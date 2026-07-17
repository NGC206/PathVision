"""Resource Manager for PathVision Runtime v2.0."""

from __future__ import annotations

import logging
import os
import sys
import time
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

# Add project root to path if needed
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import AppConfig
from perception.pathvision_trt import TRTPathVisionEngine
from perception.depth_trt import TRTDepthEngine
from reasoning.qwen_llama import QwenLlamaReasoner
from speech.kokoro import KokoroSpeaker

LOGGER = logging.getLogger(__name__)


class AsyncCamera:
    """Thread-safe non-blocking camera device wrapper with deadlock-prevention timeouts."""

    def __init__(self, camera_index: int, width: int, height: int, backend_dshow: bool = True) -> None:
        self._camera_index = camera_index
        self._width = width
        self._height = height
        self._backend = cv2.CAP_DSHOW if backend_dshow else cv2.CAP_MSMF
        
        self._lock = threading.Lock()
        self._io_lock = threading.Lock() # Lock for blocking video capture IO calls
        
        # Performance timings (in ms)
        self.time_spent_waiting_mutex_ms = 0.0
        self.time_spent_in_read_ms = 0.0
        self.time_spent_copying_frame_ms = 0.0
        self.main_thread_wait_read_latest_ms = 0.0

        LOGGER.info("Initializing VideoCapture at index %d...", camera_index)
        self._cap = cv2.VideoCapture(camera_index, self._backend)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if not self._cap.isOpened():
            LOGGER.error("Failed to open camera device index: %d", camera_index)
            raise RuntimeError(f"Could not open camera at index {camera_index}")

        self._latest_frame: np.ndarray | None = None
        self._running = True
        self._thread = threading.Thread(target=self._run, name="CameraReaderThread", daemon=True)
        self._thread.start()
        LOGGER.info("AsyncCamera thread started.")

    def _run(self) -> None:
        """Background frame grabbing loop."""
        while True:
            with self._lock:
                if not self._running:
                    break
                cap = self._cap
                if not cap.isOpened():
                    time.sleep(0.01)
                    continue

            # Read frame with io_lock acquire timeout to prevent C++ block deadlocks
            t_read_start = time.perf_counter()
            acquired = self._io_lock.acquire(timeout=1.5)
            if acquired:
                try:
                    ok, frame = cap.read()
                except Exception as e:
                    LOGGER.error("Exception in VideoCapture.read: %s", e)
                    ok, frame = False, None
                finally:
                    self._io_lock.release()
                
                self.time_spent_in_read_ms = (time.perf_counter() - t_read_start) * 1000.0
                
                if ok and frame is not None:
                    t_copy_start = time.perf_counter()
                    with self._lock:
                        self._latest_frame = frame
                    self.time_spent_copying_frame_ms = (time.perf_counter() - t_copy_start) * 1000.0
                else:
                    time.sleep(0.01)
            else:
                LOGGER.warning("CameraReaderThread timed out acquiring self._io_lock.")
                time.sleep(0.1)

    def read_latest(self) -> np.ndarray | None:
        """Thread-safe fetch of latest frame."""
        t_wait_start = time.perf_counter()
        with self._lock:
            self.main_thread_wait_read_latest_ms = (time.perf_counter() - t_wait_start) * 1000.0
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def change_camera(self, new_index: int) -> bool:
        """Safely switch camera index on the fly."""
        LOGGER.info("Attempting safely timed camera index switch to: %d", new_index)
        acquired_lock = self._lock.acquire(timeout=2.0)
        if not acquired_lock:
            LOGGER.error("ResourceManager lock timed out trying to change camera index.")
            return False

        try:
            acquired_io = self._io_lock.acquire(timeout=2.0)
            if not acquired_io:
                LOGGER.error("ResourceManager io_lock timed out trying to release VideoCapture.")
                return False

            try:
                self._cap.release()
                time.sleep(0.5) # allow hardware driver cool-down
                
                new_cap = cv2.VideoCapture(new_index, self._backend)
                new_cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
                new_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
                
                if not new_cap.isOpened():
                    LOGGER.error("Failed to open camera index %d on the fly.", new_index)
                    # Revert immediately to old index
                    self._cap = cv2.VideoCapture(self._camera_index, self._backend)
                    self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
                    self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
                    return False
                
                self._cap = new_cap
                self._camera_index = new_index
                self._latest_frame = None
                LOGGER.info("Camera index successfully switched to %d.", new_index)
                return True
            finally:
                self._io_lock.release()
        finally:
            self._lock.release()

    def close(self) -> None:
        """Release VideoCapture safely."""
        LOGGER.info("Closing AsyncCamera...")
        with self._lock:
            self._running = False
        
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
            
        acquired_lock = self._lock.acquire(timeout=2.0)
        if acquired_lock:
            try:
                acquired_io = self._io_lock.acquire(timeout=2.0)
                if acquired_io:
                    try:
                        self._cap.release()
                        LOGGER.info("AsyncCamera VideoCapture released.")
                    finally:
                        self._io_lock.release()
                else:
                    LOGGER.warning("Could not acquire io_lock during camera release.")
            finally:
                self._lock.release()
        else:
            LOGGER.warning("Could not acquire lock during camera release.")


class ResourceManager:
    """Manages the lifecycle, priority streams, and warmups of all hardware resources."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        
        # Primary Hardware resources
        self.camera: AsyncCamera | None = None
        self.cuda_stream_high: torch.cuda.Stream | None = None
        self.pathvision_engine: TRTPathVisionEngine | None = None
        self.depth_engine: TRTDepthEngine | None = None
        self.reasoner: QwenLlamaReasoner | None = None
        self.speaker: KokoroSpeaker | None = None
        
        LOGGER.info("ResourceManager initialized.")

    def load_hardware_and_runtimes(self) -> None:
        """Allocates hardware streams and instantiates perception engines."""
        # 1. Discover CUDA Device
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. CUDA GPU is required for real-time operation.")
        torch.cuda.set_device(0)
        
        # 2. Allocate Priority CUDA Streams
        LOGGER.info("Allocating high-priority CUDA Stream for Fast Runtime...")
        self.cuda_stream_high = torch.cuda.Stream(priority=-1)
        
        # 3. Load Camera
        idx = self.config.camera.index
        width = self.config.camera.width
        height = self.config.camera.height
        dshow = self.config.camera.backend_dshow
        
        LOGGER.info("Instantiating AsyncCamera...")
        self.camera = AsyncCamera(idx, width, height, dshow)
        
        # 4. Load TensorRT engines
        path_engine_path = self.config.engines.pathvision
        depth_engine_path = self.config.engines.depth_anything
        
        LOGGER.info("Loading PathVision TRT Engine from: %s", path_engine_path)
        self.pathvision_engine = TRTPathVisionEngine(str(path_engine_path))
        
        LOGGER.info("Loading Depth TRT Engine from: %s", depth_engine_path)
        self.depth_engine = TRTDepthEngine(
            engine_path=depth_engine_path,
            input_width=self.config.depth.input_width,
            input_height=self.config.depth.input_height,
            mean=self.config.depth.mean,
            std=self.config.depth.std,
        )

        # Apply high priority stream to engines if supported
        if hasattr(self.pathvision_engine, "stream"):
            self.pathvision_engine.stream = self.cuda_stream_high

    def load_reasoning_and_speech(self) -> None:
        """Loads and compiles GGUF and Kokoro models."""
        # Load Reasoner
        if self.config.reasoning.enabled:
            LOGGER.info("Loading Qwen llama.cpp Model...")
            self.reasoner = QwenLlamaReasoner(self.config)
        
        # Load Speech
        if self.config.speech.enabled:
            LOGGER.info("Loading Kokoro Speech Pipeline...")
            self.speaker = KokoroSpeaker(
                enabled=self.config.speech.enabled,
                voice=self.config.speech.voice,
                language_code=self.config.speech.language_code,
                speed=self.config.speech.speed,
                sample_rate=self.config.speech.sample_rate,
            )

    def warmup_resources(self) -> None:
        """Execute warming up runs across all models to prevent first-run latencies."""
        LOGGER.info("Executing staged resource warmup...")
        
        # 1. Warmup PathVision TRT
        if self.pathvision_engine:
            LOGGER.info("Warming up PathVision TRT Engine...")
            dummy_input = torch.zeros(self.pathvision_engine.meta.input_shape, dtype=self.pathvision_engine.meta.input_dtype)
            with torch.cuda.stream(self.cuda_stream_high):
                for _ in range(3):
                    self.pathvision_engine.infer(dummy_input)
            LOGGER.info("PathVision TRT Engine warmed up.")
            
        # 2. Warmup Depth Anything TRT
        if self.depth_engine:
            LOGGER.info("Warming up Depth Anything TRT Engine...")
            dummy_depth_input = np.zeros((self.config.depth.input_height, self.config.depth.input_width, 3), dtype=np.uint8)
            for _ in range(3):
                self.depth_engine.infer(dummy_depth_input)
            LOGGER.info("Depth Anything TRT Engine warmed up.")
            
        # 3. Warmup llama.cpp
        if self.reasoner and self.reasoner.model:
            LOGGER.info("Warming up llama.cpp Model...")
            dummy_payload = {
                "timestamp": "2026-07-16T15:00:00Z",
                "path_visible": 1,
                "path_center_x": 160,
                "path_width_px": 240,
                "path_width_ratio": 0.75,
                "safe_area_ratio": 0.65,
                "nearest_obstacle_distance": 2.50,
                "scene_confidence": 0.85,
                "danger_state": "safe",
                "navigation_recommendation": "FORWARD",
                "navigation_confidence": 0.90,
            }
            dummy_memories = ["System initialized."]
            # Execute first warmup (compiles CUDA contexts)
            self.reasoner.generate(dummy_payload, dummy_memories, "guidance")
            LOGGER.info("llama.cpp warmed up.")

        # 4. Warmup Kokoro
        if self.speaker:
            LOGGER.info("Warming up Kokoro Speech synthesizer...")
            self.speaker.warmup() # Loads KPipeline and triggers sounddevice streams
            LOGGER.info("Kokoro Speech synthesizer warmed up.")

    def release_all(self) -> None:
        """Shutdown and release all hardware/model bindings safely."""
        LOGGER.info("Initiating resource manager cleanup...")
        
        # 1. Release camera
        if self.camera:
            try:
                self.camera.close()
            except Exception as e:
                LOGGER.error("Error closing AsyncCamera: %s", e)
                
        # 2. Release llama.cpp
        if self.reasoner:
            try:
                self.reasoner.close()
            except Exception as e:
                LOGGER.error("Error closing Reasoner: %s", e)

        # 3. Release Speech
        if self.speaker:
            try:
                self.speaker.close()
            except Exception as e:
                LOGGER.error("Error closing Speaker: %s", e)

        LOGGER.info("All resources safely released.")
