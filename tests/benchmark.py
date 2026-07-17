"""Production pipeline benchmark script for PathVision Final.

Measures startup time, model loading, individual processing stage latencies,
and CUDA memory allocations under simulated runtime conditions.
"""

from __future__ import annotations

import logging
import time
import sys
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Mock camera and speech classes before importing modules
import runtime.resource_manager

class BenchmarkCamera:
    """Mock camera that returns consistent test frames without camera I/O overhead."""
    def __init__(self, camera_index: int, width: int, height: int, backend_dshow: bool = True) -> None:
        self._camera_index = camera_index
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.frame[320:, 160:480] = [128, 128, 128]
        
    def read_latest(self) -> np.ndarray:
        return self.frame.copy()
        
    def close(self) -> None:
        pass

class BenchmarkSpeaker:
    def __init__(self, enabled: bool, voice: str, language_code: str, speed: float, sample_rate: int) -> None:
        pass
    def warmup(self) -> None:
        pass
    def speak(self, text: str, blocking: bool = False, priority: int = 2) -> None:
        pass
    def close(self) -> None:
        pass

runtime.resource_manager.AsyncCamera = BenchmarkCamera
runtime.resource_manager.KokoroSpeaker = BenchmarkSpeaker

from config import load_config
from runtime.runtime_manager import RuntimeManager

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("benchmark")


def run_benchmark(num_frames: int = 100) -> None:
    """Run end-to-end benchmark on the actual production pipeline."""
    LOGGER.info("==================================================")
    LOGGER.info("Starting PathVision Final Production Benchmark...")
    LOGGER.info("==================================================")

    # 1. Measure Startup Time
    t_start = time.perf_counter()
    config = load_config()
    
    # Disable preview to run headlessly
    object.__setattr__(config.runtime, "show_preview", False)
    
    manager = RuntimeManager(config)
    
    # Start manager but mock camera and speaker to run synchronously
    manager.start()
    
    # Stop the background scheduler thread to avoid concurrent TRT execution collisions
    manager.scheduler.stop()
    
    t_startup = time.perf_counter() - t_start
    LOGGER.info("Startup / Model Loading completed in: %.3f seconds", t_startup)

    # Pre-allocate lists for latency logs
    cam_latencies = []
    preproc_latencies = []
    pv_infer_latencies = []
    decoder_latencies = []
    postproc_latencies = []
    depth_latencies = []
    fusion_latencies = []
    e2e_latencies = []

    # Warmup loop to clear CUDA graph caches
    LOGGER.info("Warming up CUDA contexts...")
    for _ in range(10):
        frame = manager.resource_manager.camera.read_latest()
        frame_small, input_cpu = manager.scheduler.preprocessor.run(frame)
        logits_gpu = manager.resource_manager.pathvision_engine.infer(input_cpu)
        class_map, safe_prob = manager.scheduler.decoder.run(logits_gpu)
        safe_mask = manager.scheduler.postproc.run(class_map, safe_prob)
        depth_map = manager.resource_manager.depth_engine.infer(frame)
        _ = manager.scene_fusion.build(safe_mask_u8=safe_mask, depth_map=depth_map)

    LOGGER.info("Running %d benchmark iterations...", num_frames)
    
    # Enable PyTorch CUDA profiling
    torch.cuda.reset_peak_memory_stats()
    
    for i in range(num_frames):
        t_frame_start = time.perf_counter()
        
        # Stage 1: Camera Grab
        t0 = time.perf_counter()
        frame = manager.resource_manager.camera.read_latest()
        cam_latencies.append((time.perf_counter() - t0) * 1000.0)
        
        # Stage 2: Preprocessor
        t0 = time.perf_counter()
        frame_small, input_cpu = manager.scheduler.preprocessor.run(frame)
        preproc_latencies.append((time.perf_counter() - t0) * 1000.0)
        
        # Stage 3: PathVision Segmentation Inference (TRT)
        t0 = time.perf_counter()
        logits_gpu = manager.resource_manager.pathvision_engine.infer(input_cpu)
        pv_infer_latencies.append((time.perf_counter() - t0) * 1000.0)
        
        # Stage 4: Segmentation Decoding
        t0 = time.perf_counter()
        class_map, safe_prob = manager.scheduler.decoder.run(logits_gpu)
        decoder_latencies.append((time.perf_counter() - t0) * 1000.0)
        
        # Stage 5: Safe Mask Post-processing
        t0 = time.perf_counter()
        safe_mask = manager.scheduler.postproc.run(class_map, safe_prob)
        postproc_latencies.append((time.perf_counter() - t0) * 1000.0)
        
        # Stage 6: Depth Anything Inference (TRT)
        t0 = time.perf_counter()
        depth_map = manager.resource_manager.depth_engine.infer(frame)
        depth_latencies.append((time.perf_counter() - t0) * 1000.0)
        
        # Stage 7: Scene Fusion & Navigation logic
        t0 = time.perf_counter()
        scene = manager.scene_fusion.build(safe_mask_u8=safe_mask, depth_map=depth_map)
        fusion_latencies.append((time.perf_counter() - t0) * 1000.0)
        
        # Telemetry & memory updates
        manager.telemetry_queue.put((0, scene))
        manager.memory.add(scene)
        
        e2e_latencies.append((time.perf_counter() - t_frame_start) * 1000.0)

    # 2. Get Memory & GPU Stats
    ram_usage_mb = 0.0
    try:
        import psutil
        process = psutil.Process(os.getpid())
        ram_usage_mb = process.memory_info().rss / (1024.0 * 1024.0)
    except ImportError:
        pass

    peak_vram_bytes = torch.cuda.max_memory_allocated()
    peak_vram_mb = peak_vram_bytes / (1024.0 * 1024.0)

    # 3. Measure Shutdown Time
    t_shutdown_start = time.perf_counter()
    manager.stop()
    t_shutdown = time.perf_counter() - t_shutdown_start

    # Output detailed metrics report
    print("\n" + "=" * 50)
    print("PATHVISION FINAL BENCHMARK METRICS REPORT")
    print("=" * 50)
    print(f"Startup Time        : {t_startup:.3f} seconds")
    print(f"Shutdown Time       : {t_shutdown:.3f} seconds")
    print(f"Peak VRAM Usage     : {peak_vram_mb:.2f} MB")
    if ram_usage_mb > 0.0:
        print(f"RAM Usage           : {ram_usage_mb:.2f} MB")
    
    print("-" * 50)
    print("STAGE LATENCIES (ms)")
    print("-" * 50)
    print(f"Camera Grab         - Mean: {np.mean(cam_latencies):.2f} ms | Std: {np.std(cam_latencies):.2f} ms")
    print(f"Frame Preprocessing - Mean: {np.mean(preproc_latencies):.2f} ms | Std: {np.std(preproc_latencies):.2f} ms")
    print(f"PathVision TRT      - Mean: {np.mean(pv_infer_latencies):.2f} ms | Std: {np.std(pv_infer_latencies):.2f} ms")
    print(f"Logits Decoding     - Mean: {np.mean(decoder_latencies):.2f} ms | Std: {np.std(decoder_latencies):.2f} ms")
    print(f"Mask Post-processing- Mean: {np.mean(postproc_latencies):.2f} ms | Std: {np.std(postproc_latencies):.2f} ms")
    print(f"Depth Anything TRT  - Mean: {np.mean(depth_latencies):.2f} ms | Std: {np.std(depth_latencies):.2f} ms")
    print(f"Scene Fusion & Nav  - Mean: {np.mean(fusion_latencies):.2f} ms | Std: {np.std(fusion_latencies):.2f} ms")
    print(f"End-to-End Latency  - Mean: {np.mean(e2e_latencies):.2f} ms | Std: {np.std(e2e_latencies):.2f} ms")
    print(f"Calculated Loop FPS : {1000.0 / np.mean(e2e_latencies):.1f} FPS")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    run_benchmark(100)
