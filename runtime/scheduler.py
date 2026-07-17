"""Scheduler for executing the 30 FPS Fast Vision Runtime loop in PathVision v2.0."""

from __future__ import annotations

import logging
import time
import threading
from datetime import datetime, timezone
import numpy as np
import torch

from runtime.world_model import WorldModel, ScenePacket, NavMeshRepresentation
from runtime.event_bus import EventBus, Event, EventPriority, EventType
from runtime.resource_manager import ResourceManager
from config import AppConfig

# Import fusion models
from perception.pathvision_trt import FramePreprocessor, SegmentationDecoder, SafeMaskPostProcessor

LOGGER = logging.getLogger(__name__)


class Scheduler:
    """Manages the fast visual/inference loop at a deterministic 30 FPS."""

    def __init__(
        self,
        resource_manager: ResourceManager,
        world_model: WorldModel,
        event_bus: EventBus,
        config: AppConfig,
        scene_fusion: Any, # scene fusion model passed in from runtime manager
        health_monitor: Any = None, # optional health monitor for heartbeats
    ) -> None:
        self.rm = resource_manager
        self.wm = world_model
        self.eb = event_bus
        self.config = config
        self.scene_fusion = scene_fusion
        self._health_monitor = health_monitor

        self._running = False
        self._thread: threading.Thread | None = None
        self._latest_timings: dict[str, float] = {}
        self.frame_id = 0

        # Preprocessors and Decoders
        self.preprocessor = FramePreprocessor(
            model_w=config.pathvision.model_width,
            model_h=config.pathvision.model_height,
            input_dtype=self.rm.pathvision_engine.meta.input_dtype,
        )
        self.decoder = SegmentationDecoder(
            model_h=config.pathvision.model_height,
            model_w=config.pathvision.model_width,
            logits_dtype=self.rm.pathvision_engine.meta.output_dtype,
        )
        self.postproc = SafeMaskPostProcessor(
            model_h=config.pathvision.model_height,
            model_w=config.pathvision.model_width,
            safe_class_id=config.pathvision.safe_class_id,
            prob_threshold=config.pathvision.safe_probability_threshold,
        )

        # Track state transitions for event trigger checks
        self._last_command = ""
        self._last_safety = ""

    def start(self) -> None:
        """Start the fast runtime thread."""
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="FastVisionPipeline", daemon=True)
        self._thread.start()
        LOGGER.info("Scheduler started background vision pipeline loop.")

    def stop(self) -> None:
        """Gracefully stop scheduler loop."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        LOGGER.info("Scheduler stopped.")

    def get_latest_timings(self) -> dict[str, float]:
        """Fetch the latencies of the last execution loop."""
        return dict(self._latest_timings)

    def _loop(self) -> None:
        """Continuous execution loop targeting 30 FPS (33ms loop intervals)."""
        target_frame_time = 1.0 / 30.0 # ~33.3ms
        
        while self._running:
            t_loop_start = time.perf_counter()

            # Heartbeat to HealthMonitor
            if self._health_monitor:
                self._health_monitor.heartbeat("FastVisionPipeline")

            # 1. Grab raw webcam frame
            t0 = time.perf_counter()
            frame = self.rm.camera.read_latest()
            dt_cam = (time.perf_counter() - t0) * 1000.0
            
            if frame is None:
                # Sleep briefly and try again
                time.sleep(0.005)
                continue

            self.frame_id += 1

            # 2. Resize & Preprocessing
            t0 = time.perf_counter()
            frame_small, input_cpu = self.preprocessor.run(frame)
            dt_preproc = (time.perf_counter() - t0) * 1000.0

            # 3. Async inference triggers inside the high-priority CUDA stream
            t0 = time.perf_counter()
            with torch.cuda.stream(self.rm.cuda_stream_high):
                self.rm.pathvision_engine.infer_async(input_cpu)
                self.rm.depth_engine.infer_async(frame)
            dt_infer = (time.perf_counter() - t0) * 1000.0

            # 4. Synchronize PathVision and run decoder
            t0 = time.perf_counter()
            logits_gpu = self.rm.pathvision_engine.synchronize()
            class_map, safe_prob = self.decoder.run(logits_gpu)
            dt_decoder = (time.perf_counter() - t0) * 1000.0

            # 5. Mask Post-processing
            t0 = time.perf_counter()
            safe_mask = self.postproc.run(class_map, safe_prob)
            dt_postproc = (time.perf_counter() - t0) * 1000.0

            # 6. Synchronize Depth TRT
            t0 = time.perf_counter()
            depth_map = self.rm.depth_engine.synchronize()
            dt_depth = (time.perf_counter() - t0) * 1000.0

            # 7. Scene Fusion (Fuses path mask, depth map, builds NavigationMesh)
            t0 = time.perf_counter()
            scene = self.scene_fusion.build(safe_mask_u8=safe_mask, depth_map=depth_map)
            dt_fusion = (time.perf_counter() - t0) * 1000.0

            # 8. Create Navigation Mesh Representation
            t0 = time.perf_counter()
            # Extract centerline, boundaries, curvature properties from built mesh
            center_line = list(scene.nav_mesh.centerline)
            left_boundary = []
            right_boundary = []
            for node in scene.nav_mesh.nodes.values():
                if node.is_left:
                    left_boundary.append((node.x, node.y))
                else:
                    right_boundary.append((node.x, node.y))
                    
            nav_mesh_rep = NavMeshRepresentation(
                center_line=center_line,
                left_boundary=left_boundary,
                right_boundary=right_boundary,
                walkable_corridor=safe_mask,
                curvature=scene.nav_mesh.curvature_index,
                forward_distance=float(len(center_line) * 0.1), # approximation
                clearance=scene.nav_mesh.nearest_obstacle_dist,
                confidence=scene.scene_confidence,
            )
            dt_mesh = (time.perf_counter() - t0) * 1000.0

            # 9. Write ScenePacket to WorldModel atomically
            t0 = time.perf_counter()
            packet = ScenePacket(
                frame_id=self.frame_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                frame=frame_small.copy(),
                depth_map=depth_map.copy(),
                safe_mask=safe_mask.copy(),
                nav_mesh=scene.nav_mesh,
                nav_mesh_rep=nav_mesh_rep,
                path_geometry=scene.path_geometry,
                safety=scene.safety,
                navigation=scene.navigation,
                situation=scene.situation if hasattr(scene, "situation") else None,
                scene_confidence=scene.scene_confidence,
                guidance="",
            )
            self.wm.update(packet)
            dt_wm = (time.perf_counter() - t0) * 1000.0

            # 10. Check Safety/Command events and publish to EventBus
            curr_command = scene.navigation.command.value
            curr_safety = scene.safety.state.value

            if curr_command != self._last_command:
                # Trigger Priority 2 Navigation event
                self.eb.publish(Event(
                    priority=EventPriority.NAVIGATION,
                    event_type=EventType.NAV_COMMAND,
                    payload={"command": curr_command, "previous": self._last_command},
                    timestamp=t_loop_start
                ))
                self._last_command = curr_command

            if curr_safety != self._last_safety:
                # Map safety state transitions to high-priority events
                prio = EventPriority.HAZARD
                evt_type = EventType.OBSTACLE_DETECTED
                if curr_safety == "danger":
                    prio = EventPriority.EMERGENCY
                    evt_type = EventType.EMERGENCY_STOP
                
                self.eb.publish(Event(
                    priority=prio,
                    event_type=evt_type,
                    payload={"safety": curr_safety, "previous": self._last_safety},
                    timestamp=t_loop_start
                ))
                self._last_safety = curr_safety

            # Record latencies
            loop_duration = time.perf_counter() - t_loop_start
            self._latest_timings = {
                "camera_capture": dt_cam,
                "preprocessing": dt_preproc,
                "pathvision_trt": dt_infer,
                "logits_decoding": dt_decoder,
                "mask_postproc": dt_postproc,
                "depth_trt": dt_depth,
                "scene_fusion": dt_fusion,
                "nav_mesh": dt_mesh,
                "world_model": dt_wm,
                "total_loop": loop_duration * 1000.0,
            }

            # 11. Enforce 30 FPS loop rate
            sleep_time = target_frame_time - loop_duration
            if sleep_time > 0:
                time.sleep(sleep_time)
