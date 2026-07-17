"""PathVision Final v2.0 — Entry Point and Visual Runtime Loop Manager."""

from __future__ import annotations

import logging
import queue
import time
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import psutil

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))

from config import load_config, AppConfig
from runtime.runtime_manager import RuntimeManager
from runtime.runtime_state import RuntimeState
from runtime.event_bus import Event, EventPriority, EventType
from reasoning.situation_manager import InteractionMode
from learning.feedback import FeedbackLabel
from perception.pathvision_trt import Visualizer

LOGGER = logging.getLogger("pathvision")


class KeyboardListener:
    """Hooks global keyboard events using Windows User32 DLL shortcuts."""

    def __init__(self) -> None:
        self.has_user32 = False
        if sys.platform == "win32":
            try:
                import ctypes
                self.user32 = ctypes.windll.user32
                self.has_user32 = True
                LOGGER.info("Windows KeyboardListener loaded successfully.")
            except Exception as e:
                LOGGER.warning("Could not initialize Windows User32 Keyboard hook: %s", e)

    def is_key_just_pressed(self, vk_code: int) -> bool:
        """Check if a virtual key code is pressed (checks high-order bit)."""
        if not self.has_user32:
            return False
        # GetAsyncKeyState returns 16-bit integer. High-order bit indicates key state.
        return bool(self.user32.GetAsyncKeyState(vk_code) & 0x8000)


# Virtual Key Codes
VK_ESCAPE = 0x1B
VK_Q = 0x51
VK_S = 0x53
VK_E = 0x45
VK_D = 0x44
VK_K = 0x4B
VK_C = 0x43
VK_W = 0x57
VK_R = 0x52
VK_V = 0x56
VK_A = 0x41


class PathVisionApplication:
    """Manages the visual preview window and keyboard shortcut hooks on the main thread."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.manager = RuntimeManager(config)
        self.keyboard = KeyboardListener()
        self._last_key_action_ts: dict[str, float] = {}
        self._key_cooldown_s = 0.35
        self.visualizer = Visualizer(
            model_h=config.pathvision.model_height,
            model_w=config.pathvision.model_width,
            display_scale=config.pathvision.display_scale,
        )

    def start(self) -> None:
        """Start the runtime manager and run the GUI preview loop."""
        self.manager.start()
        
        # Staged visual window initialization
        if self.config.runtime.show_preview:
            cv2.namedWindow("PathVision Final", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("PathVision Final", 640, 600) # Include depth bar height
            
        LOGGER.info("Starting GUI preview and keyboard polling loop on main thread...")
        
        last_difficult_capture_ts = 0.0
        last_telemetry_ts = 0.0
        
        try:
            while self.manager._running:
                t_loop_start = time.perf_counter()
                
                # Check Global Keyboard Shortcuts
                requested_exit = False
                requested_scan_orientation = False
                requested_scan_description = False
                requested_camera_change = False
                
                if self.keyboard.has_user32:
                    if self.keyboard.is_key_just_pressed(VK_Q) or self.keyboard.is_key_just_pressed(VK_ESCAPE):
                        requested_exit = True
                    elif self.keyboard.is_key_just_pressed(VK_S):
                        requested_scan_orientation = True
                    elif self.keyboard.is_key_just_pressed(VK_D):
                        requested_scan_description = True
                    elif self.keyboard.is_key_just_pressed(VK_K):
                        requested_camera_change = True
                    elif self.keyboard.is_key_just_pressed(VK_C):
                        self._log_feedback("C", "Operator Feedback: Correct Path")
                        self.manager.telemetry_queue.put((2, (time.time(), FeedbackLabel.CORRECT, "Operator verified path")))
                    elif self.keyboard.is_key_just_pressed(VK_W):
                        self._log_feedback("W", "Operator Feedback: Wrong Direction")
                        self.manager.telemetry_queue.put((2, (time.time(), FeedbackLabel.WRONG_DIRECTION, "Operator reported wrong direction")))
                    elif self.keyboard.is_key_just_pressed(VK_R):
                        self._log_feedback("R", "Operator Feedback: Too Risky")
                        self.manager.telemetry_queue.put((2, (time.time(), FeedbackLabel.TOO_RISKY, "Operator reported too risky")))
                    elif self.keyboard.is_key_just_pressed(VK_V):
                        self._log_feedback("V", "Operator Feedback: Too Conservative")
                        self.manager.telemetry_queue.put((2, (time.time(), FeedbackLabel.TOO_CONSERVATIVE, "Operator reported too conservative")))

                # Poll OpenCV Window Key Events (as fallback / non-Windows shortcut handler)
                key = -1
                if self.config.runtime.show_preview:
                    key = cv2.waitKey(1) & 0xFF
                    if key != 255 and key != -1:
                        if not self.keyboard.has_user32:
                            if key in (ord("q"), 27): # q or ESC
                                requested_exit = True
                            elif key in (ord("s"), ord("S")):
                                requested_scan_orientation = True
                            elif key in (ord("d"), ord("D")):
                                requested_scan_description = True
                            elif key in (ord("k"), ord("K")):
                                requested_camera_change = True
                            elif key == ord("c") or key == ord("C"):
                                self._log_feedback("C", "Operator Feedback: Correct Path")
                                self.manager.telemetry_queue.put((2, (time.time(), FeedbackLabel.CORRECT, "Operator verified path")))
                            elif key == ord("w") or key == ord("W"):
                                self._log_feedback("W", "Operator Feedback: Wrong Direction")
                                self.manager.telemetry_queue.put((2, (time.time(), FeedbackLabel.WRONG_DIRECTION, "Operator reported wrong direction")))
                            elif key == ord("r") or key == ord("R"):
                                self._log_feedback("R", "Operator Feedback: Too Risky")
                                self.manager.telemetry_queue.put((2, (time.time(), FeedbackLabel.TOO_RISKY, "Operator reported too risky")))
                            elif key == ord("v") or key == ord("V"):
                                self._log_feedback("V", "Operator Feedback: Too Conservative")
                                self.manager.telemetry_queue.put((2, (time.time(), FeedbackLabel.TOO_CONSERVATIVE, "Operator reported too conservative")))

                # Execute requested actions via event bus publications
                if requested_exit:
                    LOGGER.info("Shutdown requested via keyboard shortcut.")
                    break
                    
                if requested_scan_orientation:
                    if self._allow_key_action("scan_orientation"):
                        self.manager.event_bus.publish(Event(
                            priority=EventPriority.COGNITIVE,
                            event_type=EventType.SCAN_REQUEST,
                            payload={"mode": InteractionMode.ORIENTATION}
                        ))
                    
                if requested_scan_description:
                    if self._allow_key_action("scan_description"):
                        self.manager.event_bus.publish(Event(
                            priority=EventPriority.COGNITIVE,
                            event_type=EventType.SCAN_REQUEST,
                            payload={"mode": InteractionMode.DESCRIPTION}
                        ))
                    
                if requested_camera_change:
                    if self._allow_key_action("camera_cycle"):
                        self._cycle_camera()

                # Get latest state packet from WorldModel
                packet = self.manager.world_model.get_latest()
                if packet is not None:
                    now = time.perf_counter()
                    
                    # Throttled telemetry logging — once per second max to avoid queue flooding
                    if now - last_telemetry_ts >= 1.0:
                        last_telemetry_ts = now
                        # Only log if the queue is not already backed up
                        if self.manager.telemetry_queue.qsize() < 10:
                            try:
                                self.manager.telemetry_queue.put_nowait((0, packet))
                            except queue.Full:
                                pass  # Drop telemetry sample if queue is full
                    
                    # Dataset Auto-Labeler evaluations — throttled at 5 second intervals
                    if self.config.learning.enabled and (now - last_difficult_capture_ts >= 5.0):
                        try:
                            capture = self.manager.auto_labeler.evaluate(packet)
                            if capture.should_capture:
                                last_difficult_capture_ts = now
                                try:
                                    self.manager.telemetry_queue.put_nowait((1, (packet.frame.copy(), packet, capture.reason)))
                                except queue.Full:
                                    pass
                        except Exception as e:
                            LOGGER.debug("Auto-labeler evaluation error: %s", e)

                    # Render preview overlays
                    if self.config.runtime.show_preview:
                        try:
                            # Draw safety mask overlays & navigation steer vectors
                            preview = self.visualizer.draw(
                                frame_320=packet.frame,
                                class_map=(packet.safe_mask > 0).astype(np.uint8),
                                filtered_safe_mask=packet.safe_mask,
                                command=packet.navigation.command.value,
                                center_x=packet.path_geometry.center_x,
                                fps=30.0,
                                depth_map=packet.depth_map,
                                safe_mask_full=packet.safe_mask,
                            )
                            
                            # Draw Left boundary, right boundary, centerline overlay
                            self._draw_mesh_overlay(preview, packet.nav_mesh_rep)
                            
                            # Draw HUD sidebar debug info
                            self._draw_hud(preview, packet)
                            
                            cv2.imshow("PathVision Final", preview)
                        except Exception as e:
                            LOGGER.debug("Preview rendering error: %s", e)
                
                # Yield CPU — target ~30 FPS for GUI loop
                elapsed = time.perf_counter() - t_loop_start
                sleep_ms = max(0.001, 0.033 - elapsed)
                time.sleep(sleep_ms)
        except KeyboardInterrupt:
            LOGGER.info("Application interrupted.")
        finally:
            self.manager.stop()

    def _log_feedback(self, key: str, label: str) -> None:
        print("\n------------------------------------------------")
        print(f"Key : {key}")
        print(f"Action : {label}")
        print("Status : Logged")
        print("------------------------------------------------\n")
        LOGGER.info("Feedback registered: %s (Key: %s)", label, key)

    def _allow_key_action(self, action: str) -> bool:
        """Debounce repeated key actions to avoid event flooding."""
        now = time.perf_counter()
        prev = self._last_key_action_ts.get(action, 0.0)
        if now - prev < self._key_cooldown_s:
            return False
        self._last_key_action_ts[action] = now
        return True

    def _cycle_camera(self) -> None:
        """Cycle camera indices based on dynamic available device scans."""
        LOGGER.info("Dynamic Camera Cycle Triggered.")
        available = []
        scan_backend = cv2.CAP_DSHOW if self.config.camera.backend_dshow else cv2.CAP_ANY
        for i in range(4):
            try:
                cap = cv2.VideoCapture(i, scan_backend)
                if cap.isOpened():
                    available.append(i)
                    cap.release()
            except Exception:
                pass
                
        if not available:
            return
            
        try:
            curr_pos = available.index(self.manager.resource_manager.camera._camera_index)
            next_pos = (curr_pos + 1) % len(available)
            new_idx = available[next_pos]
        except ValueError:
            new_idx = available[0]
            
        success = self.manager.resource_manager.camera.change_camera(new_idx)
        if success:
            LOGGER.info("Switched camera device to index: %d", new_idx)
        else:
            LOGGER.error("Failed to cycle camera index to: %d", new_idx)

    def _draw_mesh_overlay(self, preview: np.ndarray, rep: NavMeshRepresentation) -> None:
        """Draw centerline tracks, boundary graphs, and look-ahead averages on preview canvas."""
        h, w = preview.shape[:2]
        
        # Draw centerline track (Yellow centerline track)
        for pt in rep.center_line:
            # Map coordinates
            px = int(pt[0] * (w / self.config.pathvision.model_width))
            py = int(pt[1] * (h / self.config.pathvision.model_height))
            cv2.circle(preview, (px, py), 2, (0, 255, 255), -1)

        # Draw boundaries (Left is Red, Right is Blue)
        for pt in rep.left_boundary:
            px = int(pt[0] * (w / self.config.pathvision.model_width))
            py = int(pt[1] * (h / self.config.pathvision.model_height))
            cv2.circle(preview, (px, py), 2, (0, 0, 255), -1)
            
        for pt in rep.right_boundary:
            px = int(pt[0] * (w / self.config.pathvision.model_width))
            py = int(pt[1] * (h / self.config.pathvision.model_height))
            cv2.circle(preview, (px, py), 2, (255, 0, 0), -1)

    def _draw_hud(self, preview: np.ndarray, packet: ScenePacket) -> None:
        """Draw sidebar HUD overlays on preview canvas."""
        h, w = preview.shape[:2]
        sidebar_w = 270
        sidebar_mask = preview[:, :sidebar_w].copy()
        cv2.rectangle(sidebar_mask, (0, 0), (sidebar_w, h), (15, 15, 15), -1)
        cv2.addWeighted(preview[:, :sidebar_w], 0.3, sidebar_mask, 0.7, 0, preview[:, :sidebar_w])
        
        t = self.manager.scheduler.get_latest_timings()
        stats = self.manager.health_monitor.get_stats()
        
        lines = [
            "--- PATHVISION RUNTIME v2.0 ---",
            f"State        : {self.manager.state_machine.current_state.value}",
            f"Camera Index : {self.manager.resource_manager.camera._camera_index}",
            f"Steer Command: {packet.navigation.command.value}",
            f"Safety State : {packet.safety.state.value.upper()}",
            f"Clearance    : {packet.nav_mesh_rep.clearance:.2f} m",
            f"Curvature    : {packet.nav_mesh_rep.curvature:.2f}",
            "--- RESOURCE STATS ---",
            f"CPU Usage    : {stats.get('cpu_percent', 0.0):.1f}%",
            f"System RAM   : {stats.get('ram_used_mb', 0.0):.1f} MB",
            f"GPU VRAM     : {stats.get('gpu_mem_used_mb', 0.0):.1f} MB",
            "--- STAGE LATENCY (ms) ---",
            f"Camera Grab  : {t.get('camera_capture', 0.0):.2f} ms",
            f"Preprocess   : {t.get('preprocessing', 0.0):.2f} ms",
            f"Path TRT     : {t.get('pathvision_trt', 0.0):.2f} ms",
            f"Decoder      : {t.get('logits_decoding', 0.0):.2f} ms",
            f"Depth TRT    : {t.get('depth_trt', 0.0):.2f} ms",
            f"Scene Fusion : {t.get('scene_fusion', 0.0):.2f} ms",
            f"Mesh Draw    : {t.get('nav_mesh', 0.0):.2f} ms",
            f"Total Loop   : {t.get('total_loop', 0.0):.2f} ms",
        ]
        
        y = 22
        for line in lines:
            cv2.putText(preview, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(preview, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (240, 240, 240), 1, cv2.LINE_AA)
            y += 18


def _setup_logging(config: AppConfig) -> None:
    logging.basicConfig(
        level=getattr(logging, config.runtime.log_level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> None:
    # Restrict PyTorch CPU threads to prevent core saturation during Kokoro speech synthesis
    torch.set_num_threads(2)
    config = load_config()
    _setup_logging(config)
    LOGGER.info("Starting PathVision Final Application...")
    app = PathVisionApplication(config)
    app.start()


if __name__ == "__main__":
    main()