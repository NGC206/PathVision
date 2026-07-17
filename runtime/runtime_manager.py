"""Runtime Manager for PathVision Runtime v2.0."""

from __future__ import annotations

import logging
import time
import queue
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from config import AppConfig
from runtime.runtime_state import RuntimeState, RuntimeStateMachine
from runtime.event_bus import EventBus, Event, EventPriority, EventType
from runtime.world_model import WorldModel, ScenePacket, NavMeshRepresentation
from runtime.resource_manager import ResourceManager
from runtime.scheduler import Scheduler
from runtime.health_monitor import HealthMonitor

# Perception / Navigation dependencies
from perception.scene_fusion import SceneFusion
from navigation.path_geometry import PathGeometryAnalyzer
from navigation.safety import SafetyEvaluator
from navigation.decision import NavigationDecisionEngine
from reasoning.situation_manager import SituationManager, InteractionMode, SituationType
from reasoning.scene_memory import SceneMemory
from reasoning.conversation_memory import ConversationMemory
from reasoning.prompts import fallback_instruction
from learning.scene_logger import SceneLogger
from learning.dataset_builder import DatasetBuilder
from learning.feedback import FeedbackStore, FeedbackLabel
from learning.auto_label import AutoLabeler

LOGGER = logging.getLogger(__name__)


class RuntimeManager:
    """Central director governing resource allocations, task schedules, and boot states."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        
        # State Machine
        self.state_machine = RuntimeStateMachine(RuntimeState.BOOTING)
        
        # Core Infrastructure
        self.event_bus = EventBus()
        self.world_model = WorldModel()
        self.resource_manager = ResourceManager(config)
        
        # Pipelines & Processors (will be constructed during load stages)
        self.scene_fusion: SceneFusion | None = None
        self.situation_manager: SituationManager | None = None
        self.conversation_memory: ConversationMemory | None = None
        self.memory: SceneMemory | None = None
        self.scheduler: Scheduler | None = None
        self.health_monitor: HealthMonitor | None = None
        
        # Logging & Telemetry
        self.scene_logger = SceneLogger(config.learning.scene_log_path)
        self.dataset_builder = DatasetBuilder(config.learning.dataset_output_dir)
        self.feedback_store = FeedbackStore(config.learning.feedback_log_path)
        self.auto_labeler = AutoLabeler(config.learning.capture_confidence_threshold)
        
        self.telemetry_queue: queue.Queue[tuple[int, Any] | None] = queue.Queue(maxsize=20)
        self.telemetry_tasks_created = 0
        self.telemetry_tasks_consumed = 0
        self.telemetry_write_latencies: list[float] = []
        self.telemetry_max_write_latency_ms = 0.0
        self.telemetry_avg_write_latency_ms = 0.0
        self._last_difficult_capture_ts = 0.0
        
        self._running = False
        self._telemetry_thread: threading.Thread | None = None
        self._reasoning_thread: threading.Thread | None = None
        self._last_emergency_ts = 0.0
        self._last_hazard_ts = 0.0
        self._last_recovery_ts = 0.0
        self._last_manual_scan_ts = 0.0
        self._scan_lock = threading.Lock()
        self._manual_scan_active = threading.Event()
        self._reasoning_singleflight = threading.Lock()
        self._reasoning_cooldown_until = 0.0
        self._reasoner_thread_active = False
        
        # Setup EventBus subscribers
        self.event_bus.subscribe(EventType.SCAN_REQUEST, self._handle_scan_request)
        self.event_bus.subscribe(EventType.SPEECH_REQUEST, self._handle_speech_request)
        self.event_bus.subscribe(EventType.NAV_COMMAND, self._handle_nav_command)
        self.event_bus.subscribe(EventType.EMERGENCY_STOP, self._handle_emergency)
        self.event_bus.subscribe(EventType.OBSTACLE_DETECTED, self._handle_hazard)
        
        LOGGER.info("RuntimeManager initialized.")

    def start(self) -> None:
        """Execute the staged boot sequence."""
        self._running = True
        
        # Start EventBus
        self.event_bus.start()

        # Step 1: Discover Hardware
        self.state_machine.transition_to(RuntimeState.DISCOVERING_HARDWARE)
        self._discover_hardware()

        # Step 2: Load Models & Contexts
        self.state_machine.transition_to(RuntimeState.LOADING_MODELS)
        self._load_models()

        # Step 3: Warmup Runtimes
        self.state_machine.transition_to(RuntimeState.WARMING_UP)
        self.resource_manager.warmup_resources()

        # Step 4: System Ready
        self.state_machine.transition_to(RuntimeState.READY)
        self._announce_system_online()

        # Step 5: Start Navigation & Workers
        self.state_machine.transition_to(RuntimeState.NAVIGATING)
        self._start_worker_threads()

    def stop(self) -> None:
        """Shutdown all runtimes and release resources gracefully."""
        self.state_machine.transition_to(RuntimeState.SHUTTING_DOWN)
        LOGGER.info("Shutting down RuntimeManager...")
        
        self._running = False
        
        # Stop Scheduler
        if self.scheduler:
            self.scheduler.stop()
            
        # Stop Health Monitor
        if self.health_monitor:
            self.health_monitor.stop()
            
        # Stop Telemetry Worker
        if self._telemetry_thread and self._telemetry_thread.is_alive():
            self.telemetry_queue.put(None)
            self._telemetry_thread.join(timeout=2.0)
            
        # Stop Event Bus
        self.event_bus.stop()
        
        # Release GPU Engines & Camera via Resource Manager
        self.resource_manager.release_all()
        
        cv2.destroyAllWindows()
        LOGGER.info("RuntimeManager stopped successfully.")

    def _discover_hardware(self) -> None:
        """Detect CUDA, RTX device, and scan camera indices."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not supported on this hardware.")
            
        # Scan indices 0..3
        available_cameras = []
        scan_backend = cv2.CAP_DSHOW if self.config.camera.backend_dshow else cv2.CAP_ANY
        LOGGER.info("Scanning camera indices 0..3 for active hardware...")
        for i in range(4):
            try:
                cap = cv2.VideoCapture(i, scan_backend)
                if cap.isOpened():
                    available_cameras.append(i)
                    cap.release()
            except Exception as e:
                LOGGER.warning("Camera index %d scan failed: %s", i, e)

        LOGGER.info("Camera scan results: %s", available_cameras)
        if not available_cameras:
            LOGGER.error("No cameras detected! Defaulting to index %d", self.config.camera.index)
        elif self.config.camera.index not in available_cameras:
            fallback = available_cameras[0]
            LOGGER.warning("Configured camera index %d unavailable. Falling back to index %d", self.config.camera.index, fallback)
            # Override configured index safely for dataclass-based config.
            self.config = replace(self.config, camera=replace(self.config.camera, index=fallback))

    def _load_models(self) -> None:
        """Instantiate models and build fusion structures."""
        # Load ResourceManager bindings
        self.resource_manager.load_hardware_and_runtimes()
        self.resource_manager.load_reasoning_and_speech()
        
        # Instantiate fusion processors
        geometry = PathGeometryAnalyzer()
        safety = SafetyEvaluator(
            min_safe_area_ratio=self.config.navigation.min_safe_area_ratio,
            min_bottom_width_ratio=self.config.navigation.min_bottom_width_ratio,
            minimum_clearance=self.config.navigation.minimum_clearance,
            caution_clearance=self.config.navigation.caution_clearance,
        )
        decision = NavigationDecisionEngine(
            frame_width=self.config.pathvision.model_width,
            deadband_ratio=self.config.navigation.deadband_ratio,
        )
        self.scene_fusion = SceneFusion(
            geometry=geometry,
            safety=safety,
            decision=decision,
            nearest_obstacle_quantile=self.config.depth.nearest_obstacle_quantile,
        )
        
        self.situation_manager = SituationManager()
        self.conversation_memory = ConversationMemory(reassurance_interval=45.0)
        self.memory = SceneMemory(max_items=8)

        # Setup HealthMonitor (created before Scheduler so it can receive heartbeats)
        self.health_monitor = HealthMonitor(
            event_bus=self.event_bus,
            resource_manager=self.resource_manager,
            scheduler=None,  # Will be set after scheduler creation
        )

        # Setup Scheduler
        self.scheduler = Scheduler(
            resource_manager=self.resource_manager,
            world_model=self.world_model,
            event_bus=self.event_bus,
            config=self.config,
            scene_fusion=self.scene_fusion,
            health_monitor=self.health_monitor,
        )
        self.health_monitor.scheduler = self.scheduler

    def _announce_system_online(self) -> None:
        """Announce system online using speech synthesizer."""
        if self.resource_manager.speaker:
            LOGGER.info("Announcing system online...")
            self.resource_manager.speaker.speak("PathVision system online.", blocking=True)
            time.sleep(0.5)
            self.resource_manager.speaker.speak("Ready to scan.", blocking=True)

    def _start_worker_threads(self) -> None:
        """Launch background run loops and register watchdogs."""
        # 1. Start Telemetry Thread
        self._telemetry_thread = threading.Thread(
            target=self._telemetry_worker,
            name="TelemetryWorkerThread",
            daemon=True
        )
        self._telemetry_thread.start()
        
        # 2. Start Scheduler
        self.scheduler.start()
        
        # 3. Start Health Monitor
        self.health_monitor.start()

        # 4. Start reasoning worker thread
        self._reasoning_thread = threading.Thread(
            target=self._reasoning_worker,
            name="QwenReasonerThread",
            daemon=True
        )
        self._reasoning_thread.start()
        
        # Register Scheduler and Telemetry in Health Monitor
        self.health_monitor.register_worker(
            "FastVisionPipeline",
            self.scheduler.start,
            self.scheduler.stop
        )

    def _telemetry_worker(self) -> None:
        """Flush queues to disk — bounded and fault-tolerant."""
        while self._running:
            try:
                task = self.telemetry_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if task is None:
                break

            t_start = time.perf_counter()
            task_id, data = task
            try:
                if task_id == 0:
                    # SceneLogger now supports ScenePacket-like objects directly.
                    self.scene_logger.write(data)
                elif task_id == 1:
                    frame_small, scene_packet, reason = data
                    self.dataset_builder.save_difficult_scene(frame_small, scene_packet, reason)
                elif task_id == 2:
                    timestamp, label, comment = data
                    self.feedback_store.add(timestamp, label, comment)
            except Exception as e:
                LOGGER.debug("Telemetry write failed: %s", e)

            dt_write = (time.perf_counter() - t_start) * 1000.0
            self.telemetry_tasks_consumed += 1
            self.telemetry_write_latencies.append(dt_write)
            if len(self.telemetry_write_latencies) > 100:
                self.telemetry_write_latencies.pop(0)
            self.telemetry_avg_write_latency_ms = float(np.mean(self.telemetry_write_latencies))
            self.telemetry_max_write_latency_ms = max(self.telemetry_max_write_latency_ms, dt_write)
            self.telemetry_queue.task_done()

    def _reasoning_worker(self) -> None:
        """Reactive reasoning worker loop that processes events from the EventBus."""
        # Polling prompt/guidance updates
        last_reason_ts = 0.0
        while self._running:
            time.sleep(0.1)
            
            try:
                # Send heartbeat to HealthMonitor
                self.health_monitor.heartbeat("QwenReasonerThread")
                
                packet = self.world_model.get_latest()
                if packet is None:
                    continue
                    
                now = time.perf_counter()
                if self._manual_scan_active.is_set():
                    continue
                if now < self._reasoning_cooldown_until:
                    continue

                time_since_reason = now - last_reason_ts
                if time_since_reason < self.config.reasoning.update_interval_seconds:
                    continue

                # Resolve situation
                situation, mode = self.situation_manager.resolve(
                    scene=packet,
                    is_startup_scan=False
                )
                
                command = packet.navigation.command.value
                should_reason = (
                    situation != self.conversation_memory.last_situation
                    or command != self.conversation_memory.last_navigation_command
                    or now - self.conversation_memory.last_speech_time >= self.conversation_memory.reassurance_interval
                )
                
                if should_reason and self.resource_manager.reasoner:
                    last_reason_ts = now
                    # Build mock recent summaries
                    recent_summaries = self.memory.recent_summaries()
                    scene_payload = self._build_scene_payload(packet)
                    
                    # Non-blocking generation call
                    response = self._safe_generate(
                        scene_payload=scene_payload,
                        recent_summaries=recent_summaries,
                        mode=mode,
                    )
                    
                    if response:
                        if not self.conversation_memory.should_speak(
                            text=response,
                            situation=situation,
                            command=command,
                            mode=mode,
                            time_now=now,
                        ):
                            continue
                        
                        # Update conversation memory immediately to prevent race conditions
                        self.conversation_memory.update(
                            text=response,
                            situation=situation,
                            command=command,
                            time_now=now,
                        )
                        # Append to SceneMemory to log the context history
                        self.memory.add(packet)
                        
                        # Enqueue Priority 3 Cognitive event
                        self.event_bus.publish(Event(
                            priority=EventPriority.COGNITIVE,
                            event_type=EventType.SPEECH_REQUEST,
                            payload={"text": response, "situation": situation, "command": command, "mode": mode},
                            timestamp=now
                        ))
            except Exception as e:
                LOGGER.error("Error in reasoning worker iteration: %s", e, exc_info=True)

    def _handle_scan_request(self, event: Event) -> None:
        """Asynchronously triggers reasoning scan task without blocking navigation."""
        if self.state_machine.current_state == RuntimeState.ENVIRONMENT_ANALYSIS:
            LOGGER.warning("Environment scan already in progress. Ignoring request.")
            return

        LOGGER.info("Asynchronous environment scan request received.")
        packet = self.world_model.get_latest()
        if packet is None:
            return
            
        mode = event.payload.get("mode", InteractionMode.ORIENTATION)
        
        # Play immediate auditory/VUI confirmation feedback
        if self.resource_manager.speaker:
            feedback_text = "Scanning environment." if mode == InteractionMode.ORIENTATION else "Running deep scan."
            self.resource_manager.speaker.speak(feedback_text, blocking=False, priority=1)
            # Sync conversation memory to prevent redundant prompt generation from overlapping immediately
            now = time.perf_counter()
            self.conversation_memory.update(feedback_text, SituationType.ENVIRONMENT_TRANSITION, packet.navigation.command.value, now)
        
        # Start background scan thread
        def _bg_scan():
            self.state_machine.transition_to(RuntimeState.ENVIRONMENT_ANALYSIS)
            if self.resource_manager.reasoner:
                recent_summaries = self.memory.recent_summaries()
                
                # Build enriched context dict
                scene_payload = {
                    "path_visible": int(packet.path_geometry.path_visible),
                    "path_center_x": -1 if packet.path_geometry.center_x is None else packet.path_geometry.center_x,
                    "path_width_ratio": round(packet.path_geometry.bottom_width_ratio, 3),
                    "safe_area_ratio": round(packet.path_geometry.safe_area_ratio, 3),
                    "nearest_obstacle_distance": round(packet.nav_mesh_rep.clearance, 3),
                    "scene_confidence": round(packet.scene_confidence, 3),
                    "danger_state": packet.safety.state.value,
                    "navigation_recommendation": packet.navigation.command.value,
                    "navigation_confidence": round(packet.navigation.confidence, 3),
                }
                
                response = self.resource_manager.reasoner.generate(
                    scene_payload=scene_payload,
                    recent_memories=recent_summaries,
                    mode_str=mode.value
                )
                if response:
                    # Enqueue speech event
                    self.event_bus.publish(Event(
                        priority=EventPriority.COGNITIVE,
                        event_type=EventType.SPEECH_REQUEST,
                        payload={"text": response, "situation": SituationType.ENVIRONMENT_TRANSITION, "command": packet.navigation.command.value},
                        timestamp=time.perf_counter()
                    ))
                    # Update conversation/scene memories
                    now = time.perf_counter()
                    self.conversation_memory.update(response, SituationType.ENVIRONMENT_TRANSITION, packet.navigation.command.value, now)
                    self.memory.add(packet)
            self.state_machine.transition_to(RuntimeState.NAVIGATING)
            
        threading.Thread(target=_bg_scan, name="BGScanWorker", daemon=True).start()

    def _handle_speech_request(self, event: Event) -> None:
        """Forward speech events to Kokoro Speaker."""
        text = event.payload.get("text", "")
        situation = event.payload.get("situation", SituationType.SAFE_PATH_FORWARD)
        command = event.payload.get("command", "")
        mode = event.payload.get("mode", InteractionMode.GUIDANCE)
        if not text:
            return
        if self.resource_manager.speaker:
            self.resource_manager.speaker.speak(text, blocking=False, priority=event.priority.value)
            self.conversation_memory.update(
                text=text,
                situation=situation,
                command=command,
                time_now=time.perf_counter(),
            )

    def _handle_nav_command(self, event: Event) -> None:
        cmd = event.payload.get("command", "STOP")
        LOGGER.debug("Decoupled Event Handler: Steering command changed to: %s", cmd)

    def _handle_emergency(self, event: Event) -> None:
        now = time.perf_counter()
        if now - self._last_emergency_ts < 3.0:
            return  # Cooldown: don't repeat emergency within 3 seconds
        self._last_emergency_ts = now
        LOGGER.warning("EMERGENCY EVENT DETECTED: Safety is in danger state!")
        if self.resource_manager.speaker:
            self.resource_manager.speaker.speak("STOP. Danger.", blocking=False, priority=0)
            self.conversation_memory.update("STOP. Danger.", SituationType.LOST_SAFE_PATH, "STOP", now)

    def _handle_hazard(self, event: Event) -> None:
        now = time.perf_counter()
        safety = event.payload.get("safety", "caution")
        previous = event.payload.get("previous", "")
        
        if safety == "safe":
            # Obstacle cleared recovery transition
            if previous in ("caution", "danger"):
                if now - self._last_recovery_ts < 5.0:
                    return  # Cooldown: don't repeat recovery announcement within 5 seconds
                self._last_recovery_ts = now
                LOGGER.info("OBSTACLE CLEARED EVENT DETECTED: Safety recovered to safe.")
                if self.resource_manager.speaker:
                    self.resource_manager.speaker.speak("Obstacles cleared. You can continue moving forward.", blocking=False, priority=1)
                    self.conversation_memory.update("Obstacles cleared. You can continue moving forward.", SituationType.SAFE_PATH_FORWARD, "FORWARD", now)
            return

        # Otherwise, safety state is caution
        if now - self._last_hazard_ts < 5.0:
            return  # Cooldown: don't repeat hazard within 5 seconds
        self._last_hazard_ts = now
        LOGGER.warning("HAZARD EVENT DETECTED: Safety caution triggered.")
        if self.resource_manager.speaker:
            self.resource_manager.speaker.speak("Caution. Obstacle ahead.", blocking=False, priority=1)
            self.conversation_memory.update("Caution. Obstacle ahead.", SituationType.OBSTACLE_APPROACHING, "SLOW", now)

    @staticmethod
    def _build_scene_payload(packet: ScenePacket) -> dict[str, Any]:
        """Build a stable structured payload for Qwen from ScenePacket."""
        return {
            "timestamp": str(packet.timestamp),
            "path_visible": int(packet.path_geometry.path_visible),
            "path_center_x": -1 if packet.path_geometry.center_x is None else int(packet.path_geometry.center_x),
            "path_width_px": int(packet.path_geometry.bottom_width_px),
            "path_width_ratio": float(packet.path_geometry.bottom_width_ratio),
            "safe_area_ratio": float(packet.path_geometry.safe_area_ratio),
            "nearest_obstacle_distance": float(packet.nav_mesh_rep.clearance if packet.nav_mesh_rep else 0.0),
            "scene_confidence": float(packet.scene_confidence),
            "danger_state": packet.safety.state.value,
            "navigation_recommendation": packet.navigation.command.value,
            "navigation_confidence": float(packet.navigation.confidence),
        }

    def _safe_generate(
        self,
        scene_payload: dict[str, Any],
        recent_summaries: list[str],
        mode: InteractionMode,
    ) -> str:
        """Run reasoner generation with timeout guard; fallback on timeout/hang."""
        now = time.perf_counter()
        if now < self._reasoning_cooldown_until:
            return fallback_instruction(scene_payload, mode.value)

        # Check thread status to prevent concurrent llama.cpp execution thread leaks
        if self._reasoner_thread_active:
            LOGGER.warning("[LLAMA.CPP] Reasoner thread is active. Returning fallback to prevent thread leak.")
            return fallback_instruction(scene_payload, mode.value)

        if not self._reasoning_singleflight.acquire(timeout=0.05):
            return fallback_instruction(scene_payload, mode.value)
        try:
            reasoner = self.resource_manager.reasoner
            if reasoner is None:
                return fallback_instruction(scene_payload, mode.value)

            result: dict[str, str] = {"text": ""}
            error: dict[str, Exception] = {}
            
            self._reasoner_thread_active = True

            def _run() -> None:
                try:
                    result["text"] = reasoner.generate(
                        scene_payload=scene_payload,
                        recent_memories=recent_summaries,
                        mode_str=mode.value,
                    )
                except Exception as exc:
                    error["exc"] = exc
                finally:
                    self._reasoner_thread_active = False

            worker = threading.Thread(target=_run, name="ReasonerCall", daemon=True)
            worker.start()
            worker.join(timeout=self.config.reasoning.generation_timeout_seconds)
            if worker.is_alive():
                LOGGER.warning("Reasoner timeout in mode=%s. Using fallback.", mode.value)
                self._reasoning_cooldown_until = time.perf_counter() + 2.0
                return fallback_instruction(scene_payload, mode.value)
            if "exc" in error:
                LOGGER.warning("Reasoner error in mode=%s: %s. Using fallback.", mode.value, error["exc"])
                return fallback_instruction(scene_payload, mode.value)
            text = result.get("text", "").strip()
            if not text:
                return ""
            return text
        finally:
            self._reasoning_singleflight.release()
