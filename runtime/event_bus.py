"""Priority-based Event Bus for PathVision Runtime v2.0."""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Callable

LOGGER = logging.getLogger(__name__)


class EventPriority(IntEnum):
    """Categorized event priority levels where lower number means higher priority."""

    EMERGENCY = 0    # Emergency Stop, Collision, Fall, Immediate Danger
    HAZARD = 1       # Obstacle, Road Edge, Vehicle, Drop
    NAVIGATION = 2   # Continue, Slight Left, Slight Right
    COGNITIVE = 3    # Environment Scan, Descriptions, Conversation
    DIAGNOSTICS = 4  # Telemetry, Dataset, Learning


class EventType(str, Enum):
    """Specific event categories propagated over the bus."""

    EMERGENCY_STOP = "emergency_stop"
    COLLISION_WARNING = "collision_warning"
    FALL_DETECTED = "fall_detected"
    
    OBSTACLE_DETECTED = "obstacle_detected"
    ROAD_EDGE_NEARBY = "road_edge_nearby"
    DROP_HAZARD = "drop_hazard"
    
    NAV_COMMAND = "nav_command"
    
    SCAN_REQUEST = "scan_request"
    SCAN_COMPLETED = "scan_completed"
    SPEECH_REQUEST = "speech_request"
    
    TELEMETRY_LOG = "telemetry_log"
    DATASET_SAMPLE = "dataset_sample"


@dataclass(order=True)
class Event:
    """Immutable event payload sent over the Event Bus."""

    priority: EventPriority
    event_type: EventType = field(compare=False)
    payload: dict[str, Any] = field(default_factory=dict, compare=False)
    timestamp: float = field(default_factory=lambda: 0.0, compare=False)


class EventBus:
    """Asynchronous, priority-routed event subscription and delivery registry."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._listeners: dict[EventType, list[Callable[[Event], None]]] = {}
        self._global_listeners: list[Callable[[Event], None]] = []
        self._queue: queue.PriorityQueue[Event] = queue.PriorityQueue()
        self._running = False
        self._worker_thread: threading.Thread | None = None
        LOGGER.info("EventBus initialized.")

    def subscribe(self, event_type: EventType, callback: Callable[[Event], None]) -> None:
        """Register a callback for a specific event type."""
        with self._lock:
            if event_type not in self._listeners:
                self._listeners[event_type] = []
            if callback not in self._listeners[event_type]:
                self._listeners[event_type].append(callback)
            LOGGER.debug("Callback subscribed to: %s", event_type.value)

    def subscribe_all(self, callback: Callable[[Event], None]) -> None:
        """Register a callback for all event publications."""
        with self._lock:
            if callback not in self._global_listeners:
                self._global_listeners.append(callback)

    def publish(self, event: Event) -> None:
        """Enqueue an event for asynchronous dispatch based on priority."""
        self._queue.put(event)
        LOGGER.debug("Published event: %s (Priority: %d)", event.event_type.value, event.priority.value)

    def start(self) -> None:
        """Start the background event dispatch loop."""
        with self._lock:
            if self._running:
                return
            self._running = True
            self._worker_thread = threading.Thread(target=self._dispatch_loop, name="EventBusWorker", daemon=True)
            self._worker_thread.start()
            LOGGER.info("EventBus dispatcher started.")

    def stop(self) -> None:
        """Shutdown the event dispatch system."""
        with self._lock:
            if not self._running:
                return
            self._running = False
        
        # Unblock the queue with a dummy diagnostic event to drop out of get()
        self._queue.put(Event(priority=EventPriority.DIAGNOSTICS, event_type=EventType.TELEMETRY_LOG))
        
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
        LOGGER.info("EventBus dispatcher stopped.")

    def _dispatch_loop(self) -> None:
        """Continuously pop events from PriorityQueue and invoke callbacks."""
        while self._running:
            try:
                event = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if not self._running:
                break

            # Snapshot listeners under a single short lock
            with self._lock:
                callbacks = list(self._global_listeners)
                if event.event_type in self._listeners:
                    callbacks.extend(self._listeners[event.event_type])

            # Invoke callbacks outside the lock to avoid blocking publishers
            for callback in callbacks:
                try:
                    callback(event)
                except Exception as e:
                    LOGGER.exception("Error executing event callback: %s", e)
            
            self._queue.task_done()
