"""Thread-safe state machine for PathVision Runtime v2.0."""

from __future__ import annotations

import logging
import threading
from enum import Enum

LOGGER = logging.getLogger(__name__)


class RuntimeState(str, Enum):
    """Categorized runtime states for PathVision orchestration."""

    BOOTING = "BOOTING"
    INITIALIZING = "INITIALIZING"
    DISCOVERING_HARDWARE = "DISCOVERING_HARDWARE"
    LOADING_MODELS = "LOADING_MODELS"
    WARMING_UP = "WARMING_UP"
    READY = "READY"
    NAVIGATING = "NAVIGATING"
    ENVIRONMENT_ANALYSIS = "ENVIRONMENT_ANALYSIS"
    ERROR = "ERROR"
    SHUTTING_DOWN = "SHUTTING_DOWN"


class RuntimeStateMachine:
    """Manages thread-safe transitions and callbacks for the global state machine."""

    def __init__(self, initial_state: RuntimeState = RuntimeState.BOOTING) -> None:
        self._lock = threading.Lock()
        self._state = initial_state
        self._listeners: list[callable] = []
        LOGGER.info("RuntimeStateMachine initialized at state: %s", self._state.value)

    @property
    def current_state(self) -> RuntimeState:
        """Get the current runtime state."""
        with self._lock:
            return self._state

    def transition_to(self, new_state: RuntimeState) -> None:
        """Transition state and notify all registered listeners."""
        with self._lock:
            old_state = self._state
            if old_state == new_state:
                return
            self._state = new_state
            LOGGER.info("State Transition: %s -> %s", old_state.value, new_state.value)
            listeners_copy = list(self._listeners)

        # Notify listeners outside the lock to avoid deadlocks
        for listener in listeners_copy:
            try:
                listener(old_state, new_state)
            except Exception as e:
                LOGGER.exception("Error in state transition listener: %s", e)

    def register_listener(self, listener: callable) -> None:
        """Register a callback for state transition notifications."""
        with self._lock:
            if listener not in self._listeners:
                self._listeners.append(listener)
