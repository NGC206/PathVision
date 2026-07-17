"""Thread-safe World Model and Immutable ScenePacket for PathVision v2.0."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any
import numpy as np

from perception.navigation_mesh import NavigationMesh
from navigation.path_geometry import PathGeometryResult
from navigation.safety import SafetyAssessment
from navigation.decision import NavigationDecisionResult
from reasoning.situation_manager import SituationType

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class NavMeshRepresentation:
    """Immutable, thread-safe representation of the Navigation Mesh."""

    center_line: list[tuple[float, float]]
    left_boundary: list[tuple[float, float]]
    right_boundary: list[tuple[float, float]]
    walkable_corridor: np.ndarray
    spline_coefficients: list[float] = field(default_factory=list)
    curvature: float = 0.0
    forward_distance: float = 0.0
    clearance: float = 1.0
    confidence: float = 1.0


@dataclass(frozen=True)
class ScenePacket:
    """Versioned, read-only snapshot representing the system's entire world state."""

    frame_id: int
    timestamp: str
    frame: np.ndarray
    depth_map: np.ndarray
    safe_mask: np.ndarray
    nav_mesh: NavigationMesh
    nav_mesh_rep: NavMeshRepresentation
    path_geometry: PathGeometryResult
    safety: SafetyAssessment
    navigation: NavigationDecisionResult
    situation: SituationType
    scene_confidence: float
    lighting: str = "normal"
    detected_objects: list[dict[str, Any]] = field(default_factory=list)
    guidance: str = ""
    version: int = 2


class WorldModel:
    """Thread-safe atomic container storing the latest system ScenePacket."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest_packet: ScenePacket | None = None
        LOGGER.info("WorldModel initialized.")

    def update(self, packet: ScenePacket) -> None:
        """Atomically set the latest scene packet."""
        with self._lock:
            # Set array flags writeable = False to enforce immutability
            if packet.frame is not None:
                packet.frame.flags.writeable = False
            if packet.depth_map is not None:
                packet.depth_map.flags.writeable = False
            if packet.safe_mask is not None:
                packet.safe_mask.flags.writeable = False
            self._latest_packet = packet
            LOGGER.debug("WorldModel updated with Frame ID: %d", packet.frame_id)

    def get_latest(self) -> ScenePacket | None:
        """Retrieve the latest scene packet."""
        with self._lock:
            return self._latest_packet
