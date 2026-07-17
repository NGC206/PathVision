"""Situation Manager to classify environmental context and select conversation modes."""

from __future__ import annotations

import logging
from enum import Enum

LOGGER = logging.getLogger(__name__)


class InteractionMode(str, Enum):
    """Supported user interaction modes."""

    ORIENTATION = "orientation"
    GUIDANCE = "guidance"
    ALERT = "alert"
    DESCRIPTION = "description"
    SCENE_CONTEXT = "scene_context"


class SituationType(str, Enum):
    """Categorized environmental situations."""

    INITIAL_SCAN = "initial_scan"
    SAFE_PATH_FORWARD = "safe_path_forward"
    PATH_NARROWING = "path_narrowing"
    OBSTACLE_APPROACHING = "obstacle_approaching"
    LOST_SAFE_PATH = "lost_safe_path"
    TURNING_LEFT = "turning_left"
    TURNING_RIGHT = "turning_right"
    WAITING = "waiting"
    ENVIRONMENT_TRANSITION = "environment_transition"


class SituationManager:
    """Classifies user surroundings and selects the active InteractionMode."""

    def __init__(self) -> None:
        self._prev_scene: Scene | None = None
        self._waiting_frames = 0
        LOGGER.info("SituationManager initialized.")

    def resolve(
        self,
        scene: object,
        is_startup_scan: bool,
        manual_mode_override: InteractionMode | None = None,
    ) -> tuple[SituationType, InteractionMode]:
        """Classify the current scene situation and map it to an interaction mode."""
        if manual_mode_override is not None:
            # Handle user requested action (like D for description or S for scan)
            situation = SituationType.SAFE_PATH_FORWARD
            if manual_mode_override == InteractionMode.ORIENTATION:
                situation = SituationType.INITIAL_SCAN
            elif manual_mode_override in (InteractionMode.DESCRIPTION, InteractionMode.SCENE_CONTEXT):
                situation = SituationType.ENVIRONMENT_TRANSITION
            return situation, manual_mode_override

        if is_startup_scan:
            return SituationType.INITIAL_SCAN, InteractionMode.ORIENTATION

        # Check for safety alerts (Emergency)
        state_str = scene.safety.state.value
        if state_str == "danger":
            return SituationType.LOST_SAFE_PATH, InteractionMode.ALERT
        if state_str == "caution":
            return SituationType.OBSTACLE_APPROACHING, InteractionMode.ALERT

        # Check for turning requirements
        cmd = scene.navigation.command.value
        if cmd == "LEFT":
            return SituationType.TURNING_LEFT, InteractionMode.ALERT
        if cmd == "RIGHT":
            return SituationType.TURNING_RIGHT, InteractionMode.ALERT

        # Detect environmental transition (moving rooms, crossing doorways)
        # We look for a sudden, significant change in depth profiles
        is_transition = False
        current_depth_mean = self._extract_depth_mean(scene)
        if self._prev_scene is not None:
            prev_depth_mean = self._extract_depth_mean(self._prev_scene)
            depth_diff = abs(current_depth_mean - prev_depth_mean)
            # If mean depth shifts by > 0.25 on a 0-1 scale, we crossed a doorway/entered a new space
            if depth_diff > 0.25:
                is_transition = True
                LOGGER.info("Environmental transition detected: depth mean shifted by %.2f", depth_diff)

        # Track waiting behavior (stationary for a long time)
        if cmd == "STOP":
            self._waiting_frames += 1
        else:
            self._waiting_frames = 0

        # Resolve primary walking situations
        if is_transition:
            situation = SituationType.ENVIRONMENT_TRANSITION
            mode = InteractionMode.ORIENTATION
        elif self._waiting_frames > 60:  # ~2.5 seconds at 25 FPS
            situation = SituationType.WAITING
            mode = InteractionMode.GUIDANCE
        elif scene.path_geometry.bottom_width_ratio < 0.15:
            situation = SituationType.PATH_NARROWING
            mode = InteractionMode.GUIDANCE
        else:
            situation = SituationType.SAFE_PATH_FORWARD
            mode = InteractionMode.GUIDANCE

        self._prev_scene = scene
        return situation, mode

    @staticmethod
    def _extract_depth_mean(scene: object) -> float:
        """Extract depth mean robustly from Scene or ScenePacket-like objects."""
        depth = getattr(scene, "depth", None)
        if depth is not None and hasattr(depth, "mean_value"):
            return float(getattr(depth, "mean_value"))
        if isinstance(depth, (float, int)):
            return float(depth)
        nav_mesh_rep = getattr(scene, "nav_mesh_rep", None)
        if nav_mesh_rep is not None and hasattr(nav_mesh_rep, "clearance"):
            return float(getattr(nav_mesh_rep, "clearance"))
        return 0.5
