"""Navigation decision logic based on scene safety and path geometry."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
import numpy as np

from navigation.path_geometry import PathGeometryResult
from navigation.safety import DangerState, SafetyAssessment
from perception.navigation_mesh import NavigationMesh

LOGGER = logging.getLogger(__name__)


class NavigationCommand(str, Enum):
    """Set of discrete movement recommendations."""

    FORWARD = "FORWARD"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    SLOW = "SLOW"
    STOP = "STOP"


@dataclass(frozen=True)
class NavigationDecisionResult:
    """Output of the decision module."""

    command: NavigationCommand
    confidence: float
    rationale: tuple[str, ...]


class NavigationDecisionEngine:
    """Map geometry + safety + mesh into a deterministic movement command."""

    def __init__(
        self,
        frame_width: int,
        deadband_ratio: float,
        blocked_confirm_frames: int = 3,
        unblock_confirm_frames: int = 2,
    ) -> None:
        self.frame_width = frame_width
        self.center_x = frame_width // 2
        self.deadband_px = int(frame_width * deadband_ratio)
        self.blocked_confirm_frames = blocked_confirm_frames
        self.unblock_confirm_frames = unblock_confirm_frames
        self._mesh_blocked_streak = 0
        self._mesh_clear_streak = 0
        LOGGER.info(
            "NavigationDecisionEngine initialized | width=%d center=%d deadband=%d px blocked_confirm=%d",
            frame_width,
            self.center_x,
            self.deadband_px,
            blocked_confirm_frames,
        )

    def decide(
        self,
        geometry: PathGeometryResult,
        safety: SafetyAssessment,
        nav_mesh: NavigationMesh | None = None,
    ) -> NavigationDecisionResult:
        """Generate a movement command and confidence."""
        rationale: list[str] = []

        # Apply mesh blockage hysteresis to avoid single-frame STOP spikes.
        mesh_blocked_raw = nav_mesh is not None and nav_mesh.is_blocked
        if mesh_blocked_raw:
            self._mesh_blocked_streak += 1
            self._mesh_clear_streak = 0
        else:
            self._mesh_clear_streak += 1
            if self._mesh_clear_streak >= self.unblock_confirm_frames:
                self._mesh_blocked_streak = 0
        mesh_blocked_persistent = self._mesh_blocked_streak >= self.blocked_confirm_frames

        # Emergency Stop only for true danger/no path/persistent blockage.
        if safety.state == DangerState.DANGER or not geometry.path_visible or mesh_blocked_persistent:
            if mesh_blocked_persistent:
                rationale.append("mesh_blocked")
            else:
                rationale.append("danger_or_no_path")
            result = NavigationDecisionResult(
                command=NavigationCommand.STOP,
                confidence=min(0.95, max(0.55, 1.0 - safety.score)),
                rationale=tuple(rationale + list(safety.reasons)),
            )
            LOGGER.debug("Navigation Decision: STOP (reason=%s)", result.rationale)
            return result

        # Compute centerline steering target using Navigation Mesh if available
        steer_x: float | None = None
        if nav_mesh is not None and len(nav_mesh.centerline) > 0:
            # Weighted average favoring mid-range/look-ahead nodes to anticipate turns
            # Centerline points are sorted from bottom-up (y_start to y_end)
            pts = nav_mesh.centerline
            if len(pts) >= 3:
                # Give higher weight to the look-ahead points
                weights = np.linspace(1.0, 2.0, len(pts))
                xs = np.array([pt[0] for pt in pts])
                steer_x = float(np.average(xs, weights=weights))
                rationale.append("mesh_lookahead")
            else:
                steer_x = float(np.mean([pt[0] for pt in pts]))
                rationale.append("mesh_mean")
        else:
            # Fall back to geometry centroid
            steer_x = float(geometry.center_x) if geometry.center_x is not None else None
            if steer_x is not None:
                rationale.append("fallback_geometry")

        if steer_x is None:
            rationale.append("missing_steering_target")
            result = NavigationDecisionResult(
                command=NavigationCommand.SLOW,
                confidence=0.45,
                rationale=tuple(rationale + list(safety.reasons)),
            )
            LOGGER.debug("Navigation Decision: SLOW (missing target)")
            return result

        offset = steer_x - self.center_x
        if abs(offset) <= self.deadband_px:
            # If mesh is temporarily uncertain, prefer SLOW over STOP.
            if mesh_blocked_raw and not mesh_blocked_persistent:
                command = NavigationCommand.SLOW
                rationale.append("temporary_mesh_uncertainty")
            else:
                command = NavigationCommand.FORWARD if safety.state == DangerState.SAFE else NavigationCommand.SLOW
            rationale.append("centered_path")
        elif offset < 0:
            command = NavigationCommand.LEFT
            rationale.append("path_left")
        else:
            command = NavigationCommand.RIGHT
            rationale.append("path_right")

        confidence = self._confidence_from_offset(abs(offset), safety.score)
        result = NavigationDecisionResult(
            command=command,
            confidence=confidence,
            rationale=tuple(rationale + list(safety.reasons)),
        )
        LOGGER.debug(
            "Navigation Decision: %s | confidence=%.2f offset=%.1f",
            command.value,
            confidence,
            offset,
        )
        return result

    def _confidence_from_offset(self, abs_offset_px: float, safety_score: float) -> float:
        offset_penalty = min(1.0, abs_offset_px / max(self.frame_width * 0.5, 1.0))
        confidence = (0.70 * safety_score) + (0.30 * (1.0 - offset_penalty))
        return max(0.0, min(1.0, float(confidence)))
