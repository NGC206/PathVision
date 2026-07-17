"""Safety assessment for fused scene information."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

from navigation.path_geometry import PathGeometryResult

LOGGER = logging.getLogger(__name__)


class DangerState(str, Enum):
    """Discrete safety state used by the navigation planner."""

    SAFE = "safe"
    CAUTION = "caution"
    DANGER = "danger"


@dataclass(frozen=True)
class SafetyAssessment:
    """Outcome of the safety evaluation stage."""

    state: DangerState
    score: float
    reasons: tuple[str, ...]


class SafetyEvaluator:
    """Compute danger state and confidence score from geometry + depth."""

    def __init__(
        self,
        min_safe_area_ratio: float,
        min_bottom_width_ratio: float,
        minimum_clearance: float,
        caution_clearance: float,
    ) -> None:
        self.min_safe_area_ratio = min_safe_area_ratio
        self.min_bottom_width_ratio = min_bottom_width_ratio
        self.minimum_clearance = minimum_clearance
        self.caution_clearance = caution_clearance
        LOGGER.info(
            "SafetyEvaluator initialized | min_area=%.2f min_width=%.2f clearance_min=%.2f clearance_caution=%.2f",
            min_safe_area_ratio,
            min_bottom_width_ratio,
            minimum_clearance,
            caution_clearance,
        )

    def evaluate(
        self,
        geometry: PathGeometryResult,
        nearest_obstacle_distance: float,
        scene_confidence: float,
    ) -> SafetyAssessment:
        """Evaluate scene safety from geometric path quality and depth clearance."""
        reasons: list[str] = []

        if not geometry.path_visible:
            reasons.append("no_path_visible")
        if geometry.safe_area_ratio < self.min_safe_area_ratio:
            reasons.append("low_safe_area")
        if geometry.bottom_width_ratio < self.min_bottom_width_ratio:
            reasons.append("narrow_walkway")
        if nearest_obstacle_distance < self.minimum_clearance:
            reasons.append("obstacle_too_close")

        if reasons:
            state = DangerState.DANGER
        elif nearest_obstacle_distance < self.caution_clearance:
            state = DangerState.CAUTION
            reasons.append("limited_clearance")
        elif scene_confidence < 0.45:
            state = DangerState.CAUTION
            reasons.append("low_scene_confidence")
        else:
            state = DangerState.SAFE
            reasons.append("path_clear")

        score = self._compute_score(geometry, nearest_obstacle_distance, scene_confidence, state)
        result = SafetyAssessment(state=state, score=score, reasons=tuple(reasons))
        LOGGER.debug(
            "Safety Assessed: %s | score=%.2f reasons=%s",
            state.value.upper(),
            score,
            reasons,
        )
        return result

    def _compute_score(
        self,
        geometry: PathGeometryResult,
        nearest_obstacle_distance: float,
        scene_confidence: float,
        state: DangerState,
    ) -> float:
        width_term = min(1.0, geometry.bottom_width_ratio / max(self.min_bottom_width_ratio, 1e-6))
        area_term = min(1.0, geometry.safe_area_ratio / max(self.min_safe_area_ratio, 1e-6))
        clearance_term = min(1.0, nearest_obstacle_distance / max(self.caution_clearance, 1e-6))
        score = (0.35 * width_term) + (0.30 * area_term) + (0.20 * clearance_term) + (0.15 * scene_confidence)

        if state == DangerState.DANGER:
            score *= 0.45
        elif state == DangerState.CAUTION:
            score *= 0.75
        return max(0.0, min(1.0, float(score)))
