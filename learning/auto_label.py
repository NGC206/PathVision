"""Auto-label heuristics to identify difficult scenes for dataset capture."""

from __future__ import annotations

import logging
from dataclasses import dataclass

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class AutoLabelDecision:
    """Capture decision for learning dataset generation."""

    should_capture: bool
    reason: str


class AutoLabeler:
    """Heuristic rules that mark hard examples for offline retraining."""

    def __init__(self, capture_confidence_threshold: float) -> None:
        self._capture_threshold = capture_confidence_threshold
        LOGGER.info("AutoLabeler initialized with capture threshold: %.2f", capture_confidence_threshold)

    def evaluate(self, scene: object) -> AutoLabelDecision:
        """Return whether a scene should be captured and why."""
        safety = getattr(scene, "safety", None)
        danger_state = getattr(getattr(safety, "state", None), "value", "safe")
        scene_confidence = float(getattr(scene, "scene_confidence", 1.0))
        navigation = getattr(scene, "navigation", None)
        nav_conf = float(getattr(navigation, "confidence", 1.0))
        path_geometry = getattr(scene, "path_geometry", None)
        bottom_width_ratio = float(getattr(path_geometry, "bottom_width_ratio", 1.0))

        if danger_state == "danger":
            LOGGER.info("Auto-label trigger: danger state detected.")
            return AutoLabelDecision(True, "danger_state")
        if scene_confidence < self._capture_threshold:
            LOGGER.info("Auto-label trigger: low scene confidence (%.2f < %.2f).", scene_confidence, self._capture_threshold)
            return AutoLabelDecision(True, "low_scene_confidence")
        if nav_conf < 0.45:
            LOGGER.info("Auto-label trigger: low navigation confidence (%.2f).", nav_conf)
            return AutoLabelDecision(True, "low_navigation_confidence")
        if bottom_width_ratio < 0.12:
            LOGGER.info("Auto-label trigger: very narrow path bottom width ratio (%.2f).", bottom_width_ratio)
            return AutoLabelDecision(True, "very_narrow_path")
        return AutoLabelDecision(False, "confident_scene")
