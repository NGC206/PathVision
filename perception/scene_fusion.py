"""Scene fusion utilities that combine path segmentation and depth."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import cv2
import numpy as np

from perception.navigation_mesh import NavigationMesh, MeshBuilder
from navigation.decision import NavigationDecisionEngine, NavigationDecisionResult
from navigation.path_geometry import PathGeometryAnalyzer, PathGeometryResult
from navigation.safety import SafetyAssessment, SafetyEvaluator

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DepthSummary:
    """Compact depth descriptors used for navigation and reasoning."""

    min_value: float
    max_value: float
    mean_value: float
    nearest_obstacle_distance: float
    confidence: float


@dataclass(frozen=True)
class Scene:
    """Lightweight fused scene object used throughout the runtime pipeline."""

    timestamp: str
    path_geometry: PathGeometryResult
    depth: DepthSummary
    safety: SafetyAssessment
    navigation: NavigationDecisionResult
    scene_confidence: float
    qwen_prompt_data: dict[str, float | int | str]
    nav_mesh: NavigationMesh
    safe_mask: np.ndarray = field(repr=False)
    depth_map: np.ndarray = field(repr=False)


class SceneFusion:
    """Build a fused scene state from safe-path mask and depth map."""

    def __init__(
        self,
        geometry: PathGeometryAnalyzer,
        safety: SafetyEvaluator,
        decision: NavigationDecisionEngine,
        nearest_obstacle_quantile: float = 0.10,
    ) -> None:
        self._geometry = geometry
        self._safety = safety
        self._decision = decision
        self._nearest_quantile = nearest_obstacle_quantile
        self._mesh_builder = MeshBuilder()
        LOGGER.info(
            "SceneFusion initialized | nearest_quantile=%.2f",
            nearest_obstacle_quantile,
        )

    def build(self, safe_mask_u8: np.ndarray, depth_map: np.ndarray) -> Scene:
        """Fuse model outputs into a single scene object."""
        geometry = self._geometry.analyze(safe_mask_u8)
        depth_resized = self._resize_depth(depth_map, safe_mask_u8.shape)
        depth_summary = self._summarize_depth(depth_resized, safe_mask_u8)

        # Build Navigation Mesh
        nav_mesh = self._mesh_builder.build(safe_mask_u8, depth_resized)

        scene_confidence = self._scene_confidence(geometry, depth_summary)
        safety = self._safety.evaluate(
            geometry=geometry,
            nearest_obstacle_distance=depth_summary.nearest_obstacle_distance,
            scene_confidence=scene_confidence,
        )
        navigation = self._decision.decide(geometry=geometry, safety=safety, nav_mesh=nav_mesh)
        timestamp = datetime.now(timezone.utc).isoformat()

        qwen_prompt_data: dict[str, float | int | str] = {
            "timestamp": timestamp,
            "path_visible": int(geometry.path_visible),
            "path_center_x": -1 if geometry.center_x is None else geometry.center_x,
            "path_width_px": geometry.bottom_width_px,
            "path_width_ratio": round(geometry.bottom_width_ratio, 3),
            "safe_area_ratio": round(geometry.safe_area_ratio, 3),
            "nearest_obstacle_distance": round(depth_summary.nearest_obstacle_distance, 3),
            "scene_confidence": round(scene_confidence, 3),
            "danger_state": safety.state.value,
            "navigation_recommendation": navigation.command.value,
            "navigation_confidence": round(navigation.confidence, 3),
        }

        LOGGER.debug(
            "Fused Scene created | safety=%s command=%s confidence=%.2f obstacle_dist=%.2f",
            safety.state.value,
            navigation.command.value,
            scene_confidence,
            depth_summary.nearest_obstacle_distance,
        )

        return Scene(
            timestamp=timestamp,
            path_geometry=geometry,
            depth=depth_summary,
            safety=safety,
            navigation=navigation,
            scene_confidence=scene_confidence,
            qwen_prompt_data=qwen_prompt_data,
            nav_mesh=nav_mesh,
            safe_mask=safe_mask_u8,
            depth_map=depth_resized,
        )

    @staticmethod
    def _resize_depth(depth_map: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
        target_h, target_w = target_shape
        if depth_map.shape[:2] == (target_h, target_w):
            return depth_map.astype(np.float32, copy=False)
        return cv2.resize(depth_map, (target_w, target_h), interpolation=cv2.INTER_LINEAR).astype(np.float32, copy=False)

    def _summarize_depth(
        self, depth_map: np.ndarray, safe_mask: np.ndarray | None = None,
    ) -> DepthSummary:
        depth = np.clip(depth_map.astype(np.float32, copy=False), 0.0, 1.0)

        # General statistics over the full frame
        d_min = float(depth.min())
        d_max = float(depth.max())
        d_mean = float(depth.mean())

        # Nearest-obstacle distance: prefer safe-mask region to avoid
        # false triggers from walls / floor outside the traversable path.
        if safe_mask is not None and np.any(safe_mask > 0):
            nearest = float(np.quantile(depth[safe_mask > 0], self._nearest_quantile))
        else:
            nearest = float(np.quantile(depth, self._nearest_quantile))

        spread = d_max - d_min
        confidence = max(0.0, min(1.0, spread))
        return DepthSummary(
            min_value=d_min,
            max_value=d_max,
            mean_value=d_mean,
            nearest_obstacle_distance=nearest,
            confidence=confidence,
        )

    @staticmethod
    def _scene_confidence(geometry: PathGeometryResult, depth_summary: DepthSummary) -> float:
        width_term = min(1.0, geometry.bottom_width_ratio * 2.0)
        area_term = min(1.0, geometry.safe_area_ratio * 2.5)
        confidence = (0.45 * area_term) + (0.35 * width_term) + (0.20 * depth_summary.confidence)
        return max(0.0, min(1.0, float(confidence)))
