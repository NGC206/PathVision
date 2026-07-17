"""Offline dataset export for difficult scenes with error protections."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


class DatasetBuilder:
    """Store RGB/depth/mask snapshots and metadata for retraining. Safe from I/O errors."""

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir
        try:
            self._output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            LOGGER.warning("Could not create dataset output directory: %s", exc)

    def save_difficult_scene(
        self,
        rgb_frame_bgr: np.ndarray,
        scene: object,
        reason: str,
    ) -> Path | None:
        """Save one difficult-scene sample. Fails gracefully on file errors."""
        try:
            timestamp = str(getattr(scene, "timestamp", "unknown"))
            sample_id = timestamp.replace(":", "-").replace(".", "-")
            sample_dir = self._output_dir / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)

            rgb_path = sample_dir / "rgb.png"
            depth_path = sample_dir / "depth.npy"
            mask_path = sample_dir / "safe_mask.png"
            meta_path = sample_dir / "meta.json"

            # Perform write operations with defensive guards
            cv2.imwrite(str(rgb_path), rgb_frame_bgr)
            depth_map = getattr(scene, "depth_map", None)
            safe_mask = getattr(scene, "safe_mask", None)
            if depth_map is None or safe_mask is None:
                return None
            np.save(depth_path, depth_map.astype(np.float32, copy=False))
            cv2.imwrite(str(mask_path), safe_mask)

            safety = getattr(scene, "safety", None)
            navigation = getattr(scene, "navigation", None)
            path_geometry = getattr(scene, "path_geometry", None)
            scene_confidence = float(getattr(scene, "scene_confidence", 0.0))
            qwen_prompt_data = getattr(scene, "qwen_prompt_data", {}) or {}
            depth_obj = getattr(scene, "depth", None)
            nav_mesh_rep = getattr(scene, "nav_mesh_rep", None)
            nearest = None
            dmin = None
            dmax = None
            dmean = None
            dconf = None
            if depth_obj is not None and hasattr(depth_obj, "nearest_obstacle_distance"):
                nearest = getattr(depth_obj, "nearest_obstacle_distance", None)
                dmin = getattr(depth_obj, "min_value", None)
                dmax = getattr(depth_obj, "max_value", None)
                dmean = getattr(depth_obj, "mean_value", None)
                dconf = getattr(depth_obj, "confidence", None)
            elif nav_mesh_rep is not None:
                nearest = getattr(nav_mesh_rep, "clearance", None)

            metadata: dict[str, Any] = {
                "timestamp": timestamp,
                "reason": reason,
                "scene_confidence": scene_confidence,
                "danger_state": getattr(getattr(safety, "state", None), "value", "unknown"),
                "navigation_command": getattr(getattr(navigation, "command", None), "value", "UNKNOWN"),
                "navigation_confidence": float(getattr(navigation, "confidence", 0.0)),
                "path_geometry": {
                    "path_visible": bool(getattr(path_geometry, "path_visible", False)),
                    "center_x": getattr(path_geometry, "center_x", None),
                    "safe_area_ratio": float(getattr(path_geometry, "safe_area_ratio", 0.0)),
                    "bottom_width_px": int(getattr(path_geometry, "bottom_width_px", 0)),
                    "bottom_width_ratio": float(getattr(path_geometry, "bottom_width_ratio", 0.0)),
                },
                "depth": {
                    "min": dmin,
                    "max": dmax,
                    "mean": dmean,
                    "nearest_obstacle_distance": nearest,
                    "confidence": dconf,
                },
                "qwen_prompt_data": qwen_prompt_data,
            }
            
            with meta_path.open("w", encoding="utf-8") as handle:
                json.dump(metadata, handle, ensure_ascii=True, indent=2)

            LOGGER.info("Saved difficult scene sample: %s", sample_dir)
            return sample_dir
        except (OSError, IOError) as exc:
            LOGGER.warning("Failed to save difficult scene sample due to I/O error: %s", exc)
            return None
        except Exception as exc:
            LOGGER.warning("Unexpected error saving difficult scene sample: %s", exc)
            return None
