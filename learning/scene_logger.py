"""Structured scene logging utilities for offline analysis with error protection."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


class SceneLogger:
    """Append compact scene summaries to a JSONL log file. Safe from write failures."""

    def __init__(self, log_path: Path) -> None:
        self._log_path = log_path
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            LOGGER.warning("Could not create logs directory: %s", exc)

    def write(self, scene: Any) -> None:
        """Persist one fused scene summary in JSONL format. Fails gracefully on I/O errors."""
        try:
            record = self._scene_to_record(scene)
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        except (OSError, IOError) as exc:
            LOGGER.warning("Failed to write telemetry log to %s: %s", self._log_path, exc)
        except Exception as exc:
            LOGGER.warning("Unexpected error during scene logging: %s", exc)

    @staticmethod
    def _scene_to_record(scene: Any) -> dict[str, Any]:
        depth_obj = getattr(scene, "depth", None)
        if depth_obj is not None and hasattr(depth_obj, "mean_value"):
            depth_record: dict[str, Any] = asdict(depth_obj)
        else:
            clearance = None
            nav_mesh_rep = getattr(scene, "nav_mesh_rep", None)
            if nav_mesh_rep is not None:
                clearance = getattr(nav_mesh_rep, "clearance", None)
            depth_record = {
                "min_value": None,
                "max_value": None,
                "mean_value": clearance,
                "nearest_obstacle_distance": clearance,
                "confidence": None,
            }

        qwen_payload = getattr(scene, "qwen_prompt_data", None)
        if qwen_payload is None:
            qwen_payload = {}

        return {
            "timestamp": str(getattr(scene, "timestamp", "")),
            "path_geometry": asdict(getattr(scene, "path_geometry")),
            "depth": depth_record,
            "safety": {
                "state": getattr(getattr(scene, "safety"), "state").value,
                "score": getattr(getattr(scene, "safety"), "score"),
                "reasons": list(getattr(getattr(scene, "safety"), "reasons")),
            },
            "navigation": {
                "command": getattr(getattr(scene, "navigation"), "command").value,
                "confidence": getattr(getattr(scene, "navigation"), "confidence"),
                "rationale": list(getattr(getattr(scene, "navigation"), "rationale")),
            },
            "scene_confidence": float(getattr(scene, "scene_confidence")),
            "qwen_prompt_data": qwen_payload,
        }
