"""Short-horizon scene memory ring buffer used to stabilize reasoning prompts."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SceneMemoryItem:
    """Compact scene memory entry."""

    timestamp: str
    recommendation: str
    danger_state: str
    confidence: float
    nearest_obstacle_distance: float


class SceneMemory:
    """Fixed-size ring buffer of recent scene summaries. Safe and logged."""

    def __init__(self, max_items: int = 8) -> None:
        self._items: deque[SceneMemoryItem] = deque(maxlen=max_items)
        LOGGER.info("SceneMemory initialized with capacity %d.", max_items)

    def add(self, scene: object) -> None:
        """Append a scene summary to memory."""
        timestamp = str(getattr(scene, "timestamp", ""))
        navigation = getattr(scene, "navigation", None)
        safety = getattr(scene, "safety", None)
        scene_confidence = float(getattr(scene, "scene_confidence", 0.0))
        depth = getattr(scene, "depth", None)
        nav_mesh_rep = getattr(scene, "nav_mesh_rep", None)

        nearest_obstacle_distance = 0.0
        if depth is not None and hasattr(depth, "nearest_obstacle_distance"):
            nearest_obstacle_distance = float(getattr(depth, "nearest_obstacle_distance"))
        elif nav_mesh_rep is not None and hasattr(nav_mesh_rep, "clearance"):
            nearest_obstacle_distance = float(getattr(nav_mesh_rep, "clearance"))

        item = SceneMemoryItem(
            timestamp=timestamp,
            recommendation=getattr(getattr(navigation, "command", ""), "value", "UNKNOWN"),
            danger_state=getattr(getattr(safety, "state", ""), "value", "unknown"),
            confidence=scene_confidence,
            nearest_obstacle_distance=nearest_obstacle_distance,
        )
        self._items.append(item)
        LOGGER.debug("Memory appended: cmd=%s, danger=%s", item.recommendation, item.danger_state)

    def recent_summaries(self) -> list[str]:
        """Return readable summaries for prompt construction."""
        return [
            (
                f"{item.timestamp} | cmd={item.recommendation} | "
                f"danger={item.danger_state} | conf={item.confidence:.2f} | "
                f"clearance={item.nearest_obstacle_distance:.2f}"
            )
            for item in self._items
        ]
