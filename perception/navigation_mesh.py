from __future__ import annotations

import time
import logging
import numpy as np
import cv2
from dataclasses import dataclass, field

LOGGER = logging.getLogger(__name__)

@dataclass
class MeshNode:
    """Represents a geometric node in the corridor boundary graph."""
    node_id: int
    is_left: bool
    x: float
    y: float
    depth_nearest: float = 1.0
    depth_mean: float = 1.0
    depth_gradient: float = 0.0
    clearance: float = 1.0
    confidence: float = 1.0
    local_width: float = 0.0
    curvature: float = 0.0
    slope: float = 0.0
    obstacle_distance: float = 1.0
    walkable_score: float = 1.0
    timestamp: float = field(default_factory=time.perf_counter)
    blocked: bool = False

@dataclass
class NavigationMesh:
    """Coroutine mesh representing safe corridors and boundary geometry."""
    nodes: dict[int, MeshNode] = field(default_factory=dict)
    edges: list[tuple[int, int]] = field(default_factory=list)
    centerline: list[tuple[float, float]] = field(default_factory=list)
    average_width: float = 0.0
    nearest_obstacle_dist: float = 1.0
    curvature_index: float = 0.0
    is_blocked: bool = False

class MeshBuilder:
    """Builds, smooths, and tracks a Navigation Mesh from a Stable Mask and Depth Map."""

    def __init__(
        self,
        vertical_step: int = 10,
        temporal_alpha: float = 0.4,
        clearance_threshold: float = 0.06
    ) -> None:
        self.vertical_step = vertical_step
        self.temporal_alpha = temporal_alpha
        self.clearance_threshold = clearance_threshold
        
        # Track previous nodes for temporal consistency: { (y, is_left): MeshNode }
        self._prev_nodes: dict[tuple[int, bool], MeshNode] = {}
        self._next_node_id = 0
        self._smoothed_clearance: float | None = None
        LOGGER.info(
            "MeshBuilder initialized | vertical_step=%d alpha=%.2f clearance_threshold=%.2f",
            vertical_step, temporal_alpha, clearance_threshold
        )

    def build(self, safe_mask: np.ndarray, depth_map: np.ndarray) -> NavigationMesh:
        """Construct the NavigationMesh from stable mask and depth map."""
        h, w = safe_mask.shape
        nodes: dict[int, MeshNode] = {}
        edges: list[tuple[int, int]] = []
        centerline: list[tuple[float, float]] = []
        
        # Sample rows vertically from bottom-up (ROI region)
        # Typically the corridor is in the lower 60% of the frame
        y_start = h - 2
        y_end = int(h * 0.4)
        rows_to_sample = range(y_start, y_end, -self.vertical_step)
        
        row_nodes_map: dict[int, tuple[int, int]] = {} # y -> (left_node_id, right_node_id)
        
        total_width = 0.0
        width_count = 0
        row_clearances: list[float] = []
        blocked_row_streak = 0
        max_blocked_row_streak = 0
        
        for y in rows_to_sample:
            row_mask = safe_mask[y, :]
            xs = np.flatnonzero(row_mask > 0)
            
            if len(xs) < 5: # Not enough walkable pixels on this row
                continue
                
            x_l_raw = float(xs[0])
            x_r_raw = float(xs[-1])
            
            # Temporal Tracking & Smoothing
            prev_l = self._prev_nodes.get((y, True))
            prev_r = self._prev_nodes.get((y, False))
            
            if prev_l is not None:
                x_l = self.temporal_alpha * x_l_raw + (1.0 - self.temporal_alpha) * prev_l.x
            else:
                x_l = x_l_raw
                
            if prev_r is not None:
                x_r = self.temporal_alpha * x_r_raw + (1.0 - self.temporal_alpha) * prev_r.x
            else:
                x_r = x_r_raw
            
            # Calculate Local width
            width = x_r - x_l
            if width <= 0:
                continue
                
            total_width += width
            width_count += 1
            
            # Depth sampling at node coordinates (3x3 window)
            depth_l = self._sample_depth(depth_map, int(x_l), y)
            depth_r = self._sample_depth(depth_map, int(x_r), y)
            row_clearance = self._sample_corridor_clearance(depth_map, int(x_l), int(x_r), y)
            row_clearances.append(row_clearance)

            # Local blockage criteria with robust corridor signal.
            blocked_l = depth_l["nearest"] < self.clearance_threshold
            blocked_r = depth_r["nearest"] < self.clearance_threshold
            row_blocked = row_clearance < self.clearance_threshold
            if row_blocked:
                blocked_row_streak += 1
                max_blocked_row_streak = max(max_blocked_row_streak, blocked_row_streak)
            else:
                blocked_row_streak = 0

            # Create Node IDs
            id_l = self._get_next_id()
            id_r = self._get_next_id()
            
            # Curvature calculation relative to previous step below
            curvature = 0.0
            prev_y = y + self.vertical_step
            if prev_y in row_nodes_map:
                prev_l_id, prev_r_id = row_nodes_map[prev_y]
                prev_center = (nodes[prev_l_id].x + nodes[prev_r_id].x) / 2.0
                curr_center = (x_l + x_r) / 2.0
                curvature = curr_center - prev_center
            
            # Instantiate Nodes
            node_l = MeshNode(
                node_id=id_l,
                is_left=True,
                x=x_l,
                y=float(y),
                depth_nearest=depth_l["nearest"],
                depth_mean=depth_l["mean"],
                clearance=depth_l["nearest"],
                local_width=width,
                curvature=curvature,
                walkable_score=0.0 if (blocked_l or row_blocked) else 1.0,
                blocked=(blocked_l or row_blocked)
            )
            node_r = MeshNode(
                node_id=id_r,
                is_left=False,
                x=x_r,
                y=float(y),
                depth_nearest=depth_r["nearest"],
                depth_mean=depth_r["mean"],
                clearance=depth_r["nearest"],
                local_width=width,
                curvature=curvature,
                walkable_score=0.0 if (blocked_r or row_blocked) else 1.0,
                blocked=(blocked_r or row_blocked)
            )
            
            nodes[id_l] = node_l
            nodes[id_r] = node_r
            
            # Cache for next frame
            self._prev_nodes[(y, True)] = node_l
            self._prev_nodes[(y, False)] = node_r
            row_nodes_map[y] = (id_l, id_r)
            
            # Connect lateral edge (left-right neighbors)
            edges.append((id_l, id_r))
            
            # Connect longitudinal edges (connect to row below)
            if prev_y in row_nodes_map:
                p_l, p_r = row_nodes_map[prev_y]
                edges.append((p_l, id_l))
                edges.append((p_r, id_r))
                
            # Local centerline rerouting
            if blocked_l and blocked_r:
                # Both blocked
                pass
            elif blocked_l:
                # Reroute towards right node
                center_x = x_l + width * 0.75
                centerline.append((center_x, float(y)))
            elif blocked_r:
                # Reroute towards left node
                center_x = x_l + width * 0.25
                centerline.append((center_x, float(y)))
            else:
                # Center centerline
                center_x = (x_l + x_r) / 2.0
                centerline.append((center_x, float(y)))

        # Clean up stale nodes from cache if they weren't matched this frame
        current_ys = set(row_nodes_map.keys())
        for key in list(self._prev_nodes.keys()):
            if key[0] not in current_ys:
                self._prev_nodes.pop(key)
                
        # Calculate summary parameters
        avg_w = total_width / max(width_count, 1)
        curv_index = 0.0
        if len(centerline) >= 2:
            curv_index = float(np.mean([abs(n.curvature) for n in nodes.values()]))
            
        if row_clearances:
            robust_clearance = float(np.quantile(np.asarray(row_clearances, dtype=np.float32), 0.10))
        else:
            robust_clearance = 1.0

        if self._smoothed_clearance is None:
            self._smoothed_clearance = robust_clearance
        else:
            self._smoothed_clearance = (
                (1.0 - self.temporal_alpha) * self._smoothed_clearance
                + self.temporal_alpha * robust_clearance
            )

        # Block only when low clearance persists across consecutive rows.
        is_blocked = (
            max_blocked_row_streak >= 3
            and self._smoothed_clearance < self.clearance_threshold
        )

        return NavigationMesh(
            nodes=nodes,
            edges=edges,
            centerline=centerline,
            average_width=avg_w,
            nearest_obstacle_dist=float(self._smoothed_clearance),
            curvature_index=curv_index,
            is_blocked=is_blocked
        )

    def _sample_depth(self, depth_map: np.ndarray, x: int, y: int) -> dict[str, float]:
        """Sample local window depth metrics to check local clearance."""
        h, w = depth_map.shape[:2]
        # Restrict bounds
        x = max(2, min(w - 3, x))
        y = max(2, min(h - 3, y))
        
        window = depth_map[y-2:y+3, x-2:x+3]
        return {
            "nearest": float(np.min(window)),
            "mean": float(np.mean(window))
        }

    def _sample_corridor_clearance(self, depth_map: np.ndarray, x_l: int, x_r: int, y: int) -> float:
        """Estimate robust corridor clearance for one row using quantile depth."""
        h, w = depth_map.shape[:2]
        y = max(2, min(h - 3, y))
        left = max(0, min(w - 1, min(x_l, x_r)))
        right = max(0, min(w - 1, max(x_l, x_r)))
        if right - left < 6:
            return 1.0
        band = depth_map[max(0, y - 2):min(h, y + 3), left:right]
        return float(np.quantile(band, 0.10))

    def _get_next_id(self) -> int:
        self._next_node_id += 1
        return self._next_node_id
