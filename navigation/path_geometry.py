"""Path geometry extraction utilities for safe-walk mask analysis."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PathGeometryResult:
    """Geometry summary derived from the trusted safe-path mask."""

    path_visible: bool
    center_x: int | None
    safe_area_ratio: float
    bottom_width_px: int
    bottom_width_ratio: float
    safe_pixel_count: int
    frame_width: int
    frame_height: int


class PathGeometryAnalyzer:
    """Extract center and walkable-width metrics from a binary safe mask."""

    def __init__(
        self,
        roi_start_ratio: float = 0.60,
        bottom_band_start_ratio: float = 0.80,
    ) -> None:
        self.roi_start_ratio = roi_start_ratio
        self.bottom_band_start_ratio = bottom_band_start_ratio
        LOGGER.info(
            "PathGeometryAnalyzer initialized | roi_start_ratio=%.2f bottom_band_start_ratio=%.2f",
            roi_start_ratio,
            bottom_band_start_ratio,
        )

    def analyze(self, safe_mask_u8: np.ndarray) -> PathGeometryResult:
        """Compute path geometry from a safe-walk mask.

        Args:
            safe_mask_u8: Binary image where non-zero pixels indicate walkable area.
        """
        if safe_mask_u8.ndim != 2:
            LOGGER.error("Safe mask must be a 2D array, got ndim=%d", safe_mask_u8.ndim)
            raise ValueError("safe_mask_u8 must be a 2D array")

        frame_h, frame_w = safe_mask_u8.shape
        roi_start = int(frame_h * self.roi_start_ratio)
        bottom_band_start = int(frame_h * self.bottom_band_start_ratio)

        roi = safe_mask_u8[roi_start:, :]
        safe_pixel_count = int(cv2.countNonZero(roi))
        total_roi_pixels = max(roi.shape[0] * roi.shape[1], 1)
        safe_area_ratio = float(safe_pixel_count / total_roi_pixels)

        center_x: int | None = None
        if safe_pixel_count > 0:
            _, xs = np.where(roi > 0)
            center_x = int(np.mean(xs))

        bottom_band = safe_mask_u8[bottom_band_start:, :]
        bottom_width_px = self._max_horizontal_span(bottom_band)
        bottom_width_ratio = float(bottom_width_px / max(frame_w, 1))

        result = PathGeometryResult(
            path_visible=center_x is not None,
            center_x=center_x,
            safe_area_ratio=safe_area_ratio,
            bottom_width_px=bottom_width_px,
            bottom_width_ratio=bottom_width_ratio,
            safe_pixel_count=safe_pixel_count,
            frame_width=frame_w,
            frame_height=frame_h,
        )
        LOGGER.debug(
            "Geometry analyzed | visible=%s center_x=%s area_ratio=%.3f bottom_width_ratio=%.3f",
            result.path_visible,
            result.center_x,
            result.safe_area_ratio,
            result.bottom_width_ratio,
        )
        return result

    @staticmethod
    def _max_horizontal_span(mask: np.ndarray) -> int:
        """Return the widest horizontal walkable span in a mask strip."""
        if cv2.countNonZero(mask) == 0:
            return 0
        widths: list[int] = []
        for row in mask:
            non_zero = np.flatnonzero(row > 0)
            if non_zero.size:
                widths.append(int(non_zero[-1] - non_zero[0] + 1))
        return max(widths, default=0)
