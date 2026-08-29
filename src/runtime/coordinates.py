from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from geometry.homography import Homography, load_calibration


class CoordinateTransformer:
    """Centralized vectorized image/world coordinate transformation."""

    def __init__(self, calibration_file: str):
        self.calibration_file = str(calibration_file)
        self.homography = Homography(load_calibration(Path(calibration_file)))

    @classmethod
    def from_config(cls, cfg: dict) -> Optional["CoordinateTransformer"]:
        homography_cfg = cfg.get("homography", {})
        enabled = bool(homography_cfg.get("enabled", False))
        calibration_file = homography_cfg.get("calibration_file")

        if not enabled:
            return None
        if not calibration_file:
            raise ValueError("homography.enabled=true requires homography.calibration_file")

        return cls(calibration_file)

    def points_to_world(self, points: np.ndarray) -> np.ndarray:
        if points.shape[0] == 0:
            return np.empty((0, 2), dtype=np.float64)
        return self.homography.project_pixels_to_world(points)

    def bboxes_to_world(self, bboxes: np.ndarray) -> np.ndarray:
        if bboxes.shape[0] == 0:
            return np.empty((0, 4), dtype=np.float64)
        return self.homography.project_bboxes_to_world(bboxes).astype(np.float64, copy=False)

    def polygon_to_world(self, points) -> np.ndarray:
        arr = np.asarray(points, dtype=np.float32)
        if arr.shape[0] == 0:
            return np.empty((0, 2), dtype=np.float64)
        return self.homography.project_polygon_to_world(arr)

    def line_to_world(self, points) -> np.ndarray:
        arr = np.asarray(points, dtype=np.float32)
        if arr.shape != (2, 2):
            raise ValueError("line points must have shape (2, 2)")
        return self.homography.project_pixels_to_world(arr)
