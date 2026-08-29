from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from geometry.primitives import Line, Polygon
from geometry.spatial import (
    bbox_any_in_polygon,
    bbox_center_in_polygon,
    bbox_corners_in_polygon,
    image_line_coefficients,
    line_coefficients,
    points_in_polygon,
    polygon_to_mask,
)


class GeometryEngine:
    """Vectorized spatial cache engine for image-space or world-space coordinates."""

    def __init__(
        self,
        lines: Dict[str, Line],
        polygons: Dict[str, Polygon],
        polygon_mode: str = "center",
        coordinate_space: str = "image",
        line_vicinity_thresholds: Optional[Dict[str, float]] = None,
    ):
        self.coordinate_space = self._normalize_coordinate_space(coordinate_space)
        self._lines = lines
        self._polygons = polygons
        self._polygon_mode = polygon_mode
        self._line_vicinity_thresholds = line_vicinity_thresholds or {}

        self._line_ids = list(lines.keys())
        self._line_id_to_idx = {
            line_id: idx
            for idx, line_id in enumerate(self._line_ids)
        }
        self._polygon_masks = self._build_polygon_masks(polygons)

        self._A = None
        self._B = None
        self._C = None
        self._norm = None
        self._thresh = None
        self._build_line_cache(lines)

    @staticmethod
    def _normalize_coordinate_space(value: str) -> str:
        normalized = str(value).lower()
        if normalized not in {"image", "world"}:
            raise ValueError("coordinate_space must be one of: image, world")
        return normalized

    def _build_line_cache(
        self,
        lines: Dict[str, Line],
    ) -> None:
        dtype = np.float64 if self.coordinate_space == "world" else np.float32
        line_count = len(self._line_ids)

        if line_count == 0:
            self._A = np.zeros((0,), dtype=dtype)
            self._B = np.zeros((0,), dtype=dtype)
            self._C = np.zeros((0,), dtype=dtype)
            self._norm = np.ones((0,), dtype=dtype)
            self._thresh = np.zeros((0,), dtype=dtype)
            return

        coeffs = np.asarray([
            self._line_coefficients(lines[line_id])
            for line_id in self._line_ids
        ], dtype=dtype)

        self._A = coeffs[:, 0]
        self._B = coeffs[:, 1]
        self._C = coeffs[:, 2]
        self._norm = coeffs[:, 3]
        self._thresh = np.asarray([
            self._line_threshold(line_id)
            for line_id in self._line_ids
        ], dtype=dtype)

    def _line_coefficients(self, line: Line) -> np.ndarray:
        points = np.asarray(line.points, dtype=np.float64)
        if self.coordinate_space == "image":
            return image_line_coefficients(points)
        return line_coefficients(points)

    def _line_threshold(self, line_id: str) -> float:
        return float(self._line_vicinity_thresholds.get(line_id, 0.0))

    def _build_polygon_masks(
        self,
        polygons: Dict[str, Polygon],
    ) -> Dict[str, Tuple[np.ndarray, int, int]]:
        if self.coordinate_space != "image":
            return {}

        return {
            polygon_id: polygon_to_mask(np.asarray(polygon.points, dtype=np.float64))
            for polygon_id, polygon in polygons.items()
        }

    def line_normal(self, line_id: str) -> Optional[np.ndarray]:
        line_idx = self._line_id_to_idx[line_id]
        direction = np.asarray(
            [self._A[line_idx], self._B[line_idx]],
            dtype=np.float64,
        )
        norm = np.linalg.norm(direction)
        if norm <= 1e-12:
            return None
        return direction / norm

    def compute(
        self,
        points: np.ndarray,
        bboxes: Optional[np.ndarray] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        if points is None:
            raise ValueError("points must be provided")

        x = points[:, 0]
        y = points[:, 1]

        obs_count = points.shape[0]
        line_count = len(self._line_ids)
        distance_dtype = np.float64 if self.coordinate_space == "world" else np.float32

        if line_count > 0:
            signed_distance = (
                self._A[:, None] * x[None, :]
                + self._B[:, None] * y[None, :]
                + self._C[:, None]
            ) / self._norm[:, None]

            distance = np.abs(signed_distance)
            line_cache = {
                "distance": distance,
                "sign": np.sign(signed_distance).astype(np.int8),
                "vicinity_mask": distance < self._thresh[:, None],
            }
        else:
            line_cache = {
                "distance": np.zeros((0, obs_count), dtype=distance_dtype),
                "sign": np.zeros((0, obs_count), dtype=np.int8),
                "vicinity_mask": np.zeros((0, obs_count), dtype=bool),
            }

        polygon_cache = {
            polygon_id: self._polygon_mask(polygon_id, polygon, bboxes, points).reshape(-1)
            for polygon_id, polygon in self._polygons.items()
        }

        return line_cache, polygon_cache

    def _polygon_mask(
        self,
        polygon_id: str,
        polygon: Polygon,
        bboxes: Optional[np.ndarray],
        points: np.ndarray,
    ) -> np.ndarray:
        if self.coordinate_space == "world":
            return points_in_polygon(
                points,
                np.asarray(polygon.points, dtype=np.float64),
            ).reshape(-1, 1)

        mask, x_min, y_min = self._polygon_masks[polygon_id]

        if self._polygon_mode == "center":
            return bbox_center_in_polygon(
                mask,
                x_min,
                y_min,
                bboxes=bboxes,
                points=points,
            )

        if self._polygon_mode == "corners":
            if bboxes is None:
                raise ValueError("bboxes must be provided for polygon mode 'corners'")
            return bbox_corners_in_polygon(
                mask,
                x_min,
                y_min,
                bboxes=bboxes,
            )

        if self._polygon_mode == "any":
            if bboxes is None:
                raise ValueError("bboxes must be provided for polygon mode 'any'")
            return bbox_any_in_polygon(
                mask,
                x_min,
                y_min,
                bboxes=bboxes,
            )

        raise ValueError(f"Unknown polygon mode: {self._polygon_mode}")
