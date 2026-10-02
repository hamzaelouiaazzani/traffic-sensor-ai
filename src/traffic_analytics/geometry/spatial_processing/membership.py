from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw

from traffic_analytics.geometry.spatial_processing.primitives import Polygon


class IPolygonMembership(ABC):
    """Compiled geometric point-in-polygon membership backend."""

    @property
    @abstractmethod
    def polygon_ids(self) -> tuple[str, ...]:
        """Polygon IDs ordered exactly like membership output columns."""
        raise NotImplementedError

    @property
    @abstractmethod
    def polygon_id_to_index(self) -> dict[str, int]:
        """Mapping from polygon ID to membership output column index."""
        raise NotImplementedError

    @abstractmethod
    def compute(self, points: np.ndarray) -> np.ndarray:
        """Return bool[N, P] for the compiled polygon set."""
        raise NotImplementedError


@dataclass(frozen=True)
class _CompiledRayPolygon:
    polygon_id: str
    x0: np.ndarray
    y0: np.ndarray
    x1: np.ndarray
    y1: np.ndarray
    denominator: np.ndarray


class RayCastingPolygonMembership(IPolygonMembership):
    """
    Coordinate-space-agnostic membership using compiled ray-casting edges.

    This backend operates directly on continuous 2D coordinates and is suitable
    for image-space or planar world-space points when all inputs share the same
    coordinate representation. Boundary points are treated as members.
    """

    def __init__(
        self,
        polygons: Sequence[Polygon],
        dtype: np.dtype | type = np.float32,
    ):
        self._dtype = np.dtype(dtype)
        self._compiled = tuple(
            _compile_ray_polygon(polygon, self._dtype)
            for polygon in polygons
        )
        self._polygon_ids = tuple(polygon.polygon_id for polygon in polygons)
        self._polygon_id_to_index = {
            polygon_id: idx
            for idx, polygon_id in enumerate(self._polygon_ids)
        }

    @property
    def polygon_ids(self) -> tuple[str, ...]:
        return self._polygon_ids

    @property
    def polygon_id_to_index(self) -> dict[str, int]:
        return dict(self._polygon_id_to_index)

    def compute(self, points: np.ndarray) -> np.ndarray:
        points_arr = _validate_points(points, self._dtype)
        if not self._compiled:
            return np.zeros((points_arr.shape[0], 0), dtype=bool)

        columns = [
            _points_in_compiled_ray_polygon(points_arr, polygon)
            for polygon in self._compiled
        ]
        return np.column_stack(columns).astype(bool, copy=False)


@dataclass(frozen=True)
class _CompiledRasterPolygon:
    polygon_id: str
    mask: np.ndarray
    x_min: int
    y_min: int


class ImageRasterMaskPolygonMembership(IPolygonMembership):
    """
    Image-space membership using pre-rasterized pixel masks.

    This backend is defined only for image-coordinate geometry, where polygon
    vertices and observation points are expressed in pixel coordinates. It must
    not be used for planar world-space coordinates. Metric coordinates require
    continuous geometric processing and are handled by
    RayCastingPolygonMembership.

    Selection of this backend is the responsibility of Stage-2 runtime assembly
    according to the resolved coordinate policy. Floating-point image points are
    mapped to pixel indices by `pixel_index_policy`: "floor" means the
    containing pixel, and "round" means nearest integer pixel.
    """

    def __init__(
        self,
        polygons: Sequence[Polygon],
        pixel_index_policy: str = "floor",
        dtype: np.dtype | type = np.float32,
    ):
        if pixel_index_policy not in {"floor", "round"}:
            raise ValueError("pixel_index_policy must be one of: floor, round")

        self.pixel_index_policy = pixel_index_policy
        self._dtype = np.dtype(dtype)
        self._compiled = tuple(
            _compile_raster_polygon(polygon)
            for polygon in polygons
        )
        self._polygon_ids = tuple(polygon.polygon_id for polygon in polygons)
        self._polygon_id_to_index = {
            polygon_id: idx
            for idx, polygon_id in enumerate(self._polygon_ids)
        }

    @property
    def polygon_ids(self) -> tuple[str, ...]:
        return self._polygon_ids

    @property
    def polygon_id_to_index(self) -> dict[str, int]:
        return dict(self._polygon_id_to_index)

    def compute(self, points: np.ndarray) -> np.ndarray:
        points_arr = _validate_points(points, self._dtype)
        if not self._compiled:
            return np.zeros((points_arr.shape[0], 0), dtype=bool)

        columns = [
            self._membership_for_polygon(points_arr, polygon)
            for polygon in self._compiled
        ]
        return np.column_stack(columns).astype(bool, copy=False)

    def _membership_for_polygon(
        self,
        points: np.ndarray,
        polygon: _CompiledRasterPolygon,
    ) -> np.ndarray:
        x_idx = self._to_pixel_index(points[:, 0] - polygon.x_min)
        y_idx = self._to_pixel_index(points[:, 1] - polygon.y_min)

        in_bounds = (
            (x_idx >= 0)
            & (x_idx < polygon.mask.shape[1])
            & (y_idx >= 0)
            & (y_idx < polygon.mask.shape[0])
        )

        x_safe = np.clip(x_idx, 0, polygon.mask.shape[1] - 1)
        y_safe = np.clip(y_idx, 0, polygon.mask.shape[0] - 1)
        values = polygon.mask[y_safe, x_safe].astype(bool)
        values[~in_bounds] = False
        return values

    def _to_pixel_index(self, values: np.ndarray) -> np.ndarray:
        if self.pixel_index_policy == "round":
            return np.rint(values).astype(np.int64)
        return np.floor(values).astype(np.int64)


def _validate_points(points: np.ndarray, dtype: np.dtype) -> np.ndarray:
    arr = np.asarray(points)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"points must have shape (N, 2); got {arr.shape}")
    if not np.issubdtype(arr.dtype, np.floating):
        return arr.astype(dtype, copy=False)
    return arr


def _compile_ray_polygon(
    polygon: Polygon,
    dtype: np.dtype,
) -> _CompiledRayPolygon:
    vertices = np.asarray(polygon.points, dtype=dtype)
    x0 = vertices[:, 0].copy()
    y0 = vertices[:, 1].copy()
    x1 = np.roll(x0, -1)
    y1 = np.roll(y0, -1)
    horizontal = np.isclose(y1 - y0, 0.0)
    denominator = np.where(horizontal, 1.0, y1 - y0).astype(dtype, copy=False)

    for arr in (x0, y0, x1, y1, denominator):
        arr.setflags(write=False)

    return _CompiledRayPolygon(
        polygon_id=polygon.polygon_id,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        denominator=denominator,
    )


def _points_in_compiled_ray_polygon(
    points: np.ndarray,
    polygon: _CompiledRayPolygon,
) -> np.ndarray:
    x = points[:, 0][:, None]
    y = points[:, 1][:, None]

    boundary = _points_on_polygon_boundary(points, polygon)

    x_intersection = (
        (polygon.x1 - polygon.x0)
        * (y - polygon.y0)
        / polygon.denominator
    ) + polygon.x0
    crosses = ((polygon.y0 > y) != (polygon.y1 > y)) & (x < x_intersection)
    inside = np.count_nonzero(crosses, axis=1) % 2 == 1

    return inside | boundary


def _points_on_polygon_boundary(
    points: np.ndarray,
    polygon: _CompiledRayPolygon,
) -> np.ndarray:
    px = points[:, 0][:, None]
    py = points[:, 1][:, None]

    edge_x = polygon.x1 - polygon.x0
    edge_y = polygon.y1 - polygon.y0
    rel_x = px - polygon.x0
    rel_y = py - polygon.y0

    cross = edge_x * rel_y - edge_y * rel_x
    within_x = (
        (px >= np.minimum(polygon.x0, polygon.x1))
        & (px <= np.maximum(polygon.x0, polygon.x1))
    )
    within_y = (
        (py >= np.minimum(polygon.y0, polygon.y1))
        & (py <= np.maximum(polygon.y0, polygon.y1))
    )
    return np.any(np.isclose(cross, 0.0) & within_x & within_y, axis=1)


def _compile_raster_polygon(polygon: Polygon) -> _CompiledRasterPolygon:
    vertices = np.asarray(polygon.points)
    x_min = int(np.floor(vertices[:, 0].min()))
    y_min = int(np.floor(vertices[:, 1].min()))
    x_max = int(np.ceil(vertices[:, 0].max()))
    y_max = int(np.ceil(vertices[:, 1].max()))

    width = x_max - x_min + 1
    height = y_max - y_min + 1
    if width <= 0 or height <= 0:
        raise ValueError("invalid polygon raster bounds")

    shifted = [
        (float(x - x_min), float(y - y_min))
        for x, y in vertices
    ]
    image = Image.new("L", (width, height), 0)
    ImageDraw.Draw(image).polygon(shifted, outline=1, fill=1)
    mask = np.asarray(image, dtype=np.uint8)
    mask.setflags(write=False)

    return _CompiledRasterPolygon(
        polygon_id=polygon.polygon_id,
        mask=mask,
        x_min=x_min,
        y_min=y_min,
    )
