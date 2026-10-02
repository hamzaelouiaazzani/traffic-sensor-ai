from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from traffic_analytics.geometry.spatial_processing.membership import (
    IPolygonMembership,
    RayCastingPolygonMembership,
)
from traffic_analytics.geometry.spatial_processing.primitives import Line, Polygon


@dataclass(frozen=True)
class _CompiledLines:
    line_ids: tuple[str, ...]
    p0: np.ndarray
    direction: np.ndarray
    inv_norm: np.ndarray
    id_to_index: dict[str, int]


class SpatialProcessor:
    """
    Pure geometric processor for compiled Stage-2 spatial primitives.

    Static lines and polygons are supplied once during construction and compiled
    into dense numerical execution structures. Each analytics period then
    supplies only dynamic ``points`` with shape ``(N, 2)``. Polygon membership
    and signed line distance are geometric relations only; temporal/event
    semantics, traffic areas, vicinity thresholds, coordinate-policy selection,
    and metric meaning remain outside this package.
    """

    def __init__(
        self,
        lines: Sequence[Line] = (),
        polygons: Sequence[Polygon] = (),
        polygon_membership: IPolygonMembership | None = None,
        dtype: np.dtype | type = np.float32,
    ):
        self._dtype = np.dtype(dtype)
        self._lines = _compile_lines(lines, self._dtype)
        self.polygon_membership = (
            polygon_membership
            if polygon_membership is not None
            else RayCastingPolygonMembership(polygons, dtype=self._dtype)
        )

    @property
    def line_ids(self) -> tuple[str, ...]:
        """Line IDs ordered exactly like signed-distance output columns."""
        return self._lines.line_ids

    @property
    def line_id_to_index(self) -> dict[str, int]:
        """Mapping from line ID to signed-distance output column index."""
        return dict(self._lines.id_to_index)

    @property
    def polygon_ids(self) -> tuple[str, ...]:
        """Polygon IDs ordered exactly like membership output columns."""
        return self.polygon_membership.polygon_ids

    @property
    def polygon_id_to_index(self) -> dict[str, int]:
        """Mapping from polygon ID to membership output column index."""
        return self.polygon_membership.polygon_id_to_index

    def line_normal(self, line_id: str) -> np.ndarray | None:
        """
        Return the unit normal of a compiled line, or None if unknown.

        This supports downstream traffic estimators that project vehicles along
        the direction orthogonal to a configured flow line.
        """
        idx = self._lines.id_to_index.get(line_id)
        if idx is None:
            return None
        direction = self._lines.direction[idx]
        inv_norm = self._lines.inv_norm[idx]
        normal = np.asarray(
            [-direction[1] * inv_norm, direction[0] * inv_norm],
            dtype=self._dtype,
        )
        normal.setflags(write=False)
        return normal

    def compute_polygon_membership(self, points: np.ndarray) -> np.ndarray:
        """
        Return bool[N, P] point-in-polygon membership.

        Column p corresponds exactly to ``self.polygon_ids[p]``.
        """
        return self.polygon_membership.compute(points)

    def compute_signed_distances(self, points: np.ndarray) -> np.ndarray:
        """
        Return float[N, L] signed point-to-line distances.

        Column l corresponds exactly to ``self.line_ids[l]``. The sign is
        defined by each ordered line endpoint pair p0 -> p1: positive values lie
        to the left of the directed line, negative values lie to the right, and
        zero lies on the line.
        """
        points_arr = _validate_points(points, self._dtype)
        if not self._lines.line_ids:
            return np.zeros((points_arr.shape[0], 0), dtype=self._dtype)

        rel = points_arr[:, None, :] - self._lines.p0[None, :, :]
        cross = (
            self._lines.direction[None, :, 0] * rel[:, :, 1]
            - self._lines.direction[None, :, 1] * rel[:, :, 0]
        )
        return cross * self._lines.inv_norm[None, :]


def _compile_lines(
    lines: Sequence[Line],
    dtype: np.dtype,
) -> _CompiledLines:
    line_list = tuple(lines)
    line_ids = tuple(line.line_id for line in line_list)
    id_to_index = {
        line_id: idx
        for idx, line_id in enumerate(line_ids)
    }

    if not line_list:
        empty_2d = np.empty((0, 2), dtype=dtype)
        empty_1d = np.empty((0,), dtype=dtype)
        for arr in (empty_2d, empty_1d):
            arr.setflags(write=False)
        return _CompiledLines(
            line_ids=line_ids,
            p0=empty_2d,
            direction=empty_2d,
            inv_norm=empty_1d,
            id_to_index=id_to_index,
        )

    line_points = np.stack([line.points for line in line_list], axis=0).astype(
        dtype,
        copy=False,
    )
    p0 = line_points[:, 0, :].copy()
    p1 = line_points[:, 1, :]
    direction = (p1 - p0).astype(dtype, copy=False)
    norm = np.linalg.norm(direction, axis=1)
    if np.any(norm <= 1e-12):
        raise ValueError("Line requires two distinct endpoints")

    inv_norm = (1.0 / norm).astype(dtype, copy=False)
    for arr in (p0, direction, inv_norm):
        arr.setflags(write=False)

    return _CompiledLines(
        line_ids=line_ids,
        p0=p0,
        direction=direction,
        inv_norm=inv_norm,
        id_to_index=id_to_index,
    )


def _validate_points(points: np.ndarray, dtype: np.dtype) -> np.ndarray:
    arr = np.asarray(points)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"points must have shape (N, 2); got {arr.shape}")
    if not np.issubdtype(arr.dtype, np.floating):
        return arr.astype(dtype, copy=False)
    return arr
