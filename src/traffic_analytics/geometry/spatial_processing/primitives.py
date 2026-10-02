from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Sequence

import numpy as np


@dataclass(frozen=True, init=False)
class Point:
    """Immutable 2D point primitive represented internally as float[2]."""

    model_fields: ClassVar[dict] = {"coordinates": None}
    coordinates: np.ndarray

    def __init__(self, coordinates: Sequence[float] | np.ndarray):
        arr = _readonly_array(coordinates, expected_shape=(2,), name="Point")
        object.__setattr__(self, "coordinates", arr)

    def as_array(self) -> np.ndarray:
        """Return the point coordinates as a read-only NumPy array."""
        return self.coordinates


@dataclass(frozen=True, init=False)
class Line:
    """
    Ordered 2D line segment primitive.

    `points` has shape (2, 2). Endpoint order is preserved and defines the
    positive side used by signed-distance calculations: endpoint 0 -> endpoint 1.
    `line_id` is optional external metadata and is not used by pure geometry.
    """

    model_fields: ClassVar[dict] = {"line_id": None, "points": None}
    line_id: str
    points: np.ndarray

    def __init__(
        self,
        points: Sequence[Sequence[float]] | np.ndarray,
        line_id: str = "",
    ):
        arr = _readonly_array(points, expected_shape=(2, 2), name="Line")
        if np.allclose(arr[0], arr[1]):
            raise ValueError("Line requires two distinct endpoints")

        object.__setattr__(self, "points", arr)
        object.__setattr__(self, "line_id", str(line_id))

    @property
    def start(self) -> np.ndarray:
        return self.points[0]

    @property
    def end(self) -> np.ndarray:
        return self.points[1]


@dataclass(frozen=True, init=False)
class Polygon:
    """
    2D polygon primitive represented by a variable-length vertex array.

    `points` has shape (V, 2), V >= 3. `polygon_id` is optional external
    metadata and is not used by pure geometry.
    """

    model_fields: ClassVar[dict] = {"polygon_id": None, "points": None}
    polygon_id: str
    points: np.ndarray

    def __init__(
        self,
        points: Sequence[Sequence[float]] | np.ndarray,
        polygon_id: str = "",
    ):
        arr = _readonly_vertices(points, name="Polygon")
        if len({(float(x), float(y)) for x, y in arr}) < 3:
            raise ValueError("Polygon must have at least 3 distinct vertices")
        if np.isclose(_shoelace_area(arr), 0.0):
            raise ValueError("Polygon vertices do not form a valid polygon")

        object.__setattr__(self, "points", arr)
        object.__setattr__(self, "polygon_id", str(polygon_id))

    def area(self) -> float:
        """Return the absolute polygon area in the input coordinate units."""
        return float(abs(_shoelace_area(self.points)) * 0.5)

    def canonical_polygon(self):
        """Return a deterministic tuple representation independent of rotation."""
        pts = [tuple(map(float, point)) for point in self.points]
        n = len(pts)
        rotations = [
            tuple(pts[i:] + pts[:i])
            for i in range(n)
        ]
        reversed_pts = pts[::-1]
        reversed_rotations = [
            tuple(reversed_pts[i:] + reversed_pts[:i])
            for i in range(n)
        ]
        return min(rotations + reversed_rotations)


def _readonly_array(
    values: Sequence[float] | Sequence[Sequence[float]] | np.ndarray,
    *,
    expected_shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    arr = np.array(values, dtype=np.float64, copy=True)
    if arr.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}; got {arr.shape}")
    arr.setflags(write=False)
    return arr


def _readonly_vertices(
    values: Sequence[Sequence[float]] | np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    arr = np.array(values, dtype=np.float64, copy=True)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name} vertices must have shape (V, 2); got {arr.shape}")
    if arr.shape[0] < 3:
        raise ValueError(f"{name} requires at least 3 vertices")
    arr.setflags(write=False)
    return arr


def _shoelace_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
