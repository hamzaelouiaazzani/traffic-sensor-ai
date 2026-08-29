from __future__ import annotations

from typing import Optional, Tuple
import math

import numpy as np
from PIL import Image, ImageDraw


def image_line_coefficients(points: np.ndarray) -> np.ndarray:
    """Return canonical image-space line coefficients plus the line norm."""

    arr = np.asarray(points, dtype=np.float64)
    if arr.shape != (2, 2):
        raise ValueError("line points must have shape (2, 2)")

    (x1, y1), (x2, y2) = arr
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2

    if not np.allclose([a, b, c], np.round([a, b, c])):
        return line_coefficients(arr)

    a_i, b_i, c_i = int(round(a)), int(round(b)), int(round(c))
    gcd_value = math.gcd(math.gcd(abs(a_i), abs(b_i)), abs(c_i)) or 1
    a_i, b_i, c_i = a_i // gcd_value, b_i // gcd_value, c_i // gcd_value

    if a_i < 0 or (a_i == 0 and b_i < 0):
        a_i, b_i, c_i = -a_i, -b_i, -c_i

    norm = np.sqrt(a_i * a_i + b_i * b_i)
    if norm <= 1e-12:
        raise ValueError("line requires two distinct points")
    return np.asarray([a_i, b_i, c_i, norm], dtype=np.float64)


def line_coefficients(points: np.ndarray) -> np.ndarray:
    """Return Ax + By + C = 0 coefficients plus the line norm."""

    arr = np.asarray(points, dtype=np.float64)
    if arr.shape != (2, 2):
        raise ValueError("line points must have shape (2, 2)")

    (x1, y1), (x2, y2) = arr
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    norm = np.sqrt(a * a + b * b)
    if norm <= 1e-12:
        raise ValueError("line requires two distinct points")
    return np.asarray([a, b, c, norm], dtype=np.float64)


def polygon_to_mask(polygon: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """
    Rasterize a polygon into a tight image-space mask.

    Returns the mask plus its top-left image origin.
    """

    if polygon.ndim != 2 or polygon.shape[1] != 2:
        raise ValueError("polygon must be shape (N,2)")

    xs = polygon[:, 0]
    ys = polygon[:, 1]

    x_min = int(math.floor(xs.min()))
    y_min = int(math.floor(ys.min()))
    x_max = int(math.ceil(xs.max()))
    y_max = int(math.ceil(ys.max()))

    width = x_max - x_min + 1
    height = y_max - y_min + 1
    if width <= 0 or height <= 0:
        raise ValueError("invalid polygon bbox")

    shifted = [(float(x - x_min), float(y - y_min)) for x, y in polygon]
    img = Image.new("L", (width, height), 0)
    ImageDraw.Draw(img).polygon(shifted, outline=1, fill=1)

    return np.asarray(img, dtype=np.uint8), x_min, y_min


def bbox_center_in_polygon(
    mask: np.ndarray,
    x_min: int,
    y_min: int,
    bboxes: Optional[np.ndarray] = None,
    points: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return an (N, 1) mask for image-space point membership."""

    if points is not None:
        centers_x = points[:, 0]
        centers_y = points[:, 1]
    elif bboxes is not None:
        centers_x = (bboxes[:, 0] + bboxes[:, 2]) * 0.5
        centers_y = (bboxes[:, 1] + bboxes[:, 3]) * 0.5
    else:
        raise ValueError("points or bboxes must be provided")

    x_idx = np.floor(centers_x - x_min).astype(int)
    y_idx = np.floor(centers_y - y_min).astype(int)

    x_idx_clipped = np.clip(x_idx, 0, mask.shape[1] - 1)
    y_idx_clipped = np.clip(y_idx, 0, mask.shape[0] - 1)

    in_bounds = (
        (x_idx >= 0) & (x_idx < mask.shape[1])
        & (y_idx >= 0) & (y_idx < mask.shape[0])
    )

    vals = mask[y_idx_clipped, x_idx_clipped].astype(bool)
    vals[~in_bounds] = False

    return vals.reshape(-1, 1)


def bbox_corners_in_polygon(
    mask: np.ndarray,
    x_min: int,
    y_min: int,
    bboxes: np.ndarray,
    points: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return an (N, 1) mask for bboxes with any corner inside an image-space polygon."""

    corners = np.stack([
        bboxes[:, [0, 1]],
        bboxes[:, [2, 1]],
        bboxes[:, [0, 3]],
        bboxes[:, [2, 3]],
    ], axis=1)

    x_idx = np.floor(corners[:, :, 0] - x_min).astype(int)
    y_idx = np.floor(corners[:, :, 1] - y_min).astype(int)

    x_idx_clipped = np.clip(x_idx, 0, mask.shape[1] - 1)
    y_idx_clipped = np.clip(y_idx, 0, mask.shape[0] - 1)

    in_bounds = (
        (x_idx >= 0) & (x_idx < mask.shape[1])
        & (y_idx >= 0) & (y_idx < mask.shape[0])
    )

    vals = mask[y_idx_clipped, x_idx_clipped].astype(bool)
    vals[~in_bounds] = False

    return np.any(vals, axis=1).reshape(-1, 1)


def bbox_any_in_polygon(
    mask: np.ndarray,
    x_min: int,
    y_min: int,
    bboxes: np.ndarray,
    points: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return an (N, 1) mask for bboxes with any covered pixel inside an image-space polygon."""

    n = bboxes.shape[0]
    h, w = mask.shape

    x_starts = np.floor(bboxes[:, 0] - x_min).astype(int)
    x_ends = np.ceil(bboxes[:, 2] - x_min).astype(int)
    y_starts = np.floor(bboxes[:, 1] - y_min).astype(int)
    y_ends = np.ceil(bboxes[:, 3] - y_min).astype(int)

    x_starts_clipped = np.clip(x_starts, 0, w - 1)
    x_ends_clipped = np.clip(x_ends, -1, w - 1)
    y_starts_clipped = np.clip(y_starts, 0, h - 1)
    y_ends_clipped = np.clip(y_ends, -1, h - 1)

    x_lengths = np.maximum(0, x_ends_clipped - x_starts_clipped + 1)
    y_lengths = np.maximum(0, y_ends_clipped - y_starts_clipped + 1)

    if x_lengths.sum() == 0 or y_lengths.sum() == 0:
        return np.zeros((n, 1), dtype=bool)

    max_x = int(x_lengths.max())
    max_y = int(y_lengths.max())

    x_offsets = np.arange(max_x)
    y_offsets = np.arange(max_y)

    x_grid = x_starts_clipped[:, None] + x_offsets[None, :]
    y_grid = y_starts_clipped[:, None] + y_offsets[None, :]

    x_grid = np.clip(x_grid, 0, w - 1)
    y_grid = np.clip(y_grid, 0, h - 1)

    x_coords = np.broadcast_to(x_grid[:, None, :], (n, max_y, max_x))
    y_coords = np.broadcast_to(y_grid[:, :, None], (n, max_y, max_x))

    x_valid = x_offsets[None, :] < x_lengths[:, None]
    y_valid = y_offsets[None, :] < y_lengths[:, None]
    valid = np.logical_and(y_valid[:, :, None], x_valid[:, None, :])

    mask_vals = mask[y_coords, x_coords].astype(bool)
    mask_vals[~valid] = False

    return np.any(mask_vals, axis=(1, 2)).reshape(-1, 1)


def points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Vectorized point-in-polygon test for arbitrary coordinate systems."""

    n_obs = points.shape[0]
    if n_obs == 0:
        return np.zeros((0,), dtype=bool)

    polygon = np.asarray(polygon, dtype=np.float64)
    x = points[:, 0]
    y = points[:, 1]

    x_min = polygon[:, 0].min()
    x_max = polygon[:, 0].max()
    y_min = polygon[:, 1].min()
    y_max = polygon[:, 1].max()
    bbox_mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
    inside = np.zeros((n_obs,), dtype=bool)

    candidate_idx = np.flatnonzero(bbox_mask)
    if candidate_idx.size == 0:
        return inside

    cx = x[candidate_idx]
    cy = y[candidate_idx]
    candidate_inside = np.zeros((candidate_idx.size,), dtype=bool)

    xj, yj = polygon[-1]
    for xi, yi in polygon:
        intersects = ((yi > cy) != (yj > cy)) & (
            cx < ((xj - xi) * (cy - yi) / ((yj - yi) + 1e-12) + xi)
        )
        candidate_inside ^= intersects
        xj, yj = xi, yi

    inside[candidate_idx] = candidate_inside
    return inside
