from dataclasses import dataclass
from pydantic import BaseModel, field_validator, model_validator, Field, PrivateAttr
from typing import List, Optional, Tuple, Dict
from math import gcd
import numpy as np

from geometry.spatial import (
    polygon_to_mask,
    bbox_center_in_polygon,
    bbox_corners_in_polygon,
    bbox_any_in_polygon,
    line_coefficients,
    points_in_polygon,
)

# --- Basic ---
Point = Tuple[int, int]


class Line(BaseModel):
    """
    Line

    Purpose:
        Represent a 2D line used for geometric filtering and crossing detection.

    Configuration Attributes (from YAML):
        - line_id: unique identifier
        - points: ((x1, y1), (x2, y2)) defining the line
        - vicinity: optional normalized threshold (relative to frame size)

    Cached Attributes (computed once):
        - _A, _B, _C: canonical line equation coefficients (Ax + By + C = 0)
        - _idx: index assigned by GeometryEngine (for vectorized access)

    Notes:
        - Canonical form ensures uniqueness (no duplicated representations)
        - All heavy computations are done once (constructor phase)
    """

    # =================================================
    # CONFIGURATION ATTRIBUTES
    # =================================================
    line_id: str
    points: Tuple[Point, Point]
    vicinity: Optional[float] = None
    vicinity_world_m: Optional[float] = None

    # =================================================
    # CACHED ATTRIBUTES (PRIVATE)
    # =================================================
    _A: int = PrivateAttr(default=None)
    _B: int = PrivateAttr(default=None)
    _C: int = PrivateAttr(default=None)

    _idx: int = PrivateAttr(default=None)

    # =================================================
    # VALIDATION
    # =================================================
    @field_validator("points")
    def valid_line(cls, pts):
        if pts[0] == pts[1]:
            raise ValueError("A line requires two distinct points")
        return pts

    @field_validator("vicinity")
    def non_negative_vicinity(cls, v):
        if v is not None and v < 0:
            raise ValueError("vicinity must be >= 0")
        return v

    @field_validator("vicinity_world_m")
    def non_negative_world_vicinity(cls, v):
        if v is not None and v < 0:
            raise ValueError("vicinity_world_m must be >= 0")
        return v

    # =================================================
    # PRECOMPUTE (CRITICAL)
    # =================================================
    @model_validator(mode="after")
    def compute_abc(self):
        """
        Compute canonical line equation Ax + By + C = 0
        with normalization for uniqueness.
        """
        (x1, y1), (x2, y2) = self.points

        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2

        # normalize using gcd
        g = gcd(gcd(abs(A), abs(B)), abs(C)) or 1
        A, B, C = A // g, B // g, C // g

        # enforce canonical sign
        if A < 0 or (A == 0 and B < 0):
            A, B, C = -A, -B, -C

        self._A, self._B, self._C = A, B, C
        return self

    # =================================================
    # PUBLIC METHODS
    # =================================================
    def canonical(self) -> Tuple[int, int, int]:
        """
        Returns:
            (A, B, C) canonical line coefficients
        """
        return self._A, self._B, self._C
        


        
class Polygon(BaseModel):
    """
    Pure geometric polygon primitive.
    No runtime processing policy inside.
    """

    # =================================================
    # CONFIGURATION
    # =================================================
    polygon_id: str
    points: List[Point] = Field(min_length=3)
    distance_meters: Optional[float] = None

    # =================================================
    # CACHED ATTRIBUTES
    # =================================================
    _mask: np.ndarray = PrivateAttr()
    _x_min: int = PrivateAttr()
    _y_min: int = PrivateAttr()
    _area: float = PrivateAttr()

    # =================================================
    # VALIDATION
    # =================================================
    @field_validator("points")
    def valid_polygon(cls, pts):

        if len(set(pts)) < 3:
            raise ValueError(
                "Polygon must have at least 3 distinct points"
            )

        area = 0

        for i in range(len(pts)):

            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % len(pts)]

            area += x1 * y2 - x2 * y1

        if area == 0:
            raise ValueError(
                "Points do not form a valid polygon"
            )

        return pts

    @field_validator("distance_meters")
    def positive_distance(cls, v):

        if v is not None and v <= 0:
            raise ValueError(
                "distance_meters must be > 0"
            )

        return v

    # =================================================
    # PRECOMPUTE (ONE TIME)
    # =================================================
    @model_validator(mode="after")
    def compute_cache(self):

        poly_np = np.array(
            self.points,
            dtype=np.int32
        )

        # mask + bbox
        self._mask, self._x_min, self._y_min = (
            polygon_to_mask(poly_np)
        )

        # area (shoelace)
        x = poly_np[:, 0]
        y = poly_np[:, 1]

        self._area = 0.5 * abs(
            np.dot(x, np.roll(y, -1))
            - np.dot(y, np.roll(x, -1))
        )

        return self

    # =================================================
    # UTILITIES
    # =================================================
    def area(self) -> float:
        return self._area

    def canonical_polygon(self):

        pts = self.points
        n = len(pts)

        rotations = [
            tuple(pts[i:] + pts[:i])
            for i in range(n)
        ]

        rev = pts[::-1]

        rev_rotations = [
            tuple(rev[i:] + rev[:i])
            for i in range(n)
        ]

        return min(rotations + rev_rotations)
        





@dataclass(frozen=True)
class SpatialLine:
    """Runtime line primitive for transformed coordinate spaces."""

    line_id: str
    points: np.ndarray
    vicinity: Optional[float] = None
    vicinity_world_m: Optional[float] = None

    def __post_init__(self):
        points = np.asarray(self.points, dtype=np.float64)
        if points.shape != (2, 2):
            raise ValueError("line points must have shape (2, 2)")
        object.__setattr__(self, "points", points)


@dataclass(frozen=True)
class SpatialPolygon:
    """Runtime polygon primitive for transformed coordinate spaces."""

    polygon_id: str
    points: np.ndarray
    distance_meters: Optional[float] = None

    def __post_init__(self):
        points = np.asarray(self.points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 3:
            raise ValueError("polygon points must have shape (N, 2) with N >= 3")
        object.__setattr__(self, "points", points)


class GeometryEngine:
    """
    Fully vectorized geometry engine for image or world coordinates.
    """

    # =================================================
    # CONSTRUCTOR
    # =================================================
    def __init__(
        self,
        lines: dict,
        polygons: dict,
        frame_size: int = 1000,
        polygon_mode: str = "center",
        coordinate_space: str = "image",
    ):

        # -------------------------
        # CONFIG
        # -------------------------
        self.coordinate_space = self._normalize_coordinate_space(coordinate_space)
        self._line_ids = list(lines.keys())

        self._line_id_to_idx = {
            lid: i
            for i, lid in enumerate(self._line_ids)
        }

        self._lines = lines

        self._polygons = polygons

        self._polygon_mode = polygon_mode

        # -------------------------
        # LINE CACHE
        # -------------------------
        self._A = None
        self._B = None
        self._C = None
        self._norm = None
        self._thresh = None

        # -------------------------
        # BUILD CACHE
        # -------------------------
        self._build_line_cache(
            lines,
            frame_size
        )

    @staticmethod
    def _normalize_coordinate_space(value: str) -> str:
        normalized = str(value).lower()
        if normalized not in {"image", "world"}:
            raise ValueError("coordinate_space must be one of: image, world")
        return normalized

    @staticmethod
    def _line_points(line) -> np.ndarray:
        return np.asarray(line.points, dtype=np.float64)

    def _line_threshold(self, line, frame_size: int) -> float:
        if self.coordinate_space == "world":
            return float(0.0 if line.vicinity_world_m is None else line.vicinity_world_m)

        vicinity = 0.0 if line.vicinity is None else line.vicinity
        return float(vicinity * frame_size)

    def _line_abc_norm(self, line) -> Tuple[float, float, float, float]:
        if self.coordinate_space == "image" and hasattr(line, "canonical"):
            a, b, c = line.canonical()
            norm = np.sqrt(a * a + b * b)
            if norm <= 1e-12:
                raise ValueError("line requires two distinct points")
            return float(a), float(b), float(c), float(norm)

        coeff = line_coefficients(self._line_points(line))
        return float(coeff[0]), float(coeff[1]), float(coeff[2]), float(coeff[3])

    # =================================================
    # BUILD LINE CACHE
    # =================================================
    def _build_line_cache(
        self,
        lines: dict,
        frame_size: int,
    ):
    
        L = len(self._line_ids)
    
    
        # ---------------------------------
        # Build cache
        # ---------------------------------
    
        dtype = np.float64 if self.coordinate_space == "world" else np.float32

        if L > 0:
    
            ABC = np.array([
                self._line_abc_norm(lines[lid])
                for lid in self._line_ids
            ], dtype=dtype)
    
            self._A = ABC[:, 0]
            self._B = ABC[:, 1]
            self._C = ABC[:, 2]
            self._norm = ABC[:, 3]
            self._thresh = np.array([
                self._line_threshold(lines[lid], frame_size)
                for lid in self._line_ids
            ], dtype=dtype)
    
        # ---------------------------------
        # Empty cache
        # ---------------------------------
    
        else:
    
            self._A = np.zeros((0,), dtype=dtype)
            self._B = np.zeros((0,), dtype=dtype)
            self._C = np.zeros((0,), dtype=dtype)
    
            self._norm = np.ones((0,), dtype=dtype)
            self._thresh = np.zeros((0,), dtype=dtype)
            
    # =================================================
    # POLYGON MASK
    # =================================================
    def _polygon_mask(
        self,
        poly: Polygon,
        bboxes: np.ndarray,
        points: np.ndarray,
    ):
        if self.coordinate_space == "world":
            return points_in_polygon(
                points,
                np.asarray(poly.points, dtype=np.float64),
            ).reshape(-1, 1)

        if self._polygon_mode == "center":

            return bbox_center_in_polygon(
                poly._mask,
                poly._x_min,
                poly._y_min,
                bboxes=bboxes,
                points=points,
            )

        elif self._polygon_mode == "corners":

            return bbox_corners_in_polygon(
                poly._mask,
                poly._x_min,
                poly._y_min,
                bboxes=bboxes,
            )

        elif self._polygon_mode == "any":

            return bbox_any_in_polygon(
                poly._mask,
                poly._x_min,
                poly._y_min,
                bboxes=bboxes,
            )

        else:
            raise ValueError(
                f"Unknown polygon mode: {self._polygon_mode}"
            )

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

    # =================================================
    # MAIN COMPUTE
    # =================================================
    def compute(
        self,
        points: np.ndarray,
        bboxes: Optional[np.ndarray] = None,
    ):
    
        if points is None:
            raise ValueError("points must be provided")
    
        x = points[:, 0]
        y = points[:, 1]
    
        N = points.shape[0]
        L = len(self._line_ids)
        distance_dtype = np.float64 if self.coordinate_space == "world" else np.float32
    
        # ---------------------------------
        # LINE FEATURES
        # ---------------------------------
    
        if L > 0:
    
            d = (
                self._A[:, None] * x[None, :]
                + self._B[:, None] * y[None, :]
                + self._C[:, None]
            ) / self._norm[:, None]
    
            sign = np.sign(d).astype(np.int8)
    
            abs_d = np.abs(d)
    
            vicinity_mask = (
                abs_d < self._thresh[:, None]
            )
            
            line_cache = {
                "distance": abs_d,
                "sign": sign,
                "vicinity_mask": vicinity_mask,
            }
    
        else:
    
            line_cache = {
                "distance": np.zeros((0, N), dtype=distance_dtype),
                "sign": np.zeros((0, N), dtype=np.int8),
                "vicinity_mask": np.zeros((0, N), dtype=bool),
            }
    
        # ---------------------------------
        # POLYGON FEATURES
        # ---------------------------------
    
        polygon_cache = {}
    
        for pid, poly in self._polygons.items():
    
            polygon_cache[pid] = self._polygon_mask(
                poly,
                bboxes,
                points,
            ).reshape(-1)
    
        return line_cache, polygon_cache
    


# --- Area ---
class Area(BaseModel):

    area_id: str
    name: str
    area_type: str = "lane"

    enable: bool = True
    description: str = ""

    flow_line: Optional[Line] = None
    zone: Optional[Polygon] = None

    eligible_metrics: List[str] = []
    ineligible_metrics: Dict[str, str] = Field(
        default_factory=dict
    )


    @model_validator(mode="after")
    def at_least_one_defined(self):

        if not (
            self.flow_line
            or self.zone
        ):
            raise ValueError(
                "At least one of flow_line or zone must be provided"
            )

        return self

    @field_validator("area_type")
    def valid_area_type(cls, value):
        supported = {"lane", "direction", "mixed", "entire"}
        if value not in supported:
            raise ValueError(
                f"area_type must be one of: {sorted(supported)}"
            )
        return value
