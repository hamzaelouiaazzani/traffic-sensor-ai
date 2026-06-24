from pydantic import BaseModel, field_validator, model_validator, Field, PrivateAttr
from typing import List, Optional, Tuple, Set, Dict, Any, Callable
from math import gcd
import numpy as np

from utils.helper_functions import (
    polygon_to_mask,
    bbox_center_in_polygon,
    bbox_corners_in_polygon,
    bbox_any_in_polygon,
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
        


        
# --- Speed Line Pair ---
class SpeedLinePair(BaseModel):
    
    line_pair_id: str
    line1: Line
    line2: Line
    distance_meters: float

    @model_validator(mode="after")
    def normalize_and_validate(self):
    
        # distinct check
        if self.line1.canonical() == self.line2.canonical():
            raise ValueError("line1 and line2 must be different")
    
        return self

    @field_validator("distance_meters")
    def non_zero_distance(cls, v):
        if v <= 0:
            raise ValueError("distance_meters must be > 0")
        return v    
    
    def canonical_pair(self):
        l1 = self.line1.canonical()
        l2 = self.line2.canonical()
        return tuple(sorted([l1, l2]))
        

    def between_mask_from_cache(self, cache):
        # get indices of both lines
        idx1 = self.line1.get_idx()
        idx2 = self.line2.get_idx()
    
        # get signed distances (N,)
        d1 = cache["distance"][idx1]
        d2 = cache["distance"][idx2]
    
        # between mask: opposite signs or touching
        mask = ((d1 * d2) <= 0).astype(int)
    
        return d1, d2, mask

    




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
        





class GeometryEngine:
    """
    Fully vectorized geometry engine.
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
    ):

        # -------------------------
        # CONFIG
        # -------------------------
        self._line_ids = list(lines.keys())

        self._line_id_to_idx = {
            lid: i
            for i, lid in enumerate(self._line_ids)
        }

        self._polygons = polygons

        self._polygon_mode = polygon_mode

        # -------------------------
        # LINE CACHE
        # -------------------------
        self._A = None
        self._B = None
        self._C = None
        self._norm = None
        self._vicinity = None
        self._thresh = None

        # -------------------------
        # BUILD CACHE
        # -------------------------
        # print(f"I am Here before self._build_line_cache")
        self._build_line_cache(
            lines,
            frame_size
        )
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
    
        if L > 0:
    
            ABC = np.array([
                lines[lid].canonical()
                for lid in self._line_ids
            ], dtype=np.float32)
    
            self._A = ABC[:, 0]
            self._B = ABC[:, 1]
            self._C = ABC[:, 2]
    
            self._norm = np.sqrt(
                self._A**2 + self._B**2
            )
    
            self._vicinity = np.array([
                (
                    0.0
                    if lines[lid].vicinity is None
                    else lines[lid].vicinity
                )
                for lid in self._line_ids
            ], dtype=np.float32)
    
            self._thresh = (
                self._vicinity * frame_size
            )
    
        # ---------------------------------
        # Empty cache
        # ---------------------------------
    
        else:
    
            self._A = np.zeros((0,), dtype=np.float32)
            self._B = np.zeros((0,), dtype=np.float32)
            self._C = np.zeros((0,), dtype=np.float32)
    
            self._norm = np.ones((0,), dtype=np.float32)
    
            self._vicinity = np.zeros((0,), dtype=np.float32)
    
            self._thresh = np.zeros((0,), dtype=np.float32)
            
    # =================================================
    # POLYGON MASK
    # =================================================
    def _polygon_mask(
        self,
        poly: Polygon,
        bboxes: np.ndarray,
        points: np.ndarray,
    ):

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

            # print(f"x.shape: {x.shape}")
            # print(f"y.shape: {y.shape}")
            # print(f"self._A.shape: {self._A.shape}")
            # print(f"self._B.shape: {self._B.shape}")
            # print(f"self._C.shape: {self._C.shape}")
            # print(f"d.shape: {d.shape}")
            # print(f"self._norm.shape: {self._norm.shape}")
            # print(f"self._thresh.shape: {self._thresh.shape}")
            # print(f"sign.shape: {sign.shape}")
            # print(f"abs_d.shape: {abs_d.shape}")
            # print(f"vicinity_mask.shape: {vicinity_mask.shape}")


            
            line_cache = {
                "distance": abs_d,
                "sign": sign,
                "vicinity_mask": vicinity_mask,
            }
    
        else:
    
            line_cache = {
                "distance": np.zeros((0, N), dtype=np.float32),
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

    enable: bool = True
    description: str = ""

    flow_line: Optional[Line] = None
    zone: Optional[Polygon] = None

    metrics_names: List[str] = []
    eligible_metrics: List[str] = []
    metrics: Dict[str, Any] = Field(
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
