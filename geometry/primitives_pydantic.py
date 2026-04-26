from pydantic import BaseModel, field_validator, model_validator, Field, PrivateAttr
from typing import List, Optional, Tuple, Set, Dict, Any
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

# --- Line ---
class Line(BaseModel):
    line_id: str
    points: tuple[Point, Point]
    vicinity: Optional[float] = None


    _A: int = PrivateAttr(default=None)
    _B: int = PrivateAttr(default=None)
    _C: int = PrivateAttr(default=None)

    _idx: int = PrivateAttr(default=None)

    # Pre-crossing state per ID
    _pre_crossing_state: Dict[str, Dict[str, Any]] = PrivateAttr(default_factory=dict)

    # Per-ID crossing results (t_c and optional metadata)
    _crossing_events: Dict[str, Dict[str, Any]] = PrivateAttr(default_factory=dict)
    
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
        
    @model_validator(mode="after")
    def compute_abc(self):
        (x1, y1), (x2, y2) = self.points
        A = y2 - y1
        B = x1 - x2
        C = x2*y1 - x1*y2
    
        g = gcd(gcd(abs(A), abs(B)), abs(C)) or 1
        A, B, C = A//g, B//g, C//g
    
        if A < 0 or (A == 0 and B < 0):
            A, B, C = -A, -B, -C
    
        self._A, self._B, self._C = A, B, C
        return self
    
    def canonical(self):
        return (self._A , self._B , self._C)

    def set_idx(self, idx: int) -> None:
        self._idx = idx

    def get_idx(self) -> int:
        return self._idx

    def get_crossed_mask(self, track_ids: np.ndarray) -> np.ndarray:
        return np.array([tid in self._crossing_events for tid in track_ids], dtype=bool)
    
    def get_pre_crossing_state(self) -> Dict[str, Dict[str, Any]]:
        return self._pre_crossing_state
    
    def update_pre_crossing_state(self, track_id: str, dist: float, sign: int, timestamp: float, frame: int) -> None:
        self._pre_crossing_state[track_id] = {
            "dist": dist,
            "sign": sign,
            "timestamp": timestamp,
            "frame": frame,
        }



        
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
    polygon_id: str
    points: List[Point] = Field(min_length=3)
    distance_meters: Optional[float] = None

    # --- Cached ---
    _mask: np.ndarray = PrivateAttr()
    _x_min: int = PrivateAttr()
    _y_min: int = PrivateAttr()
    _area: float = PrivateAttr()

    # --- Validation ---
    @field_validator("points")
    def valid_polygon(cls, pts):
        if len(set(pts)) < 3:
            raise ValueError("Polygon must have at least 3 distinct points")

        area = 0
        for i in range(len(pts)):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % len(pts)]
            area += x1 * y2 - x2 * y1

        if area == 0:
            raise ValueError("Points do not form a valid polygon")

        return pts

    @field_validator("distance_meters")
    def positive_distance(cls, v):
        if v is not None and v <= 0:
            raise ValueError("distance_meters must be > 0")
        return v

    # --- Precompute (CRITICAL for performance) ---
    @model_validator(mode="after")
    def compute_cache(self):
        poly_np = np.array(self.points, dtype=np.int32)

        # mask + bbox
        self._mask, self._x_min, self._y_min = polygon_to_mask(poly_np)

        # area (shoelace)
        x = poly_np[:, 0]
        y = poly_np[:, 1]
        self._area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

        return self

    # --- Core methods (used in metrics pipeline) ---
    def contains_bboxes(self, bboxes: np.ndarray, mode: str = "center") -> np.ndarray:
        if mode == "center":
            return bbox_center_in_polygon(bboxes, self._mask, self._x_min, self._y_min)
        elif mode == "corners":
            return bbox_corners_in_polygon(bboxes, self._mask, self._x_min, self._y_min)
        elif mode == "any":
            return bbox_any_in_polygon(bboxes, self._mask, self._x_min, self._y_min)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def mask_and_count(self, bboxes: np.ndarray, mode: str = "center"):
        mask = self.contains_bboxes(bboxes, mode)
        return mask, mask.sum()

    def area(self) -> float:
        return self._area

    def canonical_polygon(self):
        pts = self.points
        n = len(pts)
    
        # generate all rotations
        rotations = [tuple(pts[i:] + pts[:i]) for i in range(n)]
        
        # also reversed rotations
        rev = pts[::-1]
        rev_rotations = [tuple(rev[i:] + rev[:i]) for i in range(n)]
    
        # take lexicographically smallest
        return min(rotations + rev_rotations)




class GeometryEngine:
    def __init__(self, lines: dict, polygons: dict, frame_size=1000):

        # =========================
        # 1. LINES (vectorized)
        # =========================
        self.line_ids = list(lines.keys())
        for i, lid in enumerate(self.line_ids):
            lines[lid].set_idx(i)   
            
        if len(self.line_ids) > 0:
            ABC = np.array([lines[lid].canonical() for lid in self.line_ids])  # (L,3)
        
            self.A = ABC[:, 0][:, None]
            self.B = ABC[:, 1][:, None]
            self.C = ABC[:, 2][:, None]
            self.norm = np.sqrt(self.A**2 + self.B**2)
        
            self.vicinity = np.array([
                0.0 if lines[lid].vicinity is None else lines[lid].vicinity
                for lid in self.line_ids
            ])[:, None]
        
            self.thresh = self.vicinity * frame_size
        
        else:
            # 👉 EMPTY SAFE STRUCTURES
            self.A = np.zeros((0, 1))
            self.B = np.zeros((0, 1))
            self.C = np.zeros((0, 1))
            self.norm = np.ones((0, 1))  # avoid division issues
            self.vicinity = np.zeros((0, 1))
            self.thresh = np.zeros((0, 1))

        # =========================
        # 2. POLYGONS (cached objects)
        # =========================
        self.polygons = polygons  # {area_id: Polygon}

    # =========================
    # 3. INTERNAL: bbox → centers (ONLY for lines)
    # =========================
    def _centers_from_bboxes(self, bboxes: np.ndarray):
        return np.stack([
            (bboxes[:, 0] + bboxes[:, 2]) * 0.5,
            (bboxes[:, 1] + bboxes[:, 3]) * 0.5
        ], axis=1)

    # =========================
    # 4. MAIN COMPUTE (ONE PASS)
    # =========================
    def compute(self, bboxes: np.ndarray):

        # ---------------------------------
        # A. POINTS for line computations
        # ---------------------------------
        points = self._centers_from_bboxes(bboxes)  # (N,2)

        x = points[:, 0][None, :]
        y = points[:, 1][None, :]

        # ---------------------------------
        # B. LINE FEATURES (vectorized)
        # ---------------------------------
        if self.A.shape[0] > 0:
            d = (self.A * x + self.B * y + self.C) / self.norm  # (L,N)
        
            abs_d = np.abs(d)
            vicinity_mask = (abs_d < self.thresh).astype(int)
        
            line_cache = {
                "distance": d,
                "sign": np.sign(d),
                "vicinity_mask": vicinity_mask
            }
        else:
            # 👉 EMPTY OUTPUT (consistent API)
            N = bboxes.shape[0]
            line_cache = {
                "distance": np.zeros((0, N)),
                "sign": np.zeros((0, N)),
                "vicinity_mask": np.zeros((0, N), dtype=int)
            } 
            
        # ---------------------------------
        # C. POLYGON FEATURES (use bboxes directly)
        # ---------------------------------
        polygon_cache = {}

        for pid, poly in self.polygons.items():
            polygon_cache[pid] = poly.contains_bboxes(bboxes).reshape(-1)  # (N,1)

        # ---------------------------------
        # D. FINAL CACHE
        # ---------------------------------
        return {
            "line": line_cache,
            "polygon": polygon_cache
        }

        


    
# --- Area ---
class Area(BaseModel):
    name: str
    enable: bool
    description: str
    flow_line: Optional[Line] = None
    speed_pair: Optional[SpeedLinePair] = None
    zone: Optional[Polygon] = None
    
    @model_validator(mode="after")
    def at_least_one_defined(self):
        if not (self.flow_line or self.speed_pair or self.zone):
            raise ValueError("At least one of flow_line, speed_pair, or zone must be provided")
        return self


