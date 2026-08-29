from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator


Coordinate = Tuple[float, float]


class Line(BaseModel):
    """Coordinate-agnostic geometric line primitive."""

    line_id: str
    points: Tuple[Coordinate, Coordinate]

    @field_validator("points")
    def valid_line(cls, pts):
        if pts[0] == pts[1]:
            raise ValueError("A line requires two distinct points")
        return pts


class Polygon(BaseModel):
    """Coordinate-agnostic geometric polygon primitive."""

    polygon_id: str
    points: List[Coordinate] = Field(min_length=3)

    @field_validator("points")
    def valid_polygon(cls, pts):
        if len(set(pts)) < 3:
            raise ValueError("Polygon must have at least 3 distinct points")

        if _shoelace_area(pts) == 0:
            raise ValueError("Points do not form a valid polygon")

        return pts

    def area(self) -> float:
        return float(abs(_shoelace_area(self.points)) * 0.5)

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


class Area(BaseModel):
    """Logical monitoring area composed from geometric primitives."""

    area_id: str
    name: str
    area_type: str = "lane"

    enable: bool = True
    description: str = ""

    flow_line: Optional[Line] = None
    zone: Optional[Polygon] = None
    distance_meters: Optional[float] = None

    eligible_metrics: List[str] = Field(default_factory=list)
    ineligible_metrics: Dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def at_least_one_defined(self):
        if not (self.flow_line or self.zone):
            raise ValueError("At least one of flow_line or zone must be provided")

        return self

    @field_validator("area_type")
    def valid_area_type(cls, value):
        supported = {"lane", "direction", "mixed", "entire"}
        if value not in supported:
            raise ValueError(f"area_type must be one of: {sorted(supported)}")
        return value

    @field_validator("distance_meters")
    def positive_distance(cls, value):
        if value is not None and value <= 0:
            raise ValueError("distance_meters must be > 0")
        return value


def _shoelace_area(points: List[Coordinate]) -> float:
    poly = np.asarray(points, dtype=np.float64)
    x = poly[:, 0]
    y = poly[:, 1]
    return float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
