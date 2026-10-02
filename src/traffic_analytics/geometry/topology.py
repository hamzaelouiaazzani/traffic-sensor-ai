from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from traffic_analytics.geometry.spatial_processing.primitives import Line, Polygon


@dataclass
class Area:
    """Semantic monitoring area composed from geometric primitives."""

    area_id: str
    name: str
    area_type: str = "lane"
    enable: bool = True
    description: str = ""
    flow_line: Optional[Line] = None
    zone: Optional[Polygon] = None
    distance_meters: Optional[float] = None
    eligible_metrics: List[str] = field(default_factory=list)
    ineligible_metrics: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not (self.flow_line or self.zone):
            raise ValueError("At least one of flow_line or zone must be provided")

        supported = {"lane", "direction", "mixed", "entire"}
        if self.area_type not in supported:
            raise ValueError(f"area_type must be one of: {sorted(supported)}")

        if self.distance_meters is not None and self.distance_meters <= 0:
            raise ValueError("distance_meters must be > 0")
