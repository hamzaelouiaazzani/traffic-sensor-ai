from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class DensityResult:
    average_density: float = 0.0
    average_density_by_class: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float32)
    )
    current_density: float = 0.0
    current_density_by_class: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float32)
    )


@dataclass
class FlowResult:
    average_flow: float = 0.0
    average_flow_by_class: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float32)
    )


@dataclass
class SpaceHeadwayResult:
    average_space_headway: float | None = None
    current_space_headway: float | None = None


@dataclass
class TimeHeadwayResult:
    average_time_headway: float | None = None
    current_time_headway: float | None = None


@dataclass
class OccupancyResult:
    average_occupancy: float = 0.0
    current_occupancy: float = 0.0


@dataclass
class PeriodSpatialResult:
    coordinate_space: str
    points: np.ndarray
    bboxes: np.ndarray
    line_cache: Dict[str, np.ndarray]
    polygon_cache: Dict[str, np.ndarray]
    line_ids: List[str]
    line_id_to_idx: Dict[str, int]


@dataclass
class PeriodEvents:
    crossed_masks: np.ndarray
    crossings_by_area: Dict[str, Dict[str, np.ndarray]]
