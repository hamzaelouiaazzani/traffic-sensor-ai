import logging
import numpy as np
from dataclasses import dataclass, field
from collections import deque

@dataclass
class CountResult:
    cumulative_count: int = 0
    cumulative_counts_by_class: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int64)
    )

    new_count: int = 0
    new_counts_by_class: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int64)
    )




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

    


class DensityMetric:
    def __init__(
        self,
        distance_meters: float,
        num_classes: int = 4,
    ):
        self.distance_meters = distance_meters
        self.num_classes = num_classes

        self.density_history = []
        self.density_by_class_history = []

    def compute(
        self,
        polygon_mask: np.ndarray,
        det_cls: np.ndarray,
    ) -> DensityResult:

        if polygon_mask is None or self.distance_meters <= 0:
            return DensityResult(
                average_density=0.0,
                average_density_by_class=np.zeros(
                    self.num_classes, dtype=np.float32
                ),
                current_density=0.0,
                current_density_by_class=np.zeros(
                    self.num_classes, dtype=np.float32
                ),
            )

        selected_classes = det_cls[polygon_mask]

        counts_by_class = np.bincount(
            selected_classes,
            minlength=self.num_classes,
        )

        current_density_by_class = (
            counts_by_class / self.distance_meters
        ).astype(np.float32)

        current_density = float(
            current_density_by_class.sum()
        )

        self.density_history.append(current_density)
        self.density_by_class_history.append(
            current_density_by_class
        )

        average_density = float(
            np.mean(self.density_history)
        )

        average_density_by_class = np.mean(
            self.density_by_class_history,
            axis=0,
        ).astype(np.float32)

        return DensityResult(
            average_density=average_density,
            average_density_by_class=average_density_by_class.copy(),
            current_density=current_density,
            current_density_by_class=current_density_by_class.copy(),
        )

        

class Counter:
    def __init__(self, counter_logic="counter_5", ttl_seconds=5.0, num_classes=4):
        self.counter_logic = counter_logic
        self.ttl_seconds = ttl_seconds
        self.num_classes = num_classes

        # lifecycle state
        self._counted_ids = {}  # track_id -> last_seen_time

        # aggregation state
        self.cumulative_counts_by_class = np.zeros(num_classes, dtype=np.int64)
        self.cumulative_count = 0

        # -------------------------
        # Bind strategy ONCE
        # -------------------------
        if counter_logic == "counter_2":
            self._counter_fn = lambda c, v: c
        elif counter_logic == "counter_3":
            self._counter_fn = lambda c, v: v
        elif counter_logic == "counter_4":
            self._counter_fn = lambda c, v: c & v
        elif counter_logic == "counter_5":
            self._counter_fn = lambda c, v: c | v
        else:
            raise ValueError(f"Unknown counter type: {counter_logic}")

    # -------------------------
    # TTL cleanup (time-based)
    # -------------------------
    def _cleanup(self, current_time: float):
        to_delete = [
            tid for tid, last in self._counted_ids.items()
            if (current_time - last) > self.ttl_seconds
        ]
        for tid in to_delete:
            del self._counted_ids[tid]

    # -------------------------
    # New IDs mask
    # -------------------------
    def _new_ids_mask(self, track_ids: np.ndarray) -> np.ndarray:
        return np.array([tid not in self._counted_ids for tid in track_ids], dtype=bool)

    # -------------------------
    # Main compute
    # -------------------------
    def compute(
        self,
        track_ids: np.ndarray,
        det_cls: np.ndarray,
        current_time: float,
        crossed_mask: np.ndarray = None,
        vicinity_mask: np.ndarray = None,
        polygon_mask: np.ndarray = None,
    ):
        # cleanup
        self._cleanup(current_time)

        # strategy
        counter_mask = self._counter_fn(crossed_mask, vicinity_mask)

        # new IDs only
        new_ids_mask = self._new_ids_mask(track_ids)

        # print(f"crossed_mask shape is {crossed_mask.shape}")
        # print(f"crossed_mask  is {crossed_mask}")
        
        # print(f"vicinity_mask shape is {vicinity_mask.shape}")
        # print(f"vicinity_mask is {vicinity_mask}")

        # print(f"counter_mask shape is {counter_mask.shape}")
        # print(f"new_ids_mask shape is {new_ids_mask.shape}")
        
        mask = counter_mask & new_ids_mask

        # optional ROI
        if polygon_mask is not None:
            mask &= polygon_mask
            # print(f"mask shape is {mask.shape}")
            # print(f"polygon_mask shape is {polygon_mask.shape}")
            # print(f"track_ids shape is {track_ids.shape}")

        selected_ids = track_ids[mask]
        selected_classes = det_cls[mask]

        # -------------------------
        # Update lifecycle state
        # -------------------------
        for tid in selected_ids:
            self._counted_ids[tid] = current_time

        # -------------------------
        # Update aggregation
        # -------------------------
        self.cumulative_count += len(selected_ids)

        counts = np.bincount(selected_classes, minlength=self.num_classes)
        self.cumulative_counts_by_class += counts

        return CountResult(
            cumulative_count=self.cumulative_count,
            cumulative_counts_by_class=self.cumulative_counts_by_class.copy(),
            new_count=len(selected_ids),
            new_counts_by_class=counts.copy()
        )

    

class FlowMetric:
    def __init__(self, num_classes: int = 4):
        self.num_classes = num_classes

    def compute(
        self,
        cumulative_count: int,
        cumulative_counts_by_class: np.ndarray,
        current_time: float,
    ) -> FlowResult:

        if current_time <= 0:
            return FlowResult(
                average_flow=0.0,
                average_flow_by_class=np.zeros(
                    self.num_classes,
                    dtype=np.float32,
                ),
            )

        average_flow = cumulative_count / current_time

        average_flow_by_class = (
            cumulative_counts_by_class / current_time
        ).astype(np.float32)

        return FlowResult(
            average_flow=float(average_flow),
            average_flow_by_class=average_flow_by_class.copy(),
        )



        


class SpaceHeadwayMetric:
    def __init__(self, direction: np.ndarray):
        self.direction = direction / np.linalg.norm(direction)

        self.headway_history = []
        self.average_space_headway = None

    def compute(
        self,
        points: np.ndarray,
        polygon_mask: np.ndarray,
    ) -> SpaceHeadwayResult:

        pts = points[polygon_mask]
        if pts.shape[0] < 2:
            return SpaceHeadwayResult(
                average_space_headway=self.average_space_headway,
                current_space_headway=None,
            )

        s = pts @ self.direction
        s_sorted = np.sort(s)

        headways = np.diff(s_sorted)
        
        if headways.size == 0:
            return SpaceHeadwayResult()

        current_space_headway = float(headways.mean())
        

        self.headway_history.append(
            current_space_headway
        )

        self.average_space_headway = float(
            np.mean(self.headway_history)
        )

        return SpaceHeadwayResult(
            average_space_headway=self.average_space_headway,
            current_space_headway=current_space_headway,
        )

        

class TimeHeadwayMetric:
    def __init__(self):
        self._last_time = None
        self._headway_history = []
        self.average_time_headway = None

    def compute(self, new_crossings) -> TimeHeadwayResult:

        if not new_crossings:
            return TimeHeadwayResult(
                average_time_headway=self.average_time_headway,
                current_time_headway=None,
            )

        sum_hw = 0.0
        count = 0

        for t in new_crossings:

            if self._last_time is not None:
                sum_hw += (t - self._last_time)
                count += 1

            self._last_time = t

        if count == 0:
            return TimeHeadwayResult(
                average_time_headway=self.average_time_headway,
                current_time_headway=None,
            )

        current_time_headway = sum_hw / count

        self._headway_history.append(
            current_time_headway
        )

        self.average_time_headway = (
            sum(self._headway_history)
            / len(self._headway_history)
        )

        return TimeHeadwayResult(
            average_time_headway=float(
                self.average_time_headway
            ),
            current_time_headway=float(
                current_time_headway
            ),
        )

class OccupancyMetric:
    def __init__(self):
        self._num_frames = 0
        self._num_occupied_frames = 0

    def compute(
        self,
        vicinity_mask: np.ndarray,
        polygon_mask: np.ndarray,
    ) -> OccupancyResult:

        mask = vicinity_mask & polygon_mask

        current_occupancy = float(mask.any())

        self._num_frames += 1
        self._num_occupied_frames += current_occupancy

        average_occupancy = (
            self._num_occupied_frames
            / self._num_frames
        )

        return OccupancyResult(
            average_occupancy=float(average_occupancy),
            current_occupancy=current_occupancy,
        )