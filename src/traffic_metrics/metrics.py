import logging
import numpy as np
from dataclasses import dataclass, field
from collections import deque


@dataclass
class CountResult:
    total_count: int = 0                # cumulative
    counts_by_class: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    new_count: int = 0                  # 🔥 incremental (this frame)
    new_counts_by_class: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))




    
class DensityMetric:
    def compute(self, area, cache, class_ids, num_classes=None):
        # ❌ No ROI → no density
        if area.zone is None:
            logging.warning(f"{area.name}: no polygon (ROI), density undefined.")
            return None

        # ❌ No metric length → invalid density
        L = area.zone.distance_meters
        if L is None or L <= 0:
            logging.warning(f"{area.name}: invalid distance_meters, density undefined.")
            return None

        # ✔ polygon mask
        mask = cache["polygon"][area.zone.polygon_id].astype(bool)

        # ✔ selected classes
        selected_classes = class_ids[mask]

        # infer number of classes
        if num_classes is None:
            num_classes = int(class_ids.max()) + 1 if class_ids.size else 0

        # ✔ per-class counts
        counts_by_class = np.bincount(selected_classes, minlength=num_classes)

        # ✔ density per class
        density_by_class = counts_by_class / L

        # ✔ total density
        total_density = density_by_class.sum()

        return total_density, density_by_class

    
class Counter:
    def __init__(self, counter_type="counter_5"):
        self.counter_type = counter_type
        self._counted_ids = set()
        self._counts_by_class = np.zeros(0, dtype=np.int64)

    # -------------------------
    # New IDs filter
    # -------------------------
    def get_new_ids_mask(self, track_ids: np.ndarray) -> np.ndarray:
        return ~np.isin(track_ids, list(self._counted_ids))

    # -------------------------
    # Counter logic
    # -------------------------
    def _counter_mask(self, crossed_mask, vicinity_mask):
        if self.counter_type == "counter_2":
            return crossed_mask
        elif self.counter_type == "counter_3":
            return vicinity_mask
        elif self.counter_type == "counter_4":
            return crossed_mask & vicinity_mask
        elif self.counter_type == "counter_5":
            return crossed_mask | vicinity_mask
        else:
            raise ValueError(f"Unknown counter type: {self.counter_type}")

    # -------------------------
    # Main compute
    # -------------------------
    def compute(self, area, cache, track_ids, class_ids, num_classes=None):
        line = area.flow_line
        idx = line.get_idx()

        crossed_mask = line.get_crossed_mask(track_ids)
        vicinity_mask = cache["line"]["vicinity_mask"][idx].astype(bool)

        counter_mask = self._counter_mask(crossed_mask, vicinity_mask)
        new_ids_mask = self.get_new_ids_mask(track_ids)

        mask = counter_mask & new_ids_mask

        if area.zone is not None:
            poly_mask = cache["polygon"][area.zone.polygon_id].astype(bool)
            mask = mask & poly_mask

        selected_ids = track_ids[mask]
        selected_classes = class_ids[mask]

        # update counted IDs
        self._counted_ids.update(selected_ids.tolist())

        # infer number of classes
        if num_classes is None:
            num_classes = int(class_ids.max()) + 1 if class_ids.size else 0

        # ensure capacity
        if len(self._counts_by_class) < num_classes:
            new = np.zeros(num_classes, dtype=np.int64)
            new[:len(self._counts_by_class)] = self._counts_by_class
            self._counts_by_class = new

        # accumulate counts
        counts = np.bincount(selected_classes, minlength=num_classes)
        self._counts_by_class[:len(counts)] += counts

        return CountResult(
            total_count=len(self._counted_ids),
            counts_by_class=self._counts_by_class.copy(),
            new_count=len(selected_ids),                        # 🔥 key for Flow
            new_counts_by_class=counts                          # per-class increment
        )





class FlowMetric:
    def __init__(self, counter, time_window_sec=60.0, fps=30.0, output_unit="veh/h"):
        self.counter = counter
        self.time_window_sec = time_window_sec
        self.fps = fps
        self.output_unit = output_unit

        self._window_size = int(time_window_sec * fps)

        # FIFO buffers (total + per-class)
        self._buffer = deque(maxlen=self._window_size)
        self._buffer_by_class = deque(maxlen=self._window_size)

    # -------------------------
    # Main update (per frame)
    # -------------------------
    def update(self, area, cache, track_ids, class_ids):
        # 1. get incremental counts (NO recomputation)
        result = self.counter.compute(area, cache, track_ids, class_ids)

        new_count = result.new_count
        new_counts_by_class = result.new_counts_by_class

        # 2. push to buffers
        self._buffer.append(new_count)
        self._buffer_by_class.append(new_counts_by_class)

        # 3. aggregate window
        total = sum(self._buffer)

        # handle per-class aggregation safely
        if self._buffer_by_class:
            max_len = max(len(x) for x in self._buffer_by_class)
            counts_by_class = np.zeros(max_len, dtype=np.float64)

            for arr in self._buffer_by_class:
                counts_by_class[:len(arr)] += arr
        else:
            counts_by_class = np.zeros(0)

        # 4. normalize by time
        flow = total / self.-
        flow_by_class = counts_by_class / self.time_window_sec

        # 5. unit conversion
        if self.output_unit == "veh/h":
            flow *= 3600
            flow_by_class *= 3600

        return flow, flow_by_class