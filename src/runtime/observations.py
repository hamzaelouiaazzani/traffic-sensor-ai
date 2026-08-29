from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from runtime.timing import FrameTiming


class PeriodObservationBufferOverflow(BufferError):
    """Raised when a bounded period observation buffer cannot accept more rows."""


@dataclass(frozen=True)
class PeriodObservationBatch:
    """Column-oriented observation batch passed from perception to analytics."""

    timestamp: np.ndarray
    frame_id: np.ndarray
    track_id: np.ndarray
    class_id: np.ndarray
    points: np.ndarray
    bboxes: np.ndarray
    is_context: np.ndarray
    period_frame_ids: np.ndarray
    period_timestamps: np.ndarray
    period_idx: int
    timing_mode: str
    fps: Optional[float] = None

    @property
    def observation_count(self) -> int:
        return int(self.track_id.shape[0])

    @property
    def frame_count(self) -> int:
        return int(self.period_frame_ids.shape[0])

    @property
    def active_mask(self) -> np.ndarray:
        return ~self.is_context

    @property
    def active_observation_count(self) -> int:
        return int(np.count_nonzero(self.active_mask))

    @property
    def start_frame(self) -> Optional[int]:
        if self.frame_count == 0:
            return None
        return int(self.period_frame_ids[0])

    @property
    def end_frame(self) -> Optional[int]:
        if self.frame_count == 0:
            return None
        return int(self.period_frame_ids[-1])

    @property
    def source_frames_elapsed(self) -> int:
        if self.frame_count == 0:
            return 0
        return int(self.period_frame_ids[-1] - self.period_frame_ids[0] + 1)

    @property
    def duration_seconds(self) -> float:
        if self.frame_count == 0:
            return 0.0

        if self.timing_mode == "frame" and self.fps is not None and self.fps > 0:
            return max(self.source_frames_elapsed / self.fps, 1.0 / self.fps)

        elapsed = float(self.period_timestamps[-1] - self.period_timestamps[0])
        if elapsed > 0:
            return elapsed
        if self.fps is not None and self.fps > 0:
            return 1.0 / self.fps
        return 0.0

    def final_observations_by_track(self) -> "PeriodObservationBatch":
        """Return one latest non-context observation per track for boundary continuity."""

        active = self.active_mask
        if not np.any(active):
            return self._slice(np.zeros((0,), dtype=np.int64), context_override=True)

        idx = np.flatnonzero(active)
        order = np.lexsort((self.frame_id[idx], self.timestamp[idx], self.track_id[idx]))
        ordered_idx = idx[order]
        ordered_track_ids = self.track_id[ordered_idx]

        keep = np.ones((ordered_idx.shape[0],), dtype=bool)
        keep[:-1] = ordered_track_ids[:-1] != ordered_track_ids[1:]
        latest_idx = ordered_idx[keep]

        latest_order = np.argsort(self.timestamp[latest_idx], kind="stable")
        return self._slice(latest_idx[latest_order], context_override=True)

    def boundary_observations(self, end_time: float, max_age_seconds: float) -> "PeriodObservationBatch":
        """Return latest observations per track that may still affect the next period."""

        if self.observation_count == 0:
            return self._slice(np.zeros((0,), dtype=np.int64), context_override=True)

        order = np.lexsort((self.frame_id, self.timestamp, self.track_id))
        ordered_idx = order
        ordered_track_ids = self.track_id[ordered_idx]

        keep = np.ones((ordered_idx.shape[0],), dtype=bool)
        keep[:-1] = ordered_track_ids[:-1] != ordered_track_ids[1:]
        latest_idx = ordered_idx[keep]

        if max_age_seconds is not None:
            recent = (float(end_time) - self.timestamp[latest_idx]) <= float(max_age_seconds)
            latest_idx = latest_idx[recent]

        latest_order = np.argsort(self.timestamp[latest_idx], kind="stable")
        return self._slice(latest_idx[latest_order], context_override=True)

    def _slice(
        self,
        idx: np.ndarray,
        context_override: Optional[bool] = None,
    ) -> "PeriodObservationBatch":
        is_context = self.is_context[idx].copy()
        if context_override is not None:
            is_context[:] = bool(context_override)

        return PeriodObservationBatch(
            timestamp=self.timestamp[idx].copy(),
            frame_id=self.frame_id[idx].copy(),
            track_id=self.track_id[idx].copy(),
            class_id=self.class_id[idx].copy(),
            points=self.points[idx].copy(),
            bboxes=self.bboxes[idx].copy(),
            is_context=is_context,
            period_frame_ids=np.empty((0,), dtype=np.int64),
            period_timestamps=np.empty((0,), dtype=np.float64),
            period_idx=self.period_idx,
            timing_mode=self.timing_mode,
            fps=self.fps,
        )


class PeriodObservationBuffer:
    """Bounded preallocated storage for one reporting period."""

    def __init__(self, max_observations: int):
        max_observations = int(max_observations)
        if max_observations <= 0:
            raise ValueError("max_observations must be positive")

        self.max_observations = max_observations
        self.timestamp = np.empty((max_observations,), dtype=np.float64)
        self.frame_id = np.empty((max_observations,), dtype=np.int64)
        self.track_id = np.empty((max_observations,), dtype=np.int64)
        self.class_id = np.empty((max_observations,), dtype=np.int32)
        self.points = np.empty((max_observations, 2), dtype=np.float32)
        self.bboxes = np.empty((max_observations, 4), dtype=np.float32)
        self.is_context = np.empty((max_observations,), dtype=bool)
        self._size = 0
        self._period_frame_ids = []
        self._period_timestamps = []

    @property
    def size(self) -> int:
        return self._size

    @property
    def frame_count(self) -> int:
        return len(self._period_frame_ids)

    def clear(self) -> None:
        self._size = 0
        self._period_frame_ids.clear()
        self._period_timestamps.clear()

    def add_context(self, batch: Optional[PeriodObservationBatch]) -> None:
        if batch is None or batch.observation_count == 0:
            return

        self._append_arrays(
            timestamp=batch.timestamp,
            frame_id=batch.frame_id,
            track_id=batch.track_id,
            class_id=batch.class_id,
            points=batch.points,
            bboxes=batch.bboxes,
            is_context=True,
        )

    def append_frame(
        self,
        timing: FrameTiming,
        track_ids: np.ndarray,
        class_ids: np.ndarray,
        points: np.ndarray,
        bboxes: np.ndarray,
    ) -> None:
        self._period_frame_ids.append(timing.frame_id)
        self._period_timestamps.append(timing.timestamp)

        n = int(track_ids.shape[0])
        if n == 0:
            return

        self._append_arrays(
            timestamp=np.full((n,), timing.timestamp, dtype=np.float64),
            frame_id=np.full((n,), timing.frame_id, dtype=np.int64),
            track_id=track_ids.astype(np.int64, copy=False),
            class_id=class_ids.astype(np.int32, copy=False),
            points=points.astype(np.float32, copy=False),
            bboxes=bboxes.astype(np.float32, copy=False),
            is_context=False,
        )

    def freeze(
        self,
        period_idx: int,
        timing_mode: str,
        fps: Optional[float],
    ) -> PeriodObservationBatch:
        size = self._size
        return PeriodObservationBatch(
            timestamp=self.timestamp[:size].copy(),
            frame_id=self.frame_id[:size].copy(),
            track_id=self.track_id[:size].copy(),
            class_id=self.class_id[:size].copy(),
            points=self.points[:size].copy(),
            bboxes=self.bboxes[:size].copy(),
            is_context=self.is_context[:size].copy(),
            period_frame_ids=np.asarray(self._period_frame_ids, dtype=np.int64),
            period_timestamps=np.asarray(self._period_timestamps, dtype=np.float64),
            period_idx=int(period_idx),
            timing_mode=timing_mode,
            fps=fps,
        )

    def _append_arrays(
        self,
        timestamp: np.ndarray,
        frame_id: np.ndarray,
        track_id: np.ndarray,
        class_id: np.ndarray,
        points: np.ndarray,
        bboxes: np.ndarray,
        is_context: bool,
    ) -> None:
        n = int(track_id.shape[0])
        if self._size + n > self.max_observations:
            raise PeriodObservationBufferOverflow(
                f"Period observation capacity exceeded: requested {self._size + n}, "
                f"capacity {self.max_observations}"
            )

        end = self._size + n
        self.timestamp[self._size:end] = timestamp
        self.frame_id[self._size:end] = frame_id
        self.track_id[self._size:end] = track_id
        self.class_id[self._size:end] = class_id
        self.points[self._size:end] = points
        self.bboxes[self._size:end] = bboxes
        self.is_context[self._size:end] = is_context
        self._size = end
