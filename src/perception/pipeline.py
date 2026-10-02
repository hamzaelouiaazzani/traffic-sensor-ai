from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from runtime.observations import PeriodObservationBuffer
from runtime.timing import FrameTiming, TimingPolicy
from utils.profilers import Profile
from perception.vision_io.frame_producer import RealTimeSimulationProducer



@dataclass(frozen=True)
class TimedFramePipe:
    """Pipe data from acquisition/timing into detection."""

    frame: Any
    timing: FrameTiming


@dataclass(frozen=True)
class DetectionPipe:
    """Pipe data from detection into tracking."""

    timed_frame: TimedFramePipe
    detections: np.ndarray


@dataclass(frozen=True)
class TrackingPipe:
    """Pipe data from tracking into observation extraction."""

    timed_frame: TimedFramePipe
    tracks: np.ndarray


@dataclass(frozen=True)
class ObservationPipe:
    """Pipe data from observation extraction into the period buffer sink."""

    timing: FrameTiming
    points: np.ndarray
    track_ids: np.ndarray
    class_ids: np.ndarray


class FrameAcquisitionFilter:
    """
    Pipeline filter that pulls one frame and attaches authoritative timing.

    ADD role: instantiated filter element for acquisition/timing. It preserves
    producer-specific acquisition behavior while exposing one stable pipe data
    contract to downstream filters.
    """

    def __init__(self, producer, timing_policy: TimingPolicy, latest_frame_store=None):
        self.producer = producer
        self.timing_policy = timing_policy
        self.latest_frame_store = latest_frame_store

    def pull(self, processing_latency: float = 0.0) -> Optional[TimedFramePipe]:
        if isinstance(self.producer, RealTimeSimulationProducer):
            frame = self.producer.next_frame(processing_latency=processing_latency)
        else:
            frame = self.producer.next_frame()

        if frame is None:
            return None

        timing = self.timing_policy.resolve(frame)
        if self.latest_frame_store is not None:
            self.latest_frame_store.update(frame.data, timing.frame_id)

        return TimedFramePipe(frame=frame, timing=timing)


class DetectionFilter:
    """
    Pipeline filter that transforms a timed frame into canonical detections.

    The detector implementation remains replaceable behind its stable contract.
    """

    def __init__(self, detector, profile: Profile):
        self.detector = detector
        self.profile = profile

    def process(self, pipe: TimedFramePipe) -> DetectionPipe:
        with self.profile:
            detections = self.detector.detect_to_track(pipe.frame.data)
        return DetectionPipe(timed_frame=pipe, detections=detections)


class TrackingFilter:
    """
    Pipeline filter that transforms detections into tracked object rows.

    The tracker is intentionally stateful and must be reused across periods by
    the runtime assembly.
    """

    def __init__(self, tracker, profile: Profile):
        self.tracker = tracker
        self.profile = profile

    def process(self, pipe: DetectionPipe) -> TrackingPipe:
        with self.profile:
            tracks = self.tracker.update(
                pipe.detections,
                pipe.timed_frame.frame.data,
            )
        return TrackingPipe(timed_frame=pipe.timed_frame, tracks=tracks)


class ObservationExtractionFilter:
    """
    Pipeline filter that transforms tracker output into observation columns.

    The canonical vehicle point is always the geometric center of the tracked
    bounding box; this is an architectural invariant, not a runtime policy.
    """

    def process(self, pipe: TrackingPipe) -> ObservationPipe:
        points, track_ids, class_ids = extract_tracking_outputs(pipe.tracks)
        return ObservationPipe(
            timing=pipe.timed_frame.timing,
            points=points,
            track_ids=track_ids,
            class_ids=class_ids,
        )


class ObservationBufferSink:
    """
    Pipeline sink backed by PeriodObservationBuffer.

    ADD role: instantiated bounded temporal accumulator element. It is the data
    staging endpoint of Stage 1, not a filter.
    """

    def __init__(self, buffer: PeriodObservationBuffer):
        self.buffer = buffer

    def consume(self, pipe: ObservationPipe) -> None:
        self.buffer.append_frame(
            timing=pipe.timing,
            track_ids=pipe.track_ids,
            class_ids=pipe.class_ids,
            points=pipe.points,
        )


def extract_tracking_outputs(tracks: np.ndarray):
    if tracks is None or len(tracks) == 0:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
        )

    tracks = np.asarray(tracks)
    if tracks.ndim != 2 or tracks.shape[1] < 7:
        raise ValueError("tracker output must have at least 7 columns")

    bboxes = tracks[:, :4].astype(np.float32, copy=False)

    points = np.empty((bboxes.shape[0], 2), dtype=np.float32)
    points[:, 0] = (bboxes[:, 0] + bboxes[:, 2]) * 0.5
    points[:, 1] = (bboxes[:, 1] + bboxes[:, 3]) * 0.5

    track_ids = tracks[:, 4].astype(np.int32)
    class_ids = tracks[:, 6].astype(np.int32)

    return points, track_ids, class_ids
