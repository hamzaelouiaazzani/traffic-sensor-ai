from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Optional

import torch

from perception.pipeline import (
    DetectionFilter,
    FrameAcquisitionFilter,
    ObservationBufferSink,
    ObservationExtractionFilter,
    TrackingFilter,
)
from runtime.observations import PeriodObservationBuffer
from runtime.timing import FrameTiming, TimingPolicy
from utils.profilers import Profile


@dataclass(frozen=True)
class PerceptionRuntimeStep:
    """
    Result of one Stage-1 pipeline execution step.

    The step reports what happened while processing one frame. It does not
    decide period boundaries; callers decide whether to request another step.
    """

    timing: Optional[FrameTiming]
    frame_processed: bool
    end_of_stream: bool
    processing_latency_seconds: float = 0.0


class PerceptionRuntime:
    """
    Runtime executor for the perception pipeline.

    ADD role: execution/orchestration element for Stage 1 only. It invokes the
    instantiated perception filters in order and sinks observations into a
    caller-owned PeriodObservationBuffer. It intentionally does not own
    reporting-period stop policy.
    """

    def __init__(
        self,
        producer,
        detector,
        tracker,
        timing_policy: TimingPolicy,
        buffer: PeriodObservationBuffer,
        device=None,
        latest_frame_store=None,
    ):
        self.buffer = buffer
        self.detection_profile = Profile(device=device)
        self.tracking_profile = Profile(device=device)
        self._last_processing_latency = 0.0

        self._acquisition = FrameAcquisitionFilter(
            producer=producer,
            timing_policy=timing_policy,
            latest_frame_store=latest_frame_store,
        )
        self._detection = DetectionFilter(detector, self.detection_profile)
        self._tracking = TrackingFilter(tracker, self.tracking_profile)
        self._observation_extraction = ObservationExtractionFilter()
        self._sink = ObservationBufferSink(buffer)

    @property
    def detection_seconds(self) -> float:
        return self.detection_profile.t

    @property
    def tracking_seconds(self) -> float:
        return self.tracking_profile.t

    def process_next_frame(self) -> PerceptionRuntimeStep:
        """
        Pull one frame through the Stage-1 perception pipeline.

        The caller owns all period policy, including when to stop requesting
        frames and when to freeze the buffer.
        """
        with torch.inference_mode():
            timed_frame = self._acquisition.pull(
                processing_latency=self._last_processing_latency,
            )
            if timed_frame is None:
                return PerceptionRuntimeStep(
                    timing=None,
                    frame_processed=False,
                    end_of_stream=True,
                    processing_latency_seconds=0.0,
                )

            frame_processing_start = time.perf_counter()
            detections = self._detection.process(timed_frame)
            tracks = self._tracking.process(detections)
            observations = self._observation_extraction.process(tracks)
            self._sink.consume(observations)

            self._last_processing_latency = time.perf_counter() - frame_processing_start
            return PerceptionRuntimeStep(
                timing=timed_frame.timing,
                frame_processed=True,
                end_of_stream=False,
                processing_latency_seconds=self._last_processing_latency,
            )
