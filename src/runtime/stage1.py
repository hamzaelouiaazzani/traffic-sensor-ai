from __future__ import annotations

from dataclasses import dataclass
from threading import Event
import time
from typing import Optional

import numpy as np
import torch

from runtime.continuity import ContinuityPolicy
from runtime.observations import PeriodObservationBatch, PeriodObservationBuffer
from runtime.timing import TimingPolicy
from utils.profilers import Profile
from video_io.frame_producer import RealTimeSimulationProducer


@dataclass(frozen=True)
class Stage1PeriodResult:
    batch: Optional[PeriodObservationBatch]
    frames_processed: int
    end_of_stream: bool
    detection_seconds: float = 0.0
    tracking_seconds: float = 0.0


def extract_tracking_outputs(tracks: np.ndarray):
    if tracks is None or len(tracks) == 0:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
        )

    tracks = np.asarray(tracks)
    if tracks.ndim != 2 or tracks.shape[1] < 7:
        raise ValueError("tracker output must have at least 7 columns")

    bboxes = tracks[:, :4].astype(np.float32, copy=False)

    points = np.empty((bboxes.shape[0], 2), dtype=np.float32)
    points[:, 0] = (bboxes[:, 0] + bboxes[:, 2]) * 0.5
    points[:, 1] = bboxes[:, 3]

    track_ids = tracks[:, 4].astype(np.int32)
    class_ids = tracks[:, 6].astype(np.int32)

    return points, bboxes, track_ids, class_ids


def process_stage1_period(
    producer,
    detector,
    tracker,
    timing_policy: TimingPolicy,
    period_seconds: float,
    max_observations: int,
    period_idx: int,
    continuity_policy: ContinuityPolicy,
    boundary_context: Optional[PeriodObservationBatch] = None,
    device=None,
    stop_event: Optional[Event] = None,
    latest_frame_store=None,
) -> Stage1PeriodResult:
    buffer = PeriodObservationBuffer(max_observations)
    if continuity_policy.enabled:
        buffer.add_context(boundary_context)

    det_profile = Profile(device=device)
    track_profile = Profile(device=device)

    first_timing = None
    frames_processed = 0
    end_of_stream = False
    last_processing_latency = 0.0

    with torch.inference_mode():
        while True:
            if stop_event is not None and stop_event.is_set():
                break

            if isinstance(producer, RealTimeSimulationProducer):
                frame = producer.next_frame(processing_latency=last_processing_latency)
            else:
                frame = producer.next_frame()

            if frame is None:
                end_of_stream = True
                break

            frame_processing_start = time.perf_counter()
            timing = timing_policy.resolve(frame)
            if first_timing is None:
                first_timing = timing

            if latest_frame_store is not None:
                latest_frame_store.update(frame.data, timing.frame_id)

            with det_profile:
                detections = detector.detect_to_track(frame.data)

            with track_profile:
                tracks = tracker.update(detections, frame.data)

            points, bboxes, track_ids, class_ids = extract_tracking_outputs(tracks)
            buffer.append_frame(
                timing=timing,
                track_ids=track_ids,
                class_ids=class_ids,
                points=points,
                bboxes=bboxes,
            )
            frames_processed += 1
            last_processing_latency = time.perf_counter() - frame_processing_start

            if timing_policy.period_elapsed_seconds(first_timing, timing) >= period_seconds:
                break

    if frames_processed == 0:
        return Stage1PeriodResult(
            batch=None,
            frames_processed=0,
            end_of_stream=end_of_stream,
            detection_seconds=det_profile.t,
            tracking_seconds=track_profile.t,
        )

    batch = buffer.freeze(
        period_idx=period_idx,
        timing_mode=timing_policy.mode,
        fps=timing_policy.fps,
    )
    return Stage1PeriodResult(
        batch=batch,
        frames_processed=frames_processed,
        end_of_stream=end_of_stream,
        detection_seconds=det_profile.t,
        tracking_seconds=track_profile.t,
    )
