import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from config.load_build import load_config
from runtime.continuity import boundary_batch_for_next_period, resolve_continuity_policy
from runtime.observations import PeriodObservationBuffer
from runtime.periods import resolve_period_policy
from runtime.perception_runtime import PerceptionRuntime
from runtime.timing import TimingPolicy


def build_tracker(cfg: dict):
    from perception.tracking.factory import build_tracker_from_config

    return build_tracker_from_config(cfg["tracker"])


def save_perception_batch(batch, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"period_{batch.period_idx:06d}_perception.npz"
    np.savez_compressed(
        output_path,
        timestamp=batch.timestamp,
        frame_id=batch.frame_id,
        track_id=batch.track_id,
        class_id=batch.class_id,
        points=batch.points,
        is_context=batch.is_context,
        period_frame_ids=batch.period_frame_ids,
        period_timestamps=batch.period_timestamps,
        period_idx=np.asarray(batch.period_idx, dtype=np.int64),
        timing_mode=np.asarray(batch.timing_mode),
        fps=np.asarray(np.nan if batch.fps is None else batch.fps, dtype=np.float64),
    )
    return output_path


def run_perception(
    config_path: str,
    output_dir: str,
    source: Optional[str] = None,
    fps: Optional[float] = None,
    period_mins: Optional[float] = None,
    max_periods: Optional[int] = None,
) -> List[Path]:
    cfg = load_config(config_path)

    from perception.detection.factory import build_detector_from_config
    from scripts.sensor_pipeline import (
        build_frame_producer,
        resolve_capacity_fps,
        resolve_detector_device,
        resolve_fps,
        resolve_source,
    )

    source = resolve_source(cfg, source)

    fps = resolve_fps(cfg, fps)
    period_policy = resolve_period_policy(
        cfg=cfg,
        fps=resolve_capacity_fps(cfg, fps),
        period_mins_override=period_mins,
    )
    continuity_policy = resolve_continuity_policy(cfg)

    detector = build_detector_from_config(cfg["detector"])

    _ = resolve_detector_device(detector)

    tracker = build_tracker(cfg)

    timing_policy = TimingPolicy.from_config(cfg, fps_override=fps)
    producer = build_frame_producer(source, cfg, fps)

    output_paths = []
    boundary_context = None
    period_idx = 0

    producer.start()
    try:
        while max_periods is None or period_idx < max_periods:
            buffer = PeriodObservationBuffer(period_policy.max_observations)
            if continuity_policy.enabled:
                buffer.add_context(boundary_context)

            perception_runtime = PerceptionRuntime(
                producer=producer,
                detector=detector,
                tracker=tracker,
                timing_policy=timing_policy,
                buffer=buffer,
            )

            first_timing = None
            frames_processed = 0
            end_of_stream = False
            while True:
                step = perception_runtime.process_next_frame()
                if step.end_of_stream:
                    end_of_stream = True
                    break
                if not step.frame_processed:
                    break

                frames_processed += 1
                if first_timing is None:
                    first_timing = step.timing

                if timing_policy.period_elapsed_seconds(first_timing, step.timing) >= period_policy.period_seconds:
                    break

            if frames_processed == 0:
                break

            batch = buffer.freeze(
                period_idx=period_idx,
                timing_mode=timing_policy.mode,
                fps=timing_policy.fps,
            )

            output_path = save_perception_batch(batch, Path(output_dir))
            output_paths.append(output_path)

            print(
                f"saved {output_path} "
                f"frames={batch.frame_count} "
                f"observations={batch.observation_count} "
                f"context={batch.observation_count - batch.active_observation_count}"
            )

            boundary_context = boundary_batch_for_next_period(
                batch=batch,
                policy=continuity_policy,
            )

            period_idx += 1

            if end_of_stream:
                break

    finally:
        producer.release()
        if hasattr(tracker, "close"):
            tracker.close()
        if hasattr(detector, "close"):
            detector.close()

    return output_paths


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run perception runtime only and export PeriodObservationBatch files."
    )
    parser.add_argument("--config", default="config/traffic_metrics.yaml", help="YAML runtime config path.")
    parser.add_argument("--output", default="outputs/perception", help="Directory for per-period .npz exports.")
    parser.add_argument("--source", default=None, help="Override input.source from config.")
    parser.add_argument("--fps", type=float, default=None, help="Override config FPS.")
    parser.add_argument("--period-mins", type=float, default=None, help="Override analytics period length in minutes.")
    parser.add_argument("--max-periods", type=int, default=None, help="Optional limit for inspection runs.")
    return parser.parse_args()


def main():
    args = parse_args()
    run_perception(
        config_path=args.config,
        output_dir=args.output,
        source=args.source,
        fps=args.fps,
        period_mins=args.period_mins,
        max_periods=args.max_periods,
    )


if __name__ == "__main__":
    main()
