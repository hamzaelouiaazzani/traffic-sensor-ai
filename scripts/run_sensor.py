import argparse

import numpy as np
import torch

from config.load_build import (
    build_areas,
    build_runtime_components,
    load_config,
)
from detection.factory import build_detector
from tracking.track import Tracker
from utils.profilers import Profile
from video_io.frame_producer import DirectFrameProducer


def extract_tracking_outputs(tracks: np.ndarray):
    if tracks is None or len(tracks) == 0:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
        )

    bboxes = tracks[:, :4]

    points = np.empty((bboxes.shape[0], 2), dtype=np.float32)
    points[:, 0] = (bboxes[:, 0] + bboxes[:, 2]) * 0.5
    points[:, 1] = bboxes[:, 3]

    track_ids = tracks[:, 4].astype(np.int32)
    det_cls = tracks[:, 6].astype(np.int32)

    return points, bboxes, track_ids, det_cls


def compute_area_metrics(
    areas,
    geometry_engine,
    points,
    track_ids,
    det_cls,
    current_time,
    line_cache,
    polygon_cache,
    crossed_masks,
    new_crossings,
):
    results = {}

    for area in areas:
        area_results = {}

        polygon_mask = None
        if area.zone is not None:
            polygon_mask = polygon_cache[area.zone.polygon_id]

        crossed_mask = None
        vicinity_mask = None
        new_area_crossings = []

        if area.flow_line is not None:
            line_idx = geometry_engine._line_id_to_idx[area.flow_line.line_id]
            crossed_mask = crossed_masks[line_idx]
            vicinity_mask = line_cache["vicinity_mask"][line_idx]
            new_area_crossings = new_crossings.get(area.area_id, [])

        if "flow" in area.metrics:
            counter_res = area.metrics["counter"].compute(
                track_ids=track_ids,
                det_cls=det_cls,
                current_time=current_time,
                crossed_mask=crossed_mask,
                vicinity_mask=vicinity_mask,
                polygon_mask=polygon_mask,
            )

            area_results["flow"] = area.metrics["flow"].compute(
                cumulative_count=counter_res.cumulative_count,
                cumulative_counts_by_class=counter_res.cumulative_counts_by_class,
                current_time=current_time,
            )

        if "density" in area.metrics:
            area_results["density"] = area.metrics["density"].compute(
                polygon_mask=polygon_mask,
                det_cls=det_cls,
            )

        if "occupancy" in area.metrics:
            area_results["occupancy"] = area.metrics["occupancy"].compute(
                vicinity_mask=vicinity_mask,
                polygon_mask=polygon_mask,
            )

        if "space_headway" in area.metrics:
            area_results["space_headway"] = area.metrics["space_headway"].compute(
                points=points,
                polygon_mask=polygon_mask,
            )

        if "time_headway" in area.metrics:
            area_results["time_headway"] = area.metrics["time_headway"].compute(
                new_area_crossings
            )

        results[area.area_id] = area_results

    return results


def build_period_state(cfg):
    areas = build_areas(cfg)
    geometry_engine, crossing_estimator = build_runtime_components(areas, cfg)

    tracker = Tracker(
        method=cfg["tracker"]["method"],
        reid_model=cfg["tracker"]["reid_model"],
        classes=cfg["tracker"]["classes"],
        device=cfg["tracker"]["device"],
        half=cfg["tracker"]["half"],
        per_class=cfg["tracker"]["per_class"],
    )

    return areas, geometry_engine, crossing_estimator, tracker


def process_period(
    producer,
    detector,
    cfg,
    fps,
    period_frames,
    period_idx,
    device,
):
    areas, geometry_engine, crossing_estimator, tracker = build_period_state(cfg)

    traffic_metrics_profile = Profile()
    det_profile = Profile(device=device)
    track_profile = Profile(device=device)

    first_frame_idx = None
    last_frame_idx = None
    area_metrics = None
    frames_processed = 0
    source_frames_elapsed = 0
    end_of_stream = False

    with torch.inference_mode():
        while True:
            frame = producer.next_frame()

            if frame is None:
                end_of_stream = True
                break

            if first_frame_idx is None:
                first_frame_idx = frame.read_idx

            print(frame.read_idx)
            last_frame_idx = frame.read_idx
            frames_processed += 1
            source_frames_elapsed = frame.read_idx - first_frame_idx + 1
            current_time = max(
                source_frames_elapsed / fps,
                1.0 / fps,
            )

            with det_profile:
                ready_to_track_array = detector.detect_to_track(frame.data)

            with track_profile:
                tracks = tracker.update(
                    ready_to_track_array,
                    frame.data,
                )

            with traffic_metrics_profile:
                points, bboxes, track_ids, det_cls = extract_tracking_outputs(tracks)

                line_cache, polygon_cache = geometry_engine.compute(
                    points,
                    bboxes,
                )

                crossed_masks, new_crossings = crossing_estimator.update(
                    track_ids,
                    current_time,
                    line_cache,
                    polygon_cache,
                )

                area_metrics = compute_area_metrics(
                    areas=areas,
                    geometry_engine=geometry_engine,
                    points=points,
                    track_ids=track_ids,
                    det_cls=det_cls,
                    current_time=current_time,
                    line_cache=line_cache,
                    polygon_cache=polygon_cache,
                    crossed_masks=crossed_masks,
                    new_crossings=new_crossings,
                )

            if source_frames_elapsed >= period_frames:
                break

    return {
        "period_idx": period_idx,
        "start_frame": first_frame_idx,
        "end_frame": last_frame_idx,
        "frames_processed": frames_processed,
        "source_frames_elapsed": source_frames_elapsed,
        "end_of_stream": end_of_stream,
        "area_metrics": area_metrics,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run smart-sensor traffic metrics over consecutive video periods."
    )
    parser.add_argument("--source", "--video", dest="source", required=True)
    parser.add_argument("--config", default="config/traffic_metrics.yaml")
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--period-mins", type=float, default=5.0)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    fps = args.fps or cfg["general_params"]["fps"]
    period_frames = max(1, int(args.period_mins * 60 * fps))

    detector = build_detector(
        model_name=cfg["detector"]["model_name"],
        conf=cfg["detector"]["confidence"],
    )
    device = detector.predictor.device

    producer = DirectFrameProducer(args.source)
    producer.start()

    period_idx = 0

    try:
        while True:
            result = process_period(
                producer=producer,
                detector=detector,
                cfg=cfg,
                fps=fps,
                period_frames=period_frames,
                period_idx=period_idx,
                device=device,
            )

            if result["frames_processed"] == 0:
                break

            print(result)
            period_idx += 1

            if result["end_of_stream"]:
                break

    except KeyboardInterrupt:
        pass

    finally:
        producer.release()


if __name__ == "__main__":
    main()
