from __future__ import annotations

import numpy as np

from traffic_metrics.models import (
    DensityResult,
    FlowResult,
    OccupancyResult,
    SpaceHeadwayResult,
    TimeHeadwayResult,
)


def normalize_counter_logic(value: str) -> str:
    aliases = {
        "counter_2": "crossed",
        "counter_3": "vicinity",
        "counter_4": "crossed_and_vicinity",
        "counter_5": "crossed_or_vicinity",
        "crossed": "crossed",
        "vicinity": "vicinity",
        "crossed_and_vicinity": "crossed_and_vicinity",
        "crossed_or_vicinity": "crossed_or_vicinity",
    }
    key = str(value).lower()
    if key not in aliases:
        raise ValueError(f"Unknown counter logic: {value}")
    return aliases[key]


def estimate_flow(
    *,
    batch,
    active_mask: np.ndarray,
    crossed_mask: np.ndarray | None,
    vicinity_mask: np.ndarray | None,
    polygon_mask: np.ndarray | None,
    counted_state: dict[int, float],
    counter_logic: str,
    continuity_max_age_seconds: float,
    num_classes: int,
) -> FlowResult:
    if crossed_mask is None and vicinity_mask is None:
        return FlowResult(
            average_flow=0.0,
            average_flow_by_class=np.zeros(num_classes, dtype=np.float32),
        )

    counter_mask = select_counter_mask(crossed_mask, vicinity_mask, counter_logic)
    mask = counter_mask & active_mask
    if polygon_mask is not None:
        mask &= polygon_mask

    selected_idx = np.flatnonzero(mask)
    counts_by_class = np.zeros(num_classes, dtype=np.int64)
    if selected_idx.size:
        end_time = batch.period_timestamps[-1] if batch.frame_count else 0.0
        counted_classes = count_unique_with_ttl(
            batch=batch,
            selected_idx=selected_idx,
            state=counted_state,
            ttl_seconds=continuity_max_age_seconds,
        )
        if counted_classes.size:
            counts_by_class = np.bincount(
                counted_classes,
                minlength=num_classes,
            ).astype(np.int64, copy=False)
        cleanup_counted_ids(counted_state, end_time, continuity_max_age_seconds)

    duration = batch.duration_seconds
    if duration <= 0:
        return FlowResult(
            average_flow=0.0,
            average_flow_by_class=np.zeros(num_classes, dtype=np.float32),
        )

    cumulative_count = int(counts_by_class.sum())
    return FlowResult(
        average_flow=float(cumulative_count / duration),
        average_flow_by_class=(counts_by_class / duration).astype(np.float32),
    )


def estimate_density(
    *,
    area,
    batch,
    active_mask: np.ndarray,
    polygon_mask: np.ndarray | None,
    num_classes: int,
) -> DensityResult:
    empty = np.zeros(num_classes, dtype=np.float32)
    if polygon_mask is None or area.zone.distance_meters is None or area.zone.distance_meters <= 0:
        return DensityResult(
            average_density=0.0,
            average_density_by_class=empty.copy(),
            current_density=0.0,
            current_density_by_class=empty.copy(),
        )

    counts_by_frame_class = counts_by_frame_class_for_batch(
        batch=batch,
        mask=active_mask & polygon_mask,
        num_classes=num_classes,
    )
    densities = counts_by_frame_class / float(area.zone.distance_meters)

    if densities.shape[0] == 0:
        return DensityResult(
            average_density=0.0,
            average_density_by_class=empty.copy(),
            current_density=0.0,
            current_density_by_class=empty.copy(),
        )

    average_by_class = densities.mean(axis=0).astype(np.float32)
    current_by_class = densities[-1].astype(np.float32)
    return DensityResult(
        average_density=float(average_by_class.sum()),
        average_density_by_class=average_by_class.copy(),
        current_density=float(current_by_class.sum()),
        current_density_by_class=current_by_class.copy(),
    )


def estimate_time_occupancy(
    *,
    batch,
    active_mask: np.ndarray,
    vicinity_mask: np.ndarray | None,
    polygon_mask: np.ndarray | None,
) -> OccupancyResult:
    if batch.frame_count == 0 or vicinity_mask is None or polygon_mask is None:
        return OccupancyResult(average_occupancy=0.0, current_occupancy=0.0)

    mask = active_mask & vicinity_mask & polygon_mask
    occupied = np.zeros(batch.frame_count, dtype=bool)
    selected_idx = np.flatnonzero(mask)
    if selected_idx.size:
        frame_idx = np.searchsorted(batch.period_frame_ids, batch.frame_id[selected_idx])
        occupied[np.unique(frame_idx)] = True

    return OccupancyResult(
        average_occupancy=float(occupied.mean()),
        current_occupancy=float(occupied[-1]),
    )


def estimate_space_headway(
    *,
    area,
    batch,
    points: np.ndarray,
    direction: np.ndarray | None,
    active_mask: np.ndarray,
    polygon_mask: np.ndarray | None,
) -> SpaceHeadwayResult:
    if polygon_mask is None or area.flow_line is None or batch.frame_count == 0:
        return SpaceHeadwayResult()
    if direction is None:
        return SpaceHeadwayResult()

    mask = active_mask & polygon_mask
    selected_idx = np.flatnonzero(mask)
    if selected_idx.size < 2:
        return SpaceHeadwayResult()

    selected_frame_ids = batch.frame_id[selected_idx]
    selected_points = points[selected_idx]
    headways = []
    current_space_headway = None

    for frame_id in batch.period_frame_ids:
        frame_points = selected_points[selected_frame_ids == frame_id]
        if frame_points.shape[0] < 2:
            current_space_headway = None
            continue

        projected = frame_points @ direction
        diffs = np.diff(np.sort(projected))
        if diffs.size == 0:
            current_space_headway = None
            continue

        current_space_headway = float(diffs.mean())
        headways.append(current_space_headway)

    if not headways:
        return SpaceHeadwayResult(
            average_space_headway=None,
            current_space_headway=current_space_headway,
        )

    return SpaceHeadwayResult(
        average_space_headway=float(np.mean(headways)),
        current_space_headway=current_space_headway,
    )


def estimate_time_headway(
    *,
    timestamps: np.ndarray,
    previous_timestamp: float | None,
) -> TimeHeadwayResult:
    if timestamps.size == 0:
        return TimeHeadwayResult(
            average_time_headway=None,
            current_time_headway=None,
        )

    if previous_timestamp is None:
        headways = np.diff(timestamps)
    else:
        headways = np.diff(np.concatenate(([previous_timestamp], timestamps)))

    if headways.size == 0:
        return TimeHeadwayResult(
            average_time_headway=None,
            current_time_headway=None,
        )

    return TimeHeadwayResult(
        average_time_headway=float(headways.mean()),
        current_time_headway=float(headways[-1]),
    )


def select_counter_mask(
    crossed_mask: np.ndarray | None,
    vicinity_mask: np.ndarray | None,
    counter_logic: str,
) -> np.ndarray:
    if crossed_mask is None:
        crossed_mask = np.zeros_like(vicinity_mask, dtype=bool)
    if vicinity_mask is None:
        vicinity_mask = np.zeros_like(crossed_mask, dtype=bool)

    if counter_logic == "crossed":
        return crossed_mask
    if counter_logic == "vicinity":
        return vicinity_mask
    if counter_logic == "crossed_and_vicinity":
        return crossed_mask & vicinity_mask
    return crossed_mask | vicinity_mask


def count_unique_with_ttl(
    *,
    batch,
    selected_idx: np.ndarray,
    state: dict[int, float],
    ttl_seconds: float,
) -> np.ndarray:
    order = np.lexsort((batch.timestamp[selected_idx], batch.track_id[selected_idx]))
    ordered_idx = selected_idx[order]
    counted_classes = []

    for idx in ordered_idx:
        track_id = int(batch.track_id[idx])
        timestamp = float(batch.timestamp[idx])
        last_seen = state.get(track_id)
        if last_seen is None or (timestamp - last_seen) > ttl_seconds:
            counted_classes.append(int(batch.class_id[idx]))
            state[track_id] = timestamp

    return np.asarray(counted_classes, dtype=np.int32)


def counts_by_frame_class_for_batch(
    *,
    batch,
    mask: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    counts = np.zeros((batch.frame_count, num_classes), dtype=np.int64)
    selected_idx = np.flatnonzero(mask)
    if selected_idx.size == 0 or batch.frame_count == 0:
        return counts

    frame_idx = np.searchsorted(batch.period_frame_ids, batch.frame_id[selected_idx])
    class_ids = batch.class_id[selected_idx]
    np.add.at(counts, (frame_idx, class_ids), 1)
    return counts


def cleanup_counted_ids(
    state: dict[int, float],
    current_time: float,
    ttl_seconds: float,
) -> None:
    expired = [
        track_id for track_id, last_seen in state.items()
        if current_time - last_seen > ttl_seconds
    ]
    for track_id in expired:
        del state[track_id]
