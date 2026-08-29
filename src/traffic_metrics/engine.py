from __future__ import annotations

from typing import Dict

import numpy as np

from geometry.engine import GeometryEngine
from geometry.primitives import Line, Polygon
from runtime.coordinates import CoordinateTransformer
from runtime.continuity import (
    AnalyticsContinuityContext,
    boundary_batch_for_next_period,
    resolve_continuity_policy,
)
from runtime.observations import PeriodObservationBatch
from traffic_metrics.estimators import (
    estimate_density,
    estimate_flow,
    estimate_space_headway,
    estimate_time_headway,
    estimate_time_occupancy,
    normalize_counter_logic,
)
from traffic_metrics.models import (
    PeriodEvents,
    PeriodSpatialResult,
)


class PeriodAnalyticsEngine:
    """Vectorized period-level traffic analytics over a frozen observation batch."""

    def __init__(
        self,
        areas,
        geometry_engine,
        cfg: dict,
        num_classes: int,
    ):
        self.areas = [area for area in areas if area.enable]
        self.geometry_engine = geometry_engine
        self.cfg = cfg
        self.num_classes = int(num_classes)
        if self.num_classes <= 0:
            raise ValueError("detector num_classes must be positive")

        flow_cfg = _metric_config(cfg, "flow")
        counter_cfg = flow_cfg.get("counter", {})
        self.counter_logic = normalize_counter_logic(
            counter_cfg.get("logic", "crossed_or_vicinity")
        )
        self.continuity_policy = resolve_continuity_policy(cfg)
        self.continuity_max_age_seconds = self.continuity_policy.max_age_seconds

        self.coordinate_policy = cfg.get("geometry", {}).get("coordinate_space", "image")
        self.transformer = CoordinateTransformer.from_config(cfg)
        self.world_line_vicinity_threshold = _world_line_vicinity_threshold(cfg)
        self.coordinate_space = self._resolve_coordinate_space()
        if self.coordinate_space == "world":
            self.geometry_engine = self._build_world_geometry_engine(geometry_engine)

        self._line_ids = list(self.geometry_engine._line_ids)
        self._line_id_to_idx = dict(self.geometry_engine._line_id_to_idx)

    def compute_period(
        self,
        batch: PeriodObservationBatch,
        continuity: AnalyticsContinuityContext,
    ) -> Dict[str, object]:
        spatial = self.compute_spatial(batch)
        events = self.derive_crossings(batch, spatial)
        area_metrics = self.compute_metrics(batch, spatial, events, continuity)
        continuity.boundary_batch = boundary_batch_for_next_period(
            batch=batch,
            policy=self.continuity_policy,
        )
        return area_metrics

    def compute_spatial(self, batch: PeriodObservationBatch) -> PeriodSpatialResult:
        if self.coordinate_space == "world":
            points = self.transformer.points_to_world(batch.points)
            bboxes = self.transformer.bboxes_to_world(batch.bboxes)
        else:
            points = batch.points
            bboxes = batch.bboxes

        line_cache, polygon_cache = self.geometry_engine.compute(points, bboxes)

        return PeriodSpatialResult(
            coordinate_space=self.coordinate_space,
            points=points,
            bboxes=bboxes,
            line_cache=line_cache,
            polygon_cache=polygon_cache,
            line_ids=self._line_ids,
            line_id_to_idx=self._line_id_to_idx,
        )

    def derive_crossings(
        self,
        batch: PeriodObservationBatch,
        spatial: PeriodSpatialResult,
    ) -> PeriodEvents:
        signs = spatial.line_cache["sign"]
        distances = spatial.line_cache["distance"]
        line_count, obs_count = signs.shape
        crossed_masks = np.zeros((line_count, obs_count), dtype=bool)
        crossings_by_area = {
            area.area_id: _empty_crossing_arrays()
            for area in self.areas
        }

        if line_count > 0 and obs_count >= 2:
            order = np.lexsort((batch.frame_id, batch.timestamp, batch.track_id))
            prev_idx = order[:-1]
            curr_idx = order[1:]
            same_track = batch.track_id[prev_idx] == batch.track_id[curr_idx]
            curr_active = ~batch.is_context[curr_idx]
            comparable = same_track & curr_active

            if np.any(comparable):
                sign_change = (
                    signs[:, prev_idx] * signs[:, curr_idx] < 0
                ) & comparable[None, :]

                if np.any(sign_change):
                    line_indices, pair_indices = np.where(sign_change)
                    current_observation_indices = curr_idx[pair_indices]
                    crossed_masks[line_indices, current_observation_indices] = True

                    prev_dist = distances[line_indices, prev_idx[pair_indices]]
                    curr_dist = distances[line_indices, current_observation_indices]
                    denom = prev_dist + curr_dist
                    alpha = np.divide(
                        prev_dist,
                        denom,
                        out=np.full(prev_dist.shape, 0.5, dtype=np.float64),
                        where=denom > 1e-6,
                    )
                    timestamps = (
                        batch.timestamp[prev_idx[pair_indices]]
                        + alpha * (
                            batch.timestamp[current_observation_indices]
                            - batch.timestamp[prev_idx[pair_indices]]
                        )
                    )

                    for area in self.areas:
                        if area.flow_line is None:
                            continue

                        line_idx = spatial.line_id_to_idx.get(area.flow_line.line_id)
                        if line_idx is None:
                            continue

                        area_event_mask = line_indices == line_idx
                        if area.zone is not None:
                            polygon_mask = spatial.polygon_cache.get(area.zone.polygon_id)
                            if polygon_mask is None:
                                area_event_mask &= False
                            else:
                                area_event_mask &= polygon_mask[prev_idx[pair_indices]]

                        selected = np.flatnonzero(area_event_mask)
                        if selected.size == 0:
                            continue

                        event_order = np.argsort(timestamps[selected], kind="stable")
                        selected = selected[event_order]
                        crossings_by_area[area.area_id] = {
                            "timestamp": timestamps[selected].astype(np.float64, copy=False),
                            "track_id": batch.track_id[current_observation_indices[selected]].astype(
                                np.int64,
                                copy=False,
                            ),
                            "class_id": batch.class_id[current_observation_indices[selected]].astype(
                                np.int32,
                                copy=False,
                            ),
                            "line_id": np.asarray(
                                [area.flow_line.line_id] * selected.size,
                                dtype=object,
                            ),
                        }

        self._append_ttl_fallback_crossings(batch, spatial, crossings_by_area)

        return PeriodEvents(
            crossed_masks=crossed_masks,
            crossings_by_area=crossings_by_area,
        )

    def _append_ttl_fallback_crossings(
        self,
        batch: PeriodObservationBatch,
        spatial: PeriodSpatialResult,
        crossings_by_area: Dict[str, Dict[str, np.ndarray]],
    ) -> None:
        if batch.frame_count == 0 or batch.observation_count == 0 or self.continuity_max_age_seconds <= 0:
            return

        end_time = float(batch.period_timestamps[-1])

        for area in self.areas:
            if area.flow_line is None:
                continue

            line_idx = spatial.line_id_to_idx.get(area.flow_line.line_id)
            if line_idx is None:
                continue

            area_mask = np.ones(batch.observation_count, dtype=bool)
            if area.zone is not None:
                polygon_mask = spatial.polygon_cache.get(area.zone.polygon_id)
                if polygon_mask is None:
                    continue
                area_mask &= polygon_mask

            vicinity_mask = spatial.line_cache["vicinity_mask"][line_idx] & area_mask
            candidate_tracks = np.unique(batch.track_id[vicinity_mask])
            if candidate_tracks.size == 0:
                continue

            area_indices = np.flatnonzero(area_mask)
            order = np.lexsort((
                batch.frame_id[area_indices],
                batch.timestamp[area_indices],
                batch.track_id[area_indices],
            ))
            ordered_idx = area_indices[order]
            ordered_track_ids = batch.track_id[ordered_idx]

            keep_latest = np.ones((ordered_idx.shape[0],), dtype=bool)
            keep_latest[:-1] = ordered_track_ids[:-1] != ordered_track_ids[1:]
            latest_idx = ordered_idx[keep_latest]
            latest_track_ids = batch.track_id[latest_idx]

            selection = np.isin(latest_track_ids, candidate_tracks)
            existing_tracks = crossings_by_area[area.area_id]["track_id"]
            if existing_tracks.size:
                selection &= ~np.isin(latest_track_ids, existing_tracks)
            selection &= (end_time - batch.timestamp[latest_idx]) > self.continuity_max_age_seconds

            selected_idx = latest_idx[selection]
            if selected_idx.size:
                additions = {
                    "timestamp": batch.timestamp[selected_idx].astype(np.float64, copy=False),
                    "track_id": batch.track_id[selected_idx].astype(np.int64, copy=False),
                    "class_id": batch.class_id[selected_idx].astype(np.int32, copy=False),
                    "line_id": np.asarray(
                        [area.flow_line.line_id] * selected_idx.size,
                        dtype=object,
                    ),
                }
                crossings_by_area[area.area_id] = _merge_crossing_arrays(
                    crossings_by_area[area.area_id],
                    additions,
                )

    def compute_metrics(
        self,
        batch: PeriodObservationBatch,
        spatial: PeriodSpatialResult,
        events: PeriodEvents,
        continuity: AnalyticsContinuityContext,
    ) -> Dict[str, object]:
        results = {}
        active_mask = batch.active_mask

        for area in self.areas:
            area_results = {}
            polygon_mask = None
            if area.zone is not None:
                polygon_mask = spatial.polygon_cache.get(area.zone.polygon_id)

            line_idx = None
            vicinity_mask = None
            crossed_mask = None
            if area.flow_line is not None:
                line_idx = spatial.line_id_to_idx[area.flow_line.line_id]
                vicinity_mask = spatial.line_cache["vicinity_mask"][line_idx]
                crossed_mask = events.crossed_masks[line_idx]

            if "flow" in area.eligible_metrics:
                area_results["flow"] = estimate_flow(
                    batch=batch,
                    active_mask=active_mask,
                    crossed_mask=crossed_mask,
                    vicinity_mask=vicinity_mask,
                    polygon_mask=polygon_mask,
                    counted_state=continuity.counted_ids_by_area.setdefault(area.area_id, {}),
                    counter_logic=self.counter_logic,
                    continuity_max_age_seconds=self.continuity_max_age_seconds,
                    num_classes=self.num_classes,
                )

            if "density" in area.eligible_metrics:
                area_results["density"] = estimate_density(
                    area=area,
                    batch=batch,
                    active_mask=active_mask,
                    polygon_mask=polygon_mask,
                    num_classes=self.num_classes,
                )

            if "time_occupancy" in area.eligible_metrics:
                area_results["time_occupancy"] = estimate_time_occupancy(
                    batch=batch,
                    active_mask=active_mask,
                    vicinity_mask=vicinity_mask,
                    polygon_mask=polygon_mask,
                )

            if "space_headway" in area.eligible_metrics:
                area_results["space_headway"] = estimate_space_headway(
                    area=area,
                    batch=batch,
                    points=spatial.points,
                    direction=self.geometry_engine.line_normal(area.flow_line.line_id),
                    active_mask=active_mask,
                    polygon_mask=polygon_mask,
                )

            if "time_headway" in area.eligible_metrics:
                event_payload = events.crossings_by_area.get(area.area_id, _empty_crossing_arrays())
                timestamps = event_payload["timestamp"]
                area_results["time_headway"] = estimate_time_headway(
                    timestamps=timestamps,
                    previous_timestamp=continuity.last_crossing_timestamp_by_area.get(area.area_id),
                )
                if timestamps.size:
                    continuity.last_crossing_timestamp_by_area[area.area_id] = float(timestamps[-1])

            results[area.area_id] = area_results

        return results

    def _resolve_coordinate_space(self) -> str:
        policy = str(self.coordinate_policy).lower()
        if policy == "image":
            return "image"
        if policy == "world":
            if self.transformer is None:
                raise ValueError("geometry.coordinate_space='world' requires homography.enabled=true")
            return "world"
        if policy == "auto":
            if self.transformer is not None and self._all_world_vicinity_available():
                return "world"
            return "image"
        raise ValueError("geometry.coordinate_space must be one of: image, world, auto")

    def _all_world_vicinity_available(self) -> bool:
        for area in self.areas:
            if area.flow_line is not None and self.world_line_vicinity_threshold is None:
                return False
        return True

    def _build_world_geometry_engine(self, image_geometry_engine) -> GeometryEngine:
        if not self._all_world_vicinity_available():
            raise ValueError(
                "world-space geometry requires every active flow line to define vicinity.world_m"
            )

        lines = {}
        for line_id in image_geometry_engine._line_ids:
            line = image_geometry_engine._lines[line_id]
            lines[line_id] = Line(
                line_id=line.line_id,
                points=_as_point_tuples(self.transformer.line_to_world(line.points)),
            )

        polygons = {}
        for polygon_id, polygon in image_geometry_engine._polygons.items():
            polygons[polygon_id] = Polygon(
                polygon_id=polygon_id,
                points=_as_point_tuples(self.transformer.polygon_to_world(polygon.points)),
            )

        threshold = 0.0 if self.world_line_vicinity_threshold is None else float(self.world_line_vicinity_threshold)
        line_thresholds = {
            line_id: threshold
            for line_id in lines
        }

        return GeometryEngine(
            lines=lines,
            polygons=polygons,
            polygon_mode=image_geometry_engine._polygon_mode,
            coordinate_space="world",
            line_vicinity_thresholds=line_thresholds,
        )


def _metric_config(cfg: dict, metric_name: str) -> dict:
    return cfg.get("metrics", {}).get(metric_name, {})


def _world_line_vicinity_threshold(cfg: dict) -> float | None:
    value = cfg.get("geometry", {}).get("vicinity", {}).get("world_m")
    return None if value is None else float(value)


def _as_point_tuples(points: np.ndarray):
    return [
        (float(x), float(y))
        for x, y in np.asarray(points, dtype=np.float64)
    ]


def _empty_crossing_arrays() -> Dict[str, np.ndarray]:
    return {
        "timestamp": np.empty((0,), dtype=np.float64),
        "track_id": np.empty((0,), dtype=np.int64),
        "class_id": np.empty((0,), dtype=np.int32),
        "line_id": np.empty((0,), dtype=object),
    }


def _merge_crossing_arrays(
    existing: Dict[str, np.ndarray],
    additions: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    if existing["timestamp"].size == 0:
        merged = additions
    else:
        merged = {
            key: np.concatenate((existing[key], additions[key]))
            for key in existing
        }

    order = np.argsort(merged["timestamp"], kind="stable")
    return {
        key: values[order]
        for key, values in merged.items()
    }
