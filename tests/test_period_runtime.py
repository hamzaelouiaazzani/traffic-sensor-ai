import tempfile
import unittest
from pathlib import Path

import numpy as np

from config.load_build import (
    build_areas,
    build_geometry_engine,
    get_input_fps,
    load_config,
)
from geometry.homography import Homography, compute_homography, save_calibration_yaml
from geometry.primitives import GeometryEngine
from runtime.observations import (
    PeriodObservationBatch,
    PeriodObservationBuffer,
    PeriodObservationBufferOverflow,
)
from runtime.continuity import (
    AnalyticsContinuityContext,
    ContinuityPolicy,
    boundary_batch_for_next_period,
)
from runtime.periods import (
    resolve_max_obs_per_frame,
    resolve_period_policy,
    resolve_period_seconds,
)
from runtime.stage1 import process_stage1_period
from runtime.timing import FrameTiming, TimingPolicy
from traffic_metrics.engine import PeriodAnalyticsEngine
from video_io.frame_producer import Frame


def _bbox_from_point(x, y):
    return [x - 1.0, y - 2.0, x + 1.0, y]


def _batch(rows, period_frames=None, period_timestamps=None, fps=1.0):
    if rows:
        arr = np.asarray(rows, dtype=object)
        timestamp = arr[:, 0].astype(np.float64)
        frame_id = arr[:, 1].astype(np.int64)
        track_id = arr[:, 2].astype(np.int64)
        class_id = arr[:, 3].astype(np.int32)
        points = arr[:, 4:6].astype(np.float32)
        is_context = arr[:, 6].astype(bool) if arr.shape[1] > 6 else np.zeros(len(rows), dtype=bool)
        bboxes = np.asarray([_bbox_from_point(x, y) for x, y in points], dtype=np.float32)
    else:
        timestamp = np.empty((0,), dtype=np.float64)
        frame_id = np.empty((0,), dtype=np.int64)
        track_id = np.empty((0,), dtype=np.int64)
        class_id = np.empty((0,), dtype=np.int32)
        points = np.empty((0, 2), dtype=np.float32)
        bboxes = np.empty((0, 4), dtype=np.float32)
        is_context = np.empty((0,), dtype=bool)

    if period_frames is None:
        active_frames = frame_id[~is_context]
        period_frames = sorted(set(active_frames.tolist()))

    if period_timestamps is None:
        by_frame = {}
        for frame, ts, context in zip(frame_id, timestamp, is_context):
            if not context and int(frame) not in by_frame:
                by_frame[int(frame)] = float(ts)
        period_timestamps = [by_frame[int(frame)] for frame in period_frames]

    return PeriodObservationBatch(
        timestamp=timestamp,
        frame_id=frame_id,
        track_id=track_id,
        class_id=class_id,
        points=points,
        bboxes=bboxes,
        is_context=is_context,
        period_frame_ids=np.asarray(period_frames, dtype=np.int64),
        period_timestamps=np.asarray(period_timestamps, dtype=np.float64),
        period_idx=0,
        timing_mode="frame",
        fps=fps,
    )


def _metric(enabled, scope, coord, **extra):
    payload = {
        "enabled": enabled,
        "spatial_scope": scope,
        "coordinate_requirement": coord,
    }
    payload.update(extra)
    return payload


def _config(
    *,
    area_type="lane",
    area_enabled=True,
    line=True,
    zone=True,
    coordinate_space="image",
    homography_enabled=False,
    calibration_file=None,
    image_vicinity=0.02,
    world_vicinity=0.2,
    flow_enabled=True,
    density_enabled=False,
    time_occupancy_enabled=True,
    space_headway_enabled=False,
    time_headway_enabled=True,
    speed_enabled=True,
    continuity_enabled=True,
    continuity_seconds=30.0,
    input_fps=1.0,
):
    geometry = {
        "coordinate_space": coordinate_space,
        "frame_size_reference": 100,
        "polygon_membership_mode": "center",
        "vicinity": {
            "image": image_vicinity,
            "world_m": world_vicinity,
        },
        "lines": [],
        "areas": [
            {
                "area_id": "area_1",
                "area_type": area_type,
                "enabled": area_enabled,
            }
        ],
    }

    if line:
        geometry["lines"].append(
            {
                "line_id": "line_a",
                "points": [[0, 50], [100, 50]],
            }
        )
        geometry["areas"][0]["flow_line_id"] = "line_a"

    if zone:
        geometry["areas"][0]["zone"] = {
            "points": [[0, 0], [100, 0], [100, 100], [0, 100]],
            "distance_meters": 100.0,
        }

    return {
        "input": {"source": "synthetic.mp4", "fps": input_fps},
        "timing": {"mode": "frame"},
        "analytics": {"period_seconds": 300, "max_obs_per_frame": 50},
        "continuity": {"enabled": continuity_enabled, "max_age_seconds": continuity_seconds},
        "homography": {
            "enabled": homography_enabled,
            "calibration_file": calibration_file,
        },
        "detector": {
            "model_name": "synthetic",
            "confidence": 0.5,
        },
        "tracker": {},
        "geometry": geometry,
        "metrics": {
            "flow": _metric(
                flow_enabled,
                "any_area",
                "image_or_world",
                counter={"logic": "crossed"},
            ),
            "density": _metric(density_enabled, "any_area", "world_required"),
            "space_occupancy": _metric(True, "any_area", "world_required"),
            "time_occupancy": _metric(time_occupancy_enabled, "any_area", "image_or_world"),
            "space_headway": _metric(space_headway_enabled, "lane_only", "world_required"),
            "time_headway": _metric(time_headway_enabled, "lane_only", "image_or_world"),
            "speed": _metric(speed_enabled, "any_area", "world_required"),
        },
    }


def _runtime(cfg, num_classes=4):
    areas = build_areas(cfg, num_classes=num_classes)
    geometry_engine = build_geometry_engine(areas, cfg)
    analytics = PeriodAnalyticsEngine(areas, geometry_engine, cfg, num_classes=num_classes)
    return areas, geometry_engine, analytics, AnalyticsContinuityContext()


class _DummyProducer:
    def __init__(self, frames):
        self.frames = list(frames)
        self.index = 0

    def next_frame(self):
        if self.index >= len(self.frames):
            return None
        frame = self.frames[self.index]
        self.index += 1
        return frame


class _DummyDetector:
    def detect_to_track(self, frame):
        return np.empty((0, 6), dtype=np.float32)


class _StatefulDummyTracker:
    def __init__(self):
        self.calls = 0

    def update(self, detections, frame):
        self.calls += 1
        return np.asarray(
            [[0.0, 0.0, 2.0, 2.0, 100 + self.calls, 1.0, 0]],
            dtype=np.float32,
        )


class PeriodRuntimeTest(unittest.TestCase):
    def test_current_yaml_loads_successfully(self):
        cfg = load_config("config/traffic_metrics.yaml")
        areas = build_areas(cfg, num_classes=4)
        geometry_engine = build_geometry_engine(areas, cfg)
        analytics = PeriodAnalyticsEngine(areas, geometry_engine, cfg, num_classes=4)

        self.assertEqual(len(areas), 5)
        self.assertEqual(len(geometry_engine._line_ids), 1)
        self.assertEqual(analytics.coordinate_space, "image")

    def test_input_fps_is_used_by_frame_timing(self):
        cfg = _config(input_fps=10.0)
        self.assertEqual(get_input_fps(cfg), 10.0)

        policy = TimingPolicy.from_config(cfg)
        start = policy.resolve(Frame(data=None, timestamp=1000.0, read_idx=0))
        later = policy.resolve(Frame(data=None, timestamp=9999.0, read_idx=10))

        self.assertAlmostEqual(later.timestamp, 1.0)
        self.assertAlmostEqual(policy.period_elapsed_seconds(start, later), 1.1)

    def test_period_policy_derives_max_observations_from_fps_and_per_frame_limit(self):
        cfg = _config(input_fps=30.0)
        cfg["analytics"]["period_seconds"] = 2
        cfg["analytics"]["max_obs_per_frame"] = 20

        policy = resolve_period_policy(cfg, fps=15.0)

        self.assertEqual(policy.period_seconds, 2.0)
        self.assertEqual(policy.max_obs_per_frame, 20)
        self.assertEqual(policy.max_observations, 600)

    def test_period_seconds_is_positive_and_clamped_to_one_hour(self):
        cfg = _config()
        cfg["analytics"]["period_seconds"] = 7200

        self.assertEqual(resolve_period_seconds(cfg), 3600.0)

        cfg["analytics"]["period_seconds"] = 0
        with self.assertRaises(ValueError):
            resolve_period_seconds(cfg)

    def test_max_obs_per_frame_defaults_and_clamps(self):
        cfg = _config()
        del cfg["analytics"]["max_obs_per_frame"]
        self.assertEqual(resolve_max_obs_per_frame(cfg), 50)

        cfg["analytics"]["max_obs_per_frame"] = 5
        self.assertEqual(resolve_max_obs_per_frame(cfg), 10)

        cfg["analytics"]["max_obs_per_frame"] = 500
        self.assertEqual(resolve_max_obs_per_frame(cfg), 50)

    def test_timing_mode_unix_uses_acquisition_timestamp(self):
        cfg = _config(input_fps=10.0)
        cfg["timing"]["mode"] = "unix"
        policy = TimingPolicy.from_config(cfg)
        start = policy.resolve(Frame(data=None, timestamp=1000.0, read_idx=0))
        later = policy.resolve(Frame(data=None, timestamp=1005.25, read_idx=50))

        self.assertAlmostEqual(later.timestamp, 1005.25)
        self.assertAlmostEqual(later.elapsed_seconds, 5.25)
        self.assertAlmostEqual(policy.period_elapsed_seconds(start, later), 5.25)

    def test_global_geometry_vicinity_is_applied_to_all_lines(self):
        cfg = _config(image_vicinity=0.05)
        cfg["geometry"]["lines"].append(
            {
                "line_id": "line_b",
                "points": [[0, 20], [100, 20]],
            }
        )
        cfg["geometry"]["areas"].append(
            {
                "area_id": "area_2",
                "area_type": "lane",
                "enabled": True,
                "flow_line_id": "line_b",
                "zone": {
                    "points": [[0, 0], [100, 0], [100, 100], [0, 100]],
                    "distance_meters": 100.0,
                },
            }
        )

        areas, geometry_engine, _, _ = _runtime(cfg)

        self.assertEqual(len(areas), 2)
        self.assertEqual(set(geometry_engine._line_ids), {"line_a", "line_b"})
        np.testing.assert_allclose(geometry_engine._thresh, [5.0, 5.0])

    def test_geometry_lines_and_areas_load_from_geometry_section(self):
        cfg = _config()
        areas = build_areas(cfg, num_classes=4)

        self.assertEqual(areas[0].area_id, "area_1")
        self.assertEqual(areas[0].flow_line.line_id, "line_a")

    def test_area_type_validation(self):
        cfg = _config(area_type="parking")

        with self.assertRaises(ValueError):
            build_areas(cfg, num_classes=4)

    def test_lane_only_metrics_are_skipped_for_non_lane_areas(self):
        cfg = _config(area_type="entire", time_headway_enabled=True)
        areas = build_areas(cfg, num_classes=4)

        self.assertNotIn("time_headway", areas[0].eligible_metrics)
        self.assertIn("lane_only", areas[0].ineligible_metrics["time_headway"])

    def test_any_area_metrics_are_eligible_for_all_supported_area_types(self):
        for area_type in ("lane", "direction", "mixed", "entire"):
            cfg = _config(area_type=area_type, time_occupancy_enabled=True)
            areas = build_areas(cfg, num_classes=4)
            self.assertIn("flow", areas[0].eligible_metrics)
            self.assertIn("time_occupancy", areas[0].eligible_metrics)

    def test_world_required_metrics_are_unavailable_without_homography(self):
        cfg = _config(density_enabled=True, space_headway_enabled=True)
        areas = build_areas(cfg, num_classes=4)

        self.assertNotIn("density", areas[0].eligible_metrics)
        self.assertNotIn("space_headway", areas[0].eligible_metrics)
        self.assertEqual(areas[0].ineligible_metrics["density"], "world coordinates unavailable")

    def test_image_or_world_metrics_remain_available_without_homography(self):
        cfg = _config()
        areas = build_areas(cfg, num_classes=4)

        self.assertIn("flow", areas[0].eligible_metrics)
        self.assertIn("time_occupancy", areas[0].eligible_metrics)
        self.assertIn("time_headway", areas[0].eligible_metrics)

    def test_homography_enabled_world_required_eligibility_works(self):
        calibration = compute_homography(
            image_points=[(0, 0), (100, 0), (100, 100), (0, 100)],
            world_points=[(0, 0), (10, 0), (10, 10), (0, 10)],
            method="ls",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            calibration_path = Path(temp_dir) / "calibration.yaml"
            save_calibration_yaml(calibration, calibration_path)
            cfg = _config(
                coordinate_space="world",
                homography_enabled=True,
                calibration_file=str(calibration_path),
                density_enabled=True,
                space_headway_enabled=True,
            )
            areas, _, analytics, continuity = _runtime(cfg)

            self.assertIn("density", areas[0].eligible_metrics)
            self.assertIn("space_headway", areas[0].eligible_metrics)

            batch = _batch(
                [
                    (0.0, 0, 1, 0, 50.0, 49.0),
                    (1.0, 1, 1, 0, 50.0, 51.0),
                ]
            )
            result = analytics.compute_period(batch, continuity)["area_1"]
            self.assertIn("density", result)

    def test_continuity_max_age_controls_boundary_context_and_duplicate_counts(self):
        cfg = _config(continuity_seconds=5.0)
        _, _, analytics, continuity = _runtime(cfg)
        continuity.counted_ids_by_area["area_1"] = {9: 0.0}

        batch = _batch(
            [(3.0, 3, 9, 0, 50.0, 49.0)],
            period_frames=[3, 7],
            period_timestamps=[3.0, 7.0],
        )
        result = analytics.compute_period(batch, continuity)["area_1"]["flow"]

        self.assertAlmostEqual(result.average_flow, 0.0)
        self.assertEqual(continuity.boundary_batch.observation_count, 1)
        self.assertEqual(analytics.continuity_max_age_seconds, 5.0)

    def test_continuity_disabled_does_not_carry_boundary_observations(self):
        cfg = _config(continuity_enabled=False, continuity_seconds=5.0)
        _, _, analytics, continuity = _runtime(cfg)
        batch = _batch(
            [(3.0, 3, 9, 0, 50.0, 49.0)],
            period_frames=[3, 7],
            period_timestamps=[3.0, 7.0],
        )

        analytics.compute_period(batch, continuity)

        self.assertIsNone(continuity.boundary_batch)
        self.assertIsNone(boundary_batch_for_next_period(batch, analytics.continuity_policy))

    def test_stage1_does_not_inject_context_when_continuity_is_disabled(self):
        tracker = _StatefulDummyTracker()
        timing_policy = TimingPolicy(mode="frame", fps=1.0)
        producer = _DummyProducer(
            [
                Frame(data=np.zeros((2, 2, 3), dtype=np.uint8), timestamp=10.0, read_idx=0),
                Frame(data=np.zeros((2, 2, 3), dtype=np.uint8), timestamp=11.0, read_idx=1),
            ]
        )
        disabled = ContinuityPolicy(enabled=False, max_age_seconds=30.0)
        carried_context = _batch([(0.0, 0, 7, 0, 1.0, 2.0)])

        first = process_stage1_period(
            producer=producer,
            detector=_DummyDetector(),
            tracker=tracker,
            timing_policy=timing_policy,
            period_seconds=1.0,
            max_observations=10,
            period_idx=0,
            continuity_policy=disabled,
            boundary_context=carried_context,
        )
        second = process_stage1_period(
            producer=producer,
            detector=_DummyDetector(),
            tracker=tracker,
            timing_policy=timing_policy,
            period_seconds=1.0,
            max_observations=10,
            period_idx=1,
            continuity_policy=disabled,
            boundary_context=first.batch,
        )

        self.assertEqual(tracker.calls, 2)
        self.assertEqual(first.batch.observation_count, 1)
        self.assertEqual(second.batch.observation_count, 1)
        self.assertFalse(np.any(first.batch.is_context))
        self.assertFalse(np.any(second.batch.is_context))
        self.assertEqual(int(second.batch.track_id[0]), 102)

    def test_per_area_metric_lists_are_not_expected(self):
        cfg = _config()
        self.assertNotIn("metrics", cfg["geometry"]["areas"][0])
        areas = build_areas(cfg, num_classes=4)

        self.assertIn("flow", areas[0].eligible_metrics)

    def test_detector_num_classes_is_used_for_metric_vectors(self):
        cfg = _config()
        cfg["metrics"]["defaults"] = {"num_classes": 99}
        _, _, analytics, continuity = _runtime(cfg, num_classes=2)
        batch = _batch(
            [
                (0.0, 0, 1, 0, 50.0, 49.0),
                (1.0, 1, 1, 0, 50.0, 51.0),
            ]
        )

        result = analytics.compute_period(batch, continuity)["area_1"]["flow"]

        self.assertEqual(result.average_flow_by_class.shape, (2,))

    def test_disabled_metrics_and_areas_are_not_processed(self):
        metric_cfg = _config(flow_enabled=False)
        metric_areas = build_areas(metric_cfg, num_classes=4)
        self.assertNotIn("flow", metric_areas[0].eligible_metrics)

        area_cfg = _config(area_enabled=False)
        self.assertEqual(build_areas(area_cfg, num_classes=4), [])

    def test_batch_homography_matches_pointwise_projection(self):
        calibration = compute_homography(
            image_points=[(0, 0), (100, 0), (100, 100), (0, 100)],
            world_points=[(0, 0), (10, 0), (10, 10), (0, 10)],
            method="ls",
        )
        homography = Homography(calibration)
        points = np.asarray([[5, 5], [50, 75], [90, 10]], dtype=np.float32)

        batched = homography.project_pixels_to_world(points)
        pointwise = np.vstack([homography.project_pixels_to_world(point)[0] for point in points])

        np.testing.assert_allclose(batched, pointwise, atol=1e-5)

    def test_period_spatial_image_mode_matches_existing_geometry_engine(self):
        cfg = _config()
        _, geometry_engine, analytics, _ = _runtime(cfg)
        batch = _batch(
            [
                (0.0, 0, 1, 0, 40.0, 49.0),
                (0.0, 0, 2, 1, 60.0, 47.0),
                (1.0, 1, 1, 0, 40.0, 51.0),
                (1.0, 1, 2, 1, 60.0, 47.0),
            ]
        )

        spatial = analytics.compute_spatial(batch)
        expected_line_cache, expected_polygon_cache = geometry_engine.compute(batch.points, batch.bboxes)

        for key in expected_line_cache:
            np.testing.assert_array_equal(spatial.line_cache[key], expected_line_cache[key])
        for key in expected_polygon_cache:
            np.testing.assert_array_equal(spatial.polygon_cache[key], expected_polygon_cache[key])

    def test_period_spatial_world_mode_uses_geometry_engine_contract(self):
        calibration = compute_homography(
            image_points=[(0, 0), (100, 0), (100, 100), (0, 100)],
            world_points=[(0, 0), (10, 0), (10, 10), (0, 10)],
            method="ls",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            calibration_path = Path(temp_dir) / "calibration.yaml"
            save_calibration_yaml(calibration, calibration_path)
            cfg = _config(
                coordinate_space="world",
                homography_enabled=True,
                calibration_file=str(calibration_path),
                density_enabled=True,
            )
            _, image_geometry_engine, analytics, _ = _runtime(cfg)
            batch = _batch(
                [
                    (0.0, 0, 1, 0, 50.0, 49.0),
                    (1.0, 1, 1, 0, 50.0, 51.0),
                ]
            )

            spatial = analytics.compute_spatial(batch)
            expected_points = analytics.transformer.points_to_world(batch.points)
            expected_bboxes = analytics.transformer.bboxes_to_world(batch.bboxes)
            expected_line_cache, expected_polygon_cache = analytics.geometry_engine.compute(
                expected_points,
                expected_bboxes,
            )

            self.assertEqual(image_geometry_engine.coordinate_space, "image")
            self.assertEqual(analytics.geometry_engine.coordinate_space, "world")
            self.assertEqual(spatial.coordinate_space, "world")
            np.testing.assert_allclose(spatial.points, expected_points, atol=1e-6)
            np.testing.assert_allclose(spatial.bboxes, expected_bboxes, atol=1e-6)
            for key in expected_line_cache:
                np.testing.assert_array_equal(spatial.line_cache[key], expected_line_cache[key])
            for key in expected_polygon_cache:
                np.testing.assert_array_equal(spatial.polygon_cache[key], expected_polygon_cache[key])
            self.assertTrue(np.all(spatial.line_cache["vicinity_mask"][0]))
            self.assertTrue(np.all(spatial.polygon_cache["area_1"]))

    def test_period_metrics_match_controlled_formulas(self):
        cfg = _config()
        _, _, analytics, continuity = _runtime(cfg)
        batch = _batch(
            [
                (0.0, 0, 1, 0, 40.0, 49.0),
                (0.0, 0, 2, 1, 60.0, 47.0),
                (1.0, 1, 1, 0, 40.0, 51.0),
                (1.0, 1, 2, 1, 60.0, 47.0),
            ]
        )

        result = analytics.compute_period(batch, continuity)["area_1"]

        self.assertAlmostEqual(result["flow"].average_flow, 0.5)
        np.testing.assert_allclose(result["flow"].average_flow_by_class, [0.5, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(result["time_occupancy"].average_occupancy, 1.0)
        self.assertAlmostEqual(result["time_occupancy"].current_occupancy, 1.0)

    def test_boundary_context_preserves_crossing_across_reporting_periods(self):
        cfg = _config()
        _, _, analytics, continuity = _runtime(cfg)

        period_1 = _batch([(0.0, 0, 7, 0, 50.0, 49.0)])
        result_1 = analytics.compute_period(period_1, continuity)["area_1"]["flow"]
        self.assertAlmostEqual(result_1.average_flow, 0.0)
        self.assertEqual(continuity.boundary_batch.observation_count, 1)
        self.assertTrue(bool(continuity.boundary_batch.is_context[0]))

        buffer = PeriodObservationBuffer(max_observations=4)
        buffer.add_context(continuity.boundary_batch)
        buffer.append_frame(
            timing=FrameTiming(frame_id=1, timestamp=1.0, elapsed_seconds=0.0, unix_timestamp=1.0),
            track_ids=np.asarray([7], dtype=np.int64),
            class_ids=np.asarray([0], dtype=np.int32),
            points=np.asarray([[50.0, 51.0]], dtype=np.float32),
            bboxes=np.asarray([_bbox_from_point(50.0, 51.0)], dtype=np.float32),
        )

        period_2 = buffer.freeze(period_idx=1, timing_mode="frame", fps=1.0)
        result_2 = analytics.compute_period(period_2, continuity)["area_1"]["flow"]

        self.assertAlmostEqual(result_2.average_flow, 1.0)

    def test_time_headway_uses_cross_period_last_crossing(self):
        cfg = _config()
        _, _, analytics, continuity = _runtime(cfg)

        period_1 = _batch(
            [
                (0.0, 0, 1, 0, 50.0, 49.0),
                (1.0, 1, 1, 0, 50.0, 51.0),
            ]
        )
        result_1 = analytics.compute_period(period_1, continuity)["area_1"]["time_headway"]
        self.assertIsNone(result_1.average_time_headway)

        period_2 = _batch(
            [
                (2.0, 2, 2, 0, 50.0, 49.0),
                (3.0, 3, 2, 0, 50.0, 51.0),
            ]
        )
        result_2 = analytics.compute_period(period_2, continuity)["area_1"]["time_headway"]

        self.assertAlmostEqual(result_2.average_time_headway, 2.0)
        self.assertAlmostEqual(result_2.current_time_headway, 2.0)

    def test_empty_line_only_and_zone_only_batches_are_valid(self):
        empty_line_cache, empty_polygon_cache = GeometryEngine(lines={}, polygons={}).compute(
            points=np.empty((0, 2), dtype=np.float32),
            bboxes=np.empty((0, 4), dtype=np.float32),
        )
        self.assertEqual(empty_line_cache["distance"].shape, (0, 0))
        self.assertEqual(empty_polygon_cache, {})

        line_only_cfg = _config(line=True, zone=False, time_occupancy_enabled=False, time_headway_enabled=False)
        _, _, line_only_analytics, line_only_continuity = _runtime(line_only_cfg)
        line_only_batch = _batch(
            [
                (0.0, 0, 1, 0, 50.0, 49.0),
                (1.0, 1, 1, 0, 50.0, 51.0),
            ]
        )
        line_only_result = line_only_analytics.compute_period(
            line_only_batch,
            line_only_continuity,
        )["area_1"]["flow"]
        self.assertAlmostEqual(line_only_result.average_flow, 0.5)

        zone_only_cfg = _config(
            line=False,
            zone=True,
            flow_enabled=False,
            density_enabled=False,
            time_occupancy_enabled=False,
            time_headway_enabled=False,
        )
        _, _, zone_only_analytics, zone_only_continuity = _runtime(zone_only_cfg)
        zone_only_batch = _batch([], period_frames=[0], period_timestamps=[0.0])
        zone_only_result = zone_only_analytics.compute_period(
            zone_only_batch,
            zone_only_continuity,
        )["area_1"]
        self.assertEqual(zone_only_result, {})

    def test_period_observation_buffer_capacity_is_explicit(self):
        buffer = PeriodObservationBuffer(max_observations=1)
        timing = FrameTiming(frame_id=0, timestamp=0.0, elapsed_seconds=0.0, unix_timestamp=0.0)

        buffer.append_frame(
            timing=timing,
            track_ids=np.asarray([1], dtype=np.int64),
            class_ids=np.asarray([0], dtype=np.int32),
            points=np.asarray([[50.0, 49.0]], dtype=np.float32),
            bboxes=np.asarray([_bbox_from_point(50.0, 49.0)], dtype=np.float32),
        )

        with self.assertRaises(PeriodObservationBufferOverflow):
            buffer.append_frame(
                timing=timing,
                track_ids=np.asarray([2], dtype=np.int64),
                class_ids=np.asarray([0], dtype=np.int32),
                points=np.asarray([[50.0, 51.0]], dtype=np.float32),
                bboxes=np.asarray([_bbox_from_point(50.0, 51.0)], dtype=np.float32),
            )


if __name__ == "__main__":
    unittest.main()
