# Repository Context for AI Coding Agents

This is the architectural handoff for `traffic-sensor-ai`. It is intended for future developers and AI coding agents working in this repository without rediscovering the whole system from scratch.

Always read `AGENTS.md` before changing files. For Python commands, use the existing `smart_sensor` Conda environment:

```powershell
C:\Users\hamza\anaconda3\Scripts\conda.exe run -n smart_sensor <command>
```

## System Purpose

`traffic-sensor-ai` is a Python computer-vision smart traffic sensor. The runtime reads frames from an OpenCV-compatible source, runs object detection, feeds detections into a continuous BoxMOT tracker, applies configured spatial geometry, derives crossing events, computes eligible traffic metrics, and optionally publishes period-level metrics over MQTT.

The primary runtime is a synchronous two-stage pipeline:

```text
Stage 1: acquisition -> TimingPolicy -> detection -> tracking -> canonical observations -> PeriodObservationBuffer
Stage 2: PeriodObservationBatch -> optional coordinate transform -> GeometryEngine -> events -> metrics -> period report
```

Stage 2 is intentionally synchronous. There is no `analytics.async_stage2` switch.

## Current Configuration Schema

`config/traffic_metrics.yaml` is the authoritative runtime schema. Do not reintroduce old fallback sections unless an explicit external interface requires it.

Top-level responsibilities:

```yaml
input:
  source: ...
  fps: ...

frame_processing:
  producer: direct | sampled_offline | realtime_simulation
  sampling:
    enabled: ...
    policy: ...
    stride: ...

timing:
  mode: frame | unix

analytics:
  period_seconds: ...
  max_obs_per_frame: ...

continuity:
  enabled: ...
  max_age_seconds: ...

homography:
  enabled: ...
  calibration_file: ...

geometry:
  coordinate_space: image | world | auto
  frame_size_reference: ...
  polygon_membership_mode: center | corners | any
  vicinity:
    image: ...
    world_m: ...
  lines: [...]
  areas: [...]

metrics:
  flow: ...
  density: ...
  space_occupancy: ...
  time_occupancy: ...
  space_headway: ...
  time_headway: ...
  speed: ...
```

## Runtime Assembly

Primary runtime code lives in `scripts/sensor_pipeline.py`.

Assembly order:

1. Load YAML with `config/load_build.py::load_config`.
2. Resolve source from CLI override or `input.source`.
3. Resolve FPS from CLI override or `input.fps`.
4. Build detector through `detection/factory.py::build_detector`.
5. Read detector metadata from `detector.num_classes` and `detector.class_names`.
6. Build typed areas from `geometry.areas` and lines from `geometry.lines`.
7. Resolve valid area/metric combinations from direct `metrics.*` definitions.
8. Build one continuous `Tracker`.
9. Build one centralized `TimingPolicy`.
10. Build `GeometryEngine` and `PeriodAnalyticsEngine`.
11. Run Stage 1 per reporting period, freeze `PeriodObservationBatch`, then run synchronous Stage 2.

The tracker is deliberately built once and preserved across reporting periods.

## Runtime Period Policy

Reporting-period sizing is centralized in `src/runtime/periods.py`.

`analytics.period_seconds` is validated as positive and clamped to a maximum of 3600 seconds. The bounded observation-buffer capacity is not configured directly. It is derived after the effective processing FPS is known:

```text
max_observations = int(period_seconds * effective_fps * max_obs_per_frame)
```

`analytics.max_obs_per_frame` defaults to 50 and is clamped to `[10, 50]`. `PeriodObservationBuffer` still receives the derived concrete capacity and still raises `PeriodObservationBufferOverflow` when exceeded.

## Stage 1 Contract

Stage 1 writes observations into `src/runtime/observations.py::PeriodObservationBuffer`, then freezes a `PeriodObservationBatch`.

`PeriodObservationBatch` columns:

| Field | Shape | Meaning |
|---|---:|---|
| `timestamp` | `(O,)` | Authoritative timestamp from `TimingPolicy`. |
| `frame_id` | `(O,)` | Source frame index. |
| `track_id` | `(O,)` | Tracker identity. |
| `class_id` | `(O,)` | Detector/tracker class ID. |
| `points` | `(O, 2)` | Bottom-center object point. |
| `bboxes` | `(O, 4)` | Tracked bbox. |
| `is_context` | `(O,)` | Prior-period context row flag. |
| `period_frame_ids` | `(F,)` | Active source frames in the period. |
| `period_timestamps` | `(F,)` | Active frame timestamps in the period. |

`O` is observation count and `F` is processed frame count. Context rows are retained only for continuity and excluded from period metric values with `batch.active_mask`.

`PeriodObservationBuffer` is bounded and raises `PeriodObservationBufferOverflow`; it must not silently discard observations.

Stage 1 reusable period execution lives in `src/runtime/stage1.py`. Full runtime orchestration and the standalone exporter both use this same loop for acquisition, timing, detection, tracking, canonical observation extraction, and buffer freezing.

Standalone Stage 1 export utility:

```powershell
C:\Users\hamza\anaconda3\Scripts\conda.exe run -n smart_sensor python scripts\run_stage1.py --source video.mp4 --config config\traffic_metrics.yaml --output outputs\stage1 --max-periods 1
```

It exports one `.npz` per period with the batch columns above.

## Timing

`src/runtime/timing.py::TimingPolicy` owns timestamp semantics:

- `timing.mode: frame` uses `frame.read_idx / input.fps`.
- `timing.mode: unix` uses the acquisition Unix timestamp on `Frame.timestamp`.

`input.fps` belongs to the source configuration. Do not duplicate FPS under `timing`.

## Geometry and Homography

Spatial configuration belongs under `geometry`.

- `geometry.lines` defines line IDs and points.
- `geometry.areas` defines `area_id`, `area_type`, `enabled`, optional `zone`, and optional `flow_line_id`.
- `geometry.vicinity.image` is the global image-space line vicinity threshold.
- `geometry.vicinity.world_m` is the global world-space line vicinity threshold.

Supported `area_type` values:

- `lane`
- `direction`
- `mixed`
- `entire`

`src/geometry/primitives.py::Area` validates `area_type`.

Homography is centralized in `src/runtime/coordinates.py`. Static lines and polygons are transformed once during `PeriodAnalyticsEngine` initialization when world coordinates are selected. Dynamic batch `points` and `bboxes` are transformed vectorially once in Stage 2. Individual metrics must not call homography.

The geometry subsystem owns all spatial reasoning in both image and world modes:

- line signed distance,
- line sign,
- vicinity masks,
- polygon membership,
- spatial cache shape and semantics.

`src/geometry/primitives.py::GeometryEngine` exposes the single Stage 2 spatial-computation contract. Image-space engines use raster polygon masks and pixel-scaled vicinity thresholds. World-space engines use transformed `SpatialLine` / `SpatialPolygon` inputs, metric thresholds, and vectorized point-in-polygon logic from `src/geometry/spatial.py`. Both paths return the same `(line_cache, polygon_cache)` structure.

`src/traffic_metrics/engine.py::PeriodAnalyticsEngine` selects the coordinate space, invokes `CoordinateTransformer` when needed, delegates all spatial cache computation to `GeometryEngine`, then derives events and dispatches metric estimators.

Coordinate policy:

- `image`: no homography, image-space geometry.
- `world`: requires enabled homography and calibration file.
- `auto`: selects world when enabled homography and world vicinity are available; otherwise image.

## Area x Metric Eligibility

Eligibility is centralized in `config/load_build.py::resolve_eligible_metrics`.

An eligible metric name is attached to an area only when all are true:

- `metrics.<metric>.enabled` is true.
- The metric backend is implemented.
- `spatial_scope` is compatible with `area.area_type`.
- `coordinate_requirement` is satisfied.
- Required topology exists, such as zone and/or flow line.

Supported scopes:

- `any_area`: lane, direction, mixed, entire.
- `lane_only`: only `area_type == lane`.

Supported coordinate requirements:

- `image_or_world`: allowed in image or world space.
- `world_required`: allowed only when world coordinates are configured and available.

World-required metrics are not silently computed in image coordinates.

## Metric Backend Status

Current status under the redesigned config:

| Metric | Status | Notes |
|---|---|---|
| `flow` | Implemented | Uses line crossing/vicinity masks and detector-derived class vector size. |
| `density` | Implemented when world coordinates are available | Skipped in image mode because config marks it `world_required`. |
| `space_occupancy` | Architecture-only / not implemented | Skipped with an ineligible reason; no validated backend exists yet. |
| `time_occupancy` | Implemented | Maps to the existing temporal occupancy semantics: fraction of frames occupied near line and inside zone. |
| `space_headway` | Implemented when world coordinates are available and area is lane | Skipped outside lane or without world coordinates. |
| `time_headway` | Implemented | Lane-only, image or world. Preserves cross-period last crossing timestamp. |
| `speed` | Architecture-only / not implemented | No validated speed metric backend was found; config support is extensible but computation is skipped. |

Metric result dataclasses live in `src/traffic_metrics/models.py`. Metric algorithms live in `src/traffic_metrics/estimators.py`. `src/traffic_metrics/engine.py` coordinates Stage 2 and dispatches eligible metrics by name.

## Detector Metadata

Detector adapters implement the common `IDetector` metadata contract:

```python
detector.class_names
detector.num_classes
```

Runtime metric vector sizing comes from `detector.num_classes`, not configuration.

Backends:

- Ultralytics: normalized from loaded model `names` metadata.
- Torchvision: normalized from weights enum `meta["categories"]`.

## Continuity

One global continuity horizon is configured:

```yaml
continuity:
  enabled: true
  max_age_seconds: 30
```

Continuity policy is centralized in `src/runtime/continuity.py`.

`continuity.enabled` controls whether prior-period boundary observations are injected as context rows into the next `PeriodObservationBatch`. When disabled, no previous-period observations are inserted and `is_context` has no carried rows. The detector/tracker pipeline remains continuous because the tracker object is created once and reused independently of analytical boundary context.

`continuity.max_age_seconds` controls:

- retained boundary observations,
- crossing TTL fallback in Stage 2,
- duplicate flow-count suppression,
- time-headway continuity retention shape for future bounded state handling.

Do not add separate crossing/counter TTLs.

## Component Map

| Component | Responsibility |
|---|---|
| `config/load_build.py` | Current-schema YAML assembly, metric eligibility, area/geometry construction. |
| `scripts/sensor_pipeline.py` | Runtime orchestration around reusable Stage 1 and synchronous Stage 2 call. |
| `scripts/run_sensor.py` | Direct CLI entry point. |
| `scripts/run_stage1.py` | Stage 1-only `.npz` batch exporter. |
| `scripts/sensor_daemon.py` | MQTT-controlled daemon entry point. |
| `src/runtime/periods.py` | Reporting-period validation and derived buffer-capacity policy. |
| `src/runtime/continuity.py` | Boundary context policy and cross-period analytics state container. |
| `src/runtime/stage1.py` | Reusable Stage 1 period loop and canonical tracker-output extraction. |
| `src/runtime/timing.py` | Centralized timing policy. |
| `src/runtime/observations.py` | Bounded period buffer and batch representation. |
| `src/runtime/coordinates.py` | Centralized optional homography. |
| `src/traffic_metrics/engine.py` | Stage 2 vectorized analytics orchestration. |
| `src/traffic_metrics/models.py` | Traffic analytics/result contracts. |
| `src/traffic_metrics/estimators.py` | Current vectorized traffic metric algorithms. |
| `src/geometry/primitives.py` | Line, Polygon, transformed spatial primitives, typed Area, GeometryEngine. |
| `src/geometry/spatial.py` | Shared image/world spatial algorithms used by GeometryEngine. |
| `src/detection/*` | Detector adapters and metadata contract. |
| `src/tracking/track.py` | BoxMOT tracker wrapper. |
| `src/video_io/frame_producer.py` | Direct, sampled offline, and real-time simulation producers. |
| `visualization/draw_geometry.py` | Draws current-schema `geometry.lines` and area zones. |
| `tests/test_period_runtime.py` | Current-schema deterministic runtime tests. |

## Verification

Compile:

```powershell
C:\Users\hamza\anaconda3\Scripts\conda.exe run -n smart_sensor python -m compileall config src scripts tests visualization
```

Tests:

```powershell
C:\Users\hamza\anaconda3\Scripts\conda.exe run -n smart_sensor python -m unittest discover tests
```

Latest test result:

```text
Ran 31 tests
OK
```

Current config smoke check:

```powershell
C:\Users\hamza\anaconda3\Scripts\conda.exe run -n smart_sensor python -c "from config.load_build import load_config, build_areas, build_geometry_engine; from traffic_metrics.engine import PeriodAnalyticsEngine; cfg=load_config('config/traffic_metrics.yaml'); areas=build_areas(cfg, num_classes=4); engine=build_geometry_engine(areas,cfg); analytics=PeriodAnalyticsEngine(areas,engine,cfg,num_classes=4); print(len(areas), len(engine._line_ids), analytics.coordinate_space, {a.area_id: sorted(a.eligible_metrics) for a in areas})"
```

Latest result:

```text
5 1 image {'area_1': ['flow', 'time_headway', 'time_occupancy'], 'area_2': ['flow', 'time_headway', 'time_occupancy'], 'area_3': ['flow', 'time_headway', 'time_occupancy'], 'area_4': ['flow', 'time_headway', 'time_occupancy'], 'area_5': ['flow']}
```

`flake8` is configured in `setup.cfg`, but it is not installed in the tested `smart_sensor` environment.

## Known Limitations

- `space_occupancy` has no validated scientific backend yet.
- `speed` has no validated metric backend yet.
- `smart_sensor.service` hard-codes paths and runtime arguments.
- `README.md` is empty.

## Architectural Invariants

- Preserve the two-stage architecture.
- Do not rebuild tracker state per reporting period.
- Do not silently discard rows from `PeriodObservationBuffer`.
- Keep detector output canonical as `(N, 6) = [x1, y1, x2, y2, score, class_id]`.
- Use detector metadata, not config, for class vector sizing.
- Keep homography centralized.
- Do not compute `world_required` metrics in image space.
- Keep global line vicinity under `geometry.vicinity`.
- Keep runtime policy in the current YAML structure.
