# Repository Context for AI Coding Agents

This is the technical architectural handoff for `traffic-sensor-ai`. It is intended for future developers and AI coding agents working on the repository, especially when making code changes without rediscovering the system from scratch. It describes what the repository currently implements, how the runtime is assembled, how frames move through the traffic-sensing pipeline, and where common modifications should be made.

## Quick Start

The primary runtime entry points currently present are:

```powershell
$env:PYTHONPATH = "$PWD\src;$PWD\scripts;$PWD"
python scripts\run_sensor.py --source path\to\video.mp4 --config config\traffic_metrics.yaml --period-mins 5 --sensor-id camera_1
```

With MQTT publishing enabled:

```powershell
$env:PYTHONPATH = "$PWD\src;$PWD\scripts;$PWD"
python scripts\run_sensor.py --source 0 --mqtt-broker-host 10.101.100.47 --sensor-id camera_1
```

For MQTT-controlled daemon mode:

```powershell
$env:PYTHONPATH = "$PWD\src;$PWD\scripts;$PWD"
python scripts\sensor_daemon.py --source 0 --mqtt-broker-host 10.101.100.47 --sensor-id camera_1
```

`smart_sensor.service` provides a Linux systemd deployment example that launches `scripts/sensor_daemon.py` from `/home/hamza/traffic-sensor-ai`.

## System Purpose

`traffic-sensor-ai` is a Python computer-vision smart traffic sensor. The implemented runtime reads frames from an OpenCV-compatible source, runs object detection, feeds detections into a BoxMOT tracker, applies configurable line and polygon geometry, extracts line-crossing events, computes per-area traffic metrics, and optionally publishes period-level results over MQTT.

The repository also contains dataset conversion, fine-tuning, visualization, homography, and frame-sampling utilities. Not all of those utilities are wired into the primary runtime path.

## Architecture Overview

Current primary runtime architecture:

```text
CLI or MQTT daemon
      |
Configuration loading
      |
Detector construction
      |
Period loop
      |
Direct OpenCV frame acquisition
      |
Detector output: [x1, y1, x2, y2, score, class_id]
      |
BoxMOT tracker output
      |
Track point extraction
      |
GeometryEngine line and polygon masks
      |
CrossingExtractor line-crossing events
      |
Per-area metric computation
      |
Console result and optional MQTT metrics publish
```

Daemon mode adds an outer lifecycle:

```text
BOOTING -> READY -> PROCESSING -> READY
    |        |          |
    |        |          run_sensor(...)
    |        latest-frame acquisition for snapshots
    MQTT client, command subscriptions, configuration receiver
```

## Runtime Processing Workflow

The main processing code is in `scripts/sensor_pipeline.py`.

`run_sensor()` performs process-level setup:

1. Loads YAML with `config/load_build.py::load_config`.
2. Resolves FPS from the CLI argument or `general_params.fps`.
3. Builds one detector with `detection/factory.py::build_detector`.
4. Opens a `video_io/frame_producer.py::DirectFrameProducer`.
5. Creates an optional `communication/services.py::MetricsPublisherService`.
6. Processes consecutive periods by calling `process_period()`.

`process_period()` owns the per-period loop:

1. Builds fresh areas, `GeometryEngine`, `CrossingExtractor`, and `Tracker` through `build_period_state()`.
2. Reads frames one at a time from `DirectFrameProducer.next_frame()`.
3. Updates `LatestFrameStore` if daemon snapshots are enabled.
4. Computes the period-relative timestamp from source frame indices and FPS.
5. Runs `detector.detect_to_track(frame.data)`.
6. Runs `tracker.update(detections, frame.data)`.
7. Converts tracks into bottom-center points, bboxes, track IDs, and class IDs with `extract_tracking_outputs()`.
8. Computes line and polygon caches with `GeometryEngine.compute(points, bboxes)`.
9. Updates crossing state with `CrossingExtractor.update(...)`.
10. Computes metrics for every area in `compute_area_metrics(...)`.
11. Ends the period once `source_frames_elapsed >= period_frames` or the source ends.

Important: detector construction persists across periods, but tracker, crossing extractor, area metric objects, and their histories are rebuilt at the start of every period.

## Major Components and Responsibilities

| Component | Responsibility | Inputs | Outputs | Important Dependencies |
|---|---|---|---|---|
| `scripts/run_sensor.py` | CLI entry point for direct period processing | CLI source, config, MQTT options | Calls `run_sensor()` | `scripts/sensor_pipeline.py`, MQTT client |
| `scripts/sensor_daemon.py` | MQTT-controlled lifecycle daemon | MQTT commands, video source, config path | Starts/stops processing, serves snapshots/configuration | `communication/*`, `scripts/sensor_pipeline.py` |
| `scripts/sensor_pipeline.py` | Primary pipeline orchestration | Frames, config, detector, tracker, geometry | Period result dictionaries and optional MQTT metrics | Detection, tracking, geometry, crossing, metrics |
| `config/load_build.py` | YAML loading and runtime area/geometry/metric assembly | `config/traffic_metrics.yaml` | `Area` objects, `GeometryEngine`, `CrossingExtractor` | `geometry.primitives`, `traffic_metrics.metrics` |
| `src/video_io/frame_producer.py` | OpenCV frame acquisition and optional sampling/buffering utilities | Video path, camera index, RTSP-like source | `Frame` objects | OpenCV |
| `src/detection/*` | Detector interface, detector factory, detector implementations | Image frame | Canonical detection array `(N, 6)` | Ultralytics, Torchvision, NumPy |
| `src/tracking/track.py` | BoxMOT tracker wrapper | Detector array and frame | BoxMOT track array | BoxMOT |
| `src/geometry/primitives.py` | Lines, polygons, areas, vectorized geometry masks | Track points, bboxes, configured geometry | `line_cache`, `polygon_cache` | Pydantic, NumPy, PIL mask helpers |
| `src/crossing/crossing_estimation.py` | Stateful line-crossing event extraction | Track IDs, line cache, polygon cache, timestamp | `crossed_masks`, per-polygon crossing timestamps | NumPy |
| `src/traffic_metrics/metrics.py` | Stateful metric implementations | Masks, classes, track IDs, timestamps, crossing events | Dataclass result objects | NumPy |
| `src/communication/*` | MQTT topics, payloads, publish/subscribe, offline queue, snapshots/configuration | Period results, MQTT messages, frames | MQTT messages, queued messages, saved config/snapshots | Paho MQTT, OpenCV |

## Data Flow and Shared Intermediate Results

Detection is run once per processed frame. The resulting canonical detector array is shared with tracking rather than recomputed for each metric.

The tracker output is converted once into:

- `points`: `(N, 2)` bottom-center points derived from tracked bboxes.
- `bboxes`: `(N, 4)` tracked bounding boxes.
- `track_ids`: `(N,)` integer track IDs.
- `det_cls`: `(N,)` integer class IDs.

`GeometryEngine.compute()` runs once per frame and produces shared geometry caches:

- `line_cache["distance"]`: `(L, N)` absolute distances to configured lines.
- `line_cache["sign"]`: `(L, N)` signed side of each line.
- `line_cache["vicinity_mask"]`: `(L, N)` line-vicinity booleans.
- `polygon_cache`: `{polygon_id: (N,) bool mask}`.

`CrossingExtractor.update()` consumes the shared geometry caches and emits:

- `crossed_masks`: `(L, N)` booleans for frame-level line crossings.
- `new_crossings`: mapping from polygon or area ID to crossing timestamps.

Metrics then reuse the same track IDs, classes, masks, and events. The repository therefore implements a shared one-pass per-frame detection/tracking/geometry path for the metrics wired in `scripts/sensor_pipeline.py`.

## Control Flow and Orchestration

Direct CLI control flow:

```text
scripts/run_sensor.py::main
    -> optional SmartSensorMqttClient.connect()
    -> scripts/sensor_pipeline.py::run_sensor()
    -> repeated process_period()
    -> optional MetricsPublisherService.publish_period_metrics()
    -> producer.release()
    -> optional MQTT disconnect
```

Daemon control flow:

```text
scripts/sensor_daemon.py::main
    -> SensorDaemon.start()
    -> initialize MQTT topics, client, services, command handlers
    -> subscribe to sensors/<sensor_id>/commands/#
    -> READY frame acquisition for snapshots
    -> START command spawns processing thread
    -> STOP command sets stop_event
    -> processing thread returns daemon to READY
```

Error handling is limited. `SensorDaemon._run_processing()` logs exceptions and returns to READY. `run_sensor()` catches `KeyboardInterrupt` and releases the producer in `finally`. MQTT publish failures are queued when `queue_on_failure=True`.

## Configuration Architecture

Primary configuration file:

- `config/traffic_metrics.yaml`

Primary loader and builder:

- `config/load_build.py`

The config is read with `yaml.safe_load()` and is not schema-validated as a complete configuration object. Validation occurs mainly when Pydantic models are instantiated for `Line`, `Polygon`, and `Area`.

Important YAML sections:

| Section | Used by current primary runtime | Notes |
|---|---:|---|
| `detector.model_name` | Yes | Selects Ultralytics or Torchvision detector in `src/detection/factory.py`. |
| `detector.confidence` | Yes | Passed as `conf` to detector construction. |
| `tracker.*` | Yes | Passed to `src/tracking/track.py::Tracker`. |
| `lines` | Yes | Used to build flow lines. |
| `areas` | Yes | Defines enabled flag, zones, flow lines, and requested metrics. |
| `metrics.flow.counter_logic` | Yes | Controls counter logic used by `Counter`. |
| `metrics.flow.ttl_seconds` | Yes | Controls counter counted-ID cleanup. |
| `general_params.frame_size` | Yes | Converts normalized line vicinity to pixels. |
| `general_params.fps` | Yes | Default FPS if CLI does not override. |
| `general_params.num_classes` | Yes | Metric vector size. |
| `general_params.crossing_ttl` | Yes | Crossing fallback TTL. |
| `frame_grabber.*` | Not in current `run_sensor()` path | Present in YAML, but `run_sensor()` uses the CLI `source` and `DirectFrameProducer` without sampling. |

`Area.enable` is stored on `Area` but current `build_areas()` and `compute_area_metrics()` do not filter disabled areas.

## Runtime Component Assembly

`config/load_build.py::build_areas()` constructs one `Area` per YAML `areas` item. For each area:

- `zone` becomes a `Polygon` with `polygon_id` equal to the area ID.
- `flow_line_id` is looked up in YAML `lines` and becomes a `Line`.
- requested metrics are filtered through `get_eligible_metrics()`.
- metric objects are instantiated from `METRIC_REGISTRY`.

Metric eligibility currently works as follows:

- `flow` requires a flow line.
- `density` requires a zone.
- `occupancy`, `space_headway`, and `time_headway` require both a zone and a flow line.

`build_runtime_components()` extracts unique lines and polygons from areas, builds a `GeometryEngine`, and builds a `CrossingExtractor` using the geometry engine's line ID ordering.

The detector is selected separately by `src/detection/factory.py::build_detector()`.

The tracker is constructed directly in `scripts/sensor_pipeline.py::build_period_state()` using the tracker config.

## Frame Acquisition and Sampling

The `Frame` dataclass in `src/video_io/frame_producer.py` carries:

- `data`: image array.
- `timestamp`: wall-clock timestamp from `time.time()`.
- `read_idx`: source-read frame index.
- `processed_idx`: optional processing index, not assigned by the current primary loop.

Implemented acquisition classes:

- `FrameGrabber`: OpenCV `VideoCapture` wrapper. It normalizes digit strings such as `"0"` into integer camera indices. For non-camera string sources, it sleeps by source FPS to regulate playback.
- `DirectFrameProducer`: simple sequential producer used by `scripts/sensor_pipeline.py` and daemon READY snapshot acquisition.
- `FrameProducer`: threaded producer with `CircularFrameBuffer` and sampling.
- `OfflineSampledFrameProducer`: sequential sampled producer for offline evaluation.
- `RealTimeSimulationProducer`: prerecorded-video simulation with latency-based frame skipping.

Current primary runtime uses `DirectFrameProducer`, so it performs no configured stride, buffering, sampling, or latency-based dropping.

## Detection

`src/detection/interface.py::IDetector` defines the common detector contract:

```text
np.ndarray shape (N, 6)
[x1, y1, x2, y2, score, class_id]
```

Implemented detector paths:

- `src/detection/ultralytics_detectors.py::UltralyticsDetector` for model names containing `yolo` or `rtdetr`. The factory expects weights at `models/<model_name>.pt`.
- `src/detection/torchvision_detectors.py::TorchvisionDetector` for names listed in `TORCHVISION_MODELS`.

`src/detection/factory.py::TORCHVISION_MODELS` currently includes:

- `fasterrcnn_resnet50_fpn`
- `fasterrcnn_resnet50_fpn_v2`
- `retinanet_resnet50_fpn`
- `ssd300_vgg16`
- `ssdlite320_mobilenet_v3_large`
- `fcos_resnet50_fpn`

`UltralyticsDetector` warms the model during construction. The current code raises `DetectorError` for unsupported Ultralytics families but does not import `DetectorError` in `src/detection/ultralytics_detectors.py`.

## Tracking

`src/tracking/track.py::Tracker` wraps BoxMOT:

- Clears `BaseTrack` counts on construction.
- Calls `boxmot.tracker_zoo.create_tracker()`.
- Uses `get_tracker_config(method)`.
- Passes `reid_model`, `device`, `half`, and `per_class`.
- Applies optional class filtering before `tracker.update()`.

Tracker output is treated by `scripts/sensor_pipeline.py::extract_tracking_outputs()` as:

```text
tracks[:, :4] -> bbox
tracks[:, 4]  -> track_id
tracks[:, 6]  -> class_id
```

The tracker is rebuilt for every period, which resets identity state at period boundaries.

## Geometry and Traffic-Space Model

Primary geometry lives in `src/geometry/primitives.py`.

Implemented concepts:

- `Line`: two configured points plus optional normalized `vicinity`. It caches canonical `A, B, C` coefficients for `Ax + By + C = 0`.
- `Polygon`: at least three points plus optional `distance_meters`. It caches a tight raster mask, bounding-box offset, and polygon area.
- `Area`: logical monitoring unit with optional `zone`, optional `flow_line`, and metric objects.
- `GeometryEngine`: vectorized line distances/signs/vicinity masks and polygon membership masks.
- `SpeedLinePair`: present as a geometry primitive, but no active integration was found in the primary runtime.

Configured road-space model in `config/traffic_metrics.yaml`:

- Global `lines` define virtual lines.
- `areas[].zone.points` define polygons or ROIs.
- `areas[].flow_line_id` links an area to a line.
- `areas[].metrics` requests metrics for that area.

`GeometryEngine` defaults to polygon membership mode `"center"`, which uses configured or derived object points against the polygon mask. Other modes, `"corners"` and `"any"`, exist in code but are not exposed by the current YAML builder.

Homography support is implemented in `src/geometry/homography.py`, with an example calibration in `config/cam01_homography.yaml`. It supports image/world projection, bbox projection, polygon projection, local scale, and bird's-eye warping. Current repository inspection indicates this homography module is not wired into `scripts/sensor_pipeline.py`.

## Traffic Metrics

Implemented metrics are in `src/traffic_metrics/metrics.py`.

| Metric | Responsible class | Required upstream information | Temporal type | State maintained | Output |
|---|---|---|---|---|---|
| Vehicle count helper | `Counter` | track IDs, classes, current time, crossed mask, vicinity mask, optional polygon mask | Event/filter based | counted IDs with TTL, cumulative counts | `CountResult` |
| Flow | `FlowMetric` | cumulative count and current period time from `Counter` | Period average | None inside `FlowMetric` | `FlowResult` |
| Density | `DensityMetric` | polygon mask, classes, `distance_meters` | Instantaneous plus historical average | density history and class-density history | `DensityResult` |
| Occupancy | `OccupancyMetric` | line vicinity mask and polygon mask | Frame occupancy plus historical average | frame count and occupied-frame count | `OccupancyResult` |
| Space headway | `SpaceHeadwayMetric` | tracked points in polygon, direction from flow line coefficients | Instantaneous plus historical average | headway history | `SpaceHeadwayResult` |
| Time headway | `TimeHeadwayMetric` | new crossing timestamps for the area | Event based | last crossing time and headway history | `TimeHeadwayResult` |

Flow uses a special `Counter` object stored under `area.metrics["counter"]`; `Counter` is not listed as an area metric output. `counter_logic` values are implemented as:

- `counter_2`: crossed mask only.
- `counter_3`: vicinity mask only.
- `counter_4`: crossed and vicinity.
- `counter_5`: crossed or vicinity.

All metric histories are per-period because metric objects are rebuilt at the start of each period.

## State and Temporal Processing

Stateful runtime owners:

- `Tracker` owns BoxMOT tracking state. It is reset per period.
- `CrossingExtractor` owns per-line, per-track sign/distance/vicinity/polygon history. It is reset per period.
- `Counter` owns counted IDs and cumulative counts. It is reset per period.
- `DensityMetric`, `OccupancyMetric`, `SpaceHeadwayMetric`, and `TimeHeadwayMetric` own histories or last-event state. They are reset per period.
- `LatestFrameStore` owns a thread-safe copy of the latest frame for snapshots.
- `SmartSensorMqttClient` owns connection state and an `OfflineMessageQueue`.
- `SensorDaemon` owns lifecycle state, command handlers, processing thread, stop event, and READY acquisition service.

Within a period, `current_time` is calculated from source frame index elapsed since the first period frame. It is not taken from `Frame.timestamp`.

## Extensibility Mechanisms

| Extension | Current mechanism | Files normally modified | Pipeline modification needed |
|---|---|---|---|
| Add a detector | Implement `IDetector`, return `(N, 6)`, update factory selection | `src/detection/interface.py`, new or existing detector file, `src/detection/factory.py` | Usually no, if factory returns compatible detector |
| Add a Torchvision detector | Add model and weights mapping | `src/detection/torchvision_detectors.py`, `src/detection/factory.py` | No, if output stays `(N, 6)` |
| Add a tracker method | Rely on BoxMOT `create_tracker()` and config name | `config/traffic_metrics.yaml`; possibly `src/tracking/track.py` | No, if BoxMOT supports the method |
| Add a traffic metric | Add metric class/result, registry entry, eligibility rules, and compute call | `src/traffic_metrics/metrics.py`, `config/load_build.py`, `scripts/sensor_pipeline.py`, `config/traffic_metrics.yaml` | Yes, current metric dispatch is explicit |
| Add a geometry primitive | Add primitive and builder/runtime consumption | `src/geometry/primitives.py`, `config/load_build.py`, pipeline consumer | Usually yes |
| Expose polygon mode | Add config key and pass it to `GeometryEngine` | `config/traffic_metrics.yaml`, `config/load_build.py` | No if existing modes are sufficient |
| Change frame sampling | Wire existing sampled producers into runtime | `src/video_io/frame_producer.py`, `scripts/sensor_pipeline.py`, config | Yes, current runtime hard-codes `DirectFrameProducer` |
| Add output or communication mechanism | Add service/client and call from period result publishing | `src/communication/*`, `scripts/run_sensor.py`, `scripts/sensor_daemon.py`, `scripts/sensor_pipeline.py` | Usually yes |
| Add MQTT command | Add topic property, service or handler, registration | `src/communication/topics.py`, `scripts/sensor_daemon.py`, possibly `src/communication/services.py` | No for pipeline unless command affects processing |

The repository has factory-like selection for detectors and BoxMOT tracker methods, but metrics and frame producers are not fully plug-in configured.

## Communication and External Interfaces

MQTT support is implemented in `src/communication/*` and used by `scripts/run_sensor.py` and `scripts/sensor_daemon.py`.

Topic construction is centralized in `src/communication/topics.py::SensorTopics`.

For `root="sensors"` and `sensor_id="camera_1"`:

| Topic | Direction | Payload | Implemented handling |
|---|---|---|---|
| `sensors/camera_1/metrics` | publish | JSON metrics payload from `build_metrics_payload()` | Published after each processed period |
| `sensors/camera_1/status` | publish | JSON status payload builder exists | Topic and builder exist; no active periodic publisher found |
| `sensors/camera_1/commands/snapshot` | subscribe | bytes, content ignored | Publishes latest JPEG frame to snapshot response |
| `sensors/camera_1/responses/snapshot` | publish | JPEG bytes | Used by `SnapshotProviderService` |
| `sensors/camera_1/commands/configuration` | subscribe | YAML file bytes | Saves to configured config path |
| `sensors/camera_1/responses/configuration` | publish | Not actively used by daemon | Topic exists |
| `sensors/camera_1/commands/start` | subscribe | bytes, content ignored | Starts processing thread if READY and configured |
| `sensors/camera_1/responses/start` | publish | Not actively used by daemon | Topic exists |
| `sensors/camera_1/commands/stop` | subscribe | bytes, content ignored | Sets processing stop event |
| `sensors/camera_1/responses/stop` | publish | Not actively used by daemon | Topic exists |
| `sensors/camera_1/commands/reboot` | subscribe | bytes, content ignored | Calls `sudo reboot` |
| `sensors/camera_1/responses/reboot` | publish | Not actively used by daemon | Topic exists |

`src/communication/mqtt_client.py::SmartSensorMqttClient` uses Paho MQTT callback API v2, starts the network loop with `loop_start()`, tracks `connected`, and flushes queued messages after a successful connection.

`src/communication/offline_queue.py::OfflineMessageQueue` stores failed publishes as JSON files in `outputs/pending_mqtt` or a configured subdirectory.

## Edge-AI Deployment

Edge deployment evidence in the repository:

- `smart_sensor.service` is a systemd service for Linux deployment.
- `restart_sensor.sh` interactively writes `/etc/default/smart_sensor`, reloads systemd, and restarts `smart_sensor.service`.
- `requirements_deploy.txt` is a smaller runtime dependency list and notes that Jetson Orin Nano should use torch/torchvision builds matching JetPack/CUDA.
- `config/traffic_metrics.yaml` defaults tracker `device` to `"cuda"` and `half` to `true`.
- `src/utils/profilers.py` can measure CUDA event timing when the device starts with `"cuda"` and CUDA is available.

No Dockerfile, TensorRT integration, packaged installer, or explicit Jetson startup environment loader was found. `smart_sensor.service` currently hard-codes command-line values and does not reference `/etc/default/smart_sensor`, although `restart_sensor.sh` writes that file.

## Outputs, Logging, and Persistence

Current outputs:

- `scripts/sensor_pipeline.py` prints frame indices and period result dictionaries to stdout.
- MQTT metrics JSON is published to `SensorTopics.metrics` when an MQTT client is provided.
- Failed MQTT publishes are persisted as JSON files in `outputs/pending_mqtt/...`.
- Snapshot responses are JPEG bytes over MQTT.
- `SnapshotReceiverService` can save snapshot JPEGs under `outputs/snapshots`, but no active integration was found in the primary runtime.
- Configuration commands save uploaded YAML bytes to the configured config path.

Logging:

- `scripts/sensor_daemon.py` configures Python logging in daemon mode.
- Some components still use `print()` for diagnostics.

## Experimental and Evaluation Infrastructure

Runtime-adjacent but not primary pipeline:

- `src/video_io/frame_producer.py::OfflineSampledFrameProducer`
- `src/video_io/frame_producer.py::RealTimeSimulationProducer`
- `src/video_io/frame_producer.py::FrameProducer`
- `src/geometry/homography.py`
- `src/utils/shape_setter.py`
- `visualization/draw_geometry.py`

Detection fine-tuning and dataset utilities:

- `src/detection/fintuning/dawn_to_ultralytics.py`
- `src/detection/fintuning/coco_to_yolo.py`
- `src/detection/fintuning/restructure_yolo_dataset.py`
- `src/detection/fintuning/annotation_standarizer.py`
- `src/detection/fintuning/yolo_dataset_analysis.py`
- `src/detection/fintuning/yolo_visualizer.py`
- `src/detection/fintuning/fintuning.ipynb`
- `src/detection/fintuning/testing utilities from previous models.ipynb`
- `src/detection/fintuning/predictions/`

Notebook:

- `notebooks/test_geometry.ipynb`

These files support research, visualization, dataset preparation, or exploratory work. They are not part of the current primary deployable sensor loop unless explicitly wired in later.

## Important Files

```text
config/
  traffic_metrics.yaml          Main runtime configuration
  load_build.py                 Config-to-runtime area/geometry/metric assembly
  cam01_homography.yaml         Example homography calibration

scripts/
  run_sensor.py                 Direct CLI runtime entry point
  sensor_daemon.py              MQTT-controlled daemon entry point
  sensor_pipeline.py            Primary period/frame processing pipeline

src/
  communication/
    topics.py                   MQTT topic hierarchy
    payloads.py                 Metrics/status JSON payload builders
    mqtt_client.py              Paho MQTT wrapper and offline flush
    offline_queue.py            JSON-backed offline publish queue
    services.py                 Metrics, configuration, and snapshot services
  crossing/
    crossing_estimation.py      Stateful crossing extraction
  detection/
    interface.py                Detector contract
    factory.py                  Detector selection
    ultralytics_detectors.py    YOLO/RTDETR detector adapter
    torchvision_detectors.py    Torchvision detector adapter
  geometry/
    primitives.py               Lines, polygons, areas, geometry engine
    homography.py               Calibration and projection utilities
  tracking/
    track.py                    BoxMOT tracker wrapper
  traffic_metrics/
    metrics.py                  Metric result dataclasses and metric implementations
  video_io/
    frame_producer.py           Frame producers and OpenCV acquisition
  utils/
    helper_functions.py         Polygon mask and bbox-in-polygon helpers
    profilers.py                CPU/CUDA timing helper
    shape_setter.py             Interactive geometry selection tools

deployment/root:
  requirements.txt              Large development/runtime environment pin list
  requirements_deploy.txt       Smaller deployment dependency list
  smart_sensor.service          systemd daemon example
  restart_sensor.sh             Interactive service configuration/restart helper
  setup.py                      Package metadata, reads requirements_deploy.txt
```

## Common Modification Targets

| Change | Primary Files / Modules | Notes |
|---|---|---|
| Change main processing order | `scripts/sensor_pipeline.py` | Preserve shared detector/tracker/geometry results unless intentionally redesigning the architecture. |
| Add or change config keys | `config/traffic_metrics.yaml`, `config/load_build.py` | There is no full config schema; add validation where needed. |
| Add detector | `src/detection/interface.py`, `src/detection/factory.py`, detector implementation file | Must return canonical `(N, 6)` detections. |
| Change model path rules | `src/detection/factory.py` | Ultralytics weights are currently assumed under `models/<model_name>.pt`. |
| Add tracker support | `src/tracking/track.py`, `config/traffic_metrics.yaml` | BoxMOT method must be supported by `get_tracker_config()`. |
| Add traffic metric | `src/traffic_metrics/metrics.py`, `config/load_build.py`, `scripts/sensor_pipeline.py` | Metric registry alone is insufficient; compute dispatch is explicit. |
| Change crossing logic | `src/crossing/crossing_estimation.py` | This class owns per-line, per-track temporal crossing state. |
| Change ROI/line behavior | `src/geometry/primitives.py`, `src/utils/helper_functions.py`, `config/load_build.py` | Preserve vectorized masks for real-time use. |
| Wire frame sampling into runtime | `scripts/sensor_pipeline.py`, `src/video_io/frame_producer.py`, `config/traffic_metrics.yaml` | Config contains sampling-like keys but current runtime does not consume them. |
| Change MQTT topics | `src/communication/topics.py`, `scripts/sensor_daemon.py`, `src/communication/services.py` | Keep topic and handler registration synchronized. |
| Change metric payload shape | `src/communication/payloads.py`, `src/communication/services.py` | Payload conversion handles dataclasses and NumPy values. |
| Change deployment command | `smart_sensor.service`, `restart_sensor.sh`, `requirements_deploy.txt` | Service currently does not source `/etc/default/smart_sensor`. |

## Architectural Invariants

- Detector implementations must produce `(N, 6)` arrays ordered as `[x1, y1, x2, y2, score, class_id]`.
- The tracker output layout assumed by `extract_tracking_outputs()` must remain compatible with `tracks[:, :4]`, `tracks[:, 4]`, and `tracks[:, 6]`.
- Per-frame detection and tracking should remain shared upstream work for metrics, not rerun separately per metric.
- Geometry line ordering must stay consistent between `GeometryEngine` and `CrossingExtractor`.
- Area metric objects are stateful; rebuild/reset behavior at period boundaries affects reported averages.
- Polygon IDs currently equal area IDs when built from `areas[].zone`.
- Metric computation assumes masks align with the current frame's track array length.
- Runtime config remains separate from detector/tracker/metric implementation code, even though not all config keys are currently wired.
- Frame copies should be avoided in the hot path; `LatestFrameStore` intentionally copies only for snapshot safety.

## Known Limitations / Technical Debt

- `config/traffic_metrics.yaml` contains `frame_grabber` settings, but the primary runtime hard-codes `DirectFrameProducer`.
- `Area.enable` is stored but not used to skip disabled areas.
- Tracker, crossing, and metric state reset every period, so track continuity and averages do not span period boundaries.
- `CrossingExtractor.update()` calls `np.vstack()` on polygon masks and will fail if there are no polygons in `polygon_cache`.
- `visualization/draw_geometry.py` imports `config.loader`, but the repository contains `config/load_build.py`; it also expects `cfg["polygons"]`, while the main config stores zones under `areas`.
- `CircularFrameBuffer.dropped_frames` returns `_dropped_frames`, but that attribute is not initialized.
- `UltralyticsDetector` references `DetectorError` without importing it.
- `SpaceHeadwayMetric` normalizes the supplied direction and can fail for a zero vector.
- `smart_sensor.service` hard-codes source, sensor ID, broker host, and paths. `restart_sensor.sh` writes `/etc/default/smart_sensor`, but the service file does not consume it.
- `src/communication/services.py` contains duplicated imports and hard-coded Windows defaults in some helper services.
- `build_status_payload()` and `SensorTopics.status` exist, but no active status publishing loop was found.
- Homography and `SpeedLinePair` exist but are not integrated into the primary metrics pipeline.
- `README.md` is empty.
- There is no committed pytest suite; `.pytest_cache` exists but tests are not present in the source tree.
- The repository contains generated prediction label artifacts under `src/detection/fintuning/predictions/`.

## Suggested Verification

After source changes, use static or import checks appropriate to the touched area. The following commands are supported by repository files but were not run while creating this document:

```powershell
$env:PYTHONPATH = "$PWD\src;$PWD\scripts;$PWD"
python -c "from config.load_build import load_config, build_areas, build_runtime_components; cfg=load_config('config/traffic_metrics.yaml'); areas=build_areas(cfg); engine,crossing=build_runtime_components(areas,cfg); print(len(areas), len(engine._line_ids))"
```

```powershell
$env:PYTHONPATH = "$PWD\src;$PWD\scripts;$PWD"
python -c "from detection.factory import build_detector; from video_io.frame_producer import DirectFrameProducer; from communication.topics import SensorTopics; print('imports ok')"
```

If `flake8` is installed:

```powershell
flake8 src config scripts visualization
```

Representative runtime smoke checks require model weights and a valid source:

```powershell
$env:PYTHONPATH = "$PWD\src;$PWD\scripts;$PWD"
python scripts\run_sensor.py --source path\to\video.mp4 --config config\traffic_metrics.yaml --period-mins 0.1
```

MQTT daemon smoke check requires a broker:

```powershell
$env:PYTHONPATH = "$PWD\src;$PWD\scripts;$PWD"
python scripts\sensor_daemon.py --source 0 --mqtt-broker-host <broker-host> --sensor-id camera_1
```

For metric changes, add focused deterministic checks for:

- Counter TTL and duplicate-ID behavior.
- Crossing sign-change and TTL fallback behavior.
- Geometry output shapes for empty and non-empty track arrays.
- Polygon membership masks.
- Per-period metric reset behavior.

For deployment changes, verify:

- `smart_sensor.service` paths and environment match the target machine.
- `requirements_deploy.txt` matches the Jetson/PyTorch/CUDA installation strategy.
- MQTT broker host, sensor ID, and source are supplied consistently.
