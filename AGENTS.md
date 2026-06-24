# AGENTS.md



## Codex Working Rules

### Repository Understanding

* Before responding to any request, read AGENTS.md.
* Use AGENTS.md as the primary source of repository-wide knowledge.
* Do not scan the entire repository unless necessary.
* Only open files directly related to the current task.

### Code Modification

* Prefer modifying existing code over creating new files.
* Maintain consistency with existing architecture and coding style.
* Avoid introducing duplicate functionality.

### Execution Policy

* Do NOT run scripts, tests, benchmarks, training jobs, or experiments unless explicitly requested.
* Perform static code analysis only.
* When proposing changes, explain the reasoning before implementation.

### Performance Awareness

* Be aware that some classes maintain state across frames and across execution time.
* Before modifying a class, analyze whether it contains cached data, accumulated metrics, queues, buffers, frame history, trackers, counters, or other persistent state.
* Preserve stateful behavior unless explicitly instructed otherwise.

### Smart Sensor Specific Rules

* Target platform: Jetson Orin Nano.
* Optimize for memory efficiency and real-time performance.
* Avoid unnecessary frame copies.
* Consider CPU, GPU, memory, and I/O implications of changes.
* Detection → Tracking → Counting → Metrics pipeline must remain consistent.

### Analysis Workflow

1. Read AGENTS.md.
2. Identify relevant files.
3. Read only those files.
4. Analyze dependencies.
5. Propose changes.
6. Implement changes.
7. Do not execute code unless explicitly requested.




## Project Purpose

This repository is a Python computer-vision project for road traffic monitoring. It combines object detection, multi-object tracking, geometry filtering, line-crossing estimation, homography calibration, and traffic indicator computation.

The package name in `setup.py` is `traffic-vision-ai`. The intended runtime is a video analytics pipeline that:

1. Reads frames from videos, webcams, or streams.
2. Runs vehicle/object detection with Ultralytics or Torchvision models.
3. Feeds detections into a BoxMOT tracker.
4. Computes geometric relationships against configured lines and zones.
5. Extracts line crossings and traffic metrics such as flow, density, occupancy, time headway, and space headway.

## Repository Structure

```text
.
├── config/
│   ├── cam01_homography.yaml
│   ├── load_build.py
│   └── traffic_metrics.yaml
├── inputs/
├── models/
├── notebooks/
│   └── test_geometry.ipynb
├── scripts/
│   └── run_geometry.py
├── src/
│   ├── crossing/
│   │   └── crossing_estimation.py
│   ├── detection/
│   │   ├── factory.py
│   │   ├── interface.py
│   │   ├── torchvision_detectors.py
│   │   ├── ultralytics_detectors.py
│   │   └── fintuning/
│   ├── geometry/
│   │   ├── homography.py
│   │   └── primitives.py
│   ├── tracking/
│   │   └── track.py
│   ├── traffic_metrics/
│   │   └── metrics.py
│   ├── utils/
│   │   ├── helper_functions.py
│   │   ├── profilers.py
│   │   └── shape_setter.py
│   └── video_io/
│       └── frame_producer.py
├── visualization/
│   └── draw_geometry.py
├── requirements.txt
├── setup.cfg
└── setup.py
```

Notes:

- `README.md` currently exists but is empty.
- The working tree may contain ongoing renames from `configs/` and `src/io/` to `config/` and `src/video_io/`. Do not revert unrelated changes.
- Model weights, datasets, generated media, run outputs, and caches are intentionally ignored by `.gitignore`.

## Key Components

### Configuration Builder

- File: `config/load_build.py`
- Responsibilities:
  - Load YAML configuration with `yaml.safe_load`.
  - Build `Line`, `Polygon`, and `Area` objects.
  - Determine which metrics are eligible for each area.
  - Build metric objects.
  - Build a `GeometryEngine` and `CrossingExtractor`.

The main config file is `config/traffic_metrics.yaml`. It defines frame source settings, detector settings, tracker settings, line geometry, monitoring areas, metric settings, and general runtime parameters.

### Geometry

- File: `src/geometry/primitives.py`
- Core classes:
  - `Line`: validated two-point line with cached canonical equation coefficients.
  - `Polygon`: validated polygon with cached mask, bounding box, and area.
  - `Area`: logical monitoring area combining a zone, a flow line, and metrics.
  - `GeometryEngine`: vectorized computation of line distance/sign/vicinity masks and polygon membership masks.

Geometry inputs are pixel coordinates. `GeometryEngine.compute()` expects object center points and optionally bounding boxes. It returns:

- `line_cache`: `distance`, `sign`, and `vicinity_mask` arrays.
- `polygon_cache`: mapping from polygon ID to boolean masks.

### Homography

- File: `src/geometry/homography.py`
- Provides calibration and projection helpers:
  - `compute_homography()`
  - `load_calibration()`
  - `save_calibration_yaml()`
  - `Homography.project_pixels_to_world()`
  - `Homography.project_world_to_pixels()`
  - `Homography.project_bboxes_to_world()`
  - `Homography.warp_to_birdseye()`

The sample calibration is stored in `config/cam01_homography.yaml`.

### Detection

- Files:
  - `src/detection/interface.py`
  - `src/detection/factory.py`
  - `src/detection/ultralytics_detectors.py`
  - `src/detection/torchvision_detectors.py`

All detectors must follow the `IDetector` interface. The canonical detector output is:

```text
np.ndarray shape (N, 6)
[x1, y1, x2, y2, score, class_id]
```

`build_detector()` chooses:

- Ultralytics models for names containing `yolo` or `rtdetr`; weights are expected at `models/<model_name>.pt`.
- Torchvision detection models for supported model names such as `fasterrcnn_resnet50_fpn`, `retinanet_resnet50_fpn`, and `ssd300_vgg16`.

### Tracking

- File: `src/tracking/track.py`
- Wraps BoxMOT trackers via `boxmot.tracker_zoo.create_tracker`.
- Default method is `ocsort`.
- Optional class filtering is applied before tracker update.

Tracker input should be the detector canonical `(N, 6)` array. Tracker output is whatever the selected BoxMOT tracker returns.

### Crossing Estimation

- File: `src/crossing/crossing_estimation.py`
- `CrossingExtractor` keeps per-line, per-track state.
- It detects true crossings via sign changes across a configured line.
- It also handles disappeared objects with TTL fallback if an object was recently near a line.

Inputs include track IDs, current timestamp, `line_cache`, and `polygon_cache`.

### Traffic Metrics

- File: `src/traffic_metrics/metrics.py`
- Metrics implemented:
  - `Counter`
  - `FlowMetric`
  - `DensityMetric`
  - `OccupancyMetric`
  - `TimeHeadwayMetric`
  - `SpaceHeadwayMetric`

Metric classes are stateful. Reuse the same metric object across frames when computing rolling or cumulative values.

### Video IO

- File: `src/video_io/frame_producer.py`
- Provides:
  - `Frame`
  - `CircularFrameBuffer`
  - `FrameGrabber`
  - `FrameSampler`
  - `FrameProducer`
  - `RealTimeSimulationProducer`
  - `OfflineSampledFrameProducer`
  - `DirectFrameProducer`

This module supports direct frame reads, threaded production, offline sampled processing, and simulated real-time frame dropping.

### Visualization

- File: `visualization/draw_geometry.py`
- Intended to draw configured lines and polygons on frames.
- Current implementation imports `config.loader`, but the present config builder file is `config/load_build.py`. Treat this as a known drift before relying on the script.

### Finetuning Utilities

- Directory: `src/detection/fintuning/`
- Contains dataset conversion, annotation standardization, YOLO visualization, and dataset analysis helpers.
- The directory also contains notebooks and raw prediction labels. Be careful with large data artifacts and generated outputs.

## Environment Setup

Recommended setup from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For local imports, either install the package in editable mode or set `PYTHONPATH`:

```powershell
pip install -e .
```

or:

```powershell
$env:PYTHONPATH = "$PWD\src;$PWD"
```

Important: `setup.py` reads `requirements.txt` directly into `install_requires`. Because `requirements.txt` contains comments and blank lines, editable install may fail in some packaging environments. If that happens, prefer `pip install -r requirements.txt` plus `PYTHONPATH`, or clean requirement parsing in `setup.py`.

## Dependencies

Declared in `requirements.txt`:

- Core numeric/scientific: `numpy`, `scipy`, `pandas`, `scikit-learn`
- Computer vision: `opencv-python`, `pillow`
- Deep learning: `torch`, `torchvision`, `tensorboard`
- Detection/support tooling: `lapx`, `ultralytics-thop`, `yacs`, `gdown`, `GitPython`
- Visualization/analysis: `matplotlib`, `seaborn`, `tqdm`
- Utilities: `PyYAML`, `requests`, `loguru`, `psutil`, `py-cpuinfo`, `regex`, `ftfy`, `pre-commit`

Imported but not currently declared in `requirements.txt`:

- `pydantic` is required by `src/geometry/primitives.py`.
- `ultralytics` is required by `src/detection/ultralytics_detectors.py`.
- `boxmot` is required by `src/tracking/track.py`.
- `flake8` is configured in `setup.cfg` but not declared in `requirements.txt`.
- `pytest` is useful for test runs but no committed pytest suite currently exists.

Install missing runtime dependencies explicitly when needed:

```powershell
pip install pydantic ultralytics boxmot flake8 pytest
```

GPU workflows require a compatible PyTorch/CUDA installation. Follow the official PyTorch install selector for the target CUDA version rather than assuming the generic `requirements.txt` wheel is correct.

## Build and Run Commands

### Import/Smoke Checks

From the repository root:

```powershell
$env:PYTHONPATH = "$PWD\src;$PWD"
python -c "from geometry.primitives import Line, Polygon, GeometryEngine; print('geometry ok')"
python -c "from config.load_build import load_config, build_areas, build_runtime_components; print('config builder ok')"
```

### Build Runtime Components

```powershell
$env:PYTHONPATH = "$PWD\src;$PWD"
python -c "from config.load_build import load_config, build_areas, build_runtime_components; cfg=load_config('config/traffic_metrics.yaml'); areas=build_areas(cfg); engine,crossing=build_runtime_components(areas,cfg); print(len(areas), len(engine._line_ids))"
```

### Geometry Demo Script

The intended command shape is:

```powershell
$env:PYTHONPATH = "$PWD\src;$PWD"
python scripts\run_geometry.py --video path\to\video.mp4 --config config\traffic_metrics.yaml --model models\yolo11n_finetuned.pt
```

Known issue: `scripts/run_geometry.py` and `visualization/draw_geometry.py` currently import `config.loader`, but the repository contains `config/load_build.py`. Fix the import path or provide a compatible `config/loader.py` before using this script.

### Linting

`setup.cfg` configures flake8:

```powershell
flake8 src config scripts visualization
```

Configured conventions include:

- Maximum line length: 120
- Selected checks: `E`, `W`, `F`
- Ignored checks: `E731`, `F405`, `E402`, `W504`, `W605`, `E741`

## Testing Procedures

There is no committed Python unit test suite at the time this file was generated. Current testing appears to be exploratory/manual, mainly through notebooks such as `notebooks/test_geometry.ipynb`.

Recommended validation for code changes:

1. Run import smoke checks for touched modules.
2. Run `flake8 src config scripts visualization` if `flake8` is installed.
3. For geometry changes, add or run small deterministic checks for:
   - invalid line and polygon validation
   - polygon mask membership
   - `GeometryEngine.compute()` output shapes
   - line sign changes and crossing extraction
4. For detector changes, verify canonical `(N, 6)` output with an image/frame.
5. For tracker changes, verify compatibility with BoxMOT and class filtering.
6. For metrics changes, test stateful behavior across multiple frames/timestamps.

Suggested future pytest layout:

```text
tests/
├── test_geometry_primitives.py
├── test_geometry_engine.py
├── test_crossing_estimation.py
├── test_metrics.py
└── test_config_build.py
```

## Coding Conventions

- Keep source packages under `src/`.
- Prefer absolute package imports that work with `PYTHONPATH=src`, for example `from geometry.primitives import Line`.
- Preserve the detector output contract: `(N, 6)` with `[x1, y1, x2, y2, score, class_id]`.
- Use vectorized NumPy operations for per-frame geometry and metric paths.
- Keep runtime metric and crossing classes stateful when they represent temporal state.
- Validate geometry inputs early. Existing geometry models use Pydantic validators.
- Avoid hard-coding local machine paths in reusable code. `config/traffic_metrics.yaml` currently contains an absolute Windows video path and should be overridden for other machines.
- Do not commit model weights, videos, generated images, datasets, run outputs, or caches.
- Be cautious with notebooks: avoid committing large output cells unless intentionally preserving an experiment.

## Configuration Notes

`config/traffic_metrics.yaml` includes:

- `frame_grabber`: source path, stride/sampling options, queue size, fallback FPS, dynamic resolution flag.
- `detector`: model name and confidence.
- `tracker`: BoxMOT method, ReID model path, device, half precision, class filtering.
- `lines`: line IDs, endpoints, and normalized vicinity thresholds.
- `areas`: area IDs, names, enabled flags, metrics, optional polygon zones, and flow line IDs.
- `metrics.flow`: counter logic and TTL.
- `general_params`: frame size, FPS, class count, and crossing TTL.

Metric eligibility is derived from available geometry:

- `flow` requires a flow line.
- `density` requires a zone.
- `occupancy`, `space_headway`, and `time_headway` require both zone and flow line.

## Deployment and Runtime Expectations

There is no packaged service, Dockerfile, CI workflow, or deployment manifest in the repository.

A practical deployment is currently a Python environment running the pipeline directly on a machine with:

- Python 3.8 or newer.
- OpenCV access to the target video source, camera, or RTSP stream.
- Model weights under `models/`.
- Correct YAML geometry/calibration for the camera.
- Optional CUDA-capable GPU for real-time deep learning performance.
- Compatible PyTorch, Ultralytics, BoxMOT, and tracker/ReID weights.

Before deploying to a new camera or scene:

1. Calibrate homography if world-coordinate metrics are needed.
2. Update line and polygon coordinates in `config/traffic_metrics.yaml`.
3. Validate detector class IDs against `general_params.num_classes`.
4. Confirm tracker method and ReID weights are available.
5. Run a short video sample and inspect geometry overlays and metric outputs.

## Important Constraints and Known Issues

- `README.md` is empty, so this file is currently the main operational guide.
- `requirements.txt` is incomplete for current imports: add `pydantic`, `ultralytics`, and `boxmot` as needed.
- `setup.py` may not robustly parse commented `requirements.txt` lines into `install_requires`.
- `scripts/run_geometry.py` and `visualization/draw_geometry.py` reference `config.loader`, which is not present in the current tree.
- `visualization/draw_geometry.py` expects `cfg["polygons"]`, while `config/traffic_metrics.yaml` defines polygons under `areas[].zone`. Update this before relying on visualization output.
- `CircularFrameBuffer.dropped_frames` references `_dropped_frames`, but that attribute is not initialized in the current class.
- `UltralyticsDetector` raises `DetectorError` but does not import it from `detection.interface`.
- `SpaceHeadwayMetric` expects a direction vector that can be normalized; avoid zero vectors.
- `CrossingExtractor.update()` stacks polygon masks with `np.vstack`; handle empty polygon caches if adding line-only workflows.
- Many defaults assume Windows paths and CUDA (`tracker.device: cuda`, `half: true`).
- The repository contains generated prediction labels under `src/detection/fintuning/predictions/`; avoid expanding generated data in source directories.

## Agent Workflow Guidance

When modifying this repository:

1. Inspect the current working tree first. There may be user changes in progress.
2. Keep changes narrowly scoped; do not rewrite the pipeline structure unless asked.
3. Preserve the canonical detector and tracker data contracts.
4. Add focused tests or smoke checks for the module touched.
5. Document any newly required external files, weights, or environment variables.
6. Do not remove or regenerate notebooks, datasets, model files, or prediction artifacts unless explicitly requested.
