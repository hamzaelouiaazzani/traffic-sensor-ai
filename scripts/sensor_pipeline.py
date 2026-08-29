from threading import Event
from typing import Optional

from config.load_build import (
    build_areas,
    build_geometry_engine,
    get_input_fps,
    get_input_source,
    load_config,
)
from communication.mqtt_client import SmartSensorMqttClient
from communication.services import LatestFrameStore, MetricsPublisherService
from communication.topics import SensorTopics
from detection.factory import build_detector
from runtime.continuity import AnalyticsContinuityContext
from runtime.periods import (
    resolve_period_policy,
    resolve_period_seconds as resolve_policy_period_seconds,
)
from runtime.stage1 import (
    extract_tracking_outputs as _extract_tracking_outputs,
    process_stage1_period,
)
from runtime.timing import TimingPolicy
from tracking.track import Tracker
from traffic_metrics.engine import PeriodAnalyticsEngine
from utils.profilers import Profile
from video_io.frame_producer import (
    DirectFrameProducer,
    OfflineSampledFrameProducer,
    RealTimeSimulationProducer,
)


def extract_tracking_outputs(tracks):
    return _extract_tracking_outputs(tracks)


def build_runtime_state(cfg: dict, fps: float, detector):
    areas = build_areas(cfg, num_classes=detector.num_classes)
    geometry_engine = build_geometry_engine(areas, cfg)

    tracker = Tracker(
        method=cfg["tracker"]["method"],
        reid_model=cfg["tracker"]["reid_model"],
        classes=cfg["tracker"]["classes"],
        device=cfg["tracker"]["device"],
        half=cfg["tracker"]["half"],
        per_class=cfg["tracker"]["per_class"],
    )

    timing_policy = TimingPolicy.from_config(cfg, fps_override=fps)
    analytics_engine = PeriodAnalyticsEngine(
        areas=areas,
        geometry_engine=geometry_engine,
        cfg=cfg,
        num_classes=detector.num_classes,
    )
    continuity = AnalyticsContinuityContext()

    return tracker, timing_policy, analytics_engine, continuity


def build_frame_producer(source: str, cfg: dict, fps: float):
    frame_cfg = cfg.get("frame_processing", {})
    sampling_cfg = frame_cfg.get("sampling", {})
    sampling_enabled = bool(sampling_cfg.get("enabled", False))
    producer_type = frame_cfg.get("producer", "direct")

    if producer_type == "direct" and sampling_enabled:
        producer_type = "sampled_offline"

    if producer_type == "direct":
        return DirectFrameProducer(source)

    if producer_type == "sampled_offline":
        effective_fps, sampling_type = resolve_sampling_settings(sampling_cfg, fps)
        return OfflineSampledFrameProducer(
            source=source,
            fps=fps,
            effective_fps=effective_fps,
            sampling_type=sampling_type,
            window_size=int(sampling_cfg.get("window_size") or 30),
        )

    if producer_type == "realtime_simulation":
        return RealTimeSimulationProducer(source)

    raise ValueError(
        f"Unsupported frame_processing.producer '{producer_type}'. "
        "Supported values: direct, sampled_offline, realtime_simulation."
    )


def resolve_sampling_settings(sampling_cfg: dict, fps: float):
    policy = sampling_cfg.get("policy", "every_frame")
    stride = int(sampling_cfg.get("stride", 1))
    if stride <= 0:
        raise ValueError("frame_processing.sampling.stride must be positive")

    if policy in {"every_frame", "none"}:
        return float(fps), "deterministic"

    if policy in {"fixed_stride", "periodic", "periodic_stride", "periodic_sampling"}:
        return float(sampling_cfg.get("effective_fps") or (fps / stride)), "deterministic"

    if policy in {"burst", "burst_sampling"}:
        return float(sampling_cfg.get("effective_fps") or (fps * max(stride - 1, 1) / stride)), "deterministic"

    if policy in {"random", "random_sampling"}:
        effective_fps = float(sampling_cfg.get("effective_fps") or fps)
        return effective_fps, "stochastic"

    raise ValueError(f"Unsupported frame_processing.sampling.policy '{policy}'")


def process_period(
    producer,
    detector,
    tracker,
    timing_policy: TimingPolicy,
    analytics_engine: PeriodAnalyticsEngine,
    continuity: AnalyticsContinuityContext,
    period_seconds: float,
    max_observations: int,
    period_idx: int,
    device,
    stop_event: Optional[Event] = None,
    latest_frame_store: Optional[LatestFrameStore] = None,
):
    analytics_profile = Profile()
    stage1_result = process_stage1_period(
        producer=producer,
        detector=detector,
        tracker=tracker,
        timing_policy=timing_policy,
        period_seconds=period_seconds,
        max_observations=max_observations,
        period_idx=period_idx,
        continuity_policy=analytics_engine.continuity_policy,
        boundary_context=continuity.boundary_batch,
        device=device,
        stop_event=stop_event,
        latest_frame_store=latest_frame_store,
    )
    batch = stage1_result.batch

    if batch is None:
        return {
            "period_idx": period_idx,
            "start_frame": None,
            "end_frame": None,
            "frames_processed": 0,
            "source_frames_elapsed": 0,
            "end_of_stream": stage1_result.end_of_stream,
            "area_metrics": None,
        }

    with analytics_profile:
        area_metrics = analytics_engine.compute_period(batch, continuity)

    return {
        "period_idx": period_idx,
        "start_frame": batch.start_frame,
        "end_frame": batch.end_frame,
        "frames_processed": stage1_result.frames_processed,
        "source_frames_elapsed": batch.source_frames_elapsed,
        "end_of_stream": stage1_result.end_of_stream,
        "area_metrics": area_metrics,
        "processing_stats": {
            "observations": batch.active_observation_count,
            "context_observations": int(batch.observation_count - batch.active_observation_count),
            "timing_mode": timing_policy.mode,
            "period_duration_seconds": batch.duration_seconds,
            "detection_seconds": stage1_result.detection_seconds,
            "tracking_seconds": stage1_result.tracking_seconds,
            "analytics_seconds": analytics_profile.t,
        },
    }


def run_sensor(
    source: Optional[str] = None,
    config_path: str = "config/traffic_metrics.yaml",
    fps: Optional[float] = None,
    period_mins: Optional[float] = None,
    sensor_id: str = "camera_1",
    mqtt_client: Optional[SmartSensorMqttClient] = None,
    stop_event: Optional[Event] = None,
    latest_frame_store: Optional[LatestFrameStore] = None,
) -> None:
    cfg = load_config(config_path)

    source = resolve_source(cfg, source)
    fps = resolve_fps(cfg, fps)
    capacity_fps = resolve_capacity_fps(cfg, fps)
    period_policy = resolve_period_policy(cfg, capacity_fps, period_mins)
    period_seconds = period_policy.period_seconds
    max_observations = period_policy.max_observations

    detector = build_detector(
        model_name=cfg["detector"]["model_name"],
        conf=cfg["detector"]["confidence"],
    )
    device = resolve_detector_device(detector)

    tracker, timing_policy, analytics_engine, continuity = build_runtime_state(cfg, fps, detector)

    producer = build_frame_producer(source, cfg, fps)
    producer.start()

    metrics_publisher = None
    if mqtt_client is not None:
        topics = SensorTopics(sensor_id=sensor_id)
        metrics_publisher = MetricsPublisherService(mqtt_client, topics)

    period_idx = 0

    try:
        while True:
            result = process_period(
                producer=producer,
                detector=detector,
                tracker=tracker,
                timing_policy=timing_policy,
                analytics_engine=analytics_engine,
                continuity=continuity,
                period_seconds=period_seconds,
                max_observations=max_observations,
                period_idx=period_idx,
                device=device,
                stop_event=stop_event,
                latest_frame_store=latest_frame_store,
            )

            if result["frames_processed"] == 0:
                break

            print(result)
            if metrics_publisher is not None:
                metrics_publisher.publish_period_metrics(result)

            period_idx += 1

            if stop_event is not None and stop_event.is_set():
                break

            if result["end_of_stream"]:
                break

    except KeyboardInterrupt:
        pass

    finally:
        producer.release()
        if hasattr(detector, "close"):
            detector.close()


def resolve_source(cfg: dict, source_override: Optional[str]) -> str:
    if source_override is not None:
        return source_override
    return get_input_source(cfg)


def resolve_fps(cfg: dict, fps_override: Optional[float]) -> float:
    fps_value = fps_override or get_input_fps(cfg)
    fps_value = float(fps_value)
    if fps_value <= 0:
        raise ValueError("FPS must be positive")
    return fps_value


def resolve_period_seconds(cfg: dict, period_mins_override: Optional[float]) -> float:
    return resolve_policy_period_seconds(cfg, period_mins_override)


def resolve_capacity_fps(cfg: dict, fps: float) -> float:
    frame_cfg = cfg.get("frame_processing", {})
    sampling_cfg = frame_cfg.get("sampling", {})
    sampling_enabled = bool(sampling_cfg.get("enabled", False))
    producer_type = frame_cfg.get("producer", "direct")

    if producer_type == "sampled_offline" or (producer_type == "direct" and sampling_enabled):
        effective_fps, _ = resolve_sampling_settings(sampling_cfg, fps)
        return effective_fps

    return float(fps)


def resolve_detector_device(detector):
    if hasattr(detector, "predictor") and hasattr(detector.predictor, "device"):
        return detector.predictor.device
    return getattr(detector, "device", None)
