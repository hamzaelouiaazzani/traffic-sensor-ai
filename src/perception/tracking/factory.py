import importlib
from typing import Any, Callable, Dict, Optional

from perception.tracking.interface import TrackerError, validate_tracker_contract


TrackerBuilder = Callable[..., object]
_TRACKER_BUILDERS: Dict[str, TrackerBuilder] = {}


def register_tracker_backend(name: str, builder: TrackerBuilder) -> None:
    """
    Register a tracker backend builder behind the stable tracker contract.

    Builders receive backend-specific keyword arguments and must return an
    object exposing update(detections, frame).
    """
    normalized = name.strip().lower()
    if not normalized:
        raise TrackerError("tracker backend name must be non-empty")
    _TRACKER_BUILDERS[normalized] = builder


def _load_class(class_path: str):
    if ":" in class_path:
        module_name, class_name = class_path.split(":", 1)
    else:
        module_name, class_name = class_path.rsplit(".", 1)

    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _build_boxmot_tracker(**kwargs):
    from perception.tracking.boxmot_trackers import BoxMOTTracker

    return BoxMOTTracker(**kwargs)


def _build_custom_tracker(class_path: Optional[str] = None, **kwargs):
    if not class_path:
        raise TrackerError(
            "custom tracker backend requires tracker.class_path "
            "(for example, 'my_package.trackers:MyTracker')"
        )

    tracker_cls = _load_class(class_path)
    return tracker_cls(**kwargs)


register_tracker_backend("boxmot", _build_boxmot_tracker)
register_tracker_backend("custom", _build_custom_tracker)


def build_tracker(
    backend: str = "boxmot",
    class_path: Optional[str] = None,
    **kwargs,
):
    """Build a tracker adapter through configuration-time binding."""
    backend = backend or kwargs.pop("framework", None) or kwargs.pop("backend", None) or "boxmot"
    class_path = class_path or kwargs.pop("class_path", None) or kwargs.pop("custom_class", None)

    normalized_backend = backend.strip().lower()
    if normalized_backend not in _TRACKER_BUILDERS:
        raise TrackerError(f"Unsupported tracker backend: {backend}")

    if normalized_backend == "custom":
        tracker = _TRACKER_BUILDERS[normalized_backend](class_path=class_path, **kwargs)
    else:
        tracker = _TRACKER_BUILDERS[normalized_backend](**kwargs)
    return validate_tracker_contract(tracker)


def build_tracker_from_config(tracker_cfg: Dict[str, Any]):
    backend = tracker_cfg.get("backend") or tracker_cfg.get("framework") or "boxmot"
    normalized_backend = backend.strip().lower() if isinstance(backend, str) else backend
    tracker_kwargs = dict(tracker_cfg.get("kwargs", {}))
    if normalized_backend != "custom":
        for key in ("method", "reid_model", "classes", "device", "half", "per_class"):
            if key in tracker_cfg and key not in tracker_kwargs:
                tracker_kwargs[key] = tracker_cfg[key]

    return build_tracker(
        backend=backend,
        class_path=tracker_cfg.get("class_path") or tracker_cfg.get("custom_class"),
        **tracker_kwargs,
    )
