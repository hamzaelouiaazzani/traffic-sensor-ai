import importlib
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from perception.detection.interface import DetectorError, validate_detector_contract


TORCHVISION_MODELS = {
    "fasterrcnn_resnet50_fpn",
    "fasterrcnn_resnet50_fpn_v2",
    "retinanet_resnet50_fpn",
    "ssd300_vgg16",
    "ssdlite320_mobilenet_v3_large",
    "fcos_resnet50_fpn",
}

DetectorBuilder = Callable[..., object]
_DETECTOR_BUILDERS: Dict[str, DetectorBuilder] = {}


def register_detector_backend(name: str, builder: DetectorBuilder) -> None:
    """
    Register a detector backend builder behind the stable detector contract.

    This supports component replacement without changing downstream runtime
    code. Builders receive model_name plus backend-specific keyword arguments
    and must return an object satisfying validate_detector_contract().
    """
    normalized = name.strip().lower()
    if not normalized:
        raise DetectorError("detector backend name must be non-empty")
    _DETECTOR_BUILDERS[normalized] = builder


def _load_class(class_path: str):
    if ":" in class_path:
        module_name, class_name = class_path.split(":", 1)
    else:
        module_name, class_name = class_path.rsplit(".", 1)

    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _build_ultralytics_detector(model_name: str, **kwargs):
    from perception.detection.ultralytics_detectors import UltralyticsDetector

    model_path = Path("models") / f"{model_name}.pt"
    return UltralyticsDetector(model_name=str(model_path), **kwargs)


def _build_torchvision_detector(model_name: str, **kwargs):
    from perception.detection.torchvision_detectors import TorchvisionDetector

    return TorchvisionDetector(model_name=model_name, **kwargs)


def _build_custom_detector(model_name: str, class_path: Optional[str] = None, **kwargs):
    if not class_path:
        raise DetectorError(
            "custom detector backend requires detector.class_path "
            "(for example, 'my_package.detectors:MyDetector')"
        )

    detector_cls = _load_class(class_path)
    return detector_cls(model_name=model_name, **kwargs)


register_detector_backend("ultralytics", _build_ultralytics_detector)
register_detector_backend("torchvision", _build_torchvision_detector)
register_detector_backend("custom", _build_custom_detector)


def build_detector(
    model_name: str,
    backend: Optional[str] = None,
    class_path: Optional[str] = None,
    **kwargs
):
    """
    Build a detector adapter through configuration-time binding.

    backend/framework may be supplied explicitly. If omitted, the factory keeps
    the historical model-name inference for Ultralytics and torchvision models.
    """
    backend = backend or kwargs.pop("framework", None) or kwargs.pop("backend", None)
    class_path = class_path or kwargs.pop("class_path", None) or kwargs.pop("custom_class", None)

    if backend is None:
        name = model_name.lower()
        if "yolo" in name or "rtdetr" in name:
            backend = "ultralytics"
        elif name in TORCHVISION_MODELS:
            backend = "torchvision"
        else:
            raise DetectorError(
                f"Unsupported detector '{model_name}'. "
                "Set detector.backend and detector.class_path for custom adapters."
            )

    normalized_backend = backend.strip().lower()
    if normalized_backend not in _DETECTOR_BUILDERS:
        raise DetectorError(f"Unsupported detector backend: {backend}")

    if normalized_backend == "custom":
        detector = _DETECTOR_BUILDERS[normalized_backend](
            model_name=model_name,
            class_path=class_path,
            **kwargs,
        )
    else:
        detector = _DETECTOR_BUILDERS[normalized_backend](
            model_name=model_name,
            **kwargs,
        )
    return validate_detector_contract(detector)


def build_detector_from_config(detector_cfg: Dict[str, Any]):
    backend = detector_cfg.get("backend") or detector_cfg.get("framework")
    normalized_backend = backend.strip().lower() if isinstance(backend, str) else backend
    detector_kwargs = dict(detector_cfg.get("kwargs", {}))
    if normalized_backend != "custom" and "confidence" in detector_cfg and "conf" not in detector_kwargs:
        detector_kwargs["conf"] = detector_cfg["confidence"]

    return build_detector(
        model_name=detector_cfg["model_name"],
        backend=backend,
        class_path=detector_cfg.get("class_path") or detector_cfg.get("custom_class"),
        **detector_kwargs,
    )
