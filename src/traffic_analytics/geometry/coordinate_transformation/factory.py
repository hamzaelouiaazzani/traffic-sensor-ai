from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, Optional

from traffic_analytics.geometry.coordinate_transformation.interface import (
    CoordinateTransformError,
    validate_coordinate_transformer_contract,
)


TransformerBuilder = Callable[..., object]
_TRANSFORMER_BUILDERS: Dict[str, TransformerBuilder] = {}


def register_coordinate_transformer_backend(name: str, builder: TransformerBuilder) -> None:
    """
    Register a coordinate-transform backend behind the stable transformer contract.

    Builders receive backend-specific keyword arguments and must return an object
    satisfying validate_coordinate_transformer_contract().
    """

    normalized = name.strip().lower()
    if not normalized:
        raise CoordinateTransformError("coordinate transformer backend name must be non-empty")
    _TRANSFORMER_BUILDERS[normalized] = builder


def _load_class(class_path: str):
    if ":" in class_path:
        module_name, class_name = class_path.split(":", 1)
    else:
        module_name, class_name = class_path.rsplit(".", 1)

    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _build_homography_transformer(calibration_file: Optional[str] = None, **kwargs):
    if not calibration_file:
        raise CoordinateTransformError(
            "homography coordinate transformer requires homography.calibration_file"
        )

    from traffic_analytics.geometry.coordinate_transformation.homography import (
        HomographyTransformer,
    )

    return HomographyTransformer(calibration_file=calibration_file)


def _build_custom_transformer(class_path: Optional[str] = None, **kwargs):
    if not class_path:
        raise CoordinateTransformError(
            "custom coordinate transformer backend requires coordinate_transform.class_path "
            "(for example, 'my_package.transforms:MyTransformer')"
        )

    transformer_cls = _load_class(class_path)
    return transformer_cls(**kwargs)


register_coordinate_transformer_backend("homography", _build_homography_transformer)
register_coordinate_transformer_backend("custom", _build_custom_transformer)


def build_coordinate_transformer(
    backend: str = "homography",
    class_path: Optional[str] = None,
    **kwargs,
):
    """Build a coordinate transformer through configuration-time binding."""

    backend = backend or kwargs.pop("method", None) or kwargs.pop("backend", None) or "homography"
    class_path = class_path or kwargs.pop("class_path", None) or kwargs.pop("custom_class", None)

    normalized_backend = backend.strip().lower()
    if normalized_backend not in _TRANSFORMER_BUILDERS:
        raise CoordinateTransformError(f"Unsupported coordinate transformer backend: {backend}")

    if normalized_backend == "custom":
        transformer = _TRANSFORMER_BUILDERS[normalized_backend](
            class_path=class_path,
            **kwargs,
        )
    else:
        transformer = _TRANSFORMER_BUILDERS[normalized_backend](**kwargs)
    return validate_coordinate_transformer_contract(transformer)


def build_coordinate_transformer_from_config(cfg: Dict[str, Any]):
    """
    Build the configured planar coordinate transformer.

    Current configs remain compatible through the top-level ``homography``
    section. Future methods can be selected through ``coordinate_transform``.
    """

    transform_cfg = cfg.get("coordinate_transform", {})
    homography_cfg = cfg.get("homography", {})
    homography_enabled = bool(homography_cfg.get("enabled", False))

    backend = transform_cfg.get("backend") or transform_cfg.get("method")
    if backend is None:
        if not homography_enabled:
            return None
        backend = "homography"

    normalized_backend = backend.strip().lower() if isinstance(backend, str) else backend
    if normalized_backend == "none":
        return None

    transformer_kwargs = dict(transform_cfg.get("kwargs", {}))
    if normalized_backend == "homography":
        calibration_file = (
            transform_cfg.get("calibration_file")
            or homography_cfg.get("calibration_file")
        )
        transformer_kwargs.setdefault("calibration_file", calibration_file)

    return build_coordinate_transformer(
        backend=backend,
        class_path=transform_cfg.get("class_path") or transform_cfg.get("custom_class"),
        **transformer_kwargs,
    )


def coordinate_transformer_config_available(cfg: Dict[str, Any]) -> bool:
    """Return whether config contains enough information to bind a transformer."""

    transform_cfg = cfg.get("coordinate_transform", {})
    homography_cfg = cfg.get("homography", {})
    backend = transform_cfg.get("backend") or transform_cfg.get("method")

    if backend is None:
        return bool(homography_cfg.get("enabled", False)) and bool(
            homography_cfg.get("calibration_file")
        )

    normalized_backend = backend.strip().lower()
    if normalized_backend == "none":
        return False
    if normalized_backend == "custom":
        return bool(transform_cfg.get("class_path") or transform_cfg.get("custom_class"))
    if normalized_backend == "homography":
        return bool(transform_cfg.get("calibration_file") or homography_cfg.get("calibration_file"))

    return False
