from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class CoordinateTransformError(Exception):
    """Raised for coordinate-transform adapter or binding failures."""


class IPlanarCoordinateTransformer(ABC):
    """
    Stable interface for vectorized 2D planar coordinate transformation.

    Stage 2 depends on this contract only. Candidate implementations may use
    homography, another calibrated planar mapping, or a custom mapping, but
    must expose these vectorized services without leaking implementation
    details to geometry processing or traffic estimators.

    Implementations must accept point arrays whose final dimension is 2:
        (..., 2)

    The same operation is used for vehicle points, polygon vertices, line
    endpoints, or any other planar point collection.
    """

    @abstractmethod
    def image_to_world(self, points: np.ndarray) -> np.ndarray:
        """Map image-space point arrays with final dimension 2 to world space."""
        raise NotImplementedError

    @abstractmethod
    def world_to_image(self, points: np.ndarray) -> np.ndarray:
        """Map world-space point arrays with final dimension 2 to image space."""
        raise NotImplementedError


def validate_coordinate_transformer_contract(transformer: Any) -> IPlanarCoordinateTransformer:
    """Validate the stable coordinate-transform contract used by Stage 2."""

    required_methods = (
        "image_to_world",
        "world_to_image",
    )
    missing = [
        name for name in required_methods
        if not callable(getattr(transformer, name, None))
    ]
    if missing:
        raise CoordinateTransformError(
            "coordinate transformer does not satisfy the runtime contract: "
            + ", ".join(missing)
        )

    return transformer
