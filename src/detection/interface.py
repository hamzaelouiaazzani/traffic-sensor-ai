# detectors/interface.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping, Sequence, Tuple, Union
import numpy as np


class DetectorError(Exception):
    """Raised for detector-specific failures."""
    pass


class IDetector(ABC):
    """
    Unified detector interface aligned with Ultralytics-style detectors.

    Canonical detector output:
        np.ndarray of shape (N, 6)
        [x1, y1, x2, y2, score, class_id]

    All detectors (Ultralytics, torchvision, TensorRT, custom)
    MUST be able to produce this format.

    Metadata contract:
        class_names: backend-independent ordered class-name tuple.
        num_classes: number of classes exposed by the loaded detector model.
    """

    # -------------------------
    # Lifecycle
    # -------------------------

    @abstractmethod
    def __init__(self, model_name: str, **kwargs: Dict[str, Any]):
        """Initialize detector resources (model, device, precision, etc.)."""
        raise NotImplementedError

    @abstractmethod
    def warmup(self, imgsz: Any = None) -> None:
        """Optional warmup to reduce first-inference latency."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Release model / GPU / TensorRT resources."""
        raise NotImplementedError

    # -------------------------
    # Model metadata
    # -------------------------

    @property
    @abstractmethod
    def class_names(self) -> Tuple[str, ...]:
        """Ordered class names as exposed by the loaded detector backend."""
        raise NotImplementedError

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Number of detector classes, derived from class_names."""
        raise NotImplementedError

    # -------------------------
    # Pipeline hooks (logical)
    # -------------------------

    @abstractmethod
    def preprocess(self, array_frame: np.ndarray):
        """
        Prepare input for inference.
        Return type is framework-specific.
        """
        raise NotImplementedError

    @abstractmethod
    def infer(self, preprocessed_input, **kwargs):
        """
        Run model forward pass.
        Return type is framework-specific.
        """
        raise NotImplementedError

    @abstractmethod
    def postprocess(
        self,
        raw_output,
        preprocessed_input,
        array_frame: np.ndarray,
    ) -> np.ndarray:
        """
        Convert raw output to canonical detector format.

        MUST return:
            np.ndarray (N, 6)
            columns = [x1, y1, x2, y2, score, class_id]
        """
        raise NotImplementedError

    # -------------------------
    # Public API (PRIMARY)
    # -------------------------

    @abstractmethod
    def detect_to_track(self, array_frame: np.ndarray, **kwargs) -> np.ndarray:
        """
        Run full detection pipeline on a single frame.

        MUST return:
            np.ndarray (N, 6)
            [x1, y1, x2, y2, score, class_id]

        This output is tracker-ready (e.g., BoxMOT).
        """
        raise NotImplementedError


def normalize_class_names(names: Union[Mapping[int, str], Sequence[str]]) -> Tuple[str, ...]:
    """
    Normalize backend metadata into the IDetector class_names representation.

    Backends commonly expose names either as a sequence indexed by class ID
    or as a class-id keyed mapping. The detector interface uses a tuple so
    downstream code receives an immutable, backend-independent value.
    """
    if isinstance(names, Mapping):
        ordered = [
            str(names[class_id])
            for class_id in sorted(names)
        ]
    else:
        ordered = [str(name) for name in names]

    if not ordered:
        raise DetectorError("detector metadata must contain at least one class name")

    return tuple(ordered)
