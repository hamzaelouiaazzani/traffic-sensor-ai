# detectors/interface.py
from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence, Tuple, Union
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

    All detector adapters (Ultralytics, torchvision, TensorRT, custom)
    MUST be able to produce this format.

    Metadata contract:
        class_names: backend-independent ordered class-name tuple.
        num_classes: number of classes exposed by the loaded detector model.
    """

    # -------------------------
    # Lifecycle
    # -------------------------

    def __init__(self, model_name: str, **kwargs: Any):
        """Initialize detector resources (model, device, precision, etc.)."""
        super().__init__()

    def warmup(self, imgsz: Any = None) -> None:
        """Optional warmup to reduce first-inference latency."""
        return None

    def close(self) -> None:
        """Release model / GPU / TensorRT resources."""
        return None

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

    def preprocess(self, array_frame: np.ndarray):
        """
        Optional adapter hook to prepare input for inference.

        Downstream runtime code does not call this method directly; custom
        detectors may implement detect_to_track() without exposing framework
        internals here.
        """
        raise NotImplementedError

    def infer(self, preprocessed_input, **kwargs):
        """
        Optional adapter hook to run a framework-specific forward pass.

        Downstream runtime code does not call this method directly.
        """
        raise NotImplementedError

    def postprocess(
        self,
        raw_output,
        preprocessed_input,
        array_frame: np.ndarray,
    ) -> np.ndarray:
        """
        Optional adapter hook to convert raw output to canonical detector format.

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


def validate_detector_contract(detector: Any) -> IDetector:
    """
    Validate the stable detector contract used by downstream runtime code.

    This keeps component replacement focused on the required interface:
    metadata plus canonical tracker-ready detections. Framework-specific hooks
    such as preprocess/infer/postprocess remain adapter internals.
    """
    missing = []
    if not callable(getattr(detector, "detect_to_track", None)):
        missing.append("detect_to_track(frame)")
    if not hasattr(detector, "class_names"):
        missing.append("class_names")
    if not hasattr(detector, "num_classes"):
        missing.append("num_classes")

    if missing:
        raise DetectorError(
            "detector adapter does not satisfy the runtime contract: "
            + ", ".join(missing)
        )

    class_names = detector.class_names
    num_classes = detector.num_classes
    if not isinstance(class_names, tuple):
        raise DetectorError("detector.class_names must be an ordered tuple")
    if not isinstance(num_classes, int) or num_classes <= 0:
        raise DetectorError("detector.num_classes must be a positive integer")
    if len(class_names) != num_classes:
        raise DetectorError("detector.num_classes must match len(detector.class_names)")

    return detector
