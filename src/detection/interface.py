# detectors/interface.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
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

