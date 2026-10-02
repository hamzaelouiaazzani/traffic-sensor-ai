from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class TrackerError(Exception):
    """Raised for tracker adapter or binding failures."""
    pass


class ITracker(ABC):
    """
    Stable tracker interface used by the perception runtime.

    This is the only behavior Stage 1 depends on. Tracker frameworks such as
    BoxMOT must be adapted to this contract; downstream runtime code should not
    import framework-specific tracker modules directly.

    Canonical detector input:
        np.ndarray of shape (N, 6)
        [x1, y1, x2, y2, score, class_id]

    Expected tracker output for current downstream observation extraction:
        np.ndarray with at least seven columns:
        [x1, y1, x2, y2, track_id, ..., class_id, ...]
    """

    @abstractmethod
    def update(self, detections: np.ndarray, frame: Any) -> np.ndarray:
        """
        Update tracker state for one frame and return active tracks.

        Implementations may maintain persistent identity state across calls.
        The returned track array must preserve the columns consumed by
        perception.pipeline.extract_tracking_outputs().
        """
        raise NotImplementedError

    def close(self) -> None:
        """Release optional tracker resources."""
        return None


def validate_tracker_contract(tracker: Any) -> ITracker:
    """Validate the stable tracker contract used by runtime code."""
    if not callable(getattr(tracker, "update", None)):
        raise TrackerError("tracker adapter must expose update(detections, frame)")
    return tracker
