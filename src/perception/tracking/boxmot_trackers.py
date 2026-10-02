from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from boxmot.tracker_zoo import (
    create_tracker,
    get_tracker_config,
)
from boxmot.trackers.bytetrack.basetrack import (
    BaseTrack,
)

from perception.tracking.interface import ITracker


class BoxMOTTracker(ITracker):
    """
    BoxMOT tracker adapter behind the stable tracker interface.
    """

    def __init__(
        self,
        method: str = "ocsort",
        reid_model: str = "osnet_x0_25_market1501.pt",
        classes: Optional[List[int]] = None,
        device: str = "",
        half: bool = False,
        per_class: bool = True,
    ):
        BaseTrack.clear_count()

        self.method = method
        self.classes = classes
        self.tracker = create_tracker(
            tracker_type=method,
            tracker_config=get_tracker_config(method),
            reid_weights=Path(reid_model),
            device=device,
            half=half,
            per_class=per_class,
        )

    def update(
        self,
        detections: np.ndarray,
        frame: Any,
    ) -> np.ndarray:
        if self.classes is not None:
            cls_ids = detections[:, 5].astype(int)
            mask = np.isin(cls_ids, self.classes)
            detections = detections[mask]

        return self.tracker.update(detections, frame)
