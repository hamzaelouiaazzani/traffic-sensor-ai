import cv2
import numpy as np
from config.load_build import load_config


def annotate_geometry(frame: np.ndarray, config_path: str = "config/traffic_metrics.yaml") -> np.ndarray:
    """
    Draw lines + polygons on frame.
    """
    cfg = load_config(config_path)

    geometry_cfg = cfg["geometry"]
    lines = geometry_cfg.get("lines", [])
    areas = geometry_cfg.get("areas", [])

    img = frame.copy()

    # --- draw lines
    for line in lines:
        (x1, y1), (x2, y2) = line["points"]
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # --- draw polygons
    for area in areas:
        zone = area.get("zone")
        if zone is None:
            continue
        pts = np.array(zone["points"], dtype=np.int32)
        cv2.polylines(img, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

    return img
