import cv2
import numpy as np
from config.loader import load_config


def annotate_geometry(frame: np.ndarray, config_path: str = "config/traffic_metrics.yaml") -> np.ndarray:
    """
    Draw lines + polygons on frame.
    """
    cfg = load_config(config_path)

    lines = cfg["lines"]
    polygons = cfg["polygons"]

    img = frame.copy()

    # --- draw lines
    for line in lines.values():
        (x1, y1), (x2, y2) = line.points
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # --- draw polygons
    for poly in polygons.values():
        pts = np.array(poly.points, dtype=np.int32)
        cv2.polylines(img, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

    return img