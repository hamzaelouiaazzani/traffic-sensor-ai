# geometry/coordinate_transformation/homography.py
from __future__ import annotations
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import cv2

from traffic_analytics.geometry.coordinate_transformation.interface import (
    IPlanarCoordinateTransformer,
)

try:
    import yaml
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False


Pixel = Tuple[float, float]    # image-space (x, y) pixels
World = Tuple[float, float]    # world-space (X, Y) meters


@dataclass(frozen=True)
class Calibration:
    """
    Immutable container for a calibration/homography.
    - homography: 3x3 np.ndarray mapping image-space pixels -> world-space meters
    - homography_inv: inverse world-space -> image-space mapping
    - units: usually "meters"
    - timestamp: epoch seconds when created
    - source: string describing how this calibration was produced
    - rmse: reprojection error (meters)
    - version: semantic-ish version string
    """
    homography: np.ndarray
    homography_inv: np.ndarray
    units: str
    timestamp: float
    source: str
    rmse: float
    version: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "homography": self.homography.tolist(),
            "homography_inv": self.homography_inv.tolist(),
            "units": self.units,
            "timestamp": self.timestamp,
            "source": self.source,
            "rmse": float(self.rmse),
            "version": self.version,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Calibration":
        H = np.asarray(d["homography"], dtype=np.float64)
        H_inv = np.asarray(d.get("homography_inv") or np.linalg.inv(H), dtype=np.float64)
        return Calibration(
            homography=H,
            homography_inv=H_inv,
            units=d.get("units", "meters"),
            timestamp=float(d.get("timestamp", time.time())),
            source=str(d.get("source", "unknown")),
            rmse=float(d.get("rmse", 1e9)),
            version=str(d.get("version", "v0")),
        )

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        if _HAS_YAML and path.suffix.lower() in (".yml", ".yaml"):
            with open(path, "w") as f:
                yaml.safe_dump(payload, f)
        else:
            # fallback to json
            with open(path, "w") as f:
                json.dump(payload, f, indent=2)


# -------------------------
# Core homography functions
# -------------------------
def _ensure_numpy_pts(pts: Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(pts, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("Points must be Nx2 array-like (x,y).")
    return arr


def compute_homography(
    image_points: Sequence[Pixel],
    world_points: Sequence[World],
    method: str = "ransac",
    ransac_thresh_px: float = 3.0,
    source: str = "manual",
    version: str = "v1",
    units: str = "meters",
) -> Calibration:
    """
    Compute homography that maps image-space points to world-space points.

    - image_points: list of image-space (x, y) pixel coordinates
    - world_points: list of world-space (X, Y) coordinates in meters (must lie on the plane)
    - method: "ransac" or "ls" (least-squares)
    - ransac_thresh_px: threshold for RANSAC reprojection in pixels
    """
    img = _ensure_numpy_pts(image_points)
    wrd = _ensure_numpy_pts(world_points)
    if img.shape[0] != wrd.shape[0]:
        raise ValueError("Number of image points must equal number of world points.")
    if img.shape[0] < 4:
        raise ValueError("At least 4 correspondences required for homography.")

    # OpenCV expects float32 for findHomography
    src = img.astype(np.float32)
    dst = wrd.astype(np.float32)

    if method.lower() == "ransac" and img.shape[0] >= 4:
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransacReprojThreshold=ransac_thresh_px)
    else:
        H, mask = cv2.findHomography(src, dst, 0)

    if H is None:
        raise RuntimeError("Homography estimation failed (H is None).")

    # Ensure invertible
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        raise RuntimeError("Estimated homography is singular / not invertible.")

    rmse_m = reprojection_rmse(img, wrd, H)  # note: world units (meters) because dst are meters
    cal = Calibration(
        homography=H.astype(np.float64),
        homography_inv=H_inv.astype(np.float64),
        units=units,
        timestamp=time.time(),
        source=source,
        rmse=rmse_m,
        version=version,
    )
    return cal


def reprojection_rmse(image_points: np.ndarray, world_points: np.ndarray, H: np.ndarray) -> float:
    """
    Compute RMSE (in world units) between projected image_points (via H) and given world_points.
    - image_points: Nx2 (pixels)
    - world_points: Nx2 (meters)
    - H: 3x3 mapping image -> world
    Returns scalar RMSE in same units as world_points.
    """
    img = _ensure_numpy_pts(image_points)
    wrd = _ensure_numpy_pts(world_points)
    # convert to homogeneous
    ones = np.ones((img.shape[0], 1), dtype=np.float64)
    img_h = np.hstack([img, ones])  # Nx3
    proj = (H @ img_h.T).T  # Nx3
    proj_xy = proj[:, :2] / proj[:, 2:3]
    diffs = proj_xy - wrd
    dists = np.linalg.norm(diffs, axis=1)
    rmse = float(np.sqrt(np.mean(dists ** 2)))
    return rmse



# -------------------------
# Utility: load/save config
# -------------------------
def save_calibration_yaml(cal: Calibration, path: Path) -> None:
    if not _HAS_YAML and path.suffix.lower() in (".yml", ".yaml"):
        # fallback to json if yaml not installed
        path = path.with_suffix(".json")
    cal.save(path)


def load_calibration(path: Path) -> Calibration:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    if _HAS_YAML and path.suffix.lower() in (".yml", ".yaml"):
        with open(path, "r") as f:
            payload = yaml.safe_load(f)
    else:
        with open(path, "r") as f:
            payload = json.load(f)
    return Calibration.from_dict(payload)



# -------------------------
# Projection helpers / API
# -------------------------
class Homography:
    """Encapsulates a computed image→world homography and related helpers:
       project_pixels_to_world, project_world_to_pixels, compute_local_scale, warp_to_birdseye, reprojection_rmse.
    """
    
    def __init__(self, calibration: Calibration):
        self.cal = calibration
        # cache float32 forms for OpenCV convenience
        self._H = calibration.homography.astype(np.float32)
        self._H_inv = calibration.homography_inv.astype(np.float32)

    def project_pixels_to_world(self, pixels: Sequence[Pixel]) -> np.ndarray:
        """
        Image-space pixels -> world-space coordinates in meters.

        Accepts arrays whose final dimension is 2 and preserves the input shape.
        """
        return _project_points(pixels, self._H)

    def project_world_to_pixels(self, world_points: Sequence[World]) -> np.ndarray:
        """
        World-space meters -> image-space pixels.

        Accepts arrays whose final dimension is 2 and preserves the input shape.
        """
        return _project_points(world_points, self._H_inv)

    def reprojection_rmse(self, image_points: Sequence[Pixel], world_points: Sequence[World]) -> float:
        img = _ensure_numpy_pts(image_points)
        wrd = _ensure_numpy_pts(world_points)
        return reprojection_rmse(img, wrd, self.cal.homography)

    def compute_local_scale(self, pixel: Pixel, delta_pixels: float = 1.0) -> Tuple[float, float, float]:
        """
        Approximate local meters-per-pixel in X and Y by finite differences:
        returns (m_per_px_x, m_per_px_y, m_per_px_avg)
        Note: 'x' means +delta along image x axis (right), 'y' means +delta along image y axis (down).
        """
        x, y = float(pixel[0]), float(pixel[1])
        p = np.array([[x, y], [x + delta_pixels, y], [x, y + delta_pixels]], dtype=np.float32)
        world = cv2.perspectiveTransform(p.reshape(-1, 1, 2), self._H).reshape(-1, 2)
        d_x = np.linalg.norm(world[1] - world[0])
        d_y = np.linalg.norm(world[2] - world[0])
        avg = (d_x + d_y) / 2.0
        return float(d_x / delta_pixels), float(d_y / delta_pixels), float(avg / 1.0)

    def warp_to_birdseye(
        self,
        frame: np.ndarray,
        dst_size: Tuple[int, int],
        origin_world: Tuple[float, float] = (0.0, 0.0),
        meters_per_pixel: float = 0.02,
        border_value: int = 0
    ) -> np.ndarray:
        """
        Produce a top-down (bird's-eye) view image.

        - frame: source image (HxWxC)
        - dst_size: (width_px, height_px) of desired bird image
        - origin_world: (X_min, Y_min) world coordinate that will map to pixel (0,0) in bird image
        - meters_per_pixel: scale for bird image (meters -> pixels)
        Returns warped image of shape (dst_h, dst_w, C).
        Implementation detail:
          M = A @ H_img2world, where A maps world-space meters -> bird_image pixels.
        """
        dst_w, dst_h = dst_size
        s = 1.0 / float(meters_per_pixel)  # pixels per meter
        X0, Y0 = float(origin_world[0]), float(origin_world[1])

        # A: world-space meters -> bird pixels
        A = np.array([
            [s, 0.0, -s * X0],
            [0.0, s, -s * Y0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        # H maps image -> world; we want M mapping image -> bird pixels
        M = (A @ self.cal.homography).astype(np.float32)
        bird = cv2.warpPerspective(frame, M, (dst_w, dst_h), flags=cv2.INTER_LINEAR, borderValue=border_value)
        return bird

class HomographyTransformer(IPlanarCoordinateTransformer):
    """Homography-backed adapter for the planar coordinate-transform contract."""

    def __init__(self, calibration_file: str | Path):
        self.calibration_file = str(calibration_file)
        self._homography = Homography(load_calibration(Path(calibration_file)))

    def image_to_world(self, points: np.ndarray) -> np.ndarray:
        return self._homography.project_pixels_to_world(points)

    def world_to_image(self, points: np.ndarray) -> np.ndarray:
        return self._homography.project_world_to_pixels(points)



def _project_points(points: Sequence[Sequence[float]], matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32)
    if arr.shape == (2,):
        original_shape = arr.shape
        flat = arr.reshape(1, 2)
    else:
        if arr.ndim < 2 or arr.shape[-1] != 2:
            raise ValueError("points must have final dimension 2")
        original_shape = arr.shape
        flat = arr.reshape(-1, 2)

    if flat.shape[0] == 0:
        return np.empty(original_shape, dtype=np.float64)

    projected = cv2.perspectiveTransform(flat.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    return projected.reshape(original_shape).astype(np.float64)
