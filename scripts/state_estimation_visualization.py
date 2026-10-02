from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from perception.detection.factory import build_detector
from perception.tracking.factory import build_tracker
from runtime.timing import TimingPolicy
from perception.vision_io.frame_producer import Frame

from boxmot.trackers.bytetrack.bytetrack import STrack
from boxmot.utils import logger as BOXMOT_LOGGER


STATE_VECTOR_DESCRIPTION = "x, y, a, h, vx, vy, va, vh"
CHI2_95_2D = 5.9915


@dataclass
class UpdateDiagnostic:
    frame: int
    timestamp: float
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    measurement_bbox: List[float]
    measurement_xyah: List[float]
    predicted_bbox: List[float]
    predicted_state: List[float]
    prior_covariance: List[List[float]]
    updated_bbox: List[float]
    updated_state: List[float]
    posterior_covariance: List[List[float]]
    prediction_measurement_distance: float
    prediction_update_distance: float
    iou_prediction_observation: float
    iou_update_observation: float
    prior_position_covariance: List[List[float]]
    posterior_position_covariance: List[List[float]]
    prior_uncertainty_ellipse: Optional[Dict[str, float]]
    posterior_uncertainty_ellipse: Optional[Dict[str, float]]
    update_kind: str


@dataclass
class Candidate:
    diagnostic: UpdateDiagnostic
    frame_bgr: np.ndarray
    score: float


class ByteTrackDiagnosticHook:
    def __init__(self, class_names: Tuple[str, ...]):
        self.class_names = class_names
        self.current_frame = -1
        self.current_timestamp = 0.0
        self.records_by_frame: Dict[int, List[UpdateDiagnostic]] = {}
        self._original_update = None
        self._original_reactivate = None

    def install(self):
        self._original_update = STrack.update
        self._original_reactivate = STrack.re_activate
        hook = self

        def instrumented_update(track, new_track, frame_id):
            return hook._capture_update(track, new_track, frame_id, "update", hook._original_update)

        def instrumented_reactivate(track, new_track, frame_id, new_id=False):
            return hook._capture_update(
                track,
                new_track,
                frame_id,
                "re_activate",
                hook._original_reactivate,
                new_id=new_id,
            )

        STrack.update = instrumented_update
        STrack.re_activate = instrumented_reactivate

    def uninstall(self):
        if self._original_update is not None:
            STrack.update = self._original_update
        if self._original_reactivate is not None:
            STrack.re_activate = self._original_reactivate

    def _capture_update(self, track, new_track, frame_id, update_kind, original_method, **kwargs):
        prior_mean = None if track.mean is None else track.mean.copy()
        prior_covariance = None if track.covariance is None else track.covariance.copy()
        measurement_xyah = new_track.xyah.copy()
        measurement_bbox = new_track.xyxy.copy()
        confidence = float(new_track.conf)
        class_id = int(new_track.cls)

        result = original_method(track, new_track, frame_id, **kwargs)

        if prior_mean is not None and prior_covariance is not None:
            posterior_mean = track.mean.copy()
            posterior_covariance = track.covariance.copy()
            diagnostic = build_diagnostic(
                frame=self.current_frame,
                timestamp=self.current_timestamp,
                track_id=int(track.id),
                class_id=class_id,
                class_name=class_name_for_id(class_id, self.class_names),
                confidence=confidence,
                measurement_bbox=measurement_bbox,
                measurement_xyah=measurement_xyah,
                predicted_state=prior_mean,
                prior_covariance=prior_covariance,
                updated_state=posterior_mean,
                posterior_covariance=posterior_covariance,
                update_kind=update_kind,
            )
            self.records_by_frame.setdefault(self.current_frame, []).append(diagnostic)

        return result


def class_name_for_id(class_id: int, class_names: Tuple[str, ...]) -> str:
    if 0 <= class_id < len(class_names):
        return class_names[class_id]
    return f"class_{class_id}"


def xyah_to_xyxy(state_xyah: np.ndarray) -> np.ndarray:
    x, y, aspect, height = state_xyah[:4]
    width = aspect * height
    return np.array(
        [
            x - width * 0.5,
            y - height * 0.5,
            x + width * 0.5,
            y + height * 0.5,
        ],
        dtype=np.float64,
    )


def bbox_center(bbox: np.ndarray) -> np.ndarray:
    return np.array([(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5], dtype=np.float64)


def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def uncertainty_ellipse(covariance_2x2: np.ndarray) -> Optional[Dict[str, float]]:
    if covariance_2x2.shape != (2, 2):
        return None
    if not np.all(np.isfinite(covariance_2x2)):
        return None
    covariance_2x2 = (covariance_2x2 + covariance_2x2.T) * 0.5
    values, vectors = np.linalg.eigh(covariance_2x2)
    if np.any(values <= 0):
        return None
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    angle = math.degrees(math.atan2(vectors[1, 0], vectors[0, 0]))
    return {
        "width": float(2.0 * math.sqrt(CHI2_95_2D * values[0])),
        "height": float(2.0 * math.sqrt(CHI2_95_2D * values[1])),
        "angle_degrees": float(angle),
    }


def build_diagnostic(
    frame: int,
    timestamp: float,
    track_id: int,
    class_id: int,
    class_name: str,
    confidence: float,
    measurement_bbox: np.ndarray,
    measurement_xyah: np.ndarray,
    predicted_state: np.ndarray,
    prior_covariance: np.ndarray,
    updated_state: np.ndarray,
    posterior_covariance: np.ndarray,
    update_kind: str,
) -> UpdateDiagnostic:
    predicted_bbox = xyah_to_xyxy(predicted_state[:4])
    updated_bbox = xyah_to_xyxy(updated_state[:4])
    measurement_center = bbox_center(measurement_bbox)
    predicted_center = predicted_state[:2].astype(np.float64)
    updated_center = updated_state[:2].astype(np.float64)
    prior_position_covariance = prior_covariance[:2, :2].copy()
    posterior_position_covariance = posterior_covariance[:2, :2].copy()

    return UpdateDiagnostic(
        frame=int(frame),
        timestamp=float(timestamp),
        track_id=track_id,
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        measurement_bbox=measurement_bbox.astype(float).tolist(),
        measurement_xyah=measurement_xyah.astype(float).tolist(),
        predicted_bbox=predicted_bbox.astype(float).tolist(),
        predicted_state=predicted_state.astype(float).tolist(),
        prior_covariance=prior_covariance.astype(float).tolist(),
        updated_bbox=updated_bbox.astype(float).tolist(),
        updated_state=updated_state.astype(float).tolist(),
        posterior_covariance=posterior_covariance.astype(float).tolist(),
        prediction_measurement_distance=float(np.linalg.norm(predicted_center - measurement_center)),
        prediction_update_distance=float(np.linalg.norm(predicted_center - updated_center)),
        iou_prediction_observation=float(bbox_iou(predicted_bbox, measurement_bbox)),
        iou_update_observation=float(bbox_iou(updated_bbox, measurement_bbox)),
        prior_position_covariance=prior_position_covariance.astype(float).tolist(),
        posterior_position_covariance=posterior_position_covariance.astype(float).tolist(),
        prior_uncertainty_ellipse=uncertainty_ellipse(prior_position_covariance),
        posterior_uncertainty_ellipse=uncertainty_ellipse(posterior_position_covariance),
        update_kind=update_kind,
    )


def diagnostic_score(diagnostic: UpdateDiagnostic, rng: random.Random) -> float:
    area = bbox_area(np.asarray(diagnostic.updated_bbox))
    if area < 500:
        return -1.0
    return (
        diagnostic.prediction_measurement_distance
        + 80.0 * (1.0 - diagnostic.iou_prediction_observation)
        + 1e-6 * rng.random()
    )


def bbox_area(bbox: np.ndarray) -> float:
    return float(max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1]))


def draw_box(ax, bbox, color, label, linestyle="-", linewidth=2.5):
    x1, y1, x2, y2 = bbox
    rect = plt.Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        fill=False,
        edgecolor=color,
        linewidth=linewidth,
        linestyle=linestyle,
    )
    ax.add_patch(rect)
    ax.text(
        x1,
        max(8, y1 - 10),
        label,
        color="white",
        fontsize=10,
        bbox={"facecolor": color, "edgecolor": "none", "alpha": 0.88, "pad": 3},
    )


def draw_ellipse(ax, center, ellipse, color, label):
    if ellipse is None:
        return
    patch = plt.matplotlib.patches.Ellipse(
        xy=center,
        width=ellipse["width"],
        height=ellipse["height"],
        angle=ellipse["angle_degrees"],
        fill=False,
        edgecolor=color,
        linewidth=2.0,
        linestyle=":",
    )
    ax.add_patch(patch)
    ax.text(
        center[0],
        center[1],
        label,
        color=color,
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": color, "alpha": 0.75, "pad": 2},
    )


def render_panel(ax, frame_bgr: np.ndarray, diagnostic: UpdateDiagnostic, panel_label: Optional[str] = None):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    ax.imshow(frame_rgb)
    ax.axis("off")
    if panel_label:
        ax.set_title(panel_label, fontsize=13, fontweight="bold")

    observation = np.asarray(diagnostic.measurement_bbox)
    prediction = np.asarray(diagnostic.predicted_bbox)
    update = np.asarray(diagnostic.updated_bbox)

    draw_box(
        ax,
        observation,
        "#1b9e77",
        f"Observation z(k)\nconf={diagnostic.confidence:.2f}, {diagnostic.class_name}",
        linestyle="-",
        linewidth=2.8,
    )
    draw_box(ax, prediction, "#d95f02", "Prediction x̂⁻(k)", linestyle="--", linewidth=2.5)
    draw_box(ax, update, "#377eb8", "Updated estimate x̂(k)", linestyle="-", linewidth=2.5)

    predicted_center = np.asarray(diagnostic.predicted_state[:2])
    updated_center = np.asarray(diagnostic.updated_state[:2])
    draw_ellipse(
        ax,
        predicted_center,
        diagnostic.prior_uncertainty_ellipse,
        "#d95f02",
        "95% prior\nposition uncertainty",
    )
    draw_ellipse(
        ax,
        updated_center,
        diagnostic.posterior_uncertainty_ellipse,
        "#377eb8",
        "95% posterior\nposition uncertainty",
    )

    info = (
        f"Frame {diagnostic.frame} | t={diagnostic.timestamp:.2f}s\n"
        f"Track ID {diagnostic.track_id} | {diagnostic.class_name}"
    )
    ax.text(
        0.012,
        0.985,
        info,
        transform=ax.transAxes,
        va="top",
        ha="left",
        color="white",
        fontsize=10,
        bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.65, "pad": 4},
    )


def save_individual_figure(candidate: Candidate, out_dir: Path) -> Path:
    diagnostic = candidate.diagnostic
    out_path = out_dir / f"frame_{diagnostic.frame:04d}_state_estimation.png"
    fig, ax = plt.subplots(figsize=(16, 9), dpi=180)
    render_panel(ax, candidate.frame_bgr, diagnostic)
    fig.tight_layout(pad=0)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out_path


def save_combined_figure(candidates: List[Candidate], out_dir: Path) -> Path:
    out_path = out_dir / "state_estimation_examples.png"
    fig, axes = plt.subplots(len(candidates), 1, figsize=(15, 8 * len(candidates)), dpi=170)
    if len(candidates) == 1:
        axes = [axes]
    fig.suptitle("Detection, Prediction, and State Update in Multi-Object Tracking", fontsize=20, fontweight="bold")
    for idx, (ax, candidate) in enumerate(zip(axes, candidates), start=1):
        render_panel(ax, candidate.frame_bgr, candidate.diagnostic, panel_label=f"({chr(96 + idx)}) Example {idx}")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return out_path


def process_video(args) -> Tuple[List[Candidate], List[UpdateDiagnostic], Tuple[str, ...], float, int]:
    rng = random.Random(args.seed)
    BOXMOT_LOGGER.remove()
    cap = cv2.VideoCapture(str(args.source))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.source}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or args.fps)
    if fps <= 0:
        fps = args.fps
    timing_policy = TimingPolicy(mode="frame", fps=fps)

    detector = build_detector(
        model_name=args.model_name,
        conf=args.conf,
        device=args.device,
    )
    tracker = build_tracker(
        backend="boxmot",
        method="bytetrack",
        reid_model=args.reid_model,
        classes=None,
        device=args.tracker_device,
        half=False,
        per_class=args.per_class,
    )

    hook = ByteTrackDiagnosticHook(detector.class_names)
    hook.install()

    bins: List[Optional[Candidate]] = [None, None, None]
    all_records: List[UpdateDiagnostic] = []

    try:
        with torch.inference_mode():
            read_idx = 0
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break

                timing = timing_policy.resolve(
                    Frame(data=frame_bgr, timestamp=0.0, read_idx=read_idx)
                )
                hook.current_frame = timing.frame_id
                hook.current_timestamp = timing.timestamp

                detections = detector.detect_to_track(frame_bgr)
                tracker.update(detections, frame_bgr)

                frame_records = hook.records_by_frame.pop(timing.frame_id, [])
                all_records.extend(frame_records)
                if frame_records:
                    best_record = max(frame_records, key=lambda record: diagnostic_score(record, rng))
                    score = diagnostic_score(best_record, rng)
                    bin_idx = min(2, int((read_idx / max(total_frames, 1)) * 3))
                    if score > 0 and (bins[bin_idx] is None or score > bins[bin_idx].score):
                        bins[bin_idx] = Candidate(
                            diagnostic=best_record,
                            frame_bgr=frame_bgr.copy(),
                            score=score,
                        )

                read_idx += 1
                if args.max_frames is not None and read_idx >= args.max_frames:
                    break
    finally:
        hook.uninstall()
        cap.release()
        if hasattr(detector, "close"):
            detector.close()
        if hasattr(tracker, "close"):
            tracker.close()

    selected = [candidate for candidate in bins if candidate is not None]
    if len(selected) < 3:
        by_frame = {candidate.diagnostic.frame for candidate in selected}
        fallback_records = sorted(
            all_records,
            key=lambda record: diagnostic_score(record, rng),
            reverse=True,
        )
        for record in fallback_records:
            if record.frame in by_frame:
                continue
            frame = load_frame_at(args.source, record.frame)
            selected.append(Candidate(diagnostic=record, frame_bgr=frame, score=diagnostic_score(record, rng)))
            by_frame.add(record.frame)
            if len(selected) == 3:
                break

    selected = sorted(selected[:3], key=lambda candidate: candidate.diagnostic.frame)
    return selected, all_records, detector.class_names, fps, total_frames


def load_frame_at(source: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(source))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Failed to reload frame {frame_idx} from {source}")
        return frame
    finally:
        cap.release()


def print_diagnostic_report(diagnostics: List[UpdateDiagnostic]):
    for diagnostic in diagnostics:
        print("-" * 50)
        print(f"Frame: {diagnostic.frame}")
        print(f"Timestamp: {diagnostic.timestamp:.6f} s")
        print(f"Track ID: {diagnostic.track_id}")
        print(f"Class: {diagnostic.class_name} ({diagnostic.class_id})")
        print("")
        print("OBSERVATION")
        print(f"Detection bbox: {format_vector(diagnostic.measurement_bbox)}")
        print(f"Detection center: {format_vector(bbox_center(np.asarray(diagnostic.measurement_bbox)).tolist())}")
        print(f"Confidence: {diagnostic.confidence:.6f}")
        print("")
        print("PRIOR / PREDICTION")
        print(f"Predicted bbox: {format_vector(diagnostic.predicted_bbox)}")
        print(f"Predicted state vector [{STATE_VECTOR_DESCRIPTION}]: {format_vector(diagnostic.predicted_state)}")
        print(f"Predicted center: {format_vector(diagnostic.predicted_state[:2])}")
        print(f"Prior covariance: {format_matrix(diagnostic.prior_covariance)}")
        print(f"Position covariance: {format_matrix(diagnostic.prior_position_covariance)}")
        print(f"Uncertainty ellipse parameters: {diagnostic.prior_uncertainty_ellipse}")
        print("")
        print("POSTERIOR / UPDATED ESTIMATE")
        print(f"Updated bbox: {format_vector(diagnostic.updated_bbox)}")
        print(f"Updated state vector [{STATE_VECTOR_DESCRIPTION}]: {format_vector(diagnostic.updated_state)}")
        print(f"Updated center: {format_vector(diagnostic.updated_state[:2])}")
        print(f"Posterior covariance: {format_matrix(diagnostic.posterior_covariance)}")
        print(f"Position covariance: {format_matrix(diagnostic.posterior_position_covariance)}")
        print(f"Posterior uncertainty ellipse parameters: {diagnostic.posterior_uncertainty_ellipse}")
        print("")
        print("VISUAL DIFFERENCES")
        print(
            "Prediction center <-> measurement center: "
            f"{diagnostic.prediction_measurement_distance:.3f} px"
        )
        print(
            "Prediction center <-> updated-estimate center: "
            f"{diagnostic.prediction_update_distance:.3f} px"
        )
        print(f"IoU predicted bbox <-> observation bbox: {diagnostic.iou_prediction_observation:.4f}")
        print(f"IoU updated bbox <-> observation bbox: {diagnostic.iou_update_observation:.4f}")
    print("-" * 50)


def format_vector(values) -> str:
    return "[" + ", ".join(f"{float(value):.3f}" for value in values) + "]"


def format_matrix(values) -> str:
    rows = ["[" + ", ".join(f"{float(value):.3f}" for value in row) + "]" for row in values]
    return "[" + "; ".join(rows) + "]"


def print_slide_text(selected: List[Candidate]):
    print("")
    print("READY-TO-COPY SLIDE TEXT")
    print("OBSERVATION: The detector provides a noisy measurement z(k) of the object's image-space bounding box.")
    print("PREDICTION: BoxMOT ByteTrack predicts the state x̂⁻(k) before seeing the current measurement.")
    print("UPDATE: The matched detection is used to correct the Kalman state, producing x̂(k).")
    print("UNCERTAINTY: The covariance is the Kalman estimator uncertainty; ellipses show 95% image-plane center uncertainty.")
    print("TRUE STATE: No true state is available from the video alone; manual annotation is required.")
    for idx, candidate in enumerate(selected, start=1):
        d = candidate.diagnostic
        print(
            f"Example {idx}: On frame {d.frame}, the prediction was "
            f"{d.prediction_measurement_distance:.1f} px from the detector measurement and the posterior moved "
            f"{d.prediction_update_distance:.1f} px from the prior toward the associated observation."
        )


def print_validation():
    print("")
    print("SCIENTIFIC VALIDATION")
    print("1. Observation is the STrack new_track measurement passed into the matched ByteTrack update/re_activate call.")
    print("2. Prediction is copied from track.mean immediately after ByteTrack multi_predict/association and before correction.")
    print("3. Updated estimate is copied from track.mean after the original BoxMOT correction method returns.")
    print("4. Prior/posterior covariance are copied alongside their corresponding pre/post state vectors.")
    print("5. Bbox conversion uses ByteTrack's XYAH convention: width = aspect_ratio * height.")
    print("6. Confidence is the detector confidence stored in new_track.conf.")
    print("7. Track ID, class, measurement, prior, and posterior come from the same track update in the same frame.")
    print("8. Tracking is run sequentially from frame 0 through the video; selected examples are chosen after continuous updates.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create state-estimation visualizations from BoxMOT ByteTrack diagnostics."
    )
    parser.add_argument("--source", type=Path, default=ROOT / "p1c1.mp4")
    parser.add_argument("--model-name", default="yolo11n_finetuned")
    parser.add_argument("--conf", type=float, default=0.50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tracker-device", default="")
    parser.add_argument("--reid-model", default="osnet_x0_25_market1501.pt")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--per-class", action="store_true", default=False)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "state_estimation_visualization")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected, records, class_names, fps, total_frames = process_video(args)
    if len(selected) < 3:
        raise RuntimeError(f"Only found {len(selected)} usable matched-update examples; need 3.")

    individual_paths = [save_individual_figure(candidate, args.output_dir) for candidate in selected]
    combined_path = save_combined_figure(selected, args.output_dir)
    diagnostics = [candidate.diagnostic for candidate in selected]

    diagnostics_path = args.output_dir / "state_estimation_diagnostics.json"
    diagnostics_path.write_text(
        json.dumps([asdict(diagnostic) for diagnostic in diagnostics], indent=2),
        encoding="utf-8",
    )

    print("FILES INSPECTED")
    print("AGENTS.md: repository rules and execution policy.")
    print("REPOSITORY_CONTEXT.md: architecture map and current runtime contracts.")
    print("src/perception/detection/factory.py: detector construction for yolo11n_finetuned.")
    print("src/perception/detection/interface.py: canonical detector output contract.")
    print("src/perception/detection/ultralytics_detectors.py: model loading and detector output conversion.")
    print("src/perception/tracking/interface.py: stable tracker contract.")
    print("src/perception/tracking/factory.py: tracker backend binding.")
    print("src/perception/tracking/boxmot_trackers.py: BoxMOT tracker adapter.")
    print("src/runtime/perception.py: per-frame detection -> tracking contract.")
    print("src/vision_io/frame_producer.py: repository Frame metadata convention.")
    print("boxmot/trackers/bytetrack/bytetrack.py: STrack update, association flow, and output columns.")
    print("boxmot/motion/kalman_filters/aabb/xyah_kf.py: actual Kalman state convention.")
    print("boxmot/motion/kalman_filters/aabb/base_kalman_filter.py: predict/update covariance semantics.")
    print("boxmot/configs/trackers/bytetrack.yaml: default ByteTrack thresholds.")
    print("")
    print("BYTE TRACK STATE REPRESENTATION")
    print(f"State vector: [{STATE_VECTOR_DESCRIPTION}]")
    print("The measurement is XYAH: center x, center y, aspect ratio, height.")
    print("Visualization bboxes are converted with width = aspect_ratio * height.")
    print("")
    print("GROUND TRUTH")
    print("No true state is available from the video alone; manual annotation is required.")
    print("The UA-DETRAC annotation CSV present in this repo does not contain a p1c1/P1C1 sequence match.")
    print("")
    print(f"Processed frames sequentially: {total_frames}")
    print(f"FPS used for timestamps: {fps:.6f}")
    print(f"Matched-update diagnostics collected: {len(records)}")
    print("Selected frames and timestamps:")
    for diagnostic in diagnostics:
        print(f"  frame {diagnostic.frame}, timestamp {diagnostic.timestamp:.3f} s, track {diagnostic.track_id}")
    print("")
    print("OUTPUT FILES")
    for path in individual_paths:
        print(path)
    print(combined_path)
    print(diagnostics_path)
    print("")
    print_diagnostic_report(diagnostics)
    print_slide_text(selected)
    print_validation()


if __name__ == "__main__":
    main()
