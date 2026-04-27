#!/usr/bin/env python3
import random
import shutil
from pathlib import Path

# -------- CONFIG --------
SRC_ROOT = Path(r"C:\Users\hamza\Datasets\TrafficDatasets\Dawn_traffic_conditions")
DST_ROOT = Path(r"C:\Users\hamza\Datasets\TrafficDatasets\DAWN_ULTRALYTICS")

TRAIN_RATIO = 0.8
RANDOM_SEED = 42

CLASSES = [
    None,
    "person",
    "bicycle",
    "car",
    "motorcycle",
    None,
    "bus",
    "train",
    "truck",
]

CONDITIONS = ["Fog", "Rain", "Sand", "Snow"]

# -----------------------

def main():
    random.seed(RANDOM_SEED)

    images_out = {
        "train": DST_ROOT / "images" / "train",
        "test": DST_ROOT / "images" / "test",
    }
    labels_out = {
        "train": DST_ROOT / "labels" / "train",
        "test": DST_ROOT / "labels" / "test",
    }

    for p in list(images_out.values()) + list(labels_out.values()):
        p.mkdir(parents=True, exist_ok=True)

    samples = []

    # Collect all image–label pairs
    for cond in CONDITIONS:
        cond_dir = SRC_ROOT / cond
        label_dir = cond_dir / f"{cond}_YOLO_darknet"

        for img in cond_dir.iterdir():
            if img.suffix.lower() not in [".jpg", ".png", ".jpeg"]:
                continue

            label = label_dir / f"{img.stem}.txt"
            if label.exists():
                samples.append((img, label))

    random.shuffle(samples)
    split_idx = int(len(samples) * TRAIN_RATIO)

    splits = {
        "train": samples[:split_idx],
        "test": samples[split_idx:],
    }

    # Copy files
    for split, items in splits.items():
        for img, lbl in items:
            shutil.copy(img, images_out[split] / img.name)
            shutil.copy(lbl, labels_out[split] / lbl.name)

    # Write data.yaml
    with open(DST_ROOT / "data.yaml", "w") as f:
        f.write(f"path: {DST_ROOT.resolve()}\n")
        f.write("train: images/train\n")
        f.write("val: images/test\n\n")
        f.write(f"nc: {len(CLASSES)}\n")
        f.write("names:\n")
        for i, name in enumerate(CLASSES):
            f.write(f"  {i}: {name}\n")

    print("✅ DAWN dataset converted to Ultralytics YOLO format.")

if __name__ == "__main__":
    main()