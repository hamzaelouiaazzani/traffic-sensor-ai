#!/usr/bin/env python3
import shutil
from pathlib import Path

# -------- PATHS --------

SRC_ROOT = Path(r"C:\Users\hamza\Datasets\TrafficDatasets\vehicles")
DST_ROOT = Path(r"C:\Users\hamza\Datasets\TrafficDatasets\vehicles_ultralytics")

SPLITS = {
    "train": "train",
    "valid": "val",
    "test": "test"
}

CLASS_NAMES = ['Bus', 'Jeepney', 'Motorcycle', 'Tricycle', 'Van', 'cars', 'truck']

def main():
    for split_src, split_dst in SPLITS.items():
        # Create destination folders
        (DST_ROOT / "images" / split_dst).mkdir(parents=True, exist_ok=True)
        (DST_ROOT / "labels" / split_dst).mkdir(parents=True, exist_ok=True)

        # Source folders
        src_images = SRC_ROOT / split_src / "images"
        src_labels = SRC_ROOT / split_src / "labels"

        # Copy images
        for img in src_images.iterdir():
            shutil.copy(img, DST_ROOT / "images" / split_dst / img.name)

        # Copy labels
        for lbl in src_labels.iterdir():
            shutil.copy(lbl, DST_ROOT / "labels" / split_dst / lbl.name)

    # Write data.yaml
    yaml_path = DST_ROOT / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {DST_ROOT.resolve()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n\n")
        f.write(f"nc: {len(CLASS_NAMES)}\n")
        f.write("names:\n")
        for i, name in enumerate(CLASS_NAMES):
            f.write(f"  {i}: {name}\n")

    print("✅ Dataset successfully restructured for Ultralytics.")

if __name__ == "__main__":
    main()
