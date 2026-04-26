import json
import shutil
from pathlib import Path

# -------- CONFIG --------
COCO_ROOT = Path(r"C:\Users\hamza\Datasets\TrafficDatasets\UA_DETRAC_robotflow")     # input
YOLO_ROOT = Path(r"C:\Users\hamza\Datasets\TrafficDatasets\UA_DETRAC_robotflow_yolo")     # output
SPLITS = {"train": "train", "valid": "val"}

YOLO_ROOT.mkdir(parents=True, exist_ok=True)

def convert_split(split_name, yolo_split):
    coco_dir = COCO_ROOT / split_name
    with open(coco_dir / "_annotations.coco.json", "r") as f:
        coco = json.load(f)

    images = {i["id"]: i for i in coco["images"]}
    categories = {c["id"]: c["name"] for c in coco["categories"]}
    class_map = {cid: i for i, cid in enumerate(sorted(categories))}

    (YOLO_ROOT / "images" / yolo_split).mkdir(parents=True, exist_ok=True)
    (YOLO_ROOT / "labels" / yolo_split).mkdir(parents=True, exist_ok=True)

    labels = {img_id: [] for img_id in images}

    for ann in coco["annotations"]:
        img = images[ann["image_id"]]
        w, h = img["width"], img["height"]
        x, y, bw, bh = ann["bbox"]

        xc = (x + bw / 2) / w
        yc = (y + bh / 2) / h
        bw /= w
        bh /= h

        cls = class_map[ann["category_id"]]
        labels[ann["image_id"]].append(
            f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
        )

    for img_id, img in images.items():
        shutil.copy(
            coco_dir / img["file_name"],
            YOLO_ROOT / "images" / yolo_split / img["file_name"]
        )
        with open(
            YOLO_ROOT / "labels" / yolo_split / f"{Path(img['file_name']).stem}.txt",
            "w"
        ) as f:
            f.write("\n".join(labels[img_id]))

    return categories, class_map


def main():
    YOLO_ROOT.mkdir(parents=True, exist_ok=True)
    all_categories = None
    class_map = None

    for split, yolo_split in SPLITS.items():
        all_categories, class_map = convert_split(split, yolo_split)

    with open(YOLO_ROOT / "uadetrac_roboflow.yaml", "w") as f:
        f.write(f"path: {YOLO_ROOT.resolve()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n\n")
        f.write(f"nc: {len(all_categories)}\n")
        f.write("names:\n")
        for cid, name in sorted(all_categories.items(), key=lambda x: class_map[x[0]]):
            f.write(f"  {class_map[cid]}: {name}\n")

    print("✅ COCO → YOLO conversion done.")


if __name__ == "__main__":
    main()