#!/usr/bin/env python3
import yaml
import cv2
from pathlib import Path

class YOLOVisualizer:
    def __init__(self, dataset_root: str, yaml_name: str = "data.yaml"):
        self.root = Path(dataset_root)

        with open(self.root / yaml_name, "r") as f:
            cfg = yaml.safe_load(f)
        self.names = cfg["names"]

        # ---- logging dataset statistics ----
        print("[YOLOVisualizer] Dataset loaded:")
        for split in ["train", "val", "test"]:
            img_dir = self.root / "images" / split
            n_imgs = len(list(img_dir.iterdir())) if img_dir.exists() else 0
            print(f"  - {split}: {n_imgs} images")


    def _load_image_and_labels(self, split: str, image_name: str):
        img_path = self.root / "images" / split / image_name
        lbl_path = self.root / "labels" / split / f"{Path(image_name).stem}.txt"
        print(f"The .txt file lables corresponding to this image is: {lbl_path}")
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        h, w = image.shape[:2]
        boxes = []

        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    cls, xc, yc, bw, bh = map(float, line.split())
                    x1 = int((xc - bw / 2) * w)
                    y1 = int((yc - bh / 2) * h)
                    x2 = int((xc + bw / 2) * w)
                    y2 = int((yc + bh / 2) * h)
                    boxes.append((int(cls), x1, y1, x2, y2))

        return image, boxes

    def _draw(self, image, boxes):
        for cls, x1, y1, x2, y2 in boxes:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = self.names[int(cls)]
            cv2.putText(image, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return image

    def visualize_by_index(self, index: int, split: str = "train"):
        images = sorted((self.root / "images" / split).iterdir())
        image_name = images[index].name
        self.visualize_by_filename(image_name, split)

    def visualize_by_filename(self, filename: str, split: str = "train"):
        print(filename)
        image, boxes = self._load_image_and_labels(split, filename)
        image = self._draw(image, boxes)
        cv2.imshow(f"{split} | {filename}", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()