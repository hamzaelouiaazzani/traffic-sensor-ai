#!/usr/bin/env python3
import shutil
from pathlib import Path
import yaml

class YOLODatasetStandardizer:
    """
    Standardizes multiple YOLO-format datasets into one unified dataset
    by remapping class IDs and preserving YOLO geometry.
    """

    def __init__(self, standardized_root: str, standard_names: list):
        self.std_root = Path(standardized_root)
        self.standard_names = standard_names
        self.splits = ["train", "val", "test"]

        # create folder structure if missing
        for split in self.splits:
            (self.std_root / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.std_root / "labels" / split).mkdir(parents=True, exist_ok=True)

        # create / overwrite authoritative data.yaml
        self._write_data_yaml()

        print("[Standardizer] Initialized standardized dataset at:")
        print(f"  {self.std_root}")

    def _write_data_yaml(self):
        yaml_path = self.std_root / "data.yaml"
        with open(yaml_path, "w") as f:
            f.write(f"path: {self.std_root.resolve()}\n")
            f.write("train: images/train\n")
            f.write("val: images/val\n")
            f.write("test: images/test\n\n")
            f.write(f"nc: {len(self.standard_names)}\n")
            f.write("names:\n")
            for i, name in enumerate(self.standard_names):
                f.write(f"  {i}: {name}\n")

    def add_dataset(self, dataset_root: str, class_id_map: dict):
        """
        Add a YOLO-format dataset into the standardized dataset
        using a class-ID remapping.

        class_id_map example:
            {0: None, 1: 0, 2: 1, 3: 1, 4: 1}
        """
        dataset_root = Path(dataset_root)

        print(f"\n[Standardizer] Adding dataset: {dataset_root}")

        for split in self.splits:
            src_img_dir = dataset_root / "images" / split
            src_lbl_dir = dataset_root / "labels" / split

            if not src_img_dir.exists() or not src_lbl_dir.exists():
                continue

            dst_img_dir = self.std_root / "images" / split
            dst_lbl_dir = self.std_root / "labels" / split

            for lbl_file in src_lbl_dir.glob("*.txt"):
                new_lines = []

                with open(lbl_file, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if not parts:
                            continue

                        old_cls = int(parts[0])

                        # skip removed classes
                        if old_cls not in class_id_map:
                            continue
                        if class_id_map[old_cls] is None:
                            continue

                        new_cls = class_id_map[old_cls]
                        new_lines.append(
                            " ".join([str(new_cls)] + parts[1:])
                        )

                # skip images with no remaining annotations
                if not new_lines:
                    continue

                # copy image
                img_path = next(
                    (src_img_dir / f"{lbl_file.stem}{ext}"
                     for ext in [".jpg", ".png", ".jpeg"]
                     if (src_img_dir / f"{lbl_file.stem}{ext}").exists()),
                    None
                )

                if img_path is None:
                    continue

                shutil.copy(img_path, dst_img_dir / img_path.name)

                # write standardized label
                with open(dst_lbl_dir / lbl_file.name, "w") as f:
                    f.write("\n".join(new_lines))

        print("  ✔ Dataset merged successfully")
