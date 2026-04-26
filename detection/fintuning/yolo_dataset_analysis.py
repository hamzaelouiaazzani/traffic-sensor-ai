#!/usr/bin/env python3
import random
from pathlib import Path
import yaml
import matplotlib.pyplot as plt

class YOLODatasetStats:
    def __init__(self, dataset_root: str, yaml_name: str = "data.yaml"):
        self.root = Path(dataset_root)

        with open(self.root / yaml_name, "r") as f:
            cfg = yaml.safe_load(f)

        self.names = cfg["names"]
        self.splits = ["train", "val", "test"]

    def _label_files(self, split):
        lbl_dir = self.root / "labels" / split
        return list(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []

    def class_frequencies(self, split: str | None = None):
        """
        Returns dict: {class_name: count}
        If split is None → all splits combined
        """
        counts = {name: 0 for name in self.names.values()}

        splits = [split] if split else self.splits
        for sp in splits:
            for lbl in self._label_files(sp):
                with open(lbl) as f:
                    for line in f:
                        cls_id = int(line.split()[0])
                        counts[self.names[cls_id]] += 1

        return counts

    def plot_class_distribution(self, split: str | None = None):
        counts = self.class_frequencies(split)
        classes = list(counts.keys())
        values = list(counts.values())

        title = f"Class Distribution ({split})" if split else "Class Distribution (All)"
        plt.figure(figsize=(10, 5))
        plt.bar(classes, values)
        plt.title(title)
        plt.ylabel("Frequency")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.show()
        