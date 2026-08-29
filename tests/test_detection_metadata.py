import unittest

import numpy as np

from detection.interface import IDetector, normalize_class_names


class DummyDetector(IDetector):
    def __init__(self, model_name: str, **kwargs):
        self._class_names = normalize_class_names(kwargs["names"])

    @property
    def class_names(self):
        return self._class_names

    @property
    def num_classes(self):
        return len(self._class_names)

    def warmup(self, imgsz=None):
        return None

    def close(self):
        return None

    def preprocess(self, array_frame: np.ndarray):
        return array_frame

    def infer(self, preprocessed_input, **kwargs):
        return preprocessed_input

    def postprocess(self, raw_output, preprocessed_input, array_frame):
        return np.zeros((0, 6), dtype=np.float32)

    def detect_to_track(self, array_frame: np.ndarray, **kwargs):
        return np.zeros((0, 6), dtype=np.float32)


class DetectorMetadataContractTest(unittest.TestCase):
    def test_normalize_sequence_class_names(self):
        self.assertEqual(
            normalize_class_names(["car", "truck"]),
            ("car", "truck"),
        )

    def test_normalize_mapping_class_names_by_class_id(self):
        self.assertEqual(
            normalize_class_names({2: "bus", 0: "car", 1: "truck"}),
            ("car", "truck", "bus"),
        )

    def test_detector_contract_exposes_class_names_and_num_classes(self):
        detector = DummyDetector("dummy", names={1: "truck", 0: "car"})

        self.assertEqual(detector.class_names, ("car", "truck"))
        self.assertEqual(detector.num_classes, 2)
        self.assertEqual(detector.detect_to_track(np.zeros((4, 4, 3))).shape, (0, 6))


if __name__ == "__main__":
    unittest.main()
