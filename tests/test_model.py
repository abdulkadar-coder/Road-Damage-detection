from __future__ import annotations

import uuid
import unittest
from types import ModuleType
from unittest.mock import patch
from pathlib import Path

from road_damage.model import (
    ModelLoadError,
    class_code_for_id,
    load_road_damage_model,
    validate_model,
)


class DetectorMetadata:
    task = "detect"

    def __init__(self, names):
        self.names = names


class ModelValidationTests(unittest.TestCase):
    def test_accepts_four_rdd_codes(self):
        model = DetectorMetadata({0: "D00", 1: "D10", 2: "D20", 3: "D40"})
        self.assertEqual(validate_model(model), {0: "D00", 1: "D10", 2: "D20", 3: "D40"})

    def test_accepts_prepared_dataset_class_names(self):
        model = DetectorMetadata([
            "Longitudinal crack",
            "Transverse crack",
            "Alligator crack",
            "Pothole",
        ])
        self.assertEqual(class_code_for_id(model.names, 3), "D40")

    def test_rejects_generic_coco_checkpoint_classes(self):
        model = DetectorMetadata(["person", "car", "truck"])
        with self.assertRaisesRegex(ModelLoadError, "not one of the RDD2022 classes"):
            validate_model(model)

    def test_rejects_non_detection_task(self):
        model = DetectorMetadata(["D00", "D10", "D20", "D40"])
        model.task = "segment"
        with self.assertRaisesRegex(ModelLoadError, "task='detect'"):
            validate_model(model)

    def test_rejects_duplicate_or_missing_rdd_class(self):
        model = DetectorMetadata(["D00", "D10", "D20", "D20"])
        with self.assertRaisesRegex(ModelLoadError, "exactly the four"):
            validate_model(model)

    def test_corrupt_checkpoint_is_reported_without_fallback(self):
        bad_file = Path(__file__).resolve().parent / f"corrupt-{uuid.uuid4().hex}.pt"
        bad_file.write_bytes(b"not a torch checkpoint")
        fake_ultralytics = ModuleType("ultralytics")

        def reject_checkpoint(path):
            raise RuntimeError("invalid checkpoint archive")

        fake_ultralytics.YOLO = reject_checkpoint
        try:
            with patch.dict("sys.modules", {"ultralytics": fake_ultralytics}):
                with self.assertRaisesRegex(ModelLoadError, "corrupted or incompatible"):
                    load_road_damage_model(bad_file)
        finally:
            bad_file.unlink(missing_ok=True)
    def test_missing_weights_fail_without_fallback(self):
        missing = Path(__file__).resolve().parent / f"missing-{uuid.uuid4().hex}.pt"
        with self.assertRaisesRegex(ModelLoadError, "not found"):
            load_road_damage_model(missing)


if __name__ == "__main__":
    unittest.main()

