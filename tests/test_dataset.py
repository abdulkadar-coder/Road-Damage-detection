from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from scripts.train import validate_dataset_yaml


class DatasetConfigTests(unittest.TestCase):
    def _write_dataset(self, names):
        directory = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        root = Path(directory.name)
        (root / "images" / "train").mkdir(parents=True)
        (root / "images" / "val").mkdir(parents=True)
        config = {
            "path": str(root),
            "train": "images/train",
            "val": "images/val",
            "names": {index: name for index, name in enumerate(names)},
        }
        data_yaml = root / "data.yaml"
        data_yaml.write_text(yaml.safe_dump(config), encoding="utf-8")
        return directory, data_yaml

    def test_accepts_generated_four_class_data_yaml(self):
        directory, data_yaml = self._write_dataset([
            "Longitudinal crack", "Transverse crack", "Alligator crack", "Pothole"
        ])
        with directory:
            self.assertTrue(validate_dataset_yaml(data_yaml)["train"])

    def test_rejects_wrong_class_order(self):
        directory, data_yaml = self._write_dataset([
            "Pothole", "Transverse crack", "Alligator crack", "Longitudinal crack"
        ])
        with directory:
            with self.assertRaisesRegex(ValueError, "class IDs/names must be in this order"):
                validate_dataset_yaml(data_yaml)


if __name__ == "__main__":
    unittest.main()

