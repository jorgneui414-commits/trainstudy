"""Tests for the versioned YOLO dataset builder."""

from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yolo_dataset_builder as builder


DETECTION_LINE = "0 0.5 0.5 0.2 0.2\n"
OBB_LINE = "0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n"


class YoloDatasetBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.raw_dir = self.root / "raw_labeled"
        self.dataset_dir = self.root / "datasets"
        self.raw_dir.mkdir()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def make_source(
        self,
        count: int = 10,
        *,
        label_text: str = DETECTION_LINE,
        classes: str = "part\n",
    ) -> None:
        (self.raw_dir / "classes.txt").write_text(classes, encoding="utf-8")
        for index in range(count):
            stem = f"sample_{index:02d}"
            (self.raw_dir / f"{stem}.jpg").write_bytes(f"image-{index}".encode())
            (self.raw_dir / f"{stem}.txt").write_text(label_text, encoding="utf-8")

    def split_names(self, version_dir: Path) -> dict[str, set[str]]:
        return {
            split: {path.stem for path in (version_dir / "images" / split).iterdir()}
            for split in builder.SPLIT_NAMES
        }

    def test_builds_7_2_1_version_and_absolute_yaml(self) -> None:
        self.make_source(classes="bolt\nnut\n")

        yaml_path = builder.build_yolo_dataset(self.raw_dir, self.dataset_dir)
        version_dir = yaml_path.parent

        self.assertTrue(yaml_path.is_absolute())
        self.assertEqual(
            {split: len(names) for split, names in self.split_names(version_dir).items()},
            {"train": 7, "val": 2, "test": 1},
        )
        for split in builder.SPLIT_NAMES:
            image_names = self.split_names(version_dir)[split]
            label_names = {path.stem for path in (version_dir / "labels" / split).iterdir()}
            self.assertEqual(image_names, label_names)

        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        self.assertTrue(Path(data["path"]).is_absolute())
        self.assertEqual(Path(data["path"]), version_dir)
        self.assertEqual(data["train"], "images/train")
        self.assertEqual(data["val"], "images/val")
        self.assertEqual(data["test"], "images/test")
        self.assertEqual(data["names"], {0: "bolt", 1: "nut"})
        self.assertEqual((version_dir / "classes.txt").read_text(encoding="utf-8"), "bolt\nnut\n")
        self.assertEqual(
            (self.dataset_dir / "data.yaml").read_text(encoding="utf-8"),
            yaml_path.read_text(encoding="utf-8"),
        )

    def test_seed_is_reproducible_and_versions_are_preserved_on_collision(self) -> None:
        self.make_source()
        fixed_now = datetime(2026, 7, 28, 12, 34, 56)

        with mock.patch.object(builder, "datetime", wraps=datetime) as mocked_datetime:
            mocked_datetime.now.return_value = fixed_now
            first_yaml = builder.build_yolo_dataset(self.raw_dir, self.dataset_dir, random_seed=7)
            second_yaml = builder.build_yolo_dataset(self.raw_dir, self.dataset_dir, random_seed=7)

        self.assertEqual(first_yaml.parent.name, "20260728_123456")
        self.assertEqual(second_yaml.parent.name, "20260728_123456_01")
        self.assertTrue(first_yaml.exists())
        self.assertTrue(second_yaml.exists())
        self.assertEqual(self.split_names(first_yaml.parent), self.split_names(second_yaml.parent))
        stable = yaml.safe_load((self.dataset_dir / "data.yaml").read_text(encoding="utf-8"))
        self.assertEqual(Path(stable["path"]), second_yaml.parent)

    def test_detection_and_obb_each_build_but_mixing_is_rejected(self) -> None:
        with self.subTest("detection"):
            self.make_source(label_text=DETECTION_LINE)
            self.assertTrue(builder.build_yolo_dataset(self.raw_dir, self.dataset_dir).exists())

        # Use fresh roots because successful builds intentionally preserve history.
        self.tearDown()
        self.setUp()
        with self.subTest("obb"):
            self.make_source(label_text=OBB_LINE)
            self.assertTrue(builder.build_yolo_dataset(self.raw_dir, self.dataset_dir).exists())

        (self.raw_dir / "sample_00.txt").write_text(DETECTION_LINE, encoding="utf-8")
        with self.assertRaisesRegex(builder.DatasetValidationError, "cannot be mixed"):
            builder.build_yolo_dataset(self.raw_dir, self.dataset_dir)

    def test_one_empty_label_is_allowed_but_all_empty_is_rejected(self) -> None:
        self.make_source()
        (self.raw_dir / "sample_00.txt").write_text("", encoding="utf-8")
        self.assertTrue(builder.build_yolo_dataset(self.raw_dir, self.dataset_dir).exists())

        for label in self.raw_dir.glob("sample_*.txt"):
            label.write_text("\n", encoding="utf-8")
        with self.assertRaisesRegex(builder.DatasetValidationError, "All label files are empty"):
            builder.build_yolo_dataset(self.raw_dir, self.dataset_dir)

    def test_missing_or_duplicate_classes_are_rejected(self) -> None:
        (self.raw_dir / "one.jpg").write_bytes(b"image")
        (self.raw_dir / "one.txt").write_text(DETECTION_LINE, encoding="utf-8")

        with self.assertRaises(FileNotFoundError):
            builder.build_yolo_dataset(self.raw_dir, self.dataset_dir)

        for classes, message in (("\n", "at least one"), ("part\npart\n", "duplicate")):
            with self.subTest(classes=classes):
                (self.raw_dir / "classes.txt").write_text(classes, encoding="utf-8")
                with self.assertRaisesRegex(builder.DatasetValidationError, message):
                    builder.build_yolo_dataset(self.raw_dir, self.dataset_dir)

    def test_duplicate_image_stem_missing_label_and_orphan_label_are_rejected(self) -> None:
        self.make_source()
        (self.raw_dir / "sample_00.png").write_bytes(b"duplicate")
        with self.assertRaisesRegex(builder.DatasetValidationError, "Duplicate image stems"):
            builder.build_yolo_dataset(self.raw_dir, self.dataset_dir)

        (self.raw_dir / "sample_00.png").unlink()
        (self.raw_dir / "sample_00.txt").unlink()
        with self.assertRaisesRegex(builder.DatasetValidationError, "missing labels"):
            builder.build_yolo_dataset(self.raw_dir, self.dataset_dir)

        (self.raw_dir / "sample_00.txt").write_text(DETECTION_LINE, encoding="utf-8")
        (self.raw_dir / "orphan.txt").write_text(DETECTION_LINE, encoding="utf-8")
        with self.assertRaisesRegex(builder.DatasetValidationError, "orphan labels"):
            builder.build_yolo_dataset(self.raw_dir, self.dataset_dir)

    def test_invalid_label_values_are_rejected(self) -> None:
        cases = {
            "wrong column count": ("0 0.5 0.5 0.2 0.2 0.1\n", "expected 5 columns"),
            "non-integer class": ("0.0 0.5 0.5 0.2 0.2\n", "must be an integer"),
            "class out of range": ("1 0.5 0.5 0.2 0.2\n", "outside the valid range"),
            "not numeric": ("0 value 0.5 0.2 0.2\n", "must be a number"),
            "not finite": ("0 nan 0.5 0.2 0.2\n", "must be finite"),
            "not normalized": ("0 1.1 0.5 0.2 0.2\n", "normalized"),
        }
        for name, (label, message) in cases.items():
            with self.subTest(name=name):
                self.tearDown()
                self.setUp()
                self.make_source(label_text=label)
                with self.assertRaisesRegex(builder.DatasetValidationError, message):
                    builder.build_yolo_dataset(self.raw_dir, self.dataset_dir)

    def test_ratios_and_empty_splits_are_rejected(self) -> None:
        self.make_source(count=2)

        invalid_ratios = (
            (0.0, 0.5, 0.5),
            (0.7, 0.2, 0.2),
            (float("nan"), 0.5, 0.5),
        )
        for ratios in invalid_ratios:
            with self.subTest(ratios=ratios):
                with self.assertRaises(builder.DatasetValidationError):
                    builder.build_yolo_dataset(
                        self.raw_dir,
                        self.dataset_dir,
                        train_ratio=ratios[0],
                        val_ratio=ratios[1],
                        test_ratio=ratios[2],
                    )

        with self.assertRaisesRegex(builder.DatasetValidationError, "empty split"):
            builder.build_yolo_dataset(self.raw_dir, self.dataset_dir)

    def test_failed_rebuild_keeps_stable_yaml_and_history_unchanged(self) -> None:
        self.make_source()
        first_yaml = builder.build_yolo_dataset(self.raw_dir, self.dataset_dir)
        stable_before = (self.dataset_dir / "data.yaml").read_bytes()
        versions_before = {path.name for path in (self.dataset_dir / "versions").iterdir()}

        (self.raw_dir / "sample_00.txt").write_text("0 nan 0.5 0.2 0.2\n", encoding="utf-8")
        with self.assertRaises(builder.DatasetValidationError):
            builder.build_yolo_dataset(self.raw_dir, self.dataset_dir)

        self.assertTrue(first_yaml.exists())
        self.assertEqual((self.dataset_dir / "data.yaml").read_bytes(), stable_before)
        self.assertEqual(
            {path.name for path in (self.dataset_dir / "versions").iterdir()}, versions_before
        )
        self.assertEqual(list(self.dataset_dir.glob(".staging_*")), [])

    def test_stable_yaml_write_failure_rolls_back_new_version(self) -> None:
        self.make_source()
        first_yaml = builder.build_yolo_dataset(self.raw_dir, self.dataset_dir)
        stable_before = (self.dataset_dir / "data.yaml").read_bytes()
        versions_before = {path.name for path in (self.dataset_dir / "versions").iterdir()}

        with mock.patch.object(builder.os, "replace", side_effect=OSError("simulated failure")):
            with self.assertRaisesRegex(OSError, "simulated failure"):
                builder.build_yolo_dataset(self.raw_dir, self.dataset_dir)

        self.assertTrue(first_yaml.exists())
        self.assertEqual((self.dataset_dir / "data.yaml").read_bytes(), stable_before)
        self.assertEqual(
            {path.name for path in (self.dataset_dir / "versions").iterdir()}, versions_before
        )
        self.assertEqual(list(self.dataset_dir.glob(".staging_*")), [])


if __name__ == "__main__":
    unittest.main()
