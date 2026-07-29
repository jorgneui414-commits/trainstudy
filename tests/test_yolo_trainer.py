"""Tests for YOLO training validation without downloading a model or training."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yolo_trainer as trainer


DETECTION_LINE = "0 0.5 0.5 0.2 0.2\n"
OBB_LINE = "0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n"


class FakeCuda:
    def __init__(self, *, available: bool, count: int = 1) -> None:
        self.available = available
        self.count = count

    def is_available(self) -> bool:
        return self.available

    def device_count(self) -> int:
        return self.count


class FakeTorch:
    def __init__(self, *, available: bool, count: int = 1) -> None:
        self.cuda = FakeCuda(available=available, count=count)


class FakeModel:
    def __init__(self, task: str, save_dir: Path) -> None:
        self.task = task
        self.save_dir = save_dir
        self.train_kwargs: dict[str, object] | None = None

    def train(self, **kwargs: object) -> SimpleNamespace:
        self.train_kwargs = kwargs
        return SimpleNamespace(save_dir=self.save_dir)


class YoloTrainerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.data_yaml = self.root / "data.yaml"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def make_dataset(self, label_text: str = DETECTION_LINE) -> Path:
        dataset_root = self.root / "dataset"
        for split_name in trainer.SPLIT_NAMES:
            image_dir = dataset_root / "images" / split_name
            label_dir = dataset_root / "labels" / split_name
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            (image_dir / f"{split_name}.bmp").write_bytes(b"image")
            (label_dir / f"{split_name}.txt").write_text(label_text, encoding="utf-8")

        self.data_yaml.write_text(
            yaml.safe_dump(
                {
                    "path": dataset_root.as_posix(),
                    "train": "images/train",
                    "val": "images/val",
                    "test": "images/test",
                    "names": {0: "part"},
                },
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        return dataset_root

    def make_options(self, **changes: object) -> trainer.TrainingOptions:
        values: dict[str, object] = {
            "data_yaml": self.data_yaml,
            "task": "detect",
            "model": "yolo11n.pt",
            "device": "cpu",
            "gpu_index": 0,
            "epochs": 200,
            "image_size": 640,
            "batch_size": 8,
            "patience": 50,
            "workers": 4,
            "cache": False,
            "amp": True,
            "seed": 42,
            "output_dir": self.root / "runs" / "train",
            "run_name": "unit_test",
            "exist_ok": False,
            "save_period": 10,
        }
        values.update(changes)
        return trainer.TrainingOptions(**values)  # type: ignore[arg-type]

    def test_inspect_detect_and_obb_dataset_tasks(self) -> None:
        dataset_root = self.make_dataset(DETECTION_LINE)
        detected = trainer.inspect_dataset_task(self.data_yaml)

        self.assertEqual(detected.task, "detect")
        self.assertEqual(detected.dataset_root, dataset_root.resolve())
        self.assertEqual(detected.label_file_count, 3)
        self.assertEqual(detected.annotation_count, 3)

        self.tearDown()
        self.setUp()
        self.make_dataset(OBB_LINE)
        self.assertEqual(trainer.inspect_dataset_task(self.data_yaml).task, "obb")

    def test_missing_data_yaml_explains_that_step_two_is_required(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "02_build_yolo_dataset"):
            trainer.inspect_dataset_task(self.data_yaml)

    def test_mixed_label_columns_are_rejected(self) -> None:
        dataset_root = self.make_dataset(DETECTION_LINE)
        (dataset_root / "labels" / "val" / "val.txt").write_text(OBB_LINE, encoding="utf-8")

        with self.assertRaisesRegex(trainer.TrainingConfigurationError, "不能混用"):
            trainer.inspect_dataset_task(self.data_yaml)

    def test_cpu_and_gpu_device_choices(self) -> None:
        self.assertEqual(trainer.resolve_training_device("cpu", 0), "cpu")
        self.assertEqual(
            trainer.resolve_training_device("gpu", 0, torch_module=FakeTorch(available=True)),
            0,
        )

    def test_unavailable_or_invalid_gpu_is_rejected(self) -> None:
        with self.assertRaisesRegex(trainer.TrainingRuntimeError, "未检测到 CUDA"):
            trainer.resolve_training_device("gpu", 0, torch_module=FakeTorch(available=False))
        with self.assertRaisesRegex(trainer.TrainingRuntimeError, "不可用"):
            trainer.resolve_training_device("gpu", 1, torch_module=FakeTorch(available=True, count=1))

    def test_dataset_task_mismatch_is_rejected_before_model_load(self) -> None:
        self.make_dataset(DETECTION_LINE)
        factory_called = False

        def factory(_model: str) -> FakeModel:
            nonlocal factory_called
            factory_called = True
            return FakeModel("obb", self.root / "unused")

        with self.assertRaisesRegex(trainer.TrainingConfigurationError, "不匹配"):
            trainer.train_yolo(
                self.make_options(task="obb", model="yolo11n-obb.pt"),
                yolo_factory=factory,
                printer=lambda _message: None,
            )
        self.assertFalse(factory_called)

    def test_model_task_mismatch_is_rejected(self) -> None:
        self.make_dataset()
        model = FakeModel("obb", self.root / "unused")

        with self.assertRaisesRegex(trainer.TrainingConfigurationError, "模型"):
            trainer.train_yolo(
                self.make_options(),
                yolo_factory=lambda _model: model,
                printer=lambda _message: None,
            )
        self.assertIsNone(model.train_kwargs)

    def test_gpu_device_is_forwarded_when_cuda_is_available(self) -> None:
        self.make_dataset()
        model = FakeModel("detect", self.root / "runs" / "train" / "gpu_test")

        trainer.train_yolo(
            self.make_options(device="gpu", run_name="gpu_test"),
            yolo_factory=lambda _model: model,
            torch_module=FakeTorch(available=True),
            printer=lambda _message: None,
        )

        self.assertIsNotNone(model.train_kwargs)
        assert model.train_kwargs is not None
        self.assertEqual(model.train_kwargs["device"], 0)

    def test_obb_model_and_obb_labels_can_train_together(self) -> None:
        self.make_dataset(OBB_LINE)
        model = FakeModel("obb", self.root / "runs" / "train" / "obb_test")

        trainer.train_yolo(
            self.make_options(task="obb", model="yolo11n-obb.pt", run_name="obb_test"),
            yolo_factory=lambda _model: model,
            printer=lambda _message: None,
        )

        self.assertIsNotNone(model.train_kwargs)
        assert model.train_kwargs is not None
        self.assertEqual(model.train_kwargs["device"], "cpu")
        self.assertEqual(model.task, "obb")

    def test_training_options_are_forwarded_to_compatible_model(self) -> None:
        self.make_dataset()
        expected_save_dir = self.root / "runs" / "train" / "unit_test"
        model = FakeModel("detect", expected_save_dir)
        messages: list[str] = []

        actual_save_dir = trainer.train_yolo(
            self.make_options(),
            yolo_factory=lambda _model: model,
            printer=messages.append,
        )

        self.assertEqual(actual_save_dir, expected_save_dir.resolve())
        self.assertIsNotNone(model.train_kwargs)
        assert model.train_kwargs is not None
        self.assertEqual(model.train_kwargs["data"], str(self.data_yaml.resolve()))
        self.assertEqual(model.train_kwargs["device"], "cpu")
        self.assertEqual(model.train_kwargs["epochs"], 200)
        self.assertEqual(model.train_kwargs["imgsz"], 640)
        self.assertEqual(model.train_kwargs["batch"], 8)
        self.assertEqual(model.train_kwargs["patience"], 50)
        self.assertEqual(model.train_kwargs["workers"], 4)
        self.assertEqual(model.train_kwargs["cache"], False)
        self.assertEqual(model.train_kwargs["amp"], True)
        self.assertEqual(model.train_kwargs["seed"], 42)
        self.assertEqual(model.train_kwargs["save_period"], 10)
        self.assertTrue(any("开始 YOLO 训练" in message for message in messages))


if __name__ == "__main__":
    unittest.main()
