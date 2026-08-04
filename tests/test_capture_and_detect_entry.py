"""第六步一次性入口的资源释放测试。"""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "06_capture_and_detect.py"
SPEC = importlib.util.spec_from_file_location("capture_and_detect_06", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
capture_script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = capture_script
SPEC.loader.exec_module(capture_script)


class FakeCamera:
    def __init__(self, _sdk_path: Path) -> None:
        self.open_calls = 0
        self.release_calls = 0

    def open(self, *, availability_timeout_seconds: float) -> None:
        self.open_calls += 1

    def release(self) -> None:
        self.release_calls += 1


class CaptureAndDetectEntryTests(unittest.TestCase):
    def test_main_loads_once_opens_detects_prints_and_releases(self) -> None:
        model = SimpleNamespace(task="obb")
        camera = FakeCamera(Path("sdk.dll"))
        expected = {
            "captured_at": "2026-07-30T00:00:00.000Z",
            "image_width": 1920,
            "image_height": 1080,
            "coordinate_frame": "kinect_camera",
            "coordinate_unit": "mm",
            "detections": [{"part_type": "螺丝刀"}],
        }
        yolo_factory = mock.Mock(return_value=model)

        with (
            mock.patch.object(capture_script.config, "INFERENCE_MODEL_PATH", SCRIPT_PATH),
            mock.patch.object(capture_script, "KinectV2Camera", return_value=camera),
            mock.patch.object(capture_script, "capture_and_detect", return_value=expected) as detect,
            mock.patch.dict(sys.modules, {"ultralytics": SimpleNamespace(YOLO=yolo_factory)}),
            mock.patch("builtins.print") as print_mock,
        ):
            capture_script.main()

        yolo_factory.assert_called_once_with(str(SCRIPT_PATH))
        self.assertEqual(camera.open_calls, 1)
        detect.assert_called_once_with(camera, model)
        self.assertEqual(camera.release_calls, 1)
        printed = json.loads(print_mock.call_args.args[0])
        self.assertEqual(printed, expected)


if __name__ == "__main__":
    unittest.main()
