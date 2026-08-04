"""第六步单次 Kinect OBB 深度检测的纯模拟测试。"""

from __future__ import annotations

import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest import mock

import numpy as np

import kinect_obb_detection


class FakeObb:
    """只实现 parse_yolo_result() 会访问的 OBB 结果字段。"""

    def __init__(
        self,
        polygons: np.ndarray,
        classes: np.ndarray,
        confidences: np.ndarray,
    ) -> None:
        self.xyxyxyxy = polygons
        self.cls = classes
        self.conf = confidences

    def __len__(self) -> int:
        return len(self.xyxyxyxy)


class FakeCamera:
    def __init__(self, frame: SimpleNamespace) -> None:
        self.frame = frame
        self.read_calls = 0
        self.timeout_seconds: float | None = None

    def read(self, *, timeout_seconds: float) -> SimpleNamespace:
        self.read_calls += 1
        self.timeout_seconds = timeout_seconds
        return self.frame


class FakeModel:
    def __init__(self, result: SimpleNamespace, *, task: str = "obb") -> None:
        self.task = task
        self.result = result
        self.predict_calls = 0
        self.predict_kwargs: dict[str, object] = {}

    def predict(self, **kwargs: object) -> list[SimpleNamespace]:
        self.predict_calls += 1
        self.predict_kwargs = kwargs
        return [self.result]


def make_frame() -> SimpleNamespace:
    """构造三个彩色映射点：两个深度有效，一个深度无效。"""

    depth_mm = np.array([[1000, 0], [0, 1200]], dtype=np.uint16)
    depth_to_color_xy = np.array(
        [[[25.0, 25.0], [75.0, 25.0]], [[25.0, 75.0], [75.0, 75.0]]],
        dtype=np.float32,
    )
    camera_xyz_mm = np.array(
        [
            [[100.0, -50.0, 1000.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [300.0, 200.0, 1200.0]],
        ],
        dtype=np.float32,
    )
    return SimpleNamespace(
        color_bgr=np.zeros((100, 100, 3), dtype=np.uint8),
        depth_mm=depth_mm,
        depth_to_color_xy=depth_to_color_xy,
        depth_to_camera_xyz_mm=camera_xyz_mm,
    )


def rectangle(center_x: float, center_y: float) -> np.ndarray:
    return np.array(
        [
            [center_x - 15.0, center_y - 10.0],
            [center_x + 15.0, center_y - 10.0],
            [center_x + 15.0, center_y + 10.0],
            [center_x - 15.0, center_y + 10.0],
        ],
        dtype=np.float32,
    )


def make_result(
    polygons: list[np.ndarray],
    class_ids: list[int],
    names: dict[int, str],
) -> SimpleNamespace:
    polygon_array = (
        np.asarray(polygons, dtype=np.float32).reshape((-1, 4, 2))
        if polygons
        else np.empty((0, 4, 2), dtype=np.float32)
    )
    return SimpleNamespace(
        names=names,
        obb=FakeObb(
            polygon_array,
            np.asarray(class_ids, dtype=np.float32),
            np.full(len(class_ids), 0.9, dtype=np.float32),
        ),
    )


class KinectObbDetectionTests(unittest.TestCase):
    def capture(
        self,
        result: SimpleNamespace,
    ) -> tuple[dict[str, object], FakeCamera, FakeModel]:
        camera = FakeCamera(make_frame())
        model = FakeModel(result)
        # 小型模拟深度图每个目标只有一个映射点，因此把最小样本数临时设为 1。
        with mock.patch.object(
            kinect_obb_detection.config,
            "DEPTH_MIN_VALID_SAMPLES",
            1,
        ):
            output = kinect_obb_detection.capture_and_detect(camera, model)
        return output, camera, model

    def test_camera_and_model_are_each_called_exactly_once(self) -> None:
        output, camera, model = self.capture(
            make_result([rectangle(25.0, 25.0)], [0], {0: "螺丝刀"})
        )

        self.assertEqual(camera.read_calls, 1)
        self.assertEqual(model.predict_calls, 1)
        self.assertIs(model.predict_kwargs["source"], camera.frame.color_bgr)
        self.assertFalse(model.predict_kwargs["save"])
        self.assertEqual(len(output["detections"]), 1)

    def test_empty_obb_result_returns_empty_detections(self) -> None:
        output, camera, model = self.capture(make_result([], [], {}))

        self.assertEqual(output["detections"], [])
        self.assertEqual(camera.read_calls, 1)
        self.assertEqual(model.predict_calls, 1)

    def test_invalid_depth_is_filtered_and_chinese_name_is_preserved(self) -> None:
        output, _camera, _model = self.capture(
            make_result(
                [rectangle(25.0, 25.0), rectangle(75.0, 25.0)],
                [0, 1],
                {0: "螺丝刀", 1: "无效深度零件"},
            )
        )

        detections = output["detections"]
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["part_type"], "螺丝刀")
        self.assertTrue(detections[0]["depth_valid"])

    def test_all_valid_targets_and_robot_fields_are_returned(self) -> None:
        output, _camera, _model = self.capture(
            make_result(
                [rectangle(25.0, 25.0), rectangle(75.0, 75.0)],
                [0, 1],
                {0: "螺丝刀", 1: "支架"},
            )
        )

        detections = output["detections"]
        self.assertEqual([item["part_type"] for item in detections], ["螺丝刀", "支架"])
        self.assertEqual([item["distance_mm"] for item in detections], [1000.0, 1200.0])
        required_fields = {
            "class_id",
            "part_type",
            "confidence",
            "center_x",
            "center_y",
            "bbox_width",
            "bbox_height",
            "obb_points",
            "orientation_deg",
            "distance_mm",
            "camera_x_mm",
            "camera_y_mm",
            "camera_z_mm",
        }
        for detection in detections:
            self.assertTrue(required_fields.issubset(detection))
            self.assertGreaterEqual(float(detection["orientation_deg"]), 0.0)
            self.assertLess(float(detection["orientation_deg"]), 180.0)

        self.assertEqual(output["image_width"], 100)
        self.assertEqual(output["image_height"], 100)
        self.assertEqual(output["coordinate_frame"], "kinect_camera")
        self.assertEqual(output["coordinate_unit"], "mm")
        captured_at = str(output["captured_at"])
        self.assertTrue(captured_at.endswith("Z"))
        self.assertIsNotNone(datetime.fromisoformat(captured_at.replace("Z", "+00:00")))

    def test_non_obb_model_is_rejected_before_camera_or_prediction(self) -> None:
        camera = FakeCamera(make_frame())
        model = FakeModel(make_result([], [], {}), task="detect")

        with self.assertRaisesRegex(ValueError, "只接受.*obb"):
            kinect_obb_detection.capture_and_detect(camera, model)

        self.assertEqual(camera.read_calls, 0)
        self.assertEqual(model.predict_calls, 0)

    def test_camera_read_failure_is_distinguished_from_model_failure(self) -> None:
        class FailingCamera(FakeCamera):
            def read(self, *, timeout_seconds: float) -> SimpleNamespace:
                self.read_calls += 1
                self.timeout_seconds = timeout_seconds
                raise RuntimeError("no synchronized frame")

        camera = FailingCamera(make_frame())
        model = FakeModel(make_result([], [], {}))

        with self.assertRaisesRegex(
            kinect_obb_detection.KinectFrameError,
            "Kinect RGB-D 取帧失败",
        ):
            kinect_obb_detection.capture_and_detect(camera, model)

        self.assertEqual(camera.read_calls, 1)
        self.assertEqual(model.predict_calls, 0)


if __name__ == "__main__":
    unittest.main()
