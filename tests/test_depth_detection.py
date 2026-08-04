"""Detect/OBB 解析、毫米制 Kinect 深度融合和预览测试。

这些测试使用很小的 NumPy 数组模拟 YOLO 与 Kinect 数据，因此不需要连接真实相机，
可以快速验证第五步的核心计算是否在以后修改代码时被意外破坏。
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from depth_detection import (
    colorize_depth,
    enrich_detections_with_depth,
    parse_yolo_result,
)


class FakeBoxes:
    """模拟 Ultralytics Detect 结果中本测试真正会使用的 boxes 属性。"""

    def __init__(self, xyxy: np.ndarray, classes: np.ndarray, confidences: np.ndarray) -> None:
        self.xyxy = xyxy
        self.cls = classes
        self.conf = confidences

    def __len__(self) -> int:
        return len(self.xyxy)


class FakeObb:
    """模拟 Ultralytics OBB 结果，避免测试时加载真实模型和 GPU。"""

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


def horizontal_detection() -> dict[str, object]:
    """创建一个覆盖 100x100 彩色图的基础二维检测框。"""

    return {
        "part_type": "part",
        "class_id": 0,
        "confidence": 0.9,
        "x1": 0.0,
        "y1": 0.0,
        "x2": 100.0,
        "y2": 100.0,
        "center_x": 50.0,
        "center_y": 50.0,
        "bbox_width": 100.0,
        "bbox_height": 100.0,
        "orientation_deg": None,
        "box_type": "horizontal",
    }


class DepthDetectionTests(unittest.TestCase):
    def test_detect_result_uses_horizontal_box_without_angle(self) -> None:
        # SimpleNamespace 可临时组合出 result.names 和 result.boxes 两个属性。
        result = SimpleNamespace(
            names={2: "bolt"},
            boxes=FakeBoxes(
                np.array([[10.0, 20.0, 30.0, 60.0]], dtype=np.float32),
                np.array([2.0], dtype=np.float32),
                np.array([0.75], dtype=np.float32),
            ),
        )

        detection = parse_yolo_result(result, task="detect")[0]

        self.assertEqual(detection["part_type"], "bolt")
        self.assertEqual(detection["center_x"], 20.0)
        self.assertEqual(detection["center_y"], 40.0)
        self.assertIsNone(detection["orientation_deg"])
        self.assertEqual(detection["box_type"], "horizontal")

    def test_obb_result_includes_polygon_and_orientation(self) -> None:
        # 四个点组成没有旋转的矩形，所以预期方向角是 0 度。
        result = SimpleNamespace(
            names={0: "plate"},
            obb=FakeObb(
                np.array([[[1.0, 2.0], [5.0, 2.0], [5.0, 4.0], [1.0, 4.0]]]),
                np.array([0.0]),
                np.array([0.8]),
            ),
        )

        detection = parse_yolo_result(result, task="obb")[0]

        self.assertEqual(detection["box_type"], "obb")
        self.assertEqual(
            detection["obb_points"],
            [[1.0, 2.0], [5.0, 2.0], [5.0, 4.0], [1.0, 4.0]],
        )
        self.assertAlmostEqual(float(detection["orientation_deg"]), 0.0)

    def test_obb_orientation_uses_long_axis_and_ignores_corner_order(self) -> None:
        # 图像坐标 y 向下，因此 30 度表示长轴从右方向顺时针倾斜 30 度。
        angle_rad = np.deg2rad(30.0)
        long_axis = np.array([np.cos(angle_rad), np.sin(angle_rad)]) * 20.0
        short_axis = np.array([-np.sin(angle_rad), np.cos(angle_rad)]) * 5.0
        center = np.array([50.0, 50.0])
        inclined = np.array(
            [
                center - long_axis - short_axis,
                center + long_axis - short_axis,
                center + long_axis + short_axis,
                center - long_axis + short_axis,
            ]
        )

        cases = {
            "horizontal": (
                np.array([[10.0, 20.0], [50.0, 20.0], [50.0, 30.0], [10.0, 30.0]]),
                0.0,
            ),
            "vertical": (
                np.array([[20.0, 10.0], [30.0, 10.0], [30.0, 50.0], [20.0, 50.0]]),
                90.0,
            ),
            "inclined": (inclined, 30.0),
            # 故意打乱起点和顺序；长轴角度仍应与上一个案例相同。
            "shuffled": (inclined[[2, 0, 3, 1]], 30.0),
        }
        for label, (polygon, expected_angle) in cases.items():
            with self.subTest(label=label):
                result = SimpleNamespace(
                    names={0: "part"},
                    obb=FakeObb(
                        polygon.reshape((1, 4, 2)),
                        np.array([0.0]),
                        np.array([0.9]),
                    ),
                )

                orientation = float(
                    parse_yolo_result(result, task="obb")[0]["orientation_deg"]
                )

                self.assertGreaterEqual(orientation, 0.0)
                self.assertLess(orientation, 180.0)
                self.assertAlmostEqual(orientation, expected_angle, places=5)

    def test_empty_results_and_invalid_task_are_handled(self) -> None:
        # 第一维长度为 0 表示这一帧没有检测到任何目标。
        detect_result = SimpleNamespace(
            names={},
            boxes=FakeBoxes(np.empty((0, 4)), np.empty(0), np.empty(0)),
        )
        obb_result = SimpleNamespace(
            names={},
            obb=FakeObb(np.empty((0, 4, 2)), np.empty(0), np.empty(0)),
        )

        self.assertEqual(parse_yolo_result(detect_result, task="detect"), [])
        self.assertEqual(parse_yolo_result(obb_result, task="obb"), [])
        with self.assertRaisesRegex(ValueError, "detect.*obb"):
            parse_yolo_result(detect_result, task="segment")

    def test_central_roi_returns_only_millimeter_fields(self) -> None:
        # 构造 4x4 深度像素，并把它们映射到彩色图的四个横纵坐标位置。
        axis = np.array([10.0, 40.0, 60.0, 90.0], dtype=np.float32)
        color_x, color_y = np.meshgrid(axis, axis)
        depth_to_color = np.stack([color_x, color_y], axis=-1)

        # 外圈为 4000 mm，中心 2x2 为 1000 mm；30% 中央 ROI 应只选中中心四点。
        depth_mm = np.full((4, 4), 4000, dtype=np.uint16)
        depth_mm[1:3, 1:3] = 1000
        camera_xyz_mm = np.zeros((4, 4, 3), dtype=np.float32)
        camera_xyz_mm[..., 0] = 100.0
        camera_xyz_mm[..., 1] = -200.0
        camera_xyz_mm[..., 2] = depth_mm.astype(np.float32)

        detection = enrich_detections_with_depth(
            [horizontal_detection()],
            depth_mm,
            depth_to_color,
            camera_xyz_mm,
            image_width=100,
            image_height=100,
            roi_ratio=0.30,
            min_valid_samples=4,
        )[0]

        self.assertTrue(detection["depth_valid"])
        self.assertEqual(detection["depth_sample_count"], 4)
        self.assertEqual(detection["distance_mm"], 1000.0)
        self.assertEqual(detection["camera_x_mm"], 100.0)
        self.assertEqual(detection["camera_y_mm"], -200.0)
        self.assertEqual(detection["camera_z_mm"], 1000.0)
        # 这些是重构前的米制或机器人字段，防止以后被误加回精简接口。
        for removed_field in (
            "depth_z",
            "camera_x",
            "camera_y",
            "camera_z",
            "robot_x_mm",
            "robot_y_mm",
            "robot_z_mm",
        ):
            self.assertNotIn(removed_field, detection)

    def test_too_few_valid_samples_keeps_2d_detection(self) -> None:
        # 四个深度像素中只有一个有效，而配置要求至少两个有效点。
        depth_mm = np.array([[0, 1000], [0, 0]], dtype=np.uint16)
        depth_to_color = np.full((2, 2, 2), 50.0, dtype=np.float32)
        camera_xyz_mm = np.zeros((2, 2, 3), dtype=np.float32)
        camera_xyz_mm[..., 2] = depth_mm.astype(np.float32)

        detection = enrich_detections_with_depth(
            [horizontal_detection()],
            depth_mm,
            depth_to_color,
            camera_xyz_mm,
            image_width=100,
            image_height=100,
            min_valid_samples=2,
        )[0]

        self.assertFalse(detection["depth_valid"])
        self.assertEqual(detection["depth_sample_count"], 1)
        self.assertIsNone(detection["distance_mm"])
        self.assertIsNone(detection["camera_x_mm"])
        self.assertIsNone(detection["camera_y_mm"])
        self.assertIsNone(detection["camera_z_mm"])
        self.assertEqual(detection["center_x"], 50.0)
        self.assertEqual(detection["center_y"], 50.0)

    def test_depth_outside_reliable_range_is_invalid(self) -> None:
        # 默认最大深度为 4500 mm，因此 4900 mm 的点全部应被过滤。
        depth_mm = np.full((3, 3), 4900, dtype=np.uint16)
        depth_to_color = np.full((3, 3, 2), 50.0, dtype=np.float32)
        camera_xyz_mm = np.zeros((3, 3, 3), dtype=np.float32)
        camera_xyz_mm[..., 2] = 4900.0

        detection = enrich_detections_with_depth(
            [horizontal_detection()],
            depth_mm,
            depth_to_color,
            camera_xyz_mm,
            image_width=100,
            image_height=100,
        )[0]

        self.assertFalse(detection["depth_valid"])
        self.assertEqual(detection["depth_sample_count"], 0)

    def test_depth_preview_marks_invalid_pixels_black(self) -> None:
        # 0 和 5000 超出指定范围，应在伪彩色预览中保持黑色。
        preview = colorize_depth(
            np.array([[0, 500], [4500, 5000]], dtype=np.uint16),
            min_depth_mm=500,
            max_depth_mm=4500,
        )

        self.assertEqual(preview.shape, (2, 2, 3))
        self.assertTrue(np.array_equal(preview[0, 0], [0, 0, 0]))
        self.assertTrue(np.array_equal(preview[1, 1], [0, 0, 0]))
        self.assertFalse(np.array_equal(preview[0, 1], [0, 0, 0]))


if __name__ == "__main__":
    # 允许直接运行本测试文件，也允许由 unittest discover 自动发现。
    unittest.main()
