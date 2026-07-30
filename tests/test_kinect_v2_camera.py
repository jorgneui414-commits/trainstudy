"""不连接真实硬件时验证 Kinect v2 读取器的边界行为。

这里不测试真实取帧，而是验证单位转换、错误顺序和参数校验；因此普通开发环境也能运行。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from kinect_v2_camera import KinectV2Camera, _meters_to_millimeters


class KinectV2CameraTests(unittest.TestCase):
    def test_camera_space_coordinates_are_converted_to_millimeters(self) -> None:
        # SDK 的 CameraSpacePoint 原始单位是米，包含正负方向坐标。
        camera_xyz_m = np.array([[[0.1, -0.2, 1.0]]], dtype=np.float32)

        camera_xyz_mm = _meters_to_millimeters(camera_xyz_m)

        np.testing.assert_allclose(camera_xyz_mm, [[[100.0, -200.0, 1000.0]]])
        # 转换函数应返回新数组，不能顺便把调用者的米制原数组改掉。
        np.testing.assert_allclose(camera_xyz_m, [[[0.1, -0.2, 1.0]]])

    def test_missing_sdk_is_reported_before_loading_pythonnet(self) -> None:
        # 临时目录中的路径一定不存在，用它验证程序能给出清楚的 SDK 缺失提示。
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_path = Path(temp_dir) / "Microsoft.Kinect.dll"
            camera = KinectV2Camera(missing_path)
            with self.assertRaisesRegex(FileNotFoundError, "Kinect SDK"):
                camera.open()

        self.assertFalse(camera.is_open)

    def test_read_before_open_is_rejected(self) -> None:
        # 未调用 open() 时没有读取器，read() 应立即拒绝这种调用顺序。
        camera = KinectV2Camera("unused.dll")
        with self.assertRaisesRegex(RuntimeError, "尚未打开"):
            camera.read()

    def test_non_positive_timeouts_are_rejected(self) -> None:
        # 等待时间为 0 没有实际意义；open() 和 read() 都必须验证该参数。
        camera = KinectV2Camera("unused.dll")
        with self.assertRaises(ValueError):
            camera.open(availability_timeout_seconds=0)
        with self.assertRaises(RuntimeError):
            camera.read(timeout_seconds=0)


if __name__ == "__main__":
    unittest.main()
