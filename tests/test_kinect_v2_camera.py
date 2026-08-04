"""不连接真实硬件时验证 Kinect v2 读取器的缓冲区和边界行为。"""

from __future__ import annotations

import tempfile
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import numpy as np

from kinect_v2_camera import KinectV2Camera, _meters_to_millimeters


class FakeDisposable:
    def __init__(self) -> None:
        self.dispose_calls = 0

    def Dispose(self) -> None:
        self.dispose_calls += 1


class FakeColorFrame(FakeDisposable):
    def __init__(
        self,
        target: np.ndarray,
        values: np.ndarray,
        *,
        error: Exception | None = None,
    ) -> None:
        super().__init__()
        self.target = target
        self.values = values
        self.error = error
        self.copy_calls: list[tuple[object, int, object]] = []

    def CopyConvertedFrameDataToIntPtr(
        self,
        pointer: object,
        byte_count: int,
        image_format: object,
    ) -> None:
        self.copy_calls.append((pointer, byte_count, image_format))
        if self.error is not None:
            raise self.error
        np.copyto(self.target, self.values)


class FakeDepthFrame(FakeDisposable):
    def __init__(self, target: np.ndarray, values: np.ndarray) -> None:
        super().__init__()
        self.target = target
        self.values = values
        self.copy_calls: list[tuple[object, int]] = []

    def CopyFrameDataToIntPtr(self, pointer: object, byte_count: int) -> None:
        self.copy_calls.append((pointer, byte_count))
        np.copyto(self.target, self.values)


class FakeCoordinateMapper:
    def __init__(
        self,
        color_target: np.ndarray,
        color_values: np.ndarray,
        camera_target: np.ndarray,
        camera_values: np.ndarray,
    ) -> None:
        self.color_target = color_target
        self.color_values = color_values
        self.camera_target = camera_target
        self.camera_values = camera_values
        self.color_calls: list[tuple[object, int, object, int]] = []
        self.camera_calls: list[tuple[object, int, object, int]] = []

    def MapDepthFrameToColorSpaceUsingIntPtr(
        self,
        depth_pointer: object,
        depth_bytes: int,
        color_pointer: object,
        color_bytes: int,
    ) -> None:
        self.color_calls.append(
            (depth_pointer, depth_bytes, color_pointer, color_bytes)
        )
        np.copyto(self.color_target, self.color_values)

    def MapDepthFrameToCameraSpaceUsingIntPtr(
        self,
        depth_pointer: object,
        depth_bytes: int,
        camera_pointer: object,
        camera_bytes: int,
    ) -> None:
        self.camera_calls.append(
            (depth_pointer, depth_bytes, camera_pointer, camera_bytes)
        )
        np.copyto(self.camera_target, self.camera_values)


class FakeMultiFrame(FakeDisposable):
    def __init__(
        self,
        color_frame: FakeColorFrame | None,
        depth_frame: FakeDepthFrame | None,
    ) -> None:
        super().__init__()
        self.ColorFrameReference = SimpleNamespace(AcquireFrame=lambda: color_frame)
        self.DepthFrameReference = SimpleNamespace(AcquireFrame=lambda: depth_frame)


class FakeReader(FakeDisposable):
    def __init__(self, results: list[object]) -> None:
        super().__init__()
        self.results = list(results)

    def AcquireLatestFrame(self) -> object | None:
        if not self.results:
            return None
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class FakeSensor:
    def __init__(self) -> None:
        self.close_calls = 0

    def Close(self) -> None:
        self.close_calls += 1


class FakeIntPtr:
    def __init__(self, address: int) -> None:
        self.address = address

    def ToInt64(self) -> int:
        return self.address


class FakeOpenSensor(FakeSensor):
    def __init__(self) -> None:
        super().__init__()
        self.IsAvailable = True
        self.open_calls = 0
        self.reader = FakeReader([])
        self.ColorFrameSource = SimpleNamespace(
            FrameDescription=SimpleNamespace(Width=2, Height=1)
        )
        self.DepthFrameSource = SimpleNamespace(
            FrameDescription=SimpleNamespace(Width=2, Height=1)
        )
        self.CoordinateMapper = object()

    def Open(self) -> None:
        self.open_calls += 1

    def OpenMultiSourceFrameReader(self, _frame_types: int) -> FakeReader:
        return self.reader


class KinectV2CameraTests(unittest.TestCase):
    def make_open_camera(self) -> KinectV2Camera:
        camera = KinectV2Camera("unused.dll")
        camera.COLOR_WIDTH = 2
        camera.COLOR_HEIGHT = 1
        camera.DEPTH_WIDTH = 2
        camera.DEPTH_HEIGHT = 1
        camera._color_buffer = np.empty((1, 2, 4), dtype=np.uint8)
        camera._depth_buffer = np.empty((1, 2), dtype=np.uint16)
        camera._color_space_points = np.empty((1, 2, 2), dtype=np.float32)
        camera._camera_space_points = np.empty((1, 2, 3), dtype=np.float32)
        camera._color_buffer_pointer = object()
        camera._depth_buffer_pointer = object()
        camera._color_space_points_pointer = object()
        camera._camera_space_points_pointer = object()
        camera._color_image_format = object()
        camera._sensor = FakeSensor()
        return camera

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

    def test_open_allocates_contiguous_numpy_buffers_and_matching_pointers(self) -> None:
        sensor = FakeOpenSensor()
        pythonnet_module = ModuleType("pythonnet")
        pythonnet_module.load = mock.Mock()
        clr_module = ModuleType("clr")
        clr_module.AddReference = mock.Mock()
        system_module = ModuleType("System")
        system_module.IntPtr = FakeIntPtr
        microsoft_module = ModuleType("Microsoft")
        microsoft_module.__path__ = []
        kinect_module = ModuleType("Microsoft.Kinect")
        kinect_module.ColorImageFormat = SimpleNamespace(Bgra=object())
        kinect_module.FrameSourceTypes = SimpleNamespace(Color=1, Depth=2)
        kinect_module.KinectSensor = SimpleNamespace(GetDefault=lambda: sensor)
        microsoft_module.Kinect = kinect_module

        with mock.patch.dict(
            sys.modules,
            {
                "pythonnet": pythonnet_module,
                "clr": clr_module,
                "System": system_module,
                "Microsoft": microsoft_module,
                "Microsoft.Kinect": kinect_module,
            },
        ):
            camera = KinectV2Camera(Path(__file__))
            camera.open()

        pythonnet_module.load.assert_called_once_with("netfx")
        clr_module.AddReference.assert_called_once_with(str(Path(__file__)))
        expected_buffers = (
            (camera._color_buffer, (1, 2, 4), np.uint8),
            (camera._depth_buffer, (1, 2), np.uint16),
            (camera._color_space_points, (1, 2, 2), np.float32),
            (camera._camera_space_points, (1, 2, 3), np.float32),
        )
        for buffer, shape, dtype in expected_buffers:
            self.assertEqual(buffer.shape, shape)
            self.assertEqual(buffer.dtype, dtype)
            self.assertTrue(buffer.flags.c_contiguous)
        pointer_pairs = (
            (camera._color_buffer_pointer, camera._color_buffer),
            (camera._depth_buffer_pointer, camera._depth_buffer),
            (camera._color_space_points_pointer, camera._color_space_points),
            (camera._camera_space_points_pointer, camera._camera_space_points),
        )
        for pointer, buffer in pointer_pairs:
            self.assertEqual(pointer.ToInt64(), buffer.ctypes.data)

        camera.release()
        self.assertEqual(sensor.open_calls, 1)
        self.assertEqual(sensor.close_calls, 1)
        self.assertEqual(sensor.reader.dispose_calls, 1)

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

    def test_read_uses_intptr_buffers_and_returns_independent_arrays(self) -> None:
        camera = self.make_open_camera()
        expected_bgra = np.array(
            [[[10, 20, 30, 255], [40, 50, 60, 255]]],
            dtype=np.uint8,
        )
        expected_depth = np.array([[1000, 2000]], dtype=np.uint16)
        expected_color_points = np.array(
            [[[100.5, 200.5], [300.5, 400.5]]],
            dtype=np.float32,
        )
        expected_camera_points_m = np.array(
            [[[0.1, -0.2, 1.0], [0.3, 0.4, 2.0]]],
            dtype=np.float32,
        )
        color_frame = FakeColorFrame(
            camera._color_buffer,
            expected_bgra,
        )
        depth_frame = FakeDepthFrame(
            camera._depth_buffer,
            expected_depth,
        )
        multi_frame = FakeMultiFrame(color_frame, depth_frame)
        mapper = FakeCoordinateMapper(
            camera._color_space_points,
            expected_color_points,
            camera._camera_space_points,
            expected_camera_points_m,
        )
        camera._coordinate_mapper = mapper
        camera._reader = FakeReader([multi_frame])

        frame = camera.read(timeout_seconds=0.1)

        self.assertEqual(
            color_frame.copy_calls,
            [
                (
                    camera._color_buffer_pointer,
                    camera._color_buffer.nbytes,
                    camera._color_image_format,
                )
            ],
        )
        self.assertEqual(
            depth_frame.copy_calls,
            [(camera._depth_buffer_pointer, camera._depth_buffer.nbytes)],
        )
        self.assertEqual(
            mapper.color_calls,
            [
                (
                    camera._depth_buffer_pointer,
                    camera._depth_buffer.nbytes,
                    camera._color_space_points_pointer,
                    camera._color_space_points.nbytes,
                )
            ],
        )
        self.assertEqual(
            mapper.camera_calls,
            [
                (
                    camera._depth_buffer_pointer,
                    camera._depth_buffer.nbytes,
                    camera._camera_space_points_pointer,
                    camera._camera_space_points.nbytes,
                )
            ],
        )
        np.testing.assert_array_equal(frame.color_bgr, expected_bgra[..., :3])
        np.testing.assert_array_equal(frame.depth_mm, expected_depth)
        np.testing.assert_allclose(frame.depth_to_color_xy, expected_color_points)
        np.testing.assert_allclose(
            frame.depth_to_camera_xyz_mm,
            expected_camera_points_m * 1000.0,
        )
        self.assertEqual(frame.color_bgr.dtype, np.uint8)
        self.assertEqual(frame.depth_mm.dtype, np.uint16)
        self.assertEqual(frame.depth_to_color_xy.dtype, np.float32)
        self.assertEqual(frame.depth_to_camera_xyz_mm.dtype, np.float32)

        # 修改复用缓冲区不能污染已经返回给调用方的这一帧。
        camera._depth_buffer.fill(0)
        camera._color_space_points.fill(0)
        camera._camera_space_points.fill(0)
        np.testing.assert_array_equal(frame.depth_mm, expected_depth)
        np.testing.assert_allclose(frame.depth_to_color_xy, expected_color_points)
        np.testing.assert_allclose(
            frame.depth_to_camera_xyz_mm,
            expected_camera_points_m * 1000.0,
        )
        self.assertEqual(color_frame.dispose_calls, 1)
        self.assertEqual(depth_frame.dispose_calls, 1)
        self.assertEqual(multi_frame.dispose_calls, 1)

    def test_partial_frame_releases_acquired_objects(self) -> None:
        camera = self.make_open_camera()
        color_frame = FakeColorFrame(
            camera._color_buffer,
            np.zeros_like(camera._color_buffer),
        )
        multi_frame = FakeMultiFrame(color_frame, None)
        camera._reader = FakeReader([multi_frame, RuntimeError("stop after partial")])

        with self.assertRaisesRegex(RuntimeError, "stop after partial"):
            camera.read(timeout_seconds=0.1)

        self.assertEqual(color_frame.dispose_calls, 1)
        self.assertEqual(multi_frame.dispose_calls, 1)
        self.assertEqual(color_frame.copy_calls, [])

    def test_copy_failure_releases_all_frames(self) -> None:
        camera = self.make_open_camera()
        color_frame = FakeColorFrame(
            camera._color_buffer,
            np.zeros_like(camera._color_buffer),
            error=ValueError("copy failed"),
        )
        depth_frame = FakeDepthFrame(
            camera._depth_buffer,
            np.zeros_like(camera._depth_buffer),
        )
        multi_frame = FakeMultiFrame(color_frame, depth_frame)
        camera._reader = FakeReader([multi_frame])

        with self.assertRaisesRegex(ValueError, "copy failed"):
            camera.read(timeout_seconds=0.1)

        self.assertEqual(color_frame.dispose_calls, 1)
        self.assertEqual(depth_frame.dispose_calls, 1)
        self.assertEqual(multi_frame.dispose_calls, 1)

    def test_release_clears_numpy_buffers_and_pointers(self) -> None:
        camera = self.make_open_camera()
        reader = FakeReader([])
        sensor = camera._sensor
        camera._reader = reader

        camera.release()

        self.assertEqual(reader.dispose_calls, 1)
        self.assertEqual(sensor.close_calls, 1)
        self.assertFalse(camera.is_open)
        for attribute in (
            "_color_buffer",
            "_depth_buffer",
            "_color_space_points",
            "_camera_space_points",
            "_color_buffer_pointer",
            "_depth_buffer_pointer",
            "_color_space_points_pointer",
            "_camera_space_points_pointer",
        ):
            self.assertIsNone(getattr(camera, attribute), attribute)


if __name__ == "__main__":
    unittest.main()
