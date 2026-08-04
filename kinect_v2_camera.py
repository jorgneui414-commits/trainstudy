"""Microsoft Kinect v2 彩色帧、深度帧和坐标映射读取接口。

本模块只封装 Kinect for Windows SDK 2.0，不负责 YOLO、窗口显示或网络通信。
所有方法都应由同一个线程调用，避免多个线程同时访问 Kinect SDK 对象。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class KinectFrame:
    """一组由 MultiSourceFrame 同步取得的 Kinect v2 RGB-D 数据。

    深度和相机三维坐标均使用毫米；彩色映射坐标使用像素。
    """

    # OpenCV 使用的彩色图，形状为 (彩色高度, 彩色宽度, 3)，通道顺序是 B、G、R。
    color_bgr: np.ndarray
    # 每个深度像素测得的距离，形状为 (深度高度, 深度宽度)，单位是毫米。
    depth_mm: np.ndarray
    # 每个深度像素在彩色图中的对应位置，最后一维依次保存彩色图的 x、y 像素坐标。
    depth_to_color_xy: np.ndarray
    # 每个深度像素在 Kinect 相机坐标系中的 X、Y、Z，最后一维长度为 3，单位是毫米。
    depth_to_camera_xyz_mm: np.ndarray


class KinectV2Camera:
    """通过官方 Microsoft.Kinect.dll 读取 Kinect v2。"""

    # Kinect v2 的标准彩色分辨率和深度分辨率；open() 后还会从真实设备重新读取一次。
    COLOR_WIDTH = 1920
    COLOR_HEIGHT = 1080
    DEPTH_WIDTH = 512
    DEPTH_HEIGHT = 424

    def __init__(self, sdk_assembly_path: Path | str) -> None:
        self.sdk_assembly_path = Path(sdk_assembly_path)

        # 下面三个变量保存 SDK 设备对象。类型写成 Any，是因为它们运行时才由 .NET 创建，
        # Pylance 无法像分析普通 Python 类那样提前识别具体类型。
        self._sensor: Any | None = None
        self._reader: Any | None = None
        self._coordinate_mapper: Any | None = None

        # SDK 通过 IntPtr 直接写入这些连续 NumPy 缓冲区。每帧复用同一批数组，避免
        # 先写 .NET 数组、固定内存、再复制到 NumPy 的中转过程。
        self._color_buffer: np.ndarray | None = None
        self._depth_buffer: np.ndarray | None = None
        self._color_space_points: np.ndarray | None = None
        self._camera_space_points: np.ndarray | None = None
        self._color_buffer_pointer: Any | None = None
        self._depth_buffer_pointer: Any | None = None
        self._color_space_points_pointer: Any | None = None
        self._camera_space_points_pointer: Any | None = None
        self._color_image_format: Any | None = None

    @property
    def is_open(self) -> bool:
        """传感器和同步帧读取器都已创建时，才认为相机已经打开。"""

        return self._sensor is not None and self._reader is not None

    def open(self, *, availability_timeout_seconds: float = 5.0) -> None:
        """加载 SDK、打开默认 Kinect，并等待设备进入可用状态。"""

        if self.is_open:
            return
        if availability_timeout_seconds <= 0:
            raise ValueError("availability_timeout_seconds 必须大于 0。")
        if not self.sdk_assembly_path.is_file():
            raise FileNotFoundError(
                f"找不到 Kinect SDK 程序集：{self.sdk_assembly_path}。"
                "请安装 Kinect for Windows SDK 2.0，或修改 config.KINECT_SDK_ASSEMBLY_PATH。"
            )

        try:
            import pythonnet

            # Kinect SDK 是 .NET 程序集，不是普通 Python 包：先启动 .NET Framework，
            # 再通过 clr.AddReference() 把 Microsoft.Kinect.dll 加入当前 Python 进程。
            pythonnet.load("netfx")
            import clr

            clr.AddReference(str(self.sdk_assembly_path))

            # 这些命名空间由 pythonnet 在运行时动态提供。程序可以正常导入，但 Pylance
            # 无法静态找到对应的 .py 文件，所以只在这两处忽略 reportMissingImports。
            from System import IntPtr  # pyright: ignore[reportMissingImports]
            from Microsoft.Kinect import (  # pyright: ignore[reportMissingImports]
                ColorImageFormat,
                FrameSourceTypes,
                KinectSensor,
            )
        except (ImportError, RuntimeError, OSError) as exc:
            raise RuntimeError(
                "无法加载 Kinect v2 Python/SDK 依赖。请在当前环境安装 requirements.txt，"
                "并确认 Kinect for Windows SDK 2.0 已安装。"
            ) from exc

        sensor = KinectSensor.GetDefault()
        if sensor is None:
            raise RuntimeError("Kinect SDK 未找到默认 Kinect v2 设备。")

        try:
            # Open() 发出启动命令；IsAvailable=True 才代表设备已经可以真正提供数据。
            sensor.Open()
            deadline = time.monotonic() + float(availability_timeout_seconds)
            while not bool(sensor.IsAvailable) and time.monotonic() < deadline:
                time.sleep(0.05)
            if not bool(sensor.IsAvailable):
                raise RuntimeError(
                    "Kinect v2 已被 SDK 枚举，但当前不可用。请在 Kinect Studio 左上角断开设备并关闭软件，"
                    "确认没有其他程序占用相机后重试。"
                )

            # 按位或把 Color 和 Depth 两种数据源组合起来，要求 SDK 返回同一时刻的 RGB-D 帧。
            frame_types = FrameSourceTypes.Color | FrameSourceTypes.Depth
            reader = sensor.OpenMultiSourceFrameReader(frame_types)
            if reader is None:
                raise RuntimeError("Kinect v2 无法创建彩色+深度同步帧读取器。")

            color_description = sensor.ColorFrameSource.FrameDescription
            depth_description = sensor.DepthFrameSource.FrameDescription
            self.COLOR_WIDTH = int(color_description.Width)
            self.COLOR_HEIGHT = int(color_description.Height)
            self.DEPTH_WIDTH = int(depth_description.Width)
            self.DEPTH_HEIGHT = int(depth_description.Height)

            # NumPy 默认创建 C 连续数组。SDK 的 IntPtr 重载会按字节直接填满这些缓冲区；
            # 两种映射点在官方程序集中分别是连续的 2 个和 3 个 float32。
            self._color_buffer = np.empty(
                (self.COLOR_HEIGHT, self.COLOR_WIDTH, 4),
                dtype=np.uint8,
            )
            self._depth_buffer = np.empty(
                (self.DEPTH_HEIGHT, self.DEPTH_WIDTH),
                dtype=np.uint16,
            )
            self._color_space_points = np.empty(
                (self.DEPTH_HEIGHT, self.DEPTH_WIDTH, 2),
                dtype=np.float32,
            )
            self._camera_space_points = np.empty(
                (self.DEPTH_HEIGHT, self.DEPTH_WIDTH, 3),
                dtype=np.float32,
            )
            self._color_buffer_pointer = IntPtr(int(self._color_buffer.ctypes.data))
            self._depth_buffer_pointer = IntPtr(int(self._depth_buffer.ctypes.data))
            self._color_space_points_pointer = IntPtr(
                int(self._color_space_points.ctypes.data)
            )
            self._camera_space_points_pointer = IntPtr(
                int(self._camera_space_points.ctypes.data)
            )
            self._color_image_format = ColorImageFormat.Bgra
            self._coordinate_mapper = sensor.CoordinateMapper
            self._reader = reader
            self._sensor = sensor
        except Exception:
            try:
                sensor.Close()
            except Exception:
                pass
            self._clear_state()
            raise

    def read(self, *, timeout_seconds: float = 2.0) -> KinectFrame:
        """等待并返回一组新的同步彩色/深度帧及 SDK 坐标映射。"""

        if not self.is_open:
            raise RuntimeError("Kinect v2 尚未打开，请先调用 open()。")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds 必须大于 0。")

        # deadline 是本次 read() 最晚结束的时间，避免设备断流后程序永远卡在循环中。
        deadline = time.monotonic() + float(timeout_seconds)
        while time.monotonic() < deadline:
            # AcquireLatestFrame() 只取最新一组组合帧；当前还没有新帧时会返回 None。
            multi_frame = self._reader.AcquireLatestFrame()
            if multi_frame is None:
                time.sleep(0.001)
                continue

            color_frame = None
            depth_frame = None
            try:
                # MultiSourceFrame 是外层容器，还要分别取得其中的彩色子帧和深度子帧。
                color_frame = multi_frame.ColorFrameReference.AcquireFrame()
                depth_frame = multi_frame.DepthFrameReference.AcquireFrame()
                if color_frame is None or depth_frame is None:
                    continue

                # 使用 SDK 官方 IntPtr 重载，把彩色和深度数据直接写入 NumPy 缓冲区。
                color_frame.CopyConvertedFrameDataToIntPtr(
                    self._color_buffer_pointer,
                    int(self._color_buffer.nbytes),
                    self._color_image_format,
                )
                depth_frame.CopyFrameDataToIntPtr(
                    self._depth_buffer_pointer,
                    int(self._depth_buffer.nbytes),
                )

                # 同一深度像素可以被映射成两种结果：
                # 1. 它位于彩色图的哪个像素；2. 它在相机坐标系中的三维位置。
                self._coordinate_mapper.MapDepthFrameToColorSpaceUsingIntPtr(
                    self._depth_buffer_pointer,
                    int(self._depth_buffer.nbytes),
                    self._color_space_points_pointer,
                    int(self._color_space_points.nbytes),
                )
                self._coordinate_mapper.MapDepthFrameToCameraSpaceUsingIntPtr(
                    self._depth_buffer_pointer,
                    int(self._depth_buffer.nbytes),
                    self._camera_space_points_pointer,
                    int(self._camera_space_points.nbytes),
                )

                # 内部缓冲区会被下一次 read() 覆盖，因此返回独立数组。cvtColor 和毫米
                # 转换本身会创建新数组；深度及彩色映射显式 copy()。
                return KinectFrame(
                    color_bgr=cv2.cvtColor(self._color_buffer, cv2.COLOR_BGRA2BGR),
                    depth_mm=self._depth_buffer.copy(),
                    depth_to_color_xy=self._color_space_points.copy(),
                    depth_to_camera_xyz_mm=_meters_to_millimeters(
                        self._camera_space_points
                    ),
                )
            finally:
                # .NET 帧持有相机资源，成功、跳过或异常时都必须释放。
                _dispose(depth_frame)
                _dispose(color_frame)
                _dispose(multi_frame)

        raise RuntimeError(
            f"在 {timeout_seconds:.1f} 秒内没有取得 Kinect v2 的同步彩色+深度帧。"
            "请检查相机是否被 Kinect Studio 或其他程序占用。"
        )

    def release(self) -> None:
        """释放同步帧读取器和 Kinect 设备。"""

        # 先停止读取器，再关闭传感器，最后清空 Python 保存的对象引用。
        _dispose(self._reader)
        if self._sensor is not None:
            try:
                self._sensor.Close()
            except Exception:
                pass
        self._clear_state()

    def _clear_state(self) -> None:
        """清空所有运行时对象，使 is_open 恢复为 False。"""

        self._reader = None
        self._sensor = None
        self._coordinate_mapper = None
        self._color_buffer = None
        self._depth_buffer = None
        self._color_space_points = None
        self._camera_space_points = None
        self._color_buffer_pointer = None
        self._depth_buffer_pointer = None
        self._color_space_points_pointer = None
        self._camera_space_points_pointer = None
        self._color_image_format = None

    def __enter__(self) -> "KinectV2Camera":
        # 支持 with KinectV2Camera(...) as camera: 这种自动释放的写法。
        self.open()
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.release()


def _meters_to_millimeters(camera_xyz_m: np.ndarray) -> np.ndarray:
    """把 Kinect SDK 的相机坐标从米转换为独立的毫米数组。"""

    # 每个坐标乘以 1000，例如 1.2 米会变成 1200 毫米。
    return np.asarray(camera_xyz_m, dtype=np.float32) * 1000.0


def _dispose(value: Any | None) -> None:
    """安全释放一个可能为空的 .NET 对象。"""

    if value is None:
        return
    try:
        value.Dispose()
    except Exception:
        pass
