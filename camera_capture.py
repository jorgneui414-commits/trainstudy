"""使用 OpenCV 从本地 USB 摄像头采集编号 JPG 图像。"""

from __future__ import annotations

import math
import platform
import time
from pathlib import Path

import cv2


# 采集到的图片统一保存为 0001.jpg、0002.jpg 这样的格式，便于后续标注。
IMAGE_SUFFIX = ".jpg"
IMAGE_NUMBER_WIDTH = 4
PREVIEW_WINDOW_NAME = "摄像头采集"
EXIT_KEYS = {ord("q"), 27}


def _validate_camera_id(camera_id: int) -> None:
    if isinstance(camera_id, bool) or not isinstance(camera_id, int) or camera_id < 0:
        raise ValueError("camera_id 必须是非负整数。")


def _validate_positive_integer(value: int | None, name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} 必须是正整数或 None。")


def _validate_capture_options(
    camera_id: int,
    image_count: int,
    interval_seconds: float,
    frame_width: int | None,
    frame_height: int | None,
) -> None:
    _validate_camera_id(camera_id)
    if isinstance(image_count, bool) or not isinstance(image_count, int) or image_count <= 0:
        raise ValueError("image_count 必须是正整数。")
    if isinstance(interval_seconds, bool) or not isinstance(interval_seconds, (int, float)):
        raise TypeError("interval_seconds 必须是数字。")
    if not math.isfinite(float(interval_seconds)) or interval_seconds < 0:
        raise ValueError("interval_seconds 必须是大于等于 0 的有限数字。")
    _validate_positive_integer(frame_width, "frame_width")
    _validate_positive_integer(frame_height, "frame_height")


def _next_image_index(output_dir: Path) -> int:
    """返回所有以纯数字为文件名的文件中，下一个可用的编号。

    例如 ``0001.txt`` 会像 ``0001.jpg`` 一样占用编号 1，
    因此图像采集不会覆盖将来可能出现的 YOLO 标注文件。
    """

    indices = [
        int(path.stem)
        for path in output_dir.iterdir()
        if path.is_file() and path.stem.isdigit()
    ]
    return max(indices, default=0) + 1


def _is_index_reserved(output_dir: Path, index: int) -> bool:
    return any(
        path.is_file() and path.stem.isdigit() and int(path.stem) == index
        for path in output_dir.iterdir()
    )


def open_camera(camera_id: int):
    """按当前系统打开本地摄像头，Windows 优先使用 DirectShow 后端。"""

    _validate_camera_id(camera_id)
    try:
        # DirectShow 对 Windows 下的普通 USB 摄像头通常更稳定；其他系统使用默认后端。
        if platform.system() == "Windows":
            return cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        return cv2.VideoCapture(camera_id)
    except Exception as exc:
        raise RuntimeError(f"无法创建摄像头 {camera_id}：{exc}") from exc


def capture_images(
    camera_id: int,
    output_dir: str | Path,
    image_count: int,
    interval_seconds: float,
    *,
    frame_width: int | None = None,
    frame_height: int | None = None,
    show_preview: bool = True,
) -> list[Path]:
    """从本地 USB 摄像头采集编号 JPG 图像。

    首帧有效画面会立即保存。后续图像按 ``interval_seconds`` 设定的
    间隔保存。在预览窗口中按 ``q`` 或 ``Esc`` 可提前停止采集。
    """

    _validate_capture_options(camera_id, image_count, interval_seconds, frame_width, frame_height)

    # 输出目录不存在时自动创建；已有数字编号会被保留，不会覆盖旧图片或标签。
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    next_index = _next_image_index(destination)
    saved_images: list[Path] = []
    last_saved_at: float | None = None
    camera = open_camera(camera_id)

    try:
        if not camera.isOpened():
            raise RuntimeError(f"无法打开摄像头 {camera_id}。")

        if frame_width is not None:
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, float(frame_width))
        if frame_height is not None:
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, float(frame_height))

        while len(saved_images) < image_count:
            # 每次先读取最新画面，预览窗口才会保持实时刷新。
            ok, image_bgr = camera.read()
            if not ok or image_bgr is None:
                raise RuntimeError(f"从摄像头 {camera_id} 读取帧失败。")

            if show_preview:
                cv2.imshow(PREVIEW_WINDOW_NAME, image_bgr)
                key = cv2.waitKey(1) & 0xFF
                # 用户可以在达到目标数量前主动结束采集。
                if key in EXIT_KEYS:
                    break

            # monotonic 不受系统时间修改影响，适合用来计算采集间隔。
            now = time.monotonic()
            if last_saved_at is not None and now - last_saved_at < interval_seconds:
                continue

            # 防止采集过程中目录又出现同编号文件，保存前再检查一次。
            while _is_index_reserved(destination, next_index):
                next_index += 1

            image_path = destination / f"{next_index:0{IMAGE_NUMBER_WIDTH}d}{IMAGE_SUFFIX}"
            if not cv2.imwrite(str(image_path), image_bgr):
                # 少数写入失败场景会留下半成品文件，尝试清理后再报告错误。
                try:
                    image_path.unlink(missing_ok=True)
                except OSError:
                    pass
                raise RuntimeError(f"OpenCV 保存图像失败：{image_path}")

            saved_images.append(image_path)
            last_saved_at = now
            next_index += 1
    finally:
        # 无论提前退出、读取失败还是保存失败，都要释放摄像头占用。
        camera.release()
        if show_preview:
            try:
                cv2.destroyWindow(PREVIEW_WINDOW_NAME)
            except cv2.error:
                pass

    return saved_images
