from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2

from camera_capture import PREVIEW_WINDOW_NAME, capture_images


class FakeVideoCapture:
    def __init__(self, *, opened: bool = True, frames: list[tuple[bool, object]] | None = None) -> None:
        self.opened = opened
        self.frames = frames or [(True, object())]
        self.read_count = 0
        self.released = False
        self.set_calls: list[tuple[int, float]] = []

    def isOpened(self) -> bool:
        return self.opened

    def set(self, property_id: int, value: float) -> bool:
        self.set_calls.append((property_id, value))
        return True

    def read(self) -> tuple[bool, object]:
        frame = self.frames[self.read_count % len(self.frames)]
        self.read_count += 1
        return frame

    def release(self) -> None:
        self.released = True


def write_image(path: str, _image: object) -> bool:
    Path(path).write_bytes(b"test image")
    return True


def write_then_fail(path: str, _image: object) -> bool:
    Path(path).write_bytes(b"partial image")
    return False


class CaptureImagesTests(unittest.TestCase):
    def test_windows_uses_directshow_and_sets_requested_size(self) -> None:
        capture = FakeVideoCapture()
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch("camera_capture.platform.system", return_value="Windows"),
                patch("camera_capture.cv2.VideoCapture", return_value=capture) as video_capture,
                patch("camera_capture.cv2.imwrite", side_effect=write_image),
            ):
                saved = capture_images(0, temp_dir, 1, 0, frame_width=640, frame_height=480, show_preview=False)

        self.assertEqual([path.name for path in saved], ["0001.jpg"])
        video_capture.assert_called_once_with(0, cv2.CAP_DSHOW)
        self.assertEqual(
            capture.set_calls,
            [(cv2.CAP_PROP_FRAME_WIDTH, 640.0), (cv2.CAP_PROP_FRAME_HEIGHT, 480.0)],
        )
        self.assertTrue(capture.released)

    def test_non_windows_uses_opencv_default_backend(self) -> None:
        capture = FakeVideoCapture()
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch("camera_capture.platform.system", return_value="Linux"),
                patch("camera_capture.cv2.VideoCapture", return_value=capture) as video_capture,
                patch("camera_capture.cv2.imwrite", side_effect=write_image),
            ):
                capture_images(0, temp_dir, 1, 0, show_preview=False)

        video_capture.assert_called_once_with(0)

    def test_numeric_images_and_labels_reserve_image_numbers(self) -> None:
        capture = FakeVideoCapture()
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            existing_image = output_dir / "0006.jpg"
            existing_image.write_bytes(b"do not overwrite")
            (output_dir / "0007.txt").write_text("label", encoding="utf-8")
            (output_dir / "8.json").write_text("metadata", encoding="utf-8")
            (output_dir / "not-a-number.txt").write_text("ignored", encoding="utf-8")
            with (
                patch("camera_capture.cv2.VideoCapture", return_value=capture),
                patch("camera_capture.cv2.imwrite", side_effect=write_image),
            ):
                saved = capture_images(0, output_dir, 2, 0, show_preview=False)

            self.assertEqual(existing_image.read_bytes(), b"do not overwrite")

        self.assertEqual([path.name for path in saved], ["0009.jpg", "0010.jpg"])

    def test_first_image_is_immediate_and_later_images_wait_for_interval(self) -> None:
        capture = FakeVideoCapture()
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch("camera_capture.cv2.VideoCapture", return_value=capture),
                patch("camera_capture.cv2.imwrite", side_effect=write_image),
                patch("camera_capture.time.monotonic", side_effect=[10.0, 10.2, 11.0]),
            ):
                saved = capture_images(0, temp_dir, 2, 1.0, show_preview=False)

        self.assertEqual([path.name for path in saved], ["0001.jpg", "0002.jpg"])
        self.assertEqual(capture.read_count, 3)

    def test_q_or_escape_stops_preview_before_saving(self) -> None:
        for key in (ord("q"), 27):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temp_dir:
                capture = FakeVideoCapture()
                with (
                    patch("camera_capture.cv2.VideoCapture", return_value=capture),
                    patch("camera_capture.cv2.imshow") as imshow,
                    patch("camera_capture.cv2.waitKey", return_value=key),
                    patch("camera_capture.cv2.destroyWindow") as destroy_window,
                    patch("camera_capture.cv2.imwrite", side_effect=write_image),
                ):
                    saved = capture_images(0, temp_dir, 1, 0, show_preview=True)

                self.assertEqual(saved, [])
                imshow.assert_called_once_with(PREVIEW_WINDOW_NAME, unittest.mock.ANY)
                destroy_window.assert_called_once_with(PREVIEW_WINDOW_NAME)
                self.assertTrue(capture.released)

    def test_invalid_camera_id_is_rejected_before_opening(self) -> None:
        for camera_id in (-1, True, "0"):
            with self.subTest(camera_id=camera_id), tempfile.TemporaryDirectory() as temp_dir:
                with patch("camera_capture.cv2.VideoCapture") as video_capture:
                    with self.assertRaises(ValueError):
                        capture_images(camera_id, temp_dir, 1, 0, show_preview=False)  # type: ignore[arg-type]

                video_capture.assert_not_called()

    def test_invalid_capture_options_are_rejected_before_opening(self) -> None:
        invalid_options = (
            {"image_count": 0},
            {"interval_seconds": -0.1},
            {"interval_seconds": float("inf")},
            {"frame_width": 0},
            {"frame_height": "480"},
        )
        for options in invalid_options:
            with self.subTest(options=options), tempfile.TemporaryDirectory() as temp_dir:
                with patch("camera_capture.cv2.VideoCapture") as video_capture:
                    with self.assertRaises((TypeError, ValueError)):
                        capture_images(0, temp_dir, 1, 0, show_preview=False, **options)

                video_capture.assert_not_called()

    def test_camera_open_failure_releases_device(self) -> None:
        capture = FakeVideoCapture(opened=False)
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("camera_capture.cv2.VideoCapture", return_value=capture):
                with self.assertRaisesRegex(RuntimeError, "无法打开摄像头 0"):
                    capture_images(0, temp_dir, 1, 0, show_preview=False)

        self.assertTrue(capture.released)

    def test_frame_read_failure_releases_device(self) -> None:
        capture = FakeVideoCapture(frames=[(False, None)])
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("camera_capture.cv2.VideoCapture", return_value=capture):
                with self.assertRaisesRegex(RuntimeError, "从摄像头 0 读取帧失败"):
                    capture_images(0, temp_dir, 1, 0, show_preview=False)

        self.assertTrue(capture.released)

    def test_image_write_failure_releases_device_and_does_not_create_output(self) -> None:
        capture = FakeVideoCapture()
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch("camera_capture.cv2.VideoCapture", return_value=capture),
                patch("camera_capture.cv2.imwrite", side_effect=write_then_fail),
            ):
                with self.assertRaisesRegex(RuntimeError, "OpenCV 保存图像失败"):
                    capture_images(0, temp_dir, 1, 0, show_preview=False)

            self.assertEqual(list(Path(temp_dir).iterdir()), [])

        self.assertTrue(capture.released)

    def test_preview_window_is_destroyed_when_a_read_fails(self) -> None:
        capture = FakeVideoCapture(frames=[(False, None)])
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch("camera_capture.cv2.VideoCapture", return_value=capture),
                patch("camera_capture.cv2.destroyWindow") as destroy_window,
            ):
                with self.assertRaises(RuntimeError):
                    capture_images(0, temp_dir, 1, 0, show_preview=True)

        destroy_window.assert_called_once_with(PREVIEW_WINDOW_NAME)
        self.assertTrue(capture.released)


if __name__ == "__main__":
    unittest.main()
