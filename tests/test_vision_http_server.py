"""第七步同步 HTTP/JSON 服务的假相机、假视觉函数测试。"""

from __future__ import annotations

import json
import socket
import threading
import unittest
from http.client import HTTPConnection
from http.server import HTTPServer, ThreadingHTTPServer
from unittest import mock

import vision_http_server
from kinect_obb_detection import KinectFrameError
from vision_http_server import VisionHTTPServer


class FakeCamera:
    """只实现 HTTP 服务管理生命周期时会访问的相机接口。"""

    def __init__(self) -> None:
        self.is_open = False
        self.open_calls = 0
        self.release_calls = 0
        self.open_errors: list[Exception | None] = []

    def open(self, *, availability_timeout_seconds: float) -> None:
        self.open_calls += 1
        if self.open_errors:
            error = self.open_errors.pop(0)
            if error is not None:
                raise error
        self.is_open = True

    def release(self) -> None:
        self.release_calls += 1
        self.is_open = False


def make_detection_result(
    detections: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "captured_at": "2026-08-03T08:00:00.000Z",
        "image_width": 1920,
        "image_height": 1080,
        "coordinate_frame": "kinect_camera",
        "coordinate_unit": "mm",
        "detections": detections if detections is not None else [],
    }


class VisionHTTPServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.camera = FakeCamera()
        self.model = object()
        self.capture = mock.Mock(return_value=make_detection_result())
        self.save_patch = mock.patch.object(
            vision_http_server,
            "save_detection_result",
            return_value=9001,
        )
        self.save_detection_result = self.save_patch.start()
        self.addCleanup(self.save_patch.stop)
        self.server = VisionHTTPServer(
            ("127.0.0.1", 0),
            camera=self.camera,
            model=self.model,
            capture_function=self.capture,
        )
        self.server_thread = threading.Thread(
            target=self.server.serve_forever,
            daemon=True,
        )
        self.server_thread.start()
        self.server_closed = False

    def tearDown(self) -> None:
        self._close_server()

    def _close_server(self) -> None:
        if self.server_closed:
            return
        self.server.shutdown()
        self.server_thread.join(timeout=2.0)
        self.assertFalse(self.server_thread.is_alive())
        self.server.server_close()
        self.server_closed = True

    def _request(
        self,
        method: str,
        path: str,
        body: bytes | None = None,
    ) -> tuple[int, str, bytes, dict[str, object]]:
        host, port = self.server.server_address
        connection = HTTPConnection(host, port, timeout=2.0)
        headers = {"Content-Type": "application/json; charset=utf-8"}
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        raw_body = response.read()
        content_type = response.getheader("Content-Type", "")
        connection.close()
        return (
            response.status,
            content_type,
            raw_body,
            json.loads(raw_body.decode("utf-8")),
        )

    def _post_command(
        self,
        payload: dict[str, object],
        *,
        path: str = "/vision/command",
    ) -> tuple[int, str, bytes, dict[str, object]]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        return self._request("POST", path, body)

    def test_uses_standard_single_threaded_http_server(self) -> None:
        self.assertIsInstance(self.server, HTTPServer)
        self.assertNotIsInstance(self.server, ThreadingHTTPServer)

    def test_health_returns_json_without_opening_camera(self) -> None:
        status, content_type, _raw_body, payload = self._request("GET", "/health")

        self.assertEqual(status, 200)
        self.assertEqual(payload, {"status": "ok"})
        self.assertEqual(content_type, "application/json; charset=utf-8")
        self.assertEqual(self.camera.open_calls, 0)
        self.capture.assert_not_called()

    def test_valid_command_calls_capture_once_and_preserves_chinese_and_id(self) -> None:
        detections = [{"part_type": "螺丝刀", "camera_z_mm": 1000.0}]
        result = make_detection_result(detections)
        call_order: list[str] = []
        self.capture.side_effect = lambda *_args: (
            call_order.append("capture") or result
        )
        self.save_detection_result.side_effect = lambda *_args: (
            call_order.append("save") or 9001
        )
        command_id = " pick_0001 "

        status, _content_type, raw_body, payload = self._post_command(
            {"command": "capture_and_detect", "command_id": command_id}
        )

        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["command"], "capture_and_detect")
        self.assertEqual(payload["command_id"], command_id)
        self.assertEqual(payload["captured_at"], "2026-08-03T08:00:00.000Z")
        self.assertEqual(payload["coordinate_frame"], "kinect_camera")
        self.assertEqual(payload["coordinate_unit"], "mm")
        self.assertEqual(payload["detection_count"], 1)
        self.assertEqual(payload["detections"][0]["part_type"], "螺丝刀")
        self.assertTrue(payload["mysql_saved"])
        self.assertEqual(payload["mysql_request_id"], 9001)
        self.assertIsNone(payload["mysql_error"])
        self.assertIn("螺丝刀".encode("utf-8"), raw_body)
        self.assertEqual(self.camera.open_calls, 1)
        self.capture.assert_called_once_with(self.camera, self.model)
        self.save_detection_result.assert_called_once_with(
            command_id,
            result["captured_at"],
            detections,
        )
        self.assertIs(self.save_detection_result.call_args.args[2], detections)
        self.assertEqual(payload["detections"], detections)
        self.assertEqual(call_order, ["capture", "save"])

    def test_zero_detections_returns_200(self) -> None:
        status, _content_type, _raw_body, payload = self._post_command(
            {"command": "capture_and_detect", "command_id": "pick_empty"}
        )

        self.assertEqual(status, 200)
        self.assertEqual(payload["detection_count"], 0)
        self.assertEqual(payload["detections"], [])
        self.assertTrue(payload["mysql_saved"])
        self.assertEqual(payload["mysql_request_id"], 9001)
        self.assertIsNone(payload["mysql_error"])
        self.save_detection_result.assert_called_once_with(
            "pick_empty",
            "2026-08-03T08:00:00.000Z",
            self.capture.return_value["detections"],
        )

    def test_database_failure_keeps_200_and_does_not_expose_password(self) -> None:
        detections = [{"part_type": "支架", "camera_z_mm": 1200.0}]
        self.capture.return_value = make_detection_result(detections)
        self.save_detection_result.side_effect = RuntimeError(
            "mysql://vision_user:super-secret-password@127.0.0.1/vision_study"
        )

        status, _content_type, raw_body, payload = self._post_command(
            {"command": "capture_and_detect", "command_id": "pick_db_failed"}
        )

        decoded_body = raw_body.decode("utf-8")
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["detections"], detections)
        self.assertFalse(payload["mysql_saved"])
        self.assertIsNone(payload["mysql_request_id"])
        self.assertEqual(payload["mysql_error"], "mysql_save_failed")
        self.assertNotIn("super-secret-password", decoded_body)
        self.assertNotIn("mysql://", decoded_body)
        self.save_detection_result.assert_called_once_with(
            "pick_db_failed",
            "2026-08-03T08:00:00.000Z",
            detections,
        )

    def test_invalid_requests_return_400_without_opening_camera(self) -> None:
        cases = (
            (b"{invalid", "invalid_json"),
            (json.dumps({"command_id": "pick_1"}).encode(), "missing_command"),
            (
                json.dumps({"command": "capture_and_detect"}).encode(),
                "missing_command_id",
            ),
            (
                json.dumps(
                    {"command": "capture_and_detect", "command_id": "   "}
                ).encode(),
                "empty_command_id",
            ),
            (
                json.dumps(
                    {"command": "detect_saved_image", "command_id": "pick_1"}
                ).encode(),
                "unsupported_command",
            ),
        )

        for body, label in cases:
            with self.subTest(label=label):
                status, _content_type, _raw_body, payload = self._request(
                    "POST",
                    "/vision/command",
                    body,
                )
                self.assertEqual(status, 400)
                self.assertEqual(payload["status"], "error")

        self.assertEqual(self.camera.open_calls, 0)
        self.capture.assert_not_called()

    def test_unknown_get_and_post_paths_return_404(self) -> None:
        get_status, _content_type, _raw_body, get_payload = self._request(
            "GET",
            "/missing",
        )
        post_status, _content_type, _raw_body, post_payload = self._post_command(
            {"command": "capture_and_detect", "command_id": "pick_1"},
            path="/missing",
        )

        self.assertEqual(get_status, 404)
        self.assertEqual(post_status, 404)
        self.assertEqual(get_payload["status"], "error")
        self.assertEqual(post_payload["status"], "error")
        self.capture.assert_not_called()

    def test_camera_open_failure_returns_503_and_next_command_retries(self) -> None:
        self.camera.open_errors = [RuntimeError("device busy"), None]
        command = {"command": "capture_and_detect", "command_id": "pick_retry"}

        first_status, _content_type, _raw_body, first_payload = self._post_command(
            command
        )
        self.save_detection_result.assert_not_called()
        second_status, _content_type, _raw_body, _second_payload = self._post_command(
            command
        )

        self.assertEqual(first_status, 503)
        self.assertEqual(first_payload["command_id"], "pick_retry")
        self.assertEqual(second_status, 200)
        self.assertEqual(self.camera.open_calls, 2)
        self.assertEqual(self.camera.release_calls, 1)
        self.capture.assert_called_once_with(self.camera, self.model)
        self.save_detection_result.assert_called_once()

    def test_frame_failure_releases_camera_and_next_command_reopens(self) -> None:
        self.capture.side_effect = [
            KinectFrameError("frame timeout"),
            make_detection_result(),
        ]
        command = {"command": "capture_and_detect", "command_id": "pick_retry"}

        first_status, _content_type, _raw_body, first_payload = self._post_command(
            command
        )
        self.save_detection_result.assert_not_called()
        second_status, _content_type, _raw_body, _second_payload = self._post_command(
            command
        )

        self.assertEqual(first_status, 503)
        self.assertEqual(first_payload["command_id"], "pick_retry")
        self.assertEqual(second_status, 200)
        self.assertEqual(self.camera.open_calls, 2)
        self.assertEqual(self.camera.release_calls, 1)
        self.assertEqual(self.capture.call_count, 2)
        self.save_detection_result.assert_called_once()

    def test_model_or_inference_failure_returns_500(self) -> None:
        self.capture.side_effect = RuntimeError("inference failed")

        status, _content_type, _raw_body, payload = self._post_command(
            {"command": "capture_and_detect", "command_id": "pick_failed"}
        )

        self.assertEqual(status, 500)
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["command_id"], "pick_failed")
        self.assertTrue(self.camera.is_open)
        self.assertEqual(self.camera.release_calls, 0)
        self.save_detection_result.assert_not_called()

    def test_close_releases_camera_and_http_port(self) -> None:
        status, _content_type, _raw_body, _payload = self._post_command(
            {"command": "capture_and_detect", "command_id": "pick_close"}
        )
        self.assertEqual(status, 200)
        address = self.server.server_address

        self._close_server()

        self.assertFalse(self.camera.is_open)
        self.assertEqual(self.camera.release_calls, 1)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(address)


if __name__ == "__main__":
    unittest.main()
