"""第十步模拟机器人客户端和本机 HTTP 端到端测试。"""

from __future__ import annotations

import importlib.util
import io
import socket
import sys
import threading
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import vision_http_server
from kinect_obb_detection import KinectFrameError
from vision_http_server import VisionHTTPServer


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "10_robot_client.py"
SPEC = importlib.util.spec_from_file_location("robot_client_10", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
robot_client = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = robot_client
SPEC.loader.exec_module(robot_client)


def make_detection(
    part_type: str = "螺丝刀",
    *,
    class_id: int = 0,
    camera_x_mm: float = 100.0,
) -> dict[str, object]:
    """生成沿用第六步字段名的一个模拟 OBB+深度目标。"""

    return {
        "class_id": class_id,
        "part_type": part_type,
        "confidence": 0.9567,
        "center_x": 500.0,
        "center_y": 300.0,
        "bbox_width": 120.0,
        "bbox_height": 60.0,
        "obb_points": [[440.0, 270.0], [560.0, 270.0], [560.0, 330.0], [440.0, 330.0]],
        "orientation_deg": 12.5,
        "distance_mm": 1000.0,
        "camera_x_mm": camera_x_mm,
        "camera_y_mm": -50.0,
        "camera_z_mm": 1000.0,
        "depth_valid": True,
    }


def make_capture_result(
    detections: list[dict[str, object]],
) -> dict[str, object]:
    """生成现有 capture_and_detect() 的顶层结果。"""

    return {
        "captured_at": "2026-08-03T08:00:00.000Z",
        "image_width": 1920,
        "image_height": 1080,
        "coordinate_frame": "kinect_camera",
        "coordinate_unit": "mm",
        "detections": detections,
    }


def make_http_success(
    command_id: str,
    detections: list[dict[str, object]],
    *,
    mysql_saved: bool = True,
) -> dict[str, object]:
    """生成现有 /vision/command 的成功 JSON。"""

    return {
        "status": "ok",
        "command": "capture_and_detect",
        "command_id": command_id,
        "captured_at": "2026-08-03T08:00:00.000Z",
        "coordinate_frame": "kinect_camera",
        "coordinate_unit": "mm",
        "detection_count": len(detections),
        "detections": detections,
        "mysql_saved": mysql_saved,
        "mysql_request_id": 9001 if mysql_saved else None,
        "mysql_error": None if mysql_saved else "mysql_save_failed",
    }


class RobotClientParsingTests(unittest.TestCase):
    def _run_with_command_response(
        self,
        status: int,
        response: dict[str, object],
    ) -> tuple[int, str, mock.Mock]:
        request_mock = mock.Mock(
            side_effect=[
                (200, {"status": "ok"}),
                (status, response),
            ]
        )
        output = io.StringIO()
        with (
            mock.patch.object(robot_client, "request_json", request_mock),
            mock.patch.object(
                robot_client.uuid,
                "uuid4",
                return_value=SimpleNamespace(hex="fixed_command_id"),
            ),
            mock.patch.object(
                robot_client.time,
                "perf_counter",
                side_effect=[10.0, 10.25],
            ),
            redirect_stdout(output),
        ):
            exit_code = robot_client.run_once("127.0.0.1", 8008)
        return exit_code, output.getvalue(), request_mock

    def test_parses_one_multiple_and_empty_success_responses(self) -> None:
        cases = (
            ("one", [make_detection()], ("类别: 螺丝刀",)),
            (
                "multiple",
                [
                    make_detection(),
                    make_detection("支架", class_id=2, camera_x_mm=200.0),
                ],
                ("类别: 螺丝刀", "类别: 支架"),
            ),
            ("empty", [], ()),
        )

        for label, detections, expected_names in cases:
            with self.subTest(label=label):
                command_id = "robot_fixed_command_id"
                exit_code, output, request_mock = self._run_with_command_response(
                    200,
                    make_http_success(command_id, detections),
                )

                self.assertEqual(exit_code, 0)
                self.assertIn("健康检查 HTTP 状态码: 200", output)
                self.assertIn("HTTP 状态码: 200", output)
                self.assertIn(f"command_id: {command_id}", output)
                self.assertIn(f"detection_count: {len(detections)}", output)
                self.assertIn("mysql_saved: true", output)
                self.assertIn("请求总耗时: 0.250 秒", output)
                for expected_name in expected_names:
                    self.assertIn(expected_name, output)
                if not detections:
                    self.assertNotIn("目标 1:", output)

                self.assertEqual(
                    request_mock.call_args_list,
                    [
                        mock.call("127.0.0.1", 8008, "GET", "/health"),
                        mock.call(
                            "127.0.0.1",
                            8008,
                            "POST",
                            "/vision/command",
                            payload={
                                "command": "capture_and_detect",
                                "command_id": command_id,
                            },
                        ),
                    ],
                )

    def test_displays_400_503_and_500_json_errors(self) -> None:
        for status, message in (
            (400, "请求体不是合法 JSON。"),
            (503, "Kinect 无法打开。"),
            (500, "模型或推理内部错误。"),
        ):
            with self.subTest(status=status):
                exit_code, output, _request_mock = self._run_with_command_response(
                    status,
                    {"status": "error", "error": message},
                )

                self.assertEqual(exit_code, 1)
                self.assertIn(f"HTTP 状态码: {status}", output)
                self.assertIn(f"错误: {message}", output)
                self.assertIn("command_id: robot_fixed_command_id", output)

    def test_mysql_failure_keeps_visual_success_and_detection_output(self) -> None:
        detections = [make_detection("计算模块", class_id=4)]
        exit_code, output, _request_mock = self._run_with_command_response(
            200,
            make_http_success(
                "robot_fixed_command_id",
                detections,
                mysql_saved=False,
            ),
        )

        self.assertEqual(exit_code, 0)
        self.assertIn("类别: 计算模块", output)
        self.assertIn("置信度: 0.9567", output)
        self.assertIn("角度: 12.50 度", output)
        self.assertIn("相机 XYZ (mm): (100.00, -50.00, 1000.00)", output)
        self.assertIn("mysql_saved: false", output)


class RobotClientEndToEndSimulationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.camera = SimpleNamespace(is_open=False)

        def open_camera(*, availability_timeout_seconds: float) -> None:
            self.assertGreater(availability_timeout_seconds, 0.0)
            self.camera.is_open = True

        def release_camera() -> None:
            self.camera.is_open = False

        self.camera.open = mock.Mock(side_effect=open_camera)
        self.camera.release = mock.Mock(side_effect=release_camera)
        self.capture = mock.Mock(return_value=make_capture_result([make_detection()]))
        self.save_patch = mock.patch.object(
            vision_http_server,
            "save_detection_result",
            return_value=9001,
        )
        self.save_detection_result = self.save_patch.start()
        self.server = VisionHTTPServer(
            ("127.0.0.1", 0),
            camera=self.camera,
            model=object(),
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
        self.save_patch.stop()

    def _close_server(self) -> None:
        if self.server_closed:
            return
        self.server.shutdown()
        self.server_thread.join(timeout=2.0)
        self.assertFalse(self.server_thread.is_alive())
        self.server.server_close()
        self.server_closed = True

    def _run_client(self) -> tuple[int, str]:
        host, port = self.server.server_address
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = robot_client.run_once(host, port)
        return exit_code, output.getvalue()

    def test_real_http_round_trip_handles_one_multiple_and_empty_results(self) -> None:
        one = [make_detection()]
        multiple = [
            make_detection(),
            make_detection("支架", class_id=2, camera_x_mm=200.0),
        ]
        empty: list[dict[str, object]] = []
        self.capture.side_effect = [
            make_capture_result(one),
            make_capture_result(multiple),
            make_capture_result(empty),
        ]
        self.save_detection_result.side_effect = [9101, 9102, 9103]

        results = [self._run_client() for _ in range(3)]

        self.assertEqual([item[0] for item in results], [0, 0, 0])
        self.assertIn("detection_count: 1", results[0][1])
        self.assertIn("类别: 螺丝刀", results[0][1])
        self.assertIn("detection_count: 2", results[1][1])
        self.assertIn("类别: 支架", results[1][1])
        self.assertIn("detection_count: 0", results[2][1])
        self.assertNotIn("目标 1:", results[2][1])
        self.assertEqual(self.save_detection_result.call_count, 3)
        self.assertIs(self.save_detection_result.call_args_list[0].args[2], one)
        self.assertIs(self.save_detection_result.call_args_list[1].args[2], multiple)
        self.assertIs(self.save_detection_result.call_args_list[2].args[2], empty)

        saved_command_ids = [
            item.args[0] for item in self.save_detection_result.call_args_list
        ]
        self.assertEqual(len(set(saved_command_ids)), 3)
        for command_id, (_exit_code, output) in zip(saved_command_ids, results):
            self.assertIn(f"command_id: {command_id}", output)

    def test_mysql_failure_then_next_command_recovers(self) -> None:
        self.save_detection_result.side_effect = [RuntimeError("mysql stopped"), 9202]

        first_exit, first_output = self._run_client()
        second_exit, second_output = self._run_client()

        self.assertEqual(first_exit, 0)
        self.assertIn("detection_count: 1", first_output)
        self.assertIn("mysql_saved: false", first_output)
        self.assertEqual(second_exit, 0)
        self.assertIn("detection_count: 1", second_output)
        self.assertIn("mysql_saved: true", second_output)
        self.assertEqual(self.save_detection_result.call_count, 2)

    def test_kinect_503_and_inference_500_are_displayed(self) -> None:
        self.capture.side_effect = [
            KinectFrameError("frame timeout"),
            RuntimeError("inference failed"),
        ]

        first_exit, first_output = self._run_client()
        second_exit, second_output = self._run_client()

        self.assertEqual(first_exit, 1)
        self.assertIn("HTTP 状态码: 503", first_output)
        self.assertEqual(second_exit, 1)
        self.assertIn("HTTP 状态码: 500", second_output)
        self.assertEqual(self.camera.open.call_count, 2)
        self.assertEqual(self.camera.release.call_count, 1)
        self.save_detection_result.assert_not_called()

    def test_twenty_serial_commands_keep_single_camera_owner(self) -> None:
        active_captures = 0
        maximum_active_captures = 0
        capture_thread_ids: list[int] = []

        def guarded_capture(_camera: object, _model: object) -> dict[str, object]:
            nonlocal active_captures, maximum_active_captures
            active_captures += 1
            maximum_active_captures = max(maximum_active_captures, active_captures)
            capture_thread_ids.append(threading.get_ident())
            try:
                return make_capture_result([make_detection()])
            finally:
                active_captures -= 1

        self.capture.side_effect = guarded_capture
        self.save_detection_result.side_effect = lambda *_args: (
            9300 + self.save_detection_result.call_count
        )

        results = [self._run_client() for _ in range(20)]

        self.assertTrue(all(exit_code == 0 for exit_code, _output in results))
        self.assertEqual(self.capture.call_count, 20)
        self.assertEqual(self.camera.open.call_count, 1)
        self.assertEqual(maximum_active_captures, 1)
        self.assertEqual(len(set(capture_thread_ids)), 1)
        saved_command_ids = {
            item.args[0] for item in self.save_detection_result.call_args_list
        }
        self.assertEqual(len(saved_command_ids), 20)

    def test_client_shutdown_releases_camera_and_http_port(self) -> None:
        exit_code, _output = self._run_client()
        self.assertEqual(exit_code, 0)
        address = self.server.server_address

        self._close_server()

        self.assertFalse(self.camera.is_open)
        self.camera.release.assert_called_once_with()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(address)


if __name__ == "__main__":
    unittest.main()
