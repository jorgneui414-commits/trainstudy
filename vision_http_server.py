"""第七步：把一次 Kinect OBB 检测暴露为同步 HTTP/JSON 接口。

服务直接复用 ``capture_and_detect(camera, model)``。标准库 ``HTTPServer`` 会在
``serve_forever()`` 所在线程中串行处理请求，因此不会并发访问 Kinect。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import urlsplit

import config
from kinect_obb_detection import KinectFrameError, capture_and_detect
from mysql_detection_store import save_detection_result


SUPPORTED_COMMAND = "capture_and_detect"
CaptureFunction = Callable[[Any, Any], dict[str, object]]


class _KinectUnavailableError(RuntimeError):
    """本次请求无法打开 Kinect，或无法从 Kinect 取得一帧。"""


class VisionHTTPServer(HTTPServer):
    """持有启动时加载的模型，并串行执行机器人视觉命令。"""

    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        *,
        camera: Any,
        model: Any,
        capture_function: CaptureFunction | None = None,
    ) -> None:
        self.camera = camera
        self.model = model
        self.capture_function = (
            capture_and_detect if capture_function is None else capture_function
        )
        self._camera_needs_release = bool(getattr(camera, "is_open", False))
        super().__init__(server_address, _VisionRequestHandler)

    def capture_once(self) -> dict[str, object]:
        """按需打开 Kinect，并复用第六步接口执行恰好一次检测。"""

        self._ensure_camera_open()
        try:
            result = self.capture_function(self.camera, self.model)
        except KinectFrameError as exc:
            # 断流后立即释放；下一条合法命令会再次进入 _ensure_camera_open()。
            self._release_camera()
            raise _KinectUnavailableError(str(exc)) from exc

        if not isinstance(result, dict):
            raise TypeError("capture_and_detect() 必须返回字典。")
        return result

    def _ensure_camera_open(self) -> None:
        """仅在首条合法命令或上次断流后的下一条合法命令中打开设备。"""

        if bool(getattr(self.camera, "is_open", False)):
            self._camera_needs_release = True
            return

        # 即使 open() 只完成了部分初始化，异常路径也会调用 release() 清理。
        self._camera_needs_release = True
        try:
            self.camera.open(
                availability_timeout_seconds=(
                    config.KINECT_AVAILABILITY_TIMEOUT_SECONDS
                )
            )
            if not bool(getattr(self.camera, "is_open", False)):
                raise RuntimeError("Kinect open() 返回后设备仍未处于打开状态。")
        except Exception as exc:
            self._release_camera()
            raise _KinectUnavailableError(f"Kinect 打开失败：{exc}") from exc

    def _release_camera(self) -> None:
        """释放当前可能持有的 Kinect 资源；重复调用不会重复释放。"""

        if not self._camera_needs_release:
            return
        try:
            self.camera.release()
        except Exception:
            # 关闭 HTTP 端口不能被相机释放异常阻断。
            pass
        finally:
            self._camera_needs_release = False

    def server_close(self) -> None:
        """关闭监听端口前释放 Kinect。"""

        try:
            self._release_camera()
        finally:
            super().server_close()


class _VisionRequestHandler(BaseHTTPRequestHandler):
    """只实现第七步规定的两个 HTTP 路由。"""

    server: VisionHTTPServer

    def do_GET(self) -> None:  # noqa: N802 - 标准库要求使用该方法名
        if self._request_path() != "/health":
            self._send_error_json(HTTPStatus.NOT_FOUND, "未知路径。")
            return
        self._send_json(HTTPStatus.OK, {"status": "ok"})

    def do_POST(self) -> None:  # noqa: N802 - 标准库要求使用该方法名
        if self._request_path() != "/vision/command":
            self._send_error_json(HTTPStatus.NOT_FOUND, "未知路径。")
            return

        try:
            payload = self._read_json_object()
            command, command_id = self._validate_command(payload)
        except ValueError as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
            return

        try:
            result = self.server.capture_once()
        except _KinectUnavailableError:
            self._send_error_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "Kinect 无法打开或本次取帧失败。",
                command=command,
                command_id=command_id,
            )
            return
        except Exception as exc:
            self.log_error("模型或推理内部错误：%s", exc)
            self._send_error_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "模型或推理内部错误。",
                command=command,
                command_id=command_id,
            )
            return

        try:
            response = self._build_success_response(result, command, command_id)
        except Exception as exc:
            self.log_error("检测结果无法编码为约定 JSON：%s", exc)
            self._send_error_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "模型或推理内部错误。",
                command=command,
                command_id=command_id,
            )
            return

        # 第九步直接保存准备返回机器人的同一组字段；数据库失败不抹掉视觉结果。
        try:
            mysql_request_id = save_detection_result(
                command_id,
                response["captured_at"],
                response["detections"],
            )
        except Exception as exc:
            # 只记录异常类型并返回固定标识，避免密码或连接字符串进入 HTTP 响应。
            self.log_error("MySQL 检测结果写入失败（%s）。", type(exc).__name__)
            response["mysql_saved"] = False
            response["mysql_request_id"] = None
            response["mysql_error"] = "mysql_save_failed"
        else:
            response["mysql_saved"] = True
            response["mysql_request_id"] = mysql_request_id
            response["mysql_error"] = None

        try:
            self._send_json(HTTPStatus.OK, response)
        except Exception as exc:
            self.log_error("检测结果无法编码为约定 JSON：%s", exc)
            self._send_error_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "模型或推理内部错误。",
                command=command,
                command_id=command_id,
            )

    def _request_path(self) -> str:
        """只取 URL 路径部分，不把查询字符串误当成路由名。"""

        return urlsplit(self.path).path

    def _read_json_object(self) -> dict[str, object]:
        """读取请求体，并要求顶层 JSON 是对象。"""

        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise ValueError("请求体必须包含 JSON。")
        try:
            content_length = int(raw_length)
        except ValueError as exc:
            raise ValueError("Content-Length 必须是非负整数。") from exc
        if content_length < 0:
            raise ValueError("Content-Length 必须是非负整数。")

        raw_body = self.rfile.read(content_length)
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("请求体不是合法的 UTF-8 JSON。") from exc
        if not isinstance(payload, dict):
            raise ValueError("请求 JSON 顶层必须是对象。")
        return payload

    @staticmethod
    def _validate_command(payload: dict[str, object]) -> tuple[str, str]:
        """校验当前唯一支持的命令，并保留原始 command_id。"""

        if "command" not in payload:
            raise ValueError("缺少必填字段 command。")
        command = payload["command"]
        if command != SUPPORTED_COMMAND:
            raise ValueError(f"当前只支持 command={SUPPORTED_COMMAND!r}。")

        if "command_id" not in payload:
            raise ValueError("缺少必填字段 command_id。")
        command_id = payload["command_id"]
        if not isinstance(command_id, str) or not command_id.strip():
            raise ValueError("command_id 必须是非空字符串。")
        return SUPPORTED_COMMAND, command_id

    @staticmethod
    def _build_success_response(
        result: dict[str, object],
        command: str,
        command_id: str,
    ) -> dict[str, object]:
        """把第六步结果加入稳定的 HTTP 命令字段和检测数量。"""

        captured_at = result.get("captured_at")
        detections = result.get("detections")
        if not isinstance(captured_at, str) or not captured_at:
            raise ValueError("检测结果缺少 captured_at。")
        if result.get("coordinate_frame") != "kinect_camera":
            raise ValueError("检测结果坐标系不是 kinect_camera。")
        if result.get("coordinate_unit") != "mm":
            raise ValueError("检测结果坐标单位不是 mm。")
        if not isinstance(detections, list):
            raise ValueError("检测结果 detections 必须是列表。")

        response: dict[str, object] = {
            "status": "ok",
            "command": command,
            "command_id": command_id,
            "captured_at": captured_at,
            "coordinate_frame": "kinect_camera",
            "coordinate_unit": "mm",
            "detection_count": len(detections),
            "detections": detections,
        }
        # 第六步已有图像尺寸字段；原样沿用，不为 HTTP 再造另一套名称。
        for field_name in ("image_width", "image_height"):
            if field_name in result:
                response[field_name] = result[field_name]
        return response

    def _send_error_json(
        self,
        status: HTTPStatus,
        message: str,
        *,
        command: str | None = None,
        command_id: str | None = None,
    ) -> None:
        """使用统一 JSON 结构返回 HTTP 错误。"""

        payload: dict[str, object] = {"status": "error", "error": message}
        if command is not None:
            payload["command"] = command
        if command_id is not None:
            payload["command_id"] = command_id
        self._send_json(status, payload)

    def _send_json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
        """以 UTF-8 JSON 返回响应，中文类别名不转义。"""

        # allow_nan=False 保证响应符合 JSON 标准；异常浮点值按内部错误处理。
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
