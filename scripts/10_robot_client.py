"""第十步：用一个只收发 JSON 的简单客户端模拟机器人请求。

客户端先检查 ``GET /health``，再为本次命令生成唯一 ``command_id``，并按
第七步已经确定的协议调用 ``POST /vision/command``。它只展示检测结果，不选择
抓取目标，也不会发送任何机器人运动指令。
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from http.client import HTTPConnection
from pathlib import Path


# 当前文件位于 scripts 子目录；加入项目根目录后可直接运行本脚本。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from vision_http_server import SUPPORTED_COMMAND


_REQUEST_TIMEOUT_SECONDS = 30.0


def request_json(
    host: str,
    port: int,
    method: str,
    path: str,
    *,
    payload: dict[str, object] | None = None,
    timeout_seconds: float = _REQUEST_TIMEOUT_SECONDS,
) -> tuple[int, dict[str, object]]:
    """发送一次 HTTP 请求，并要求服务端返回 JSON 对象。"""

    body: bytes | None = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"

    connection = HTTPConnection(host, port, timeout=timeout_seconds)
    try:
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        status = response.status
        content_type = response.getheader("Content-Type", "")
        raw_body = response.read()
    finally:
        connection.close()

    media_type = content_type.split(";", maxsplit=1)[0].strip().lower()
    if media_type != "application/json":
        raise ValueError(f"服务端返回的 Content-Type 不是 JSON：{content_type!r}。")
    try:
        response_payload = json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("服务端返回的响应不是合法的 UTF-8 JSON。") from exc
    if not isinstance(response_payload, dict):
        raise ValueError("服务端返回的 JSON 顶层必须是对象。")
    return status, response_payload


def print_command_response(
    http_status: int,
    response: dict[str, object],
    sent_command_id: str,
) -> bool:
    """按机器人验收所需字段打印命令响应，返回视觉命令是否成功。"""

    print(f"HTTP 状态码: {http_status}")
    returned_command_id = response.get("command_id", sent_command_id)
    print(f"command_id: {returned_command_id}")

    if http_status != 200 or response.get("status") != "ok":
        print(f"错误: {response.get('error', '服务端返回未知错误。')}")
        return False

    if returned_command_id != sent_command_id:
        raise ValueError("响应 command_id 与本次请求不一致。")

    captured_at = response.get("captured_at")
    detection_count = response.get("detection_count")
    detections = response.get("detections")
    mysql_saved = response.get("mysql_saved")
    if not isinstance(captured_at, str) or not captured_at:
        raise ValueError("成功响应缺少非空 captured_at。")
    if (
        not isinstance(detection_count, int)
        or isinstance(detection_count, bool)
        or detection_count < 0
    ):
        raise ValueError("成功响应中的 detection_count 必须是非负整数。")
    if not isinstance(detections, list) or len(detections) != detection_count:
        raise ValueError("成功响应中的 detections 与 detection_count 不一致。")
    if not isinstance(mysql_saved, bool):
        raise ValueError("成功响应缺少布尔字段 mysql_saved。")

    print(f"captured_at: {captured_at}")
    print(f"detection_count: {detection_count}")
    for index, detection in enumerate(detections, start=1):
        if not isinstance(detection, dict):
            raise ValueError("detections 中的目标必须是 JSON 对象。")
        try:
            part_type = detection["part_type"]
            confidence = float(detection["confidence"])
            orientation_deg = float(detection["orientation_deg"])
            camera_x_mm = float(detection["camera_x_mm"])
            camera_y_mm = float(detection["camera_y_mm"])
            camera_z_mm = float(detection["camera_z_mm"])
        except KeyError as exc:
            raise ValueError(f"目标缺少字段 {exc.args[0]}。") from exc
        except (TypeError, ValueError) as exc:
            raise ValueError("目标的置信度、角度或相机 XYZ 不是数字。") from exc
        if not isinstance(part_type, str) or not part_type:
            raise ValueError("目标类别 part_type 必须是非空字符串。")

        print(f"目标 {index}:")
        print(f"  类别: {part_type}")
        print(f"  置信度: {confidence:.4f}")
        print(f"  角度: {orientation_deg:.2f} 度")
        print(
            "  相机 XYZ (mm): "
            f"({camera_x_mm:.2f}, {camera_y_mm:.2f}, {camera_z_mm:.2f})"
        )
    print(f"mysql_saved: {str(mysql_saved).lower()}")
    return True


def run_once(host: str, port: int) -> int:
    """完成一次健康检查和一次检测命令；成功返回 0，失败返回 1。"""

    started_at = time.perf_counter()
    try:
        health_status, health_response = request_json(host, port, "GET", "/health")
        print(f"健康检查 HTTP 状态码: {health_status}")
        if health_status != 200 or health_response.get("status") != "ok":
            print(f"健康检查错误: {health_response.get('error', '服务不可用。')}")
            return 1

        command_id = f"robot_{uuid.uuid4().hex}"
        command_payload = {
            "command": SUPPORTED_COMMAND,
            "command_id": command_id,
        }
        command_status, command_response = request_json(
            host,
            port,
            "POST",
            "/vision/command",
            payload=command_payload,
        )
        return 0 if print_command_response(
            command_status,
            command_response,
            command_id,
        ) else 1
    except Exception as exc:
        print(f"客户端请求失败: {exc}")
        return 1
    finally:
        elapsed_seconds = time.perf_counter() - started_at
        print(f"请求总耗时: {elapsed_seconds:.3f} 秒")


def main() -> None:
    """使用现有 HTTP 地址运行一次模拟机器人命令。"""

    exit_code = run_once(config.VISION_HTTP_HOST, config.VISION_HTTP_PORT)
    if exit_code != 0:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
