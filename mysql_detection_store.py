"""第八步：独立创建 MySQL 表并事务写入一次检测结果。

本模块只处理配置、SQL 和事务，不打开 Kinect、不加载模型，也不启动或调用 HTTP。
一条 ``vision_request`` 与它的全部 ``vision_detection`` 始终共用同一个事务。
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any

import pymysql

import config


_CREATE_VISION_REQUEST_SQL = """
CREATE TABLE IF NOT EXISTS vision_request (
    request_id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    command_id VARCHAR(128) NOT NULL,
    captured_at DATETIME(3) NOT NULL,
    detection_count INT UNSIGNED NOT NULL,
    status VARCHAR(32) NOT NULL,
    error_message TEXT NULL,
    created_at TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
    PRIMARY KEY (request_id),
    INDEX idx_vision_request_command_id (command_id),
    INDEX idx_vision_request_captured_at (captured_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

_CREATE_VISION_DETECTION_SQL = """
CREATE TABLE IF NOT EXISTS vision_detection (
    detection_id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    request_id BIGINT UNSIGNED NOT NULL,
    class_id INT NOT NULL,
    part_type VARCHAR(128) NOT NULL,
    confidence DOUBLE NOT NULL,
    center_x DOUBLE NOT NULL,
    center_y DOUBLE NOT NULL,
    bbox_width DOUBLE NOT NULL,
    bbox_height DOUBLE NOT NULL,
    obb_points_json TEXT NOT NULL,
    orientation_deg DOUBLE NOT NULL,
    distance_mm DOUBLE NOT NULL,
    camera_x_mm DOUBLE NOT NULL,
    camera_y_mm DOUBLE NOT NULL,
    camera_z_mm DOUBLE NOT NULL,
    created_at TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
    PRIMARY KEY (detection_id),
    INDEX idx_vision_detection_request_id (request_id),
    CONSTRAINT fk_vision_detection_request
        FOREIGN KEY (request_id) REFERENCES vision_request(request_id)
        ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

_INSERT_VISION_REQUEST_SQL = """
INSERT INTO vision_request
    (command_id, captured_at, detection_count, status, error_message)
VALUES
    (%s, %s, %s, %s, %s);
"""

_INSERT_VISION_DETECTION_SQL = """
INSERT INTO vision_detection
    (request_id, class_id, part_type, confidence, center_x, center_y,
     bbox_width, bbox_height, obb_points_json, orientation_deg, distance_mm,
     camera_x_mm, camera_y_mm, camera_z_mm)
VALUES
    (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
"""


def initialize_database() -> None:
    """连接已配置的数据库并创建第八步所需的两个表。"""

    connection = _open_connection()
    try:
        with connection.cursor() as cursor:
            # 先建父表，再建带外键的明细表。
            cursor.execute(_CREATE_VISION_REQUEST_SQL)
            cursor.execute(_CREATE_VISION_DETECTION_SQL)
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def save_detection_result(
    command_id: str,
    captured_at: str,
    detections: Iterable[Mapping[str, object]],
) -> int:
    """保存一条成功检测命令及其全部目标，并返回 ``request_id``。

    ``captured_at`` 直接沿用 ``capture_and_detect()`` 的 UTC ISO 8601 字符串；
    ``detections`` 直接沿用该接口返回的检测字典，不创建另一套字段命名。
    """

    if not isinstance(command_id, str) or not command_id.strip():
        raise ValueError("command_id 必须是非空字符串。")
    captured_datetime = _parse_captured_at(captured_at)

    try:
        detection_items = list(detections)
    except TypeError as exc:
        raise ValueError("detections 必须是可迭代的检测字典集合。") from exc
    if any(not isinstance(item, Mapping) for item in detection_items):
        raise ValueError("detections 中的每一项都必须是检测字典。")

    connection = _open_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                _INSERT_VISION_REQUEST_SQL,
                (
                    command_id,
                    captured_datetime,
                    len(detection_items),
                    "ok",
                    None,
                ),
            )
            request_id = int(cursor.lastrowid)
            if request_id <= 0:
                raise RuntimeError("写入 vision_request 后未取得有效 request_id。")

            # 多目标逐条写入，但直到全部成功后才统一提交。
            for detection in detection_items:
                cursor.execute(
                    _INSERT_VISION_DETECTION_SQL,
                    _build_detection_parameters(request_id, detection),
                )
        connection.commit()
        return request_id
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _open_connection() -> Any:
    """校验集中配置，并创建显式关闭自动提交的 PyMySQL 连接。"""

    connection_options = _mysql_connection_options()
    return pymysql.connect(
        **connection_options,
        autocommit=False,
    )


def _mysql_connection_options() -> dict[str, object]:
    """读取并校验 config.py 中的六项 MySQL 配置。"""

    values = {
        "MYSQL_HOST": config.MYSQL_HOST,
        "MYSQL_USER": config.MYSQL_USER,
        "MYSQL_PASSWORD": config.MYSQL_PASSWORD,
        "MYSQL_DATABASE": config.MYSQL_DATABASE,
        "MYSQL_CHARSET": config.MYSQL_CHARSET,
    }
    missing = [
        name
        for name, value in values.items()
        if not isinstance(value, str) or not value.strip()
    ]
    if missing:
        env_names = ", ".join(
            name.replace("MYSQL_", "VISION_MYSQL_") for name in missing
        )
        raise ValueError(
            "MySQL 配置缺失："
            f"{', '.join(missing)}。请在运行前设置环境变量：{env_names}。"
        )

    try:
        port = int(config.MYSQL_PORT)
    except (TypeError, ValueError) as exc:
        raise ValueError("MySQL 配置 MYSQL_PORT 必须是整数。") from exc
    if not 1 <= port <= 65535:
        raise ValueError("MySQL 配置 MYSQL_PORT 必须在 1 到 65535 之间。")

    charset = str(values["MYSQL_CHARSET"])
    if charset.lower() != "utf8mb4":
        raise ValueError("MySQL 配置 MYSQL_CHARSET 必须是 utf8mb4。")

    return {
        "host": str(values["MYSQL_HOST"]),
        "port": port,
        "user": str(values["MYSQL_USER"]),
        "password": str(values["MYSQL_PASSWORD"]),
        "database": str(values["MYSQL_DATABASE"]),
        "charset": charset,
    }


def _parse_captured_at(value: str) -> datetime:
    """把现有 UTC ISO 8601 字段转换成 MySQL DATETIME(3) 参数。"""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("captured_at 必须是非空的 ISO 8601 字符串。")
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError("captured_at 必须是合法的 ISO 8601 时间。") from exc
    if parsed.tzinfo is None:
        raise ValueError("captured_at 必须包含时区；当前检测接口使用 UTC 的 Z 后缀。")
    # MySQL DATETIME 不保存时区；统一换算到 UTC 后去掉 tzinfo，语义仍与原字段一致。
    return parsed.astimezone(timezone.utc).replace(tzinfo=None)


def _build_detection_parameters(
    request_id: int,
    detection: Mapping[str, object],
) -> tuple[object, ...]:
    """按 vision_detection 的列顺序生成参数化 SQL 参数。"""

    try:
        obb_points_json = json.dumps(
            detection["obb_points"],
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    except KeyError as exc:
        raise ValueError("检测字典缺少必填字段 obb_points。") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError("obb_points 必须能够序列化为标准 JSON。") from exc

    return (
        request_id,
        _required_int(detection, "class_id"),
        _required_text(detection, "part_type"),
        _required_float(detection, "confidence"),
        _required_float(detection, "center_x"),
        _required_float(detection, "center_y"),
        _required_float(detection, "bbox_width"),
        _required_float(detection, "bbox_height"),
        obb_points_json,
        _required_float(detection, "orientation_deg"),
        _required_float(detection, "distance_mm"),
        _required_float(detection, "camera_x_mm"),
        _required_float(detection, "camera_y_mm"),
        _required_float(detection, "camera_z_mm"),
    )


def _required_text(detection: Mapping[str, object], field_name: str) -> str:
    value = detection.get(field_name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"检测字段 {field_name} 必须是非空字符串。")
    return value


def _required_int(detection: Mapping[str, object], field_name: str) -> int:
    try:
        return int(detection[field_name])
    except KeyError as exc:
        raise ValueError(f"检测字典缺少必填字段 {field_name}。") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(f"检测字段 {field_name} 必须是整数。") from exc


def _required_float(detection: Mapping[str, object], field_name: str) -> float:
    try:
        result = float(detection[field_name])
    except KeyError as exc:
        raise ValueError(f"检测字典缺少必填字段 {field_name}。") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(f"检测字段 {field_name} 必须是数字。") from exc
    if not math.isfinite(result):
        raise ValueError(f"检测字段 {field_name} 必须是有限数字。")
    return result
