"""第八步 MySQL 建表、参数化写入和事务测试。"""

from __future__ import annotations

import json
import unittest
from datetime import datetime
from unittest import mock

import mysql_detection_store


class FakeCursor:
    """记录 execute() 调用，并可在指定调用处模拟数据库失败。"""

    def __init__(
        self,
        *,
        request_id: int = 9001,
        fail_on_execute: int | None = None,
    ) -> None:
        self.request_id = request_id
        self.fail_on_execute = fail_on_execute
        self.execute_calls: list[tuple[str, tuple[object, ...] | None]] = []
        self.lastrowid = 0

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        return False

    def execute(
        self,
        sql: str,
        parameters: tuple[object, ...] | None = None,
    ) -> None:
        self.execute_calls.append((sql, parameters))
        call_number = len(self.execute_calls)
        if self.fail_on_execute == call_number:
            raise RuntimeError("simulated insert failure")
        if "INSERT INTO vision_request" in sql:
            self.lastrowid = self.request_id


class FakeConnection:
    def __init__(self, cursor: FakeCursor | None = None) -> None:
        self.fake_cursor = cursor or FakeCursor()
        self.cursor_calls = 0
        self.commit_calls = 0
        self.rollback_calls = 0
        self.close_calls = 0

    def cursor(self) -> FakeCursor:
        self.cursor_calls += 1
        return self.fake_cursor

    def commit(self) -> None:
        self.commit_calls += 1

    def rollback(self) -> None:
        self.rollback_calls += 1

    def close(self) -> None:
        self.close_calls += 1


def make_detection(
    *,
    class_id: int = 0,
    part_type: str = "螺丝刀",
    offset: float = 0.0,
) -> dict[str, object]:
    return {
        "class_id": class_id,
        "part_type": part_type,
        "confidence": 0.95,
        "center_x": 100.0 + offset,
        "center_y": 200.0 + offset,
        "bbox_width": 30.0,
        "bbox_height": 20.0,
        "obb_points": [
            [85.0 + offset, 190.0 + offset],
            [115.0 + offset, 190.0 + offset],
            [115.0 + offset, 210.0 + offset],
            [85.0 + offset, 210.0 + offset],
        ],
        "orientation_deg": 15.0,
        "distance_mm": 1000.0 + offset,
        "camera_x_mm": 100.0 + offset,
        "camera_y_mm": -50.0 + offset,
        "camera_z_mm": 1000.0 + offset,
    }


class MySQLDetectionStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        config_patch = mock.patch.multiple(
            mysql_detection_store.config,
            MYSQL_HOST="127.0.0.1",
            MYSQL_PORT="3306",
            MYSQL_USER="vision_user",
            MYSQL_PASSWORD="test-only-password",
            MYSQL_DATABASE="vision_study",
            MYSQL_CHARSET="utf8mb4",
        )
        config_patch.start()
        self.addCleanup(config_patch.stop)

    def test_missing_config_is_reported_before_connecting(self) -> None:
        with (
            mock.patch.object(mysql_detection_store.config, "MYSQL_PASSWORD", ""),
            mock.patch.object(mysql_detection_store.pymysql, "connect") as connect,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "MYSQL_PASSWORD.*VISION_MYSQL_PASSWORD",
            ):
                mysql_detection_store.save_detection_result(
                    "cmd-missing-config",
                    "2026-08-03T08:00:00.000Z",
                    [],
                )

        connect.assert_not_called()

    def test_initialize_database_creates_both_tables_and_commits(self) -> None:
        connection = FakeConnection()
        with mock.patch.object(
            mysql_detection_store.pymysql,
            "connect",
            return_value=connection,
        ) as connect:
            mysql_detection_store.initialize_database()

        connect.assert_called_once_with(
            host="127.0.0.1",
            port=3306,
            user="vision_user",
            password="test-only-password",
            database="vision_study",
            charset="utf8mb4",
            autocommit=False,
        )
        self.assertEqual(len(connection.fake_cursor.execute_calls), 2)
        request_sql = connection.fake_cursor.execute_calls[0][0]
        detection_sql = connection.fake_cursor.execute_calls[1][0]
        self.assertIn("CREATE TABLE IF NOT EXISTS vision_request", request_sql)
        self.assertIn("CREATE TABLE IF NOT EXISTS vision_detection", detection_sql)
        self.assertIn("FOREIGN KEY (request_id)", detection_sql)
        self.assertIn("DEFAULT CHARSET=utf8mb4", request_sql)
        self.assertIn("DEFAULT CHARSET=utf8mb4", detection_sql)
        self.assertEqual(connection.commit_calls, 1)
        self.assertEqual(connection.rollback_calls, 0)
        self.assertEqual(connection.close_calls, 1)

    def test_zero_detections_writes_only_request_and_closes(self) -> None:
        connection = FakeConnection(FakeCursor(request_id=101))
        with mock.patch.object(
            mysql_detection_store.pymysql,
            "connect",
            return_value=connection,
        ):
            request_id = mysql_detection_store.save_detection_result(
                "cmd-zero",
                "2026-08-03T08:00:00.123Z",
                [],
            )

        self.assertEqual(request_id, 101)
        self.assertEqual(len(connection.fake_cursor.execute_calls), 1)
        request_sql, request_parameters = connection.fake_cursor.execute_calls[0]
        self.assertIn("INSERT INTO vision_request", request_sql)
        self.assertEqual(request_sql.count("%s"), 5)
        self.assertEqual(
            request_parameters,
            (
                "cmd-zero",
                datetime(2026, 8, 3, 8, 0, 0, 123000),
                0,
                "ok",
                None,
            ),
        )
        self.assertEqual(connection.commit_calls, 1)
        self.assertEqual(connection.rollback_calls, 0)
        self.assertEqual(connection.close_calls, 1)

    def test_single_detection_uses_parameters_and_utf8_json(self) -> None:
        connection = FakeConnection(FakeCursor(request_id=202))
        detection = make_detection()
        with mock.patch.object(
            mysql_detection_store.pymysql,
            "connect",
            return_value=connection,
        ):
            request_id = mysql_detection_store.save_detection_result(
                "cmd-single",
                "2026-08-03T08:00:00.000Z",
                [detection],
            )

        self.assertEqual(request_id, 202)
        self.assertEqual(len(connection.fake_cursor.execute_calls), 2)
        detection_sql, parameters = connection.fake_cursor.execute_calls[1]
        assert parameters is not None
        self.assertEqual(detection_sql.count("%s"), 14)
        self.assertNotIn("螺丝刀", detection_sql)
        self.assertEqual(parameters[0], 202)
        self.assertEqual(parameters[1], 0)
        self.assertEqual(parameters[2], "螺丝刀")
        self.assertEqual(json.loads(str(parameters[8])), detection["obb_points"])
        self.assertNotIn("\\u", str(parameters[8]))
        self.assertEqual(connection.commit_calls, 1)
        self.assertEqual(connection.rollback_calls, 0)
        self.assertEqual(connection.close_calls, 1)

    def test_multiple_detections_share_one_request_id_and_transaction(self) -> None:
        connection = FakeConnection(FakeCursor(request_id=303))
        detections = [
            make_detection(),
            make_detection(class_id=2, part_type="支架", offset=50.0),
        ]
        with mock.patch.object(
            mysql_detection_store.pymysql,
            "connect",
            return_value=connection,
        ):
            request_id = mysql_detection_store.save_detection_result(
                "cmd-multiple",
                "2026-08-03T08:00:00.000Z",
                detections,
            )

        self.assertEqual(request_id, 303)
        calls = connection.fake_cursor.execute_calls
        self.assertEqual(len(calls), 3)
        assert calls[0][1] is not None
        self.assertEqual(calls[0][1][2], 2)
        assert calls[1][1] is not None and calls[2][1] is not None
        self.assertEqual([calls[1][1][0], calls[2][1][0]], [303, 303])
        self.assertEqual([calls[1][1][2], calls[2][1][2]], ["螺丝刀", "支架"])
        self.assertEqual(connection.cursor_calls, 1)
        self.assertEqual(connection.commit_calls, 1)
        self.assertEqual(connection.rollback_calls, 0)
        self.assertEqual(connection.close_calls, 1)

    def test_detection_insert_failure_rolls_back_and_closes(self) -> None:
        # 第 1 次是请求行，第 2 次是第一个目标，第 3 次模拟第二个目标写入失败。
        cursor = FakeCursor(request_id=404, fail_on_execute=3)
        connection = FakeConnection(cursor)
        with mock.patch.object(
            mysql_detection_store.pymysql,
            "connect",
            return_value=connection,
        ):
            with self.assertRaisesRegex(RuntimeError, "simulated insert failure"):
                mysql_detection_store.save_detection_result(
                    "cmd-rollback",
                    "2026-08-03T08:00:00.000Z",
                    [
                        make_detection(),
                        make_detection(class_id=2, part_type="支架", offset=50.0),
                    ],
                )

        self.assertEqual(len(cursor.execute_calls), 3)
        self.assertEqual(connection.commit_calls, 0)
        self.assertEqual(connection.rollback_calls, 1)
        self.assertEqual(connection.close_calls, 1)


if __name__ == "__main__":
    unittest.main()
