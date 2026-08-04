"""第八步 MySQL 独立初始化入口测试。"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "08_init_mysql.py"
SPEC = importlib.util.spec_from_file_location("init_mysql_08", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
init_script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = init_script
SPEC.loader.exec_module(init_script)


class InitMySQLEntryTests(unittest.TestCase):
    def test_main_only_initializes_tables(self) -> None:
        with (
            mock.patch.object(init_script, "initialize_database") as initialize,
            mock.patch("builtins.print") as print_mock,
        ):
            init_script.main()

        initialize.assert_called_once_with()
        print_mock.assert_called_once_with(
            "MySQL 表初始化完成：vision_request、vision_detection"
        )


if __name__ == "__main__":
    unittest.main()
