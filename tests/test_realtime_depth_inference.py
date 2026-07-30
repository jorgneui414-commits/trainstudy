"""第五个脚本的任务选择、配置校验和精简接口测试。

测试只导入入口脚本中的辅助函数，不会执行 main()，因此不会打开 Kinect 或加载模型。
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


# 文件名以数字开头，不能写成普通的 import 语句，所以使用 importlib 按路径导入。
SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "05_realtime_depth_inference.py"
)
SPEC = importlib.util.spec_from_file_location("realtime_depth_inference_05", SCRIPT_PATH)
# spec 和 loader 是 Python 根据文件路径创建的“模块加载说明”；正常路径下两者都应存在。
assert SPEC is not None and SPEC.loader is not None
depth_script = importlib.util.module_from_spec(SPEC)
# 先登记到 sys.modules，再执行模块代码，行为与正常 import 更接近。
sys.modules[SPEC.name] = depth_script
SPEC.loader.exec_module(depth_script)


class RealtimeDepthInferenceTests(unittest.TestCase):
    def test_runtime_config_accepts_detect_and_obb(self) -> None:
        # mock.patch.object 只在 with 块内临时改配置，离开后会自动恢复原值。
        with mock.patch.object(depth_script.config, "YOLO_TASK", "detect"):
            self.assertEqual(depth_script._validate_runtime_config(), "detect")
        with mock.patch.object(depth_script.config, "YOLO_TASK", " OBB "):
            self.assertEqual(depth_script._validate_runtime_config(), "obb")

    def test_runtime_config_rejects_invalid_task_and_depth_range(self) -> None:
        # 第五步只支持 detect/obb，也要求最小深度严格小于最大深度。
        with mock.patch.object(depth_script.config, "YOLO_TASK", "segment"):
            with self.assertRaisesRegex(ValueError, "YOLO_TASK"):
                depth_script._validate_runtime_config()

        with (
            mock.patch.object(depth_script.config, "DEPTH_MIN_MM", 2000),
            mock.patch.object(depth_script.config, "DEPTH_MAX_MM", 1000),
        ):
            with self.assertRaisesRegex(ValueError, "DEPTH_MIN_MM"):
                depth_script._validate_runtime_config()

    def test_model_task_must_match_configured_task(self) -> None:
        # SimpleNamespace 在这里充当只含 task 属性的最小假模型。
        depth_script._validate_model_task(SimpleNamespace(task="detect"), "detect")
        depth_script._validate_model_task(SimpleNamespace(task="obb"), "obb")

        with self.assertRaisesRegex(ValueError, "YOLO_TASK"):
            depth_script._validate_model_task(SimpleNamespace(task="obb"), "detect")
        with self.assertRaisesRegex(RuntimeError, "无法.*识别"):
            depth_script._validate_model_task(SimpleNamespace(task=None), "detect")

    def test_robot_http_and_image_saving_interfaces_were_removed(self) -> None:
        # 这是精简重构的回归测试：这些旧接口以后也不应被入口脚本重新引入。
        for removed_name in (
            "CommandBroker",
            "PendingCommand",
            "RobotVisionHTTPServer",
            "RobotVisionHandler",
            "_build_command_response",
            "_save_command_images",
        ):
            self.assertFalse(hasattr(depth_script, removed_name), removed_name)


if __name__ == "__main__":
    unittest.main()
