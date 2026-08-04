"""第七步 HTTP 入口的模型加载和关闭测试。"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "07_vision_http_server.py"
SPEC = importlib.util.spec_from_file_location("vision_http_server_07", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
server_script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = server_script
SPEC.loader.exec_module(server_script)


class VisionHTTPServerEntryTests(unittest.TestCase):
    def test_main_loads_model_once_and_closes_server(self) -> None:
        model = SimpleNamespace(task="obb")
        camera = SimpleNamespace()
        server = mock.Mock()
        server.serve_forever.return_value = None
        yolo_factory = mock.Mock(return_value=model)
        server_factory = mock.Mock(return_value=server)

        with (
            mock.patch.object(server_script.config, "INFERENCE_MODEL_PATH", SCRIPT_PATH),
            mock.patch.object(server_script.config, "VISION_HTTP_HOST", "127.0.0.1"),
            mock.patch.object(server_script.config, "VISION_HTTP_PORT", 8008),
            mock.patch.object(
                server_script,
                "KinectV2Camera",
                return_value=camera,
            ) as camera_factory,
            mock.patch.object(server_script, "VisionHTTPServer", server_factory),
            mock.patch.dict(
                sys.modules,
                {"ultralytics": SimpleNamespace(YOLO=yolo_factory)},
            ),
            mock.patch("builtins.print"),
        ):
            server_script.main()

        yolo_factory.assert_called_once_with(str(SCRIPT_PATH))
        camera_factory.assert_called_once_with(
            server_script.config.KINECT_SDK_ASSEMBLY_PATH
        )
        server_factory.assert_called_once_with(
            ("127.0.0.1", 8008),
            camera=camera,
            model=model,
        )
        server.serve_forever.assert_called_once_with()
        server.server_close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
