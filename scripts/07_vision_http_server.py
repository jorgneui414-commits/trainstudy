"""第七步入口：启动同步、单线程的 Kinect OBB HTTP 服务。"""

from __future__ import annotations

import sys
from pathlib import Path


# 当前文件位于 scripts 子目录；加入项目根目录后可直接运行本脚本。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from kinect_v2_camera import KinectV2Camera
from vision_http_server import VisionHTTPServer


def main() -> None:
    """加载一次模型并启动服务；Kinect 延迟到第一条合法命令再打开。"""

    model_path = Path(config.INFERENCE_MODEL_PATH)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"找不到 OBB 推理模型：{model_path}。"
            "请把 OBB best.pt 放到 config.INFERENCE_MODEL_PATH 指定的位置。"
        )

    # 模型只在服务器启动时加载一次，后续所有串行请求复用同一个对象。
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    camera = KinectV2Camera(config.KINECT_SDK_ASSEMBLY_PATH)
    server = VisionHTTPServer(
        (config.VISION_HTTP_HOST, config.VISION_HTTP_PORT),
        camera=camera,
        model=model,
    )
    print(
        "Kinect OBB HTTP 服务已启动："
        f"http://{config.VISION_HTTP_HOST}:{config.VISION_HTTP_PORT}"
    )
    try:
        # HTTPServer 不创建请求线程；所有命令在这里按到达顺序串行处理。
        server.serve_forever()
    except KeyboardInterrupt:
        print("收到退出信号，正在关闭 Kinect 和 HTTP 端口。")
    finally:
        server.server_close()


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        raise SystemExit(f"Kinect OBB HTTP 服务未启动：{exc}") from exc
