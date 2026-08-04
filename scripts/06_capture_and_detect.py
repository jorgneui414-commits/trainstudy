"""第六步入口：执行一次 Kinect v2 OBB 深度检测并打印结果。"""

from __future__ import annotations

import json
import sys
from pathlib import Path


# 当前文件位于 scripts 子目录；加入项目根目录后可直接运行本脚本。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from kinect_obb_detection import capture_and_detect
from kinect_v2_camera import KinectV2Camera


def main() -> None:
    """加载一次 OBB 模型、打开 Kinect、检测一帧、打印并释放设备。"""

    model_path = Path(config.INFERENCE_MODEL_PATH)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"找不到 OBB 推理模型：{model_path}。"
            "请把 OBB best.pt 放到 config.INFERENCE_MODEL_PATH 指定的位置。"
        )

    # 模型只在调用方加载一次，然后原样传给 capture_and_detect()。
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    camera = KinectV2Camera(config.KINECT_SDK_ASSEMBLY_PATH)
    try:
        camera.open(
            availability_timeout_seconds=config.KINECT_AVAILABILITY_TIMEOUT_SECONDS
        )
        result = capture_and_detect(camera, model)
        # ensure_ascii=False 让模型结果中的中文类别名称直接显示。
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        camera.release()


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        raise SystemExit(f"Kinect OBB 单次深度检测未完成：{exc}") from exc
