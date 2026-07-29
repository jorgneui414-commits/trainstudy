"""使用普通 USB 相机进行实时 YOLO 推理验证。"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from camera_capture import EXIT_KEYS, open_camera


WINDOW_NAME = "YOLO 实时推理"


def main() -> None:
    """检查配置相机后，持续显示带 YOLO 预测结果的实时画面。"""

    model_path = Path(config.INFERENCE_MODEL_PATH)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"找不到推理模型：{model_path}。请先训练模型，或修改 config.INFERENCE_MODEL_PATH。"
        )

    # 只尝试 config.CAMERA_ID；不会自动扫描其他编号，避免误选多相机环境中的设备。
    camera = open_camera(config.CAMERA_ID)
    try:
        if not camera.isOpened():
            raise RuntimeError(
                f"未检测到可用摄像头 CAMERA_ID={config.CAMERA_ID}。"
                "请连接普通 UVC 相机、确认 Windows 已识别设备，或修改 config.CAMERA_ID。"
            )

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, float(config.FRAME_WIDTH))
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, float(config.FRAME_HEIGHT))
        ok, frame = camera.read()
        if not ok or frame is None:
            raise RuntimeError(
                f"摄像头 CAMERA_ID={config.CAMERA_ID} 已打开，但无法读取首帧。"
                "请检查相机占用、隐私权限和连接状态。"
            )

        # 相机可用后才加载模型，避免未接硬件时浪费 GPU 初始化时间。
        from ultralytics import YOLO

        model = YOLO(str(model_path))
        actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("实时推理已启动。")
        print(f"模型：{model_path}")
        print(f"设备：{config.INFERENCE_DEVICE}")
        print(f"相机：{config.CAMERA_ID}（实际分辨率：{actual_width}x{actual_height}）")
        print("按 q 或 Esc 退出。")

        # 首帧已在相机连通性检查中读取，后续循环每次读取一帧并完成预测、标注和显示。
        frame_started_at = time.perf_counter()
        while True:
            results = model.predict(
                source=frame,
                conf=config.INFERENCE_CONFIDENCE_THRESHOLD,
                imgsz=config.TRAIN_IMAGE_SIZE,
                device=config.INFERENCE_DEVICE,
                verbose=False,
                save=False,
            )
            # Ultralytics 可能返回只读数组；复制后 OpenCV 才能安全叠加 FPS 文字。
            annotated_frame = (results[0].plot() if results else frame).copy()

            # 后续帧从读取开始计时，显示单帧读取、推理和标注处理的 FPS。
            elapsed_seconds = max(time.perf_counter() - frame_started_at, 1e-9)
            fps = 1.0 / elapsed_seconds
            cv2.putText(
                annotated_frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(WINDOW_NAME, annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in EXIT_KEYS:
                break

            frame_started_at = time.perf_counter()
            ok, frame = camera.read()
            if not ok or frame is None:
                raise RuntimeError(f"从摄像头 CAMERA_ID={config.CAMERA_ID} 读取帧失败。")
    finally:
        camera.release()
        try:
            cv2.destroyWindow(WINDOW_NAME)
        except cv2.error:
            pass


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"实时推理未启动：{exc}") from exc
