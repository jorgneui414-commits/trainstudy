"""第五步入口：用 Kinect v2 运行 Detect/OBB 并显示毫米制三维坐标。

这个脚本只负责编排运行顺序：读取配置 -> 打开 Kinect -> YOLO 推理 ->
融合深度 -> 显示窗口。相机读取和深度计算的具体实现分别放在独立模块中。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import cv2


# 当前文件位于 scripts 子目录，parents[1] 才是项目根目录。
# 把根目录加入模块搜索路径后，直接运行本脚本也能导入 config.py 等模块。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from camera_capture import EXIT_KEYS
from depth_detection import (
    colorize_depth,
    draw_detections,
    enrich_detections_with_depth,
    parse_yolo_result,
)
from kinect_v2_camera import KinectV2Camera


# 窗口名称集中定义，后续若想改标题只需要修改这里。
COLOR_WINDOW_NAME = "Kinect v2 YOLO + Depth"
DEPTH_WINDOW_NAME = "Kinect v2 Depth"


def main() -> None:
    """持续读取 Kinect RGB-D 帧、运行 YOLO 并显示目标三维坐标。

    正常情况下循环一直运行，直到用户按 q 或 Esc；发生错误或主动退出时，
    finally 代码块都会释放 Kinect，避免设备一直被当前 Python 进程占用。
    """

    # 先检查配置，避免相机打开后才发现任务名称或深度参数填写错误。
    selected_task = _validate_runtime_config()
    model_path = Path(config.INFERENCE_MODEL_PATH)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"找不到推理模型：{model_path}。请先训练模型，或修改 config.INFERENCE_MODEL_PATH。"
        )

    # 创建对象不会立即占用设备，真正打开 Kinect 的操作发生在 camera.open()。
    camera = KinectV2Camera(config.KINECT_SDK_ASSEMBLY_PATH)
    try:
        camera.open(availability_timeout_seconds=config.KINECT_AVAILABILITY_TIMEOUT_SECONDS)

        # 相机确认可用后再初始化 GPU 模型，避免硬件不可用时浪费初始化时间。
        from ultralytics import YOLO

        model = YOLO(str(model_path))
        # 防止用 detect 的解析方式读取 OBB 模型，或反过来读取。
        _validate_model_task(model, selected_task)

        print("Kinect v2 深度实时检测已启动。")
        print(f"任务：{selected_task}")
        print(f"模型：{model_path}")
        print(f"推理设备：{config.INFERENCE_DEVICE}，输入尺寸：{config.TRAIN_IMAGE_SIZE}")
        print("三维坐标单位：毫米（Kinect 相机坐标系）")
        print("按 q 或 Esc 退出。")

        while True:
            # perf_counter() 适合测量一帧处理耗时，不受系统时间调整影响。
            frame_started_at = time.perf_counter()

            # frame 同时包含彩色图、毫米深度图以及 SDK 计算好的两种坐标映射。
            frame = camera.read(timeout_seconds=config.KINECT_FRAME_TIMEOUT_SECONDS)

            # predict() 返回“每张输入图片一个结果”的列表；这里每次只输入一帧。
            results = model.predict(
                source=frame.color_bgr,
                conf=config.INFERENCE_CONFIDENCE_THRESHOLD,
                imgsz=config.TRAIN_IMAGE_SIZE,
                device=config.INFERENCE_DEVICE,
                verbose=False,
                save=False,
            )
            detections_2d = (
                parse_yolo_result(results[0], task=selected_task) if results else []
            )
            # OpenCV 图像形状是 (高度, 宽度, 通道数)，这里只需要前两个值。
            image_height, image_width = frame.color_bgr.shape[:2]

            # 给每个二维框补充 distance_mm 和相机坐标 camera_x/y/z_mm。
            detections = enrich_detections_with_depth(
                detections_2d,
                frame.depth_mm,
                frame.depth_to_color_xy,
                frame.depth_to_camera_xyz_mm,
                image_width=image_width,
                image_height=image_height,
                roi_ratio=config.DEPTH_ROI_RATIO,
                min_depth_mm=config.DEPTH_MIN_MM,
                max_depth_mm=config.DEPTH_MAX_MM,
                min_valid_samples=config.DEPTH_MIN_VALID_SAMPLES,
            )

            # max(..., 1e-9) 防止极端情况下除以 0；FPS = 1 / 单帧耗时。
            elapsed_seconds = max(time.perf_counter() - frame_started_at, 1e-9)
            annotated = draw_detections(
                frame.color_bgr,
                detections,
                fps=1.0 / elapsed_seconds,
            )
            depth_preview = colorize_depth(
                frame.depth_mm,
                min_depth_mm=config.DEPTH_MIN_MM,
                max_depth_mm=config.DEPTH_MAX_MM,
            )
            cv2.imshow(COLOR_WINDOW_NAME, annotated)
            cv2.imshow(DEPTH_WINDOW_NAME, depth_preview)
            # waitKey(1) 既刷新 OpenCV 窗口，也读取按键；& 0xFF 取得低 8 位键值。
            if cv2.waitKey(1) & 0xFF in EXIT_KEYS:
                break
    finally:
        # 即使推理、绘图或取帧抛出异常，也必须执行这两句清理代码。
        camera.release()
        cv2.destroyAllWindows()


def _validate_runtime_config() -> str:
    """检查第五步运行前必须满足的配置，并返回规范化后的任务名称。"""

    raw_task = config.YOLO_TASK
    if not isinstance(raw_task, str) or raw_task.strip().lower() not in {"detect", "obb"}:
        raise ValueError("YOLO_TASK 必须是 'detect' 或 'obb'。")
    selected_task = raw_task.strip().lower()

    # float()/int() 既允许配置写成 2，也允许写成 2.0，并统一后续比较方式。
    if float(config.KINECT_AVAILABILITY_TIMEOUT_SECONDS) <= 0:
        raise ValueError("KINECT_AVAILABILITY_TIMEOUT_SECONDS 必须大于 0。")
    if float(config.KINECT_FRAME_TIMEOUT_SECONDS) <= 0:
        raise ValueError("KINECT_FRAME_TIMEOUT_SECONDS 必须大于 0。")
    roi_ratio = float(config.DEPTH_ROI_RATIO)
    if not 0 < roi_ratio <= 1:
        raise ValueError("DEPTH_ROI_RATIO 必须位于 (0, 1]。")
    if int(config.DEPTH_MIN_MM) <= 0 or int(config.DEPTH_MIN_MM) >= int(config.DEPTH_MAX_MM):
        raise ValueError("深度范围必须满足 0 < DEPTH_MIN_MM < DEPTH_MAX_MM。")
    if int(config.DEPTH_MIN_VALID_SAMPLES) <= 0:
        raise ValueError("DEPTH_MIN_VALID_SAMPLES 必须大于 0。")
    return selected_task


def _validate_model_task(model: Any, expected_task: str) -> None:
    """确认权重文件声明的任务与 config.YOLO_TASK 完全一致。"""

    raw_task = getattr(model, "task", None)
    if not isinstance(raw_task, str) or not raw_task.strip():
        raise RuntimeError("无法从推理模型中识别 Detect/OBB 任务类型。")
    actual_task = raw_task.strip().lower()
    if actual_task != expected_task:
        raise ValueError(
            f"模型任务是 {actual_task!r}，但 YOLO_TASK 设置为 {expected_task!r}。"
            "请在 config.py 中选择匹配的任务和 INFERENCE_MODEL_PATH。"
        )


if __name__ == "__main__":
    # 只有“直接运行本文件”时才进入 main；测试导入本模块时不会自动打开 Kinect。
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        # 把常见启动错误整理成一行中文提示，同时用非零退出码告诉终端运行失败。
        raise SystemExit(f"Kinect 深度实时推理未启动：{exc}") from exc
