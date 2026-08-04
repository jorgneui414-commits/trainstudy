"""第六步：封装一次 Kinect v2 OBB 深度检测。

调用方负责加载一个 OBB 模型并打开 Kinect；本模块只编排一次取帧、一次推理、
既有结果解析和既有深度融合，不管理模型、不打开设备，也不选择机器人目标。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import config
from depth_detection import enrich_detections_with_depth, parse_yolo_result


class KinectFrameError(RuntimeError):
    """Kinect 在本次 ``camera.read()`` 中未能返回可用 RGB-D 帧。"""


def capture_and_detect(camera: Any, model: Any) -> dict[str, object]:
    """读取一组 RGB-D 帧并返回全部深度有效的 OBB 检测。

    ``camera`` 必须已由调用方打开，``model`` 必须是调用方只加载一次的 Ultralytics
    OBB 模型。本函数每次调用恰好执行一次 ``camera.read()`` 和一次
    ``model.predict()``；没有检测或没有有效深度都正常返回空 ``detections``。
    """

    model_task = getattr(model, "task", None)
    if not isinstance(model_task, str) or model_task.strip().lower() != "obb":
        raise ValueError("capture_and_detect() 只接受 task='obb' 的模型。")

    try:
        frame = camera.read(timeout_seconds=config.KINECT_FRAME_TIMEOUT_SECONDS)
    except Exception as exc:
        # HTTP 层需要把相机取帧失败映射为 503，同时让模型/推理异常保持为 500。
        # 单独包装 read() 的异常即可准确区分来源，不改变成功返回结构。
        raise KinectFrameError(f"Kinect RGB-D 取帧失败：{exc}") from exc
    captured_at = (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )
    image_height, image_width = frame.color_bgr.shape[:2]

    results = model.predict(
        source=frame.color_bgr,
        conf=config.INFERENCE_CONFIDENCE_THRESHOLD,
        imgsz=config.TRAIN_IMAGE_SIZE,
        device=config.INFERENCE_DEVICE,
        verbose=False,
        save=False,
    )
    detections_2d = parse_yolo_result(results[0], task="obb") if results else []
    detections_with_depth = enrich_detections_with_depth(
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

    # 第六步只把三维坐标可靠的目标交给机器人；不排序，也不挑选“最佳”目标。
    valid_detections = [
        detection
        for detection in detections_with_depth
        if bool(detection.get("depth_valid"))
    ]
    return {
        "captured_at": captured_at,
        "image_width": int(image_width),
        "image_height": int(image_height),
        "coordinate_frame": "kinect_camera",
        "coordinate_unit": "mm",
        "detections": valid_detections,
    }
