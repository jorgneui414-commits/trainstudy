"""YOLO Detect/OBB 结果解析、Kinect 毫米制深度融合和可视化。

本模块只处理数据，不负责打开相机或运行模型。主要流程是：
1. 把 Ultralytics 的 Detect/OBB 结果整理成统一字典；
2. 在每个检测框中心区域寻找对应的有效深度点；
3. 补充相机坐标系中的 X/Y/Z（毫米），并绘制结果。
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

import cv2
import numpy as np


_SUPPORTED_TASKS = {"detect", "obb"}


def parse_yolo_result(result: Any, *, task: str) -> list[dict[str, object]]:
    """把一帧 Ultralytics 结果转换成统一的检测字典列表。

    Detect 和 OBB 在 Ultralytics 中使用不同属性保存结果。这里统一输出类别、
    置信度、中心点和框尺寸；OBB 还会额外输出四个角点及旋转角度。
    """

    selected_task = _normalize_task(task)
    # names 通常是 {类别编号: 类别名称}，例如 {0: "bolt"}。
    names = getattr(result, "names", {}) or {}

    if selected_task == "obb":
        # OBB 的 xyxyxyxy 形状是 (目标数, 4, 2)：每个目标有 4 个 (x, y) 角点。
        obb = getattr(result, "obb", None)
        if obb is None or len(obb) == 0:
            return []
        polygons = _to_numpy(obb.xyxyxyxy).reshape((-1, 4, 2))
        class_ids = _to_numpy(obb.cls).reshape(-1)
        confidences = _to_numpy(obb.conf).reshape(-1)
        return [
            _build_obb_detection(polygon, int(class_id), float(confidence), names)
            # zip 会把同一个目标的角点、类别编号和置信度配成一组。
            for polygon, class_id, confidence in zip(polygons, class_ids, confidences)
        ]

    # Detect 的 xyxy 每行依次是左上角 (x1, y1) 和右下角 (x2, y2)。
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    xyxy_rows = _to_numpy(boxes.xyxy).reshape((-1, 4))
    class_ids = _to_numpy(boxes.cls).reshape(-1)
    confidences = _to_numpy(boxes.conf).reshape(-1)
    return [
        _build_horizontal_detection(row, int(class_id), float(confidence), names)
        for row, class_id, confidence in zip(xyxy_rows, class_ids, confidences)
    ]


def enrich_detections_with_depth(
    detections: Iterable[Mapping[str, object]],
    depth_mm: np.ndarray,
    depth_to_color_xy: np.ndarray,
    depth_to_camera_xyz_mm: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    roi_ratio: float = 0.30,
    min_depth_mm: int = 500,
    max_depth_mm: int = 4500,
    min_valid_samples: int = 5,
) -> list[dict[str, object]]:
    """在 Detect/OBB 框中央区域采样深度并补充毫米制相机坐标。

    ``depth_mm`` 的每个元素是一个深度像素的距离。另两个映射数组告诉我们：
    这个深度像素落在彩色图的哪个位置，以及它在相机坐标系中的 X/Y/Z。
    三个数组的前两维必须都是 Kinect 深度图的高和宽。
    """

    _validate_depth_inputs(
        depth_mm,
        depth_to_color_xy,
        depth_to_camera_xyz_mm,
        image_width=image_width,
        image_height=image_height,
        roi_ratio=roi_ratio,
        min_depth_mm=min_depth_mm,
        max_depth_mm=max_depth_mm,
        min_valid_samples=min_valid_samples,
    )

    # ``[..., 0]`` 表示保留前面所有行和列，只取最后一维的第 0 个值。
    # 因此 color_x/color_y 都是与 depth_mm 一样大的二维数组。
    color_x = depth_to_color_xy[..., 0]
    color_y = depth_to_color_xy[..., 1]
    camera_x_mm = depth_to_camera_xyz_mm[..., 0]
    camera_y_mm = depth_to_camera_xyz_mm[..., 1]
    camera_z_mm = depth_to_camera_xyz_mm[..., 2]

    # 布尔数组 base_valid 会逐像素记录“能否用于计算”。& 表示所有条件都要满足：
    # 映射坐标不能是无穷值、必须落在彩色图内、深度要在可靠量程内，且 XYZ 有效。
    base_valid = (
        np.isfinite(color_x)
        & np.isfinite(color_y)
        & (color_x >= 0)
        & (color_x < image_width)
        & (color_y >= 0)
        & (color_y < image_height)
        & (depth_mm >= min_depth_mm)
        & (depth_mm <= max_depth_mm)
        & np.isfinite(camera_x_mm)
        & np.isfinite(camera_y_mm)
        & np.isfinite(camera_z_mm)
        & (camera_z_mm > 0)
    )

    enriched: list[dict[str, object]] = []
    for detection in detections:
        # 复制字典再补充深度字段，避免意外修改调用者传入的二维检测结果。
        det = dict(detection)
        center_x = float(det["center_x"])
        center_y = float(det["center_y"])

        # 只使用框中央的小区域，能降低框内背景或边缘空洞对结果的影响。
        roi_width = max(float(det["bbox_width"]) * roi_ratio, 1.0)
        roi_height = max(float(det["bbox_height"]) * roi_ratio, 1.0)
        roi_x1 = max(0.0, center_x - roi_width / 2.0)
        roi_x2 = min(float(image_width), center_x + roi_width / 2.0)
        roi_y1 = max(0.0, center_y - roi_height / 2.0)
        roi_y2 = min(float(image_height), center_y + roi_height / 2.0)

        # 深度图与彩色图分辨率不同。这里不是直接用彩色图坐标索引 depth_mm，
        # 而是用 SDK 给出的 color_x/color_y 找出落进当前彩色 ROI 的深度像素。
        sample_mask = (
            base_valid
            & (color_x >= roi_x1)
            & (color_x < roi_x2)
            & (color_y >= roi_y1)
            & (color_y < roi_y2)
        )
        valid_count = int(np.count_nonzero(sample_mask))
        if valid_count < min_valid_samples:
            # 有效点不够时只令三维字段无效，二维类别和检测框仍然保留并显示。
            det.update(_invalid_depth_fields(valid_count))
            enriched.append(det)
            continue

        sampled_depth_mm = np.asarray(depth_mm[sample_mask], dtype=np.float64)
        sampled_camera_xyz_mm = np.asarray(
            depth_to_camera_xyz_mm[sample_mask],
            dtype=np.float64,
        )
        # 中位数比单个中心像素或平均值更不容易被深度噪点、孔洞影响。
        # axis=0 表示分别计算所有采样点的 X、Y、Z 中位数。
        median_camera_xyz_mm = np.median(sampled_camera_xyz_mm, axis=0)
        det.update(
            {
                "distance_mm": float(np.median(sampled_depth_mm)),
                "camera_x_mm": float(median_camera_xyz_mm[0]),
                "camera_y_mm": float(median_camera_xyz_mm[1]),
                "camera_z_mm": float(median_camera_xyz_mm[2]),
                "depth_valid": True,
                "depth_sample_count": valid_count,
            }
        )
        enriched.append(det)
    return enriched


def colorize_depth(
    depth_mm: np.ndarray,
    *,
    min_depth_mm: int = 500,
    max_depth_mm: int = 4500,
) -> np.ndarray:
    """把 Kinect 原始毫米深度图转换为便于人眼观察的 BGR 伪彩色图。"""

    if depth_mm.ndim != 2:
        raise ValueError("depth_mm 必须是二维数组。")
    if min_depth_mm >= max_depth_mm:
        raise ValueError("min_depth_mm 必须小于 max_depth_mm。")

    # OpenCV 的颜色映射需要 0~255 的单通道图，所以先把有效深度线性缩放。
    valid = (depth_mm >= min_depth_mm) & (depth_mm <= max_depth_mm)
    normalized = np.zeros(depth_mm.shape, dtype=np.uint8)
    if np.any(valid):
        scaled = (depth_mm[valid].astype(np.float32) - min_depth_mm) / (
            max_depth_mm - min_depth_mm
        )
        # 近处使用暖色、远处使用冷色。
        normalized[valid] = np.clip((1.0 - scaled) * 255.0, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    # 0、超量程等无效深度统一显示为黑色，避免被误认为真实距离。
    colored[~valid] = 0
    return colored


def draw_detections(
    frame: np.ndarray,
    detections: Iterable[Mapping[str, object]],
    *,
    fps: float | None = None,
) -> np.ndarray:
    """在彩色帧副本上绘制检测框、置信度和毫米制 X/Y/Z。"""

    if frame is None:
        raise ValueError("frame 不能为空。")
    # 在副本上绘图，调用者仍可继续使用没有文字和框线的原始彩色帧。
    output = frame.copy()
    for det in detections:
        # 绿色代表三维深度有效，橙色代表只有二维检测结果。
        color = (0, 180, 0) if bool(det.get("depth_valid")) else (0, 165, 255)
        x1 = int(round(float(det["x1"])))
        y1 = int(round(float(det["y1"])))
        x2 = int(round(float(det["x2"])))
        y2 = int(round(float(det["y2"])))
        points = det.get("obb_points")
        if points:
            # OBB 用四角点画旋转多边形；Detect 没有角点，所以走下面的水平矩形分支。
            polygon = np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(output, [polygon], True, color, 2)
        else:
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

        label = f"{det.get('part_type', 'unknown')} {float(det.get('confidence', 0.0)):.2f}"
        orientation_deg = det.get("orientation_deg")
        if orientation_deg is not None:
            label += f" angle={float(orientation_deg):.1f}deg"

        if bool(det.get("depth_valid")):
            coordinate_text = (
                f"X={float(det['camera_x_mm']):.0f} "
                f"Y={float(det['camera_y_mm']):.0f} "
                f"Z={float(det['camera_z_mm']):.0f} mm"
            )
        else:
            coordinate_text = "XYZ=invalid"

        label_y = max(22, y1 - 32)
        coordinate_y = max(46, y1 - 8)
        text_x = max(0, x1)
        cv2.putText(
            output,
            label,
            (text_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            coordinate_text,
            (text_x, coordinate_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )

    if fps is not None:
        cv2.putText(
            output,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return output


def _normalize_task(task: str) -> str:
    """去除任务名称两端空格并统一成小写，同时拒绝不支持的任务。"""

    if not isinstance(task, str):
        raise ValueError("task 必须是 'detect' 或 'obb'。")
    selected_task = task.strip().lower()
    if selected_task not in _SUPPORTED_TASKS:
        raise ValueError("task 必须是 'detect' 或 'obb'。")
    return selected_task


def _build_horizontal_detection(
    xyxy: np.ndarray,
    class_id: int,
    confidence: float,
    names: Any,
) -> dict[str, object]:
    """把一个 Detect 水平框整理成深度融合所需的统一字段。"""

    x1, y1, x2, y2 = [float(value) for value in xyxy]
    bbox_width = max(0.0, x2 - x1)
    bbox_height = max(0.0, y2 - y1)
    return {
        "part_type": _class_name(names, class_id),
        "class_id": class_id,
        "confidence": confidence,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "center_x": x1 + bbox_width / 2.0,
        "center_y": y1 + bbox_height / 2.0,
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
        "orientation_deg": None,
        "box_type": "horizontal",
    }


def _build_obb_detection(
    polygon: np.ndarray,
    class_id: int,
    confidence: float,
    names: Any,
) -> dict[str, object]:
    """把一个 OBB 四角点框整理成与 Detect 兼容的统一字段。"""

    points = [[float(x), float(y)] for x, y in polygon]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    return {
        "part_type": _class_name(names, class_id),
        "class_id": class_id,
        "confidence": confidence,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "center_x": sum(xs) / 4.0,
        "center_y": sum(ys) / 4.0,
        "bbox_width": x2 - x1,
        "bbox_height": y2 - y1,
        "orientation_deg": _polygon_angle_deg(points),
        "obb_points": points,
        "box_type": "obb",
    }


def _invalid_depth_fields(valid_count: int) -> dict[str, object]:
    """生成深度无效时的统一字段，None 表示当前没有可靠三维值。"""

    return {
        "distance_mm": None,
        "camera_x_mm": None,
        "camera_y_mm": None,
        "camera_z_mm": None,
        "depth_valid": False,
        "depth_sample_count": valid_count,
    }


def _validate_depth_inputs(
    depth_mm: np.ndarray,
    depth_to_color_xy: np.ndarray,
    depth_to_camera_xyz_mm: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    roi_ratio: float,
    min_depth_mm: int,
    max_depth_mm: int,
    min_valid_samples: int,
) -> None:
    """在计算前检查数组形状和深度参数，尽早给出容易理解的错误。"""

    if depth_mm.ndim != 2:
        raise ValueError("depth_mm 必须是二维数组。")
    expected_shape = depth_mm.shape
    if depth_to_color_xy.shape != (*expected_shape, 2):
        raise ValueError("depth_to_color_xy 的形状必须为 (深度高, 深度宽, 2)。")
    if depth_to_camera_xyz_mm.shape != (*expected_shape, 3):
        raise ValueError("depth_to_camera_xyz_mm 的形状必须为 (深度高, 深度宽, 3)。")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_width 和 image_height 必须大于 0。")
    if not math.isfinite(roi_ratio) or not 0 < roi_ratio <= 1:
        raise ValueError("roi_ratio 必须位于 (0, 1]。")
    if min_depth_mm <= 0 or min_depth_mm >= max_depth_mm:
        raise ValueError("深度范围必须满足 0 < min_depth_mm < max_depth_mm。")
    if min_valid_samples <= 0:
        raise ValueError("min_valid_samples 必须大于 0。")


def _polygon_angle_deg(points: list[list[float]]) -> float:
    """根据 OBB 第一个角点到第二个角点的方向计算角度。"""

    dx = points[1][0] - points[0][0]
    dy = points[1][1] - points[0][1]
    return math.degrees(math.atan2(dy, dx))


def _class_name(names: Any, class_id: int) -> str:
    """兼容字典或列表形式的类别名称；找不到时生成备用名称。"""

    if isinstance(names, Mapping):
        return str(names.get(class_id, f"class_{class_id}"))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return f"class_{class_id}"


def _to_numpy(value: Any) -> np.ndarray:
    """把 PyTorch 张量或普通序列安全转换为 NumPy 数组。

    推理张量可能位于 GPU，必须先与梯度图分离并搬到 CPU，才能调用 numpy()。
    测试中直接传入的 NumPy 数组也能通过最后一行正常处理。
    """

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)
