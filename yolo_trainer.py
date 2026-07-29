"""YOLO Detect / OBB 训练前校验与训练调用。

本模块不在导入时加载 PyTorch 或 Ultralytics。这样普通的数据集校验、
自动测试以及 CPU/GPU 配置错误都能给出更直接的提示，也避免测试时下载模型。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml


# 步骤 02 生成的数据集固定分为训练、验证、测试三部分。
SPLIT_NAMES = ("train", "val", "test")
# 标签中一行的列数能反映任务类型：普通框是 5 列，旋转框是 9 列。
LABEL_COLUMNS_TO_TASK = {5: "detect", 9: "obb"}
SUPPORTED_TASKS = frozenset(LABEL_COLUMNS_TO_TASK.values())


class TrainingConfigurationError(ValueError):
    """训练配置、模型或数据集彼此不匹配时抛出的异常。"""


class TrainingRuntimeError(RuntimeError):
    """所选 CPU/GPU 设备在当前环境无法使用时抛出的异常。"""


@dataclass(frozen=True)
class TrainingOptions:
    """从 ``config.py`` 汇总而来的训练参数。

    把大量参数放入一个对象后，03 入口只需要组装它，真正训练逻辑则不依赖
    全局变量，因而更容易阅读和测试。
    """

    # 数据集位置、模型类型和运行设备。
    data_yaml: str | Path
    task: str
    model: str | Path
    device: str
    gpu_index: int
    # 一轮训练的主要超参数。
    epochs: int
    image_size: int
    batch_size: int
    patience: int
    workers: int
    cache: bool | str
    amp: bool
    seed: int
    output_dir: str | Path
    run_name: str
    exist_ok: bool = False
    save_period: int = -1


@dataclass(frozen=True)
class DatasetTaskInfo:
    """扫描已构建数据集后得到的标签任务信息。"""

    task: str
    dataset_root: Path
    label_file_count: int
    annotation_count: int


def _normalize_task(task: str) -> str:
    # 允许用户写成 "Detect"，内部统一转为小写，后续比较时就不会受大小写影响。
    if not isinstance(task, str):
        raise TrainingConfigurationError("YOLO_TASK 必须是字符串：'detect' 或 'obb'。")

    normalized = task.strip().casefold()
    if normalized not in SUPPORTED_TASKS:
        choices = "、".join(sorted(SUPPORTED_TASKS))
        raise TrainingConfigurationError(f"YOLO_TASK 必须是以下值之一：{choices}。")
    return normalized


def _require_positive_integer(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TrainingConfigurationError(f"{name} 必须是正整数。")


def _validate_options(options: TrainingOptions) -> None:
    # 在下载模型、占用显存之前尽早发现明显的配置错误。
    _normalize_task(options.task)
    if not str(options.model).strip():
        raise TrainingConfigurationError("YOLO_MODEL 不能为空。")

    if not isinstance(options.device, str) or options.device.strip().casefold() not in {"cpu", "gpu"}:
        raise TrainingConfigurationError("TRAIN_DEVICE 必须是 'cpu' 或 'gpu'。")

    if isinstance(options.gpu_index, bool) or not isinstance(options.gpu_index, int) or options.gpu_index < 0:
        raise TrainingConfigurationError("TRAIN_GPU_INDEX 必须是大于等于 0 的整数。")

    for value, name in (
        (options.epochs, "TRAIN_EPOCHS"),
        (options.image_size, "TRAIN_IMAGE_SIZE"),
        (options.batch_size, "TRAIN_BATCH_SIZE"),
        (options.patience, "TRAIN_PATIENCE"),
        (options.workers, "TRAIN_WORKERS"),
    ):
        _require_positive_integer(value, name)

    if options.cache not in (True, False, "ram", "disk"):
        raise TrainingConfigurationError("TRAIN_CACHE 只能是 True、False、'ram' 或 'disk'。")
    if not isinstance(options.amp, bool):
        raise TrainingConfigurationError("TRAIN_AMP 必须是布尔值。")
    if isinstance(options.seed, bool) or not isinstance(options.seed, int):
        raise TrainingConfigurationError("TRAIN_SEED 必须是整数。")
    if not isinstance(options.exist_ok, bool):
        raise TrainingConfigurationError("TRAIN_EXIST_OK 必须是布尔值。")
    if isinstance(options.save_period, bool) or not isinstance(options.save_period, int):
        raise TrainingConfigurationError("TRAIN_SAVE_PERIOD 必须是整数。")
    if not str(options.run_name).strip():
        raise TrainingConfigurationError("TRAIN_RUN_NAME 不能为空。")


def _read_dataset_root(data_yaml: Path) -> Path:
    if not data_yaml.is_file():
        raise FileNotFoundError(
            f"找不到数据集配置文件：{data_yaml}。请先运行 scripts/02_build_yolo_dataset.py。"
        )

    try:
        data = yaml.safe_load(data_yaml.read_text(encoding="utf-8-sig"))
    except yaml.YAMLError as exc:
        raise TrainingConfigurationError(f"无法读取 data.yaml：{data_yaml}") from exc

    if not isinstance(data, dict):
        raise TrainingConfigurationError(f"data.yaml 必须是键值配置：{data_yaml}")
    for split_name in SPLIT_NAMES:
        if split_name not in data:
            raise TrainingConfigurationError(f"data.yaml 缺少 {split_name!r} 数据集划分。")

    # data.yaml 的 path 既可以是绝对路径，也可以是相对于 data.yaml 的相对路径。
    root_value = data.get("path", data_yaml.parent)
    if not isinstance(root_value, (str, Path)):
        raise TrainingConfigurationError("data.yaml 中的 path 必须是目录路径。")

    dataset_root = Path(root_value).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = data_yaml.parent / dataset_root
    dataset_root = dataset_root.resolve()
    if not dataset_root.is_dir():
        raise TrainingConfigurationError(f"data.yaml 指向的数据集目录不存在：{dataset_root}")
    return dataset_root


def inspect_dataset_task(data_yaml: str | Path) -> DatasetTaskInfo:
    """扫描标签，判断数据集用于普通 Detect 还是 OBB 训练。

    本项目步骤 02 会把标签放在 ``labels/train``、``labels/val``、
    ``labels/test``。空标签文件表示“这张图片没有目标”，属于合法负样本；
    但所有标签都为空时，模型没有可学习的目标，因此不能开始训练。
    """

    yaml_path = Path(data_yaml).expanduser().resolve()
    dataset_root = _read_dataset_root(yaml_path)
    column_count: int | None = None
    label_file_count = 0
    annotation_count = 0

    for split_name in SPLIT_NAMES:
        # 图片和标签在步骤 02 中按同样的 train/val/test 名称分目录保存。
        label_dir = dataset_root / "labels" / split_name
        if not label_dir.is_dir():
            raise TrainingConfigurationError(f"找不到 {split_name} 标签目录：{label_dir}")

        label_paths = sorted(label_dir.glob("*.txt"), key=lambda path: path.name.casefold())
        if not label_paths:
            raise TrainingConfigurationError(f"{split_name} 标签目录为空：{label_dir}")

        label_file_count += len(label_paths)
        for label_path in label_paths:
            try:
                text = label_path.read_text(encoding="utf-8-sig")
            except UnicodeDecodeError as exc:
                raise TrainingConfigurationError(f"标签文件不是 UTF-8 编码：{label_path}") from exc

            for line_number, raw_line in enumerate(text.splitlines(), start=1):
                line = raw_line.strip()
                if not line:
                    # 空行或空标签文件不计作目标，但允许作为负样本存在。
                    continue

                # 这里只检查列数来区分任务；数值范围已由步骤 02 的构建器校验过。
                current_columns = len(line.split())
                if current_columns not in LABEL_COLUMNS_TO_TASK:
                    raise TrainingConfigurationError(
                        f"标签格式错误：{label_path}:{line_number} 有 {current_columns} 列；"
                        "普通 Detect 需要 5 列，OBB 需要 9 列。"
                    )
                if column_count is None:
                    column_count = current_columns
                elif current_columns != column_count:
                    raise TrainingConfigurationError(
                        f"Detect 和 OBB 标签不能混用：{label_path}:{line_number} 的列数与前面的标签不同。"
                    )
                annotation_count += 1

    if column_count is None:
        raise TrainingConfigurationError("所有标签文件均为空，无法开始训练。")

    return DatasetTaskInfo(
        task=LABEL_COLUMNS_TO_TASK[column_count],
        dataset_root=dataset_root,
        label_file_count=label_file_count,
        annotation_count=annotation_count,
    )


def _import_torch() -> Any:
    # 只有选择 GPU 时才导入 torch，CPU 配置的参数校验不需要提前加载它。
    try:
        import torch
    except ImportError as exc:
        raise TrainingRuntimeError(
            "未安装 PyTorch，无法使用 GPU 训练。请使用 part_yolo_gpu 环境运行本脚本。"
        ) from exc
    return torch


def resolve_training_device(
    device: str,
    gpu_index: int,
    *,
    torch_module: Any | None = None,
) -> str | int:
    """把易读的 ``cpu`` / ``gpu`` 配置转换成 Ultralytics 所需的设备值。

    Ultralytics 以字符串 ``"cpu"`` 表示 CPU，以整数 ``0``、``1`` 等表示 GPU 编号。
    """

    if not isinstance(device, str):
        raise TrainingConfigurationError("TRAIN_DEVICE 必须是 'cpu' 或 'gpu'。")

    normalized = device.strip().casefold()
    if normalized == "cpu":
        # 明确返回 "cpu"，而不是让框架自行选择设备，避免意外占用显卡。
        return "cpu"
    if normalized != "gpu":
        raise TrainingConfigurationError("TRAIN_DEVICE 必须是 'cpu' 或 'gpu'。")

    if isinstance(gpu_index, bool) or not isinstance(gpu_index, int) or gpu_index < 0:
        raise TrainingConfigurationError("TRAIN_GPU_INDEX 必须是大于等于 0 的整数。")

    # 单元测试会传入假的 torch 对象；实际运行时才导入真实 PyTorch。
    torch_module = torch_module if torch_module is not None else _import_torch()
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not cuda.is_available():
        raise TrainingRuntimeError(
            "TRAIN_DEVICE 设置为 'gpu'，但当前 PyTorch 未检测到 CUDA。"
            "请使用 E:\\anaconda\\envs\\part_yolo_gpu\\python.exe 运行，"
            "或把 TRAIN_DEVICE 改为 'cpu'。"
        )

    device_count = cuda.device_count()
    if gpu_index >= device_count:
        raise TrainingRuntimeError(
            f"TRAIN_GPU_INDEX={gpu_index} 不可用；当前只检测到 {device_count} 张 CUDA GPU。"
        )
    return gpu_index


def _import_yolo_factory() -> Callable[[str], Any]:
    # 延迟导入可让数据集检查和单元测试无需初始化 Ultralytics。
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise TrainingRuntimeError(
            "未安装 ultralytics。请先在 part_yolo_gpu 环境中执行 requirements.txt 的安装命令。"
        ) from exc
    return YOLO


def _model_task(model: Any) -> str:
    # 预训练权重本身也带有任务类型，例如 yolo11n.pt 是 detect，-obb.pt 是 obb。
    task = getattr(model, "task", None)
    if not isinstance(task, str):
        raise TrainingConfigurationError("无法从所选模型识别任务类型，请换用兼容的 YOLO Detect 或 OBB 模型。")
    return task.casefold()


def _save_dir_from_training(model: Any, training_result: Any, options: TrainingOptions) -> Path:
    save_dir = getattr(training_result, "save_dir", None)
    if save_dir is None:
        save_dir = getattr(getattr(model, "trainer", None), "save_dir", None)
    if save_dir is None:
        # Ultralytics 正常训练会提供 save_dir；这个后备路径仅用于兼容不同版本的返回值。
        save_dir = Path(options.output_dir) / options.run_name
    return Path(save_dir).expanduser().resolve()


def train_yolo(
    options: TrainingOptions,
    *,
    yolo_factory: Callable[[str], Any] | None = None,
    torch_module: Any | None = None,
    printer: Callable[[str], None] = print,
) -> Path:
    """依次校验数据、设备、模型，然后调用 Ultralytics 开始训练。

    校验顺序很重要：先确认标签任务正确，再确认 GPU 可用，最后才下载/加载模型，
    这样配置错误不会浪费下载时间或占用显存。``yolo_factory`` 和 ``torch_module``
    是给自动测试替换真实库用的，日常运行时不需要填写。
    """

    _validate_options(options)
    selected_task = _normalize_task(options.task)
    dataset_info = inspect_dataset_task(options.data_yaml)
    if dataset_info.task != selected_task:
        # 不能把普通水平框“当作”旋转框训练；两种标注表达的信息不同。
        format_hint = (
            "OBB 训练需要 9 列四角点标签；请先重新标注并构建数据集。"
            if selected_task == "obb"
            else "普通 Detect 训练需要 5 列中心点、宽高标签；请检查 YOLO_TASK 或重新构建数据集。"
        )
        raise TrainingConfigurationError(
            f"YOLO_TASK={selected_task!r} 与数据集标签格式不匹配；"
            f"当前数据集是 {dataset_info.task!r}。"
            + format_hint
        )

    resolved_device = resolve_training_device(
        options.device,
        options.gpu_index,
        torch_module=torch_module,
    )
    # 到这里才加载模型：模型名可能触发首次预训练权重下载。
    factory = yolo_factory if yolo_factory is not None else _import_yolo_factory()
    model = factory(str(options.model))
    actual_model_task = _model_task(model)
    if actual_model_task != selected_task:
        raise TrainingConfigurationError(
            f"模型 {options.model!s} 的任务是 {actual_model_task!r}，"
            f"但 YOLO_TASK 设置为 {selected_task!r}。请让模型、标签和配置三者一致。"
        )

    device_text = "CPU" if resolved_device == "cpu" else f"GPU {resolved_device}"
    printer("\n开始 YOLO 训练。")
    printer(f"模型：{options.model}")
    printer(f"任务：{selected_task}")
    printer(f"设备：{device_text}")
    printer(f"数据集：{Path(options.data_yaml).expanduser().resolve()}")
    printer(
        "训练参数："
        f"epochs={options.epochs}, imgsz={options.image_size}, batch={options.batch_size}, "
        f"patience={options.patience}"
    )

    # 这些关键字参数与 config.py 一一对应，便于初学者从配置追到训练行为。
    training_result = model.train(
        data=str(Path(options.data_yaml).expanduser().resolve()),
        epochs=options.epochs,
        imgsz=options.image_size,
        batch=options.batch_size,
        patience=options.patience,
        device=resolved_device,
        workers=options.workers,
        cache=options.cache,
        amp=options.amp,
        seed=options.seed,
        project=str(Path(options.output_dir).expanduser().resolve()),
        name=options.run_name,
        exist_ok=options.exist_ok,
        save=True,
        save_period=options.save_period,
        plots=True,
    )
    save_dir = _save_dir_from_training(model, training_result, options)
    printer("\nYOLO 训练结束。")
    printer(f"结果目录：{save_dir}")
    printer(f"最佳权重：{save_dir / 'weights' / 'best.pt'}")
    printer(f"最后权重：{save_dir / 'weights' / 'last.pt'}")
    return save_dir


__all__ = [
    "DatasetTaskInfo",
    "LABEL_COLUMNS_TO_TASK",
    "SUPPORTED_TASKS",
    "TrainingConfigurationError",
    "TrainingOptions",
    "TrainingRuntimeError",
    "inspect_dataset_task",
    "resolve_training_device",
    "train_yolo",
]
