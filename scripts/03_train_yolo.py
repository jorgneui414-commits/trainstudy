"""按 config.py 的设置训练 YOLO Detect 或 OBB 模型。"""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 直接运行 scripts/03_train_yolo.py 时，Python 默认只搜索 scripts 目录；
# 把项目根目录加入搜索路径后，才能导入同级的 config.py 和 yolo_trainer.py。
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from yolo_trainer import TrainingOptions, train_yolo


def main() -> None:
    # 所有经常调整的参数均放在 config.py；此入口只负责把配置传给训练模块。
    options = TrainingOptions(
        data_yaml=config.DATA_YAML_PATH,
        task=config.YOLO_TASK,
        model=config.YOLO_MODEL,
        device=config.TRAIN_DEVICE,
        gpu_index=config.TRAIN_GPU_INDEX,
        epochs=config.TRAIN_EPOCHS,
        image_size=config.TRAIN_IMAGE_SIZE,
        batch_size=config.TRAIN_BATCH_SIZE,
        patience=config.TRAIN_PATIENCE,
        workers=config.TRAIN_WORKERS,
        cache=config.TRAIN_CACHE,
        amp=config.TRAIN_AMP,
        seed=config.TRAIN_SEED,
        output_dir=config.TRAIN_OUTPUT_DIR,
        run_name=config.TRAIN_RUN_NAME,
        exist_ok=config.TRAIN_EXIST_OK,
        save_period=config.TRAIN_SAVE_PERIOD,
    )
    train_yolo(options)


if __name__ == "__main__":
    # 只有直接执行本文件时才开始训练；被其他文件导入时不会自动训练。
    main()
