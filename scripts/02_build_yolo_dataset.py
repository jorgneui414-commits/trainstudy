"""按 config.py 的设置构建一个带版本号的 YOLO 数据集。"""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from yolo_dataset_builder import build_yolo_dataset


def main() -> None:
    data_yaml = build_yolo_dataset(
        raw_dir=config.RAW_LABELED_DIR,
        dataset_dir=config.DATASETS_DIR,
        train_ratio=config.TRAIN_RATIO,
        val_ratio=config.VAL_RATIO,
        test_ratio=config.TEST_RATIO,
        random_seed=config.RANDOM_SEED,
    )

    version_dir = data_yaml.parent
    print("\nYOLO 数据集构建完成。")
    print(f"版本目录：{version_dir}")
    print(f"配置文件：{data_yaml}")
    print(f"稳定入口：{config.DATA_YAML_PATH}")


if __name__ == "__main__":
    main()

