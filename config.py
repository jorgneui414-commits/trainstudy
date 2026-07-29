"""新项目的集中配置。

通常只需要修改本文件中的参数，不需要改动采集或数据集构建逻辑。
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

# 普通 USB 相机参数
# 0 代表系统识别到的第一台摄像头；多台摄像头时可尝试改为 1、2 等。
CAMERA_ID = 0
# 这是向摄像头请求的分辨率，部分设备可能按自身支持的最接近尺寸返回画面。
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
# 本次最多保存多少张图，以及两次自动保存之间至少间隔多少秒。
CAPTURE_IMAGE_COUNT = 1
CAPTURE_INTERVAL_SECONDS = 0.5

# 数据路径
DATA_DIR = PROJECT_ROOT / "data"
RAW_LABELED_DIR = DATA_DIR / "raw_labeled"
DATASETS_DIR = DATA_DIR / "datasets"
DATASET_VERSIONS_DIR = DATASETS_DIR / "versions"
DATA_YAML_PATH = DATASETS_DIR / "data.yaml"

# YOLO 数据集拆分参数
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1
RANDOM_SEED = 42
