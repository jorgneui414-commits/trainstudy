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

# YOLO 训练参数（步骤 03）
# 任务只能选 "detect"（普通水平框）或 "obb"（旋转框）。当前数据集是 5 列普通框，
# 因此第一次训练请保持为 "detect"；OBB 需要先重新标注为 9 列四角点格式。
YOLO_TASK = "detect"
# Detect 常用模型：yolo11n.pt、yolo26n.pt、yolov8n.pt。
# OBB 常用模型：yolo11n-obb.pt、yolo26n-obb.pt、yolov8n-obb.pt。
# n/s/m/l/x 依次通常代表从轻量到大型；模型越大，显存占用和训练时间通常越高。
YOLO_MODEL = "yolo11n.pt"

# "gpu" 使用下方编号的 NVIDIA GPU；改成 "cpu" 后会明确使用 CPU，不会访问显卡。
TRAIN_DEVICE = "gpu"
TRAIN_GPU_INDEX = 0

# 精度优先的首轮训练设置。6GB 显存的 RTX 3060 使用 YOLO11n 时，batch=8 通常较稳妥；
# 如果改用更大的模型或提高图片尺寸后显存不足，请优先减小 TRAIN_BATCH_SIZE。
TRAIN_EPOCHS = 200
TRAIN_IMAGE_SIZE = 640
TRAIN_BATCH_SIZE = 8
# 验证集连续 50 轮没有提升时提前停止，避免无效训练和明显过拟合。
TRAIN_PATIENCE = 50
TRAIN_WORKERS = 4
# 原图约 2GB，默认不缓存到内存；可按机器内存改为 "disk" 或 "ram"。
TRAIN_CACHE = False
# GPU 训练建议保持自动混合精度，以减少显存占用并提高速度；CPU 训练也可保持 True。
TRAIN_AMP = True
TRAIN_SEED = 42

# 每次训练会保存到 runs/train/<TRAIN_RUN_NAME>；同名已存在时自动增加编号，不覆盖旧结果。
TRAIN_OUTPUT_DIR = PROJECT_ROOT / "runs" / "train"
TRAIN_RUN_NAME = "yolo11n_detect"
TRAIN_EXIST_OK = False
# 每 10 个 epoch 额外保存一次检查点；设为 -1 可只保留 best.pt 和 last.pt。
TRAIN_SAVE_PERIOD = 10

# 实时推理参数（步骤 04）
# 每次训练后请确认这里指向本次要验证的 best.pt；当前为已完成的 yolo11n_detect 训练结果。
INFERENCE_MODEL_PATH = TRAIN_OUTPUT_DIR / "yolo11n_detect" / "weights" / "best.pt"
# Ultralytics 中 0 表示第一张 NVIDIA GPU；如需只用 CPU，可改为 "cpu"。
INFERENCE_DEVICE = 0
# 低于该置信度的预测不会显示在实时画面中。
INFERENCE_CONFIDENCE_THRESHOLD = 0.25

# Kinect v2 深度实时推理参数（步骤 05）
# 第五步直接复用上面的 YOLO_TASK、INFERENCE_MODEL_PATH、INFERENCE_DEVICE、
# INFERENCE_CONFIDENCE_THRESHOLD 和 TRAIN_IMAGE_SIZE，避免为同一含义重复设置参数。
# YOLO_TASK 决定解析普通水平框（detect）还是旋转框（obb），模型路径必须与任务匹配。
# 使用 Kinect for Windows SDK 2.0 自带的 Microsoft.Kinect.dll；无需填写相机内参，
# 彩色/深度对齐和相机 XYZ 坐标都由 SDK CoordinateMapper 计算。
KINECT_SDK_ASSEMBLY_PATH = Path(
    r"C:\Program Files\Microsoft SDKs\Kinect\v2.0_1409\Assemblies\Microsoft.Kinect.dll"
)
# 打开设备后最多等待多少秒，确认 Kinect 已经进入可用状态。
KINECT_AVAILABILITY_TIMEOUT_SECONDS = 5.0
# 每次读取同步彩色+深度帧时的最长等待时间，不是整段程序的运行时长。
KINECT_FRAME_TIMEOUT_SECONDS = 2.0

# Kinect v2 深度原始单位为毫米。检测框中央 30% 区域内只保留 500~4500 mm，
# 至少 5 个有效映射点才返回深度；相机 X/Y/Z 在读取后也统一转换为毫米。
# 0.30 表示只采样检测框中心宽度和高度各 30% 的区域，减少背景像素干扰。
DEPTH_ROI_RATIO = 0.30
# 超出这个可靠量程的像素不会参与三维坐标计算。
DEPTH_MIN_MM = 500
DEPTH_MAX_MM = 4500
# 有效点少于这个数量时仍保留二维检测框，但三维坐标会标记为无效。
DEPTH_MIN_VALID_SAMPLES = 5
