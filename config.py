"""VisionStudyProject 的集中配置。

参数按照“共享路径 -> 普通相机 -> 数据集 -> YOLO -> Kinect -> HTTP -> MySQL”
排列。一个参数如果被多个入口共用，只在这里保留一份，并在注释中标明使用它的脚本。
日常切换相机、模型、训练任务或服务地址时，只需修改本文件，不要改入口脚本。
"""

import os
from pathlib import Path


# =============================================================================
# 0. 项目与数据路径（scripts/01、02、03 共用）
# =============================================================================

# 当前项目根目录；根据 config.py 所在位置自动计算，通常不需要手动修改。
PROJECT_ROOT = Path(__file__).resolve().parent

# 项目全部数据的根目录，原始数据和构建后的数据集都放在这里。
DATA_DIR = PROJECT_ROOT / "data"

# 原始图片、同名 YOLO 标签和 classes.txt 的目录；步骤 01 写入，步骤 02 读取。
RAW_LABELED_DIR = DATA_DIR / "raw_labeled"

# 构建后数据集的根目录；步骤 02 会在这里维护稳定入口和历史版本。
DATASETS_DIR = DATA_DIR / "datasets"

# 每次成功构建的数据集版本目录，保留该名称供项目其他模块统一引用。
DATASET_VERSIONS_DIR = DATASETS_DIR / "versions"

# 最新一次成功构建的数据集配置；步骤 02 更新，步骤 03 从这里开始训练。
DATA_YAML_PATH = DATASETS_DIR / "data.yaml"


# =============================================================================
# 1. 普通 USB 相机（scripts/01_capture_images.py、04_realtime_inference.py）
# =============================================================================

# Windows 摄像头编号；0 是第一台，只打开此编号，不会自动扫描或切换设备。
CAMERA_ID = 0

# 向普通相机请求的彩色画面宽度（像素）；实际宽度取决于相机支持能力。
FRAME_WIDTH = 1280

# 向普通相机请求的彩色画面高度（像素）；实际高度取决于相机支持能力。
FRAME_HEIGHT = 720

# 步骤 01 本次最多保存的图片数量。
CAPTURE_IMAGE_COUNT = 1

# 步骤 01 连续保存两张图片之间至少等待的秒数；0 表示不额外等待。
CAPTURE_INTERVAL_SECONDS = 0.5


# =============================================================================
# 2. YOLO 数据集构建（scripts/02_build_yolo_dataset.py）
# =============================================================================

# 原始样本中分配给训练集的比例。
TRAIN_RATIO = 0.7

# 原始样本中分配给验证集的比例。
VAL_RATIO = 0.2

# 原始样本中分配给测试集的比例；三项比例相加必须等于 1.0。
TEST_RATIO = 0.1

# 拆分数据集时的随机种子；保持不变可以复现同一组拆分结果。
RANDOM_SEED = 42


# =============================================================================
# 3. YOLO 训练与推理共享参数
#    YOLO_TASK：scripts/03、05 共用
#    TRAIN_IMAGE_SIZE：scripts/03、04、05、06、07 共用
# =============================================================================

# YOLO 任务类型："detect" 是普通水平框，"obb" 是旋转框。
# 当前值用于 OBB 推理；用现有 5 列零件数据运行步骤 03 前必须改回 "detect"。
YOLO_TASK = "obb"

# YOLO 训练和推理的输入图像尺寸（像素）；值越大通常越占显存、处理越慢。
# 变量名为兼容已有入口保留为 TRAIN_IMAGE_SIZE，推理步骤也复用这个值。
TRAIN_IMAGE_SIZE = 640


# =============================================================================
# 4. YOLO 模型训练（scripts/03_train_yolo.py）
# =============================================================================

# 训练使用的初始模型名称或权重路径，任务类型必须与 YOLO_TASK 和标签格式一致。
# Detect 示例：yolo11n.pt；OBB 示例：yolo11n-obb.pt。
YOLO_MODEL = "yolo11n.pt"

# 训练设备："gpu" 使用 NVIDIA GPU，"cpu" 明确只使用 CPU。
TRAIN_DEVICE = "gpu"

# TRAIN_DEVICE="gpu" 时使用的 GPU 编号；0 表示第一张 NVIDIA GPU。
TRAIN_GPU_INDEX = 0

# 最多训练多少个 epoch。
TRAIN_EPOCHS = 200

# 每次训练送入模型的图片数量；显存不足时应优先减小此值。
TRAIN_BATCH_SIZE = 8

# 验证指标连续多少个 epoch 没有提升后提前停止训练。
TRAIN_PATIENCE = 50

# 数据加载使用的工作进程数量；Windows 下过大可能增加启动开销。
TRAIN_WORKERS = 4

# 训练数据缓存方式：False 不缓存，也可设置为 True、"ram" 或 "disk"。
TRAIN_CACHE = False

# 是否启用自动混合精度；GPU 训练通常保持 True 以降低显存占用。
TRAIN_AMP = True

# Ultralytics 训练过程使用的随机种子。
TRAIN_SEED = 42

# 所有训练结果的父目录。
TRAIN_OUTPUT_DIR = PROJECT_ROOT / "runs" / "train"

# 本次训练的结果目录名称；应根据实际任务修改，例如 yolo11n_detect 或 yolo11n_obb。
TRAIN_RUN_NAME = "yolo11n_detect"

# 是否允许直接复用同名结果目录；False 时 Ultralytics 会使用新的递增目录。
TRAIN_EXIST_OK = False

# 每隔多少个 epoch 额外保存一次检查点；-1 表示不定期额外保存。
TRAIN_SAVE_PERIOD = 10


# =============================================================================
# 5. YOLO 实时推理（scripts/04、05、06、07 共用）
# =============================================================================

# 推理时加载的权重文件；当前指向本地收据 OBB 示例模型。
INFERENCE_MODEL_PATH = PROJECT_ROOT / "models" / "notepay_yolov8s_obb_receipt.pt"

# Ultralytics 推理设备；0 表示第一张 GPU，"cpu" 表示只使用 CPU。
INFERENCE_DEVICE = 0

# 最低检测置信度；低于此值的预测不会进入后续显示或深度融合。
INFERENCE_CONFIDENCE_THRESHOLD = 0.65


# =============================================================================
# 6. Kinect v2 与深度融合（scripts/05、06、07 共用）
# =============================================================================

# Kinect for Windows SDK 2.0 的 Microsoft.Kinect.dll 路径。
# 彩色/深度对齐和相机 XYZ 坐标由 SDK CoordinateMapper 计算，无需手填相机内参。
KINECT_SDK_ASSEMBLY_PATH = Path(
    r"C:\Program Files\Microsoft SDKs\Kinect\v2.0_1409\Assemblies\Microsoft.Kinect.dll"
)

# 打开 Kinect 后等待设备进入可用状态的最长秒数。
KINECT_AVAILABILITY_TIMEOUT_SECONDS = 5.0

# 每次读取一组同步彩色帧和深度帧的最长等待秒数。
KINECT_FRAME_TIMEOUT_SECONDS = 2.0

# 在检测框中心取样区域的宽、高比例；0.30 表示各取检测框尺寸的 30%。
DEPTH_ROI_RATIO = 0.30

# 允许参与深度融合的最小距离（毫米）。
DEPTH_MIN_MM = 500

# 允许参与深度融合的最大距离（毫米）。
DEPTH_MAX_MM = 4500

# 至少需要多少个有效映射点才返回可靠的距离和相机 X/Y/Z。
DEPTH_MIN_VALID_SAMPLES = 5


# =============================================================================
# 7. 机器人视觉 HTTP 地址（scripts/07、10 共用）
# =============================================================================

# HTTP 服务监听地址；127.0.0.1 只允许本机访问，局域网访问时改为服务电脑的明确地址。
VISION_HTTP_HOST = "127.0.0.1"

# HTTP 服务监听端口，模拟机器人客户端也会连接这个端口。
VISION_HTTP_PORT = 8008


# =============================================================================
# 8. MySQL 持久化（scripts/08 和 vision_http_server.py 共用）
# =============================================================================

# MySQL 配置在 Python 启动时从环境变量读取。密码不写入本文件，也不要打印到日志。

# MySQL 主机地址；未设置 VISION_MYSQL_HOST 时默认连接本机。
MYSQL_HOST = os.environ.get("VISION_MYSQL_HOST", "127.0.0.1")

# MySQL TCP 端口；以字符串读取，数据库模块会再校验并转换为整数。
MYSQL_PORT = os.environ.get("VISION_MYSQL_PORT", "3306")

# MySQL 登录用户名；必须在启动 Python 前设置 VISION_MYSQL_USER。
MYSQL_USER = os.environ.get("VISION_MYSQL_USER", "")

# MySQL 登录密码；只从 VISION_MYSQL_PASSWORD 读取，不提供硬编码默认密码。
MYSQL_PASSWORD = os.environ.get("VISION_MYSQL_PASSWORD", "")

# 已经存在的目标数据库名称；初始化脚本只建表，不负责创建数据库。
MYSQL_DATABASE = os.environ.get("VISION_MYSQL_DATABASE", "")

# 数据库连接字符集；固定使用 utf8mb4 才能完整保存中文类别和 OBB JSON。
MYSQL_CHARSET = os.environ.get("VISION_MYSQL_CHARSET", "utf8mb4")
