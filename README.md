# VisionStudyProject

VisionStudyProject 是一个独立的 Windows 视觉项目，覆盖从图片采集、YOLO 数据集构建和模型训练，到 Kinect v2 RGB-D 深度定位、HTTP/JSON 服务和 MySQL 持久化的完整流程。

同级目录中的 `VisionTeachDemo` 只作为阅读参考；本项目不导入、不复制，也不修改其中的代码和数据。

这个项目有两条主线：

- 数据与模型主线：普通 USB 相机采集图片 → 人工标注 → 构建版本化 YOLO 数据集 → 训练 Detect 或 OBB 模型 → 实时推理。
- Kinect 服务主线：接收 JSON 命令 → 获取一组 Kinect 彩色/深度同步帧 → 执行 OBB 推理和深度融合 → 尝试写入 MySQL → 返回同一组 JSON 检测结果。

MySQL 只负责保存结果，不是机器人与视觉服务之间的通信通道。模拟机器人客户端也只收发 JSON，不包含任何机器人运动控制。

## 1. 项目功能

### 1.1 图片、数据集和模型

- 使用 OpenCV 打开 `config.CAMERA_ID` 指定的普通 UVC USB 相机并采集编号图片。
- 校验 `classes.txt`、图片与同名标签是否完整配对。
- 同时支持 YOLO Detect 5 列标签和 OBB 9 列标签，但同一数据集禁止混用。
- 按固定随机种子拆分 train、val、test，并保留每次成功构建的数据集版本。
- 在训练前检查数据集任务、预训练模型任务和 `YOLO_TASK` 是否一致。
- 可以明确选择 CPU 或 NVIDIA GPU；GPU 不可用时不会静默回退到 CPU。

### 1.2 实时视觉与 Kinect 深度定位

- 普通 USB 相机可以加载 `INFERENCE_MODEL_PATH` 指定的权重进行实时推理。
- Kinect v2 使用官方 Kinect for Windows SDK 2.0 同步读取彩色帧和深度帧。
- SDK `CoordinateMapper` 负责深度到彩色图的对齐，以及 Kinect 相机坐标 X/Y/Z 的计算。
- 连续预览入口同时支持 Detect 和 OBB；单次检测、HTTP 服务只接受 OBB 模型。
- 深度、距离和相机坐标统一使用毫米，二维坐标仍使用彩色图像素。
- OBB 方向角取旋转框长轴：向右为 0°，沿图像顺时针增加，范围为 `[0, 180)`。

### 1.3 HTTP、MySQL 和模拟机器人客户端

- `GET /health` 提供健康检查，不会打开 Kinect。
- `POST /vision/command` 同步执行一次 Kinect OBB 深度检测。
- 服务基于标准库单线程 `HTTPServer`，所有检测命令串行处理，不会并发读取 Kinect。
- 视觉成功后，服务用同一组字段尝试写入 MySQL。
- MySQL 写入失败不会丢失视觉结果：HTTP 仍返回 200，并用 `mysql_saved=false` 表示没有保存。
- 模拟客户端先检查服务，再生成 `robot_<UUID>` 命令编号并发送一次现有 JSON 命令。

完整在线数据流如下：

~~~mermaid
flowchart LR
    client["机器人或模拟客户端"] --> health["GET /health"]
    client --> command["POST /vision/command"]
    command --> server["单线程视觉服务"]
    server --> frame["一组 Kinect RGB-D 同步帧"]
    frame --> inference["OBB 推理"]
    inference --> depth["深度融合与毫米制相机坐标"]
    depth --> mysql["尝试写入 MySQL"]
    depth --> response["HTTP JSON 视觉结果"]
    mysql --> response
    response --> client
~~~

## 2. 运行入口一览

所有入口脚本都应从项目根目录运行。

| 入口 | 用途 | 需要的外部条件 | 主要输出 |
| --- | --- | --- | --- |
| `scripts/01_capture_images.py` | 普通相机采集图片 | UVC USB 相机 | `data/raw_labeled/` 中的 JPG |
| `scripts/02_build_yolo_dataset.py` | 校验并构建数据集 | 已完成的图片、标签、`classes.txt` | 版本目录和稳定 `data.yaml` |
| `scripts/03_train_yolo.py` | 训练 Detect 或 OBB | 匹配的标注、模型和设备配置 | `runs/train/.../weights/best.pt` |
| `scripts/04_realtime_inference.py` | 普通相机实时推理 | UVC 相机、推理权重 | 带检测结果和 FPS 的窗口 |
| `scripts/05_realtime_depth_inference.py` | Kinect Detect/OBB 连续预览 | Kinect v2、匹配任务的权重 | 彩色检测窗口和深度窗口 |
| `scripts/06_capture_and_detect.py` | 单次 Kinect OBB 检测 | Kinect v2、OBB 权重 | 一次 JSON 结果 |
| `scripts/07_vision_http_server.py` | 启动 Kinect OBB HTTP 服务 | OBB 权重；Kinect 在首条命令时使用 | `/health`、`/vision/command` |
| `scripts/08_init_mysql.py` | 创建 MySQL 表 | 已存在的数据库和有权限的账号 | `vision_request`、`vision_detection` |
| 第 09 步，无独立脚本 | 把 HTTP 成功结果接入 MySQL | 已集成在 `vision_http_server.py` | HTTP 响应中的 MySQL 状态 |
| `scripts/10_robot_client.py` | 模拟机器人发起一次命令 | 已启动的 HTTP 服务 | 终端中的 JSON 检测摘要 |

第 09 步没有 `09_*.py`，因为它不是一个单独运行的程序，而是“视觉成功后调用已有 MySQL 写入接口”的集成逻辑。

## 3. 项目边界

当前项目负责“视觉检测和结果交付”，不负责以下功能：

- 不发送机器人运动指令。
- 不选择、排序或只返回一个“最佳抓取目标”。
- 不把 Kinect 相机坐标直接转换为机器人坐标。
- 不包含手眼标定、抓取姿态求解或完整 6D 位姿。
- 不通过 HTTP 传输图片，也不保存运行时彩色图或深度图。
- 不自动扫描相机，不会在配置设备不可用时偷偷切换到其他设备。
- 不提供多线程 Kinect 并发读取。

如果后续要让机器人使用 `camera_x_mm`、`camera_y_mm`、`camera_z_mm` 执行运动，必须先完成相机与机器人之间的标定和坐标变换；当前坐标只能解释为 Kinect 相机坐标系中的位置。

## 4. 运行前准备

### 4.1 进入项目并固定解释器

下面命令都在 PowerShell 中执行。每次打开新终端，先运行：

~~~powershell
Set-Location E:\trainstudy_demo\VisionStudyProject

$PythonCpu = "E:\anaconda\envs\part_yolo_cpu\python.exe"
$PythonGpu = "E:\anaconda\envs\part_yolo_gpu\python.exe"
~~~

`&` 是 PowerShell 的调用运算符，用来执行变量中保存的程序路径。后续命令使用完整解释器路径，避免误用系统 Python、错误的 Conda 环境或失效的 `pip` 启动器。

当前项目使用 64 位 Python 3.10。安装项目依赖：

~~~powershell
& $PythonCpu -m pip install -r requirements.txt
& $PythonGpu -m pip install -r requirements.txt
~~~

主要依赖包括 NumPy、OpenCV、PyYAML、Ultralytics、pythonnet 和 PyMySQL。Kinect 入口必须使用安装了 `pythonnet==3.0.5` 的 64 位环境。

### 4.2 首次创建 GPU 环境

如果本机还没有 `part_yolo_gpu`，可以从现有 CPU 环境克隆，再安装项目使用的 CUDA 版 PyTorch：

~~~powershell
& "E:\anaconda\Scripts\conda.exe" create --name part_yolo_gpu --clone part_yolo_cpu -y

& $PythonGpu -m pip install --force-reinstall --no-deps `
  torch==2.13.0 torchvision==0.28.0 `
  --index-url https://download.pytorch.org/whl/cu130

& $PythonGpu -m pip install -r requirements.txt
~~~

这个环境不需要另外安装完整 CUDA Toolkit；PyTorch wheel 已包含运行时组件。确认当前解释器和 CUDA 状态：

~~~powershell
& $PythonGpu -c "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA GPU')"
~~~

只有 `torch.cuda.is_available()` 为 `True` 才表示当前 Python 真正能使用 GPU。仅在 `config.py` 中填写 `INFERENCE_DEVICE = 0` 或 `TRAIN_DEVICE = "gpu"` 并不能让 CPU 版 PyTorch 自动获得 CUDA。

### 4.3 硬件和服务依赖

按准备运行的功能选择对应条件：

- 图片采集、普通实时推理：一台 Windows 能识别的 UVC USB 相机。
- Kinect 功能：Xbox One Kinect / Kinect v2、供电适配器、USB 3.0、Kinect for Windows Runtime 2.0 和 SDK 2.0。
- Kinect SDK 程序集：默认路径为 `C:\Program Files\Microsoft SDKs\Kinect\v2.0_1409\Assemblies\Microsoft.Kinect.dll`。
- GPU 推理或训练：CUDA 可用的 NVIDIA GPU 环境；也可以按配置明确使用 CPU。
- MySQL 持久化：已经创建好的 MySQL 数据库、账号，以及该账号对目标数据库的建表和写入权限。

Kinect Studio 可以用于检查设备，但运行 Python 前必须在 Kinect Studio 左上角断开 Kinect，并关闭会占用相机的窗口。Python 进程和 Kinect Studio 不能同时独占设备。

## 5. 当前数据、模型和配置关系

### 5.1 本工作副本中的两类模型

项目当前同时存在两条不同用途的模型路径，不能混为一谈：

1. 零件 Detect 模型

   - 原始数据位于 `data/raw_labeled/`，标签是 5 列 Detect 格式。
   - 类别顺序为：螺丝刀、扳手、支架、直波天线、计算模块。
   - 已构建的数据集稳定入口是 `data/datasets/data.yaml`。
   - 对应权重位于 `runs/train/yolo11n_detect/weights/best.pt`。
   - 该权重可用于普通相机推理和 Kinect Detect 连续预览，但不能用于第 06、07 步的 OBB 接口。

2. 收据 OBB 示例模型

   - 文件位于 `models/notepay_yolov8s_obb_receipt.pt`。
   - 当前 `config.INFERENCE_MODEL_PATH` 指向该模型，`YOLO_TASK` 为 `"obb"`。
   - 它用于运行 OBB 四角点、方向角和 Kinect 深度调用链，类别来自权重自身的收据类别。
   - 它不是本项目五类零件的 OBB 模型，不能代替真实零件 OBB 标注和训练。
   - 模型来源、SHA-256 和许可证说明见 [models/README.md](models/README.md)。

因此，当前 `config.py` 适合运行 OBB 推理入口，但不能直接拿当前 5 列 Detect 数据执行第 03 步。重新训练零件 Detect 前，必须先把 `YOLO_TASK`、`YOLO_MODEL` 和训练名称切回下一节给出的 Detect 组合。

`data/raw_labeled/`、`data/datasets/`、`runs/` 和 `*.pt` 默认被 Git 忽略。换电脑或重新克隆仓库时，需要自行准备原始数据、重新构建数据集，并复制或重新训练权重。

### 5.2 必须保持一致的三件事

开始训练或 Kinect 连续预览前，下面三者必须表达同一种任务：

- 数据集标签：Detect 每行 5 列；OBB 每行 9 列。
- `YOLO_TASK`：`"detect"` 或 `"obb"`。
- 模型类型：普通 `yolo11n.pt` 属于 Detect；`yolo11n-obb.pt` 和 OBB 权重属于 OBB。

项目不会把 Detect 水平框自动伪装成 OBB，也不会在任务不匹配时继续运行。

### 5.3 常用配置组合

用当前五类零件 Detect 数据重新训练时：

~~~python
YOLO_TASK = "detect"
YOLO_MODEL = "yolo11n.pt"
TRAIN_RUN_NAME = "yolo11n_detect"
~~~

训练完成后，用零件 Detect 权重运行普通相机或 Kinect 连续预览：

~~~python
YOLO_TASK = "detect"
INFERENCE_MODEL_PATH = TRAIN_OUTPUT_DIR / "yolo11n_detect" / "weights" / "best.pt"
~~~

使用当前收据 OBB 示例权重运行 OBB 入口时：

~~~python
YOLO_TASK = "obb"
INFERENCE_MODEL_PATH = PROJECT_ROOT / "models" / "notepay_yolov8s_obb_receipt.pt"
INFERENCE_CONFIDENCE_THRESHOLD = 0.65
~~~

训练真实零件 OBB 模型时，必须先把数据重新标注成 9 列 OBB 格式并重新构建数据集，然后使用：

~~~python
YOLO_TASK = "obb"
YOLO_MODEL = "yolo11n-obb.pt"
TRAIN_RUN_NAME = "yolo11n_obb"
~~~

`YOLO_TASK` 同时被训练入口和 Kinect 连续预览入口使用。切换用途后，应先重新检查 `config.py`，不要只修改模型路径。

### 5.4 其他常用配置

| 配置 | 含义 |
| --- | --- |
| `CAMERA_ID` | 普通 USB 相机编号；项目只打开这个编号 |
| `FRAME_WIDTH`、`FRAME_HEIGHT` | 向普通相机请求的分辨率 |
| `CAPTURE_IMAGE_COUNT` | 本次最多采集的图片数量 |
| `CAPTURE_INTERVAL_SECONDS` | 自动保存两张图片之间的最小间隔 |
| `TRAIN_RATIO`、`VAL_RATIO`、`TEST_RATIO` | 数据集拆分比例，默认 0.7、0.2、0.1 |
| `RANDOM_SEED` | 数据集随机拆分种子，默认 42 |
| `TRAIN_DEVICE`、`TRAIN_GPU_INDEX` | 训练使用 CPU 或指定 GPU |
| `INFERENCE_DEVICE` | 推理设备；`0` 表示第 1 张 GPU，`"cpu"` 表示 CPU |
| `INFERENCE_CONFIDENCE_THRESHOLD` | 推理置信度阈值 |
| `KINECT_AVAILABILITY_TIMEOUT_SECONDS` | 打开 Kinect 后等待设备可用的时间 |
| `KINECT_FRAME_TIMEOUT_SECONDS` | 单次同步 RGB-D 取帧的最长等待时间 |
| `DEPTH_ROI_RATIO` | 在检测框中央区域采样深度，默认 0.30 |
| `DEPTH_MIN_MM`、`DEPTH_MAX_MM` | 参与融合的有效深度范围，默认 500～4500 mm |
| `DEPTH_MIN_VALID_SAMPLES` | 返回可靠三维坐标所需的最少映射点，默认 5 |
| `VISION_HTTP_HOST`、`VISION_HTTP_PORT` | HTTP 监听地址，默认 `127.0.0.1:8008` |

## 6. 从采集到训练

### 6.1 采集普通相机图片

先在 `config.py` 中设置相机编号、分辨率、数量和间隔，然后运行：

~~~powershell
& $PythonCpu scripts\01_capture_images.py
~~~

程序只打开 `CAMERA_ID` 指定的相机。图片保存到 `data/raw_labeled/`，名称按 `0001.jpg`、`0002.jpg` 递增；已有编号不会被覆盖。第一张立即保存，后续按配置间隔保存。按 `q` 或 `Esc` 提前结束。

这个入口只负责普通 UVC 图片采集，不读取 Kinect 深度。

### 6.2 准备 YOLO 标注

原始目录采用平铺结构，每张图片必须有一个同名标签：

~~~text
data/raw_labeled/
├── classes.txt
├── 0001.jpg
├── 0001.txt
├── 0002.bmp
└── 0002.txt
~~~

`classes.txt` 使用 UTF-8 编码，每行一个类别；行号从 0 开始，对应标签中的 `class_id`。

Detect 标签每行 5 列：

~~~text
class_id x_center y_center width height
~~~

OBB 标签每行 9 列：

~~~text
class_id x1 y1 x2 y2 x3 y3 x4 y4
~~~

所有坐标必须归一化到 `[0, 1]`。构建器会拒绝以下情况：

- 图片缺少同名标签，或标签没有同名图片。
- 同一 stem 对应多种图片扩展名。
- `classes.txt` 缺失、为空或包含重复类别。
- 类别编号越界。
- 坐标不是数字、包含 NaN/无穷值或超出 `[0, 1]`。
- Detect 和 OBB 标签混用。

空标签文件可以表示没有目标的负样本，但整个数据集不能全部为空。默认拆分比例要求 train、val、test 都非空，因此至少需要 5 组图片和标签。

### 6.3 构建版本化数据集

~~~powershell
& $PythonCpu scripts\02_build_yolo_dataset.py
~~~

成功后结构类似：

~~~text
data/datasets/
├── data.yaml
└── versions/
    └── 20260804_120000/
        ├── classes.txt
        ├── data.yaml
        ├── images/{train,val,test}/
        └── labels/{train,val,test}/
~~~

每次构建都会创建新版本；旧版本不会被覆盖。只有新版本全部完成后，稳定入口 `data/datasets/data.yaml` 才会更新。

生成的 `data.yaml` 使用当前版本目录的绝对路径。如果项目整体移动到了其他磁盘或目录，应重新运行第 02 步，不要继续使用指向旧位置的 YAML。

### 6.4 训练 YOLO

先按第 5 节让标签、`YOLO_TASK` 和 `YOLO_MODEL` 保持一致，再运行：

~~~powershell
& $PythonGpu scripts\03_train_yolo.py
~~~

如需明确用 CPU：

~~~python
TRAIN_DEVICE = "cpu"
~~~

训练入口会先检查数据和设备，再加载预训练模型。首次使用某个 Ultralytics 模型名称时可能需要联网下载权重。训练输出保存在 `runs/train/<TRAIN_RUN_NAME>/`；常用文件是：

- `weights/best.pt`：训练过程中验证指标最好的权重。
- `weights/last.pt`：最后一个 epoch 的权重。

要运行后续推理，必须把 `INFERENCE_MODEL_PATH` 改为实际要使用的权重。

## 7. 运行实时推理和 Kinect 单次检测

### 7.1 普通 USB 相机实时推理

配置 `CAMERA_ID`、`INFERENCE_MODEL_PATH`、`INFERENCE_DEVICE` 和置信度后运行：

~~~powershell
& $PythonGpu scripts\04_realtime_inference.py
~~~

程序会先确认配置相机能够读到首帧，再加载模型。窗口显示模型绘制结果和 FPS，按 `q` 或 `Esc` 退出。该入口不保存图片或视频。

如果当前使用收据 OBB 示例模型，可以让普通相机拍摄收据；如果使用零件 Detect 权重，则应拍摄对应零件。`part_type` 始终来自权重自身的 `model.names`。

### 7.2 Kinect v2 运行前检查

运行第 05、06、07 步前确认：

- `KINECT_SDK_ASSEMBLY_PATH` 指向真实存在的 `Microsoft.Kinect.dll`。
- 当前 Python 是 64 位 3.10，并已安装 `pythonnet==3.0.5`。
- Kinect Studio 已与设备断开，其他相机程序也已关闭。
- `INFERENCE_MODEL_PATH` 指向存在的权重。
- 第 05 步的 `YOLO_TASK` 与权重任务一致。
- 第 06、07 步使用的权重必须是 OBB。

YOLO 只对 Kinect 彩色帧推理，深度图不参与模型训练。每个深度像素由 SDK 映射到彩色图和 Kinect 相机坐标，再在检测框中央区域取有效点的中位数。

### 7.3 Kinect Detect/OBB 连续预览

~~~powershell
& $PythonGpu scripts\05_realtime_depth_inference.py
~~~

入口持续执行：

1. 获取一组同步彩色帧和深度帧。
2. 按 `YOLO_TASK` 解析 Detect 或 OBB 结果。
3. 在每个框中央 `DEPTH_ROI_RATIO` 区域筛选有效深度映射点。
4. 计算 `distance_mm` 和相机 `X/Y/Z` 中位数。
5. 显示彩色检测窗口和深度伪彩色窗口。

绿色框表示三维深度有效，橙色框表示只有二维检测结果。深度无效时，第 05 步仍保留并显示二维目标。按 `q` 或 `Esc` 释放 Kinect 并退出。

### 7.4 单次 Kinect OBB 深度检测

~~~powershell
& $PythonGpu scripts\06_capture_and_detect.py
~~~

这个入口加载一次 OBB 模型、打开一次 Kinect、读取一组 RGB-D 帧、执行一次推理，并把 JSON 打印到终端，随后释放设备。

内部 `capture_and_detect(camera, model)` 的边界是：

- 相机和模型由调用方创建并持有；函数不会重新打开相机或重新加载模型。
- 每次调用恰好执行一次 `camera.read()` 和一次 `model.predict()`。
- 固定按 OBB 解析。
- 返回全部 `depth_valid=true` 的目标，不排序，也不选择单个目标。
- 没有检测或没有可靠深度时，正常返回空 `detections`。
- 不启动 HTTP、不连接 MySQL、不显示窗口、不保存图片。

## 8. 启动 HTTP、MySQL 和模拟机器人客户端

### 8.1 先选择运行模式

| 模式 | 做法 | 成功响应中的 MySQL 字段 |
| --- | --- | --- |
| 仅视觉服务 | 不设置 MySQL 账号，直接启动第 07 步 | 服务仍会尝试保存，但返回 `mysql_saved=false` |
| 视觉服务 + MySQL | 启动前设置全部 `VISION_MYSQL_*` 环境变量 | 保存成功时返回 `mysql_saved=true` 和请求编号 |

当前没有单独的“禁用 MySQL”开关。视觉结果构建成功后，服务总会尝试一次数据库写入；数据库失败只影响 MySQL 状态，不会把已经得到的视觉结果改成 HTTP 失败。

### 8.2 仅启动视觉服务

在终端 A 中：

~~~powershell
Set-Location E:\trainstudy_demo\VisionStudyProject
$PythonGpu = "E:\anaconda\envs\part_yolo_gpu\python.exe"

& $PythonGpu scripts\07_vision_http_server.py
~~~

默认监听 `http://127.0.0.1:8008`。模型在服务启动时加载一次；Kinect 在第一条合法检测命令到达时才打开。按 `Ctrl+C` 退出时会释放 Kinect 并关闭 HTTP 端口。

### 8.3 配置 MySQL 并启动完整服务

第 08 步只在“已经存在的数据库”中创建表，不创建数据库或账号。先让管理员准备数据库和账号，并授予建表和写入权限。

在终端 A 中设置非密码变量；尖括号内容必须替换为真实值：

~~~powershell
Set-Location E:\trainstudy_demo\VisionStudyProject
$PythonGpu = "E:\anaconda\envs\part_yolo_gpu\python.exe"

$env:VISION_MYSQL_HOST = "127.0.0.1"
$env:VISION_MYSQL_PORT = "3306"
$env:VISION_MYSQL_USER = "<MySQL用户名>"
$env:VISION_MYSQL_DATABASE = "<已存在的数据库名>"
$env:VISION_MYSQL_CHARSET = "utf8mb4"
~~~

使用隐藏输入读取密码，避免把明文密码写进 `config.py`、README 或 PowerShell 命令历史：

~~~powershell
$SecurePassword = Read-Host "请输入 MySQL 密码" -AsSecureString
$PasswordPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($SecurePassword)

try {
    $env:VISION_MYSQL_PASSWORD = `
        [Runtime.InteropServices.Marshal]::PtrToStringBSTR($PasswordPointer)
} finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($PasswordPointer)
}

if ([string]::IsNullOrEmpty($env:VISION_MYSQL_PASSWORD)) {
    throw "没有输入 MySQL 密码"
}
~~~

首次运行时，在同一终端中初始化表并启动服务：

~~~powershell
try {
    & $PythonGpu scripts\08_init_mysql.py
    & $PythonGpu scripts\07_vision_http_server.py
} finally {
    Remove-Item Env:VISION_MYSQL_PASSWORD -ErrorAction SilentlyContinue
    Remove-Variable SecurePassword, PasswordPointer -ErrorAction SilentlyContinue
}
~~~

`08_init_mysql.py` 使用 `CREATE TABLE IF NOT EXISTS`，重复执行不会删除已有数据。以后启动服务时可以省略第 08 步，但仍需在启动 Python 前设置环境变量并读取密码。

PowerShell 环境变量只对当前终端及其子进程有效。`config.py` 在 Python 进程启动时读取这些值，因此服务启动后再修改环境变量不会更新当前连接配置。

### 8.4 运行模拟机器人客户端

保持终端 A 中的服务运行。打开终端 B，重新定义项目目录和解释器变量：

~~~powershell
Set-Location E:\trainstudy_demo\VisionStudyProject
$PythonGpu = "E:\anaconda\envs\part_yolo_gpu\python.exe"

& $PythonGpu scripts\10_robot_client.py
~~~

客户端会：

1. 请求 `GET /health`。
2. 生成一个 `robot_<UUID>` 格式的 `command_id`。
3. 向 `POST /vision/command` 发送 `capture_and_detect`。
4. 打印 HTTP 状态、检测时间、目标数量、类别、置信度、方向角、相机 XYZ、MySQL 状态和本次请求耗时。

它不会发送图片，也不会发送机器人运动指令。

### 8.5 用 PowerShell 直接调用 HTTP

健康检查：

~~~powershell
Invoke-RestMethod `
  -Method Get `
  -Uri "http://127.0.0.1:8008/health"
~~~

发送检测命令：

~~~powershell
$Body = @{
    command = "capture_and_detect"
    command_id = "pick_0001"
} | ConvertTo-Json -Compress

Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8008/vision/command" `
  -ContentType "application/json; charset=utf-8" `
  -Body $Body
~~~

`command_id` 必须是非空字符串，由调用方生成并用于关联 HTTP 响应和 MySQL 请求记录。

## 9. HTTP 接口约定

### 9.1 路由

~~~http
GET /health
~~~

响应：

~~~json
{
  "status": "ok"
}
~~~

健康检查本身不会打开 Kinect，但服务进程必须已经成功加载模型并启动。

检测请求：

~~~http
POST /vision/command
Content-Type: application/json

{
  "command": "capture_and_detect",
  "command_id": "pick_0001"
}
~~~

### 9.2 成功响应

下面示例展示实际字段结构；数值和类别仅用于说明：

~~~json
{
  "status": "ok",
  "command": "capture_and_detect",
  "command_id": "pick_0001",
  "captured_at": "2026-08-04T04:00:00.000Z",
  "coordinate_frame": "kinect_camera",
  "coordinate_unit": "mm",
  "detection_count": 1,
  "detections": [
    {
      "part_type": "part_a",
      "class_id": 0,
      "confidence": 0.9567,
      "x1": 716.2,
      "y1": 379.4,
      "x2": 1043.8,
      "y2": 600.6,
      "center_x": 880.0,
      "center_y": 490.0,
      "bbox_width": 327.6,
      "bbox_height": 221.2,
      "orientation_deg": 12.5,
      "obb_points": [
        [750.9, 379.4],
        [1043.8, 444.4],
        [1009.1, 600.6],
        [716.2, 535.6]
      ],
      "box_type": "obb",
      "distance_mm": 1002.0,
      "camera_x_mm": 100.0,
      "camera_y_mm": -50.0,
      "camera_z_mm": 1000.0,
      "depth_valid": true,
      "depth_sample_count": 24
    }
  ],
  "image_width": 1920,
  "image_height": 1080,
  "mysql_saved": true,
  "mysql_request_id": 9001,
  "mysql_error": null
}
~~~

没有目标也是成功结果：HTTP 200、`detection_count=0`、`detections=[]`。如果数据库未配置或写入失败，视觉字段保持不变，三个 MySQL 字段变为：

~~~json
{
  "mysql_saved": false,
  "mysql_request_id": null,
  "mysql_error": "mysql_save_failed"
}
~~~

服务不会把数据库异常详情、账号、密码或连接字符串放进 HTTP 响应。

### 9.3 HTTP 状态码

| 状态码 | 含义 |
| --- | --- |
| 200 | 视觉命令成功；零目标和 MySQL 写入失败也属于这种情况 |
| 400 | JSON 非法、缺少字段、`command_id` 为空或命令不受支持 |
| 404 | 请求路径不是 `/health` 或 `/vision/command` |
| 503 | Kinect 无法打开，或本次没有取得同步 RGB-D 帧 |
| 500 | 模型、推理或结果编码发生内部错误 |

Kinect 取帧失败后，服务会释放设备。下一条合法命令会重新尝试打开 Kinect。模型或推理异常不会被错误地标记为 Kinect 503。

## 10. MySQL 保存逻辑

数据库层只有两个公开入口：

~~~python
initialize_database()
request_id = save_detection_result(command_id, captured_at, detections)
~~~

两个表的职责如下：

| 表 | 作用 |
| --- | --- |
| `vision_request` | 每条命令一行，保存 `command_id`、UTC 检测时间、目标数量和状态 |
| `vision_detection` | 每个目标一行，通过 `request_id` 关联命令，保存类别、框、角度、距离和相机 XYZ |

一条请求及其全部目标共用一个事务：

- 先写一条 `vision_request`。
- 再逐条写入 `vision_detection`。
- 所有 SQL 成功后才 `commit`。
- 任一步失败都 `rollback`。
- 零目标仍写入一条 `vision_request`，但不写检测明细。
- `obb_points` 使用 UTF-8 JSON 文本保存。

HTTP 层把准备返回客户端的 `command_id`、`captured_at` 和 `detections` 原样交给这个接口，不会重新推理、重新筛选或创建另一套字段名称。

## 11. 结果字段怎么理解

- `captured_at`：完成 Kinect 取帧后的 UTC ISO 8601 时间，使用 `Z` 后缀。
- `coordinate_frame`：固定为 `kinect_camera`。
- `coordinate_unit`：固定为 `mm`。
- `center_x`、`center_y`、`obb_points`：彩色图像素坐标；`bbox_width`、`bbox_height` 是四角点外接水平框的宽和高，不是旋转矩形的两条边长。
- `distance_mm`：检测框中央有效深度样本的中位数。
- `camera_x_mm`、`camera_y_mm`、`camera_z_mm`：同一区域有效 Kinect 相机坐标的中位数。
- `orientation_deg`：OBB 长轴的图像平面角度，不是机器人末端姿态。
- `part_type`：来自模型权重的类别名称，不由 `classes.txt` 在推理时重新覆盖。

第 05 步为了观察效果会保留深度无效的二维检测；第 06 步、HTTP 和 MySQL 只交付 `depth_valid=true` 的 OBB 检测。因此服务响应中的目标一定包含可用的距离和相机 XYZ 字段。

## 12. 常见运行问题

### 12.1 使用了错误的 Python

先确认当前命令实际调用的解释器：

~~~powershell
& $PythonGpu -c "import sys; print(sys.executable)"
~~~

项目命令统一使用 `python.exe -m pip`，不要依赖可能指向其他环境的独立 `pip.exe` 或系统 `python`。

### 12.2 模型、任务和标签不匹配

典型错误包括：

- 5 列 Detect 数据集配 `YOLO_TASK="obb"`。
- `yolo11n.pt` 配 `YOLO_TASK="obb"`。
- Detect `best.pt` 用于第 06、07 步。
- 第 05 步配置 `YOLO_TASK="detect"`，但加载 OBB 权重。

解决方法不是忽略检查，而是回到第 5 节，让标签格式、任务名称和模型类型保持一致。

### 12.3 Kinect 找不到、被占用或取帧超时

按错误阶段分别检查：

- 找不到 `Microsoft.Kinect.dll`：安装 Kinect SDK 2.0，或修改程序集路径。
- SDK 找不到默认设备：检查 Kinect 供电、USB 3.0、Windows 设备管理器和驱动。
- 设备已枚举但不可用：断开 Kinect Studio，并关闭其他占用相机的程序。
- 能打开但收不到同步帧：关闭 Kinect Studio、Configuration Verifier 和相关进程后重试。
- 开始能取帧，随后间歇超时：在 Windows 声音设置中找到“麦克风阵列（Xbox NUI Sensor）”，将“音频增强”关闭。

单纯增大 `KINECT_FRAME_TIMEOUT_SECONDS` 不能解决设备被占用或数据流已经中断的问题。第 05 步遇到取帧错误会退出；HTTP 服务会释放 Kinect，并在下一条合法命令时重新打开。

### 12.4 `mysql_saved=false`

依次确认：

- 数据库和账号已经存在。
- `VISION_MYSQL_USER`、`VISION_MYSQL_PASSWORD`、`VISION_MYSQL_DATABASE` 均在启动 Python 前设置。
- 端口是 1～65535 的整数。
- 字符集为 `utf8mb4`。
- 账号拥有建表和写入权限。
- MySQL 服务正在监听配置的主机和端口。

不要把真实密码写进 `config.py`，也不要为了排错把密码打印到终端或 HTTP 响应。

### 12.5 其他电脑无法访问 HTTP

默认 `VISION_HTTP_HOST="127.0.0.1"` 只允许本机访问。确实需要局域网客户端时，明确改为服务电脑的局域网地址，并按现场网络策略配置 Windows 防火墙。

当前 HTTP 接口没有身份认证和 TLS，不应直接暴露到不可信网络。

### 12.6 找不到数据或权重

大文件默认不提交到 Git。新工作副本需要重新准备：

- `data/raw_labeled/` 中的图片、标签和 `classes.txt`。
- 通过第 02 步生成的 `data/datasets/data.yaml`。
- 自己训练的 `best.pt`，或按 [models/README.md](models/README.md) 准备明确来源的 OBB 权重。

## 13. 项目结构

~~~text
VisionStudyProject/
├── config.py                         # 集中配置
├── camera_capture.py                 # 普通相机采集
├── yolo_dataset_builder.py           # 数据校验、拆分和版本化
├── yolo_trainer.py                   # 训练前检查和 YOLO 训练
├── kinect_v2_camera.py               # Kinect SDK RGB-D 读取与坐标映射
├── depth_detection.py                # Detect/OBB 解析、深度融合和绘制
├── kinect_obb_detection.py            # 单次 OBB 取帧、推理和结果封装
├── vision_http_server.py              # HTTP 路由、Kinect 生命周期和 MySQL 集成
├── mysql_detection_store.py           # MySQL 建表和事务写入
├── scripts/
│   ├── 01_capture_images.py
│   ├── 02_build_yolo_dataset.py
│   ├── 03_train_yolo.py
│   ├── 04_realtime_inference.py
│   ├── 05_realtime_depth_inference.py
│   ├── 06_capture_and_detect.py
│   ├── 07_vision_http_server.py
│   ├── 08_init_mysql.py
│   └── 10_robot_client.py
├── data/
│   ├── raw_labeled/                  # 原始图片、标签、classes.txt
│   ├── datasets/                     # 稳定 data.yaml 和版本目录
│   └── runtime/                      # 被 Git 忽略的临时运行资料
├── models/
│   └── README.md                     # 本地模型来源、校验值和许可说明
├── runs/                              # 被 Git 忽略的训练输出
├── requirements.txt
└── GIT_WORKFLOW.md                    # Git 分支、提交、PR 和同步说明
~~~

Git 初学者需要发布新功能、创建 PR 或在合并后同步本地 `main` 时，可阅读 [GIT_WORKFLOW.md](GIT_WORKFLOW.md)。
