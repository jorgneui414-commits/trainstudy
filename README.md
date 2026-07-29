# VisionStudyProject

这是一个独立的视觉学习项目。`VisionTeachDemo` 只用于阅读和参考，本项目不会导入、复制或修改旧项目中的代码与数据。

当前项目完成四件事：

1. 用 OpenCV 从本地普通 USB 相机采集图片。
2. 校验扁平目录中的 YOLO 标注，拆分训练集、验证集和测试集，并生成 `data.yaml`。
3. 使用 Ultralytics YOLO 训练普通 Detect 或 OBB 模型。
4. 使用训练好的模型，对普通 USB 相机画面进行实时推理验证。

## 项目结构

```text
VisionStudyProject/
├── config.py                    # 经常调整的参数
├── camera_capture.py            # OpenCV 普通相机采集逻辑
├── yolo_dataset_builder.py      # YOLO 数据校验、拆分和版本管理
├── yolo_trainer.py              # 训练前校验、设备选择和 YOLO 训练调用
├── scripts/
│   ├── 01_capture_images.py
│   ├── 02_build_yolo_dataset.py
│   ├── 03_train_yolo.py
│   └── 04_realtime_inference.py
├── data/
│   ├── raw_labeled/             # 原始图片、标签和 classes.txt
│   └── datasets/versions/       # 每次构建出的数据集版本
├── runs/                         # 自动生成的训练结果（不提交到 Git）
└── tests/
```

## 1. 准备 Python 环境

下面的命令都在 PowerShell 中执行。项目沿用现有的 `part_yolo_cpu` 环境：

```powershell
Set-Location E:\trainstudy_demo\VisionStudyProject
E:\anaconda\envs\part_yolo_cpu\python.exe -m pip install -r requirements.txt
```

后续示例直接写解释器的完整路径，这样不会误用系统中的其他 Python。

### GPU 训练环境

采集和构建数据集仍可使用 `part_yolo_cpu`。训练 GPU 模型时，使用独立的
`part_yolo_gpu` 环境，避免修改原来的 CPU 环境。首次创建时运行：

```powershell
E:\anaconda\Scripts\conda.exe create --name part_yolo_gpu --clone part_yolo_cpu -y
E:\anaconda\envs\part_yolo_gpu\python.exe -m pip install --force-reinstall --no-deps torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
E:\anaconda\envs\part_yolo_gpu\python.exe -m pip install -r requirements.txt
```

无需另外安装完整 CUDA Toolkit；PyTorch 的 CUDA wheel 已包含训练所需运行库。安装后可以用下面的命令确认它真正使用的是 GPU：

```powershell
E:\anaconda\envs\part_yolo_gpu\python.exe -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

本项目的 RTX 3060 Laptop GPU 应显示为 CUDA 可用。若结果为 `False`，请不要直接开始训练，先检查是否误用了 `part_yolo_cpu` 解释器。

## 2. 修改集中配置

打开 `config.py`，通常只需调整下面几项：

- `CAMERA_ID = 0`：系统中的第 0 个 USB 摄像头；若接入多个相机，可尝试 `1`、`2` 等编号。
- `FRAME_WIDTH = 1280`、`FRAME_HEIGHT = 720`：请求的画面尺寸；实际尺寸取决于相机是否支持。
- `CAPTURE_IMAGE_COUNT = 1`：本次最多保存多少张图片。
- `CAPTURE_INTERVAL_SECONDS = 0.5`：两次自动保存之间的秒数。
- `RAW_LABELED_DIR`：采集图片及后续 YOLO 标注所在的目录。
- `TRAIN_RATIO`、`VAL_RATIO`、`TEST_RATIO`：默认分别为 `0.7`、`0.2`、`0.1`。
- `RANDOM_SEED = 42`：固定随机拆分结果，便于复现实验。
- `YOLO_TASK`、`YOLO_MODEL`：选择普通 Detect 或 OBB 模型。
- `TRAIN_DEVICE`：填写 `"gpu"` 使用 NVIDIA 显卡，填写 `"cpu"` 强制使用 CPU。
- `TRAIN_EPOCHS`、`TRAIN_IMAGE_SIZE`、`TRAIN_BATCH_SIZE`：训练轮数、输入尺寸和每批图片数。
- `INFERENCE_MODEL_PATH`：实时推理所加载的 `best.pt` 路径；完成新的训练后，请确认它指向要验证的结果。
- `INFERENCE_DEVICE = 0`：实时推理默认使用第一张 NVIDIA GPU；可改为 `"cpu"`。
- `INFERENCE_CONFIDENCE_THRESHOLD = 0.25`：低于该置信度的目标不会显示。

## 3. 采集图片

先确认 USB 相机已连接，再运行：

```powershell
E:\anaconda\envs\part_yolo_cpu\python.exe scripts\01_capture_images.py
```

程序会显示实时画面，并将图片保存到 `data/raw_labeled/`。文件采用四位数字连续编号，例如 `0001.jpg`、`0002.jpg`；已有文件不会被覆盖。第一张会立即保存，后续图片按配置的间隔保存。按 `q` 或 `Esc` 可以提前结束。

自动测试不会启动真实相机。第一次连接硬件时，应先把 `CAPTURE_IMAGE_COUNT` 设为较小的值，确认画面、分辨率和保存目录都正确。

当前阶段仅支持本地普通 USB 相机；具体 SDK 等型号确定后再单独设计。

## 4. 准备 YOLO 原始标注

把文件平铺放到 `data/raw_labeled/`。每张图片必须有一个同名 `.txt` 标签，并且必须有 `classes.txt`：

```text
data/raw_labeled/
├── classes.txt
├── 0001.jpg
├── 0001.txt
├── 0002.jpg
└── 0002.txt
```

`classes.txt` 每行写一个类别，行号从 0 开始对应标签中的类别编号：

```text
part_a
part_b
```

支持两种标签格式，所有坐标都必须是 `[0, 1]` 范围内的归一化数值：

- YOLO Detection（每行 5 列）：`class_id x_center y_center width height`
- YOLO OBB（每行 9 列）：`class_id x1 y1 x2 y2 x3 y3 x4 y4`

同一个数据集不能混用 5 列和 9 列格式。单个空标签可以表示没有目标的负样本，但所有标签都为空时不会构建数据集。程序还会拒绝缺失标签、孤立标签、重复文件 stem、非法类别编号、`NaN`/无穷值和越界坐标。

默认比例要求训练、验证、测试三组都非空，因此至少准备 5 组图片与标签；数据越多，拆分比例越有代表性。

## 5. 构建 YOLO 数据集

运行：

```powershell
E:\anaconda\envs\part_yolo_cpu\python.exe scripts\02_build_yolo_dataset.py
```

一次成功构建会生成类似下面的目录：

```text
data/datasets/
├── data.yaml                         # 始终指向最新成功版本
└── versions/20260728_163000/
    ├── classes.txt
    ├── data.yaml
    ├── images/{train,val,test}/
    └── labels/{train,val,test}/
```

版本名使用时间戳；同一秒重复构建时会自动增加序号。旧版本会被保留。只有新版本全部完成后，稳定入口 `data/datasets/data.yaml` 才会更新，所以一次失败不会破坏上一次可用的数据集。

## 6. 训练 YOLO 模型

先确认已经完成步骤 5，且 `data/datasets/data.yaml` 存在。默认配置会使用：

- `YOLO_TASK = "detect"`
- `YOLO_MODEL = "yolo11n.pt"`
- GPU 0（RTX 3060）
- 200 epochs、`imgsz=640`、`batch=8`、早停耐心值 50

运行：

```powershell
E:\anaconda\envs\part_yolo_gpu\python.exe scripts\03_train_yolo.py
```

首次运行时 Ultralytics 会下载预训练权重。每次训练的输出会保存到
`runs/train/` 下的独立目录，其中 `weights/best.pt` 是验证集表现最好的权重，
`weights/last.pt` 是最后一个 epoch 的权重。

### 切换 CPU/GPU

在 `config.py` 中修改：

```python
TRAIN_DEVICE = "gpu"  # 使用 TRAIN_GPU_INDEX 指定的显卡
# TRAIN_DEVICE = "cpu"  # 强制使用 CPU
```

选择 `gpu` 时，脚本会先检查 PyTorch 是否检测到 CUDA；检查失败会停止并提示使用
`part_yolo_gpu`，不会偷偷改用 CPU。

### 切换 Detect/OBB 模型

普通目标检测使用 5 列标签，例如：

```python
YOLO_TASK = "detect"
YOLO_MODEL = "yolo11n.pt"
```

OBB 使用 9 列四角点标签，例如：

```python
YOLO_TASK = "obb"
YOLO_MODEL = "yolo11n-obb.pt"
```

当前数据集是 5 列普通 Detect 标签，因此不能直接改成 OBB 训练。脚本会严格拦截
这种组合；请先用真实旋转框重新标注、运行步骤 02 构建新版本，再切换 OBB 配置。

## 7. 普通相机实时推理

先连接一台可被 Windows 识别的普通 UVC USB 相机，再运行：

```powershell
E:\anaconda\envs\part_yolo_gpu\python.exe scripts\04_realtime_inference.py
```

脚本只会打开 `config.CAMERA_ID` 指定的相机，不会自动扫描或切换到其他设备。它会先
读取首帧确认相机可用，然后使用 `INFERENCE_MODEL_PATH` 中的模型进行实时推理。窗口会
显示检测框、类别、置信度和 FPS；按 `q` 或 `Esc` 正常退出。当前阶段不保存图片或视频。

如果没有检测到相机，脚本会提示检查连接、Windows 设备识别、相机隐私权限，或修改
`CAMERA_ID` 后重新运行。

## 8. 运行测试

```powershell
E:\anaconda\envs\part_yolo_gpu\python.exe -m unittest discover -s tests -v
```

测试使用模拟相机、模拟 PyTorch/Ultralytics 模型和临时目录，不会打开真实相机、下载权重或启动正式训练，也不会修改 `VisionTeachDemo`。
