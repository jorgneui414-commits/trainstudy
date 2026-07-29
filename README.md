# VisionStudyProject

这是一个独立的视觉学习项目。`VisionTeachDemo` 只用于阅读和参考，本项目不会导入、复制或修改旧项目中的代码与数据。

当前项目完成两件事：

1. 用 OpenCV 从本地普通 USB 相机采集图片。
2. 校验扁平目录中的 YOLO 标注，拆分训练集、验证集和测试集，并生成 `data.yaml`。

## 项目结构

```text
VisionStudyProject/
├── config.py                    # 经常调整的参数
├── camera_capture.py            # OpenCV 普通相机采集逻辑
├── yolo_dataset_builder.py      # YOLO 数据校验、拆分和版本管理
├── scripts/
│   ├── 01_capture_images.py
│   └── 02_build_yolo_dataset.py
├── data/
│   ├── raw_labeled/             # 原始图片、标签和 classes.txt
│   └── datasets/versions/       # 每次构建出的数据集版本
└── tests/
```

## 1. 准备 Python 环境

下面的命令都在 PowerShell 中执行。项目沿用现有的 `part_yolo_cpu` 环境：

```powershell
Set-Location E:\trainstudy_demo\VisionStudyProject
E:\anaconda\envs\part_yolo_cpu\python.exe -m pip install -r requirements.txt
```

后续示例直接写解释器的完整路径，这样不会误用系统中的其他 Python。

## 2. 修改集中配置

打开 `config.py`，通常只需调整下面几项：

- `CAMERA_ID = 0`：系统中的第 0 个 USB 摄像头；若接入多个相机，可尝试 `1`、`2` 等编号。
- `FRAME_WIDTH = 1280`、`FRAME_HEIGHT = 720`：请求的画面尺寸；实际尺寸取决于相机是否支持。
- `CAPTURE_IMAGE_COUNT = 1`：本次最多保存多少张图片。
- `CAPTURE_INTERVAL_SECONDS = 0.5`：两次自动保存之间的秒数。
- `RAW_LABELED_DIR`：采集图片及后续 YOLO 标注所在的目录。
- `TRAIN_RATIO`、`VAL_RATIO`、`TEST_RATIO`：默认分别为 `0.7`、`0.2`、`0.1`。
- `RANDOM_SEED = 42`：固定随机拆分结果，便于复现实验。

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

## 6. 运行测试

```powershell
E:\anaconda\envs\part_yolo_cpu\python.exe -m unittest discover -s tests -v
```

测试使用模拟相机和临时目录，不会打开真实相机，也不会修改 `VisionTeachDemo`。
