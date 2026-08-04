# 本地 OBB 测试模型

## `notepay_yolov8s_obb_receipt.pt`

- 用途：检测购物收据中的旋转文字区域，适合用常见收据验证本项目的 OBB 四角点、方向角和 Kinect 深度融合调用链。
- 来源：<https://huggingface.co/NeoCode77/notepay-yolo-receipt>
- 下载日期：2026-08-04
- 文件大小：6,678,540 字节
- SHA-256：`05b139a95cdcaa49ca1524d4b66a0273d90dbfa83affbfc73980a0b3668f1e08`
- 模型仓库声明的许可证：MIT。项目还使用 Ultralytics；其开源版本为 AGPL-3.0，闭源或商业部署应另行核对 Ultralytics Enterprise 许可：<https://www.ultralytics.com/license>

权重已在 `E:\anaconda\envs\part_yolo_gpu\python.exe`、Ultralytics 8.4.92 中实际加载：

- `model.task == "obb"`
- 内部模型类型为 `ultralytics.nn.tasks.OBBModel`
- 权重内置类别为：`0 line_item`、`1 nama_toko`、`2 tanggal_waktu`、`3 total_belanja`

权重内置类别顺序是运行时真值。模型仓库 README 把前两个类别编号写成了相反顺序，因此不要手工覆盖 `model.names`。

本地冒烟测试使用 CORD v2 的公开收据样图（CC BY 4.0，NAVER Clova）：
<https://huggingface.co/datasets/naver-clova-ix/cord-v2>。在 RTX 3060 上，项目自己的
`parse_yolo_result(..., task="obb")` 成功解析出 24 个旋转框。样图和标注预览保存在被 Git
忽略的 `data/runtime/model_tests/` 中。该结果只证明模型与本项目接口兼容，不代表对中文收据、真实 Kinect 深度或生产场景的精度验收；模型仓库也没有发布完整精度指标。

`config.INFERENCE_MODEL_PATH` 已指向此权重，步骤 06、07 可直接使用。若要运行步骤 05 的连续 OBB 预览，请先把 `config.YOLO_TASK` 临时改成 `"obb"`；当前 5 列 Detect 数据集不能用它直接执行步骤 03 的 OBB 训练。
