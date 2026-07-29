"""按 config.py 的设置使用普通 USB 相机采集图片。"""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from camera_capture import capture_images


def main() -> None:
    # 采集参数统一从 config.py 读取，修改配置后无需改动本入口脚本。
    saved_paths = capture_images(
        camera_id=config.CAMERA_ID,
        output_dir=config.RAW_LABELED_DIR,
        image_count=config.CAPTURE_IMAGE_COUNT,
        interval_seconds=config.CAPTURE_INTERVAL_SECONDS,
        frame_width=config.FRAME_WIDTH,
        frame_height=config.FRAME_HEIGHT,
    )

    # capture_images 返回实际成功写入的图片路径，用户提前退出时数量可能小于目标数量。
    print("\n图片采集结束。")
    print(f"保存数量：{len(saved_paths)}/{config.CAPTURE_IMAGE_COUNT}")
    print(f"保存目录：{config.RAW_LABELED_DIR}")


if __name__ == "__main__":
    main()
