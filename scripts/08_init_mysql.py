"""第八步入口：在已配置的 MySQL 数据库中创建检测结果表。"""

from __future__ import annotations

import sys
from pathlib import Path


# 当前文件位于 scripts 子目录；加入项目根目录后可直接运行本脚本。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mysql_detection_store import initialize_database


def main() -> None:
    """创建 vision_request 和 vision_detection，不启动相机或 HTTP。"""

    initialize_database()
    print("MySQL 表初始化完成：vision_request、vision_detection")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"MySQL 表初始化失败：{exc}") from exc
