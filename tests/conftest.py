"""pytest 公共配置：把项目根加进 sys.path，让 tests/ 能 import rag / config。

（项目是扁平布局，没打包，装了 pytest 之后直接 `python -m pytest` 就能跑。）
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Windows 控制台默认 GBK，断言里的中文会变成乱码，失败时根本看不出在比什么
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
