"""统一日志：写到 data/logs/workbench.log，同时打到控制台。

只配置 "rag" 这一支 logger，不动 root：uvicorn 自带 handler，
挂到 root 上会变成控制台双份输出。

之前全项目一条日志都没有，出问题（比如「知识库怎么空了」「刚才那问为什么失败」）
只能靠前端看，事后完全无从追查。
"""
from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)-7s %(name)-16s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"
_MAX_BYTES = 2 * 1024 * 1024
_BACKUPS = 3

_configured = False


def setup(base_dir: str | Path, level: int = logging.INFO) -> Path | None:
    """配置日志。返回日志文件路径；目录建不出来时退回只打控制台。"""
    global _configured
    if _configured:
        return None
    _configured = True

    logger = logging.getLogger("rag")
    logger.setLevel(level)
    logger.propagate = False  # 避免被 root 再打一遍
    if logger.handlers:
        return None

    fmt = logging.Formatter(_FORMAT, _DATEFMT)

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(fmt)
    logger.addHandler(console)

    try:
        log_dir = Path(base_dir) / "data" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / "workbench.log"
        fh = logging.handlers.RotatingFileHandler(
            path, maxBytes=_MAX_BYTES, backupCount=_BACKUPS, encoding="utf-8"
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except OSError as e:  # 磁盘只读之类
        logger.warning("日志文件不可写，仅输出到控制台：%s", e)
        return None
    return path
