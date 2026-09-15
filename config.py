"""配置读写：config.json 保存界面可调参数，.env 保存 API Key。"""
import json
import os
from pathlib import Path

from dotenv import load_dotenv, set_key

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"
ENV_PATH = BASE_DIR / ".env"

DEFAULT_CONFIG = {
    "llm_model": "qwen-plus",
    "embedding_model": "text-embedding-v3",
    "dimensions": 1024,
    "chunk_strategy": "recursive",  # 切分策略：recursive=递归字符切分 / structure=按标题等级切块
    "structure_level": 2,  # structure 策略下：块边界到第几级标题（1~6）
    "chunk_size": 500,
    "chunk_overlap": 50,
    "top_k": 5,
    "recall_k": 20,  # 粗排召回数：向量检索先取这么多，再交给 rerank 精排
    "neighbor_window": 1,  # 邻块扩展：命中块前后各带 n 块一起送进上下文（0=关闭）
    "rerank_enabled": True,
    "rerank_model": "gte-rerank-v2",
    "retrieval_min_score": 0.0,  # 相似度阈值：低于此的命中不展示
    "clean_enabled": True,
    "clean_noise": True,
    "clean_dedup": True,
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    # DashScope 原生接口，rerank 走这里（OpenAI 兼容模式不含 rerank）
    "native_base_url": "https://dashscope.aliyuncs.com/api/v1",
}

# 各字段类型集合：save/load 时把表单提交的字符串矫正回正确类型
_BOOL_FIELDS = {"clean_enabled", "clean_noise", "clean_dedup", "rerank_enabled"}
_INT_FIELDS = {
    "dimensions",
    "structure_level",
    "chunk_size",
    "chunk_overlap",
    "top_k",
    "recall_k",
    "neighbor_window",
}
_FLOAT_FIELDS = {"retrieval_min_score"}


def _defaults() -> dict:
    return dict(DEFAULT_CONFIG)


def _coerce(key: str, value):
    """把外部传来的值（表单可能是字符串）矫正为字段应有类型；无法解析返回 None。"""
    if value is None:
        return None
    if key in _BOOL_FIELDS:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("1", "true", "yes", "on")
    if key in _INT_FIELDS:
        try:
            return int(float(str(value).strip()))
        except ValueError:
            return None
    if key in _FLOAT_FIELDS:
        try:
            return float(str(value).strip())
        except ValueError:
            return None
    if not isinstance(value, str):
        return None
    return value


def _fill(merged: dict, src: dict) -> dict:
    for k in DEFAULT_CONFIG:
        if k not in src:
            continue
        v = _coerce(k, src[k])
        if v is not None:
            merged[k] = v
    return merged


_cache: dict = {"mtime": None, "cfg": None}


def load_config() -> dict:
    """读取 config.json；缺失字段补默认值，并把字段矫正为正确类型。

    每个请求都会调用，用 mtime 做缓存，文件没变就不重复解析。
    """
    try:
        mtime = CONFIG_PATH.stat().st_mtime_ns
    except OSError:
        mtime = None
    if _cache["cfg"] is not None and _cache["mtime"] == mtime:
        return dict(_cache["cfg"])

    cfg = _defaults()
    if mtime is not None:
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            _fill(cfg, data)
        except (json.JSONDecodeError, OSError):
            pass
    _cache["mtime"] = mtime
    _cache["cfg"] = cfg
    return dict(cfg)


def save_config(cfg: dict) -> dict:
    """仅保留合法字段并落盘（含类型矫正），返回保存后的配置。"""
    merged = _fill(_defaults(), cfg)
    CONFIG_PATH.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _cache["mtime"] = None  # 让下次 load_config 重新读盘
    _cache["cfg"] = None
    return merged


def get_api_key() -> str:
    load_dotenv(ENV_PATH)
    return os.environ.get("DASHSCOPE_API_KEY", "").strip()


def save_api_key(key: str) -> None:
    key = (key or "").strip()
    if not ENV_PATH.exists():
        ENV_PATH.write_text("", encoding="utf-8")
    if key:
        set_key(str(ENV_PATH), "DASHSCOPE_API_KEY", key)
    load_dotenv(ENV_PATH, override=True)


def has_api_key() -> bool:
    return bool(get_api_key())


def mask_key(key: str) -> str:
    if not key:
        return ""
    if len(key) <= 8:
        return "*" * len(key)
    return key[:4] + "*" * 8 + key[-4:]
