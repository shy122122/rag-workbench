"""公网演示配套：访问门禁、调用配额，以及随镜像分发的示例向量库。

只有设了 DEMO_PASSWORD 才会启用门禁，本地 `uvicorn app:app` 完全不受影响。

为什么要门禁：这个工作台调的是真实付费模型，公网裸奔等于把账号额度挂出去；
而且免费 PaaS 的文件系统不持久，容器一重建向量库就没了，所以还得有一份随镜像走的示例库。
"""
from __future__ import annotations

import base64
import hmac
import logging
import os
import shutil
import time
from collections import defaultdict, deque
from datetime import date
from pathlib import Path

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

log = logging.getLogger("rag.deploy")

# 平台探测存活用，必须免鉴权，否则会被判成不健康而反复重启
OPEN_PATHS = {"/healthz"}

# 真正花钱的接口：每次调用都要打百炼。只对这些计入配额。
COSTLY_PATHS = {
    "/api/query",
    "/api/match",
    "/api/knowledge/upload",
    "/api/knowledge/load-examples",
    "/api/test-key",
}

DEFAULT_DAILY_LIMIT = 200  # 全局每日上限，防的是「额度被吃光」而不是某个人的手速
DEFAULT_RATE_PER_MIN = 20  # 单 IP 每分钟上限，防的是连点


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        log.warning("%s=%r 不是整数，按默认值 %d 处理", name, raw, default)
        return default


def enabled() -> bool:
    """没设密码就当作本地开发，门禁整体不生效。"""
    return bool(os.environ.get("DEMO_PASSWORD", "").strip())


def _expected_user() -> str:
    return os.environ.get("DEMO_USER", "demo").strip() or "demo"


def _check_auth(header: str) -> bool:
    """校验 Basic 头。用 compare_digest 逐字节比，避免按字符提前返回泄漏密码长度。"""
    if not header.lower().startswith("basic "):
        return False
    try:
        raw = base64.b64decode(header[6:].strip(), validate=True).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return False
    user, sep, password = raw.partition(":")
    if not sep:
        return False
    want_user = _expected_user().encode()
    want_pass = os.environ.get("DEMO_PASSWORD", "").encode()
    return hmac.compare_digest(user.encode(), want_user) and hmac.compare_digest(
        password.encode(), want_pass
    )


def _client_ip(request) -> str:
    """Render/Caddy 这类反代后面，request.client.host 永远是对端网关，得看 X-Forwarded-For。"""
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class Gate:
    """内存态计数。单实例部署够用，容器重启后归零——可接受。"""

    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._day = date.today()
        self._used_today = 0

    def _roll_day(self) -> None:
        today = date.today()
        if today != self._day:
            log.info("跨天，调用配额重置（昨日用掉 %d 次）", self._used_today)
            self._day = today
            self._used_today = 0
            self._hits.clear()

    def charge(self, ip: str) -> str | None:
        """记一次计费调用。返回 None 表示放行，否则是拒绝原因。"""
        self._roll_day()

        rate = _env_int("DEMO_RATE_PER_MIN", DEFAULT_RATE_PER_MIN)
        if rate:
            now = time.monotonic()
            q = self._hits[ip]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= rate:
                return f"操作太快了，每分钟最多 {rate} 次，请稍候再试"
            q.append(now)

        limit = _env_int("DEMO_DAILY_LIMIT", DEFAULT_DAILY_LIMIT)
        if limit and self._used_today >= limit:
            return "今日演示额度已用完（这是为了防止 API 额度被耗尽）。如需继续，请联系我。"
        self._used_today += 1
        return None

    def usage(self) -> dict:
        self._roll_day()
        return {
            "used": self._used_today,
            "limit": _env_int("DEMO_DAILY_LIMIT", DEFAULT_DAILY_LIMIT),
        }


GATE = Gate()


class GateMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        # 没设密码就整个空转。少了这一句，本地不配 DEMO_PASSWORD 时会被自己的门禁锁死——
        # _check_auth 拿空密码去比永远不会通过。
        if not enabled():
            return await call_next(request)

        path = request.url.path
        if path in OPEN_PATHS:
            return await call_next(request)

        if not _check_auth(request.headers.get("authorization", "")):
            ip = _client_ip(request)
            log.warning("未通过鉴权：%s %s（来自 %s）", request.method, path, ip)
            return JSONResponse(
                {"ok": False, "message": "需要密码才能访问。用户名 demo，密码请向作者索取。"},
                status_code=401,
                # realm 只能是 ASCII：HTTP 头按 latin-1 编码，塞中文会直接 500。
                # charset 声明了凭据按 UTF-8 传，和下面 _check_auth 的解码方式对齐。
                headers={"WWW-Authenticate": 'Basic realm="RAG Workbench", charset="UTF-8"'},
            )

        if request.method != "GET" and path in COSTLY_PATHS:
            reason = GATE.charge(_client_ip(request))
            if reason:
                log.warning("配额拦截：%s %s -> %s", request.method, path, reason)
                return JSONResponse({"ok": False, "message": reason}, status_code=429)

        return await call_next(request)


# ---------------------------------------------------------------- 示例向量库
def seed_kb_if_empty(base: Path) -> bool:
    """容器重建后 data/kb 是空的，拿镜像里那份示例库顶上，免得面试官打开是个空壳。

    只认 meta.json 存不存在：它和 vectors.npy 同生共死，有一个就说明库有内容。
    """
    kb_dir = base / "data" / "kb"
    seed_dir = base / "seed" / "kb"
    if (kb_dir / "meta.json").exists():
        return False
    if not (seed_dir / "meta.json").exists():
        return False
    kb_dir.mkdir(parents=True, exist_ok=True)
    for f in seed_dir.iterdir():
        if f.is_file():
            shutil.copy2(f, kb_dir / f.name)
    log.info("data/kb 为空，已从 seed/kb 恢复示例向量库")
    return True
