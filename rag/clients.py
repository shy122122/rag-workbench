"""共享 HTTP 客户端缓存。

AsyncOpenAI 内部维护连接池，每次请求都新建等于丢掉连接复用、反复 TCP 握手，
且旧实例从不关闭。这里按 (base_url, api_key) 缓存复用。
"""
from __future__ import annotations

import threading

from openai import AsyncOpenAI

_clients: dict[tuple[str, str], AsyncOpenAI] = {}
_lock = threading.Lock()


def get_openai_client(base_url: str, api_key: str) -> AsyncOpenAI:
    key = (base_url, api_key)
    client = _clients.get(key)
    if client is not None:
        return client
    with _lock:
        client = _clients.get(key)
        if client is None:
            client = AsyncOpenAI(base_url=base_url, api_key=api_key)
            _clients[key] = client
        return client


async def close_all() -> None:
    """进程退出时关闭连接池，避免解释器收尾时刷出一堆 asyncgen 异常。"""
    with _lock:
        clients = list(_clients.values())
        _clients.clear()
    for client in clients:
        try:
            await client.close()
        except Exception:  # noqa: BLE001
            pass
