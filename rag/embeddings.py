"""百炼向量化封装：走 OpenAI 兼容接口 /v1/embeddings。"""
from __future__ import annotations

import asyncio
import random
import threading
from typing import Iterable

from .clients import get_openai_client
from .errors import ModelCallError

BATCH = 25  # 单请求最大文本条数（百炼上限）
_MAX_RETRY = 3  # 429 / 网络抖动时的重试次数
_CACHE_MAX = 4096  # 向量缓存条数上限；超出直接清空重建，够用且简单

_cache: dict[tuple, list[float]] = {}
_cache_lock = threading.Lock()


def _cache_get(key: tuple) -> list[float] | None:
    with _cache_lock:
        return _cache.get(key)


def _cache_put(key: tuple, vec: list[float]) -> None:
    with _cache_lock:
        if len(_cache) >= _CACHE_MAX:
            _cache.clear()
        _cache[key] = vec


class Embedder:
    def __init__(self, base_url: str, api_key: str, model: str, dimensions: int):
        self.model = model
        self.dimensions = dimensions
        self._client = get_openai_client(base_url, api_key)

    async def _embed_batch(self, batch: list[str]) -> tuple[list[list[float]], int]:
        """单批向量化，失败按指数退避重试。返回 (向量, tokens)。"""
        last_err: Exception | None = None
        for attempt in range(_MAX_RETRY):
            try:
                resp = await self._client.embeddings.create(
                    model=self.model, input=batch, dimensions=self.dimensions
                )
                ordered = sorted(resp.data, key=lambda d: d.index)
                return [item.embedding for item in ordered], resp.usage.total_tokens
            except Exception as e:  # noqa: BLE001  openai 各类错误统一重试
                last_err = e
                if attempt < _MAX_RETRY - 1:
                    await asyncio.sleep(0.5 * (2**attempt) + random.uniform(0, 0.3))
        raise ModelCallError(f"向量化失败（已重试 {_MAX_RETRY} 次）：{last_err}") from last_err

    async def embed_texts(
        self, texts: Iterable[str], on_batch=None, concurrency: int = 4
    ) -> tuple[list[list[float]], dict]:
        """批量向量化。返回 (向量列表, 用量统计)。

        相同 (模型, 维度, 文本) 命中缓存直接复用——反复用同一问题试不同 Top-K 时，
        问题向量不必重算。
        """
        texts = list(texts)
        if not texts:
            return [], {"batches": 0, "total_tokens": 0, "cached": 0}

        keys = [(self.model, self.dimensions, t) for t in texts]
        vectors: list[list[float] | None] = [None] * len(texts)

        miss_idx: list[int] = []
        for i, key in enumerate(keys):
            hit = _cache_get(key)
            if hit is None:
                miss_idx.append(i)
            else:
                vectors[i] = hit
        cached_n = len(texts) - len(miss_idx)

        if not miss_idx:
            return vectors, {"batches": 0, "total_tokens": 0, "cached": cached_n}

        batches = [
            miss_idx[i : i + BATCH] for i in range(0, len(miss_idx), BATCH)
        ]
        total_tokens = 0
        sem = asyncio.Semaphore(concurrency)
        lock = asyncio.Lock()

        async def run_one(idxs: list[int]) -> None:
            nonlocal total_tokens
            async with sem:
                vecs, tokens = await self._embed_batch([texts[i] for i in idxs])
            async with lock:
                total_tokens += tokens
                for i, v in zip(idxs, vecs):
                    vectors[i] = v
                    _cache_put(keys[i], v)
            if on_batch:
                on_batch(len(idxs))

        await asyncio.gather(*(run_one(b) for b in batches))
        return vectors, {
            "batches": len(batches),
            "total_tokens": total_tokens,
            "cached": cached_n,
        }
