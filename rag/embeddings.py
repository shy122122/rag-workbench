"""百炼向量化封装：走 OpenAI 兼容接口 /v1/embeddings。"""
from __future__ import annotations

import asyncio
from typing import Iterable

from openai import AsyncOpenAI

from .errors import ApiKeyMissing

BATCH = 25  # 单请求最大文本条数


class Embedder:
    def __init__(self, base_url: str, api_key: str, model: str, dimensions: int):
        self.model = model
        self.dimensions = dimensions
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    async def embed_texts(
        self, texts: Iterable[str], on_batch=None, concurrency: int = 4
    ) -> tuple[list[list[float]], dict]:
        """批量向量化。返回 (向量列表, 用量统计)。on_batch 每批完成回调。"""
        texts = list(texts)
        if not texts:
            return [], {"batches": 0, "total_tokens": 0}
        batches = [texts[i : i + BATCH] for i in range(0, len(texts), BATCH)]
        total_tokens = 0
        sem = asyncio.Semaphore(concurrency)

        async def run_one(batch: list[str]):
            nonlocal total_tokens
            async with sem:
                resp = await self._client.embeddings.create(
                    model=self.model, input=batch, dimensions=self.dimensions
                )
                total_tokens += resp.usage.total_tokens
                if on_batch:
                    on_batch(len(batch))
                ordered = sorted(resp.data, key=lambda d: d.index)
                return [item.embedding for item in ordered]

        results = await asyncio.gather(*(run_one(b) for b in batches))
        vectors = [v for r in results for v in r]
        return vectors, {"batches": len(batches), "total_tokens": total_tokens}
