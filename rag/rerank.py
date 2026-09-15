"""百炼精排（rerank）封装。

rerank 不在 OpenAI 兼容路径里，走 DashScope 原生接口：
POST {native_base_url}/services/rerank/text-rerank/text-rerank

与向量检索的区别：向量检索是「query 和文档各自编码再比距离」（双塔，快但粗），
rerank 是把 query 和文档拼在一起送进模型逐对打分（cross-encoder，慢但准）。
所以标准做法是：向量检索粗排召回一批，rerank 精排出最终 Top-K。
"""
from __future__ import annotations

import httpx2

from .errors import ModelCallError

_PATH = "/services/rerank/text-rerank/text-rerank"


class Reranker:
    def __init__(self, native_base_url: str, api_key: str, model: str, timeout: float = 30.0):
        self.model = model
        self._url = native_base_url.rstrip("/") + _PATH
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._timeout = timeout

    async def rerank(
        self, query: str, documents: list[str], top_n: int | None = None
    ) -> tuple[list[dict], dict]:
        """逐对打分并按相关性降序返回。

        返回 (results, usage)，results[i] = {"index": 原始下标, "relevance_score": float}。
        注意 relevance_score 是 cross-encoder 打分，量纲与余弦相似度不同，不要跨模型比较。
        """
        if not documents:
            return [], {"total_tokens": 0}

        params: dict = {"return_documents": False}
        if top_n:
            params["top_n"] = min(int(top_n), len(documents))
        body = {
            "model": self.model,
            "input": {"query": query, "documents": documents},
            "parameters": params,
        }

        try:
            async with httpx2.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(self._url, headers=self._headers, json=body)
        except httpx2.HTTPError as e:
            raise ModelCallError(f"精排接口请求失败：{e}") from e

        if resp.status_code != 200:
            raise ModelCallError(f"精排接口返回 {resp.status_code}：{resp.text[:200]}")

        try:
            payload = resp.json()
            raw = payload["output"]["results"]
        except (ValueError, KeyError, TypeError) as e:
            raise ModelCallError(f"精排接口返回格式异常：{resp.text[:200]}") from e

        results = [
            {"index": int(r["index"]), "relevance_score": float(r["relevance_score"])}
            for r in raw
        ]
        results.sort(key=lambda r: r["relevance_score"], reverse=True)
        usage = payload.get("usage") or {}
        return results, {"total_tokens": usage.get("total_tokens", 0)}
