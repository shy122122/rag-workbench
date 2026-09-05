"""百炼流式生成封装：走 OpenAI 兼容接口 /v1/chat/completions。"""
from __future__ import annotations

from typing import Awaitable, Callable

from openai import AsyncOpenAI

from .errors import ModelCallError

TokenHandler = Callable[[str], Awaitable[None]]


class LLM:
    def __init__(self, base_url: str, api_key: str, model: str, temperature: float = 0.3):
        self.model = model
        self.temperature = temperature
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    async def stream(
        self, messages: list[dict], on_token: TokenHandler | None = None
    ) -> dict:
        """流式生成。on_token 每次收到增量调用。返回用量统计。"""
        try:
            stream = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                temperature=self.temperature,
            )
        except Exception as e:  # openai.AuthenticationError/APIError 等
            raise ModelCallError(f"生成接口调用失败：{e}") from e

        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                if on_token:
                    await on_token(delta.content)
            if chunk.usage:
                usage = {
                    "prompt_tokens": chunk.usage.prompt_tokens or 0,
                    "completion_tokens": chunk.usage.completion_tokens or 0,
                }
        return usage
