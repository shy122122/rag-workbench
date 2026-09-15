"""百炼流式生成封装：走 OpenAI 兼容接口 /v1/chat/completions。"""
from __future__ import annotations

from typing import Awaitable, Callable

from .clients import get_openai_client
from .errors import ModelCallError

TokenHandler = Callable[[str], Awaitable[None]]


class LLM:
    def __init__(self, base_url: str, api_key: str, model: str, temperature: float = 0.3):
        self.model = model
        self.temperature = temperature
        self._client = get_openai_client(base_url, api_key)

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
                # 不带这个，流式响应里的 usage 全是 0，界面上的输入/输出 tokens 就永远是 0
                stream_options={"include_usage": True},
            )
        except Exception as e:  # openai.AuthenticationError/APIError 等
            raise ModelCallError(f"生成接口调用失败：{e}") from e

        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        # 必须用 async with：中途被取消（前端「停止」按钮）时如果没关流，
        # 底层 HTTP 响应生成器会悬挂到解释器退出才被回收，抛一串
        # "generator didn't stop after athrow()" 报错，并泄漏连接。
        async with stream:
            async for chunk in stream:
                # usage 只在最后一个分块上，而那个分块的 choices 是空列表，
                # 所以必须先取 usage 再判断 choices，否则用量永远是 0。
                if chunk.usage:
                    usage = {
                        "prompt_tokens": chunk.usage.prompt_tokens or 0,
                        "completion_tokens": chunk.usage.completion_tokens or 0,
                    }
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    if on_token:
                        await on_token(delta.content)
        return usage
