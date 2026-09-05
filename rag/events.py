"""步骤事件模型与 SSE 编码。"""
from __future__ import annotations

import json


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def step_event(stage: str, status: str, data: dict | None = None) -> str:
    return sse("step", {"stage": stage, "status": status, "data": data or {}})
