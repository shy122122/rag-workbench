"""流程编排：建库 ingest() 与问答 query() 两条可视化流水线。"""
from __future__ import annotations

import asyncio
import math
import time
from pathlib import Path
from typing import Awaitable, Callable

import config
from . import chunker, cleaner, loader
from .embeddings import Embedder
from .errors import ApiKeyMissing
from .events import sse, step_event
from .llm import LLM
from .store import VectorStore, new_doc_id

# 步骤定义：(键, 中文标签)
INGEST_STAGES = [
    ("parse", "解析文档"),
    ("clean", "数据清洗"),
    ("chunk", "文本切分"),
    ("embed", "文本向量化"),
    ("index", "写入索引"),
]
QUERY_STAGES = [
    ("q_embed", "问题向量化"),
    ("retrieve", "向量检索 Top-K"),
    ("context", "组装上下文"),
    ("generate", "流式生成回答"),
]
MATCH_STAGES = [
    ("m_embed", "文本向量化"),
    ("m_retrieve", "知识库匹配 Top-K"),
    ("m_prompt", "组装解读请求"),
    ("m_interpret", "模型解读回答"),
]
STAGE_LABELS = dict(INGEST_STAGES + QUERY_STAGES + MATCH_STAGES)

SYSTEM_PROMPT = (
    "你是一个严谨的中文知识问答助手。请严格依据下面提供的【参考资料】回答用户问题，"
    "不要编造资料中没有的信息；当资料不足以回答时，请明确说明资料中没有相关内容。\n"
    "回答时在引用到某条资料的语句后标注来源编号，例如 [1][2]，编号必须与参考资料条目标号一致；"
    "如未引用任何资料则不要标注编号。请用简体中文、条理清晰地作答。"
)

MATCH_SYSTEM_PROMPT = (
    "你是一个 RAG 内容匹配解读助手。用户提供了一段【待匹配文本】，系统已从知识库检索到若干"
    "【参考资料】（按相似度从高到低编号 [1][2]...）。请解读：待匹配文本与哪些资料主题相关、"
    "契合点在何处，资料能对这段文本起到补充或支撑作用的地方；若检索到的资料与文本关系不大或没有"
    "检索到资料，请如实说明，不要编造。引用资料时用 [n] 标注，编号必须与参考资料条目标号一致。"
    "请用简体中文、条理清晰地作答。"
)


def _now() -> int:
    return time.perf_counter()


def get_embedder(cfg: dict) -> Embedder:
    key = config.get_api_key()
    if not key:
        raise ApiKeyMissing("尚未配置阿里云百炼 API Key，请先在「设置」中填写。")
    return Embedder(cfg["base_url"], key, cfg["embedding_model"], int(cfg["dimensions"]))


def get_llm(cfg: dict) -> LLM:
    key = config.get_api_key()
    if not key:
        raise ApiKeyMissing("尚未配置阿里云百炼 API Key，请先在「设置」中填写。")
    return LLM(cfg["base_url"], key, cfg["llm_model"])


# ---------------------------------------------------------------- 建库流程
async def ingest_document(
    file_path: str | Path, store: VectorStore, cfg: dict
) -> tuple[list[dict], str, str]:
    """处理单个文档，返回 (步骤转写, doc_id, 错误信息)。"""
    file_path = Path(file_path)
    doc_name = file_path.name
    doc_id = new_doc_id()
    transcript: list[dict] = []

    def finish_stage(stage: str, ok: bool, detail: dict, ms: int) -> None:
        transcript.append(
            {
                "stage": stage,
                "label": STAGE_LABELS[stage],
                "ok": ok,
                "ms": ms,
                "detail": detail,
            }
        )

    # 1) 解析
    t0 = _now()
    try:
        res = loader.load_text(file_path)
        text, meta = res["text"], res["meta"]
    except loader.LoadError as e:
        finish_stage("parse", False, {"error": str(e)}, int((_now() - t0) * 1000))
        return transcript, doc_id, str(e)
    finish_stage(
        "parse",
        True,
        {
            "source": meta["source"],
            "chars": meta["chars"],
            "pages": meta.get("pages"),
            "preview": text[:300],
        },
        int((_now() - t0) * 1000),
    )

    # 1.5) 数据清洗（可选）
    t0 = _now()
    noise_on = bool(cfg["clean_noise"])
    dedup_on = bool(cfg["clean_dedup"])
    clean_ms = int((_now() - t0) * 1000)
    if cfg["clean_enabled"]:
        try:
            cleaned, report = cleaner.clean(text, noise=noise_on, dedup=dedup_on)
            finish_stage(
                "clean",
                True,
                {
                    "enabled": True,
                    "noise": noise_on,
                    "dedup": dedup_on,
                    "message": f"删除 {report['removed_lines']} 行 / {report['removed_chars']} 字",
                    **report,
                },
                clean_ms,
            )
            text = cleaned
        except Exception as e:  # noqa: BLE001
            finish_stage(
                "clean",
                False,
                {"enabled": True, "error": f"清洗异常，已跳过：{e}"},
                clean_ms,
            )
    else:
        finish_stage(
            "clean",
            True,
            {
                "enabled": False,
                "noise": noise_on,
                "dedup": dedup_on,
                "message": "未启用数据清洗（可在「设置」中开启，改动后需重新建库）",
            },
            clean_ms,
        )

    # 2) 切分
    chunk_size = max(50, int(cfg["chunk_size"]))
    overlap = min(max(0, int(cfg["chunk_overlap"])), chunk_size - 1)
    t0 = _now()
    chunks = chunker.split_text(text, chunk_size, overlap)
    if not chunks:
        finish_stage("chunk", False, {"error": "切分结果为空"}, int((_now() - t0) * 1000))
        return transcript, doc_id, "切分结果为空"
    samples = [
        {"chunk_index": i, "len": len(c), "head": c[:140]} for i, c in enumerate(chunks[:12])
    ]
    finish_stage(
        "chunk",
        True,
        {"count": len(chunks), "chunk_size": chunk_size, "overlap": overlap, "samples": samples},
        int((_now() - t0) * 1000),
    )

    # 3) 向量化
    items = [
        {"text": c, "chunk_index": i, "chars": len(c)} for i, c in enumerate(chunks)
    ]
    t0 = _now()
    try:
        embedder = get_embedder(cfg)

        def on_batch(done: int) -> None:
            pass  # 批间无需事件，量不大，直接异步并发

        vectors, usage = await embedder.embed_texts(
            (it["text"] for it in items), on_batch=on_batch
        )
    except ApiKeyMissing as e:
        finish_stage("embed", False, {"error": str(e)}, int((_now() - t0) * 1000))
        return transcript, doc_id, str(e)
    except Exception as e:  # noqa: BLE001
        finish_stage("embed", False, {"error": f"向量化失败：{e}"}, int((_now() - t0) * 1000))
        return transcript, doc_id, f"向量化失败：{e}"
    finish_stage(
        "embed",
        True,
        {
            "model": cfg["embedding_model"],
            "dimensions": cfg["dimensions"],
            "batches": usage["batches"],
            "tokens": usage["total_tokens"],
            "vector_head": [round(float(x), 4) for x in vectors[0][:6]] if vectors else [],
        },
        int((_now() - t0) * 1000),
    )

    # 4) 入库
    t0 = _now()
    new_chunks = store.add_doc(doc_id, doc_name, items, vectors)
    stats = store.stats()
    finish_stage(
        "index",
        True,
        {"doc_name": doc_name, "new_chunks": new_chunks, **stats},
        int((_now() - t0) * 1000),
    )
    return transcript, doc_id, ""


# ---------------------------------------------------------------- 问答流程(SSE)
async def run_query(
    question: str,
    store: VectorStore,
    cfg: dict,
    emit: Callable[[str], Awaitable[None]],
    top_k: int | None = None,
    min_score: float | None = None,
) -> None:
    """按步骤跑问答，通过 emit 推送 SSE。top_k/min_score 缺省取 config。"""
    started = _now()
    answer_parts: list[str] = []

    async def emit_step(stage: str, status: str, detail: dict | None = None) -> None:
        await emit(step_event(stage, status, detail))

    # 1) 问题向量化
    t0 = _now()
    try:
        embedder = get_embedder(cfg)
        await emit_step("q_embed", "run")
        qvec, usage = await embedder.embed_texts([question])
        ms = int((_now() - t0) * 1000)
        await emit_step(
            "q_embed",
            "done",
            {
                "model": cfg["embedding_model"],
                "dimensions": cfg["dimensions"],
                "ms": ms,
                "tokens": usage["total_tokens"],
                "vector_head": [round(float(x), 4) for x in qvec[0][:6]],
            },
        )
    except ApiKeyMissing as e:
        await emit_step("q_embed", "error", {"error": str(e)})
        return
    except Exception as e:  # noqa: BLE001
        await emit_step("q_embed", "error", {"error": f"向量化失败：{e}"})
        return

    # 2) 检索 Top-K
    k = max(1, int(top_k if top_k is not None else cfg["top_k"]))
    mscore = float(min_score if min_score is not None else cfg["retrieval_min_score"])
    t0 = _now()
    hits = store.search(qvec[0], top_k=k, min_score=mscore)
    ms = int((_now() - t0) * 1000)
    stats = store.stats()
    above = store.count_above(qvec[0], mscore)
    await emit_step(
        "retrieve",
        "done",
        {
            "top_k": k,
            "min_score": mscore,
            "above": above,  # 高于阈值的命中总数（用于提示被过滤了多少）
            "total_chunks": stats["chunks"],
            "dim": stats["dim"],
            "hits": hits,
            "ms": ms,
        },
    )

    # 3) 组装上下文
    t0 = _now()
    context = _build_context(hits)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"【参考资料】\n{context}\n\n【用户问题】\n{question}"},
    ]
    if not hits:
        messages[-1]["content"] = (
            "知识库中没有检索到相关资料。请直接说明知识库中缺少相关内容，"
            "并基于常识尽量简要作答。\n\n【用户问题】\n" + question
        )
    ms = int((_now() - t0) * 1000)
    est_tokens = sum(_est_tokens(m["content"]) for m in messages)
    await emit_step(
        "context",
        "done",
        {"messages": messages, "hits_used": len(hits), "est_tokens": est_tokens, "ms": ms},
    )

    # 4) 流式生成
    t0 = _now()
    await emit_step("generate", "run")

    async def on_token(tok: str) -> None:
        answer_parts.append(tok)
        await emit(sse("token", {"text": tok}))

    try:
        llm = get_llm(cfg)
        usage = await llm.stream(messages, on_token=on_token)
    except ApiKeyMissing as e:
        await emit_step("generate", "error", {"error": str(e)})
        return
    except Exception as e:  # noqa: BLE001
        await emit_step("generate", "error", {"error": f"生成失败：{e}"})
        return
    ms = int((_now() - t0) * 1000)
    answer = "".join(answer_parts)
    await emit_step(
        "generate",
        "done",
        {"ms": ms, "chars": len(answer), "usage": usage, "answer": answer},
    )

    # 完成
    await emit(
        sse(
            "done",
            {
                "elapsed_ms": int((_now() - started) * 1000),
                "question": question,
                "answer": answer,
                "references": _refs(hits),
            },
        )
    )


async def run_match(
    text: str,
    store: VectorStore,
    cfg: dict,
    emit: Callable[[str], Awaitable[None]],
    with_llm: bool = False,
    top_k: int | None = None,
    min_score: float | None = None,
) -> None:
    """内容匹配：把一段文本拿去和知识库做相似度匹配，可选模型解读。

    阶段 m_embed → m_retrieve →（with_llm）m_prompt → m_interpret。
    """
    started = _now()
    k = max(1, int(top_k if top_k is not None else cfg["top_k"]))
    mscore = float(min_score if min_score is not None else cfg["retrieval_min_score"])
    answer_parts: list[str] = []

    async def emit_step(stage: str, status: str, detail: dict | None = None) -> None:
        await emit(step_event(stage, status, detail))

    # 1) 待匹配文本向量化
    t0 = _now()
    try:
        embedder = get_embedder(cfg)
        await emit_step("m_embed", "run")
        qvec, usage = await embedder.embed_texts([text])
        ms = int((_now() - t0) * 1000)
        await emit_step(
            "m_embed",
            "done",
            {
                "model": cfg["embedding_model"],
                "dimensions": cfg["dimensions"],
                "ms": ms,
                "tokens": usage["total_tokens"],
                "vector_head": [round(float(x), 4) for x in qvec[0][:6]],
            },
        )
    except ApiKeyMissing as e:
        await emit_step("m_embed", "error", {"error": str(e)})
        return
    except Exception as e:  # noqa: BLE001
        await emit_step("m_embed", "error", {"error": f"向量化失败：{e}"})
        return

    # 2) 知识库匹配 Top-K
    t0 = _now()
    hits = store.search(qvec[0], top_k=k, min_score=mscore)
    ms = int((_now() - t0) * 1000)
    stats = store.stats()
    above = store.count_above(qvec[0], mscore)
    await emit_step(
        "m_retrieve",
        "done",
        {
            "top_k": k,
            "min_score": mscore,
            "above": above,
            "total_chunks": stats["chunks"],
            "dim": stats["dim"],
            "hits": hits,
            "ms": ms,
        },
    )

    if not with_llm:
        await emit(
            sse(
                "done",
                {
                    "elapsed_ms": int((_now() - started) * 1000),
                    "mode": "match",
                    "text": text,
                    "answer": None,
                    "references": _refs(hits),
                    "hits": hits,
                },
            )
        )
        return

    # 3) 组装解读请求
    t0 = _now()
    context = _build_context(hits)
    messages = [
        {"role": "system", "content": MATCH_SYSTEM_PROMPT},
        {"role": "user", "content": f"【待匹配文本】\n{text}\n\n【参考资料】\n{context}"},
    ]
    if not hits:
        messages[-1]["content"] = (
            "知识库中没有检索到与这段文本相关的资料。请如实说明未匹配到资料，"
            "不要编造关系。\n\n【待匹配文本】\n" + text
        )
    ms = int((_now() - t0) * 1000)
    est_tokens = sum(_est_tokens(m["content"]) for m in messages)
    await emit_step(
        "m_prompt",
        "done",
        {"text": text, "messages": messages, "hits_used": len(hits),
         "est_tokens": est_tokens, "ms": ms},
    )

    # 4) 模型解读（流式）
    t0 = _now()
    await emit_step("m_interpret", "run")

    async def on_token(tok: str) -> None:
        answer_parts.append(tok)
        await emit(sse("token", {"text": tok}))

    try:
        llm = get_llm(cfg)
        usage = await llm.stream(messages, on_token=on_token)
    except ApiKeyMissing as e:
        await emit_step("m_interpret", "error", {"error": str(e)})
        return
    except Exception as e:  # noqa: BLE001
        await emit_step("m_interpret", "error", {"error": f"解读失败：{e}"})
        return
    ms = int((_now() - t0) * 1000)
    answer = "".join(answer_parts)
    await emit_step(
        "m_interpret",
        "done",
        {"ms": ms, "chars": len(answer), "usage": usage, "answer": answer},
    )

    # 完成
    await emit(
        sse(
            "done",
            {
                "elapsed_ms": int((_now() - started) * 1000),
                "mode": "match",
                "text": text,
                "answer": answer,
                "references": _refs(hits),
                "hits": hits,
            },
        )
    )


def _refs(hits: list[dict]) -> list[dict]:
    return [
        {"rank": h["rank"], "doc_name": h["doc_name"], "id": h["id"]}
        for h in hits
    ]


def _build_context(hits: list[dict]) -> str:
    """把命中的分块拼成带 [n] 编号的资料文本。"""
    blocks = []
    for h in hits:
        blocks.append(f"[{h['rank']}] 来源《{h['doc_name']}》(片段 {h['chunk_index'] + 1})\n{h['text']}")
    return "\n\n".join(blocks)


def _est_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 1.5))  # 中英混排粗略估算
