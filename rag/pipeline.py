"""流程编排：建库 ingest() 与问答 query() 两条可视化流水线。"""
from __future__ import annotations

import asyncio
import logging
import math
import re
import time
from pathlib import Path
from typing import Awaitable, Callable

import config
from . import chunker, cleaner, loader
from .embeddings import Embedder
from .errors import ApiKeyMissing, DimMismatch
from .events import sse, step_event
from .llm import LLM
from .rerank import Reranker
from .store import VectorStore, new_doc_id

log = logging.getLogger("rag.pipeline")

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
    ("retrieve", "向量粗排召回"),
    ("rerank", "精排重排序"),
    ("context", "组装上下文"),
    ("generate", "流式生成回答"),
]
MATCH_STAGES = [
    ("m_embed", "文本向量化"),
    ("m_retrieve", "向量粗排召回"),
    ("m_rerank", "精排重排序"),
    ("m_prompt", "组装解读请求"),
    ("m_interpret", "模型解读回答"),
]
STAGE_LABELS = dict(INGEST_STAGES + QUERY_STAGES + MATCH_STAGES)

# 粗排之外再多带回多少条「被 Top-K 截断」的命中（只带摘要，用于让用户看清被丢掉了什么）
TRUNCATED_EXTRA = 20

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

# 带历史问答时追加到系统提示词后面。历史里的 [n] 指的是「上一轮」的参考资料，
# 直接留着会诱导模型在新回答里沿用旧编号、指到错误来源，所以拼进上下文前先抹掉。
MULTI_TURN_NOTE = (
    "\n\n【多轮对话】下面带有若干轮历史问答，它们只用于理解用户问题中的指代和上下文，"
    "回答当前问题时请只依据本轮【参考资料】；引用编号 [n] 一律只指本轮【参考资料】的条目号。"
)

_CITE_RE = re.compile(r"\[\d+\]")


def _strip_cites(text: str) -> str:
    """抹掉历史回答里的 [n] 引用标记，避免误导当前这轮的引用编号。"""
    return _CITE_RE.sub("", text)


def _now() -> int:
    return time.perf_counter()


def _get_key() -> str:
    key = config.get_api_key()
    if not key:
        raise ApiKeyMissing("尚未配置阿里云百炼 API Key，请先在「设置」中填写。")
    return key


def get_embedder(cfg: dict) -> Embedder:
    return Embedder(cfg["base_url"], _get_key(), cfg["embedding_model"], int(cfg["dimensions"]))


def get_llm(cfg: dict) -> LLM:
    return LLM(cfg["base_url"], _get_key(), cfg["llm_model"])


def get_reranker(cfg: dict) -> Reranker:
    base = cfg.get("native_base_url") or "https://dashscope.aliyuncs.com/api/v1"
    return Reranker(base, _get_key(), cfg["rerank_model"])


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

    # 1) 解析（解析 PDF 是同步阻塞的，丢到线程里免得卡住事件循环）
    t0 = _now()
    try:
        res = await asyncio.to_thread(loader.load_text, file_path)
        text, meta = res["text"], res["meta"]
    except loader.LoadError as e:
        log.warning("解析失败：%s -> %s", doc_name, e)
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
    if cfg["clean_enabled"]:
        try:
            cleaned, report = await asyncio.to_thread(
                cleaner.clean, text, noise_on, dedup_on
            )
            text = cleaned
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
                int((_now() - t0) * 1000),
            )
        except Exception as e:  # noqa: BLE001
            finish_stage(
                "clean",
                False,
                {"enabled": True, "error": f"清洗异常，已跳过：{e}"},
                int((_now() - t0) * 1000),
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
            int((_now() - t0) * 1000),
        )

    # 2) 切分（纯 CPU，同样丢线程）
    t0 = _now()
    chunk_size = max(50, int(cfg["chunk_size"]))
    overlap = min(max(0, int(cfg["chunk_overlap"])), chunk_size - 1)
    strategy = str(cfg.get("chunk_strategy") or chunker.CHUNK_STRATEGY_RECURSIVE)
    structure_level = None
    if strategy == chunker.CHUNK_STRATEGY_STRUCTURE:
        structure_level = max(1, min(6, int(cfg.get("structure_level") or 2)))
        chunks = await asyncio.to_thread(
            chunker.split_by_structure, text, chunk_size, overlap, structure_level
        )
    else:
        strategy = chunker.CHUNK_STRATEGY_RECURSIVE
        chunks = await asyncio.to_thread(chunker.split_text, text, chunk_size, overlap)
    if not chunks:
        finish_stage("chunk", False, {"error": "切分结果为空"}, int((_now() - t0) * 1000))
        return transcript, doc_id, "切分结果为空"
    samples = [
        {"chunk_index": i, "len": len(c), "head": c[:140]} for i, c in enumerate(chunks[:12])
    ]
    finish_stage(
        "chunk",
        True,
        {
            "count": len(chunks),
            "strategy": strategy,
            "structure_level": structure_level,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "samples": samples,
        },
        int((_now() - t0) * 1000),
    )

    # 3) 向量化
    items = [
        {"text": c, "chunk_index": i, "chars": len(c)} for i, c in enumerate(chunks)
    ]
    t0 = _now()
    try:
        embedder = get_embedder(cfg)
        vectors, usage = await embedder.embed_texts((it["text"] for it in items))
    except ApiKeyMissing as e:
        finish_stage("embed", False, {"error": str(e)}, int((_now() - t0) * 1000))
        return transcript, doc_id, str(e)
    except Exception as e:  # noqa: BLE001
        log.exception("向量化失败：%s（%d 块）", doc_name, len(items))
        finish_stage("embed", False, {"error": f"向量化失败：{e}"}, int((_now() - t0) * 1000))
        return transcript, doc_id, f"向量化失败：{e}"
    finish_stage(
        "embed",
        True,
        {
            "model": cfg["embedding_model"],
            "dimensions": cfg["dimensions"],
            "batches": usage["batches"],
            "cached": usage["cached"],
            "tokens": usage["total_tokens"],
            "vector_head": [round(float(x), 4) for x in vectors[0][:6]] if vectors else [],
        },
        int((_now() - t0) * 1000),
    )

    # 4) 入库（写盘全量序列化，丢线程）
    t0 = _now()
    new_chunks = await asyncio.to_thread(store.add_doc, doc_id, doc_name, items, vectors)
    stats = store.stats()
    finish_stage(
        "index",
        True,
        {"doc_name": doc_name, "new_chunks": new_chunks, **stats},
        int((_now() - t0) * 1000),
    )
    return transcript, doc_id, ""


# ---------------------------------------------------------------- 共享检索段
def _rerank_items(hits: list[dict], results: list[dict]) -> list[dict]:
    """把 rerank 结果贴回原文命中，带上「粗排名次」以便前端做前后对照。"""
    items = []
    for new_rank, r in enumerate(results, 1):
        src = hits[r["index"]]
        items.append(
            {
                "rank": new_rank,
                "prev_rank": src["rank"],
                "delta": src["rank"] - new_rank,  # >0 上升，<0 下降，0 不变
                "score": round(r["relevance_score"], 4),
                "cosine_score": src["score"],
                "id": src["id"],
                "doc_id": src["doc_id"],
                "doc_name": src["doc_name"],
                "chunk_index": src["chunk_index"],
                "chars": src["chars"],
                "text": src["text"],
            }
        )
    return items


async def _retrieve_and_rerank(
    *,
    query_text: str,
    qvec,
    store: VectorStore,
    cfg: dict,
    top_k: int,
    min_score: float,
    emit_step,
    retrieve_stage: str,
    rerank_stage: str,
    doc_ids: set[str] | None = None,
) -> tuple[list[dict], str | None]:
    """共享的「向量粗排 → rerank 精排」两段式检索。返回 (最终命中, 错误信息)。"""
    rerank_on = bool(cfg.get("rerank_enabled"))
    recall_k = max(top_k, int(cfg.get("recall_k") or top_k)) if rerank_on else top_k

    # --- 粗排：向量检索 ---
    t0 = _now()
    try:
        hits, truncated, above = store.search(
            qvec, top_k=recall_k, min_score=min_score, extra=TRUNCATED_EXTRA,
            doc_ids=doc_ids,
        )
    except DimMismatch as e:
        await emit_step(retrieve_stage, "error", {"error": str(e)})
        return [], str(e)
    stats = store.stats()
    await emit_step(
        retrieve_stage,
        "done",
        {
            "top_k": recall_k,
            "min_score": min_score,
            "above": above,  # 高于阈值的命中总数（用于提示过滤掉了多少）
            "total_chunks": stats["chunks"],
            "dim": stats["dim"],
            "hits": hits,
            "truncated": truncated,  # Top-K 之外还有什么
            # 本次实际检索的范围：None=全库。留档用，便于回看某一轮是在哪些文档上跑的
            "scope_docs": None if doc_ids is None else len(doc_ids),
            "total_docs": stats["docs"],
            "ms": int((_now() - t0) * 1000),
        },
    )

    # --- 精排：rerank ---
    t0 = _now()
    if not rerank_on:
        await emit_step(
            rerank_stage,
            "done",
            {"enabled": False, "message": "未启用精排（可在「设置」中开启，开启后先粗排召回再精排）",
             "ms": int((_now() - t0) * 1000)},
        )
        return hits[:top_k], None

    if not hits:
        await emit_step(
            rerank_stage,
            "done",
            {"enabled": True, "model": cfg["rerank_model"], "items": [], "dropped": [],
             "reason": "粗排没有命中，无需精排", "ms": int((_now() - t0) * 1000)},
        )
        return [], None

    await emit_step(rerank_stage, "run")
    try:
        reranker = get_reranker(cfg)
        results, usage = await reranker.rerank(
            query_text, [h["text"] for h in hits], top_n=top_k
        )
    except ApiKeyMissing as e:
        await emit_step(rerank_stage, "error", {"error": str(e)})
        return [], str(e)
    except Exception as e:  # noqa: BLE001
        log.exception("精排失败（候选 %d 条）", len(hits))
        await emit_step(rerank_stage, "error", {"error": f"精排失败：{e}"})
        return [], f"精排失败：{e}"

    items = _rerank_items(hits, results)
    final_ids = {it["id"] for it in items}
    dropped = [
        {"prev_rank": h["rank"], "score": h["score"], "doc_name": h["doc_name"],
         "chunk_index": h["chunk_index"]}
        for h in hits
        if h["rank"] <= top_k and h["id"] not in final_ids
    ]
    await emit_step(
        rerank_stage,
        "done",
        {
            "enabled": True,
            "model": cfg["rerank_model"],
            "recall_k": len(hits),
            "top_k": top_k,
            "tokens": usage["total_tokens"],
            "items": items,  # 精排后，含 prev_rank / delta 供前后对照
            "dropped": dropped,  # 粗排进了 Top-K、精排被挤出去的
            "changed": sum(1 for it in items if it["delta"] != 0),
            "ms": int((_now() - t0) * 1000),
        },
    )
    return items, None


def _neighbor_expanded_hits(hits: list[dict], store: VectorStore, window: int) -> list[dict]:
    """给每个命中块挂上左右邻块文本（small-to-big）。"""
    if window <= 0:
        return hits
    out = []
    for h in hits:
        rows = store.neighbors(h["doc_id"], h["chunk_index"], window)
        parts, neighbors = [], []
        for m in rows:
            if m["chunk_index"] == h["chunk_index"]:
                parts.append(m["text"])
                continue
            neighbors.append({"chunk_index": m["chunk_index"], "chars": m["chars"]})
            if m["chunk_index"] < h["chunk_index"]:
                parts.insert(0, m["text"])
            else:
                parts.append(m["text"])
        h = dict(h)
        h["context_text"] = "\n".join(parts)
        h["neighbors"] = sorted(neighbors, key=lambda n: n["chunk_index"])
        out.append(h)
    return out


# ---------------------------------------------------------------- 问答流程(SSE)
async def run_query(
    question: str,
    store: VectorStore,
    cfg: dict,
    emit: Callable[[str], Awaitable[None]],
    top_k: int | None = None,
    min_score: float | None = None,
    history: list[dict] | None = None,
) -> None:
    """按步骤跑问答，通过 emit 推送 SSE。top_k/min_score 缺省取 config。

    history 是最近几轮 [{"question", "answer"}]，仅用于理解指代，不参与检索。
    """
    started = _now()
    history = history or []
    answer_parts: list[str] = []

    async def emit_step(stage: str, status: str, detail: dict | None = None) -> None:
        await emit(step_event(stage, status, detail))

    # 1) 问题向量化
    t0 = _now()
    try:
        embedder = get_embedder(cfg)
        await emit_step("q_embed", "run")
        qvec, usage = await embedder.embed_texts([question])
        await emit_step(
            "q_embed",
            "done",
            {
                "model": cfg["embedding_model"],
                "dimensions": cfg["dimensions"],
                "ms": int((_now() - t0) * 1000),
                "tokens": usage["total_tokens"],
                "cached": usage["cached"],
                "vector_head": [round(float(x), 4) for x in qvec[0][:6]],
            },
        )
    except ApiKeyMissing as e:
        await emit_step("q_embed", "error", {"error": str(e)})
        return
    except Exception as e:  # noqa: BLE001
        await emit_step("q_embed", "error", {"error": f"向量化失败：{e}"})
        return

    # 2)+3) 粗排 → 精排
    k = max(1, int(top_k if top_k is not None else cfg["top_k"]))
    mscore = float(min_score if min_score is not None else cfg["retrieval_min_score"])
    hits, err = await _retrieve_and_rerank(
        query_text=question, qvec=qvec[0], store=store, cfg=cfg, top_k=k,
        min_score=mscore, emit_step=emit_step,
        retrieve_stage="retrieve", rerank_stage="rerank",
    )
    if err:
        return

    # 3) 组装上下文（命中块 + 邻块扩展）
    t0 = _now()
    window = max(0, int(cfg.get("neighbor_window") or 0))
    expanded = _neighbor_expanded_hits(hits, store, window)
    context = _build_context(expanded)
    if hits:
        current = f"【参考资料】\n{context}\n\n【用户问题】\n{question}"
    else:
        current = (
            "知识库中没有检索到相关资料。请直接说明知识库中缺少相关内容，"
            "并基于常识尽量简要作答。\n\n【用户问题】\n" + question
        )
    messages = [{"role": "system", "content": SYSTEM_PROMPT + (MULTI_TURN_NOTE if history else "")}]
    for h in history:
        messages.append({"role": "user", "content": h["question"]})
        messages.append({"role": "assistant", "content": _strip_cites(h["answer"])})
    messages.append({"role": "user", "content": current})

    est_tokens = sum(_est_tokens(m["content"]) for m in messages)
    await emit_step(
        "context",
        "done",
        {
            "messages": messages,
            "hits_used": len(hits),
            "est_tokens": est_tokens,
            "neighbor_window": window,
            "history_turns": len(history),
            "expanded": [
                {"rank": h["rank"], "doc_name": h["doc_name"], "chunk_index": h["chunk_index"],
                 "chars": h["chars"], "neighbors": h.get("neighbors", []),
                 "context_chars": len(h.get("context_text", h["text"]))}
                for h in expanded
            ],
            "ms": int((_now() - t0) * 1000),
        },
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
        log.exception("生成失败：%s", question[:80])
        await emit_step("generate", "error", {"error": f"生成失败：{e}"})
        return
    answer = "".join(answer_parts)
    await emit_step(
        "generate",
        "done",
        {"ms": int((_now() - t0) * 1000), "chars": len(answer), "usage": usage, "answer": answer},
    )

    elapsed = int((_now() - started) * 1000)
    log.info(
        "问答完成 %.1fs｜命中 %d 条｜上文 %d 轮｜tokens 入 %s / 出 %s｜%s",
        elapsed / 1000, len(hits), len(history),
        usage.get("prompt_tokens"), usage.get("completion_tokens"), question[:60],
    )
    # 完成
    await emit(
        sse(
            "done",
            {
                "elapsed_ms": elapsed,
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
    doc_ids: set[str] | None = None,
) -> None:
    """内容匹配：把一段文本拿去和知识库做相似度匹配，可选模型解读。

    doc_ids 限定参与匹配的文档范围（None = 全库）。
    阶段 m_embed → m_retrieve → m_rerank →（with_llm）m_prompt → m_interpret。
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
        await emit_step(
            "m_embed",
            "done",
            {
                "model": cfg["embedding_model"],
                "dimensions": cfg["dimensions"],
                "ms": int((_now() - t0) * 1000),
                "tokens": usage["total_tokens"],
                "cached": usage["cached"],
                "vector_head": [round(float(x), 4) for x in qvec[0][:6]],
            },
        )
    except ApiKeyMissing as e:
        await emit_step("m_embed", "error", {"error": str(e)})
        return
    except Exception as e:  # noqa: BLE001
        await emit_step("m_embed", "error", {"error": f"向量化失败：{e}"})
        return

    # 2)+3) 粗排 → 精排
    hits, err = await _retrieve_and_rerank(
        query_text=text, qvec=qvec[0], store=store, cfg=cfg, top_k=k,
        min_score=mscore, emit_step=emit_step,
        retrieve_stage="m_retrieve", rerank_stage="m_rerank",
        doc_ids=doc_ids,
    )
    if err:
        return

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

    # 4) 组装解读请求
    t0 = _now()
    window = max(0, int(cfg.get("neighbor_window") or 0))
    expanded = _neighbor_expanded_hits(hits, store, window)
    context = _build_context(expanded)
    messages = [
        {"role": "system", "content": MATCH_SYSTEM_PROMPT},
        {"role": "user", "content": f"【待匹配文本】\n{text}\n\n【参考资料】\n{context}"},
    ]
    if not hits:
        messages[-1]["content"] = (
            "知识库中没有检索到与这段文本相关的资料。请如实说明未匹配到资料，"
            "不要编造关系。\n\n【待匹配文本】\n" + text
        )
    est_tokens = sum(_est_tokens(m["content"]) for m in messages)
    await emit_step(
        "m_prompt",
        "done",
        {"text": text, "messages": messages, "hits_used": len(hits),
         "est_tokens": est_tokens, "neighbor_window": window, "ms": int((_now() - t0) * 1000)},
    )

    # 5) 模型解读（流式）
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
    answer = "".join(answer_parts)
    await emit_step(
        "m_interpret",
        "done",
        {"ms": int((_now() - t0) * 1000), "chars": len(answer), "usage": usage, "answer": answer},
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
    """把命中的分块拼成带 [n] 编号的资料文本；命中块可能已含邻块扩展文本。"""
    blocks = []
    for h in hits:
        body = h.get("context_text") or h["text"]
        blocks.append(
            f"[{h['rank']}] 来源《{h['doc_name']}》(片段 {h['chunk_index'] + 1})\n{body}"
        )
    return "\n\n".join(blocks)


def _est_tokens(text: str) -> int:
    """粗略估算 token：中文约 0.75 token/字，其余约 3.5 字符/token。"""
    cjk = sum(1 for ch in text if "一" <= ch <= "鿿")
    other = len(text) - cjk
    return max(1, math.ceil(cjk * 0.75 + other / 3.5))
