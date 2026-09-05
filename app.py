"""FastAPI 入口：静态页面 + 知识库管理 + 建库/问答路由。"""
from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path

import config
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from rag import pipeline
from rag.errors import ApiKeyMissing
from rag.events import sse
from rag.store import VectorStore

BASE = config.BASE_DIR
STATIC_DIR = BASE / "static"
KB = VectorStore(BASE / "data" / "kb")

app = FastAPI(title="RAG 可视化工作台")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health() -> dict:
    return {"ok": True, "stats": KB.stats()}


# ---------------------------------------------------------------- 配置
@app.get("/api/config")
async def get_config() -> dict:
    return {"config": config.load_config(), "has_key": config.has_api_key()}


@app.post("/api/config")
async def post_config(payload: dict) -> dict:
    cfg = config.load_config()
    api_key = payload.get("api_key", "")
    for k in payload:
        if k in config.DEFAULT_CONFIG:
            cfg[k] = payload[k]
    config.save_config(cfg)
    if api_key:
        config.save_api_key(api_key)
    return {"config": config.load_config(), "has_key": config.has_api_key()}


@app.post("/api/test-key")
async def test_key() -> dict:
    cfg = config.load_config()
    if not config.has_api_key():
        return {"ok": False, "message": "尚未填写 API Key"}
    try:
        embedder = pipeline.get_embedder(cfg)
        await embedder.embed_texts(["连通性测试"])
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "message": f"调用失败：{e}"}
    return {"ok": True, "message": "连通成功，API Key 可用"}


# ---------------------------------------------------------------- 知识库
@app.get("/api/knowledge/list")
async def list_knowledge() -> dict:
    return {"docs": KB.list_docs(), "stats": KB.stats()}


@app.delete("/api/knowledge")
async def clear_knowledge() -> dict:
    KB.clear()
    return {"ok": True, "stats": KB.stats()}


@app.delete("/api/knowledge/{doc_id}")
async def delete_doc(doc_id: str) -> dict:
    removed = KB.remove_doc(doc_id)
    return {"ok": True, "removed": removed, "stats": KB.stats()}


def _safe_name(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z一-鿿._-]+", "_", Path(name).name)


@app.post("/api/knowledge/upload")
async def upload_knowledge(file: UploadFile = File(...)) -> dict:
    name = _safe_name(file.filename or "upload.bin")
    tmp = BASE / "data" / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    path = tmp / f"{time.time_ns()}-{name}"
    content = await file.read()
    if len(content) > 100 * 1024 * 1024:
        return JSONResponse({"ok": False, "message": "文件超过 100MB 限制"}, status_code=400)
    path.write_bytes(content)
    try:
        transcript, doc_id, err = await pipeline.ingest_document(path, KB, config.load_config())
        return {
            "ok": not err,
            "doc_name": name,
            "doc_id": doc_id,
            "error": err or None,
            "transcript": transcript,
        }
    finally:
        path.unlink(missing_ok=True)


@app.post("/api/knowledge/load-examples")
async def load_examples() -> dict:
    ex_dir = BASE / "examples"
    files = sorted(p for p in ex_dir.iterdir() if p.is_file())
    # 幂等：移除同名旧文档，避免重复点击堆出重复内容
    existing = {d["doc_name"] for d in KB.list_docs()}
    for d in KB.list_docs():
        if d["doc_name"] in {p.name for p in files}:
            KB.remove_doc(d["doc_id"])
    docs = []
    fail = 0
    for p in files:
        replaced = p.name in existing
        try:
            transcript, doc_id, err = await pipeline.ingest_document(p, KB, config.load_config())
        except Exception as e:  # noqa: BLE001
            transcript, doc_id, err = [], "", str(e)
        docs.append(
            {
                "doc_name": p.name,
                "replaced": replaced,
                "ok": not err,
                "error": err or None,
                "transcript": transcript,
            }
        )
        if err:
            fail += 1
    return {"ok": fail == 0, "docs": docs, "stats": KB.stats()}


# ---------------------------------------------------------------- 问答(SSE)
def _num(body: dict, key: str, cast):
    v = body.get(key)
    if v is None or v == "":
        return None
    try:
        return cast(v)
    except (TypeError, ValueError):
        return None


def _to_bool(v, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _stream(run) -> StreamingResponse:
    """把 run(emit) 的异步流程包装成 SSE 流。run 是接收 emit 的可调用对象。"""

    async def gen():
        queue: asyncio.Queue = asyncio.Queue()

        async def emit(s: str) -> None:
            await queue.put(s)

        runner = asyncio.create_task(run(emit))

        async def drive() -> None:
            try:
                await runner
            except Exception as e:  # noqa: BLE001
                try:
                    await emit(sse("error", {"message": f"流程异常：{e}"}))
                except Exception:  # noqa: BLE001
                    pass
            finally:
                await queue.put(None)

        watcher = asyncio.create_task(drive())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item
        finally:
            watcher.cancel()
            if not runner.done():
                runner.cancel()

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/query")
async def query(body: dict):
    question = (body.get("question") or "").strip()
    if not question:
        return JSONResponse({"ok": False, "message": "问题不能为空"}, status_code=400)
    cfg = config.load_config()
    return _stream(
        lambda emit: pipeline.run_query(
            question,
            KB,
            cfg,
            emit,
            top_k=_num(body, "top_k", int),
            min_score=_num(body, "min_score", float),
        )
    )


@app.post("/api/match")
async def match(body: dict):
    text = (body.get("text") or "").strip()
    if not text:
        return JSONResponse({"ok": False, "message": "待匹配文本不能为空"}, status_code=400)
    cfg = config.load_config()
    return _stream(
        lambda emit: pipeline.run_match(
            text,
            KB,
            cfg,
            emit,
            with_llm=_to_bool(body.get("with_llm")),
            top_k=_num(body, "top_k", int),
            min_score=_num(body, "min_score", float),
        )
    )
