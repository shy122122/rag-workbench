"""FastAPI 入口：静态页面 + 知识库管理 + 建库/问答路由。"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path

import config
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from rag import clients, logging_setup, pipeline
from rag.errors import StoreCorrupt
from rag.events import sse
from rag.store import VectorStore

BASE = config.BASE_DIR
STATIC_DIR = BASE / "static"

LOG_PATH = logging_setup.setup(BASE)
log = logging.getLogger("rag.app")

KB = VectorStore(BASE / "data" / "kb")

MAX_UPLOAD = 100 * 1024 * 1024  # 单文件上限
_READ_CHUNK = 1024 * 1024  # 边收边写，不要把整个文件攒在内存里


@asynccontextmanager
async def lifespan(_: FastAPI):
    log.info("工作台启动｜日志文件：%s", LOG_PATH or "(仅控制台)")
    st = KB.stats()
    log.info("知识库载入：%s 篇 / %s 块 / %s 维", st["docs"], st["chunks"], st["dim"])
    if st["problems"]:
        log.warning("知识库存在数据问题：%s", "；".join(st["problems"]))
    yield
    await clients.close_all()
    log.info("工作台退出")


app = FastAPI(title="RAG 可视化工作台", lifespan=lifespan)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """记录每个请求。注意 SSE 接口的耗时就到这里为止，真正的流程耗时由 pipeline 记。"""
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = int((time.perf_counter() - t0) * 1000)
    level = logging.WARNING if response.status_code >= 400 else logging.INFO
    log.log(level, "%s %s -> %s (%dms)", request.method, request.url.path,
            response.status_code, ms)
    return response


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
        return {"ok": False, "message": f"向量化接口调用失败：{e}"}

    if not cfg.get("rerank_enabled"):
        return {"ok": True, "message": "连通成功，API Key 可用（精排未启用，已跳过）"}
    try:
        reranker = pipeline.get_reranker(cfg)
        await reranker.rerank("连通性测试", ["测试文档"], top_n=1)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "message": f"向量化正常，但精排接口调用失败：{e}"}
    return {"ok": True, "message": "连通成功，向量化与精排接口均可用"}


# ---------------------------------------------------------------- 知识库
@app.get("/api/knowledge/list")
async def list_knowledge() -> dict:
    return {"docs": KB.list_docs(), "stats": KB.stats()}


@app.delete("/api/knowledge")
async def clear_knowledge() -> dict:
    before = KB.stats()
    KB.clear()
    log.info("清空知识库（原有 %s 篇 / %s 块）", before["docs"], before["chunks"])
    return {"ok": True, "stats": KB.stats()}


@app.delete("/api/knowledge/{doc_id}")
async def delete_doc(doc_id: str) -> dict:
    try:
        removed = KB.remove_doc(doc_id)
    except StoreCorrupt as e:
        return JSONResponse({"ok": False, "message": str(e)}, status_code=409)
    log.info("删除文档 %s（%s 块）", doc_id, removed)
    return {"ok": True, "removed": removed, "stats": KB.stats()}


def _safe_name(name: str) -> str:
    """把用户给的文件名压成安全的单段文件名。

    Path(name).name 先砍掉所有目录成分（`../../etc/passwd` → `passwd`），
    再白名单化剩余字符。全被过滤光时兜个底，避免落盘写出 `1712345-` 这种名字。
    """
    return re.sub(r"[^0-9A-Za-z一-鿿._-]+", "_", Path(name).name) or "upload.bin"


async def _save_upload(file: UploadFile, path: Path) -> str:
    """边收边写盘，超过上限立刻停。返回错误信息，空串表示成功。

    旧写法是 `await file.read()` 整个读进内存再判断大小——100MB 的限制
    在一个 1GB 的文件面前等于没有，读的时候内存就先炸了。
    """
    size = 0
    with path.open("wb") as fh:
        while True:
            chunk = await file.read(_READ_CHUNK)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_UPLOAD:
                return f"文件超过 {MAX_UPLOAD // 1024 // 1024}MB 限制"
            fh.write(chunk)
    if size == 0:
        return "文件是空的"
    return ""


@app.post("/api/knowledge/upload")
async def upload_knowledge(file: UploadFile = File(...)) -> dict:
    name = _safe_name(file.filename or "upload.bin")
    tmp = BASE / "data" / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    path = tmp / f"{time.time_ns()}-{name}"
    try:
        err = await _save_upload(file, path)
        if err:
            log.warning("上传被拒：%s（%s）", name, err)
            return JSONResponse({"ok": False, "message": err}, status_code=400)
        log.info("开始建库：%s", name)
        transcript, doc_id, err = await pipeline.ingest_document(path, KB, config.load_config())
        if err:
            log.error("建库失败：%s -> %s", name, err)
        else:
            log.info("建库完成：%s -> %s 块", name,
                     next((s["detail"].get("new_chunks") for s in transcript
                           if s["stage"] == "index"), "?"))
        return {
            "ok": not err,
            "doc_name": name,
            "doc_id": doc_id,
            "error": err or None,
            "transcript": transcript,
        }
    except StoreCorrupt as e:
        return JSONResponse({"ok": False, "message": str(e)}, status_code=409)
    finally:
        path.unlink(missing_ok=True)


@app.post("/api/knowledge/load-examples")
async def load_examples() -> dict:
    ex_dir = BASE / "examples"
    files = sorted(p for p in ex_dir.iterdir() if p.is_file())
    # 幂等：移除同名旧文档，避免重复点击堆出重复内容
    try:
        existing = {d["doc_name"] for d in KB.list_docs()}
        for d in KB.list_docs():
            if d["doc_name"] in {p.name for p in files}:
                KB.remove_doc(d["doc_id"])
    except StoreCorrupt as e:
        return JSONResponse({"ok": False, "message": str(e)}, status_code=409)
    log.info("载入内置示例 %d 篇", len(files))
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
    if fail:
        log.error("内置示例载入失败 %d/%d 篇", fail, len(files))
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


HISTORY_MAX_TURNS = 4  # 只带最近几轮进上下文
HISTORY_MAX_CHARS = 4000  # 单轮上限
HISTORY_SCAN = HISTORY_MAX_TURNS * 4  # 从请求体尾部最多回看多少条


def _clean_history(raw) -> list[dict]:
    """校验前端传来的多轮上文。

    前端是唯一来源，但请求体是不可信输入：必须限条数、限长度，
    否则一段伪造的超长 history 就能把上下文顶爆、把调用费用打上去。
    """
    if not isinstance(raw, list):
        return []
    out: list[dict] = []
    # 先切片再逐条校验：列表长度本身不可信，不能让它逼着我们对百万条做字符串处理。
    # 回看窗口比实际要留的轮数大，这样夹在中间的空壳不会把有效轮次挤出窗口。
    for h in raw[-HISTORY_SCAN:]:
        if not isinstance(h, dict):
            continue
        q = str(h.get("question") or "").strip()
        a = str(h.get("answer") or "").strip()
        if not q or not a:
            continue
        out.append({"question": q[:HISTORY_MAX_CHARS], "answer": a[:HISTORY_MAX_CHARS]})
    return out[-HISTORY_MAX_TURNS:]


DOC_ID_MAX = 200  # 一次最多接受多少个文档 id，别让超大数组拖着做字符串处理


def _clean_doc_ids(raw) -> set[str] | None:
    """校验前端传来的「参与匹配的文档范围」。

    返回 None 表示全库（没传、传空、传了非法值都按全库处理——这是默认语义，
    也是唯一安全的回退：范围过滤只应该让结果变少，不该因为参数畸形把整库变成 0 命中）。
    """
    if not isinstance(raw, list):
        return None
    out = {
        str(x).strip()[:128]
        for x in raw[:DOC_ID_MAX]
        if isinstance(x, (str, int)) and str(x).strip()
    }
    return out or None


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
            except asyncio.CancelledError:
                log.info("流程被客户端中断")
                raise
            except Exception as e:  # noqa: BLE001
                # 异常要落日志：前端只看到一句提示，事后得能查到栈
                log.exception("流程异常")
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
    history = _clean_history(body.get("history"))
    log.info("问答开始：%s%s", question[:80],
             f"（带 {len(history)} 轮上文）" if history else "")
    return _stream(
        lambda emit: pipeline.run_query(
            question,
            KB,
            cfg,
            emit,
            top_k=_num(body, "top_k", int),
            min_score=_num(body, "min_score", float),
            history=history,
        )
    )


@app.post("/api/match")
async def match(body: dict):
    text = (body.get("text") or "").strip()
    if not text:
        return JSONResponse({"ok": False, "message": "待匹配文本不能为空"}, status_code=400)
    cfg = config.load_config()
    scope = _clean_doc_ids(body.get("doc_ids"))
    log.info("内容匹配开始：%s（范围 %s）", text[:60],
             "全库" if scope is None else f"{len(scope)} 篇")
    return _stream(
        lambda emit: pipeline.run_match(
            text,
            KB,
            cfg,
            emit,
            with_llm=_to_bool(body.get("with_llm")),
            top_k=_num(body, "top_k", int),
            min_score=_num(body, "min_score", float),
            doc_ids=scope,
        )
    )
