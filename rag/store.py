"""本地向量库：numpy 归一化向量 + meta.json，按行对齐，支持持久化。

规模适合个人知识库（数万块以内），余弦检索即点积。

落盘走「先写临时文件、再 os.replace 原子替换」。直接覆盖写的话，进程在序列化
几 MB JSON 的中途被杀会留下半截文件，下次载入解析失败，整个库被当成空的，
再存一次盘用户的数据就真没了——实测复现过，不是理论风险。
"""
from __future__ import annotations

import io
import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path

import numpy as np

from .errors import DimMismatch, StoreCorrupt

log = logging.getLogger("rag.store")


class VectorStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._meta_path = self.root / "meta.json"
        self._vec_path = self.root / "vectors.npy"
        self._lock = threading.RLock()
        self._meta: list[dict] = []
        self._vectors = np.empty((0, 0), dtype=np.float32)
        self.problems: list[str] = []  # 载入时发现的数据问题，经 stats() 上报给界面
        self._blocked: str | None = None  # 非空则禁止写入，防止覆盖掉还能抢救的原文件
        self._load()

    # ---------- 基础读写 ----------
    def _read_meta(self, notes: list[str]) -> tuple[list[dict] | None, bool]:
        """读 meta.json。返回 (内容, 是否出错)。文件不存在不算错（全新库）。"""
        if not self._meta_path.exists():
            return None, False
        try:
            data = json.loads(self._meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
            notes.append(f"meta.json 解析失败：{e}。原文件已原样保留，未做任何改动")
            return None, True
        if not isinstance(data, list):
            notes.append("meta.json 顶层不是数组。原文件已原样保留，未做任何改动")
            return None, True
        return data, False

    def _read_vecs(self, notes: list[str]) -> tuple[np.ndarray | None, bool]:
        if not self._vec_path.exists():
            return None, False
        try:
            return np.load(self._vec_path), False
        except (OSError, ValueError) as e:
            notes.append(f"vectors.npy 读取失败：{e}。原文件已原样保留，未做任何改动")
            return None, True

    def _load(self) -> None:
        """载入并按行对齐。任何异常都只记录问题、不清空数据。

        旧实现遇到解析失败或行数不一致就把内存里的库置空，紧接着一次 _save()
        就把空库覆盖回磁盘——用户的整个知识库静默消失，且没有任何提示。
        """
        notes: list[str] = []
        meta, meta_err = self._read_meta(notes)
        vecs, vec_err = self._read_vecs(notes)

        if meta_err or vec_err:
            self.problems = notes
            self._blocked = "；".join(notes)
            log.warning("向量库载入失败，按空库启动并锁定写入（磁盘文件未改动）：%s", self._blocked)
            return

        if (meta is None) != (vecs is None):
            notes.append(
                "meta.json 与 vectors.npy 只剩其中一个，无法还原向量库。"
                "已按空库启动，原文件未做改动"
            )
            self.problems = notes
            self._blocked = notes[-1]
            log.warning(notes[-1])
            return

        if meta is None:  # 两个文件都不存在：全新空库
            self.problems = []
            return

        if vecs is not None and int(vecs.shape[0]) != len(meta):
            # 两侧行数对不上，只可能是「meta 已换新、vectors 还没换」这类写盘中断。
            # 行是追加有序的，共同前缀必然自洽：截到共同长度，损失只限于最后一批，
            # 远好过把整个库丢掉。
            keep = min(len(meta), int(vecs.shape[0]))
            notes.append(
                f"meta.json（{len(meta)} 条）与 vectors.npy（{vecs.shape[0]} 行）行数不一致，"
                f"已按共同前缀保留前 {keep} 条；最后一次写入可能未完成"
            )
            log.warning(notes[-1])
            meta = meta[:keep]
            vecs = vecs[:keep]

        self._meta = meta
        self._vectors = vecs.astype(np.float32) if vecs is not None else np.empty((0, 0), dtype=np.float32)
        self.problems = notes

    def _save(self) -> None:
        """整份原子落盘。读到的要么是上一份完整内容，要么是新的一份，不会半截。"""
        meta_bytes = json.dumps(self._meta, ensure_ascii=False).encode("utf-8")
        buf = io.BytesIO()
        np.save(buf, self._vectors)
        self._replace(self._meta_path, meta_bytes)
        self._replace(self._vec_path, buf.getvalue())
        # 磁盘和内存现在完全一致，之前载入时报的问题（如行数不一致）已经修好了。
        # 不清掉的话界面上的告警横幅会一直挂着，用户以为没救回来。
        self.problems = []

    @staticmethod
    def _replace(path: Path, data: bytes) -> None:
        """写临时文件再 os.replace。同一卷上 os.replace 是原子的。"""
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)

    # ---------- 增删改查 ----------
    def writable(self) -> str | None:
        """返回禁止写入的原因；None 表示可写。"""
        return self._blocked

    def _guard(self) -> None:
        if self._blocked:
            raise StoreCorrupt(
                f"{self._blocked}。为避免覆盖掉还能抢救的原文件，已暂停写入；"
                "请先备份 data/kb 目录，再点「清空知识库」确认放弃原有数据。"
            )

    def add_doc(
        self,
        doc_id: str,
        doc_name: str,
        items: list[dict],
        vectors: np.ndarray,
    ) -> int:
        """追加一个文档。items 与 vectors 行对齐；同 doc_id 旧块先删除。返回本次入库块数。"""
        with self._lock:
            self._guard()
            self._remove_rows_of(doc_id)
            vecs = np.asarray(vectors, dtype=np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / np.maximum(norms, 1e-12)  # 防零向量除出 NaN
            created = time.strftime("%Y-%m-%d %H:%M:%S")
            new_meta = []
            for it, v in zip(items, vecs):
                new_meta.append(
                    {
                        "id": f"{doc_id}#{it['chunk_index']}",
                        "doc_id": doc_id,
                        "doc_name": doc_name,
                        "chunk_index": it["chunk_index"],
                        "chars": len(it["text"]),
                        "text": it["text"],
                        "created_at": created,
                    }
                )
            if self._vectors.size == 0:
                self._vectors = vecs
            else:
                self._vectors = np.concatenate([self._vectors, vecs], axis=0)
            self._meta.extend(new_meta)
            self._save()
            return len(new_meta)

    def _scores(self, query_vec: np.ndarray) -> np.ndarray:
        """查询向量与全库的余弦相似度。库中向量入库时已归一化，这里只需归一化查询向量。"""
        q = np.asarray(query_vec, dtype=np.float32).ravel()
        dim = int(self._vectors.shape[1])
        if q.shape[0] != dim:
            raise DimMismatch(
                f"查询向量是 {q.shape[0]} 维，库中向量是 {dim} 维。"
                "通常是改了嵌入模型或向量维度却没重建库——请在设置里核对维度后"
                "「清空知识库」再重新建库。"
            )
        norm = float(np.linalg.norm(q))
        if norm > 0:
            q = q / norm
        return (self._vectors @ q).ravel()

    @staticmethod
    def _hit(m: dict, score: float, rank: int) -> dict:
        return {
            "rank": rank,
            "score": round(score, 4),
            "id": m["id"],
            "doc_id": m["doc_id"],
            "doc_name": m["doc_name"],
            "chunk_index": m["chunk_index"],
            "chars": m["chars"],
            "text": m["text"],
        }

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 5,
        min_score: float = 0.0,
        extra: int = 0,
        doc_ids: set[str] | None = None,
    ) -> tuple[list[dict], list[dict], int]:
        """向量粗排。

        返回 (Top-K 命中, 紧随其后被截断的命中, 高于阈值的总命中数)。
        above 顺带在这里算出并返回，调用方不必再算一遍全量相似度；
        truncated 只带文本前 160 字，用于告诉用户「Top-K 之外还有什么」。

        doc_ids 为 None 表示全库；给了集合就只在范围内的文档上排名。
        必须**先在子集内排名再取 Top-K**——先取全库 Top-K 再过滤的话，
        范围一窄就会把名额浪费在范围外的块上，最后空手而归。
        """
        with self._lock:
            if not self._meta:
                return [], [], 0
            if doc_ids is None:
                cand = np.arange(len(self._meta))
            else:
                cand = np.fromiter(
                    (i for i, m in enumerate(self._meta) if m["doc_id"] in doc_ids),
                    dtype=np.intp,
                )
            if cand.size == 0:
                return [], [], 0
            scores = self._scores(query_vec)[cand]
            above = int(np.count_nonzero(scores >= min_score))
            order = np.argsort(-scores)  # 降序；后续分数只会更低
            hits: list[dict] = []
            truncated: list[dict] = []
            for rank_pos in order:
                s = float(scores[rank_pos])
                if s < min_score:
                    break
                m = self._meta[int(cand[rank_pos])]
                if len(hits) < top_k:
                    hits.append(self._hit(m, s, len(hits) + 1))
                elif len(truncated) < extra:
                    t = self._hit(m, s, len(hits) + len(truncated) + 1)
                    t["text"] = t["text"][:160]
                    t["cut"] = t["chars"] > 160
                    truncated.append(t)
                else:
                    break
            return hits, truncated, above

    def neighbors(self, doc_id: str, chunk_index: int, window: int = 1) -> list[dict]:
        """取同文档内 chunk_index ± window 的邻近块（含自己），按原文顺序返回。

        用于 small-to-big：命中的块是「检索单元」，边界可能切断了语义，
        把左右邻块一并送进上下文能补回被切断的上下文。
        """
        if window <= 0:
            return []
        lo, hi = chunk_index - window, chunk_index + window
        with self._lock:
            rows = [
                m
                for m in self._meta
                if m["doc_id"] == doc_id and lo <= m["chunk_index"] <= hi
            ]
        rows.sort(key=lambda m: m["chunk_index"])
        return rows

    def remove_doc(self, doc_id: str) -> int:
        with self._lock:
            self._guard()
            return self._remove_rows_of(doc_id)

    def _remove_rows_of(self, doc_id: str) -> int:
        keep_idx = [i for i, m in enumerate(self._meta) if m["doc_id"] != doc_id]
        removed = len(self._meta) - len(keep_idx)
        if removed == 0:
            return 0
        self._meta = [self._meta[i] for i in keep_idx]
        if keep_idx:
            self._vectors = self._vectors[keep_idx]
        else:
            self._vectors = np.empty((0, 0), dtype=np.float32)
        self._save()
        return removed

    def clear(self) -> None:
        """清空知识库。这是用户明确「放弃原有数据」的动作，因此同时解除写入锁定。"""
        with self._lock:
            self._meta = []
            self._vectors = np.empty((0, 0), dtype=np.float32)
            self._blocked = None
            self.problems = []
            self._save()
            log.info("知识库已清空")

    def list_docs(self) -> list[dict]:
        """按文档聚合统计，便于侧栏展示。"""
        with self._lock:
            agg: dict[str, dict] = {}
            for m in self._meta:
                a = agg.setdefault(
                    m["doc_id"],
                    {
                        "doc_id": m["doc_id"],
                        "doc_name": m["doc_name"],
                        "chunks": 0,
                        "chars": 0,
                        "created_at": m["created_at"],
                    },
                )
                a["chunks"] += 1
                a["chars"] += m["chars"]
            return sorted(agg.values(), key=lambda d: d["created_at"], reverse=True)

    def stats(self) -> dict:
        with self._lock:
            dim = int(self._vectors.shape[1]) if self._vectors.size else 0
            return {
                "docs": len(self.list_docs()),
                "chunks": len(self._meta),
                "dim": dim,
                "problems": list(self.problems),  # 载入时发现的数据问题，空列表＝一切正常
                "writable": self._blocked is None,
            }


def new_doc_id() -> str:
    return uuid.uuid4().hex[:12]
