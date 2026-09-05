"""本地向量库：numpy 归一化向量 + meta.json，按行对齐，支持持久化。

规模适合个人知识库（数万块以内），余弦检索即点积。
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path

import numpy as np


class VectorStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._meta_path = self.root / "meta.json"
        self._vec_path = self.root / "vectors.npy"
        self._lock = threading.RLock()
        self._meta: list[dict] = []
        self._vectors = np.empty((0, 0), dtype=np.float32)
        self._load()

    # ---------- 基础读写 ----------
    def _load(self) -> None:
        meta: list[dict] = []
        vecs = None
        if self._meta_path.exists():
            try:
                meta = json.loads(self._meta_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                meta = []
        if self._vec_path.exists():
            try:
                vecs = np.load(self._vec_path)
            except OSError:
                vecs = None
        if vecs is not None and len(meta) > 0 and vecs.shape[0] == len(meta):
            self._meta = meta
            self._vectors = vecs.astype(np.float32)
        else:
            self._meta = []
            self._vectors = np.empty((0, 0), dtype=np.float32)

    def _save(self) -> None:
        self._meta_path.write_text(
            json.dumps(self._meta, ensure_ascii=False), encoding="utf-8"
        )
        np.save(self._vec_path, self._vectors)

    # ---------- 增删改查 ----------
    def add_doc(
        self,
        doc_id: str,
        doc_name: str,
        items: list[dict],
        vectors: np.ndarray,
    ) -> int:
        """追加一个文档。items 与 vectors 行对齐；同 doc_id 旧块先删除。返回本次入库块数。"""
        with self._lock:
            self._remove_rows_of(doc_id)
            vecs = np.asarray(vectors, dtype=np.float32)
            vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
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

    def search(self, query_vec: np.ndarray, top_k: int = 5, min_score: float = 0.0) -> list[dict]:
        """返回 Top-K，score 为余弦相似度。query_vec 未归一化会自动归一。"""
        with self._lock:
            n = len(self._meta)
            if n == 0:
                return []
            q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
            q = q / np.linalg.norm(q)
            scores = (self._vectors @ q.T).ravel()
            order = np.argsort(-scores)
            hits = []
            for pos in order:
                s = float(scores[pos])
                if s < min_score or len(hits) >= top_k:
                    break  # 已排序，后续分数只会更低
                m = self._meta[pos]
                hits.append(
                    {
                        "rank": len(hits) + 1,
                        "score": round(s, 4),
                        "id": m["id"],
                        "doc_id": m["doc_id"],
                        "doc_name": m["doc_name"],
                        "chunk_index": m["chunk_index"],
                        "chars": m["chars"],
                        "text": m["text"],
                    }
                )
            return hits

    def count_above(self, query_vec: np.ndarray, min_score: float = 0.0) -> int:
        """统计相似度 >= min_score 的分块总数，用于提示"被阈值过滤了多少"。"""
        with self._lock:
            n = len(self._meta)
            if n == 0:
                return 0
            q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
            q = q / np.linalg.norm(q)
            scores = (self._vectors @ q.T).ravel()
            return int(np.count_nonzero(scores >= min_score))

    def remove_doc(self, doc_id: str) -> int:
        with self._lock:
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
        with self._lock:
            self._meta = []
            self._vectors = np.empty((0, 0), dtype=np.float32)
            self._save()

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
            return {"docs": len(self.list_docs()), "chunks": len(self._meta), "dim": dim}


def new_doc_id() -> str:
    return uuid.uuid4().hex[:12]
