"""向量库落盘与损坏恢复的行为约束。

这里的用例基本都是「真实发生过 / 差点发生」的问题，不是凑覆盖率：
旧实现在 meta.json 解析失败时会把内存库置空，紧接着一次写盘就把用户的
整个知识库覆盖没了，而且界面上没有任何提示。
"""
import json

import numpy as np
import pytest

from rag.errors import DimMismatch, StoreCorrupt
from rag.store import VectorStore, new_doc_id

DIM = 4


def _items(texts):
    return [{"chunk_index": i, "text": t} for i, t in enumerate(texts)]


def _vecs(n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((n, DIM), dtype=np.float32)


def _fill(store, name="doc.md", texts=("甲", "乙", "丙"), seed=0):
    doc_id = new_doc_id()
    store.add_doc(doc_id, name, _items(list(texts)), _vecs(len(texts), seed))
    return doc_id


# ---------------------------------------------------------------- 正常往返
def test_round_trip(tmp_path):
    s = VectorStore(tmp_path)
    doc_id = _fill(s)
    assert s.stats() == {"docs": 1, "chunks": 3, "dim": DIM, "problems": [], "writable": True}

    again = VectorStore(tmp_path)
    assert again.stats()["chunks"] == 3
    assert again.list_docs()[0]["doc_id"] == doc_id
    assert again.list_docs()[0]["chunks"] == 3


def test_new_dir_starts_empty(tmp_path):
    s = VectorStore(tmp_path / "还不存在")
    assert s.stats()["chunks"] == 0
    assert s.stats()["problems"] == []
    assert s.list_docs() == []


def test_vectors_are_normalized_on_ingest(tmp_path):
    s = VectorStore(tmp_path)
    s.add_doc(new_doc_id(), "d", _items(["a", "b"]), np.array([[3, 4, 0, 0], [0, 0, 0, 2]], np.float32))
    norms = np.linalg.norm(s._vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_zero_vector_does_not_produce_nan(tmp_path):
    s = VectorStore(tmp_path)
    s.add_doc(new_doc_id(), "d", _items(["零向量"]), np.zeros((1, DIM), np.float32))
    assert np.isfinite(s._vectors).all()


def test_same_doc_id_replaces_old_chunks(tmp_path):
    s = VectorStore(tmp_path)
    doc_id = new_doc_id()
    s.add_doc(doc_id, "d", _items(["旧1", "旧2", "旧3"]), _vecs(3))
    s.add_doc(doc_id, "d", _items(["新1"]), _vecs(1, seed=9))
    assert s.stats()["chunks"] == 1
    assert s.list_docs()[0]["chunks"] == 1


# ---------------------------------------------------------------- 损坏恢复
def test_broken_meta_does_not_wipe_the_store(tmp_path):
    """meta.json 半截必须：不报告为空库、禁止写入、原文件原样保留。"""
    s = VectorStore(tmp_path)
    _fill(s)

    raw = (tmp_path / "meta.json").read_bytes()
    (tmp_path / "meta.json").write_bytes(raw[: len(raw) // 2])

    broken = VectorStore(tmp_path)
    st = broken.stats()
    assert st["writable"] is False
    assert st["problems"], "载入失败必须给出可展示的原因"
    assert "解析失败" in st["problems"][0]

    # 关键：此时写盘会抛，而不是把空库盖回磁盘
    with pytest.raises(StoreCorrupt):
        broken.add_doc(new_doc_id(), "x", _items(["x"]), _vecs(1))
    with pytest.raises(StoreCorrupt):
        broken.remove_doc("任意")

    # 原文件没被碰过，用户还能自己抢救
    assert (tmp_path / "meta.json").read_bytes() == raw[: len(raw) // 2]


def test_broken_meta_survives_reload_after_manual_fix(tmp_path):
    s = VectorStore(tmp_path)
    _fill(s)
    good = (tmp_path / "meta.json").read_text(encoding="utf-8")

    (tmp_path / "meta.json").write_text("{ 坏", encoding="utf-8")
    assert VectorStore(tmp_path).stats()["writable"] is False

    (tmp_path / "meta.json").write_text(good, encoding="utf-8")
    fixed = VectorStore(tmp_path)
    assert fixed.stats() == {"docs": 1, "chunks": 3, "dim": DIM, "problems": [], "writable": True}


def test_meta_not_a_list_is_treated_as_corrupt(tmp_path):
    s = VectorStore(tmp_path)
    _fill(s)
    (tmp_path / "meta.json").write_text('{"oops": 1}', encoding="utf-8")
    st = VectorStore(tmp_path).stats()
    assert st["writable"] is False
    assert "顶层不是数组" in st["problems"][0]


def test_broken_vectors_locks_writes(tmp_path):
    s = VectorStore(tmp_path)
    _fill(s)
    (tmp_path / "vectors.npy").write_bytes(b"this is not a npy file")

    broken = VectorStore(tmp_path)
    assert broken.stats()["writable"] is False
    assert "vectors.npy" in broken.stats()["problems"][0]


def test_row_mismatch_truncates_to_common_prefix(tmp_path):
    """写盘中断（meta 已换、vectors 未换）应截到共同前缀，而不是丢整库。"""
    s = VectorStore(tmp_path)
    _fill(s, texts=("1", "2", "3", "4", "5"))

    meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    (tmp_path / "meta.json").write_text(
        json.dumps(meta + [dict(meta[-1], id="额外#9", chunk_index=9)], ensure_ascii=False),
        encoding="utf-8",
    )

    recovered = VectorStore(tmp_path)
    st = recovered.stats()
    assert st["chunks"] == 5, "应保留两侧都有的 5 条"
    assert st["writable"] is True, "截断后仍可写，否则用户要连能救的数据一起清掉"
    assert "行数不一致" in st["problems"][0]

    # 截断之后继续建库要正常
    _fill(recovered, name="新文档.md", texts=("新",), seed=7)
    assert recovered.stats()["chunks"] == 6
    assert recovered.stats()["problems"] == []


def test_only_one_file_present_locks_writes(tmp_path):
    s = VectorStore(tmp_path)
    _fill(s)
    (tmp_path / "vectors.npy").unlink()

    half = VectorStore(tmp_path)
    st = half.stats()
    assert st["writable"] is False
    assert "只剩其中一个" in st["problems"][0]


def test_clear_releases_the_write_lock(tmp_path):
    """「清空知识库」是用户明确放弃原数据的动作，必须能解锁。"""
    s = VectorStore(tmp_path)
    _fill(s)
    (tmp_path / "meta.json").write_text("{ 坏", encoding="utf-8")

    blocked = VectorStore(tmp_path)
    with pytest.raises(StoreCorrupt):
        blocked.add_doc(new_doc_id(), "x", _items(["x"]), _vecs(1))

    blocked.clear()
    assert blocked.stats() == {"docs": 0, "chunks": 0, "dim": 0, "problems": [], "writable": True}
    _fill(blocked, name="重建.md")
    assert VectorStore(tmp_path).stats()["chunks"] == 3


def test_no_tmp_files_left_behind(tmp_path):
    s = VectorStore(tmp_path)
    _fill(s)
    s.remove_doc(s.list_docs()[0]["doc_id"])
    leftovers = [p.name for p in tmp_path.iterdir() if p.suffix == ".tmp"]
    assert leftovers == []


# ---------------------------------------------------------------- 检索
def test_search_orders_by_cosine_and_reports_truncated(tmp_path):
    s = VectorStore(tmp_path)
    s.add_doc(
        new_doc_id(),
        "d",
        _items(["x 方向", "y 方向", "z 方向"]),
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], np.float32),
    )
    hits, truncated, above = s.search(np.array([0.9, 0.1, 0, 0], np.float32), top_k=1, extra=1)
    assert [h["text"] for h in hits] == ["x 方向"]
    assert hits[0]["rank"] == 1
    assert truncated[0]["text"] == "y 方向"
    assert truncated[0]["rank"] == 2
    assert above == 3


def test_search_min_score_filters(tmp_path):
    s = VectorStore(tmp_path)
    s.add_doc(
        new_doc_id(), "d", _items(["x 方向", "y 方向"]),
        np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32),
    )
    hits, _, above = s.search(np.array([1, 0, 0, 0], np.float32), top_k=5, min_score=0.5)
    assert [h["text"] for h in hits] == ["x 方向"]
    assert above == 1


def test_search_on_empty_store(tmp_path):
    assert VectorStore(tmp_path).search(np.zeros(DIM, np.float32)) == ([], [], 0)


# ---------------------------------------------------------------- 匹配范围过滤
def test_search_doc_filter_restricts_to_scope(tmp_path):
    s = VectorStore(tmp_path)
    s.add_doc(new_doc_id(), "甲.md", _items(["甲的 x"]), np.array([[1, 0, 0, 0]], np.float32))
    s.add_doc(new_doc_id(), "乙.md", _items(["乙的 x"]), np.array([[0.9, 0.1, 0, 0]], np.float32))
    docs = {d["doc_name"]: d["doc_id"] for d in s.list_docs()}

    q = np.array([1, 0, 0, 0], np.float32)
    all_hits, _, above_all = s.search(q, top_k=5)
    assert [h["doc_name"] for h in all_hits] == ["甲.md", "乙.md"]
    assert above_all == 2

    hits, _, above = s.search(q, top_k=5, doc_ids={docs["乙.md"]})
    assert [h["doc_name"] for h in hits] == ["乙.md"]
    assert above == 1, "above 也只该数范围内的"


def test_search_doc_filter_ranks_within_subset_not_globally(tmp_path):
    """先在全库取 Top-K 再过滤是错的：范围外的块会把名额占光。"""
    s = VectorStore(tmp_path)
    # 范围外的 3 条全都比范围内的更像
    s.add_doc(new_doc_id(), "范围外.md", _items(["外1", "外2", "外3"]),
              np.array([[1, 0, 0, 0], [0.99, 0, 0, 0], [0.98, 0, 0, 0]], np.float32))
    s.add_doc(new_doc_id(), "范围内.md", _items(["内1", "内2"]),
              np.array([[0.5, 0.5, 0, 0], [0.4, 0.6, 0, 0]], np.float32))
    inner = {d["doc_name"]: d["doc_id"] for d in s.list_docs()}["范围内.md"]

    hits, _, _ = s.search(np.array([1, 0, 0, 0], np.float32), top_k=2, doc_ids={inner})
    assert [h["text"] for h in hits] == ["内1", "内2"], "范围内排名第一的必须拿到名额"


def test_search_doc_filter_unknown_ids_yield_nothing(tmp_path):
    s = VectorStore(tmp_path)
    _fill(s)
    assert s.search(np.ones(DIM, np.float32), doc_ids={"不存在的 id"}) == ([], [], 0)


def test_search_doc_filter_empty_set_yields_nothing(tmp_path):
    """空集合是「一篇都不选」，与 None（全库）必须区分开。"""
    s = VectorStore(tmp_path)
    _fill(s)
    assert s.search(np.ones(DIM, np.float32), doc_ids=set()) == ([], [], 0)
    assert s.search(np.ones(DIM, np.float32), doc_ids=None)[2] == 3


def test_search_doc_filter_still_reports_truncated(tmp_path):
    s = VectorStore(tmp_path)
    s.add_doc(new_doc_id(), "甲.md", _items(["甲的 a", "甲的 b", "甲的 c"]),
              np.array([[1, 0, 0, 0], [0.9, 0.1, 0, 0], [0.8, 0.2, 0, 0]], np.float32))
    s.add_doc(new_doc_id(), "乙.md", _items(["乙的 a"]), np.array([[0.95, 0.05, 0, 0]], np.float32))
    jia = {d["doc_name"]: d["doc_id"] for d in s.list_docs()}["甲.md"]

    hits, truncated, above = s.search(
        np.array([1, 0, 0, 0], np.float32), top_k=1, extra=1, doc_ids={jia}
    )
    assert [h["text"] for h in hits] == ["甲的 a"]
    assert [t["text"] for t in truncated] == ["甲的 b"], "截断项也该只在范围内"
    assert above == 3


def test_search_dim_mismatch_raises_chinese_error(tmp_path):
    s = VectorStore(tmp_path)
    _fill(s)
    with pytest.raises(DimMismatch) as e:
        s.search(np.zeros(8, np.float32))
    assert "8 维" in str(e.value) and "4 维" in str(e.value)


def test_truncated_text_is_clipped_but_flags_cut(tmp_path):
    s = VectorStore(tmp_path)
    long_text = "长" * 300
    s.add_doc(
        new_doc_id(), "d", _items(["短", long_text]),
        np.array([[1, 0, 0, 0], [0.99, 0.01, 0, 0]], np.float32),
    )
    hits, truncated, _ = s.search(np.array([1, 0, 0, 0], np.float32), top_k=1, extra=1)
    assert "cut" not in hits[0]
    assert len(truncated[0]["text"]) == 160
    assert truncated[0]["cut"] is True


# ---------------------------------------------------------------- 邻块扩展
def test_neighbors_stay_inside_one_document(tmp_path):
    s = VectorStore(tmp_path)
    a = new_doc_id()
    b = new_doc_id()
    s.add_doc(a, "a.md", _items(["a0", "a1", "a2", "a3"]), _vecs(4))
    s.add_doc(b, "b.md", _items(["b0", "b1", "b2"]), _vecs(3, seed=3))

    got = s.neighbors(a, 0, window=1)
    assert [m["text"] for m in got] == ["a0", "a1"], "边界处只取到本块 + 下一块"
    got = s.neighbors(b, 1, window=1)
    assert [m["text"] for m in got] == ["b0", "b1", "b2"]
    assert all(m["doc_id"] == b for m in got), "窗口不得滑到别的文档"


def test_neighbors_disabled_by_zero_window(tmp_path):
    s = VectorStore(tmp_path)
    a = _fill(s)
    assert s.neighbors(a, 1, window=0) == []


# ---------------------------------------------------------------- 删除
def test_remove_doc_keeps_rows_aligned(tmp_path):
    s = VectorStore(tmp_path)
    keep = new_doc_id()
    drop = new_doc_id()
    s.add_doc(drop, "drop.md", _items(["d0", "d1"]), _vecs(2))
    s.add_doc(keep, "keep.md", _items(["k0", "k1", "k2"]), _vecs(3, seed=5))

    assert s.remove_doc(drop) == 2
    assert s.stats()["chunks"] == 3
    assert s.search(np.array([1, 0, 0, 0], np.float32), top_k=9)[2] == 3
    # 重载后行仍然对齐
    assert VectorStore(tmp_path).stats()["chunks"] == 3


def test_remove_missing_doc_is_a_noop(tmp_path):
    s = VectorStore(tmp_path)
    _fill(s)
    assert s.remove_doc("不存在") == 0
    assert s.stats()["chunks"] == 3


def test_remove_last_doc_leaves_a_working_empty_store(tmp_path):
    s = VectorStore(tmp_path)
    doc_id = _fill(s)
    s.remove_doc(doc_id)
    assert s.stats()["chunks"] == 0
    _fill(s, name="再来.md")
    assert s.stats()["chunks"] == 3
    assert VectorStore(tmp_path).stats()["chunks"] == 3


def test_list_docs_aggregates_and_sorts_by_time(tmp_path):
    s = VectorStore(tmp_path)
    s.add_doc("d1", "一.md", _items(["ab", "c"]), _vecs(2))
    s.add_doc("d2", "二.md", _items(["def"]), _vecs(1, seed=2))
    docs = {d["doc_name"]: d for d in s.list_docs()}
    assert docs["一.md"]["chunks"] == 2 and docs["一.md"]["chars"] == 3
    assert docs["二.md"]["chunks"] == 1 and docs["二.md"]["chars"] == 3


def test_meta_json_is_written_as_readable_utf8(tmp_path):
    s = VectorStore(tmp_path)
    s.add_doc("d1", "中文文档.md", _items(["内容"]), _vecs(1))
    raw = (tmp_path / "meta.json").read_text(encoding="utf-8")
    assert "中文文档.md" in raw, "不应写成 \\uXXXX 转义，日志和排障都要能直接看"
