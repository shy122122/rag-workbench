"""两种切分策略的行为约束。

用户点「清空知识库重建」最常见的触发原因就是改了切分参数，
所以这里的重点是：块大小上限、块间重叠、标题边界、以及退化路径。
"""
from rag.chunker import STRATEGY_LABELS, split_by_structure, split_text


# ---------------------------------------------------------------- 递归字符切分
def test_empty_and_blank_input():
    assert split_text("") == []
    assert split_text("   \n\n  ") == []
    assert split_text(None) == []


def test_short_text_is_one_chunk():
    assert split_text("就一句话。", chunk_size=500) == ["就一句话。"]


def test_chunk_size_zero_returns_whole_text():
    text = "甲" * 2000
    assert split_text(text, chunk_size=0) == [text]


def test_every_chunk_respects_the_size_cap():
    text = "。".join(f"第{i}个句子" for i in range(300))
    chunks = split_text(text, chunk_size=100, overlap=10)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)


def test_cut_prefers_sentence_boundary():
    """切点应落在句末标点后，而不是把句子拦腰截断。"""
    text = "。".join("句" * 10 for _ in range(40))
    chunks = split_text(text, chunk_size=64, overlap=0)
    assert all(c.endswith("。") for c in chunks[:-1])


def test_consecutive_chunks_overlap():
    text = "".join(f"{i:04d}" for i in range(200))  # 无分隔符，只能硬切
    chunks = split_text(text, chunk_size=100, overlap=20)
    assert len(chunks) >= 2
    assert chunks[0][-20:] == chunks[1][:20], "相邻块应有 overlap 个字符的重叠"


def test_always_makes_progress_even_with_huge_overlap():
    """overlap ≥ chunk_size 时旧实现会原地打转，这里必须仍然收敛。"""
    text = "。".join("内容" * 3 for _ in range(60))
    chunks = split_text(text, chunk_size=30, overlap=999)
    assert 1 < len(chunks) < 400
    assert all(len(c) <= 30 for c in chunks)


def test_no_empty_chunks_are_emitted():
    text = "\n\n\n" + "正文" * 300 + "\n\n\n"
    assert all(c.strip() for c in split_text(text, chunk_size=50, overlap=5))


def test_internal_whitespace_is_collapsed():
    chunks = split_text("甲    乙\t\t丙　丁，后面还有很长很长很长很长很长的一段话用来触发切分逻辑。" * 4, 60, 0)
    assert all("  " not in c for c in chunks)
    assert all("\t" not in c for c in chunks)


def test_reassembled_chunks_keep_all_content():
    """有重叠时内容只会重复，不该丢。"""
    text = "。".join(f"第{i}句内容" for i in range(120))
    chunks = split_text(text, chunk_size=80, overlap=15)
    body = "".join(chunks)
    for i in range(120):
        assert f"第{i}句内容" in body


# ---------------------------------------------------------------- 按结构切分
DOC = """# 总标题

开篇的一段引言文字。

## 第一章

第一章的正文内容。

### 1.1 小节

小节正文。

### 1.2 小节

又一段正文。

## 第二章

第二章的正文内容。
"""


def test_structure_splits_on_the_cut_level():
    chunks = split_by_structure(DOC, chunk_size=1000, cut_level=2)
    assert len(chunks) == 3, "## 级切分应得到 序章 / 第一章 / 第二章 三块"
    assert chunks[1].startswith("## 第一章")
    assert chunks[2].startswith("## 第二章")
    assert "1.1 小节" in chunks[1] and "1.2 小节" in chunks[1]


def test_structure_without_headings_falls_back_to_chars():
    text = "没有标题的一段话。" * 80
    assert split_by_structure(text, chunk_size=100, overlap=0, cut_level=2) == split_text(
        text, 100, 0
    )


def test_structure_chunk_size_zero_returns_whole_text():
    assert split_by_structure(DOC, chunk_size=0) == [DOC.strip()]


def test_structure_empty_input():
    assert split_by_structure("") == []
    assert split_by_structure(None) == []


def test_oversized_block_is_split_by_finer_headings():
    body = "\n".join(f"第{i}行的正文内容，用来把这一块撑大。" for i in range(60))
    doc = f"## 大章节\n\n### 甲\n{body}\n\n### 乙\n{body}\n"
    chunks = split_by_structure(doc, chunk_size=300, cut_level=2)
    assert len(chunks) >= 4, "超过上限时应在 ### 处继续拆"
    assert any(c.startswith("### 甲") for c in chunks)
    assert any(c.startswith("### 乙") for c in chunks)


def test_oversized_block_without_finer_headings_falls_back_to_chars():
    doc = "## 只有正文的一章\n" + "甲" * 2000
    chunks = split_by_structure(doc, chunk_size=200, cut_level=2)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_headings_are_not_left_orphaned():
    """连续标题（中间没正文）应聚在同一个块首，而不是各自成块。"""
    doc = "# 标题\n\n## 第一节\n\n### 子标题\n\n正文内容。\n"
    chunks = split_by_structure(doc, chunk_size=1000, cut_level=2)
    assert len(chunks) == 1
    assert chunks[0].startswith("# 标题")
    assert "正文内容。" in chunks[0]


def test_cut_level_controls_the_boundary():
    as_h1 = split_by_structure(DOC, chunk_size=1000, cut_level=1)
    assert len(as_h1) == 1, "只有 # 级才是边界时整篇一块"
    as_h2 = split_by_structure(DOC, chunk_size=1000, cut_level=2)
    assert len(as_h2) == 3


def test_cut_level_is_clamped():
    assert split_by_structure(DOC, chunk_size=1000, cut_level=99) == split_by_structure(
        DOC, chunk_size=1000, cut_level=6
    )
    assert split_by_structure(DOC, chunk_size=1000, cut_level=-3) == split_by_structure(
        DOC, chunk_size=1000, cut_level=1
    )


def test_hash_without_space_is_not_a_heading():
    """markdown 里 `## 标题` 要有空格才是标题；`##不是标题` 必须当正文。"""
    doc = "##不是标题\n\n# 真标题\n\n正文。\n"
    chunks = split_by_structure(doc, chunk_size=1000, cut_level=1)
    assert chunks == ["##不是标题", "# 真标题\n正文。"]


def test_structure_keeps_every_heading():
    chunks = split_by_structure(DOC, chunk_size=1000)
    joined = "\n".join(chunks)
    for h in ("# 总标题", "## 第一章", "### 1.1 小节", "### 1.2 小节", "## 第二章"):
        assert h in joined


def test_strategy_labels_match_config_keys():
    assert set(STRATEGY_LABELS) == {"recursive", "structure"}
    assert all(isinstance(v, str) and v for v in STRATEGY_LABELS.values())
