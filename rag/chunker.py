"""文本切分器：提供两种可对比的策略。

- recursive：递归字符切分（默认）。按 chunk_size 字符计量、块间按 overlap 滑动，切点
  优先落在段落空行/换行/句末标点处（_SEPARATORS 逐级找）。
- structure：按标题等级切分。块边界由 markdown 标题等级决定（cut_level，默认 ## 级一个
  章节成一块，块首带标题行）；chunk_size 在这里只是「单块字符上限」——某候选块超过上限时
  先用更细的标题再拆，没有任何更细标题才整块按字符兜底拆（此时 overlap 才生效）。
"""
from __future__ import annotations

import re

# 策略键与中文标签（config 里存键，界面显示中文）
CHUNK_STRATEGY_RECURSIVE = "recursive"
CHUNK_STRATEGY_STRUCTURE = "structure"
STRATEGY_LABELS = {
    CHUNK_STRATEGY_RECURSIVE: "递归字符切分",
    CHUNK_STRATEGY_STRUCTURE: "按段落结构",
}

_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "…", " ", "、", ","]
_HEAD_LVL_RE = re.compile(r"^(#{1,6})[ \t]+\S")  # 捕获 # 个数即标题等级 1~6


def split_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """把长文本切成若干块（按字符数计量）。chunk_size<=0 时整段一块。"""
    text = (text or "").strip()
    if not text:
        return []
    if chunk_size <= 0:
        return [text]

    text = _collapse_spaces(text)
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    n = len(text)
    i = 0
    while i < n:
        end = _find_cut(text, i, chunk_size)
        end = min(end, n)
        chunk = text[i:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        # 下一窗口起点：后退 overlap 个字符实现重叠，同时保证有进展
        i = max(end - overlap, i + 1)
    return chunks


def _collapse_spaces(text: str) -> str:
    out: list[str] = []
    for ch in text:
        if ch in " \t　":
            if out and out[-1] != " ":
                out.append(" ")
        else:
            out.append(ch)
    return "".join(out)


def _find_cut(text: str, start: int, chunk_size: int) -> int:
    """在 text[start : start+chunk_size] 内找最合适的分隔点，返回其后的位置。"""
    lo = start + max(1, int(chunk_size * 0.5))  # 至少切走一半，避免碎块
    hi = min(start + chunk_size, len(text))
    window = text[lo:hi]
    for sep in _SEPARATORS:
        idx = window.rfind(sep)
        if idx >= 0:
            return lo + idx + len(sep)
    return hi  # 区间内无分隔符，硬切


def split_by_structure(
    text: str, chunk_size: int = 500, overlap: int = 50, cut_level: int = 2
) -> list[str]:
    """按标题等级切块：等级 ≤ cut_level 的标题决定块边界，块首带标题行。

    处理规则：
    - 解析 markdown 标题（# ～ ######）与正文行；整篇没有标题的文档退化为 split_text()。
    - 从一条「等级 ≤ cut_level」的标题起，到下一条同级/更粗标题前，构成一个候选块；
      等级更高的标题不作为切点，保留在块内作小节定位；连续标题累积成块首前缀（避免孤标题）。
    - 候选块字符数 ≤ chunk_size → 直接成块；
      超过上限时：块内存在更细标题 → 按这些更细标题再拆（递归，块首各自带标题）；
      没有任何更细标题 → 整块按 split_text() 字符兜底拆。

    因此正常块之间没有字符重叠，overlap 只作用于「整块字符兜底拆」那一步。
    chunk_size<=0 时整段一块。
    """
    text = (text or "").strip()
    if not text:
        return []
    if chunk_size <= 0:
        return [text]

    rows = _scan_rows(text)
    if not any(lvl for lvl, _ in rows):  # 没有任何标题 → 没有结构可用
        return split_text(text, chunk_size, overlap)

    cut_level = max(1, min(6, int(cut_level)))
    chunks: list[str] = []
    for block in _base_blocks(rows, cut_level):
        chunks.extend(_bounded(block, cut_level, chunk_size, overlap))
    return chunks


def _scan_rows(text: str) -> list[tuple[int, str]]:
    """把文本扫成 (level, text) 行序列；level=0 表示普通正文行，1~6 表示 # 标题行。"""
    rows: list[tuple[int, str]] = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        m = _HEAD_LVL_RE.match(s)
        if m:
            rows.append((len(m.group(1)), s))
        else:
            rows.append((0, s))
    return rows


def _base_blocks(rows: list[tuple[int, str]], cut: int) -> list[list[tuple[int, str]]]:
    """按「等级 ≤ cut 的标题」把行切成候选块，块首尽量带标题、避免孤标题。

    连续标题（无正文）会累积成块首前缀；遇到含正文的块后，下一条边界标题才结束本块。
    """
    blocks: list[list[tuple[int, str]]] = []
    cur: list[tuple[int, str]] = []
    has_body = False

    def is_boundary(r: tuple[int, str]) -> bool:
        lvl = r[0]
        return 0 < lvl <= cut

    for r in rows:
        if is_boundary(r):
            if cur and has_body:  # 上一块已含正文 → 该标题开启新块
                blocks.append(cur)
                cur = []
                has_body = False
            cur.append(r)
        else:
            cur.append(r)
            if r[0] == 0:
                has_body = True
    if cur:
        blocks.append(cur)
    return blocks


def _bounded(
    rows: list[tuple[int, str]], lo: int, cap: int, overlap: int
) -> list[str]:
    """把一个候选块压到 ≤ cap：优先用更细标题（等级 > lo）再拆，否则字符兜底。"""
    if _rows_len(rows) <= cap:
        return [_rows_text(rows)]

    inner = sorted({lvl for lvl, _ in rows if 0 < lvl and lvl > lo})
    if not inner:  # 没有更细标题可作边界 → 整块字符兜底（overlap 在此生效）
        return split_text(_rows_text(rows), cap, overlap)

    nxt = inner[0]
    groups: list[list[tuple[int, str]]] = []
    cur: list[tuple[int, str]] = []
    for r in rows:
        if 0 < r[0] == nxt:  # 在最接近的下一级标题处断开
            if cur:
                groups.append(cur)
            cur = [r]
        else:
            cur.append(r)
    if cur:
        groups.append(cur)

    out: list[str] = []
    for g in groups:
        out.extend(_bounded(g, nxt, cap, overlap))
    return out


def _rows_len(rows: list[tuple[int, str]]) -> int:
    return sum(len(t) for _, t in rows) + max(0, len(rows) - 1)


def _rows_text(rows: list[tuple[int, str]]) -> str:
    return "\n".join(t for _, t in rows).strip()
