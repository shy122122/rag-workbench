"""递归字符切分器：中文友好，按优先级找分隔符切块并保留重叠。

分隔符优先级：段落空行 > 换行 > 句末标点(。！？…;) > 空格/逗号 > 单字符兜底。
"""
from __future__ import annotations

_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "…", " ", "、", ","]


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
