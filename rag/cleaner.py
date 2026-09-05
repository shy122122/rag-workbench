"""入库前数据清洗：基础规范化 + 可选的噪声行剔除与全篇去重复行。

clean() 返回 (清洗后文本, 报告)。报告供界面展示"删了什么、删了几行几字"。
"""
from __future__ import annotations

import re

# 统一换行前先清掉的零宽/控制字符：BOM(U+FEFF)、零宽空格(U+200B)、
# 零宽不连字(U+200C)、零宽连字(U+200D)、词连接符(U+2060)
_ZERO_WIDTH = str.maketrans({ord(c): None for c in "﻿​‌‍⁠"})

# 行内连续空格压平（正文里缩进/表格对齐在切分前意义不大）
_COLLAPSE = re.compile(r"[ \t]{2,}")

# 噪声行：页码 / 页眉页脚
_PAGE_CN = re.compile(r"^第\s*[0-9一二三四五六七八九十百千万]{1,8}\s*页\s*$")
_PAGE_CN_TOTAL = re.compile(
    r"^第\s*[0-9一二三四五六七八九十百千万]{1,8}\s*页\s*(?:共|/)\s*[0-9]+\s*页?\s*$"
)
_PAGE_SLASH = re.compile(r"^[0-9]{1,4}\s*/\s*[0-9]{1,4}\s*页?\s*$")
_PAGE_EN = re.compile(r"^page\s*[0-9]{1,5}(\s*of\s*[0-9]{1,5})?\s*$", re.IGNORECASE)
_PAGE_CN_TOTAL2 = re.compile(r"^共\s*[0-9]{1,5}\s*页\s*$")

# 纯分隔线：整行由 - _ = · * ~ — 与空白构成
_SEP = re.compile(r"^[\s\-—_=·*~—]+$")

_PAGE_RULES = (_PAGE_CN, _PAGE_CN_TOTAL, _PAGE_SLASH, _PAGE_EN, _PAGE_CN_TOTAL2)


def _is_noise_line(ln: str) -> bool:
    """整行是否像页码/页眉页脚/分隔线。ln 应为已去除首尾空白的非空行。"""
    for rx in _PAGE_RULES:
        if rx.match(ln):
            return True
    return len(ln) >= 3 and bool(_SEP.match(ln))


def clean(text: str, noise: bool = True, dedup: bool = True) -> tuple[str, dict]:
    """清洗文本，返回 (清洗结果, 报告)。

    报告结构：{rules, before_chars, after_chars, removed_lines, removed_chars,
    samples: [{reason, text}] 被删行示例前 8 条}。
    """
    before = len(text)
    removed: list[dict] = []
    rules: list[str] = []

    # ---- 1) 基础规范化（始终执行）----
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = t.translate(_ZERO_WIDTH)
    lines = [_COLLAPSE.sub(" ", ln.strip()) for ln in t.split("\n")]
    rules.append("基础规范化：统一换行、去零宽字符、行首尾去空白、连续空格压平")

    # ---- 2) 噪声行剔除（页码/页眉页脚/分隔线）----
    if noise:
        kept: list[str] = []
        n_noise = 0
        for ln in lines:
            if ln and _is_noise_line(ln):
                removed.append({"reason": "页码/页眉/分隔线", "text": ln})
                n_noise += 1
            else:
                kept.append(ln)
        lines = kept
        rules.append(
            f"剔除噪声行（页码/页眉/分隔线）{n_noise} 行"
            if n_noise
            else "噪声行规则已启用：未发现页码/页眉/分隔线"
        )

    # ---- 3) 全篇去重复行（保留首次出现，仅针对非空行）----
    if dedup:
        kept = []
        seen: set[str] = set()
        n_dup = 0
        for ln in lines:
            if ln and ln in seen:
                removed.append({"reason": "全篇重复行", "text": ln})
                n_dup += 1
            else:
                if ln:
                    seen.add(ln)
                kept.append(ln)
        lines = kept
        rules.append(
            f"去除全篇重复行 {n_dup} 行（保留首次出现）"
            if n_dup
            else "去重复行规则已启用：未发现重复行"
        )

    # ---- 4) 压缩空行：连续空行压到至多 1，并去掉首尾空行 ----
    out: list[str] = []
    prev_blank = False
    for ln in lines:
        if not ln:
            if prev_blank:
                continue
            prev_blank = True
        else:
            prev_blank = False
        out.append(ln)
    while out and not out[0]:
        out.pop(0)
    while out and not out[-1]:
        out.pop()
    rules.append("压缩连续空行，仅保留至多 1 个空行用于分段")

    cleaned = "\n".join(out)
    after = len(cleaned)
    report = {
        "rules": rules,
        "before_chars": before,
        "after_chars": after,
        "removed_lines": len(removed),
        "removed_chars": before - after,
        "samples": [{"reason": r["reason"], "text": r["text"]} for r in removed[:8]],
    }
    return cleaned, report
