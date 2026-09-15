"""入库前清洗的行为约束。

清洗是「静默改数据」的操作——删错了用户很难发现，所以既要有删除效果，
也要保证开关关掉时一个字都没动。报告（rules / samples）是界面唯一的解释来源。
"""
from rag.cleaner import clean


def _lines(report):
    return [s["text"] for s in report["samples"]]


# ---------------------------------------------------------------- 基础规范化
def test_normalization_always_runs():
    text = "﻿甲​乙\r\n丙\r丁\r\n\r\n\r\n戊"
    out, rep = clean(text, noise=False, dedup=False)
    assert "﻿" not in out and "​" not in out
    assert "\r" not in out
    assert out == "甲乙\n丙\n丁\n\n戊"
    assert rep["rules"], "即使只做基础规范化也要给出说明"


def test_inline_spaces_collapsed_and_lines_trimmed():
    out, _ = clean("  甲    乙  \n\t丙\t\t丁\t", noise=False, dedup=False)
    assert out == "甲 乙\n丙 丁"


def test_consecutive_blank_lines_are_compressed():
    out, _ = clean("甲\n\n\n\n\n乙", noise=False, dedup=False)
    assert out == "甲\n\n乙"


def test_leading_and_trailing_blank_lines_are_stripped():
    out, _ = clean("\n\n\n甲\n乙\n\n\n", noise=False, dedup=False)
    assert out == "甲\n乙"


def test_before_after_chars_are_reported():
    _, rep = clean("甲    乙", noise=False, dedup=False)
    assert rep["before_chars"] == len("甲    乙")
    assert rep["after_chars"] == len("甲 乙")


# ---------------------------------------------------------------- 噪声行
NOISE_CASES = [
    "第 3 页",
    "第3页",
    "第 12 页 共 30 页",
    "第 3 页 / 12 页",
    "3 / 12",
    "3/12页",
    "Page 7",
    "Page 7 of 12",
    "PAGE 7 OF 12",
    "共 30 页",
]


def test_page_number_variants_are_removed():
    for line in NOISE_CASES:
        out, rep = clean(f"正文甲\n{line}\n正文乙", noise=True, dedup=False)
        assert out == "正文甲\n正文乙", f"{line!r} 应被当作页码删除"
        assert _lines(rep) == [line]


SEPARATOR_CASES = ["----------", "___", "====", "·····", "~~~~~", "————"]


def test_separator_lines_are_removed():
    for line in SEPARATOR_CASES:
        out, _ = clean(f"甲\n{line}\n乙", noise=True, dedup=False)
        assert out == "甲\n乙", f"{line!r} 应被当作分隔线删除"


def test_short_separators_are_kept():
    """两个字符以内不算分隔线——正文里出现「--」的概率比页眉高。"""
    out, _ = clean("甲\n--\n乙", noise=True, dedup=False)
    assert out == "甲\n--\n乙"


def test_noise_removal_can_be_turned_off():
    text = "正文甲\n第 3 页\n----------\n正文乙"
    out, rep = clean(text, noise=False, dedup=False)
    assert out == text
    assert rep["removed_lines"] == 0
    assert not any("噪声" in r for r in rep["rules"])


def test_body_text_looking_like_page_numbers_is_kept():
    for line in ["第 3 章 概述", "3 / 12 的记录", "页码在中间 第 3 页 后面还有字"]:
        out, _ = clean(f"甲\n{line}\n乙", noise=True, dedup=False)
        assert line in out, f"{line!r} 是正文，不能删"


# ---------------------------------------------------------------- 去重复行
def test_duplicate_lines_keep_the_first_occurrence():
    out, rep = clean("表头\n正文甲\n表头\n正文乙\n表头", noise=True, dedup=True)
    assert out == "表头\n正文甲\n正文乙"
    assert rep["removed_lines"] == 2
    assert _lines(rep) == ["表头", "表头"]
    assert rep["samples"][0]["reason"] == "全篇重复行"


def test_blank_lines_are_not_treated_as_duplicates():
    out, rep = clean("甲\n\n乙\n\n丙", noise=False, dedup=True)
    assert out == "甲\n\n乙\n\n丙"
    assert rep["removed_lines"] == 0


def test_dedup_can_be_turned_off():
    text = "表头\n正文甲\n表头"
    out, _ = clean(text, noise=False, dedup=False)
    assert out == text


def test_blank_line_does_not_reset_duplicate_detection():
    """「表头 / 空行 / 表头」也是重复，空行不该把它隔断。"""
    out, _ = clean("表头\n\n表头\n正文", noise=True, dedup=True)
    assert out == "表头\n\n正文"


# ---------------------------------------------------------------- 报告
def test_report_shape_is_stable():
    out, rep = clean("甲\n第 1 页\n甲", noise=True, dedup=True)
    assert set(rep) == {
        "rules", "before_chars", "after_chars", "removed_lines", "removed_chars", "samples"
    }
    assert isinstance(rep["rules"], list) and len(rep["rules"]) >= 1
    assert rep["removed_lines"] == 2
    assert rep["removed_chars"] == rep["before_chars"] - rep["after_chars"]
    assert out == "甲"


def test_samples_are_capped_at_eight():
    text = "\n".join([f"正文{i}" for i in range(20)] + ["第 1 页"] * 15)
    _, rep = clean(text, noise=True, dedup=False)
    assert rep["removed_lines"] == 15
    assert len(rep["samples"]) == 8


def test_rules_mention_when_nothing_was_found():
    _, rep = clean("干净的正文。", noise=True, dedup=True)
    assert rep["removed_lines"] == 0
    assert any("未发现" in r for r in rep["rules"])


def test_clean_of_empty_input():
    out, rep = clean("", noise=True, dedup=True)
    assert out == ""
    assert rep["removed_lines"] == 0
    assert rep["before_chars"] == 0
