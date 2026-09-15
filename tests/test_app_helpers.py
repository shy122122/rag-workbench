"""请求体边界上的几个纯函数的约束。

这些函数是「不可信输入」唯一进来的地方：请求体里的 history、上传的文件名、
Top-K / 阈值。它们错了不会报错，只会安静地放大成本或写出目录，所以必须钉住。
"""
import pytest

import app


# ---------------------------------------------------------------- 上传文件名
@pytest.mark.parametrize(
    "raw",
    [
        "../../../etc/passwd",
        "..\\..\\Windows\\System32\\config",
        "/etc/shadow",
        "C:\\Windows\\win.ini",
        "....//....//evil.txt",
    ],
)
def test_safe_name_strips_directory_components(raw):
    out = app._safe_name(raw)
    assert "/" not in out and "\\" not in out
    assert ".." not in out, "上跳符号必须被清掉"
    assert out


def test_safe_name_keeps_chinese_and_extension():
    assert app._safe_name("磷酸铁锂 电池/报告 v2.pdf") == "报告_v2.pdf"
    assert app._safe_name("说明.md") == "说明.md"


def test_safe_name_never_returns_empty_for_garbage():
    assert app._safe_name("///") != ""
    assert app._safe_name("") != ""


# ---------------------------------------------------------------- 多轮上文
def test_history_rejects_non_list():
    assert app._clean_history(None) == []
    assert app._clean_history("不是数组") == []
    assert app._clean_history({"question": "甲", "answer": "乙"}) == []


def test_history_drops_incomplete_turns():
    raw = [
        {"question": "有问有答", "answer": "答"},
        {"question": "只有问题", "answer": ""},
        {"question": "", "answer": "只有回答"},
        {"question": "  ", "answer": "  "},
        {"question": "答是空的", "answer": None},
        "不是对象",
    ]
    assert app._clean_history(raw) == [{"question": "有问有答", "answer": "答"}]


def test_history_keeps_only_the_last_few_turns():
    raw = [{"question": f"q{i}", "answer": f"a{i}"} for i in range(20)]
    out = app._clean_history(raw)
    assert len(out) == app.HISTORY_MAX_TURNS
    assert out[-1]["question"] == "q19", "保留的应是最近几轮，不是最早几轮"


def test_history_caps_each_field_length():
    out = app._clean_history([{"question": "问" * 99999, "answer": "答" * 99999}])
    assert len(out[0]["question"]) == app.HISTORY_MAX_CHARS
    assert len(out[0]["answer"]) == app.HISTORY_MAX_CHARS


def test_history_window_backfills_past_junk():
    """夹在中间的空壳不该把有效轮次挤出窗口。"""
    raw = [{"question": "有效", "answer": "答"}]
    raw += [{"question": "", "answer": ""} for _ in range(app.HISTORY_MAX_TURNS * 2)]
    assert app._clean_history(raw) == [{"question": "有效", "answer": "答"}]


def test_history_scan_is_bounded():
    """超长请求体不该被逐条遍历——只看尾部窗口。"""
    raw = [{"question": f"q{i}", "answer": f"a{i}"} for i in range(app.HISTORY_SCAN + 500)]
    out = app._clean_history(raw)
    assert len(out) == app.HISTORY_MAX_TURNS
    assert out[-1]["question"] == f"q{app.HISTORY_SCAN + 499}"


def test_history_stringifies_non_string_fields():
    out = app._clean_history([{"question": 123, "answer": ["a"]}])
    assert out == [{"question": "123", "answer": "['a']"}]


def test_history_trims_whitespace():
    out = app._clean_history([{"question": "  问  ", "answer": "\n答\t"}])
    assert out == [{"question": "问", "answer": "答"}]


# ---------------------------------------------------------------- 匹配范围
def test_doc_ids_absent_or_empty_means_whole_kb():
    """None 是「全库」。过滤只该让结果变少，绝不能因为参数畸形把整库变成 0 命中。"""
    assert app._clean_doc_ids(None) is None
    assert app._clean_doc_ids([]) is None
    assert app._clean_doc_ids(["", "  "]) is None
    assert app._clean_doc_ids("不是数组") is None
    assert app._clean_doc_ids({"a": 1}) is None


def test_doc_ids_keeps_valid_ids():
    assert app._clean_doc_ids(["a", "b"]) == {"a", "b"}
    assert app._clean_doc_ids([" a ", "b"]) == {"a", "b"}, "两侧空白要剥掉"
    assert app._clean_doc_ids([123]) == {"123"}, "数字 id 也认"


def test_doc_ids_drops_garbage_entries_but_keeps_the_rest():
    assert app._clean_doc_ids(["a", None, {"x": 1}, ["b"], "b"]) == {"a", "b"}


def test_doc_ids_is_bounded():
    """超长数组不该被逐条处理成字符串。"""
    out = app._clean_doc_ids([f"d{i}" for i in range(app.DOC_ID_MAX + 500)])
    assert len(out) == app.DOC_ID_MAX


def test_doc_ids_caps_each_id_length():
    out = app._clean_doc_ids(["x" * 9999])
    assert len(next(iter(out))) == 128


# ---------------------------------------------------------------- 参数解析
def test_num_returns_none_for_blank_and_bad_values():
    assert app._num({}, "top_k", int) is None
    assert app._num({"top_k": ""}, "top_k", int) is None
    assert app._num({"top_k": None}, "top_k", int) is None
    assert app._num({"top_k": "abc"}, "top_k", int) is None
    assert app._num({"top_k": "3"}, "top_k", int) == 3
    assert app._num({"min_score": "0.5"}, "min_score", float) == 0.5


def test_to_bool_accepts_common_truthy_spellings():
    for v in [True, 1, "1", "true", "TRUE", " yes ", "on"]:
        assert app._to_bool(v) is True, v
    for v in [False, 0, "0", "false", "no", "off", "", None, "随便"]:
        assert app._to_bool(v) is False, v


def test_to_bool_default_only_applies_to_missing_values():
    assert app._to_bool(None, default=True) is True
    assert app._to_bool("", default=True) is False, "显式传了空串按 false，不该回落到默认值"


# ---------------------------------------------------------------- 上传大小限制
def test_upload_limit_is_100mb():
    assert app.MAX_UPLOAD == 100 * 1024 * 1024
    assert app._READ_CHUNK <= app.MAX_UPLOAD
