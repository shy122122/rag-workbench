"""公网演示门禁的约束。

这层是唯一挡在「真实付费模型」前面的东西，漏一个洞就是别人的手速换你的账单，
所以每一条拒绝路径都要钉死：没密码、密码错、太频繁、额度用完。

注意不用 starlette 的 TestClient——它依赖 httpx，而本项目跟着 openai 3.x 用的是
httpx2（重命名分支），装不了。这里直接把中间件当 ASGI 应用调，不引新依赖。
"""
import asyncio
import base64
import json
from pathlib import Path

import pytest

from rag import deploy


# ---------------------------------------------------------------- 测试夹具
async def _ok_app(scope, receive, send):
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b'{"ok": true}'})


def _basic(user: str, password: str) -> str:
    raw = f"{user}:{password}".encode()
    return "Basic " + base64.b64encode(raw).decode()


def _call(method: str, path: str, auth: str | None = None, ip: str = "1.2.3.4"):
    """跑一次请求，返回 (状态码, 响应体, 响应头)。"""
    headers = [("x-forwarded-for", ip)]
    if auth:
        headers.append(("authorization", auth))

    scope = {
        "type": "http", "http_version": "1.1", "method": method, "path": path,
        "raw_path": path.encode(), "query_string": b"", "root_path": "",
        "scheme": "http", "headers": [(k.encode(), v.encode()) for k, v in headers],
        "client": (ip, 12345), "server": ("testserver", 80),
    }
    sent: list[dict] = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(msg):
        sent.append(msg)

    asyncio.run(deploy.GateMiddleware(_ok_app)(scope, receive, send))
    start = next(m for m in sent if m["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in sent
                    if m["type"] == "http.response.body")
    headers = {k.decode().lower(): v.decode() for k, v in start.get("headers", [])}
    return start["status"], body, headers


@pytest.fixture
def gated(monkeypatch):
    """开一个干净的门禁：设好密码，并把配额计数器重置。"""
    monkeypatch.setenv("DEMO_PASSWORD", "面试官专用")
    monkeypatch.setenv("DEMO_USER", "demo")
    monkeypatch.delenv("DEMO_DAILY_LIMIT", raising=False)
    monkeypatch.delenv("DEMO_RATE_PER_MIN", raising=False)
    monkeypatch.setattr(deploy, "GATE", deploy.Gate())


# ---------------------------------------------------------------- 鉴权
def test_gate_off_by_default(monkeypatch):
    """没设密码就该完全空转，否则本地开发第一步就被自己挡住。"""
    monkeypatch.delenv("DEMO_PASSWORD", raising=False)
    assert deploy.enabled() is False
    assert _call("GET", "/api/query")[0] == 200


@pytest.mark.parametrize("method,path", [("GET", "/"), ("GET", "/api/config"),
                                         ("POST", "/api/query"), ("DELETE", "/api/knowledge")])
def test_rejects_without_credentials(gated, method, path):
    assert _call(method, path)[0] == 401


def test_401_carries_www_authenticate(gated):
    """缺了这个头浏览器不会弹登录框，访客只会看到一段 JSON，不知道要输密码。"""
    status, _, headers = _call("GET", "/api/config")
    assert status == 401
    assert headers.get("www-authenticate", "").lower().startswith("basic ")


def test_401_message_is_actionable(gated):
    """拒绝了得说清楚怎么进来，不然面试官只会以为站点挂了。"""
    _, body, _ = _call("GET", "/api/config")
    assert "密码" in body.decode("utf-8")


def test_rejects_wrong_password(gated):
    assert _call("GET", "/api/config", _basic("demo", "猜的"))[0] == 401


def test_rejects_wrong_user(gated):
    assert _call("GET", "/api/config", _basic("admin", "面试官专用"))[0] == 401


def test_accepts_correct_credentials(gated):
    assert _call("GET", "/api/config", _basic("demo", "面试官专用"))[0] == 200


@pytest.mark.parametrize("header", ["", "Basic", "Basic !!!不是base64",
                                    "Bearer abc", "Basic " + base64.b64encode(b"no-colon").decode()])
def test_malformed_auth_header_does_not_crash(gated, header):
    """鉴权头是未可信输入：解不开就当作没通过，不能把 500 抛给访客。"""
    assert _call("GET", "/api/config", header)[0] == 401


def test_healthz_is_open(gated):
    """平台探活必须免鉴权，否则实例会被判成不健康而反复重启。"""
    assert _call("GET", "/healthz")[0] == 200


# ---------------------------------------------------------------- 配额
def test_rate_limit_blocks_burst(gated, monkeypatch):
    monkeypatch.setenv("DEMO_RATE_PER_MIN", "3")
    monkeypatch.setattr(deploy, "GATE", deploy.Gate())
    codes = [_call("POST", "/api/query", _basic("demo", "面试官专用"))[0] for _ in range(4)]
    assert codes[:3] == [200, 200, 200]
    assert codes[3] == 429


def test_rate_limit_is_per_ip(gated, monkeypatch):
    monkeypatch.setenv("DEMO_RATE_PER_MIN", "1")
    monkeypatch.setattr(deploy, "GATE", deploy.Gate())
    assert _call("POST", "/api/query", _basic("demo", "面试官专用"), ip="1.1.1.1")[0] == 200
    assert _call("POST", "/api/query", _basic("demo", "面试官专用"), ip="1.1.1.1")[0] == 429
    assert _call("POST", "/api/query", _basic("demo", "面试官专用"), ip="2.2.2.2")[0] == 200


def test_daily_limit_blocks_across_ips(gated, monkeypatch):
    """每日额度是全局的：换 IP 也绕不过去，否则它挡不住「额度被吃光」这件事。"""
    monkeypatch.setenv("DEMO_DAILY_LIMIT", "2")
    monkeypatch.delenv("DEMO_RATE_PER_MIN", raising=False)
    monkeypatch.setattr(deploy, "GATE", deploy.Gate())
    auth = _basic("demo", "面试官专用")
    assert _call("POST", "/api/query", auth, ip="1.1.1.1")[0] == 200
    assert _call("POST", "/api/query", auth, ip="2.2.2.2")[0] == 200
    assert _call("POST", "/api/query", auth, ip="3.3.3.3")[0] == 429


def test_free_endpoints_do_not_consume_quota(gated, monkeypatch):
    """读配置、列文档不该计入额度，否则面试官随便点点就把额度耗光了。"""
    monkeypatch.setenv("DEMO_DAILY_LIMIT", "1")
    monkeypatch.setattr(deploy, "GATE", deploy.Gate())
    auth = _basic("demo", "面试官专用")
    for _ in range(5):
        assert _call("GET", "/api/config", auth)[0] == 200
    assert _call("POST", "/api/query", auth)[0] == 200  # 额度还在
    assert _call("POST", "/api/query", auth)[0] == 429


def test_zero_limit_means_unlimited(gated, monkeypatch):
    monkeypatch.setenv("DEMO_DAILY_LIMIT", "0")
    monkeypatch.setenv("DEMO_RATE_PER_MIN", "0")
    monkeypatch.setattr(deploy, "GATE", deploy.Gate())
    auth = _basic("demo", "面试官专用")
    assert all(_call("POST", "/api/query", auth)[0] == 200 for _ in range(10))


def test_bad_limit_value_falls_back(monkeypatch):
    monkeypatch.setenv("DEMO_DAILY_LIMIT", "三百")
    assert deploy._env_int("DEMO_DAILY_LIMIT", 200) == 200


# ---------------------------------------------------------------- 示例向量库
def _make_kb(root: Path, docs: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "meta.json").write_text(json.dumps([{"doc_name": f"d{i}"} for i in range(docs)]),
                                    encoding="utf-8")
    (root / "vectors.npy").write_bytes(b"\x00" * 8)


def test_seed_restores_empty_kb(tmp_path):
    """容器重建后 data/kb 是空的，得有东西顶上，否则面试官打开是个空壳。"""
    _make_kb(tmp_path / "seed" / "kb", 3)
    assert deploy.seed_kb_if_empty(tmp_path) is True
    assert (tmp_path / "data" / "kb" / "meta.json").exists()
    assert (tmp_path / "data" / "kb" / "vectors.npy").exists()


def test_seed_does_not_touch_existing_kb(tmp_path):
    """已有内容就不许动——访客上传的文档不能因为一次重启被示例覆盖掉。"""
    _make_kb(tmp_path / "seed" / "kb", 3)
    _make_kb(tmp_path / "data" / "kb", 7)
    before = (tmp_path / "data" / "kb" / "meta.json").read_text(encoding="utf-8")
    assert deploy.seed_kb_if_empty(tmp_path) is False
    assert (tmp_path / "data" / "kb" / "meta.json").read_text(encoding="utf-8") == before


def test_seed_absent_is_noop(tmp_path):
    """没有种子目录（本地开发就是这样）时不能报错。"""
    assert deploy.seed_kb_if_empty(tmp_path) is False
    assert not (tmp_path / "data" / "kb" / "meta.json").exists()


def test_client_ip_prefers_forwarded_header():
    """反代后面 request.client.host 永远是网关地址，按它限流等于全局限流。"""
    class _Req:
        class client:
            host = "10.0.0.1"
        headers = {"x-forwarded-for": "203.0.113.7, 10.0.0.1"}

    assert deploy._client_ip(_Req()) == "203.0.113.7"
