# rag-workbench · 开发协作说明

个人自建「真实可用 + 逐步可视化」的 RAG 工作台（2026-09-05 起）：上传文档 → 解析 → 切分 → 向量化 → 入库 → 检索 → 组装上下文 → 流式生成回答，前端逐步点亮每一步并展示中间结果（分块文本 / 向量维度 / Top-K 相似度 / 完整 Prompt / 带来源引用的回答）。

## 启动与访问

- 依赖装在项目 `.venv`。启动命令（在项目根）：
  - `.venv/Scripts/python -m uvicorn app:app --port 8000` 或 `source .venv/Scripts/activate && uvicorn app:app --port 8000`
  - 页面 http://127.0.0.1:8000
  - 根目录还有 `start.bat`（双击用）。
- 后端改动后需**重启** uvicorn 才生效（平时没开 `--reload`）。
- 前端静态文件改动后浏览器强刷（Ctrl+F5）。

## 跑测试

```bash
.venv/Scripts/python -m pytest      # 约 3 秒
```

- Windows 默认 GBK 控制台会把中文断言打成乱码，先 `set PYTHONUTF8=1`；
  根目录 `test.bat` 已替你设好（`PYTHONUTF8=1` + `PYTHONIOENCODING=utf-8`），双击即可。
- `tests/conftest.py` 会把项目根加进 `sys.path` 并把 stdout/stderr 切成 UTF-8，
  所以直接 `pytest` 也不会因编码问题挂。
- 用例集中在「出过问题的地方」：向量库损坏后的恢复路径、切分的块大小与重叠边界、
  清洗的误删风险、以及请求体里不可信的文件名与多轮上文。

## 配置

- API Key 只存 `.env`（`DASHSCOPE_API_KEY`）；界面可调参数存 `config.json`（模型 qwen-plus / text-embedding-v3 / dimensions 1024 / chunk_strategy recursive|structure / structure_level 2 / chunk_size 500 / chunk_overlap 50 / top_k 5 / retrieval_min_score / 清洗开关 / **rerank_enabled、rerank_model、recall_k、neighbor_window**），两者都走 `config.py`。
- 模型走阿里云百炼 OpenAI 兼容接口（base_url `https://dashscope.aliyuncs.com/compatible-mode/v1`）；**精排不走兼容接口**，走百炼原生接口 `native_base_url = https://dashscope.aliyuncs.com/api/v1` 的 `/services/rerank/text-rerank/text-rerank`（model `gte-rerank-v2`）。
- `config.load_config()` 带 mtime 缓存：外部直接改 `config.json` 也会生效，但写入必须走 `config.save_config()`（会失效缓存）。

## 目录速览

- `app.py` FastAPI 入口 + 全部 REST/SSE 路由；`static/` 单页 Tailwind 前端（全中文）。
- `rag/`：loader(解析 pdf/docx/txt/md)、cleaner(入库前清洗：规范化/去页码页眉/去重复行)、chunker(递归字符切分 / 按段落结构两种策略)、embeddings、llm(流式)、rerank(百炼原生精排)、clients(AsyncOpenAI 客户端缓存)、store(本地向量库)、events(SSE)、errors(带中文提示的领域异常)、logging_setup(控制台 + 轮转文件日志，logger 名 `"rag"`，`propagate=False`)、pipeline(建库/问答/内容匹配三条可视化流水线)。
- `tests/` pytest 用例 + `conftest.py`（UTF-8 输出）；根目录 `pytest.ini`、`test.bat`。
- `data/kb/` 向量库持久化（numpy+JSON，重启不丢）；`data/logs/workbench.log` 运行日志（2MB × 4 轮转）；`examples/` 内置示例（点击一键载入）。

## 检索链路

`q_embed → retrieve(向量粗排，召回 recall_k 条) → rerank(精排取 top_k) → context(邻块扩展后拼 Prompt) → generate`

- 精排关掉时 `recall_k` 无效，直接取 top_k。
- `neighbor_window` 控制邻块扩展：命中块前后各带 N 块进上下文（small-to-big），窗口只在本篇文档内滑动。
- 精排开启后会额外返回「紧随其后被召回数截断的那几条」，用来判断召回数是否设小了。
- 内容匹配可限定**文档范围**（`store.search(doc_ids=…)`）。必须**先在子集内排名再取 Top-K**——
  先取全库 Top-K 再过滤的话，范围外的块会把名额占光，范围一窄就空手而归。
  步骤事件里的 `scope_docs` / `total_docs` 记录了当轮实际范围；`above` 也只在范围内计数。

## 前端（static/，无构建，Tailwind Play CDN + 设计变量）

- **回答渲染**：CDN 加载 `marked@12` + `DOMPurify@3`（jsDelivr）。模型输出**不可信**，
  一律 `DOMPurify.sanitize(marked.parse(text))`，禁 `style`/`iframe`/`form` 等标签与 `style` 属性。
  CDN 挂了会退回纯文本（`MD_OK` 判定），不会白屏。
- `[n]` 引用在 **Markdown 渲染之后**用 TreeWalker 注入（`attachCites`），跳过 `CODE/PRE/A` 的父节点，
  所以代码块里的 `[1]` 不会变成引用；只有本回合真实存在的 rank 才挂，幽灵编号保持纯文本。
- Tailwind preflight 会清掉 list 符号与标题字号，必须靠 `.md` 那段 CSS 给回来——删了会退化成一行行平文本。
- 流式回答用 `requestAnimationFrame` 节流重渲染（`scheduleLiveRender`），别每个 token 都 innerHTML。
- **对话历史**存 `localStorage["rag.history"]`（键与常量在 app.js「对话历史」段），
  刷新不丢；追问只带最近 `HIST_TURNS`（3）轮，后端还会再截一次。
  只有**拿到完整回答**才入库——半截回答当上文会污染下一轮的指代。
- **主题**：`data-theme="light|dark"` 挂 `<html>`，颜色全走 CSS 变量；
  `<head>` 里有一段 pre-paint IIFE 读 `localStorage["rag.theme"]`，避免刷新闪白。
  重色块用 `color-mix(in srgb, var(--accent) N%, transparent)`，深色下自动跟着变。
- **KB 告警横幅**：`#kb-alert`，数据来自 `/api/health` 的 `problems` / `writable=false`。
  成功写入后 `store._save()` 会清空 `problems`，否则修好了横幅还一直挂着。
- **匹配范围**：内容匹配模式下可勾选参与匹配的文档（`#match-scope`，默认全选）。
  前端 `scope === null` 表示全选，这样**新上传的文档会自动进范围**；
  删文档时 `refreshSidebar` 会把已失效的 id 从选择里摘掉。
  选了 0 篇不允许提交（前端拦）。`/api/match` 的 `doc_ids` 为空/非法一律按**全库**处理。

## 数据安全

- `meta.json` / `vectors.npy` 按行对齐、追加有序，所以**共同前缀必然自洽**。
- 落盘走「先写临时文件 → `os.replace` 原子替换」：磁盘上要么是上一份完整内容、要么是新的一份。
- 载入时行数不一致 → 按共同前缀截断（只损失最后一批），不丢整库；
  文件彻底读不出来 → 按空库启动但**禁止任何写入**，原文件一字节不动，界面给横幅。
  此时点「清空知识库」才会解锁——那是用户明确放弃原数据的动作。

## 已确认的技术要点（避免再踩坑）

- **切分/嵌入参数改动后需「清空知识库」重建**；维度不一致现在会抛 `DimMismatch`（中文提示），不再是一串 numpy 英文报错。
- 精排/召回数/邻块窗口改动**无需重建**，立即生效。
- store 向量化曾对 `list` 调 `.astype()` 崩溃——embed_texts 返回 Python list，须先 `np.asarray` 再转类型。
- openai 3.x 依赖 `httpx2`（httpx 的重命名分支），不要再去 `import httpx`。
- 流式生成要传 `stream_options={"include_usage": True}`，且**必须先读 `chunk.usage` 再判断 `chunk.choices`**——携带 usage 的最后一个分块 choices 是空列表，顺序反了用量就恒为 0。
- 「载入示例」已做幂等（同名文档先删后建）。
- 不支持旧版 .doc 与无文字层的扫描版 PDF。

## 验证状态

- 2026-09-05 已用用户真 Key 端到端验证：建库 3 篇 9 块、问答检索+生成+[n] 引用正常。
- 2026-09-15 新增精排 / 邻块扩展 / 截断展示 / 停止按钮 / 流式用量修复，后端已用真 Key 冒烟通过（粗排 20 → 精排 3，名次变动与挤出项均正确）。
- 2026-09-15 产品化改造，全部用真 Key 验证过：
  - **数据安全**：原子写盘 + 行数不一致按共同前缀截断 + 读不出时锁写，`store._save()` 成功后清空 `problems`。
  - **日志**：`data/logs/workbench.log` 记录请求 / 建库 / 问答 / 异常堆栈。
  - **多轮对话**：带 1000 轮 × 5000 字的恶意 history 被截到 4 轮（事件 78 条、无 error）；
    对照组证明有效——同一个「那它的缺点呢？」，带上文答的是上一篇的主题，不带就答偏。
  - **前端**（无头 Edge + CDP 实测，非静态检查）：Markdown 的 h2/strong/em/ul/ol/blockquote/code/pre/table 全部渲染，
    列表符号为 `disc`，代码块里的 `[1]` 未被误挂引用，幽灵编号保持纯文本；
    XSS 样本（`onerror` / `<script>`）被 DOMPurify 拦下；13 个引用可点且点击后 `rank-flash` 命中 1 处；
    深色模式正文/面板对比度 13.99:1，图标与 `localStorage` 均正确切换；
    历史折叠块里的引用同样可点。**唯一的控制台输出是 Edge 自带跟踪防护对 jsDelivr 的 WARNING，与本项目无关。**
  - 84 个 pytest 用例全过。
