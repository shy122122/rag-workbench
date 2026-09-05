# RAG 可视化工作台

一个「真实可用 + 逐步可见」的个人 RAG 工作台。它真正跑通
上传文档 → 解析 → 切分 → 向量化 → 入库 → 检索 → 组装上下文 → 生成回答 的全流程，
同时在界面上逐步点亮每一步，展示**每一个步骤的中间结果**：
分块文本、向量维度与头部、Top-K 相似度分数条、实际发给大模型的完整 Prompt、
以及带来源引用的流式回答。

- 后端：Python + FastAPI（本地运行，数据全部留在本机）
- 模型服务：阿里云百炼（OpenAI 兼容接口），生成用 qwen 系列，向量用 text-embedding-v3
- 向量库：本地文件持久化（numpy + JSON），重启不丢
- 前端：单页 Tailwind，无需构建

## 目录结构

```
rag-workbench/
├─ app.py              # FastAPI 入口与全部路由（含 SSE）
├─ config.py           # config.json(参数) + .env(API Key)
├─ rag/
│  ├─ loader.py        # 解析 pdf/docx/txt/md
│  ├─ chunker.py       # 中文友好的递归切分
│  ├─ embeddings.py    # 百炼向量化
│  ├─ llm.py           # 百炼流式生成
│  ├─ store.py         # 本地向量库（增/查/删/清空/持久化）
│  ├─ events.py        # 步骤事件与 SSE 编码
│  └─ pipeline.py      # 建库 / 问答两条可视化流水线
├─ static/             # 前端工作台
├─ examples/           # 内置示例文档（一键载入）
└─ data/kb/            # 运行期生成的向量库
```

## 快速开始

1. 安装依赖（国内网络建议加清华镜像）：

   ```bash
   python -m venv .venv
   source .venv/Scripts/activate        # Windows Git Bash
   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

2. 启动：

   ```bash
   uvicorn app:app --reload --port 8000
   ```

3. 打开 http://127.0.0.1:8000 ，
   在右上角「设置」里填入你的 [阿里云百炼 API Key](https://bailian.console.aliyun.com/) 并测试连通。

4. 点左侧「示例」一键载入 3 篇内置文档，观察**建库流水**逐步执行；
   然后在上方输入框提问，观察**问答流水**每一步实时点亮。

## 使用提示

- 切分大小/重叠、Top-K、向量维度、模型都可在「设置」中调整；改动切分或嵌入参数后需「清空知识库」重建。
- 支持上传 .pdf / .docx / .txt / .md；旧版 .doc 与扫描版 PDF（无文字层）无法解析。
- API Key 只保存在项目本地的 `.env`，界面显示掩码，不会上传任何服务器。

## 常见问题

- **问答/建库报「未配置 API Key」**：先到「设置」里填写并点“测试连接”。
- **检索总是 0 命中**：确认知识库非空（看顶部统计芯片），并检查嵌入维度等参数一致后重建库。
- **答案质量差**：先看“向量检索”分数，再展开“组装上下文”核对注入的参考资料，最后才考虑换更强的生成模型。
