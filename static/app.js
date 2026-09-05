/* RAG 可视化工作台前端逻辑 */
"use strict";

// ---------------------------------------------------------------- 工具
const $ = (id) => document.getElementById(id);
const esc = (v) =>
  String(v ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");

function toast(msg, isErr = false) {
  const t = $("toast");
  t.textContent = msg;
  t.style.background = isErr ? "#b91c1c" : "#1e293b";
  t.classList.remove("hidden");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => t.classList.add("hidden"), isErr ? 4500 : 2200);
}

function fmtMs(ms) {
  if (ms == null) return "";
  return ms >= 1000 ? (ms / 1000).toFixed(2) + " s" : ms + " ms";
}

async function api(path, opts) {
  const res = await fetch(path, opts);
  let data = null;
  try { data = await res.json(); } catch { data = null; }
  if (!res.ok) throw new Error((data && data.message) || `请求失败(${res.status})`);
  return data;
}

// ---------------------------------------------------------------- 状态
const QUERY_ORDER = ["q_embed", "retrieve", "context", "generate"];
const MATCH_ORDER = ["m_embed", "m_retrieve"];
const MATCH_ORDER_LLM = [...MATCH_ORDER, "m_prompt", "m_interpret"];

const STAGE_LABEL = {
  q_embed: "问题向量化", retrieve: "向量检索 Top-K", context: "组装上下文", generate: "流式生成回答",
  m_embed: "文本向量化", m_retrieve: "知识库匹配 Top-K", m_prompt: "组装解读请求", m_interpret: "模型解读回答",
};

const flow = {
  running: false, mode: "query", order: QUERY_ORDER, ansStage: "generate",
  withLlm: false, question: "", text: "",
  steps: {}, references: [], hits: [], totalMs: 0,
};

// ---------------------------------------------------------------- 统计与知识库
async function refreshSidebar() {
  const kb = await api("/api/knowledge/list").catch(() => null);
  if (!kb) return;
  const { docs, stats } = kb;
  const chip = $("stat-chip");
  if (stats.chunks > 0) {
    chip.textContent = `知识库 · ${stats.docs} 篇 / ${stats.chunks} 块 / ${stats.dim} 维`;
    chip.classList.remove("hidden");
  } else {
    chip.classList.add("hidden");
  }
  const box = $("doc-list");
  if (!docs.length) {
    box.innerHTML =
      `<div class="text-xs text-slate-400 text-center py-6"><i class="fa fa-inbox fa-lg block mb-2"></i>知识库为空<br/>点击上方「示例」或上传文档开始建库</div>`;
    return;
  }
  box.innerHTML = docs.map((d) => `
    <div class="group flex items-start gap-2 px-2 py-2 rounded-lg hover:bg-slate-50">
      <div class="mt-0.5 w-7 h-7 rounded-md bg-indigo-50 text-indigo-500 flex items-center justify-center flex-none">
        <i class="fa fa-file-text-o text-sm"></i>
      </div>
      <div class="min-w-0 flex-1">
        <div class="text-xs font-medium truncate" title="${esc(d.doc_name)}">${esc(d.doc_name)}</div>
        <div class="text-[10px] text-slate-400 mono">${d.chunks} 块 · ${d.chars} 字符</div>
      </div>
      <button data-del="${esc(d.doc_id)}" class="opacity-0 group-hover:opacity-100 text-rose-400 hover:text-rose-600 text-xs p-1" title="删除本文档">
        <i class="fa fa-trash"></i>
      </button>
    </div>`).join("");
  box.querySelectorAll("[data-del]").forEach((btn) =>
    btn.addEventListener("click", async () => {
      await api("/api/knowledge/" + btn.dataset.del, { method: "DELETE" }).catch(() => {});
      refreshSidebar();
    })
  );
}

// ---------------------------------------------------------------- 步骤卡片
const STATUS_ICON = {
  pending: `<i class="fa fa-circle-o text-slate-300"></i>`,
  run: `<span class="w-3.5 h-3.5 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin inline-block"></span>`,
  done: `<i class="fa fa-check-circle text-emerald-500"></i>`,
  error: `<i class="fa fa-times-circle text-rose-500"></i>`,
};
const STATUS_RING = {
  run: "border-indigo-300 bg-indigo-50",
  error: "border-rose-300 bg-rose-50",
};

function stepCard(key, label, status, ms, bodyHtml, open = true) {
  const ring = STATUS_RING[status] || "border-slate-200 bg-white";
  const icon = STATUS_ICON[status] || STATUS_ICON.pending;
  return `
  <details class="step-card rounded-xl border ${ring} shadow-sm overflow-hidden" ${open ? "open" : ""}>
    <summary class="flex items-center gap-3 px-4 py-3 select-none">
      <span class="w-4 text-center flex-none">${icon}</span>
      <span class="text-sm font-medium flex-none">${label}</span>
      <span class="text-[10px] px-1.5 py-0.5 rounded bg-black/5 text-slate-400 mono flex-none">${key}</span>
      <span class="text-xs text-slate-400 mono ml-auto">${fmtMs(ms)}</span>
      <i class="fa fa-chevron-right chev text-xs text-slate-300"></i>
    </summary>
    <div class="px-4 pb-4 pt-1 border-t border-slate-100 bg-white/60">${bodyHtml}</div>
  </details>`;
}

// 向量头部展示
function vecHeadHtml(head) {
  if (!head || !head.length) return "";
  return `<div class="text-[11px] text-slate-400 mb-1">向量头部 6 维（embedding 归一化后）</div>
    <code class="mono text-[11px] text-indigo-600 bg-indigo-50 px-2.5 py-1.5 rounded block overflow-x-auto">[${head.map(esc).join(", ")} …]</code>`;
}

function metaChips(items) {
  return `<div class="flex flex-wrap gap-2 text-[11px] mb-2">
    ${items.filter(([k, v]) => v != null).map(([k, v]) =>
      `<span class="px-2 py-0.5 rounded-md bg-slate-100 text-slate-500"><span class="text-slate-400">${k}</span> <b class="mono">${esc(v)}</b></span>`).join("")}</div>`;
}

// 把回答文本中的 [n] 引用转成可点击定位的 span（仅限真实存在的编号）
function citeAnswerHtml(text, refs) {
  const set = new Set((refs || []).map((r) => Number(r.rank)));
  const s = esc(text);
  let out = "", last = 0, m;
  const re = /(\[\d+\])/g;
  while ((m = re.exec(s))) {
    out += s.slice(last, m.index);
    const n = parseInt(m[1].slice(1), 10);
    out += set.has(n)
      ? `<span class="cite" data-rank="${n}" title="定位到检索命中 #${n}">${m[1]}</span>`
      : m[1];
    last = m.index + m[0].length;
  }
  out += s.slice(last);
  return out;
}

// 流式回答 / 解读卡片（generate 与 m_interpret 共用）
function answerCardHtml(d, liveText, title, style) {
  const refs = flow.references || [];
  const final = (d.answer || "").trim();
  const body = final
    ? citeAnswerHtml(d.answer, refs)
    : (liveText ? esc(liveText) : '<span class="text-slate-300 italic">等待回答…</span>');
  return metaChips([["耗时", d.ms], ["字符数", d.chars], ["输入 tokens", (d.usage || {}).prompt_tokens], ["输出 tokens", (d.usage || {}).completion_tokens]]) +
    `<div class="rounded-lg border ${style} p-3">
       <div class="text-[11px] text-slate-400 font-medium mb-1.5"><i class="fa fa-commenting-o mr-1"></i>${title}</div>
       <div id="ans-text" class="pre-wrap text-sm text-slate-800 leading-7">${body}</div>
     </div>
     ${refs.length ? `
     <div class="mt-2">
       <div class="text-[11px] text-slate-400 mb-1">引用来源（点击回答里的 [n] 可定位到上方检索原文）</div>
       <div class="flex flex-wrap gap-1.5">${refs.map((r) =>
         `<span class="inline-flex items-center gap-1 text-[11px] px-2 py-1 rounded-md bg-slate-100 text-slate-500 border border-slate-200"><b>${r.rank}</b>${esc(r.doc_name)}</span>`).join("")}
       </div>
     </div>` : ""}`;
}

function detailHtml(stage, d, liveText) {
  d = d || {};
  if (d.error) return `<div class="text-sm text-rose-600 bg-rose-50 rounded-lg px-3 py-2">${esc(d.error)}</div>`;
  switch (stage) {
    case "q_embed":
    case "m_embed":
      return metaChips([["模型", d.model], ["维度", d.dimensions], ["耗时", fmtMs(d.ms)], ["tokens", d.tokens]]) + vecHeadHtml(d.vector_head);
    case "retrieve":
    case "m_retrieve": {
      const aboveZero = d.above != null;
      const threshNote = aboveZero
        ? `<div class="text-[11px] rounded-lg px-2.5 py-1.5 mb-2 flex items-start gap-1.5 ${(d.hits && d.hits.length) ? "text-slate-400 bg-slate-50" : "text-amber-600 bg-amber-50"}">
            <i class="fa fa-filter mt-0.5"></i><span>相似度阈值 <b>${d.min_score}</b>：高于阈值的命中共 <b>${d.above}</b> 条 · 展示前 ${d.top_k ?? "?"} 条${(d.top_k != null && d.above > d.top_k) ? "（其余被 Top-K 截断）" : ""}${d.above === 0 ? "，当前没有任何命中达到阈值" : ""}</span></div>`
        : "";
      const empty = !d.hits || !d.hits.length;
      if (empty) {
        return metaChips([["Top-K", d.top_k], ["阈值", d.min_score], ["库规模", `${d.total_chunks ?? 0} 块`], ["维度", d.dim ?? "-"], ["耗时", fmtMs(d.ms)]]) +
          threshNote +
          `<div class="text-xs text-slate-400 bg-slate-50 rounded-lg px-3 py-2"><i class="fa fa-frown-o mr-1"></i>${
            d.total_chunks
              ? "没有达到展示条件的命中。可尝试调低相似度阈值、换个表述，或向知识库补充内容。"
              : "知识库为空，先在上方「示例」载入或上传文档。"}</div>`;
      }
      return metaChips([["Top-K", d.top_k], ["阈值", d.min_score], ["库规模", `${d.total_chunks} 块`], ["维度", d.dim], ["耗时", fmtMs(d.ms)]]) +
        threshNote +
        d.hits.map((h) => {
          const w = Math.max(2, Math.min(100, Math.round(h.score * 100)));
          return `
          <div class="rounded-lg border border-slate-100 mb-2 overflow-hidden" data-rank="${h.rank}">
            <div class="flex items-center gap-3 px-3 py-2 bg-slate-50/70">
              <span class="w-5 h-5 rounded text-[10px] flex-none flex items-center justify-center ${h.rank === 1 ? "bg-amber-400 text-white" : "bg-slate-300 text-white"}">${h.rank}</span>
              <div class="flex-1 min-w-0">
                <div class="flex items-baseline justify-between">
                  <span class="text-xs font-medium truncate" title="${esc(h.doc_name)}">${esc(h.doc_name)} · 片段 ${h.chunk_index + 1}</span>
                  <span class="mono text-[11px] ${h.score >= 0.55 ? "text-emerald-600" : "text-amber-600"}">${(h.score * 100).toFixed(1)}%</span>
                </div>
                <div class="h-1.5 rounded-full bg-slate-200 mt-1 overflow-hidden"><div class="h-full rounded-full ${h.score >= 0.55 ? "bg-emerald-400" : "bg-amber-400"}" style="width:${w}%"></div></div>
              </div>
            </div>
            <details class="bg-white"><summary class="px-3 py-1.5 text-[11px] text-slate-400 hover:text-indigo-500 flex items-center gap-1"><i class="fa fa-chevron-right chev text-[9px]"></i>查看检索到的原文</summary>
            <div class="px-3 pb-3"><pre class="wrap text-xs text-slate-600 leading-relaxed">${esc(h.text)}</pre></div></details>
          </div>`;
        }).join("");
    }
    case "context":
    case "m_prompt": {
      const msgs = d.messages || [];
      const user = msgs.find((m) => m.role === "user") || {};
      const system = msgs.find((m) => m.role === "system") || {};
      const copy = esc(user.content || "");
      return metaChips([["命中资料", `${d.hits_used} 条`], ["估算 tokens", d.est_tokens], ["耗时", fmtMs(d.ms)]]) +
        `<div class="space-y-2 text-xs">
          <div>
            <div class="flex items-center justify-between mb-1">
              <span class="font-medium text-slate-500"><i class="fa fa-terminal text-slate-400 mr-1"></i>system 系统提示词</span>
            </div>
            <pre class="wrap bg-slate-900 text-slate-100 rounded-lg p-3 leading-relaxed">${esc(system.content || "")}</pre>
          </div>
          <div>
            <div class="flex items-center justify-between mb-1">
              <span class="font-medium text-slate-500"><i class="fa fa-send-o text-slate-400 mr-1"></i>user 注入给模型的内容（参考资料 + 问题）</span>
              <button class="copy-btn text-[11px] px-2 py-0.5 rounded bg-slate-100 hover:bg-slate-200 text-slate-500" data-copy="${copy}">复制</button>
            </div>
            <details open class="rounded-lg border border-slate-200 bg-white"><summary class="px-3 py-1.5 text-[11px] text-slate-400 flex items-center gap-1"><i class="fa fa-chevron-right chev text-[9px]"></i>展开完整 Prompt</summary>
            <pre class="wrap p-3 text-slate-700 leading-relaxed">${esc(user.content || "")}</pre></details>
          </div>
        </div>`;
    }
    case "generate":
      return answerCardHtml(d, liveText, "最终回答（模型生成）", "bg-gradient-to-br from-slate-50 to-indigo-50/40 border-indigo-100");
    case "m_interpret":
      return answerCardHtml(d, liveText, "模型解读（检索结果与待匹配文本的关系）", "bg-gradient-to-br from-amber-50/60 to-orange-50/40 border-amber-100");
    // 建库：数据清洗
    case "clean": {
      if (!d.enabled) {
        return `<div class="text-xs text-slate-400 bg-slate-50 rounded-lg px-3 py-2"><i class="fa fa-info-circle mr-1"></i>${esc(d.message || "未启用数据清洗")}</div>`;
      }
      if (d.error) return `<div class="text-sm text-rose-600 bg-rose-50 rounded-lg px-3 py-2">${esc(d.error)}</div>`;
      const removed = d.removed_lines || 0;
      const chips = metaChips([
        ["删除", `${removed} 行 / ${d.removed_chars} 字`],
        ["清洗前", `${d.before_chars} 字`], ["清洗后", `${d.after_chars} 字`],
        ["噪声行", d.noise ? "开" : "关"], ["去重复", d.dedup ? "开" : "关"],
      ]);
      const rules = (d.rules || []).map((r) =>
        `<li class="flex items-start gap-1.5"><i class="fa fa-check-circle-o text-emerald-400 mt-0.5"></i><span>${esc(r)}</span></li>`).join("");
      const samples = (d.samples || []).map((s) =>
        `<div class="rounded border border-slate-100 px-2.5 py-1.5 mb-1.5 flex items-start gap-2">
           <span class="flex-none text-[10px] px-1.5 py-0.5 rounded ${s.reason.includes("重复") ? "bg-violet-100 text-violet-600" : "bg-rose-100 text-rose-500"}">${esc(s.reason)}</span>
           <div class="min-w-0 flex-1"><pre class="wrap text-[11px] text-slate-400 leading-relaxed max-h-16 overflow-hidden">${esc(s.text)}</pre></div>
         </div>`).join("");
      return chips +
        (rules ? `<ul class="space-y-0.5 mb-2 text-xs text-slate-500">${rules}</ul>` : "") +
        (samples ? `<div class="text-[11px] text-slate-400 mb-1">被删除内容${d.removed_lines > 8 ? "（示例前 8 条）" : ""}</div>${samples}` : "") +
        (removed === 0 ? `<div class="text-[11px] text-emerald-600 bg-emerald-50 rounded-lg px-2.5 py-1.5"><i class="fa fa-check-circle mr-1"></i>未发现需要清理的内容</div>` : "");
    }
    // 建库四步
    case "parse":
      return metaChips([["来源", d.source], ["字符数", d.chars], ["页数", d.pages]]) +
        `<div class="text-[11px] text-slate-400 mb-1">解析出的文本预览</div><pre class="wrap text-xs text-slate-600 bg-slate-50 rounded-lg p-2.5">${esc(d.preview)}${esc((d.preview || "").length >= 300 ? "\n……" : "")}</pre>`;
    case "chunk":
      return metaChips([["分块数", d.count], ["chunk_size", d.chunk_size], ["overlap", d.overlap]]) +
        (d.samples || []).map((s) =>
          `<div class="rounded border border-slate-100 px-2.5 py-1.5 mb-1.5">
            <div class="text-[11px] mono text-slate-400 mb-0.5">#${s.chunk_index} · ${s.len} 字符</div>
            <div class="text-xs text-slate-600 leading-relaxed line-clamp-2">${esc(s.head)}${(s.len > 140) ? "…" : ""}</div>
          </div>`).join("") +
        (d.count > 12 ? `<div class="text-[11px] text-slate-400">… 其余 ${d.count - 12} 块略</div>` : "");
    case "embed":
      return metaChips([["模型", d.model], ["维度", d.dimensions], ["批次数", d.batches], ["tokens", d.tokens], ["耗时", fmtMs(d.ms)]]) + vecHeadHtml(d.vector_head);
    case "index":
      return metaChips([["本次新增", `${d.new_chunks} 块`], ["库中文档", d.docs + " 篇"], ["库总块数", d.chunks + " 块"], ["维度", d.dim]]) +
        `<div class="text-[11px] text-emerald-600 bg-emerald-50 rounded-lg px-3 py-2"><i class="fa fa-check-circle mr-1"></i>已写入本地向量库并持久化，重启不丢失。</div>`;
    default:
      return `<pre class="wrap text-[11px] text-slate-500">${esc(JSON.stringify(d, null, 2))}</pre>`;
  }
}

// ---------------------------------------------------------------- 流程绘制（问答 / 内容匹配）
function clip(s, n) {
  s = String(s || "");
  return s.length > n ? s.slice(0, n) + "…" : s;
}

function emptyStateHtml() {
  if (flow.mode === "match") {
    return `<div class="text-slate-400 text-sm text-center py-10">
      <i class="fa fa-link fa-2x block mb-3 text-slate-200"></i>
      输入一段文本或句子，这里将展示它与知识库的相似度匹配全过程。<br/>
      <span class="text-xs">文本向量化 → Top-K 匹配（含相似度分数）${flow.withLlm ? " → 组装解读请求 → 模型解读" : ""}</span></div>`;
  }
  return `<div class="text-slate-400 text-sm text-center py-10">
    <i class="fa fa-comments-o fa-2x block mb-3 text-slate-200"></i>
    提交一个问题，这里将按步骤实时点亮 RAG 问答全过程。<br/>
    <span class="text-xs">问题向量化 → Top-K 检索（含相似度分数）→ 组装上下文（完整 Prompt）→ 流式生成</span></div>`;
}

function setFlowTitle() {
  const t = $("flow-title");
  if (t) t.textContent = flow.mode === "match" ? "内容匹配流水" : "问答生成流水";
}

function flowHasInput() { return !!(flow.question || flow.text); }

function refreshExportBtn() {
  const has = flow.order.some((k) => flow.steps[k] && flow.steps[k].status === "done");
  $("btn-export").classList.toggle("hidden", !has);
}

function drawQueryFlow() {
  const body = $("query-body");
  const live = flow._liveText || "";
  setFlowTitle();
  if (!flowHasInput()) {
    body.innerHTML = emptyStateHtml();
    $("query-meta").textContent = "";
    refreshExportBtn();
    return;
  }
  const cards = flow.order.map((key) => {
    const st = flow.steps[key] || { status: "pending", ms: null };
    const streaming = key === flow.ansStage && st.status === "run" && live;
    return stepCard(key, STAGE_LABEL[key], st.status, st.ms, detailHtml(key, st.data, streaming ? live : ""), st.data != null);
  }).join("");
  body.innerHTML =
    `<div class="pipeline-rail relative">
       <div class="space-y-3">${cards}</div>
     </div>
     ${flow.running ? `<div class="mt-3 text-center text-[11px] text-indigo-400"><span class="inline-block w-2.5 h-2.5 border-2 border-indigo-400 border-t-transparent rounded-full animate-spin align-middle"></span> ${flow.mode === "match" ? "正在匹配（可选解读）…" : "正在执行检索生成流水…"}</div>` : ""}`;
  body.querySelectorAll(".copy-btn").forEach((b) =>
    b.addEventListener("click", async () => {
      await navigator.clipboard.writeText(b.dataset.copy).catch(() => {});
      toast("已复制完整 Prompt");
    })
  );
  refreshExportBtn();
}

function resetFlow(mode) {
  flow.mode = mode || "query";
  flow.order = flow.mode === "match" ? (flow.withLlm ? MATCH_ORDER_LLM : MATCH_ORDER) : QUERY_ORDER;
  flow.ansStage = flow.mode === "match" ? (flow.withLlm ? "m_interpret" : null) : "generate";
  flow.running = true;
  flow.steps = {};
  flow.references = [];
  flow.totalMs = 0;
  flow._liveText = "";
}

function applyStep(ev) {
  const { stage, status, data } = ev;
  flow.steps[stage] = { status, data, ms: data && data.ms };
  if (status === "run" && stage === flow.ansStage) flow._liveText = "";
  if (status === "done" && stage === flow.ansStage) flow.running = false;
}

// ---------------------------------------------------------------- SSE 问答 / 匹配
function currentTopk() {
  const v = parseInt($("ctl-topk").value, 10);
  return Number.isFinite(v) ? Math.min(20, Math.max(1, v)) : 5;
}
function currentMscore() {
  const v = parseFloat($("ctl-mscore").value);
  return Number.isFinite(v) ? Math.min(1, Math.max(0, v)) : 0;
}
function runParams() {
  return { top_k: currentTopk(), min_score: currentMscore() };
}
let persistTimer = null;
function persistRunCtl() {
  clearTimeout(persistTimer);
  persistTimer = setTimeout(async () => {
    try {
      await api("/api/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(runParams()),
      });
    } catch (e) { toast("保存 Top-K / 阈值失败：" + e.message, true); }
  }, 350);
}

async function submitQuestion(question) {
  question = (question || "").trim();
  if (!question) return;
  if (flow.running) { toast("上一轮问答尚未结束"); return; }
  flow.mode = "query"; flow.withLlm = false; flow.question = question; flow.text = "";
  resetFlow("query");
  drawQueryFlow();
  $("query-meta").textContent = "问题：" + clip(question, 40);
  await streamFlow("/api/query", { question, ...runParams() });
}

async function submitMatch(text) {
  text = (text || "").trim();
  if (!text) return;
  if (flow.running) { toast("上一轮内容匹配尚未结束"); return; }
  flow.mode = "match"; flow.withLlm = $("with-llm").checked; flow.question = ""; flow.text = text;
  resetFlow("match");
  drawQueryFlow();
  $("query-meta").textContent = "待匹配文本：" + clip(text, 40);
  await streamFlow("/api/match", { text, with_llm: flow.withLlm, ...runParams() });
}

async function streamFlow(path, body) {
  try {
    const res = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok || !res.body) {
      let msg = "流程接口不可用";
      try { msg = (await res.json()).message || msg; } catch { /* 非 JSON */ }
      throw new Error(msg);
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buf = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n\n")) >= 0) {
        const block = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        handleSseBlock(block);
      }
    }
  } catch (e) {
    flow.running = false;
    const key = flow.ansStage || flow.order[flow.order.length - 1];
    flow.steps[key] = { status: "error", data: { error: String(e.message || e) } };
    drawQueryFlow();
    toast(String(e.message || e), true);
  } finally {
    if (flow.running) {
      // 正常以 done 收尾时已在 done 处理器重置 running；这里兜底
      flow.running = false;
      const last = flow.order[flow.order.length - 1];
      if (flow.steps[last] && flow.steps[last].status === "done") drawQueryFlow();
    }
  }
}

function handleSseBlock(block) {
  let ev = "";
  const dataLines = [];
  block.split("\n").forEach((line) => {
    const t = line.trim();
    if (t.startsWith("event:")) ev = t.slice(6).trim();
    else if (t.startsWith("data:")) dataLines.push(t.slice(5).trim());
  });
  if (!ev || !dataLines.length) return;
  let payload = null;
  try { payload = JSON.parse(dataLines.join("\n")); } catch { return; }

  if (ev === "step") {
    applyStep(payload);
    drawQueryFlow();
  } else if (ev === "token") {
    flow._liveText += payload.text || "";
    const ans = $("ans-text");
    if (ans) ans.textContent = flow._liveText;
  } else if (ev === "error") {
    flow.running = false;
    const key = flow.ansStage || flow.order[flow.order.length - 1];
    flow.steps[key] = { status: "error", data: { error: payload.message || "未知错误" } };
    drawQueryFlow();
    toast(payload.message || "流程出错", true);
  } else if (ev === "done") {
    flow.running = false;
    flow.totalMs = payload.elapsed_ms || 0;
    flow.references = payload.references || [];
    // 把流式文本固化回答步，保证重绘不丢
    if (flow.ansStage) {
      const st = flow.steps[flow.ansStage] || (flow.steps[flow.ansStage] = {});
      st.data = st.data || {};
      st.data.answer = payload.answer || st.data.answer;
      st.data.chars = (st.data.answer || "").length;
    }
    if (flow.mode === "match") {
      $("query-meta").textContent =
        `待匹配文本："${clip(payload.text || flow.text, 30)}" · 命中 ${(payload.references || []).length} 条` +
        (payload.answer ? " · 已解读" : "") + ` · 总耗时 ${fmtMs(payload.elapsed_ms)}`;
    } else {
      $("query-meta").textContent = `问题："${payload.question || flow.question}" · 总耗时 ${fmtMs(payload.elapsed_ms)}`;
    }
    drawQueryFlow();
  }
}

// ---------------------------------------------------------------- 建库流水
function addIngestLog(entry) {
  const box = $("ingest-log");
  if (!entry) return;
  const ok = entry.ok;
  const icon = ok
    ? `<i class="fa fa-check-circle text-emerald-500"></i>`
    : `<i class="fa fa-times-circle text-rose-500"></i>`;
  const title = (entry.doc_name || "文档").slice(0, 40);
  const time = new Date().toLocaleTimeString("zh-CN", { hour12: false });
  const steps = entry.transcript || [];
  const body = steps.length
    ? `<div class="space-y-2 mt-2 pt-2 border-t border-slate-100">${steps.map((s) => {
        const ic = s.ok ? STATUS_ICON.done : STATUS_ICON.error;
        return `<details class="rounded-lg border border-slate-100">
          <summary class="flex items-center gap-2 px-2.5 py-1.5 text-xs select-none">
            <span>${ic}</span><span class="text-slate-600">${esc(s.label)}</span>
            <span class="mono text-[10px] text-slate-400 ml-auto">${fmtMs(s.ms)}</span>
            <i class="fa fa-chevron-right chev text-[9px] text-slate-300"></i>
          </summary>
          <div class="px-2.5 pb-2.5">${detailHtml(s.stage, s.detail)}</div>
        </details>`;
      }).join("")}</div>`
    : (entry.error ? `<div class="text-rose-500 text-xs mt-1.5">${esc(entry.error)}</div>` : "");
  const el = document.createElement("div");
  el.className = "rounded-lg border border-slate-200 p-2.5 mb-2 text-left";
  el.innerHTML = `<div class="flex items-center gap-2 text-xs">${icon}<span class="font-medium truncate">${esc(title)}</span><span class="mono text-[10px] text-slate-300 ml-auto">${time}</span></div>${body}`;
  box.prepend(el);
  box.querySelector(".text-center")?.remove();
}

// ---------------------------------------------------------------- 上传 / 示例
function doUploadFiles(files) {
  files = Array.from(files || []).filter((f) => /\.(pdf|docx|txt|md|markdown)$/i.test(f.name));
  if (!files.length) { toast("仅支持 .pdf / .docx / .txt / .md", true); return; }
  (async () => {
    for (const f of files) {
      const hint = $("upload-hint");
      hint.classList.remove("hidden");
      hint.innerHTML = `<i class="fa fa-spinner fa-spin mr-1 text-indigo-500"></i>正在建库：${esc(f.name)}…`;
      try {
        const fd = new FormData();
        fd.append("file", f);
        const data = await api("/api/knowledge/upload", { method: "POST", body: fd });
        addIngestLog(data);
        if (data.error) hint.innerHTML = `<span class="text-rose-500">${esc(f.name)} 建库失败：${esc(data.error)}</span>`;
        else hint.innerHTML = `<span class="text-emerald-600">${esc(f.name)} 建库完成，新增 ${esc((data.transcript || []).find(s => s.stage === "index")?.detail?.new_chunks ?? "?")} 块</span>`;
      } catch (e) {
        hint.innerHTML = `<span class="text-rose-500">${esc(f.name)}：${esc(e.message)}</span>`;
      }
      await new Promise((r) => setTimeout(r, 400));
    }
    setTimeout(() => hint.classList.add("hidden"), 1600);
    refreshSidebar();
  })();
}

async function loadExamples() {
  const box = $("ingest-log");
  box.innerHTML = `<div class="text-center py-2"><i class="fa fa-spinner fa-spin text-indigo-400 text-lg"></i><div class="text-[11px] text-slate-400 mt-1">正在逐篇解析并向量化内置示例…</div></div>`;
  try {
    const data = await api("/api/knowledge/load-examples", { method: "POST" });
    (data.docs || []).forEach((d) => addIngestLog(d));
    const okN = (data.docs || []).filter((d) => d.ok).length;
    if (!data.ok) toast(`部分示例建库失败（成功 ${okN}/${(data.docs || []).length}）`, true);
    else toast(`已载入 ${okN} 篇示例文档`);
  } catch (e) {
    toast("载入示例失败：" + e.message, true);
  }
  refreshSidebar();
}

// ---------------------------------------------------------------- 模式切换 / 引用定位 / 导出
function setMode(mode) {
  if (flow.running) { toast("流程进行中，请结束后再切换模式"); return; }
  flow.mode = mode;
  document.querySelectorAll(".mode-tab").forEach((b) => b.classList.toggle("active", b.dataset.mode === mode));
  const isMatch = mode === "match";
  $("chip-row").classList.toggle("hidden", isMatch);
  $("llm-wrap").classList.toggle("hidden", !isMatch);
  $("llm-wrap").classList.toggle("flex", isMatch);
  $("q-input").placeholder = isMatch
    ? "粘贴要与知识库匹配的文本 / 句子，例如：磷酸铁锂电池和三元锂电池在寿命上有哪些差异…"
    : "输入问题，例如：RAG 中的 Top-K 是什么？分块策略要注意什么？";
  $("btn-send-label").textContent = isMatch ? "匹配" : "提问";
  $("mode-hint").textContent = isMatch ? "将整段文本与知识库分块逐条比对相似度" : "";
  flow.question = ""; flow.text = ""; flow.withLlm = false;
  flow.steps = {}; flow.references = []; flow.totalMs = 0;
  drawQueryFlow();
}

function focusRank(rank) {
  const el = document.querySelector(`[data-rank="${rank}"]`);
  if (!el) { toast(`未找到检索命中 #${rank}`); return; }
  el.scrollIntoView({ behavior: "smooth", block: "center" });
  el.classList.remove("rank-flash");
  void el.offsetWidth; // 重新触发 CSS 动画
  el.classList.add("rank-flash");
}

function buildMarkdown() {
  const L = [];
  L.push(`# RAG 可视化工作台 · ${flow.mode === "match" ? "内容匹配" : "问答生成"}导出`);
  L.push("");
  L.push(`- 时间：${new Date().toLocaleString("zh-CN", { hour12: false })}`);
  L.push(`- 输入：${flow.mode === "match" ? flow.text : flow.question}`);
  L.push(`- Top-K：${currentTopk()} · 相似度阈值：${currentMscore()}`);
  if (flow.totalMs) L.push(`- 总耗时：${fmtMs(flow.totalMs)}`);
  flow.order.forEach((key) => {
    const st = flow.steps[key];
    if (!st || st.status !== "done" || !st.data) return;
    const d = st.data;
    L.push("", `## ${STAGE_LABEL[key]}`, "");
    L.push(`- 耗时：${st.ms != null ? fmtMs(st.ms) : "-"}`);
    switch (key) {
      case "q_embed":
      case "m_embed":
        L.push(`- 模型：${d.model}（${d.dimensions} 维） · tokens：${d.tokens}`);
        break;
      case "retrieve":
      case "m_retrieve": {
        if (d.min_score != null && d.above != null) {
          L.push(`- 阈值 ${d.min_score}：高于阈值共 ${d.above} 条 · 展示前 ${d.top_k} 条`);
        }
        if (!(d.hits || []).length) { L.push("（无命中）"); break; }
        (d.hits || []).forEach((h) => {
          L.push(`${h.rank}. 《${h.doc_name}》片段 ${h.chunk_index + 1} — 相似度 ${(h.score * 100).toFixed(1)}%`);
          L.push(`   > ${h.text.replace(/\n/g, "\n   > ")}`);
        });
        break;
      }
      case "context":
      case "m_prompt": {
        (d.messages || []).forEach((m) => {
          L.push(`### ${m.role === "system" ? "system 系统提示词" : "user 用户内容"}`, "", m.content || "");
        });
        break;
      }
      case "generate":
      case "m_interpret":
        L.push((d.answer || "").trim() || "（无内容）");
        break;
      case "clean":
        L.push(`- 清洗前 ${d.before_chars} 字 → 清洗后 ${d.after_chars} 字（删除 ${d.removed_lines} 行 / ${d.removed_chars} 字）`);
        (d.rules || []).forEach((r) => L.push(`- ${r}`));
        (d.samples || []).forEach((s) => L.push(`  - [${s.reason}] ${s.text}`));
        break;
      default:
        break;
    }
  });
  L.push("");
  return L.join("\n");
}

function exportMarkdown() {
  const md = buildMarkdown();
  const fname = `rag-${flow.mode === "match" ? "match" : "qa"}-${new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19)}.md`;
  const blob = new Blob([md], { type: "text/markdown;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = fname;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
  if (navigator.clipboard) {
    navigator.clipboard.writeText(md)
      .then(() => toast("已导出并复制 Markdown"))
      .catch(() => toast("已导出 Markdown（已触发下载）"));
  } else {
    toast("已导出 Markdown（已触发下载）");
  }
}

async function loadRunControls() {
  try {
    const c = await api("/api/config");
    $("ctl-topk").value = c.config.top_k;
    $("ctl-mscore").value = c.config.retrieval_min_score;
  } catch { /* 保持默认 */ }
}

// ---------------------------------------------------------------- 设置
const DIMS = {
  "text-embedding-v3": [1024, 768, 512, 256, 128, 64],
  "text-embedding-v4": [2048, 1536, 1024, 768, 512, 256, 128, 64],
};
function fillDims(model, keep) {
  const list = DIMS[model] || [1024, 768, 512];
  const sel = $("cfg-dim");
  const prev = Number(keep != null ? keep : sel.value) || list[0];
  sel.innerHTML = list.map((d) => `<option>${d}</option>`).join("");
  sel.value = String(list.includes(prev) ? prev : list[0]);
}

async function openSettings() {
  try {
    const c = await api("/api/config");
    const cfg = c.config;
    $("cfg-llm").value = cfg.llm_model;
    $("cfg-emb").value = cfg.embedding_model;
    fillDims(cfg.embedding_model, cfg.dimensions);
    $("cfg-clean-en").checked = !!cfg.clean_enabled;
    $("cfg-clean-noise").checked = !!cfg.clean_noise;
    $("cfg-clean-dedup").checked = !!cfg.clean_dedup;
    $("cfg-cs").value = cfg.chunk_size;
    $("cfg-co").value = cfg.chunk_overlap;
    $("cfg-key").value = "";
    $("cfg-key").placeholder = c.has_key ? "已保存（留空则保持不变）" : "sk-… 填写你的百炼 API Key";
    const ks = $("key-status");
    ks.textContent = c.has_key ? "已配置" : "未配置";
    ks.className = "text-xs px-2 py-0.5 rounded-full " + (c.has_key ? "bg-emerald-100 text-emerald-600" : "bg-rose-100 text-rose-500");
    $("settings-modal").classList.remove("hidden");
  } catch (e) { toast("读取设置失败：" + e.message, true); }
}

async function saveSettings() {
  const n = (v, d) => { const x = parseInt(v, 10); return isNaN(x) ? d : x; };
  const payload = {
    llm_model: $("cfg-llm").value,
    embedding_model: $("cfg-emb").value,
    dimensions: n($("cfg-dim").value, 1024),
    chunk_size: Math.max(50, n($("cfg-cs").value, 500)),
    chunk_overlap: Math.max(0, n($("cfg-co").value, 50)),
    clean_enabled: $("cfg-clean-en").checked,
    clean_noise: $("cfg-clean-noise").checked,
    clean_dedup: $("cfg-clean-dedup").checked,
  };
  const key = $("cfg-key").value.trim();
  if (key) payload.api_key = key;
  fillDims($("cfg-emb").value, $("cfg-dim").value); // 维度选项随嵌入模型联动
  payload.dimensions = n($("cfg-dim").value, 1024);
  try {
    await api("/api/config", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    toast("设置已保存");
    $("settings-modal").classList.add("hidden");
  } catch (e) { toast("保存失败：" + e.message, true); }
}

// ---------------------------------------------------------------- 初始化
function init() {
  refreshSidebar();

  $("btn-settings").addEventListener("click", openSettings);
  $("btn-close-settings").addEventListener("click", () => $("settings-modal").classList.add("hidden"));
  $("settings-modal").addEventListener("click", (e) => { if (e.target.id === "settings-modal") $("settings-modal").classList.add("hidden"); });
  $("btn-save-settings").addEventListener("click", saveSettings);
  $("cfg-emb").addEventListener("change", () => fillDims($("cfg-emb").value));
  $("btn-test").addEventListener("click", async () => {
    $("btn-test").disabled = true; $("btn-test").textContent = "测试中…";
    try { const r = await api("/api/test-key", { method: "POST" }); toast(r.message, !r.ok); }
    catch (e) { toast("测试失败：" + e.message, true); }
    $("btn-test").disabled = false; $("btn-test").textContent = "测试连接";
  });

  $("btn-examples").addEventListener("click", loadExamples);
  $("btn-clear").addEventListener("click", async () => {
    if (!confirm("确定清空整个知识库？此操作不可恢复。")) return;
    await api("/api/knowledge", { method: "DELETE" }).catch(() => {});
    $("ingest-log").innerHTML = `<div class="text-center text-slate-400 py-2">已清空知识库</div>`;
    refreshSidebar();
  });

  const dz = $("dropzone");
  const fi = $("file-input");
  dz.addEventListener("click", () => fi.click());
  fi.addEventListener("change", () => { doUploadFiles(fi.files); fi.value = ""; });
  ["dragover", "dragenter"].forEach((e) => dz.addEventListener(e, (ev) => { ev.preventDefault(); dz.classList.add("border-indigo-400", "text-indigo-400"); }));
  ["dragleave", "drop"].forEach((e) => dz.addEventListener(e, (ev) => { ev.preventDefault(); dz.classList.remove("border-indigo-400", "text-indigo-400"); }));
  dz.addEventListener("drop", (ev) => doUploadFiles(ev.dataTransfer.files));

  // 模式切换
  document.querySelectorAll(".mode-tab").forEach((b) =>
    b.addEventListener("click", () => setMode(b.dataset.mode))
  );

  // 问答 / 匹配 提交（textarea 不触发隐式提交，用 Enter 处理）
  const composerSubmit = () => {
    const v = $("q-input").value;
    if (flow.mode === "match") submitMatch(v);
    else submitQuestion(v);
  };
  $("query-form").addEventListener("submit", (e) => { e.preventDefault(); composerSubmit(); });
  $("q-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); composerSubmit(); }
  });
  document.querySelectorAll(".q-chip").forEach((c) =>
    c.addEventListener("click", () => { $("q-input").value = c.textContent.trim(); composerSubmit(); })
  );

  // Top-K / 阈值：改动即保存（防抖）
  ["ctl-topk", "ctl-mscore"].forEach((id) => {
    const el = $(id);
    el.addEventListener("input", persistRunCtl);
    el.addEventListener("change", () => { persistRunCtl(); el.value = id === "ctl-topk" ? currentTopk() : currentMscore(); });
  });

  // 点击回答中的 [n] 定位到检索命中
  $("query-body").addEventListener("click", (e) => {
    const c = e.target.closest(".cite");
    if (c) focusRank(c.dataset.rank);
  });

  // 导出 MD
  $("btn-export").addEventListener("click", exportMarkdown);

  setMode("query");
  loadRunControls();
}

document.addEventListener("DOMContentLoaded", init);
