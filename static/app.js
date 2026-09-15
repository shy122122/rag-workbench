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

// 线性描边 SVG 图标库（替代 Font Awesome；stroke 走 currentColor）
function svg(p, w = 14, h = 14, sw = 1.6) {
  return `<svg viewBox="0 0 24 24" width="${w}" height="${h}" fill="none" stroke="currentColor" stroke-width="${sw}" stroke-linecap="round" stroke-linejoin="round">${p}</svg>`;
}
const IC = {
  db:     svg('<ellipse cx="12" cy="5.5" rx="8" ry="2.8"/><path d="M4 5.5V18c0 1.6 3.6 2.8 8 2.8s8-1.2 8-2.8V5.5"/><path d="M4 11.8C4 13.4 7.6 14.6 12 14.6s8-1.2 8-2.8"/>'),
  file:   svg('<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5"/>'),
  inbox:  svg('<path d="M3 13h4l2 3h6l2-3h4"/><path d="M3 13V6a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2v7"/><path d="M3 13v5a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-5"/>'),
  trash:  svg('<path d="M3 6h18"/><path d="M8 6V4a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v2"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6M14 11v6"/>'),
  chevR:  svg('<path d="M9 6l6 6-6 6"/>', 13, 13, 1.8),
  filter: svg('<path d="M3 5h18l-7 8.5v5.5l-4 2v-7.5z"/>'),
  frown:  svg('<circle cx="12" cy="12" r="8.6"/><path d="M8.8 15.6a4.6 4.6 0 0 1 6.4 0"/><path d="M9 9.6h.01M15 9.6h.01"/>'),
  check:  svg('<path d="M5 12l4.6 4.6L19 7"/>', 12, 12, 2),
  x:      svg('<path d="M6 6l12 12M18 6L6 18"/>', 12, 12, 2),
  term:   svg('<rect x="3" y="4.5" width="18" height="15" rx="2"/><path d="M7 9l3 3-3 3M13 15h4"/>', 14, 14, 1.5),
  bub:    svg('<path d="M21 11.5a8 8 0 0 1-8 8H5l2.5-2.5A8 8 0 1 1 21 11.5z"/>', 15, 15, 1.4),
  link:   svg('<path d="M10 14a4.5 4.5 0 0 0 6.6.5l2.6-2.6a4.5 4.5 0 0 0-6.4-6.4L11.6 6.9"/><path d="M14 10a4.5 4.5 0 0 0-6.6-.5L4.8 12.1a4.5 4.5 0 0 0 6.4 6.4L12.4 17"/>', 15, 15, 1.4),
  seg:    svg('<path d="M6 4v16M6 6l14 4-8 3-6 5"/>'),  // 片段/段落（分块）
};

function toast(msg, isErr = false) {
  const t = $("toast");
  t.textContent = msg;
  t.style.background = isErr ? "var(--bad)" : "var(--codebg)";
  t.style.color = isErr ? "#fff" : "var(--codeink)";
  t.classList.remove("hidden");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => t.classList.add("hidden"), isErr ? 4500 : 2200);
}

function fmtMs(ms) {
  if (ms == null) return "";
  // 后端给的是整数毫秒，0 代表「不到 1 毫秒」而不是没计时。
  // 直接显示 "0 ms" 会被当成仪表没接好，本地那些纯计算的步骤（粗排、拼上下文）全中招。
  if (ms === 0) return "<1 ms";
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
const QUERY_ORDER = ["q_embed", "retrieve", "rerank", "context", "generate"];
const MATCH_ORDER = ["m_embed", "m_retrieve", "m_rerank"];
const MATCH_ORDER_LLM = [...MATCH_ORDER, "m_prompt", "m_interpret"];

const STAGE_LABEL = {
  q_embed: "问题向量化", retrieve: "向量粗排召回", rerank: "精排重排序",
  context: "组装上下文", generate: "流式生成回答",
  m_embed: "文本向量化", m_retrieve: "向量粗排召回", m_rerank: "精排重排序",
  m_prompt: "组装解读请求", m_interpret: "模型解读回答",
};

const flow = {
  running: false, mode: "query", order: QUERY_ORDER, ansStage: "generate",
  withLlm: false, question: "", text: "",
  steps: {}, references: [], hits: [], totalMs: 0,
  abort: null, // 当前流式请求的 AbortController，供「停止」按钮中断
  stopped: false, // 本轮是否被用户手动停止
  history: [], // 本会话已完成的问答轮次，追问时取最后几轮当上文
};

// 内容匹配的文档范围。scope 为 null 表示「全选」——比维护一个全量集合省事，
// 也能让新上传的文档自动进入范围（用户没主动排除过，就不该被排除在外）。
let allDocs = [];
let scope = null;

// ---------------------------------------------------------------- 统计与知识库
function refreshSidebar() {
  const box = $("doc-list");
  api("/api/knowledge/list").then((kb) => {
    const { docs, stats } = kb;
    allDocs = docs;
    // 删掉的文档要从已选范围里摘掉，否则范围里会残留已不存在的 id
    if (scope) {
      const alive = new Set(docs.map((d) => d.doc_id));
      scope = new Set([...scope].filter((id) => alive.has(id)));
    }
    renderScope();
    const chip = $("stat-chip");
    if (stats.chunks > 0) {
      chip.textContent = `知识库 · ${stats.docs} 篇 / ${stats.chunks} 块 / ${stats.dim} 维`;
      chip.classList.remove("hidden");
    } else {
      chip.classList.add("hidden");
    }
    if (!docs.length) {
      renderKbAlert(stats);
      box.innerHTML =
        `<div class="text-center text-[var(--ink3)] py-7">
           <span class="inline-flex text-[var(--ink4)] mb-2">${IC.inbox}</span>
           <div class="text-[12px]">知识库为空</div>
           <div class="text-[11px] text-[var(--ink4)] mt-0.5">点击上方「示例」或上传文档开始建库</div>
         </div>`;
      return;
    }
    renderKbAlert(stats);
    box.innerHTML = docs.map((d) => `
      <div class="group flex items-start gap-2.5 px-2 py-2 rounded-lg hover:bg-[var(--pane2)]">
        <span class="mt-0.5 w-7 h-7 rounded-md border border-[var(--line2)] text-[var(--ink3)] flex items-center justify-center flex-none">${IC.file}</span>
        <div class="min-w-0 flex-1">
          <div class="text-[12px] font-medium truncate" title="${esc(d.doc_name)}">${esc(d.doc_name)}</div>
          <div class="num text-[10.5px] text-[var(--ink4)]">${d.chunks} 块 · ${d.chars} 字符</div>
        </div>
        <button data-del="${esc(d.doc_id)}" class="opacity-0 group-hover:opacity-100 text-[var(--ink3)] hover:text-[var(--bad)] text-xs p-1.5 rounded-md hover:bg-[var(--bad-soft)]" title="删除本文档">
          <span class="inline-flex">${IC.trash}</span>
        </button>
      </div>`).join("");
    box.querySelectorAll("[data-del]").forEach((btn) =>
      btn.addEventListener("click", () => {
        api("/api/knowledge/" + btn.dataset.del, { method: "DELETE" }).then(refreshSidebar).catch(() => {});
      })
    );
  }).catch(() => {});
}

// ---------------------------------------------------------------- 匹配范围
function scopeSelected() {
  return scope === null ? new Set(allDocs.map((d) => d.doc_id)) : scope;
}

function renderScope() {
  const list = $("scope-list");
  const sel = scopeSelected();
  const n = allDocs.length;
  const picked = allDocs.filter((d) => sel.has(d.doc_id)).length;
  const cnt = $("scope-count");
  cnt.textContent = n ? (picked === n ? `全部 ${n} 篇` : `已选 ${picked} / ${n} 篇`) : "知识库为空";
  cnt.style.color = n && !picked ? "var(--bad)" : "";
  list.innerHTML = n
    ? allDocs.map((d) => {
        const on = sel.has(d.doc_id);
        return `<button type="button" data-scope="${esc(d.doc_id)}"
          class="scope-chip${on ? " on" : ""}" title="${esc(d.doc_name)} · ${d.chunks} 块">
          <span class="scope-tick">${on ? IC.check : ""}</span>
          <span class="truncate max-w-[180px]">${esc(d.doc_name)}</span>
        </button>`;
      }).join("")
    : `<span class="text-[11.5px] text-[var(--ink4)] py-1">知识库里还没有文档，先上传或点「示例」载入。</span>`;
  list.querySelectorAll("[data-scope]").forEach((btn) =>
    btn.addEventListener("click", () => {
      const cur = scopeSelected();
      const id = btn.dataset.scope;
      if (cur.has(id)) cur.delete(id);
      else cur.add(id);
      scope = cur;
      renderScope();
    })
  );
}

function setScopeAll(on) {
  scope = on ? null : new Set();
  renderScope();
}

// ---------------------------------------------------------------- 步骤卡片
// 切分策略中文名（后端传键名 strategy）
const CHUNK_LABEL = { recursive: "递归字符切分", structure: "按段落结构" };
const STRATEGY_HINT = {
  recursive: "每块约「切分大小」字符，块与块之间按「切分重叠」字符滑动重叠；标题行不特殊处理。",
  structure: "块按所选「标题等级」为界（如 H2：一个二级章节成一块，块首带标题）。章节超过「单块字符上限」时先按更细标题再拆，没有更细标题才按字符兜底拆；无标题文档退化为字符切分。",
};

// 每条 rail 步骤：左侧信号节点（状态决定外观）+ 右侧“抽屉”详情
function stepCard(key, label, status, ms, bodyHtml, open = true) {
  return `
  <div class="srow st-${status}">
    <span class="snode"><span class="nd"></span></span>
    <details class="slot ${open ? "" : ""}" ${open ? "open" : ""}>
      <summary class="flex items-center gap-2 pl-3 pr-2 py-2 min-h-[46px] select-none">
        <span class="text-[13px] font-medium">${label}</span>
        <span class="lbl hidden md:inline">${key}</span>
        <span class="num text-[11px] text-[var(--ink4)] ml-auto flex-none">${fmtMs(ms)}</span>
        <span class="chev text-[var(--ink3)] flex-none">${IC.chevR}</span>
      </summary>
      <div class="px-3 py-2.5 rounded-b-[7px] text-[13px]">${bodyHtml}</div>
    </details>
  </div>`;
}

// 向量头部展示
function vecHeadHtml(head) {
  if (!head || !head.length) return "";
  return `<div class="lbl mb-1.5 !tracking-[.05em]">向量头部 6 维 · embedding 归一化后</div>
    <code class="num text-[11px] block overflow-x-auto px-3 py-2 hair rounded-md bg-[var(--pane2)]" style="color:var(--accent-deep)">[${head.map(esc).join(", ")} …]</code>`;
}

// 键值元数据行（Stripe 式，替代彩色胶囊堆叠）
function metaChips(items) {
  return `<div class="kvs">
    ${items.filter(([k, v]) => v != null).map(([k, v]) =>
      `<span class="kv"><span class="k">${k}</span><span class="v">${esc(v)}</span></span>`).join("")}</div>`;
}

// ---------------------------------------------------------------- 回答渲染
// 模型输出的是 Markdown。以前整段 esc() 当纯文本贴出来，**加粗**、## 标题、- 列表
// 全成了字面上的字符——这是别人一眼就能看出「没做完」的地方。
const MD_OK = typeof window.marked !== "undefined" && typeof window.DOMPurify !== "undefined";
if (MD_OK) window.marked.setOptions({ gfm: true, breaks: true });

// Markdown → 安全 HTML。模型输出属于不可信输入，必须过一遍 DOMPurify：
// 它能返回 <img onerror=…>，直接塞 innerHTML 就是 XSS。
function mdToHtml(text) {
  const s = String(text ?? "");
  if (!s) return "";
  if (!MD_OK) return esc(s); // CDN 被墙 / 断网时退回纯文本，不至于整块空掉
  try {
    return window.DOMPurify.sanitize(window.marked.parse(s), {
      FORBID_TAGS: ["style", "form", "input", "textarea", "iframe", "object", "embed"],
      FORBID_ATTR: ["style"],
    });
  } catch (e) {
    return esc(s);
  }
}

// 把文本节点里的 [n] 换成可点击的引用标签。
// 必须走 DOM 而不是字符串拼接：拼字符串就得先 esc 再插 <span>，
// 那样渲染出来的 <strong>/<li> 会被一起转义掉。
function attachCites(root, refs) {
  const set = new Set((refs || []).map((r) => Number(r.rank)).filter(Number.isFinite));
  if (!set.size || !root) return;
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      if (!/\[\d+\]/.test(node.nodeValue)) return NodeFilter.FILTER_REJECT;
      // 代码块里的 [1] 是代码，不是引用
      for (let p = node.parentNode; p && p !== root; p = p.parentNode) {
        if (/^(CODE|PRE|A)$/.test(p.nodeName)) return NodeFilter.FILTER_REJECT;
      }
      return NodeFilter.FILTER_ACCEPT;
    },
  });
  const targets = [];
  let n;
  while ((n = walker.nextNode())) targets.push(n);

  targets.forEach((node) => {
    const s = node.nodeValue;
    const frag = document.createDocumentFragment();
    const re = /\[(\d+)\]/g;
    let last = 0, m, hit = false;
    while ((m = re.exec(s))) {
      const rank = parseInt(m[1], 10);
      if (!set.has(rank)) continue;
      if (m.index > last) frag.appendChild(document.createTextNode(s.slice(last, m.index)));
      const span = document.createElement("span");
      span.className = "cite";
      span.dataset.rank = String(rank);
      span.title = `定位到检索命中 #${rank}`;
      span.textContent = m[0];
      frag.appendChild(span);
      last = m.index + m[0].length;
      hit = true;
    }
    if (!hit) return;
    if (last < s.length) frag.appendChild(document.createTextNode(s.slice(last)));
    node.parentNode.replaceChild(frag, node);
  });
}

// 渲染一段回答 → HTML 字符串。先在游离节点上改完 DOM 再取 innerHTML。
function renderAnswer(text, refs) {
  const host = document.createElement("div");
  host.innerHTML = mdToHtml(text);
  attachCites(host, refs);
  return host.innerHTML;
}

// 流式回答 / 解读卡片（generate 与 m_interpret 共用）
function answerCardHtml(d, liveText, title) {
  const refs = flow.references || [];
  const final = (d.answer || "").trim();
  const streaming = !final && !!liveText;
  const body = final
    ? renderAnswer(d.answer, refs)
    : (streaming ? renderAnswer(liveText, refs) : `<span class="ink3 italic">等待回答…</span>`);
  return metaChips([["耗时", fmtMs(d.ms)], ["字符数", d.chars], ["输入 tokens", (d.usage || {}).prompt_tokens], ["输出 tokens", (d.usage || {}).completion_tokens]]) +
    `<div class="mt-2 hair rounded-lg px-4 py-3 bg-[var(--pane2)]">
       <div class="lbl mb-2">${title}</div>
       <div id="ans-text" class="md${streaming ? " streaming" : ""}">${body}</div>
     </div>
     ${refs.length ? `
     <div class="mt-2.5 flex flex-wrap items-baseline gap-x-2 gap-y-1.5 text-[11px] ink2">
       <span class="lbl !tracking-[.05em] flex-none">引用来源 · 点击回答里的 [n] 定位到上方原文</span>
       ${refs.map((r) =>
         `<span class="inline-flex items-center gap-1.5 pl-1 pr-2 py-0.5 rounded-md hair bg-[var(--panel)]">
            <b class="rank-tag !w-[17px] !h-[17px] !rounded text-[var(--amber)]" style="background:var(--amber-soft)">${r.rank}</b>
            ${esc(r.doc_name)}</span>`).join("")}
     </div>` : ""}`;
}

// 召回数之外被截断的命中：只列摘要，用来判断召回数 / Top-K 是不是设小了
function truncatedHtml(d) {
  const t = d.truncated || [];
  if (!t.length) return "";
  // 每条都可展开看正文开头 160 字（后端已裁好并标了 cut），
  // 否则 19 行只看到一堆同名前缀的文档名，判断不出「截断里有没有相关的」。
  const row = (h) => `
    <details class="hair-b">
      <summary class="flex items-baseline gap-2 py-1.5 text-[11.5px] cursor-pointer select-none">
        <span class="chev flex-none">${IC.chevR}</span>
        <span class="num ink4 w-6 flex-none text-right">${h.rank}</span>
        <span class="ink2 truncate" title="${esc(h.doc_name)}">${esc(h.doc_name)} <span class="ink4">· 片段 ${h.chunk_index + 1}</span></span>
        <span class="num ml-auto flex-none ink3">${(h.score * 100).toFixed(1)}%</span>
      </summary>
      <pre class="wrap text-[11.5px] leading-relaxed px-2 pb-2 ml-7" style="color:var(--ink3)">${esc(h.text)}${h.cut ? "…" : ""}</pre>
    </details>`;
  return `
    <details class="mt-1 hair rounded-lg overflow-hidden bg-[var(--panel)]">
      <summary class="px-3 py-2 text-[11.5px] ink3 flex items-center gap-1.5 select-none">
        <span class="chev">${IC.chevR}</span>
        紧随其后、被召回数截断的 ${t.length} 条 · 展开可看正文开头，里面若有相关的，说明召回数设小了
      </summary>
      <div class="px-3 pb-2">
        ${t.map(row).join("")}
      </div>
    </details>`;
}

// 精排：左右双列对照 —— 左「不精排时本来会用的 Top-K」对右「精排后的 Top-K」
function rerankDetailHtml(d, stageKey) {
  if (d.error) {
    return `<div class="text-[13px] rounded-lg px-3 py-2 flex items-start gap-2" style="color:var(--bad);background:var(--bad-soft)">${IC.x}<span>${esc(d.error)}</span></div>`;
  }
  if (d.enabled === undefined) {
    // status=run 时后端没带 detail，此时还不能断言「未启用」
    return `<div class="text-[12px] ink3 bg-[var(--pane2)] hair rounded-lg px-3 py-2">正在调用精排模型为候选块打分…</div>`;
  }
  if (!d.enabled) {
    return `<div class="text-[12px] ink3 bg-[var(--pane2)] hair rounded-lg px-3 py-2 flex items-start gap-1.5">${IC.filter}<span>${esc(d.message || "未启用精排")}</span></div>`;
  }
  const items = d.items || [];
  if (!items.length) {
    return `<div class="text-[12px] ink2 bg-[var(--pane2)] hair rounded-lg px-3 py-2 flex items-start gap-2">${IC.frown}<span>${esc(d.reason || "精排后没有结果")}</span></div>`;
  }

  const retrieveKey = stageKey === "m_rerank" ? "m_retrieve" : "retrieve";
  const coarse = ((flow.steps[retrieveKey] || {}).data || {}).hits || [];
  const topK = d.top_k || items.length;
  const baseline = coarse.slice(0, topK); // 不做精排时本来会用的那几条
  const finalIds = new Set(items.map((it) => it.id));

  const leftRow = (h) => `
    <div class="rr-row ${finalIds.has(h.id) ? "" : "rr-out"}" data-lkey="${esc(h.id)}">
      <span class="rank-tag !w-[18px] !h-[18px] !rounded flex-none" style="background:#ebe4d6;color:var(--ink2)">${h.rank}</span>
      <span class="text-[11.5px] truncate min-w-0 flex-1" title="${esc(h.doc_name)}">${esc(h.doc_name)} <span class="ink4">· 片段 ${h.chunk_index + 1}</span></span>
      <span class="num text-[10.5px] ink3 flex-none">${(h.score * 100).toFixed(1)}%</span>
    </div>`;

  const rightRow = (it) => {
    const delta = it.delta;
    const cls = delta > 0 ? "rr-up" : delta < 0 ? "rr-down" : "rr-same";
    const label = delta > 0 ? `↑${delta} 原 #${it.prev_rank}`
      : delta < 0 ? `↓${-delta} 原 #${it.prev_rank}`
      : "名次持平";
    return `
    <div class="rr-row ${cls}" data-rkey="${esc(it.id)}">
      <span class="rank-tag !w-[18px] !h-[18px] !rounded flex-none" style="color:var(--amber);background:var(--amber-soft)">${it.rank}</span>
      <span class="text-[11.5px] truncate min-w-0 flex-1" title="${esc(it.doc_name)}">${esc(it.doc_name)} <span class="ink4">· 片段 ${it.chunk_index + 1}</span></span>
      <span class="num text-[10.5px] ink2 flex-none">${it.score.toFixed(3)}</span>
      <span class="mini-badge flex-none px-1.5 py-0.5 rounded" style="background:var(--pane3);color:var(--ink3)">${
        it.prev_rank > topK ? `粗排 #${it.prev_rank} 杀入` : label}</span>
    </div>`;
  };

  const dropped = d.dropped || [];
  return metaChips([
    ["模型", d.model], ["召回", `${d.recall_k} 条`], ["精排至", `${items.length} 条`],
    ["名次变动", `${d.changed ?? 0} 条`], ["tokens", d.tokens], ["耗时", fmtMs(d.ms)],
  ]) +
    `<div class="flex items-start gap-1.5 text-[11.5px] ink3 mb-2">${IC.filter}<span>左＝不做精排时本来会用的 ${topK} 条，右＝精排后的最终结果。连线标出每一名的去向，变灰＝被挤出。</span></div>
     <div class="rr-wrap">
       <div class="rr-cols">
         <div class="rr-col">
           <div class="lbl mb-1.5">向量粗排 Top-${topK} · 余弦相似度</div>
           ${baseline.map(leftRow).join("") || `<div class="text-[11.5px] ink4">无命中</div>`}
         </div>
         <div class="rr-col">
           <div class="lbl mb-1.5">精排结果 Top-${items.length} · cross-encoder 打分</div>
           ${items.map(rightRow).join("")}
         </div>
       </div>
       <svg class="rr-links"></svg>
     </div>` +
    (dropped.length
      ? `<div class="mt-2 text-[11.5px] rounded-lg px-2.5 py-1.5 flex items-start gap-1.5" style="color:var(--bad);background:var(--bad-soft)">${IC.x}
           <span>被精排挤出 Top-${topK}：${dropped.map((x) => `粗排 #${x.prev_rank}《${esc(x.doc_name)}》片段 ${x.chunk_index + 1}`).join("、")}</span></div>`
      : "");
}

// 在双列之间画出「粗排名次 → 精排名次」的贝塞尔连线（需要等 DOM 布局完成后测量）
function drawRerankLinks(wrap) {
  const svg = wrap.querySelector(".rr-links");
  const cols = wrap.querySelector(".rr-cols");
  if (!svg || !cols) return;
  const box = cols.getBoundingClientRect();
  if (!box.width || !box.height) return;
  svg.setAttribute("viewBox", `0 0 ${box.width} ${box.height}`);
  svg.setAttribute("width", box.width);
  svg.setAttribute("height", box.height);

  const rightById = new Map();
  wrap.querySelectorAll("[data-rkey]").forEach((r) => rightById.set(r.dataset.rkey, r));

  const paths = [];
  wrap.querySelectorAll("[data-lkey]").forEach((lr) => {
    const rr = rightById.get(lr.dataset.lkey);
    if (!rr) return; // 被挤出的条目：不画线，靠左侧置灰表示
    const a = lr.getBoundingClientRect();
    const b = rr.getBoundingClientRect();
    const x1 = a.right - box.left, y1 = a.top + a.height / 2 - box.top;
    const x2 = b.left - box.left, y2 = b.top + b.height / 2 - box.top;
    const mid = (x1 + x2) / 2;
    const cls = rr.classList.contains("rr-up") ? "up" : rr.classList.contains("rr-down") ? "down" : "same";
    paths.push(`<path class="rrl rrl-${cls}" d="M${x1},${y1} C${mid},${y1} ${mid},${y2} ${x2},${y2}"/>`);
  });
  svg.innerHTML = paths.join("");
}

function detailHtml(stage, d, liveText) {
  d = d || {};
  if (d.error) return `<div class="text-[13px] rounded-lg px-3 py-2 flex items-start gap-2" style="color:var(--bad);background:var(--bad-soft)">${IC.x}<span>${esc(d.error)}</span></div>`;
  switch (stage) {
    case "q_embed":
    case "m_embed":
      return metaChips([["模型", d.model], ["维度", d.dimensions], ["耗时", fmtMs(d.ms)], ["tokens", d.tokens]]) + vecHeadHtml(d.vector_head);
    case "retrieve":
    case "m_retrieve": {
      const aboveZero = d.above != null;
      const threshNote = aboveZero
        ? `<div class="text-[11.5px] rounded-lg px-2.5 py-1.5 mb-2.5 flex items-start gap-1.5 ${(d.hits && d.hits.length) ? "ink3 bg-[var(--pane2)]" : ""}" style="${(d.hits && d.hits.length) ? "" : "color:var(--amber);background:var(--amber-soft)"}">
            ${IC.filter}<span>相似度阈值 <b class="num">${d.min_score}</b>：高于阈值的命中共 <b class="num">${d.above}</b> 条 · 展示前 ${d.top_k ?? "?"} 条${(d.top_k != null && d.above > d.top_k) ? "（其余被 Top-K 截断）" : ""}${d.above === 0 ? "，当前没有任何命中达到阈值" : ""}</span></div>`
        : "";
      const empty = !d.hits || !d.hits.length;
      if (empty) {
        return metaChips([["召回", d.top_k], ["阈值", d.min_score], ["库规模", `${d.total_chunks ?? 0} 块`], ["维度", d.dim ?? "-"], ["耗时", fmtMs(d.ms)]]) +
          threshNote +
          `<div class="text-[12px] ink2 bg-[var(--pane2)] hair rounded-lg px-3 py-2 flex items-start gap-2">${IC.frown}<span>${
            d.total_chunks
              ? "没有达到展示条件的命中。可尝试调低相似度阈值、换个表述，或向知识库补充内容。"
              : "知识库为空，先在上方「示例」载入或上传文档。"}</span></div>` +
          truncatedHtml(d);
      }
      return metaChips([["召回", d.top_k], ["阈值", d.min_score], ["库规模", `${d.total_chunks} 块`], ["维度", d.dim], ["耗时", fmtMs(d.ms)]]) +
        threshNote +
        d.hits.map((h) => {
          const w = Math.max(2, Math.min(100, Math.round(h.score * 100)));
          const hi = h.score >= 0.55;
          return `
          <div class="hit mb-2 overflow-hidden" data-rank="${h.rank}">
            <div class="hit-hd flex items-center gap-2.5 px-3 py-2">
              <span class="rank-tag ${h.rank === 1 ? "" : "ink3"}" style="${h.rank === 1 ? "color:var(--amber);background:var(--amber-soft)" : "background:#ebe4d6"}">${h.rank}</span>
              <div class="flex-1 min-w-0">
                <div class="flex items-baseline justify-between gap-3">
                  <span class="text-[12px] font-medium truncate" title="${esc(h.doc_name)}">${esc(h.doc_name)} <span class="ink3 font-normal">· 片段 ${h.chunk_index + 1}</span></span>
                  <span class="num text-[11px] flex-none" style="color:${hi ? "var(--good)" : "var(--amber)"}">${(h.score * 100).toFixed(1)}%</span>
                </div>
                <div class="bar mt-1.5"><i style="width:${w}%;background:${hi ? "var(--good)" : "var(--amber)"};opacity:.85"></i></div>
              </div>
            </div>
            <details class="bg-[var(--panel)]"><summary class="px-3 py-1.5 text-[11px] ink3 hover:ink2 flex items-center gap-1.5">
              <span class="chev">${IC.chevR}</span>查看检索到的原文</summary>
            <div class="px-3 pb-3"><pre class="wrap text-[12px] leading-relaxed text-[var(--ink2)]">${esc(h.text)}</pre></div></details>
          </div>`;
        }).join("") +
        truncatedHtml(d);
    }
    case "rerank":
    case "m_rerank":
      return rerankDetailHtml(d, stage);
    case "context":
    case "m_prompt": {
      const msgs = d.messages || [];
      // 必须取「最后一条」user：带上历史之后，第一条 user 是上一轮的提问，
      // 把历史当成本轮注入内容展示出来是错的。
      const users = msgs.filter((m) => m.role === "user");
      const user = users[users.length - 1] || {};
      const system = msgs.find((m) => m.role === "system") || {};
      const turns = [];
      for (let i = 0; i + 1 < users.length; i++) {
        turns.push([users[i].content || "", (msgs[msgs.indexOf(users[i]) + 1] || {}).content || ""]);
      }
      const copy = esc(user.content || "");
      const expanded = d.expanded || [];
      const grown = expanded.filter((e) => (e.neighbors || []).length);
      const nbChip = d.neighbor_window
        ? ["邻块扩展", grown.length ? `±${d.neighbor_window} 块（${grown.length} 条已扩展）` : `±${d.neighbor_window} 块（本页无邻块）`]
        : ["邻块扩展", "关闭"];
      const histHtml = turns.length ? `
        <details class="hair rounded-lg bg-[var(--panel)] overflow-hidden">
          <summary class="px-3 py-1.5 text-[11px] ink3 flex items-center gap-1.5 select-none">
            <span class="chev">${IC.chevR}</span>上文 ${turns.length} 轮 · 只用于理解指代，引用编号只算本轮资料</summary>
          <div class="px-3 pb-2.5 space-y-2">
            ${turns.map(([q, a], i) => `
              <div class="text-[12px]">
                <div class="ink2"><b class="ink4 mr-1.5">问${i + 1}</b>${esc(clip(q, 160))}</div>
                <div class="ink3 mt-0.5"><b class="ink4 mr-1.5">答${i + 1}</b>${esc(clip(a, 160))}</div>
              </div>`).join("")}
          </div>
        </details>` : "";
      const expandNote = grown.length ? `
        <details class="mt-2.5 hair rounded-lg bg-[var(--panel)]">
          <summary class="px-3 py-2 text-[11.5px] ink3 flex items-center gap-1.5 select-none">
            <span class="chev">${IC.chevR}</span>邻块扩展明细 · 实际送进模型的原文范围</summary>
          <div class="px-3 pb-2.5">
            ${expanded.map((e) => `
              <div class="flex items-baseline gap-2 text-[11.5px] pt-1.5">
                <b class="rank-tag !w-[17px] !h-[17px] !rounded flex-none" style="color:var(--amber);background:var(--amber-soft)">${e.rank}</b>
                <span class="ink2 truncate">${esc(e.doc_name)} <span class="ink4">· 片段 ${e.chunk_index + 1}</span></span>
                <span class="num ml-auto flex-none ink3">${e.chars} → ${e.context_chars} 字${
                  (e.neighbors || []).length
                    ? ` <span style="color:var(--accent)">含邻块 ${e.neighbors.map((n) => n.chunk_index + 1).join("/")}</span>`
                    : ""}</span>
              </div>`).join("")}
          </div>
        </details>` : "";
      return metaChips([["命中资料", `${d.hits_used} 条`], nbChip, ["估算 tokens", d.est_tokens], ["耗时", fmtMs(d.ms)]]) +
        (histHtml ? `<div class="mt-2.5">${histHtml}</div>` : "") +
        `<div class="space-y-2.5 text-[12.5px] mt-2.5">
          <div>
            <div class="flex items-center justify-between mb-1.5">
              <span class="font-medium ink2 flex items-center gap-1.5">${IC.term}<span class="lbl !tracking-[.05em] !text-[11px]">SYSTEM · 系统提示词</span></span>
            </div>
            <pre class="wrap cblock p-3 leading-relaxed text-[12.5px]">${esc(system.content || "")}</pre>
          </div>
          <div>
            <div class="flex items-center justify-between mb-1.5">
              <span class="font-medium ink2 flex items-center gap-1.5">${IC.seg}<span class="lbl !tracking-[.05em] !text-[11px]">USER · 注入给模型的内容（参考资料 + 问题）</span></span>
              <button class="copy-btn mini-badge px-2 py-0.5 rounded-md btn-line" data-copy="${copy}">复制</button>
            </div>
            <details open class="hair rounded-lg bg-[var(--pane2)]"><summary class="px-3 py-1.5 text-[11px] ink3 flex items-center gap-1.5">
              <span class="chev">${IC.chevR}</span>展开完整 Prompt</summary>
            <pre class="wrap p-3 text-[12.5px] leading-relaxed text-[var(--ink2)]">${esc(user.content || "")}</pre></details>
          </div>
        </div>` + expandNote;
    }
    case "generate":
      return answerCardHtml(d, liveText, "FINAL · 最终回答（模型生成）");
    case "m_interpret":
      return answerCardHtml(d, liveText, "INTERP · 模型解读（检索结果与待匹配文本的关系）");
    // 建库：数据清洗
    case "clean": {
      if (!d.enabled) {
        return `<div class="text-[12px] ink3 bg-[var(--pane2)] hair rounded-lg px-3 py-2">${IC.filter} ${esc(d.message || "未启用数据清洗")}</div>`;
      }
      if (d.error) return `<div class="text-[13px] rounded-lg px-3 py-2" style="color:var(--bad);background:var(--bad-soft)">${esc(d.error)}</div>`;
      const removed = d.removed_lines || 0;
      const chips = metaChips([
        ["删除", `${removed} 行 / ${d.removed_chars} 字`],
        ["清洗前", `${d.before_chars} 字`], ["清洗后", `${d.after_chars} 字`],
        ["噪声行", d.noise ? "开" : "关"], ["去重复", d.dedup ? "开" : "关"],
      ]);
      const rules = (d.rules || []).map((r) =>
        `<li class="flex items-start gap-2"><span class="mt-1" style="color:var(--good)">${IC.check}</span><span>${esc(r)}</span></li>`).join("");
      const samples = (d.samples || []).map((s) =>
        `<div class="hair bg-[var(--pane2)] rounded-md px-2.5 py-1.5 mb-1.5 flex items-start gap-2">
           <span class="flex-none mini-badge px-1.5 py-0.5 rounded ${s.reason.includes("重复") ? "text-[var(--amber)]" : "text-[var(--bad)]"}" style="background:${s.reason.includes("重复") ? "var(--amber-soft)" : "var(--bad-soft)"}">${esc(s.reason)}</span>
           <div class="min-w-0 flex-1"><pre class="wrap text-[11px] ink3 leading-relaxed max-h-16 overflow-hidden">${esc(s.text)}</pre></div>
         </div>`).join("");
      return chips +
        (rules ? `<ul class="space-y-1 my-2 text-[12px] ink2">${rules}</ul>` : "") +
        (samples ? `<div class="lbl mt-1 mb-1.5 !tracking-[.05em]">被删除内容${d.removed_lines > 8 ? "（示例前 8 条）" : ""}</div>${samples}` : "") +
        (removed === 0 ? `<div class="text-[11.5px] mt-1 inline-flex items-center gap-1.5 rounded-md px-2.5 py-1" style="color:var(--good);background:var(--good-soft)">${IC.check}未发现需要清理的内容</div>` : "");
    }
    // 建库四步
    case "parse":
      return metaChips([["来源", d.source], ["字符数", d.chars], ["页数", d.pages]]) +
        `<div class="lbl mt-1 mb-1.5 !tracking-[.05em]">解析出的文本预览</div><pre class="wrap text-[12px] text-[var(--ink2)] bg-[var(--pane2)] hair rounded-md p-2.5">${esc(d.preview)}${esc((d.preview || "").length >= 300 ? "\n……" : "")}</pre>`;
    case "chunk": {
      const chips = [
        ["分块数", d.count],
        ["策略", CHUNK_LABEL[d.strategy] || d.strategy || "递归字符切分"],
      ];
      if (d.strategy === "structure") chips.push(["标题等级", "H" + (d.structure_level || 2)]);
      chips.push(["上限", d.chunk_size], ["overlap", d.overlap]);
      return metaChips(chips) +
        (d.samples || []).map((s) =>
          `<div class="hair bg-[var(--pane2)] rounded-md px-2.5 py-1.5 mb-1.5">
            <div class="num text-[10.5px] ink4 mb-0.5">#${s.chunk_index} · ${s.len} 字符</div>
            <div class="text-[12px] text-[var(--ink2)] leading-relaxed line-clamp-2">${esc(s.head)}${(s.len > 140) ? "…" : ""}</div>
          </div>`).join("") +
        (d.count > 12 ? `<div class="text-[11px] ink3">… 其余 ${d.count - 12} 块略</div>` : "");
    }
    case "embed":
      return metaChips([
        ["模型", d.model],
        ["维度", d.dimensions],
        ["批次数", d.batches],
        ...(d.cached ? [["缓存命中", `${d.cached} 条（未重复计费）`]] : []),
        ["tokens", d.tokens],
        ["耗时", fmtMs(d.ms)],
      ]) + vecHeadHtml(d.vector_head);
    case "index":
      return metaChips([["本次新增", `${d.new_chunks} 块`], ["库中文档", d.docs + " 篇"], ["库总块数", d.chunks + " 块"], ["维度", d.dim]]) +
        `<div class="mt-1.5 inline-flex items-center gap-1.5 text-[12px] rounded-md px-2.5 py-1.5" style="color:var(--good);background:var(--good-soft)">${IC.check}已写入本地向量库并持久化，重启不丢失。</div>`;
    default:
      return `<pre class="wrap text-[11px] ink3">${esc(JSON.stringify(d, null, 2))}</pre>`;
  }
}

// ---------------------------------------------------------------- 流程绘制（问答 / 内容匹配）
function clip(s, n) {
  s = String(s || "");
  return s.length > n ? s.slice(0, n) + "…" : s;
}

function emptyStateHtml() {
  const glyph = flow.mode === "match" ? IC.link : IC.bub;
  const line = flow.mode === "match"
    ? `<div class="text-[13px]">输入一段文本或句子，这里将展示它与知识库的相似度匹配全过程。</div>
       <div class="num text-[11px] ink3 mt-1.5">m_embed → m_retrieve（粗排召回）→ m_rerank（精排重排，含前后名次对照）${flow.withLlm ? " → m_prompt → m_interpret" : ""}</div>`
    : `<div class="text-[13px]">提交一个问题，这里将按步骤实时点亮 RAG 问答全过程。</div>
       <div class="num text-[11px] ink3 mt-1.5">q_embed → retrieve（粗排召回）→ rerank（精排重排）→ context（完整 Prompt）→ generate（流式）</div>`;
  return `<div class="text-center py-12 flex flex-col items-center gap-2 rise">
    <span class="inline-flex text-[var(--ink4)] opacity-80">${glyph}</span>${line}</div>`;
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
  $("btn-stop").classList.toggle("hidden", !flow.running);
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
    `<div class="rail">
       <div class="space-y-1">${cards}</div>
     </div>
     ${flow.running ? `<div class="mt-3 flex items-center justify-center gap-2 text-[11px]" style="color:var(--accent)">
        <span class="inline-block w-3 h-3 rounded-full border-2" style="border-color:var(--accent);border-top-color:transparent"></span>
        <span class="num">${flow.mode === "match" ? "RUN · 正在匹配（可选解读）…" : "RUN · 正在执行检索生成流水…"}</span></div>` : ""}`;
  body.querySelectorAll(".copy-btn").forEach((b) =>
    b.addEventListener("click", () => {
      navigator.clipboard.writeText(b.dataset.copy).then(() => toast("已复制完整 Prompt")).catch(() => {});
    })
  );
  // 精排连线要等布局完成才能测量坐标；折叠展开会改变布局，需重画
  redrawRerankLinks(body);
  body.querySelectorAll(".slot").forEach((det) =>
    det.addEventListener("toggle", () => { if (det.open) redrawRerankLinks(body); })
  );
  refreshExportBtn();
}

function redrawRerankLinks(root) {
  const wraps = (root || document).querySelectorAll(".rr-wrap");
  if (!wraps.length) return;
  requestAnimationFrame(() => wraps.forEach(drawRerankLinks));
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
  flow.abort = null;
  flow.stopped = false;
}

function applyStep(ev) {
  const { stage, status, data } = ev;
  flow.steps[stage] = { status, data, ms: data && data.ms };
  if (status === "run" && stage === flow.ansStage) flow._liveText = "";
  if (status === "done" && stage === flow.ansStage) flow.running = false;
  // 上下文一拼好就知道 [n] 指向谁了。等到 done 才挂引用的话，
  // 整个流式过程里回答里的 [1] 都是死的，最后一下才变可点。
  if (status === "done" && (stage === "context" || stage === "m_prompt")) {
    const ex = (data && data.expanded) || [];
    if (ex.length) flow.references = ex.map((e) => ({ rank: e.rank, doc_name: e.doc_name }));
  }
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
  persistTimer = setTimeout(() => {
    api("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(runParams()),
    }).catch((e) => toast("保存 Top-K / 阈值失败：" + e.message, true));
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
  // 注意：取的是「提问之前」的历史，本轮自己不能进上文
  await streamFlow("/api/query", {
    question,
    history: flow.history.slice(-HIST_TURNS).map((t) => ({ question: t.q, answer: t.a })),
    ...runParams(),
  });
}

async function submitMatch(text) {
  text = (text || "").trim();
  if (!text) return;
  if (flow.running) { toast("上一轮内容匹配尚未结束"); return; }
  const picked = [...scopeSelected()];
  if (allDocs.length && !picked.length) { toast("至少要选一篇文档才能匹配"); return; }
  flow.mode = "match"; flow.withLlm = $("with-llm").checked; flow.question = ""; flow.text = text;
  resetFlow("match");
  drawQueryFlow();
  $("query-meta").textContent = "待匹配文本：" + clip(text, 40);
  await streamFlow("/api/match", {
    text, with_llm: flow.withLlm, doc_ids: picked, ...runParams(),
  });
}

async function streamFlow(path, body) {
  flow.abort = new AbortController();
  // 极端时序：刚点「停止」时请求还没建好，此时 flow.abort 还是 null，
  // 这里补一次中断，避免流程在「已停止」之后又自己跑起来。
  if (flow.stopped) flow.abort.abort();
  try {
    const res = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: flow.abort.signal,
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
    // 用户点「停止」触发的中断不是错误，保持当前进度并标记为已中断即可
    if (e && e.name === "AbortError") {
      markStopped();
    } else {
      flow.running = false;
      const key = flow.ansStage || flow.order[flow.order.length - 1];
      flow.steps[key] = { status: "error", data: { error: String(e.message || e) } };
      drawQueryFlow();
      toast(String(e.message || e), true);
    }
  } finally {
    flow.abort = null;
    if (flow.running) {
      // 正常以 done 收尾时已在 done 处理器重置 running；这里兜底
      flow.running = false;
      const last = flow.order[flow.order.length - 1];
      if (flow.steps[last] && flow.steps[last].status === "done") drawQueryFlow();
    }
  }
}

function markStopped() {
  flow.running = false;
  const key = flow.order.find((k) => !flow.steps[k] || flow.steps[k].status === "run") || flow.ansStage;
  if (key) {
    const st = flow.steps[key];
    flow.steps[key] = { status: "stopped", ms: st ? st.ms : null, data: st ? st.data : { note: "已中断" } };
  }
  drawQueryFlow();
  toast("已停止本次流程");
}

// 流式期间刷回答区。按帧节流：每个 token 都 parse 一遍 Markdown 会明显掉帧，
// 一帧一次视觉上完全看不出区别。
let liveRaf = 0;
function scheduleLiveRender() {
  if (liveRaf) return;
  liveRaf = requestAnimationFrame(() => {
    liveRaf = 0;
    if (!flow.running) return; // 收尾的整块重绘已经画完了，别用半截文本把它盖回去
    const ans = $("ans-text");
    if (!ans || !flow._liveText) return;
    ans.innerHTML = renderAnswer(flow._liveText, flow.references);
    ans.classList.add("streaming");
  });
}

function stopFlow() {
  if (!flow.running) return;
  flow.stopped = true;
  if (flow.abort) flow.abort.abort();
  else markStopped(); // 请求还没建好时的兜底，streamFlow 里会再补一次中断
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
    scheduleLiveRender();
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
      // 只有拿到完整回答才入库：半截回答当上文会污染下一轮的指代
      const answer = (payload.answer || "").trim();
      if (answer) {
        flow.history.push({
          q: payload.question || flow.question,
          a: answer,
          refs: payload.references || [],
          ms: payload.elapsed_ms || 0,
          ts: Date.now(),
        });
        if (flow.history.length > HIST_MAX) flow.history = flow.history.slice(-HIST_MAX);
        saveHistory();
        renderHistory();
      }
    }
    drawQueryFlow();
  }
}

// ---------------------------------------------------------------- 建库流水
function addIngestLog(entry) {
  const box = $("ingest-log");
  if (!entry) return;
  const ok = entry.ok;
  const title = (entry.doc_name || "文档").slice(0, 40);
  const time = new Date().toLocaleTimeString("zh-CN", { hour12: false });
  const steps = entry.transcript || [];
  const body = steps.length
    ? `<div class="space-y-1.5 mt-2 pt-2 hair-t">${steps.map((s) => {
        const ic = s.ok ? `<span style="color:var(--good)">${IC.check}</span>` : `<span style="color:var(--bad)">${IC.x}</span>`;
        return `<details class="hair rounded-md overflow-hidden bg-[var(--panel)]">
          <summary class="flex items-center gap-2 px-2.5 py-1.5 text-[12px] select-none">
            <span class="flex-none inline-flex">${ic}</span><span class="ink2">${esc(s.label)}</span>
            <span class="num text-[10px] ink4 ml-auto flex-none">${fmtMs(s.ms)}</span>
            <span class="chev text-[var(--ink4)] flex-none">${IC.chevR}</span>
          </summary>
          <div class="px-2.5 pb-2.5">${detailHtml(s.stage, s.detail)}</div>
        </details>`;
      }).join("")}</div>`
    : (entry.error ? `<div class="text-[12px] mt-1.5" style="color:var(--bad)">${esc(entry.error)}</div>` : "");
  const el = document.createElement("div");
  el.className = "rise hair rounded-md bg-[var(--panel)] px-2.5 py-2 mb-2 text-left";
  el.innerHTML = `<div class="flex items-center gap-2 text-[12px]">
      <span class="inline-flex flex-none" style="color:${ok ? "var(--good)" : "var(--bad)"}">${ok ? IC.check : IC.x}</span>
      <span class="font-medium truncate">${esc(title)}</span>
      <span class="num text-[10px] text-[var(--ink4)] ml-auto flex-none">${time}</span></div>${body}`;
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
      hint.innerHTML = `<span class="inline-flex items-center gap-2" style="color:var(--accent)">
        <span class="inline-block w-3 h-3 rounded-full border-2" style="border-color:var(--accent);border-top-color:transparent;animation:spin 1s linear infinite"></span>
        正在建库：${esc(f.name)}…</span>`;
      try {
        const fd = new FormData();
        fd.append("file", f);
        const data = await api("/api/knowledge/upload", { method: "POST", body: fd });
        addIngestLog(data);
        if (data.error) hint.innerHTML = `<span style="color:var(--bad)">${esc(f.name)} 建库失败：${esc(data.error)}</span>`;
        else hint.innerHTML = `<span style="color:var(--good)">${esc(f.name)} 建库完成，新增 ${esc((data.transcript || []).find(s => s.stage === "index")?.detail?.new_chunks ?? "?")} 块</span>`;
      } catch (e) {
        hint.innerHTML = `<span style="color:var(--bad)">${esc(f.name)}：${esc(e.message)}</span>`;
      }
      await new Promise((r) => setTimeout(r, 400));
    }
    setTimeout(() => hint.classList.add("hidden"), 1600);
    refreshSidebar();
  })();
}

async function loadExamples() {
  const box = $("ingest-log");
  box.innerHTML = `<div class="text-center py-3 flex flex-col items-center gap-1.5" style="color:var(--accent)">
      <span class="inline-block w-4 h-4 rounded-full border-2" style="border-color:var(--accent);border-top-color:transparent;animation:spin 1s linear infinite"></span>
      <div class="num text-[11px]">正在逐篇解析并向量化内置示例…</div></div>`;
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
  document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("on", b.dataset.mode === mode));
  const isMatch = mode === "match";
  $("chip-row").classList.toggle("hidden", isMatch);
  $("match-scope").classList.toggle("hidden", !isMatch);
  $("llm-wrap").classList.toggle("hidden", !isMatch);
  $("llm-wrap").classList.toggle("flex", isMatch);
  if (isMatch) renderScope();
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
  L.push(`# RAG 工作台 · ${flow.mode === "match" ? "内容匹配" : "问答生成"}导出`);
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
        if ((d.truncated || []).length) {
          L.push("", `以下 ${d.truncated.length} 条被召回数截断，未能进入候选（若有相关的说明召回数设小了）：`);
          d.truncated.forEach((h) => {
            L.push(`- ${h.rank}. 《${h.doc_name}》片段 ${h.chunk_index + 1} — 相似度 ${(h.score * 100).toFixed(1)}%`);
          });
        }
        break;
      }
      case "rerank":
      case "m_rerank": {
        if (d.error) { L.push(`（失败）${d.error}`); break; }
        if (!d.enabled) { L.push(`（未启用）${d.message || ""}`); break; }
        L.push(
          `- 模型：${d.model} · 粗排召回 ${d.recall_k} 条 → 精排取前 ${d.top_k} 条` +
          ` · 名次变动 ${d.changed} 条 · tokens：${d.tokens}`
        );
        (d.items || []).forEach((it) => {
          const delta = it.delta;
          const move = delta > 0 ? `↑${delta}（粗排 #${it.prev_rank}）`
            : delta < 0 ? `↓${-delta}（粗排 #${it.prev_rank}）` : "持平";
          L.push(
            `${it.rank}. 《${it.doc_name}》片段 ${it.chunk_index + 1}` +
            ` — 精排分 ${it.score} · 余弦 ${(it.cosine_score * 100).toFixed(1)}% · 名次${move}`
          );
          L.push(`   > ${it.text.replace(/\n/g, "\n   > ")}`);
        });
        if ((d.dropped || []).length) {
          L.push("", "精排后被挤出 Top-K 的（粗排原本能进）：");
          d.dropped.forEach((h) => {
            L.push(`- 粗排 #${h.prev_rank}《${h.doc_name}》片段 ${h.chunk_index + 1} — 余弦 ${(h.score * 100).toFixed(1)}%`);
          });
        }
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

function updateStrategyHint() {
  const isStruct = $("cfg-strategy").value === "structure";
  $("strategy-hint").textContent = STRATEGY_HINT[isStruct ? "structure" : "recursive"];
}

function applyStrategyUI() {
  const isStruct = $("cfg-strategy").value === "structure";
  $("cfg-level-wrap").classList.toggle("hidden", !isStruct);
  $("lbl-cs").textContent = isStruct ? "单块字符上限" : "切分大小（字符）";
  updateStrategyHint();
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
    $("cfg-strategy").value = cfg.chunk_strategy || "recursive";
    $("cfg-level").value = String(cfg.structure_level || 2);
    applyStrategyUI();
    $("cfg-cs").value = cfg.chunk_size;
    $("cfg-co").value = cfg.chunk_overlap;
    $("cfg-rerank-en").checked = !!cfg.rerank_enabled;
    $("cfg-rerank-model").value = cfg.rerank_model || "gte-rerank-v2";
    $("cfg-recall").value = cfg.recall_k != null ? cfg.recall_k : 20;
    $("cfg-window").value = cfg.neighbor_window != null ? cfg.neighbor_window : 1;
    $("cfg-key").value = "";
    $("cfg-key").placeholder = c.has_key ? "已保存（留空则保持不变）" : "sk-… 填写你的百炼 API Key";
    const ks = $("key-status");
    ks.textContent = c.has_key ? "已配置" : "未配置";
    ks.className = "badge " + (c.has_key ? "b-good" : "b-bad");
    // 演示环境 Key 由服务端托管：留着输入框能填、点了却报 403，不如直接说明白
    $("cfg-key").disabled = !!c.demo;
    if (c.demo) {
      $("cfg-key").placeholder = "演示环境由服务端托管，无需填写";
      ks.textContent = "服务端托管";
      ks.className = "badge b-good";
    }
    $("settings-modal").classList.remove("hidden");
  } catch (e) { toast("读取设置失败：" + e.message, true); }
}

async function saveSettings() {
  const n = (v, d) => { const x = parseInt(v, 10); return isNaN(x) ? d : x; };
  const payload = {
    llm_model: $("cfg-llm").value,
    embedding_model: $("cfg-emb").value,
    dimensions: n($("cfg-dim").value, 1024),
    chunk_strategy: $("cfg-strategy").value,
    structure_level: Math.max(1, Math.min(6, n($("cfg-level").value, 2))),
    chunk_size: Math.max(50, n($("cfg-cs").value, 500)),
    chunk_overlap: Math.max(0, n($("cfg-co").value, 50)),
    clean_enabled: $("cfg-clean-en").checked,
    clean_noise: $("cfg-clean-noise").checked,
    clean_dedup: $("cfg-clean-dedup").checked,
    rerank_enabled: $("cfg-rerank-en").checked,
    rerank_model: $("cfg-rerank-model").value,
    recall_k: Math.max(1, Math.min(100, n($("cfg-recall").value, 20))),
    neighbor_window: Math.max(0, Math.min(5, n($("cfg-window").value, 1))),
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

// ---------------------------------------------------------------- 对话历史
// 刷新一下页面就把问过的东西全丢掉，是「玩具」和「工具」的分界；
// 而且追问（「那它呢？」）本来就依赖上文，没历史就答不对。
// 存 localStorage：纯本地、不开后端也在。
const HIST_KEY = "rag.history";
const HIST_MAX = 20;        // 列表最多留几轮
const HIST_TURNS = 3;       // 追问实际带几轮上文（后端还会再截一次，两边都限才安全）
const HIST_ANS_CAP = 4000;  // 单条回答入库上限，别把 localStorage 撑爆

function loadHistory() {
  try {
    const raw = JSON.parse(localStorage.getItem(HIST_KEY) || "[]");
    if (!Array.isArray(raw)) return [];
    return raw
      .filter((t) => t && typeof t.q === "string" && typeof t.a === "string")
      .slice(-HIST_MAX);
  } catch (e) {
    return [];
  }
}

function saveHistory() {
  try {
    localStorage.setItem(HIST_KEY, JSON.stringify(
      flow.history.slice(-HIST_MAX).map((t) => ({
        q: t.q, a: t.a.slice(0, HIST_ANS_CAP), refs: t.refs || [], ms: t.ms || 0, ts: t.ts,
      }))
    ));
  } catch (e) {
    // 隐私模式或配额满。问答本身不受影响，静默降级即可，没必要弹窗吓人
  }
}

function renderHistory() {
  const box = $("hist-list");
  const list = flow.history;
  $("hist-count").textContent = list.length ? `${list.length} 轮` : "";
  if (!list.length) {
    box.innerHTML = `<div class="text-center text-[var(--ink3)] py-6">
        <div class="text-[12px]">还没有对话</div>
        <div class="text-[11px] text-[var(--ink4)] mt-0.5">问过的问题会留在这里，刷新不丢</div>
      </div>`;
    return;
  }
  box.innerHTML = list.map((t, i) => {
    const n = i + 1;
    const ctx = n > list.length - HIST_TURNS;
    return `
    <details class="hist-turn hair-b">
      <summary class="flex items-start gap-2 px-2 py-1.5 rounded-md hover:bg-[var(--pane2)] cursor-pointer select-none">
        <b class="num mt-0.5 rank-tag !w-[18px] !h-[18px] !rounded flex-none" style="color:var(--accent-deep);background:var(--accent-soft)">${n}</b>
        <div class="min-w-0 flex-1">
          <div class="text-[12px] truncate" title="${esc(t.q)}">${esc(clip(t.q, 80))}</div>
          <div class="num text-[10.5px] text-[var(--ink4)] truncate">
            ${ctx ? `<span style="color:var(--accent)">带上文</span> · ` : ""}${t.a.length} 字${t.ms ? " · " + fmtMs(t.ms) : ""}
          </div>
        </div>
        <button data-drop="${i}" class="flex-none opacity-40 hover:opacity-100 text-[var(--ink4)] hover:text-[var(--bad)] p-0.5 rounded" title="从历史中移除">${IC.x}</button>
      </summary>
      <div class="px-2 pb-2.5" data-body="${i}"></div>
    </details>`;
  }).join("");

  box.querySelectorAll(".hist-turn").forEach((det) => {
    det.addEventListener("toggle", () => {
      if (!det.open) return;
      const i = Number(det.querySelector("[data-body]").dataset.body);
      renderHistBody(i, det.querySelector("[data-body]"));
    });
  });
  box.querySelectorAll("[data-drop]").forEach((btn) =>
    btn.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      flow.history.splice(Number(btn.dataset.drop), 1);
      saveHistory();
      renderHistory();
      toast("已从历史移除");
    })
  );
}

// 展开某一条时才渲染回答（20 轮全量渲染 Markdown 没必要）
function renderHistBody(i, host) {
  const t = flow.history[i];
  if (!t || host.dataset.done === "1") return;
  host.dataset.done = "1";
  const refs = t.refs || [];
  host.innerHTML =
    `<div class="mt-1.5 hair rounded-lg px-3 py-2.5 bg-[var(--pane2)]">
       <div class="lbl mb-2">FINAL · 该轮回答（历史记录）</div>
       <div class="md">${renderAnswer(t.a, refs)}</div>
     </div>` +
    (refs.length ? `<div class="mt-2 flex flex-wrap items-baseline gap-x-2 gap-y-1.5 text-[11px] ink2">
        <span class="lbl !tracking-[.05em] flex-none">引用来源</span>
        ${refs.map((r) => `<span class="inline-flex items-center gap-1.5 pl-1 pr-2 py-0.5 rounded-md hair bg-[var(--panel)]">
            <b class="rank-tag !w-[17px] !h-[17px] !rounded text-[var(--amber)]" style="background:var(--amber-soft)">${r.rank}</b>
            ${esc(r.doc_name)}</span>`).join("")}
      </div>` : "") +
    `<div class="mt-2 flex items-center gap-2">
       <button class="hist-again mini-badge px-2.5 py-1 rounded-md btn-line" data-again="${i}">填入输入框</button>
     </div>`;
  host.querySelector("[data-again]")?.addEventListener("click", () => {
    $("q-input").value = flow.history[i].q;
    $("q-input").focus();
    toast("已填入输入框");
  });
}

function newChat() {
  if (!flow.history.length) { toast("当前没有对话记录"); return; }
  if (!confirm(`开始新对话？当前 ${flow.history.length} 轮记录会被清空（不再作为追问上文）。`)) return;
  flow.history = [];
  saveHistory();
  renderHistory();
  $("q-input").value = "";
  toast("已开始新对话");
}

// ---------------------------------------------------------------- 知识库数据问题横幅
function renderKbAlert(stats) {
  const box = $("kb-alert");
  const problems = (stats && stats.problems) || [];
  if (!problems.length) {
    box.classList.add("hidden");
    box.innerHTML = "";
    return;
  }
  const blocked = stats.writable === false;
  box.classList.remove("hidden");
  box.innerHTML = `
    <div class="alert px-3 py-2.5 text-[12px]">
      <div class="flex items-start gap-2">
        <span class="flex-none inline-flex mt-px" style="color:var(--bad)">${IC.frown}</span>
        <div class="min-w-0">
          <div class="alert-t">知识库载入时发现问题${blocked ? "，已暂停写入" : "，已尽量恢复"}</div>
          <ul class="mt-1 space-y-0.5 ink2 list-disc pl-4">${problems.map((p) => `<li>${esc(p)}</li>`).join("")}</ul>
          ${blocked
            ? `<div class="mt-1.5 ink3">磁盘上的文件未被改动。建议先备份 data/kb 目录，再点「清空知识库」确认放弃原有数据。</div>`
            : ""}
        </div>
      </div>
    </div>`;
}

// ---------------------------------------------------------------- 主题
function setTheme(t) {
  document.documentElement.setAttribute("data-theme", t);
  try { localStorage.setItem("rag.theme", t); } catch (e) { /* 隐私模式 */ }
}

function toggleTheme() {
  const cur = document.documentElement.getAttribute("data-theme") === "dark" ? "dark" : "light";
  setTheme(cur === "dark" ? "light" : "dark");
}

// ---------------------------------------------------------------- 初始化
function init() {
  refreshSidebar();

  flow.history = loadHistory();
  renderHistory();
  $("btn-new-chat").addEventListener("click", newChat);
  $("btn-theme").addEventListener("click", toggleTheme);

  $("btn-settings").addEventListener("click", openSettings);
  $("btn-close-settings").addEventListener("click", () => $("settings-modal").classList.add("hidden"));
  $("settings-modal").addEventListener("click", (e) => { if (e.target.id === "settings-modal") $("settings-modal").classList.add("hidden"); });
  $("btn-save-settings").addEventListener("click", saveSettings);
  $("cfg-emb").addEventListener("change", () => fillDims($("cfg-emb").value));
  $("cfg-strategy").addEventListener("change", applyStrategyUI);
  $("btn-test").addEventListener("click", async () => {
    $("btn-test").disabled = true; $("btn-test").textContent = "测试中…";
    try { const r = await api("/api/test-key", { method: "POST" }); toast(r.message, !r.ok); }
    catch (e) { toast("测试失败：" + e.message, true); }
    $("btn-test").disabled = false; $("btn-test").textContent = "测试连接";
  });

  $("btn-examples").addEventListener("click", loadExamples);
  $("btn-clear").addEventListener("click", () => {
    if (!confirm("确定清空整个知识库？此操作不可恢复。")) return;
    api("/api/knowledge", { method: "DELETE" }).then(() => {
      $("ingest-log").innerHTML = `<div class="text-center ink3 py-2">已清空知识库</div>`;
      refreshSidebar();
    }).catch(() => {});
  });

  const dz = $("dropzone");
  const fi = $("file-input");
  dz.addEventListener("click", () => fi.click());
  fi.addEventListener("change", () => { doUploadFiles(fi.files); fi.value = ""; });
  ["dragover", "dragenter"].forEach((e) => dz.addEventListener(e, (ev) => { ev.preventDefault(); dz.classList.add("drop-on"); }));
  ["dragleave", "drop"].forEach((e) => dz.addEventListener(e, (ev) => { ev.preventDefault(); dz.classList.remove("drop-on"); }));
  dz.addEventListener("drop", (ev) => doUploadFiles(ev.dataTransfer.files));

  // 模式切换
  document.querySelectorAll(".tab").forEach((b) =>
    b.addEventListener("click", () => setMode(b.dataset.mode))
  );

  // 匹配范围：全选 / 全不选
  $("scope-all").addEventListener("click", () => setScopeAll(true));
  $("scope-none").addEventListener("click", () => setScopeAll(false));

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

  // 停止当前流式流程
  $("btn-stop").addEventListener("click", stopFlow);
  window.addEventListener("resize", () => redrawRerankLinks($("query-body")));

  setMode("query");
  loadRunControls();
}

document.addEventListener("DOMContentLoaded", init);
