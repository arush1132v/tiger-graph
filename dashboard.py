"""
RAG Pipeline Comparison Dashboard
Interactive Streamlit app for comparing LLM-Only, Vector RAG, and GraphRAG
Metrics: Token usage, latency, cost per query, answer accuracy (LLM-as-Judge + BERTScore)
"""
import streamlit as st
import time
from datetime import datetime
from pipeline_1_llm import get_pipeline_1_metadata
from pipeline_2_rag import get_pipeline_2_metadata
from pipeline_3_graphrag import get_pipeline_3_metadata


# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Benchmark",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Global Styles ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

/* ── Root & Base ── */
:root {
    --bg:       #0a0b0f;
    --surface:  #12141a;
    --border:   #1e2130;
    --p1:       #ff5c5c;
    --p2:       #00d4aa;
    --p3:       #4f9eff;
    --text:     #e8eaf0;
    --muted:    #6b7280;
    --mono:     'Space Mono', monospace;
    --sans:     'Syne', sans-serif;
}

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--sans) !important;
}

/* Streamlit chrome */
.stApp { background-color: var(--bg) !important; }
section[data-testid="stSidebar"] { background-color: var(--surface) !important; border-right: 1px solid var(--border); }
.block-container { padding: 2rem 2.5rem !important; max-width: 1400px; }

/* ── Typography ── */
h1, h2, h3 { font-family: var(--sans) !important; font-weight: 800 !important; letter-spacing: -0.02em; }
code, .mono { font-family: var(--mono) !important; }

/* ── Header Banner ── */
.rag-header {
    background: linear-gradient(135deg, #12141a 0%, #0d1422 50%, #12141a 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2.5rem 2rem 2rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.rag-header::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(79,158,255,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.rag-header::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 100px;
    width: 160px; height: 160px;
    background: radial-gradient(circle, rgba(0,212,170,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.rag-header h1 {
    font-size: 2rem !important;
    background: linear-gradient(90deg, #e8eaf0 30%, #4f9eff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.5rem !important;
}
.rag-header p {
    color: var(--muted);
    margin: 0;
    font-size: 0.9rem;
    line-height: 1.6;
}

/* ── Pipeline Badge Labels ── */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 100px;
    font-size: 0.7rem;
    font-weight: 700;
    font-family: var(--mono);
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.badge-p1 { background: rgba(255,92,92,0.15); color: var(--p1); border: 1px solid rgba(255,92,92,0.3); }
.badge-p2 { background: rgba(0,212,170,0.12); color: var(--p2); border: 1px solid rgba(0,212,170,0.25); }
.badge-p3 { background: rgba(79,158,255,0.12); color: var(--p3); border: 1px solid rgba(79,158,255,0.25); }

/* ── Pipeline Result Card ── */
.pipeline-result {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    position: relative;
}
.pipeline-result.p1 { border-top: 3px solid var(--p1); }
.pipeline-result.p2 { border-top: 3px solid var(--p2); }
.pipeline-result.p3 { border-top: 3px solid var(--p3); }

.pipeline-result h3 {
    margin: 0 0 1rem !important;
    font-size: 1rem !important;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Metric Grid ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.75rem;
    margin-bottom: 1.25rem;
}
.metric-cell {
    background: #0a0b0f;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.75rem 1rem;
}
.metric-cell .label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.3rem;
    font-family: var(--mono);
}
.metric-cell .value {
    font-size: 1.25rem;
    font-weight: 800;
    font-family: var(--mono);
}
.metric-cell .sub {
    font-size: 0.7rem;
    color: var(--muted);
    font-family: var(--mono);
    margin-top: 0.15rem;
}

/* Colour per pipeline */
.p1 .metric-cell .value { color: var(--p1); }
.p2 .metric-cell .value { color: var(--p2); }
.p3 .metric-cell .value { color: var(--p3); }

/* ── Accuracy Chips ── */
.accuracy-row {
    display: flex;
    gap: 0.75rem;
    align-items: center;
    margin-bottom: 1rem;
    flex-wrap: wrap;
}
.judge-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 5px 12px;
    border-radius: 100px;
    font-size: 0.75rem;
    font-weight: 700;
    font-family: var(--mono);
}
.judge-pass { background: rgba(0,212,170,0.15); color: #00d4aa; border: 1px solid rgba(0,212,170,0.3); }
.judge-fail { background: rgba(255,92,92,0.15); color: #ff5c5c; border: 1px solid rgba(255,92,92,0.3); }
.bert-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 5px 12px;
    border-radius: 100px;
    font-size: 0.75rem;
    font-weight: 700;
    font-family: var(--mono);
    background: rgba(255,255,255,0.05);
    border: 1px solid var(--border);
    color: var(--text);
}

/* ── Answer Box ── */
.answer-box {
    background: #0a0b0f;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem 1.25rem;
    font-size: 0.88rem;
    line-height: 1.7;
    color: #c8cad4;
}
.answer-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.5rem;
    font-family: var(--mono);
}

/* ── Comparison Table ── */
.compare-table {
    width: 100%;
    border-collapse: collapse;
    font-family: var(--mono);
    font-size: 0.82rem;
    margin-top: 1rem;
}
.compare-table th {
    background: var(--surface);
    color: var(--muted);
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.6rem 1rem;
    text-align: left;
    border-bottom: 1px solid var(--border);
}
.compare-table td {
    padding: 0.75rem 1rem;
    border-bottom: 1px solid var(--border);
    color: var(--text);
}
.compare-table tr:last-child td { border-bottom: none; }
.compare-table .winner { font-weight: 700; }
.compare-table .winner-p1 { color: var(--p1); }
.compare-table .winner-p2 { color: var(--p2); }
.compare-table .winner-p3 { color: var(--p3); }

/* ── Query Input ── */
.stTextArea textarea {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: var(--sans) !important;
    font-size: 0.9rem !important;
}
.stTextArea textarea:focus {
    border-color: var(--p3) !important;
    box-shadow: 0 0 0 2px rgba(79,158,255,0.15) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: #1a1d28 !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    font-family: var(--mono) !important;
    font-size: 0.8rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    padding: 0.6rem 1.2rem !important;
    transition: all 0.15s !important;
}
.stButton > button:hover {
    border-color: var(--p3) !important;
    background: rgba(79,158,255,0.1) !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1e3a5f 0%, #1a4080 100%) !important;
    border-color: rgba(79,158,255,0.5) !important;
    color: #a8d4ff !important;
}

/* ── Sidebar ── */
.stSidebar .stCheckbox > label { color: var(--text) !important; font-family: var(--mono) !important; font-size: 0.82rem !important; }
.stSidebar .stSelectbox label, .stSidebar h3, .stSidebar p { color: var(--text) !important; }
[data-testid="stSidebarContent"] { padding: 1.5rem 1rem !important; }

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* ── Section Header ── */
.section-head {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 2rem 0 1rem;
}
.section-head h2 {
    font-size: 1rem !important;
    margin: 0 !important;
    color: var(--muted);
    font-weight: 700 !important;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    font-family: var(--mono) !important;
}
.section-line { flex: 1; height: 1px; background: var(--border); }

/* ── Sources ── */
.source-item {
    background: #0a0b0f;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.8rem;
    font-family: var(--mono);
    color: var(--muted);
    line-height: 1.5;
}
.source-id {
    color: var(--p3);
    font-weight: 700;
    margin-bottom: 0.3rem;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* ── Status banner ── */
.status-bar {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.75rem 1.25rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    font-family: var(--mono);
    font-size: 0.8rem;
    color: var(--muted);
    margin-bottom: 1.5rem;
}
.status-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--p2); flex-shrink: 0; }

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #0a0b0f !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--muted) !important;
    font-family: var(--mono) !important;
    font-size: 0.78rem !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Constants ────────────────────────────────────────────────────────────────
PIPELINE_META = {
    "pipeline_1": {
        "label": "LLM-ONLY",
        "full":  "Pipeline 1 — LLM Only",
        "cls":   "p1",
        "badge": "badge-p1",
        "color": "#ff5c5c",
        "icon":  "◆",
        "cost_per_1k_prompt":     0.000125,   # $ per 1k tokens (adjust to your model)
        "cost_per_1k_completion": 0.000375,
    },
    "pipeline_2": {
        "label": "VECTOR RAG",
        "full":  "Pipeline 2 — Vector RAG",
        "cls":   "p2",
        "badge": "badge-p2",
        "color": "#00d4aa",
        "icon":  "●",
        "cost_per_1k_prompt":     0.000125,
        "cost_per_1k_completion": 0.000375,
    },
    "pipeline_3": {
        "label": "GRAPHRAG",
        "full":  "Pipeline 3 — GraphRAG",
        "cls":   "p3",
        "badge": "badge-p3",
        "color": "#4f9eff",
        "icon":  "▲",
        "cost_per_1k_prompt":     0.000125,
        "cost_per_1k_completion": 0.000375,
    },
}

EXAMPLE_QUESTIONS = [
    "What are the core metrics tracked in the dataset?",
    "Explain the main patterns in the data",
    "What anomalies or outliers exist?",
    "Summarize the key findings",
]


# ─── Helpers ──────────────────────────────────────────────────────────────────
def fmt_time(seconds: float) -> str:
    return f"{seconds * 1000:.0f}ms" if seconds < 1 else f"{seconds:.2f}s"


def fmt_cost(dollars: float) -> str:
    if dollars < 0.001:
        return f"${dollars * 100:.4f}¢"
    return f"${dollars:.5f}"


def calc_cost(meta: dict, prompt_tokens: int, completion_tokens: int) -> float:
    return (
        prompt_tokens     / 1000 * meta["cost_per_1k_prompt"]
        + completion_tokens / 1000 * meta["cost_per_1k_completion"]
    )


def bert_score_bar(score: float, color: str) -> str:
    """Return inline HTML progress bar for BERTScore."""
    pct = score * 100
    return f"""
    <div style="display:flex;align-items:center;gap:0.5rem;margin-top:0.25rem;">
        <div style="flex:1;height:4px;background:#1e2130;border-radius:4px;overflow:hidden;">
            <div style="width:{pct:.1f}%;height:100%;background:{color};border-radius:4px;"></div>
        </div>
        <span style="font-family:var(--mono);font-size:0.7rem;color:{color};min-width:40px;">{pct:.1f}%</span>
    </div>
    """


# ─── Render a single pipeline's result ────────────────────────────────────────
def render_pipeline_result(key: str, result: dict, response_time: float):
    meta = PIPELINE_META[key]
    cls  = meta["cls"]

    # ── Extract / derive metrics ──────────────────────────────────────────────
    prompt_tokens     = result.get("prompt_tokens", 0)
    completion_tokens = result.get("completion_tokens", 0)
    total_tokens      = prompt_tokens + completion_tokens
    cost              = calc_cost(meta, prompt_tokens, completion_tokens)

    # Accuracy signals
    judge_verdict = result.get("llm_judge_verdict", None)   # "PASS" | "FAIL" | None
    bert_score    = result.get("bert_score", None)          # float 0-1 | None

    # ── Card HTML ─────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="pipeline-result {cls}">
        <h3>
            <span class="badge {meta['badge']}">{meta['label']}</span>
            <span style="color:var(--muted);font-size:0.85rem;font-weight:400;">{meta['full']}</span>
        </h3>

        <div class="metric-grid">
            <div class="metric-cell">
                <div class="label">Latency</div>
                <div class="value">{fmt_time(response_time)}</div>
                <div class="sub">end-to-end</div>
            </div>
            <div class="metric-cell">
                <div class="label">Prompt Tokens</div>
                <div class="value">{prompt_tokens:,}</div>
                <div class="sub">input</div>
            </div>
            <div class="metric-cell">
                <div class="label">Completion Tokens</div>
                <div class="value">{completion_tokens:,}</div>
                <div class="sub">output · {total_tokens:,} total</div>
            </div>
            <div class="metric-cell">
                <div class="label">Cost / Query</div>
                <div class="value">{fmt_cost(cost)}</div>
                <div class="sub">USD estimate</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # ── Accuracy row ──────────────────────────────────────────────────────────
    accuracy_html = '<div class="accuracy-row">'
    accuracy_html += '<span style="font-family:var(--mono);font-size:0.65rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.1em;">Accuracy →</span>'

    if judge_verdict is not None:
        chip_cls = "judge-pass" if judge_verdict == "PASS" else "judge-fail"
        icon     = "✓" if judge_verdict == "PASS" else "✗"
        accuracy_html += f'<span class="judge-chip {chip_cls}">{icon} LLM Judge: {judge_verdict}</span>'
    else:
        accuracy_html += '<span class="judge-chip" style="opacity:0.4;">LLM Judge: —</span>'

    if bert_score is not None:
        accuracy_html += f'<span class="bert-chip">BERTScore {bert_score:.3f}</span>'
        accuracy_html += bert_score_bar(bert_score, meta["color"])
    else:
        accuracy_html += '<span class="bert-chip" style="opacity:0.4;">BERTScore —</span>'

    accuracy_html += '</div>'
    st.markdown(accuracy_html, unsafe_allow_html=True)

    # ── Answer ────────────────────────────────────────────────────────────────
    st.markdown(f"""
        <div class="answer-label">Answer</div>
        <div class="answer-box">{result.get('answer', '—')}</div>
    </div>  <!-- close .pipeline-result -->
    """, unsafe_allow_html=True)

    # ── Sources expander ──────────────────────────────────────────────────────
    sources = result.get("sources", [])
    if sources:
        with st.expander(f"📚 Retrieved sources ({len(sources)})"):
            for s in sources:
                st.markdown(f"""
                <div class="source-item">
                    <div class="source-id">Source {s.get('id', '?')}</div>
                    {s.get('content', '')}
                </div>
                """, unsafe_allow_html=True)


# ─── Comparison summary table ──────────────────────────────────────────────────
def render_comparison_table(results: dict):
    st.markdown("""
    <div class="section-head">
        <div class="section-line"></div>
        <h2>Side-by-Side Comparison</h2>
        <div class="section-line"></div>
    </div>
    """, unsafe_allow_html=True)

    rows = []
    for key, (result, rt) in results.items():
        meta  = PIPELINE_META[key]
        pt    = result.get("prompt_tokens", 0)
        ct    = result.get("completion_tokens", 0)
        cost  = calc_cost(meta, pt, ct)
        bert  = result.get("bert_score")
        judge = result.get("llm_judge_verdict")
        rows.append({
            "key":    key,
            "meta":   meta,
            "time":   rt,
            "tokens": pt + ct,
            "cost":   cost,
            "bert":   bert if bert else 0.0,
            "judge":  judge,
        })

    # Find winners
    def winner_idx(key, reverse=False):
        vals = [r[key] for r in rows]
        fn = max if reverse else min
        return rows.index(next(r for r in rows if r[key] == fn(vals)))

    fast_idx  = winner_idx("time")
    cheap_idx = winner_idx("cost")
    tok_idx   = winner_idx("tokens")
    acc_idx   = winner_idx("bert", reverse=True) if any(r["bert"] for r in rows) else None

    def cell(value, i, winner_i, cls):
        tag = f'class="winner {cls}"' if i == winner_i else ""
        crown = " 🏆" if i == winner_i else ""
        return f"<td {tag}>{value}{crown}</td>"

    header = """
    <table class="compare-table">
        <thead><tr>
            <th>Pipeline</th>
            <th>Latency</th>
            <th>Total Tokens</th>
            <th>Cost / Query</th>
            <th>LLM Judge</th>
            <th>BERTScore</th>
        </tr></thead>
        <tbody>
    """

    body = ""
    for i, r in enumerate(rows):
        meta  = r["meta"]
        judge_html = ""
        if r["judge"] == "PASS":
            judge_html = '<span class="judge-chip judge-pass" style="font-size:0.7rem;padding:2px 8px;">✓ PASS</span>'
        elif r["judge"] == "FAIL":
            judge_html = '<span class="judge-chip judge-fail" style="font-size:0.7rem;padding:2px 8px;">✗ FAIL</span>'
        else:
            judge_html = '<span style="color:var(--muted);">—</span>'

        bert_html = f"{r['bert']:.3f}" if r["bert"] else '<span style="color:var(--muted);">—</span>'

        body += f"<tr>"
        body += f'<td><span class="badge {meta["badge"]}">{meta["label"]}</span></td>'
        body += cell(fmt_time(r["time"]),    i, fast_idx,  f"winner-{meta['cls']}")
        body += cell(f'{r["tokens"]:,}',    i, tok_idx,   f"winner-{meta['cls']}")
        body += cell(fmt_cost(r["cost"]),   i, cheap_idx, f"winner-{meta['cls']}")
        body += f"<td>{judge_html}</td>"
        bert_td = cell(bert_html, i, acc_idx, f"winner-{meta['cls']}") if acc_idx is not None else f"<td>{bert_html}</td>"
        body += bert_td
        body += "</tr>"

    st.markdown(header + body + "</tbody></table>", unsafe_allow_html=True)


# ─── Main App ─────────────────────────────────────────────────────────────────
def main():

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="rag-header">
        <h1>⚗️ RAG Benchmark Dashboard</h1>
        <p>
            One query. Three pipelines. Full metrics.&nbsp;&nbsp;·&nbsp;&nbsp;
            <strong>LLM-Only</strong> baseline &nbsp;·&nbsp;
            <strong>Vector RAG</strong> (embedding retrieval) &nbsp;·&nbsp;
            <strong>GraphRAG</strong> (knowledge graph retrieval)
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Pipelines")
        run_p1 = st.checkbox("◆ Pipeline 1 — LLM Only",    value=True)
        run_p2 = st.checkbox("● Pipeline 2 — Vector RAG",   value=True)
        run_p3 = st.checkbox("▲ Pipeline 3 — GraphRAG",     value=False)

        st.markdown("---")
        st.markdown("### 📝 Example Questions")
        selected = st.selectbox("", ["Custom…"] + EXAMPLE_QUESTIONS, label_visibility="collapsed")

        st.markdown("---")
        st.markdown("""
        <p style="font-size:0.75rem;color:var(--muted);font-family:var(--mono);line-height:1.8;">
        <strong style="color:var(--text);">Metrics explained</strong><br>
        <span style="color:var(--p1);">◆</span> Tokens → prompt + completion<br>
        <span style="color:var(--p2);">●</span> Cost → tokens × model price<br>
        <span style="color:var(--p3);">▲</span> LLM Judge → PASS / FAIL grade<br>
        &nbsp;&nbsp;&nbsp;BERTScore → semantic similarity
        </p>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <p style="font-size:0.72rem;color:var(--muted);font-family:var(--mono);line-height:1.8;">
        <strong style="color:var(--text);">Setup</strong><br>
        1. Set GOOGLE_API_KEY in .env<br>
        2. Run ingest_data.py for vector DB<br>
        3. TigerGraph for Pipeline 3
        </p>
        """, unsafe_allow_html=True)

    # ── Query Input ───────────────────────────────────────────────────────────
    default_q = "" if selected == "Custom…" else selected
    user_q = st.text_area(
        "Query",
        value=default_q,
        height=90,
        placeholder="Enter your question and run all selected pipelines at once…",
        label_visibility="collapsed",
    )

    col_run, col_all, col_space = st.columns([1.2, 1, 5])
    with col_run:
        run_btn = st.button("▶ Run Selected", type="primary", use_container_width=True)
    with col_all:
        all_btn = st.button("⊞ Compare All", use_container_width=True)

    # ── Execute ───────────────────────────────────────────────────────────────
    if run_btn or all_btn:
        q = user_q.strip()

        if not q:
            st.error("⚠ Please enter a question first.")
            return

        if all_btn:
            run_p1 = run_p2 = run_p3 = True

        if not (run_p1 or run_p2 or run_p3):
            st.warning("Select at least one pipeline in the sidebar.")
            return

        # Status banner
        ts = datetime.now().strftime("%H:%M:%S")
        pipelines_running = sum([run_p1, run_p2, run_p3])
        st.markdown(f"""
        <div class="status-bar">
            <div class="status-dot"></div>
            Running {pipelines_running} pipeline{"s" if pipelines_running > 1 else ""} &nbsp;·&nbsp;
            Query received at {ts} &nbsp;·&nbsp;
            <em style="color:var(--muted);">"{q[:80]}{'…' if len(q)>80 else ''}"</em>
        </div>
        """, unsafe_allow_html=True)

        pipeline_map = [
            ("pipeline_1", run_p1, get_pipeline_1_metadata, "Pipeline 1 — LLM Only"),
            ("pipeline_2", run_p2, get_pipeline_2_metadata, "Pipeline 2 — Vector RAG"),
            ("pipeline_3", run_p3, get_pipeline_3_metadata, "Pipeline 3 — GraphRAG"),
        ]

        results = {}
        for key, should_run, fn, label in pipeline_map:
            if not should_run:
                continue
            with st.spinner(f"Running {label}…"):
                t0 = time.time()
                try:
                    result = fn(q)
                    results[key] = (result, time.time() - t0)
                except Exception as e:
                    st.error(f"**{label}** error: {e}")

        # ── Results section ───────────────────────────────────────────────────
        if results:
            st.markdown("""
            <div class="section-head">
                <div class="section-line"></div>
                <h2>Pipeline Outputs</h2>
                <div class="section-line"></div>
            </div>
            """, unsafe_allow_html=True)

            for key, (result, rt) in results.items():
                render_pipeline_result(key, result, rt)

        # ── Comparison table ──────────────────────────────────────────────────
        if len(results) > 1:
            render_comparison_table(results)


if __name__ == "__main__":
    main()