"""
app.py — Medical RAG Assistant
Gradio 5 compatible, CPU Basic, HuggingFace Spaces
Fix: show_api=False + api_name=False on all events stops the
     json_schema_to_python_type crash on gr.Chatbot type="messages"
"""

import time
import gradio as gr

from rag_pipeline import MedicalRAGPipeline, RetrievedChunk
from evaluation   import evaluate, run_batch_evaluation, format_batch_results

# ── Init pipeline ─────────────────────────────────────────────
print("🚀 Initialising Medical RAG pipeline...")
pipeline = MedicalRAGPipeline()
print("✅ Pipeline ready")

# ── Examples ──────────────────────────────────────────────────
EXAMPLES = [
    "What are the symptoms and signs of heart failure?",
    "How is blood pressure measured and what is hypertension?",
    "What does a normal lung examination sound like?",
    "How do you examine the abdomen for liver enlargement?",
    "What are the signs of depression in a patient interview?",
    "What is the difference between systolic and diastolic murmurs?",
    "How do you perform a neurological examination?",
    "What causes wheezing and how is it detected during examination?",
]

SYS_COLORS = {
    "cardiovascular":"#e74c3c","respiratory":"#3498db","neurological":"#9b59b6",
    "gastrointestinal":"#27ae60","musculoskeletal":"#f39c12","dermatological":"#e91e63",
    "endocrine":"#16a085","psychiatric":"#c0392b","general":"#607d8b",
}

# ── Helpers ───────────────────────────────────────────────────
def _sources_html(chunks, timings):
    if not chunks:
        return "<p style='color:#999;padding:8px'>No sources retrieved.</p>"
    html = ""
    for i, c in enumerate(chunks):
        m     = c.metadata
        sys   = m.get("body_system","general")
        color = SYS_COLORS.get(sys,"#607d8b")
        html += f"""
        <div style="border-left:3px solid {color};padding:10px 14px;margin-bottom:10px;
                    background:#1e2530;border-radius:0 6px 6px 0;">
            <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
                <b style="color:#e8eaf6;font-size:0.83rem;">[{i+1}] {m.get('chapter_name','')[:45]}</b>
                <span style="background:{color}33;color:{color};border:1px solid {color}66;
                             border-radius:12px;padding:1px 8px;font-size:0.70rem;">{sys}</span>
            </div>
            <div style="color:#90caf9;font-size:0.76rem;margin-bottom:5px;">
                {m.get('section_name','')[:55]} · p.{m.get('page_start','?')}
            </div>
            <div style="color:#b0bec5;font-size:0.79rem;line-height:1.5;">
                {c.text[:260]}{"..." if len(c.text)>260 else ""}
            </div>
            <div style="margin-top:5px;color:#546e7a;font-size:0.71rem;">
                Relevance: <b style="color:#80cbc4">{c.final_score:.3f}</b> &nbsp;
                CE: <b style="color:#80cbc4">{c.ce_score:.2f}</b> &nbsp;
                Type: <b style="color:#80cbc4">{m.get('chunk_type','text')}</b>
            </div>
        </div>"""
    timing_str = " · ".join(
        f"{k.replace('_ms','')}:{v}ms" for k,v in timings.items() if 'ms' in k
    )
    return (f"<div style='font-family:monospace;font-size:0.70rem;color:#6e7681;"
            f"background:#0d1117;border:1px solid #30363d;border-radius:6px;"
            f"padding:6px 10px;margin-bottom:10px;'>⏱ {timing_str}</div>{html}")


def _metrics_html(r):
    def bar(val, color):
        return (f'<div style="height:7px;background:#1e2530;border-radius:3px;overflow:hidden;">'
                f'<div style="height:100%;width:{int(val*100)}%;background:{color};">'
                f'</div></div>')
    rows = ""
    for label, val, color in [
        ("Answer Relevancy",  r.answer_relevancy,      "#42a5f5"),
        ("Faithfulness",      r.faithfulness,           "#66bb6a"),
        ("Context Precision", r.context_precision,      "#ffa726"),
        ("Context Recall",    r.context_recall,         "#ab47bc"),
        ("Diversity",         r.retrieval_diversity,    "#26c6da"),
        ("Completeness",      r.response_completeness,  "#ec407a"),
    ]:
        rows += (f'<div style="margin-bottom:9px;">'
                 f'<div style="display:flex;justify-content:space-between;margin-bottom:2px;">'
                 f'<span style="color:#b0bec5;font-size:0.77rem;">{label}</span>'
                 f'<span style="color:{color};font-weight:700;font-size:0.77rem;">{val:.2f}</span>'
                 f'</div>{bar(val,color)}</div>')
    oc = "#66bb6a" if r.overall_score>=0.6 else "#ffa726" if r.overall_score>=0.4 else "#ef5350"
    return (f'<div style="text-align:center;padding:10px;margin-bottom:12px;'
            f'background:linear-gradient(135deg,#1e2530,#263043);'
            f'border-radius:8px;border:1px solid #30363d;">'
            f'<div style="color:#6e7681;font-size:0.72rem;">OVERALL SCORE</div>'
            f'<div style="color:{oc};font-size:1.9rem;font-weight:800;">{r.overall_score:.2f}</div>'
            f'<div style="color:#6e7681;font-size:0.70rem;">/ 1.00</div></div>'
            f'{rows}'
            f'<div style="margin-top:8px;padding:6px;background:#0d1117;border-radius:5px;'
            f'font-size:0.69rem;color:#546e7a;text-align:center;">'
            f'{r.details.get("word_count",0)} words · {r.details.get("context_count",0)} chunks</div>')


# ── Chat ──────────────────────────────────────────────────────
def chat(message, history):
    if not message.strip():
        yield history, "", "", ""
        return

    # Convert Gradio 5 messages → pipeline tuples
    history_tuples = []
    msgs = history or []
    i = 0
    while i < len(msgs) - 1:
        if msgs[i]["role"] == "user" and msgs[i+1]["role"] == "assistant":
            history_tuples.append((msgs[i]["content"], msgs[i+1]["content"]))
            i += 2
        else:
            i += 1

    history = list(msgs)
    history.append({"role": "user",      "content": message})
    history.append({"role": "assistant", "content": "⏳ Thinking... (1–2 min on CPU)"})
    yield history, "", "", ""

    answer, chunks, timings = pipeline.run(message, history_tuples)
    ctx_texts    = [c.text for c in chunks]
    eval_result  = evaluate(message, answer, ctx_texts)

    # Prepend confidence warning if pipeline flagged low confidence
    display_answer = answer
    if timings.get("low_confidence"):
        display_answer = (
            "⚠️ **Low Confidence** — the retrieved context may not fully cover this question. "
            "Answer may be incomplete or loosely grounded. Please verify with a healthcare professional.\n\n"
            + answer
        )

    history[-1] = {"role": "assistant", "content": display_answer}
    yield history, "", _sources_html(chunks, timings), _metrics_html(eval_result)


def run_evaluation():
    return format_batch_results(run_batch_evaluation(pipeline))


# ── CSS ───────────────────────────────────────────────────────
CSS = """
body, .gradio-container {
    background: #0d1117 !important;
    font-family: 'Segoe UI', sans-serif !important;
}
textarea, input[type=text] {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    border-radius: 6px !important;
    font-size: 0.87rem !important;
}
textarea:focus, input[type=text]:focus {
    border-color: #58a6ff !important;
    outline: none !important;
}
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
.queue-status, .eta-bar, [class*="queue"], [class*="eta"] {
    display: none !important;
}
"""

# ── Build UI ──────────────────────────────────────────────────
with gr.Blocks(css=CSS, theme=gr.themes.Base(), title="Medical RAG Assistant") as demo:

    gr.Markdown("""
# 🏥 Medical RAG Assistant
**Grounded in** *Bates' Guide to Physical Examination and History Taking*

`BAAI/bge-large-en-v1.5` · `Qwen2.5-1.5B-Instruct` · `ChromaDB` · `Hybrid Search` · `Cross-Encoder` · `MMR`
---
""")

    with gr.Tabs():

        # ── Tab 1: Chat ───────────────────────────────────────
        with gr.Tab("💬 Chat"):
            with gr.Row():

                with gr.Column(scale=6):
                    chatbot = gr.Chatbot(
                        value      = [],
                        height     = 500,
                        show_label = False,
                        type       = "messages",
                    )
                    with gr.Row():
                        msg_box  = gr.Textbox(
                            placeholder = "Ask about a disease, symptom, or examination...",
                            show_label  = False,
                            scale       = 8,
                            lines       = 1,
                            max_lines   = 4,
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=1, min_width=70)
                    clear_btn = gr.Button("🗑 Clear", variant="secondary", size="sm")
                    gr.Examples(
                        examples = [[e] for e in EXAMPLES],
                        inputs   = [msg_box],
                        label    = "Examples",
                    )

                with gr.Column(scale=4):
                    with gr.Tabs():
                        with gr.Tab("📚 Sources"):
                            sources_box = gr.HTML(
                                "<p style='color:#6e7681;padding:10px;font-size:0.82rem;'>"
                                "Sources appear after your first question.</p>"
                            )
                        with gr.Tab("📊 Metrics"):
                            metrics_box = gr.HTML(
                                "<p style='color:#6e7681;padding:10px;font-size:0.82rem;'>"
                                "Metrics appear after your first question.</p>"
                            )

        # ── Tab 2: Batch Eval ─────────────────────────────────
        with gr.Tab("🧪 Batch Evaluation"):
            gr.Markdown("Run pipeline on 10 standard queries and compute quality metrics.")
            eval_btn    = gr.Button("▶ Run Batch Evaluation", variant="primary")
            eval_output = gr.Markdown("*Click to start — takes 3–8 min on CPU.*")

        # ── Tab 3: About ──────────────────────────────────────
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
### Pipeline Stages
| Stage | Component |
|-------|-----------|
| Query Expansion | Medical synonym dictionary + exam-bias suffix |
| Dense Retrieval | ChromaDB + BAAI/bge-large-en-v1.5 |
| Sparse Retrieval | BM25Okapi |
| Fusion | Reciprocal Rank Fusion (60% dense + 40% BM25) |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Diversification | MMR (λ=0.65) |
| Generation | Qwen2.5-1.5B-Instruct (local CPU) |

### Enterprise Features
| Feature | Detail |
|---------|--------|
| Confidence Warning | Shown when top chunk score < 0.35 or retrieval is sparse |
| Query Audit Log | Every query logged to `/tmp/medical_rag/query_log.jsonl` |
| Source Attribution | Retrieved chunks shown with chapter, section, page, relevance score |
| Similarity Filtering | SIMILARITY_FLOOR=0.35 — only high-quality chunks enter the pipeline |

### Evaluation Metrics
Answer Relevancy · Faithfulness · Context Precision · Context Recall · Diversity · Completeness

---
⚕️ *Educational purposes only. Always consult a qualified healthcare professional.*
""")

    # ── Events — api_name=False prevents schema introspection ─
    send_btn.click(
        fn       = chat,
        inputs   = [msg_box, chatbot],
        outputs  = [chatbot, msg_box, sources_box, metrics_box],
        api_name = False,
    )
    msg_box.submit(
        fn       = chat,
        inputs   = [msg_box, chatbot],
        outputs  = [chatbot, msg_box, sources_box, metrics_box],
        api_name = False,
    )
    clear_btn.click(
        fn       = lambda: ([], "", "", ""),
        outputs  = [chatbot, msg_box, sources_box, metrics_box],
        api_name = False,
    )
    eval_btn.click(
        fn       = run_evaluation,
        outputs  = [eval_output],
        api_name = False,
    )

# ── Launch — show_api=False is the critical fix ───────────────
if __name__ == "__main__":
    demo.queue(
        max_size        = 3,      # max 3 requests queued — beyond this users get a busy message
        default_concurrency_limit = 1,  # CPU can only handle 1 at a time
    )
    demo.launch(
        server_name = "0.0.0.0",
        server_port = 7860,
        show_api    = False,   # ← prevents json_schema_to_python_type crash
    )