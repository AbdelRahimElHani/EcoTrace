"""EcoTrace Gradio web interface for greenwashing detection."""

import json
import shutil
import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).parent))

from src.data.pdf_extractor import process_esg_report
from src.pipeline import EcoTracePipeline
from src.retrieval.bi_encoder import BiEncoderRetriever

INDEX_PATH    = "models/bi_encoder_index"
MODEL_PATH    = "models/cross_encoder"
ESG_DIR       = "data/esg_reports"
SENTENCES_DIR = "data/processed/esg_sentences"

_pipeline: EcoTracePipeline | None = None


def get_pipeline() -> EcoTracePipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = EcoTracePipeline(
            index_path=INDEX_PATH,
            verifier_model=MODEL_PATH,
        )
    return _pipeline


# ── HTML helpers ─────────────────────────────────────────────────────────────

def _verdict_badge(verdict: str, score: float) -> str:
    cfg = {
        "HIGH RISK":     ("#dc2626", "#fee2e2", "🔴"),
        "MODERATE RISK": ("#d97706", "#fef3c7", "🟡"),
        "LOW RISK":      ("#16a34a", "#dcfce7", "🟢"),
    }
    fg, bg, icon = cfg.get(verdict, ("#6b7280", "#f3f4f6", "⚪"))
    return f"""
    <div style="display:flex;gap:16px;align-items:center;flex-wrap:wrap;margin:12px 0;">
      <div style="background:{bg};color:{fg};border:2px solid {fg};border-radius:10px;
                  padding:14px 28px;font-size:1.5rem;font-weight:800;letter-spacing:.05em;">
        {icon}&nbsp;{verdict}
      </div>
      <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;
                  padding:14px 24px;font-size:1.1rem;color:#334155;">
        Risk Score:&nbsp;<b style="font-size:1.4rem;">{score:.2f}</b>&nbsp;/ 1.00
      </div>
    </div>"""


def _evidence_cards(retrieved: list) -> str:
    if not retrieved:
        return "<p style='color:#94a3b8;font-style:italic;'>No evidence retrieved.</p>"
    html = "<h4 style='margin:16px 0 10px;color:#334155;'>Retrieved Evidence</h4>"
    for item in retrieved:
        pct       = int(item["score"] * 100)
        bar_color = "#22c55e" if pct >= 70 else "#f59e0b" if pct >= 50 else "#94a3b8"
        html += f"""
        <div style="border:1px solid #e2e8f0;border-radius:10px;padding:14px 16px;
                    margin-bottom:10px;background:#ffffff;">
          <div style="display:flex;justify-content:space-between;align-items:center;
                      margin-bottom:6px;">
            <span style="font-weight:700;color:#475569;font-size:.88rem;">
              Evidence [{item['rank']}]
            </span>
            <span style="background:#e0f2fe;color:#0369a1;border-radius:6px;
                         padding:2px 10px;font-size:.82rem;font-weight:600;">
              {pct}% match
            </span>
          </div>
          <div style="background:#e2e8f0;border-radius:4px;height:4px;margin-bottom:10px;">
            <div style="background:{bar_color};width:{pct}%;border-radius:4px;height:4px;">
            </div>
          </div>
          <p style="margin:0;color:#1e293b;line-height:1.6;font-size:.94rem;">
            "{item['sentence']}"
          </p>
        </div>"""
    return html


def _nli_bars(nli_results: list) -> str:
    if not nli_results:
        return ""
    probs   = nli_results[0]["probabilities"]
    support = probs.get("SUPPORT", 0)
    neutral = probs.get("NEUTRAL", 0)
    refute  = probs.get("REFUTE",  0)

    def bar(label, value, color):
        pct = round(value * 100)
        return f"""
        <div style="margin-bottom:10px;">
          <div style="display:flex;justify-content:space-between;
                      font-size:.9rem;margin-bottom:4px;">
            <span style="font-weight:700;color:{color};">{label}</span>
            <span style="color:#475569;">{pct}%</span>
          </div>
          <div style="background:#e2e8f0;border-radius:6px;height:12px;">
            <div style="background:{color};width:{pct}%;border-radius:6px;height:12px;">
            </div>
          </div>
        </div>"""

    return (
        "<h4 style='margin:16px 0 12px;color:#334155;'>"
        "NLI Probability Breakdown (top evidence)</h4>"
        + bar("✅ SUPPORT", support, "#16a34a")
        + bar("➖ NEUTRAL", neutral, "#d97706")
        + bar("❌ REFUTE",  refute,  "#dc2626")
    )


def _company_list_html() -> str:
    d = Path(SENTENCES_DIR)
    if not d.exists():
        return "<p style='color:#94a3b8'>No companies indexed yet.</p>"
    items = []
    for f in sorted(d.glob("*.json")):
        with open(f, encoding="utf-8") as jf:
            data = json.load(jf)
        n = len(data.get("sentences", []))
        items.append(
            f"<li style='margin-bottom:8px;padding:8px 12px;background:#f8fafc;"
            f"border-radius:8px;border:1px solid #e2e8f0;'>"
            f"<b style='color:#0f172a;'>{f.stem}</b>"
            f"<span style='color:#64748b;margin-left:8px;font-size:.88rem;'>"
            f"{n} environmental sentences</span></li>"
        )
    if not items:
        return "<p style='color:#94a3b8'>No companies indexed yet.</p>"
    return (
        "<ul style='margin:0;padding:0;list-style:none;'>"
        + "".join(items)
        + "</ul>"
    )


# ── Core logic ────────────────────────────────────────────────────────────────

def analyze(claim: str):
    if not claim.strip():
        err = "<p style='color:#dc2626;font-weight:600;'>⚠️ Please enter a claim.</p>"
        return err, "", ""

    pipe   = get_pipeline()
    result = pipe.analyze(claim)

    return (
        _verdict_badge(result["verdict"], result["greenwashing_score"]),
        _evidence_cards(result["retrieved_evidence"]),
        _nli_bars(result["nli_results"]),
    )


def index_pdf(pdf_file, progress=gr.Progress(track_tqdm=True)):
    if pdf_file is None:
        return (
            "<p style='color:#d97706;font-weight:600;'>⚠️ No file selected.</p>",
            _company_list_html(),
        )

    pdf_path = Path(pdf_file.name)
    company  = pdf_path.stem

    progress(0.05, desc=f"Copying {pdf_path.name}…")
    dest = Path(ESG_DIR) / pdf_path.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(pdf_file.name, str(dest))

    progress(0.25, desc="Extracting text and filtering environmental sentences…")
    out_json = Path(SENTENCES_DIR) / f"{company}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    process_esg_report(str(dest), str(out_json))

    progress(0.55, desc="Rebuilding search index (this may take a moment)…")
    all_sentences, seen = [], set()
    for jf in sorted(Path(SENTENCES_DIR).glob("*.json")):
        with open(jf, encoding="utf-8") as f:
            data = json.load(f)
        for s in data.get("sentences", []):
            if s not in seen:
                seen.add(s)
                all_sentences.append(s)

    retriever = BiEncoderRetriever()
    retriever.build_index(all_sentences)
    retriever.save_index(INDEX_PATH)

    progress(0.92, desc="Reloading pipeline retriever…")
    pipe = get_pipeline()
    pipe.retriever.load_index(INDEX_PATH)

    with open(out_json, encoding="utf-8") as f:
        n = len(json.load(f).get("sentences", []))

    progress(1.0, desc="Done!")
    status = (
        f"<div style='background:#dcfce7;border:1px solid #16a34a;border-radius:10px;"
        f"padding:14px 18px;color:#15803d;font-weight:600;'>"
        f"✅ <b>{company}</b> indexed successfully!<br>"
        f"<span style='font-weight:400;'>Added <b>{n}</b> environmental sentences. "
        f"Total index: <b>{len(all_sentences)}</b> sentences.</span></div>"
    )
    return status, _company_list_html()


# ── UI layout ─────────────────────────────────────────────────────────────────

EXAMPLES = [
    "We have achieved carbon neutrality across all our operations.",
    "Our packaging uses 100% recycled materials.",
    "We will be fully electric by 2030.",
    "We have zero methane emissions from our operations.",
    "We invest more in clean energy than any other oil company.",
    "Our supply chain is 100% deforestation-free.",
    "We recycle 95% of our operational waste.",
    "All our tier-1 suppliers meet our environmental standards.",
]

CSS = """
#header  { text-align:center; padding:24px 0 8px; }
#header h1 { font-size:2.4rem; font-weight:900; color:#0f172a; margin:0; }
#header p  { color:#64748b; font-size:1.05rem; margin:6px 0 0; }
footer { display:none !important; }
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="EcoTrace") as demo:

        gr.HTML("""
        <div id="header">
          <h1>🌿 EcoTrace</h1>
          <p>Automated Greenwashing Detection — ClimateBERT NLI · Dense Retrieval · Risk Scoring</p>
        </div>
        """)

        with gr.Tabs():

            # ── Tab 1: Analyze ────────────────────────────────────────────
            with gr.Tab("🔍  Analyze a Claim"):
                with gr.Row(equal_height=False):

                    with gr.Column(scale=2):
                        claim_box = gr.Textbox(
                            label="Corporate ESG Claim",
                            placeholder=(
                                'Paste or type any environmental claim, e.g.\n'
                                '"We have achieved carbon neutrality across all our operations."'
                            ),
                            lines=4,
                        )
                        with gr.Row():
                            analyze_btn = gr.Button(
                                "⚡ Analyze", variant="primary", scale=3
                            )
                            clear_btn = gr.Button(
                                "🗑 Clear", variant="secondary", scale=1
                            )

                        gr.Examples(
                            examples=[[e] for e in EXAMPLES],
                            inputs=[claim_box],
                            label="Quick examples — click any to load",
                            examples_per_page=8,
                        )

                    with gr.Column(scale=3):
                        verdict_out  = gr.HTML(label="Verdict")
                        evidence_out = gr.HTML(label="Evidence")
                        nli_out      = gr.HTML(label="NLI Breakdown")

                analyze_btn.click(
                    fn=analyze,
                    inputs=[claim_box],
                    outputs=[verdict_out, evidence_out, nli_out],
                )
                clear_btn.click(
                    fn=lambda: ("", "", "", ""),
                    outputs=[claim_box, verdict_out, evidence_out, nli_out],
                )

            # ── Tab 2: Add Company PDF ────────────────────────────────────
            with gr.Tab("📄  Add a Company PDF"):
                gr.Markdown("""
Upload any company's ESG or sustainability report PDF.
EcoTrace will extract environmental sentences from it, embed them into the search index,
and make them available immediately for claim analysis.

**Supported companies so far:** Amazon · Apple · ExxonMobil · Shell · Volkswagen
                """)

                with gr.Row(equal_height=False):
                    with gr.Column():
                        pdf_input = gr.File(
                            label="Upload ESG Report PDF",
                            file_types=[".pdf"],
                            file_count="single",
                        )
                        index_btn = gr.Button("📥 Index PDF", variant="primary")
                        gr.Markdown(
                            "_Indexing typically takes 30–90 seconds depending on PDF size._",
                        )
                        index_status = gr.HTML()

                    with gr.Column():
                        gr.Markdown("### 📋 Currently Indexed Companies")
                        company_list_out = gr.HTML(value=_company_list_html())

                index_btn.click(
                    fn=index_pdf,
                    inputs=[pdf_input],
                    outputs=[index_status, company_list_out],
                )

    return demo


if __name__ == "__main__":
    print("Loading EcoTrace pipeline — please wait (~10 seconds)...")
    get_pipeline()
    print("Pipeline ready. Launching browser...")
    build_ui().launch(inbrowser=True, theme=gr.themes.Soft(), css=CSS)
