# 🌿 EcoTrace — Automated Greenwashing Detection

EcoTrace is an end-to-end NLP pipeline that fact-checks corporate ESG claims against a company's **own** sustainability report data. It combines dense retrieval, domain-adapted NLI, and a principled risk score to flag misleading environmental statements.

---

## How It Works

```
You provide a claim:
  "We have achieved carbon neutrality across all our operations."

EcoTrace searches that company's own ESG report for relevant evidence:
  "Scope 1 & 2 emissions increased 3% YoY, totalling 45M tonnes CO₂."

NLI model compares claim vs. evidence:
  REFUTE → S_gw = 0.93 → 🔴 HIGH RISK
```

The pipeline has four stages:

| Stage | Model | Output |
|-------|-------|--------|
| Retrieval | `all-MiniLM-L6-v2` (bi-encoder) | Top-K relevant sentences from the ESG corpus |
| Verification | `climatebert/distilroberta-base-climate-f` (fine-tuned) | P(SUPPORT), P(NEUTRAL), P(REFUTE) |
| Scoring | S_gw = w₁·P(Refute) + w₂·(1−P(Support)) | Risk score 0–1 |
| Explanation | LLM (OpenAI / Anthropic / Ollama) | 3–4 sentence natural-language justification |

---

## Results

Evaluated on a manually annotated gold standard of **50 claim–evidence pairs** from 5 major ESG reports:

| Model | Accuracy | Macro-F1 | Evidence |
|-------|----------|----------|----------|
| TF-IDF + Logistic Regression | 28% | 25.2% | Oracle (gold) |
| Zero-shot RoBERTa NLI | 40% | 34.1% | Oracle (gold) |
| **EcoTrace end-to-end** | **28%** | **25.1%** | **Retrieved** |
| ClimateBERT NLI (component only) | — | **98.2%** | Oracle (val) |

> EcoTrace matches TF-IDF with oracle evidence while retrieving its own evidence — a harder task. The NLI component alone reaches 98.2% macro-F1, confirming that retrieval quality is the primary bottleneck.

---

## Quickstart

### Option A — Web Interface (recommended)

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python app.py
```

Your browser opens at `http://localhost:7860`. Select a company, type a claim, click **Analyze**.

### Option B — Command Line

```bash
# Analyze a single custom claim
python demo.py \
  --index_path models/bi_encoder_index \
  --verifier_model models/cross_encoder \
  --claim "We have achieved carbon neutrality across all our operations."

# Run the 3 built-in demo cases
python demo.py \
  --index_path models/bi_encoder_index \
  --verifier_model models/cross_encoder
```

---

## Full Setup (train from scratch)

```bash
# 1. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. (Optional) Configure LLM for explanations
cp .env.example .env
# Add OPENAI_API_KEY or ANTHROPIC_API_KEY — pipeline works without it

# 3. Download and preprocess CLIMATE-FEVER (auto-downloads from HuggingFace)
python -m scripts.download_data

# 4. Place ESG report PDFs in data/esg_reports/ then build the index
python -m scripts.index_esg_reports --pdf_dir data/esg_reports/

# 5. Fine-tune the cross-encoder NLI model (~8 hours on CPU, much faster on GPU)
python -m scripts.train_cross_encoder --epochs 5

# 6. Evaluate against the gold standard
python -m scripts.evaluate \
  --index_path models/bi_encoder_index \
  --verifier_model models/cross_encoder

# 7. Run baselines for comparison
python -m scripts.run_baselines
```

---

## Adding a New Company

Drop the company's sustainability report PDF into `data/esg_reports/` and re-index:

```bash
python -m scripts.index_esg_reports --pdf_dir data/esg_reports/
```

Or use the **"Add a Company PDF"** tab in the web interface — no command line needed.

---

## Repository Structure

```
ecotrace/
├── app.py                          # Gradio web interface
├── demo.py                         # CLI demo (3 canonical cases + custom claim)
├── src/
│   ├── pipeline.py                 # EcoTracePipeline (4-stage orchestrator)
│   ├── data/
│   │   ├── preprocess.py           # CLIMATE-FEVER loading & stratified splits
│   │   ├── pdf_extractor.py        # PyMuPDF + spaCy sentence extraction
│   │   └── augmentation.py         # Back-translation (EN→DE→EN) augmentation
│   ├── retrieval/
│   │   └── bi_encoder.py           # Dense retriever (all-MiniLM-L6-v2)
│   ├── verification/
│   │   ├── cross_encoder.py        # ClimateBERT NLI classifier
│   │   └── scorer.py               # Greenwashing risk score S_gw
│   ├── explainability/
│   │   └── explainer.py            # LLM explanation (OpenAI/Anthropic/Ollama)
│   └── evaluation/
│       └── metrics.py              # Macro-F1, Recall@K
├── scripts/
│   ├── download_data.py            # Download CLIMATE-FEVER, create splits
│   ├── train_cross_encoder.py      # Fine-tune ClimateBERT on NLI splits
│   ├── index_esg_reports.py        # Extract sentences & build retriever index
│   ├── evaluate.py                 # Full pipeline evaluation on gold standard
│   └── run_baselines.py            # TF-IDF+LR and zero-shot RoBERTa baselines
├── data/
│   ├── gold_standard/
│   │   └── gold_standard.json      # 50 annotated claim–evidence triples
│   ├── processed/
│   │   ├── train.json              # CLIMATE-FEVER train split (6,141 pairs)
│   │   ├── val.json                # Val split (767 pairs)
│   │   ├── test.json               # Test split (767 pairs)
│   │   └── esg_sentences/          # Per-company extracted sentence JSONs
│   └── esg_reports/                # Place ESG report PDFs here
├── models/                         # Saved model checkpoints (gitignored)
├── results/
│   ├── baseline_results.json       # TF-IDF + RoBERTa results
│   ├── evaluation_report.json      # EcoTrace end-to-end results
│   └── training_report.json        # ClimateBERT fine-tuning metrics
├── tests/                          # Unit tests (pytest)
├── report/
│   └── ecotrace_report.md          # Full project report
├── requirements.txt
└── .env.example
```

---

## Greenwashing Risk Score

$$S_{gw} = w_1 \cdot P(\text{Refute}) + w_2 \cdot (1 - P(\text{Support}))$$

| Score | Verdict |
|-------|---------|
| S_gw < 0.3 | 🟢 LOW RISK |
| 0.3 ≤ S_gw < 0.6 | 🟡 MODERATE RISK |
| S_gw ≥ 0.6 | 🔴 HIGH RISK |

Default weights: **w₁ = 0.7** (refutation), **w₂ = 0.3** (non-support).

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Dense retrieval | `sentence-transformers/all-MiniLM-L6-v2` |
| NLI verification | `climatebert/distilroberta-base-climate-f` (fine-tuned) |
| PDF extraction | PyMuPDF + spaCy sentencizer |
| Training framework | HuggingFace Transformers + Trainer |
| Web interface | Gradio |
| Baselines | scikit-learn TF-IDF + `roberta-large-mnli` |

---

## Run Tests

```bash
pytest tests/ -v
```

---

## LLM Explanation Providers

Configure `DEFAULT_LLM_PROVIDER` in `.env`. The pipeline works fully without an API key — explanations gracefully fall back to "Explanation unavailable."

| Provider | Env Var | Model |
|----------|---------|-------|
| OpenAI | `OPENAI_API_KEY` | `gpt-4o-mini` |
| Anthropic | `ANTHROPIC_API_KEY` | `claude-3-haiku` |
| Local (Ollama) | `OLLAMA_BASE_URL` | any Ollama model |
