# EcoTrace — Automated Greenwashing Detection Pipeline

Detect logical contradictions between corporate marketing claims and technical environmental evidence using a 4-stage NLP pipeline.

```
[Claim] → [Retrieval: all-MiniLM-L6-v2]
        → [Verification: ClimateBERT NLI → SUPPORT/NEUTRAL/REFUTE]
        → [Scoring: S_gw = w1·P(Refute) + w2·(1−P(Support))]
        → [Explainability: LLM → Reasoning paragraph]
        → [Output: Verdict + Risk Score + Reasoning]
```

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Configure API keys
cp .env.example .env
# Edit .env with your OPENAI_API_KEY or ANTHROPIC_API_KEY

# 3. Download and preprocess CLIMATE-FEVER
python scripts/download_data.py

# 4. Place ESG report PDFs in data/esg_reports/ then index them
python scripts/index_esg_reports.py --pdf_dir data/esg_reports/

# 5. Fine-tune the cross-encoder
python scripts/train_cross_encoder.py --epochs 5

# 6. Run the demo
python demo.py
```

## Repository Structure

```
ecotrace/
├── src/
│   ├── data/           # preprocess, pdf_extractor, augmentation
│   ├── retrieval/      # bi_encoder (all-MiniLM-L6-v2)
│   ├── verification/   # cross_encoder (ClimateBERT), scorer
│   ├── explainability/ # LLM explainer (OpenAI/Anthropic/Ollama)
│   ├── evaluation/     # metrics (P/R/F1, Recall@K)
│   └── pipeline.py     # EcoTracePipeline end-to-end
├── scripts/            # download_data, train, index, evaluate
├── notebooks/demo.ipynb
├── demo.py
├── data/gold_standard/gold_standard.json   # 50 annotated triples
└── report/ecotrace_report_outline.md
```

## Greenwashing Risk Score

$$S_{gw} = w_1 \cdot P(\text{Refute}) + w_2 \cdot (1 - P(\text{Support}))$$

- Default weights: w1=0.7, w2=0.3 (constraint: w1+w2=1)
- `S_gw < 0.3` → **LOW RISK**
- `0.3 ≤ S_gw < 0.6` → **MODERATE RISK**
- `S_gw ≥ 0.6` → **HIGH RISK**

## Run Tests

```bash
pytest tests/ -v
```

## LLM Providers

Set `DEFAULT_LLM_PROVIDER` in `.env`:
- `openai` — requires `OPENAI_API_KEY`
- `anthropic` — requires `ANTHROPIC_API_KEY`
- `local` — requires Ollama running at `OLLAMA_BASE_URL`
