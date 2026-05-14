# EcoTrace: Automated Greenwashing Detection Pipeline
## Report Outline (20 pages)

---

### 1. Abstract (0.5 pages)
- Problem: corporations publish ESG claims that contradict technical environmental data
- Approach: 4-stage NLP pipeline (retrieval → NLI verification → risk scoring → LLM explanation)
- Key results: macro-F1, Recall@5, comparison against TF-IDF and zero-shot baselines

---

### 2. Introduction (1.5 pages)
- Greenwashing definition and real-world harm
- ESG reporting context and scale (volume of corporate sustainability claims)
- Gap: no automated, explainable fact-checking system for environmental claims
- Contributions:
  1. End-to-end EcoTrace pipeline
  2. Gold standard of 50 annotated claim–evidence triples
  3. Fine-tuned ClimateBERT NLI model
  4. Explainable greenwashing risk score S_gw

---

### 3. Related Work (2 pages)
- CLIMATE-FEVER (Diggelmann et al., 2020): climate claim fact-checking dataset
- ClimateBERT (Webersinke et al., 2022): domain-adapted BERT for climate text
- SNLI / MNLI: foundational NLI corpora and models
- Sentence-BERT (Reimers & Gurevych, 2019): dense retrieval foundations
- FEVER (Thorne et al., 2018): claim verification benchmark
- RAG (Lewis et al., 2020): retrieval-augmented generation framework
- Prior greenwashing detection work (rule-based, keyword, shallow ML)

---

### 4. Dataset & Data Engineering (2 pages)
- CLIMATE-FEVER statistics: 1,535 claims, label distribution
- Preprocessing pipeline: flatten → stratified split (80/10/10, seed=42)
- ESG Report corpus: ExxonMobil, Shell, Amazon, Apple, Volkswagen PDFs
  - PyMuPDF extraction + spaCy sentence splitting
  - Environmental keyword filtering (9 keywords)
  - Sentence count per company
- Gold standard: 50 triples (10 HIGH / 10 LOW / 10 NEUTRAL minimum)
- Back-translation augmentation: EN→DE→EN on REFUTE class (ratio=1.0)
- Figure: Label distribution bar chart before and after augmentation

---

### 5. System Architecture (3 pages)
- Full pipeline diagram (claim → retrieval → NLI → scoring → explanation)

**Stage 1 — Retrieval**
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Cosine similarity: $\text{sim}(q,d) = \frac{\mathbf{q}\cdot\mathbf{d}}{\|\mathbf{q}\|\|\mathbf{d}\|}$
- Index construction: batch_size=64, L2-normalized embeddings
- Top-K retrieval (default K=5)

**Stage 2 — Verification**
- Model: `climatebert/distilroberta-base-climate-f` (fine-tuned)
- Input: [CLS] claim [SEP] evidence [SEP], max_length=512
- Output: P(SUPPORT), P(NEUTRAL), P(REFUTE)
- Loss: $\mathcal{L} = -\sum_{c \in \{S,N,R\}} y_c \log \hat{p}_c$

**Stage 3 — Scoring**
- $S_{gw} = w_1 \cdot P(\text{Refute}) + w_2 \cdot (1 - P(\text{Support}))$
- Default: w1=0.7, w2=0.3 (w1+w2=1)
- Thresholds: <0.3 LOW | 0.3–0.6 MODERATE | ≥0.6 HIGH

**Stage 4 — Explainability**
- System prompt: expert ESG analyst persona
- 3-4 sentence structure: claim assertion / evidence finding / verdict rationale
- Providers: OpenAI gpt-4o-mini / Anthropic Claude / Ollama (local)

---

### 6. Experiments & Training (2 pages)
- Table: Hyperparameter configuration
- Training curves: loss and macro-F1 vs epoch for ClimateBERT fine-tuning
- EarlyStoppingCallback(patience=2) — epoch stopped at
- w1/w2 ablation: grid over w1 ∈ {0.5, 0.6, 0.7, 0.8, 0.9} on gold standard
- Figure: Ablation heatmap (w1 vs macro-F1)

---

### 7. Results & Analysis (3 pages)
- Table: Main results — EcoTrace vs baselines (TF-IDF+LR, Zero-shot RoBERTa)

| Model | Precision | Recall | Macro-F1 |
|---|---|---|---|
| TF-IDF + Logistic Regression | - | - | - |
| Zero-shot RoBERTa NLI | - | - | - |
| **EcoTrace (ours)** | **-** | **-** | **-** |

- Figure: Confusion matrix (3×3 for SUPPORT/NEUTRAL/REFUTE)
- Figure: Precision/Recall/F1 per class bar chart
- Table: Recall@K for K ∈ {1, 3, 5}
- Figure: Recall@K curve
- Figure: Greenwashing score histogram (gold standard distribution)

---

### 8. Qualitative Analysis (2 pages)
Full EcoTrace output for all three demo cases:

**Case 1 — HIGH RISK:** "We have achieved carbon neutrality across all our operations."
- Retrieved evidence, NLI verdict (REFUTE), S_gw > 0.7, 3-4 sentence explanation

**Case 2 — MODERATE RISK:** "We are committed to reducing emissions by 2050."
- Retrieved evidence, NLI verdict (NEUTRAL), S_gw 0.3–0.6, explanation noting commitment vs action gap

**Case 3 — Failure Mode:** "Our supply chain is 100% deforestation-free."
- Demonstrates vocabulary mismatch: retriever returns AWS energy sentences
- Analysis: deforestation vocabulary absent from ESG index → retrieval failure

---

### 9. Discussion & Limitations (1.5 pages)
- Vocabulary mismatch: domain gap between deforestation/biodiversity and energy corpus
- Hallucination risk: LLM explainer may fabricate plausible-sounding reasoning
- Label subjectivity: NLI for greenwashing is ambiguous (NEUTRAL vs REFUTE edge cases)
- PDF extraction quality: tabular data and headers produce noise
- Scope 3 emissions: indirect emissions rarely mentioned in corporate sentences
- Future work: FAISS indexing for scale, multilingual support, cross-lingual ESG corpora

---

### 10. Conclusion & Future Work (1 page)
- EcoTrace: first end-to-end, explainable greenwashing detection pipeline
- Outperforms both classical and zero-shot baselines on macro-F1
- Demonstrates principled failure mode analysis (Case 3)
- Future: real-time monitoring, regulatory API integration, multi-document fusion

---

### 11. References (1 page)
1. Diggelmann et al. (2020). CLIMATE-FEVER: A Dataset for Verification of Real-World Climate Claims. *NeurIPS 2020 Workshop*.
2. Webersinke et al. (2022). ClimateBERT: A Pre-trained Language Model for Climate-Related Text. *AAAI 2022*.
3. Bowman et al. (2015). A Large Annotated Corpus for Learning Natural Language Inference. *EMNLP 2015*.
4. Reimers & Gurevych (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP 2019*.
5. Thorne et al. (2018). FEVER: A Large-scale Dataset for Fact Extraction and VERification. *NAACL 2018*.
6. Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS 2020*.

---

### Figures Checklist
- [ ] Architecture pipeline diagram
- [ ] Label distribution bar chart (before/after augmentation)
- [ ] Training loss and macro-F1 curves
- [ ] Confusion matrix (3×3)
- [ ] Precision/Recall/F1 per-class bar chart
- [ ] Recall@K curve (K=1,3,5)
- [ ] Greenwashing score histogram
- [ ] w1/w2 ablation heatmap

### Tables Checklist
- [ ] Hyperparameter configuration
- [ ] Main results vs baselines (P/R/F1)
- [ ] Recall@K table
- [ ] w1/w2 ablation results
