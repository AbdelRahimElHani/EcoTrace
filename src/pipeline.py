"""EcoTrace end-to-end greenwashing detection pipeline."""

import logging
import os
import time
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

from src.explainability.explainer import GreenwashingExplainer
from src.retrieval.bi_encoder import BiEncoderRetriever
from src.verification.cross_encoder import GreenwashingVerifier
from src.verification.scorer import aggregate_scores, compute_greenwashing_score, get_verdict

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EcoTracePipeline:
    """Four-stage greenwashing detection pipeline.

    Stages:
        1. Retrieval  — bi-encoder cosine similarity
        2. Verification — cross-encoder NLI (SUPPORT / NEUTRAL / REFUTE)
        3. Scoring    — S_gw = w1*P(Refute) + w2*(1-P(Support))
        4. Explainability — LLM reasoning paragraph
    """

    def __init__(
        self,
        index_path: str | None = None,
        verifier_model: str = GreenwashingVerifier.BASE_MODEL,
        llm_provider: str | None = None,
        llm_model: str | None = None,
        top_k: int | None = None,
        w1: float = 0.7,
        w2: float = 0.3,
    ) -> None:
        """Initialize all pipeline components.

        Args:
            index_path: Path to a pre-built bi-encoder index directory.
            verifier_model: HuggingFace checkpoint for the cross-encoder.
            llm_provider: LLM backend ("openai", "anthropic", "local").
            llm_model: Model name for the LLM explainer.
            top_k: Number of evidence sentences to retrieve per claim.
            w1: Weight on P(Refute) in the greenwashing score.
            w2: Weight on (1 - P(Support)) in the greenwashing score.
        """
        self.top_k = top_k or int(os.environ.get("TOP_K_RETRIEVAL", 5))
        self.w1 = w1
        self.w2 = w2

        self.retriever = BiEncoderRetriever()
        if index_path and Path(index_path).exists():
            self.retriever.load_index(index_path)

        self.verifier = GreenwashingVerifier(verifier_model)

        provider = llm_provider or os.environ.get("DEFAULT_LLM_PROVIDER", "openai")
        model = llm_model or os.environ.get("DEFAULT_LLM_MODEL", "gpt-4o-mini")
        self.explainer = GreenwashingExplainer(provider=provider, model=model)

    def analyze(self, claim: str) -> Dict:
        """Run the full four-stage pipeline on a single claim.

        Args:
            claim: Corporate marketing claim to analyse.

        Returns:
            Dict with keys:
                claim, retrieved_evidence, nli_results,
                greenwashing_score, verdict, reasoning, processing_time_ms.
        """
        start = time.time()

        # Stage 1 — Retrieval
        retrieved = self.retriever.retrieve(claim, top_k=self.top_k)

        # Stage 2 — NLI Verification
        nli_results = []
        for item in retrieved:
            nli_out = self.verifier.predict(claim, item["sentence"])
            nli_results.append(
                {
                    "sentence": item["sentence"],
                    "retrieval_score": item["score"],
                    "label": nli_out["label"],
                    "probabilities": nli_out["probabilities"],
                }
            )

        # Stage 3 — Greenwashing Scoring
        per_evidence_scores = [
            compute_greenwashing_score(r["probabilities"], self.w1, self.w2)
            for r in nli_results
        ]
        greenwashing_score = aggregate_scores(per_evidence_scores, method="max")
        verdict = get_verdict(greenwashing_score)

        # Stage 4 — Explainability
        evidence_texts = [r["sentence"] for r in nli_results]
        try:
            reasoning = self.explainer.explain(claim, evidence_texts, verdict, greenwashing_score)
        except Exception as exc:
            logger.warning(f"Explainer failed: {exc}")
            reasoning = "Explanation unavailable (LLM call failed)."

        elapsed_ms = (time.time() - start) * 1000

        return {
            "claim": claim,
            "retrieved_evidence": retrieved,
            "nli_results": nli_results,
            "greenwashing_score": greenwashing_score,
            "verdict": verdict,
            "reasoning": reasoning,
            "processing_time_ms": round(elapsed_ms, 1),
        }
