"""Cross-encoder NLI verifier using ClimateBERT or a fallback model."""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ClimateBERT fine-tuned on climate NLI; fallback if not available
_BASE_MODEL = "climatebert/distilroberta-base-climate-f"
_FALLBACK_MODEL = "cross-encoder/nli-deberta-v3-small"

# Label order varies by model checkpoint — detected at load time
_CLIMATEBERT_LABELS = ["SUPPORT", "NEUTRAL", "REFUTE"]
_DEBERTA_LABELS = ["contradiction", "neutral", "entailment"]  # DeBERTa convention
_DEBERTA_REMAP = {"contradiction": "REFUTE", "neutral": "NEUTRAL", "entailment": "SUPPORT"}


class GreenwashingVerifier:
    """Sequence-classification NLI model for greenwashing claim verification.

    Attributes:
        BASE_MODEL: Default ClimateBERT checkpoint.
    """

    BASE_MODEL = _BASE_MODEL

    def __init__(self, model_name: str = _BASE_MODEL) -> None:
        """Load tokenizer and model; fall back gracefully on load failure.

        Args:
            model_name: HuggingFace checkpoint identifier.
        """
        self._load_model(model_name)

    def _load_model(self, model_name: str) -> None:
        """Attempt to load model_name; fall back to _FALLBACK_MODEL on failure."""
        try:
            logger.info(f"Loading verifier model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.eval()
            self._model_name = model_name
            self._is_fallback = model_name == _FALLBACK_MODEL
            logger.info(f"Model loaded: {model_name}")
        except Exception as exc:
            logger.warning(f"Failed to load {model_name}: {exc}. Trying fallback...")
            self.tokenizer = AutoTokenizer.from_pretrained(_FALLBACK_MODEL)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                _FALLBACK_MODEL
            )
            self.model.eval()
            self._model_name = _FALLBACK_MODEL
            self._is_fallback = True
            logger.info("Fallback model loaded.")

    def _decode_logits(self, logits: torch.Tensor) -> Dict:
        """Convert raw logits to a probability dict with unified label names.

        Args:
            logits: 1-D tensor of raw model scores.

        Returns:
            Dict with keys SUPPORT, NEUTRAL, REFUTE mapped to float probabilities.
        """
        probs = torch.softmax(logits, dim=-1).tolist()

        if self._is_fallback:
            label_keys = [
                _DEBERTA_REMAP[self.model.config.id2label[i]]
                for i in range(len(probs))
            ]
        else:
            label_keys = _CLIMATEBERT_LABELS

        prob_dict = {k: 0.0 for k in ("SUPPORT", "NEUTRAL", "REFUTE")}
        for key, prob in zip(label_keys, probs):
            prob_dict[key] = float(prob)

        predicted_label = max(prob_dict, key=prob_dict.__getitem__)
        return {"label": predicted_label, "probabilities": prob_dict}

    def predict(self, claim: str, evidence: str) -> Dict:
        """Run NLI inference on a single claim-evidence pair.

        Args:
            claim: The marketing claim to verify.
            evidence: The evidence sentence to check against.

        Returns:
            Dict: {"label": str, "probabilities": {"SUPPORT": float, "NEUTRAL": float, "REFUTE": float}}.
        """
        inputs = self.tokenizer(
            claim, evidence, return_tensors="pt", truncation=True, max_length=512
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits[0]
        return self._decode_logits(logits)

    def predict_batch(self, pairs: List[Tuple[str, str]]) -> List[Dict]:
        """Run NLI inference on a batch of claim-evidence pairs.

        Args:
            pairs: List of (claim, evidence) tuples.

        Returns:
            List of result dicts matching predict() output format.
        """
        results = []
        for claim, evidence in pairs:
            results.append(self.predict(claim, evidence))
        return results

    def save(self, path: str) -> None:
        """Save the model and tokenizer to a directory.

        Args:
            path: Output directory path.
        """
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(out))
        self.tokenizer.save_pretrained(str(out))
        logger.info(f"Model saved to {out}")

    def load(self, path: str) -> None:
        """Load model and tokenizer from a directory.

        Args:
            path: Directory containing saved model files.
        """
        self._load_model(path)
