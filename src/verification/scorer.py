"""Greenwashing risk scoring and verdict assignment."""

import logging
from typing import Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_W1_DEFAULT = 0.7
_W2_DEFAULT = 0.3

_THRESHOLD_LOW = 0.3
_THRESHOLD_HIGH = 0.6


def compute_greenwashing_score(
    probabilities: Dict[str, float],
    w1: float = _W1_DEFAULT,
    w2: float = _W2_DEFAULT,
) -> float:
    """Compute the greenwashing risk score from NLI probabilities.

    S_gw = w1 * P(Refute) + w2 * (1 - P(Support))

    Args:
        probabilities: Dict with keys SUPPORT, NEUTRAL, REFUTE as floats.
        w1: Weight on P(Refute). Must satisfy w1 + w2 == 1.
        w2: Weight on (1 - P(Support)).

    Returns:
        Greenwashing risk score in [0, 1].
    """
    if abs(w1 + w2 - 1.0) > 1e-6:
        raise ValueError(f"Weights must sum to 1; got w1={w1}, w2={w2}")

    p_refute = probabilities.get("REFUTE", 0.0)
    p_support = probabilities.get("SUPPORT", 0.0)
    score = w1 * p_refute + w2 * (1.0 - p_support)
    return float(min(max(score, 0.0), 1.0))


def get_verdict(score: float) -> str:
    """Map a greenwashing risk score to a human-readable verdict.

    Args:
        score: Greenwashing risk score in [0, 1].

    Returns:
        "LOW RISK", "MODERATE RISK", or "HIGH RISK".
    """
    if score < _THRESHOLD_LOW:
        return "LOW RISK"
    if score < _THRESHOLD_HIGH:
        return "MODERATE RISK"
    return "HIGH RISK"


def aggregate_scores(scores: List[float], method: str = "max") -> float:
    """Aggregate multiple per-evidence scores into a single claim-level score.

    Args:
        scores: List of per-evidence greenwashing scores.
        method: Aggregation strategy — "max", "mean", or "min".

    Returns:
        Aggregated score.
    """
    if not scores:
        return 0.0

    if method == "max":
        return float(max(scores))
    if method == "mean":
        return float(sum(scores) / len(scores))
    if method == "min":
        return float(min(scores))

    raise ValueError(f"Unknown aggregation method: {method}")
