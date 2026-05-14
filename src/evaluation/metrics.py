"""Classification and retrieval evaluation metrics for EcoTrace."""

import logging
from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
) -> Dict:
    """Compute precision, recall, F1, accuracy, and confusion matrix.

    Args:
        y_true: Ground-truth integer labels.
        y_pred: Predicted integer labels.

    Returns:
        Dict with keys: precision, recall, f1, accuracy, confusion_matrix, report.
    """
    return {
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "report": classification_report(y_true, y_pred, zero_division=0),
    }


def compute_recall_at_k(
    queries: List[str],
    relevant_docs: List[List[str]],
    retrieved_docs: List[List[str]],
    k: int,
) -> float:
    """Compute mean Recall@K across all queries.

    Recall@K = |Relevant ∩ Retrieved_K| / |Relevant|

    Args:
        queries: Query strings (unused beyond length validation).
        relevant_docs: Per-query list of ground-truth relevant documents.
        retrieved_docs: Per-query list of retrieved documents (any length).
        k: Cutoff for retrieval.

    Returns:
        Mean Recall@K across all queries (float in [0, 1]).
    """
    if len(queries) != len(relevant_docs) or len(queries) != len(retrieved_docs):
        raise ValueError("queries, relevant_docs, and retrieved_docs must have equal length.")

    recall_scores = []
    for rel, ret in zip(relevant_docs, retrieved_docs):
        rel_set = set(rel)
        ret_at_k = set(ret[:k])
        if len(rel_set) == 0:
            recall_scores.append(1.0)
        else:
            recall_scores.append(len(rel_set & ret_at_k) / len(rel_set))

    return float(np.mean(recall_scores))


def evaluate_pipeline(
    pipeline,
    test_data: List[Dict],
    k_values: List[int] = None,
) -> Dict:
    """Run the full EcoTrace pipeline on test_data and collect all metrics.

    Args:
        pipeline: An EcoTracePipeline instance with an analyze(claim) method.
        test_data: List of dicts with keys: claim, evidence (list), label (int).
        k_values: Recall@K cutoffs to compute (default [1, 3, 5]).

    Returns:
        Dict with classification metrics and Recall@K values.
    """
    if k_values is None:
        k_values = [1, 3, 5]

    label_to_int = {"SUPPORT": 0, "REFUTE": 1, "NEUTRAL": 2}
    y_true, y_pred = [], []
    queries, relevant_docs, retrieved_docs = [], [], []

    for item in test_data:
        claim = item["claim"]
        result = pipeline.analyze(claim)

        pred_label_str = result.get("nli_results", [{}])[0].get("label", "NEUTRAL")
        y_pred.append(label_to_int.get(pred_label_str, 2))
        y_true.append(int(item.get("label", 2)))

        queries.append(claim)
        relevant_docs.append(item.get("evidence", []))
        retrieved_docs.append(
            [e["sentence"] for e in result.get("retrieved_evidence", [])]
        )

    clf_metrics = compute_classification_metrics(y_true, y_pred)

    recall_at_k = {
        f"recall@{k}": compute_recall_at_k(queries, relevant_docs, retrieved_docs, k)
        for k in k_values
    }

    logger.info(f"Pipeline evaluation complete — macro-F1: {clf_metrics['f1']:.4f}")
    return {**clf_metrics, **recall_at_k}
