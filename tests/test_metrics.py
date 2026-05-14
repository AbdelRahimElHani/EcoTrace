"""Unit tests for evaluation/metrics.py."""

import pytest

from src.evaluation.metrics import (
    compute_classification_metrics,
    compute_recall_at_k,
)


class TestComputeClassificationMetrics:
    def test_perfect_predictions(self):
        y = [0, 1, 2, 0, 1, 2]
        metrics = compute_classification_metrics(y, y)
        assert metrics["accuracy"] == 1.0
        assert metrics["f1"] == 1.0

    def test_keys_present(self):
        y_true = [0, 1, 2]
        y_pred = [0, 2, 1]
        metrics = compute_classification_metrics(y_true, y_pred)
        for key in ("precision", "recall", "f1", "accuracy", "confusion_matrix", "report"):
            assert key in metrics

    def test_confusion_matrix_shape(self):
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 0, 2, 1, 1, 2]
        cm = compute_classification_metrics(y_true, y_pred)["confusion_matrix"]
        assert len(cm) == 3 and all(len(row) == 3 for row in cm)


class TestComputeRecallAtK:
    def test_perfect_retrieval(self):
        queries = ["q1"]
        relevant = [["doc_a", "doc_b"]]
        retrieved = [["doc_a", "doc_b", "doc_c"]]
        assert compute_recall_at_k(queries, relevant, retrieved, k=2) == 1.0

    def test_no_relevant_retrieved(self):
        queries = ["q1"]
        relevant = [["doc_a"]]
        retrieved = [["doc_x", "doc_y"]]
        assert compute_recall_at_k(queries, relevant, retrieved, k=2) == 0.0

    def test_k_cutoff_respected(self):
        queries = ["q1"]
        relevant = [["doc_b"]]
        retrieved = [["doc_a", "doc_b"]]
        # doc_b is at position 2; k=1 should miss it
        assert compute_recall_at_k(queries, relevant, retrieved, k=1) == 0.0
        assert compute_recall_at_k(queries, relevant, retrieved, k=2) == 1.0

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            compute_recall_at_k(["q1", "q2"], [["a"]], [["b"]], k=1)
