"""Unit tests for verification/scorer.py."""

import pytest

from src.verification.scorer import (
    aggregate_scores,
    compute_greenwashing_score,
    get_verdict,
)


class TestComputeGreenwashingScore:
    def test_high_refute(self):
        probs = {"SUPPORT": 0.05, "NEUTRAL": 0.05, "REFUTE": 0.90}
        score = compute_greenwashing_score(probs)
        assert score > 0.6

    def test_high_support(self):
        probs = {"SUPPORT": 0.90, "NEUTRAL": 0.05, "REFUTE": 0.05}
        score = compute_greenwashing_score(probs)
        assert score < 0.3

    def test_neutral_case(self):
        probs = {"SUPPORT": 0.33, "NEUTRAL": 0.34, "REFUTE": 0.33}
        score = compute_greenwashing_score(probs)
        assert 0.0 <= score <= 1.0

    def test_weights_must_sum_to_one(self):
        probs = {"SUPPORT": 0.5, "NEUTRAL": 0.3, "REFUTE": 0.2}
        with pytest.raises(ValueError):
            compute_greenwashing_score(probs, w1=0.6, w2=0.6)

    def test_formula_correctness(self):
        probs = {"SUPPORT": 0.2, "NEUTRAL": 0.3, "REFUTE": 0.5}
        expected = 0.7 * 0.5 + 0.3 * (1 - 0.2)
        assert abs(compute_greenwashing_score(probs) - expected) < 1e-6


class TestGetVerdict:
    def test_low_risk(self):
        assert get_verdict(0.1) == "LOW RISK"

    def test_boundary_low(self):
        assert get_verdict(0.29) == "LOW RISK"

    def test_moderate_risk(self):
        assert get_verdict(0.45) == "MODERATE RISK"

    def test_high_risk(self):
        assert get_verdict(0.75) == "HIGH RISK"

    def test_boundary_high(self):
        assert get_verdict(0.6) == "HIGH RISK"


class TestAggregateScores:
    def test_max(self):
        assert aggregate_scores([0.1, 0.9, 0.5]) == 0.9

    def test_mean(self):
        assert abs(aggregate_scores([0.2, 0.4, 0.6], method="mean") - 0.4) < 1e-6

    def test_min(self):
        assert aggregate_scores([0.3, 0.7, 0.5], method="min") == 0.3

    def test_empty_returns_zero(self):
        assert aggregate_scores([]) == 0.0

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            aggregate_scores([0.5], method="median")
