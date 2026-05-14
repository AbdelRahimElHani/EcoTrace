"""Unit tests for retrieval/bi_encoder.py (uses real model — requires network on first run)."""

import tempfile
from pathlib import Path

import pytest

from src.retrieval.bi_encoder import BiEncoderRetriever


@pytest.fixture(scope="module")
def retriever():
    """Shared retriever instance loaded once per test module."""
    r = BiEncoderRetriever()
    sentences = [
        "Our carbon emissions decreased by 30% this year.",
        "We invested $2 billion in renewable energy projects.",
        "The company uses 100% recycled packaging materials.",
        "Net zero target set for 2040 across all operations.",
        "GHG intensity fell by 15% compared to 2020 baseline.",
    ]
    r.build_index(sentences)
    return r


class TestBiEncoderRetriever:
    def test_retrieve_returns_correct_count(self, retriever):
        results = retriever.retrieve("carbon emissions reduction", top_k=3)
        assert len(results) == 3

    def test_retrieve_result_keys(self, retriever):
        results = retriever.retrieve("renewable energy", top_k=1)
        assert all("sentence" in r and "score" in r and "rank" in r for r in results)

    def test_scores_are_in_valid_range(self, retriever):
        results = retriever.retrieve("climate change", top_k=5)
        for r in results:
            assert -1.0 <= r["score"] <= 1.0

    def test_ranks_are_sequential(self, retriever):
        results = retriever.retrieve("net zero emissions", top_k=3)
        assert [r["rank"] for r in results] == [1, 2, 3]

    def test_save_and_load_index(self, retriever):
        with tempfile.TemporaryDirectory() as tmpdir:
            retriever.save_index(tmpdir)
            new_retriever = BiEncoderRetriever()
            new_retriever.load_index(tmpdir)
            results = new_retriever.retrieve("renewable energy", top_k=2)
            assert len(results) == 2

    def test_empty_index_raises(self):
        r = BiEncoderRetriever()
        with pytest.raises(RuntimeError):
            r.retrieve("any query")
