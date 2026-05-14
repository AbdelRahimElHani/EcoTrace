"""Unit tests for data/preprocess.py (offline, no HuggingFace download)."""

import pytest

from src.data.preprocess import LABEL_MAP, flatten_to_nli_pairs, stratified_split


def _make_fake_dataset():
    """Build a minimal dict mimicking CLIMATE-FEVER structure."""

    class FakeExample(dict):
        pass

    examples = []
    label_cycle = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]
    for i in range(30):
        ex = {
            "claim": f"Claim number {i}",
            "claim_label": label_cycle[i % 3],
            "evidences": [{"evidence": f"Evidence sentence {i}"}],
        }
        examples.append(ex)

    return {"test": examples}


class TestFlattenToNliPairs:
    def test_output_keys(self):
        dataset = _make_fake_dataset()
        pairs = flatten_to_nli_pairs(dataset)
        assert all("claim" in p and "evidence" in p and "label" in p for p in pairs)

    def test_label_mapping(self):
        dataset = _make_fake_dataset()
        pairs = flatten_to_nli_pairs(dataset)
        labels = {p["label"] for p in pairs}
        assert labels == {0, 1, 2}

    def test_count(self):
        dataset = _make_fake_dataset()
        pairs = flatten_to_nli_pairs(dataset)
        assert len(pairs) == 30  # 30 examples × 1 evidence each


class TestStratifiedSplit:
    def test_split_sizes_sum_to_total(self):
        pairs = [{"claim": f"c{i}", "evidence": f"e{i}", "label": i % 3} for i in range(100)]
        train, val, test = stratified_split(pairs, test_size=0.1, val_size=0.1)
        assert len(train) + len(val) + len(test) == 100

    def test_reproducible_with_seed(self):
        pairs = [{"claim": f"c{i}", "evidence": f"e{i}", "label": i % 3} for i in range(60)]
        t1, v1, te1 = stratified_split(pairs, seed=42)
        t2, v2, te2 = stratified_split(pairs, seed=42)
        assert [p["claim"] for p in t1] == [p["claim"] for p in t2]

    def test_all_labels_in_train(self):
        pairs = [{"claim": f"c{i}", "evidence": f"e{i}", "label": i % 3} for i in range(90)]
        train, _, _ = stratified_split(pairs)
        train_labels = {p["label"] for p in train}
        assert train_labels == {0, 1, 2}
