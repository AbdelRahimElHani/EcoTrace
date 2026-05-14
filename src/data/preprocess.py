"""CLIMATE-FEVER dataset loading, flattening, splitting, and saving."""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LABEL_MAP = {"SUPPORTS": 0, "REFUTES": 1, "NOT_ENOUGH_INFO": 2}
LABEL_NAMES = {0: "SUPPORT", 1: "REFUTE", 2: "NEUTRAL"}


def load_climate_fever() -> datasets.DatasetDict:
    """Load the CLIMATE-FEVER dataset from HuggingFace."""
    logger.info("Loading CLIMATE-FEVER dataset...")
    dataset = datasets.load_dataset("climate_fever")
    logger.info(f"Loaded {len(dataset['test'])} examples")
    return dataset


def flatten_to_nli_pairs(dataset: datasets.DatasetDict) -> List[Dict]:
    """Flatten CLIMATE-FEVER claims+evidences into NLI pairs.

    Args:
        dataset: Raw CLIMATE-FEVER DatasetDict.

    Returns:
        List of dicts with keys: claim, evidence, label (int).
    """
    pairs = []
    split = dataset["test"]  # CLIMATE-FEVER only has test split

    for example in split:
        claim = example["claim"]
        for evidence_item in example["evidences"]:
            evidence_text = evidence_item["evidence"]
            # Use claim-level label
            label_str = example["claim_label"]
            label = LABEL_MAP.get(label_str, 2)
            pairs.append({"claim": claim, "evidence": evidence_text, "label": label})

    logger.info(f"Flattened to {len(pairs)} NLI pairs")
    return pairs


def stratified_split(
    data: List[Dict],
    test_size: float = 0.1,
    val_size: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Stratified split into train/val/test sets.

    Args:
        data: List of NLI pair dicts.
        test_size: Fraction for test set.
        val_size: Fraction for validation set.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train, val, test) lists.
    """
    random.seed(seed)

    # Group by label
    by_label: Dict[int, List[Dict]] = {}
    for item in data:
        label = item["label"]
        by_label.setdefault(label, []).append(item)

    train_all, val_all, test_all = [], [], []

    for label, items in by_label.items():
        random.shuffle(items)
        n = len(items)
        n_test = max(1, int(n * test_size))
        n_val = max(1, int(n * val_size))

        test_all.extend(items[:n_test])
        val_all.extend(items[n_test : n_test + n_val])
        train_all.extend(items[n_test + n_val :])

    random.shuffle(train_all)
    logger.info(
        f"Split sizes — train: {len(train_all)}, val: {len(val_all)}, test: {len(test_all)}"
    )
    return train_all, val_all, test_all


def save_splits(
    train: List[Dict],
    val: List[Dict],
    test: List[Dict],
    output_dir: str,
) -> None:
    """Save train/val/test splits as JSON files.

    Args:
        train: Training examples.
        val: Validation examples.
        test: Test examples.
        output_dir: Directory to write JSON files.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name, split in [("train", train), ("val", val), ("test", test)]:
        path = out / f"{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(split, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(split)} examples to {path}")


if __name__ == "__main__":
    dataset = load_climate_fever()
    pairs = flatten_to_nli_pairs(dataset)
    train, val, test = stratified_split(pairs)
    save_splits(train, val, test, "data/processed")
