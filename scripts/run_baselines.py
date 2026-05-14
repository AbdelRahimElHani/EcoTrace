"""Baseline models for greenwashing detection on the gold standard.

Baselines:
    1. TF-IDF + Logistic Regression (trained on CLIMATE-FEVER)
    2. Zero-shot RoBERTa NLI (roberta-large-mnli, no training)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LABEL_STR_TO_INT = {"SUPPORT": 0, "REFUTE": 1, "NEUTRAL": 2}
LABEL_INT_TO_STR = {0: "SUPPORT", 1: "REFUTE", 2: "NEUTRAL"}
TARGET_NAMES = ["SUPPORT", "REFUTE", "NEUTRAL"]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_gold_standard(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    claims = [d["claim"] for d in data]
    evidences = [d["evidence"] for d in data]
    labels = [LABEL_STR_TO_INT[d["label"]] for d in data]
    return claims, evidences, labels


def load_train_data(train_json: str):
    """Load CLIMATE-FEVER train split produced by download_data.py."""
    with open(train_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = [d["claim"] + " [SEP] " + d["evidence"] for d in data]
    labels = [d["label"] for d in data]
    return texts, labels


def _download_and_prepare(output_dir: str):
    """Fallback: download CLIMATE-FEVER and build splits on the fly."""
    logger.info("Training split not found — downloading CLIMATE-FEVER...")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.data.preprocess import (
        flatten_to_nli_pairs,
        load_climate_fever,
        save_splits,
        stratified_split,
    )
    dataset = load_climate_fever()
    pairs = flatten_to_nli_pairs(dataset)
    train, val, test = stratified_split(pairs, seed=42)
    save_splits(train, val, test, output_dir)
    logger.info("Splits saved to %s", output_dir)
    train_path = Path(output_dir) / "train.json"
    return load_train_data(str(train_path))


def get_train_data(processed_dir: str):
    train_path = Path(processed_dir) / "train.json"
    if train_path.exists():
        logger.info("Loading training data from %s", train_path)
        return load_train_data(str(train_path))
    return _download_and_prepare(processed_dir)


# ---------------------------------------------------------------------------
# Baseline 1: TF-IDF + Logistic Regression
# ---------------------------------------------------------------------------

def run_tfidf_lr(train_texts, train_labels, test_texts, test_labels):
    logger.info("Running Baseline 1: TF-IDF + Logistic Regression")
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=50_000, sublinear_tf=True)
    X_train = vec.fit_transform(train_texts)
    X_test = vec.transform(test_texts)

    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=42)
    clf.fit(X_train, train_labels)

    preds = clf.predict(X_test)
    return _report("TF-IDF + Logistic Regression", test_labels, preds)


# ---------------------------------------------------------------------------
# Baseline 2: Zero-shot RoBERTa NLI
# ---------------------------------------------------------------------------

def run_roberta_zeroshot(test_claims, test_evidences, test_labels):
    logger.info("Running Baseline 2: Zero-shot RoBERTa NLI (roberta-large-mnli)")
    from transformers import pipeline as hf_pipeline

    nli = hf_pipeline(
        "text-classification",
        model="roberta-large-mnli",
        device=-1,          # CPU; set to 0 for GPU
        truncation=True,
        max_length=512,
    )

    # roberta-large-mnli label mapping: ENTAILMENT→SUPPORT, NEUTRAL→NEUTRAL, CONTRADICTION→REFUTE
    roberta_map = {
        "ENTAILMENT": 0,
        "NEUTRAL": 2,
        "CONTRADICTION": 1,
    }

    preds = []
    for claim, evidence in zip(test_claims, test_evidences):
        text = claim + " </s></s> " + evidence
        out = nli(text)[0]
        label_str = out["label"].upper()
        preds.append(roberta_map.get(label_str, 2))

    return _report("Zero-shot RoBERTa NLI", test_labels, preds)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _report(name: str, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(
        y_true, y_pred, target_names=TARGET_NAMES, zero_division=0
    )
    logger.info("\n=== %s ===\nAccuracy: %.4f  Macro-F1: %.4f\n%s", name, acc, macro_f1, report)
    return {
        "model": name,
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "classification_report": report,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Run baseline models on gold standard")
    p.add_argument("--gold_standard", default="data/gold_standard/gold_standard.json")
    p.add_argument("--processed_dir", default="data/processed")
    p.add_argument("--output", default="results/baseline_results.json")
    p.add_argument(
        "--skip_roberta",
        action="store_true",
        help="Skip the zero-shot RoBERTa baseline (large download)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Load gold standard
    gs_path = Path(args.gold_standard)
    if not gs_path.exists():
        logger.error("Gold standard not found at %s", gs_path)
        sys.exit(1)
    claims, evidences, labels = load_gold_standard(str(gs_path))
    test_texts = [c + " [SEP] " + e for c, e in zip(claims, evidences)]
    logger.info("Gold standard loaded: %d examples", len(labels))

    # Training data for TF-IDF + LR
    train_texts, train_labels = get_train_data(args.processed_dir)
    logger.info("Training set: %d examples", len(train_labels))

    results = []

    # Baseline 1
    results.append(run_tfidf_lr(train_texts, train_labels, test_texts, labels))

    # Baseline 2
    if not args.skip_roberta:
        results.append(run_roberta_zeroshot(claims, evidences, labels))
    else:
        logger.info("Skipping RoBERTa baseline (--skip_roberta set)")

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
