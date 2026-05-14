"""Evaluate the full EcoTrace pipeline against the gold standard."""

import argparse
import json
import logging
from pathlib import Path

from src.evaluation.metrics import evaluate_pipeline
from src.pipeline import EcoTracePipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate EcoTrace pipeline on gold standard")
    p.add_argument(
        "--gold_standard",
        type=str,
        default="data/gold_standard/gold_standard.json",
    )
    p.add_argument(
        "--index_path",
        type=str,
        default="models/bi_encoder_index",
        help="Pre-built bi-encoder index directory",
    )
    p.add_argument(
        "--verifier_model",
        type=str,
        default="models/cross_encoder",
        help="Fine-tuned cross-encoder directory (or HF model ID)",
    )
    p.add_argument(
        "--output",
        type=str,
        default="results/evaluation_report.json",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.gold_standard, "r", encoding="utf-8") as f:
        gold_data = json.load(f)

    label_map = {"SUPPORT": 0, "REFUTE": 1, "NEUTRAL": 2}
    test_data = [
        {
            "claim": item["claim"],
            "evidence": [item["evidence"]],
            "label": label_map.get(item["label"], 2),
        }
        for item in gold_data
    ]

    logger.info("Initializing pipeline...")
    pipeline = EcoTracePipeline(
        index_path=args.index_path,
        verifier_model=args.verifier_model,
    )

    logger.info("Running evaluation...")
    metrics = evaluate_pipeline(pipeline, test_data, k_values=[1, 3, 5])

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        # confusion_matrix is already a list; remove verbose report for JSON
        json_metrics = {k: v for k, v in metrics.items() if k != "report"}
        json.dump(json_metrics, f, indent=2)

    logger.info(f"\n{metrics['report']}")
    logger.info(f"Recall@1: {metrics.get('recall@1', 'N/A'):.4f}")
    logger.info(f"Recall@3: {metrics.get('recall@3', 'N/A'):.4f}")
    logger.info(f"Recall@5: {metrics.get('recall@5', 'N/A'):.4f}")
    logger.info(f"Evaluation report saved to {out_path}")


if __name__ == "__main__":
    main()
