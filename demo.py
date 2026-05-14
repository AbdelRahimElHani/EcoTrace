"""EcoTrace interactive demo — three canonical test cases."""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Add project root to path so src imports work when run directly
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import EcoTracePipeline  # noqa: E402

_SEPARATOR = "=" * 51

_DEMO_CASES = [
    {
        "name": "Case 1 — HIGH RISK (typical)",
        "claim": "We have achieved carbon neutrality across all our operations.",
        "expected": "REFUTE, S_gw > 0.7",
    },
    {
        "name": "Case 2 — MODERATE RISK (edge)",
        "claim": "We are committed to reducing emissions by 2050.",
        "expected": "NEUTRAL, S_gw 0.3–0.6",
    },
    {
        "name": "Case 3 — Retriever Failure (vocabulary mismatch)",
        "claim": "Our supply chain is 100% deforestation-free.",
        "expected": "Retriever may return unrelated AWS/energy sentences — expected failure mode",
    },
]


def print_result(result: dict, case_name: str) -> None:
    """Render pipeline output in the canonical CLI format."""
    print(_SEPARATOR)
    print("  EcoTrace - Greenwashing Detection System")
    print(_SEPARATOR)
    print(f"[{case_name}]")
    print(f'CLAIM: "{result["claim"]}"')
    print()
    print("RETRIEVED EVIDENCE:")
    for item in result["retrieved_evidence"]:
        print(f'  [{item["rank"]}] (score: {item["score"]:.2f}) "{item["sentence"]}"')
    print()
    print(f'NLI VERDICT:  {result["nli_results"][0]["label"] if result["nli_results"] else "N/A"}')
    print(f'RISK SCORE:   {result["greenwashing_score"]:.2f} / 1.00')
    print(f'VERDICT:      {result["verdict"]}')
    print(f'REASONING:    {result["reasoning"]}')
    print(f'Processing Time: {result["processing_time_ms"] / 1000:.2f}s')
    print(_SEPARATOR)
    print()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EcoTrace demo")
    p.add_argument(
        "--index_path",
        type=str,
        default="models/bi_encoder_index",
        help="Path to pre-built bi-encoder index",
    )
    p.add_argument(
        "--verifier_model",
        type=str,
        default="models/cross_encoder",
        help="Fine-tuned cross-encoder directory or HF model ID",
    )
    p.add_argument(
        "--claim",
        type=str,
        default=None,
        help="Run a single custom claim instead of the three demo cases",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("Initializing EcoTrace pipeline...")
    pipeline = EcoTracePipeline(
        index_path=args.index_path,
        verifier_model=args.verifier_model,
    )

    if args.claim:
        result = pipeline.analyze(args.claim)
        print_result(result, "Custom Claim")
        return

    for case in _DEMO_CASES:
        print(f"\nRunning {case['name']}...")
        print(f"(Expected: {case['expected']})")
        result = pipeline.analyze(case["claim"])
        print_result(result, case["name"])


if __name__ == "__main__":
    main()
