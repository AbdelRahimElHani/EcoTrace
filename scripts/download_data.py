"""Download and preprocess CLIMATE-FEVER dataset."""

import argparse
import logging
from pathlib import Path

from src.data.preprocess import (
    flatten_to_nli_pairs,
    load_climate_fever,
    save_splits,
    stratified_split,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and preprocess CLIMATE-FEVER")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Directory to save train/val/test JSON splits",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("Starting CLIMATE-FEVER download and preprocessing...")

    dataset = load_climate_fever()
    pairs = flatten_to_nli_pairs(dataset)
    train, val, test = stratified_split(pairs, seed=args.seed)

    output_dir = Path(args.output_dir)
    save_splits(train, val, test, str(output_dir))

    logger.info("Done. Splits saved to %s", output_dir)


if __name__ == "__main__":
    main()
