"""Index ESG report PDFs for bi-encoder retrieval."""

import argparse
import json
import logging
from pathlib import Path

from src.data.pdf_extractor import process_esg_report
from src.retrieval.bi_encoder import BiEncoderRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract and index ESG report PDFs")
    p.add_argument(
        "--pdf_dir",
        type=str,
        default="data/esg_reports",
        help="Directory containing ESG report PDFs",
    )
    p.add_argument(
        "--processed_dir",
        type=str,
        default="data/processed/esg_sentences",
        help="Where to save extracted sentence JSON files",
    )
    p.add_argument(
        "--index_path",
        type=str,
        default="models/bi_encoder_index",
        help="Where to save the bi-encoder index",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pdf_dir = Path(args.pdf_dir)
    processed_dir = Path(args.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {pdf_dir}. Place ESG report PDFs there.")
        return

    logger.info(f"Found {len(pdf_files)} PDFs to process")

    all_sentences = []
    for pdf_path in pdf_files:
        output_path = processed_dir / f"{pdf_path.stem}.json"
        logger.info(f"Processing: {pdf_path.name}")
        process_esg_report(str(pdf_path), str(output_path))

        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        all_sentences.extend(data.get("sentences", []))

    # Deduplicate while preserving order
    seen = set()
    unique_sentences = []
    for s in all_sentences:
        if s not in seen:
            seen.add(s)
            unique_sentences.append(s)

    logger.info(f"Total unique environmental sentences: {len(unique_sentences)}")

    retriever = BiEncoderRetriever()
    retriever.build_index(unique_sentences)
    retriever.save_index(args.index_path)

    logger.info(f"Index saved to {args.index_path}")


if __name__ == "__main__":
    main()
