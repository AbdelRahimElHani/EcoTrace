"""PDF text extraction and environmental sentence filtering for ESG reports."""

import json
import logging
from pathlib import Path
from typing import List

import fitz  # PyMuPDF
import spacy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ENVIRONMENTAL_KEYWORDS = [
    "carbon", "emission", "CO2", "renewable", "net zero",
    "GHG", "climate", "energy", "waste",
]

_NLP = None


def _get_nlp() -> spacy.Language:
    """Lazy-load a blank spaCy pipeline with sentencizer (fast, no parser needed)."""
    global _NLP
    if _NLP is None:
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        nlp.max_length = 10_000_000
        _NLP = nlp
    return _NLP


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract full text from a PDF file using PyMuPDF.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Concatenated text from all pages.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    text_parts = []
    with fitz.open(str(path)) as doc:
        for page in doc:
            text_parts.append(page.get_text("text"))

    text = "\n".join(text_parts)
    logger.info(f"Extracted {len(text)} characters from {path.name}")
    return text


def sentence_tokenize(text: str) -> List[str]:
    """Split text into sentences using spaCy en_core_web_sm.

    Args:
        text: Raw text string.

    Returns:
        List of sentence strings.
    """
    nlp = _get_nlp()
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    logger.info(f"Tokenized into {len(sentences)} sentences")
    return sentences


def filter_environmental_sentences(sentences: List[str]) -> List[str]:
    """Keep only sentences containing environmental keywords.

    Args:
        sentences: List of sentence strings.

    Returns:
        Filtered list containing only environment-relevant sentences.
    """
    filtered = [
        s for s in sentences
        if any(kw.lower() in s.lower() for kw in ENVIRONMENTAL_KEYWORDS)
    ]
    logger.info(
        f"Filtered {len(filtered)} environmental sentences from {len(sentences)} total"
    )
    return filtered


def process_esg_report(pdf_path: str, output_path: str) -> None:
    """Full pipeline: extract → tokenize → filter → save.

    Args:
        pdf_path: Path to the ESG report PDF.
        output_path: Path to write the filtered sentences JSON.
    """
    text = extract_text_from_pdf(pdf_path)
    sentences = sentence_tokenize(text)
    env_sentences = filter_environmental_sentences(sentences)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    company_name = Path(pdf_path).stem
    data = {"source": company_name, "sentences": env_sentences}

    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(env_sentences)} sentences to {out}")
