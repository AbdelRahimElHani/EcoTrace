"""Back-translation augmentation for the REFUTE class."""

import logging
from typing import Dict, List

from deep_translator import GoogleTranslator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def back_translate(text: str, intermediate_lang: str = "de") -> str:
    """Translate text to an intermediate language and back to English.

    Args:
        text: Original English text.
        intermediate_lang: BCP-47 language code for the pivot language.

    Returns:
        Back-translated English string.
    """
    try:
        pivot = GoogleTranslator(source="en", target=intermediate_lang).translate(text)
        back = GoogleTranslator(source=intermediate_lang, target="en").translate(pivot)
        return back
    except Exception as exc:
        logger.warning(f"Back-translation failed for text snippet: {exc}")
        return text  # Fallback to original on failure


def augment_dataset(
    data: List[Dict],
    target_label: int = 1,
    ratio: float = 1.0,
) -> List[Dict]:
    """Augment a dataset by back-translating examples of a target label.

    Args:
        data: List of dicts with keys: claim, evidence, label.
        target_label: Label integer to augment (default 1 = REFUTE).
        ratio: Fraction of target-label examples to augment (1.0 = 100%).

    Returns:
        Original data plus augmented examples.
    """
    target_examples = [d for d in data if d["label"] == target_label]
    n_augment = max(1, int(len(target_examples) * ratio))
    to_augment = target_examples[:n_augment]

    logger.info(
        f"Augmenting {n_augment}/{len(target_examples)} label-{target_label} examples..."
    )

    augmented = []
    for i, item in enumerate(to_augment):
        new_claim = back_translate(item["claim"])
        new_evidence = back_translate(item["evidence"])
        augmented.append(
            {"claim": new_claim, "evidence": new_evidence, "label": target_label}
        )
        if (i + 1) % 50 == 0:
            logger.info(f"  Augmented {i + 1}/{n_augment}")

    logger.info(f"Added {len(augmented)} augmented examples")
    return data + augmented
