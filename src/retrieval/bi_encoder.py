"""Bi-encoder retriever using sentence-transformers and cosine similarity."""

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BiEncoderRetriever:
    """Dense retriever built on sentence-transformers with numpy cosine similarity.

    Attributes:
        MODEL_NAME: Default HuggingFace model identifier.
    """

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        """Initialize the retriever and load the encoder model.

        Args:
            model_name: HuggingFace sentence-transformers model identifier.
        """
        logger.info(f"Loading bi-encoder model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self._sentences: List[str] = []
        self._embeddings: np.ndarray = np.array([])

    def build_index(self, sentences: List[str]) -> None:
        """Encode sentences and store the embedding matrix.

        Args:
            sentences: List of document sentences to index.
        """
        if not sentences:
            raise ValueError("Cannot build index from empty sentence list.")

        self._sentences = sentences
        logger.info(f"Encoding {len(sentences)} sentences (batch_size=64)...")
        self._embeddings = self.model.encode(
            sentences, batch_size=64, show_progress_bar=True, convert_to_numpy=True
        )
        # L2-normalize for fast cosine via dot product
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-9, norms)
        self._embeddings = self._embeddings / norms
        logger.info("Index built successfully.")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve top-k sentences most similar to the query.

        Args:
            query: The claim text to search against the index.
            top_k: Number of results to return.

        Returns:
            List of dicts: [{"sentence": str, "score": float, "rank": int}].
        """
        if self._embeddings.size == 0:
            raise RuntimeError("Index is empty. Call build_index() first.")

        query_vec = self.model.encode([query], convert_to_numpy=True)
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-9)

        # Cosine similarity via dot product (embeddings are already L2-normalized)
        scores = self._embeddings @ query_vec.T  # shape: (N, 1)
        scores = scores.squeeze()

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [
            {
                "sentence": self._sentences[int(idx)],
                "score": float(scores[int(idx)]),
                "rank": rank + 1,
            }
            for rank, idx in enumerate(top_indices)
        ]
        return results

    def save_index(self, path: str) -> None:
        """Persist the sentence list and embedding matrix to disk.

        Args:
            path: Directory path (or prefix) to save index files.
        """
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        np.save(str(out / "embeddings.npy"), self._embeddings)
        with open(out / "sentences.json", "w", encoding="utf-8") as f:
            json.dump(self._sentences, f, ensure_ascii=False)
        logger.info(f"Index saved to {out}")

    def load_index(self, path: str) -> None:
        """Load a previously saved index from disk.

        Args:
            path: Directory path containing embeddings.npy and sentences.json.
        """
        p = Path(path)
        self._embeddings = np.load(str(p / "embeddings.npy"))
        with open(p / "sentences.json", "r", encoding="utf-8") as f:
            self._sentences = json.load(f)
        logger.info(f"Loaded index: {len(self._sentences)} sentences from {p}")
