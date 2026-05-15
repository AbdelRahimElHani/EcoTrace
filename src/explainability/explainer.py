"""LLM-backed explainer for greenwashing verdicts."""

import logging
import os
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are an expert ESG analyst specializing in greenwashing detection. "
    "Given a corporate claim, retrieved evidence, a verdict, and a risk score, "
    "write exactly 3-4 sentences explaining: "
    "(1) what the claim asserts, "
    "(2) what the evidence shows, "
    "(3) why the verdict and score were assigned. "
    "Be precise and cite specific details from the evidence."
)


class GreenwashingExplainer:
    """Generates natural-language explanations for greenwashing verdicts via LLM.

    Supports OpenAI, Anthropic, and local Ollama backends.
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
    ) -> None:
        """Initialize the explainer with a specific LLM backend.

        Args:
            provider: "openai", "anthropic", or "local" (Ollama).
            model: Model identifier (provider-specific).
        """
        self.provider = provider
        self.model = model
        self._client = self._build_client(provider)

    def _build_client(self, provider: str):
        """Instantiate the appropriate API client.

        Args:
            provider: Backend name.

        Returns:
            Client instance or None for local/Ollama.
        """
        if provider == "openai":
            import openai

            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                logger.warning("OPENAI_API_KEY not set — LLM explanations disabled.")
                return None
            return openai.OpenAI(api_key=api_key)

        if provider == "anthropic":
            import anthropic

            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                logger.warning("ANTHROPIC_API_KEY not set — LLM explanations disabled.")
                return None
            return anthropic.Anthropic(api_key=api_key)

        if provider == "local":
            return None  # Ollama uses requests directly

        raise ValueError(f"Unknown provider: {provider}")

    def explain(
        self,
        claim: str,
        evidence: List[str],
        verdict: str,
        score: float,
    ) -> str:
        """Generate a 3-4 sentence explanation for a greenwashing verdict.

        Args:
            claim: The original corporate claim.
            evidence: List of retrieved evidence sentences.
            verdict: "LOW RISK", "MODERATE RISK", or "HIGH RISK".
            score: Greenwashing risk score in [0, 1].

        Returns:
            Explanation paragraph as a string.
        """
        evidence_block = "\n".join(
            f"  [{i+1}] {sent}" for i, sent in enumerate(evidence)
        )
        user_message = (
            f"CLAIM: {claim}\n\n"
            f"EVIDENCE:\n{evidence_block}\n\n"
            f"VERDICT: {verdict}\n"
            f"RISK SCORE: {score:.2f}"
        )

        if self._client is None and self.provider != "local":
            return "Explanation unavailable (no API key configured)."

        if self.provider == "openai":
            return self._call_openai(user_message)
        if self.provider == "anthropic":
            return self._call_anthropic(user_message)
        if self.provider == "local":
            return self._call_ollama(user_message)

        return "Explanation unavailable."

    def _call_openai(self, user_message: str) -> str:
        """Call OpenAI chat completion API."""
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_tokens=300,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    def _call_anthropic(self, user_message: str) -> str:
        """Call Anthropic Messages API."""
        response = self._client.messages.create(
            model=self.model,
            max_tokens=300,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text.strip()

    def _call_ollama(self, user_message: str) -> str:
        """Call local Ollama REST API."""
        import json

        import requests

        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            "stream": False,
        }
        resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
