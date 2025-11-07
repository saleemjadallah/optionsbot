"""Prompt templates tailored for Jeffrey."""

from __future__ import annotations

from typing import Dict


class PromptLibrary:
    def __init__(self) -> None:
        self.prompts: Dict[str, str] = {
            "claude": self._claude_prompt(),
            "perplexity": self._perplexity_prompt(),
            "openai": self._openai_prompt(),
        }

    def get_prompt(self, model: str) -> str:
        return self.prompts.get(model, self.prompts["claude"])

    @staticmethod
    def _claude_prompt() -> str:
        return (
            "You are Jeffrey, a veteran options strategist embedded inside a hybrid "
            "automation stack. Provide deeply reasoned answers with numbered steps, "
            "identify risk factors, and close with a concise action checklist."
        )

    @staticmethod
    def _perplexity_prompt() -> str:
        return (
            "You are Jeffrey's market intelligence wing. Summarize the most recent "
            "news, cite sources, and highlight option-specific impacts (volatility, "
            "flow, liquidity)."
        )

    @staticmethod
    def _openai_prompt() -> str:
        return (
            "You are Jeffrey the operator. Provide direct answers, calculations, or "
            "config help in under 200 words. Return tables when useful."
        )
