"""Perplexity client for real-time market intelligence."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import httpx


class PerplexityClient:
    BASE_URL = "https://api.perplexity.ai/chat/completions"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        default_model = "sonar-pro"
        self.model = model or os.getenv("PERPLEXITY_MODEL", default_model)

    async def chat(
        self,
        query: str,
        system_prompt: str,
        trading_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.api_key:
            return {"answer": "Perplexity API key missing.", "error": True}

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._build_system_prompt(system_prompt, trading_context)},
                {"role": "user", "content": query},
            ],
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(self.BASE_URL, json=payload, headers=headers)
            if response.status_code != 200:
                try:
                    detail = response.json()
                except ValueError:
                    detail = response.text
                return {
                    "answer": f"Perplexity error ({response.status_code}): {detail}",
                    "error": True,
                }
            data = response.json()
            return {
                "answer": data.get("choices", [{}])[0].get("message", {}).get("content", ""),
                "sources": data.get("citations", []),
                "timestamp": data.get("created"),
                "error": False,
            }

    @staticmethod
    def _build_system_prompt(base: str, context: Optional[Dict[str, Any]]) -> str:
        if not context:
            return base

        sections = [base]

        universe = context.get("universe_ideas", [])[:3]
        if universe:
            highlight = "\n".join(
                f"- {idea.get('symbol')}: {idea.get('suggested_strategy')}" for idea in universe
            )
            sections.append(f"Focus tickers:\n{highlight}")

        favorites = context.get("favorite_strategies") or []
        if favorites:
            fav_lines = "\n".join(
                f"- {entry.get('symbol')}: {entry.get('strategy')} ({entry.get('signal') or 'saved idea'})"
                for entry in favorites[:3]
            )
            sections.append(f"Saved strategies:\n{fav_lines}")

        trade_digest = context.get("recent_trade_digest")
        if trade_digest:
            sections.append(f"Recent trades to reference: {trade_digest}")

        return "\n\n".join(sections)
