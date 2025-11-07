"""Perplexity client for real-time market intelligence."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import httpx


class PerplexityClient:
    BASE_URL = "https://api.perplexity.ai/chat/completions"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        self.model = model or os.getenv("PERPLEXITY_MODEL", "pplx-70b-chat")

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
                return {
                    "answer": f"Perplexity error: {response.text}",
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
        latest = context.get("universe_ideas", [])[:3]
        highlight = "\n".join(
            f"- {idea.get('symbol')}: {idea.get('suggested_strategy')}" for idea in latest
        )
        return f"{base}\nFocus tickers:\n{highlight}"
