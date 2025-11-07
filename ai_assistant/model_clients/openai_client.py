"""OpenAI client used for fast operational responses."""

from __future__ import annotations

import os
from typing import Dict, List, Optional

try:
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover - optional dependency
    AsyncOpenAI = None  # type: ignore


class OpenAIClient:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_MODEL", "o4-mini")
        self.client = AsyncOpenAI(api_key=self.api_key) if (AsyncOpenAI and self.api_key) else None

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        tools: Optional[List[Dict]] = None,
    ) -> Dict:
        if not self.client:
            return {"content": "OpenAI client is not configured.", "tool_calls": None, "finish_reason": "error"}

        params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"

        response = await self.client.chat.completions.create(**params)
        choice = response.choices[0]
        return {
            "content": choice.message.content,
            "tool_calls": choice.message.tool_calls,
            "finish_reason": choice.finish_reason,
        }
