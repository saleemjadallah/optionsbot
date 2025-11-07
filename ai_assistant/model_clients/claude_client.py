"""Anthropic Claude client with context-aware prompts."""

from __future__ import annotations

import os
from typing import Any, AsyncGenerator, Dict, List, Optional

try:
    import anthropic
except ImportError:  # pragma: no cover - dependency optional in some deployments
    anthropic = None  # type: ignore


class ClaudeClient:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model or os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20240620")
        self.max_tokens = 6000
        self.temperature = 0.7
        self.client = (
            anthropic.AsyncAnthropic(api_key=self.api_key) if (anthropic and self.api_key) else None
        )

    async def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str,
        trading_context: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> AsyncGenerator[str, None]:
        if not self.client:
            yield "Claude client is not configured."
            return

        enhanced_system = self._enhance_system(system_prompt, trading_context or {})

        if stream:
            async with self.client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=enhanced_system,
                messages=messages,
            ) as events:
                async for text in events.text_stream:
                    yield text
            return

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=enhanced_system,
            messages=messages,
        )
        yield response.content[0].text

    def _enhance_system(self, prompt: str, context: Dict[str, Any]) -> str:
        context_lines = [
            f"Portfolio Value: ${context.get('portfolio_value', 0):,.2f}",
            f"Available Capital: ${context.get('available_capital', 0):,.2f}",
            f"Positions: {context.get('position_count', 0)}",
            f"Delta: {context.get('delta', 0):.2f}",
            f"Gamma: {context.get('gamma', 0):.4f}",
            f"Theta: {context.get('theta', 0):.2f}",
            f"Vega: {context.get('vega', 0):.2f}",
        ]
        return f"{prompt}\n\nCurrent Context:\n" + "\n".join(context_lines)
