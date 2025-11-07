"""High-level orchestrator for Jeffrey, the multi-model trading assistant."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Dict, Optional, List

from ai_assistant.context_manager import TradingContextManager
from ai_assistant.conversation_history import ConversationHistory
from ai_assistant.model_router import ModelRouter, QueryClassification
from ai_assistant.prompt_templates import PromptLibrary
from ai_assistant.model_clients.claude_client import ClaudeClient
from ai_assistant.model_clients.perplexity_client import PerplexityClient
from ai_assistant.model_clients.openai_client import OpenAIClient


@dataclass
class AssistantResponse:
    """Normalized response structure returned by Jeffrey."""

    text: str
    model: str
    query_type: str
    confidence: float
    reasoning: str
    metadata: Dict[str, str]


class JeffreyAssistant:
    """Routes user prompts to the best-fitting AI model and tracks history."""

    def __init__(self, api_client=None) -> None:
        self.context_manager = TradingContextManager(api_client=api_client)
        self.history = ConversationHistory()
        self.router = ModelRouter()
        self.prompts = PromptLibrary()
        self.clients = {
            "claude": ClaudeClient(),
            "perplexity": PerplexityClient(),
            "openai": OpenAIClient(),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_session_id(self, session_id: Optional[str]) -> str:
        if session_id:
            return session_id
        return self.history.create_session()

    def ask(self, session_id: str, query: str) -> AssistantResponse:
        """Synchronous wrapper for Streamlit/UI callers."""

        return asyncio.run(self._ask_async(session_id, query))

    async def _ask_async(self, session_id: str, query: str) -> AssistantResponse:
        context_snapshot = self.context_manager.build_context()
        classification = self.router.classify_query(
            query,
            context={"last_query_type": self.history.last_query_type(session_id)},
        )

        self.history.append_user_message(session_id, query)

        model_key = classification.preferred_model
        client = self.clients.get(model_key)
        if client is None:
            text = (
                "Jeffrey couldn't reach the %s model because it isn't configured. "
                "Please check your API keys." % model_key
            )
            response_text = text
        else:
            response_text = await self._invoke_model(
                client=client,
                model_key=model_key,
                session_id=session_id,
                query=query,
                context_snapshot=context_snapshot,
                classification=classification,
            )

        self.history.append_assistant_message(
            session_id,
            response_text,
            query_type=classification.query_type.name,
        )

        return AssistantResponse(
            text=response_text,
            model=model_key,
            query_type=classification.query_type.name,
            confidence=classification.confidence,
            reasoning=classification.reasoning,
            metadata={
                "keywords": ", ".join(classification.keywords) or "n/a",
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _invoke_model(
        self,
        client,
        model_key: str,
        session_id: str,
        query: str,
        context_snapshot: Dict,
        classification: QueryClassification,
    ) -> str:
        """Send the query to the selected model and normalize the response."""

        system_prompt = self.prompts.get_prompt(model_key)
        messages = self.history.format_for_model(session_id)

        if model_key == "claude":
            chunks: List[str] = []
            async for piece in client.chat(
                messages=messages,
                system_prompt=system_prompt,
                trading_context=context_snapshot,
                stream=False,
            ):
                chunks.append(piece)
            return "".join(chunks).strip()

        if model_key == "perplexity":
            payload = await client.chat(
                query=query,
                system_prompt=system_prompt,
                trading_context=context_snapshot,
            )
            if payload.get("error"):
                return payload["answer"]
            answer = payload.get("answer") or "No answer received."
            sources = payload.get("sources") or []
            if sources:
                formatted_sources = "\n\nSources:\n" + "\n".join(sources)
            else:
                formatted_sources = ""
            return f"{answer}{formatted_sources}"

        if model_key == "openai":
            payload = messages
            if not payload or payload[0].get("role") != "system":
                payload = [{"role": "system", "content": system_prompt}] + payload
            result = await client.chat(messages=payload)
            content = result.get("content") or "No response received."
            return content.strip()

        return "This query type is not supported yet."
