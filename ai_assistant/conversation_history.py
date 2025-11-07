"""Conversation storage utilities for Jeffrey."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Message:
    role: str
    content: str
    query_type: Optional[str] = None


@dataclass
class SessionHistory:
    session_id: str
    messages: List[Message] = field(default_factory=list)

    def to_chat_messages(self) -> List[Dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in self.messages]

    def last_query_type(self) -> Optional[str]:
        for message in reversed(self.messages):
            if message.role == "assistant" and message.query_type:
                return message.query_type
        return None


class ConversationHistory:
    """In-memory session store (Streamlit already runs per-user sessions)."""

    def __init__(self) -> None:
        self.sessions: Dict[str, SessionHistory] = {}

    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = SessionHistory(session_id=session_id)
        return session_id

    def get(self, session_id: str) -> SessionHistory:
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionHistory(session_id=session_id)
        return self.sessions[session_id]

    def append_user_message(self, session_id: str, content: str) -> None:
        session = self.get(session_id)
        session.messages.append(Message(role="user", content=content))

    def append_assistant_message(self, session_id: str, content: str, query_type: Optional[str]) -> None:
        session = self.get(session_id)
        session.messages.append(
            Message(role="assistant", content=content, query_type=query_type)
        )

    def format_for_model(self, session_id: str) -> List[Dict[str, str]]:
        session = self.get(session_id)
        return session.to_chat_messages()

    def reset(self, session_id: str) -> None:
        self.sessions[session_id] = SessionHistory(session_id=session_id)

    def last_query_type(self, session_id: str) -> Optional[str]:
        session = self.get(session_id)
        return session.last_query_type()
