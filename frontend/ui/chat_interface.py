"""Streamlit chat interface for Jeffrey the trading assistant."""

from __future__ import annotations

import streamlit as st

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from ai_assistant import JeffreyAssistant
from frontend.utils.api_client import TradingBotAPI


def get_assistant() -> JeffreyAssistant:
    if "jeffrey_assistant" not in st.session_state:
        api_client = TradingBotAPI()
        st.session_state.jeffrey_assistant = JeffreyAssistant(api_client=api_client)
    return st.session_state.jeffrey_assistant


def render_chat_interface() -> None:
    st.header("🧠 Jeffrey – AI Trading Partner")
    st.caption("Routes each question to Anthropic, Perplexity, or OpenAI automatically.")

    assistant = get_assistant()
    if "jeffrey_session" not in st.session_state:
        st.session_state.jeffrey_session = assistant.get_session_id(None)

    session_id = st.session_state.jeffrey_session

    history = assistant.history.get(session_id).messages
    for message in history:
        with st.chat_message(message.role):
            st.markdown(message.content)

    prompt = st.chat_input("Ask Jeffrey anything about the system, markets, or risk...")
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Jeffrey is thinking..."):
                response = assistant.ask(session_id, prompt)
                st.markdown(response.text)
                st.caption(
                    f"Model: {response.model} • Intent: {response.query_type} • Confidence: {response.confidence:.0%}"
                )

    if st.button("Reset Conversation", type="secondary"):
        assistant.history.reset(session_id)
        st.rerun()
