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
        st.session_state.jeffrey_api = api_client
        st.session_state.jeffrey_assistant = JeffreyAssistant(api_client=api_client)
    if "jeffrey_api" not in st.session_state:
        st.session_state.jeffrey_api = TradingBotAPI()
    return st.session_state.jeffrey_assistant


def render_chat_interface() -> None:
    st.header("🧠 Jeffrey – AI Trading Partner")
    st.caption("Routes each question to Anthropic, Perplexity, or OpenAI automatically.")

    assistant = get_assistant()
    api_client: TradingBotAPI = st.session_state.jeffrey_api
    if "jeffrey_session" not in st.session_state:
        st.session_state.jeffrey_session = assistant.get_session_id(None)
        st.session_state.jeffrey_chat_loaded = False

    session_id = st.session_state.jeffrey_session

    account_number = None
    if api_client.can_use_tastytrade():
        try:
            account_number = api_client.get_account_number()
        except Exception:
            account_number = None

    if account_number and not st.session_state.get("jeffrey_chat_loaded"):
        history = api_client.fetch_chat_history(session_id)
        if history:
            assistant.history.set_messages(
                session_id,
                [{"role": m["role"], "content": m["content"]} for m in history],
            )
        st.session_state.jeffrey_chat_loaded = True

    history = assistant.history.get(session_id).messages
    for message in history:
        with st.chat_message(message.role):
            st.markdown(message.content)

    prompt = st.chat_input("Ask Jeffrey anything about the system, markets, or risk...")
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        if account_number:
            api_client.log_chat_message(session_id, "user", prompt)
        with st.chat_message("assistant"):
            with st.spinner("Jeffrey is thinking..."):
                response = assistant.ask(session_id, prompt)
                st.markdown(response.text)
                st.caption(
                    f"Model: {response.model} • Intent: {response.query_type} • Confidence: {response.confidence:.0%}"
                )
        if account_number:
            api_client.log_chat_message(session_id, "assistant", response.text)

    if st.button("Reset Conversation", type="secondary"):
        assistant.history.reset(session_id)
        st.session_state.jeffrey_chat_loaded = False
        st.rerun()
