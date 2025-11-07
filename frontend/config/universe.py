"""
Universe configuration for proactive strategy scans.

Defines the default symbol universe and helpers to parse overrides from
environment variables or user input provided through the Streamlit UI.
"""

from __future__ import annotations

import os
from typing import Iterable, List


DEFAULT_UNIVERSE: List[str] = [
    "SPY",
    "QQQ",
    "IWM",
    "AAPL",
    "MSFT",
    "TSLA",
    "NVDA",
    "AMD",
    "GOOGL",
    "XLK",
]


def _clean_symbol(symbol: str) -> str:
    """Normalize a user supplied ticker string."""
    return "".join(ch for ch in symbol.upper() if ch.isalnum() or ch in {".", "-"})


def parse_symbol_list(values: Iterable[str]) -> List[str]:
    """
    Convert an iterable of candidate strings into a de-duplicated list of tickers.
    Empty entries are ignored and output order matches the first occurrence.
    """
    seen = set()
    cleaned: List[str] = []

    for raw in values:
        symbol = _clean_symbol(raw.strip())
        if not symbol or symbol in seen:
            continue
        cleaned.append(symbol)
        seen.add(symbol)

    return cleaned


def load_universe_from_env() -> List[str]:
    """
    Read default universe overrides from environment variables.

    Supports comma-separated values via UNIVERSE_SYMBOLS or WATCHLIST_SYMBOLS
    (the latter is kept for compatibility with earlier docs).
    """
    env_value = os.getenv("UNIVERSE_SYMBOLS") or os.getenv("WATCHLIST_SYMBOLS")
    if not env_value:
        return []

    candidates = [piece.strip() for piece in env_value.split(",")]
    return parse_symbol_list(candidates)


def get_default_universe() -> List[str]:
    """
    Return the default universe merged with environment overrides.

    Environment variables take precedence; if none are defined the baked in
    DEFAULT_UNIVERSE list is used.
    """
    env_symbols = load_universe_from_env()
    if env_symbols:
        return env_symbols
    return list(DEFAULT_UNIVERSE)
