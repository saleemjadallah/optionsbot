"""Lightweight helpers that LLMs can call for quick heuristics."""

from __future__ import annotations

from typing import Dict


def classify_vol_environment(iv_rank: float) -> str:
    if iv_rank >= 0.7:
        return "high"
    if iv_rank <= 0.3:
        return "low"
    return "normal"


def recommended_strategy(iv_rank: float) -> str:
    bucket = classify_vol_environment(iv_rank)
    mapping: Dict[str, str] = {
        "high": "Consider premium-selling structures like iron condors or strangles.",
        "normal": "Directional verticals or calendars balance risk vs. reward.",
        "low": "Debit spreads or long gamma plays (straddles/strangles) shine in low IV.",
    }
    return mapping[bucket]
