"""Simple helpers for system configuration questions."""

from __future__ import annotations

from typing import Dict


def summarize_config(config: Dict) -> str:
    parts = []
    for key, value in config.items():
        parts.append(f"{key}: {value}")
    return "; ".join(parts)
