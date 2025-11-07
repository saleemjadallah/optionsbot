"""Utility helpers used by LLM tools for quick analytics."""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def summarize_series(prices: List[float]) -> Dict[str, float]:
    if not prices:
        return {"mean": 0.0, "stdev": 0.0, "change_pct": 0.0}
    arr = np.array(prices)
    return {
        "mean": float(arr.mean()),
        "stdev": float(arr.std()),
        "change_pct": float(((arr[-1] - arr[0]) / arr[0]) * 100 if arr[0] else 0.0),
    }
