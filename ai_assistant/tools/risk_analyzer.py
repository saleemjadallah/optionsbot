"""Risk helpers for Jeffrey."""

from __future__ import annotations

from typing import Dict


def classify_risk(delta: float, theta: float, gross_exposure: float) -> Dict[str, str]:
    assessment = []
    if abs(delta) > 500:
        assessment.append("High directional exposure")
    if theta < -250:
        assessment.append("Large negative theta bleed")
    if gross_exposure > 1_000_000:
        assessment.append("Significant gross notional deployed")
    if not assessment:
        assessment.append("Risk posture within normal bounds")
    return {
        "summary": ", ".join(assessment),
        "delta": f"{delta:+.1f}",
        "theta": f"{theta:+.1f}",
        "gross": f"${gross_exposure:,.0f}",
    }
