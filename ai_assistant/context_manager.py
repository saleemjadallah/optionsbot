"""Collects trading context for the AI assistant."""

from __future__ import annotations

from typing import Any, Dict, Optional

from frontend.utils.api_client import TradingBotAPI


class TradingContextManager:
    """Pulls account, risk, and strategy data for prompt enrichment."""

    def __init__(self, api_client: Optional[TradingBotAPI] = None) -> None:
        self.api = api_client or TradingBotAPI()

    def build_context(self) -> Dict[str, Any]:
        context: Dict[str, Any] = {}
        try:
            portfolio = self.api.get_portfolio_status()
        except Exception:
            portfolio = {}
        try:
            risk = self.api.get_risk_metrics()
        except Exception:
            risk = {}
        try:
            strategy = self.api.get_strategy_context()
        except Exception:
            strategy = {}

        context.update(
            {
                "portfolio_value": portfolio.get("portfolio_value"),
                "available_capital": portfolio.get("available_cash"),
                "position_count": portfolio.get("positions_count"),
                "daily_pnl": portfolio.get("daily_pnl"),
                "delta": risk.get("total_delta"),
                "gamma": risk.get("total_gamma"),
                "theta": risk.get("total_theta"),
                "vega": risk.get("total_vega"),
                "gross_exposure": risk.get("gross_exposure"),
                "risk_score": risk.get("risk_score"),
                "universe_symbols": strategy.get("universe_symbols", []),
                "universe_ideas": strategy.get("universe_ideas", []),
            }
        )
        return context
