"""Collects trading context for the AI assistant."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

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
        try:
            recent_trades = self.api.get_recent_trades(limit=8)
        except Exception:
            recent_trades = []
        try:
            saved_strategies = self.api.get_saved_strategies(limit=6)
        except Exception:
            saved_strategies = []

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
                "recent_trades": recent_trades,
                "favorite_strategies": saved_strategies,
                "favorite_symbols": [
                    entry.get("symbol")
                    for entry in saved_strategies
                    if entry.get("symbol")
                ],
                "recent_trade_digest": self._summarize_trades(recent_trades),
                "favorite_strategy_digest": self._summarize_strategies(saved_strategies),
            }
        )
        return context

    @staticmethod
    def _summarize_trades(trades: List[Dict[str, Any]]) -> str:
        """Return a short sentence covering the latest trades for context prompts."""
        if not trades:
            return ""
        snippets = []
        for trade in trades[:3]:
            symbol = trade.get("symbol", "??")
            action = (trade.get("action") or "").split()[0] if trade.get("action") else "Trade"
            qty = trade.get("quantity")
            price = trade.get("price")
            pnl = trade.get("realized_pnl")
            size_prefix = ""
            if qty:
                try:
                    qty_float = float(qty)
                    if qty_float.is_integer():
                        size_prefix = f"{int(qty_float)} "
                    else:
                        size_prefix = f"{qty_float:.1f} "
                except (TypeError, ValueError):
                    size_prefix = ""
            phrases = [f"{action.title()} {size_prefix}{symbol}".strip()]
            if price:
                phrases.append(f"@ {price:.2f}")
            if pnl:
                phrases.append(f"PNL {pnl:+.0f}")
            snippets.append(" ".join(phrases))
        return "; ".join(snippets)

    @staticmethod
    def _summarize_strategies(strategies: List[Dict[str, Any]]) -> str:
        """Create a compact summary of the user's pinned strategy ideas."""
        if not strategies:
            return ""
        parts = []
        for item in strategies[:3]:
            symbol = item.get("symbol", "??")
            strat = item.get("strategy") or "Idea"
            signal = item.get("signal")
            detail = f"{symbol} {strat}"
            if signal:
                detail += f" ({signal})"
            parts.append(detail)
        return "; ".join(parts)
