"""
Thin client that pulls live data from the Tastytrade session used by the
Streamlit frontend. No mock data or backend fallbacks – everything flows
directly from the authenticated sandbox (or production) account.
"""

from __future__ import annotations

from datetime import datetime, time
from typing import Any, Dict, List, Optional

from config.universe import get_default_universe
from utils.tastytrade_auth import get_auth_manager
from utils.strategy_engine import StrategyEngine

try:  # pragma: no cover - zoneinfo availability depends on python version
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore

US_EASTERN = ZoneInfo("America/New_York") if ZoneInfo else None
REGULAR_OPEN = time(9, 30)
REGULAR_CLOSE = time(16, 0)


class TradingBotAPI:
    def __init__(self):
        self.auth_manager = get_auth_manager()

    # ------------------------------------------------------------------
    # Session helpers
    # ------------------------------------------------------------------

    def can_use_tastytrade(self) -> bool:
        """Return True when an authenticated Tastytrade session is active."""
        return self.auth_manager.is_authenticated()

    def _require_session(self) -> None:
        if not self.can_use_tastytrade():
            raise RuntimeError(
                "Tastytrade session not available. Authenticate on the Tastytrade tab."
            )

    def get_account_number(self) -> str:
        self._require_session()
        return self.auth_manager.get_account_number()

    # ------------------------------------------------------------------
    # Portfolio data
    # ------------------------------------------------------------------

    def get_portfolio_status(self) -> Dict[str, float]:
        """
        Compute high-level portfolio metrics from the live Tastytrade balance
        and positions.
        """
        self._require_session()

        account = self.auth_manager.get_account_number()
        balance = self.auth_manager.get_account_balance(account)
        positions = self.auth_manager.get_account_positions(account)

        portfolio_value = self._safe_float(balance.get("net-liquidating-value"))
        available_cash = self._safe_float(balance.get("cash-balance"))
        buying_power = self._safe_float(balance.get("equity-buying-power"))

        day_realized = self._safe_float(balance.get("current-day-realized-pl"))
        day_unrealized = self._safe_float(balance.get("current-day-unrealized-pl"))
        daily_pnl = day_realized + day_unrealized

        if daily_pnl == 0:
            daily_pnl = sum(self._safe_float(pos.get("unrealized-gain-loss")) for pos in positions)

        return {
            "account_number": account,
            "portfolio_value": portfolio_value,
            "available_cash": available_cash,
            "buying_power": buying_power,
            "daily_pnl": daily_pnl,
            "positions_count": len(positions),
        }

    def get_positions(self) -> List[Dict]:
        """Return the live positions list from Tastytrade."""
        self._require_session()
        account = self.auth_manager.get_account_number()
        return self.auth_manager.get_account_positions(account)

    # ------------------------------------------------------------------
    # Risk metrics
    # ------------------------------------------------------------------

    def get_risk_metrics(self) -> Dict[str, float]:
        """Aggregate Greeks and basic exposures from current positions."""
        positions = self.get_positions()

        total_delta = sum(self._safe_float(pos.get("delta")) for pos in positions)
        total_gamma = sum(self._safe_float(pos.get("gamma")) for pos in positions)
        total_theta = sum(self._safe_float(pos.get("theta")) for pos in positions)
        total_vega = sum(self._safe_float(pos.get("vega")) for pos in positions)
        gross_exposure = sum(abs(self._safe_float(pos.get("unrealized-gain-loss"))) for pos in positions)

        risk_score = min(
            10.0,
            max(0.0, abs(total_delta) * 2 + gross_exposure / 100000),
        )

        return {
            "total_delta": total_delta,
            "total_gamma": total_gamma,
            "total_theta": total_theta,
            "total_vega": total_vega,
            "gross_exposure": gross_exposure,
            "positions_tracked": len(positions),
            "risk_score": risk_score,
        }

    # ------------------------------------------------------------------
    # Strategy ideas & execution
    # ------------------------------------------------------------------

    def get_strategy_context(
        self,
        universe_symbols: Optional[List[str]] = None,
        edge_threshold: Optional[float] = None,
        confidence_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Return both current exposures and recommended trades."""
        self._require_session()
        try:
            positions = self.get_positions()
        except Exception:
            return {"positions": [], "ideas": [], "exposures": []}

        engine = StrategyEngine(self.auth_manager)
        if edge_threshold is not None:
            engine.ENSEMBLE_EDGE_THRESHOLD = edge_threshold
        if confidence_threshold is not None:
            engine.ENSEMBLE_CONFIDENCE_THRESHOLD = confidence_threshold
        universe = universe_symbols or get_default_universe()

        try:
            exposures = engine.summarize_equity_exposure(positions)
        except Exception:
            exposures = []

        try:
            ideas = engine.build_recommendations(positions)
        except Exception:
            ideas = []

        try:
            universe_ideas = engine.build_universe_recommendations(universe)
        except Exception:
            universe_ideas = []

        return {
            "positions": positions,
            "ideas": ideas,
            "universe_ideas": universe_ideas,
            "exposures": exposures,
            "universe_symbols": universe,
            "market_state": self._market_state(),
        }

    def get_trade_opportunities(self) -> List[Dict]:
        """Generate strategy ideas from current portfolio context."""
        context = self.get_strategy_context()
        live_ideas = context.get("ideas", [])
        scout_ideas = context.get("universe_ideas", [])
        return live_ideas + scout_ideas

    def dry_run_order(self, order: Dict) -> Dict:
        self._require_session()
        return self.auth_manager.dry_run_order(order)

    def place_order(self, order: Dict) -> Dict:
        self._require_session()
        return self.auth_manager.place_order(order)

    def get_watchlist_symbols(self, watchlist_name: Optional[str] = None) -> List[str]:
        """Fetch symbols from the user's Tastytrade watchlist."""
        self._require_session()
        try:
            return self.auth_manager.get_watchlist_symbols(watchlist_name)
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _market_state(self) -> Dict[str, Any]:
        """Classify whether we are inside regular trading hours."""
        now = datetime.now(US_EASTERN) if US_EASTERN else datetime.utcnow()
        is_weekday = now.weekday() < 5
        in_session = REGULAR_OPEN <= now.time() <= REGULAR_CLOSE
        is_open = is_weekday and in_session
        label = "On-Market Analysis" if is_open else "Off-Market Snapshot"
        basis = (
            "Live quotes captured during regular session hours."
            if is_open
            else "Quotes captured outside regular hours (uses latest available prices)."
        )
        return {
            "is_open": is_open,
            "label": label,
            "basis": basis,
            "as_of": now.isoformat(),
        }

    @staticmethod
    def _safe_float(value) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
