"""
Thin client that pulls live data from the Tastytrade session used by the
Streamlit frontend. Most flows rely directly on the authenticated sandbox
(or production) account, with a light local cache for favorites when the
backend persistence layer is unavailable.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, time
from typing import Any, Dict, List, Optional

import requests
try:
    import streamlit as st
except Exception:  # pragma: no cover - streamlit not needed for tests
    st = None  # type: ignore
from config.universe import get_default_universe
from utils.favorites_store import LocalFavoriteStore
from utils.strategy_engine import StrategyEngine
from utils.tastytrade_auth import get_auth_manager

try:  # pragma: no cover - zoneinfo availability depends on python version
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore

US_EASTERN = ZoneInfo("America/New_York") if ZoneInfo else None
REGULAR_OPEN = time(9, 30)
REGULAR_CLOSE = time(16, 0)


logger = logging.getLogger(__name__)


def _get_setting(key: str) -> str:
    """Resolve configuration from env vars or Streamlit secrets."""
    value = os.getenv(key)
    if value:
        return str(value)
    if st is not None:
        secret_value = st.secrets.get(key)  # type: ignore[attr-defined]
        if secret_value is not None:
            return str(secret_value)
    return ""


def _resolve_backend_base_url() -> str:
    for key in ("ENSEMBLE_SERVICE_URL", "TRADING_BOT_API_URL"):
        candidate = _get_setting(key).strip()
        if candidate:
            return candidate.rstrip("/")
    return ""


class TradingBotAPI:
    def __init__(self):
        self.auth_manager = get_auth_manager()
        self.backend_base_url = _resolve_backend_base_url()
        self.local_favorites = LocalFavoriteStore()

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

    def get_recent_trades(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Return the most recent live trades from the Tastytrade transaction feed."""
        if limit <= 0 or not self.can_use_tastytrade():
            return []

        account = self.get_account_number()
        try:
            transactions = self.auth_manager.get_account_transactions(
                account_number=account,
                per_page=max(limit * 2, limit),
                types=["Trade"],
            )
        except Exception as exc:
            logger.debug("Failed to fetch recent trades: %s", exc)
            return []

        trades: List[Dict[str, Any]] = []
        for txn in transactions:
            normalized = self._normalize_trade(txn)
            if not normalized:
                continue
            trades.append(normalized)
            if len(trades) >= limit:
                break
        return trades

    def get_saved_strategies(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Return a lightweight view of saved favorites for quick context injection."""
        favorites = self.fetch_favorites()
        summaries: List[Dict[str, Any]] = []
        for fav in favorites:
            snapshot = fav.get("snapshot", {}) if isinstance(fav, dict) else {}
            summaries.append(
                {
                    "idea_id": fav.get("idea_id"),
                    "symbol": snapshot.get("symbol") or fav.get("symbol"),
                    "strategy": snapshot.get("suggested_strategy") or fav.get("strategy"),
                    "signal": snapshot.get("signal"),
                    "edge": (snapshot.get("metrics") or {}).get("edge_pct"),
                    "notes": snapshot.get("trade_idea") or snapshot.get("rationale"),
                }
            )
            if len(summaries) >= limit:
                break
        return summaries

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
    # Backend persistence helpers
    # ------------------------------------------------------------------

    def _backend_available(self) -> bool:
        return bool(self.backend_base_url)

    def fetch_favorites(self) -> List[Dict[str, Any]]:
        if not self.can_use_tastytrade():
            return []
        account = self.get_account_number()
        if self._backend_available():
            try:
                resp = requests.get(
                    f"{self.backend_base_url}/favorites/{account}",
                    timeout=10,
                )
                resp.raise_for_status()
                data = resp.json()
                favorites = data.get("favorites", [])
                if isinstance(favorites, list):
                    self.local_favorites.replace_all(account, favorites)
                    return favorites
            except requests.RequestException as exc:
                logger.debug("Failed to fetch favorites from backend: %s", exc)
        return self.local_favorites.list(account)

    def save_favorite(self, idea_id: str, idea: Dict[str, Any]) -> bool:
        if not self.can_use_tastytrade():
            return False
        account = self.get_account_number()
        payload = {
            "account_number": account,
            "idea_id": idea_id,
            "symbol": idea.get("symbol", ""),
            "strategy": idea.get("suggested_strategy", ""),
            "snapshot": idea,
        }
        if self._backend_available():
            try:
                resp = requests.post(
                    f"{self.backend_base_url}/favorites",
                    json=payload,
                    timeout=10,
                )
                resp.raise_for_status()
                self.local_favorites.save(account, payload)
                return True
            except requests.RequestException as exc:
                if not self._should_fallback_to_local(exc):
                    detail = self._http_error_detail(exc)
                    logger.warning("Failed to save favorite: %s", detail)
                    raise RuntimeError(detail) from exc
                logger.warning("Favorites backend unavailable, saving locally: %s", exc)
        self.local_favorites.save(account, payload)
        return True

    def delete_favorite(self, idea_id: str) -> bool:
        if not self.can_use_tastytrade():
            return False
        account = self.get_account_number()
        if self._backend_available():
            try:
                resp = requests.delete(
                    f"{self.backend_base_url}/favorites/{account}/{idea_id}",
                    timeout=10,
                )
                resp.raise_for_status()
                self.local_favorites.delete(account, idea_id)
                return True
            except requests.RequestException as exc:
                if not self._should_fallback_to_local(exc):
                    detail = self._http_error_detail(exc)
                    logger.warning("Failed to delete favorite: %s", detail)
                    raise RuntimeError(detail) from exc
                logger.warning("Favorites backend unavailable, deleting locally: %s", exc)
        return self.local_favorites.delete(account, idea_id)

    def fetch_chat_history(self, session_id: str) -> List[Dict[str, Any]]:
        if not self._backend_available() or not self.can_use_tastytrade():
            return []
        account = self.get_account_number()
        try:
            resp = requests.get(
                f"{self.backend_base_url}/chat/logs/{account}/{session_id}",
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("messages", [])
        except requests.RequestException as exc:
            logger.debug("Failed to fetch chat history: %s", exc)
            return []

    def log_chat_message(self, session_id: str, role: str, content: str) -> None:
        if not self._backend_available() or not self.can_use_tastytrade():
            return
        account = self.get_account_number()
        try:
            requests.post(
                f"{self.backend_base_url}/chat/logs",
                json={
                    "account_number": account,
                    "session_id": session_id,
                    "role": role,
                    "content": content,
                },
                timeout=10,
            ).raise_for_status()
        except requests.RequestException as exc:
            logger.debug("Failed to log chat message: %s", exc)

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

    def _normalize_trade(self, txn: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Condense a raw Tastytrade transaction into the fields Jeffrey needs."""
        txn_type = str(
            txn.get("transaction-type") or txn.get("type") or ""
        ).strip().lower()
        if txn_type != "trade":
            return None

        symbol = (
            txn.get("symbol")
            or txn.get("underlying-symbol")
            or txn.get("underlying_symbol")
        )
        action = txn.get("action") or txn.get("transaction-sub-type")
        quantity = self._safe_float(txn.get("quantity"))
        price = self._safe_float(txn.get("price"))
        value = self._safe_float(txn.get("value"))
        realized = (
            txn.get("realized-pl")
            or txn.get("realized-pnl")
            or txn.get("realized-gain-loss")
            or txn.get("realized-gain")
            or txn.get("realized-loss")
            or 0.0
        )
        fees = (
            self._safe_float(txn.get("commission"))
            + self._safe_float(txn.get("clearing-fees"))
            + self._safe_float(txn.get("regulatory-fees"))
            + self._safe_float(txn.get("other-fees"))
        )
        executed_at = txn.get("executed-at") or txn.get("transaction-date")

        return {
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "price": price,
            "value": value,
            "realized_pnl": self._safe_float(realized),
            "fees": fees,
            "instrument_type": txn.get("instrument-type"),
            "description": txn.get("description"),
            "executed_at": executed_at,
        }

    @staticmethod
    def _http_error_detail(exc: requests.RequestException) -> str:
        response = getattr(exc, "response", None)
        if response is None:
            return str(exc)
        try:
            payload = response.json()
        except ValueError:
            return response.text or str(exc)
        detail = (
            payload.get("detail")
            or payload.get("message")
            or payload.get("error")
            or payload
        )
        return str(detail)

    @staticmethod
    def _should_fallback_to_local(exc: requests.RequestException) -> bool:
        response = getattr(exc, "response", None)
        if response is None:
            return True
        return int(getattr(response, "status_code", 0)) >= 500
