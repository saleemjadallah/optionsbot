"""
Strategy recommendation engine powered by live Tastytrade data.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

import pandas as pd

from tastytrade.instruments import Option, OptionType, get_option_chain
from tastytrade.market_data import InstrumentType, MarketData, get_market_data

from utils.tastytrade_auth import TastytradeAuthManager

BACKEND_ROOT = Path(__file__).resolve().parents[2] / "backend"
ENABLE_FRONTEND_ENSEMBLE = os.getenv("ENABLE_FRONTEND_ENSEMBLE", "").strip().lower() in {"1", "true", "yes"}

if ENABLE_FRONTEND_ENSEMBLE and BACKEND_ROOT.exists() and str(BACKEND_ROOT) not in sys.path:
    sys.path.append(str(BACKEND_ROOT))

if ENABLE_FRONTEND_ENSEMBLE:
    try:
        from strategies.model_ensemble import ModelEnsemble, StrategyType  # type: ignore
    except Exception:  # pragma: no cover - backend optional
        ModelEnsemble = None  # type: ignore
        StrategyType = None  # type: ignore
else:
    ModelEnsemble = None  # type: ignore
    StrategyType = None  # type: ignore


@dataclass
class UnderlyingExposure:
    symbol: str
    long_shares: float = 0.0
    long_cost: float = 0.0
    short_shares: float = 0.0
    net_quantity: float = 0.0
    net_delta: float = 0.0
    net_gamma: float = 0.0
    net_theta: float = 0.0
    net_vega: float = 0.0
    unrealized_pnl: float = 0.0

    @property
    def average_price(self) -> float:
        if self.long_shares <= 0:
            return 0.0
        return self.long_cost / self.long_shares

    @property
    def gross_shares(self) -> float:
        return self.long_shares + self.short_shares

    def to_dict(self) -> Dict[str, float]:
        data = asdict(self)
        data["average_price"] = self.average_price
        data["gross_shares"] = self.gross_shares
        return data


class StrategyEngine:
    """
    Generates trade ideas (covered calls, protective puts, etc.) based on live
    account exposure and option chain data.
    """

    MIN_DTE = 7
    MAX_DTE = 45
    COVERED_CALL_OTM = 0.05
    PROTECTIVE_PUT_OTM = 0.05
    UNIVERSE_STRIKE_SPAN = 0.10
    UNIVERSE_MAX_OPTIONS = 120
    ENSEMBLE_EDGE_THRESHOLD = 0.02
    ENSEMBLE_CONFIDENCE_THRESHOLD = 0.4
    REQUEST_INTERVAL = float(os.getenv("TASTYTRADE_REQUEST_INTERVAL", "0.25"))

    def __init__(self, auth_manager: TastytradeAuthManager):
        self.auth = auth_manager
        self.session = auth_manager.get_sdk_session()
        self.chain_cache: Dict[str, Dict[date, List[Option]]] = {}
        self.option_quote_cache: Dict[str, MarketData] = {}
        self._ensemble: Optional[ModelEnsemble] = None
        self.logger = logging.getLogger(__name__)
        self.ensemble_service_url = (os.getenv("ENSEMBLE_SERVICE_URL") or "").strip()
        self.ensemble_risk_level = os.getenv("ENSEMBLE_RISK_LEVEL", "moderate")
        self._last_request_time: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_recommendations(self, positions: List[Dict]) -> List[Dict]:
        exposures = self._aggregate_exposures(positions)
        if not exposures:
            return []

        ideas: List[Dict] = []
        for exposure in exposures.values():
            try:
                underlying_quote = self._get_market_data(exposure.symbol, InstrumentType.EQUITY)
            except Exception:
                continue

            underlying_price = self._market_price(underlying_quote)
            if underlying_price <= 0:
                continue

            if exposure.net_quantity > 0:
                idea = self._covered_call_idea(exposure, underlying_price)
                if idea:
                    ideas.append(idea)

                if exposure.unrealized_pnl < 0:
                    hedge = self._protective_put_idea(exposure, underlying_price)
                    if hedge:
                        ideas.append(hedge)

            gamma_adjust = self._gamma_scalping_adjustment(exposure, underlying_price)
            if gamma_adjust:
                ideas.append(gamma_adjust)

            if exposure.net_delta < -150:
                hedge = self._protective_call_idea(exposure, underlying_price)
                if hedge:
                    ideas.append(hedge)

        ideas.sort(key=lambda x: abs(x.get("metrics", {}).get("unrealized_pnl", 0)), reverse=True)
        return ideas

    def build_universe_recommendations(self, symbols: List[str]) -> List[Dict]:
        """
        Generate proactive trade ideas by delegating to the external ensemble service
        when available. Falls back to local execution only if the full ModelEnsemble
        stack is enabled.
        """
        unique_symbols = self._normalize_symbols(symbols)
        if not unique_symbols:
            return []

        if self.ensemble_service_url:
            try:
                base_data = self._fetch_remote_option_snapshot(unique_symbols)
                if not base_data:
                    base_data = self._build_local_payload(unique_symbols)
                if not base_data.get("symbols"):
                    return []
                payload = self._compose_service_payload(base_data)
                response = requests.post(
                    self._service_endpoint("/ensemble/ideas"),
                    json=payload,
                    timeout=float(os.getenv("ENSEMBLE_SERVICE_TIMEOUT", "20")),
                )
                response.raise_for_status()
                data = response.json() or {}
                ideas = data.get("ideas", [])
                if isinstance(ideas, list):
                    return ideas
                return []
            except Exception as exc:  # pragma: no cover - network dependent
                self.logger.warning("Ensemble service request failed: %s", exc)

        if not ModelEnsemble or not StrategyType:
            return []

        return self._build_universe_recommendations_local(unique_symbols)

    def summarize_equity_exposure(self, positions: List[Dict]) -> List[Dict]:
        exposures = self._aggregate_exposures(positions)
        return [exp.to_dict() for exp in exposures.values()]

    def _normalize_symbols(self, symbols: List[str]) -> List[str]:
        unique: List[str] = []
        seen = set()
        for raw in symbols or []:
            sym = (raw or "").strip().upper()
            if not sym or sym in seen:
                continue
            unique.append(sym)
            seen.add(sym)
        return unique

    def _service_endpoint(self, path: str) -> str:
        base = self.ensemble_service_url.rstrip("/")
        path = path if path.startswith("/") else f"/{path}"
        return f"{base}{path}"

    def _compose_service_payload(self, data: Dict[str, Any]) -> Dict[str, Any]:
        payload = dict(data)
        payload["risk_level"] = self.ensemble_risk_level
        payload["min_edge"] = self.ENSEMBLE_EDGE_THRESHOLD
        payload["min_confidence"] = self.ENSEMBLE_CONFIDENCE_THRESHOLD
        return payload

    def _build_local_payload(self, symbols: List[str]) -> Dict[str, Any]:
        option_payload: Dict[str, List[Dict[str, Any]]] = {}
        market_payload: Dict[str, Dict[str, List[float]]] = {}

        for symbol in symbols:
            try:
                equity_quote = self._get_market_data(symbol, InstrumentType.EQUITY)
            except Exception as exc:
                self.logger.debug("Failed to fetch market data for %s: %s", symbol, exc)
                continue

            underlying_price = self._market_price(equity_quote)
            if underlying_price <= 0:
                continue

            chain_df = self._prepare_option_chain_dataframe(symbol, underlying_price)
            if chain_df.empty:
                continue

            option_payload[symbol] = chain_df.to_dict("records")
            history_df = self._bootstrap_market_history(equity_quote)
            market_payload[symbol] = {
                "close": [float(x) for x in history_df["close"].tolist()],
                "volume": [float(x) for x in history_df["volume"].tolist()],
            }

        return {
            "symbols": list(option_payload.keys()),
            "option_chains": option_payload,
            "market_data": market_payload,
        }

    def _fetch_remote_option_snapshot(self, symbols: List[str]) -> Optional[Dict[str, Any]]:
        if not self.ensemble_service_url:
            return None
        if not symbols:
            return None
        try:
            response = requests.post(
                self._service_endpoint("/market-data/options"),
                json={
                    "symbols": symbols,
                    "max_options": self.UNIVERSE_MAX_OPTIONS,
                    "strike_span": self.UNIVERSE_STRIKE_SPAN,
                },
                timeout=float(os.getenv("ENSEMBLE_SERVICE_TIMEOUT", "20")),
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            self.logger.warning("Market data service failed, falling back to local chains: %s", exc)
            return None

    def _build_universe_recommendations_local(self, symbols: List[str]) -> List[Dict]:
        option_chains: Dict[str, pd.DataFrame] = {}
        option_records: Dict[str, List[Dict[str, Any]]] = {}
        market_history: Dict[str, pd.DataFrame] = {}
        underlying_prices: Dict[str, float] = {}

        for symbol in symbols:
            try:
                equity_quote = self._get_market_data(symbol, InstrumentType.EQUITY)
            except Exception:
                continue

            underlying_price = self._market_price(equity_quote)
            if underlying_price <= 0:
                continue

            chain_df = self._prepare_option_chain_dataframe(symbol, underlying_price)
            if chain_df.empty:
                continue

            option_chains[symbol] = chain_df
            option_records[symbol] = chain_df.to_dict("records")
            market_history[symbol] = self._bootstrap_market_history(equity_quote)
            underlying_prices[symbol] = underlying_price

        if not option_chains:
            return []

        ensemble = self._get_ensemble()
        if not ensemble:
            return []

        try:
            predictions = self._run_async_task(
                ensemble.analyze_universe(option_chains, market_history)
            )
        except Exception as exc:
            self.logger.warning("Local ensemble execution failed: %s", exc)
            return []

        ideas: List[Dict] = []
        for prediction in predictions:
            symbol = getattr(prediction, "symbol", None)
            if not symbol or symbol not in option_records:
                continue
            if abs(getattr(prediction, "edge_magnitude", 0.0)) < self.ENSEMBLE_EDGE_THRESHOLD:
                continue
            if getattr(prediction, "confidence_score", 0.0) < self.ENSEMBLE_CONFIDENCE_THRESHOLD:
                continue

            idea = self._translate_prediction(
                prediction,
                option_records[symbol],
                underlying_prices.get(symbol, 0.0),
            )
            if idea:
                ideas.append(idea)

        ideas.sort(key=lambda item: item.get("metrics", {}).get("edge_pct", 0.0), reverse=True)
        return ideas

    # ------------------------------------------------------------------
    # Universe helpers
    # ------------------------------------------------------------------

    def _get_ensemble(self) -> Optional[ModelEnsemble]:
        if not ModelEnsemble:
            return None
        if self._ensemble is None:
            try:
                self._ensemble = ModelEnsemble(risk_level="moderate")
            except Exception:
                self._ensemble = None
        return self._ensemble

    @staticmethod
    def _run_async_task(coro):
        try:
            return asyncio.run(coro)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(coro)
            finally:
                asyncio.set_event_loop(None)
                loop.close()

    def _prepare_option_chain_dataframe(self, symbol: str, underlying_price: float) -> pd.DataFrame:
        chain = self._get_option_chain(symbol)
        if not chain:
            return pd.DataFrame()

        rows: List[Dict[str, Any]] = []
        lower_bound = underlying_price * (1 - self.UNIVERSE_STRIKE_SPAN)
        upper_bound = underlying_price * (1 + self.UNIVERSE_STRIKE_SPAN)

        filtered: List[Option] = []
        for options in chain.values():
            for opt in options:
                if opt.days_to_expiration < self.MIN_DTE or opt.days_to_expiration > self.MAX_DTE:
                    continue
                strike = float(opt.strike_price)
                if strike <= 0:
                    continue
                if strike < lower_bound or strike > upper_bound:
                    continue
                filtered.append(opt)

        if not filtered:
            return pd.DataFrame()

        filtered.sort(
            key=lambda opt: (
                opt.days_to_expiration,
                abs(float(opt.strike_price) - underlying_price),
                0 if opt.option_type == OptionType.CALL else 1,
            )
        )

        for option in filtered[: self.UNIVERSE_MAX_OPTIONS]:
            quote = self._get_option_quote(option.symbol)
            bid = float(getattr(quote, "bid", 0.0) or 0.0) if quote else 0.0
            ask = float(getattr(quote, "ask", 0.0) or 0.0) if quote else 0.0
            mid = self._option_mark(quote) if quote else 0.0
            if mid <= 0 and bid and ask:
                mid = (bid + ask) / 2.0
            elif mid <= 0:
                mid = max(bid, ask)

            rows.append(
                {
                    "symbol": option.symbol,
                    "strike": float(option.strike_price),
                    "expiration": option.expiration_date.isoformat(),
                    "option_type": "call" if option.option_type == OptionType.CALL else "put",
                    "bid": bid,
                    "ask": ask,
                    "mid": mid,
                    "days_to_expiration": option.days_to_expiration,
                }
            )

        return pd.DataFrame(rows)

    def _get_option_quote(self, option_symbol: str) -> Optional[MarketData]:
        if option_symbol in self.option_quote_cache:
            return self.option_quote_cache[option_symbol]
        try:
            quote = self._get_market_data(option_symbol, InstrumentType.EQUITY_OPTION)
        except Exception:
            quote = None
        if quote:
            self.option_quote_cache[option_symbol] = quote
        else:
            # Cache the miss briefly to avoid hammering for unavailable contracts
            self.option_quote_cache[option_symbol] = None
        return quote

    def _bootstrap_market_history(self, quote: MarketData) -> pd.DataFrame:
        price = max(self._market_price(quote), 1.0)
        base_volume = float(
            getattr(quote, "volume", None)
            or getattr(quote, "total_volume", None)
            or 1_000_000
        )

        closes = [price * (1 + 0.002 * math.sin(idx)) for idx in range(30)]
        volumes = [max(1.0, base_volume * (0.9 + 0.02 * math.cos(idx))) for idx in range(30)]

        return pd.DataFrame({"close": closes, "volume": volumes})

    def _translate_prediction(
        self,
        prediction: Any,
        option_records: List[Dict[str, Any]],
        underlying_price: float,
    ) -> Optional[Dict]:
        if not StrategyType:
            return None

        strategy = getattr(prediction, "recommended_strategy", None)
        builder_map = {
            StrategyType.DIRECTIONAL: self._build_directional_vertical,
            StrategyType.VOLATILITY_ARBITRAGE: self._build_iron_condor,
            StrategyType.GAMMA_SCALPING: self._build_long_straddle,
            StrategyType.DISPERSION: self._build_long_strangle,
            StrategyType.MARKET_MAKING: self._build_iron_butterfly,
        }

        builder = builder_map.get(strategy)
        if not builder:
            return None

        idea = builder(prediction, option_records, underlying_price)
        if not idea:
            return None

        metrics = idea.setdefault("metrics", {})
        metrics["edge_pct"] = getattr(prediction, "edge_magnitude", 0.0) * 100
        metrics["confidence_pct"] = getattr(prediction, "confidence_score", 0.0) * 100
        metrics["consensus_price"] = getattr(prediction, "consensus_price", 0.0)
        metrics["market_price"] = getattr(prediction, "market_price", 0.0)
        metrics["model_disagreement"] = getattr(prediction, "model_disagreement", 0.0)
        metrics["best_model"] = getattr(prediction, "best_model", "unknown")
        idea["origin"] = "universe"
        idea["source"] = "Universe Scan"
        return idea

    @staticmethod
    def _collect_expiration_records(
        records: List[Dict[str, Any]], expiration: str
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        calls = sorted(
            [row for row in records if row.get("option_type") == "call" and row.get("expiration") == expiration],
            key=lambda row: float(row.get("strike", 0.0)),
        )
        puts = sorted(
            [row for row in records if row.get("option_type") == "put" and row.get("expiration") == expiration],
            key=lambda row: float(row.get("strike", 0.0)),
        )
        return calls, puts

    @staticmethod
    def _find_option_record(
        records: List[Dict[str, Any]],
        expiration: str,
        option_type: str,
        strike: float,
        tolerance: float = 0.01,
    ) -> Optional[Dict[str, Any]]:
        for row in records:
            if (
                row.get("expiration") == expiration
                and row.get("option_type") == option_type
                and abs(float(row.get("strike", 0.0)) - float(strike)) <= tolerance
            ):
                return row
        return None

    @staticmethod
    def _find_partner_option(
        options: List[Dict[str, Any]],
        base_strike: float,
        prefer_higher: bool,
    ) -> Optional[Dict[str, Any]]:
        candidate = None
        best_diff = float("inf")
        for row in options:
            strike = float(row.get("strike", 0.0))
            diff = strike - base_strike
            if prefer_higher and diff > 1e-6 and diff < best_diff:
                candidate = row
                best_diff = diff
            elif not prefer_higher and diff < -1e-6 and abs(diff) < best_diff:
                candidate = row
                best_diff = abs(diff)
        return candidate

    @staticmethod
    def _find_option_near_target(
        options: List[Dict[str, Any]], target: float
    ) -> Optional[Dict[str, Any]]:
        candidate = None
        best_diff = float("inf")
        for row in options:
            strike = float(row.get("strike", 0.0))
            diff = abs(strike - target)
            if diff < best_diff:
                best_diff = diff
                candidate = row
        return candidate

    @staticmethod
    def _find_otm_call(
        calls: List[Dict[str, Any]], underlying_price: float
    ) -> Optional[Dict[str, Any]]:
        otm = [row for row in calls if float(row.get("strike", 0.0)) >= underlying_price]
        if otm:
            return min(otm, key=lambda row: float(row.get("strike", 0.0)))
        if calls:
            return max(calls, key=lambda row: float(row.get("strike", 0.0)))
        return None

    @staticmethod
    def _find_otm_put(
        puts: List[Dict[str, Any]], underlying_price: float
    ) -> Optional[Dict[str, Any]]:
        otm = [row for row in puts if float(row.get("strike", 0.0)) <= underlying_price]
        if otm:
            return max(otm, key=lambda row: float(row.get("strike", 0.0)))
        if puts:
            return min(puts, key=lambda row: float(row.get("strike", 0.0)))
        return None

    @staticmethod
    def _mid_price(row: Dict[str, Any], fallback: float = 0.0) -> float:
        mid = float(row.get("mid", 0.0) or 0.0)
        bid = float(row.get("bid", 0.0) or 0.0)
        ask = float(row.get("ask", 0.0) or 0.0)
        if mid > 0:
            return mid
        if bid > 0 and ask > 0:
            return (bid + ask) / 2.0
        if bid > 0:
            return bid
        if ask > 0:
            return ask
        return fallback

    def _compose_idea(
        self,
        symbol: str,
        strategy_label: str,
        signal: str,
        rationale: str,
        trade_summary: str,
        metrics: Dict[str, Any],
        legs: List[Dict[str, Any]],
        limit_price: float,
        category: str,
    ) -> Dict:
        return {
            "symbol": symbol,
            "suggested_strategy": strategy_label,
            "signal": signal,
            "rationale": rationale,
            "trade_idea": trade_summary,
            "metrics": metrics,
            "order_example": {
                "time-in-force": "Day",
                "order-type": "Limit",
                "price": round(max(limit_price, 0.01), 2),
                "legs": legs,
                "source": "streamlit-app",
            },
            "category": category,
        }

    def _build_directional_vertical(
        self,
        prediction: Any,
        option_records: List[Dict[str, Any]],
        underlying_price: float,
    ) -> Optional[Dict]:
        expiration = getattr(prediction, "expiry", "")
        option_type = str(getattr(prediction, "option_type", "call")).lower()
        strike = float(getattr(prediction, "strike", 0.0))

        base = self._find_option_record(option_records, expiration, option_type, strike)
        if not base:
            return None

        calls, puts = self._collect_expiration_records(option_records, expiration)
        edge_positive = getattr(prediction, "edge_magnitude", 0.0) > 0

        if option_type == "call":
            partner = self._find_partner_option(calls, strike, prefer_higher=True)
            if not partner:
                return None
            buy_primary = edge_positive
            long_leg = base if buy_primary else partner
            short_leg = partner if buy_primary else base
            strategy_label = "Universe • Bull Call Vertical" if buy_primary else "Universe • Bear Call Credit Spread"
            category = "Directional" if buy_primary else "Premium Capture"
        else:
            partner = self._find_partner_option(puts, strike, prefer_higher=False)
            if not partner:
                return None
            buy_primary = edge_positive
            long_leg = base if buy_primary else partner
            short_leg = partner if buy_primary else base
            strategy_label = "Universe • Bear Put Vertical" if buy_primary else "Universe • Bull Put Credit Spread"
            category = "Directional" if buy_primary else "Income"

        long_price = self._mid_price(long_leg, fallback=getattr(prediction, "consensus_price", 0.0))
        short_price = self._mid_price(short_leg, fallback=getattr(prediction, "market_price", 0.0))

        if buy_primary:
            net = max(long_price - short_price, 0.0)
            trade_type = "Debit"
        else:
            net = max(short_price - long_price, 0.0)
            trade_type = "Credit"

        width = abs(float(short_leg.get("strike", 0.0)) - float(long_leg.get("strike", 0.0)))
        if width <= 0 or net <= 0:
            return None

        multiplier = 100
        if buy_primary:
            max_loss = net * multiplier
            max_gain = max((width - net) * multiplier, 0.0)
        else:
            max_gain = net * multiplier
            max_loss = max((width - net) * multiplier, 0.0)

        if max_loss <= 0 and buy_primary:
            return None

        legs = [
            {
                "instrument-type": "Equity Option",
                "symbol": long_leg.get("symbol", ""),
                "action": "Buy to Open",
                "quantity": 1,
            },
            {
                "instrument-type": "Equity Option",
                "symbol": short_leg.get("symbol", ""),
                "action": "Sell to Open",
                "quantity": 1,
            },
        ]

        call_put = "Call" if option_type == "call" else "Put"
        direction = "bullish" if (option_type == "call") == buy_primary else "bearish"
        signal = (
            f"{call_put} vertical expresses a {direction} view with ensemble edge "
            f"{getattr(prediction, 'edge_magnitude', 0.0):.1%} at "
            f"{getattr(prediction, 'confidence_score', 0.0):.0%} confidence."
        )
        rationale = (
            f"Consensus price ${getattr(prediction, 'consensus_price', 0.0):.2f} vs market "
            f"${getattr(prediction, 'market_price', 0.0):.2f} supports the structure."
        )

        short_strike = float(short_leg.get("strike", 0.0))
        long_strike = float(long_leg.get("strike", 0.0))
        if option_type == "call":
            breakeven = long_strike + net if buy_primary else short_strike + net
        else:
            breakeven = long_strike - net if buy_primary else short_strike - net

        metrics = {
            "expiration": expiration,
            "contracts": 1,
            "trade_type": trade_type,
            "spread_width": width,
            "net_debit": net if buy_primary else 0.0,
            "net_credit": net if not buy_primary else 0.0,
            "max_gain": max_gain,
            "max_loss": max_loss,
            "breakeven": breakeven,
            "underlying_price": underlying_price,
        }

        summary = (
            f"{'Buy' if buy_primary else 'Sell'} {call_put} at {long_leg.get('strike')} "
            f"vs {'sell' if buy_primary else 'buy'} {call_put} at {short_leg.get('strike')} "
            f"for ~${net:.2f} {trade_type.lower()}."
        )

        return self._compose_idea(
            getattr(prediction, "symbol", ""),
            strategy_label,
            signal,
            rationale,
            summary,
            metrics,
            legs,
            net,
            category,
        )

    def _build_iron_condor(
        self,
        prediction: Any,
        option_records: List[Dict[str, Any]],
        underlying_price: float,
    ) -> Optional[Dict]:
        expiration = getattr(prediction, "expiry", "")
        calls, puts = self._collect_expiration_records(option_records, expiration)
        if not calls or not puts:
            return None

        short_call = self._find_otm_call(calls, underlying_price)
        short_put = self._find_otm_put(puts, underlying_price)
        if not short_call or not short_put:
            return None

        long_call = self._find_partner_option(calls, float(short_call.get("strike", 0.0)), prefer_higher=True)
        long_put = self._find_partner_option(puts, float(short_put.get("strike", 0.0)), prefer_higher=False)
        if not long_call or not long_put:
            return None

        short_call_price = self._mid_price(short_call, fallback=getattr(prediction, "market_price", 0.0))
        short_put_price = self._mid_price(short_put, fallback=getattr(prediction, "market_price", 0.0))
        long_call_price = self._mid_price(long_call, fallback=short_call_price * 0.5)
        long_put_price = self._mid_price(long_put, fallback=short_put_price * 0.5)

        credit = max(short_call_price + short_put_price - long_call_price - long_put_price, 0.0)
        if credit <= 0:
            return None

        call_width = float(long_call.get("strike", 0.0)) - float(short_call.get("strike", 0.0))
        put_width = float(short_put.get("strike", 0.0)) - float(long_put.get("strike", 0.0))
        max_width = max(call_width, put_width)
        if max_width <= 0:
            return None

        multiplier = 100
        max_gain = credit * multiplier
        max_loss = max((max_width - credit) * multiplier, 0.0)

        legs = [
            {"instrument-type": "Equity Option", "symbol": short_call.get("symbol", ""), "action": "Sell to Open", "quantity": 1},
            {"instrument-type": "Equity Option", "symbol": long_call.get("symbol", ""), "action": "Buy to Open", "quantity": 1},
            {"instrument-type": "Equity Option", "symbol": short_put.get("symbol", ""), "action": "Sell to Open", "quantity": 1},
            {"instrument-type": "Equity Option", "symbol": long_put.get("symbol", ""), "action": "Buy to Open", "quantity": 1},
        ]

        metrics = {
            "expiration": expiration,
            "contracts": 1,
            "net_credit": credit,
            "max_gain": max_gain,
            "max_loss": max_loss,
            "upper_breakeven": float(short_call.get("strike", 0.0)) + credit,
            "lower_breakeven": float(short_put.get("strike", 0.0)) - credit,
            "underlying_price": underlying_price,
            "spread_width_call": call_width,
            "spread_width_put": put_width,
        }

        signal = (
            f"Iron condor harvests {credit:.2f} credit where ensemble flags "
            f"{getattr(prediction, 'model_disagreement', 0.0):.2f} model disagreement."
        )
        rationale = (
            "Short volatility stance seeks to capture mean reversion while defined wings cap risk."
        )
        summary = (
            f"Sell {getattr(prediction, 'symbol', '')} {expiration} iron condor collecting ~${credit:.2f} credit."
        )

        return self._compose_idea(
            getattr(prediction, "symbol", ""),
            "Universe • Iron Condor (Vol Arb)",
            signal,
            rationale,
            summary,
            metrics,
            legs,
            credit,
            "Volatility",
        )

    def _build_long_straddle(
        self,
        prediction: Any,
        option_records: List[Dict[str, Any]],
        underlying_price: float,
    ) -> Optional[Dict]:
        expiration = getattr(prediction, "expiry", "")
        calls, puts = self._collect_expiration_records(option_records, expiration)
        if not calls or not puts:
            return None

        call = self._find_option_near_target(calls, underlying_price)
        put = self._find_option_near_target(puts, underlying_price)
        if not call or not put:
            return None

        strike_call = float(call.get("strike", 0.0))
        strike_put = float(put.get("strike", 0.0))
        strike_center = (strike_call + strike_put) / 2.0

        call_price = self._mid_price(call, fallback=getattr(prediction, "market_price", 0.0))
        put_price = self._mid_price(put, fallback=getattr(prediction, "market_price", 0.0))
        debit = call_price + put_price
        if debit <= 0:
            return None

        legs = [
            {"instrument-type": "Equity Option", "symbol": call.get("symbol", ""), "action": "Buy to Open", "quantity": 1},
            {"instrument-type": "Equity Option", "symbol": put.get("symbol", ""), "action": "Buy to Open", "quantity": 1},
        ]

        metrics = {
            "expiration": expiration,
            "contracts": 1,
            "net_debit": debit,
            "max_loss": debit * 100,
            "max_gain": "Unlimited",
            "upper_breakeven": strike_call + debit,
            "lower_breakeven": strike_put - debit,
            "underlying_price": underlying_price,
        }

        signal = (
            f"Long straddle targets gamma capture with {getattr(prediction, 'edge_magnitude', 0.0):.1%} edge."
        )
        rationale = (
            "Owning both call and put keeps delta flexible while profiting from movement or volatility expansion."
        )
        summary = (
            f"Buy call {strike_call:.2f} and put {strike_put:.2f} for ~${debit:.2f} debit."
        )

        return self._compose_idea(
            getattr(prediction, "symbol", ""),
            "Universe • Long Straddle (Gamma)",
            signal,
            rationale,
            summary,
            metrics,
            legs,
            debit,
            "Gamma Scalping",
        )

    def _build_long_strangle(
        self,
        prediction: Any,
        option_records: List[Dict[str, Any]],
        underlying_price: float,
    ) -> Optional[Dict]:
        expiration = getattr(prediction, "expiry", "")
        calls, puts = self._collect_expiration_records(option_records, expiration)
        if not calls or not puts:
            return None

        call = self._find_otm_call(calls, underlying_price)
        put = self._find_otm_put(puts, underlying_price)
        if not call or not put:
            return None

        call_price = self._mid_price(call, fallback=getattr(prediction, "market_price", 0.0))
        put_price = self._mid_price(put, fallback=getattr(prediction, "market_price", 0.0))
        debit = call_price + put_price
        if debit <= 0:
            return None

        legs = [
            {"instrument-type": "Equity Option", "symbol": call.get("symbol", ""), "action": "Buy to Open", "quantity": 1},
            {"instrument-type": "Equity Option", "symbol": put.get("symbol", ""), "action": "Buy to Open", "quantity": 1},
        ]

        strike_call = float(call.get("strike", 0.0))
        strike_put = float(put.get("strike", 0.0))

        metrics = {
            "expiration": expiration,
            "contracts": 1,
            "net_debit": debit,
            "max_loss": debit * 100,
            "max_gain": "Unlimited",
            "upper_breakeven": strike_call + debit,
            "lower_breakeven": strike_put - debit,
            "underlying_price": underlying_price,
        }

        signal = (
            f"Long strangle targets dispersion payoff with {getattr(prediction, 'confidence_score', 0.0):.0%} confidence."
        )
        rationale = (
            "Out-of-the-money wings profit from outsized moves while keeping cost contained."
        )
        summary = (
            f"Buy OTM call {strike_call:.2f} and put {strike_put:.2f} for ~${debit:.2f} debit."
        )

        return self._compose_idea(
            getattr(prediction, "symbol", ""),
            "Universe • Long Strangle (Dispersion)",
            signal,
            rationale,
            summary,
            metrics,
            legs,
            debit,
            "Dispersion",
        )

    def _build_iron_butterfly(
        self,
        prediction: Any,
        option_records: List[Dict[str, Any]],
        underlying_price: float,
    ) -> Optional[Dict]:
        expiration = getattr(prediction, "expiry", "")
        calls, puts = self._collect_expiration_records(option_records, expiration)
        if not calls or not puts:
            return None

        center_call = self._find_option_near_target(calls, underlying_price)
        center_put = self._find_option_near_target(puts, underlying_price)
        if not center_call or not center_put:
            return None

        center_strike = (float(center_call.get("strike", 0.0)) + float(center_put.get("strike", 0.0))) / 2.0
        short_call = self._find_option_near_target(calls, center_strike)
        short_put = self._find_option_near_target(puts, center_strike)
        if not short_call or not short_put:
            return None

        long_call = self._find_partner_option(calls, float(short_call.get("strike", 0.0)), prefer_higher=True)
        long_put = self._find_partner_option(puts, float(short_put.get("strike", 0.0)), prefer_higher=False)
        if not long_call or not long_put:
            return None

        short_call_price = self._mid_price(short_call, fallback=getattr(prediction, "market_price", 0.0))
        short_put_price = self._mid_price(short_put, fallback=getattr(prediction, "market_price", 0.0))
        long_call_price = self._mid_price(long_call, fallback=short_call_price * 0.5)
        long_put_price = self._mid_price(long_put, fallback=short_put_price * 0.5)

        credit = max(short_call_price + short_put_price - long_call_price - long_put_price, 0.0)
        if credit <= 0:
            return None

        wing_width_call = float(long_call.get("strike", 0.0)) - float(short_call.get("strike", 0.0))
        wing_width_put = float(short_put.get("strike", 0.0)) - float(long_put.get("strike", 0.0))
        max_width = max(wing_width_call, wing_width_put)
        if max_width <= 0:
            return None

        multiplier = 100
        max_gain = credit * multiplier
        max_loss = max((max_width - credit) * multiplier, 0.0)

        legs = [
            {"instrument-type": "Equity Option", "symbol": short_call.get("symbol", ""), "action": "Sell to Open", "quantity": 1},
            {"instrument-type": "Equity Option", "symbol": long_call.get("symbol", ""), "action": "Buy to Open", "quantity": 1},
            {"instrument-type": "Equity Option", "symbol": short_put.get("symbol", ""), "action": "Sell to Open", "quantity": 1},
            {"instrument-type": "Equity Option", "symbol": long_put.get("symbol", ""), "action": "Buy to Open", "quantity": 1},
        ]

        metrics = {
            "expiration": expiration,
            "contracts": 1,
            "net_credit": credit,
            "max_gain": max_gain,
            "max_loss": max_loss,
            "upper_breakeven": float(short_call.get("strike", 0.0)) + credit,
            "lower_breakeven": float(short_put.get("strike", 0.0)) - credit,
            "underlying_price": underlying_price,
            "wing_width_call": wing_width_call,
            "wing_width_put": wing_width_put,
        }

        signal = (
            f"Iron butterfly pins price action around {center_strike:.2f} to capture market-making edge."
        )
        rationale = (
            "Collect premium near the money while defined wings enforce disciplined risk limits."
        )
        summary = (
            f"Sell ATM call/put and buy wings for ~${credit:.2f} credit."
        )

        return self._compose_idea(
            getattr(prediction, "symbol", ""),
            "Universe • Iron Butterfly (Market Making)",
            signal,
            rationale,
            summary,
            metrics,
            legs,
            credit,
            "Market Making",
        )

    # ------------------------------------------------------------------
    # Idea builders
    # ------------------------------------------------------------------

    def _covered_call_idea(self, exposure: UnderlyingExposure, underlying_price: float) -> Optional[Dict]:
        option = self._select_option(
            symbol=exposure.symbol,
            option_type=OptionType.CALL,
            target_price=underlying_price * (1 + self.COVERED_CALL_OTM),
        )
        if not option:
            return None

        quote = self._get_market_data(option.symbol, InstrumentType.EQUITY_OPTION)
        premium = self._option_mark(quote)
        if premium <= 0:
            return None

        shares_per_contract = max(1, int(option.shares_per_contract))
        shares_available = max(0.0, exposure.long_shares)
        contracts = int(shares_available // shares_per_contract)
        if contracts <= 0:
            return None
        credit = premium * shares_per_contract * contracts
        yield_pct = credit / (underlying_price * shares_per_contract * contracts)
        annualized_yield = yield_pct * (365 / option.days_to_expiration)

        return {
            "symbol": exposure.symbol,
            "suggested_strategy": "Ensemble • Covered Call Income",
            "signal": "Harvest premium against long shares",
            "rationale": (
                f"Long {int(exposure.net_quantity)} shares with unrealized P&L "
                f"${exposure.unrealized_pnl:+,.2f}. Target OTM call generates income."
            ),
            "trade_idea": (
                f"Sell {contracts}x {exposure.symbol} {option.expiration_date:%Y-%m-%d} "
                f"{'C' if option.option_type == OptionType.CALL else 'P'} {float(option.strike_price):.2f} "
                "to collect premium while capping upside."
            ),
            "metrics": {
                "underlying_price": underlying_price,
                "strike": float(option.strike_price),
                "expiration": option.expiration_date.isoformat(),
                "days_to_expiration": option.days_to_expiration,
                "premium": credit,
                "contracts": contracts,
                "yield_pct": yield_pct * 100,
                "annualized_yield_pct": annualized_yield * 100,
                "unrealized_pnl": exposure.unrealized_pnl,
                "net_shares": exposure.net_quantity,
                "category": "Income",
            },
            "order_example": {
                "time-in-force": "Day",
                "order-type": "Limit",
                "price": round(premium, 2),
                "legs": [
                    {
                        "instrument-type": "Equity Option",
                        "symbol": option.symbol,
                        "action": "Sell to Open",
                        "quantity": contracts,
                    }
                ],
                "source": "streamlit-app",
            },
            "category": "Income",
        }

    def _protective_put_idea(self, exposure: UnderlyingExposure, underlying_price: float) -> Optional[Dict]:
        option = self._select_option(
            symbol=exposure.symbol,
            option_type=OptionType.PUT,
            target_price=underlying_price * (1 - self.PROTECTIVE_PUT_OTM),
            prefer_otm=False,
        )
        if not option:
            return None

        quote = self._get_market_data(option.symbol, InstrumentType.EQUITY_OPTION)
        premium = self._option_mark(quote)
        if premium <= 0:
            return None

        shares_per_contract = max(1, int(option.shares_per_contract))
        contracts = max(1, math.ceil(exposure.net_quantity / shares_per_contract))
        debit = premium * shares_per_contract * contracts

        return {
            "symbol": exposure.symbol,
            "suggested_strategy": "Ensemble • Protective Put",
            "signal": "Hedge downside on losing position",
            "rationale": (
                f"Unrealized loss ${exposure.unrealized_pnl:+,.2f}. Protective put limits further drawdowns."
            ),
            "trade_idea": (
                f"Buy {contracts}x {exposure.symbol} {option.expiration_date:%Y-%m-%d} "
                f"{'P' if option.option_type == OptionType.PUT else 'C'} {float(option.strike_price):.2f} "
                "to cap downside risk."
            ),
            "metrics": {
                "underlying_price": underlying_price,
                "strike": float(option.strike_price),
                "expiration": option.expiration_date.isoformat(),
                "days_to_expiration": option.days_to_expiration,
                "cost": debit,
                "contracts": contracts,
                "breakeven": float(option.strike_price) - premium,
                "unrealized_pnl": exposure.unrealized_pnl,
                "net_shares": exposure.net_quantity,
                "category": "Hedge",
            },
            "order_example": {
                "time-in-force": "Day",
                "order-type": "Limit",
                "price": round(premium, 2),
                "legs": [
                    {
                        "instrument-type": "Equity Option",
                        "symbol": option.symbol,
                        "action": "Buy to Open",
                        "quantity": contracts,
                    }
                ],
                "source": "streamlit-app",
            },
            "category": "Hedge",
        }

    def _gamma_scalping_adjustment(self, exposure: UnderlyingExposure, underlying_price: float) -> Optional[Dict]:
        delta_shares = exposure.net_delta
        gamma_value = exposure.net_gamma

        if abs(delta_shares) > 25:
            return None
        if abs(gamma_value) < 0.5:
            return None

        shares_to_trade = int(round(-delta_shares))
        if shares_to_trade == 0:
            return None

        action = "Buy" if shares_to_trade > 0 else "Sell"

        return {
            "symbol": exposure.symbol,
            "suggested_strategy": "Gamma Scalping Adjustment",
            "signal": "Rebalance underlying to keep delta-neutral while exploiting gamma.",
            "rationale": (
                f"Net delta {delta_shares:+.1f} shares with gamma {gamma_value:+.3f}. "
                f"Adjust underlying to recenter before next scalping cycle."
            ),
            "trade_idea": (
                f"{action} {abs(shares_to_trade)} shares of {exposure.symbol} at ~${underlying_price:.2f} "
                "to return delta to neutral while maintaining gamma exposure."
            ),
            "metrics": {
                "net_delta_shares": delta_shares,
                "net_gamma": gamma_value,
                "shares_to_trade": shares_to_trade,
                "action": action,
                "unrealized_pnl": exposure.unrealized_pnl,
                "category": "Gamma Scalping",
            },
            "order_example": {
                "time-in-force": "Day",
                "order-type": "Market",
                "legs": [
                    {
                        "instrument-type": "Equity",
                        "symbol": exposure.symbol,
                        "action": action,
                        "quantity": abs(shares_to_trade),
                    }
                ],
                "source": "streamlit-app",
            },
            "category": "Gamma Scalping",
        }

    def _protective_call_idea(self, exposure: UnderlyingExposure, underlying_price: float) -> Optional[Dict]:
        option = self._select_option(
            symbol=exposure.symbol,
            option_type=OptionType.CALL,
            target_price=underlying_price * (1 + self.COVERED_CALL_OTM),
            prefer_otm=False,
        )
        if not option:
            return None

        quote = self._get_market_data(option.symbol, InstrumentType.EQUITY_OPTION)
        premium = self._option_mark(quote)
        if premium <= 0:
            return None

        shares_per_contract = max(1, int(option.shares_per_contract))
        contracts = max(1, math.ceil(-exposure.net_quantity / shares_per_contract))
        debit = premium * shares_per_contract * contracts

        return {
            "symbol": exposure.symbol,
            "suggested_strategy": "Ensemble • Protective Call",
            "signal": "Cap upside risk on short equity exposure",
            "rationale": (
                f"Short {abs(int(exposure.net_quantity))} shares. Buying calls limits losses "
                f"if {exposure.symbol} rallies sharply."
            ),
            "trade_idea": (
                f"Buy {contracts}x {exposure.symbol} {option.expiration_date:%Y-%m-%d} "
                f"{float(option.strike_price):.2f} call(s) as insurance against rallies."
            ),
            "metrics": {
                "underlying_price": underlying_price,
                "strike": float(option.strike_price),
                "expiration": option.expiration_date.isoformat(),
                "days_to_expiration": option.days_to_expiration,
                "cost": debit,
                "contracts": contracts,
                "unrealized_pnl": exposure.unrealized_pnl,
                "net_shares": exposure.net_quantity,
                "category": "Hedge",
            },
            "order_example": {
                "time-in-force": "Day",
                "order-type": "Limit",
                "price": round(premium, 2),
                "legs": [
                    {
                        "instrument-type": "Equity Option",
                        "symbol": option.symbol,
                        "action": "Buy to Open",
                        "quantity": contracts,
                    }
                ],
                "source": "streamlit-app",
            },
            "category": "Hedge",
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _aggregate_exposures(self, positions: List[Dict]) -> Dict[str, UnderlyingExposure]:
        exposures: Dict[str, UnderlyingExposure] = {}

        for pos in positions:
            instrument = str(pos.get("instrument-type", "")).lower()
            symbol = (pos.get("symbol") or "").strip()
            underlying = (pos.get("underlying-symbol") or pos.get("underlying_symbol") or "").strip()

            if instrument == "equity":
                base_symbol = symbol
            elif "option" in instrument:
                base_symbol = underlying or symbol.split()[0]
            else:
                continue

            if not base_symbol:
                continue

            exp = exposures.setdefault(base_symbol, UnderlyingExposure(symbol=base_symbol))
            qty = self._safe_float(pos.get("quantity"))
            pnl = self._safe_float(pos.get("unrealized-gain-loss"))
            delta = self._safe_float(pos.get("delta"))
            gamma = self._safe_float(pos.get("gamma"))
            theta = self._safe_float(pos.get("theta"))
            vega = self._safe_float(pos.get("vega"))
            multiplier = (
                self._safe_float(pos.get("multiplier"))
                or self._safe_float(pos.get("quantity-multiplier"))
                or self._safe_float(pos.get("shares-per-contract"))
                or 100.0
            )

            exp.unrealized_pnl += pnl

            if instrument == "equity":
                if qty > 0:
                    exp.long_shares += qty
                    avg_price = self._safe_float(pos.get("average-open-price"))
                    exp.long_cost += avg_price * qty
                elif qty < 0:
                    exp.short_shares += abs(qty)
                exp.net_quantity += qty
                exp.net_delta += qty
            else:
                exp.net_delta += delta * qty * multiplier
                exp.net_gamma += gamma * qty * multiplier
                exp.net_theta += theta * qty * multiplier
                exp.net_vega += vega * qty * multiplier

        return {k: v for k, v in exposures.items()}

    def _select_option(
        self,
        symbol: str,
        option_type: OptionType,
        target_price: float,
        prefer_otm: bool = True,
    ) -> Optional[Option]:
        chain = self._get_option_chain(symbol)
        candidates: List[Option] = []

        for options in chain.values():
            for opt in options:
                if opt.option_type != option_type:
                    continue
                if not (self.MIN_DTE <= opt.days_to_expiration <= self.MAX_DTE):
                    continue
                candidates.append(opt)

        if not candidates:
            return None

        if prefer_otm:
            filtered = [opt for opt in candidates if float(opt.strike_price) >= target_price]
            if filtered:
                candidates = filtered

        return min(
            candidates,
            key=lambda opt: (
                abs(float(opt.strike_price) - target_price),
                opt.days_to_expiration,
            ),
        )

    def _get_option_chain(self, symbol: str) -> Dict[date, List[Option]]:
        if symbol in self.chain_cache:
            return self.chain_cache[symbol]
        try:
            chain = get_option_chain(self.session, symbol)
        except Exception:
            chain = {}
        self.chain_cache[symbol] = chain
        return chain

    def _get_market_data(self, symbol: str, instrument_type: InstrumentType) -> MarketData:
        key = f"{instrument_type.value}:{symbol.upper()}"
        self._respect_rate_limit(key)
        quote = get_market_data(self.session, symbol, instrument_type)
        self._last_request_time[key] = time.monotonic()
        return quote

    def _respect_rate_limit(self, key: str) -> None:

        interval = max(0.0, self.REQUEST_INTERVAL)
        if interval == 0.0:
            return
        last = self._last_request_time.get(key)
        if last is None:
            return
        elapsed = time.monotonic() - last
        if elapsed < interval:
            time.sleep(interval - elapsed)

    @staticmethod
    def _market_price(data: MarketData) -> float:
        for attr in ("mark", "mid", "last", "close", "prev_close", "open"):
            value = getattr(data, attr, None)
            if value is not None:
                return float(value)
        return 0.0

    @staticmethod
    def _option_mark(data: MarketData) -> float:
        for attr in ("mark", "mid", "last", "ask"):
            value = getattr(data, attr, None)
            if value is not None:
                return float(value)
        bid = getattr(data, "bid", None)
        ask = getattr(data, "ask", None)
        if bid is not None and ask is not None:
            return float(bid + ask) / 2.0
        return 0.0

    @staticmethod
    def _safe_float(value: Optional[Decimal]) -> float:
        if value is None:
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
