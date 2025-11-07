"""Universe scanning utilities for the ensemble service."""

from __future__ import annotations

import asyncio
import math
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import HTTPException

from strategies.model_ensemble import ModelEnsemble, StrategyType

from .schemas import EnsembleIdea, OptionRecord, UniverseScanRequest


class UniverseScanner:
    """Wraps the ModelEnsemble to transform option data into actionable ideas."""

    STRIKE_SPAN = 0.15
    MAX_OPTIONS = 180

    def __init__(self, risk_level: str = "moderate"):
        self.risk_level = risk_level
        self.ensemble = ModelEnsemble(risk_level=risk_level)

    async def scan(self, request: UniverseScanRequest) -> List[EnsembleIdea]:
        """Run ensemble analysis across the provided universe."""
        option_frames = {}
        lookup_records: Dict[str, List[Dict[str, Any]]] = {}

        for symbol in request.symbols:
            records = request.option_chains.get(symbol, [])
            if not records:
                continue
            frame, dict_records = self._option_records_to_frame(records)
            if frame.empty:
                continue
            option_frames[symbol] = frame
            lookup_records[symbol] = dict_records

        if not option_frames:
            return []

        market_frames = {
            symbol: self._market_history_to_frame(request.market_data.get(symbol))
            for symbol in option_frames
        }

        predictions = await self._invoke_ensemble(option_frames, market_frames)

        ideas: List[EnsembleIdea] = []
        for prediction in predictions:
            symbol = getattr(prediction, "symbol", None)
            if not symbol or symbol not in lookup_records:
                continue

            if abs(getattr(prediction, "edge_magnitude", 0.0)) < request.min_edge:
                continue
            if getattr(prediction, "confidence_score", 0.0) < request.min_confidence:
                continue

            idea_payload = self._translate_prediction(
                prediction,
                lookup_records[symbol],
            )
            if idea_payload:
                ideas.append(EnsembleIdea(**idea_payload))

        return ideas

    async def _invoke_ensemble(
        self,
        option_frames: Dict[str, pd.DataFrame],
        market_frames: Dict[str, pd.DataFrame],
    ):
        """Run the asynchronous ensemble call with a fresh event loop if needed."""
        coroutine = self.ensemble.analyze_universe(option_frames, market_frames)
        try:
            return await coroutine
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(coroutine)
            finally:
                loop.close()

    @staticmethod
    def _option_records_to_frame(records: Iterable[OptionRecord]) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """Convert option records into a pandas frame and list of dicts."""
        rows: List[Dict[str, Any]] = []
        dict_records: List[Dict[str, Any]] = []

        for rec in records:
            if rec.mid is None:
                mid = UniverseScanner._compute_mid(rec.bid, rec.ask)
            else:
                mid = rec.mid

            row = {
                "symbol": rec.symbol,
                "strike": float(rec.strike),
                "expiration": rec.expiration.isoformat(),
                "option_type": rec.option_type.lower(),
                "bid": float(rec.bid) if rec.bid is not None else None,
                "ask": float(rec.ask) if rec.ask is not None else None,
                "mid": float(mid) if mid is not None else None,
                "days_to_expiration": rec.days_to_expiration,
            }
            rows.append(row)
            dict_records.append(row.copy())

        frame = pd.DataFrame(rows)
        return frame, dict_records

    @staticmethod
    def _market_history_to_frame(history) -> pd.DataFrame:
        """Convert market history to DataFrame expected by ensemble."""
        if history is None:
            # fallback synthetic history
            closes = [100.0 + math.sin(idx / 4) for idx in range(30)]
            volumes = [1_000_000 for _ in range(30)]
        else:
            closes = history.close
            if history.volume:
                volumes = history.volume
            else:
                avg = float(np.mean(history.close)) if history.close else 1.0
                volumes = [avg * 1_000 for _ in history.close]

        return pd.DataFrame({"close": closes, "volume": volumes})

    @staticmethod
    def _compute_mid(bid: Optional[float], ask: Optional[float]) -> Optional[float]:
        """Compute midpoint price with reasonable fallback."""
        if bid is not None and ask is not None and bid > 0 and ask > 0:
            return (bid + ask) / 2.0
        if bid and bid > 0:
            return bid
        if ask and ask > 0:
            return ask
        return None

    def _translate_prediction(
        self,
        prediction: Any,
        option_records: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Convert ensemble prediction into fully fleshed out trade idea."""
        mapping = {
            StrategyType.DIRECTIONAL: self._build_directional_vertical,
            StrategyType.VOLATILITY_ARBITRAGE: self._build_iron_condor,
            StrategyType.GAMMA_SCALPING: self._build_long_straddle,
            StrategyType.DISPERSION: self._build_long_strangle,
            StrategyType.MARKET_MAKING: self._build_iron_butterfly,
        }

        builder = mapping.get(getattr(prediction, "recommended_strategy", None))
        if not builder:
            return None

        idea = builder(prediction, option_records)
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
        idea["symbol"] = getattr(prediction, "symbol", "")
        return idea

    # ------------------------------------------------------------------
    # Trade translation helpers (mirrors frontend logic)
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_expiration_records(
        records: List[Dict[str, Any]],
        expiration: str,
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
    def _find_option_near_target(options: List[Dict[str, Any]], target: float) -> Optional[Dict[str, Any]]:
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
    def _find_otm_call(calls: List[Dict[str, Any]], underlying_price: float) -> Optional[Dict[str, Any]]:
        otm = [row for row in calls if float(row.get("strike", 0.0)) >= underlying_price]
        if otm:
            return min(otm, key=lambda row: float(row.get("strike", 0.0)))
        if calls:
            return max(calls, key=lambda row: float(row.get("strike", 0.0)))
        return None

    @staticmethod
    def _find_otm_put(puts: List[Dict[str, Any]], underlying_price: float) -> Optional[Dict[str, Any]]:
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

    def _build_directional_vertical(
        self,
        prediction: Any,
        option_records: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        expiration = getattr(prediction, "expiry", "")
        option_type = str(getattr(prediction, "option_type", "call")).lower()
        strike = float(getattr(prediction, "strike", 0.0))
        underlying_price = float(getattr(prediction, "market_price", 0.0) or 0.0)

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
            strategy_label = "Ensemble • Bull Call Vertical" if buy_primary else "Ensemble • Bear Call Credit Spread"
            category = "Directional" if buy_primary else "Premium Capture"
        else:
            partner = self._find_partner_option(puts, strike, prefer_higher=False)
            if not partner:
                return None
            buy_primary = edge_positive
            long_leg = base if buy_primary else partner
            short_leg = partner if buy_primary else base
            strategy_label = "Ensemble • Bear Put Vertical" if buy_primary else "Ensemble • Bull Put Credit Spread"
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

        summary = (
            f"{'Buy' if buy_primary else 'Sell'} {call_put} {long_leg.get('strike')} "
            f"vs {'sell' if buy_primary else 'buy'} {call_put} {short_leg.get('strike')} "
            f"for ~${net:.2f} {trade_type.lower()}."
        )

        return self._compose_idea(
            strategy_label,
            signal,
            rationale,
            summary,
            {
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
            },
            [
                {"instrument-type": "Equity Option", "symbol": long_leg.get("symbol", ""), "action": "Buy to Open", "quantity": 1},
                {"instrument-type": "Equity Option", "symbol": short_leg.get("symbol", ""), "action": "Sell to Open", "quantity": 1},
            ],
            net,
            category,
        )

    def _build_iron_condor(
        self,
        prediction: Any,
        option_records: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        expiration = getattr(prediction, "expiry", "")
        underlying_price = float(getattr(prediction, "market_price", 0.0) or 0.0)
        calls, puts = self._collect_expiration_records(option_records, expiration)
        if not calls or not puts:
            return None

        short_call = self._find_otm_call(calls, underlying_price)
        short_put = self._find_otm_put(puts, underlying_price)
        if not short_call or not short_put:
            return None

        long_call = self._find_partner_option(calls, float(short_call.get("strike", 0.0)), prefer_higher=True)

