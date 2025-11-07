"""Market data utilities that combine Tastytrade chains with DXLink quotes."""

from __future__ import annotations

import asyncio
import os
import ssl
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from tastytrade.instruments import Option, get_option_chain
from tastytrade.market_data import InstrumentType, MarketData, get_market_data
from tastytrade.session import Session
from tastytrade.streamer import DXLinkStreamer
from tastytrade.dxfeed import Greeks, Quote, TheoPrice


def _to_float(value: Optional[Decimal]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class TastytradeSessionManager:
    """Manages a reusable Tastytrade session using username/password env vars."""

    def __init__(self) -> None:
        self.username = os.getenv("TASTYTRADE_USERNAME")
        self.password = os.getenv("TASTYTRADE_PASSWORD")
        if not self.username or not self.password:
            raise RuntimeError("TASTYTRADE_USERNAME and TASTYTRADE_PASSWORD must be set for DXLink service.")

        self.is_test = os.getenv("TASTYTRADE_SANDBOX", "true").lower() in {"1", "true", "yes"}
        self.session: Optional[Session] = None
        self.lock = asyncio.Lock()

    async def get_session(self) -> Session:
        async with self.lock:
            if self.session is None or self._session_expired(self.session):
                loop = asyncio.get_running_loop()
                self.session = await loop.run_in_executor(
                    None,
                    lambda: Session(
                        login=self.username,
                        password=self.password,
                        remember_me=True,
                        is_test=self.is_test,
                    ),
                )
        return self.session

    @staticmethod
    def _session_expired(session: Session) -> bool:
        safety_window = timedelta(minutes=5)
        return datetime.now(timezone.utc) >= (session.session_expiration - safety_window)


class DxLinkCollector:
    """Fetches DXLink Quote/Greeks/TheoPrice snapshots for a list of streamer symbols."""

    def __init__(self, timeout: float = 3.0) -> None:
        self.timeout = timeout

    async def collect(
        self,
        session: Session,
        symbols: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        if not symbols:
            return {}

        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        async with DXLinkStreamer(session, ssl_context=ssl_context) as streamer:
            await streamer.subscribe(Quote, symbols)
            await streamer.subscribe(Greeks, symbols)
            await streamer.subscribe(TheoPrice, symbols)
            await asyncio.sleep(0.3)  # allow initial data to populate

            quotes = await self._drain_events(streamer, Quote, symbols)
            greeks = await self._drain_events(streamer, Greeks, symbols)
            theos = await self._drain_events(streamer, TheoPrice, symbols)

        payload: Dict[str, Dict[str, Any]] = {}
        for sym in symbols:
            quote = quotes.get(sym)
            greek = greeks.get(sym)
            theo = theos.get(sym)

            payload[sym] = {
                "quote": self._quote_to_dict(quote),
                "greeks": self._greeks_to_dict(greek),
                "theo": self._theo_to_dict(theo),
            }
        return payload

    async def _drain_events(
        self,
        streamer: DXLinkStreamer,
        event_type,
        symbols: List[str],
    ) -> Dict[str, Any]:
        deadline = asyncio.get_event_loop().time() + self.timeout
        store: Dict[str, Any] = {}

        while asyncio.get_event_loop().time() < deadline and len(store) < len(symbols):
            event = streamer.get_event_nowait(event_type)
            if event is not None:
                store[getattr(event, "event_symbol", getattr(event, "eventSymbol", "")) or event.eventSymbol] = event
                continue
            await asyncio.sleep(0.05)

        return store

    @staticmethod
    def _quote_to_dict(quote: Optional[Quote]) -> Dict[str, Optional[float]]:
        if quote is None:
            return {}
        return {
            "bid": _to_float(quote.bid_price),
            "ask": _to_float(quote.ask_price),
            "last": _to_float(getattr(quote, "last_price", None)),
            "bid_size": _to_float(quote.bid_size),
            "ask_size": _to_float(quote.ask_size),
        }

    @staticmethod
    def _greeks_to_dict(greek: Optional[Greeks]) -> Dict[str, Optional[float]]:
        if greek is None:
            return {}
        return {
            "delta": _to_float(greek.delta),
            "gamma": _to_float(greek.gamma),
            "theta": _to_float(greek.theta),
            "rho": _to_float(greek.rho),
            "vega": _to_float(greek.vega),
            "iv": _to_float(greek.volatility),
        }

    @staticmethod
    def _theo_to_dict(theo: Optional[TheoPrice]) -> Dict[str, Optional[float]]:
        if theo is None:
            return {}
        return {
            "theo_price": _to_float(theo.price),
            "underlying_price": _to_float(theo.underlying_price),
        }


@dataclass
class OptionSnapshot:
    symbol: str
    strike: float
    expiration: str
    option_type: str
    days_to_expiration: int
    streamer_symbol: str
    record: Dict[str, Any]


class MarketDataService:
    """High level orchestrator that builds enriched option chains."""

    MIN_DTE = 7
    MAX_DTE = 45

    def __init__(self) -> None:
        self.session_manager = TastytradeSessionManager()
        self.dxlink = DxLinkCollector()

    async def build_snapshot(
        self,
        symbols: List[str],
        max_options: int,
        strike_span: float,
    ) -> Dict[str, Any]:
        session = await self.session_manager.get_session()
        loop = asyncio.get_running_loop()

        option_payload: Dict[str, List[Dict[str, Any]]] = {}
        market_data_payload: Dict[str, Dict[str, List[float]]] = {}
        underlying_prices: Dict[str, float] = {}
        streamer_map: Dict[str, Tuple[str, Dict[str, Any]]] = {}

        for symbol in symbols:
            chain = await loop.run_in_executor(None, lambda: get_option_chain(session, symbol))
            try:
                equity_quote: MarketData = await loop.run_in_executor(
                    None, lambda: get_market_data(session, symbol, InstrumentType.EQUITY)
                )
                underlying_price = self._market_price(equity_quote)
            except Exception:
                underlying_price = 0.0
                equity_quote = None

            underlying_prices[symbol] = underlying_price
            selected, records = self._select_options(chain, underlying_price, strike_span, max_options)
            option_payload[symbol] = records
            for opt in selected:
                streamer_map[opt.streamer_symbol] = (symbol, opt.record)

            market_data_payload[symbol] = self._bootstrap_history(equity_quote)

        streamer_symbols = list(streamer_map.keys())
        dx_metrics = await self.dxlink.collect(session, streamer_symbols)

        for dx_symbol, (underlying_symbol, record) in streamer_map.items():
            metrics = dx_metrics.get(dx_symbol, {})
            quote = metrics.get("quote", {})
            greeks = metrics.get("greeks", {})
            theo = metrics.get("theo", {})
            record["bid"] = quote.get("bid")
            record["ask"] = quote.get("ask")
            record["last"] = quote.get("last")
            record["bid_size"] = quote.get("bid_size")
            record["ask_size"] = quote.get("ask_size")
            record["mid"] = self._mid_from_quote(quote)
            record["greeks"] = greeks
            record["theo_price"] = theo.get("theo_price")
            if theo.get("underlying_price"):
                underlying_prices[underlying_symbol] = theo["underlying_price"]

        for records in option_payload.values():
            for rec in records:
                rec.pop("streamer_symbol", None)

        return {
            "symbols": symbols,
            "option_chains": option_payload,
            "market_data": market_data_payload,
            "underlying_prices": underlying_prices,
        }

    def _select_options(
        self,
        chain: Dict[Any, List[Option]],
        underlying_price: float,
        strike_span: float,
        max_options: int,
    ) -> Tuple[List[OptionSnapshot], List[Dict[str, Any]]]:
        records: List[OptionSnapshot] = []

        lower = underlying_price * (1 - strike_span)
        upper = underlying_price * (1 + strike_span)

        for expiration, options in chain.items():
            for opt in options:
                if opt.days_to_expiration < self.MIN_DTE or opt.days_to_expiration > self.MAX_DTE:
                    continue
                strike = float(opt.strike_price)
                if underlying_price > 0 and (strike < lower or strike > upper):
                    continue
                record = {
                    "symbol": opt.symbol,
                    "strike": strike,
                    "expiration": opt.expiration_date.isoformat(),
                    "option_type": "call" if opt.option_type.value.upper().startswith("C") else "put",
                    "days_to_expiration": opt.days_to_expiration,
                    "shares_per_contract": int(opt.shares_per_contract or 100),
                }
                records.append(
                    OptionSnapshot(
                        symbol=opt.symbol,
                        strike=strike,
                        expiration=record["expiration"],
                        option_type=record["option_type"],
                        days_to_expiration=opt.days_to_expiration,
                        streamer_symbol=opt.streamer_symbol,
                        record=record,
                    )
                )

        records.sort(
            key=lambda snap: (
                abs(snap.strike - underlying_price),
                snap.days_to_expiration,
            )
        )
        selected = records[:max_options]
        return selected, [snap.record | {"streamer_symbol": snap.streamer_symbol} for snap in selected]

    @staticmethod
    def _market_price(data: Optional[MarketData]) -> float:
        if not data:
            return 0.0
        for attr in ("mark", "mid", "last", "close", "prev_close", "open"):
            value = getattr(data, attr, None)
            if value is not None:
                return float(value)
        return 0.0

    @staticmethod
    def _bootstrap_history(quote: Optional[MarketData]) -> Dict[str, List[float]]:
        price = max(MarketDataService._market_price(quote), 1.0)
        volume = float(
            getattr(quote, "volume", None)
            or getattr(quote, "total_volume", None)
            or 1_000_000
        )
        closes = [price * (1 + 0.0015 * np.sin(idx / 3.0)) for idx in range(30)]
        volumes = [max(1.0, volume * (0.95 + 0.02 * np.cos(idx / 4.0))) for idx in range(30)]
        return {"close": closes, "volume": volumes}

    @staticmethod
    def _mid_from_quote(quote: Dict[str, Optional[float]]) -> Optional[float]:
        bid = quote.get("bid")
        ask = quote.get("ask")
        if bid and ask:
            return (bid + ask) / 2.0
        return bid or ask
