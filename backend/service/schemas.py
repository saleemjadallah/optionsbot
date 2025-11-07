"""Pydantic schemas for the ensemble microservice."""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, validator


class OptionRecord(BaseModel):
    """Single option observation from the frontend fetcher."""

    symbol: str = Field(..., description="Full option symbol understood by Tastytrade.")
    strike: float = Field(..., gt=0)
    expiration: date = Field(..., description="ISO formatted expiration date.")
    option_type: Literal["call", "put"]
    bid: Optional[float] = Field(None, ge=0)
    ask: Optional[float] = Field(None, ge=0)
    mid: Optional[float] = Field(None, ge=0)
    days_to_expiration: Optional[int] = Field(None, ge=0)

    @validator("option_type")
    def normalize_type(cls, value: str) -> str:  # noqa: D401
        """Ensure option type is lowercased."""
        return value.lower()


class MarketHistory(BaseModel):
    """Minimal market history required by the ensemble."""

    close: List[float] = Field(..., description="Recent closing prices (most recent last).")
    volume: Optional[List[float]] = Field(None, description="Recent volume series aligned with close.")

    @validator("close")
    def require_enough_points(cls, value: List[float]) -> List[float]:  # noqa: D401
        """Require at least a handful of data points for basic stats."""
        if len(value) < 5:
            raise ValueError("close series must contain at least 5 data points")
        return value

    @validator("volume")
    def validate_volume_length(cls, value: Optional[List[float]], values) -> Optional[List[float]]:  # noqa: D401
        """If volume is provided it must align with close length."""
        if value is not None and "close" in values and len(value) != len(values["close"]):
            raise ValueError("volume series must match close series length")
        return value


class UniverseScanRequest(BaseModel):
    """Request payload for /ensemble/ideas endpoint."""

    symbols: List[str] = Field(..., description="Unique list of symbols to evaluate.")
    option_chains: Dict[str, List[OptionRecord]] = Field(
        ..., description="Map of symbol -> list of option records."
    )
    market_data: Dict[str, MarketHistory] = Field(
        ..., description="Map of symbol -> market history used for regime detection."
    )
    risk_level: Literal["low", "moderate", "high"] = Field(
        "moderate", description="Portfolio risk appetite controlling ensemble weighting."
    )
    min_edge: float = Field(0.02, ge=0.0, le=0.2, description="Minimum edge magnitude to include.")
    min_confidence: float = Field(
        0.4, ge=0.0, le=1.0, description="Minimum ensemble confidence to include."
    )

    @validator("symbols")
    def dedupe_symbols(cls, value: List[str]) -> List[str]:
        """Dedupe symbols while preserving order."""
        seen = set()
        deduped = []
        for sym in value:
            sym_clean = sym.strip().upper()
            if not sym_clean or sym_clean in seen:
                continue
            seen.add(sym_clean)
            deduped.append(sym_clean)
        if not deduped:
            raise ValueError("symbols list cannot be empty")
        return deduped


class EnsembleIdea(BaseModel):
    """Simplified trade idea returned by the ensemble service."""

    symbol: str
    suggested_strategy: str
    signal: str
    rationale: str
    trade_idea: str
    category: str
    metrics: Dict[str, float]
    order_example: Dict[str, object]
    source: str = "universe"
    origin: str = "universe"


class UniverseScanResponse(BaseModel):
    """Response envelope for universe scan."""

    ideas: List[EnsembleIdea]
    generated_at: str
    symbols_evaluated: List[str]
    risk_level: str


class MarketDataRequest(BaseModel):
    """Request payload for enriched option chain snapshots."""

    symbols: List[str] = Field(..., description="List of underlying symbols to fetch.")
    max_options: int = Field(
        180, ge=1, le=400, description="Maximum number of option contracts per symbol."
    )
    strike_span: float = Field(
        0.15, ge=0.01, le=0.5, description="Strike distance from ATM (fractional)."
    )


class MarketDataResponse(BaseModel):
    """Option chain metadata plus streaming quotes."""

    symbols: List[str]
    option_chains: Dict[str, List[Dict[str, Any]]]
    market_data: Dict[str, Dict[str, List[float]]]
    underlying_prices: Dict[str, float]
