"""FastAPI application exposing the ensemble universe scan."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .market_data import MarketDataService
from .schemas import (
    EnsembleIdea,
    MarketDataRequest,
    MarketDataResponse,
    UniverseScanRequest,
    UniverseScanResponse,
)
from .universe_scanner import UniverseScanner

logger = logging.getLogger("ensemble-service")


def create_app() -> FastAPI:
    """Factory to build the FastAPI application."""
    app = FastAPI(title="Options Ensemble Service", version="1.0.0")

    allow_origins = os.getenv("CORS_ALLOW_ORIGINS", "").split(",")
    allow_origins = [origin.strip() for origin in allow_origins if origin.strip()]
    if not allow_origins:
        allow_origins = ["*"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    state: dict[str, Optional[object]] = {"scanner": None, "market_service": None}

    @app.on_event("startup")
    async def startup_event() -> None:
        risk_level = os.getenv("ENSEMBLE_RISK_LEVEL", "moderate").lower()
        if risk_level not in {"low", "moderate", "high"}:
            risk_level = "moderate"
        state["scanner"] = UniverseScanner(risk_level=risk_level)
        try:
            state["market_service"] = MarketDataService()
        except RuntimeError as exc:
            logger.warning("MarketDataService unavailable: %s", exc)

    @app.get("/health")
    async def healthcheck() -> dict[str, str]:
        scanner_ready = state["scanner"] is not None
        return {
            "status": "ok" if scanner_ready else "initializing",
            "scanner": "ready" if scanner_ready else "initializing",
            "timestamp": datetime.utcnow().isoformat(),
        }

    @app.post("/ensemble/ideas", response_model=UniverseScanResponse)
    async def generate_ideas(payload: UniverseScanRequest) -> UniverseScanResponse:
        scanner = state["scanner"]
        if scanner is None:
            raise HTTPException(status_code=503, detail="Scanner not initialized yet.")

        ideas: list[EnsembleIdea] = await scanner.scan(payload)
        symbols = {idea.symbol for idea in ideas if idea.symbol}
        return UniverseScanResponse(
            ideas=ideas,
            generated_at=datetime.utcnow().isoformat(),
            symbols_evaluated=sorted(symbols),
            risk_level=scanner.risk_level,
        )

    @app.post("/market-data/options", response_model=MarketDataResponse)
    async def market_data_endpoint(payload: MarketDataRequest) -> MarketDataResponse:
        service = state.get("market_service")
        if service is None:
            raise HTTPException(status_code=503, detail="Market data service is not configured.")
        snapshot = await service.build_snapshot(
            symbols=payload.symbols,
            max_options=payload.max_options,
            strike_span=payload.strike_span,
        )
        return MarketDataResponse(**snapshot)

    return app
app = create_app()
