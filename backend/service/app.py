"""FastAPI application exposing the ensemble universe scan."""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .market_data import MarketDataService
from .repository import (
    delete_favorite,
    get_chat_history,
    get_settings,
    list_favorites,
    log_chat_message,
    save_favorite,
    upsert_settings,
)
from .db import init_db
from .schemas import (
    EnsembleIdea,
    FavoriteListResponse,
    FavoritePayload,
    MarketDataRequest,
    MarketDataResponse,
    ChatLogEntry,
    ChatHistoryResponse,
    JeffreySettingsPayload,
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

    state: dict[str, Optional[object]] = {"scanner": None, "market_service": None, "db_ready": None}

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
        try:
            state["db_ready"] = init_db()
            if not state["db_ready"]:
                logger.warning("Favorites DB not configured.")
        except Exception as exc:
            state["db_ready"] = False
            logger.warning("Failed to initialize database: %s", exc)

    @app.get("/")
    async def root() -> dict[str, object]:
        """Root endpoint with API information."""
        return {
            "name": "Options Ensemble Service",
            "version": "1.0.0",
            "status": "running",
            "endpoints": {
                "health": "/health",
                "market_data": "POST /market-data/options",
                "ensemble_ideas": "POST /ensemble/ideas",
                "docs": "/docs",
                "openapi": "/openapi.json"
            },
            "features": [
                "DXLink real-time streaming",
                "Greeks calculation",
                "Model ensemble trading ideas",
                "Risk-adjusted strategies"
            ],
            "timestamp": datetime.utcnow().isoformat(),
        }

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

    # ----------------------------------------------------------------------
    # Favorites endpoints
    # ----------------------------------------------------------------------

    @app.get("/favorites/{account_number}", response_model=FavoriteListResponse)
    async def favorites_list(account_number: str) -> FavoriteListResponse:
        loop = asyncio.get_running_loop()
        try:
            favorites = await loop.run_in_executor(None, lambda: list_favorites(account_number))
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return FavoriteListResponse(favorites=favorites)

    @app.post("/favorites", response_model=FavoritePayload)
    async def favorites_save(payload: FavoritePayload) -> FavoritePayload:
        loop = asyncio.get_running_loop()
        try:
            saved = await loop.run_in_executor(None, lambda: save_favorite(payload.dict()))
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return FavoritePayload(**saved)

    @app.delete("/favorites/{account_number}/{idea_id}")
    async def favorites_delete(account_number: str, idea_id: str) -> dict[str, str]:
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, lambda: delete_favorite(account_number, idea_id))
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {"status": "deleted"}

    # ----------------------------------------------------------------------
    # Chat log endpoints
    # ----------------------------------------------------------------------

    @app.post("/chat/logs")
    async def chat_log(payload: ChatLogEntry) -> dict[str, str]:
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, lambda: log_chat_message(payload.dict()))
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {"status": "logged"}

    @app.get("/chat/logs/{account_number}/{session_id}", response_model=ChatHistoryResponse)
    async def chat_history(account_number: str, session_id: str) -> ChatHistoryResponse:
        loop = asyncio.get_running_loop()
        try:
            messages = await loop.run_in_executor(None, lambda: get_chat_history(account_number, session_id))
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return ChatHistoryResponse(messages=messages)

    # ----------------------------------------------------------------------
    # Jeffrey settings
    # ----------------------------------------------------------------------

    @app.get("/settings/{account_number}", response_model=JeffreySettingsPayload)
    async def get_user_settings(account_number: str) -> JeffreySettingsPayload:
        loop = asyncio.get_running_loop()
        try:
            settings = await loop.run_in_executor(None, lambda: get_settings(account_number))
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return JeffreySettingsPayload(**settings)

    @app.put("/settings/{account_number}", response_model=JeffreySettingsPayload)
    async def update_user_settings(account_number: str, payload: JeffreySettingsPayload) -> JeffreySettingsPayload:
        loop = asyncio.get_running_loop()
        try:
            updated = await loop.run_in_executor(
                None, lambda: upsert_settings(account_number, payload.settings)
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return JeffreySettingsPayload(**updated)

    return app


app = create_app()
