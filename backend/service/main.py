"""Uvicorn entry point for running the ensemble service."""

from __future__ import annotations

import os

import uvicorn

from .app import create_app


def run() -> None:
    """Launch the FastAPI application with uvicorn."""
    host = os.getenv("SERVICE_HOST", "0.0.0.0")
    port_value = os.getenv("SERVICE_PORT") or os.getenv("PORT") or "8000"
    port = int(port_value)
    reload = os.getenv("SERVICE_RELOAD", "false").lower() in {"1", "true", "yes"}

    uvicorn.run(
        create_app(),
        host=host,
        port=port,
        reload=reload,
        log_level=os.getenv("SERVICE_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    run()
