"""Persistence helpers for favorites, chat logs, and settings."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from sqlalchemy import delete, insert, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from .db import chat_logs, get_engine, jeffrey_settings, user_favorites


def _ensure_engine():
    engine = get_engine()
    if engine is None:
        raise RuntimeError("Database is not configured for this service.")
    return engine


def save_favorite(payload: Dict[str, Any]) -> Dict[str, Any]:
    engine = _ensure_engine()
    stmt = insert(user_favorites).values(**payload)
    if engine.url.get_dialect().name == "postgresql":
        stmt = pg_insert(user_favorites).values(**payload).on_conflict_do_update(
            index_elements=[user_favorites.c.account_number, user_favorites.c.idea_id],
            set_={
                "snapshot": payload["snapshot"],
                "symbol": payload["symbol"],
                "strategy": payload["strategy"],
            },
        )
    with engine.begin() as conn:
        conn.execute(stmt)
    return payload


def list_favorites(account_number: str) -> List[Dict[str, Any]]:
    engine = _ensure_engine()
    stmt = (
        select(
            user_favorites.c.account_number,
            user_favorites.c.idea_id,
            user_favorites.c.symbol,
            user_favorites.c.strategy,
            user_favorites.c.snapshot,
            user_favorites.c.created_at,
        )
        .where(user_favorites.c.account_number == account_number)
        .order_by(user_favorites.c.created_at.desc())
    )
    with engine.connect() as conn:
        rows = conn.execute(stmt).mappings().all()
    return [dict(row) for row in rows]


def delete_favorite(account_number: str, idea_id: str) -> None:
    engine = _ensure_engine()
    stmt = delete(user_favorites).where(
        (user_favorites.c.account_number == account_number)
        & (user_favorites.c.idea_id == idea_id)
    )
    with engine.begin() as conn:
        conn.execute(stmt)


def log_chat_message(payload: Dict[str, Any]) -> None:
    engine = _ensure_engine()
    stmt = insert(chat_logs).values(**payload)
    with engine.begin() as conn:
        conn.execute(stmt)


def get_chat_history(account_number: str, session_id: str) -> List[Dict[str, Any]]:
    engine = _ensure_engine()
    stmt = (
        select(
            chat_logs.c.account_number,
            chat_logs.c.session_id,
            chat_logs.c.role,
            chat_logs.c.content,
            chat_logs.c.created_at,
        )
        .where(
            (chat_logs.c.account_number == account_number)
            & (chat_logs.c.session_id == session_id)
        )
        .order_by(chat_logs.c.created_at.asc())
    )
    with engine.connect() as conn:
        rows = conn.execute(stmt).mappings().all()
    return [dict(row) for row in rows]


def upsert_settings(account_number: str, settings: Dict[str, Any]) -> Dict[str, Any]:
    engine = _ensure_engine()
    stmt = insert(jeffrey_settings).values(account_number=account_number, settings=settings)
    if engine.url.get_dialect().name == "postgresql":
        stmt = pg_insert(jeffrey_settings).values(
            account_number=account_number, settings=settings
        ).on_conflict_do_update(index_elements=[jeffrey_settings.c.account_number], set_={"settings": settings})
    with engine.begin() as conn:
        conn.execute(stmt)
    return {"account_number": account_number, "settings": settings}


def get_settings(account_number: str) -> Dict[str, Any]:
    engine = _ensure_engine()
    stmt = select(jeffrey_settings.c.settings).where(jeffrey_settings.c.account_number == account_number)
    with engine.connect() as conn:
        row = conn.execute(stmt).first()
    if not row:
        return {"account_number": account_number, "settings": {}}
    return {"account_number": account_number, "settings": row[0]}
