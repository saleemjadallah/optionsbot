"""Database utilities for the ensemble service."""

from __future__ import annotations

import os
from typing import Optional
from urllib.parse import quote_plus

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    func,
    create_engine,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine


def _build_db_url_from_pg_env() -> Optional[str]:
    """Construct a SQLAlchemy URL from Railway-style PG* env vars."""
    host = os.getenv("PGHOST") or os.getenv("POSTGRES_HOST")
    user = os.getenv("PGUSER") or os.getenv("POSTGRES_USER")
    password = os.getenv("PGPASSWORD") or os.getenv("POSTGRES_PASSWORD")
    database = os.getenv("PGDATABASE") or os.getenv("POSTGRES_DB")
    port = os.getenv("PGPORT") or os.getenv("POSTGRES_PORT") or "5432"

    if not all([host, user, password, database]):
        return None

    safe_user = quote_plus(user)
    safe_password = quote_plus(password)
    safe_db = quote_plus(database)
    url = f"postgresql://{safe_user}:{safe_password}@{host}:{port}/{safe_db}"
    os.environ.setdefault("FAVORITES_DATABASE_URL", url)
    return url


def _resolve_database_url() -> Optional[str]:
    direct = os.getenv("FAVORITES_DATABASE_URL")
    if direct:
        return direct

    fallback = os.getenv("DATABASE_URL")
    if fallback:
        os.environ.setdefault("FAVORITES_DATABASE_URL", fallback)
        return fallback

    return _build_db_url_from_pg_env()


DATABASE_URL = _resolve_database_url()
metadata = MetaData()

user_favorites = Table(
    "user_favorites",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("account_number", String, nullable=False),
    Column("idea_id", String, nullable=False),
    Column("symbol", String, nullable=False),
    Column("strategy", String, nullable=False),
    Column("snapshot", JSONB if DATABASE_URL and "postgres" in DATABASE_URL else JSON, nullable=False),
    Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
    Column("updated_at", DateTime(timezone=True), server_default=func.now(), onupdate=func.now()),
    UniqueConstraint("account_number", "idea_id", name="uq_fav_account_idea"),
)

chat_logs = Table(
    "chat_logs",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("account_number", String, nullable=False),
    Column("session_id", String, nullable=False),
    Column("role", String, nullable=False),
    Column("content", Text, nullable=False),
    Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
)

jeffrey_settings = Table(
    "jeffrey_settings",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("account_number", String, nullable=False, unique=True),
    Column("settings", JSONB if DATABASE_URL and "postgres" in DATABASE_URL else JSON, nullable=False),
    Column("updated_at", DateTime(timezone=True), server_default=func.now(), onupdate=func.now()),
)

_engine: Optional[Engine] = None


def get_engine() -> Optional[Engine]:
    global _engine
    if DATABASE_URL and _engine is None:
        _engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    return _engine


def init_db() -> bool:
    engine = get_engine()
    if engine is None:
        return False
    metadata.create_all(engine)
    return True
