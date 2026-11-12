"""Hardening-suite fixtures.

Every run provisions a disposable PostgreSQL database and uses a dedicated
Redis instance. Nothing here talks to a hosted model or vector service: the
only test double is the intent classifier's model call.

Environment (all optional, defaults match the CI services and the local
commands in ``back-end/README``-style docs in AGENTS.md):

* ``HARDENING_POSTGRES_ADMIN_URL`` - server to create the disposable database on.
* ``HARDENING_REDIS_URL`` - dedicated Redis instance (flushed between tests).
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import psycopg
import pytest
import redis
from psycopg import sql

POSTGRES_ADMIN_URL = os.environ.get(
    "HARDENING_POSTGRES_ADMIN_URL",
    "postgresql://postgres:postgres@localhost:5432/postgres",
)
REDIS_URL = os.environ.get("HARDENING_REDIS_URL", "redis://localhost:6380/0")

# Synthetic configuration: the app refuses to start without these, but no
# test ever uses them to reach a hosted provider.
os.environ["GROQ_API_KEY"] = "synthetic-groq-key"
os.environ["GOOGLE_GEMINI_API_KEY"] = "synthetic-gemini-key"
os.environ["PINECONE_API_KEY"] = "synthetic-pinecone-key"
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

_DB_NAME = f"nova_hardening_{uuid.uuid4().hex[:12]}"


def _database_url() -> str:
    parts = urlsplit(POSTGRES_ADMIN_URL)
    return urlunsplit(parts._replace(path=f"/{_DB_NAME}"))


@pytest.fixture(scope="session", autouse=True)
def disposable_database() -> Iterator[str]:
    """Create a fresh database before the app is imported; drop it afterwards.

    Failing to reach PostgreSQL fails the suite. It is never skipped.
    """
    with psycopg.connect(POSTGRES_ADMIN_URL, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(_DB_NAME)))
    url = _database_url()
    os.environ["DATABASE_URL"] = url
    try:
        yield url
    finally:
        _close_app_pools()
        with psycopg.connect(POSTGRES_ADMIN_URL, autocommit=True) as admin:
            admin.execute(
                sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(
                    sql.Identifier(_DB_NAME)
                )
            )


def _close_app_pools() -> None:
    import sys

    graph = sys.modules.get("app.agent.graph")
    if graph is not None:
        graph.pool.close()
    database = sys.modules.get("app.core.database")
    if database is not None:
        database.engine.dispose()


@pytest.fixture(scope="session")
def redis_client() -> Iterator[redis.Redis]:
    client = redis.Redis.from_url(REDIS_URL)
    client.ping()  # fails the suite if the dedicated Redis is not running
    yield client
    client.flushdb()
    client.close()


@pytest.fixture(autouse=True)
def clean_state(disposable_database: str, redis_client: redis.Redis) -> Iterator[None]:
    """Give every test empty application tables and an empty Redis database."""
    _truncate_application_tables(disposable_database)
    redis_client.flushdb()
    yield


def _truncate_application_tables(url: str) -> None:
    with psycopg.connect(url, autocommit=True) as conn:
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
                " AND tablename <> 'checkpoint_migrations'"
            )
        ]
        if tables:
            names = sql.SQL(", ").join(sql.Identifier(t) for t in tables)
            conn.execute(sql.SQL("TRUNCATE {} RESTART IDENTITY CASCADE").format(names))


@pytest.fixture(scope="session")
def app(disposable_database: str) -> Any:
    # Import only after DATABASE_URL points at the disposable database.
    from app.main import app as fastapi_app

    return fastapi_app


@pytest.fixture
def client(app: Any) -> Iterator[Any]:
    from fastapi.testclient import TestClient

    # Entering the context runs the lifespan, which creates the app tables.
    with TestClient(app) as test_client:
        yield test_client
