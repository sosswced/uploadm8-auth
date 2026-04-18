"""
Shared asyncpg JSON / JSONB codec registration.

Idempotent-safe: registering twice or unsupported servers logs at DEBUG and continues.
Used by app.py, worker.py, and migration tooling.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

import asyncpg

logger = logging.getLogger("uploadm8.asyncpg_json")


def json_param_encoder(value: Any) -> str:
    """Encoder for asyncpg: strings pass through (caller may already have json.dumps'd)."""
    if isinstance(value, str):
        return value
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


async def apply_asyncpg_json_codecs(conn, log: logging.Logger | None = None) -> None:
    """Register json + jsonb codecs on one connection (pool init callback)."""
    lg = log or logger
    enc: Callable[[Any], str] = json_param_encoder
    for name in ("json", "jsonb"):
        try:
            await conn.set_type_codec(
                name,
                encoder=enc,
                decoder=json.loads,
                schema="pg_catalog",
            )
        except (asyncpg.PostgresError, asyncpg.InterfaceError, ValueError, TypeError, OSError) as e:
            lg.debug("asyncpg %s codec not registered: %s", name, e)
