"""promo_label_backfill must pass a Python int to asyncpg ($1::int)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from services.promo_label_backfill import backfill_promo_outcome_labels


def test_backfill_passes_int_lookback_to_asyncpg():
    conn = MagicMock()
    conn.fetch = AsyncMock(return_value=[])

    out = asyncio.run(backfill_promo_outcome_labels(conn, lookback_days=730))

    assert out == {"inserted": 0, "candidates": 0}
    conn.fetch.assert_awaited_once()
    args = conn.fetch.await_args.args
    assert len(args) == 2
    assert isinstance(args[1], int)
    assert args[1] == 730


def test_backfill_coerces_string_lookback_to_int():
    """Env/config mishaps must not reintroduce str args (asyncpg TypeError)."""
    conn = MagicMock()
    conn.fetch = AsyncMock(return_value=[])

    asyncio.run(backfill_promo_outcome_labels(conn, lookback_days="420"))  # type: ignore[arg-type]

    assert isinstance(conn.fetch.await_args.args[1], int)
    assert conn.fetch.await_args.args[1] == 420
