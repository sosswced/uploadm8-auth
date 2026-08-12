"""Account group membership validation, resolve, and disconnect prune."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from services.account_groups import (
    GROUPS_NO_MATCHING_ACCOUNTS,
    prune_account_id_from_groups,
    resolve_group_ids_to_target_accounts,
    validate_account_ids_for_user,
)


def test_validate_account_ids_accepts_owned_tokens():
    conn = AsyncMock()
    conn.fetch = AsyncMock(return_value=[{"id": "tok-1"}, {"id": "tok-2"}])

    async def _run():
        return await validate_account_ids_for_user(conn, "user-1", ["tok-1", "tok-2"])

    assert asyncio.run(_run()) == ["tok-1", "tok-2"]


def test_validate_account_ids_rejects_foreign_or_native_ids():
    conn = AsyncMock()
    conn.fetch = AsyncMock(return_value=[{"id": "tok-1"}])

    async def _run():
        await validate_account_ids_for_user(conn, "user-1", ["tok-1", "native-open-id"])

    with pytest.raises(HTTPException) as exc:
        asyncio.run(_run())
    assert exc.value.status_code == 400
    assert "native-open-id" in str(exc.value.detail)


def test_resolve_group_ids_unions_and_filters_platforms():
    conn = AsyncMock()

    async def _fetch(sql, *args):
        sql_l = str(sql).lower()
        if "from account_groups" in sql_l:
            return [
                {"id": "g1", "account_ids": ["tok-yt", "tok-tt", "ghost"]},
            ]
        if "from platform_tokens" in sql_l and "platform" in sql_l:
            return [
                {"id": "tok-yt", "platform": "youtube"},
                {"id": "tok-tt", "platform": "tiktok"},
            ]
        return [{"id": "tok-yt"}, {"id": "tok-tt"}]

    conn.fetch = AsyncMock(side_effect=_fetch)

    async def _run():
        return await resolve_group_ids_to_target_accounts(
            conn, "user-1", ["g1"], platforms=["youtube"]
        )

    resolved, gids = asyncio.run(_run())
    assert gids == ["g1"]
    assert resolved == ["tok-yt"]


def test_resolve_group_ids_empty_raises():
    conn = AsyncMock()

    async def _fetch(sql, *args):
        sql_l = str(sql).lower()
        if "from account_groups" in sql_l:
            return [{"id": "g1", "account_ids": ["ghost"]}]
        return []

    conn.fetch = AsyncMock(side_effect=_fetch)

    async def _run():
        await resolve_group_ids_to_target_accounts(conn, "user-1", ["g1"])

    with pytest.raises(HTTPException) as exc:
        asyncio.run(_run())
    assert exc.value.status_code == 400
    assert GROUPS_NO_MATCHING_ACCOUNTS in str(exc.value.detail)


def test_prune_account_id_from_groups():
    conn = AsyncMock()
    conn.execute = AsyncMock(return_value="UPDATE 2")

    async def _run():
        return await prune_account_id_from_groups(conn, "user-1", "tok-dead")

    assert asyncio.run(_run()) == 2
    conn.execute.assert_awaited_once()
    args = conn.execute.await_args.args
    assert args[1] == "user-1"
    assert args[2] == "tok-dead"


def test_prune_skips_blank_account_id():
    conn = AsyncMock()

    async def _run():
        return await prune_account_id_from_groups(conn, "user-1", "  ")

    assert asyncio.run(_run()) == 0
    conn.execute.assert_not_called()
