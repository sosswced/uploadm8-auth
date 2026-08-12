"""
Account group helpers — validate members and resolve group_ids to platform_tokens.id.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Set

from fastapi import HTTPException


async def fetch_user_platform_token_ids(conn, user_id: str) -> Set[str]:
    """Return set of live platform_tokens.id strings owned by user."""
    rows = await conn.fetch(
        """
        SELECT id::text FROM platform_tokens
         WHERE user_id = $1 AND revoked_at IS NULL
        """,
        user_id,
    )
    return {str(r["id"]) for r in rows}


async def prune_account_id_from_groups(conn, user_id: str, account_id: str) -> int:
    """
    Remove a disconnected platform_tokens.id from every group owned by user.

    Returns the number of groups updated.
    """
    aid = str(account_id or "").strip()
    if not aid or not user_id:
        return 0
    status = await conn.execute(
        """
        UPDATE account_groups
           SET account_ids = array_remove(account_ids, $2::text),
               updated_at = NOW()
         WHERE user_id = $1
           AND $2::text = ANY(account_ids)
        """,
        user_id,
        aid,
    )
    try:
        return int(str(status).split()[-1])
    except (TypeError, ValueError, IndexError):
        return 0


async def validate_account_ids_for_user(
    conn,
    user_id: str,
    account_ids: Optional[Sequence[str]],
) -> List[str]:
    """
    Ensure every account_id exists in platform_tokens for this user.
    Returns normalized list (may be empty).
    """
    if not account_ids:
        return []
    ids = [str(a).strip() for a in account_ids if a]
    if not ids:
        return []
    owned = await fetch_user_platform_token_ids(conn, user_id)
    invalid = [a for a in ids if a not in owned]
    if invalid:
        raise HTTPException(
            400,
            f"Unknown or unauthorized account_ids: {', '.join(invalid[:5])}"
            + ("…" if len(invalid) > 5 else ""),
        )
    return ids


GROUPS_NO_MATCHING_ACCOUNTS = (
    "Selected groups contain no matching platform accounts for this upload. "
    "Check group membership and selected platforms."
)


async def resolve_group_ids_to_target_accounts(
    conn,
    user_id: str,
    group_ids: Optional[Sequence[str]],
    platforms: Optional[Sequence[str]] = None,
    *,
    raise_on_empty: bool = True,
) -> tuple[List[str], List[str]]:
    """
    Load account_groups for user, union account_ids, intersect with owned tokens.

    When platforms is provided, only include accounts whose platform is in that set.

    Returns (resolved_target_account_ids, normalized_group_ids).
    Raises HTTPException 400 if group_ids provided but resolve to zero accounts
    (unless ``raise_on_empty=False`` — used by cost estimate to avoid console spam).
    """
    if not group_ids:
        return [], []

    gids = [str(g).strip() for g in group_ids if g]
    if not gids:
        return [], []

    rows = await conn.fetch(
        """
        SELECT id::text AS id, account_ids
        FROM account_groups
        WHERE user_id = $1 AND id = ANY($2::uuid[])
        """,
        user_id,
        gids,
    )
    found = {str(r["id"]) for r in rows}
    missing = [g for g in gids if g not in found]
    if missing:
        raise HTTPException(404, f"Group not found: {', '.join(missing[:3])}")

    union_ids: List[str] = []
    seen: Set[str] = set()
    for r in rows:
        for aid in r["account_ids"] or []:
            s = str(aid)
            if s not in seen:
                seen.add(s)
                union_ids.append(s)

    owned = await fetch_user_platform_token_ids(conn, user_id)
    resolved = [a for a in union_ids if a in owned]

    if platforms:
        plat_set = {p.lower() for p in platforms if p}
        if plat_set:
            tok_rows = await conn.fetch(
                """
                SELECT id::text AS id, platform
                FROM platform_tokens
                WHERE user_id = $1 AND id = ANY($2::uuid[])
                """,
                user_id,
                resolved or ["00000000-0000-0000-0000-000000000000"],
            )
            resolved = [
                str(r["id"])
                for r in tok_rows
                if str(r["id"]) in resolved and str(r["platform"]).lower() in plat_set
            ]

    if not resolved:
        if raise_on_empty:
            raise HTTPException(400, GROUPS_NO_MATCHING_ACCOUNTS)
        return [], gids

    return resolved, gids
