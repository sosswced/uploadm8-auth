"""Admin CRUD for the dynamic generic hard-ban registry."""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

import core.state
from core.deps import require_admin
from services.generic_hard_ban import (
    admin_report,
    apply_add,
    apply_remove,
    apply_restore,
    load_overlay_from_db,
    persist_overlay,
    refresh_from_db,
    set_auto_promote_after,
)

router = APIRouter(prefix="/api/admin", tags=["admin", "generic-hard-ban"])


class SlugListBody(BaseModel):
    slugs: List[str] = Field(default_factory=list, max_length=64)


class AutoPromoteBody(BaseModel):
    auto_promote_after: int = Field(2, ge=0, le=50)


def _actor(user: dict) -> str:
    return str(user.get("email") or user.get("id") or "admin")[:120]


def _clean_slugs(raw: List[str]) -> List[str]:
    from services.generic_hard_ban import normalize_ban_slug

    out: List[str] = []
    seen: set = set()
    for item in raw or []:
        slug = normalize_ban_slug(item)
        if not slug or len(slug) > 48 or slug in seen:
            continue
        seen.add(slug)
        out.append(slug)
        if len(out) >= 64:
            break
    return out


@router.get("/generic-hard-ban")
async def get_generic_hard_ban(user: dict = Depends(require_admin)):
    """Full registry: effective ban list, added/removed, learned hits."""
    if core.state.db_pool is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    await refresh_from_db(core.state.db_pool)
    return admin_report()


@router.post("/generic-hard-ban/add")
async def add_generic_hard_ban(body: SlugListBody, user: dict = Depends(require_admin)):
    if core.state.db_pool is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    slugs = _clean_slugs(body.slugs)
    if not slugs:
        raise HTTPException(status_code=400, detail="slugs required")
    o = await load_overlay_from_db(core.state.db_pool)
    o = apply_add(o, slugs, by=_actor(user))
    await persist_overlay(core.state.db_pool, o, merge_from_cache=True)
    return admin_report()


@router.post("/generic-hard-ban/remove")
async def remove_generic_hard_ban(body: SlugListBody, user: dict = Depends(require_admin)):
    """Unban / exclude slugs (fixes false positives). Removed always wins."""
    if core.state.db_pool is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    slugs = _clean_slugs(body.slugs)
    if not slugs:
        raise HTTPException(status_code=400, detail="slugs required")
    o = await load_overlay_from_db(core.state.db_pool)
    o = apply_remove(o, slugs, by=_actor(user))
    await persist_overlay(core.state.db_pool, o, merge_from_cache=True)
    return admin_report()


@router.post("/generic-hard-ban/restore")
async def restore_generic_hard_ban(body: SlugListBody, user: dict = Depends(require_admin)):
    """Undo a remove so builtin/added/learned can ban the slug again."""
    if core.state.db_pool is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    slugs = _clean_slugs(body.slugs)
    if not slugs:
        raise HTTPException(status_code=400, detail="slugs required")
    o = await load_overlay_from_db(core.state.db_pool)
    o = apply_restore(o, slugs, by=_actor(user))
    await persist_overlay(core.state.db_pool, o, merge_from_cache=True)
    return admin_report()


@router.put("/generic-hard-ban/auto-promote")
async def put_auto_promote(body: AutoPromoteBody, user: dict = Depends(require_admin)):
    """How many scrub hits before a learned slug auto-joins the ban list (0=manual)."""
    if core.state.db_pool is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    o = await load_overlay_from_db(core.state.db_pool)
    o = set_auto_promote_after(o, body.auto_promote_after, by=_actor(user))
    await persist_overlay(core.state.db_pool, o, merge_from_cache=True)
    return admin_report()
