"""Presign insert, cost estimation, and telemetry key helpers."""

from __future__ import annotations

import json
import uuid
from typing import Any, Optional

from fastapi import HTTPException

from core.models import UploadInit
from core.wallet import atomic_reserve_tokens, get_wallet
from routers.preferences import get_user_prefs_for_upload
from services.account_groups import resolve_group_ids_to_target_accounts
from services.billing_service_weights import fetch_service_weights_map
from services.workspace import billing_user_id, can_upload_in_workspace, get_workspace_for_user
from services.upload.schedule_guard import (
    build_smart_schedule_for_upload,
    schedule_slot_iso,
    validate_presign_schedule,
)
from stages.ai_service_costs import compute_presign_put_aic_costs
from stages.entitlements import entitlements_to_dict, get_entitlements_from_user

from services.upload.hashtags import _to_hash_tags
from services.upload.prefs import (
    merge_upload_init_thumbnail_preferences,
    merge_upload_init_tiktok_post_settings,
    normalize_user_prefs_snapshot,
)
from services.upload.tiktok import (
    _tiktok_target_account_ids,
    _validate_tiktok_post_settings_for_upload,
)


def _json_for_upload_row(obj: Any) -> str:
    """DB-backed dicts may contain datetime/UUID/decimals — asyncpg returns native types."""
    return json.dumps(obj, default=str)


def telemetry_r2_key_for_upload(user_id: str, upload_id: str, has_telemetry: bool) -> Optional[str]:
    """Companion .map object key saved on uploads so the worker can fetch it."""
    if not has_telemetry:
        return None
    return f"uploads/{user_id}/{upload_id}/telemetry.map"


async def presign_create_upload(conn, data: UploadInit, user: dict) -> dict:
    """
    Insert upload row and reserve tokens. Mutates ``data`` (hashtags, privacy, defaults).

    Returns dict with upload_id, r2_key, put_cost, aic_cost, user_prefs, smart_schedule (or None).
    """
    member_id = str(user["id"])
    ws_ctx = await get_workspace_for_user(conn, member_id)
    if ws_ctx and not can_upload_in_workspace(ws_ctx):
        raise HTTPException(403, "Viewers cannot create uploads")
    billing_user = ws_ctx.owner_row if ws_ctx else user
    bill_id = billing_user_id(ws_ctx, member_id)

    db_ent = await conn.fetchrow(
        "SELECT subscription_tier, role, flex_enabled FROM users WHERE id = $1",
        bill_id,
    )
    user_for_ent = dict(billing_user)
    if db_ent:
        for _k in ("subscription_tier", "role", "flex_enabled"):
            _v = db_ent.get(_k)
            if _v is not None:
                user_for_ent[_k] = _v
    ent_cost = get_entitlements_from_user(user_for_ent)
    plan = entitlements_to_dict(ent_cost)

    if getattr(data, "hashtags", None) is None:
        data.hashtags = []
    if getattr(data, "platforms", None) is None:
        data.platforms = []

    group_ids_raw = getattr(data, "group_ids", None) or []
    resolved_group_ids: list[str] = []
    if group_ids_raw:
        resolved_accounts, resolved_group_ids = await resolve_group_ids_to_target_accounts(
            conn,
            bill_id,
            group_ids_raw,
            data.platforms,
        )
        data.target_accounts = resolved_accounts

    user_prefs = await get_user_prefs_for_upload(conn, bill_id)
    normalize_user_prefs_snapshot(user_prefs)
    merge_upload_init_thumbnail_preferences(user_prefs, data)
    merge_upload_init_tiktok_post_settings(user_prefs, data)

    tiktok_account_ids = await _tiktok_target_account_ids(conn, bill_id, data)
    _validate_tiktok_post_settings_for_upload(data, tiktok_account_ids)

    if not getattr(data, "privacy", None):
        data.privacy = user_prefs["default_privacy"]

    combined = _to_hash_tags(getattr(data, "hashtags", []) or [])

    if user_prefs.get("ai_hashtags_enabled") and plan.get("ai"):
        pass

    blocked = set(
        _to_hash_tags(user_prefs.get("blocked_hashtags", []) or user_prefs.get("blockedHashtags", []))
    )
    combined = [h for h in combined if h and h not in blocked]
    data.hashtags = list(dict.fromkeys(combined))[: int(user_prefs.get("max_hashtags", 30))]

    use_ai_checkbox = bool(getattr(data, "use_ai", False))

    num_publish_targets = len(data.target_accounts) if data.target_accounts else len(data.platforms)
    db_weights = await fetch_service_weights_map(conn)
    put_cost, aic_cost, billing_breakdown = compute_presign_put_aic_costs(
        ent_cost,
        num_publish_targets=num_publish_targets,
        file_size=getattr(data, "file_size", None),
        duration_hint=None,
        has_telemetry=bool(getattr(data, "has_telemetry", False)),
        use_ai_checkbox=use_ai_checkbox,
        user_prefs=user_prefs,
        num_thumbnails_override=None,
        service_weights_map=db_weights,
    )

    pending_count = await conn.fetchval(
        """SELECT COUNT(*) FROM uploads
           WHERE user_id = $1
           AND status IN ('pending','staged','queued','processing','ready_to_publish')""",
        bill_id,
    )
    if pending_count >= ent_cost.queue_depth:
        raise HTTPException(
            429,
            detail={
                "code": "queue_depth_exceeded",
                "message": (
                    f"Queue limit reached ({pending_count}/{ent_cost.queue_depth} uploads pending). "
                    "Wait for existing uploads to complete or upgrade your plan."
                ),
                "hint": "Upgrade your plan for a higher queue depth.",
                "topup_url": "/settings.html#billing",
                "ledger_url": "/settings.html#token-balances",
            },
        )

    upload_id = str(uuid.uuid4())
    schedule_seed = (
        str(getattr(data, "smart_schedule_seed", "") or "").strip() or upload_id
    )
    r2_key = f"uploads/{bill_id}/{upload_id}/{data.filename}"
    telemetry_r2_key = telemetry_r2_key_for_upload(
        bill_id,
        upload_id,
        bool(getattr(data, "has_telemetry", False)),
    )

    ws_id = ws_ctx.workspace_id if ws_ctx else None

    validate_presign_schedule(data)

    smart_schedule = None
    schedule_mode = (getattr(data, "schedule_mode", None) or "immediate").strip().lower()
    if schedule_mode == "smart":
        days = getattr(data, "smart_schedule_days", 14)
        smart_schedule = await build_smart_schedule_for_upload(
            conn,
            bill_id,
            data.platforms,
            num_days=days,
            random_seed=schedule_seed,
        )
        if not smart_schedule:
            raise HTTPException(
                500,
                detail={
                    "code": "schedule_generation_failed",
                    "message": "Could not generate smart schedule times. Try again or pick a manual time.",
                },
            )

    scheduled_time = getattr(data, "scheduled_time", None)
    schedule_metadata = None

    if schedule_mode == "smart" and smart_schedule:
        schedule_metadata = {p: schedule_slot_iso(dt) for p, dt in smart_schedule.items()}
        scheduled_time = min(smart_schedule.values())

    vm_id = getattr(data, "vehicle_make_id", None)
    vmd_id = getattr(data, "vehicle_model_id", None)
    if vm_id is None:
        vm_id = user_prefs.get("default_vehicle_make_id")
    if vmd_id is None:
        vmd_id = user_prefs.get("default_vehicle_model_id")
    if vm_id is not None and vmd_id is not None:
        ok = await conn.fetchrow(
            "SELECT 1 FROM vehicle_models WHERE id = $1 AND make_id = $2",
            vmd_id,
            vm_id,
        )
        if not ok:
            raise HTTPException(400, "Invalid vehicle model for selected make")
    elif vmd_id is not None and vm_id is None:
        raise HTTPException(400, "vehicle_make_id required when vehicle_model_id is set")

    await conn.execute(
        """
            INSERT INTO uploads (
                id, user_id, r2_key, telemetry_r2_key, filename, file_size, platforms,
                title, caption, hashtags, privacy, status, scheduled_time,
                schedule_mode, put_reserved, aic_reserved, billing_breakdown, schedule_metadata,
                user_preferences, target_accounts, vehicle_make_id, vehicle_model_id, group_ids,
                workspace_id, created_by_user_id
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, 'pending', $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24)
        """,
        upload_id,
        bill_id,
        r2_key,
        telemetry_r2_key,
        data.filename,
        data.file_size,
        data.platforms,
        data.title,
        data.caption,
        data.hashtags,
        data.privacy,
        scheduled_time,
        data.schedule_mode,
        put_cost,
        aic_cost,
        json.dumps(billing_breakdown, default=str),
        _json_for_upload_row(schedule_metadata) if schedule_metadata else None,
        _json_for_upload_row(user_prefs),
        data.target_accounts or [],
        vm_id,
        vmd_id,
        resolved_group_ids or [],
        ws_id,
        member_id,
    )

    ledger_meta = {"billing_breakdown": billing_breakdown}
    if ws_ctx:
        ledger_meta["actor_user_id"] = member_id
        ledger_meta["workspace_id"] = ws_id
    reserved = await atomic_reserve_tokens(
        conn, bill_id, put_cost, aic_cost, upload_id, ledger_meta=ledger_meta
    )
    if not reserved:
        await conn.execute("DELETE FROM uploads WHERE id = $1", upload_id)
        fresh_wallet = await get_wallet(conn, bill_id)
        put_avail = fresh_wallet["put_balance"] - fresh_wallet["put_reserved"]
        aic_avail = fresh_wallet["aic_balance"] - fresh_wallet["aic_reserved"]
        if put_avail < put_cost:
            raise HTTPException(
                429,
                {
                    "code": "insufficient_put",
                    "message": f"Insufficient PUT tokens ({put_avail} available, {put_cost} needed).",
                    "topup_url": "/settings.html#billing",
                    "ledger_url": "/settings.html#token-balances",
                },
            )
        raise HTTPException(
            429,
            {
                "code": "insufficient_aic",
                "message": f"Insufficient AIC credits ({aic_avail} available, {aic_cost} needed).",
                "topup_url": "/settings.html#billing",
                "ledger_url": "/settings.html#token-balances",
            },
        )

    return {
        "upload_id": upload_id,
        "r2_key": r2_key,
        "telemetry_r2_key": telemetry_r2_key,
        "put_cost": put_cost,
        "aic_cost": aic_cost,
        "billing_breakdown": billing_breakdown,
        "user_prefs": user_prefs,
        "smart_schedule": smart_schedule,
    }


async def estimate_upload_costs(conn, data: UploadInit, user: dict) -> dict:
    """
    Preview PUT/AIC costs and queue depth without inserting a row or reserving tokens.
    """
    member_id = str(user["id"])
    ws_ctx = await get_workspace_for_user(conn, member_id)
    if ws_ctx and not can_upload_in_workspace(ws_ctx):
        raise HTTPException(403, "Viewers cannot create uploads")
    billing_user = ws_ctx.owner_row if ws_ctx else user
    bill_id = billing_user_id(ws_ctx, member_id)

    db_ent = await conn.fetchrow(
        "SELECT subscription_tier, role, flex_enabled FROM users WHERE id = $1",
        bill_id,
    )
    user_for_ent = dict(billing_user)
    if db_ent:
        for _k in ("subscription_tier", "role", "flex_enabled"):
            _v = db_ent.get(_k)
            if _v is not None:
                user_for_ent[_k] = _v
    ent_cost = get_entitlements_from_user(user_for_ent)

    platforms = list(getattr(data, "platforms", None) or [])
    target_accounts = list(getattr(data, "target_accounts", None) or [])
    group_ids_raw = getattr(data, "group_ids", None) or []
    if group_ids_raw:
        resolved_accounts, _ = await resolve_group_ids_to_target_accounts(
            conn, bill_id, group_ids_raw, platforms
        )
        target_accounts = resolved_accounts

    user_prefs = await get_user_prefs_for_upload(conn, bill_id)
    use_ai_checkbox = bool(getattr(data, "use_ai", False))
    num_publish_targets = len(target_accounts) if target_accounts else len(platforms)
    db_weights = await fetch_service_weights_map(conn)
    put_cost, aic_cost, billing_breakdown = compute_presign_put_aic_costs(
        ent_cost,
        num_publish_targets=num_publish_targets,
        file_size=getattr(data, "file_size", None),
        duration_hint=None,
        has_telemetry=bool(getattr(data, "has_telemetry", False)),
        use_ai_checkbox=use_ai_checkbox,
        user_prefs=user_prefs,
        num_thumbnails_override=None,
        service_weights_map=db_weights,
    )

    pending_count = await conn.fetchval(
        """SELECT COUNT(*) FROM uploads
           WHERE user_id = $1
           AND status IN ('pending','staged','queued','processing','ready_to_publish')""",
        bill_id,
    )
    queue_depth = ent_cost.queue_depth
    queue_depth_remaining = max(0, queue_depth - int(pending_count or 0))

    wallet = await get_wallet(conn, bill_id)
    put_avail = wallet["put_balance"] - wallet["put_reserved"]
    aic_avail = wallet["aic_balance"] - wallet["aic_reserved"]

    return {
        "put_cost": put_cost,
        "aic_cost": aic_cost,
        "billing_breakdown": billing_breakdown,
        "queue_depth": queue_depth,
        "queue_depth_pending": int(pending_count or 0),
        "queue_depth_remaining": queue_depth_remaining,
        "put_available": put_avail,
        "aic_available": aic_avail,
        "sufficient_put": put_avail >= put_cost,
        "sufficient_aic": aic_avail >= aic_cost,
    }
