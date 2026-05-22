"""
UploadM8 Admin Router — all /api/admin/* endpoints.
Admin HTTP surface; prefer thin handlers and ``services/`` for new logic.
"""

import importlib.util
import json
import os
import re
import uuid
import secrets
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import asyncpg
import bcrypt
import httpx
from fastapi import APIRouter, BackgroundTasks, Body, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

import core.state
from core.state import admin_settings_cache
from core.deps import get_current_user, require_admin, require_master_admin
from core.wallet import get_wallet
from core.notifications import discord_notify, notify_weekly_costs
from core.audit import log_admin_audit
from core.helpers import _now_utc, get_plan, _tier_is_upgrade, _valid_uuid, _safe_json, _safe_col
from core.sql_allowlist import (
    USERS_UPDATE_COLUMNS_ADMIN,
    assert_set_fragments_columns,
    assert_wallet_balance_column,
)
from core.config import (
    FRONTEND_URL,
    ADMIN_DISCORD_WEBHOOK_URL,
    COMMUNITY_DISCORD_WEBHOOK_URL,
)
from core.models import (
    AdminUserUpdate,
    AdminWalletAdjust,
    AdminUpdateEmailIn,
    AdminResetPasswordIn,
    AnnouncementRequest,
    NotificationSettings,
    AdminWalletDisputePatch,
)
from stages.entitlements import (
    TIER_CONFIG,
    TOPUP_PRODUCTS,
    PUBLIC_TIER_SLUGS,
    get_entitlements_for_tier,
    get_effective_tier_config,
    get_effective_topup_products,
    get_tiers_for_api,
    entitlements_to_dict,
    normalize_tier,
    tier_cfg_to_api_dict,
)
from services.admin_kpi_finance import build_admin_costs_summary, fetch_stripe_refunds_window
from services.admin_kpi_reliability import (
    fetch_admin_attach_metrics,
    fetch_admin_growth_metrics,
    fetch_admin_reliability_metrics,
)
from services.canonical_engagement import ROLLUP_VERSION, compute_admin_engagement_totals
from services import metric_definitions as metric_definitions_svc
from services.ml_hub_config import get_ml_hub_urls, ml_hub_huggingface_dict
from services.ml_observability import hf_write_token
from services.wallet_disputes import admin_patch_wallet_dispute, list_admin_wallet_disputes
from stages.emails import (
    send_email_change_email,
    send_admin_reset_password_email,
    send_announcement_email,
    send_friends_family_welcome_email,
    send_agency_welcome_email,
    send_master_admin_welcome_email,
    send_admin_wallet_topup_email,
    send_admin_tier_switch_email,
)

logger = logging.getLogger("uploadm8-api")

router = APIRouter(prefix="/api/admin", tags=["admin"])


class BillingWeightsPut(BaseModel):
    weights: Dict[str, int] = Field(default_factory=dict)
    put_cost_rules: Optional[Dict[str, int]] = None


class BillingCatalogPut(BaseModel):
    tier_overrides: Dict[str, Any] = Field(default_factory=dict)
    topup_overrides: Dict[str, Any] = Field(default_factory=dict)
    tier_service_overrides: Dict[str, Dict[str, bool]] = Field(default_factory=dict)


def _tier_json_key(tier: Any) -> str:
    """JSON object keys must be strings; DB subscription_tier can be NULL."""
    s = tier if tier is not None else ""
    s = str(s).strip() if s else ""
    return s if s else "free"


def _platform_upload_mix_sql() -> str:
    """Per-platform upload counts — valid on PostgreSQL (unnest + GROUP BY column)."""
    return """
        SELECT p AS platform, COUNT(*)::int AS uploads
        FROM uploads u,
             LATERAL unnest(COALESCE(u.platforms, ARRAY[]::text[])) AS u_p(p)
        WHERE u.created_at >= $1
        GROUP BY p
    """


def _tier_list_price_usd(tier: Any) -> float:
    """Monthly list price from TIER_CONFIG (get_plan / entitlements_to_dict omit price)."""
    slug = normalize_tier(str(tier) if tier is not None else "free")
    return float((get_effective_tier_config().get(slug) or {}).get("price", 0) or 0)


@router.get("/ml/observability-overview")
async def ml_observability_overview(user: dict = Depends(require_admin)):
    """
    Single-pane status for local ML artifacts + HF + Trackio wiring.
    No secrets are returned; only boolean/config health and local paths.
    """
    root = Path(__file__).resolve().parents[1]
    dataset_path = root / "data" / "ml" / "promo_targeting_train_v1.parquet"
    baseline_report_path = root / "data" / "ml" / "promo_targeting_baseline_report.json"

    hf_token = hf_write_token()
    trackio_project = (os.environ.get("TRACKIO_PROJECT") or "").strip()
    trackio_space = (os.environ.get("TRACKIO_SPACE_ID") or "").strip()
    try:
        trackio_package_importable = importlib.util.find_spec("trackio") is not None
    except Exception:
        trackio_package_importable = False

    hf_links = ml_hub_huggingface_dict()
    hub_urls = get_ml_hub_urls()
    summary: Dict[str, Any] = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "local": {
            "repo_root": str(root),
            "dataset_path": str(dataset_path),
            "dataset_exists": dataset_path.exists(),
            "baseline_report_path": str(baseline_report_path),
            "baseline_report_exists": baseline_report_path.exists(),
        },
        "huggingface": {
            "token_configured": bool(hf_token),
            **hf_links,
        },
        "hub_pages": {
            "dataset_url_configured": bool(hub_urls.get("dataset_url")),
            "trackio_space_url_configured": bool(hub_urls.get("trackio_space_url")),
        },
        "trackio": {
            "project_configured": bool(trackio_project),
            "project": trackio_project,
            "space_configured": bool(trackio_space),
            "space_id": trackio_space,
            "package_importable": trackio_package_importable,
        },
    }

    db: Dict[str, Any] = {}
    try:
        async with core.state.db_pool.acquire() as conn:
            db["uploads_count"] = int(await conn.fetchval("SELECT COUNT(*)::int FROM uploads") or 0)
            db["ml_outcome_labels_count"] = int(
                await conn.fetchval("SELECT COUNT(*)::int FROM ml_outcome_labels") or 0
            )
            db["marketing_touchpoint_deliveries_count"] = int(
                await conn.fetchval("SELECT COUNT(*)::int FROM marketing_touchpoint_deliveries") or 0
            )
            db["m8_model_runs_count"] = int(
                await conn.fetchval("SELECT COUNT(*)::int FROM m8_model_runs") or 0
            )
            db["upload_quality_scores_daily_count"] = int(
                await conn.fetchval("SELECT COUNT(*)::int FROM upload_quality_scores_daily") or 0
            )
            try:
                db["visual_entity_catalog_count"] = int(
                    await conn.fetchval("SELECT COUNT(*)::int FROM user_visual_entity_catalog") or 0
                )
                db["visual_entity_creator_count"] = int(
                    await conn.fetchval(
                        "SELECT COUNT(DISTINCT user_id)::int FROM user_visual_entity_catalog"
                    )
                    or 0
                )
            except Exception:
                db["visual_entity_catalog_count"] = 0
                db["visual_entity_creator_count"] = 0
            lr = await conn.fetchrow(
                """
                SELECT id::text AS id, trained_at, model_version, train_row_count, val_mae_log1p_views,
                       COALESCE(related_ops_incident_ids, ARRAY[]::uuid[]) AS related_ops_incident_ids
                FROM m8_model_runs
                ORDER BY trained_at DESC
                LIMIT 1
                """
            )
            if lr:
                db["latest_m8_model_run"] = {
                    "id": lr["id"],
                    "trained_at": lr["trained_at"].isoformat() if lr.get("trained_at") else None,
                    "model_version": lr["model_version"],
                    "train_row_count": int(lr["train_row_count"] or 0),
                    "val_mae_log1p_views": float(lr["val_mae_log1p_views"])
                    if lr.get("val_mae_log1p_views") is not None
                    else None,
                    "related_ops_incident_ids": [str(x) for x in (lr["related_ops_incident_ids"] or [])],
                }
            else:
                db["latest_m8_model_run"] = None
    except Exception as e:
        db["error"] = str(e)

    summary["database"] = db
    return summary


@router.get("/ml/observability-trends")
async def ml_observability_trends(days: int = Query(30, ge=7, le=90), user: dict = Depends(require_admin)):
    """
    Daily trend snapshots for ML observability page.
    """
    out: Dict[str, Any] = {"days": int(days), "as_of": datetime.now(timezone.utc).isoformat(), "series": {}}
    try:
        async with core.state.db_pool.acquire() as conn:
            outcome = await conn.fetch(
                """
                WITH d AS (
                    SELECT generate_series(
                        (CURRENT_DATE - ($1::int - 1)),
                        CURRENT_DATE,
                        INTERVAL '1 day'
                    )::date AS day
                )
                SELECT
                    d.day,
                    COALESCE(x.n, 0)::int AS count
                FROM d
                LEFT JOIN (
                    SELECT DATE(created_at) AS day, COUNT(*)::int AS n
                    FROM ml_outcome_labels
                    WHERE created_at >= (CURRENT_DATE - ($1::int - 1))
                    GROUP BY DATE(created_at)
                ) x ON x.day = d.day
                ORDER BY d.day
                """,
                int(days),
            )
            touchpoints = await conn.fetch(
                """
                WITH d AS (
                    SELECT generate_series(
                        (CURRENT_DATE - ($1::int - 1)),
                        CURRENT_DATE,
                        INTERVAL '1 day'
                    )::date AS day
                )
                SELECT
                    d.day,
                    COALESCE(x.n, 0)::int AS count
                FROM d
                LEFT JOIN (
                    SELECT DATE(created_at) AS day, COUNT(*)::int AS n
                    FROM marketing_touchpoint_deliveries
                    WHERE created_at >= (CURRENT_DATE - ($1::int - 1))
                    GROUP BY DATE(created_at)
                ) x ON x.day = d.day
                ORDER BY d.day
                """,
                int(days),
            )
            runs = await conn.fetch(
                """
                WITH d AS (
                    SELECT generate_series(
                        (CURRENT_DATE - ($1::int - 1)),
                        CURRENT_DATE,
                        INTERVAL '1 day'
                    )::date AS day
                )
                SELECT
                    d.day,
                    COALESCE(x.n, 0)::int AS count
                FROM d
                LEFT JOIN (
                    SELECT DATE(trained_at) AS day, COUNT(*)::int AS n
                    FROM m8_model_runs
                    WHERE trained_at >= (CURRENT_DATE - ($1::int - 1))
                    GROUP BY DATE(trained_at)
                ) x ON x.day = d.day
                ORDER BY d.day
                """,
                int(days),
            )

            def _pack(rows: List[Any]) -> List[Dict[str, Any]]:
                return [{"day": str(r["day"]), "count": int(r["count"] or 0)} for r in rows]

            out["series"] = {
                "ml_outcome_labels": _pack(outcome),
                "marketing_touchpoint_deliveries": _pack(touchpoints),
                "m8_model_runs": _pack(runs),
            }
    except Exception as e:
        out["error"] = str(e)
    return out


# ============================================================
# Time range helpers (used by several KPI endpoints)
# ============================================================

_RANGE_PRESETS_MINUTES = {
    "24h": 24 * 60,
    "7d": 7 * 24 * 60,
    "30d": 30 * 24 * 60,
    "90d": 90 * 24 * 60,
    "6m": 180 * 24 * 60,
    "1y": 365 * 24 * 60,
}

def _range_to_minutes(range_str: str | None, default_minutes: int) -> int:
    r = (range_str or "").strip()
    if not r:
        return default_minutes
    if r in _RANGE_PRESETS_MINUTES:
        return _RANGE_PRESETS_MINUTES[r]
    m = re.fullmatch(r"(\d{1,4})d", r)
    if m:
        days = int(m.group(1))
        # Guardrails: 1 day .. 10 years
        days = max(1, min(days, 3650))
        return days * 24 * 60
    return default_minutes

def _range_label(range_str: str | None, fallback: str = "30d") -> str:
    r = (range_str or "").strip()
    return r if r else fallback


# ============================================================
# Announcement helpers
# ============================================================

def _normalize_announcement_channels(channels_in, send_email: bool, send_discord_community: bool, send_user_webhooks: bool):
    """Return normalized channel list: ['email','discord_community','user_webhook']"""
    out = []
    # New-style list/dict support (if frontend starts sending channels explicitly)
    if isinstance(channels_in, dict):
        for k, v in channels_in.items():
            if v is True:
                out.append(str(k))
    elif isinstance(channels_in, list):
        out.extend([str(c) for c in channels_in])

    # Legacy booleans always win (existing UI contract)
    if send_email and "email" not in out:
        out.append("email")
    if send_discord_community and "discord_community" not in out:
        out.append("discord_community")
    if send_user_webhooks and "user_webhook" not in out and "user_webhooks" not in out:
        out.append("user_webhook")

    # Normalize key spelling
    out = ["user_webhook" if c == "user_webhooks" else c for c in out]
    # De-dupe, preserve order
    seen = set()
    norm = []
    for c in out:
        c = (c or "").strip()
        if not c:
            continue
        if c in seen:
            continue
        if c not in ("email", "discord_community", "user_webhook"):
            continue
        seen.add(c)
        norm.append(c)
    return norm

def _channels_to_store_map(channels_list):
    """Persist in the same shape your DB currently uses (json text map)."""
    store = {"email": False, "discord_community": False, "user_webhooks": False}
    for c in channels_list:
        if c == "email":
            store["email"] = True
        elif c == "discord_community":
            store["discord_community"] = True
        elif c == "user_webhook":
            store["user_webhooks"] = True
    return json.dumps(store)

async def _discord_post_raw(webhook_url: str, *, content: str = None, embeds: list = None):
    """Return (ok, err). Unlike discord_notify(), this does not swallow failures."""
    if not webhook_url:
        return False, "missing webhook url"
    payload = {}
    if content:
        payload["content"] = content[:1999]
    if embeds:
        payload["embeds"] = embeds
    try:
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.post(webhook_url, json=payload)
        if r.status_code in (200, 204):
            return True, ""
        return False, f"{r.status_code}: {r.text[:300]}"
    except Exception as e:
        return False, str(e)

async def _insert_delivery_intents(conn, announcement_id: str, *, sender_user_id: str, recipients: list, channels_list: list, title: str, body: str):
    """
    Insert queued intents into announcement_deliveries, idempotent by (announcement_id,user_id,channel).
    - email: one row per recipient (destination=email)
    - user_webhook: one row per recipient with webhook configured (destination=webhook)
    - discord_community: single row owned by sender (destination=global webhook)
    """
    now = _now_utc()

    inserts = []

    # Community webhook is a single delivery owned by sender
    if "discord_community" in channels_list and COMMUNITY_DISCORD_WEBHOOK_URL:
        inserts.append((announcement_id, sender_user_id, "discord_community", COMMUNITY_DISCORD_WEBHOOK_URL, "queued", None, now))

    # Per-user deliveries
    for r in recipients:
        uid = str(r.get("id") or r.get("user_id") or "")
        email = (r.get("email") or "").strip()
        webhook = (r.get("discord_webhook") or "").strip()

        if not uid:
            continue

        if "email" in channels_list and email:
            inserts.append((announcement_id, uid, "email", email, "queued", None, now))

        if "user_webhook" in channels_list and webhook:
            inserts.append((announcement_id, uid, "user_webhook", webhook, "queued", None, now))

    if not inserts:
        return 0

    await conn.executemany(
        """
        INSERT INTO announcement_deliveries
          (announcement_id, user_id, channel, destination, status, error, created_at)
        SELECT $1::uuid, $2::uuid, $3::text, $4::text, $5::text, $6::text, $7::timestamptz
        WHERE NOT EXISTS (
          SELECT 1 FROM announcement_deliveries
          WHERE announcement_id = $1::uuid AND user_id = $2::uuid AND channel = $3::text
        )
        """,
        inserts,
    )
    return len(inserts)

async def _execute_announcement_deliveries(conn, announcement_id: str, title: str, body: str):
    """Execute queued deliveries for an announcement and update rollups."""
    from services.marketing_promo_media import resolve_promo_presigned_url

    ann_row = await conn.fetchrow(
        "SELECT promo_media FROM announcements WHERE id = $1::uuid",
        announcement_id,
    )
    hero_url = ""
    if ann_row:
        u = await resolve_promo_presigned_url(conn, ann_row.get("promo_media"))
        hero_url = (u or "").strip()

    rows = await conn.fetch(
        """
        SELECT id, channel, destination
        FROM announcement_deliveries
        WHERE announcement_id = $1::uuid AND status = 'queued'
        ORDER BY id ASC
        """,
        announcement_id,
    )

    email_sent = 0
    discord_sent = 0
    webhook_sent = 0

    for r in rows:
        delivery_id = r["id"]
        ch = r["channel"]
        dest = (r["destination"] or "").strip()

        ok = False
        err = ""

        if ch == "email":
            try:
                await send_announcement_email(dest, title, body, hero_image_url=hero_url)
                ok = True
                email_sent += 1
            except Exception as e:
                ok = False
                err = str(e)

        elif ch == "discord_community":
            emb: Dict[str, Any] = {"title": f"\U0001f4e2 {title}", "description": body, "color": 0xf97316}
            if hero_url:
                emb["image"] = {"url": hero_url[:2048]}
            ok, err = await _discord_post_raw(dest, embeds=[emb])
            if ok:
                discord_sent += 1

        elif ch == "user_webhook":
            emb2: Dict[str, Any] = {"title": f"\U0001f4e2 {title}", "description": body, "color": 0xf97316}
            if hero_url:
                emb2["image"] = {"url": hero_url[:2048]}
            ok, err = await _discord_post_raw(dest, embeds=[emb2])
            if ok:
                webhook_sent += 1

        status = "sent" if ok else "failed"
        await conn.execute(
            """
            UPDATE announcement_deliveries
            SET status = $2,
                error = $3,
                sent_at = CASE WHEN $2='sent' THEN NOW() ELSE sent_at END
            WHERE id = $1
            """,
            delivery_id,
            status,
            (err or None),
        )

    await conn.execute(
        """
        UPDATE announcements
        SET email_sent = COALESCE(email_sent,0) + $2,
            discord_sent = COALESCE(discord_sent,0) + $3,
            webhook_sent = COALESCE(webhook_sent,0) + $4
        WHERE id = $1::uuid
        """,
        announcement_id,
        int(email_sent),
        int(discord_sent),
        int(webhook_sent),
    )

    return {"email": email_sent, "discord_community": discord_sent, "user_webhook": webhook_sent}


# ============================================================
# Discord notification helpers
# ============================================================

def get_notif_settings():
    return admin_settings_cache.get("notifications", {})

def get_admin_webhook():
    settings = get_notif_settings()
    return settings.get("admin_webhook_url") or ADMIN_DISCORD_WEBHOOK_URL

def mask_email(email: str) -> str:
    if not email or "@" not in email: return email
    local, domain = email.split("@", 1)
    return f"{local[:2]}***@{domain}" if len(local) > 2 else f"{local[0]}***@{domain}"


async def notify_revenue_event(event_type: str, email: str, tier: str, amount: float, stripe_id: str = None, extra_fields: list = None):
    """Send revenue event notification to Discord"""
    settings = get_notif_settings()
    webhook = get_admin_webhook()
    if not webhook: return
    setting_map = {"mrr_charge": "notify_mrr_charge", "topup": "notify_topup", "upgrade": "notify_upgrade", "downgrade": "notify_downgrade", "cancel": "notify_cancel", "refund": "notify_refund"}
    if setting_map.get(event_type) and not settings.get(setting_map[event_type], True): return
    event_config = {
        "mrr_charge": {"emoji": "\U0001f4b0", "color": 0x22c55e, "title": "MRR Charge"},
        "topup": {"emoji": "\U0001f4b3", "color": 0x8b5cf6, "title": "Top-up Purchase"},
        "upgrade": {"emoji": "\u2b06\ufe0f", "color": 0x3b82f6, "title": "Plan Upgrade"},
        "downgrade": {"emoji": "\u2b07\ufe0f", "color": 0xf59e0b, "title": "Plan Downgrade"},
        "cancel": {"emoji": "\u274c", "color": 0xef4444, "title": "Subscription Cancelled"},
        "refund": {"emoji": "\u21a9\ufe0f", "color": 0xf97316, "title": "Refund Processed"},
    }
    cfg = event_config.get(event_type, {"emoji": "\U0001f4ca", "color": 0x6b7280, "title": event_type.title()})
    fields = [{"name": "User", "value": mask_email(email), "inline": True}, {"name": "Tier", "value": tier.title(), "inline": True}, {"name": "Amount", "value": f"${amount:.2f}", "inline": True}]
    if stripe_id: fields.append({"name": "Stripe ID", "value": f"`{stripe_id[:20]}...`" if len(stripe_id) > 20 else f"`{stripe_id}`", "inline": False})
    if extra_fields: fields.extend(extra_fields)
    await discord_notify(webhook, embeds=[{"title": f"{cfg['emoji']} {cfg['title']}", "color": cfg["color"], "fields": fields, "footer": {"text": "UploadM8 Revenue Alert"}, "timestamp": _now_utc().isoformat()}])


async def notify_cost_report(report_type: str, costs: dict, period: str = "Weekly"):
    """Send cost report notification to Discord"""
    settings = get_notif_settings()
    webhook = get_admin_webhook()
    if not webhook: return
    setting_map = {"openai": "notify_openai_cost", "storage": "notify_storage_cost", "compute": "notify_compute_cost", "weekly": "notify_weekly_report"}
    if setting_map.get(report_type) and not settings.get(setting_map[report_type], True): return
    if report_type == "weekly":
        total_cost = costs.get("openai", 0) + costs.get("storage", 0) + costs.get("compute", 0)
        revenue = costs.get("revenue", 0)
        margin = revenue - total_cost
        margin_pct = (margin / max(revenue, 1)) * 100
        embed = {"title": "\U0001f4ca Weekly Cost Report", "color": 0x3b82f6, "fields": [
            {"name": "OpenAI", "value": f"${costs.get('openai', 0):.2f}", "inline": True},
            {"name": "Storage", "value": f"${costs.get('storage', 0):.2f}", "inline": True},
            {"name": "Compute", "value": f"${costs.get('compute', 0):.2f}", "inline": True},
            {"name": "Total COGS", "value": f"${total_cost:.2f}", "inline": True},
            {"name": "Revenue", "value": f"${revenue:.2f}", "inline": True},
            {"name": "Margin", "value": f"${margin:.2f} ({margin_pct:.1f}%)", "inline": True},
        ], "footer": {"text": f"UploadM8 {period} Report"}, "timestamp": _now_utc().isoformat()}
    else:
        titles = {"openai": "\U0001f916 OpenAI Cost", "storage": "\U0001f4be Storage Cost", "compute": "\u26a1 Compute Cost"}
        embed = {"title": titles.get(report_type, "\U0001f4c8 Cost Report"), "color": 0xef4444, "fields": [
            {"name": "Cost", "value": f"${costs.get('amount', 0):.2f}", "inline": True},
            {"name": "Units", "value": str(costs.get("units", "N/A")), "inline": True},
            {"name": "vs Last Week", "value": f"{costs.get('change', 0):+.1f}%", "inline": True},
        ], "footer": {"text": f"UploadM8 {period} Report"}, "timestamp": _now_utc().isoformat()}
    await discord_notify(webhook, embeds=[embed])


async def notify_billing_reminder(reminder_type: str, date: str, amount: float = None, service: str = None):
    """Send billing calendar reminder to Discord"""
    settings = get_notif_settings()
    webhook = get_admin_webhook()
    if not webhook: return
    setting_map = {"stripe_payout": "notify_stripe_payout", "cloud_billing": "notify_cloud_billing", "render_renewal": "notify_render_renewal"}
    if setting_map.get(reminder_type) and not settings.get(setting_map[reminder_type], True): return
    config = {"stripe_payout": {"emoji": "\U0001f4b8", "color": 0x6366f1, "title": "Stripe Payout Coming"}, "cloud_billing": {"emoji": "\u2601\ufe0f", "color": 0xf97316, "title": "Cloud Billing Reminder"}, "render_renewal": {"emoji": "\U0001f680", "color": 0x06b6d4, "title": "Render Renewal Reminder"}}
    cfg = config.get(reminder_type, {"emoji": "\U0001f4c5", "color": 0x6b7280, "title": "Billing Reminder"})
    fields = [{"name": "Date", "value": date, "inline": True}]
    if service: fields.append({"name": "Service", "value": service, "inline": True})
    if amount: fields.append({"name": "Est. Amount", "value": f"${amount:.2f}", "inline": True})
    await discord_notify(webhook, embeds=[{"title": f"{cfg['emoji']} {cfg['title']}", "description": "Upcoming billing event in 2 days", "color": cfg["color"], "fields": fields, "footer": {"text": "UploadM8 Billing Calendar"}, "timestamp": _now_utc().isoformat()}])


# ============================================================
# Internal announcement sender (called by the two POST routes)
# ============================================================

async def send_announcement(data: AnnouncementRequest, background_tasks: BackgroundTasks, user: dict):
    """Creates announcement + idempotent delivery intents, then executes queued deliveries."""
    title = (data.title or "").strip()
    body = (data.body or "").strip()
    if not title or not body:
        raise HTTPException(status_code=400, detail="title and body are required")

    async with core.state.db_pool.acquire() as conn:
        # ----------------------------
        # Resolve recipients (banned excluded)
        # ----------------------------
        query = """
            SELECT
              u.id::text AS id,
              u.email,
              u.subscription_tier,
              COALESCE(
                NULLIF(TRIM(us.discord_webhook), ''),
                NULLIF(TRIM(up.discord_webhook), ''),
                NULLIF(TRIM(COALESCE(u.preferences->>'discordWebhook', u.preferences->>'discord_webhook')), ''),
                ''
              ) AS discord_webhook
            FROM users u
            LEFT JOIN user_settings us ON us.user_id = u.id
            LEFT JOIN user_preferences up ON up.user_id = u.id
            WHERE COALESCE(u.status,'active') <> 'banned'
        """
        params = []

        if getattr(data, "target", None) == "paid":
            query += " AND u.subscription_tier NOT IN ('free')"
        elif getattr(data, "target", None) == "free":
            query += " AND u.subscription_tier = 'free'"
        elif getattr(data, "target", None) in ("specific_tiers", "tiers") and getattr(data, "target_tiers", None):
            params.append(list(data.target_tiers))
            query += f" AND u.subscription_tier = ANY(${len(params)}::text[]) "

        recipients = await conn.fetch(query, *params) if params else await conn.fetch(query)
        recipients_list = [dict(r) for r in recipients]

        # ----------------------------
        # Normalize channels
        # ----------------------------
        channels_list = _normalize_announcement_channels(
            getattr(data, "channels", None) if hasattr(data, "channels") else None,
            bool(getattr(data, "send_email", False)),
            bool(getattr(data, "send_discord_community", False)),
            bool(getattr(data, "send_user_webhooks", False)),
        )
        if not channels_list:
            raise HTTPException(status_code=400, detail="No channels selected")

        # Persist in the DB using your current storage shape (json text map)
        channels_store = _channels_to_store_map(channels_list)

        # ----------------------------
        # Insert announcement row
        # ----------------------------
        ann_id = str(uuid.uuid4())
        pm_raw = getattr(data, "promo_media", None)
        pm_row: Dict[str, Any] = pm_raw if isinstance(pm_raw, dict) else {}
        await conn.execute(
            """
            INSERT INTO announcements (id, title, body, channels, target, target_tiers, created_by, promo_media)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb)
            """,
            ann_id,
            title,
            body,
            channels_store,
            getattr(data, "target", "all"),
            getattr(data, "target_tiers", None),
            user["id"],
            json.dumps(pm_row),
        )

        # ----------------------------
        # Insert idempotent delivery intents
        # ----------------------------
        await _insert_delivery_intents(
            conn,
            ann_id,
            sender_user_id=str(user["id"]),
            recipients=recipients_list,
            channels_list=channels_list,
            title=title,
            body=body,
        )

        # ----------------------------
        # Execute fanout (background). For debugging, you can run inline.
        # ----------------------------
        async def _run():
            async with core.state.db_pool.acquire() as c2:
                return await _execute_announcement_deliveries(c2, ann_id, title, body)

        background_tasks.add_task(_run)

    return {"status": "queued", "announcement_id": ann_id, "recipients": len(recipients_list), "channels": channels_list}


# ============================================================
# USER MANAGEMENT
# ============================================================

@router.get("/users")
async def admin_get_users(search: Optional[str] = None, tier: Optional[str] = None, limit: int = 50, offset: int = 0, user: dict = Depends(require_admin)):
    query = "SELECT id, email, name, role, subscription_tier, subscription_status, status, created_at, last_active_at FROM users WHERE 1=1"
    params = []
    if search:
        params.append(f"%{search}%")
        query += f" AND (email ILIKE ${len(params)} OR name ILIKE ${len(params)})"
    if tier:
        params.append(tier)
        query += f" AND subscription_tier = ${len(params)}"
    params.extend([limit, offset])
    query += f" ORDER BY created_at DESC LIMIT ${len(params)-1} OFFSET ${len(params)}"

    async with core.state.require_pool().acquire() as conn:
        users = await conn.fetch(query, *params)
        total = await conn.fetchval("SELECT COUNT(*) FROM users")
    rows = []
    for u in users:
        rows.append(
            {
                "id": str(u["id"]),
                "email": u["email"],
                "name": u["name"],
                "role": u["role"],
                "subscription_tier": u["subscription_tier"],
                "subscription_status": u["subscription_status"],
                "status": u["status"],
                "created_at": u["created_at"].isoformat() if u.get("created_at") else None,
                "last_active_at": u["last_active_at"].isoformat() if u.get("last_active_at") else None,
            }
        )
    return {"users": rows, "total": total}

@router.put("/users/{user_id}")
async def admin_update_user(user_id: str, data: AdminUserUpdate, request: Request, background_tasks: BackgroundTasks, user: dict = Depends(require_admin)):
    _ADMIN_USER_COLS = USERS_UPDATE_COLUMNS_ADMIN
    updates, params = [], [user_id]
    changes = {}
    if data.subscription_tier:
        updates.append(f"{_safe_col('subscription_tier', _ADMIN_USER_COLS)} = ${len(params)+1}")
        params.append(data.subscription_tier)
        changes["subscription_tier"] = data.subscription_tier
    if data.role and user.get("role") == "master_admin":
        updates.append(f"{_safe_col('role', _ADMIN_USER_COLS)} = ${len(params)+1}")
        params.append(data.role)
        changes["role"] = data.role
    if data.status:
        updates.append(f"{_safe_col('status', _ADMIN_USER_COLS)} = ${len(params)+1}")
        params.append(data.status)
        changes["status"] = data.status
    if data.flex_enabled is not None:
        updates.append(f"{_safe_col('flex_enabled', _ADMIN_USER_COLS)} = ${len(params)+1}")
        params.append(data.flex_enabled)
        changes["flex_enabled"] = data.flex_enabled
    if updates:
        assert_set_fragments_columns(updates, USERS_UPDATE_COLUMNS_ADMIN)
        async with core.state.db_pool.acquire() as conn:
            # Fetch target before updating so we have old tier
            _target = await conn.fetchrow("SELECT email, name, subscription_tier FROM users WHERE id = $1", user_id)
            await conn.execute(f"UPDATE users SET {', '.join(updates)}, updated_at = NOW() WHERE id = $1", *params)
            await log_admin_audit(conn, user_id=user_id, admin=user, action="ADMIN_UPDATE_USER",
                                  details={"changes": changes}, request=request,
                                  resource_type="user", resource_id=user_id)

        # Fire tier-change and special welcome emails when subscription_tier changed
        if data.subscription_tier and _target:
            _te = _target["email"]
            _tn = _target["name"] or "there"
            _old = _target["subscription_tier"] or "free"
            _new = data.subscription_tier
            if _old != _new:
                background_tasks.add_task(
                    send_admin_tier_switch_email,
                    _te, _tn, _old, _new, "", _tier_is_upgrade(_old, _new),
                )
            # Special heartfelt welcome for privileged tiers
            if _new == "friends_family":
                background_tasks.add_task(send_friends_family_welcome_email, _te, _tn)
            elif _new == "agency":
                background_tasks.add_task(send_agency_welcome_email, _te, _tn)
            elif _new == "master_admin":
                background_tasks.add_task(send_master_admin_welcome_email, _te, _tn)

    return {"status": "updated"}

@router.post("/users/{user_id}/ban")
async def admin_ban_user(user_id: str, request: Request, user: dict = Depends(require_admin)):
    async with core.state.db_pool.acquire() as conn:
        target = await conn.fetchrow("SELECT email FROM users WHERE id = $1", user_id)
        await conn.execute("UPDATE users SET status = 'banned' WHERE id = $1", user_id)
        await log_admin_audit(conn, user_id=user_id, admin=user, action="ADMIN_BAN_USER",
                              details={"target_email": target["email"] if target else None},
                              request=request, resource_type="user", resource_id=user_id,
                              severity="WARNING")
    return {"status": "banned"}

@router.post("/users/{user_id}/unban")
async def admin_unban_user(user_id: str, request: Request, user: dict = Depends(require_admin)):
    async with core.state.db_pool.acquire() as conn:
        target = await conn.fetchrow("SELECT email FROM users WHERE id = $1", user_id)
        await conn.execute("UPDATE users SET status = 'active' WHERE id = $1", user_id)
        await log_admin_audit(conn, user_id=user_id, admin=user, action="ADMIN_UNBAN_USER",
                              details={"target_email": target["email"] if target else None},
                              request=request, resource_type="user", resource_id=user_id)
    return {"status": "unbanned"}


@router.put("/users/{user_id}/email")
async def admin_change_email(user_id: str, payload: AdminUpdateEmailIn, request: Request, background_tasks: BackgroundTasks, user: dict = Depends(get_current_user)):
    require_admin(user)
    new_email = payload.email.lower().strip()

    async with core.state.db_pool.acquire() as conn:
        exists = await conn.fetchval(
            "SELECT 1 FROM users WHERE LOWER(email)=LOWER($1) AND id <> $2",
            new_email,
            user_id,
        )
        if exists:
            raise HTTPException(status_code=409, detail="Email already in use")

        old = await conn.fetchrow("SELECT email, name FROM users WHERE id=$1", user_id)
        if not old:
            raise HTTPException(status_code=404, detail="User not found")

        verification_token = secrets.token_urlsafe(32)

        await conn.execute(
            """
            INSERT INTO email_changes (user_id, old_email, new_email, changed_by_admin_id, verification_token)
            VALUES ($1::uuid, $2, $3, $4::uuid, $5)
            """,
            user_id,
            old["email"],
            new_email,
            user["id"],
            verification_token,
        )

        await conn.execute(
            "UPDATE users SET email=$1, email_verified=false, updated_at=NOW() WHERE id=$2",
            new_email,
            user_id,
        )

        await log_admin_audit(
            conn,
            user_id=user_id,
            admin=user,
            action="ADMIN_CHANGE_EMAIL",
            details={"old_email": old["email"], "new_email": new_email},
            request=request,
        )

    # Send verification email to the new address (name = target user, not admin)
    _verify_link = f"{FRONTEND_URL}/verify-email?token={verification_token}"
    _target_name = old.get("name") or "there"
    background_tasks.add_task(
        send_email_change_email,
        new_email, old["email"], _target_name, _verify_link
    )

    return {"ok": True, "email": new_email}


@router.post("/users/{user_id}/reset-password")
async def admin_reset_password(user_id: str, payload: AdminResetPasswordIn, request: Request, background_tasks: BackgroundTasks, user: dict = Depends(get_current_user)):
    require_admin(user)
    temp = payload.temp_password
    pw_hash = bcrypt.hashpw(temp.encode("utf-8"), bcrypt.gensalt(12)).decode("utf-8")

    async with core.state.db_pool.acquire() as conn:
        target = await conn.fetchrow("SELECT id, role FROM users WHERE id=$1", user_id)
        if not target:
            raise HTTPException(status_code=404, detail="User not found")
        if target["role"] == "master_admin" and user.get("role") != "master_admin":
            raise HTTPException(status_code=403, detail="Cannot reset master_admin password")

        await conn.execute(
            """
            UPDATE users
            SET password_hash=$1,
                must_reset_password=true,
                updated_at=NOW()
            WHERE id=$2
            """,
            pw_hash,
            user_id,
        )

        await conn.execute(
            """
            INSERT INTO password_resets (user_id, reset_by_admin_id, temp_password_hash, force_change, expires_at)
            VALUES ($1::uuid, $2::uuid, $3, TRUE, NOW() + INTERVAL '7 days')
            """,
            user_id,
            user["id"],
            pw_hash,
        )

        await log_admin_audit(
            conn,
            user_id=user_id,
            admin=user,
            action="ADMIN_RESET_PASSWORD",
            details={"must_reset_password": True},
            request=request,
        )

    # Email the user their temporary password
    async with core.state.db_pool.acquire() as _ec:
        _tgt = await _ec.fetchrow("SELECT email, name FROM users WHERE id=$1", user_id)
    if _tgt:
        background_tasks.add_task(send_admin_reset_password_email, _tgt["email"], _tgt["name"] or "there", payload.temp_password)

    return {"ok": True}


@router.post("/users/assign-tier")
async def admin_assign_tier(user_id: str = Query(...), tier: str = Query(...), request: Request = None, background_tasks: BackgroundTasks = None, user: dict = Depends(require_master_admin)):
    if tier not in TIER_CONFIG:
        raise HTTPException(400, "Invalid tier")
    async with core.state.db_pool.acquire() as conn:
        old = await conn.fetchrow("SELECT subscription_tier, email, name FROM users WHERE id = $1", user_id)
        await conn.execute("UPDATE users SET subscription_tier = $1 WHERE id = $2", tier, user_id)
        await log_admin_audit(conn, user_id=user_id, admin=user, action="ADMIN_ASSIGN_TIER",
                              details={"old_tier": old["subscription_tier"] if old else None, "new_tier": tier,
                                       "target_email": old["email"] if old else None},
                              request=request, resource_type="user", resource_id=user_id,
                              severity="WARNING")

    if old and background_tasks:
        _te  = old["email"]
        _tn  = old["name"] or "there"
        _old = old["subscription_tier"] or "free"
        if _old != tier:
            background_tasks.add_task(
                send_admin_tier_switch_email,
                _te, _tn, _old, tier, "", _tier_is_upgrade(_old, tier),
            )
        if tier == "friends_family":
            background_tasks.add_task(send_friends_family_welcome_email, _te, _tn)
        elif tier == "agency":
            background_tasks.add_task(send_agency_welcome_email, _te, _tn)
        elif tier == "master_admin":
            background_tasks.add_task(send_master_admin_welcome_email, _te, _tn)

    return {"status": "assigned", "tier": tier}


# ============================================================
# AUDIT
# ============================================================

@router.get("/audit")
async def admin_audit(
    user_id: Optional[str] = None,
    event_category: Optional[str] = None,
    action: Optional[str] = None,
    severity: Optional[str] = None,
    source: str = "all",           # "all" | "admin" | "system"
    limit: int = 100,
    offset: int = 0,
    user: dict = Depends(get_current_user)
):
    """
    Corporate-grade audit log endpoint.
    - Returns rolling 6-month window across both admin_audit_log and system_event_log
    - Supports filtering by category, action, severity, user, source table
    - Auto-purges records older than 6 months on each call (once per hour max via in-memory flag)
    - Pagination via limit/offset
    """
    require_admin(user)
    limit = max(1, min(limit, 500))
    offset = max(0, offset)

    def _ser(v):
        if v is None: return None
        if hasattr(v, "isoformat"): return v.isoformat()
        if isinstance(v, uuid.UUID): return str(v)
        if isinstance(v, (dict, list)): return v
        return v

    def _ser_row(r: dict) -> dict:
        return {k: _ser(v) for k, v in r.items()}

    try:
        async with core.state.db_pool.acquire() as conn:
            # -- Background purge (rolling 6-month window) --
            try:
                await conn.execute(
                    "DELETE FROM admin_audit_log WHERE created_at < NOW() - INTERVAL '6 months'"
                )
                await conn.execute(
                    "DELETE FROM system_event_log WHERE created_at < NOW() - INTERVAL '6 months'"
                )
            except Exception as purge_err:
                logger.warning(f"[audit] Purge skipped: {purge_err}")

            items = []

            # -- Build admin_audit_log query --
            if source in ("all", "admin"):
                try:
                    aq = """
                        SELECT
                            id, 'admin_audit' AS log_source,
                            COALESCE(event_category, 'ADMIN') AS event_category,
                            action,
                            COALESCE(actor_user_id, admin_id) AS actor_id,
                            admin_email AS actor_email,
                            user_id AS target_user_id,
                            resource_type, resource_id,
                            details, ip_address, user_agent,
                            COALESCE(severity, 'INFO') AS severity,
                            COALESCE(outcome, 'SUCCESS') AS outcome,
                            created_at
                        FROM admin_audit_log
                        WHERE created_at >= NOW() - INTERVAL '6 months'
                    """
                    aargs: List[Any] = []
                    if user_id:
                        aargs.append(user_id)
                        aq += f" AND (user_id = ${len(aargs)}::uuid OR admin_id = ${len(aargs)}::uuid)"
                    if event_category:
                        aargs.append(event_category.upper())
                        aq += f" AND UPPER(COALESCE(event_category,'ADMIN')) = ${len(aargs)}"
                    if action:
                        aargs.append(f"%{action.upper()}%")
                        aq += f" AND UPPER(action) LIKE ${len(aargs)}"
                    if severity:
                        aargs.append(severity.upper())
                        aq += f" AND UPPER(COALESCE(severity,'INFO')) = ${len(aargs)}"

                    admin_rows = await conn.fetch(aq, *aargs)
                    items.extend([_ser_row(dict(r)) for r in admin_rows])
                except Exception as ae:
                    logger.warning(f"[audit] admin_audit_log query failed: {ae}")

            # -- Build system_event_log query --
            if source in ("all", "system"):
                try:
                    sq = """
                        SELECT
                            id, 'system_event' AS log_source,
                            event_category,
                            action,
                            user_id AS actor_id,
                            NULL AS actor_email,
                            user_id AS target_user_id,
                            resource_type, resource_id,
                            details, ip_address, user_agent,
                            COALESCE(severity, 'INFO') AS severity,
                            COALESCE(outcome, 'SUCCESS') AS outcome,
                            created_at
                        FROM system_event_log
                        WHERE created_at >= NOW() - INTERVAL '6 months'
                    """
                    sargs: List[Any] = []
                    if user_id:
                        sargs.append(user_id)
                        sq += f" AND user_id = ${len(sargs)}::uuid"
                    if event_category:
                        sargs.append(event_category.upper())
                        sq += f" AND UPPER(event_category) = ${len(sargs)}"
                    if action:
                        sargs.append(f"%{action.upper()}%")
                        sq += f" AND UPPER(action) LIKE ${len(sargs)}"
                    if severity:
                        sargs.append(severity.upper())
                        sq += f" AND UPPER(COALESCE(severity,'INFO')) = ${len(sargs)}"

                    sys_rows = await conn.fetch(sq, *sargs)
                    items.extend([_ser_row(dict(r)) for r in sys_rows])
                except Exception as se:
                    logger.warning(f"[audit] system_event_log query failed: {se}")

            # Sort combined by created_at desc, paginate
            items.sort(key=lambda x: x.get("created_at") or "", reverse=True)
            total = len(items)
            page = items[offset: offset + limit]

            # Enrich with actor name from users table where possible
            actor_ids = list({x["actor_id"] for x in page if x.get("actor_id")})
            actor_map = {}
            if actor_ids:
                try:
                    urows = await conn.fetch(
                        "SELECT id, email, name FROM users WHERE id = ANY($1::uuid[])",
                        [str(a) for a in actor_ids]
                    )
                    for ur in urows:
                        actor_map[str(ur["id"])] = {"email": ur["email"], "name": ur["name"]}
                except Exception:
                    pass

            for item in page:
                aid = str(item.get("actor_id") or "")
                if aid in actor_map:
                    item["actor_email"] = item.get("actor_email") or actor_map[aid]["email"]
                    item["actor_name"] = actor_map[aid]["name"]

        return {
            "items": page,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": (offset + limit) < total,
        }

    except Exception as e:
        logger.error(f"[admin_audit] Error fetching audit log: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Audit log error: {str(e)}")


_DATA_INTEGRITY_SQL_FILTER = """
    created_at >= $1
    AND (
        UPPER(COALESCE(event_category, '')) LIKE '%INTEGRITY%'
        OR UPPER(COALESCE(action, '')) LIKE '%INTEGRITY%'
        OR UPPER(COALESCE(action, '')) LIKE '%ROLLUP%'
        OR UPPER(COALESCE(action, '')) LIKE '%RECONCILE%'
        OR UPPER(COALESCE(action, '')) LIKE '%DATA_QUALITY%'
    )
"""


@router.get("/audit/data-integrity")
async def admin_audit_data_integrity(
    severity: Optional[str] = Query(None),
    since_hours: int = Query(72, ge=1, le=720),
    limit: int = Query(200, ge=1, le=500),
    offset: int = Query(0, ge=0),
    user: dict = Depends(require_admin),
):
    """Data-integrity style events for admin-data-integrity.html and dashboard badge."""
    since = _now_utc() - timedelta(hours=since_hours)
    sev = (severity or "").strip().upper() or None
    cap = min(2500, max(limit + offset, 500) * 5)
    merged: List[Dict[str, Any]] = []

    async with core.state.db_pool.acquire() as conn:
        for table in ("admin_audit_log", "system_event_log"):
            try:
                q = f"""
                    SELECT action, details,
                           COALESCE(severity, 'INFO') AS severity,
                           COALESCE(outcome, 'SUCCESS') AS outcome,
                           user_id, created_at
                    FROM {table}
                    WHERE {_DATA_INTEGRITY_SQL_FILTER}
                """
                args: List[Any] = [since]
                if sev:
                    args.append(sev)
                    q += f" AND UPPER(COALESCE(severity, 'INFO')) = ${len(args)}"
                q += f" ORDER BY created_at DESC LIMIT {cap}"
                for r in await conn.fetch(q, *args):
                    d = _safe_json(dict(r).get("details"), {})
                    merged.append(
                        {
                            "created_at": r["created_at"].isoformat() if r.get("created_at") else None,
                            "severity": r.get("severity"),
                            "action": r.get("action"),
                            "outcome": r.get("outcome"),
                            "user_id": str(r["user_id"]) if r.get("user_id") else None,
                            "details": d if isinstance(d, dict) else {"raw": d},
                        }
                    )
            except Exception as e:
                logger.warning("data-integrity %s: %s", table, e)

    merged.sort(key=lambda x: x.get("created_at") or "", reverse=True)

    def _cnt(pred):
        return sum(1 for x in merged if pred(x))

    summary = {
        "total": len(merged),
        "failed": _cnt(lambda x: str(x.get("outcome") or "").upper() != "SUCCESS"),
        "corrected": _cnt(lambda x: "CORRECT" in str(x.get("action") or "").upper()),
        "errors": _cnt(lambda x: str(x.get("severity") or "").upper() == "ERROR"),
        "warnings": _cnt(lambda x: str(x.get("severity") or "").upper() == "WARNING"),
    }
    page = merged[offset : offset + limit]
    return {"summary": summary, "items": page}


# ============================================================
# ANALYTICS
# ============================================================

@router.get("/analytics/users")
async def admin_analytics_users(user: dict = Depends(get_current_user)):
    require_admin(user)

    async with core.state.db_pool.acquire() as conn:
        total_users = await conn.fetchval("SELECT COUNT(*) FROM users")
        active_users = await conn.fetchval("SELECT COUNT(*) FROM users WHERE status='active'")
        banned_users = await conn.fetchval("SELECT COUNT(*) FROM users WHERE status='banned'")
        paid_users = await conn.fetchval("SELECT COUNT(*) FROM users WHERE subscription_tier <> 'free'")
        new_users_30d = await conn.fetchval("SELECT COUNT(*) FROM users WHERE created_at >= NOW() - INTERVAL '30 days'")
        new_users_7d = await conn.fetchval("SELECT COUNT(*) FROM users WHERE created_at >= NOW() - INTERVAL '7 days'")
        new_users_24h = await conn.fetchval("SELECT COUNT(*) FROM users WHERE created_at >= NOW() - INTERVAL '24 hours'")

    return {
        "total_users": int(total_users or 0),
        "active_users": int(active_users or 0),
        "banned_users": int(banned_users or 0),
        "paid_users": int(paid_users or 0),
        "new_users_30d": int(new_users_30d or 0),
        "new_users_7d": int(new_users_7d or 0),
        "new_users_24h": int(new_users_24h or 0),
    }


@router.get("/analytics/revenue")
async def admin_analytics_revenue(user: dict = Depends(get_current_user)):
    require_admin(user)

    async with core.state.db_pool.acquire() as conn:
        launch_count = await conn.fetchval("SELECT COUNT(*) FROM users WHERE subscription_tier='launch'")
        creator_lite_count = await conn.fetchval("SELECT COUNT(*) FROM users WHERE subscription_tier='creator_lite'")
        creator_pro_count = await conn.fetchval("SELECT COUNT(*) FROM users WHERE subscription_tier='creator_pro'")
        studio_count = await conn.fetchval("SELECT COUNT(*) FROM users WHERE subscription_tier='studio'")
        agency_count = await conn.fetchval("SELECT COUNT(*) FROM users WHERE subscription_tier='agency'")

    return {
        "mrr_estimate": 0.0,
        "launch_count": int(launch_count or 0),
        "creator_lite_count": int(creator_lite_count or 0),
        "creator_pro_count": int(creator_pro_count or 0),
        "studio_count": int(studio_count or 0),
        "agency_count": int(agency_count or 0),
    }


@router.get("/analytics/overview")
async def admin_analytics_overview(
    range: Optional[str] = Query(None, description="Time window: 7d|30d|90d|6m|1y|Nd (custom)"),
    days: Optional[int] = Query(None, ge=1, le=3650, description="Compatibility: days=N"),
    user: dict = Depends(require_admin),
):
    """
    Admin analytics overview for the selected time window.

    Contract (frontend expects):
      - total_users
      - new_users
      - paid_users
      - mrr_estimate
      - range

    Supports:
      - ?range=7d|30d|90d|6m|1y
      - ?range=45d (custom, guarded 1-3650)
      - ?days=45 (compat)
    """
    # -------- range parsing (authoritative) --------
    preset_map = {"7d": 7, "30d": 30, "90d": 90, "6m": 180, "1y": 365}
    window_days = None

    if days is not None:
        window_days = int(days)

    if range:
        r = str(range).strip().lower()
        if r in preset_map:
            window_days = preset_map[r]
        else:
            m = re.match(r"^(\d{1,4})d$", r)
            if m:
                window_days = int(m.group(1))
            else:
                raise HTTPException(status_code=400, detail="Invalid range. Use 7d|30d|90d|6m|1y or Nd (e.g. 45d).")

    if window_days is None:
        window_days = 30

    if window_days < 1 or window_days > 3650:
        raise HTTPException(status_code=400, detail="Range out of bounds. Use 1-3650 days.")

    since = _now_utc() - timedelta(days=window_days)

    # -------- KPI aggregation --------
    # We intentionally avoid a dependency on a billing_events table (not guaranteed to exist).
    # Instead, derive paid_users + mrr_estimate from users.subscription_tier + users.subscription_status.
    paid_tiers = ["launch", "creator_pro", "studio", "agency"]

    async with core.state.db_pool.acquire() as conn:
        total_users = await conn.fetchval("SELECT COUNT(*) FROM users")

        new_users = await conn.fetchval(
            "SELECT COUNT(*) FROM users WHERE created_at >= $1",
            since,
        )

        paid_rows = await conn.fetch(
            """
            SELECT subscription_tier, COUNT(*)::bigint AS c
            FROM users
            WHERE subscription_status = 'active'
              AND subscription_tier = ANY($1::text[])
            GROUP BY subscription_tier
            """,
            paid_tiers,
        )

    counts = {row["subscription_tier"]: int(row["c"]) for row in paid_rows}
    paid_users = sum(counts.values())

    # MRR estimate based on PLAN_CONFIG prices
    mrr_estimate = 0.0
    for tier, c in counts.items():
        mrr_estimate += _tier_list_price_usd(tier) * c

    return {
        "total_users": int(total_users or 0),
        "new_users": int(new_users or 0),
        "paid_users": int(paid_users or 0),
        "mrr_estimate": float(mrr_estimate),
        "range": f"{window_days}d",
    }


# ============================================================
# KPI ENDPOINTS
# ============================================================

@router.get("/kpi/overview")
async def kpi_overview(range: str = "30d", user: dict = Depends(require_admin)):
    minutes = _range_to_minutes(range, default_minutes=30 * 24 * 60)
    since = _now_utc() - timedelta(minutes=minutes)

    async with core.state.db_pool.acquire() as conn:
        new_users = await conn.fetchval("SELECT COUNT(*) FROM users WHERE created_at >= $1", since)
        total_users = await conn.fetchval("SELECT COUNT(*) FROM users")
        paid_users = await conn.fetchval("SELECT COUNT(*) FROM users WHERE subscription_tier NOT IN ('free', 'master_admin', 'friends_family')")

        upload_stats = await conn.fetchrow("""
            SELECT COUNT(*)::int AS total, SUM(CASE WHEN status IN ('completed','succeeded') THEN 1 ELSE 0 END)::int AS completed,
            SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END)::int AS failed,
            COALESCE(SUM(views), 0)::bigint AS views, COALESCE(SUM(likes), 0)::bigint AS likes
            FROM uploads WHERE created_at >= $1
        """, since)

        revenue = await conn.fetchrow("SELECT COALESCE(SUM(amount), 0)::decimal AS total, COALESCE(SUM(CASE WHEN source = 'subscription' THEN amount ELSE 0 END), 0)::decimal AS subscriptions, COALESCE(SUM(CASE WHEN source = 'topup' THEN amount ELSE 0 END), 0)::decimal AS topups FROM revenue_tracking WHERE created_at >= $1", since)

        mrr_data = await conn.fetch("SELECT subscription_tier, COUNT(*) AS count FROM users WHERE subscription_tier NOT IN ('free', 'master_admin', 'friends_family', 'lifetime') AND subscription_status = 'active' GROUP BY subscription_tier")
        mrr = sum(_tier_list_price_usd(r["subscription_tier"]) * r["count"] for r in mrr_data)

        tiers = await conn.fetch("SELECT subscription_tier, COUNT(*)::int AS count FROM users GROUP BY subscription_tier")

    return {
        "users": {"new": new_users, "total": total_users, "paid": paid_users},
        "uploads": {"total": upload_stats["total"] if upload_stats else 0, "completed": upload_stats["completed"] if upload_stats else 0, "failed": upload_stats["failed"] if upload_stats else 0, "success_rate": ((upload_stats["completed"] or 0) / max(upload_stats["total"] or 1, 1)) * 100},
        "engagement": {"views": upload_stats["views"] if upload_stats else 0, "likes": upload_stats["likes"] if upload_stats else 0},
        "revenue": {"total": float(revenue["total"]) if revenue else 0, "subscriptions": float(revenue["subscriptions"]) if revenue else 0, "topups": float(revenue["topups"]) if revenue else 0, "mrr": mrr},
        "tiers": {_tier_json_key(t["subscription_tier"]): t["count"] for t in tiers},
    }


@router.get("/kpis")
async def get_admin_kpis(range: str = Query("30d"), user: dict = Depends(require_admin)):
    """Combined KPI endpoint that returns all metrics in one call"""
    minutes = _range_to_minutes(range, 43200)
    since = _now_utc() - timedelta(minutes=minutes)
    prev_since = since - timedelta(minutes=minutes)

    async with core.state.require_pool().acquire() as conn:
        # Users
        total_users = await conn.fetchval("SELECT COUNT(*) FROM users")
        new_users = await conn.fetchval("SELECT COUNT(*) FROM users WHERE created_at >= $1", since)
        prev_users = await conn.fetchval("SELECT COUNT(*) FROM users WHERE created_at >= $1 AND created_at < $2", prev_since, since)
        paid_users = await conn.fetchval("SELECT COUNT(*) FROM users WHERE subscription_tier NOT IN ('free', 'master_admin', 'friends_family', 'lifetime') AND subscription_status = 'active'")
        active_users = await conn.fetchval("SELECT COUNT(DISTINCT user_id) FROM uploads WHERE created_at >= $1", since)

        new_users_change = ((new_users - prev_users) / max(prev_users, 1)) * 100 if prev_users > 0 else 0

        # MRR
        mrr_data = await conn.fetch("""
            SELECT subscription_tier, COUNT(*) AS count FROM users
            WHERE subscription_tier NOT IN ('free', 'master_admin', 'friends_family', 'lifetime')
            AND subscription_status = 'active' GROUP BY subscription_tier
        """)
        total_mrr = sum(_tier_list_price_usd(r["subscription_tier"]) * r["count"] for r in mrr_data)
        mrr_by_tier = {
            _tier_json_key(r["subscription_tier"]): _tier_list_price_usd(r["subscription_tier"]) * r["count"]
            for r in mrr_data
        }

        # Tier breakdown
        tier_data = await conn.fetch(
            "SELECT COALESCE(subscription_tier, 'free') AS tier, COUNT(*)::int AS count "
            "FROM users GROUP BY COALESCE(subscription_tier, 'free')"
        )
        tier_breakdown = {str(t["tier"] or "free"): t["count"] for t in tier_data}

        # Revenue
        revenue = await conn.fetchrow(
            """
            SELECT COALESCE(SUM(amount), 0)::decimal AS total,
                COALESCE(SUM(CASE WHEN source = 'topup' THEN amount ELSE 0 END), 0)::decimal AS topups,
                COUNT(*) FILTER (WHERE source = 'topup')::int AS topup_cnt
            FROM revenue_tracking WHERE created_at >= $1
            """,
            since,
        )

        until = _now_utc()
        cost_summary = await build_admin_costs_summary(
            conn, since=since, until=until, range_key=range,
        )
        openai_cost = cost_summary["openai_cost"]
        openai_calls = cost_summary["openai_calls"]
        google_cloud_cost = cost_summary["google_cloud_cost"]
        pikzels_cost = cost_summary["pikzels_cost"]
        pikzels_calls = cost_summary["pikzels_calls"]
        twelvelabs_cost = cost_summary["twelvelabs_cost"]
        serpapi_cost = cost_summary["serpapi_cost"]
        other_provider_cost = cost_summary["other_provider_cost"]
        storage_cost = cost_summary["storage_cost"]
        storage_gb = cost_summary["storage_gb"]
        bandwidth_cost = cost_summary["bandwidth_cost"]
        bandwidth_tb = cost_summary["bandwidth_tb"]
        compute_cost = cost_summary["compute_cost"]
        stripe_fees = cost_summary["stripe_fees"]
        stripe_fee_txns = cost_summary["stripe_fee_txns"]
        mailgun_cost = cost_summary["mailgun_cost"]
        mailgun_emails_sent = cost_summary["mailgun_emails_sent"]
        postgres_cost = cost_summary["postgres_cost"]
        postgres_size_gb = cost_summary["postgres_size_gb"]
        redis_cost = cost_summary["redis_cost"]
        redis_memory_mb = cost_summary["redis_memory_mb"]
        total_costs = cost_summary["total_costs"]
        cost_providers = cost_summary.get("providers") or []
        cost_provenance = cost_summary.get("cost_provenance") or {}

        gross_margin = ((total_mrr - total_costs) / max(total_mrr, 1)) * 100 if total_mrr > 0 else 0

        # Uploads
        upload_stats = await conn.fetchrow("""
            SELECT COUNT(*)::int AS total, SUM(CASE WHEN status IN ('completed','succeeded') THEN 1 ELSE 0 END)::int AS completed,
            SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END)::int AS failed,
            COALESCE(SUM(views), 0)::bigint AS views, COALESCE(SUM(likes), 0)::bigint AS likes
            FROM uploads WHERE created_at >= $1
        """, since)
        total_uploads = upload_stats["total"] if upload_stats else 0
        successful_uploads = upload_stats["completed"] if upload_stats else 0
        success_rate = (successful_uploads / max(total_uploads, 1)) * 100

        prev_uploads = await conn.fetchval("SELECT COUNT(*) FROM uploads WHERE created_at >= $1 AND created_at < $2", prev_since, since)
        uploads_change = ((total_uploads - prev_uploads) / max(prev_uploads, 1)) * 100 if prev_uploads > 0 else 0
        cost_per_upload = cost_summary["cost_per_upload"]

        w_min = float(os.environ.get("KPI_WHISPER_ASSUMED_MINUTES_PER_UPLOAD", "0.5") or 0.5)
        w_usd = float(os.environ.get("KPI_WHISPER_USD_PER_MINUTE", "0.006") or 0.006)
        whisper_cost_estimate_usd = float(successful_uploads or 0) * w_min * w_usd

        # Platform distribution
        platform_data = await conn.fetch(_platform_upload_mix_sql(), since)
        platform_distribution = {
            str(p["platform"] or "unknown").lower(): p["uploads"] for p in platform_data if p["platform"]
        }

        queue_depth = await conn.fetchval("SELECT COUNT(*) FROM uploads WHERE status IN ('pending', 'queued', 'processing')")

        # Funnels
        funnel_connected = await conn.fetchval("SELECT COUNT(DISTINCT u.id) FROM users u JOIN platform_tokens pt ON u.id = pt.user_id WHERE u.created_at >= $1", since)
        funnel_uploaded = await conn.fetchval("SELECT COUNT(DISTINCT user_id) FROM uploads WHERE created_at >= $1", since)
        funnel_signup_connect = (funnel_connected / max(new_users, 1)) * 100
        funnel_connect_upload = (funnel_uploaded / max(funnel_connected, 1)) * 100

        cancellations = await conn.fetchval("SELECT COUNT(*) FROM users WHERE subscription_status = 'cancelled' AND updated_at >= $1", since)

        admin_eng = await compute_admin_engagement_totals(
            conn, window_start=since, window_end_exclusive=until,
        )
        reliability = await fetch_admin_reliability_metrics(conn, since, until)
        attach = await fetch_admin_attach_metrics(conn, since, until)

    return {
        "total_mrr": total_mrr, "mrr_change": 0, "mrr_by_tier": mrr_by_tier,
        "mrr_launch": mrr_by_tier.get("launch", 0), "mrr_creator_pro": mrr_by_tier.get("creator_pro", 0),
        "mrr_studio": mrr_by_tier.get("studio", 0), "mrr_agency": mrr_by_tier.get("agency", 0),
        "launch_users": int(tier_breakdown.get("creator_lite") or tier_breakdown.get("launch") or 0),
        "creator_lite_users": int(tier_breakdown.get("creator_lite") or tier_breakdown.get("launch") or 0),
        "creator_pro_users": tier_breakdown.get("creator_pro", 0),
        "studio_users": tier_breakdown.get("studio", 0), "agency_users": tier_breakdown.get("agency", 0),
        "topup_revenue": float(revenue["topups"]) if revenue else 0,
        "topup_count": int(revenue["topup_cnt"] or 0) if revenue else 0,
        "arpu": round(total_mrr / max(total_users, 1), 2), "arpa": round(total_mrr / max(paid_users, 1), 2),
        "refunds": 0, "refund_count": 0,
        "openai_cost": openai_cost, "openai_calls": openai_calls,
        "google_cloud_cost": google_cloud_cost, "pikzels_cost": pikzels_cost, "pikzels_calls": pikzels_calls,
        "twelvelabs_cost": twelvelabs_cost, "serpapi_cost": serpapi_cost, "other_provider_cost": other_provider_cost,
        "storage_cost": storage_cost, "storage_gb": storage_gb,
        "compute_cost": compute_cost, "bandwidth_cost": bandwidth_cost, "bandwidth_tb": bandwidth_tb,
        "stripe_fees": stripe_fees, "stripe_fee_txns": stripe_fee_txns,
        "mailgun_cost": mailgun_cost, "mailgun_emails_sent": mailgun_emails_sent,
        "postgres_cost": postgres_cost, "postgres_size_gb": postgres_size_gb,
        "redis_cost": redis_cost, "redis_memory_mb": redis_memory_mb,
        "providers": cost_providers, "cost_provenance": cost_provenance,
        "total_costs": total_costs, "cost_per_upload": round(cost_per_upload, 4),
        "gross_margin": round(gross_margin, 1), "gross_margin_change": 0,
        "funnel_signups": new_users, "funnel_connected": funnel_connected, "funnel_uploaded": funnel_uploaded,
        "funnel_signup_connect": round(funnel_signup_connect, 1), "funnel_connect_upload": round(funnel_connect_upload, 1),
        "free_to_paid_rate": round((paid_users / max(total_users, 1)) * 100, 1), "free_to_paid_change": 0,
        "cancellations": cancellations, "cancellation_rate": round((cancellations / max(paid_users, 1)) * 100, 1),
        "failed_payments": 0, "payment_failure_rate": 0,
        "total_uploads": total_uploads, "successful_uploads": successful_uploads, "success_rate": round(success_rate, 1),
        "transcode_fail_rate": reliability.get("transcode_fail_rate", 0),
        "platform_fail_rate": reliability.get("platform_fail_rate", 0),
        "retry_rate": reliability.get("retry_rate", 0),
        "avg_process_time": reliability.get("avg_process_time", 0),
        "avg_transcode_time": reliability.get("avg_transcode_time") or 0,
        "cancel_rate": reliability.get("cancel_rate", 0),
        "queue_depth": reliability.get("queue_depth", queue_depth or 0),
        "ai_attach_rate": attach.get("ai_attach_rate", 0),
        "topup_attach_rate": attach.get("topup_attach_rate", 0),
        "flex_adoption_rate": attach.get("flex_adoption_rate", 0),
        "new_users": new_users, "new_users_change": round(new_users_change, 1), "uploads_change": round(uploads_change, 1),
        "active_users": active_users or 0,
        "total_views": admin_eng.get("views", 0),
        "total_likes": admin_eng.get("likes", 0),
        "total_comments": admin_eng.get("comments", 0),
        "total_shares": admin_eng.get("shares", 0),
        "total_views_raw": admin_eng.get("raw_upload_sum_views", 0),
        "total_likes_raw": admin_eng.get("raw_upload_sum_likes", 0),
        "engagement_crosswalk": metric_definitions_svc.engagement_crosswalk(),
        "metric_definitions": metric_definitions_svc.admin_kpi_data_provenance(rollup_version=ROLLUP_VERSION),
        "engagement_provenance": admin_eng.get("kpi_sources"),
        "avg_uploads_per_user": round(total_uploads / max(active_users or 1, 1), 1),
        "tier_breakdown": tier_breakdown, "platform_distribution": platform_distribution,
        "whisper_cost_estimate_usd": round(whisper_cost_estimate_usd, 4),
        "whisper_estimate": {
            "effective_minutes_per_upload": w_min,
            "usd_per_minute": w_usd,
        },
    }


@router.get("/kpi/margins")
async def kpi_margins(range: str = "30d", user: dict = Depends(require_admin)):
    minutes = {"7d": 10080, "30d": 43200, "6m": 262800}.get(range, 43200)
    since = _now_utc() - timedelta(minutes=minutes)

    async with core.state.db_pool.acquire() as conn:
        costs = await conn.fetchrow("""
            SELECT
                COALESCE(SUM(CASE WHEN category = 'openai' THEN cost_usd ELSE 0 END), 0)::decimal AS openai,
                COALESCE(SUM(CASE WHEN category = 'storage' THEN cost_usd ELSE 0 END), 0)::decimal AS storage,
                COALESCE(SUM(CASE WHEN category = 'compute' THEN cost_usd ELSE 0 END), 0)::decimal AS compute,
                COALESCE(SUM(CASE WHEN category IN ('stripe_fees','mailgun','bandwidth','postgres','redis') THEN cost_usd ELSE 0 END), 0)::decimal AS other
            FROM cost_tracking WHERE created_at >= $1
        """, since)
        revenue = await conn.fetchval("SELECT COALESCE(SUM(amount), 0) FROM revenue_tracking WHERE created_at >= $1", since)

        tier_data = await conn.fetch("""
            SELECT u.subscription_tier, COUNT(up.id)::int AS uploads, 0::decimal AS cost
            FROM users u LEFT JOIN uploads up ON up.user_id = u.id AND up.created_at >= $1
            GROUP BY u.subscription_tier
        """, since)

        platform_data = await conn.fetch(
            """
            SELECT p AS platform, COUNT(*)::int AS uploads, 0::decimal AS cost
            FROM uploads u,
                 LATERAL unnest(COALESCE(u.platforms, ARRAY[]::text[])) AS u_p(p)
            WHERE u.created_at >= $1
            GROUP BY p
            """,
            since,
        )

    total_cost = float(costs["openai"] or 0) + float(costs["storage"] or 0) + float(costs["compute"] or 0) + float(costs.get("other") or 0)
    gross_margin = float(revenue or 0) - total_cost

    return {
        "costs": {"openai": float(costs["openai"] or 0), "storage": float(costs["storage"] or 0), "compute": float(costs["compute"] or 0), "other": float(costs.get("other") or 0), "total": total_cost},
        "revenue": float(revenue or 0),
        "gross_margin": gross_margin,
        "margin_pct": (gross_margin / max(float(revenue or 1), 1)) * 100,
        "by_tier": {
            _tier_json_key(t["subscription_tier"]): {"uploads": t["uploads"], "cost": float(t["cost"])}
            for t in tier_data
        },
        "by_platform": {
            str(p["platform"] or "unknown").lower(): {"uploads": p["uploads"], "cost": float(p["cost"])}
            for p in platform_data
            if p["platform"]
        },
    }

@router.get("/kpi/burn")
async def kpi_burn(range: str = "30d", user: dict = Depends(require_admin)):
    minutes = {"7d": 10080, "30d": 43200}.get(range, 43200)
    since = _now_utc() - timedelta(minutes=minutes)

    async with core.state.db_pool.acquire() as conn:
        token_stats = await conn.fetchrow("""
            SELECT COALESCE(SUM(CASE WHEN token_type = 'put' AND delta < 0 THEN ABS(delta) ELSE 0 END), 0)::int AS put_spent,
            COALESCE(SUM(CASE WHEN token_type = 'aic' AND delta < 0 THEN ABS(delta) ELSE 0 END), 0)::int AS aic_spent,
            COALESCE(SUM(CASE WHEN reason = 'topup' THEN delta ELSE 0 END), 0)::int AS tokens_purchased
            FROM token_ledger WHERE created_at >= $1
        """, since)

        quota_data = await conn.fetch("""
            SELECT u.id, u.subscription_tier, w.put_balance, w.put_reserved
            FROM users u JOIN wallets w ON w.user_id = u.id WHERE u.status = 'active'
        """)

        hitting_quota = sum(1 for u in quota_data if (u["put_balance"] - u["put_reserved"]) <= get_plan(u["subscription_tier"]).get("put_daily", 1))
        total_active = len(quota_data)

    return {
        "put_spent": token_stats["put_spent"] if token_stats else 0,
        "aic_spent": token_stats["aic_spent"] if token_stats else 0,
        "tokens_purchased": token_stats["tokens_purchased"] if token_stats else 0,
        "users_hitting_quota": hitting_quota,
        "total_active_users": total_active,
        "quota_hit_pct": (hitting_quota / max(total_active, 1)) * 100,
    }

@router.get("/kpi/funnels")
async def kpi_funnels(user: dict = Depends(require_admin)):
    async with core.state.db_pool.acquire() as conn:
        signups_24h = await conn.fetchval("SELECT COUNT(*) FROM users WHERE created_at >= NOW() - INTERVAL '24 hours'")
        first_uploads_24h = await conn.fetchval("SELECT COUNT(DISTINCT user_id) FROM uploads WHERE user_id IN (SELECT id FROM users WHERE created_at >= NOW() - INTERVAL '24 hours')")

        total_free = await conn.fetchval("SELECT COUNT(*) FROM users WHERE subscription_tier = 'free'")
        converted = await conn.fetchval("SELECT COUNT(*) FROM users WHERE subscription_tier NOT IN ('free', 'master_admin', 'friends_family') AND subscription_status = 'active'")

        ai_users = await conn.fetchval("SELECT COUNT(DISTINCT user_id) FROM uploads WHERE aic_spent > 0")
        total_uploaders = await conn.fetchval("SELECT COUNT(DISTINCT user_id) FROM uploads")

        topup_users = await conn.fetchval("SELECT COUNT(DISTINCT user_id) FROM token_ledger WHERE reason = 'topup'")
        flex_users = await conn.fetchval("SELECT COUNT(*) FROM users WHERE flex_enabled = TRUE")
        churned = await conn.fetchval("SELECT COUNT(*) FROM users WHERE subscription_status = 'cancelled' AND updated_at >= NOW() - INTERVAL '30 days'")

    return {
        "signup_to_upload_24h": {"signups": signups_24h, "uploads": first_uploads_24h, "rate": (first_uploads_24h / max(signups_24h, 1)) * 100},
        "free_to_paid": {"free": total_free, "paid": converted, "rate": (converted / max(total_free + converted, 1)) * 100},
        "ai_usage": {"users": ai_users, "total": total_uploaders, "rate": (ai_users / max(total_uploaders, 1)) * 100},
        "topup_adoption": {"users": topup_users, "total": total_uploaders, "rate": (topup_users / max(total_uploaders, 1)) * 100},
        "flex_adoption": {"users": flex_users},
        "churn_30d": churned,
    }


@router.get("/kpi/revenue")
async def get_kpi_revenue(range: str = Query("30d"), user: dict = Depends(require_admin)):
    minutes = _range_to_minutes(range, 43200)
    since = _now_utc() - timedelta(minutes=minutes)
    until = _now_utc()
    async with core.state.require_pool().acquire() as conn:
        total_users = await conn.fetchval("SELECT COUNT(*) FROM users")
        paid_users = await conn.fetchval("SELECT COUNT(*) FROM users WHERE subscription_tier NOT IN ('free', 'master_admin', 'friends_family', 'lifetime') AND subscription_status = 'active'") or 1
        mrr_data = await conn.fetch("SELECT subscription_tier, COUNT(*) AS count FROM users WHERE subscription_tier NOT IN ('free', 'master_admin', 'friends_family', 'lifetime') AND subscription_status = 'active' GROUP BY subscription_tier")
        total_mrr = sum(_tier_list_price_usd(r["subscription_tier"]) * r["count"] for r in mrr_data)
        topup = await conn.fetchval("SELECT COALESCE(SUM(amount), 0) FROM revenue_tracking WHERE source = 'topup' AND created_at >= $1", since)
        topup_n = await conn.fetchval(
            "SELECT COUNT(*)::int FROM revenue_tracking WHERE source = 'topup' AND created_at >= $1",
            since,
        )
    refunds_total, refunds_count = await fetch_stripe_refunds_window(since, until)
    return {
        "total_mrr": total_mrr,
        "mrr_change": 0,
        "mrr_by_tier": {},
        "topup_total": float(topup or 0),
        "topup_count": int(topup_n or 0),
        "arpu": round(total_mrr / max(total_users, 1), 2),
        "arpa": round(total_mrr / max(paid_users, 1), 2),
        "ltv": round((total_mrr / max(paid_users, 1)) * 12, 2),
        "refunds_total": refunds_total,
        "refunds_count": refunds_count,
        "refunds_change": 0,
    }


@router.get("/kpi/costs")
async def get_kpi_costs(range: str = Query("30d"), user: dict = Depends(require_admin)):
    minutes = _range_to_minutes(range, 43200)
    since = _now_utc() - timedelta(minutes=minutes)
    until = _now_utc()
    async with core.state.db_pool.acquire() as conn:
        summary = await build_admin_costs_summary(
            conn, since=since, until=until, range_key=range,
        )
    return {**summary, "costs_change": 0}


@router.get("/kpi/growth")
async def get_kpi_growth(range: str = Query("30d"), user: dict = Depends(require_admin)):
    minutes = _range_to_minutes(range, 43200)
    since = _now_utc() - timedelta(minutes=minutes)
    until = _now_utc()
    async with core.state.db_pool.acquire() as conn:
        return await fetch_admin_growth_metrics(conn, since, until)


@router.get("/kpi/reliability")
async def get_kpi_reliability(range: str = Query("30d"), user: dict = Depends(require_admin)):
    minutes = _range_to_minutes(range, 43200)
    since = _now_utc() - timedelta(minutes=minutes)
    until = _now_utc()
    async with core.state.require_pool().acquire() as conn:
        return await fetch_admin_reliability_metrics(conn, since, until)


@router.post("/kpi/refresh")
async def trigger_kpi_refresh(background_tasks: BackgroundTasks, user: dict = Depends(require_admin)):
    """
    Manually trigger KPI data collection from Stripe, OpenAI, Mailgun, etc.
    Runs in background; results appear in cost_tracking within ~30s.
    """
    from stages.kpi_collector import run_kpi_collect
    async def _run():
        try:
            summary = await run_kpi_collect(core.state.db_pool)
            logger.info(f"KPI refresh complete: {summary}")
        except Exception as e:
            logger.warning(f"KPI refresh failed: {e}")
    background_tasks.add_task(_run)
    return {"status": "started", "message": "KPI collection running in background"}


@router.get("/kpi/recognition")
async def get_kpi_recognition(
    range: str = Query("30d"),
    limit: int = Query(20, ge=1, le=100),
    user: dict = Depends(require_admin),
):
    """Recognition KPIs powered by upload_recognition_summary + video_recognition.

    Returns:
      - ``hydration_avg`` / ``hydration_p50`` / ``hydration_p90`` —
        deterministic-evidence quality of recent uploads (0..1).
      - ``coverage_pct`` — fraction of recent uploads that have any
        recognition summary at all (good for ops health checks).
      - ``top_objects`` / ``top_logos`` / ``top_text`` — most-detected items
        across the platform in the window (drives the "what content lives
        here" widget on the admin dashboard).
      - ``with_people_pct`` — share of clips with at least one person segment.
    """
    minutes = _range_to_minutes(range, 43200)
    since = _now_utc() - timedelta(minutes=minutes)

    async with core.state.require_pool().acquire() as conn:
        # Health: how many uploads in window vs how many got a summary
        upload_total = await conn.fetchval(
            "SELECT COUNT(*)::int FROM uploads WHERE created_at >= $1",
            since,
        )
        summary_total = await conn.fetchval(
            """
            SELECT COUNT(*)::int
              FROM upload_recognition_summary rs
              JOIN uploads u ON u.id = rs.upload_id
             WHERE u.created_at >= $1
            """,
            since,
        )

        stats = await conn.fetchrow(
            """
            SELECT
                COALESCE(AVG(rs.hydration_score), 0)::double precision  AS hydration_avg,
                COALESCE(percentile_cont(0.5)  WITHIN GROUP (ORDER BY rs.hydration_score), 0)::double precision AS hydration_p50,
                COALESCE(percentile_cont(0.9)  WITHIN GROUP (ORDER BY rs.hydration_score), 0)::double precision AS hydration_p90,
                COALESCE(SUM(CASE WHEN rs.has_people THEN 1 ELSE 0 END), 0)::int AS with_people,
                COALESCE(SUM(rs.object_track_count), 0)::bigint   AS object_total,
                COALESCE(SUM(rs.logo_count), 0)::bigint           AS logo_total,
                COALESCE(SUM(rs.text_detection_count), 0)::bigint AS text_total,
                COALESCE(AVG(rs.coverage_seconds), 0)::double precision AS avg_coverage_seconds
              FROM upload_recognition_summary rs
              JOIN uploads u ON u.id = rs.upload_id
             WHERE u.created_at >= $1
            """,
            since,
        )

        # Top descriptions across all per-detection rows in the window.
        top_objects = await conn.fetch(
            """
            SELECT lower(description) AS d, COUNT(*)::bigint AS n
              FROM video_recognition vr
              JOIN uploads u ON u.id = vr.upload_id
             WHERE vr.kind = 'object'
               AND u.created_at >= $1
               AND length(description) > 0
             GROUP BY lower(description)
             ORDER BY n DESC
             LIMIT $2
            """,
            since,
            limit,
        )
        top_logos = await conn.fetch(
            """
            SELECT lower(description) AS d, COUNT(*)::bigint AS n
              FROM video_recognition vr
              JOIN uploads u ON u.id = vr.upload_id
             WHERE vr.kind = 'logo'
               AND u.created_at >= $1
               AND length(description) > 0
             GROUP BY lower(description)
             ORDER BY n DESC
             LIMIT $2
            """,
            since,
            limit,
        )
        top_text = await conn.fetch(
            """
            SELECT lower(description) AS d, COUNT(*)::bigint AS n
              FROM video_recognition vr
              JOIN uploads u ON u.id = vr.upload_id
             WHERE vr.kind = 'text'
               AND u.created_at >= $1
               AND length(description) > 0
             GROUP BY lower(description)
             ORDER BY n DESC
             LIMIT $2
            """,
            since,
            limit,
        )

    entity_catalog_total = 0
    entity_buckets: Dict[str, Any] = {}
    try:
        from services.visual_entity_memory import fetch_platform_bucket_kpis
        from core.visual_entity_taxonomy import narrative_bucket_labels

        async with core.state.require_pool().acquire() as conn_ec:
            entity_catalog_total = await conn_ec.fetchval(
                "SELECT COUNT(*)::int FROM user_visual_entity_catalog"
            )
        bucket_kpis = await fetch_platform_bucket_kpis(
            core.state.db_pool, since=since, limit=min(limit, 12)
        )
        labels = narrative_bucket_labels()
        for key, items in (bucket_kpis.get("buckets") or {}).items():
            if items:
                entity_buckets[key] = {
                    "label": labels.get(key, key),
                    "top": items,
                }
    except Exception as e:
        logger.debug("kpi recognition entity buckets: %s", e)

    coverage_pct = (
        (summary_total or 0) / max(upload_total or 1, 1) * 100.0
    ) if upload_total else 0.0
    with_people_pct = (
        (stats["with_people"] or 0) / max(summary_total or 1, 1) * 100.0
    ) if summary_total else 0.0

    return {
        "range": range,
        "uploads_in_window": int(upload_total or 0),
        "uploads_with_recognition": int(summary_total or 0),
        "coverage_pct": round(coverage_pct, 2),
        "hydration_avg": round(float(stats["hydration_avg"] or 0), 4),
        "hydration_p50": round(float(stats["hydration_p50"] or 0), 4),
        "hydration_p90": round(float(stats["hydration_p90"] or 0), 4),
        "with_people_pct": round(with_people_pct, 2),
        "object_total": int(stats["object_total"] or 0),
        "logo_total": int(stats["logo_total"] or 0),
        "text_total": int(stats["text_total"] or 0),
        "avg_coverage_seconds": round(float(stats["avg_coverage_seconds"] or 0), 2),
        "top_objects": [{"label": r["d"], "count": int(r["n"])} for r in top_objects],
        "top_logos": [{"label": r["d"], "count": int(r["n"])} for r in top_logos],
        "top_text": [{"label": r["d"], "count": int(r["n"])} for r in top_text],
        "entity_catalog_total": int(entity_catalog_total or 0),
        "entity_buckets": entity_buckets,
    }


@router.post("/ml/sync-visual-entities")
async def admin_sync_visual_entities_hub(
    user: dict = Depends(require_admin),
    limit_users: int = Query(200, ge=1, le=2000),
):
    """
    Push aggregated visual entity catalogs to the configured Hugging Face dataset
    (``UM8_HF_DATASET_REPO``, config ``visual_entities``). Requires ``HF_TOKEN``.
    """
    from services.ml_entity_hub_sync import build_platform_visual_entity_export, push_visual_entities_to_hub

    rows = await build_platform_visual_entity_export(
        core.state.db_pool, limit_per_user=limit_users
    )
    result = push_visual_entities_to_hub(rows)
    if not result.get("ok"):
        raise HTTPException(
            status_code=400 if result.get("skipped") else 502,
            detail=result.get("error") or result.get("reason") or "sync failed",
        )
    return result


@router.get("/kpi/usage")
async def get_kpi_usage(range: str = Query("30d"), user: dict = Depends(require_admin)):
    minutes = _range_to_minutes(range, 43200)
    since = _now_utc() - timedelta(minutes=minutes)
    until = _now_utc()
    async with core.state.db_pool.acquire() as conn:
        active = await conn.fetchval("SELECT COUNT(DISTINCT user_id) FROM uploads WHERE created_at >= $1", since)
        uploads = await conn.fetchval("SELECT COUNT(*) FROM uploads WHERE created_at >= $1", since)
        new_users = await conn.fetchval("SELECT COUNT(*) FROM users WHERE created_at >= $1", since)
        prev_since = since - timedelta(minutes=minutes)
        prev_users = await conn.fetchval("SELECT COUNT(*) FROM users WHERE created_at >= $1 AND created_at < $2", prev_since, since)
        admin_eng = await compute_admin_engagement_totals(
            conn, window_start=since, window_end_exclusive=until,
        )
    chg = ((new_users - prev_users) / max(prev_users, 1)) * 100 if prev_users > 0 else 0
    return {"active_users": active or 0, "active_users_change": 0, "total_uploads": uploads or 0, "uploads_change": 0,
            "new_users": new_users or 0, "new_users_change": round(chg, 1),
            "total_views": admin_eng.get("views", 0), "total_likes": admin_eng.get("likes", 0),
            "total_comments": admin_eng.get("comments", 0), "total_shares": admin_eng.get("shares", 0),
            "total_views_raw": admin_eng.get("raw_upload_sum_views", 0),
            "avg_uploads_per_user": round((uploads or 0) / max(active or 1, 1), 1)}


# ============================================================
# CHARTS
# ============================================================

@router.get("/chart/revenue")
async def get_chart_revenue(period: str = Query("30d"), user: dict = Depends(require_admin)):
    days = int(period.replace("d", "")) if period.endswith("d") and period[:-1].isdigit() else 30
    since = _now_utc() - timedelta(days=days)
    async with core.state.db_pool.acquire() as conn:
        rows = await conn.fetch("SELECT DATE(created_at) as date, COALESCE(SUM(amount), 0)::decimal as revenue FROM revenue_tracking WHERE created_at >= $1 GROUP BY DATE(created_at) ORDER BY date", since)
    data = {r["date"]: float(r["revenue"]) for r in rows}
    labels, values, current, end = [], [], since.date(), _now_utc().date()
    while current <= end:
        labels.append(current.strftime("%b %d"))
        values.append(data.get(current, 0))
        current += timedelta(days=1)
    return {"labels": labels, "values": values}


@router.get("/chart/users")
async def get_chart_users(period: str = Query("30d"), user: dict = Depends(require_admin)):
    days = int(period.replace("d", "")) if period.endswith("d") and period[:-1].isdigit() else 30
    since = _now_utc() - timedelta(days=days)
    async with core.state.require_pool().acquire() as conn:
        rows = await conn.fetch("SELECT DATE(created_at) as date, COUNT(*)::int as users FROM users WHERE created_at >= $1 GROUP BY DATE(created_at) ORDER BY date", since)
    data = {r["date"]: r["users"] for r in rows}
    labels, values, current, end = [], [], since.date(), _now_utc().date()
    while current <= end:
        labels.append(current.strftime("%b %d"))
        values.append(data.get(current, 0))
        current += timedelta(days=1)
    return {"labels": labels, "values": values}


# ============================================================
# BILLING — pipeline AIC weights (master admin)
# ============================================================


@router.get("/billing/service-weights")
async def admin_get_billing_service_weights(user: dict = Depends(require_master_admin)):
    """Live per-service AIC weights from ``billing_service_weights`` merged with code defaults."""
    if core.state.db_pool is None:
        raise HTTPException(503, "Database unavailable")
    from stages.ai_service_costs import SERVICE_WEIGHTS, merge_service_weights_from_db, service_catalog
    from services.billing_service_weights import fetch_service_weights_map
    from services.billing_catalog import (
        PUT_COST_DEFAULTS,
        effective_put_cost_rules,
        merge_put_cost_overrides,
    )

    async with core.state.require_pool().acquire() as conn:
        raw = await fetch_service_weights_map(conn)
    merged = merge_service_weights_from_db(raw)
    put_db = dict(core.state.billing_catalog_cache.get("put_cost_overrides") or {})
    put_effective = effective_put_cost_rules()
    return {
        "code_defaults": dict(SERVICE_WEIGHTS),
        "db_weights": raw,
        "effective": merged,
        "catalog": service_catalog(merged),
        "put_cost_defaults": dict(PUT_COST_DEFAULTS),
        "put_cost_db_overrides": put_db,
        "put_cost_effective": put_effective,
    }


@router.put("/billing/service-weights")
async def admin_put_billing_service_weights(
    body: BillingWeightsPut,
    user: dict = Depends(require_master_admin),
):
    """Upsert AIC weights (only ``SERVICE_WEIGHTS`` keys) and optional PUT debit constants."""
    if core.state.db_pool is None:
        raise HTTPException(503, "Database unavailable")
    if not body.weights and body.put_cost_rules is None:
        raise HTTPException(400, "Provide weights and/or put_cost_rules")
    from services.billing_service_weights import upsert_service_weights
    from services.billing_catalog import save_put_cost_overrides, validate_put_cost_overrides

    n = 0
    put_n = 0
    async with core.state.require_pool().acquire() as conn:
        if body.weights:
            n = await upsert_service_weights(conn, body.weights, updated_by=str(user["id"]))
        if body.put_cost_rules is not None:
            validated = validate_put_cost_overrides(body.put_cost_rules)
            await save_put_cost_overrides(conn, validated, str(user["id"]))
            put_n = len(validated)
    return {"status": "updated", "rows_written": n, "put_cost_rows_written": put_n}


@router.get("/billing/catalog")
async def admin_get_billing_catalog(user: dict = Depends(require_master_admin)):
    """DB overrides for public tiers + top-ups; merged values for UI."""
    if core.state.db_pool is None:
        raise HTTPException(503, "Database unavailable")
    from services.billing_catalog import (
        effective_tier_services_matrix,
        fetch_catalog_row,
        tier_config_before_billing_overrides,
        topup_products_before_billing_overrides,
    )
    from services.stripe_lookup_keys import subscription_keys_from_catalog
    from stages.ai_service_costs import service_catalog

    async with core.state.require_pool().acquire() as conn:
        row = await fetch_catalog_row(conn)
        subscription_stripe_keys = await subscription_keys_from_catalog(conn)
    tier_cfg = get_effective_tier_config()
    public = {s: tier_cfg.get(s, {}) for s in ("free", "creator_lite", "creator_pro", "studio", "agency")}
    tier_base = tier_config_before_billing_overrides()
    baseline_public_tiers = {s: tier_base.get(s, {}) for s in ("free", "creator_lite", "creator_pro", "studio", "agency")}
    topups = []
    top_base = topup_products_before_billing_overrides()
    baseline_topups = {}
    for lk, meta in get_effective_topup_products().items():
        wallet = meta.get("wallet") or ""
        kind = "bundle" if wallet == "bundle" else wallet
        row_d = {
            "lookup_key": lk,
            "kind": kind,
            "wallet": meta.get("wallet"),
            "amount": meta.get("amount"),
            "put": meta.get("put"),
            "aic": meta.get("aic"),
            "price_usd": meta.get("price_usd") or meta.get("price"),
        }
        topups.append(row_d)
        bmeta = top_base.get(lk) or {}
        baseline_topups[lk] = {
            "lookup_key": lk,
            "wallet": bmeta.get("wallet"),
            "amount": bmeta.get("amount"),
            "put": bmeta.get("put"),
            "aic": bmeta.get("aic"),
            "price_usd": bmeta.get("price_usd") or bmeta.get("price"),
        }
    svc_matrix = effective_tier_services_matrix()
    return {
        "tier_overrides": dict(core.state.billing_catalog_cache.get("tier_overrides") or {}),
        "topup_overrides": dict(core.state.billing_catalog_cache.get("topup_overrides") or {}),
        "tier_service_overrides": dict(core.state.billing_catalog_cache.get("tier_service_overrides") or {}),
        "effective_tier_services": svc_matrix,
        "service_catalog": service_catalog(),
        "effective_public_tiers": public,
        "baseline_public_tiers": baseline_public_tiers,
        "topups": topups,
        "baseline_topups": baseline_topups,
        "subscription_stripe_keys": subscription_stripe_keys,
        "catalog_loaded_at": core.state.catalog_pricing_cache.get("loaded_at"),
        "last_sync_at": core.state.billing_catalog_cache.get("last_sync_at"),
        "last_sync_ok": core.state.billing_catalog_cache.get("last_sync_ok"),
        "last_sync_error": core.state.billing_catalog_cache.get("last_sync_error"),
        "last_sync_detail": core.state.billing_catalog_cache.get("last_sync_detail"),
        "updated_at": row["updated_at"].isoformat() if row and row.get("updated_at") else None,
        "merge_order": "TIER_CONFIG → catalog_products → billing_catalog tier_overrides",
    }


@router.get("/billing/effective-tiers")
async def admin_get_effective_tiers(user: dict = Depends(require_master_admin)):
    """
    Live merged public tier matrix for admin billing UIs.
    Same values as ``GET /api/pricing`` and runtime entitlement checks (DB-backed).
    """
    by_slug = {row["slug"]: row for row in get_tiers_for_api()}
    tiers = [by_slug[s] for s in PUBLIC_TIER_SLUGS if s in by_slug]
    return {
        "tiers": tiers,
        "tier_slugs": list(PUBLIC_TIER_SLUGS),
        "catalog_loaded_at": core.state.catalog_pricing_cache.get("loaded_at"),
        "billing_catalog_updated_at": core.state.billing_catalog_cache.get("updated_at"),
        "merge_order": "TIER_CONFIG → catalog_products → billing_catalog tier_overrides",
    }


@router.put("/billing/catalog")
async def admin_put_billing_catalog(
    body: BillingCatalogPut,
    request: Request,
    background_tasks: BackgroundTasks,
    user: dict = Depends(require_master_admin),
):
    """Replace tier/topup override JSON; triggers async PNG regen + Stripe sync."""
    if core.state.db_pool is None:
        raise HTTPException(503, "Database unavailable")
    from services.billing_catalog import save_catalog_overrides
    from services.catalog_publish import refresh_pricing_caches_after_catalog_sync

    async with core.state.require_pool().acquire() as conn:
        await save_catalog_overrides(
            conn,
            body.tier_overrides,
            body.topup_overrides,
            str(user["id"]),
            body.tier_service_overrides,
        )
        await refresh_pricing_caches_after_catalog_sync(conn)
        await log_admin_audit(
            conn,
            user_id=str(user["id"]),
            admin=user,
            action="ADMIN_BILLING_CATALOG_SAVE",
            details={
                "tier_slugs": list(body.tier_overrides.keys()),
                "topup_keys": list(body.topup_overrides.keys()),
                "tier_service_slugs": list(body.tier_service_overrides.keys()),
            },
            request=request,
            resource_type="billing_catalog",
            resource_id="1",
            severity="INFO",
        )

    background_tasks.add_task(run_billing_catalog_sync_job)
    return {"status": "saved", "background_sync": "started"}


@router.get("/settings")
async def get_admin_settings(user: dict = Depends(require_master_admin)):
    return admin_settings_cache


@router.put("/settings")
async def update_admin_settings(settings: dict, user: dict = Depends(require_master_admin)):
    if "watermark_burn_text" in settings:
        from stages.db import sanitize_watermark_burn_text

        settings = dict(settings)
        settings["watermark_burn_text"] = sanitize_watermark_burn_text(
            settings.get("watermark_burn_text")
        )
    core.state.admin_settings_cache.update(settings)
    async with core.state.db_pool.acquire() as conn:
        await conn.execute("UPDATE admin_settings SET settings_json = $1, updated_at = NOW() WHERE id = 1", json.dumps(core.state.admin_settings_cache))
    return {"status": "updated", "settings": core.state.admin_settings_cache}


@router.get("/calculator/pricing")
async def get_admin_calculator_pricing(user: dict = Depends(require_master_admin)):
    """
    Live pricing and entitlements for the admin Business Calculator.
    Merged from catalog_products + billing_catalog (same as /api/pricing).
    """
    revenue_tiers = {}
    tier_cfg = get_effective_tier_config()
    for slug in PUBLIC_TIER_SLUGS:
        cfg = tier_cfg.get(slug, {})
        row = tier_cfg_to_api_dict(slug, cfg)
        revenue_tiers[slug] = {
            "name": row["name"],
            "price": row["price"],
            "price_annual": row.get("price_annual"),
            "put_daily": row.get("put_daily", 0),
            "put_monthly": row.get("put_monthly", 0),
            "aic_monthly": row.get("aic_monthly", 0),
            "queue_depth": row.get("queue_depth", 0),
            "lookahead_hours": row.get("lookahead_hours", 0),
            "max_accounts": row.get("max_accounts", 0),
            "max_accounts_per_platform": row.get("max_accounts_per_platform", 0),
            "trial_days": row.get("trial_days", 0),
            "team_seats": row.get("team_seats", 1),
            "priority_class": row.get("priority_class"),
            "ai_depth": row.get("ai_depth"),
            "analytics": row.get("analytics"),
        }
    # Sales-led floor for admin calculator only — not a billable Stripe tier in this repo.
    revenue_tiers["enterprise"] = {
        "name": "Enterprise (quote)",
        "price": 399.0,
        "put_monthly": 50000,
        "aic_monthly": 15000,
        "queue_depth": 999999,
        "lookahead_hours": 168,
        "max_accounts_per_platform": 100,
    }

    # Internal tiers (Friends & Family, Lifetime, Master Admin) -- $0 revenue, full infra cost
    internal_tiers = {}
    for slug in ("friends_family", "lifetime", "master_admin"):
        cfg = tier_cfg.get(slug, {})
        internal_tiers[slug] = {
            "name": cfg.get("name", slug.replace("_", " ").title()),
            "price": 0,
            "put_monthly": cfg.get("put_monthly", 0),
            "aic_monthly": cfg.get("aic_monthly", 0),
        }

    # Top-up packs (amounts from entitlements; prices come from Stripe)
    topup_packs = []
    for lookup_key, meta in get_effective_topup_products().items():
        wallet = meta.get("wallet", "")
        amount = meta.get("amount", 0)
        _price = meta.get("price_usd") or meta.get("price")
        if str(wallet) == "bundle":
            label = f"{int(meta.get('put') or 0)} PUT + {int(meta.get('aic') or 0)} AIC bundle"
        else:
            label = f"{wallet.upper()} {amount} Pack"
        topup_packs.append({
            "lookup_key": lookup_key,
            "wallet": wallet,
            "amount": amount,
            "put": int(meta.get("put") or 0),
            "aic": int(meta.get("aic") or 0),
            "price_usd": _price,   # canonical field (new frontend reads this)
            "price": _price,       # legacy alias  (older clients / backcompat)
            "label": label,
        })

    return {
        "revenue_tiers": revenue_tiers,
        "internal_tiers": internal_tiers,
        "topup_packs": topup_packs,
        "catalog_loaded_at": core.state.catalog_pricing_cache.get("loaded_at"),
        "merge_order": "TIER_CONFIG → catalog_products → billing_catalog tier_overrides",
    }


@router.get("/workers/heartbeat")
async def admin_workers_heartbeat(user: dict = Depends(require_master_admin)):
    """
    Liveness rows written by worker heartbeat (see worker.run_heartbeat_loop).
    Table is auto-created by the worker on first run.
    """
    if core.state.db_pool is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    try:
        async with core.state.db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    worker_id,
                    last_seen_at,
                    started_at,
                    worker_concurrency,
                    publish_concurrency,
                    version,
                    EXTRACT(EPOCH FROM (NOW() - last_seen_at))::int AS seconds_since_last_beat,
                    CASE
                        WHEN NOW() - last_seen_at < INTERVAL '30 seconds' THEN 'alive'
                        WHEN NOW() - last_seen_at < INTERVAL '2 minutes' THEN 'stale'
                        ELSE 'dead'
                    END AS status
                FROM worker_heartbeat
                ORDER BY last_seen_at DESC
                """
            )
    except asyncpg.UndefinedTableError:
        return {"workers": [], "note": "worker_heartbeat not initialized yet (start a worker)."}
    out = []
    for r in rows:
        d = dict(r)
        for k, v in list(d.items()):
            if hasattr(v, "isoformat"):
                d[k] = v.isoformat()
        out.append(d)
    return {"workers": out}


# ============================================================
# WEEKLY REPORT
# ============================================================

@router.post("/weekly-report")
async def trigger_weekly_report(user: dict = Depends(require_master_admin)):
    since = _now_utc() - timedelta(days=7)
    async with core.state.db_pool.acquire() as conn:
        costs = await conn.fetchrow("""
            SELECT COALESCE(SUM(CASE WHEN category = 'openai' THEN cost_usd ELSE 0 END), 0)::decimal AS openai,
            COALESCE(SUM(CASE WHEN category = 'storage' THEN cost_usd ELSE 0 END), 0)::decimal AS storage,
            COALESCE(SUM(CASE WHEN category = 'compute' THEN cost_usd ELSE 0 END), 0)::decimal AS compute
            FROM cost_tracking WHERE created_at >= $1
        """, since)
        revenue = await conn.fetchval("SELECT COALESCE(SUM(amount), 0) FROM revenue_tracking WHERE created_at >= $1", since)

    await notify_weekly_costs(float(costs["openai"] or 0), float(costs["storage"] or 0), float(costs["compute"] or 0), float(revenue or 0))
    return {"status": "sent"}


# ============================================================
# ANNOUNCEMENTS
# ============================================================

@router.post("/announcements/send")
async def _announcements_send(data: AnnouncementRequest, background_tasks: BackgroundTasks, user: dict = Depends(require_admin)):
    """Legacy /send path."""
    return await send_announcement(data, background_tasks, user)


@router.get("/announcements")
async def get_announcements(limit: int = 20, user: dict = Depends(require_admin)):
    async with core.state.db_pool.acquire() as conn:
        anns = await conn.fetch("SELECT * FROM announcements ORDER BY created_at DESC LIMIT $1", limit)
    return [dict(a) for a in anns]


@router.post("/announcements")
async def post_announcements(data: AnnouncementRequest, background_tasks: BackgroundTasks, user: dict = Depends(require_admin)):
    """Announcement endpoint at the path frontend expects"""
    return await send_announcement(data, background_tasks, user)


# ============================================================
# NOTIFICATION SETTINGS
# ============================================================

@router.get("/notification-settings")
async def get_notification_settings(user: dict = Depends(require_admin)):
    """Get Discord notification settings"""
    settings = admin_settings_cache.get("notifications", {})
    return {
        "notify_mrr_charge": settings.get("notify_mrr_charge", True),
        "notify_topup": settings.get("notify_topup", True),
        "notify_upgrade": settings.get("notify_upgrade", True),
        "notify_downgrade": settings.get("notify_downgrade", True),
        "notify_cancel": settings.get("notify_cancel", True),
        "notify_refund": settings.get("notify_refund", True),
        "notify_openai_cost": settings.get("notify_openai_cost", True),
        "notify_storage_cost": settings.get("notify_storage_cost", True),
        "notify_compute_cost": settings.get("notify_compute_cost", True),
        "notify_weekly_report": settings.get("notify_weekly_report", True),
        "notify_stripe_payout": settings.get("notify_stripe_payout", True),
        "notify_cloud_billing": settings.get("notify_cloud_billing", True),
        "notify_render_renewal": settings.get("notify_render_renewal", True),
        "stripe_payout_day": settings.get("stripe_payout_day", 15),
        "cloud_billing_day": settings.get("cloud_billing_day", 1),
        "render_renewal_day": settings.get("render_renewal_day", 7),
        "admin_webhook_url": settings.get("admin_webhook_url", ADMIN_DISCORD_WEBHOOK_URL or ""),
    }


@router.put("/notification-settings")
async def update_notification_settings(settings: NotificationSettings, user: dict = Depends(require_admin)):
    """Update Discord notification settings"""
    notif_settings = {
        "notify_mrr_charge": settings.notify_mrr_charge,
        "notify_topup": settings.notify_topup,
        "notify_upgrade": settings.notify_upgrade,
        "notify_downgrade": settings.notify_downgrade,
        "notify_cancel": settings.notify_cancel,
        "notify_refund": settings.notify_refund,
        "notify_openai_cost": settings.notify_openai_cost,
        "notify_storage_cost": settings.notify_storage_cost,
        "notify_compute_cost": settings.notify_compute_cost,
        "notify_weekly_report": settings.notify_weekly_report,
        "notify_stripe_payout": settings.notify_stripe_payout,
        "notify_cloud_billing": settings.notify_cloud_billing,
        "notify_render_renewal": settings.notify_render_renewal,
        "stripe_payout_day": settings.stripe_payout_day,
        "cloud_billing_day": settings.cloud_billing_day,
        "render_renewal_day": settings.render_renewal_day,
        "admin_webhook_url": settings.admin_webhook_url,
    }
    core.state.admin_settings_cache["notifications"] = notif_settings
    async with core.state.db_pool.acquire() as conn:
        await conn.execute("UPDATE admin_settings SET settings_json = $1, updated_at = NOW() WHERE id = 1", json.dumps(core.state.admin_settings_cache))
    return {"status": "updated", "settings": notif_settings}


@router.post("/test-webhook")
async def test_webhook(data: dict, user: dict = Depends(require_admin)):
    """Send a test message to the provided Discord webhook"""
    webhook_url = data.get("webhook_url", "").strip()
    if not webhook_url:
        raise HTTPException(400, "Webhook URL required")
    if not webhook_url.startswith("https://discord.com/api/webhooks/"):
        raise HTTPException(400, "Invalid Discord webhook URL")
    test_embed = {
        "title": "\U0001f514 UploadM8 Webhook Test",
        "description": "If you see this message, your webhook is configured correctly!",
        "color": 0x22c55e,
        "fields": [
            {"name": "Status", "value": "\u2705 Connected", "inline": True},
            {"name": "Tested By", "value": user.get("email", "Admin"), "inline": True},
        ],
        "footer": {"text": "UploadM8 Admin Notifications"},
        "timestamp": _now_utc().isoformat()
    }
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(webhook_url, json={"embeds": [test_embed]})
            if r.status_code not in (200, 204):
                raise HTTPException(400, f"Discord returned status {r.status_code}")
    except httpx.TimeoutException:
        raise HTTPException(400, "Webhook request timed out")
    except Exception as e:
        raise HTTPException(400, f"Failed to send: {str(e)}")
    return {"status": "sent"}


# ============================================================
# BILLING REMINDERS
# ============================================================

@router.post("/check-billing-reminders")
async def check_billing_reminders(user: dict = Depends(require_admin)):
    """Check and send billing reminders for upcoming dates"""
    settings = get_notif_settings()
    today = _now_utc().day
    reminders_sent = []
    stripe_day = settings.get("stripe_payout_day", 15)
    cloud_day = settings.get("cloud_billing_day", 1)
    render_day = settings.get("render_renewal_day", 7)
    if (stripe_day - 2) == today or (stripe_day - 2 + 28) % 28 == today:
        await notify_billing_reminder("stripe_payout", f"Day {stripe_day} of this month", service="Stripe Payouts")
        reminders_sent.append("stripe_payout")
    if (cloud_day - 2) == today or (cloud_day - 2 + 28) % 28 == today:
        await notify_billing_reminder("cloud_billing", f"Day {cloud_day} of this month", service="AWS/Cloudflare")
        reminders_sent.append("cloud_billing")
    if (render_day - 2) == today or (render_day - 2 + 28) % 28 == today:
        await notify_billing_reminder("render_renewal", f"Day {render_day} of this month", service="Render Hosting")
        reminders_sent.append("render_renewal")
    return {"status": "checked", "reminders_sent": reminders_sent}


# ============================================================
# WALLET ADMIN
# ============================================================

@router.get("/users/{user_id}/wallet")
async def admin_get_user_wallet(
    user_id: str,
    admin: dict = Depends(require_admin),
):
    """Return the wallet + recent ledger for any user (admin only)."""
    if not _valid_uuid(user_id):
        raise HTTPException(400, "Invalid user ID")
    async with core.state.db_pool.acquire() as conn:
        target = await conn.fetchrow(
            "SELECT id, email, name, subscription_tier FROM users WHERE id = $1",
            user_id,
        )
        if not target:
            raise HTTPException(404, "User not found")

        wallet = await get_wallet(conn, target["id"])

        ledger = await conn.fetch(
            """
            SELECT token_type, delta, reason, upload_id, meta, created_at
            FROM   token_ledger
            WHERE  user_id = $1
            ORDER  BY created_at DESC
            LIMIT  100
            """,
            target["id"],
        )

    plan = get_plan(target["subscription_tier"] or "free")
    tier_slug = normalize_tier(target["subscription_tier"] or "free")
    tier_cfg = get_effective_tier_config()
    plan_row = tier_cfg_to_api_dict(tier_slug, tier_cfg.get(tier_slug, {}))

    return {
        "user": {
            "id":    str(target["id"]),
            "email": target["email"],
            "name":  target["name"],
            "tier":  target["subscription_tier"],
        },
        "wallet": {
            "put_balance":  int(wallet.get("put_balance", 0)),
            "aic_balance":  int(wallet.get("aic_balance", 0)),
            "put_reserved": int(wallet.get("put_reserved", 0)),
            "aic_reserved": int(wallet.get("aic_reserved", 0)),
        },
        "plan": plan_row,
        "plan_limits": {
            "put_daily":   int(plan_row.get("put_daily", plan.get("put_daily", 0))),
            "put_monthly": int(plan_row.get("put_monthly", plan.get("put_monthly", 0))),
            "aic_monthly": int(plan_row.get("aic_monthly", plan.get("aic_monthly", 0))),
        },
        "ledger": [
            {
                "token_type": r.get("token_type"),
                "delta": int(r.get("delta", 0)),
                "reason": r.get("reason"),
                "upload_id": str(r["upload_id"]) if r.get("upload_id") else None,
                "meta": r.get("meta"),
                "created_at": r["created_at"].isoformat() if r.get("created_at") else None,
            }
            for r in ledger
        ],
    }

@router.post("/users/{user_id}/wallet/adjust")
async def admin_adjust_wallet(
    user_id:  str,
    payload:  AdminWalletAdjust,
    request:  Request,
    background_tasks: BackgroundTasks,
    admin:    dict = Depends(require_admin),
):
    """
    Manually adjust a user's PUT or AIC balance.

    Modes:
      set      -> set balance to exactly this amount
      add      -> add tokens to current balance
      subtract -> subtract tokens (clamped to 0 -- never goes negative)

    All changes are written to token_ledger and admin_audit_log.
    """
    if not _valid_uuid(user_id):
        raise HTTPException(400, "Invalid user ID")
    async with core.state.db_pool.acquire() as conn:
        target = await conn.fetchrow(
            "SELECT id, email, name FROM users WHERE id = $1", user_id
        )
        if not target:
            raise HTTPException(404, "User not found")

        wallet = await get_wallet(conn, target["id"])
        col_name = assert_wallet_balance_column("put_balance" if payload.wallet == "put" else "aic_balance")
        col = col_name
        before = int(wallet.get(col, 0) or 0)

        if payload.mode == "set":
            new_val = int(payload.amount)
            delta   = new_val - before
        elif payload.mode == "add":
            new_val = before + int(payload.amount)
            delta   = int(payload.amount)
        else:  # subtract
            new_val = max(0, before - int(payload.amount))
            delta   = new_val - before  # negative

        await conn.execute(
            f"UPDATE wallets SET {col} = $1, updated_at = NOW() WHERE user_id = $2",
            new_val,
            target["id"],
        )

        meta_json = json.dumps({
            "ref_type": "admin_adjust",
            "admin_id": str(admin.get("id", "")),
            "admin_email": admin.get("email") or "",
            "mode": payload.mode,
            "before": before,
            "after": new_val,
            "reason": payload.reason,
        })
        await conn.execute(
            """
            INSERT INTO token_ledger (user_id, token_type, delta, reason, meta)
            VALUES ($1, $2, $3, $4, $5::jsonb)
            """,
            target["id"],
            payload.wallet,
            int(delta),
            f"admin_{payload.mode}_{payload.wallet}",
            meta_json,
        )

        try:
            await conn.execute(
                """
                INSERT INTO admin_audit_log
                    (user_id, admin_id, admin_email, action, details, ip_address)
                VALUES
                    ($1, $2, $3, $4, $5::jsonb, $6)
                """,
                target["id"],
                admin.get("id"),
                admin.get("email"),
                "ADMIN_WALLET_ADJUST",
                json.dumps({
                    "wallet": payload.wallet,
                    "mode":   payload.mode,
                    "amount": int(payload.amount),
                    "delta":  int(delta),
                    "before": before,
                    "after":  new_val,
                    "reason": payload.reason,
                }),
                request.client.host if request.client else None,
            )
        except Exception:
            # admin_audit_log may not exist in some environments; ledger is the source of truth
            pass

    # Send notification email to user for grants (add/set with positive delta only)
    if delta > 0 and payload.mode in ("add", "set"):
        background_tasks.add_task(
            send_admin_wallet_topup_email,
            target["email"],
            target["name"] or "there",
            payload.wallet,
            int(delta),
            new_val,
            payload.reason or "Tokens credited to your account by the UploadM8 team.",
        )

    return {
        "ok":     True,
        "wallet": payload.wallet,
        "mode":   payload.mode,
        "before": before,
        "after":  new_val,
        "delta":  int(delta),
    }


# ============================================================
# LEADERBOARD & MISC
# ============================================================

@router.get("/leaderboard")
async def get_leaderboard(range: str = Query("30d"), sort: str = Query("uploads"), user: dict = Depends(require_admin)):
    minutes = {"7d": 10080, "30d": 43200, "90d": 129600}.get(range, 43200)
    since = _now_utc() - timedelta(minutes=minutes)
    async with core.state.require_pool().acquire() as conn:
        if sort == "revenue":
            rows = await conn.fetch("SELECT u.id, u.name, u.email, u.subscription_tier, COALESCE(SUM(r.amount), 0)::decimal AS revenue, COUNT(DISTINCT up.id)::int AS uploads FROM users u LEFT JOIN revenue_tracking r ON u.id = r.user_id AND r.created_at >= $1 LEFT JOIN uploads up ON u.id = up.user_id AND up.created_at >= $1 GROUP BY u.id ORDER BY revenue DESC LIMIT 10", since)
        else:
            rows = await conn.fetch("SELECT u.id, u.name, u.email, u.subscription_tier, 0::decimal AS revenue, COUNT(up.id)::int AS uploads FROM users u LEFT JOIN uploads up ON u.id = up.user_id AND up.created_at >= $1 GROUP BY u.id ORDER BY uploads DESC LIMIT 10", since)
    return [{"id": str(r["id"]), "name": r["name"] or "Unknown", "email": r["email"], "tier": r["subscription_tier"] or "free", "uploads": r["uploads"] or 0, "revenue": float(r["revenue"] or 0), "views": 0} for r in rows]


@router.get("/countries")
async def get_countries(range: str = Query("30d"), user: dict = Depends(require_admin)):
    """Return user count by country. Populated from CF-IPCountry header at registration."""
    days = int(range.replace("d", "")) if range.endswith("d") and range[:-1].isdigit() else 30
    since = _now_utc() - timedelta(days=days)
    try:
        async with core.state.db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT country, COUNT(*) AS users
                FROM users
                WHERE country IS NOT NULL
                  AND created_at >= $1
                GROUP BY country
                ORDER BY users DESC
                LIMIT 50
                """,
                since,
            )
        return [{"country": r["country"], "users": int(r["users"])} for r in rows]
    except Exception:
        return []  # column may not exist yet on older deployments


@router.get("/activity")
async def get_admin_activity(limit: int = Query(10), user: dict = Depends(require_admin)):
    async with core.state.db_pool.acquire() as conn:
        signups = await conn.fetch("SELECT 'signup' as type, id as user_id, name, email, created_at FROM users ORDER BY created_at DESC LIMIT $1", limit // 2)
        uploads = await conn.fetch("SELECT 'upload' as type, user_id, filename, status, created_at FROM uploads ORDER BY created_at DESC LIMIT $1", limit // 2)
        payments = await conn.fetch("SELECT 'payment' as type, user_id, amount, source, created_at FROM revenue_tracking ORDER BY created_at DESC LIMIT $1", limit // 2)
    activities = []
    for s in signups:
        activities.append({"id": str(s["user_id"]), "user_id": str(s["user_id"]), "type": "signup", "description": f"{s['name'] or s['email']} signed up", "created_at": s["created_at"].isoformat() if s["created_at"] else None})
    for u in uploads:
        activities.append({"id": str(uuid.uuid4()), "user_id": str(u["user_id"]), "type": "upload", "description": f"Uploaded {u['filename'] or 'video'} ({u['status']})", "created_at": u["created_at"].isoformat() if u["created_at"] else None})
    for p in payments:
        activities.append({"id": str(uuid.uuid4()), "user_id": str(p["user_id"]) if p["user_id"] else None, "type": "payment", "description": f"${float(p['amount']):.2f} - {p['source'] or 'payment'}", "created_at": p["created_at"].isoformat() if p["created_at"] else None})
    activities.sort(key=lambda x: x["created_at"] or "", reverse=True)
    return activities[:limit]


@router.get("/top-users")
async def get_admin_top_users(limit: int = Query(5), sort: str = Query("revenue"), user: dict = Depends(require_admin)):
    async with core.state.db_pool.acquire() as conn:
        if sort == "revenue":
            rows = await conn.fetch("SELECT u.id, u.name, u.email, u.subscription_tier, COALESCE(SUM(r.amount), 0)::decimal AS revenue, COUNT(DISTINCT up.id)::int AS uploads FROM users u LEFT JOIN revenue_tracking r ON u.id = r.user_id LEFT JOIN uploads up ON u.id = up.user_id GROUP BY u.id ORDER BY revenue DESC LIMIT $1", limit)
        else:
            rows = await conn.fetch("SELECT u.id, u.name, u.email, u.subscription_tier, 0::decimal AS revenue, COUNT(up.id)::int AS uploads FROM users u LEFT JOIN uploads up ON u.id = up.user_id GROUP BY u.id ORDER BY uploads DESC LIMIT $1", limit)
    return [{"id": str(r["id"]), "name": r["name"] or "Unknown", "email": r["email"], "tier": r["subscription_tier"] or "free", "subscription_tier": r["subscription_tier"] or "free", "revenue": float(r["revenue"] or 0), "uploads": r["uploads"] or 0} for r in rows]


@router.get("/operational-incidents")
async def admin_operational_incidents(
    limit: int = Query(80, le=500),
    offset: int = Query(0, ge=0),
    source: Optional[str] = Query(None, description="Filter by source (worker|web|api|...)"),
    incident_type: Optional[str] = Query(None, description="Filter by incident_type"),
    since_hours: Optional[int] = Query(None, ge=1, le=24 * 90, description="Only return incidents in the last N hours"),
    incident_id: Optional[str] = Query(
        None,
        description="Exact operational_incidents.id (UUID); skips since_hours so deep links work",
    ),
    user_id: Optional[str] = Query(None, description="Filter by user_id (UUID)"),
    upload_id: Optional[str] = Query(None, description="Filter by upload_id (UUID)"),
    q: Optional[str] = Query(None, description="Substring search across subject/body"),
    user: dict = Depends(get_current_user),
):
    """Operational incident log (upload failures, bug reports, client errors).

    Supports filtering by source/type/time/user/upload/search. Returns the rows
    plus aggregate counts so the admin UI can render summary cards.
    """
    if user.get("role") not in ("admin", "master_admin"):
        raise HTTPException(403, "Admin only")

    if core.state.db_pool is None:
        raise HTTPException(503, "Database not ready")

    where: List[str] = []
    params: List[Any] = []
    single_incident_mode = False

    def _add(clause: str, value: Any) -> None:
        params.append(value)
        where.append(clause.replace("$$", f"${len(params)}"))

    if incident_id:
        iid = incident_id.strip()
        try:
            uuid.UUID(iid)
        except ValueError as exc:
            raise HTTPException(400, "Invalid incident_id") from exc
        _add("i.id = $$::uuid", iid)
        single_incident_mode = True
    if source:
        _add("source = $$", source[:50])
    if incident_type:
        _add("incident_type = $$", incident_type[:120])
    if since_hours and not single_incident_mode:
        _add("created_at >= NOW() - ($$ || ' hours')::interval", str(int(since_hours)))
    if user_id:
        try:
            uuid.UUID(user_id)
            _add("user_id = $$::uuid", user_id)
        except ValueError as exc:
            raise HTTPException(400, "Invalid user_id") from exc
    if upload_id:
        try:
            uuid.UUID(upload_id)
            _add("upload_id = $$::uuid", upload_id)
        except ValueError as exc:
            raise HTTPException(400, "Invalid upload_id") from exc
    if q:
        like = f"%{q[:200]}%"
        params.append(like)
        idx_a = len(params)
        params.append(like)
        idx_b = len(params)
        where.append(f"(subject ILIKE ${idx_a} OR COALESCE(body,'') ILIKE ${idx_b})")

    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    rows: List[Any] = []
    summary: Dict[str, Any] = {"total": 0, "by_source": {}, "by_type": {}, "last_24h": 0}
    try:
        async with core.state.db_pool.acquire() as conn:
            limit_idx = len(params) + 1
            offset_idx = len(params) + 2
            rows = await conn.fetch(
                f"""
                SELECT i.id, i.source, i.incident_type, i.user_id, i.upload_id, i.subject,
                       LEFT(COALESCE(i.body, ''), 800) AS body_preview,
                       i.details, i.screenshot_r2_key,
                       i.email_sent_at, i.discord_sent_at, i.created_at,
                       u.email AS user_email
                FROM operational_incidents i
                LEFT JOIN users u ON u.id = i.user_id
                {where_sql}
                ORDER BY i.created_at DESC
                LIMIT ${limit_idx} OFFSET ${offset_idx}
                """,
                *params,
                limit,
                offset,
            )

            total_row = await conn.fetchrow(
                f"SELECT COUNT(*)::int AS n FROM operational_incidents i {where_sql}",
                *params,
            )
            summary["total"] = int(total_row["n"]) if total_row else 0

            agg_source = await conn.fetch(
                f"""
                SELECT source, COUNT(*)::int AS n
                FROM operational_incidents i {where_sql}
                GROUP BY source ORDER BY n DESC LIMIT 20
                """,
                *params,
            )
            summary["by_source"] = {r["source"]: int(r["n"]) for r in agg_source}

            agg_type = await conn.fetch(
                f"""
                SELECT incident_type, COUNT(*)::int AS n
                FROM operational_incidents i {where_sql}
                GROUP BY incident_type ORDER BY n DESC LIMIT 25
                """,
                *params,
            )
            summary["by_type"] = {r["incident_type"]: int(r["n"]) for r in agg_type}

            if single_incident_mode:
                summary["last_24h"] = 0
            else:
                last24 = await conn.fetchrow(
                    f"""
                    SELECT COUNT(*)::int AS n
                    FROM operational_incidents i
                    {where_sql + (' AND ' if where else 'WHERE ')}created_at >= NOW() - INTERVAL '24 hours'
                    """,
                    *params,
                )
                summary["last_24h"] = int(last24["n"]) if last24 else 0
    except asyncpg.exceptions.UndefinedTableError:
        logger.warning("operational_incidents: table missing (run migrations)")
        return {"incidents": [], "summary": summary, "limit": limit, "offset": offset}

    out: List[Dict[str, Any]] = []
    for r in rows:
        d = dict(r)
        d["id"] = str(d["id"])
        d["user_id"] = str(d["user_id"]) if d.get("user_id") else None
        d["upload_id"] = str(d["upload_id"]) if d.get("upload_id") else None
        d["details"] = _safe_json(d.get("details"), {})
        d["created_at"] = d["created_at"].isoformat() if d.get("created_at") else None
        d["email_sent_at"] = d["email_sent_at"].isoformat() if d.get("email_sent_at") else None
        d["discord_sent_at"] = d["discord_sent_at"].isoformat() if d.get("discord_sent_at") else None
        out.append(d)
    return {
        "incidents": out,
        "summary": summary,
        "limit": limit,
        "offset": offset,
    }


@router.get("/wallet-disputes")
async def admin_wallet_disputes(
    limit: int = Query(80, le=200),
    offset: int = Query(0, ge=0),
    status: Optional[str] = Query(None, description="open|in_review|resolved|rejected"),
    user: dict = Depends(get_current_user),
):
    if user.get("role") not in ("admin", "master_admin"):
        raise HTTPException(403, "Admin only")
    if core.state.db_pool is None:
        raise HTTPException(503, "Database not ready")
    if status and status not in ("open", "in_review", "resolved", "rejected"):
        raise HTTPException(400, "Invalid status filter")
    try:
        return await list_admin_wallet_disputes(
            core.state.db_pool, limit=limit, offset=offset, status=status
        )
    except asyncpg.exceptions.UndefinedTableError:
        return {"items": [], "total": 0, "limit": limit, "offset": offset}


@router.patch("/wallet-disputes/{dispute_id}")
async def admin_wallet_disputes_patch(
    dispute_id: str,
    data: AdminWalletDisputePatch,
    user: dict = Depends(get_current_user),
):
    if user.get("role") not in ("admin", "master_admin"):
        raise HTTPException(403, "Admin only")
    if core.state.db_pool is None:
        raise HTTPException(503, "Database not ready")
    if data.status is None and data.admin_internal_note is None and data.resolution_message is None:
        raise HTTPException(400, "No fields to update")
    try:
        row = await admin_patch_wallet_dispute(
            core.state.db_pool,
            dispute_id=dispute_id,
            status=data.status,
            admin_internal_note=data.admin_internal_note,
            resolution_message=data.resolution_message,
        )
        async with core.state.db_pool.acquire() as conn:
            await log_admin_audit(
                conn,
                user_id=str(row.get("user_id") or user.get("id")),
                admin=user,
                action="wallet_dispute_patch",
                details={"dispute_id": dispute_id, "status": data.status, "resolution_set": data.resolution_message is not None},
            )
        return {"dispute": row}
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except LookupError as e:
        raise HTTPException(404, str(e)) from e
    except Exception as e:
        logger.warning("admin wallet_disputes_patch failed: %s", e)
        raise HTTPException(500, "Update failed") from e


@router.get("/all-failures")
async def admin_all_failures(
    limit: int = Query(150, le=500),
    since_hours: int = Query(24, ge=1, le=24 * 30),
    source_filter: Optional[str] = Query(None, description="incident|upload|event"),
    user: dict = Depends(get_current_user),
):
    """Unified failure feed merging three sources into one timeline:

    1. ``operational_incidents`` rows (worker errors, bug reports, JS errors,
       API 500s).
    2. ``uploads`` rows where ``status='failed'`` (in case the worker failed
       to record an incident).
    3. ``system_event_log`` rows where ``severity='ERROR'`` or
       ``outcome='FAILURE'`` (OAuth failures, webhook failures, etc).

    De-duplicated by ``upload_id`` where applicable (incidents win over upload
    rows for the same upload_id). Returned newest-first.
    """
    if user.get("role") not in ("admin", "master_admin"):
        raise HTTPException(403, "Admin only")
    if core.state.db_pool is None:
        raise HTTPException(503, "Database not ready")

    show_inc = source_filter in (None, "", "incident")
    show_up = source_filter in (None, "", "upload")
    show_evt = source_filter in (None, "", "event")

    incidents: List[Dict[str, Any]] = []
    failed_uploads: List[Dict[str, Any]] = []
    failure_events: List[Dict[str, Any]] = []
    seen_upload_ids: set = set()

    async with core.state.db_pool.acquire() as conn:
        if show_inc:
            try:
                rows = await conn.fetch(
                    """
                    SELECT i.id, i.source, i.incident_type, i.user_id, i.upload_id,
                           i.subject, LEFT(COALESCE(i.body,''), 400) AS body_preview,
                           i.details, i.created_at, u.email AS user_email
                    FROM operational_incidents i
                    LEFT JOIN users u ON u.id = i.user_id
                    WHERE i.created_at >= NOW() - ($1 || ' hours')::interval
                    ORDER BY i.created_at DESC
                    LIMIT $2
                    """,
                    str(since_hours),
                    limit,
                )
                for r in rows:
                    uid = str(r["upload_id"]) if r["upload_id"] else None
                    if uid:
                        seen_upload_ids.add(uid)
                    incidents.append({
                        "kind": "incident",
                        "id": str(r["id"]),
                        "source": r["source"],
                        "incident_type": r["incident_type"],
                        "user_id": str(r["user_id"]) if r["user_id"] else None,
                        "user_email": r["user_email"],
                        "upload_id": uid,
                        "subject": r["subject"],
                        "body_preview": r["body_preview"],
                        "details": _safe_json(r["details"], {}),
                        "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                    })
            except asyncpg.exceptions.UndefinedTableError:
                pass

        if show_up:
            try:
                rows = await conn.fetch(
                    """
                    SELECT up.id, up.user_id, up.filename, up.error_code, up.error_detail,
                           up.platforms, up.status,
                           COALESCE(up.processing_finished_at, up.updated_at, up.created_at) AS ts,
                           u.email AS user_email
                    FROM uploads up
                    LEFT JOIN users u ON u.id = up.user_id
                    WHERE up.status = 'failed'
                      AND COALESCE(up.processing_finished_at, up.updated_at, up.created_at)
                          >= NOW() - ($1 || ' hours')::interval
                    ORDER BY ts DESC
                    LIMIT $2
                    """,
                    str(since_hours),
                    limit,
                )
                for r in rows:
                    uid = str(r["id"])
                    if uid in seen_upload_ids:
                        continue
                    failed_uploads.append({
                        "kind": "upload",
                        "id": uid,
                        "source": "upload",
                        "incident_type": (r["error_code"] or "upload_failed")[:120],
                        "user_id": str(r["user_id"]) if r["user_id"] else None,
                        "user_email": r["user_email"],
                        "upload_id": uid,
                        "subject": f"Upload failed: {r['filename'] or uid}",
                        "body_preview": (r["error_detail"] or "")[:400],
                        "details": {
                            "filename": r["filename"],
                            "platforms": list(r["platforms"] or []),
                            "error_code": r["error_code"],
                        },
                        "created_at": r["ts"].isoformat() if r["ts"] else None,
                    })
            except asyncpg.exceptions.UndefinedTableError:
                pass

        if show_evt:
            try:
                rows = await conn.fetch(
                    """
                    SELECT e.id, e.user_id, e.event_category, e.action, e.resource_type,
                           e.resource_id, e.details, e.severity, e.outcome, e.created_at,
                           u.email AS user_email
                    FROM system_event_log e
                    LEFT JOIN users u ON u.id = e.user_id
                    WHERE (e.severity = 'ERROR' OR e.outcome = 'FAILURE')
                      AND e.created_at >= NOW() - ($1 || ' hours')::interval
                    ORDER BY e.created_at DESC
                    LIMIT $2
                    """,
                    str(since_hours),
                    limit,
                )
                for r in rows:
                    failure_events.append({
                        "kind": "event",
                        "id": str(r["id"]),
                        "source": (r["event_category"] or "event").lower()[:50],
                        "incident_type": (r["action"] or "event")[:120],
                        "user_id": str(r["user_id"]) if r["user_id"] else None,
                        "user_email": r["user_email"],
                        "upload_id": r["resource_id"] if r["resource_type"] == "upload" else None,
                        "subject": f"{r['event_category']}: {r['action']}",
                        "body_preview": "",
                        "details": {
                            **(_safe_json(r["details"], {}) or {}),
                            "severity": r["severity"],
                            "outcome": r["outcome"],
                            "resource_type": r["resource_type"],
                            "resource_id": r["resource_id"],
                        },
                        "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                    })
            except asyncpg.exceptions.UndefinedTableError:
                pass

    merged = incidents + failed_uploads + failure_events
    merged.sort(key=lambda x: x.get("created_at") or "", reverse=True)
    merged = merged[:limit]

    summary = {
        "total": len(merged),
        "incidents": len(incidents),
        "failed_uploads": len(failed_uploads),
        "failure_events": len(failure_events),
        "since_hours": since_hours,
    }
    return {"failures": merged, "summary": summary}


def _clip_text(v: Any, *, max_chars: int = 16_000) -> str:
    s = "" if v is None else str(v)
    if len(s) <= max_chars:
        return s
    return s[: max_chars - len("...[truncated]")] + "...[truncated]"


def _extract_ai_artifact_subset(output_artifacts: Dict[str, Any]) -> Dict[str, Any]:
    """Return only AI/hydration-relevant artifact keys to keep payload focused."""
    if not isinstance(output_artifacts, dict):
        return {}
    keep: Dict[str, Any] = {}
    for k, raw in output_artifacts.items():
        key = str(k or "").strip()
        if not key:
            continue
        low = key.lower()
        if not (
            low.startswith("m8_")
            or "hydration" in low
            or "caption" in low
            or "content_attribution" in low
            or "ai_pipeline_trace" in low
            or "prompt" in low
            or "thumbnail" in low
            or "pikzels" in low
            or "provider_error" in low
            or low == "error"
        ):
            continue
        val = raw
        if isinstance(raw, str):
            parsed = _safe_json(raw, None)
            val = parsed if parsed is not None else _clip_text(raw, max_chars=20_000)
        keep[key] = val
    return keep


def _parse_artifact_json_with_error(output_artifacts: Dict[str, Any], key: str) -> tuple[Any, Optional[str]]:
    """Parse output_artifacts[key] and return (value, error_message)."""
    if not isinstance(output_artifacts, dict):
        return None, "output_artifacts_not_dict"
    if key not in output_artifacts:
        return None, "missing_key"
    raw = output_artifacts.get(key)
    if raw is None:
        return None, "null_value"
    if isinstance(raw, (dict, list)):
        return raw, None
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return None, "empty_string"
        try:
            return json.loads(s), None
        except Exception as exc:
            return None, f"json_parse_error: {exc}"
    return None, f"unsupported_type:{type(raw).__name__}"


def _normalize_platform_results(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    if isinstance(raw, dict):
        out: List[Dict[str, Any]] = []
        for k, v in raw.items():
            if isinstance(v, dict):
                row = dict(v)
                row.setdefault("platform", str(k))
                out.append(row)
        return out
    return []


def _looks_like_http_url(v: Any) -> bool:
    s = str(v or "").strip().lower()
    return s.startswith("http://") or s.startswith("https://")


def _mask_discord_webhook_preview(url: str) -> str:
    s = (url or "").strip()
    if not s:
        return ""
    if len(s) <= 44:
        return "https://…/(hidden)"
    return f"{s[:26]}…{s[-14:]}"


def _owner_discord_webhook_from_ai_trace_row(row: Dict[str, Any]) -> Optional[str]:
    for k in ("_trace_us_dw", "_trace_upr_dw", "_trace_u_dwh", "_trace_u_dwsn"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _collect_upload_diagnostics(
    *,
    output_artifacts: Dict[str, Any],
    hydration_payload: Any,
    ai_trace_blob: Any,
    thumbnail_trace: Any,
    pikzels_prompt_by_platform: Any,
    platform_results: List[Dict[str, Any]],
    error_code: Optional[str],
    error_detail: Optional[str],
) -> Dict[str, Any]:
    expected = [
        "hydration_payload",
        "thumbnail_trace",
        "pikzels_prompt_by_platform",
        "ai_pipeline_trace_v1",
        "provider_error_trace",
    ]
    parse_errors: Dict[str, str] = {}
    missing_artifacts: List[str] = []
    for k in expected:
        _, err = _parse_artifact_json_with_error(output_artifacts, k)
        if err == "missing_key":
            missing_artifacts.append(k)
        elif err:
            parse_errors[k] = err

    hp = hydration_payload if isinstance(hydration_payload, dict) else {}
    ev = hp.get("evidence") if isinstance(hp.get("evidence"), dict) else {}
    hydration_missing: List[str] = []
    for k in ("category", "anchor_phrase", "evidence", "signal_hashtags", "fusion_summary", "hydration_story", "trace_id"):
        val = hp.get(k)
        if k == "signal_hashtags":
            if not isinstance(val, list):
                hydration_missing.append(k)
        elif val is None or str(val).strip() == "":
            hydration_missing.append(k)
    for lane in ("geo", "osd", "music", "speech", "vision", "trill"):
        if not isinstance(ev.get(lane), dict):
            hydration_missing.append(f"evidence.{lane}")

    broken_links: List[Dict[str, Any]] = []
    for pr in platform_results:
        ok = bool(pr.get("success"))
        if not ok:
            continue
        candidates = [
            pr.get("platform_url"),
            pr.get("url"),
            pr.get("video_url"),
            pr.get("post_url"),
            pr.get("share_url"),
            pr.get("permalink"),
        ]
        present = [str(x).strip() for x in candidates if str(x or "").strip()]
        valid = [u for u in present if _looks_like_http_url(u)]
        if not valid:
            broken_links.append(
                {
                    "platform": str(pr.get("platform") or ""),
                    "reason": "successful_publish_missing_valid_url",
                    "url_candidates": present[:6],
                    "platform_video_id": pr.get("platform_video_id"),
                }
            )

    ai_events = len((ai_trace_blob.get("events") or [])) if isinstance(ai_trace_blob, dict) else 0
    thumb_events = len(thumbnail_trace) if isinstance(thumbnail_trace, list) else 0
    pikzels_platforms = len(pikzels_prompt_by_platform) if isinstance(pikzels_prompt_by_platform, dict) else 0

    return {
        "error_code_present": bool(error_code),
        "error_detail_present": bool(str(error_detail or "").strip()),
        "missing_artifacts": missing_artifacts,
        "artifact_parse_errors": parse_errors,
        "hydration_missing_keys": hydration_missing,
        "ai_event_count": ai_events,
        "thumbnail_event_count": thumb_events,
        "pikzels_prompt_platform_count": pikzels_platforms,
        "platform_results_count": len(platform_results),
        "broken_links": broken_links,
        "broken_links_count": len(broken_links),
    }


def _serialize_upload_ai_trace_row(
    row: Dict[str, Any],
    *,
    include_events: bool,
    include_raw_artifacts: bool,
) -> Dict[str, Any]:
    output_artifacts = _safe_json(row.get("output_artifacts"), {}) or {}
    ai_trace_blob, _ = _parse_artifact_json_with_error(output_artifacts, "ai_pipeline_trace_v1")
    hydration_payload, _ = _parse_artifact_json_with_error(output_artifacts, "hydration_payload")
    thumbnail_trace, _ = _parse_artifact_json_with_error(output_artifacts, "thumbnail_trace")
    pikzels_prompt_by_platform, _ = _parse_artifact_json_with_error(output_artifacts, "pikzels_prompt_by_platform")
    provider_error_trace, _ = _parse_artifact_json_with_error(output_artifacts, "provider_error_trace")
    events = list(ai_trace_blob.get("events") or []) if isinstance(ai_trace_blob, dict) else []
    if not isinstance(thumbnail_trace, list):
        thumbnail_trace = []
    if not isinstance(pikzels_prompt_by_platform, dict):
        pikzels_prompt_by_platform = {}
    if not isinstance(provider_error_trace, list):
        provider_error_trace = []
    platform_results = _normalize_platform_results(row.get("platform_results"))
    diagnostics = _collect_upload_diagnostics(
        output_artifacts=output_artifacts,
        hydration_payload=hydration_payload,
        ai_trace_blob=ai_trace_blob,
        thumbnail_trace=thumbnail_trace,
        pikzels_prompt_by_platform=pikzels_prompt_by_platform,
        platform_results=platform_results,
        error_code=row.get("error_code"),
        error_detail=row.get("error_detail"),
    )
    diagnostics["provider_error_count"] = len(provider_error_trace)

    try:
        from stages.notify_stage import _is_allowed_discord_webhook_url

        wh = _owner_discord_webhook_from_ai_trace_row(row)
        diagnostics["owner_discord_webhook_configured"] = bool(wh)
        diagnostics["owner_discord_webhook_valid_url"] = bool(wh and _is_allowed_discord_webhook_url(wh))
        diagnostics["owner_discord_webhook_preview"] = (
            _mask_discord_webhook_preview(wh) if wh else None
        )
    except Exception:
        diagnostics["owner_discord_webhook_configured"] = False
        diagnostics["owner_discord_webhook_valid_url"] = False
        diagnostics["owner_discord_webhook_preview"] = None

    raw_artifacts = None
    if include_raw_artifacts:
        raw_artifacts = {}
        for k, raw in (output_artifacts or {}).items():
            key = str(k or "").strip()
            if not key:
                continue
            val = raw
            if isinstance(raw, str):
                parsed = _safe_json(raw, None)
                val = parsed if parsed is not None else _clip_text(raw, max_chars=40_000)
            raw_artifacts[key] = val

    return {
        "upload_id": str(row.get("id") or ""),
        "user_id": str(row.get("user_id") or "") if row.get("user_id") else None,
        "user_email": row.get("user_email"),
        "filename": row.get("filename"),
        "status": row.get("status"),
        "platforms": list(row.get("platforms") or []),
        "processing_stage": row.get("processing_stage"),
        "processing_progress": row.get("processing_progress"),
        "error_code": row.get("error_code"),
        "error_detail": _clip_text(row.get("error_detail"), max_chars=2000) if row.get("error_detail") else None,
        "created_at": row.get("created_at").isoformat() if row.get("created_at") else None,
        "updated_at": row.get("updated_at").isoformat() if row.get("updated_at") else None,
        "processing_started_at": row.get("processing_started_at").isoformat() if row.get("processing_started_at") else None,
        "processing_finished_at": row.get("processing_finished_at").isoformat() if row.get("processing_finished_at") else None,
        "ai_title": row.get("ai_title"),
        "ai_caption": row.get("ai_caption"),
        "ai_generated_title": row.get("ai_generated_title"),
        "ai_generated_caption": row.get("ai_generated_caption"),
        "ai_generated_hashtags": list(row.get("ai_generated_hashtags") or []),
        "hydration_payload": hydration_payload,
        "thumbnail_trace": thumbnail_trace,
        "pikzels_prompt_by_platform": pikzels_prompt_by_platform,
        "provider_error_trace": provider_error_trace,
        "platform_results": platform_results,
        "diagnostics": diagnostics,
        "artifact_keys": sorted(list((output_artifacts or {}).keys())),
        "ai_artifacts": _extract_ai_artifact_subset(output_artifacts),
        "raw_artifacts": raw_artifacts,
        "ai_trace": {
            "event_count": int((ai_trace_blob or {}).get("event_count") or len(events)),
            "by_stage": (ai_trace_blob or {}).get("by_stage") if isinstance(ai_trace_blob, dict) else {},
            "events": events if include_events else [],
        },
    }


@router.get("/upload-ai-trace")
async def admin_upload_ai_trace(
    upload_id: Optional[str] = Query(None),
    upload_ids: Optional[str] = Query(None, description="Comma/newline/space-separated upload UUIDs"),
    q: Optional[str] = Query(None, description="Search by upload id, filename, or user email"),
    status: Optional[str] = Query(None),
    since_hours: int = Query(72, ge=1, le=24 * 3650),
    include_success: bool = Query(True),
    include_events: bool = Query(False, description="Include full ai_trace events array"),
    include_raw_artifacts: bool = Query(False, description="Include all output_artifacts keys/values"),
    limit: int = Query(50, ge=1, le=300),
    offset: int = Query(0, ge=0),
    user: dict = Depends(get_current_user),
):
    """
    Admin diagnostics feed for upload AI flow.
    Returns hydration payload, ai trace summary/events, and relevant prompt artifacts.
    """
    if user.get("role") not in ("admin", "master_admin"):
        raise HTTPException(403, "Admin only")
    if core.state.db_pool is None:
        raise HTTPException(503, "Database not ready")

    if upload_id and not _valid_uuid(upload_id):
        raise HTTPException(400, "Invalid upload_id")
    raw_ids: List[str] = []
    if upload_ids:
        raw_ids = [s for s in re.split(r"[\s,;]+", str(upload_ids).strip()) if s]
        raw_ids = [s.strip().strip('"').strip("'") for s in raw_ids]
        raw_ids = [s for s in raw_ids if s]
        # Allow pasted snippets like: data-id="uuid"
        cleaned: List[str] = []
        for s in raw_ids:
            if "data-id=" in s.lower():
                m = re.search(r'data-id=["\']([0-9a-fA-F-]{36})["\']', s)
                if m:
                    cleaned.append(m.group(1))
                continue
            cleaned.append(s)
        raw_ids = cleaned
        # Be tolerant: drop malformed IDs instead of failing the whole request.
        raw_ids = [s for s in raw_ids if _valid_uuid(s)]

    where_parts: List[str] = []
    params: List[Any] = []

    if upload_id:
        params.append(upload_id)
        where_parts.append(f"up.id = ${len(params)}::uuid")
    elif raw_ids:
        placeholders = []
        for uid in raw_ids[:500]:
            params.append(uid)
            placeholders.append(f"${len(params)}::uuid")
        where_parts.append(f"up.id IN ({', '.join(placeholders)})")
    else:
        params.append(str(since_hours))
        where_parts.append(
            f"COALESCE(up.processing_finished_at, up.updated_at, up.created_at) >= NOW() - (${len(params)} || ' hours')::interval"
        )
        if not include_success:
            where_parts.append("up.status <> 'completed'")
        if status:
            params.append(str(status))
            where_parts.append(f"up.status = ${len(params)}")
        if q:
            params.append(f"%{str(q).strip()}%")
            p = len(params)
            where_parts.append(
                f"(CAST(up.id AS text) ILIKE ${p} OR COALESCE(up.filename,'') ILIKE ${p} OR COALESCE(u.email,'') ILIKE ${p})"
            )

    where_sql = " AND ".join(where_parts) if where_parts else "TRUE"

    query = f"""
        SELECT
            up.id, up.user_id, up.filename, up.status, up.platforms,
            up.processing_stage, up.processing_progress,
            up.error_code, up.error_detail,
            up.created_at, up.updated_at, up.processing_started_at, up.processing_finished_at,
            up.ai_title, up.ai_caption, up.ai_generated_title, up.ai_generated_caption,
            up.ai_generated_hashtags, up.output_artifacts, up.platform_results,
            u.email AS user_email,
            us.discord_webhook AS _trace_us_dw,
            upr.discord_webhook AS _trace_upr_dw,
            u.preferences->>'discordWebhook' AS _trace_u_dwh,
            u.preferences->>'discord_webhook' AS _trace_u_dwsn
        FROM uploads up
        LEFT JOIN users u ON u.id = up.user_id
        LEFT JOIN user_settings us ON us.user_id = up.user_id
        LEFT JOIN user_preferences upr ON upr.user_id = up.user_id
        WHERE {where_sql}
        ORDER BY COALESCE(up.processing_finished_at, up.updated_at, up.created_at) DESC
        LIMIT ${len(params) + 1}
        OFFSET ${len(params) + 2}
    """
    run_params = list(params) + [limit, offset]

    async with core.state.db_pool.acquire() as conn:
        rows = await conn.fetch(query, *run_params)

    items = [
        _serialize_upload_ai_trace_row(
            dict(r),
            include_events=bool(include_events or upload_id),
            include_raw_artifacts=bool(include_raw_artifacts or upload_id),
        )
        for r in rows
    ]
    return {
        "count": len(items),
        "limit": limit,
        "offset": offset,
        "since_hours": since_hours,
        "include_success": bool(include_success),
        "include_events": bool(include_events or upload_id),
        "include_raw_artifacts": bool(include_raw_artifacts or upload_id),
        "items": items,
    }


@router.get("/tiktok-webhook-log")
async def admin_tiktok_webhook_log(
    limit: int = Query(50, le=500),
    offset: int = Query(0),
    event: Optional[str] = Query(None),
    user: dict = Depends(get_current_user),
):
    """Read the tiktok_webhook_events log (admin only)."""
    if user.get("role") not in ("admin", "master_admin"):
        raise HTTPException(403, "Admin only")

    async with core.state.db_pool.acquire() as conn:
        where = "WHERE event = $3" if event else ""
        params = [limit, offset] + ([event] if event else [])
        rows = await conn.fetch(
            f"""
            SELECT id, client_key, event, create_time, user_openid,
                   content, processed_at, sig_verified, handling_notes
            FROM tiktok_webhook_events
            {where}
            ORDER BY processed_at DESC
            LIMIT $1 OFFSET $2
            """,
            *params,
        )
        total = await conn.fetchval(
            "SELECT COUNT(*) FROM tiktok_webhook_events" + (" WHERE event=$1" if event else ""),
            *([event] if event else []),
        )

    return {
        "total": total,
        "records": [
            {
                **dict(r),
                "id": str(r["id"]),
                "processed_at": r["processed_at"].isoformat() if r["processed_at"] else None,
                "content": _safe_json(r["content"], {}),
            }
            for r in rows
        ],
    }


class TrillNameRejectIn(BaseModel):
    reason: str = Field("", max_length=500)


@router.get("/trill-display-names/pending")
async def admin_trill_display_names_pending(
    limit: int = Query(50, ge=1, le=200),
    user: dict = Depends(require_master_admin),
):
    async with core.state.db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT up.user_id, u.email, up.trill_public_name_pending, up.trill_public_name,
                   up.trill_public_name_status, up.updated_at
            FROM user_preferences up
            JOIN users u ON u.id = up.user_id
            WHERE up.trill_public_name_status = 'pending'
              AND NULLIF(btrim(up.trill_public_name_pending), '') IS NOT NULL
            ORDER BY up.updated_at DESC
            LIMIT $1
            """,
            limit,
        )
    return {
        "rows": [
            {
                "user_id": str(r["user_id"]),
                "email": r["email"],
                "pending": r["trill_public_name_pending"],
                "current_approved": r["trill_public_name"],
                "updated_at": r["updated_at"].isoformat() if r.get("updated_at") else None,
            }
            for r in rows
        ]
    }


@router.post("/trill-display-names/{target_user_id}/approve")
async def admin_trill_display_name_approve(
    target_user_id: str,
    user: dict = Depends(require_master_admin),
):
    if not _valid_uuid(target_user_id):
        raise HTTPException(400, "Invalid user id")
    admin_id = user["id"]
    async with core.state.db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            UPDATE user_preferences
            SET trill_public_name = NULLIF(btrim(trill_public_name_pending), ''),
                trill_public_name_pending = NULL,
                trill_public_name_status = 'approved',
                trill_public_name_rejection_reason = NULL,
                trill_public_name_reviewed_at = NOW(),
                trill_public_name_reviewed_by = $2::uuid,
                updated_at = NOW()
            WHERE user_id = $1::uuid AND trill_public_name_status = 'pending'
            RETURNING trill_public_name
            """,
            target_user_id,
            admin_id,
        )
    if not row:
        raise HTTPException(404, "No pending name for this user")
    return {"ok": True, "approved": row["trill_public_name"]}


@router.post("/trill-display-names/{target_user_id}/reject")
async def admin_trill_display_name_reject(
    target_user_id: str,
    body: TrillNameRejectIn,
    user: dict = Depends(require_master_admin),
):
    if not _valid_uuid(target_user_id):
        raise HTTPException(400, "Invalid user id")
    admin_id = user["id"]
    reason = (body.reason or "").strip() or "Rejected"
    async with core.state.db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            UPDATE user_preferences
            SET trill_public_name_pending = NULL,
                trill_public_name_status = 'rejected',
                trill_public_name_rejection_reason = $3,
                trill_public_name_reviewed_at = NOW(),
                trill_public_name_reviewed_by = $2::uuid,
                updated_at = NOW()
            WHERE user_id = $1::uuid AND trill_public_name_status = 'pending'
            RETURNING trill_public_name
            """,
            target_user_id,
            admin_id,
            reason[:500],
        )
    if not row:
        raise HTTPException(404, "No pending name for this user")
    return {"ok": True, "kept_approved": row["trill_public_name"], "reason": reason}
