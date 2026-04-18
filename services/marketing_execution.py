"""
DB-driven marketing execution: campaigns → audience match → email / Discord / in-app events.
Requires master-admin approval for outbound (email / discord / mixed) before sends.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import asyncpg
import httpx

from services.marketing_compliance import (
    assert_channel_rate_ok,
    is_suppressed,
    recent_delivery_exists,
    render_template,
    user_discord_marketing_allowed,
    user_email_marketing_allowed,
)
from services.wallet_marketing import _user_campaign_features
from stages.entitlements import normalize_tier
from stages.emails.base import send_email, mailgun_ready

logger = logging.getLogger("uploadm8.marketing_execution")

OUTBOUND_CHANNELS = frozenset({"email", "discord", "mixed"})


def _campaign_channel(ch: Optional[str]) -> str:
    return (ch or "in_app").strip().lower()[:50]


def _needs_approval_row(ch: str) -> bool:
    return _campaign_channel(ch) in OUTBOUND_CHANNELS


def _parse_targeting(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            j = json.loads(raw)
            return j if isinstance(j, dict) else {}
        except Exception:
            return {}
    return {}


async def _admin_discord_webhook(conn: asyncpg.Connection) -> Optional[str]:
    env_url = (os.environ.get("MARKETING_DISCORD_WEBHOOK_URL") or "").strip()
    if env_url.startswith("https://discord.com/api/webhooks/"):
        return env_url
    try:
        val = await conn.fetchval(
            "SELECT settings_json->'notifications'->>'admin_webhook_url' FROM admin_settings WHERE id = 1"
        )
        u = str(val).strip() if val else ""
        if u.startswith("https://discord.com/api/webhooks/"):
            return u
    except Exception:
        pass
    return None


async def send_discord_webhook(webhook_url: str, content: str) -> bool:
    if not webhook_url.startswith("https://discord.com/api/webhooks/"):
        return False
    body = {"content": (content or "")[:1900]}
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(webhook_url, json=body)
        if r.status_code >= 400:
            logger.warning("Discord webhook HTTP %s", r.status_code)
            return False
        return True
    except Exception as e:
        logger.warning("Discord webhook error: %s", e)
        return False


async def maybe_alert_pending_approvals(
    conn: asyncpg.Connection,
    *,
    min_interval_hours: int = 2,
) -> Dict[str, Any]:
    """
    If campaigns await approval, ping admin Discord (deduped).
    """
    pending = await conn.fetchval(
        """
        SELECT COUNT(*)::int FROM marketing_campaigns
        WHERE status = 'pending_approval'
          AND LOWER(COALESCE(channel, '')) IN ('email', 'discord', 'mixed')
        """
    )
    n = int(pending or 0)
    if n <= 0:
        return {"sent": False, "reason": "no_pending", "pending_count": 0}

    row = await conn.fetchrow(
        "SELECT last_sent_at FROM marketing_admin_alerts WHERE alert_key = $1",
        "pending_campaign_approval",
    )
    now = datetime.now(timezone.utc)
    if row and row["last_sent_at"]:
        delta = now - row["last_sent_at"]
        if delta.total_seconds() < max(3600, min_interval_hours * 3600):
            return {"sent": False, "reason": "deduped", "pending_count": n}

    webhook = await _admin_discord_webhook(conn)
    if not webhook:
        return {"sent": False, "reason": "no_webhook", "pending_count": n}

    msg = (
        f"**UploadM8 Marketing Ops** — {n} campaign(s) **await master admin approval** "
        f"before email/Discord sends. Open Marketing Ops to review."
    )
    ok = await send_discord_webhook(webhook, msg)
    await conn.execute(
        """
        INSERT INTO marketing_admin_alerts (alert_key, last_sent_at)
        VALUES ('pending_campaign_approval', NOW())
        ON CONFLICT (alert_key) DO UPDATE SET last_sent_at = NOW()
        """,
    )
    return {"sent": ok, "reason": "pinged" if ok else "discord_failed", "pending_count": n}


async def list_pending_approval_campaigns(conn: asyncpg.Connection) -> List[Dict[str, Any]]:
    rows = await conn.fetch(
        """
        SELECT id, name, objective, channel, status, schedule_at, targeting, template_subject,
               created_at, updated_at, range_key
        FROM marketing_campaigns
        WHERE status = 'pending_approval'
          AND LOWER(COALESCE(channel, '')) IN ('email', 'discord', 'mixed')
        ORDER BY created_at DESC
        LIMIT 100
        """
    )
    out: List[Dict[str, Any]] = []
    for r in rows:
        tg = _parse_targeting(r["targeting"])
        out.append(
            {
                "id": str(r["id"]),
                "name": r["name"],
                "objective": r["objective"] or "",
                "channel": r["channel"],
                "status": r["status"],
                "schedule_at": r["schedule_at"].isoformat() if r.get("schedule_at") else None,
                "targeting": tg,
                "template_subject": r["template_subject"] or "",
                "created_at": r["created_at"].isoformat() if r.get("created_at") else None,
                "range_key": str(r["range_key"] or "30d"),
            }
        )
    return out


def _user_matches_campaign(
    tier: str,
    feats: Dict[str, Any],
    targeting: Dict[str, Any],
) -> bool:
    tier_norm = normalize_tier(tier or "free")
    tiers = [normalize_tier(t) for t in (targeting.get("tiers") or []) if t]
    if tiers and tier_norm not in tiers:
        return False
    try:
        min_u = int(targeting.get("min_uploads_30d") or 0)
    except (TypeError, ValueError):
        min_u = 0
    if int(feats.get("uploads_window") or 0) < min_u:
        return False
    try:
        min_ctr = float(targeting.get("min_nudge_ctr_pct") or 0)
    except (TypeError, ValueError):
        min_ctr = 0.0
    if float(feats.get("nudge_ctr_pct") or 0) < min_ctr:
        return False
    try:
        min_score = float(targeting.get("min_enterprise_fit_score") or 0)
    except (TypeError, ValueError):
        min_score = 0.0
    if float(feats.get("enterprise_fit_score") or 0) < min_score:
        return False
    if bool(targeting.get("require_no_revenue_7d")) and float(feats.get("revenue_7d") or 0) > 0:
        return False
    return True


async def _iter_candidate_users(conn: asyncpg.Connection, limit: int = 2500):
    rows = await conn.fetch(
        """
        SELECT id, email, name, subscription_tier, role
        FROM users
        WHERE COALESCE(subscription_tier, '') NOT IN ('master_admin', 'friends_family', 'lifetime')
          AND COALESCE(role, '') NOT IN ('master_admin')
          AND COALESCE(status, 'active') = 'active'
        ORDER BY last_active_at DESC NULLS LAST
        LIMIT $1
        """,
        limit,
    )
    return rows


async def run_marketing_execution_tick(
    conn: asyncpg.Connection,
    *,
    max_per_campaign: int = 30,
    max_total_sends: int = 120,
) -> Dict[str, Any]:
    """
    Process active, approved outbound campaigns: queue sends, write marketing_events, update runs.
    """
    run_id = uuid.uuid4()
    await conn.execute(
        """
        INSERT INTO marketing_automation_runs (id, status, mode, meta)
        VALUES ($1::uuid, 'running', 'touchpoints_v1', '{}'::jsonb)
        """,
        run_id,
    )
    summary: Dict[str, Any] = {
        "run_id": str(run_id),
        "campaigns": [],
        "email_sent": 0,
        "discord_sent": 0,
        "in_app_written": 0,
        "skipped": 0,
        "errors": [],
    }
    total = 0
    users_eval = 0

    campaigns = await conn.fetch(
        """
        SELECT *
        FROM marketing_campaigns
        WHERE status = 'active'
          AND LOWER(COALESCE(channel, '')) IN ('email', 'discord', 'mixed')
          AND approved_at IS NOT NULL
          AND (schedule_at IS NULL OR schedule_at <= NOW())
        ORDER BY COALESCE(schedule_at, created_at) ASC
        LIMIT 20
        """
    )

    webhook = await _admin_discord_webhook(conn)

    for camp in campaigns:
        if total >= max_total_sends:
            break
        cid = str(camp["id"])
        ch = _campaign_channel(camp["channel"])
        targeting = _parse_targeting(camp["targeting"])
        range_key = str(camp.get("range_key") or "30d")
        camp_sent = 0

        ok_e, reason_e = await assert_channel_rate_ok(conn, "email")
        if ch in ("email", "mixed") and not ok_e:
            summary["errors"].append(f"campaign {cid}: {reason_e}")
            continue

        subj_tpl = camp.get("template_subject") or "A note from UploadM8"
        html_tpl = camp.get("template_body_html") or ""
        text_tpl = camp.get("template_body_text") or ""
        disc_tpl = camp.get("discord_message_text") or text_tpl or ""

        users = await _iter_candidate_users(conn, 2500)
        for urow in users:
            if camp_sent >= max_per_campaign or total >= max_total_sends:
                break
            uid = str(urow["id"])
            users_eval += 1
            tier = str(urow.get("subscription_tier") or "free")
            feats = await _user_campaign_features(conn, uid, range_key)
            if not _user_matches_campaign(tier, feats, targeting):
                continue
            if await recent_delivery_exists(conn, user_id=uid, campaign_id=cid, within_days=7):
                summary["skipped"] += 1
                continue

            ctx = {
                "name": (urow.get("name") or "there").split(" ")[0][:80],
                "first_name": (urow.get("name") or "there").split(" ")[0][:80],
                "app_url": os.environ.get("FRONTEND_URL", "https://app.uploadm8.com"),
            }

            touched = False

            if ch in ("email", "mixed"):
                skip_email = False
                if await is_suppressed(conn, uid, "email"):
                    skip_email = True
                elif not await user_email_marketing_allowed(conn, uid):
                    skip_email = True
                email = (urow.get("email") or "").strip()
                if not skip_email and email and "@" in email:
                    subject = render_template(str(subj_tpl), ctx)[:500]
                    html = render_template(str(html_tpl), ctx) or render_template(
                        "<p>Hi {{name}},</p><p>" + (camp.get("objective") or "Thanks for using UploadM8.") + "</p>",
                        ctx,
                    )
                    did = uuid.uuid4()
                    await conn.execute(
                        """
                        INSERT INTO marketing_touchpoint_deliveries (
                            id, user_id, channel, subject, body_text, body_html, status, scheduled_at, meta, campaign_id
                        )
                        VALUES ($1::uuid, $2::uuid, 'email', $3, $4, $5, 'pending', NOW(),
                                $6::jsonb, $7::uuid)
                        """,
                        did,
                        uid,
                        subject,
                        text_tpl and render_template(str(text_tpl), ctx),
                        html,
                        json.dumps({"campaign_id": cid, "run_id": str(run_id)}),
                        cid,
                    )
                    if mailgun_ready():
                        try:
                            await send_email(email, subject, html)
                            await conn.execute(
                                """
                                UPDATE marketing_touchpoint_deliveries
                                SET status = 'sent', sent_at = NOW() WHERE id = $1::uuid
                                """,
                                did,
                            )
                            await conn.execute(
                                """
                                INSERT INTO marketing_events (user_id, event_type, payload)
                                VALUES ($1::uuid, 'campaign_email_sent', $2::jsonb)
                                """,
                                uid,
                                json.dumps({"campaign_id": cid, "channel": ch}),
                            )
                            touched = True
                            camp_sent += 1
                            total += 1
                            summary["email_sent"] += 1
                        except Exception as e:
                            await conn.execute(
                                """
                                UPDATE marketing_touchpoint_deliveries
                                SET status = 'failed', error_detail = $2
                                WHERE id = $1::uuid
                                """,
                                did,
                                str(e)[:500],
                            )
                            summary["errors"].append(f"email {uid}: {e}")
                    else:
                        await conn.execute(
                            """
                            UPDATE marketing_touchpoint_deliveries
                            SET status = 'failed', error_detail = 'mailgun_not_configured'
                            WHERE id = $1::uuid
                            """,
                            did,
                        )

            if ch == "email" and not touched:
                continue

            if ch in ("discord", "mixed"):
                diskip = False
                if await is_suppressed(conn, uid, "discord"):
                    diskip = True
                elif not await user_discord_marketing_allowed(conn, uid):
                    diskip = True
                if not diskip:
                    line = render_template(str(disc_tpl), ctx)[:1800]
                    if not line.strip():
                        line = f"Hi {ctx['name']} — " + (camp.get("objective") or "UploadM8 update")[:500]
                    did = uuid.uuid4()
                    await conn.execute(
                        """
                        INSERT INTO marketing_touchpoint_deliveries (
                            id, user_id, channel, subject, body_text, body_html, status, scheduled_at, meta, campaign_id
                        )
                        VALUES ($1::uuid, $2::uuid, 'discord_in_app', '', $3, '', 'sent', NOW(),
                                $4::jsonb, $5::uuid)
                        """,
                        did,
                        uid,
                        line,
                        json.dumps(
                            {
                                "campaign_id": cid,
                                "run_id": str(run_id),
                                "delivery": "in_app_discord_prompt",
                            }
                        ),
                        cid,
                    )
                    await conn.execute(
                        """
                        INSERT INTO marketing_events (user_id, event_type, payload)
                        VALUES ($1::uuid, 'campaign_discord_prompt', $2::jsonb)
                        """,
                        uid,
                        json.dumps(
                            {
                                "campaign_id": cid,
                                "title": (camp.get("name") or "UploadM8")[:120],
                                "body": line[:2000],
                                "cta_href": ctx.get("app_url", "") + "/settings.html",
                                "cta_label": "Notification settings",
                            }
                        ),
                    )
                    touched = True
                    camp_sent += 1
                    total += 1
                    summary["discord_sent"] += 1
                    summary["in_app_written"] += 1
                elif ch == "discord":
                    continue

            if ch == "discord" and not touched:
                continue

        if camp_sent and webhook and ch in ("discord", "mixed", "email"):
            await send_discord_webhook(
                webhook,
                f"**Marketing run** · campaign **{camp.get('name') or cid[:8]}** · touches **{camp_sent}** users "
                f"(channel `{ch}`). Email uses Mailgun; Discord leg is in-app + community (no PII on webhook).",
            )

        summary["campaigns"].append({"id": cid, "sent": camp_sent})

    await conn.execute(
        """
        UPDATE marketing_automation_runs
        SET finished_at = NOW(),
            status = 'ok',
            users_evaluated = $2,
            users_messaged = $3,
            email_sent = $4,
            discord_sent = $5,
            in_app_written = $6,
            skipped_dedupe = $7,
            error_detail = $8,
            meta = $9::jsonb
        WHERE id = $1::uuid
        """,
        run_id,
        users_eval,
        total,
        summary["email_sent"],
        summary["discord_sent"],
        summary["in_app_written"],
        summary["skipped"],
        "; ".join(summary["errors"])[:2000] if summary["errors"] else None,
        json.dumps({"summary": summary}),
    )
    return summary


def approval_required_for_channel(channel: str) -> bool:
    return _needs_approval_row(channel)
