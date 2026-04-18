"""
Opt-in multi-channel marketing automation (email, Discord webhook, in-app wallet nudges).

Requires MARKETING_AUTOMATION_ENABLED=1. Uses aggregate user context + OpenAI when
OPENAI_API_KEY is set; otherwise deterministic copy. Designed for high-tier users with
unused platform connection headroom.
"""

from __future__ import annotations

import html
import json
import logging
import os
import uuid
from typing import Any, Dict, List, Optional, Tuple

import httpx

from stages.entitlements import get_entitlements_for_tier, normalize_tier
from services.growth_intelligence import fetch_user_engagement_snapshot
from stages.emails.base import (
    FRONTEND_URL,
    cta_button,
    email_shell,
    intro_row,
    mailgun_ready,
    send_email,
)

logger = logging.getLogger("uploadm8.marketing_automation")

SEGMENT_KEY = "high_tier_platform_headroom_v1"

MARKETING_AUTOMATION_ENABLED = os.environ.get("MARKETING_AUTOMATION_ENABLED", "").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
MARKETING_AUTOMATION_INTERVAL_SEC = int(os.environ.get("MARKETING_AUTOMATION_INTERVAL_SEC", str(4 * 3600)))
MARKETING_AUTOMATION_MAX_USERS = int(os.environ.get("MARKETING_AUTOMATION_MAX_USERS", "80"))
MARKETING_AUTOMATION_DEDUPE_DAYS = int(os.environ.get("MARKETING_AUTOMATION_DEDUPE_DAYS", "14"))

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_TOUCHPOINT_MODEL", "gpt-4o-mini")

ELIGIBLE_TIERS = frozenset(
    {"creator_pro", "studio", "agency", "friends_family", "lifetime"}
)


def _channels_enabled() -> Tuple[bool, bool, bool]:
    raw = os.environ.get("MARKETING_AUTOMATION_CHANNELS", "email,discord,in_app").lower()
    parts = {p.strip() for p in raw.split(",") if p.strip()}
    return ("email" in parts, "discord" in parts, "in_app" in parts)


def _qualifies(tier: str, n_conn: int, max_ac: int, per_pf: int) -> bool:
    if max_ac <= 0 or n_conn >= max_ac:
        return False
    headroom = max_ac - n_conn
    t = normalize_tier(tier)
    if t not in ELIGIBLE_TIERS:
        return False
    if t in ("studio", "agency", "friends_family", "lifetime"):
        return headroom >= 2 or (n_conn >= 2 and headroom >= 1)
    if t == "creator_pro":
        return headroom >= 2
    return False


async def _recent_touchpoint(conn, user_id: str) -> bool:
    row = await conn.fetchval(
        """
        SELECT 1 FROM marketing_touchpoint_deliveries
        WHERE user_id = $1::uuid
          AND meta->>'segment_key' = $2
          AND created_at >= NOW() - ($3::int * INTERVAL '1 day')
        LIMIT 1
        """,
        user_id,
        SEGMENT_KEY,
        MARKETING_AUTOMATION_DEDUPE_DAYS,
    )
    return row is not None


async def _fetch_discord_webhook(conn, user_id: str) -> Optional[str]:
    row = await conn.fetchrow(
        """
        SELECT discord_webhook FROM user_settings WHERE user_id = $1::uuid
        """,
        user_id,
    )
    if row and row["discord_webhook"]:
        u = str(row["discord_webhook"]).strip()
        return u or None
    return None


async def _synthesize_copy(
    *,
    name: str,
    email: str,
    tier: str,
    n_conn: int,
    max_ac: int,
    per_pf: int,
    engagement: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, str], bool]:
    """Returns (fields, used_openai)."""
    plat_url = f"{FRONTEND_URL.rstrip('/')}/platforms.html"
    headroom = max_ac - n_conn
    eng = engagement or {}
    er = float(eng.get("engagement_rate_pct") or 0) if eng.get("samples_30d") else 0.0
    eng_hint = ""
    if eng.get("samples_30d") and er > 0:
        eng_hint = (
            f" Your recent posts average ~{er:.2f}% engagement (likes+comments+shares vs views) — "
            "more connected channels help you scale what's already working."
        )
    base = {
        "email_subject": f"{name.split()[0] if name else 'Hey'} — use your {tier.replace('_', ' ')} account slots",
        "email_body_plain": (
            f"Your {tier.replace('_', ' ')} plan supports up to {max_ac} connected channels "
            f"({n_conn} active logins today). Cross-posting more accounts usually lifts reach — "
            f"add another YouTube, TikTok, Instagram, or Facebook login in UploadM8."
            + eng_hint
        ),
        "in_app_title": "Connect more platforms on your plan",
        "in_app_body": (
            f"You have about {headroom} open connection slots ({n_conn}/{max_ac} used). "
            f"Link another channel to widen distribution."
            + (f" Engagement on your last uploads is ~{er:.2f}% — scale winners to more surfaces." if er > 0 else "")
        ),
        "discord_text": (
            f"**UploadM8** · {tier.replace('_', ' ')} plan\n"
            f"You can connect up to **{max_ac}** accounts ({n_conn} now). "
            f"Open {plat_url} to add another channel."
        ),
    }
    if not (OPENAI_API_KEY or "").strip():
        return base, False
    try:
        payload = {
            "name": name,
            "email": email,
            "tier": tier,
            "connected": n_conn,
            "max_accounts": max_ac,
            "per_platform_cap": per_pf,
            "headroom": headroom,
            "platforms_url": plat_url,
            "engagement_30d": engagement or {},
        }
        async with httpx.AsyncClient(timeout=45.0) as client:
            r = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": OPENAI_MODEL,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You write concise SaaS lifecycle marketing. Output JSON with keys: "
                                "email_subject (string), email_body_plain (string, <=520 chars), "
                                "in_app_title (string), in_app_body (string, <=320 chars), "
                                "discord_text (string, markdown, <=400 chars). "
                                "Tone: helpful, not hype. Mention connecting more social accounts. "
                                "If engagement_30d has samples, you may briefly reference strong or weak "
                                "likes/comments/shares vs views — never fabricate numbers."
                            ),
                        },
                        {"role": "user", "content": json.dumps(payload)},
                    ],
                    "temperature": 0.35,
                },
            )
        if r.status_code >= 400:
            return base, False
        data = r.json()
        txt = (data.get("choices") or [{}])[0].get("message", {}).get("content") or "{}"
        parsed = json.loads(txt)
        if isinstance(parsed, dict):
            for k in base:
                if parsed.get(k):
                    base[k] = str(parsed[k])[: 800 if k == "email_body_plain" else 400]
        return base, True
    except Exception as e:
        logger.warning("touchpoint OpenAI failed: %s", e)
        return base, False


async def run_touchpoint_cycle(pool) -> Dict[str, Any]:
    """
    Evaluate users, generate AI/deterministic copy, deliver across enabled channels.
    """
    out: Dict[str, Any] = {
        "ok": True,
        "enabled": MARKETING_AUTOMATION_ENABLED,
        "users_messaged": 0,
        "email_sent": 0,
        "discord_sent": 0,
        "in_app_written": 0,
        "skipped_dedupe": 0,
    }
    if not MARKETING_AUTOMATION_ENABLED:
        return out

    want_email, want_discord, want_in_app = _channels_enabled()
    run_id = uuid.uuid4()

    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO marketing_automation_runs (id, status, mode, segment_key, meta)
            VALUES ($1, 'running', 'touchpoints_v1', $2, '{}'::jsonb)
            """,
            run_id,
            SEGMENT_KEY,
        )

        rows = await conn.fetch(
            """
            SELECT u.id, u.email, u.name, COALESCE(u.subscription_tier, 'free') AS subscription_tier,
              (SELECT COUNT(*)::int FROM platform_tokens pt
               WHERE pt.user_id = u.id AND pt.revoked_at IS NULL) AS n_conn
            FROM users u
            WHERE LOWER(TRIM(COALESCE(u.subscription_tier, 'free'))) = ANY(
                SELECT LOWER(TRIM(x)) FROM unnest($1::text[]) AS t(x)
            )
              AND COALESCE(u.role, '') NOT IN ('master_admin')
              AND COALESCE(u.email_verified, true) IS NOT FALSE
            """,
            list(ELIGIBLE_TIERS),
        )

        evaluated = 0
        for r in rows:
            if out["users_messaged"] >= MARKETING_AUTOMATION_MAX_USERS:
                break
            uid = str(r["id"])
            tier = str(r["subscription_tier"] or "free")
            tier_n = normalize_tier(tier)
            ent = get_entitlements_for_tier(tier_n)
            n_conn = int(r["n_conn"] or 0)
            evaluated += 1
            if not _qualifies(tier_n, n_conn, ent.max_accounts, ent.max_accounts_per_platform):
                continue
            if await _recent_touchpoint(conn, uid):
                out["skipped_dedupe"] += 1
                continue

            name = (r["name"] or "there").strip()
            email = (r["email"] or "").strip()
            if not email:
                continue

            touched_any = False
            eng_ctx = await fetch_user_engagement_snapshot(conn, uid)

            fields, used_ai = await _synthesize_copy(
                name=name,
                email=email,
                tier=tier_n,
                n_conn=n_conn,
                max_ac=ent.max_accounts,
                per_pf=ent.max_accounts_per_platform,
                engagement=eng_ctx,
            )
            meta_base = {
                "segment_key": SEGMENT_KEY,
                "tier": tier_n,
                "used_openai": used_ai,
                "connected": n_conn,
                "max_accounts": ent.max_accounts,
                "engagement_rate_pct_30d": eng_ctx.get("engagement_rate_pct"),
                "engagement_samples_30d": eng_ctx.get("samples_30d"),
            }

            # In-app: replace prior pending rows for this user/channel
            if want_in_app:
                await conn.execute(
                    """
                    DELETE FROM marketing_touchpoint_deliveries
                    WHERE user_id = $1::uuid AND channel = 'in_app' AND status = 'pending'
                    """,
                    uid,
                )
                await conn.execute(
                    """
                    INSERT INTO marketing_touchpoint_deliveries (
                        user_id, channel, subject, body_text, status, meta
                    )
                    VALUES ($1::uuid, 'in_app', $2, $3, 'pending', $4::jsonb)
                    """,
                    uid,
                    fields["in_app_title"][:500],
                    fields["in_app_body"],
                    json.dumps(meta_base),
                )
                out["in_app_written"] += 1
                touched_any = True

            if want_email and mailgun_ready():
                body_html = email_shell(
                    intro_row(fields["email_subject"], html.escape(fields["email_body_plain"]))
                    + cta_button("Open Platforms", f"{FRONTEND_URL.rstrip('/')}/platforms.html"),
                    preheader_text=fields["email_subject"][:120],
                )
                eid = uuid.uuid4()
                await conn.execute(
                    """
                    INSERT INTO marketing_touchpoint_deliveries (
                        id, user_id, channel, subject, body_text, body_html, status, meta
                    )
                    VALUES ($1::uuid, $2::uuid, 'email', $3, $4, $5, 'pending', $6::jsonb)
                    """,
                    eid,
                    uid,
                    fields["email_subject"][:500],
                    fields["email_body_plain"],
                    body_html,
                    json.dumps({**meta_base, "phase": "queued"}),
                )
                try:
                    await send_email(email, fields["email_subject"][:500], body_html)
                    await conn.execute(
                        """
                        UPDATE marketing_touchpoint_deliveries
                        SET status = 'sent', sent_at = NOW()
                        WHERE id = $1::uuid
                        """,
                        eid,
                    )
                    out["email_sent"] += 1
                    touched_any = True
                except Exception as ex:
                    logger.warning("touchpoint email fail user=%s: %s", uid, ex)
                    await conn.execute(
                        """
                        UPDATE marketing_touchpoint_deliveries
                        SET status = 'failed', error_detail = $2
                        WHERE id = $1::uuid
                        """,
                        eid,
                        str(ex)[:500],
                    )

            if want_discord:
                wh = await _fetch_discord_webhook(conn, uid)
                if wh and wh.startswith("http"):
                    did = uuid.uuid4()
                    await conn.execute(
                        """
                        INSERT INTO marketing_touchpoint_deliveries (
                            id, user_id, channel, subject, body_text, status, meta
                        )
                        VALUES ($1::uuid, $2::uuid, 'discord', $3, $4, 'pending', $5::jsonb)
                        """,
                        did,
                        uid,
                        "UploadM8 · connect more accounts",
                        fields["discord_text"],
                        json.dumps(meta_base),
                    )
                    try:
                        async with httpx.AsyncClient(timeout=15.0) as client:
                            dr = await client.post(wh, json={"content": fields["discord_text"][:1800]})
                        if dr.status_code < 400:
                            await conn.execute(
                                """
                                UPDATE marketing_touchpoint_deliveries
                                SET status = 'sent', sent_at = NOW()
                                WHERE id = $1::uuid
                                """,
                                did,
                            )
                            out["discord_sent"] += 1
                            touched_any = True
                        else:
                            await conn.execute(
                                """
                                UPDATE marketing_touchpoint_deliveries
                                SET status = 'failed', error_detail = $2
                                WHERE id = $1::uuid
                                """,
                                did,
                                f"HTTP {dr.status_code}"[:200],
                            )
                    except Exception as ex:
                        logger.warning("touchpoint discord fail user=%s: %s", uid, ex)
                        await conn.execute(
                            """
                            UPDATE marketing_touchpoint_deliveries
                            SET status = 'failed', error_detail = $2
                            WHERE id = $1::uuid
                            """,
                            did,
                            str(ex)[:500],
                        )

            if touched_any:
                out["users_messaged"] += 1

        await conn.execute(
            """
            UPDATE marketing_automation_runs
            SET finished_at = NOW(),
                status = 'completed',
                users_evaluated = $2,
                users_messaged = $3,
                email_sent = $4,
                discord_sent = $5,
                in_app_written = $6,
                skipped_dedupe = $7
            WHERE id = $1::uuid
            """,
            run_id,
            evaluated,
            out["users_messaged"],
            out["email_sent"],
            out["discord_sent"],
            out["in_app_written"],
            out["skipped_dedupe"],
        )

    logger.info(
        "[marketing-automation] cycle | messaged=%s email=%s discord=%s in_app=%s dedupe_skip=%s",
        out["users_messaged"],
        out["email_sent"],
        out["discord_sent"],
        out["in_app_written"],
        out["skipped_dedupe"],
    )
    return out


async def pending_in_app_as_opportunities(conn, user_id: str, links: Dict[str, str]) -> List[Dict[str, Any]]:
    rows = await conn.fetch(
        """
        SELECT id, subject, body_text, meta
        FROM marketing_touchpoint_deliveries
        WHERE user_id = $1::uuid AND channel = 'in_app' AND status = 'pending'
          AND scheduled_at <= NOW()
        ORDER BY scheduled_at DESC
        LIMIT 4
        """,
        user_id,
    )
    opps: List[Dict[str, Any]] = []
    for r in rows:
        meta = r["meta"] if isinstance(r["meta"], dict) else {}
        opps.append(
            {
                "type": "ai_touchpoint_platform_headroom",
                "severity": "promo" if (meta or {}).get("tier") in ("studio", "agency") else "info",
                "title": r["subject"] or "Grow reach with more channels",
                "body": r["body_text"] or "",
                "cta_label": "Connect platforms",
                "cta_link": links.get("platforms", "/platforms.html"),
                "metadata": {
                    "touchpoint_delivery_id": str(r["id"]),
                    "segment_key": (meta or {}).get("segment_key"),
                    "tier": (meta or {}).get("tier"),
                },
            }
        )
    return opps
