"""
Operational incidents: durable log + admin email + Discord for failures and bug reports.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

from core.config import ADMIN_OPS_EMAIL

logger = logging.getLogger("uploadm8.ops_incidents")

ERROR_DISCORD_WEBHOOK_URL = os.environ.get("ERROR_DISCORD_WEBHOOK_URL", "")
ADMIN_DISCORD_WEBHOOK_URL = os.environ.get("ADMIN_DISCORD_WEBHOOK_URL", "")

# Default dedupe window for alerts: one alert per (source, incident_type)
# bucket within this many seconds. Override via OPS_ALERT_DEDUPE_SECONDS env.
DEFAULT_ALERT_DEDUPE_SECONDS = int(os.environ.get("OPS_ALERT_DEDUPE_SECONDS", "900"))

# In-memory fallback for alert dedupe when Redis is unavailable. Maps a bucket
# key to the unix timestamp of the most recent alert. Single-process only —
# Redis is preferred for multi-worker deployments.
_LOCAL_ALERT_BUCKET: Dict[str, float] = {}


def _discord_url() -> str:
    return (ERROR_DISCORD_WEBHOOK_URL or ADMIN_DISCORD_WEBHOOK_URL or "").strip()


async def _send_discord_embed(
    title: str,
    description: str,
    color: int = 0xEF4444,
    *,
    fields: Optional[List[Dict[str, Any]]] = None,
    footer: Optional[str] = None,
) -> None:
    wh = _discord_url()
    if not wh:
        return
    embed: Dict[str, Any] = {
        "title": title[:256],
        "color": color,
        "description": description[:1800],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if fields:
        embed["fields"] = [
            {
                "name": str(f.get("name") or "")[:256],
                "value": str(f.get("value") or "")[:1024],
                "inline": bool(f.get("inline", True)),
            }
            for f in fields[:25]
            if f
        ]
    if footer:
        embed["footer"] = {"text": str(footer)[:2048]}
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            await client.post(wh, json={"embeds": [embed]})
    except Exception as e:
        logger.warning("ops_incidents discord: %s", e)


def _severity_color(severity: str) -> int:
    s = (severity or "").lower()
    if s == "critical":
        return 0xEF4444
    if s == "warning":
        return 0xF59E0B
    return 0x3B82F6


def _watchdog_discord_fields(details: Dict[str, Any], incident_id: Any) -> List[Dict[str, Any]]:
    fleet = details.get("fleet") if isinstance(details.get("fleet"), dict) else {}
    fields: List[Dict[str, Any]] = []
    if fleet:
        fields.append(
            {
                "name": "Fleet",
                "value": (
                    f"alive={fleet.get('alive_count', '?')} "
                    f"stale={fleet.get('stale_count', '?')} "
                    f"dead={fleet.get('dead_count', '?')}"
                ),
                "inline": True,
            }
        )
        fields.append(
            {
                "name": "Slots",
                "value": (
                    f"proc {fleet.get('process_slots_in_use', '?')}/{fleet.get('process_capacity', '?')} "
                    f"· mem_warn={fleet.get('workers_memory_warn', '?')}"
                ),
                "inline": True,
            }
        )
        if fleet.get("max_memory_rss_mb") is not None:
            fields.append(
                {
                    "name": "Max RSS",
                    "value": f"{fleet.get('max_memory_rss_mb')} MB",
                    "inline": True,
                }
            )
        if fleet.get("max_load_1m") is not None:
            fields.append(
                {
                    "name": "Load 1m",
                    "value": str(fleet.get("max_load_1m")),
                    "inline": True,
                }
            )
    if details.get("queues_pending") is not None:
        fields.append(
            {"name": "Redis pending", "value": str(details.get("queues_pending")), "inline": True}
        )
    render_summary = details.get("render_summary") if isinstance(details.get("render_summary"), dict) else {}
    if render_summary.get("critical_count"):
        fields.append(
            {
                "name": "Render critical",
                "value": str(render_summary.get("critical_count")),
                "inline": True,
            }
        )
    ev = details.get("event") if isinstance(details.get("event"), dict) else None
    if ev:
        fields.append(
            {
                "name": "Render event",
                "value": f"{ev.get('type')} @ {ev.get('timestamp')}",
                "inline": False,
            }
        )
    fields.append({"name": "Incident", "value": str(incident_id), "inline": False})
    return fields


async def _send_ops_email(subject: str, html_body: str) -> None:
    if not ADMIN_OPS_EMAIL:
        return
    try:
        from stages.emails.base import send_email, MAIL_FROM

        await send_email(ADMIN_OPS_EMAIL, subject[:200], html_body, from_addr=MAIL_FROM)
    except Exception as e:
        logger.warning("ops_incidents email: %s", e)


def _parse_uuid(val: Any) -> Optional[uuid.UUID]:
    if val is None or val == "":
        return None
    try:
        return val if isinstance(val, uuid.UUID) else uuid.UUID(str(val))
    except (ValueError, TypeError):
        return None


async def _claim_alert_slot(bucket_key: str, ttl_seconds: int) -> bool:
    """Try to claim an alert slot for ``bucket_key``.

    Returns True if this caller should send the alert (slot was free), False
    if a recent alert already exists in the dedupe window. Uses Redis SET NX
    EX when available, falls back to an in-process dict otherwise. Failure
    of the dedupe check itself defaults to allowing the alert (fail-open) so
    we don't silently swallow real incidents because Redis is down.
    """
    try:
        import core.state as _state

        rc = getattr(_state, "redis_client", None)
        if rc is not None:
            try:
                ok = await rc.set(
                    name=f"ops:alert:dedupe:{bucket_key}",
                    value=str(int(time.time())),
                    ex=int(ttl_seconds),
                    nx=True,
                )
                return bool(ok)
            except Exception as e:
                logger.debug("alert dedupe redis failed (%s); falling back to local", e)
    except Exception:
        pass

    now = time.time()
    last = _LOCAL_ALERT_BUCKET.get(bucket_key)
    if last is not None and (now - last) < ttl_seconds:
        return False
    _LOCAL_ALERT_BUCKET[bucket_key] = now
    if len(_LOCAL_ALERT_BUCKET) > 5000:
        cutoff = now - max(ttl_seconds, 3600)
        for k in [k for k, v in _LOCAL_ALERT_BUCKET.items() if v < cutoff]:
            _LOCAL_ALERT_BUCKET.pop(k, None)
    return True


async def record_operational_incident(
    pool,
    *,
    source: str,
    incident_type: str,
    subject: str,
    body: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    user_id: Any = None,
    upload_id: Any = None,
    screenshot_r2_key: Optional[str] = None,
    send_alerts: bool = True,
    alert_email: bool = True,
    alert_discord: bool = True,
    dedupe_seconds: Optional[int] = None,
    dedupe_key: Optional[str] = None,
    bypass_dedupe: bool = False,
) -> Optional[str]:
    """Insert ``operational_incidents`` row; optionally alert ops via email + Discord.

    Alerts are rate-limited / deduped: at most one alert per ``dedupe_key``
    (default ``f"{source}:{incident_type}"``) per ``dedupe_seconds`` window
    (default ``OPS_ALERT_DEDUPE_SECONDS`` env, 900s). The DB row is ALWAYS
    written regardless of dedupe so the admin incidents page sees every
    failure — only the noisy alerting channels are throttled.

    Returns the incident id (str) or None if the DB write failed.
    """
    uid = _parse_uuid(user_id)
    upid = _parse_uuid(upload_id)
    det = details if isinstance(details, dict) else {}
    src = (source or "unknown")[:50]
    itype = (incident_type or "unknown")[:120]
    subj = (subject or itype)[:2000]
    bod = (body or "")[:8000] if body else None
    shot = (screenshot_r2_key or "")[:512] or None

    new_id: Optional[uuid.UUID] = None
    try:
        from core.db_pool import acquire_db

        async with acquire_db(pool) as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO operational_incidents
                    (source, incident_type, user_id, upload_id, subject, body, details, screenshot_r2_key)
                VALUES ($1, $2, $3::uuid, $4::uuid, $5, $6, $7::jsonb, $8)
                RETURNING id
                """,
                src,
                itype,
                uid,
                upid,
                subj,
                bod,
                json.dumps(det, default=str),
                shot,
            )
            if row:
                new_id = row["id"]
    except Exception as e:
        logger.error("record_operational_incident DB failed: %s", e)
        return None

    if not send_alerts or not new_id:
        return str(new_id) if new_id else None

    # Dedupe alerts so a flood of identical failures (e.g. a downstream
    # provider outage producing 1000 api_500s in a minute) doesn't spam
    # Discord/email. The DB row above was already written either way.
    dkey = (dedupe_key or f"{src}:{itype}").lower()[:200]
    ttl = int(dedupe_seconds if dedupe_seconds is not None else DEFAULT_ALERT_DEDUPE_SECONDS)
    should_alert = True
    if not bypass_dedupe and ttl > 0:
        should_alert = await _claim_alert_slot(dkey, ttl)
    if not should_alert:
        logger.debug("ops alert deduped (key=%s ttl=%ss)", dkey, ttl)
        return str(new_id)

    email_ok = False
    discord_ok = False
    if alert_email and ADMIN_OPS_EMAIL:
        try:
            safe = json.dumps(det, indent=2, default=str)[:4000]
            html = (
                f"<h2>{subj}</h2><pre style='white-space:pre-wrap;font-size:12px'>{safe}</pre>"
                f"<p>Incident ID: {new_id}</p>"
                f"<p style='color:#888;font-size:11px'>Further '{itype}' alerts suppressed for {ttl}s.</p>"
            )
            await _send_ops_email(f"[UploadM8] {itype}", html)
            email_ok = True
        except Exception:
            pass
    if alert_discord and _discord_url():
        try:
            severity = str(det.get("severity") or ("critical" if "fail" in itype else "warning"))
            color = _severity_color(severity)
            fields = None
            footer = None
            if src == "worker_fleet_watchdog" or itype.startswith("render_event_"):
                fields = _watchdog_discord_fields(det, new_id)
                frontend = (os.environ.get("FRONTEND_URL") or "https://app.uploadm8.com").rstrip("/")
                footer = f"{frontend}/admin-incidents.html?incident_id={new_id}"
                await _send_discord_embed(
                    f"🛡 Fleet: {itype}",
                    f"**{subj}**\n{(bod or '')[:400]}",
                    color=color,
                    fields=fields,
                    footer=footer,
                )
            else:
                await _send_discord_embed(
                    f"Ops: {itype}",
                    f"**{subj}**\n```json\n{json.dumps(det, indent=2, default=str)[:1200]}\n```\n`{new_id}`\n"
                    f"_Further '{itype}' alerts suppressed for {ttl}s._",
                    color=color,
                )
            discord_ok = True
        except Exception:
            pass

    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE operational_incidents
                   SET email_sent_at = CASE WHEN $2 THEN NOW() ELSE email_sent_at END,
                       discord_sent_at = CASE WHEN $3 THEN NOW() ELSE discord_sent_at END
                 WHERE id = $1::uuid
                """,
                new_id,
                email_ok,
                discord_ok,
            )
    except Exception:
        pass

    return str(new_id)
