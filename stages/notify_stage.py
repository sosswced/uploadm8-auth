"""
UploadM8 Notification Stage
===========================
Send notifications via Discord webhooks and email.

Notifications:
- User webhook: Upload status (success/fail)
- Admin webhook: Signup, trial, MRR, errors
- Email: Welcome, upgrade confirmation, promotions
"""

import os
import json
import re
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple
from urllib.parse import urlparse
import httpx

from core.helpers import sanitize_hashtag_body

from .context import JobContext, PlatformResult
from . import db as db_stage
from .publish_stage import resolve_privacy_level

logger = logging.getLogger("uploadm8-worker")


def append_notification_delivery_record(ctx: JobContext, record: Dict[str, Any]) -> None:
    """Append one delivery row into ``ctx.output_artifacts['notification_delivery']``.

    Last 120 rows kept. Safe no-op when artifacts dict is missing.
    """
    arts = getattr(ctx, "output_artifacts", None)
    if not isinstance(arts, dict):
        return
    key = "notification_delivery"
    rows: List[Dict[str, Any]] = []
    raw = arts.get(key)
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                rows = parsed
        except Exception:
            rows = []
    row = dict(record)
    row.setdefault("ts", datetime.now(timezone.utc).isoformat())
    rows.append(row)
    rows = rows[-120:]
    try:
        arts[key] = json.dumps(rows, default=str)[:48000]
    except Exception:
        pass


async def append_notification_delivery_to_upload_db(
    db_pool_: Any,
    upload_id: str,
    record: Dict[str, Any],
) -> None:
    """Append a delivery diagnostic row onto ``uploads.output_artifacts['notification_delivery']``.

    Mirrors :func:`append_notification_delivery_record` but persists without ``JobContext``
    (e.g. background verify webhook / email paths).
    """
    if not db_pool_ or not upload_id:
        return

    row = dict(record)
    row.setdefault("ts", datetime.now(timezone.utc).isoformat())

    try:
        async with db_pool_.acquire() as conn:
            async with conn.transaction():
                rec = await conn.fetchrow(
                    "SELECT output_artifacts FROM uploads WHERE id = $1 FOR UPDATE",
                    str(upload_id),
                )
                if not rec:
                    return
                arts_raw = rec.get("output_artifacts")
                arts: Dict[str, Any] = (
                    dict(arts_raw)
                    if isinstance(arts_raw, dict)
                    else {}
                )

                rows: List[Any] = []
                key_nd = "notification_delivery"
                raw_nd = arts.get(key_nd)

                def _consume_parsed(parsed: Any) -> None:
                    nonlocal rows
                    if isinstance(parsed, list):
                        rows = parsed

                if isinstance(raw_nd, str) and raw_nd.strip():
                    try:
                        j = json.loads(raw_nd)
                        _consume_parsed(j)
                    except Exception:
                        rows = []
                elif isinstance(raw_nd, list):
                    rows = raw_nd

                rows.append(row)
                rows = rows[-120:]
                arts[key_nd] = json.dumps(rows, default=str)[:48000]

                await conn.execute(
                    """
                    UPDATE uploads
                       SET output_artifacts = $2::jsonb,
                           updated_at = NOW()
                     WHERE id = $1
                    """,
                    str(upload_id),
                    arts,
                )
    except ImportError:
        logger.debug(
            "append_notification_delivery_to_upload_db: asyncpg missing — skipping"
        )
    except Exception as ex:
        logger.debug(
            "append_notification_delivery_to_upload_db failed upload=%s: %s",
            upload_id,
            ex,
        )


async def user_wants_upload_email_notifications(db_pool_: Any, user_id_: Any) -> bool:
    """Per-user toggle for pipeline upload emails — default TRUE when prefs row missing."""

    if not db_pool_ or not user_id_:
        return True
    try:
        async with db_pool_.acquire() as _pconn:
            _prefs = await _pconn.fetchrow(
                "SELECT email_notifications FROM user_preferences WHERE user_id = $1",
                user_id_,
            )
    except Exception as _pe:
        logger.debug("user email-pref lookup failed (default TRUE): %s", _pe)
        return True
    if not _prefs:
        return True
    val = _prefs.get("email_notifications")
    return True if val is None else bool(val)


def _parse_notification_delivery_from_artifacts(arts: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = arts.get("notification_delivery")
    rows: List[Dict[str, Any]] = []
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                rows = [x for x in parsed if isinstance(x, dict)]
        except Exception:
            return []
    elif isinstance(raw, list):
        rows = [x for x in raw if isinstance(x, dict)]
    return rows


async def _record_terminal_comms_provider_trace(
    db_pool_: Any,
    ctx: JobContext,
    raw_status: str,
) -> None:
    """Append a synthetic ``provider_error_trace`` row summarizing user email/Discord outcomes.

    Correlates terminal pipeline status with the last delivery attempts persisted under
    ``output_artifacts['notification_delivery']`` (read from DB when available).
    """
    try:
        from services.provider_error_trace import append_provider_error
    except Exception:
        return

    arts: Dict[str, Any] = {}
    uid = str(getattr(ctx, "upload_id", "") or "")
    if db_pool_ and uid:
        try:
            async with db_pool_.acquire() as conn:
                rec = await conn.fetchrow(
                    "SELECT output_artifacts FROM uploads WHERE id = $1",
                    uid,
                )
            if rec and isinstance(rec.get("output_artifacts"), dict):
                arts = dict(rec["output_artifacts"])
        except Exception:
            arts = {}
    if not arts and isinstance(getattr(ctx, "output_artifacts", None), dict):
        arts = dict(ctx.output_artifacts or {})

    rows = _parse_notification_delivery_from_artifacts(arts)
    tail = rows[-48:] if rows else []
    email_rows = [r for r in tail if r.get("channel") == "user_upload_email"]
    discord_rows = [r for r in tail if r.get("channel") == "user_upload_discord"]
    last_email = email_rows[-1] if email_rows else None
    last_discord = discord_rows[-1] if discord_rows else None

    def _pick(d: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not d:
            return None
        return {
            "ok": d.get("ok"),
            "kind": d.get("kind"),
            "reason": d.get("reason"),
        }

    summary: Dict[str, Any] = {
        "terminal_status": raw_status,
        "last_user_upload_email": _pick(last_email),
        "last_user_upload_discord": _pick(last_discord),
        "email_delivery_rows": len(email_rows),
        "discord_delivery_rows": len(discord_rows),
    }

    tier = "ok"
    for d in (last_email, last_discord):
        if not d or d.get("ok") is not False:
            continue
        reason = str(d.get("reason") or "")
        if reason == "no_webhook_configured":
            continue
        tier = "degraded"

    append_provider_error(
        ctx,
        provider="uploadm8_terminal_comms",
        stage="notify_upload_terminal",
        operation="delivery_summary",
        message=json.dumps(summary, default=str)[:1300],
        result_tier=tier,
    )


async def _fetch_upload_costs_for_notify(db_pool_: Any, upload_id: Any) -> Tuple[int, int]:
    if not db_pool_:
        return 0, 0
    try:
        async with db_pool_.acquire() as conn:
            prow = await conn.fetchrow(
                "SELECT put_cost, aic_cost, put_reserved, aic_reserved FROM uploads WHERE id = $1",
                upload_id,
            )
        if not prow:
            return 0, 0
        put_c = prow["put_cost"] or prow["put_reserved"] or 0
        aic_c = prow["aic_cost"] or prow["aic_reserved"] or 0
        return int(put_c), int(aic_c)
    except Exception:
        return 0, 0


async def notify_upload_terminal(
    db_pool_: Any,
    ctx: JobContext,
    upload_id: str,
    *,
    status: str,
    scene_story: str = "",
) -> None:
    """Unified terminal upload comms: user email / Discord where applicable + admin paging.

    Accepted ``status``: succeeded / success / partial / failed / cancelled / degraded
    (``degraded`` is treated like ``partial`` for user-facing semantics).
    """
    raw_status = (status or "").strip().lower()
    st = raw_status
    if st not in (
        "succeeded",
        "success",
        "partial",
        "failed",
        "cancelled",
        "degraded",
        "staged",
    ):
        return
    if st == "success":
        st = "succeeded"
    elif st == "degraded":
        st = "partial"
    # ``staged`` = processing finished, publish deferred to scheduled_time (keep raw for branching).

    user_email = None
    user_name = "there"
    try:
        u = getattr(ctx, "user_record", None) or {}
        user_email = u.get("email") if u else None
        user_name = (u.get("name") if u else None) or "there"
    except Exception:
        pass

    wants_email = await user_wants_upload_email_notifications(
        db_pool_, str(ctx.user_id) if ctx.user_id else None
    )

    if user_email and wants_email:
        try:
            from stages.emails.uploads import (
                send_upload_completed_email,
                send_upload_failed_email,
                send_upload_staged_processing_email,
            )
            from stages.emails import build_upload_completed_email_extensions

            platforms = list(ctx.platforms or [])
            put_cost, aic_cost = await _fetch_upload_costs_for_notify(db_pool_, ctx.upload_id)

            if raw_status == "staged":
                from stages.emails.base import mailgun_ready

                if not mailgun_ready():
                    await append_notification_delivery_to_upload_db(
                        db_pool_,
                        str(upload_id),
                        {
                            "channel": "user_upload_email",
                            "ok": False,
                            "kind": "upload_staged_processing",
                            "reason": "mailgun_not_ready",
                        },
                    )
                else:
                    st_label = ""
                    try:
                        st_raw = getattr(ctx, "scheduled_time", None)
                        if st_raw is not None:
                            if hasattr(st_raw, "strftime"):
                                st_label = st_raw.strftime("%Y-%m-%d %H:%M UTC")
                            else:
                                st_label = str(st_raw)
                    except Exception:
                        st_label = ""
                    await send_upload_staged_processing_email(
                        user_email,
                        user_name,
                        ctx.filename or upload_id,
                        platforms,
                        str(upload_id),
                        scheduled_at_label=st_label or "your scheduled time",
                    )
                    await append_notification_delivery_to_upload_db(
                        db_pool_,
                        str(upload_id),
                        {"channel": "user_upload_email", "ok": True, "kind": "upload_staged_processing"},
                    )
            elif st in ("succeeded", "partial"):
                put_bal = aic_bal = None
                try:
                    async with db_pool_.acquire() as _wconn:
                        _wrow = await _wconn.fetchrow(
                            "SELECT put_balance, aic_balance FROM wallets WHERE user_id = $1::uuid",
                            ctx.user_id,
                        )
                    if _wrow:
                        put_bal = int(_wrow["put_balance"] or 0)
                        aic_bal = int(_wrow["aic_balance"] or 0)
                except Exception:
                    pass
                brand_ctx = None
                try:
                    from services.white_label import load_effective_brand_context

                    async with db_pool_.acquire() as _bconn:
                        brand_ctx = await load_effective_brand_context(_bconn, str(ctx.user_id))
                except Exception:
                    pass
                extras = build_upload_completed_email_extensions(
                    ctx, put_balance=put_bal, aic_balance=aic_bal
                )
                if scene_story:
                    extras["scene_story"] = scene_story[:1600]
                try:
                    from services.upload_notification_preview import (
                        resolve_upload_notification_preview_https_url,
                    )

                    preview_u = await resolve_upload_notification_preview_https_url(
                        db_pool_, ctx
                    )
                    if preview_u:
                        extras["preview_image_url"] = preview_u
                except Exception:
                    pass

                dedupe_kind = (
                    "upload_completed_ok"
                    if st == "succeeded"
                    else (
                        "upload_completed_partial_strict"
                        if raw_status == "degraded"
                        else "upload_completed_partial"
                    )
                )

                from stages.emails.base import mailgun_ready

                if not mailgun_ready():
                    await append_notification_delivery_to_upload_db(
                        db_pool_,
                        str(upload_id),
                        {
                            "channel": "user_upload_email",
                            "ok": False,
                            "kind": dedupe_kind,
                            "reason": "mailgun_not_ready",
                        },
                    )
                else:
                    dur = int(extras.pop("duration_seconds", 0) or 0)
                    if brand_ctx:
                        extras["brand"] = brand_ctx
                    await send_upload_completed_email(
                        user_email,
                        user_name,
                        ctx.filename or upload_id,
                        ctx.get_success_platforms() or platforms,
                        int(put_cost or 0),
                        int(aic_cost or 0),
                        str(upload_id),
                        dur,
                        **extras,
                    )
                    await append_notification_delivery_to_upload_db(
                        db_pool_,
                        str(upload_id),
                        {"channel": "user_upload_email", "ok": True, "kind": dedupe_kind},
                    )
            else:
                err_reason = getattr(ctx, "error_message", "") or ""
                err_stage = getattr(ctx, "current_stage", "") or ""
                if raw_status == "cancelled":
                    err_reason = err_reason or "Upload cancelled before publish."

                from stages.emails.base import mailgun_ready

                if not mailgun_ready():
                    await append_notification_delivery_to_upload_db(
                        db_pool_,
                        str(upload_id),
                        {
                            "channel": "user_upload_email",
                            "ok": False,
                            "kind": "upload_failed_email",
                            "reason": "mailgun_not_ready",
                        },
                    )
                else:
                    fail_brand = None
                    try:
                        from services.white_label import load_effective_brand_context

                        async with db_pool_.acquire() as _bconn:
                            fail_brand = await load_effective_brand_context(_bconn, str(ctx.user_id))
                    except Exception:
                        pass
                    await send_upload_failed_email(
                        user_email,
                        user_name,
                        ctx.filename or upload_id,
                        platforms,
                        err_reason,
                        str(upload_id),
                        err_stage,
                        scene_story=scene_story[:1600] if scene_story else "",
                        brand=fail_brand,
                    )
                    await append_notification_delivery_to_upload_db(
                        db_pool_,
                        str(upload_id),
                        {"channel": "user_upload_email", "ok": True, "kind": "upload_failed_email"},
                    )
        except Exception as _email_err:
            logger.warning(f"[{upload_id}] Upload email failed (non-fatal): {_email_err}")
            try:
                await append_notification_delivery_to_upload_db(
                    db_pool_,
                    str(upload_id),
                    {
                        "channel": "user_upload_email",
                        "ok": False,
                        "kind": "upload_email_exception",
                        "error": str(_email_err)[:500],
                    },
                )
            except Exception:
                pass

    # User Discord: early failures/cancel, deferred "staged" processing complete, or
    # publish path (run_notify_stage) for success/partial/fail.
    if raw_status == "staged":
        try:
            wh_sq = await _resolve_user_discord_webhook(ctx, db_pool=db_pool_)
            if not wh_sq:
                await append_notification_delivery_to_upload_db(
                    db_pool_,
                    str(upload_id),
                    {
                        "channel": "user_upload_discord",
                        "ok": False,
                        "kind": "upload_staged_discord",
                        "reason": "no_webhook_configured",
                    },
                )
            elif not _is_allowed_discord_webhook_url(wh_sq):
                await append_notification_delivery_to_upload_db(
                    db_pool_,
                    str(upload_id),
                    {
                        "channel": "user_upload_discord",
                        "ok": False,
                        "kind": "upload_staged_discord",
                        "reason": "invalid_webhook_url",
                    },
                )
            else:
                staged_discord_ok = False
                setattr(ctx, "_notify_terminal_staged", True)
                try:
                    staged_discord_ok = await send_user_upload_notification(wh_sq, ctx, db_pool=db_pool_)
                finally:
                    try:
                        delattr(ctx, "_notify_terminal_staged")
                    except Exception:
                        setattr(ctx, "_notify_terminal_staged", False)
                await append_notification_delivery_to_upload_db(
                    db_pool_,
                    str(upload_id),
                    {
                        "channel": "user_upload_discord",
                        "ok": bool(staged_discord_ok),
                        "kind": "upload_staged_discord",
                    },
                )
        except Exception as e:
            logger.debug(f"[{upload_id}] staged discord notify: {e}")
            try:
                await append_notification_delivery_to_upload_db(
                    db_pool_,
                    str(upload_id),
                    {
                        "channel": "user_upload_discord",
                        "ok": False,
                        "kind": "upload_staged_discord",
                        "error": str(e)[:400],
                    },
                )
            except Exception:
                pass
    # User Discord on early failures (publish path already runs notify_stage).
    elif st in ("failed", "cancelled") and not (ctx.platform_results or []):
        try:
            wh_quick = await _resolve_user_discord_webhook(ctx, db_pool=db_pool_)
            discord_early_ok = False
            prev_state = getattr(ctx, "state", None)
            try:
                if st == "cancelled":
                    setattr(ctx, "state", "cancelled")
                discord_early_ok = await run_notify_stage(ctx, db_pool_)
            finally:
                if prev_state is not None:
                    setattr(ctx, "state", prev_state)
            if not wh_quick:
                await append_notification_delivery_to_upload_db(
                    db_pool_,
                    str(upload_id),
                    {
                        "channel": "user_upload_discord",
                        "ok": False,
                        "kind": "early_terminal_run_notify_stage",
                        "reason": "no_webhook_configured",
                        "detail": st,
                    },
                )
            else:
                await append_notification_delivery_to_upload_db(
                    db_pool_,
                    str(upload_id),
                    {
                        "channel": "user_upload_discord",
                        "ok": bool(discord_early_ok),
                        "kind": "early_terminal_run_notify_stage",
                        "detail": st,
                    },
                )
        except Exception as e:
            logger.debug(f"[{upload_id}] notify_stage on failure path: {e}")
            try:
                await append_notification_delivery_to_upload_db(
                    db_pool_,
                    str(upload_id),
                    {
                        "channel": "user_upload_discord",
                        "ok": False,
                        "kind": "early_terminal_run_notify_stage",
                        "error": str(e)[:400],
                    },
                )
            except Exception:
                pass

    try:
        await _record_terminal_comms_provider_trace(db_pool_, ctx, raw_status)
    except Exception:
        logger.debug("[%s] terminal comms provider trace skipped", upload_id, exc_info=True)

    # Admin paging — partial / failed (canonical), plus explicit degraded semantics.
    if st in ("partial", "failed") or raw_status == "degraded":
        admin_st = raw_status if raw_status == "degraded" else st
        if admin_st not in ("partial", "failed"):
            admin_st = "partial"
        try:
            await notify_admin_upload_status(
                ctx,
                status=admin_st,
                upload_id=str(upload_id),
                db_pool=db_pool_,
                scene_story=scene_story or "",
            )
        except Exception as e:
            logger.warning(f"[{upload_id}] admin upload-status notify failed: {e}")


# Configuration
ADMIN_DISCORD_WEBHOOK_URL = os.environ.get("ADMIN_DISCORD_WEBHOOK_URL", "")
SIGNUP_DISCORD_WEBHOOK_URL = os.environ.get("SIGNUP_DISCORD_WEBHOOK_URL", "")
TRIAL_DISCORD_WEBHOOK_URL = os.environ.get("TRIAL_DISCORD_WEBHOOK_URL", "")
MRR_DISCORD_WEBHOOK_URL = os.environ.get("MRR_DISCORD_WEBHOOK_URL", "")
ERROR_DISCORD_WEBHOOK_URL = os.environ.get("ERROR_DISCORD_WEBHOOK_URL", "")

MAILGUN_API_KEY = os.environ.get("MAILGUN_API_KEY", "")
MAILGUN_DOMAIN = os.environ.get("MAILGUN_DOMAIN", "")
MAIL_FROM = os.environ.get("MAIL_FROM", "UploadM8 <no-reply@uploadm8.com>")
FRONTEND_URL = os.environ.get("FRONTEND_URL", "https://app.uploadm8.com")

_PLATFORM_LABELS = {
    "tiktok": "TikTok",
    "youtube": "YouTube",
    "instagram": "Instagram",
    "facebook": "Facebook",
}


def _platform_label(slug: str) -> str:
    k = (slug or "").lower()
    return _PLATFORM_LABELS.get(k, (slug or "Platform").title())


def _flatten_hashtag_raw(raw: Any) -> List[str]:
    """Turn stored hashtag payloads (list, JSON string, junk strings) into token strings without '#'."""
    if raw is None:
        return []
    if isinstance(raw, list):
        candidates: List[Any] = list(raw)
    elif isinstance(raw, str):
        s = raw.strip()
        if not s:
            return []
        if s.startswith("["):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    candidates = parsed
                else:
                    candidates = [s]
            except json.JSONDecodeError:
                candidates = [s]
        else:
            candidates = [s]
    else:
        candidates = [raw]

    out: List[str] = []
    for item in candidates:
        piece = str(item).strip()
        if not piece:
            continue
        # Pull word-like tokens from messy strings (e.g. #"[\"tester\" #"qwe"]")
        for m in re.finditer(r"#?([A-Za-z0-9_]{2,50})", piece):
            body = sanitize_hashtag_body(m.group(1))
            if body:
                out.append(body)
    # De-dupe preserving order
    seen: set = set()
    uniq: List[str] = []
    for b in out:
        if b.lower() in seen:
            continue
        seen.add(b.lower())
        uniq.append(b)
    return uniq


def _hashtags_for_discord_line(tokens: List[str]) -> str:
    return " ".join(f"#{t}" for t in tokens if t)


def _build_hashtags_by_platform_block(ctx: JobContext) -> str:
    """Effective caption hashtags per target platform (matches publish merge order)."""
    plat_sources = [r.platform for r in (ctx.platform_results or []) if r.success]
    if not plat_sources:
        plat_sources = list(ctx.platforms or [])
    seen: set = set()
    lines: List[str] = []
    for pl in plat_sources:
        key = (pl or "").lower()
        if not key or key in seen:
            continue
        seen.add(key)
        tags = ctx.get_effective_hashtags(key)
        if not tags:
            continue
        line = f"**{_platform_label(key)}:** {' '.join(tags)}"
        lines.append(line)
    return "\n".join(lines)


def _build_m8_ai_hashtags_block(ctx: JobContext) -> str:
    """Per-platform AI hashtag variants from M8 (when they differ from a single global list)."""
    m8 = getattr(ctx, "m8_platform_hashtags", None) or {}
    if not isinstance(m8, dict) or not m8:
        return ""
    lines: List[str] = []
    for pl in sorted(m8.keys()):
        raw = m8.get(pl) or []
        if not isinstance(raw, list) or not raw:
            continue
        flat = _flatten_hashtag_raw(raw)
        if not flat:
            continue
        lines.append(f"**{_platform_label(str(pl))}:** {_hashtags_for_discord_line(flat)}")
    return "\n".join(lines)


def _canonical_privacy(ctx: JobContext) -> str:
    p = (getattr(ctx, "privacy", None) or "public").strip().lower()
    if p not in ("public", "unlisted", "private"):
        return "public"
    return p


def _tiktok_status_lines(ctx: JobContext, result: PlatformResult) -> List[str]:
    lines: List[str] = []
    payload = result.response_payload or {}
    level = payload.get("tiktok_privacy_level") or resolve_privacy_level(_canonical_privacy(ctx), "tiktok")
    canon = (payload.get("upload_privacy") or _canonical_privacy(ctx) or "public").strip().lower()

    if not result.platform_url and not result.platform_video_id:
        lines.append(
            "TikTok is still processing this upload. If it is not on your profile yet, open the TikTok app and check **Inbox** or **Drafts**."
        )

    if level == "SELF_ONLY":
        if canon == "unlisted":
            lines.append(
                "You chose **unlisted** — TikTok received **Only you**. It may appear in **Drafts** or **Inbox** until you post publicly from the app; links may not work for others until then."
            )
        else:
            lines.append(
                "Posted as **Only you** on TikTok — check **Drafts** or **Inbox** if it is not on your profile yet. A share link may not work for others until you publish publicly."
            )
    elif level == "MUTUAL_FOLLOW_FRIENDS":
        lines.append(
            "Posted with **friends / mutual followers** visibility on TikTok — links may not work for everyone."
        )
    elif canon in ("unlisted", "private") and level == "PUBLIC_TO_EVERYONE":
        lines.append(
            f"Upload privacy was **{canon}** — confirm visibility in the TikTok app if the link behaves unexpectedly."
        )
    return lines


def _normalize_post_url(result: PlatformResult) -> Optional[str]:
    u = (result.platform_url or "").strip()
    if u.startswith("http"):
        plat = (result.platform or "").lower()
        if plat == "facebook" and "facebook.com/video/" in u and "/watch/" not in u:
            vid = getattr(result, "platform_video_id", None)
            if vid:
                return f"https://www.facebook.com/watch/?v={vid}"
        return u
    return None


def _fallback_post_url(result: PlatformResult) -> Optional[str]:
    vid = getattr(result, "platform_video_id", None)
    if not vid:
        return None
    plat = (result.platform or "").lower()
    if plat == "tiktok":
        handle = getattr(result, "account_username", None) or ""
        h = str(handle).strip().lstrip("@")
        if h:
            return f"https://www.tiktok.com/@{h}/video/{vid}"
        return f"https://www.tiktok.com/video/{vid}"
    if plat == "youtube":
        payload = getattr(result, "response_payload", None) or {}
        if isinstance(payload, dict) and payload.get("youtube_long_form_rights_guard"):
            return f"https://www.youtube.com/watch?v={vid}"
        return f"https://www.youtube.com/shorts/{vid}"
    if plat == "facebook":
        return f"https://www.facebook.com/watch/?v={vid}"
    return None


async def fetch_user_discord_webhook_from_db(db_pool, user_id: str) -> Optional[str]:
    """Resolve Discord webhook from DB (user_settings → user_preferences → users.preferences)."""
    if db_pool is None or not user_id:
        return None
    try:
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT COALESCE(
                  NULLIF(TRIM(us.discord_webhook), ''),
                  NULLIF(TRIM(up.discord_webhook), ''),
                  NULLIF(TRIM(COALESCE(
                    u.preferences->>'discordWebhook',
                    u.preferences->>'discord_webhook'
                  )), '')
                ) AS url
                FROM users u
                LEFT JOIN user_settings us ON us.user_id = u.id
                LEFT JOIN user_preferences up ON up.user_id = u.id
                WHERE u.id = $1
                """,
                user_id,
            )
        if row and row["url"]:
            url = str(row["url"]).strip()
            return url or None
    except Exception as e:
        logger.warning("Could not resolve user discord webhook from DB user=%s: %s", user_id, e)
    return None


async def _resolve_user_discord_webhook(ctx: JobContext, db_pool=None) -> Optional[str]:
    """Find the user's Discord webhook URL.

    Checks ``ctx.user_settings`` first (snake_case + camelCase), and falls back to a
    direct DB ``COALESCE`` across ``user_settings``, ``user_preferences``, and
    ``users.preferences`` JSONB when the in-memory copy is missing or stale. This
    mirrors the lookup used by ``POST /api/settings/test-discord-webhook`` so the
    pipeline stays consistent with what the user verified in the UI.
    """
    us = ctx.user_settings or {}
    raw = us.get("discord_webhook") or us.get("discordWebhook")
    if isinstance(raw, str):
        raw = raw.strip() or None
    if raw:
        return raw

    uid = getattr(ctx, "user_id", None)
    if not uid:
        return None
    return await fetch_user_discord_webhook_from_db(db_pool, str(uid))


def _is_allowed_discord_webhook_url(url: str) -> bool:
    """True for official Discord webhook URLs (including legacy discordapp.com and PTB/canary)."""
    try:
        u = urlparse((url or "").strip())
        if u.scheme != "https" or not u.hostname:
            return False
        host = u.hostname.lower()
        if host not in (
            "discord.com",
            "discordapp.com",
            "canary.discord.com",
            "ptb.discord.com",
        ):
            return False
        return (u.path or "").startswith("/api/webhooks/")
    except Exception:
        return False


def extract_pikzels_preview_image_url(data: Any) -> Optional[str]:
    """Best-effort HTTPS image URL from a Pikzels v2 JSON body (flat or nested ``data``)."""
    if not isinstance(data, dict):
        return None
    keys = ("output", "image_url", "url", "pikzels_cdn_url", "preview_url", "thumbnail_url")
    for k in keys:
        v = data.get(k)
        if isinstance(v, str) and v.startswith("https://"):
            return v.strip()[:2048]
    nested = data.get("data")
    if isinstance(nested, dict):
        for k in keys:
            v = nested.get(k)
            if isinstance(v, str) and v.startswith("https://"):
                return v.strip()[:2048]
    return None


def _pikzels_score_embed_description(response_data: Any, operation: str, upload_id: Optional[str]) -> Optional[str]:
    """Build markdown description for score/analyze; returns None if no score in body."""
    data = response_data if isinstance(response_data, dict) else {}
    nested = data.get("data") if isinstance(data.get("data"), dict) else {}
    score = data.get("main_score")
    if score is None:
        score = data.get("score")
    if score is None and nested:
        score = nested.get("main_score", nested.get("score"))
    if score is None:
        return None
    sugg_raw = data.get("suggestion") or nested.get("suggestion") or ""
    sugg = str(sugg_raw).strip()[:900]
    op_label = (operation or "score").strip()[:80]
    parts = [f"Pikzels **{op_label}**", "", f"**Score:** {score}"]
    if sugg:
        parts.extend(["", sugg])
    uid = (upload_id or "").strip()
    if uid:
        parts.extend(["", f"Upload id: `{uid}`"])
    return "\n".join(parts)[:1800]


async def notify_user_pikzels_generation(
    db_pool,
    user_id: str,
    *,
    operation: str,
    response_data: Any,
    upload_id: Optional[str] = None,
    source_image_url: Optional[str] = None,
) -> None:
    """
    When the user has configured a Discord webhook (same resolution as upload notify),
    send a compact embed. Image flows use ``image``; score flows add analysis text and,
    when available, the analyzed thumbnail (response CDN URL, or ``source_image_url``).
    """
    if not db_pool or not user_id:
        return
    wh = await fetch_user_discord_webhook_from_db(db_pool, user_id)
    if not wh or not _is_allowed_discord_webhook_url(wh):
        return

    op_low = (operation or "").lower()
    is_score = "score" in op_low

    img = extract_pikzels_preview_image_url(response_data)
    if not img and source_image_url:
        su = str(source_image_url).strip()
        if su.startswith("https://"):
            img = su[:2048]

    score_desc = _pikzels_score_embed_description(response_data, operation, upload_id) if is_score else None

    if not img and not score_desc:
        return

    embed: Dict[str, Any] = {
        "title": "📊 Pikzels analyze" if is_score else "🖼️ Pikzels output",
        "color": 0x10B981 if is_score else 0x5865F2,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "footer": {"text": "UploadM8 · Pikzels"},
    }

    if score_desc:
        embed["description"] = score_desc
    elif img:
        op = (operation or "pikzels").strip()[:80]
        desc_lines = [f"Pikzels **{op}** finished with a new image."]
        uid = (upload_id or "").strip()
        if uid:
            desc_lines.append(f"Upload / context id: `{uid}`")
        embed["description"] = "\n".join(desc_lines)[:1800]
    else:
        return

    if img:
        embed["image"] = {"url": img}

    ok_pz = bool(await _send_discord_webhook(wh, embeds=[embed]))
    if upload_id:
        try:
            await append_notification_delivery_to_upload_db(
                db_pool,
                str(upload_id),
                {
                    "channel": "user_upload_discord",
                    "ok": ok_pz,
                    "kind": "pikzels_preview",
                    "operation": (operation or "")[:120],
                },
            )
        except Exception:
            pass


async def run_notify_stage(ctx: JobContext, db_pool=None) -> bool:
    """
    Send notifications for completed upload.

    Process:
    1. Resolve user's Discord webhook (ctx.user_settings → DB fallback)
    2. Send embed with title / caption / hashtags / per-platform results
    3. Log explicit reason when no notification is sent

    Returns:
        True if a user Discord webhook was sent and returned 200/204; False if skipped or failed.
    """
    ctx.mark_stage("notify")

    user_webhook = await _resolve_user_discord_webhook(ctx, db_pool=db_pool)

    if not user_webhook:
        logger.info(
            f"[{ctx.upload_id}] notify: no user discord webhook configured — skipping user notification"
        )
        return False

    if not _is_allowed_discord_webhook_url(user_webhook):
        logger.warning(
            f"[{ctx.upload_id}] notify: user discord webhook is not a recognized Discord webhook URL — skipping"
        )
        return False

    logger.info(
        f"[{ctx.upload_id}] notify: sending user discord webhook "
        f"(success={ctx.is_success()} partial={ctx.is_partial_success()})"
    )
    return bool(await send_user_upload_notification(user_webhook, ctx, db_pool=db_pool))


async def send_user_upload_notification(webhook_url: str, ctx: JobContext, db_pool=None) -> bool:
    """Send upload status to user's Discord webhook.

    Embed includes:
      - Status (success / partial / failed / staged)
      - AI-generated title, caption, and hashtags
      - Per-platform result with clickable post links

    Returns True when the Discord POST succeeded (200/204); False on skip, HTTP error, or exception.
    """
    try:
        if getattr(ctx, "_notify_terminal_staged", False):
            color = 0x3b82f6
            status_title = "📅 Upload processed — scheduled publish"
            st_raw = getattr(ctx, "scheduled_time", None)
            st_lbl = ""
            try:
                if st_raw is not None and hasattr(st_raw, "strftime"):
                    st_lbl = st_raw.strftime("%Y-%m-%d %H:%M UTC")
                elif st_raw is not None:
                    st_lbl = str(st_raw)
            except Exception:
                st_lbl = ""
            status_desc = (
                "Your video finished processing and is queued for automatic publish "
                + (f"at **{st_lbl}** (UTC). " if st_lbl else "at your scheduled time. ")
                + "You can still edit or cancel from the queue before then."
            )
            video_title = (
                ctx.get_effective_title()
                if hasattr(ctx, "get_effective_title")
                else (getattr(ctx, "title", None) or ctx.filename or "Untitled")
            )
            embed: Dict[str, Any] = {
                "title": status_title,
                "description": status_desc,
                "color": color,
                "fields": [
                    {"name": "📹 Title", "value": str(video_title)[:256], "inline": False},
                    {"name": "📤 Platforms", "value": ", ".join(sorted(set(ctx.platforms or []))) or "—", "inline": True},
                    {"name": "📁 File", "value": (ctx.filename or "—")[:80], "inline": True},
                ],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "footer": {"text": "UploadM8"},
            }
            ok = await _send_discord_webhook(webhook_url, embeds=[embed])
            return bool(ok)

        cancelled = str(getattr(ctx, "state", "") or "").lower() == "cancelled"

        is_success = ctx.is_success() and not cancelled

        if cancelled:
            color = 0x6b7280
            status_title = "🛑 Upload Cancelled"
            status_desc = "This upload was cancelled before it finished publishing."
        elif is_success:
            color = 0x22c55e        # green
            status_title = "✅ Upload Completed"
            status_desc  = "Your video has been published successfully!"
        elif ctx.is_partial_success():
            color = 0xf97316        # orange
            status_title = "⚠️ Partial Upload"
            status_desc  = "Some platforms failed — check your queue for details."
        else:
            color = 0xef4444        # red
            status_title = "❌ Upload Failed"
            status_desc  = f"Upload failed: {ctx.error_message or 'Unknown error'}"

        # ── Content (same precedence as publish: explicit user copy, then hydrated AI) ─
        video_title = (
            ctx.get_effective_title()
            if hasattr(ctx, "get_effective_title")
            else (
                getattr(ctx, "title", None)
                or getattr(ctx, "ai_title", None)
                or ctx.filename
                or "Untitled"
            )
        )
        video_caption = (
            ctx.get_effective_caption()
            if hasattr(ctx, "get_effective_caption")
            else (getattr(ctx, "caption", None) or getattr(ctx, "ai_caption", None) or "")
        )
        raw_tags = getattr(ctx, "ai_hashtags", None)
        if raw_tags is None:
            raw_tags = getattr(ctx, "hashtags", None) or []
        flat_ai = _flatten_hashtag_raw(raw_tags)
        hashtag_str = _hashtags_for_discord_line(flat_ai)

        by_plat = _build_hashtags_by_platform_block(ctx)
        if not by_plat and hashtag_str:
            by_plat = ""

        # ── Build fields ─────────────────────────────────────────────────────
        fields: List[dict] = [
            {"name": "📹 Title",    "value": str(video_title)[:256],  "inline": False},
        ]

        if video_caption:
            cap_val = str(video_caption)
            fields.append({
                "name": "📝 Caption",
                "value": (cap_val[:500] + "…") if len(cap_val) > 500 else cap_val,
                "inline": False,
            })

        if by_plat:
            fields.append({
                "name": "🏷️ Hashtags (by platform)",
                "value": by_plat[:1020],
                "inline": False,
            })
        elif hashtag_str:
            fields.append({
                "name": "🏷️ Hashtags",
                "value": hashtag_str[:1020],
                "inline": False,
            })

        m8_block = _build_m8_ai_hashtags_block(ctx)
        if m8_block:
            fields.append({
                "name": "🧠 AI hashtag variants (M8)",
                "value": m8_block[:1020],
                "inline": False,
            })

        # Platforms summary (unique platforms; multi-account may repeat)
        plat_set = set(ctx.platforms) if ctx.platforms else set()
        fields.append({"name": "📤 Platforms", "value": ", ".join(sorted(plat_set)) or "None", "inline": True})
        fields.append({"name": "📁 File",      "value": (ctx.filename or "—")[:80],          "inline": True})

        preview_url = None
        try:
            from services.upload_notification_preview import (
                resolve_upload_notification_preview_https_url,
                thumbnail_quality_summary_text,
            )

            tq = thumbnail_quality_summary_text(ctx)
            if tq:
                fields.append({"name": "🖼️ Thumbnail", "value": tq[:1020], "inline": False})
            try:
                from services.uploads_handlers import pikzels_template_thumbnail_warning

                pw = pikzels_template_thumbnail_warning(getattr(ctx, "output_artifacts", None) or {})
                if pw:
                    hint = pw.get("message") or ""
                    if pw.get("skip_reason"):
                        hint += f"\nReason: {pw['skip_reason']}"
                    hint += "\nFix: Settings → Thumbnail Studio (enable auto-thumbnails + studio engine, pipeline Auto)."
                    fields.append(
                        {
                            "name": "⚠️ Thumbnail Studio",
                            "value": hint[:1020],
                            "inline": False,
                        }
                    )
            except Exception:
                pass
            preview_url = await resolve_upload_notification_preview_https_url(db_pool, ctx)
        except Exception as e:
            logger.debug("[%s] upload notify preview prep skipped: %s", getattr(ctx, "upload_id", ""), e)

        try:
            from stages.youtube_copyright_shorts import get_youtube_copyright_notice

            yt_note = get_youtube_copyright_notice(ctx)
            if yt_note and yt_note.get("message"):
                fields.append(
                    {
                        "name": "🎵 YouTube / music (ACR)",
                        "value": str(yt_note.get("message") or "")[:900],
                        "inline": False,
                    }
                )
        except Exception:
            pass

        # ── Scene Story (fused VI + Vision + OSD + telemetry + audio paragraph) ─
        try:
            scene_story_value = ""
            if isinstance(getattr(ctx, "output_artifacts", None), dict):
                scene_story_value = str(ctx.output_artifacts.get("scene_story") or "").strip()
            if scene_story_value:
                fields.append({
                    "name": "📖 Scene Story",
                    "value": (scene_story_value[:1017] + "…") if len(scene_story_value) > 1020 else scene_story_value,
                    "inline": False,
                })
        except Exception:
            pass

        # ── Per-platform/account results with live post URLs ────────────────────
        for result in ctx.platform_results:
            icon      = "✅" if result.success else "❌"
            plat_name = result.platform.title()
            account_label = (
                getattr(result, "account_username", None)
                or getattr(result, "account_name", None)
                or getattr(result, "account_id", None)
            )
            if account_label:
                field_name = f"{icon} {plat_name} ({account_label})"
            else:
                field_name = f"{icon} {plat_name}"

            if result.success:
                url = _normalize_post_url(result) or _fallback_post_url(result)
                chunks: List[str] = []
                if url:
                    chunks.append(f"[View post]({url})")
                elif result.publish_id:
                    chunks.append(f"Accepted — publish_id: `{result.publish_id}`")
                else:
                    chunks.append("Published ✓")

                plat_lc = (result.platform or "").lower()
                if plat_lc == "tiktok":
                    for line in _tiktok_status_lines(ctx, result):
                        if line:
                            chunks.append(line)

                if plat_lc == "instagram" and not url:
                    chunks.append(
                        "Instagram link not available yet — open Instagram or wait for queue sync to refresh the permalink."
                    )

                value = "\n".join(chunks)[:1020]
            else:
                raw_err = result.error_message or result.error_code or "Unknown error"
                value = str(raw_err)[:256]

            fields.append({"name": field_name, "value": value, "inline": False})

        embed: Dict[str, Any] = {
            "title":       status_title,
            "description": status_desc,
            "color":       color,
            "fields":      fields,
            "timestamp":   datetime.now(timezone.utc).isoformat(),
            "footer":      {"text": "UploadM8"},
        }
        if preview_url:
            embed["image"] = {"url": preview_url[:2048]}

        ok_discord = await _send_discord_webhook(webhook_url, embeds=[embed])
        append_notification_delivery_record(
            ctx,
            {
                "channel": "user_upload_discord",
                "ok": bool(ok_discord),
                "kind": "upload_status",
                "upload_id": str(getattr(ctx, "upload_id", "") or ""),
            },
        )
        return bool(ok_discord)

    except Exception as e:
        logger.exception(f"[{ctx.upload_id}] User webhook notification failed: {e}")
        return False


async def notify_user_publish_rejected(
    db_pool,
    *,
    user_id: str,
    upload_id: str,
    platform: str,
    detail: str = "",
    verify_outcome: str = "rejected",
) -> None:
    """Notify user when verification finds the platform rejected/removed the post or failed to confirm."""
    if not db_pool or not user_id or not upload_id:
        return
    plat_key = (platform or "").strip().lower()
    if plat_key not in ("tiktok", "youtube"):
        return

    outcome = (verify_outcome or "rejected").strip().lower()
    is_failed = outcome == "failed"
    discord_kind = "publish_verify_failed" if is_failed else "publish_verify_rejected"
    email_kind = "publish_verify_failed_email" if is_failed else "publish_verify_rejected_email"
    filename = ""
    email = ""
    name = "there"
    wants_email = True
    try:
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT u.email AS email,
                       COALESCE(NULLIF(TRIM(u.name), ''), 'there') AS display_name,
                       ups.filename AS filename,
                       up.email_notifications AS email_notifications
                FROM uploads ups
                JOIN users u ON u.id = ups.user_id
                LEFT JOIN user_preferences up ON up.user_id = u.id
                WHERE ups.id = $1 AND ups.user_id = $2::uuid
                """,
                upload_id,
                user_id,
            )
            if row:
                email = str(row["email"] or "").strip()
                name = str(row["display_name"] or "there")
                fn = row.get("filename")
                filename = str(fn).strip() if fn else ""
                pref = row.get("email_notifications")
                wants_email = True if pref is None else bool(pref)
    except Exception as e:
        logger.warning("publish_rejected: lookup failed upload=%s: %s", upload_id, e)

    plat_label = _platform_label(plat_key)
    reason = (detail or "").strip() or (
        (
            f"{plat_label} verification could not confirm this post is live "
            "(API error, timeout, or missing publish)."
        )
        if is_failed
        else (
            f"{plat_label} verification reported this upload as rejected, removed, "
            "or not publicly accessible."
        )
    )

    embed_title = (
        f"⚠️ {plat_label}: verification could not confirm post"
        if is_failed
        else f"⚠️ {plat_label}: post rejected or removed"
    )

    wh = await fetch_user_discord_webhook_from_db(db_pool, user_id)
    discord_ok = False
    if wh and _is_allowed_discord_webhook_url(wh):
        try:
            discord_ok = bool(
                await _send_discord_webhook(
                    wh,
                    embeds=[
                        {
                            "title": embed_title,
                            "description": (
                                f"{(filename or 'Your video')} — verification failed.\n\n"
                                f"{reason[:900]}"
                            ),
                            "color": 0xef4444,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "footer": {"text": "UploadM8 · platform verification"},
                        }
                    ],
                )
            )
        except Exception as e:
            logger.warning("publish_rejected Discord failed upload=%s: %s", upload_id, e)
            discord_ok = False
    try:
        await append_notification_delivery_to_upload_db(
            db_pool,
            str(upload_id),
            {
                "channel": "user_upload_discord",
                "ok": bool(discord_ok),
                "kind": discord_kind,
                "platform": plat_key,
            },
        )
    except Exception:
        pass

    if email and wants_email:
        email_ok = False
        try:
            from stages.emails.base import mailgun_ready
            from stages.emails.uploads import send_upload_failed_email

            if mailgun_ready():
                await send_upload_failed_email(
                    email,
                    name,
                    filename or upload_id,
                    [plat_key],
                    reason,
                    str(upload_id),
                    "platform_verify",
                    "",
                )
                email_ok = True
        except Exception as e:
            logger.warning("publish_rejected email failed upload=%s: %s", upload_id, e)
        try:
            await append_notification_delivery_to_upload_db(
                db_pool,
                str(upload_id),
                {
                    "channel": "user_upload_email",
                    "ok": bool(email_ok),
                    "kind": email_kind,
                    "platform": plat_key,
                },
            )
        except Exception:
            pass


async def notify_user_publish_confirmed(
    db_pool,
    *,
    user_id: str,
    upload_id: str,
    platform: str,
    post_url: str = "",
) -> None:
    """
    Step-B confirmation: platform APIs report the post is fully live (e.g. TikTok
    PUBLISH_COMPLETE with video_id). Discord webhook ping only (no email).
    """
    if not db_pool or not user_id or not upload_id:
        return
    plat_key = (platform or "").strip().lower()
    if plat_key not in ("tiktok", "youtube"):
        return

    filename = ""
    try:
        async with db_pool.acquire() as conn:
            fn = await conn.fetchval("SELECT filename FROM uploads WHERE id = $1", upload_id)
            if fn:
                filename = str(fn)
    except Exception as e:
        logger.warning("publish_confirmed: could not load filename upload=%s: %s", upload_id, e)

    plat_label = _platform_label(plat_key)
    wh = await fetch_user_discord_webhook_from_db(db_pool, user_id)
    discord_ok = False
    if wh and _is_allowed_discord_webhook_url(wh):
        desc_parts = [
            (filename or "Your video") + f" is confirmed live on **{plat_label}**.",
        ]
        if (post_url or "").strip().startswith("http"):
            desc_parts.append(f"[View post]({post_url.strip()})")
        try:
            discord_ok = bool(
                await _send_discord_webhook(
                    wh,
                    embeds=[
                        {
                            "title": f"✅ {plat_label} confirmed",
                            "description": "\n\n".join(desc_parts)[:1800],
                            "color": 0x22c55e,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "footer": {"text": "UploadM8 · publish confirmation"},
                        }
                    ],
                )
            )
        except Exception as e:
            logger.warning("publish_confirmed Discord failed upload=%s: %s", upload_id, e)
            discord_ok = False
    try:
        await append_notification_delivery_to_upload_db(
            db_pool,
            str(upload_id),
            {
                "channel": "user_upload_discord",
                "ok": bool(discord_ok),
                "kind": "publish_verify_confirmed",
                "platform": plat_key,
            },
        )
    except Exception:
        pass


# ============================================================
# Admin Notifications
# ============================================================

async def _get_admin_webhook(db_pool=None) -> str:
    # Priority: explicit env override -> admin_settings saved webhook
    if ADMIN_DISCORD_WEBHOOK_URL:
        return ADMIN_DISCORD_WEBHOOK_URL
    if db_pool is None:
        return ""
    try:
        wh = await db_stage.load_admin_notification_webhook(db_pool)
        return wh or ""
    except Exception:
        return ""


async def notify_admin_signup(email: str, name: str, tier: str = "free", db_pool=None):
    """Notify admin of new user signup."""
    webhook = SIGNUP_DISCORD_WEBHOOK_URL or (await _get_admin_webhook(db_pool))
    if not webhook:
        return
    
    embed = {
        "title": "🎉 New User Signup",
        "color": 0x3b82f6,
        "fields": [
            {"name": "Email", "value": email, "inline": True},
            {"name": "Name", "value": name, "inline": True},
            {"name": "Tier", "value": tier, "inline": True},
        ],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    
    await _send_discord_webhook(webhook, embeds=[embed])


async def notify_admin_trial_started(email: str, name: str, plan: str, db_pool=None):
    """Notify admin of new trial signup."""
    webhook = TRIAL_DISCORD_WEBHOOK_URL or (await _get_admin_webhook(db_pool))
    if not webhook:
        return
    
    embed = {
        "title": "🚀 Trial Started",
        "color": 0x8b5cf6,
        "fields": [
            {"name": "Email", "value": email, "inline": True},
            {"name": "Name", "value": name, "inline": True},
            {"name": "Plan", "value": plan, "inline": True},
        ],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    
    await _send_discord_webhook(webhook, embeds=[embed])


async def notify_admin_mrr_collected(amount: float, customer_email: str, plan: str, db_pool=None):
    """Notify admin of MRR collection."""
    webhook = MRR_DISCORD_WEBHOOK_URL or (await _get_admin_webhook(db_pool))
    if not webhook:
        return
    
    embed = {
        "title": "💰 MRR Collected",
        "color": 0x22c55e,
        "fields": [
            {"name": "Amount", "value": f"${amount:.2f}", "inline": True},
            {"name": "Customer", "value": customer_email, "inline": True},
            {"name": "Plan", "value": plan, "inline": True},
        ],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    
    await _send_discord_webhook(webhook, embeds=[embed])


async def notify_admin_upload_status(
    ctx: "JobContext",
    *,
    status: str,
    upload_id: str,
    db_pool=None,
    scene_story: str = "",
) -> None:
    """Per-upload admin notification for partial / failed outcomes.

    Always fires admin Discord embed AND admin ops email (ADMIN_OPS_EMAIL via
    services.ops_incidents.record_operational_incident) regardless of user prefs
    so the operator sees every partial/failed pipeline. Successes do NOT page admin.

    The DB row is always written (see ops_incidents) so the admin-incidents page
    sees every status; alerting channels dedupe within OPS_ALERT_DEDUPE_SECONDS.
    """
    st = (status or "").strip().lower()
    if st not in ("partial", "failed"):
        return  # success does not page admin

    if not db_pool:
        return

    try:
        from services.ops_incidents import record_operational_incident
    except Exception as e:
        logger.debug("notify_admin_upload_status: ops_incidents import failed: %s", e)
        return

    succeeded = []
    failed = []
    try:
        succeeded = list(ctx.get_success_platforms() or [])
        failed = list(ctx.get_failed_platforms() or [])
    except Exception:
        pass

    details: Dict[str, Any] = {
        "upload_id": str(upload_id) if upload_id else None,
        "user_id": str(getattr(ctx, "user_id", "") or "") or None,
        "filename": getattr(ctx, "filename", None),
        "platforms": list(getattr(ctx, "platforms", None) or []),
        "succeeded_platforms": succeeded,
        "failed_platforms": failed,
        "error_code": getattr(ctx, "error_code", None),
        "error_message": (getattr(ctx, "error_message", "") or "")[:1000],
        "current_stage": getattr(ctx, "current_stage", None),
        "ai_title": str(getattr(ctx, "ai_title", "") or "")[:240],
    }
    viol = list(getattr(ctx, "google_multimodal_strict_violations", None) or [])
    if viol:
        details["google_multimodal_strict_violations"] = viol[:30]

    arts_gap = getattr(ctx, "output_artifacts", None)
    if isinstance(arts_gap, dict):
        _gaps_raw = arts_gap.get("google_multimodal_gaps")
        if _gaps_raw:
            details["google_multimodal_gaps"] = str(_gaps_raw)[:2000]
        mq_art = arts_gap.get("metadata_quality_report")
        if mq_art:
            details["metadata_quality_report_snippet"] = str(mq_art)[:2000]

    mq_ctx = getattr(ctx, "metadata_quality_violations", None) or []
    if mq_ctx:
        details["metadata_quality_violations"] = mq_ctx[:40]

    if scene_story:
        details["scene_story"] = scene_story[:600]

    plat_lines: List[str] = []
    for r in getattr(ctx, "platform_results", None) or []:
        try:
            icon = "OK" if r.success else "FAIL"
            label = r.platform
            if getattr(r, "account_username", None):
                label = f"{label} (@{r.account_username})"
            reason = ""
            if not r.success:
                reason = f": {r.error_code or 'UNKNOWN'} — {(r.error_message or '')[:200]}"
            plat_lines.append(f"{icon} {label}{reason}")
        except Exception:
            continue

    body = (
        f"Filename: {details['filename'] or '—'}\n"
        f"Platforms requested: {', '.join(details['platforms']) or '—'}\n"
        f"Succeeded: {', '.join(succeeded) or '—'}\n"
        f"Failed: {', '.join(failed) or '—'}\n"
        f"Error: {details['error_code'] or '—'} — {details['error_message'] or '—'}\n"
        f"Stage: {details['current_stage'] or '—'}\n"
        f"Title: {details['ai_title'] or '—'}\n"
        + ("Scene story: " + scene_story[:600] + "\n" if scene_story else "")
        + ("\nPer-platform:\n" + "\n".join(plat_lines) if plat_lines else "")
    )[:8000]

    try:
        await record_operational_incident(
            db_pool,
            source="upload",
            incident_type=f"upload_{st}",
            subject=f"Upload {st}: {details['filename'] or upload_id}",
            body=body,
            details=details,
            user_id=details.get("user_id"),
            upload_id=details.get("upload_id"),
            alert_email=True,
            alert_discord=True,
        )
    except Exception as e:
        logger.warning("notify_admin_upload_status record failed: %s", e)


async def notify_admin_error(error_type: str, details: dict, db_pool=None):
    """Notify admin of system error — DB incident row + email, then Discord embed."""
    if db_pool:
        try:
            from services.ops_incidents import record_operational_incident

            await record_operational_incident(
                db_pool,
                source="worker",
                incident_type=str(error_type)[:120],
                subject=f"Worker: {error_type}",
                body=str(details.get("error") or details.get("message") or "")[:8000],
                details=dict(details) if isinstance(details, dict) else {"raw": str(details)},
                user_id=details.get("user_id"),
                upload_id=details.get("upload_id"),
                alert_discord=False,
            )
        except Exception as ex:
            logger.warning("notify_admin_error incident log failed: %s", ex)

    webhook = ERROR_DISCORD_WEBHOOK_URL or (await _get_admin_webhook(db_pool))
    if not webhook:
        return

    embed = {
        "title": f"🚨 Error: {error_type}",
        "color": 0xEF4444,
        "description": f"```json\n{json.dumps(details, indent=2, default=str)[:1500]}\n```",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    await _send_discord_webhook(webhook, embeds=[embed])


async def notify_admin_worker_start(db_pool=None):
    """Notify admin that worker has started."""
    webhook = await _get_admin_webhook(db_pool)
    if not webhook:
        return
    
    await _send_discord_webhook(
        webhook,
        content="🟢 UploadM8 Worker started"
    )


async def notify_admin_worker_stop(db_pool=None):
    """Notify admin that worker has stopped."""
    webhook = await _get_admin_webhook(db_pool)
    if not webhook:
        return
    
    await _send_discord_webhook(
        webhook,
        content="🔴 UploadM8 Worker stopped"
    )


async def notify_admin_daily_summary(data: dict, db_pool=None):
    """Send daily summary to admin."""
    webhook = await _get_admin_webhook(db_pool)
    if not webhook:
        return
    
    embed = {
        "title": "📊 Daily Summary",
        "color": 0x3b82f6,
        "fields": [
            {"name": "New Users", "value": str(data.get("new_users", 0)), "inline": True},
            {"name": "Uploads", "value": str(data.get("uploads", 0)), "inline": True},
            {"name": "MRR", "value": f"${data.get('mrr', 0):.2f}", "inline": True},
            {"name": "OpenAI Cost", "value": f"${data.get('openai_cost', 0):.2f}", "inline": True},
            {"name": "Storage Used", "value": f"{data.get('storage_gb', 0):.2f} GB", "inline": True},
            {"name": "Active Users", "value": str(data.get("active_users", 0)), "inline": True},
        ],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    
    await _send_discord_webhook(webhook, embeds=[embed])


# ============================================================
# Email Notifications
# ============================================================

async def send_welcome_email(email: str, name: str):
    """Send welcome email to new user."""
    if not MAILGUN_API_KEY or not MAILGUN_DOMAIN:
        logger.info(f"Welcome email skipped (no Mailgun): {email}")
        return
    
    subject = "Welcome to UploadM8! 🎉"
    html = f"""
    <h1>Welcome to UploadM8, {name}!</h1>
    <p>Thanks for signing up. You're now ready to upload videos to multiple platforms with a single click.</p>
    <h2>Getting Started:</h2>
    <ol>
        <li>Connect your social media accounts (TikTok, YouTube, Instagram, Facebook)</li>
        <li>Upload your first video</li>
        <li>Let our AI generate titles, captions, and hashtags</li>
        <li>Publish to all platforms at once!</li>
    </ol>
    <p><a href="{FRONTEND_URL}/dashboard.html" style="background: #f97316; color: white; padding: 12px 24px; text-decoration: none; border-radius: 8px;">Go to Dashboard</a></p>
    <p>If you have any questions, just reply to this email.</p>
    <p>- The UploadM8 Team</p>
    """
    
    await _send_mailgun_email(email, subject, html)


async def send_upgrade_email(email: str, name: str, new_tier: str):
    """Send upgrade confirmation email."""
    if not MAILGUN_API_KEY or not MAILGUN_DOMAIN:
        return
    
    subject = f"Welcome to {new_tier.title()}! 🚀"
    html = f"""
    <h1>Upgrade Confirmed!</h1>
    <p>Hi {name},</p>
    <p>Your account has been upgraded to <strong>{new_tier.title()}</strong>!</p>
    <p>You now have access to:</p>
    <ul>
        <li>Higher upload limits</li>
        <li>AI-powered captions and thumbnails</li>
        <li>Smart scheduling</li>
        <li>And more!</li>
    </ul>
    <p><a href="{FRONTEND_URL}/dashboard.html">Go to Dashboard</a></p>
    <p>- The UploadM8 Team</p>
    """
    
    await _send_mailgun_email(email, subject, html)


async def send_tier_change_email(email: str, name: str, old_tier: str, new_tier: str, is_upgrade: bool):
    """Send tier change notification email."""
    if not MAILGUN_API_KEY or not MAILGUN_DOMAIN:
        return
    
    if is_upgrade:
        subject = f"🎉 Upgraded to {new_tier.title()}"
        message = f"Your account has been upgraded from {old_tier.title()} to {new_tier.title()}!"
    else:
        subject = f"Plan Changed to {new_tier.title()}"
        message = f"Your account has been changed from {old_tier.title()} to {new_tier.title()}."
    
    html = f"""
    <h1>Plan Changed</h1>
    <p>Hi {name},</p>
    <p>{message}</p>
    <p><a href="{FRONTEND_URL}/settings.html">View your account settings</a></p>
    <p>- The UploadM8 Team</p>
    """
    
    await _send_mailgun_email(email, subject, html)


# ============================================================
# Internal Helpers
# ============================================================

async def _send_discord_webhook(webhook_url: str, content: str = None, embeds: List[dict] = None) -> bool:
    """POST to a Discord webhook. Returns True on 200/204, False otherwise."""
    if not webhook_url:
        return False

    payload = {}
    if content:
        payload["content"] = content
    if embeds:
        payload["embeds"] = embeds

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(webhook_url, json=payload)
            if response.status_code not in (200, 204):
                body = (response.text or "")[:500]
                logger.warning(
                    f"Discord webhook failed: HTTP {response.status_code} body={body}"
                )
                return False
            return True
    except Exception as e:
        logger.warning(f"Discord webhook error: {e}")
        return False


async def _send_mailgun_email(to: str, subject: str, html: str):
    """Send email via Mailgun API."""
    if not MAILGUN_API_KEY or not MAILGUN_DOMAIN:
        return
    
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"https://api.mailgun.net/v3/{MAILGUN_DOMAIN}/messages",
                auth=("api", MAILGUN_API_KEY),
                data={
                    "from": MAIL_FROM,
                    "to": to,
                    "subject": subject,
                    "html": html,
                }
            )
            
            if response.status_code != 200:
                logger.warning(f"Mailgun send failed: {response.status_code}")
            else:
                logger.info(f"Email sent to {to}: {subject}")
                
    except Exception as e:
        logger.warning(f"Mailgun error: {e}")
