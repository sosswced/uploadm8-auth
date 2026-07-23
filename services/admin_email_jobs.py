"""
Admin email job runners.

Each job is a coroutine that takes ``db_pool`` and returns a summary dict::

    {
        "job":           "trial_reminders",
        "sent":          12,
        "skipped":       3,
        "errors":        0,
        "details":       {...},  # job-specific extra info
    }

Jobs are designed to be IDEMPOTENT. They claim rows (UPDATE … WHERE marker IS NULL
RETURNING) before sending, and only keep the claim when Mailgun accepts the
message — so concurrent runs and Mailgun-down skips do not re-spam or burn
idempotency windows.

Triggers:
  - ``POST /api/admin/email-jobs/run``  (manual, master admin UI)
  - ``run_admin_email_jobs_loop()``     (worker-side daily cron)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import asyncpg

from stages.emails import (
    send_trial_ending_reminder_email,
    send_monthly_user_kpi_digest_email,
    send_admin_weekly_kpi_digest_email,
    send_scheduled_publish_alert_email,
)

logger = logging.getLogger("uploadm8-admin-jobs")

# Worker cron: run the four email jobs once per day (default 24h).
ADMIN_EMAIL_JOBS_INTERVAL_SEC = max(
    3600, int(os.environ.get("ADMIN_EMAIL_JOBS_INTERVAL_SEC", str(24 * 3600)))
)
WEEKLY_ADMIN_DIGEST_MIN_INTERVAL_DAYS = 6


# ─────────────────────────────────────────────────────────────────────────────
# Run ledger helpers (admin_email_job_runs table — migration 1050 / 1090)
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_triggered_by(triggered_by: str) -> str:
    raw = (triggered_by or "manual").strip() or "manual"
    return raw[:255]


async def _start_run(pool: asyncpg.Pool, job: str, triggered_by: str) -> Optional[str]:
    try:
        async with pool.acquire() as conn:
            run_id = await conn.fetchval(
                """
                INSERT INTO admin_email_job_runs (job, triggered_by)
                VALUES ($1, $2)
                RETURNING id::text
                """,
                job,
                _normalize_triggered_by(triggered_by),
            )
        return run_id
    except Exception as e:
        logger.warning("admin_email_job_runs insert failed for %s: %s", job, e)
        return None


async def _finish_run(
    pool: asyncpg.Pool,
    run_id: Optional[str],
    sent: int,
    skipped: int,
    errors: int,
    summary: Dict[str, Any],
    error_message: Optional[str] = None,
) -> None:
    if not run_id:
        return
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE admin_email_job_runs
                SET finished_at   = NOW(),
                    sent_count    = $2,
                    skipped_count = $3,
                    error_count   = $4,
                    summary       = $5::jsonb,
                    error_message = $6
                WHERE id = $1::uuid
                """,
                run_id,
                int(sent),
                int(skipped),
                int(errors),
                json.dumps(summary, default=str),
                (error_message or None),
            )
    except Exception as e:
        logger.warning("admin_email_job_runs update failed for run=%s: %s", run_id, e)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Trial reminders  (3 days before trial_end)
# ─────────────────────────────────────────────────────────────────────────────

TRIAL_REMINDER_WINDOW_DAYS = 4


async def run_trial_reminders(
    pool: asyncpg.Pool, *, triggered_by: str = "manual"
) -> Dict[str, Any]:
    """Email every trialing user whose trial_end is within the next 4 days
    and who hasn't already received a reminder.

    Claims ``users.trial_reminder_sent`` before send; rolls back the claim if
    Mailgun does not accept the message.
    """
    job = "trial_reminders"
    run_id = await _start_run(pool, job, triggered_by)
    sent = skipped = errors = 0
    sent_emails: List[str] = []

    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT id, email, name, subscription_tier, trial_end
                FROM users
                WHERE subscription_status = 'trialing'
                  AND trial_end IS NOT NULL
                  AND trial_end > NOW()
                  AND trial_end <= NOW() + INTERVAL '{TRIAL_REMINDER_WINDOW_DAYS} days'
                  AND trial_reminder_sent IS NULL
                  AND status = 'active'
                  AND email IS NOT NULL
                  AND email <> ''
                """
            )

        for r in rows:
            email = r["email"]
            try:
                async with pool.acquire() as conn:
                    claimed = await conn.fetchval(
                        """
                        UPDATE users
                        SET trial_reminder_sent = NOW()
                        WHERE id = $1 AND trial_reminder_sent IS NULL
                        RETURNING id
                        """,
                        r["id"],
                    )
                if not claimed:
                    skipped += 1
                    continue

                trial_end_dt = r["trial_end"]
                if trial_end_dt.tzinfo is None:
                    trial_end_dt = trial_end_dt.replace(tzinfo=timezone.utc)
                days_left = max(1, (trial_end_dt - datetime.now(timezone.utc)).days)
                ok = await send_trial_ending_reminder_email(
                    email=email,
                    name=r["name"] or "there",
                    tier=r["subscription_tier"] or "creator_pro",
                    trial_end_date=trial_end_dt.strftime("%B %d, %Y"),
                    days_left=days_left,
                )
                if not ok:
                    async with pool.acquire() as conn:
                        await conn.execute(
                            "UPDATE users SET trial_reminder_sent = NULL WHERE id = $1",
                            r["id"],
                        )
                    skipped += 1
                    continue
                sent += 1
                sent_emails.append(email)
            except Exception as e:
                logger.warning("trial_reminder failed for %s: %s", email, e)
                try:
                    async with pool.acquire() as conn:
                        await conn.execute(
                            "UPDATE users SET trial_reminder_sent = NULL WHERE id = $1",
                            r["id"],
                        )
                except Exception:
                    pass
                errors += 1

        summary = {
            "queried":     len(rows),
            "sample_recipients": sent_emails[:10],
            "window_days": TRIAL_REMINDER_WINDOW_DAYS,
        }
        await _finish_run(pool, run_id, sent, skipped, errors, summary)
        return {"job": job, "sent": sent, "skipped": skipped, "errors": errors, "details": summary}
    except Exception as e:
        logger.exception("run_trial_reminders failed: %s", e)
        await _finish_run(pool, run_id, sent, skipped, errors, {}, str(e))
        return {"job": job, "sent": sent, "skipped": skipped, "errors": errors + 1, "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# 2. Monthly per-user KPI digest (last 30 days)
# ─────────────────────────────────────────────────────────────────────────────

MONTHLY_DIGEST_MIN_INTERVAL_DAYS = 25  # never send twice within ~a month


async def _user_monthly_metrics(conn: asyncpg.Connection, user_id) -> Dict[str, int]:
    """Pull last-30-day upload + engagement metrics for one user."""
    upload_row = await conn.fetchrow(
        """
        SELECT
          COUNT(*)                               AS uploads,
          COUNT(*) FILTER (WHERE status IN ('succeeded','completed','partial'))
                                                 AS completed,
          COALESCE(SUM(put_spent), 0)            AS put_used,
          COALESCE(SUM(aic_spent), 0)            AS aic_used
        FROM uploads
        WHERE user_id = $1
          AND created_at >= NOW() - INTERVAL '30 days'
        """,
        user_id,
    )

    engagement_row = await conn.fetchrow(
        """
        SELECT
          COALESCE(SUM(views),    0) AS views,
          COALESCE(SUM(likes),    0) AS likes,
          COALESCE(SUM(comments), 0) AS comments,
          COALESCE(SUM(shares),   0) AS shares
        FROM platform_content_items
        WHERE user_id = $1
          AND COALESCE(published_at, updated_at) >= NOW() - INTERVAL '30 days'
        """,
        user_id,
    )

    wallet_row = await conn.fetchrow(
        "SELECT put_balance, aic_balance FROM wallets WHERE user_id = $1",
        user_id,
    )

    uploads_n   = int(upload_row["uploads"] or 0)
    completed_n = int(upload_row["completed"] or 0)
    success_pct = int(round((completed_n / uploads_n) * 100)) if uploads_n else 0

    return {
        "uploads":          uploads_n,
        "success_rate_pct": success_pct,
        "put_used":         int(upload_row["put_used"] or 0),
        "aic_used":         int(upload_row["aic_used"] or 0),
        "views":            int(engagement_row["views"]    or 0),
        "likes":            int(engagement_row["likes"]    or 0),
        "comments":         int(engagement_row["comments"] or 0),
        "shares":           int(engagement_row["shares"]   or 0),
        "put_balance":      int((wallet_row or {}).get("put_balance") or 0) if wallet_row else 0,
        "aic_balance":      int((wallet_row or {}).get("aic_balance") or 0) if wallet_row else 0,
    }


async def run_monthly_user_digest(
    pool: asyncpg.Pool, *, triggered_by: str = "manual"
) -> Dict[str, Any]:
    """Send a per-user 30-day KPI digest to everyone who:
       - has digest_emails enabled (Settings → Digest Emails)
       - has email_notifications enabled (master email kill-switch / legacy opt-out)
       - is NOT a brand-new account (>= 30 days old)
       - hasn't received a digest in the last MONTHLY_DIGEST_MIN_INTERVAL_DAYS days
       - has at least one upload in the past 30 days (no point digesting nothing)
    """
    job = "monthly_user_digest"
    run_id = await _start_run(pool, job, triggered_by)
    sent = skipped = errors = 0
    period_label = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%b %d") + " — " + datetime.now(timezone.utc).strftime("%b %d")

    try:
        async with pool.acquire() as conn:
            user_rows = await conn.fetch(
                f"""
                SELECT u.id, u.email, u.name, u.subscription_tier, u.last_monthly_digest_sent_at
                FROM users u
                LEFT JOIN user_preferences up ON up.user_id = u.id
                WHERE u.status = 'active'
                  AND u.email IS NOT NULL AND u.email <> ''
                  AND u.created_at <= NOW() - INTERVAL '30 days'
                  AND COALESCE(up.email_notifications, TRUE) = TRUE
                  AND COALESCE(up.digest_emails, TRUE) = TRUE
                  AND (
                        u.last_monthly_digest_sent_at IS NULL
                     OR u.last_monthly_digest_sent_at < NOW() - INTERVAL '{MONTHLY_DIGEST_MIN_INTERVAL_DAYS} days'
                  )
                """
            )

        for r in user_rows:
            prev_sent_at = r["last_monthly_digest_sent_at"]
            try:
                async with pool.acquire() as conn:
                    metrics = await _user_monthly_metrics(conn, r["id"])
                    if metrics["uploads"] == 0:
                        skipped += 1
                        continue

                    claimed = await conn.fetchval(
                        f"""
                        UPDATE users
                        SET last_monthly_digest_sent_at = NOW()
                        WHERE id = $1
                          AND (
                                last_monthly_digest_sent_at IS NULL
                             OR last_monthly_digest_sent_at < NOW() - INTERVAL '{MONTHLY_DIGEST_MIN_INTERVAL_DAYS} days'
                          )
                        RETURNING id
                        """,
                        r["id"],
                    )
                if not claimed:
                    skipped += 1
                    continue

                ok = await send_monthly_user_kpi_digest_email(
                    email=r["email"],
                    name=r["name"] or "there",
                    tier=r["subscription_tier"] or "free",
                    period_label=period_label,
                    uploads=metrics["uploads"],
                    success_rate_pct=metrics["success_rate_pct"],
                    views=metrics["views"],
                    likes=metrics["likes"],
                    put_used=metrics["put_used"],
                    aic_used=metrics["aic_used"],
                    put_balance=metrics["put_balance"],
                    aic_balance=metrics["aic_balance"],
                    comments=metrics["comments"],
                    shares=metrics["shares"],
                )
                if not ok:
                    async with pool.acquire() as conn:
                        await conn.execute(
                            "UPDATE users SET last_monthly_digest_sent_at = $2 WHERE id = $1",
                            r["id"],
                            prev_sent_at,
                        )
                    skipped += 1
                    continue
                sent += 1
            except Exception as e:
                logger.warning("monthly_user_digest failed for %s: %s", r["email"], e)
                try:
                    async with pool.acquire() as conn:
                        await conn.execute(
                            "UPDATE users SET last_monthly_digest_sent_at = $2 WHERE id = $1",
                            r["id"],
                            prev_sent_at,
                        )
                except Exception:
                    pass
                errors += 1

        summary = {
            "queried":      len(user_rows),
            "period_label": period_label,
        }
        await _finish_run(pool, run_id, sent, skipped, errors, summary)
        return {"job": job, "sent": sent, "skipped": skipped, "errors": errors, "details": summary}
    except Exception as e:
        logger.exception("run_monthly_user_digest failed: %s", e)
        await _finish_run(pool, run_id, sent, skipped, errors, {}, str(e))
        return {"job": job, "sent": sent, "skipped": skipped, "errors": errors + 1, "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# 3. Weekly admin KPI digest
# ─────────────────────────────────────────────────────────────────────────────

PAID_TIERS = ("creator_lite", "creator_pro", "studio", "agency", "enterprise")


async def _admin_recipients(conn: asyncpg.Connection) -> List[Tuple[Any, str, str, Any]]:
    """Active admin/master_admin users eligible for ops mail."""
    rows = await conn.fetch(
        f"""
        SELECT id, email, name, last_weekly_admin_digest_sent_at
        FROM users
        WHERE role IN ('admin', 'master_admin')
          AND status = 'active'
          AND email IS NOT NULL
          AND email <> ''
          AND (
                last_weekly_admin_digest_sent_at IS NULL
             OR last_weekly_admin_digest_sent_at < NOW() - INTERVAL '{WEEKLY_ADMIN_DIGEST_MIN_INTERVAL_DAYS} days'
          )
        """
    )
    return [
        (r["id"], r["email"], r["name"] or "Admin", r["last_weekly_admin_digest_sent_at"])
        for r in rows
    ]


async def run_weekly_admin_digest(
    pool: asyncpg.Pool, *, triggered_by: str = "manual"
) -> Dict[str, Any]:
    """Send the weekly KPI digest to every admin/master_admin user.

    Per-recipient ``last_weekly_admin_digest_sent_at`` prevents re-spam within
    WEEKLY_ADMIN_DIGEST_MIN_INTERVAL_DAYS.
    """
    job = "weekly_admin_digest"
    run_id = await _start_run(pool, job, triggered_by)
    sent = skipped = errors = 0

    week_label = (
        (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%b %d")
        + " — "
        + datetime.now(timezone.utc).strftime("%b %d, %Y")
    )

    try:
        async with pool.acquire() as conn:
            recipients = await _admin_recipients(conn)
            if not recipients:
                summary = {"reason": "no admin recipients (or all within digest window)"}
                await _finish_run(pool, run_id, 0, 0, 0, summary)
                return {"job": job, "sent": 0, "skipped": 0, "errors": 0, "details": summary}

            kpi_row = await conn.fetchrow(
                """
                SELECT
                  (SELECT COUNT(*) FROM users)                                               AS total_users,
                  (SELECT COUNT(*) FROM users WHERE created_at >= NOW() - INTERVAL '7 days') AS new_users,
                  (SELECT COUNT(*) FROM users
                    WHERE subscription_status IN ('active','trialing')
                      AND subscription_tier IN ('creator_lite','creator_pro','studio','agency','enterprise')) AS paid_users,
                  (SELECT COUNT(*) FROM users
                    WHERE subscription_status = 'trialing'
                      AND subscription_tier IN ('creator_lite','creator_pro','studio','agency','enterprise')) AS trialing_paid,
                  (SELECT COUNT(*) FROM uploads
                    WHERE created_at >= NOW() - INTERVAL '7 days')                            AS uploads_7d,
                  (SELECT COUNT(*) FROM uploads
                    WHERE created_at >= NOW() - INTERVAL '7 days'
                      AND status IN ('succeeded','completed','partial'))                      AS uploads_ok_7d,
                  (SELECT COALESCE(SUM(amount), 0) FROM revenue_tracking
                    WHERE created_at >= NOW() - INTERVAL '7 days')                            AS revenue_7d,
                  (SELECT COALESCE(SUM(cost_usd), 0) FROM cost_tracking
                    WHERE created_at >= NOW() - INTERVAL '7 days')                            AS cost_7d
                """
            )

        uploads_7d    = int(kpi_row["uploads_7d"] or 0)
        uploads_ok_7d = int(kpi_row["uploads_ok_7d"] or 0)
        upload_ok_pct = int(round((uploads_ok_7d / uploads_7d) * 100)) if uploads_7d else 0
        revenue_7d    = float(kpi_row["revenue_7d"] or 0)
        cost_7d       = float(kpi_row["cost_7d"] or 0)
        margin_pct    = ((revenue_7d - cost_7d) / revenue_7d * 100.0) if revenue_7d > 0 else 0.0

        for user_id, email, name, prev_sent_at in recipients:
            try:
                async with pool.acquire() as conn:
                    claimed = await conn.fetchval(
                        f"""
                        UPDATE users
                        SET last_weekly_admin_digest_sent_at = NOW()
                        WHERE id = $1
                          AND (
                                last_weekly_admin_digest_sent_at IS NULL
                             OR last_weekly_admin_digest_sent_at < NOW() - INTERVAL '{WEEKLY_ADMIN_DIGEST_MIN_INTERVAL_DAYS} days'
                          )
                        RETURNING id
                        """,
                        user_id,
                    )
                if not claimed:
                    skipped += 1
                    continue

                ok = await send_admin_weekly_kpi_digest_email(
                    email=email,
                    name=name,
                    week_label=week_label,
                    total_users=int(kpi_row["total_users"] or 0),
                    new_users=int(kpi_row["new_users"] or 0),
                    paid_users=int(kpi_row["paid_users"] or 0),
                    uploads=uploads_7d,
                    revenue=revenue_7d,
                    cost=cost_7d,
                    margin_pct=margin_pct,
                    upload_success_pct=upload_ok_pct,
                    trialing_paid_users=int(kpi_row["trialing_paid"] or 0),
                )
                if not ok:
                    async with pool.acquire() as conn:
                        await conn.execute(
                            "UPDATE users SET last_weekly_admin_digest_sent_at = $2 WHERE id = $1",
                            user_id,
                            prev_sent_at,
                        )
                    skipped += 1
                    continue
                sent += 1
            except Exception as e:
                logger.warning("weekly_admin_digest failed for %s: %s", email, e)
                try:
                    async with pool.acquire() as conn:
                        await conn.execute(
                            "UPDATE users SET last_weekly_admin_digest_sent_at = $2 WHERE id = $1",
                            user_id,
                            prev_sent_at,
                        )
                except Exception:
                    pass
                errors += 1

        summary = {
            "week_label":         week_label,
            "recipients":         len(recipients),
            "total_users":        int(kpi_row["total_users"] or 0),
            "new_users_7d":       int(kpi_row["new_users"] or 0),
            "uploads_7d":         uploads_7d,
            "upload_success_pct": upload_ok_pct,
            "revenue_7d":         revenue_7d,
            "cost_7d":            cost_7d,
            "margin_pct":         margin_pct,
        }
        await _finish_run(pool, run_id, sent, skipped, errors, summary)
        return {"job": job, "sent": sent, "skipped": skipped, "errors": errors, "details": summary}
    except Exception as e:
        logger.exception("run_weekly_admin_digest failed: %s", e)
        await _finish_run(pool, run_id, sent, skipped, errors, {}, str(e))
        return {"job": job, "sent": sent, "skipped": skipped, "errors": errors + 1, "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# 4. Scheduled publish alerts
#    - "upcoming": user has a publish due in the next 24h → friendly heads-up
#    - "stuck":    a scheduled publish that should have fired >30min ago but
#                  is still ready_to_publish → tell the user to retry/reschedule
# ─────────────────────────────────────────────────────────────────────────────

UPCOMING_WINDOW_HOURS = 24
STUCK_THRESHOLD_MINUTES = 30


async def _claim_upload_alert(conn: asyncpg.Connection, upload_id, key: str) -> bool:
    """Atomically set output_artifacts[key] if unset. Returns True if this caller won."""
    claimed = await conn.fetchval(
        """
        UPDATE uploads
        SET output_artifacts = COALESCE(output_artifacts, '{}'::jsonb)
                            || jsonb_build_object($2::text, NOW()::text)
        WHERE id = $1
          AND COALESCE(output_artifacts->>$2, '') = ''
        RETURNING id
        """,
        upload_id,
        key,
    )
    return bool(claimed)


async def _release_upload_alert(conn: asyncpg.Connection, upload_id, key: str) -> None:
    await conn.execute(
        """
        UPDATE uploads
        SET output_artifacts = COALESCE(output_artifacts, '{}'::jsonb) - $2::text
        WHERE id = $1
        """,
        upload_id,
        key,
    )


async def run_scheduled_publish_alerts(
    pool: asyncpg.Pool, *, triggered_by: str = "manual"
) -> Dict[str, Any]:
    """Send 'heads up' and 'stuck publish' alerts based on scheduled_time."""
    job = "scheduled_publish_alerts"
    run_id = await _start_run(pool, job, triggered_by)
    sent = skipped = errors = 0
    upcoming_sent = stuck_sent = stuck_staged_sent = 0

    try:
        async with pool.acquire() as conn:
            # ── Upcoming uploads in the next 24h, alerted at most once each.
            upcoming = await conn.fetch(
                f"""
                SELECT u.id, u.user_id, u.filename, u.scheduled_time,
                       us.email, us.name
                FROM uploads u
                JOIN users us ON us.id = u.user_id
                LEFT JOIN user_preferences up ON up.user_id = u.user_id
                WHERE u.status IN ('staged','ready_to_publish')
                  AND u.scheduled_time IS NOT NULL
                  AND u.scheduled_time BETWEEN NOW() AND NOW() + INTERVAL '{UPCOMING_WINDOW_HOURS} hours'
                  AND COALESCE(up.email_notifications, TRUE) = TRUE
                  AND COALESCE(up.scheduled_alert_emails, TRUE) = TRUE
                  AND us.status = 'active'
                  AND us.email IS NOT NULL AND us.email <> ''
                  AND COALESCE(u.output_artifacts->>'scheduled_alert_sent', '') = ''
                """
            )

            for r in upcoming:
                try:
                    if not await _claim_upload_alert(conn, r["id"], "scheduled_alert_sent"):
                        skipped += 1
                        continue
                    ok = await send_scheduled_publish_alert_email(
                        email=r["email"],
                        name=r["name"] or "there",
                        filename=r["filename"] or "your upload",
                        scheduled_at_label=r["scheduled_time"].strftime("%B %d, %Y %H:%M UTC"),
                        status="upcoming",
                        upload_id=str(r["id"]),
                    )
                    if not ok:
                        await _release_upload_alert(conn, r["id"], "scheduled_alert_sent")
                        skipped += 1
                        continue
                    sent += 1
                    upcoming_sent += 1
                except Exception as e:
                    logger.warning("scheduled upcoming alert failed for upload %s: %s", r["id"], e)
                    try:
                        await _release_upload_alert(conn, r["id"], "scheduled_alert_sent")
                    except Exception:
                        pass
                    errors += 1

            # ── Stuck publishes (should have fired but didn't)
            stuck = await conn.fetch(
                f"""
                SELECT u.id, u.user_id, u.filename, u.scheduled_time,
                       us.email, us.name, u.error_code, u.error_detail
                FROM uploads u
                JOIN users us ON us.id = u.user_id
                LEFT JOIN user_preferences up ON up.user_id = u.user_id
                WHERE u.status = 'ready_to_publish'
                  AND u.scheduled_time IS NOT NULL
                  AND u.scheduled_time < NOW() - INTERVAL '{STUCK_THRESHOLD_MINUTES} minutes'
                  AND COALESCE(up.email_notifications, TRUE) = TRUE
                  AND COALESCE(up.scheduled_alert_emails, TRUE) = TRUE
                  AND us.status = 'active'
                  AND us.email IS NOT NULL AND us.email <> ''
                  AND COALESCE(u.output_artifacts->>'scheduled_stuck_alert_sent', '') = ''
                """
            )

            for r in stuck:
                try:
                    if not await _claim_upload_alert(conn, r["id"], "scheduled_stuck_alert_sent"):
                        skipped += 1
                        continue
                    ok = await send_scheduled_publish_alert_email(
                        email=r["email"],
                        name=r["name"] or "there",
                        filename=r["filename"] or "your upload",
                        scheduled_at_label=r["scheduled_time"].strftime("%B %d, %Y %H:%M UTC"),
                        status="stuck",
                        reason=(r["error_detail"] or r["error_code"] or "Job did not fire on time"),
                        upload_id=str(r["id"]),
                    )
                    if not ok:
                        await _release_upload_alert(conn, r["id"], "scheduled_stuck_alert_sent")
                        skipped += 1
                        continue
                    sent += 1
                    stuck_sent += 1
                except Exception as e:
                    logger.warning("scheduled stuck alert failed for upload %s: %s", r["id"], e)
                    try:
                        await _release_upload_alert(conn, r["id"], "scheduled_stuck_alert_sent")
                    except Exception:
                        pass
                    errors += 1

            # ── Stuck staged (past scheduled_time but never entered processing)
            stuck_staged = await conn.fetch(
                f"""
                SELECT u.id, u.user_id, u.filename, u.scheduled_time,
                       us.email, us.name, u.error_code, u.error_detail
                FROM uploads u
                JOIN users us ON us.id = u.user_id
                LEFT JOIN user_preferences up ON up.user_id = u.user_id
                WHERE u.status = 'staged'
                  AND u.scheduled_time IS NOT NULL
                  AND u.scheduled_time < NOW() - INTERVAL '{STUCK_THRESHOLD_MINUTES} minutes'
                  AND COALESCE(up.email_notifications, TRUE) = TRUE
                  AND COALESCE(up.scheduled_alert_emails, TRUE) = TRUE
                  AND us.status = 'active'
                  AND us.email IS NOT NULL AND us.email <> ''
                  AND COALESCE(u.output_artifacts->>'staged_stuck_alert_sent', '') = ''
                """
            )

            for r in stuck_staged:
                try:
                    if not await _claim_upload_alert(conn, r["id"], "staged_stuck_alert_sent"):
                        skipped += 1
                        continue
                    ok = await send_scheduled_publish_alert_email(
                        email=r["email"],
                        name=r["name"] or "there",
                        filename=r["filename"] or "your upload",
                        scheduled_at_label=r["scheduled_time"].strftime("%B %d, %Y %H:%M UTC"),
                        status="stuck",
                        reason=(
                            r["error_detail"]
                            or r["error_code"]
                            or "Upload is staged but processing never started"
                        ),
                        upload_id=str(r["id"]),
                    )
                    if not ok:
                        await _release_upload_alert(conn, r["id"], "staged_stuck_alert_sent")
                        skipped += 1
                        continue
                    sent += 1
                    stuck_staged_sent += 1
                except Exception as e:
                    logger.warning("staged stuck alert failed for upload %s: %s", r["id"], e)
                    try:
                        await _release_upload_alert(conn, r["id"], "staged_stuck_alert_sent")
                    except Exception:
                        pass
                    errors += 1

        summary = {
            "upcoming_sent":     upcoming_sent,
            "stuck_sent":        stuck_sent,
            "stuck_staged_sent": stuck_staged_sent,
            "upcoming_window_h": UPCOMING_WINDOW_HOURS,
            "stuck_threshold_m": STUCK_THRESHOLD_MINUTES,
        }
        await _finish_run(pool, run_id, sent, skipped, errors, summary)
        return {"job": job, "sent": sent, "skipped": skipped, "errors": errors, "details": summary}
    except Exception as e:
        logger.exception("run_scheduled_publish_alerts failed: %s", e)
        await _finish_run(pool, run_id, sent, skipped, errors, {}, str(e))
        return {"job": job, "sent": sent, "skipped": skipped, "errors": errors + 1, "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# Top-level dispatcher (used by the admin runner endpoint and the cron loop)
# ─────────────────────────────────────────────────────────────────────────────

ADMIN_EMAIL_JOBS = {
    "trial_reminders":          run_trial_reminders,
    "monthly_user_digest":      run_monthly_user_digest,
    "weekly_admin_digest":      run_weekly_admin_digest,
    "scheduled_publish_alerts": run_scheduled_publish_alerts,
}


async def run_admin_email_job(
    pool: asyncpg.Pool, job: str, *, triggered_by: str = "manual"
) -> Dict[str, Any]:
    """Run a single named job, or every email job when ``job=='all'``.

    ``all`` runs only the four ADMIN_EMAIL_JOBS entries — never marketing.
    """
    job_norm = (job or "").strip().lower()
    if job_norm == "all":
        results = []
        for name, fn in ADMIN_EMAIL_JOBS.items():
            results.append(await fn(pool, triggered_by=triggered_by))
        return {"job": "all", "ran": [r["job"] for r in results], "results": results}

    fn = ADMIN_EMAIL_JOBS.get(job_norm)
    if fn is None:
        return {"job": job, "error": f"unknown job: {job}"}
    return await fn(pool, triggered_by=triggered_by)


async def run_admin_email_jobs_loop(pool: asyncpg.Pool, shutdown_event: asyncio.Event) -> None:
    """Worker background loop: run all four email jobs once per interval."""
    logger.info(
        "[admin-email-jobs] loop started | interval=%ss",
        ADMIN_EMAIL_JOBS_INTERVAL_SEC,
    )
    # Stagger after KPI collector / analytics so cold start does not pile on.
    try:
        await asyncio.wait_for(asyncio.shield(shutdown_event.wait()), timeout=180)
        return
    except asyncio.TimeoutError:
        pass

    while not shutdown_event.is_set():
        try:
            result = await run_admin_email_job(pool, "all", triggered_by="cron:worker")
            logger.info(
                "[admin-email-jobs] tick complete | ran=%s",
                result.get("ran"),
            )
        except Exception as e:
            logger.warning("[admin-email-jobs] tick failed: %s", e)

        try:
            await asyncio.wait_for(
                asyncio.shield(shutdown_event.wait()),
                timeout=ADMIN_EMAIL_JOBS_INTERVAL_SEC,
            )
            break
        except asyncio.TimeoutError:
            pass

    logger.info("[admin-email-jobs] loop stopped")
