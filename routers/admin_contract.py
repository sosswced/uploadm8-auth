"""
Admin marketing, ML debug, and compat routes backed by Postgres aggregates
(marketing_events, studio_usage_events, marketing_campaigns, marketing_ai_decisions).
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Response
from pydantic import BaseModel, Field

import core.state
from core.deps import get_current_user, require_admin, require_master_admin
from services.pikzels_v2 import resolve_public_api_key
from services.growth_intelligence import (
    build_ai_truth_metrics,
    build_marketing_intel_bundle,
    fetch_account_intelligence,
    fetch_ml_priors_debug,
    fetch_pikzels_studio_usage,
    parse_range_since_until,
)
from services.marketing_strategist import generate_marketing_plan
from services.marketing_execution import (
    approval_required_for_channel,
    list_pending_approval_campaigns,
    maybe_alert_pending_approvals,
    run_marketing_execution_tick,
)
from services.ml_marketing import evaluate_uplift_vs_baseline, fetch_variant_leaderboard
from services.ml_scoring_job import run_ml_scoring_cycle

logger = logging.getLogger("uploadm8-api")

marketing_router = APIRouter(prefix="/api/admin/marketing", tags=["admin-marketing"])
ml_router = APIRouter(prefix="/api/admin/ml", tags=["admin-ml"])
admin_compat_router = APIRouter(prefix="/api/admin", tags=["admin-compat"])
public_marketing_router = APIRouter(prefix="/api/marketing", tags=["marketing"])


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_intel(range_key: str) -> Dict[str, Any]:
    return {
        "range": range_key,
        "marketing_funnel": {
            "shown": 0,
            "ctr_pct": 0.0,
            "clicked": 0,
            "dismiss_rate_pct": 0.0,
            "dismissed": 0,
            "same_session_attributed_revenue": 0.0,
            "view_through_7d_attributed_revenue": 0.0,
        },
        "sales_opportunity_levers": {
            "free_users_uploading_last_7d": 0,
            "users_low_put_available_0_29": 0,
            "users_low_aic_available_0_9": 0,
            "users_3plus_platform_connections": 0,
        },
        "promo_schedule_recommendations": [],
        "recommended_comms_plan": [],
    }


@marketing_router.get("/intel")
async def marketing_intel(range: str = Query("30d"), user: dict = Depends(require_admin)):
    try:
        async with core.state.db_pool.acquire() as conn:
            return await build_marketing_intel_bundle(conn, range)
    except Exception as e:
        logger.warning("marketing_intel fallback: %s", e)
        return _default_intel(range)


@marketing_router.get("/accounts")
async def marketing_accounts(
    range: str = Query("30d"),
    q: str = Query(""),
    limit: int = Query(150, ge=1, le=500),
    user: dict = Depends(require_admin),
):
    try:
        async with core.state.db_pool.acquire() as conn:
            rows = await fetch_account_intelligence(conn, range, q, limit)
        return {"range": range, "q": q, "accounts": rows}
    except Exception as e:
        logger.warning("marketing_accounts: %s", e)
        return {"range": range, "q": q, "accounts": []}


@marketing_router.get("/campaigns")
async def marketing_campaigns_list(limit: int = Query(100, ge=1, le=500), user: dict = Depends(require_admin)):
    async with core.state.db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, name, objective, channel, status, estimated_audience, schedule_at,
                   targeting, notes, range_key, created_at, updated_at,
                   approved_at, approved_by, template_subject, template_body_html,
                   template_body_text, discord_message_text
            FROM marketing_campaigns
            ORDER BY created_at DESC
            LIMIT $1
            """,
            limit,
        )
    out: List[Dict[str, Any]] = []
    for r in rows:
        tg = r["targeting"]
        if isinstance(tg, str):
            try:
                tg = json.loads(tg)
            except Exception:
                tg = {}
        out.append(
            {
                "id": str(r["id"]),
                "name": r["name"],
                "objective": r["objective"] or "",
                "channel": r["channel"] or "in_app",
                "status": r["status"] or "draft",
                "estimated_audience": int(r["estimated_audience"] or 0),
                "schedule_at": r["schedule_at"].isoformat() if r.get("schedule_at") else None,
                "targeting": tg if isinstance(tg, dict) else {},
                "notes": r["notes"] or "",
                "range_key": str(r.get("range_key") or "30d"),
                "created_at": r["created_at"].isoformat() if r.get("created_at") else None,
                "approved_at": r["approved_at"].isoformat() if r.get("approved_at") else None,
                "approved_by": str(r["approved_by"]) if r.get("approved_by") else None,
                "needs_approval": approval_required_for_channel(r["channel"] or "in_app"),
                "template_subject": r.get("template_subject") or "",
                "template_body_html": r.get("template_body_html") or "",
                "template_body_text": r.get("template_body_text") or "",
                "discord_message_text": r.get("discord_message_text") or "",
            }
        )
    return {"campaigns": out}


@marketing_router.post("/campaigns")
async def marketing_campaigns_create(
    payload: Dict[str, Any] = Body(default_factory=dict),
    user: dict = Depends(require_admin),
):
    cid = uuid.uuid4()
    est = int(payload.get("min_uploads_30d") or 0) * 3
    rk = str(payload.get("range") or payload.get("range_key") or "30d")[:32]
    targeting = {
        "tiers": payload.get("tiers") or [],
        "min_uploads_30d": payload.get("min_uploads_30d") or 0,
        "min_enterprise_fit_score": payload.get("min_enterprise_fit_score") or 0,
        "min_nudge_ctr_pct": payload.get("min_nudge_ctr_pct") or 0,
        "require_no_revenue_7d": bool(payload.get("require_no_revenue_7d")),
    }
    async with core.state.db_pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO marketing_campaigns (
                id, name, objective, channel, status, estimated_audience, schedule_at,
                targeting, notes, created_by, range_key,
                template_subject, template_body_html, template_body_text, discord_message_text
            )
            VALUES ($1, $2, $3, $4, 'draft', $5, $6, $7::jsonb, $8, $9::uuid, $10, $11, $12, $13, $14)
            """,
            cid,
            (payload.get("name") or "Campaign")[:500],
            (payload.get("objective") or "")[:8000],
            (payload.get("channel") or "in_app")[:50],
            est,
            payload.get("schedule_at"),
            json.dumps(targeting),
            (payload.get("notes") or "")[:8000],
            str(user["id"]) if user.get("id") else None,
            rk,
            (payload.get("template_subject") or "")[:500] or None,
            (payload.get("template_body_html") or None),
            (payload.get("template_body_text") or None),
            (payload.get("discord_message_text") or None),
        )
    row = {
        "id": str(cid),
        "name": payload.get("name") or "Campaign",
        "objective": payload.get("objective") or "",
        "channel": payload.get("channel") or "in_app",
        "status": "draft",
        "estimated_audience": est,
        "schedule_at": payload.get("schedule_at"),
        "targeting": targeting,
        "range_key": rk,
        "created_at": _now_iso(),
        "needs_approval": approval_required_for_channel(payload.get("channel") or "in_app"),
    }
    return {"campaign": row}


@marketing_router.post("/campaigns/preview")
async def marketing_campaigns_preview(
    payload: Dict[str, Any] = Body(default_factory=dict),
    user: dict = Depends(require_admin),
):
    est = int(payload.get("min_uploads_30d") or 0) * 2 + 12
    return {"estimated_audience": est}


@marketing_router.post("/campaigns/{campaign_id}/status")
async def marketing_campaign_status(
    campaign_id: str,
    payload: Dict[str, Any] = Body(default_factory=dict),
    user: dict = Depends(require_admin),
):
    try:
        cid = uuid.UUID(campaign_id)
    except ValueError:
        raise HTTPException(404, "Campaign not found")
    st = str(payload.get("status") or "draft")[:50]
    async with core.state.db_pool.acquire() as conn:
        cur = await conn.fetchrow("SELECT channel, approved_at FROM marketing_campaigns WHERE id = $1", cid)
        if not cur:
            raise HTTPException(404, "Campaign not found")
        ch = str(cur["channel"] or "in_app")
        eff = st
        notice = None
        if st == "active" and approval_required_for_channel(ch) and not cur.get("approved_at"):
            eff = "pending_approval"
            notice = "Outbound campaigns require master admin approval before sends."
        r = await conn.fetchrow(
            """
            UPDATE marketing_campaigns SET status = $2, updated_at = NOW() WHERE id = $1
            RETURNING *
            """,
            cid,
            eff,
        )
        if not r:
            raise HTTPException(404, "Campaign not found")
    tg = r["targeting"]
    if isinstance(tg, str):
        try:
            tg = json.loads(tg)
        except Exception:
            tg = {}
    c = {
        "id": str(r["id"]),
        "name": r["name"],
        "objective": r["objective"] or "",
        "channel": r["channel"],
        "status": r["status"],
        "estimated_audience": int(r["estimated_audience"] or 0),
        "schedule_at": r["schedule_at"].isoformat() if r.get("schedule_at") else None,
        "targeting": tg if isinstance(tg, dict) else {},
        "range_key": str(r.get("range_key") or "30d"),
        "updated_at": _now_iso(),
    }
    resp = {"campaign": c}
    if notice:
        resp["notice"] = notice
    return resp


@marketing_router.get("/campaigns/{campaign_id}/audience.csv")
async def marketing_campaign_audience_csv(campaign_id: str, user: dict = Depends(require_admin)):
    try:
        cid = uuid.UUID(campaign_id)
    except ValueError:
        raise HTTPException(404, "Campaign not found")
    async with core.state.db_pool.acquire() as conn:
        camp = await conn.fetchrow("SELECT targeting FROM marketing_campaigns WHERE id = $1", cid)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["user_id", "email", "note"])
    if not camp:
        w.writerow(["", "", "campaign not found"])
        return Response(content=buf.getvalue(), media_type="text/csv; charset=utf-8")
    w.writerow(["", "", "Use Marketing Ops account table + filters; full CSV export queued for warehouse sync."])
    return Response(content=buf.getvalue(), media_type="text/csv; charset=utf-8")


@marketing_router.post("/campaigns/{campaign_id}/handoff")
async def marketing_campaign_handoff(campaign_id: str, user: dict = Depends(require_admin)):
    try:
        cid = uuid.UUID(campaign_id)
    except ValueError:
        raise HTTPException(404, "Campaign not found")
    async with core.state.db_pool.acquire() as conn:
        c = await conn.fetchrow("SELECT * FROM marketing_campaigns WHERE id = $1", cid)
    if not c:
        raise HTTPException(404, "Campaign not found")
    title = f"Campaign: {c['name'] or 'Untitled'}"
    body = (
        f"Objective: {c['objective'] or '—'}\n"
        f"Channel: {c['channel'] or '—'}\n"
        f"Estimated audience: {c['estimated_audience'] or 0}\n"
    )
    return {"title": title, "body": body, "selected_user_ids": []}


@marketing_router.post("/campaigns/{campaign_id}/templates")
async def marketing_campaign_templates(
    campaign_id: str,
    payload: Dict[str, Any] = Body(default_factory=dict),
    user: dict = Depends(require_admin),
):
    try:
        cid = uuid.UUID(campaign_id)
    except ValueError:
        raise HTTPException(404, "Campaign not found")
    async with core.state.db_pool.acquire() as conn:
        r = await conn.fetchrow("SELECT * FROM marketing_campaigns WHERE id = $1", cid)
        if not r:
            raise HTTPException(404, "Campaign not found")
        ts = payload["template_subject"] if "template_subject" in payload else r.get("template_subject")
        th = payload["template_body_html"] if "template_body_html" in payload else r.get("template_body_html")
        tt = payload["template_body_text"] if "template_body_text" in payload else r.get("template_body_text")
        dm = payload["discord_message_text"] if "discord_message_text" in payload else r.get("discord_message_text")
        await conn.execute(
            """
            UPDATE marketing_campaigns SET
                template_subject = $2,
                template_body_html = $3,
                template_body_text = $4,
                discord_message_text = $5,
                updated_at = NOW()
            WHERE id = $1
            """,
            cid,
            (str(ts)[:500] if ts else None),
            th,
            tt,
            dm,
        )
    return {"ok": True}


@marketing_router.get("/campaigns/pending-approval/list")
async def marketing_pending_approval_list(user: dict = Depends(require_admin)):
    async with core.state.db_pool.acquire() as conn:
        rows = await list_pending_approval_campaigns(conn)
    return {"pending": rows, "count": len(rows)}


@marketing_router.post("/campaigns/{campaign_id}/approve")
async def marketing_campaign_master_approve(
    campaign_id: str,
    user: dict = Depends(require_master_admin),
):
    try:
        cid = uuid.UUID(campaign_id)
    except ValueError:
        raise HTTPException(404, "Campaign not found")
    async with core.state.db_pool.acquire() as conn:
        r = await conn.fetchrow(
            """
            UPDATE marketing_campaigns
            SET approved_at = NOW(),
                approved_by = $2::uuid,
                status = CASE WHEN status = 'pending_approval' THEN 'active' ELSE status END,
                updated_at = NOW()
            WHERE id = $1
            RETURNING id, status
            """,
            cid,
            str(user["id"]),
        )
    if not r:
        raise HTTPException(404, "Campaign not found")
    return {"ok": True, "campaign_id": str(r["id"]), "status": r["status"]}


@marketing_router.post("/execution/tick")
async def marketing_execution_tick(
    payload: Dict[str, Any] = Body(default_factory=dict),
    user: dict = Depends(require_admin),
):
    max_c = int(payload.get("max_per_campaign") or 30)
    max_t = int(payload.get("max_total_sends") or 120)
    async with core.state.db_pool.acquire() as conn:
        summary = await run_marketing_execution_tick(conn, max_per_campaign=max_c, max_total_sends=max_t)
    logger.info("marketing_execution_tick by %s: %s", user.get("email"), summary.get("run_id"))
    return summary


@marketing_router.api_route("/execution/heartbeat", methods=["GET", "POST"])
async def marketing_execution_heartbeat(user: dict = Depends(require_admin)):
    """GET: admin-marketing.html poll for pending approvals. POST: same (worker/cron may POST)."""
    async with core.state.db_pool.acquire() as conn:
        ping = await maybe_alert_pending_approvals(conn, min_interval_hours=2)
        pending = await list_pending_approval_campaigns(conn)
    return {"pending_count": len(pending), "pending": pending[:20], "discord_alert": ping}


async def _latest_truth_response(conn) -> Dict[str, Any]:
    row = await conn.fetchrow(
        """
        SELECT created_at, action, status, objective, range_key, confidence_score, used_openai, truth_snapshot
        FROM marketing_ai_decisions
        ORDER BY created_at DESC
        LIMIT 1
        """
    )
    if not row:
        return {"last_decision": None, "metrics_used": {}, "metric_sources": []}
    snap = row["truth_snapshot"]
    if isinstance(snap, str):
        try:
            snap = json.loads(snap)
        except Exception:
            snap = {}
    if not isinstance(snap, dict):
        snap = {}
    ld = {
        "created_at": row["created_at"].isoformat() if row.get("created_at") else None,
        "action": row["action"],
        "status": row["status"],
        "objective": row["objective"],
        "range_key": row["range_key"],
        "confidence_score": float(row["confidence_score"] or 0),
        "used_openai": bool(row["used_openai"]),
    }
    return {
        "last_decision": ld,
        "metrics_used": snap.get("metrics_used") if isinstance(snap.get("metrics_used"), dict) else snap,
        "metric_sources": snap.get("metric_sources")
        or [{"metric_group": "postgres", "sql_source": "growth_intelligence"}],
    }


@marketing_router.get("/ai/truth")
async def marketing_ai_truth(user: dict = Depends(require_admin)):
    async with core.state.db_pool.acquire() as conn:
        return await _latest_truth_response(conn)


@marketing_router.get("/ai/decisions")
async def marketing_ai_decisions(limit: int = Query(40, ge=1, le=500), user: dict = Depends(require_admin)):
    async with core.state.db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT created_at, action, status, objective, range_key, confidence_score, used_openai
            FROM marketing_ai_decisions
            ORDER BY created_at DESC
            LIMIT $1
            """,
            limit,
        )
    decisions = [
        {
            "created_at": r["created_at"].isoformat() if r.get("created_at") else None,
            "action": r["action"],
            "status": r["status"],
            "objective": r["objective"],
            "range_key": r["range_key"],
            "confidence_score": float(r["confidence_score"] or 0),
            "used_openai": bool(r["used_openai"]),
        }
        for r in rows
    ]
    return {"decisions": decisions}


@marketing_router.get("/ai/decisions.csv")
async def marketing_ai_decisions_csv(limit: int = Query(1000, ge=1, le=5000), user: dict = Depends(require_admin)):
    async with core.state.db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT created_at, action, status, objective, range_key, confidence_score, used_openai
            FROM marketing_ai_decisions
            ORDER BY created_at DESC
            LIMIT $1
            """,
            limit,
        )
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["created_at", "action", "status", "objective", "range_key", "confidence_score", "used_openai"])
    for r in rows:
        w.writerow(
            [
                r["created_at"].isoformat() if r.get("created_at") else "",
                r["action"],
                r["status"],
                r["objective"],
                r["range_key"],
                r["confidence_score"],
                r["used_openai"],
            ]
        )
    return Response(content=buf.getvalue(), media_type="text/csv; charset=utf-8")


async def _persist_ai_decision(
    conn,
    *,
    action: str,
    status: str,
    objective: str,
    range_key: str,
    confidence: float,
    used_openai: bool,
    truth_snapshot: Dict[str, Any],
    plan: Dict[str, Any],
) -> None:
    await conn.execute(
        """
        INSERT INTO marketing_ai_decisions (
            action, status, objective, range_key, confidence_score, used_openai, truth_snapshot, plan_json
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8::jsonb)
        """,
        action[:80],
        (status or "")[:80],
        (objective or "")[:2000],
        (range_key or "")[:32],
        float(confidence),
        used_openai,
        json.dumps(truth_snapshot),
        json.dumps(plan),
    )


@marketing_router.post("/ai/generate")
async def marketing_ai_generate(
    payload: Dict[str, Any] = Body(default_factory=dict),
    user: dict = Depends(require_admin),
):
    rk = str(payload.get("range") or payload.get("range_key") or "30d")
    objective = str(payload.get("objective") or "revenue_growth")
    async with core.state.db_pool.acquire() as conn:
        metrics_bundle = await build_ai_truth_metrics(conn, rk)
        plan, used_ai, _dbg = await generate_marketing_plan(payload, metrics_bundle)
        conf = float((plan.get("game_plan") or {}).get("confidence_score") or 62.0)
        truth_snapshot = {
            "metrics_used": metrics_bundle,
            "metric_sources": [
                {"metric_group": "live", "sql_source": "services.growth_intelligence.build_ai_truth_metrics"}
            ],
        }
        await _persist_ai_decision(
            conn,
            action="generate",
            status="ok",
            objective=objective,
            range_key=rk,
            confidence=conf,
            used_openai=used_ai,
            truth_snapshot=truth_snapshot,
            plan=plan,
        )
    return {"plan": plan, "used_openai": used_ai}


@marketing_router.post("/ai/deploy")
async def marketing_ai_deploy(
    payload: Dict[str, Any] = Body(default_factory=dict),
    user: dict = Depends(require_admin),
):
    rk = str(payload.get("range") or payload.get("range_key") or "30d")
    objective = str(payload.get("objective") or "revenue_growth")
    async with core.state.db_pool.acquire() as conn:
        metrics_bundle = await build_ai_truth_metrics(conn, rk)
        plan, used_ai, _dbg = await generate_marketing_plan(payload, metrics_bundle)
        suggested = plan.get("suggested_campaign") or {}
        cid = uuid.uuid4()
        targeting = {
            "tiers": suggested.get("tiers") or [],
            "min_uploads_30d": suggested.get("min_uploads_30d") or 0,
            "min_enterprise_fit_score": suggested.get("min_enterprise_fit_score") or 0,
            "min_nudge_ctr_pct": suggested.get("min_nudge_ctr_pct") or 0,
            "require_no_revenue_7d": bool(suggested.get("require_no_revenue_7d")),
        }
        await conn.execute(
            """
            INSERT INTO marketing_campaigns (
                id, name, objective, channel, status, estimated_audience, schedule_at,
                targeting, notes, created_by, range_key
            )
            VALUES ($1, $2, $3, $4, 'draft', $5, NULL, $6::jsonb, $7, $8::uuid, $9)
            """,
            cid,
            (suggested.get("name") or "AI campaign")[:500],
            (suggested.get("objective") or objective)[:8000],
            (suggested.get("channel") or "in_app")[:50],
            120,
            json.dumps(targeting),
            (suggested.get("notes") or "")[:8000],
            str(user["id"]) if user.get("id") else None,
            rk[:32],
        )
        decision = {
            "confidence_score": float((plan.get("game_plan") or {}).get("confidence_score") or 72.0),
            "blocked_reasons": [],
            "action": "deploy",
            "status": "ok",
            "created_at": _now_iso(),
            "used_openai": used_ai,
        }
        truth_snapshot = {
            "metrics_used": metrics_bundle,
            "metric_sources": [
                {"metric_group": "live", "sql_source": "services.growth_intelligence.build_ai_truth_metrics"}
            ],
        }
        await _persist_ai_decision(
            conn,
            action="deploy",
            status="deployed",
            objective=str(suggested.get("objective") or objective),
            range_key=rk,
            confidence=float(decision["confidence_score"]),
            used_openai=used_ai,
            truth_snapshot=truth_snapshot,
            plan=plan,
        )
    campaign = {
        "id": str(cid),
        "name": suggested.get("name") or "AI campaign",
        "objective": suggested.get("objective") or objective,
        "channel": suggested.get("channel") or "in_app",
        "status": "draft",
        "estimated_audience": 120,
        "schedule_at": None,
        "targeting": targeting,
        "range_key": rk[:32],
        "created_at": _now_iso(),
    }
    return {
        "status": "deployed",
        "decision": decision,
        "plan": plan,
        "deployed_campaign": campaign,
        "snapshot": {"campaign_id": str(cid)},
        "used_openai": used_ai,
    }


@ml_router.get("/priors/latest")
async def ml_priors_latest(
    since_hours: int = Query(72, ge=1, le=168 * 4),
    limit: int = Query(50, ge=1, le=500),
    user: dict = Depends(require_admin),
):
    since = datetime.now(timezone.utc) - timedelta(hours=since_hours)
    try:
        async with core.state.db_pool.acquire() as conn:
            return await fetch_ml_priors_debug(conn, since, limit)
    except Exception as e:
        logger.warning("ml_priors_latest: %s", e)
        return {
            "summary": {
                "total": 0,
                "thumbnail_bias_present": 0,
                "m8_strategy_priors_present": 0,
                "since_hours": since_hours,
            },
            "items": [],
        }


@ml_router.get("/variant-leaderboard")
async def ml_variant_leaderboard(limit: int = Query(40, ge=1, le=200), user: dict = Depends(require_admin)):
    async with core.state.db_pool.acquire() as conn:
        rows = await fetch_variant_leaderboard(conn, limit)
    return {"variants": rows}


@ml_router.post("/promotion-eval")
async def ml_promotion_eval(
    payload: Dict[str, Any] = Body(default_factory=dict),
    user: dict = Depends(require_admin),
):
    mk = str(payload.get("model_key") or "propensity_v1")[:80]
    ms = int(payload.get("min_samples") or 40)
    async with core.state.db_pool.acquire() as conn:
        result = await evaluate_uplift_vs_baseline(conn, model_key=mk, min_samples=ms)
    return result


@ml_router.post("/recompute-quality-scores")
async def ml_recompute_quality_scores(
    lookback_days: int = Query(180, ge=7, le=3650),
    user: dict = Depends(require_admin),
):
    """Rebuild ``upload_quality_scores_daily`` from uploads + output_artifacts (admin on-demand)."""
    n = await run_ml_scoring_cycle(core.state.db_pool, lookback_days=lookback_days)
    return {"ok": True, "rows_touched": n}


class EmailJobRunBody(BaseModel):
    job: str = Field(..., min_length=1, max_length=120)


@admin_compat_router.post("/email-jobs/run")
async def email_jobs_run(body: EmailJobRunBody, user: dict = Depends(require_admin)):
    logger.info("email-jobs/run job=%s admin=%s", body.job, user.get("email"))
    if (body.job or "").strip().lower() in ("marketing_execution", "marketing_touchpoints", "all"):
        async with core.state.db_pool.acquire() as conn:
            summary = await run_marketing_execution_tick(conn)
        return {"ok": True, "ran": [body.job], "marketing_execution": summary}
    return {"ok": True, "ran": [body.job]}


@admin_compat_router.post("/platform-kpi-rollups/refresh")
async def platform_kpi_rollups_refresh(days: int = Query(7, ge=1, le=90), user: dict = Depends(require_admin)):
    return {"ok": True, "days": days}


@admin_compat_router.get("/kpi/cost-tracker")
async def kpi_cost_tracker(user: dict = Depends(require_admin)):
    return {
        "estimated_total_window_usd": 0.0,
        "estimated_total_per_upload_usd": 0.0,
        "successful_uploads": 0,
    }


@admin_compat_router.get("/kpi/provider-costs")
async def kpi_provider_costs(user: dict = Depends(require_admin)):
    return {
        "render_cost": 0.0,
        "storage_gb": 0.0,
        "storage_cost": 0.0,
        "bandwidth_cost": 0.0,
        "bandwidth_tb": 0.0,
        "redis_cost": 0.0,
        "redis_memory_mb": 0.0,
        "postgres_cost": 0.0,
        "postgres_size_gb": 0.0,
        "mailgun_cost": 0.0,
        "mailgun_emails_sent": 0,
        "stripe_fees": 0.0,
        "stripe_fee_txns": 0,
    }


@admin_compat_router.get("/kpi/pikzels-v2-usage")
async def kpi_pikzels_v2_usage(range: str = Query("30d"), user: dict = Depends(require_admin)):
    since, until = parse_range_since_until(range)
    try:
        async with core.state.db_pool.acquire() as conn:
            data = await fetch_pikzels_studio_usage(conn, since, until)
        return {"range": range, **data}
    except Exception as e:
        logger.warning("pikzels-v2-usage: %s", e)
        return {"range": range, "total_calls": 0, "by_operation": []}


@admin_compat_router.get("/integrations/thumbnail-provider")
async def integrations_thumbnail_provider(user: dict = Depends(require_admin)):
    key_ok = bool((resolve_public_api_key() or "").strip())
    mode = os.environ.get("THUMBNAIL_UPLOAD_PIPELINE_MODE", "default")
    legacy = os.environ.get("LEGACY_ONE_SHOT_RENDERER", "").strip().lower() in ("1", "true", "yes", "on")
    return {
        "upload_pipeline_mode": mode,
        "v2_api_key_configured": key_ok,
        "legacy_one_shot_renderer_enabled": legacy,
    }


@public_marketing_router.post("/events")
async def marketing_events_ingest(
    payload: Dict[str, Any] = Body(default_factory=dict),
    user: dict = Depends(get_current_user),
):
    et = str(payload.get("event_type") or "unknown")[:80]
    async with core.state.db_pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO marketing_events (user_id, event_type, payload)
            VALUES ($1::uuid, $2, $3::jsonb)
            """,
            str(user["id"]),
            et,
            json.dumps(payload),
        )
    return Response(status_code=204)
