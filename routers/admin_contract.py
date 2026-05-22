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
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Response
from pydantic import BaseModel, Field

import core.state
from core.audit import log_admin_audit
from core.deps import get_current_user, require_admin, require_master_admin
from services.pikzels_v2 import resolve_public_api_key
from services.admin_kpi_finance import build_cost_tracker_payload, build_provider_costs_payload
from services.growth_intelligence import (
    build_ai_truth_metrics,
    build_marketing_intel_bundle,
    fetch_account_intelligence,
    fetch_ml_priors_debug,
    fetch_pikzels_studio_usage,
    parse_range_since_until,
)
from services.pikzels_thumbnail_kpi import fetch_pikzels_template_render_kpi
from services.marketing_promo_media import (
    _promo_media_dict,
    build_platform_avatars_html_row,
    fetch_user_marketing_visual_context,
    generate_and_store_promo_image,
    resolve_promo_presigned_url,
    user_visual_context_enabled_for_campaign,
)
from services.marketing_execution import (
    approval_required_for_channel,
    count_campaign_audience,
    list_pending_approval_campaigns,
    maybe_alert_pending_approvals,
    run_marketing_execution_tick,
    _iter_candidate_users,
    _parse_targeting,
    _user_matches_campaign,
)
from services.marketing_image import generate_marketing_image
from services.wallet_marketing import _user_campaign_features
from stages.entitlements import normalize_tier
from services.admin_email_jobs import (
    ADMIN_EMAIL_JOBS,
    run_admin_email_job,
)
from services.ml_marketing import (
    evaluate_uplift_vs_baseline,
    fetch_marketing_conversion_proxy,
    fetch_variant_leaderboard,
    infer_promo_media_defaults,
)
from services.ml_observability import OptionalTrackioRun
from services.ml_scoring_job import run_ml_scoring_cycle

logger = logging.getLogger("uploadm8-api")

marketing_router = APIRouter(prefix="/api/admin/marketing", tags=["admin-marketing"])


class MarketingPreviewRequest(BaseModel):
    """Body for ``POST /api/admin/marketing/preview`` — dry-run personalized product card."""

    user_id: str = Field(..., description="UUID of the user to render for")
    kind: str = Field(
        ...,
        description="topup_aic | topup_put | sub_upgrade | win_back | trial_remind",
    )
    amount: Optional[int] = Field(None, description="For top-up offers: e.g. 50/100/250/500/1000")
    headline: Optional[str] = None
    use_llm: bool = True
    use_pikzels: bool = True
    upload_to_r2: bool = True
    debit_wallet: bool = False
    campaign_id: Optional[str] = None
    variant_id: Optional[str] = None


class MarketingTicketResolveBody(BaseModel):
    decision: str = Field(..., description="approved | rejected")
    notes: Optional[str] = Field(None, max_length=8000)


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
                   template_body_text, discord_message_text, promo_media
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
        pm = r.get("promo_media")
        if isinstance(pm, str):
            try:
                pm = json.loads(pm)
            except Exception:
                pm = {}
        if not isinstance(pm, dict):
            pm = {}
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
                "promo_media": pm,
            }
        )
    return {"campaigns": out}


@marketing_router.post("/campaigns")
async def marketing_campaigns_create(
    payload: Dict[str, Any] = Body(default_factory=dict),
    user: dict = Depends(require_admin),
):
    cid = uuid.uuid4()
    rk = str(payload.get("range") or payload.get("range_key") or "30d")[:32]
    targeting = {
        "tiers": payload.get("tiers") or [],
        "min_uploads_30d": payload.get("min_uploads_30d") or 0,
        "min_enterprise_fit_score": payload.get("min_enterprise_fit_score") or 0,
        "min_nudge_ctr_pct": payload.get("min_nudge_ctr_pct") or 0,
        "require_no_revenue_7d": bool(payload.get("require_no_revenue_7d")),
    }
    pm0 = payload.get("promo_media")
    if pm0 is not None and not isinstance(pm0, dict):
        pm0 = {}
    apply_defaults = bool(payload.get("apply_promo_defaults") or payload.get("apply_promo_media_defaults"))
    async with core.state.db_pool.acquire() as conn:
        aud = await count_campaign_audience(conn, targeting=targeting, range_key=rk)
        est = int(aud.get("matched_count") or 0)
        merged_pm: Dict[str, Any] = dict(pm0) if isinstance(pm0, dict) else {}
        if apply_defaults:
            inferred = await infer_promo_media_defaults(conn)
            merged_pm = {**inferred, **merged_pm}
        promo_json = json.dumps(merged_pm)
        await conn.execute(
            """
            INSERT INTO marketing_campaigns (
                id, name, objective, channel, status, estimated_audience, schedule_at,
                targeting, notes, created_by, range_key,
                template_subject, template_body_html, template_body_text, discord_message_text,
                promo_media
            )
            VALUES ($1, $2, $3, $4, 'draft', $5, $6, $7::jsonb, $8, $9::uuid, $10, $11, $12, $13, $14, $15::jsonb)
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
            promo_json,
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
    rk = str(payload.get("range") or payload.get("range_key") or "30d")[:32]
    targeting = {
        "tiers": payload.get("tiers") or [],
        "min_uploads_30d": payload.get("min_uploads_30d") or 0,
        "min_enterprise_fit_score": payload.get("min_enterprise_fit_score") or 0,
        "min_nudge_ctr_pct": payload.get("min_nudge_ctr_pct") or 0,
        "require_no_revenue_7d": bool(payload.get("require_no_revenue_7d")),
    }
    cid = payload.get("campaign_id")
    async with core.state.db_pool.acquire() as conn:
        aud = await count_campaign_audience(
            conn,
            targeting=targeting,
            range_key=rk,
            campaign_id=str(cid) if cid else None,
        )
    return aud


@marketing_router.post("/campaigns/{campaign_id}/audience-count")
async def marketing_campaign_audience_count(
    campaign_id: str,
    user: dict = Depends(require_admin),
):
    try:
        cid = uuid.UUID(campaign_id)
    except ValueError:
        raise HTTPException(404, "Campaign not found")
    async with core.state.db_pool.acquire() as conn:
        camp = await conn.fetchrow(
            "SELECT targeting, range_key FROM marketing_campaigns WHERE id = $1",
            cid,
        )
        if not camp:
            raise HTTPException(404, "Campaign not found")
        aud = await count_campaign_audience(
            conn,
            targeting=camp["targeting"],
            range_key=str(camp.get("range_key") or "30d"),
            campaign_id=str(cid),
        )
        est = int(aud.get("matched_count") or 0)
        await conn.execute(
            "UPDATE marketing_campaigns SET estimated_audience = $2, updated_at = NOW() WHERE id = $1",
            cid,
            est,
        )
    return aud


_KINDS_MARKETING_CARD = frozenset(
    {"topup_aic", "topup_put", "sub_upgrade", "win_back", "trial_remind"}
)


@marketing_router.post("/preview")
async def admin_marketing_preview(
    body: MarketingPreviewRequest,
    user: dict = Depends(require_master_admin),
):
    """Render a personalized marketing card for one user without sending email/Discord."""
    try:
        uuid.UUID(str(body.user_id).strip())
    except ValueError:
        raise HTTPException(status_code=400, detail="user_id must be a valid UUID")
    k = str(body.kind or "").strip().lower()
    if k not in _KINDS_MARKETING_CARD:
        raise HTTPException(
            status_code=400,
            detail=f"kind must be one of: {', '.join(sorted(_KINDS_MARKETING_CARD))}",
        )
    async with core.state.db_pool.acquire() as conn:
        return await generate_marketing_image(
            conn,
            user_id=str(body.user_id).strip(),
            kind=k,
            campaign_id=(str(body.campaign_id).strip() if body.campaign_id else None),
            variant_id=(str(body.variant_id).strip()[:128] if body.variant_id else None),
            headline=(str(body.headline).strip() or None) if body.headline is not None else None,
            amount=body.amount,
            upload_to_r2=body.upload_to_r2,
            use_llm=body.use_llm,
            use_pikzels=body.use_pikzels,
            debit_wallet=body.debit_wallet,
            channel=None,
        )


@marketing_router.post("/campaigns/{campaign_id}/render-images")
async def admin_marketing_render_campaign_images(
    campaign_id: str,
    force: bool = Query(False, description="If true, overwrite existing rendered_image_url rows"),
    user: dict = Depends(require_master_admin),
):
    """Pre-render personalized cards for up to 1000 users; stores URLs in marketing_campaign_audience."""
    try:
        cid = uuid.UUID(campaign_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Campaign not found")

    track = OptionalTrackioRun("marketing_bulk_render_cards")
    track.start(config={"campaign_id": campaign_id, "force": bool(force)})
    try:
        async with core.state.db_pool.acquire() as conn:
            camp = await conn.fetchrow(
                """
                SELECT id, targeting, range_key, channel, promo_media
                FROM marketing_campaigns
                WHERE id = $1
                """,
                cid,
            )
            if not camp:
                raise HTTPException(status_code=404, detail="Campaign not found")

            targeting = _parse_targeting(camp["targeting"])
            range_key = str(camp.get("range_key") or "30d")
            ch = str(camp.get("channel") or "in_app").strip().lower()
            pm = _promo_media_dict(camp.get("promo_media"))
            ck = str(pm.get("card_kind") or "topup_aic").strip().lower()
            if ck not in _KINDS_MARKETING_CARD:
                ck = "topup_aic"
            amt_raw = pm.get("amount")
            amt_i: Optional[int] = None
            if amt_raw is not None:
                try:
                    amt_i = int(amt_raw)
                except (TypeError, ValueError):
                    amt_i = None
            hl_raw = pm.get("headline")
            hl_opt = str(hl_raw).strip() if hl_raw else None
            if hl_opt == "":
                hl_opt = None

            pending = await conn.fetch(
                """
                SELECT user_id::text AS user_id
                FROM marketing_campaign_audience
                WHERE campaign_id = $1
                  AND ($2::bool OR rendered_image_url IS NULL)
                LIMIT 1000
                """,
                cid,
                force,
            )
            uids: List[str] = [str(r["user_id"]) for r in pending]
            if not uids:
                users = await _iter_candidate_users(conn, 2500)
                for urow in users:
                    if len(uids) >= 1000:
                        break
                    uid = str(urow["id"])
                    tier = str(urow.get("subscription_tier") or "free")
                    feats = await _user_campaign_features(conn, uid, range_key)
                    if not _user_matches_campaign(tier, feats, targeting):
                        continue
                    if not force:
                        ex = await conn.fetchrow(
                            """
                            SELECT rendered_image_url
                            FROM marketing_campaign_audience
                            WHERE campaign_id = $1 AND user_id = $2::uuid
                            """,
                            cid,
                            uid,
                        )
                        if ex and (ex.get("rendered_image_url") or "").strip():
                            continue
                    uids.append(uid)

            rendered = 0
            errors = 0
            for uid in uids:
                try:
                    res = await generate_marketing_image(
                        conn,
                        user_id=uid,
                        kind=ck,
                        campaign_id=str(cid),
                        variant_id=None,
                        headline=hl_opt,
                        amount=amt_i,
                        upload_to_r2=True,
                        use_llm=bool(pm.get("use_llm", True)),
                        use_pikzels=bool(pm.get("use_pikzels", True)),
                        debit_wallet=bool(pm.get("debit_wallet_for_card", False)),
                        channel=ch if ch in ("email", "discord", "mixed") else None,
                        features_range_key=range_key,
                    )
                    if res.get("image_url") and not res.get("error"):
                        await conn.execute(
                            """
                            INSERT INTO marketing_campaign_audience (
                                campaign_id, user_id, rendered_image_url, variant_id, rendered_at
                            )
                            VALUES ($1, $2::uuid, $3, $4, NOW())
                            ON CONFLICT (campaign_id, user_id) DO UPDATE SET
                                rendered_image_url = EXCLUDED.rendered_image_url,
                                variant_id = EXCLUDED.variant_id,
                                rendered_at = EXCLUDED.rendered_at
                            """,
                            cid,
                            uid,
                            res["image_url"],
                            str(res.get("variant_id") or "")[:128] or None,
                        )
                        rendered += 1
                    else:
                        errors += 1
                except Exception:
                    errors += 1
                    continue

            out = {
                "campaign_id": str(cid),
                "rendered": rendered,
                "errors": errors,
                "candidates": len(uids),
            }
        track.log(
            {
                "status": 1,
                "rendered": out["rendered"],
                "errors": out["errors"],
                "candidates": out["candidates"],
            }
        )
        return out
    except HTTPException as he:
        det = he.detail
        if not isinstance(det, str):
            det = str(det)
        track.log({"status": 0, "error": det[:300]})
        raise
    except Exception as e:
        track.log({"status": 0, "error": str(e)[:300]})
        raise
    finally:
        track.finish()


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
        from services.marketing_approval_tickets import ensure_open_ticket_for_pending_campaign

        _sub_uid = None
        if user.get("id"):
            try:
                _sub_uid = uuid.UUID(str(user["id"]))
            except ValueError:
                _sub_uid = None
        await ensure_open_ticket_for_pending_campaign(
            conn,
            campaign_id=cid,
            submitted_by=_sub_uid,
            channel=str(r.get("channel") or ""),
            status=str(r.get("status") or ""),
        )
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
        if "promo_media" in payload or bool(payload.get("apply_promo_defaults") or payload.get("apply_promo_media_defaults")):
            pm = payload.get("promo_media") if isinstance(payload.get("promo_media"), dict) else {}
            if not isinstance(pm, dict):
                pm = {}
            raw_pm = r.get("promo_media")
            if isinstance(raw_pm, dict):
                base = dict(raw_pm)
            elif isinstance(raw_pm, str):
                try:
                    parsed = json.loads(raw_pm)
                    base = dict(parsed) if isinstance(parsed, dict) else {}
                except Exception:
                    base = {}
            else:
                base = {}
            merged = {**base, **pm}
            if bool(payload.get("apply_promo_defaults") or payload.get("apply_promo_media_defaults")):
                inferred = await infer_promo_media_defaults(conn)
                merged = {**inferred, **merged}
            await conn.execute(
                """
                UPDATE marketing_campaigns SET promo_media = $2::jsonb, updated_at = NOW() WHERE id = $1
                """,
                cid,
                json.dumps(merged),
            )
    return {"ok": True}


@marketing_router.post("/campaigns/{campaign_id}/promo-media/generate")
async def marketing_campaign_promo_media_generate(
    campaign_id: str,
    payload: Dict[str, Any] = Body(default_factory=dict),
    user: dict = Depends(require_admin),
):
    """Preview/regenerate platform Pikzels hero for a campaign segment (writes R2 + ``promo_media``)."""
    try:
        cid = uuid.UUID(campaign_id)
    except ValueError:
        raise HTTPException(404, "Campaign not found")
    seg = str(payload.get("segment_bucket") or "all").strip()[:120] or "all"
    preview_uid = str(payload.get("preview_user_id") or "").strip()
    uv: Optional[Dict[str, Any]] = None
    pm: Dict[str, Any] = {}
    async with core.state.db_pool.acquire() as conn:
        r = await conn.fetchrow("SELECT * FROM marketing_campaigns WHERE id = $1", cid)
        if not r:
            raise HTTPException(404, "Campaign not found")
        pm = r.get("promo_media") or {}
        if isinstance(pm, str):
            try:
                pm = json.loads(pm)
            except Exception:
                pm = {}
        if not isinstance(pm, dict):
            pm = {}
        if isinstance(payload.get("promo_media"), dict):
            pm = {**pm, **payload["promo_media"]}
        if payload.get("mode"):
            pm["mode"] = str(payload.get("mode"))[:32]
        if not str(pm.get("mode") or "").strip() or str(pm.get("mode")).lower() in ("none", ""):
            pm["mode"] = "pikzels_static"
        await conn.execute(
            "UPDATE marketing_campaigns SET promo_media = $2::jsonb, updated_at = NOW() WHERE id = $1",
            cid,
            json.dumps(pm),
        )
        if preview_uid and user_visual_context_enabled_for_campaign(pm):
            try:
                uv = await fetch_user_marketing_visual_context(conn, preview_uid)
            except Exception:
                uv = {}
        _rk, url, _vid = await generate_and_store_promo_image(
            conn,
            entity_kind="campaign",
            entity_id=str(cid),
            promo_media=pm,
            title=str(r.get("name") or "UploadM8"),
            objective=str(r.get("objective") or ""),
            segment_bucket=seg,
            marketing_user_id=preview_uid if preview_uid and user_visual_context_enabled_for_campaign(pm) else None,
            user_visual=uv,
        )
        row = await conn.fetchrow("SELECT promo_media FROM marketing_campaigns WHERE id = $1", cid)
        pm_out = row["promo_media"] if row else {}
        if not url:
            url = await resolve_promo_presigned_url(conn, pm_out)
    extras: Dict[str, Any] = {}
    if preview_uid and uv and user_visual_context_enabled_for_campaign(pm):
        h = build_platform_avatars_html_row(uv.get("platforms"))
        if h:
            extras["marketing_platform_avatars_html"] = h
    return {"ok": True, "segment_bucket": seg, "promo_media": pm_out, "presigned_url": url, "extras": extras}


@marketing_router.get("/campaigns/pending-approval/list")
async def marketing_pending_approval_list(user: dict = Depends(require_admin)):
    async with core.state.db_pool.acquire() as conn:
        rows = await list_pending_approval_campaigns(conn)
    return {"pending": rows, "count": len(rows)}


@marketing_router.post("/campaigns/{campaign_id}/approve")
async def marketing_campaign_master_approve(
    campaign_id: str,
    payload: Dict[str, Any] = Body(default_factory=dict),
    user: dict = Depends(require_master_admin),
):
    try:
        cid = uuid.UUID(campaign_id)
    except ValueError:
        raise HTTPException(404, "Campaign not found")
    try:
        master_uuid = uuid.UUID(str(user["id"]))
    except (ValueError, TypeError):
        raise HTTPException(400, "Invalid master user id")

    from services.marketing_approval_tickets import (
        channel_requires_ticket,
        ensure_open_ticket_for_pending_campaign,
        resolve_ticket_master,
    )

    async with core.state.db_pool.acquire() as conn:
        camp = await conn.fetchrow(
            "SELECT id, channel, status FROM marketing_campaigns WHERE id = $1",
            cid,
        )
        if not camp:
            raise HTTPException(404, "Campaign not found")
        if not channel_requires_ticket(str(camp["channel"] or "")):
            r = await conn.fetchrow(
                """
                UPDATE marketing_campaigns
                SET approved_at = NOW(),
                    approved_by = $2,
                    status = 'active',
                    updated_at = NOW()
                WHERE id = $1
                RETURNING id, status
                """,
                cid,
                master_uuid,
            )
            out = {"ok": True, "campaign_id": str(r["id"]), "status": r["status"], "ticket_skipped": True}
            if bool(payload.get("run_tick")):
                async with core.state.db_pool.acquire() as conn2:
                    out["execution_tick"] = await run_marketing_execution_tick(conn2)
            return out

        tid = await conn.fetchval(
            """
            SELECT id FROM marketing_approval_tickets
            WHERE campaign_id = $1 AND status IN ('open', 'in_review')
            ORDER BY created_at DESC
            LIMIT 1
            """,
            cid,
        )
        if not tid:
            ch = str(camp["channel"] or "")
            st = str(camp["status"] or "").strip().lower()
            if channel_requires_ticket(ch) and st != "pending_approval":
                # UI "Master approve" expects a one-step resolve; tickets are only auto-created
                # when status is pending_approval, so move outbound campaigns into that queue first.
                if st in ("draft", "paused", "scheduled", "rejected", "active"):
                    await conn.execute(
                        """
                        UPDATE marketing_campaigns
                        SET status = 'pending_approval', updated_at = NOW()
                        WHERE id = $1
                        """,
                        cid,
                    )
                    st = "pending_approval"
            await ensure_open_ticket_for_pending_campaign(
                conn,
                campaign_id=cid,
                submitted_by=master_uuid,
                channel=ch,
                status=st,
            )
            tid = await conn.fetchval(
                """
                SELECT id FROM marketing_approval_tickets
                WHERE campaign_id = $1 AND status IN ('open', 'in_review')
                ORDER BY created_at DESC
                LIMIT 1
                """,
                cid,
            )
        if not tid:
            raise HTTPException(
                400,
                "No marketing approval ticket. Set the campaign to Pending approval first, "
                "or use Operational incidents → Marketing.",
            )
        try:
            await resolve_ticket_master(
                conn,
                ticket_id=tid,
                master_user_id=master_uuid,
                decision="approved",
                notes=None,
            )
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        except LookupError as e:
            raise HTTPException(404, str(e)) from e
        r2 = await conn.fetchrow(
            "SELECT id, status, approved_at FROM marketing_campaigns WHERE id = $1",
            cid,
        )
        out = {
            "ok": True,
            "campaign_id": str(r2["id"]),
            "status": r2["status"],
            "ticket_id": str(tid),
            "approved_at": r2["approved_at"].isoformat() if r2.get("approved_at") else None,
        }
        if bool(payload.get("run_tick")):
            out["execution_tick"] = await run_marketing_execution_tick(conn)
        return out


@marketing_router.get("/approval-tickets")
async def marketing_approval_tickets_list(
    status: Optional[str] = Query(None, description="open | in_review | approved | rejected"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    user: dict = Depends(require_admin),
):
    from services.marketing_approval_tickets import list_tickets

    st = (status or "").strip().lower()[:24] or None
    if st and st not in ("open", "in_review", "approved", "rejected"):
        raise HTTPException(400, "Invalid status filter")
    async with core.state.db_pool.acquire() as conn:
        rows = await list_tickets(conn, status=st, limit=limit, offset=offset)
    return {"items": rows, "count": len(rows)}


@marketing_router.post("/approval-tickets/{ticket_id}/resolve")
async def marketing_approval_ticket_resolve(
    ticket_id: str,
    body: MarketingTicketResolveBody,
    user: dict = Depends(require_master_admin),
):
    try:
        tid = uuid.UUID(ticket_id)
        master_uuid = uuid.UUID(str(user["id"]))
    except ValueError:
        raise HTTPException(400, "Invalid ticket or user id")
    from services.marketing_approval_tickets import resolve_ticket_master

    dec = (body.decision or "").strip().lower()
    if dec not in ("approved", "rejected"):
        raise HTTPException(400, "decision must be approved or rejected")
    try:
        async with core.state.db_pool.acquire() as conn:
            out = await resolve_ticket_master(
                conn,
                ticket_id=tid,
                master_user_id=master_uuid,
                decision=dec,
                notes=body.notes,
            )
    except LookupError:
        raise HTTPException(404, "Ticket or campaign not found")
    except ValueError as e:
        raise HTTPException(400, str(e))
    return out


@marketing_router.post("/campaigns/{campaign_id}/submit-approval-ticket")
async def marketing_submit_approval_ticket(
    campaign_id: str,
    user: dict = Depends(require_admin),
):
    """Create a new open approval ticket (e.g. after rejection or manual resubmit)."""
    try:
        cid = uuid.UUID(campaign_id)
        uid = uuid.UUID(str(user["id"]))
    except ValueError:
        raise HTTPException(400, "Invalid campaign or user id")
    from services.marketing_approval_tickets import submit_ticket_for_campaign

    try:
        async with core.state.db_pool.acquire() as conn:
            out = await submit_ticket_for_campaign(conn, campaign_id=cid, submitted_by=uid)
    except LookupError:
        raise HTTPException(404, "Campaign not found")
    except ValueError as e:
        raise HTTPException(400, str(e))
    return out


@marketing_router.post("/execution/tick")
async def marketing_execution_tick(
    payload: Dict[str, Any] = Body(default_factory=dict),
    user: dict = Depends(require_admin),
):
    max_c = int(payload.get("max_per_campaign") or 30)
    max_t = int(payload.get("max_total_sends") or 120)
    track = OptionalTrackioRun("marketing_execution_tick")
    track.start(config={"max_per_campaign": max_c, "max_total_sends": max_t})
    try:
        async with core.state.db_pool.acquire() as conn:
            summary = await run_marketing_execution_tick(
                conn, max_per_campaign=max_c, max_total_sends=max_t
            )
        logger.info("marketing_execution_tick by %s: %s", user.get("email"), summary.get("run_id"))
        track.log(
            {
                "status": 1,
                "email_sent": int(summary.get("email_sent") or 0),
                "discord_sent": int(summary.get("discord_sent") or 0),
                "in_app_written": int(summary.get("in_app_written") or 0),
                "skipped": int(summary.get("skipped") or 0),
                "errors_n": len(summary.get("errors") or []),
            }
        )
        return summary
    except Exception as e:
        track.log({"status": 0, "error": str(e)[:300]})
        raise
    finally:
        track.finish()


@marketing_router.api_route("/execution/heartbeat", methods=["GET", "POST"])
async def marketing_execution_heartbeat(user: dict = Depends(require_admin)):
    """GET: admin-marketing.html poll for pending approvals. POST: same (worker/cron may POST)."""
    async with core.state.db_pool.acquire() as conn:
        ping = await maybe_alert_pending_approvals(conn, min_interval_hours=2)
        pending = await list_pending_approval_campaigns(conn)
    return {"pending_count": len(pending), "pending": pending[:20], "discord_alert": ping}


@marketing_router.get("/promo-media-defaults")
async def marketing_promo_media_defaults(user: dict = Depends(require_admin)):
    """Suggested ``promo_media`` from aggregate wallet levers + recent marketing-style variant ids."""
    async with core.state.db_pool.acquire() as conn:
        return await infer_promo_media_defaults(conn)


@marketing_router.get("/reports/uplift")
async def marketing_reports_uplift(
    model_key: str = Query("propensity_v1"),
    min_samples: int = Query(40, ge=10, le=5000),
    record_audit: bool = Query(False, description="If true, append a row to ml_model_promotion_audit"),
    user: dict = Depends(require_admin),
):
    mk = str(model_key or "propensity_v1")[:80]
    async with core.state.db_pool.acquire() as conn:
        return await evaluate_uplift_vs_baseline(
            conn, model_key=mk, min_samples=min_samples, record_audit=record_audit
        )


@marketing_router.get("/reports/variant-leaderboard")
async def marketing_reports_variant_leaderboard(
    limit: int = Query(40, ge=1, le=200), user: dict = Depends(require_admin)
):
    async with core.state.db_pool.acquire() as conn:
        rows = await fetch_variant_leaderboard(conn, limit)
    return {"variants": rows}


@marketing_router.get("/reports/conversion-proxy")
async def marketing_reports_conversion_proxy(
    days: int = Query(90, ge=7, le=365),
    limit_rows: int = Query(80, ge=10, le=500),
    user: dict = Depends(require_admin),
):
    async with core.state.db_pool.acquire() as conn:
        return await fetch_marketing_conversion_proxy(conn, days=days, limit_rows=limit_rows)


@marketing_router.get("/reports/bundle")
async def marketing_reports_bundle(
    days: int = Query(90, ge=7, le=365),
    leaderboard_limit: int = Query(40, ge=1, le=200),
    model_key: str = Query("propensity_v1"),
    min_samples: int = Query(40, ge=10, le=5000),
    user: dict = Depends(require_admin),
):
    """Single round-trip for Marketing Ops ML cards (read-only uplift — no audit row)."""
    mk = str(model_key or "propensity_v1")[:80]
    async with core.state.db_pool.acquire() as conn:
        uplift = await evaluate_uplift_vs_baseline(
            conn, model_key=mk, min_samples=min_samples, record_audit=False
        )
        variants = await fetch_variant_leaderboard(conn, leaderboard_limit)
        conv = await fetch_marketing_conversion_proxy(conn, days=days, limit_rows=120)
    return {
        "uplift": uplift,
        "variant_leaderboard": variants,
        "conversion_proxy": conv,
    }


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
        ch_ins = (suggested.get("channel") or "in_app")[:50]
        ch_low = ch_ins.strip().lower()
        # In-app and discount nudges do not use outbound templates; activate immediately so
        # wallet opportunities and promo CTAs match live data without an extra draft step.
        initial_status = "active" if ch_low in ("in_app", "discount") else "draft"
        await conn.execute(
            """
            INSERT INTO marketing_campaigns (
                id, name, objective, channel, status, estimated_audience, schedule_at,
                targeting, notes, created_by, range_key
            )
            VALUES ($1, $2, $3, $4, $10, $5, NULL, $6::jsonb, $7, $8::uuid, $9)
            """,
            cid,
            (suggested.get("name") or "AI campaign")[:500],
            (suggested.get("objective") or objective)[:8000],
            ch_ins,
            120,
            json.dumps(targeting),
            (suggested.get("notes") or "")[:8000],
            str(user["id"]) if user.get("id") else None,
            rk[:32],
            initial_status,
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
    ch_out = str(suggested.get("channel") or "in_app")[:50]
    st_out = "active" if ch_out.strip().lower() in ("in_app", "discount") else "draft"
    campaign = {
        "id": str(cid),
        "name": suggested.get("name") or "AI campaign",
        "objective": suggested.get("objective") or objective,
        "channel": ch_out,
        "status": st_out,
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
    track = OptionalTrackioRun("api_recompute_quality_scores")
    if track.start(config={"lookback_days": int(lookback_days)}):
        track.log({"rows_touched": int(n), "lookback_days": int(lookback_days), "source": "admin_api"})
        track.finish()
    return {"ok": True, "rows_touched": n}


class M8RunRelatedIncidentsBody(BaseModel):
    """Replace ``related_ops_incident_ids`` on an ``m8_model_runs`` row (postmortem linking)."""

    incident_ids: List[str] = Field(default_factory=list)


@ml_router.put("/m8-model-runs/{run_id}/related-incidents")
async def ml_m8_model_run_put_related_incidents(
    run_id: str,
    body: M8RunRelatedIncidentsBody,
    user: dict = Depends(require_admin),
):
    """Attach operational incident UUIDs to an M8 training run (admin UI + external postmortems)."""
    try:
        rid = uuid.UUID(str(run_id).strip())
    except ValueError as exc:
        raise HTTPException(400, "Invalid run_id") from exc
    clean: List[uuid.UUID] = []
    for raw in (body.incident_ids or [])[:48]:
        try:
            clean.append(uuid.UUID(str(raw).strip()))
        except ValueError as exc:
            raise HTTPException(400, f"Invalid incident id: {raw!r}") from exc
    if core.state.db_pool is None:
        raise HTTPException(503, "Database not ready")
    async with core.state.db_pool.acquire() as conn:
        async with conn.transaction():
            status = await conn.execute(
                "UPDATE m8_model_runs SET related_ops_incident_ids = $2::uuid[] WHERE id = $1::uuid",
                rid,
                clean,
            )
            if status == "UPDATE 0":
                raise HTTPException(404, "Model run not found")
            await log_admin_audit(
                conn,
                user_id=str(user.get("id") or ""),
                admin=user,
                action="m8_model_run_related_incidents",
                details={"run_id": str(rid), "related_ops_incident_ids": [str(x) for x in clean]},
            )
    return {"ok": True, "run_id": str(rid), "related_ops_incident_ids": [str(x) for x in clean]}


class EmailJobRunBody(BaseModel):
    job: str = Field(..., min_length=1, max_length=120)


@admin_compat_router.post("/email-jobs/run")
async def email_jobs_run(body: EmailJobRunBody, user: dict = Depends(require_admin)):
    """Run an admin email job on demand.

    Supported job names:
      - trial_reminders
      - monthly_user_digest
      - weekly_admin_digest
      - scheduled_publish_alerts
      - marketing_execution / marketing_touchpoints
      - all  (runs every email job above + marketing_execution)
    """
    job = (body.job or "").strip().lower()
    admin_email = user.get("email") or "?"
    logger.info("email-jobs/run job=%s admin=%s", job, admin_email)

    response: Dict[str, Any] = {"ok": True, "job": job, "ran": []}

    # 1. Marketing tick (kept compatible with the previous behaviour)
    if job in ("marketing_execution", "marketing_touchpoints", "all"):
        try:
            async with core.state.db_pool.acquire() as conn:
                response["marketing_execution"] = await run_marketing_execution_tick(conn)
            response["ran"].append("marketing_execution")
        except Exception as e:
            logger.exception("marketing_execution tick failed: %s", e)
            response["marketing_execution_error"] = str(e)

    # 2. Admin email jobs (the four buttons in the admin UI + 'all')
    if job in ADMIN_EMAIL_JOBS or job == "all":
        try:
            result = await run_admin_email_job(
                core.state.db_pool, job, triggered_by=f"manual:{admin_email}"
            )
            response["email_jobs"] = result
            if job == "all" and isinstance(result.get("ran"), list):
                response["ran"].extend(result["ran"])
            elif "job" in result:
                response["ran"].append(result["job"])
        except Exception as e:
            logger.exception("admin email job %s failed: %s", job, e)
            response["ok"] = False
            response["error"] = str(e)

    if not response["ran"]:
        response["ok"] = False
        response["error"] = f"unknown job: {body.job}"

    return response


@admin_compat_router.post("/platform-kpi-rollups/refresh")
async def platform_kpi_rollups_refresh(days: int = Query(7, ge=1, le=90), user: dict = Depends(require_admin)):
    return {"ok": True, "days": days}


@admin_compat_router.get("/kpi/cost-tracker")
async def kpi_cost_tracker(range: str = Query("30d"), user: dict = Depends(require_admin)):
    async with core.state.db_pool.acquire() as conn:
        return await build_cost_tracker_payload(conn, range_key=range)


@admin_compat_router.get("/kpi/provider-costs")
async def kpi_provider_costs(range: str = Query("30d"), user: dict = Depends(require_admin)):
    async with core.state.db_pool.acquire() as conn:
        return await build_provider_costs_payload(conn, range_key=range)


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


@admin_compat_router.get("/kpi/pikzels-template-render")
async def kpi_pikzels_template_render(range: str = Query("30d"), user: dict = Depends(require_admin)):
    """Share of completed uploads that used PIL template thumbs while PIKZELS_API_KEY is set."""
    since, until = parse_range_since_until(range)
    try:
        async with core.state.db_pool.acquire() as conn:
            data = await fetch_pikzels_template_render_kpi(conn, since, until)
        return {"range": range, **data}
    except Exception as e:
        logger.warning("pikzels-template-render kpi: %s", e)
        return {
            "range": range,
            "pikzels_api_key_configured": bool((resolve_public_api_key() or "").strip()),
            "total_completed_uploads": 0,
            "template_render_count": 0,
            "template_render_pct": 0.0,
            "error": str(e),
        }


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


@public_marketing_router.get("/o/{token}")
async def marketing_email_open_pixel(token: str):
    """1×1 tracking pixel for campaign email opens (signed token)."""
    import base64

    from services.marketing_promo_media import verify_tracking_token
    from services.ml_marketing import record_outcome_label

    px = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMB/ax9pZkAAAAASUVORK5CYII="
    )
    pl = verify_tracking_token(token)
    if not pl or str(pl.get("t")) != "open":
        return Response(content=px, media_type="image/png")
    uid = str(pl.get("u") or "")
    if not uid:
        return Response(content=px, media_type="image/png")
    cid = str(pl.get("c") or "")
    vid = str(pl.get("v") or "")
    did = str(pl.get("d") or "")
    try:
        async with core.state.db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO marketing_events (user_id, event_type, payload)
                VALUES ($1::uuid, 'campaign_email_open', $2::jsonb)
                """,
                uid,
                json.dumps(
                    {
                        "campaign_id": cid,
                        "delivery_id": did,
                        "variant_id": vid,
                        "promo_variant_id": vid,
                    }
                ),
            )
            await record_outcome_label(
                conn,
                user_id=uid,
                upload_id=None,
                variant_id=vid or None,
                feature_snapshot={"campaign_id": cid, "delivery_id": did},
                label_json={"email_open": True},
            )
    except Exception:
        logger.debug("marketing open pixel skip", exc_info=True)
    return Response(content=px, media_type="image/png")


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
        meta = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        cid = str(meta.get("campaign_id") or "").strip()
        if cid and et in ("shown", "clicked", "dismissed", "converted"):
            try:
                from services.ml_marketing import record_outcome_label

                vid = str(meta.get("promo_variant_id") or meta.get("variant_id") or "")[:128]
                await record_outcome_label(
                    conn,
                    user_id=str(user["id"]),
                    upload_id=None,
                    variant_id=vid or None,
                    feature_snapshot={
                        "campaign_id": cid,
                        "wallet_banner_event": et,
                        "nudge_type": payload.get("nudge_type"),
                        "page": payload.get("page"),
                    },
                    label_json={f"wallet_banner_{et}": True},
                )
            except Exception:
                logger.debug("marketing_events ml label skip", exc_info=True)
    return Response(status_code=204)
