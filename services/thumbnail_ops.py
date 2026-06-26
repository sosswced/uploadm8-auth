"""
Operational alerts for thumbnail / Pikzels pipeline gaps.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("uploadm8-worker.thumbnail_ops")


async def record_pikzels_template_render_incident(
    db_pool: Any,
    *,
    upload_id: str,
    user_id: Optional[str],
    render_method: str,
    studio_report: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Fire when PIKZELS_API_KEY is configured but the upload used PIL template render.
    """
    if (render_method or "").strip().lower() not in ("template", "none", ""):
        return
    try:
        from stages.pikzels_api import studio_renderer_enabled
    except Exception:
        return
    if not studio_renderer_enabled():
        return

    skip_reason = ""
    if isinstance(studio_report, dict):
        skip_reason = str(studio_report.get("skip_reason") or "").strip()
    body_lines = [
        f"upload_id={upload_id}",
        f"thumbnail_render_method={render_method}",
        f"skip_reason={skip_reason or '(styled block skipped or studio ineligible)'}",
    ]
    if isinstance(studio_report, dict):
        try:
            body_lines.append("studio_render_report=" + json.dumps(studio_report, default=str)[:2000])
        except Exception:
            pass

    try:
        from services.ops_incidents import record_operational_incident

        await record_operational_incident(
            db_pool,
            source="thumbnail",
            incident_type="pikzels_template_fallback",
            subject=f"Pikzels key set but template thumbnail for upload {upload_id}",
            body="\n".join(body_lines),
            details={
                "upload_id": str(upload_id),
                "user_id": user_id,
                "thumbnail_render_method": render_method,
                "skip_reason": skip_reason,
                "studio_render_report": studio_report if isinstance(studio_report, dict) else {},
            },
            user_id=user_id,
            upload_id=str(upload_id),
            alert_email=True,
            alert_discord=True,
        )
        logger.warning(
            "[%s] pikzels_template_fallback ops incident (PIKZELS_API_KEY set, render=template) reason=%s",
            upload_id,
            skip_reason or "unknown",
        )
    except Exception as e:
        logger.debug("[%s] pikzels_template_fallback incident skipped: %s", upload_id, e)


async def record_pikzels_render_failures_incident(
    db_pool: Any,
    *,
    upload_id: str,
    user_id: Optional[str],
    output_artifacts: Any,
) -> None:
    """
    Emit one consolidated ops incident when ``pikzels_render_failures`` is present
    on upload output_artifacts (worker thumbnail stage).
    """
    try:
        arts = output_artifacts if isinstance(output_artifacts, dict) else {}
        raw_pf = arts.get("pikzels_render_failures")
        if isinstance(raw_pf, str) and raw_pf.strip():
            failures = json.loads(raw_pf)
        elif isinstance(raw_pf, list):
            failures = raw_pf
        else:
            return
        if not isinstance(failures, list) or not failures:
            return

        from services.ops_incidents import record_operational_incident

        rows = []
        for f in failures:
            if not isinstance(f, dict):
                continue
            plat = str(f.get("platform") or "unknown").lower()
            status_code = f.get("http_status")
            status_label = (
                str(status_code) if status_code not in (None, "", "unknown") else "unknown"
            )
            msg = str(f.get("message") or "")[:1000]
            rows.append(
                {
                    "platform": plat,
                    "http_status": status_code,
                    "http_label": status_label,
                    "message": msg,
                }
            )
        if not rows:
            return

        plat_summary = ",".join(r["platform"] for r in rows[:12])
        lines = "\n".join(
            f"  • {r['platform']}: HTTP {r['http_label']} — {(r['message'] or '')[:280]}"
            for r in rows
        )
        body = lines or "Pikzels studio renderer did not return usable images."
        types_suffix = ":".join(f"{r['platform']}:{r['http_label']}" for r in rows[:4])[:80]
        pikzels_payment_required = all(str(r.get("http_label") or "") == "402" for r in rows)
        await record_operational_incident(
            db_pool,
            source="thumbnail",
            incident_type=(
                f"pikzels_render_failed:{types_suffix}" if types_suffix else "pikzels_render_failed"
            )[:120],
            subject=(f"Pikzels render failed ({len(rows)} platform(s)): {plat_summary}")[:200],
            body=body[:8000],
            details={
                "upload_id": str(upload_id),
                "user_id": str(user_id) if user_id else None,
                "failures": rows,
            },
            user_id=str(user_id) if user_id else None,
            upload_id=str(upload_id),
            alert_email=not pikzels_payment_required,
            alert_discord=not pikzels_payment_required,
        )
    except Exception as e:
        logger.debug("[%s] pikzels failure scan: %s", upload_id, e)


async def record_pikzels_studio_ineligible_incident(
    db_pool: Any,
    *,
    upload_id: str,
    user_id: Optional[str],
    studio_report: Dict[str, Any],
) -> None:
    """Fire when API key is on but studio_eligible was false (prefs/entitlement)."""
    try:
        from stages.pikzels_api import studio_renderer_enabled
    except Exception:
        return
    if not studio_renderer_enabled():
        return
    if studio_report.get("studio_eligible"):
        return
    skip = str(studio_report.get("skip_reason") or "studio ineligible").strip()
    try:
        from services.ops_incidents import record_operational_incident

        await record_operational_incident(
            db_pool,
            source="thumbnail",
            incident_type="pikzels_studio_ineligible",
            subject=f"Pikzels configured but studio skipped for upload {upload_id}",
            body=f"skip_reason={skip}\n{json.dumps(studio_report, default=str)[:1800]}",
            details={
                "upload_id": str(upload_id),
                "user_id": user_id,
                "studio_render_report": studio_report,
            },
            user_id=user_id,
            upload_id=str(upload_id),
            alert_email=False,
            alert_discord=True,
        )
    except Exception as e:
        logger.debug("[%s] pikzels_studio_ineligible incident skipped: %s", upload_id, e)
