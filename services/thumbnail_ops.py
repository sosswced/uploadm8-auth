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
