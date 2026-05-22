"""Back-fill missing upload diagnostics for already-finished uploads.

Writes synthesized ``hydration_report`` and ``studio_render_report`` into
``uploads.output_artifacts`` so admin trace has deterministic explanations
without requiring re-upload/reprocess.

Default mode is dry-run (no DB writes).
By default only missing reports are targeted; use ``--force`` to regenerate
and overwrite existing synthesized reports too.

Examples:
  python -m tools.backfill_upload_diag_reports --upload-id e0f77697-efbd-4bec-b8fb-fd853bb270c5 --apply
  python -m tools.backfill_upload_diag_reports --user-id 0af99456-1002-49f8-8554-e4d4405e5884 --limit 50 --apply
  python -m tools.backfill_upload_diag_reports --since-hours 720 --limit 200 --apply
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

import asyncpg
from dotenv import load_dotenv

from stages.entitlements import get_entitlements_from_user
from stages.thumbnail_stage import (
    normalize_thumbnail_render_pipeline,
    pikzels_studio_eligible_for_styled_thumbnail,
)
from stages.pikzels_api import studio_renderer_enabled

load_dotenv()

UUID_RE = re.compile(r"^[0-9a-fA-F-]{36}$")


def _json_val(raw: Any) -> Any:
    if raw is None:
        return None
    if isinstance(raw, (dict, list)):
        return raw
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except Exception:
            return raw
    return raw


def _json_obj(raw: Any) -> Dict[str, Any]:
    v = _json_val(raw)
    return dict(v) if isinstance(v, dict) else {}


def _json_list(raw: Any) -> List[Any]:
    v = _json_val(raw)
    return list(v) if isinstance(v, list) else []


def _first_text(*values: Any, limit: int = 280) -> str:
    for v in values:
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            v = " ".join(str(x).strip() for x in v if str(x).strip())
        txt = " ".join(str(v).strip().split())
        if txt:
            return txt[:limit]
    return ""


def _evidence_present(evidence: Dict[str, Any]) -> bool:
    if not isinstance(evidence, dict):
        return False
    for v in evidence.values():
        if isinstance(v, dict) and any(bool(x) for x in v.values()):
            return True
        if isinstance(v, list) and len(v) > 0:
            return True
        if isinstance(v, str) and v.strip():
            return True
        if isinstance(v, (int, float)) and v not in (0, 0.0):
            return True
    return False


def _synth_hydration_report(
    row: Dict[str, Any],
    artifacts: Dict[str, Any],
    *,
    force: bool = False,
) -> Dict[str, Any]:
    existing = _json_obj(artifacts.get("hydration_report"))
    if existing and not force:
        return existing

    payload = _json_obj(artifacts.get("hydration_payload"))
    payload_ev = _json_obj(payload.get("evidence"))
    payload_tags = payload.get("signal_hashtags")
    tags: List[str] = []
    if isinstance(payload_tags, list):
        tags = [str(t).strip() for t in payload_tags if str(t).strip()]
    elif isinstance(payload_tags, str):
        tags = [t for t in re.split(r"[\s,]+", payload_tags) if t]

    anchor = _first_text(
        payload.get("anchor_phrase"),
        row.get("ai_generated_title"),
        row.get("ai_title"),
        row.get("title"),
        row.get("filename"),
        limit=240,
    )
    story = _first_text(
        payload.get("hydration_story"),
        artifacts.get("hydration_story"),
        row.get("ai_generated_caption"),
        row.get("ai_caption"),
        row.get("caption"),
        limit=900,
    )

    ev: Dict[str, Any] = payload_ev if payload_ev else {}
    if not ev:
        ev = {
            "geo": {},
            "osd": {},
            "music": {},
            "speech": {},
            "vision": {},
            "trill": {},
            "source": "backfill:no_hydration_payload",
        }

    present = bool(anchor or story or tags or _evidence_present(ev))

    return {
        "evidence_present": present,
        "used_fallback_anchor": not bool(payload.get("anchor_phrase")),
        "rewrote_caption": False,
        "rewrote_title": False,
        "purged_seed_tags": 0,
        "added_evidence_tags": len(tags),
        "anchor": anchor,
        "hydration_story": story,
        "evidence_tags": tags,
        "evidence": ev,
        "warnings": [
            "synthesized_by_backfill_upload_diag_reports",
            ("from_hydration_payload" if payload else "missing_hydration_payload"),
        ],
    }


def _synth_studio_render_report(
    row: Dict[str, Any],
    artifacts: Dict[str, Any],
    user_prefs: Dict[str, Any],
    user_record: Dict[str, Any],
    *,
    force: bool = False,
) -> Dict[str, Any]:
    existing = _json_obj(artifacts.get("studio_render_report"))
    if existing and not force:
        return existing

    ent = get_entitlements_from_user(user_record or {})
    can_custom = bool(getattr(ent, "can_custom_thumbnails", False))
    can_ai_style = bool(getattr(ent, "can_ai_thumbnail_styling", False))
    styled_enabled = bool(user_prefs.get("styled_thumbnails", user_prefs.get("styledThumbnails", True)))
    auto_thumbnails = bool(user_prefs.get("auto_thumbnails") or user_prefs.get("autoThumbnails"))
    render_pipeline_pref = normalize_thumbnail_render_pipeline(user_prefs)
    ready = bool(studio_renderer_enabled())

    studio_eligible = bool(
        pikzels_studio_eligible_for_styled_thumbnail(
            user_prefs, ent, require_auto_thumbnails=False
        )
    )

    skip_reason = None
    if not can_custom:
        skip_reason = "tier lacks can_custom_thumbnails"
    elif not styled_enabled:
        skip_reason = "user pref styled_thumbnails=false"
    elif render_pipeline_pref == "none":
        skip_reason = "thumbnailRenderPipeline=none"
    elif not ready:
        skip_reason = "pikzels api key missing/disabled"
    elif not studio_eligible:
        skip_reason = "not studio eligible by prefs/entitlements"

    trace_events = _json_list(artifacts.get("thumbnail_trace"))
    trace_names: List[str] = []
    for ev in trace_events:
        if isinstance(ev, dict) and ev.get("event"):
            trace_names.append(str(ev.get("event")))

    platform_map = _json_obj(artifacts.get("platform_thumbnail_map"))
    platform_render_methods: Dict[str, Dict[str, Any]] = {}
    for p in ("youtube", "instagram", "facebook", "tiktok"):
        if p in platform_map:
            platform_render_methods[p] = {
                "rendered": True,
                "source": "platform_thumbnail_map",
                "path": str(platform_map.get(p) or "")[:180],
            }

    hr = _json_obj(artifacts.get("hydration_report"))
    evidence_anchor = _first_text(
        hr.get("anchor"),
        _json_obj(artifacts.get("hydration_payload")).get("anchor_phrase"),
        limit=500,
    )

    return {
        "pikzels_api_key_configured": ready,
        "can_custom_thumbnails": can_custom,
        "can_ai_thumbnail_styling": can_ai_style,
        "styled_thumbnails_pref": styled_enabled,
        "render_pipeline_pref": render_pipeline_pref,
        "auto_thumbnails_pref": auto_thumbnails,
        "studio_eligible": studio_eligible,
        "persona_kind": _first_text(
            user_prefs.get("thumbnail_persona_kind"),
            user_prefs.get("thumbnailPersonaKind"),
            limit=80,
        )
        or None,
        "persona_uuid": _first_text(
            user_prefs.get("thumbnail_pikzels_persona_id"),
            user_prefs.get("thumbnailPikzelsPersonaId"),
            user_prefs.get("thumbnail_default_persona_id"),
            user_prefs.get("thumbnailDefaultPersonaId"),
            limit=80,
        )
        or None,
        "render_steps": trace_names[:24],
        "platform_render_methods": platform_render_methods,
        "hydration_pikzels_edit": _json_obj(artifacts.get("hydration_pikzels_edit")),
        "evidence_anchor": evidence_anchor or None,
        "skip_reason": skip_reason,
        "raw_frame_only": bool(artifacts.get("thumbnail_raw_frame_only", not bool(platform_map))),
        "synthesized": True,
        "synthesized_source": "backfill_upload_diag_reports_v1",
    }


def _iter_target_ids(rows: Iterable[asyncpg.Record]) -> List[str]:
    out: List[str] = []
    for row in rows:
        uid = str(row.get("id") or "")
        if UUID_RE.match(uid):
            out.append(uid)
    return out


async def run(args: argparse.Namespace) -> int:
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        print("ERROR: DATABASE_URL missing")
        return 2

    ssl = "require" if "sslmode=require" in dsn else None
    conn = await asyncpg.connect(dsn=dsn, ssl=ssl)
    updated = 0
    scanned = 0
    try:
        target_ids: List[str] = []
        if args.upload_id:
            target_ids = [u.strip() for u in args.upload_id if u and UUID_RE.match(u.strip())]
        else:
            where: List[str] = []
            params: List[Any] = []
            if args.user_id:
                params.append(args.user_id)
                where.append(f"up.user_id = ${len(params)}::uuid")
            params.append(int(args.since_hours))
            where.append(
                f"COALESCE(up.processing_finished_at, up.updated_at, up.created_at) >= NOW() - make_interval(hours => ${len(params)}::int)"
            )
            if not args.include_processing:
                where.append("up.status IN ('completed','published','ready_to_publish')")

            if not args.force:
                missing_pred = (
                    "(up.output_artifacts IS NULL OR "
                    " NOT (up.output_artifacts ? 'hydration_report') OR "
                    " NOT (up.output_artifacts ? 'studio_render_report'))"
                )
                where.append(missing_pred)
            params.append(int(args.limit))
            q = f"""
                SELECT up.id
                FROM uploads up
                WHERE {' AND '.join(where)}
                ORDER BY COALESCE(up.processing_finished_at, up.updated_at, up.created_at) DESC
                LIMIT ${len(params)}
            """
            rows = await conn.fetch(q, *params)
            target_ids = _iter_target_ids(rows)

        if not target_ids:
            print("No matching uploads.")
            return 0

        print(f"Targets: {len(target_ids)} upload(s)")
        for upload_id in target_ids:
            row = await conn.fetchrow(
                """
                SELECT
                    up.*,
                    u.preferences AS users_preferences,
                    u.subscription_tier,
                    u.role
                FROM uploads up
                JOIN users u ON u.id = up.user_id
                WHERE up.id = $1::uuid
                """,
                upload_id,
            )
            if not row:
                continue
            scanned += 1
            d = dict(row)
            artifacts = _json_obj(d.get("output_artifacts"))
            upload_prefs = _json_obj(d.get("user_preferences"))
            user_prefs = _json_obj(d.get("users_preferences"))
            merged_prefs = dict(user_prefs)
            merged_prefs.update({k: v for k, v in upload_prefs.items() if v is not None})

            user_record = {
                "subscription_tier": d.get("subscription_tier"),
                "role": d.get("role"),
                "flex_enabled": d.get("flex_enabled"),
            }
            hyd = _synth_hydration_report(d, artifacts, force=bool(args.force))
            if not artifacts.get("hydration_story") and hyd.get("hydration_story"):
                artifacts["hydration_story"] = hyd.get("hydration_story")
            artifacts["hydration_report"] = hyd

            srr = _synth_studio_render_report(
                d,
                artifacts,
                merged_prefs,
                user_record,
                force=bool(args.force),
            )
            artifacts["studio_render_report"] = srr

            patch = {
                "hydration_report": hyd,
                "studio_render_report": srr,
            }
            if hyd.get("hydration_story"):
                patch["hydration_story"] = hyd.get("hydration_story")

            before_hr = bool(_json_obj(_json_obj(d.get("output_artifacts")).get("hydration_report")))
            before_srr = bool(_json_obj(_json_obj(d.get("output_artifacts")).get("studio_render_report")))
            print(
                f"- {upload_id} | status={d.get('status')} "
                f"| hydration_report: {'present' if before_hr else 'missing'} -> set "
                f"| studio_render_report: {'present' if before_srr else 'missing'} -> set "
                f"| skip_reason={patch['studio_render_report'].get('skip_reason')!r}"
            )

            if args.apply:
                await conn.execute(
                    """
                    UPDATE uploads
                    SET output_artifacts = COALESCE(output_artifacts, '{}'::jsonb) || $2::jsonb,
                        updated_at = NOW()
                    WHERE id = $1::uuid
                    """,
                    upload_id,
                    json.dumps(patch, default=str),
                )
                updated += 1

        print(
            f"\nDone. scanned={scanned} "
            f"updated={updated} mode={'APPLY' if args.apply else 'DRY-RUN'} "
            f"force={bool(args.force)}"
        )
    finally:
        await conn.close()
    return 0


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Back-fill hydration_report + studio_render_report in uploads.output_artifacts"
    )
    p.add_argument(
        "--upload-id",
        action="append",
        default=[],
        help="Target upload UUID (repeatable)",
    )
    p.add_argument("--user-id", default="", help="Limit to one user UUID")
    p.add_argument("--since-hours", type=int, default=24 * 30, help="Lookback window when not using --upload-id")
    p.add_argument("--limit", type=int, default=100, help="Max uploads to scan when not using --upload-id")
    p.add_argument(
        "--include-processing",
        action="store_true",
        help="Include non-completed uploads too",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Write changes to DB (default is dry-run preview)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Regenerate reports even when hydration_report/studio_render_report already exist",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    code = asyncio.run(run(args))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
