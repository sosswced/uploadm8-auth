"""Diagnostic: dump Pikzels / studio renderer status for the most recent partial uploads.

Reads ``output_artifacts.studio_render_report`` and ``output_artifacts.thumbnail_trace``
from the latest ``status='partial'`` (or any non-completed) uploads in the DB so we
can see exactly which gate fell through and why per-platform thumbnails fell back
to template / black-bar covers.

Usage:
    python -m tools.diag_pikzels_latest [n_uploads]

Defaults: n_uploads = 5. Pulls regardless of user_id (whole-table sweep).
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Optional

import asyncpg
from dotenv import load_dotenv

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass


def _safe_json(blob: Any) -> Any:
    if blob is None:
        return None
    if isinstance(blob, (dict, list)):
        return blob
    if isinstance(blob, str):
        try:
            return json.loads(blob)
        except Exception:
            return blob
    return blob


def _short(blob: Any, max_chars: int = 1200) -> str:
    if blob is None:
        return "<none>"
    if isinstance(blob, (dict, list)):
        s = json.dumps(blob, default=str, indent=2)
    else:
        s = str(blob)
    if len(s) > max_chars:
        return s[:max_chars] + " …"
    return s


def _platform_render_summary(report: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    pm = report.get("platform_render_methods") if isinstance(report, dict) else None
    if isinstance(pm, dict):
        for platform, info in pm.items():
            if not isinstance(info, dict):
                continue
            attempted = info.get("attempted") or []
            succeeded_with = info.get("succeeded_with")
            out.append(
                f"  - {platform}: succeeded_with={succeeded_with!r} attempted={attempted}"
            )
    return out


def _thumbnail_trace_pikzels_events(trace: Any) -> List[Dict[str, Any]]:
    """Pull pikzels-related events from a thumbnail_trace artifact."""
    events: List[Dict[str, Any]] = []
    if isinstance(trace, dict):
        ev_list = trace.get("events") or trace.get("trace") or []
        if isinstance(ev_list, list):
            for e in ev_list:
                if not isinstance(e, dict):
                    continue
                key = str(e.get("event") or e.get("name") or "").lower()
                if "pikzels" in key or "studio" in key or "render" in key:
                    events.append(e)
    elif isinstance(trace, list):
        for e in trace:
            if not isinstance(e, dict):
                continue
            key = str(e.get("event") or e.get("name") or "").lower()
            if "pikzels" in key or "studio" in key or "render" in key:
                events.append(e)
    return events


async def diag(n_uploads: int) -> None:
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        print("DATABASE_URL not set", file=sys.stderr)
        sys.exit(2)

    conn = await asyncpg.connect(dsn)
    try:
        rows = await conn.fetch(
            """
            SELECT id, user_id, status, platforms, filename, created_at,
                   thumbnail_r2_key, output_artifacts
            FROM uploads
            WHERE status IN ('partial', 'failed', 'completed', 'succeeded')
              AND output_artifacts IS NOT NULL
            ORDER BY created_at DESC
            LIMIT $1
            """,
            int(n_uploads),
        )
    finally:
        await conn.close()

    if not rows:
        print("No completed/partial/failed uploads with output_artifacts found.")
        return

    print(f"=== Pikzels render diagnosis (latest {len(rows)} uploads) ===\n")
    for row in rows:
        arts = _safe_json(row.get("output_artifacts")) or {}
        if not isinstance(arts, dict):
            arts = {}

        srr = _safe_json(arts.get("studio_render_report")) or {}
        if not isinstance(srr, dict):
            srr = {}
        tt = _safe_json(arts.get("thumbnail_trace"))
        ppp = _safe_json(arts.get("pikzels_prompt_by_platform")) or {}
        ptmap = _safe_json(arts.get("platform_thumbnail_map")) or {}

        print("─" * 80)
        print(f"upload_id={row['id']}  user={row['user_id']}  status={row['status']}")
        print(f"file={row.get('filename')!r}  created={row.get('created_at')}")
        print(f"platforms={list(row.get('platforms') or [])}")
        print(f"thumbnail_r2_key={row.get('thumbnail_r2_key')!r}")
        print()
        print("studio_render_report:")
        print(f"  pikzels_api_key_configured = {srr.get('pikzels_api_key_configured')!r}")
        print(f"  studio_eligible            = {srr.get('studio_eligible')!r}")
        print(f"  can_custom_thumbnails      = {srr.get('can_custom_thumbnails')!r}")
        print(f"  can_ai_thumbnail_styling   = {srr.get('can_ai_thumbnail_styling')!r}")
        print(f"  styled_thumbnails_pref     = {srr.get('styled_thumbnails_pref')!r}")
        print(f"  render_pipeline_pref       = {srr.get('render_pipeline_pref')!r}")
        print(f"  auto_thumbnails_pref       = {srr.get('auto_thumbnails_pref')!r}")
        print(f"  persona_kind / persona_uuid = {srr.get('persona_kind')!r} / {srr.get('persona_uuid')!r}")
        print(f"  evidence_anchor            = {srr.get('evidence_anchor')!r}")
        print(f"  render_steps               = {srr.get('render_steps')!r}")
        print(f"  raw_frame_only             = {srr.get('raw_frame_only')!r}")
        print(f"  skip_reason                = {srr.get('skip_reason')!r}")
        for line in _platform_render_summary(srr):
            print(line)

        if isinstance(ptmap, dict) and ptmap:
            print()
            print(f"platform_thumbnail_map: {list(ptmap.keys())}")

        if isinstance(ppp, dict) and ppp:
            print()
            print("pikzels_prompt_by_platform:")
            for plat, prompt in ppp.items():
                if not isinstance(prompt, str):
                    continue
                snippet = prompt[:280].replace("\n", " ")
                print(f"  - {plat}: {snippet} …" if len(prompt) > 280 else f"  - {plat}: {snippet}")

        events = _thumbnail_trace_pikzels_events(tt)
        if events:
            print()
            print(f"thumbnail_trace pikzels-related events ({len(events)}):")
            for e in events[:12]:
                print(f"  - {_short(e, 360)}")
        print()

    print("─" * 80)
    print("Done.")


def _parse_args(argv: List[str]) -> int:
    n = 5
    if len(argv) >= 2:
        try:
            n = max(1, min(int(argv[1]), 50))
        except ValueError:
            print(f"Invalid n_uploads={argv[1]!r}, using default 5", file=sys.stderr)
    return n


if __name__ == "__main__":
    n_uploads = _parse_args(sys.argv)
    try:
        asyncio.run(diag(n_uploads))
    except KeyboardInterrupt:
        pass
