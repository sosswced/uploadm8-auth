#!/usr/bin/env python3
"""Backtest stock motorsport / hype openers under Style×Tone×Voice mixes.

Scans recent uploads for cliché openers (e.g. "Start your engines") and
correlates them with caption creative prefs + hydration wipe reasons.

Usage:
    python -m tools.backtest_prose_hype_cliches
    python -m tools.backtest_prose_hype_cliches --days 45 --limit 80
    python -m tools.backtest_prose_hype_cliches --json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from core.prose_cliche_patterns import backtest_opener_patterns, motorsport_opener_hits


def _j(v: Any) -> Any:
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return v
    return v


def _hits(text: str) -> list[str]:
    t = text or ""
    return [p.pattern for p in backtest_opener_patterns() if p.search(t)]


async def run(*, days: int, limit: int) -> dict[str, Any]:
    import asyncpg
    from core.caption_creative import (
        DEFAULT_CAPTION_STYLE,
        DEFAULT_CAPTION_TONE,
        DEFAULT_CAPTION_VOICE,
        normalize_caption_style,
        normalize_caption_tone,
        normalize_caption_voice,
    )
    from services.m8_grounding_pass import is_formula_stub_caption, persona_voice_required

    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        raise SystemExit("DATABASE_URL missing")

    conn = await asyncpg.connect(dsn)
    try:
        rows = await conn.fetch(
            """
            SELECT id::text AS id, status, title, caption,
                   ai_generated_title, ai_generated_caption,
                   user_preferences, output_artifacts, created_at
            FROM uploads
            WHERE created_at > now() - ($1::text || ' days')::interval
              AND (
                coalesce(title,'') ~* 'start your engine|buckle up|fasten your seatbelt|hit the (road|gas)'
                OR coalesce(caption,'') ~* 'start your engine|buckle up|fasten your seatbelt'
                OR coalesce(ai_generated_title,'') ~* 'start your engine|buckle up|fasten your seatbelt'
                OR coalesce(output_artifacts::text,'') ILIKE '%start your engine%'
              )
            ORDER BY created_at DESC
            LIMIT $2
            """,
            str(int(days)),
            int(limit),
        )

        persona_rows = await conn.fetch(
            """
            SELECT id::text AS id, title, user_preferences, output_artifacts, created_at
            FROM uploads
            WHERE created_at > now() - ($1::text || ' days')::interval
              AND status = 'succeeded'
              AND output_artifacts::text ILIKE '%"persona_required": true%'
            ORDER BY created_at DESC
            LIMIT $2
            """,
            str(min(int(days), 30)),
            int(limit),
        )
    finally:
        await conn.close()

    engines: list[dict[str, Any]] = []
    combo_hits: Counter[str] = Counter()
    for r in rows:
        arts = _j(r["output_artifacts"]) or {}
        if not isinstance(arts, dict):
            arts = {}
        hyd = _j(arts.get("hydration_report")) or {}
        if not isinstance(hyd, dict):
            hyd = {}
        prefs = _j(r["user_preferences"]) or {}
        if not isinstance(prefs, dict):
            prefs = {}
        trace = _j(arts.get("ai_pipeline_trace_v1")) or {}
        caption_payload = None
        if isinstance(trace, dict):
            for e in trace.get("events") or []:
                if e.get("stage") == "caption":
                    caption_payload = e.get("payload")
        before = str(hyd.get("title_before") or "")
        style = normalize_caption_style(
            prefs.get("captionStyle") or prefs.get("caption_style") or ""
        )
        tone = normalize_caption_tone(
            prefs.get("captionTone") or prefs.get("caption_tone") or ""
        )
        voice = normalize_caption_voice(
            prefs.get("captionVoice") or prefs.get("caption_voice") or ""
        )
        combo = f"{style}|{tone}|{voice}"
        blob = " | ".join(
            [
                before,
                str(r["title"] or ""),
                str(r["ai_generated_title"] or ""),
                str((caption_payload or {}).get("ai_title") or "")
                if isinstance(caption_payload, dict)
                else "",
            ]
        )
        hits = _hits(blob)
        if hits:
            combo_hits[combo] += 1
        engines.append(
            {
                "id": r["id"],
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                "status": r["status"],
                "combo": combo,
                "persona_required": persona_voice_required(prefs),
                "defaults": {
                    "style": style == DEFAULT_CAPTION_STYLE,
                    "tone": tone == DEFAULT_CAPTION_TONE,
                    "voice": voice == DEFAULT_CAPTION_VOICE,
                },
                "title_before": before[:160],
                "final_title": str(r["title"] or "")[:160],
                "caption_ai_title": (
                    (caption_payload or {}).get("ai_title")
                    if isinstance(caption_payload, dict)
                    else None
                ),
                "wipe_reason": hyd.get("wipe_reason"),
                "receipt_rejected": hyd.get("receipt_rejected"),
                "generic_rejected": hyd.get("generic_rejected"),
                "before_is_stub": is_formula_stub_caption(before),
                "hype_hits": hits,
            }
        )

    persona_combos: Counter[str] = Counter()
    for r in persona_rows:
        prefs = _j(r["user_preferences"]) or {}
        if not isinstance(prefs, dict):
            prefs = {}
        style = normalize_caption_style(
            prefs.get("captionStyle") or prefs.get("caption_style") or ""
        )
        tone = normalize_caption_tone(
            prefs.get("captionTone") or prefs.get("caption_tone") or ""
        )
        voice = normalize_caption_voice(
            prefs.get("captionVoice") or prefs.get("caption_voice") or ""
        )
        persona_combos[f"{style}|{tone}|{voice}"] += 1

    return {
        "days": days,
        "cliche_upload_count": len(engines),
        "cliche_by_combo": combo_hits.most_common(),
        "persona_required_combos": persona_combos.most_common(15),
        "uploads": engines,
        "notes": [
            "stub/receipt → is_formula_stub_caption; wipe_reason=receipt_rejected.",
            "generic/hype → core.prose_cliche_patterns; wipe_reason=generic_rejected.",
            "freestyle×hype×teacher historically emitted 'Start your engines' (stock hype, not an MPH receipt).",
            f"motorsport sample hits helper: {motorsport_opener_hits('Start your engines!')}",
        ],
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=45)
    ap.add_argument("--limit", type=int, default=80)
    ap.add_argument("--json", action="store_true")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to write full JSON report",
    )
    args = ap.parse_args()
    report = asyncio.run(run(days=args.days, limit=args.limit))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if args.json:
        print(json.dumps(report, indent=2, default=str))
        return
    print(f"days={report['days']} cliche_uploads={report['cliche_upload_count']}")
    print("cliche_by_combo:", report["cliche_by_combo"])
    print("persona_required_combos:", report["persona_required_combos"])
    for u in report["uploads"][:20]:
        print(
            f"- {u['id'][:8]}… combo={u['combo']} before={u['title_before']!r} "
            f"wipe={u['wipe_reason']} hits={u['hype_hits']}"
        )
    for n in report["notes"]:
        print(f"note: {n}")


if __name__ == "__main__":
    main()
