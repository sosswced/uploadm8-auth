#!/usr/bin/env python3
"""
One-shot TUP-style proof upload with randomized creative knobs.

Picks:
  - random .mp4+.map from E2E_MEDIA_LIBRARY (TUP default PNW folder)
  - random single platform (tiktok|youtube|instagram|facebook)
  - random caption style / tone / voice (non-default preferred)

Then runs the live demo journey, waits for terminal status, and asserts:
  - titles/captions are NOT receipt templates under the persona oracle
  - every fact_ledger_v1.publishable class is cited in title∪caption∪tags

Usage:
  python scripts/agent/tup_random_voice_proof.py --json
  python scripts/agent/tup_random_voice_proof.py --json --skip-pikzels --headless
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except Exception:
    pass

ARTIFACTS = ROOT / "tests" / "e2e" / "artifacts"
_DEFAULT_MEDIA_LIBRARY = r"G:\My Drive\pnw 256\F\F\Normal"
PLATFORMS = ("tiktok", "youtube", "instagram", "facebook")


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pick_creative() -> dict[str, str]:
    from core.caption_creative import (
        CAPTION_STYLES,
        CAPTION_TONES,
        CAPTION_VOICES,
        DEFAULT_CAPTION_STYLE,
        DEFAULT_CAPTION_TONE,
        DEFAULT_CAPTION_VOICE,
    )

    # Prefer non-default knobs so persona_voice_required is True (proof of the fix).
    styles = [s for s in CAPTION_STYLES if s != DEFAULT_CAPTION_STYLE] or list(CAPTION_STYLES)
    tones = [t for t in CAPTION_TONES if t != DEFAULT_CAPTION_TONE] or list(CAPTION_TONES)
    voices = [v for v in CAPTION_VOICES if v != DEFAULT_CAPTION_VOICE] or list(CAPTION_VOICES)
    return {
        "captionStyle": random.choice(styles),
        "captionTone": random.choice(tones),
        "captionVoice": random.choice(voices),
    }


def _configure_env(
    *,
    platform: str,
    creative: dict[str, str],
    video: Path,
    telemetry: Path | None,
    skip_pikzels: bool,
    headless: bool,
) -> None:
    os.environ["E2E_TUP"] = "1"
    os.environ["E2E_UPLOAD_PLATFORMS"] = platform
    os.environ["E2E_USE_PERSONA"] = "1"
    os.environ["E2E_SKIP_MUTATIONS"] = "1"
    os.environ["E2E_WORKER_SAFE"] = "1"
    os.environ["E2E_YOUTUBE_COPYRIGHT_TRIM"] = "1"
    os.environ["RATE_LIMIT_LOOPBACK_BYPASS"] = "1"
    os.environ["E2E_HEADED"] = "0" if headless else "1"
    if skip_pikzels:
        os.environ["E2E_SKIP_PIKZELS"] = "1"
    lib = (os.environ.get("E2E_MEDIA_LIBRARY") or "").strip()
    if not lib and Path(_DEFAULT_MEDIA_LIBRARY).is_dir():
        lib = _DEFAULT_MEDIA_LIBRARY
    if lib:
        os.environ["E2E_MEDIA_LIBRARY"] = lib
    os.environ["E2E_TEST_VIDEO"] = str(video)
    if telemetry:
        os.environ["E2E_TEST_TELEMETRY_MAP"] = str(telemetry)
    else:
        os.environ.pop("E2E_TEST_TELEMETRY_MAP", None)
    # Pin creative knobs for preference POST + DB proof.
    os.environ["E2E_CAPTION_STYLE"] = creative["captionStyle"]
    os.environ["E2E_CAPTION_TONE"] = creative["captionTone"]
    os.environ["E2E_CAPTION_VOICE"] = creative["captionVoice"]
    base = (os.environ.get("E2E_BASE_URL") or "").strip().rstrip("/")
    if not (
        base.startswith("http://127.")
        or base.startswith("http://localhost")
        or base.startswith("http://[::1]")
    ):
        os.environ["E2E_BASE_URL"] = "http://127.0.0.1:8000"


def _apply_creative_prefs(base_url: str, creative: dict[str, str], log_note) -> dict[str, Any]:
    from tests.e2e.helpers.auth import api_client, close_api_client

    payload = {
        **creative,
        "caption_style": creative["captionStyle"],
        "caption_tone": creative["captionTone"],
        "caption_voice": creative["captionVoice"],
        "randomizeCaptionCreative": False,
        "randomize_caption_creative": False,
        "captionCreativePickMode": "off",
        "caption_creative_pick_mode": "off",
        "autoCaptions": True,
        "auto_captions": True,
        "aiServiceCaptionWriter": True,
        "ai_service_caption_writer": True,
    }
    client = api_client()
    try:
        r = client.post("/api/settings/preferences", json=payload)
        ok = r.status_code < 400
        log_note(
            f"Creative prefs POST → HTTP {r.status_code} "
            f"style={creative['captionStyle']} tone={creative['captionTone']} "
            f"voice={creative['captionVoice']}"
        )
        body: Any
        try:
            body = r.json()
        except Exception:
            body = (r.text or "")[:400]
        return {"ok": ok, "status": r.status_code, "body": body, "payload": payload}
    finally:
        close_api_client(client)


def _parse_jsonish(val: Any) -> Any:
    if isinstance(val, str) and val[:1] in "{[":
        try:
            return json.loads(val)
        except Exception:
            return val
    return val


async def _fetch_upload_proof(upload_id: str) -> dict[str, Any]:
    import asyncpg
    from services.fact_ledger import FactEntry, FactLedger
    from services.m8_grounding_pass import is_formula_stub_caption, persona_voice_required

    url = os.environ.get("DATABASE_URL") or ""
    if not url:
        return {"ok": False, "error": "DATABASE_URL missing"}
    conn = await asyncpg.connect(url)
    try:
        row = await conn.fetchrow(
            """
            SELECT id, status, title, caption, hashtags,
                   ai_generated_title, ai_generated_caption, ai_generated_hashtags,
                   platforms, user_preferences, output_artifacts, created_at, completed_at
            FROM uploads WHERE id=$1::uuid
            """,
            upload_id,
        )
        if not row:
            return {"ok": False, "error": "upload_not_found", "upload_id": upload_id}
        data = dict(row)
        for k in ("user_preferences", "output_artifacts", "platforms"):
            data[k] = _parse_jsonish(data.get(k))

        title = str(data.get("title") or data.get("ai_generated_title") or "")
        caption = str(data.get("caption") or data.get("ai_generated_caption") or "")
        tags_raw = data.get("hashtags") or data.get("ai_generated_hashtags") or []
        tags_raw = _parse_jsonish(tags_raw)
        if not isinstance(tags_raw, (list, tuple)):
            tags_raw = [tags_raw] if tags_raw else []
        tags = [str(t).lstrip("#") for t in tags_raw if t]

        prefs = data.get("user_preferences") or {}
        if not isinstance(prefs, dict):
            prefs = {}
        art = data.get("output_artifacts") or {}
        if not isinstance(art, dict):
            art = {}

        m8_titles = _parse_jsonish(art.get("m8_platform_titles") or {}) or {}
        m8_caps = _parse_jsonish(art.get("m8_platform_captions") or {}) or {}
        m8_tags = _parse_jsonish(art.get("m8_platform_hashtags") or {}) or {}
        hyd = _parse_jsonish(art.get("hydration_report") or {}) or {}
        fl_raw = _parse_jsonish(art.get("fact_ledger_v1") or {}) or {}
        if not isinstance(m8_titles, dict):
            m8_titles = {}
        if not isinstance(m8_caps, dict):
            m8_caps = {}
        if not isinstance(m8_tags, dict):
            m8_tags = {}
        if not isinstance(hyd, dict):
            hyd = {}
        if not isinstance(fl_raw, dict):
            fl_raw = {}

        pl_titles = [str(t or "") for t in m8_titles.values()]
        pl_caps = [str(c or "") for c in m8_caps.values()]
        pl_tag_lists: list[str] = []
        for tv in m8_tags.values():
            if isinstance(tv, (list, tuple)):
                pl_tag_lists.extend(str(x).lstrip("#") for x in tv if x)
            elif tv:
                pl_tag_lists.append(str(tv).lstrip("#"))
        cite_title = " ".join([title] + pl_titles)
        cite_caption = " ".join([caption] + pl_caps)
        cite_tags = list(dict.fromkeys(tags + pl_tag_lists))

        title_receipt = is_formula_stub_caption(title)
        caption_receipt = is_formula_stub_caption(caption)
        platform_checks: dict[str, Any] = {}
        for pl, t in m8_titles.items():
            platform_checks[f"{pl}_title"] = {
                "text": str(t or "")[:200],
                "is_receipt": is_formula_stub_caption(str(t or "")),
            }
        for pl, c in m8_caps.items():
            platform_checks[f"{pl}_caption"] = {
                "text": str(c or "")[:240],
                "is_receipt": is_formula_stub_caption(str(c or "")),
            }

        any_platform_receipt = any(bool(v.get("is_receipt")) for v in platform_checks.values())
        persona = persona_voice_required(prefs)

        publishable: list[str] = []
        missing_classes: list[str] = []
        ledger_ok = True
        if fl_raw:
            facts_in: dict[str, FactEntry] = {}
            raw_facts = fl_raw.get("facts") or {}
            if isinstance(raw_facts, dict):
                for cls, meta in raw_facts.items():
                    if not isinstance(meta, dict):
                        continue
                    if not meta.get("publishable", True):
                        continue
                    val = str(meta.get("value") or "").strip()
                    if not val:
                        continue
                    facts_in[str(cls)] = FactEntry(
                        cls=str(cls),
                        value=val,
                        source=str(meta.get("source") or ""),
                        confidence=str(meta.get("confidence") or "high"),
                        publishable=True,
                        slug=str(meta.get("slug") or ""),
                    )
            ledger = FactLedger(
                version=int(fl_raw.get("version") or 1),
                facts=facts_in,
                upload_id=str(upload_id),
            )
            publishable = ledger.publishable_classes() if facts_in else list(
                fl_raw.get("publishable") or []
            )
            missing_classes = ledger.missing_in_text(
                cite_title,
                cite_caption,
                cite_tags,
                include_hashtags=True,
            )
            ledger_ok = not missing_classes
        else:
            ledger_ok = False
            missing_classes = ["fact_ledger_v1_missing"]

        status = str(data.get("status") or "")
        terminal = status in (
            "succeeded",
            "completed",
            "failed",
            "error",
            "cancelled",
            "canceled",
        ) or status.startswith("succeed")
        voice_ok = (not title_receipt) and (not caption_receipt) and (not any_platform_receipt)
        proof_ok = bool(
            terminal and voice_ok and ledger_ok and status.startswith("succeed")
        )
        return {
            "ok": True,
            "upload_id": upload_id,
            "status": status,
            "terminal": terminal,
            "persona_required": persona,
            "title": title,
            "caption": caption,
            "hashtags": cite_tags[:40],
            "title_is_receipt": title_receipt,
            "caption_is_receipt": caption_receipt,
            "platform_checks": platform_checks,
            "hydration_persona_required": hyd.get("persona_required"),
            "hydration_receipt_rejected": hyd.get("receipt_rejected"),
            "hydration_title_trail": {
                "title_before": hyd.get("title_before"),
                "title_after": hyd.get("title_after"),
                "wipe_reason": hyd.get("wipe_reason"),
            },
            "fact_ledger": {
                "present": bool(fl_raw),
                "publishable": publishable,
                "missing_in_title_caption_tags": missing_classes,
                "cite_ok": ledger_ok,
            },
            "voice_ok": voice_ok,
            "ledger_ok": ledger_ok,
            "proof_ok": proof_ok,
            "prefs_snapshot": {
                "captionStyle": prefs.get("captionStyle") or prefs.get("caption_style"),
                "captionTone": prefs.get("captionTone") or prefs.get("caption_tone"),
                "captionVoice": prefs.get("captionVoice") or prefs.get("caption_voice"),
            },
        }
    finally:
        await conn.close()


def _poll_proof(upload_ids: list[str], *, timeout_min: int) -> dict[str, Any]:
    deadline = time.time() + max(60, timeout_min * 60)
    last: dict[str, Any] = {"ok": False, "error": "no_upload_ids"}
    ids = [str(u) for u in upload_ids if u]
    if not ids:
        return last
    while time.time() < deadline:
        for uid in ids:
            last = asyncio.run(_fetch_upload_proof(uid))
            if last.get("terminal"):
                return last
        time.sleep(20)
    last["timed_out"] = True
    return last


def _verdict(proof: dict[str, Any]) -> str:
    if proof.get("proof_ok"):
        return "PASS — non-receipt + all fact_ledger publishable classes cited"
    if proof.get("ok") and not proof.get("terminal"):
        return "PENDING — upload not terminal yet"
    if proof.get("terminal") and proof.get("voice_ok") and not proof.get("ledger_ok"):
        miss = (proof.get("fact_ledger") or {}).get("missing_in_title_caption_tags") or []
        return f"FAIL — fact_ledger cite gaps: {miss}"
    if proof.get("terminal") and not proof.get("voice_ok"):
        return "FAIL — receipt template still present"
    return "FAIL — receipt template still present or upload failed"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Random TUP voice/title proof upload")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--skip-pikzels", action="store_true", default=True)
    parser.add_argument("--force-pikzels", action="store_true")
    parser.add_argument("--pipeline-timeout-min", type=int, default=90)
    parser.add_argument("--platform", default="", help="Force platform (else random)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    if args.seed:
        random.seed(args.seed)
    else:
        random.seed()

    skip_pikzels = bool(args.skip_pikzels) and not args.force_pikzels
    platform = (args.platform or "").strip().lower() or random.choice(PLATFORMS)
    creative = _pick_creative()

    from tests.e2e.helpers.media_library import (
        clear_media_pair_cache,
        e2e_media_library,
        pick_random_media_pair,
    )

    lib = e2e_media_library()
    if not lib and Path(_DEFAULT_MEDIA_LIBRARY).is_dir():
        os.environ["E2E_MEDIA_LIBRARY"] = _DEFAULT_MEDIA_LIBRARY
        lib = e2e_media_library()
    clear_media_pair_cache()
    pair = pick_random_media_pair(force_new=True)
    if not pair:
        out = {
            "ok": False,
            "error": "no_media_pair",
            "library": str(lib) if lib else None,
            "hint": "Set E2E_MEDIA_LIBRARY to the TUP PNW folder",
        }
        print(json.dumps(out, indent=2 if args.json else None))
        return 2
    video, telemetry = pair

    _configure_env(
        platform=platform,
        creative=creative,
        video=video,
        telemetry=telemetry,
        skip_pikzels=skip_pikzels,
        headless=args.headless,
    )

    from tests.e2e.helpers.api_ready import wait_for_api_ready
    from tests.e2e.helpers.browser_session import single_browser_session
    from tests.e2e.helpers.config import e2e_base_url
    from tests.e2e.helpers.live_demo import LiveDemoLog, run_live_demo_journey

    base = e2e_base_url()
    ready = wait_for_api_ready(base, timeout_s=120.0, require_db=True)
    if not ready.get("ok"):
        out = {"ok": False, "error": "api_not_ready", "ready": ready}
        print(json.dumps(out, indent=2))
        return 2

    plan = {
        "started_at": _utc(),
        "base_url": base,
        "library": str(lib) if lib else os.environ.get("E2E_MEDIA_LIBRARY"),
        "video": str(video),
        "telemetry": str(telemetry) if telemetry else None,
        "platform": platform,
        "creative": creative,
        "skip_pikzels": skip_pikzels,
        "pipeline_timeout_min": args.pipeline_timeout_min,
    }
    print(json.dumps({"phase": "plan", **plan}, indent=2), flush=True)

    log = LiveDemoLog()
    journey: dict[str, Any] = {}
    prefs_result = _apply_creative_prefs(base, creative, log.note)

    with single_browser_session(headed=not args.headless) as page:
        journey = run_live_demo_journey(
            page,
            base,
            video=video,
            telemetry=telemetry,
            pipeline_timeout_s=float(args.pipeline_timeout_min) * 60.0,
            api_per_page=1,
            include_slow_api=False,
            skip_api_smoke=True,
            log=log,
        )

    upload_ids = list(journey.get("upload_ids") or [])
    proof = _poll_proof(upload_ids, timeout_min=max(5, args.pipeline_timeout_min))

    out = {
        "ok": bool(proof.get("proof_ok")),
        "generated_at": _utc(),
        "plan": plan,
        "prefs_apply": {
            "ok": prefs_result.get("ok"),
            "status": prefs_result.get("status"),
        },
        "journey": {
            "ok": journey.get("ok"),
            "upload_ids": upload_ids,
            "steps_tail": (log.steps[-12:] if getattr(log, "steps", None) else None),
        },
        "proof": proof,
        "verdict": _verdict(proof),
    }

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = ARTIFACTS / f"tup_random_voice_proof_{stamp}.json"
    path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    out["artifact"] = str(path)

    print(json.dumps(out, indent=2, default=str))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
