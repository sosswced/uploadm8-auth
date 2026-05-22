"""One-shot diagnostic: dump hydration_report + studio_render_report + persona linkage
for the most recent uploads of a given user, so we can see exactly which gate
fell through ("[hydration] NO evidence...", "[thumb-renderer] skip_reason...",
or "merge_pikzels_thumbnail_persona_id silent miss").

Usage:
    python -m tools.diag_hydration_persona [user_id] [n_uploads]

Defaults: user_id = the one in debug-0d13f7.log, n_uploads = 5.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any, Dict

import asyncpg
from dotenv import load_dotenv

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass

DEFAULT_USER_ID = "0af99456-1002-49f8-8554-e4d4405e5884"


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


def _short(blob: Any, max_chars: int = 600) -> str:
    if blob is None:
        return "<none>"
    if isinstance(blob, (dict, list)):
        s = json.dumps(blob, default=str, indent=2)
    else:
        s = str(blob)
    if len(s) > max_chars:
        return s[:max_chars] + " …"
    return s


async def diag(user_id: str, n_uploads: int) -> None:
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        print("ERROR: DATABASE_URL not set in environment / .env")
        sys.exit(2)

    conn = await asyncpg.connect(dsn=dsn, ssl="require" if "sslmode=require" in dsn else None)
    try:
        print(f"\n=== USER {user_id} ===")
        urow = await conn.fetchrow(
            "SELECT id, email, preferences FROM users WHERE id = $1::uuid",
            user_id,
        )
        if not urow:
            print("user not found")
            return
        print(f"email={urow['email']}")

        prefs = _safe_json(urow["preferences"]) or {}
        relevant_pref_keys = [
            "captionVoice", "caption_voice",
            "captionStyle", "caption_style",
            "captionTone",  "caption_tone",
            "autoCaptions", "auto_captions",
            "autoThumbnails", "auto_thumbnails",
            "styledThumbnails", "styled_thumbnails",
            "aiHashtagsEnabled", "ai_hashtags_enabled",
            "thumbnailPersonaEnabled", "thumbnail_persona_enabled",
            "thumbnailDefaultPersonaId", "thumbnail_default_persona_id",
            "thumbnailPersonaStrength", "thumbnail_persona_strength",
            "thumbnailStudioEnabled", "thumbnail_studio_enabled",
            "thumbnailStudioEngineEnabled", "thumbnail_studio_engine_enabled",
            "thumbnailRenderPipeline", "thumbnail_render_pipeline",
        ]
        print("\n--- relevant users.preferences ---")
        for k in relevant_pref_keys:
            if k in prefs:
                print(f"  {k} = {prefs[k]!r}")

        # ── Persona linkage ───────────────────────────────────────────────
        pid = prefs.get("thumbnail_default_persona_id") or prefs.get("thumbnailDefaultPersonaId")
        print(f"\n--- persona linkage check ---")
        print(f"  thumbnail_default_persona_id = {pid!r}")
        if pid:
            cp = await conn.fetchrow(
                """SELECT id, name, profile_json
                     FROM creator_personas
                    WHERE id = $1::uuid AND user_id = $2::uuid""",
                pid, user_id,
            )
            if not cp:
                print("  ! creator_personas row NOT FOUND (UUID belongs to another user, or was deleted)")
            else:
                print(f"  creator_personas.name = {cp['name']!r}")
                prof = _safe_json(cp["profile_json"]) or {}
                pkz_in_profile = prof.get("pikzels_pikzonality_id") if isinstance(prof, dict) else None
                print(f"  profile_json.pikzels_pikzonality_id = {pkz_in_profile!r}")

                pua = await conn.fetchrow(
                    """SELECT pikzels_pikzonality_id::text AS pkz, status, kind, updated_at, created_at
                         FROM pikzels_user_assets
                        WHERE user_id = $1::uuid AND local_persona_id = $2::uuid AND kind='persona'
                        ORDER BY updated_at DESC NULLS LAST, created_at DESC LIMIT 1""",
                    user_id, pid,
                )
                if not pua:
                    print("  ! pikzels_user_assets row NOT FOUND for this persona — persona is NOT pushed to Pikzels")
                else:
                    print(f"  pikzels_user_assets: status={pua['status']} pkz_uuid={pua['pkz']!r}")
                    if pua["status"] != "linked":
                        print("  ! status != 'linked' — merge_pikzels_thumbnail_persona_id will NOT inject this UUID")
                    if not pua["pkz"]:
                        print("  ! pikzels_pikzonality_id is NULL — Pikzels will not see the persona")

        # ── Recent uploads & their artifacts ──────────────────────────────
        cols = await conn.fetch(
            "SELECT column_name FROM information_schema.columns WHERE table_name='uploads'"
        )
        col_names = {c["column_name"] for c in cols}
        wanted = ["id", "created_at", "status", "filename", "title", "caption", "hashtags",
                  "ai_generated_title", "ai_generated_caption", "ai_generated_hashtags",
                  "output_artifacts", "user_preferences"]
        select_parts = [f"{c}::text AS id" if c == "id" else c for c in wanted if c in col_names]
        rows = await conn.fetch(
            f"""SELECT {', '.join(select_parts)}
                 FROM uploads
                WHERE user_id = $1::uuid
                ORDER BY created_at DESC
                LIMIT $2""",
            user_id, n_uploads,
        )
        print(f"\n--- last {len(rows)} uploads ---")
        for i, row in enumerate(rows, 1):
            d = dict(row)
            arts = _safe_json(d.get("output_artifacts")) or {}
            hr   = _safe_json(arts.get("hydration_report")) if isinstance(arts, dict) else None
            srr  = _safe_json(arts.get("studio_render_report")) if isinstance(arts, dict) else None
            tbj  = _safe_json(arts.get("thumbnail_brief_json")) if isinstance(arts, dict) else None

            print(f"\n  [{i}] {d.get('id')}  created={d.get('created_at')}  status={d.get('status')}")
            print(f"       filename = {d.get('filename')!r}")
            print(f"       title    = {d.get('title')!r}")
            print(f"       caption  = {(d.get('caption') or '')[:80]!r}")
            print(f"       ai_generated_title    = {d.get('ai_generated_title')!r}")
            print(f"       ai_generated_caption  = {(d.get('ai_generated_caption') or '')[:240]!r}")
            print(f"       hashtags              = {d.get('hashtags')}")
            print(f"       ai_generated_hashtags = {d.get('ai_generated_hashtags')}")
            up = _safe_json(d.get('user_preferences')) or {}
            if isinstance(up, dict):
                print(f"       user_preferences keys = {sorted(list(up.keys()))[:30]}")
                interesting = ['captionVoice', 'caption_voice', 'thumbnailDefaultPersonaId',
                               'thumbnail_default_persona_id', 'thumbnailPersonaEnabled',
                               'autoCaptions', 'autoThumbnails', 'styledThumbnails',
                               'aiHashtagsEnabled']
                for k in interesting:
                    if k in up:
                        print(f"         {k} = {up[k]!r}")

            print(f"\n       --- hydration_report ---")
            if not isinstance(hr, dict):
                print(f"       <missing or non-dict>: {hr!r}")
            else:
                print(f"       evidence_present     = {hr.get('evidence_present')}")
                print(f"       rewrote_caption      = {hr.get('rewrote_caption')}")
                print(f"       rewrote_title        = {hr.get('rewrote_title')}")
                print(f"       purged_seed_tags     = {hr.get('purged_seed_tags')}")
                print(f"       added_evidence_tags  = {hr.get('added_evidence_tags')}")
                print(f"       anchor               = {hr.get('anchor')!r}")
                print(f"       evidence_tags        = {hr.get('evidence_tags')}")
                print(f"       warnings             = {hr.get('warnings')}")
                ev = hr.get("evidence") or {}
                if isinstance(ev, dict):
                    print(f"       evidence.geo            = {ev.get('geo')}")
                    print(f"       evidence.speed          = {ev.get('speed')}")
                    print(f"       evidence.osd            = {ev.get('osd')}")
                    print(f"       evidence.music          = {ev.get('music')}")
                    print(f"       evidence.transcript     = {_short(ev.get('transcript'), 200)}")
                    print(f"       evidence.vision         = {_short(ev.get('vision'), 200)}")
                    print(f"       evidence.video_intelligence = {_short(ev.get('video_intelligence'), 200)}")

            print(f"\n       --- studio_render_report ---")
            if not isinstance(srr, dict):
                print(f"       <missing or non-dict>: {srr!r}")
            else:
                print(f"       skip_reason            = {srr.get('skip_reason')}")
                print(f"       studio_eligible        = {srr.get('studio_eligible')}")
                print(f"       persona_kind           = {srr.get('persona_kind')}")
                print(f"       persona_uuid           = {srr.get('persona_uuid')}")
                print(f"       render_steps           = {srr.get('render_steps')}")
                print(f"       platform_render_methods = {_short(srr.get('platform_render_methods'), 240)}")
                print(f"       evidence_anchor        = {srr.get('evidence_anchor')!r}")
                print(f"       pikzels_api_key_configured = {srr.get('pikzels_api_key_configured')}")
                print(f"       can_custom_thumbnails  = {srr.get('can_custom_thumbnails')}")
                print(f"       styled_thumbnails_pref = {srr.get('styled_thumbnails_pref')}")
                print(f"       render_pipeline_pref   = {srr.get('render_pipeline_pref')}")

            print(f"\n       --- thumbnail brief (selected_headline / notes) ---")
            if isinstance(tbj, dict):
                print(f"       selected_headline = {tbj.get('selected_headline')!r}")
                print(f"       headline_options  = {tbj.get('headline_options')}")
                print(f"       badge_text        = {tbj.get('badge_text')!r}")
                print(f"       notes             = {(tbj.get('notes') or '')[:160]!r}")
                print(f"       geo_context       = {(tbj.get('geo_context') or '')[:140]!r}")
                print(f"       osd_context       = {(tbj.get('osd_context') or '')[:140]!r}")
                print(f"       music_context     = {(tbj.get('music_context') or '')[:140]!r}")
                print(f"       speech_context    = {(tbj.get('speech_context') or '')[:140]!r}")
            else:
                print(f"       <missing or non-dict>: {tbj!r}")
    finally:
        await conn.close()


def main() -> None:
    user_id   = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_USER_ID
    n_uploads = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    asyncio.run(diag(user_id, n_uploads))


if __name__ == "__main__":
    main()
