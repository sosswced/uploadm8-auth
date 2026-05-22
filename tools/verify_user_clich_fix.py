"""Verify the new hydration enforcer would have rewritten the user's
actual production output. Mirrors the data we saw in
``tools/diag_hydration_persona.py`` for upload 96321a61-… etc.
"""
from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass

from stages.context import JobContext
from services.hydration_enforcer import enforce_hydration


CASES = [
    {
        "upload_id": "96321a61-fb53-4eb3-a239-245f39fdb07f",
        "filename":  "20250301_0058_CAM_EVNT.MP4",
        "ai_title":  "Highway Symphony: A Journey Unfolds 🚗",
        "ai_caption": "Endless road ahead. Watch serenity meet motion.",
    },
    {
        "upload_id": "319df896-a369-404c-aaae-b6d43e938a79",
        "filename":  "20250227_0035_CAM_EVNT.MP4",
        "ai_title":  "Blooming Roads: A Scenic Drive 🌸",
        "ai_caption": "Witness the road transform! Purple blooms stun.",
    },
    {
        "upload_id": "414c52eb-581f-4ec1-a47e-31027cd29acf",
        "filename":  "20250301_0058_CAM_EVNT.MP4",
        "ai_title":  "Open Road Odyssey 🌅🚗",
        "ai_caption": "Cruise under vast skies! Endless horizons await.",
    },
    {
        "upload_id": "2be2c7b3-3a4e-47e7-a441-d9a750b5fc4d",
        "filename":  "20250227_0035_CAM_EVNT.MP4",
        "ai_title":  "Desert Drive: Vibrant Blooms on the Horizon 🌸🚗",
        "ai_caption": "Watch the road transform! Colorful blooms meet desert sands.",
    },
]


def main() -> None:
    print("=" * 78)
    print("Verifying hydration enforcer against the user's actual production output.")
    print("Each case has NO evidence pool (telemetry/Vision/audio empty in DB),")
    print("so the new fallback anchor must rewrite the generic AI clichés.")
    print("=" * 78)
    fails = 0
    for case in CASES:
        ctx = JobContext(job_id="diag", upload_id=case["upload_id"], user_id="0af99456")
        ctx.filename = case["filename"]
        ctx.thumbnail_category = "automotive"
        ctx.ai_title   = case["ai_title"]
        ctx.ai_caption = case["ai_caption"]
        ctx.user_settings = {
            "thumbnail_persona_display_name": "gloc",
            "thumbnailPersonaDisplayName": "gloc",
        }

        report = enforce_hydration(ctx)

        new_title   = ctx.ai_title or ""
        new_caption = ctx.ai_caption or ""
        print()
        print(f"# upload {case['upload_id']}  filename={case['filename']}")
        print(f"  evidence_present     = {report['evidence_present']}")
        print(f"  used_fallback_anchor = {report['used_fallback_anchor']}")
        print(f"  rewrote_caption      = {report['rewrote_caption']}")
        print(f"  rewrote_title        = {report['rewrote_title']}")
        print(f"  anchor               = {report['anchor']!r}")
        print(f"  before title         = {case['ai_title']!r}")
        print(f"  after  title         = {new_title!r}")
        print(f"  before caption       = {case['ai_caption']!r}")
        print(f"  after  caption       = {new_caption!r}")
        if new_caption == case["ai_caption"]:
            print("  !! caption was NOT rewritten — fix did not catch this cliché")
            fails += 1
        if new_title == case["ai_title"]:
            print("  !! title was NOT rewritten — fix did not catch this cliché")
            fails += 1
    print()
    print(f"Result: {fails} miss(es) across {2 * len(CASES)} expected rewrites.")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
