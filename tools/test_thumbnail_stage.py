"""
Minimal local runner for stages/thumbnail_stage.run_thumbnail_stage.

Usage (from repo root):
  python tools/test_thumbnail_stage.py path/to/video.mp4

Writes a copy of the best thumbnail JPEG next to the script:
  tools/_test_thumb_output.jpg

Requires:
  - OPENAI_API_KEY optional (sharpest-frame fallback works without GPT vision)
  - FFmpeg/ffprobe on PATH
  - Video file readable by FFmpeg

JobContext fields that matter for this stage:
  - processed_video_path or local_video_path
  - temp_dir, entitlements (max_thumbnails / should_generate_thumbnails)
  - user_settings: auto_thumbnails / autoThumbnails = true
  - platforms (for styled path); optional audio_context for mood (else category from title/filename)
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path, help="Path to a video file")
    parser.add_argument(
        "--tier",
        default="creator_pro",
        help="Entitlements tier slug (use creator_pro/studio — 'pro' alone maps to free)",
    )
    parser.add_argument(
        "--no-styled",
        action="store_true",
        help="Disable styled-thumbnails branch (faster smoke test)",
    )
    args = parser.parse_args()

    video = args.video.resolve()
    if not video.exists():
        print(f"ERROR: file not found: {video}")
        sys.exit(1)

    from stages.context import JobContext
    from stages.entitlements import get_entitlements_for_tier
    from stages.errors import SkipStage
    from stages.thumbnail_stage import run_thumbnail_stage

    out = Path(__file__).resolve().parent / "_test_thumb_output.jpg"

    ctx = None
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            us = {
                "auto_thumbnails": True,
                "autoThumbnails": True,
                "styled_thumbnails": False if args.no_styled else True,
                "styledThumbnails": False if args.no_styled else True,
            }
            ctx = JobContext(
                job_id="tool-thumb-1",
                upload_id="tool-upload-2",
                user_id="tool-user-1",
                idempotency_key="tool-thumb-1",
                filename=video.name,
                platforms=["youtube"],
                title="Test video",
                local_video_path=video,
                temp_dir=tmp_path,
                entitlements=get_entitlements_for_tier(args.tier),
                user_settings=us,
                audio_context={
                    "thumbnail_mood": "bold_dramatic",
                    "emotional_tone": "hype_energetic",
                    "category": "general",
                },
            )

            try:
                ctx = await run_thumbnail_stage(ctx)
            except SkipStage as e:
                print(f"SKIP: {e.reason}")
                sys.exit(2)

            # Must copy before temp_dir is deleted — path may be str or Path
            raw = ctx.thumbnail_path or ctx.output_artifacts.get("thumbnail")
            thumb_file = Path(raw) if raw else None
            if thumb_file and thumb_file.exists():
                shutil.copy2(thumb_file, out)
                print(f"\nOK: best frame → {out}")
                print(f"     selection: {ctx.output_artifacts.get('thumbnail_selection_method', '?')}")
            else:
                print(
                    "No thumbnail_path on context (or file missing before copy). "
                    f"thumbnail_path={ctx.thumbnail_path!r} artifact={ctx.output_artifacts.get('thumbnail')!r}"
                )
                sys.exit(3)
    finally:
        try:
            from stages.playwright_stage import close_browser

            await close_browser()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
