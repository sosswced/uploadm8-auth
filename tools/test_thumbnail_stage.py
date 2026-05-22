"""
Minimal local runner for stages/thumbnail_stage.run_thumbnail_stage.

Usage (from repo root):
  python tools/test_thumbnail_stage.py path/to/video.mp4

Writes JPEG copies under tools/_test_thumbnails/ (or --out-dir):
  thumb_winner.jpg, thumb_youtube.jpg, thumb_raw_best.jpg, thumb_meta.json

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
import json
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


def _configure_console_encoding() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


async def main() -> None:
    _configure_console_encoding()
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
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for thumb_winner.jpg + per-platform JPEGs (default: tools/_test_thumbnails)",
    )
    parser.add_argument(
        "--platforms",
        default="youtube,tiktok,instagram,facebook",
        help="Comma-separated platforms for styled thumbnail render",
    )
    args = parser.parse_args()

    video = args.video.resolve()
    if not video.exists():
        print(f"ERROR: file not found: {video}")
        sys.exit(1)

    from stages.context import JobContext
    from stages.entitlements import get_entitlements_for_tier
    from stages.errors import SkipStage, ThumbnailError
    from stages.thumbnail_stage import run_thumbnail_stage
    from tools.thumbnail_output_helpers import persist_thumbnail_outputs, print_thumbnail_outputs

    out_dir = (args.out_dir or (Path(__file__).resolve().parent / "_test_thumbnails")).resolve()
    platforms = [p.strip().lower() for p in args.platforms.split(",") if p.strip()] or ["youtube"]

    ctx = None
    saved: dict = {}
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
                platforms=platforms,
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
            except ThumbnailError as e:
                if not (getattr(ctx, "thumbnail_path", None) and Path(ctx.thumbnail_path).exists()):
                    print(f"FAIL: {e}")
                    sys.exit(3)
                print(f"WARN: {e}")
                print("     (raw/styled files exist — persisting outputs anyway)")

            saved = persist_thumbnail_outputs(ctx, out_dir)
            if not saved:
                print(
                    "No thumbnail files to copy. "
                    f"thumbnail_path={ctx.thumbnail_path!r} artifact={ctx.output_artifacts.get('thumbnail')!r}"
                )
                sys.exit(3)

            print(f"\nOK: thumbnails written under {out_dir}")
            print_thumbnail_outputs(saved, ctx)
            meta_path = out_dir / "thumb_meta.json"
            if meta_path.exists():
                print(f"\nMeta: {meta_path}")
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    if meta.get("headline"):
                        print(f"  headline: {meta['headline']}")
                except json.JSONDecodeError:
                    pass
    finally:
        pass


if __name__ == "__main__":
    asyncio.run(main())
