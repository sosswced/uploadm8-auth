"""
Minimal local runner for stages/audio_stage.run_audio_context_stage.

Usage (from repo root):
  python tools/test_audio_stage.py path/to/video.mp4

Requires:
  - OPENAI_API_KEY (or set in .env)
  - AUDIO_STAGE_ENABLED=true (default)
  - Video with an audio stream, duration between AUDIO_MIN_DURATION_SECS and AUDIO_MAX_DURATION_SECS

JobContext fields that matter for this stage:
  - platform_videos | processed_video_path | local_video_path  → source file
  - video_info: { "audio_codec", "duration" }  → from ffprobe below
  - temp_dir, user_settings (use_audio_context, audio_transcription), entitlements
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
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


def _ffprobe_video(path: Path) -> dict:
    """Best-effort probe; audio_stage reads video_info like transcode_stage."""
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_streams",
                "-show_format",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if proc.returncode != 0:
            return {"audio_codec": "aac", "duration": 30.0}
        data = json.loads(proc.stdout)
        dur = float((data.get("format") or {}).get("duration") or 30.0)
        ac = ""
        for s in data.get("streams") or []:
            if s.get("codec_type") == "audio":
                ac = s.get("codec_name") or ""
                break
        return {"audio_codec": ac, "duration": dur}
    except Exception:
        return {"audio_codec": "aac", "duration": 30.0}


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path, help="Path to an MP4 (or other FFmpeg-readable) file")
    parser.add_argument("--tier", default="pro", help="Entitlements tier slug for get_entitlements_for_tier")
    args = parser.parse_args()

    video = args.video.resolve()
    if not video.exists():
        print(f"ERROR: file not found: {video}")
        sys.exit(1)

    from stages.audio_stage import run_audio_context_stage
    from stages.context import JobContext
    from stages.entitlements import get_entitlements_for_tier
    from stages.errors import SkipStage

    vi = _ffprobe_video(video)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        ctx = JobContext(
            job_id="tool-audio-1",
            upload_id="tool-upload-1",
            user_id="tool-user-1",
            idempotency_key="tool-audio-1",
            filename=video.name,
            platforms=["youtube"],
            local_video_path=video,
            temp_dir=tmp_path,
            video_info=vi,
            entitlements=get_entitlements_for_tier(args.tier),
            user_settings={
                "use_audio_context": True,
                "useAudioContext": True,
                "audio_transcription": True,
                "audioTranscription": True,
            },
        )

        try:
            ctx = await run_audio_context_stage(ctx)
        except SkipStage as e:
            print(f"SKIP: {e.reason}")
            sys.exit(2)

    ac = ctx.audio_context or {}
    print("\n=== audio_context (summary) ===")
    for k in (
        "category",
        "subcategory",
        "emotional_tone",
        "thumbnail_mood",
        "caption_style",
        "copyright_risk",
        "music_detected",
    ):
        if k in ac:
            print(f"  {k}: {ac[k]}")
    tr = (ac.get("transcript") or "").strip()
    if tr:
        print(f"  transcript ({len(tr)} chars): {tr[:400]}{'…' if len(tr) > 400 else ''}")
    if ctx.ai_transcript:
        print(f"\nai_transcript ({len(ctx.ai_transcript)} chars) matches transcript pipeline output.")


if __name__ == "__main__":
    asyncio.run(main())
