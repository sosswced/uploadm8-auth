"""Regenerate frontend/js/caption-creative.generated.js from core.caption_creative."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.caption_creative import (  # noqa: E402
    DEFAULT_CAPTION_STYLE,
    DEFAULT_CAPTION_TONE,
    DEFAULT_CAPTION_VOICE,
    ui_style_options,
    ui_tone_options,
    ui_voice_options,
)

OUT = ROOT / "frontend" / "js" / "caption-creative.generated.js"


def main() -> int:
    payload = {
        "styles": ui_style_options(),
        "tones": ui_tone_options(),
        "voices": ui_voice_options(),
        "defaults": {
            "style": DEFAULT_CAPTION_STYLE,
            "tone": DEFAULT_CAPTION_TONE,
            "voice": DEFAULT_CAPTION_VOICE,
        },
    }
    body = json.dumps(payload, indent=2)
    # Keep JS object keys unquoted for readability parity with prior file.
    text = f"""/**
 * Auto-aligned with core/caption_creative.py — do not hand-edit allowlists.
 * Regenerate: python tools/generate_caption_creative_js.py
 */
(function (global) {{
  'use strict';
  var CAPTION_CREATIVE = {body};
  global.CAPTION_CREATIVE = CAPTION_CREATIVE;
}})(typeof window !== 'undefined' ? window : globalThis);
"""
    OUT.write_text(text, encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
