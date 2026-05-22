#!/usr/bin/env python
"""Print non-secret ML / Hugging Face hub wiring status (loads repo-root .env)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

try:
    from dotenv import load_dotenv

    load_dotenv(_REPO / ".env")
except ImportError:
    pass

from services.ml_hub_config import get_ml_hub_urls
from services.ml_observability import hf_env_status


def main() -> int:
    urls = get_ml_hub_urls()
    hf_ok, hf_reason = hf_env_status(require_write_token=False)
    trackio_proj = bool((os.environ.get("TRACKIO_PROJECT") or "").strip())
    trackio_space = bool((os.environ.get("TRACKIO_SPACE_ID") or "").strip())

    print("UploadM8 ML hub wiring (no secrets printed)")
    print(f"  dataset_repo:     {urls.get('dataset_repo') or '(unset)'}")
    print(f"  dataset_url:      {'set' if urls.get('dataset_url') else '(unset)'}")
    print(f"  trackio_space:    {urls.get('trackio_space_path') or '(unset)'}")
    print(f"  trackio_space_url: {'set' if urls.get('trackio_space_url') else '(unset)'}")
    print(f"  HF_TOKEN:         {'configured' if hf_ok else 'not configured'} ({hf_reason})")
    print(f"  TRACKIO_PROJECT:  {'set' if trackio_proj else '(unset)'}")
    print(f"  TRACKIO_SPACE_ID: {'set' if trackio_space else '(unset)'}")

    if urls.get("dataset_url") and urls.get("trackio_space_url") and hf_ok:
        print("\n[ok] Hub URLs and HF token look ready for scripts and admin chips.")
        return 0
    print("\n[warn] Set UM8_HF_DATASET_REPO (or *_URL), UM8_TRACKIO_SPACE_PATH (or *_URL), and HF_TOKEN in .env, then restart the API.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
