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
from services.ml_engine_config import get_ml_engine_config, ml_engine_public_dict
from services.ml_observability import hf_env_status


def main() -> int:
    urls = get_ml_hub_urls()
    hf_ok, hf_reason = hf_env_status(require_write_token=False)
    trackio_proj = bool((os.environ.get("TRACKIO_PROJECT") or "").strip())
    trackio_space = bool((os.environ.get("TRACKIO_SPACE_ID") or "").strip())

    engine = get_ml_engine_config()
    eng = ml_engine_public_dict(engine)

    print("UploadM8 ML hub wiring (no secrets printed)")
    print(f"  dataset_repo:     {urls.get('dataset_repo') or '(unset)'}")
    print(f"  model_repo:       {urls.get('model_repo') or eng.get('model_repo') or '(unset)'}")
    print(f"  dataset_url:      {'set' if urls.get('dataset_url') else '(unset)'}")
    print(f"  trackio_space:    {urls.get('trackio_space_path') or '(unset)'}")
    print(f"  trackio_space_url: {'set' if urls.get('trackio_space_url') else '(unset)'}")
    print(f"  HF_TOKEN:         {'configured' if hf_ok else 'not configured'} ({hf_reason})")
    print(f"  TRACKIO_PROJECT:  {'set' if trackio_proj else '(unset)'}")
    print(f"  TRACKIO_SPACE_ID: {'set' if trackio_space else '(unset)'}")
    print(f"  ML engine:        {'enabled' if engine.enabled else 'disabled'} | stack_ready={engine.stack_ready}")
    if engine.use_hf_jobs:
        print(f"  HF Jobs:          flavor={engine.jobs_flavor} timeout={engine.jobs_timeout}")

    if urls.get("dataset_url") and urls.get("trackio_space_url") and hf_ok and engine.model_repo:
        print("\n[ok] Hub URLs, model repo, and HF token look ready for the ML engine.")
        return 0
    if urls.get("dataset_url") and urls.get("trackio_space_url") and hf_ok:
        print("\n[warn] Set UM8_HF_MODEL_REPO and UM8_ML_ENGINE_ENABLED=1 for full automation.")
        return 1
    print("\n[warn] Set UM8_HF_DATASET_REPO (or *_URL), UM8_TRACKIO_SPACE_PATH (or *_URL), and HF_TOKEN in .env, then restart the API.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
