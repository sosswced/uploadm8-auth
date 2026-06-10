#!/usr/bin/env python3
"""Print effective pipeline/ML env vs code defaults (loads .env if present)."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
try:
    from dotenv import load_dotenv

    load_dotenv(_REPO / ".env")
except ImportError:
    pass

from core.pipeline_env_defaults import pipeline_env_audit
from services.ml_hub_config import get_ml_hub_urls
from services.ml_engine_config import get_ml_engine_config
from services.promo_targeting_model import ml_targeting_enabled


def main() -> None:
    print("=== Pipeline env audit (.env vs code defaults) ===")
    for r in pipeline_env_audit():
        flag = "SET" if r["env_set"] else "default"
        print(f"{r['name']:40} [{flag:7}] effective={r['effective']!r}")

    hub = get_ml_hub_urls()
    cfg = get_ml_engine_config()
    print()
    print("=== ML hub (effective) ===")
    print("promo dataset:     ", hub.get("dataset_repo"))
    print("promo model:       ", hub.get("model_repo"))
    print("content dataset:   ", hub.get("content_dataset_repo"))
    print("content model:     ", hub.get("content_model_repo"))
    print("ml_engine enabled: ", cfg.enabled)
    print("run_content_success:", cfg.run_content_success)
    print("content_stack_ready:", cfg.content_stack_ready)
    print("ml_targeting:      ", ml_targeting_enabled())


if __name__ == "__main__":
    main()
