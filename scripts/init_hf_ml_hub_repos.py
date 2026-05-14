#!/usr/bin/env python
# /// script
# dependencies = [
#   "huggingface_hub>=0.24.0,<1.0.0",
#   "python-dotenv>=1.0.0,<2.0.0",
# ]
# ///
"""
Ensure Hugging Face dataset + Space repos exist for UploadM8 ML Hub links.

Requires HF_TOKEN with write access. Does not configure the Space runtime,
Trackio secrets, or Hub Buckets (those are separate from app R2).

Example (repo root, .env with HF_TOKEN):

  uv run scripts/init_hf_ml_hub_repos.py \\
    --dataset-repo YOUR_ORG/uploadm8-promo-targeting-v1 \\
    --trackio-space YOUR_ORG/uploadm8-trackio

Then set UM8_HF_DATASET_REPO, UM8_TRACKIO_SPACE_PATH (or *_URL overrides),
TRACKIO_PROJECT, and TRACKIO_SPACE_ID in .env and restart the API.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import create_repo

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
load_dotenv(_REPO_ROOT / ".env")

from services.ml_observability import hf_env_status  # noqa: E402


def _token() -> str:
    ok, reason = hf_env_status(require_write_token=True)
    if not ok:
        raise SystemExit(f"HF env check failed: {reason}")
    t = (os.environ.get("HF_TOKEN") or "").strip()
    if not t:
        raise SystemExit("HF_TOKEN is required")
    return t


def _validate_repo_id(s: str, label: str) -> str:
    rid = s.strip()
    if "/" not in rid or rid.startswith("/") or rid.endswith("/"):
        raise SystemExit(f"{label} must look like namespace/name (got {s!r})")
    return rid


def main() -> int:
    p = argparse.ArgumentParser(description="Create HF dataset + Trackio Space repos (exist_ok).")
    p.add_argument("--dataset-repo", required=True, help="Dataset repo id, e.g. org/uploadm8-promo-v1")
    p.add_argument("--trackio-space", required=True, help="Space repo id, e.g. org/uploadm8-trackio")
    p.add_argument("--private", action="store_true", help="Create private dataset and Space.")
    p.add_argument(
        "--space-sdk",
        default="gradio",
        choices=("gradio", "docker"),
        help="HF Space SDK template (empty Space shell only).",
    )
    args = p.parse_args()
    token = _token()
    ds = _validate_repo_id(args.dataset_repo, "--dataset-repo")
    sp = _validate_repo_id(args.trackio_space, "--trackio-space")

    create_repo(repo_id=ds, token=token, private=bool(args.private), repo_type="dataset", exist_ok=True)
    print(f"Dataset repo ready: {ds}")

    create_repo(
        repo_id=sp,
        token=token,
        private=bool(args.private),
        repo_type="space",
        space_sdk=args.space_sdk,
        exist_ok=True,
    )
    print(f"Space repo ready: {sp}")

    print()
    print("Add to .env (then restart the API):")
    print(f"  UM8_HF_DATASET_REPO={ds}")
    print(f"  UM8_TRACKIO_SPACE_PATH={sp}")
    print("  TRACKIO_PROJECT=<your Trackio project name>")
    print(f"  TRACKIO_SPACE_ID={sp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
