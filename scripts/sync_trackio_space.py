#!/usr/bin/env python
# /// script
# dependencies = [
#   "trackio>=0.25.0,<1.0.0",
#   "python-dotenv>=1.0.0,<2.0.0",
#   "huggingface_hub>=0.26.0",
# ]
# ///
"""
Sync local Trackio metrics to the UploadM8 HF Space bucket.

Run after training jobs so the Space dashboard shows runs immediately::

    .venv\\Scripts\\python.exe scripts/sync_trackio_space.py

Requires HF_TOKEN and TRACKIO_PROJECT / UM8_TRACKIO_SPACE_PATH (or TRACKIO_SPACE_ID) in .env.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(_REPO_ROOT / ".env")


def main() -> int:
    p = argparse.ArgumentParser(description="Sync local Trackio project to HF Space bucket.")
    p.add_argument(
        "--project",
        default=(os.environ.get("TRACKIO_PROJECT") or "uploadm8-ml").strip(),
    )
    p.add_argument(
        "--space-id",
        default=(
            (os.environ.get("TRACKIO_SPACE_ID") or os.environ.get("UM8_TRACKIO_SPACE_PATH") or "").strip()
        ),
    )
    p.add_argument("--force", action="store_true", help="Overwrite remote DB without prompting.")
    args = p.parse_args()
    if not args.space_id:
        print("Set TRACKIO_SPACE_ID or UM8_TRACKIO_SPACE_PATH", file=sys.stderr)
        return 1

    import trackio
    from huggingface_hub import HfApi

    from services.ml_observability import hf_write_token

    token = hf_write_token()
    if not token:
        print("HF_TOKEN required", file=sys.stderr)
        return 1

    bucket_id = f"{args.space_id}-bucket"
    api = HfApi(token=token)
    try:
        from trackio.deploy import _ensure_bucket_mounted_at_data

        _ensure_bucket_mounted_at_data(args.space_id, bucket_id, hf_api=api)
    except Exception as exc:
        print(f"[warn] bucket volume attach skipped: {exc}")

    print(f"Syncing {args.project!r} -> {args.space_id} (bucket {bucket_id}) …")
    trackio.sync(project=args.project, space_id=args.space_id, force=args.force, bucket_id=bucket_id)
    api.restart_space(args.space_id)
    print(f"Done. Open https://huggingface.co/spaces/{args.space_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
