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

By default uses **bucket upload + Space restart** (fast, reliable for local dev). The Trackio
library's built-in ``sync()`` also waits up to 180s for the live Space API to confirm each run;
that often times out when the Space is cold or has a stale run named ``trackio-``. Use
``--verify`` only when you need that confirmation step.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(_REPO_ROOT / ".env")


def _env_truthy(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in ("1", "true", "yes", "on")


def _warn_stale_local_runs(project: str) -> None:
    try:
        from trackio.sqlite_storage import SQLiteStorage

        runs = Counter(
            log["run"] for log in SQLiteStorage.get_all_logs_for_sync(project)
        )
    except Exception:
        return
    if not runs:
        print(f"[info] No local Trackio logs to sync for project {project!r}.")
        return
    bad = [r for r in runs if not r or r.endswith("-")]
    print(f"[info] Local runs pending sync: {dict(runs)}")
    if bad:
        print(
            "[warn] Run name(s) end with '-' or are empty — often from "
            "TRACKIO_RUN_ID=trackio-${RENDER_GIT_COMMIT} in .env while not on Render. "
            "Unset TRACKIO_RUN_ID for local dev; OptionalTrackioRun will use the script run name."
        )


def _ensure_space_vars(api, space_id: str, *, project: str, bucket_id: str) -> None:
    import huggingface_hub

    for key, value in (
        ("TRACKIO_PROJECT", project),
        ("TRACKIO_BUCKET_ID", bucket_id),
    ):
        try:
            huggingface_hub.add_space_variable(space_id, key, value)
        except Exception as exc:
            print(f"[warn] could not set Space variable {key}: {exc}")


def _sync_bucket_and_restart(
    *,
    api,
    project: str,
    space_id: str,
    bucket_id: str,
) -> None:
    from trackio.bucket_storage import create_bucket_if_not_exists, upload_project_to_bucket
    from trackio.deploy import _ensure_bucket_mounted_at_data, create_space_if_not_exists
    from trackio.sqlite_storage import SQLiteStorage

    create_bucket_if_not_exists(bucket_id, private=None)
    upload_project_to_bucket(project, bucket_id)
    print(f"* Project data uploaded to bucket: https://huggingface.co/buckets/{bucket_id}")

    create_space_if_not_exists(space_id, bucket_id=bucket_id, private=None)
    try:
        _ensure_bucket_mounted_at_data(space_id, bucket_id, hf_api=api)
    except Exception as exc:
        print(f"[warn] bucket volume attach skipped: {exc}")

    _ensure_space_vars(api, space_id, project=project, bucket_id=bucket_id)
    SQLiteStorage.set_project_metadata(project, "space_id", space_id)

    print(f"* Restarting Space {space_id} (loads bucket at /data/trackio) …")
    api.restart_space(space_id)
    print(
        "* Skipped live API verify (use --verify for trackio.sync wait). "
        "Open the Space after status is Running — cold start can take 1–3 minutes."
    )


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
    p.add_argument(
        "--verify",
        action="store_true",
        help=(
            "Use trackio.sync() and wait up to ~180s for the live Space API to confirm runs. "
            "Often fails locally when the Space is sleeping or a stale run exists (e.g. trackio-)."
        ),
    )
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
    # Fast path unless explicitly verifying (or TRACKIO_SYNC_VERIFY=1 for CI).
    use_fast_path = not args.verify and not _env_truthy("TRACKIO_SYNC_VERIFY")

    _warn_stale_local_runs(args.project)
    print(f"Syncing {args.project!r} -> {args.space_id} (bucket {bucket_id}) …")

    if use_fast_path:
        _sync_bucket_and_restart(
            api=api,
            project=args.project,
            space_id=args.space_id,
            bucket_id=bucket_id,
        )
    else:
        try:
            from trackio.deploy import _ensure_bucket_mounted_at_data

            _ensure_bucket_mounted_at_data(args.space_id, bucket_id, hf_api=api)
        except Exception as exc:
            print(f"[warn] bucket volume attach skipped: {exc}")
        trackio.sync(
            project=args.project,
            space_id=args.space_id,
            force=args.force,
            bucket_id=bucket_id,
        )
        _ensure_space_vars(api, args.space_id, project=args.project, bucket_id=bucket_id)
        api.restart_space(args.space_id)

    print(f"Done. Open https://huggingface.co/spaces/{args.space_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
