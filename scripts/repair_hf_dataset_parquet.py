#!/usr/bin/env python
# /// script
# dependencies = [
#   "datasets>=2.20.0,<4.0.0",
#   "huggingface_hub>=0.26.0,<1.0.0",
#   "pandas>=2.0.0,<3.0.0",
#   "pyarrow>=15.0.0",
#   "python-dotenv>=1.0.0,<2.0.0",
# ]
# ///
"""
Repair Hugging Face dataset parquet files that use Arrow UUID types (breaks Dataset Viewer).

Downloads *.parquet from a Hub dataset repo, coerces UUID/datetime columns to strings,
and re-uploads the same paths so the viewer can parse splits again.

  .venv\\Scripts\\python.exe scripts/repair_hf_dataset_parquet.py --repo-id cedy243/uploadm8-promo-targeting-v1
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
load_dotenv(_REPO_ROOT / ".env")

from services.hf_dataset_export import coerce_dataframe_for_hf, parquet_bytes_hf_safe, read_parquet_hf_safe
from services.ml_observability import hf_write_token


def _token() -> str:
    t = hf_write_token()
    if not t:
        raise SystemExit("HF_TOKEN or HUGGING_FACE_HUB_TOKEN required")
    return t


def repair_repo(repo_id: str, *, dry_run: bool = False) -> None:
    token = _token()
    api = HfApi(token=token)
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    parquet_files = [f for f in files if f.lower().endswith(".parquet")]
    if not parquet_files:
        print(f"No parquet files in {repo_id} — nothing to repair.")
        return
    print(f"Found {len(parquet_files)} parquet file(s) in {repo_id}")
    for path_in_repo in parquet_files:
        print(f"  repairing {path_in_repo} …")
        local = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=path_in_repo, token=token)

        df = read_parquet_hf_safe(local)
        safe = coerce_dataframe_for_hf(df)
        if dry_run:
            print(f"    dry-run: {len(safe)} rows, columns={list(safe.columns)}")
            if len(safe) and "touchpoint_id" in safe.columns:
                print(f"    sample touchpoint_id: {safe.iloc[0]['touchpoint_id']}")
            continue
        payload = parquet_bytes_hf_safe(safe)
        api.upload_file(
            path_or_fileobj=payload,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Repair parquet for HF Dataset Viewer (string UUIDs): {path_in_repo}",
        )
        print(f"    uploaded {path_in_repo} ({len(safe)} rows)")
    if not dry_run:
        print(f"Done. Open https://huggingface.co/datasets/{repo_id} and refresh the viewer.")


def main() -> int:
    p = argparse.ArgumentParser(description="Repair HF dataset parquet UUID columns for Dataset Viewer.")
    p.add_argument(
        "--repo-id",
        default=(os.environ.get("UM8_HF_DATASET_REPO") or "").strip(),
        help="Hub dataset repo (default: UM8_HF_DATASET_REPO)",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if not args.repo_id:
        raise SystemExit("Provide --repo-id or set UM8_HF_DATASET_REPO")
    repair_repo(args.repo_id, dry_run=bool(args.dry_run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
