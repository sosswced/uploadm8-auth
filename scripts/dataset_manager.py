#!/usr/bin/env python
# /// script
# dependencies = [
#   "datasets>=2.20.0,<4.0.0",
#   "huggingface_hub>=0.24.0,<1.0.0",
#   "pandas>=2.0.0,<3.0.0",
#   "pyarrow>=15.0.0",
#   "python-dotenv>=1.0.0,<2.0.0",
# ]
# ///
"""
Phase 1 Hugging Face dataset manager for UploadM8.

Examples:
  uv run scripts/dataset_manager.py init --repo-id "yourname/uploadm8-promo-v1" --private
  uv run scripts/dataset_manager.py push --repo-id "yourname/uploadm8-promo-v1" --input data/ml/promo_training_v1.parquet
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from datasets import Dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
load_dotenv(_REPO_ROOT / ".env")

from services.ml_observability import hf_env_status, hf_write_token
from services.hf_dataset_export import coerce_dataframe_for_hf, coerce_rows_for_hf


def _require_hf_token() -> str:
    ok, reason = hf_env_status(require_write_token=True)
    if not ok:
        raise SystemExit(f"HF env check failed: {reason}")
    token = hf_write_token()
    if not token:
        raise SystemExit("HF_TOKEN or HUGGING_FACE_HUB_TOKEN is required")
    return token


def _init_repo(repo_id: str, private: bool) -> None:
    token = _require_hf_token()
    create_repo(repo_id=repo_id, token=token, private=private, repo_type="dataset", exist_ok=True)
    print(f"Dataset repo ready: {repo_id}")


def _push_file(repo_id: str, input_path: str, split: str, private: bool) -> None:
    token = _require_hf_token()
    _init_repo(repo_id=repo_id, private=private)

    p = Path(input_path)
    if not p.exists():
        raise SystemExit(f"Input not found: {input_path}")

    if p.suffix.lower() == ".parquet":
        import pandas as pd

        df = coerce_dataframe_for_hf(pd.read_parquet(str(p)))
        ds = Dataset.from_pandas(df, preserve_index=False)
    elif p.suffix.lower() == ".jsonl":
        ds = Dataset.from_json(str(p))
    elif p.suffix.lower() == ".csv":
        ds = Dataset.from_csv(str(p))
    else:
        raise SystemExit("Input must be .parquet, .jsonl, or .csv")

    ds.push_to_hub(repo_id, token=token, split=split, private=private)
    print(f"Pushed split '{split}' with {len(ds)} rows to {repo_id}")


def _write_config(repo_id: str, system_prompt: str) -> None:
    token = _require_hf_token()
    api = HfApi(token=token)
    content = json.dumps(
        {
            "project": "uploadm8-phase1",
            "system_prompt": system_prompt,
            "notes": "Pricing/promo/marketing ML dataset configuration",
        },
        indent=2,
    )
    api.upload_file(
        path_or_fileobj=content.encode("utf-8"),
        path_in_repo="config.json",
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"Uploaded config.json to {repo_id}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="UploadM8 HF dataset manager.")
    sub = p.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="Create/init HF dataset repo.")
    init.add_argument("--repo-id", required=True)
    init.add_argument("--private", action="store_true")

    push = sub.add_parser("push", help="Push a local dataset file to HF.")
    push.add_argument("--repo-id", required=True)
    push.add_argument("--input", required=True)
    push.add_argument("--split", default="train")
    push.add_argument("--private", action="store_true")

    cfg = sub.add_parser("config", help="Upload dataset config/system prompt.")
    cfg.add_argument("--repo-id", required=True)
    cfg.add_argument("--system-prompt", required=True)
    return p


def main() -> int:
    args = _build_parser().parse_args()
    if args.command == "init":
        _init_repo(args.repo_id, bool(args.private))
        return 0
    if args.command == "push":
        _push_file(args.repo_id, args.input, args.split, bool(args.private))
        return 0
    if args.command == "config":
        _write_config(args.repo_id, args.system_prompt)
        return 0
    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
