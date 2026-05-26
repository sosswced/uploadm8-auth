#!/usr/bin/env python
# /// script
# dependencies = [
#   "pandas>=2.0.0,<3.0.0",
#   "pyarrow>=15.0.0",
#   "scikit-learn>=1.4.0,<2.0.0",
#   "datasets>=2.20.0,<4.0.0",
#   "huggingface_hub>=0.26.0",
#   "trackio>=0.25.0,<1.0.0",
#   "pyyaml>=6.0.0,<7.0.0",
# ]
# ///
"""
HF Jobs UV entrypoint: train promo uplift baseline from Hub dataset, push eval results.

Mounted env (set by ``services/ml_engine`` when submitting the job):
  UM8_HF_DATASET_REPO, UM8_HF_MODEL_REPO, HF_TOKEN (secret), TRACKIO_PROJECT

Docs: https://huggingface.co/docs/huggingface_hub/guides/jobs
      https://huggingface.co/docs/trl/en/jobs_training (TRL Jobs use the same HF Jobs API)
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from services.ml_eval_hub import (
    ensure_model_repo,
    push_dataset_eval_yaml,
    push_model_card_metrics,
    push_model_eval_results,
)
from services.ml_observability import OptionalTrackioRun, hf_write_token


def _load_dataset_parquet(dataset_repo: str) -> "Path":
    from datasets import load_dataset

    token = hf_write_token()
    ds = load_dataset(dataset_repo, split="train", token=token or None)
    tmp = Path(tempfile.mkdtemp(prefix="um8_promo_"))
    out = tmp / "train.parquet"
    ds.to_pandas().to_parquet(out, index=False)
    return out


def main() -> int:
    dataset_repo = (os.environ.get("UM8_HF_DATASET_REPO") or "").strip()
    model_repo = (os.environ.get("UM8_HF_MODEL_REPO") or "").strip()
    if not dataset_repo or not model_repo:
        print("UM8_HF_DATASET_REPO and UM8_HF_MODEL_REPO are required", file=sys.stderr)
        return 1
    if not hf_write_token():
        print("HF_TOKEN secret required", file=sys.stderr)
        return 1

    track = OptionalTrackioRun("promo_uplift_hf_job_train")
    track.start(config={"dataset_repo": dataset_repo, "model_repo": model_repo})

    try:
        parquet = _load_dataset_parquet(dataset_repo)
        # Import train logic from repo script module path
        import importlib.util

        train_path = _REPO / "scripts" / "train_promo_uplift_baseline.py"
        spec = importlib.util.spec_from_file_location("um8_train_promo", train_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load {train_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        class Args:
            input = str(parquet)
            report_out = str(parquet.parent / "report.json")

        report = mod._run(Args())
        track.log(report)

        ensure_model_repo(model_repo)
        try:
            push_dataset_eval_yaml(dataset_repo)
        except Exception as exc:
            print(f"[warn] eval.yaml: {exc}", file=sys.stderr)
        push_model_eval_results(model_repo, dataset_repo=dataset_repo, report=report)
        push_model_card_metrics(model_repo, report)

        out = Path(Args.report_out)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0 if report.get("status") != "insufficient_label_variance" else 2
    except Exception as exc:
        track.log({"status": 0, "error": str(exc)[:300]})
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        track.finish()


if __name__ == "__main__":
    raise SystemExit(main())
