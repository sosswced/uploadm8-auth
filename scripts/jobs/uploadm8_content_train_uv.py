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
HF Jobs UV entrypoint: train the content-success / hottest-topic model from the Hub
dataset (split ``content_success``) and push eval results + ranked topics.

Mounted env (set by ``services/ml_engine`` when submitting the job):
  UM8_HF_DATASET_REPO, UM8_HF_MODEL_REPO, HF_TOKEN (secret), TRACKIO_PROJECT

Reuses the *current buckets*: same dataset repo (different split), same model repo
(separate ``.eval_results/uploadm8_content_success.yaml`` + ``uploadm8_content_metrics.json``),
and the same Trackio project/Space bucket as the promo loop.

Docs: https://huggingface.co/docs/huggingface_hub/guides/jobs
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
    push_content_card_metrics,
    push_content_eval_results,
    push_dataset_eval_yaml,
)
from services.ml_observability import OptionalTrackioRun, hf_write_token


def _load_dataset_parquet(dataset_repo: str) -> "Path":
    from datasets import load_dataset

    token = hf_write_token()
    ds = load_dataset(dataset_repo, split="content_success", token=token or None)
    tmp = Path(tempfile.mkdtemp(prefix="um8_content_"))
    out = tmp / "content_success.parquet"
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

    track = OptionalTrackioRun("content_success_hf_job_train")
    track.start(config={"dataset_repo": dataset_repo, "model_repo": model_repo})

    try:
        parquet = _load_dataset_parquet(dataset_repo)
        import importlib.util

        train_path = _REPO / "scripts" / "train_content_success_model.py"
        spec = importlib.util.spec_from_file_location("um8_train_content", train_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load {train_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        class Args:
            input = str(parquet)
            report_out = str(parquet.parent / "report.json")

        report = mod._run(Args())
        track.log({k: v for k, v in report.items() if k != "rankings"})

        if report.get("status") == "ok":
            ensure_model_repo(model_repo)
            try:
                push_dataset_eval_yaml(dataset_repo)
            except Exception as exc:
                print(f"[warn] eval.yaml: {exc}", file=sys.stderr)
            push_content_eval_results(model_repo, dataset_repo=dataset_repo, report=report)
            push_content_card_metrics(model_repo, report)

        out = Path(Args.report_out)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in report.items() if k != "rankings"}, indent=2))
        return 0 if report.get("status") in ("ok", "insufficient_rows", "insufficient_label_variance") else 2
    except Exception as exc:
        track.log({"status": 0, "error": str(exc)[:300]})
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        track.finish()


if __name__ == "__main__":
    raise SystemExit(main())
