"""
Push UploadM8 training metrics to Hugging Face (eval.yaml on dataset, eval_results on model).

See https://huggingface.co/docs/hub/eval-results
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from services.ml_observability import hf_write_token

EVAL_YAML = """name: UploadM8 Promo Targeting
description: >
  Promo touchpoint conversion uplift labels exported from UploadM8
  (marketing_touchpoint_deliveries + ml_outcome_labels).
evaluation_framework: uploadm8-promo

tasks:
  - id: promo_uplift_roc_auc
    config: default
    split: train
  - id: promo_uplift_average_precision
    config: default
    split: train
  - id: promo_uplift_lift_at_10pct
    config: default
    split: train
"""


def _require_token() -> str:
    token = hf_write_token()
    if not token:
        raise RuntimeError("HF_TOKEN is required for Hub eval push")
    return token


def ensure_model_repo(repo_id: str, *, private: bool = False) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=_require_token())
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)


def push_dataset_eval_yaml(dataset_repo: str) -> None:
    """Upload ``eval.yaml`` so the dataset can act as an UploadM8 benchmark."""
    from huggingface_hub import HfApi

    api = HfApi(token=_require_token())
    api.upload_file(
        path_or_fileobj=EVAL_YAML.encode("utf-8"),
        path_in_repo="eval.yaml",
        repo_id=dataset_repo,
        repo_type="dataset",
        commit_message="UploadM8: add promo targeting eval.yaml benchmark spec",
    )


def build_eval_results_entries(
    *,
    dataset_repo: str,
    report: Dict[str, Any],
    task_prefix: str = "promo_uplift",
) -> List[Dict[str, Any]]:
    """Map baseline train report JSON to Hub ``.eval_results/*.yaml`` rows."""
    if report.get("status") == "insufficient_label_variance":
        return []
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    notes = str(report.get("model") or "uploadm8_baseline")
    entries: List[Dict[str, Any]] = []

    mapping = (
        ("roc_auc", f"{task_prefix}_roc_auc"),
        ("average_precision", f"{task_prefix}_average_precision"),
        ("lift_at_10pct", f"{task_prefix}_lift_at_10pct"),
    )
    for metric_key, task_id in mapping:
        val = report.get(metric_key)
        if val is None:
            continue
        try:
            fval = float(val)
        except (TypeError, ValueError):
            continue
        entries.append(
            {
                "dataset": {"id": dataset_repo, "task_id": task_id},
                "value": round(fval, 6),
                "date": today,
                "notes": notes,
            }
        )
    return entries


def _yaml_dump_entries(entries: List[Dict[str, Any]]) -> str:
    try:
        import yaml  # type: ignore

        return yaml.safe_dump(entries, sort_keys=False, allow_unicode=True)
    except ImportError:
        # Minimal fallback without PyYAML dependency.
        lines: List[str] = []
        for row in entries:
            ds = row.get("dataset") or {}
            lines.append("- dataset:")
            lines.append(f"    id: {ds.get('id')}")
            lines.append(f"    task_id: {ds.get('task_id')}")
            lines.append(f"  value: {row.get('value')}")
            if row.get("date"):
                lines.append(f'  date: "{row["date"]}"')
            if row.get("notes"):
                lines.append(f'  notes: "{row["notes"]}"')
        return "\n".join(lines) + "\n"


def push_model_eval_results(
    model_repo: str,
    *,
    dataset_repo: str,
    report: Dict[str, Any],
    path_in_repo: str = ".eval_results/uploadm8_promo_uplift.yaml",
) -> Optional[str]:
    """
    Upload evaluation YAML to the model repo. Returns commit URL fragment or None if skipped.
    """
    entries = build_eval_results_entries(dataset_repo=dataset_repo, report=report)
    if not entries:
        return None
    from huggingface_hub import HfApi

    body = _yaml_dump_entries(entries)
    api = HfApi(token=_require_token())
    info = api.upload_file(
        path_or_fileobj=body.encode("utf-8"),
        path_in_repo=path_in_repo,
        repo_id=model_repo,
        repo_type="model",
        commit_message="UploadM8 ML engine: promo uplift eval results",
    )
    return getattr(info, "commit_url", None) or path_in_repo


def push_model_card_metrics(model_repo: str, report: Dict[str, Any]) -> None:
    """Append a short metrics JSON blob under ``uploadm8_metrics.json`` on the model repo."""
    from huggingface_hub import HfApi

    payload = {
        "task": report.get("task"),
        "model": report.get("model"),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "metrics": {
            k: report.get(k)
            for k in (
                "roc_auc",
                "average_precision",
                "lift_at_10pct",
                "lift_at_20pct",
                "train_rows",
                "test_rows",
                "base_positive_rate_test",
            )
            if report.get(k) is not None
        },
    }
    api = HfApi(token=_require_token())
    api.upload_file(
        path_or_fileobj=json.dumps(payload, indent=2).encode("utf-8"),
        path_in_repo="uploadm8_metrics.json",
        repo_id=model_repo,
        repo_type="model",
        commit_message="UploadM8 ML engine: metrics snapshot",
    )
