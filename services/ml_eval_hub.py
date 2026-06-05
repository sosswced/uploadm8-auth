"""
Push UploadM8 training metrics to Hugging Face (eval.yaml on dataset, eval_results on model).

See https://huggingface.co/docs/hub/eval-results
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from services.ml_observability import hf_write_token

EVAL_YAML = """name: UploadM8 ML Hub
description: >
  UploadM8 training/eval tables exported from the app DB. Two loops share this
  dataset repo (separate splits / "buckets"):
    * promo targeting uplift (marketing_touchpoint_deliveries + ml_outcome_labels) — split: train
    * content success / hottest-topic (per upload x platform engagement + upload-flow
      topic features from uploads.platform_results + output_artifacts) — split: content_success
evaluation_framework: uploadm8

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
  - id: content_success_roc_auc
    config: default
    split: content_success
  - id: content_success_average_precision
    config: default
    split: content_success
  - id: content_success_lift_at_10pct
    config: default
    split: content_success
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
    if report.get("status") in ("insufficient_label_variance", "insufficient_rows"):
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
    task_prefix: str = "promo_uplift",
    commit_message: str = "UploadM8 ML engine: promo uplift eval results",
) -> Optional[str]:
    """
    Upload evaluation YAML to the model repo. Returns commit URL fragment or None if skipped.
    """
    entries = build_eval_results_entries(
        dataset_repo=dataset_repo, report=report, task_prefix=task_prefix
    )
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
        commit_message=commit_message,
    )
    return getattr(info, "commit_url", None) or path_in_repo


def push_content_eval_results(
    model_repo: str,
    *,
    dataset_repo: str,
    report: Dict[str, Any],
) -> Optional[str]:
    """Eval-results for the content-success / hottest-topic model (its own bucket)."""
    return push_model_eval_results(
        model_repo,
        dataset_repo=dataset_repo,
        report=report,
        path_in_repo=".eval_results/uploadm8_content_success.yaml",
        task_prefix="content_success",
        commit_message="UploadM8 ML engine: content success eval results",
    )


_DEFAULT_METRIC_KEYS = (
    "roc_auc",
    "average_precision",
    "lift_at_10pct",
    "lift_at_20pct",
    "train_rows",
    "test_rows",
    "base_positive_rate_test",
)


def push_model_card_metrics(
    model_repo: str,
    report: Dict[str, Any],
    *,
    path_in_repo: str = "uploadm8_metrics.json",
    metric_keys: tuple = _DEFAULT_METRIC_KEYS,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Append a short metrics JSON blob to the model repo (one file per loop/bucket)."""
    from huggingface_hub import HfApi

    payload: Dict[str, Any] = {
        "task": report.get("task"),
        "model": report.get("model"),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "metrics": {k: report.get(k) for k in metric_keys if report.get(k) is not None},
    }
    if extra:
        payload.update(extra)
    api = HfApi(token=_require_token())
    api.upload_file(
        path_or_fileobj=json.dumps(payload, indent=2).encode("utf-8"),
        path_in_repo=path_in_repo,
        repo_id=model_repo,
        repo_type="model",
        commit_message="UploadM8 ML engine: metrics snapshot",
    )


def push_content_card_metrics(model_repo: str, report: Dict[str, Any]) -> None:
    """Content-success metrics + the ranked hottest topics/content, as its own file."""
    rankings = report.get("rankings") or {}
    push_model_card_metrics(
        model_repo,
        report,
        path_in_repo="uploadm8_content_metrics.json",
        extra={
            "rows": report.get("rows"),
            "top_topics": (rankings.get("top_topics") or [])[:10],
            "top_hashtags": (rankings.get("top_hashtags") or [])[:10],
            "top_platform_topic": (rankings.get("top_platform_topic") or [])[:10],
            "top_packaging": (rankings.get("top_packaging") or [])[:10],
        },
    )
