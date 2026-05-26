"""
Environment configuration for the UploadM8 ML / AI engine (dataset → train → eval → Hub).

Single source for automation flags consumed by ``services/ml_engine``,
``routers/admin``, and ``services/ml_hub_config``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from services.ml_hub_config import get_ml_hub_urls
from services.ml_observability import hf_env_status, hf_write_token


def _env(name: str, default: str = "") -> str:
    v = (os.environ.get(name) or "").strip()
    return v if v else default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class MLEngineConfig:
    enabled: bool
    interval_seconds: int
    dataset_lookback_days: int
    dataset_limit: int
    min_train_rows: int
    dataset_repo: Optional[str]
    model_repo: Optional[str]
    model_url: Optional[str]
    local_dataset_path: str
    local_report_path: str
    use_hf_jobs: bool
    jobs_flavor: str
    jobs_timeout: str
    jobs_namespace: Optional[str]
    eval_task_id: str
    sync_trackio_after_run: bool
    run_quality_scoring: bool
    quality_scoring_lookback_days: int

    @property
    def hf_token_ok(self) -> bool:
        ok, _ = hf_env_status(require_write_token=False)
        return ok

    @property
    def stack_ready(self) -> bool:
        return bool(
            self.enabled
            and self.hf_token_ok
            and self.dataset_repo
            and self.model_repo
        )


def get_ml_engine_config() -> MLEngineConfig:
    hub = get_ml_hub_urls()
    model_repo = _env("UM8_HF_MODEL_REPO") or None
    model_url = _env("UM8_HF_MODEL_URL")
    if not model_url and model_repo:
        model_url = f"https://huggingface.co/{model_repo}"

    return MLEngineConfig(
        enabled=_env_bool("UM8_ML_ENGINE_ENABLED", default=False),
        interval_seconds=max(3600, _env_int("UM8_ML_ENGINE_INTERVAL_SECONDS", 86400)),
        dataset_lookback_days=max(7, _env_int("UM8_ML_ENGINE_DATASET_LOOKBACK_DAYS", 180)),
        dataset_limit=max(100, _env_int("UM8_ML_ENGINE_DATASET_LIMIT", 250000)),
        min_train_rows=max(2, _env_int("UM8_ML_ENGINE_MIN_TRAIN_ROWS", 8)),
        dataset_repo=hub.get("dataset_repo"),
        model_repo=model_repo,
        model_url=model_url or None,
        local_dataset_path=_env(
            "UM8_ML_ENGINE_DATASET_PATH", "data/ml/promo_targeting_train_v1.parquet"
        ),
        local_report_path=_env(
            "UM8_ML_ENGINE_REPORT_PATH", "data/ml/promo_targeting_baseline_report.json"
        ),
        use_hf_jobs=_env_bool("UM8_ML_ENGINE_USE_HF_JOBS", default=False),
        jobs_flavor=_env("UM8_HF_JOBS_FLAVOR", "cpu-basic"),
        jobs_timeout=_env("UM8_HF_JOBS_TIMEOUT", "2h"),
        jobs_namespace=_env("UM8_HF_JOBS_NAMESPACE") or None,
        eval_task_id=_env("UM8_ML_EVAL_TASK_ID", "promo_uplift_roc_auc"),
        sync_trackio_after_run=_env_bool("UM8_ML_ENGINE_SYNC_TRACKIO", default=True),
        run_quality_scoring=_env_bool("UM8_ML_ENGINE_RUN_QUALITY_SCORING", default=True),
        quality_scoring_lookback_days=max(
            7, _env_int("ML_SCORING_LOOKBACK_DAYS", 180)
        ),
    )


def ml_engine_public_dict(cfg: Optional[MLEngineConfig] = None) -> Dict[str, Any]:
    """Non-secret status for admin UI / observability."""
    c = cfg or get_ml_engine_config()
    token = hf_write_token()
    return {
        "enabled": c.enabled,
        "interval_seconds": c.interval_seconds,
        "stack_ready": c.stack_ready,
        "hf_token_configured": bool(token),
        "dataset_repo": c.dataset_repo,
        "model_repo": c.model_repo,
        "model_url": c.model_url,
        "use_hf_jobs": c.use_hf_jobs,
        "jobs_flavor": c.jobs_flavor,
        "jobs_timeout": c.jobs_timeout,
        "jobs_namespace": c.jobs_namespace,
        "eval_task_id": c.eval_task_id,
        "docs": {
            "jobs": "https://huggingface.co/docs/huggingface_hub/guides/jobs",
            "trl_jobs": "https://huggingface.co/docs/trl/en/jobs_training",
            "eval_results": "https://huggingface.co/docs/hub/eval-results",
            "datasets": "https://huggingface.co/docs/datasets/index",
            "trl": "https://huggingface.co/docs/trl/index",
        },
    }
