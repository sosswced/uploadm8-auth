"""
UploadM8 ML / AI engine — automated dataset → Hub → train → eval → Trackio cycle.

Runs locally by default; optional Hugging Face Jobs GPU/CPU offload via
``UM8_ML_ENGINE_USE_HF_JOBS=1`` (see https://huggingface.co/docs/huggingface_hub/guides/jobs).
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import asyncpg

from services.ml_engine_config import MLEngineConfig, get_ml_engine_config, ml_engine_public_dict
from services.ml_eval_hub import (
    ensure_model_repo,
    push_dataset_eval_yaml,
    push_model_card_metrics,
    push_model_eval_results,
)
from services.ml_observability import OptionalTrackioRun, hf_write_token

logger = logging.getLogger("uploadm8.ml_engine")

_REPO_ROOT = Path(__file__).resolve().parents[1]
_STATE_PATH = _REPO_ROOT / "data" / "ml" / "engine_state.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_engine_state() -> Dict[str, Any]:
    if not _STATE_PATH.is_file():
        return {}
    try:
        return json.loads(_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_engine_state(state: Dict[str, Any]) -> None:
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _run_subprocess(cmd: list[str], *, timeout: int = 1800) -> Dict[str, Any]:
    proc = subprocess.run(
        cmd,
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout_tail": (proc.stdout or "")[-4000:],
        "stderr_tail": (proc.stderr or "")[-4000:],
    }


async def _record_promo_model_run(pool: asyncpg.Pool, report: Dict[str, Any]) -> Optional[str]:
    if report.get("status") == "insufficient_label_variance":
        return None
    run_id = uuid.uuid4()
    metrics = dict(report)
    features = report.get("feature_columns") or []
    train_rows = int(report.get("train_rows") or 0)
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO m8_model_runs (
                id, model_version, train_row_count, val_mae_log1p_views,
                features_used, train_config, metrics
            ) VALUES ($1, $2, $3, NULL, $4::jsonb, $5::jsonb, $6::jsonb)
            """,
            run_id,
            str(report.get("model") or "promo_uplift_v1")[:40],
            train_rows,
            json.dumps(features),
            json.dumps({"engine": "uploadm8_ml_engine", "task": report.get("task")}),
            json.dumps(metrics),
        )
    return str(run_id)


def _read_report(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        return {"status": "missing_report", "path": str(p)}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        return {"status": "invalid_report", "error": str(e)[:200]}


def _step_build_dataset(cfg: MLEngineConfig) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "build_promo_training_dataset.py"),
        "--lookback-days",
        str(cfg.dataset_lookback_days),
        "--limit",
        str(cfg.dataset_limit),
        "--output",
        cfg.local_dataset_path,
    ]
    if cfg.dataset_repo:
        cmd.extend(["--push-to", cfg.dataset_repo, "--split", "train"])
    return _run_subprocess(cmd)


def _step_train_local(cfg: MLEngineConfig) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "train_promo_uplift_baseline.py"),
        "--input",
        cfg.local_dataset_path,
        "--report-out",
        cfg.local_report_path,
    ]
    return _run_subprocess(cmd)


def _step_push_eval(cfg: MLEngineConfig, report: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"ok": True, "skipped": []}
    if not cfg.dataset_repo or not cfg.model_repo:
        out["ok"] = False
        out["error"] = "dataset_repo and model_repo required for eval push"
        return out
    try:
        ensure_model_repo(cfg.model_repo)
        try:
            push_dataset_eval_yaml(cfg.dataset_repo)
            out["dataset_eval_yaml"] = True
        except Exception as e:
            out["dataset_eval_yaml_error"] = str(e)[:300]
        eval_path = push_model_eval_results(
            cfg.model_repo,
            dataset_repo=cfg.dataset_repo,
            report=report,
        )
        out["eval_results_path"] = eval_path
        push_model_card_metrics(cfg.model_repo, report)
        out["metrics_json"] = True
    except Exception as e:
        out["ok"] = False
        out["error"] = str(e)[:400]
    return out


def _step_submit_hf_job(cfg: MLEngineConfig) -> Dict[str, Any]:
    token = hf_write_token()
    if not token:
        return {"ok": False, "error": "HF_TOKEN missing for HF Jobs"}
    script = _REPO_ROOT / "scripts" / "jobs" / "uploadm8_promo_train_uv.py"
    if not script.is_file():
        return {"ok": False, "error": f"UV job script missing: {script}"}
    try:
        from huggingface_hub import run_uv_job
    except ImportError as e:
        return {"ok": False, "error": f"huggingface_hub run_uv_job unavailable: {e}"}

    env = {
        "UM8_HF_DATASET_REPO": cfg.dataset_repo or "",
        "UM8_HF_MODEL_REPO": cfg.model_repo or "",
        "UM8_ML_EVAL_TASK_ID": cfg.eval_task_id,
        "TRACKIO_PROJECT": (__import__("os").environ.get("TRACKIO_PROJECT") or "uploadm8-ml"),
    }
    raw_space = (__import__("os").environ.get("TRACKIO_SPACE_ID") or "").strip()
    if raw_space:
        env["TRACKIO_SPACE_ID"] = raw_space

    kwargs: Dict[str, Any] = {
        "flavor": cfg.jobs_flavor,
        "secrets": {"HF_TOKEN": token},
        "env": env,
        "timeout": cfg.jobs_timeout,
    }
    if cfg.jobs_namespace:
        kwargs["namespace"] = cfg.jobs_namespace

    try:
        job = run_uv_job(str(script), **kwargs)
        return {
            "ok": True,
            "job_id": getattr(job, "id", None),
            "job_url": getattr(job, "url", None),
            "status": getattr(getattr(job, "status", None), "stage", None),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)[:400]}


def _step_sync_trackio() -> Dict[str, Any]:
    space = (__import__("os").environ.get("UM8_TRACKIO_SPACE_PATH") or "").strip()
    project = (__import__("os").environ.get("TRACKIO_PROJECT") or "uploadm8-ml").strip()
    if not space:
        return {"ok": True, "skipped": "UM8_TRACKIO_SPACE_PATH unset"}
    cmd = [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "sync_trackio_space.py"),
    ]
    return _run_subprocess(cmd, timeout=600)


async def run_ml_engine_cycle(
    pool: Optional[asyncpg.Pool],
    *,
    cfg: Optional[MLEngineConfig] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Full automation cycle. Returns structured result (also persisted to engine_state.json).
    """
    c = cfg or get_ml_engine_config()
    if not c.enabled and not force:
        return {"ok": False, "skipped": True, "reason": "UM8_ML_ENGINE_ENABLED is off"}

    if not c.stack_ready and not force:
        return {
            "ok": False,
            "skipped": True,
            "reason": "ML engine stack not ready (token, dataset_repo, model_repo, enabled)",
            "config": ml_engine_public_dict(c),
        }

    track = OptionalTrackioRun("ml_engine_cycle")
    track.start(config=ml_engine_public_dict(c))

    result: Dict[str, Any] = {
        "ok": False,
        "started_at": _now_iso(),
        "config": ml_engine_public_dict(c),
        "steps": {},
    }

    try:
        if c.run_quality_scoring and pool is not None:
            from services.ml_scoring_job import run_ml_scoring_cycle

            qs = await run_ml_scoring_cycle(pool, lookback_days=c.quality_scoring_lookback_days)
            result["steps"]["quality_scoring"] = {"ok": qs is not None, "rows": qs}

        build = await asyncio.to_thread(_step_build_dataset, c)
        result["steps"]["build_dataset"] = build
        if not build.get("ok"):
            result["error"] = "dataset build failed"
            return result

        if c.use_hf_jobs:
            job = await asyncio.to_thread(_step_submit_hf_job, c)
            result["steps"]["hf_jobs_train"] = job
            result["ok"] = bool(job.get("ok"))
            if not job.get("ok"):
                result["error"] = job.get("error") or "HF Jobs submit failed"
            else:
                result["mode"] = "hf_jobs_async"
                result["message"] = (
                    "Training submitted to Hugging Face Jobs — metrics will land on the model repo "
                    "and Trackio when the job completes."
                )
        else:
            train = await asyncio.to_thread(_step_train_local, c)
            result["steps"]["train_local"] = train
            if not train.get("ok"):
                result["error"] = "local training failed"
                return result

            report = _read_report(c.local_report_path)
            result["report"] = report
            if report.get("status") == "insufficient_label_variance":
                result["ok"] = True
                result["warning"] = report.get("message")
            elif int(report.get("train_rows") or 0) < c.min_train_rows:
                result["ok"] = False
                result["error"] = f"Too few train rows ({report.get('train_rows')})"
            else:
                eval_push = await asyncio.to_thread(_step_push_eval, c, report)
                result["steps"]["push_eval"] = eval_push
                if pool is not None:
                    run_id = await _record_promo_model_run(pool, report)
                    result["m8_model_run_id"] = run_id
                result["ok"] = bool(eval_push.get("ok"))
                if not eval_push.get("ok"):
                    result["error"] = eval_push.get("error") or "eval push failed"

        if c.sync_trackio_after_run and result.get("ok"):
            sync = await asyncio.to_thread(_step_sync_trackio)
            result["steps"]["sync_trackio"] = sync

    except Exception as e:
        logger.exception("ml engine cycle failed")
        result["error"] = str(e)[:400]
    finally:
        result["finished_at"] = _now_iso()
        track.log(
            {
                "ok": result.get("ok"),
                "mode": result.get("mode", "local"),
                "error": result.get("error"),
                "steps": list((result.get("steps") or {}).keys()),
            }
        )
        track.finish()
        prev = load_engine_state()
        prev["last_run"] = result
        prev["updated_at"] = _now_iso()
        save_engine_state(prev)

    return result
