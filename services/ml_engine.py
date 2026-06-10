"""
UploadM8 ML / AI engine — automated dataset → Hub → train → eval → Trackio cycle.

Runs locally by default; optional Hugging Face Jobs GPU/CPU offload via
``UM8_ML_ENGINE_USE_HF_JOBS=1`` (see https://huggingface.co/docs/huggingface_hub/guides/jobs).
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import asyncpg

from services.ml_engine_config import MLEngineConfig, get_ml_engine_config, ml_engine_public_dict
from services.ml_eval_hub import (
    ensure_model_repo,
    push_content_card_metrics,
    push_content_dataset_eval_yaml,
    push_content_eval_results,
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


def _which_uv() -> Optional[str]:
    return shutil.which("uv")


def _script_cmd(script_rel: str) -> List[str]:
    """Prefer ``uv run`` for PEP 723 scripts (isolated deps e.g. datasets)."""
    script = str(_REPO_ROOT / script_rel)
    uv = _which_uv()
    if uv:
        return [uv, "run", script]
    return [sys.executable, script]


def _datasets_available() -> bool:
    try:
        import datasets  # noqa: F401

        return True
    except ImportError:
        return False


async def _alert_ml_engine_failure(
    pool: Optional[asyncpg.Pool],
    result: Dict[str, Any],
) -> None:
    """Ops incident when cycle fails for reasons other than blocked_on_data."""
    if result.get("ok") or result.get("skipped"):
        return
    cycle_status = str(result.get("cycle_status") or result.get("status") or "")
    if cycle_status == "blocked_on_data":
        return
    if pool is None:
        return
    try:
        from services.ops_incidents import record_operational_incident

        err = str(result.get("error") or result.get("reason") or "unknown")[:800]
        build = (result.get("steps") or {}).get("build_dataset") or {}
        build_tail = (build.get("stderr_tail") or build.get("stdout_tail") or "").strip()[-600:]
        await record_operational_incident(
            pool,
            source="ml_engine",
            incident_type="ml_engine_cycle_failed",
            subject="ML engine cycle failed",
            body=f"error={err}\nbuild_tail={build_tail}",
            details={
                "cycle_status": cycle_status,
                "error": err,
                "steps": list((result.get("steps") or {}).keys()),
                "finished_at": result.get("finished_at"),
            },
            dedupe_key="ml_engine:cycle_failed",
            dedupe_seconds=3600,
        )
    except Exception as e:
        logger.warning("ml engine failure alert skipped: %s", e)


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


async def _record_content_model_run(pool: asyncpg.Pool, report: Dict[str, Any]) -> Optional[str]:
    if report.get("status") in ("insufficient_label_variance", "insufficient_rows"):
        return None
    run_id = uuid.uuid4()
    metrics = {k: v for k, v in report.items() if k != "rankings"}
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
            str(report.get("model") or "content_success_v1")[:40],
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


def _step_build_dataset(cfg: MLEngineConfig, lookback_days: Optional[int] = None) -> Dict[str, Any]:
    lb = int(lookback_days or cfg.dataset_lookback_days)
    cmd = [
        *_script_cmd("scripts/build_promo_training_dataset.py"),
        "--lookback-days",
        str(lb),
        "--limit",
        str(cfg.dataset_limit),
        "--output",
        cfg.local_dataset_path,
    ]
    if cfg.dataset_repo:
        cmd.extend(["--push-to", cfg.dataset_repo, "--split", "train"])
    return _run_subprocess(cmd)


def _step_train_local(cfg: MLEngineConfig) -> Dict[str, Any]:
    model_out = str(_REPO_ROOT / "data" / "ml" / "promo_uplift_model.joblib")
    cmd = [
        *_script_cmd("scripts/train_promo_uplift_baseline.py"),
        "--input",
        cfg.local_dataset_path,
        "--report-out",
        cfg.local_report_path,
        "--model-out",
        model_out,
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


def _step_build_content_dataset(cfg: MLEngineConfig, lookback_days: Optional[int] = None) -> Dict[str, Any]:
    lb = int(lookback_days or cfg.dataset_lookback_days)
    cmd = [
        *_script_cmd("scripts/build_content_success_dataset.py"),
        "--lookback-days",
        str(lb),
        "--limit",
        str(cfg.dataset_limit),
        "--output",
        cfg.content_local_dataset_path,
    ]
    if cfg.content_dataset_repo:
        cmd.extend(["--push-to", cfg.content_dataset_repo, "--split", "train"])
    return _run_subprocess(cmd)


def _step_train_content(cfg: MLEngineConfig) -> Dict[str, Any]:
    model_out = str(_REPO_ROOT / "data" / "ml" / "content_success_model.joblib")
    cmd = [
        *_script_cmd("scripts/train_content_success_model.py"),
        "--input",
        cfg.content_local_dataset_path,
        "--report-out",
        cfg.content_local_report_path,
        "--model-out",
        model_out,
    ]
    return _run_subprocess(cmd)


def _step_push_content_eval(cfg: MLEngineConfig, report: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"ok": True, "skipped": []}
    if not cfg.content_dataset_repo or not cfg.content_model_repo:
        out["ok"] = False
        out["error"] = "content_dataset_repo and content_model_repo required for content eval push"
        return out
    try:
        ensure_model_repo(cfg.content_model_repo)
        try:
            push_content_dataset_eval_yaml(cfg.content_dataset_repo)
            out["dataset_eval_yaml"] = True
        except Exception as e:
            out["dataset_eval_yaml_error"] = str(e)[:300]
        eval_path = push_content_eval_results(
            cfg.content_model_repo,
            dataset_repo=cfg.content_dataset_repo,
            report=report,
        )
        out["eval_results_path"] = eval_path
        push_content_card_metrics(cfg.content_model_repo, report)
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


_INSUFFICIENT = ("insufficient_label_variance", "insufficient_rows")


def _widen_lookbacks(c: MLEngineConfig) -> list:
    """Lookback windows to try, doubling up to the cold-start cap."""
    lbs = [int(c.dataset_lookback_days)]
    if c.cold_start_auto_widen:
        w = int(c.dataset_lookback_days)
        while w < c.cold_start_max_lookback_days:
            w = min(int(c.cold_start_max_lookback_days), w * 2)
            if w != lbs[-1]:
                lbs.append(w)
    return lbs


_CHAMPION_EPSILON = 0.005


async def _passes_champion_gate(
    pool: Optional[asyncpg.Pool], report: Dict[str, Any], task: str
) -> tuple:
    """Promote only if the challenger beats the last published champion on ROC AUC.

    m8_model_runs records only promoted runs, so the latest row for a task is the
    reigning champion. No champion → first model is promoted.
    """
    if pool is None:
        return True, None
    new_roc = float(report.get("roc_auc") or 0.0)
    try:
        row = await pool.fetchrow(
            """
            SELECT metrics
            FROM m8_model_runs
            WHERE train_config->>'task' = $1
            ORDER BY trained_at DESC
            LIMIT 1
            """,
            task,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("champion gate query failed (%s); promoting by default", e)
        return True, None
    if not row or row.get("metrics") in (None, ""):
        return True, None
    raw = row["metrics"]
    try:
        champ = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except Exception:
        return True, None
    champ_roc = float(champ.get("roc_auc") or 0.0)
    if new_roc + 1e-9 >= champ_roc + _CHAMPION_EPSILON:
        return True, None
    return False, (
        f"roc_auc {new_roc:.3f} did not beat champion {champ_roc:.3f} (+{_CHAMPION_EPSILON})"
    )


async def _run_promo_loop(
    c: MLEngineConfig, pool: Optional[asyncpg.Pool], result: Dict[str, Any]
) -> None:
    """Promo-targeting loop. Populates ``result`` in place; never aborts the cycle."""
    if not c.dataset_repo or not c.model_repo:
        result["steps"]["build_dataset"] = {"ok": False, "skipped": "promo dataset/model repo unset"}
        return

    # HF Jobs path stays single-shot (async training elsewhere).
    if c.use_hf_jobs:
        build = await asyncio.to_thread(_step_build_dataset, c)
        result["steps"]["build_dataset"] = build
        if not build.get("ok"):
            result["error"] = "dataset build failed"
            detail = (build.get("stderr_tail") or build.get("stdout_tail") or "").strip()
            if detail:
                result["build_detail"] = detail[-800:]
            return
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
        return

    report: Optional[Dict[str, Any]] = None
    last_build: Optional[Dict[str, Any]] = None
    last_train: Optional[Dict[str, Any]] = None
    used_lookback = c.dataset_lookback_days

    for lb in _widen_lookbacks(c):
        last_build = await asyncio.to_thread(_step_build_dataset, c, lb)
        if not last_build.get("ok"):
            result["steps"]["build_dataset"] = last_build
            result["error"] = "dataset build failed"
            detail = (last_build.get("stderr_tail") or last_build.get("stdout_tail") or "").strip()
            if detail:
                result["build_detail"] = detail[-800:]
                logger.warning("ml engine promo build failed (rc=%s): %s", last_build.get("returncode"), detail[-1200:])
            return
        last_train = await asyncio.to_thread(_step_train_local, c)
        if not last_train.get("ok"):
            result["steps"]["build_dataset"] = last_build
            result["steps"]["train_local"] = last_train
            result["error"] = "local training failed"
            return
        report = _read_report(c.local_report_path)
        used_lookback = lb
        if report.get("status") not in _INSUFFICIENT:
            break

    seeded = False
    if report is not None and report.get("status") in _INSUFFICIENT and c.seed_bootstrap:
        try:
            from services.ml_seed import seed_promo_parquet

            added = seed_promo_parquet(c.local_dataset_path, c.seed_rows)
            seeded = True
            last_train = await asyncio.to_thread(_step_train_local, c)
            report = _read_report(c.local_report_path)
            logger.info("promo seed bootstrap added %s rows", added)
        except Exception as e:
            logger.warning("promo seed bootstrap failed: %s", e)

    result["steps"]["build_dataset"] = last_build
    result["steps"]["train_local"] = last_train
    result["report"] = report
    result["lookback_days_used"] = used_lookback
    if seeded:
        result["seeded"] = True
    await _finalize_publish(
        c, pool, result, report, seeded,
        task="promo_targeting_uplift_baseline",
        push_step=_step_push_eval,
        record_run=_record_promo_model_run,
    )


async def _run_content_loop(
    c: MLEngineConfig, pool: Optional[asyncpg.Pool]
) -> Dict[str, Any]:
    """Content-success loop, independent of promo. Returns its own sub-result."""
    content: Dict[str, Any] = {"ok": False}
    if not c.content_dataset_repo:
        content["skipped"] = "content_dataset_repo unset"
        return content
    try:
        report: Optional[Dict[str, Any]] = None
        last_build: Optional[Dict[str, Any]] = None
        last_train: Optional[Dict[str, Any]] = None
        used_lookback = c.dataset_lookback_days

        for lb in _widen_lookbacks(c):
            last_build = await asyncio.to_thread(_step_build_content_dataset, c, lb)
            if not last_build.get("ok"):
                content["build_dataset"] = last_build
                content["error"] = "content dataset build failed"
                detail = (last_build.get("stderr_tail") or last_build.get("stdout_tail") or "").strip()
                if detail:
                    content["build_detail"] = detail[-800:]
                return content
            last_train = await asyncio.to_thread(_step_train_content, c)
            if not last_train.get("ok"):
                content["build_dataset"] = last_build
                content["train_local"] = last_train
                content["error"] = "content training failed"
                return content
            report = _read_report(c.content_local_report_path)
            used_lookback = lb
            if report.get("status") not in _INSUFFICIENT:
                break

        seeded = False
        if report is not None and report.get("status") in _INSUFFICIENT and c.seed_bootstrap:
            try:
                from services.ml_seed import seed_content_parquet

                added = seed_content_parquet(c.content_local_dataset_path, c.seed_rows)
                seeded = True
                last_train = await asyncio.to_thread(_step_train_content, c)
                report = _read_report(c.content_local_report_path)
                logger.info("content seed bootstrap added %s rows", added)
            except Exception as e:
                logger.warning("content seed bootstrap failed: %s", e)

        content["build_dataset"] = last_build
        content["train_local"] = last_train
        content["report"] = report
        content["lookback_days_used"] = used_lookback
        if seeded:
            content["seeded"] = True
        await _finalize_publish(
            c, pool, content, report, seeded,
            task="content_success_hotness",
            push_step=_step_push_content_eval,
            record_run=_record_content_model_run,
        )
    except Exception as e:
        logger.exception("content-success loop failed")
        content["error"] = str(e)[:400]
    return content


async def _finalize_publish(
    c: MLEngineConfig,
    pool: Optional[asyncpg.Pool],
    out: Dict[str, Any],
    report: Optional[Dict[str, Any]],
    seeded: bool,
    *,
    task: str,
    push_step,
    record_run,
) -> None:
    """Shared train→publish gate: data status, quality threshold, champion gate."""
    if report is None:
        out["ok"] = False
        out["error"] = "no training report"
        return

    status = report.get("status")
    if status in _INSUFFICIENT:
        # Keep the cycle healthy; the loop is simply waiting on real data.
        out["ok"] = True
        out["status"] = "blocked_on_data"
        out["warning"] = report.get("message")
        return

    if int(report.get("train_rows") or 0) < c.min_train_rows:
        out["ok"] = False
        out["error"] = f"Too few train rows ({report.get('train_rows')})"
        return

    roc = float(report.get("roc_auc") or 0.0)
    if seeded:
        out["ok"] = True
        out["status"] = "trained_not_published"
        out["reason_not_published"] = "seeded model not promoted to real bucket"
        return
    if roc < c.publish_min_roc_auc:
        out["ok"] = True
        out["status"] = "trained_not_published"
        out["reason_not_published"] = f"roc_auc {roc:.3f} < publish threshold {c.publish_min_roc_auc:.3f}"
        return

    promote, champ_reason = await _passes_champion_gate(pool, report, task)
    if not promote:
        out["ok"] = True
        out["status"] = "trained_not_published"
        out["reason_not_published"] = champ_reason or "did not beat champion"
        return

    eval_push = await asyncio.to_thread(push_step, c, report)
    # Promo records under result["steps"]; content keeps its own push_eval key.
    if task.startswith("promo"):
        out.setdefault("steps", {})["push_eval"] = eval_push
    else:
        out["push_eval"] = eval_push
    if pool is not None:
        out["m8_model_run_id"] = await record_run(pool, report)
    out["ok"] = bool(eval_push.get("ok"))
    out["status"] = "published" if eval_push.get("ok") else out.get("status")
    if not eval_push.get("ok"):
        out["error"] = eval_push.get("error") or "eval push failed"


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

    if not c.stack_ready and not c.content_stack_ready and not force:
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
        if c.dataset_repo and not _datasets_available() and not _which_uv():
            result["preflight_warning"] = (
                "datasets package not importable and uv not on PATH; "
                "dataset --push-to may fail — install datasets or uv"
            )
            logger.warning(result["preflight_warning"])

        if c.run_quality_scoring and pool is not None:
            from services.ml_scoring_job import run_ml_scoring_cycle

            qs = await run_ml_scoring_cycle(
                pool,
                lookback_days=c.quality_scoring_lookback_days,
                emit_trackio=False,
            )
            result["steps"]["quality_scoring"] = {"ok": qs is not None, "rows": qs}

        if pool is not None:
            try:
                from services.promo_label_backfill import backfill_promo_outcome_labels

                async with pool.acquire() as conn:
                    bf = await backfill_promo_outcome_labels(
                        conn, lookback_days=c.cold_start_max_lookback_days
                    )
                result["steps"]["promo_label_backfill"] = {"ok": True, **bf}
            except Exception as bf_e:
                result["steps"]["promo_label_backfill"] = {"ok": False, "error": str(bf_e)[:300]}

        if pool is not None:
            try:
                from services.m8_publish_hour_model import train_m8_publish_hour_priors, training_lookback_days_from_env

                m8_metrics = await train_m8_publish_hour_priors(
                    pool, lookback_days=training_lookback_days_from_env()
                )
                result["steps"]["m8_publish_hour_priors"] = {
                    "ok": bool(m8_metrics),
                    "rows": int((m8_metrics or {}).get("pci_rows") or 0),
                }
            except Exception as m8_e:
                result["steps"]["m8_publish_hour_priors"] = {"ok": False, "error": str(m8_e)[:300]}

        await _run_promo_loop(c, pool, result)

        if c.run_content_success:
            result["content"] = await _run_content_loop(c, pool)

        content_ok = bool((result.get("content") or {}).get("ok"))
        if c.sync_trackio_after_run and (result.get("ok") or content_ok):
            sync = await asyncio.to_thread(_step_sync_trackio)
            result["steps"]["sync_trackio"] = sync

    except Exception as e:
        logger.exception("ml engine cycle failed")
        result["error"] = str(e)[:400]
    finally:
        result["finished_at"] = _now_iso()
        content_status = (result.get("content") or {}).get("status")
        statuses = [s for s in (result.get("status"), content_status) if s]
        if statuses and all(s == "blocked_on_data" for s in statuses):
            result["cycle_status"] = "blocked_on_data"
        elif "published" in statuses:
            result["cycle_status"] = "published"
        elif statuses:
            result["cycle_status"] = statuses[0]
        track.log(
            {
                "ok": result.get("ok"),
                "mode": result.get("mode", "local"),
                "status": result.get("status"),
                "content_status": content_status,
                "cycle_status": result.get("cycle_status"),
                "error": result.get("error"),
                "steps": list((result.get("steps") or {}).keys()),
            }
        )
        track.finish()
        prev = load_engine_state()
        prev["last_run"] = result
        prev["updated_at"] = _now_iso()
        save_engine_state(prev)
        await _alert_ml_engine_failure(pool, result)

    return result
