#!/usr/bin/env python3
"""Verify all 22 upload/ML implementation phases are wired in code (static audit)."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

try:
    from dotenv import load_dotenv

    load_dotenv(_REPO / ".env")
except ImportError:
    pass


def _read(rel: str) -> str:
    p = _REPO / rel
    return p.read_text(encoding="utf-8", errors="replace") if p.is_file() else ""


def _ok(label: str, cond: bool, detail: str = "") -> tuple[bool, str]:
    status = "OK" if cond else "MISSING"
    msg = f"[{status}] {label}"
    if detail:
        msg += f" — {detail}"
    return cond, msg


def main() -> int:
    worker = _read("worker.py")
    checks: list[tuple[bool, str]] = []

    checks.append(_ok("1 single-pass watermark", "WATERMARK_SINGLE_PASS" in worker and "watermark_single_pass" in worker))
    checks.append(_ok("2 pipeline manifest", "init_pipeline_diag" in worker and "persist_pipeline_manifest" in worker))
    checks.append(_ok("3 unified stage timeouts", "pipeline_stage_budgets" in worker and "STAGE_TIMEOUT_CAPTION" in worker))
    checks.append(_ok("4 twelve labs parallel", "TWELVE_LABS_PARALLEL" in worker))
    checks.append(_ok("5 redis streams", "xreadgroup_one" in worker and "run_stream_reclaim_loop" in worker))
    checks.append(_ok("6 durable upload funnel", "upload_funnel_events" in _read("migrations/runtime_migrations.py")))
    checks.append(_ok("7 ml uv/datasets", "_script_cmd" in _read("services/ml_engine.py") and "_datasets_available" in _read("services/ml_engine.py")))
    checks.append(_ok("8 ml failure alerts", "ml_engine_cycle_failed" in _read("services/ml_engine.py")))
    checks.append(_ok("9 promo label backfill", "promo_label_backfill" in _read("services/ml_engine.py")))
    checks.append(_ok("10 content-success loop", "run_content_success" in _read("services/ml_engine_config.py")))
    checks.append(_ok("11 m8 publish-hour priors", "train_m8_publish_hour_priors" in _read("services/ml_engine.py")))
    checks.append(_ok("12 model artifact sync", "sync_ml_models_from_hub" in _read("services/ml_model_sync.py")))
    checks.append(_ok("13 promo marketing wire", "score_user_propensity" in _read("services/marketing_execution.py")))
    checks.append(_ok("14 content rankings UI", "content_rankings" in _read("services/ai_insights_hub.py") and "renderContentRankings" in _read("frontend/js/smart-insights-page.js")))
    checks.append(_ok("15 runtime hotness", "score_presign_init" in _read("routers/uploads_lifecycle.py") and "score_upload_context" in worker))
    checks.append(_ok("16 evidence coach hints", "coach_hints" in worker and "upload_evidence" in _read("services/growth_intelligence.py")))
    checks.append(_ok("17 publish quality UX", "publish_quality_notice" in _read("frontend/queue.html")))
    checks.append(_ok("18 selective AI trace", "ai_trace_enabled_for_ctx" in _read("services/pipeline_ai_trace.py")))
    checks.append(_ok("19 m8 strategy context", "build_m8_strategy_context" in _read("stages/m8_strategy_context.py")))
    checks.append(_ok("20 heavy-slot semaphore", "_heavy_semaphore" in worker))
    checks.append(_ok("21 worker lane ops health", "worker_lane" in _read("routers/ops.py")))
    checks.append(_ok("22 deferred publish checkpoint", "try_resume_from_checkpoint" in worker and "save_post_publish_checkpoint" in worker))

    from core.pipeline_env_defaults import effective_pipeline_env

    env = effective_pipeline_env()
    checks.append(_ok("env defaults centralized", env.get("WATERMARK_SINGLE_PASS") is True))
    checks.append(_ok("funnel terminal helper", "emit_funnel_terminal_if_needed" in _read("services/upload_funnel.py")))

    print("UploadM8 — 22-phase implementation verification\n")
    failed = 0
    for ok, msg in checks:
        print(msg)
        if not ok:
            failed += 1
    print()
    if failed:
        print(f"{failed} check(s) failed.")
        return 1
    print("All phase checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
