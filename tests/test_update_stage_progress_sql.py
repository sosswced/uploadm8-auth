"""Regression: asyncpg AmbiguousParameterError on stage progress writes."""

from __future__ import annotations

import inspect

from stages.db import update_stage_progress


def test_update_stage_progress_casts_stage_param_as_text():
    src = inspect.getsource(update_stage_progress)
    assert "$2::text" in src
    assert "jsonb_build_object('last_processing_stage', $2::text)" in src
    assert "WHERE id = $1::uuid" in src
    # Must not assign bare $2 to a varchar column alongside $2::text (ambiguous).
    assert "SET processing_stage    = $2," not in src
    assert "SET processing_stage = $2," not in src


def test_update_stage_progress_never_regresses_percent():
    src = inspect.getsource(update_stage_progress)
    assert "GREATEST" in src
    assert "$3::int >= COALESCE(processing_progress, 0)" in src


def test_update_stage_progress_logs_failures_not_silent():
    src = inspect.getsource(update_stage_progress)
    assert "update_stage_progress failed" in src
