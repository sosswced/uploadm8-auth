"""Regression: do not re-bind module ``r2_stage`` inside ``run_processing_pipeline``.

A nested ``from stages import r2 as r2_stage`` makes ``r2_stage`` a local cell for
the whole function. Nested download closures then raise:

  cannot access free variable 'r2_stage' where it is not associated with a value
  in enclosing scope
"""

from __future__ import annotations

from pathlib import Path


def test_worker_single_module_level_r2_stage_import():
    text = Path("worker.py").read_text(encoding="utf-8")
    assert text.count("from stages import r2 as r2_stage") == 1


def test_run_processing_pipeline_r2_stage_not_local_cell():
    import worker

    co = worker.run_processing_pipeline.__code__
    assert "r2_stage" not in (co.co_varnames or ())
    assert "r2_stage" not in (co.co_cellvars or ())
