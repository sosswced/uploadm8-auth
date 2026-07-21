"""Unit coverage for local-dev gap closures (billing UUID, trackio nest, HF pickle)."""

from __future__ import annotations

import os

import pandas as pd


def test_coerce_updated_by_uuid_rejects_calibration_labels():
    from services.billing_service_weights import coerce_updated_by_uuid

    assert coerce_updated_by_uuid("calibration:2026-07-v3") is None
    assert coerce_updated_by_uuid("sync_from_code") is None
    assert coerce_updated_by_uuid(None) is None
    assert coerce_updated_by_uuid("") is None
    uid = "0af99456-1002-49f8-8554-e4d4405e5884"
    assert coerce_updated_by_uuid(uid) == uid


def test_trackio_nested_disable_blocks_env():
    from services.ml_observability import trackio_env_enabled

    prev_nested = os.environ.get("UM8_TRACKIO_NESTED_DISABLE")
    prev_project = os.environ.get("TRACKIO_PROJECT")
    try:
        os.environ["TRACKIO_PROJECT"] = "uploadm8-ml"
        os.environ.pop("UM8_TRACKIO_NESTED_DISABLE", None)
        assert trackio_env_enabled() is True
        os.environ["UM8_TRACKIO_NESTED_DISABLE"] = "1"
        assert trackio_env_enabled() is False
    finally:
        if prev_nested is None:
            os.environ.pop("UM8_TRACKIO_NESTED_DISABLE", None)
        else:
            os.environ["UM8_TRACKIO_NESTED_DISABLE"] = prev_nested
        if prev_project is None:
            os.environ.pop("TRACKIO_PROJECT", None)
        else:
            os.environ["TRACKIO_PROJECT"] = prev_project


def test_datasets_pickle_bug_detector():
    from services.hf_dataset_export import _is_datasets_pickle_bug

    assert _is_datasets_pickle_bug(
        TypeError("Pickler._batch_setitems() takes 2 positional arguments but 3 were given")
    )
    assert _is_datasets_pickle_bug(
        RuntimeError("when serializing datasets.table.InMemoryTable state")
    )
    assert not _is_datasets_pickle_bug(ValueError("unrelated"))


def test_push_dataframe_to_hub_falls_back_on_pickle_bug(monkeypatch):
    from services import hf_dataset_export as mod

    calls = {"parquet": 0}

    class BoomDataset:
        @staticmethod
        def from_pandas(*_a, **_k):
            raise TypeError(
                "Pickler._batch_setitems() takes 2 positional arguments but 3 were given"
            )

    class FakeDatasets:
        Dataset = BoomDataset

    def fake_parquet(*_a, **_k):
        calls["parquet"] += 1
        return {"ok": True, "mode": "parquet", "rows": 1, "repo_id": "org/ds"}

    monkeypatch.setitem(__import__("sys").modules, "datasets", FakeDatasets)
    monkeypatch.setattr(mod, "push_parquet_to_hub", fake_parquet)

    df = pd.DataFrame([{"user_id": "a", "converted_7d": 1}])
    out = mod.push_dataframe_to_hub(df, repo_id="org/ds", token="x" * 20, split="train")
    assert out["ok"] is True
    assert out["mode"] == "parquet"
    assert calls["parquet"] == 1
