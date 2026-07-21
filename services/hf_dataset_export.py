"""
Hugging Face Dataset Viewer-safe exports.

Parquet files with PyArrow ``uuid`` extension columns break HF's dataset inspector
(SplitsNotFoundError). Coerce UUIDs and datetimes to strings before push.
"""

from __future__ import annotations

import logging
import tempfile
import uuid
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def uuid_to_hf_string(value: Any) -> Optional[str]:
    """Convert UUID / 16-byte binary / string to canonical hyphenated UUID text."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, bytes) and len(value) == 16:
        try:
            return str(uuid.UUID(bytes=value))
        except ValueError:
            pass
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return str(uuid.UUID(text))
        except ValueError:
            return text
    return str(value)


def _scalar_hf_safe(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, bytes):
        as_uuid = uuid_to_hf_string(value)
        if as_uuid and len(value) == 16:
            return as_uuid
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (dict, list)):
        return value
    return value


def coerce_row_dict_for_hf(row: Dict[str, Any]) -> Dict[str, Any]:
    return {str(k): _scalar_hf_safe(v) for k, v in row.items()}


def coerce_rows_for_hf(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [coerce_row_dict_for_hf(r) for r in rows]


def _is_id_column(name: str) -> bool:
    n = name.lower()
    return n.endswith("_id") or (n.endswith("id") and n not in ("valid", "grid"))


def coerce_dataframe_for_hf(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy safe for ``to_parquet`` + Hugging Face Dataset Viewer."""
    if df is None or df.empty:
        return df.copy() if df is not None else pd.DataFrame()
    out = df.copy()
    for col in out.columns:
        series = out[col]
        dtype_str = str(series.dtype).lower()
        if _is_id_column(col) or "uuid" in dtype_str:
            out[col] = series.map(uuid_to_hf_string)
            continue
        if series.dtype == object:
            out[col] = series.map(_scalar_hf_safe)
            continue
        if pd.api.types.is_datetime64_any_dtype(series):
            out[col] = series.map(lambda x: x.isoformat() if pd.notna(x) else None)
    return out


def read_parquet_hf_safe(path: str) -> pd.DataFrame:
    """Read parquet and normalize UUID columns (fixes garbled binary IDs in viewer)."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pq.read_table(path)
        columns = []
        for i, field in enumerate(table.schema):
            col = table.column(i)
            type_str = str(field.type).lower()
            if "uuid" in type_str or _is_id_column(field.name):
                py_vals = col.to_pylist()
                str_vals = [uuid_to_hf_string(v) for v in py_vals]
                columns.append(pa.array(str_vals, type=pa.string()))
            else:
                columns.append(col)
        table = pa.table({field.name: columns[i] for i, field in enumerate(table.schema)})
        return table.to_pandas()
    except Exception:
        return coerce_dataframe_for_hf(pd.read_parquet(path))


def parquet_bytes_hf_safe(df: pd.DataFrame) -> bytes:
    import io

    safe = coerce_dataframe_for_hf(df)
    for col in safe.columns:
        if _is_id_column(col):
            safe[col] = safe[col].astype("string")
    buf = io.BytesIO()
    safe.to_parquet(buf, index=False)
    return buf.getvalue()


def _is_datasets_pickle_bug(exc: BaseException) -> bool:
    msg = str(exc)
    return (
        "_batch_setitems" in msg
        or "when serializing datasets.table" in msg
        or ("Pickler" in msg and "positional arguments" in msg)
    )


def push_parquet_to_hub(
    df: pd.DataFrame,
    *,
    repo_id: str,
    token: str,
    split: str = "train",
    private: bool = False,
) -> Dict[str, Any]:
    """Upload a Viewer-safe parquet file (avoids datasets/dill pickle on Py3.14)."""
    from huggingface_hub import HfApi

    safe = coerce_dataframe_for_hf(df)
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="um8-hf-ds-") as td:
        path = Path(td) / f"{split}.parquet"
        safe.to_parquet(path, index=False)
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=f"data/{split}.parquet",
            repo_id=repo_id,
            repo_type="dataset",
        )
    return {"ok": True, "mode": "parquet", "rows": int(len(safe)), "repo_id": repo_id}


def push_dataframe_to_hub(
    df: pd.DataFrame,
    *,
    repo_id: str,
    token: str,
    split: str = "train",
    private: bool = False,
) -> Dict[str, Any]:
    """
    Push rows to a Hub dataset.

    Prefers ``datasets.Dataset.push_to_hub``; on Python 3.14 / dill pickle bugs,
    falls back to a direct parquet upload so the ML engine cycle can continue.
    """
    if df is None or df.empty:
        return {"ok": False, "skipped": "empty", "repo_id": repo_id}
    safe = coerce_dataframe_for_hf(df)
    try:
        from datasets import Dataset

        ds = Dataset.from_pandas(safe, preserve_index=False)
        ds.push_to_hub(repo_id, token=token, split=split, private=private)
        return {"ok": True, "mode": "datasets", "rows": int(len(ds)), "repo_id": repo_id}
    except Exception as e:
        if not _is_datasets_pickle_bug(e) and not isinstance(e, TypeError):
            raise
        logger.warning(
            "datasets Hub push failed (%s); falling back to parquet upload for %s",
            str(e)[:200],
            repo_id,
        )
        return push_parquet_to_hub(
            safe, repo_id=repo_id, token=token, split=split, private=private
        )
