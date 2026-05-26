"""
Hugging Face Dataset Viewer-safe exports.

Parquet files with PyArrow ``uuid`` extension columns break HF's dataset inspector
(SplitsNotFoundError). Coerce UUIDs and datetimes to strings before push.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

import pandas as pd


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
