from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

_TRACE_KEY = "provider_error_trace"
_MAX_ENTRIES = 80
_MAX_JSON_CHARS = 180_000
_PROVIDER_DOCS = {
    "google_vision": "https://cloud.google.com/vision/docs/reference/rest",
    "google_video_intelligence": "https://cloud.google.com/video-intelligence/docs/reference/rest",
    "twelvelabs": "https://docs.twelvelabs.io/api-reference",
    "pikzels": "https://docs.pikzels.com/",
}


def _safe_str(v: Any, *, max_chars: int = 1200) -> str:
    s = "" if v is None else str(v)
    if len(s) <= max_chars:
        return s
    return s[: max_chars - len("...[truncated]")] + "...[truncated]"


def append_provider_error(
    ctx: Any,
    *,
    provider: str,
    stage: str,
    operation: str,
    message: Any,
    http_status: Optional[int] = None,
    provider_code: Optional[str] = None,
    provider_request_id: Optional[str] = None,
    url: Optional[str] = None,
    response_body_snippet: Optional[str] = None,
    exception_type: Optional[str] = None,
    result_tier: str = "failed",
) -> None:
    """Append normalized provider error row into ctx.output_artifacts[_TRACE_KEY].

    ``result_tier``: ``ok`` | ``degraded`` | ``failed`` — used by admin trace / ops
    (``failed`` = user-visible or blocking; ``degraded`` = soft skip / partial data).
    """
    arts = getattr(ctx, "output_artifacts", None)
    if not isinstance(arts, dict):
        return
    raw = arts.get(_TRACE_KEY)
    rows = []
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                rows = parsed
        except Exception:
            rows = []

    row: Dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "provider": str(provider or "").strip().lower(),
        "stage": str(stage or "").strip(),
        "operation": str(operation or "").strip(),
        "message": _safe_str(message, max_chars=1400),
        "upload_id": str(getattr(ctx, "upload_id", "") or ""),
        "result_tier": str(result_tier or "failed").strip().lower(),
    }
    p = row["provider"]
    if p in _PROVIDER_DOCS:
        row["provider_docs"] = _PROVIDER_DOCS[p]
    if http_status is not None:
        row["http_status"] = int(http_status)
    if provider_code:
        row["provider_code"] = _safe_str(provider_code, max_chars=120)
    if provider_request_id:
        row["provider_request_id"] = _safe_str(provider_request_id, max_chars=180)
    if url:
        row["url"] = _safe_str(url, max_chars=500)
    if response_body_snippet:
        row["response_body_snippet"] = _safe_str(response_body_snippet, max_chars=1600)
    if exception_type:
        row["exception_type"] = _safe_str(exception_type, max_chars=120)

    rows.append(row)
    if len(rows) > _MAX_ENTRIES:
        rows = rows[-_MAX_ENTRIES:]
    try:
        arts[_TRACE_KEY] = json.dumps(rows, default=str)[:_MAX_JSON_CHARS]
    except Exception:
        return


def normalize_provider_trace_rows(rows: Any) -> List[Dict[str, Any]]:
    """Return a list of trace dicts with ``result_tier`` defaulted for admin UIs."""
    if not isinstance(rows, list):
        return []
    out: List[Dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        row = dict(r)
        tier = str(row.get("result_tier") or "failed").strip().lower()
        if tier not in ("ok", "degraded", "failed"):
            tier = "failed"
        row["result_tier"] = tier
        out.append(row)
    return out


__all__ = ["append_provider_error", "normalize_provider_trace_rows"]
