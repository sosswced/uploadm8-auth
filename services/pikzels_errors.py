"""
Pikzels HTTP API — error and rate-limit documentation alignment.

Public error envelope (programmatic ``code`` + human ``message``):

  https://docs.pikzels.com/errors

Concurrency / 429 behavior (default 10 in-flight requests per API key):

  https://docs.pikzels.com/rate-limits
"""

from __future__ import annotations

from typing import Any, List

PIKZELS_DOCS_ERRORS_URL = "https://docs.pikzels.com/errors"
PIKZELS_DOCS_RATE_LIMITS_URL = "https://docs.pikzels.com/rate-limits"


def format_pikzels_error_message(data: Any, *, max_len: int = 800) -> str:
    """
    Turn a Pikzels JSON error body into a short string for logs / ``engine_error`` / toasts.

    Prefer the documented ``{"error": {"code": "…", "message": "…"}}`` shape, then fall back
    to FastAPI-style ``detail`` lists and loose ``message`` keys.
    """
    if not isinstance(data, dict):
        return "upstream_error"

    err = data.get("error")
    if isinstance(err, dict):
        code = err.get("code")
        msg = err.get("message")
        cs = str(code).strip() if code is not None else ""
        ms = str(msg).strip() if msg is not None else ""
        if cs and ms:
            return f"{cs}: {ms}"[:max_len]
        if ms:
            return ms[:max_len]
        if cs:
            return cs[:max_len]

    detail = data.get("detail")
    if isinstance(detail, list):
        parts: List[str] = []
        for it in detail[:10]:
            if isinstance(it, dict):
                parts.append(str(it.get("message") or it.get("code") or it)[:220])
            else:
                parts.append(str(it)[:220])
        if parts:
            return "; ".join(parts)[:max_len]

    for key in ("issues", "errors"):
        arr = data.get(key)
        if isinstance(arr, list) and len(arr) > 0:
            return format_pikzels_error_message({"detail": arr}, max_len=max_len)

    if detail is not None and not isinstance(detail, (dict, list)):
        return str(detail)[:max_len]

    raw = data.get("raw")
    if raw is not None and str(raw).strip():
        return str(raw).strip()[:max_len]

    msg = data.get("message") or data.get("detail")
    if msg is not None and str(msg).strip():
        return str(msg).strip()[:max_len]

    # Last resort: compact JSON so ops never see an empty "non-2xx" placeholder.
    try:
        import json

        blob = json.dumps(data, default=str)
        if blob and blob not in ("{}", "null"):
            return blob[:max_len]
    except Exception:
        pass
    return "upstream_error"
