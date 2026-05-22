"""Stable HTTP error payloads for API clients (machine-readable `code` + `message`)."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException


def api_problem(
    status_code: int,
    *,
    code: str,
    message: str,
    **extra: Any,
) -> HTTPException:
    """
    Return a FastAPI HTTPException whose JSON body is {"detail": {code, message, ...}}.

    Keeps auth, billing, and upload surfaces consistent for integrators and the SPA.
    """
    body: dict[str, Any] = {"code": code, "message": message}
    for key, val in extra.items():
        if val is not None:
            body[key] = val
    return HTTPException(status_code=status_code, detail=body)
