"""Normalize uploads.target_accounts TEXT[] values to platform_tokens UUID strings."""

from __future__ import annotations

import json
import uuid
from typing import Any, List


def normalize_target_account_uuids(raw: Any) -> List[str]:
    """
    uploads.target_accounts is TEXT[]; values must be platform_tokens.id UUID strings.
    Invalid entries (or a mistaken string stored as JSON) must not reach ANY($1::uuid[]).
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return []
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                raw = parsed
            else:
                return []
        except json.JSONDecodeError:
            return []
    if not isinstance(raw, (list, tuple)):
        return []
    out: List[str] = []
    seen: set[str] = set()
    for x in raw:
        if x is None:
            continue
        s = str(x).strip()
        if not s or s in seen:
            continue
        try:
            out.append(str(uuid.UUID(s)))
            seen.add(s)
        except (ValueError, AttributeError, TypeError):
            continue
    return out
