"""Hashtag parsing and normalization for upload presign/complete."""

from __future__ import annotations

import json
import re
from typing import Any, List

from core.helpers import _safe_json, sanitize_hashtag_body


def _split_tags(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return []
        try:
            maybe = json.loads(s)
            if isinstance(maybe, list):
                v = maybe
            else:
                v = s
        except Exception:
            v = s
    if isinstance(v, str):
        return [p for p in re.split(r"[\s,]+", v.strip()) if p]
    if isinstance(v, (list, tuple, set)):
        return [str(x).strip() for x in v if str(x).strip()]
    s = str(v).strip()
    return [s] if s else []


def _to_hash_tags(v: Any) -> List[str]:
    out: List[str] = []
    for t in _split_tags(v):
        body = sanitize_hashtag_body(t)
        if body:
            out.append(f"#{body}")
    return out


def _normalize_hashtags_list(raw: Any) -> List[str]:
    tags = _safe_json(raw, [])
    if isinstance(tags, list):
        out: List[str] = []
        for t in tags:
            if not t:
                continue
            body = sanitize_hashtag_body(str(t))
            if body:
                out.append(f"#{body}")
        return out
    if isinstance(tags, str) and tags.strip():
        body = sanitize_hashtag_body(tags)
        return [f"#{body}"] if body else []
    return []
