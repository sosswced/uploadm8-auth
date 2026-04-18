"""
Async client for Pikzels public API v2 (https://api.pikzels.com).

Auth: X-Api-Key — resolved via services.pikzels_v2.resolve_public_api_key (PIKZELS_API_KEY preferred).
OpenAPI lists some XOR pairs as both required; we send an empty string for the unused side.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Tuple

import httpx

from services.pikzels_v2 import PUBLIC_BASE, resolve_public_api_key

_log = logging.getLogger("uploadm8.pikzels_v2")


def pikzels_api_key() -> str:
    """Same as resolve_public_api_key; kept for callers that import this name."""
    return resolve_public_api_key()


def pikzels_timeout_seconds() -> float:
    return float(os.environ.get("PIKZELS_API_TIMEOUT_SECONDS") or os.environ.get("PIKZELS_TIMEOUT_SECONDS", "120") or 120)


def coerce_pikzels_v2_image_base64_fields(body: Dict[str, Any]) -> None:
    """
    Pikzels v2 validates ``image_base64`` / ``*_base64`` / ``image_base64s[]`` when present:
    non-empty values must be full ``data:image/*;base64,...`` URLs (not raw base64).

    Empty strings must be **omitted** — Pikzels treats ``\"\"`` as present and fails
    ("must start with data:image/") even when ``image_url`` is the active XOR side.
    """
    for key in ("image_base64", "support_image_base64", "face_image_base64", "mask_base64"):
        if key not in body:
            continue
        v = body.get(key)
        if not isinstance(v, str):
            continue
        t = v.strip()
        if not t:
            body.pop(key, None)
            continue
        low = t.lower()
        if low.startswith("data:image") and ";base64," in low:
            continue
        body[key] = f"data:image/jpeg;base64,{t}"[:14_000_000]
    arr = body.get("image_base64s")
    if isinstance(arr, list):
        if not arr:
            body.pop("image_base64s", None)
        else:
            out: List[str] = []
            for x in arr:
                if not isinstance(x, str):
                    continue
                t = x.strip()
                if not t:
                    continue
                low = t.lower()
                if low.startswith("data:image") and ";base64," in low:
                    out.append(t[:14_000_000])
                else:
                    out.append(f"data:image/jpeg;base64,{t}"[:14_000_000])
            body["image_base64s"] = out


def normalize_url_or_base64(body: Dict[str, Any], url_key: str, b64_key: str) -> None:
    """Ensure exactly one of url / base64 is non-empty; set the other to ''."""
    u = str(body.get(url_key) or "").strip() if body.get(url_key) is not None else ""
    b = str(body.get(b64_key) or "").strip() if body.get(b64_key) is not None else ""
    if u and b:
        body[url_key], body[b64_key] = u, ""
    elif u:
        body[url_key], body[b64_key] = u, ""
    elif b:
        body[url_key], body[b64_key] = "", b
    else:
        body[url_key], body[b64_key] = "", ""


def trim_pikzonality_images(body: Dict[str, Any]) -> None:
    """
    Persona/style creation: exactly 3 refs from urls XOR base64s.

    Pikzels rejects bodies that include *both* keys (even with one empty array):
    "Provide either image_urls or image_base64s, not both".
    """
    raw_u = body.pop("image_urls", None)
    raw_b = body.pop("image_base64s", None)
    clean_u = [
        str(x).strip()
        for x in (raw_u if isinstance(raw_u, list) else [])
        if str(x).strip()
    ]
    clean_b = [
        str(x).strip()
        for x in (raw_b if isinstance(raw_b, list) else [])
        if str(x).strip()
    ]
    if len(clean_u) >= 3:
        body["image_urls"] = clean_u[:3]
    elif len(clean_b) >= 3:
        body["image_base64s"] = clean_b[:3]


async def pikzels_v2_post(path: str, json_body: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    key = pikzels_api_key()
    if not key:
        return 503, {
            "error": "missing_api_key",
            "message": "Set PIKZELS_API_KEY (preferred) or THUMB_RENDER_API_KEY for https://api.pikzels.com.",
        }
    if isinstance(json_body, dict):
        coerce_pikzels_v2_image_base64_fields(json_body)
    url = f"{PUBLIC_BASE}{path}"
    headers = {"X-Api-Key": key, "Content-Type": "application/json"}
    timeout = pikzels_timeout_seconds()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(url, headers=headers, json=json_body)
            data: Dict[str, Any]
            try:
                data = r.json() if r.content else {}
            except Exception:
                data = {"raw": (r.text or "")[:4000]}
            if r.status_code >= 400:
                _log.warning("pikzels POST %s HTTP %s: %s", path, r.status_code, str(data)[:300])
            return r.status_code, data if isinstance(data, dict) else {"data": data}
    except httpx.TimeoutException:
        return 504, {"error": "timeout", "message": "Pikzels API request timed out."}
    except httpx.HTTPError as e:
        _log.warning("pikzels POST %s failed: %s", path, e)
        return 502, {"error": "upstream", "message": str(e)}


async def pikzels_v2_get(path: str) -> Tuple[int, Dict[str, Any]]:
    key = pikzels_api_key()
    if not key:
        return 503, {
            "error": "missing_api_key",
            "message": "Set PIKZELS_API_KEY (preferred) or THUMB_RENDER_API_KEY for https://api.pikzels.com.",
        }
    url = f"{PUBLIC_BASE}{path}"
    headers = {"X-Api-Key": key}
    timeout = pikzels_timeout_seconds()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(url, headers=headers)
            try:
                data = r.json() if r.content else {}
            except Exception:
                data = {"raw": (r.text or "")[:4000]}
            return r.status_code, data if isinstance(data, dict) else {"data": data}
    except httpx.TimeoutException:
        return 504, {"error": "timeout", "message": "Pikzels API request timed out."}
    except httpx.HTTPError as e:
        _log.warning("pikzels GET %s failed: %s", path, e)
        return 502, {"error": "upstream", "message": str(e)}
