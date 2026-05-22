"""
ACRCloud music identification (Identify API v1).

Uses multipart POST with HMAC-SHA1 signature. Configure via environment:

- ACRCLOUD_HOST — e.g. identify-us-west-2.acrcloud.com (no scheme)
- ACRCLOUD_ACCESS_KEY / ACR_ACCESS_KEY
- ACRCLOUD_ACCESS_SECRET / ACR_ACCESS_SECRET

Docs: https://docs.acrcloud.com/reference/identification-api/identification-api
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger("uploadm8-worker.acrcloud")

# ACR recommends short samples; keep upload small and fast.
ACR_MAX_SAMPLE_BYTES = int(os.environ.get("ACRCLOUD_MAX_SAMPLE_BYTES", "4800000"))  # < 5 MB
ACR_TIMEOUT_SECS = float(os.environ.get("ACRCLOUD_TIMEOUT_SECS", "25"))


def acrcloud_credentials_from_env() -> Optional[tuple[str, str, str]]:
    """Return (host, access_key, access_secret) if fully configured."""
    host = (os.environ.get("ACRCLOUD_HOST") or os.environ.get("ACR_HOST") or "").strip()
    key = (
        os.environ.get("ACRCLOUD_ACCESS_KEY")
        or os.environ.get("ACR_ACCESS_KEY")
        or ""
    ).strip()
    secret = (
        os.environ.get("ACRCLOUD_ACCESS_SECRET")
        or os.environ.get("ACR_ACCESS_SECRET")
        or os.environ.get("ACRCLOUD_SECRET_KEY")
        or ""
    ).strip()
    if not host or not key or not secret:
        return None
    host = host.replace("https://", "").replace("http://", "").split("/")[0].strip()
    return host, key, secret


def _acr_signature(access_key: str, access_secret: str, data_type: str) -> tuple[str, str]:
    http_method = "POST"
    http_uri = "/v1/identify"
    signature_version = "1"
    timestamp = str(int(time.time()))
    string_to_sign = "\n".join(
        [http_method, http_uri, access_key, data_type, signature_version, timestamp]
    )
    digest = hmac.HMAC(
        access_secret.encode("utf-8"),
        string_to_sign.encode("utf-8"),
        hashlib.sha1,
    ).digest()
    sign = base64.b64encode(digest).decode("ascii")
    return timestamp, sign


def parse_acr_identify_response(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize ACR JSON into audio_context / scene_graph friendly fields.

    Returns keys: music_detected, music_title, music_artist, music_genre,
    copyright_risk, music_fingerprint (best-effort id string), acr_raw_status_code.
    """
    out: Dict[str, Any] = {
        "music_detected": False,
        "music_title": "",
        "music_artist": "",
        "music_genre": "",
        "copyright_risk": False,
        "music_fingerprint": None,
        "acr_raw_status_code": None,
    }
    try:
        st = payload.get("status") or {}
        code = st.get("code")
        out["acr_raw_status_code"] = code
        if code not in (0, None):
            return out
        meta = payload.get("metadata") or {}
        tracks = meta.get("music")
        if not isinstance(tracks, list) or not tracks:
            return out
        hit = tracks[0]
        if not isinstance(hit, dict):
            return out
        title = str(hit.get("title") or "").strip()
        artists = hit.get("artists") or []
        artist_name = ""
        if isinstance(artists, list) and artists and isinstance(artists[0], dict):
            artist_name = str(artists[0].get("name") or "").strip()
        genres = hit.get("genres") or []
        genre = ""
        if isinstance(genres, list) and genres and isinstance(genres[0], dict):
            genre = str(genres[0].get("name") or "").strip()
        score = float(hit.get("score") or 0)
        out["music_detected"] = bool(title or artist_name or score >= 50)
        out["music_title"] = title[:300]
        out["music_artist"] = artist_name[:300]
        out["music_genre"] = genre[:120]
        # Commercial catalogue match → treat as third-party / rights-sensitive.
        out["copyright_risk"] = out["music_detected"] and (score >= 40 or bool(title))
        ext = hit.get("external_ids")
        if isinstance(ext, dict):
            for k in ("isrc", "upc", "acrid", "spotify", "deezer"):
                v = ext.get(k)
                if v:
                    out["music_fingerprint"] = f"{k}:{v}"
                    break
        if out["music_fingerprint"] is None and hit.get("acrid"):
            out["music_fingerprint"] = str(hit.get("acrid"))
    except Exception as e:
        logger.debug("ACRCloud response parse skipped: %s", e)
    return out


def identify_music_file_sync(
    audio_path: Path,
    *,
    mime: str = "audio/wav",
) -> Dict[str, Any]:
    """
    Blocking call: POST sample to ACRCloud /v1/identify.

    On failure returns empty dict {} (caller treats as no match).
    """
    cred = acrcloud_credentials_from_env()
    if not cred:
        return {}
    host, access_key, access_secret = cred
    if not audio_path.is_file():
        return {}

    raw = audio_path.read_bytes()
    if len(raw) > ACR_MAX_SAMPLE_BYTES:
        raw = raw[:ACR_MAX_SAMPLE_BYTES]

    data_type = "audio"
    timestamp, signature = _acr_signature(access_key, access_secret, data_type)
    url = f"https://{host}/v1/identify"
    fields = {
        "access_key": access_key,
        "sample_bytes": str(len(raw)),
        "data_type": data_type,
        "signature_version": "1",
        "signature": signature,
        "timestamp": timestamp,
    }
    try:
        with httpx.Client(timeout=ACR_TIMEOUT_SECS) as client:
            resp = client.post(
                url,
                data=fields,
                files={"sample": (audio_path.name, raw, mime)},
            )
        if resp.status_code != 200:
            logger.warning(
                "ACRCloud HTTP %s: %s",
                resp.status_code,
                (resp.text or "")[:400],
            )
            return {}
        payload = resp.json()
    except (httpx.HTTPError, json.JSONDecodeError, OSError, ValueError) as e:
        logger.warning("ACRCloud identify failed: %s", e)
        return {}

    if not isinstance(payload, dict):
        return {}
    return parse_acr_identify_response(payload)
