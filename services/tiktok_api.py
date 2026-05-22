"""
TikTok Open API v2 — URL and response envelope rules (single source for app, worker, jobs).

Docs: video list / video query use ?fields= in the query string, not the JSON body.
"""
from __future__ import annotations

import logging
from typing import Any, Mapping, Optional, Tuple
from urllib.parse import quote

import httpx

logger = logging.getLogger(__name__)

TIKTOK_USER_INFO_URL = "https://open.tiktokapis.com/v2/user/info/"
# Scope tiers (https://developers.tiktok.com/bulletin/user-info-scope-migration):
# - user.info.basic: open_id, display_name, avatar_*
# - user.info.profile: username, bio_description, is_verified, profile_deep_link
# - user.info.stats: follower_count, following_count, likes_count, video_count
TIKTOK_USER_INFO_FIELDS_BASIC = (
    "open_id,union_id,display_name,avatar_url,avatar_url_100,avatar_large_url"
)
TIKTOK_USER_INFO_FIELDS_PROFILE = (
    "open_id,union_id,username,display_name,avatar_url,avatar_url_100,avatar_large_url,"
    "bio_description,is_verified,profile_deep_link"
)
TIKTOK_USER_INFO_FIELDS_STATS = "follower_count,following_count,likes_count,video_count"
# Legacy alias — do not request username/stats here (breaks basic-only tokens).
TIKTOK_USER_INFO_FIELDS_FULL = TIKTOK_USER_INFO_FIELDS_PROFILE

# Single list used by catalog sync and any caller of ``tiktok_video_list_url`` — no short/long split;
# all published videos are requested with the same field set.
TIKTOK_VIDEO_LIST_FIELDS = (
    "id,title,video_description,duration,cover_image_url,share_url,"
    "view_count,like_count,comment_count,share_count,create_time"
)
TIKTOK_VIDEO_QUERY_FIELDS = "id,view_count,like_count,comment_count,share_count"


def tiktok_video_list_url() -> str:
    return (
        "https://open.tiktokapis.com/v2/video/list/"
        f"?fields={quote(TIKTOK_VIDEO_LIST_FIELDS)}"
    )


def tiktok_video_query_url() -> str:
    return (
        "https://open.tiktokapis.com/v2/video/query/"
        f"?fields={quote(TIKTOK_VIDEO_QUERY_FIELDS)}"
    )


def tiktok_envelope_error(body: Any) -> Optional[str]:
    """TikTok often returns HTTP 200 with error.code != 'ok' in JSON."""
    if not isinstance(body, dict):
        return None
    err = body.get("error")
    if not isinstance(err, dict):
        return None
    code = err.get("code")
    if code is None or str(code).lower() == "ok":
        return None
    return (err.get("message") or str(code) or "tiktok_api_error")[:500]


def tiktok_parse_oauth_token_response(resp_json: Any) -> Tuple[str, str, Optional[str], dict]:
    """
    Normalize TikTok /v2/oauth/token/ JSON (flat or wrapped under ``data``).
    Returns (access_token, open_id, refresh_token_or_none, merged_dict_for_blob).
    """
    if not isinstance(resp_json, dict):
        return "", "", None, {}
    nested = resp_json.get("data")
    src: dict = nested if isinstance(nested, dict) else resp_json
    access = str(src.get("access_token") or resp_json.get("access_token") or "").strip()
    open_id = str(src.get("open_id") or resp_json.get("open_id") or "").strip()
    refresh = src.get("refresh_token")
    if refresh is None:
        refresh = resp_json.get("refresh_token")
    rft: Optional[str]
    if refresh is None:
        rft = None
    elif isinstance(refresh, str):
        rft = refresh
    else:
        rft = str(refresh)
    return access, open_id, rft, src


def tiktok_extract_user_from_info_body(body: Any) -> dict:
    """Pull the user object from /v2/user/info/ response variants."""
    if not isinstance(body, dict):
        return {}
    data = body.get("data")
    if isinstance(data, dict):
        user = data.get("user")
        if isinstance(user, dict):
            return user
        if any(k in data for k in ("open_id", "display_name", "avatar_url", "avatar_url_100")):
            return data
    user = body.get("user")
    if isinstance(user, dict):
        return user
    return {}


def tiktok_identity_from_user_object(user_obj: Mapping[str, Any]) -> dict:
    """
    Map TikTok user JSON to platform_tokens-style fields.
    Returns keys: account_id, account_name, account_username, account_avatar (may be "").
    """
    if not isinstance(user_obj, Mapping):
        return {
            "account_id": "",
            "account_name": "",
            "account_username": "",
            "account_avatar": "",
        }
    incoming_open_id = (
        str(user_obj.get("open_id") or "").strip()
        or str(user_obj.get("user_id") or "").strip()
    )
    display_name = (
        str(user_obj.get("display_name") or "").strip()
        or str(user_obj.get("name") or "").strip()
        or str(user_obj.get("nickname") or "").strip()
    )
    username = (
        str(user_obj.get("username") or "").strip()
        or str(user_obj.get("unique_id") or "").strip()
        or str(user_obj.get("username_normalized") or "").strip()
    )
    avatar_url = (
        str(user_obj.get("avatar_large_url") or "").strip()
        or str(user_obj.get("avatar_url_100") or "").strip()
        or str(user_obj.get("avatar_url_hd") or "").strip()
        or str(user_obj.get("avatar_url") or "").strip()
        or str(user_obj.get("profile_image") or "").strip()
        or str(user_obj.get("avatar") or "").strip()
    )
    return {
        "account_id": incoming_open_id,
        "account_name": display_name,
        "account_username": username,
        "account_avatar": avatar_url,
    }


def _merge_tiktok_user_objects(base: Mapping[str, Any], extra: Mapping[str, Any]) -> dict:
    out = dict(base) if isinstance(base, Mapping) else {}
    if not isinstance(extra, Mapping):
        return out
    for key, val in extra.items():
        if val is None:
            continue
        if isinstance(val, str) and not val.strip():
            continue
        if key not in out or not out.get(key):
            out[key] = val
    return out


async def _tiktok_user_info_get(
    client: httpx.AsyncClient,
    access_token: str,
    fields: str,
) -> Tuple[Optional[dict], str]:
    """Single GET /v2/user/info/; returns (user_obj or None, log_hint)."""
    headers = {
        "Authorization": f"Bearer {access_token.strip()}",
        "Accept": "application/json",
    }
    try:
        resp = await client.get(
            TIKTOK_USER_INFO_URL,
            params={"fields": fields},
            headers=headers,
        )
        try:
            body = resp.json() if resp.content else {}
        except Exception:
            body = {}
        env_err = tiktok_envelope_error(body)
        if resp.status_code != 200 or env_err:
            return None, (
                f"GET fields={fields[:48]} HTTP {resp.status_code} envelope={env_err!r}"
            )
        user_obj = tiktok_extract_user_from_info_body(body)
        if user_obj:
            return user_obj, "ok"
        return None, f"GET fields={fields[:48]} empty user payload"
    except Exception as e:
        return None, f"GET fields={fields[:48]} exception: {type(e).__name__}: {e}"


async def fetch_tiktok_user_profile_for_oauth(
    client: httpx.AsyncClient,
    access_token: str,
) -> dict:
    """
    Fetch display name + avatar after OAuth using ``user.info.basic`` fields only.

    TikTok scope migration: ``open_id``, ``display_name``, and ``avatar_*`` live under
    ``user.info.basic``. The old code incorrectly included ``username`` in the basic
    field list; ``username`` requires ``user.info.profile`` and caused the whole
    ``/v2/user/info/`` call to fail — empty name/avatar and ui-avatars placeholders.
    """
    if not access_token or not str(access_token).strip():
        return tiktok_identity_from_user_object({})

    user_obj, hint = await _tiktok_user_info_get(
        client, access_token, TIKTOK_USER_INFO_FIELDS_BASIC
    )
    if user_obj:
        ident = tiktok_identity_from_user_object(user_obj)
        logger.info(
            "TikTok user/info (basic) has_name=%s has_avatar=%s open_id=%s",
            bool(ident.get("account_name")),
            bool(ident.get("account_avatar")),
            bool(ident.get("account_id")),
        )
        return ident

    logger.warning("TikTok user/info (basic) failed: %s", hint)
    return tiktok_identity_from_user_object({})
