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
# Stats fields require user.info.stats; if a token lacks it, retry with basic-only.
TIKTOK_USER_INFO_FIELDS_BASIC = (
    "open_id,union_id,display_name,avatar_url,avatar_url_100,avatar_large_url"
)
TIKTOK_USER_INFO_FIELDS_FULL = TIKTOK_USER_INFO_FIELDS_BASIC + (
    ",follower_count,following_count,likes_count"
)

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


async def fetch_tiktok_user_profile_for_oauth(
    client: httpx.AsyncClient,
    access_token: str,
) -> dict:
    """
    Fetch display name + avatar after OAuth. Tries GET (documented), then POST (some configs),
    then retries with fewer fields if stats fields are rejected.
    Returns tiktok_identity_from_user_object shape (may be empty strings).
    """
    if not access_token or not str(access_token).strip():
        return tiktok_identity_from_user_object({})

    headers = {
        "Authorization": f"Bearer {access_token.strip()}",
        "Accept": "application/json",
    }
    last_log: str = ""

    for fields, label in (
        (TIKTOK_USER_INFO_FIELDS_FULL, "full"),
        (TIKTOK_USER_INFO_FIELDS_BASIC, "basic"),
    ):
        for method in ("get", "post"):
            try:
                if method == "get":
                    resp = await client.get(
                        TIKTOK_USER_INFO_URL,
                        params={"fields": fields},
                        headers=headers,
                    )
                else:
                    resp = await client.post(
                        TIKTOK_USER_INFO_URL,
                        headers={**headers, "Content-Type": "application/json"},
                        json={"fields": fields},
                    )
                try:
                    body = resp.json() if resp.content else {}
                except Exception:
                    body = {}
                env_err = tiktok_envelope_error(body)
                if resp.status_code != 200 or env_err:
                    last_log = (
                        f"TikTok user/info {method.upper()} ({label}) HTTP {resp.status_code} "
                        f"envelope={env_err!r}"
                    )
                    logger.warning("%s body_snip=%r", last_log, str(body)[:400])
                    continue
                user_obj = tiktok_extract_user_from_info_body(body)
                ident = tiktok_identity_from_user_object(user_obj)
                if ident.get("account_name") or ident.get("account_avatar") or ident.get("account_id"):
                    logger.info(
                        "TikTok user/info ok via %s %s (has_name=%s has_avatar=%s)",
                        method,
                        label,
                        bool(ident.get("account_name")),
                        bool(ident.get("account_avatar")),
                    )
                    return ident
                last_log = f"TikTok user/info {method} ({label}) empty user payload"
                logger.warning("%s keys=%r", last_log, list(body.keys())[:20])
            except Exception as e:
                last_log = f"TikTok user/info {method} ({label}) exception: {type(e).__name__}: {e}"
                logger.warning(last_log)

    if last_log:
        logger.warning("TikTok user profile: all attempts failed; last=%s", last_log)
    return tiktok_identity_from_user_object({})
