"""
Resolve OAuth tokens for per-upload analytics sync.

platform_results rows store:
  - token_row_id: platform_tokens.id (UUID) — canonical key for token_map_by_id
  - account_id: platform-native id (e.g. YouTube channel UC…) — must NOT be used alone
    to index token_map_by_id (that map is keyed by UUID). Use token_map_by_plat_account
    for (platform, account_id) fallback when token_row_id is missing (legacy rows).
"""
from __future__ import annotations

from typing import Any, Dict, Tuple, List


def _as_token_list(v: Any) -> List[dict]:
    """Normalize platform fallback storage into a list of decrypted token dicts."""
    if not v:
        return []
    if isinstance(v, list):
        return [t for t in v if isinstance(t, dict)]
    if isinstance(v, dict):
        return [v]
    return []


def resolve_token_for_platform_result(
    pr: Dict[str, Any],
    token_map_by_id: Dict[str, dict],
    token_map_by_plat_account: Dict[Tuple[str, str], dict],
    token_map_by_platform: Dict[str, dict],
) -> dict:
    """Pick the correct decrypted token dict for one platform_results entry."""
    # Backwards-compatible wrapper around the new candidate-based resolver.
    candidates = resolve_token_candidates_for_platform_result(
        pr=pr,
        token_map_by_id=token_map_by_id,
        token_map_by_plat_account=token_map_by_plat_account,
        token_map_by_platform=token_map_by_platform,
    )
    return candidates[0] if candidates else {}


def resolve_token_candidates_for_platform_result(
    pr: Dict[str, Any],
    token_map_by_id: Dict[str, dict],
    token_map_by_plat_account: Dict[Tuple[str, str], dict],
    token_map_by_platform: Dict[str, Any],
) -> List[dict]:
    """
    Return decrypted token candidates for one `platform_results` entry.

    If the entry includes `token_row_id`/`token_id` or `account_id`, return the single matching token.
    Otherwise, return *all* active tokens for that platform (caller may try them in order).
    """
    plat = str(pr.get("platform") or "").lower()

    tid = pr.get("token_row_id") or pr.get("token_id")
    if tid:
        tok = token_map_by_id.get(str(tid))
        if tok:
            return [tok]

    aid = pr.get("account_id")
    if aid is not None and str(aid).strip() != "":
        a = str(aid).strip()
        tok = token_map_by_plat_account.get((plat, a))
        if tok:
            return [tok]
        tok = token_map_by_id.get(a)
        if tok:
            return [tok]

    # Legacy/ambiguous rows: try all active tokens for that platform.
    return _as_token_list(token_map_by_platform.get(plat))


def build_plat_account_token_map(token_rows, decrypt_fn) -> Dict[Tuple[str, str], dict]:
    """
    (platform, account_id) -> decrypted token. account_id is platform_tokens.account_id
    (YouTube channel id, TikTok open id, etc.).
    """
    out: Dict[Tuple[str, str], dict] = {}
    for tr in token_rows:
        try:
            blob = tr["token_blob"]
            if not blob:
                continue
            dec = decrypt_fn(blob)
            if not dec:
                continue
            plat = tr["platform"]
            aid = tr.get("account_id")
            if aid is not None and str(aid).strip() != "":
                out[(str(plat), str(aid).strip())] = dec
        except Exception:
            continue
    return out
