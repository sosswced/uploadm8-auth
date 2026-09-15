"""
Resolve OAuth tokens for per-upload analytics sync.

platform_results rows store:
  - token_row_id: platform_tokens.id (UUID) — canonical key for token_map_by_id
  - account_id: platform-native id (e.g. YouTube channel UC…) — must NOT be used alone
    to index token_map_by_id (that map is keyed by UUID). Use token_map_by_plat_account
    for (platform, account_id) fallback when token_row_id is missing (legacy rows).
"""
from __future__ import annotations

from typing import Any, Dict, Tuple, List, Optional


def _as_token_list(v: Any) -> List[dict]:
    """Normalize platform fallback storage into a list of decrypted token dicts."""
    if not v:
        return []
    if isinstance(v, list):
        return [t for t in v if isinstance(t, dict)]
    if isinstance(v, dict):
        return [v]
    return []


def build_plat_account_token_row_map(token_rows: Any, decrypt_fn: Any) -> Dict[Tuple[str, str], Tuple[str, dict]]:
    """
    (platform, account_id) -> (platform_tokens.id, decrypted_token).
    Used to attach the correct token_row_id when account_id is present on platform_results.
    """
    out: Dict[Tuple[str, str], Tuple[str, dict]] = {}
    for tr in token_rows or []:
        try:
            blob = tr["token_blob"]
            if not blob:
                continue
            dec = decrypt_fn(blob)
            if not dec:
                continue
            plat = str(tr.get("platform") or "").lower()
            aid = tr.get("account_id")
            if aid is not None and str(aid).strip() != "":
                out[(plat, str(aid).strip())] = (str(tr["id"]), dec)
        except Exception:
            continue
    return out


def build_platform_token_row_list(token_rows: Any, decrypt_fn: Any) -> Dict[str, List[Tuple[str, dict]]]:
    """
    platform -> [(token_row_id, decrypted_token), ...] for legacy multi-token fallback.
    """
    out: Dict[str, List[Tuple[str, dict]]] = {}
    for tr in token_rows or []:
        try:
            blob = tr["token_blob"]
            if not blob:
                continue
            dec = decrypt_fn(blob)
            if not dec:
                continue
            plat = str(tr.get("platform") or "").lower()
            out.setdefault(plat, []).append((str(tr["id"]), dec))
        except Exception:
            continue
    return out


def _dedupe_token_pairs(pairs: List[Tuple[str, dict]]) -> List[Tuple[str, dict]]:
    """Prefer first occurrence; identity by token_row_id when set, else id(token dict)."""
    out: List[Tuple[str, dict]] = []
    seen: set[str] = set()
    for tid, tok in pairs:
        if not tok:
            continue
        key = str(tid).strip() if tid else f"obj:{id(tok)}"
        if key in seen:
            continue
        seen.add(key)
        out.append((str(tid) if tid else "", tok))
    return out


def resolve_token_candidates_with_row_ids(
    pr: Dict[str, Any],
    token_map_by_id: Dict[str, dict],
    token_map_by_plat_account: Dict[Tuple[str, str], dict],
    plat_account_row_map: Dict[Tuple[str, str], Tuple[str, dict]],
    platform_token_rows: Dict[str, List[Tuple[str, dict]]],
) -> List[Tuple[str, dict]]:
    """
    Return (platform_tokens.id, decrypted_token) pairs for one platform_results entry.

    Order: explicit token_row_id (if still live), then (platform, account_id), then all
    live tokens for the platform. Never exclusive-bind to a live-but-wrong row after
    reconnect / account_id rewrite — callers try until a platform API returns metrics.
    """
    plat = str(pr.get("platform") or "").lower()
    ordered: List[Tuple[str, dict]] = []

    tid = pr.get("token_row_id") or pr.get("token_id")
    if tid:
        tok = token_map_by_id.get(str(tid))
        if tok:
            ordered.append((str(tid), tok))

    aid = pr.get("account_id")
    if aid is not None and str(aid).strip() != "":
        a = str(aid).strip()
        pair = plat_account_row_map.get((plat, a))
        if pair:
            ordered.append(pair)
        else:
            tok = token_map_by_plat_account.get((plat, a))
            if tok:
                ordered.append(("", tok))
            else:
                tok = token_map_by_id.get(a)
                if tok:
                    ordered.append(("", tok))

    for pair in platform_token_rows.get(plat) or []:
        ordered.append(pair)

    return _dedupe_token_pairs(ordered)


def resolve_token_candidates_for_platform_result(
    pr: Dict[str, Any],
    token_map_by_id: Dict[str, dict],
    token_map_by_plat_account: Dict[Tuple[str, str], dict],
    token_map_by_platform: Dict[str, Any],
    *,
    plat_account_row_map: Optional[Dict[Tuple[str, str], Tuple[str, dict]]] = None,
    platform_token_rows: Optional[Dict[str, List[Tuple[str, dict]]]] = None,
) -> List[dict]:
    """
    Return decrypted token candidates for one `platform_results` entry.

    Prefer token_row_id / account_id matches first, then other active tokens for that
    platform so a stale exclusive bind after reconnect cannot block sync.
    """
    if plat_account_row_map is not None and platform_token_rows is not None:
        pairs = resolve_token_candidates_with_row_ids(
            pr, token_map_by_id, token_map_by_plat_account, plat_account_row_map, platform_token_rows
        )
        return [p[1] for p in pairs if p[1]]

    plat = str(pr.get("platform") or "").lower()
    ordered: List[dict] = []

    tid = pr.get("token_row_id") or pr.get("token_id")
    if tid:
        tok = token_map_by_id.get(str(tid))
        if tok:
            ordered.append(tok)

    aid = pr.get("account_id")
    if aid is not None and str(aid).strip() != "":
        a = str(aid).strip()
        tok = token_map_by_plat_account.get((plat, a))
        if tok:
            ordered.append(tok)
        else:
            tok = token_map_by_id.get(a)
            if tok:
                ordered.append(tok)

    for tok in _as_token_list(token_map_by_platform.get(plat)):
        ordered.append(tok)

    # Deduplicate by object identity (legacy path stores one dict per platform key).
    out: List[dict] = []
    seen: set[int] = set()
    for tok in ordered:
        if not tok:
            continue
        i = id(tok)
        if i in seen:
            continue
        seen.add(i)
        out.append(tok)
    return out


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


def token_row_id_from_pair(token_row_id: str, pr: Dict[str, Any]) -> str:
    """Prefer non-empty token_row_id from resolution; fall back to entry."""
    s = str(token_row_id or "").strip()
    if s:
        return s
    return str(pr.get("token_row_id") or pr.get("token_id") or "").strip()
