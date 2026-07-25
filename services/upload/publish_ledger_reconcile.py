"""Reconcile stuck uploads from Step A ``publish_attempts`` ledger.

When platforms already accepted a publish but the worker died before
``mark_processing_completed``, ``uploads.status`` can stay ``processing`` while
verify keeps polling accepted attempts. This module terminalizes from the ledger
**without re-publishing**.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("uploadm8-worker")


def attempt_row_to_platform_result(row: Dict[str, Any]) -> Dict[str, Any]:
    """Map a ``publish_attempts`` row to a ``platform_results`` dict."""
    status = str(row.get("status") or "").strip().lower()
    success = status == "accepted"
    verify = str(row.get("verify_status") or "").strip() or "pending"
    tid = row.get("token_row_id")
    out = {
        "platform": str(row.get("platform") or "").strip().lower(),
        "success": success,
        "platform_video_id": row.get("platform_post_id"),
        "platform_url": row.get("platform_url"),
        "publish_id": row.get("publish_id"),
        "attempt_id": str(row.get("id") or "") or None,
        "http_status": row.get("http_status"),
        "error_code": None if success else (row.get("error_code") or "PUBLISH_FAILED"),
        "error_message": None if success else (row.get("error_message") or ""),
        "verify_status": verify,
        "views": 0,
        "likes": 0,
    }
    if tid:
        out["token_row_id"] = str(tid)
    return out


def terminal_state_from_attempt_rows(rows: Sequence[Dict[str, Any]]) -> Optional[str]:
    """Return ``succeeded`` / ``partial`` when any attempt is accepted; else None.

    Does not return ``failed`` — absence of accepted rows is not proof of
    publish failure (ledger may be incomplete). Caller must not re-publish when
    this returns a terminal success state.
    """
    if not rows:
        return None
    any_ok = any(str(r.get("status") or "").strip().lower() == "accepted" for r in rows)
    if not any_ok:
        return None
    any_fail = any(str(r.get("status") or "").strip().lower() == "failed" for r in rows)
    if any_fail:
        return "partial"
    return "succeeded"


def platform_results_from_attempts(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build platform_results list from ledger rows (accepted + failed).

    Dedupes by platform + post identity so re-publish storms do not inflate the
    queue UI to dozens of identical chips.
    """
    out: List[Dict[str, Any]] = []
    for row in rows or []:
        st = str(row.get("status") or "").strip().lower()
        if st not in ("accepted", "failed"):
            continue
        pr = attempt_row_to_platform_result(dict(row))
        if pr.get("platform"):
            out.append(pr)
    try:
        from services.uploads_api import dedupe_platform_result_entries

        return dedupe_platform_result_entries(out)
    except Exception:
        return out


def has_fresh_pending_attempts(
    rows: Sequence[Dict[str, Any]],
    *,
    max_age_seconds: int = 1800,
    now: Optional[datetime] = None,
) -> bool:
    """True when any pending ledger row is younger than ``max_age_seconds``.

    A fresh pending slot means Step A is likely in-flight on another publisher.
    Dispatching a second publish while it runs risks a real double-post the
    moment the first slot gets abandoned. Pending rows without timestamps
    count as fresh (fail-closed).
    """
    ref = now or datetime.now(timezone.utc)
    for row in rows or []:
        if str(row.get("status") or "").strip().lower() != "pending":
            continue
        ts = row.get("updated_at") or row.get("created_at")
        if ts is None:
            return True
        if getattr(ts, "tzinfo", None) is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if (ref - ts).total_seconds() < max_age_seconds:
            return True
    return False


def has_accepted_publish_attempts(rows: Sequence[Dict[str, Any]]) -> bool:
    return any(str(r.get("status") or "").strip().lower() == "accepted" for r in rows or [])


async def recover_deferred_publish_false_failure(
    db_pool,
    upload_id: str,
    *,
    user_id: Optional[str] = None,
    ctx: Any = None,
) -> Dict[str, Any]:
    """Recover post-publish errors when the ledger already shows accepts.

    Covers the false-failure class where TikTok (etc.) accepted but the worker
    blew up on ``platform_results`` jsonb binds (``expected str, got list``) or
    a stale consumer left ``ctx.platform_results`` empty. Hydrates ``ctx`` from
    the ledger when needed, then terminalizes via
    :func:`reconcile_stuck_processing_from_ledger`.

    Returns ``recovered``, ``state``, ``reason``, ``hydrated``.
    """
    out: Dict[str, Any] = {
        "recovered": False,
        "state": None,
        "reason": "not_attempted",
        "hydrated": False,
        "upload_id": str(upload_id or ""),
    }
    if not db_pool or not upload_id:
        out["reason"] = "missing_pool_or_id"
        return out

    attempts = await load_publish_attempts_for_upload(db_pool, upload_id)
    if not has_accepted_publish_attempts(attempts):
        out["reason"] = "no_accepted_attempts"
        return out

    ctx_ok = False
    try:
        ctx_ok = bool(
            ctx is not None
            and (
                getattr(ctx, "is_success", lambda: False)()
                or getattr(ctx, "is_partial_success", lambda: False)()
            )
        )
    except Exception:
        ctx_ok = False

    if ctx is not None and not ctx_ok:
        try:
            from services.deferred_publish_schedule import hydrate_platform_results_into_ctx

            hydrate_platform_results_into_ctx(
                ctx, platform_results_from_attempts(attempts)
            )
            out["hydrated"] = True
            try:
                ctx_ok = bool(ctx.is_success() or ctx.is_partial_success())
            except Exception:
                ctx_ok = False
        except Exception as hyd_err:
            logger.warning(
                "[%s] recover_deferred_publish_false_failure hydrate failed: %s",
                upload_id,
                hyd_err,
            )

    rec = await reconcile_stuck_processing_from_ledger(
        db_pool,
        upload_id,
        user_id=user_id,
        force=True,
    )
    state = str(rec.get("state") or "").strip().lower()
    if rec.get("ok") and state in ("succeeded", "partial", "completed"):
        out["recovered"] = True
        out["state"] = "partial" if state == "partial" else "succeeded"
        out["reason"] = str(rec.get("reason") or "reconciled_from_ledger")
        if ctx is not None and out["state"]:
            try:
                ctx.state = out["state"]
            except Exception:
                pass
        return out

    # Ledger write raced / already terminal elsewhere — still treat local
    # successes as recovered so callers skip mark_processing_failed.
    if ctx_ok:
        try:
            out["state"] = (
                "partial" if ctx.is_partial_success() else "succeeded"
            )
        except Exception:
            out["state"] = "succeeded"
        out["recovered"] = True
        out["reason"] = str(rec.get("reason") or "ctx_success_ledger_miss")
        return out

    out["reason"] = str(rec.get("reason") or "ledger_reconcile_failed")
    return out


def filter_pending_targets_against_accepted_ledger(
    pending_targets: Sequence[Tuple[str, Optional[str]]],
    attempt_rows: Sequence[Dict[str, Any]],
    *,
    existing_platform_results: Optional[Sequence[Any]] = None,
) -> Tuple[List[Tuple[str, Optional[str]]], List[Dict[str, Any]]]:
    """Drop targets already covered by accepted ``publish_attempts``.

    Prefer token-scoped match when both the target and ledger row have
    ``token_row_id``. Legacy rows (null token) still consume platform buckets
    in order so pre-migration ledgers keep working.

    Returns ``(still_pending, synthetic_results)`` where synthetic_results should
    be appended onto ``ctx.platform_results`` so finalize/UI see the prior posts.
    """
    accepted = [
        dict(r)
        for r in (attempt_rows or [])
        if str(r.get("status") or "").strip().lower() == "accepted"
        and str(r.get("platform") or "").strip()
    ]
    if not pending_targets or not accepted:
        return list(pending_targets), []

    accepted_by_token: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    accepted_legacy_by_plat: Dict[str, List[Dict[str, Any]]] = {}
    for row in accepted:
        plat = str(row.get("platform") or "").strip().lower()
        tid = str(row.get("token_row_id") or "").strip()
        if tid:
            accepted_by_token.setdefault((plat, tid), []).append(row)
        else:
            accepted_legacy_by_plat.setdefault(plat, []).append(row)

    # Consume ledger slots already reflected in platform_results (token first).
    for r in existing_platform_results or []:
        if isinstance(r, dict):
            ok = bool(r.get("success"))
            plat = str(r.get("platform") or "").strip().lower()
            tid = str(
                r.get("token_row_id") or r.get("account_id") or r.get("token_id") or ""
            ).strip()
        else:
            ok = bool(getattr(r, "success", False))
            plat = str(getattr(r, "platform", "") or "").strip().lower()
            tid = str(
                getattr(r, "token_row_id", None)
                or getattr(r, "account_id", None)
                or getattr(r, "token_id", None)
                or ""
            ).strip()
        if not ok or not plat:
            continue
        if tid:
            bucket = accepted_by_token.get((plat, tid)) or []
            if bucket:
                bucket.pop(0)
                continue
        legacy = accepted_legacy_by_plat.get(plat) or []
        if legacy:
            legacy.pop(0)

    still_pending: List[Tuple[str, Optional[str]]] = []
    synthetic: List[Dict[str, Any]] = []

    for platform, token_id in pending_targets:
        plat = str(platform or "").strip().lower()
        tid = str(token_id or "").strip()
        row = None
        if tid:
            bucket = accepted_by_token.get((plat, tid)) or []
            if bucket:
                row = bucket.pop(0)
        if row is None:
            legacy = accepted_legacy_by_plat.get(plat) or []
            if legacy:
                row = legacy.pop(0)
        if row is None:
            still_pending.append((platform, token_id))
            continue
        pr = attempt_row_to_platform_result(row)
        if tid:
            pr["token_row_id"] = tid
        synthetic.append(pr)
        logger.info(
            "ledger idempotent skip platform=%s token=%s attempt=%s",
            plat,
            (tid[:8] + "…") if tid else "-",
            pr.get("attempt_id"),
        )

    return still_pending, synthetic


async def hydrate_ctx_from_accepted_ledger(
    ctx: Any,
    db_pool,
    pending_targets: Sequence[Tuple[str, Optional[str]]],
) -> List[Tuple[str, Optional[str]]]:
    """Load accepted ledger into ctx and return targets that still need API calls."""
    from stages.context import PlatformResult

    upload_id = str(getattr(ctx, "upload_id", "") or "")
    attempts = await load_publish_attempts_for_upload(db_pool, upload_id)
    still, synthetic = filter_pending_targets_against_accepted_ledger(
        pending_targets,
        attempts,
        existing_platform_results=getattr(ctx, "platform_results", None) or [],
    )
    for pr in synthetic:
        ctx.platform_results.append(
            PlatformResult(
                platform=str(pr.get("platform") or ""),
                success=bool(pr.get("success")),
                platform_video_id=pr.get("platform_video_id"),
                platform_url=pr.get("platform_url"),
                publish_id=pr.get("publish_id"),
                token_row_id=pr.get("token_row_id"),
                attempt_id=pr.get("attempt_id"),
                http_status=pr.get("http_status"),
                error_code=pr.get("error_code"),
                error_message=pr.get("error_message"),
                verify_status=str(pr.get("verify_status") or "pending"),
            )
        )
    return still


class LedgerLoadError(RuntimeError):
    """Raised when publish_attempts cannot be loaded (fail-closed callers)."""


async def load_publish_attempts_for_upload(db_pool, upload_id: str) -> List[Dict[str, Any]]:
    """Load ledger rows for an upload (accepted/failed/pending).

    Missing table / missing columns → empty list (pre-migration).
    Other DB errors → ``LedgerLoadError`` so publish fan-out can fail closed.
    """
    if not db_pool or not upload_id:
        return []
    try:
        async with db_pool.acquire() as conn:
            try:
                rows = await conn.fetch(
                    """
                    SELECT id, upload_id, user_id, platform, status,
                           token_row_id,
                           platform_post_id, platform_url, publish_id,
                           http_status, error_code, error_message, verify_status,
                           created_at, updated_at
                      FROM publish_attempts
                     WHERE upload_id = $1::uuid
                     ORDER BY created_at ASC
                    """,
                    upload_id,
                )
            except Exception as table_err:
                err_name = type(table_err).__name__
                msg = str(table_err).lower()
                if (
                    "undefinedtable" in err_name.lower()
                    or "undefinedcolumn" in err_name.lower()
                    or "does not exist" in msg
                    or "undefined_table" in msg
                    or "undefined_column" in msg
                ):
                    logger.debug(
                        "load_publish_attempts_for_upload skipped upload=%s: %s",
                        upload_id,
                        table_err,
                    )
                    return []
                raise LedgerLoadError(
                    f"publish_attempts load failed for {upload_id}: {table_err}"
                ) from table_err
        return [dict(r) for r in rows]
    except LedgerLoadError:
        raise
    except Exception as e:
        raise LedgerLoadError(
            f"publish_attempts load failed for {upload_id}: {e}"
        ) from e


async def count_accepted_publish_attempts(db_pool, upload_id: str) -> int:
    rows = await load_publish_attempts_for_upload(db_pool, upload_id)
    return sum(1 for r in rows if str(r.get("status") or "").lower() == "accepted")


def expected_publish_slots(
    upload_row: Optional[Dict[str, Any]],
    *,
    live_token_ids: Optional[Sequence[str]] = None,
) -> int:
    """How many platform/account posts this upload intended.

    When ``live_token_ids`` is provided, revoked/missing ``target_accounts`` are
    excluded so ledger reconcile can terminalize after all reachable accepts.
    """
    if not upload_row:
        return 0
    targets = upload_row.get("target_accounts")
    if isinstance(targets, str):
        try:
            import json as _json

            targets = _json.loads(targets) if targets.strip() else []
        except Exception:
            targets = []
    if isinstance(targets, list) and targets:
        ids = [str(t).strip() for t in targets if str(t).strip()]
        if live_token_ids is not None:
            live = {str(x).strip() for x in live_token_ids if str(x).strip()}
            ids = [t for t in ids if t in live]
            if ids:
                return len(ids)
            # All listed tokens dead — fall through to platforms length.
        else:
            return len(ids)
    plats = upload_row.get("platforms")
    if isinstance(plats, str):
        try:
            import json as _json

            plats = _json.loads(plats) if plats.strip() else []
        except Exception:
            plats = []
    if isinstance(plats, list) and plats:
        return len([p for p in plats if str(p).strip()])
    return 0


async def resolve_live_target_token_ids(db_pool, upload_row: Optional[Dict[str, Any]]) -> Optional[List[str]]:
    """Return non-revoked platform_tokens ids from upload.target_accounts, or None."""
    if not db_pool or not upload_row:
        return None
    targets = upload_row.get("target_accounts")
    if isinstance(targets, str):
        try:
            import json as _json

            targets = _json.loads(targets) if targets.strip() else []
        except Exception:
            targets = []
    ids = [str(t).strip() for t in (targets or []) if str(t).strip()]
    if not ids:
        return None
    try:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id::text AS id
                  FROM platform_tokens
                 WHERE id = ANY($1::uuid[])
                   AND revoked_at IS NULL
                """,
                ids,
            )
        return [str(r["id"]) for r in rows]
    except Exception as e:
        logger.debug("resolve_live_target_token_ids failed: %s", e)
        return None


def ledger_covers_expected_slots(
    attempt_rows: Sequence[Dict[str, Any]],
    upload_row: Optional[Dict[str, Any]],
    *,
    live_token_ids: Optional[Sequence[str]] = None,
) -> bool:
    """True when accepted attempts are enough to terminalize (no remaining targets)."""
    accepted_n = sum(
        1 for r in attempt_rows or [] if str(r.get("status") or "").lower() == "accepted"
    )
    if accepted_n <= 0:
        return False
    expected = expected_publish_slots(upload_row, live_token_ids=live_token_ids)
    if expected <= 0:
        # Unknown target count — only safe when every non-pending row is accepted
        # and at least one accepted exists (legacy heal / single-platform).
        non_pending = [
            r
            for r in (attempt_rows or [])
            if str(r.get("status") or "").lower() in ("accepted", "failed")
        ]
        return bool(non_pending) and all(
            str(r.get("status") or "").lower() == "accepted" for r in non_pending
        )
    return accepted_n >= expected


async def reconcile_stuck_processing_from_ledger(
    db_pool,
    upload_id: str,
    *,
    user_id: Optional[str] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Terminalize a stuck upload from accepted publish_attempts (no re-publish).

    Safe when ``status`` is ``processing`` (or still ``ready_to_publish`` with
    empty platform_results after a false reclaim). Returns a result dict:
    ``ok``, ``state``, ``reason``, ``accepted_count``, ``platform_results_count``.

    Refuses to mark ``succeeded`` when accepted count is below expected publish
    slots (unless ``force=True``) so mid-fan-out crashes can finish remaining
    platforms via ledger-aware ``publish_stage`` skip.
    """
    result: Dict[str, Any] = {
        "ok": False,
        "state": None,
        "reason": "not_attempted",
        "accepted_count": 0,
        "platform_results_count": 0,
        "expected_slots": 0,
        "upload_id": str(upload_id or ""),
    }
    if not db_pool or not upload_id:
        result["reason"] = "missing_pool_or_id"
        return result

    attempts = await load_publish_attempts_for_upload(db_pool, upload_id)
    accepted_n = sum(1 for r in attempts if str(r.get("status") or "").lower() == "accepted")
    result["accepted_count"] = accepted_n
    state = terminal_state_from_attempt_rows(attempts)
    if not state:
        result["reason"] = "no_accepted_attempts"
        return result

    platform_results = platform_results_from_attempts(attempts)
    result["platform_results_count"] = len(platform_results)
    if not platform_results:
        result["reason"] = "empty_platform_results"
        return result

    from stages.asyncpg_json_codecs import json_param_encoder

    pr_bind = json_param_encoder(platform_results)
    finished = datetime.now(timezone.utc)

    try:
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT status, platform_results, platforms, target_accounts
                  FROM uploads
                 WHERE id = $1::uuid
                """,
                upload_id,
            )
            if not row:
                result["reason"] = "upload_not_found"
                return result
            upload_snap = dict(row)
            live_ids = await resolve_live_target_token_ids(db_pool, upload_snap)
            expected = expected_publish_slots(upload_snap, live_token_ids=live_ids)
            result["expected_slots"] = expected
            cur_status = str(row.get("status") or "").strip().lower()
            if cur_status in ("succeeded", "partial", "completed", "failed", "cancelled"):
                result["ok"] = True
                result["state"] = cur_status
                result["reason"] = "already_terminal"
                return result
            if cur_status not in ("processing", "ready_to_publish", "queued"):
                result["reason"] = f"unsupported_status:{cur_status}"
                return result

            if not force and not ledger_covers_expected_slots(
                attempts, upload_snap, live_token_ids=live_ids
            ):
                result["reason"] = "accepted_below_expected"
                return result

            tag = await conn.execute(
                """
                UPDATE uploads
                   SET status = $2,
                       processing_finished_at = COALESCE(processing_finished_at, $3),
                       completed_at = CASE
                           WHEN $2 IN ('succeeded', 'partial') THEN COALESCE(completed_at, $3)
                           ELSE completed_at
                       END,
                       platform_results = $4::jsonb,
                       processing_stage = 'done',
                       processing_progress = 100,
                       error_code = NULL,
                       error_detail = NULL,
                       output_artifacts = (
                           COALESCE(output_artifacts, '{}'::jsonb) - 'failure_phase'
                       ),
                       updated_at = NOW()
                 WHERE id = $1::uuid
                   AND status IN ('processing', 'ready_to_publish', 'queued')
                """,
                upload_id,
                state,
                finished,
                pr_bind,
            )
            if str(tag or "") == "UPDATE 0":
                result["reason"] = "race_lost"
                return result
    except Exception as e:
        logger.exception(
            "[%s] reconcile_stuck_processing_from_ledger failed: %s", upload_id, e
        )
        result["reason"] = f"db_error:{e}"
        return result

    result["ok"] = True
    result["state"] = state
    result["reason"] = "reconciled_from_ledger"
    logger.warning(
        "[%s] ledger reconcile → %s (accepted=%s expected=%s results=%s user=%s)",
        upload_id,
        state,
        accepted_n,
        result.get("expected_slots"),
        len(platform_results),
        user_id or "",
    )
    return result
