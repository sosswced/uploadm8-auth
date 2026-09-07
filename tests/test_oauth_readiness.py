"""OAuth durability: refresh-grant expiry, failure grace, and scheduled-work risk.

Long Smart Schedule windows mean a connection must be trusted months or years
ahead, so these tests pin the two behaviours that make that safe: a provider
outage must not retire a connection, and a post scheduled past refresh-grant
death must be reported as at risk.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from core.platform_token_expiry import (
    is_permanent_oauth_error,
    parse_refresh_expires_at,
    refresh_expiry_state,
    refresh_token_alive_at,
)
from services.oauth_readiness import (
    RISK_GRANT_EXPIRES_FIRST,
    RISK_NEEDS_RECONNECT,
    RISK_NO_CONNECTION,
    STATE_DEGRADED,
    STATE_EXPIRING,
    STATE_NEEDS_RECONNECT,
    STATE_OK,
    STATE_STALE,
    assess_scheduled_risk,
    evaluate_connection,
    summarize_connections,
)
from services.platform_oauth_refresh import (
    OAUTH_FAIL_GRACE_ATTEMPTS,
    OAUTH_RETRY_BACKOFF_MAX_SEC,
    _retry_backoff_seconds,
    _should_declare_dead,
)

NOW = datetime(2026, 9, 7, 12, 0, tzinfo=timezone.utc)


# ── Provider error classification ────────────────────────────────────────────


def test_permanent_errors_are_recognized():
    for err in (
        "invalid_grant",
        "Token has been revoked",
        "OAuth2 parameter error",
        "refresh token expired",
        "unauthorized_client",
    ):
        assert is_permanent_oauth_error(err) is True, err


def test_transient_errors_are_not_permanent():
    """A blip must never read as a revoked grant."""
    for err in (
        "timed out",
        "Connection reset by peer",
        "500 Internal Server Error",
        "503 Service Unavailable",
        "rate limit exceeded",
        "temporary failure in name resolution",
        TimeoutError("read timeout"),
        None,
        "",
    ):
        assert is_permanent_oauth_error(err) is False, err


# ── Refresh-grant expiry ─────────────────────────────────────────────────────


def test_absolute_refresh_expiry_wins():
    blob = {"refresh_token": "rt", "refresh_expires_at": "2027-01-01T00:00:00+00:00"}
    got = parse_refresh_expires_at(blob, platform="tiktok", now=NOW)
    assert got == datetime(2027, 1, 1, tzinfo=timezone.utc)


def test_refresh_expires_in_is_relative_to_obtained_at():
    blob = {
        "refresh_token": "rt",
        "refresh_expires_in": 86400,
        "access_obtained_at": NOW.isoformat(),
    }
    assert parse_refresh_expires_at(blob, platform="tiktok", now=NOW) == NOW + timedelta(days=1)


def test_platform_default_lifetime_used_when_provider_silent():
    """TikTok never sends refresh_expires_in on refresh; we still know the ceiling."""
    blob = {"refresh_token": "rt", "access_obtained_at": NOW.isoformat()}
    got = parse_refresh_expires_at(blob, platform="tiktok", now=NOW)
    assert got == NOW + timedelta(days=365)


def test_no_refresh_token_has_no_grant_expiry():
    assert parse_refresh_expires_at({}, platform="tiktok", now=NOW) is None


def test_refresh_non_expiring_flag_respected():
    blob = {"refresh_token": "rt", "refresh_non_expiring": True, "access_obtained_at": NOW.isoformat()}
    assert parse_refresh_expires_at(blob, platform="tiktok", now=NOW) is None


def test_refresh_expiry_state_transitions():
    obtained = {"refresh_token": "rt", "access_obtained_at": NOW.isoformat()}
    assert refresh_expiry_state(obtained, platform="tiktok", now=NOW) == "ok"
    # 5 days before a 365d grant dies.
    late = NOW + timedelta(days=360)
    assert refresh_expiry_state(obtained, platform="tiktok", now=late) == "expiring_soon"
    assert refresh_expiry_state(obtained, platform="tiktok", now=NOW + timedelta(days=400)) == "expired"


def test_refresh_token_alive_at_gates_long_horizon_publishing():
    blob = {"refresh_token": "rt", "access_obtained_at": NOW.isoformat()}
    assert refresh_token_alive_at(blob, NOW + timedelta(days=30), platform="tiktok", now=NOW) is True
    assert refresh_token_alive_at(blob, NOW + timedelta(days=400), platform="tiktok", now=NOW) is False
    # Unknown ceiling must not block scheduling.
    assert refresh_token_alive_at({}, NOW + timedelta(days=4000), platform="tiktok", now=NOW) is True


# ── Failure grace: an outage is not a dead connection ────────────────────────


def _live_blob():
    return {
        "refresh_token": "rt",
        "access_obtained_at": NOW.isoformat(),
        "expires_at": (NOW + timedelta(hours=1)).isoformat(),
    }


def test_permanent_error_retires_connection_immediately():
    assert _should_declare_dead(
        platform="tiktok", blob=_live_blob(), fail_count=1,
        error="invalid_grant", now=NOW,
    ) is True


def test_single_transient_failure_never_retires_connection():
    assert _should_declare_dead(
        platform="tiktok", blob=_live_blob(), fail_count=1,
        error="503 Service Unavailable", now=NOW,
    ) is False


def test_repeated_transient_failures_survive_while_access_token_valid():
    """A long provider outage must not mass-flag healthy accounts."""
    assert _should_declare_dead(
        platform="tiktok", blob=_live_blob(),
        fail_count=OAUTH_FAIL_GRACE_ATTEMPTS + 10,
        error="timed out", now=NOW,
    ) is False


def test_repeated_failures_retire_once_access_token_has_expired():
    blob = dict(_live_blob(), expires_at=(NOW - timedelta(minutes=5)).isoformat())
    assert _should_declare_dead(
        platform="tiktok", blob=blob,
        fail_count=OAUTH_FAIL_GRACE_ATTEMPTS,
        error="timed out", now=NOW,
    ) is True
    # ...but not before the grace threshold is reached.
    assert _should_declare_dead(
        platform="tiktok", blob=blob,
        fail_count=OAUTH_FAIL_GRACE_ATTEMPTS - 1,
        error="timed out", now=NOW,
    ) is False


def test_expired_refresh_grant_retires_connection():
    blob = {
        "refresh_token": "rt",
        "access_obtained_at": (NOW - timedelta(days=400)).isoformat(),
    }
    assert _should_declare_dead(
        platform="tiktok", blob=blob, fail_count=1, error="timed out", now=NOW,
    ) is True


def test_retry_backoff_grows_and_is_capped():
    seq = [_retry_backoff_seconds(n) for n in range(1, 12)]
    assert seq == sorted(seq)
    assert seq[0] < seq[-1]
    assert max(seq) <= OAUTH_RETRY_BACKOFF_MAX_SEC
    assert _retry_backoff_seconds(0) == _retry_backoff_seconds(1)


class _FakeConn:
    """Records the UPDATE statements the keepalive writes."""

    def __init__(self):
        self.calls = []

    async def execute(self, sql, *args):
        self.calls.append((sql, args))
        return "UPDATE 1"


class _FakePool:
    def __init__(self):
        self.conn = _FakeConn()

    def acquire(self):
        conn = self.conn

        class _Ctx:
            async def __aenter__(self):
                return conn

            async def __aexit__(self, *exc):
                return False

        return _Ctx()


def _written_sql(pool):
    return " ".join(sql for sql, _ in pool.conn.calls)


def test_not_due_scan_refreshes_mirrors_without_touching_health():
    """A revoked account can still hold a valid access token — that is not health."""
    import asyncio

    from services.platform_oauth_refresh import record_keepalive_scan

    pool = _FakePool()
    asyncio.run(
        record_keepalive_scan(
            pool, "11111111-1111-1111-1111-111111111111",
            platform="tiktok", blob=_live_blob(),
        )
    )
    sql = _written_sql(pool)
    assert "access_expires_at" in sql
    assert "refresh_expires_at" in sql
    assert "oauth_last_verified_at" in sql
    # Must not declare a possibly-dead connection healthy, or reset its tally.
    assert "oauth_health" not in sql
    assert "oauth_fail_count" not in sql


def test_transient_failure_writes_backoff_but_leaves_health_alone():
    import asyncio

    from services.platform_oauth_refresh import record_keepalive_failure

    pool = _FakePool()
    dead = asyncio.run(
        record_keepalive_failure(
            pool, "11111111-1111-1111-1111-111111111111",
            platform="tiktok", blob=_live_blob(),
            error="503 Service Unavailable", fail_count=1,
        )
    )
    sql = _written_sql(pool)
    assert dead is False
    assert "oauth_next_retry_at" in sql
    assert "oauth_fail_count = COALESCE(oauth_fail_count, 0) + 1" in sql
    assert "needs_reconnection" not in sql


def test_permanent_failure_writes_needs_reconnection():
    import asyncio

    from services.platform_oauth_refresh import record_keepalive_failure

    pool = _FakePool()
    dead = asyncio.run(
        record_keepalive_failure(
            pool, "11111111-1111-1111-1111-111111111111",
            platform="tiktok", blob=_live_blob(),
            error="invalid_grant", fail_count=1,
        )
    )
    assert dead is True
    assert "oauth_health" in _written_sql(pool)


def test_successful_refresh_clears_failure_state():
    import asyncio

    from services.platform_oauth_refresh import record_keepalive_success

    pool = _FakePool()
    asyncio.run(
        record_keepalive_success(
            pool, "11111111-1111-1111-1111-111111111111",
            platform="tiktok", blob=_live_blob(),
        )
    )
    sql = _written_sql(pool)
    assert "oauth_fail_count" in sql
    assert "oauth_next_retry_at" in sql
    assert "oauth_last_verified_at" in sql


def test_keepalive_sweep_covers_every_token_table():
    """A row parked in a legacy table must not be stranded unrefreshed."""
    import inspect

    from services import platform_oauth_refresh as por

    src = inspect.getsource(por.sweep_platform_token_keepalive)
    listing = src.split("for table in _oauth_tables():")[1].split("except Exception")[0]
    # Old behaviour stopped at the first table that returned rows.
    assert "break" not in listing.replace("if remaining <= 0:\n                    break", "")
    assert "rows.extend" in listing


def test_reconnect_clears_keepalive_failure_state():
    """Reconnecting must not leave the account in backoff or reading as degraded."""
    import inspect

    from routers import oauth as oauth_router
    from services import oauth_meta_finish
    from services.platform_oauth_refresh import OAUTH_RECONNECT_RESET_SQL

    for col in (
        "oauth_health = 'ok'",
        "oauth_fail_count = 0",
        "oauth_next_retry_at = NULL",
        "oauth_reconnect_alert_at = NULL",
        "access_expires_at = NULL",
        "refresh_expires_at = NULL",
    ):
        assert col in OAUTH_RECONNECT_RESET_SQL, col

    # Both connect paths apply the reset rather than only stamping health.
    for mod in (oauth_router, oauth_meta_finish):
        src = inspect.getsource(mod)
        assert "OAUTH_RECONNECT_RESET_SQL" in src
        assert "last_oauth_reconnect_at = NOW(),\n                                oauth_health" not in src


# ── Connection evaluation ────────────────────────────────────────────────────


def _row(**over):
    base = {
        "id": "tok-1",
        "user_id": "user-1",
        "platform": "tiktok",
        "account_name": "demo",
        "oauth_health": "ok",
        "oauth_fail_count": 0,
        "oauth_last_verified_at": NOW - timedelta(hours=1),
        "access_expires_at": NOW + timedelta(hours=1),
        "refresh_expires_at": NOW + timedelta(days=300),
    }
    base.update(over)
    return base


def test_healthy_connection_is_ok_with_a_publish_horizon():
    got = evaluate_connection(_row(), now=NOW)
    assert got["state"] == STATE_OK
    assert got["publishable_days"] == 300
    assert got["publishable_until"] is not None


def test_needs_reconnection_health_surfaces():
    got = evaluate_connection(_row(oauth_health="needs_reconnection"), now=NOW)
    assert got["state"] == STATE_NEEDS_RECONNECT


def test_expired_grant_surfaces_even_when_health_says_ok():
    got = evaluate_connection(_row(refresh_expires_at=NOW - timedelta(days=1)), now=NOW)
    assert got["state"] == STATE_NEEDS_RECONNECT
    assert got["publishable_days"] == 0


def test_grant_expiring_soon_is_flagged_before_it_breaks():
    got = evaluate_connection(_row(refresh_expires_at=NOW + timedelta(days=3)), now=NOW)
    assert got["state"] == STATE_EXPIRING


def test_consecutive_failures_report_degraded_not_dead():
    got = evaluate_connection(_row(oauth_fail_count=3), now=NOW)
    assert got["state"] == STATE_DEGRADED
    assert "3 consecutive" in got["reasons"][0]


def test_never_verified_reads_as_unverified():
    """A stalled sweep must be visible rather than looking healthy."""
    assert evaluate_connection(_row(oauth_last_verified_at=None), now=NOW)["state"] == STATE_STALE
    stale = _row(oauth_last_verified_at=NOW - timedelta(days=30))
    assert evaluate_connection(stale, now=NOW)["state"] == STATE_STALE


def test_no_known_grant_expiry_has_open_horizon():
    got = evaluate_connection(_row(refresh_expires_at=None), now=NOW)
    assert got["publishable_until"] is None
    assert got["publishable_days"] is None


# ── Scheduled-work risk ──────────────────────────────────────────────────────


def _conn(**over):
    return evaluate_connection(_row(**over), now=NOW)


def _upload(**over):
    base = {
        "id": "up-1",
        "user_id": "user-1",
        "platforms": ["tiktok"],
        "target_accounts": ["tok-1"],
        "scheduled_time": NOW + timedelta(days=10),
    }
    base.update(over)
    return base


def test_post_inside_grant_window_is_not_at_risk():
    out = assess_scheduled_risk([_upload()], [_conn()], now=NOW)
    assert out["slots_at_risk"] == 0
    assert out["slots_checked"] == 1


def test_post_scheduled_past_grant_expiry_is_at_risk():
    """The core long-horizon guarantee: a 2-year post on a 300-day grant is flagged."""
    up = _upload(scheduled_time=NOW + timedelta(days=730))
    out = assess_scheduled_risk([up], [_conn()], now=NOW)
    assert out["slots_at_risk"] == 1
    assert out["by_reason"][RISK_GRANT_EXPIRES_FIRST] == 1
    assert out["at_risk"][0]["days_out"] == 730
    assert out["uploads_at_risk"] == 1
    assert out["users_affected"] == 1


def test_dead_connection_puts_its_scheduled_posts_at_risk():
    out = assess_scheduled_risk(
        [_upload()], [_conn(oauth_health="needs_reconnection")], now=NOW
    )
    assert out["by_reason"][RISK_NEEDS_RECONNECT] == 1


def test_missing_connection_is_reported():
    out = assess_scheduled_risk([_upload(target_accounts=[])], [], now=NOW)
    assert out["by_reason"][RISK_NO_CONNECTION] == 1


def test_any_surviving_target_account_clears_the_risk():
    up = _upload(target_accounts=["tok-1", "tok-2"], scheduled_time=NOW + timedelta(days=500))
    conns = [
        _conn(id="tok-1", refresh_expires_at=NOW + timedelta(days=100)),
        _conn(id="tok-2", refresh_expires_at=NOW + timedelta(days=900)),
    ]
    assert assess_scheduled_risk([up], conns, now=NOW)["slots_at_risk"] == 0


def test_already_due_slots_are_not_evaluated():
    past = _upload(scheduled_time=NOW - timedelta(days=1))
    out = assess_scheduled_risk([past], [_conn()], now=NOW)
    assert out["slots_checked"] == 0
    assert out["slots_at_risk"] == 0


def test_per_platform_times_beat_the_upload_level_time():
    """Smart Schedule stores a different slot per platform; risk is per slot."""
    up = _upload(
        platforms=["tiktok", "youtube"],
        target_accounts=[],
        platform_times={
            "tiktok": NOW + timedelta(days=10),
            "youtube": NOW + timedelta(days=800),
        },
    )
    conns = [
        _conn(id="tok-1", platform="tiktok", refresh_expires_at=NOW + timedelta(days=300)),
        _conn(id="tok-2", platform="youtube", refresh_expires_at=NOW + timedelta(days=300)),
    ]
    out = assess_scheduled_risk([up], conns, now=NOW)
    assert out["slots_checked"] == 2
    assert out["slots_at_risk"] == 1
    assert out["at_risk"][0]["platform"] == "youtube"


def test_unlimited_grant_covers_any_horizon():
    up = _upload(scheduled_time=NOW + timedelta(days=4000))
    out = assess_scheduled_risk([up], [_conn(refresh_expires_at=None)], now=NOW)
    assert out["slots_at_risk"] == 0


def test_risk_detail_is_capped_but_counts_stay_exact():
    ups = [
        _upload(id=f"up-{i}", scheduled_time=NOW + timedelta(days=730))
        for i in range(25)
    ]
    out = assess_scheduled_risk(ups, [_conn()], now=NOW, detail_limit=10)
    assert out["slots_at_risk"] == 25
    assert len(out["at_risk"]) == 10
    assert out["at_risk_truncated"] is True


# ── Reconnect alert wiring ───────────────────────────────────────────────────


def test_reconnect_alert_job_is_registered_and_runnable_by_admin():
    import inspect

    from api.schemas.admin_requests import AdminEmailJobRunRequest
    from services.admin_email_jobs import ADMIN_EMAIL_JOBS

    assert "oauth_reconnect_alerts" in ADMIN_EMAIL_JOBS
    allowed = AdminEmailJobRunRequest.model_fields["job"].annotation
    assert "oauth_reconnect_alerts" in str(allowed)
    # Reachable from the cron dispatcher, which runs every registered job.
    assert inspect.iscoroutinefunction(ADMIN_EMAIL_JOBS["oauth_reconnect_alerts"])


def test_reconnect_alert_respects_email_opt_out_and_repeat_window():
    import inspect

    from services import oauth_reconnect_alerts as alerts

    recipients_sql = inspect.getsource(alerts._notifiable_recipients)
    assert "COALESCE(up.email_notifications, TRUE) = TRUE" in recipients_sql
    assert "u.status = 'active'" in recipients_sql

    # Claim-before-send keeps a long unattended run from re-spamming.
    claim_sql = inspect.getsource(alerts._claim_reconnect_alert)
    assert "oauth_reconnect_alert_at IS NULL" in claim_sql
    assert "RETURNING id" in claim_sql
    assert alerts.RECONNECT_ALERT_REPEAT_DAYS >= 1


def test_at_risk_counts_per_account_are_exact_despite_detail_cap():
    """Reconnect emails quote these counts, so truncation must not reach them."""
    ups = [
        _upload(id=f"up-{i}", scheduled_time=NOW + timedelta(days=730))
        for i in range(30)
    ]
    out = assess_scheduled_risk(ups, [_conn()], now=NOW, detail_limit=0)
    assert out["at_risk"] == []
    assert out["by_user_platform"]["user-1|tiktok"] == 30


def test_readiness_payload_keeps_the_keys_the_accounts_ui_reads():
    """frontend/js/connection-readiness.js binds to these names — pin them."""
    conn = _conn()
    for key in ("token_row_id", "platform", "state", "publishable_until", "publishable_days"):
        assert key in conn, key

    risk = assess_scheduled_risk(
        [_upload(scheduled_time=NOW + timedelta(days=730))], [conn], now=NOW
    )
    for key in ("slots_at_risk", "uploads_at_risk", "by_user_platform"):
        assert key in risk, key

    summary = summarize_connections([conn])
    for key in ("needs_reconnection", "expiring_soon", "total"):
        assert key in summary, key

    # The UI keys per-account risk as "<user_id>|<platform>".
    assert "user-1|tiktok" in risk["by_user_platform"]

    # States the UI has styling for.
    assert conn["state"] in (
        STATE_OK, STATE_DEGRADED, STATE_EXPIRING, STATE_STALE, STATE_NEEDS_RECONNECT,
    )


def test_summary_rolls_up_states_and_soonest_expiry():
    conns = [
        _conn(id="a"),
        _conn(id="b", oauth_health="needs_reconnection"),
        _conn(id="c", refresh_expires_at=NOW + timedelta(days=2)),
    ]
    out = summarize_connections(conns)
    assert out["total"] == 3
    assert out["needs_reconnection"] == 1
    assert out["expiring_soon"] == 1
    assert out["soonest_grant_expiry"] is not None
    assert "tiktok" in out["by_platform"]
