"""Settings Security / Billing / Token balances wiring checks."""

from __future__ import annotations

import inspect

from routers import me as me_router
from services.notification_prefs import maybe_queue_password_changed_email


def test_settings_password_endpoint_queues_security_email():
    src = inspect.getsource(me_router.update_password_settings)
    assert "maybe_queue_password_changed_email" in src
    assert "BackgroundTasks" in src or "background" in src


def test_maybe_queue_helper_exists():
    assert callable(maybe_queue_password_changed_email)


def test_billing_upload_estimate_uses_service_weights():
    from routers import billing as billing_router

    src = inspect.getsource(billing_router.billing_upload_estimate)
    assert "fetch_service_weights_map" in src
    assert "compute_presign_put_aic_costs" in src
    assert "get_user_prefs_for_upload" in src
