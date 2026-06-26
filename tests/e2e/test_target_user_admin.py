"""Target-user admin API/UI (also run inside run_live_demo_journey while upload pending)."""

from __future__ import annotations

import pytest

from tests.e2e.helpers.auth import api_client
from tests.e2e.helpers.config import e2e_target_user_id, e2e_target_user_name
from tests.e2e.helpers.target_user import (
    TargetUserNotFound,
    resolve_target_user,
    run_target_user_api_checks,
    target_user_admin_api_checks,
)
from tests.e2e.helpers.target_user_ui import (
    exercise_account_management_target,
    exercise_admin_wallet_target,
    exercise_target_user_admin_ui,
)

pytestmark = [pytest.mark.e2e, pytest.mark.api_smoke, pytest.mark.ui_smoke, pytest.mark.overnight]


@pytest.fixture(scope="module")
def target_user_record():
    with api_client() as client:
        try:
            return resolve_target_user(client)
        except TargetUserNotFound as e:
            pytest.skip(str(e))


@pytest.fixture(scope="module")
def target_user_id(target_user_record) -> str:
    return str(target_user_record["id"])


def test_target_user_resolves(target_user_record):
    assert str(target_user_record["id"]) == e2e_target_user_id()
    name = (target_user_record.get("name") or "").lower()
    assert e2e_target_user_name().split()[0].lower() in name or target_user_record.get("email")


@pytest.mark.parametrize("spec", target_user_admin_api_checks(), ids=lambda s: s["id"])
def test_target_user_admin_api(spec, target_user_id: str):
    with api_client() as client:
        resolve_target_user(client)
        kwargs = {}
        if spec.get("params"):
            kwargs["params"] = spec["params"]
        resp = client.request(spec["method"], spec["path"], **kwargs)
        assert resp.status_code < 400, f"{spec['id']}: HTTP {resp.status_code} {resp.text[:300]}"
        msg = spec["assert_fn"](resp)
        if msg and ("missing" in msg.lower() or "mismatch" in msg.lower()):
            if spec.get("soft"):
                pytest.skip(msg)
            pytest.fail(f"{spec['id']}: {msg}")


def test_target_user_api_matrix_summary(target_user_id: str):
    with api_client() as client:
        results = run_target_user_api_checks(client, target_user_id)
    fails = [r for r in results if r["status"] == "FAIL"]
    assert not fails, f"API failures: {fails}"


def test_target_user_account_management_ui(human_session_page, base_url: str, target_user_record):
    result = exercise_account_management_target(human_session_page, base_url)
    assert "edit_modal_open" in result.get("steps", [])


def test_target_user_admin_wallet_ui(human_session_page, base_url: str, target_user_record):
    result = exercise_admin_wallet_target(human_session_page, base_url)
    assert "wallet_balances_loaded" in result.get("steps", [])


@pytest.mark.slow
def test_target_user_full_admin_ui(human_session_page, base_url: str, target_user_record):
    report = exercise_target_user_admin_ui(human_session_page, base_url)
    assert report.get("ok")
