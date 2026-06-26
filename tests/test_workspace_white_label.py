"""Unit tests for workspace access helpers and white-label owner scoping."""

from __future__ import annotations

from services.white_label import (
    can_manage_white_label,
    resolve_white_label_logo_url,
    white_label_owner_user_id,
)
from services.workspace import (
    can_manage_billing,
    can_manage_platforms,
    can_upload_from_user,
    resolve_billing_user_id,
    workspace_capabilities,
    workspace_role_from_user,
)


def test_resolve_billing_user_id_prefers_owner():
    user = {"id": "member-1", "billing_user_id": "owner-1"}
    assert resolve_billing_user_id(user) == "owner-1"


def test_workspace_role_defaults_owner_for_solo():
    assert workspace_role_from_user({"id": "solo"}) == "owner"


def test_viewer_cannot_manage_platforms_or_billing():
    user = {"workspace": {"role": "viewer", "id": "ws-1"}}
    assert can_manage_platforms(user) is False
    assert can_manage_billing(user) is False
    assert can_upload_from_user(user) is False


def test_editor_can_upload_but_not_billing():
    user = {"workspace": {"role": "editor", "id": "ws-1"}}
    assert can_upload_from_user(user) is True
    assert can_manage_billing(user) is False


def test_workspace_capabilities_shape():
    caps = workspace_capabilities({"workspace": {"role": "admin"}})
    assert caps["role"] == "admin"
    assert caps["can_manage_team"] is True
    assert caps["can_upload"] is True


def test_white_label_owner_uses_billing_user():
    user = {"id": "m1", "billing_user_id": "o1"}
    assert white_label_owner_user_id(user) == "o1"


def test_can_manage_white_label_for_editor():
    user = {"workspace": {"role": "editor"}}
    assert can_manage_white_label(user) is False


def test_resolve_white_label_logo_url_https_passthrough():
    assert resolve_white_label_logo_url("https://example.com/logo.png") == "https://example.com/logo.png"


def test_viewer_cannot_edit_settings_guard():
    from fastapi import HTTPException
    from services.workspace import require_can_edit_settings

    try:
        require_can_edit_settings({"workspace": {"role": "viewer"}})
        assert False, "expected HTTPException"
    except HTTPException as exc:
        assert exc.status_code == 403


def test_workspace_member_billing_id():
    user = {"id": "member", "billing_user_id": "owner"}
    assert resolve_billing_user_id(user) != user["id"]
