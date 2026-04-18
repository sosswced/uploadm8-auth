"""Session resolution when Bearer is stale but cookie is valid."""
from unittest.mock import patch

from core.deps import _resolve_user_id_from_session


def test_prefers_valid_bearer():
    with patch("core.deps.verify_access_jwt") as mock_verify:

        def side_effect(tok):
            if tok == "bearer-ok":
                return "user-b"
            if tok == "cookie-ok":
                return "user-c"
            return None

        mock_verify.side_effect = side_effect
        uid, reason = _resolve_user_id_from_session(
            "Bearer bearer-ok", {"uploadm8_access": "cookie-ok"}
        )
        assert uid == "user-b"
        assert reason == ""


def test_falls_back_to_cookie_when_bearer_invalid():
    with patch("core.deps.verify_access_jwt") as mock_verify:

        def side_effect(tok):
            if tok == "stale-bearer":
                return None
            if tok == "cookie-ok":
                return "user-from-cookie"
            return None

        mock_verify.side_effect = side_effect
        uid, reason = _resolve_user_id_from_session(
            "Bearer stale-bearer", {"uploadm8_access": "cookie-ok"}
        )
        assert uid == "user-from-cookie"
        assert reason == ""


def test_cookie_only_still_works():
    with patch("core.deps.verify_access_jwt") as mock_verify:
        mock_verify.side_effect = lambda tok: "u1" if tok == "c" else None
        uid, reason = _resolve_user_id_from_session(None, {"uploadm8_access": "c"})
        assert uid == "u1"
        assert reason == ""


def test_missing_when_no_credentials():
    uid, reason = _resolve_user_id_from_session(None, {})
    assert uid is None
    assert reason == "missing"


def test_invalid_when_both_present_but_unverifiable():
    with patch("core.deps.verify_access_jwt", return_value=None):
        uid, reason = _resolve_user_id_from_session(
            "Bearer x", {"uploadm8_access": "y"}
        )
        assert uid is None
        assert reason == "invalid"
