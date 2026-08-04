"""TikTok export settings resolve — stale token UUID / reconnect rematch."""

from services.tiktok_api import resolve_tiktok_post_settings_for_account


def _entry(**overrides):
    base = {
        "privacy_level": "FOLLOWER_OF_CREATOR",
        "user_consent": True,
        "allow_comment": True,
        "allow_duet": False,
        "allow_stitch": False,
    }
    base.update(overrides)
    return base


def test_resolve_exact_token_row_id():
    raw = {"by_account": {"aaa": _entry()}}
    out = resolve_tiktok_post_settings_for_account(raw, "aaa")
    assert out and out["privacy_level"] == "FOLLOWER_OF_CREATOR"


def test_resolve_single_stale_key_when_live_token_differs():
    """Ghost target UUID in by_account; publish uses current primary token id."""
    raw = {"by_account": {"710dd785-fcf1-472f-9b27-8e2e348047ca": _entry()}}
    out = resolve_tiktok_post_settings_for_account(
        raw, "46bdc2b9-2536-4a83-ae58-b8bdf7c76a79"
    )
    assert out and out["privacy_level"] == "FOLLOWER_OF_CREATOR"


def test_resolve_by_platform_account_open_id():
    raw = {
        "by_account": {
            "old-token": _entry(platform_account_id="-000m0n2wq8CvaviqRKFF-AT6aGWaMEovNqE"),
            "other": _entry(
                privacy_level="PUBLIC_TO_EVERYONE",
                platform_account_id="other-open-id",
            ),
        }
    }
    out = resolve_tiktok_post_settings_for_account(
        raw,
        "new-token-uuid",
        platform_account_id="-000m0n2wq8CvaviqRKFF-AT6aGWaMEovNqE",
    )
    assert out and out["privacy_level"] == "FOLLOWER_OF_CREATOR"


def test_resolve_missing_when_multi_and_no_match():
    raw = {
        "by_account": {
            "a": _entry(),
            "b": _entry(privacy_level="PUBLIC_TO_EVERYONE"),
        }
    }
    assert (
        resolve_tiktok_post_settings_for_account(raw, "c", platform_account_id="x")
        is None
    )
