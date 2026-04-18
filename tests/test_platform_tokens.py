"""Unit tests for platform token DB slug consistency (publish + verify)."""

from stages.platform_tokens import platform_tokens_db_key


def test_platform_tokens_db_key_oauth_slugs():
    assert platform_tokens_db_key("tiktok") == "tiktok"
    assert platform_tokens_db_key("YouTube") == "youtube"
    assert platform_tokens_db_key("INSTAGRAM") == "instagram"
    assert platform_tokens_db_key("facebook") == "facebook"


def test_platform_tokens_db_key_google_alias():
    assert platform_tokens_db_key("google") == "youtube"


def test_platform_tokens_db_key_unknown_pass_through():
    assert platform_tokens_db_key("some_future_platform") == "some_future_platform"
