"""Unit tests for TikTok export settings validation."""

import pytest
from fastapi import HTTPException

from services.upload.tiktok import _validate_tiktok_by_account_settings


def test_tiktok_settings_required_structured_error():
    with pytest.raises(HTTPException) as exc:
        _validate_tiktok_by_account_settings(None, ["acc-1"])
    assert exc.value.status_code == 400
    detail = exc.value.detail
    assert isinstance(detail, dict)
    assert detail.get("code") == "tiktok_settings_required"


def test_tiktok_settings_incomplete_accounts():
    with pytest.raises(HTTPException) as exc:
        _validate_tiktok_by_account_settings({"by_account": {}}, ["acc-1", "acc-2"])
    detail = exc.value.detail
    assert detail.get("code") == "tiktok_settings_incomplete"
