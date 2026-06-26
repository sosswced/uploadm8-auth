"""R2 storage guard and complete-time source verification."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from services.upload.complete import complete_upload_transaction
from services.upload.r2_storage_guard import (
    ERROR_SOURCE_NOT_IN_R2,
    SOURCE_NOT_IN_R2_MESSAGE,
    classify_r2_head_not_found,
)
from services.retry_policy import classify_retry_error
from services.upload.schedule_guard import UPLOAD_ERROR_MESSAGES


def test_classify_r2_head_not_found_matches_boto_message():
    exc = Exception(
        "An error occurred (404) when calling the HeadObject operation: Not Found"
    )
    out = classify_r2_head_not_found(exc)
    assert out == (ERROR_SOURCE_NOT_IN_R2, SOURCE_NOT_IN_R2_MESSAGE)


def test_classify_r2_head_not_found_ignores_other_errors():
    assert classify_r2_head_not_found(Exception("connection reset")) is None


def test_source_not_in_r2_message_in_upload_error_messages():
    assert UPLOAD_ERROR_MESSAGES[ERROR_SOURCE_NOT_IN_R2] == SOURCE_NOT_IN_R2_MESSAGE


def test_source_not_in_r2_blocks_auto_and_user_retry():
    decision = classify_retry_error(ERROR_SOURCE_NOT_IN_R2)
    assert decision.allowed is False
    assert decision.http_status == 409


def test_complete_rejects_missing_r2_object():
    upload_row = {
        "id": "adb8a4f6-2ac1-4e26-a394-bb2a9859b458",
        "user_id": "user-1",
        "status": "pending",
        "schedule_mode": "immediate",
        "r2_key": "uploads/user-1/u1/video.mp4",
        "platforms": ["tiktok"],
    }
    executed: list[tuple] = []

    class FakeConn:
        async def fetchrow(self, *_a, **_k):
            return upload_row

        async def execute(self, sql, *args):
            executed.append((sql, args))

    async def _run():
        with patch(
            "services.upload.complete.get_user_prefs_for_upload",
            new=AsyncMock(return_value={}),
        ), patch(
            "services.upload.complete.validate_upload_row_tiktok_settings",
            new=AsyncMock(return_value=None),
        ), patch(
            "services.upload.complete.asyncio.to_thread",
            new=AsyncMock(return_value=False),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await complete_upload_transaction(FakeConn(), upload_row["id"], "user-1", {})
            return exc_info

    exc_info = asyncio.run(_run())
    assert exc_info.value.status_code == 409
    assert exc_info.value.detail["code"] == ERROR_SOURCE_NOT_IN_R2
    assert "video file" in exc_info.value.detail["message"].lower()
    sql_blob = " ".join(sql for sql, _ in executed).lower()
    assert "error_code" in sql_blob
    assert any(
        ERROR_SOURCE_NOT_IN_R2 in (args or ())
        for _sql, args in executed
    )
