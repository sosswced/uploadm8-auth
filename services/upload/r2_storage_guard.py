"""R2 presence checks for upload complete and pipeline error classification."""

from __future__ import annotations

from typing import Any, Literal, Optional, Tuple

from core.r2 import r2_object_head_status

ERROR_SOURCE_NOT_IN_R2 = "SOURCE_NOT_IN_R2"
ERROR_STORAGE_CHECK_UNAVAILABLE = "STORAGE_CHECK_UNAVAILABLE"

SOURCE_NOT_IN_R2_MESSAGE = (
    "The video file did not finish uploading to storage. "
    "Re-upload the file, then complete the upload again."
)

R2HeadStatus = Literal["present", "missing", "unknown"]


def upload_row_source_r2_key(upload_row: Any) -> str:
    if not upload_row:
        return ""
    key = upload_row.get("r2_key") if isinstance(upload_row, dict) else upload_row["r2_key"]
    return str(key or "").strip()


def upload_source_head_status(upload_row: Any) -> R2HeadStatus:
    """``present`` / ``missing`` / ``unknown`` — never conflate unknown with missing."""
    key = upload_row_source_r2_key(upload_row)
    if not key:
        return "missing"
    return r2_object_head_status(key)  # type: ignore[return-value]


def upload_source_present_in_r2(upload_row: Any) -> bool:
    return upload_source_head_status(upload_row) == "present"


def upload_source_definitely_missing_in_r2(upload_row: Any) -> bool:
    return upload_source_head_status(upload_row) == "missing"


async def mark_source_not_in_r2_failed(
    conn: Any,
    upload_id: str,
    *,
    detail: str | None = None,
) -> None:
    text = (detail or SOURCE_NOT_IN_R2_MESSAGE).strip()[:4000]
    await conn.execute(
        """
        UPDATE uploads
        SET status = 'failed',
            error_code = $2,
            error_detail = $3,
            updated_at = NOW()
        WHERE id = $1
        """,
        upload_id,
        ERROR_SOURCE_NOT_IN_R2,
        text,
    )


def classify_r2_head_not_found(exc: BaseException) -> Optional[Tuple[str, str]]:
    """Map boto HeadObject 404 (missing source object) to a user-facing error code."""
    msg = str(exc or "")
    low = msg.lower()
    if "headobject" not in low:
        return None
    if "404" not in msg and "not found" not in low:
        return None
    return ERROR_SOURCE_NOT_IN_R2, SOURCE_NOT_IN_R2_MESSAGE
