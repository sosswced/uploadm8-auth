"""Pikzels error formatting — never emit empty ops messages."""

from services.pikzels_errors import format_pikzels_error_message


def test_format_prefers_documented_error_envelope():
    assert (
        format_pikzels_error_message({"error": {"code": "VALIDATION_ERROR", "message": "bad prompt"}})
        == "VALIDATION_ERROR: bad prompt"
    )


def test_format_uses_raw_non_json_body():
    assert "content policy" in format_pikzels_error_message({"raw": "content policy rejected"}).lower()


def test_format_never_empty_on_blank_object():
    msg = format_pikzels_error_message({})
    assert msg
    assert msg == "upstream_error"


def test_format_dumps_opaque_dict():
    msg = format_pikzels_error_message({"foo": "bar", "n": 1})
    assert "foo" in msg and "bar" in msg
