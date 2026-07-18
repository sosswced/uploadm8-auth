"""Unit tests for Meta signed_request parsing (Facebook data deletion callback)."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json

from routers.meta_compliance import parse_meta_signed_request


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _make_signed_request(payload: dict, secret: str) -> str:
    raw = _b64url(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    sig = hmac.new(secret.encode("utf-8"), raw.encode("utf-8"), hashlib.sha256).digest()
    return f"{_b64url(sig)}.{raw}"


def test_parse_meta_signed_request_ok():
    secret = "test_app_secret"
    payload = {"algorithm": "HMAC-SHA256", "user_id": "218471", "issued_at": 1291836800}
    signed = _make_signed_request(payload, secret)
    data = parse_meta_signed_request(signed, secret)
    assert data is not None
    assert data["user_id"] == "218471"


def test_parse_meta_signed_request_bad_sig():
    secret = "test_app_secret"
    payload = {"algorithm": "HMAC-SHA256", "user_id": "218471"}
    signed = _make_signed_request(payload, secret)
    assert parse_meta_signed_request(signed, "wrong_secret") is None


def test_parse_meta_signed_request_empty():
    assert parse_meta_signed_request("", "secret") is None
    assert parse_meta_signed_request("no-dot", "secret") is None
    assert parse_meta_signed_request("a.b", "") is None
