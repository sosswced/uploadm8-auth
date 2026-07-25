"""asyncpg JSON param encoder: lists/dicts encode; pre-dumped strings pass through."""

from __future__ import annotations

import json

from stages.asyncpg_json_codecs import json_param_encoder
from stages.db import _jsonb_bind


def test_json_param_encoder_list_for_platform_results():
    payload = [{"platform": "tiktok", "success": True}]
    encoded = json_param_encoder(payload)
    assert isinstance(encoded, str)
    assert json.loads(encoded) == payload


def test_json_param_encoder_passthrough_string():
    raw = json.dumps({"a": 1})
    assert json_param_encoder(raw) == raw


def test_json_param_encoder_dict():
    assert json.loads(json_param_encoder({"x": 2})) == {"x": 2}


def test_jsonb_bind_encodes_platform_results_list():
    """Regression: worker pool without codecs + raw list → asyncpg DataError."""
    payload = [{"platform": "tiktok", "success": True, "publish_id": "v_pub"}]
    bound = _jsonb_bind(payload)
    assert isinstance(bound, str)
    assert json.loads(bound) == payload
    assert _jsonb_bind(None) is None
    assert _jsonb_bind(bound) == bound  # already-encoded string passes through


def test_helpers_fallback_encoder_passthrough_string():
    """Fallback codecs must not double-encode pre-bound jsonb strings."""
    import asyncio

    import stages.asyncpg_json_codecs as codecs_mod
    from core.helpers import _init_asyncpg_codecs

    registered = {}

    class FakeConn:
        async def set_type_codec(self, name, encoder=None, decoder=None, schema=None):
            registered[name] = encoder

    async def fail_apply(*_a, **_k):
        raise RuntimeError("forced apply fail")

    orig = codecs_mod.apply_asyncpg_json_codecs
    codecs_mod.apply_asyncpg_json_codecs = fail_apply
    try:
        asyncio.run(_init_asyncpg_codecs(FakeConn()))
    finally:
        codecs_mod.apply_asyncpg_json_codecs = orig

    assert "jsonb" in registered
    raw = json.dumps([{"platform": "tiktok"}])
    assert registered["jsonb"](raw) == raw
    assert json.loads(registered["jsonb"]([{"a": 1}])) == [{"a": 1}]
