"""asyncpg JSON param encoder: lists/dicts encode; pre-dumped strings pass through."""

from __future__ import annotations

import json

from stages.asyncpg_json_codecs import json_param_encoder


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
