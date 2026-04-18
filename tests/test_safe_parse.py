"""Unit tests for stages.safe_parse canonical JSON helpers."""

import json

import pytest

from stages.safe_parse import json_dict, json_list


def test_json_dict_from_dict():
    assert json_dict({"a": 1}) == {"a": 1}


def test_json_dict_from_str():
    assert json_dict('{"x": 2}', default={"z": 0}) == {"x": 2}


def test_json_dict_invalid_returns_default():
    assert json_dict("not json", default={"k": True}) == {"k": True}


def test_json_dict_list_json_returns_default():
    assert json_dict("[1,2]", default={}) == {}


def test_json_list_from_list():
    assert json_list([{"p": "tiktok"}]) == [{"p": "tiktok"}]


def test_json_list_from_str():
    raw = json.dumps([1, 2, 3])
    assert json_list(raw) == [1, 2, 3]


def test_json_list_dict_json_returns_default():
    assert json_list('{"a": 1}', default=[]) == []


@pytest.mark.parametrize("raw", [None, "", "   "])
def test_json_empty_defaults(raw):
    assert json_dict(raw, default={"d": 1}) == {"d": 1}
    assert json_list(raw, default=[0]) == [0]
