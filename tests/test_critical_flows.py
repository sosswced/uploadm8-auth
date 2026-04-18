"""
Critical-path helpers: SQL allowlist unit test + optional app smoke when env is set.
"""
from __future__ import annotations

import os

import pytest


def test_sql_allowlist_rejects_unknown_column():
    from core.sql_allowlist import assert_set_fragments_columns

    with pytest.raises(ValueError, match="disallowed"):
        assert_set_fragments_columns(["evil = $1"], frozenset({"name"}))


def test_sql_allowlist_rejects_unknown_relation():
    from core.sql_allowlist import OAUTH_TOKEN_STORAGE_TABLES, assert_relation_name

    with pytest.raises(ValueError, match="disallowed"):
        assert_relation_name("other_table", OAUTH_TOKEN_STORAGE_TABLES)


@pytest.mark.skipif(
    not os.environ.get("JWT_SECRET")
    or not os.environ.get("TOKEN_ENC_KEYS")
    or os.environ.get("UPLOADM8_INTEGRATION", "").strip().lower() not in ("1", "true", "yes"),
    reason="Needs JWT_SECRET, TOKEN_ENC_KEYS, UPLOADM8_INTEGRATION=1",
)
def test_app_module_import_smoke():
    import app  # noqa: F401
