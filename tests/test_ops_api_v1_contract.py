"""GET /api/v1 documents canonical /api paths — it is not a route alias."""

from __future__ import annotations

import inspect

from routers import ops as ops_mod


def test_api_v1_contract_does_not_claim_aliased_paths():
    src = inspect.getsource(ops_mod.api_v1_contract)
    assert "/api/v1/auth/login" not in src
    assert 'canonical_prefix": "/api"' in src or "canonical_prefix" in src
    assert "/api/auth/login" in src
