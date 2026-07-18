"""E2E auth helper retry behaviour (unit-level, no live API)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx

from tests.e2e.helpers.auth import api_request_with_retry


def test_api_request_retries_on_401():
    expired = httpx.Response(401, request=httpx.Request("GET", "http://test/api/uploads"))
    ok = httpx.Response(200, request=httpx.Request("GET", "http://test/api/uploads"))

    client = MagicMock(spec=httpx.Client)
    client.request.side_effect = [expired, ok]

    with patch("tests.e2e.helpers.auth.refresh_api_client") as refresh:
        refresh.return_value = client
        resp, active = api_request_with_retry(client, "GET", "/api/uploads", attempts=2)

    assert resp.status_code == 200
    assert refresh.called
    assert client.request.call_count == 2
