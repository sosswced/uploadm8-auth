"""Unit-test session guards (non-e2e).

Disable Sentry so intentional logger.error / fail-closed paths in unit tests
cannot open or regress production issues (e.g. UPLOADM8-AP).
"""

from __future__ import annotations

import os

# Must run before app/worker import paths call init_sentry_* with .env DSN.
os.environ["SENTRY_DSN"] = ""
# before_send drop for any residual LoggingIntegration (UPLOADM8-AP).
os.environ["SENTRY_DROP_PYTEST_EVENTS"] = "1"

try:
    import sentry_sdk

    sentry_sdk.init(dsn=None)
except Exception:
    pass
