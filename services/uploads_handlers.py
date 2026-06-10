"""
Upload route business logic: presign insert, complete transaction, list/detail shaping, queue stats.

Backward-compat shim — import from ``services.upload`` or this module interchangeably.
Routers keep HTTP wiring (cookies, Request, presigned URL generation, enqueue, audit).
"""

from services.upload import *  # noqa: F403
from services.upload import __all__  # noqa: F401
