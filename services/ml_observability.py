"""
Optional ML observability helpers for HF + Trackio integration.

This module is intentionally fail-safe:
- If trackio is missing or not configured, calls become no-ops.
- If HF token is missing, scripts can skip push operations cleanly.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("uploadm8.ml_observability")


def hf_write_token() -> str:
    """Write-capable Hub token: ``HF_TOKEN`` first, then ``HUGGING_FACE_HUB_TOKEN``."""
    return (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()


def hf_env_status(*, require_write_token: bool = False) -> Tuple[bool, str]:
    """
    Validate Hugging Face env configuration for dataset operations.
    """
    token = hf_write_token()
    if not token:
        return False, "Hub token is not set (HF_TOKEN or HUGGING_FACE_HUB_TOKEN)"
    if require_write_token and len(token) < 16:
        return False, "Hub token appears invalid/too short"
    return True, "ok"


def trackio_env_enabled() -> bool:
    """
    Trackio is enabled only when explicit project configuration is present.
    """
    project = (os.environ.get("TRACKIO_PROJECT") or "").strip()
    return bool(project)


class OptionalTrackioRun:
    """
    Lightweight wrapper around trackio with safe fallbacks.
    """

    def __init__(self, run_name: str, *, default_project: str = "uploadm8-ml"):
        self.run_name = run_name
        self.default_project = default_project
        self._enabled = False
        self._trackio = None

    def start(self, *, config: Optional[Dict[str, Any]] = None) -> bool:
        if not trackio_env_enabled():
            return False
        try:
            import trackio  # type: ignore

            project = (os.environ.get("TRACKIO_PROJECT") or self.default_project).strip()
            raw_space_id = (os.environ.get("TRACKIO_SPACE_ID") or "").strip()
            space_id = raw_space_id or None
            if space_id and ("/" not in space_id or space_id.endswith("/") or space_id.startswith("/")):
                logger.warning("TRACKIO_SPACE_ID is invalid; ignoring value")
                space_id = None
            run_id = (os.environ.get("TRACKIO_RUN_ID") or "").strip()
            if not run_id or "${" in run_id or run_id.endswith("-"):
                run_id = self.run_name
            kwargs: Dict[str, Any] = {
                "project": project,
                "name": run_id,
                "config": config or {},
            }
            if space_id:
                kwargs["space_id"] = space_id
            trackio.init(**kwargs)
            self._enabled = True
            self._trackio = trackio
            return True
        except Exception as e:
            logger.warning("trackio init skipped: %s", e)
            self._enabled = False
            self._trackio = None
            return False

    def log(self, payload: Dict[str, Any]) -> None:
        if not self._enabled or self._trackio is None:
            return
        try:
            self._trackio.log(payload)
        except Exception as e:
            logger.warning("trackio log failed: %s", e)

    def finish(self) -> None:
        if not self._enabled or self._trackio is None:
            return
        try:
            self._trackio.finish()
        except Exception as e:
            logger.warning("trackio finish failed: %s", e)
        finally:
            # Mark inactive so a second finish()/log() is a safe no-op and we never
            # call into a torn-down global trackio run.
            self._enabled = False
