"""Router lint + OpenAPI GET smoke distributed across page visits while upload pending."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import httpx

from tests.e2e.helpers.auth import api_get_with_retry, api_request_with_retry
from tests.e2e.helpers.human_pace import CHROME_UA, pause_between_requests
from tests.e2e.helpers.openapi_catalog import (
    ApiGetCase,
    SLOW_PATH_RE,
    assert_acceptable_status,
    build_context,
    fetch_openapi,
    iter_read_smoke_cases,
)
from tests.e2e.helpers.upload_pace import worker_safe_mode

ROOT = Path(__file__).resolve().parents[3]


def _smoke_timeout() -> httpx.Timeout:
    read_s = float(os.environ.get("E2E_SMOKE_READ_TIMEOUT_S", "12"))
    return httpx.Timeout(connect=10.0, read=read_s, write=10.0, pool=10.0)


@dataclass
class BackgroundCheckRunner:
    """
    Queues router lint, target-user admin API, and OpenAPI GET cases.
    Each ``run_page_batch`` call advances the queues while the browser is on a page.
    """

    log: Any
    api_per_page: int = 8
    include_slow_api: bool = False
    skip_api_smoke: bool = False

    _api_cases: list[ApiGetCase] = field(default_factory=list, repr=False)
    _deferred_api_cases: list[ApiGetCase] = field(default_factory=list, repr=False)
    _api_idx: int = 0
    _router_names: list[str] = field(default_factory=list, repr=False)
    _router_errors: dict[str, list[str]] = field(default_factory=dict, repr=False)
    _router_idx: int = 0
    _target_specs: list[dict[str, Any]] = field(default_factory=list, repr=False)
    _target_idx: int = 0
    _records: list[dict[str, Any]] = field(default_factory=list, repr=False)
    _static: dict[str, Any] = field(default_factory=dict, repr=False)

    def prepare(self, client: httpx.Client) -> None:
        if self.include_slow_api:
            os.environ["E2E_INCLUDE_SLOW_API"] = "1"

        if not self.skip_api_smoke:
            try:
                openapi = fetch_openapi(client)
                ctx = build_context(client)
                all_cases = list(iter_read_smoke_cases(openapi, ctx=ctx, safe_only=True))
                if worker_safe_mode():
                    active: list[ApiGetCase] = []
                    deferred: list[ApiGetCase] = []
                    for case in all_cases:
                        if SLOW_PATH_RE.search(case.path):
                            deferred.append(case)
                        else:
                            active.append(case)
                    self._api_cases = active
                    self._deferred_api_cases = deferred
                    self.log.note(
                        f"API sweep queued: {len(active)} now, {len(deferred)} deferred until upload done"
                    )
                else:
                    self._api_cases = all_cases
                    self.log.note(f"API sweep queued: {len(self._api_cases)} GET endpoints")
            except Exception as e:
                self.log.note(f"API sweep prepare failed: {e}")

        try:
            from tools.lint_routers import ROUTERS, collect_lint_errors

            _errors, per_file = collect_lint_errors()
            self._router_errors = per_file
            self._router_names = sorted(
                f"routers/{p.name}"
                for p in ROUTERS.glob("*.py")
                if p.name != "__init__.py"
            )
            self.log.note(f"Router lint queued: {len(self._router_names)} modules")
        except Exception as e:
            self.log.note(f"Router lint prepare failed: {e}")

        from tests.e2e.helpers.target_user import target_user_admin_api_checks

        self._target_specs = target_user_admin_api_checks()

    def run_startup_static(self) -> dict[str, Any]:
        """One-shot consistency grep (fast) before the page tour."""
        script = ROOT / "tools" / "check_consistency.py"
        if not script.is_file():
            self._static["consistency"] = {"ok": True, "status": "SKIP", "detail": "script missing"}
            return self._static

        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        ok = proc.returncode == 0
        detail = (proc.stderr or proc.stdout or "").strip()[-500:]
        row = {
            "category": "consistency",
            "id": "consistency.all",
            "status": "PASS" if ok else "FAIL",
            "detail": detail,
        }
        self._records.append(row)
        self._static["consistency"] = {"ok": ok, **row}
        self.log.note(f"Consistency check: {row['status']}")
        return self._static

    def run_page_batch(
        self,
        client: httpx.Client,
        page_rel: str,
        *,
        include_target_user: bool = True,
        include_router_lint: bool = True,
    ) -> tuple[dict[str, Any], httpx.Client]:
        batch: dict[str, Any] = {
            "page": page_rel,
            "router": None,
            "api": [],
            "target_user": None,
        }

        if include_router_lint and self._router_idx < len(self._router_names):
            rname = self._router_names[self._router_idx]
            errs = self._router_errors.get(rname, [])
            status = "FAIL" if errs else "PASS"
            row = {
                "category": "router",
                "id": rname,
                "status": status,
                "detail": errs[0][:300] if errs else "within baseline",
            }
            self._records.append(row)
            batch["router"] = row
            self._router_idx += 1
            self.log.note(f"  lint {rname}: {status}")

        for case in self._next_api_slice(self.api_per_page):
            row, client = self._run_api_case(client, case)
            batch["api"].append(row)
            self._records.append(row)

        if include_target_user and self._target_idx < len(self._target_specs):
            row, client = self._run_target_user_case(client, self._target_specs[self._target_idx])
            batch["target_user"] = row
            self._records.append(row)
            self._target_idx += 1

        pause_between_requests()
        return batch, client

    def drain_remaining_api(self, client: httpx.Client) -> tuple[int, httpx.Client]:
        """Finish API queue (including deferred heavy reads) after upload completes."""
        if self._deferred_api_cases:
            self.log.note(f"Running {len(self._deferred_api_cases)} deferred API checks…")
            self._api_cases.extend(self._deferred_api_cases)
            self._deferred_api_cases = []
        ran = 0
        while self._api_idx < len(self._api_cases):
            case = self._api_cases[self._api_idx]
            self._api_idx += 1
            row, client = self._run_api_case(client, case)
            self._records.append(row)
            ran += 1
            if ran % 10 == 0:
                pause_between_requests()
        return ran, client

    def api_remaining(self) -> int:
        return max(0, len(self._api_cases) - self._api_idx)

    def summary(self) -> dict[str, Any]:
        fails = [r for r in self._records if r.get("status") == "FAIL"]
        api_rows = [r for r in self._records if r.get("category") == "api"]
        router_rows = [r for r in self._records if r.get("category") == "router"]
        tu_rows = [r for r in self._records if r.get("category") == "target_user"]
        return {
            "ok": len(fails) == 0,
            "total_checks": len(self._records),
            "failures": len(fails),
            "api": {
                "total": len(self._api_cases) + len(self._deferred_api_cases),
                "run": len(api_rows),
                "deferred_pending": len(self._deferred_api_cases),
                "fail": sum(1 for r in api_rows if r.get("status") == "FAIL"),
            },
            "router": {
                "total": len(self._router_names),
                "run": len(router_rows),
                "fail": sum(1 for r in router_rows if r.get("status") == "FAIL"),
            },
            "target_user": {
                "total": len(self._target_specs),
                "run": len(tu_rows),
                "fail": sum(1 for r in tu_rows if r.get("status") == "FAIL"),
            },
            "static": self._static,
            "failed_samples": fails[:12],
        }

    def _next_api_slice(self, n: int) -> list[ApiGetCase]:
        if not self._api_cases or self._api_idx >= len(self._api_cases):
            return []
        end = min(self._api_idx + n, len(self._api_cases))
        chunk = self._api_cases[self._api_idx : end]
        self._api_idx = end
        return chunk

    def _run_api_case(self, client: httpx.Client, case: ApiGetCase) -> tuple[dict[str, Any], httpx.Client]:
        row = {"category": "api", "id": case.id, "path": case.path, "status": "PASS", "detail": ""}
        try:
            r, client = api_get_with_retry(
                client,
                case.path,
                timeout=_smoke_timeout(),
                headers={"User-Agent": CHROME_UA},
            )
            assert_acceptable_status(r.status_code, case)
            row["detail"] = f"HTTP {r.status_code}"
        except AssertionError as e:
            row["status"] = "FAIL"
            row["detail"] = str(e)[:400]
        except httpx.HTTPError as e:
            row["status"] = "FAIL"
            row["detail"] = f"transport: {e}"[:400]
        return row, client

    def _run_target_user_case(self, client: httpx.Client, spec: dict[str, Any]) -> tuple[dict[str, Any], httpx.Client]:
        row = {
            "category": "target_user",
            "id": spec["id"],
            "path": spec["path"],
            "status": "PASS",
            "detail": "",
        }
        try:
            kwargs: dict[str, Any] = {}
            if spec.get("params"):
                kwargs["params"] = spec["params"]
            fn: Callable[[httpx.Response], str] = spec["assert_fn"]
            resp, client = api_request_with_retry(client, spec["method"], spec["path"], **kwargs)
            if resp.status_code >= 400:
                row["status"] = "FAIL"
                row["detail"] = f"HTTP {resp.status_code}: {resp.text[:200]}"
            else:
                msg = fn(resp)
                if msg and ("missing" in msg.lower() or "mismatch" in msg.lower()):
                    row["status"] = "FAIL" if not spec.get("soft") else "PASS"
                    row["detail"] = msg
                else:
                    row["detail"] = msg or "ok"
        except Exception as e:
            row["status"] = "FAIL"
            row["detail"] = str(e)[:300]
        return row, client
