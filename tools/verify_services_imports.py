#!/usr/bin/env python
"""
Verify every module under ``services/`` and every import of it from backend code.

Usage:
  python tools/verify_services_imports.py          # fail on broken imports
  python tools/verify_services_imports.py --report # also print 122-module wiring map

Exit 0 when all imports resolve; exit 1 on missing modules/symbols.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import pathlib
import re
import sys
from collections import defaultdict

_CODEGEN_IMPORT_RE = re.compile(
    r"^\s*from\s+services\.([a-zA-Z_][\w]*)\s+import\s+",
    re.MULTILINE,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SKIP_DIRS = {
    "venv",
    ".venv",
    ".venv-lockregen",
    "__pycache__",
    ".pytest_cache",
    ".git",
    "node_modules",
    "frontend",
    "tests",
}
SCAN_ROOTS = [
    "routers",
    "api",
    "core",
    "stages",
    "jobs",
    "scripts",
    "tools",
    "app.py",
    "worker.py",
    "telemetry_trill.py",
]
SERVICES_DIR = ROOT / "services"
RUNTIME_PREFIXES = ("routers.", "app", "worker", "stages.", "jobs.")


def _should_skip(path: pathlib.Path) -> bool:
    return any(part in SKIP_DIRS for part in path.parts)


def _iter_py_files(scan: str) -> list[pathlib.Path]:
    p = ROOT / scan
    if scan.endswith(".py"):
        return [p] if p.exists() else []
    if not p.is_dir():
        return []
    return [f for f in p.rglob("*.py") if not _should_skip(f)]


def _service_modules() -> dict[str, pathlib.Path]:
    mods: dict[str, pathlib.Path] = {}
    for py in sorted(SERVICES_DIR.glob("*.py")):
        if py.name == "__init__.py":
            continue
        mods[f"services.{py.stem}"] = py
    return mods


def _resolve_import(node: ast.ImportFrom) -> list[tuple[str, str]]:
    """Return list of (module, symbol) referenced by an import node."""
    out: list[tuple[str, str]] = []
    if not node.module:
        return out
    if node.module == "services":
        for alias in node.names:
            if alias.name == "*":
                continue
            out.append((f"services.{alias.name}", alias.name))
        return out
    if not node.module.startswith("services."):
        return out
    for alias in node.names:
        if alias.name == "*":
            out.append((node.module, "*"))
        else:
            out.append((node.module, alias.name))
    return out


def _is_submodule_import(mod: str, sym: str, all_mods: dict[str, pathlib.Path]) -> bool:
    """``from services import foo`` imports the ``services.foo`` module, not attr ``foo``."""
    return mod in all_mods and mod == f"services.{sym}"


def _collect_codegen_references() -> dict[str, set[str]]:
    """Router codegen scripts embed ``from services...`` inside string templates."""
    refs: dict[str, set[str]] = defaultdict(set)
    tools_dir = ROOT / "tools"
    if not tools_dir.is_dir():
        return refs
    for py in tools_dir.glob("*.py"):
        if py.name == "verify_services_imports.py":
            continue
        try:
            text = py.read_text(encoding="utf-8")
        except Exception:
            continue
        for match in _CODEGEN_IMPORT_RE.finditer(text):
            refs[f"services.{match.group(1)}"].add(f"codegen:{py.name}")
    return refs


def _collect_references() -> dict[str, set[str]]:
    refs: dict[str, set[str]] = defaultdict(set)
    for scan in [*SCAN_ROOTS, "services"]:
        for py in _iter_py_files(scan):
            if py.name == "verify_services_imports.py":
                continue
            try:
                tree = ast.parse(py.read_text(encoding="utf-8"))
            except Exception:
                continue
            src = str(py.relative_to(ROOT))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom):
                    continue
                for mod, _sym in _resolve_import(node):
                    refs[mod].add(src)
    for mod, sources in _collect_codegen_references().items():
        refs[mod].update(sources)
    return refs


def _audit_imports(all_mods: dict[str, pathlib.Path]) -> tuple[list, list]:
    fail_mod: list[tuple[str, ...]] = []
    fail_sym: list[tuple[str, ...]] = []

    for scan in SCAN_ROOTS:
        for py in _iter_py_files(scan):
            if py.name == "verify_services_imports.py":
                continue
            try:
                tree = ast.parse(py.read_text(encoding="utf-8"))
            except Exception:
                continue
            rel = str(py.relative_to(ROOT))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom):
                    continue
                for mod, sym in _resolve_import(node):
                    if mod not in all_mods:
                        fail_mod.append((rel, mod))
                        continue
                    try:
                        m = importlib.import_module(mod)
                    except Exception as exc:
                        fail_mod.append((rel, mod, repr(exc)))
                        continue
                    if sym == "*" or _is_submodule_import(mod, sym, all_mods):
                        continue
                    if not hasattr(m, sym):
                        fail_sym.append((rel, mod, sym))
    return fail_mod, fail_sym


def _norm_ref(path: str) -> str:
    return path.replace("\\", "/")


def _classify_module(mod: str, refs: set[str]) -> str:
    if not refs:
        return "unwired"
    norm = {_norm_ref(r) for r in refs}
    external = [r for r in norm if not r.startswith("services/")]
    if not external:
        return "internal"
    runtime = [
        r
        for r in external
        if r.startswith(RUNTIME_PREFIXES) or r in ("app.py", "worker.py", "telemetry_trill.py")
    ]
    if runtime:
        return "runtime"
    if any(
        r.startswith("tools/")
        or r.startswith("scripts/")
        or r.startswith("jobs/")
        or r.startswith("codegen:")
        for r in external
    ):
        return "tooling"
    if external:
        return "other"
    return "internal"


def _print_report(all_mods: dict[str, pathlib.Path], refs: dict[str, set[str]]) -> None:
    import_fail: list[str] = []
    for mod in sorted(all_mods):
        try:
            importlib.import_module(mod)
        except Exception as exc:
            import_fail.append(f"{mod}: {type(exc).__name__}: {exc}")

    buckets: dict[str, list[str]] = defaultdict(list)
    for mod in sorted(all_mods):
        buckets[_classify_module(mod, refs.get(mod, set()))].append(mod)

    print(f"SERVICE_MODULES {len(all_mods)}")
    print(f"IMPORTABLE {len(all_mods) - len(import_fail)}")
    print(f"IMPORT_ERRORS {len(import_fail)}")
    for line in import_fail:
        print(f"  {line}")

    for label in ("runtime", "internal", "tooling", "unwired", "other"):
        items = buckets.get(label, [])
        print(f"{label.upper()} {len(items)}")
        for mod in items:
            sample = sorted(refs.get(mod, set()))[:3]
            extra = f"  <- {', '.join(sample)}" if sample else ""
            print(f"  {mod}{extra}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify services/ import wiring")
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print full 122-module wiring map after validation",
    )
    args = parser.parse_args()

    all_mods = _service_modules()
    fail_mod, fail_sym = _audit_imports(all_mods)

    if fail_mod:
        print("MODULE_FAILURES", len(fail_mod))
        for row in fail_mod:
            print("MOD |", " | ".join(row))
    if fail_sym:
        print("SYMBOL_FAILURES", len(fail_sym))
        for row in fail_sym:
            print("SYM |", " | ".join(row))

    if args.report:
        print()
        _print_report(all_mods, _collect_references())

    if not fail_mod and not fail_sym:
        if not args.report:
            print(f"OK: all services imports resolve ({len(all_mods)} modules)")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
