"""Strip erroneous m. prefix for module-level functions defined in routers/admin.py."""
from __future__ import annotations

import ast
import re
from pathlib import Path


def main() -> None:
    path = Path(__file__).resolve().parent.parent / "routers" / "admin.py"
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    local = {
        n.name
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for name in sorted(local, key=len, reverse=True):
        pat = r"\bm\." + re.escape(name) + r"\b"
        text, n = re.subn(pat, name, text)
        if n:
            print(f"{name}: {n}")
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
