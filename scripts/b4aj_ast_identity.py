"""[B4] AST identity gate for the docstring-only review-fix pass.

The standing gate for a comment/docstring-only pass (CLAUDE.md, AGENTS.md):
parse before and after, strip docstring nodes, compare ``ast.dump``.

TIER A strips docstring nodes only -- it catches a deleted dict line, a changed
number, a renamed identifier or a dropped argument. TIER B additionally
normalizes EVERY string constant, so it still catches all of that on a pass
that legitimately rewrites user-facing strings while being blind to the string
contents themselves; tier B's blind spot is why the changed strings are also
reviewed individually.

Usage (from the checkout root, PYTHONPATH set to it)::

    python scripts/b4aj_ast_identity.py <rev> <path> [<path> ...]
"""

import ast
import subprocess
import sys
from pathlib import Path


def _strip(tree, normalize_strings):
    """Return ``tree`` with docstrings removed, optionally string-normalized."""
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:] or [ast.Pass()]
    if normalize_strings:
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                node.value = "<str>"
    return tree


def dump(source, normalize_strings):
    return ast.dump(
        _strip(ast.parse(source), normalize_strings), annotate_fields=True
    )


def main(argv):
    rev, paths = argv[0], argv[1:]
    ok = True
    for rel in paths:
        before = subprocess.run(
            ["git", "show", f"{rev}:{rel}"],
            capture_output=True, text=True, check=True,
        ).stdout
        after = Path(rel).read_text()
        a = dump(before, False) == dump(after, False)
        b = dump(before, True) == dump(after, True)
        ok = ok and a and b
        print(f"{rel}")
        print(f"    tier A (docstrings stripped)              : "
              f"{'IDENTICAL' if a else 'DIFFERS'}")
        print(f"    tier B (+ every string constant normalized): "
              f"{'IDENTICAL' if b else 'DIFFERS'}")
        print(f"    bytes {len(before)} -> {len(after)}")
    print("=" * 78)
    print(
        f"AST IDENTITY vs {rev}: "
        + ("PASS -- the pass is docstring-only" if ok else "FAIL")
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
