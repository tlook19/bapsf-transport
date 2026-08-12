"""Two-tier AST identity for the R1 documentation-only round.

The standing gate for any comment/docstring-only pass: parse before and after,
strip docstring nodes, compare ``ast.dump``. Tier B additionally normalizes
EVERY string constant, so it still catches a deleted dict line, a changed
number, a renamed identifier or a dropped argument even where the pass
legitimately rewrites a user-facing string.

Tier A passing already proves the pass touched only docstrings and comments
(comments never reach the AST at all). Tier B is the belt-and-braces check that
nothing structural moved under cover of the string edits. Tier B's one blind
spot is the CONTENT of the changed strings, which has to be read by eye -- the
report lists every string constant that moved so the reviewer can do exactly
that.

Usage (from <checkout>/cablp, with PYTHONPATH set to that same cablp):
    python scripts/r1vb_ast_identity.py <base-ref> <path> [<path> ...]
"""

import argparse
import ast
import subprocess
import sys


def _strip_docstrings(tree):
    """Return ``tree`` with every docstring node removed, in place."""
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                   ast.AsyncFunctionDef)
        ):
            continue
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:]
            if not node.body:
                node.body = [ast.Pass()]
    return tree


def _normalize_strings(tree):
    """Return ``tree`` with every string constant replaced by one sentinel."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = "<str>"
    return tree


def _strings(tree):
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


def dumps(source, tier):
    tree = _strip_docstrings(ast.parse(source))
    if tier == "B":
        tree = _normalize_strings(tree)
    return ast.dump(tree)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("base_ref")
    p.add_argument("paths", nargs="+")
    args = p.parse_args(argv)

    failures = []
    for path in args.paths:
        before = subprocess.run(
            ["git", "show", f"{args.base_ref}:cablp/{path}"],
            capture_output=True, text=True, check=True,
        ).stdout
        with open(path, encoding="utf-8") as handle:
            after = handle.read()

        print(f"{path}")
        if before == after:
            print("  UNCHANGED (byte-identical to the base ref)")
            print()
            continue
        for tier in ("A", "B"):
            ok = dumps(before, tier) == dumps(after, tier)
            label = ("docstrings stripped" if tier == "A"
                     else "docstrings stripped + every string normalized")
            print(f"  tier {tier} ({label}): "
                  f"{'IDENTICAL' if ok else 'DIFFERS'}")
            if not ok:
                failures.append(f"{path}: tier {tier} differs")

        # Tier B's blind spot, made visible: the string constants that moved.
        b_before, b_after = _strings(
            _strip_docstrings(ast.parse(before))
        ), _strings(_strip_docstrings(ast.parse(after)))
        moved = [s for s in b_after if s not in b_before]
        dropped = [s for s in b_before if s not in b_after]
        print(f"  non-docstring string constants added: {len(moved)}, "
              f"removed: {len(dropped)}")
        for s in moved:
            print(f"    + {s!r}")
        for s in dropped:
            print(f"    - {s!r}")
        print()

    if failures:
        print(f"AST IDENTITY: FAIL -- {failures}")
        return 1
    print("AST IDENTITY: PASS -- every touched module is structurally "
          "identical at both tiers, so the round changed docstrings and "
          "comments only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
