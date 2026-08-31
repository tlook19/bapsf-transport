"""[perf-batch-1] Tier-A/B AST identity of one file against a committed revision.

Used to prove that a post-gate edit was comment- or docstring-only, so the
gate transcripts taken before it still apply. Comments never reach the AST at
all, so a tier-A match (docstring nodes stripped) proves a comment-only edit;
tier B additionally normalizes every string constant and would still catch a
deleted dict line, a changed number, a renamed identifier or a dropped
argument.

    python scripts/pb1_ast_identity.py <rev> <path> [<path> ...]
"""

import argparse
import ast
import subprocess
import sys


def _strip_docstrings(tree):
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
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
    return tree


def _normalize_strings(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = "<str>"
    return tree


def dumps(source):
    tier_a = _strip_docstrings(ast.parse(source))
    a = ast.dump(tier_a)
    tier_b = _normalize_strings(_strip_docstrings(ast.parse(source)))
    return a, ast.dump(tier_b)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("rev")
    ap.add_argument("paths", nargs="+")
    args = ap.parse_args(argv)

    failures = []
    for path in args.paths:
        old = subprocess.run(
            ["git", "show", f"{args.rev}:{path}"],
            capture_output=True, text=True, check=True,
        ).stdout
        with open(path) as handle:
            new = handle.read()
        old_a, old_b = dumps(old)
        new_a, new_b = dumps(new)
        tier_a = old_a == new_a
        tier_b = old_b == new_b
        print(f"{path}: tier A {'identical' if tier_a else 'MOVED'}, "
              f"tier B {'identical' if tier_b else 'MOVED'}")
        if not tier_a:
            failures.append(path)
    if failures:
        print("AST IDENTITY FAIL: " + ", ".join(failures))
        return 1
    print(f"AST IDENTITY OK vs {args.rev}: the edit reached no executable node")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
