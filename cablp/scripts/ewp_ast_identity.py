"""AST-identity gate for the docstring/comment-only correction pass.

The standing rule for any comment/docstring-only pass: parse before and after,
strip docstring nodes, compare ``ast.dump``. A docstring edit that silently
deleted a dict line, changed a number, renamed an identifier or dropped an
argument shows up here and nowhere else.

Tier A strips docstring nodes only. Tier B additionally normalizes EVERY string
constant, which is what a pass that legitimately rewrites user-facing strings
(error messages, argparse help) needs. This pass is docstrings only, so tier A
must already be identical; tier B is reported as corroboration.

Usage (from <checkout>/cablp, PYTHONPATH set to that same cablp):

    python scripts/ewp_ast_identity.py --capture BEFORE.json  FILE [FILE ...]
    python scripts/ewp_ast_identity.py --verify  BEFORE.json  FILE [FILE ...]
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path


def _strip_docstrings(tree, normalize_strings=False):
    """Return a copy of ``tree`` with docstring nodes removed."""
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
            node.body = body[1:]
            # A body that was nothing but a docstring still needs a statement.
            if not node.body:
                node.body = [ast.Pass()]
    if normalize_strings:
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                node.value = ""
    return tree


def digests(path):
    """Return (tier_a, tier_b) ast.dump digests for one source file."""
    src = Path(path).read_text()
    out = []
    for normalize in (False, True):
        tree = _strip_docstrings(ast.parse(src), normalize_strings=normalize)
        dump = ast.dump(tree, annotate_fields=True, include_attributes=False)
        out.append(hashlib.sha256(dump.encode()).hexdigest())
    return tuple(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--capture", metavar="JSON", help="write the before-state")
    g.add_argument("--verify", metavar="JSON", help="compare against it")
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()

    current = {str(Path(f)): digests(f) for f in args.files}

    if args.capture:
        Path(args.capture).write_text(json.dumps(current, indent=2, sort_keys=True))
        print(f"captured {len(current)} file(s) -> {args.capture}")
        for name, (a, b) in sorted(current.items()):
            print(f"  {name}\n    tierA {a}\n    tierB {b}")
        return

    before = json.loads(Path(args.verify).read_text())
    ok = True
    print("AST-IDENTITY GATE (docstring/comment-only pass)")
    for name in sorted(current):
        a_now, b_now = current[name]
        a_was, b_was = before.get(name, (None, None))
        a_ok = a_now == a_was
        b_ok = b_now == b_was
        ok &= a_ok
        print(f"  {name}")
        print(f"    tierA (docstrings stripped)      "
              f"{'IDENTICAL' if a_ok else 'CHANGED'}  {a_now}")
        print(f"    tierB (+string constants nulled) "
              f"{'IDENTICAL' if b_ok else 'CHANGED'}  {b_now}")
        if not a_ok:
            print(f"    was tierA {a_was}")
    missing = sorted(set(before) - set(current))
    if missing:
        ok = False
        print(f"  MISSING from this run: {missing}")
    print()
    print("RESULT:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
