"""Two-tier AST-identity gate for this pass, base-vs-worktree, from git.

The standing rule for any comment/docstring-only pass: parse before and
after, strip docstring nodes, compare ``ast.dump``. Tier A strips docstring
nodes only; tier B additionally normalizes EVERY string constant, so a pass
that legitimately rewrites user-facing strings still catches a deleted dict
line, a changed number, a renamed identifier or a dropped argument.

This variant reads the BEFORE side straight out of ``git show <base>:<path>``
rather than from a captured sidecar, so it cannot be run against a stale
capture. It also reports, for any file that is NOT tier-A identical, the
exact set of differing constant VALUES -- which is what makes a deliberate
one-value change auditable instead of merely flagged.

Files are declared with the identity they are claimed to have:

* ``prose`` -- tier A AND tier B must be identical (docstrings/comments only);
* ``value`` -- tier A is EXPECTED to differ, and the difference must reduce to
  the declared ``(old, new)`` constant swap and nothing else.

Usage (from the worktree root, PYTHONPATH set to it)::

    python scripts/dacc_ast_identity.py [--base <rev>]
"""

import argparse
import ast
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_BASE = "3e32967"

#: Files this pass edits, with the identity each is claimed to have.
PROSE_FILES = (
    "cablp/solvers/_sim1d/core/validation.py",
    "cablp/solvers/_sim1d/solver.py",
    "cablp/solvers/_sim1d/physics/kinetic_dvm.py",
    "scripts/verify_sim1d_k2_dvm.py",
)

#: The one file carrying a deliberate value change, and the swap it carries.
VALUE_FILES = {
    "cablp/solvers/_sim1d/core/config.py": (1.0, 0.40),
}


def _strip(tree, normalize_strings):
    """Return ``tree`` with docstrings removed (and strings blanked in B)."""
    for node in ast.walk(tree):
        if not isinstance(
            node,
            (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
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
                node.value = ""
    return tree


def _dump(src, normalize_strings):
    tree = _strip(ast.parse(src), normalize_strings)
    return ast.dump(tree, annotate_fields=True, include_attributes=False)


def _digests(src):
    return tuple(
        hashlib.sha256(_dump(src, n).encode()).hexdigest()
        for n in (False, True)
    )


def _constants(src):
    """Return the multiset of non-docstring constant values, as a counter."""
    tree = _strip(ast.parse(src), normalize_strings=False)
    counts = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant):
            key = (type(node.value).__name__, repr(node.value))
            counts[key] = counts.get(key, 0) + 1
    return counts


def _base_source(rev, rel):
    out = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{rev}:{rel}"],
        capture_output=True, text=True, check=True,
    )
    return out.stdout


def _const_delta(before, after):
    """Return the constants that appear more/fewer times after the change."""
    keys = set(before) | set(after)
    delta = {}
    for key in keys:
        d = after.get(key, 0) - before.get(key, 0)
        if d:
            delta[key] = d
    return delta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=DEFAULT_BASE)
    args = ap.parse_args(argv)

    print(f"AST identity, base {args.base} vs worktree")
    print("=" * 78)
    ok = True

    for rel in PROSE_FILES:
        before = _base_source(args.base, rel)
        after = (ROOT / rel).read_text()
        b_a, b_b = _digests(before)
        a_a, a_b = _digests(after)
        same_a, same_b = b_a == a_a, b_b == a_b
        ok = ok and same_a and same_b
        print(f"[prose] {rel}")
        print(f"    tierA {'IDENTICAL' if same_a else 'DIFFERS'}  {a_a}")
        print(f"    tierB {'IDENTICAL' if same_b else 'DIFFERS'}  {a_b}")
        if not same_a:
            print(f"    constant delta: {_const_delta(_constants(before), _constants(after))}")

    for rel, (old, new) in VALUE_FILES.items():
        before = _base_source(args.base, rel)
        after = (ROOT / rel).read_text()
        b_a, b_b = _digests(before)
        a_a, a_b = _digests(after)
        delta = _const_delta(_constants(before), _constants(after))
        expected = {
            ("float", repr(old)): -1,
            ("float", repr(new)): +1,
        }
        exact = delta == expected
        ok = ok and exact
        print(f"[value] {rel}  (declared swap {old!r} -> {new!r})")
        print(f"    tierA {'IDENTICAL' if b_a == a_a else 'DIFFERS'} (a differ is EXPECTED here)")
        print(f"    tierB {'IDENTICAL' if b_b == a_b else 'DIFFERS'} (a differ is EXPECTED here)")
        print(f"    constant delta observed: {delta}")
        print(f"    constant delta expected: {expected}")
        print(f"    reduces to the declared swap and nothing else: {exact}")

    print("=" * 78)
    print("AST IDENTITY GATE: PASS" if ok else "AST IDENTITY GATE: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
