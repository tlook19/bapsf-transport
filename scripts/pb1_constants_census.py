"""[perf-batch-1] Changed-constants census: every numeric literal, base vs tip.

A hoist, a cache and a skip add no numbers. This parses each changed file at
the base commit and at the working tree, collects every numeric Constant node
in the AST as a MULTISET (so a literal that merely moved is not flagged, and a
literal that changed value or count is), and reports the difference.

    python scripts/pb1_constants_census.py 75a2fa1

Numbers reported as ADDED must each be accounted for in the report.
"""

import argparse
import ast
import collections
import subprocess
import sys


def literals(source):
    """Return a multiset of every numeric literal in ``source``."""
    counts = collections.Counter()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Constant) and isinstance(
            node.value, (int, float, complex)
        ) and not isinstance(node.value, bool):
            counts[repr(node.value)] += 1
    return counts


def changed_files(base):
    out = subprocess.run(
        ["git", "diff", "--name-only", f"{base}..HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout
    return [p for p in out.split() if p.endswith(".py")]


def show(base, path):
    out = subprocess.run(
        ["git", "show", f"{base}:{path}"], capture_output=True, text=True
    )
    return out.stdout if out.returncode == 0 else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    args = ap.parse_args(argv)

    added_total = collections.Counter()
    removed_total = collections.Counter()
    for path in changed_files(args.base):
        old = show(args.base, path)
        new = show("HEAD", path)
        old_counts = literals(old) if old is not None else collections.Counter()
        new_counts = literals(new) if new is not None else collections.Counter()
        added = new_counts - old_counts
        removed = old_counts - new_counts
        status = "NEW FILE" if old is None else "changed"
        print(f"--- {path} ({status})")
        if not added and not removed:
            print("    no numeric literal added or removed")
        for value, n in sorted(added.items()):
            print(f"    +{n}x {value}")
        for value, n in sorted(removed.items()):
            print(f"    -{n}x {value}")
        if old is not None:
            added_total.update(added)
            removed_total.update(removed)

    print()
    print("=== census over PRE-EXISTING files (new instrument files excluded) ===")
    if not added_total and not removed_total:
        print("NO NUMERIC CONSTANT MOVED")
        return 0
    for value, n in sorted(added_total.items()):
        print(f"  ADDED    {n}x {value}")
    for value, n in sorted(removed_total.items()):
        print(f"  REMOVED  {n}x {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
