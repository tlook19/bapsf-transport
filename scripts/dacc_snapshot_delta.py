"""Show that the config-snapshot rotation is exactly ONE CHANGED VALUE.

Changing a default moves every case digest in
``cablp/solvers/_sim1d/config_snapshots.json``, so a rotated snapshot cannot
by itself distinguish "one value moved" from "a key was added, removed or
renamed as well". This rebuilds each snapshot case on the current tree, puts
``neutral_kinetic_dvm_accommodation`` BACK to the base default, and checks
that every digest returns to the value a named base commit recorded. If every
case comes back, the rotation carries nothing but that one value.

The key COUNTS are checked against the base too, which is the complementary
half: a restored digest proves no value moved, an unchanged
``parameter_count`` / ``flag_count`` proves no key came or went.

Usage (from the worktree root, PYTHONPATH set to it)::

    python scripts/dacc_snapshot_delta.py [--base <rev>]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from audit_sim1d_configs import (  # noqa: E402
    SNAPSHOT_PATH,
    config_cases,
    config_digest,
)

#: The one key this branch re-values, with its base and adopted defaults.
KEY = "neutral_kinetic_dvm_accommodation"
BASE_VALUE = 1.0
ADOPTED_VALUE = 0.40

#: The staging tip this branch is built on.
DEFAULT_BASE = "3e32967"


def base_snapshot(rev):
    """Return the committed snapshot payload at ``rev``."""
    rel = SNAPSHOT_PATH.relative_to(Path(__file__).resolve().parents[1])
    out = subprocess.run(
        ["git", "show", f"{rev}:{rel.as_posix()}"],
        capture_output=True, text=True, check=True,
        cwd=str(Path(__file__).resolve().parents[1]),
    ).stdout
    return json.loads(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=DEFAULT_BASE)
    args = parser.parse_args(argv)
    base = base_snapshot(args.base)
    current = json.loads(SNAPSHOT_PATH.read_text())
    print(f"base {args.base}: parameters={base['parameter_count']}, "
          f"flags={base['flag_count']}")
    print(f"branch          : parameters={current['parameter_count']}, "
          f"flags={current['flag_count']}")
    print(f"re-valued key   : {KEY}  {BASE_VALUE!r} -> {ADOPTED_VALUE!r}")
    print("=" * 78)
    counts_ok = (
        current["parameter_count"] == base["parameter_count"]
        and current["flag_count"] == base["flag_count"]
    )
    print(f"key counts unchanged (no key added or removed): {counts_ok}")
    print("=" * 78)
    ok = counts_ok
    for name, (params, flags) in sorted(config_cases().items()):
        restored = dict(params)
        got_now = restored.get(KEY, "<ABSENT>")
        restored[KEY] = BASE_VALUE
        got = config_digest(restored, flags)
        want = base["cases"][name]["sha256"]
        same = got == want
        adopted_ok = got_now == ADOPTED_VALUE
        ok = ok and same and adopted_ok
        print(f"{name}")
        print(f"    resolved {KEY} = {got_now!r}")
        print(f"    resolves to the adopted default: {adopted_ok}")
        print(f"    digest with the key restored to {BASE_VALUE!r}: {got}")
        print(f"    base commit digest                    : {want}")
        print(f"    {'MATCH' if same else 'DIFFERS'}")
        print(f"    (branch digest as committed           : "
              f"{current['cases'][name]['sha256']})")
    print("=" * 78)
    print(
        "ROTATION IS A PURE VALUE CHANGE" if ok else "ROTATION CARRIES MORE"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
