"""Show that the config-snapshot rotation is exactly ONE ADDED KEY.

Adding a configuration key moves every digest in
``cablp/solvers/_sim1d/config_snapshots.json``, so a rotated snapshot cannot
by itself distinguish "a key was added" from "a value moved". This rebuilds
each snapshot case on the current tree, REMOVES the one added key, and checks
that the digest returns to the value a named base commit recorded. If every
case comes back, the rotation carries no value change.

Usage (from the checkout root, PYTHONPATH set to it)::

    python scripts/k2_dvm_wall_reflection_snapshot_delta.py [--base <rev>]
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

#: The key this branch adds. A params-namespace selector, default-inert.
ADDED_KEY = "neutral_kinetic_dvm_wall_reflection"

#: The staging tip this branch was cut from.
DEFAULT_BASE = "3e7d386"


def base_snapshot(rev):
    """Return the committed snapshot payload at ``rev``."""
    rel = SNAPSHOT_PATH.relative_to(Path(__file__).resolve().parents[1])
    out = subprocess.run(
        ["git", "show", f"{rev}:{rel.as_posix()}"],
        capture_output=True, text=True, check=True,
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
    print(f"added key       : {ADDED_KEY!r}")
    print("=" * 78)
    ok = (
        current["parameter_count"] == base["parameter_count"] + 1
        and current["flag_count"] == base["flag_count"]
    )
    for name, (params, flags) in sorted(config_cases().items()):
        stripped = dict(params)
        value = stripped.pop(ADDED_KEY, "<ABSENT>")
        got = config_digest(stripped, flags)
        want = base["cases"][name]["sha256"]
        same = got == want
        ok = ok and same and value == "specular"
        print(f"{name}")
        print(f"    resolved value            : {value!r}")
        print(f"    digest with the key removed: {got}")
        print(f"    base commit digest         : {want}")
        print(f"    {'MATCH' if same else 'DIFFERS'}")
    print("=" * 78)
    print("ROTATION IS A PURE KEY ADDITION" if ok else "ROTATION CARRIES MORE")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
