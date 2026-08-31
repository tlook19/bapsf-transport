"""Show that the config-snapshot rotation is exactly ONE ADDED KEY.

The B6 twin of ``b4aj_snapshot_delta.py`` / ``b5cj_snapshot_delta.py``, and the
same statement: adding configuration keys moves every digest in
``cablp/solvers/_sim1d/config_snapshots.json``, so a rotated snapshot cannot by
itself distinguish "a key was added" from "a value moved". This rebuilds each
snapshot case on the current tree, REMOVES the key this branch adds, and checks
that the digest returns to the value a named base commit recorded. If every
case comes back, the rotation carries no value change.

The added key is checked to resolve to its declared default -- the channel OFF
-- so "no value moved" covers the new key as well as the old ones.

B6's key is in the FLAGS namespace, not params: it is the kinetic twin of the
fluid ``neutral_baffles`` FLAG and is filed beside it, which is where
``core/config.py`` owns that pair. The B4/B5 jets sit in params because their
own fluid twins do. That is why the counts checked below move the FLAG count
and leave the parameter count alone -- the mirror image of B4's and B5's
rotations, and the one thing a reader comparing the three files will notice.

Usage (from the checkout root, PYTHONPATH set to it)::

    python scripts/b6bf_snapshot_delta.py [--base <rev>]
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

#: The key this branch adds, with the value it must resolve to. FLAGS
#: namespace, default-inert.
ADDED_FLAGS = {"neutral_kinetic_dvm_baffles": False}

#: The branch base: ``campaign`` == ``agent-staging`` == ``origin/campaign``.
DEFAULT_BASE = "75a2fa1"


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
    print(f"added flags     : {sorted(ADDED_FLAGS)}")
    print("=" * 78)
    ok = (
        current["parameter_count"] == base["parameter_count"]
        and current["flag_count"] == base["flag_count"] + len(ADDED_FLAGS)
    )
    for name, (params, flags) in sorted(config_cases().items()):
        stripped = dict(flags)
        values = {
            key: stripped.pop(key, "<ABSENT>") for key in sorted(ADDED_FLAGS)
        }
        got = config_digest(params, stripped)
        want = base["cases"][name]["sha256"]
        same = got == want
        defaults_ok = all(
            values[key] == expected for key, expected in ADDED_FLAGS.items()
        )
        ok = ok and same and defaults_ok
        print(f"{name}")
        for key in sorted(values):
            print(f"    resolved {key} = {values[key]!r}")
        print(f"    resolved to the declared default: {defaults_ok}")
        print(f"    digest with the key removed: {got}")
        print(f"    base commit digest         : {want}")
        print(f"    {'MATCH' if same else 'DIFFERS'}")
    print("=" * 78)
    print("ROTATION IS A PURE KEY ADDITION" if ok else "ROTATION CARRIES MORE")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
