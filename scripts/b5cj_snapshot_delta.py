"""Show that the config-snapshot rotation is exactly FOUR ADDED KEYS.

Adding configuration keys moves every digest in
``cablp/solvers/_sim1d/config_snapshots.json``, so a rotated snapshot cannot
by itself distinguish "keys were added" from "a value moved". This rebuilds
each snapshot case on the current tree, REMOVES the four keys this branch
adds, and checks that the digest returns to the value a named base commit
recorded. If every case comes back, the rotation carries no value change.

The four are also checked to resolve to the declared defaults -- the channel
OFF, and the two reflection coefficients at the fluid channel's shipped
``cathode_jet_R_N`` / ``cathode_jet_R_E`` -- so "no value moved" covers the
new keys as well as the old ones.

Usage (from the checkout root, PYTHONPATH set to it)::

    python scripts/b5cj_snapshot_delta.py [--base <rev>]
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

#: The keys this branch adds, with the value each must resolve to. All four
#: are params-namespace and all four are default-inert.
ADDED_KEYS = {
    "neutral_kinetic_dvm_cathode_jet": False,
    "neutral_kinetic_dvm_cathode_jet_R_N": 0.34,
    "neutral_kinetic_dvm_cathode_jet_R_E": 0.18,
    "neutral_kinetic_dvm_cathode_jet_T_launch_eV": None,
}

#: The B3 branch tip this member is stacked on.
DEFAULT_BASE = "fee4568"


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
    print(f"added keys      : {sorted(ADDED_KEYS)}")
    print("=" * 78)
    ok = (
        current["parameter_count"] == base["parameter_count"] + len(ADDED_KEYS)
        and current["flag_count"] == base["flag_count"]
    )
    for name, (params, flags) in sorted(config_cases().items()):
        stripped = dict(params)
        values = {
            key: stripped.pop(key, "<ABSENT>") for key in sorted(ADDED_KEYS)
        }
        got = config_digest(stripped, flags)
        want = base["cases"][name]["sha256"]
        same = got == want
        defaults_ok = all(
            values[key] == expected for key, expected in ADDED_KEYS.items()
        )
        ok = ok and same and defaults_ok
        print(f"{name}")
        for key in sorted(values):
            print(f"    resolved {key} = {values[key]!r}")
        print(f"    resolved to the declared defaults: {defaults_ok}")
        print(f"    digest with the keys removed: {got}")
        print(f"    base commit digest          : {want}")
        print(f"    {'MATCH' if same else 'DIFFERS'}")
    print("=" * 78)
    print("ROTATION IS A PURE KEY ADDITION" if ok else "ROTATION CARRIES MORE")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
