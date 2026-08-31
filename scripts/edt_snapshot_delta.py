"""Show that the config-snapshot rotation is exactly THREE ADDED KEYS.

Adding configuration keys moves every digest in
``cablp/solvers/_sim1d/config_snapshots.json``, so a rotated snapshot cannot by
itself distinguish "keys were added" from "a value moved". This rebuilds each
snapshot case on the current tree, REMOVES the three keys this branch adds --
two params and one flag, so BOTH namespaces are stripped -- and checks that the
digest returns to the value a named base commit recorded. If every case comes
back, the rotation carries no value change.

The three are also checked to resolve to their declared defaults, so "no value
moved" covers the new keys as well as the old ones.

Usage (from the checkout root, PYTHONPATH set to it)::

    python scripts/edt_snapshot_delta.py [--base <rev>]
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

#: The params-namespace keys this branch adds, with the value each must
#: resolve to. Both are the DECLARED endpoints of a bracket, and both are
#: refused at anything but these values while the flag is off.
ADDED_PARAMS = {
    "electron_drift_charge_death": "cell_1",
    "electron_drift_anode_handshake": "sheath_row_closes_all",
}

#: The flags-namespace key this branch adds. Default off, bit-exact off.
ADDED_FLAGS = {
    "electron_drift_transport": False,
}

#: The staging tip this branch is based on (rebased 2026-08-31 onto the B6
#: merge plus its identity-only golden rotation and provenance entry).
DEFAULT_BASE = "056a733"


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
    print(f"added params    : {sorted(ADDED_PARAMS)}")
    print(f"added flags     : {sorted(ADDED_FLAGS)}")
    print("=" * 78)
    ok = (
        current["parameter_count"] == base["parameter_count"] + len(ADDED_PARAMS)
        and current["flag_count"] == base["flag_count"] + len(ADDED_FLAGS)
    )
    for name, (params, flags) in sorted(config_cases().items()):
        stripped_params = dict(params)
        stripped_flags = dict(flags)
        values = {
            key: stripped_params.pop(key, "<ABSENT>")
            for key in sorted(ADDED_PARAMS)
        }
        values.update(
            {
                key: stripped_flags.pop(key, "<ABSENT>")
                for key in sorted(ADDED_FLAGS)
            }
        )
        got = config_digest(stripped_params, stripped_flags)
        want = base["cases"][name]["sha256"]
        same = got == want
        expected = {**ADDED_PARAMS, **ADDED_FLAGS}
        defaults_ok = all(
            values[key] == value for key, value in expected.items()
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
