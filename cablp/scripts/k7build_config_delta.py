"""K7 config-snapshot delta: exactly the keys this change adds, and nothing else.

Resolves every config-complete campaign case on BOTH checkouts -- this one and
a reference one named on the command line (the base the branch was cut from) --
and diffs the resolved ``input_dict`` / ``input_flags`` key by key. Adding a
config key legitimately changes every snapshot digest, which is why the digest
gate alone cannot tell "three new keys" from "three new keys and a moved
default". This does.

The reference side is loaded by running the reference checkout's own
``audit_sim1d_configs`` in a SUBPROCESS with its own ``PYTHONPATH``: importing
two copies of ``cablp`` into one interpreter is not possible, and shelling out
keeps each side resolving its own defaults with its own code.

Usage (from <checkout>/cablp, with PYTHONPATH set to that same cablp):
    python scripts/k7build_config_delta.py --reference /path/to/other/cablp
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from audit_sim1d_configs import config_cases

REPORT = Path(__file__).resolve().parent / "k7build_config_delta.txt"

# The keys K7 adds. Pre-registered here so the gate compares against a stated
# expectation rather than reporting whatever it happens to find.
EXPECTED_ADDED_PARAMS = {
    "heating_anomalous_tail_cathode_boundary",
    "heating_anomalous_tail_energy_keying",
    "heating_anomalous_tail_phi_c_fraction",
}
EXPECTED_ADDED_FLAGS = set()

_DUMP = """
import json, sys
sys.path.insert(0, sys.argv[1])
from audit_sim1d_configs import config_cases
print(json.dumps({
    name: {"params": params, "flags": flags}
    for name, (params, flags) in config_cases().items()
}))
"""


def reference_cases(cablp_dir):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(cablp_dir)
    out = subprocess.run(
        [sys.executable, "-c", _DUMP, str(Path(cablp_dir) / "scripts")],
        cwd=str(cablp_dir), env=env, capture_output=True, text=True,
        check=True,
    )
    return json.loads(out.stdout)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--reference",
        default="/Users/tlook/bapsf/bapsf-transport/cablp",
        help="the reference checkout's cablp/ directory",
    )
    args = ap.parse_args(argv)

    out = []

    def P(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        out.append(s)

    ref = reference_cases(Path(args.reference).resolve())
    mine = {
        name: {"params": params, "flags": flags}
        for name, (params, flags) in config_cases().items()
    }

    P("# k7build_config_delta -- resolved-config key delta vs the base checkout")
    P(f"# reference: {args.reference}")
    P(f"# expected added params: {sorted(EXPECTED_ADDED_PARAMS)}")
    P(f"# expected added flags : {sorted(EXPECTED_ADDED_FLAGS)}")
    P("")

    failures = []
    assert set(ref) == set(mine), (sorted(ref), sorted(mine))
    for name in sorted(mine):
        for kind, expected_added in (
            ("params", EXPECTED_ADDED_PARAMS),
            ("flags", EXPECTED_ADDED_FLAGS),
        ):
            a = ref[name][kind]
            b = mine[name][kind]
            added = set(b) - set(a)
            removed = set(a) - set(b)
            changed = {
                k: (a[k], b[k]) for k in set(a) & set(b) if a[k] != b[k]
            }
            P(f"== {name} [{kind}]  base={len(a)} branch={len(b)}")
            P(f"   added   : {sorted(added)}")
            P(f"   removed : {sorted(removed)}")
            P(f"   changed : "
              + (json.dumps(changed, sort_keys=True) if changed else "{}"))
            P("   new-key values: "
              + json.dumps({k: b[k] for k in sorted(added)}, sort_keys=True))
            if added != expected_added or removed or changed:
                failures.append((name, kind, sorted(added), sorted(removed),
                                 changed))
        P("")

    if failures:
        P(f"CONFIG DELTA FAILED: {len(failures)} case/kind(s) off the "
          "pre-registered delta")
        for f in failures:
            P(f"   {f}")
    else:
        P("CONFIG DELTA PASSED: every case adds exactly the pre-registered "
          "keys, removes nothing, and moves no existing default.")

    REPORT.write_text("\n".join(out) + "\n")
    P(f"# wrote {REPORT}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
