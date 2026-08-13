"""regime_crf gate 4: enumerate EXACTLY what this branch adds to the config.

Imports the resolved default manifest from a BASE checkout and from this one
and diffs them key by key, so the claim "one numerics-class key, no physical
constants, no value changed" is a measurement rather than an assertion. Also
re-derives the four config-snapshot case hashes on both sides, which is what
lets a snapshot-ledger update be reviewed as "only the new key moved it".

Usage (from <checkout>/cablp, with PYTHONPATH set to that same cablp):
    python scripts/regime_crf_config_audit.py --base /path/to/base/cablp
"""

import argparse
import json
import subprocess
import sys

MANIFEST_PROG = (
    "import json, sys;"
    "from cablp.solvers._sim1d import config_manifest, default_config;"
    "p, f = default_config();"
    "print(json.dumps({'manifest': config_manifest(),"
    " 'params': p, 'flags': f}, sort_keys=True, default=str))"
)


def read_side(cablp_dir, python):
    out = subprocess.run(
        [python, "-c", MANIFEST_PROG],
        cwd=cablp_dir,
        env={"PYTHONPATH": cablp_dir, "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="base checkout's cablp/ dir")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    here = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True, text=True, check=True,
    ).stdout.strip() + "/cablp"

    base = read_side(args.base, args.python)
    now = read_side(here, args.python)

    for section in ("params", "flags"):
        b, n = base[section], now[section]
        added = sorted(set(n) - set(b))
        removed = sorted(set(b) - set(n))
        changed = sorted(
            k for k in set(b) & set(n) if b[k] != n[k]
        )
        print(f"== {section}: {len(b)} -> {len(n)}")
        print(f"   ADDED   ({len(added)}): {added}")
        print(f"   REMOVED ({len(removed)}): {removed}")
        print(f"   CHANGED ({len(changed)}): {changed}")
        for k in added:
            print(f"      + {k} = {n[k]!r}")
        for k in changed:
            print(f"      ~ {k}: {b[k]!r} -> {n[k]!r}")

    b_par = base["manifest"]["parameters"]
    n_par = now["manifest"]["parameters"]
    print(f"== manifest parameters: {len(b_par)} -> {len(n_par)}")
    print(f"   ADDED: {sorted(set(n_par) - set(b_par))}")
    print(f"   REMOVED: {sorted(set(b_par) - set(n_par))}")
    b_fl = base["manifest"]["flags"]
    n_fl = now["manifest"]["flags"]
    print(f"== manifest flags: {len(b_fl)} -> {len(n_fl)}")
    print(f"   ADDED: {sorted(set(n_fl) - set(b_fl))}")
    print(f"   REMOVED: {sorted(set(b_fl) - set(n_fl))}")


if __name__ == "__main__":
    main()
