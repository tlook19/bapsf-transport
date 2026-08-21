"""Prove the config-snapshot delta is EXACTLY ``cathode_jet_hot_carrier``.

`cablp/solvers/_sim1d/config_snapshots.json` is a reviewed fixture: a changed
default, precedence rule or driver choice fails the smoke before a solver run
can silently inherit it. Adding one params key legitimately moves it, and the
discipline is to prove the move rather than to accept it: DELETING the new key
from the live manifest and from every case's resolved config must reproduce the
previously committed hashes bit for bit. If it does, the fixture's only delta is
the new key and the regeneration is a bookkeeping act; if it does not, something
else moved and the regeneration would bury it.

    python scripts/t23c_config_snapshot_delta.py --key cathode_jet_hot_carrier
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
if str(_SCRIPTS.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS.parent))

from audit_sim1d_configs import (  # noqa: E402
    SNAPSHOT_PATH,
    config_cases,
    config_digest,
    current_snapshots,
)

from cablp.solvers._sim1d import config_manifest  # noqa: E402


def _sha(obj):
    return hashlib.sha256(
        json.dumps(
            obj, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", default="cathode_jet_hot_carrier")
    args = parser.parse_args()
    key = args.key

    committed = json.loads(SNAPSHOT_PATH.read_text())
    live = current_snapshots()

    manifest = config_manifest()
    if key not in manifest["parameters"]:
        raise SystemExit(f"{key!r} is not a params-namespace key")
    stripped = {
        "parameters": {
            name: value
            for name, value in manifest["parameters"].items()
            if name != key
        },
        "flags": manifest["flags"],
    }
    stripped.update(
        {k: v for k, v in manifest.items() if k not in ("parameters", "flags")}
    )

    ok = True
    print("=" * 74)
    print(f"CONFIG-SNAPSHOT DELTA PROOF -- one new params key: {key}")
    print("=" * 74)
    print(f"  committed parameter_count {committed['parameter_count']}")
    print(f"  live      parameter_count {live['parameter_count']}")
    print(f"  committed flag_count      {committed['flag_count']}")
    print(f"  live      flag_count      {live['flag_count']}")
    ok &= live["parameter_count"] == committed["parameter_count"] + 1
    ok &= live["flag_count"] == committed["flag_count"]

    rebuilt = _sha(stripped)
    print()
    print("  manifest_sha256 with the key REMOVED from the live manifest")
    print(f"    rebuilt   {rebuilt}")
    print(f"    committed {committed['manifest_sha256']}")
    match = rebuilt == committed["manifest_sha256"]
    ok &= match
    print(f"    {'MATCH' if match else 'MISMATCH'}")

    print()
    print("  per-case sha256 with the key REMOVED from the resolved config")
    for name, (params, flags) in sorted(config_cases().items()):
        if key not in params:
            print(f"    {name:44s} key ABSENT from this case -- unexpected")
            ok = False
            continue
        pruned = {k: v for k, v in params.items() if k != key}
        rebuilt_case = config_digest(pruned, flags)
        want = committed["cases"][name]["sha256"]
        same = rebuilt_case == want
        ok &= same
        print(f"    {name:44s} {'MATCH' if same else 'MISMATCH'}")
        if not same:
            print(f"      rebuilt   {rebuilt_case}")
            print(f"      committed {want}")

    print()
    print("PROOF " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
