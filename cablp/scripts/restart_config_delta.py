"""Prove the config-snapshot change is EXACTLY the new restart_from key.

The snapshot tripwire (``audit_sim1d_configs.verify_snapshots``) fires on any
change to the resolved LAPDSim1D config. Registering ``restart_from`` moves it
legitimately, but "legitimately" has to be demonstrated, not asserted: this
recomputes every case digest with ``restart_from`` REMOVED from the resolved
params and checks the result against the digests the snapshot recorded BEFORE
the key existed. A match proves the key's addition is the whole delta -- no
default was altered, no key was dropped.

Usage:  python scripts/restart_config_delta.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_sim1d_configs import (  # noqa: E402
    config_cases,
    config_digest,
    current_snapshots,
)

from cablp.solvers._sim1d.core.config import (  # noqa: E402
    config_manifest,
    input_dict_template_1d,
)

NEW_KEY = "restart_from"

#: Digests recorded in cablp/solvers/_sim1d/config_snapshots.json at base
#: commit 4e4dd27, i.e. before restart_from was registered.
BASE = {
    "parameter_count": 225,
    "flag_count": 40,
    "manifest_sha256":
        "8894d278d15ac7f59ceb5d9924e5e1d177e3d16eea51231f497bf5abd85c12ca",
    "cases": {
        "compare_sim1d_es1":
            "bdf62a892c25c91d407bf227070c7f17115289ccad7f3d2352c73721a486a48a",
        "production_golden":
            "dea3ae5d06f16210745949c28f1cd177453a239017dc7051c70a529c5bb1b743",
        "run_m6_point_es1_sgp3400_defaults":
            "851ca65187893380579d0135a4a1003128477a47c6e81dcffc33a7ee3d4b4125",
        "run_mechanism_ladder_es1_defaults":
            "3320364096f3578027446a468da4ef65da8fba163666578290a51f2b63e01f9a",
    },
}


def main():
    failures = []
    actual = current_snapshots()

    print(f"parameter_count: {BASE['parameter_count']} -> "
          f"{actual['parameter_count']} "
          f"(delta {actual['parameter_count'] - BASE['parameter_count']})")
    if actual["parameter_count"] != BASE["parameter_count"] + 1:
        failures.append("parameter_count moved by something other than +1")
    print(f"flag_count: {BASE['flag_count']} -> {actual['flag_count']} "
          "(unchanged)" if actual["flag_count"] == BASE["flag_count"]
          else f"flag_count CHANGED: {actual['flag_count']}")
    if actual["flag_count"] != BASE["flag_count"]:
        failures.append("flag_count changed")

    if NEW_KEY not in input_dict_template_1d:
        failures.append(f"{NEW_KEY} is not registered at all")
    elif input_dict_template_1d[NEW_KEY] is not None:
        failures.append(f"{NEW_KEY} default is not None")
    else:
        print(f"{NEW_KEY} registered with default None: yes")

    source = config_manifest()["parameters"][NEW_KEY]["source"]
    print(f"{NEW_KEY} owned by defaults group: {source}\n")

    print("per-case digests with restart_from stripped back out:")
    for name, (params, flags) in sorted(config_cases().items()):
        stripped = {k: v for k, v in params.items() if k != NEW_KEY}
        if len(stripped) != len(params) - 1:
            failures.append(f"{name}: {NEW_KEY} absent from resolved params")
            continue
        digest = config_digest(stripped, flags)
        ok = digest == BASE["cases"][name]
        print(f"  {'MATCH  ' if ok else 'DIFFER '} {name}")
        if not ok:
            print(f"      base   {BASE['cases'][name]}")
            print(f"      actual {digest}")
            failures.append(f"{name}: stripped digest != base digest")

    print()
    if failures:
        print(f"OVERALL: FAIL -- {len(failures)} problem(s): {failures}")
        return 1
    print("OVERALL: PASS -- every case digest reproduces its pre-restart value "
          "once restart_from is removed, so registering that one default-None "
          "key is the ENTIRE config delta. Updating config_snapshots.json is "
          "therefore recording a proven change, not silencing a tripwire.")
    print(f"\nnew manifest_sha256: {actual['manifest_sha256']}")
    print(json.dumps(actual["cases"], sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
