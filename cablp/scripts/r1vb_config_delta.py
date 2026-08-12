"""Prove the config-snapshot change is EXACTLY the new circuit-bound flag.

The snapshot tripwire (``audit_sim1d_configs``) fires on any change to the
resolved LAPDSim1D config. Registering ``cathode_circuit_voltage_bound`` moves
it legitimately, but "legitimately" has to be demonstrated, not asserted: this
recomputes every case digest with the new flag REMOVED from the resolved flags
and checks the result against the digests the snapshot recorded BEFORE the key
existed. A match proves the key's addition is the whole delta -- no default was
altered, no key was dropped, and no parameter moved.

Same pattern as ``restart_config_delta.py`` / ``sprobe_config_delta.py``; the
only structural difference is that this key lives in the FLAGS namespace, so
the count that must move by +1 is ``flag_count`` and ``parameter_count`` must
be untouched.

Usage:  python scripts/r1vb_config_delta.py
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
    input_flags_template_1d,
)

NEW_KEY = "cathode_circuit_voltage_bound"

#: Digests recorded in cablp/solvers/_sim1d/config_snapshots.json at base
#: commit 338bb2e, i.e. before cathode_circuit_voltage_bound was registered.
BASE = {
    "parameter_count": 226,
    "flag_count": 40,
    "manifest_sha256":
        "60425291d7bb077ed54d0130640ba34caaa798546cabab4ee93fd31997cd91d0",
    "cases": {
        "compare_sim1d_es1":
            "480a39e4e0fa9129ad00851c20cccb67e38777675f692992d877a422c3bb706b",
        "production_golden":
            "83d8f36168387fda3f3927ec699539cd7a6dec0f85878bedb2e00e5c06318585",
        "run_m6_point_es1_sgp3400_defaults":
            "d23d5c7eea215fc7dc3afa2021d53a07ea22e04f77dee6a9ed6b6993a8a2465e",
        "run_mechanism_ladder_es1_defaults":
            "d4d2d1704b1581c592eb3c900f38077ac0c930fb91fd84c39f69742f97cae2b3",
    },
}


def main():
    failures = []
    actual = current_snapshots()

    print(f"flag_count: {BASE['flag_count']} -> {actual['flag_count']} "
          f"(delta {actual['flag_count'] - BASE['flag_count']})")
    if actual["flag_count"] != BASE["flag_count"] + 1:
        failures.append("flag_count moved by something other than +1")
    print(f"parameter_count: {BASE['parameter_count']} -> "
          f"{actual['parameter_count']} "
          + ("(unchanged)" if actual["parameter_count"]
             == BASE["parameter_count"] else "CHANGED"))
    if actual["parameter_count"] != BASE["parameter_count"]:
        failures.append("parameter_count changed")

    if NEW_KEY not in input_flags_template_1d:
        failures.append(f"{NEW_KEY} is not registered at all")
    elif input_flags_template_1d[NEW_KEY] is not False:
        failures.append(f"{NEW_KEY} default is not False")
    else:
        print(f"{NEW_KEY} registered with default False: yes")

    if NEW_KEY in config_manifest()["parameters"]:
        failures.append(f"{NEW_KEY} leaked into the PARAMETER namespace")
    else:
        print(f"{NEW_KEY} is a flag, not a parameter: yes")
    print()

    print(f"per-case digests with {NEW_KEY} stripped back out:")
    for name, (params, flags) in sorted(config_cases().items()):
        stripped = {k: v for k, v in flags.items() if k != NEW_KEY}
        if len(stripped) != len(flags) - 1:
            failures.append(f"{name}: {NEW_KEY} absent from resolved flags")
            continue
        if flags[NEW_KEY] is not False:
            failures.append(f"{name}: {NEW_KEY} resolves to something but False")
        digest = config_digest(params, stripped)
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
    print("OVERALL: PASS -- every case digest reproduces its pre-R1 value once "
          f"{NEW_KEY} is removed, so registering that one default-False flag "
          "is the ENTIRE config delta. Updating config_snapshots.json is "
          "therefore recording a proven change, not silencing a tripwire.")
    print(f"\nnew manifest_sha256: {actual['manifest_sha256']}")
    print(json.dumps(actual["cases"], sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
