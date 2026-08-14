"""Prove the config-snapshot change is EXACTLY the new transpiration selector.

The snapshot tripwire (``audit_sim1d_configs.verify_snapshots``) fires on any
change to the resolved LAPDSim1D config. The NBL pass-2 decoupled two-channel
build adds exactly ONE key -- ``neutral_knudsen_temperature``, the
thermal-transpiration selector, registered at its ratified v1-primary value
``"frozen"``. Everything else pass 2 builds rides the ``neutral_energy`` flag
that pass 1 already registered, so it moves no config surface at all.

"Exactly one key" has to be demonstrated, not asserted: this recomputes every
case digest with that parameter REMOVED from the resolved config and checks the
result against the digests the snapshot recorded at the branch base. A match
proves its addition is the whole delta -- no default was altered, no key was
dropped.

Usage:  PYTHONPATH=<checkout>/cablp python scripts/fnb2_config_delta.py
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

from cablp.solvers._sim1d import (  # noqa: E402
    config_manifest,
    default_config,
)

NEW_PARAMS = ("neutral_knudsen_temperature",)
NEW_PARAM_DEFAULTS = {"neutral_knudsen_temperature": "frozen"}

#: Digests recorded in cablp/solvers/_sim1d/config_snapshots.json at the branch
#: base commit c9ddcbe (the merged NBL pass-1 En core), i.e. before the
#: transpiration selector was registered.
BASE = {
    "parameter_count": 255,
    "flag_count": 47,
    "manifest_sha256":
        "e634521b6f00bf76327a779da49fddbe18e030f337e7dc0c668bf4176d85e04d",
    "cases": {
        "compare_sim1d_es1":
            "d00ca3eb93a9eca8d3dcd1b5219479ff4fb08eea8e7ba1cda60b9f60e2fcbb75",
        "production_golden":
            "0943a8e8128601e10d65bdb8033e99079389216672f0951de11baf67fd3dd6c6",
        "run_m6_point_es1_sgp3400_defaults":
            "338c9e5e991dd5a0687fe5e3bfc373f916cd431316ca3546718ce45bc0b13a65",
        "run_mechanism_ladder_es1_defaults":
            "6b513f6d7441ae9c60b2ffb50d3978242a0e4e94e8beba7cf459a3afe3fdac59",
    },
}


def main():
    failures = []
    actual = current_snapshots()
    params_template, flags_template = default_config()

    print(f"new parameters ({len(NEW_PARAMS)}): {list(NEW_PARAMS)}")
    print("new flags: none -- pass 2 rides the pass-1 neutral_energy flag\n")

    print(f"parameter_count: {BASE['parameter_count']} -> "
          f"{actual['parameter_count']} "
          f"(delta {actual['parameter_count'] - BASE['parameter_count']})")
    if actual["parameter_count"] != BASE["parameter_count"] + len(NEW_PARAMS):
        failures.append(
            f"parameter_count moved by something other than +{len(NEW_PARAMS)}"
        )
    print(f"flag_count: {BASE['flag_count']} -> {actual['flag_count']} "
          f"(delta {actual['flag_count'] - BASE['flag_count']})")
    if actual["flag_count"] != BASE["flag_count"]:
        failures.append("flag_count moved, but pass 2 registers no new flag")

    for name, expected in NEW_PARAM_DEFAULTS.items():
        if name not in params_template:
            failures.append(f"{name} is not registered at all")
        elif params_template[name] != expected:
            failures.append(f"{name} default is not {expected!r}")
        else:
            print(f"{name} registered at {params_template[name]!r}")
    if flags_template.get("neutral_energy") is not False:
        failures.append("neutral_energy is no longer default-off")
    print("neutral_energy still default-off: "
          f"{flags_template.get('neutral_energy') is False}")

    manifest = config_manifest()
    sources = {
        manifest["parameters"][name]["source"]
        for name in NEW_PARAMS
        if name in manifest["parameters"]
    }
    print(f"owned by defaults group(s): {sorted(sources)}\n")

    print("per-case digests with the transpiration selector stripped back out:")
    for name, (params, flags) in sorted(config_cases().items()):
        stripped_p = {k: v for k, v in params.items() if k not in NEW_PARAMS}
        if len(stripped_p) != len(params) - len(NEW_PARAMS):
            failures.append(f"{name}: new params absent from resolved config")
            continue
        digest = config_digest(stripped_p, flags)
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
    print("OVERALL: PASS -- every case digest reproduces its pre-branch value "
          "once the neutral_knudsen_temperature parameter is removed, so "
          "registering it is the ENTIRE config delta. Updating "
          "config_snapshots.json is therefore recording a proven change, not "
          "silencing a tripwire.")
    print(f"\nnew manifest_sha256: {actual['manifest_sha256']}")
    print(json.dumps(actual["cases"], sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
