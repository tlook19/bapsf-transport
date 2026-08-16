"""Prove the config-snapshot change is EXACTLY the new neutral-energy keys.

The snapshot tripwire (``audit_sim1d_configs.verify_snapshots``) fires on any
change to the resolved LAPDSim1D config. Registering the NBL pass-1 neutral
energy field moves it legitimately, but "legitimately" has to be demonstrated,
not asserted: this recomputes every case digest with the
``neutral_energy_wall_accommodation`` parameter and the ``neutral_energy`` flag
REMOVED from the resolved config, and checks the result against the digests the
snapshot recorded BEFORE the keys existed. A match proves their addition is the
whole delta -- no default was altered, no key was dropped.

Usage:  PYTHONPATH=<checkout>/cablp python scripts/fnb1_config_delta.py
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

NEW_PARAMS = ("neutral_energy_wall_accommodation",)
NEW_FLAG = "neutral_energy"

#: Digests recorded in cablp/solvers/_sim1d/config_snapshots.json at base
#: commit add9b1e, i.e. before the neutral-energy keys were registered.
BASE = {
    "parameter_count": 254,
    "flag_count": 46,
    "manifest_sha256":
        "e8ee896d53b37895898fcff269049dfbc3ae19a6eafcdae84b3629714b86d541",
    "cases": {
        "compare_sim1d_es1":
            "82a9763e2b20811a87b35a0e465505e433e3709b5ff6f85db98324ab70f80027",
        "production_golden":
            "8edbd2d96eef961da43c6fcde5a27ead8da433a8715f27a862177dfa96de27b6",
        "run_m6_point_es1_sgp3400_defaults":
            "ed227a91cc8008e3009ba14b1dbea8c13b560879bba24e6441e7df609583812c",
        "run_mechanism_ladder_es1_defaults":
            "b135d70c10940f17e8b324263a24b44dd2b340d87a5b205d3027a733c26a3d56",
    },
}


def main():
    failures = []
    actual = current_snapshots()
    params_template, flags_template = default_config()

    print(f"new parameters ({len(NEW_PARAMS)}): {list(NEW_PARAMS)}")
    print(f"new flag: {NEW_FLAG}\n")

    print(f"parameter_count: {BASE['parameter_count']} -> "
          f"{actual['parameter_count']} "
          f"(delta {actual['parameter_count'] - BASE['parameter_count']})")
    if actual["parameter_count"] != BASE["parameter_count"] + len(NEW_PARAMS):
        failures.append(
            f"parameter_count moved by something other than +{len(NEW_PARAMS)}"
        )
    print(f"flag_count: {BASE['flag_count']} -> {actual['flag_count']} "
          f"(delta {actual['flag_count'] - BASE['flag_count']})")
    if actual["flag_count"] != BASE["flag_count"] + 1:
        failures.append("flag_count moved by something other than +1")

    for name in NEW_PARAMS:
        if name not in params_template:
            failures.append(f"{name} is not registered at all")
        elif params_template[name] != 0.40:
            failures.append(f"{name} default is not 0.40")
    if flags_template.get(NEW_FLAG) is not False:
        failures.append(f"{NEW_FLAG} default is not False")
    print("alpha_E registered at 0.40: "
          f"{params_template.get('neutral_energy_wall_accommodation')!r}")
    print(f"{NEW_FLAG} registered default-off: "
          f"{flags_template.get(NEW_FLAG) is False}")

    manifest = config_manifest()
    sources = {
        manifest["parameters"][name]["source"]
        for name in NEW_PARAMS
        if name in manifest["parameters"]
    }
    print(f"owned by defaults group(s): {sorted(sources)}\n")

    print("per-case digests with the neutral-energy keys stripped back out:")
    for name, (params, flags) in sorted(config_cases().items()):
        stripped_p = {k: v for k, v in params.items() if k not in NEW_PARAMS}
        stripped_f = {k: v for k, v in flags.items() if k != NEW_FLAG}
        if len(stripped_p) != len(params) - len(NEW_PARAMS):
            failures.append(f"{name}: new params absent from resolved config")
            continue
        if len(stripped_f) != len(flags) - 1:
            failures.append(f"{name}: {NEW_FLAG} absent from resolved flags")
            continue
        digest = config_digest(stripped_p, stripped_f)
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
          "once the neutral_energy_wall_accommodation parameter and the "
          "default-off neutral_energy flag are removed, so registering them is "
          "the ENTIRE config delta. Updating config_snapshots.json is "
          "therefore recording a proven change, not silencing a tripwire.")
    print(f"\nnew manifest_sha256: {actual['manifest_sha256']}")
    print(json.dumps(actual["cases"], sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
