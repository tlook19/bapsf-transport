"""Prove the config-snapshot change is EXACTLY the new probe-source keys.

The snapshot tripwire (``audit_sim1d_configs.verify_snapshots``) fires on any
change to the resolved LAPDSim1D config. Registering the ad-hoc probe source
moves it legitimately, but "legitimately" has to be demonstrated, not asserted:
this recomputes every case digest with the ten ``neutral_probe_*`` parameters
and the ``neutral_probe_source`` flag REMOVED from the resolved config, and
checks the result against the digests the snapshot recorded BEFORE the keys
existed. A match proves their addition is the whole delta -- no default was
altered, no key was dropped.

Usage:  python scripts/sprobe_config_delta.py
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
    neutral_probe_source_defaults,
)
from cablp.solvers._sim1d import (  # noqa: E402
    config_manifest,
    default_config,
)

NEW_PARAMS = tuple(sorted(neutral_probe_source_defaults()))
NEW_FLAG = "neutral_probe_source"

#: Digests recorded in cablp/solvers/_sim1d/config_snapshots.json at base
#: commit 6cb1abf, i.e. before the probe-source keys were registered.
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
    params_template, flags_template = default_config()

    print(f"new parameters ({len(NEW_PARAMS)}): {list(NEW_PARAMS)}")
    print(f"new flag: {NEW_FLAG}\n")

    print(f"parameter_count: {BASE['parameter_count']} -> "
          f"{actual['parameter_count']} "
          f"(delta {actual['parameter_count'] - BASE['parameter_count']})")
    if actual["parameter_count"] != BASE["parameter_count"] + len(NEW_PARAMS):
        failures.append(
            "parameter_count moved by something other than "
            f"+{len(NEW_PARAMS)}"
        )
    print(f"flag_count: {BASE['flag_count']} -> {actual['flag_count']} "
          f"(delta {actual['flag_count'] - BASE['flag_count']})")
    if actual["flag_count"] != BASE["flag_count"] + 1:
        failures.append("flag_count moved by something other than +1")

    for name in NEW_PARAMS:
        if name not in params_template:
            failures.append(f"{name} is not registered at all")
        elif params_template[name] is not None:
            failures.append(f"{name} default is not None")
    if flags_template.get(NEW_FLAG) is not False:
        failures.append(f"{NEW_FLAG} default is not False")
    print("all ten parameters registered with default None: "
          f"{all(params_template.get(n, 0) is None for n in NEW_PARAMS)}")
    print(f"{NEW_FLAG} registered default-off: "
          f"{flags_template.get(NEW_FLAG) is False}")

    sources = {
        config_manifest()["parameters"][name]["source"]
        for name in NEW_PARAMS
        if name in config_manifest()["parameters"]
    }
    print(f"owned by defaults group(s): {sorted(sources)}\n")

    print("per-case digests with the probe keys stripped back out:")
    for name, (params, flags) in sorted(config_cases().items()):
        stripped_p = {k: v for k, v in params.items() if k not in NEW_PARAMS}
        stripped_f = {k: v for k, v in flags.items() if k != NEW_FLAG}
        if len(stripped_p) != len(params) - len(NEW_PARAMS):
            failures.append(f"{name}: probe params absent from resolved config")
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
          "once the ten default-None probe parameters and the default-off "
          "neutral_probe_source flag are removed, so registering them is the "
          "ENTIRE config delta. Updating config_snapshots.json is therefore "
          "recording a proven change, not silencing a tripwire.")
    print(f"\nnew manifest_sha256: {actual['manifest_sha256']}")
    print(json.dumps(actual["cases"], sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
