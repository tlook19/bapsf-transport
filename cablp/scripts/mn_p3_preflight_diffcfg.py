"""P3 pre-flight: NO-SOLVE config diff of the compensated arm vs the ES1 reference.

Run artifact (untracked).  Mimics run_m6_point.main()'s config construction
EXACTLY (its ES_OPERATING-driven `extra` dict, then run_model's
default_config + PARAM_OVERRIDES + FLAG_OVERRIDES stack) WITHOUT constructing
or solving the sim, then diffs against the reference artifact's
params_json/flags_json.

Usage:
    python scripts/mn_p3_preflight_diffcfg.py <reference.h5> <sgp> [k=v ...]
        [--flag k=v ...]
"""
import json
import sys
from pathlib import Path

import h5py

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))

from cablp.solvers._sim1d import default_config  # noqa: E402
import compare_sim1d_es1 as cmp_es1  # noqa: E402
import run_m6_point as m6  # noqa: E402
from run_mechanism_ladder import ES_OPERATING  # noqa: E402


def build_m6(es, nx, sgp, extra_kv, flag_kv):
    """Reproduce run_m6_point.main()'s extra/flags_extra, then run_model's stack."""
    op = ES_OPERATING[es]
    extra = {
        "nx": nx,
        "V_bank": op["V_bank"],
        "cathode_solver_model": "current_driven",
        "beam_deposition_model": "csda",
        "beam_anomalous_model": "quasilinear",
        "cathode_emission_profile": "gaussian",
        "cathode_warming_model": "power_balance",
        "T_s": op["Ts_standby_K"],
        "cathode_Ts_base_K": op["Ts_standby_K"],
        "cathode_heat_capacity_J_per_K": 120.0,
        "cathode_emissivity": 0.7,
        "phi_wf": 2.869,
        "cathode_surface_model": "ads_des",
        "cathode_phiwf_clean_eV": 2.809,
        "cathode_cleaning_sigma_cm2": 3.5e-16,
        "cathode_cleaning_E_th_eV": 20.0,
        "Te_birth_ionization": m6.ELECTRON_BIRTH_POLICY,
        "gas_puff_mode": "square",
        "S_gp": sgp,
    }
    # --no-smooth NOT passed => presheath smoothing, as run_m6_point does
    extra["cathode_sample_smoothing"] = "presheath"
    flags_extra = {}
    for k, v in extra_kv:
        try:
            extra[k] = json.loads(v)
        except json.JSONDecodeError:
            extra[k] = v
    for k, v in flag_kv:
        try:
            flags_extra[k] = json.loads(v)
        except json.JSONDecodeError:
            flags_extra[k] = v

    # run_model's own stack (nx, exchange_model default "knudsen",
    # drag_closure None -- run_m6_point never passes it)
    params, flags = default_config()
    params.update(cmp_es1.PARAM_OVERRIDES)
    flags.update(cmp_es1.FLAG_OVERRIDES)
    flags.update(flags_extra)
    params["neutral_exchange_model"] = "knudsen"
    params["nx"] = nx
    params.update(extra)
    return params, flags


def read_ref(path):
    with h5py.File(path, "r") as f:
        return (json.loads(f.attrs["params_json"]),
                json.loads(f.attrs["flags_json"]))


def diff(name, ref, new):
    changed, added, removed = [], [], []
    for k in sorted(set(ref) | set(new)):
        if k not in ref:
            added.append((k, new[k]))
        elif k not in new:
            removed.append((k, ref[k]))
        elif ref[k] != new[k]:
            changed.append((k, ref[k], new[k]))
    print(f"  {name}: {len(changed)} CHANGED, {len(added)} ADDED, "
          f"{len(removed)} REMOVED")
    for k, a, b in changed:
        print(f"    CHANGED  {k}: {a!r} -> {b!r}")
    for k, v in added:
        print(f"    ADDED    {k}: <MISSING in ref> -> {v!r}")
    for k, v in removed:
        print(f"    REMOVED  {k}: {v!r} -> <absent>")
    return changed, added, removed


def main():
    ref_path, sgp = sys.argv[1], float(sys.argv[2])
    rest = sys.argv[3:]
    extra_kv, flag_kv, mode = [], [], "extra"
    for tok in rest:
        if tok == "--flag":
            mode = "flag"
            continue
        k, v = tok.split("=", 1)
        (extra_kv if mode == "extra" else flag_kv).append((k, v))

    rp, rf = read_ref(ref_path)
    print(f"reference: {ref_path}")
    print(f"  ref params keys={len(rp)}  flags keys={len(rf)}")
    print(f"  candidate: run_m6_point --es 1 --nx 240 --sgp {sgp}")
    print(f"    --extra {' '.join(f'{k}={v}' for k, v in extra_kv)}")
    print(f"    --extra-flag {' '.join(f'{k}={v}' for k, v in flag_kv)}")

    p, fl = build_m6(1, cmp_es1.PRODUCTION_NX, sgp, extra_kv, flag_kv)
    print("\n=== CANDIDATE vs REFERENCE ===")
    pc, pa, pr = diff("params", rp, p)
    fc, fa, fr = diff("flags", rf, fl)

    print("\n=== VERDICT ===")
    print("  REQUIRED: params exactly {S_gp: 3000 -> 4056.38}, "
          "flags exactly {neutral_momentum: False -> True},")
    print("  modulo keys ABSENT from the reference that sit at their "
          "default_config() defaults.")
    ok_p = ([k for k, _a, _b in pc] == ["S_gp"]) and not pr
    ok_f = ([k for k, _a, _b in fc] == ["neutral_momentum"]) and not fa and not fr
    print(f"  params CHANGED set == ['S_gp'] and none REMOVED : {ok_p}")
    print(f"  flags  CHANGED set == ['neutral_momentum']      : {ok_f}")
    if pa:
        print(f"  params ADDED (absent from reference): {[k for k, _ in pa]}")
        d, _ = default_config()
        for k, v in pa:
            at_default = k in d and d[k] == v
            print(f"    {k}: candidate={v!r}  at default_config() default: "
                  f"{at_default}"
                  + ("" if k in d else "  (key not in default_config())"))
    ok = ok_p and ok_f
    print(f"  PRE-FLIGHT: {'PASS' if ok else 'FAIL -- DO NOT RUN'}"
          + (" (with ADDED-key riders reported above)" if pa and ok else ""))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
