"""P1 pre-flight: NO-SOLVE config diff of the mn1z arm vs the ES1 reference h5.

Run artifact (untracked). Rebuilds the config exactly as
compare_sim1d_es1.run_model() does -- WITHOUT constructing or solving the sim --
and diffs params/flags against the reference artifact's params_json/flags_json.

Usage:
    python scripts/mn_p1_preflight_diffcfg.py <reference.h5>
"""
import json
import sys
from pathlib import Path

import h5py

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from cablp.solvers._sim1d import default_config  # noqa: E402

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "cmp_es1", str(_HERE / "compare_sim1d_es1.py")
)
cmp_es1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cmp_es1)


def build(drag_closure, tau_afterglow=0.006, nx=None, exchange_model="knudsen"):
    """Mirror run_model()'s config construction exactly, minus the solve."""
    if nx is None:
        nx = cmp_es1.PRODUCTION_NX
    params, flags = default_config()
    params.update(cmp_es1.PARAM_OVERRIDES)
    flags.update(cmp_es1.FLAG_OVERRIDES)
    params["neutral_exchange_model"] = exchange_model
    if nx is not None:
        params["nx"] = nx
    if drag_closure == "slip":
        params["ion_neutral_drag_model"] = "slip"
        params["b_ion_neutral_drag"] = 1.0
    elif drag_closure == "neutral_momentum":
        params["ion_neutral_drag_model"] = "constant"
        params["b_ion_neutral_drag"] = 1.0
        flags["neutral_momentum"] = True
    elif drag_closure == "neutral_momentum_two_zone":
        params["ion_neutral_drag_model"] = "constant"
        params["b_ion_neutral_drag"] = 1.0
        params["neutral_momentum_radial"] = "two_zone"
        flags["neutral_momentum"] = True
    elif drag_closure not in (None, "constant"):
        raise ValueError(f"unknown drag_closure {drag_closure!r}")
    # `extra` is applied LAST by run_model; the CLI puts tau_afterglow there.
    extra = {}
    if tau_afterglow is not None:
        extra["tau_afterglow"] = tau_afterglow
    params.update(extra)
    return params, flags


def read_ref(path):
    with h5py.File(path, "r") as f:
        p = json.loads(f.attrs["params_json"])
        fl = json.loads(f.attrs["flags_json"])
    return p, fl


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
        print(f"    ADDED    {k}: <absent> -> {v!r}")
    for k, v in removed:
        print(f"    REMOVED  {k}: {v!r} -> <absent>")
    return len(changed) + len(added) + len(removed)


def main():
    ref_path = sys.argv[1]
    rp, rf = read_ref(ref_path)
    print(f"reference: {ref_path}")
    print(f"  ref params keys={len(rp)}  flags keys={len(rf)}")
    print(f"  PRODUCTION_NX = {cmp_es1.PRODUCTION_NX}")

    total = {}
    for label, dc in (("CONTROL rebuild (no --drag-closure)", None),
                      ("ARM rebuild (--drag-closure neutral_momentum)",
                       "neutral_momentum")):
        print(f"\n=== {label} ===")
        p, fl = build(dc)
        np_ = diff("params", rp, p)
        nf_ = diff("flags", rf, fl)
        total[label] = (np_, nf_)

    print("\n=== VERDICT ===")
    c = total["CONTROL rebuild (no --drag-closure)"]
    a = total["ARM rebuild (--drag-closure neutral_momentum)"]
    print(f"  control: params {c[0]} deltas, flags {c[1]} deltas "
          f"(REQUIRED 0 / 0)  -> {'OK' if c == (0, 0) else 'STOP'}")
    print(f"  arm:     params {a[0]} deltas, flags {a[1]} deltas "
          f"(REQUIRED 0 / 1)  -> {'OK' if a == (0, 1) else 'STOP'}")
    ok = c == (0, 0) and a == (0, 1)
    print(f"  PRE-FLIGHT: {'PASS' if ok else 'FAIL -- DO NOT RUN'}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
