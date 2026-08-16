"""fa1 -- A1 read: knee/breakdown timing, foot duration, pedestal e-fold class.

READ-ONLY. Two saved runs in, a table out; no solver, no config rebuild.

The e-fold estimator is IMPORTED from ``covcal_read.pedestal_efold`` rather
than re-implemented, so this read cannot drift from the definition that
produced the machine's 713-725 us class. That class is the MEASURED pedestal
e-fold (``clump_leg2_pedestal_vs_model.txt``: d(ln n)/dt = 1.4019 /ms over the
-4.0..-0.5 ms fit -> 713.3 us) and the band used by ``covcal_efold_read.py``.

CAVEAT carried from covcal_efold_read.py and NOT removable here: the band was
established on a different closure set (nx=60, coverage closure on). These
runs are nx=240 without the coverage closure, so the comparison is a CLASS
comparison, not a like-for-like gate.

Usage:
    python scripts/fa1_efold.py --arm ARM.h5 --ref REF.h5
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from covcal_read import pedestal_efold  # noqa: E402

BAND_LO_US, BAND_HI_US = 713.0, 725.0


def _s(x):
    return x.decode() if isinstance(x, bytes) else str(x)


def load(path):
    d = {}
    with h5py.File(path, "r") as f:
        d["path"] = path
        d["attrs"] = {k: v for k, v in f.attrs.items()
                      if k not in ("params_json", "flags_json")}
        d["status"] = str(f.attrs["run_status"]) if "run_status" in f.attrs \
            else "(absent: pre-opt-in run)"
        d["steps"] = int(f.attrs["steps"])
        d["final_time"] = float(f.attrs["final_time"])
        d["t_bd"] = float(f.attrs["t_breakdown_trigger"])
        d["t_pbd"] = float(f.attrs["t_prebreakdown_trigger"])
        p = json.loads(f.attrs["params_json"])
        d["S_gp"] = p["S_gp"]
        d["t"] = np.asarray(f["time"], float)
        act = np.asarray(f["geometry"]["plasma_active"], bool)
        d["meann"] = np.asarray(f["n"], float)[:, act].mean(axis=1)
        d["I"] = np.asarray(f["cathode_diagnostics"]["circuit_I_loop"], float)
        ev = f["phase_events"]
        d["ev"] = list(zip(np.asarray(ev["time"], float),
                           [_s(v) for v in np.asarray(ev["phase"])],
                           [_s(v) for v in np.asarray(ev["reason"])]))
    return d


def report(d, tag):
    print(f"\n--- {tag}: {d['path']}  (S_gp = {d['S_gp']:.0f} sccm) ---")
    print(f"  run_status  {d['status']}")
    print(f"  steps       {d['steps']}   final_time {d['final_time'] * 1e3:.4f} ms")
    print(f"  t_prebreakdown_trigger {d['t_pbd'] * 1e3:.6f} ms")
    print(f"  t_breakdown_trigger    {d['t_bd'] * 1e3:.6f} ms")
    print(f"  FOOT duration (prebreakdown->breakdown trigger) "
          f"{(d['t_bd'] - d['t_pbd']) * 1e3:.6f} ms")
    print("  phase events:")
    for tt, ph, rs in d["ev"]:
        print(f"    {tt * 1e3:10.5f} ms  -> {ph:16s}  ({rs})")
    I = d["I"]
    t = d["t"]
    print(f"  I_loop: max {I.max():.6g} A at {t[int(np.argmax(I))] * 1e3:.3f} ms")
    for thr in (2.0, 132.0, 1000.0, 2000.0, 3000.0):
        hit = np.flatnonzero(I >= thr)
        when = f"{t[hit[0]] * 1e3:.5f} ms" if hit.size else "NEVER"
        print(f"    first reaches {thr:8g} A : {when}")
    out = pedestal_efold(t, d["meann"], d["t_bd"])
    if out is None:
        print("  pedestal e-fold: NOT COMPUTABLE (fewer than 2 pre-bd saves)")
        return None
    leg, a, b, gr, ef = out
    print(f"  e-fold leg: {leg}")
    print(f"    saves in leg   {b - a + 1}  (index {a}..{b})")
    print(f"    <n>(a) {d['meann'][a]:.6e} -> <n>(b) {d['meann'][b]:.6e} cm^-3")
    if gr is None:
        print("    NOT COMPUTABLE (mean n did not grow across the leg)")
        return None
    print(f"    growth rate    {gr:.6g} /s   ({gr / 1e3:.6g} /ms)")
    band = ("IN BAND" if BAND_LO_US <= ef * 1e6 <= BAND_HI_US
            else "OUT OF BAND")
    print(f"    PEDESTAL E-FOLD  TAU = {ef * 1e6:.4f} us   "
          f"[band {BAND_LO_US:.0f}-{BAND_HI_US:.0f} us: {band}]")
    return ef


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--ref", required=True)
    args = ap.parse_args()
    print("=" * 78)
    print("fa1 A1 -- knee/breakdown timing, foot duration, pedestal e-fold")
    print("=" * 78)
    ref = load(args.ref)
    arm = load(args.arm)
    ef_r = report(ref, "REF")
    ef_a = report(arm, "ARM")
    print("\n--- ARM vs REF ---")
    print(f"  foot duration  REF {(ref['t_bd'] - ref['t_pbd']) * 1e3:.6f} ms"
          f"   ARM {(arm['t_bd'] - arm['t_pbd']) * 1e3:.6f} ms"
          f"   ratio {(arm['t_bd'] - arm['t_pbd']) / (ref['t_bd'] - ref['t_pbd']):.4f}")
    print(f"  t_breakdown    REF {ref['t_bd'] * 1e3:.6f} ms"
          f"   ARM {arm['t_bd'] * 1e3:.6f} ms"
          f"   delta {(arm['t_bd'] - ref['t_bd']) * 1e3:+.6f} ms")
    if ef_r and ef_a:
        print(f"  pedestal e-fold REF {ef_r * 1e6:.4f} us"
              f"   ARM {ef_a * 1e6:.4f} us   ratio {ef_a / ef_r:.4f}")


if __name__ == "__main__":
    main()
