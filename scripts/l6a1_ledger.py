"""L6 piece-1: END-REGION SOURCE-ATTRIBUTED LEDGER over a post-breakdown window.

Report-only. Follows scripts/l6_tpmc_background_check.py: the COLUMN row nn
integrates on the plasma volume Vp, the ANNULUS row nn_a on Vm - Vp.
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mc_neutrals import (  # noqa: E402
    assert_end_recycle_routed_live,
    boundary_recycle_row,
)

path = sys.argv[1]
lo, hi = float(sys.argv[2]), float(sys.argv[3])
routed_guard = len(sys.argv) > 4 and sys.argv[4] == "--guard"

CHANNELS = [
    ("face-recycle", None),          # resolved to the live boundary row
    ("exchange", "neutral_zone_exchange"),
    ("hot-wall", "neutral_energy_wall"),
    ("hot-channel", "neutral_hot_channel"),
    ("recombination", "recombination_3b_loss"),
    ("neutral_sources", "neutral_sources"),
]

with h5py.File(path, "r") as f:
    t0 = float(f.attrs["t_breakdown_trigger"])
    t_ms = (f["time"][:] - t0) * 1e3
    m = (t_ms >= lo) & (t_ms <= hi)
    row, stance = boundary_recycle_row(f)
    flags = json.loads(f.attrs.get("flags_json", "{}"))
    Vm = f["geometry/neutral_volume_cm3"][:]
    Vp = f["geometry/plasma_volume_cm3"][:]
    Va = np.maximum(Vm - Vp, 0.0)
    z = f["geometry/z_cm"][:]
    roles = [r.decode() if isinstance(r, bytes) else str(r)
             for r in f["geometry/cell_role"][:]]
    nn_w = np.mean(f["nn"][:][m], axis=0)
    nn_a_w = (np.mean(f["nn_a"][:][m], axis=0)
              if "nn_a" in f else None)

    print(f"artifact : {path}")
    print(f"window   : {lo}-{hi} ms after breakdown  ({int(m.sum())} saves, "
          f"t_ms {t_ms[m].min():.3f}..{t_ms[m].max():.3f})")
    print(f"live boundary row : rhs_terms/{row}  (characteristic_boundary={stance})")
    print(f"stance   : neutral_two_zone={flags.get('neutral_two_zone')}, "
          f"end_recycle_to_annulus={flags.get('end_recycle_to_annulus')}")

    cath = [i for i, r in enumerate(roles) if r == "cathode"]
    coll = [i for i, r in enumerate(roles) if r == "collector"]
    endc = [i for i, r in enumerate(roles) if r == "end"]
    end_region = sorted(endc + coll)

    print("\n=== END-REGION CELL COLUMN/ANNULUS DENSITY "
          "(time-averaged over the window) ===")
    print(f"{'cell':>5}{'role':>11}{'z_cm':>10}"
          f"{'nn [cm^-3]':>16}{'nn_a [cm^-3]':>16}{'Vp':>13}{'V_ann':>13}")
    for i in end_region:
        na = f"{nn_a_w[i]:.6e}" if nn_a_w is not None else "n/a"
        print(f"{i:>5}{roles[i]:>11}{z[i]:>10.1f}"
              f"{nn_w[i]:>16.6e}{na:>16}{Vp[i]:>13.5e}{Va[i]:>13.5e}")
    print(f"{'--':>5}{'cathode':>11}{z[cath[-1]]:>10.1f}"
          f"{nn_w[cath[-1]]:>16.6e}"
          f"{(f'{nn_a_w[cath[-1]]:.6e}' if nn_a_w is not None else 'n/a'):>16}"
          f"{Vp[cath[-1]]:>13.5e}{Va[cath[-1]]:>13.5e}")

    print("\n=== SOURCE-ATTRIBUTED RATES [particles/s], window-averaged ===")
    print("  column contribution = <nn-rate> * Vp ; annulus = <nn_a-rate> * V_ann")
    hdr = (f"{'channel':<22}{'cell':>5}{'role':>11}"
           f"{'column [1/s]':>18}{'annulus [1/s]':>18}{'total [1/s]':>18}")
    ba_new = ba_ann_row = None
    for label, key in CHANNELS:
        k = row if key is None else key
        if f"rhs_terms/{k}/nn" not in f:
            print(f"\n{label:<22} rhs_terms/{k}: ABSENT")
            continue
        col = np.mean(f[f"rhs_terms/{k}/nn"][:][m], axis=0)
        ann = (np.mean(f[f"rhs_terms/{k}/nn_a"][:][m], axis=0)
               if f"rhs_terms/{k}/nn_a" in f else None)
        print(f"\n--- {label}  (rhs_terms/{k}) ---")
        print(hdr)
        tot_c = tot_a = 0.0
        for i in sorted(set(end_region + cath)):
            c = float(col[i] * Vp[i])
            a = float(ann[i] * Va[i]) if ann is not None else 0.0
            if c == 0.0 and a == 0.0:
                continue
            tot_c += c
            tot_a += a
            print(f"{'':<22}{i:>5}{roles[i]:>11}{c:>18.6e}"
                  f"{a:>18.6e}{c + a:>18.6e}")
        print(f"{'  END-REGION+CATH TOTAL':<22}{'':>5}{'':>11}"
              f"{tot_c:>18.6e}{tot_a:>18.6e}{tot_c + tot_a:>18.6e}")
        gc = float((col * Vp).sum())
        ga = float((ann * Va).sum()) if ann is not None else 0.0
        print(f"{'  WHOLE-GRID TOTAL':<22}{'':>5}{'':>11}"
              f"{gc:>18.6e}{ga:>18.6e}{gc + ga:>18.6e}")
        if key is None:
            ba_new = col * Vp + (ann * Va if ann is not None else 0.0)
            ba_ann_row = ann

    print("\n=== PER-FACE RECYCLE RATE [particles/s] "
          f"(rhs_terms/{row}, zone-correct read) ===")
    print(f"{'face':<14}{'cell':>5}{'column':>18}{'annulus':>18}{'total':>18}")
    col = np.mean(f[f"rhs_terms/{row}/nn"][:][m], axis=0)
    ann = (np.mean(f[f"rhs_terms/{row}/nn_a"][:][m], axis=0)
           if f"rhs_terms/{row}/nn_a" in f else None)
    for lab, cells in (("cathode", cath), ("collector", coll)):
        c = float((col[cells] * Vp[cells]).sum())
        a = float((ann[cells] * Va[cells]).sum()) if ann is not None else 0.0
        print(f"{lab:<14}{cells[-1]:>5}{c:>18.6e}{a:>18.6e}{c + a:>18.6e}")
    ct = float((col * Vp).sum())
    at = float((ann * Va).sum()) if ann is not None else 0.0
    print(f"{'TOTAL':<14}{'':>5}{ct:>18.6e}{at:>18.6e}{ct + at:>18.6e}")

    print("\nWHERE THE ROUTED STREAM LANDS:")
    if ann is None:
        print("  annulus row rhs_terms/%s/nn_a ABSENT" % row)
    else:
        nzc = np.flatnonzero(col != 0.0)
        nza = np.flatnonzero(ann != 0.0)
        print(f"  column row  nonzero cells: {nzc.tolist()} "
              f"(roles {[roles[i] for i in nzc]})")
        print(f"  annulus row nonzero cells: {nza.tolist()} "
              f"(roles {[roles[i] for i in nza]})")
        print(f"  annulus row identically zero: {not np.any(ann)}")

if routed_guard:
    print("\n=== assert_end_recycle_routed_live GUARD ===")
    try:
        assert_end_recycle_routed_live(
            ba_new, ba_ann_row, path=path, window_ms=(lo, hi)
        )
        print("  zone-correct read: PASSED (guard did not fire)")
    except ValueError as exc:
        print("  zone-correct read: FIRED ->", str(exc).splitlines()[0])
    # positive control: the column-only read on a routed artifact must fire
    try:
        assert_end_recycle_routed_live(
            np.zeros_like(ba_new), ba_ann_row, path=path, window_ms=(lo, hi)
        )
        print("  column-only read : DID NOT FIRE  <-- routed rows NOT live")
    except ValueError as exc:
        print("  column-only read : FIRED (routed rows ARE live) ->",
              str(exc).splitlines()[0])
