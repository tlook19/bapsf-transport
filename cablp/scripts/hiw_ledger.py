"""Hot-channel conservation ledger and landing-fraction table for the
internal-wall flag (neutral_hot_internal_wall).

LEDGER (the probe_q4 arithmetic). Under neutral_two_zone the cold books are
nn on the column volume Vp and nn_a on the annulus V_ann = Vm - Vp, and the
saved rhs_terms rows are POST-mask, which is the point: the residual is
whatever the plasma-topology mask deleted.

    cx debit          sum over cells of  neutral_cx_channel/nn   * Vp
    hot cold return   sum of  neutral_hot_channel/{nn * Vp, nn_a * V_ann}
    hot ionization    sum of  neutral_hot_channel/n * Vp   (leaves the
                      neutral books legitimately: an in-flight ionization
                      is a plasma source)

    R_neutral = cx debit + hot cold return
    R_total   = R_neutral + hot ionization

R_total is the number that must close: every hot atom is born out of the cold
gas and ends either back in it (wall) or in the plasma (in-flight ionization);
re-CX replaces the atom it consumes and moves no inventory.

    hiw_ledger.py RUN.h5 [t_lo_ms t_hi_ms]

TABLE. With the run's own geometry, the isotropic flight kernel is rebuilt
with the wall off and on, and the per-birth-cell landing mass falling over
plasma-dead cells is tabulated for the cells near the cathode disc.
"""
import json
import sys

import h5py
import numpy as np

sys.path.insert(
    0, "/home/trloo/bapsf/bapsf-transport/.claude/worktrees/agent-ab01a8a131b1e6369/cablp"
)

from cablp.solvers._sim1d import LAPDSim1D  # noqa: E402
from cablp.solvers._sim1d.physics.hot_neutrals import (  # noqa: E402
    ballistic_flight_kernels,
    flight_wall_bounds,
)

path = sys.argv[1]
lo = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
hi = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0e9

with h5py.File(path, "r") as f:
    params = json.loads(f.attrs["params_json"])
    flags = json.loads(f.attrs["flags_json"])
    t0 = float(f.attrs["t_breakdown_trigger"])
    t_ms = (f["time"][:] - t0) * 1e3
    m = (t_ms >= lo) & (t_ms <= hi)
    Vp = f["geometry/plasma_volume_cm3"][:]
    Vm = f["geometry/neutral_volume_cm3"][:]
    Va = np.maximum(Vm - Vp, 0.0)
    roles = [r.decode() if isinstance(r, bytes) else str(r)
             for r in f["geometry/cell_role"][:]]

    def row(term, name, vol):
        key = f"rhs_terms/{term}/{name}"
        if key not in f:
            return np.zeros_like(Vp)
        return np.mean(f[key][:][m], axis=0) * vol

    cx_debit = float(row("neutral_cx_channel", "nn", Vp).sum())
    hot_cold = float(
        row("neutral_hot_channel", "nn", Vp).sum()
        + row("neutral_hot_channel", "nn_a", Va).sum()
    )
    hot_ion = float(row("neutral_hot_channel", "n", Vp).sum())
    hot_wall_births = (
        float(np.mean(f["hot_wall"][:][m], axis=0).dot(Vp))
        if "hot_wall" in f else float("nan")
    )
    hot_ion_births = (
        float(np.mean(f["hot_ionized"][:][m], axis=0).dot(Vp))
        if "hot_ionized" in f else float("nan")
    )
    hot_recx_births = (
        float(np.mean(f["hot_recx"][:][m], axis=0).dot(Vp))
        if "hot_recx" in f else float("nan")
    )
    end_fold = (
        float(np.mean(f["hot_end_fraction"][:][m], axis=0).max())
        if "hot_end_fraction" in f else float("nan")
    )

wall = bool(flags.get("neutral_hot_internal_wall", False))
print(f"artifact : {path}")
print(f"window   : {lo}-{hi} ms after breakdown  ({int(m.sum())} saves, "
      f"t_ms {t_ms[m].min():.4f}..{t_ms[m].max():.4f})")
print(f"stance   : neutral_hot_internal_wall={wall}, "
      f"neutral_two_zone={flags.get('neutral_two_zone')}, "
      f"neutral_energy={flags.get('neutral_energy')}, "
      f"cathode_neutral_jet={params.get('cathode_neutral_jet')}")

print("\n=== HOT-CHANNEL PARTICLE LEDGER [atoms/s], window-averaged, "
      "ALL cells x volumes (POST-mask rows) ===")
print(f"  cx debit          (cx.nn * Vp)                 {cx_debit: .6e}")
print(f"  hot cold return   (hot.nn*Vp + hot.nn_a*V_ann) {hot_cold: .6e}")
print(f"  R_neutral         = cx debit + cold return     {cx_debit + hot_cold: .6e}")
print(f"  hot ionization    (hot.n * Vp)                 {hot_ion: .6e}")
print(f"  R_total           = R_neutral + ionization     "
      f"{cx_debit + hot_cold + hot_ion: .6e}")
print(f"  |R_total| / |cx debit|                         "
      f"{abs(cx_debit + hot_cold + hot_ion) / max(abs(cx_debit), 1e-300): .6e}")
print("\n  birth-cell fate readings (diagnostics, pre-mask) [atoms/s]:")
print(f"    hot_wall * Vp     {hot_wall_births: .6e}")
print(f"    hot_ionized * Vp  {hot_ion_births: .6e}")
print(f"    hot_recx * Vp     {hot_recx_births: .6e}   (replaces, moves no inventory)")
print(f"    max per-cell hot_end_fraction  {end_fold: .6e}")

# ---------------------------------------------------------------- the table
geom = LAPDSim1D(dict(params), dict(flags)).geometry
dead = ~np.asarray(geom.plasma_active, dtype=bool)
k_off = ballistic_flight_kernels(geom, internal_wall=False)
k_on = ballistic_flight_kernels(geom, internal_wall=True)
zlo, zhi, clo, chi = flight_wall_bounds(geom, internal_wall=True)
closed = np.flatnonzero(~np.asarray(geom.plasma_open, dtype=bool))
absorbing = np.flatnonzero(np.asarray(geom.plasma_absorbing, dtype=bool))

print(f"\n=== FLIGHT KERNEL, this run's geometry ({geom.cells} cells) ===")
print(f"closed plasma faces {closed.tolist()} ; absorbing faces "
      f"{absorbing.tolist()} ; plasma-dead cells "
      f"{np.flatnonzero(dead).tolist()}")
print(f"segments (cell_lo, cell_hi): {sorted(set(zip(clo.tolist(), chi.tolist())))}")
print("\nLANDING MASS OVER PLASMA-DEAD CELLS, per birth cell "
      "(kernel row sums; the row itself always sums to 1)")
print(f"{'cell':>5}{'role':>13}{'z_cm':>10}"
      f"{'dead-landing OFF':>19}{'dead-landing ON':>18}"
      f"{'fold OFF':>14}{'fold ON':>14}")
show = [i for i in range(geom.cells) if i <= max(closed[:2].max(), 6) + 6]
for i in show:
    print(f"{i:>5}{roles[i]:>13}{geom.z_cm[i]:>10.2f}"
          f"{k_off[0][i][dead].sum():>19.6e}{k_on[0][i][dead].sum():>18.6e}"
          f"{k_off[2][i]:>14.6e}{k_on[2][i]:>14.6e}")
live_off = k_off[0][~dead][:, dead].sum()
live_on = k_on[0][~dead][:, dead].sum()
print(f"\nsummed over ALL live birth cells: dead-landing mass "
      f"OFF {live_off:.6e}  ON {live_on:.6e}")
print(f"summed over ALL dead birth cells: live-landing mass "
      f"OFF {k_off[0][dead][:, ~dead].sum():.6e}  "
      f"ON {k_on[0][dead][:, ~dead].sum():.6e}")
print("row-sum closure: OFF max|1-sum| "
      f"{np.abs(k_off[0].sum(axis=1) - 1).max():.3e}  ON "
      f"{np.abs(k_on[0].sum(axis=1) - 1).max():.3e}")
