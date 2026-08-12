"""What the residual beam power past the beam's IONIZING range actually IS.

MEASUREMENT ONLY. This script proposes no remedy and changes no behaviour: it
decomposes the ``Ee`` beam power the tracer's quasi-static balance is handed on
passive cells that the rays still reach and still deposit in, but no longer
IONIZE in, and reports where every erg of it comes from. The question it
answers was opened by ``regime_pb_balance_table.py`` section G, where the
balance refuses past that range because the ionization source has collapsed to
a denormal while the deposited power has not.

"Beyond the ionizing range", not "beyond the range": the output below shows
raw plasma-heating banks on cells 2-39, so the rays reach them. What stops is
ionization, once the primary drops below threshold.

Three decompositions, per cell:

(a) RAW vs SMOOTHED. The deposition module banks an extensive per-cell power
    [erg/s]; ``_beam_ionization_sources`` then divides by the plasma volume and,
    when ``beam_deposition_smoothing_cm > 0``, pushes the density through a
    conservative kernel. Both are printed, so a cell that banks nothing of its
    own and still carries power is visible as such.

(b) KERNEL PROVENANCE. The smoothing is a matrix, so the split is exact rather
    than estimated: the smoothed extensive power at cell ``i`` is
    ``sum_j W[i, j] * raw_ext[j]``, and the share of it contributed by
    ``j != i`` is the fraction that ORIGINATED IN OTHER CELLS. The largest
    contributing source cells are named.

(c) VOLUME. ``Vp`` per cell beside the rest, because the kernel conserves the
    EXTENSIVE power and the balance consumes a DENSITY: the same erg/s landing
    in a smaller cell is a larger source term.

Usage (from <checkout>/cablp, with PYTHONPATH set to that same cablp):
    python scripts/regime_pb_pnet_decomposition.py --nx 20
"""

import argparse
import sys
import warnings

import numpy as np

from regime_r2_overlap_gate import build_config

from cablp.solvers._sim1d import LAPDSim1D
from cablp.solvers._sim1d.core.geometry import gap_cell_indices
from cablp.solvers._sim1d.physics.cathode import (
    _beam_smoothing_matrix,
    beam_anomalous_power_density,
)
from cablp.solvers._sim1d.physics.tracer import TracerBalanceError

#: The cells section G sampled, plus the one the solver refused on.
DEFAULT_CELLS = (13, 22, 31, 32, 40)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=20)
    parser.add_argument("--t-end", type=float, default=1.0e-4)
    parser.add_argument("--max-steps", type=int, default=200000)
    parser.add_argument(
        "--cells", type=int, nargs="*", default=list(DEFAULT_CELLS)
    )
    args = parser.parse_args(argv)
    warnings.simplefilter("ignore")

    print("== regime_pb: what the residual P_net past the IONIZING range IS")
    print("   MEASUREMENT ONLY -- no remedy is proposed or applied here.")
    print(f"   stance regime_r2_overlap_gate.build_config, nx={args.nx}")

    params, flags = build_config(args.nx, True)
    params["dt_save"] = 1.0e-5
    sim = LAPDSim1D(params, flags)
    status = "ran to the window end"
    try:
        sim.run(t_end=args.t_end, max_steps=args.max_steps)
    except TracerBalanceError as error:
        status = "refused"
        print(f"\n   tracer arm REFUSED at t = {float(sim._time):.6g} s")
        print(f"   {error}")
    time = float(sim._time)
    print(f"\n   decomposing at t = {time:.6g} s ({status})")

    geometry = sim._geometry
    Vp = np.asarray(geometry.plasma_volume_cm3, dtype=float)
    smoothing_cm = float(
        sim._input_dict.get("beam_deposition_smoothing_cm", 0.0)
    )
    cathode_solve = sim._cathode_solve or sim.solve_cathode_boundary(
        state=sim.state, time=time, update_cache=False
    )

    # -- the module's own extensive banks, before anything is done to them.
    raw_heat = np.zeros_like(Vp)
    raw_rad = np.zeros_like(Vp)
    raw_cost = np.zeros_like(Vp)
    raw_anom = np.zeros_like(Vp)
    raw_term = np.zeros_like(Vp)
    for dep in (getattr(cathode_solve, "beam_deposition", None) or {}).values():
        if dep is None:
            continue
        raw_heat += np.asarray(dep.plasma_heating_erg_s, dtype=float)
        raw_rad += np.asarray(dep.radiated_erg_s, dtype=float)
        raw_cost += np.asarray(dep.ionization_cost_erg_s, dtype=float)
        raw_anom += np.asarray(dep.heating_anomalous_erg_s, dtype=float)
        raw_term += np.asarray(dep.heating_terminal_erg_s, dtype=float)

    # -- the kernel, and the exact provenance split it implies.
    if smoothing_cm > 0.0:
        W = np.asarray(_beam_smoothing_matrix(geometry, smoothing_cm))
        W = W.toarray() if hasattr(W, "toarray") else np.asarray(W)
    else:
        W = np.eye(int(geometry.cells))
    smoothed_heat_ext = W @ raw_heat
    self_ext = np.diagonal(W) * raw_heat
    foreign_ext = smoothed_heat_ext - self_ext

    # -- what the balance is actually handed, and the reconstruction check.
    S, P_net, P_full = sim._tracer_beam_rows(sim.state, cathode_solve, time)
    P_ql = beam_anomalous_power_density(
        **sim._tracer_beam_kwargs(sim.state, cathode_solve, time)
    )
    ohmic = np.zeros_like(Vp)
    beam_result = cathode_solve.beam_result
    for end, result in ((0, beam_result.result), (-1, beam_result.result_twin)):
        if result is None:
            continue
        gap = np.asarray(gap_cell_indices(geometry, end=end), dtype=int)
        ohmic[gap] += float(result.P_ohmic) * 1.0e7 / Vp[gap] / gap.size
    reconstructed = smoothed_heat_ext / Vp
    passive = sim._tracer_passive

    print(f"   smoothing width {smoothing_cm:g} cm; "
          f"P_ohmic total {sum(float(r.P_ohmic) for r in (beam_result.result, beam_result.result_twin) if r is not None):.5g} W; "
          f"gap cells {np.asarray(gap_cell_indices(geometry, end=0)).tolist()}")

    # -- IDENTIFICATION. If P_full is the smoothed plasma-heating density and
    # nothing else, the residual is named exactly and the question is closed.
    off_gap = np.ones(int(geometry.cells), dtype=bool)
    off_gap[np.asarray(gap_cell_indices(geometry, end=0), dtype=int)] = False
    denom = np.where(np.abs(P_full) > 0.0, np.abs(P_full), 1.0)
    mismatch = np.abs(P_full - reconstructed) / denom
    worst_cell = int(np.argmax(np.where(off_gap, mismatch, 0.0)))
    print(f"   IDENTITY CHECK  P_full == smoothed(plasma_heating)/Vp off the "
          f"gap: worst relative mismatch {float(mismatch[worst_cell]):.3g} "
          f"at cell {worst_cell}")
    print("     (the smoothed radiated and ionization-cost banks cancel the "
          "excitation and cost ROWS exactly, so anything left is the heating "
          "bank alone)")

    print()
    z_cm = np.asarray(geometry.z_cm, dtype=float)
    dz_cm = np.asarray(geometry.length_cm, dtype=float)
    header = (
        f"{'cell':>5} {'pas':>4} {'z [cm]':>9} {'dz':>7} {'Vp [cm^3]':>11} "
        f"{'raw_heat[erg/s]':>16} {'raw_dens':>11} {'smoothed_dens':>14} "
        f"{'P_full':>11} {'foreign %':>10} {'S':>11}"
    )
    print(header)
    print("   " + "-" * (len(header) - 3))
    for cell in args.cells:
        cell = int(cell)
        tot = smoothed_heat_ext[cell]
        foreign_pct = 100.0 * foreign_ext[cell] / tot if tot != 0.0 else 0.0
        print(
            f"{cell:>5} {str(bool(passive[cell]))[:4]:>4} {z_cm[cell]:>9.5g} "
            f"{dz_cm[cell]:>7.4g} {Vp[cell]:>11.5g} "
            f"{raw_heat[cell]:>16.5g} {raw_heat[cell] / Vp[cell]:>11.5g} "
            f"{reconstructed[cell]:>14.5g} {P_full[cell]:>11.5g} "
            f"{foreign_pct:>10.4g} {S[cell]:>11.5g}"
        )
    print(f"   mesh: the kernel width is {smoothing_cm:g} cm and the cells it "
          "redistributes across are these dz, so its reach in CELLS is set by "
          "the local mesh spacing, not by a fixed stencil")

    print()
    print("   where each cell's smoothed power CAME FROM (top source cells, "
          "share of that cell's smoothed extensive power):")
    for cell in args.cells:
        cell = int(cell)
        contrib = W[cell] * raw_heat
        tot = contrib.sum()
        if tot == 0.0:
            print(f"     cell {cell}: nothing reaches it")
            continue
        order = np.argsort(contrib)[::-1][:4]
        parts = ", ".join(
            f"cell {int(j)} {100.0 * contrib[int(j)] / tot:.3g}%"
            for j in order
            if contrib[int(j)] > 0.0
        )
        print(f"     cell {cell}: {parts}")

    print()
    print("   the raw banks that feed the kernel (where the rays actually "
          "stopped):")
    live = np.flatnonzero(raw_heat > 0.0)
    print(f"     cells banking any plasma heating: {live.tolist()}")
    for cell in live:
        cell = int(cell)
        print(f"       cell {cell}: heat {raw_heat[cell]:.5g} erg/s "
              f"(anomalous {raw_anom[cell]:.5g}, terminal {raw_term[cell]:.5g}, "
              f"radiated {raw_rad[cell]:.5g}, cost {raw_cost[cell]:.5g})")
    print(f"     total banked plasma heating "
          f"{float(raw_heat.sum()) * 1.0e-7:.5g} W; kernel conserves it to "
          f"{abs(float(smoothed_heat_ext.sum() / raw_heat.sum()) - 1.0):.3g} "
          "relative")
    terminal_cell = int(np.argmax(raw_term))
    if raw_term[terminal_cell] > 0.0:
        print(f"     the primary's END-OF-RANGE TERMINAL DUMP is banked whole "
              f"in cell {terminal_cell}: {raw_term[terminal_cell]:.5g} erg/s, "
              f"{100.0 * raw_term[terminal_cell] / raw_heat.sum():.4g}% of all "
              f"banked plasma heating, in one cell of dz {dz_cm[terminal_cell]:g} cm")

    print()
    print("   P_ql on the sampled cells (the anomalous channel, for scale): "
          + ", ".join(f"cell {int(c)} {P_ql[int(c)]:.4g}" for c in args.cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
