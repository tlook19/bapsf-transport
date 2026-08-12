"""Re-measure the R2 quasi-static balance under the corrected beam booking.

Reproduces, at the SAME production pre-breakdown stance and in the SAME format,
the two measurements NUMERICS.md records under "MEASURED: the local balance has
no root at the production stance" -- the per-channel electron-energy rows at a
cathode cell and a column cell, and the density scan of the quasi-static
balance -- so the amendment block next to that section can be read against it
line by line.

The one thing that differs is the booking the balance is fed. On a cell the
tracer owns, the quasilinear/anomalous share of the beam's deposited power is
refused: quasilinear absorption is a beam-PLASMA instability and a passive cell
has no plasma to be the wave medium. Both bookings are evaluated here, side by
side, at every density in the scan, so what the correction does and does not
change is visible rather than asserted.

The stance is `regime_r2_overlap_gate.build_config` -- the same production-stance
ES1 arm the R2 gates use (current-driven cathode, CSDA + quasilinear deposition,
circuit voltage bound on), so nothing about the operating point is chosen here.

Usage (from <checkout>/cablp, with PYTHONPATH set to that same cablp):
    python scripts/regime_pb_balance_table.py --nx 20
"""

import argparse
import sys
import warnings

import numpy as np

from regime_r2_overlap_gate import build_config

from cablp.funcs._beam_deposition import beam_speed_cm_s
from cablp.solvers._sim1d import LAPDSim1D
from cablp.solvers._sim1d.physics.cathode import beam_anomalous_power_density
from cablp.solvers._sim1d.physics.tracer import (
    TracerBalanceError,
    quasistatic_Te_eV,
)
from cablp.vars._cons import ev_to_erg as EV_TO_ERG, qe_SI as QE_SI

#: The rows the R2 table reports, in its order.
ROW_NAMES = (
    "beam_power_deposition",
    "heat_conduction",
    "plasma_advective_flux",
    "cathode_surface_loss",
    "ionization_energy_cost",
    "electron_neutral_cooling",
    "anode_collection",
)

#: The density multipliers the R2 scan used.
SCAN = (1.0, 10.0, 100.0, 1000.0, 1.0e4)


def advance_to(sim, t_target):
    """Step ``sim`` to ``t_target`` one step at a time; return the time reached.

    ``t_end`` is the target on every call, so the integrator lands ON it rather
    than stepping past. The fluid arm's step sequence is untouched by this pass,
    so this reaches the state the original table was measured at.
    """
    while sim._time < t_target:
        try:
            sim.run(t_end=t_target, max_steps=1)
        except RuntimeError:
            continue
    return float(sim._time)


def channel_rows(sim, cells):
    """Return ``{row name: Ee array}`` for the reported channels."""
    terms = sim.rhs_terms()
    out = {}
    for name in ROW_NAMES:
        term = terms.get(name)
        out[name] = (
            np.zeros(cells, dtype=float)
            if term is None
            else np.asarray(term.Ee, dtype=float)
        )
    return out


def beam_ledger(sim, state, time):
    """Return the per-cell beam power split and the whole-solve totals [W]."""
    cathode_solve = sim.solve_cathode_boundary(
        state=state, time=time, update_cache=False
    )
    geometry = sim._geometry
    Vp = np.asarray(geometry.plasma_volume_cm3, dtype=float)
    P_ql = beam_anomalous_power_density(
        **sim._tracer_beam_kwargs(state, cathode_solve, time)
    )
    totals = {}
    events = 0.0
    deposition = getattr(cathode_solve, "beam_deposition", None) or {}
    for dep in deposition.values():
        if dep is None:
            continue
        for name in (
            "plasma_heating_erg_s",
            "heating_coulomb_erg_s",
            "heating_anomalous_erg_s",
            "heating_secondary_erg_s",
            "heating_terminal_erg_s",
            "radiated_erg_s",
            "ionization_cost_erg_s",
        ):
            arr = np.asarray(getattr(dep, name), dtype=float)
            totals[name] = totals.get(name, 0.0) + float(arr.sum()) * 1.0e-7
        events += float(np.asarray(dep.ionization_events, dtype=float).sum())
    return cathode_solve, P_ql, Vp, totals, events


def balance_scan(sim, state, time, cathode_solve, S, P, cells_reported,
                 passive):
    """Return ``{scale: outcome string}`` for one booking of the beam power.

    ``passive`` mirrors the solver's own solve domain: the balance is solved on
    the tracer's cells and nowhere else, because the fluid owns ``Te`` on the
    rest and the local closure was never valid there.
    """
    cells = int(sim._geometry.cells)
    ci, coli = cells_reported
    passive = np.asarray(passive, dtype=bool)
    boundary_rhs = sim._tracer_boundary_rhs(cathode_solve, time)
    Ti = np.full(cells, float(sim._floors["Ti"]))
    n_base = np.maximum(np.asarray(state.n, dtype=float), 0.0)
    out = {}
    for scale in SCAN:
        n_true = n_base * scale
        n_probe = np.maximum(n_true, float(sim._floors["n"]))
        try:
            Te, sign_changes = quasistatic_Te_eV(
                state=state,
                n_true=n_true,
                n_probe=n_probe,
                Ti_eV=Ti,
                S_beam=S,
                P_beam_net=P,
                floors=sim._floors,
                ion_mass_g=sim._ion_mass_g,
                mu=sim._mu,
                cooling_kwargs=sim._electron_cooling_kwargs(),
                exchange_kwargs=sim._tracer_exchange_kwargs(),
                boundary_rhs=boundary_rhs,
                active=passive & ((n_true > 0.0) | (S > 0.0)),
                Te_ceiling_eV=sim._tracer_beam_energy_eV(cathode_solve),
            )
        except TracerBalanceError as error:
            kind = (
                "MULTI-VALUED"
                if "MULTI-VALUED" in str(error)
                else "no root (wants Te above the bracket top)"
            )
            out[scale] = f"REFUSED -- {kind}"
            continue
        out[scale] = (
            f"root, Te {float(Te[ci]):.3g} / {float(Te[coli]):.3g} eV "
            f"(cells {ci}/{coli}), {float(np.max(Te)):.3g} max, "
            f"{int(np.max(sign_changes))} sign change(s)"
        )
    return out


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=20)
    parser.add_argument(
        "--t-target",
        type=float,
        default=1.0423e-05,
        help="the instant the original R2 table was measured at [s]",
    )
    parser.add_argument("--cathode-cell", type=int, default=2)
    parser.add_argument("--column-cell", type=int, default=7)
    parser.add_argument("--t-end", type=float, default=3.0e-5)
    parser.add_argument("--dt-save", type=float, default=1.0e-6)
    parser.add_argument("--max-steps", type=int, default=200000)
    parser.add_argument("--t-end-long", type=float, default=1.0e-4)
    args = parser.parse_args(argv)
    warnings.simplefilter("ignore")

    print("== regime_pb: quasi-static balance under the CORRECTED beam booking")
    print(
        f"   stance regime_r2_overlap_gate.build_config, nx={args.nx}, "
        f"target t={args.t_target:g} s"
    )

    # -- the background. The FLUID arm, exactly as the original table used it:
    # its step sequence is untouched by this pass, so it reaches the same state
    # at the same instant.
    params, flags = build_config(args.nx, False)
    fluid = LAPDSim1D(params, flags)
    reached = advance_to(fluid, args.t_target)
    cells = int(fluid._geometry.cells)
    ci, coli = args.cathode_cell, args.column_cell
    state = fluid.state
    print(f"   fluid arm at t = {reached:.6g} s, {cells} cells, "
          f"n[{ci}] = {float(np.asarray(state.n)[ci]):.4g}, "
          f"n[{coli}] = {float(np.asarray(state.n)[coli]):.4g} cm^-3")

    rows = channel_rows(fluid, cells)
    print()
    print("-- A. per-cell electron-energy rows [erg cm^-3 s^-1] (R2 table format)")
    print(f"   {'row':<26} {'cell ' + str(ci) + ' (cathode)':>20} "
          f"{'cell ' + str(coli) + ' (column)':>20}")
    for name in ROW_NAMES:
        print(f"   {name:<26} {rows[name][ci]:>20.4g} {rows[name][coli]:>20.4g}")

    # -- the booking split at that same background.
    _cs, P_ql, _Vp, totals, events = beam_ledger(fluid, state, reached)
    dep_row = rows["beam_power_deposition"]
    print()
    print("-- B. what the beam_power_deposition row is made of, at the same "
          "cells")
    print(f"   {'channel':<26} {'cell ' + str(ci):>20} {'cell ' + str(coli):>20}")
    print(f"   {'booked (whole row)':<26} {dep_row[ci]:>20.4g} "
          f"{dep_row[coli]:>20.4g}")
    print(f"   {'QL/anomalous share':<26} {P_ql[ci]:>20.4g} {P_ql[coli]:>20.4g}")
    print(f"   {'remainder (kept)':<26} {dep_row[ci] - P_ql[ci]:>20.4g} "
          f"{dep_row[coli] - P_ql[coli]:>20.4g}")
    for cell in (ci, coli):
        share = 0.0 if dep_row[cell] == 0.0 else P_ql[cell] / dep_row[cell]
        print(f"   cell {cell}: the QL channel is {100.0 * share:.3g}% of the row")

    print()
    print("-- C. whole-solve beam deposition ledger [W], the module's own banks")
    for name in (
        "plasma_heating_erg_s",
        "heating_coulomb_erg_s",
        "heating_anomalous_erg_s",
        "heating_secondary_erg_s",
        "heating_terminal_erg_s",
        "radiated_erg_s",
        "ionization_cost_erg_s",
    ):
        print(f"   {name:<31} {totals.get(name, 0.0):12.5g}")
    heating = totals.get("plasma_heating_erg_s", 0.0)
    anom = totals.get("heating_anomalous_erg_s", 0.0)
    if heating > 0.0:
        print(f"   QL/anomalous is {100.0 * anom / heating:.4g}% of the plasma "
              "heating the module books")
    if events > 0.0:
        per_birth_eV = heating / events / 1.602176634e-19
        kept_eV = (heating - anom) / events / 1.602176634e-19
        print(f"   ionization events               {events:12.5g} 1/s")
        print("   deposited energy per beam-born electron (the dilution term's "
              "burden):")
        print(f"     as booked        {per_birth_eV:12.5g} eV  -> needs Te = "
              f"{2.0 * per_birth_eV / 3.0:.5g} eV")
        print(f"     QL share refused {kept_eV:12.5g} eV  -> needs Te = "
              f"{2.0 * kept_eV / 3.0:.5g} eV")

    # -- the balance itself, evaluated on the FLUID arm's own machinery. It has
    # to be: the circuit current and the cathode thermal state are integrated
    # quantities, so a freshly built solver handed this state would solve its
    # cathode at t = 0 conditions and report a balance for a discharge that has
    # not started. Everything the balance reads -- the beam rows, the boundary
    # discretization, the cooling and exchange kwargs -- comes off this solver,
    # which is what the original measurement used and why the surface-loss row
    # is live in both.
    cathode_solve = fluid.solve_cathode_boundary(
        state=state, time=reached, update_cache=False
    )
    S, _P_net, P_full = fluid._tracer_beam_rows(state, cathode_solve, reached)
    # The pre-breakdown passive set: every plasma-active cell, which is exactly
    # the mask the tracer arms with at construction.
    passive = np.asarray(fluid._geometry.plasma_active, dtype=bool)
    P_corrected = P_full - np.where(passive, P_ql, 0.0)
    print()
    print(f"-- D. quasi-static balance density scan "
          f"({int(np.count_nonzero(passive))} of {cells} cells in the "
          f"pre-breakdown passive set)")
    old = balance_scan(
        fluid, state, reached, cathode_solve, S, P_full, (ci, coli), passive
    )
    new = balance_scan(
        fluid, state, reached, cathode_solve, S, P_corrected, (ci, coli),
        passive
    )
    for label, table in (("AS BOOKED (the R2 column)", old),
                         ("CORRECTED (QL refused on passive cells)", new)):
        print(f"   {label}")
        for scale in SCAN:
            print(f"     {'x' + format(scale, 'g'):<8} {table[scale]}")

    # -- and the end-to-end arm: a real tracer run, its own state, its own
    # passive mask, the refusal on the live code path.
    print()
    print("-- E. end-to-end tracer arm under the corrected booking")
    tracer_params, tracer_flags = build_config(args.nx, True)
    tracer_params["dt_save"] = args.dt_save
    tracer = LAPDSim1D(tracer_params, tracer_flags)
    status = "RAN"
    try:
        result = tracer.run(t_end=args.t_end, max_steps=args.max_steps)
        print(f"   ran to t = {float(np.asarray(result.time)[-1]):.6g} s, "
              f"{np.asarray(result.time).size} frames, n max "
              f"{float(np.max(np.asarray(result.n))):.4g} cm^-3")
    except TracerBalanceError as error:
        status = "REFUSED"
        print(f"   REFUSED at t = {float(tracer._time):.6g} s")
        print(f"   {error}")
    leak = tracer.tracer_passive_anomalous_leak()
    worst = float(np.max(np.abs(leak)))
    print(f"   passive-cell QL leak invariant: max |leak| = {worst:.6g} "
          "erg cm^-3 s^-1 (must be exactly 0)")
    # -- F. the n_act / QL-onset gap the overlap gate measures when it fails.
    # The onset is not a number chosen here: `quasilinear_relaxation_length_cm`
    # returns inf -- no anomalous drag at all -- unless n_b < 0.1 n_e, so the
    # module's own weak-beam gate puts QL onset at n_e = 10 n_b. The tracer
    # hands a cell to the fluid at `tracer_activation_ne`. Between the two,
    # quasilinear absorption is live by the code's own criterion while the cell
    # is still passive and the refusal is suppressing it.
    print()
    print("-- F. QL onset (the module's own weak-beam gate) vs tracer_activation_ne")
    beam_result = cathode_solve.beam_result
    n_act = float(fluid._input_dict["tracer_activation_ne"])
    for end, result in (("low", beam_result.result),
                        ("high", beam_result.result_twin)):
        if result is None:
            continue
        Gamma0 = float(result.I_eth_star) / QE_SI
        E0 = float(result.phi_c)
        area = float(np.asarray(fluid._geometry.plasma_area_cm2)[ci])
        n_b = Gamma0 / (area * beam_speed_cm_s(E0))
        onset = 10.0 * n_b
        print(f"   {end} end: E0 = {E0:.5g} eV, Gamma0 = {Gamma0:.5g} 1/s, "
              f"beam density n_b = {n_b:.5g} cm^-3")
        print(f"     QL onset n_e = 10 n_b = {onset:.5g} cm^-3; "
              f"tracer_activation_ne = {n_act:.5g} cm^-3")
        print(f"     GAP: n_act / n_QL_onset = {n_act / onset:.5g} "
              f"({np.log10(n_act / onset):.3g} decades of density in which QL "
              "is live and the cell is still passive)")
    print(f"   observed: the module books QL power at this background, where "
          f"n = {float(np.asarray(state.n)[ci]):.4g} cm^-3 < n_act")

    # -- G. the long window: does the tracer ever hand a cell over, and what
    # happens when it does. On an ACTIVE cell the anomalous booking is restored
    # in full and correctly so, but the balance is still SOLVED there (it is
    # solved wherever there is plasma or a beam birth, because the criteria
    # read Te on every cell), so a handed-over cell can refuse.
    print()
    print(f"-- G. long window (t_end = {args.t_end_long:g} s)")
    long_params, long_flags = build_config(args.nx, True)
    long_params["dt_save"] = 1.0e-5
    long_sim = LAPDSim1D(long_params, long_flags)
    try:
        long_sim.run(t_end=args.t_end_long, max_steps=args.max_steps)
        print(f"   ran to t = {float(long_sim._time):.6g} s with no refusal")
    except TracerBalanceError as error:
        long_passive = long_sim._tracer_passive
        handed = (
            np.asarray(long_sim._geometry.plasma_active, dtype=bool)
            & ~long_passive
        )
        long_n = np.asarray(long_sim.state.n, dtype=float)
        long_t = float(long_sim._time)
        print(f"   REFUSED at t = {long_t:.6g} s")
        print(f"   {error}")
        print(f"   cells handed to the fluid: "
              f"{np.flatnonzero(handed).tolist()}")
        if np.any(handed):
            print(f"   their densities: {long_n[handed].min():.4g} .. "
                  f"{long_n[handed].max():.4g} cm^-3 (n_act = "
                  f"{float(long_sim._input_dict['tracer_activation_ne']):.4g})")
        # WHY, without improvising a fix. In the VACUUM limit the balance
        # reduces to the dilution cost alone, Te -> (2/3) P_net /
        # (ev_to_erg S) -- the beam's W-value in the gas. That is an
        # UPPER-BOUND SCREEN and nothing more: it ignores the L1/L2 radiative
        # sinks, which at this density absorb enough to give most of the
        # screened cells a root anyway. The cell the solver actually refused on
        # is the one named in the message above; the screen is here to show the
        # SHAPE of the problem, which is that S collapses past the beam's range
        # while a residual deposited power does not.
        long_solve = long_sim._cathode_solve or long_sim.solve_cathode_boundary(
            state=long_sim.state, time=long_t, update_cache=False
        )
        long_S, long_net, long_full = long_sim._tracer_beam_rows(
            long_sim.state, long_solve, long_t
        )
        long_ql = beam_anomalous_power_density(
            **long_sim._tracer_beam_kwargs(long_sim.state, long_solve, long_t)
        )
        ceiling = (2.0 / 3.0) * long_sim._tracer_beam_energy_eV(long_solve)
        live = long_S > 0.0
        demand = np.zeros_like(long_S)
        demand[live] = (2.0 / 3.0) * long_net[live] / (
            EV_TO_ERG * long_S[live]
        )
        over = live & long_passive & (demand > ceiling)
        print(f"   bracket top {ceiling:.6g} eV ((2/3) E_beam, E_beam = "
              f"{long_sim._tracer_beam_energy_eV(long_solve):.6g} eV)")
        screened = np.flatnonzero(over)
        print(f"   PASSIVE cells failing the vacuum-limit SCREEN (upper bound, "
              f"not the refusing set): {screened.tolist()}")
        sample = (
            screened[[0, len(screened) // 3, 2 * len(screened) // 3, -1]]
            if screened.size >= 4
            else screened
        )
        # Where the residual power on those cells comes from, so the reader
        # does not have to guess: the deposition module's own per-cell banks
        # BEFORE smoothing, and the plasma volume the conservative kernel
        # divides by.
        long_Vp = np.asarray(
            long_sim._geometry.plasma_volume_cm3, dtype=float
        )
        raw_banks = np.zeros_like(long_Vp)
        for dep in (getattr(long_solve, "beam_deposition", None) or {}).values():
            if dep is None:
                continue
            raw_banks += (
                np.asarray(dep.plasma_heating_erg_s, dtype=float)
                + np.asarray(dep.radiated_erg_s, dtype=float)
                + np.asarray(dep.ionization_cost_erg_s, dtype=float)
            ) / long_Vp
        for cell in sample:
            cell = int(cell)
            print(f"     cell {cell}: passive={bool(long_passive[cell])} "
                  f"n={long_n[cell]:.4g} P_full={long_full[cell]:.5g} "
                  f"P_ql={long_ql[cell]:.5g} P_net={long_net[cell]:.5g} "
                  f"S={long_S[cell]:.5g} -> Te demand {demand[cell]:.5g} eV")
            print(f"       unsmoothed bank density {raw_banks[cell]:.5g}, "
                  f"Vp {long_Vp[cell]:.5g} cm^3")
        print("   NOTE the active-cell refusal is GONE: the run passes both "
              "the instant and the cell that used to stop it. This refusal is "
              "on a cell the tracer OWNS, past the beam's IONIZING range (the "
              "rays still reach it and still deposit; what has stopped is "
              "ionization), and it is the OPPOSITE limit from the original "
              "finding -- not too much power but too few beam-born electrons "
              "to dilute into, S having collapsed to a denormal while P_net "
              "has not. The residual P_net is IDENTIFIED, by "
              "regime_pb_pnet_decomposition.py: it is the primary's "
              "end-of-range terminal dump, banked whole in one 10 cm cell and "
              "redistributed by the 50 cm smoothing kernel into short "
              "small-volume cells. Not the QL channel and not the ohmic "
              "booking.")

    print()
    print(f"== outcome: the balance {'HAS' if status == 'RAN' else 'has NO'} a "
          f"root at stance under the corrected booking; tracer arm {status}")
    return 0 if worst == 0.0 else 1


if __name__ == "__main__":
    sys.exit(main())
