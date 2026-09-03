"""Regime-R2 handoff integrity: tracer leg -> restart export -> full solver.

PRE-REGISTERED INVARIANTS (fixed before the check was implemented):

I1  CONFIG IDENTITY. Stage 2 is built from the exported payload and must pass
    ``results.restart.check_restart_compatibility``: same grid, same packed
    field layout, same structural closure keys. The handoff is a STATE
    TRANSFER, and this is what makes the stored fields mean the same thing on
    both sides of it. Stage 2 differs from stage 1 in exactly one key --
    ``regime_tracer`` off -- which is deliberately NOT structural, because
    "the conducting leg hands off to a differently configured main arm" is the
    case the restart machinery exists for.

I2  FINITENESS. Every conservative and derived field of the resumed state is
    finite (``core.state.assert_finite_state``), and no density is negative.

I3  TWO-PART LEDGER CLOSURE. The closed passive/active interface means the run
    carries two separately closed inventories. Across the handoff the total
    plasma inventory ``sum(n * V_plasma)`` is a RELABELLING, not a source, so
    stage 2's opening inventory must equal stage 1's closing inventory to
    ``LEDGER_RTOL`` relative. The restart carries ``y`` verbatim, so the
    honest expectation is bit-identical and the tolerance is there to make the
    assertion quotable, not to absorb a discrepancy.

I4  THE INTERFACE FLUX IS ZERO BY CONSTRUCTION, NOT BY OVERSIGHT. Every
    passive/active face in the tracer's geometry view is closed (zero plasma
    and heat transmission) AND the Rusanov face flux evaluated there carries
    exactly zero particle and thermal-energy flux, while the same state on the
    BASE geometry -- interface open -- drives a nonzero particle flux through
    those same faces. This is the term the design DROPS (it is the
    seed-transport neglect, NUMERICS.md), and the check exists so a reader can
    see it is zero deliberately.

    Asserted at the FACE, not on a cell RHS row: ``_mask_inactive_rhs`` writes
    literal zeros onto every cell the tracer owns, so a cell-row assertion
    would be true whatever the flux did and would say nothing about whether the
    ACTIVE neighbour lost plasma across the interface.

A fourth outcome, BLOCKED, is not a pass: the tracer can refuse to produce a
number (``TracerBalanceError``) at the operating point. That is reported with
the refusal verbatim.

READ THIS BEFORE QUOTING A PASS FROM THIS SCRIPT
------------------------------------------------
**A PASS here is CONDITIONAL on the Picard refresh cadence, and is not a claim
that the tracer is usable at this stance.** The per-cell quasi-static electron
energy balance has NO ROOT at the production pre-breakdown stance -- parallel
heat conduction and the boundary losses are each an order of magnitude larger
than every local radiative channel, and a per-cell object cannot see conduction
at all. The measurement is in ``_sim1d/NUMERICS.md``, section "MEASURED: the
local balance has no root at the production stance".

This script nevertheless reaches a PASS because ``gamma`` and ``Te`` are
refreshed on a tolerance, not every step: on this run's save lattice the
refreshes happen to land at states where a root still exists, so the refusal is
never evaluated. ``regime_r2_overlap_gate.py``, at a finer ``dt_save`` over the
SAME window and the SAME configuration, refreshes at the failing state and is
BLOCKED. Whether the tracer runs at all therefore depends on the refresh
cadence -- which is itself evidence that the balance is marginal here, not that
this configuration is sound.

So a PASS means: the handoff MACHINERY (config identity, finiteness, ledger
closure, the closed interface) is correct. It does not mean the tracer's ``Te``
closure describes this leg. The printed output repeats this next to the
refresh count so the caveat cannot be separated from the result.

Usage (from <checkout>/cablp, with PYTHONPATH set to that same cablp):
    python scripts/regime_r2_handoff_check.py --nx 20 --t-handoff 2e-5
"""

import argparse
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

from regime_r2_overlap_gate import build_config

from cablp.solvers._sim1d import LAPDSim1D
from cablp.solvers._sim1d.core.state import assert_finite_state
from cablp.solvers._sim1d.physics.flux import rusanov_fluxes
from cablp.solvers._sim1d.physics.tracer import TracerBalanceError
from cablp.solvers._sim1d.results.restart import (
    load_restart_state,
    save_restart_state,
)

#: Relative tolerance on I3. The transfer copies ``y`` verbatim, so this is a
#: quotable bound on an expected-exact identity, not slack for a mismatch.
LEDGER_RTOL = 1.0e-12


def plasma_inventory(sim):
    """Return ``sum(n * V_plasma)`` [particles] -- the whole-column ledger."""
    return float(
        np.sum(
            np.asarray(sim.state.n, dtype=float)
            * np.asarray(sim.geometry.plasma_volume_cm3, dtype=float)
        )
    )


def _faces(sim, geometry):
    """Return the Rusanov face fluxes this state produces on ``geometry``."""
    return rusanov_fluxes(
        state=sim.state,
        floors=sim.floors,
        ion_mass_g=sim.ion_mass_g,
        mu=sim._mu,
        geometry=geometry,
        active_plasma_topology=sim._active_plasma_topology,
        wave_speed=sim._hyperbolic_wave_speed,
        energy_consistent=sim._hyperbolic_energy_consistent,
    )


def print_cadence_caveat(sim):
    """Print, next to the refresh count, what a PASS from this script does not mean.

    The caveat has to travel WITH the result: a printed PASS that a reader
    cannot reconcile against what NUMERICS.md says about the Te closure is
    worse than no output at all, and a gate log outside the repository is not
    a substitute.

    What it says was rewritten when the passive-cell beam power booking was
    corrected (NUMERICS.md, "Corrected beam power booking on passive cells").
    The balance now HAS a root at this stance, so the old wording -- no root
    here, overlap gate BLOCKED -- became false the moment that landed. The
    reasons a PASS is still conditional are different ones, and they are the
    ones printed below.
    """
    refreshes = 0
    census = getattr(sim, "_tracer_census", None)
    if census:
        refreshes = int(census.get("refreshes", 0))
    print(
        f"  CADENCE CAVEAT (refreshes={refreshes}): a PASS below is "
        "CONDITIONAL. Under the corrected passive-cell beam power booking the "
        "quasi-static Te balance DOES have a root at this stance, so this run "
        "is no longer living off a lucky refresh cadence -- but the closure is "
        "still not certified by a PASS here, for two measured reasons. (1) It "
        "refuses further into the ramp: past the beam's IONIZING range the "
        "ionization source collapses while the deposited power does not, so "
        "the dilution denominator goes to zero (measured at t=7.476e-05 s, "
        "cell 32; "
        "regime_pb_balance_table.py (at commit 48be9a4, retired "
        "2026-09-03) section G). (2) The two-sided overlap "
        "gate FAILS at the measured gap between tracer_activation_ne and the "
        "quasilinear onset density, so the two descriptions are not shown to "
        f"agree anywhere. The {refreshes} Picard refresh(es) still set which "
        "states the balance is evaluated at. A PASS therefore certifies the "
        "handoff MACHINERY, not that the tracer's Te closure describes this leg."
    )


def check_interface_row(sim, failures):
    """I4: the dropped interface term is zero by construction, not by oversight.

    Asserted on the FACE FLUXES themselves rather than on a cell RHS row. A
    cell row would be vacuous: ``_mask_inactive_rhs`` writes literal zeros onto
    every cell the tracer owns, so "the row is zero there" is true whatever the
    flux did, and would say nothing about whether the ACTIVE neighbour lost
    plasma across the interface. The face is where the two descriptions meet
    and the only place the invariant has content.
    """
    if sim._tracer is None or not bool(np.any(sim._tracer_passive)):
        failures.append("I4: the tracer owns no cells, so I4 is vacuous here")
        return
    view = sim._plasma_geometry()
    dead = ~sim._plasma_active_mask()
    interfaces = [
        face
        for face in range(1, int(sim.geometry.cells))
        if dead[face - 1] != dead[face]
    ]
    if not interfaces:
        failures.append("I4: no passive/active interface face exists to check")
        return
    for face in interfaces:
        if view.plasma_open[face]:
            failures.append(f"I4: interface face {face} is not closed")
        if view.plasma_transmission[face] != 0.0:
            failures.append(f"I4: interface face {face} passes plasma")
        if view.heat_transmission[face] != 0.0:
            failures.append(f"I4: interface face {face} passes heat")

    closed = _faces(sim, view)
    for face in interfaces:
        for name in ("n", "Ee", "Ei"):
            value = float(np.asarray(getattr(closed, name), dtype=float)[face])
            if value != 0.0:
                failures.append(
                    f"I4: interface face {face} carried {name} flux {value:.6g}; "
                    "a passive/active face must carry none"
                )
    # ANTI-VACUITY: on the BASE geometry the same faces are open, and the same
    # state drives a nonzero particle flux through them. Without this the zeros
    # above could simply mean nothing was flowing anywhere.
    opened = _faces(sim, sim._geometry)
    open_n = np.asarray(opened.n, dtype=float)
    if not np.any(np.abs(open_n[interfaces]) > 0.0):
        failures.append(
            "I4 is vacuous: with the interface OPEN these faces carry no "
            "particle flux either, so closing them proved nothing"
        )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=20)
    parser.add_argument("--t-handoff", type=float, default=2.0e-5)
    parser.add_argument("--t-resume", type=float, default=2.2e-5)
    parser.add_argument("--max-steps", type=int, default=20000)
    args = parser.parse_args(argv)

    warnings.simplefilter("ignore")
    print(
        f"regime_r2 handoff check: nx={args.nx} "
        f"t_handoff={args.t_handoff:g} s t_resume={args.t_resume:g} s"
    )
    failures = []

    # --- stage 1: the conducting leg, via the tracer ---
    params1, flags1 = build_config(args.nx, True)
    stage1 = LAPDSim1D(params1, flags1)
    try:
        stage1.run(t_end=args.t_handoff, max_steps=args.max_steps)
    except TracerBalanceError as error:
        print("  stage 1 (tracer leg): REFUSED to produce a number")
        print(f"    {error}")
        print("regime_r2 handoff check: BLOCKED (tracer leg could not run)")
        return 2
    census = stage1._tracer_census_line()
    if census:
        print(f"  {census}")
    print_cadence_caveat(stage1)
    check_interface_row(stage1, failures)
    inventory_before = plasma_inventory(stage1)
    print(
        f"  stage 1 closed at t={stage1.time:.6g} s, "
        f"inventory {inventory_before:.12g} particles"
    )

    with tempfile.TemporaryDirectory() as tmp:
        payload_path = Path(tmp) / "regime_r2_handoff.h5"
        save_restart_state(payload_path, stage1)
        payload = load_restart_state(payload_path)
        print(
            f"  payload: format ok, t={payload['time']:.6g} s, "
            f"cells={payload['cells']}, fields={list(payload['state_fields'])}"
        )

        # --- stage 2: the full solver, tracer OFF, resumed from the payload ---
        params2, flags2 = build_config(args.nx, False)
        params2["restart_from"] = str(payload_path)
        try:
            # I1: construction runs check_restart_compatibility; an
            # incompatible payload raises here and nowhere later.
            stage2 = LAPDSim1D(params2, flags2)
        except ValueError as error:
            failures.append(f"I1: the resumed stage refused the payload: {error}")
            print("regime_r2 handoff check: FAIL")
            for item in failures:
                print(f"    {item}")
            return 1
        print("  I1 config identity: the payload was accepted by stage 2")
        if stage2._tracer is not None:
            failures.append("I1: stage 2 must resume with the tracer OFF")

        inventory_after = plasma_inventory(stage2)
        rel = abs(inventory_after - inventory_before) / max(
            abs(inventory_before), 1.0
        )
        print(
            f"  I3 ledger: stage 2 opens at {inventory_after:.12g} particles, "
            f"relative change {rel:.3g} (tolerance {LEDGER_RTOL:g})"
        )
        if not rel <= LEDGER_RTOL:
            failures.append(
                f"I3: the handoff moved the plasma inventory by {rel:.6g} "
                "relative; a state transfer must relabel, not source"
            )

        stage2.run(t_end=args.t_resume, max_steps=args.max_steps)
        try:
            assert_finite_state(stage2.state, stage2.derived)
        except ValueError as error:
            failures.append(f"I2: {error}")
        if float(np.min(np.asarray(stage2.state.n, dtype=float))) < 0.0:
            failures.append("I2: the resumed run produced a negative density")
        print(
            f"  I2 finiteness: stage 2 ran to t={stage2.time:.6g} s, "
            f"n in [{float(np.min(stage2.state.n)):.4g}, "
            f"{float(np.max(stage2.state.n)):.4g}] cm^-3"
        )

    if failures:
        print("regime_r2 handoff check: FAIL")
        for item in failures:
            print(f"    {item}")
        return 1
    # The verdict string itself carries the qualifier, so a grep for the
    # result line cannot pick up the PASS and leave the caveat behind.
    print(
        "regime_r2 handoff check: PASS (CONDITIONAL -- handoff machinery only; "
        "see the CADENCE CAVEAT above, the Te closure is not certified here)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
