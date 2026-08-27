"""P3: the per-ray energy ledger closes under ``terminal_nonlocal``.

PRE-REGISTERED BEFORE THE SELECTOR WAS RUN.

THE IDENTITY
------------
For every CSDA ray the solver fires, at the probe state,

    launched  =  along-ray booked  +  walked-to-ledger  +  transmitted

with

* ``launched``          = ``Gamma0 * E0`` = ``I_eth_star * phi_c`` (W -> erg/s
                          is exactly 1e7), the ray's own budget as the cathode
                          solve issued it;
* ``along-ray booked``  = ``plasma_heating + radiated + ionization_cost +
                          anode_intercepted``, everything the column and the
                          electrodes kept;
* ``walked-to-ledger``  = ``end_loss_low + end_loss_high``, which under this
                          selector is the walked TERMINAL population alone;
* ``transmitted``       = ``transmitted_flux * transmitted_energy_eV``, which
                          this selector deliberately leaves OUTSIDE the ledger.

Registered tolerance: 1e-9 relative, the same bar the smoke suite holds the
other two selectors to.

THE CHARGE SIDE is checked in the same pass, because the point of the
selector's second booking is that energy and charge go to different places and
neither is counted twice: the escaping terminal FLUX is reported by the module,
and with ``regime_vessel_node`` armed the node's electron wall current must be
exactly ``qe * (transmitted_flux + terminal_escape_flux)`` summed over rays.

ANTI-VACUITY (registered as part of the check, not added after it passed)
-------------------------------------------------------------------------
A conservation check that cannot fail proves nothing, and the failure this one
exists to catch is specific: a build that walks the terminal population but
books it locally anyway, or one that walks it and then loses it. Both are
simulated by CORRUPTING an otherwise-passing result -- dropping the ledger
term, and inflating one along-ray bank by a relative 1e-6 -- and the same
expression must reject both. If either corruption passes, the check is
declared vacuous and this script exits non-zero whatever the honest arms did.

That is not a formality here. MEASURED: on the FLUID arm the walked term is
identically zero at both stance states -- the column is dense enough that the
terminal population thermalizes before it reaches an end, which is the walk
collapsing onto the local rule as designed -- so the dropped-ledger corruption
is a no-op there and the check reports itself vacuous. The walked term only
exists where the selector was built to act: the TRACER arm's late, thin state.
Both arms are therefore probed, and the corruption is applied to the ray with
the largest walked term rather than to whichever ray came first.

Usage (from <checkout>/cablp, with PYTHONPATH set to that same cablp):
    python scripts/regime_tnl_p3_ledger.py
"""

import argparse
import dataclasses
import sys
import warnings

import numpy as np

from regime_pb_balance_table import advance_to
from regime_r2_overlap_gate import build_config

from cablp.solvers._sim1d import LAPDSim1D
from cablp.vars._cons import ev_to_erg as EV_TO_ERG, qe_SI as QE_SI

#: Registered relative tolerance for the per-ray identity.
RTOL = 1.0e-9


def ray_terms(dep):
    """Return ``(booked, walked, transmitted)`` [erg/s] for one ray."""
    booked = (
        float(np.sum(dep.plasma_heating_erg_s))
        + float(np.sum(dep.radiated_erg_s))
        + float(np.sum(dep.ionization_cost_erg_s))
        + float(dep.anode_intercepted_erg_s)
    )
    walked = float(dep.end_loss_low_erg_s) + float(dep.end_loss_high_erg_s)
    transmitted = (
        float(dep.transmitted_flux) * float(dep.transmitted_energy_eV)
        * EV_TO_ERG
    )
    return booked, walked, transmitted


def residual(dep, launched):
    """Relative departure from the registered identity for one ray."""
    booked, walked, transmitted = ray_terms(dep)
    return abs(booked + walked + transmitted - launched) / launched


def launched_erg_s(solve):
    """Return ``{end index: Gamma0*E0}`` [erg/s] from the cathode solve."""
    beam = solve.beam_result
    out = {}
    for index, result in ((0, beam.result), (-1, beam.result_twin)):
        if result is not None:
            out[index] = float(result.I_eth_star) * float(result.phi_c) * 1.0e7
    return out


def check_state(sim, time, label):
    """Check every ray of one solve; return ``(rows, worst, escaped)``."""
    solve = sim.solve_cathode_boundary(
        state=sim.state, time=time, update_cache=False
    )
    deposition = getattr(solve, "beam_deposition", None) or {}
    budgets = launched_erg_s(solve)
    rows, worst, escaped = [], 0.0, 0.0
    for index, dep in sorted(deposition.items(), key=lambda kv: str(kv[0])):
        if dep is None or index not in budgets:
            continue
        launched = budgets[index]
        if launched <= 0.0:
            continue
        booked, walked, transmitted = ray_terms(dep)
        rel = residual(dep, launched)
        worst = max(worst, rel)
        escaped += walked
        rows.append({
            "label": label, "ray": index, "launched": launched,
            "booked": booked, "walked": walked, "transmitted": transmitted,
            "rel": rel, "flux": float(dep.terminal_escape_flux_per_s),
            "dep": dep,
        })
    return rows, worst, escaped


def report_rows(rows):
    print(f"   {'ray':>5} {'launched':>13} {'booked':>13} {'walked':>13} "
          f"{'transmitted':>13} {'rel resid':>11} {'escape flux':>12}")
    for r in rows:
        print(f"   {r['ray']:>5} {r['launched']:13.6g} {r['booked']:13.6g} "
              f"{r['walked']:13.6g} {r['transmitted']:13.6g} "
              f"{r['rel']:11.3e} {r['flux']:12.6g}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=20)
    parser.add_argument(
        "--t-probe", type=float, nargs="+",
        default=(1.0423e-5, 7.49713e-5),
        help="FLUID-arm states to check: the balance table's stance, and the "
             "instant the shipped selector refuses at",
    )
    parser.add_argument(
        "--t-tracer", type=float, nargs="+", default=(1.0e-5, 2.0e-5, 1.0e-4),
        help="TRACER-arm states to check. The first two are where the walked "
             "terminal population ESCAPES; the last is where the ray no "
             "longer stops at all and there is no terminal population to walk",
    )
    parser.add_argument("--selector", default="terminal_nonlocal")
    args = parser.parse_args(argv)
    warnings.simplefilter("ignore")

    print("== regime_tnl P3: the per-ray ledger under "
          f"beam_product_transport={args.selector!r}")
    print(f"   identity: launched = along-ray booked + walked-to-ledger + "
          f"transmitted, to {RTOL:g} relative")
    print(f"   stance regime_r2_overlap_gate.build_config({args.nx}, False)")

    params, flags = build_config(
        args.nx, False, {"beam_product_transport": args.selector}
    )
    sim = LAPDSim1D(params, flags)

    all_rows, worst, escaped_total = [], 0.0, 0.0
    for t_probe in sorted(args.t_probe):
        reached = advance_to(sim, t_probe)
        rows, arm_worst, escaped = check_state(sim, reached, f"t={reached:g}")
        print()
        print(f"-- FLUID arm, state t = {reached:.6g} s")
        report_rows(rows)
        all_rows.extend(rows)
        worst = max(worst, arm_worst)
        escaped_total += escaped

    # The TRACER arm: the thin late state the selector exists for, and the
    # only place in this scenario where the walked term is nonzero.
    tparams, tflags = build_config(
        args.nx, True, {"beam_product_transport": args.selector}
    )
    tsim = LAPDSim1D(tparams, tflags)
    for t_probe in sorted(args.t_tracer):
        reached = advance_to(tsim, t_probe)
        rows, arm_worst, escaped = check_state(tsim, reached, f"t={reached:g}")
        print()
        print(f"-- TRACER arm, state t = {reached:.6g} s")
        report_rows(rows)
        all_rows.extend(rows)
        worst = max(worst, arm_worst)
        escaped_total += escaped
    print()
    print(f"   worst relative residual over every ray and state: {worst:.3e} "
          f"(registered bar {RTOL:g})")
    closed = worst <= RTOL

    # -- the charge side, on the same rays.
    print()
    print("-- charge: the node's electron wall current vs the module's fluxes")
    vparams, vflags = build_config(
        args.nx, True, {"beam_product_transport": args.selector}
    )
    vflags["regime_vessel_node"] = True
    vsim = LAPDSim1D(vparams, vflags)
    # Read at the state where the walk ESCAPES, or the charge channel it
    # exists to book is zero and the comparison says nothing.
    vreached = advance_to(vsim, max(args.t_tracer[:2] or args.t_tracer))
    vsim._cathode_solve = vsim.solve_cathode_boundary(
        state=vsim.state, time=vreached, update_cache=False
    )
    deposition = getattr(vsim._cathode_solve, "beam_deposition", None) or {}
    flux_t = sum(float(d.transmitted_flux) for d in deposition.values() if d)
    flux_e = sum(
        float(d.terminal_escape_flux_per_s) for d in deposition.values() if d
    )
    I_e = vsim._vessel_electron_wall_current_A()
    expect = QE_SI * max(flux_t + flux_e, 0.0)
    charge_ok = abs(I_e - expect) <= 1e-12 * max(expect, 1.0)
    print(f"   armed={vsim._beam_terminal_wall_charge}  "
          f"transmitted flux {flux_t:.6g} 1/s, terminal escape flux "
          f"{flux_e:.6g} 1/s")
    print(f"   I_e_wall = {I_e:.6g} A, expected qe*(sum) = {expect:.6g} A -> "
          f"{'MATCH' if charge_ok else 'MISMATCH'}")

    # -- ANTI-VACUITY. Corrupt a PASSING result two ways; both must be caught.
    print()
    print("-- anti-vacuity: a broken booking must fail the same expression")
    # The corruption must be applied where the walked term EXISTS, or the
    # dropped-ledger arm is testing nothing (see the docstring's measurement).
    probe = max(all_rows, key=lambda r: (r["walked"], r["launched"]))
    launched = probe["launched"]
    dep = probe["dep"]
    caught = {}
    lost = dataclasses.replace(dep, end_loss_low_erg_s=0.0,
                               end_loss_high_erg_s=0.0)
    caught["walked energy dropped (banked local instead)"] = residual(
        lost, launched
    )
    inflated = dataclasses.replace(
        dep, plasma_heating_erg_s=dep.plasma_heating_erg_s * (1.0 + 1.0e-6)
    )
    caught["one along-ray bank inflated by 1e-6"] = residual(
        inflated, launched
    )
    vacuous = False
    for name, rel in caught.items():
        verdict = "CAUGHT" if rel > RTOL else "MISSED -- CHECK IS VACUOUS"
        if rel <= RTOL:
            vacuous = True
        print(f"   {name:<45} rel {rel:.3e}  {verdict}")

    print()
    escape_seen = escaped_total > 0.0
    print(f"== outcome: ledger {'CLOSES' if closed else 'DOES NOT CLOSE'}; "
          f"charge {'consistent' if charge_ok else 'INCONSISTENT'}; "
          f"anti-vacuity {'ok' if not vacuous else 'FAILED'}")
    print(f"   walked-to-ledger energy seen over the probed states: "
          f"{escaped_total:.6g} erg/s"
          + ("" if escape_seen else
             "   <-- nothing escaped at these states: the identity is real "
             "but the walked term was not exercised here"))
    return 0 if (closed and charge_ok and not vacuous) else 1


if __name__ == "__main__":
    sys.exit(main())
