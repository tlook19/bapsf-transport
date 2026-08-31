"""Build gates for the electron drift-transport + EMF-work operator (edt).

**These gates were registered BEFORE the operator was implemented** and are not
moved after seeing results (AGENTS.md, "Briefs and reports"). Each names its
QUANTITY, its MEASUREMENT SITE, and its FIXTURE. Gates 2-4 and 6 are properties
of a SAVED state and of the advisor consult's own algebra, so they were
measurable before the solver-side code existed at all;
``scripts/edt_consult_pins.py`` is the standalone evaluator that measured them
at the unmodified base commit, and this suite checks the implementation against
those same readings.

Run from the checkout root::

    PYTHONPATH=<checkout> python scripts/verify_sim1d_edt.py

Exit 0 = every gated statement passed. A failure is a DELIVERABLE: it is
printed with its numbers and the suite exits 1. Never relax a tolerance here to
make a gate pass.

--------------------------------------------------------------------------
GATE REGISTRY
--------------------------------------------------------------------------

**G1 -- bit-inertness with the flag off.**
  QUANTITY: the accepted-step trajectory, the golden config identity, and the
  RHS term rows.
  SITE: ``scripts/golden_digest_gate.py`` (4,000-step chain digest, all five
  checkpoints and the final digest) and ``scripts/edt_bitinert_ab.py``.
  FIXTURE: the golden config at nx=60 for the digest; ``default_config()``
  for the moment and kinetic-DVM A/B routes.
  PASS: every checkpoint and the final digest unchanged from the committed
  reference; the config identity moves ONLY by the addition of the three new
  keys, proven by a strip control that reproduces the base identity
  bit-for-bit THROUGH THE GATE'S OWN EXPRESSION (the two golden references
  legitimately carry different identities, so each must be computed through
  its own); both routes row-by-row bit-identical to base with only the new
  all-zero rows one-sided. The digest and A/B legs run outside this suite;
  the strip control is checked here.

**G2 -- the volume identity.**
  QUANTITY: ``total - (boundary_in - boundary_out + W_EMF)``, relative to the
  larger side.
  SITE: the operator's own named rows on ONE accepted step.
  FIXTURE: the golden config at nx=60, and the ES1 source region on
  ``scripts/mgcr1_confirm.h5``'s saved state.
  PASS: <= 1e-10 relative on every arm of the bracket, at a state where the
  operator is NOT vacuous (a zero-current state satisfies the identity
  trivially and would gate nothing).

**G3 -- the cathode-face handshake.**
  QUANTITY: the drift enthalpy-plus-thermal-force influx at the cathode face,
  ``2.21 T_e Gamma_d A``, window-mean over 0.1-20.1 ms.
  SITE: ``edt_cathode_face_handshake_W``, via the standalone evaluator on the
  saved state.
  FIXTURE: ``scripts/mgcr1_confirm.h5`` (data in hand; no new run).
  PASS: +14.8 kW within the consult's bracket-A/B spread, which the consult
  puts below 15 %; this suite gates at 15 % ROW-RELATIVE.

**G4 -- the compression piece.**
  QUANTITY: the pressure-drift work summed over the gap cells, window-mean
  over 0.1-20.1 ms, reported ROW-RELATIVE and throughput-normalized
  (AGENTS.md, "Negative controls gate on the ROW-RELATIVE normalization").
  SITE: ``edt_pressure_drift_work_W``.
  FIXTURE: ``scripts/mgcr1_confirm.h5``.
  PASS: +13.6 kW within 15 % ROW-RELATIVE. The throughput-normalized figure is
  REPORTED, never gated -- a piece that is ~4 % of P_prim reads as "small" on
  that normalization however wrong it is.

**G5 -- the J = 0 limit (negative control at the statement level).**
  QUANTITY: every cell of the operator's total row.
  SITE: the ``electron_drift_transport`` RHS term, and the operator called
  directly at zero current.
  FIXTURE: ``default_config()`` with the plasma on and the drive off, the flag
  ARMED; and a LIVE discharge state with the currents set to zero.
  PASS: exactly zero on every cell -- not "small", zero. Both forms are
  checked because the configuration form alone would only exercise the
  "no solve to read" guard and would say nothing about the arithmetic.

**G6 -- the afterglow clause (REPORTED, NOT GATED).**
  QUANTITY: the operator's net over the source region during afterglow.
  SITE: the operator's total row.
  FIXTURE: ``scripts/mgcr1_confirm.h5`` at t = 26 ms AND over the whole
  afterglow window (t > 20.1 ms). Both, because the consult's -0.5...-0.6 kW
  belongs to a 213 A tail and 213 A is the afterglow WINDOW's mean loop
  current -- at the 26 ms instant the loop carries ~12 A. The clause exists so
  that "the term is zero in afterglow" is never stated flatly.

Companion gates that are NOT this suite's to run, and where they live: smoke
(``scripts/smoke_sim1d.py``), the DVM suite (``verify_sim1d_k2_dvm.py``), the
digest gate (``scripts/golden_digest_gate.py``), the snapshot delta
(``scripts/edt_snapshot_delta.py``) and the A/B bit-inertness reader
(``scripts/edt_bitinert_ab.py``).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from cablp.solvers._sim1d import LAPDSim1D, default_config
from cablp.solvers._sim1d.physics.sources import (
    electron_drift_transport_rhs as _operator,
)

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from b5cj_bitinert_ab import step_once  # noqa: E402
from baseline_sim1d import build_baseline_config  # noqa: E402
from edt_consult_pins import (  # noqa: E402
    ANODE_HANDSHAKE_CHOICES,
    CHARGE_DEATH_CHOICES,
    SavedGeometry,
    _window_mean_rows,
    evaluate,
)

#: The fixture the consult measured its pins on. Data in hand: no new run.
#: Resolved against this script's own directory rather than the working
#: directory, because run artifacts live beside the scripts and a worktree's
#: copy of ``scripts/`` does not carry them -- pass ``--h5`` there.
DEFAULT_H5 = str(Path(__file__).resolve().parent / "mgcr1_confirm.h5")

#: The consult's window, in seconds.
WINDOW = (1.0e-4, 2.01e-2)

#: The base commit's golden config identity, measured at 75a2fa1 through the
#: digest gate's OWN expression. G1's strip control must reproduce it.
BASE_CONFIG_IDENTITY = (
    "8974b3ec46a944947e6f080ef48e973ecaaf51163e979d14b874fdb02f57563c"
)

#: The keys this branch adds, by namespace, for G1's strip control.
ADDED_PARAMS = ("electron_drift_charge_death", "electron_drift_anode_handshake")
ADDED_FLAGS = ("electron_drift_transport",)

#: Accepted steps the golden-config leg of G2 walks. A cost knob, not physics:
#: the operator is non-vacuous from step 1 there (the pre-breakdown cathode
#: solve already carries a current), so this only buys a richer state.
GOLDEN_STEPS = 200

#: The identity's bar, as registered.
IDENTITY_TOLERANCE = 1e-10

#: G3 and G4's pins and their bar, as registered.
G3_TARGET_KW, G4_TARGET_KW, PIN_TOLERANCE = 14.8, 13.6, 0.15


class Report:
    """Collects gate outcomes so one failure does not hide the others."""

    def __init__(self):
        self.failures = []

    def check(self, gate, ok, line):
        print(f"[{'PASS' if ok else 'FAIL'}] {gate} {line}")
        if not ok:
            self.failures.append(gate)

    def note(self, gate, line):
        print(f"[    ] {gate} {line}")


def _armed_golden(charge_death, anode_handshake):
    params, flags = build_baseline_config({"max_steps_action": "stop"})
    params = dict(params)
    flags = dict(flags)
    flags["electron_drift_transport"] = True
    params["electron_drift_charge_death"] = charge_death
    params["electron_drift_anode_handshake"] = anode_handshake
    return params, flags


def gate1_strip_control(report):
    """The added keys move the golden config identity and nothing else."""
    from golden_digest_gate import DIGEST_PARAM_OVERRIDES, config_identity

    params, flags = build_baseline_config(DIGEST_PARAM_OVERRIDES)
    live = config_identity(params, flags)
    stripped_params = {
        k: v for k, v in params.items() if k not in ADDED_PARAMS
    }
    stripped_flags = {k: v for k, v in flags.items() if k not in ADDED_FLAGS}
    recovered = config_identity(stripped_params, stripped_flags)
    report.note("G1", f"identity on this branch      {live}")
    report.note("G1", f"identity with the keys strip {recovered}")
    report.note("G1", f"identity at base 75a2fa1     {BASE_CONFIG_IDENTITY}")
    report.check(
        "G1",
        recovered == BASE_CONFIG_IDENTITY and live != BASE_CONFIG_IDENTITY,
        "strip control: the identity moves ONLY by the three added keys "
        "(computed through the digest gate's own expression)",
    )


def _identity(rows, support_slice):
    """Return ``(total, boundary_net + W_EMF, relative residual)``."""
    total = float(
        (
            rows["edt_enthalpy_convection_W"]
            + rows["edt_pressure_drift_work_W"]
            + rows["edt_thermal_force_flux_W"]
            + rows["edt_emf_work_W"]
        )[support_slice].sum()
    )
    rhs = (
        rows["edt_boundary_in_W"]
        - rows["edt_boundary_out_W"]
        + rows["edt_W_EMF_W"]
    )
    scale = max(abs(total), abs(rhs), 1.0)
    return total, rhs, abs(total - rhs) / scale


def gate2_golden(report, steps):
    """The volume identity on the golden config at nx=60, every arm."""
    for charge_death in CHARGE_DEATH_CHOICES:
        for anode_handshake in ANODE_HANDSHAKE_CHOICES:
            params, flags = _armed_golden(charge_death, anode_handshake)
            sim = LAPDSim1D(input_dict=params, input_flags=flags)
            for _ in range(steps):
                step_once(sim)
            rows = sim._electron_drift_rows
            spec = sim._electron_drift
            support = slice(spec["launch_cell"], spec["anode_face"])
            total, rhs, rel = _identity(rows, support)
            arm = f"[{charge_death}/{anode_handshake}]"
            report.check(
                "G2",
                rel <= IDENTITY_TOLERANCE and total != 0.0,
                f"golden nx={sim.geometry.cells} {arm} after {steps} steps: "
                f"total={total:.6e} W, boundary+W_EMF={rhs:.6e} W, "
                f"relative residual {rel:.3e} (bar {IDENTITY_TOLERANCE:.0e}), "
                f"non-vacuous={total != 0.0}",
            )


def gate2_es1(report, geom, h5):
    """The volume identity on the ES1 source region, from the saved state."""
    t = h5["time"][:]
    index = int(np.argmin(np.abs(t - 1.0e-2)))
    cd = h5["cathode_diagnostics"]
    for charge_death in CHARGE_DEATH_CHOICES:
        for anode_handshake in ANODE_HANDSHAKE_CHOICES:
            res = evaluate(
                geom,
                h5["Te"][index, :],
                h5["n"][index, :],
                float(cd["circuit_I_loop"][index]),
                float(cd["source_I_eth_star"][index]),
                charge_death,
                anode_handshake,
            )
            c, last = geom.cathode_cell, geom.last_source_cell
            total = float(res["total_W"][c : last + 1].sum())
            rhs = (
                res["cathode_face_flux_W"]
                + res["cathode_face_work_W"]
                - res["anode_face_flux_W"]
                - res["anode_face_work_W"]
                + res["W_EMF_W"]
            )
            scale = max(abs(total), abs(rhs), 1.0)
            rel = abs(total - rhs) / scale
            arm = f"[{charge_death}/{anode_handshake}]"
            report.check(
                "G2",
                rel <= IDENTITY_TOLERANCE and total != 0.0,
                f"ES1 nx={geom.cells} at t={t[index] * 1e3:.3f} ms {arm}: "
                f"total={total:.6e} W, boundary+W_EMF={rhs:.6e} W, "
                f"relative residual {rel:.3e} (bar {IDENTITY_TOLERANCE:.0e}), "
                f"non-vacuous={total != 0.0}",
            )


def gates34(report, geom, h5):
    """The cathode-face handshake and the compression piece, window-mean."""
    cd = h5["cathode_diagnostics"]
    t = h5["time"][:]
    sel = np.flatnonzero((t >= WINDOW[0]) & (t <= WINDOW[1]))
    throughput_W = float(np.nanmean(cd["source_P_prim"][sel]))
    c, last = geom.cathode_cell, geom.last_source_cell
    spreads = {}
    for charge_death in CHARGE_DEATH_CHOICES:
        rows = _window_mean_rows(
            h5, geom, WINDOW[0], WINDOW[1], charge_death, "export_counts"
        )
        handshake_kW = rows["cathode_face_flux_W"] * 1e-3
        compression_kW = (
            float(rows["pressure_work_W"][c + 1 : last + 1].sum()) * 1e-3
        )
        spreads[charge_death] = (handshake_kW, compression_kW)
        if charge_death != "cell_1":
            continue
        for gate, value, target, label in (
            ("G3", handshake_kW, G3_TARGET_KW, "cathode-face handshake"),
            ("G4", compression_kW, G4_TARGET_KW, "compression piece"),
        ):
            row_relative = abs(value - target) / abs(target)
            normalized = value * 1e3 / throughput_W
            report.check(
                gate,
                row_relative <= PIN_TOLERANCE,
                f"{label}: {value:+.3f} kW against the consult's "
                f"{target:+.1f} kW -- ROW-RELATIVE {row_relative:.4f} "
                f"(bar {PIN_TOLERANCE:.2f}); throughput-normalized "
                f"{normalized:.4f} of P_prim {throughput_W * 1e-3:.1f} kW "
                "(reported, not gated)",
            )
    a, b = spreads["cell_1"], spreads["cell_2"]
    report.note(
        "G3",
        f"bracket A/B spread on the handshake: {a[0]:+.3f} / {b[0]:+.3f} kW "
        f"({abs(b[0] - a[0]) / abs(a[0]):.4f} relative)",
    )
    report.note(
        "G4",
        f"bracket A/B spread on the compression: {a[1]:+.3f} / {b[1]:+.3f} kW",
    )


def gate5(report):
    """J = 0: the operator is exactly zero, by configuration and by arithmetic."""
    params, flags = default_config()
    params = dict(params)
    flags = dict(flags)
    flags["electron_drift_transport"] = True
    flags["cathode_coupling"] = False
    flags["Plasma"] = True
    sim = LAPDSim1D(input_dict=params, input_flags=flags)
    step_once(sim)
    term = sim.rhs_terms()["electron_drift_transport"]
    nonzero = int(np.count_nonzero(term.Ee))
    report.check(
        "G5",
        nonzero == 0,
        "no drive (cathode_coupling off), flag ARMED: the term's Ee row has "
        f"{nonzero} non-zero cells of {sim.geometry.cells} -- exactly zero "
        "required",
    )

    # The configuration form above exercises only the "no solve to read a
    # current from" guard. The arithmetic statement is the one that matters:
    # on a LIVE state, with the currents set to zero, every cell must still be
    # exactly zero.
    params, flags = _armed_golden("cell_1", "export_counts")
    live = LAPDSim1D(input_dict=params, input_flags=flags)
    for _ in range(GOLDEN_STEPS):
        step_once(live)
    rhs, rows = _operator(
        state=live.state,
        floors=live._floors,
        ion_mass_g=live._ion_mass_g,
        geometry=live._plasma_geometry(),
        spec=live._electron_drift,
        I_tot_A=0.0,
        I_beam_A=0.0,
    )
    nonzero = int(np.count_nonzero(rhs.Ee))
    report.check(
        "G5",
        nonzero == 0 and rows["edt_total_W"] == 0.0,
        f"live discharge state at zero current: {nonzero} non-zero cells, "
        f"total={rows['edt_total_W']!r} -- exactly zero required",
    )


def gate6(report, geom, h5, afterglow_lo=2.01e-2):
    """The afterglow clause. Reported, never gated on."""
    t = h5["time"][:]
    cd = h5["cathode_diagnostics"]
    index = int(np.argmin(np.abs(t - 2.6e-2)))
    c, last = geom.cathode_cell, geom.last_source_cell
    for charge_death in CHARGE_DEATH_CHOICES:
        res = evaluate(
            geom,
            h5["Te"][index, :],
            h5["n"][index, :],
            float(cd["circuit_I_loop"][index]),
            float(cd["source_I_eth_star"][index]),
            charge_death,
            "export_counts",
        )
        instant = float(res["total_W"][c : last + 1].sum()) * 1e-3
        rows = _window_mean_rows(
            h5, geom, afterglow_lo, float(t[-1]), charge_death, "export_counts"
        )
        window = float(rows["total_W"][c : last + 1].sum()) * 1e-3
        report.note(
            "G6",
            f"[{charge_death}] instant t={t[index] * 1e3:.3f} ms: "
            f"{instant:+.4f} kW at I_tot="
            f"{float(cd['circuit_I_loop'][index]):.1f} A; afterglow WINDOW "
            f"{afterglow_lo * 1e3:.1f}-{t[-1] * 1e3:.1f} ms: {window:+.4f} kW "
            f"at mean I_tot={rows['_I_tot_mean']:.1f} A "
            "(consult: -0.5...-0.6 kW on a 213 A tail)",
        )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--h5", default=DEFAULT_H5)
    ap.add_argument("--golden-steps", type=int, default=GOLDEN_STEPS)
    ap.add_argument(
        "--registration",
        action="store_true",
        help="print the gate registry and exit without running anything",
    )
    args = ap.parse_args(argv)
    if args.registration:
        print(__doc__)
        return 0

    import h5py

    report = Report()
    gate1_strip_control(report)
    gate2_golden(report, args.golden_steps)
    with h5py.File(args.h5, "r") as h5:
        geom = SavedGeometry(h5)
        geom.check_uniform_area()
        print(
            f"[    ] fixture {args.h5}: cells={geom.cells}, "
            f"cathode_cell={geom.cathode_cell}, anode_face={geom.anode_face}"
        )
        gate2_es1(report, geom, h5)
        gates34(report, geom, h5)
        gate6(report, geom, h5)
    gate5(report)

    print("=" * 78)
    if report.failures:
        print(f"edt build gates: FAILED {sorted(set(report.failures))}")
        return 1
    print("edt build gates: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
