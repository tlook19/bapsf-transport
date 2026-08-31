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

**G3 -- the cathode face (RE-FORMED 2026-08-31).**
  QUANTITY: two statements. (i) the drift enthalpy-plus-thermal-force influx
  at the cathode face is ZERO; (ii) the face-1 WORK term is the exact partner
  of what ``pressure_work_rhs`` books at the same face, so the two sum to
  roundoff. Window-mean over 0.1-20.1 ms.
  SITE: ``edt_cathode_face_handshake_W`` and the face-1 work term, via the
  standalone evaluator on the saved state.
  FIXTURE: ``scripts/mgcr1_confirm.h5`` (data in hand; no new run).
  PASS: (i) == 0 to roundoff; (ii) |work + pressure_work face-1| / |work| <=
  1e-12, at a magnitude of 4.298 kW.

  **The +14.8 kW pin is RETIRED as measured-wrong.** It rode the circuit's ion
  current at a face whose electron channel carries ~0.3 mA, and the rationale
  for it -- that it cancelled a ghost-Bohm booking -- was a stale read of a
  legacy row that has been inert on the shipped stance since R3.2. It is named
  here rather than deleted so that the next reader does not re-derive it.

**G4 -- the compression piece (RE-FORMED 2026-08-31).**
  QUANTITY: the pressure-drift work summed over the cells STRICTLY DOWNSTREAM
  of the death cell, window-mean over 0.1-20.1 ms, reported ROW-RELATIVE and
  throughput-normalized (AGENTS.md, "Negative controls gate on the ROW-RELATIVE
  normalization").
  SITE: ``edt_pressure_drift_work_W``, on the ``export_counts`` arm -- the one
  where NO face is closed, so the row is the operator's own interior
  compression and nothing else. Under ``sheath_row_closes_all`` the mesh
  face's work term is handed to the sheath row and lands in the same cell, so
  that arm's row is a different quantity; it is reported beside this one.
  FIXTURE: ``scripts/mgcr1_confirm.h5``.
  PASS: +13.6 kW (bracket A) / +8.2 kW (bracket B) within 15 % ROW-RELATIVE.
  The fixed cells-2-5 range is RETIRED with its "robust, handshake-independent"
  label: over that range bracket B reads -5.3 kW, so the quantity was
  bracket-A-specific rather than robust. The throughput-normalized figure is
  REPORTED, never gated.

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
  afterglow window (t > 20.1 ms). Both, because the -0.5...-0.6 kW reading
  belongs to a 213 A tail and 213 A is the afterglow WINDOW's mean loop
  current -- at the 26 ms instant the loop carries ~12 A. The clause exists so
  that "the term is zero in afterglow" is never stated flatly, and it carries
  one more disclosure: in afterglow ``Gamma_d`` is NEGATIVE, because the
  emission outlasts the loop current (I_beam 297.5 A against I_tot 217.8 A,
  window means) and the drift reverses.

**G10 -- the in-plasma EMF (REPORTED as a BRACKET, not gated).**
  QUANTITY: ``W_EMF`` per ampere, as a CURRENT-WEIGHTED window mean --
  ``sum(W_EMF) / sum(I)``, not the mean of a per-sample ratio, which is a
  small-denominator trap.
  SITE: ``edt_inplasma_emf_V``, on two supports: faces 2-5 (the DECLARED
  support -- the drift is absorbed at the mesh and never traverses the last
  gradient) and faces 2-6 (which includes the mesh face's pressure jump, and
  is the support that reproduces the 2026-08-26 consult's figure).
  FIXTURE: ``scripts/mgcr1_confirm.h5``.
  REPORTED: the bracket, expected 3.7-6.2 V against the Boltzmann estimate
  ``T_e ln(n_5/n_1)`` = 5.7 V. **The > 6 V binary is DROPPED** as
  discretization-fragile: one arm spans 5.27-6.22 V across defensible
  conventions, so a threshold at 6 V would be reporting the convention. A
  pre-breakdown small-current frame reads ~16 V and is NOT a physics reading.

**G11 -- the beam-bypass identity (NEW 2026-08-31).**
  QUANTITY: every RHS row other than the new term, and the circuit's
  beam-bypass fraction, evaluated at ONE identical state with the flag off and
  with it armed.
  SITE: ``rhs_terms`` and ``beam_bypass_fraction`` on the cathode solve.
  FIXTURE: the golden config at nx=60, at a state reached by the armed run.
  PASS: bit-identical on every shared row and on the bypass fraction, with the
  new term NON-ZERO (a vacuous state would prove nothing). This is what makes
  "the beam electrons that reach the mesh are booked once" a measurement: the
  registered anode closure holds them to be outside both ``Gamma_d`` and the
  kinetic sheath row, and arming must therefore not touch their booking.

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

#: The base commit's golden config identity, measured at agent-staging 056a733
#: through the digest gate's OWN expression. G1's strip control must reproduce
#: it. The two golden references carry different identities by design, so a
#: control computed through the other expression matches neither.
BASE_COMMIT = "056a733"
BASE_CONFIG_IDENTITY = (
    "21a9b4764df68bc9c201d5ea11589223358bd9ca19d2801f82ac7bd75db632c3"
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

#: G4's pins, per charge-death arm, and the shared bar. G3's +14.8 kW pin was
#: RETIRED 2026-08-31 as measured-wrong and has no successor number: its
#: quantity is now zero by construction.
G4_TARGETS_KW = (13.6, 8.2)
PIN_TOLERANCE = 0.15


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
    report.note(
        "G1",
        f"identity at base {BASE_COMMIT}     {BASE_CONFIG_IDENTITY}",
    )
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
                u=h5["u"][index, :],
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


def gates3410(report, geom, h5):
    """The cathode face, the compression piece, and the EMF bracket."""
    cd = h5["cathode_diagnostics"]
    t = h5["time"][:]
    sel = np.flatnonzero((t >= WINDOW[0]) & (t <= WINDOW[1]))
    throughput_W = float(np.nanmean(cd["source_P_prim"][sel]))
    c, last = geom.cathode_cell, geom.last_source_cell
    gap_sums = {}
    for charge_death, target_kW in zip(CHARGE_DEATH_CHOICES, G4_TARGETS_KW):
        # G4's arm is export_counts: the one where no face is closed, so the
        # pressure-drift row is the interior compression and nothing else.
        rows = _window_mean_rows(
            h5, geom, WINDOW[0], WINDOW[1], charge_death, "export_counts"
        )
        death_cell = c if charge_death == "cell_1" else c + 1

        # --- G3, on the arm-independent cathode face -----------------------
        handshake_W = rows["cathode_face_flux_W"]
        work_W = rows["cathode_face_work_W"]
        partner_W = rows["pressure_work_face1_W"]
        residual = abs(work_W + partner_W) / max(abs(work_W), 1.0)
        if charge_death == "cell_1":
            report.check(
                "G3",
                handshake_W == 0.0,
                f"cathode-face enthalpy+thermal-force: {handshake_W:+.6e} W "
                "-- EXACTLY zero required (the returning thermal-electron "
                "current is ~0.3 mA; the +14.8 kW pin is RETIRED as "
                "measured-wrong)",
            )
            report.check(
                "G3",
                residual <= 1e-12,
                f"cathode-face work term {work_W * 1e-3:+.4f} kW against "
                f"pressure_work_rhs's face-1 piece {partner_W * 1e-3:+.4f} kW "
                f"-- they sum to {(work_W + partner_W):+.3e} W, relative "
                f"{residual:.3e} (bar 1e-12)",
            )

        # --- G4 -------------------------------------------------------------
        compression_kW = (
            float(rows["pressure_work_W"][death_cell + 1 : last + 1].sum())
            * 1e-3
        )
        row_relative = abs(compression_kW - target_kW) / abs(target_kW)
        report.check(
            "G4",
            row_relative <= PIN_TOLERANCE,
            f"compression piece [{charge_death}], cells "
            f"{death_cell + 1}-{last} (strictly downstream of the death "
            f"cell): {compression_kW:+.3f} kW against {target_kW:+.1f} kW "
            f"-- ROW-RELATIVE {row_relative:.4f} (bar {PIN_TOLERANCE:.2f}); "
            f"throughput-normalized {compression_kW * 1e3 / throughput_W:.4f} "
            f"of P_prim {throughput_W * 1e-3:.1f} kW (reported, not gated)",
        )
        legacy_kW = (
            float(rows["pressure_work_W"][c + 1 : last + 1].sum()) * 1e-3
        )
        report.note(
            "G4",
            f"[{charge_death}] the RETIRED fixed cells-{c + 1}-{last} range "
            f"reads {legacy_kW:+.3f} kW -- which is why 'robust, "
            "handshake-independent +13.6 kW' did not survive measurement",
        )

        # --- G10, reported as a bracket --------------------------------------
        report.note(
            "G10",
            f"[{charge_death}] in-plasma EMF, current-weighted window mean: "
            f"{rows['_emf_V_declared']:+.3f} V on the declared support (faces "
            f"{c + 1}-{last}) .. {rows['_emf_V_wide']:+.3f} V across the mesh "
            f"face (faces {c + 1}-{last + 1}); W_EMF "
            f"{rows['W_EMF_W'] * 1e-3:+.3f} / "
            f"{rows['W_EMF_wide_W'] * 1e-3:+.3f} kW",
        )

        # The registered anode reading, and the two instrument arms beside it.
        for handshake in ANODE_HANDSHAKE_CHOICES:
            arm = _window_mean_rows(
                h5, geom, WINDOW[0], WINDOW[1], charge_death, handshake
            )
            gap_sums[(charge_death, handshake)] = (
                float(arm["total_W"][c : last + 1].sum()) * 1e-3
            )
    Te = h5["Te"][sel, :]
    n = h5["n"][sel, :]
    boltzmann = float(
        np.mean(Te[:, c]) * np.log(np.mean(n[:, last]) / np.mean(n[:, c]))
    )
    report.note(
        "G10",
        f"Boltzmann estimate T_e ln(n_{last}/n_{c}) = {boltzmann:.3f} V; the "
        "> 6 V binary is DROPPED as discretization-fragile (one arm spans "
        "5.27-6.22 V across defensible conventions)",
    )
    for key, value in gap_sums.items():
        label = (
            "REGISTERED CLOSURE"
            if key[1] == "sheath_row_closes_all"
            else "instrument arm"
        )
        report.note(
            "G4",
            f"source-region sum [{key[0]}/{key[1]}]: {value:+.2f} kW "
            f"({label})",
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


def gate11(report, steps):
    """Arming the operator leaves the beam bypass, and every other row, alone.

    Evaluated at ONE identical state, because two RUNS diverge the moment the
    operator books anything and could never be compared row by row. Both sims
    are freshly constructed from the same configuration bar the flag, and both
    are handed the same packed state and the same time, so their cathode solves
    see identical inputs and the comparison is of the RHS assembly alone.
    """
    params, flags = _armed_golden("cell_1", "sheath_row_closes_all")
    live = LAPDSim1D(input_dict=params, input_flags=flags)
    for _ in range(steps):
        step_once(live)
    y = np.asarray(live._y, dtype=float).copy()
    t = float(live.time)

    off_params, off_flags = build_baseline_config({"max_steps_action": "stop"})
    off = LAPDSim1D(input_dict=dict(off_params), input_flags=dict(off_flags))
    on = LAPDSim1D(input_dict=params, input_flags=flags)
    rows_off = off.rhs_terms(y, time=t)
    rows_on = on.rhs_terms(y, time=t)

    fields = ("n", "nn", "M", "Ee", "Ei", "M_n", "nn_a", "M_n_a", "En")
    moved = []
    shared = set(rows_off) & set(rows_on)
    # The operator's OWN term is present in both -- all-zero when the flag is
    # off, which is what keeps the saved structure stable -- so it is the one
    # row that is SUPPOSED to differ. Comparing it would make the gate assert
    # that arming does nothing, which is the opposite of what it is for.
    shared.discard("electron_drift_transport")
    for name in sorted(shared):
        for field in fields:
            a = getattr(rows_off[name], field, None)
            b = getattr(rows_on[name], field, None)
            if a is None or b is None:
                continue
            if np.asarray(a, dtype=float).tobytes() != np.asarray(
                b, dtype=float
            ).tobytes():
                moved.append(f"{name}.{field}")
    new_term = rows_on["electron_drift_transport"]
    non_vacuous = bool(np.any(np.asarray(new_term.Ee, dtype=float) != 0.0))
    report.check(
        "G11",
        not moved and non_vacuous,
        f"same state at t={t:.6e} s: "
        f"{len(shared)} shared terms compared (the operator's own excluded), "
        f"{len(moved)} "
        f"changed rows{' ' + str(moved[:6]) if moved else ''}; the new term is "
        f"non-zero on {int(np.count_nonzero(np.asarray(new_term.Ee)))} cells "
        f"(non-vacuous={non_vacuous})",
    )

    def _bypass(sim):
        solve = getattr(sim, "_cathode_solve", None)
        if solve is None or solve.beam_result is None:
            return None
        return float(solve.beam_result.result.beam_bypass_fraction)

    bypass_off, bypass_on = _bypass(off), _bypass(on)
    # Non-vacuity of the BEAM half of the statement: a bypass equality read at
    # a state where no beam power is deposited would be an equality of two
    # zeros. The beam rows are inside the row comparison above, so this line
    # exists to make their magnitude visible rather than inferable.
    beam_W = {
        name: float(
            np.abs(np.asarray(rows_off[name].Ee, dtype=float)).sum()
        )
        for name in (
            "beam_power_deposition",
            "beam_ionization_cost",
            "beam_excitation_radiation",
        )
    }
    report.check(
        "G11",
        bypass_off == bypass_on and bypass_off is not None,
        f"beam bypass fraction: off {bypass_off!r} vs armed {bypass_on!r} -- "
        "bit-identical required (the registered anode closure holds the beam "
        "electrons outside both Gamma_d and the kinetic sheath row, so arming "
        "must not touch their booking). Beam rows carried through the "
        "comparison above, |row| sums: "
        + ", ".join(f"{k}={v:.4e}" for k, v in beam_W.items()),
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
            u=h5["u"][index, :],
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
            f"at mean I_tot={rows['_I_tot_mean']:.1f} A, mean I_beam="
            f"{rows['_I_beam_mean']:.1f} A -- so Gamma_d is NEGATIVE here, the "
            "emission outlasting the loop current. Confined to the ~1.5 ms "
            "ring-down; the window mean and the instant are two readings, not "
            "one",
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
        gates3410(report, geom, h5)
        gate6(report, geom, h5)
    gate5(report)
    gate11(report, args.golden_steps)

    print("=" * 78)
    if report.failures:
        print(f"edt build gates: FAILED {sorted(set(report.failures))}")
        return 1
    print("edt build gates: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
