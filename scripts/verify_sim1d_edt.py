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
  SITE: ``edt_pressure_drift_work_W``, on the REGISTERED closure (the shipped
  default, and the headline) and on the ``export_counts`` instrument arm --
  the one where NO face is closed, so that row is the operator's own interior
  compression and nothing else.
  FIXTURE: ``scripts/mgcr1_confirm.h5``.
  PASS: under the REGISTERED closure +35.4 kW (bracket A) / +29.9 kW
  (bracket B), and on the instrument arms +13.6 / +8.2 kW, each within 15 %
  ROW-RELATIVE. BOTH are gated and both are labelled, because quoting either
  alone was a review finding: under ``sheath_row_closes_all`` the mesh face's
  work term is handed to the sheath row and lands in these same cells, so the
  row is a different quantity there, not a different value of one quantity.
  The fixed cells-2-5 range is RETIRED with its "robust,
  handshake-independent" label: over that range bracket B reads -5.3 kW, so
  the quantity was bracket-A-specific rather than robust. The
  throughput-normalized figure is REPORTED, never gated.

**G5 -- the J = 0 limit (negative control at the statement level).**
  QUANTITY: (i) every cell of the operator's total row at zero current;
  (ii) the magnitude of the CLOSURE-FAMILY discontinuity the guard removes.
  SITE: the ``electron_drift_transport`` RHS term; the operator called
  directly at zero current; and the face currents rebuilt with the guard
  BYPASSED.
  FIXTURE: ``default_config()`` with the plasma on and the drive off, the flag
  ARMED; and a LIVE discharge state (``export_counts`` arm, ``GOLDEN_STEPS``
  steps) with the currents set to zero.
  PASS: (i) exactly zero on every cell -- not "small", zero; (ii) the
  unguarded arithmetic leaves ``G5_DISCONTINUITY_W`` on the launch cell and
  nowhere else, matching its closed form to 1e-12.

  Both are needed, and (ii) is the one that earns its place. The operator
  answers J = 0 with a GUARD, not with arithmetic, so calling it at zero
  current only reaches an early return: (i) alone certifies that the guard
  fires and says nothing about what it is for. The cathode-face work channel
  rides the difference velocity ``u_e - u_i`` and does not vanish with the
  current, so the guard is cutting out a real residue; pinning its size is
  what stops a future change to that channel moving it silently.

  WHAT THE RESIDUE IS (wording corrected 2026-08-31 (Tom)). It is a
  discontinuity in the CLOSURE FAMILY, not in the physics. The GUARDED zero
  IS the continuum limit -- at J = 0 the two species leave together, the
  plasma is ambipolar and the ion-velocity pressure work is already exact --
  while the residue is the DRIVEN face closure evaluated outside its own
  validity, where the repelling-sheath statement its work channel encodes no
  longer holds. So the number below is not a physical jump, and it is
  fixture-specific by nature: it equals T_e[launch] x that face's ion current
  on the state it is measured at, which is why the fixture is pinned with it.
  A run crosses this boundary ONCE, at cathode-solve shutoff.

**G6 -- the afterglow clause (REPORTED, NOT GATED).**
  QUANTITY: the operator's net over the source region during afterglow.
  SITE: the operator's total row, on ALL THREE anode readings, with the
  registered closure as the HEADLINE and the other two labelled INSTRUMENT
  ARM. An earlier form reported ``export_counts`` alone and unlabelled, which
  gave the afterglow term the OPPOSITE SIGN to the shipped default.
  FIXTURE: ``scripts/mgcr1_confirm.h5`` at t = 26 ms AND over the whole
  afterglow window (t > 20.1 ms). Both, because the window's mean loop current
  is 218 A while at the 26 ms instant the loop carries ~12 A -- the term is
  confined to the ~1.5 ms ring-down and those are two readings, not one.
  REPORTED: under the registered closure, +0.25 (`cell_1`) / +0.12 (`cell_2`)
  kW over the window and +1.6 / -1.0 W at the instant; the instrument arms run
  the other way, ``export_counts`` giving -0.68 / -0.81 kW. The earlier
  "-0.5...-0.6 kW" figure is RETIRED: it was the as-built number, and the
  cathode amendment moved it. One more disclosure rides here: in afterglow
  ``Gamma_d`` is NEGATIVE, because the emission outlasts the loop current
  (I_beam 297.5 A against I_tot 217.8 A, window means) and the drift
  reverses.

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

**G12 -- the twin is tied to the kernel (NEW 2026-08-31).**
  QUANTITY: the standalone evaluator's per-cell rows against the SHIPPED
  operator's ``Ee`` row times cell volume.
  SITE: ``edt_consult_pins.evaluate`` against
  ``sources.electron_drift_transport_rhs``, on ONE saved state.
  FIXTURE: ``scripts/mgcr1_confirm.h5`` at t = 10 ms, on ALL SIX arms
  (``CHARGE_DEATH_CHOICES`` x ``ANODE_HANDSHAKE_CHOICES``).
  PASS: <= 1e-12 relative on every cell, on every arm, and both exactly zero
  outside the operator's support. **Not bit-identical, by design**: the kernel
  reconstructs ``T_e`` and ``u`` from the conservative state while the twin
  reads the saved primitives, so the two differ by that round trip.

  This exists because gates 3, 4, 6 and 10 are all measured by the TWIN while
  only G2's golden leg exercises the shipped kernel. A twin that drifted would
  let every one of those pins certify the twin instead of the code that ships.

Companion gates that are NOT this suite's to run, and where they live: smoke
(``scripts/smoke_sim1d.py``), the DVM suite (``verify_sim1d_k2_dvm.py``), the
digest gate (``scripts/golden_digest_gate.py``), the snapshot delta
(``scripts/edt_snapshot_delta.py``) and the A/B bit-inertness reader
(``scripts/edt_bitinert_ab.py``).
"""

import argparse
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np

from cablp.constants import ev_to_erg
from cablp.solvers._sim1d import LAPDSim1D, default_config
from cablp.solvers._sim1d.core.state import ConservativeState1D, derive_state
from cablp.solvers._sim1d.physics.sources import (
    _drift_face_currents,
    _drift_face_values,
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

#: The golden config identity WITHOUT this member's three keys, computed
#: through the digest gate's OWN expression. G1's strip control must reproduce
#: it. The two golden references carry different identities by design, so a
#: control computed through the other expression matches neither.
#:
#: ROTATED with the golden references when [legacy-boundary-retirement] removed
#: `characteristic_boundary` and `neutral_kinetic_dvm_tn_feedback` (retired
#: 2026-08-31 (Tom)). The previous value, 21a9b476..., was this same quantity
#: while those two keys still existed; stripping the three edt keys AND
#: restoring those two reproduces it bit-for-bit, which is the proof that the
#: move is those two keys and nothing else.
BASE_COMMIT = "1fc05c9 minus the edt keys"
BASE_CONFIG_IDENTITY = (
    "7f2eadcb0b0610fa1ab6c8cd4fe174d61227ce1e2973e1d146cbbb1e91993d87"
)

#: The keys this branch adds, by namespace, for G1's strip control.
#:
#: EXTENDED 2026-08-31 by the two cathode-jet arming keys. They are not this
#: member's physics -- they belong to the arming criterion -- but G1 measures
#: the identity of the WHOLE golden config against a fixed pre-edt baseline,
#: so every key added downstream of that baseline has to be stripped for the
#: control to reach it. The alternative was to re-baseline
#: ``BASE_CONFIG_IDENTITY`` on every unrelated config addition, which would
#: destroy exactly the property it exists to pin: that this baseline is the
#: config with the edt keys REMOVED, not merely some earlier config. Adding a
#: name here is inert unless that key really is present, because the strip is
#: a dict comprehension over keys that exist.
ADDED_PARAMS = (
    "electron_drift_charge_death",
    "electron_drift_anode_handshake",
    "neutral_jet_arm_current_A",
    "neutral_jet_disarm_current_A",
)
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

#: G4's pin under the REGISTERED closure, per charge-death arm. The mesh face's
#: work term is handed to the sheath row there and lands in the same cells, so
#: this row is a different quantity from the instrument-arm one above -- both
#: are gated, because quoting either alone was the review finding.
G4_TARGETS_CLOSURE_KW = (35.4, 29.9)

#: The discontinuity the J = 0 guard removes, in watts on the launch cell,
#: measured at the ``export_counts`` arm after ``GOLDEN_STEPS`` steps. It is
#: fixture-specific by nature -- it rides that state's own u_1 and n_1 -- so
#: the fixture is pinned with it.
G5_DISCONTINUITY_W = 0.372957143


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
        f"strip control: the identity moves ONLY by the "
        f"{len(ADDED_PARAMS) + len(ADDED_FLAGS)} keys added since this "
        f"baseline (computed through the digest gate's own expression)",
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

        # --- G4, on BOTH readings, each labelled -----------------------------
        # Quoting one alone was a review finding: under the registered closure
        # the mesh face's work term is handed to the sheath row and lands in
        # these same cells, so the row is a different QUANTITY there, not a
        # different value of one quantity.
        closure_rows = _window_mean_rows(
            h5,
            geom,
            WINDOW[0],
            WINDOW[1],
            charge_death,
            "sheath_row_closes_all",
        )
        closure_target = G4_TARGETS_CLOSURE_KW[
            CHARGE_DEATH_CHOICES.index(charge_death)
        ]
        for label, source, target in (
            (
                "REGISTERED CLOSURE (sheath_row_closes_all)",
                closure_rows,
                closure_target,
            ),
            ("INSTRUMENT ARM (export_counts)", rows, target_kW),
        ):
            value_kW = (
                float(
                    source["pressure_work_W"][death_cell + 1 : last + 1].sum()
                )
                * 1e-3
            )
            row_relative = abs(value_kW - target) / abs(target)
            report.check(
                "G4",
                row_relative <= PIN_TOLERANCE,
                f"compression piece [{charge_death}] {label}, cells "
                f"{death_cell + 1}-{last} (strictly downstream of the death "
                f"cell): {value_kW:+.3f} kW against {target:+.1f} kW -- "
                f"ROW-RELATIVE {row_relative:.4f} (bar {PIN_TOLERANCE:.2f}); "
                f"throughput-normalized {value_kW * 1e3 / throughput_W:.4f} "
                f"of P_prim {throughput_W * 1e-3:.1f} kW (reported, not "
                "gated)",
            )
        compression_kW = (
            float(rows["pressure_work_W"][death_cell + 1 : last + 1].sum())
            * 1e-3
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

    # Both halves below run on a LIVE discharge state with the currents set to
    # zero. They are DIFFERENT statements and the gate needs both, because the
    # operator answers this case with a guard rather than with arithmetic.
    params, flags = _armed_golden("cell_1", "export_counts")
    live = LAPDSim1D(input_dict=params, input_flags=flags)
    for _ in range(GOLDEN_STEPS):
        step_once(live)

    # (i) THE GUARD. What the operator actually books at J = 0: exactly zero.
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
        f"guarded: live discharge state at zero current -- {nonzero} non-zero "
        f"cells, total={rows['edt_total_W']!r}, exactly zero required",
    )

    # (ii) THE ARITHMETIC UNDER THE GUARD. Calling the operator with zero
    # currents only reaches its early return, so (i) alone certifies the guard
    # and says nothing about what the guard is FOR. This half rebuilds the
    # face currents with the guard bypassed -- the helper never had it; only
    # the outer function does -- and measures the discontinuity the guard
    # exists to cut out. It is a DOCUMENTED magnitude, not a defect: the
    # cathode-face work channel rides the difference velocity u_e - u_i and
    # does not vanish with the current, so at J = 0 the unguarded arithmetic
    # leaves this much on the launch cell. Pinning it means a future change to
    # that channel cannot move the discontinuity silently.
    geom = live._plasma_geometry()
    spec = live._electron_drift
    derived = derive_state(
        live.state, floors=live._floors, ion_mass_g=live._ion_mass_g
    )
    Te = np.asarray(derived.Te, dtype=float)
    n = np.maximum(np.asarray(live.state.n, dtype=float), live._floors["n"])
    n_face = _drift_face_values(n, geom)
    u_face = _drift_face_values(np.asarray(derived.u, dtype=float), geom)
    lo_flux, hi_flux, lo_work, hi_work = _drift_face_currents(
        geom,
        spec,
        0.0,
        0.0,
        n,
        u_face,
        np.asarray(geom.plasma_face_area_cm2, dtype=float),
    )
    launch = spec["launch_cell"]
    index = np.arange(geom.cells)
    w_lo = np.divide(
        lo_work,
        n_face[index],
        out=np.zeros(geom.cells),
        where=n_face[index] > 0.0,
    )
    w_hi = np.divide(
        hi_work,
        n_face[index + 1],
        out=np.zeros(geom.cells),
        where=n_face[index + 1] > 0.0,
    )
    unguarded = -Te * n * (w_hi - w_lo)
    measured_W = float(unguarded[launch])
    closed_form_W = float(Te[launch] * lo_work[launch])
    residual = abs(measured_W - closed_form_W) / max(abs(measured_W), 1.0)
    report.check(
        "G5",
        abs(measured_W - G5_DISCONTINUITY_W) / abs(G5_DISCONTINUITY_W) <= 1e-6
        and residual <= 1e-12
        and int(np.count_nonzero(unguarded)) == 1,
        f"unguarded arithmetic at zero current: {measured_W:.9f} W on the "
        f"launch cell against the pinned {G5_DISCONTINUITY_W:.6f} W, and "
        f"against its closed form T_e[launch] x lo_work[launch] = "
        f"{closed_form_W:.9f} W (relative {residual:.3e}); "
        f"{int(np.count_nonzero(unguarded))} cell carries it (1 required). "
        "This is the DOCUMENTED discontinuity the guard removes, measured at "
        f"the export_counts arm after {GOLDEN_STEPS} steps",
    )


def gate11(report, steps):
    """Arming the operator leaves the beam bypass, and every other row, alone.

    Evaluated on ONE sim at ONE state, by toggling its presence gate.

    Two RUNS diverge the moment the operator books anything, so they could
    never be compared row by row. Two freshly-built sims handed the same packed
    state do not work either, and the reason is worth recording: the drive
    phase is carried by TRIGGER STATE, not by the ``time`` argument, so a sim
    that has never stepped is still pre-drive and books no beam at all -- an
    earlier form of this gate compared two zeros and passed vacuously.

    So the sim is advanced into the driven phase once, and its resolved
    ``_electron_drift`` record -- the presence gate the whole operator hangs
    off, and a single attribute by design -- is toggled between two evaluations
    of the same state. Everything else (circuit lag, phase triggers, cathode
    solve inputs, ``y``, ``t``) is then identical by construction rather than
    by argument passing, and the comparison is of the RHS assembly alone.
    """
    params, flags = _armed_golden("cell_1", "sheath_row_closes_all")
    live = LAPDSim1D(input_dict=params, input_flags=flags)
    for _ in range(steps):
        step_once(live)
    y = np.asarray(live._y, dtype=float).copy()
    t = float(live.time)

    def _bypass(sim):
        solve = getattr(sim, "_cathode_solve", None)
        if solve is None or solve.beam_result is None:
            return None
        return float(solve.beam_result.result.beam_bypass_fraction)

    spec = live._electron_drift
    rows_on = live.rhs_terms(y, time=t)
    bypass_on = _bypass(live)
    live._electron_drift = None
    rows_off = live.rhs_terms(y, time=t)
    bypass_off = _bypass(live)
    live._electron_drift = spec

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

    # NON-VACUITY of the BEAM half of the statement, gated rather than noted:
    # an identity read where no beam power is deposited is an equality of
    # zeros, and an earlier form of this gate passed exactly that way.
    beam_W = {
        name: float(np.abs(np.asarray(rows_off[name].Ee, dtype=float)).sum())
        for name in (
            "beam_power_deposition",
            "beam_ionization_cost",
            "beam_excitation_radiation",
        )
    }
    report.check(
        "G11",
        all(v > 0.0 for v in beam_W.values()),
        "beam is LIVE at the compared state (non-vacuity), |row| sums: "
        + ", ".join(f"{k}={v:.4e}" for k, v in beam_W.items()),
    )
    report.check(
        "G11",
        bypass_off == bypass_on and bypass_off is not None,
        f"beam bypass fraction: off {bypass_off!r} vs armed {bypass_on!r} -- "
        "bit-identical required (the registered anode closure holds the beam "
        "electrons outside both Gamma_d and the kinetic sheath row, so arming "
        "must not touch their booking)",
    )


def gate12(report, geom, h5):
    """The standalone evaluator's rows equal the SHIPPED operator's.

    Gates 3, 4, 6 and 10 are all measured by ``edt_consult_pins.evaluate``, a
    second implementation of the same algebra, while only G2's golden leg
    exercises ``sources.electron_drift_transport_rhs``. A twin that drifts from
    the kernel would let every pin above certify the twin instead of the code
    that ships -- so this ties them together on one saved state.

    The kernel is handed a geometry rebuilt from the saved file. ``plasma_open``
    and ``plasma_face_live_cell`` are derived so that the kernel's
    ``_drift_face_values`` reproduces the evaluator's ``_face_mean`` rule
    exactly: a face is OPEN only where both neighbours are live, and otherwise
    takes its one live cell.

    NOT bit-identical, and the reason is stated rather than tuned around: the
    kernel reconstructs ``T_e`` and ``u`` from the CONSERVATIVE state
    (``derive_state``) while the evaluator reads the saved primitives, so the
    two differ by the round trip through ``Ee = 3/2 n T_e`` -- a relative
    1e-16 on the inputs. The density floor is set to zero here so that the
    kernel's ``max(n, floor)`` and the evaluator's raw ``n`` are the same
    array; the floor is a real and deliberate difference between the two
    paths, and this gate is about the ARITHMETIC, not about floor policy.
    """
    index = int(np.argmin(np.abs(h5["time"][:] - 1.0e-2)))
    cd = h5["cathode_diagnostics"]
    I_tot = float(cd["circuit_I_loop"][index])
    I_beam = float(cd["source_I_eth_star"][index])
    Te = np.asarray(h5["Te"][index, :], dtype=float)
    n = np.asarray(h5["n"][index, :], dtype=float)
    u = np.asarray(h5["u"][index, :], dtype=float)

    active = np.asarray(geom.plasma_active, dtype=bool)
    cells = geom.cells
    plasma_open = np.zeros(cells + 1, dtype=bool)
    live_cell = np.full(cells + 1, -1, dtype=int)
    for face in range(cells + 1):
        lo_ok = face - 1 >= 0 and active[face - 1]
        hi_ok = face < cells and active[face]
        plasma_open[face] = bool(lo_ok and hi_ok)
        if lo_ok and not hi_ok:
            live_cell[face] = face - 1
        elif hi_ok and not lo_ok:
            live_cell[face] = face
    shipped_geometry = SimpleNamespace(
        cells=cells,
        plasma_open=plasma_open,
        plasma_face_live_cell=live_cell,
        plasma_face_area_cm2=geom.plasma_face_area_cm2,
        plasma_volume_cm3=geom.plasma_volume_cm3,
        length_cm=geom.length_cm,
    )
    ion_mass_g = 6.6464731e-24  # helium; cancels exactly, see below
    state = ConservativeState1D(
        n=n.copy(),
        nn=np.zeros(cells),
        # M and Ee are built from the saved primitives and inverted again by
        # derive_state, so ion_mass_g cancels and only the round trip remains.
        M=ion_mass_g * n * u,
        Ee=1.5 * n * Te * ev_to_erg,
        Ei=np.zeros(cells),
    )
    floors = {"n": 0.0, "nn": 0.0, "Te": 0.0, "Ti": 0.0}

    worst = 0.0
    worst_where = ""
    compared = 0
    # ALL SIX arms: charge_death x anode_handshake. The gate previously
    # compared only cell_1/sheath_row_closes_all -- the registered closure --
    # which left the twin unconstrained on the five other arms that gates 3, 4,
    # 6 and 10 actually measure through it.
    for charge_death in CHARGE_DEATH_CHOICES:
        for handshake in ANODE_HANDSHAKE_CHOICES:
            spec = {
                "charge_death": charge_death,
                "anode_handshake": handshake,
                "cathode_face": geom.cathode_face,
                "anode_face": geom.anode_face,
                "launch_cell": geom.cathode_cell,
            }
            rhs, _ = _operator(
                state=state,
                floors=floors,
                ion_mass_g=ion_mass_g,
                geometry=shipped_geometry,
                spec=spec,
                I_tot_A=I_tot,
                I_beam_A=I_beam,
            )
            shipped_W = (
                np.asarray(rhs.Ee, dtype=float)
                * np.asarray(geom.plasma_volume_cm3, dtype=float)
                * 1.0e-7
            )
            twin = evaluate(
                geom, Te, n, I_tot, I_beam, charge_death, handshake, u=u
            )
            twin_W = np.asarray(twin["total_W"], dtype=float)
            compared += 1
            scale = max(float(np.abs(twin_W).max()), 1.0)
            diff = float(np.abs(shipped_W - twin_W).max())
            if diff / scale > worst:
                worst = diff / scale
                worst_where = f"{charge_death}/{handshake}"
            # Outside the support both must be exactly zero, or one of them is
            # booking into a cell the other does not touch at all.
            outside = np.ones(cells, dtype=bool)
            outside[geom.cathode_cell : geom.anode_face] = False
            clean = not np.any(shipped_W[outside]) and not np.any(
                twin_W[outside]
            )
            report.check(
                "G12",
                diff / scale <= 1e-12 and clean,
                f"[{charge_death}/{handshake}] shipped kernel vs standalone "
                f"twin on the saved state at t=10 ms: worst per-cell "
                f"difference {diff:.3e} W on a {scale:.3e} W row, relative "
                f"{diff / scale:.3e} (bar 1e-12); both exactly zero outside "
                f"the support: {clean}",
            )
    report.note(
        "G12",
        f"worst over the {compared} compared arms "
        f"(charge_death x anode_handshake): {worst:.3e} relative "
        f"({worst_where}). NOT bit-identical by design -- the kernel "
        "reconstructs T_e and u from the conservative state while the twin "
        "reads the saved primitives",
    )


def gate6(report, geom, h5, afterglow_lo=2.01e-2):
    """The afterglow clause, on EVERY anode reading. Reported, never gated on.

    The registered closure is the headline. An earlier form of this gate
    evaluated ``export_counts`` alone and unlabelled, which reported the
    afterglow term with the OPPOSITE SIGN to the shipped default.
    """
    t = h5["time"][:]
    cd = h5["cathode_diagnostics"]
    index = int(np.argmin(np.abs(t - 2.6e-2)))
    c, last = geom.cathode_cell, geom.last_source_cell
    for handshake in ANODE_HANDSHAKE_CHOICES:
        label = (
            "REGISTERED CLOSURE"
            if handshake == "sheath_row_closes_all"
            else "INSTRUMENT ARM"
        )
        for charge_death in CHARGE_DEATH_CHOICES:
            res = evaluate(
                geom,
                h5["Te"][index, :],
                h5["n"][index, :],
                float(cd["circuit_I_loop"][index]),
                float(cd["source_I_eth_star"][index]),
                charge_death,
                handshake,
                u=h5["u"][index, :],
            )
            instant_W = float(res["total_W"][c : last + 1].sum())
            rows = _window_mean_rows(
                h5, geom, afterglow_lo, float(t[-1]), charge_death, handshake
            )
            window = float(rows["total_W"][c : last + 1].sum()) * 1e-3
            report.note(
                "G6",
                f"[{charge_death}/{handshake}] {label}: window "
                f"{afterglow_lo * 1e3:.1f}-{t[-1] * 1e3:.1f} ms "
                f"{window:+.4f} kW at mean I_tot={rows['_I_tot_mean']:.1f} A; "
                f"instant t={t[index] * 1e3:.3f} ms {instant_W:+.2f} W at "
                f"I_tot={float(cd['circuit_I_loop'][index]):.1f} A",
            )
    rows = _window_mean_rows(
        h5, geom, afterglow_lo, float(t[-1]), "cell_1", "sheath_row_closes_all"
    )
    report.note(
        "G6",
        "the term is confined to the ~1.5 ms ring-down, and Gamma_d is "
        f"NEGATIVE across this window (mean I_tot {rows['_I_tot_mean']:.1f} A "
        f"against mean I_beam {rows['_I_beam_mean']:.1f} A, a net drift of "
        f"{rows['_I_tot_mean'] - rows['_I_beam_mean']:+.1f} A): the emission "
        "outlasts the loop current and the drift reverses. The window mean "
        "and the instant are two readings, not one",
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
        gate12(report, geom, h5)
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
