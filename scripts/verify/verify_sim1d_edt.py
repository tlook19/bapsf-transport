"""Build gates for the electron drift-transport + EMF-work operator (edt).

**These gates were registered BEFORE the operator was implemented** and are not
moved after seeing results (the standing pre-registration discipline: gates are
written down before anything is run or implemented). Each names its
QUANTITY, its MEASUREMENT SITE, and its FIXTURE. Gates 2-4, 6, 10 and 12 are
properties of a SAVED state and of the advisor consult's own algebra, so they
were measurable before the solver-side code existed at all;
``scripts/score/edt_consult_pins.py`` is the standalone evaluator those gates run
beside the shipped operator. Most of them are now LIVE RELATIONS -- recomputed
from whatever fixture ``--h5`` names and gated against a self-consistency
identity assembled on that same run, not against a fixed prior reading -- and
each registry entry below says which of the consult's original pinned numbers
were RETIRED as stale and replaced this way.

Run from the checkout root::

    PYTHONPATH=<checkout> python scripts/verify/verify_sim1d_edt.py \
        --h5 <saved sim1d run>.h5

``--registration`` prints the registry below and exits without running
anything.

**THE FIXTURE IS NAMED ON THE COMMAND LINE, NEVER IN THIS FILE.** ``--h5``
carries its path and defaults to ``DEFAULT_FIXTURE`` under the artifacts root,
because the fixture is a RUN ARTIFACT -- it lives outside the repo and is
superseded whenever the configuration of record moves -- while this file is
code. A filename written into the registry becomes a dangling pointer the
moment its artifact is superseded or its directory is emptied, and every
reading recorded against that name then describes a run nobody can open. This
suite was in exactly that state: the registry named one retired fluid-closure
artifact at seven sites and the file no longer existed anywhere.

So the registry states the PROPERTIES the fixture must HAVE, and the suite
checks at open the ones it can:

  * **the reference configuration.** The file's ``configuration_name`` root
    attribute must equal ``baseline_sim1d.PRODUCTION_STANCE``, and a file
    that says otherwise -- or says nothing, which is how a trajectory written
    before configurations were named reads -- is REFUSED rather than guessed
    at. ``configuration_identity`` is PRINTED beside it and never compared:
    it is the identity of the configuration at the vintage the artifact was
    written, so pinning it here would fail at the first unrelated key
    rotation, for reasons with no bearing on this suite.
  * **the ES1 source region present**, which is what ``SavedGeometry`` reads
    out of the saved ``geometry`` group: exactly one ``cathode`` cell, at
    least one ``column`` cell downstream of it, and a uniform cell area
    across the support between them (G2, G3, G4, G6, G10 and G12 all live on
    that support, and the face-area rebuild is only exact where the area is
    uniform).
  * **per-sample ``Te``, ``n`` and ``u`` profiles** and the
    ``cathode_diagnostics`` currents ``circuit_I_loop`` and
    ``source_I_eth_star``, plus ``source_P_prim`` for G4's reported
    throughput normalization.
  * **a DRIVEN window and an afterglow.** Saved samples across 0.1-20.1 ms
    carrying a non-zero loop current -- without one G2's non-vacuity clause
    fails and G4's identities would be equalities of zeros -- and samples
    past 20.1 ms, which is what G6's window and its 26 ms instant read.

Any current artifact with those properties runs this suite, and the readings
move with it. That is the point: the gates are properties of whatever run
``--h5`` names, and no gated statement in this file is a record of one
particular run. Numbers still quoted in the registry below against the RETIRED
consult artifact are records of what THAT run measured -- kept so the next
reader does not re-derive them -- never thresholds and never readings of the
configuration of record.

Exit 0 = every gated statement passed. A failure is a DELIVERABLE: it is
printed with its numbers and the suite exits 1. Never relax a tolerance here to
make a gate pass.

--------------------------------------------------------------------------
GATE REGISTRY
--------------------------------------------------------------------------

**G1 -- bit-inertness with the flag off.**
  QUANTITY: the accepted-step trajectory, the golden config identity, and the
  RHS term rows.
  SITE: ``scripts/gates/golden_digest_gate.py`` (4,000-step chain digest, all five
  checkpoints and the final digest) and ``scripts/edt_bitinert_ab.py``
  (retired; see commit 48be9a4).
  FIXTURE: the golden config at nx=60 for the digest; ``default_config()``
  for the moment and kinetic-DVM A/B routes.
  PASS: every checkpoint and the final digest unchanged from the committed
  reference; the config identity carries this member's keys, certified by
  two LIVE-computed controls -- PRESENCE (the declared key list is exactly
  the ``electron_drift_`` keys the live resolved config carries, and
  stripping them moves that config's identity) and SAME-OBJECT (the identity
  this suite computes for the live resolved digest config equals the
  ``config_identity`` in the digest gate's own committed reference, so both
  gates are reading one config); both routes row-by-row bit-identical to
  base with only the new all-zero rows one-sided. The digest and A/B legs
  run outside this suite; the identity controls are checked here.

  WHAT THE IDENTITY CONTROLS DO NOT CERTIFY: that the identity moves ONLY by
  this member's keys relative to some earlier state of the configuration.
  That claim needs a base identity fixed before the member existed, and the
  identity hashes the WHOLE resolved config -- so any such pin parts company
  with the live value at the first unrelated key addition or removal, for
  reasons that have no bearing on these keys, and the control then fails on
  its own bookkeeping instead of on its subject. The controls above are
  computed live on every run and hold across any such rotation.

**G2 -- the volume identity.**
  QUANTITY: ``total - (boundary_in - boundary_out + W_EMF)``, relative to the
  larger side.
  SITE: the operator's own named rows on ONE accepted step.
  FIXTURE: the golden config at nx=60, and the ES1 source region on the
  ``--h5`` fixture's saved state.
  PASS: <= 1e-10 relative on every arm of the bracket, at a state where the
  operator is NOT vacuous (a zero-current state satisfies the identity
  trivially and would gate nothing).

**G3 -- the cathode face.**
  QUANTITY: two statements. (i) the drift enthalpy-plus-thermal-force influx
  at the cathode face is ZERO; (ii) the face-1 WORK term is the exact partner
  of what ``pressure_work_rhs`` books at the same face, so the two sum to
  roundoff. Window-mean over 0.1-20.1 ms.
  SITE: ``edt_cathode_face_handshake_W`` and the face-1 work term, via the
  standalone evaluator on the saved state.
  FIXTURE: the ``--h5`` fixture's saved state (data in hand; no new run).
  PASS: (i) == 0 to roundoff; (ii) |work + pressure_work face-1| / |work| <=
  1e-12, at a magnitude of 4.298 kW.

  **The +14.8 kW pin is RETIRED as measured-wrong.** It rode the circuit's ion
  current at a face whose electron channel carries ~0.3 mA, and the rationale
  for it -- that it cancelled a ghost-Bohm booking -- was a stale read of a
  legacy row that has been inert on the shipped stance since R3.2. It is named
  here rather than deleted so that the next reader does not re-derive it.

**G4 -- the compression piece (two LIVE relations; the kW pins are RETIRED).**
  QUANTITY: two statements about the pressure-drift work row, both computed
  from the SAME run on every pass, window-mean over 0.1-20.1 ms.
  (i) SUMMATION BY PARTS -- the row summed over the operator's whole support
  equals ``W_EMF_pressure + cathode_face_work - anode_face_work``, the three
  face-and-interior powers the evaluator assembles independently of it.
  (ii) THE CLOSURE IDENTITY -- the row summed over the cells STRICTLY
  DOWNSTREAM of the death cell, under the REGISTERED closure MINUS the same
  sum on the ``export_counts`` instrument arm, equals the mesh face's work
  power that ``sheath_row_closes_all`` hands to the kinetic sheath row.
  SITE: ``edt_pressure_drift_work_W`` with the evaluator's own face powers, on
  the REGISTERED closure (the shipped default, and the headline) and on the
  ``export_counts`` instrument arm -- the one where NO face is closed, so that
  row is the operator's own interior compression and nothing else.
  FIXTURE: the ``--h5`` fixture, over the window.
  PASS: both relations <= 1e-12 relative, with NON-VACUITY gated beside each
  (finite operands, a non-zero row sum, a non-zero difference) and the SIGN of
  the closure difference gated positive. The kW figures are REPORTED with the
  throughput normalization, never gated.

  **The four kW pins are RETIRED: +35.4 / +29.9 kW under the registered
  closure and +13.6 / +8.2 kW on the instrument arms, each within 15 %
  ROW-RELATIVE.** They were measured on the retired consult artifact and they
  are properties of THAT run's state, not of the operator: on a current
  artifact at the configuration of record the two instrument-arm rows read
  0.29 and 0.43 row-relative against them and this gate failed at base and tip
  alike, on its own staleness rather than on its subject. They are named here
  rather than deleted so the next reader does not re-derive them. The fixed
  cells-2-5 range is RETIRED with its "robust, handshake-independent" label
  for the same class of reason: over that range bracket B reads -5.3 kW, so
  the quantity was bracket-A-specific rather than robust.

  WHY THESE TWO RELATIONS, AND WHAT EACH ONE EARNS. Relation (ii) IS the
  review finding that made both pins be quoted together, turned into a
  measurement: under ``sheath_row_closes_all`` the mesh face's work term is
  handed to the sheath row and lands in these same cells, so the row is a
  different QUANTITY there, not a different value of one quantity -- and (ii)
  names the difference exactly, so a future change that made the closure move
  anything ELSE in these cells breaks it. It cannot, on its own, say the
  compression row is right: a wholly wrong row would still satisfy it as long
  as the two arms differed by that one face power. That is what (i) is for --
  it ties the row itself to quantities assembled by a different route (the
  interior pressure-gradient sum and the two bounding face work powers), so a
  misbooking inside the row fails it. G2 gates the volume identity, which is
  (i) PLUS the enthalpy/thermal-force telescoping; compensating errors between
  those rows and this one pass G2 and fail (i).

  NEGATIVE CONTROLS, pre-registered in ``G4_NEGATIVE_CONTROLS`` and run via
  ``--g4-negative-control``, one per statement, and under either one this gate
  MUST fail. ``emf-operand-scale`` moves the ``W_EMF_pressure`` operand off
  the value the row was built against, which (i)'s residual catches five
  orders above its bar; ``mesh-work-drop`` zeroes the mesh face's work power,
  which (ii) catches both on its residual and on the non-vacuity guard's dead
  operand. A control that does not fire means the comparison has gone inert.

  THE SIGN IS GATED, and it is the one clause the identities cannot supply:
  both sides of (ii) carry the same face current, so a global sign error in it
  flips them together and the equality survives. The gated quantity is
  ``T_e[last] x n[last] x hi_work[last] / n_face[last + 1]``, and the mesh face
  NEVER carries beam current -- ``_face_pairs`` puts the beam only on the faces
  within ``beam_faces_through`` of the cathode cell, which the anode face is
  not, on either arm, for a source region of the length this header admits --
  so its sign is the sign of ``I_tot`` alone, and that is positive throughout
  the run: the discharge loop hands off to the open-circuit solve before the
  device current turns non-positive. What DOES reverse in afterglow is
  ``Gamma_d = I_tot - I_beam`` in the beam-carrying SOURCE cells, which is what
  G6's note reports, and it never reaches this face. So restricting the clause
  to the driven window is deliberate conservatism, not a necessity. What it
  earns is the sign convention of the face WORK channel -- ``_face_pairs``'s
  ``hi_work`` and the ``anode_face_work_W`` built from it -- which a flip would
  carry through both sides of (ii) untouched, and which ``mesh-work-drop``
  demonstrates it catches.

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
  unguarded arithmetic leaves the residue on the launch cell and nowhere
  else, and it equals the closed form ``T_e[launch] x lo_work[launch]``
  computed LIVE from the same run, to 1e-12.

  Both are needed, and (ii) is the one that earns its place. The operator
  answers J = 0 with a GUARD, not with arithmetic, so calling it at zero
  current only reaches an early return: (i) alone certifies that the guard
  fires and says nothing about what it is for. The cathode-face work channel
  rides the difference velocity ``u_e - u_i`` and does not vanish with the
  current, so the guard is cutting out a real residue; tying its size to the
  closed form is what stops a future change to that channel moving it
  silently.

  WHY A CLOSED FORM AND NOT A PINNED SCALAR. The comparison is between two
  routes to the same quantity on ONE run: the measured route assembles the
  residue the way the operator does, from the face work currents divided by
  the face densities and multiplied back by the cell density, while the
  closed form takes the launch cell's electron temperature times the launch
  face's work current directly. They agree only if the high face contributes
  nothing at zero current, the cathode face's density normalization cancels
  against the cell it is taken from, and the sign convention is the one the
  operator uses -- so a change to the work channel breaks the agreement.
  Both routes are recomputed on every run, so the statement survives
  configuration rotations that move the run's state without touching that
  channel. A pinned scalar cannot: it is the residue's VALUE on one vintage
  of the configuration, and any unrelated key rotation moves the state the
  residue rides on, after which the pin fails on its own staleness rather
  than on its subject. NON-VACUITY is gated with it -- the measured residue
  finite and strictly positive, and each closed-form operand finite and
  non-zero -- because an equality of zeros would pass while certifying
  nothing. NEGATIVE CONTROLS, pre-registered in ``G5_NEGATIVE_CONTROLS`` and
  run via ``--g5-negative-control``: perturbing the temperature operand must
  fail the comparison, and taking the closed form one cell below the launch
  cell must fail the non-vacuity guard on a dead work operand.

  WHAT THE RESIDUE IS. It is a
  discontinuity in the CLOSURE FAMILY, not in the physics. The GUARDED zero
  IS the continuum limit -- at J = 0 the two species leave together, the
  plasma is ambipolar and the ion-velocity pressure work is already exact --
  while the residue is the DRIVEN face closure evaluated outside its own
  validity, where the repelling-sheath statement its work channel encodes no
  longer holds. So the residue is not a physical jump, and it is
  state-specific by nature: it equals T_e[launch] x that face's ion current
  on the state it is measured at, which is exactly why it is gated against
  that product rather than against a number. A run crosses this boundary
  ONCE, at cathode-solve shutoff.

**G6 -- the afterglow clause (REPORTED, NOT GATED).**
  QUANTITY: the operator's net over the source region during afterglow.
  SITE: the operator's total row, on ALL THREE anode readings, with the
  registered closure as the HEADLINE and the other two labelled INSTRUMENT
  ARM. An earlier form reported ``export_counts`` alone and unlabelled, which
  gave the afterglow term the OPPOSITE SIGN to the shipped default.
  FIXTURE: the ``--h5`` fixture at t = 26 ms AND over the whole
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
  FIXTURE: the ``--h5`` fixture.
  REPORTED: the bracket, expected 3.7-6.2 V against the Boltzmann estimate
  ``T_e ln(n_5/n_1)`` = 5.7 V. **The > 6 V binary is DROPPED** as
  discretization-fragile: one arm spans 5.27-6.22 V across defensible
  conventions, so a threshold at 6 V would be reporting the convention. A
  pre-breakdown small-current frame reads ~16 V and is NOT a physics reading.

**G11 -- the beam-bypass identity.**
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

**G12 -- the twin is tied to the kernel.**
  QUANTITY: the standalone evaluator's per-cell rows against the SHIPPED
  operator's ``Ee`` row times cell volume.
  SITE: ``edt_consult_pins.evaluate`` against
  ``sources.electron_drift_transport_rhs``, on ONE saved state.
  FIXTURE: the ``--h5`` fixture at t = 10 ms, on ALL SIX arms
  (``CHARGE_DEATH_CHOICES`` x ``ANODE_HANDSHAKE_CHOICES``).
  PASS: <= 1e-12 relative on every cell, on every arm, and both exactly zero
  outside the operator's support. **Not bit-identical, by design**: the kernel
  reconstructs ``T_e`` and ``u`` from the conservative state while the twin
  reads the saved primitives, so the two differ by that round trip.

  This exists because gates 3, 4, 6 and 10 are all measured by the TWIN while
  only G2's golden leg exercises the shipped kernel. A twin that drifted would
  let every one of those pins certify the twin instead of the code that ships.

Companion gates that are NOT this suite's to run, and where they live: smoke
(``scripts/gates/smoke_sim1d.py``), the DVM suite (``verify_sim1d_k2_dvm.py``), the
digest gate (``scripts/gates/golden_digest_gate.py``), the snapshot delta
(``scripts/edt_snapshot_delta.py`` (retired; see commit 48be9a4)) and
the A/B bit-inertness reader
(``scripts/edt_bitinert_ab.py`` (retired; see commit 48be9a4)).
"""

import argparse
import json
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

SCRIPT_DIR = Path(__file__).resolve().parents[1]
# scripts/ sibling imports: the seven purpose subdirectories on sys.path.
import sys as _sys
from pathlib import Path as _Path
for _sub in ("atomic", "gates", "kinetic", "run", "score", "stance",
             "verify"):
    _dir = str(_Path(__file__).resolve().parents[1] / _sub)
    if _dir not in _sys.path:
        _sys.path.insert(0, _dir)

from baseline_sim1d import (  # noqa: E402
    PRODUCTION_STANCE,
    build_baseline_config,
)
from edt_consult_pins import (  # noqa: E402
    ANODE_HANDSHAKE_CHOICES,
    CHARGE_DEATH_CHOICES,
    SavedGeometry,
    _launched_current_A,
    _window_mean_rows,
    evaluate,
)
from golden_digest_gate import DIGEST_PARAM_OVERRIDES  # noqa: E402
from stance_config import _exit_unresolved  # noqa: E402

#: The documented default fixture: the ES1 arm of the re-anchor continuity
#: pair, a saved run at the configuration of record with the ES1 source region
#: present, a driven 0.1-20.1 ms window and an afterglow past it. It is a PATH
#: under the artifacts root, not a name resolved inside the repo, because run
#: artifacts live outside the repo -- and it is a DEFAULT rather than a pin:
#: ``--h5`` overrides it, and any artifact carrying the properties listed at
#: the top of this file runs the suite. When this one is superseded, point
#: ``--h5`` at the successor; the gated statements do not move with it.
DEFAULT_FIXTURE = (
    Path.home()
    / "bapsf/artifacts/reanchor_continuity_pair_2026-09-04"
    / "g1atrim_es1_d0e9748.h5"
)

#: The consult's window, in seconds.
WINDOW = (1.0e-4, 2.01e-2)

#: The prefix this member's config keys share, and the keys themselves by
#: namespace. G1's presence control scans the live resolved config for the
#: prefix and requires the scan to return exactly these names, so a key
#: dropped from the lists -- or one that no longer exists in the config --
#: fails the gate instead of silently narrowing the control to whatever is
#: left. Only this member's own keys belong here: the lists are the subject
#: of the presence claim, not a strip list reaching back to some earlier
#: state of the configuration.
EDT_KEY_PREFIX = "electron_drift_"
ADDED_PARAMS = (
    "electron_drift_charge_death",
    "electron_drift_anode_handshake",
)
ADDED_FLAGS = ("electron_drift_transport",)

#: Accepted steps the golden-config leg of G2 walks. A cost knob, not physics:
#: the operator is non-vacuous from step 1 there (the pre-breakdown cathode
#: solve already carries a current), so this only buys a richer state.
GOLDEN_STEPS = 200

#: The label every emitted line carries when ``--any-configuration`` bypasses
#: a lineage mismatch: it marks the whole transcript as readings of whatever
#: configuration the fixture names, not of the reference configuration, so no
#: line can be quoted as a reading of the latter.
LABEL = "NOT-A-REFERENCE-READING"

#: The identity's bar, as registered.
IDENTITY_TOLERANCE = 1e-10

#: G4's bar on its two live relations. Both compare two assemblies of the same
#: window-mean quantity that differ by summation order and, for the closure
#: identity, by one subtraction -- so the arithmetic bounds them at a few ulp
#: of the larger row. The residuals below are normalized by
#: ``max(|measured|, 1 W)``, which leaves four orders of headroom over that
#: bound and many orders below the drift a pin on either VALUE would report.
#: G3's +14.8 kW pin, like G4's four, is RETIRED as a record of the retired
#: consult artifact rather than a property of the operator.
G4_IDENTITY_TOLERANCE = 1e-12

#: G4's pre-registered negative controls, selected by ``--g4-negative-control``
#: or by passing the name to :func:`gates3410`. One per gated statement, and
#: each perturbs exactly ONE side of its comparison so the gate MUST fail:
#: ``emf-operand-scale`` moves the ``W_EMF_pressure`` operand off the value the
#: pressure-work row was assembled against, which the summation-by-parts
#: residual catches; ``mesh-work-drop`` zeroes the mesh face's work power, so
#: the closure identity's predicted difference is a dead operand, which both
#: that residual and the non-vacuity guard catch. A control that does not fire
#: means the comparison has gone inert.
G4_NEGATIVE_CONTROLS = ("emf-operand-scale", "mesh-work-drop")

#: The relative perturbation ``emf-operand-scale`` applies. Small enough that
#: it is plainly a perturbation of one operand rather than a different
#: quantity, and the residual it produces still sits about five orders above
#: the bar.
G4_OPERAND_PERTURBATION = 1.0e-6

#: G5's bar on the measured-versus-closed-form residual. The two routes to
#: the residue differ by ONE divide-then-multiply round trip through a
#: bit-identical density (the measured route forms ``n * (lo_work / n_face)``
#: at a closed cathode face where ``n_face`` IS the launch cell's own ``n``;
#: the closed form uses ``lo_work`` directly), so the arithmetic bounds their
#: difference at about two ulp of the product -- order 1e-16 relative. The
#: residual below is normalized by ``max(|measured|, 1 W)``, so on a
#: sub-watt residue this bar reads as 1e-12 W absolute: four orders of
#: headroom over the arithmetic bound, and nine orders below the drift a
#: stale pin on the residue's VALUE would report.
G5_CLOSED_FORM_TOLERANCE = 1e-12

#: G5's pre-registered negative controls, selected by ``--g5-negative-control``
#: or by passing the name to :func:`gate5`. Each perturbs exactly ONE side of
#: the comparison and the gate MUST fail: ``operand-scale`` moves the
#: temperature operand off the value the measured route used, which the
#: residual catches; ``launch-offset`` takes the closed form one cell below
#: the launch cell, where no work current is booked, which the non-vacuity
#: guard catches on a dead operand. A control that does not fire means the
#: comparison has gone inert.
G5_NEGATIVE_CONTROLS = ("operand-scale", "launch-offset")

#: The relative perturbation ``operand-scale`` applies to the temperature
#: operand. Small enough that it is plainly a perturbation and not a
#: different quantity, and the residual it produces still sits five orders
#: above the tolerance.
G5_OPERAND_PERTURBATION = 1.0e-6


class Report:
    """Collects gate outcomes so one failure does not hide the others.

    ``label``, when set (to :data:`LABEL`), is prefixed onto every line
    :meth:`check` and :meth:`note` print. It stays ``None`` for a normal run
    against the reference configuration, in which case every printed line is
    byte-identical to a build with no labelling concept at all.
    """

    def __init__(self):
        self.failures = []
        self.label = None

    def _prefix(self):
        return f"{self.label} " if self.label else ""

    def check(self, gate, ok, line):
        print(f"[{'PASS' if ok else 'FAIL'}] {self._prefix()}{gate} {line}")
        if not ok:
            self.failures.append(gate)

    def note(self, gate, line):
        print(f"[    ] {self._prefix()}{gate} {line}")


def step_once(sim):
    """Advance one step through the production step-acceptance path."""
    split = sim._flags.get("implicit_heat_conduction", False)
    diag = sim.suggest_timestep(include_heat_conduction=not split)

    def generate():
        attempt, retries, reason, events = sim._attempt_step_with_retries(
            dt=diag.dt, operator_split=None, diag=diag,
        )
        return attempt, (retries, reason, events)

    return sim._accept_step_with_picard(generate)


def _armed_golden(charge_death, anode_handshake):
    params, flags = build_baseline_config(DIGEST_PARAM_OVERRIDES)
    params = dict(params)
    flags = dict(flags)
    flags["electron_drift_transport"] = True
    params["electron_drift_charge_death"] = charge_death
    params["electron_drift_anode_handshake"] = anode_handshake
    return params, flags


def gate1_strip_control(report):
    """This member's keys are in the golden config identity's payload.

    Every reference this control compares against is computed or read LIVE,
    in-process. It cannot be otherwise: the identity hashes the WHOLE
    resolved config, so a hash pinned at a pre-member state of the code stops
    describing "the config minus these keys" the moment any UNRELATED key is
    added or removed downstream of it. A control built on such a pin then
    fails on its own stale bookkeeping, which says nothing about the keys it
    exists to check.

    PRESENCE, in two parts. COVERAGE: ``ADDED_PARAMS``/``ADDED_FLAGS`` is
        exactly the set of ``EDT_KEY_PREFIX`` keys the live resolved config
        carries, so the control cannot shrink to a subset of the member's
        keys without failing. IDENTITY: stripping those keys out of the live
        resolved config moves its identity -- if it did not, the keys are
        absent from the payload the identity hashes over and every identity
        claim about them is vacuous.
    SAME-OBJECT: the identity this suite computes equals the
        ``config_identity`` recorded in the digest gate's own committed
        reference -- the file that gate's ``--verify`` compares a fresh run
        against -- so this suite and the digest gate are hashing one config
        rather than two that merely resemble each other. That reference
        rotates at each reviewed recapture, as a side effect of the work that
        moves the config, so it never needs a hand-update here.

    What no live control can supply is the ONLY-these-keys claim against an
    earlier baseline: that needs a base identity fixed before the member
    existed, and such a pin is precisely what stops being true at the next
    unrelated rotation.
    """
    from golden_digest_gate import DEFAULT_REFERENCE, digest_config_identity

    params, flags = build_baseline_config(DIGEST_PARAM_OVERRIDES)
    live = digest_config_identity(params, flags)
    stripped_params = {
        k: v for k, v in params.items() if k not in ADDED_PARAMS
    }
    stripped_flags = {k: v for k, v in flags.items() if k not in ADDED_FLAGS}
    stripped = digest_config_identity(stripped_params, stripped_flags)
    declared = set(ADDED_PARAMS) | set(ADDED_FLAGS)
    carried = {k for k in (*params, *flags) if k.startswith(EDT_KEY_PREFIX)}
    reference_identity = json.loads(DEFAULT_REFERENCE.read_text())[
        "config_identity"
    ]

    report.note("G1", f"identity, live resolved config     {live}")
    report.note("G1", f"identity, this member's keys strip {stripped}")
    report.note(
        "G1", f"identity, digest gate's reference  {reference_identity}"
    )
    report.check(
        "G1",
        declared == carried,
        f"presence control, coverage: the {len(declared)} declared keys are "
        f"exactly the {EDT_KEY_PREFIX!r} keys the live config carries "
        f"(declared only: {sorted(declared - carried)}; "
        f"carried only: {sorted(carried - declared)})",
    )
    report.check(
        "G1",
        stripped != live,
        f"presence control, identity: stripping "
        f"{', '.join(sorted(declared))} moves the live config identity",
    )
    report.check(
        "G1",
        live == reference_identity,
        "same-object control: this suite's live identity equals the "
        f"config_identity committed in {DEFAULT_REFERENCE.name}",
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
                float(_launched_current_A(cd, index)),
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


def gates3410(report, geom, h5, g4_negative_control=None):
    """The cathode face, the compression piece, and the EMF bracket.

    G4's two relations are recomputed from the same run on every pass, so they
    survive the configuration rotations that move the state the row rides on --
    which is exactly what the four kW pins they replace could not do. See the
    registry entry for what each one earns, and why the sign is gated on its
    own.

    ``g4_negative_control`` selects a pre-registered perturbation from
    ``G4_NEGATIVE_CONTROLS``; under either one G4 MUST fail.
    """
    if g4_negative_control is not None and (
        g4_negative_control not in G4_NEGATIVE_CONTROLS
    ):
        raise ValueError(
            f"unknown G4 negative control {g4_negative_control!r}; the "
            f"pre-registered ones are {list(G4_NEGATIVE_CONTROLS)}"
        )
    cd = h5["cathode_diagnostics"]
    t = h5["time"][:]
    sel = np.flatnonzero((t >= WINDOW[0]) & (t <= WINDOW[1]))
    throughput_W = float(np.nanmean(cd["source_P_prim"][sel]))
    c, last = geom.cathode_cell, geom.last_source_cell
    gap_sums = {}
    for charge_death in CHARGE_DEATH_CHOICES:
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

        # --- G4, two LIVE relations on BOTH readings, each labelled ----------
        # The kW pins these replace were properties of the retired consult
        # artifact's state, not of the operator, and they failed on their own
        # staleness. Both statements below are recomputed from this run on
        # every pass.
        closure_rows = _window_mean_rows(
            h5,
            geom,
            WINDOW[0],
            WINDOW[1],
            charge_death,
            "sheath_row_closes_all",
        )
        gated = {}
        for label, source in (
            ("REGISTERED CLOSURE (sheath_row_closes_all)", closure_rows),
            ("INSTRUMENT ARM (export_counts)", rows),
        ):
            gated_W = float(
                source["pressure_work_W"][death_cell + 1 : last + 1].sum()
            )
            gated[label] = gated_W
            # (i) SUMMATION BY PARTS over the operator's own support. The row
            # is a per-cell p_e times a face-velocity difference; summed over
            # the support the interior faces telescope into the pressure half
            # of W_EMF, leaving the two bounding face work powers. Those three
            # are assembled by the evaluator on a different route from the row
            # itself, so a misbooking inside the row breaks the equality.
            support_W = float(source["pressure_work_W"][c : last + 1].sum())
            emf_pressure_W = source["W_EMF_pressure_W"]
            if g4_negative_control == "emf-operand-scale":
                emf_pressure_W = emf_pressure_W * (
                    1.0 + G4_OPERAND_PERTURBATION
                )
            face_in_W = source["cathode_face_work_W"]
            face_out_W = source["anode_face_work_W"]
            closed_form_W = emf_pressure_W + face_in_W - face_out_W
            residual = abs(support_W - closed_form_W) / max(
                abs(support_W), 1.0
            )
            # NON-VACUITY, gated: an inert row and dead operands would satisfy
            # the equality while certifying nothing. ``face_out_W`` is EXACTLY
            # zero under the registered closure by construction -- that is what
            # the closure IS -- so it is required finite and no more.
            live = bool(
                np.isfinite(support_W)
                and support_W != 0.0
                and np.isfinite(emf_pressure_W)
                and emf_pressure_W != 0.0
                and np.isfinite(face_in_W)
                and face_in_W != 0.0
                and np.isfinite(face_out_W)
            )
            control = (
                "" if g4_negative_control != "emf-operand-scale"
                else "NEGATIVE CONTROL 'emf-operand-scale' armed -- must "
                     "FAIL. "
            )
            report.check(
                "G4",
                live and residual <= G4_IDENTITY_TOLERANCE,
                f"{control}compression piece [{charge_death}] {label}, cells "
                f"{death_cell + 1}-{last} (strictly downstream of the death "
                f"cell): {gated_W * 1e-3:+.3f} kW, throughput-normalized "
                f"{gated_W / throughput_W:.4f} of P_prim "
                f"{throughput_W * 1e-3:.1f} kW (both REPORTED, not gated). "
                f"SUMMATION BY PARTS over the support cells {c}-{last}: row "
                f"sum {support_W * 1e-3:+.6f} kW against W_EMF_pressure "
                f"{emf_pressure_W * 1e-3:+.6f} + cathode-face work "
                f"{face_in_W * 1e-3:+.6f} - anode-face work "
                f"{face_out_W * 1e-3:+.6f} = {closed_form_W * 1e-3:+.6f} kW "
                f"-- residual {residual:.3e} (bar "
                f"{G4_IDENTITY_TOLERANCE:.0e}); non-vacuity: row sum and both "
                f"live operands finite and non-zero={live}",
            )

        # (ii) THE CLOSURE IDENTITY. The two arms' pressure-work rows differ in
        # ONE cell and by ONE quantity: the mesh face's work power, which
        # ``sheath_row_closes_all`` hands to the kinetic sheath row. That is
        # what makes the two readings different QUANTITIES rather than two
        # values of one, and naming the difference exactly is what would catch
        # a future closure that moved anything else in these cells.
        delta_W = (
            gated["REGISTERED CLOSURE (sheath_row_closes_all)"]
            - gated["INSTRUMENT ARM (export_counts)"]
        )
        mesh_work_W = (
            rows["anode_face_work_W"] - closure_rows["anode_face_work_W"]
        )
        if g4_negative_control == "mesh-work-drop":
            mesh_work_W = 0.0
        residual = abs(delta_W - mesh_work_W) / max(abs(delta_W), 1.0)
        live = bool(
            np.isfinite(delta_W)
            and delta_W != 0.0
            and np.isfinite(mesh_work_W)
            and mesh_work_W != 0.0
        )
        # THE SIGN, gated separately: both sides carry the same face current,
        # so a global sign error in it flips them together and the equality
        # survives. The mesh face never carries beam current -- _face_pairs
        # puts the beam only on the faces within beam_faces_through of the
        # cathode cell, which the anode face is not -- so this power's sign is
        # sign(I_tot) alone, positive throughout the run. Gamma_d's afterglow
        # reversal is a SOURCE-cell statement (G6) and never reaches this face,
        # so the window restriction below is conservatism, not necessity. What
        # the clause catches is a flipped sign convention in the face work
        # channel, which mesh-work-drop demonstrates.
        sign_ok = bool(mesh_work_W > 0.0)
        control = (
            "" if g4_negative_control != "mesh-work-drop"
            else "NEGATIVE CONTROL 'mesh-work-drop' armed -- must FAIL. "
        )
        report.check(
            "G4",
            live and sign_ok and residual <= G4_IDENTITY_TOLERANCE,
            f"{control}closure identity [{charge_death}], cells "
            f"{death_cell + 1}-{last}: REGISTERED CLOSURE minus INSTRUMENT "
            f"ARM "
            f"= {delta_W * 1e-3:+.6f} kW against the mesh-face work power the "
            f"sheath row takes over, {mesh_work_W * 1e-3:+.6f} kW -- residual "
            f"{residual:.3e} (bar {G4_IDENTITY_TOLERANCE:.0e}); non-vacuity: "
            f"difference and mesh-face power finite and non-zero={live}, sign "
            f"positive over the driven window={sign_ok}",
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


def gate5(report, negative_control=None):
    """J = 0: the operator is exactly zero, by configuration and by arithmetic.

    The second half measures the residue the guard removes and gates it
    against its CLOSED FORM, computed live from the same run: the launch
    cell's electron temperature times the launch face's work current,
    ``T_e[launch] x lo_work[launch]``. The units close on their own -- eV
    times amperes is watts -- and the equality is a real statement about the
    operator's arithmetic rather than a restatement of it, because it holds
    only if the high face contributes nothing at zero current, the cathode
    face's density normalization cancels against the cell it is taken from,
    and the sign is the operator's. A change to the work channel breaks it.

    WHAT A PINNED SCALAR CANNOT DO. The residue is state-specific: it rides
    the launch cell's own ``n`` and ``u`` at the step this gate stops at, so
    its VALUE moves whenever an unrelated configuration key rotates the run
    that reaches that step. A pin on that value then reports its own
    staleness -- a mismatch that says nothing about the channel the gate
    exists to watch. Both sides of the closed-form comparison are recomputed
    on every run, so the statement is invariant under those rotations while
    staying sensitive to the thing it gates.

    NON-VACUITY is gated, not assumed: the measured residue must be finite
    and strictly positive and each closed-form operand finite and non-zero,
    so an inert state cannot pass this as an equality of zeros.

    ``negative_control`` selects a pre-registered perturbation from
    ``G5_NEGATIVE_CONTROLS``; under either one this gate MUST fail.
    """
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
    # leaves this much on the launch cell. Gating it against its closed form
    # means a future change to that channel cannot move the discontinuity
    # silently, and does so without a scalar that ages out of date.
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

    # The closed form is built from the same run's own quantities. Under a
    # pre-registered negative control one side of it is perturbed, and the
    # checks below must then report a failure rather than absorb it.
    closed_form_cell = launch
    perturbation = 1.0
    if negative_control == "launch-offset":
        closed_form_cell = launch - 1
    elif negative_control == "operand-scale":
        perturbation = 1.0 + G5_OPERAND_PERTURBATION
    elif negative_control is not None:
        raise ValueError(
            f"unknown G5 negative control {negative_control!r}; the "
            f"pre-registered ones are {list(G5_NEGATIVE_CONTROLS)}"
        )

    measured_W = float(unguarded[launch])
    Te_operand = float(Te[closed_form_cell]) * perturbation
    work_operand = float(lo_work[closed_form_cell])
    closed_form_W = Te_operand * work_operand
    residual = abs(measured_W - closed_form_W) / max(abs(measured_W), 1.0)
    carried = int(np.count_nonzero(unguarded))
    # NON-VACUITY, gated: a zero or non-finite residue, or a closed form built
    # on a dead operand, would make the equality below an equality of zeros.
    measured_live = bool(np.isfinite(measured_W) and measured_W > 0.0)
    operands_live = bool(
        np.isfinite(Te_operand)
        and Te_operand != 0.0
        and np.isfinite(work_operand)
        and work_operand != 0.0
    )
    control = (
        "" if negative_control is None
        else f"NEGATIVE CONTROL {negative_control!r} armed -- must FAIL. "
    )
    report.check(
        "G5",
        measured_live
        and operands_live
        and residual <= G5_CLOSED_FORM_TOLERANCE
        and carried == 1,
        f"{control}unguarded arithmetic at zero current: {measured_W:.9f} W "
        f"on launch cell {launch} against its closed form "
        f"T_e[{closed_form_cell}] x lo_work[{closed_form_cell}] = "
        f"{Te_operand:.9e} eV x {work_operand:.9e} A = {closed_form_W:.9f} W "
        f"-- residual {residual:.3e} (bar "
        f"{G5_CLOSED_FORM_TOLERANCE:.0e}); non-vacuity: measured finite and "
        f"positive={measured_live}, both operands finite and "
        f"non-zero={operands_live}; {carried} cell carries it (1 required). "
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
    I_beam = float(_launched_current_A(cd, index))
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
                float(_launched_current_A(cd, index)),
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


def _fixture_lineage(h5, path, any_configuration=False):
    """Return ``(configuration_name, configuration_identity, off_reference)``.

    The name is CHECKED against the reference configuration and the identity is
    only carried out to be printed. That asymmetry is the point: a run at a
    different configuration is a different plasma, and readings taken off it
    are not readings of the configuration of record -- which is how this suite
    came to record a fluid-closure arm's numbers as though they were the
    stance's. The identity, by contrast, is the WHOLE resolved config's hash at
    the vintage the artifact was written, so it rotates at every unrelated key
    addition; comparing it here would fail on that bookkeeping rather than on
    the fixture, so it is reported and never gated.

    A file whose root attributes name no configuration -- how a trajectory
    written before configurations were named reads -- is treated the same as
    any other mismatch.

    With ``any_configuration`` false (the default), a mismatch RAISES
    ``ValueError`` and ``off_reference`` is never returned as ``True`` --
    :func:`main` converts that into a clean ``SystemExit(2)``, the
    ``_exit_unresolved`` precedent from ``scripts/stance/stance_config.py``,
    rather than let a caller's mistyped fixture surface as a traceback. With
    ``any_configuration`` true, a mismatch does not raise: the mismatched name
    and identity are returned instead, and the caller labels every line of the
    run as :data:`LABEL` so nothing in it can be quoted as a reading of the
    reference configuration.
    """
    def _attr(key):
        value = h5.attrs.get(key)
        if isinstance(value, bytes):
            value = value.decode()
        if value is None or str(value) in ("", "None"):
            return None
        return str(value)

    name = _attr("configuration_name")
    identity = _attr("configuration_identity")
    if name != PRODUCTION_STANCE:
        message = (
            f"the fixture {path} names configuration {name!r}, not the "
            f"reference configuration {PRODUCTION_STANCE!r}. This suite's "
            "saved-state gates read a run at the configuration of record "
            "with the ES1 source region present, a driven 0.1-20.1 ms window "
            "and an afterglow past it; see this file's header for the full "
            "list. A trajectory written before configurations were named "
            "carries no name and is refused rather than guessed at."
        )
        if not any_configuration:
            raise ValueError(message)
        return name, identity, True
    return name, identity, False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--h5",
        default=str(DEFAULT_FIXTURE),
        help="the saved sim1d run the gates read their fixture state from, "
             "under the artifacts root; it must carry the properties listed "
             "at the top of this file, and its configuration_name is checked "
             "at open. Defaults to DEFAULT_FIXTURE",
    )
    ap.add_argument("--golden-steps", type=int, default=GOLDEN_STEPS)
    ap.add_argument(
        "--g4-negative-control",
        choices=G4_NEGATIVE_CONTROLS,
        default=None,
        help="arm one of G4's pre-registered negative controls; the suite "
             "MUST then report G4 as FAILED, and a clean run afterwards MUST "
             "restore the pass",
    )
    ap.add_argument(
        "--g5-negative-control",
        choices=G5_NEGATIVE_CONTROLS,
        default=None,
        help="arm one of G5's pre-registered negative controls; the suite "
             "MUST then report G5 as FAILED, and a clean run afterwards MUST "
             "restore the pass",
    )
    ap.add_argument(
        "--any-configuration",
        action="store_true",
        help="run the saved-state gates even when --h5's configuration_name "
             "does not match the reference configuration, instead of "
             "refusing. The mismatched name and identity are printed loudly "
             "once at open, and every line this run prints -- every gate "
             "line, the summary and the final verdict -- is prefixed "
             f"{LABEL!r}: the readings become readings of whatever "
             "configuration the fixture names, not of the reference, and no "
             "line may be quoted as one. Exit-code semantics are unchanged "
             "(0 = all pass, 1 = failures); the label carries the caveat, "
             "not the exit code",
    )
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
    with h5py.File(args.h5, "r") as h5:
        try:
            name, identity, off_reference = _fixture_lineage(
                h5, args.h5, args.any_configuration
            )
        except ValueError as error:
            _exit_unresolved(error)
        if off_reference:
            report.label = LABEL
            print(
                f"{LABEL}: fixture {args.h5} names configuration {name!r}, "
                f"not the reference configuration {PRODUCTION_STANCE!r} "
                f"(identity {identity}); continuing because "
                "--any-configuration was armed. No statement in this run is "
                "a reading of the reference configuration."
            )
        gate1_strip_control(report)
        gate2_golden(report, args.golden_steps)
        geom = SavedGeometry(h5)
        geom.check_uniform_area()
        prefix = f"{LABEL} " if off_reference else ""
        print(
            f"[    ] {prefix}fixture {args.h5}: configuration {name} "
            f"(identity {identity}), cells={geom.cells}, "
            f"cathode_cell={geom.cathode_cell}, anode_face={geom.anode_face}"
        )
        gate2_es1(report, geom, h5)
        gates3410(report, geom, h5, args.g4_negative_control)
        gate6(report, geom, h5)
        gate12(report, geom, h5)
        gate5(report, args.g5_negative_control)
        gate11(report, args.golden_steps)

    print("=" * 78)
    prefix = f"{report.label} " if report.label else ""
    if report.failures:
        print(f"{prefix}edt build gates: FAILED {sorted(set(report.failures))}")
        return 1
    print(f"{prefix}edt build gates: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
