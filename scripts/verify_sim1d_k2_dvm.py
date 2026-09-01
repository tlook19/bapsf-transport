"""K2a transient-DVM coupling-integrity gate suite (E3 + limit cases).

Pre-registered gates for the in-solver transient deterministic velocity-grid
neutral arm, ``neutral_model = "kinetic_dvm"`` (default off, gated on
``neutral_two_zone``). The E3 list is the coupling-integrity block of the
neutral-architecture evaluation package; the L gates are the limit cases the
operator can be held to exactly or tightly.

Gates:
  I1  particle inventory, distribution form: over a multi-step window the
      change in ``sum(f V)`` equals (all births - all losses) to roundoff --
      substep B creates exactly what substep A destroyed, per channel
  I2  particle inventory, domain form: the change in the full inventory
      (distributions plus the lagged end-return buffers) equals
      (external births - ionization - pumping) to roundoff, every internal
      channel having cancelled
  I3  ledger completeness: every channel the engine declares is present in
      every ledger it emits, and no ledger entry is unaccounted
  I4  the same inventory closure and an independent transfer reconstruction
      on the R5 STAND-IN expanded-end geometry (retired keys, kept as a
      stand-in -- not production; see R5_STANDIN_PARAMS), where the column
      and annulus areas jump at the plenum constriction and the end
      expansion -- the case the throat-face flux form exists for and the one
      the uniform default geometry cannot exercise
  J1  the bounded-chord annulus flight classes satisfy the two-dimensional
      mean-chord theorem, ``pi (Rm - Rp) / 2``, which nothing in their
      derivation was fitted to; and every class flight time is sharper than
      the exponential the rate arm implies
  J2  the bounded-chord jump operator routes every launched particle to
      exactly one outcome, and the running engine closes both ledger forms
      and reproduces the booked transfer on the R5 stand-in expanded-end
      geometry -- the I4 statement, made against the jump kernel
  J3  naming the shipped ``annulus_flights = "rates"`` is bit-identical to
      not naming it at all
  I5  the same two statements on a zero-annulus geometry, plus the
      statement that nothing leaks into a cell whose annulus has no volume
  I6  ENERGY ledger identity: on a synthetic CLOSED BOX -- no annulus, no
      mesh, no pumping, specular end walls, no external source -- the change
      in the box's total energy equals the CX/elastic exchange plus the
      ionization consumption to roundoff, every disarmed row booking zero;
      then the same two closure forms with EVERY channel armed at once, once
      per annulus treatment, and on the in-solver default arm -- all at the
      tolerance the particle ledger is held to
  S1  recycle identity: what the arm sources at a plasma-terminating surface
      equals what the boundary term removed from the plasma there,
      per face, on all three geometries of RECYCLE_GEOMETRIES -- the shipped
      uniform bore, the R5 stand-in (whose plenum obstruction puts the
      cathode's live cell at index 2 rather than at the mesh start) and the
      PRODUCTION machine read from the stance file; and the arm deposits it
      in that same cell
  S2  per-end pump fidelity: at each end the sticking coefficient times that
      end's OWN open area times the incident one-way flux is the configured
      pumping speed, with ``pump_elbow_conductance_lps`` folded in series on
      a plenum end -- on the production machine, whose two ends differ in
      area by 3.63x, with unequal speeds at the two ends
  C1  momentum transfer antisymmetry: the fluid coupling term's M row is
      exactly minus the kinetic momentum moment per cell, to roundoff
  C2  energy transfer antisymmetry: the fluid coupling term's Ei row is
      exactly the kinetic energy moment closed with the same bulk-kinetic
      decomposition the conservative birth booking uses, to roundoff
  C4  the booked transfer rebuilt by an INDEPENDENT route -- substep A's
      marched state recovered algebraically from the public post-update
      state, birth moments taken from the analytic Maxwellian targets
      rather than from the engine's bin sums -- so the decomposition
      itself is under test and not merely copied
  C5  elastic-channel calibration: the isotropic BGK rate carries the
      one-half momentum-transfer factor exactly, and the resulting rate
      coefficient equals the closed form 0.5 <sigma_iso v_rel> to roundoff
  C3  zone-exchange antisymmetry: V_col * nu_x == V_ann * nu_xp cell by
      cell, so the column/annulus channel moves particles without creating
      or destroying any
  R1  rejected attempts mutate nothing: a rejected step attempt leaves the
      distributions, the pending buffers, the coupling accumulators and the
      ledger bit-identical to the pre-attempt state
  R2  trial RHS evaluations mutate nothing: repeated rhs() calls at
      arbitrary states do not advance the neutral clock
  P1  default off: the shipped configuration builds no arm, exposes no arm
      term, and its packed RHS is bit-identical to a build that has never
      heard of the arm
  P2  presence gating: with the arm ON but not yet engaged, the fluid terms
      keep their exact rows; once engaged the neutral rows and the
      superseded ion-transfer rows are exactly zero and the coupling term
      carries them
  P3  moment consistency: the saved ``nn`` field IS the zeroth moment of the
      column distribution (floored), and ``nn_a`` the annulus moment
  D1  the coupling term BOUNDS the timestep: a drain injected into the
      tick-frozen transfer shortens dt_surface_loss in exact inverse
      proportion and takes the active constraint -- the inverse of the
      K2d diagnosis, where a 1e12 erg/cm^3/s injected drain moved the
      suggested dt by exactly zero
  D2  honest constraints: every bound the engaged arm makes phantom (it
      zeroes the row the bound describes) reads inf and cannot be named
      active_constraint, while the unstripped form of that same bound is
      demonstrably finite -- the value that used to set the step
  D3  the floor-aware relax defers, never destroys: inert on a healthy
      step (applied == booked bit-exactly), engaged under a drain no
      admissible step could carry, and closing
      ``applied_cum + debt == booked_cum`` per cell to roundoff
  D4  the wall recycle enters as a directed inflow at its own face: the
      injected flux integral equals what was fed, nothing appears
      upstream of the face, and the emitting cell does not retain the
      whole return the way the superseded stationary birth did
  D5  the counted ionization debit is capped at the POST-REBIRTH
      inventory: on a frozen synthetic cell the booked count is paid with
      zero shortfall for every booking up to the cell's whole inventory,
      while the pre-fix ordering -- the same debit against the marched
      state alone, run as a negative control -- short-falls above the
      closed-form threshold and not below it; plus the ledger, handshake
      and positivity statements per tick on the live arm
  B1  counted boundary inflow, closed box: the four boundary-inflow
      channels handed as PARTICLES are injected at exactly that count and
      independently of the tick length, each landing at its own face or
      birth cell with nothing upstream/downstream/elsewhere; the same
      numbers handed as RATES are the negative control and scale with dt
  B2  the in-solver counted-source ledger: every tick's handed count is
      exactly what the ACCEPTED steps since the last tick accumulated, per
      cell and per channel, a rejected attempt moving it by zero; resetting
      the accumulator after the tick instead of before is the negative
      control and over-counts
  B3  the counted channels carry their own emission energy, with every
      channel armed at once: both ledgers close and each counted channel's
      energy row is its count times the mean energy of its own spectrum,
      rebuilt from the velocity grid. Emitting the cathode face at the WALL
      temperature is the negative control -- not one particle moves, so B1
      and B2 cannot see it and only this statement does
  WR1 cylindrical-wall detailed balance, closed box: the net wall energy
      exchange equals the accommodated share's velocity-resolution offset,
      pinned per alpha at the fixture's grid (the continuum zero-net
      statement was retired as the pin 2026-08-31 -- a discretized wall does
      not obey it), and the offset is shown to be EXACTLY the accommodated
      share; the statement the reflection selector owns holds at roundoff
      too (the two arms exchange the same wall energy at every alpha, and
      the pure reflection limit alpha = 0 exchanges none). Two negative
      controls: perturbing the accommodated share off the pinned offset, and
      solving the re-emission temperature against the continuum <E> = 2kT
      instead of the discrete moment
  WR2 the wall-reflection selector in-solver: it reaches the engine, moves
      the state, and both ledgers keep closing over several production-arm
      ticks; at alpha = 1 the two values are bit-identical, in-solver and
      on the bare engine under both annulus treatments. Dropping the
      selector on the way to the engine -- the silent-inert defect -- is the
      negative control
  WR3 the diffuse-elastic wall return, every channel armed: its count is
      (1 - alpha) times the landings exactly, its energy (1 - alpha) times
      the incident wall energy exactly as DISCRETE moments, and its net v_z
      zero. Booking that re-emission at the WALL MEAN energy is the negative
      control -- not one particle moves, so nothing else in this suite sees
      it
  CJ1 the cathode-side energetic recycle, closed box: the counted recycle
      splits into the R_N energetic backscatter and the thermal remainder,
      the two counts sum to the handed count exactly, and both ledgers
      close. Birthing the atoms and leaving their energy row at zero is the
      negative control -- the particle ledger cannot see it
  CJ2 the same channel in-solver: injected == counted per tick, the birth's
      energy row is the count times the placed spectrum's discrete mean, and
      the CUMULATIVE cathode-ledger ``backscatter`` row equals the
      cumulative birth energy plus what the next tick is still owed.
      Forming the debit from a tick-time (phi_c + Ti) on the window count is
      the negative control
  CJ3 the launch energy is the debited energy, every channel armed: the row
      against R_E times the counted incident energy, and the disclosed
      analytic-vs-discrete faithfulness number of the launch spectrum.
      Launching on the 300 K cosine-wall spectrum is the negative control --
      not one particle moves and both ledgers still close, so only the
      row-relative cross-book catches it; both normalizations are reported
  CJ4 the DVM cathode jet and ``cathode_jet_surface_debit`` refuse to arm
      together: two independent debits of the same R_E share
  AJ1 anode jet, closed box: the counted anode stream splits into the R_N
      energetic share and the thermal remainder summing to the handed count
      EXACTLY, and both ledgers close. Birthing the atoms with the energy row
      left at zero is the negative control -- the particle ledger cannot see
      it, the energy ledger can
  AJ2 in-solver, engaged arm, jet armed: injected == counted per tick; the
      birth energy row is the count times the placed spectrum's discrete
      mean; and the CUMULATIVE anode-book ``backscatter`` row equals the
      cumulative birth energy plus what the next tick is still owed. Forming
      the debit from a tick-time (phi_a + Ti) on the window count is the
      negative control
  AJ3 every channel armed: the ledger's ``birth_anode_jet`` energy row
      against R_E times the incident energy the anode book was debited by,
      reported BOTH row-relative and throughput-normalized, plus the
      analytic-vs-discrete faithfulness number of the launch spectrum.
      Launching on the 300 K cosine-wall spectrum is the negative control
  AJ4 the two presence-gated MOMENTUM rows: ``momentum_anode_jet`` against an
      independent rebuild of the signed sum over sides of m <v_z> count from
      the discrete launch spectra, and ``momentum_mesh_absorbed`` against the
      MIRROR ANTISYMMETRY of a one-sided seed reflected about the mesh face.
      Launching both sides into +z, and tallying the mesh interception with
      |v_z| instead of the signed v_z, are the two negative controls
  AJ5 the DVM anode jet and the fluid ``anode_neutral_jet`` refuse to arm
      together: two independent directed re-emissions of the same collected
      stream (the G28 statement, called directly -- see the gate's docstring)
  AJ6 the ruled zero-incident closure: a cell whose committed incident energy
      is exactly zero launches NOTHING and is born wholly thermal, per cell
      and not per tick, with cells carrying positive incident energy in the
      same tick launching normally beside it. Dropping the one mask -- the
      pre-ruling arithmetic -- is the negative control, and it RAISES
  BF1 closed box, engine-only: one annular baffle throttles the ANNULUS by
      exactly its transparency t_f = open_ann / A_ann and leaves the COLUMN
      bit-identical, with the particle ledger closed. Routing the same
      throttle through the anode MESH -- which blocks the column too -- is the
      negative control, and the column-untouched statement fails there
  BF2 THE PLAN GATE, the matched case: a sealed 300 K tube carrying the
      g1atrim baffle face's own radii with a density step across the face.
      The DVM's net annulus current per unit density difference against the
      fluid's series orifice 0.25 vbar open_ann -- ~1 with the baffle on,
      A_ann / open_ann (the plan's "~1.75x", measured) with it off, at (48,12)
      and (64,24) and with the discrete-grid gap MEASURED rather than assumed
  BF3 in-solver on the engaged production arm with the stance baffle armed:
      the baffle rows non-zero ONLY on the face's flanking cells, the energy
      ledger closed in BOTH the row-relative and throughput-normalized forms,
      and momentum_baffle_absorbed held to AJ4's mirror antisymmetry
  G1..G27, G29..G32 construction refusals: each unsupported configuration
      raises a ValueError at construction naming the offender. G2 is the
      model-preset resolver's refusal half -- an explicitly-set family member
      the selection cannot carry, refused ONCE with the whole member set named
  X1  the resolver's other half: naming ``neutral_model='kinetic_dvm'`` on an
      otherwise untouched ``default_config()`` constructs, every member of
      the family resolved to the value the selection requires and none of
      them hand-cleared
  L1  free-streaming, exact uniform stationarity: a spatially uniform
      distribution in one velocity bin with a matched inflow is stationary
      to roundoff under the transport march
  L2  free-streaming, exact centre-of-mass displacement: with no collisions
      and no mass leaving the domain, <z> advances by exactly v_z * t
  L3  free-streaming vs the analytic attenuation: the steady discrete
      solution matches its closed form to roundoff and converges to
      exp(-nu z / v) at first order under axial refinement
  L4  equilibrium preservation: a Maxwellian at the wall temperature with
      balanced sources stays put to tolerance over many updates
  L5  wall flux balance: incident == accommodated + reflected exactly, at
      the cylindrical wall and at both end walls, for any accommodation

The conservation and antisymmetry gates (I1, I2, I4, I5, S1, C1, C2, C3, C4) are
statements about the OPERATOR, not about the rate values it is handed, so the
suite runs them once per value of ``neutral_kinetic_dvm_exchange`` -- the same
gate functions at the same tolerances, with only the closure rebound. The
default-off, refusal and limit-case gates are about the shipped default and
run once.

Artifacts: this script writes nothing. The transcript is the artifact; the
caller redirects it (``k2_dvm_verify.txt`` by campaign convention).

Usage (from <checkout>/cablp, with PYTHONPATH set to that same cablp):
    python scripts/verify_sim1d_k2_dvm.py
"""
import hashlib
import shutil
import sys
import tempfile
import warnings
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from cablp.atomic.cross_sections import (
    phelps_he_isotropic_cm2,
    phelps_iso_rate_cm3_s,
    phelps_momentum_transfer_rate_cm3_s,
)
from cablp.solvers._sim1d import (
    KINETIC_DVM_INCOMPATIBLE_DEFAULTS,
    LAPDSim1D,
    default_config,
    save_restart_state,
)
from cablp.solvers._sim1d.core.geometry import (
    absorbing_live_cells_by_role,
    build_geometry,
    is_plenum_cell,
)
from cablp.solvers._sim1d.core.validation import (
    refuse_anode_backscatter_double_book,
    refuse_cathode_backscatter_double_book,
)
from cablp.solvers._sim1d.physics.kinetic_dvm import (
    ELASTIC_BGK_MOMENTUM_FACTOR,
    EXCHANGE_MODELS,
    LEDGER_BIRTH_CHANNELS,
    LEDGER_BOOKKEEPING,
    LEDGER_ENERGY_BIRTH_CHANNELS,
    LEDGER_ENERGY_BOOKKEEPING,
    LEDGER_ENERGY_LOSS_CHANNELS,
    LEDGER_ENERGY_NET_CHANNELS,
    LEDGER_EXTERNAL_BIRTHS,
    LEDGER_LOSS_CHANNELS,
    LEDGER_MOMENTUM_DIAGNOSTICS,
    TransientDVM,
    ledger_energy_residual,
    ledger_residual,
)
from cablp.solvers._sim1d.physics.kinetic_neutrals import (
    EV,
    KB,
    M_HE,
    annulus_chord_classes,
)
from cablp.solvers._sim1d.physics.neutrals import (
    _effective_pump_speed,
    knudsen_flow_coefficients,
    neutral_thermal_speed,
    neutral_zone_volumes,
    two_zone_knudsen_coefficients,
)
from cablp.constants import kb_cgs, m_He_cgs

# The stance loader, for the committed stance FILE. It is this module's only
# scripts/ import, and it is deliberate: the geometry the gates below call
# PRODUCTION must come from the artifact production is run from, not from a
# dict restated here that can go stale against it (it did -- see
# PRODUCTION_GEOMETRY_KEYS).
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from stance_config import load_stance  # noqa: E402

CADENCE_S = 2.5e-5
ROUNDOFF_REL = 1.0e-12

# Which zone-exchange closure the gates below build. The conservation and
# antisymmetry statements are properties of the OPERATOR, not of the rate
# values it is handed, so every gate that makes one is run once per selector
# value; ``main`` rebinds this between the two passes. Everything else (the
# default-off, refusal and limit-case gates) runs on the shipped default only,
# which is what those gates are about.
EXCHANGE_MODEL = "cauchy_chord"

# Closed form of <sigma_iso v_rel>: the Phelps isotropic cross section is
# 7.63e-20 E^-0.5 m^2 with E = m g^2 / (4 eV) the equal-mass relative
# collision energy, so sigma_iso * g is velocity-INDEPENDENT and its
# Maxwellian average is that same constant, at any temperature.
CLOSED_ISO_RATE_CM3_S = 2.0 * 7.63e-16 * np.sqrt(EV / M_HE)

# Registered bracket on the kinetic/fluid RATE-COEFFICIENT ratio gated by C6
# (re-registered 2026-08-23e; supersedes the 2026-08-05 [1.0, 1.7]). Both
# sides approximate the SAME two-Maxwellian average, so the bracket is a
# DERIVED residual budget rather than a factor-wide tripwire: the lower edge
# is exact (every residual is sign-definite >= 0) and the upper edge is the
# worst-case residual sum with headroom. See gate_c6's docstring for the
# per-point decomposition and for the two structural errors it must catch.
RATE_COEFF_RATIO_BRACKET = (1.00, 1.10)

# Predicted per-point residual sums [%] over the C6 probe points
# Ti = 0.1/0.5/2/8 eV, interpolation + Jensen + drift-neglect, recorded so a
# future move in the observed ratios can be CLASSIFIED rather than merely
# re-bracketed. Reported alongside the observed ratios; not itself gated.
RATE_COEFF_RESIDUAL_PREDICTED_PCT = (0.95, 1.9, 2.8, 4.4)


# The measured incompatibility set for ``neutral_model = "kinetic_dvm"`` now
# lives IN THE SOLVER (``core/model_families.py``) and is imported above: the
# resolver applies it at construction, and this fixture reads the same tuple
# so the arms below and the solver can never disagree about what the arm
# refuses. Entries are ``(namespace, key, required_value, why)``.
#
# ``arm_config`` still applies it explicitly. That is now belt-and-braces --
# the resolver sets every member left at its config default -- but it keeps
# this suite's arms building IDENTICALLY whatever the package defaults happen
# to be, which is what the base was for.


# --------------------------------------------------------------- harness


def arm_config(**overrides):
    """Return the (input_dict, input_flags) of a minimal ON-arm build."""
    d, fl = default_config()
    d = dict(d)
    fl = dict(fl)
    fl["neutral_two_zone"] = True
    # The kinetic-compatible base. Applied BEFORE ``overrides`` below, so a
    # caller -- in particular a refusal gate in REFUSALS -- can still arm any
    # of these back on top and get the refusal it is there to test.
    for _space, _key, _value, _why in KINETIC_DVM_INCOMPATIBLE_DEFAULTS:
        (fl if _space == "flags" else d)[_key] = _value
    d["neutral_model"] = "kinetic_dvm"
    d["neutral_kinetic_dvm_cadence_s"] = CADENCE_S
    d["neutral_kinetic_dvm_exchange"] = EXCHANGE_MODEL
    # DECLARED ARM, not an inherited default (declared 2026-08-30, when the
    # package default moved to 'diffuse_elastic'). Every pinned number in
    # this suite that touches a cylindrical-wall return was measured with the
    # non-accommodated share placed in its INCIDENT bin, so the suite names
    # that arm at its one shared construction site rather than inheriting
    # whatever the package ships. Before this line 26 of the suite's 27 build
    # sites took the default implicitly; the flip would have moved all of
    # them silently. The WR gates override it per-arm and so still exercise
    # BOTH values -- which is the point: the arm under test is chosen by the
    # gate, never by the package.
    d["neutral_kinetic_dvm_wall_reflection"] = "specular"
    for key, value in overrides.items():
        if key in fl or key.startswith("flag:"):
            fl[key.removeprefix("flag:")] = value
        else:
            d[key] = value
    return d, fl


def make_sim(**overrides):
    d, fl = arm_config(**overrides)
    return LAPDSim1D(input_dict=d, input_flags=fl)


def advance_one_step(sim, operator_split=None):
    """Advance one step through the PRODUCTION step-acceptance path.

    ``LAPDSim1D.advance_one_step`` raises on the FIRST rejected candidate. That
    is not how a run steps: ``run()`` hands the candidate to
    ``_attempt_step_with_retries``, which re-attempts at ``DT_REJECT_FACTOR``
    times the dt (0.5) up to ``max_step_retries`` (8) and only raises once the
    retries are exhausted or the dt would fall below ``dt_min``. A gate that
    stepped the raising API was therefore asserting something stricter than the
    solver ships -- that no candidate is ever rejected -- and a single rejected
    candidate aborted the suite (measured 2026-08-23).

    This is the same three calls ``run()`` makes around one step, without
    run()'s dt-growth ramp and output-cadence caps: those shape the step
    SCHEDULE, which no gate here is about, whereas the retry is what decides
    whether a step is accepted at all.
    """
    split = (
        sim._flags.get("implicit_heat_conduction", False)
        if operator_split is None
        else operator_split
    )
    diag = sim.suggest_timestep(include_heat_conduction=not split)

    def _generate():
        attempt, retries, reason, events = sim._attempt_step_with_retries(
            dt=diag.dt,
            operator_split=operator_split,
            diag=diag,
        )
        return attempt, (retries, reason, events)

    result, _attempt, _extra = sim._accept_step_with_picard(_generate)
    return result


def run_until_updates(sim, n_updates, max_steps=6000):
    """Advance until the neutral clock has ticked ``n_updates`` times."""
    ledgers = []
    steps = 0
    while sim._dvm.updates < n_updates and steps < max_steps:
        before = sim._dvm.updates
        advance_one_step(sim)
        steps += 1
        if sim._dvm.updates > before:
            ledgers.append(dict(sim._dvm.last_ledger))
    if sim._dvm.updates < n_updates:
        raise RuntimeError(
            f"only {sim._dvm.updates} neutral updates in {steps} steps"
        )
    return ledgers


def uniform_tube(nz, length_cm=1600.0, Rp=15.0, Rm=50.0):
    """Return a synthetic strictly-uniform coaxial geometry.

    The device geometries all carry shorter source cells and expanded end
    cells, so the cell area and the cell length both vary along z. The
    free-streaming and equilibrium identities below are exact only on a
    constant-area, constant-dz mesh -- on a varying one they acquire
    genuine (correct) mesh terms. Building the limit cases on this tube
    keeps them statements about the OPERATOR rather than about the device
    mesh; the device mesh is exercised by every other gate here.
    """
    dz = np.full(nz, length_cm / nz)
    Rp_cm = np.full(nz, Rp)
    Rm_cm = np.full(nz, Rm)
    return SimpleNamespace(
        cells=nz,
        length_cm=dz,
        Rp_cm=Rp_cm,
        Rm_cm=Rm_cm,
        plasma_volume_cm3=np.pi * Rp_cm**2 * dz,
        neutral_volume_cm3=np.pi * Rm_cm**2 * dz,
    )


def bare_dvm(nz=12, nvz=16, nvp=6, **kwargs):
    """Build a standalone DVM on the uniform straight tube."""
    kwargs.setdefault("exchange_model", EXCHANGE_MODEL)
    return TransientDVM(geometry=uniform_tube(nz), nvz=nvz, nvp=nvp, **kwargs)


def zero_plasma(dvm):
    nz = dvm.nz
    return {
        "n_i": np.zeros(nz),
        "Ti_eV": np.full(nz, 0.026),
        "u_i": np.zeros(nz),
        "nu_ion": np.zeros(nz),
    }


# R5 STAND-IN GEOMETRY (retired keys, kept as a stand-in -- not production).
#
# The R5 parametric flare: the end vessel expands to a 1 m neutral radius over
# 10 cells with the plasma held at ``Rp = 15`` cm, and the plenum choke
# (``Rcs = 40``, ``Lcs = 25``) constricts the annulus in front of the cathode.
# Both are ANNULUS area jumps, which is what the throat-face flux form in
# ``_march`` exists to handle; the column area is uniform throughout.
#
# These keys were RETIRED by the G1 measured geometry (compare_sim1d_es1.py
# records the retirement, and the two area machineries are mutually exclusive
# by construction), so this block is NOT production geometry and is no longer
# labelled as one. It is kept for the two things it is the only geometry here
# to supply:
#
#   * a plenum obstruction, which puts the cathode's live cell at index 2
#     rather than at the mesh start -- the offset the positional-constant
#     deposit defect S1 guards against needs in order to be a test at all;
#   * a coarse three-radius annulus with an exactly-representable chord split,
#     on which the J2 flight-map routing residual is EXACTLY zero. On the
#     280-cell stance mesh the same residual is 3.3e-16 -- roundoff, not a
#     routing error, but the gate's statement is exact-zero and is not
#     relaxed here.
#
# ``collector_length_cm`` is pinned at the R5 value. The config default dropped
# 100 -> 7.8 at R2a, and inheriting it subdivided the ten-cell end block into
# 0.78 cm cells -- a mesh R5 never had, and the one on which the explicit
# neutral-diffusion checkerboard measured 2026-08-23 appeared.
R5_STANDIN_PARAMS = {
    "Rp": 15.0,
    "R_cath": 15.0,
    "Rcs": 40.0,
    "Lcs": 25.0,
    "Rsup": 0.0,
    "collector_length_cm": 100.0,
    "end_expansion_cells": 10,
    "end_expansion_machine_radius_cm": 100.0,
    "end_expansion_plasma_radius_cm": 15.0,
    # Gap pinned with the region: the fixed source span runs from the anode
    # face outward, so its far end rides the pinned gap or the span stops
    # being a whole number of source_region_dz_cm cells. ``L_cath`` is the
    # same physical distance and moves with it.
    "cathode_anode_gap_cm": 50.0,
    "L_cath": 50.0,
    "source_region_length_cm": 100.0,
    "source_region_dz_cm": 10.0,
}
R5_STANDIN_FLAGS = {
    "end_expansion_geometry": True,
    "source_fixed_grid": True,
}
#: The same package in ``arm_config`` override form (``flag:`` prefixed flags).
R5_STANDIN_GEOMETRY_KEYS = {
    **R5_STANDIN_PARAMS,
    **{f"flag:{key}": value for key, value in R5_STANDIN_FLAGS.items()},
}


def expanded_end_geometry():
    """Return the R5 stand-in expanded-end geometry (see R5_STANDIN_PARAMS)."""
    d, fl = default_config()
    d = dict(d)
    fl = dict(fl)
    d.update(R5_STANDIN_PARAMS)
    fl.update(R5_STANDIN_FLAGS)
    return LAPDSim1D(input_dict=d, input_flags=fl).geometry


def zero_annulus_tube(nz=6, blocked=(2, 3)):
    """Return a tube whose ``blocked`` cells have NO annulus at all.

    ``Rm == Rp`` there, so ``neutral_zone_volumes`` returns ``V_ann = 0``:
    the zone-exchange and wall rates are forced to zero, the annulus face
    areas vanish, and the cells must neither receive nor emit annulus
    particles. Nothing in the device geometries produces this, so it is the
    degenerate case the engine's ``inv_va`` guards are written for.
    """
    dz = np.full(nz, 100.0)
    Rp_cm = np.full(nz, 15.0)
    Rm_cm = np.full(nz, 50.0)
    Rm_cm[list(blocked)] = 15.0
    return SimpleNamespace(
        cells=nz,
        length_cm=dz,
        Rp_cm=Rp_cm,
        Rm_cm=Rm_cm,
        plasma_volume_cm3=np.pi * Rp_cm**2 * dz,
        neutral_volume_cm3=np.pi * Rm_cm**2 * dz,
    )


def closed_box_dvm(nz=8, nvz=16, nvp=6):
    """Return a DVM with every external ENERGY channel disarmed.

    A straight tube with NO annulus (``Rm == Rp``, so
    :func:`neutral_zone_volumes` returns ``V_ann = 0`` in every cell): the
    cylindrical wall and the two zone-exchange channels then carry nothing
    at all. No anode mesh, no pumping, and SPECULAR end walls
    (``accommodation = 0``), which return what reaches them into the exact
    bin mirror and therefore at the incident energy. No external source of
    any kind.

    What is left is the closed box the energy identity is registered on:
    the only channels with any energy in them are the CX/elastic exchange
    against the ion background and the ionization a partner consumes, so
    the change in the box's total energy must be exactly those two
    bookings.
    """
    dvm = TransientDVM(
        geometry=uniform_tube(nz, Rp=15.0, Rm=15.0),
        nvz=nvz,
        nvp=nvp,
        accommodation=0.0,
        exchange_model=EXCHANGE_MODEL,
        s_L=0.0,
        s_R=0.0,
    )
    dvm.seed_from_density(np.full(nz, 1.0e13), np.zeros(nz), T_K=400.0)
    return dvm


#: The energy rows the closed box disarms. Each must book exactly zero
#: there, which is what makes the I6 residual a statement about the two
#: channels that ARE live rather than about a cancellation between many.
DISARMED_ENERGY_ROWS = (
    "loss_wall",
    "loss_mesh_blocked",
    "loss_baffle_blocked",
    "loss_pump_L",
    "loss_pump_R",
    "birth_wall_accommodated",
    "birth_wall_reflected",
    "birth_mesh_reemit",
    "birth_baffle_reemit",
    "loss_closed_face_blocked",
    "birth_closed_face_reemit",
    "birth_puff",
    "birth_recombination",
    "birth_cathode_face",
    "birth_collector_face",
    "birth_anode",
    "net_surface_wall",
    "net_surface_mesh",
    "net_surface_baffle",
    "net_surface_closed_face",
    "net_surface_end_L",
    "net_surface_end_R",
)


def all_channels_energy_closure(annulus_flights, nz=12):
    """Worst energy-ledger residuals with EVERY channel armed at once.

    The closed box of :func:`closed_box_dvm` proves the identity on the two
    channels it leaves live; it cannot see a mis-booked surface, because it
    never fires one. This fires all of them on the uniform tube -- an anode
    mesh, partial pumping at both ends, a puff, volume recombination, an
    anode rebirth and both recycle faces -- and does it once per annulus
    treatment, since the bounded-chord arm books its wall, mesh and end
    channels through the flight kernel rather than through the march.

    Returns ``(distribution, domain)`` worst relative residuals.
    """
    dvm = TransientDVM(
        geometry=uniform_tube(nz),
        nvz=16,
        nvp=6,
        s_L=0.3,
        s_R=0.3,
        accommodation=0.4,
        exchange_model=EXCHANGE_MODEL,
        annulus_flights=annulus_flights,
        mesh_face=nz // 2,
        transparency=0.642,
    )
    dvm.seed_from_density(np.full(nz, 1.0e13), np.full(nz, 1.0e13))
    sources = {
        "recombination": np.full(nz, 1.0e15),
        "puff": np.zeros(nz),
        "anode": np.zeros(nz),
        "cathode_face": np.zeros(nz),
        "collector_face": np.zeros(nz),
    }
    sources["puff"][3] = 3.0e17
    sources["anode"][nz // 2] = 2.0e16
    sources["cathode_face"][0] = 5.0e16
    sources["collector_face"][-1] = 4.0e16
    plasma = geometry_plasma(nz)
    worst_dist = 0.0
    worst_dom = 0.0
    for _ in range(6):
        r = ledger_energy_residual(
            dvm.update(CADENCE_S, sources=sources, **plasma)
        )
        worst_dist = max(worst_dist, abs(r["distribution_rel"]))
        worst_dom = max(worst_dom, abs(r["domain_rel"]))
    return worst_dist, worst_dom


def geometry_plasma(nz):
    """A deliberately non-uniform plasma background for the geometry gates."""
    return {
        "n_i": np.full(nz, 5.0e12),
        "Ti_eV": np.linspace(0.5, 3.0, nz),
        "u_i": np.linspace(-1.0e5, 3.0e5, nz),
        "nu_ion": np.full(nz, 2.0e3),
    }


def independent_transfer(dvm, dt, plasma, rec, T_s_K=None):
    """Rebuild ``(M, Ei, S)_transfer`` by a route independent of the engine.

    ``_book_transfer`` builds these from substep A's tallies and from bin
    sums over the birth distributions. This rebuilds them WITHOUT either:

    - substep A's marched state is recovered algebraically from the public
      post-update ``f_c``. The only column births are the collisional
      rebirths and recombination, both at the local ion Maxwellian ``M_i``,
      so ``f_c = f_march + (x + rec dt / V) M_i`` with ``x`` the rebirth
      density. ``x`` satisfies the linear relation ``x = A - x B`` in the
      rates, which inverts exactly -- no call into the march.
    - the birth momentum and energy come from the ANALYTIC Maxwellian
      moments ``m u`` and ``(1/2) m u^2 + (3/2) k T_i``, so the gate also
      tests that the projection really is moment-exact rather than assuming
      it, and would fail on a wrong birth temperature or drift.

    Valid whenever the column receives no mesh, anode or end-face source
    (the caller supplies only recombination and an annulus puff); zone
    exchange, the cylindrical wall and the end buffers may all be live,
    because those act on the annulus or inside the march.

    The CLOSED-FACE re-emission is the one other column birth, and on a
    geometry that has one it must come off ``f_c`` before the inversion for
    the same reason ``rec_births`` does: left in, it is misread as
    collisional rebirth and the recovered ``f_march`` is wrong. It is
    rebuilt here from the engine's published per-side blocked COUNTS times
    this function's own half-flux spectra -- the spectra are constructed
    independently, so a side re-emitted at the wrong temperature or into
    the wrong direction still shows up as a transfer error rather than
    cancelling.
    """
    g = dvm.g
    T_s_K = dvm.T_wall_K if T_s_K is None else float(T_s_K)
    Ti = np.asarray(plasma["Ti_eV"], dtype=float)
    u = np.asarray(plasma["u_i"], dtype=float)
    # The engine floors the birth temperature at 0.02 eV; match it exactly.
    Ti_used = np.maximum(Ti, 0.02)
    M_i = np.empty((dvm.nz, g.nvz, g.nvp))
    for i in range(dvm.nz):
        M_i[i] = g.maxwellian(float(Ti_used[i]), float(u[i]))
    inv_vc = np.where(
        dvm.V_col > 0.0, 1.0 / np.maximum(dvm.V_col, 1e-300), 0.0
    )
    rec_dt = np.asarray(rec, dtype=float) * dt
    rec_births = (rec_dt * inv_vc)[:, None, None] * M_i

    closed_births = np.zeros_like(dvm.f_c)
    for _face, d_in, cell, surface in dvm._closed_emitters:
        emit = -d_in
        T_side = T_s_K if surface else dvm.T_wall_K
        count = float(dvm.last_closed_counts[emit][cell])
        closed_births[cell] += (
            count * inv_vc[cell] * g.half_flux_spectrum(T_side, emit)
        )

    nu_cx, nu_el = dvm.collision_frequencies(
        plasma["n_i"], Ti, u
    )
    nu_coll = nu_cx + nu_el
    residual = dvm.f_c - rec_births - closed_births
    A = (nu_coll * residual * dt).sum(axis=(1, 2))
    B = (nu_coll * M_i * dt).sum(axis=(1, 2))
    x = A / (1.0 + B)
    f_march = residual - x[:, None, None] * M_i

    vol = dvm.V_col[:, None, None]
    L_ion = np.asarray(plasma["nu_ion"], dtype=float)[:, None, None] * (
        f_march * dt * vol
    )
    L_coll = nu_coll * f_march * dt * vol
    VZ = g.VZ[None, :, :]
    V2 = g.V2[None, :, :]
    N_ion = L_ion.sum(axis=(1, 2))
    P_ion = M_HE * (L_ion * VZ).sum(axis=(1, 2))
    E_ion = 0.5 * M_HE * (L_ion * V2).sum(axis=(1, 2))
    P_coll_lost = M_HE * (L_coll * VZ).sum(axis=(1, 2))
    E_coll_lost = 0.5 * M_HE * (L_coll * V2).sum(axis=(1, 2))

    N_coll = x * dvm.V_col
    e_birth = 0.5 * M_HE * u**2 + 1.5 * Ti_used * EV
    P = P_ion + (P_coll_lost - M_HE * u * N_coll) - M_HE * u * rec_dt
    E = E_ion + (E_coll_lost - e_birth * N_coll) - e_birth * rec_dt
    S = N_ion - rec_dt

    scale = 1.0 / (np.maximum(dvm.V_col, 1e-300) * dt)
    M_t = P * scale
    S_t = S * scale
    Ei_t = E * scale - u * M_t + 0.5 * M_HE * u**2 * S_t
    return M_t, Ei_t, S_t


def transfer_reconstruction_error(dvm, dt, plasma, rec, T_s_K=None):
    """Return the worst relative error of :func:`independent_transfer`."""
    want = independent_transfer(dvm, dt, plasma, rec, T_s_K=T_s_K)
    got = (dvm.M_transfer, dvm.Ei_transfer, dvm.S_transfer)
    worst = 0.0
    for engine, rebuilt in zip(got, want):
        scale = max(np.max(np.abs(rebuilt)), 1e-300)
        worst = max(worst, float(np.max(np.abs(engine - rebuilt)) / scale))
    return worst


def geometry_closure(geom, label, annulus_flights="rates"):
    """Run a DVM on ``geom`` and return (worst residuals, transfer error).

    Everything physical is live -- zone exchange, the cylindrical wall,
    partial end pumping, a puff into the annulus and volume recombination
    into the column -- so the closure statement is made against a working
    engine, not a stripped one.
    """
    nz = int(np.asarray(geom.length_cm).size)
    dvm = TransientDVM(
        geometry=geom, nvz=16, nvp=6, s_L=0.3, s_R=0.3,
        exchange_model=EXCHANGE_MODEL, annulus_flights=annulus_flights,
    )
    seed_ann = np.where(dvm.V_ann > 0.0, 1.0e13, 0.0)
    dvm.seed_from_density(np.full(nz, 1.0e13), seed_ann)
    plasma = geometry_plasma(nz)
    rec = np.full(nz, 1.0e15)
    puff = np.zeros(nz)
    puff[int(np.argmax(dvm.V_ann > 0.0))] = 3.0e17
    dt = CADENCE_S
    worst_dist = 0.0
    worst_dom = 0.0
    for _ in range(6):
        led = dvm.update(
            dt, sources={"recombination": rec, "puff": puff}, **plasma
        )
        r = ledger_residual(led)
        worst_dist = max(worst_dist, abs(r["distribution_rel"]))
        worst_dom = max(worst_dom, abs(r["domain_rel"]))
    transfer_err = transfer_reconstruction_error(dvm, dt, plasma, rec)
    return dvm, worst_dist, worst_dom, transfer_err


#: The stance of record. Its committed file is the ONE artifact that
#: constructs the production configuration, so the geometry below is READ FROM
#: IT rather than restated here.
PRODUCTION_STANCE = "g1atrim"

#: The ``input_dict`` keys of the stance's MACHINE GEOMETRY package. All five
#: are mesh-coupled and travel together: ``nx`` sizes the far column, the two
#: radius profiles carry one entry per mesh cell (they are built offline by
#: scripts/g1_build_profiles.py from the measured census, and the vessel
#: profile is a staircase), and the baffle arrays are the apertures that go
#: with that machine. Nothing else of the stance is taken -- the operating
#: point, the closure family and the shaped initial fill are not geometry, and
#: several of them are keys a ``kinetic_dvm`` arm refuses outright (see
#: KINETIC_DVM_INCOMPATIBLE_DEFAULTS).
STANCE_GEOMETRY_PARAMS = (
    "nx",
    "plasma_radius_profile_cm",
    "machine_radius_profile_cm",
    "neutral_baffle_positions_cm",
    "neutral_baffle_clear_radii_cm",
)
#: The two flags that package requires. ``prescribed_area_geometry`` is what
#: makes the per-cell radii the geometry; ``neutral_baffles`` is what makes the
#: baffle arrays live. They travel as part of the MACHINE rather than as a
#: kinetic input: since B6 the DVM march does read the baffle faces, but only
#: when its own default-off ``neutral_kinetic_dvm_baffles`` flag arms them on
#: top of these, so this package alone still describes the machine and nothing
#: kinetic (the BF gates arm the kinetic flag explicitly).
STANCE_GEOMETRY_FLAGS = ("prescribed_area_geometry", "neutral_baffles")


def _production_geometry_keys():
    """Return the stance's geometry package as ``arm_config`` overrides.

    Read from the committed stance file through the same loader
    ``run_m6_point.py`` and ``baseline_sim1d.py`` use, so this fixture's
    "production geometry" cannot drift from what production runs.
    """
    stance = load_stance(PRODUCTION_STANCE)
    keys = {name: stance.params[name] for name in STANCE_GEOMETRY_PARAMS}
    keys.update(
        {f"flag:{name}": stance.flags[name] for name in STANCE_GEOMETRY_FLAGS}
    )
    return keys


#: The PRODUCTION machine geometry: the g1atrim stance's measured per-cell
#: plasma and vessel radii, its baffles and the mesh they are sized to.
#:
#: This constant previously restated the R5 parametric flare instead (now
#: R5_STANDIN_PARAMS above), which the G1 measured geometry had retired, and
#: rebuilt it on a config whose ``collector_length_cm`` default had since
#: dropped 100 -> 7.8 -- producing ten 0.78 cm end cells no stance ever ran.
#: The gates that step a solver on it were rejecting candidate steps on that
#: mesh alone (measured 2026-08-23).
PRODUCTION_GEOMETRY_KEYS = _production_geometry_keys()


#: The geometries S1 makes its statement on, and what each one is there for.
#: The invariant is a property of the arm, so it must hold on all three: the
#: shipped uniform bore, the R5 stand-in whose plenum obstruction pushes the
#: cathode's live cell off the mesh start, and the production machine.
RECYCLE_GEOMETRIES = (
    ("default", {}),
    ("R5-standin", R5_STANDIN_GEOMETRY_KEYS),
    (PRODUCTION_STANCE, PRODUCTION_GEOMETRY_KEYS),
)


def recycle_identity(geometry_keys, steps=40):
    """Compare the arm's wall-return channels with the plasma actually removed.

    The DESIGN INVARIANT: whatever the plasma-terminating boundary term takes
    out of the plasma at an absorbing face, the arm re-injects as neutrals at
    that same face -- per face, to roundoff. The two sides are read
    independently: the channel rates from ``_kinetic_channel_rates`` (what the
    arm will source), the removal from the boundary term's PLASMA row
    ``-n * V_plasma`` (what left the plasma). Nothing here reads the ``nn``
    return row the implementation samples, so a channel that samples the wrong
    cell cannot satisfy both sides at once.

    The stance sweep this used to run (both values of the retired
    ``characteristic_boundary`` flag) went with that flag on 2026-08-31 (Tom);
    the invariant itself is unchanged and is now asserted against the single
    surviving operator, still across all three geometries.
    """
    overrides = {
        "neutral_kinetic_dvm_nvz": 16,
        "neutral_kinetic_dvm_nvp": 6,
    }
    overrides.update(geometry_keys)
    sim = make_sim(**overrides)
    for _ in range(steps):
        advance_one_step(sim)
    geom = sim.geometry
    roles = np.asarray(geom.cell_role)
    Vp = np.asarray(geom.plasma_volume_cm3, dtype=float)
    state = sim.state
    term = sim.characteristic_boundary_rhs(state=state)
    removed = -np.asarray(term.n, dtype=float) * Vp
    rates = sim._kinetic_channel_rates(state, sim.derived, sim.time)
    by_role = absorbing_live_cells_by_role(geom)
    faces = []
    for role, key in (("cathode", "cath_cells"), ("collector", "coll_cells")):
        channel = np.asarray(rates[key], dtype=float)
        cells = list(by_role.get(role, ()))
        elsewhere = np.delete(channel, cells) if cells else channel
        for cell in cells:
            faces.append(
                {
                    "role": role,
                    "cell": int(cell),
                    "recycled": float(channel[cell]),
                    "removed": float(removed[cell]),
                    "rel": float(
                        abs(channel[cell] - removed[cell])
                        / max(abs(removed[cell]), 1e-300)
                    ),
                    # Nothing of this channel may sit anywhere but its own
                    # faces -- the D3 mis-deposit would show up here.
                    "off_face": float(np.sum(np.abs(elsewhere))),
                }
            )
    return sim, roles, faces


def fmt(x):
    return f"{x:.6e}"


# ------------------------------------------------------------- E3 gates


def gate_i1():
    sim = make_sim()
    ledgers = run_until_updates(sim, 6)
    worst = max(abs(ledger_residual(led)["distribution_rel"]) for led in ledgers)
    ok = worst < ROUNDOFF_REL
    return (
        "I1 particle inventory (distribution form) closes to roundoff",
        ok,
        f"{len(ledgers)} updates, worst |residual|/scale = {fmt(worst)} "
        f"(tol {fmt(ROUNDOFF_REL)})",
    )


def gate_i2():
    sim = make_sim()
    ledgers = run_until_updates(sim, 6)
    worst = max(abs(ledger_residual(led)["domain_rel"]) for led in ledgers)
    # And the same statement accumulated across the whole window, which is
    # the multi-step form the E3 list asks for.
    total = 0.0
    span = 0.0
    for led in ledgers:
        r = ledger_residual(led)
        total += r["domain"]
        span += r["scale"]
    ok = worst < ROUNDOFF_REL and abs(total) / span < ROUNDOFF_REL
    return (
        "I2 particle inventory (domain form) closes to roundoff",
        ok,
        f"per-update worst {fmt(worst)}, window-accumulated "
        f"{fmt(abs(total) / span)} (tol {fmt(ROUNDOFF_REL)})",
    )


def gate_i3():
    sim = make_sim()
    ledgers = run_until_updates(sim, 2)
    led = ledgers[-1]
    missing = [
        f"loss_{name}" for name in LEDGER_LOSS_CHANNELS
        if f"loss_{name}" not in led
    ] + [
        f"birth_{name}" for name in LEDGER_BIRTH_CHANNELS
        if f"birth_{name}" not in led
    ] + [k for k in ("loss_pump_L", "loss_pump_R") if k not in led]
    bookkeeping = set(LEDGER_BOOKKEEPING)
    missing += [k for k in LEDGER_BOOKKEEPING if k not in led]
    # The two MOMENTUM diagnostic rows are PRESENCE-GATED on the anode jet, so
    # they are declared-but-optional rather than declared-and-required. This
    # fixture arms no jet, so the statement here is the STRONGER one: neither
    # row may be present at all. That keeps a presence-gated row from becoming
    # a hole in the completeness gate -- an always-on row could hide behind
    # "optional" -- while AJ4 makes the armed statement about their values.
    momentum = set(LEDGER_MOMENTUM_DIAGNOSTICS)
    present_momentum = sorted(momentum & set(led))
    unaccounted = [
        k
        for k in led
        if k not in bookkeeping
        and k not in momentum
        and not k.startswith("loss_")
        and not k.startswith("birth_")
    ]
    unaccounted += [
        f"{k} (presence-gated row present with no anode jet armed)"
        for k in present_momentum
    ]
    # The nested ENERGY sub-ledger, held to the same standard: every
    # declared row present, and no row present that is not declared.
    energy = led.get("energy", {})
    declared_energy = (
        {f"loss_{name}" for name in LEDGER_ENERGY_LOSS_CHANNELS}
        | {f"birth_{name}" for name in LEDGER_ENERGY_BIRTH_CHANNELS}
        | {f"net_{name}" for name in LEDGER_ENERGY_NET_CHANNELS}
        | set(LEDGER_ENERGY_BOOKKEEPING)
    )
    missing += [
        f"energy.{k}" for k in sorted(declared_energy) if k not in energy
    ]
    unaccounted += [
        f"energy.{k}" for k in energy if k not in declared_energy
    ]
    ok = not missing and not unaccounted
    return (
        "I3 ledger completeness: every declared channel booked",
        ok,
        f"{len(led) - len(bookkeeping)} particle channel entries, "
        f"{len(energy)} energy entries, "
        f"{len(present_momentum)} presence-gated momentum rows (0 required "
        "with no anode jet armed); "
        f"missing={missing or 'none'}; unaccounted={unaccounted or 'none'}",
    )


def gate_i4():
    geom = expanded_end_geometry()
    dvm, worst_dist, worst_dom, transfer_err = geometry_closure(
        geom, "expanded end"
    )
    area_ann = dvm.V_ann / dvm.dz
    jumps = area_ann[1:] / np.maximum(area_ann[:-1], 1e-300)
    jumps = sorted({round(float(r), 3) for r in jumps if abs(r - 1.0) > 1e-9})
    ok = (
        worst_dist < ROUNDOFF_REL
        and worst_dom < ROUNDOFF_REL
        and transfer_err < ROUNDOFF_REL
    )
    return (
        "I4 expanded-end R5 stand-in geometry: closure and transfer exact "
        "across the area jumps",
        ok,
        f"{dvm.nz} cells, annulus area-jump ratios {jumps}; worst "
        f"distribution {fmt(worst_dist)}, domain {fmt(worst_dom)}, "
        f"independent transfer {fmt(transfer_err)} (tol {fmt(ROUNDOFF_REL)})",
    )


def gate_i6():
    """Energy ledger: closed-box identity, and the armed arm at I1/I2 class.

    The registered gate of B0a. The first statement is the closed box
    (:func:`closed_box_dvm`): with every surface and external row disarmed,
    the change in the box's total energy IS the exchange plus the
    consumption booking, to roundoff, and each disarmed row books zero.
    That is a statement about the two live channels and nothing else.

    The second and third statements are the reason the ledger exists at
    all -- a closed box cannot catch a mis-booked wall, mesh, pump, puff or
    recycle row, because it never fires one. So the same two closure forms
    are made with EVERY channel armed at once, once per annulus treatment
    (:func:`all_channels_energy_closure`), and then on the fully-armed
    in-solver default arm, which is the coupled path with the counted
    ionization handshake in it. All of them at ROUNDOFF_REL, the tolerance
    I1 and I2 hold the particle ledger to.
    """
    nz = 8
    dvm = closed_box_dvm(nz)
    plasma = geometry_plasma(nz)
    worst_dist = 0.0
    worst_dom = 0.0
    worst_disarmed = 0.0
    live = float("inf")
    for _ in range(6):
        led = dvm.update(CADENCE_S, **plasma)
        r = ledger_energy_residual(led)
        e = led["energy"]
        worst_dist = max(worst_dist, abs(r["distribution_rel"]))
        worst_dom = max(worst_dom, abs(r["domain_rel"]))
        for key in DISARMED_ENERGY_ROWS:
            worst_disarmed = max(worst_disarmed, abs(e[key]) / r["scale"])
        # Non-vacuity, worst update of the window: the two channels the
        # box is closed around must actually be carrying energy on every
        # update, or the identity being checked is 0 == 0.
        live = min(
            live,
            min(
                abs(e["net_exchange_cx"]) + abs(e["net_exchange_elastic"]),
                abs(e["loss_ionization"]),
            )
            / r["scale"],
        )

    channels = {
        name: all_channels_energy_closure(name)
        for name in ("rates", "bounded_chord")
    }
    armed = run_until_updates(make_sim(), 6)
    armed_dist = max(
        abs(ledger_energy_residual(led)["distribution_rel"]) for led in armed
    )
    armed_dom = max(
        abs(ledger_energy_residual(led)["domain_rel"]) for led in armed
    )
    ok = (
        worst_dist < ROUNDOFF_REL
        and worst_dom < ROUNDOFF_REL
        and worst_disarmed < ROUNDOFF_REL
        and live > 1.0e-6
        and armed_dist < ROUNDOFF_REL
        and armed_dom < ROUNDOFF_REL
        and all(
            max(pair) < ROUNDOFF_REL for pair in channels.values()
        )
    )
    return (
        "I6 energy ledger: closed box closes on exchange + consumption, "
        "every armed channel closes at the particle-ledger tolerance",
        ok,
        f"closed box (6 updates): distribution {fmt(worst_dist)}, domain "
        f"{fmt(worst_dom)}, worst disarmed row {fmt(worst_disarmed)}, live "
        f"channel share {fmt(live)} (> 1e-6 required); every channel armed: "
        + "; ".join(
            f"{name} distribution {fmt(d)}, domain {fmt(m)}"
            for name, (d, m) in channels.items()
        )
        + f"; in-solver arm ({len(armed)} updates): distribution "
        f"{fmt(armed_dist)}, domain {fmt(armed_dom)} "
        f"(tol {fmt(ROUNDOFF_REL)})",
    )


def gate_j1():
    """The bounded-chord classes are the cosine-weighted chords, analytically.

    The three class means are checked against an identity nothing in their
    derivation was fitted to: the two-dimensional mean-chord theorem. The
    annulus cross-section has mean chord ``pi A / P = pi (Rm - Rp) / 2``
    over all its surfaces, and the same average taken over the CLASSES is
    the perimeter-weighted mix of the outer-wall branch (the view factor
    splitting ``c_wi`` from ``c_ww``) and the inner-surface branch
    ``c_io``. The two must agree, and the only error is the sampling of the
    emission angle.
    """
    geom = expanded_end_geometry()
    Rp = np.asarray(geom.Rp_cm, dtype=float)
    Rm = np.asarray(geom.Rm_cm, dtype=float)
    F, c_ww, c_wi, c_io, v_ww, v_wi, v_io = annulus_chord_classes(Rp, Rm)
    wall_mean = (1.0 - F) * c_ww + F * c_wi
    mixed = (Rm * wall_mean + Rp * c_io) / (Rm + Rp)
    analytic = np.pi * (Rm - Rp) / 2.0
    err = float(np.max(np.abs(mixed - analytic) / analytic))
    sharp = {
        "ww": c_ww**2 / np.maximum(v_ww, 1e-300),
        "wi": c_wi**2 / np.maximum(v_wi, 1e-300),
        "io": c_io**2 / np.maximum(v_io, 1e-300),
    }
    # An exponential flight time -- what the rate arm's nuw and nuxp imply --
    # has mean^2/var exactly 1. Every class must be narrower than that; the
    # measured duct values are the ~10 and ~200 the kernel was built for.
    ok = err < 1.0e-4 and all(float(np.min(s)) > 1.0 for s in sharp.values())
    duct = int(np.argmax(np.isclose(Rm, np.min(Rm))))
    return (
        "J1 bounded-chord classes satisfy the 2D mean-chord theorem",
        ok,
        f"max relative departure from pi (Rm - Rp) / 2 = {fmt(err)} "
        f"(tol 1e-4, the angle sampling); duct cell {duct} "
        f"(Rp={Rp[duct]:g}, Rm={Rm[duct]:g}) chords "
        f"ww {c_ww[duct]:.3f} wi {c_wi[duct]:.3f} io {c_io[duct]:.3f} cm, "
        f"mean^2/var ww {sharp['ww'][duct]:.2f} wi {sharp['wi'][duct]:.2f} "
        f"io {sharp['io'][duct]:.2f} (an exponential flight time is 1.00)",
    )


def gate_j2():
    """The jump operator conserves on the R5 stand-in expanded-end geometry.

    The I4 statement, made against the bounded-chord annulus: the routing
    map itself must send every launched particle to exactly one outcome,
    and the running engine must close both ledger forms and reproduce the
    booked transfer independently, across the same annulus area jumps the
    throat-face flux form exists for.
    """
    geom = expanded_end_geometry()
    dvm, worst_dist, worst_dom, transfer_err = geometry_closure(
        geom, "expanded end", annulus_flights="bounded_chord"
    )
    area_ann = dvm.V_ann / dvm.dz
    jumps = area_ann[1:] / np.maximum(area_ann[:-1], 1e-300)
    jumps = sorted({round(float(r), 3) for r in jumps if abs(r - 1.0) > 1e-9})
    routing = dvm.flights.residual
    parts = sum(dvm.f_flight.values())
    split = float(np.max(np.abs(dvm.f_a - parts)))
    ok = (
        worst_dist < ROUNDOFF_REL
        and worst_dom < ROUNDOFF_REL
        and transfer_err < ROUNDOFF_REL
        and routing == 0.0
        and split == 0.0
    )
    return (
        "J2 bounded-chord annulus: routing and closure exact across the "
        "area jumps",
        ok,
        f"{dvm.nz} cells, annulus area-jump ratios {jumps}; flight-map "
        f"routing residual {fmt(routing)} (exact 0 required); "
        f"f_a - sum(in-flight classes) {fmt(split)} (exact 0 required); "
        f"worst distribution {fmt(worst_dist)}, domain {fmt(worst_dom)}, "
        f"independent transfer {fmt(transfer_err)} (tol {fmt(ROUNDOFF_REL)})",
    )


def gate_j3():
    """Naming the shipped selector changes nothing, bit for bit."""
    geom = expanded_end_geometry()
    nz = int(np.asarray(geom.length_cm).size)
    states = []
    for kwargs in ({}, {"annulus_flights": "rates"}):
        dvm = TransientDVM(
            geometry=geom, nvz=16, nvp=6, s_L=0.3, s_R=0.3,
            exchange_model=EXCHANGE_MODEL, **kwargs
        )
        dvm.seed_from_density(
            np.full(nz, 1.0e13), np.where(dvm.V_ann > 0.0, 1.0e13, 0.0)
        )
        plasma = geometry_plasma(nz)
        rec = np.full(nz, 1.0e15)
        for _ in range(6):
            dvm.update(CADENCE_S, sources={"recombination": rec}, **plasma)
        states.append((dvm.f_c.copy(), dvm.f_a.copy(),
                       dvm.M_transfer.copy(), dvm.Ei_transfer.copy()))
    same = all(
        a.tobytes() == b.tobytes() for a, b in zip(states[0], states[1])
    )
    return (
        "J3 annulus_flights='rates' is bit-identical to not naming it",
        same,
        "f_c, f_a and both transfer rows compared as raw bytes after 6 "
        f"updates on the expanded-end geometry: "
        f"{'identical' if same else 'DIFFER'}",
    )


def gate_i5():
    geom = zero_annulus_tube()
    dvm, worst_dist, worst_dom, transfer_err = geometry_closure(
        geom, "zero annulus"
    )
    empty = dvm.V_ann <= 0.0
    leaked = float(np.max(dvm.annulus_density()[empty]))
    ok = (
        worst_dist < ROUNDOFF_REL
        and worst_dom < ROUNDOFF_REL
        and transfer_err < ROUNDOFF_REL
        and leaked == 0.0
    )
    return (
        "I5 zero-annulus geometry: closure and transfer exact, no leakage "
        "into volumeless cells",
        ok,
        f"{dvm.nz} cells, {int(empty.sum())} with V_ann = 0; worst "
        f"distribution {fmt(worst_dist)}, domain {fmt(worst_dom)}, "
        f"independent transfer {fmt(transfer_err)}; annulus density in the "
        f"volumeless cells {fmt(leaked)} (exactly zero required)",
    )


def gate_s1():
    lines = []
    ok = True
    for geometry_name, geometry_keys in RECYCLE_GEOMETRIES:
        sim, roles, faces = recycle_identity(geometry_keys)
        label = f"{geometry_name}"
        for face in faces:
            cell = face["cell"]
            # The invariant, plus the two structural statements the
            # positional-constant defect violated: the channel is live and
            # it sits on the role-resolved cell -- which is cell 2 on the
            # R5 stand-in, whose plenum obstruction is the only geometry
            # here that moves the cathode off the mesh start.
            face_ok = (
                face["rel"] < ROUNDOFF_REL
                and face["removed"] > 0.0
                and face["recycled"] > 0.0
                and face["off_face"] == 0.0
                and str(roles[cell]) == face["role"]
            )
            ok = ok and face_ok
            lines.append(
                f"{label} {face['role']}@cell{cell}: "
                f"recycled {fmt(face['recycled'])} vs removed "
                f"{fmt(face['removed'])}, rel {fmt(face['rel'])}, "
                f"off-face {fmt(face['off_face'])}"
            )
        dep = (
            f"{label} deposit cells: cath={sim._dvm.cath_cell} "
            f"coll={sim._dvm.coll_cell}"
        )
        expected = {
            f["role"]: f["cell"] for f in faces
        }
        dep_ok = (
            sim._dvm.cath_cell == expected.get("cathode")
            and sim._dvm.coll_cell == expected.get("collector")
        )
        ok = ok and dep_ok
        lines.append(dep + f" (matches the sampled faces: {dep_ok})")
    return (
        "S1 recycle identity: what the arm re-injects equals what the "
        "boundary removed, per face, on all three geometries",
        ok,
        ("\n        ").join(lines) + f"\n        tol {fmt(ROUNDOFF_REL)}",
    )


def gate_s2():
    """Per-end pump fidelity: each end realizes the speed it was configured.

    The engine books ``loss_pump_END = s_END * (counted end-face outflow)``
    and the march counts that outflow through THAT end's own open area, so
    the sticking coefficient is the only place the configured pumping speed
    enters. The identity that pins it, per end, is

        s_END * A_END * vbar(300 K) / 4  ==  S_eff_END

    -- area x sticking x incident flux, read as a speed -- with ``S_eff`` the
    configured speed after the plenum elbow's series conductance. The booked
    half is taken off a LIVE ledger rather than assumed.

    Run on the PRODUCTION machine, whose plenum and collector faces differ in
    area by 3.63x, with unequal speeds at the two ends and the elbow set, so
    one shared area, a swapped end, or a dropped elbow cannot satisfy both
    ends at once. ``vbar`` is rebuilt here from ``m_He_cgs`` -- the neutral
    mass the DVM march itself carries, and so the one the sticking
    probability must be normalized by for the realized speed to be the
    configured one -- taken from the constants module rather than read back
    through the routine under test: this gate is about the AREA and the
    elbow.
    """
    S_L, S_R, C_elbow = 2000.0, 5000.0, 4000.0
    overrides = {
        "neutral_kinetic_dvm_nvz": 16,
        "neutral_kinetic_dvm_nvp": 6,
        "S_pump_L": S_L,
        "S_pump_R": S_R,
        "pump_elbow_conductance_lps": C_elbow,
    }
    overrides.update(PRODUCTION_GEOMETRY_KEYS)
    sim = make_sim(**overrides)
    dvm = sim._dvm
    geom = sim.geometry
    ledger = run_until_updates(sim, 2)[-1]
    vbar = np.sqrt(8.0 * kb_cgs * 300.0 / (np.pi * m_He_cgs))
    lines = []
    ok = True
    areas = {}
    for label, index, speed, sticking, out_key, pump_key in (
        ("L", 0, S_L, dvm.s_L, "loss_end_out_L", "loss_pump_L"),
        ("R", -1, S_R, dvm.s_R, "loss_end_out_R", "loss_pump_R"),
    ):
        # The march's own end-face area: both zones cross the same face.
        area = float(dvm.face_c[index] + dvm.face_a[index])
        areas[label] = area
        plenum = is_plenum_cell(geom, index)
        s_eff = _effective_pump_speed(speed, C_elbow if plenum else None)
        realized = sticking * area * vbar / 4.0 / 1.0e3
        rel = abs(realized - s_eff) / s_eff
        incident = float(ledger[out_key])
        booked = float(ledger[pump_key])
        book_rel = abs(booked - sticking * incident) / max(booked, 1e-300)
        end_ok = (
            rel < ROUNDOFF_REL
            and book_rel < ROUNDOFF_REL
            # Non-vacuity: the clip is not binding (which would make the
            # identity false by design) and gas is actually arriving.
            and 0.0 < sticking < 1.0
            and incident > 0.0
        )
        ok = ok and end_ok
        lines.append(
            f"end {label} (cell {index}, plenum={plenum}): area "
            f"{fmt(area)} cm^2, s={fmt(sticking)}, realized "
            f"{fmt(realized)} L/s vs S_eff {fmt(s_eff)} L/s, rel "
            f"{fmt(rel)}; booked {fmt(booked)} vs s*incident "
            f"{fmt(sticking * incident)}, rel {fmt(book_rel)}"
        )
    # The fixture must actually exercise what the identity distinguishes:
    # two different end areas, and an elbow that bites on the plenum end
    # and is absent on the collector end.
    area_ratio = areas["R"] / areas["L"]
    elbow_bites = _effective_pump_speed(S_L, C_elbow) < S_L
    fixture_ok = (
        area_ratio > 1.1
        and elbow_bites
        and is_plenum_cell(geom, 0)
        and not is_plenum_cell(geom, -1)
    )
    ok = ok and fixture_ok
    lines.append(
        f"fixture: end-area ratio R/L = {area_ratio:.6f}, elbow bites on the "
        f"plenum end = {elbow_bites}, collector end takes no elbow = "
        f"{not is_plenum_cell(geom, -1)}"
    )
    return (
        "S2 per-end pump fidelity: sticking x own-end area x incident flux "
        "is the configured speed, elbow in series on the plenum end",
        ok,
        ("\n        ").join(lines) + f"\n        tol {fmt(ROUNDOFF_REL)}",
    )


def gate_c1():
    sim = make_sim()
    run_until_updates(sim, 3)
    dvm = sim._dvm
    term = sim.rhs_terms()["neutral_kinetic_dvm_coupling"]
    active = np.asarray(sim._geometry.plasma_active, dtype=bool)
    lhs = np.asarray(term.M, dtype=float)
    rhs = np.where(active, dvm.M_transfer, 0.0)
    err = np.max(np.abs(lhs - rhs))
    scale = max(np.max(np.abs(rhs)), 1e-300)
    ok = err == 0.0 or err / scale < ROUNDOFF_REL
    return (
        "C1 momentum transfer is exactly minus the kinetic moment",
        ok,
        f"max |fluid M - kinetic moment| = {fmt(err)} on scale {fmt(scale)}",
    )


def gate_c2():
    sim = make_sim()
    run_until_updates(sim, 3)
    dvm = sim._dvm
    term = sim.rhs_terms()["neutral_kinetic_dvm_coupling"]
    active = np.asarray(sim._geometry.plasma_active, dtype=bool)
    lhs = np.asarray(term.Ei, dtype=float)
    rhs = np.where(active, dvm.Ei_transfer, 0.0)
    err = np.max(np.abs(lhs - rhs))
    scale = max(np.max(np.abs(rhs)), 1e-300)
    # NB this gate deliberately makes ONE statement -- that the fluid Ei row
    # is the engine's booked Ei transfer on the active cells. The K2a build
    # carried a second "independent reconstruction" leg here that computed
    # its two sides from the same expression and so could only ever print
    # 0.0; it was removed rather than repaired in place, because the
    # decomposition it meant to test needs inputs this sim-level gate does
    # not have. C4 makes that statement properly.
    ok = err == 0.0 or err / scale < ROUNDOFF_REL
    return (
        "C2 ion energy transfer is the kinetic energy moment, bulk removed",
        ok,
        f"max |fluid Ei - kinetic closure| = {fmt(err)} on scale {fmt(scale)} "
        f"(the decomposition itself is gated by C4)",
    )


def gate_c3():
    dvm = bare_dvm()
    lhs = dvm.V_col[:, None] * dvm.nux
    rhs = dvm.V_ann[:, None] * dvm.nuxp
    scale = max(np.max(np.abs(rhs)), 1e-300)
    err = np.max(np.abs(lhs - rhs))
    ok = err / scale < ROUNDOFF_REL
    return (
        "C3 zone-exchange conductance is antisymmetric",
        ok,
        f"max |V_col nu_x - V_ann nu_xp| = {fmt(err)} on scale {fmt(scale)}",
    )


def gate_c4():
    """The booked transfer, rebuilt from the post-update state alone."""
    nz = 10
    dvm = bare_dvm(nz=nz, nvz=16, nvp=6)
    dvm.s_L = 0.3
    dvm.s_R = 0.3
    dvm.seed_from_density(np.full(nz, 1.0e13), np.full(nz, 1.0e13))
    plasma = geometry_plasma(nz)
    rec = np.full(nz, 1.0e15)
    dt = CADENCE_S
    for _ in range(3):
        dvm.update(dt, sources={"recombination": rec}, **plasma)
    want = independent_transfer(dvm, dt, plasma, rec)
    rows = []
    ok = True
    for name, engine, rebuilt in zip(
        ("M", "Ei", "S"),
        (dvm.M_transfer, dvm.Ei_transfer, dvm.S_transfer),
        want,
    ):
        scale = max(np.max(np.abs(rebuilt)), 1e-300)
        err = float(np.max(np.abs(engine - rebuilt)) / scale)
        rows.append(f"{name} {fmt(err)}")
        ok = ok and err < ROUNDOFF_REL
    return (
        "C4 booked transfer matches an independent reconstruction",
        ok,
        "relative error vs the rebuilt moments: "
        + ", ".join(rows)
        + f" (tol {fmt(ROUNDOFF_REL)}); marched state recovered from the "
        "public post-update f_c, birth moments taken analytically",
    )


def gate_c5():
    """The isotropic-elastic BGK rate carries the one-half factor exactly.

    A BGK full-replacement event transfers the whole ``m (v - u_i)``, which
    is right for backscatter and twice the isotropic angular average, so
    the isotropic rate is halved. Two statements are exact and are gated:
    the factor is applied bit-exactly, and the resulting rate coefficient
    is the closed form ``0.5 <sigma_iso v_rel>`` -- well defined at ANY
    temperature because ``sigma_iso ~ 1/v`` makes ``sigma_iso v``
    velocity-independent. The cx channel has genuine velocity dependence
    and no single-number correspondence, so its ratio is reported only.
    """
    nz = 4
    dvm = bare_dvm(nz=nz, nvz=48, nvp=12)
    g = dvm.g
    n_i = np.full(nz, 5.0e12)
    Ti = np.array([0.1, 0.5, 2.0, 8.0])
    u = np.array([0.0, 1.0e5, -2.0e5, 5.0e5])
    nu_cx, nu_el = dvm.collision_frequencies(n_i, Ti, u)

    # (a) the halving itself, against the unhalved expression. The g_eff here
    # is a TRANSCRIPTION of collision_frequencies' own thermal floor and must
    # track it: the ratio below isolates the 1/2 factor only when both sides
    # are built on the same g_eff. (sigma_iso g_eff is analytically
    # g_eff-independent but not bitwise so, which is what part (a) resolves.)
    w2 = (g.VZ[None, :, :] - u[:, None, None]) ** 2 + (g.VP**2)[None, :, :]
    g_eff = np.sqrt(w2 + 8.0 * Ti[:, None, None] * EV / (np.pi * M_HE))
    E_rel = np.maximum(0.25 * M_HE * g_eff**2 / EV, 1e-9)
    unhalved = n_i[:, None, None] * phelps_he_isotropic_cm2(E_rel) * g_eff
    ratio = nu_el / unhalved
    halved_exactly = bool(
        np.all(ratio == ELASTIC_BGK_MOMENTUM_FACTOR)
    ) and ELASTIC_BGK_MOMENTUM_FACTOR == 0.5

    # (b) the rate coefficient against the closed form, and its constancy.
    k_el = nu_el / n_i[:, None, None]
    want = 0.5 * CLOSED_ISO_RATE_CM3_S
    closed_err = float(np.max(np.abs(k_el - want)) / want)

    # (c) the off arm is exactly zero.
    off = bare_dvm(nz=nz, nvz=16, nvp=6, elastic_model="off")
    _, nu_el_off = off.collision_frequencies(
        n_i, Ti, u
    )
    off_zero = bool(np.all(nu_el_off == 0.0))

    ok = halved_exactly and closed_err < ROUNDOFF_REL and off_zero

    # Reported context: how far the fluid TABLE sits from its own closed
    # form, and the total effective rate against a 300 K neutral Maxwellian,
    # which the halving moves toward the fluid operator but cannot make exact
    # (the cx channel's g_eff interpolation is not a Maxwellian rate
    # average). The table's quadrature used to run on a linear E grid, where
    # the E^-0.5 factor left the integrand non-smooth at the origin and the
    # table landed ~1.2e-3 BELOW the closed form; it now integrates in
    # sqrt(E) and agrees to ~3e-7, so the table is no longer the looser of
    # the two references.
    table = float(phelps_iso_rate_cm3_s(0.5))
    table_rel = table / CLOSED_ISO_RATE_CM3_S - 1.0
    fM = g.maxwellian(300.0 * KB / EV, 0.0)
    totals = []
    for i in range(nz):
        k_kin = float(((nu_cx[i] + nu_el[i]) * fM).sum()) / n_i[i]
        k_pre = k_kin + float((nu_el[i] * fM).sum()) / n_i[i]
        k_fluid = float(
            phelps_momentum_transfer_rate_cm3_s(0.5 * (Ti[i] + 300.0 * KB / EV))
        )
        totals.append(
            f"Ti={Ti[i]:g}: {k_kin / k_fluid:.3f} (was {k_pre / k_fluid:.3f})"
        )
    return (
        "C5 elastic BGK rate carries the 1/2 momentum-transfer factor",
        ok,
        f"nu_el/unhalved exactly {ELASTIC_BGK_MOMENTUM_FACTOR} everywhere: "
        f"{halved_exactly}; k_el vs closed 0.5<sigma_iso v> = "
        f"{fmt(want)} cm^3/s, relative error {fmt(closed_err)} "
        f"(tol {fmt(ROUNDOFF_REL)}); elastic_model='off' exactly zero: "
        f"{off_zero}. Context (not gated): the tabulated "
        f"phelps_iso_rate_cm3_s is {table_rel:+.2e} off its own closed form "
        f"(quadrature); total k/k_fluid vs a 300 K Maxwellian -- "
        + "; ".join(totals),
    )


def gate_c6():
    """The kinetic/fluid rate-coefficient ratio stays inside its bracket.

    WHAT IS MEASURED. Both arms are approximations to the SAME object -- the
    ``k_b + 0.5 k_iso`` channel-weighted rate coefficient of a two-Maxwellian
    He/He+ pair -- so this is a residual budget, not a kinetic effect scored
    against a fluid one. The quantity is a NUMBER-WEIGHTED rate coefficient
    against a STATIONARY 300 K neutral Maxwellian: the kinetic arm's per-bin
    ``(nu_cx + nu_el)/n_i`` averaged over ``g.maxwellian(300 K, u = 0)``,
    divided by ``phelps_momentum_transfer_rate_cm3_s((Ti + Tn)/2)``. It is
    NOT a total drag; the momentum-weighted ratio is a different number and
    differs by up to 1.4 %.

    The fluid side is EXACT in the drift-free two-Maxwellian limit (it is
    ``<sigma g>`` over the reduced-mass relative Maxwellian at
    ``T_eff = (Ti + Tn)/2``) and it neglects the ion drift. The kinetic side
    evaluates ``n_i sigma(E(g_eff)) g_eff`` per velocity bin with the
    ion-only thermal floor. THREE residuals separate them, and each is
    SIGN-DEFINITE >= 0 for the Phelps pair, which is why the lower edge is
    exactly 1.0 and not "1 +- quadrature":

    (i)   the ``sqrt(w^2 + c_bar^2)`` interpolation is an UPPER bound on
          ``<|v - u_i|>``, by <= +2.5 %, peaking near ``w/a ~ 1.2-1.5`` --
          which is exactly where a 300 K neutral sits against 0.03-0.1 eV
          ions;
    (ii)  single-energy evaluation ``sigma(E(g_eff)) g_eff`` against the
          average ``<sigma g>`` (Jensen, with ``sigma_b`` monotone
          DECREASING in E: ``d ln sigma_b/dE = -0.15/(E + 5) -
          0.25/(1000 + E) < 0``), <= +2.5 % for ``Ti <= 8`` eV;
    (iii) the fluid rate's neglect of the drift, ``~ x^2/3`` with
          ``x = u_i/a``, ~2 % at the largest drift probed here.

    PER-POINT DECOMPOSITION, recorded so a future move can be CLASSIFIED
    rather than merely re-bracketed. Predicted residual sums over the four
    probe points (``RATE_COEFF_RESIDUAL_PREDICTED_PCT``) are 0.95/1.9/2.8/
    4.4 %; observed ratios are 1.0091/1.0183/1.0258/1.0428. Grid quadrature
    is NOT part of the residual: 48x12, 96x24 and 192x48 agree to four
    decimals.

    THE BRACKET. Lower edge 1.00 is exact by the sign-definiteness above (the
    exact erf form leaves the (0.1 eV, u = 0) point at 1.0006, and the 0.9 %
    headroom there is deterministic and grid-converged). Upper edge 1.10 is
    the derived worst-case residual budget (<= ~7.5 %) with headroom, chosen
    from the admissible range [1.08, 1.15]. It still catches both structural
    errors this operator has actually exhibited: the reduced-mass thermal
    floor prints >= 1.22, and dropping the isotropic channel's one-half
    momentum-transfer factor prints ~1.39/1.25/1.17/1.12. The superseded
    [1.0, 1.7] caught NEITHER.

    The one-half factor is additionally gated exactly, bit-for-bit, by C5;
    the two rows are complementary rather than redundant.
    """
    nz = 4
    dvm = bare_dvm(nz=nz, nvz=48, nvp=12)
    g = dvm.g
    n_i = np.full(nz, 5.0e12)
    Ti = np.array([0.1, 0.5, 2.0, 8.0])
    u = np.array([0.0, 1.0e5, -2.0e5, 5.0e5])
    nu_cx, nu_el = dvm.collision_frequencies(n_i, Ti, u)
    Tn_eV = 300.0 * KB / EV
    fM = g.maxwellian(Tn_eV, 0.0)

    ratios = []
    for i in range(nz):
        k_kin = float(((nu_cx[i] + nu_el[i]) * fM).sum()) / n_i[i]
        k_fluid = float(
            phelps_momentum_transfer_rate_cm3_s(0.5 * (Ti[i] + Tn_eV))
        )
        ratios.append(k_kin / k_fluid if k_fluid else float("inf"))

    lo, hi = RATE_COEFF_RATIO_BRACKET
    ok = all(lo <= r <= hi for r in ratios)
    worst = min(ratios, key=lambda r: min(r - lo, hi - r))
    return (
        "C6 kinetic/fluid rate-coefficient ratio inside its bracket",
        ok,
        f"bracket [{lo:.2f}, {hi:.2f}] (re-registered 2026-08-23e as a "
        f"DERIVED residual budget; number-weighted rate coefficient against "
        f"a stationary 300 K Maxwellian, NOT a total drag); "
        + "; ".join(
            f"Ti={Ti[i]:g}: {ratios[i]:.4f} "
            f"(predicted +{RATE_COEFF_RESIDUAL_PREDICTED_PCT[i]:g}%, "
            f"observed +{100.0 * (ratios[i] - 1.0):.2f}%)"
            for i in range(nz)
        )
        + f"; closest to an edge: {worst:.4f}",
    )


def _dvm_fingerprint(dvm):
    return tuple(
        np.ascontiguousarray(a, dtype=float).tobytes()
        for a in (
            dvm.f_c,
            dvm.f_a,
            dvm.pend_L_c,
            dvm.pend_R_c,
            dvm.pend_L_a,
            dvm.pend_R_a,
            dvm.M_transfer,
            dvm.Ei_transfer,
            dvm.S_transfer,
            dvm.Tn_col_eV,
        )
    ) + (dvm.updates, tuple(sorted((dvm.last_ledger or {}).items())))


def gate_r1():
    sim = make_sim()
    run_until_updates(sim, 2)
    before = _dvm_fingerprint(sim._dvm)
    # A rejected attempt: build one and discard it, exactly as the retry
    # path does. The attempt machinery is what a dt-retry re-runs.
    for _ in range(4):
        sim._attempt_step(dt=1.0e-7)
        sim._attempt_step(dt=1.0e-12)
    after = _dvm_fingerprint(sim._dvm)
    ok = before == after
    return (
        "R1 rejected attempts leave the kinetic state bit-identical",
        ok,
        f"8 discarded attempts; fingerprint match={ok} "
        f"(updates {sim._dvm.updates})",
    )


def gate_r2():
    sim = make_sim()
    run_until_updates(sim, 2)
    before = _dvm_fingerprint(sim._dvm)
    y = sim._y.copy()
    for scale in (1.0, 1.05, 0.95):
        sim.rhs(y=y * scale)
        sim.rhs_terms(y=y * scale)
    after = _dvm_fingerprint(sim._dvm)
    ok = before == after
    return (
        "R2 trial RHS evaluations do not advance the neutral clock",
        ok,
        f"6 trial evaluations at perturbed states; match={ok}",
    )


def gate_p1():
    d, fl = default_config()
    ref = LAPDSim1D(input_dict=dict(d), input_flags=dict(fl))
    for _ in range(20):
        advance_one_step(ref)
    terms = ref.rhs_terms()
    no_arm_term = not any("dvm" in name for name in terms)
    no_arm = ref._dvm is None
    # Bit-exactness of the off path against a build with the two-zone state
    # on but the arm still off -- the nearest neighbour configuration.
    fl2 = dict(fl)
    fl2["neutral_two_zone"] = True
    alt = LAPDSim1D(input_dict=dict(d), input_flags=fl2)
    for _ in range(20):
        advance_one_step(alt)
    alt_clean = alt._dvm is None and not any(
        "dvm" in name for name in alt.rhs_terms()
    )
    ok = no_arm and no_arm_term and alt_clean
    return (
        "P1 default off: no arm, no arm term, in either neutral stance",
        ok,
        f"shipped build arm={ref._dvm}; two-zone build clean={alt_clean}; "
        f"{len(terms)} terms, none named for the arm",
    )


def gate_p2():
    sim = make_sim()
    # Before engagement the fluid terms must be untouched.
    pre = sim.rhs_terms()
    pre_neutral_live = np.any(pre["neutral_sources"].nn != 0.0) or np.any(
        pre["ionization_birth"].nn != 0.0
    )
    pre_coupling_zero = np.all(
        pre["neutral_kinetic_dvm_coupling"].M == 0.0
    ) and np.all(pre["neutral_kinetic_dvm_coupling"].Ei == 0.0)
    run_until_updates(sim, 2)
    post = sim.rhs_terms()
    neutral_rows_zero = all(
        np.all(np.asarray(t.nn, dtype=float) == 0.0)
        and (t.nn_a is None or np.all(np.asarray(t.nn_a, dtype=float) == 0.0))
        for name, t in post.items()
        if name != "neutral_kinetic_dvm_coupling"
    )
    superseded = sim._DVM_TRANSFER_TERMS | sim._DVM_BIRTH_TERMS
    transfer_zero = all(
        np.all(np.asarray(post[name].M, dtype=float) == 0.0)
        and np.all(np.asarray(post[name].Ei, dtype=float) == 0.0)
        for name in superseded
        if name in post
    )
    births_live = np.any(post["ionization_birth"].n != 0.0)
    coupling_live = np.any(post["neutral_kinetic_dvm_coupling"].M != 0.0)
    ok = (
        pre_neutral_live
        and pre_coupling_zero
        and neutral_rows_zero
        and transfer_zero
        and births_live
        and coupling_live
    )
    return (
        "P2 presence gating: fluid rows handed over exactly at engagement",
        ok,
        f"pre-engage fluid neutral rows live={pre_neutral_live}, coupling "
        f"zero={pre_coupling_zero}; post-engage neutral rows zero="
        f"{neutral_rows_zero}, superseded M/Ei zero={transfer_zero}, "
        f"particle rows still live={births_live}, coupling live={coupling_live}",
    )


def gate_p3():
    sim = make_sim()
    run_until_updates(sim, 3)
    floor = sim.floors["nn"]
    col = np.maximum(sim._dvm.column_density(), floor)
    ann = np.maximum(sim._dvm.annulus_density(), floor)
    ok = np.array_equal(sim.state.nn, col) and np.array_equal(
        sim.state.nn_a, ann
    )
    return (
        "P3 moment consistency: saved nn IS the column zeroth moment",
        ok,
        f"nn == moment: {np.array_equal(sim.state.nn, col)}; "
        f"nn_a == annulus moment: {np.array_equal(sim.state.nn_a, ann)}",
    )


# --------------------------------------------------------- K2d D gates


def engaged_production_sim(**overrides):
    """Return an ENGAGED arm on the PRODUCTION machine geometry."""
    kwargs = {
        "neutral_kinetic_dvm_nvz": 16,
        "neutral_kinetic_dvm_nvp": 6,
        **PRODUCTION_GEOMETRY_KEYS,
    }
    kwargs.update(overrides)
    sim = make_sim(**kwargs)
    run_until_updates(sim, 1)
    return sim


# The bounds the engaged arm's supersession makes PHANTOM: it zeroes the row
# each one describes, so a value here can only misreport. They must read inf
# while the arm is engaged, and the term rows they describe must be zero.
PHANTOM_BOUNDS = {
    "dt_ion_charge_exchange": ("ion_charge_exchange", "Ei"),
    "dt_ion_neutral_drag": ("ion_neutral_drag", "M"),
    "dt_neutral_exchange": ("neutral_exchange", "nn"),
    "dt_neutral_sources": ("neutral_sources", "nn"),
}


def gate_d1():
    """The coupling term bounds dt (the inverse of the crash's guard proof)."""
    sim = engaged_production_sim()
    base = sim.suggest_timestep()
    responses = []
    ok = True
    for drain in (1.0e10, 1.0e11, 1.0e12):
        sim._dvm.Ei_transfer = np.full(sim._geometry.cells, -float(drain))
        diag = sim.suggest_timestep()
        responses.append((drain, diag.dt_surface_loss, diag.active_constraint))
    # It must MOVE (the defect was a bound that could not see the term at
    # all), it must move the right way, and it must scale like 1/drain --
    # a floor-margin bound on a linear drain is exactly inversely
    # proportional, which no accidental coupling would reproduce.
    ok = ok and all(r[1] < base.dt_surface_loss for r in responses)
    ok = ok and responses[0][1] > responses[1][1] > responses[2][1]
    ratio = responses[0][1] / responses[2][1]
    ok = ok and abs(ratio - 100.0) < 1.0e-6 * 100.0
    ok = ok and all(r[2] == "surface_loss" for r in responses)
    detail = f"unbounded-drain baseline dt_surface_loss = {fmt(base.dt_surface_loss)}"
    for drain, dt_sl, constraint in responses:
        detail += (
            f"\n        Ei_transfer = -{fmt(drain)} erg/cm3/s -> "
            f"dt_surface_loss {fmt(dt_sl)} ({constraint})"
        )
    detail += f"\n        x100 drain -> x{ratio:.6f} shorter bound (exactly 100 required)"
    return (
        "D1 the DVM coupling drain BOUNDS the timestep, inversely and "
        "through surface_loss",
        ok,
        detail,
    )


def gate_d2():
    """No superseded term's bound survives to be reported or to set dt."""
    from cablp.solvers._sim1d.core.timestep import ion_charge_exchange_timestep

    sim = engaged_production_sim()
    # The phantom is REAL as a number -- this is what used to set dt while
    # naming a term whose applied row is identically zero.
    phantom = ion_charge_exchange_timestep(
        state=sim.state,
        floors=sim.floors,
        ion_mass_g=sim._ion_mass_g,
        ion_charge_exchange_kwargs=sim._ion_charge_exchange_kwargs(),
        density_dt_fraction=0.25,
        plasma_active=(
            sim._geometry.plasma_active if sim._active_plasma_topology else None
        ),
    )
    terms = sim.rhs_terms()
    lines = [
        f"unstripped ion_charge_exchange bound (the phantom) = {fmt(phantom)} s"
    ]
    ok = np.isfinite(phantom)
    constraints = set()
    for _ in range(25):
        diag = sim.suggest_timestep()
        constraints.add(diag.active_constraint)
        for field, (term, row) in PHANTOM_BOUNDS.items():
            ok = ok and getattr(diag, field) == np.inf
            ok = ok and np.all(
                np.asarray(getattr(terms[term], row), dtype=float) == 0.0
            )
        advance_one_step(sim)
        terms = sim.rhs_terms()
    withdrawn = {name.removeprefix("dt_") for name in PHANTOM_BOUNDS}
    ok = ok and not (constraints & withdrawn)
    for field, (term, row) in PHANTOM_BOUNDS.items():
        lines.append(
            f"{field} = inf while the applied {term}.{row} row is exactly zero"
        )
    lines.append(
        f"active_constraint over 25 engaged steps: {sorted(constraints)}"
    )
    return (
        "D2 every phantom bound is withdrawn: active_constraint names only "
        "terms the step applies",
        ok,
        ("\n        ").join(lines),
    )


def gate_d3():
    """The floor-aware relax defers transfer; it never destroys any."""
    sim = engaged_production_sim()
    # Inert first: with no limiting the applied rate is the booked rate
    # BIT-exactly, so the relax cannot perturb a healthy run.
    advance_one_step(sim)
    quiet = sim.dvm_transfer_ledger()
    quiet_limited = quiet["relax_limited_steps"]
    # Now a drain no admissible step could carry: 1e12 erg/cm3/s against
    # margins of order 1e3 erg/cm3 is the crash's regime, amplified.
    cells = sim._geometry.cells
    sim._dvm.Ei_transfer = np.full(cells, -1.0e12)
    sim._dvm.M_transfer = np.full(cells, -1.0e3)
    survived = True
    try:
        for _ in range(30):
            advance_one_step(sim)
    except Exception as error:  # noqa: BLE001 - the gate's whole point
        survived = False
        crash = f"{type(error).__name__}: {error}"
    ledger = sim.dvm_transfer_ledger()
    engaged_cells = int(np.count_nonzero(ledger["relax_cell_steps"]))
    ok = (
        survived
        and quiet_limited == 0
        and ledger["relax_limited_steps"] > 0
        and ledger["Ei"]["rel"] < ROUNDOFF_REL
        and ledger["M"]["rel"] < ROUNDOFF_REL
        and np.any(np.abs(sim._dvm.Ei_debt) > 0.0)
    )
    detail = (
        f"quiet run: {quiet['relax_steps']} steps, "
        f"{quiet_limited} limited (0 required -- the relax is inert on a "
        f"healthy step and applied == booked bit-exactly)\n        "
        f"forced -1e12 erg/cm3/s: survived={survived}, "
        f"{ledger['relax_limited_steps']} of {ledger['relax_steps']} steps "
        f"limited over {engaged_cells} cells\n        "
        f"ledger |applied_cum + debt - booked_cum| / scale: "
        f"Ei {fmt(ledger['Ei']['rel'])}, M {fmt(ledger['M']['rel'])} "
        f"(tol {fmt(ROUNDOFF_REL)})\n        "
        f"outstanding Ei debt max {fmt(float(np.max(np.abs(sim._dvm.Ei_debt))))} "
        f"erg/cm3 -- withheld, not discarded"
    )
    if not survived:
        detail += f"\n        CRASH {crash}"
    return (
        "D3 floor-aware relax: the withheld transfer is re-ledgered, not lost",
        ok,
        detail,
    )


def gate_d4():
    """The recycle channel enters as a directed inflow at its own face."""
    sim = make_sim(
        neutral_kinetic_dvm_nvz=16,
        neutral_kinetic_dvm_nvp=6,
        **PRODUCTION_GEOMETRY_KEYS,
    )
    geom = sim.geometry
    dvm = TransientDVM(geometry=geom, nvz=16, nvp=6,
                       exchange_model=EXCHANGE_MODEL)
    cath = dvm.cath_cell
    fed = 1.0e18
    # The DECIDED production neutral-clock tick. The transport statement is
    # about what one tick does, so it must be measured at one.
    dt = 1.0e-5
    dvm.f_c[:] = 0.0
    dvm.f_a[:] = 0.0
    src = np.zeros(dvm.nz)
    src[cath] = fed
    led = dvm.update(
        dt,
        n_i=np.zeros(dvm.nz),
        Ti_eV=np.full(dvm.nz, 0.026),
        u_i=np.zeros(dvm.nz),
        nu_ion=np.zeros(dvm.nz),
        sources={"cathode_face": src},
        T_s_K=1910.0,
    )
    mass = dvm.f_c.sum(axis=(1, 2)) * dvm.V_col
    mass_a = dvm.f_a.sum(axis=(1, 2)) * dvm.V_ann
    injected = dvm.total_inventory()
    expected = fed * dt
    # 1. Every fed particle arrived. The ghost density is the counted
    #    particles divided by exactly the ``|v_z| A dt`` the march multiplies
    #    them by, so the inflow FLUX INTEGRAL across the face equals the
    #    removed flux the plasma reported -- S1's identity, carried through
    #    the face into the distribution. Nothing is pumped here (s_L = s_R =
    #    0) and the end returns are buffered, so the domain inventory IS the
    #    injected count.
    flux_rel = abs(injected - expected) / expected
    # 2. It came off the SURFACE, travelling: nothing upstream of the
    #    emitting face, and within this one tick part of the return has
    #    already moved downstream. The superseded form wrote
    #    ``f_c[cell] += counts * spectrum / V`` AFTER the march, which by
    #    construction leaves the whole return standing in the emitting cell
    #    for the entire tick -- the re-ignition mechanism.
    upstream = float(mass[:cath].sum()) + float(mass_a[:cath].sum())
    retained = float(mass[cath]) / injected
    downstream = float(mass[cath + 1:].sum())
    drift = float(dvm.column_drift()[cath])
    ok = (
        flux_rel < ROUNDOFF_REL
        and upstream == 0.0
        and 0.0 < retained < 1.0
        and downstream > 0.0
        and drift > 0.0
        and abs(ledger_residual(led)["distribution_rel"]) < ROUNDOFF_REL
    )
    return (
        "D4 wall recycle enters as a directed face inflow: flux integral "
        "== fed, and it travels",
        ok,
        f"fed {fmt(expected)} particles at cell {cath} over one {dt:g} s "
        f"tick; domain inventory {fmt(injected)}, relative error "
        f"{fmt(flux_rel)} (tol {fmt(ROUNDOFF_REL)})\n        "
        f"upstream of the emitting face: {fmt(upstream)} (exactly zero "
        f"required); retained in the emitting cell {retained:.4f}, moved "
        f"downstream {fmt(downstream)} particles\n        "
        f"(the superseded in-cell birth retains 1.0000 by construction)"
        f"\n        column drift at the emitting cell {fmt(drift)} cm/s "
        f"(> 0 required); ledger residual "
        f"{fmt(abs(ledger_residual(led)['distribution_rel']))}",
    )


#: [D5] The synthetic cell's per-tick budget, as fractions of the pre-tick
#: inventory I0. ``MARCHED`` is the ionization the march itself took (a
#: NON-conserving loss); ``CONSERVING`` is the CX/elastic pair, which
#: leaves the cell and returns to it within the same tick at the same
#: count. Nothing else leaves the synthetic cell, so the largest booking
#: it can pay is I0 itself, and the pre-fix ordering's ceiling was
#: ``1 - CONSERVING``.
D5_MARCHED = 0.10
D5_CONSERVING = 0.35
D5_BOOKED_FRACTIONS = (
    0.0, 0.10, 0.30, 0.50, 0.65, 0.66, 0.70, 0.80, 0.90, 0.99, 1.0,
)


def d5_synthetic_cell(dvm, booked_fraction, I0=1.0e18):
    """Build one frozen synthetic cell-tick and debit it both ways.

    The cell holds ``I0`` atoms before the tick. The march takes
    ``D5_MARCHED * I0`` of them as ionization and ``D5_CONSERVING * I0``
    as the CX/elastic pair; the pair is re-born in the same cell, at the
    same count, in the ion Maxwellian. Nothing else enters or leaves, so
    after the tick the cell holds ``I0`` less whatever the ionization
    channel finally takes.

    Returns ``(post, pre, population, counts)`` -- the debit taken against
    the post-rebirth population (the shipped ordering), the same debit
    taken against the marched state alone (the pre-fix ordering, kept as
    the negative control), the post-rebirth population itself so the
    caller can check positivity on the array the solver would carry
    forward, and the per-cell ``I0``. Both debits go through the SHIPPED
    :meth:`TransientDVM._debit_booked_ionization`; the ONLY difference
    between them is which population it is handed, which is the whole of
    the defect and the whole of the fix.

    The ``dvm`` must be freshly constructed and never stepped, so its
    ``ion_debt`` is zero and the debit's target is the booking itself.
    """
    g = dvm.g
    vol_c = dvm.V_col[:, None, None]
    # A drifting Maxwellian for the marched remnant and a hotter, shifted
    # one for the re-births, so the two populations are distinguishable in
    # velocity and a debit that draws on the wrong one is visible.
    shape_march = g.maxwellian(0.30, 0.0)[None, :, :] * np.ones(dvm.nz)[
        :, None, None
    ]
    shape_birth = g.maxwellian(1.20, 2.0e5)[None, :, :] * np.ones(dvm.nz)[
        :, None, None
    ]

    def at_count(shape, count):
        mass = (shape * vol_c).sum(axis=(1, 2))
        return shape * (count / mass)[:, None, None]

    counts = np.full(dvm.nz, I0)
    marched_n = D5_MARCHED * counts
    conserving_n = D5_CONSERVING * counts
    f_march = at_count(shape_march, counts - marched_n - conserving_n)
    births = at_count(shape_birth, conserving_n)
    # The march's own frequency tally, per bin, at exactly ``marched_n``.
    L_ion = at_count(shape_march, marched_n) * vol_c
    booked = booked_fraction * counts
    return (
        dvm._debit_booked_ionization(booked, L_ion, f_march + births, vol_c),
        dvm._debit_booked_ionization(booked, L_ion, f_march, vol_c),
        f_march + births,
        counts,
    )


def gate_d5():
    """The counted ionization debit is capped at the POST-REBIRTH inventory.

    The march removes ionization, charge exchange and elastic scattering
    from the pre-tick population together, but only ionization is a real
    loss: the CX/elastic pair is re-born in the same cell, at the same
    count, within the same tick. The count the plasma booked must therefore
    be measured against the marched state PLUS those re-births -- the atoms
    the cell actually holds when the tick ends. Capped at the marched state
    alone, a cell is told it cannot pay a booking smaller than its own
    inventory, the positivity limiter fires against atoms that never left,
    and the resulting debt can never retire.

    Two statements:

    (a) SYNTHETIC, frozen (:func:`d5_synthetic_cell`). A cell holding
        ``I0`` loses ``D5_MARCHED`` to the march's ionization and
        ``D5_CONSERVING`` to the pair, the pair returning in full. Against
        the post-rebirth population the debit succeeds with zero shortfall
        and an exactly non-negative distribution for EVERY booking up to
        ``I0``, which is the whole inventory and therefore the most any
        booking could ask for. Against the marched state -- the pre-fix
        ordering, run here as a negative control on the same numbers --
        the shortfall appears at bookings above ``1 - D5_CONSERVING``,
        sharply at that closed-form threshold and not before, which is the
        defect reproduced on demand.

    (b) LIVE. Over a window of the in-solver default arm, per tick: the
        particle and energy ledgers close at ROUNDOFF_REL, the handshake
        identity ``ion_removed_cum + ion_debt == ion_booked_cum`` holds per
        cell at the same tolerance, the column distribution never goes
        negative, and the arm is actually booking ionization -- without
        which every identity above is 0 == 0.
    """
    dvm = closed_box_dvm(nz=4)
    threshold = 1.0 - D5_CONSERVING
    worst_short = 0.0
    worst_removed = 0.0
    worst_negative = 0.0
    post_clean = True
    control_sharp = True
    control_fired = 0
    for frac in D5_BOOKED_FRACTIONS:
        post, pre, population, counts = d5_synthetic_cell(dvm, frac)
        scale = counts
        worst_short = max(
            worst_short, float(np.max(np.abs(post["shortfall"]) / scale))
        )
        worst_removed = max(
            worst_removed,
            float(np.max(np.abs(post["removed"] - post["booked"]) / scale)),
        )
        # Positivity, as the array the solver would carry forward.
        remaining = population - post["drop"]
        worst_negative = max(worst_negative, float(-np.min(remaining)))
        post_clean = post_clean and not np.any(post["limited"])
        # The negative control must fire above the closed-form threshold
        # and stay quiet below it -- a control that failed everywhere would
        # not have localized anything.
        fired = bool(np.any(pre["limited"]))
        control_fired += int(fired)
        control_sharp = control_sharp and (fired == (frac > threshold))

    # (b) the live arm
    sim = make_sim()
    ledgers = run_until_updates(sim, 8)
    dvm_live = sim._dvm
    worst_part = max(
        max(abs(ledger_residual(led)["distribution_rel"]),
            abs(ledger_residual(led)["domain_rel"]))
        for led in ledgers
    )
    worst_energy = max(
        max(abs(ledger_energy_residual(led)["distribution_rel"]),
            abs(ledger_energy_residual(led)["domain_rel"]))
        for led in ledgers
    )
    booked_cum = np.asarray(dvm_live.ion_booked_cum, dtype=float)
    identity = np.asarray(dvm_live.ion_removed_cum, dtype=float) + np.asarray(
        dvm_live.ion_debt, dtype=float
    ) - booked_cum
    ion_scale = max(float(np.max(np.abs(booked_cum))), 1e-300)
    worst_identity = float(np.max(np.abs(identity))) / ion_scale
    live_negative = float(-np.min(dvm_live.f_c))
    booking = float(np.sum(booked_cum))

    ok = (
        worst_short <= ROUNDOFF_REL
        and worst_removed <= ROUNDOFF_REL
        and worst_negative <= 0.0
        and post_clean
        and control_sharp
        and control_fired > 0
        and worst_part < ROUNDOFF_REL
        and worst_energy < ROUNDOFF_REL
        and worst_identity < ROUNDOFF_REL
        and live_negative <= 0.0
        and booking > 0.0
    )
    return (
        "D5 counted ionization debits against the post-rebirth inventory",
        ok,
        f"synthetic cell, {len(D5_BOOKED_FRACTIONS)} bookings up to the "
        f"whole inventory (march takes {D5_MARCHED:.2f} I0, the CX/elastic "
        f"pair {D5_CONSERVING:.2f} I0 and returns it): worst |shortfall|/I0 "
        f"{fmt(worst_short)}, worst |removed - booked|/I0 "
        f"{fmt(worst_removed)}, most negative bin {fmt(worst_negative)}, "
        f"limiter never fired ({post_clean})\n        "
        f"negative control on the pre-fix ordering (marched state alone): "
        f"fired at {control_fired} of {len(D5_BOOKED_FRACTIONS)} bookings, "
        f"exactly those above the closed-form threshold "
        f"1 - {D5_CONSERVING:.2f} = {threshold:.2f} ({control_sharp})\n        "
        f"live arm, {len(ledgers)} ticks: particle ledger {fmt(worst_part)}, "
        f"energy ledger {fmt(worst_energy)}, handshake "
        f"|removed_cum + debt - booked_cum|/booked {fmt(worst_identity)}, "
        f"most negative f_c {fmt(live_negative)}, booked "
        f"{fmt(booking)} particles (tol {fmt(ROUNDOFF_REL)})",
    )


# ------------------------------------------------- construction refusals


REFUSALS = (
    (
        "G1 two-zone required",
        dict(),
        "neutral_two_zone",
        lambda d, fl: (fl.__setitem__("neutral_two_zone", False), None)[1],
    ),
    # The single-key refusal this gate used to arm -- ``neutral_momentum``
    # set back to True on the cleared base -- is unreachable BY CONSTRUCTION
    # since the model-preset resolver landed (2026-08-23h/aj/ak): the flag is
    # a MEMBER of the ``neutral_model='kinetic_dvm'`` family and True is its
    # config default, so a caller cannot distinguish "I chose True" from "I
    # left it alone" and the resolver clears it rather than refusing. X1
    # below pins that resolution. What is still reachable, and what this gate
    # now arms, is an EXPLICIT family conflict: a member whose default is
    # already compatible, set to a value the selection refuses. The one
    # collected error names the complete member set, so ``neutral_momentum``
    # is still in it and the pinned string is still under test.
    (
        "G2 explicit kinetic_dvm member refused, whole member set named",
        dict(),
        "neutral_momentum",
        lambda d, fl: (
            fl.__setitem__("neutral_hot_birth_drift", True),
            None,
        )[1],
    ),
    (
        "G3 non-helium refused",
        dict(),
        "gas_type",
        lambda d, fl: (d.__setitem__("gas_type", "H"), None)[1],
    ),
    (
        "G4 non-positive cadence refused",
        dict(),
        "neutral_kinetic_dvm_cadence_s",
        lambda d, fl: (
            d.__setitem__("neutral_kinetic_dvm_cadence_s", 0.0),
            None,
        )[1],
    ),
    (
        "G5 accommodation outside [0, 1] refused",
        dict(),
        "neutral_kinetic_dvm_accommodation",
        lambda d, fl: (
            d.__setitem__("neutral_kinetic_dvm_accommodation", 1.5),
            None,
        )[1],
    ),
    (
        "G6 unknown elastic model refused",
        dict(),
        "neutral_kinetic_dvm_elastic",
        lambda d, fl: (
            d.__setitem__("neutral_kinetic_dvm_elastic", "bilinear"),
            None,
        )[1],
    ),
    (
        "G7 Picard coupling refused",
        dict(),
        "coupled_circuit_picard",
        lambda d, fl: (fl.__setitem__("coupled_circuit_picard", True), None)[1],
    ),
    (
        "G8 puff local ionization refused",
        dict(),
        "gas_puff_local_ionization_fraction",
        lambda d, fl: (
            d.__setitem__("gas_puff_local_ionization_fraction", 0.2),
            None,
        )[1],
    ),
    (
        "G10 odd v_z bin count refused",
        dict(),
        "nvz",
        lambda d, fl: (d.__setitem__("neutral_kinetic_dvm_nvz", 47), None)[1],
    ),
    (
        "G11 unknown zone-exchange closure refused",
        dict(),
        "neutral_kinetic_dvm_exchange",
        lambda d, fl: (
            d.__setitem__("neutral_kinetic_dvm_exchange", "cauchy"),
            None,
        )[1],
    ),
    (
        "G12 unknown annulus-flight treatment refused",
        dict(),
        "neutral_kinetic_dvm_annulus_flights",
        lambda d, fl: (
            d.__setitem__("neutral_kinetic_dvm_annulus_flights", "chord"),
            None,
        )[1],
    ),
    (
        "G13 bounded-chord annulus without the DVM arm refused",
        dict(),
        "neutral_kinetic_dvm_annulus_flights",
        lambda d, fl: (
            d.__setitem__(
                "neutral_kinetic_dvm_annulus_flights", "bounded_chord"
            ),
            d.__setitem__("neutral_model", "moment"),
            # Passes today only because the annulus guard is checked before
            # the wall-reflection one; dropped here so the gate tests its own
            # subject rather than a guard ORDERING.
            off_arm(d),
        )[2],
    ),
    (
        "G14 bounded-chord annulus without the two-zone flag refused",
        dict(),
        "neutral_two_zone",
        lambda d, fl: (
            d.__setitem__(
                "neutral_kinetic_dvm_annulus_flights", "bounded_chord"
            ),
            fl.__setitem__("neutral_two_zone", False),
            None,
        )[2],
    ),
    (
        "G16 unknown wall-reflection spectrum refused",
        dict(),
        "neutral_kinetic_dvm_wall_reflection",
        lambda d, fl: (
            d.__setitem__(
                "neutral_kinetic_dvm_wall_reflection", "diffuse"
            ),
            None,
        )[1],
    ),
    # G17/G18 MIRRORED 2026-08-30, at the flip of the package default from
    # 'specular' to 'diffuse_elastic'. Both gates assert that the selector's
    # NON-DEFAULT arm is refused where it cannot act -- off the DVM arm (G17)
    # and without the two-zone flag (G18). Which literal instantiates "the
    # non-default arm" is decided by the default, so a default flip MUST
    # carry these two literals with it: the statements are unchanged in
    # intent and in cardinality, and mirroring them is the necessary
    # companion of the flip, not a weakening. Left unmirrored, G17 asserts a
    # refusal that the flip makes impossible (measured: no refusal raised).
    (
        "G17 specular wall reflection without the DVM arm refused",
        dict(),
        "neutral_kinetic_dvm_wall_reflection",
        lambda d, fl: (
            d.__setitem__(
                "neutral_kinetic_dvm_wall_reflection", "specular"
            ),
            d.__setitem__("neutral_model", "moment"),
            None,
        )[2],
    ),
    (
        "G18 specular wall reflection without the two-zone flag "
        "refused",
        dict(),
        "neutral_two_zone",
        lambda d, fl: (
            d.__setitem__(
                "neutral_kinetic_dvm_wall_reflection", "specular"
            ),
            fl.__setitem__("neutral_two_zone", False),
            None,
        )[2],
    ),
    (
        "G19 DVM cathode jet without the DVM arm refused",
        dict(),
        "neutral_kinetic_dvm_cathode_jet",
        lambda d, fl: (
            d.__setitem__("neutral_kinetic_dvm_cathode_jet", True),
            d.__setitem__("neutral_model", "moment"),
            off_arm(d),
        )[2],
    ),
    (
        "G20 DVM cathode jet coefficients without the arm refused",
        dict(),
        "neutral_kinetic_dvm_cathode_jet_R_N",
        lambda d, fl: (
            d.__setitem__("neutral_kinetic_dvm_cathode_jet_R_N", 0.5),
            d.__setitem__("neutral_model", "moment"),
            off_arm(d),
        )[2],
    ),
    (
        "G21 DVM cathode jet with R_E > R_N refused",
        dict(),
        "R_E",
        lambda d, fl: (
            d.__setitem__("neutral_kinetic_dvm_cathode_jet", True),
            d.__setitem__("neutral_kinetic_dvm_cathode_jet_R_E", 0.5),
            None,
        )[2],
    ),
    (
        "G22 DVM cathode jet with a non-positive launch smear refused",
        dict(),
        "T_launch_eV",
        lambda d, fl: (
            d.__setitem__("neutral_kinetic_dvm_cathode_jet", True),
            d.__setitem__(
                "neutral_kinetic_dvm_cathode_jet_T_launch_eV", 0.0
            ),
            None,
        )[2],
    ),
    (
        "G23 DVM cathode jet without cathode coupling refused",
        dict(),
        "cathode_coupling",
        lambda d, fl: (
            d.__setitem__("neutral_kinetic_dvm_cathode_jet", True),
            fl.__setitem__("cathode_coupling", False),
            None,
        )[2],
    ),
    # G24-G27 and G29 are the B4 anode-side twins of G19-G23, in the same
    # order and built the same way. The PAIR statement that would be "G28" --
    # this channel against the fluid ``anode_neutral_jet`` -- is not here for
    # the reason CJ4 is not here: it is unreachable through LAPDSim1D, so a
    # gate driven that way would be testing the model-family resolver rather
    # than the guard. It is ``gate_aj5``, which calls the guard directly.
    (
        "G24 DVM anode jet without the DVM arm refused",
        dict(),
        "neutral_kinetic_dvm_anode_jet",
        lambda d, fl: (
            d.__setitem__("neutral_kinetic_dvm_anode_jet", True),
            d.__setitem__("neutral_model", "moment"),
            off_arm(d),
        )[2],
    ),
    (
        "G25 DVM anode jet coefficients without the arm refused",
        dict(),
        "neutral_kinetic_dvm_anode_jet_R_N",
        lambda d, fl: (
            d.__setitem__("neutral_kinetic_dvm_anode_jet_R_N", 0.5),
            d.__setitem__("neutral_model", "moment"),
            off_arm(d),
        )[2],
    ),
    (
        "G26 DVM anode jet with R_E > R_N refused",
        dict(),
        "R_E",
        lambda d, fl: (
            d.__setitem__("neutral_kinetic_dvm_anode_jet", True),
            d.__setitem__("neutral_kinetic_dvm_anode_jet_R_E", 0.8),
            None,
        )[2],
    ),
    (
        "G27 DVM anode jet with a non-positive launch smear refused",
        dict(),
        "T_launch_eV",
        lambda d, fl: (
            d.__setitem__("neutral_kinetic_dvm_anode_jet", True),
            d.__setitem__(
                "neutral_kinetic_dvm_anode_jet_T_launch_eV", 0.0
            ),
            None,
        )[2],
    ),
    (
        "G29 DVM anode jet without cathode coupling refused",
        dict(),
        "cathode_coupling",
        lambda d, fl: (
            d.__setitem__("neutral_kinetic_dvm_anode_jet", True),
            fl.__setitem__("cathode_coupling", False),
            None,
        )[2],
    ),
    # --- B6 baffle interception. G32 (a clear radius below the local column
    # radius) is NOT here: its solver-route refusal belongs to core.geometry
    # and fires before the kinetic arm is built, so it is made as its own gate
    # that states BOTH owners -- see gate_bf_g32.
    (
        "G30 DVM baffle interception without the DVM arm refused",
        dict(),
        "neutral_kinetic_dvm_baffles",
        lambda d, fl: (
            fl.__setitem__("neutral_kinetic_dvm_baffles", True),
            d.__setitem__("neutral_model", "moment"),
            off_arm(d),
        )[2],
    ),
    (
        "G31 DVM baffle interception without the fluid baffles refused",
        dict(),
        "neutral_baffles",
        lambda d, fl: (
            fl.__setitem__("neutral_kinetic_dvm_baffles", True),
            None,
        )[1],
    ),
    (
        "G15 gas puff into a cell with no annulus refused",
        dict(),
        "V_ann",
        lambda d, fl: (d.__setitem__("Rm", d["Rp"]), None)[1],
    ),
)


def off_arm(d):
    """Drop the suite's declared wall-reflection arm; return ``None``.

    ``arm_config`` DECLARES ``wall_reflection = "specular"`` so the suite's
    pinned numbers name their arm instead of inheriting a default. Off the DVM
    arm that declaration is meaningless -- there is no wall share to place --
    and since the package default moved to ``"diffuse_elastic"`` the solver's
    off-arm inertness guard REFUSES a declared ``"specular"`` there, by
    design. A gate that leaves the arm to test something else would therefore
    trip that guard before reaching its own subject: measured, G19 and G20
    raised the wall-reflection message instead of their own.

    So an off-arm gate drops the key and lets it resolve to whatever the
    package ships, which is the inert value by construction. Written as a
    helper rather than a literal so no gate hard-codes which value is
    currently the default -- the coupling that broke the guard in the first
    place.
    """
    d.pop("neutral_kinetic_dvm_wall_reflection", None)
    return None


def make_refusal_gate(label, offender, mutate):
    def gate():
        d, fl = arm_config()
        mutate(d, fl)
        try:
            LAPDSim1D(input_dict=d, input_flags=fl)
        except ValueError as exc:
            names = offender in str(exc)
            return label, names, f"raised naming {offender!r}: {str(exc)[:88]}"
        return label, False, "no ValueError raised"

    return gate


def gate_x1():
    """Family members left at their config default are RESOLVED, not refused.

    Every arm in this file used to be unconstructible from
    ``default_config()``: the package ships each member of the
    ``neutral_model='kinetic_dvm'`` family armed, and each one had to be
    cleared by hand before the next refusal appeared. Since the model-preset
    resolver landed the selection alone is enough -- so this gate names the
    selection, hands over an OTHERWISE UNTOUCHED default config, and checks
    that every member came out at the value the family requires.
    """
    d, fl = default_config()
    d["neutral_model"] = "kinetic_dvm"
    d["neutral_kinetic_dvm_cadence_s"] = CADENCE_S
    d["neutral_kinetic_dvm_exchange"] = EXCHANGE_MODEL
    label = (
        "X1 kinetic_dvm resolves its member set from an untouched "
        "default_config()"
    )
    try:
        sim = LAPDSim1D(input_dict=d, input_flags=fl)
    except ValueError as exc:
        return label, False, f"construction raised: {str(exc)[:120]}"
    got_d, got_fl = sim.get_config()
    wrong = [
        f"{space}:{key}={(got_fl if space == 'flags' else got_d).get(key)!r}"
        f" (required {required!r})"
        for space, key, required, _why in KINETIC_DVM_INCOMPATIBLE_DEFAULTS
        if (got_fl if space == "flags" else got_d).get(key) != required
    ]
    n = len(KINETIC_DVM_INCOMPATIBLE_DEFAULTS)
    return (
        label,
        not wrong,
        f"{n} members resolved, none hand-cleared; mismatches: "
        f"{', '.join(wrong) if wrong else 'none'}",
    )


# ------------------------------------------------------- limit cases


def gate_l1():
    """Uniform profile in one bin, matched inflow: stationary to roundoff."""
    dvm = bare_dvm(nz=16, nvz=16, nvp=6, accommodation=1.0)
    g = dvm.g
    b = int(np.argmax(g.vz > 0))  # slowest forward bin
    nz = dvm.nz
    dens = 1.0e12
    dvm.f_c[:, b, 0] = dens
    dt = 1.0e-6
    # Matched inflow: the buffered particles that would arrive at the left
    # face in one step at this density.
    dvm.pend_L_c[b, 0] = dens * abs(g.vz[b]) * dvm.face_c[0] * dt
    bg = zero_plasma(dvm)
    dvm.nux[...] = 0.0
    dvm.nuxp[...] = 0.0
    dvm.nuw[...] = 0.0
    before = dvm.f_c.copy()
    dvm.update(dt, **bg)
    err = np.max(np.abs(dvm.f_c - before)) / dens
    ok = err < 1.0e-14
    return (
        "L1 free streaming: uniform profile with matched inflow is stationary",
        ok,
        f"max relative drift over one update = {fmt(err)} (tol 1e-14)",
    )


def gate_l2():
    """Centre of mass advances by exactly v_z * t while mass is interior.

    The domain is long enough that the implicit scheme's (globally
    supported) tail is at machine zero when it reaches the far end, so
    "while mass is interior" is satisfied to roundoff rather than
    approximately: the identity ``d<z>/dt = v`` then holds exactly, with
    no reliance on the scheme being non-diffusive.
    """
    dvm = bare_dvm(nz=40, nvz=16, nvp=6)
    g = dvm.g
    b = int(np.argmax(g.vz > 0))
    v = float(g.vz[b])
    dvm.f_c[:, :, :] = 0.0
    dvm.f_c[2, b, 0] = 1.0e12
    dvm.nux[...] = 0.0
    dvm.nuxp[...] = 0.0
    dvm.nuw[...] = 0.0
    dvm.s_L = 0.0
    dvm.s_R = 0.0
    zc = np.cumsum(dvm.dz) - 0.5 * dvm.dz
    bg = zero_plasma(dvm)

    def com():
        w = dvm.f_c.sum(axis=(1, 2)) * dvm.V_col
        return float((zc * w).sum() / w.sum())

    # dt small enough that the packet stays clear of both ends.
    dt = 0.25 * float(dvm.dz[0]) / v
    n = 8
    c0 = com()
    for _ in range(n):
        dvm.update(dt, **bg)
    c1 = com()
    expected = v * dt * n
    err = abs((c1 - c0) - expected) / expected
    # The identity is exact only while the mass is interior, so the gate
    # also confirms that what reached the ends is at machine zero relative
    # to the packet -- otherwise a passing displacement could be an
    # accident of two boundary terms cancelling.
    escaped = dvm.pending_inventory()
    total = dvm.f_inventory() + escaped
    escaped_frac = escaped / total
    ok = err < 1.0e-12 and escaped_frac < 1.0e-15
    return (
        "L2 free streaming: <z> advances by exactly v_z t",
        ok,
        f"measured {fmt(c1 - c0)} cm vs analytic {fmt(expected)} cm, "
        f"relative error {fmt(err)}; escaped fraction {fmt(escaped_frac)} "
        f"(tol 1e-15)",
    )


def _steady_attenuation(nz, optical_depth):
    """Return (discrete steady profile, closed form, analytic continuum).

    ``optical_depth`` is ``nu L / v`` across the whole tube, so the
    absorbing rate is set from the tested velocity bin and the comparison
    against ``exp(-nu z / v)`` stays on an O(1) scale at every resolution.
    """
    dvm = bare_dvm(nz=nz, nvz=16, nvp=6)
    g = dvm.g
    b = int(np.argmax(g.vz > 0))
    v = float(g.vz[b])
    length = float(dvm.dz.sum())
    nu = float(optical_depth) * v / length
    dvm.nux[...] = 0.0
    dvm.nuxp[...] = 0.0
    dvm.nuw[...] = 0.0
    dvm.s_L = 1.0
    dvm.s_R = 1.0
    dt = 50.0 * float(dvm.dz[0]) / v  # deep into the steady limit
    nz_ = dvm.nz
    bg = {
        "n_i": np.zeros(nz_),
        "Ti_eV": np.full(nz_, 0.026),
        "u_i": np.zeros(nz_),
        "nu_ion": np.full(nz_, nu),
    }
    dens = 1.0e12
    inflow = dens * v * dvm.face_c[0] * dt
    for _ in range(400):
        dvm.pend_L_c[b, 0] = inflow
        dvm.update(dt, **bg)
    f = dvm.f_c[:, b, 0]
    # Closed form of the discrete backward-Euler upwind steady state on a
    # uniform grid: each cell attenuates by lam / (lam + nu).
    lam = v / float(dvm.dz[0])
    ratio = lam / (lam + nu)
    closed = f[0] * ratio ** np.arange(nz_)
    zc = np.cumsum(dvm.dz) - 0.5 * dvm.dz
    analytic = f[0] * np.exp(-nu * (zc - zc[0]) / v)
    return f, closed, analytic


def gate_l3():
    optical_depth = 2.0
    errs = []
    for nz in (24, 48, 96):
        f, closed, analytic = _steady_attenuation(nz, optical_depth)
        closed_err = np.max(np.abs(f - closed)) / f[0]
        # Compare on the shared physical coordinate: the half-domain point,
        # which sits exactly L/2 downstream of the first cell centre at
        # every resolution.
        mid = f.size // 2
        errs.append((nz, closed_err, abs(f[mid] - analytic[mid]) / analytic[mid]))
    orders = [
        np.log(errs[i][2] / errs[i + 1][2]) / np.log(2.0)
        for i in range(len(errs) - 1)
    ]
    closed_ok = all(e[1] < 1.0e-12 for e in errs)
    order_ok = all(0.8 <= o <= 1.3 for o in orders)
    ok = closed_ok and order_ok
    detail = "; ".join(
        f"nz={n}: closed {fmt(c)}, analytic {fmt(a)}" for n, c, a in errs
    )
    return (
        "L3 free streaming: discrete closed form exact, analytic 1st order",
        ok,
        f"{detail}; measured orders "
        + ", ".join(f"{o:.3f}" for o in orders),
    )


def gate_l4():
    """A wall-temperature Maxwellian in a closed box stays put, and stays.

    NB the tolerances here were REVISED after the first run of this suite;
    the original 2e-2 density bound was a blind guess that the measurement
    showed to be mis-specified for the shipped velocity grid. What the
    measurement establishes:

    - the closed box conserves particles EXACTLY (roundoff), so nothing is
      created or destroyed while the state relaxes;
    - the seeded continuum Maxwellian is not exactly the DISCRETE
      operator's fixed point, and the state relaxes toward that fixed
      point by ~2.5% in the column/annulus density split. The offset is a
      velocity-resolution property, not a coupling defect: the shared
      production grid is sized for the 10 eV charge-exchange tail, so a
      300 K gas occupies only the few bins inside ``v_fine`` and the
      ``v_perp``-weighted zone-exchange and wall rates see it coarsely.
      Refining the grid reduces the temperature offset steadily
      (2.4e-2 -> 8.4e-3 from 32x10 to 96x32) while the density split
      converges to ~2.1e-2, which is the coarse-near-zero resolution.
    - the drift SATURATES rather than growing: that is the stability
      statement this gate actually protects, and it is checked directly by
      comparing an early and a late window.

    The E3 coupling gates above are untouched by this revision.
    """
    dvm = bare_dvm(nz=16, nvz=32, nvp=10, accommodation=1.0)
    dvm.s_L = 0.0
    dvm.s_R = 0.0
    dens = 1.0e13
    dvm.seed_from_density(np.full(dvm.nz, dens), np.full(dvm.nz, dens))
    bg = zero_plasma(dvm)
    dt = 2.5e-5
    n0 = dvm.column_density().copy()
    a0 = dvm.annulus_density().copy()
    T0 = dvm.column_temperature_eV().copy()
    start = float((n0 * dvm.V_col + a0 * dvm.V_ann).sum())
    marks = {}
    for k in range(1, 401):
        dvm.update(dt, **bg)
        if k in (40, 200, 400):
            marks[k] = (
                np.max(np.abs(dvm.column_density() - n0)) / dens,
                np.max(np.abs(dvm.annulus_density() - a0)) / dens,
                np.max(np.abs(dvm.column_temperature_eV() - T0))
                / float(np.max(T0)),
                dvm.column_density().copy(),
            )
    inv_rel = abs(dvm.total_inventory() - start) / start
    early = float(np.max(np.abs(marks[200][3] - marks[40][3]))) / dens
    late = float(np.max(np.abs(marks[400][3] - marks[200][3]))) / dens
    bounded = marks[400][0] < 5.0e-2 and marks[400][2] < 1.0e-1
    saturating = late < early
    ok = bounded and saturating and inv_rel < 1.0e-12
    return (
        "L4 equilibrium: closed box conserves exactly, drift bounded and "
        "saturating",
        ok,
        f"offset from the seeded Maxwellian at 400 updates: dn/n "
        f"{fmt(marks[400][0])}, dn_a/n {fmt(marks[400][1])}, dTn/Tn "
        f"{fmt(marks[400][2])} (bounds 5e-2 / 1e-1); windowed drift "
        f"40->200 {fmt(early)} then 200->400 {fmt(late)} "
        f"(saturating={saturating}); inventory drift {fmt(inv_rel)}",
    )


def gate_l5():
    """Wall flux balance: incident == accommodated + reflected exactly."""
    rows = []
    ok = True
    for alpha in (0.0, 0.35, 1.0):
        dvm = bare_dvm(nz=10, nvz=16, nvp=6, accommodation=alpha)
        dvm.s_L = 0.0
        dvm.s_R = 0.0
        dvm.seed_from_density(
            np.full(dvm.nz, 1.0e13), np.full(dvm.nz, 1.0e13)
        )
        led = dvm.update(1.0e-5, **zero_plasma(dvm))
        incident = led["loss_wall"]
        returned = (
            led["birth_wall_accommodated"] + led["birth_wall_reflected"]
        )
        wall_err = abs(incident - returned) / max(incident, 1e-300)
        ends_in = led["loss_end_out_L"] + led["loss_end_out_R"]
        # With zero sticking the whole end outflow is buffered for return.
        buffered = dvm.pending_inventory()
        end_err = abs(ends_in - buffered) / max(ends_in, 1e-300)
        rows.append((alpha, wall_err, end_err))
        ok = ok and wall_err < ROUNDOFF_REL and end_err < ROUNDOFF_REL
    detail = "; ".join(
        f"alpha={a}: wall {fmt(w)}, ends {fmt(e)}" for a, w, e in rows
    )
    return (
        "L5 wall flux balance: incident == accommodated + reflected",
        ok,
        detail,
    )


# ------------------------------------------ B1 counted boundary channels


#: [B1] The counted boundary-inflow fixture, in PARTICLES per tick.
#: Deliberately unequal across the four channels so a channel that received
#: another's count -- or that landed at another's cell -- could not pass.
B1_FED_COUNTS = {
    "cathode_face": 5.0e16,
    "collector_face": 4.0e16,
    "recombination": 3.0e16,
    "anode": 2.0e16,
}

#: [B1] The cathode surface temperature the counted arm emits at. Different
#: from the 300 K wall so the two recycle faces carry different spectra --
#: which is what makes the B3 wrong-energy negative control bite.
B1_T_S_K = 1910.0


def b1_no_loss_box(nz=12):
    """Return an empty DVM with no pumping and no external loss channel.

    Started at zero (``f_c = f_a = 0``, no pending buffers) so the domain
    inventory after one update IS what the external channels injected, and
    with ``s_L = s_R = 0`` and no plasma so nothing leaves: the wall, mesh
    and zone channels are internal and conserve particles exactly, and the
    end returns are buffered inside the inventory. Anything the box holds
    afterwards therefore came in through the source channels under test.
    """
    dvm = bare_dvm(nz=nz, accommodation=0.4)
    dvm.s_L = 0.0
    dvm.s_R = 0.0
    dvm.f_c[:] = 0.0
    dvm.f_a[:] = 0.0
    return dvm


def b1_home_cells(dvm):
    """Return each B1 channel's own cell on ``dvm``'s grid.

    The face channels belong to the cells the engine emits them from; the
    two volume births are placed a third and two thirds along, well clear
    of both faces and of each other, so that "landed at its own cell" and
    "landed at another channel's" are different answers.
    """
    return {
        "cathode_face": int(dvm.cath_cell),
        "collector_face": int(dvm.coll_cell),
        "recombination": dvm.nz // 3,
        "anode": 2 * dvm.nz // 3,
    }


def b1_fed_rows(dvm):
    """Return the B1 fixture as per-cell rows on ``dvm``'s grid."""
    homes = b1_home_cells(dvm)
    rows = {name: np.zeros(dvm.nz) for name in B1_FED_COUNTS}
    for name, count in B1_FED_COUNTS.items():
        rows[name][homes[name]] = count
    return rows


def gate_b1():
    """Counted boundary inflow: injected == counted, per face, at any dt.

    STATEMENT 1 of the B1 three (the closed-box synthetic case). The four
    boundary-inflow channels are handed as PARTICLES, so what the box holds
    after one tick is the fed count exactly and does not depend on the tick
    length at all -- the whole point of the counted handshake, and the
    property the rate path cannot have.

    Per FACE, on the PRODUCTION machine (whose cathode face is NOT the
    domain end, so the statement is not vacuous the way it is on the
    uniform tube): fed one channel at a time, the counted particles put
    EXACTLY ZERO on the far side of their own emitting face -- the march
    cannot move anything against the direction it was launched in -- and
    the cathode's return drifts downstream while the collector's drifts
    upstream. The two volume births stay in the cell they were fed (they
    are substep-B births, applied after the march).

    A channel's share at ANOTHER channel's cell is bounded rather than
    exactly zero, because the march is implicit and therefore has no finite
    signal speed: a cell hundreds of cells downstream receives an
    exponentially small tail within one tick. It is measured against the
    fed count and held below the roundoff tolerance, which a genuine
    mis-deposit -- a channel launched at the wrong face -- would exceed by
    twenty orders of magnitude.

    NEGATIVE CONTROL: the same numbers handed through ``sources`` instead,
    which multiplies by ``dt``. The injected inventory is then proportional
    to the tick length and equals the fed count at neither of the two,
    which is exactly the first-order-in-cadence sampling error this member
    removes -- and it fails at THIS statement.
    """
    fed_total = float(sum(B1_FED_COUNTS.values()))
    dts = (1.0e-5, 4.0e-5)
    counted = []
    rows_ok = True
    resid = 0.0
    for dt in dts:
        dvm = b1_no_loss_box()
        rows = b1_fed_rows(dvm)
        led = dvm.update(
            dt, source_counts=rows, T_s_K=B1_T_S_K, **zero_plasma(dvm)
        )
        counted.append(dvm.total_inventory())
        resid = max(resid, abs(ledger_residual(led)["distribution_rel"]))
        for name, fed in B1_FED_COUNTS.items():
            rows_ok = rows_ok and led[f"birth_{name}"] == fed
    counted_rel = max(abs(c - fed_total) / fed_total for c in counted)
    dt_rel = abs(counted[0] - counted[1]) / fed_total

    # Per-face / per-cell placement on the PRODUCTION machine, one channel
    # at a time.
    machine = make_sim(
        neutral_kinetic_dvm_nvz=16,
        neutral_kinetic_dvm_nvp=6,
        **PRODUCTION_GEOMETRY_KEYS,
    ).geometry
    places = []
    place_ok = True
    for name in B1_FED_COUNTS:
        dvm = TransientDVM(
            geometry=machine, nvz=16, nvp=6, exchange_model=EXCHANGE_MODEL,
        )
        dvm.s_L = 0.0
        dvm.s_R = 0.0
        dvm.f_c[:] = 0.0
        dvm.f_a[:] = 0.0
        homes = b1_home_cells(dvm)
        rows = b1_fed_rows(dvm)
        dvm.update(
            1.0e-5,
            source_counts={name: rows[name]},
            T_s_K=B1_T_S_K,
            **zero_plasma(dvm),
        )
        mass = dvm.f_c.sum(axis=(1, 2)) * dvm.V_col
        mass_a = dvm.f_a.sum(axis=(1, 2)) * dvm.V_ann
        home = homes[name]
        foreign = float(sum(
            mass[cell] + mass_a[cell]
            for other, cell in homes.items() if other != name
        ))
        if name == "cathode_face":
            extra = float(mass[:home].sum() + mass_a[:home].sum())
            directed = float(dvm.column_drift()[home]) > 0.0
            where = f"upstream of face {home}"
        elif name == "collector_face":
            extra = float(mass[home + 1:].sum() + mass_a[home + 1:].sum())
            directed = float(dvm.column_drift()[home]) < 0.0
            where = f"downstream of face {home}"
        else:
            extra = float(
                mass.sum() - mass[home] + mass_a.sum() - mass_a[home]
            )
            directed = True
            where = f"outside birth cell {home}"
        got = dvm.total_inventory()
        rel = abs(got - B1_FED_COUNTS[name]) / B1_FED_COUNTS[name]
        at_home = float(mass[home] + mass_a[home]) > 0.0
        foreign_rel = foreign / B1_FED_COUNTS[name]
        places.append((name, home, where, extra, foreign_rel, rel))
        place_ok = (
            place_ok
            and extra == 0.0
            and foreign_rel < ROUNDOFF_REL
            and at_home
            and directed
            and rel < ROUNDOFF_REL
        )

    # NEGATIVE CONTROL: the same numbers as RATES.
    control = []
    for dt in dts:
        dvm = b1_no_loss_box()
        rows = b1_fed_rows(dvm)
        dvm.update(dt, sources=rows, T_s_K=B1_T_S_K, **zero_plasma(dvm))
        control.append(dvm.total_inventory())
    control_ratio = control[1] / control[0]
    control_fails = (
        abs(control_ratio - dts[1] / dts[0]) < 1.0e-9
        and abs(control[0] - fed_total) / fed_total > 0.1
        and abs(control[1] - fed_total) / fed_total > 0.1
    )

    ok = (
        counted_rel < ROUNDOFF_REL
        and dt_rel < ROUNDOFF_REL
        and rows_ok
        and resid < ROUNDOFF_REL
        and place_ok
        and control_fails
    )
    detail = (
        f"fed {fmt(fed_total)} particles across "
        f"{sorted(B1_FED_COUNTS)}; box inventory {fmt(counted[0])} at "
        f"dt={dts[0]:g} s and {fmt(counted[1])} at dt={dts[1]:g} s -- "
        f"relative error {fmt(counted_rel)}, dt-dependence {fmt(dt_rel)} "
        f"(tol {fmt(ROUNDOFF_REL)}); every birth_* ledger row equals its fed "
        f"count exactly: {rows_ok}; particle residual {fmt(resid)}"
    )
    for name, home, where, extra, foreign_rel, rel in places:
        detail += (
            f"\n        {name} fed alone at cell {home}: {fmt(extra)} "
            f"particles {where} (exactly zero required), share at the other "
            f"channels' cells {fmt(foreign_rel)} of the fed count (tol "
            f"{fmt(ROUNDOFF_REL)}), injected relative error {fmt(rel)}"
        )
    detail += (
        f"\n        NEGATIVE CONTROL (same numbers as RATES): inventory "
        f"{fmt(control[0])} -> {fmt(control[1])}, ratio {control_ratio:.9f} "
        f"= the dt ratio {dts[1] / dts[0]:.9f}, and neither equals the fed "
        f"count -- the sampling error the counted path removes "
        f"(control behaves as required: {control_fails})"
    )
    return (
        "B1 counted boundary inflow: injected == counted particles, per "
        "face, independent of the tick length",
        ok,
        detail,
    )


def gate_b2():
    """The in-solver counted-source ledger: handed == accumulated, accepted only.

    STATEMENT 2 of the B1 three. On the engaged production arm every tick's
    handed count must be exactly the tally the ACCEPTED steps since the last
    tick put into the accumulator -- nothing dropped, nothing counted twice,
    nothing contributed by an attempt the solver rejected. Per cell, per
    channel, over several ticks::

        sum(handed over ticks) + outstanding accumulator
            == sum(accepted attempts' bookings)

    The two sides are read from opposite ends of the handshake: the left
    from what ``TransientDVM.update`` actually received, the right from the
    per-attempt tallies ``_accept_step_attempt`` folded in.

    NEGATIVE CONTROL: restore the accumulator after the tick instead of
    before it -- the reset-order defect the pattern exists to prevent --
    and the identity over-counts, because each tick then re-hands every
    earlier tick's particles. A bare rejected attempt is also run and must
    move the accumulator by exactly zero.
    """
    sim = make_sim(
        neutral_kinetic_dvm_nvz=16,
        neutral_kinetic_dvm_nvp=6,
        **PRODUCTION_GEOMETRY_KEYS,
    )
    channels = LAPDSim1D._DVM_COUNTED_SOURCES

    def measure(target, ticks=4):
        handed = {n: np.zeros(target.geometry.cells) for n in channels}
        accepted = {n: np.zeros(target.geometry.cells) for n in channels}
        update = TransientDVM.update
        accept = LAPDSim1D._accept_step_attempt

        def spy_update(self, dt, **kwargs):
            for name, row in (kwargs.get("source_counts") or {}).items():
                handed[name] = handed[name] + np.asarray(row, dtype=float)
            return update(self, dt, **kwargs)

        def spy_accept(self, attempt):
            booking = getattr(attempt, "source_booking", None)
            if booking is not None:
                for name, row in booking.items():
                    accepted[name] = accepted[name] + row
            return accept(self, attempt)

        TransientDVM.update = spy_update
        LAPDSim1D._accept_step_attempt = spy_accept
        try:
            run_until_updates(target, ticks)
        finally:
            TransientDVM.update = update
            LAPDSim1D._accept_step_attempt = accept
        worst = 0.0
        for name in channels:
            left = handed[name] + target._dvm_source_booked[name]
            scale = max(float(np.max(np.abs(accepted[name]))), 1e-300)
            worst = max(worst, float(np.max(np.abs(left - accepted[name]))) / scale)
        return worst, handed, accepted

    worst, handed, accepted = measure(sim)

    # A rejected attempt moves nothing: build one and drop it on the floor.
    before = {n: sim._dvm_source_booked[n].copy() for n in channels}
    sim._attempt_step(dt=sim.suggest_timestep().dt)
    rejected_moved = any(
        not np.array_equal(before[n], sim._dvm_source_booked[n])
        for n in channels
    )

    # NEGATIVE CONTROL: reset AFTER the update instead of before it.
    control = make_sim(
        neutral_kinetic_dvm_nvz=16,
        neutral_kinetic_dvm_nvp=6,
        **PRODUCTION_GEOMETRY_KEYS,
    )
    advance = LAPDSim1D._dvm_advance

    def late_reset(self, dt_neutral):
        held = {n: row.copy() for n, row in self._dvm_source_booked.items()}
        out = advance(self, dt_neutral)
        self._dvm_source_booked = held
        return out

    LAPDSim1D._dvm_advance = late_reset
    try:
        control_worst, _, _ = measure(control)
    finally:
        LAPDSim1D._dvm_advance = advance

    ok = (
        worst < ROUNDOFF_REL
        and not rejected_moved
        and control_worst > 1.0e-3
    )
    totals = ", ".join(
        f"{name} {fmt(float(handed[name].sum()))}" for name in channels
    )
    return (
        "B2 in-solver counted-source ledger: handed == accumulated over "
        "accepted steps, per cell",
        ok,
        f"4 ticks, particles handed: {totals}\n        "
        f"worst |handed + outstanding - accepted| / scale = {fmt(worst)} "
        f"(tol {fmt(ROUNDOFF_REL)})\n        "
        f"a rejected attempt moved the accumulator: {rejected_moved} "
        f"(False required)\n        "
        f"NEGATIVE CONTROL (reset AFTER the update, the double-count "
        f"defect): worst residual {fmt(control_worst)} -- the identity "
        f"breaks, as required",
    )


def gate_b3():
    """Counted channels carry the RIGHT energy, with every channel armed.

    STATEMENT 3 of the B1 three, and the one the B0a review made binding:
    a surface booked at the wrong energy is INVISIBLE to the closed-box and
    in-solver statements above, because it moves no particle. With every
    channel armed at once -- pumping at both ends, an anode mesh, a wall, a
    puff, and all four counted channels -- both ledgers close, and each
    counted channel's energy row is its own count times the mean energy of
    the spectrum it is emitted into, rebuilt here from the velocity grid
    rather than read back from the engine.

    NEGATIVE CONTROL: emit the cathode face at the WALL temperature instead
    of the cathode's. Not one particle moves -- the particle ledger closes
    to the same roundoff and every ``birth_*`` count is unchanged -- while
    the cathode-face energy row moves by a large factor. Only this
    statement sees it.
    """
    def armed(T_s_K):
        nz = 12
        dvm = TransientDVM(
            geometry=uniform_tube(nz),
            nvz=16,
            nvp=6,
            s_L=0.3,
            s_R=0.3,
            accommodation=0.4,
            exchange_model=EXCHANGE_MODEL,
            mesh_face=nz // 2,
            transparency=0.642,
        )
        dvm.seed_from_density(np.full(nz, 1.0e13), np.full(nz, 1.0e13))
        rows = b1_fed_rows(dvm)
        puff = np.zeros(nz)
        puff[3] = 3.0e17
        plasma = geometry_plasma(nz)
        led = dvm.update(
            CADENCE_S,
            source_counts=rows,
            sources={"puff": puff / CADENCE_S},
            T_s_K=T_s_K,
            **plasma,
        )
        return dvm, led, rows, plasma

    dvm, led, rows, plasma = armed(B1_T_S_K)
    g = dvm.g
    M_i = np.stack([
        g.maxwellian(max(float(t), 0.02), float(u))
        for t, u in zip(plasma["Ti_eV"], plasma["u_i"])
    ])
    E_Mi = (M_i * dvm.E_bin).sum(axis=(1, 2))
    expected = {
        "cathode_face": float(rows["cathode_face"].sum())
        * dvm._energy_of(g.half_flux_spectrum(B1_T_S_K, +1)),
        "collector_face": float(rows["collector_face"].sum())
        * dvm._energy_of(g.half_flux_spectrum(dvm.T_wall_K, -1)),
        "anode": float(rows["anode"].sum()) * dvm.E_wall_mean,
        "recombination": float((rows["recombination"] * E_Mi).sum()),
    }
    energy = led["energy"]
    energy_rows = []
    energy_ok = True
    for name, want in expected.items():
        got = energy[f"birth_{name}"]
        rel = abs(got - want) / max(abs(want), 1e-300)
        energy_rows.append((name, got, want, rel))
        energy_ok = energy_ok and rel < ROUNDOFF_REL
    part = abs(ledger_residual(led)["distribution_rel"])
    part_dom = abs(ledger_residual(led)["domain_rel"])
    ener = abs(ledger_energy_residual(led)["distribution_rel"])
    ener_dom = abs(ledger_energy_residual(led)["domain_rel"])

    # NEGATIVE CONTROL: the same particles, booked at the wrong temperature.
    _, wrong_led, _, _ = armed(dvm.T_wall_K)
    counts_unchanged = all(
        wrong_led[f"birth_{name}"] == led[f"birth_{name}"]
        for name in LEDGER_EXTERNAL_BIRTHS
    )
    wrong_part = abs(ledger_residual(wrong_led)["distribution_rel"])
    wrong_energy = wrong_led["energy"]["birth_cathode_face"]
    energy_moved = (
        abs(wrong_energy - energy["birth_cathode_face"])
        / abs(energy["birth_cathode_face"])
    )
    control_ok = (
        counts_unchanged and wrong_part < ROUNDOFF_REL and energy_moved > 0.5
    )

    ok = (
        energy_ok
        and part < ROUNDOFF_REL
        and part_dom < ROUNDOFF_REL
        and ener < ROUNDOFF_REL
        and ener_dom < ROUNDOFF_REL
        and control_ok
    )
    detail = (
        f"every channel armed, counted inflow at {B1_T_S_K:g} K: particle "
        f"residual dist {fmt(part)} / domain {fmt(part_dom)}, energy "
        f"residual dist {fmt(ener)} / domain {fmt(ener_dom)} "
        f"(tol {fmt(ROUNDOFF_REL)})"
    )
    for name, got, want, rel in energy_rows:
        detail += (
            f"\n        birth_{name}: ledger {fmt(got)} erg vs count x "
            f"spectrum mean {fmt(want)} erg, relative {fmt(rel)}"
        )
    detail += (
        f"\n        NEGATIVE CONTROL (cathode face emitted at the WALL "
        f"temperature): every birth_* particle count unchanged "
        f"({counts_unchanged}), particle residual {fmt(wrong_part)} -- still "
        f"closed -- while birth_cathode_face energy moves by "
        f"{energy_moved:.3f} of itself; invisible to statements 1 and 2, "
        f"caught here (control behaves as required: {control_ok})"
    )
    return (
        "B3 counted channels carry their own emission energy, every channel "
        "armed",
        ok,
        detail,
    )


# ------------------------------------- B3 cylindrical-wall reflection

#: [WR] Accommodation values the wall-reflection pins are taken at: the two
#: values the surface-physics program brackets, the value between them, and an
#: arbitrary interior point that is not a round number, so a statement that
#: only holds on the registered triple cannot pass.
WR_ALPHAS = (0.35, 0.40, 0.46, 0.7307)

#: [WR] The two values of ``neutral_kinetic_dvm_wall_reflection``, shipped
#: default first.
WR_MODES = ("specular", "diffuse_elastic")

#: [WR1] The net wall energy exchange as a fraction of the incident wall
#: energy, per ``neutral_kinetic_dvm_exchange`` closure and per accommodation
#: coefficient, on the ``wr_box`` fixture (sealed 10-cell tube, 16 x 6
#: velocity grid). The wall does NOT exchange zero net energy with a gas
#: already at its own temperature: the accommodated share re-emits on the
#: discrete ``wall_emission_spectrum`` while the wall absorbs at
#: ``nu_w ~ vp`` times the volume Maxwellian, and the two agree only in the
#: continuum. What is left is the accommodated share's velocity-RESOLUTION
#: offset, and it is what these values pin.
#:
#: The offset is a property of the DISCRETIZATION, not of any wall model: it
#: scales with alpha exactly (statement (b) in ``gate_wr1`` is that identity)
#: and it falls with refinement -- 4.1e-2 here at alpha = 0.35 against 5.5e-3
#: on a 96 x 32 grid. It also depends on the exchange closure, which sets how
#: much of the column reaches the wall, which is why this table is keyed by
#: closure and covers every model the suite re-runs these gates under. Any
#: change to the fixture's grid or to the closure set means re-measuring and
#: re-registering these values.
WR1_OFFSET_PINS = {
    "cauchy_chord": {
        0.35: 4.0503988540753155e-02,
        0.40: 4.6290272618003600e-02,
        0.46: 5.3233813510704095e-02,
        0.7307: 8.4560755504938020e-02,
    },
    "geometric": {
        0.35: 4.0188576570484996e-02,
        0.40: 4.5929801794839980e-02,
        0.46: 5.2819272064065830e-02,
        0.7307: 8.3902265428723980e-02,
    },
}

#: [WR1] Relative tolerance the pinned offsets above are held to. The fixture
#: is deterministic, so the only spread it must absorb is the last-ulp
#: difference between the two reflection arms (worst 2.9e-15 as measured); a
#: tolerance this tight makes the pin a real constraint rather than a band.
WR1_OFFSET_REL_TOL = 1.0e-6


def wr1_pins():
    """Return the ``WR1_OFFSET_PINS`` row for the exchange closure in force.

    Raises ``KeyError`` naming the closure if the suite is re-run under one
    the table does not pin -- a silently unpinned pass is exactly what this
    member exists to prevent.
    """
    try:
        return WR1_OFFSET_PINS[EXCHANGE_MODEL]
    except KeyError:
        raise KeyError(
            f"WR1 has no pinned wall-exchange offsets for "
            f"neutral_kinetic_dvm_exchange = {EXCHANGE_MODEL!r}; measure them "
            f"on the wr_box fixture and register them in WR1_OFFSET_PINS"
        ) from None

#: [WR1] Relative perturbation of the accommodated share the pin's negative
#: control injects. It must be large against ``WR1_OFFSET_REL_TOL`` and small
#: enough to be the size of a plausible defect rather than a different physics
#: problem.
WR1_CONTROL_PERTURBATION = 1.0e-3


def wr_box(alpha, mode, nz=10, nvz=16, nvp=6):
    """Return a sealed uniform tube seeded at the wall temperature.

    No pumping and no plasma, so the only surfaces with any energy in them
    are the cylindrical wall and the two specular end planes, and the gas
    starts at the same temperature the wall re-emits at. That is the
    fixture the detailed-balance statement is taken on.
    """
    dvm = bare_dvm(
        nz=nz, nvz=nvz, nvp=nvp, accommodation=alpha, wall_reflection=mode
    )
    dvm.s_L = 0.0
    dvm.s_R = 0.0
    dvm.seed_from_density(np.full(nz, 1.0e13), np.full(nz, 1.0e13))
    return dvm


def wr_wall_rows(dvm, led):
    """Return ``(net, accommodated-share offset, loss)`` of the wall channel."""
    e = led["energy"]
    offset = dvm.accommodation * (
        e["loss_wall"] - led["loss_wall"] * dvm.E_wall_mean
    )
    return e["net_surface_wall"], offset, e["loss_wall"]


def gate_wr1():
    """Wall detailed balance: the wall exchange IS the accommodated offset.

    STATEMENT 1 of the B3 three (the closed-box synthetic case). A uniform
    gas at the wall temperature in a sealed tube, at four accommodation
    coefficients and under BOTH values of
    ``neutral_kinetic_dvm_wall_reflection``.

    WHAT THE PIN IS. A wall facing a gas already at its own temperature
    exchanges no net energy in the continuum, and that zero-net statement was
    this member's original pin. It is NOT the behaviour of a discretized
    wall and is retired as a pin (re-registered 2026-08-31, Tom): the
    accommodated share re-emits on the discrete ``wall_emission_spectrum``
    while the wall absorbs at ``nu_w ~ vp`` times the volume Maxwellian, and
    those two agree bin-by-bin only in the continuum limit. The residue is
    the accommodated share's velocity-RESOLUTION offset. The registered
    statement is therefore that the net wall energy exchange EQUALS that
    offset, at its measured per-alpha value, and three things are measured:

    (a) The REGISTERED PIN: ``|net_surface_wall|`` over the incident wall
        energy, against its pinned value in ``WR1_OFFSET_PINS`` to
        ``WR1_OFFSET_REL_TOL`` relative, at all four alpha and under both
        arms. The fixture is deterministic, so this is a per-alpha NUMBER and
        not a band. The offset is a property of the DISCRETIZATION: it is
        present unchanged at the base commit (reproduce with
        ``k2_dvm_wall_detailed_balance_base_probe.py``), it scales with alpha
        exactly, it falls with refinement, and it moves with the exchange
        closure -- so the pins are keyed by ``EXCHANGE_MODEL`` as well as by
        alpha, covering the closure the suite re-runs this member under, and
        an unpinned closure raises rather than passing silently
        (``wr1_pins``).

    (b) What that offset EXACTLY is, which is the identity justifying (a)'s
        reading and which localizes the offset away from this member:
        ``net_surface_wall == alpha * (E_incident - N_incident *
        E_wall_mean)`` to roundoff, under both values. Every term on the
        right belongs to the ACCOMMODATED share; the non-accommodated share
        -- the only thing the selector touches -- contributes nothing to it.

    (c) The pin the selector owns, at roundoff: the two arms exchange the
        SAME wall energy at every alpha (the diffuse-elastic return is
        elastic, so it moves no energy across the surface), and at
        ``alpha = 0`` -- the pure reflection limit, where the accommodated
        offset is switched off -- the net wall exchange is exactly zero under
        both.

    TWO NEGATIVE CONTROLS, one per pinned statement:

    * (a)'s: perturb the accommodated share by ``WR1_CONTROL_PERTURBATION``
      relative and re-measure against the SAME pins. Every alpha and both
      arms move off their pin by that perturbation, far outside the
      tolerance, while identity (b) still closes at roundoff -- so a
      mis-sized accommodated share is caught by the pinned value and by
      nothing else here.
    * (c)'s: solve the re-emission temperature against the CONTINUUM relation
      ``<E> = 2 k T`` instead of against the spectrum's discrete mean energy
      -- the analytic-target booking the member is written to avoid. The two
      arms then stop exchanging the same wall energy and the alpha = 0 limit
      stops being zero, while the particle ledger closes to the same roundoff
      throughout.
    """
    pins = wr1_pins()
    pinned = []
    exact = []
    same_arm = []
    for alpha in WR_ALPHAS:
        by_mode = {}
        pin = pins[alpha]
        for mode in WR_MODES:
            dvm = wr_box(alpha, mode)
            led = dvm.update(1.0e-5, **zero_plasma(dvm))
            net, offset, loss = wr_wall_rows(dvm, led)
            by_mode[mode] = net
            ratio = abs(net) / loss
            pinned.append((alpha, mode, ratio, pin, abs(ratio - pin) / pin))
            exact.append(
                (alpha, mode, abs(net - offset) / max(abs(offset), 1e-300))
            )
        spec, diff = by_mode["specular"], by_mode["diffuse_elastic"]
        same_arm.append(
            (alpha, spec, diff, abs(spec - diff) / max(abs(spec), 1e-300))
        )
    zero_alpha = []
    for mode in WR_MODES:
        dvm = wr_box(0.0, mode)
        led = dvm.update(1.0e-5, **zero_plasma(dvm))
        net, _offset, loss = wr_wall_rows(dvm, led)
        zero_alpha.append((mode, abs(net) / loss))

    pin_ok = all(dev < WR1_OFFSET_REL_TOL for _a, _m, _r, _p, dev in pinned)
    exact_ok = all(rel < ROUNDOFF_REL for _a, _m, rel in exact)
    same_ok = all(rel < ROUNDOFF_REL for _a, _s, _d, rel in same_arm)
    zero_ok = all(rel < ROUNDOFF_REL for _m, rel in zero_alpha)

    # NEGATIVE CONTROL for (a): a mis-sized accommodated share. Re-measure at
    # a perturbed alpha against the SAME pins -- every alpha and both arms
    # must leave the tolerance, while identity (b) still closes at roundoff,
    # so the pinned VALUE is what catches it and not the identity.
    off_control = []
    off_control_exact = 0.0
    for alpha in WR_ALPHAS:
        pin = pins[alpha]
        for mode in WR_MODES:
            dvm = wr_box(alpha * (1.0 + WR1_CONTROL_PERTURBATION), mode)
            led = dvm.update(1.0e-5, **zero_plasma(dvm))
            net, offset, loss = wr_wall_rows(dvm, led)
            off_control.append(abs(abs(net) / loss - pin) / pin)
            off_control_exact = max(
                off_control_exact,
                abs(net - offset) / max(abs(offset), 1e-300),
            )
    off_control_fails = (
        min(off_control) > WR1_OFFSET_REL_TOL
        and off_control_exact < ROUNDOFF_REL
    )

    # NEGATIVE CONTROL for (c): the continuum target, not the discrete one.
    solve = TransientDVM._solve_wall_return_spectra

    def continuum(self, e_bar):
        from cablp.solvers._sim1d.physics.kinetic_dvm import (
            M_HE as _M,
            _cosine_wall_spectra,
        )
        return _cosine_wall_spectra(
            self.g, np.sqrt(np.asarray(e_bar, dtype=float) / (2.0 * _M))
        )

    TransientDVM._solve_wall_return_spectra = continuum
    try:
        control_same = []
        control_part = 0.0
        for alpha in WR_ALPHAS:
            nets = []
            for mode in WR_MODES:
                dvm = wr_box(alpha, mode)
                led = dvm.update(1.0e-5, **zero_plasma(dvm))
                nets.append(led["energy"]["net_surface_wall"])
                control_part = max(
                    control_part, abs(ledger_residual(led)["distribution_rel"])
                )
            control_same.append(
                abs(nets[0] - nets[1]) / max(abs(nets[0]), 1e-300)
            )
        dvm = wr_box(0.0, "diffuse_elastic")
        led = dvm.update(1.0e-5, **zero_plasma(dvm))
        control_zero = abs(
            led["energy"]["net_surface_wall"]
        ) / led["energy"]["loss_wall"]
    finally:
        TransientDVM._solve_wall_return_spectra = solve
    control_fails = (
        max(control_same) > 1.0e-3
        and control_zero > 1.0e-3
        and control_part < ROUNDOFF_REL
    )

    ok = (
        pin_ok
        and exact_ok
        and same_ok
        and zero_ok
        and off_control_fails
        and control_fails
    )
    worst_pin = max(dev for _a, _m, _r, _p, dev in pinned)
    detail = (
        "(a) REGISTERED PIN -- |net wall energy exchange| / incident wall "
        "energy against its pinned per-alpha value: worst relative "
        f"{fmt(worst_pin)} over "
        f"{sorted(set(a for a, _m, _r, _p, _d in pinned))}, both arms, at "
        f"neutral_kinetic_dvm_exchange = {EXCHANGE_MODEL!r} "
        f"(tol {fmt(WR1_OFFSET_REL_TOL)}). The exchange is NOT zero: it is "
        "the accommodated share's velocity-resolution offset, a property of "
        "the discretization, present unchanged at the base commit (reproduce "
        "with k2_dvm_wall_detailed_balance_base_probe.py)"
    )
    for alpha, mode, ratio, pin, dev in pinned:
        detail += (
            f"\n            alpha={alpha:g} [{mode}]: {fmt(ratio)} vs pin "
            f"{fmt(pin)}, relative {fmt(dev)}"
        )
    detail += (
        "\n        NEGATIVE CONTROL (accommodated share perturbed by "
        f"{fmt(WR1_CONTROL_PERTURBATION)} relative, same pins): every alpha "
        f"and both arms leave the pin by at least {fmt(min(off_control))} "
        f"while identity (b) still closes at {fmt(off_control_exact)} -- "
        f"caught only by the pinned value (control behaves as required: "
        f"{off_control_fails})"
    )
    detail += (
        "\n        (b) that offset is EXACTLY the accommodated share: worst "
        "|net - alpha (E_inc - N_inc E_wall_mean)| / offset = "
        f"{fmt(max(rel for _a, _m, rel in exact))} (tol {fmt(ROUNDOFF_REL)}); "
        "the non-accommodated share contributes nothing to it"
    )
    detail += "\n        (c) the pin this member owns, at roundoff:"
    for alpha, spec, diff, rel in same_arm:
        detail += (
            f"\n            alpha={alpha:g}: net wall exchange specular "
            f"{fmt(spec)} erg vs diffuse_elastic {fmt(diff)} erg, relative "
            f"{fmt(rel)}"
        )
    for mode, rel in zero_alpha:
        detail += (
            f"\n            alpha=0 [{mode}] (pure reflection): "
            f"|net| / incident = {fmt(rel)}"
        )
    detail += (
        "\n        NEGATIVE CONTROL (continuum <E> = 2kT target instead of "
        f"the discrete moment): the arms diverge by up to "
        f"{fmt(max(control_same))} and the alpha=0 limit reads "
        f"{fmt(control_zero)}, while the particle ledger still closes at "
        f"{fmt(control_part)} -- caught only by statement (c) "
        f"(control behaves as required: {control_fails})"
    )
    return (
        "WR1 cylindrical-wall detailed balance: the net wall energy exchange "
        "is the accommodated share's velocity-resolution offset at its "
        "pinned per-alpha value, and the reflection selector moves no energy "
        "across the wall",
        ok,
        detail,
    )


def gate_wr2():
    """The selector is READ in-solver, and degenerates at full accommodation.

    STATEMENT 2 of the B3 three. On the engaged production arm the two
    values must reach the ENGINE (a selector the solver validates and then
    drops is the silent-inert trap this repo refuses) and must actually move
    the kinetic state, while both ledgers keep closing over several ticks.
    The complementary half is the degeneracy the pair has by construction:
    at ``alpha = 1`` there is no non-accommodated share to place, so the two
    values must produce BIT-IDENTICAL distributions -- checked here both
    in-solver and, on the standalone engine, under both annulus treatments,
    since the jump arm places the same array as a wall launch.

    NEGATIVE CONTROL, owned by this statement: drop the selector on the way
    to the engine -- the un-threaded-config defect -- and the in-solver
    trajectories become identical at an interior alpha, so "the arm is read"
    fails here while every ledger statement above still passes.
    """
    keys = dict(
        neutral_kinetic_dvm_nvz=16,
        neutral_kinetic_dvm_nvp=6,
        neutral_kinetic_dvm_accommodation=0.40,
        **PRODUCTION_GEOMETRY_KEYS,
    )

    def run(mode, alpha=None, ticks=3):
        over = dict(keys, neutral_kinetic_dvm_wall_reflection=mode)
        if alpha is not None:
            over["neutral_kinetic_dvm_accommodation"] = alpha
        sim = make_sim(**over)
        ledgers = run_until_updates(sim, ticks)
        return sim, ledgers

    sim_s, led_s = run("specular")
    sim_d, led_d = run("diffuse_elastic")
    threaded = (
        sim_s._dvm.wall_reflection == "specular"
        and sim_d._dvm.wall_reflection == "diffuse_elastic"
    )
    moved = not np.array_equal(sim_s._dvm.f_a, sim_d._dvm.f_a)
    part = max(
        abs(ledger_residual(led)["domain_rel"])
        for led in led_s + led_d
    )
    ener = max(
        abs(ledger_energy_residual(led)["domain_rel"])
        for led in led_s + led_d
    )

    # Degeneracy at full accommodation, in-solver and on the bare engine.
    one_s, _ = run("specular", alpha=1.0)
    one_d, _ = run("diffuse_elastic", alpha=1.0)
    degenerate = np.array_equal(
        one_s._dvm.f_c, one_d._dvm.f_c
    ) and np.array_equal(one_s._dvm.f_a, one_d._dvm.f_a)
    bare = {}
    for flights in ("rates", "bounded_chord"):
        states = []
        for mode in WR_MODES:
            dvm = bare_dvm(
                nz=10, nvz=16, nvp=6, accommodation=1.0,
                wall_reflection=mode, annulus_flights=flights,
            )
            dvm.seed_from_density(
                np.full(dvm.nz, 1.0e13), np.full(dvm.nz, 1.0e13)
            )
            for _ in range(3):
                dvm.update(1.0e-5, **zero_plasma(dvm))
            states.append((dvm.f_c.copy(), dvm.f_a.copy()))
        bare[flights] = np.array_equal(
            states[0][0], states[1][0]
        ) and np.array_equal(states[0][1], states[1][1])

    # NEGATIVE CONTROL: the selector never reaches the engine.
    build = TransientDVM.__init__

    def dropped(self, **kwargs):
        kwargs.pop("wall_reflection", None)
        return build(self, **kwargs)

    TransientDVM.__init__ = dropped
    try:
        ctrl_s, _ = run("specular", ticks=2)
        ctrl_d, _ = run("diffuse_elastic", ticks=2)
    finally:
        TransientDVM.__init__ = build
    control_fails = np.array_equal(ctrl_s._dvm.f_a, ctrl_d._dvm.f_a)

    ok = (
        threaded
        and moved
        and part < ROUNDOFF_REL
        and ener < ROUNDOFF_REL
        and degenerate
        and all(bare.values())
        and control_fails
    )
    return (
        "WR2 in-solver: the wall-reflection selector reaches the engine, "
        "moves the state, and degenerates bit-exactly at alpha = 1",
        ok,
        f"3 ticks per arm on the production machine at alpha=0.40: engine "
        f"carries the selected value ({threaded}), the two arms' annulus "
        f"distributions differ ({moved}); worst domain residual particle "
        f"{fmt(part)} / energy {fmt(ener)} (tol {fmt(ROUNDOFF_REL)})\n        "
        f"alpha=1 degeneracy: in-solver bit-identical ({degenerate}); bare "
        f"engine bit-identical under annulus_flights='rates' "
        f"({bare['rates']}) and 'bounded_chord' ({bare['bounded_chord']})\n"
        f"        NEGATIVE CONTROL (selector dropped on the way to the "
        f"engine): the two arms' distributions become identical "
        f"({control_fails}) -- the silent-inert defect, caught only here",
    )


def gate_wr3():
    """The diffuse-elastic return conserves count, energy and axial momentum.

    STATEMENT 3 of the B3 three, with EVERY channel armed at once -- pumping
    at both ends, an anode mesh, a wall, a puff, a live plasma and all four
    counted boundary channels. Per tick, on the array the engine actually
    placed (``TransientDVM.last_wall_return``, in particles per bin):

      * its COUNT is ``(1 - alpha)`` times the wall landings exactly;
      * its ENERGY is ``(1 - alpha)`` times the incident wall energy exactly,
        both sides taken as DISCRETE moments of the bins -- which is what
        makes the reflection elastic rather than approximately so;
      * its net ``v_z`` is zero to roundoff, so the surface hands the gas no
        axial momentum;

    and both ledgers close at the tolerance the particle ledger is held to.

    NEGATIVE CONTROL, owned by this statement: book the re-emission at the
    WALL MEAN energy -- the analytic-target booking, and the one a
    ``birth_wall_reflected`` row would take if it were written like the
    accommodated one. Not one particle moves, so WR1 and WR2 and every
    particle statement in this suite are blind to it, while the energy
    ledger's residual leaves roundoff and the mis-booked row moves by more
    than a tenth of itself.
    """
    def armed(alpha, mode, nz=12):
        dvm = TransientDVM(
            geometry=uniform_tube(nz),
            nvz=16,
            nvp=6,
            s_L=0.3,
            s_R=0.3,
            accommodation=alpha,
            wall_reflection=mode,
            exchange_model=EXCHANGE_MODEL,
            mesh_face=nz // 2,
            transparency=0.642,
        )
        dvm.seed_from_density(np.full(nz, 1.0e13), np.full(nz, 1.0e13))
        rows = b1_fed_rows(dvm)
        puff = np.zeros(nz)
        puff[3] = 3.0e17
        led = dvm.update(
            CADENCE_S,
            source_counts=rows,
            sources={"puff": puff / CADENCE_S},
            T_s_K=B1_T_S_K,
            **geometry_plasma(nz),
        )
        return dvm, led

    rows = []
    ok = True
    for alpha in WR_ALPHAS:
        dvm, led = armed(alpha, "diffuse_elastic")
        wr = dvm.last_wall_return
        want_n = (1.0 - alpha) * led["loss_wall"]
        want_e = (1.0 - alpha) * led["energy"]["loss_wall"]
        got_n = float(wr.sum())
        got_e = dvm._energy_of(wr)
        vz = float((wr * dvm.g.VZ[None, :, :]).sum())
        vz_scale = float((wr * np.abs(dvm.g.VZ[None, :, :])).sum())
        n_rel = abs(got_n - want_n) / want_n
        e_rel = abs(got_e - want_e) / want_e
        vz_rel = abs(vz) / max(vz_scale, 1e-300)
        part = abs(ledger_residual(led)["distribution_rel"])
        part_dom = abs(ledger_residual(led)["domain_rel"])
        ener = abs(ledger_energy_residual(led)["distribution_rel"])
        ener_dom = abs(ledger_energy_residual(led)["domain_rel"])
        rows.append((alpha, got_n, n_rel, e_rel, vz_rel, part, ener))
        ok = ok and max(
            n_rel, e_rel, vz_rel, part, part_dom, ener, ener_dom
        ) < ROUNDOFF_REL

    # NEGATIVE CONTROL: the reflected row booked at the WALL MEAN.
    book = TransientDVM._book_energy_ledger

    def wall_mean(self, **kwargs):
        out = book(self, **kwargs)
        was = out["birth_wall_reflected"]
        now = (1.0 - kwargs["alpha"]) * float(
            kwargs["N_wall"].sum()
        ) * self.E_wall_mean
        out["birth_wall_reflected"] = now
        out["net_surface_wall"] = out["net_surface_wall"] + was - now
        return out

    clean_dvm, clean_led = armed(0.40, "diffuse_elastic")
    TransientDVM._book_energy_ledger = wall_mean
    try:
        _, wrong_led = armed(0.40, "diffuse_elastic")
    finally:
        TransientDVM._book_energy_ledger = book
    counts_unchanged = all(
        wrong_led[f"birth_{name}"] == clean_led[f"birth_{name}"]
        for name in LEDGER_BIRTH_CHANNELS
    )
    wrong_part = abs(ledger_residual(wrong_led)["distribution_rel"])
    wrong_ener = abs(ledger_energy_residual(wrong_led)["distribution_rel"])
    row_moved = abs(
        wrong_led["energy"]["birth_wall_reflected"]
        - clean_led["energy"]["birth_wall_reflected"]
    ) / abs(clean_led["energy"]["birth_wall_reflected"])
    control_fails = (
        counts_unchanged
        and wrong_part < ROUNDOFF_REL
        and wrong_ener > 1.0e-6
        and row_moved > 0.1
    )
    ok = ok and control_fails

    detail = (
        "every channel armed (both ends pumping, anode mesh, wall, puff, "
        "live plasma, all four counted channels):"
    )
    for alpha, got_n, n_rel, e_rel, vz_rel, part, ener in rows:
        detail += (
            f"\n        alpha={alpha:g}: re-emitted {fmt(got_n)} particles, "
            f"count rel {fmt(n_rel)}, energy rel {fmt(e_rel)}, net v_z "
            f"{fmt(vz_rel)}; ledger residual particle {fmt(part)} / energy "
            f"{fmt(ener)} (tol {fmt(ROUNDOFF_REL)})"
        )
    detail += (
        f"\n        NEGATIVE CONTROL (re-emission booked at the WALL MEAN "
        f"energy): every birth_* particle count unchanged "
        f"({counts_unchanged}) and the particle ledger still closes at "
        f"{fmt(wrong_part)}, while birth_wall_reflected moves by "
        f"{row_moved:.3f} of itself and the energy residual goes to "
        f"{fmt(wrong_ener)} (control behaves as required: {control_fails})"
    )
    return (
        "WR3 diffuse-elastic wall return: count, energy and net v_z exact, "
        "every channel armed",
        ok,
        detail,
    )


# ------------------------------------------------------- B2 closed faces

#: [B2] Which interior face the synthetic closed-face tubes carry. Chosen off
#: the middle so the two sides have different cell counts and a statement that
#: only holds by symmetry cannot pass.
CF_FACE = 3
CF_CELLS = 8

#: [B2] The cathode surface temperature the closed-face gates emit at, and the
#: same 1910 K the B1 fixture uses -- far from the 300 K wall, so a side
#: booked at the wrong one is a large move rather than a subtle one.
CF_T_S_K = 1910.0


def closed_face_tube(nz=CF_CELLS, face=CF_FACE, Rp=15.0, Rm=50.0,
                     length_cm=800.0):
    """Return a uniform tube carrying ONE interior closed plasma face.

    The device geometries reach a closed face only through their typed cell
    roles, and every one of them puts it at the cathode. This fixture states
    the face directly -- ``plasma_open`` false at ``face`` and at the two
    domain ends, which is the array shape the resolved geometry hands the
    engine -- so the transport statement is about the OPERATOR at a closed
    face rather than about the machine that happens to have one.

    ``Rm == Rp`` leaves the tube with NO annulus, which is what makes the
    zero-transmission statement a bit-exact identity: the zone-exchange and
    wall rates are then identically zero, so the axial column flux is the
    ONLY route from one side of the face to the other. Passing ``Rm > Rp``
    restores the annulus and with it the bypass around the disc.
    """
    dz = np.full(nz, length_cm / nz)
    Rp_cm = np.full(nz, Rp)
    Rm_cm = np.full(nz, Rm)
    plasma_open = np.ones(nz + 1, dtype=bool)
    plasma_open[0] = False
    plasma_open[nz] = False
    plasma_open[face] = False
    return SimpleNamespace(
        cells=nz,
        length_cm=dz,
        Rp_cm=Rp_cm,
        Rm_cm=Rm_cm,
        plasma_volume_cm3=np.pi * Rp_cm**2 * dz,
        neutral_volume_cm3=np.pi * Rm_cm**2 * dz,
        plasma_open=plasma_open,
        plasma_absorbing=np.zeros(nz + 1, dtype=bool),
    )


def cf_open_the_face(dvm):
    """Disable the closed-face BC in place -- the B2 negative control.

    Strips the resolved topology and nothing else, so the run is the SAME
    engine on the SAME geometry with only the zero-transmission condition
    removed. That is what makes the control a control: the traffic it lets
    across the face is exactly the traffic the BC exists to stop.
    """
    dvm.closed_faces = ()
    dvm._closed_face = [False] * (dvm.nz + 1)
    dvm._closed_emitters = ()
    return dvm


def cf_box(seed_dead, seed_live, closed=True, Rm=15.0, nz=CF_CELLS,
           face=CF_FACE):
    """Return a sealed closed-face box seeded per side, in cm^-3."""
    dvm = TransientDVM(
        geometry=closed_face_tube(nz=nz, face=face, Rm=Rm),
        nvz=16,
        nvp=6,
        accommodation=0.4,
        exchange_model=EXCHANGE_MODEL,
        s_L=0.0,
        s_R=0.0,
    )
    if not closed:
        cf_open_the_face(dvm)
    col = np.zeros(nz)
    ann = np.zeros(nz)
    col[:face] = seed_dead
    col[face:] = seed_live
    if Rm > 15.0:
        ann[:face] = seed_dead
        ann[face:] = seed_live
    dvm.seed_from_density(col, ann, T_K=400.0)
    return dvm


def cf_block_inventory(dvm, face=CF_FACE):
    """Return the ``(low, high)`` side inventories, pending buffers included.

    Each domain end belongs to the side it terminates, so a side's buffered
    end return is part of that side's books and the two numbers partition
    the whole box exactly.
    """
    V = dvm.V_col
    low = float((dvm.f_c[:face] * V[:face, None, None]).sum())
    low += float((dvm.f_a[:face] * dvm.V_ann[:face, None, None]).sum())
    low += float(dvm.pend_L_c.sum() + dvm.pend_L_a.sum())
    high = float((dvm.f_c[face:] * V[face:, None, None]).sum())
    high += float((dvm.f_a[face:] * dvm.V_ann[face:, None, None]).sum())
    high += float(dvm.pend_R_c.sum() + dvm.pend_R_a.sum())
    return low, high


def gate_cf1():
    """Closed faces transmit exactly nothing, and each side's books balance.

    STATEMENT 1 of the B2 three (the closed-box synthetic case), and it
    carries both pre-registered gates of the plan row.

    THE TRANSMISSION PIN, as a hard identity rather than a tolerance: on a
    tube with no annulus at all the axial column flux is the only route
    across the face, so seeding gas on ONE side alone must leave the other
    side's column at EXACTLY zero -- bit-exact, not to roundoff -- for as
    many updates as it is asked to hold.

    THE PLENUM-TRAFFIC LEDGER: seeded on BOTH sides, the sealed box has no
    pumping and no plasma, and the face returns to each side exactly what
    that side delivered, so each side's own inventory is separately
    conserved to roundoff. That is the traffic statement per side, which a
    whole-box total cannot make: a leak from one side into the other
    cancels in the total and shows up here.

    NON-VACUITY: with the annulus restored the dead side's ANNULUS does
    fill, because the annulus is the clear bore around the disc and is
    deliberately not blocked. Without that check the exact zero above would
    also be reported by an engine that had simply stopped.

    NEGATIVE CONTROL: strip the resolved closed faces and nothing else.
    Column particles then cross into the far side, so the exact-zero
    identity fails at this statement, which is the one that owns it.
    """
    updates = 8
    plasma = zero_plasma(cf_box(0.0, 1.0e13))

    def walk(dvm):
        worst_p = 0.0
        worst_e = 0.0
        for _ in range(updates):
            led = dvm.update(CADENCE_S, **plasma)
            worst_p = max(
                worst_p, abs(ledger_residual(led)["distribution_rel"])
            )
            worst_e = max(
                worst_e, abs(ledger_energy_residual(led)["distribution_rel"])
            )
        return worst_p, worst_e

    # (a) the transmission pin: gas on the high side only, no annulus.
    sealed = cf_box(0.0, 1.0e13)
    worst_p, worst_e = walk(sealed)
    crossed = float(
        (sealed.f_c[:CF_FACE] * sealed.V_col[:CF_FACE, None, None]).sum()
    )
    blocked_traffic = float(sealed.last_ledger["loss_closed_face_blocked"])

    # (b) the per-side traffic ledger: both sides seeded, sealed box.
    both = cf_box(2.0e13, 1.0e13)
    before = cf_block_inventory(both)
    walk(both)
    after = cf_block_inventory(both)
    side_rel = tuple(
        abs(a - b) / max(abs(b), 1e-300) for a, b in zip(after, before)
    )

    # (c) non-vacuity: with an annulus, the dead side's annulus DOES fill.
    bypass = cf_box(0.0, 1.0e13, Rm=50.0)
    walk(bypass)
    dead_annulus = float(
        (bypass.f_a[:CF_FACE] * bypass.V_ann[:CF_FACE, None, None]).sum()
    )

    # NEGATIVE CONTROL: the same box with the closed faces stripped.
    control = cf_box(0.0, 1.0e13, closed=False)
    walk(control)
    control_crossed = float(
        (control.f_c[:CF_FACE] * control.V_col[:CF_FACE, None, None]).sum()
    )

    ok = (
        crossed == 0.0
        and blocked_traffic > 0.0
        and max(side_rel) < ROUNDOFF_REL
        and worst_p < ROUNDOFF_REL
        and worst_e < ROUNDOFF_REL
        and dead_annulus > 0.0
        and control_crossed > 0.0
    )
    return (
        "CF1 closed faces transmit exactly zero; each side's traffic balances",
        ok,
        f"{updates} updates, closed face {CF_FACE} of {CF_CELLS} cells, no "
        f"annulus\n        "
        f"column particles across the face: {crossed!r} (exactly 0.0 "
        f"required) against {fmt(blocked_traffic)} blocked at the face on "
        f"the last update alone\n        "
        f"per-side inventory drift, both sides seeded: low {fmt(side_rel[0])}"
        f", high {fmt(side_rel[1])} (tol {fmt(ROUNDOFF_REL)})\n        "
        f"ledger residuals: particle {fmt(worst_p)}, energy {fmt(worst_e)} "
        f"(tol {fmt(ROUNDOFF_REL)})\n        "
        f"NON-VACUITY (annulus restored): dead-side annulus holds "
        f"{fmt(dead_annulus)} particles, so the bore around the disc is open "
        f"and the exact zero above is the COLUMN block\n        "
        f"NEGATIVE CONTROL (closed faces stripped): "
        f"{fmt(control_crossed)} column particles cross -- the identity "
        f"fails without the BC, as required",
    )


def gate_cf2():
    """The in-solver closed-face ledger on the production machine.

    STATEMENT 2 of the B2 three. On the engaged production arm the closed
    face is the CATHODE DISC, resolved from the typed cell roles and not
    from any configuration that names it: the plenum sits on one side and
    the live cathode cell on the other. Over several ticks the channel must
    carry real traffic, book the same count blocked as re-emitted, and
    leave both ledgers closed to roundoff -- the statement the synthetic
    box cannot make, because only the coupled arm runs the face with every
    plasma channel, the recycle ghost and the real mesh alongside it.

    NEGATIVE CONTROL: re-emit HALF of what the face blocked. Not a particle
    crosses the face either way, so the transmission pin of statement 1 is
    untouched; the ledger stops closing, which is what this statement owns.
    """
    def measure(sim, ticks=4):
        ledgers = run_until_updates(sim, ticks)
        worst_p = max(
            abs(ledger_residual(led)["distribution_rel"]) for led in ledgers
        )
        worst_e = max(
            abs(ledger_energy_residual(led)["distribution_rel"])
            for led in ledgers
        )
        pair = max(
            abs(led["loss_closed_face_blocked"]
                - led["birth_closed_face_reemit"])
            for led in ledgers
        )
        traffic = min(led["loss_closed_face_blocked"] for led in ledgers)
        return worst_p, worst_e, pair, traffic

    sim = make_sim(
        neutral_kinetic_dvm_nvz=16,
        neutral_kinetic_dvm_nvp=6,
        **PRODUCTION_GEOMETRY_KEYS,
    )
    dvm = sim._dvm
    geom = sim.geometry
    roles = np.asarray(geom.cell_role)
    faces = dvm.closed_faces
    sides = tuple(
        (int(cell), str(roles[cell]))
        for _face, _d, cell, _surface in dvm._closed_emitters
    )
    topology_ok = (
        faces == tuple(int(f) for f in geom.cathode_face_indices)
        and {role for _cell, role in sides} == {"plenum", "cathode"}
        and any(
            surface for _face, _d, _cell, surface in dvm._closed_emitters
        )
    )
    worst_p, worst_e, pair, traffic = measure(sim)

    # NEGATIVE CONTROL: give back half of what the face stopped.
    control = make_sim(
        neutral_kinetic_dvm_nvz=16,
        neutral_kinetic_dvm_nvp=6,
        **PRODUCTION_GEOMETRY_KEYS,
    )
    spectra = TransientDVM._closed_face_spectra

    def half_back(self, T_s_K):
        return {
            key: 0.5 * value
            for key, value in spectra(self, T_s_K).items()
        }

    TransientDVM._closed_face_spectra = half_back
    try:
        control_p, _control_e, _control_pair, _t = measure(control)
    finally:
        TransientDVM._closed_face_spectra = spectra

    ok = (
        topology_ok
        and traffic > 0.0
        and pair == 0.0
        and worst_p < ROUNDOFF_REL
        and worst_e < ROUNDOFF_REL
        and control_p > 1.0e-6
    )
    return (
        "CF2 in-solver closed-face ledger on the production machine",
        ok,
        f"closed faces {faces} == cathode faces "
        f"{tuple(int(f) for f in geom.cathode_face_indices)}; sides "
        f"{sides} (topology as required: {topology_ok})\n        "
        f"4 ticks: smallest per-tick traffic {fmt(traffic)} particles, "
        f"|blocked - re-emitted| {pair!r} (exactly 0.0 required)\n        "
        f"ledger residuals: particle {fmt(worst_p)}, energy {fmt(worst_e)} "
        f"(tol {fmt(ROUNDOFF_REL)})\n        "
        f"NEGATIVE CONTROL (half the blocked count re-emitted): particle "
        f"residual {fmt(control_p)} -- the ledger breaks while nothing "
        f"crosses the face, as required",
    )


def gate_cf3():
    """Each closed-face side re-emits at ITS OWN surface temperature.

    STATEMENT 3 of the B2 three, and the one the B0a standing lesson makes
    binding: a surface re-emitting at the wrong temperature moves no
    particle at all, so statements 1 and 2 close to exactly the same
    roundoff whether it is right or wrong. Only an every-channel-armed
    ENERGY statement sees it.

    Armed on the production machine with pumping, the anode mesh, the
    cylindrical wall, a puff and all four counted inflows live at once, the
    closed-face energy birth row must equal, side by side, that side's own
    blocked count times the mean energy of the half-flux spectrum it is
    emitted into -- the cathode side at the live surface temperature, the
    plenum side at the wall's -- rebuilt here from the velocity grid rather
    than read back from the engine.

    NEGATIVE CONTROL: emit the CATHODE side of the face at the wall
    temperature. Every particle count is unchanged and the particle ledger
    closes to the same roundoff, while the closed-face energy row collapses
    towards the cold spectrum.
    """
    geom = make_sim(**PRODUCTION_GEOMETRY_KEYS).geometry
    nz = geom.cells

    def armed(wrong_temperature=False):
        dvm = TransientDVM(
            geometry=geom,
            nvz=16,
            nvp=6,
            s_L=0.3,
            s_R=0.3,
            accommodation=0.4,
            exchange_model=EXCHANGE_MODEL,
            mesh_face=int(geom.anode_face_indices[0]),
            transparency=0.642,
        )
        if wrong_temperature:
            # The whole face at the WALL temperature: the cathode side's
            # own spectrum is the only thing this changes.
            dvm._closed_surface_dirs = ()
            for emit in (+1, -1):
                dvm._closed_wall_spectra[(True, emit)] = (
                    dvm.g.half_flux_spectrum(dvm.T_wall_K, emit)
                )
        dvm.seed_from_density(
            np.full(nz, 1.0e13), np.full(nz, 1.0e13), T_K=400.0
        )
        rows = {name: np.zeros(nz) for name in B1_FED_COUNTS}
        rows["cathode_face"][dvm.cath_cell] = B1_FED_COUNTS["cathode_face"]
        rows["collector_face"][dvm.coll_cell] = B1_FED_COUNTS["collector_face"]
        rows["recombination"][nz // 3] = B1_FED_COUNTS["recombination"]
        rows["anode"][2 * nz // 3] = B1_FED_COUNTS["anode"]
        puff = np.zeros(nz)
        puff[nz // 4] = 3.0e17
        led = dvm.update(
            CADENCE_S,
            source_counts=rows,
            sources={"puff": puff / CADENCE_S},
            T_s_K=CF_T_S_K,
            **geometry_plasma(nz),
        )
        return dvm, led

    dvm, led = armed()
    # Rebuild the expected energy from the per-side blocked counts. The
    # engine's own per-side split is re-derived here from the emitters, so a
    # side attributed to the wrong temperature cannot satisfy both.
    g = dvm.g
    want = 0.0
    per_side = []
    for _face, d_in, cell, surface in dvm._closed_emitters:
        emit = -d_in
        T_side = CF_T_S_K if surface else dvm.T_wall_K
        count = float(dvm.last_closed_counts[emit][cell])
        contribution = count * dvm._energy_of(
            g.half_flux_spectrum(T_side, emit)
        )
        per_side.append((cell, T_side, count, contribution))
        want += contribution
    got = led["energy"]["birth_closed_face_reemit"]
    energy_rel = abs(got - want) / max(abs(want), 1e-300)

    part = abs(ledger_residual(led)["distribution_rel"])
    part_dom = abs(ledger_residual(led)["domain_rel"])
    ener = abs(ledger_energy_residual(led)["distribution_rel"])
    ener_dom = abs(ledger_energy_residual(led)["domain_rel"])

    # NEGATIVE CONTROL: the cathode side booked at the wall temperature.
    _wrong_dvm, wrong_led = armed(wrong_temperature=True)
    counts_unchanged = (
        wrong_led["birth_closed_face_reemit"]
        == led["birth_closed_face_reemit"]
        and all(
            wrong_led[f"birth_{name}"] == led[f"birth_{name}"]
            for name in LEDGER_EXTERNAL_BIRTHS
        )
    )
    wrong_part = abs(ledger_residual(wrong_led)["distribution_rel"])
    energy_moved = abs(
        wrong_led["energy"]["birth_closed_face_reemit"] - got
    ) / max(abs(got), 1e-300)
    control_ok = (
        counts_unchanged and wrong_part < ROUNDOFF_REL and energy_moved > 0.5
    )

    ok = (
        energy_rel < ROUNDOFF_REL
        and part < ROUNDOFF_REL
        and part_dom < ROUNDOFF_REL
        and ener < ROUNDOFF_REL
        and ener_dom < ROUNDOFF_REL
        and control_ok
    )
    detail = (
        f"every channel armed on the production machine, cathode surface at "
        f"{CF_T_S_K:g} K\n        "
        f"birth_closed_face_reemit: ledger {fmt(got)} erg vs per-side counts "
        f"x spectrum means {fmt(want)} erg, relative {fmt(energy_rel)} "
        f"(tol {fmt(ROUNDOFF_REL)})"
    )
    for cell, T_side, count, contribution in per_side:
        detail += (
            f"\n        side cell {cell} at {T_side:g} K: {fmt(count)} "
            f"particles carrying {fmt(contribution)} erg"
        )
    detail += (
        f"\n        ledger residuals: particle dist {fmt(part)} / domain "
        f"{fmt(part_dom)}, energy dist {fmt(ener)} / domain {fmt(ener_dom)}"
        f"\n        NEGATIVE CONTROL (cathode side emitted at the WALL "
        f"temperature): every particle count unchanged "
        f"({counts_unchanged}), particle residual {fmt(wrong_part)} -- still "
        f"closed -- while the closed-face energy row moves by "
        f"{energy_moved:.3f} of itself; invisible to CF1 and CF2, caught "
        f"here (control behaves as required: {control_ok})"
    )
    return (
        "CF3 closed-face sides carry their own emission energy, every "
        "channel armed",
        ok,
        detail,
    )


# ------------------------------- B5 cathode-side energetic recycle (jet)


#: [B5] The reflection coefficients under test. They MIRROR the fluid
#: channel's shipped ``cathode_jet_R_N`` / ``cathode_jet_R_E``, which is what
#: makes the two arms statements about the same surface.
CJ_R_N = 0.34
CJ_R_E = 0.18

#: [B5] The velocity grid these gates run on. Every other gate in this file
#: runs the coarse ``(16, 6)`` grid; the jet cannot. Its launch spectrum is a
#: tens-of-eV directed beam smeared onto the axial axis, and the ``(16, 6)``
#: grid's bin at that speed is itself tens of eV wide, so the narrowest
#: spectrum it resolves carries more thermal energy than the beam has -- the
#: engine's guard says exactly that and RAISES. These gates therefore run the
#: SHIPPED grid, on which the projection lands at machine precision.
CJ_NVZ = 48
CJ_NVP = 12

#: [B5] Incident energy per collected ion [eV], ``phi_c + Ti``: a
#: production-class cathode sheath, so the launch is the energetic beam the
#: channel exists to carry rather than a near-thermal one that would pass the
#: cross-book by being indistinguishable from the thermal share.
CJ_PHI_TI_EV = 62.0

#: [B5] Counted cathode recycle over one tick [particles].
CJ_COUNT = 5.0e16

#: [B5] Registered ceiling on the cross-book |booked - debited| / debited.
#: The two are the same committed number by construction, so this is a
#: roundoff budget, not a physics tolerance.
CJ_CROSS_BOOK_REL = 1.0e-8

#: [B5] Registered ceiling on the analytic-vs-discrete faithfulness number:
#: the launch spectrum's DISCRETE mean energy against the ``(1/2) m v_back^2``
#: it is built to carry. It is the convergence tolerance of the moment
#: compensation, and a solve that misses it raises rather than launching.
CJ_MOMENT_REL = 1.0e-10


def cj_spec(R_N=CJ_R_N, R_E=CJ_R_E, T_launch_eV=None):
    return {"R_N": R_N, "R_E": R_E, "T_launch_eV": T_launch_eV}


def cj_closed_box(nz=8, jet=True, cls=None):
    """Return the closed box of :func:`closed_box_dvm` with the jet armed.

    Same disarmed box -- no annulus (``Rm == Rp``), no mesh, no pumping,
    specular end walls -- so the only energy channels are the CX/elastic
    exchange, the ionization a partner consumes, and the cathode recycle
    this member splits. On the jet grid, and empty, so what the box holds
    afterwards came in through the channel under test.
    """
    builder = TransientDVM if cls is None else cls
    dvm = builder(
        geometry=uniform_tube(nz, Rp=15.0, Rm=15.0),
        nvz=CJ_NVZ,
        nvp=CJ_NVP,
        accommodation=0.0,
        exchange_model=EXCHANGE_MODEL,
        s_L=0.0,
        s_R=0.0,
        cathode_jet=cj_spec() if jet else None,
    )
    dvm.seed_from_density(np.full(nz, 1.0e13), np.zeros(nz), T_K=400.0)
    return dvm


def cj_feed(dvm, count=CJ_COUNT, per_ion_eV=CJ_PHI_TI_EV):
    """Return ``(source_counts, incident_erg)`` for one counted tick."""
    rows = {"cathode_face": np.zeros(dvm.nz)}
    rows["cathode_face"][dvm.cath_cell] = count
    incident = np.zeros(dvm.nz)
    incident[dvm.cath_cell] = count * per_ion_eV * EV
    return rows, incident


class _CJUnbookedEnergy(TransientDVM):
    """Harness defect: the jet births atoms and forgets their energy row.

    The one failure the closed-box statement exists to catch -- a channel
    added to the distribution without an entry in the energy ledger. Every
    particle count is untouched, so the PARTICLE ledger closes exactly as it
    did; only the energy identity sees it.
    """

    def _book_energy_ledger(self, **kwargs):
        kwargs["e_birth_cathode_jet"] = 0.0
        return TransientDVM._book_energy_ledger(self, **kwargs)


class _CJWallSpectrum(TransientDVM):
    """Harness defect: the jet launches on the 300 K cosine-wall spectrum.

    The B0a-class defect -- the counted particles arrive, the ledger closes
    against itself, and the energy the surface was debited is simply not the
    energy the gas received. Statement 3's cross-book is the only statement
    that sees it.
    """

    def _cathode_jet_launch_spectrum(self, e_launch, cell):
        return self.M_wall


def gate_cj1():
    """Cathode jet, closed box: both ledgers close and the split conserves.

    STATEMENT 1 of the B5 three. In the disarmed box the counted cathode
    recycle is the only external channel, so what the split does is visible
    on its own: the ``R_N`` share becomes an energetic volume birth and the
    remainder the thermal face inflow, the two counts sum to the handed
    count EXACTLY, and both the particle and the energy ledger close at the
    tolerance I1/I2/I6 hold the arm to.

    Non-vacuity is asserted rather than assumed: the jet's energy row must
    carry a real share of the tick's energy throughput, or the identity
    under test is 0 == 0. It does -- the beam is ``(R_E/R_N)(phi_c + Ti)``
    per atom against a 400 K seed.

    NEGATIVE CONTROL: birth the same atoms and book their energy row at zero
    (:class:`_CJUnbookedEnergy`) -- an energy channel added to the
    distribution and left out of the ledger. The particle ledger closes to
    the same roundoff and the ENERGY distribution residual goes to order
    one, so the control fails at THIS statement.
    """
    nz = 8
    worst_part = 0.0
    worst_ener = 0.0
    worst_split = 0.0
    live = float("inf")
    dvm = cj_closed_box(nz)
    plasma = geometry_plasma(nz)
    for _ in range(5):
        rows, incident = cj_feed(dvm)
        led = dvm.update(
            CADENCE_S,
            source_counts=rows,
            cathode_jet_incident_erg=incident,
            **plasma,
        )
        p = ledger_residual(led)
        e = ledger_energy_residual(led)
        worst_part = max(
            worst_part, abs(p["distribution_rel"]), abs(p["domain_rel"])
        )
        worst_ener = max(
            worst_ener, abs(e["distribution_rel"]), abs(e["domain_rel"])
        )
        split = led["birth_cathode_jet"] + led["birth_cathode_face"]
        worst_split = max(worst_split, abs(split - CJ_COUNT) / CJ_COUNT)
        live = min(
            live, abs(led["energy"]["birth_cathode_jet"]) / e["scale"]
        )

    control = cj_closed_box(nz, cls=_CJUnbookedEnergy)
    control_part = 0.0
    control_ener = 0.0
    for _ in range(5):
        rows, incident = cj_feed(control)
        led = control.update(
            CADENCE_S,
            source_counts=rows,
            cathode_jet_incident_erg=incident,
            **plasma,
        )
        control_part = max(
            control_part, abs(ledger_residual(led)["distribution_rel"])
        )
        control_ener = max(
            control_ener,
            abs(ledger_energy_residual(led)["distribution_rel"]),
        )

    ok = (
        worst_part < ROUNDOFF_REL
        and worst_ener < ROUNDOFF_REL
        and worst_split < ROUNDOFF_REL
        and live > 1.0e-6
        and control_part < ROUNDOFF_REL
        and control_ener > 1.0e-3
    )
    return (
        "CJ1 cathode jet, closed box: particle and energy ledgers close, "
        "the recycle split conserves the counted stream",
        ok,
        f"5 ticks at {fmt(CJ_COUNT)} counted particles, "
        f"{CJ_PHI_TI_EV} eV incident per ion, R_N={CJ_R_N} R_E={CJ_R_E}: "
        f"worst particle residual {fmt(worst_part)}, worst energy residual "
        f"{fmt(worst_ener)}, |jet + thermal - handed| / handed "
        f"{fmt(worst_split)} (tol {fmt(ROUNDOFF_REL)}); jet share of the "
        f"energy throughput {fmt(live)} (> 1e-6 required)\n        "
        f"NEGATIVE CONTROL (birth the atoms, book their energy row at zero): "
        f"particle residual {fmt(control_part)} -- unchanged, as the defect "
        f"moves no particle -- and energy distribution residual "
        f"{fmt(control_ener)}, which is the statement failing"
    )


def gate_cj2():
    """In-solver: one committed pair feeds both the birth and the debit.

    STATEMENT 2 of the B5 three. On the engaged arm with the jet armed, per
    tick and cumulatively over the window:

    * every counted cathode particle is injected -- ``birth_cathode_jet +
      birth_cathode_face`` equals the count the tick was handed;
    * the birth's ENERGY row is the count times the discrete mean of the
      spectrum that was placed, which is true by construction and is
      re-derived here from the ledger rather than read back from the engine;
    * the CUMULATIVE surface debit -- the cathode energy ledger's named
      ``backscatter`` row, booked per ACCEPTED step -- equals the cumulative
      ``birth_cathode_jet`` energy the ticks handed the gas, plus ``R_E``
      times the accumulator the next tick has not yet been given. That is
      the created-once identity: the surface gave up exactly what the gas
      received, and what it has not yet received is still owed.

    NEGATIVE CONTROL: form the same debit from a TICK-TIME reading of
    ``(phi_c + Ti)`` applied to the whole window's count -- the sampling the
    stage-weighted accumulator exists to avoid. The plasma moves inside a
    tick, so that debit and the counted one differ by far more than the
    identity's roundoff, and the control fails at THIS statement.
    """
    sim = make_sim(
        nx=24,
        neutral_kinetic_dvm_cathode_jet=True,
        neutral_kinetic_dvm_nvz=CJ_NVZ,
        neutral_kinetic_dvm_nvp=CJ_NVP,
    )
    handed = []
    update = TransientDVM.update

    def spy_update(self, dt, **kwargs):
        rows = (kwargs.get("source_counts") or {}).get("cathode_face")
        incident = kwargs.get("cathode_jet_incident_erg")
        record = {
            "count": float(np.sum(rows)) if rows is not None else 0.0,
            "incident": (
                float(np.sum(incident)) if incident is not None else 0.0
            ),
            "phi_ti_tick_eV": float(
                np.max(np.asarray(kwargs["Ti_eV"], dtype=float))
            ),
        }
        led = update(self, dt, **kwargs)
        record["led"] = dict(led)
        handed.append(record)
        return led

    TransientDVM.update = spy_update
    try:
        run_until_updates(sim, 4)
        # A few more accepted steps, so the window ends BETWEEN ticks and
        # the "what the next tick is still owed" term of the identity is
        # non-zero rather than trivially satisfied. Any tick that fires in
        # them is recorded by the same spy.
        for _ in range(5):
            advance_one_step(sim)
    finally:
        TransientDVM.update = update

    worst_count = 0.0
    worst_row = 0.0
    births = 0.0
    for rec in handed:
        led = rec["led"]
        injected = led["birth_cathode_jet"] + led["birth_cathode_face"]
        worst_count = max(
            worst_count,
            abs(injected - rec["count"]) / max(rec["count"], 1e-300),
        )
        row = led["energy"]["birth_cathode_jet"]
        births += row
        # count x discrete-spectrum mean, rebuilt from the ledger: the row
        # divided by the count IS that mean, so the statement is that the
        # per-atom energy it implies is the committed launch energy.
        if led["birth_cathode_jet"] > 0.0:
            per_atom = row / led["birth_cathode_jet"]
            target = (CJ_R_E / CJ_R_N) * rec["incident"] / rec["count"]
            worst_row = max(worst_row, abs(per_atom - target) / target)

    debited_erg = sim._cathode_energy_ledger_J["backscatter"] * 1.0e7
    outstanding = CJ_R_E * float(np.sum(sim._dvm_cathode_jet_energy_booked))
    identity = abs(debited_erg - (births + outstanding)) / max(
        births + outstanding, 1e-300
    )

    # NEGATIVE CONTROL: the tick-time reading applied to the window count.
    control_debit = sum(
        CJ_R_E * rec["count"] * rec["phi_ti_tick_eV"] * EV for rec in handed
    )
    control_rel = abs(control_debit - births) / max(births, 1e-300)

    ok = (
        len(handed) >= 4
        and births > 0.0
        and outstanding > 0.0
        and worst_count < ROUNDOFF_REL
        and worst_row < CJ_CROSS_BOOK_REL
        and identity < ROUNDOFF_REL
        and control_rel > 1.0e-3
    )
    return (
        "CJ2 in-solver cathode jet: injected == counted, and the cumulative "
        "surface debit == the cumulative birth energy + what is still owed",
        ok,
        f"{len(handed)} ticks, {fmt(births)} erg of backscatter born; "
        f"worst |injected - counted| / counted {fmt(worst_count)} "
        f"(tol {fmt(ROUNDOFF_REL)}); worst per-atom launch energy against "
        f"(R_E/R_N)(phi_c + Ti) {fmt(worst_row)} "
        f"(tol {fmt(CJ_CROSS_BOOK_REL)})\n        "
        f"cathode ledger backscatter row {fmt(debited_erg)} erg vs births "
        f"{fmt(births)} + outstanding {fmt(outstanding)} erg: relative "
        f"{fmt(identity)} (tol {fmt(ROUNDOFF_REL)})\n        "
        f"NEGATIVE CONTROL (tick-time (phi_c + Ti) on the window count): "
        f"debit {fmt(control_debit)} erg, {fmt(control_rel)} relative from "
        f"the counted one -- the sampling error the stage accumulator "
        f"removes, and it fails this statement"
    )


def gate_cj3():
    """Every channel armed: the gas receives what the surface was debited.

    STATEMENT 3 of the B5 three, and the one the B0a standing lesson makes
    binding: a backscatter launched at the WRONG energy moves exactly the
    right number of particles, so statements 1 and 2 close to the same
    roundoff whether the spectrum is right or wrong. Only a cross-book
    between the two ledgers sees it.

    With the cylindrical wall, an anode mesh, pumping at both ends, a puff,
    volume recombination, an anode rebirth and both recycle faces all live
    at once, and once per annulus treatment:

    * ``count * <E>_spectrum`` -- the ledger's ``birth_cathode_jet`` energy
      row -- against ``R_E`` times the incident energy the surface was
      debited by, at ``CJ_CROSS_BOOK_REL``;
    * the analytic-vs-discrete faithfulness number: the same spectrum's
      discrete mean energy against the ``(1/2) m v_back^2`` it is built to
      carry, at ``CJ_MOMENT_REL``. It is a DISCLOSED number rather than an
      assumption -- the compensation solve is what makes the two agree, and
      a solve that gives up raises instead of launching.

    NEGATIVE CONTROL: launch the same counted atoms on the 300 K cosine-wall
    spectrum (:class:`_CJWallSpectrum`) -- the surface still debited for the
    energetic beam, the gas given the cold one. BOTH normalizations are
    reported, because which one a gate reads decides what it can catch: the
    ROW-RELATIVE form (against the debit itself) is what this gate tests,
    and the THROUGHPUT-NORMALIZED form (against the tick's whole energy
    scale) is the one a ledger-residual gate would have seen -- and the
    ledger residual sees nothing at all, because the engine books the row
    from the spectrum it placed either way.
    """
    nz = 12
    results = {}
    for flights in ("rates", "bounded_chord"):
        for wrong in (False, True):
            cls = _CJWallSpectrum if wrong else TransientDVM
            dvm = cls(
                geometry=uniform_tube(nz),
                nvz=CJ_NVZ,
                nvp=CJ_NVP,
                s_L=0.3,
                s_R=0.3,
                accommodation=0.4,
                exchange_model=EXCHANGE_MODEL,
                annulus_flights=flights,
                mesh_face=nz // 2,
                transparency=0.642,
                cathode_jet=cj_spec(),
            )
            dvm.seed_from_density(np.full(nz, 1.0e13), np.full(nz, 1.0e13))
            sources = {
                "recombination": np.full(nz, 1.0e15),
                "puff": np.zeros(nz),
                "anode": np.zeros(nz),
                "collector_face": np.zeros(nz),
            }
            sources["puff"][3] = 3.0e17
            sources["anode"][nz // 2] = 2.0e16
            sources["collector_face"][-1] = 4.0e16
            plasma = geometry_plasma(nz)
            cross = 0.0
            moment = 0.0
            throughput = 0.0
            resid = 0.0
            for _ in range(4):
                rows, incident = cj_feed(dvm)
                led = dvm.update(
                    CADENCE_S,
                    sources=sources,
                    source_counts=rows,
                    cathode_jet_incident_erg=incident,
                    **plasma,
                )
                row = led["energy"]["birth_cathode_jet"]
                debited = CJ_R_E * float(np.sum(incident))
                cross = max(cross, abs(row - debited) / debited)
                e = ledger_energy_residual(led)
                throughput = max(
                    throughput, abs(row - debited) / e["scale"]
                )
                resid = max(
                    resid,
                    abs(e["distribution_rel"]),
                    abs(e["domain_rel"]),
                )
                # The faithfulness number, rebuilt independently of the
                # engine: the spectrum the launch energy implies, and its
                # own discrete mean energy against that launch energy.
                e_launch = (CJ_R_E / CJ_R_N) * CJ_PHI_TI_EV * EV
                spec = TransientDVM._cathode_jet_launch_spectrum(
                    dvm, e_launch, int(dvm.cath_cell)
                )
                got = 0.5 * M_HE * float((spec * dvm.g.V2).sum())
                moment = max(moment, abs(got - e_launch) / e_launch)
            results[(flights, wrong)] = (cross, moment, throughput, resid)

    good = [results[(f, False)] for f in ("rates", "bounded_chord")]
    bad = [results[(f, True)] for f in ("rates", "bounded_chord")]
    ok = (
        all(c < CJ_CROSS_BOOK_REL for c, _m, _t, _r in good)
        and all(m < CJ_MOMENT_REL for _c, m, _t, _r in good)
        and all(r < ROUNDOFF_REL for _c, _m, _t, r in good)
        and all(c > 0.5 for c, _m, _t, _r in bad)
        and all(r < ROUNDOFF_REL for _c, _m, _t, r in bad)
    )
    detail = (
        f"4 ticks per arm, every channel armed, {CJ_PHI_TI_EV} eV incident "
        f"per ion (launch {(CJ_R_E / CJ_R_N) * CJ_PHI_TI_EV:.4g} eV per atom)"
    )
    for flights in ("rates", "bounded_chord"):
        c, m, _t, r = results[(flights, False)]
        detail += (
            f"\n        {flights}: cross-book |row - R_E E_incident| / "
            f"(R_E E_incident) {fmt(c)} (tol {fmt(CJ_CROSS_BOOK_REL)}); "
            f"analytic-vs-discrete launch energy {fmt(m)} "
            f"(tol {fmt(CJ_MOMENT_REL)}); energy-ledger residual {fmt(r)}"
        )
    for flights in ("rates", "bounded_chord"):
        c, _m, t, r = results[(flights, True)]
        detail += (
            f"\n        NEGATIVE CONTROL [{flights}] (launch on the 300 K "
            f"cosine-wall spectrum): row-relative {fmt(c)} -- detected -- "
            f"while throughput-normalized it is {fmt(t)} and the energy "
            f"ledger still closes at {fmt(r)}, so only the row-relative "
            f"cross-book catches it"
        )
    return (
        "CJ3 cathode jet, every channel armed: the booked launch energy IS "
        "the debited surface energy",
        ok,
        detail,
    )


def gate_cj4():
    """The two cathode backscatter books cannot be armed together.

    The pairing refusal, tested as its own statement rather than through the
    model-family resolver. ``cathode_jet_surface_debit`` is its own member of
    ``KINETIC_DVM_INCOMPATIBLE_DEFAULTS`` with required value ``False``, and
    the resolver runs before the guards, so the guard sees ``False`` on every
    config this arm can be built from -- and unconditionally so, since the
    template ships the debit ``True`` while the resolver reads "explicitly
    set" as "differs from the template". A refusal gate driven through
    ``LAPDSim1D`` would therefore be testing the resolver rather than the
    guard. The guard is called directly instead, on the pair and on each half
    alone, so it stays a live statement about the PAIR if that resolver
    membership is ever relaxed.
    """
    both = {
        "neutral_kinetic_dvm_cathode_jet": True,
        "cathode_jet_surface_debit": True,
    }
    singles = (
        {"neutral_kinetic_dvm_cathode_jet": True},
        {"cathode_jet_surface_debit": True},
        {},
    )
    raised = ""
    try:
        refuse_cathode_backscatter_double_book(both)
    except ValueError as exc:
        raised = str(exc)
    quiet = True
    for single in singles:
        try:
            refuse_cathode_backscatter_double_book(single)
        except ValueError:
            quiet = False
    names = (
        "neutral_kinetic_dvm_cathode_jet" in raised
        and "cathode_jet_surface_debit" in raised
    )
    ok = bool(raised) and names and quiet
    return (
        "CJ4 the DVM cathode jet and cathode_jet_surface_debit refuse to "
        "arm together (both books debit the same R_E)",
        ok,
        f"pair refused naming both keys: {names}; each half alone and the "
        f"empty config raise nothing: {quiet}; message: {raised[:96]!r}",
    )


# ------------------------------ cathode-jet ARMING CRITERION (the latch)
#
# The criterion ships INERT (``neutral_jet_arm_current_A = 0`` declares no
# criterion), so every gate below names its own arm/disarm pair explicitly.
# That is deliberate: the registered 50/25 pair belongs to the M1 arms, not to
# the package, and a suite that inherited it from the defaults would stop
# testing the thing it is about the moment the defaults moved.


#: The REGISTERED ARM values, set explicitly by the M1 arms. NOT defaults.
JA_ARM_A = 50.0
JA_DISARM_A = 25.0

#: An arm threshold BELOW the ion current the FIRST accepted step books, so
#: the latch arms at the end of step 1. Measured on this arm: step 1 books
#: I_i = 2.657700e-02 A, and the cathode jet books no energy at all on that
#: step (the first booking is step 2). That is what lets the
#: immediately-armed run be BIT-IDENTICAL to the inert one rather than merely
#: close -- the one censored step had nothing to censor.
JA_IMMEDIATE_ARM_A = 0.02

#: Steps each latch comparison runs: long enough for the jet to book on many
#: steps and for the neutral clock to tick at least once.
JA_STEPS = 40

#: Steps the FLUID parity legs run. Shorter only because the fluid arm carries
#: the full moment neutral package and costs more per step; the statement is
#: the same one.
JA_FLUID_STEPS = 12

#: Step cap for the JA5 identity run, which continues past ``JA_STEPS`` until
#: the neutral clock has ticked at least once so BOTH right-hand terms of the
#: identity carry energy.
JA_IDENTITY_MAX_STEPS = 900

#: Steps the JA8 restart leg takes AFTER the handoff, on each of the three
#: continuation legs. Chosen for what it has to resolve, not for a horizon: the
#: negative control's re-arming has to be visible INSIDE the window, and on
#: this arm the disarmed resume re-arms at the end of its own first step, so a
#: window of a few steps already carries the divergence and the re-arm. Kept
#: small because the fluid arm carries the full moment neutral package and the
#: leg runs three times.
JA_RESTART_STEPS = 6

#: The DVM cathode-jet arm these gates ride, on the same coefficients the B5
#: gates above use.
JA_JET_KW = {
    "neutral_kinetic_dvm_cathode_jet": True,
    "neutral_kinetic_dvm_cathode_jet_R_N": CJ_R_N,
    "neutral_kinetic_dvm_cathode_jet_R_E": CJ_R_E,
}


def _ja_solve(I_i):
    """A stand-in cathode solve carrying one booked ion current."""
    return SimpleNamespace(
        beam_result=SimpleNamespace(result=SimpleNamespace(I_i=float(I_i)))
    )


def _ja_bits(y):
    """Raw uint64 view of a state vector, for bit-identity comparisons."""
    return np.ascontiguousarray(y, dtype=float).view(np.uint64)


def _ja_run(steps=JA_STEPS, drive=None, **overrides):
    """Run a DVM cathode-jet arm and return its end-of-run fingerprint.

    ``drive``, when given, maps a step index to a latch state forced BEFORE
    that step. It is how the arm/disarm/re-arm gate reaches latch histories
    this arm's monotonically rising ion current cannot produce on its own; the
    gates that are about the latch's OWN decisions never use it.
    """
    sim = make_sim(**JA_JET_KW, **overrides)
    tick_energy = 0.0
    ticks = 0
    I_i_max = 0.0
    for k in range(steps):
        if drive is not None and k in drive:
            sim._jet_armed = bool(drive[k])
        before = sim._dvm.updates
        advance_one_step(sim)
        solve = sim._cathode_solve
        if solve is not None and solve.beam_result is not None:
            candidate = float(solve.beam_result.result.I_i)
            if np.isfinite(candidate):
                I_i_max = max(I_i_max, candidate)
        if sim._dvm.updates > before:
            ticks += 1
            tick_energy += float(
                sim._dvm.last_ledger["energy"]["birth_cathode_jet"]
            )
    return {
        "y": np.asarray(sim._y, dtype=float).copy(),
        "backscatter_J": float(sim._cathode_energy_ledger_J["backscatter"]),
        "pending_erg": float(np.sum(sim._dvm_cathode_jet_energy_booked)),
        "tick_energy_erg": tick_energy,
        "ticks": ticks,
        "censored": int(sim._jet_arming_censored_steps),
        "transitions": int(sim._jet_arming_transitions),
        "I_i_max": I_i_max,
        "arming_active": bool(sim._jet_arming_active),
    }


def _ja_fluid_sim(**overrides):
    """A FLUID cathode-jet build (the shipped moment package, no DVM arm)."""
    d, fl = default_config()
    d = dict(d)
    fl = dict(fl)
    d.update(overrides)
    return LAPDSim1D(input_dict=d, input_flags=fl)


def _ja_fluid_run(steps=JA_FLUID_STEPS, **overrides):
    """Run a fluid cathode-jet arm and return its end-of-run fingerprint."""
    sim = _ja_fluid_sim(**overrides)
    spec_live = 0
    I_i_max = 0.0
    for _ in range(steps):
        advance_one_step(sim)
        solve = sim._cathode_solve
        if solve is not None and solve.beam_result is not None:
            candidate = float(solve.beam_result.result.I_i)
            if np.isfinite(candidate):
                I_i_max = max(I_i_max, candidate)
        if sim._cathode_jet_spec(sim._cathode_solve) is not None:
            spec_live += 1
    return {
        "y": np.asarray(sim._y, dtype=float).copy(),
        "Ts_K": float(sim._cathode_Ts_K),
        "spec_live": spec_live,
        "censored": int(sim._jet_arming_censored_steps),
        "transitions": int(sim._jet_arming_transitions),
        "I_i_max": I_i_max,
    }


def gate_ja1():
    """JA1 sub-threshold: nothing is launched AND nothing is debited.

    QUANTITY: both sides of the conservation pair the criterion gates -- the
    launch (the tick's ``birth_cathode_jet`` energy plus the not-yet-ticked
    accumulator) and the debit (the surface energy ledger's ``backscatter``
    row).
    SITE: the DVM cathode-jet channel through ``LAPDSim1D``'s accepted-step
    path, on the kinetic arm.
    FIXTURE: ``arm_config`` + the B5 jet coefficients, at the REGISTERED
    50 A / 25 A pair, run ``JA_STEPS`` accepted steps.
    PASS: exactly zero on BOTH sides -- not small, zero -- with the latch
    never arming, over a run whose ion current stayed below the disarm
    threshold throughout.

    Showing both sides is the whole point. A censoring that suppressed the
    launch but left the debit standing would book the surface for atoms that
    were never born, and the totals would still close.
    """
    r = _ja_run(
        neutral_jet_arm_current_A=JA_ARM_A,
        neutral_jet_disarm_current_A=JA_DISARM_A,
    )
    sub_threshold = r["I_i_max"] < JA_DISARM_A
    ok = (
        r["arming_active"]
        and sub_threshold
        and r["pending_erg"] == 0.0
        and r["tick_energy_erg"] == 0.0
        and r["backscatter_J"] == 0.0
        and r["transitions"] == 0
        and r["censored"] == JA_STEPS
    )
    return (
        "JA1 arming criterion, sub-threshold: the cathode jet launches "
        "NOTHING and the surface is debited NOTHING",
        ok,
        f"{JA_STEPS} accepted steps at arm={fmt(JA_ARM_A)} A / "
        f"disarm={fmt(JA_DISARM_A)} A; worst booked I_i over the run "
        f"{fmt(r['I_i_max'])} A, below the disarm threshold throughout: "
        f"{sub_threshold}\n        "
        f"LAUNCH side: tick birth_cathode_jet energy "
        f"{fmt(r['tick_energy_erg'])} erg over {r['ticks']} ticks, "
        f"not-yet-ticked accumulator {fmt(r['pending_erg'])} erg "
        f"(exactly 0 required)\n        "
        f"DEBIT side: surface backscatter row {fmt(r['backscatter_J'])} J "
        f"(exactly 0 required); latch transitions {r['transitions']}, "
        f"censored steps {r['censored']} of {JA_STEPS}",
    )


def gate_ja2():
    """JA2 supra-threshold: an armed latch is bit-identical to no criterion.

    QUANTITY: the packed conservative state ``y`` at raw uint64, plus both
    sides of the conservation pair.
    SITE / FIXTURE: as JA1, comparing an arm whose latch arms at the end of
    step 1 (``JA_IMMEDIATE_ARM_A``, below the current step 1 books) against
    the INERT declaration ``arm = 0``.
    PASS: every bit of ``y`` identical, and both ledger sides identical.

    The one censored step is step 1, on which this arm's cathode jet books no
    energy at all -- so the comparison is exact rather than approximate, and
    it says what it should: once armed, the criterion is not in the way.
    """
    armed = _ja_run(
        neutral_jet_arm_current_A=JA_IMMEDIATE_ARM_A,
        neutral_jet_disarm_current_A=0.0,
    )
    inert = _ja_run(neutral_jet_arm_current_A=0.0)
    bits_ok = np.array_equal(_ja_bits(armed["y"]), _ja_bits(inert["y"]))
    ledger_ok = (
        armed["backscatter_J"] == inert["backscatter_J"]
        and armed["pending_erg"] == inert["pending_erg"]
        and armed["tick_energy_erg"] == inert["tick_energy_erg"]
    )
    non_vacuous = inert["tick_energy_erg"] + inert["pending_erg"] > 0.0
    ok = (
        bits_ok
        and ledger_ok
        and non_vacuous
        and armed["transitions"] == 1
        and armed["censored"] == 1
    )
    return (
        "JA2 arming criterion, supra-threshold: once armed the latch is "
        "bit-identical to declaring no criterion at all",
        ok,
        f"{JA_STEPS} accepted steps; armed arm={fmt(JA_IMMEDIATE_ARM_A)} A "
        f"(arms at the end of step 1) against the inert arm=0 declaration\n"
        f"        state vector bit-identical at raw uint64: {bits_ok}; "
        f"backscatter row {fmt(armed['backscatter_J'])} vs "
        f"{fmt(inert['backscatter_J'])} J, accumulator "
        f"{fmt(armed['pending_erg'])} vs {fmt(inert['pending_erg'])} erg, "
        f"tick energy {fmt(armed['tick_energy_erg'])} vs "
        f"{fmt(inert['tick_energy_erg'])} erg\n        "
        f"latch armed once ({armed['transitions']} transitions) and censored "
        f"exactly the one pre-booking step ({armed['censored']}); the "
        f"comparison is NON-VACUOUS -- the inert run really does launch "
        f"({non_vacuous})",
    )


def gate_ja3():
    """JA3 the hysteresis holds and does not chatter.

    QUANTITY: the latch state after each of the crossing sequence
    30 -> 60 -> 40 -> 20 A, at the registered 50/25 pair.
    SITE: ``LAPDSim1D._update_jet_arming_latch``, driven directly on
    stand-in solves so the sequence is exactly the registered one rather than
    whatever a run happens to produce.
    PASS: disarmed at 30 (never armed), ARMS at 60, STAYS ARMED at 40, and
    DISARMS at 20 -- two transitions, not four.

    The 40 A step is the gate. A bare single-threshold comparator at 50 A
    would drop the jet there and pick it up again on the next excursion
    above; the band is what stops a current dwelling near the threshold from
    switching the jet on and off step after step.
    """
    sim = make_sim(
        **JA_JET_KW,
        neutral_jet_arm_current_A=JA_ARM_A,
        neutral_jet_disarm_current_A=JA_DISARM_A,
    )
    sequence = (30.0, 60.0, 40.0, 20.0)
    expected = (False, True, True, False)
    observed = []
    for I_i in sequence:
        sim._update_jet_arming_latch(_ja_solve(I_i))
        observed.append(bool(sim._jet_armed))
    # What a single-threshold comparator at the ARM level would have done, so
    # the no-chatter claim is measured against a STATED alternative rather
    # than asserted. The two agree everywhere except the 40 A step -- which is
    # the whole difference between a band and a threshold, and is why the
    # comparison is made state-by-state rather than by counting flips (over
    # this four-point sequence both forms happen to move twice; the naive one
    # simply never comes back).
    naive = tuple(I_i >= JA_ARM_A for I_i in sequence)
    diverges_at_40 = naive[2] is False and observed[2] is True
    ok = (
        tuple(observed) == expected
        and sim._jet_arming_transitions == 2
        and diverges_at_40
    )
    return (
        "JA3 arming criterion: latched hysteresis over a 30->60->40->20 A "
        "crossing, no chatter",
        ok,
        f"arm={fmt(JA_ARM_A)} A, disarm={fmt(JA_DISARM_A)} A; sequence "
        f"{sequence} -> armed {tuple(observed)} (expected {expected})\n"
        f"        latch transitions {sim._jet_arming_transitions} "
        f"(2 required: one arm at 60 A, one disarm at 20 A). The 40 A step "
        f"stays ARMED: {observed[2]}\n        "
        f"a bare single-threshold comparator at {fmt(JA_ARM_A)} A would read "
        f"{naive} over the same sequence and DROP the jet at 40 A; the latch "
        f"holds it ({diverges_at_40}). That one step is the chatter the band "
        f"removes -- a current oscillating about the arm threshold toggles "
        f"the comparator every crossing and leaves the latch alone",
    )


def gate_ja4():
    """JA4 NEGATIVE CONTROL: the censoring is real, not vacuous.

    QUANTITY: the launch and debit a LATCH-DISABLED evaluation books over the
    same steps on which the registered 50/25 pair books nothing.
    SITE / FIXTURE: two runs of the JA1 arm, one at ``arm = 0`` (no criterion
    -- the pre-change behaviour) and one at 50/25.
    PASS: the disabled run launches AND debits strictly positive amounts
    while the armed-config run is exactly zero on both.

    Without this leg JA1 proves nothing: a jet that never fires under ANY
    configuration would pass it. This is the leg that shows the zeros are the
    criterion's doing.
    """
    disabled = _ja_run(neutral_jet_arm_current_A=0.0)
    armed = _ja_run(
        neutral_jet_arm_current_A=JA_ARM_A,
        neutral_jet_disarm_current_A=JA_DISARM_A,
    )
    launched = disabled["tick_energy_erg"] + disabled["pending_erg"]
    ok = (
        launched > 0.0
        and disabled["backscatter_J"] > 0.0
        and armed["tick_energy_erg"] == 0.0
        and armed["pending_erg"] == 0.0
        and armed["backscatter_J"] == 0.0
    )
    return (
        "JA4 arming criterion NEGATIVE CONTROL: the latch-disabled "
        "evaluation launches and debits where the registered pair does not",
        ok,
        f"same {JA_STEPS}-step arm, same state history, two declarations\n"
        f"        latch DISABLED (arm=0, the pre-change behaviour): launched "
        f"{fmt(launched)} erg (ticks {fmt(disabled['tick_energy_erg'])} + "
        f"accumulator {fmt(disabled['pending_erg'])}), surface debited "
        f"{fmt(disabled['backscatter_J'])} J\n        "
        f"latch at arm={fmt(JA_ARM_A)}/disarm={fmt(JA_DISARM_A)} A: launched "
        f"{fmt(armed['tick_energy_erg'] + armed['pending_erg'])} erg, "
        f"debited {fmt(armed['backscatter_J'])} J (exactly 0 on both)\n"
        f"        the censored energy is REAL and its magnitude is the "
        f"disabled run's own booking",
    )


def gate_ja5():
    """JA5 the cumulative backscatter identity survives arm/disarm/re-arm.

    QUANTITY: the B5 cumulative identity

        surface ``backscatter`` row
            == sum over ticks of ``birth_cathode_jet`` energy
               + R_E x the not-yet-ticked accumulator

    SITE / FIXTURE: the JA1 arm, run ``JA_STEPS`` steps with the latch DRIVEN
    disarmed part-way and re-armed later, so the run carries censored and
    uncensored steps on both sides of at least one neutral tick.
    PASS: the identity closes to ``ROUNDOFF_REL``, with no new ledger row.

    The identity is what makes the criterion safe: a censored step
    contributes zero to BOTH sides, so the books stay paired across an
    arbitrary latch history rather than only on runs that never switch.
    """
    drive = {JA_STEPS // 3: False, 2 * JA_STEPS // 3: True}
    sim = make_sim(
        **JA_JET_KW,
        neutral_jet_arm_current_A=JA_IMMEDIATE_ARM_A,
        neutral_jet_disarm_current_A=0.0,
    )
    tick_energy = 0.0
    ticks = 0
    steps = 0
    # Run past the drive sequence AND past the first neutral tick, so the
    # identity is tested with energy on BOTH of its right-hand terms: the
    # ticked ``birth_cathode_jet`` rows and the accumulator still in hand.
    while steps < JA_IDENTITY_MAX_STEPS and (
        steps < JA_STEPS or ticks < 1
    ):
        if steps in drive:
            sim._jet_armed = drive[steps]
        before = sim._dvm.updates
        advance_one_step(sim)
        steps += 1
        if sim._dvm.updates > before:
            ticks += 1
            tick_energy += float(
                sim._dvm.last_ledger["energy"]["birth_cathode_jet"]
            )
    lhs = float(sim._cathode_energy_ledger_J["backscatter"])
    pending_J = (
        CJ_R_E * float(np.sum(sim._dvm_cathode_jet_energy_booked)) * 1.0e-7
    )
    rhs = tick_energy * 1.0e-7 + pending_J
    scale = max(abs(lhs), abs(rhs), 1e-300)
    rel = abs(lhs - rhs) / scale
    censored = int(sim._jet_arming_censored_steps)
    ok = (
        rel < ROUNDOFF_REL
        and censored > 0
        and censored < steps
        and lhs > 0.0
        and ticks >= 1
        and tick_energy > 0.0
    )
    return (
        "JA5 arming criterion: the cumulative backscatter identity holds "
        "across an arm/disarm/re-arm sequence, with no new ledger row",
        ok,
        f"{steps} accepted steps, latch driven disarmed at step "
        f"{JA_STEPS // 3} and re-armed at step {2 * JA_STEPS // 3}; "
        f"{censored} censored and {steps - censored} live steps over "
        f"{ticks} neutral ticks\n        "
        f"backscatter row {fmt(lhs)} J == tick birth energy "
        f"{fmt(tick_energy * 1.0e-7)} J + R_E x accumulator "
        f"{fmt(pending_J)} J = {fmt(rhs)} J; relative {fmt(rel)} "
        f"(tol {fmt(ROUNDOFF_REL)})\n        "
        f"the run carries BOTH censored and live steps ({censored} of "
        f"{steps}), which is what makes this a statement about the latch "
        f"history rather than about a run that never switched",
    )


def gate_ja6():
    """JA6 FLUID parity: the same latch, the same two statements.

    QUANTITY: (a) sub-threshold -- a fluid arm at 50/25 against one with the
    cathode jet ABSENT (``cathode_neutral_jet=False`` and its debit off);
    (b) supra-threshold -- an immediately-arming fluid arm against the inert
    declaration.
    SITE: the fluid ``cathode_neutral_jet`` channel and the cathode surface
    power balance's retention factor, through ``LAPDSim1D``.
    FIXTURE: ``default_config()`` (which ships the fluid jet and its debit
    armed), ``JA_FLUID_STEPS`` accepted steps.
    PASS: (a) bit-identical to the jet-absent build -- which is ZERO SOURCE
    and ZERO DEBIT in one statement, since the absent build books neither;
    (b) bit-identical to the inert declaration.

    (a) is deliberately an equivalence against ABSENCE rather than two
    separate "is zero" reads: the momentum source and the retention factor
    are different quantities in different equations, and the only way to say
    both are gone at once is to compare against the build that has neither.
    """
    sub = _ja_fluid_run(
        neutral_jet_arm_current_A=JA_ARM_A,
        neutral_jet_disarm_current_A=JA_DISARM_A,
    )
    absent = _ja_fluid_run(
        cathode_neutral_jet=False,
        cathode_jet_surface_debit=False,
        # The template ships the 'total_reflected' reading, which is refused
        # without the jet it rescales. It is read only by the launch-speed
        # spec, which the absent build never calls, so standing it down is
        # what lets this build exist at all -- and it cannot perturb the
        # comparison, because nothing consults it.
        cathode_jet_energy_convention="legacy",
    )
    # The supra-threshold leg is a MATCHED-STATE A/B rather than a whole-run
    # comparison, and the fluid channel is why. Its jet books from the FIRST
    # accepted step (the DVM channel's first booking is step 2), and the latch
    # necessarily starts disarmed -- there is no discharge current yet -- so
    # an armed fluid run and an inert one differ on step 1 no matter how low
    # the arm threshold is set. That difference is the criterion WORKING, not
    # the thing this leg is about. So: advance one sim until the latch is
    # armed, then step it twice from the SAME state through the solver's own
    # Picard snapshot -- once under the criterion, once with the criterion
    # stood down -- and compare. That is the actual claim: with the latch
    # armed, the criterion is not in the way.
    supra_sim = _ja_fluid_sim(
        neutral_jet_arm_current_A=JA_IMMEDIATE_ARM_A,
        neutral_jet_disarm_current_A=0.0,
    )
    for _ in range(JA_FLUID_STEPS):
        advance_one_step(supra_sim)
    latch_armed = bool(supra_sim._jet_armed)
    snap = supra_sim._picard_snapshot()
    advance_one_step(supra_sim)
    supra = {
        "y": np.asarray(supra_sim._y, dtype=float).copy(),
        "Ts_K": float(supra_sim._cathode_Ts_K),
    }
    supra_sim._picard_restore(snap)
    supra_sim._jet_arming_active = False
    advance_one_step(supra_sim)
    inert = {
        "y": np.asarray(supra_sim._y, dtype=float).copy(),
        "Ts_K": float(supra_sim._cathode_Ts_K),
    }
    sub_ok = (
        np.array_equal(_ja_bits(sub["y"]), _ja_bits(absent["y"]))
        and sub["Ts_K"] == absent["Ts_K"]
        and sub["spec_live"] == 0
        and sub["censored"] == JA_FLUID_STEPS
    )
    supra_ok = (
        latch_armed
        and np.array_equal(_ja_bits(supra["y"]), _ja_bits(inert["y"]))
        and supra["Ts_K"] == inert["Ts_K"]
    )
    non_vacuous = not np.array_equal(
        _ja_bits(sub["y"]), _ja_bits(_ja_fluid_run()["y"])
    )
    ok = sub_ok and supra_ok and non_vacuous
    return (
        "JA6 arming criterion, FLUID parity: sub-threshold books zero source "
        "AND zero debit; supra-threshold is bit-identical to base",
        ok,
        f"{JA_FLUID_STEPS} accepted steps on the shipped fluid package; "
        f"worst booked I_i {fmt(sub['I_i_max'])} A\n        "
        f"SUB-THRESHOLD (arm={fmt(JA_ARM_A)}/disarm={fmt(JA_DISARM_A)} A) "
        f"bit-identical to the jet-ABSENT build: {sub_ok} -- jet spec live "
        f"on {sub['spec_live']} steps (0 required), censored "
        f"{sub['censored']} of {JA_FLUID_STEPS}, T_s {fmt(sub['Ts_K'])} K "
        f"against the absent build's {fmt(absent['Ts_K'])} K\n        "
        f"SUPRA-THRESHOLD, matched-state A/B after {JA_FLUID_STEPS} steps "
        f"(latch armed: {latch_armed}): one step under the criterion vs one "
        f"step with it stood down, from the SAME state through the solver's "
        f"Picard snapshot -- bit-identical: {supra_ok}, T_s "
        f"{fmt(supra['Ts_K'])} K against {fmt(inert['Ts_K'])} K\n        "
        f"NON-VACUOUS: the censored run really does differ from the live "
        f"shipped build ({non_vacuous}), so the sub-threshold equivalence is "
        f"a censoring result and not an identity between two identical runs",
    )


def gate_ja7():
    """JA7 the shipped INERT default is structurally inert.

    QUANTITY: the presence gate, the latch census, and the trajectory.
    SITE / FIXTURE: the JA1 arm built three ways -- with both keys omitted
    from ``input_dict`` entirely (so the template default applies), with both
    named explicitly at 0, and at the registered 50/25 pair.
    PASS: the two arm=0 builds agree bit-for-bit AND report the presence gate
    OFF with zero censored steps; and both differ from the 50/25 build.

    The differing leg is what stops this being a tautology: it shows the
    comparison can tell the two apart, so the agreement of the first two is
    a statement rather than an artefact.

    The corresponding statement AT THE GOLDEN STANCE -- that the inert
    default leaves the 4,000-step digest trajectory exact while the config
    identity moves by exactly these two keys -- is the digest additive proof,
    which is where the shipped-stance claim is actually made.
    """
    default = _ja_run()
    explicit = _ja_run(
        neutral_jet_arm_current_A=0.0, neutral_jet_disarm_current_A=0.0
    )
    criterion = _ja_run(
        neutral_jet_arm_current_A=JA_ARM_A,
        neutral_jet_disarm_current_A=JA_DISARM_A,
    )
    agree = np.array_equal(_ja_bits(default["y"]), _ja_bits(explicit["y"]))
    differs = not np.array_equal(
        _ja_bits(default["y"]), _ja_bits(criterion["y"])
    )
    inert_ok = (
        not default["arming_active"]
        and not explicit["arming_active"]
        and default["censored"] == 0
        and default["transitions"] == 0
    )
    ok = agree and differs and inert_ok
    return (
        "JA7 arming criterion: the shipped inert default (arm = 0) declares "
        "no criterion and cannot reach the latch",
        ok,
        f"{JA_STEPS} accepted steps; keys omitted vs both named at 0 "
        f"bit-identical: {agree}\n        "
        f"presence gate OFF on both ({not default['arming_active']}), "
        f"censored steps {default['censored']}, latch transitions "
        f"{default['transitions']} (0 required on each)\n        "
        f"and BOTH differ from the {fmt(JA_ARM_A)}/{fmt(JA_DISARM_A)} A "
        f"build ({differs}), so the agreement above is a measurement and not "
        f"a comparison of one run with itself",
    )


#: The four latch attributes the restart record carries, in the solver's own
#: register order. JA8 reads the register itself rather than restating the
#: names, so a member added to the carriage without a gate cannot pass here by
#: going unnoticed.
JA_RESTART_LATCH_ATTRS = LAPDSim1D._RESTART_JET_ARMING_ATTRS


def _ja8_sim(**overrides):
    """A restart-eligible FLUID cathode-jet build at an arming criterion.

    Two departures from :func:`_ja_fluid_sim`, both forced by the restart:
    ``neutral_equilibration`` is cleared (a resume REFUSES it at construction,
    because ``start_simulation`` would overwrite the restored state, and the
    unbroken leg clears it too so all three legs carry one config), and the
    criterion is named here so every leg arms on the same pair.

    The arm is ``JA_IMMEDIATE_ARM_A``: the latch has to be ARMED at the handoff
    for the carriage to be under test at all, and this arm's booked ion current
    reaches the registered 50 A pair nowhere inside a window a gate can afford.
    """
    d, fl = default_config()
    d = dict(d)
    fl = dict(fl)
    fl["neutral_equilibration"] = False
    d["neutral_jet_arm_current_A"] = JA_IMMEDIATE_ARM_A
    d["neutral_jet_disarm_current_A"] = 0.0
    d.update(overrides)
    return LAPDSim1D(input_dict=d, input_flags=fl)


def _ja8_latch(sim):
    """The latch's four carried members, comparable at raw bits.

    The transition clock is compared through its uint64 view rather than as a
    float so the not-yet-transitioned value (NaN) compares equal to itself: a
    gate that used ``==`` there would read a correctly carried NaN as a
    mismatch and a dropped one as a mismatch too, and could not tell them
    apart.
    """
    armed, censored, transitions, last_s = (
        getattr(sim, name) for name in JA_RESTART_LATCH_ATTRS
    )
    return (
        bool(armed),
        int(censored),
        int(transitions),
        int(_ja_bits(np.asarray([last_s], dtype=float))[0]),
    )


def _ja8_advance(sim, steps=JA_RESTART_STEPS):
    """Step ``sim`` and fingerprint what the cathode jet did while stepping.

    ``censored_entry`` is the LAUNCH observable: the number of continuation
    steps ENTERED with the jet censored, sampled from
    ``_cathode_jet_censored`` before each step rather than after it. Before is
    the only correct side. The latch advances at the END of the accepted step,
    so a probe taken afterwards reports the state the NEXT step will run under
    and would read a censored first step as an uncensored one.

    It is also the only launch probe that is uncontaminated here. Building the
    jet spec would read ``_cathode_solve``, which the restart deliberately
    DROPS and re-establishes on its first step, so a spec-based count would
    measure that omission on the resumed leg instead of the latch.

    The surface energy ledger with ``T_s`` is the DEBIT side. Both ride
    alongside the state so the continuation claim is about the channel the
    latch gates, not only about the packed vector agreeing.
    """
    censored_entry = 0
    for _ in range(steps):
        if sim._cathode_jet_censored():
            censored_entry += 1
        advance_one_step(sim)
    return {
        "y": np.asarray(sim._y, dtype=float).copy(),
        "Ts_K": float(sim._cathode_Ts_K),
        "ledger": dict(sim._cathode_energy_ledger_J),
        "censored_entry": censored_entry,
        "latch": _ja8_latch(sim),
    }


def _ja8_strip_latch(payload_path, destination):
    """Copy a payload and DELETE its four latch rows -- the pre-change record.

    This is how the negative control reaches a record written before the
    carriage existed without keeping a stale binary fixture around: the fields
    are removed from the ``cathode`` group, which is byte-for-byte the shape a
    payload from the previous build has.
    """
    shutil.copyfile(payload_path, destination)
    removed = []
    with h5py.File(destination, "r+") as h5:
        group = h5["cathode"]
        for name in JA_RESTART_LATCH_ATTRS:
            for suffix in ("", "__int", "__bool", "__none"):
                if f"{name}{suffix}" in group.attrs:
                    del group.attrs[f"{name}{suffix}"]
                    removed.append(f"{name}{suffix}")
    return removed


def gate_ja8():
    """JA8 the arming latch SURVIVES a restart, and matters when it does not.

    QUANTITY: the four carried latch members across an export/resume, and the
    continuation the resumed run then produces -- its packed state, its
    cathode surface energy ledger, ``T_s``, and the number of steps on which
    the jet spec was actually built.
    SITE: ``LAPDSim1D._RESTART_JET_ARMING_ATTRS`` written by
    ``restart_payload`` and read by ``_apply_restart_payload``, through
    ``save_restart_state``.
    FIXTURE: the shipped fluid package at ``JA_IMMEDIATE_ARM_A``, run
    ``JA_FLUID_STEPS`` accepted steps to a handoff at which the latch is ARMED
    and its census NONZERO, then ``JA_RESTART_STEPS`` further steps unbroken
    and restarted.
    PASS: (a) the resumed solver's four members are bit-identical to the
    producing solver's at the handoff instant; (b) after the same further
    steps the restarted run is bit-identical to the unbroken one on state,
    ledger, ``T_s``, launch count AND census; (c) the NEGATIVE CONTROL -- the
    same payload with the four rows deleted, which is what a record written
    before the carriage looks like -- comes up DISARMED, warns that it did,
    and DIVERGES from the unbroken run.

    (c) is what makes (a) and (b) statements rather than tautologies. Without
    it a carriage that restored nothing would still pass (b) on any window
    where the latch happened not to gate anything. It also pins the
    old-record behaviour the reader is promised: a payload missing these rows
    LOADS -- it does not raise -- and the run says out loud that it resumed
    disarmed.
    """
    unbroken = _ja8_sim()
    for _ in range(JA_FLUID_STEPS):
        advance_one_step(unbroken)
    handoff_latch = _ja8_latch(unbroken)
    # The census must be NONZERO at the handoff or the carriage is untested:
    # a latch carried as (disarmed, 0, 0, NaN) is indistinguishable from one
    # that was never carried at all.
    census_live = (
        handoff_latch[0] and handoff_latch[1] > 0 and handoff_latch[2] > 0
    )
    with tempfile.TemporaryDirectory() as tmp:
        payload = Path(tmp) / "ja8_handoff.restart.h5"
        save_restart_state(payload, unbroken)
        stripped = Path(tmp) / "ja8_stripped.restart.h5"
        removed = _ja8_strip_latch(payload, stripped)

        resumed = _ja8_sim(restart_from=str(payload))
        resumed_latch = _ja8_latch(resumed)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            control = _ja8_sim(restart_from=str(stripped))
            warned = any(
                "cathode-jet arming latch" in str(w.message) for w in caught
            )
        control_latch = _ja8_latch(control)
        # Read AT LOAD, before any of the three legs steps: the control
        # re-arms on its own first step, so a transition clock sampled after
        # the continuation would report the re-arm and not the seed.
        control_seed_s = float(control._jet_arming_last_transition_s)
        handoff_transition_s = float(unbroken._jet_arming_last_transition_s)

        after_unbroken = _ja8_advance(unbroken)
        after_resumed = _ja8_advance(resumed)
        after_control = _ja8_advance(control)

    carried = resumed_latch == handoff_latch
    continues = (
        np.array_equal(
            _ja_bits(after_unbroken["y"]), _ja_bits(after_resumed["y"])
        )
        and after_resumed["Ts_K"] == after_unbroken["Ts_K"]
        and after_resumed["ledger"] == after_unbroken["ledger"]
        and after_resumed["censored_entry"] == after_unbroken["censored_entry"]
        and after_resumed["latch"] == after_unbroken["latch"]
    )
    # The control must come up DISARMED with an empty census -- the
    # constructor's seed -- and its continuation must then leave the unbroken
    # trajectory. Both halves are required: a control that diverged while
    # still resuming armed would be measuring something else.
    control_disarmed = control_latch[:3] == (False, 0, 0)
    control_seed_nan = not np.isfinite(control_seed_s)
    control_diverges = not np.array_equal(
        _ja_bits(after_unbroken["y"]), _ja_bits(after_control["y"])
    )
    # Name the MECHANISM of the divergence rather than only its existence: the
    # control has to actually enter a step censored, and the two carried legs
    # have to enter none. Without this the equality above could be satisfied
    # by both legs censoring everything.
    control_censors = after_control["censored_entry"] > 0
    carried_legs_launch = (
        after_unbroken["censored_entry"] == 0
        and after_resumed["censored_entry"] == 0
    )
    ok = (
        census_live
        and carried
        and continues
        and carried_legs_launch
        and len(removed) == len(JA_RESTART_LATCH_ATTRS)
        and warned
        and control_disarmed
        and control_seed_nan
        and control_diverges
        and control_censors
    )
    return (
        "JA8 arming criterion: the latch and its census survive a restart, "
        "and a record without them resumes disarmed and diverges",
        ok,
        f"handoff after {JA_FLUID_STEPS} accepted steps at "
        f"arm={fmt(JA_IMMEDIATE_ARM_A)} A; latch ARMED with a nonzero census: "
        f"{census_live} -- armed={handoff_latch[0]}, censored steps "
        f"{handoff_latch[1]}, transitions {handoff_latch[2]}, last transition "
        f"{fmt(handoff_transition_s)} s\n        "
        f"CARRIED: all {len(JA_RESTART_LATCH_ATTRS)} members "
        f"({', '.join(JA_RESTART_LATCH_ATTRS)}) bit-identical in the resumed "
        f"solver: {carried}\n        "
        f"CONTINUATION over {JA_RESTART_STEPS} further steps, unbroken vs "
        f"restarted: state bit-identical, ledger, T_s "
        f"{fmt(after_resumed['Ts_K'])} K vs {fmt(after_unbroken['Ts_K'])} K, "
        f"steps entered CENSORED {after_resumed['censored_entry']} vs "
        f"{after_unbroken['censored_entry']} (0 required on both -- both "
        f"launch on every step: {carried_legs_launch}), census "
        f"{after_resumed['latch'][:3]} vs {after_unbroken['latch'][:3]} -- "
        f"all equal: {continues}\n        "
        f"NEGATIVE CONTROL, the same payload with its {len(removed)} latch "
        f"rows deleted ({', '.join(removed)}) -- the shape a record written "
        f"before the carriage has: it LOADS rather than raising, warns "
        f"({warned}), and comes up disarmed with an empty census "
        f"({control_disarmed and control_seed_nan}: armed="
        f"{control_latch[0]}, censored {control_latch[1]}, transitions "
        f"{control_latch[2]}, last transition {fmt(control_seed_s)} s)\n        "
        f"and its continuation DIVERGES from the unbroken run "
        f"({control_diverges}) -- it re-censors the jet it should have "
        f"resumed launching, entering {after_control['censored_entry']} of "
        f"{JA_RESTART_STEPS} steps censored against the unbroken run's "
        f"{after_unbroken['censored_entry']} ({control_censors}), and re-arms at "
        f"{fmt(control._jet_arming_last_transition_s)} s instead of holding "
        f"the handoff's {fmt(handoff_transition_s)} s. That divergence is the "
        f"measure of what the carriage is worth",
    )


# --------------------------------- B4 anode-side energetic recycle (jet)


#: [B4] The reflection coefficients under test. They MIRROR the fluid
#: channel's shipped ``anode_jet_R_N`` / ``anode_jet_R_E``, which is what makes
#: the two arms statements about the same surface (He -> Mo).
AJ_R_N = 0.63
AJ_R_E = 0.41

#: [B4] The velocity grid these gates run on, for the reason ``CJ_NVZ`` states:
#: a directed beam smeared onto the coarse ``(16, 6)`` axis carries more
#: thermal energy in its narrowest resolvable spectrum than the beam has, and
#: the engine's guard says so and RAISES. Measured on the shipped ``(48, 12)``
#: axis the grid-tied smear leaves the margin ``e / (3/2 k T_launch)`` between
#: 3.1 and 5.4 over ``0.005-100 eV`` (``scripts/b4aj_smear_margin_probe.py``)
#: -- the stretched axis makes it scale-free -- so the projection lands at
#: machine precision at every launch energy this channel can reach.
AJ_NVZ = 48
AJ_NVP = 12

#: [B4] Incident energy per collected ion [eV], ``phi_a + Ti``, at the mesh's
#: LOW-z flanking cell. The high-z side is fed at half of it (``aj_feed``), so
#: the two sides carry DIFFERENT per-atom launch energies and the aggregate
#: identities below are not a single-cell special case.
AJ_PHI_TI_EV = 40.0

#: [B4] Counted anode collection over one tick [particles], split across the
#: two flanking cells by ``AJ_SHARES``.
AJ_COUNT = 5.0e16

#: [B4] How ``aj_feed`` splits the counted stream across the mesh's low-z and
#: high-z flanking cells. Deliberately unequal: an equal split would let a
#: sign error in one direction cancel against the other.
AJ_SHARES = (0.6, 0.4)

#: [B4] Registered ceiling on the cross-book |booked - debited| / debited.
#: The two are the same committed number by construction, so this is a
#: roundoff budget, not a physics tolerance.
AJ_CROSS_BOOK_REL = 1.0e-8

#: [B4] Registered ceiling on the analytic-vs-discrete faithfulness number of
#: the launch spectrum: its DISCRETE mean energy against the
#: ``(1/2) m v_back^2`` it is built to carry. It is the convergence tolerance
#: of the moment compensation, and a solve that misses it raises.
AJ_MOMENT_REL = 1.0e-10

#: [B4] Registered ceiling on the two MOMENTUM statements of AJ4: the row
#: against its independent rebuild, and the mirror antisymmetry of the mesh
#: interception. Both are the same arithmetic reached by two routes, so this
#: is a roundoff budget.
AJ_MOMENTUM_REL = 1.0e-12


def aj_spec(R_N=AJ_R_N, R_E=AJ_R_E, T_launch_eV=None):
    return {"R_N": R_N, "R_E": R_E, "T_launch_eV": T_launch_eV}


def aj_closed_box(nz=8, jet=True, cls=None, transparency=1.0):
    """Return the closed box of :func:`closed_box_dvm` with the anode jet armed.

    Same disarmed box as :func:`cj_closed_box` -- no annulus (``Rm == Rp``), no
    pumping, specular end walls -- with ONE addition the anode channel cannot
    do without: a mesh face, because the launch DIRECTION is defined against
    it. The mesh ships fully TRANSPARENT here (``transparency = 1.0``), so it
    intercepts nothing and the box stays disarmed apart from the channel under
    test; :func:`aj_mesh_box` is where the interception is armed.
    """
    builder = TransientDVM if cls is None else cls
    dvm = builder(
        geometry=uniform_tube(nz, Rp=15.0, Rm=15.0),
        nvz=AJ_NVZ,
        nvp=AJ_NVP,
        accommodation=0.0,
        exchange_model=EXCHANGE_MODEL,
        s_L=0.0,
        s_R=0.0,
        mesh_face=nz // 2,
        transparency=transparency,
        anode_jet=aj_spec() if jet else None,
    )
    dvm.seed_from_density(np.full(nz, 1.0e13), np.zeros(nz), T_K=400.0)
    return dvm


def aj_feed(dvm, count=AJ_COUNT, per_ion_eV=AJ_PHI_TI_EV):
    """Return ``(source_counts, incident_erg)`` for one counted anode tick.

    Both flanking cells of the mesh face are fed, at unequal shares and at
    unequal per-ion incident energies (the high-z side at half the low-z
    side's), so both launch directions are live and the two sides cannot
    cancel each other's errors.
    """
    face = int(dvm.mesh_face)
    rows = {"anode": np.zeros(dvm.nz)}
    incident = np.zeros(dvm.nz)
    for cell, share, energy in (
        (face - 1, AJ_SHARES[0], per_ion_eV),
        (face, AJ_SHARES[1], 0.5 * per_ion_eV),
    ):
        rows["anode"][cell] = share * count
        incident[cell] = share * count * energy * EV
    return rows, incident


def aj_mesh_box(nz=12, cls=None, side="low"):
    """Return an OPAQUE-mesh column box seeded on one side of the mesh.

    Column-only (``Rm == Rp``), collisionless (the caller hands
    :func:`zero_plasma`, so ``n_i = 0`` kills CX and elastic and ``nu_ion = 0``
    kills ionization), no external sources, specular end walls, and a mesh at
    the centre face that stops everything reaching it
    (``transparency = 0.0``). The only channel with any axial momentum in it is
    therefore the mesh interception, which is what AJ4's second statement is
    about.

    ``side`` seeds the column in the cells BELOW the mesh face (``"low"``) or
    in their exact mirror image above it (``"high"``). The box is uniform and
    the mesh sits at its centre, so the two are reflections of one another and
    the intercepted axial momentum must be equal and OPPOSITE.
    """
    face = nz // 2
    fill = np.zeros(nz)
    if side == "low":
        fill[:face] = 1.0e13
    else:
        fill[face:] = 1.0e13
    builder = TransientDVM if cls is None else cls
    dvm = builder(
        geometry=uniform_tube(nz, Rp=15.0, Rm=15.0),
        nvz=AJ_NVZ,
        nvp=AJ_NVP,
        accommodation=0.0,
        exchange_model=EXCHANGE_MODEL,
        s_L=0.0,
        s_R=0.0,
        mesh_face=face,
        transparency=0.0,
        anode_jet=aj_spec(),
    )
    dvm.seed_from_density(fill, np.zeros(nz), T_K=400.0)
    return dvm


class _AJUnbookedEnergy(TransientDVM):
    """Harness defect: the jet births atoms and forgets their energy row.

    The AJ1 twin of :class:`_CJUnbookedEnergy`: a channel added to the
    distribution without an entry in the energy ledger. Every particle count is
    untouched, so the PARTICLE ledger closes exactly as it did; only the energy
    identity sees it.
    """

    def _book_energy_ledger(self, **kwargs):
        kwargs["e_birth_anode_jet"] = 0.0
        return TransientDVM._book_energy_ledger(self, **kwargs)


class _AJWallSpectrum(TransientDVM):
    """Harness defect: the jet launches on the 300 K cosine-wall spectrum.

    The B0a-class defect, anode side -- the counted particles arrive, the
    ledger closes against itself, and the energy the anode book was debited is
    simply not the energy the gas received.
    """

    def _anode_jet_launch_spectrum(self, e_launch, cell, direction):
        return self.M_wall


class _AJOneSidedLaunch(TransientDVM):
    """Harness defect: both sides of the mesh launch into ``+z``.

    The defect the SIGNED momentum row exists to catch. Every particle count,
    every energy row and both ledger residuals are untouched -- a launch
    spectrum's energy does not know its sign -- so only ``momentum_anode_jet``
    sees it.
    """

    def _anode_jet_launch_spectrum(self, e_launch, cell, direction):
        return TransientDVM._anode_jet_launch_spectrum(
            self, e_launch, cell, 1.0
        )


def aj_feed_mixed(dvm, count=AJ_COUNT, per_ion_eV=AJ_PHI_TI_EV):
    """Return ``(source_counts, incident_erg)`` with BOTH branches live.

    Four fed cells, two on each side of the mesh face, and on each side one
    carries a positive per-ion incident energy and one carries exactly zero:

    ==========  =========================  ================================
    cell        side                       branch
    ==========  =========================  ================================
    ``face-2``  low-z  (launch ``-z``)     ZERO incident -> wholly thermal
    ``face-1``  low-z  (launch ``-z``)     positive      -> splits ``R_N``
    ``face``    high-z (launch ``+z``)     positive      -> splits ``R_N``
    ``face+1``  high-z (launch ``+z``)     ZERO incident -> wholly thermal
    ==========  =========================  ================================

    The shares are unequal for the reason :func:`aj_feed`'s are, and the two
    zero cells sit on OPPOSITE sides so a per-side rather than per-cell
    implementation of the rule would be caught.
    """
    face = int(dvm.mesh_face)
    rows = {"anode": np.zeros(dvm.nz)}
    incident = np.zeros(dvm.nz)
    plan = (
        (face - 2, 0.30, 0.0),
        (face - 1, 0.25, per_ion_eV),
        (face, 0.20, 0.5 * per_ion_eV),
        (face + 1, 0.25, 0.0),
    )
    for cell, share, energy in plan:
        rows["anode"][cell] = share * count
        incident[cell] = share * count * energy * EV
    return rows, incident


class _AJZeroThroughJet(TransientDVM):
    """Harness defect: send the ZERO-incident cells down the jet path anyway.

    The pre-ruling behaviour, reinstated as AJ6's negative control: split
    ``R_N`` off every fed cell regardless of whether it carries any committed
    incident energy. It is the exact arithmetic this member ran before the
    fluid-parity ruling, restored by dropping ONE mask, so the control tests
    the ruling itself rather than a strawman.
    """

    def _split_anode_recycle(self, anode, incident_erg):
        thermal, jet, jet_energy = TransientDVM._split_anode_recycle(
            self, anode, incident_erg
        )
        if jet is None:
            return thermal, jet, jet_energy
        counts = np.asarray(anode, dtype=float)
        unmasked = float(self.anode_jet["R_N"]) * counts
        return counts - unmasked, unmasked, jet_energy


class _AJMeshUnsignedMomentum(TransientDVM):
    """Harness defect: the mesh tallies ``|v_z|`` instead of the signed ``v_z``.

    The mirror-antisymmetry statement's negative control. The intercepted
    COUNT and its energy are unchanged, so the ledgers close; what breaks is
    that a mirrored pair of runs no longer cancels.
    """

    def _mesh_axial_momentum_weight(self):
        return M_HE * np.abs(self.g.VZ)


def gate_aj1():
    """Anode jet, closed box: both ledgers close and the split conserves.

    STATEMENT 1 of the B4 four, and the AJ twin of CJ1. In the disarmed box
    the counted anode collection is the only external channel, so what the
    split does is visible on its own: the ``R_N`` share becomes a directed
    energetic volume birth on the side it was collected from and the remainder
    the thermal ``M_wall`` rebirth, the two counts sum to the handed count
    EXACTLY, and both the particle and the energy ledger close at the tolerance
    I1/I2/I6 hold the arm to.

    Non-vacuity is asserted rather than assumed: the jet's energy row must
    carry a real share of the tick's energy throughput, or the identity under
    test is 0 == 0.

    NEGATIVE CONTROL: birth the same atoms and book their energy row at zero
    (:class:`_AJUnbookedEnergy`). The particle ledger closes to the same
    roundoff and the ENERGY distribution residual goes to order one, so the
    control fails at THIS statement.
    """
    nz = 8
    worst_part = 0.0
    worst_ener = 0.0
    worst_split = 0.0
    live = float("inf")
    dvm = aj_closed_box(nz)
    plasma = geometry_plasma(nz)
    for _ in range(5):
        rows, incident = aj_feed(dvm)
        led = dvm.update(
            CADENCE_S,
            source_counts=rows,
            anode_jet_incident_erg=incident,
            **plasma,
        )
        p = ledger_residual(led)
        e = ledger_energy_residual(led)
        worst_part = max(
            worst_part, abs(p["distribution_rel"]), abs(p["domain_rel"])
        )
        worst_ener = max(
            worst_ener, abs(e["distribution_rel"]), abs(e["domain_rel"])
        )
        split = led["birth_anode_jet"] + led["birth_anode"]
        worst_split = max(worst_split, abs(split - AJ_COUNT) / AJ_COUNT)
        live = min(
            live, abs(led["energy"]["birth_anode_jet"]) / e["scale"]
        )

    control = aj_closed_box(nz, cls=_AJUnbookedEnergy)
    control_part = 0.0
    control_ener = 0.0
    for _ in range(5):
        rows, incident = aj_feed(control)
        led = control.update(
            CADENCE_S,
            source_counts=rows,
            anode_jet_incident_erg=incident,
            **plasma,
        )
        control_part = max(
            control_part, abs(ledger_residual(led)["distribution_rel"])
        )
        control_ener = max(
            control_ener,
            abs(ledger_energy_residual(led)["distribution_rel"]),
        )

    ok = (
        worst_part < ROUNDOFF_REL
        and worst_ener < ROUNDOFF_REL
        and worst_split < ROUNDOFF_REL
        and live > 1.0e-6
        and control_part < ROUNDOFF_REL
        and control_ener > 1.0e-3
    )
    return (
        "AJ1 anode jet, closed box: particle and energy ledgers close, the "
        "anode split conserves the counted stream",
        ok,
        f"5 ticks at {fmt(AJ_COUNT)} counted particles across both flanking "
        f"cells ({AJ_SHARES[0]} / {AJ_SHARES[1]}), {AJ_PHI_TI_EV} / "
        f"{0.5 * AJ_PHI_TI_EV} eV incident per ion, R_N={AJ_R_N} "
        f"R_E={AJ_R_E}: worst particle residual {fmt(worst_part)}, worst "
        f"energy residual {fmt(worst_ener)}, |jet + thermal - handed| / "
        f"handed {fmt(worst_split)} (tol {fmt(ROUNDOFF_REL)}); jet share of "
        f"the energy throughput {fmt(live)} (> 1e-6 required)\n        "
        f"NEGATIVE CONTROL (birth the atoms, book their energy row at zero): "
        f"particle residual {fmt(control_part)} -- unchanged, as the defect "
        f"moves no particle -- and energy distribution residual "
        f"{fmt(control_ener)}, which is the statement failing"
    )


def gate_aj2():
    """In-solver: one committed pair feeds both the birth and the anode book.

    STATEMENT 2 of the B4 four. On the engaged arm with the anode jet armed,
    per tick and cumulatively over the window:

    * every counted anode particle is injected -- ``birth_anode_jet +
      birth_anode`` equals the count the tick was handed;
    * the birth's ENERGY row is the count times the discrete mean of the
      spectra that were placed, re-derived here from the ledger rather than
      read back from the engine. Summed over the two sides that is
      ``(R_E/R_N) * (incident / count)`` per atom exactly, whatever the two
      sides' individual launch energies are;
    * the CUMULATIVE anode-book ``backscatter`` row -- booked per ACCEPTED
      step -- equals the cumulative ``birth_anode_jet`` energy the ticks
      handed the gas, plus ``R_E`` times the accumulator the next tick has not
      yet been given.

    NEGATIVE CONTROL: form the same debit from a TICK-TIME reading applied to
    the whole window's count -- the sampling the stage-weighted accumulator
    exists to avoid.
    """
    sim = make_sim(
        nx=24,
        neutral_kinetic_dvm_anode_jet=True,
        neutral_kinetic_dvm_nvz=AJ_NVZ,
        neutral_kinetic_dvm_nvp=AJ_NVP,
    )
    handed = []
    update = TransientDVM.update

    def spy_update(self, dt, **kwargs):
        rows = (kwargs.get("source_counts") or {}).get("anode")
        incident = kwargs.get("anode_jet_incident_erg")
        record = {
            "count": float(np.sum(rows)) if rows is not None else 0.0,
            "incident": (
                float(np.sum(incident)) if incident is not None else 0.0
            ),
            "phi_ti_tick_eV": float(
                np.max(np.asarray(kwargs["Ti_eV"], dtype=float))
            ),
        }
        led = update(self, dt, **kwargs)
        record["led"] = dict(led)
        handed.append(record)
        return led

    TransientDVM.update = spy_update
    try:
        run_until_updates(sim, 4)
        # A few more accepted steps, so the window ends BETWEEN ticks and the
        # "what the next tick is still owed" term is non-zero rather than
        # trivially satisfied.
        for _ in range(5):
            advance_one_step(sim)
    finally:
        TransientDVM.update = update

    worst_count = 0.0
    worst_row = 0.0
    births = 0.0
    for rec in handed:
        led = rec["led"]
        injected = led["birth_anode_jet"] + led["birth_anode"]
        worst_count = max(
            worst_count,
            abs(injected - rec["count"]) / max(rec["count"], 1e-300),
        )
        row = led["energy"]["birth_anode_jet"]
        births += row
        if led["birth_anode_jet"] > 0.0:
            per_atom = row / led["birth_anode_jet"]
            target = (AJ_R_E / AJ_R_N) * rec["incident"] / rec["count"]
            worst_row = max(worst_row, abs(per_atom - target) / target)

    debited_erg = sim._anode_energy_ledger_J["backscatter"] * 1.0e7
    incident_erg = sim._anode_energy_ledger_J["ion_incident"] * 1.0e7
    outstanding = AJ_R_E * float(np.sum(sim._dvm_anode_jet_energy_booked))
    identity = abs(debited_erg - (births + outstanding)) / max(
        births + outstanding, 1e-300
    )
    # The book's own internal statement: the backscatter row is exactly R_E of
    # the incident row, so "the anode absorbs 1 - R_E" is a subtraction of two
    # measured rows rather than a remembered convention.
    share = (
        abs(debited_erg - AJ_R_E * incident_erg)
        / max(AJ_R_E * incident_erg, 1e-300)
    )

    control_debit = sum(
        AJ_R_E * rec["count"] * rec["phi_ti_tick_eV"] * EV for rec in handed
    )
    control_rel = abs(control_debit - births) / max(births, 1e-300)

    ok = (
        len(handed) >= 4
        and births > 0.0
        and outstanding > 0.0
        and worst_count < ROUNDOFF_REL
        and worst_row < AJ_CROSS_BOOK_REL
        and identity < ROUNDOFF_REL
        and share < ROUNDOFF_REL
        and control_rel > 1.0e-3
    )
    return (
        "AJ2 in-solver anode jet: injected == counted, and the cumulative "
        "anode-book debit == the cumulative birth energy + what is still owed",
        ok,
        f"{len(handed)} ticks, {fmt(births)} erg of backscatter born; "
        f"worst |injected - counted| / counted {fmt(worst_count)} "
        f"(tol {fmt(ROUNDOFF_REL)}); worst per-atom launch energy against "
        f"(R_E/R_N)(phi_a + Ti) {fmt(worst_row)} "
        f"(tol {fmt(AJ_CROSS_BOOK_REL)})\n        "
        f"anode ledger backscatter row {fmt(debited_erg)} erg vs births "
        f"{fmt(births)} + outstanding {fmt(outstanding)} erg: relative "
        f"{fmt(identity)} (tol {fmt(ROUNDOFF_REL)}); ion_incident row "
        f"{fmt(incident_erg)} erg, backscatter / (R_E ion_incident) - 1 = "
        f"{fmt(share)}\n        "
        f"NEGATIVE CONTROL (tick-time (phi_a + Ti) on the window count): "
        f"debit {fmt(control_debit)} erg, {fmt(control_rel)} relative from "
        f"the counted one -- the sampling error the stage accumulator removes"
    )


def gate_aj3():
    """Every channel armed: the gas receives what the anode book was debited.

    STATEMENT 3 of the B4 four, and the one the B0a standing lesson makes
    binding: a backscatter launched at the WRONG energy moves exactly the right
    number of particles, so statements 1 and 2 close to the same roundoff
    whether the spectrum is right or wrong. Only a cross-book between the two
    ledgers sees it.

    With the cylindrical wall, an intercepting anode mesh, pumping at both
    ends, a puff, volume recombination and both recycle faces all live at once,
    and once per annulus treatment:

    * ``count * <E>_spectrum`` -- the ledger's ``birth_anode_jet`` energy row
      -- against ``R_E`` times the incident energy the anode book was debited
      by, at ``AJ_CROSS_BOOK_REL``;
    * the analytic-vs-discrete faithfulness number: the same spectrum's
      discrete mean energy against the ``(1/2) m v_back^2`` it is built to
      carry, at ``AJ_MOMENT_REL``, in BOTH launch directions.

    NEGATIVE CONTROL: launch the same counted atoms on the 300 K cosine-wall
    spectrum (:class:`_AJWallSpectrum`). BOTH normalizations are reported,
    because which one a gate reads decides what it can catch: the ROW-RELATIVE
    form is what this gate tests, and the THROUGHPUT-NORMALIZED form is the one
    a ledger-residual gate would have seen.
    """
    nz = 12
    results = {}
    for flights in ("rates", "bounded_chord"):
        for wrong in (False, True):
            cls = _AJWallSpectrum if wrong else TransientDVM
            dvm = cls(
                geometry=uniform_tube(nz),
                nvz=AJ_NVZ,
                nvp=AJ_NVP,
                s_L=0.3,
                s_R=0.3,
                accommodation=0.4,
                exchange_model=EXCHANGE_MODEL,
                annulus_flights=flights,
                mesh_face=nz // 2,
                transparency=0.642,
                anode_jet=aj_spec(),
            )
            dvm.seed_from_density(np.full(nz, 1.0e13), np.full(nz, 1.0e13))
            sources = {
                "recombination": np.full(nz, 1.0e15),
                "puff": np.zeros(nz),
                "cathode_face": np.zeros(nz),
                "collector_face": np.zeros(nz),
            }
            sources["puff"][3] = 3.0e17
            sources["cathode_face"][0] = 2.0e16
            sources["collector_face"][-1] = 4.0e16
            plasma = geometry_plasma(nz)
            cross = 0.0
            moment = 0.0
            throughput = 0.0
            resid = 0.0
            for _ in range(4):
                rows, incident = aj_feed(dvm)
                led = dvm.update(
                    CADENCE_S,
                    sources=sources,
                    source_counts=rows,
                    anode_jet_incident_erg=incident,
                    **plasma,
                )
                row = led["energy"]["birth_anode_jet"]
                debited = AJ_R_E * float(np.sum(incident))
                cross = max(cross, abs(row - debited) / debited)
                e = ledger_energy_residual(led)
                throughput = max(
                    throughput, abs(row - debited) / e["scale"]
                )
                resid = max(
                    resid,
                    abs(e["distribution_rel"]),
                    abs(e["domain_rel"]),
                )
                # The faithfulness number, rebuilt independently of the
                # engine's own booking, in BOTH launch directions.
                for direction, per_ion in (
                    (-1.0, AJ_PHI_TI_EV), (1.0, 0.5 * AJ_PHI_TI_EV)
                ):
                    e_launch = (AJ_R_E / AJ_R_N) * per_ion * EV
                    spec = TransientDVM._anode_jet_launch_spectrum(
                        dvm, e_launch, int(dvm.mesh_face), direction
                    )
                    got = 0.5 * M_HE * float((spec * dvm.g.V2).sum())
                    moment = max(moment, abs(got - e_launch) / e_launch)
            results[(flights, wrong)] = (cross, moment, throughput, resid)

    good = [results[(f, False)] for f in ("rates", "bounded_chord")]
    bad = [results[(f, True)] for f in ("rates", "bounded_chord")]
    ok = (
        all(c < AJ_CROSS_BOOK_REL for c, _m, _t, _r in good)
        and all(m < AJ_MOMENT_REL for _c, m, _t, _r in good)
        and all(r < ROUNDOFF_REL for _c, _m, _t, r in good)
        and all(c > 0.5 for c, _m, _t, _r in bad)
        and all(r < ROUNDOFF_REL for _c, _m, _t, r in bad)
    )
    detail = (
        f"4 ticks per arm, every channel armed, {AJ_PHI_TI_EV} / "
        f"{0.5 * AJ_PHI_TI_EV} eV incident per ion (launch "
        f"{(AJ_R_E / AJ_R_N) * AJ_PHI_TI_EV:.4g} / "
        f"{(AJ_R_E / AJ_R_N) * 0.5 * AJ_PHI_TI_EV:.4g} eV per atom)"
    )
    for flights in ("rates", "bounded_chord"):
        c, m, _t, r = results[(flights, False)]
        detail += (
            f"\n        {flights}: cross-book |row - R_E E_incident| / "
            f"(R_E E_incident) {fmt(c)} (tol {fmt(AJ_CROSS_BOOK_REL)}); "
            f"analytic-vs-discrete launch energy {fmt(m)} "
            f"(tol {fmt(AJ_MOMENT_REL)}); energy-ledger residual {fmt(r)}"
        )
    for flights in ("rates", "bounded_chord"):
        c, _m, t, r = results[(flights, True)]
        detail += (
            f"\n        NEGATIVE CONTROL [{flights}] (launch on the 300 K "
            f"cosine-wall spectrum): row-relative {fmt(c)} -- detected -- "
            f"while throughput-normalized it is {fmt(t)} and the energy "
            f"ledger still closes at {fmt(r)}, so only the row-relative "
            f"cross-book catches it"
        )
    return (
        "AJ3 anode jet, every channel armed: the booked launch energy IS the "
        "debited anode-book energy",
        ok,
        detail,
    )


def gate_aj4():
    """The two presence-gated momentum rows measure what they claim to.

    STATEMENT 4 of the B4 four -- the plan's "launch momentum booked against
    the surface / wire-intercepted momentum on the structure", made
    MEASURABLE. Two rows, two independent statements:

    ``momentum_anode_jet`` is rebuilt from the discrete launch spectra: for
    each fed cell, ``m * <v_z>_spectrum * count`` with the direction the cell's
    side implies, summed. The two sides launch AWAY from the mesh, so the
    signed sum is a difference of two positive magnitudes and a dropped or
    mis-signed side cannot hide in it. NEGATIVE CONTROL
    (:class:`_AJOneSidedLaunch`): launch both sides into ``+z``. Not one
    particle count, energy row or ledger residual moves -- a spectrum's energy
    does not know its sign -- and only this row sees it.

    ``momentum_mesh_absorbed`` is held to MIRROR ANTISYMMETRY, which needs no
    second implementation of the tally: a uniform collisionless column box with
    an opaque mesh at its centre, seeded only below the mesh, and the same box
    seeded in the exact mirror image above it. The atoms reaching the mesh in
    the first carry ``+z`` momentum and in the second ``-z``, in equal
    magnitude, so the two rows must sum to zero. Non-vacuity is asserted: the
    mesh must actually have intercepted particles. NEGATIVE CONTROL
    (:class:`_AJMeshUnsignedMomentum`): tally ``|v_z|`` instead of the signed
    ``v_z`` -- the intercepted COUNT and its energy are untouched, both ledgers
    close, and the mirrored pair stops cancelling.

    The transparent-mesh box of AJ1 is checked too: a mesh that intercepts
    nothing absorbs exactly zero momentum.
    """
    nz = 8
    dvm = aj_closed_box(nz)
    plasma = geometry_plasma(nz)
    worst_launch = 0.0
    worst_transparent = 0.0
    launched = 0.0
    for _ in range(3):
        rows, incident = aj_feed(dvm)
        led = dvm.update(
            CADENCE_S,
            source_counts=rows,
            anode_jet_incident_erg=incident,
            **plasma,
        )
        expected = 0.0
        for cell in np.flatnonzero(rows["anode"]):
            count = AJ_R_N * float(rows["anode"][cell])
            e_launch = AJ_R_E * float(incident[cell]) / count
            direction = -1.0 if int(cell) < int(dvm.mesh_face) else 1.0
            spec = TransientDVM._anode_jet_launch_spectrum(
                dvm, e_launch, int(cell), direction
            )
            expected += count * M_HE * float((spec * dvm.g.VZ).sum())
        got = led["momentum_anode_jet"]
        launched = max(launched, abs(expected))
        worst_launch = max(
            worst_launch, abs(got - expected) / max(abs(expected), 1e-300)
        )
        worst_transparent = max(
            worst_transparent, abs(led["momentum_mesh_absorbed"])
        )

    control = aj_closed_box(nz, cls=_AJOneSidedLaunch)
    control_launch = 0.0
    control_counts = True
    for _ in range(3):
        rows, incident = aj_feed(control)
        led = control.update(
            CADENCE_S,
            source_counts=rows,
            anode_jet_incident_erg=incident,
            **plasma,
        )
        expected = 0.0
        for cell in np.flatnonzero(rows["anode"]):
            count = AJ_R_N * float(rows["anode"][cell])
            e_launch = AJ_R_E * float(incident[cell]) / count
            direction = -1.0 if int(cell) < int(control.mesh_face) else 1.0
            spec = TransientDVM._anode_jet_launch_spectrum(
                control, e_launch, int(cell), direction
            )
            expected += count * M_HE * float((spec * control.g.VZ).sum())
        control_launch = max(
            control_launch,
            abs(led["momentum_anode_jet"] - expected)
            / max(abs(expected), 1e-300),
        )
        control_counts = control_counts and (
            abs(led["birth_anode_jet"] - AJ_R_N * AJ_COUNT)
            <= ROUNDOFF_REL * AJ_R_N * AJ_COUNT
        )

    # --- the mesh row: mirror antisymmetry of a one-sided seed
    def mesh_pair(cls=None):
        rows = []
        for side in ("low", "high"):
            box = aj_mesh_box(cls=cls, side=side)
            zero = np.zeros(box.nz)
            led = box.update(
                CADENCE_S,
                source_counts={"anode": zero.copy()},
                anode_jet_incident_erg=zero.copy(),
                **zero_plasma(box),
            )
            rows.append(
                (
                    led["momentum_mesh_absorbed"],
                    led["loss_mesh_blocked"],
                    ledger_residual(led),
                    ledger_energy_residual(led),
                )
            )
        return rows

    (p_low, n_low, r_low, e_low), (p_high, n_high, r_high, e_high) = mesh_pair()
    scale = max(abs(p_low), abs(p_high), 1e-300)
    mirror = abs(p_low + p_high) / scale
    (c_low, _cn_low, _cr_low, _ce_low), (c_high, _cn_high, _cr_high, _ce_high) = (
        mesh_pair(cls=_AJMeshUnsignedMomentum)
    )
    control_mirror = abs(c_low + c_high) / max(abs(c_low), abs(c_high), 1e-300)

    ok = (
        launched > 0.0
        and worst_launch < AJ_MOMENTUM_REL
        and worst_transparent == 0.0
        and control_counts
        and control_launch > 0.5
        and n_low > 0.0
        and n_high > 0.0
        and p_low > 0.0
        and p_high < 0.0
        and mirror < AJ_MOMENTUM_REL
        and abs(r_low["distribution_rel"]) < ROUNDOFF_REL
        and abs(e_low["distribution_rel"]) < ROUNDOFF_REL
        and control_mirror > 0.5
    )
    return (
        "AJ4 the anode jet's two momentum rows: the launched momentum is the "
        "signed sum of the placed spectra, the mesh absorbs the intercepted "
        "axial momentum",
        ok,
        f"3 ticks, {fmt(launched)} g cm/s launched: worst "
        f"|row - rebuilt| / |rebuilt| {fmt(worst_launch)} "
        f"(tol {fmt(AJ_MOMENTUM_REL)}); transparent mesh absorbs "
        f"{fmt(worst_transparent)} (exactly 0 required)\n        "
        f"NEGATIVE CONTROL (both sides launch +z): the birth count is "
        f"unchanged ({control_counts}) and the row is {fmt(control_launch)} "
        f"relative from the signed rebuild -- only this row sees it\n        "
        f"mesh mirror pair: low-seed {fmt(p_low)} g cm/s over "
        f"{fmt(n_low)} intercepted particles, high-seed {fmt(p_high)} over "
        f"{fmt(n_high)}; |sum| / |each| {fmt(mirror)} "
        f"(tol {fmt(AJ_MOMENTUM_REL)}); particle residual "
        f"{fmt(abs(r_low['distribution_rel']))}, energy residual "
        f"{fmt(abs(e_low['distribution_rel']))}\n        "
        f"NEGATIVE CONTROL (mesh tallies |v_z|): mirror sum "
        f"{fmt(control_mirror)} relative -- the pair stops cancelling"
    )


def gate_aj6():
    """Zero incident energy is a LEGAL, BOOKED state, and it is PER CELL.

    The ruled fluid-parity closure (2026-08-30): an ion that arrives with zero
    clamped incident energy backscatters nothing, so that cell's whole counted
    stream is born thermal. Under the fluid spec the same ion gives
    ``v_back = 0``, which makes the ``R_N`` share indistinguishable from
    thermal desorption; the DVM books it as such instead of inventing energy
    the ion did not bring.

    On the closed box, fed so that BOTH branches are live in the SAME tick and
    on BOTH sides of the mesh (:func:`aj_feed_mixed`):

    * the two zero cells launch exactly nothing -- ``birth_anode_jet`` is the
      ``R_N`` share of the POSITIVE cells alone, to the bit;
    * every cell still conserves: ``thermal + jet == counted`` per cell and
      bitwise, checked on the split itself rather than on its totals, so the
      rule cannot have been implemented by moving particles between rows;
    * the cross-book holds unchanged -- the energy row is ``R_E`` times the
      counted incident energy, which the zero cells contribute nothing to;
    * both ledgers close at the tolerance I1/I2/I6 hold the arm to;
    * the momentum row carries the positive cells only, against an
      independent rebuild.

    NEGATIVE CONTROL (:class:`_AJZeroThroughJet`): drop the one mask and send
    the zero cells down the jet path, which is exactly what this member did
    before the ruling. It RAISES -- the launch guard, on a per-atom launch
    energy of ``0.0`` erg -- and the gate records that the control fails by
    raising rather than by misbooking.
    """
    nz = 8
    dvm = aj_closed_box(nz)
    plasma = geometry_plasma(nz)
    face = int(dvm.mesh_face)
    zero_cells = (face - 2, face + 1)
    live_cells = (face - 1, face)

    worst_part = 0.0
    worst_ener = 0.0
    worst_zero = 0.0
    worst_jet = 0.0
    worst_split = 0.0
    worst_cell = 0.0
    worst_cross = 0.0
    worst_mom = 0.0
    for _ in range(4):
        rows, incident = aj_feed_mixed(dvm)
        counted = rows["anode"]
        # The per-cell conservation statement, taken on the SPLIT itself.
        thermal, jet, _energy = TransientDVM._split_anode_recycle(
            dvm, counted, incident
        )
        worst_cell = max(
            worst_cell, float(np.max(np.abs((thermal + jet) - counted)))
        )
        worst_zero = max(
            worst_zero, max(abs(float(jet[c])) for c in zero_cells)
        )
        led = dvm.update(
            CADENCE_S,
            source_counts=rows,
            anode_jet_incident_erg=incident,
            **plasma,
        )
        live_only = AJ_R_N * sum(float(counted[c]) for c in live_cells)
        worst_jet = max(
            worst_jet,
            abs(led["birth_anode_jet"] - live_only) / max(live_only, 1e-300),
        )
        total = float(counted.sum())
        worst_split = max(
            worst_split,
            abs(led["birth_anode_jet"] + led["birth_anode"] - total) / total,
        )
        debited = AJ_R_E * float(np.sum(incident))
        worst_cross = max(
            worst_cross,
            abs(led["energy"]["birth_anode_jet"] - debited) / debited,
        )
        expected_p = 0.0
        for cell in live_cells:
            n_jet = AJ_R_N * float(counted[cell])
            e_launch = AJ_R_E * float(incident[cell]) / n_jet
            direction = -1.0 if cell < face else 1.0
            spec = TransientDVM._anode_jet_launch_spectrum(
                dvm, e_launch, int(cell), direction
            )
            expected_p += n_jet * M_HE * float((spec * dvm.g.VZ).sum())
        worst_mom = max(
            worst_mom,
            abs(led["momentum_anode_jet"] - expected_p)
            / max(abs(expected_p), 1e-300),
        )
        p = ledger_residual(led)
        e = ledger_energy_residual(led)
        worst_part = max(
            worst_part, abs(p["distribution_rel"]), abs(p["domain_rel"])
        )
        worst_ener = max(
            worst_ener, abs(e["distribution_rel"]), abs(e["domain_rel"])
        )

    control = aj_closed_box(nz, cls=_AJZeroThroughJet)
    control_raised = ""
    try:
        rows, incident = aj_feed_mixed(control)
        control.update(
            CADENCE_S,
            source_counts=rows,
            anode_jet_incident_erg=incident,
            **plasma,
        )
    except ValueError as exc:
        control_raised = str(exc)

    ok = (
        worst_cell == 0.0
        and worst_zero == 0.0
        and worst_jet < ROUNDOFF_REL
        and worst_split < ROUNDOFF_REL
        and worst_cross < AJ_CROSS_BOOK_REL
        and worst_mom < AJ_MOMENTUM_REL
        and worst_part < ROUNDOFF_REL
        and worst_ener < ROUNDOFF_REL
        and "positive finite launch energy" in control_raised
    )
    return (
        "AJ6 anode jet, zero incident energy: that cell launches nothing and "
        "is born wholly thermal, per cell",
        ok,
        f"4 ticks, cells {list(zero_cells)} fed at ZERO incident energy and "
        f"cells {list(live_cells)} at {AJ_PHI_TI_EV} / "
        f"{0.5 * AJ_PHI_TI_EV} eV per ion, on both sides of mesh face "
        f"{face}: jet share of the zero cells {fmt(worst_zero)} (exactly 0 "
        f"required); per-cell |thermal + jet - counted| {fmt(worst_cell)} "
        f"(exactly 0 required); birth_anode_jet against R_N x the POSITIVE "
        f"cells alone {fmt(worst_jet)}; |jet + thermal - handed| / handed "
        f"{fmt(worst_split)} (tol {fmt(ROUNDOFF_REL)})\n        "
        f"cross-book {fmt(worst_cross)} (tol {fmt(AJ_CROSS_BOOK_REL)}); "
        f"momentum row vs the positive cells' rebuild {fmt(worst_mom)} "
        f"(tol {fmt(AJ_MOMENTUM_REL)}); particle residual {fmt(worst_part)}, "
        f"energy residual {fmt(worst_ener)}\n        "
        f"NEGATIVE CONTROL (drop the mask, send the zero cells down the jet "
        f"path -- the pre-ruling arithmetic): RAISES, it does not misbook -- "
        f"{control_raised[:88]!r}"
    )


def gate_aj5():
    """The two anode backscatter re-emissions cannot be armed together.

    The G28 pairing statement, tested as its own statement rather than through
    the model-family resolver -- exactly as CJ4 is, and for the same measured
    reason. ``anode_neutral_jet`` is its own member of
    ``KINETIC_DVM_INCOMPATIBLE_DEFAULTS`` with required value ``False``, and
    the resolver runs before the guards: a config that leaves it alone has it
    CLEARED before this guard is asked, and a config that sets it ``True``
    is refused by the RESOLVER, naming the whole member set. Either way a
    refusal gate driven through ``LAPDSim1D`` would be testing the resolver.
    The guard is called directly instead, on the pair and on each half alone,
    so it stays a live statement about the PAIR if that resolver membership is
    ever relaxed.
    """
    both = {
        "neutral_kinetic_dvm_anode_jet": True,
        "anode_neutral_jet": True,
    }
    singles = (
        {"neutral_kinetic_dvm_anode_jet": True},
        {"anode_neutral_jet": True},
        {},
    )
    raised = ""
    try:
        refuse_anode_backscatter_double_book(both)
    except ValueError as exc:
        raised = str(exc)
    quiet = True
    for single in singles:
        try:
            refuse_anode_backscatter_double_book(single)
        except ValueError:
            quiet = False
    names = (
        "neutral_kinetic_dvm_anode_jet" in raised
        and "anode_neutral_jet" in raised
    )
    ok = bool(raised) and names and quiet
    return (
        "AJ5 [G28] the DVM anode jet and the fluid anode_neutral_jet refuse "
        "to arm together (both re-emit the same collected stream directed)",
        ok,
        f"pair refused naming both keys: {names}; each half alone and the "
        f"empty config raise nothing: {quiet}; message: {raised[:96]!r}",
    )


# ------------------------------------------------- B6 baffle interception
#
# The registered gates of plan row B6. The fluid books each thin annular
# baffle as a zero-thickness SERIES
# ORIFICE on the annulus conductance alone (``physics/neutrals.py``:
# ``orifice = clausing_scale * 0.25 * v_th * open_ann`` with
# ``open_ann = pi (R_clear^2 - R_col^2)``, the column untouched because
# ``R_clear >= R_p``). The DVM consumed no baffle at all, so its annulus
# streamed through the FULL annulus area where only the clear ring is open.
# B6 intercepts that flux at every baffle face and re-emits it on the side it
# was intercepted from, particle-conserving, exactly as the anode-mesh channel
# does -- the same full accommodation at ``T_wall`` on the wall spectrum, which
# is the accommodation-scope correction ruled 2026-08-28 (the scalar alpha
# covers the cylinder and the ends; mesh and closed faces run alpha = 1)
# extended to baffles.

#: Relative tolerance of the B6 FLUX statements. The transparency enters the
#: march as a per-bin scaling inside a sum, so a statement formed as
#: ``sum((1 - t) F v)`` against ``(1 - t) sum(F v)`` is a DISTRIBUTIVE
#: rearrangement of the same arithmetic and closes at roundoff, not to the bit.
#: Where the scaling is exact (``t_f = 0``, where ``1.0 * F == F``) the identity
#: IS bit-exact and the gate says so separately.
BF_FLUX_REL = 1.0e-12

#: Relative tolerance of the B6 particle-ledger closure (the registered
#: ``<= 1e-14``; the suite's general roundoff class is 1e-12).
BF_LEDGER_REL = 1.0e-14

#: Relative tolerance of the B6 momentum-row antisymmetry (AJ4's class).
BF_MOMENTUM_REL = 1.0e-12

#: Double-precision unit of least precision, for the BF3 energy bound below.
DOUBLE_EPS = float(np.finfo(float).eps)

#: [bf3] BF3's ENERGY-closure bound, in ULP of the tick's own energy SCALE:
#: the gate accepts ``|distribution residual| <= k x eps x scale``.
#:
#: DERIVATION of k = 8. The quantity BF3 bounds is a WHOLE-LEDGER roundoff
#: accumulation, and the form this replaces divided it by the BAFFLE'S OWN
#: energy row -- a single channel. Numerator and denominator were therefore
#: different objects, and the resulting headroom was set by whichever way the
#: roundoff happened to fall: measured across three DVM velocity grids the
#: residual is 0.725 / 0.021 / 0.574 ULP of the ledger scale at (16,6) --
#: the registered grid -- (48,12) and (64,24), while the old form's margin
#: against ROUNDOFF_REL swung 3.83x / 134.36x / 4.99x over the same three
#: runs. The swing is entirely the numerator's: the baffle row varies by
#: 3.3 % across the grids and the ledger scale by 0.2 %, so neither
#: denominator collapses (measured 2026-08-31; a denominator FLOOR was tried
#: first and is arithmetically inert at any value at or below the ~1.5e4 erg
#: row).
#:
#: k = 8 is the next power of two above 11x the worst measured case (0.725
#: ULP), which is the headroom this suite gives a quantity that is already at
#: the floating-point noise floor. It is expressed in ULP rather than as a
#: bare relative constant because that is what the measurement is in: saying
#: "eight units in the last place of the energy that moved" states the noise
#: floor the bound is derived from, where a fixed relative tolerance would
#: hide it.
BF_ENERGY_ULP_K = 8.0

#: [bf3] Size of the injected violation the BF3 negative control adds, in the
#: same ULP units. Comfortably outside ``BF_ENERGY_ULP_K`` from any residual
#: the gate would otherwise accept, so the control is a statement about the
#: BOUND and not about how close the live residual happens to sit to it.
BF_ENERGY_CONTROL_ULP = 32.0

#: The BF1 closed box's target transparency at its one baffle face.
BF_TARGET_TRANSPARENCY = 0.5

#: BF2's registered bound on the free-molecular-vs-discrete-grid gap, on the
#: GAS-MATCHED velocity grid. The gap itself is MEASURED and stated; this is
#: only the bound the measurement has to come in under for the plan gate to be
#: a statement about the baffle rather than about the grid.
BF_GRID_GAP_MAX = 0.05


def bf_clear_radius(Rp, Rm, transparency):
    """Return the clear radius whose open ring is ``transparency`` of the annulus.

    Inverts ``t = pi (R_clear^2 - Rp^2) / (pi (Rm^2 - Rp^2))``. The realized
    transparency is read back off the engine rather than assumed here: the
    engine forms the annulus face area from ``V_ann / dz``, which is not
    bit-identical to ``pi (Rm^2 - Rp^2)``, and every BF statement is made
    against the value the engine actually used.
    """
    return float(np.sqrt(Rp**2 + float(transparency) * (Rm**2 - Rp**2)))


def bf_box(
    nz=8,
    nvz=16,
    nvp=6,
    Rp=15.0,
    Rm=50.0,
    clear=None,
    face=None,
    through_mesh=False,
    transparency=None,
    nn_col=None,
    nn_ann=None,
    T_K=300.0,
    vmax_cm_s=None,
    annulus_flights="rates",
):
    """Return a SEALED uniform tube carrying one annular baffle at ``face``.

    ``closed_box_dvm``'s class of fixture -- specular ends
    (``accommodation = 0``), no pumping, no external source, no plasma -- but
    with a REAL ANNULUS (``Rm > Rp``), because the baffle is an annulus-only
    object and the closed box proper has ``V_ann = 0`` in every cell.

    ``through_mesh`` is the BF1 NEGATIVE CONTROL routing: the same face at the
    same transparency is armed as the ANODE MESH instead, which by design
    intercepts the COLUMN as well as the annulus. Nothing test-only is added to
    the engine for it -- the control is a second, already-shipped operator that
    makes the statement B6 is required to satisfy false.
    """
    face = nz // 2 if face is None else int(face)
    kwargs = dict(
        geometry=uniform_tube(nz, Rp=Rp, Rm=Rm),
        nvz=nvz,
        nvp=nvp,
        accommodation=0.0,
        exchange_model=EXCHANGE_MODEL,
        annulus_flights=annulus_flights,
        s_L=0.0,
        s_R=0.0,
    )
    if vmax_cm_s is not None:
        kwargs["vmax_cm_s"] = float(vmax_cm_s)
    if through_mesh:
        kwargs.update(mesh_face=face, transparency=float(transparency))
    elif clear is not None:
        kwargs.update(
            baffle_faces=(face,), baffle_clear_radius_cm=(float(clear),)
        )
    dvm = TransientDVM(**kwargs)
    col = (
        np.full(nz, 1.0e13) if nn_col is None
        else np.asarray(nn_col, dtype=float)
    )
    ann = (
        np.full(nz, 1.0e13) if nn_ann is None
        else np.asarray(nn_ann, dtype=float)
    )
    dvm.seed_from_density(col, ann, T_K=T_K)
    return dvm


def bf_tick(dvm, dt=CADENCE_S):
    """Advance one tick with every external channel silent; return the ledger."""
    return dvm.update(dt, **zero_plasma(dvm))


def bf_bits(array):
    """Return the raw-uint64 sha256 of one float64 array (bit-level identity)."""
    return hashlib.sha256(
        np.ascontiguousarray(array, dtype=np.float64).view(np.uint64).tobytes()
    ).hexdigest()


def gate_bf1():
    """BF1 closed box: the baffle throttles the ANNULUS by exactly ``t_f``.

    Engine-only, on the sealed uniform tube of :func:`bf_box` with one baffle
    at the middle face and a uniform 300 K seed in BOTH zones. Four arms are
    ticked ONCE each from the identical seed, which is what makes the
    comparison exact: the march's upstream flux ``F_a_prev`` arriving at the
    baffle face is built from cells the baffle has not touched, so on the FIRST
    tick it is bit-identical across arms whatever the transparency.

    * ``Z`` -- ``R_clear = R_col``, so ``open_ann = 0`` and ``t_f = 0``. Its
      ``loss_baffle_blocked`` row IS the full one-way incident flux at the
      face, bit-exactly (``1.0 * F == F``), and PER CELL it is the one-way
      flux of ONE direction: ``+z`` traffic is tallied on the cell left of the
      face, ``-z`` traffic on the cell right of it.
    * ``B`` -- the registered ``t_f = 0.5`` box (the realized value is read off
      the engine, not assumed).
    * ``U`` -- ``R_clear`` beyond the vessel bore, so the open ring covers the
      whole annulus: ``t_f`` clips to 1, the face is NOT armed, and the arm
      must be a bit-exact no-op against ``N``.
    * ``N`` -- no baffle configured at all.

    The four statements:

    1. the annulus flux CROSSING the face is ``t_f`` times the unbaffled flux,
       per direction: ``(incident - blocked) / incident == t_f`` at
       ``BF_FLUX_REL``, on both flanking cells;
    2. the particle ledger closes at ``BF_LEDGER_REL`` on every arm;
    3. ``loss_baffle_blocked == (1 - t_f) * incident`` -- to the BIT at
       ``t_f = 0``, and at ``BF_FLUX_REL`` at ``t_f = 0.5`` where the engine
       sums ``(1 - t_f) F`` per bin while the reference scales the summed
       incident (see :data:`BF_FLUX_REL`);
    4. the COLUMN FLUX ACROSS THE FACE is unchanged to the BIT. The quantity
       is rebuilt from the post-tick state as
       ``Phi_col = ((f_c[f-1] |v_z|)_{v_z>0} + (f_c[f] |v_z|)_{v_z<0})
       face_c[f] dt`` -- the two upstream cells of the two directions, which
       the sweep solves BEFORE it reaches the face, so an operator acting at
       the face cannot have moved them -- and arm ``B``'s value must be
       raw-identical to arm ``N``'s. Arm ``U`` is additionally raw-uint64
       identical to ``N`` in BOTH zones, which is the no-op statement.

    Statement 4 is deliberately about the FLUX AT THE FACE and not about the
    whole column field, and the difference is physics rather than pedantry:
    throttling the annulus DOES move the column, one cell later, through the
    zone-exchange coupling the 2x2 march solves -- less annulus next to a cell
    means less gas exchanged into its column. That response is the coupling
    working, and a gate asserting a bit-identical ``f_c`` everywhere would be
    asserting the coupling is broken. The per-cell column response is reported
    beside the statement so the distinction is visible rather than implied.
    (Recorded because this gate's first formulation DID assert the stronger
    property and measured it false at 1.5e+09 cm^-3 on the flanking pair; the
    registered quantity was always the flux at the face.)

    NEGATIVE CONTROL: the same face at the same transparency routed through the
    ANODE MESH, which intercepts the column too. It must block STRICTLY MORE
    than the baffle, and the excess must be EXACTLY the column share the
    baffle is required not to take -- ``(1 - t_f) Phi_col`` at
    ``BF_FLUX_REL``, rebuilt independently from the state above. That single
    number carries both halves of the statement: the mesh takes the column
    share and the baffle does not, and their ANNULUS tallies must be
    bit-identical for the difference to land on the rebuild at all, which is
    the other half of "tallied exactly as the mesh's ``mesh_a``".

    BOTH OPERATORS, ONE CHANNEL PAIR: everything above is the ``rates``
    march's. Under ``bounded_chord`` the annulus is not marched at all -- it is
    carried as flights and the baffle is an annular THROAT in the frozen flight
    map -- so the interception is a different operator, and the claim that it
    books into the SAME pair is proved on the ledger rather than asserted: the
    rows non-vacuous, equal to each other, and the ledger closed. The map's own
    routing residual (every launched particle routed exactly once, which now
    sums the baffle stops too) is checked at construction and RAISES if it
    fails, so reaching this line at all is part of the statement.

    The engine's construction-time refusal of ``R_clear < R_col`` is asserted
    here too: it is the ENGINE-side half of G32, and the only half reachable
    without the solver's geometry (which refuses the same thing first).
    """
    nz, Rp, Rm = 8, 15.0, 50.0
    face = nz // 2
    clear_b = bf_clear_radius(Rp, Rm, BF_TARGET_TRANSPARENCY)

    arm_n = bf_box(nz=nz, Rp=Rp, Rm=Rm)
    arm_u = bf_box(nz=nz, Rp=Rp, Rm=Rm, clear=1.5 * Rm, face=face)
    arm_z = bf_box(nz=nz, Rp=Rp, Rm=Rm, clear=Rp, face=face)
    arm_b = bf_box(nz=nz, Rp=Rp, Rm=Rm, clear=clear_b, face=face)
    t_f = float(arm_b.baffle_transparency[0])
    t_u = float(arm_u.baffle_transparency[0])

    led_n = bf_tick(arm_n)
    led_u = bf_tick(arm_u)
    led_z = bf_tick(arm_z)
    led_b = bf_tick(arm_b)

    incident = np.asarray(arm_z.last_baffle_counts, dtype=float)
    blocked = np.asarray(arm_b.last_baffle_counts, dtype=float)
    per_direction = []
    for cell in (face - 1, face):
        inc = float(incident[cell])
        crossing = (inc - float(blocked[cell])) / max(inc, 1e-300)
        per_direction.append((int(cell), inc, crossing, abs(crossing - t_f)))
    worst_flux = max(row[3] for row in per_direction)

    inc_total = float(incident.sum())
    blocked_total = float(blocked.sum())
    scaled = (1.0 - t_f) * inc_total
    worst_scaling = abs(blocked_total - scaled) / max(abs(scaled), 1e-300)
    exact_at_zero = inc_total == float(led_z["loss_baffle_blocked"])

    residuals = {
        name: ledger_residual(led)
        for name, led in (
            ("N", led_n), ("U", led_u), ("Z", led_z), ("B", led_b)
        )
    }
    worst_ledger = max(
        max(abs(r["distribution_rel"]), abs(r["domain_rel"]))
        for r in residuals.values()
    )

    fc_n, fa_n = bf_bits(arm_n.f_c), bf_bits(arm_n.f_a)
    noop = (
        bf_bits(arm_u.f_c) == fc_n
        and bf_bits(arm_u.f_a) == fa_n
        and t_u == 1.0
    )

    def column_face_flux(dvm):
        """Rebuild the column particle flux crossing ``face`` over one tick."""
        g = dvm.g
        total = 0.0
        for cell, sel in ((face - 1, g.vz > 0), (face, g.vz < 0)):
            total += float(
                (dvm.f_c[cell][sel] * np.abs(g.vz[sel])[:, None]).sum()
            )
        return total * float(dvm.face_c[face]) * CADENCE_S

    phi_col_n = column_face_flux(arm_n)
    phi_col_b = column_face_flux(arm_b)
    column_flux_unchanged = phi_col_b == phi_col_n
    # Reported, not gated: the column DOES respond to the annulus being
    # throttled, one cell later, through the zone-exchange coupling.
    column_response = np.abs(arm_b.f_c - arm_n.f_c).sum(axis=(1, 2))

    # NEGATIVE CONTROL: the same throttle at the same face, through the mesh.
    control = bf_box(
        nz=nz, Rp=Rp, Rm=Rm, clear=clear_b, face=face,
        through_mesh=True, transparency=t_f,
    )
    led_c = bf_tick(control)
    control_extra = float(led_c["loss_mesh_blocked"]) - blocked_total
    column_share = (1.0 - t_f) * phi_col_n
    control_rel = abs(control_extra - column_share) / max(
        abs(column_share), 1e-300
    )

    # BOTH OPERATORS, ONE CHANNEL PAIR. The statements above are the ``rates``
    # march's. Under ``bounded_chord`` the annulus is not marched at all -- it
    # is carried as flights, and the baffle is an annular THROAT in the frozen
    # flight map -- so the interception is a different operator entirely. It
    # must book into the SAME pair, and that is proved on the ledger rather
    # than asserted: the rows must be non-vacuous, equal to each other, and
    # the map's own routing-residual check (every launched particle routed
    # exactly once, which now sums the baffle stops too) must have passed at
    # construction, which it does by raising if it has not.
    jump_arm = bf_box(
        nz=nz, Rp=Rp, Rm=Rm, clear=clear_b, face=face,
        annulus_flights="bounded_chord",
    )
    led_j = bf_tick(jump_arm)
    jump_blocked = float(led_j["loss_baffle_blocked"])
    jump_paired = jump_blocked == float(led_j["birth_baffle_reemit"])
    jump_res = ledger_residual(led_j)
    jump_ledger = max(
        abs(jump_res["distribution_rel"]), abs(jump_res["domain_rel"])
    )
    jump_map_stops = sum(
        int(jump_arm.flights.baffle_src[name].size)
        for name in jump_arm.flights.baffle_src
    )

    engine_refusal = ""
    try:
        bf_box(nz=nz, Rp=Rp, Rm=Rm, clear=Rp - 1.0, face=face)
    except ValueError as exc:
        engine_refusal = str(exc)

    ok = (
        inc_total > 0.0
        and abs(t_f - BF_TARGET_TRANSPARENCY) < 1.0e-9
        and worst_flux < BF_FLUX_REL
        and worst_ledger < BF_LEDGER_REL
        and worst_scaling < BF_FLUX_REL
        and exact_at_zero
        and column_flux_unchanged
        and noop
        and control_extra > 0.0
        and control_rel < BF_FLUX_REL
        and jump_blocked > 0.0
        and jump_paired
        and jump_map_stops > 0
        and jump_ledger < BF_LEDGER_REL
        and "clear radius" in engine_refusal
    )
    detail_rows = "; ".join(
        f"cell {cell}: incident {fmt(inc)}, crossing {fmt(frac)} "
        f"(|- t_f| {fmt(dev)})"
        for cell, inc, frac, dev in per_direction
    )
    return (
        "BF1 closed box: the annular baffle throttles the ANNULUS by exactly "
        "t_f and leaves the column bit-identical",
        ok,
        f"{nz}-cell sealed tube, Rp={Rp} Rm={Rm}, one baffle at face {face}, "
        f"R_clear={clear_b:.6f} cm -> t_f={t_f:.12f}; one tick per arm from "
        f"the identical seed\n        "
        f"crossing per direction -- {detail_rows} (tol {fmt(BF_FLUX_REL)})\n"
        f"        blocked vs (1 - t_f) x incident {fmt(worst_scaling)} "
        f"relative (tol {fmt(BF_FLUX_REL)}); the same identity at t_f = 0 is "
        f"BIT-exact: {exact_at_zero}\n        "
        f"worst particle-ledger residual over the four arms "
        f"{fmt(worst_ledger)} (tol {fmt(BF_LEDGER_REL)}); the t_f = 1 arm is "
        f"a bit-exact no-op in both zones: {noop}\n        "
        f"COLUMN FLUX ACROSS THE FACE {phi_col_b!r} vs the no-baffle arm's "
        f"{phi_col_n!r}: raw-identical {column_flux_unchanged}. Reported, not "
        f"gated -- the per-cell column RESPONSE through the zone-exchange "
        f"coupling, which is the coupling working and not a leak: "
        f"{np.array2string(column_response, precision=3)}\n        "
        f"NEGATIVE CONTROL (same t_f at the same face through the anode MESH, "
        f"which blocks the column too): the mesh blocks {fmt(control_extra)} "
        f"particles MORE than the baffle, against an independent rebuild of "
        f"(1 - t_f) x the column face flux {fmt(column_share)} -- "
        f"{fmt(control_rel)} relative (tol {fmt(BF_FLUX_REL)}); the excess is "
        f"EXACTLY the column share the baffle does not take, which also "
        f"requires the two ANNULUS tallies to be bit-identical\n        "
        f"BOTH OPERATORS, ONE PAIR: under annulus_flights='bounded_chord' the "
        f"annulus is flights and the baffle is an annular THROAT in the frozen "
        f"map ({jump_map_stops} routed stop entries); it books "
        f"{fmt(jump_blocked)} particles into the SAME pair, blocked == "
        f"re-emitted: {jump_paired}, worst ledger residual "
        f"{fmt(jump_ledger)} (tol {fmt(BF_LEDGER_REL)})\n        "
        f"engine refuses R_clear < R_col naming it: {engine_refusal[:88]!r}"
    )


def gate_bf2():
    """BF2 THE PLAN GATE: the DVM's baffled annulus throughput IS the fluid's.

    The matched case the plan row registers. A sealed 300 K tube with NO plasma
    carrying the g1atrim baffle face's own radii -- ``R_clear``, the vessel
    bore ``Rm`` and the face-average column radius ``R_col``, all read from the
    committed stance geometry rather than restated here -- and a DENSITY STEP
    in the annulus across the baffle face (two reservoirs). Over the first tick,
    while the step is still sharp, the net annulus particle current across the
    face per unit density difference is compared with the fluid's own
    zero-thickness series orifice ``0.25 * v_th * open_ann`` at
    ``clausing_scale = 1``.

    The current is read off the engine's OWN face tallies rather than rebuilt:
    a ``t_f = 0`` twin's per-cell ``loss_baffle_blocked`` is the one-way
    incident flux at the face in each direction, bit-exactly, and on the first
    tick it is the flux the UNBAFFLED tube would have passed (the march's
    upstream sweep never sees the face). The baffled arm's crossing is that
    incident less its own blocked row.

    Three statements, at (48, 12) and (64, 24) bins:

    * OFF/ON STRUCTURE (exact) -- ``ratio_off == ratio_on * A_ann / open_ann``
      at ``BF_FLUX_REL``. Both ratios carry the same discrete-grid factor, so
      this isolates the geometric statement from the quadrature one: the ONLY
      thing the baffle changes is the area, by exactly ``A_ann / open_ann``.
      That factor is the plan's predicted "~1.75x", MEASURED here.
    * BAFFLE ON -- ``DVM / fluid``, expected ``~1``. The residual is the
      free-molecular-vs-discrete-grid gap: the DVM's one-way flux is a finite
      sum ``sum_bins f |v_z| A`` over a stretched grid while the fluid's is the
      continuum ``n vbar / 4``. It is MEASURED at both bin counts and on BOTH
      grid extents, and EVERY row must come in under
      :data:`BF_GRID_GAP_MAX` with refinement improving it on both.
    * TWO GRID EXTENTS, both gated. The matched extent is sized to the 300 K
      gas the box actually holds; the SHIPPED one is the production
      construction (``vmax ~ 4 v_th(Ti_cap) + 1.5 u_cap``), sized for ion drift
      caps, so a cold gas occupies a small part of it. Running both separates
      "the baffle passes the right area" from "the velocity grid resolves a
      cold Maxwellian", which are different questions; gating both says the
      answer to the second is not currently load-bearing for the first (the
      shipped rows come in with 28-57x margin on the bound).

    The fluid's own conductances at that face on the real stance geometry are
    quoted beside the bare orifice, so the reference number is on record with
    the tube conductance it sits in series with rather than only in isolation.
    """
    d, fl = arm_config(**PRODUCTION_GEOMETRY_KEYS)
    geom = build_geometry(d, fl)
    stance_face = int(
        np.asarray(geom.neutral_baffle_face_indices, dtype=int)[0]
    )
    clear = float(
        np.asarray(geom.neutral_baffle_clear_radius_cm, dtype=float)[0]
    )
    Rp_stance = np.asarray(geom.Rp_cm, dtype=float)
    Rm_stance = np.asarray(geom.Rm_cm, dtype=float)
    R_col = 0.5 * (
        float(Rp_stance[stance_face - 1]) + float(Rp_stance[stance_face])
    )
    Rm = 0.5 * (
        float(Rm_stance[stance_face - 1]) + float(Rm_stance[stance_face])
    )

    v_th = neutral_thermal_speed(Tn_K=300.0, mu_neutral=4.0)
    open_ann = np.pi * (clear**2 - R_col**2)
    orifice = 0.25 * v_th * open_ann
    fluid_two_zone = float(
        two_zone_knudsen_coefficients(geom, 300.0, 4.0)[1][stance_face - 1]
    )
    fluid_single = float(
        knudsen_flow_coefficients(geom, 300.0, 4.0)[stance_face - 1]
    )

    nz = 12
    face = nz // 2
    n_hi, n_lo = 2.0e13, 1.0e13
    dn = n_hi - n_lo
    step = np.where(np.arange(nz) < face, n_hi, n_lo).astype(float)
    # A tick short enough that transport and zone exchange inside it stay far
    # below the quadrature gap the statement is about. Both zones carry the
    # same step, so the zone channel is in detailed balance at t = 0.
    dt = 1.0e-8
    # The gas-matched grid extent: six thermal speeds of the 300 K gas the box
    # actually holds.
    matched_vmax = 6.0 * np.sqrt(KB * 300.0 / M_HE)

    rows = []
    for label, vmax in (("matched", matched_vmax), ("shipped", None)):
        for nvz, nvp in ((48, 12), (64, 24)):
            kw = dict(
                nz=nz, nvz=nvz, nvp=nvp, Rp=R_col, Rm=Rm, face=face,
                nn_col=step, nn_ann=step, vmax_cm_s=vmax,
            )
            arm_z = bf_box(clear=R_col, **kw)
            arm_b = bf_box(clear=clear, **kw)
            t_f = float(arm_b.baffle_transparency[0])
            A_ann = float(arm_b.face_a[face])
            bf_tick(arm_z, dt)
            bf_tick(arm_b, dt)
            inc = np.asarray(arm_z.last_baffle_counts, dtype=float)
            blk = np.asarray(arm_b.last_baffle_counts, dtype=float)
            net_off = (float(inc[face - 1]) - float(inc[face])) / dt
            net_on = (
                (float(inc[face - 1]) - float(blk[face - 1]))
                - (float(inc[face]) - float(blk[face]))
            ) / dt
            ratio_on = (net_on / dn) / orifice
            ratio_off = (net_off / dn) / orifice
            geometric = A_ann / open_ann
            rows.append(
                {
                    "label": label,
                    "grid": (nvz, nvp),
                    "vmax": float(arm_b.g.vz.max()),
                    "t_f": t_f,
                    "geometric": geometric,
                    "ratio_on": ratio_on,
                    "ratio_off": ratio_off,
                    "structure": abs(
                        ratio_off - ratio_on * geometric
                    ) / max(abs(ratio_on * geometric), 1e-300),
                }
            )

    worst_structure = max(r["structure"] for r in rows)
    gaps = {}
    improves = True
    for label in ("matched", "shipped"):
        pair = [r for r in rows if r["label"] == label]
        coarse, fine = (abs(r["ratio_on"] - 1.0) for r in pair)
        gaps[label] = (coarse, fine)
        improves = improves and fine <= coarse
    worst_gap = max(g for pair in gaps.values() for g in pair)
    ok = (
        worst_structure < BF_FLUX_REL
        and worst_gap < BF_GRID_GAP_MAX
        and improves
    )
    detail = "\n        ".join(
        f"{r['label']:8s} grid {str(r['grid']):9s} vmax {r['vmax']:.3e} cm/s: "
        f"t_f {r['t_f']:.9f}; DVM/fluid ON {r['ratio_on']:.9f}, OFF "
        f"{r['ratio_off']:.9f}; OFF/(ON x A_ann/open_ann) - 1 = "
        f"{r['structure']:.3e}"
        for r in rows
    )
    return (
        "BF2 [THE PLAN GATE] matched case: the baffled DVM annulus throughput "
        "equals the fluid series orifice; unbaffled it overstates it by "
        "A_ann / open_ann",
        ok,
        f"g1atrim baffle face {stance_face}: R_clear {clear} cm, R_col (face "
        f"average) {R_col:.6f} cm, Rm {Rm:.4f} cm; open_ann {open_ann:.6f} "
        f"cm^2; A_ann/open_ann {rows[0]['geometric']:.9f} (the plan's "
        f"'~1.75x', MEASURED)\n        "
        f"fluid reference: 0.25 vbar open_ann = {orifice:.6e} cm^3/s at 300 K "
        f"(vbar {v_th:.6e} cm/s, clausing_scale 1)\n        "
        f"sealed {nz}-cell tube, annulus step {n_hi:.3e} -> {n_lo:.3e} cm^-3 "
        f"across the face, one {dt:.1e} s tick\n        {detail}\n        "
        f"OFF/ON structural identity worst {worst_structure:.3e} "
        f"(tol {fmt(BF_FLUX_REL)}); MEASURED free-molecular-vs-discrete-grid "
        f"gap, gas-matched grid {gaps['matched'][0]:.3e} at (48, 12) and "
        f"{gaps['matched'][1]:.3e} at (64, 24), shipped grid extent "
        f"{gaps['shipped'][0]:.3e} and {gaps['shipped'][1]:.3e} "
        f"(bound {BF_GRID_GAP_MAX} on EVERY row); refining improves it on "
        f"both extents: {improves}\n        "
        f"fluid conductances at the same face on the real stance geometry "
        f"(quoted, not gated): two_zone_knudsen_coefficients annulus "
        f"{fluid_two_zone:.6e} cm^3/s, knudsen_flow_coefficients "
        f"{fluid_single:.6e} cm^3/s -- each the tube in SERIES with the "
        f"orifice above"
    )


def gate_bf3():
    """BF3 in-solver: the baffle books where it should and closes both ledgers.

    The stance baffle armed on the ENGAGED production arm
    (``engaged_production_sim`` + ``neutral_kinetic_dvm_baffles``), ticked, and
    read three ways:

    * PLACEMENT -- the per-cell baffle counts are non-zero ONLY on the two
      cells flanking the stance's baffle face, and exactly zero everywhere
      else. A channel depositing on a positional constant rather than on the
      face it belongs to (the S1 defect class) fails here.
    * ENERGY CLOSURE -- the every-channel energy ledger. The gate asserts an
      ABSOLUTE bound in ULP of the tick's own energy scale,
      ``|distribution residual| <= BF_ENERGY_ULP_K x eps x scale`` (amended
      2026-08-31; see that constant for the derivation and the measurement it
      rests on). Both older normalizations are still REPORTED per the
      2026-08-30 rule -- throughput-normalized, and row-relative against the
      baffle's own energy row -- because a misbooking that moves only its own
      row by O(1) shows up in the row-relative figure. What changed is which
      one is load-bearing: the residual here is a WHOLE-LEDGER roundoff
      accumulation, so dividing it by a SINGLE channel's row compared two
      different objects and left the headroom to the luck of the rounding
      (measured: a 3.83x / 134.36x / 4.99x swing across three velocity grids
      whose denominators vary by 3.3 % and 0.2 %). The ULP form states the
      floating-point noise floor the bound is derived from.
      NEGATIVE CONTROL: an injected ``BF_ENERGY_CONTROL_ULP`` violation must
      trip the bound, so a pass cannot come from an unreachable tolerance.
    * MOMENTUM -- ``momentum_baffle_absorbed``. Its VALUE statement is AJ4's
      MIRROR ANTISYMMETRY, and for AJ4's stated reason: a second implementation
      of the tally inside the gate would be the same reduction over the same
      arrays and would agree with a sign error. A symmetric sealed box seeded
      only below the baffle face, and the same box seeded in the exact mirror
      image above it, must give rows summing to zero. The IN-SOLVER row is then
      required present, finite and non-vacuous -- the half the mirror box
      cannot make.
    """
    sim = engaged_production_sim(
        **{"flag:neutral_kinetic_dvm_baffles": True}
    )
    ledgers = run_until_updates(sim, 3)
    geom = sim.geometry
    face = int(np.asarray(geom.neutral_baffle_face_indices, dtype=int)[0])
    flanking = sorted({face - 1, face})
    dvm = sim._dvm

    counts = np.asarray(dvm.last_baffle_counts, dtype=float)
    off_face = float(np.sum(np.abs(np.delete(counts, flanking))))
    on_face = float(np.sum(np.abs(counts[flanking])))

    led = ledgers[-1]
    e = led["energy"]
    e_res = ledger_energy_residual(led)
    n_res = ledger_residual(led)
    row = abs(e["loss_baffle_blocked"]) + abs(e["birth_baffle_reemit"])
    row_relative = abs(e_res["distribution"]) / max(row, 1e-300)
    throughput_normalized = abs(e_res["distribution_rel"])
    # [bf3] THE ENERGY LEG THE GATE ASSERTS: the absolute distribution
    # residual against a bound of BF_ENERGY_ULP_K units in the last place of
    # the tick's own energy scale. See that constant for the derivation and
    # for what the row-relative form did instead.
    energy_abs = abs(e_res["distribution"])
    energy_bound = BF_ENERGY_ULP_K * DOUBLE_EPS * abs(e_res["scale"])
    energy_ulp = (
        energy_abs / (DOUBLE_EPS * abs(e_res["scale"]))
        if e_res["scale"]
        else float("inf")
    )
    energy_ok = energy_abs <= energy_bound
    # [bf3] NEGATIVE CONTROL for the new form, at the statement level: perturb
    # the residual by a violation just outside the bound and confirm the bound
    # REJECTS it. Without this the gate could be passing because the bound is
    # unreachable rather than because the ledger closes.
    control_injected = BF_ENERGY_CONTROL_ULP * DOUBLE_EPS * abs(e_res["scale"])
    control_trips = not (energy_abs + control_injected <= energy_bound)

    def mirror_row(side):
        nz, Rp, Rm = 8, 15.0, 50.0
        f = nz // 2
        profile = np.zeros(nz)
        profile[:f] = 1.0e13
        if side == "high":
            profile = profile[::-1].copy()
        box = bf_box(
            nz=nz, Rp=Rp, Rm=Rm,
            clear=bf_clear_radius(Rp, Rm, BF_TARGET_TRANSPARENCY),
            face=f, nn_col=np.zeros(nz), nn_ann=profile,
        )
        return bf_tick(box)

    led_low, led_high = mirror_row("low"), mirror_row("high")
    p_low = float(led_low["momentum_baffle_absorbed"])
    p_high = float(led_high["momentum_baffle_absorbed"])
    n_low = float(led_low["loss_baffle_blocked"])
    n_high = float(led_high["loss_baffle_blocked"])
    mirror = abs(p_low + p_high) / max(abs(p_low), abs(p_high), 1e-300)

    in_solver = led.get("momentum_baffle_absorbed")
    solver_row_ok = (
        in_solver is not None
        and np.isfinite(in_solver)
        and led["loss_baffle_blocked"] > 0.0
        and led["loss_baffle_blocked"] == led["birth_baffle_reemit"]
    )

    ok = (
        on_face > 0.0
        and off_face == 0.0
        and energy_ok
        and abs(n_res["distribution_rel"]) < BF_LEDGER_REL
        and solver_row_ok
        and n_low > 0.0
        and n_high > 0.0
        and p_low > 0.0
        and p_high < 0.0
        and mirror < BF_MOMENTUM_REL
        and control_trips
    )
    return (
        "BF3 in-solver, stance baffle armed: booked on the flanking cells "
        "only, both ledgers closed, the momentum row antisymmetric",
        ok,
        f"{geom.cells}-cell production mesh, baffle at face {face}; "
        f"{len(ledgers)} ticks; per-cell baffle counts {fmt(on_face)} on the "
        f"flanking pair {flanking} and {fmt(off_face)} everywhere else "
        f"(exactly 0 required)\n        "
        f"energy closure THE GATE: |distribution residual| "
        f"{fmt(energy_abs)} erg = {energy_ulp:.3f} ULP of the tick's energy "
        f"scale {fmt(abs(e_res['scale']))} erg, against a bound of "
        f"{BF_ENERGY_ULP_K:.0f} ULP = {fmt(energy_bound)} erg\n        "
        f"NEGATIVE CONTROL: an injected {fmt(BF_ENERGY_CONTROL_ULP)}-ULP "
        f"conservation violation ({fmt(control_injected)} erg) trips the "
        f"bound: {control_trips}\n        "
        f"INFORMATIONAL, not gated -- row-relative against the baffle's own "
        f"energy row {fmt(row_relative)} (row {fmt(row)} erg; this is the "
        f"form this gate asserted before 2026-08-31, whose margin was "
        f"grid-dependent), throughput-normalized "
        f"{fmt(throughput_normalized)}; particle "
        f"distribution residual {fmt(abs(n_res['distribution_rel']))} "
        f"(tol {fmt(BF_LEDGER_REL)})\n        "
        f"in-solver rows: loss_baffle_blocked "
        f"{fmt(led['loss_baffle_blocked'])} == birth_baffle_reemit "
        f"{fmt(led['birth_baffle_reemit'])}; momentum_baffle_absorbed "
        f"{fmt(float(in_solver) if in_solver is not None else float('nan'))} "
        f"g cm/s, present and finite: {solver_row_ok}\n        "
        f"mirror pair (AJ4's form): low-seed {fmt(p_low)} g cm/s over "
        f"{fmt(n_low)} intercepted particles, high-seed {fmt(p_high)} over "
        f"{fmt(n_high)}; |sum| / |each| {fmt(mirror)} "
        f"(tol {fmt(BF_MOMENTUM_REL)})"
    )


def gate_bf_g32():
    """G32 a clear radius below the local column radius is REFUSED, both routes.

    Two statements, because the refusal has two owners and only one of them is
    new. In the SOLVER route ``core.geometry._neutral_baffle_spec`` already
    demands ``Rp <= R_clear < Rm`` at every baffle face and raises before the
    kinetic arm is built at all, so the kinetic flag INHERITS that refusal
    rather than repeating it -- the gate arms the flag and quotes the message
    it actually gets, so a future relaxation of the geometry guard surfaces
    here. In the ENGINE route (a ``TransientDVM`` built directly, as the
    fixtures above do) nothing stands in front, and the engine's own refusal is
    all that is between a caller and an annulus silently sealed shut.
    """
    d, fl = arm_config(**PRODUCTION_GEOMETRY_KEYS)
    fl["neutral_kinetic_dvm_baffles"] = True
    d["neutral_baffle_clear_radii_cm"] = [1.0]
    solver_msg = ""
    try:
        LAPDSim1D(input_dict=d, input_flags=fl)
    except ValueError as exc:
        solver_msg = str(exc)

    engine_msg = ""
    try:
        bf_box(nz=8, Rp=15.0, Rm=50.0, clear=14.0, face=4)
    except ValueError as exc:
        engine_msg = str(exc)

    ok = (
        "clear radius" in solver_msg
        and "clear radius" in engine_msg
        and "R_col" in engine_msg
    )
    return (
        "G32 baffle clear radius below the local column radius refused "
        "(solver route inherits core.geometry; engine route is B6's own)",
        ok,
        f"solver route raised: {solver_msg[:96]!r}\n        "
        f"engine route raised: {engine_msg[:96]!r}",
    )


# ------------------------------------------------------------------ main


# The conservation and antisymmetry gates. Each is a statement about the
# OPERATOR -- that substep B creates exactly what substep A destroyed, that the
# fluid gain is minus the kinetic moment, that the zone channel moves particles
# without making any -- so it must hold whatever rate values the exchange
# closure hands the march. These are re-run once per value of
# ``neutral_kinetic_dvm_exchange``.
CONSERVATION_GATES = ("gate_i1", "gate_i2", "gate_i4", "gate_i5",
                      "gate_i6",
                      "gate_j2",
                      "gate_s1",
                      "gate_c1", "gate_c2", "gate_c3", "gate_c4",
                      "gate_d3", "gate_d4", "gate_d5",
                      "gate_b1", "gate_b3",
                      "gate_cf1", "gate_cf3",
                      "gate_wr1", "gate_wr3",
                      "gate_cj1", "gate_cj3",
                      "gate_aj1", "gate_aj3", "gate_aj4", "gate_aj6",
                      "gate_bf1")


def main():
    gates = [
        gate_i1,
        gate_i2,
        gate_i3,
        gate_i4,
        gate_i5,
        gate_i6,
        gate_j1,
        gate_j2,
        gate_j3,
        gate_s1,
        gate_s2,
        gate_c1,
        gate_c2,
        gate_c3,
        gate_c4,
        gate_c5,
        gate_c6,
        gate_r1,
        gate_r2,
        gate_p1,
        gate_p2,
        gate_p3,
        gate_d1,
        gate_d2,
        gate_d3,
        gate_d4,
        gate_d5,
        gate_b1,
        gate_b2,
        gate_b3,
        gate_cf1,
        gate_cf2,
        gate_cf3,
        gate_wr1,
        gate_wr2,
        gate_wr3,
        gate_cj1,
        gate_cj2,
        gate_cj3,
        gate_cj4,
        gate_ja1,
        gate_ja2,
        gate_ja3,
        gate_ja4,
        gate_ja5,
        gate_ja6,
        gate_ja7,
        gate_ja8,
        gate_aj1,
        gate_aj2,
        gate_aj3,
        gate_aj4,
        gate_aj5,
        gate_aj6,
        gate_bf1,
        gate_bf2,
        gate_bf3,
        gate_bf_g32,
        gate_x1,
    ]
    gates += [
        make_refusal_gate(label, offender, mutate)
        for label, _, offender, mutate in REFUSALS
    ]
    gates += [gate_l1, gate_l2, gate_l3, gate_l4, gate_l5]
    print("K2a transient-DVM coupling-integrity gate suite (E3 + limit cases)")
    print("=" * 78)
    print(f"accepted command line: {' '.join(sys.argv)}")
    print(f"neutral clock cadence under test: {CADENCE_S} s (PROVISIONAL)")
    print(f"zone-exchange closure of the main pass: {EXCHANGE_MODEL!r}")
    print("=" * 78)
    all_ok = True
    for g in gates:
        name, ok, detail = g()
        all_ok = all_ok and ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        print(f"        {detail}")
    all_ok = run_exchange_pass(EXCHANGE_MODELS, gates) and all_ok
    print("=" * 78)
    print("K2a DVM gates:", "ALL PASS" if all_ok else "FAILURES PRESENT")
    return 0 if all_ok else 1


def run_exchange_pass(models, gates):
    """Re-run the conservation/antisymmetry gates under the other closures.

    Same gate functions, same tolerances -- only the module-level
    ``EXCHANGE_MODEL`` the three construction sites read is rebound, so the
    second pass cannot drift from the first.
    """
    global EXCHANGE_MODEL
    by_name = {g.__name__: g for g in gates if hasattr(g, "__name__")}
    subset = [by_name[n] for n in CONSERVATION_GATES if n in by_name]
    ok_all = True
    original = EXCHANGE_MODEL
    for model in models:
        if model == original:
            continue
        print("-" * 78)
        print(f"conservation/antisymmetry gates re-run under "
              f"neutral_kinetic_dvm_exchange = {model!r}")
        EXCHANGE_MODEL = model
        try:
            for g in subset:
                name, ok, detail = g()
                ok_all = ok_all and ok
                print(f"[{'PASS' if ok else 'FAIL'}] [{model}] {name}")
                print(f"        {detail}")
        finally:
            EXCHANGE_MODEL = original
    return ok_all


if __name__ == "__main__":
    sys.exit(main())
