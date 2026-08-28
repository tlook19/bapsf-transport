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
      equals what the ACTIVE boundary term removed from the plasma there,
      per face, on all three geometries of RECYCLE_GEOMETRIES -- the shipped
      uniform bore, the R5 stand-in (whose plenum obstruction puts the
      cathode's live cell at index 2 rather than at the mesh start) and the
      PRODUCTION machine read from the stance file -- in both stances of
      ``characteristic_boundary``; and the arm deposits it in that same cell
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
  G1..G14 construction refusals: each unsupported configuration raises a
      ValueError at construction naming the offender. G2 is the model-preset
      resolver's refusal half -- an explicitly-set family member the
      selection cannot carry, refused ONCE with the whole member set named
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
import sys
from pathlib import Path
from types import SimpleNamespace

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
)
from cablp.solvers._sim1d.core.geometry import (
    absorbing_live_cells_by_role,
    is_plenum_cell,
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
    neutral_zone_volumes,
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
    candidate aborted the suite (campaign log 2026-08-23o).

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
# neutral-diffusion checkerboard reported in campaign log 2026-08-23o appeared.
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
    "loss_pump_L",
    "loss_pump_R",
    "birth_wall_accommodated",
    "birth_wall_reflected",
    "birth_mesh_reemit",
    "birth_puff",
    "birth_recombination",
    "birth_cathode_face",
    "birth_collector_face",
    "birth_anode",
    "net_surface_wall",
    "net_surface_mesh",
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


def independent_transfer(dvm, dt, plasma, rec):
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
    """
    g = dvm.g
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

    nu_cx, nu_el = dvm.collision_frequencies(
        plasma["n_i"], Ti, u
    )
    nu_coll = nu_cx + nu_el
    residual = dvm.f_c - rec_births
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


def transfer_reconstruction_error(dvm, dt, plasma, rec):
    """Return the worst relative error of :func:`independent_transfer`."""
    want = independent_transfer(dvm, dt, plasma, rec)
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
#: baffle arrays live. The DVM march itself does not read the baffle faces (the
#: fluid neutral operator does), so they travel as part of the machine rather
#: than as a kinetic input.
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
#: mesh alone (campaign log 2026-08-23o).
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


def recycle_identity(geometry_keys, characteristic_boundary, steps=40):
    """Compare the arm's wall-return channels with the plasma actually removed.

    The DESIGN INVARIANT: whatever the active plasma-terminating boundary
    term takes out of the plasma at an absorbing face, the arm re-injects as
    neutrals at that same face -- per face, to roundoff, in either stance.
    The two sides are read independently: the channel rates from
    ``_kinetic_channel_rates`` (what the arm will source), the removal from
    the boundary term's PLASMA row ``-n * V_plasma`` (what left the plasma).
    Nothing here reads the ``nn`` return row the implementation samples, so a
    channel that samples the wrong operator, or the wrong cell, cannot satisfy
    both sides at once.
    """
    overrides = {
        "neutral_kinetic_dvm_nvz": 16,
        "neutral_kinetic_dvm_nvp": 6,
        "flag:characteristic_boundary": bool(characteristic_boundary),
    }
    overrides.update(geometry_keys)
    sim = make_sim(**overrides)
    for _ in range(steps):
        advance_one_step(sim)
    geom = sim.geometry
    roles = np.asarray(geom.cell_role)
    Vp = np.asarray(geom.plasma_volume_cm3, dtype=float)
    state = sim.state
    term = (
        sim.characteristic_boundary_rhs(state=state)
        if characteristic_boundary
        else sim.boundary_absorption_rhs(state=state)
    )
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
    unaccounted = [
        k
        for k in led
        if k not in bookkeeping
        and not k.startswith("loss_")
        and not k.startswith("birth_")
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
        f"{len(energy)} energy entries; "
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
        for characteristic_boundary in (False, True):
            sim, roles, faces = recycle_identity(
                geometry_keys, characteristic_boundary
            )
            label = f"{geometry_name}/char={int(characteristic_boundary)}"
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
        "S1 recycle identity: what the arm re-injects equals what the active "
        "boundary removed, per face, on all three geometries and both stances",
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
        "G9 Tn feedback with characteristic_boundary refused",
        dict(),
        "characteristic_boundary",
        lambda d, fl: (
            d.__setitem__("neutral_kinetic_dvm_tn_feedback", True),
            fl.__setitem__("characteristic_boundary", True),
            None,
        )[2],
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
            None,
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
        "G15 gas puff into a cell with no annulus refused",
        dict(),
        "V_ann",
        lambda d, fl: (d.__setitem__("Rm", d["Rp"]), None)[1],
    ),
)


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
                      "gate_b1", "gate_b3")


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
