"""Build the sp3 shaped initial neutral profile nn0(z) for the foot-shape arm.

INSTRUMENT, not repo physics: this script is the VALUE PRODUCER behind the
solver's ``neutral_initial_profile`` capability, which ships no number of its
own. It writes an ``.npz`` that ``run_m6_point.py --nn0-profile-npz`` hands to
the solver as ``nn0_profile`` / ``nn0_annulus_profile``.

THE CONSTRUCTION (leg 3a of the sp campaign):

    nn0(z) = base + spread( first-flight lobe x throughput x dt_foot )

* ``base`` -- ONE OF TWO, and a result states which:

  - ``--base-from-h5 RUN.h5`` (the verdict-arm base): the ``t = 0`` column
    ``nn`` frame of an existing result, and its ``nn_a`` frame for the
    annulus. For the sp3 verdict arm that h5 is the sp1 fluid REFERENCE, so
    the base IS the reference run's own equilibrated initial profile and the
    arm's SINGLE DELTA is the foot addition on top of it. This matters
    numerically: the REF's equilibrated fill is ~2.6e12 cm^-3, about 7.5x
    BELOW the uniform ``nn0`` convention, so the two bases are not
    interchangeable.
  - the uniform ``resolve_nn0`` value for the stance config (the default).
    Retained for stances that do not start from an equilibrated seed
    (the SS/G-class conducting stances), where there is no reference frame to
    read and the configured scalar is the honest base.

    That value is now the configured ``nn0`` and nothing else. It used to
    have a fallback -- a ``None`` was resolved from a frozen gas-puff lookup
    table -- and the table is RETIRED, so a stance that leaves ``nn0 = None``
    (which is what a stance arming the per-cell ``nn0_profile`` does) is
    REFUSED here with a ValueError naming what ``nn0`` accepts, rather than
    being handed an interpolation on a superseded sccm convention. Such a
    stance has a per-cell profile to read and should be given one through
    ``--base-from-h5``.

  The two are mutually exclusive: passing ``--base-from-h5`` replaces the
  uniform base entirely, and the ledger records which was used.
* the lobe -- the gas puff's first-flight axial deposition, taken from the
  repo's own ``gas_puff_rate_profile`` at the named configuration's own puff
  keys: whichever ``gas_puff_profile`` that configuration resolves, together
  with its centre, width, throw and orifice bore and length. PROFILE-AGNOSTIC
  -- nothing here names or assumes a shape; it imports whatever puff row the
  configuration carries, never re-deriving one, so the accumulated shape is
  by construction the shape the running model deposits.
* the throughput -- AS-APPLIED, valves included: the same
  ``4.171431e17 * sccm * valves`` [particles/s] the solver applies, obtained
  from the repo's ``puff_rate`` rather than restated here. The ledger also
  prints the per-valve-nominal half, because both conventions are on the
  campaign record and a quoted number is incomplete without its convention.
* ``dt_foot`` -- the duration of the current foot the model forecloses.
  MEASURED, not fitted, and registered PER RUNG: the machine's own
  circuit-on -> 1 kA lead minus the model's own circuit-on -> 1 kA time
  (``MEASURED_LEAD_S`` - ``MODEL_1KA_S``, rounded to the 10 us the leads are
  quoted at). The model reaches 1 kA sooner than
  the machine does, and the gas that flows during the difference is the foot
  the model never sees. Its bracket is the measurement's own spread, the
  shot-to-shot standard deviation of the lead (``MEASURED_LEAD_SD_S``), so
  ``foot +- sd`` per rung -- an error bar, not a pair of modelling choices.
* ``spread`` -- carries the deposited inventory away from the lobe over
  ``dt_foot``. Three selectable kernels:

      knudsen    a conservative finite-volume axial diffusion SOLVE
      diffusive  gaussian, sigma = sqrt(2 D dt),  D = lambda vbar / 3
      ballistic  top-hat,  half-width = vbar dt

  ``knudsen`` is the REGISTERED member and is what an omitted ``--kernel``
  runs: the wall-limited finite-volume solve at
  :data:`KNUDSEN_KAPPA_REFERENCE`, gap-coupled, with the deposit released
  continuously over the foot across :data:`KNUDSEN_SUBSTEPS_DEFAULT`
  substeps. Its own registration is the three NAMED coefficient members
  below -- the reference and the two ends of its closure bracket -- and a
  member is requested by name with ``--knudsen-member``.

  ``diffusive`` and ``ballistic`` are MATRIX kernels: a stencil in ``z``
  evaluated once, its targets weighted by cell LENGTH and its columns
  normalized, applied to an inventory deposited whole at the start of the
  foot. They are RETAINED, reachable only by an explicit ``--kernel``, as
  the LEGACY REPRODUCTION ROUTE for rows built before the finite-volume
  member was registered, and they compute what they always computed.
  ``knudsen`` is not a stencil at all -- it integrates a diffusion equation
  on the builder's own mesh and zone volumes, weighted by cell VOLUME, with
  the source spread over the foot. All three conserve the injected
  inventory on the grid exactly (asserted).

NOTHING HERE IS FITTED. Every input is hardware-anchored (S_gp, valves, the
puff placement, the measured 1 kA lead), code-anchored (the lobe, the
throughput constant, the base fill, the model's own 1 kA time), or
literature-boxed (the He-He collision cross section, printed with its source
and overridable from the command line). The declared spread has two parts and
they are different kinds of thing: the ``dt_foot`` bracket is the measured
lead's error bar, so the registered foot is the value and the bracket is its
uncertainty, while the spreading COEFFICIENT's bracket is a closure the data
cannot pin, so the reference member is the value and its two named ends are
the bracket.

The kernels are stated, not assumed to be right:

* DIFFUSIVE is the random-walk limit -- the foot gas is collisional against
  the background fill, so it spreads as sqrt(t) with the elementary kinetic
  self-diffusion coefficient ``D = lambda vbar / 3``.
* BALLISTIC is the collisionless limit -- the foot gas free-streams for
  ``dt_foot``, so its support is the interval it can physically reach,
  ``vbar dt``. A TOP-HAT fills that interval flatly, which makes the reach
  literally the profile's support and so readable straight off the array. It
  is a deliberate idealization: an exactly free-streaming 3D Maxwellian
  projects onto a gaussian of sigma ``t sqrt(kT/m)`` = 0.63 ``vbar t``, a
  narrower core with tails past the top-hat edge.
* KNUDSEN is the wall-limited limit -- the gas meets the vessel wall far more
  often than it meets another atom, so the wall, not a gas-phase collision,
  is what randomizes it. Its statement is a conservation law rather than a
  displacement:

      dN_i/dt = S_i - sum_faces c_f (n_i - n_neighbour),   n_i = N_i / V_i

  on the builder's own cells, with ``V_i`` the TOTAL neutral volume of cell
  ``i``, ``S_i`` the deposit released at a constant rate over ``dt_foot``
  (``--knudsen-source-convention deposit_t0`` releases it whole at t = 0
  instead, which is the matrix members' convention), and ONE face
  conductance [cm^3/s] formula for every face:

      1/c_f = L_i/(2 D_i A_i) + L_j/(2 D_j A_j)
              + 1/2 [(1 - A_open/A_i) + (1 - A_open/A_j)] / (A_open vbar / 4)
      D_i   = kappa R_i vbar        [cm^2/s], the cell diffusivity
      A_i   = V_i / L_i             [cm^2],   the cell's open cross section
      R_i   = sqrt(A_i / pi)        [cm],     the local vessel radius

  The first two terms are the two half-cells' series resistance and the third
  is the thin-restriction (aperture) series resistance, which vanishes
  identically where nothing restricts the face. ``A_open`` is the
  configuration's OWN per-face open area, which already carries the bore
  step, the anode mesh's transparency and any neutral baffle's clear
  aperture, so this reads one number per face rather than re-deriving three.
  The aperture term is a free-molecular form and states the wide end of a
  step's resistance.

  That third term is SYMMETRIC in the two cells, which books half of a
  restriction's resistance against each side, and it therefore has two
  limits. At a plain AREA CHANGE the face is open to the smaller of the two
  cells, one bracket vanishes, and the face carries HALF the aperture term.
  At a THIN RESTRICTION between equal areas -- an anode mesh, a neutral
  baffle -- both brackets are the same non-zero number and the face carries
  the FULL term. Being symmetric, the conductance does not depend on which
  side of the face is read first.

  Because the measure is VOLUME and the two half-cell resistances are
  symmetric, a uniform DENSITY is in the operator's null space exactly: a
  source uniform per unit volume relaxes to a density continuous across a
  bore step. The matrix members' length weighting has the other fixed point,
  a uniform LINE density, so their added density steps by the area ratio at a
  bore change.

  ``kappa`` carries a REGISTRATION of three named members -- the reference
  :data:`KNUDSEN_KAPPA_REFERENCE` and the two ends of its closure bracket,
  :data:`KNUDSEN_KAPPA_SLOW` and :data:`KNUDSEN_KAPPA_FAST`. A member is
  requested by name (``--knudsen-member``), so walking the bracket is an
  invocation rather than a code edit and a row records which member it is;
  ``--knudsen-kappa`` still states a coefficient the registration does not
  carry, and the two are mutually exclusive. Naming neither runs the
  reference. The time integration is backward Euler over
  ``--knudsen-substeps`` fixed substeps (:data:`KNUDSEN_SUBSTEPS_DEFAULT`),
  each solved by an explicit tridiagonal (Thomas) sweep in plain arithmetic
  rather than a library factorization, so the rows it writes do not depend on
  which BLAS the environment linked. Backward Euler with this operator
  preserves positivity and conserves the inventory exactly; both are asserted
  at build.

  Which cells the operator transports through is the puff's own eligibility
  mask, as for the matrix members, so faces the puff gas cannot cross today
  stay closed -- except behind the anode mesh, where GAP COUPLING is part of
  the member's registration (:data:`KNUDSEN_GAP_COUPLING_REGISTERED`): the
  cathode and gap cells are carried too, coupled to the column through the
  anode mesh face at the transparency the configuration carries.
  ``--no-knudsen-gap-coupling`` selects the disclosed alternate that leaves
  that region empty; a stance carrying no anode face to couple through is
  refused unless it asks for the alternate. The plenum and the obstruction
  are never opened. Deposit landing outside the active set is re-homed to the
  nearest active cell before the solve, and the ledger reports the share.
  The domain ends are zero-flux: end pumping is not represented here.

THE STANCE IS OVERRIDABLE. ``--extra k=v`` / ``--extra-flag k=v`` carry
arbitrary ``input_dict`` / ``input_flags`` overrides into the stance the
builder assembles -- the same passthrough ``run_m6_point.py`` gives the RUN,
read by the same code (``extra_overrides.parse_extra_overrides``), so a
geometry the arm runs on is a geometry the foot is built on and a value
spelled one way here means one value there. They are applied LAST, after the
whole stance is assembled, so the grid, the puff lobe, the spread targets and
the zone volumes all see them. A key neither template owns is refused at that
parse layer; a key filed into the WRONG one of the two namespaces still
reaches ``LAPDSim1D``'s own construction-time refusal. Array-valued keys
(``plasma_radius_profile_cm`` and friends, one entry per mesh cell) come from
a file rather than a kilobyte of argv, via ``--extra-npz KEY=path.npz:array``.

Usage:

    python scripts/stance/sp3_build_nn0.py --sgp 5200 --two-zone \
        --out nn0_foot_es1.npz

    python scripts/stance/sp3_build_nn0.py --sgp 5200 --two-zone \
        --knudsen-member slow --out nn0_foot_es1_slow.npz

    python scripts/stance/sp3_build_nn0.py --sgp 5200 --two-zone \
        --kernel ballistic --out nn0_foot_es1_legacy.npz

(No kernel argument builds the REGISTERED member: the Knudsen reference at
the requested rung's registered foot. ``--dt-foot-s`` omitted takes that
foot; pass it to walk that rung's bracket. ``--knudsen-member`` and
``--knudsen-kappa`` both omitted take the reference member,
:data:`KNUDSEN_KAPPA_REFERENCE`.)
"""

import argparse
import json
import math

import numpy as np

# scripts/ sibling imports: the seven purpose subdirectories on sys.path.
import sys as _sys
from pathlib import Path as _Path
for _sub in ("atomic", "gates", "kinetic", "run", "score", "stance",
             "verify"):
    _dir = str(_Path(__file__).resolve().parents[1] / _sub)
    if _dir not in _sys.path:
        _sys.path.insert(0, _dir)

from compare_sim1d_es1 import PRODUCTION_NX, PARAM_OVERRIDES, FLAG_OVERRIDES
from extra_overrides import parse_extra_overrides
from run_mechanism_ladder import ES_OPERATING

from cablp.solvers._sim1d import LAPDSim1D, default_config, load_result_hdf5
from cablp.solvers._sim1d.core.config import resolve_nn0
from cablp.solvers._sim1d.physics.neutrals import (
    # The eligibility mask the puff itself uses. Imported rather than
    # restated: the spread must not carry gas into a cell the source is
    # forbidden from reaching (the plenum behind the cathode, the gap, the
    # end wall region), and "which cells those are" has exactly one owner.
    _PUFF_ELIGIBLE_ROLES,
    gas_puff_rate_profile,
    neutral_zone_volumes,
    puff_rate,
)
from cablp.constants import kb_cgs, m_He_cgs

#: He-He collision cross section [cm^2], hard-sphere from the Lennard-Jones
#: collision diameter sigma_LJ = 2.551 Angstrom for helium (Hirschfelder,
#: Curtiss & Bird, *Molecular Theory of Gases and Liquids*, the standard
#: viscosity-fitted He parameters; the same value is tabulated in Bird,
#: Stewart & Lightfoot, *Transport Phenomena*, App. E):
#:
#:     sigma_c = pi sigma_LJ^2 = pi (2.551e-8 cm)^2 = 2.044e-15 cm^2
#:
#: LITERATURE-BOXED, never fitted, and overridable with --sigma-hehe-cm2 (or
#: bypassed entirely with --mfp-cm) so the bracket's sensitivity to it is a
#: command-line question rather than a code edit. The diffusive reach goes as
#: sqrt(lambda) and so as sigma^-1/2: a factor 2 in the cross section is a
#: factor 1.4 in reach, which is inside the bracket's own width.
SIGMA_HE_HE_CM2 = 2.044e-15
SIGMA_HE_HE_SOURCE = (
    "hard sphere from the Lennard-Jones He collision diameter "
    "sigma_LJ = 2.551 Angstrom (Hirschfelder/Curtiss/Bird; Bird/Stewart/"
    "Lightfoot App. E): sigma_c = pi sigma_LJ^2"
)

#: The MEASURED circuit-on -> 1 kA lead of each ES rung [s], and the
#: shot-to-shot standard deviation of that lead over the rung's shots [s].
#: Machine quantities, read off the discharge-current records, never fitted.
MEASURED_LEAD_S = {1: 5.95e-3, 2: 6.75e-3, 3: 6.77e-3}
MEASURED_LEAD_SD_S = {1: 0.09e-3, 2: 0.02e-3, 3: 0.09e-3}

#: The MODEL's own circuit-on -> 1 kA time at each rung's operating point [s],
#: measured off the model's discharge current. Code-anchored, not fitted.
#:
#: What the subtraction in :func:`registered_foot_s` does with these is a CLOCK
#: ALIGNMENT: the same 1 kA threshold is read on the machine's discharge
#: current and on the model's, and the foot is the interval between the two
#: crossings. These times are therefore where the model's clock sits relative
#: to the machine's, not a property of the gas.
#:
#: THE TOLERANCE RULE THE REGISTRATION FOLLOWS. The model's 1 kA time is read
#: ONCE, at the REGISTERED FILL -- the initial fill the reference configuration
#: carries -- and is RE-REGISTERED only when it moves by MORE than that rung's
#: measured lead sd (:data:`MEASURED_LEAD_SD_S`) at any rung. A smaller move
#: sits inside the foot's own error bar, where the foot it would produce is
#: indistinguishable from the registered one; re-reading on such a move would
#: also make the fill and the time it is built from chase each other, since the
#: fill changes the model's approach to 1 kA and the time then changes the
#: fill. A move larger than the sd is outside the bracket the foot is stated
#: with, and all three rungs are then re-read together so one registration
#: holds across the ladder.
MODEL_1KA_S = {1: 0.067e-3, 2: 0.089e-3, 3: 0.139e-3}

#: The two LEGACY MATRIX kernels: the stencil members ``spread_matrix``
#: builds, retained as the reproduction route for rows built before the
#: finite-volume member was registered. They are reachable only by an
#: explicit ``--kernel``, and the ledger's ``kernel_bracket`` field names
#: this pair.
KERNELS = ("diffusive", "ballistic")

#: The wall-limited finite-volume spreading member's ``--kernel`` name, the
#: full set of selectable members, and the member an omitted ``--kernel``
#: builds. ``knudsen`` is the REGISTERED spreading member: its coefficient's
#: own registration is :data:`KNUDSEN_MEMBERS` below, and the settings that
#: complete it -- gap coupling, the source convention and the substep count --
#: are named in this module rather than left to argparse defaults.
KNUDSEN_KERNEL = "knudsen"
SELECTABLE_KERNELS = (KNUDSEN_KERNEL,) + KERNELS
KERNEL_REGISTERED = KNUDSEN_KERNEL

#: THE WALL-LIMITED OPERATOR'S REGISTRATION.
#:
#: ``kappa`` is the dimensionless coefficient in the cell diffusivity
#: ``D_i = kappa R_i vbar`` [cm^2/s]. Three values are REGISTERED and carry
#: names, so a member is requested by what it is rather than by a number
#: retyped on each invocation:
#:
#: ``KNUDSEN_KAPPA_REFERENCE``
#:     the reference member, the long-tube Knudsen diffusivity's own
#:     coefficient. This is the value an omitted ``--knudsen-kappa`` supplies,
#:     so the builder reproduces the reference member without being told the
#:     number.
#: ``KNUDSEN_KAPPA_SLOW`` / ``KNUDSEN_KAPPA_FAST``
#:     the two ends of the registered closure bracket, the shortest and the
#:     longest reach the coefficient is allowed to take. They are ALTERNATIVES
#:     to the reference and to each other -- a result names which member
#:     produced it -- and they are ends of a bracket, not error bars.
#:
#: ``KNUDSEN_MEMBERS`` maps the name ``--knudsen-member`` accepts to the value
#: it selects. A member name and an explicit ``--knudsen-kappa`` are mutually
#: exclusive: they are two ways of saying the same thing, and accepting both
#: would let them disagree. Any coefficient outside this table is stated on the
#: command line and recorded in the ledger; none is written here.
KNUDSEN_KAPPA_REFERENCE = 2.0 / 3.0
KNUDSEN_KAPPA_SLOW = 0.45
KNUDSEN_KAPPA_FAST = 0.90
KNUDSEN_MEMBERS = {
    "reference": KNUDSEN_KAPPA_REFERENCE,
    "slow": KNUDSEN_KAPPA_SLOW,
    "fast": KNUDSEN_KAPPA_FAST,
}
#: The member name an omitted ``--knudsen-member`` and an omitted
#: ``--knudsen-kappa`` together resolve to, and the name the ledger records for
#: a coefficient that is not one of the registered three.
KNUDSEN_MEMBER_DEFAULT = "reference"
KNUDSEN_MEMBER_UNREGISTERED = "unregistered"
#: Fixed number of equal backward-Euler substeps the foot is integrated over
#: when ``--knudsen-substeps`` is omitted [1]. Fixed rather than adaptive so
#: that a rebuild of the same invocation writes the same bytes.
KNUDSEN_SUBSTEPS_DEFAULT = 600
#: The two source conventions, in the order argparse offers them. ``continuous``
#: releases the deposit at a constant rate over ``dt_foot``; ``deposit_t0``
#: releases it whole at t = 0, which is what the matrix kernels do and is the
#: control the free-space variance check needs.
KNUDSEN_SOURCE_CONVENTIONS = ("continuous", "deposit_t0")
#: The cell roles gap coupling adds to the active set, on top of the puff's own
#: eligible roles.
KNUDSEN_GAP_COUPLED_ROLES = ("cathode", "gap")
#: The REGISTERED setting of gap coupling for the wall-limited member: whether
#: the region behind the anode mesh is carried, and so whether the operator can
#: place gas there at all. It is part of the member's registration rather than
#: an option with a convenient default, which is why it is named here and why
#: turning it off is an explicit switch (``--no-knudsen-gap-coupling``) that
#: selects a DISCLOSED ALTERNATE rather than a quieter default.
KNUDSEN_GAP_COUPLING_REGISTERED = True
#: The cell roles the operator never transports through, in either mode.
KNUDSEN_BLOCKED_ROLES = ("plenum", "obstruction")
#: Axial stations the ledger reports the added density at [cm]: the puff
#: station, the two cells flanking the source-bore step, and four cells down
#: the column. Reporting only; nothing keys off them.
KNUDSEN_PROBE_Z_CM = (60.0, 98.0, 107.0, 200.0, 300.0, 470.0)
#: Bar on the operator's own relative inventory error [1]. The march is
#: exactly conservative in exact arithmetic -- each face's two contributions
#: are equal and opposite, so the face sum telescopes -- and what this bounds
#: is therefore the ROUNDOFF of the substeps' tridiagonal solves, which
#: accumulates over them as a random walk rather than to a fixed figure. It is
#: set where every substep count and march length the operator accepts stays
#: inside it; at the registered substep count over a foot the error measures
#: about three orders of magnitude smaller, and the build's own conservation
#: check downstream of this one holds the rows a builder actually writes to the
#: tighter bar the builder has always used.
KNUDSEN_CONSERVATION_REL_TOL = 1.0e-10


#: The resolution the measured leads are quoted at [s]: 10 us, the number of
#: decimal places ``round`` takes to get there.
FOOT_QUANTUM_DECIMALS = 5


def registered_foot_s(es):
    """Return the registered ``dt_foot`` for ES rung ``es`` [s].

    The foot is the gas the model never sees: the machine's measured
    circuit-on -> 1 kA lead minus the model's own circuit-on -> 1 kA time at
    the same rung. A difference of two measured times, so MEASURED, not a fit.

    The difference is ROUNDED to 10 us, the resolution the measured leads are
    quoted at: carrying the raw subtraction's trailing digits would state a
    foot to a precision the measurement does not have. The rounded values are
    the feet OF RECORD -- ES1 0.00588, ES2 0.00666, ES3 0.00663 s -- and they
    are what an omitted ``--dt-foot-s`` supplies, so the builder reproduces a
    committed fill without being told the number.
    """
    raw = MEASURED_LEAD_S[es] - MODEL_1KA_S[es]
    return round(raw, FOOT_QUANTUM_DECIMALS)


def dt_foot_bracket_s(es):
    """Return ``(low, high)`` = registered foot +- the measured lead's sd [s].

    The rung's error bar on ``dt_foot``, printed with every ledger so a run
    always shows where in it the corner sits. It is an uncertainty, not a pair
    of modelling choices: the registered foot is the value, and the two ends
    are what the shot-to-shot spread of the lead allows. It centres on the
    ROUNDED foot, because that is the registered value; the sd is not rounded.
    """
    foot = registered_foot_s(es)
    sd = MEASURED_LEAD_SD_S[es]
    return (foot - sd, foot + sd)

#: Axial band the sp1 response map named as the required-source location
#: [cm]; reported for orientation only, nothing keys off it.
SP1_BAND_Z_CM = (790.0, 1045.0)


def parse_npz_overrides(items):
    """Return ``({key: value}, {key: provenance})`` from ``KEY=path.npz:array``.

    The array-valued route into the stance. A per-mesh-cell profile
    (``plasma_radius_profile_cm``, ``machine_radius_profile_cm``, ...) is
    hundreds of numbers and belongs in a file, not in argv; this reads the
    named array out of the named ``.npz`` and hands it over as a plain Python
    list, exactly the form the config templates take. A 0-d entry (an ``.npz``
    may hold scalars alongside its profiles) becomes the scalar itself, so one
    file can carry a whole geometry -- the machine length and the puff centre
    as readily as the profiles. KEY is the CONFIG key and ``arrayname`` the
    name inside the file; they need not agree, which is what lets one file
    hold several candidate profiles under distinguishing names.

    The script -- not the solver -- does the file I/O, as everywhere else in
    this campaign. The returned provenance records where each value came from
    and its shape, never the values themselves, which would bloat the ledger
    the output ``.npz`` carries.
    """
    values, provenance = {}, {}
    for item in items:
        key, sep, reference = item.partition("=")
        if not sep or not key or not reference:
            raise ValueError(
                f"npz override {item!r} is not of the form "
                "key=path.npz:arrayname"
            )
        path, sep, name = reference.rpartition(":")
        if not sep or not path or not name:
            raise ValueError(
                f"npz override {item!r} names no array: the value must be "
                "path.npz:arrayname"
            )
        with np.load(path, allow_pickle=False) as data:
            if name not in data:
                raise ValueError(
                    f"{path} carries no array {name!r}; it holds "
                    f"{sorted(data.files)}"
                )
            array = np.asarray(data[name])
        values[key] = array.item() if array.ndim == 0 else array.tolist()
        provenance[key] = {
            "source": reference,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }
    return values, provenance


def stance_config(es, nx, sgp, two_zone, extra_params=None, extra_flags=None):
    """Return (params, flags) for the production stance, as run_model builds it.

    ``extra_params`` and ``extra_flags`` are applied LAST, after the stance is
    fully assembled, so every consumer downstream of this function -- the
    solver geometry, the puff lobe, the spreading kernel's eligible targets,
    the zone volumes and the uniform base -- reads the overridden values. That
    ordering is the point: a geometry override that arrived earlier could be
    overwritten by the stance itself.
    """
    params, flags = default_config()
    params.update(PARAM_OVERRIDES)
    flags.update(FLAG_OVERRIDES)
    op = ES_OPERATING[es]
    params["nx"] = nx
    params["S_gp"] = float(sgp)
    params["V_bank"] = op["V_bank"]
    if two_zone:
        flags["neutral_two_zone"] = True
    if extra_params:
        params.update(extra_params)
    if extra_flags:
        flags.update(extra_flags)
    # The geometry is all this script needs from the solver, and the flags
    # that decide it are already set. Equilibration is a start_simulation()
    # behaviour and never runs at construction, so this costs one build.
    return params, flags


def base_profiles_from_h5(path, cells, two_zone):
    """Return ``(base_column, base_annulus)`` from a result's ``t = 0`` frames.

    The base is the run's INITIAL neutral state, so the first saved frame must
    actually be the initial one: a result whose first sample sits at ``t > 0``
    (a ``t_save_start`` beyond zero) is refused rather than silently treated as
    an initial condition it is not.

    ``base_annulus`` is the ``nn_a`` frame under the two-zone closure and
    ``None`` without it. The closure must MATCH: a two-zone build needs an
    ``nn_a`` to read, and a single-field build refuses a two-zone source,
    because collapsing two zones into one field is a modelling choice this
    script does not get to make silently.
    """
    result = load_result_hdf5(path)
    time = np.asarray(result.time, dtype=float)
    if time.size == 0 or time[0] != 0.0:
        raise ValueError(
            f"{path} does not save a t = 0 frame (first saved time "
            f"{time[0] if time.size else 'none'!r}), so its first frame is "
            "not an initial condition; rerun the source with t_save_start = 0"
        )
    base_col = np.array(result.nn[0], dtype=float).reshape(-1)
    if base_col.size != int(cells):
        raise ValueError(
            f"{path} has {base_col.size} cells, this stance has {cells}; the "
            "base profile and the run it seeds must be on the same grid "
            "(check --nx and the geometry keys)"
        )
    source_nn_a = getattr(result, "nn_a", None)
    if two_zone:
        if source_nn_a is None:
            raise ValueError(
                f"--two-zone needs an annulus base, but {path} carries no "
                "nn_a (it is a single-field run). Use a two-zone source, or "
                "drop --two-zone"
            )
        base_ann = np.array(source_nn_a[0], dtype=float).reshape(-1)
        if base_ann.size != int(cells):
            raise ValueError(
                f"{path} nn_a has {base_ann.size} cells, expected {cells}"
            )
    else:
        if source_nn_a is not None:
            raise ValueError(
                f"{path} is a TWO-ZONE run but this build is single-field; "
                "folding its two zones into one neutral field is a modelling "
                "choice, not a conversion. Pass --two-zone"
            )
        base_ann = None
    for label, arr in (("nn", base_col), ("nn_a", base_ann)):
        if arr is None:
            continue
        if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
            raise ValueError(
                f"{path} frame 0 {label} must be finite and > 0 (got min "
                f"{float(np.min(arr)):.6g}); the solver refuses such a profile "
                "as an initial condition and so does this"
            )
    return base_col, base_ann


def mean_speed_cm_s(T_K, mass_g):
    """Return the Maxwellian mean speed sqrt(8 k T / (pi m)) [cm/s]."""
    return math.sqrt(8.0 * kb_cgs * float(T_K) / (math.pi * float(mass_g)))


def spread_matrix(geometry, kernel, width_cm):
    """Return the inventory-conserving spreading operator ``W`` [1].

    ``W[i, j]`` is the fraction of the inventory deposited in source cell
    ``j`` that ends up in target cell ``i``. Columns sum to exactly 1 (to
    roundoff), which is what makes the spread conservative on the grid
    regardless of the kernel's shape or the domain's finite extent -- mass
    that would leave the ends is returned to the reachable cells rather than
    deleted, the discrete stand-in for reflecting walls.

    Targets are restricted to the puff's own eligible roles, weighted by cell
    length so a refinement of the grid converges rather than redistributing.

    A source cell from which the kernel reaches NO eligible target gets an
    identically-zero column. That is a real possibility on a coarse grid with
    a narrow kernel, and it is harmless exactly when such a cell carries no
    inventory -- which is the caller's to check, because only the caller knows
    the deposit. Silence here would delete particles; the conservation check in
    :func:`build` is what turns that into a loud failure.
    """
    if not (math.isfinite(width_cm) and width_cm > 0.0):
        raise ValueError(
            "the spreading kernel needs a finite width > 0 (got "
            f"{width_cm!r}); a zero-width spread is the NULL CONTROL and is "
            "handled by dt_foot = 0, which deposits nothing and never reaches "
            "this function"
        )
    z = np.asarray(geometry.z_cm, dtype=float)
    length = np.asarray(geometry.length_cm, dtype=float)
    eligible = np.array(
        [role in _PUFF_ELIGIBLE_ROLES for role in geometry.cell_role], dtype=bool
    )
    dz = z[:, None] - z[None, :]
    if kernel == "diffusive":
        raw = np.exp(-0.5 * (dz / float(width_cm)) ** 2)
    elif kernel == "ballistic":
        raw = (np.abs(dz) <= float(width_cm)).astype(float)
    else:
        raise ValueError(f"kernel must be one of {list(KERNELS)} (got {kernel!r})")
    raw = raw * (length * eligible)[:, None]
    column_sum = raw.sum(axis=0)
    return np.divide(
        raw, column_sum, out=np.zeros_like(raw), where=column_sum > 0.0
    )


def knudsen_member_name(kappa):
    """Return the registered member name of a spreading coefficient [str].

    One of the keys of :data:`KNUDSEN_MEMBERS` when ``kappa`` is exactly that
    member's value, and :data:`KNUDSEN_MEMBER_UNREGISTERED` otherwise. The
    ledger records the name beside the number so a row says which member of
    the registration produced it, whether the invocation asked for the member
    by name or stated its coefficient.
    """
    for name, value in KNUDSEN_MEMBERS.items():
        if float(kappa) == float(value):
            return name
    return KNUDSEN_MEMBER_UNREGISTERED


def knudsen_active_mask(cell_role, gap_coupling=False):
    """Return the boolean mask of cells the wall-limited operator carries gas in.

    The puff's own eligible roles, which is what makes the operator's reach
    the reach the source itself is allowed -- plus, under ``gap_coupling``,
    the roles in :data:`KNUDSEN_GAP_COUPLED_ROLES`, which opens the region
    behind the anode mesh to the column through that face. The roles in
    :data:`KNUDSEN_BLOCKED_ROLES` are never active in either mode.

    Cells outside the mask keep the base exactly: nothing is transported into
    or out of them, and deposit landing in one is re-homed before the solve.
    """
    roles = [str(role) for role in cell_role]
    active_roles = set(_PUFF_ELIGIBLE_ROLES)
    if gap_coupling:
        active_roles |= set(KNUDSEN_GAP_COUPLED_ROLES)
    active_roles -= set(KNUDSEN_BLOCKED_ROLES)
    return np.array([role in active_roles for role in roles], dtype=bool)


def knudsen_face_conductances(
    z_cm, length_cm, neutral_volume_cm3, face_open_area_cm2, active,
    vbar_cm_s, kappa,
):
    """Return ``(index, conductance)`` for the active chain [1], [cm^3/s].

    ``index`` is the ascending array of active cell indices and
    ``conductance[a]`` is the conductance of the face between ``index[a]`` and
    ``index[a + 1]``, from the one formula the module docstring states. Faces
    are numbered as the mesh numbers them, so the face between cells ``k`` and
    ``k + 1`` is entry ``k + 1`` of ``face_open_area_cm2``, whose length is
    one more than the number of cells.

    The restriction member of that formula is SYMMETRIC in the two cells,

        1/2 [(1 - A_open/A_lo) + (1 - A_open/A_hi)] / (A_open vbar / 4)

    so it books half of a restriction's resistance against each side. It has
    two limits, and they are the two things a face can be. At a plain AREA
    CHANGE the face is open to the smaller of the two cells, ``A_open`` equals
    that area, one bracket vanishes and the face carries HALF the aperture
    term. At a THIN RESTRICTION between equal areas -- a mesh, a baffle -- both
    brackets are the same non-zero number and the face carries the FULL term.
    Because the expression is symmetric under exchanging the two cells, the
    conductance does not depend on which side is read first.

    The active set must be CONTIGUOUS. A hole in it would put two cells that
    do not share a face on either side of one conductance, and the half-cell
    resistances would then understate the distance between them by the whole
    hole; rather than invent a bridging rule this raises.
    """
    z = np.asarray(z_cm, dtype=float)
    length = np.asarray(length_cm, dtype=float)
    volume = np.asarray(neutral_volume_cm3, dtype=float)
    face_open = np.asarray(face_open_area_cm2, dtype=float)
    if face_open.size != z.size + 1:
        raise ValueError(
            f"the per-face open area has {face_open.size} entries for "
            f"{z.size} cells; a mesh has one more face than cells"
        )
    if not (math.isfinite(kappa) and float(kappa) > 0.0):
        raise ValueError(
            f"the wall-limited operator needs a finite kappa > 0 (got "
            f"{kappa!r}); kappa scales the cell diffusivity D = kappa R vbar"
        )
    index = np.flatnonzero(np.asarray(active, dtype=bool))
    if index.size < 2:
        raise ValueError(
            f"the wall-limited operator needs at least two active cells to "
            f"have a face (got {index.size}); check the cell roles"
        )
    if not np.array_equal(index, np.arange(index[0], index[-1] + 1)):
        raise ValueError(
            "the wall-limited operator needs a CONTIGUOUS active set, and "
            f"this one has {int(index[-1] - index[0] + 1 - index.size)} "
            "inactive cells inside its span; a face between two cells that "
            "do not touch has no half-cell resistance this can state"
        )
    area = volume / length
    radius = np.sqrt(area / math.pi)
    diffusivity = float(kappa) * radius * float(vbar_cm_s)
    lo, hi = index[:-1], index[1:]
    open_area = face_open[hi]
    if np.any(open_area <= 0.0):
        raise ValueError(
            "a face inside the active set has no open area, so the aperture "
            "resistance is infinite; such a face is closed and its cells "
            "must not both be active"
        )
    restriction = 0.5 * (
        (1.0 - open_area / area[lo]) + (1.0 - open_area / area[hi])
    )
    resistance = (
        length[lo] / (2.0 * diffusivity[lo] * area[lo])
        + length[hi] / (2.0 * diffusivity[hi] * area[hi])
        + restriction / (open_area * float(vbar_cm_s) / 4.0)
    )
    return index, 1.0 / resistance


def _thomas_sweep(lower, diag, upper, rhs):
    """Return the solution of a tridiagonal system by the Thomas sweep.

    ``diag`` is the main diagonal, ``lower[a]`` the entry of row ``a + 1`` in
    column ``a`` and ``upper[a]`` the entry of row ``a`` in column ``a + 1``.
    Written out rather than handed to a library so the arithmetic is fixed by
    this source and not by which BLAS the environment linked.
    """
    n = int(diag.size)
    sweep_upper = [0.0] * n
    sweep_rhs = [0.0] * n
    low = lower.tolist()
    up = upper.tolist()
    dia = diag.tolist()
    right = rhs.tolist()
    pivot = dia[0]
    sweep_upper[0] = up[0] / pivot
    sweep_rhs[0] = right[0] / pivot
    for k in range(1, n):
        pivot = dia[k] - low[k - 1] * sweep_upper[k - 1]
        if k < n - 1:
            sweep_upper[k] = up[k] / pivot
        sweep_rhs[k] = (right[k] - low[k - 1] * sweep_rhs[k - 1]) / pivot
    solution = [0.0] * n
    solution[n - 1] = sweep_rhs[n - 1]
    for k in range(n - 2, -1, -1):
        solution[k] = sweep_rhs[k] - sweep_upper[k] * solution[k + 1]
    return np.array(solution, dtype=float)


def knudsen_spread(
    z_cm, length_cm, neutral_volume_cm3, face_open_area_cm2, active,
    deposited, dt_foot_s, vbar_cm_s,
    kappa=KNUDSEN_KAPPA_REFERENCE,
    substeps=KNUDSEN_SUBSTEPS_DEFAULT,
    source_convention=KNUDSEN_SOURCE_CONVENTIONS[0],
):
    """Return ``(accumulated, report)`` for the wall-limited spreading solve.

    ``accumulated`` is the inventory per cell [particles] after ``dt_foot_s``,
    zero outside the active set, and ``report`` is a dict of the quantities
    the ledger prints: the re-homed share of the deposit, the substep, the
    smallest and largest face conductance, and the two asserted properties.

    The solve is the module docstring's conservation law, integrated by
    backward Euler over ``substeps`` equal substeps. Each substep solves
    ``(I + h K V^-1) N = N + h S`` -- a tridiagonal M-matrix system, so the
    step is positivity-preserving -- and the face sum telescopes, so the
    inventory is conserved to roundoff. Both are asserted here rather than
    left to the caller.
    """
    volume = np.asarray(neutral_volume_cm3, dtype=float)
    deposit = np.array(deposited, dtype=float).reshape(-1).copy()
    mask = np.asarray(active, dtype=bool)
    if str(source_convention) not in KNUDSEN_SOURCE_CONVENTIONS:
        raise ValueError(
            f"source convention must be one of "
            f"{list(KNUDSEN_SOURCE_CONVENTIONS)} (got {source_convention!r})"
        )
    substeps = int(substeps)
    if substeps < 1:
        raise ValueError(
            f"the wall-limited operator needs at least one substep (got "
            f"{substeps}); the substep count is fixed, never adaptive"
        )
    index, conductance = knudsen_face_conductances(
        z_cm, length_cm, volume, face_open_area_cm2, mask, vbar_cm_s, kappa,
    )
    # RE-HOMING. The lobe can deposit into a cell the operator does not carry
    # gas in -- behind the anode mesh under the default active set. Such gas
    # is placed in the nearest active cell BEFORE the solve, so it is spread
    # rather than deleted, and the share is reported because it is a
    # modelling choice and not an arithmetic detail.
    z = np.asarray(z_cm, dtype=float)
    injected = float(np.sum(deposit))
    rehomed = 0.0
    for cell in np.flatnonzero(~mask):
        stranded = float(deposit[cell])
        if stranded == 0.0:
            continue
        nearest = int(index[int(np.argmin(np.abs(z[index] - z[cell])))])
        deposit[nearest] += stranded
        deposit[cell] = 0.0
        rehomed += stranded

    cell_volume = volume[index]
    size = int(index.size)
    step = float(dt_foot_s) / substeps
    diag = np.ones(size, dtype=float)
    diag[:-1] += step * conductance / cell_volume[:-1]
    diag[1:] += step * conductance / cell_volume[1:]
    upper = -step * conductance / cell_volume[1:]
    lower = -step * conductance / cell_volume[:-1]

    if str(source_convention) == "continuous":
        state = np.zeros(size, dtype=float)
        source = deposit[index] / float(dt_foot_s)
    else:
        state = deposit[index].copy()
        source = np.zeros(size, dtype=float)
    for _ in range(substeps):
        state = _thomas_sweep(lower, diag, upper, state + step * source)

    accumulated = np.zeros(z.size, dtype=float)
    accumulated[index] = state
    minimum = float(np.min(state))
    assert minimum >= 0.0, (
        f"the wall-limited solve went negative: min {minimum:.9e} particles. "
        "Backward Euler on this operator is positivity-preserving, so a "
        "negative cell means the operator is not the one documented"
    )
    held = float(np.sum(state))
    conservation_rel = abs(held - injected) / max(injected, 1e-300)
    assert conservation_rel < KNUDSEN_CONSERVATION_REL_TOL, (
        f"the wall-limited solve lost inventory: in {injected:.9e}, out "
        f"{held:.9e}, rel {conservation_rel:.3e}"
    )
    report = {
        "kappa": float(kappa),
        "member": knudsen_member_name(kappa),
        "substeps": substeps,
        "substep_s": step,
        "source_convention": str(source_convention),
        "active_cells": size,
        "active_span_cells": [int(index[0]), int(index[-1])],
        "rehomed_fraction": (
            0.0 if injected == 0.0 else rehomed / injected
        ),
        "face_conductance_min_cm3_s": float(np.min(conductance)),
        "face_conductance_max_cm3_s": float(np.max(conductance)),
        "min_cell_particles": minimum,
        "conservation_rel": conservation_rel,
    }
    return accumulated, report


def knudsen_added_rows(z_cm, length_cm, neutral_volume_cm3, accumulated, active):
    """Return the ledger rows describing where the added inventory landed.

    Three readings of one array. ``z50``/``z90``/``z99`` are the axial
    positions [cm] below which that fraction of the ADDED inventory sits, so
    they say how far the operator carried the foot; they are quantiles of the
    added inventory in ``z``, interpolated on its cumulative sum. The bore-step
    row is the added DENSITY on each side of a bore step, and their ratio,
    which is the number a length-weighted kernel fixes at the area ratio and a
    volume-consistent one drives to 1 for a uniform source. The step reported
    is the largest area change among the faces the added inventory reaches --
    those at or below ``z90`` -- so it is the step the fill actually crosses
    rather than whichever change happens to be the largest on the whole mesh.
    The probe row is the added density at :data:`KNUDSEN_PROBE_Z_CM`, each at
    its nearest cell.
    """
    z = np.asarray(z_cm, dtype=float)
    volume = np.asarray(neutral_volume_cm3, dtype=float)
    added = np.asarray(accumulated, dtype=float)
    density = added / volume
    total = float(np.sum(added))
    cumulative = np.cumsum(added) / max(total, 1e-300)
    z50 = float(np.interp(0.5, cumulative, z))
    z90 = float(np.interp(0.9, cumulative, z))
    z99 = float(np.interp(0.99, cumulative, z))
    index = np.flatnonzero(np.asarray(active, dtype=bool))
    area = volume / np.asarray(length_cm, dtype=float)
    lo, hi = index[:-1], index[1:]
    ratio = np.maximum(area[lo], area[hi]) / np.minimum(area[lo], area[hi])
    reached = z[hi] <= z90
    if not np.any(reached):
        reached = np.ones(ratio.size, dtype=bool)
    step = int(np.flatnonzero(reached)[int(np.argmax(ratio[reached]))])
    narrow, wide = (lo[step], hi[step]) if area[lo[step]] < area[hi[step]] else (
        hi[step], lo[step]
    )
    return {
        "added_z50_cm": z50,
        "added_z90_cm": z90,
        "added_z99_cm": z99,
        "bore_step_area_ratio": float(ratio[step]),
        "bore_step_narrow_z_cm": float(z[narrow]),
        "bore_step_wide_z_cm": float(z[wide]),
        "bore_step_narrow_added_cm3": float(density[narrow]),
        "bore_step_wide_added_cm3": float(density[wide]),
        "bore_step_added_density_ratio": float(
            density[narrow] / density[wide]
            if density[wide] != 0.0 else float("inf")
        ),
        "probe_added_density_cm3": [
            [float(station), float(z[int(np.argmin(np.abs(z - station)))]),
             float(density[int(np.argmin(np.abs(z - station)))])]
            for station in KNUDSEN_PROBE_Z_CM
        ],
    }


def build(args):
    """Return (profiles, ledger) for the requested corner of the bracket."""
    # npz-sourced values first, so an inline --extra can still override any of
    # them -- the same precedence run_m6_point.py gives its file-sourced nn0.
    npz_params, npz_provenance = parse_npz_overrides(args.extra_npz)
    inline_params = parse_extra_overrides(args.extra, "--extra")
    extra_params = dict(npz_params)
    extra_params.update(inline_params)
    extra_flags = parse_extra_overrides(args.extra_flag, "--extra-flag")
    params, flags = stance_config(
        args.es, args.nx, args.sgp, args.two_zone,
        extra_params=extra_params, extra_flags=extra_flags,
    )
    if params["gas_type"] != "He":
        raise ValueError(
            "sp3_build_nn0 is helium-only: the collision cross section and the "
            f"thermal speed are both He-He (stance gas_type={params['gas_type']!r})"
        )
    geometry = LAPDSim1D(dict(params), dict(flags)).geometry
    cells = int(geometry.cells)

    V_chamber_all = np.asarray(geometry.neutral_volume_cm3, dtype=float)
    V_col_all, V_ann_all = neutral_zone_volumes(geometry)
    if args.base_from_h5 is None:
        base_col_profile, base_ann_profile = None, None
        base_scalar = float(resolve_nn0(params))
        base_col = np.full(cells, base_scalar)
        base_ann = np.full(cells, base_scalar) if args.two_zone else None
        base_source = "resolve_nn0 at the stance config (shipped convention)"
    else:
        base_col_profile, base_ann_profile = base_profiles_from_h5(
            args.base_from_h5, cells, args.two_zone
        )
        base_col = base_col_profile
        base_ann = base_ann_profile
        base_source = f"t=0 frames of {args.base_from_h5}"
    # The scalar density the mean free path is evaluated at. For a profile
    # base that is the CHAMBER-VOLUME-WEIGHTED MEAN over the whole grid (total
    # neutral particles / total neutral volume, both zones counted) -- the
    # honest single number for a spread that is computed once for the grid.
    # lambda goes as 1/n and the diffusive reach as sqrt(lambda), so the choice
    # is a weak one, and --mfp-cm overrides it outright.
    if base_ann is None:
        base_particles = float(np.sum(base_col * V_chamber_all))
    else:
        base_particles = float(
            np.sum(base_col * V_col_all + base_ann * V_ann_all)
        )
    base_density = base_particles / float(np.sum(V_chamber_all))
    Tn_K = float(args.tn_k if args.tn_k is not None else params["Tn_K"])
    vbar = mean_speed_cm_s(Tn_K, m_He_cgs)

    # --- the deposited inventory -------------------------------------------
    # gas_puff_rate_profile returns [cm^-3 s^-1] against the CHAMBER volume,
    # normalized so that sum(rate * V_chamber) is the whole throughput.
    rate = gas_puff_rate_profile(
        geometry,
        params["S_gp"],
        params["gas_puff_valves"],
        profile=params["gas_puff_profile"],
        z_cm=params["gas_puff_z_cm"],
        sigma_cm=params["gas_puff_sigma_cm"],
        throw_cm=params["gas_puff_throw_cm"],
        orifice_id_cm=params["gas_puff_orifice_id_cm"],
        orifice_length_cm=params["gas_puff_orifice_length_cm"],
        end=0,
    )
    V_chamber = V_chamber_all
    deposited = rate * V_chamber * float(args.dt_foot_s)  # particles per cell

    throughput_applied = puff_rate(params["S_gp"], params["gas_puff_valves"], 1.0)
    throughput_nominal = puff_rate(params["S_gp"], 1, 1.0)
    injected_applied = throughput_applied * float(args.dt_foot_s)
    injected_nominal = throughput_nominal * float(args.dt_foot_s)

    # --- the spread ---------------------------------------------------------
    if args.mfp_cm is not None:
        mfp = float(args.mfp_cm)
        mfp_source = "explicit --mfp-cm"
    else:
        # Like-particle mean free path: the sqrt(2) is the relative-speed
        # correction for a test particle moving through its own species.
        mfp = 1.0 / (math.sqrt(2.0) * base_density * float(args.sigma_hehe_cm2))
        mfp_source = (
            f"1 / (sqrt(2) n sigma) at n = the base's chamber-volume-weighted "
            f"mean = {base_density:.6g} cm^-3, "
            f"sigma = {args.sigma_hehe_cm2:.6g} cm^2 [{SIGMA_HE_HE_SOURCE}]"
        )
    D_cm2_s = mfp * vbar / 3.0
    knudsen_report = None
    if args.kernel == KNUDSEN_KERNEL:
        # The wall-limited member has no single width: its reach is set by a
        # per-cell diffusivity and the mesh's own face areas, so the ledger
        # names the operator instead of a number it does not have.
        width = None
        width_label = (
            "no single width: a finite-volume solve with D_i = kappa R_i vbar"
        )
    elif args.kernel == "diffusive":
        width = math.sqrt(2.0 * D_cm2_s * float(args.dt_foot_s))
        width_label = "gaussian sigma = sqrt(2 D dt)"
    else:
        width = vbar * float(args.dt_foot_s)
        width_label = "top-hat half-width = vbar dt"

    if float(args.dt_foot_s) == 0.0:
        # THE NULL CONTROL (dt_foot = 0): nothing was deposited, so there is
        # nothing to spread and no kernel is built. Short-circuited rather
        # than passed through a zero-width kernel or a zero-length solve,
        # neither of which is defined.
        accumulated = np.zeros(cells, dtype=float)
    elif args.kernel == KNUDSEN_KERNEL:
        gap_coupling = bool(args.knudsen_gap_coupling)
        if gap_coupling and np.asarray(geometry.anode_face_indices).size == 0:
            raise ValueError(
                "gap coupling carries the region behind the anode mesh to the "
                "column THROUGH that mesh face, and this stance carries no "
                "anode face to couple through; it is the wall-limited "
                "member's REGISTERED setting, so a stance without that face "
                "must ask for the disclosed alternate explicitly with "
                "--no-knudsen-gap-coupling"
            )
        active = knudsen_active_mask(geometry.cell_role, gap_coupling)
        accumulated, knudsen_report = knudsen_spread(
            geometry.z_cm,
            geometry.length_cm,
            V_chamber,
            geometry.neutral_face_area_cm2,
            active,
            deposited,
            float(args.dt_foot_s),
            vbar,
            kappa=float(args.knudsen_kappa),
            substeps=int(args.knudsen_substeps),
            source_convention=args.knudsen_source_convention,
        )
        knudsen_report["gap_coupling"] = gap_coupling
        knudsen_report.update(
            knudsen_added_rows(
                geometry.z_cm, geometry.length_cm, V_chamber, accumulated,
                active,
            )
        )
    else:
        spread = spread_matrix(geometry, args.kernel, width)
        # A source cell the kernel cannot carry out of is only safe if it
        # holds nothing; otherwise the spread would delete its particles
        # silently.
        unreachable = spread.sum(axis=0) <= 0.0
        if np.any(deposited[unreachable] != 0.0):
            raise ValueError(
                "the spreading kernel reaches no eligible cell from a source "
                "cell that carries deposited gas, so the spread would delete "
                f"it: {int(np.count_nonzero(deposited[unreachable] != 0.0))} "
                f"such cells at kernel width {width:.6g} cm. Widen the kernel "
                "or refine the grid"
            )
        accumulated = spread @ deposited  # particles per cell after spreading

    grid_in = float(deposited.sum())
    grid_out = float(accumulated.sum())
    conservation_rel = abs(grid_out - grid_in) / max(grid_in, 1e-300)
    assert conservation_rel < 1e-12, (
        f"spreading kernel lost inventory: in {grid_in:.9e}, out "
        f"{grid_out:.9e}, rel {conservation_rel:.3e}"
    )

    # --- routing into the neutral field(s) ---------------------------------
    V_col, V_ann = V_col_all, V_ann_all
    add_col = np.zeros(cells, dtype=float)
    add_ann = np.zeros(cells, dtype=float) if args.two_zone else None
    if not args.two_zone:
        # One chamber-mean neutral field: the whole cell volume holds it.
        add_col = accumulated / V_chamber
    elif args.zone == "chamber":
        # Radially well-mixed: both zones rise by the same density, so the
        # particle count is conserved cell by cell (V_col + V_ann = V_chamber).
        add_col = accumulated / V_chamber
        add_ann = accumulated / V_chamber
    elif args.zone == "annulus":
        # The shipped first-flight routing: the pipe enters at the wall, so
        # the puff feeds the annulus first and annulus-free cells fall back to
        # the column -- exactly as neutral_source_sink_rhs routes it.
        has_ann = V_ann > 0.0
        add_ann = np.where(has_ann, accumulated / np.maximum(V_ann, 1e-300), 0.0)
        add_col = np.where(has_ann, 0.0, accumulated / np.maximum(V_col, 1e-300))
    elif args.zone == "column":
        add_col = accumulated / np.maximum(V_col, 1e-300)
    else:
        raise ValueError(f"unknown --zone {args.zone!r}")

    nn0_profile = base_col + add_col
    nn0_annulus_profile = None if add_ann is None else base_ann + add_ann

    # Round-trip particle check: the densities written out must hold the
    # inventory the spread produced, in whichever zone(s) it was routed to.
    if args.two_zone:
        held = float(np.sum(add_col * V_col + add_ann * V_ann))
    else:
        held = float(np.sum(add_col * V_chamber))
    routing_rel = abs(held - grid_in) / max(grid_in, 1e-300)
    assert routing_rel < 1e-10, (
        f"zone routing lost inventory: spread {grid_in:.9e}, held {held:.9e}, "
        f"rel {routing_rel:.3e}"
    )

    ledger = {
        "es": args.es,
        "nx": args.nx,
        "cells": cells,
        "S_gp_sccm": float(params["S_gp"]),
        "gas_puff_valves": int(params["gas_puff_valves"]),
        "gas_puff_profile": params["gas_puff_profile"],
        "gas_puff_z_cm": float(params["gas_puff_z_cm"]),
        "gas_puff_throw_cm": float(params["gas_puff_throw_cm"]),
        "two_zone": bool(args.two_zone),
        "zone": args.zone if args.two_zone else "single-field",
        "base_kind": "uniform" if args.base_from_h5 is None else "profile_h5",
        "base_source": base_source,
        "base_from_h5": args.base_from_h5,
        "base_mean_density_cm3": base_density,
        "base_column_min_cm3": float(np.min(base_col)),
        "base_column_max_cm3": float(np.max(base_col)),
        "base_annulus_min_cm3": (
            None if base_ann is None else float(np.min(base_ann))
        ),
        "base_annulus_max_cm3": (
            None if base_ann is None else float(np.max(base_ann))
        ),
        "Tn_K": Tn_K,
        "vbar_cm_s": vbar,
        "dt_foot_s": float(args.dt_foot_s),
        "dt_foot_bracket_s": list(dt_foot_bracket_s(args.es)),
        "kernel": args.kernel,
        "kernel_bracket": list(KERNELS),
        "kernel_width_cm": width,
        "kernel_width_label": width_label,
        "mfp_cm": mfp,
        "mfp_source": mfp_source,
        "sigma_hehe_cm2": float(args.sigma_hehe_cm2),
        "D_cm2_s": D_cm2_s,
        "throughput_as_applied_per_s": throughput_applied,
        "throughput_nominal_per_valve_per_s": throughput_nominal,
        "injected_atoms_as_applied": injected_applied,
        "injected_atoms_nominal_per_valve": injected_nominal,
        "grid_inventory_before_spread": grid_in,
        "grid_inventory_after_spread": grid_out,
        "spread_conservation_rel": conservation_rel,
        "zone_routing_conservation_rel": routing_rel,
    }
    # The stance overrides, PRESENCE-GATED: an invocation that overrides
    # nothing writes exactly the ledger it always wrote, and so exactly the
    # same output bytes. The inline and file-sourced overrides are recorded
    # separately because they are separate provenance -- a literal value given
    # on the command line, versus a named array in a named file (recorded as
    # source, shape and dtype; the values themselves would bloat the ledger
    # the output npz carries, and they are already IN the output's own grid).
    # The wall-limited member's own parameters and readings, PRESENCE-GATED
    # the same way: a matrix-kernel invocation writes exactly the ledger it
    # always wrote, and so exactly the same output bytes.
    if knudsen_report is not None:
        ledger["knudsen"] = knudsen_report
    if inline_params:
        ledger["extra_params"] = inline_params
    if npz_provenance:
        ledger["extra_params_from_npz"] = npz_provenance
    if extra_flags:
        ledger["extra_flags"] = extra_flags
    return (nn0_profile, nn0_annulus_profile, base_col, base_ann,
            geometry, ledger)


def print_ledger(
    nn0_profile, nn0_annulus_profile, base_col, base_ann, geometry, ledger
):
    """Print the inventory ledger and the profile's headline numbers."""
    z = np.asarray(geometry.z_cm, dtype=float)
    print("=== sp3 shaped-nn0 construction ===")
    print(
        f"stance: ES{ledger['es']} nx={ledger['nx']} cells={ledger['cells']} "
        f"S_gp={ledger['S_gp_sccm']:g} sccm x {ledger['gas_puff_valves']} valves, "
        f"puff {ledger['gas_puff_profile']} at z={ledger['gas_puff_z_cm']:g} cm, "
        f"throw {ledger['gas_puff_throw_cm']:g} cm"
    )
    # Presence-gated exactly as the ledger entries are: an invocation that
    # overrides nothing prints what it always printed.
    for label, key in (
        ("stance overrides [params]", "extra_params"),
        ("stance overrides [flags]", "extra_flags"),
    ):
        if key in ledger:
            print(f"{label}: " + ", ".join(
                f"{k}={v!r}" for k, v in sorted(ledger[key].items())
            ))
    if "extra_params_from_npz" in ledger:
        print("stance overrides [params from npz]: " + ", ".join(
            f"{k}<-{spec['source']} {spec['dtype']}{tuple(spec['shape'])}"
            for k, spec in sorted(ledger["extra_params_from_npz"].items())
        ))
    print(
        f"base [{ledger['base_kind']}]: {ledger['base_source']}; "
        f"column {ledger['base_column_min_cm3']:.6g}..."
        f"{ledger['base_column_max_cm3']:.6g}"
        + (
            ""
            if ledger["base_annulus_min_cm3"] is None
            else f", annulus {ledger['base_annulus_min_cm3']:.6g}..."
            f"{ledger['base_annulus_max_cm3']:.6g}"
        )
        + f"; chamber-mean {ledger['base_mean_density_cm3']:.6g} cm^-3"
    )
    print(
        f"bracket corner: dt_foot={ledger['dt_foot_s']:.6g} s of "
        f"{ledger['dt_foot_bracket_s']}, kernel={ledger['kernel']!r} "
        f"(registered {KERNEL_REGISTERED!r}; legacy matrix "
        f"{ledger['kernel_bracket']})"
    )
    print(
        f"thermal: Tn={ledger['Tn_K']:g} K, vbar={ledger['vbar_cm_s']:.6g} cm/s; "
        f"mfp={ledger['mfp_cm']:.6g} cm ({ledger['mfp_source']}); "
        f"D={ledger['D_cm2_s']:.6g} cm^2/s"
    )
    if ledger["kernel_width_cm"] is None:
        print(f"kernel width: {ledger['kernel_width_label']}")
    else:
        print(
            f"kernel width: {ledger['kernel_width_cm']:.6g} cm "
            f"({ledger['kernel_width_label']})"
        )
    # Presence-gated exactly as the ledger entry is.
    if "knudsen" in ledger:
        k = ledger["knudsen"]
        print(
            f"knudsen operator: member={k['member']!r} "
            f"kappa={k['kappa']:.6g} of "
            f"[{KNUDSEN_KAPPA_SLOW:.6g}, {KNUDSEN_KAPPA_REFERENCE:.6g}, "
            f"{KNUDSEN_KAPPA_FAST:.6g}], "
            f"substeps={k['substeps']} of {k['substep_s']:.6g} s, "
            f"source={k['source_convention']}, "
            f"gap_coupling={k['gap_coupling']}"
        )
        print(
            f"  active cells: {k['active_cells']} "
            f"(mesh {k['active_span_cells'][0]}..{k['active_span_cells'][1]}); "
            f"deposit re-homed into the active set: "
            f"{k['rehomed_fraction']:.6g}"
        )
        print(
            f"  face conductance: {k['face_conductance_min_cm3_s']:.6g}..."
            f"{k['face_conductance_max_cm3_s']:.6g} cm^3/s; "
            f"min cell {k['min_cell_particles']:.6g} atoms; "
            f"inventory rel err {k['conservation_rel']:.3e}"
        )
        print(
            f"  added inventory reach: z50={k['added_z50_cm']:.6g} "
            f"z90={k['added_z90_cm']:.6g} z99={k['added_z99_cm']:.6g} cm"
        )
        print(
            f"  bore step (area ratio {k['bore_step_area_ratio']:.6g}) at "
            f"z={k['bore_step_narrow_z_cm']:.6g}|"
            f"{k['bore_step_wide_z_cm']:.6g} cm: added "
            f"{k['bore_step_narrow_added_cm3']:.6g} / "
            f"{k['bore_step_wide_added_cm3']:.6g} cm^-3 = "
            f"ratio {k['bore_step_added_density_ratio']:.6g}"
        )
        print("  added density at the probe stations [cm^-3]:")
        for station, z_cell, value in k["probe_added_density_cm3"]:
            print(
                f"     z={station:8.2f} (cell z={z_cell:8.2f}): {value:.6g}"
            )
    print("--- inventory ledger ---")
    print(
        f"throughput as-applied (valves in): "
        f"{ledger['throughput_as_applied_per_s']:.6g} /s"
        f"  ->  S_gp x dt_foot = {ledger['injected_atoms_as_applied']:.6g} atoms"
    )
    print(
        f"throughput per-valve-nominal:      "
        f"{ledger['throughput_nominal_per_valve_per_s']:.6g} /s"
        f"  ->  S_gp x dt_foot = {ledger['injected_atoms_nominal_per_valve']:.6g} atoms"
    )
    print(
        f"deposited on grid (first-flight lobe): "
        f"{ledger['grid_inventory_before_spread']:.6g} atoms"
    )
    print(
        f"after spreading:                       "
        f"{ledger['grid_inventory_after_spread']:.6g} atoms  "
        f"(rel err {ledger['spread_conservation_rel']:.3e})"
    )
    print(
        f"held by the written densities:         "
        f"{ledger['grid_inventory_after_spread']:.6g} atoms  "
        f"(rel err {ledger['zone_routing_conservation_rel']:.3e})"
    )
    print("--- profile (enhancement is CELL-LOCAL: profile / base at that cell) ---")
    for label, prof, base in (
        ("column" if ledger["two_zone"] else "nn", nn0_profile, base_col),
        ("annulus", nn0_annulus_profile, base_ann),
    ):
        if prof is None:
            continue
        ratio = np.asarray(prof, dtype=float) / np.asarray(base, dtype=float)
        print(
            f"{label:>8}: min {float(np.min(prof)):.6g}  max "
            f"{float(np.max(prof)):.6g}  mean {float(np.mean(prof)):.6g} cm^-3 "
            f"(peak enhancement x{float(np.max(ratio)):.4g})"
        )
        for z_band in SP1_BAND_Z_CM:
            i = int(np.argmin(np.abs(z - z_band)))
            print(
                f"          z={z[i]:8.2f} cm (sp1 band {z_band:g}): "
                f"{float(prof[i]):.6g} cm^-3 = base x {float(ratio[i]):.6g} "
                f"(base {float(base[i]):.6g})"
            )


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Build the sp3 shaped initial neutral profile npz."
    )
    p.add_argument("--es", type=int, choices=(1, 2, 3), default=1)
    p.add_argument("--nx", type=int, default=PRODUCTION_NX)
    p.add_argument("--sgp", type=float, required=True,
                   help="gas puff level [sccm]; must match the verdict run's")
    p.add_argument("--two-zone", action="store_true",
                   help="build for the neutral_two_zone closure (writes an "
                        "annulus profile as well); must match the run's")
    p.add_argument("--zone", choices=("chamber", "annulus", "column"),
                   default="chamber",
                   help="two-zone routing of the accumulated inventory. "
                        "'chamber' (default) raises both zones by the same "
                        "density -- the radially well-mixed convention, whose "
                        "justification is that the free-molecular zone "
                        "exchange time is ms-class, comparable to the foot. "
                        "'annulus' is the un-mixed extreme and reproduces the "
                        "shipped first-flight routing exactly; 'column' is the "
                        "fully-mixed-inward extreme. The routing is a "
                        "DISCLOSED DEGENERACY (sp2), so a result states it")
    p.add_argument("--base-from-h5", default=None,
                   help="read the base profile from an existing result's t=0 "
                        "frames (column nn, and nn_a for the annulus) instead "
                        "of the uniform resolve_nn0 convention. THE VERDICT-ARM "
                        "BASE: with the sp1 fluid reference here, the base is "
                        "that run's own equilibrated initial profile and the "
                        "arm's single delta is the foot addition on top of it. "
                        "Mutually exclusive with the uniform base by "
                        "construction -- passing this replaces it, and the "
                        "ledger records which was used")
    p.add_argument("--dt-foot-s", type=float, default=None,
                   help="foot duration [s]. DEFAULT: the requested rung's "
                        "REGISTERED foot, the measured circuit-on -> 1 kA "
                        "lead minus the model's own circuit-on -> 1 kA time "
                        "-- "
                        + ", ".join(
                            f"ES{es} {registered_foot_s(es):.6g}"
                            for es in sorted(MEASURED_LEAD_S)
                        )
                        + " s. The registered bracket is that foot +- the "
                        "shot-to-shot sd of the lead ("
                        + ", ".join(
                            f"ES{es} +-{MEASURED_LEAD_SD_S[es]:.6g}"
                            for es in sorted(MEASURED_LEAD_SD_S)
                        )
                        + " s), so passing a value walks the measurement's "
                        "own error bar. 0 is the explicit NULL CONTROL: no "
                        "foot addition at all, so the output is the base "
                        "itself")
    p.add_argument("--kernel", choices=SELECTABLE_KERNELS,
                   default=KERNEL_REGISTERED,
                   help="spreading member. OMITTED builds the REGISTERED "
                        f"member {KERNEL_REGISTERED!r}, the volume-consistent "
                        "finite-volume solve, configured by the --knudsen-* "
                        "options below; it ignores --sigma-hehe-cm2 / "
                        "--mfp-cm, which set the matrix diffusive member's "
                        f"width. {list(KERNELS)} are the LEGACY MATRIX "
                        "kernels, retained to reproduce rows built before the "
                        "finite-volume member was registered")
    p.add_argument("--knudsen-member", choices=sorted(KNUDSEN_MEMBERS),
                   default=None,
                   help="the REGISTERED member of the wall-limited spreading "
                        "coefficient to run: "
                        + ", ".join(
                            f"{name} (kappa {KNUDSEN_MEMBERS[name]:.6g})"
                            for name in sorted(KNUDSEN_MEMBERS)
                        )
                        + f". Omitting both this and --knudsen-kappa runs "
                        f"{KNUDSEN_MEMBER_DEFAULT!r}. Mutually exclusive with "
                        "--knudsen-kappa: the two say the same thing and would "
                        "be free to disagree")
    p.add_argument("--knudsen-kappa", type=float, default=None,
                   help="dimensionless coefficient in the wall-limited cell "
                        "diffusivity D_i = kappa R_i vbar [1], stated as a "
                        "number instead of named with --knudsen-member. This "
                        "is the route to a coefficient the registration does "
                        "not carry; the ledger records such a value as "
                        f"{KNUDSEN_MEMBER_UNREGISTERED!r}")
    p.add_argument("--knudsen-substeps", type=int,
                   default=KNUDSEN_SUBSTEPS_DEFAULT,
                   help="number of equal backward-Euler substeps the foot is "
                        f"integrated over (default {KNUDSEN_SUBSTEPS_DEFAULT}"
                        "). Fixed, never adaptive, so a rebuild of the same "
                        "invocation writes the same bytes")
    p.add_argument("--knudsen-gap-coupling", dest="knudsen_gap_coupling",
                   action="store_true", default=None,
                   help="carry gas through the cathode and gap cells as well, "
                        "coupled to the column through the anode mesh face at "
                        "the transparency the configuration carries. This is "
                        "the REGISTERED setting of the wall-limited member "
                        f"({'ON' if KNUDSEN_GAP_COUPLING_REGISTERED else 'OFF'}"
                        "), so passing it restates the registration rather "
                        "than changing anything")
    p.add_argument("--no-knudsen-gap-coupling", dest="knudsen_gap_coupling",
                   action="store_false",
                   help="leave the region behind the anode mesh uncoupled, so "
                        "the operator places no gas there at all. This is the "
                        "DISCLOSED ALTERNATE to the registered setting above, "
                        "not a lighter default; a row built with it says so")
    p.add_argument("--knudsen-source-convention",
                   choices=KNUDSEN_SOURCE_CONVENTIONS,
                   default=KNUDSEN_SOURCE_CONVENTIONS[0],
                   help="how the deposit enters the solve: 'continuous' "
                        "(default) releases it at a constant rate over "
                        "dt_foot, 'deposit_t0' releases it whole at t = 0, "
                        "which is the matrix kernels' convention and is the "
                        "control a free-space reach check needs")
    p.add_argument("--sigma-hehe-cm2", type=float, default=SIGMA_HE_HE_CM2,
                   help="He-He collision cross section [cm^2] setting the mean "
                        f"free path (default {SIGMA_HE_HE_CM2:g}: "
                        f"{SIGMA_HE_HE_SOURCE})")
    p.add_argument("--mfp-cm", type=float, default=None,
                   help="mean free path [cm] stated directly, bypassing the "
                        "cross section; for auditing the diffusive reach "
                        "against an independently quoted lambda")
    p.add_argument("--tn-k", type=float, default=None,
                   help="neutral temperature [K]; default is the stance Tn_K")
    p.add_argument("--selfcheck", action="store_true",
                   help="NULL-CONSTRUCTION SELF-CHECK: requires --base-from-h5 "
                        "and --dt-foot-s 0. After writing, re-reads the npz "
                        "and the source h5 and asserts the written profiles "
                        "equal the h5's t=0 frames EXACTLY, in every zone. It "
                        "checks the plumbing end to end -- that the base "
                        "really came from the h5, survived the routing, and "
                        "round-tripped through the file -- rather than the "
                        "arithmetic of adding zero")
    p.add_argument("--extra", nargs="*", default=(),
                   help="additional k=v input_dict (params) overrides, "
                        "read and typed by "
                        "extra_overrides.parse_extra_overrides exactly as "
                        "run_m6_point.py --extra reads them: each value takes "
                        "the TYPE its key carries in the configuration "
                        "template. Applied AFTER the whole stance is "
                        "assembled, so the geometry keys (Lm, "
                        "end_wall_length_cm, gas_puff_z_cm, ...) take effect "
                        "everywhere this script reads the config. A value "
                        "that cannot be read as its key's type, and a key "
                        "neither template owns, are refused here; a key filed "
                        "into the wrong namespace still raises at LAPDSim1D "
                        "construction")
    p.add_argument("--extra-flag", nargs="*", default=(),
                   help="additional k=v input_flags overrides, as "
                        "run_m6_point.py --extra-flag; same ordering and read "
                        "by the same parse layer as --extra, where every flag "
                        "key carries bool, so these take true or false")
    p.add_argument("--extra-npz", nargs="*", default=(),
                   help="array-valued params override, KEY=path.npz:arrayname "
                        "-- reads the named array out of the named .npz and "
                        "files it under KEY, e.g. "
                        "plasma_radius_profile_cm=scripts/g1_profiles.npz:"
                        "plasma_radius_profile_cm_off. This is the route for "
                        "the per-mesh-cell geometry profiles, which are "
                        "hundreds of numbers and do not belong in argv; a 0-d "
                        "entry in the .npz becomes the scalar itself, so one "
                        "file can carry a whole geometry. Applied BEFORE "
                        "--extra, so an inline value still overrides a "
                        "file-sourced one")
    p.add_argument("--out", required=True, help="output .npz path")
    args = p.parse_args(argv)

    # The default foot is the requested rung's registered one, so it cannot be
    # an argparse default (it is not known until --es is read).
    if args.dt_foot_s is None:
        args.dt_foot_s = registered_foot_s(args.es)

    if args.dt_foot_s < 0.0 or not math.isfinite(args.dt_foot_s):
        p.error("--dt-foot-s must be finite and >= 0 (0 is the null control)")
    if args.zone != "chamber" and not args.two_zone:
        p.error("--zone is a two-zone routing choice; pass --two-zone or drop it")
    # The --knudsen-* options configure ONE member. Under a matrix kernel they
    # would be silent inert controls, which is exactly the class of mistake
    # this campaign refuses at construction rather than discovering in a
    # ledger.
    if args.kernel != KNUDSEN_KERNEL and (
        args.knudsen_member is not None
        or args.knudsen_kappa is not None
        or args.knudsen_substeps != KNUDSEN_SUBSTEPS_DEFAULT
        or args.knudsen_gap_coupling is not None
        or args.knudsen_source_convention != KNUDSEN_SOURCE_CONVENTIONS[0]
    ):
        p.error(
            f"the --knudsen-* options configure the {KNUDSEN_KERNEL!r} "
            f"member and are inert under --kernel {args.kernel}; pass "
            f"--kernel {KNUDSEN_KERNEL} or drop them"
        )
    # THE MEMBER. A registered name and an explicit coefficient are two
    # spellings of one quantity, so taking both would let an invocation carry
    # two answers; taking neither is the reference member.
    if args.knudsen_member is not None and args.knudsen_kappa is not None:
        p.error(
            "--knudsen-member and --knudsen-kappa both state the spreading "
            "coefficient; pass one. "
            f"--knudsen-member {args.knudsen_member} is kappa "
            f"{KNUDSEN_MEMBERS[args.knudsen_member]:.6g}"
        )
    if args.knudsen_kappa is None:
        member = args.knudsen_member or KNUDSEN_MEMBER_DEFAULT
        args.knudsen_kappa = KNUDSEN_MEMBERS[member]
    if not (math.isfinite(args.knudsen_kappa) and args.knudsen_kappa > 0.0):
        p.error("--knudsen-kappa must be finite and > 0")
    if args.knudsen_substeps < 1:
        p.error("--knudsen-substeps must be at least 1")
    # GAP COUPLING. Neither switch passed takes the member's REGISTERED
    # setting, which is a decision this module owns rather than an argparse
    # default that happens to be convenient.
    if args.knudsen_gap_coupling is None:
        args.knudsen_gap_coupling = KNUDSEN_GAP_COUPLING_REGISTERED
    if args.selfcheck and (
        args.base_from_h5 is None or args.dt_foot_s != 0.0
    ):
        p.error(
            "--selfcheck is the null construction: it requires "
            "--base-from-h5 and --dt-foot-s 0"
        )

    nn0_profile, nn0_annulus_profile, base_col, base_ann, geometry, ledger = (
        build(args)
    )
    print_ledger(
        nn0_profile, nn0_annulus_profile, base_col, base_ann, geometry, ledger
    )

    payload = {
        "nn0_profile": nn0_profile,
        "z_cm": np.asarray(geometry.z_cm, dtype=float),
        "provenance": json.dumps(ledger, sort_keys=True),
    }
    if nn0_annulus_profile is not None:
        payload["nn0_annulus_profile"] = nn0_annulus_profile
    np.savez(args.out, **payload)
    print(f"saved {args.out}")

    if args.selfcheck:
        source = load_result_hdf5(args.base_from_h5)
        with np.load(args.out, allow_pickle=False) as written:
            checks = [(
                "column",
                np.asarray(written["nn0_profile"], dtype=float),
                np.asarray(source.nn[0], dtype=float),
            )]
            if "nn0_annulus_profile" in written:
                checks.append((
                    "annulus",
                    np.asarray(written["nn0_annulus_profile"], dtype=float),
                    np.asarray(source.nn_a[0], dtype=float),
                ))
            ok = True
            for label, got, want in checks:
                same = np.array_equal(got, want)
                ok = ok and same
                print(
                    f"null-construction self-check [{label}]: "
                    f"{'EXACT' if same else 'DIFFERS'} vs "
                    f"{args.base_from_h5} t=0 ({got.size} cells, "
                    f"max |delta| {float(np.max(np.abs(got - want))):.3e})"
                )
        print("SELFCHECK:", "PASS" if ok else "FAIL")
        if not ok:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
