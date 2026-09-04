"""Transient deterministic velocity-grid neutral engine (K2a).

The LIVE-transient promotion of the steady generation solver in
``kinetic_neutrals.py``: instead of iterating unit-source responses to a
steady state, this module carries accepted-state distributions

    f_c(z, v_z, v_perp)   column  [cm^-3 per bin, on the column volume]
    f_a(z, v_z, v_perp)   annulus [cm^-3 per bin, on the annulus volume]

and advances them ONE step per neutral-clock tick with the same implicit
upwind march, the same sinh-stretched shared ``VGrid``, the same
moment-exact Maxwellian projection and the same cosine-wall re-emission
spectrum. The ANNULUS zone is where this module offers its choices:
``exchange_model`` selects between that same Cauchy-chord rate form and a
geometrically derived one, and ``annulus_flights`` selects whether the
annulus is carried by those rates at all or by the bounded-chord flight
classes of ``KN2ZoneJump`` (see :class:`TransientDVM` and
:class:`BoundedChordFlights`); both default to the shipped forms. Nothing
here is a second implementation of the velocity grid or of the transport
sweep -- the operators are imported or transcribed from that module so the
offline instruments and the in-solver arm keep agreeing on inputs.

Time discretization (split implicit; first order, unconditionally stable,
positivity preserving):

  A. transport + every LOSS process, fully implicit -- one backward-Euler
     upwind march per sign of ``v_z``, with the 2x2 column/annulus zone
     coupling solved exactly per (cell, bin) as in ``KN2Zone.sweep``. The
     march diagonal carries ``1/dt`` and the right-hand side ``f^n/dt``;
     every loss channel is then tallied from ``f^{n+1}``, so the tallies
     are the losses the update actually took. Under
     ``annulus_flights = "bounded_chord"`` the march is the COLUMN's alone
     -- the annulus has no advected row there -- and the annulus's own
     substep A is the flights that complete this tick, taken implicitly at
     the same backward-Euler weight.
  B. BIRTHS, at masses exactly equal to the substep-A tallies. Charge
     exchange and elastic scattering re-emit their own losses at the local
     ion Maxwellian, the cylindrical wall re-emits its own losses (split
     accommodated/reflected), the anode mesh re-emits what it intercepted,
     the interior CLOSED FACES re-emit what they blocked on the side it
     arrived from, and the external ledger (puff, recombination, recycling
     faces, anode rebirths) is added as counted particles. With the CATHODE
     JET armed the counted cathode recycle splits here: the ``R_N``
     backscattered share is born as a directed energetic volume birth
     carrying exactly the ``R_E`` energy share its surface was debited, and
     the remainder keeps the thermal face inflow. With the ANODE JET armed
     the counted anode collection splits the same way, directed AWAY from
     the mesh on the side it was collected from; what the mesh INTERCEPTED
     is untouched by that split and keeps its at-rest re-emission.

Splitting births from losses this way is what makes the inventory ledger
EXACT rather than converged: substep A never creates a particle and
substep B creates exactly the number substep A destroyed, per channel, so
the domain total closes to roundoff at every update regardless of dt.

The IONIZATION channel is the one the coupled solver shares with a fluid
partner, and it is settled in COUNTED PARTICLES, not in a rate (K2e). The
partner hands ``update`` the count it booked as ionization over the tick;
that count is what this engine debits, taken from the post-march column
population in proportion to it, with any part a cell cannot surrender
carried as ``ion_debt`` and re-offered next tick. So particle
conservation across the fluid/kinetic handshake is EXACT BY CONSTRUCTION
and independent of the neutral-clock cadence: the coupled system creates
no particle, whatever the tick length.

That is a conservation statement and nothing more. The ACCURACY of the
ionization rate is a separate question and is unchanged: the fluid rate
is evaluated against a column density that only moves when this engine
republishes it, so the rate itself remains first-order accurate in the
cadence. Shortening the cadence still buys rate accuracy; it no longer
buys conservation, because conservation is no longer approximate.

EVERY surface return enters as a directed boundary INFLOW ghost density at
the face it came off, never as a volume source in the adjacent cell,
because only the inflow form preserves an equilibrium Maxwellian exactly
(the returned flux spectrum divided by ``|v_z| A`` reproduces the volume
Maxwellian bin for bin) and only it makes the returning population
TRAVEL: a volume birth deposits its whole mass inside one cell for a
whole tick, which at a plasma-terminating surface re-ionizes on the spot
and re-ignites the cell it was supposed to be draining. That applies to
the end walls and, since K2d, to the cathode/collector recycle channels,
whose faces are interior whenever something sits behind the surface.

INTERIOR CLOSED FACES are the solid obstructions standing inside the
neutral domain: the geometry marks a face closed wherever a plasma-dead
cell (plenum, obstruction) abuts a live one, and the cathode disc against
its plenum is the canonical one. The COLUMN zone is the disc's own
footprint -- the column radius is the plasma radius, which is the cathode
radius -- so the column carries ZERO transmission across such a face:
whatever reaches it is blocked, tallied, and re-emitted as a cosine
half-flux into the side it arrived from, at that side's surface
temperature (the live cathode surface temperature on a cathode-adjacent
side, the wall temperature otherwise). The ANNULUS is deliberately NOT
blocked: it is the clear bore around the disc, and it is the only route by
which the machine's gas reaches the pump standing on the plenum behind the
cathode. Blocking is applied BEFORE the recycle ghost at the same face, so
a surface still emits into the plasma on the plasma side of itself rather
than into the obstruction behind it.

End walls remain the one LAGGED channel: their outflow is known only
after the march, so the returning particles are held in a per-end pending
buffer and injected on the next update. The buffer is part of the
inventory: closure is stated over ``sum(f V) + pending``. The recycle
faces are not lagged -- the plasma tells the arm what it removed before
the march runs, so those particles are injected in the same update.

Sign and unit conventions: CGS throughout, distributions in cm^-3 per
bin (the bin sum IS the density), ledger entries in PARTICLES (not
rates), momentum tallies in g cm/s, energy tallies in erg. The
plasma-coupling arrays the solver reads are per-second densities on the
PLASMA volume, which is exactly the column volume.
"""

import numpy as np
from scipy.special import erf as _erf

from cablp.atomic.cross_sections import (
    phelps_he_backscatter_cm2,
    phelps_he_isotropic_cm2,
)

from ..core.geometry import absorbing_live_cells_by_role
from .kinetic_neutrals import (
    EV,
    KB,
    M_HE,
    T_WALL_K,
    VGrid,
    annulus_chord_classes,
    ion_thermal_g_eff_floor_cm2_s2,
)
from .neutrals import neutral_zone_volumes


ELASTIC_MODELS = ("phelps_iso", "off")

# How the ANNULUS zone's wall-interaction and radial-exchange flights are
# taken. ``"rates"`` is the shipped algebraic-rate treatment (``nuw`` /
# ``nuxp`` in the implicit march); ``"bounded_chord"`` replaces it with the
# deterministic bounded-chord jump kernel. See ``TransientDVM``.
ANNULUS_FLIGHT_MODELS = ("rates", "bounded_chord")

# The three bounded-chord flight classes, in the order the kernel stores
# them: outer wall -> outer wall, outer wall -> inner surface (the a->c
# crossing), inner surface -> outer wall (the c->a escapes' flight).
FLIGHT_CLASSES = ("ww", "wi", "io")

# Column<->annulus zone-exchange closures. Both are algebraic rates on the
# same (cell, v_perp) index and both impose the antisymmetry through the
# actual geometry volumes; they differ in the mean chord and in how one
# surface encounter is split between the two cylinders. See
# ``TransientDVM`` for the expressions and
# ``scripts/k2_dvm_exchange_measured.txt`` for the measurement.
EXCHANGE_MODELS = ("cauchy_chord", "geometric")

# What the NON-accommodated share does at the CYLINDRICAL wall. ``"specular"``
# returns it in its incident bin (on this axisymmetric grid a specular
# reflection reverses only the unresolved radial component, so the bin is
# unchanged); ``"diffuse_elastic"`` returns it on a cosine-wall spectrum
# whose temperature parameter is solved so that the spectrum's DISCRETE mean
# energy equals the retained share's incident mean, which randomizes the
# direction while exchanging no energy with the surface. The first entry is
# this ENGINE's kwarg default, used only by direct ``TransientDVM``
# construction (test fixtures and probes); the PACKAGE default a solver-built
# run resolves is owned by ``core/config.py`` and is not this tuple's order.
# The two have differed since the 2026-08-30 adoption. The END plates keep
# the specular mirror under both
# values -- an end-wall ``v_z`` mirror already fully accommodates the
# directed axial momentum -- and the anode mesh and the interior closed faces
# are fully accommodating by construction and never read this. See
# ``TransientDVM._wall_return_counts``.
WALL_REFLECTION_MODELS = ("specular", "diffuse_elastic")

# Guards on the ``"diffuse_elastic"`` one-parameter re-emission solve. The
# discrete mean energy of the cosine-wall spectrum increases monotonically
# with the thermal speed and SATURATES once the spectrum outruns the velocity
# grid's outermost bins, so a target above that saturation value has no
# solution at all: the bracket search and the bisection both carry a hard
# iteration cap and RAISE when they hit it rather than returning a spectrum
# at the wrong energy.
WALL_ENERGY_SOLVE_MAX_BRACKET = 60
WALL_ENERGY_SOLVE_MAX_ITERS = 200
#: Relative agreement the solved spectrum's discrete mean energy must reach.
WALL_ENERGY_SOLVE_REL_TOL = 1.0e-12

# Controls of the SECANT tier of the same solve. It runs first and the
# bisection above is its fallback, so these bound the fast path only: a
# non-converged secant costs the bisection's own evaluations on top of its
# own and returns exactly what the bisection returns.
#: Residual bar, in ``|ln(<E> / e_bar)|`` and so in relative mean-energy miss,
#: the secant must reach over EVERY cell of a call before its speeds are used.
#: Two decades inside ``WALL_ENERGY_SOLVE_REL_TOL``, so a converged secant
#: cannot be the reason the final agreement check fires.
WALL_ENERGY_SECANT_REL_TOL = 1.0e-14
#: Iteration cap on the secant. Reaching it hands the whole call to the
#: bisection, which is how a saturated target reaches its refusal.
WALL_ENERGY_SECANT_MAX_ITERS = 20
#: Largest ``ln s`` step one secant iteration may take, so a flat or
#: ill-conditioned residual cannot march the thermal speed out of range
#: before the iteration cap notices it is not converging.
WALL_ENERGY_SECANT_MAX_STEP = 1.0

# How the PLASMA applies the tick-booked CX/elastic transfer between neutral
# clock ticks. The pair is a linear relaxation of the fluid momentum and ion
# energy rows towards the lost-population moments at the per-ion collision
# frequency, so "exponential" integrates it exactly over each plasma step at
# the tick's frozen (nu, target) and is unconditionally stable; "zoh" freezes
# the booked RATE instead and is oscillatory-unstable for nu*dt_tick > 2. The
# first entry is the resolved default. See the solver's transfer-hold scope
# and NUMERICS.md.
TRANSFER_HOLDS = ("exponential", "zoh")

# Relative agreement the CATHODE JET's launch spectrum must reach on its
# DISCRETE drift and energy moments before it is placed. The spectrum is a
# monoenergetic beam smeared onto the velocity grid, and the whole point of
# the channel is that the energy it hands the gas equals the energy the
# cathode surface was debited: a projection that misses its moments has
# created or destroyed energy between the two books. The tolerance is the
# convergence tolerance of ``VGrid.maxwellian``'s own compensation loop, so a
# converged solve passes by construction and a bailed one -- a conditioning
# rejection, or four iterations that did not reach the target -- RAISES.
CATHODE_JET_MOMENT_REL_TOL = 1.0e-10

# The same tolerance, and the same statement, for the ANODE JET: its launch
# spectrum's discrete moments must reach their targets before it is placed,
# because the energy it hands the gas is cross-booked against the anode energy
# book's ``backscatter`` row.
ANODE_JET_MOMENT_REL_TOL = 1.0e-10

# Rate factor on the isotropic-elastic BGK channel. A full-replacement event
# transfers m (v - u_i), which is twice the isotropic angular average
# ``mu <1 - cos th> g = m g / 2`` at equal mass; halving the collision rate
# restores the correct mean momentum (and energy) transfer per unit time. It
# is the equal-mass reduced-mass ratio ``mu/m = 1/2``, not a fitted number.
# See ``TransientDVM.collision_frequencies``.
ELASTIC_BGK_MOMENTUM_FACTOR = 0.5

# Every channel the ledger books, as (birth, loss) name pairs plus the
# one-sided channels. Named here so the verification script and the smoke
# suite can assert the ledger is complete rather than re-listing them.
LEDGER_LOSS_CHANNELS = (
    "ionization",
    "cx",
    "elastic",
    "wall",
    "mesh_blocked",
    # B6: the annulus flux a thin annular baffle intercepts at its face. Its
    # partner ``birth_baffle_reemit`` is the same count re-emitted on the side
    # it was intercepted from, so the pair cancels in the domain identity the
    # way the mesh pair does. Both rows are always PRESENT and are exactly
    # zero with no baffle armed; what is presence-gated is the code path, and
    # the momentum diagnostic below.
    "baffle_blocked",
    "closed_face_blocked",
    "end_out_L",
    "end_out_R",
)
LEDGER_BIRTH_CHANNELS = (
    "cx",
    "elastic",
    "wall_accommodated",
    "wall_reflected",
    "mesh_reemit",
    "baffle_reemit",
    "closed_face_reemit",
    "end_return_L",
    "end_return_R",
    "puff",
    "recombination",
    "cathode_face",
    "cathode_jet",
    "collector_face",
    "anode",
    "anode_jet",
)
# Losses that leave the modelled system entirely (the rest are internal
# and are paired with a birth of exactly equal mass).
LEDGER_EXTERNAL_LOSSES = ("ionization", "pump_L", "pump_R")
LEDGER_EXTERNAL_BIRTHS = (
    "puff",
    "recombination",
    "cathode_face",
    "cathode_jet",
    "collector_face",
    "anode",
    "anode_jet",
)
# External births the ENGINE splits out of another channel rather than
# receiving as a ledger entry of its own. ``cathode_jet`` is the ``R_N``
# energetic-backscatter share of ``cathode_face`` and ``anode_jet`` is the same
# share of ``anode``: the partner counts each stream ONCE, under the parent
# channel, and the split happens here. A caller naming one of these in
# ``sources``/``source_counts`` would be handing particles nothing reads, so
# :meth:`TransientDVM._check_source_channels` refuses it.
LEDGER_ENGINE_SPLIT_BIRTHS = ("cathode_jet", "anode_jet")
# PRESENCE-GATED momentum diagnostics [g cm/s per tick], emitted only when the
# anode jet is armed. They are not a momentum ledger and nothing closes over
# them: they are two named readings of the plan's "launch momentum booked
# against the surface / wire-intercepted momentum on the structure", made
# measurable rather than remembered. They are deliberately NOT named
# ``birth_*`` or ``loss_*``, because :func:`ledger_residual` sums every row
# with those prefixes and these are not particle counts.
#
# ``momentum_baffle_absorbed`` (B6) is the same kind of reading for the thin
# annular baffles and is presence-gated on THEM rather than on the anode jet:
# it is emitted exactly when at least one baffle face is armed, which is when
# there is a structure for the gas to hand axial momentum to. Like the mesh
# row it is what the structure KEPT -- the baffle re-emits at rest on the wall
# spectrum, so every bit of the intercepted axial momentum stays on it.
LEDGER_MOMENTUM_DIAGNOSTICS = (
    "momentum_anode_jet",
    "momentum_mesh_absorbed",
    "momentum_baffle_absorbed",
)
# Ledger entries that are NOT channels: the tick's own bookkeeping, the
# counted-ionization handshake's reconciliation record, and the nested
# ENERGY ledger declared below. Named here for the same reason as the
# channel tuples -- the completeness gate asserts that every key is either a
# declared channel or a declared non-channel, so a new entry cannot slip in
# unnoticed on either side.
LEDGER_BOOKKEEPING = (
    "dt",
    "inventory_before",
    "inventory_after",
    "f_inventory_before",
    "f_inventory_after",
    "ion_booked",
    "ion_debt_carried",
    "ion_limited_cells",
    "energy",
)

# The PARTICLE rows of the ledger above, in the order the saved
# ``dvm_particle_ledger`` group carries them. FLOW rows are counts of ATOMS
# that passed through a channel during one tick and are therefore additive
# over any span of ticks; STATE rows are counts of ATOMS standing at the
# instant the tick ended and are not. The remaining ledger entries are none of
# these -- a time, a cell count, a signed momentum, the nested energy ledger --
# and are excluded from both.
#
# Membership is DERIVED from the channel tuples above rather than restated, so
# a channel added there is exported without a second edit and cannot become a
# silently unexported row.
LEDGER_PARTICLE_FLOW_KEYS = (
    tuple(f"loss_{name}" for name in LEDGER_LOSS_CHANNELS)
    + ("loss_pump_L", "loss_pump_R")
    + tuple(f"birth_{name}" for name in LEDGER_BIRTH_CHANNELS)
    + ("ion_booked",)
)
LEDGER_PARTICLE_STATE_KEYS = ("inventory", "ion_debt_carried")
# The saved group's own row order: the frame's time and how many ticks it
# covers, then the flow rows, then the state rows.
LEDGER_PARTICLE_FRAME_KEYS = (
    ("time", "ticks") + LEDGER_PARTICLE_FLOW_KEYS + LEDGER_PARTICLE_STATE_KEYS
)
#: What each row of that group is, as ``name -> (unit, meaning)``. It is the
#: text the saved file carries in its own attributes, so a reader of the
#: artifact needs neither this module nor the plan that asked for it.
LEDGER_PARTICLE_ROW_DOC = {
    "time": ("s", "elapsed time at this save frame"),
    "ticks": (
        "ticks",
        "neutral clock ticks this frame's flow rows sum over; zero on a frame "
        "no tick fell in, whose flow rows are then zero because nothing "
        "happened rather than because nothing was recorded",
    ),
    "loss_ionization": (
        "atoms",
        "column neutrals the marched ionization operator destroyed",
    ),
    "loss_cx": (
        "atoms", "neutrals charge exchange removed; returned as birth_cx"
    ),
    "loss_elastic": (
        "atoms",
        "neutrals the elastic operator removed; returned as birth_elastic",
    ),
    "loss_wall": (
        "atoms",
        "landings on the cylindrical wall; returned as "
        "birth_wall_accommodated + birth_wall_reflected",
    ),
    "loss_mesh_blocked": (
        "atoms",
        "flux the anode mesh intercepted; returned as birth_mesh_reemit",
    ),
    "loss_baffle_blocked": (
        "atoms",
        "annulus flux an annular baffle face intercepted; returned as "
        "birth_baffle_reemit",
    ),
    "loss_closed_face_blocked": (
        "atoms",
        "flux an interior closed face stopped; returned as "
        "birth_closed_face_reemit",
    ),
    "loss_end_out_L": ("atoms", "flux crossing the left end plane"),
    "loss_end_out_R": ("atoms", "flux crossing the right end plane"),
    "loss_pump_L": (
        "atoms", "the pumped share of loss_end_out_L; leaves the domain"
    ),
    "loss_pump_R": (
        "atoms", "the pumped share of loss_end_out_R; leaves the domain"
    ),
    "birth_cx": ("atoms", "charge-exchange neutrals born on the ion spectrum"),
    "birth_elastic": (
        "atoms",
        "elastically scattered neutrals re-emitted on the ion spectrum",
    ),
    "birth_wall_accommodated": (
        "atoms",
        "the accommodated share of loss_wall, re-emitted at the wall "
        "temperature",
    ),
    "birth_wall_reflected": (
        "atoms", "the non-accommodated share of loss_wall"
    ),
    "birth_mesh_reemit": ("atoms", "loss_mesh_blocked re-emitted by the mesh"),
    "birth_baffle_reemit": (
        "atoms", "loss_baffle_blocked re-emitted by the baffle face"
    ),
    "birth_closed_face_reemit": (
        "atoms", "loss_closed_face_blocked re-emitted by the closed face"
    ),
    "birth_end_return_L": (
        "atoms", "the left end plane's lagged return buffer released"
    ),
    "birth_end_return_R": (
        "atoms", "the right end plane's lagged return buffer released"
    ),
    "birth_puff": ("atoms", "the configured gas puff"),
    "birth_recombination": ("atoms", "neutrals from volume recombination"),
    "birth_cathode_face": (
        "atoms",
        "thermal recycle at the cathode face, less the birth_cathode_jet "
        "share split out of it",
    ),
    "birth_cathode_jet": (
        "atoms",
        "the energetic backscatter share split out of the cathode face",
    ),
    "birth_collector_face": ("atoms", "recycle at the collector face"),
    "birth_anode": (
        "atoms",
        "thermal recycle at the anode, less the birth_anode_jet share split "
        "out of it",
    ),
    "birth_anode_jet": (
        "atoms", "the energetic backscatter share split out of the anode"
    ),
    "ion_booked": (
        "atoms",
        "the ionization count the plasma handed the engine to debit, which "
        "loss_ionization is the marched part of",
    ),
    "inventory": (
        "atoms",
        "domain inventory at this frame, including the lagged end-return "
        "buffers",
    ),
    "ion_debt_carried": (
        "atoms",
        "ionization the plasma booked that the kinetic state has not yet "
        "surrendered, standing at this frame",
    ),
}

# The ENERGY ledger, carried in the ``"energy"`` entry of every ledger the
# engine emits, in erg. Every entry is the kinetic-energy moment
# ``sum(counts * (1/2) m |v|^2)`` of exactly the per-bin particle array its
# namesake in the particle ledger counted, so the two ledgers close by the
# same algebra and to the same exactness class: the march is bin-diagonal,
# the zone coupling moves particles at fixed bin, and the per-bin energy
# weight is a constant that therefore passes straight through both.
#
# The loss rows are the particle ones plus the two pumped fractions (the
# energy that leaves with what the end planes stick); the birth rows are the
# particle ones unchanged.
LEDGER_ENERGY_LOSS_CHANNELS = LEDGER_LOSS_CHANNELS + ("pump_L", "pump_R")
LEDGER_ENERGY_BIRTH_CHANNELS = LEDGER_BIRTH_CHANNELS
# The NET rows have no particle analog, and they are the reason an energy
# ledger is a separate statement: an internal particle channel returns
# exactly what it took and cancels, an internal ENERGY channel does not. Each
# ``surface_*`` row is (what the surface received) - (what it gave back), the
# energy the gas hands to that surface over the tick; each ``exchange_*`` row
# is the same quantity for the ion population the collisional channels
# re-emit against. ``surface_end_*`` counts only what the end wall KEPT: the
# pumped share is its own loss row and the buffered return is still inside
# the inventory. With these rows present every internal channel is accounted
# and the domain identity closes.
LEDGER_ENERGY_NET_CHANNELS = (
    "surface_wall",
    "surface_mesh",
    "surface_baffle",
    "surface_closed_face",
    "surface_end_L",
    "surface_end_R",
    "exchange_cx",
    "exchange_elastic",
)
LEDGER_ENERGY_BOOKKEEPING = (
    "inventory_before",
    "inventory_after",
    "f_inventory_before",
    "f_inventory_after",
    "pending_after_L",
    "pending_after_R",
)


class TransientDVM:
    """Live transient two-zone velocity-grid neutral state.

    ``accommodation`` is the thermal accommodation coefficient of the
    stainless-steel surfaces: the accommodated fraction is re-emitted
    cosine-distributed at the wall temperature, the remaining fraction is
    reflected at the incident energy. On the axisymmetric ``(v_z,
    v_perp)`` grid a specular reflection off the cylindrical wall reverses
    only the radial component of ``v_perp`` and so is bin-preserving; off
    an end wall it reverses ``v_z``, which is the exact bin mirror of the
    symmetric stretched axis. Both are therefore represented exactly, with
    no re-projection error.

    ``wall_reflection`` selects what the NON-accommodated ``(1 - alpha)``
    share does at the CYLINDRICAL wall, and nowhere else: the end plates
    keep the ``v_z`` mirror under both values (that mirror already fully
    accommodates the directed axial momentum), and the anode mesh and the
    interior closed faces re-emit everything they intercept at a surface
    temperature, so they are fully accommodating by construction and do not
    read this at all.

    ``"specular"``
        the bin-preserving reflection described above.

    ``"diffuse_elastic"``
        the same count, per cell, returned on a cosine-wall spectrum whose
        temperature parameter is solved so that the spectrum's DISCRETE mean
        energy equals the retained share's own incident mean energy per
        atom. Count and energy are therefore both exact and the return
        carries zero net axial momentum: the surface randomizes the
        direction and exchanges no energy, which is the elastic-diffuse
        limit. The solve is a bracketed bisection on a monotone function and
        RAISES rather than falling back when the target lies outside what
        the velocity grid can re-emit (see
        :meth:`_solve_wall_return_spectra`).

    The two degenerate at ``alpha == 1``, where there is no share to place.

    ``elastic_model`` selects the polarization-elastic channel:
    ``"phelps_iso"`` adds a BGK-like relaxation toward the local ion
    Maxwellian at HALF the Phelps isotropic rate (the isotropic
    momentum-transfer average -- see :meth:`collision_frequencies` for why
    the half is there), ``"off"`` drops it (charge exchange then carries
    all ion-neutral momentum transfer). The elastic channel exists because
    the arm supersedes the fluid ion-neutral collision family wholesale
    and the fluid operator's momentum-transfer cross section is
    ``Qi + 2 Qb``; carrying only ``Qb`` would silently drop the ``Qi``
    half.

    ``exchange_model`` selects the column<->annulus zone-exchange closure,
    i.e. the per-``(cell, v_perp)`` frequencies at which a neutral crosses
    ``r = Rp`` in either direction and strikes ``r = Rm``:

    ``"cauchy_chord"``
        the three-dimensional Cauchy mean chord ``4V/S = 2 (Rm - Rp)``
        evaluated at the perpendicular speed, with the encounter split
        between the two cylinders as ``Rp/Rm : (1 - Rp/Rm)``::

            nu_total = vp / (2 (Rm - Rp))
            nu_a->c  = (Rp/Rm) nu_total        nu_a->wall = (1 - Rp/Rm) nu_total

    ``"geometric"``
        the mean chord of the cell CROSS-SECTION, ``pi A / P``, since the
        crossings of two coaxial cylinders are decided entirely by the
        motion in the ``(x, y)`` plane, with the encounter split between
        the two circles in proportion to their PERIMETERS::

            nu_total = 2 vp / (pi (Rm - Rp))
            nu_a->c  = 2 vp Rp / (pi (Rm^2 - Rp^2))
            nu_a->wall = 2 vp Rm / (pi (Rm^2 - Rp^2))
            nu_c->a  = 2 vp / (pi Rp)

        Averaged over a Maxwellian this ``nu_c->a`` is ``vbar / (2 Rp)``,
        the free-molecular column loss rate the fluid arm's
        :func:`~cablp.solvers._sim1d.physics.neutrals.neutral_zone_exchange_conductance`
        carries.

    Both branches impose ``V_col nu_c->a == V_ann nu_a->c`` through the
    actual cell volumes, so the ledger's zone channel cancels exactly
    either way. Any other value raises.

    ``annulus_flights`` selects how the ANNULUS zone's wall interaction and
    radial exchange are taken:

    ``"rates"``
        the algebraic rates above, in the implicit march: ``nuw`` removes
        annulus particles to the wall and ``nuxp`` to the column, and the
        annulus is advected axially by the same upwind sweep as the column.
        The flight-time distribution each rate implies is exponential.

    ``"bounded_chord"``
        the annulus is carried as three DETERMINISTIC flight classes
        instead (:class:`BoundedChordFlights`): a wall launch reaches the
        inner surface with the view factor ``Rp/Rm`` at the class-mean
        chord ``c_wi`` and the wall otherwise at ``c_ww``, and a column
        escape reaches the wall at ``c_io``. Each flight displaces the atom
        axially by exactly ``v_z c / v_perp`` and lasts ``c / v_perp``; the
        chords are derived numerically from the local ``(Rp, Rm)`` and
        carry no free parameter. The annulus is then NOT advected by the
        march -- the flight jump is its whole axial motion -- so ``f_a``
        becomes the sum of the three in-flight populations and the march
        runs on the column alone, with the arriving inner-surface landings
        entering as a volume source and the column's escapes leaving on the
        same diagonal rate the rate arm uses. ``exchange_model`` therefore
        still sets the column's ``nu_c->a`` escape rate under this branch;
        it is the annulus-side ``nuw`` and ``nuxp`` that the chord classes
        replace.

    Any other value raises.

    ``cathode_jet`` arms the CATHODE-SIDE ENERGETIC RECYCLE, and is ``None``
    (absent) by default: with it absent nothing below runs and every
    ``cathode_face`` particle keeps the thermal re-emission it has always
    had. Armed, it is the spec dict ``{"R_N", "R_E", "T_launch_eV"}`` and
    splits the counted ``cathode_face`` channel in two:

    * the ``R_N`` share is BACKSCATTERED, born as a volume birth in the cell
      the recycle was counted into, on a moment-compensated narrow shifted
      Maxwellian carrying ``(R_E/R_N) (phi_c + Ti)`` of kinetic energy per
      atom -- the ``"total_reflected"`` convention of
      :func:`~cablp.solvers._sim1d.physics.sources.cathode_jet_backscatter_speed`,
      so the DVM channel and the fluid one describe the same atoms;
    * the ``1 - R_N`` remainder keeps the thermal cosine half-flux inflow at
      the live surface temperature, unchanged.

    ``R_E`` is the TOTAL reflected energy fraction, so the whole channel
    carries ``R_E`` of the incident ion energy and the surface must be
    debited exactly that. The incident energy is not derived here: it is
    handed to :meth:`update` as ``cathode_jet_incident_erg``, the SAME
    committed number the partner debits its surface by, which is what makes
    the reflected energy created once rather than twice.

    ``T_launch_eV`` is a NUMERICS parameter -- the width of the smear the
    monoenergetic beam is represented by on a discrete velocity grid.
    ``None`` ties it to the grid: ``m dv_z(v_back)^2 / k_B``, one bin at the
    launch speed, which is the narrowest spectrum the grid resolves there. A
    positive float pins it instead. The drift is then solved from the
    ENERGY: ``u = sqrt(v_back^2 - 3 k T_launch / m)``, so the spectrum's
    discrete mean energy is ``(1/2) m v_back^2`` exactly rather than that
    plus the smear's own ``(3/2) k T_launch``. A projection that misses its
    moments RAISES (:meth:`_cathode_jet_launch_spectrum`).

    ``anode_jet`` is the same form for the ANODE-SIDE ENERGETIC RECYCLE, and
    is likewise ``None`` (absent) by default. Armed, it splits the counted
    ``anode`` channel -- the ions the mesh collects and neutralizes -- into an
    ``R_N`` backscattered share and the thermal remainder, on the same
    ``"total_reflected"`` convention as its cathode twin (here
    :func:`~cablp.solvers._sim1d.physics.sources.anode_jet_backscatter_speed`,
    with ``phi_a`` in place of ``phi_c``) and against the same handed-in
    committed energy, ``anode_jet_incident_erg``. Three things differ, all of
    them because the mesh is not a disc:

    * the launch is DIRECTED AWAY FROM THE MESH on the side it was collected
      from -- ``-z`` in the cell below :attr:`mesh_face`, ``+z`` at or above
      it, which is the fluid channel's own placement rule -- so an armed
      engine requires an interior mesh face and refuses without one;
    * the ``1 - R_N`` remainder keeps the at-rest ``M_wall`` rebirth in the
      cell it was counted into, not a face inflow: a wire has no half-space;
    * counts the fluid collected from the ANNULUS ride the column placement,
      because the counted ``anode`` row folds both zones onto the column and
      the thermal rebirth has always been a column birth. That is a disclosed
      placement inherited from the channel this splits, not a choice made
      here.

    A cell whose committed incident energy is exactly ZERO launches nothing
    and is born wholly thermal -- fluid parity with ``v_back = 0``, per cell
    rather than per tick, and a legal booked state rather than a refusal. See
    :meth:`_split_anode_recycle`.

    Armed, the engine also emits the two PRESENCE-GATED rows of
    :data:`LEDGER_MOMENTUM_DIAGNOSTICS`: the signed axial momentum the jet
    launched, and the axial momentum the mesh INTERCEPTED and kept (the wires
    re-emit at rest, so all of it stays on the structure).

    ``baffle_faces`` / ``baffle_clear_radius_cm`` are the THIN ANNULAR
    BAFFLES, empty (absent) by default. Each is a zero-thickness annular disc
    standing on the vessel wall at one interior mesh face, solid from its
    clear radius out to the bore and open inside it. It intercepts the share
    of the ANNULUS flux crossing that face which the ring blocks, tallies it
    with the count, the energy and the signed axial momentum, and re-emits it
    at ``T_wall_K`` on the wall spectrum in the cell it was intercepted FROM
    -- the anode mesh's channel form exactly, restricted to the annulus,
    because the disc's bore is at least the local plasma radius and the COLUMN
    passes untouched. Full accommodation at the wall temperature is the
    convention: the scalar ``accommodation`` covers the cylinder and the two
    ends, while mesh, closed faces and now baffles run at ``alpha = 1``.
    Both annulus treatments carry it -- the ``rates`` march as an interception
    at the face, the ``bounded_chord`` map as an annular throat -- and both
    book into the ONE channel pair ``baffle_blocked`` / ``baffle_reemit``.
    See :meth:`_configure_baffles` for the transparency and its refusals, and
    ``momentum_baffle_absorbed``, which is presence-gated on the baffles
    themselves rather than on the anode jet.
    """

    def __init__(
        self,
        *,
        geometry,
        nvz=48,
        nvp=12,
        accommodation=1.0,
        wall_reflection="specular",
        elastic_model="phelps_iso",
        exchange_model="cauchy_chord",
        annulus_flights="rates",
        cathode_jet=None,
        anode_jet=None,
        transparency=1.0,
        mesh_face=-999,
        baffle_faces=(),
        baffle_clear_radius_cm=(),
        s_L=0.0,
        s_R=0.0,
        T_wall_K=T_WALL_K,
        vmax_cm_s=None,
        Ti_cap_eV=10.0,
        u_cap_cm_s=2.0e6,
        grid=None,
    ):
        if elastic_model not in ELASTIC_MODELS:
            raise ValueError(
                f"elastic_model must be one of {ELASTIC_MODELS} "
                f"(got {elastic_model!r})"
            )
        if exchange_model not in EXCHANGE_MODELS:
            raise ValueError(
                f"exchange_model must be one of {EXCHANGE_MODELS} "
                f"(got {exchange_model!r})"
            )
        if annulus_flights not in ANNULUS_FLIGHT_MODELS:
            raise ValueError(
                f"annulus_flights must be one of {ANNULUS_FLIGHT_MODELS} "
                f"(got {annulus_flights!r})"
            )
        if wall_reflection not in WALL_REFLECTION_MODELS:
            raise ValueError(
                f"wall_reflection must be one of {WALL_REFLECTION_MODELS} "
                f"(got {wall_reflection!r})"
            )
        self.cathode_jet = _validated_cathode_jet(cathode_jet)
        self.anode_jet = _validated_anode_jet(anode_jet)
        self.accommodation = float(accommodation)
        self.wall_reflection = str(wall_reflection)
        self.elastic_model = str(elastic_model)
        self.exchange_model = str(exchange_model)
        self.annulus_flights = str(annulus_flights)
        self.T_wall_K = float(T_wall_K)
        self.transparency = float(transparency)
        self.mesh_face = int(mesh_face)
        self.s_L = float(s_L)
        self.s_R = float(s_R)

        self.dz = np.asarray(geometry.length_cm, dtype=float)
        self.nz = self.dz.size
        if self.anode_jet is not None and not 0 < self.mesh_face < self.nz:
            # The anode jet's launch DIRECTION is defined against the mesh
            # face, and the two cells it re-emits into are its flanking pair,
            # so a face outside (0, nz) leaves the channel with no side to
            # launch away from. This is the engine's half of the statement;
            # the solver additionally requires the geometry to carry exactly
            # one anode face and that it be this one.
            raise ValueError(
                "the DVM anode jet launches its backscatter AWAY from the "
                "anode mesh, on the side the ions were collected from, so it "
                "needs an INTERIOR mesh face with a flanking cell on each "
                f"side: mesh_face={self.mesh_face} on a {self.nz}-cell grid "
                "has none. Accepted: 0 < mesh_face < nz (a geometry with a "
                "resolved anode mesh), or anode_jet=None"
            )
        V_col, V_ann = neutral_zone_volumes(geometry)
        self.V_col = np.asarray(V_col, dtype=float)
        self.V_ann = np.asarray(V_ann, dtype=float)
        self.A_col = self.V_col / self.dz
        self.A_ann = self.V_ann / self.dz
        # Axial transport uses FACE areas, not cell areas. The shipped
        # steady march writes cell i's gain as ``lam_i f_{i-1}`` with
        # ``lam_i = |v_z|/dz_i``, which moves ``|v_z| f A_i dt`` particles
        # in while cell i-1 loses ``|v_z| f A_{i-1} dt`` -- equal only on a
        # constant-area grid. This device's end cells are expanded, so the
        # face form is required for the inventory ledger to close: one
        # area per face, taken as the throat ``min(A_left, A_right)``, the
        # free-molecular choice. Both ends are open (the pumped faces).
        self.face_c = _throat_areas(self.A_col)
        self.face_a = _throat_areas(self.A_ann)
        # Where a SCALAR cathode/collector wall return is deposited. A typed
        # geometry resolves it by role, because the live cell against the
        # cathode surface is not the first cell whenever something sits behind
        # the cathode (a plenum always does; an obstruction adds another). The
        # synthetic tubes the gate suite builds carry no roles at all: there
        # the ends ARE the surfaces, which is the fallback.
        self.cath_cell = 0
        self.coll_cell = self.nz - 1
        # Every live cell against a cathode surface, not just the first. The
        # scalar above is where a SCALAR return is deposited and is unchanged;
        # this tuple is what the closed-face re-emission asks "is this side the
        # cathode surface?", which on a twin machine has two right answers.
        self.cath_cells = ()
        if getattr(geometry, "cell_role", None) is not None:
            by_role = absorbing_live_cells_by_role(geometry)
            if by_role.get("cathode"):
                self.cath_cell = int(by_role["cathode"][0])
                self.cath_cells = tuple(
                    int(cell) for cell in by_role["cathode"]
                )
            if by_role.get("collector"):
                self.coll_cell = int(by_role["collector"][0])
        self._configure_closed_faces(geometry)
        Rp = np.asarray(geometry.Rp_cm, dtype=float)
        Rm = np.asarray(geometry.Rm_cm, dtype=float)
        self._configure_baffles(baffle_faces, baffle_clear_radius_cm, Rp)

        if grid is None:
            if vmax_cm_s is None:
                vmax_cm_s = 4.0 * np.sqrt(
                    max(float(Ti_cap_eV), 0.5) * EV / M_HE
                ) + 1.5 * float(u_cap_cm_s)
            v_fine = 0.25 * np.sqrt(KB * self.T_wall_K / M_HE)
            grid = VGrid(float(vmax_cm_s), float(vmax_cm_s), nvz, nvp, v_fine)
        self.g = grid
        g = self.g
        if g.nvz % 2:
            raise ValueError(
                "the DVM velocity grid needs an EVEN v_z bin count: an odd "
                "count places a bin at exactly v_z = 0, which neither "
                "transports nor mirrors under end-wall reflection "
                f"(got nvz={g.nvz})"
            )
        # Exact v_z -> -v_z bin map (the stretched axis is symmetric about
        # zero at half-offsets, so the mirror is a pure index reversal).
        self.mirror = np.arange(g.nvz)[::-1]

        # Zone rates, with the column<->annulus antisymmetry imposed
        # through the ACTUAL geometry volumes so that
        #     V_col * nu_x  ==  V_ann * nu_xp
        # holds to roundoff cell by cell (the particle ledger's zone
        # channel cancels exactly, which the ledger gate checks). Only the
        # mean chord and the surface split differ between the branches.
        vp = g.vp[None, :]
        Rp2 = Rp[:, None]
        Rm2 = Rm[:, None]
        gap = np.maximum(Rm2 - Rp2, 1e-9)
        if self.exchange_model == "geometric":
            # 2D Cauchy chord pi A / P on the cell cross-section; the
            # encounter splits between the two circles by perimeter.
            nu_total = 2.0 * vp / (np.pi * gap)
            self.nuxp = (Rp2 / (Rp2 + Rm2)) * nu_total
            self.nuw = (Rm2 / (Rp2 + Rm2)) * nu_total
        else:
            # Cauchy-chord branch, transcribed from KN2Zone's default:
            # the 3D chord 4V/S = 2 gap, split as Rp/Rm : 1 - Rp/Rm.
            nu_total = vp / (2.0 * gap)
            self.nuxp = (Rp2 / Rm2) * nu_total
            self.nuw = (1.0 - Rp2 / Rm2) * nu_total
        ratio = np.where(
            self.V_col > 0.0, self.V_ann / np.maximum(self.V_col, 1e-300), 0.0
        )
        self.nux = self.nuxp * ratio[:, None]
        # Cells with no annulus exchange with nothing and see no radial wall.
        no_ann = self.V_ann <= 0.0
        self.nux[no_ann] = 0.0
        self.nuxp[no_ann] = 0.0
        self.nuw[no_ann] = 0.0

        self.M_wall = g.wall_emission_spectrum(self.T_wall_K)
        self.M_cold = g.maxwellian(self.T_wall_K * KB / EV, 0.0)

        # Energy weights for the ledger. ``E_bin`` is the kinetic energy of
        # one atom in each velocity bin [erg]; the two means are that same
        # moment of the run-constant emission spectra, so a channel whose
        # birth is a counted number times a fixed spectrum is booked as that
        # product rather than by re-summing the bins.
        self.E_bin = 0.5 * M_HE * g.V2
        self.E_wall_mean = float((self.M_wall * self.E_bin).sum())
        self.E_cold_mean = float((self.M_cold * self.E_bin).sum())

        # Closed-face re-emission spectra. A closed face is a solid plate, so
        # each side takes the cosine HALF-flux directed away from it -- not
        # the cylinder's symmetric spectrum, which would launch half the
        # returned mass straight back into the plate. The wall-temperature
        # directions are run constants; the cathode-surface ones ride the
        # live surface temperature and are built per update.
        self._closed_wall_spectra = {}
        surface_dirs = []
        for _face, d_in, _cell, surface in self._closed_emitters:
            emit = -d_in
            if surface:
                if emit not in surface_dirs:
                    surface_dirs.append(emit)
            elif (False, emit) not in self._closed_wall_spectra:
                self._closed_wall_spectra[(False, emit)] = (
                    g.half_flux_spectrum(self.T_wall_K, emit)
                )
        self._closed_surface_dirs = tuple(surface_dirs)

        shape = (self.nz, g.nvz, g.nvp)
        self.f_c = np.zeros(shape)
        self.f_a = np.zeros(shape)
        # Bounded-chord arm: the annulus state is the three in-flight class
        # populations, held at their flight midpoints; ``f_a`` is their sum
        # and stays the annulus distribution every other consumer reads.
        self.flights = None
        self.f_flight = None
        self.last_flight = None
        if self.annulus_flights == "bounded_chord":
            self.flights = BoundedChordFlights(
                dz=self.dz,
                V_ann=self.V_ann,
                A_ann=self.A_ann,
                Rp_cm=Rp,
                Rm_cm=Rm,
                grid=g,
                mesh_face=self.mesh_face,
                transparency=self.transparency,
                baffles=self._baffle_throats,
            )
            self.f_flight = {
                name: np.zeros(shape) for name in FLIGHT_CLASSES
            }
            # Free-molecular equilibrium split of a wall-launched annulus:
            # the branch weights are the view factor, the residence weights
            # the class flight times, and both are ratios of the chord
            # classes, so the split is one per-cell number independent of
            # the velocity bin.
            F = self.flights.F_inner
            w_ww = (1.0 - F) * self.flights.chords["ww"]
            w_wi = F * self.flights.chords["wi"]
            denom = np.maximum(w_ww + w_wi, 1e-300)
            self._seed_split = {
                "ww": w_ww / denom,
                "wi": w_wi / denom,
                "io": np.zeros(self.nz),
            }
        # Pending end-wall returns, in PARTICLES per bin (inward half only).
        self.pend_L_c = np.zeros((g.nvz, g.nvp))
        self.pend_R_c = np.zeros((g.nvz, g.nvp))
        self.pend_L_a = np.zeros((g.nvz, g.nvp))
        self.pend_R_a = np.zeros((g.nvz, g.nvp))

        # Plasma-coupling accumulators, frozen between neutral updates and
        # read by the solver's RHS. Per-second densities on the PLASMA
        # volume (= the column volume).
        cells = np.zeros(self.nz)
        self.M_transfer = cells.copy()
        self.Ei_transfer = cells.copy()
        self.S_transfer = cells.copy()
        # The CX/elastic pair's own share of the two rows above, the per-ion
        # collision frequency that carries it, and the drift and ion-frame
        # temperature it relaxes towards. Published so the plasma side can
        # integrate the pair as the linear relaxation it is rather than as a
        # frozen rate; see _book_transfer.
        self.M_transfer_pair = cells.copy()
        self.Ei_transfer_pair = cells.copy()
        self.nu_pair = cells.copy()
        self.u_n_eff = cells.copy()
        self.T_eff_eV = np.full(self.nz, self.T_wall_K * KB / EV)
        self.Tn_col_eV = np.full(self.nz, self.T_wall_K * KB / EV)
        self.updates = 0
        self.last_ledger = None
        # Last update's closed-face traffic in PARTICLES per cell, keyed by
        # the direction the blocked particles were re-emitted into.
        self.last_closed_counts = None
        # Last update's annular-baffle interception in PARTICLES per cell, on
        # the cell each blocked particle arrived FROM. Published for the same
        # reason the closed-face counts are: the ledger rows are domain totals
        # and the per-cell placement -- that a baffle books on its own face's
        # flanking pair and nowhere else -- is not recoverable from them.
        self.last_baffle_counts = None
        # Last update's NON-accommodated cylindrical-wall return, in PARTICLES
        # per (cell, v_z bin, v_perp bin). Published because the return's
        # spectrum -- its count, its energy and its net v_z -- is the whole
        # content of the ``wall_reflection`` selector and is otherwise not
        # separable from the accommodated share once both are in ``f_a``.
        self.last_wall_return = None

        # ---- deferred transfer ledger (the K2d floor-aware relax)
        #
        # The solver may WITHHOLD part of a tick's booked transfer at a cell
        # whose ion energy the frozen drain would otherwise carry below its
        # floor inside one step. Withheld energy and momentum are not
        # discarded: they are held here as a per-cell DEBT and re-offered on
        # every later step, so the exchange stays a transfer rather than a
        # sink. Units are the transfer's own, integrated over the step:
        # erg/cm^3 and g/(cm^2 s), on the column (= plasma) volume.
        #
        # The identity these carry, per cell and at every accepted step:
        #
        #     applied_cum + debt == booked_cum
        #
        # i.e. everything the kinetic side booked has either been handed to
        # the plasma or is still owed to it. ``booked_cum`` accumulates the
        # tick-frozen rate over the step; ``applied_cum`` what the step's
        # RHS actually carried.
        self.M_debt = cells.copy()
        self.Ei_debt = cells.copy()
        # HOLD debt: the second, disjoint reason applied can differ from
        # booked. The floor debt above is "the plasma could not absorb it";
        # this one is the zero-order-hold truncation the exponential hold
        # replaces -- the tick froze a rate at the tick's state, and the
        # state moved inside the tick. Zero for the whole run under
        # ``neutral_kinetic_dvm_transfer_hold = "zoh"``, and repaid as a
        # constant source over the following tick otherwise, so the identity
        # the pair carries is
        #
        #     applied_cum + debt + hold_debt == booked_cum
        #
        # per cell at every accepted step. Same units as the debt above.
        self.M_hold_debt = cells.copy()
        self.Ei_hold_debt = cells.copy()
        self.M_booked_cum = cells.copy()
        self.Ei_booked_cum = cells.copy()
        self.M_applied_cum = cells.copy()
        self.Ei_applied_cum = cells.copy()
        # Census: steps scoped, steps where any cell was limited, and the
        # per-cell count of limited steps.
        self.relax_steps = 0
        self.relax_limited_steps = 0
        self.relax_cell_steps = cells.copy()

        # ---- counted-particle ionization ledger (the K2e handshake)
        #
        # The partner's booked count is what the ionization channel debits,
        # and a cell that cannot give up the whole count owes the rest.
        # In PARTICLES, per column cell, cumulative over the run, carrying
        #
        #     ion_removed_cum + ion_debt == ion_booked_cum
        #
        # at every update: no ionization booked by the plasma is ever
        # silently un-removed, which is the leak this ledger exists to
        # make visible from a saved file.
        self.ion_debt = cells.copy()
        self.ion_booked_cum = cells.copy()
        self.ion_removed_cum = cells.copy()
        self.ion_shortfall_updates = 0
        self.ion_shortfall_cell_updates = cells.copy()

    # ---------------------------------------------------- annular baffles

    def _configure_baffles(self, faces, clear_radius_cm, Rp):
        """Resolve the thin annular baffles and their annulus transparencies.

        A baffle is a zero-thickness annular disc standing on the vessel wall
        at one mesh face: the ring ``R_clear < r < Rm`` is solid and the bore
        ``r < R_clear`` is open. Its clear radius is at least the local column
        radius, so the PLASMA CHANNEL is untouched and the object acts on the
        annulus alone -- the same statement the fluid's series orifice makes
        (``physics/neutrals.py``, ``open_ann = pi (R_clear^2 - R_col^2)``).

        The per-face ANNULUS TRANSPARENCY is that open ring over the annulus
        area the march actually transports through, which is the face THROAT
        ``face_a[f] = min(A_ann[f-1], A_ann[f])``::

            t_f = min(pi (R_clear^2 - R_col^2) / face_a[f], 1)

        so the passed throughput ``t_f F |v_z| face_a[f]`` is exactly
        ``F |v_z| open_ann``: the open area is what passes, which is the whole
        content of the free-molecular orifice. ``R_col`` is the FACE AVERAGE
        ``(Rp[f-1] + Rp[f]) / 2``, which is the fluid's own convention at the
        same face (``_zone_face_average(Rp)[f - 1]``), so the two channels
        cannot disagree about how much ring is open.

        The clip at 1 is not cosmetic: a prescribed annulus area can be
        SMALLER than the geometric ``pi (Rm^2 - Rp^2)`` (support rods are
        subtracted from it), so an open ring can exceed the area available and
        a transparency above 1 would AMPLIFY the flux. A face at ``t_f = 1``
        is not armed at all -- it is a no-op by construction rather than by a
        multiplication by 1.0, which is what makes an unrestricting baffle
        bit-exact.

        Raises on a clear radius BELOW the local column radius: that is a disc
        covering part of the plasma channel, which this object does not model
        and which would silently seal the annulus (``open_ann <= 0``) instead.
        ``R_clear == R_col`` is legal and gives ``t_f = 0``, a fully closed
        annulus -- the same configuration the fluid accepts, where its orifice
        conductance goes to zero.
        """
        faces = np.asarray(faces, dtype=int).reshape(-1)
        radii = np.asarray(clear_radius_cm, dtype=float).reshape(-1)
        if faces.shape != radii.shape:
            raise ValueError(
                "the DVM baffle face and clear-radius arrays must have equal "
                f"length (got {faces.size} faces and {radii.size} radii)"
            )
        if faces.size and np.unique(faces).size != faces.size:
            raise ValueError(
                "the DVM baffles must sit on DISTINCT mesh faces "
                f"(got {faces.tolist()})"
            )
        if faces.size and not np.all(np.isfinite(radii)):
            raise ValueError(
                f"the DVM baffle clear radii must be finite (got {radii.tolist()})"
            )
        transparency = np.ones(faces.size)
        tau = [None] * (self.nz + 1)
        throats = []
        for i, (face, clear) in enumerate(zip(faces, radii)):
            face = int(face)
            if not 0 < face < self.nz:
                raise ValueError(
                    "a DVM baffle intercepts the annulus flux crossing ONE "
                    "interior mesh face, so it needs a flanking cell on each "
                    f"side: baffle face {face} on a {self.nz}-cell grid has "
                    f"none. Accepted: 0 < face < {self.nz}"
                )
            R_col = 0.5 * (float(Rp[face - 1]) + float(Rp[face]))
            if float(clear) < R_col:
                raise ValueError(
                    "a DVM baffle leaves the plasma channel open, so its "
                    "clear radius may not fall below the local column radius: "
                    f"face {face} has R_clear={clear} cm against "
                    f"R_col={R_col} cm (the face average of Rp). Accepted: "
                    "R_clear >= R_col, where R_clear == R_col closes the "
                    "annulus entirely"
                )
            ring = np.pi * (float(clear) ** 2 - R_col**2)
            area = float(self.face_a[face])
            if area <= 0.0 or ring >= area:
                # No annulus to restrict, or an open ring at least as wide as
                # the annulus itself: unarmed, and therefore bit-exactly inert.
                continue
            transparency[i] = ring / area
            tau[face] = float(transparency[i])
            throats.append((face, ring))
        #: The realized per-face annulus transparency, in the order the faces
        #: were given. The ONE resolved quantity published: the gates read it
        #: rather than recomputing it, because every B6 statement is made
        #: against the value the engine actually used. The constructor's own
        #: arguments are deliberately not echoed back as attributes -- an
        #: unread echo is a second copy of the truth with nothing keeping it
        #: honest.
        self.baffle_transparency = transparency
        #: Per-face transparency lookup for the march, ``None`` where no
        #: baffle is armed. A plain list so the inner loop's test is an
        #: index and an identity check rather than a numpy scalar read.
        self._baffle_tau = tau
        #: ``(face, open_area)`` for every ARMED baffle, for the bounded-chord
        #: flight map's throat convention.
        self._baffle_throats = tuple(throats)
        #: Whether any baffle actually restricts anything. Every baffle code
        #: path in the update is gated on this, so an unarmed configuration
        #: runs the arithmetic it ran before B6, bit for bit.
        self._baffle_any = bool(throats)

    # ------------------------------------------------------ closed faces

    def _configure_closed_faces(self, geometry):
        """Resolve the interior closed faces and the two sides of each.

        A face is CLOSED where the plasma stops without the domain ending:
        ``plasma_open`` false, or ``plasma_absorbing`` true, which is the
        refinement of that same set. Both come from the typed cell roles --
        a face between a plasma-dead cell (plenum, obstruction) and a live
        one -- so the geometry already carries the fact and no separate
        configuration names it. Only INTERIOR faces are taken: the two
        domain ends are closed by construction and are the end-wall
        channel, which is lagged and pumped and is not this one.

        Sets ``closed_faces`` (the face indices, ascending), ``_closed_face``
        (the same as a per-face flag the march indexes), and
        ``_closed_emitters`` -- one entry ``(face, d_in, cell, surface)``
        per face and per SIDE, where ``d_in`` is the march direction whose
        flux that side delivers to the face, ``cell`` is the side's own
        cell, and ``surface`` marks a side that is a cathode surface and so
        re-emits at the live surface temperature rather than the wall's.

        A geometry with no face topology at all -- the synthetic tubes the
        gate suite builds -- yields no closed faces, so the channel is
        absent and the march is bit-identical to one built without it.
        """
        self.closed_faces = ()
        self._closed_face = [False] * (self.nz + 1)
        self._closed_emitters = ()
        open_faces = getattr(geometry, "plasma_open", None)
        if open_faces is None:
            return
        closed = ~np.asarray(open_faces, dtype=bool)
        absorbing = getattr(geometry, "plasma_absorbing", None)
        if absorbing is not None:
            closed = closed | np.asarray(absorbing, dtype=bool)
        if closed.size != self.nz + 1:
            raise ValueError(
                "the DVM needs one plasma face flag per face: geometry "
                f"carries {closed.size} for {self.nz} cells "
                f"(expected {self.nz + 1})"
            )
        faces = tuple(
            int(face) for face in np.flatnonzero(closed) if 0 < face < self.nz
        )
        if self.mesh_face in faces:
            raise ValueError(
                f"face {self.mesh_face} is both the anode mesh and a closed "
                "plasma face: a face is one or the other, never both"
            )
        emitters = []
        for face in faces:
            self._closed_face[face] = True
            for d_in in (+1, -1):
                cell = face - 1 if d_in > 0 else face
                emitters.append((face, d_in, cell, cell in self.cath_cells))
        self.closed_faces = faces
        self._closed_emitters = tuple(emitters)

    def _closed_face_spectra(self, T_s_K):
        """Return ``{(surface, emit_direction): spectrum}`` for this update.

        The wall-temperature entries are run constants built once; the
        cathode-surface ones follow the live surface temperature and are
        built here, at most one per emitted direction.
        """
        spectra = dict(self._closed_wall_spectra)
        for emit in self._closed_surface_dirs:
            spectra[(True, emit)] = self.g.half_flux_spectrum(T_s_K, emit)
        return spectra

    # ------------------------------------------------------------ state

    def seed_from_density(self, nn_col, nn_ann, T_K=None):
        """Seed both distributions as Maxwellians at ``T_K`` (default wall)."""
        if T_K is None:
            spec = self.M_cold
        else:
            spec = self.g.maxwellian(float(T_K) * KB / EV, 0.0)
        self.f_c = np.asarray(nn_col, dtype=float)[:, None, None] * spec
        self.f_a = np.asarray(nn_ann, dtype=float)[:, None, None] * spec
        if self.f_flight is not None:
            # The seed is read as a population that has just launched off
            # the cylindrical wall: split between the two wall classes by
            # the free-molecular equilibrium weights, so the sum is the
            # requested annulus density exactly.
            for name in FLIGHT_CLASSES:
                self.f_flight[name] = (
                    self._seed_split[name][:, None, None] * self.f_a
                )
        self.pend_L_c[...] = 0.0
        self.pend_R_c[...] = 0.0
        self.pend_L_a[...] = 0.0
        self.pend_R_a[...] = 0.0

    def snapshot(self):
        """Return a deep copy of every mutable piece of the DVM state."""
        snap = {
            "f_c": self.f_c.copy(),
            "f_a": self.f_a.copy(),
            "pend_L_c": self.pend_L_c.copy(),
            "pend_R_c": self.pend_R_c.copy(),
            "pend_L_a": self.pend_L_a.copy(),
            "pend_R_a": self.pend_R_a.copy(),
            "M_transfer": self.M_transfer.copy(),
            "Ei_transfer": self.Ei_transfer.copy(),
            "S_transfer": self.S_transfer.copy(),
            "M_transfer_pair": self.M_transfer_pair.copy(),
            "Ei_transfer_pair": self.Ei_transfer_pair.copy(),
            "nu_pair": self.nu_pair.copy(),
            "u_n_eff": self.u_n_eff.copy(),
            "T_eff_eV": self.T_eff_eV.copy(),
            "Tn_col_eV": self.Tn_col_eV.copy(),
            "updates": int(self.updates),
            "M_debt": self.M_debt.copy(),
            "Ei_debt": self.Ei_debt.copy(),
            "M_hold_debt": self.M_hold_debt.copy(),
            "Ei_hold_debt": self.Ei_hold_debt.copy(),
            "M_booked_cum": self.M_booked_cum.copy(),
            "Ei_booked_cum": self.Ei_booked_cum.copy(),
            "M_applied_cum": self.M_applied_cum.copy(),
            "Ei_applied_cum": self.Ei_applied_cum.copy(),
            "relax_steps": int(self.relax_steps),
            "relax_limited_steps": int(self.relax_limited_steps),
            "relax_cell_steps": self.relax_cell_steps.copy(),
            "ion_debt": self.ion_debt.copy(),
            "ion_booked_cum": self.ion_booked_cum.copy(),
            "ion_removed_cum": self.ion_removed_cum.copy(),
            "ion_shortfall_updates": int(self.ion_shortfall_updates),
            "ion_shortfall_cell_updates": self.ion_shortfall_cell_updates.copy(),
        }
        if self.f_flight is not None:
            snap["f_flight"] = {
                name: arr.copy() for name, arr in self.f_flight.items()
            }
        return snap

    def restore(self, snap):
        """Restore a :meth:`snapshot`."""
        self.f_c = snap["f_c"].copy()
        self.f_a = snap["f_a"].copy()
        if self.f_flight is not None:
            self.f_flight = {
                name: arr.copy() for name, arr in snap["f_flight"].items()
            }
        self.pend_L_c = snap["pend_L_c"].copy()
        self.pend_R_c = snap["pend_R_c"].copy()
        self.pend_L_a = snap["pend_L_a"].copy()
        self.pend_R_a = snap["pend_R_a"].copy()
        self.M_transfer = snap["M_transfer"].copy()
        self.Ei_transfer = snap["Ei_transfer"].copy()
        self.S_transfer = snap["S_transfer"].copy()
        self.M_transfer_pair = snap["M_transfer_pair"].copy()
        self.Ei_transfer_pair = snap["Ei_transfer_pair"].copy()
        self.nu_pair = snap["nu_pair"].copy()
        self.u_n_eff = snap["u_n_eff"].copy()
        self.T_eff_eV = snap["T_eff_eV"].copy()
        self.Tn_col_eV = snap["Tn_col_eV"].copy()
        self.updates = int(snap["updates"])
        self.M_debt = snap["M_debt"].copy()
        self.Ei_debt = snap["Ei_debt"].copy()
        self.M_hold_debt = snap["M_hold_debt"].copy()
        self.Ei_hold_debt = snap["Ei_hold_debt"].copy()
        self.M_booked_cum = snap["M_booked_cum"].copy()
        self.Ei_booked_cum = snap["Ei_booked_cum"].copy()
        self.M_applied_cum = snap["M_applied_cum"].copy()
        self.Ei_applied_cum = snap["Ei_applied_cum"].copy()
        self.relax_steps = int(snap["relax_steps"])
        self.relax_limited_steps = int(snap["relax_limited_steps"])
        self.relax_cell_steps = snap["relax_cell_steps"].copy()
        self.ion_debt = snap["ion_debt"].copy()
        self.ion_booked_cum = snap["ion_booked_cum"].copy()
        self.ion_removed_cum = snap["ion_removed_cum"].copy()
        self.ion_shortfall_updates = int(snap["ion_shortfall_updates"])
        self.ion_shortfall_cell_updates = snap["ion_shortfall_cell_updates"].copy()

    # ---------------------------------------------------------- moments

    def column_density(self):
        """Column neutral density [cm^-3] -- the zeroth moment of ``f_c``."""
        return self.f_c.sum(axis=(1, 2))

    def annulus_density(self):
        """Annulus neutral density [cm^-3] -- the zeroth moment of ``f_a``."""
        return self.f_a.sum(axis=(1, 2))

    def column_drift(self):
        """Column axial drift ``<v_z>`` [cm/s]."""
        return _drift(self.f_c, self.g)

    def column_temperature_eV(self):
        """Column neutral temperature [eV] from the second central moment.

        ``T_n = (m/3) <|v - u_n|^2>`` with the three-dimensional measure
        (``v_perp`` already carries two degrees of freedom). Always
        computed; whether it FEEDS the fluid rate evaluations is a
        separate switch on the solver side.
        """
        return _temperature_eV(self.f_c, self.g)

    def f_inventory(self):
        """Particles carried by the distributions themselves."""
        return float(
            (self.f_c.sum(axis=(1, 2)) * self.V_col).sum()
            + (self.f_a.sum(axis=(1, 2)) * self.V_ann).sum()
        )

    def pending_inventory(self):
        """Particles held in the lagged end-wall return buffers."""
        return float(
            self.pend_L_c.sum()
            + self.pend_R_c.sum()
            + self.pend_L_a.sum()
            + self.pend_R_a.sum()
        )

    def total_inventory(self):
        """Domain particle inventory including the pending end buffers."""
        return self.f_inventory() + self.pending_inventory()

    def _energy_of(self, counts):
        """Kinetic energy [erg] of a per-bin PARTICLE count array.

        ``counts`` carries the two bin axes last, so both a
        ``(nz, nvz, nvp)`` volume tally and a ``(nvz, nvp)`` surface tally
        are accepted.
        """
        return float((counts * self.E_bin).sum())

    def _wall_return_counts(self, L_wall, N_wall, alpha):
        """Return the NON-accommodated cylindrical-wall share, in PARTICLES.

        ``L_wall`` is this tick's wall landings per ``(cell, v_z, v_perp)``
        and ``N_wall`` their per-cell sum; ``alpha`` is the thermal
        accommodation coefficient, so the share this method places is
        ``(1 - alpha)`` of both. The COUNT is that share exactly under either
        value of ``wall_reflection``; what the selector changes is the
        spectrum it comes back on.

        ``"specular"``
            the incident array scaled by ``(1 - alpha)``. On this
            axisymmetric grid a specular reflection off the cylinder
            reverses only the unresolved radial component of ``v_perp``, so
            the bin -- and hence the energy and the axial velocity -- is
            unchanged.

        ``"diffuse_elastic"``
            the same count, per cell, on a cosine-wall spectrum whose
            temperature parameter is solved so that the spectrum's DISCRETE
            mean energy equals the retained share's own incident mean energy
            per atom. The direction is randomized (the spectrum is symmetric
            in ``v_z``, so the return carries zero net axial momentum) while
            the energy handed to the surface by this share is zero, which is
            what makes the reflection elastic. Cells with no landings
            re-emit nothing.

        At ``alpha == 1`` there is no share to place and both values return
        the same array of exact zeros, so the pair degenerates.
        """
        if self.wall_reflection == "specular" or alpha >= 1.0:
            return (1.0 - alpha) * L_wall
        live = np.flatnonzero(N_wall > 0.0)
        if live.size == 0:
            return (1.0 - alpha) * L_wall
        # The scaled incident array is the specular RETURN, and the diffuse
        # branch below returns a spectrum instead; it is formed only on the
        # paths that hand it back, so the diffuse branch no longer scales a
        # whole landing array to use nothing of it but its shape.
        incident = (L_wall[live] * self.E_bin).sum(axis=(1, 2))
        e_bar = incident / N_wall[live]
        spectra = self._solve_wall_return_spectra(e_bar)
        out = np.zeros_like(L_wall, dtype=float)
        out[live] = ((1.0 - alpha) * N_wall[live])[:, None, None] * spectra
        return out

    def _wall_return_energy_miss(self, spectra, e_bar):
        """Return the relative DISCRETE mean-energy miss of ``spectra``.

        One value per cell: ``|<E> - e_bar| / e_bar`` with ``<E>`` contracted
        over the assembled spectrum against :attr:`E_bin`. This is the
        quantity :data:`WALL_ENERGY_SOLVE_REL_TOL` bounds, read off the
        spectrum that is actually returned rather than off the separable
        residual the solve iterates on.
        """
        got = (spectra * self.E_bin).sum(axis=(1, 2))
        return np.abs(got - e_bar) / e_bar

    def _secant_wall_return_speeds(self, e_bar):
        """Return thermal speeds matching ``e_bar``, or ``None`` if unconverged.

        A secant iteration in ``ln s`` on the residual
        ``F(ln s) = ln <E>(s) - ln e_bar``, where ``<E>`` is the SEPARABLE
        contraction of :func:`_cosine_wall_mean_energy`. Working in logs makes
        the continuum relation ``<E> = 2 m s^2`` an exactly linear residual of
        slope 2, so the seed ``s0 = sqrt(e_bar / 2 m)`` plus one Newton step at
        that known slope already lands close and the secant closes the rest.

        CONVERGENCE CRITERION: the iteration stops at the first sweep where
        ``max|F| <= WALL_ENERGY_SECANT_REL_TOL`` over every cell in the call,
        capped at ``WALL_ENERGY_SECANT_MAX_ITERS``. Every cell steps on every
        sweep -- no row is frozen once it converges -- so the count is a single
        whole-call reduction, the same shape of data dependence the bisection's
        own bracket-width test has, and a cell's answer depends on the cells it
        was solved with exactly as it did before.

        Returns ``None`` rather than a spectrum whenever the iteration does not
        reach that bar, whenever a residual or a step is not finite, and
        whenever the secant denominator vanishes on a flat residual -- which is
        what a target above the grid's saturation energy produces. The caller
        then hands the WHOLE call to the bracketed bisection, which is what
        raises on saturation.
        """
        ln_target = np.log(e_bar)

        def residual(u):
            with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                mean = _cosine_wall_mean_energy(self.g, np.exp(u))
                return np.log(mean) - ln_target

        u_prev = np.log(np.sqrt(e_bar / (2.0 * M_HE)))
        f_prev = residual(u_prev)
        if not np.all(np.isfinite(f_prev)):
            return None
        # One Newton step at the continuum slope d(ln <E>)/d(ln s) = 2, which
        # is the second point the secant needs and costs no extra evaluation.
        u = u_prev - 0.5 * f_prev
        for _ in range(WALL_ENERGY_SECANT_MAX_ITERS):
            f = residual(u)
            if not np.all(np.isfinite(f)):
                return None
            if np.all(np.abs(f) <= WALL_ENERGY_SECANT_REL_TOL):
                return np.exp(u)
            df = f - f_prev
            with np.errstate(invalid="ignore", divide="ignore"):
                step = np.where(df != 0.0, f * (u - u_prev) / df, np.nan)
            if not np.all(np.isfinite(step)):
                return None
            np.clip(
                step,
                -WALL_ENERGY_SECANT_MAX_STEP,
                WALL_ENERGY_SECANT_MAX_STEP,
                out=step,
            )
            u_prev, f_prev = u, f
            u = u - step
        return None

    def _bisect_wall_return_speeds(self, e_bar):
        """Return thermal speeds matching ``e_bar`` by bracketed bisection.

        Seeded at the continuum relation ``<E> = 2 k T = 2 m s^2``, bracketed
        outward by halving and doubling, then bisected to the floating-point
        resolution of the bracket. The mean energy SATURATES once the spectrum
        outruns the grid's outermost bins, so a target above the saturation
        value has no solution: the bracket search raises ``ValueError`` naming
        the offending cells rather than returning a speed at an energy the
        caller did not ask for.
        """

        def mean_energy(s):
            # The spectra array is this call's own and is contracted away
            # immediately, so the energy weighting is applied in place: the
            # same multiply on the same operands, one temporary fewer.
            spectra = _cosine_wall_spectra(self.g, s)
            np.multiply(spectra, self.E_bin, out=spectra)
            return spectra.sum(axis=(1, 2))

        s0 = np.sqrt(e_bar / (2.0 * M_HE))
        lo = 0.5 * s0
        hi = 2.0 * s0
        for _ in range(WALL_ENERGY_SOLVE_MAX_BRACKET):
            low = mean_energy(lo) > e_bar
            if not np.any(low):
                break
            lo = np.where(low, 0.5 * lo, lo)
        else:
            raise ValueError(
                "the energy-matched wall return could not bracket the "
                f"requested mean energy from BELOW in "
                f"{WALL_ENERGY_SOLVE_MAX_BRACKET} halvings"
            )
        for _ in range(WALL_ENERGY_SOLVE_MAX_BRACKET):
            high = mean_energy(hi) < e_bar
            if not np.any(high):
                break
            hi = np.where(high, 2.0 * hi, hi)
        else:
            bad = np.flatnonzero(mean_energy(hi) < e_bar)
            raise ValueError(
                "the energy-matched wall return could not bracket the "
                "requested mean energy from ABOVE: the cosine-wall spectrum's "
                "discrete mean energy saturates on this velocity grid, so "
                f"{bad.size} cell(s) ask for more than the grid can re-emit "
                f"(first offenders {bad[:8].tolist()}, targets "
                f"{e_bar[bad[:8]].tolist()} erg). Widen the grid "
                "(neutral_kinetic_dvm_nvz / _nvp, or the velocity cap) or "
                "select wall_reflection='specular'"
            )
        for _ in range(WALL_ENERGY_SOLVE_MAX_ITERS):
            if np.all(hi - lo <= 4.0 * np.finfo(float).eps * hi):
                break
            mid = 0.5 * (lo + hi)
            below = mean_energy(mid) < e_bar
            lo = np.where(below, mid, lo)
            hi = np.where(below, hi, mid)
        return 0.5 * (lo + hi)

    def _solve_wall_return_spectra(self, e_bar):
        """Return cosine-wall spectra at the requested DISCRETE mean energies.

        ``e_bar`` is one target kinetic energy per atom [erg] per cell. The
        discrete mean energy of :func:`_cosine_wall_spectra` rises
        monotonically with the thermal speed ``s``, so the inverse is a
        one-parameter root solve, taken in two tiers:

        * :meth:`_secant_wall_return_speeds`, a secant in ``ln s`` on the
          SEPARABLE mean energy, which needs no ``(cells, nvz, nvp)`` array per
          evaluation and closes in a handful of them;
        * :meth:`_bisect_wall_return_speeds`, the bracketed bisection, which
          the WHOLE call falls back to whenever the secant does not reach its
          residual bar -- a saturated target being the case that cannot.

        The spectrum returned is :func:`_cosine_wall_spectra` at the solved
        speed either way; the separable contraction is used for the residual
        evaluations and nowhere else.

        The mean energy SATURATES once the spectrum outruns the grid's
        outermost bins, so a target above the saturation value has no
        solution. That is a misbooked launch spectrum, not a tolerance
        question: the bracket search and the final agreement check each raise
        ``ValueError`` naming the offending cells rather than returning a
        spectrum at an energy the caller did not ask for.
        """
        e_bar = np.asarray(e_bar, dtype=float)
        if not np.all(np.isfinite(e_bar)) or np.any(e_bar <= 0.0):
            raise ValueError(
                "the energy-matched wall return needs a positive finite "
                "incident mean energy per cell (got "
                f"min {np.min(e_bar)!r}, max {np.max(e_bar)!r})"
            )

        spectra = rel = None
        s = self._secant_wall_return_speeds(e_bar)
        if s is not None:
            spectra = _cosine_wall_spectra(self.g, s)
            rel = self._wall_return_energy_miss(spectra, e_bar)
            if np.any(rel > WALL_ENERGY_SOLVE_REL_TOL):
                # The secant met its bar on the separable residual and the
                # assembled spectrum does not agree: the call goes to the
                # bisection rather than being accepted at the wrong energy.
                spectra = rel = None
        if spectra is None:
            spectra = _cosine_wall_spectra(
                self.g, self._bisect_wall_return_speeds(e_bar)
            )
            rel = self._wall_return_energy_miss(spectra, e_bar)

        if np.any(rel > WALL_ENERGY_SOLVE_REL_TOL):
            bad = np.flatnonzero(rel > WALL_ENERGY_SOLVE_REL_TOL)
            raise ValueError(
                "the energy-matched wall return did not converge in "
                f"{WALL_ENERGY_SOLVE_MAX_ITERS} bisections: {bad.size} "
                f"cell(s) miss their incident mean energy by up to "
                f"{float(np.max(rel)):.3e} relative (tolerance "
                f"{WALL_ENERGY_SOLVE_REL_TOL:.1e}, first offenders "
                f"{bad[:8].tolist()})"
            )
        return spectra

    # ------------------------------------------------- surface jet launches

    def _grid_tied_launch_temperature_eV(self, v_back):
        """Return the GRID-TIED launch smear [eV] at a speed [cm/s].

        ``m dv_z(v_back)^2 / k_B``: the axial bin containing the launch speed,
        expressed as a temperature. That is the narrowest spectrum this grid
        can carry there -- a narrower one collapses onto a single bin, where
        the two-basis moment compensation has nothing to redistribute and
        cannot reach its targets at all.

        Shared by both surface jets, which is the point: the two channels are
        one construction on two surfaces, and a second copy of this formula
        would be free to drift away from the first.
        """
        edges = self.g.vz_edges
        k = int(np.clip(np.searchsorted(edges, abs(v_back)) - 1, 0, self.g.nvz - 1))
        dv = float(edges[k + 1] - edges[k])
        return M_HE * dv * dv / EV

    def _cathode_jet_launch_temperature_eV(self, v_back):
        """Return the launch smear [eV] for a backscatter speed [cm/s].

        The configured ``T_launch_eV`` when one was named, and otherwise the
        grid-tied width of :meth:`_grid_tied_launch_temperature_eV`.
        """
        named = self.cathode_jet["T_launch_eV"]
        if named is not None:
            return float(named)
        return self._grid_tied_launch_temperature_eV(v_back)

    def _anode_jet_launch_temperature_eV(self, v_back):
        """Return the ANODE jet's launch smear [eV] at a speed [cm/s].

        The configured ``T_launch_eV`` when one was named, and otherwise the
        grid-tied width of :meth:`_grid_tied_launch_temperature_eV`.
        """
        named = self.anode_jet["T_launch_eV"]
        if named is not None:
            return float(named)
        return self._grid_tied_launch_temperature_eV(v_back)

    def _cathode_jet_launch_spectrum(self, e_launch, cell):
        """Return the backscatter launch spectrum at ``e_launch`` erg per atom.

        ``e_launch`` is the kinetic energy ONE backscattered atom leaves
        with, ``(R_E/R_N) (phi_c + Ti)`` under the ``"total_reflected"``
        convention. The returned bin masses sum to 1 and their DISCRETE mean
        energy is ``e_launch`` -- not ``e_launch`` plus the smear's own
        ``(3/2) k T_launch``, which is why the drift is solved from the
        energy rather than set to ``v_back``::

            v_back^2 = 2 e_launch / m        u^2 = v_back^2 - 3 k T_launch / m

        so ``<v_z> = u`` and ``<|v|^2> = u^2 + 3 s^2 = v_back^2`` exactly.
        The atoms leave in ``+z``, the direction the cathode's own thermal
        re-emission is injected in.

        RAISES rather than returning an approximate spectrum. The
        compensation solve inside :meth:`VGrid.maxwellian` gives up quietly
        on a numerically singular system and after four iterations, leaving
        the analytic bin masses standing at whatever moments they happen to
        have; here that is a misbooked counted channel -- the surface has
        already been debited ``R_E`` of the incident energy and this
        spectrum is what receives it -- so the achieved moments are checked
        against their targets and a miss names the cell, the shortfall and
        the two ways out.
        """
        e_launch = float(e_launch)
        if not np.isfinite(e_launch) or e_launch <= 0.0:
            raise ValueError(
                "the DVM cathode jet needs a positive finite launch energy "
                f"per atom at cell {cell} (got {e_launch!r} erg): the "
                "backscattered share carries (R_E/R_N)(phi_c + Ti), and a "
                "cell that recycles particles at zero incident energy has no "
                "energetic share to launch"
            )
        v_back = np.sqrt(2.0 * e_launch / M_HE)
        T_launch = self._cathode_jet_launch_temperature_eV(v_back)
        if not np.isfinite(T_launch) or T_launch <= 0.0:
            raise ValueError(
                "the DVM cathode jet's launch smear must be a positive "
                f"temperature (got {T_launch!r} eV at cell {cell})"
            )
        s2 = T_launch * EV / M_HE
        u2 = v_back * v_back - 3.0 * s2
        if u2 <= 0.0:
            raise ValueError(
                "the DVM cathode jet cannot represent a launch energy below "
                "its own smear: cell "
                f"{cell} asks for {e_launch / EV:.6g} eV per atom while the "
                f"launch spectrum's thermal content alone is "
                f"{1.5 * T_launch:.6g} eV (T_launch = {T_launch:.6g} eV). "
                "Accepted: a smaller neutral_kinetic_dvm_cathode_jet_"
                "T_launch_eV, or an operating point whose cathode sheath "
                "actually makes the backscatter energetic"
            )
        u = np.sqrt(u2)
        spec = self.g.maxwellian(T_launch, u, exact_moments=True)
        total = float(spec.sum())
        got_u = float((spec * self.g.VZ).sum())
        # Recomputed from V2 rather than folded through the precomputed
        # self.E_bin (== 0.5 * M_HE * g.V2) every other energy read here
        # uses. The two differ only in where the constant enters the sum --
        # scale the sum, or sum the scaled -- and they agree to roundoff:
        # measured worst 2.7e-16 relative (~1.2 ulp) over the shipped
        # (48, 12) and neighbouring grids across the production launch band,
        # scripts/dacc_v2_ebin_probe.py (at commit 48be9a4, retired
        # 2026-09-03). Cosmetic, so it is left as written.
        got_e = 0.5 * M_HE * float((spec * self.g.V2).sum())
        e_rel = abs(got_e - e_launch) / e_launch
        u_rel = abs(got_u - u) / max(abs(u), np.sqrt(s2))
        if (
            not np.isfinite(total)
            or abs(total - 1.0) > CATHODE_JET_MOMENT_REL_TOL
            or e_rel > CATHODE_JET_MOMENT_REL_TOL
            or u_rel > CATHODE_JET_MOMENT_REL_TOL
        ):
            raise ValueError(
                "the DVM cathode jet's launch spectrum did not reach its "
                f"moments at cell {cell}: density {total - 1.0:+.3e} from 1, "
                f"mean energy {e_rel:.3e} relative from "
                f"{e_launch / EV:.6g} eV, drift {u_rel:.3e} relative from "
                f"{u:.6g} cm/s (tolerance "
                f"{CATHODE_JET_MOMENT_REL_TOL:.1e}, T_launch "
                f"{T_launch:.6g} eV). The moment compensation gave up -- a "
                "singular two-basis solve, or a spectrum too narrow or too "
                "fast for this velocity grid -- and the analytic bin masses "
                "it left standing would hand the gas an energy the cathode "
                "surface was not debited. Accepted: widen the grid "
                "(neutral_kinetic_dvm_nvz / _nvp), or leave "
                "neutral_kinetic_dvm_cathode_jet_T_launch_eV unset so the "
                "smear is tied to the local bin width"
            )
        return spec

    def _anode_jet_launch_spectrum(self, e_launch, cell, direction):
        """Return the ANODE backscatter spectrum at ``e_launch`` erg per atom.

        The cathode jet's construction with ``phi_a`` in place of ``phi_c``,
        and one addition: ``direction`` is the sign of the axis the atoms
        leave along, ``-1`` for the mesh's low-z side and ``+1`` for its
        high-z one, so the drift is signed and the spectrum points AWAY from
        the wires it came off. Read
        :meth:`_cathode_jet_launch_spectrum` for why the drift is solved from
        the energy rather than set to ``v_back``, and why a projection that
        misses its moments RAISES rather than returning an approximate
        spectrum; both statements hold here unchanged, against the anode
        energy book's ``backscatter`` row instead of the cathode surface's.
        """
        e_launch = float(e_launch)
        if not np.isfinite(e_launch) or e_launch <= 0.0:
            raise ValueError(
                "the DVM anode jet needs a positive finite launch energy per "
                f"atom at cell {cell} (got {e_launch!r} erg). A cell that "
                "collects ions at ZERO incident energy is not this case and "
                "never reaches here: it launches nothing and its whole "
                "counted stream is born thermal, the fluid-parity reading of "
                "v_back = 0 (:meth:`_split_anode_recycle`). What is left for "
                "this guard is the arithmetically impossible one -- a "
                "POSITIVE committed incident energy whose per-atom launch "
                "energy came out non-finite or non-positive anyway, which "
                "means the counted (particles, energy) pair disagree with "
                "each other rather than that the ions arrived cold"
            )
        v_back = np.sqrt(2.0 * e_launch / M_HE)
        T_launch = self._anode_jet_launch_temperature_eV(v_back)
        if not np.isfinite(T_launch) or T_launch <= 0.0:
            raise ValueError(
                "the DVM anode jet's launch smear must be a positive "
                f"temperature (got {T_launch!r} eV at cell {cell})"
            )
        s2 = T_launch * EV / M_HE
        u2 = v_back * v_back - 3.0 * s2
        if u2 <= 0.0:
            raise ValueError(
                "the DVM anode jet cannot represent a launch energy below "
                "its own smear: cell "
                f"{cell} asks for {e_launch / EV:.6g} eV per atom while the "
                f"launch spectrum's thermal content alone is "
                f"{1.5 * T_launch:.6g} eV (T_launch = {T_launch:.6g} eV). "
                "Accepted: a smaller neutral_kinetic_dvm_anode_jet_"
                "T_launch_eV, or an operating point whose anode sheath "
                "actually makes the backscatter energetic"
            )
        u = float(direction) * np.sqrt(u2)
        spec = self.g.maxwellian(T_launch, u, exact_moments=True)
        total = float(spec.sum())
        got_u = float((spec * self.g.VZ).sum())
        got_e = 0.5 * M_HE * float((spec * self.g.V2).sum())
        e_rel = abs(got_e - e_launch) / e_launch
        u_rel = abs(got_u - u) / max(abs(u), np.sqrt(s2))
        if (
            not np.isfinite(total)
            or abs(total - 1.0) > ANODE_JET_MOMENT_REL_TOL
            or e_rel > ANODE_JET_MOMENT_REL_TOL
            or u_rel > ANODE_JET_MOMENT_REL_TOL
        ):
            raise ValueError(
                "the DVM anode jet's launch spectrum did not reach its "
                f"moments at cell {cell}: density {total - 1.0:+.3e} from 1, "
                f"mean energy {e_rel:.3e} relative from "
                f"{e_launch / EV:.6g} eV, drift {u_rel:.3e} relative from "
                f"{u:.6g} cm/s (tolerance "
                f"{ANODE_JET_MOMENT_REL_TOL:.1e}, T_launch "
                f"{T_launch:.6g} eV). The moment compensation gave up -- a "
                "singular two-basis solve, or a spectrum too narrow or too "
                "fast for this velocity grid -- and the analytic bin masses "
                "it left standing would hand the gas an energy the anode "
                "book did not record as leaving. Accepted: widen the grid "
                "(neutral_kinetic_dvm_nvz / _nvp), or leave "
                "neutral_kinetic_dvm_anode_jet_T_launch_eV unset so the "
                "smear is tied to the local bin width"
            )
        return spec

    def _mesh_axial_momentum_weight(self):
        """Return the per-particle AXIAL momentum weight of a velocity bin.

        ``m v_z``, signed, broadcastable over ``(nvz, nvp)``. Its own method
        for one reason: the mesh-momentum tally is a signed sum, and the
        verification suite's negative control replaces exactly this weight
        with ``m |v_z|`` to show that the sign is what the row is about.
        """
        return M_HE * self.g.VZ

    def f_energy(self):
        """Kinetic energy [erg] carried by the distributions themselves."""
        return float(
            ((self.f_c * self.E_bin).sum(axis=(1, 2)) * self.V_col).sum()
            + ((self.f_a * self.E_bin).sum(axis=(1, 2)) * self.V_ann).sum()
        )

    def pending_energy(self):
        """Kinetic energy [erg] held in the lagged end-wall return buffers."""
        return (
            self._energy_of(self.pend_L_c)
            + self._energy_of(self.pend_R_c)
            + self._energy_of(self.pend_L_a)
            + self._energy_of(self.pend_R_a)
        )

    def total_energy(self):
        """Domain energy inventory including the pending end buffers."""
        return self.f_energy() + self.pending_energy()

    # ------------------------------------------------------- the update

    def collision_frequencies(self, n_i, Ti_eV, u_i):
        """Return ``(nu_cx, nu_el)`` [1/s], shape ``(nz, nvz, nvp)``.

        BGK-like form: the loss frequency of a neutral AT velocity ``v``
        against the local ion Maxwellian, evaluated at the mean relative
        speed

            g_eff^2 = |v - u_i|^2 + 8 k T_i / (pi m_He)

        the standard interpolation between the drift-dominated and
        thermal-dominated limits. The thermal floor carries the FULL ion
        mass, not the equal-mass reduced mass: only the ions are
        Maxwellian here. The neutral's own velocity is RESOLVED by the
        velocity grid and enters exactly, through ``|v - u_i|^2``, so the
        target-averaged relative speed in the drift-free limit is the ion
        thermal speed ``sqrt(8 k T_i / (pi m_He))`` alone. Writing the
        floor as ``8 k T_i / (pi mu)`` with ``mu = m_He/2`` is the
        TWO-Maxwellian form; it counts a neutral thermal spread the grid
        already carries, and inflates ``g_eff`` by up to ``sqrt(2)`` -- and
        ``nu_cx`` with it -- for a neutral slow in the ion frame, falling
        to unity for a fast one. ``nu_el`` is insensitive to the choice:
        ``sigma_iso ~ E^-1/2`` makes ``sigma_iso g_eff`` independent of
        ``g_eff``.

        Two properties of the interpolation itself, so that it is not read
        as the exact average. (i) ``sqrt(w^2 + c_bar^2)`` is an UPPER bound
        on the exact single-neutral mean relative speed ``<|v - u_i|>`` over
        the ion Maxwellian, by at most ``+2.5 %``; the excess vanishes in
        both limits and peaks near ``w/a ~ 1.2-1.5``, with
        ``a = sqrt(2 k T_i / m_He)`` and ``w = |v - u_i|``. (ii) The cross
        sections are evaluated at the energy OF THE MEAN SPEED,
        ``E_rel = (1/2) mu g_eff^2``, which in the drift-free limit is
        ``0.64 k T_i`` -- NOT at the rate-weighted energy of the average
        ``<sigma_b g>``, which sits near ``k T_i``. For a ``sigma_b``
        monotone in E that single-energy evaluation is one-sided as well.

        Charge exchange uses the Phelps He+/He backscatter cross section
        and polarization-elastic scattering the Phelps isotropic cross
        section.

        Both channels are BGK FULL-REPLACEMENT events: the neutral is
        deleted and re-emitted at the local ion Maxwellian, so one event
        transfers the whole ``m (v - u_i)``. That is the correct weight
        for backscatter -- ``mu (1 - cos th) g`` at ``cos th = -1`` and
        ``mu = m/2`` is exactly ``m g`` -- so ``nu_cx`` is the Phelps
        backscatter rate unreduced. It is TWICE the correct weight for
        isotropic scattering, whose angular average ``<1 - cos th> = 1``
        gives ``mu g = m g / 2``. The returned ``nu_el`` therefore carries
        an explicit factor ``ELASTIC_BGK_MOMENTUM_FACTOR = 1/2``, which is
        not a tuning constant but the equal-mass ``mu/m`` ratio; with it
        the arm's effective momentum transfer takes the same CHANNEL
        WEIGHTING as the superseded fluid operator
        ``phelps_momentum_transfer_rate_cm3_s``, namely ``k_b + 0.5
        k_iso``. The two agree exactly on the ELASTIC channel only, where
        ``sigma_iso g_eff`` is constant and the per-bin rate is therefore
        the Maxwellian average; on the CX channel they do NOT, because
        ``nu_cx`` is a per-bin evaluation at ``g_eff`` rather than the rate
        average ``<sigma_b g>`` the fluid coefficient tabulates, and the
        two differ most where the neutral is slow in the ion frame. The
        factor scales the whole elastic channel, so its energy transfer and
        its rebirth throughput are reduced in the same proportion.
        """
        g = self.g
        n_i = np.asarray(n_i, dtype=float)[:, None, None]
        Ti = np.maximum(np.asarray(Ti_eV, dtype=float), 1e-6)[:, None, None]
        u = np.asarray(u_i, dtype=float)[:, None, None]
        w2 = (g.VZ[None, :, :] - u) ** 2 + (g.VP**2)[None, :, :]
        # Only the ions are Maxwellian; the neutral velocity is resolved and
        # is already in w2, so the floor carries m_He, NOT the reduced mass.
        g_eff = np.sqrt(w2 + ion_thermal_g_eff_floor_cm2_s2(Ti))
        E_rel = np.maximum(0.25 * M_HE * g_eff**2 / EV, 1e-9)
        nu_cx = n_i * phelps_he_backscatter_cm2(E_rel) * g_eff
        if self.elastic_model == "off":
            nu_el = np.zeros_like(nu_cx)
        else:
            nu_el = (
                ELASTIC_BGK_MOMENTUM_FACTOR
                * n_i
                * phelps_he_isotropic_cm2(E_rel)
                * g_eff
            )
        return nu_cx, nu_el

    def _march(self, dt, nu_c_loss, nu_a_loss, inflow_c, inflow_a,
               inject_c=None, source_c=None, column_only=False):
        """Backward-Euler implicit upwind march (substep A).

        ``nu_c_loss`` / ``nu_a_loss`` are the per-(cell, bin) NON-zone loss
        frequencies of each zone; the zone-exchange rates are added here so
        the 2x2 coupling stays exactly antisymmetric. ``inflow_*`` are
        boundary ghost DENSITIES keyed ``(-1, +1)`` by domain end.

        ``column_only`` marches the COLUMN alone, with the zone-escape rate
        still on its diagonal but no annulus row to couple to: the
        bounded-chord arm carries the annulus as flights rather than as an
        advected field, so its returning particles arrive as ``source_c``, a
        volume source in density per second, and the annulus outputs come
        back zero. Under the shipped rate arm neither argument is supplied
        and the 2x2 coupling below is the whole update.

        ``inject_c`` carries the INTERIOR-face column inflows (the recycle
        channels) as ghost densities keyed ``(face_index, direction)``. They
        are added to the upstream flux the moment the sweep crosses their
        face, which injects exactly ``|v_z| F A dt`` particles per bin -- the
        same identity the end-wall ghosts satisfy -- while the cells behind
        the face keep passing their own flux through unchanged. Injection is
        applied AFTER any mesh interception at that face: a surface emits
        into the plasma, on the plasma side of an anode mesh.

        An INTERIOR CLOSED FACE (``_closed_face``) stops the column dead:
        the whole upstream flux is tallied to the cell it came from and the
        onward flux is set to zero, so nothing crosses. It is applied after
        any mesh interception and BEFORE ``inject_c``, because the recycle
        ghost at a cathode face is emitted by that surface into the plasma
        and must not be blocked by the surface it left. The annulus is not
        touched: it is the clear bore around the disc.

        An ANNULAR BAFFLE face (``_baffle_tau``) throttles the ANNULUS alone:
        the ``1 - t_f`` share of the upstream annulus flux is tallied onto the
        cell it came from and the rest passes, in exactly the form the mesh
        block above uses for its own annulus share, while the COLUMN flux goes
        by untouched (the disc's bore is at least the plasma radius). It is
        applied after the mesh, which the geometry forbids it to coincide
        with; were the two ever placed on one face they would compose
        multiplicatively, each booking its own share, and both channels would
        still close. Nothing happens under ``column_only``: that arm carries
        the annulus as flights rather than as an advected field, and its
        baffles live in :class:`BoundedChordFlights` instead.

        Returns ``(f_c, f_a, mesh_c, mesh_a, out, mesh_E, closed, mesh_P,
        baffle)`` where
        the mesh arrays are intercepted PARTICLES per emitting cell, ``out``
        maps ``(zone, end)`` to the outgoing particles per bin, ``mesh_E``
        is the ``(column, annulus)`` pair of intercepted ENERGIES [erg] per
        emitting cell, ``closed`` is the matching
        ``(particles, energies)`` pair for the closed faces, each a dict
        keyed by the direction the blocked particles are re-emitted into, and
        ``mesh_P`` is the ``(column, annulus)`` pair of intercepted signed
        AXIAL MOMENTA [g cm/s] -- zeros unless the anode jet is armed, which
        is the only member that publishes them -- and ``baffle`` is the
        ``(particles, energies, signed axial momenta)`` triple of the annular
        baffles, per emitting cell, all zeros unless a baffle is armed.
        All of them are tallied here because these are the channels whose
        interception is summed over velocity bins inside the sweep, so their
        moments cannot be recovered from the particle tally afterwards.
        """
        g = self.g
        nz, nvz, nvp = self.nz, g.nvz, g.nvp
        f_c = np.zeros((nz, nvz, nvp))
        f_a = np.zeros((nz, nvz, nvp))
        mesh_c = np.zeros(nz)
        mesh_a = np.zeros(nz)
        mesh_c_E = np.zeros(nz)
        mesh_a_E = np.zeros(nz)
        # PRESENCE-GATED axial-momentum tallies of the same intercepted
        # particles, in the same sweep and off the same blocked arrays as the
        # counts and the energies. Armed only with the anode jet, because
        # that is the member that made the statement measurable; the weight is
        # signed, which is what distinguishes "the structure took net axial
        # momentum" from "particles hit it".
        mesh_momentum = self.anode_jet is not None
        # The baffle row is presence-gated on ITS OWN structure. The weight is
        # the same array either way and is built once; which tallies read it
        # is decided per channel, so arming one cannot switch the other on and
        # move a row that was zero.
        baffle_momentum = self._baffle_any
        P_bin = (
            self._mesh_axial_momentum_weight()
            if (mesh_momentum or baffle_momentum)
            else None
        )
        mesh_c_P = np.zeros(nz)
        mesh_a_P = np.zeros(nz)
        baffle_a = np.zeros(nz)
        baffle_a_E = np.zeros(nz)
        baffle_a_P = np.zeros(nz)
        # Closed-face tallies, keyed by the direction the blocked particles
        # are RE-EMITTED into (the reverse of the direction that delivered
        # them), on the cell of the side they arrived from.
        closed_n = {+1: np.zeros(nz), -1: np.zeros(nz)}
        closed_E = {+1: np.zeros(nz), -1: np.zeros(nz)}
        out = {}
        inv_dt = 1.0 / dt
        for direction in (+1, -1):
            if direction > 0:
                order = range(nz)
                sel = g.vz > 0
                end_in, end_out = -1, +1
            else:
                order = range(nz - 1, -1, -1)
                sel = g.vz < 0
                end_in, end_out = +1, -1
            vz = np.abs(g.vz[sel])[:, None]
            E_sel = self.E_bin[sel]
            P_all = None if P_bin is None else P_bin[sel]
            P_sel = P_all if mesh_momentum else None
            P_baf = P_all if baffle_momentum else None
            F_c_prev = inflow_c[end_in][sel]
            F_a_prev = None if column_only else inflow_a[end_in][sel]
            for i in order:
                # Upstream face carries the inflow, downstream face the
                # outflow; both are throat areas, so what leaves one cell
                # is exactly what the next receives.
                fi = i if direction > 0 else i + 1
                fo = i + 1 if direction > 0 else i
                in_c = vz * self.face_c[fi] / self.V_col[i]
                out_c = vz * self.face_c[fo] / self.V_col[i]
                if column_only:
                    if fi == self.mesh_face:
                        blocked_c = (1.0 - self.transparency) * F_c_prev
                        j = min(max(i - direction, 0), nz - 1)
                        mesh_c[j] += float(
                            (blocked_c * vz).sum() * self.face_c[fi] * dt
                        )
                        mesh_c_E[j] += float(
                            (blocked_c * vz * E_sel).sum()
                            * self.face_c[fi]
                            * dt
                        )
                        if P_sel is not None:
                            mesh_c_P[j] += float(
                                (blocked_c * vz * P_sel).sum()
                                * self.face_c[fi]
                                * dt
                            )
                        F_c_prev = self.transparency * F_c_prev
                    if self._closed_face[fi]:
                        closed_n[-direction][i - direction] += float(
                            (F_c_prev * vz).sum() * self.face_c[fi] * dt
                        )
                        closed_E[-direction][i - direction] += float(
                            (F_c_prev * vz * E_sel).sum()
                            * self.face_c[fi]
                            * dt
                        )
                        F_c_prev = np.zeros_like(F_c_prev)
                    if inject_c:
                        ghost = inject_c.get((fi, direction))
                        if ghost is not None:
                            F_c_prev = F_c_prev + ghost[sel]
                    a11 = (
                        inv_dt + out_c + nu_c_loss[i][sel]
                        + self.nux[i][None, :]
                    )
                    r1 = self.f_c[i][sel] * inv_dt + in_c * F_c_prev
                    if source_c is not None:
                        r1 = r1 + source_c[i][sel]
                    fc = r1 / a11
                    f_c[i][sel] = fc
                    F_c_prev = fc
                    continue
                if self.V_ann[i] > 0.0:
                    in_a = vz * self.face_a[fi] / self.V_ann[i]
                    out_a = vz * self.face_a[fo] / self.V_ann[i]
                else:
                    in_a = np.zeros_like(vz)
                    out_a = np.zeros_like(vz)
                if fi == self.mesh_face:
                    blocked_c = (1.0 - self.transparency) * F_c_prev
                    blocked_a = (1.0 - self.transparency) * F_a_prev
                    j = min(max(i - direction, 0), nz - 1)
                    mesh_c[j] += float(
                        (blocked_c * vz).sum() * self.face_c[fi] * dt
                    )
                    mesh_a[j] += float(
                        (blocked_a * vz).sum() * self.face_a[fi] * dt
                    )
                    mesh_c_E[j] += float(
                        (blocked_c * vz * E_sel).sum() * self.face_c[fi] * dt
                    )
                    mesh_a_E[j] += float(
                        (blocked_a * vz * E_sel).sum() * self.face_a[fi] * dt
                    )
                    if P_sel is not None:
                        mesh_c_P[j] += float(
                            (blocked_c * vz * P_sel).sum()
                            * self.face_c[fi]
                            * dt
                        )
                        mesh_a_P[j] += float(
                            (blocked_a * vz * P_sel).sum()
                            * self.face_a[fi]
                            * dt
                        )
                    F_c_prev = self.transparency * F_c_prev
                    F_a_prev = self.transparency * F_a_prev
                if self._baffle_any:
                    tau_b = self._baffle_tau[fi]
                    if tau_b is not None:
                        # ANNULUS ONLY: the disc's bore is at least the plasma
                        # radius, so the column flux crosses untouched.
                        blocked_b = (1.0 - tau_b) * F_a_prev
                        j = min(max(i - direction, 0), nz - 1)
                        baffle_a[j] += float(
                            (blocked_b * vz).sum() * self.face_a[fi] * dt
                        )
                        baffle_a_E[j] += float(
                            (blocked_b * vz * E_sel).sum()
                            * self.face_a[fi]
                            * dt
                        )
                        if P_baf is not None:
                            baffle_a_P[j] += float(
                                (blocked_b * vz * P_baf).sum()
                                * self.face_a[fi]
                                * dt
                            )
                        F_a_prev = tau_b * F_a_prev
                if self._closed_face[fi]:
                    # The COLUMN alone: the closed face is the cathode
                    # disc's own footprint, and the annulus around it is the
                    # clear bore the plenum is pumped through.
                    closed_n[-direction][i - direction] += float(
                        (F_c_prev * vz).sum() * self.face_c[fi] * dt
                    )
                    closed_E[-direction][i - direction] += float(
                        (F_c_prev * vz * E_sel).sum() * self.face_c[fi] * dt
                    )
                    F_c_prev = np.zeros_like(F_c_prev)
                if inject_c:
                    ghost = inject_c.get((fi, direction))
                    if ghost is not None:
                        F_c_prev = F_c_prev + ghost[sel]
                nux = self.nux[i][None, :]
                nuxp = self.nuxp[i][None, :]
                a11 = inv_dt + out_c + nu_c_loss[i][sel] + nux
                a12 = -nux * np.ones_like(vz)
                a21 = -nuxp * np.ones_like(vz)
                a22 = inv_dt + out_a + nu_a_loss[i][sel] + nuxp
                r1 = self.f_c[i][sel] * inv_dt + in_c * F_c_prev
                r2 = self.f_a[i][sel] * inv_dt + in_a * F_a_prev
                det = a11 * a22 - a12 * a21
                fc = (r1 * a22 - a12 * r2) / det
                fa = (a11 * r2 - a21 * r1) / det
                f_c[i][sel] = fc
                f_a[i][sel] = fa
                F_c_prev, F_a_prev = fc, fa
            # The last cell marched empties across the open domain end; its
            # downstream-face loss IS the one-way outgoing flux there.
            last = nz - 1 if direction > 0 else 0
            fo_end = nz if direction > 0 else 0
            out[("c", end_out)] = np.zeros((nvz, nvp))
            out[("a", end_out)] = np.zeros((nvz, nvp))
            out[("c", end_out)][sel] = (
                f_c[last][sel] * vz * self.face_c[fo_end] * dt
            )
            out[("a", end_out)][sel] = (
                f_a[last][sel] * vz * self.face_a[fo_end] * dt
            )
        return (
            f_c, f_a, mesh_c, mesh_a, out, (mesh_c_E, mesh_a_E),
            (closed_n, closed_E), (mesh_c_P, mesh_a_P),
            (baffle_a, baffle_a_E, baffle_a_P),
        )

    def _add_face_inflow(self, inject, counts, default_cell, direction,
                         spectrum, dt):
        """Accumulate a wall return as a directed ghost inflow at its face.

        ``counts`` are PARTICLES this update, either per cell or a scalar
        deposited at ``default_cell``. The emitting surface stands on the
        UPSTREAM face of its own cell for the direction it emits into: face
        ``i`` for a ``+z`` emitter, face ``i + 1`` for a ``-z`` one. Dividing
        the counted particles by ``|v_z| A dt`` at exactly the face area the
        march uses makes the injected count equal the counted one bin by bin,
        independent of ``dt`` -- the same construction :func:`_ghost_density`
        performs for the lagged end-wall buffers.
        """
        counts = np.asarray(counts, dtype=float)
        if counts.ndim:
            items = [
                (int(i), float(counts[i])) for i in np.flatnonzero(counts)
            ]
        elif counts:
            items = [(int(default_cell), float(counts))]
        else:
            items = []
        for cell, particles in items:
            face = cell if direction > 0 else cell + 1
            ghost = _ghost_density(
                particles * spectrum, self.face_c[face], dt, self.g
            )
            key = (face, direction)
            if key in inject:
                inject[key] = inject[key] + ghost
            else:
                inject[key] = ghost

    def update(
        self,
        dt,
        *,
        n_i,
        Ti_eV,
        u_i,
        nu_ion,
        ion_counts=None,
        sources=None,
        source_counts=None,
        cathode_jet_incident_erg=None,
        cathode_jet_counts=None,
        anode_jet_incident_erg=None,
        T_s_K=None,
    ):
        """Advance ``(f_c, f_a)`` by one neutral-clock tick of ``dt`` seconds.

        ``nu_ion`` is the velocity-BLIND ionization frequency per cell
        [1/s] -- the registered channel-1 convention; the solver derives
        it from the ionization the plasma actually books. It sets how the
        ionization sink enters the implicit march, i.e. how the surviving
        population is attenuated and transported WITHIN the tick.

        ``ion_counts`` is the separate, and stronger, statement: the
        particle count per column cell that the coupled partner booked as
        ionization over this tick. Supplied, it is what the channel
        debits, exactly, with the march's tally reconciled to it (see
        :meth:`_debit_booked_ionization`); the frequency alone cannot make
        that statement, because the count it removes is measured against
        the POST-march population while the partner booked against the
        pre-tick one. ``None`` leaves the march's own tally standing,
        which is the reading an offline caller with no partner has.
        ``sources`` holds the
        external ledger in atoms/s: ``puff`` (annulus cells),
        ``recombination`` (column cells), ``cathode_face`` /
        ``collector_face`` (column cells, or a scalar attributed to the
        role-resolved ``cath_cell`` / ``coll_cell``), ``anode`` (column
        cells).

        ``source_counts`` is the same external ledger stated in PARTICLES on
        the same keys -- what a coupled partner BOOKED over this tick -- and
        is to ``sources`` what ``ion_counts`` is to ``nu_ion``. A channel
        named there is injected at exactly that count, carrying no ``dt`` and
        so no first-order-in-cadence sampling error; a channel named in
        ``sources`` is multiplied by ``dt`` as before, which is the reading
        an offline caller with no partner has. Naming one channel in both
        raises: they are two different statements about the same particles.

        ``cathode_jet_incident_erg`` is the counted INCIDENT ion energy per
        column cell [erg] that the ``cathode_face`` particles arrived with
        over this tick -- ``sum over the tick of N (phi_c + Ti)`` at each
        contributing state, not a tick-time re-reading of it. Required
        exactly when the engine was built with a ``cathode_jet`` spec and
        refused otherwise; it is the number the partner's surface debit is
        formed from, so the energy the jet hands the gas and the energy the
        surface gave up are one committed quantity rather than two
        agreeing formulas.

        ``cathode_jet_counts`` names which of the ``cathode_face`` particles
        the directed ``R_N`` share is drawn from, for a caller whose
        incident-energy booking covers only SOME of the steps the count
        does -- which is what the cathode-jet arming criterion produces.
        ``None`` means all of them and is the pre-existing arithmetic
        exactly. All zero means the channel is ABSENT for this tick: the
        whole counted stream leaves thermally, no launch spectrum is built,
        and the ledger's jet rows are zero. See
        :meth:`_split_cathode_recycle` for why pairing the two bookings over
        the same steps is what the per-atom launch energy asserts.

        ``anode_jet_incident_erg`` is that same quantity for the ``anode``
        channel -- ``sum over the tick of N (phi_a + Ti)`` per column cell
        [erg], the number the partner's anode energy book is formed from --
        and is required exactly when the engine was built with an
        ``anode_jet`` spec and refused otherwise.

        ``T_s_K`` is the live cathode-surface temperature used for
        the cathode-adjacent surfaces (the stated special case); the wall
        temperature is used everywhere else. That covers the cathode-side
        face of an interior CLOSED FACE as well: the column carries no flux
        across such a face, and each side takes back what it delivered, at
        its own surface temperature.

        The two recycle channels enter as directed INFLOWS at their own
        faces -- a ``+z`` half-flux spectrum at the cathode's upstream face,
        a ``-z`` one at the collector's downstream face -- so they are
        transported and attacked by the loss channels within this same
        update. Every other external channel is a volume birth in substep B.

        Returns the ledger of this update: every birth and loss channel in
        PARTICLES, plus the inventory before/after, so that

            inventory_after - inventory_before == sum(births) - sum(losses)

        holds to roundoff. Its ``"energy"`` entry is the same statement in
        ERG over the same channels (see :meth:`_book_energy_ledger`), plus
        the net surface and exchange rows an energy ledger needs and a
        particle one does not.
        """
        dt = float(dt)
        if dt <= 0.0:
            raise ValueError(f"the DVM update needs a positive dt (got {dt})")
        g = self.g
        sources = {} if sources is None else sources
        source_counts = {} if source_counts is None else source_counts
        self._check_source_channels(sources, source_counts)
        T_s_K = self.T_wall_K if T_s_K is None else float(T_s_K)
        inv_before = self.total_inventory()
        f_before = self.f_inventory()
        e_inv_before = self.total_energy()
        e_f_before = self.f_energy()

        nu_ion = np.asarray(nu_ion, dtype=float)
        nu_cx, nu_el = self.collision_frequencies(n_i, Ti_eV, u_i)
        nu_c_loss = nu_ion[:, None, None] + nu_cx + nu_el
        jump = self.flights is not None
        nu_a_loss = None if jump else (
            self.nuw[:, None, :] * np.ones((self.nz, g.nvz, g.nvp))
        )

        # --- boundary inflow: last update's pending returns, as ghost
        # densities that inject exactly the buffered particle count. The
        # bounded-chord annulus is not marched, so its own buffered returns
        # are re-LAUNCHED off the end plane in substep B instead.
        inflow_c = {
            -1: _ghost_density(self.pend_L_c, self.face_c[0], dt, g),
            +1: _ghost_density(self.pend_R_c, self.face_c[-1], dt, g),
        }
        inflow_a = None if jump else {
            -1: _ghost_density(self.pend_L_a, self.face_a[0], dt, g),
            +1: _ghost_density(self.pend_R_a, self.face_a[-1], dt, g),
        }
        birth_return_L = float(self.pend_L_c.sum() + self.pend_L_a.sum())
        birth_return_R = float(self.pend_R_c.sum() + self.pend_R_a.sum())
        e_return_L = self._energy_of(self.pend_L_c) + self._energy_of(
            self.pend_L_a
        )
        e_return_R = self._energy_of(self.pend_R_c) + self._energy_of(
            self.pend_R_a
        )

        # --- the wall-return (recycle) channels, as directed inflows at the
        # faces they came off. Known before the march (the plasma reports
        # what its boundary term removed), so unlike the end walls they are
        # not lagged. Built here so the counted particles enter substep A and
        # are transported and attacked by the loss channels in the same
        # update -- the physical statement that a recycled atom leaves the
        # surface moving, rather than materializing at rest in the cell the
        # boundary was draining.
        def channel(name):
            return self._channel_counts(name, sources, source_counts, dt)

        puff = channel("puff")
        rec = channel("recombination")
        anode = channel("anode")
        cath = channel("cathode_face")
        coll = channel("collector_face")
        # The cathode-side energetic recycle splits the counted recycle
        # stream: ``R_N`` backscatters (a volume birth in substep B, below)
        # and the remainder keeps the thermal face inflow. Absent the jet
        # spec the whole stream is thermal and nothing here runs.
        cath_thermal, cath_jet, jet_energy = self._split_cathode_recycle(
            cath, cathode_jet_incident_erg, cathode_jet_counts
        )
        # The anode-side energetic recycle splits the counted mesh collection
        # the same way: ``R_N`` backscatters as a directed volume birth away
        # from the wires (substep B, below) and the remainder keeps the
        # at-rest ``M_wall`` rebirth. Absent the jet spec nothing here runs
        # and ``anode_thermal`` IS the array the channel handed over.
        anode_thermal, anode_jet, anode_jet_energy = (
            self._split_anode_recycle(anode, anode_jet_incident_erg)
        )
        inject_c = {}
        spec_cath = g.half_flux_spectrum(T_s_K, +1)
        spec_coll = g.half_flux_spectrum(self.T_wall_K, -1)
        self._add_face_inflow(
            inject_c, cath_thermal, self.cath_cell, +1, spec_cath, dt
        )
        self._add_face_inflow(
            inject_c, coll, self.coll_cell, -1, spec_coll, dt
        )

        vol_c = self.V_col[:, None, None]
        vol_a = self.V_ann[:, None, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_vc = np.where(self.V_col > 0.0, 1.0 / self.V_col, 0.0)
            inv_va = np.where(self.V_ann > 0.0, 1.0 / self.V_ann, 0.0)

        if jump:
            # --- substep A for the annulus: the flights that COMPLETE this
            # tick, taken implicitly at the class rate 1 / (chord / v_perp)
            # so the mean flight time is the class time exactly, and routed
            # through the frozen half-displacement map.
            wall_land = np.zeros((self.nz, g.nvz, g.nvp))
            inner_land = np.zeros((self.nz, g.nvz, g.nvp))
            mesh_a_bins = np.zeros((self.nz, g.nvz, g.nvp))
            baffle_a_bins = np.zeros((self.nz, g.nvz, g.nvp))
            end_a = {-1: np.zeros((g.nvz, g.nvp)), +1: np.zeros((g.nvz, g.nvp))}
            for name in FLIGHT_CLASSES:
                nu_f = self.flights.nu[name]
                surviving = 1.0 / (1.0 + nu_f * dt)
                done = self.f_flight[name] * (nu_f * dt) * surviving * vol_a
                self.f_flight[name] = self.f_flight[name] * surviving
                arrive, stopped, meshed, baffled, eL, eR = self.flights.route(
                    name, done
                )
                if name == "wi":
                    inner_land += arrive
                else:
                    wall_land += arrive
                wall_land += stopped
                mesh_a_bins += meshed
                baffle_a_bins += baffled
                end_a[-1] += eL
                end_a[+1] += eR
            source_c = inner_land * inv_vc[:, None, None] / dt
            # The march runs COLUMN-ONLY here, so its baffle triple is all
            # zeros by construction: on this arm the annulus is flights, and
            # the baffles that act on it are the throats in the map above.
            (
                f_c, f_a, mesh_c, _, out, mesh_E, closed, mesh_P, _no_baffle
            ) = self._march(
                dt, nu_c_loss, None, inflow_c, None, inject_c,
                source_c=source_c, column_only=True,
            )
            out[("a", -1)] = end_a[-1]
            out[("a", +1)] = end_a[+1]
            mesh_a = mesh_a_bins.sum(axis=(1, 2))
            # The annulus mesh tally is per BIN on this arm, so its energy
            # moment is taken directly; the column's comes from the march.
            e_loss_mesh = float(mesh_E[0].sum()) + self._energy_of(mesh_a_bins)
            # The axial-momentum moment follows the same split: the column's
            # comes from the march, the annulus's from those same bins.
            p_loss_mesh = float(mesh_P[0].sum()) + (
                float((mesh_a_bins * self._mesh_axial_momentum_weight()).sum())
                if self.anode_jet is not None
                else 0.0
            )
            # The baffle channel on this arm is entirely the flight map's, so
            # its three moments are taken off the same bins.
            baffle_a = baffle_a_bins.sum(axis=(1, 2))
            e_loss_baffle = self._energy_of(baffle_a_bins)
            p_loss_baffle = (
                float(
                    (
                        baffle_a_bins * self._mesh_axial_momentum_weight()
                    ).sum()
                )
                if self._baffle_any
                else 0.0
            )
            # the column's zone escapes, at the same implicit discretization
            # the march's diagonal used, become inner-surface launches
            escapes = self.nux[:, None, :] * f_c * dt * vol_c
            L_wall = wall_land
        else:
            (
                f_c, f_a, mesh_c, mesh_a, out, mesh_E, closed, mesh_P, baffle
            ) = self._march(
                dt, nu_c_loss, nu_a_loss, inflow_c, inflow_a, inject_c
            )
            L_wall = self.nuw[:, None, :] * f_a * dt * vol_a
            e_loss_mesh = float(mesh_E[0].sum() + mesh_E[1].sum())
            p_loss_mesh = float(mesh_P[0].sum() + mesh_P[1].sum())
            baffle_a, baffle_a_E, baffle_a_P = baffle
            e_loss_baffle = float(baffle_a_E.sum())
            p_loss_baffle = float(baffle_a_P.sum())

        # --- substep A tallies, in PARTICLES, from the marched state
        L_ion = nu_ion[:, None, None] * f_c * dt * vol_c
        L_cx = nu_cx * f_c * dt * vol_c
        L_el = nu_el * f_c * dt * vol_c

        # --- substep B: births at exactly the tallied masses
        M_i = np.empty((self.nz, g.nvz, g.nvp))
        Ti_arr = np.asarray(Ti_eV, dtype=float)
        u_arr = np.asarray(u_i, dtype=float)
        for i in range(self.nz):
            M_i[i] = g.maxwellian(max(float(Ti_arr[i]), 0.02), float(u_arr[i]))

        N_cx = L_cx.sum(axis=(1, 2))
        N_el = L_el.sum(axis=(1, 2))
        N_wall = L_wall.sum(axis=(1, 2))

        # The CX/elastic re-births are the CONSERVING half of substep B:
        # they return this tick's own losses to the same cell, at the same
        # count, in the ion Maxwellian. They are applied BEFORE the counted
        # ionization debit so that the debit's positivity cap measures the
        # inventory the cell actually holds after the tick -- net of the
        # non-conserving losses only. Debiting against the marched state
        # alone counts atoms that never left as missing, which is a
        # shortfall against an inventory the cell does not have.
        birth_cx = (N_cx * inv_vc)[:, None, None] * M_i
        birth_el = (N_el * inv_vc)[:, None, None] * M_i
        f_c = f_c + birth_cx + birth_el

        # --- the counted-particle ionization handshake (K2e). When a
        # coupled partner supplies the count it BOOKED, that count -- not
        # the march's own frequency tally -- is what leaves the column, so
        # the two sides destroy and create the same particles by
        # construction rather than by agreement of two rate formulas.
        ion = self._debit_booked_ionization(ion_counts, L_ion, f_c, vol_c)
        f_c = f_c - ion["drop"]
        L_ion = L_ion + ion["correction"]

        alpha = self.accommodation
        # The non-accommodated cylindrical-wall share, in PARTICLES: the same
        # count under either ``wall_reflection``, on the spectrum that
        # selector chooses. Taken once here because both annulus treatments
        # place the identical array -- the jump arm as a wall LAUNCH, the
        # rate arm as a volume birth.
        wall_return = self._wall_return_counts(L_wall, N_wall, alpha)
        self.last_wall_return = wall_return
        if jump:
            # Every annulus return is a LAUNCH, in particles: accommodated
            # at the cosine-wall spectrum, reflected bin-preserving (the
            # cylinder reverses only the unresolved radial component), the
            # mesh re-emitting what it intercepted at the wall temperature
            # in the cell it was intercepted from, and the puff still born
            # as a 300 K distribution at rest -- the zero-momentum
            # convention, now leaving the wall port rather than standing in
            # the cell. Last update's buffered end-plane returns re-enter
            # here too, already carrying their return spectra.
            launch_wall = (
                alpha * N_wall[:, None, None] * self.M_wall[None, :, :]
                + wall_return
                + mesh_a[:, None, None] * self.M_wall[None, :, :]
            )
            if self._baffle_any:
                # The baffle re-emits what it stopped, at the wall temperature,
                # on the side it stopped it -- as a wall LAUNCH on this arm,
                # exactly as the mesh's share above.
                launch_wall = (
                    launch_wall
                    + baffle_a[:, None, None] * self.M_wall[None, :, :]
                )
            if puff.ndim:
                launch_wall = launch_wall + puff[:, None, None] * self.M_cold
            launch_wall[0] += self.pend_L_a
            launch_wall[-1] += self.pend_R_a
            F_in = self.flights.F_inner[:, None, None]
            launches = {
                "ww": (1.0 - F_in) * launch_wall,
                "wi": F_in * launch_wall,
                "io": escapes,
            }
            f_a = np.zeros_like(f_c)
            for name in FLIGHT_CLASSES:
                self.f_flight[name] = self.f_flight[name] + (
                    self.flights.place(name, launches[name])
                    * inv_va[:, None, None]
                )
                f_a = f_a + self.f_flight[name]
            self.last_flight = {
                "annulus_to_column": inner_land.sum(axis=(1, 2)),
                "column_to_annulus": escapes.sum(axis=(1, 2)),
                "wall_landings": N_wall,
            }
        else:
            birth_wall_acc = (
                alpha * (N_wall * inv_va)[:, None, None]
                * self.M_wall[None, :, :]
            )
            # The non-accommodated share, on the spectrum ``wall_reflection``
            # chose: under ``"specular"`` its own incident bins (the
            # cylinder reverses only the radial direction, which this
            # axisymmetric grid does not resolve, so the fraction returns
            # exactly where it left), under ``"diffuse_elastic"`` an
            # energy-matched cosine spectrum.
            birth_wall_ref = wall_return * inv_va[:, None, None]
            f_a += birth_wall_acc + birth_wall_ref

        # Anode-mesh interception re-emits at the wall temperature in the
        # cell it was intercepted from, on both sides of the mesh.
        f_c += (mesh_c * inv_vc)[:, None, None] * self.M_wall[None, :, :]
        if not jump:
            f_a += (mesh_a * inv_va)[:, None, None] * self.M_wall[None, :, :]
        # Annular-baffle interception re-emits the same way and into the
        # ANNULUS only, in the cell it was intercepted from. PRESENCE-GATED so
        # that a run with no baffle armed adds nothing at all here rather than
        # adding a zero -- the difference between inert and bit-exactly inert.
        # Published per cell for the same reason the closed-face counts are.
        self.last_baffle_counts = baffle_a
        if self._baffle_any and not jump:
            f_a += (
                (baffle_a * inv_va)[:, None, None] * self.M_wall[None, :, :]
            )

        # Closed-face re-emission: what the plate stopped goes back into the
        # side it came from, as a cosine half-flux directed away from the
        # plate, at that side's own surface temperature. Every blocked
        # particle is re-emitted, so the channel conserves particles exactly
        # and the pair cancels in the domain identity the way the mesh does.
        closed_n, closed_E = closed
        # Published per side, keyed by the direction the blocked particles are
        # re-emitted into: the ledger rows are domain totals, and the SIDE is
        # what decides which surface temperature a return carries.
        self.last_closed_counts = closed_n
        n_closed = 0.0
        e_closed_reemit = 0.0
        if self.closed_faces:
            closed_spectra = self._closed_face_spectra(T_s_K)
            for _face, d_in, cell, surface in self._closed_emitters:
                emit = -d_in
                count = float(closed_n[emit][cell])
                if not count:
                    continue
                spectrum = closed_spectra[(surface, emit)]
                f_c[cell] += count * inv_vc[cell] * spectrum
                n_closed += count
                e_closed_reemit += count * self._energy_of(spectrum)
        e_closed_blocked = float(
            closed_E[+1].sum() + closed_E[-1].sum()
        )

        # --- external source ledger (counted particles this update). The two
        # recycle channels are absent here: they entered through the march as
        # face inflows above, which is the whole point of the K2d rework.
        if puff.ndim and not jump:
            # Registered channel 5: the puff is born as a 300 K Maxwellian
            # at rest -- the zero-momentum convention, as a distribution.
            f_a += (puff * inv_va)[:, None, None] * self.M_cold
        if rec.ndim:
            f_c += (rec * inv_vc)[:, None, None] * M_i
        if anode_thermal.ndim:
            f_c += (
                (anode_thermal * inv_vc)[:, None, None]
                * self.M_wall[None, :, :]
            )
        # The anode jet's backscatter share: a VOLUME birth in the cell the
        # collection was counted into, on a narrow shifted Maxwellian pointed
        # AWAY from the mesh -- ``-z`` below the mesh face, ``+z`` at or above
        # it, the fluid channel's own rule -- whose DISCRETE mean energy is
        # the committed launch energy per atom. The energy row is the count
        # times that same discrete mean and the momentum row the count times
        # its discrete DRIFT, so what the gas received and what the two rows
        # say are one number each by construction.
        n_anode_jet = 0.0
        e_anode_jet = 0.0
        p_anode_jet = 0.0
        if anode_jet is not None:
            for cell in np.flatnonzero(anode_jet):
                count = float(anode_jet[cell])
                direction = -1.0 if int(cell) < self.mesh_face else 1.0
                spectrum = self._anode_jet_launch_spectrum(
                    anode_jet_energy[cell] / count, int(cell), direction
                )
                f_c[cell] += count * inv_vc[cell] * spectrum
                n_anode_jet += count
                e_anode_jet += count * self._energy_of(spectrum)
                p_anode_jet += count * float(
                    (spectrum * self._mesh_axial_momentum_weight()).sum()
                )
        # The cathode jet's backscatter share: a VOLUME birth in the cell the
        # recycle was counted into, on a narrow shifted Maxwellian whose
        # DISCRETE mean energy is the committed launch energy per atom. The
        # ledger row is the count times that same discrete mean, so what the
        # gas received and what the row says are one number by construction.
        n_cathode_jet = 0.0
        e_cathode_jet = 0.0
        if cath_jet is not None:
            for cell in np.flatnonzero(cath_jet):
                count = float(cath_jet[cell])
                spectrum = self._cathode_jet_launch_spectrum(
                    jet_energy[cell] / count, int(cell)
                )
                f_c[cell] += count * inv_vc[cell] * spectrum
                n_cathode_jet += count
                e_cathode_jet += count * self._energy_of(spectrum)

        # --- end walls: pump what sticks, buffer what returns
        out_L = float(out[("c", -1)].sum() + out[("a", -1)].sum())
        out_R = float(out[("c", +1)].sum() + out[("a", +1)].sum())
        self.pend_L_c = _end_return(
            out[("c", -1)], self.s_L, alpha, self.mirror,
            g.half_flux_spectrum(T_s_K, +1),
        )
        self.pend_L_a = _end_return(
            out[("a", -1)], self.s_L, alpha, self.mirror,
            g.half_flux_spectrum(T_s_K, +1),
        )
        self.pend_R_c = _end_return(
            out[("c", +1)], self.s_R, alpha, self.mirror,
            g.half_flux_spectrum(self.T_wall_K, -1),
        )
        self.pend_R_a = _end_return(
            out[("a", +1)], self.s_R, alpha, self.mirror,
            g.half_flux_spectrum(self.T_wall_K, -1),
        )

        self.f_c = f_c
        self.f_a = f_a
        inv_after = self.total_inventory()
        energy = self._book_energy_ledger(
            e_inv_before=e_inv_before,
            e_f_before=e_f_before,
            alpha=alpha,
            L_ion=L_ion,
            L_cx=L_cx,
            L_el=L_el,
            L_wall=L_wall,
            N_wall=N_wall,
            wall_return=wall_return,
            N_cx=N_cx,
            N_el=N_el,
            M_i=M_i,
            mesh_c=mesh_c,
            mesh_a=mesh_a,
            e_loss_mesh=e_loss_mesh,
            baffle_a=baffle_a,
            e_loss_baffle=e_loss_baffle,
            e_closed_blocked=e_closed_blocked,
            e_closed_reemit=e_closed_reemit,
            out=out,
            e_return_L=e_return_L,
            e_return_R=e_return_R,
            puff=puff,
            rec=rec,
            anode=anode_thermal,
            cath=cath_thermal,
            coll=coll,
            spec_cath=spec_cath,
            spec_coll=spec_coll,
            e_birth_cathode_jet=e_cathode_jet,
            e_birth_anode_jet=e_anode_jet,
        )

        # --- plasma coupling: minus the moments of the kinetic operators
        self._book_ionization_ledger(ion)
        self._book_transfer(dt, L_ion, L_cx, L_el, birth_cx, birth_el, rec,
                            M_i, u_arr, np.asarray(n_i, dtype=float))
        self.Tn_col_eV = self.column_temperature_eV()
        self.updates += 1

        ledger = {
            "dt": dt,
            "inventory_before": inv_before,
            "inventory_after": inv_after,
            "f_inventory_before": f_before,
            "f_inventory_after": self.f_inventory(),
            "loss_ionization": float(L_ion.sum()),
            "loss_cx": float(L_cx.sum()),
            "loss_elastic": float(L_el.sum()),
            "loss_wall": float(N_wall.sum()),
            "loss_mesh_blocked": float(mesh_c.sum() + mesh_a.sum()),
            # The annular baffles. Exactly zero with none armed, and the pair
            # below is the same count by construction: the disc transmits
            # nothing of what it stopped and keeps nothing.
            "loss_baffle_blocked": float(baffle_a.sum()),
            "loss_closed_face_blocked": n_closed,
            "loss_end_out_L": out_L,
            "loss_end_out_R": out_R,
            "loss_pump_L": self.s_L * out_L,
            "loss_pump_R": self.s_R * out_R,
            "birth_cx": float(N_cx.sum()),
            "birth_elastic": float(N_el.sum()),
            "birth_wall_accommodated": float(alpha * N_wall.sum()),
            "birth_wall_reflected": float((1.0 - alpha) * N_wall.sum()),
            "birth_mesh_reemit": float(mesh_c.sum() + mesh_a.sum()),
            "birth_baffle_reemit": float(baffle_a.sum()),
            # The same count the closed faces blocked, by construction: the
            # plate transmits nothing and keeps nothing.
            "birth_closed_face_reemit": n_closed,
            "birth_end_return_L": birth_return_L,
            "birth_end_return_R": birth_return_R,
            "birth_puff": float(puff.sum()),
            "birth_recombination": float(rec.sum()),
            "birth_cathode_face": float(cath_thermal.sum()),
            "birth_cathode_jet": n_cathode_jet,
            "birth_collector_face": float(coll.sum()),
            "birth_anode": float(anode_thermal.sum()),
            "birth_anode_jet": n_anode_jet,
            # The counted handshake, per update: what the partner booked,
            # what this update actually debited, and what it still owes.
            # All three are zero on a standalone update (no partner count).
            "ion_booked": float(ion["booked"].sum()),
            "ion_debt_carried": float(self.ion_debt.sum()),
            "ion_limited_cells": float(np.count_nonzero(ion["limited"])),
            # The same tick in ERG, per channel; see _book_energy_ledger.
            "energy": energy,
        }
        if self.anode_jet is not None:
            # PRESENCE-GATED, so a run without the anode jet carries the
            # ledger it always carried. Two readings, not a ledger: what the
            # jet launched (signed, the two sides pointing away from the
            # mesh and so partly cancelling), and what the mesh intercepted
            # and KEPT -- the wires re-emit at rest, so every bit of the
            # intercepted axial momentum stays on the structure.
            ledger["momentum_anode_jet"] = p_anode_jet
            ledger["momentum_mesh_absorbed"] = p_loss_mesh
        if self._baffle_any:
            # PRESENCE-GATED on the baffles themselves, so a run without them
            # carries the ledger it always carried. One reading, not a ledger:
            # the signed axial momentum the discs KEPT -- they re-emit on the
            # wall spectrum, which carries none, so all of it stays on them.
            ledger["momentum_baffle_absorbed"] = p_loss_baffle
        self.last_ledger = ledger
        return ledger

    @staticmethod
    def _check_source_channels(sources, source_counts):
        """Refuse a counted external ledger this engine cannot read as written.

        A ``source_counts`` key outside :data:`LEDGER_EXTERNAL_BIRTHS` would
        be silently inert, and a channel given both as a rate and as a count
        is two different statements about the same particles with no rule
        for which wins. Both are configuration errors and both raise here,
        before anything is injected.
        """
        accepted = set(LEDGER_EXTERNAL_BIRTHS) - set(LEDGER_ENGINE_SPLIT_BIRTHS)
        split = sorted(
            (set(sources) | set(source_counts)) & set(LEDGER_ENGINE_SPLIT_BIRTHS)
        )
        if split:
            raise ValueError(
                f"DVM source channel(s) {split} are SPLIT OUT of another "
                "channel by the engine and are not fed directly: each surface "
                "jet's backscatter share is taken from its parent counted "
                "stream ('cathode_jet' from 'cathode_face', 'anode_jet' from "
                "'anode'), so feeding one here would count the same recycled "
                "particles twice"
            )
        unknown = sorted(set(source_counts) - accepted)
        if unknown:
            raise ValueError(
                f"unknown counted DVM source channel(s) {unknown} "
                "(silent/inert sources are forbidden); this engine books "
                f"{sorted(accepted)}"
            )
        both = sorted(set(sources) & set(source_counts))
        if both:
            raise ValueError(
                f"DVM source channel(s) {both} were given as a rate AND as "
                "a count: a channel is one or the other, never both"
            )

    def _channel_counts(self, name, sources, source_counts, dt):
        """Return one external channel's PARTICLES for this update.

        From ``source_counts`` when the partner counted the channel, in
        which case the count is what is injected exactly; otherwise the
        historical ``rate * dt``, which is what a standalone caller and
        every uncounted channel still use.
        """
        counted = source_counts.get(name)
        if counted is None:
            return np.asarray(sources.get(name, 0.0), dtype=float) * dt
        counted = np.asarray(counted, dtype=float)
        if counted.shape != (self.nz,):
            raise ValueError(
                f"source_counts[{name!r}] must carry one particle count per "
                f"cell (got shape {counted.shape}, expected {(self.nz,)})"
            )
        if not np.all(np.isfinite(counted)):
            raise ValueError(f"source_counts[{name!r}] must be finite")
        return counted

    def _split_cathode_recycle(self, cath, incident_erg, jet_counts=None):
        """Split the counted cathode recycle into its thermal and jet shares.

        Returns ``(thermal, jet, jet_energy)``. Without a ``cathode_jet``
        spec the whole stream is thermal, ``jet`` is ``None`` and the
        returned ``thermal`` IS the array handed in, so the off path runs
        exactly the arithmetic it ran before the channel existed.

        Armed, ``jet`` is ``R_N`` of the counted particles and
        ``jet_energy`` is ``R_E`` of the counted INCIDENT energy -- the
        total reflected energy fraction, so the per-atom launch energy is
        ``jet_energy / jet = (R_E/R_N)(phi_c + Ti)`` exactly. The thermal
        share is the REMAINDER rather than ``(1 - R_N)`` of the count, so
        the two shares sum to the counted stream to the bit and the particle
        ledger's external total is untouched by the split.

        ``jet_counts`` names WHICH of the counted particles the directed
        share is taken from, and exists because the two halves of that
        per-atom identity can be booked over DIFFERENT step sets. The
        arming criterion suppresses the incident-energy booking on censored
        steps while the recycle COUNT keeps accruing on every step, so
        ``R_N`` of the full count paired with energy from the armed steps
        alone is not ``(R_E/R_N)(phi_c + Ti)`` -- it is that number diluted
        by however much of the tick was censored, and where the whole tick
        was censored it is 0/0. Passing the armed-step counts here keeps
        numerator and denominator over the same steps, which is what the
        identity above actually asserts.

        ``None`` (the default) means "all of them", the reading that applies
        when nothing is censoring the channel, and it is bit-identical to
        the arithmetic this method ran before the parameter existed.

        A fully censored tick therefore arrives with ``jet_counts`` all
        zero, and the directed share is then exactly zero: no atom is placed
        on the launch spectrum, the spectrum is never built, and the whole
        counted stream leaves on the thermal remainder -- which is the same
        routing an engine with no ``cathode_jet`` spec at all would give it.
        That is the ratified reading: below the arming current the jet is
        ABSENT, not launched-then-censored.
        """
        if self.cathode_jet is None:
            if incident_erg is not None:
                raise ValueError(
                    "cathode_jet_incident_erg was supplied to a DVM built "
                    "with no cathode_jet spec: there is no energetic "
                    "backscatter channel to receive it, and a silently inert "
                    "energy booking is exactly the misbooking the counted "
                    "handshake exists to prevent"
                )
            return cath, None, None
        if incident_erg is None:
            raise ValueError(
                "the DVM cathode jet is armed and needs "
                "cathode_jet_incident_erg -- the counted incident ion energy "
                "[erg] per column cell that the cathode_face particles "
                "arrived with over this tick. It is the same committed "
                "number the surface debit is formed from; deriving a second "
                "one here would let the reflected energy be created twice"
            )
        incident = np.asarray(incident_erg, dtype=float)
        if incident.shape != (self.nz,):
            raise ValueError(
                "cathode_jet_incident_erg must carry one incident energy per "
                f"column cell (got shape {incident.shape}, expected "
                f"{(self.nz,)})"
            )
        if not np.all(np.isfinite(incident)) or np.any(incident < 0.0):
            raise ValueError(
                "cathode_jet_incident_erg must be finite and non-negative "
                f"(got min {float(np.min(incident))!r}, max "
                f"{float(np.max(incident))!r})"
            )
        counts = np.asarray(cath, dtype=float)
        if not counts.ndim:
            # The scalar convention deposits at the role-resolved cathode
            # cell; carrying it as the equivalent per-cell row keeps the
            # split, the birth and the face inflow on one placement rule.
            scalar = float(counts)
            counts = np.zeros(self.nz)
            counts[self.cath_cell] = scalar
        if jet_counts is None:
            directed_from = counts
        else:
            directed_from = np.asarray(jet_counts, dtype=float)
            if directed_from.shape != (self.nz,):
                raise ValueError(
                    "cathode_jet_counts must carry one count per column "
                    f"cell (got shape {directed_from.shape}, expected "
                    f"{(self.nz,)})"
                )
            if not np.all(np.isfinite(directed_from)) or np.any(
                directed_from < 0.0
            ):
                raise ValueError(
                    "cathode_jet_counts must be finite and non-negative "
                    f"(got min {float(np.min(directed_from))!r}, max "
                    f"{float(np.max(directed_from))!r})"
                )
            if np.any(directed_from > counts * (1.0 + 1.0e-12) + 1.0e-30):
                raise ValueError(
                    "cathode_jet_counts is the ARMED-STEP share of the "
                    "counted cathode recycle and cannot exceed the counted "
                    "stream itself -- a directed share drawn from more "
                    "particles than arrived would create them "
                    f"(worst excess {float(np.max(directed_from - counts))!r})"
                )
        jet = float(self.cathode_jet["R_N"]) * directed_from
        thermal = counts - jet
        jet_energy = float(self.cathode_jet["R_E"]) * incident
        return thermal, jet, jet_energy

    def _split_anode_recycle(self, anode, incident_erg):
        """Split the counted anode collection into its thermal and jet shares.

        The anode twin of :meth:`_split_cathode_recycle`, with the same
        contract: ``(thermal, jet, jet_energy)``; without an ``anode_jet``
        spec the whole stream is thermal, ``jet`` is ``None`` and the returned
        ``thermal`` IS the array handed in, so the off path runs exactly the
        arithmetic it ran before the channel existed. Armed, ``jet`` is
        ``R_N`` of the counted particles, ``jet_energy`` is ``R_E`` of the
        counted INCIDENT energy, and the thermal share is the REMAINDER rather
        than ``(1 - R_N)`` of the count, so the two sum to the counted stream
        to roundoff (measured worst 2.1e-16).

        **ZERO INCIDENT ENERGY IS A LEGAL, BOOKED STATE, PER CELL.** A cell
        whose committed incident energy is exactly zero launches NOTHING: its
        whole counted stream is born thermal, its
        ``jet`` entry is exactly ``0``, and its energy and momentum
        contributions are exactly ``0`` with them. That is FLUID PARITY, not a
        fallback -- under the fluid spec
        :func:`~cablp.solvers._sim1d.physics.sources.anode_jet_backscatter_speed`
        a zero clamped incident energy gives ``v_back = 0``, which makes the
        ``R_N`` share indistinguishable from thermal desorption; the DVM books
        it as such rather than inventing energy the ion did not bring. The
        state is REACHED, not hypothetical: the anode sheath is
        electron-attracting before breakdown, so ``max(phi_a + Ti, 0)`` is
        exactly zero over the discharge's first accepted steps while the mesh
        still collects (`scripts/b4aj_phi_a_probe.py` (at commit 48be9a4,
        retired 2026-09-03)).

        **The split is PER CELL, not per tick**: cells that carry a positive
        incident energy in the same tick launch normally beside cells that
        carry none.

        One thing differs from the cathode form. The counted ``anode`` row is
        per-cell by construction -- the partner folds the mesh's column and
        annulus collection onto the column cells it was collected at -- so
        there is no scalar convention to resolve, and a caller that supplies
        the channel as a bare RATE (the standalone convention, which produces
        a 0-d array) is arming a channel with no placement. That is refused
        rather than guessed at, unless the rate is zero, where there is
        nothing to place.
        """
        if self.anode_jet is None:
            if incident_erg is not None:
                raise ValueError(
                    "anode_jet_incident_erg was supplied to a DVM built with "
                    "no anode_jet spec: there is no energetic backscatter "
                    "channel to receive it, and a silently inert energy "
                    "booking is exactly the misbooking the counted handshake "
                    "exists to prevent"
                )
            return anode, None, None
        if incident_erg is None:
            raise ValueError(
                "the DVM anode jet is armed and needs anode_jet_incident_erg "
                "-- the counted incident ion energy [erg] per column cell "
                "that the anode-collected ions arrived with over this tick. "
                "It is the same committed number the anode energy book is "
                "formed from; deriving a second one here would let the "
                "reflected energy be created twice"
            )
        incident = np.asarray(incident_erg, dtype=float)
        if incident.shape != (self.nz,):
            raise ValueError(
                "anode_jet_incident_erg must carry one incident energy per "
                f"column cell (got shape {incident.shape}, expected "
                f"{(self.nz,)})"
            )
        if not np.all(np.isfinite(incident)) or np.any(incident < 0.0):
            raise ValueError(
                "anode_jet_incident_erg must be finite and non-negative "
                f"(got min {float(np.min(incident))!r}, max "
                f"{float(np.max(incident))!r})"
            )
        counts = np.asarray(anode, dtype=float)
        if not counts.ndim:
            scalar = float(counts)
            if scalar:
                raise ValueError(
                    "the DVM anode jet launches its backscatter AWAY from the "
                    "mesh on the side each ion was collected from, so it "
                    "needs the counted 'anode' channel PER CELL; it was given "
                    f"the scalar rate {scalar!r}, which names no side. "
                    "Accepted: source_counts['anode'] as a per-cell row, or "
                    "anode_jet=None"
                )
            counts = np.zeros(self.nz)
        jet = float(self.anode_jet["R_N"]) * counts
        # THE ZERO-INCIDENT RULE, per cell: no incident energy, no
        # backscatter. Masking the JET share (rather than the count) is what
        # makes the whole of that cell's stream fall through to ``thermal``
        # below -- ``thermal = counts - jet`` is unchanged, so the two shares
        # still sum to the counted stream to the bit whichever branch a cell
        # took, and the identity the particle ledger closes on does not
        # acquire a special case. The energy row needs no mask: it is already
        # zero exactly where this is.
        jet = np.where(incident > 0.0, jet, 0.0)
        thermal = counts - jet
        jet_energy = float(self.anode_jet["R_E"]) * incident
        return thermal, jet, jet_energy

    def _debit_booked_ionization(self, ion_counts, L_ion, f_c, vol_c):
        """Reconcile the march's ionization tally with a partner's booking.

        ``ion_counts`` is the particle count per cell that the coupled
        partner (the fluid plasma) booked as ionization over this tick --
        counted particles, not a rate. ``None`` means no partner supplied
        one: the march's own tally then stands unchanged and every entry
        below is zero, which is the standalone-engine reading and the only
        one available to an offline caller.

        With a count in hand the debit is renormalized to it. The march
        already removed ``sum(L_ion)`` from each cell; the remainder is
        taken from ``f_c`` in proportion to ``f_c`` itself, so the debit
        is velocity-BLIND over the population it draws from -- the same
        convention ``nu_ion`` itself carries -- and biases no part of that
        population over another. A negative remainder is a credit and puts
        atoms back, which is what a partner that booked less than the
        march removed is owed.

        ``f_c`` must be the column population as it stands after the
        tick's CONSERVING re-births have been applied -- the marched state
        plus the CX/elastic returns, which leave the cell's atom count
        untouched and come back within this same tick. That is the
        inventory the cell genuinely holds, and it is therefore both the
        population the debit draws from and the ceiling it is capped at;
        the two are the same array by construction, which is what keeps
        the drop from driving a bin negative. Handed the marched state
        alone the cap would measure atoms that never left as missing.

        Positivity is a hard constraint: a cell can give up at most the
        atoms it holds. The shortfall is never clipped away -- it is
        carried in ``ion_debt`` and re-offered on the next tick, exactly
        as the deferred momentum/energy transfer carries its own. The
        identity the ledger states, per cell and at every update, is

            ion_removed_cum + ion_debt == ion_booked_cum

        Returns the per-bin ``correction`` to apply to ``f_c`` (in
        particles) and the per-cell scalars the ledger books.
        """
        zeros = np.zeros(self.nz)
        if ion_counts is None:
            return {
                "drop": np.zeros_like(L_ion),
                "correction": np.zeros_like(L_ion),
                "booked": zeros,
                "removed": L_ion.sum(axis=(1, 2)),
                "shortfall": zeros.copy(),
                "limited": np.zeros(self.nz, dtype=bool),
                "counted": False,
            }
        booked = np.asarray(ion_counts, dtype=float)
        if booked.shape != (self.nz,):
            raise ValueError(
                "ion_counts must carry one particle count per column cell "
                f"(got shape {booked.shape}, expected {(self.nz,)})"
            )
        if not np.all(np.isfinite(booked)):
            raise ValueError("ion_counts must be finite")
        target = booked + self.ion_debt
        marched = L_ion.sum(axis=(1, 2))
        held = (f_c * vol_c).sum(axis=(1, 2))
        enough = held > 0.0
        frac = np.where(enough, (target - marched) / np.where(enough, held, 1.0), 0.0)
        # A cell cannot surrender more than it holds; what it cannot give
        # becomes debt rather than a negative distribution.
        applied = np.minimum(frac, 1.0)
        # As a DENSITY first: ``f_c - 1.0 * f_c`` is exactly zero, while the
        # same debit routed through a multiply-then-divide by the cell
        # volume leaves a roundoff-negative distribution behind.
        drop = applied[:, None, None] * f_c
        correction = drop * vol_c
        removed = marched + correction.sum(axis=(1, 2))
        return {
            "drop": drop,
            "correction": correction,
            "booked": booked,
            "removed": removed,
            "shortfall": target - removed,
            "limited": np.where(enough, frac > 1.0, target > marched),
            "counted": True,
        }

    def _book_ionization_ledger(self, ion):
        """Fold one update's counted-ionization reconciliation into the ledger."""
        if not ion["counted"]:
            return
        self.ion_booked_cum = self.ion_booked_cum + ion["booked"]
        self.ion_removed_cum = self.ion_removed_cum + ion["removed"]
        self.ion_debt = ion["shortfall"]
        limited = ion["limited"]
        if np.any(limited):
            self.ion_shortfall_updates += 1
            self.ion_shortfall_cell_updates = (
                self.ion_shortfall_cell_updates + limited
            )

    def _book_energy_ledger(
        self,
        *,
        e_inv_before,
        e_f_before,
        alpha,
        L_ion,
        L_cx,
        L_el,
        L_wall,
        N_wall,
        wall_return,
        N_cx,
        N_el,
        M_i,
        mesh_c,
        mesh_a,
        e_loss_mesh,
        baffle_a,
        e_loss_baffle,
        e_closed_blocked,
        e_closed_reemit,
        out,
        e_return_L,
        e_return_R,
        puff,
        rec,
        anode,
        cath,
        coll,
        spec_cath,
        spec_coll,
        e_birth_cathode_jet,
        e_birth_anode_jet,
    ):
        """Return this update's ENERGY ledger [erg], channel by channel.

        Every loss and birth row is the kinetic-energy moment of exactly the
        per-bin particle array its namesake in the particle ledger counted,
        so the distribution identity

            Delta(sum f V E) == sum(energy births) - sum(energy losses)

        closes by the same algebra that closes the particle one: the march
        is bin-diagonal, the zone coupling moves particles at fixed bin, and
        the per-bin energy weight is therefore a constant that passes
        straight through both statements. Where a birth is a counted number
        times a FIXED emission spectrum -- the cylindrical wall, the anode
        mesh, the puff, the two recycle faces, the anode rebirths -- the
        moment is that number times the spectrum's own mean energy, which is
        the same product the distribution received.

        The internal channels do NOT cancel here the way they do in
        particles: a surface returns as many atoms as it took but not the
        same energy, and the collisional channels re-emit against the ION
        population rather than against what they removed. Those net
        transfers are the ``net_*`` rows, and with them present the domain
        identity closes too (:func:`ledger_energy_residual`).

        ``surface_end_*`` is what the end wall KEPT -- the outflow less the
        pumped share, which is its own loss row, less the buffered return,
        which is still inside the inventory.
        """
        # Losses, from the arrays substep A actually removed.
        e_loss_ionization = self._energy_of(L_ion)
        e_loss_cx = self._energy_of(L_cx)
        e_loss_elastic = self._energy_of(L_el)
        e_loss_wall = self._energy_of(L_wall)
        e_loss_end_L = self._energy_of(out[("c", -1)]) + self._energy_of(
            out[("a", -1)]
        )
        e_loss_end_R = self._energy_of(out[("c", +1)]) + self._energy_of(
            out[("a", +1)]
        )
        # The pumped fraction is a per-bin multiple of the outflow, so its
        # energy is that same fraction of the outflow's energy exactly.
        e_loss_pump_L = self.s_L * e_loss_end_L
        e_loss_pump_R = self.s_R * e_loss_end_R

        # Births. The collisional and recombination channels re-emit at the
        # local ion Maxwellian, so they carry that distribution's own mean
        # energy per cell; everything else carries a fixed spectrum's.
        E_Mi = (M_i * self.E_bin).sum(axis=(1, 2))
        e_birth_cx = float((N_cx * E_Mi).sum())
        e_birth_elastic = float((N_el * E_Mi).sum())
        e_birth_wall_accommodated = (
            alpha * float(N_wall.sum()) * self.E_wall_mean
        )
        # Under ``wall_reflection = "specular"`` the non-accommodated share
        # keeps its incident bin, hence its incident energy: the cylindrical
        # wall reverses only the unresolved radial component and an end wall
        # reverses v_z, which the symmetric axis mirrors exactly. Under
        # ``"diffuse_elastic"`` it comes back on a solved spectrum instead,
        # so the row is the moment of the ARRAY that was actually placed --
        # never of the energy the solve was aiming at, which is what keeps
        # the ledger closed by construction rather than by convergence.
        if self.wall_reflection == "specular":
            e_birth_wall_reflected = (1.0 - alpha) * e_loss_wall
        else:
            e_birth_wall_reflected = self._energy_of(wall_return)
        e_birth_mesh_reemit = (
            float(mesh_c.sum()) + float(mesh_a.sum())
        ) * self.E_wall_mean
        # The baffle re-emits at the wall temperature on the wall spectrum, so
        # its birth energy is the counted number times that spectrum's own mean
        # -- the same product the annulus received, as for the mesh.
        e_birth_baffle_reemit = float(baffle_a.sum()) * self.E_wall_mean
        e_birth_puff = float(puff.sum()) * self.E_cold_mean
        e_birth_recombination = (
            float((rec * E_Mi).sum()) if rec.ndim else 0.0
        )
        e_birth_anode = float(anode.sum()) * self.E_wall_mean
        e_birth_cathode_face = float(cath.sum()) * self._energy_of(spec_cath)
        e_birth_collector_face = float(coll.sum()) * self._energy_of(spec_coll)

        e_pending_L = self._energy_of(self.pend_L_c) + self._energy_of(
            self.pend_L_a
        )
        e_pending_R = self._energy_of(self.pend_R_c) + self._energy_of(
            self.pend_R_a
        )
        return {
            "loss_ionization": e_loss_ionization,
            "loss_cx": e_loss_cx,
            "loss_elastic": e_loss_elastic,
            "loss_wall": e_loss_wall,
            "loss_mesh_blocked": e_loss_mesh,
            "loss_baffle_blocked": e_loss_baffle,
            "loss_closed_face_blocked": e_closed_blocked,
            "loss_end_out_L": e_loss_end_L,
            "loss_end_out_R": e_loss_end_R,
            "loss_pump_L": e_loss_pump_L,
            "loss_pump_R": e_loss_pump_R,
            "birth_cx": e_birth_cx,
            "birth_elastic": e_birth_elastic,
            "birth_wall_accommodated": e_birth_wall_accommodated,
            "birth_wall_reflected": e_birth_wall_reflected,
            "birth_mesh_reemit": e_birth_mesh_reemit,
            "birth_baffle_reemit": e_birth_baffle_reemit,
            "birth_closed_face_reemit": e_closed_reemit,
            "birth_end_return_L": e_return_L,
            "birth_end_return_R": e_return_R,
            "birth_puff": e_birth_puff,
            "birth_recombination": e_birth_recombination,
            "birth_cathode_face": e_birth_cathode_face,
            # Already the counted number times the DISCRETE mean energy of the
            # spectrum that was placed -- summed at the birth site rather than
            # rebuilt here, so the row cannot describe a spectrum other than
            # the one the gas received.
            "birth_cathode_jet": e_birth_cathode_jet,
            "birth_collector_face": e_birth_collector_face,
            "birth_anode": e_birth_anode,
            # As with the cathode jet: already the counted number times the
            # DISCRETE mean energy of the spectra that were placed, summed at
            # the birth site rather than rebuilt here.
            "birth_anode_jet": e_birth_anode_jet,
            "net_surface_wall": (
                e_loss_wall
                - e_birth_wall_accommodated
                - e_birth_wall_reflected
            ),
            "net_surface_mesh": e_loss_mesh - e_birth_mesh_reemit,
            # The accommodation exchange at the annular baffles: what the disc
            # took out of the annulus less what it gave back at the wall
            # temperature. Exactly zero with no baffle armed.
            "net_surface_baffle": e_loss_baffle - e_birth_baffle_reemit,
            # The accommodation exchange at the closed faces: what the plate
            # took out of the gas less what it put back at its own surface
            # temperature. Booked exactly like the other surface channels.
            "net_surface_closed_face": e_closed_blocked - e_closed_reemit,
            "net_surface_end_L": e_loss_end_L - e_loss_pump_L - e_pending_L,
            "net_surface_end_R": e_loss_end_R - e_loss_pump_R - e_pending_R,
            "net_exchange_cx": e_loss_cx - e_birth_cx,
            "net_exchange_elastic": e_loss_elastic - e_birth_elastic,
            "inventory_before": e_inv_before,
            "inventory_after": self.total_energy(),
            "f_inventory_before": e_f_before,
            "f_inventory_after": self.f_energy(),
            "pending_after_L": e_pending_L,
            "pending_after_R": e_pending_R,
        }

    def _book_transfer(self, dt, L_ion, L_cx, L_el, birth_cx, birth_el, rec,
                       M_i, u_i, n_i):
        """Book the plasma-side momentum/energy/particle transfer.

        Every entry is MINUS a measured moment of a kinetic operator, so
        the fluid gain and the kinetic loss are antisymmetric to roundoff
        by construction rather than by agreement of two formulas:

        - ionization: the plasma gains the whole momentum and energy of
          the ionized population (registered channel 1, refining the R4.2
          ``(u_n, 300 K)`` booking);
        - charge exchange and elastic scattering: the plasma gains what
          the lost neutrals carried and pays for what the replacement
          neutrals carry away;
        - recombination: the plasma pays for the born neutral.

        The energy moment is a TOTAL kinetic energy; the fluid ``Ei`` row
        is an internal energy, so the bulk term is removed with the same
        decomposition the ``ionization_birth_energy_model="conservative"``
        booking uses, ``d(KE) = u dM - (1/2) m u^2 dN``.

        The CX/elastic PAIR is additionally booked on its own
        (``M_transfer_pair``, ``Ei_transfer_pair``) with the per-ion
        collision frequency ``nu_pair`` [1/s] that carries it, because the
        pair is a linear RELAXATION of the fluid rows and the plasma-side
        integrator can hold it as one (see the solver's
        ``neutral_kinetic_dvm_transfer_hold`` selector); the ionization and
        recombination rows are a source and are not part of that pair. The
        measured moments of the LOST population are also published as
        ``u_n_eff`` [cm/s] and ``T_eff_eV``, the drift and the ion-frame
        temperature the pair relaxes the fluid towards. ``T_eff_eV``
        carries the frictional term ``(m/3k)|u_n - u_i|^2`` by
        construction: it is the second moment of the lost neutrals taken
        about the ION drift, not the neutral gas temperature.
        """
        g = self.g
        VZ = g.VZ[None, :, :]
        V2 = g.V2[None, :, :]
        vol_c = np.maximum(self.V_col, 1e-300)

        def moments(counts):
            return (
                counts.sum(axis=(1, 2)),
                M_HE * (counts * VZ).sum(axis=(1, 2)),
                0.5 * M_HE * (counts * V2).sum(axis=(1, 2)),
            )

        N_ion, P_ion, E_ion = moments(L_ion)
        N_cx_l, P_cx_l, E_cx_l = moments(L_cx + L_el)
        # Births are densities per bin; convert back to particles.
        births = (birth_cx + birth_el) * self.V_col[:, None, None]
        N_cx_b, P_cx_b, E_cx_b = moments(births)
        rec_counts = (
            np.zeros((self.nz, g.nvz, g.nvp))
            if not np.ndim(rec)
            else np.asarray(rec, dtype=float)[:, None, None] * M_i
        )
        N_rec, P_rec, E_rec = moments(rec_counts)

        P = P_ion + (P_cx_l - P_cx_b) - P_rec
        E = E_ion + (E_cx_l - E_cx_b) - E_rec
        S = N_ion - N_rec
        scale = 1.0 / (vol_c * dt)
        self.M_transfer = P * scale
        self.S_transfer = S * scale
        u = np.asarray(u_i, dtype=float)
        self.Ei_transfer = (
            E * scale - u * self.M_transfer + 0.5 * M_HE * u**2 * self.S_transfer
        )

        # --- the CX/elastic pair, booked on its own with the frequency that
        # carries it. Same decomposition, same moments; nothing above is
        # re-derived, so the pair plus the ionization/recombination rows is
        # the total to roundoff.
        M_pair = (P_cx_l - P_cx_b) * scale
        S_pair = (N_cx_l - N_cx_b) * scale
        self.M_transfer_pair = M_pair
        self.Ei_transfer_pair = (
            (E_cx_l - E_cx_b) * scale - u * M_pair + 0.5 * M_HE * u**2 * S_pair
        )
        n_arr = np.asarray(n_i, dtype=float)
        positive = n_arr > 0.0
        self.nu_pair = np.where(
            positive,
            N_cx_l / (vol_c * dt * np.where(positive, n_arr, 1.0)),
            0.0,
        )
        counted = N_cx_l > 0.0
        inv_N = np.where(counted, 1.0 / np.where(counted, N_cx_l, 1.0), 0.0)
        self.u_n_eff = P_cx_l * inv_N / M_HE
        self.T_eff_eV = (
            (2.0 / 3.0)
            * (E_cx_l - u * P_cx_l + 0.5 * M_HE * u**2 * N_cx_l)
            * inv_N
            / EV
        )


class BoundedChordFlights:
    """Frozen deterministic annulus flight maps (the K1b jump kernel).

    Built once from the geometry and the velocity grid; nothing here
    depends on the plasma, so the maps are run-constant.

    A flight of class ``X`` launched in cell ``i`` in bin ``(v_z, v_perp)``
    lasts the class time ``c_X(i) / v_perp`` and displaces the atom axially
    by ``dz = v_z c_X(i) / v_perp`` -- deterministic, at the cosine-weighted
    class-mean chord, with no free parameter. The atom is HELD at the
    midpoint of that flight and lands after a second half displacement, so
    the map this class stores is the single HALF displacement, applied once
    when the flight is launched (placement) and once when it completes
    (landing). Total displacement per flight is the full chord displacement
    and the residence centroid is the flight's own mean position.

    Along a half displacement the atom crosses faces, and three of those
    matter:

    - a **domain end plane**: the flight leaves the modelled system there.
      On the LANDING half it is booked as end outflow, which the engine's
      end-wall machinery then sticks or returns; on the PLACEMENT half it
      is only clipped, because placement moves no particle across any
      surface -- it decides which cell holds an atom that is already inside.
    - an **annulus area jump** (the throat faces the expanded ends and the
      plenum choke produce): the free-molecular throat convention the march
      already uses, ``min(A_left, A_right) / A_upstream``, is the fraction
      that passes; the rest strikes the annular step and is booked as a
      wall landing in the cell it was stopped in.
    - the **anode mesh face**: the transparency passes, the rest is booked
      on the mesh channel exactly as the march books it.
    - an **annular baffle face** (B6): a zero-thickness annular THROAT of
      area ``open_ann``, routed through the SAME free-molecular throat
      convention with ``A_throat = min(A_left, A_right, open_ann)`` -- the
      narrowest aperture in series wins -- and the stopped remainder booked on
      the baffle channel, which is the one channel pair the march's own
      interception books into as well. A baffle whose open ring is already at
      least the annulus throat changes no transmission at all and is
      therefore bit-exactly absent here, exactly as it is unarmed in the
      march.

    Every routed weight sums to one per ``(cell, bin)``: ``residual`` is the
    worst departure from that identity over the whole map, and is the
    statement that the operator moves particles without creating or
    destroying any.
    """

    def __init__(self, *, dz, V_ann, A_ann, Rp_cm, Rm_cm, grid, mesh_face,
                 transparency, baffles=()):
        g = grid
        nz = int(np.asarray(dz).size)
        nvz, nvp = g.nvz, g.nvp
        shape = (nz, nvz, nvp)
        self.nz, self.shape = nz, shape
        self.n_flat = nz * nvz * nvp
        dz = np.asarray(dz, dtype=float)
        V_ann = np.asarray(V_ann, dtype=float)
        A_ann = np.asarray(A_ann, dtype=float)
        ze = np.concatenate(([0.0], np.cumsum(dz)))
        zc = 0.5 * (ze[:-1] + ze[1:])

        (
            self.F_inner,
            c_ww,
            c_wi,
            c_io,
            self.var_ww,
            self.var_wi,
            self.var_io,
        ) = annulus_chord_classes(Rp_cm, Rm_cm)
        self.chords = {"ww": c_ww, "wi": c_wi, "io": c_io}

        # Interior-face transmissions, per direction. The throat form is the
        # march's own: what leaves a cell through a face is limited by the
        # narrower of the two cells it joins.
        tau_f = np.ones(nz + 1)
        tau_b = np.ones(nz + 1)
        is_baffle = np.zeros(nz + 1, dtype=bool)
        with np.errstate(divide="ignore", invalid="ignore"):
            throat = np.minimum(A_ann[:-1], A_ann[1:])
            # A baffle is one more aperture in the same series: the throat at
            # its face becomes the narrowest of the two cells and its own open
            # ring. An open ring at least as wide as the geometric throat
            # leaves ``throat`` untouched, which is why an unrestricting
            # baffle is bit-exactly absent from this map.
            for face, open_ann in baffles:
                is_baffle[int(face)] = True
                throat[int(face) - 1] = min(
                    float(throat[int(face) - 1]), float(open_ann)
                )
            tau_f[1:-1] = np.where(A_ann[:-1] > 0.0, throat / A_ann[:-1], 0.0)
            tau_b[1:-1] = np.where(A_ann[1:] > 0.0, throat / A_ann[1:], 0.0)
        mesh_face = int(mesh_face)
        is_mesh = np.zeros(nz + 1, dtype=bool)
        if 0 < mesh_face < nz:
            tau_f[mesh_face] *= float(transparency)
            tau_b[mesh_face] *= float(transparency)
            is_mesh[mesh_face] = True
        special = [
            f for f in range(1, nz)
            if tau_f[f] < 1.0 - 1e-15 or tau_b[f] < 1.0 - 1e-15
        ]

        no_ann = V_ann <= 0.0
        cell = np.arange(nz)[:, None, None]
        vz_pos = (g.vz > 0.0)[:, None]
        self.vz_pos = vz_pos

        self.nu = {}
        self.hold_flat = {}
        self.dest_flat = {}
        self.w_pass = {}
        self.w_end = {}
        self.stop_src = {}
        self.stop_dst = {}
        self.stop_w = {}
        self.mesh_src = {}
        self.mesh_dst = {}
        self.mesh_w = {}
        self.baffle_src = {}
        self.baffle_dst = {}
        self.baffle_w = {}
        self.residual = 0.0

        bin_flat = (np.arange(nvz)[:, None] * nvp + np.arange(nvp)[None, :])
        for name in FLIGHT_CLASSES:
            c = self.chords[name]
            nu = np.where(
                no_ann[:, None], 0.0, g.vp[None, :] / np.maximum(c, 1e-30)[:, None]
            )
            self.nu[name] = nu[:, None, :]
            half = 0.5 * (g.VZ * c[:, None, None]) / np.maximum(g.VP, 1e-30)
            z1 = zc[:, None, None] + half
            exits_L = z1 < ze[0]
            exits_R = z1 > ze[-1]
            j = np.clip(
                np.searchsorted(ze, np.clip(z1, ze[0], ze[-1])) - 1, 0, nz - 1
            )
            fwd = half > 0.0
            bwd = half < 0.0
            surv = np.ones(shape)
            hold = j.copy()
            stops = {"step": [], "mesh": [], "baffle": []}
            for faces, forward in ((special, True), (special[::-1], False)):
                for f in faces:
                    tau = tau_f[f] if forward else tau_b[f]
                    if tau >= 1.0 - 1e-15:
                        continue
                    if forward:
                        crossed = fwd & (cell < f) & (j >= f)
                        landing = f - 1
                    else:
                        crossed = bwd & (cell >= f) & (j < f)
                        landing = f
                    if not crossed.any():
                        continue
                    blocked = np.where(crossed, surv * (1.0 - tau), 0.0)
                    idx = np.flatnonzero(blocked.ravel() > 0.0)
                    if idx.size:
                        if is_mesh[f]:
                            kind = "mesh"
                        elif is_baffle[f]:
                            kind = "baffle"
                        else:
                            kind = "step"
                        stops[kind].append(
                            (idx, landing, blocked.ravel()[idx])
                        )
                    surv = np.where(crossed, surv * tau, surv)
                    if tau <= 0.0:
                        hold = np.where(
                            crossed & ((hold >= f) if forward else (hold < f)),
                            landing,
                            hold,
                        )
            hold = np.where(no_ann[:, None, None], cell, hold)
            hold = np.where(no_ann[hold], cell, hold)
            exits = exits_L | exits_R
            self.w_end[name] = np.where(exits, surv, 0.0)
            self.w_pass[name] = np.where(exits, 0.0, surv)
            self.dest_flat[name] = (j * nvz * nvp + bin_flat[None, :, :]).ravel()
            self.hold_flat[name] = (
                hold * nvz * nvp + bin_flat[None, :, :]
            ).ravel()
            for kind, src, dst, wts in (
                ("step", self.stop_src, self.stop_dst, self.stop_w),
                ("mesh", self.mesh_src, self.mesh_dst, self.mesh_w),
                ("baffle", self.baffle_src, self.baffle_dst, self.baffle_w),
            ):
                if stops[kind]:
                    src[name] = np.concatenate([s[0] for s in stops[kind]])
                    dst[name] = np.concatenate([
                        s[0] % (nvz * nvp) + s[1] * nvz * nvp
                        for s in stops[kind]
                    ])
                    wts[name] = np.concatenate([s[2] for s in stops[kind]])
                else:
                    src[name] = np.zeros(0, dtype=np.int64)
                    dst[name] = np.zeros(0, dtype=np.int64)
                    wts[name] = np.zeros(0)
            total = self.w_pass[name] + self.w_end[name]
            for src, wts in ((self.stop_src, self.stop_w),
                             (self.mesh_src, self.mesh_w),
                             (self.baffle_src, self.baffle_w)):
                np.add.at(total.reshape(-1), src[name], wts[name])
            live = ~np.broadcast_to(no_ann[:, None, None], shape)
            self.residual = max(
                self.residual, float(np.max(np.abs(total[live] - 1.0)))
            )
        if not self.residual <= 1.0e-12:
            raise ValueError(
                "the bounded-chord annulus flight map must route every "
                "launched particle exactly once (pass + stopped + through an "
                f"end plane == 1); worst departure {self.residual:.3e}"
            )

    def route(self, name, counts):
        """Route completed flights of one class.

        ``counts`` are PARTICLES per ``(cell, bin)`` completing this tick.
        Returns ``(arrive, stopped, meshed, baffled, end_L, end_R)``: the
        particles reaching the class's landing surface per destination cell
        and bin, those stopped at an annular step, those the anode mesh
        intercepted, those an annular BAFFLE intercepted, and the two
        end-plane outflows per bin.
        """
        flat = counts.reshape(-1)
        n = self.n_flat
        arrive = np.bincount(
            self.dest_flat[name],
            weights=(counts * self.w_pass[name]).reshape(-1),
            minlength=n,
        ).reshape(self.shape)
        stopped = np.bincount(
            self.stop_dst[name],
            weights=flat[self.stop_src[name]] * self.stop_w[name],
            minlength=n,
        ).reshape(self.shape)
        meshed = np.bincount(
            self.mesh_dst[name],
            weights=flat[self.mesh_src[name]] * self.mesh_w[name],
            minlength=n,
        ).reshape(self.shape)
        baffled = np.bincount(
            self.baffle_dst[name],
            weights=flat[self.baffle_src[name]] * self.baffle_w[name],
            minlength=n,
        ).reshape(self.shape)
        gone = (counts * self.w_end[name]).sum(axis=0)
        end_R = np.where(self.vz_pos, gone, 0.0)
        end_L = np.where(self.vz_pos, 0.0, gone)
        return arrive, stopped, meshed, baffled, end_L, end_R

    def place(self, name, launched):
        """Return launched PARTICLES per holding cell and bin for a class."""
        return np.bincount(
            self.hold_flat[name],
            weights=launched.reshape(-1),
            minlength=self.n_flat,
        ).reshape(self.shape)


#: The keys a ``cathode_jet`` spec carries, exactly.
CATHODE_JET_SPEC_KEYS = ("R_N", "R_E", "T_launch_eV")


def _validated_cathode_jet(spec):
    """Return a validated cathode-jet spec, or ``None`` when absent.

    ``R_E`` is the TOTAL reflected energy fraction and the ``R_N``
    backscattered particles carry all of it, so each leaves with
    ``R_E/R_N`` of the incident energy: that reading is only meaningful for
    ``0 < R_E <= R_N < 1``, the same interval
    :func:`~cablp.solvers._sim1d.physics.sources.cathode_jet_backscatter_speed`
    demands of the fluid channel under ``energy_convention =
    "total_reflected"``. A reflected atom cannot leave with more energy than
    it arrived with, and neither coefficient may be degenerate.

    ``T_launch_eV`` is ``None`` (grid-tied) or a positive finite float.
    """
    if spec is None:
        return None
    unknown = sorted(set(spec) - set(CATHODE_JET_SPEC_KEYS))
    missing = sorted(set(CATHODE_JET_SPEC_KEYS) - set(spec))
    if unknown or missing:
        raise ValueError(
            "the DVM cathode_jet spec carries exactly "
            f"{list(CATHODE_JET_SPEC_KEYS)}; got unknown {unknown} and "
            f"missing {missing}"
        )
    R_N = float(spec["R_N"])
    R_E = float(spec["R_E"])
    if not (0.0 < R_E <= R_N < 1.0):
        raise ValueError(
            "the DVM cathode jet reads R_E as the TOTAL reflected energy "
            "fraction and gives each of the R_N backscattered particles "
            "R_E/R_N of the incident energy, so it requires "
            f"0 < R_E <= R_N < 1 (got R_E={R_E}, R_N={R_N})"
        )
    T_launch = spec["T_launch_eV"]
    if T_launch is not None:
        T_launch = float(T_launch)
        if not np.isfinite(T_launch) or T_launch <= 0.0:
            raise ValueError(
                "the DVM cathode jet's T_launch_eV is the width of the smear "
                "its monoenergetic beam is represented by and must be a "
                "positive finite temperature, or None to tie it to the local "
                f"velocity-grid bin (got {T_launch!r})"
            )
    return {"R_N": R_N, "R_E": R_E, "T_launch_eV": T_launch}


#: The keys an ``anode_jet`` spec carries, exactly.
ANODE_JET_SPEC_KEYS = ("R_N", "R_E", "T_launch_eV")


def _validated_anode_jet(spec):
    """Return a validated anode-jet spec, or ``None`` when absent.

    The anode twin of :func:`_validated_cathode_jet`, on the same reading and
    the same interval: ``R_E`` is the TOTAL reflected energy fraction and the
    ``R_N`` backscattered particles carry all of it, so each leaves with
    ``R_E/R_N`` of the incident energy, which is only meaningful for
    ``0 < R_E <= R_N < 1`` -- the interval
    :func:`~cablp.solvers._sim1d.physics.sources.anode_jet_backscatter_speed`
    demands of the fluid channel under ``energy_convention =
    "total_reflected"``.

    ``T_launch_eV`` is ``None`` (grid-tied) or a positive finite float.
    """
    if spec is None:
        return None
    unknown = sorted(set(spec) - set(ANODE_JET_SPEC_KEYS))
    missing = sorted(set(ANODE_JET_SPEC_KEYS) - set(spec))
    if unknown or missing:
        raise ValueError(
            "the DVM anode_jet spec carries exactly "
            f"{list(ANODE_JET_SPEC_KEYS)}; got unknown {unknown} and "
            f"missing {missing}"
        )
    R_N = float(spec["R_N"])
    R_E = float(spec["R_E"])
    if not (0.0 < R_E <= R_N < 1.0):
        raise ValueError(
            "the DVM anode jet reads R_E as the TOTAL reflected energy "
            "fraction and gives each of the R_N backscattered particles "
            "R_E/R_N of the incident energy, so it requires "
            f"0 < R_E <= R_N < 1 (got R_E={R_E}, R_N={R_N})"
        )
    T_launch = spec["T_launch_eV"]
    if T_launch is not None:
        T_launch = float(T_launch)
        if not np.isfinite(T_launch) or T_launch <= 0.0:
            raise ValueError(
                "the DVM anode jet's T_launch_eV is the width of the smear "
                "its monoenergetic beam is represented by and must be a "
                "positive finite temperature, or None to tie it to the local "
                f"velocity-grid bin (got {T_launch!r})"
            )
    return {"R_N": R_N, "R_E": R_E, "T_launch_eV": T_launch}


def _throat_areas(cell_areas):
    """Return the ``nz+1`` face areas of a per-cell area profile [cm^2].

    Interior faces take the throat ``min`` of the two cells they join, so
    a narrowing (or a vanishing annulus) throttles the flux from both
    sides identically. The two domain-end faces take their own cell's
    area: both ends are open.
    """
    a = np.asarray(cell_areas, dtype=float)
    faces = np.empty(a.size + 1)
    faces[0] = a[0]
    faces[-1] = a[-1]
    faces[1:-1] = np.minimum(a[:-1], a[1:])
    return faces


def ledger_residual(ledger):
    """Return the ledger's particle-closure residuals.

    ``distribution``: ``Delta(sum f V)`` minus (all births - all losses),
    which is the statement that substep B creates exactly what substep A
    destroyed. ``domain``: ``Delta(inventory incl. pending)`` minus
    (external births - ionization - pumping), the physical closure with
    every internal channel cancelled.

    Both are absolute particle counts. The relative forms divide by the
    throughput PLUS the standing inventory, because each identity is a
    difference of two inventories: on a short neutral tick the throughput
    can be many orders below the inventory, and the floating-point noise
    floor of the statement is then set by the inventory, not by the
    handful of particles that moved. Normalizing by throughput alone would
    report cancellation noise as a conservation error.
    """
    births = sum(
        v for k, v in ledger.items() if k.startswith("birth_")
    )
    losses = sum(
        v
        for k, v in ledger.items()
        if k.startswith("loss_") and not k.startswith("loss_pump_")
    )
    distribution = (
        ledger["f_inventory_after"] - ledger["f_inventory_before"]
    ) - (births - losses)
    external = sum(
        ledger[f"birth_{name}"] for name in LEDGER_EXTERNAL_BIRTHS
    )
    domain = (
        ledger["inventory_after"] - ledger["inventory_before"]
    ) - (
        external
        - ledger["loss_ionization"]
        - ledger["loss_pump_L"]
        - ledger["loss_pump_R"]
    )
    throughput = births + losses + 1e-300
    scale = throughput + abs(ledger["inventory_before"])
    return {
        "distribution": distribution,
        "domain": domain,
        "throughput": throughput,
        "scale": scale,
        "distribution_rel": distribution / scale,
        "domain_rel": domain / scale,
    }


def ledger_energy_residual(ledger):
    """Return the ENERGY ledger's closure residuals [erg].

    The same two statements :func:`ledger_residual` makes about particles,
    made about the energy sub-ledger and at the same exactness class.

    ``distribution``: ``Delta(sum f V E)`` minus (all energy births - all
    energy losses), which is the statement that every energy the update
    added to or removed from the distributions was booked to a channel.

    ``domain``: ``Delta(energy incl. pending)`` minus (external energy
    births - ionization - pumping - the net surface and exchange
    transfers). Unlike the particle form nothing cancels here on its own --
    an internal particle channel returns what it took and an internal
    ENERGY channel does not -- so the ``net_*`` rows are what the internal
    channels contribute, and the identity is the physical energy balance of
    the modelled gas.

    Both are absolute energies in erg. The relative forms divide by the
    throughput PLUS the standing energy inventory, for the reason the
    particle residuals do: each identity is a difference of two
    inventories, so on a short tick the noise floor is set by the standing
    inventory rather than by the energy that moved.
    """
    e = ledger["energy"]
    births = sum(v for k, v in e.items() if k.startswith("birth_"))
    losses = sum(
        v
        for k, v in e.items()
        if k.startswith("loss_") and not k.startswith("loss_pump_")
    )
    distribution = (
        e["f_inventory_after"] - e["f_inventory_before"]
    ) - (births - losses)
    external = sum(e[f"birth_{name}"] for name in LEDGER_EXTERNAL_BIRTHS)
    net = sum(e[f"net_{name}"] for name in LEDGER_ENERGY_NET_CHANNELS)
    domain = (
        e["inventory_after"] - e["inventory_before"]
    ) - (
        external
        - e["loss_ionization"]
        - e["loss_pump_L"]
        - e["loss_pump_R"]
        - net
    )
    throughput = births + losses + 1e-300
    scale = throughput + abs(e["inventory_before"])
    return {
        "distribution": distribution,
        "domain": domain,
        "throughput": throughput,
        "scale": scale,
        "distribution_rel": distribution / scale,
        "domain_rel": domain / scale,
    }


def _ghost_density(pending, area_cm2, dt, g):
    """Convert buffered particles per bin into a boundary ghost density.

    The march injects ``|v_z| F A dt`` particles per bin, so dividing the
    buffered count by exactly that factor injects the buffered count --
    independent of ``dt``, which is what lets the neutral clock change
    cadence without leaking particles.
    """
    if area_cm2 <= 0.0 or not np.any(pending):
        return np.zeros_like(pending)
    with np.errstate(divide="ignore", invalid="ignore"):
        dens = np.where(
            np.abs(g.VZ) > 0.0,
            pending / (np.abs(g.VZ) * area_cm2 * dt),
            0.0,
        )
    return dens


def _end_return(outgoing, sticking, accommodation, mirror, spectrum):
    """Split an end-wall outflow into the buffered return, per bin.

    The pumped fraction ``sticking`` leaves. Of the rest, the
    accommodated fraction is re-emitted cosine-distributed at the surface
    temperature (``spectrum``, already an inward half-flux distribution),
    and the remainder is reflected at the incident energy, which on the
    symmetric ``v_z`` axis is the exact bin mirror.
    """
    back = (1.0 - float(sticking)) * outgoing
    total = float(back.sum())
    reflected = (1.0 - float(accommodation)) * back[mirror, :]
    accommodated = float(accommodation) * total * spectrum
    return reflected + accommodated


# Where :func:`_cosine_quadrature` parks its one entry on a velocity grid.
_COSINE_QUADRATURE_ATTR = "_cosine_wall_quadrature"

# Quadrature nodes per perpendicular bin, matching
# :meth:`VGrid.wall_emission_spectrum`'s own subsampling.
_COSINE_QUADRATURE_NODES = 64


def _cosine_quadrature(g):
    """Return the GRID-ONLY factors of the cosine-wall quadrature.

    ``(x, x_sq, dx)``: the 64 abscissae subsampling each perpendicular bin,
    their squares, and the node spacings the trapezoid weights with. All three
    are pure grid geometry -- none depends on the thermal speed -- while
    :func:`_cosine_wall_spectra` is evaluated tens of times per wall-return
    solve, so they are built once and reused.

    BOUND AND EVICTION: exactly ONE entry, stored as an attribute on the
    ``VGrid`` instance itself. There is no keyed table and nothing to grow: a
    second grid gets its own entry on itself, and an entry is released with the
    grid that owns it. The axes a ``VGrid`` is built with never change after
    construction, so the entry cannot go stale while its grid is alive.
    """
    cached = getattr(g, _COSINE_QUADRATURE_ATTR, None)
    if cached is None:
        x = np.stack([
            np.linspace(g.vp_edges[k], g.vp_edges[k + 1], _COSINE_QUADRATURE_NODES)
            for k in range(g.nvp)
        ])
        cached = (x, x[None, :, :] ** 2, np.diff(x, axis=-1))
        setattr(g, _COSINE_QUADRATURE_ATTR, cached)
    return cached


def _cosine_wall_factors(g, s):
    """Return the SEPARABLE marginals of the cosine-wall spectrum.

    ``(wz, wp)``: the unnormalized error-function ``v_z`` bin masses, shape
    ``(s.size, nvz)``, and the 64-node trapezoid of
    ``vp^2 exp(-vp^2 / 2 s^2)`` per perpendicular bin, shape
    ``(s.size, nvp)``. The spectrum is their outer product, normalized -- see
    :func:`_cosine_wall_spectra` -- so any moment of a separable weight can be
    contracted from these two factors without ever forming the
    ``(s.size, nvz, nvp)`` array.

    Raises ``ValueError`` when any ``s`` is not positive and finite.
    """
    s = np.atleast_1d(np.asarray(s, dtype=float))
    if not np.all(np.isfinite(s)) or np.any(s <= 0.0):
        raise ValueError(
            "the cosine-wall re-emission spectrum needs a positive finite "
            f"thermal speed per cell (got min {np.min(s)!r}, "
            f"max {np.max(s)!r})"
        )
    sc = s[:, None]
    ez = g.vz_edges[None, :] / (sc * np.sqrt(2.0))
    wz = 0.5 * np.diff(_erf(ez), axis=-1)
    x, x_sq, dx = _cosine_quadrature(g)
    # ``y = x_sq * exp(-0.5 * (x / s)**2)``, accumulated through ONE buffer.
    # Each step is the same operation on the same operands as the expression it
    # replaces -- ``np.square`` is what ``** 2`` dispatches to, and the two
    # scalar multiplies are commutative -- so only the number of temporaries
    # changes, never a value.
    y = x[None, :, :] / s[:, None, None]
    np.square(y, out=y)
    np.multiply(y, -0.5, out=y)
    np.exp(y, out=y)
    np.multiply(x_sq, y, out=y)
    # ``np.trapezoid(y, x=broadcast_to(x, y.shape), axis=-1)``, written out so
    # the node spacings come from the cache instead of being re-differenced off
    # a broadcast copy of ``x`` on every call. Same expression, same operand
    # values, same reduction axis and shape: numpy's own body is
    # ``(d * (y[..., 1:] + y[..., :-1]) / 2.0).sum(axis)`` with
    # ``d = diff(x, axis=-1)``, and ``dx`` broadcasts to exactly that ``d``.
    # The reduction runs on a fresh C-contiguous buffer, as it did before.
    quad = y[..., 1:] + y[..., :-1]
    np.multiply(dx, quad, out=quad)
    np.divide(quad, 2.0, out=quad)
    return wz, quad.sum(-1)


def _cosine_wall_mean_energy(g, s):
    """Return the DISCRETE mean energy per atom [erg] of the cosine spectrum.

    ``0.5 m <v_z^2 + v_perp^2>`` over :func:`_cosine_wall_spectra` at ``s``,
    contracted from the two marginals of :func:`_cosine_wall_factors` instead
    of from the assembled spectrum: the bin energy is separable
    (``E_bin = 0.5 m (v_z^2 + v_perp^2)``) and the spectrum is an outer
    product, so the mean is
    ``0.5 m [ (sum wz vz^2)(sum wp) + (sum wz)(sum wp vp^2) ] /
    [ (sum wz)(sum wp) ]`` -- ``O(nvz + nvp)`` per cell rather than
    ``O(nvz nvp)``, and no ``(cells, nvz, nvp)`` temporary at all.

    It is the residual of the wall-return energy solve, NOT the quantity the
    solve is finally checked against: that check contracts the assembled
    spectrum against ``E_bin`` and is what
    :data:`WALL_ENERGY_SOLVE_REL_TOL` bounds. The two agree to roundoff of
    the reassociated sum, which is why this form may seed the other.

    Returns non-finite values rather than raising when the marginals carry no
    normalizable mass; the caller treats that as non-convergence.
    """
    wz, wp = _cosine_wall_factors(g, s)
    sum_z = wz.sum(axis=-1)
    sum_p = wp.sum(axis=-1)
    e_z = (wz * (g.vz**2)[None, :]).sum(axis=-1)
    e_p = (wp * (g.vp**2)[None, :]).sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 0.5 * M_HE * (e_z * sum_p + sum_z * e_p) / (sum_z * sum_p)


def _cosine_wall_spectra(g, s):
    """Return cosine-wall re-emission spectra for an array of thermal speeds.

    Array transcription of :meth:`VGrid.wall_emission_spectrum` over
    ``s = sqrt(k T / m)`` [cm/s]: the same error-function ``v_z`` bin masses,
    the same 64-node trapezoid of ``vp^2 exp(-vp^2 / 2 s^2)`` on each
    perpendicular bin, and the same normalization to unit sum. Returns one
    spectrum per entry of ``s``, shape ``(s.size, nvz, nvp)``.

    It exists because the ``"diffuse_elastic"`` wall return solves a
    temperature PER CELL and evaluates the spectrum tens of times per solve;
    the scalar helper's per-bin Python quadrature makes that unaffordable on
    a device mesh. It is the same expression, not a second model of the wall.

    Raises ``ValueError`` when any ``s`` is not positive and finite, or when
    a constructed spectrum has no mass to normalize -- a spectrum that
    silently normalized to nothing would re-emit the wall's whole return at
    the wrong energy.
    """
    wz, wp = _cosine_wall_factors(g, s)
    f = wz[:, :, None] * wp[:, None, :]
    total = f.sum(axis=(1, 2))
    if not np.all(np.isfinite(total)) or np.any(total <= 0.0):
        raise ValueError(
            "the cosine-wall re-emission spectrum construction produced no "
            f"normalizable mass (worst bin total {np.min(total)!r}); the "
            "requested temperature lies outside what this velocity grid "
            "resolves"
        )
    # ``f`` is this call's own array and is not handed out anywhere else, so the
    # normalization is applied in place.
    np.divide(f, total[:, None, None], out=f)
    return f


def _drift(f, g):
    n = f.sum(axis=(1, 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(
            n > 0.0,
            (f * g.VZ[None, :, :]).sum(axis=(1, 2)) / np.maximum(n, 1e-300),
            0.0,
        )


def _temperature_eV(f, g):
    n = f.sum(axis=(1, 2))
    u = _drift(f, g)
    c2 = (g.VZ[None, :, :] - u[:, None, None]) ** 2 + (g.VP**2)[None, :, :]
    mean_c2 = np.where(
        n > 0.0, (f * c2).sum(axis=(1, 2)) / np.maximum(n, 1e-300), 0.0
    )
    return M_HE * mean_c2 / (3.0 * EV)
