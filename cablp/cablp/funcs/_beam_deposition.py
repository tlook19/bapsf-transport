"""Solver-agnostic beam deposition along a ray (B1).

Deterministic CSDA (continuous-slowing-down) integration of a monoenergetic
primary-electron beam through the column. **Pure function of the beam and the
column** — ``(E0, Gamma0, nn, ne, Te, ray)`` — with no solver state, so both
the voltage-driven and current-driven cathode formulations can consume it at
the same call sites.

Energy channels, per unit path length [eV/cm]:

- ionization: ``nn * sigma_ion(E) * (I_ion + <W_sec>(E))`` — the potential
  I_ion is banked as ionization cost (matching the solver's separate
  ``beam_ionization_cost`` term); the mean secondary energy <W_sec> is banked
  as plasma heating (the ejected electron thermalizes locally). <W_sec> uses
  the Opal-Peterson-Beaty differential shape ``dsigma/dW ~ 1/(W^2 + Ebar^2)``
  with the He shape parameter Ebar = 15.8 eV (Opal, Peterson & Beaty, At.
  Data 4, 209 (1971)) over W in [0, (E - I_ion)/2].
- excitation: ``nn * sigma_exc(E) * E_rad(E)`` — the summed Ralchenko singlet
  manifold (``_cross.He_beam_excitation_channel``); banked as radiation.
- Coulomb drag on the plasma, per ``coulomb_model``; banked as plasma heating.
- optional anomalous (beam-plasma instability) drag, per ``anomalous_model``;
  banked as plasma heating (Langmuir turbulence Landau-damps on the bulk
  near where it is driven).

Coulomb closures (both parameter-free):

- ``"fast_electron"`` (default): the physical CSDA stopping power on plasma
  electrons, ``dE/dx = 2 pi e^4 n_e lnLambda / E`` — the collision rate of a
  primary falls as 1/v^3, so at main-discharge conditions (150 eV, n_e =
  5e12) the energy e-fold is ~35 m.
- ``"legacy_tau_ei"``: ``dE/dx = E / (v(E) tau_ei(Te, ne))`` with the
  *thermal* NRL collision time — the historical `_cathode_solver._compute_l_b`
  form (~1.1 m at the same conditions). Provided for continuity experiments;
  its "Coulomb" label overestimates the classical drag ~30x.

Anomalous closure:

- ``"quasilinear"``: the beam-plasma (bump-on-tail) instability drives
  Langmuir waves at v_phase ~ v_b with cold-beam growth rate
  ``~(n_b/n_e)^(1/3) omega_pe`` (growth lengths ~mm-cm here, i.e. saturated
  essentially at the source); quasilinear diffusion then flattens the beam
  over the relaxation length

      l_QL = (n_e / n_b) * (v_b / omega_pe) * ln(n_e / n_b)

  (Vedenov-style estimate; ~0.05-0.1 m at production parameters). Modeled as
  mean-energy drag ``dE/dx = E / l_QL`` with the flux preserved (the module
  carries one mean energy, not a distribution — the plateau spread is a
  documented limitation). **Stated validity domain:** weak beam
  (``n_b << n_e``) with growth beating wave damping — the main-discharge
  column. The breakdown phase (n_b ~ n_e) is outside quasilinear theory and
  deliberately stays collisional. Requires ``beam_area_cm2`` to form
  ``n_b = Gamma0 / (A v_b)``. Order-of-magnitude closure: density-gradient
  detuning and saturation physics can lengthen it substantially; results
  using it must be presented per closure, like the drag story.

The primary is followed until it exits the domain (transmitted) or its
energy crosses ``E_stop`` (default: the lowest inelastic threshold, He 2^1S
at 20.6158 eV), where the sub-threshold residual is banked as plasma heating
in the crossing cell. Per-ray energy conservation
``Gamma0*E0 = heating + radiated + ionization cost + transmitted`` holds to
accumulated roundoff by construction (the energy decrement and the channel
banks are the same floating-point sums).

``plasma_heating_erg_s`` lumps four physically distinct deliveries, so the
result also carries them separately as DIAGNOSTIC arrays (``heating_coulomb``
/ ``heating_anomalous`` / ``heating_secondary`` / ``heating_terminal``). These
are bookkeeping only -- they re-add products the energy decrement already
formed, and the lumped bank the RHS consumes is untouched.

Non-local product transport (``product_transport``, WP-D)
---------------------------------------------------------

The two EVENT-PRODUCT channels above -- the mean secondary energy <W_sec> per
ionization, and the primary's terminal sub-threshold residual -- are banked in
their birth cell under ``product_transport="local"`` (the default, and the
historical behaviour). That is perfect local confinement, and at breakdown it
is the wrong limit: both products are BELOW every He inelastic threshold, so
their only loss channel is Coulomb slowing on the bulk, and at n_e ~ 1e10 that
costs ~1 eV per machine pass. They are near-collisionless along B exactly when
the model assumes they stop where they were born.

``product_transport="nonlocal"`` gives each product its own mini-CSDA walk
along B from its birth cell, depositing the same
``coulomb_stopping_eV_per_cm`` drag the primary feels (same ``coulomb_model``)
until it either thermalizes -- energy down to the local Maxwellian mean
``1.5*Te`` (floored at the module's own lnLambda temperature clamp, 0.1 eV),
below which the fast test-particle stopping formula has no meaning and the
electron is indistinguishable from the bulk -- or leaves an end of the domain,
where its remaining energy is booked to the END LEDGER
(``end_loss_low_erg_s`` / ``end_loss_high_erg_s``) and LEAVES the system.

Products and their directions:

- terminal residual: one walk from the stop cell in the PRIMARY's direction;
- secondaries: two half-weight walks (+z and -z) from each birth cell, the
  OPB emission being broadly isotropic -- a stated approximation.

Stated limitations of the walk: straight-line along B with no pitch-angle
diffusion; Coulomb slowing on the plasma is the ONLY loss channel (elastic
e-He transfers ~5 meV per collision, negligible); each birth cell's
secondaries walk at their flux-weighted MEAN energy, matching the module's
mean-energy treatment of the primary; a product is born at the near edge of
its birth cell along its own walk direction, so the walk traverses the whole
birth cell (cell-resolution granularity, as everywhere else in the module).

The end ledger ALSO books the transmitted primary's power
``Gamma_t * E_t`` -- computed by this module since B1 and never banked
anywhere. Under ``"nonlocal"`` the per-ray identity is therefore

    Gamma0*E0 = heating + radiated + ionization_cost + anode_intercepted
                + end_loss_low + end_loss_high

with the transmitted primary INSIDE the end-loss channel (its size is
reported separately as ``end_loss_transmitted_erg_s``, so the historical
form with ``transmitted`` as its own term is recoverable by subtraction).
``transmitted_flux`` / ``transmitted_energy_eV`` keep their meaning and their
values. Under ``"local"`` the end-loss fields are identically zero, nothing
new is booked, and the identity stays exactly the historical

    Gamma0*E0 = heating + radiated + ionization_cost + anode_intercepted
                + transmitted

**v1 is ENERGY-ONLY routing.** Particle and charge bookkeeping is UNCHANGED:
``ionization_events`` / ``excitation_events``, the fluid ``n`` rows they feed,
and the circuit currents are identical in both modes. Only where the product
ENERGY lands moves.

QL heating locality (``anomalous_transport``, WP-E)
---------------------------------------------------

The anomalous channel above banks its drag as INSTANTANEOUS LOCAL bulk
heating: the Langmuir turbulence is Landau-damped near where it is driven, so
its energy is handed to the background electrons in the cell that drove it.
That is the right picture for the wave energy, but not for the electrons the
wave heats. Quasilinear diffusion does not warm a Maxwellian in place — it
fills a fast-tail PLATEAU first, and at breakdown densities a tail electron is
collisionally decoupled (Coulomb range ~km at n_e ~ 1e10, hundreds of machine
lengths) and free-streams along B. The bulk only sees the power once those
tail electrons slow down, wherever that happens to be.

``anomalous_transport="tail_walk"`` carries that step explicitly. The
per-cell anomalous power ``P_QL(z)`` is WITHHELD from its birth cell and
re-expressed as a population of tail electrons at a single energy
``tail_energy_eV`` (``E_tail``), i.e. an equivalent tail flux
``P_QL(z) / E_tail`` launched from that cell, split 50/50 along +B and -B
(the QL plateau is driven along B and the bump-on-tail resonance is
one-sided, but the plateau electrons themselves scatter both ways — a stated
approximation, matching the secondaries' treatment above). Each population is
then walked with the SAME closed-form Coulomb machinery the WP-D products use
-- the module's own ``coulomb_stopping_eV_per_cm`` under the ray's own
``coulomb_model``, the same ``1.5*Te`` thermalization floor -- and deposits
into ``plasma_heating_erg_s`` where it actually slows. **No new physics
parameters beyond E_tail:** the walk introduces nothing the primary's drag law
does not already contain.

``E_tail`` cannot be pinned by a fluid model (the plateau energy is a kinetic
quantity), so per campaign policy it is an ASSUMPTION value and the BRACKET is
the claim: the registered central arm is 75 eV, with 30 and 150 eV as the
bracket arms, all three reported together and none of them ever fitted.

Energy that reaches a domain end still hot is booked to a SEPARATE end-surface
ledger, ``end_loss_tail_low_erg_s`` / ``end_loss_tail_high_erg_s``, and leaves
the system. It is deliberately NOT mixed into the WP-D ``end_loss_*`` fields:
those are documented, smoke-pinned, and already MEASURED (WP-D D2/D4) as the
escape of the two EVENT products, they are identically zero under
``product_transport="local"``, and the two closures switch independently — so
sharing one pair of fields would both break that invariant and silently
contaminate the WP-D diverted-fraction readout whenever both are on. The
per-ray identity under ``"tail_walk"`` gains two terms and nothing else::

    Gamma0*E0 = heating + radiated + ionization_cost + anode_intercepted
                + transmitted (or the WP-D end ledger)
                + end_loss_tail_low + end_loss_tail_high

The CONSERVATION IDENTITY the smoke suite pins is the sharper local statement:
the ray integration is bit-identical in both modes (``L_anom`` depends only on
the beam and the column, never on where its energy is banked), so the
anomalous power the ``"local"`` arm banks equals, to roundoff, what the
``"tail_walk"`` arm deposits along the walks plus what it books to the tail end
ledger.

``heating_anomalous_erg_s`` keeps its meaning as "the anomalous channel's
delivery to the electrons", so under ``"tail_walk"`` it reports the WALKED
deposition profile rather than the birth profile — the same convention WP-D
uses for ``heating_secondary`` / ``heating_terminal``.

Like WP-D this is ENERGY-ONLY: the tail electrons are an energy-transport
bookkeeping device, not a new particle species. ``ionization_events``,
``excitation_events``, every fluid particle row and every circuit current are
identical in both modes. Stated limitations, inherited from the walk: straight
lines along B, no pitch-angle diffusion, one mean energy rather than a plateau
distribution, and no sheath/ambipolar throttle at the ends — so ``"tail_walk"``
is a FREE-ESCAPE bound and the pair {local, tail_walk} brackets the truth.

Tail ionization (``tail_ionization``, K6)
-----------------------------------------

``tail_ionization="on"`` removes the ENERGY-ONLY restriction from the WP-E
tail walk: a tail electron at ``E_tail`` is above every He inelastic threshold
at all shipped rungs, so declaring that it may only Coulomb-slow is a modeling
choice, not a physical one. Under ``"on"`` each tail population is marched
with THIS MODULE'S OWN CSDA integration -- a recursive ``deposit_beam`` call
per birth cell and direction, at ``anomalous_model="none"`` (the plateau
electrons are the instability's product; they do not re-drive it) and
``product_transport="local"`` -- so it attenuates on the local COLUMN neutral
density with the same He ionization and excitation cross sections the primary
uses, at the walker's CURRENT energy, simultaneously with its Coulomb slowing.
Under ``neutral_two_zone`` the ``nn`` the caller passes IS the column channel,
and that is the density the walker attenuates on; the attic/annulus channel is
not on the walker's field line.

The walker's channels are booked into the SAME per-cell banks the primary
fills, so every downstream consumer -- the fluid particle rows, the ionization
cost sink, the radiation sink, the circuit -- sees tail-borne events through
the machinery it already has, with the ion/electron pair booked by exactly the
convention the primary's births use:

- each ionization event adds one pair to ``ionization_events`` (and its
  ``I_ion`` investment to ``ionization_cost_erg_s``);
- the mean secondary ``<W_sec>`` banks as local electron heat with the birth;
- each excitation event's threshold energy goes to ``radiated_erg_s``;
- the walker's Coulomb drag and its sub-threshold terminal residual bank as
  plasma heating where they occur;
- a walker still above ``E_stop`` at either face of its WALK WINDOW goes to the
  SAME tail end ledger the energy-only walk uses (``end_loss_tail_low/high``)
  and leaves the system. This keeps ``"tail_walk"`` a free-escape bound with or
  without the ionization channel, so ``tail_ionization`` moves ONE thing and
  the end convention is not silently swapped underneath it.

``tail_walk_window`` is REQUIRED under ``"on"`` and is the inclusive cell range
the walkers may traverse -- for the solver, the plasma-active window, so that
the cathode disc and anything behind it are a wall. It exists because the
default "the whole grid" is WRONG for a particle channel and wrong quietly: a
``-z`` walker launched near the source runs on into the cells behind the
cathode, and at K5a breakdown conditions 5-66% of the tail's ionization lands
there (measured; ``k6build_tailion_crosscheck.txt``) -- in rows the solver's
active-plasma mask zeroes, so those pairs are created and then deleted. The
energy-only walk shares the geometry but leaks only ~0.04% of ``P_QL`` into it,
which is why the defect had no visible consequence until the walk started
carrying particles. Bounding the window also makes the walk agree with the
registered offline estimate, which truncates ``-z`` walks at the cathode disc
for the same reason.

All of the walkers' heat lands in ``heating_anomalous_erg_s``, which keeps its
documented meaning ("the anomalous channel's delivery to the electrons"); the
primary's own ``heating_coulomb`` / ``heating_secondary`` / ``heating_terminal``
splits are untouched. The tail's contributions to the shared banks are ALSO
reported separately as ``ionization_events_tail``, ``excitation_events_tail``,
``ionization_cost_tail_erg_s`` and ``radiated_tail_erg_s`` -- pure bookkeeping,
identically zero under ``"off"``, and what makes the per-ray branching
statement checkable::

    P_QL = heating_anomalous + ionization_cost_tail + radiated_tail
           + end_loss_tail_low + end_loss_tail_high

i.e. every launched eV ends in exactly one of {bulk heat via thermalization,
ionization investment, secondary-birth heat, radiation, end ledger}. The
overall per-ray identity is UNCHANGED in form -- the tail's cost and radiation
join the terms that already carry the primary's.

**The depth-1 cascade truncation is MEASURED, not assumed.** Secondaries bank
locally rather than walking, which is only correct while ``<W_sec>(E_tail)``
sits below ``E_stop`` -- the point at which a secondary could itself do
something inelastic. That is true at every shipped fixed rung (1.35 / 9.89 /
16.82 eV at 30 / 75 / 150 eV against the 20.6158 eV floor) and stops being true
above ``E_tail`` = 221.406 eV, the ``<W_sec>`` crossing. Both that bar and the
``E_stop`` bar below it are COMPUTED from the thresholds themselves, never
tabulated for the shipped rungs.

Marching outside the band (``tail_ionization``, K7b)
-----------------------------------------------------

Under K7 the tail birth energy is keyed to the LIVE cathode drop,
``E_tail = f * e*phi_c(t)``, so a single run sweeps a wide range of ``E_tail``
as ``phi_c`` climbs from its cold value to its drive plateau. Refusing at the
two bars, which is what K6 did, made that unrunnable: the cold foot sits below
the lower bar and the top ``f`` arm sits above the upper one, so no keyed
ionizing arm could start. The bars now select a TREATMENT per ray instead:

- **Below** ``E_stop``: the march is REVERTED to the energy-only walk for that
  ray. This is exact physics rather than a fallback -- no He inelastic channel
  is open below the lowest threshold, so zero ionization is the answer and not
  an approximation of one. The reversion is EXACT in the implementation sense
  too: the ray takes the identical branch, over the identical floats, that
  ``tail_ionization="off"`` would take for the same call, including that
  closure's own domain convention (windowed under ``tail_reflect_face``, the
  whole grid without it). It is deliberately NOT silent: the reverted power is
  reported in ``tail_sub_threshold_power_erg_s``, so the foot-phase reversion
  is quotable from a saved trajectory.
- **Above** the ``<W_sec>`` crossing: the march is ALLOWED, with the depth-1
  truncation kept and its cost DISCLOSED rather than assumed away. Banking the
  mean secondary locally there UNDERSTATES the tail's ionization, because a
  secondary that clears ``E_stop`` could itself ionize; the understatement was
  MEASURED at <= 2.0% at ``f = 1.0`` (the 2026-08-05 sheathwalk read) and no
  cross-section extrapolation is involved in it -- the EII table covers the
  whole range. The exposure is reported in ``tail_above_bar_power_erg_s``.
- **Above the EII table edge** (``HE_EII_EPS_TOP * I_ion``, ~999.98 eV at the
  module's own ``I_ion``): still a REFUSAL, and the only one left. There
  ``He_EII_cross_lkup`` clamps to its last node, so the walk would attenuate
  on an extrapolated cross section. It IS reachable: a caller keying
  ``E_tail`` to the cathode drop at ``f = 1.0`` puts it on the edge whenever
  the sheath solve sits at its capability-limited ceiling. The edge itself is
  therefore INCLUSIVE within ``HE_EII_EDGE_REL_TOL`` (K7c) -- at the edge the
  clamped value IS the table's last node, so nothing is extrapolated and there
  is nothing to refuse -- and a genuine excess beyond that tolerance raises,
  reporting the measured relative excess.

``tail_power_erg_s`` carries the launched ``P_QL`` so the two exposures can be
read as fractions. In band both are identically zero, which is the statement
that K7b changed nothing that already ran: the split activates ONLY where the
previous code refused outright.

Stated limitations, additional to the walk's own: the walker carries one mean
energy rather than a plateau distribution, so its cross sections are evaluated
at that mean; and it is launched at the near edge of its birth cell, the same
cell-resolution granularity as everywhere else in the module.

Sheath reflection at a walk-window face (``tail_reflect_face``, K7)
-------------------------------------------------------------------

Everything above lets a walker that reaches a window face LEAVE, whatever its
energy. That is the free-escape bound, and at a face that is a biased emitting
surface it is the wrong limit rather than a conservative one: a cathode sitting
at an accelerating drop of a few hundred volts repels every electron in the
plateau, so the flux the free-escape convention deletes is in fact returned to
the column.

``tail_reflect_face`` names the ONE window face that reflects and
``tail_reflect_threshold_eV`` the energy it reflects below. A walker arriving
at that face with energy strictly below the threshold is turned around at the
SAME energy and keeps marching from the face cell; at or above the threshold it
escapes to the tail end ledger exactly as before. The comparison is general
even though the physical threshold (``e*phi_c``) exceeds every shipped plateau
energy, so the reflecting arm is not silently a "reflect everything" arm.

Consequences, all of them deliberate:

- naming a face makes ``tail_walk_window`` REQUIRED in both tail modes, and
  the energy-only walk becomes WINDOWED. Without a window the energy-only walk
  runs the whole grid and there is no face at the cathode to reflect at; with
  one, the same two faces bound both tail modes;
- the reflected walker re-crosses the face cell, the cell-resolution
  granularity every launch in this module has;
- only ONE face may reflect. Two reflecting faces trap the walker between them
  with no escape channel, which neither the closed-form walk nor the march has
  a termination convention for, and it is refused rather than approximated.

STANDING RIDER, unsized: the reflected fraction assumed here is 1. A real
cathode end is a grounded wall with an emitting disc in it, so some of the
returning tail misses the disc RADIALLY and is lost to the wall at its own
much smaller potential. This 1D walk has no radial coordinate and cannot size
that fraction; it is the one assumption that could pull the measured ~2x
reflection gain below exact. It is a documented limitation, NOT a knob -- there
is no partial-reflection coefficient to turn, precisely so that nothing can be
fitted through it.

STANDING RIDER, unchanged by this: the 30 eV bracket arm sits below the
``min(4*Te, 30 eV)`` free-escape convention the offline estimate uses at the
ends, and that convention is the estimate's, not this module's; the walk here
reflects on ``tail_reflect_threshold_eV`` alone and free-escapes on everything
else.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ._cathode_solver import _c_log_ei
from ._cross import (
    _HE_LOG_EPS,
    _HE_LOG_SIGMA,
    He_EII_cross_lkup,
    He_beam_excitation_channel_lkup,
    _he_beam_excitation_table,
)
from ._kernels import COMPILED_KERNELS as _COMPILED_KERNELS

_ERG_PER_EV = 1.602176634e-12
_ME_CGS = 9.1093837015e-28  # electron mass [g]
_E4_CGS = (4.80320425e-10) ** 4  # e^4 [esu^4]
_OMEGA_PE_COEFF = 5.64e4  # omega_pe = 5.64e4 sqrt(n_e) [rad/s] (NRL)

HE_I_ION_EV = 24.587
HE_E_STOP_EV = 20.6158  # lowest inelastic threshold (2^1S)
HE_OPB_EBAR_EV = 15.8  # Opal-Peterson-Beaty shape parameter for He
# Top of the tabulated He EII cross section, in the table's own reduced units
# eps = E / I_ion (``He_EII_cross_lkup``). Taken FROM the table rather than
# written down, so it cannot drift from the data it describes. Above it the
# lookup clamps to the last node -- i.e. it extrapolates a constant sigma --
# which is the one thing the marched tail walk must never be allowed to do
# (K7b). At the module's own I_ion this is ~999.98 eV.
HE_EII_EPS_TOP = float(np.exp(_HE_LOG_EPS[-1]))
# Fractional slack on that edge (K7c). The edge is INCLUSIVE and carries this
# tolerance: a tail energy whose relative excess over ``HE_EII_EPS_TOP *
# I_ion_eV`` is <= this clamps to the table's last node, which AT the edge is
# the node's own value and therefore not an extrapolation at all; a larger
# excess still raises. The number brackets two measurable scales:
#   floor -- the arithmetic that produces the comparison. ``E_tail`` and the
#     edge are each a product of floats (1 ULP = 1.14e-16 relative here), and
#     under phi_c keying at the ceiling ``E_tail`` also carries the
#     capability-limited sheath root-find, run at rtol = 1e-14
#     (``_cathode_solver_idriven``). That is the largest term, so the slack
#     must exceed ~1e-14 to admit an ``E_tail`` the caller meant to place AT
#     the edge;
#   ceiling -- the table's own resolution. The last two nodes are 3.7e-3 apart
#     in eps, so 1e-12 is 2.7e-10 of one node gap, and a log-linear
#     extrapolation over it (end slope dln(sigma)/dln(eps) = -0.764) would
#     move sigma by 7.6e-13 relative. Nothing physical lives in that window.
# 1e-12 sits two decades above the floor and ten below the ceiling.
HE_EII_EDGE_REL_TOL = 1.0e-12

# --- Non-local product transport (product_transport="nonlocal") -------------
# Thermalization floor of a walking product, as a multiple of the local Te.
# 3/2 is the mean energy of the local Maxwellian: at that energy the product
# is statistically indistinguishable from the bulk it is dragging against, and
# the fast test-particle stopping power (which assumes W >> Te) has stopped
# being a description of anything. NOT a tunable -- it is the bulk mean
# energy, not a fitted number, and it is deliberately not exposed as a config
# key.
_PRODUCT_FLOOR_TE_MULTIPLE = 1.5
# Absolute floor [eV], reusing the module's OWN lnLambda temperature clamp
# (``_c_log_ei(max(Te, 0.1), ne)`` in coulomb_stopping_eV_per_cm): below it the
# Coulomb logarithm is already being evaluated at a substitute temperature, so
# a walk there would be tracking a formula outside its own domain. Also the
# guard that keeps the walk finite when Te -> 0.
_PRODUCT_FLOOR_MIN_EV = 0.1
# Energy exponent p of each closure's stopping power, dE/dx = A(ne,Te) * W**p.
# Both closures are exact power laws in W -- lnLambda depends only on (ne, Te)
# -- which is what makes the product walk integrable in closed form below.
_COULOMB_STOPPING_EXPONENT = {"fast_electron": -1.0, "legacy_tau_ei": 0.5}

# --- Compiled CSDA march (opt-in; see cablp.funcs._kernels) ------------------
# The cost read of 2026-08-02 measured the substep march at ~61% numpy SCALAR
# dispatch and Python call overhead -- ~873 sub-calls per ``deposit_beam``
# call -- so the compiled unit's boundary encloses the WHOLE double loop
# rather than any leaf. Nothing here changes on the default pure path: the
# module-level bind is one ``is None`` test at import, and ``deposit_beam``
# keeps its Python march verbatim as the fallback and the equivalence target.
#
# Model selectors are integers across the boundary because the transcription
# cannot raise ``coulomb_stopping_eV_per_cm``'s ValueError from the place the
# Python raises it; an unrecognised model is simply not offered to the kernel
# and takes the Python march, which raises exactly as it always did.
_COULOMB_MODEL_CODE = {"fast_electron": 0, "legacy_tau_ei": 1}

_CSDA_MARCH = None
if _COMPILED_KERNELS is not None:
    _COMPILED_KERNELS.check_constants_beam(
        _ERG_PER_EV, _ME_CGS, _E4_CGS, _OMEGA_PE_COEFF, math.pi
    )
    _CSDA_MARCH = _COMPILED_KERNELS.csda_march

_CSDA_TABLES = None
_CSDA_TABLES_SRC = None


def _csda_tables():
    """The compiled march's view of the three cross-section tables.

    Rebuilt only when ``_cross``'s lazily-built excitation table is replaced
    (it is cached there and only rebuilt for a different ``n_max``, which
    ``deposit_beam`` never asks for), so the steady state is an identity test.
    The log-log EII table is a module constant in ``_cross`` and never moves.
    """
    global _CSDA_TABLES, _CSDA_TABLES_SRC
    src = _he_beam_excitation_table(20)
    if _CSDA_TABLES_SRC is not src:
        _CSDA_TABLES = _COMPILED_KERNELS.CsdaTables(
            _HE_LOG_EPS, _HE_LOG_SIGMA, src[1], src[2], src[3]
        )
        _CSDA_TABLES_SRC = src
    return _CSDA_TABLES


def _coulomb_stopping_coefficient(ne, Te, model):
    """Per-cell ``A`` in ``dE/dx = A * W**p`` [eV/cm], from the module's own
    ``coulomb_stopping_eV_per_cm``.

    Evaluated at ``W = 1 eV``, where ``W**p == 1`` for either exponent, so the
    coefficient carries the closure's constants, ``ne``, ``Te`` and lnLambda
    exactly as the primary's stopping power does -- the walk cannot drift away
    from the primary's drag law without ``coulomb_stopping_eV_per_cm`` itself
    changing shape (which the smoke suite checks).
    """
    return np.array(
        [
            coulomb_stopping_eV_per_cm(1.0, float(n), float(t), model=model)
            for n, t in zip(np.asarray(ne).ravel(), np.asarray(Te).ravel())
        ],
        dtype=float,
    )


def _walk_products_forward(W0_eV, flux_per_s, coeff, dz_cm, floor_eV, q):
    """Coulomb slowing walk of per-cell product populations toward +index.

    ``W0_eV[s]`` / ``flux_per_s[s]`` are the birth energy and flux of the
    population born in cell ``s``; the walk proceeds s, s+1, ... Returns
    ``(deposited_eV_per_s, exit_eV_per_s, exit_state)``: per-cell deposited
    power in ``flux*eV`` units, the scalar power carried out of the last cell,
    and ``exit_state = (birth_cells, W_exit_eV, thermalized)`` -- the PER-BIRTH
    breakdown of that scalar, which a reflecting boundary needs because the
    populations arrive at the face with different energies and only some of
    them may be below its threshold. ``exit_state`` is pure bookkeeping: it
    re-reads quantities the walk already formed and no caller that ignores it
    sees any difference.

    The integration is EXACT, not substepped. With ``dE/dx = A_j W**p`` held
    constant across a cell, ``u = W**q`` with ``q = 1 - p`` obeys
    ``du/dz = -q A_j``, so ``u`` falls linearly and a cumulative sum gives the
    energy at every cell exit in closed form. Deposition in a cell is the
    difference of the entry and exit energies, and the entry energy of a cell
    is REUSED as the exit energy of its predecessor (the identical float), so
    the per-ray sum telescopes to the birth energy exactly rather than to
    accumulated roundoff.
    """
    cells = int(dz_cm.size)
    dep = np.zeros(cells)
    W0_eV = np.asarray(W0_eV, dtype=float)
    flux_per_s = np.asarray(flux_per_s, dtype=float)
    active = np.flatnonzero((flux_per_s > 0.0) & (W0_eV > 0.0))
    if active.size == 0:
        return dep, 0.0, (active, np.zeros(0), np.zeros(0, dtype=bool))
    W0v = W0_eV[active]
    fluxv = flux_per_s[active]
    index = np.arange(cells)
    reach = index[None, :] >= active[:, None]
    # u after LEAVING each cell, per birth cell (cumulative sum starts at the
    # birth cell, so cells behind it contribute nothing and read back u = W0**q).
    delta = q * coeff * dz_cm
    u_out = np.power(W0v, q)[:, None] - np.cumsum(
        np.where(reach, delta[None, :], 0.0), axis=1
    )
    W_out = np.power(np.maximum(u_out, 0.0), 1.0 / q)
    W_in = np.empty_like(W_out)
    W_in[:, 0] = W0v
    W_in[:, 1:] = W_out[:, :-1]
    # The birth cell's entry energy is the birth energy by definition; pin it
    # rather than trusting (W0**q)**(1/q) to round back onto W0.
    W_in[np.arange(active.size), active] = W0v
    # Thermalized where the cell-exit energy has fallen to the local floor.
    stop = reach & (W_out <= floor_eV[None, :])
    has_stop = stop.any(axis=1)
    j_stop = np.where(has_stop, np.argmax(stop, axis=1), cells)
    before = reach & (index[None, :] < j_stop[:, None])
    at = reach & (index[None, :] == j_stop[:, None])
    contrib = np.where(before, W_in - W_out, 0.0) + np.where(at, W_in, 0.0)
    dep += (fluxv[:, None] * contrib).sum(axis=0)
    exit_eV = float(
        np.sum(fluxv * np.where(has_stop, 0.0, W_out[:, -1]))
    )
    return dep, exit_eV, (active, W_out[:, -1], has_stop)


def _walk_products(W0_eV, flux_per_s, direction, coeff, dz_cm, floor_eV, q):
    """Direction-aware wrapper around ``_walk_products_forward``.

    Returns ``(deposited_eV_per_s, exit_eV_per_s)``; the exit power leaves the
    HIGH-index end for ``direction > 0`` and the LOW-index end otherwise. The
    per-birth exit state is dropped -- the reflecting walk below indexes cells
    in its own traversal order and calls the forward walk directly.
    """
    if direction > 0:
        dep, exit_eV, _ = _walk_products_forward(
            W0_eV, flux_per_s, coeff, dz_cm, floor_eV, q
        )
        return dep, exit_eV
    dep, exit_eV, _ = _walk_products_forward(
        W0_eV[::-1], flux_per_s[::-1], coeff[::-1], dz_cm[::-1],
        floor_eV[::-1], q,
    )
    return dep[::-1], exit_eV


def beam_speed_cm_s(E_eV: float) -> float:
    """Nonrelativistic primary speed [cm/s] at kinetic energy E_eV."""
    return math.sqrt(2.0 * E_eV * _ERG_PER_EV / _ME_CGS)


def he_mean_secondary_energy_eV(
    E_eV: float,
    I_ion_eV: float = HE_I_ION_EV,
    ebar_eV: float = HE_OPB_EBAR_EV,
) -> float:
    """Mean secondary-electron energy [eV] per He ionization at E_eV.

    OPB shape over W in [0, (E - I)/2]:
    <W> = Ebar * ln(1 + x^2) / (2 atan(x)), x = W_max / Ebar.
    """
    W_max = 0.5 * (E_eV - I_ion_eV)
    if W_max <= 0.0:
        return 0.0
    x = W_max / ebar_eV
    return ebar_eV * math.log1p(x * x) / (2.0 * math.atan(x))


def coulomb_stopping_eV_per_cm(
    E_eV: float, ne: float, Te: float, model: str = "fast_electron"
) -> float:
    """Coulomb energy-loss rate [eV/cm] of a primary at E_eV."""
    if ne <= 0.0 or E_eV <= 0.0:
        return 0.0
    lnL = _c_log_ei(max(Te, 0.1), ne)
    if model == "fast_electron":
        return 2.0 * math.pi * _E4_CGS * ne * lnL / (E_eV * _ERG_PER_EV) / _ERG_PER_EV
    if model == "legacy_tau_ei":
        tau_ei = 3.44e5 * Te**1.5 / ne / lnL
        return E_eV / (beam_speed_cm_s(E_eV) * tau_ei)
    raise ValueError(
        f"unknown coulomb_model {model!r}; "
        "expected 'fast_electron' or 'legacy_tau_ei'"
    )


def quasilinear_relaxation_length_cm(
    E_eV: float, ne: float, n_b: float
) -> float:
    """Quasilinear beam relaxation length [cm], l_QL = (ne/nb)(v_b/w_pe)ln(ne/nb).

    Valid for a weak beam (n_b << n_e); returns inf outside that domain
    (n_b >= n_e/10) rather than extrapolating the theory into the strong-beam
    regime, and inf for an absent beam or plasma.
    """
    if ne <= 0.0 or n_b <= 0.0 or n_b >= 0.1 * ne:
        return math.inf
    omega_pe = _OMEGA_PE_COEFF * math.sqrt(ne)
    ratio = ne / n_b
    return ratio * (beam_speed_cm_s(E_eV) / omega_pe) * math.log(ratio)


@dataclass(frozen=True)
class BeamDepositionResult:
    """Per-cell deposition of one beam ray; arrays have shape (cells,).

    ionization_events   : ionization events [1/s]
    excitation_events   : excitation events [1/s]
    plasma_heating_erg_s: Coulomb + anomalous drag + secondary-electron
                          energy + sub-threshold residual [erg/s]
    heating_coulomb_erg_s   : DIAGNOSTIC split of ``plasma_heating_erg_s`` --
                          the continuous Coulomb drag on plasma electrons
                          [erg/s]
    heating_anomalous_erg_s : DIAGNOSTIC split -- the anomalous (quasilinear
                          beam-plasma) drag [erg/s]; identically zero under
                          ``anomalous_model="none"``. Under
                          ``anomalous_transport="tail_walk"`` this is the
                          WALKED deposition profile (where the tail electrons
                          actually slowed), not the birth profile.
    heating_secondary_erg_s : DIAGNOSTIC split -- the inelastic-EVENT thermal
                          residue, i.e. the mean secondary-electron energy
                          <W_sec> carried away per ionization ABOVE the
                          ``I_ion`` potential cost [erg/s]
    heating_terminal_erg_s  : DIAGNOSTIC split -- the primary's end-of-range
                          terminal dump: the sub-threshold residual banked
                          whole in the cell where E crosses ``E_stop`` [erg/s]

    The four ``heating_*`` arrays are pure BOOKKEEPING of products the energy
    decrement already forms; they are accumulated alongside, never in place
    of, ``plasma_heating_erg_s``, which keeps its exact historical value and
    is the only one the solver RHS consumes. Their sum reproduces
    ``plasma_heating_erg_s`` to floating-point associativity only (the lumped
    bank adds the three per-substep pieces before multiplying), so compare
    them with a relative tolerance, never with ``==``.

    radiated_erg_s      : excitation line radiation [erg/s]
    ionization_cost_erg_s: I_ion * ionization events [erg/s] (kept separate
                          to map onto the solver's beam_ionization_cost term)
    transmitted_flux    : primary flux leaving the far end [1/s] (reduced by
                          the anode-mesh interception, if any)
    transmitted_energy_eV: mean primary energy at exit [eV] (0 if absorbed)
    anode_intercepted_erg_s: energy the anode mesh intercepts at the anode-face
                          crossing [erg/s] (audit A15). This leaves the plasma
                          (booked to the electrode, NOT plasma_heating_erg_s);
                          0 when no interception is requested or the ray stops
                          before the anode face.
    E_entry_eV          : diagnostic: primary energy entering each cell [eV]
                          (0 for cells the ray never reaches)
    end_loss_low_erg_s  : END LEDGER, low-index end [erg/s]. Identically 0.0
                          under ``product_transport="local"``. Under
                          ``"nonlocal"`` it books (a) the remaining energy of
                          product walks that leave that end without
                          thermalizing and (b) the transmitted PRIMARY's power
                          ``Gamma_t * E_t`` when the ray exits there. This
                          energy LEAVES the system: like
                          ``anode_intercepted_erg_s`` it is NOT part of
                          ``plasma_heating_erg_s`` and must not enter any RHS
                          row.
    end_loss_high_erg_s : END LEDGER, high-index end [erg/s]; same content.
    end_loss_transmitted_erg_s: DIAGNOSTIC split -- how much of
                          ``end_loss_low + end_loss_high`` is the transmitted
                          primary rather than walked products. 0.0 under
                          ``"local"``.
    end_loss_tail_low_erg_s : WP-E TAIL END LEDGER, low-index end [erg/s].
                          Identically 0.0 under
                          ``anomalous_transport="local"`` (the default). Under
                          ``"tail_walk"`` it books the energy of QL tail
                          electrons that reach that end without thermalizing.
                          A SIBLING of ``end_loss_low_erg_s``, deliberately
                          kept separate so the WP-D product ledger keeps its
                          documented meaning and its measured values while the
                          two closures switch independently. Like the WP-D
                          ledger this energy LEAVES the system and must not
                          enter any RHS row.
    end_loss_tail_high_erg_s: WP-E TAIL END LEDGER, high-index end [erg/s];
                          same content.
    ionization_events_tail  : K6 DIAGNOSTIC SPLIT -- how much of
                          ``ionization_events`` the QL tail walkers contributed
                          [1/s]. Identically zero under ``tail_ionization="off"``
                          (the default), where the walkers cannot ionize at all.
    excitation_events_tail  : K6 DIAGNOSTIC SPLIT of ``excitation_events``
                          [1/s]; same status.
    ionization_cost_tail_erg_s: K6 DIAGNOSTIC SPLIT of ``ionization_cost_erg_s``
                          [erg/s]; same status.
    radiated_tail_erg_s     : K6 DIAGNOSTIC SPLIT of ``radiated_erg_s``
                          [erg/s]; same status.
    tail_power_erg_s        : K7b EXPOSURE LEDGER -- the total QL tail power
                          this ray launched into the walk [erg/s], i.e. the
                          withheld ``P_QL`` before any of it is deposited.
                          The DENOMINATOR of the two band fractions below.
                          Identically 0.0 under
                          ``anomalous_transport="local"``.
    tail_sub_threshold_power_erg_s: K7b -- how much of ``tail_power_erg_s`` was
                          marched with the ionizing channel REVERTED because
                          ``E_tail`` sits at or below ``E_stop_eV`` [erg/s].
                          Nonzero only under ``tail_ionization="on"``; it is
                          the whole tail power when it is nonzero, because
                          ``E_tail`` is one number per ray. What makes the
                          foot-phase reversion readable from a saved file
                          rather than being a silent no-op.
    tail_above_bar_power_erg_s: K7b -- how much of ``tail_power_erg_s`` was
                          marched ABOVE the depth-1 ``<W_sec>`` bar [erg/s],
                          i.e. under the <= 2.0% cascade understatement. Same
                          status and same all-or-nothing structure.

    The four ``*_tail`` arrays are bookkeeping, exactly like the four
    ``heating_*`` splits: they re-report a SUBSET of banks the shared arrays
    already carry, and nothing downstream consumes them in place of those
    arrays. Their purpose is that the tail channel's own energy branching
    (module docstring, K6) stays readable when the primary is filling the same
    banks.
    """

    ionization_events: np.ndarray
    excitation_events: np.ndarray
    plasma_heating_erg_s: np.ndarray
    radiated_erg_s: np.ndarray
    ionization_cost_erg_s: np.ndarray
    transmitted_flux: float
    transmitted_energy_eV: float
    anode_intercepted_erg_s: float
    E_entry_eV: np.ndarray
    heating_coulomb_erg_s: np.ndarray
    heating_anomalous_erg_s: np.ndarray
    heating_secondary_erg_s: np.ndarray
    heating_terminal_erg_s: np.ndarray
    ionization_events_tail: np.ndarray
    excitation_events_tail: np.ndarray
    ionization_cost_tail_erg_s: np.ndarray
    radiated_tail_erg_s: np.ndarray
    end_loss_low_erg_s: float = 0.0
    end_loss_high_erg_s: float = 0.0
    end_loss_transmitted_erg_s: float = 0.0
    end_loss_tail_low_erg_s: float = 0.0
    end_loss_tail_high_erg_s: float = 0.0
    tail_power_erg_s: float = 0.0
    tail_sub_threshold_power_erg_s: float = 0.0
    tail_above_bar_power_erg_s: float = 0.0


def deposit_beam(
    E0_eV: float,
    Gamma0_per_s: float,
    nn: np.ndarray,
    ne: np.ndarray,
    Te: np.ndarray,
    launch: int,
    direction: int,
    dz_cm: np.ndarray,
    *,
    I_ion_eV: float = HE_I_ION_EV,
    E_stop_eV: float = HE_E_STOP_EV,
    coulomb_model: str = "fast_electron",
    anomalous_model: str = "none",
    beam_area_cm2: np.ndarray | float | None = None,
    max_energy_fraction_per_substep: float = 0.02,
    anode_cross_index: int | None = None,
    anode_eta: float = 0.0,
    product_transport: str = "local",
    anomalous_transport: str = "local",
    tail_energy_eV: float | None = None,
    tail_ionization: str = "off",
    tail_walk_window: tuple[int, int] | None = None,
    tail_reflect_face: int | None = None,
    tail_reflect_threshold_eV: float | None = None,
    stopping_coefficient: np.ndarray | None = None,
) -> BeamDepositionResult:
    """Deposit one monoenergetic beam ray through the column (He only).

    ``E0_eV`` is the accelerating sheath drop,
    ``Gamma0_per_s`` the accepted emitted electron flux, ``nn/ne/Te`` the
    per-cell column state, and the ray is ``(launch, direction, dz_cm)``
    with ``direction`` +1 (toward increasing index) or -1. Cells behind the
    launch point receive nothing.

    ``anomalous_model``: ``"none"`` (default) or ``"quasilinear"``
    (requires ``beam_area_cm2``, scalar or per-cell, to form n_b).

    **Anode-mesh interception (audit A15).** ``anode_cross_index`` is the first
    cell on the far (column) side of the anode face along the ray; when it is
    given with ``anode_eta`` in ``[0, 1)`` the mesh intercepts the solid
    fraction ``anode_eta`` of the flux STILL STREAMING when the ray reaches
    that face (i.e. the long-mean-free-path beam that survived the gap). The
    intercepted power ``anode_eta * gamma * E`` is booked to
    ``anode_intercepted_erg_s`` (it leaves the plasma, landing on the anode,
    NOT in ``plasma_heating_erg_s``), and the surviving flux is reduced to
    ``(1 - anode_eta) * gamma`` for all subsequent deposition and ionization.
    A ray that stops in the gap never reaches the face and intercepts nothing,
    so only the survived (bypass) fraction is removed -- consistent with the
    circuit's ``eta * beam_bypass_fraction``. Per-ray energy still closes to
    roundoff::

        Gamma0*E0 = heating + radiated + cost + anode_intercepted + transmitted

    Off (``anode_cross_index is None`` or ``anode_eta == 0``) the running flux
    is the constant ``Gamma0_per_s`` throughout, so every bank is byte-for-byte
    the historical result.

    **Product transport (WP-D).** ``product_transport`` is ``"local"``
    (default) or ``"nonlocal"``; see the module docstring for the physics. Off
    (the default, and the value the gap-transmission probe call sites leave
    untouched) not one branch below changes, so the result is byte-for-byte
    the historical one and ``end_loss_*`` are identically zero. On, the
    secondary and terminal-residual banks are withheld from their birth cells
    and re-deposited along mini-CSDA walks, with what escapes an end booked to
    the end ledger. Energy-only: ``ionization_events`` and
    ``excitation_events`` -- and therefore every particle and circuit row
    downstream -- are identical in both modes.

    **QL heating locality (WP-E).** ``anomalous_transport`` is ``"local"``
    (default) or ``"tail_walk"``; see the module docstring for the physics.
    Off, no branch below changes and the result is byte-for-byte the
    historical one with ``end_loss_tail_*`` identically zero. On, the
    anomalous channel's power is withheld from its birth cell, re-expressed as
    tail electrons at ``tail_energy_eV`` (required, finite, > 0) launched
    50/50 along +-B, and walked on the same Coulomb machinery as the WP-D
    products; escapes go to the SEPARATE tail end ledger. ``"tail_walk"``
    requires an active anomalous channel (``anomalous_model="quasilinear"``) --
    with no anomalous drag there is no power to carry and the setting would be
    a silent no-op. Energy-only, exactly like WP-D. The two closures are
    independent and compose: with both on, the event products walk on the WP-D
    ledger and the QL tails on the WP-E one.

    **Tail ionization (K6).** ``tail_ionization`` is ``"off"`` (default,
    bit-exact -- the walk stays energy-only) or ``"on"``, which marches each
    tail population on this module's own CSDA integration so it ionizes and
    excites the column gas on its way; see the module docstring. Requires
    ``anomalous_transport="tail_walk"`` (there is no other walk to give the
    channel to) and ``tail_walk_window=(lo, hi)``, the inclusive cell range the
    walkers may traverse; see the module docstring for why the window has no
    safe default. Both are ValueErrors, not silent adjustments.

    **The band split (K7b).** ``tail_energy_eV`` no longer has to sit inside
    the depth-1 band; the two bars select a treatment for the ray instead of
    refusing it. At or below ``E_stop_eV`` the march REVERTS to the energy-only
    walk -- exactly the branch and exactly the floats ``"off"`` would produce
    for this call, because no inelastic channel is open there -- and the
    reverted power is reported in ``tail_sub_threshold_power_erg_s``. Above the
    ``<W_sec>(E_tail) >= E_stop_eV`` crossing the march RUNS with the depth-1
    truncation, whose cascade understatement is measured at <= 2.0%, and the
    exposed power is reported in ``tail_above_bar_power_erg_s``. Only the EII
    table edge (``HE_EII_EPS_TOP * I_ion_eV``) is still a ValueError: past it
    the ionization lookup clamps to its last node and the walk would be running
    on an extrapolated cross section. That edge is INCLUSIVE within
    ``HE_EII_EDGE_REL_TOL`` (K7c) -- ``E_tail`` there evaluates the cross
    section AT the table's last node, which is the node's own value rather than
    an extrapolation of it -- and a relative excess larger than the tolerance
    raises, reporting what it measured. In band both exposure fields are zero
    and nothing about the march changed.

    **Sheath reflection at a walk-window face (K7).** ``tail_reflect_face`` is
    ``None`` (default, bit-exact: both faces free-escape, as WP-E and K6 ship)
    or ``-1`` / ``+1``, naming the ONE face of ``tail_walk_window`` that
    reflects tail walkers. A walker arriving there is reflected -- same energy,
    reversed direction -- when its arrival energy is strictly below
    ``tail_reflect_threshold_eV`` (required, finite, > 0 with a face named),
    and free-escapes to the tail end ledger otherwise; see the module
    docstring. Naming a face also requires ``tail_walk_window``, whose faces
    become the walk's boundary in BOTH tail modes: the energy-only walk is
    windowed under reflection where it runs the whole grid without it, because
    a reflecting face has to be a face the walk actually stops at. Only one
    face may reflect: two reflecting faces trap the walker between them, which
    this closed-form walk has no termination convention for, and the caller is
    expected to refuse that configuration rather than have it silently
    approximated here.

    ``stopping_coefficient`` (cost read 2026-08-02, restructure C) is the
    per-cell ``A`` of ``dE/dx = A W**p`` that the walks below need, HOISTED to
    the caller. Left ``None`` -- the default, and every historical call -- it
    is built here by ``_coulomb_stopping_coefficient`` exactly as before.
    Supplied, it is used verbatim, so a caller that launches several rays or
    (under WP-F) several energy groups over the SAME ``(ne, Te, model)`` pays
    for the 262-iteration Python listcomp once instead of once per ray: it is
    100 us, half of the entire WP-E per-call surcharge. Bit-exact either way --
    the caller is expected to build it with this module's own
    ``_coulomb_stopping_coefficient``, and it is only read when a walk closure
    is active, so the default path never touches it.
    """
    if product_transport not in ("local", "nonlocal"):
        raise ValueError(
            f"unknown product_transport {product_transport!r}; "
            "expected 'local' or 'nonlocal'"
        )
    if anomalous_transport not in ("local", "tail_walk"):
        raise ValueError(
            f"unknown anomalous_transport {anomalous_transport!r}; "
            "expected 'local' or 'tail_walk'"
        )
    if tail_ionization not in ("off", "on"):
        raise ValueError(
            f"unknown tail_ionization {tail_ionization!r}; "
            "expected 'off' or 'on'"
        )
    if tail_ionization == "on" and anomalous_transport != "tail_walk":
        raise ValueError(
            "tail_ionization='on' requires anomalous_transport='tail_walk' "
            "(the ionizing channel belongs to the QL tail walkers; with "
            "anomalous_transport='local' there are no walkers and the "
            "setting would do nothing). anomalous_transport accepts 'local' "
            "or 'tail_walk'; tail_ionization accepts 'off' or 'on'"
        )
    if anode_eta != 0.0 and not (0.0 <= anode_eta < 1.0):
        raise ValueError(
            f"anode_eta must be in [0, 1) (got {anode_eta})"
        )
    nn = np.asarray(nn, dtype=float)
    ne = np.asarray(ne, dtype=float)
    Te = np.asarray(Te, dtype=float)
    dz_cm = np.asarray(dz_cm, dtype=float)
    cells = dz_cm.size
    if anode_cross_index is not None:
        anode_cross_index = int(anode_cross_index)
        if not 0 <= anode_cross_index < cells:
            raise ValueError(
                "anode_cross_index must index a cell in [0, cells) "
                f"(got {anode_cross_index}, cells={cells})"
            )
    if nn.shape != (cells,) or ne.shape != (cells,) or Te.shape != (cells,):
        raise ValueError("nn, ne, Te, dz_cm must share one shape (cells,)")
    if direction not in (-1, 1):
        raise ValueError(f"direction must be +1 or -1 (got {direction})")
    if anomalous_model not in ("none", "quasilinear"):
        raise ValueError(
            f"unknown anomalous_model {anomalous_model!r}; "
            "expected 'none' or 'quasilinear'"
        )
    if anomalous_model == "quasilinear":
        if beam_area_cm2 is None:
            raise ValueError("anomalous_model='quasilinear' needs beam_area_cm2")
        area = np.broadcast_to(
            np.asarray(beam_area_cm2, dtype=float), (cells,)
        )
    walk_tail = anomalous_transport == "tail_walk"
    E_tail = 0.0
    if walk_tail:
        # There must BE an anomalous channel for the walk to carry; without one
        # the setting is a silent no-op, which is exactly what the presence
        # gating exists to prevent.
        if anomalous_model == "none":
            raise ValueError(
                "anomalous_transport='tail_walk' requires an active anomalous "
                "channel (anomalous_model='quasilinear'); with no anomalous "
                "drag there is no power to carry and the setting would do "
                "nothing"
            )
        if tail_energy_eV is None:
            raise ValueError(
                "anomalous_transport='tail_walk' needs tail_energy_eV (the "
                "QL plateau energy the tail electrons are launched at)"
            )
        E_tail = float(tail_energy_eV)
        if not math.isfinite(E_tail) or E_tail <= 0.0:
            raise ValueError(
                "tail_energy_eV must be finite and > 0 (got "
                f"{tail_energy_eV})"
            )
    ionize_tail = tail_ionization == "on"
    # K7b band diagnostics for THIS ray. ``tail_sub_threshold`` means the
    # requested channel was reverted to the energy-only walk because E_tail
    # sits at or below the lowest inelastic threshold; ``tail_above_bar``
    # means it marched with the depth-1 truncation past the <W_sec> crossing.
    # At most one can be true (E_tail is one number per ray) and both are
    # false in band, which is what makes the in-band path provably untouched.
    tail_sub_threshold = False
    tail_above_bar = False
    # K7 sheath reflection. The face and its threshold are one setting: a face
    # with no threshold cannot be tested and a threshold with no face has
    # nothing to test, so each without the other is a refusal rather than a
    # default.
    reflect_face = None
    E_reflect = 0.0
    if tail_reflect_face is not None:
        if not walk_tail:
            raise ValueError(
                "tail_reflect_face requires anomalous_transport='tail_walk' "
                "(it reflects the QL tail walkers; with no walk there is "
                "nothing to reflect and the setting would do nothing)"
            )
        reflect_face = int(tail_reflect_face)
        if reflect_face not in (-1, 1):
            raise ValueError(
                "tail_reflect_face must be -1 (the walk window's low-index "
                f"face) or +1 (its high-index face), got {tail_reflect_face!r}"
            )
        if tail_reflect_threshold_eV is None:
            raise ValueError(
                "tail_reflect_face needs tail_reflect_threshold_eV (the "
                "energy below which a walker arriving at that face is turned "
                "around instead of escaping)"
            )
        E_reflect = float(tail_reflect_threshold_eV)
        if not math.isfinite(E_reflect) or E_reflect <= 0.0:
            raise ValueError(
                "tail_reflect_threshold_eV must be finite and > 0 (got "
                f"{tail_reflect_threshold_eV})"
            )
    elif tail_reflect_threshold_eV is not None:
        raise ValueError(
            "tail_reflect_threshold_eV was given without tail_reflect_face; "
            "the threshold belongs to a named reflecting face and on its own "
            "would silently do nothing"
        )
    tail_lo, tail_hi = 0, cells - 1
    if reflect_face is not None and not ionize_tail:
        # The reflecting face IS a window face, so the window is required in
        # both tail modes once reflection is on -- see the docstring.
        if tail_walk_window is None:
            raise ValueError(
                "tail_reflect_face needs tail_walk_window=(lo, hi): the "
                "reflecting face is one of that window's two faces, and "
                "without it the walk has no face to reflect at"
            )
        tail_lo, tail_hi = (int(tail_walk_window[0]), int(tail_walk_window[1]))
        if not 0 <= tail_lo <= tail_hi < cells:
            raise ValueError(
                "tail_walk_window must be an inclusive (lo, hi) cell range "
                f"with 0 <= lo <= hi < cells={cells} (got "
                f"{tail_walk_window})"
            )
    if ionize_tail:
        # THE WALK DOMAIN IS REQUIRED, not defaulted to the whole grid. This
        # module is solver-agnostic and cannot know which cells are plasma, but
        # an ionizing walk that leaves the plasma is not a small error: a ``-z``
        # walker launched near the source runs into whatever cells sit behind
        # the cathode, and at K5a conditions that is where 5-66% of its
        # ionization lands (measured, k6build_tailion_crosscheck.txt) -- births
        # into rows the solver's active-plasma mask ZEROES, i.e. pairs created
        # and silently deleted. The energy-only walk survives the same geometry
        # only because it leaks ~0.04% of P_QL there rather than most of its
        # product; a PARTICLE channel cannot be built on that domain. So the
        # caller states the window and a caller that has not thought about it
        # gets an error, not a quiet sink.
        if tail_walk_window is None:
            raise ValueError(
                "tail_ionization='on' needs tail_walk_window=(lo, hi), the "
                "inclusive cell range the tail walkers may traverse (the "
                "plasma-active window: a walker leaving it hits a wall and is "
                "booked to the tail end ledger). Without it the walk would "
                "run off into cells whose plasma rows the solver zeroes and "
                "birth pairs that are then deleted"
            )
        tail_lo, tail_hi = (int(tail_walk_window[0]), int(tail_walk_window[1]))
        if not 0 <= tail_lo <= tail_hi < cells:
            raise ValueError(
                "tail_walk_window must be an inclusive (lo, hi) cell range "
                f"with 0 <= lo <= hi < cells={cells} (got "
                f"{tail_walk_window})"
            )
        # --- The K7b band split (module docstring) -----------------------
        # The two depth-1 bars are still COMPUTED from the thresholds
        # themselves, but each now selects a treatment instead of refusing.
        # BELOW E_stop the march reverts to the energy-only walk -- exact
        # physics, no inelastic channel is open -- and ABOVE the <W_sec>
        # crossing the march runs with the depth-1 truncation and its measured
        # understatement. The ONE surviving refusal is the EII table edge:
        # there the lookup would clamp to its last node and the walk would be
        # marching on an extrapolated cross section.
        # The edge itself is INCLUSIVE, within HE_EII_EDGE_REL_TOL (K7c): AT
        # the edge the clamped cross section IS the table's endpoint value, so
        # no extrapolation happens and there is nothing to refuse. Only a
        # genuine excess -- more than the tolerance above the edge -- is
        # refused, and the message reports the measured excess so a refusal
        # can be told apart from float noise at a glance.
        _E_table_top = HE_EII_EPS_TOP * I_ion_eV
        _edge_excess = (E_tail - _E_table_top) / _E_table_top
        if _edge_excess > HE_EII_EDGE_REL_TOL:
            raise ValueError(
                "tail_ionization='on' marches the walkers on the tabulated "
                "He EII cross section, which ends at eps = E/I_ion = "
                f"{HE_EII_EPS_TOP:.6f} (i.e. "
                f"{_E_table_top:.2f} eV at I_ion_eV={I_ion_eV}); "
                f"at tail_energy_eV={E_tail} eV the lookup would clamp to its "
                "last node and the walk would attenuate on an extrapolated "
                "cross section. This is refused, not approximated (relative "
                f"excess {_edge_excess:.3e}, tolerated "
                f"{HE_EII_EDGE_REL_TOL:.1e})"
            )
        if E_tail <= E_stop_eV:
            # SUB-THRESHOLD: no He inelastic channel is open, so zero
            # ionization is not a modeling choice but the answer. Revert this
            # ray to the energy-only walk by clearing the flag -- the branches
            # below then take the identical path, on the identical floats,
            # that ``tail_ionization="off"`` would take for this same call.
            tail_sub_threshold = True
            ionize_tail = False
        elif (
            he_mean_secondary_energy_eV(E_tail, I_ion_eV=I_ion_eV)
            >= E_stop_eV
        ):
            # ABOVE THE DEPTH-1 BAR: the mean secondary can itself do
            # something inelastic, so banking it locally UNDERSTATES the
            # cascade. Allowed, with the understatement measured (<= 2.0% at
            # f = 1.0, module docstring) and the exposure reported.
            tail_above_bar = True
    if stopping_coefficient is not None:
        stopping_coefficient = np.asarray(stopping_coefficient, dtype=float)
        if stopping_coefficient.shape != (cells,):
            raise ValueError(
                "stopping_coefficient must have shape (cells,) = "
                f"{(cells,)} (got {stopping_coefficient.shape})"
            )
    frac = float(max_energy_fraction_per_substep)
    if not 0.0 < frac < 1.0:
        raise ValueError(
            "max_energy_fraction_per_substep must be in (0, 1), got "
            f"{max_energy_fraction_per_substep}"
        )

    ionization_events = np.zeros(cells)
    excitation_events = np.zeros(cells)
    heating = np.zeros(cells)  # erg/s
    radiated = np.zeros(cells)
    ionization_cost = np.zeros(cells)
    E_entry = np.zeros(cells)
    # Diagnostic splits of `heating` (see BeamDepositionResult). Accumulated
    # from the SAME products the lumped bank uses; nothing here feeds the RHS.
    heat_coulomb = np.zeros(cells)
    heat_anomalous = np.zeros(cells)
    heat_secondary = np.zeros(cells)
    heat_terminal = np.zeros(cells)
    # K6 diagnostic splits: the QL tail walkers' share of the four shared
    # banks they now write into. Allocated unconditionally (four cell-sized
    # arrays) so every result carries the same fields; they stay all-zero
    # unless the ionizing walk actually runs.
    ion_events_tail = np.zeros(cells)
    exc_events_tail = np.zeros(cells)
    ion_cost_tail = np.zeros(cells)
    radiated_tail = np.zeros(cells)
    # --- Non-local product transport (WP-D) -----------------------------
    # Under "nonlocal" the secondary and terminal-residual banks are WITHHELD
    # from their birth cells and accumulated here, then walked after the ray
    # is done. Under "local" none of this is touched or allocated.
    walk_products = product_transport == "nonlocal"
    end_loss_low = 0.0
    end_loss_high = 0.0
    end_loss_transmitted = 0.0
    if walk_products:
        sec_flux = np.zeros(cells)  # secondary electrons born per cell [1/s]
        sec_power_eV = np.zeros(cells)  # their energy [eV/s]
        terminal_cell = -1
        terminal_flux = 0.0
        terminal_E = 0.0
    # --- QL heating locality (WP-E) --------------------------------------
    # Under "tail_walk" the anomalous drag is WITHHELD from its birth cell and
    # accumulated here [eV/s], then carried by tail electrons at E_tail after
    # the ray is done. Under "local" none of this is touched or allocated.
    end_loss_tail_low = 0.0
    end_loss_tail_high = 0.0
    # K7b exposure ledger [erg/s]: the QL tail power this ray launched, and
    # how much of it was marched outside the depth-1 band. Filled once the
    # withheld power is known, below.
    tail_power = 0.0
    tail_sub_threshold_power = 0.0
    tail_above_bar_power = 0.0
    if walk_tail:
        anom_power_eV = np.zeros(cells)

    order = range(launch, cells) if direction > 0 else range(launch, -1, -1)
    E = float(E0_eV)
    absorbed = False
    # Running flux [1/s]. Constant Gamma0 unless the anode mesh intercepts part
    # of the surviving beam at its face (audit A15); every bank below multiplies
    # by this, so the off path is bit-for-bit the historical constant-flux result.
    gamma = float(Gamma0_per_s)
    anode_intercepted = 0.0  # erg/s booked to the anode, not the plasma
    intercept_active = anode_cross_index is not None and anode_eta > 0.0

    if E <= E_stop_eV:
        # Sub-threshold source: nothing inelastic can happen; the module's
        # domain is beam energies, so pass it through untouched. Under
        # "nonlocal" that pass-through IS a transmitted primary, so the end
        # ledger books it and the identity closes here too. (Defensive path:
        # the cathode wiring only launches rays with phi_c > I_ion > E_stop.)
        sub_power = float(Gamma0_per_s) * E * _ERG_PER_EV
        if walk_products and sub_power > 0.0:
            end_loss_transmitted = sub_power
            if direction > 0:
                end_loss_high = sub_power
            else:
                end_loss_low = sub_power
        return BeamDepositionResult(
            ionization_events=ionization_events,
            excitation_events=excitation_events,
            plasma_heating_erg_s=heating,
            radiated_erg_s=radiated,
            ionization_cost_erg_s=ionization_cost,
            transmitted_flux=float(Gamma0_per_s),
            transmitted_energy_eV=E,
            anode_intercepted_erg_s=0.0,
            E_entry_eV=E_entry,
            heating_coulomb_erg_s=heat_coulomb,
            heating_anomalous_erg_s=heat_anomalous,
            heating_secondary_erg_s=heat_secondary,
            heating_terminal_erg_s=heat_terminal,
            ionization_events_tail=ion_events_tail,
            excitation_events_tail=exc_events_tail,
            ionization_cost_tail_erg_s=ion_cost_tail,
            radiated_tail_erg_s=radiated_tail,
            end_loss_low_erg_s=end_loss_low,
            end_loss_high_erg_s=end_loss_high,
            end_loss_transmitted_erg_s=end_loss_transmitted,
            # A sub-threshold source drives no anomalous drag, so neither the
            # tail ledger nor the K7b exposure ledger has anything to book on
            # this path.
            end_loss_tail_low_erg_s=end_loss_tail_low,
            end_loss_tail_high_erg_s=end_loss_tail_high,
            tail_power_erg_s=tail_power,
            tail_sub_threshold_power_erg_s=tail_sub_threshold_power,
            tail_above_bar_power_erg_s=tail_above_bar_power,
        )

    # --- Compiled CSDA march (opt-in) ------------------------------------
    # Runs the whole double loop below in C and then leaves ``order`` empty so
    # the Python march is skipped. Deliberately shaped as an INSERTION rather
    # than an ``else:`` wrapped around the loop: the Python march is the
    # equivalence target and has to stay byte-for-byte reviewable, and an
    # added indent level would touch every one of its ~170 lines.
    #
    # The four preconditions are the cases the transcription does not
    # reproduce, each routed back to Python so it behaves exactly as it always
    # has: an out-of-range ``launch`` (Python's ``range`` walks negative
    # indices or raises), ``I_ion_eV == 0`` (ZeroDivisionError), an unknown
    # coulomb model (ValueError from ``coulomb_stopping_eV_per_cm``), and a
    # beam above the excitation table's ceiling (where the lookup falls back
    # to the exact manifold sum). ``E`` only ever decreases along the march,
    # so testing the launch energy settles the ceiling for the whole ray.
    _csda_ctx = None
    if (
        _CSDA_MARCH is not None
        and 0 <= launch < cells
        and I_ion_eV != 0.0
        and coulomb_model in _COULOMB_MODEL_CODE
    ):
        _csda_ctx = _csda_tables()
        if not E < _csda_ctx.exc_top:
            _csda_ctx = None
    if _csda_ctx is not None:
        (
            E,
            gamma,
            absorbed,
            anode_intercepted,
            _terminal_cell,
            _terminal_flux,
            _terminal_E,
        ) = _CSDA_MARCH(
            _csda_ctx,
            E,
            gamma,
            nn,
            ne,
            Te,
            dz_cm,
            launch,
            direction,
            I_ion_eV,
            E_stop_eV,
            frac,
            HE_OPB_EBAR_EV,
            _COULOMB_MODEL_CODE[coulomb_model],
            anomalous_model == "quasilinear",
            area if anomalous_model == "quasilinear" else None,
            anode_cross_index if intercept_active else -1,
            anode_eta,
            walk_products,
            walk_tail,
            ionization_events,
            excitation_events,
            heating,
            radiated,
            ionization_cost,
            E_entry,
            heat_coulomb,
            heat_anomalous,
            heat_secondary,
            heat_terminal,
            sec_flux if walk_products else None,
            sec_power_eV if walk_products else None,
            anom_power_eV if walk_tail else None,
        )
        if walk_products:
            terminal_cell = _terminal_cell
            terminal_flux = _terminal_flux
            terminal_E = _terminal_E
        order = ()

    for cell in order:
        # Anode-mesh interception (A15): the ray reaches the anode face only if
        # it survived the gap (a stopped beam breaks out before this cell), so
        # removing eta of the flux HERE removes exactly the long-mfp/bypass beam.
        # Book the intercepted primaries' remaining energy to the anode and carry
        # the reduced flux downstream.
        if intercept_active and cell == anode_cross_index:
            anode_intercepted += anode_eta * gamma * E * _ERG_PER_EV
            gamma *= 1.0 - anode_eta
            intercept_active = False
        E_entry[cell] = E
        remaining = float(dz_cm[cell])
        nn_c = float(nn[cell])
        ne_c = float(ne[cell])
        Te_c = float(Te[cell])
        # --- Per-cell banks, as Python floats (cost read 2026-08-02,
        # restructure B) ------------------------------------------------
        # The substep loop below used to write each bank straight into its
        # array: eight ``arr[cell] += scalar`` fancy-index stores per substep,
        # 0.825 us of a 5.70 us substep (14.5%), against ~0.01 us for a local
        # float. They accumulate here instead and are flushed once, at cell
        # exit.
        #
        # BIT-EXACT BY CONSTRUCTION, not by tolerance: ``for cell in order``
        # visits every cell EXACTLY once (a strictly monotone range) and every
        # target array starts at 0.0, so
        #
        #     arr[cell] += x1 ; arr[cell] += x2 ; ... ; arr[cell] += xn
        #
        # and
        #
        #     acc = 0.0 ; acc += x1 ; ... ; acc += xn ; arr[cell] += acc
        #
        # are the identical sequence of float64 additions from the identical
        # starting value. The flush stays a ``+=`` rather than a ``=`` so the
        # signed-zero case lands on the historical ``0.0`` too.
        acc_ionization_events = 0.0
        acc_excitation_events = 0.0
        acc_heating = 0.0
        acc_radiated = 0.0
        acc_ionization_cost = 0.0
        acc_heat_coulomb = 0.0
        acc_heat_anomalous = 0.0
        acc_heat_secondary = 0.0
        acc_heat_terminal = 0.0
        # WP-D / WP-E withholding banks; inert unless their closure is on.
        acc_sec_flux = 0.0
        acc_sec_power_eV = 0.0
        acc_anom_power_eV = 0.0
        while remaining > 0.0:
            sigma_i = (
                He_EII_cross_lkup(E / I_ion_eV) if E > I_ion_eV else 0.0
            )
            # Table-interpolated manifold channel (see _cross docstring):
            # exact-node table, ~1e-6 relative interp error, ~100x cheaper
            # than the scalar sums this loop used to spend ~80% of total
            # step time in (2026-07-21).
            sigma_x, E_rad = He_beam_excitation_channel_lkup(E)
            W_sec = he_mean_secondary_energy_eV(E, I_ion_eV=I_ion_eV)
            # channel loss rates [eV/cm]
            L_pot = nn_c * sigma_i * I_ion_eV
            L_sec = nn_c * sigma_i * W_sec
            L_exc = nn_c * sigma_x * E_rad
            L_coul = coulomb_stopping_eV_per_cm(
                E, ne_c, Te_c, model=coulomb_model
            )
            L_anom = 0.0
            if anomalous_model == "quasilinear":
                n_b = gamma / (
                    float(area[cell]) * beam_speed_cm_s(E)
                )
                l_ql = quasilinear_relaxation_length_cm(E, ne_c, n_b)
                if math.isfinite(l_ql) and l_ql > 0.0:
                    L_anom = E / l_ql
            L_tot = L_pot + L_sec + L_exc + L_coul + L_anom
            if L_tot <= 0.0:
                break  # vacuum cell: free streaming
            dz_sub = min(remaining, frac * E / L_tot)
            # Land exactly on E_stop rather than overshooting through it.
            if E - L_tot * dz_sub <= E_stop_eV:
                dz_sub = (E - E_stop_eV) / L_tot
            if dz_sub <= 0.0:
                # E sits at E_stop to roundoff: absorb the residual here.
                if walk_products:
                    terminal_cell = cell
                    terminal_flux = gamma
                    terminal_E = E
                else:
                    acc_heating += gamma * E * _ERG_PER_EV
                    acc_heat_terminal += gamma * E * _ERG_PER_EV
                E = 0.0
                absorbed = True
                break
            # Bank each channel with the identical products the energy
            # decrement uses, so conservation closes to roundoff.
            d_pot = L_pot * dz_sub
            d_sec = L_sec * dz_sub
            d_exc = L_exc * dz_sub
            d_coul = L_coul * dz_sub
            d_anom = L_anom * dz_sub
            acc_ionization_cost += gamma * d_pot * _ERG_PER_EV
            # WP-E: under "tail_walk" the anomalous decrement is withheld from
            # every local bank in this cell (both the lumped one and the
            # diagnostic split) and accumulated for the tail walks below.
            # `d_anom_local` is `d_anom` when the walk is OFF, so the two
            # expressions below are then literally the historical ones --
            # which is what makes the default path bit-exact. The ray's own
            # energy decrement further down is UNCHANGED in both modes, so the
            # trajectory, the transmitted flux and every other channel are
            # bit-identical: only the destination of this one bank moves.
            if walk_tail:
                acc_anom_power_eV += gamma * d_anom
                d_anom_local = 0.0
            else:
                d_anom_local = d_anom
            if walk_products:
                # Withhold the secondary bank from this cell; accumulate the
                # population (flux and energy) for the walks below. The flux
                # is the SAME product `ionization_events` uses, so the
                # particle rows are untouched by construction.
                acc_heating += gamma * (d_coul + d_anom_local) * _ERG_PER_EV
                acc_sec_flux += gamma * nn_c * sigma_i * dz_sub
                acc_sec_power_eV += gamma * d_sec
            else:
                acc_heating += (
                    gamma * (d_sec + d_coul + d_anom_local) * _ERG_PER_EV
                )
                acc_heat_secondary += gamma * d_sec * _ERG_PER_EV
            acc_heat_coulomb += gamma * d_coul * _ERG_PER_EV
            acc_heat_anomalous += gamma * d_anom_local * _ERG_PER_EV
            acc_radiated += gamma * d_exc * _ERG_PER_EV
            acc_ionization_events += gamma * nn_c * sigma_i * dz_sub
            acc_excitation_events += gamma * nn_c * sigma_x * dz_sub
            E -= d_pot + d_sec + d_exc + d_coul + d_anom
            remaining -= dz_sub
            if E <= E_stop_eV:
                # Sub-threshold residual: the primary can only Coulomb-drag
                # from here; bank the remainder as local plasma heating
                # (plan B1's stated closure) and end the ray. Under "nonlocal"
                # that same residual is instead walked from this cell.
                if walk_products:
                    terminal_cell = cell
                    terminal_flux = gamma
                    terminal_E = E
                else:
                    acc_heating += gamma * E * _ERG_PER_EV
                    acc_heat_terminal += gamma * E * _ERG_PER_EV
                E = 0.0
                absorbed = True
                break
        # Flush this cell's banks (see the accumulator note above). Reached on
        # every exit from the substep loop -- the vacuum-cell break, both
        # absorption breaks, and running the cell's path out -- so no cell can
        # leave its banks unwritten.
        ionization_events[cell] += acc_ionization_events
        excitation_events[cell] += acc_excitation_events
        heating[cell] += acc_heating
        radiated[cell] += acc_radiated
        ionization_cost[cell] += acc_ionization_cost
        heat_coulomb[cell] += acc_heat_coulomb
        heat_anomalous[cell] += acc_heat_anomalous
        heat_secondary[cell] += acc_heat_secondary
        heat_terminal[cell] += acc_heat_terminal
        if walk_products:
            sec_flux[cell] += acc_sec_flux
            sec_power_eV[cell] += acc_sec_power_eV
        if walk_tail:
            anom_power_eV[cell] += acc_anom_power_eV
        if absorbed:
            break

    if walk_products or walk_tail:
        # --- Product walks (WP-D) and QL tail walks (WP-E) ---------------
        # ONE set of walk machinery serves both closures: the same per-cell
        # stopping-power coefficient taken from the module's own
        # coulomb_stopping_eV_per_cm, the same closure energy exponent, and
        # the same thermalization floor; see _walk_products_forward for the
        # closed-form integration. WP-E introduces no walk physics of its own
        # -- only a different population to walk (see the module docstring).
        q = 1.0 - _COULOMB_STOPPING_EXPONENT[coulomb_model]
        # Restructure C: the caller may have built this already and be sharing
        # it across several rays / energy groups over the same (ne, Te, model).
        coeff = (
            _coulomb_stopping_coefficient(ne, Te, coulomb_model)
            if stopping_coefficient is None
            else stopping_coefficient
        )
        floor_eV = np.maximum(
            _PRODUCT_FLOOR_TE_MULTIPLE * Te, _PRODUCT_FLOOR_MIN_EV
        )

        def _walk_and_deposit(W0, flux, walk_direction, split):
            """Walk one population; deposit it and RETURN the escaping power.

            The caller books the escape to its own end ledger -- WP-D products
            to ``end_loss_*``, WP-E tails to ``end_loss_tail_*`` -- which is
            what keeps the two ledgers independently readable.
            """
            dep_eV, exit_eV = _walk_products(
                W0, flux, walk_direction, coeff, dz_cm, floor_eV, q
            )
            dep_erg = dep_eV * _ERG_PER_EV
            heating[:] += dep_erg
            split[:] += dep_erg
            return exit_eV * _ERG_PER_EV

        def _bank_walk(W0, flux, walk_direction, split):
            """Walk one product population and book its deposit and escape."""
            nonlocal end_loss_low, end_loss_high
            exit_erg = _walk_and_deposit(W0, flux, walk_direction, split)
            if walk_direction > 0:
                end_loss_high += exit_erg
            else:
                end_loss_low += exit_erg

        if walk_products and np.any(sec_flux > 0.0):
            # Flux-weighted mean secondary energy per birth cell (the module
            # carries mean energies, not distributions -- stated limitation),
            # emitted 50/50 along +z and -z (OPB emission is broadly
            # isotropic -- stated approximation).
            W_sec_cell = np.zeros(cells)
            born = sec_flux > 0.0
            W_sec_cell[born] = sec_power_eV[born] / sec_flux[born]
            half = 0.5 * sec_flux
            for walk_direction in (1, -1):
                _bank_walk(W_sec_cell, half, walk_direction, heat_secondary)
        if walk_products and terminal_flux > 0.0 and terminal_E > 0.0:
            # The terminal residual keeps the primary's direction.
            term_flux = np.zeros(cells)
            term_W = np.zeros(cells)
            term_flux[terminal_cell] = terminal_flux
            term_W[terminal_cell] = terminal_E
            _bank_walk(term_W, term_flux, direction, heat_terminal)
        if walk_tail and np.any(anom_power_eV > 0.0):
            # WP-E: re-express each cell's withheld anomalous POWER as a flux
            # of tail electrons at the single plateau energy E_tail
            # (flux = P / E_tail, so flux*E_tail returns the power to
            # roundoff), split 50/50 along +-B, and walk them on the shared
            # machinery above. The escape goes to the tail-only ledger, and
            # the deposition profile becomes the anomalous diagnostic split --
            # heating_anomalous now reports where the QL energy LANDS.
            #
            # This self-limits exactly like the WP-D walks: once n_e is high
            # the slowing length falls below a cell, the walker thermalizes in
            # its birth cell, and the closure collapses onto the local banking
            # it replaced.
            half_flux = 0.5 * (anom_power_eV / E_tail)
            # K7b: the tail power this ray actually launched, and which band
            # it was marched in. Read-only bookkeeping over a bank that is
            # already final -- nothing below consumes it.
            tail_power = float(anom_power_eV.sum()) * _ERG_PER_EV
            if tail_sub_threshold:
                tail_sub_threshold_power = tail_power
            elif tail_above_bar:
                tail_above_bar_power = tail_power
            if ionize_tail or reflect_face is not None:
                # Both WINDOWED closures stand on the same statement: the
                # window must contain every cell the QL channel drives, or
                # that cell's tail power would be dropped on the floor.
                for birth in np.flatnonzero(half_flux > 0.0):
                    if not tail_lo <= birth <= tail_hi:
                        raise ValueError(
                            f"anomalous power in cell {int(birth)} lies "
                            f"outside tail_walk_window {(tail_lo, tail_hi)}; "
                            "the window must contain every cell the QL "
                            "channel drives, or that cell's tail power would "
                            "be silently dropped"
                        )
            if ionize_tail:
                # K6: the walkers attenuate INELASTICALLY on the column gas as
                # well as Coulomb-slowing, so the closed-form integral above
                # (which knows only the Coulomb power law) cannot carry them.
                # March them on this module's own CSDA integration instead --
                # one call per birth cell and direction, the same instrument
                # the primary uses, so the cross sections, thresholds, <W_sec>
                # convention and substep control are the primary's by
                # construction rather than by transcription.
                #
                # anomalous_model="none": the plateau electrons ARE the
                # instability's product and do not re-drive it (and a walker
                # that drove its own QL drag would double-count the very power
                # being carried). product_transport="local": the depth-1
                # truncation validated at construction -- secondaries and the
                # sub-threshold terminal residual bank where they are made.
                # No anode interception: these are born in the column, not
                # streaming out of the cathode through the mesh. No recursion
                # risk: the nested call takes anomalous_transport="local".
                #
                # The march runs on the WINDOWED domain, so the window's two
                # faces are walls: a walker that reaches one is transmitted out
                # of the sliced grid and booked to the tail end ledger, exactly
                # as it would be at a true domain end. That is what keeps every
                # birth inside cells the solver actually integrates.
                win = slice(tail_lo, tail_hi + 1)
                nn_w = nn[win]
                ne_w = ne[win]
                Te_w = Te[win]
                dz_w = dz_cm[win]

                def _bank_tail_march(res):
                    """Book one marched tail population into the shared banks.

                    Shared banks: the tail's events and energy join the
                    primary's, which is what puts the born pair on the existing
                    beam-ionization birth convention and its ``I_ion``
                    investment on the existing cost sink.
                    """
                    ionization_events[win] += res.ionization_events
                    excitation_events[win] += res.excitation_events
                    ionization_cost[win] += res.ionization_cost_erg_s
                    radiated[win] += res.radiated_erg_s
                    # All of the walker's HEAT (Coulomb drag, the local
                    # <W_sec> secondaries, the terminal residual) is the
                    # anomalous channel's delivery to the electrons, so it
                    # lands in the lumped bank and in the anomalous split
                    # -- never in the primary's coulomb/secondary/terminal
                    # splits, which keep describing the primary alone.
                    heating[win] += res.plasma_heating_erg_s
                    heat_anomalous[win] += res.plasma_heating_erg_s
                    # Diagnostic splits of the four shared banks.
                    ion_events_tail[win] += res.ionization_events
                    exc_events_tail[win] += res.excitation_events
                    ion_cost_tail[win] += res.ionization_cost_erg_s
                    radiated_tail[win] += res.radiated_erg_s

                march_kwargs = dict(
                    I_ion_eV=I_ion_eV,
                    E_stop_eV=E_stop_eV,
                    coulomb_model=coulomb_model,
                    anomalous_model="none",
                    max_energy_fraction_per_substep=frac,
                )
                for birth in np.flatnonzero(half_flux > 0.0):
                    for walk_direction in (1, -1):
                        leg_dir = walk_direction
                        leg = deposit_beam(
                            E_tail,
                            float(half_flux[birth]),
                            nn_w,
                            ne_w,
                            Te_w,
                            int(birth) - tail_lo,
                            leg_dir,
                            dz_w,
                            **march_kwargs,
                        )
                        while True:
                            _bank_tail_march(leg)
                            leg_flux = float(leg.transmitted_flux)
                            leg_E = float(leg.transmitted_energy_eV)
                            # K7: at the ONE reflecting face, a walker whose
                            # arrival energy is below the sheath threshold is
                            # turned around at the same energy and marched back
                            # from the face cell. Only that face reflects, so
                            # the reversed leg cannot come back to it and this
                            # loop runs at most twice.
                            if (
                                reflect_face is not None
                                and leg_dir == reflect_face
                                and leg_flux > 0.0
                                and leg_E < E_reflect
                            ):
                                leg_dir = -leg_dir
                                leg = deposit_beam(
                                    leg_E,
                                    leg_flux,
                                    nn_w,
                                    ne_w,
                                    Te_w,
                                    0 if reflect_face < 0 else tail_hi - tail_lo,
                                    leg_dir,
                                    dz_w,
                                    **march_kwargs,
                                )
                                continue
                            # A walker still above E_stop at the window face it
                            # was heading for escapes, on the SAME free-escape
                            # convention the energy-only walk uses. Without a
                            # reflecting face no sheath or ambipolar throttle
                            # is applied at either end, which is what makes
                            # "tail_walk" the free-escape bound it is
                            # documented as with the channel on.
                            exit_erg = leg_flux * leg_E * _ERG_PER_EV
                            if leg_dir > 0:
                                end_loss_tail_high += exit_erg
                            else:
                                end_loss_tail_low += exit_erg
                            break
            elif reflect_face is not None:
                # K7 energy-only walk with one reflecting window face. The
                # closed-form integral above carries a population over a
                # SEQUENCE of cells, so a reflection is expressed by UNFOLDING
                # the path: the reflected leg is the window traversed back the
                # other way, concatenated onto the incoming leg. The entry
                # energy of the first cell of the second leg is the identical
                # float that left the last cell of the first, so the walk still
                # telescopes exactly across the bounce and the energy ledger
                # closes at roundoff, as it does without reflection.
                win = slice(tail_lo, tail_hi + 1)
                n_w = tail_hi - tail_lo + 1
                coeff_w = coeff[win]
                dz_w = dz_cm[win]
                floor_w = floor_eV[win]
                flux_w = half_flux[win]
                W0_w = np.full(n_w, E_tail)
                # Window-local cell indices in traversal order for the arm that
                # heads INTO the reflecting face, and for the arm that heads
                # away from it (which is also the reflected leg's order).
                order_hit = (
                    np.arange(n_w)[::-1] if reflect_face < 0 else np.arange(n_w)
                )
                order_away = order_hit[::-1]
                escape_at_face = 0.0     # leaves through the reflecting face
                escape_opposite = 0.0    # leaves through the other one

                def _leg(order, W0, flux):
                    return _walk_products_forward(
                        W0[order], flux[order], coeff_w[order], dz_w[order],
                        floor_w[order], q,
                    )

                def _bank_tail_walk(dep_eV, *orders):
                    """Deposit one walked population, in ITS traversal order.

                    ``orders`` maps each block of ``dep_eV`` back onto window
                    cells: one block for a plain arm, two for the unfolded
                    reflected path. Banked per population, matching the arm-by-
                    arm conversion the unreflected walk does, so a walk in
                    which nothing reflects lands on the identical floats.
                    """
                    dep_win = np.zeros(n_w)
                    for k, order in enumerate(orders):
                        dep_win[order] += dep_eV[k * n_w:(k + 1) * n_w]
                    dep_erg = dep_win * _ERG_PER_EV
                    heating[win] += dep_erg
                    heat_anomalous[win] += dep_erg

                # The arm walking AWAY from the reflecting face never meets it.
                dep_a, exit_a, _ = _leg(order_away, W0_w, flux_w)
                _bank_tail_walk(dep_a, order_away)
                escape_opposite += exit_a
                # The arm walking INTO it: test each population's ARRIVAL
                # energy against the threshold. Populations born in different
                # cells arrive with different energies, so this is a per-birth
                # split, not a whole-arm switch.
                dep_h, exit_h, (act_h, W_face, stop_h) = _leg(
                    order_hit, W0_w, flux_w
                )
                bounced = (~stop_h) & (W_face < E_reflect)
                if not bounced.any():
                    _bank_tail_walk(dep_h, order_hit)
                    escape_at_face += exit_h
                else:
                    flux_hit = flux_w[order_hit]
                    flux_bounce = np.zeros(n_w)
                    flux_escape = np.zeros(n_w)
                    flux_bounce[act_h[bounced]] = flux_hit[act_h[bounced]]
                    flux_escape[act_h[~bounced]] = flux_hit[act_h[~bounced]]
                    if np.any(flux_escape > 0.0):
                        dep_e, exit_e, _ = _walk_products_forward(
                            W0_w[order_hit], flux_escape, coeff_w[order_hit],
                            dz_w[order_hit], floor_w[order_hit], q,
                        )
                        _bank_tail_walk(dep_e, order_hit)
                        escape_at_face += exit_e
                    # The unfolded two-leg path. The face cell appears at the
                    # end of the first leg and again at the start of the second
                    # -- the reflected walker re-crosses it, the same
                    # cell-resolution granularity the marched walk has.
                    dep_u, exit_u, _ = _walk_products_forward(
                        np.concatenate([W0_w[order_hit], np.zeros(n_w)]),
                        np.concatenate([flux_bounce, np.zeros(n_w)]),
                        np.concatenate([coeff_w[order_hit], coeff_w[order_away]]),
                        np.concatenate([dz_w[order_hit], dz_w[order_away]]),
                        np.concatenate([floor_w[order_hit], floor_w[order_away]]),
                        q,
                    )
                    _bank_tail_walk(dep_u, order_hit, order_away)
                    escape_opposite += exit_u
                if reflect_face > 0:
                    end_loss_tail_high += escape_at_face * _ERG_PER_EV
                    end_loss_tail_low += escape_opposite * _ERG_PER_EV
                else:
                    end_loss_tail_low += escape_at_face * _ERG_PER_EV
                    end_loss_tail_high += escape_opposite * _ERG_PER_EV
            else:
                tail_W = np.full(cells, E_tail)
                for walk_direction in (1, -1):
                    exit_erg = _walk_and_deposit(
                        tail_W, half_flux, walk_direction, heat_anomalous
                    )
                    if walk_direction > 0:
                        end_loss_tail_high += exit_erg
                    else:
                        end_loss_tail_low += exit_erg
        if walk_products and not absorbed and gamma > 0.0 and E > 0.0:
            # The transmitted primary: computed since B1, never banked. It
            # leaves through the end the ray was heading for.
            end_loss_transmitted = gamma * E * _ERG_PER_EV
            if direction > 0:
                end_loss_high += end_loss_transmitted
            else:
                end_loss_low += end_loss_transmitted

    return BeamDepositionResult(
        ionization_events=ionization_events,
        excitation_events=excitation_events,
        plasma_heating_erg_s=heating,
        radiated_erg_s=radiated,
        ionization_cost_erg_s=ionization_cost,
        transmitted_flux=0.0 if absorbed else gamma,
        transmitted_energy_eV=0.0 if absorbed else E,
        anode_intercepted_erg_s=anode_intercepted,
        E_entry_eV=E_entry,
        heating_coulomb_erg_s=heat_coulomb,
        heating_anomalous_erg_s=heat_anomalous,
        heating_secondary_erg_s=heat_secondary,
        heating_terminal_erg_s=heat_terminal,
        ionization_events_tail=ion_events_tail,
        excitation_events_tail=exc_events_tail,
        ionization_cost_tail_erg_s=ion_cost_tail,
        radiated_tail_erg_s=radiated_tail,
        end_loss_low_erg_s=end_loss_low,
        end_loss_high_erg_s=end_loss_high,
        end_loss_transmitted_erg_s=end_loss_transmitted,
        end_loss_tail_low_erg_s=end_loss_tail_low,
        end_loss_tail_high_erg_s=end_loss_tail_high,
        tail_power_erg_s=tail_power,
        tail_sub_threshold_power_erg_s=tail_sub_threshold_power,
        tail_above_bar_power_erg_s=tail_above_bar_power,
    )


def _empty_deposition(cells, transmitted_flux=0.0, transmitted_energy_eV=0.0):
    """Return an all-zero :class:`BeamDepositionResult` of width ``cells``."""
    return BeamDepositionResult(
        ionization_events=np.zeros(cells),
        excitation_events=np.zeros(cells),
        plasma_heating_erg_s=np.zeros(cells),
        radiated_erg_s=np.zeros(cells),
        ionization_cost_erg_s=np.zeros(cells),
        transmitted_flux=float(transmitted_flux),
        transmitted_energy_eV=float(transmitted_energy_eV),
        anode_intercepted_erg_s=0.0,
        E_entry_eV=np.zeros(cells),
        heating_coulomb_erg_s=np.zeros(cells),
        heating_anomalous_erg_s=np.zeros(cells),
        heating_secondary_erg_s=np.zeros(cells),
        heating_terminal_erg_s=np.zeros(cells),
        ionization_events_tail=np.zeros(cells),
        excitation_events_tail=np.zeros(cells),
        ionization_cost_tail_erg_s=np.zeros(cells),
        radiated_tail_erg_s=np.zeros(cells),
    )


def deposit_beam_two_stream(
    E0_eV: float,
    Gamma0_per_s: float,
    *,
    f_cov: np.ndarray,
    nn_channel: np.ndarray,
    ne_channel: np.ndarray,
    nn_reservoir: np.ndarray,
    ne_reservoir: np.ndarray,
    Te: np.ndarray,
    launch: int,
    direction: int,
    dz_cm: np.ndarray,
    I_ion_eV: float = HE_I_ION_EV,
    E_stop_eV: float = HE_E_STOP_EV,
    coulomb_model: str = "fast_electron",
    anomalous_model: str = "none",
    beam_area_cm2: np.ndarray | float | None = None,
    max_energy_fraction_per_substep: float = 0.02,
    anode_cross_index: int | None = None,
    anode_eta: float = 0.0,
    product_transport: str = "local",
    anomalous_transport: str = "local",
    tail_energy_eV: float | None = None,
    tail_walk_window: tuple[int, int] | None = None,
    tail_ionization: str = "off",
    tail_reflect_face: int | None = None,
    tail_reflect_threshold_eV: float | None = None,
    stopping_coefficient: np.ndarray | None = None,
    nn_mean: np.ndarray | None = None,
    ne_mean: np.ndarray | None = None,
):
    """March one beam ray through a Z-RESOLVED two-medium column (coverage v2).

    Returns ``(channel, reservoir, flux_entry_per_s)``: one
    :class:`BeamDepositionResult` per medium -- their per-cell banks ADD to the
    ray's total, which is what the caller consumes -- and the per-cell total
    primary flux ENTERING each cell [1/s] (zero for cells the ray never
    reaches), which is the ray's own probe-independent record of where its flux
    was lost.

    **What this is.** ``deposit_beam`` marches ONE stream through ONE medium.
    The coverage closure's v1.1 split ran it twice, partitioning the emitted
    flux once at the cathode face and letting each arm march its own medium to
    the end of the column. That is exact only while the covered fraction is
    uniform in z. With ``f_cov = f_cov(z)`` the two media exchange
    cross-sectional territory as the ray advances, so the partition has to be
    re-made at every cell:

    * enter cell ``k`` with a total surviving primary flux ``Gamma`` at mean
      energy ``E``;
    * re-split by the LOCAL coverage -- ``f_cov[k]*Gamma`` into the channel
      medium (``ne_channel``, ``nn_channel``) and ``(1-f_cov[k])*Gamma`` into
      the reservoir medium (``ne_reservoir``, ``nn_reservoir``);
    * march each arm across that cell's path length with its own substepping,
      banking into that arm's own per-cell arrays;
    * re-mix at cell exit: ``Gamma' = gamma_c' + gamma_r'`` and
      ``E' = (gamma_c'*E_c' + gamma_r'*E_r')/Gamma'``.

    **The stated approximation.** Re-mixing at every cell is the statement that
    the breakdown patches DECORRELATE AXIALLY ON THE CELL SCALE: an electron
    that crossed cell ``k`` inside a patch has no memory of that when it enters
    cell ``k+1``, so the population is re-randomised over the local
    cross-section there. The re-mix is a ONE-GROUP closure on that population --
    it carries the flux-weighted mean energy forward rather than two separate
    energies -- and it conserves both flux and power identically at every
    re-partition boundary (``Gamma'*E'`` is by construction
    ``gamma_c'*E_c' + gamma_r'*E_r'``), so nothing is created or lost by the
    re-partition itself. What it discards is the SPREAD of the primary energy
    distribution, which the CSDA model does not carry in the first place.

    **Uniform ``f_cov`` does NOT reproduce v1.1.** Re-mixing at every cell is a
    different model from partitioning once at emission, even when the profile
    is flat: v1.1's channel arm keeps whatever energy the channel medium left
    it, while here both arms are pulled back to the common mean at every cell
    face. Only ``f_cov == 1`` everywhere reduces exactly -- and then to the
    SHIPPED single-medium model, which the caller obtains by calling
    ``deposit_beam`` directly rather than by entering this function.

    **The quasilinear beam density is unchanged by the split.** Each arm
    carries ``f`` (or ``1-f``) of the flux over ``f`` (or ``1-f``) of the area,
    so ``n_b = Gamma/(A*v)`` in both and equal to the mean-field value.
    ``beam_area_cm2`` is therefore the FULL column area, exactly as it is for a
    mean-field ray; the arm areas are formed here.

    **The walk closures run ON THE MEAN STATE.** ``product_transport="nonlocal"``
    (WP-D) and ``anomalous_transport="tail_walk"`` (WP-E) withhold banks during
    the march and walk them afterwards. Both arms' per-cell withheld banks --
    which are per-arm per-cell by construction, so birth LOCATIONS are the
    march's own -- feed the ONE post-march walk stage, and that stage runs on
    the mean plasma state: neither the channel view ``n/f_cov`` nor the
    reservoir floor.

    That is the same patch-decorrelation closure the re-mix above stands on,
    applied to a product instead of a primary. A field-aligned product's path
    samples the channel medium with probability ``f_cov(z)`` per cell and the
    reservoir with ``1 - f_cov(z)``, so the stopping it sees on average is
    ``f_cov*(n/f_cov) + (1-f_cov)*n_floor``, i.e. the MEAN density: the
    concentration cancels exactly the way it cancels in a volumetric bilinear
    rate. The free-stream-to-walk transition is therefore EMERGENT rather than
    imposed -- the walk's reach is set by the same density-dependent Coulomb
    blocking the primary's drag law uses, so at pedestal densities a walker
    crosses the machine (it IS a free-streamer there) and localizes as the mean
    builds. No constant and no keyword is introduced for it.

    Two second-order misattributions are accepted, in opposite directions and
    both bounded by the patch-scale argument above:

    * a product born INSIDE a channel is briefly correlated with that channel,
      whose density is higher than the mean, so it thermalizes somewhat more
      locally than this mean-field walk predicts;
    * a product born in the RESERVOIR truly sees the floor density, so its reach
      is longer than the mean walk gives it. This one vanishes at both ends of
      the closure's own life: early, when the mean IS near the floor and the two
      media barely differ, and late, when ``1 - f_cov -> 0`` and there is no
      reservoir birth left to misplace.

    ``stopping_coefficient`` is REQUIRED once either walk is active and is the
    mean-state coefficient (see ``deposit_beam``'s hoisting note). It is not
    defaulted here: this function holds two media and no single ``(ne, Te)``
    pair it could honestly build one from, so a caller that has not decided
    which state the walk runs on gets an error rather than a quiet choice.

    **Burn attribution under ``tail_ionization="on"`` is THE SAME PARTITION.**
    An ionizing walker also removes neutrals, and the medium it removed them
    from follows from the identical statement: at cell ``z`` the walker's path
    lies inside a channel for a fraction ``f_cov(z)`` of its cross-section and
    inside the reservoir for ``1 - f_cov(z)``, so its per-cell ionization
    EVENTS are split that way -- ``f_cov`` of them debit the covered column and
    ``1 - f_cov`` the reservoir. That split is expressed by banking the walker's
    events into the two ARMS with those weights, which is why it needs no
    machinery of its own: the caller already reads the reservoir arm's
    ionization rows as the reservoir debit, and the sum of the arms as the
    total. The walk itself still runs on the mean medium (``nn_mean``,
    ``ne_mean``, required with the channel on), the births are still booked to
    the mean fields, and the two misattribution bounds above carry over
    unchanged -- they are statements about the same sampling argument.

    The walked HEAT is booked to the CHANNEL arm's banks, on the same
    convention the transmitted primary already uses above: after the walk the
    energy belongs to the mean field and no medium owns it, and the caller
    consumes the two arms' sum. Splitting the ionization cost and the radiated
    energy by the same weights therefore leaves the energy side exactly where
    it shipped -- only the sum is read.
    """
    if anomalous_model not in ("none", "quasilinear"):
        raise ValueError(
            f"unknown anomalous_model {anomalous_model!r}; "
            "expected 'none' or 'quasilinear'"
        )
    if anode_eta != 0.0 and not (0.0 <= anode_eta < 1.0):
        raise ValueError(f"anode_eta must be in [0, 1) (got {anode_eta})")
    if direction not in (-1, 1):
        raise ValueError(f"direction must be +1 or -1 (got {direction})")
    dz_cm = np.asarray(dz_cm, dtype=float)
    cells = dz_cm.size
    f_cov = np.asarray(f_cov, dtype=float)
    media = tuple(
        np.asarray(a, dtype=float)
        for a in (nn_channel, ne_channel, nn_reservoir, ne_reservoir, Te)
    )
    if f_cov.shape != (cells,) or any(a.shape != (cells,) for a in media):
        raise ValueError(
            "f_cov, nn_channel, ne_channel, nn_reservoir, ne_reservoir, Te "
            f"and dz_cm must share one shape (cells,) = {(cells,)}"
        )
    if not np.all(np.isfinite(f_cov)) or np.any(f_cov <= 0.0) or np.any(
        f_cov > 1.0
    ):
        raise ValueError(
            "f_cov must be finite and in (0, 1] per cell (got min "
            f"{float(np.min(f_cov)):.6g}, max {float(np.max(f_cov)):.6g})"
        )
    nn_ch, ne_ch, nn_res, ne_res, Te = media
    if anode_cross_index is not None:
        anode_cross_index = int(anode_cross_index)
        if not 0 <= anode_cross_index < cells:
            raise ValueError(
                "anode_cross_index must index a cell in [0, cells) "
                f"(got {anode_cross_index}, cells={cells})"
            )
    area = None
    if anomalous_model == "quasilinear":
        if beam_area_cm2 is None:
            raise ValueError("anomalous_model='quasilinear' needs beam_area_cm2")
        area = np.broadcast_to(
            np.asarray(beam_area_cm2, dtype=float), (cells,)
        )
    frac = float(max_energy_fraction_per_substep)
    if not 0.0 < frac < 1.0:
        raise ValueError(
            "max_energy_fraction_per_substep must be in (0, 1), got "
            f"{max_energy_fraction_per_substep}"
        )
    if product_transport not in ("local", "nonlocal"):
        raise ValueError(
            f"unknown product_transport {product_transport!r}; "
            "expected 'local' or 'nonlocal'"
        )
    if anomalous_transport not in ("local", "tail_walk"):
        raise ValueError(
            f"unknown anomalous_transport {anomalous_transport!r}; "
            "expected 'local' or 'tail_walk'"
        )
    if tail_ionization not in ("off", "on"):
        raise ValueError(
            f"unknown tail_ionization {tail_ionization!r}; "
            "expected 'off' or 'on'"
        )
    if tail_ionization == "on" and anomalous_transport != "tail_walk":
        raise ValueError(
            "tail_ionization='on' requires anomalous_transport='tail_walk' "
            "(the ionizing channel belongs to the QL tail walkers; with "
            "anomalous_transport='local' there are no walkers and the "
            "setting would do nothing). anomalous_transport accepts 'local' "
            "or 'tail_walk'; tail_ionization accepts 'off' or 'on'"
        )
    walk_products = product_transport == "nonlocal"
    walk_tail = anomalous_transport == "tail_walk"
    ionize_tail = tail_ionization == "on"
    tail_sub_threshold = False
    tail_above_bar = False
    E_tail = 0.0
    if walk_tail:
        # Same presence gating deposit_beam applies: with no anomalous channel
        # there is no power for the walk to carry and the setting is a no-op.
        if anomalous_model == "none":
            raise ValueError(
                "anomalous_transport='tail_walk' requires an active anomalous "
                "channel (anomalous_model='quasilinear'); with no anomalous "
                "drag there is no power to carry and the setting would do "
                "nothing"
            )
        if tail_energy_eV is None:
            raise ValueError(
                "anomalous_transport='tail_walk' needs tail_energy_eV (the "
                "QL plateau energy the tail electrons are launched at)"
            )
        E_tail = float(tail_energy_eV)
        if not math.isfinite(E_tail) or E_tail <= 0.0:
            raise ValueError(
                f"tail_energy_eV must be finite and > 0 (got {tail_energy_eV})"
            )
    reflect_face = None
    E_reflect = 0.0
    if tail_reflect_face is not None:
        if not walk_tail:
            raise ValueError(
                "tail_reflect_face requires anomalous_transport='tail_walk' "
                "(it reflects the QL tail walkers; with no walk there is "
                "nothing to reflect and the setting would do nothing)"
            )
        reflect_face = int(tail_reflect_face)
        if reflect_face not in (-1, 1):
            raise ValueError(
                "tail_reflect_face must be -1 (the walk window's low-index "
                f"face) or +1 (its high-index face), got {tail_reflect_face!r}"
            )
        if tail_reflect_threshold_eV is None:
            raise ValueError(
                "tail_reflect_face needs tail_reflect_threshold_eV (the "
                "energy below which a walker arriving at that face is turned "
                "around instead of escaping)"
            )
        E_reflect = float(tail_reflect_threshold_eV)
        if not math.isfinite(E_reflect) or E_reflect <= 0.0:
            raise ValueError(
                "tail_reflect_threshold_eV must be finite and > 0 (got "
                f"{tail_reflect_threshold_eV})"
            )
    elif tail_reflect_threshold_eV is not None:
        raise ValueError(
            "tail_reflect_threshold_eV was given without tail_reflect_face; "
            "the threshold belongs to a named reflecting face and on its own "
            "would silently do nothing"
        )
    tail_lo, tail_hi = 0, cells - 1
    if reflect_face is not None and not ionize_tail:
        if tail_walk_window is None:
            raise ValueError(
                "tail_reflect_face needs tail_walk_window=(lo, hi): the "
                "reflecting face is one of that window's two faces, and "
                "without it the walk has no face to reflect at"
            )
        tail_lo, tail_hi = (int(tail_walk_window[0]), int(tail_walk_window[1]))
        if not 0 <= tail_lo <= tail_hi < cells:
            raise ValueError(
                "tail_walk_window must be an inclusive (lo, hi) cell range "
                f"with 0 <= lo <= hi < cells={cells} (got {tail_walk_window})"
            )
    if ionize_tail:
        # The walk DOMAIN is required, not defaulted to the whole grid: an
        # ionizing walker that leaves the plasma births pairs into rows the
        # solver's active mask zeroes, i.e. pairs created and silently deleted.
        if tail_walk_window is None:
            raise ValueError(
                "tail_ionization='on' needs tail_walk_window=(lo, hi), the "
                "inclusive cell range the tail walkers may traverse (the "
                "plasma-active window: a walker leaving it hits a wall and is "
                "booked to the tail end ledger). Without it the walk would "
                "run off into cells whose plasma rows the solver zeroes and "
                "birth pairs that are then deleted"
            )
        tail_lo, tail_hi = (int(tail_walk_window[0]), int(tail_walk_window[1]))
        if not 0 <= tail_lo <= tail_hi < cells:
            raise ValueError(
                "tail_walk_window must be an inclusive (lo, hi) cell range "
                f"with 0 <= lo <= hi < cells={cells} (got {tail_walk_window})"
            )
        # THE MEAN MEDIUM the walkers march through. Required for the same
        # reason the stopping coefficient is: this march holds a channel view
        # and a reservoir view, and the ruling says the walk runs on neither.
        if nn_mean is None or ne_mean is None:
            raise ValueError(
                "tail_ionization='on' under a two-stream march needs nn_mean "
                "and ne_mean: the ionizing walkers march the MEAN state, and "
                "this march carries only the channel and reservoir views, so "
                "it cannot form the mean without silently choosing a medium"
            )
        nn_mean = np.asarray(nn_mean, dtype=float)
        ne_mean = np.asarray(ne_mean, dtype=float)
        if nn_mean.shape != (cells,) or ne_mean.shape != (cells,):
            raise ValueError(
                "nn_mean and ne_mean must have one entry per grid cell "
                f"(cells={cells}); got {nn_mean.shape} and {ne_mean.shape}"
            )
        # --- The K7b band split, transcribed from deposit_beam --------------
        # The two depth-1 bars select a TREATMENT rather than refusing; only
        # the EII table edge is still a refusal, and it is edge-INCLUSIVE
        # within HE_EII_EDGE_REL_TOL (at the edge the clamped cross section IS
        # the table's endpoint value, so nothing is extrapolated).
        _E_table_top = HE_EII_EPS_TOP * I_ion_eV
        _edge_excess = (E_tail - _E_table_top) / _E_table_top
        if _edge_excess > HE_EII_EDGE_REL_TOL:
            raise ValueError(
                "tail_ionization='on' marches the walkers on the tabulated "
                "He EII cross section, which ends at eps = E/I_ion = "
                f"{HE_EII_EPS_TOP:.6f} (i.e. "
                f"{_E_table_top:.2f} eV at I_ion_eV={I_ion_eV}); "
                f"at tail_energy_eV={E_tail} eV the lookup would clamp to its "
                "last node and the walk would attenuate on an extrapolated "
                "cross section. This is refused, not approximated (relative "
                f"excess {_edge_excess:.3e}, tolerated "
                f"{HE_EII_EDGE_REL_TOL:.1e})"
            )
        if E_tail <= E_stop_eV:
            # SUB-THRESHOLD: no He inelastic channel is open, so zero
            # ionization is the answer, not a modelling choice. Revert this ray
            # to the energy-only walk by clearing the flag.
            tail_sub_threshold = True
            ionize_tail = False
        elif (
            he_mean_secondary_energy_eV(E_tail, I_ion_eV=I_ion_eV)
            >= E_stop_eV
        ):
            # ABOVE THE DEPTH-1 BAR: the mean secondary can itself do something
            # inelastic, so banking it locally understates the cascade.
            # Allowed, with the understatement measured and the exposure
            # reported.
            tail_above_bar = True
    if walk_products or walk_tail:
        # THE MEAN-STATE HAND-OFF, made structural. There is no medium here the
        # walk could fall back on: this function holds a channel view and a
        # reservoir view and the closure says the walk runs on NEITHER, so the
        # coefficient is required rather than built.
        if stopping_coefficient is None:
            raise ValueError(
                "the walk closures under a two-stream march need an explicit "
                "stopping_coefficient: the walk runs on the MEAN plasma state, "
                "and this march carries two media (channel n/f_cov and "
                "reservoir floor) but not the mean, so it cannot build one "
                "without silently choosing a medium. Pass the caller's "
                "mean-state _coulomb_stopping_coefficient(ne_mean, Te, model)"
            )
        stopping_coefficient = np.asarray(stopping_coefficient, dtype=float)
        if stopping_coefficient.shape != (cells,):
            raise ValueError(
                "stopping_coefficient must have one entry per grid cell "
                f"(cells={cells}); got shape {stopping_coefficient.shape}"
            )

    # Per-arm banks. Index 0 is the CHANNEL arm, 1 the RESERVOIR arm; they are
    # never summed here, so the caller keeps the reservoir arm's ionization
    # separable for the closure's deficit equation.
    banks = tuple(
        {
            name: np.zeros(cells)
            for name in (
                "ionization_events",
                "excitation_events",
                "heating",
                "radiated",
                "ionization_cost",
                "heat_coulomb",
                "heat_anomalous",
                "heat_secondary",
                "heat_terminal",
                "E_entry",
                # K6 diagnostic splits. Present (and zero) on both arms
                # whatever the closure, so the result shape never depends on
                # which branch ran.
                "ion_events_tail",
                "exc_events_tail",
                "ion_cost_tail",
                "radiated_tail",
            )
        }
        for _ in range(2)
    )
    flux_entry = np.zeros(cells)
    anode_intercepted = [0.0, 0.0]
    # WP-D / WP-E withholding banks, SHARED across the arms and indexed by
    # BIRTH CELL. Both arms withhold into the same per-cell slot because the
    # one walk stage below runs on the mean state for both, so a per-arm split
    # would only be summed again before it was used; the birth LOCATION, which
    # is the thing the walk consumes, is the march's own either way. The energy
    # each slot carries is a flux-weighted mean, exactly the convention
    # deposit_beam uses across the substeps within one cell.
    sec_flux = np.zeros(cells) if walk_products else None
    sec_power_eV = np.zeros(cells) if walk_products else None
    term_flux = np.zeros(cells) if walk_products else None
    term_power_eV = np.zeros(cells) if walk_products else None
    anom_power_eV = np.zeros(cells) if walk_tail else None
    end_loss_low = 0.0
    end_loss_high = 0.0
    end_loss_transmitted = 0.0
    end_loss_tail_low = 0.0
    end_loss_tail_high = 0.0
    tail_power = 0.0
    tail_sub_threshold_power = 0.0
    tail_above_bar_power = 0.0

    E = float(E0_eV)
    gamma_total = float(Gamma0_per_s)
    if E <= E_stop_eV:
        # Sub-threshold source: nothing inelastic can happen, so the ray passes
        # through untouched and the whole flux is transmitted. Mirrors
        # deposit_beam's own early return; the coverage wiring never launches
        # such a ray (phi_c > I_ion > E_stop).
        chan = _empty_deposition(cells, gamma_total, E)
        return chan, _empty_deposition(cells), flux_entry

    order = range(launch, cells) if direction > 0 else range(launch, -1, -1)
    intercept_active = anode_cross_index is not None and anode_eta > 0.0
    absorbed = False
    for cell in order:
        # Anode-mesh interception (A15). The mesh is a GEOMETRIC obstruction
        # across the whole cross-section, so it removes the same solid fraction
        # from both media; applying it to the mixed flux before the re-split is
        # therefore equivalent to applying it per arm, and books the
        # intercepted energy on the arm shares the cell is about to use.
        f = float(f_cov[cell])
        # Recorded BEFORE the mesh takes its share: ``flux_entry`` is what
        # ARRIVED at this cell, the flux counterpart of ``E_entry`` (which the
        # interception cannot move, since it removes primaries rather than
        # slowing them). A caller measuring where the ray lost flux to the
        # PLASMA -- the gap-survival ledger -- must not see a solid obstruction
        # counted as stopping.
        flux_entry[cell] = gamma_total
        if intercept_active and cell == anode_cross_index:
            lost = anode_eta * gamma_total * E * _ERG_PER_EV
            anode_intercepted[0] += f * lost
            anode_intercepted[1] += (1.0 - f) * lost
            gamma_total *= 1.0 - anode_eta
            intercept_active = False
        # The local re-split, by AREA. At f == 1 the reservoir arm has no
        # cross-section at all and is skipped entirely, so a fully-covered cell
        # costs exactly what a single-medium cell costs.
        arms = [(0, f, float(nn_ch[cell]), float(ne_ch[cell]))]
        if f < 1.0:
            arms.append(
                (1, 1.0 - f, float(nn_res[cell]), float(ne_res[cell]))
            )
        Te_c = float(Te[cell])
        dz_cell = float(dz_cm[cell])
        # n_b is formed on the TOTAL flux over the FULL area: each arm carries
        # its share of the flux over the matching share of the area, so the two
        # factors cancel and both arms see the mean-field beam density.
        area_c = 0.0 if area is None else float(area[cell])
        surviving_weight = 0.0
        surviving_energy_weight = 0.0
        for arm, weight, nn_c, ne_c in arms:
            bank = banks[arm]
            gamma = weight * gamma_total
            bank["E_entry"][cell] = E
            E_arm = E
            remaining = dz_cell
            arm_absorbed = False
            acc_ionization_events = 0.0
            acc_excitation_events = 0.0
            acc_heating = 0.0
            acc_radiated = 0.0
            acc_ionization_cost = 0.0
            acc_heat_coulomb = 0.0
            acc_heat_anomalous = 0.0
            acc_heat_secondary = 0.0
            acc_heat_terminal = 0.0
            # This arm's withheld populations in this cell; inert unless their
            # closure is on.
            acc_sec_flux = 0.0
            acc_sec_power_eV = 0.0
            acc_anom_power_eV = 0.0
            while remaining > 0.0:
                sigma_i = (
                    He_EII_cross_lkup(E_arm / I_ion_eV)
                    if E_arm > I_ion_eV
                    else 0.0
                )
                sigma_x, E_rad = He_beam_excitation_channel_lkup(E_arm)
                W_sec = he_mean_secondary_energy_eV(E_arm, I_ion_eV=I_ion_eV)
                L_pot = nn_c * sigma_i * I_ion_eV
                L_sec = nn_c * sigma_i * W_sec
                L_exc = nn_c * sigma_x * E_rad
                L_coul = coulomb_stopping_eV_per_cm(
                    E_arm, ne_c, Te_c, model=coulomb_model
                )
                L_anom = 0.0
                if anomalous_model == "quasilinear":
                    n_b = gamma_total / (area_c * beam_speed_cm_s(E_arm))
                    l_ql = quasilinear_relaxation_length_cm(E_arm, ne_c, n_b)
                    if math.isfinite(l_ql) and l_ql > 0.0:
                        L_anom = E_arm / l_ql
                L_tot = L_pot + L_sec + L_exc + L_coul + L_anom
                if L_tot <= 0.0:
                    break  # vacuum cell: free streaming
                dz_sub = min(remaining, frac * E_arm / L_tot)
                if E_arm - L_tot * dz_sub <= E_stop_eV:
                    dz_sub = (E_arm - E_stop_eV) / L_tot
                if dz_sub <= 0.0:
                    if walk_products:
                        term_flux[cell] += gamma
                        term_power_eV[cell] += gamma * E_arm
                    else:
                        acc_heating += gamma * E_arm * _ERG_PER_EV
                        acc_heat_terminal += gamma * E_arm * _ERG_PER_EV
                    E_arm = 0.0
                    arm_absorbed = True
                    break
                d_pot = L_pot * dz_sub
                d_sec = L_sec * dz_sub
                d_exc = L_exc * dz_sub
                d_coul = L_coul * dz_sub
                d_anom = L_anom * dz_sub
                acc_ionization_cost += gamma * d_pot * _ERG_PER_EV
                # WP-E: under the tail walk the anomalous decrement is withheld
                # from every local bank in this cell and carried to the walk
                # stage. ``d_anom_local`` is ``d_anom`` with the walk off, so
                # the expressions below are then literally the ones this march
                # shipped with. The ARM's own energy decrement further down is
                # unchanged in both modes, so the trajectory, the re-mix and
                # every other channel are untouched: only this bank's
                # destination moves.
                if walk_tail:
                    acc_anom_power_eV += gamma * d_anom
                    d_anom_local = 0.0
                else:
                    d_anom_local = d_anom
                if walk_products:
                    acc_heating += (
                        gamma * (d_coul + d_anom_local) * _ERG_PER_EV
                    )
                    acc_sec_flux += gamma * nn_c * sigma_i * dz_sub
                    acc_sec_power_eV += gamma * d_sec
                else:
                    acc_heating += (
                        gamma * (d_sec + d_coul + d_anom_local) * _ERG_PER_EV
                    )
                    acc_heat_secondary += gamma * d_sec * _ERG_PER_EV
                acc_heat_coulomb += gamma * d_coul * _ERG_PER_EV
                acc_heat_anomalous += gamma * d_anom_local * _ERG_PER_EV
                acc_radiated += gamma * d_exc * _ERG_PER_EV
                acc_ionization_events += gamma * nn_c * sigma_i * dz_sub
                acc_excitation_events += gamma * nn_c * sigma_x * dz_sub
                E_arm -= d_pot + d_sec + d_exc + d_coul + d_anom
                remaining -= dz_sub
                if E_arm <= E_stop_eV:
                    if walk_products:
                        term_flux[cell] += gamma
                        term_power_eV[cell] += gamma * E_arm
                    else:
                        acc_heating += gamma * E_arm * _ERG_PER_EV
                        acc_heat_terminal += gamma * E_arm * _ERG_PER_EV
                    E_arm = 0.0
                    arm_absorbed = True
                    break
            bank["ionization_events"][cell] += acc_ionization_events
            bank["excitation_events"][cell] += acc_excitation_events
            bank["heating"][cell] += acc_heating
            bank["radiated"][cell] += acc_radiated
            bank["ionization_cost"][cell] += acc_ionization_cost
            bank["heat_coulomb"][cell] += acc_heat_coulomb
            bank["heat_anomalous"][cell] += acc_heat_anomalous
            bank["heat_secondary"][cell] += acc_heat_secondary
            bank["heat_terminal"][cell] += acc_heat_terminal
            if walk_products:
                sec_flux[cell] += acc_sec_flux
                sec_power_eV[cell] += acc_sec_power_eV
            if walk_tail:
                anom_power_eV[cell] += acc_anom_power_eV
            if not arm_absorbed:
                surviving_weight += weight
                surviving_energy_weight += weight * E_arm
        # --- Re-mix -----------------------------------------------------
        # Flux and power both carry across: the surviving total is the entering
        # total times the surviving AREA share, and the common energy is the
        # area-weighted mean over the surviving arms. Weighting by area rather
        # than by flux is the same weighting -- each arm's flux is its area
        # share times the entering total -- and it stays well defined on a
        # zero-flux ray, which is a legitimate energy-only trajectory (the
        # flux-independent probe). An arm that stopped inside this cell
        # contributes to neither sum and has already banked its residual as
        # terminal heating there.
        if surviving_weight > 0.0:
            gamma_total = gamma_total * surviving_weight
            E = surviving_energy_weight / surviving_weight
        else:
            gamma_total = 0.0
            E = 0.0
            absorbed = True
            break

    if walk_products or walk_tail:
        # --- THE ONE POST-MARCH WALK STAGE, ON THE MEAN STATE -------------
        # The same closed-form machinery deposit_beam's walks use, on the same
        # closure exponent and the same thermalization floor, fed by the shared
        # per-cell birth banks above. What is coverage-specific is the MEDIUM:
        # ``coeff`` is the caller's mean-state coefficient (required, validated
        # above), never the channel view -- see this function's docstring for
        # why the concentration cancels in a decorrelated walker's path.
        # ``floor_eV`` needs Te alone, which the two media share.
        q = 1.0 - _COULOMB_STOPPING_EXPONENT[coulomb_model]
        coeff = stopping_coefficient
        floor_eV = np.maximum(
            _PRODUCT_FLOOR_TE_MULTIPLE * Te, _PRODUCT_FLOOR_MIN_EV
        )
        chan = banks[0]

        def _walk_and_deposit(W0, flux, walk_direction, split):
            """Walk one population; deposit it and RETURN the escaping power.

            The deposit lands in the CHANNEL arm's banks: the walked energy is
            the mean field's, not either medium's, and the caller consumes the
            two arms' sum (see the docstring's booking note).
            """
            dep_eV, exit_eV = _walk_products(
                W0, flux, walk_direction, coeff, dz_cm, floor_eV, q
            )
            dep_erg = dep_eV * _ERG_PER_EV
            chan["heating"] += dep_erg
            chan[split] += dep_erg
            return exit_eV * _ERG_PER_EV

        def _bank_walk(W0, flux, walk_direction, split):
            """Walk one product population and book its deposit and escape."""
            nonlocal end_loss_low, end_loss_high
            exit_erg = _walk_and_deposit(W0, flux, walk_direction, split)
            if walk_direction > 0:
                end_loss_high += exit_erg
            else:
                end_loss_low += exit_erg

        if walk_products and np.any(sec_flux > 0.0):
            # Flux-weighted mean secondary energy per BIRTH cell, over both
            # arms' contributions to that cell, emitted 50/50 along +-z.
            W_sec_cell = np.zeros(cells)
            born = sec_flux > 0.0
            W_sec_cell[born] = sec_power_eV[born] / sec_flux[born]
            half = 0.5 * sec_flux
            for walk_direction in (1, -1):
                _bank_walk(W_sec_cell, half, walk_direction, "heat_secondary")
        if walk_products and np.any(term_flux > 0.0):
            # The terminal residual keeps the PRIMARY's direction. Unlike the
            # single-medium march there can be several terminal cells: each arm
            # runs out of energy where its own medium stops it, and an arm that
            # stops leaves the survivor marching on. Carried per cell at the
            # flux-weighted mean residual, the same convention as above.
            term_W = np.zeros(cells)
            stopped = term_flux > 0.0
            term_W[stopped] = term_power_eV[stopped] / term_flux[stopped]
            _bank_walk(term_W, term_flux, direction, "heat_terminal")
        if walk_tail and np.any(anom_power_eV > 0.0):
            # WP-E: re-express each cell's withheld anomalous POWER as a flux of
            # tail electrons at the single plateau energy E_tail
            # (flux*E_tail returns the power to roundoff), split 50/50 along
            # +-B, and walk them. The deposition profile becomes the anomalous
            # diagnostic split: heating_anomalous reports where the QL energy
            # LANDS. This self-limits on the mean density -- machine-length at
            # pedestal densities, collapsing onto local banking once the mean
            # has built -- which is the emergent transition the closure claims.
            half_flux = 0.5 * (anom_power_eV / E_tail)
            tail_power = float(anom_power_eV.sum()) * _ERG_PER_EV
            if tail_sub_threshold:
                tail_sub_threshold_power = tail_power
            elif tail_above_bar:
                tail_above_bar_power = tail_power
            if ionize_tail or reflect_face is not None:
                # The windowed closure's standing requirement: the window must
                # contain every cell the QL channel drives, or that cell's tail
                # power would be dropped on the floor.
                for birth in np.flatnonzero(half_flux > 0.0):
                    if not tail_lo <= birth <= tail_hi:
                        raise ValueError(
                            f"anomalous power in cell {int(birth)} lies "
                            f"outside tail_walk_window {(tail_lo, tail_hi)}; "
                            "the window must contain every cell the QL "
                            "channel drives, or that cell's tail power would "
                            "be silently dropped"
                        )
            if ionize_tail:
                # K6 under coverage: the walkers attenuate INELASTICALLY on the
                # mean neutral column as well as Coulomb-slowing, so the
                # closed-form integral cannot carry them. March them on this
                # module's own CSDA integration over the WINDOWED MEAN medium
                # -- the same instrument the primary uses, so cross sections,
                # thresholds, <W_sec> and substep control are the primary's by
                # construction rather than by transcription.
                #
                # BURN ATTRIBUTION: every per-cell bank the marched walker
                # produces is split between the arms by the local coverage --
                # f_cov to the channel, (1 - f_cov) to the reservoir. That is
                # the decorrelation partition applied to the walker's path, and
                # it is the whole implementation: the caller already reads the
                # reservoir arm's ionization rows as the reservoir debit and
                # the arms' sum as the total, so the deficit equation picks the
                # split up with no machinery of its own. The energy rows are
                # split by the same weights, which leaves their SUM -- the only
                # thing read -- exactly where it shipped.
                win = slice(tail_lo, tail_hi + 1)
                nn_w = nn_mean[win]
                ne_w = ne_mean[win]
                Te_w = Te[win]
                dz_w = dz_cm[win]
                f_w = f_cov[win]

                def _bank_tail_march(res):
                    """Book one marched tail population, PARTITIONED by f_cov."""
                    for arm, weight in ((0, f_w), (1, 1.0 - f_w)):
                        bank = banks[arm]
                        bank["ionization_events"][win] += (
                            weight * res.ionization_events
                        )
                        bank["excitation_events"][win] += (
                            weight * res.excitation_events
                        )
                        bank["ionization_cost"][win] += (
                            weight * res.ionization_cost_erg_s
                        )
                        bank["radiated"][win] += weight * res.radiated_erg_s
                    # All of the walker's HEAT is the anomalous channel's
                    # delivery to the electrons, so it lands in the lumped bank
                    # and in the anomalous split -- never in the primary's
                    # coulomb/secondary/terminal splits, which keep describing
                    # the primary alone. Booked whole to the channel slot, like
                    # every other walked deposit above.
                    chan["heating"][win] += res.plasma_heating_erg_s
                    chan["heat_anomalous"][win] += res.plasma_heating_erg_s
                    # Diagnostic tail splits of the four shared banks, on the
                    # same partition as the banks they split.
                    for arm, weight in ((0, f_w), (1, 1.0 - f_w)):
                        bank = banks[arm]
                        bank["ion_events_tail"][win] += (
                            weight * res.ionization_events
                        )
                        bank["exc_events_tail"][win] += (
                            weight * res.excitation_events
                        )
                        bank["ion_cost_tail"][win] += (
                            weight * res.ionization_cost_erg_s
                        )
                        bank["radiated_tail"][win] += (
                            weight * res.radiated_erg_s
                        )

                march_kwargs = dict(
                    I_ion_eV=I_ion_eV,
                    E_stop_eV=E_stop_eV,
                    coulomb_model=coulomb_model,
                    anomalous_model="none",
                    max_energy_fraction_per_substep=frac,
                )
                for birth in np.flatnonzero(half_flux > 0.0):
                    for walk_direction in (1, -1):
                        leg_dir = walk_direction
                        leg = deposit_beam(
                            E_tail,
                            float(half_flux[birth]),
                            nn_w,
                            ne_w,
                            Te_w,
                            int(birth) - tail_lo,
                            leg_dir,
                            dz_w,
                            **march_kwargs,
                        )
                        while True:
                            _bank_tail_march(leg)
                            leg_flux = float(leg.transmitted_flux)
                            leg_E = float(leg.transmitted_energy_eV)
                            if (
                                reflect_face is not None
                                and leg_dir == reflect_face
                                and leg_flux > 0.0
                                and leg_E < E_reflect
                            ):
                                leg_dir = -leg_dir
                                leg = deposit_beam(
                                    leg_E,
                                    leg_flux,
                                    nn_w,
                                    ne_w,
                                    Te_w,
                                    0 if reflect_face < 0 else tail_hi - tail_lo,
                                    leg_dir,
                                    dz_w,
                                    **march_kwargs,
                                )
                                continue
                            exit_erg = leg_flux * leg_E * _ERG_PER_EV
                            if leg_dir > 0:
                                end_loss_tail_high += exit_erg
                            else:
                                end_loss_tail_low += exit_erg
                            break
            elif reflect_face is not None:
                # K7 with one reflecting window face. A reflection is expressed
                # by UNFOLDING the path: the reflected leg is the window
                # traversed back the other way, concatenated onto the incoming
                # leg, so the walk still telescopes exactly across the bounce.
                win = slice(tail_lo, tail_hi + 1)
                n_w = tail_hi - tail_lo + 1
                coeff_w = coeff[win]
                dz_w = dz_cm[win]
                floor_w = floor_eV[win]
                flux_w = half_flux[win]
                W0_w = np.full(n_w, E_tail)
                order_hit = (
                    np.arange(n_w)[::-1] if reflect_face < 0 else np.arange(n_w)
                )
                order_away = order_hit[::-1]
                escape_at_face = 0.0     # leaves through the reflecting face
                escape_opposite = 0.0    # leaves through the other one

                def _leg(order, W0, flux):
                    return _walk_products_forward(
                        W0[order], flux[order], coeff_w[order], dz_w[order],
                        floor_w[order], q,
                    )

                def _bank_tail_walk(dep_eV, *orders):
                    """Deposit one walked population, in ITS traversal order."""
                    dep_win = np.zeros(n_w)
                    for k, order in enumerate(orders):
                        dep_win[order] += dep_eV[k * n_w:(k + 1) * n_w]
                    dep_erg = dep_win * _ERG_PER_EV
                    chan["heating"][win] += dep_erg
                    chan["heat_anomalous"][win] += dep_erg

                # The arm walking AWAY from the reflecting face never meets it.
                dep_a, exit_a, _ = _leg(order_away, W0_w, flux_w)
                _bank_tail_walk(dep_a, order_away)
                escape_opposite += exit_a
                # The arm walking INTO it: populations born in different cells
                # arrive with different energies, so the threshold test is a
                # per-birth split, not a whole-arm switch.
                dep_h, exit_h, (act_h, W_face, stop_h) = _leg(
                    order_hit, W0_w, flux_w
                )
                bounced = (~stop_h) & (W_face < E_reflect)
                if not bounced.any():
                    _bank_tail_walk(dep_h, order_hit)
                    escape_at_face += exit_h
                else:
                    flux_hit = flux_w[order_hit]
                    flux_bounce = np.zeros(n_w)
                    flux_escape = np.zeros(n_w)
                    flux_bounce[act_h[bounced]] = flux_hit[act_h[bounced]]
                    flux_escape[act_h[~bounced]] = flux_hit[act_h[~bounced]]
                    if np.any(flux_escape > 0.0):
                        dep_e, exit_e, _ = _walk_products_forward(
                            W0_w[order_hit], flux_escape, coeff_w[order_hit],
                            dz_w[order_hit], floor_w[order_hit], q,
                        )
                        _bank_tail_walk(dep_e, order_hit)
                        escape_at_face += exit_e
                    dep_u, exit_u, _ = _walk_products_forward(
                        np.concatenate([W0_w[order_hit], np.zeros(n_w)]),
                        np.concatenate([flux_bounce, np.zeros(n_w)]),
                        np.concatenate(
                            [coeff_w[order_hit], coeff_w[order_away]]
                        ),
                        np.concatenate([dz_w[order_hit], dz_w[order_away]]),
                        np.concatenate(
                            [floor_w[order_hit], floor_w[order_away]]
                        ),
                        q,
                    )
                    _bank_tail_walk(dep_u, order_hit, order_away)
                    escape_opposite += exit_u
                if reflect_face > 0:
                    end_loss_tail_high += escape_at_face * _ERG_PER_EV
                    end_loss_tail_low += escape_opposite * _ERG_PER_EV
                else:
                    end_loss_tail_low += escape_at_face * _ERG_PER_EV
                    end_loss_tail_high += escape_opposite * _ERG_PER_EV
            else:
                tail_W = np.full(cells, E_tail)
                for walk_direction in (1, -1):
                    exit_erg = _walk_and_deposit(
                        tail_W, half_flux, walk_direction, "heat_anomalous"
                    )
                    if walk_direction > 0:
                        end_loss_tail_high += exit_erg
                    else:
                        end_loss_tail_low += exit_erg
        if walk_products and not absorbed and gamma_total > 0.0 and E > 0.0:
            # The transmitted primary: computed by the march, never banked. It
            # leaves through the end the ray was heading for.
            end_loss_transmitted = gamma_total * E * _ERG_PER_EV
            if direction > 0:
                end_loss_high += end_loss_transmitted
            else:
                end_loss_low += end_loss_transmitted

    results = []
    for arm, bank in enumerate(banks):
        # The transmitted primary is the mixed stream leaving the far end; it
        # is booked to the CHANNEL slot rather than split, because after the
        # last re-mix there is one population and no medium owns it. The
        # reservoir slot's transmitted flux is 0 by that convention, which is
        # why the caller must read the SUM (or ``flux_entry``) when it wants
        # the ray's survival.
        transmitted = 0.0 if (absorbed or arm == 1) else gamma_total
        results.append(
            BeamDepositionResult(
                ionization_events=bank["ionization_events"],
                excitation_events=bank["excitation_events"],
                plasma_heating_erg_s=bank["heating"],
                radiated_erg_s=bank["radiated"],
                ionization_cost_erg_s=bank["ionization_cost"],
                transmitted_flux=transmitted,
                transmitted_energy_eV=0.0 if transmitted <= 0.0 else E,
                anode_intercepted_erg_s=anode_intercepted[arm],
                E_entry_eV=bank["E_entry"],
                heating_coulomb_erg_s=bank["heat_coulomb"],
                heating_anomalous_erg_s=bank["heat_anomalous"],
                heating_secondary_erg_s=bank["heat_secondary"],
                heating_terminal_erg_s=bank["heat_terminal"],
                # K6 diagnostic splits, PARTITIONED by f_cov like the shared
                # banks they split, so the two arms' tail rows sum to the
                # walkers' own totals exactly as their parents do.
                ionization_events_tail=bank["ion_events_tail"],
                excitation_events_tail=bank["exc_events_tail"],
                ionization_cost_tail_erg_s=bank["ion_cost_tail"],
                radiated_tail_erg_s=bank["radiated_tail"],
                # The end ledgers belong to the WALK, which ran once on the
                # mean state for both arms' births, so they are booked whole to
                # the channel slot on the same convention the transmitted
                # primary uses above. The reservoir slot carries 0.0, which is
                # why a caller reading escape must read the SUM.
                end_loss_low_erg_s=0.0 if arm == 1 else end_loss_low,
                end_loss_high_erg_s=0.0 if arm == 1 else end_loss_high,
                end_loss_transmitted_erg_s=(
                    0.0 if arm == 1 else end_loss_transmitted
                ),
                end_loss_tail_low_erg_s=0.0 if arm == 1 else end_loss_tail_low,
                end_loss_tail_high_erg_s=(
                    0.0 if arm == 1 else end_loss_tail_high
                ),
                tail_power_erg_s=0.0 if arm == 1 else tail_power,
                tail_sub_threshold_power_erg_s=(
                    0.0 if arm == 1 else tail_sub_threshold_power
                ),
                tail_above_bar_power_erg_s=(
                    0.0 if arm == 1 else tail_above_bar_power
                ),
            )
        )
    return results[0], results[1], flux_entry
