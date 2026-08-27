"""Pre-breakdown passive-tracer bridge (regime R2), default off.

On a cell the tracer owns, the plasma density is the EXACT integral of the
affine scalar ODE ``dn/dt = gamma(z)*n + S(z, t)`` while the background
(circuit ramp, cathode thermal state, coverage, neutrals) owns the timestep.
``_sim1d/NUMERICS.md`` section "Regime-R2 pre-breakdown passive-tracer bridge"
is the method of record; this module is that section in executable form and the
two are meant to be read together.

Nothing here re-derives physics the solver already owns. ``gamma``, the beam
birth ``S``, and the quasi-static electron energy balance are all assembled by
calling the solver's OWN term functions on a probe state and dividing out each
channel's known homogeneity degree in ``n``. That is exact, and it means the
tracer automatically consumes whatever closure the run configured (ADAS vs
Janev rates, coverage split, CSDA vs Beer-Lambert deposition) rather than
carrying a second opinion about any of them.
"""

import math

import numpy as np

from cablp.atomic.adas import _shared_grid_tables, he_rate_temperature_range_eV
from cablp.cathode.beam_deposition import coulomb_stopping_eV_per_cm
from cablp.plasma.params import c_log, time_elec_coll
from cablp.constants import ev_to_erg, m_e_SI, qe_SI

from ..core.state import ConservativeState1D
from .cathode import beam_anomalous_power_density
from .energy import electron_cooling_rhs_terms, electron_ion_exchange_rhs
from .flux import ion_sound_speed
from .reactions import reaction_rates


#: The three passivity criteria, in census order. The census carries a fourth
#: number, ``transport_ratio``, which is NOT here because it never decides
#: passivity: it is the parallel-advection term the description DROPS, and a
#: run in which it is not small is a run whose tracer leg should not be trusted
#: (the quantified bound is the table in NUMERICS.md).
CRITERION_NAMES = ("current", "thinness", "depletion")

#: Series-switch point for :func:`phi2`. ``eps**(1/3)`` is the standard optimum
#: for a second-difference cancellation and is taken from the machine epsilon
#: rather than written as a literal, so it is a floating-point property and not
#: an unregistered threshold.
_PHI2_SERIES_X = float(np.finfo(float).eps) ** (1.0 / 3.0)


def phi1(x):
    """Return ``expm1(x)/x``, regular at ``x = 0`` where it is ``1``.

    The growth factor of the affine update's SOURCE part. ``numpy.expm1`` is
    accurate for small ``|x|``, so the only thing needing care is the removable
    singularity itself.
    """
    x = np.asarray(x, dtype=float)
    zero = x == 0.0
    safe = np.where(zero, 1.0, x)
    return np.where(zero, 1.0, np.expm1(safe) / safe)


def phi2(x):
    """Return ``(phi1(x) - 1)/x``, regular at ``x = 0`` where it is ``1/2``.

    Used ONLY by the accumulators (:func:`affine_time_integral`), never by the
    state update. The difference cancels catastrophically as ``x -> 0``, so
    below ``eps**(1/3)`` the value comes from the series
    ``1/2 + x/6 + x^2/24``; the resulting relative error is at worst ~1e-11 in
    a quantity that is compared against a 1e-2 criterion.
    """
    x = np.asarray(x, dtype=float)
    small = np.abs(x) < _PHI2_SERIES_X
    safe = np.where(small, 1.0, x)
    series = 0.5 + x / 6.0 + x * x / 24.0
    return np.where(small, series, (phi1(safe) - 1.0) / safe)


def affine_update(n, gamma, S, dt):
    """Return ``n`` after ``dt`` of ``dn/dt = gamma*n + S``, exactly.

    ``n_plus = n + n*expm1(gamma*dt) + S*dt*phi1(gamma*dt)`` -- the closed-form
    solution, written so that

    * ``gamma -> 0`` is not a special case (it gives ``n + S*dt``),
    * ``n = 0`` is a REGULAR state (it gives ``S*dt``), so a true-vacuum
      initial condition runs,
    * a decaying cell relaxes onto the exact equilibrium ``-S/gamma`` instead of
      oscillating about a density floor, and
    * there is no stability limit at all: ``dt`` is the background's to choose.

    ``n + n*expm1(x)`` rather than ``n*exp(x)`` avoids the cancellation of
    ``exp(x) - 1`` at small ``|x|``.
    """
    n = np.asarray(n, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    S = np.asarray(S, dtype=float)
    x = gamma * float(dt)
    return n + n * np.expm1(x) + S * float(dt) * phi1(x)


def affine_time_integral(n, gamma, S, dt):
    """Return ``int_0^dt n(t) dt`` for the same affine ODE, exactly.

    ``dt*(n*phi1(x) + S*dt*phi2(x))``. Read by the neutral-depletion
    accumulator (criterion c) and the conservation ledger; never by the state.
    """
    n = np.asarray(n, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    S = np.asarray(S, dtype=float)
    dt = float(dt)
    x = gamma * dt
    return dt * (n * phi1(x) + S * dt * phi2(x))


def probe_state(state, n_probe, Te_eV, Ti_eV, ion_mass_g):
    """Return ``state`` with the plasma rows replaced at a chosen ``(n, Te, Ti)``.

    The instrument the whole module is built on. Every plasma channel the
    solver owns is a homogeneous function of ``n`` of known degree at fixed
    ``(nn, Te, Ti)``, so evaluating the solver's own term function here and
    dividing by ``n_probe**degree`` recovers the channel's COEFFICIENT exactly,
    with no duplicated physics.

    ``n_probe`` is the caller's ``max(n, ne_floor)``: ``derive_state`` and the
    ADAS lookups clamp their density argument at that floor, so probing there
    means the coefficients come back at exactly the density the fluid itself
    would have used. The neutral rows pass through untouched -- they are the
    background the tracer rides on.
    """
    n_probe = np.asarray(n_probe, dtype=float)
    return ConservativeState1D(
        n=n_probe,
        nn=np.asarray(state.nn, dtype=float),
        M=np.zeros_like(n_probe),
        Ee=1.5 * n_probe * np.asarray(Te_eV, dtype=float) * ev_to_erg,
        Ei=1.5 * n_probe * np.asarray(Ti_eV, dtype=float) * ev_to_erg,
        M_n=None if state.M_n is None else np.asarray(state.M_n, dtype=float),
        nn_a=None if state.nn_a is None else np.asarray(state.nn_a, dtype=float),
        M_n_a=(
            None if state.M_n_a is None else np.asarray(state.M_n_a, dtype=float)
        ),
        En=None if state.En is None else np.asarray(state.En, dtype=float),
    )


def growth_rate(
    *,
    state,
    n_true,
    n_probe,
    Te_eV,
    Ti_eV,
    floors,
    ion_mass_g,
    reaction_kwargs,
    boundary_rhs,
):
    """Return the per-cell affine growth rate ``gamma`` [1/s].

    ``gamma = gamma_ion - gamma_rec_rad - gamma_rec_3b - gamma_boundary``, each
    channel recovered from the solver's own term function by dividing out its
    homogeneity degree in ``n``:

    ==============  ======  ==============================================
    channel         degree  source
    ==============  ======  ==============================================
    bulk ionization    1    ``reactions.reaction_rates`` ``S_ion``
    radiative recomb   2    ``reactions.reaction_rates`` ``S_rec_rad``
    three-body recomb  3    ``reactions.reaction_rates`` ``S_rec_3b``
    plasma-end loss    1    ``boundary_rhs`` ``n`` row
    ==============  ======  ==============================================

    A degree-``d`` channel evaluated at ``n_probe`` equals its coefficient times
    ``n_probe**d``, so its contribution to ``gamma`` at the TRUE density is the
    probe value divided by ``n_probe`` and multiplied by ``(n_true/n_probe)**(d-1)``.
    The ratio is unity above the density floor, which is why the identity
    ``gamma*n + S ==`` the fluid's summed ``n`` row holds bit-for-bit there.

    ``boundary_rhs`` is a callable ``probe_state -> ConservativeState1D``
    supplied by the caller, NOT a fixed choice of operator. The plasma-
    terminating faces are discretized two different ways in this package -- the
    volumetric Bohm absorption ``sources.boundary_absorption_rhs`` and the R3
    one-sided characteristic ghost-cell outflow -- and which one is live is a
    config selector. Taking either one unconditionally would make the tracer
    disagree with the fluid on whichever stance did not match; the shipped
    default is the characteristic one, so a fixed choice here would have been
    wrong in production. Both are linear in ``n`` at fixed ``Te``, so the same
    degree-1 division recovers the frequency from either.

    Parallel advection is NOT in ``gamma``: a passive cell exchanges nothing
    across its interface (NUMERICS.md, "Flux at the interface"), and that
    omission is the seed-transport neglect whose bound the census reports as
    ``transport``.
    """
    probe = probe_state(state, n_probe, Te_eV, Ti_eV, ion_mass_g)
    ratio = np.asarray(n_true, dtype=float) / n_probe

    S_ion, S_rec_rad, S_rec_3b = reaction_rates(
        state=probe,
        floors=floors,
        ion_mass_g=ion_mass_g,
        **reaction_kwargs,
    )
    gamma_ion = S_ion / n_probe
    gamma_rec_rad = (S_rec_rad / n_probe) * ratio
    gamma_rec_3b = (S_rec_3b / n_probe) * ratio * ratio

    boundary = boundary_rhs(probe)
    gamma_boundary = -np.asarray(boundary.n, dtype=float) / n_probe

    return gamma_ion - gamma_rec_rad - gamma_rec_3b - gamma_boundary


def electron_loss_coefficients(
    *,
    state,
    n_probe,
    Te_eV,
    Ti_eV,
    floors,
    ion_mass_g,
    mu,
    cooling_kwargs,
    exchange_kwargs,
    boundary_rhs,
):
    """Return ``(L1, L2)``: electron energy sinks by homogeneity degree in ``n``.

    Both are POSITIVE loss rates [erg cm^-3 s^-1] evaluated at ``n_probe``:

    * ``L1`` -- degree 1 in ``n``: the ionization energy cost, the
      electron-neutral line power, and the ``1.5*Te`` the plasma-end loss
      carries out with each lost particle.
    * ``L2`` -- degree 2: the electron-ion line power and the electron-ion
      thermal exchange.

    The boundary term is here for the same reason it is in :func:`growth_rate`,
    and comes through the same caller-supplied ``boundary_rhs`` so the two
    cannot disagree about which discretization is live: it is a genuinely
    PER-CELL sink (applied one-sidedly to the live cell at a plasma-terminating
    face), the tracer already consumes that term for the particle channel, and
    taking its particle row without its energy row would be an inconsistency in
    the tracer rather than a modelling choice. It is not, and must not be read
    as, a repair for the non-local sink the local balance is missing -- see
    NUMERICS.md.

    The Coulomb logarithm inside the exchange term carries a weak logarithmic
    density dependence that the degree split treats as part of the coefficient
    at ``n_probe``. That is the one place the split is approximate, and it is
    approximate at the level of ``ln(n/n_probe)/ln(Lambda)``.
    """
    probe = probe_state(state, n_probe, Te_eV, Ti_eV, ion_mass_g)
    cooling = electron_cooling_rhs_terms(
        state=probe,
        floors=floors,
        ion_mass_g=ion_mass_g,
        **cooling_kwargs,
    )
    exchange = electron_ion_exchange_rhs(
        state=probe,
        floors=floors,
        ion_mass_g=ion_mass_g,
        mu=mu,
        **exchange_kwargs,
    )
    boundary = boundary_rhs(probe)
    L1 = -(
        np.asarray(cooling["ionization_energy_cost"].Ee, dtype=float)
        + np.asarray(cooling["electron_neutral_cooling"].Ee, dtype=float)
        + np.asarray(boundary.Ee, dtype=float)
    )
    L2 = -(
        np.asarray(cooling["electron_ion_cooling"].Ee, dtype=float)
        + np.asarray(exchange.Ee, dtype=float)
    )
    return L1, L2


def passive_anomalous_leak(
    *,
    P_beam_net_consumed,
    P_beam_net_full,
    passive,
    beam_kwargs,
):
    """Return the deviation of a PASSIVE cell's booking from the model's policy.

    Zero, cell by cell, is the invariant. What the policy IS depends on the
    anomalous closure, and this function re-derives that too:

    * ``beam_anomalous_model="quasilinear"`` -- the FIAT leg. The balance must
      be fed the beam power MINUS the anomalous share, because that closure
      books near-total absorption by assertion and quasilinear absorption is a
      beam-PLASMA instability: a passive cell has no plasma to carry the wave.
    * ``"ql_relaxation"`` -- the MIDDLE leg. Its own onset gate and its own
      ``(n_b/2n_e)^(1/3)`` extracted fraction already decide how much a cell of
      that density absorbs, so the balance must be fed the power in FULL and
      subtracting it would delete the closure's content.
    * ``"none"`` -- there is no anomalous share, and full and net coincide.

    NUMERICS.md, "Corrected beam power booking on passive cells" and "The
    anomalous closure bracket", are the statements of record.

    The check is deliberately built the long way round rather than by asking
    the subtraction whether it subtracted:

    * ``P_beam_net_full`` is the beam power the FLUID rows carry -- the three
      ``Ee`` rows summed, with no knowledge of the tracer;
    * the anomalous share is recomputed HERE, from the deposition objects,
      through this module's own reference to
      :func:`~.cathode.beam_anomalous_power_density`, on the ``beam_kwargs``
      the beam rows themselves were built from;
    * the model key is likewise re-read HERE, straight off
      ``beam_kwargs["input_dict"]``, rather than through any predicate the
      solver shares with the subtraction -- so a build that has rebound or
      widened the keying is caught by the same mechanism that catches a build
      that removed the subtraction;
    * ``P_beam_net_consumed`` is what the balance was actually handed.

    So a build in which the refusal has been removed, disabled, or rebound to
    something that returns zeros still gets caught: the checker's own anomalous
    share does not travel through the code path it is auditing. The smoke's
    anti-vacuity case removes the refusal exactly that way and asserts this
    returns the full anomalous power.

    ACTIVE cells are excluded, not audited-and-passed: on them every closure's
    booking stands unchanged. The passive/active boundary is the gate for the
    fiat leg, which is the same statement as "the tracer handoff and QL onset
    are one event".
    """
    passive = np.asarray(passive, dtype=bool)
    refusing = str(
        beam_kwargs["input_dict"].get("beam_anomalous_model", "none")
    ) == "quasilinear"
    P_full = np.asarray(P_beam_net_full, dtype=float)
    expected = P_full
    if refusing:
        expected = P_full - beam_anomalous_power_density(**beam_kwargs)
    residual = np.asarray(P_beam_net_consumed, dtype=float) - expected
    return np.where(passive, residual, 0.0)


def _atomic_scan_temperatures_eV():
    """Return the bundled He ADF11 temperature nodes [eV].

    The scan grid for the multi-valued check is the ATOMIC DATA's own
    temperature grid: it is the resolution at which the loss coefficients can
    actually wiggle, so a sign change the data cannot represent is not looked
    for and no scan resolution has to be invented.
    """
    (_log_ne, log_te), _tables = _shared_grid_tables()
    return 10.0 ** np.asarray(log_te, dtype=float)


class TracerBalanceError(ValueError):
    """The quasi-static electron energy balance has no usable root.

    Raised rather than silently picking a branch. Either the balance wants a
    ``Te`` outside the bundled atomic-data domain (the description has left the
    data it is built from), or it is MULTI-VALUED on that domain (more than one
    sign change on the data's own temperature grid), in which case the local
    closure does not define ``Te`` and the tracer is not usable at that cell.
    Both are reportable findings.
    """


def quasistatic_Te_eV(
    *,
    state,
    n_true,
    n_probe,
    Ti_eV,
    S_beam,
    P_beam_net,
    floors,
    ion_mass_g,
    mu,
    cooling_kwargs,
    exchange_kwargs,
    boundary_rhs,
    active,
    Te_ceiling_eV=0.0,
):
    """Return ``(Te_eV, sign_changes)`` from the quasi-static energy balance.

    Solves, per cell,

        1.5 * Te * ev_to_erg * S  =  P_beam_net  -  r*L1(Te)  -  r^2*L2(Te)

    with ``r = n_true/n_probe``, i.e. ``G(Te) = 0`` for

        G(Te) = 1.5*Te*ev_to_erg*S - P_beam_net + r*L1(Te) + r^2*L2(Te).

    The left-hand side is the DILUTION cost: the model's beam ionization births
    its electron at ``Ee = 0``, so every beam-born electron has to be raised to
    the bulk temperature out of the deposited power. That term is what makes
    the balance well posed as ``n -> 0``: the vacuum limit is
    ``Te -> (2/3)*P_beam_net/(ev_to_erg*S)``, the beam's W-value in the gas,
    rather than a runaway. Bulk-ionization births carry the local ``Te`` and so
    do not appear.

    ``G(Te_lo) >= 0`` means nothing is heating the cell and ``Te`` sits at the
    bracket bottom (the temperature floor). ``G(Te_hi) < 0`` means the balance
    wants a temperature the description cannot supply and raises.

    The bracket top is ``min`` of two hard bounds, neither of them a chosen
    number: the bundled He ADF11 temperature-grid top (past which the loss
    coefficients are extrapolation, the same domain-guard role
    ``cathode_phi_c_cap_V`` plays for the sheath) and ``(2/3)*E_beam`` -- an
    electron population cannot carry a mean energy above the beam energy
    heating it. ``Te_ceiling_eV`` is that beam energy, the R1-BOUNDED one;
    passing ``0`` (no beam) leaves the atomic-data top alone.

    UNIQUENESS IS NOT CLAIMED: ``L1`` climbs steeply with ``Te`` but the
    exchange part of ``L2`` falls like ``Te**-0.5``, so monotonicity of ``G`` is
    a property of the operating point, not a theorem. The solve is therefore a
    bracketed bisection, which cannot diverge or leave the bracket, and the
    number of sign changes of ``G`` on the atomic data's own temperature grid is
    counted and returned. More than one is a raise.

    ``active`` selects the cells that HAVE a balance: a cell with neither
    plasma nor a beam birth holds no electrons, so its ``Te`` is unobservable
    (``Ee = 1.5*n*Te = 0`` for any ``Te``) and it is set to the floor by
    convention rather than solved.
    """
    active = np.asarray(active, dtype=bool)
    Te_lo_scalar, Te_hi_scalar = he_rate_temperature_range_eV()
    Te_lo_scalar = max(float(Te_lo_scalar), float(floors["Te"]))
    beam_ceiling = (2.0 / 3.0) * float(Te_ceiling_eV)
    if beam_ceiling > Te_lo_scalar:
        Te_hi_scalar = min(float(Te_hi_scalar), beam_ceiling)
    cells = np.asarray(n_probe, dtype=float).size
    ratio = np.asarray(n_true, dtype=float) / np.asarray(n_probe, dtype=float)
    S_beam = np.asarray(S_beam, dtype=float)
    P_beam_net = np.asarray(P_beam_net, dtype=float)

    def G(Te_vec):
        L1, L2 = electron_loss_coefficients(
            state=state,
            n_probe=n_probe,
            Te_eV=Te_vec,
            Ti_eV=Ti_eV,
            floors=floors,
            ion_mass_g=ion_mass_g,
            mu=mu,
            cooling_kwargs=cooling_kwargs,
            exchange_kwargs=exchange_kwargs,
            boundary_rhs=boundary_rhs,
        )
        return (
            1.5 * Te_vec * ev_to_erg * S_beam
            - P_beam_net
            + ratio * L1
            + ratio * ratio * L2
        )

    # Multi-valuedness scan on the atomic data's own temperature grid.
    scan = _atomic_scan_temperatures_eV()
    scan = scan[(scan >= Te_lo_scalar) & (scan <= Te_hi_scalar)]
    scan = np.concatenate(([Te_lo_scalar], scan, [Te_hi_scalar]))
    signs = np.sign(np.stack([G(np.full(cells, t)) for t in scan]))
    # A sample sitting exactly on zero is not a crossing on its own; carry the
    # previous nonzero sign forward so a tangency is not counted twice.
    carried = np.zeros(cells, dtype=float)
    sign_changes = np.zeros(cells, dtype=int)
    for row in signs:
        moved = (row != 0.0) & (carried != 0.0) & (row != carried)
        sign_changes += moved.astype(int)
        carried = np.where(row != 0.0, row, carried)

    lo = np.full(cells, Te_lo_scalar, dtype=float)
    hi = np.full(cells, Te_hi_scalar, dtype=float)
    G_lo = G(lo)
    G_hi = G(hi)

    at_floor = (G_lo >= 0.0) & active
    above_domain = (G_hi < 0.0) & active & ~at_floor
    if np.any(above_domain):
        cell = int(np.flatnonzero(above_domain)[0])
        raise TracerBalanceError(
            "the quasi-static electron energy balance wants Te above the "
            f"bracket top ({Te_hi_scalar:.6g} eV -- the lesser of the bundled "
            "He ADF11 grid top and two thirds of the R1-bounded beam energy) "
            f"at cell {cell} (of {cells}): the deposited beam power "
            f"{float(P_beam_net[cell]):.6g} erg cm^-3 s^-1 exceeds everything "
            "the dilution and radiative channels can absorb there. Either the "
            "tracer has left the atomic data it is built from, or the balance "
            "is asking for electrons hotter than the beam heating them; this "
            "is a finding, not something to clamp"
        )
    multi = (sign_changes > 1) & active & ~at_floor
    if np.any(multi):
        cell = int(np.flatnonzero(multi)[0])
        raise TracerBalanceError(
            "the quasi-static electron energy balance is MULTI-VALUED at cell "
            f"{cell} (of {cells}): {int(sign_changes[cell])} sign changes on "
            "the He ADF11 temperature grid, so the local closure does not "
            "define a single Te there and the tracer cannot pick a branch. "
            "Report this rather than selecting a root"
        )

    solve = active & ~at_floor
    # Bisection. Bounded and deterministic: each pass halves every bracket, so
    # 200 passes take the widest possible bracket below the double-precision
    # spacing of its own endpoints, and the loop exits as soon as every live
    # bracket is adjacent.
    for _ in range(200):
        width = hi - lo
        if not np.any(solve & (width > np.spacing(hi))):
            break
        mid = 0.5 * (lo + hi)
        G_mid = G(mid)
        take_upper = G_mid < 0.0
        lo = np.where(solve & take_upper, mid, lo)
        hi = np.where(solve & ~take_upper, mid, hi)
    Te = 0.5 * (lo + hi)
    Te = np.where(solve, Te, float(floors["Te"]))
    return Te, sign_changes


def spitzer_conductivity_S_per_m(n_cm3, Te_eV):
    """Return the Spitzer parallel conductivity ``sigma_par`` [S/m].

    ``sigma = n e^2 tau_e / m_e`` with the Braginskii electron collision time
    the rest of the package already uses (``plasma.params.time_elec_coll``,
    same ``c_log`` as the conduction and exchange terms), converted to SI so the
    conducted current in criterion (a) comes out in amperes without a statamp
    detour.
    """
    n_cm3 = np.maximum(np.asarray(n_cm3, dtype=float), 0.0)
    Te_eV = np.asarray(Te_eV, dtype=float)
    positive = n_cm3 > 0.0
    n_safe = np.where(positive, n_cm3, 1.0)
    tau_e = time_elec_coll(Te_eV, n_safe, c_log(Te_eV, n_safe, kind="ei"))
    sigma = (n_safe * 1.0e6) * qe_SI * qe_SI * tau_e / m_e_SI
    return np.where(positive, sigma, 0.0)


def conducted_current_A(*, n_cm3, Te_eV, geometry, V_dev_V, L_plasma_cm):
    """Return the current each cell actually CONDUCTS under the device drop [A].

    ``I_cond = sigma_par(n, Te) * A_plasma * (V_dev / L_plasma)`` -- Ohm's law
    on the cell's own cross-section at the mean axial field the device drop
    implies. This is the current the plasma passes, NOT the cathode's emission
    capability: conflating the two is on record as the defect this criterion was
    rewritten to fix, because what the emitter COULD supply says nothing about
    whether the column is shunting the loop.

    ``V_dev_V`` is the R1-bounded device voltage (the sheath solve's ``V_b``,
    held at or below the composed ceiling, whose circuit member
    ``cathode_circuit_bound_object`` selects). The raw atomic-data cap would
    inflate this by the same factor R1 removed. The vessel node's ``V_cm`` does
    NOT enter here: a common-mode offset moves the whole cathode/anode system
    against the wall and so cannot change the anode-to-cathode differential the
    column conducts under. It enters criterion (b)'s BEAM energy instead.

    This is deliberately an UPPER BOUND, and must be read as one. Putting the
    WHOLE device drop across the column overstates the axial field, because
    most of that drop is the cathode sheath fall rather than the column. So a
    cell that PASSES criterion (a) is certainly passive, while a cell that
    fails it may only be failing the bound -- the criterion errs toward giving
    the cell to the fluid, which is the safe direction for a bridge whose whole
    risk is holding a cell in the cheap description too long.

    The obvious refinement is the column drop ``V_b - phi_c - phi_a``, or the
    solver's own ``R_p`` network. Stage 1 does NOT take it: under the
    capability-limited branch ``phi_c`` approaches ``V_b``, so that difference
    collapses toward zero and the criterion would fail the opposite way, and
    the sheath partition it depends on is exactly what the R1 follow-up is
    still moving. Tightening this is a follow-up against a settled partition,
    not a change to make while the partition is in flight.
    """
    sigma = spitzer_conductivity_S_per_m(n_cm3, Te_eV)
    area_m2 = np.asarray(geometry.plasma_face_area_cm2[:-1], dtype=float) * 1.0e-4
    field_V_per_m = abs(float(V_dev_V)) / (max(float(L_plasma_cm), 0.0) * 1.0e-2)
    return sigma * area_m2 * field_V_per_m


def beam_plasma_thinness(*, n_cm3, Te_eV, geometry, E_beam_eV, launch_cells,
                         coulomb_model="fast_electron"):
    """Return the cumulative single-pass beam-energy fraction lost to plasma.

    Criterion (b). Along each cathode's ray the primary loses
    ``coulomb_stopping_eV_per_cm(E_beam, n_e, Te) * dz`` eV per cell to the
    PLASMA electrons; accumulating that from the launch end and dividing by the
    launch energy gives the fraction of a single pass the plasma has taken. The
    stopping power is the beam module's own -- the same function the deposition
    ray marches with -- so the criterion cannot drift away from the physics it
    is supposed to bound.

    ``launch_cells`` is a mapping ``{end: launch_cell_index}``; the cell value is
    the MAX over ends, because a cell is only thin if it is thin to every beam
    that reaches it. Returns zeros when there is no beam.
    """
    cells = int(geometry.cells)
    out = np.zeros(cells, dtype=float)
    E_beam_eV = float(E_beam_eV)
    if not (E_beam_eV > 0.0) or not launch_cells:
        return out
    n_cm3 = np.asarray(n_cm3, dtype=float)
    Te_eV = np.asarray(Te_eV, dtype=float)
    dz = np.asarray(geometry.length_cm, dtype=float)
    per_cell = np.array(
        [
            coulomb_stopping_eV_per_cm(
                E_beam_eV, float(n_cm3[i]), float(Te_eV[i]), model=coulomb_model
            )
            * dz[i]
            / E_beam_eV
            for i in range(cells)
        ],
        dtype=float,
    )
    for launch in launch_cells.values():
        launch = int(launch)
        forward = np.cumsum(per_cell[launch:])
        backward = np.cumsum(per_cell[: launch + 1][::-1])[::-1]
        ray = np.zeros(cells, dtype=float)
        ray[launch:] = forward
        ray[: launch + 1] = np.maximum(ray[: launch + 1], backward)
        out = np.maximum(out, ray)
    return out


def transport_ratio(*, gamma, Te_eV, mu, L_n_cm):
    """Return ``c_s / (L_n * gamma)``: the term the tracer DROPS, over the one it keeps.

    Not a passivity criterion -- it never gates activation -- but reported in
    the census on every step, because it is the honest statement of the tracer's
    accuracy. The quantified bound and the ``Te`` at which it stops being small
    are tabulated in NUMERICS.md ("Seed transport: the quantified neglect").
    ``inf`` where ``gamma <= 0``: a decaying cell has no growth for transport to
    be small against.
    """
    gamma = np.asarray(gamma, dtype=float)
    cs = ion_sound_speed(np.asarray(Te_eV, dtype=float), mu)
    growing = gamma > 0.0
    safe = np.where(growing, gamma, 1.0)
    return np.where(growing, cs / (float(L_n_cm) * safe), np.inf)


def bind_census(ratios):
    """Return ``(worst, binding_index)`` over the three passivity ratios.

    ``ratios`` is a mapping ``{name: array}`` covering :data:`CRITERION_NAMES`.
    The binding index is into :data:`CRITERION_NAMES`; the worst value is the
    number compared against 1 to decide passivity, so the two together are the
    ``active_constraint`` idiom applied to the passive/active interface.
    """
    stacked = np.stack([np.asarray(ratios[name], dtype=float)
                        for name in CRITERION_NAMES])
    return np.max(stacked, axis=0), np.argmax(stacked, axis=0)


def resolve_criteria(input_dict, floors):
    """Return the validated criterion constants, or raise ``ValueError``.

    Called at CONSTRUCTION time, before any step: an out-of-range criterion is a
    misconfiguration and must be loud there rather than at the first refresh.
    Each message names what the key accepts.
    """
    def _unit_interval(key):
        value = float(input_dict.get(key))
        if not (0.0 < value <= 1.0):
            raise ValueError(
                f"{key} must lie in (0, 1] -- it is the largest share of a "
                f"budget the plasma may take and still count as passive "
                f"(got {value!r})"
            )
        return value

    criteria = {
        "current": _unit_interval("tracer_passivity_current_ratio"),
        "thinness": _unit_interval("tracer_passivity_thinness"),
        "depletion": _unit_interval("tracer_passivity_depletion"),
    }
    hysteresis = float(input_dict.get("tracer_passivity_hysteresis"))
    if not hysteresis > 1.0:
        raise ValueError(
            "tracer_passivity_hysteresis must be > 1: it is the enter/exit "
            "RATIO that keeps a cell sitting on a criterion from chattering "
            "between the two descriptions, and a value of 1 or below is no "
            f"hysteresis at all (got {hysteresis!r})"
        )
    refresh_tol = float(input_dict.get("tracer_refresh_tol"))
    if refresh_tol < 0.0:
        raise ValueError(
            "tracer_refresh_tol must be >= 0 (0 refreshes the Picard "
            f"coefficients every step); got {refresh_tol!r}"
        )
    activation_ne = float(input_dict.get("tracer_activation_ne"))
    floor_n = float(floors["n"])
    if not activation_ne >= 10.0 * floor_n:
        raise ValueError(
            f"tracer_activation_ne must be at least 10x ne_floor "
            f"({10.0 * floor_n:g}); handing the fluid a cell whose density "
            "the floor clip is holding up reproduces exactly the "
            "floor-poisoned regime the tracer exists to skip (got "
            f"{activation_ne!r} against ne_floor={floor_n!r})"
        )
    band = input_dict.get("tracer_overlap_band_ne")
    try:
        band_low, band_high = (float(v) for v in band)
    except (TypeError, ValueError):
        raise ValueError(
            "tracer_overlap_band_ne must be a two-element (low, high) density "
            f"band in cm^-3 (got {band!r})"
        ) from None
    if not (0.0 < band_low < band_high):
        raise ValueError(
            "tracer_overlap_band_ne must satisfy 0 < low < high -- it is the "
            "density window in which BOTH descriptions are valid, so an empty "
            f"or inverted band has nothing to compare (got {band!r})"
        )
    overlap_rtol = float(input_dict.get("tracer_overlap_rtol"))
    if not overlap_rtol > 0.0:
        raise ValueError(
            "tracer_overlap_rtol must be positive: it is the relative "
            "agreement the two descriptions must reach inside the overlap "
            f"band (got {overlap_rtol!r})"
        )
    return {
        "criteria": criteria,
        "hysteresis": hysteresis,
        "refresh_tol": refresh_tol,
        "activation_ne": activation_ne,
        "overlap_band_ne": (band_low, band_high),
        "overlap_rtol": overlap_rtol,
    }


def relative_drift(previous, current):
    """Return the largest relative change between two background snapshots.

    The Picard cadence: ``gamma`` and the quasi-static ``Te`` are frozen until
    this exceeds ``tracer_refresh_tol``. Compared against each entry's own
    scale, so a channel that is identically zero contributes nothing rather
    than dividing by zero.
    """
    worst = 0.0
    for name, now in current.items():
        before = previous.get(name)
        if before is None:
            return math.inf
        now = np.asarray(now, dtype=float)
        before = np.asarray(before, dtype=float)
        scale = np.maximum(np.abs(before), np.abs(now))
        moving = scale > 0.0
        if not np.any(moving):
            continue
        worst = max(
            worst,
            float(np.max(np.abs(now[moving] - before[moving]) / scale[moving])),
        )
    return worst
