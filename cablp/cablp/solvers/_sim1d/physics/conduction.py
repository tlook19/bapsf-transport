import numpy as np
from scipy.linalg import solve_banded

from cablp.funcs._heat import kappa_par_elec, kappa_par_ion
from cablp.funcs._plasmaparams import c_log
from cablp.vars._cons import ev_to_erg, m_e_cgs

from ..core.state import ConservativeState1D, derive_state

# Named discretizations for the implicit heat substep, which advances
#   C dT/dt = -K T
# with the conductivity inside K frozen at the incoming state.
#
# Theta methods solve
#   (C + theta*dt*K) T_new = C*T_old - (1 - theta)*dt*K*T_old
# so theta=1 is backward Euler and theta=1/2 is Crank-Nicolson. Only theta=1 is
# L-stable: the amplification factor tends to -(1 - theta)/theta as dt*lambda
# tends to -infinity, so theta=1/2 leaves stiff modes ringing at amplitude 1
# with alternating sign, while "shifted" damps them by ~2/3 per step at the cost
# of formal second-order accuracy. See NUMERICS.md.
IMPLICIT_HEAT_SCHEME_THETA = {
    "backward_euler": 1.0,
    "shifted": 0.6,
    "crank_nicolson": 0.5,
}

# TR-BDF2 (Bank, Coughran, Fichtner, Grosse, Rose & Smith 1985): a trapezoidal
# stage out to t + gamma*dt, then a BDF2 stage through (T_old, T_gamma, T_new).
# Second-order AND L-stable -- the trapezoidal stage rings exactly as
# Crank-Nicolson does, and the BDF2 stage annihilates what it leaves behind, so
# R(-inf) = 0 instead of -1. gamma = 2 - sqrt(2) is chosen to make the two
# stages share an implicit coefficient, gamma/2 == (1 - gamma)/(2 - gamma), and
# therefore share one banded operator.
TR_BDF2 = "tr_bdf2"
_TR_BDF2_GAMMA = 2.0 - np.sqrt(2.0)
_TR_BDF2_IMPLICIT = _TR_BDF2_GAMMA / 2.0
_TR_BDF2_A = 1.0 / (_TR_BDF2_GAMMA * (2.0 - _TR_BDF2_GAMMA))
_TR_BDF2_B = -((1.0 - _TR_BDF2_GAMMA) ** 2) / (_TR_BDF2_GAMMA * (2.0 - _TR_BDF2_GAMMA))

IMPLICIT_HEAT_SCHEMES = (*IMPLICIT_HEAT_SCHEME_THETA, TR_BDF2)


def validate_implicit_heat_scheme(scheme):
    """Return ``scheme`` unchanged if it names an implemented heat scheme."""
    if scheme not in IMPLICIT_HEAT_SCHEMES:
        raise ValueError(
            "implicit_heat_scheme must be one of "
            f"{sorted(IMPLICIT_HEAT_SCHEMES)} (got {scheme!r})"
        )
    return scheme


def resolve_implicit_heat_theta(scheme):
    """Return the theta weight for a named theta-method heat scheme."""
    try:
        return IMPLICIT_HEAT_SCHEME_THETA[scheme]
    except (KeyError, TypeError):
        raise ValueError(
            "implicit_heat_scheme must be one of "
            f"{sorted(IMPLICIT_HEAT_SCHEME_THETA)} (got {scheme!r})"
        ) from None


def heat_conduction_rhs(
    state,
    floors,
    ion_mass_g,
    mu,
    geometry,
    b_epara=1.0,
    b_ipara=1.0,
    heat_conduction=True,
    ln_lambda_min=1.0,
    electron_heat_flux_limit=False,
    heat_flux_limiter_f=0.3,
    heat_flux_limiter_exponent=1.0,
):
    """Return conservative axial heat-conduction energy sources."""
    zeros = np.zeros_like(state.n, dtype=float)
    if not heat_conduction or (b_epara == 0.0 and b_ipara == 0.0):
        return ConservativeState1D(
            n=zeros,
            nn=zeros.copy(),
            M=zeros.copy(),
            Ee=zeros.copy(),
            Ei=zeros.copy(),
        )

    derived = derive_state(state, floors=floors, ion_mass_g=ion_mass_g)
    n = np.maximum(state.n, floors["n"])
    ln_lambda = np.maximum(c_log(derived.Te, n, kind="ei"), ln_lambda_min)

    conductivity_e = (
        kappa_par_elec(derived.Te, n, ln_lambda, per_particle=False)
        * ev_to_erg
        * float(b_epara)
    )
    if electron_heat_flux_limit:
        conductivity_e = flux_limited_electron_conductivity(
            conductivity_e, derived.Te, n, geometry, heat_flux_limiter_f,
            exponent=heat_flux_limiter_exponent,
        )
    qe_face = conductive_face_flux(
        temperature=derived.Te,
        conductivity=conductivity_e,
        geometry=geometry,
    )
    qi_face = conductive_face_flux(
        temperature=derived.Ti,
        conductivity=kappa_par_ion(
            derived.Ti,
            n,
            mu,
            ln_lambda,
            per_particle=False,
        )
        * ev_to_erg
        * float(b_ipara),
        geometry=geometry,
    )
    return ConservativeState1D(
        n=zeros,
        nn=zeros.copy(),
        M=zeros.copy(),
        Ee=flux_divergence_rhs(qe_face, geometry),
        Ei=flux_divergence_rhs(qi_face, geometry),
    )


def flux_limited_electron_conductivity(
    conductivity_e, Te_eV, n, geometry, f, exponent=1.0
):
    """Scale the electron conductivity per cell by the Cowie-McKee flux limiter.

    The classical (Spitzer-Harm) parallel flux ``q_SH = kappa_e |dTe/dz|`` is
    capped toward the free-streaming ceiling ``q_sat = f n Te v_the`` (with
    ``Te`` in erg and ``v_the = sqrt(Te/m_e)``) via the smooth harmonic form

        lambda = q_sat / (q_sat + q_SH),   kappa_eff = lambda * kappa_e

    so ``lambda -> 1`` (recovers Spitzer) where ``q_SH << q_sat`` and
    ``kappa_eff |dTe/dz| -> q_sat`` (the flux saturates at free-streaming) where
    ``q_SH >> q_sat``. Reducing the conductivity keeps the operator a conservative
    flux divergence. Frozen at the incoming ``Te`` like ``kappa`` itself. Audit A9
    / R5.2; ``conductivity_e`` is the already-scaled volumetric conductivity
    (``* ev_to_erg * b_epara``) the operator uses, so ``q_SH`` matches its flux.

    ``exponent`` p (default 1 = the harmonic A9, bit-exact) generalizes to a
    NON-LOCAL Knudsen suppression ``lambda = 1 / (1 + (q_SH/q_sat)^p)``. The ratio
    ``q_SH/q_sat`` is the Knudsen number ``Kn ~ lambda_e/L_T``; ``p > 1`` suppresses
    the steep-gradient (high-Kn, non-local) startup flux MUCH harder while leaving
    the shallow-gradient (low-Kn, ``Kn^p -> 0``) established column near-Spitzer --
    the "cap the front pre-heating, spare the established plasma" behaviour a
    single free-streaming factor cannot separate.
    """
    conductivity_e = np.asarray(conductivity_e, dtype=float)
    Te_eV = np.asarray(Te_eV, dtype=float)
    Te_erg = Te_eV * ev_to_erg
    v_the = np.sqrt(np.maximum(Te_erg, 0.0) / m_e_cgs)  # cm/s
    q_sat = float(f) * np.asarray(n, dtype=float) * Te_erg * v_the  # erg cm^-2 s^-1
    grad = np.gradient(Te_eV, np.asarray(geometry.z_cm, dtype=float))  # eV/cm
    q_SH = np.abs(conductivity_e) * np.abs(grad)  # erg cm^-2 s^-1
    p = float(exponent)
    if p == 1.0:
        # Harmonic A9 (bit-exact with the pre-exponent form).
        denom = q_sat + q_SH
        lam = np.where(denom > 0.0, q_sat / np.where(denom > 0.0, denom, 1.0), 1.0)
    else:
        # Non-local: lambda = 1 / (1 + (q_SH/q_sat)^p).
        ratio = np.where(q_sat > 0.0, q_SH / np.where(q_sat > 0.0, q_sat, 1.0), 0.0)
        lam = np.where(q_sat > 0.0, 1.0 / (1.0 + ratio ** p), 1.0)
    return conductivity_e * lam


def conductive_face_flux(temperature, conductivity, geometry):
    """Return signed internal-face conductive fluxes [erg cm^-2 s^-1]."""
    temperature = np.asarray(temperature, dtype=float)
    conductivity = np.asarray(conductivity, dtype=float)
    q_face = np.zeros(geometry.cells + 1, dtype=float)
    k_face = 0.5 * (conductivity[:-1] + conductivity[1:])
    q_face[1:-1] = -k_face * np.diff(temperature) / geometry.center_distance_cm
    # Faces may throttle parallel conduction independently of the other transport
    # channels: 0 at a plasma wall, (1-eta) across the anode mesh, 1 on a
    # normal interior face -- so legacy geometry is untouched.
    return q_face * geometry.heat_transmission


def flux_divergence_rhs(q_face, geometry):
    """Return ``-div(q)`` as an energy-density RHS [erg cm^-3 s^-1]."""
    face_power = geometry.plasma_face_area_cm2 * q_face
    return -(face_power[1:] - face_power[:-1]) / geometry.plasma_volume_cm3


def heat_conduction_timestep_bound(
    state,
    floors,
    ion_mass_g,
    mu,
    geometry,
    b_epara=1.0,
    b_ipara=1.0,
    heat_conduction=True,
    ln_lambda_min=1.0,
    heat_dt_fraction=0.25,
    active_cells=None,
    electron_heat_flux_limit=False,
    heat_flux_limiter_f=0.3,
    heat_flux_limiter_exponent=1.0,
):
    """Return an explicit diffusion timestep bound for heat conduction.

    The R5.2/A9 flux limiter (``electron_heat_flux_limit``) only REDUCES the
    electron conductivity, so the unlimited-conductivity bound computed here is a
    conservative (tighter) over-estimate of the limited operator's stiffness --
    accepted and ignored so the shared ``_heat_conduction_kwargs`` fits.
    """
    del electron_heat_flux_limit, heat_flux_limiter_f  # conservative: see above
    del heat_flux_limiter_exponent
    if heat_dt_fraction <= 0.0:
        raise ValueError(f"heat_dt_fraction must be positive (got {heat_dt_fraction})")
    if not heat_conduction or (b_epara == 0.0 and b_ipara == 0.0):
        return np.inf

    derived = derive_state(state, floors=floors, ion_mass_g=ion_mass_g)
    n = np.maximum(state.n, floors["n"])
    ln_lambda = np.maximum(c_log(derived.Te, n, kind="ei"), ln_lambda_min)
    capacity = 1.5 * n * ev_to_erg
    dt_e = _species_heat_timestep(
        capacity=capacity,
        conductivity=kappa_par_elec(
            derived.Te,
            n,
            ln_lambda,
            per_particle=False,
        )
        * ev_to_erg
        * float(b_epara),
        geometry=geometry,
        fraction=heat_dt_fraction,
        active_cells=active_cells,
    )
    dt_i = _species_heat_timestep(
        capacity=capacity,
        conductivity=kappa_par_ion(
            derived.Ti,
            n,
            mu,
            ln_lambda,
            per_particle=False,
        )
        * ev_to_erg
        * float(b_ipara),
        geometry=geometry,
        fraction=heat_dt_fraction,
        active_cells=active_cells,
    )
    return min(dt_e, dt_i)


def implicit_heat_conduction_step(
    state,
    floors,
    ion_mass_g,
    mu,
    geometry,
    dt,
    b_epara=1.0,
    b_ipara=1.0,
    heat_conduction=True,
    ln_lambda_min=1.0,
    implicit_heat_scheme="backward_euler",
    heat_picard_iterations=0,
    heat_picard_tol=1e-10,
    electron_heat_flux_limit=False,
    heat_flux_limiter_f=0.3,
    heat_flux_limiter_exponent=1.0,
):
    """Return a state after one implicit heat step.

    ``implicit_heat_scheme`` names one of ``IMPLICIT_HEAT_SCHEMES``: a theta
    method (``backward_euler``, ``shifted``, ``crank_nicolson``) or ``tr_bdf2``.

    ``heat_picard_iterations`` controls how the Braginskii conductivity, which
    depends on temperature as roughly T^(5/2), is evaluated:

    ``0`` (default)
        Freeze it at the incoming state. This is first-order accurate in dt no
        matter how accurate the substep discretization is, so a second-order
        scheme cannot express its order.
    ``N > 0``
        Picard-iterate up to N further times, re-evaluating the conductivity at
        the scheme's own flux evaluation point (see ``_kappa_eval_weight``)
        until the temperature stops moving by more than ``heat_picard_tol``.
        Each iteration costs one more banded solve per species.

    Note that even a fully converged Picard does not by itself make the *split
    step* second-order: ``operator_split_step`` uses Lie rather than Strang
    splitting, which is an independent first-order term. See NUMERICS.md.
    """
    if dt <= 0.0:
        raise ValueError(f"dt must be positive (got {dt})")
    scheme = validate_implicit_heat_scheme(implicit_heat_scheme)
    iterations = max(int(heat_picard_iterations), 0)
    if not heat_conduction or (b_epara == 0.0 and b_ipara == 0.0):
        return ConservativeState1D(
            n=state.n.copy(),
            nn=state.nn.copy(),
            M=state.M.copy(),
            Ee=state.Ee.copy(),
            Ei=state.Ei.copy(),
            M_n=None if state.M_n is None else state.M_n.copy(),
            nn_a=None if state.nn_a is None else state.nn_a.copy(),
            M_n_a=None if state.M_n_a is None else state.M_n_a.copy(),
            En=None if state.En is None else state.En.copy(),
        )

    derived = derive_state(state, floors=floors, ion_mass_g=ion_mass_g)
    n = np.maximum(state.n, floors["n"])
    capacity = 1.5 * n * ev_to_erg
    Te_old = derived.Te
    Ti_old = derived.Ti
    weight = _kappa_eval_weight(scheme)

    # The first pass evaluates the conductivity at Te_old/Ti_old regardless of
    # weight, because the guess starts at the incoming temperature. So
    # iterations=0 runs this loop exactly once and reproduces the original
    # frozen-conductivity step bit-for-bit.
    Ee = Ei = None
    Te_guess, Ti_guess = Te_old, Ti_old
    for _ in range(iterations + 1):
        Te_eval = weight * Te_guess + (1.0 - weight) * Te_old
        Ti_eval = weight * Ti_guess + (1.0 - weight) * Ti_old
        # Overflow guard, not physics: kappa ~ T^2.5 overflows to inf on the
        # extreme transients a bad step can produce (e.g. circuit-driven
        # power spikes), which crashes the banded solve *inside* the attempt
        # before the step-rejection machinery can see and reject the state.
        # 10 keV is far beyond anything physical here, so the clamp is inert
        # on healthy runs and merely keeps bad steps finite and rejectable.
        Te_eval = np.minimum(Te_eval, 1.0e4)
        Ti_eval = np.minimum(Ti_eval, 1.0e4)
        conductivity_e, conductivity_i = _parallel_conductivities(
            Te=Te_eval,
            Ti=Ti_eval,
            n=n,
            mu=mu,
            ln_lambda_min=ln_lambda_min,
            b_epara=b_epara,
            b_ipara=b_ipara,
        )
        if electron_heat_flux_limit:
            conductivity_e = flux_limited_electron_conductivity(
                conductivity_e, Te_eval, n, geometry, heat_flux_limiter_f,
                exponent=heat_flux_limiter_exponent,
            )
        Ee = _implicit_species_energy(
            energy=state.Ee,
            capacity=capacity,
            temperature_floor=floors["Te"],
            conductivity=conductivity_e,
            geometry=geometry,
            dt=dt,
            scheme=scheme,
        )
        Ei = _implicit_species_energy(
            energy=state.Ei,
            capacity=capacity,
            temperature_floor=floors["Ti"],
            conductivity=conductivity_i,
            geometry=geometry,
            dt=dt,
            scheme=scheme,
        )
        Te_next = Ee / capacity
        Ti_next = Ei / capacity
        if _picard_converged(Te_next, Te_guess, heat_picard_tol) and _picard_converged(
            Ti_next, Ti_guess, heat_picard_tol
        ):
            break
        Te_guess, Ti_guess = Te_next, Ti_next

    return ConservativeState1D(
        n=state.n.copy(),
        nn=state.nn.copy(),
        M=state.M.copy(),
        Ee=Ee,
        Ei=Ei,
        M_n=None if state.M_n is None else state.M_n.copy(),
        nn_a=None if state.nn_a is None else state.nn_a.copy(),
        M_n_a=None if state.M_n_a is None else state.M_n_a.copy(),
        En=None if state.En is None else state.En.copy(),
    )


def _kappa_eval_weight(scheme):
    """Return the blend weight w for evaluating kappa at w*T_new + (1-w)*T_old.

    A theta method blends its flux as theta*T_new + (1-theta)*T_old, i.e. it
    evaluates the flux at t^(n+theta), so the conductivity belongs at the
    matching temperature. theta=1 then recovers the fully implicit
    kappa(T_new); theta=1/2 gives kappa(T^(n+1/2)), which is what makes
    Crank-Nicolson second-order on this quasilinear problem. Using one midpoint
    conductivity for both endpoints rather than kappa(T_old) and kappa(T_new)
    separately is the standard linearization and stays second-order: the two
    endpoint errors are equal and opposite to leading order.

    TR-BDF2's two stages share a single banded operator -- that is the whole
    point of gamma = 2 - sqrt(2) -- so they must share one conductivity. The
    step midpoint is used, which is second-order for the composite step.
    """
    if scheme == TR_BDF2:
        return 0.5
    return resolve_implicit_heat_theta(scheme)


def _parallel_conductivities(Te, Ti, n, mu, ln_lambda_min, b_epara, b_ipara):
    """Return scaled volumetric parallel conductivities [erg cm^-1 s^-1].

    The Coulomb logarithm is an electron-ion quantity, so it is built from Te
    and shared by both species.
    """
    ln_lambda = np.maximum(c_log(Te, n, kind="ei"), ln_lambda_min)
    conductivity_e = (
        kappa_par_elec(Te, n, ln_lambda, per_particle=False)
        * ev_to_erg
        * float(b_epara)
    )
    conductivity_i = (
        kappa_par_ion(Ti, n, mu, ln_lambda, per_particle=False)
        * ev_to_erg
        * float(b_ipara)
    )
    return conductivity_e, conductivity_i


def _picard_converged(new, old, tol):
    scale = np.max(np.abs(new))
    if scale == 0.0:
        return True
    return bool(np.max(np.abs(new - old)) <= tol * scale)


def _species_heat_timestep(
    capacity, conductivity, geometry, fraction, active_cells=None
):
    face_coeff = np.zeros(geometry.cells + 1, dtype=float)
    k_face = 0.5 * (conductivity[:-1] + conductivity[1:])
    face_coeff[1:-1] = (
        geometry.plasma_face_area_cm2[1:-1]
        * k_face
        / geometry.center_distance_cm
        * geometry.heat_transmission[1:-1]
    )
    cell_coeff = (face_coeff[:-1] + face_coeff[1:]) / (
        geometry.plasma_volume_cm3 * capacity
    )
    active = cell_coeff > 0.0
    if active_cells is not None:
        active &= np.asarray(active_cells, dtype=bool)
    if not np.any(active):
        return np.inf
    return float(fraction / np.max(cell_coeff[active]))


def _implicit_species_energy(
    energy,
    capacity,
    temperature_floor,
    conductivity,
    geometry,
    dt,
    scheme="backward_euler",
):
    energy = np.asarray(energy, dtype=float)
    kwargs = dict(
        energy=energy,
        capacity=capacity,
        temperature_floor=temperature_floor,
        conductivity=conductivity,
        geometry=geometry,
        dt=dt,
    )
    if scheme == TR_BDF2:
        temperature = _tr_bdf2_temperature(**kwargs)
    else:
        temperature = _theta_temperature(
            theta=resolve_implicit_heat_theta(scheme),
            **kwargs,
        )
    # The single point at which the floor is applied, for every scheme.
    # scripts/audit_sim1d_floor_activation.py recovers the pre-clip temperature
    # by calling this function with temperature_floor=-inf, which only stays
    # valid while this remains the one place the floor is enforced.
    temperature = np.maximum(temperature, temperature_floor)
    return capacity * temperature


def _theta_temperature(
    energy,
    capacity,
    temperature_floor,
    conductivity,
    geometry,
    dt,
    theta,
):
    # The implicit operator carries theta*dt; the remaining (1 - theta)*dt of
    # the conduction is applied explicitly on the right-hand side below.
    banded = _banded_heat_operator(capacity, conductivity, geometry, theta * dt)
    rhs = energy
    if theta != 1.0:
        # _conductive_divergence is exactly -K*T_old over the same face
        # coefficients as the implicit operator, so the explicit and implicit
        # halves stay consistent by construction. theta=1 keeps rhs as the raw
        # conservative energy, reproducing the original backward-Euler solve
        # bit-for-bit.
        rhs = energy + (1.0 - theta) * dt * _conductive_divergence(
            np.maximum(energy / capacity, temperature_floor),
            conductivity,
            geometry,
        )
    return solve_banded((1, 1), banded, rhs)


def _tr_bdf2_temperature(
    energy,
    capacity,
    temperature_floor,
    conductivity,
    geometry,
    dt,
):
    old_temperature = np.maximum(energy / capacity, temperature_floor)
    # Both stages share this operator -- that is what gamma = 2 - sqrt(2) buys.
    banded = _banded_heat_operator(
        capacity,
        conductivity,
        geometry,
        _TR_BDF2_IMPLICIT * dt,
    )
    # Stage 1: trapezoidal rule out to t + gamma*dt, i.e. Crank-Nicolson over a
    # step of gamma*dt, whose implicit weight is (gamma/2)*dt = _TR_BDF2_IMPLICIT*dt.
    gamma_temperature = solve_banded(
        (1, 1),
        banded,
        energy
        + _TR_BDF2_IMPLICIT
        * dt
        * _conductive_divergence(old_temperature, conductivity, geometry),
    )
    # Stage 2: BDF2 through (T_old, T_gamma, T_new). The right-hand side needs
    # no flux evaluation -- it is just a blend of two known temperatures, and
    # _TR_BDF2_A + _TR_BDF2_B == 1 so a uniform temperature is preserved.
    # T_gamma is deliberately left unfloored: clipping between stages would
    # break the scheme's order, and the caller applies the floor once at the end.
    return solve_banded(
        (1, 1),
        banded,
        capacity * (_TR_BDF2_A * gamma_temperature + _TR_BDF2_B * old_temperature),
    )


def _banded_heat_operator(capacity, conductivity, geometry, dt):
    """Return ``C + dt*K`` in scipy banded form."""
    lower, diagonal, upper = _implicit_heat_diagonals(
        capacity=capacity,
        conductivity=conductivity,
        geometry=geometry,
        dt=dt,
    )
    banded = np.zeros((3, geometry.cells), dtype=float)
    banded[0, 1:] = upper
    banded[1, :] = diagonal
    banded[2, :-1] = lower
    return banded


def _conductive_divergence(temperature, conductivity, geometry):
    """Return ``-K*T`` for the frozen-conductivity operator [erg cm^-3 s^-1]."""
    return flux_divergence_rhs(
        conductive_face_flux(
            temperature=temperature,
            conductivity=conductivity,
            geometry=geometry,
        ),
        geometry,
    )


def _implicit_heat_diagonals(capacity, conductivity, geometry, dt):
    face_coeff = np.zeros(geometry.cells + 1, dtype=float)
    k_face = 0.5 * (conductivity[:-1] + conductivity[1:])
    # Must carry the same throttle as the explicit conductive_face_flux, or the
    # implicit substep would conduct across walls the explicit path blocks.
    face_coeff[1:-1] = (
        geometry.plasma_face_area_cm2[1:-1]
        * k_face
        / geometry.center_distance_cm
        * geometry.heat_transmission[1:-1]
    )
    left = dt * face_coeff[:-1] / geometry.plasma_volume_cm3
    right = dt * face_coeff[1:] / geometry.plasma_volume_cm3
    diagonal = capacity + left + right
    lower = -left[1:]
    upper = -right[:-1]
    return lower, diagonal, upper
