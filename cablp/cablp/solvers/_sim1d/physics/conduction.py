import numpy as np
from scipy.linalg import solve_banded

from cablp.funcs._heat import kappa_par_elec, kappa_par_ion
from cablp.funcs._plasmaparams import c_log
from cablp.vars._cons import ev_to_erg

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

    qe_face = conductive_face_flux(
        temperature=derived.Te,
        conductivity=kappa_par_elec(
            derived.Te,
            n,
            ln_lambda,
            per_particle=False,
        )
        * ev_to_erg
        * float(b_epara),
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


def conductive_face_flux(temperature, conductivity, geometry):
    """Return signed internal-face conductive fluxes [erg cm^-2 s^-1]."""
    temperature = np.asarray(temperature, dtype=float)
    conductivity = np.asarray(conductivity, dtype=float)
    q_face = np.zeros(geometry.cells + 1, dtype=float)
    k_face = 0.5 * (conductivity[:-1] + conductivity[1:])
    q_face[1:-1] = -k_face * np.diff(temperature) / geometry.center_distance_cm
    return q_face


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
):
    """Return an explicit diffusion timestep bound for heat conduction."""
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
):
    """Return a state after one frozen-conductivity implicit heat step.

    ``implicit_heat_scheme`` names one of ``IMPLICIT_HEAT_SCHEMES``: a theta
    method (``backward_euler``, ``shifted``, ``crank_nicolson``) or ``tr_bdf2``.
    Conductivity is frozen at the incoming state for every scheme, so a
    higher-order substep improves that substep's truncation error but does not
    by itself make the split step second-order accurate in time -- the frozen
    conductivity and the Lie splitting are both first-order regardless.
    """
    if dt <= 0.0:
        raise ValueError(f"dt must be positive (got {dt})")
    scheme = validate_implicit_heat_scheme(implicit_heat_scheme)
    if not heat_conduction or (b_epara == 0.0 and b_ipara == 0.0):
        return ConservativeState1D(
            n=state.n.copy(),
            nn=state.nn.copy(),
            M=state.M.copy(),
            Ee=state.Ee.copy(),
            Ei=state.Ei.copy(),
        )

    derived = derive_state(state, floors=floors, ion_mass_g=ion_mass_g)
    n = np.maximum(state.n, floors["n"])
    ln_lambda = np.maximum(c_log(derived.Te, n, kind="ei"), ln_lambda_min)
    capacity = 1.5 * n * ev_to_erg

    Ee = _implicit_species_energy(
        energy=state.Ee,
        capacity=capacity,
        temperature_floor=floors["Te"],
        conductivity=kappa_par_elec(
            derived.Te,
            n,
            ln_lambda,
            per_particle=False,
        )
        * ev_to_erg
        * float(b_epara),
        geometry=geometry,
        dt=dt,
        scheme=scheme,
    )
    Ei = _implicit_species_energy(
        energy=state.Ei,
        capacity=capacity,
        temperature_floor=floors["Ti"],
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
        dt=dt,
        scheme=scheme,
    )
    return ConservativeState1D(
        n=state.n.copy(),
        nn=state.nn.copy(),
        M=state.M.copy(),
        Ee=Ee,
        Ei=Ei,
    )


def _species_heat_timestep(capacity, conductivity, geometry, fraction):
    face_coeff = np.zeros(geometry.cells + 1, dtype=float)
    k_face = 0.5 * (conductivity[:-1] + conductivity[1:])
    face_coeff[1:-1] = (
        geometry.plasma_face_area_cm2[1:-1]
        * k_face
        / geometry.center_distance_cm
    )
    cell_coeff = (face_coeff[:-1] + face_coeff[1:]) / (
        geometry.plasma_volume_cm3 * capacity
    )
    active = cell_coeff > 0.0
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
    face_coeff[1:-1] = (
        geometry.plasma_face_area_cm2[1:-1]
        * k_face
        / geometry.center_distance_cm
    )
    left = dt * face_coeff[:-1] / geometry.plasma_volume_cm3
    right = dt * face_coeff[1:] / geometry.plasma_volume_cm3
    diagonal = capacity + left + right
    lower = -left[1:]
    upper = -right[:-1]
    return lower, diagonal, upper
