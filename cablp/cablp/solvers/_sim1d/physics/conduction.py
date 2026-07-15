import numpy as np
from scipy.linalg import solve_banded

from cablp.funcs._heat import kappa_par_elec, kappa_par_ion
from cablp.funcs._plasmaparams import c_log
from cablp.vars._cons import ev_to_erg

from ..core.state import ConservativeState1D, derive_state

# Theta weights for the implicit heat substep. The substep solves
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


def resolve_implicit_heat_theta(scheme):
    """Return the theta weight for a named implicit heat-conduction scheme."""
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

    ``implicit_heat_scheme`` selects the theta weight via
    ``IMPLICIT_HEAT_SCHEME_THETA``. Conductivity is frozen at the incoming
    state regardless of theta, so schemes below theta=1 improve the substep's
    truncation error but do not by themselves make the split step second-order
    accurate in time.
    """
    if dt <= 0.0:
        raise ValueError(f"dt must be positive (got {dt})")
    theta = resolve_implicit_heat_theta(implicit_heat_scheme)
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
        theta=theta,
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
        theta=theta,
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
    theta=1.0,
):
    energy = np.asarray(energy, dtype=float)
    # The implicit operator carries theta*dt; the remaining (1 - theta)*dt of
    # the conduction is applied explicitly on the right-hand side below.
    lower, diagonal, upper = _implicit_heat_diagonals(
        capacity=capacity,
        conductivity=conductivity,
        geometry=geometry,
        dt=theta * dt,
    )
    banded = np.zeros((3, geometry.cells), dtype=float)
    banded[0, 1:] = upper
    banded[1, :] = diagonal
    banded[2, :-1] = lower

    rhs = energy
    if theta != 1.0:
        # flux_divergence_rhs(conductive_face_flux(...)) is exactly -K*T_old,
        # built from the same face coefficients as _implicit_heat_diagonals, so
        # the explicit and implicit halves stay consistent by construction.
        # theta=1 keeps rhs as the raw conservative energy, reproducing the
        # previous backward-Euler solve bit-for-bit.
        old_temperature = np.maximum(energy / capacity, temperature_floor)
        rhs = energy + (1.0 - theta) * dt * flux_divergence_rhs(
            conductive_face_flux(
                temperature=old_temperature,
                conductivity=conductivity,
                geometry=geometry,
            ),
            geometry,
        )

    temperature = solve_banded((1, 1), banded, rhs)
    temperature = np.maximum(temperature, temperature_floor)
    return capacity * temperature


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
