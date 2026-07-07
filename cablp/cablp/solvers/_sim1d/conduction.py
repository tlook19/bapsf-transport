import numpy as np

from cablp.funcs._heat import kappa_par_elec, kappa_par_ion
from cablp.funcs._plasmaparams import c_log
from cablp.vars._cons import ev_to_erg

from .state import ConservativeState1D, derive_state


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
