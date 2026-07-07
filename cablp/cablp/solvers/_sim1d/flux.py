from dataclasses import dataclass

import numpy as np

from cablp.funcs._plasmaparams import v_ion_speed
from .state import ConservativeState1D, derive_state


@dataclass(frozen=True)
class PlasmaFaceFluxes1D:
    """Conservative plasma fluxes on cell faces."""

    n: np.ndarray
    M: np.ndarray
    Ee: np.ndarray
    Ei: np.ndarray


def ion_sound_speed(Te, mu):
    """Return the ion sound speed [cm/s] using the existing _sim3 convention."""
    return v_ion_speed(Te, mu)


def physical_fluxes(state, derived):
    """Return cell-centered physical fluxes for the conservative plasma fields."""
    return PlasmaFaceFluxes1D(
        n=state.n * derived.u,
        M=state.M * derived.u + derived.p,
        Ee=state.Ee * derived.u,
        Ei=state.Ei * derived.u,
    )


def rusanov_fluxes(state, floors, ion_mass_g, mu, geometry):
    """Build closed-boundary Rusanov fluxes for plasma conservative variables."""
    derived = derive_state(state, floors=floors, ion_mass_g=ion_mass_g)
    cell_flux = physical_fluxes(state, derived)
    cells = geometry.cells

    face_n = np.zeros(cells + 1, dtype=float)
    face_M = np.zeros(cells + 1, dtype=float)
    face_Ee = np.zeros(cells + 1, dtype=float)
    face_Ei = np.zeros(cells + 1, dtype=float)

    cs = ion_sound_speed(derived.Te, mu)
    amax = np.maximum(
        np.abs(derived.u[:-1]) + cs[:-1],
        np.abs(derived.u[1:]) + cs[1:],
    )

    face_n[1:-1] = _rusanov_face(
        cell_flux.n[:-1], cell_flux.n[1:], state.n[:-1], state.n[1:], amax
    )
    face_M[1:-1] = _rusanov_face(
        cell_flux.M[:-1], cell_flux.M[1:], state.M[:-1], state.M[1:], amax
    )
    face_Ee[1:-1] = _rusanov_face(
        cell_flux.Ee[:-1], cell_flux.Ee[1:], state.Ee[:-1], state.Ee[1:], amax
    )
    face_Ei[1:-1] = _rusanov_face(
        cell_flux.Ei[:-1], cell_flux.Ei[1:], state.Ei[:-1], state.Ei[1:], amax
    )

    # Reflecting/closed external faces: no particle or thermal-energy flux, but
    # pressure acts on the wall so a uniform stationary state has zero divergence.
    face_M[0] = derived.p[0]
    face_M[-1] = derived.p[-1]
    return PlasmaFaceFluxes1D(n=face_n, M=face_M, Ee=face_Ee, Ei=face_Ei)


def front_filling_fluxes(state, floors, ion_mass_g, mu, geometry, alpha_front=1.0):
    """Return sonic-relaxation front-filling face fluxes."""
    if alpha_front < 0:
        raise ValueError(f"alpha_front must be non-negative (got {alpha_front})")

    derived = derive_state(state, floors=floors, ion_mass_g=ion_mass_g)
    cells = geometry.cells
    face_n = np.zeros(cells + 1, dtype=float)
    face_M = np.zeros(cells + 1, dtype=float)
    face_Ee = np.zeros(cells + 1, dtype=float)
    face_Ei = np.zeros(cells + 1, dtype=float)

    cs = ion_sound_speed(derived.Te, mu)
    raw_gamma = state.n[:-1] * cs[:-1] - state.n[1:] * cs[1:]
    cap = alpha_front * np.maximum(state.n[:-1] * cs[:-1], state.n[1:] * cs[1:])
    gamma = np.clip(raw_gamma, -cap, cap)
    donor_left = gamma >= 0.0

    u_donor = np.where(donor_left, derived.u[:-1], derived.u[1:])
    n_donor = np.where(donor_left, state.n[:-1], state.n[1:])
    Ee_donor = np.where(donor_left, state.Ee[:-1], state.Ee[1:])
    Ei_donor = np.where(donor_left, state.Ei[:-1], state.Ei[1:])
    energy_floor = np.maximum(n_donor, floors["n"])

    face_n[1:-1] = gamma
    face_M[1:-1] = ion_mass_g * gamma * u_donor
    face_Ee[1:-1] = gamma * Ee_donor / energy_floor
    face_Ei[1:-1] = gamma * Ei_donor / energy_floor
    return PlasmaFaceFluxes1D(n=face_n, M=face_M, Ee=face_Ee, Ei=face_Ei)


def plasma_flux_rhs(
    state,
    floors,
    ion_mass_g,
    mu,
    geometry,
    include_front=True,
    alpha_front=1.0,
):
    """Return finite-volume RHS from conservative plasma face fluxes."""
    flux_terms = plasma_flux_rhs_terms(
        state=state,
        floors=floors,
        ion_mass_g=ion_mass_g,
        mu=mu,
        geometry=geometry,
        include_front=include_front,
        alpha_front=alpha_front,
    )
    return _add_state_rhs(
        flux_terms["plasma_advective_flux"],
        flux_terms["plasma_front_flux"],
    )


def plasma_flux_rhs_terms(
    state,
    floors,
    ion_mass_g,
    mu,
    geometry,
    include_front=True,
    alpha_front=1.0,
):
    """Return separately named conservative RHS terms from plasma face fluxes."""
    rusanov = rusanov_fluxes(
        state=state,
        floors=floors,
        ion_mass_g=ion_mass_g,
        mu=mu,
        geometry=geometry,
    )
    front = _zero_fluxes(geometry.cells)
    if include_front:
        front = front_filling_fluxes(
            state=state,
            floors=floors,
            ion_mass_g=ion_mass_g,
            mu=mu,
            geometry=geometry,
            alpha_front=alpha_front,
        )
    return {
        "plasma_advective_flux": _flux_rhs(rusanov, geometry),
        "plasma_front_flux": _flux_rhs(front, geometry),
    }


def _flux_rhs(fluxes, geometry):
    return ConservativeState1D(
        n=_flux_divergence(fluxes.n, geometry),
        nn=np.zeros(geometry.cells, dtype=float),
        M=_flux_divergence(fluxes.M, geometry),
        Ee=_flux_divergence(fluxes.Ee, geometry),
        Ei=_flux_divergence(fluxes.Ei, geometry),
    )


def _rusanov_face(flux_l, flux_r, state_l, state_r, amax):
    return 0.5 * (flux_l + flux_r) - 0.5 * amax * (state_r - state_l)


def _flux_divergence(face_flux, geometry):
    inventory_flux = geometry.plasma_face_area_cm2 * face_flux
    return -(inventory_flux[1:] - inventory_flux[:-1]) / geometry.plasma_volume_cm3


def _zero_fluxes(cells):
    zeros = np.zeros(cells + 1, dtype=float)
    return PlasmaFaceFluxes1D(
        n=zeros,
        M=zeros.copy(),
        Ee=zeros.copy(),
        Ei=zeros.copy(),
    )


def _add_state_rhs(left, right):
    return ConservativeState1D(
        n=left.n + right.n,
        nn=left.nn + right.nn,
        M=left.M + right.M,
        Ee=left.Ee + right.Ee,
        Ei=left.Ei + right.Ei,
    )
