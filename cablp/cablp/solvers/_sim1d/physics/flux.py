from dataclasses import dataclass

import numpy as np

from cablp.funcs._plasmaparams import v_ion_speed
from ..core.geometry import PLASMA_DEAD_ROLES
from ..core.state import ConservativeState1D, derive_state
from cablp.vars._cons import ev_to_erg


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
    raw = _rusanov_raw_faces(state, derived, mu, geometry)
    return _apply_face_conditions(raw, geometry, derived.p)


def _rusanov_raw_faces(state, derived, mu, geometry):
    """Return interior Rusanov faces *before* transmission or wall conditions.

    Kept separate because the intercepted (blocked) part of the raw flux is what
    the anode absorbs, so it must be known before transmission is applied.
    """
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
    return PlasmaFaceFluxes1D(n=face_n, M=face_M, Ee=face_Ee, Ei=face_Ei)


def _apply_face_conditions(faces, geometry, pressure):
    """Apply partial-blocking transmission and reflecting walls to raw faces.

    Partially blocking faces (the anode mesh) transmit only their open fraction;
    the intercepted remainder is removed by the interception term, so the mesh
    absorbs rather than reflects. Fully open faces scale by exactly 1.0.
    """
    transmission = geometry.plasma_transmission
    face_n = faces.n * transmission
    face_M = faces.M * transmission
    face_Ee = faces.Ee * transmission
    face_Ei = faces.Ei * transmission
    _apply_plasma_walls(
        geometry=geometry,
        pressure=pressure,
        face_n=face_n,
        face_M=face_M,
        face_Ee=face_Ee,
        face_Ei=face_Ei,
    )
    return PlasmaFaceFluxes1D(n=face_n, M=face_M, Ee=face_Ee, Ei=face_Ei)


def _apply_plasma_walls(geometry, pressure, face_n, face_M, face_Ee, face_Ei):
    """Impose reflecting-wall conditions on every face with ``plasma_open`` False.

    A wall carries no particle or thermal-energy flux, but pressure acts on it so
    a uniform stationary state still has zero divergence. This generalizes the
    historical external-end-only walls (the plasma domain is now bounded *inside*
    the neutral domain by the cathode surfaces, §5); the pressure is taken from
    the live plasma cell, which for the external ends is cell 0 and cell -1
    exactly as before.
    """
    roles = np.asarray(geometry.cell_role)
    dead = np.asarray([role in PLASMA_DEAD_ROLES for role in roles], dtype=bool)
    cells = roles.size
    for face in np.flatnonzero(~np.asarray(geometry.plasma_open, dtype=bool)):
        face = int(face)
        left, right = face - 1, face
        live_is_right = left < 0 or (right < cells and not dead[right])
        live = right if live_is_right else left
        face_n[face] = 0.0
        face_Ee[face] = 0.0
        face_Ei[face] = 0.0
        face_M[face] = pressure[live]


def front_filling_fluxes(state, floors, ion_mass_g, mu, geometry, alpha_front=1.0):
    """Return sonic-relaxation front-filling face fluxes."""
    raw = _front_raw_faces(
        state=state,
        floors=floors,
        ion_mass_g=ion_mass_g,
        mu=mu,
        geometry=geometry,
        alpha_front=alpha_front,
    )
    return _apply_front_conditions(raw, geometry)


def _front_raw_faces(state, floors, ion_mass_g, mu, geometry, alpha_front=1.0):
    """Return front-filling faces before transmission or wall closure."""
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


def _apply_front_conditions(faces, geometry):
    """Apply transmission and wall closure to raw front-filling faces.

    Unlike the advective flux, a wall carries *no* front flux at all -- the wall's
    momentum is the pressure term in the advective flux, not here.
    """
    transmission = geometry.plasma_transmission
    face_n = faces.n * transmission
    face_M = faces.M * transmission
    face_Ee = faces.Ee * transmission
    face_Ei = faces.Ei * transmission
    walls = ~np.asarray(geometry.plasma_open, dtype=bool)
    face_n[walls] = 0.0
    face_M[walls] = 0.0
    face_Ee[walls] = 0.0
    face_Ei[walls] = 0.0
    return PlasmaFaceFluxes1D(n=face_n, M=face_M, Ee=face_Ee, Ei=face_Ei)


def _front_fluxes(
    state, floors, ion_mass_g, mu, geometry, alpha_front, pressure=None
):
    """Return ``(raw, transmitted)`` front-filling faces."""
    raw = _front_raw_faces(
        state=state,
        floors=floors,
        ion_mass_g=ion_mass_g,
        mu=mu,
        geometry=geometry,
        alpha_front=alpha_front,
    )
    return raw, _apply_front_conditions(raw, geometry)


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
        _add_state_rhs(
            flux_terms["plasma_advective_flux"],
            flux_terms["plasma_front_flux"],
        ),
        flux_terms["anode_interception"],
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
    derived = derive_state(state, floors=floors, ion_mass_g=ion_mass_g)
    raw = _rusanov_raw_faces(state, derived, mu, geometry)
    rusanov = _apply_face_conditions(raw, geometry, derived.p)
    raw_front = _zero_fluxes(geometry.cells)
    front = raw_front
    if include_front:
        raw_front, front = _front_fluxes(
            state=state,
            floors=floors,
            ion_mass_g=ion_mass_g,
            mu=mu,
            geometry=geometry,
            alpha_front=alpha_front,
            pressure=derived.p,
        )
    # The anode intercepts whatever crosses it, advective and front alike, so the
    # sink is driven by their combined incident particle flux.
    return {
        "plasma_advective_flux": _flux_rhs(rusanov, geometry),
        "plasma_front_flux": _flux_rhs(front, geometry),
        "anode_interception": anode_interception_rhs(
            raw_face_n=raw.n + raw_front.n,
            derived=derived,
            geometry=geometry,
            ion_mass_g=ion_mass_g,
        ),
    }


def anode_interception_rhs(raw_face_n, derived, geometry, ion_mass_g):
    """Return the plasma absorbed by a partially blocking face (the anode mesh).

    Plan §11 decision 7: the anode's plasma effect is the *directed* flux it
    intercepts, so the face transmits ``(1-eta)`` and the blocked ``eta`` fraction
    is absorbed here rather than reflected. Removing it from the donor cell is
    what makes the pair conservative -- the donor loses the full incident flux
    (transmitted plus intercepted) while the far side receives only the
    transmitted part.

    The absorbed ions neutralize on the mesh and leave as gas. A wire has no
    memory of which side an ion arrived from, so the neutrals are emitted to both
    sides and are split evenly across the two flanking cells (§7); this matters
    because the mesh throttles neutral flow, so the side a neutral is born on
    decides whether it fuels the column or heads for the pump.

    Mass, momentum and thermal energy are removed together exactly as
    ``surface_neutralization_rhs`` does at a wall. The intercepted momentum is
    absorbed by the grounded anode structure -- a sink, not ion heating (§5).
    """
    zeros = np.zeros(geometry.cells, dtype=float)
    transmission = np.asarray(geometry.plasma_transmission, dtype=float)
    blocking = np.flatnonzero(
        np.asarray(geometry.plasma_open, dtype=bool) & (transmission < 1.0)
    )
    if blocking.size == 0:
        return ConservativeState1D(
            n=zeros,
            nn=zeros.copy(),
            M=zeros.copy(),
            Ee=zeros.copy(),
            Ei=zeros.copy(),
        )

    dN_loss = np.zeros(geometry.cells, dtype=float)
    dN_gain = np.zeros(geometry.cells, dtype=float)
    for face in blocking:
        face = int(face)
        flux = float(raw_face_n[face])
        # Particles/s intercepted by the solid fraction of this face.
        absorbed = (
            abs(flux)
            * (1.0 - transmission[face])
            * float(geometry.plasma_face_area_cm2[face])
        )
        if absorbed == 0.0:
            continue
        # The donor is whichever side the flow comes from.
        donor = face - 1 if flux > 0.0 else face
        dN_loss[donor] += absorbed
        dN_gain[face - 1] += 0.5 * absorbed
        dN_gain[face] += 0.5 * absorbed

    plasma_loss_rate = dN_loss / geometry.plasma_volume_cm3
    neutral_gain_rate = dN_gain / geometry.neutral_volume_cm3
    return ConservativeState1D(
        n=-plasma_loss_rate,
        nn=neutral_gain_rate,
        M=-ion_mass_g * derived.u * plasma_loss_rate,
        Ee=-1.5 * ev_to_erg * derived.Te * plasma_loss_rate,
        Ei=-1.5 * ev_to_erg * derived.Ti * plasma_loss_rate,
    )


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
