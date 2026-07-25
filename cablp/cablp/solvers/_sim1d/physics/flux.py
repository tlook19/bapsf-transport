from dataclasses import dataclass

import numpy as np

from cablp.funcs._plasmaparams import v_ion_speed
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


def plasma_wave_speed(Te, Ti, mu, wave_speed="isothermal"):
    """Return the plasma signal speed [cm/s] for the Rusanov a_max and CFL.

    ``"isothermal"`` is the historical gamma=1 electron-pressure Bohm speed
    ``sqrt(Te/m_i)`` used by both the Rusanov dissipation and the plasma CFL;
    this branch is a bit-exact passthrough of ``ion_sound_speed``.
    ``"adiabatic"`` is the exact linear acoustic speed of the implemented
    gamma=5/3 two-species ideal-gas energy system,
    ``sqrt((5/3)(Te+Ti)/m_i)`` -- the R2 spectral-radius repair (audit A3),
    which also restores the wave bound Rusanov positivity relies on.
    """
    if wave_speed == "isothermal":
        return ion_sound_speed(Te, mu)
    if wave_speed == "adiabatic":
        return v_ion_speed(Te + Ti, mu, gamma=5.0 / 3.0)
    raise ValueError(
        f"wave_speed must be 'isothermal' or 'adiabatic' (got {wave_speed!r})"
    )


def physical_fluxes(state, derived):
    """Return cell-centered physical fluxes for the conservative plasma fields."""
    return PlasmaFaceFluxes1D(
        n=state.n * derived.u,
        M=state.M * derived.u + derived.p,
        Ee=state.Ee * derived.u,
        Ei=state.Ei * derived.u,
    )


def rusanov_fluxes(
    state, floors, ion_mass_g, mu, geometry, active_plasma_topology=False,
    wave_speed="isothermal", energy_consistent=False,
    characteristic_boundary=False,
):
    """Build closed-boundary Rusanov fluxes for plasma conservative variables."""
    derived = derive_state(state, floors=floors, ion_mass_g=ion_mass_g)
    raw = _rusanov_raw_faces(
        state, derived, mu, geometry, wave_speed=wave_speed,
        energy_consistent=energy_consistent,
    )
    return _apply_face_conditions(
        raw,
        geometry,
        derived.p,
        active_plasma_topology=active_plasma_topology,
        characteristic_boundary=characteristic_boundary,
    )


def _rusanov_raw_faces(
    state, derived, mu, geometry, wave_speed="isothermal",
    energy_consistent=False,
):
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

    cs = plasma_wave_speed(derived.Te, derived.Ti, mu, wave_speed)
    amax = np.maximum(
        np.abs(derived.u[:-1]) + cs[:-1],
        np.abs(derived.u[1:]) + cs[1:],
    )

    face_n[1:-1] = _rusanov_face(
        cell_flux.n[:-1], cell_flux.n[1:], state.n[:-1], state.n[1:], amax
    )
    if energy_consistent:
        # Kinetic-energy-preserving convective momentum flux (Jameson 2008):
        # the convective part {u}{M} = 0.25(u_L+u_R)(M_L+M_R) replaces the
        # divergence-form 0.5(M_L u_L + M_R u_R). The pressure {p} and the
        # Rusanov dissipation are unchanged. This makes the discrete advective
        # kinetic energy conserved; the R2 energy-correction term then closes
        # the total-energy identity (deposit + KEP pressure work).
        u = derived.u
        conv = 0.25 * (u[:-1] + u[1:]) * (state.M[:-1] + state.M[1:])
        pbar = 0.5 * (derived.p[:-1] + derived.p[1:])
        face_M[1:-1] = conv + pbar - 0.5 * amax * (state.M[1:] - state.M[:-1])
    else:
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


def _apply_face_conditions(
    faces, geometry, pressure, active_plasma_topology=False,
    characteristic_boundary=False,
):
    """Apply partial-blocking transmission and closed-face conditions to raw faces.

    Partially blocking faces (the anode mesh) transmit only their open fraction;
    fully open faces scale by exactly 1.0.
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
        active_plasma_topology=active_plasma_topology,
        characteristic_boundary=characteristic_boundary,
    )
    return PlasmaFaceFluxes1D(n=face_n, M=face_M, Ee=face_Ee, Ei=face_Ei)


def _apply_plasma_walls(
    geometry,
    pressure,
    face_n,
    face_M,
    face_Ee,
    face_Ei,
    active_plasma_topology=False,
    characteristic_boundary=False,
):
    """Impose closed-face conditions on every face with ``plasma_open`` False.

    A closed face carries no particle or thermal-energy flux, but pressure acts on
    it so a uniform stationary state still has zero divergence. This generalizes
    the historical external-end-only walls (the plasma domain is now bounded
    *inside* the neutral domain by the cathode surfaces, §5); the pressure comes
    from the live plasma cell, which for the external ends is cell 0 and cell -1
    exactly as before.

    Absorbing surfaces are closed here too, and their loss is applied one-sidedly
    by ``sources.boundary_absorption_rhs``. It cannot be a face flux: the flux
    array telescopes, so an *interior* absorbing face (a cathode surface) would
    hand the plasma it removes to the plenum behind it instead of out of the
    domain, and would kick a plasma-dead cell with sonic momentum.
    """
    cells = geometry.cells
    for face in np.flatnonzero(~np.asarray(geometry.plasma_open, dtype=bool)):
        face = int(face)
        face_n[face] = 0.0
        face_Ee[face] = 0.0
        face_Ei[face] = 0.0
        if active_plasma_topology:
            live = int(geometry.plasma_face_live_cell[face])
            face_M[face] = 0.0 if live < 0 else pressure[live]
        else:
            # Historical selection retained exactly while the R1 topology
            # repair is default off.
            roles = np.asarray(geometry.cell_role)
            dead = ~np.asarray(geometry.plasma_active, dtype=bool)
            left, right = face - 1, face
            live_is_right = left < 0 or (right < cells and not dead[right])
            live = right if live_is_right else left
            face_M[face] = pressure[live]
    if characteristic_boundary:
        # R3.1 (SIM1D_MODEL_AUDIT_PLAN "R3.1 boundary approach"): the plasma-
        # terminating (absorbing) faces are handled by the one-sided
        # characteristic ghost-cell Bohm outflow (sources.characteristic_
        # boundary_rhs), which supplies the particle, momentum, and energy flux
        # AND its own pressure term ``M_g u_g + p_g``. So the advective flux must
        # carry NOTHING here -- keeping the reflecting closed-wall pressure
        # ``pressure[live]`` on top would double-count the wall momentum. Default
        # off => the closed-wall condition above stands and the golden is exact.
        absorbing = np.asarray(
            getattr(geometry, "plasma_absorbing", np.zeros(0)), dtype=bool
        )
        for face in np.flatnonzero(absorbing):
            face = int(face)
            face_n[face] = 0.0
            face_M[face] = 0.0
            face_Ee[face] = 0.0
            face_Ei[face] = 0.0


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
    active_plasma_topology=False,
    wave_speed="isothermal",
    energy_consistent=False,
    characteristic_boundary=False,
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
        active_plasma_topology=active_plasma_topology,
        wave_speed=wave_speed,
        energy_consistent=energy_consistent,
        characteristic_boundary=characteristic_boundary,
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
    alpha_isat=np.exp(-0.5),
    active_plasma_topology=False,
    wave_speed="isothermal",
    energy_consistent=False,
    characteristic_boundary=False,
):
    """Return separately named conservative RHS terms from plasma face fluxes."""
    rusanov = rusanov_fluxes(
        state=state,
        floors=floors,
        ion_mass_g=ion_mass_g,
        mu=mu,
        geometry=geometry,
        active_plasma_topology=active_plasma_topology,
        wave_speed=wave_speed,
        energy_consistent=energy_consistent,
        characteristic_boundary=characteristic_boundary,
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


def kep_rusanov_face_scalar(
    left,
    right,
    mu,
    ion_mass_g,
    wave_speed="isothermal",
    energy_consistent=False,
):
    """Return the R2 KEP/Rusanov face flux (F_n, F_M, F_Ee, F_Ei) for one face.

    ``left`` and ``right`` are dicts with the conservative and derived scalars of
    the two states bracketing the face (``n, M, Ee, Ei, u, p, Te, Ti``), the L
    (low-z) and R (high-z) states of a +z-oriented face. The formula is the exact
    per-face expression of ``_rusanov_raw_faces`` (KEP convective momentum flux
    when ``energy_consistent`` else the divergence form; ``plasma_wave_speed`` for
    the dissipation ``a_max``), factored so the R3.1 characteristic ghost-cell
    boundary (``sources.characteristic_boundary_rhs``) reuses the committed R2
    machinery instead of re-deriving the flux.
    """
    nL, ML, EeL, EiL = left["n"], left["M"], left["Ee"], left["Ei"]
    nR, MR, EeR, EiR = right["n"], right["M"], right["Ee"], right["Ei"]
    uL, pL = left["u"], left["p"]
    uR, pR = right["u"], right["p"]

    csL = plasma_wave_speed(left["Te"], left["Ti"], mu, wave_speed)
    csR = plasma_wave_speed(right["Te"], right["Ti"], mu, wave_speed)
    amax = max(abs(uL) + csL, abs(uR) + csR)

    f_n = 0.5 * (nL * uL + nR * uR) - 0.5 * amax * (nR - nL)
    if energy_consistent:
        conv = 0.25 * (uL + uR) * (ML + MR)
    else:
        conv = 0.5 * (ML * uL + MR * uR)
    f_M = conv + 0.5 * (pL + pR) - 0.5 * amax * (MR - ML)
    f_Ee = 0.5 * (EeL * uL + EeR * uR) - 0.5 * amax * (EeR - EeL)
    f_Ei = 0.5 * (EiL * uL + EiR * uR) - 0.5 * amax * (EiR - EiL)
    return f_n, f_M, f_Ee, f_Ei
