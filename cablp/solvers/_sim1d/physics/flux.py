from dataclasses import dataclass
import math

import numpy as np

from cablp.plasma.params import v_ion_speed
from ..core.state import ConservativeState1D, derive_state
from cablp.constants import ev_to_erg


@dataclass(frozen=True)
class PlasmaFaceFluxes1D:
    """Conservative plasma fluxes on cell faces."""

    n: np.ndarray
    M: np.ndarray
    Ee: np.ndarray
    Ei: np.ndarray


def ion_sound_speed(Te, mu):
    """Return the ion sound speed [cm/s] using the existing _sim3 convention.

    Built on ``mu`` proton masses (``9.79e5 * sqrt(Te/mu)``, i.e. an implied
    ion mass ``mu * m_p``), not the true ion mass ``ion_mass_g`` (m_He) that
    other terms use directly -- a fixed ~0.600% residual in ``m_i c^2``
    against ``Te`` at mu=4.
    """
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
):
    """Impose closed-face conditions on every face with ``plasma_open`` False.

    A closed face carries no particle or thermal-energy flux, but pressure acts on
    it so a uniform stationary state still has zero divergence. This generalizes
    the historical external-end-only walls (the plasma domain is now bounded
    *inside* the neutral domain by the cathode surfaces); the pressure comes
    from the live plasma cell, which for the external ends is cell 0 and cell -1
    exactly as before.

    Absorbing surfaces are closed here too, and their loss is applied one-sidedly
    by ``sources.characteristic_boundary_rhs``. It cannot be a face flux: the flux
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
            # Historical selection, retained exactly on the
            # ``active_plasma_topology=False`` path; the R1 topology repair is
            # on by default in the current package.
            roles = np.asarray(geometry.cell_role)
            dead = ~np.asarray(geometry.plasma_active, dtype=bool)
            left, right = face - 1, face
            live_is_right = left < 0 or (right < cells and not dead[right])
            live = right if live_is_right else left
            face_M[face] = pressure[live]
    # The plasma-terminating (absorbing) faces are handled by the one-sided
    # characteristic ghost-cell Bohm outflow (sources.characteristic_boundary_
    # rhs), which supplies the particle, momentum, and energy flux AND its own
    # pressure term ``M_g u_g + p_g``. So the advective flux must carry NOTHING
    # here -- keeping the reflecting closed-wall pressure ``pressure[live]`` on
    # top would double-count the wall momentum. Unconditional since the legacy
    # volumetric absorber and its reflecting-wall alternative were retired;
    # see commit 1fc05c9.
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
    """Return the R2 KEP/Rusanov face flux (Γ_n, Γ_M, Γ_Ee, Γ_Ei) for one face.

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


#: The Riemann face solvers ``end_wall_face_riemann_solver`` accepts.
END_WALL_FACE_RIEMANN_SOLVERS = ("exact_isothermal", "hll")


def end_wall_riemann_face_scalar(
    left,
    right,
    solver,
    mu,
    ion_mass_g,
    wave_speed="isothermal",
    energy_consistent=False,
):
    """Return one face's Riemann flux (Γ_n, Γ_M, Γ_Ee, Γ_Ei) for the end wall.

    ``left`` and ``right`` carry the same scalars ``kep_rusanov_face_scalar``
    reads (``n, M, Ee, Ei, u, p, Te, Ti``), the L (low-z) and R (high-z) states
    of a +z-oriented face. ``solver`` selects between the two implemented face
    solvers and must be one of :data:`END_WALL_FACE_RIEMANN_SOLVERS`; any other
    value raises ``ValueError`` naming the accepted set.

    Every solver here returns ALL FOUR fluxes from ONE face state, so the
    particle flux the caller books and the energy fluxes that ride it describe
    the same face.

    ``wave_speed`` and ``energy_consistent`` are read by ``"hll"`` only.
    ``"exact_isothermal"`` solves its own Riemann problem, whose sound speed is
    the isothermal one by definition, and evaluates the flux at a SINGLE face
    state, where the two-point kinetic-energy-preserving momentum average
    degenerates to the physical product ``M u``.
    """
    if solver == "exact_isothermal":
        return exact_isothermal_face_scalar(
            left, right, mu=mu, ion_mass_g=ion_mass_g
        )
    if solver == "hll":
        return hll_face_scalar(
            left,
            right,
            mu=mu,
            ion_mass_g=ion_mass_g,
            wave_speed=wave_speed,
            energy_consistent=energy_consistent,
        )
    raise ValueError(
        "end_wall_face_riemann_solver must be one of "
        f"{END_WALL_FACE_RIEMANN_SOLVERS} (got {solver!r})"
    )


def hll_face_scalar(
    left,
    right,
    mu,
    ion_mass_g,
    wave_speed="isothermal",
    energy_consistent=False,
):
    """Return the HLL face flux on the full ``(n, M, Ee, Ei)`` vector.

    The two signal speeds are the face's OWN, not a single symmetric bound:

        S_L = min(u_L - c_L, u_R - c_R),  S_R = max(u_L + c_L, u_R + c_R)

    with ``c`` from :func:`plasma_wave_speed` at the configured ``wave_speed``.
    With ``S_L >= 0`` or ``S_R <= 0`` the face is supersonic and the flux is the
    upwind physical flux of that side; otherwise it is the HLL average

        (S_R F_L - S_L F_R + S_L S_R (U_R - U_L)) / (S_R - S_L)

    written as ``w_L F_L + w_R F_R + d (U_R - U_L)`` with ``w_L + w_R = 1``.
    Setting ``S_R = -S_L = a_max`` recovers ``kep_rusanov_face_scalar`` term by
    term, which is what makes the two comparable at the same face.

    ``energy_consistent`` selects the convective momentum flux the same way the
    Rusanov path does: the kinetic-energy-preserving product of the two weighted
    means, ``(w_L M_L + w_R M_R)(w_L u_L + w_R u_R)``, in place of the weighted
    mean of the two physical convective fluxes. At ``w = 1/2`` that is the
    ``{u}{M}`` form exactly, and in either supersonic branch (``w`` a unit
    vector) both forms are the single state's physical ``M u``.
    """
    nL, ML, EeL, EiL = left["n"], left["M"], left["Ee"], left["Ei"]
    nR, MR, EeR, EiR = right["n"], right["M"], right["Ee"], right["Ei"]
    uL, pL = left["u"], left["p"]
    uR, pR = right["u"], right["p"]

    csL = plasma_wave_speed(left["Te"], left["Ti"], mu, wave_speed)
    csR = plasma_wave_speed(right["Te"], right["Ti"], mu, wave_speed)
    sL = min(uL - csL, uR - csR)
    sR = max(uL + csL, uR + csR)

    if sL >= 0.0:
        wL, wR, diss = 1.0, 0.0, 0.0
    elif sR <= 0.0:
        wL, wR, diss = 0.0, 1.0, 0.0
    else:
        denom = sR - sL
        wL = sR / denom
        wR = -sL / denom
        diss = sL * sR / denom

    f_n = wL * (nL * uL) + wR * (nR * uR) + diss * (nR - nL)
    if energy_consistent:
        conv = (wL * ML + wR * MR) * (wL * uL + wR * uR)
    else:
        conv = wL * (ML * uL) + wR * (MR * uR)
    f_M = conv + (wL * pL + wR * pR) + diss * (MR - ML)
    f_Ee = wL * (EeL * uL) + wR * (EeR * uR) + diss * (EeR - EeL)
    f_Ei = wL * (EiL * uL) + wR * (EiR * uR) + diss * (EiR - EiL)
    return f_n, f_M, f_Ee, f_Ei


def exact_isothermal_face_scalar(left, right, mu, ion_mass_g):
    """Return the exact isothermal-Riemann face flux for one face.

    The Riemann problem solved is the isothermal Euler pair in ``(n, M)``

        d_t n + d_z (n u) = 0,   d_t M + d_z (M u + p) = 0,   p = n m_i c^2

    with ``c`` the face's isothermal sound speed :func:`ion_sound_speed`, i.e.
    the same ``sqrt(Te/m_i)`` the ghost state's Bohm velocity is set at. That
    pair carries no ion partial pressure, so where ``Ti`` is not negligible this
    momentum flux is smaller than the model's own ``n (Te + Ti)`` face pressure
    by ``n_f ((Te + Ti) - m_i c^2)`` -- exactly ``n_f Ti`` plus a small
    mass-convention residual (``m_i c^2`` uses the true ion mass ``ion_mass_g``
    against a sound speed built on ``mu`` proton masses; ~0.600% of Te at
    mu=4) -- the price of a closure whose Riemann problem has a closed-form
    solution. Both sides must name the same ``Te`` (one Riemann problem has one
    sound speed); a differing pair raises rather than picking a side.

    The system has two genuinely nonlinear fields and no contact, so the star
    region is a SINGLE state ``(n*, u*)``. The wave curves are

        rarefaction   u = u_K -/+ c ln(n*/n_K)
        shock         u = u_K -/+ c (n* - n_K)/sqrt(n* n_K)

    (upper sign the 1-wave off the left state, lower the 2-wave off the right),
    and their intersection is found by a bracketed Newton iteration in
    ``ln n*``. The self-similar solution is then sampled at ``x/t = 0`` -- which
    may land in either initial state, in the star state, or inside a rarefaction
    fan -- and the flux is the physical flux of THAT face state:
    ``f_n = n_f u_f`` and ``f_M = m_i n_f u_f^2 + m_i n_f c^2``.

    ``Ee`` and ``Ei`` ride the pair as passively advected scalars: their
    specific values ``E/n`` are constant along the linearly degenerate ``u``
    field, so each is upwinded on the face velocity and transported by the very
    particle flux above, ``f_E = (E/n)_upwind * f_n``. All four fluxes therefore
    come from one face state and one ``f_n``.

    Raises ``ValueError`` on a non-positive density on either side (the wave
    curves are logarithmic there) or if the star-state iteration fails to
    converge.
    """
    cL = ion_sound_speed(left["Te"], mu)
    cR = ion_sound_speed(right["Te"], mu)
    if cL != cR:
        raise ValueError(
            "the exact isothermal Riemann face requires one sound speed on the "
            f"face: got c_L={cL!r} from Te_L={left['Te']!r} and c_R={cR!r} from "
            f"Te_R={right['Te']!r}"
        )
    c = cL
    nL, uL = left["n"], left["u"]
    nR, uR = right["n"], right["u"]
    if not (nL > 0.0 and nR > 0.0):
        raise ValueError(
            "the exact isothermal Riemann face requires a positive density on "
            f"both sides (got n_L={nL!r}, n_R={nR!r})"
        )

    n_star, u_star = _isothermal_star_state(nL, uL, nR, uR, c)
    n_f, u_f = _isothermal_face_state(nL, uL, nR, uR, n_star, u_star, c)

    f_n = n_f * u_f
    f_M = ion_mass_g * n_f * u_f * u_f + ion_mass_g * n_f * c * c
    upwind = left if u_f >= 0.0 else right
    f_Ee = (upwind["Ee"] / upwind["n"]) * f_n
    f_Ei = (upwind["Ei"] / upwind["n"]) * f_n
    return f_n, f_M, f_Ee, f_Ei


def _isothermal_wave(n_star, n_k, c):
    """Return ``(g, dg/dn*)`` of one isothermal wave curve.

    ``g`` is the velocity change across the wave in units of ``c``, signed so
    that the 1-wave gives ``u* = u_L - c g_L`` and the 2-wave ``u* = u_R +
    c g_R``: the logarithmic Riemann invariant for an expansion
    (``n* <= n_k``) and the Rankine-Hugoniot locus for a compression.
    """
    if n_star <= n_k:
        return math.log(n_star / n_k), 1.0 / n_star
    root = math.sqrt(n_star * n_k)
    return (n_star - n_k) / root, 0.5 * (1.0 + n_k / n_star) / root


def _isothermal_star_state(nL, uL, nR, uR, c):
    """Return the star state ``(n*, u*)`` of the isothermal Riemann problem.

    ``F(n*) = c (g_L + g_R) - (u_L - u_R)`` is strictly increasing in ``n*`` and
    crosses zero exactly once on ``n* > 0``, so the iteration is a Newton step
    in ``ln n*`` kept inside a bracket that the sign of ``F`` tightens at every
    pass. The first guess is the two-rarefaction solution, which is the ANSWER
    whenever both waves expand -- the case an outflow boundary is in.
    """
    n_star = math.sqrt(nL * nR) * math.exp((uL - uR) / (2.0 * c))
    lo, hi = 0.0, math.inf
    for _ in range(100):
        gL, dgL = _isothermal_wave(n_star, nL, c)
        gR, dgR = _isothermal_wave(n_star, nR, c)
        F = c * (gL + gR) - (uL - uR)
        if F > 0.0:
            hi = n_star
        else:
            lo = n_star
        dF = c * (dgL + dgR)
        candidate = n_star * math.exp(-F / (dF * n_star))
        if not math.isfinite(candidate) or candidate <= lo or candidate >= hi:
            if lo > 0.0 and math.isfinite(hi):
                candidate = math.sqrt(lo * hi)
            else:
                candidate = n_star * (0.5 if F > 0.0 else 2.0)
        converged = abs(candidate - n_star) <= 1.0e-15 * n_star
        n_star = candidate
        if converged:
            break
    else:
        raise ValueError(
            "the isothermal Riemann star state did not converge for "
            f"n_L={nL!r}, u_L={uL!r}, n_R={nR!r}, u_R={uR!r}, c={c!r}"
        )
    gL, _ = _isothermal_wave(n_star, nL, c)
    gR, _ = _isothermal_wave(n_star, nR, c)
    u_star = 0.5 * ((uL - c * gL) + (uR + c * gR))
    return n_star, u_star


def _isothermal_face_state(nL, uL, nR, uR, n_star, u_star, c):
    """Return ``(n, u)`` of the self-similar solution sampled at ``x/t = 0``.

    ``u*`` is the velocity of the star region, so its sign says which of the two
    waves the face sits behind; that wave is then a shock (one speed) or a
    rarefaction (a head and a tail speed, with the fan state in between given by
    the field's Riemann invariant at ``u = +/- c``).
    """
    if u_star >= 0.0:
        if n_star > nL:
            speed = uL - c * math.sqrt(n_star / nL)
            return (nL, uL) if speed >= 0.0 else (n_star, u_star)
        if uL - c >= 0.0:
            return nL, uL
        if u_star - c <= 0.0:
            return n_star, u_star
        return nL * math.exp(uL / c - 1.0), c
    if n_star > nR:
        speed = uR + c * math.sqrt(n_star / nR)
        return (nR, uR) if speed <= 0.0 else (n_star, u_star)
    if uR + c <= 0.0:
        return nR, uR
    if u_star + c >= 0.0:
        return n_star, u_star
    return nR * math.exp(-uR / c - 1.0), -c
