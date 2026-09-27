from dataclasses import dataclass
import math

import numpy as np

from cablp.plasma.params import bohm_sound_speed
from ..core.state import ConservativeState1D, derive_state
from cablp.constants import ev_to_erg


@dataclass(frozen=True)
class PlasmaFaceFluxes1D:
    """Conservative plasma fluxes on cell faces."""

    n: np.ndarray
    M: np.ndarray
    Ee: np.ndarray
    Ei: np.ndarray


def ion_sound_speed(Te, ion_mass_g):
    """Return the Bohm (ion sound) speed ``sqrt(Te/m_i)`` [cm/s].

    ``Te`` is the electron temperature [eV] and ``ion_mass_g`` the ion mass
    [g] -- the same ``ion_mass_g`` every other term of the model carries, so
    the sound speed, the pressures and the momentum density describe one ion.
    The gamma=1 electron-pressure form: this is the speed the sheath-edge
    (Bohm) outflow is set at and the speed the presheath depth is built from.

    The expression is :func:`~cablp.plasma.params.bohm_sound_speed`, THE ONE
    SPEC, so the fluid boundary and the cathode circuit read one number.
    """
    return bohm_sound_speed(Te, ion_mass_g)


def plasma_wave_speed(Te, Ti, ion_mass_g):
    """Return the plasma signal speed [cm/s] for the Rusanov a_max and CFL.

    The exact linear acoustic speed of the implemented gamma=5/3 two-species
    ideal-gas energy system, ``sqrt((5/3)(Te+Ti)/m_i)`` with ``Te`` and ``Ti``
    in eV and ``ion_mass_g`` in g, which is the wave bound Rusanov positivity
    relies on. The gamma=1 electron-pressure Bohm speed ``sqrt(Te/m_i)`` is
    :func:`ion_sound_speed`; it sets the sheath-edge outflow and is not a
    signal-speed bound. The implemented system is the gamma=5/3 one because
    the energy rows carry ``-p_s div u`` and the discrete total-energy flux
    carries the enthalpy; the Fourier symbol of the assembled hyperbolic core
    has phase speeds ``u +- sqrt((5/3)(Te+Ti)/m_i)`` and ``u`` twice, which is
    what the ``kep-acoustic-symbol`` smoke case measures.
    """
    return np.sqrt((5.0 / 3.0) * (Te + Ti) * ev_to_erg / ion_mass_g)


def physical_fluxes(state, derived):
    """Return cell-centered physical fluxes for the conservative plasma fields."""
    return PlasmaFaceFluxes1D(
        n=state.n * derived.u,
        M=state.M * derived.u + derived.p,
        Ee=state.Ee * derived.u,
        Ei=state.Ei * derived.u,
    )


def rusanov_fluxes(state, floors, ion_mass_g, geometry):
    """Build closed-boundary Rusanov fluxes for plasma conservative variables."""
    derived = derive_state(state, floors=floors, ion_mass_g=ion_mass_g)
    raw = _rusanov_raw_faces(state, derived, ion_mass_g, geometry)
    return _apply_face_conditions(raw, geometry, derived.p)


def _rusanov_raw_faces(state, derived, ion_mass_g, geometry):
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

    cs = plasma_wave_speed(derived.Te, derived.Ti, ion_mass_g)
    amax = np.maximum(
        np.abs(derived.u[:-1]) + cs[:-1],
        np.abs(derived.u[1:]) + cs[1:],
    )

    face_n[1:-1] = _rusanov_face(
        cell_flux.n[:-1], cell_flux.n[1:], state.n[:-1], state.n[1:], amax
    )
    # Kinetic-energy-preserving convective momentum flux (Jameson 2008): the
    # convective part {u}{M} = 0.25(u_L+u_R)(M_L+M_R) in place of the
    # divergence-form 0.5(M_L u_L + M_R u_R), with the standard pressure {p}
    # and Rusanov dissipation. This makes the discrete advective kinetic
    # energy conserved; the dissipation deposit then returns to Ei what the
    # Rusanov dissipation took from K, and the -p_s div u pressure-work row
    # pairs with the net pressure force, so the total-energy identity closes
    # per cell.
    u = derived.u
    conv = 0.25 * (u[:-1] + u[1:]) * (state.M[:-1] + state.M[1:])
    pbar = 0.5 * (derived.p[:-1] + derived.p[1:])
    face_M[1:-1] = conv + pbar - 0.5 * amax * (state.M[1:] - state.M[:-1])
    face_Ee[1:-1] = _rusanov_face(
        cell_flux.Ee[:-1], cell_flux.Ee[1:], state.Ee[:-1], state.Ee[1:], amax
    )
    face_Ei[1:-1] = _rusanov_face(
        cell_flux.Ei[:-1], cell_flux.Ei[1:], state.Ei[:-1], state.Ei[1:], amax
    )
    return PlasmaFaceFluxes1D(n=face_n, M=face_M, Ee=face_Ee, Ei=face_Ei)


def _apply_face_conditions(faces, geometry, pressure):
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
    )
    return PlasmaFaceFluxes1D(n=face_n, M=face_M, Ee=face_Ee, Ei=face_Ei)


def _apply_plasma_walls(
    geometry,
    pressure,
    face_n,
    face_M,
    face_Ee,
    face_Ei,
):
    """Impose closed-face conditions on every face with ``plasma_open`` False.

    A closed face carries no particle or thermal-energy flux, but pressure acts on
    it so a uniform stationary state still has zero divergence. The plasma
    domain is bounded *inside* the neutral domain by the cathode surfaces, so a
    closed face may be interior; the pressure comes from the face's typed live
    plasma cell (``geometry.plasma_face_live_cell``), and a closed face with no
    live cell on either side carries no momentum flux.

    Absorbing surfaces are closed here too, and their loss is applied one-sidedly
    by ``sources.characteristic_boundary_rhs``. It cannot be a face flux: the flux
    array telescopes, so an *interior* absorbing face (a cathode surface) would
    hand the plasma it removes to the plenum behind it instead of out of the
    domain, and would kick a plasma-dead cell with sonic momentum.
    """
    for face in np.flatnonzero(~np.asarray(geometry.plasma_open, dtype=bool)):
        face = int(face)
        face_n[face] = 0.0
        face_Ee[face] = 0.0
        face_Ei[face] = 0.0
        live = int(geometry.plasma_face_live_cell[face])
        face_M[face] = 0.0 if live < 0 else pressure[live]
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


def plasma_flux_rhs(
    state,
    floors,
    ion_mass_g,
    geometry,
):
    """Return finite-volume RHS from conservative plasma face fluxes."""
    flux_terms = plasma_flux_rhs_terms(
        state=state,
        floors=floors,
        ion_mass_g=ion_mass_g,
        geometry=geometry,
    )
    return flux_terms["plasma_advective_flux"]


def plasma_flux_rhs_terms(
    state,
    floors,
    ion_mass_g,
    geometry,
    alpha_isat=np.exp(-0.5),
):
    """Return separately named conservative RHS terms from plasma face fluxes."""
    rusanov = rusanov_fluxes(
        state=state,
        floors=floors,
        ion_mass_g=ion_mass_g,
        geometry=geometry,
    )
    return {
        "plasma_advective_flux": _flux_rhs(rusanov, geometry),
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


def kep_rusanov_face_scalar(left, right, ion_mass_g):
    """Return the R2 KEP/Rusanov face flux (Γ_n, Γ_M, Γ_Ee, Γ_Ei) for one face.

    ``left`` and ``right`` are dicts with the conservative and derived scalars of
    the two states bracketing the face (``n, M, Ee, Ei, u, p, Te, Ti``), the L
    (low-z) and R (high-z) states of a +z-oriented face. The formula is the exact
    per-face expression of ``_rusanov_raw_faces`` (KEP convective momentum flux;
    ``plasma_wave_speed`` for the dissipation ``a_max``), factored so the R3.1 characteristic ghost-cell
    boundary (``sources.characteristic_boundary_rhs``) reuses the committed R2
    machinery instead of re-deriving the flux.
    """
    nL, ML, EeL, EiL = left["n"], left["M"], left["Ee"], left["Ei"]
    nR, MR, EeR, EiR = right["n"], right["M"], right["Ee"], right["Ei"]
    uL, pL = left["u"], left["p"]
    uR, pR = right["u"], right["p"]

    csL = plasma_wave_speed(left["Te"], left["Ti"], ion_mass_g)
    csR = plasma_wave_speed(right["Te"], right["Ti"], ion_mass_g)
    amax = max(abs(uL) + csL, abs(uR) + csR)

    f_n = 0.5 * (nL * uL + nR * uR) - 0.5 * amax * (nR - nL)
    conv = 0.25 * (uL + uR) * (ML + MR)
    f_M = conv + 0.5 * (pL + pR) - 0.5 * amax * (MR - ML)
    f_Ee = 0.5 * (EeL * uL + EeR * uR) - 0.5 * amax * (EeR - EeL)
    f_Ei = 0.5 * (EiL * uL + EiR * uR) - 0.5 * amax * (EiR - EiL)
    return f_n, f_M, f_Ee, f_Ei


def physical_face_scalar(face):
    """Return the physical flux (Γ_n, Γ_M, Γ_Ee, Γ_Ei) of ONE state.

    ``face`` is a dict with the conservative and derived scalars
    (``n, M, Ee, Ei, u, p``) of the state the flux is evaluated at, and the
    flux returned is the +z-oriented one,

        f_n = n u,  f_M = M u + p,  f_Ee = Ee u,  f_Ei = Ei u.

    This is the two-point kernel's own physical flux with both of its states
    set to this one, so the KEP momentum average degenerates to the product
    ``M u`` and every dissipation term vanishes. It is what a MATERIAL
    surface removes: the boundary operator evaluates it at the sheath-edge
    (Bohm) state alone, since a sheath sends no wave back into the plasma and
    the density step across the sub-grid presheath is a model, not a
    discontinuity for a Riemann solver to resolve.
    """
    u = face["u"]
    return (
        face["n"] * u,
        face["M"] * u + face["p"],
        face["Ee"] * u,
        face["Ei"] * u,
    )
