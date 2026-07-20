"""Solver-agnostic beam deposition along a ray (BEAM_DEPOSITION_PLAN B1).

Deterministic CSDA (continuous-slowing-down) integration of a monoenergetic
primary-electron beam through the column. **Pure function of the beam and the
column** — ``(E0, Gamma0, nn, ne, Te, ray)`` — with no solver state, so both
the voltage-driven and current-driven cathode formulations can consume it at
the same call sites (plan §2).

Energy channels, per unit path length [eV/cm]:

- ionization: ``nn * sigma_ion(E) * (I_ion + <W_sec>(E))`` — the potential
  I_ion is banked as ionization cost (matching the solver's separate
  ``beam_ionization_cost`` term); the mean secondary energy <W_sec> is banked
  as plasma heating (the ejected electron thermalizes locally). <W_sec> uses
  the Opal-Peterson-Beaty differential shape ``dsigma/dW ~ 1/(W^2 + Ebar^2)``
  with the He shape parameter Ebar = 15.8 eV (Opal, Peterson & Beaty, At.
  Data 4, 209 (1971)) over W in [0, (E - I_ion)/2].
- excitation: ``nn * sigma_exc(E) * E_rad(E)`` — the summed Ralchenko singlet
  manifold (``_cross.He_beam_excitation_channel``); banked as radiation.
- Coulomb drag on the plasma, per ``coulomb_model``; banked as plasma heating.
- optional anomalous (beam-plasma instability) drag, per ``anomalous_model``;
  banked as plasma heating (Langmuir turbulence Landau-damps on the bulk
  near where it is driven).

Coulomb closures (both parameter-free):

- ``"fast_electron"`` (default): the physical CSDA stopping power on plasma
  electrons, ``dE/dx = 2 pi e^4 n_e lnLambda / E`` — the collision rate of a
  primary falls as 1/v^3, so at main-discharge conditions (150 eV, n_e =
  5e12) the energy e-fold is ~35 m.
- ``"legacy_tau_ei"``: ``dE/dx = E / (v(E) tau_ei(Te, ne))`` with the
  *thermal* NRL collision time — the historical `_cathode_solver._compute_l_b`
  form (~1.1 m at the same conditions). Provided for continuity experiments;
  its "Coulomb" label overestimates the classical drag ~30x (THESIS_NOTES
  item 12).

Anomalous closure:

- ``"quasilinear"``: the beam-plasma (bump-on-tail) instability drives
  Langmuir waves at v_phase ~ v_b with cold-beam growth rate
  ``~(n_b/n_e)^(1/3) omega_pe`` (growth lengths ~mm-cm here, i.e. saturated
  essentially at the source); quasilinear diffusion then flattens the beam
  over the relaxation length

      l_QL = (n_e / n_b) * (v_b / omega_pe) * ln(n_e / n_b)

  (Vedenov-style estimate; ~0.05-0.1 m at production parameters). Modeled as
  mean-energy drag ``dE/dx = E / l_QL`` with the flux preserved (the module
  carries one mean energy, not a distribution — the plateau spread is a
  documented limitation). **Stated validity domain:** weak beam
  (``n_b << n_e``) with growth beating wave damping — the main-discharge
  column. The breakdown phase (n_b ~ n_e) is outside quasilinear theory and
  deliberately stays collisional. Requires ``beam_area_cm2`` to form
  ``n_b = Gamma0 / (A v_b)``. Order-of-magnitude closure: density-gradient
  detuning and saturation physics can lengthen it substantially; results
  using it must be presented per closure, like the drag story
  (THESIS_NOTES §3).

The primary is followed until it exits the domain (transmitted) or its
energy crosses ``E_stop`` (default: the lowest inelastic threshold, He 2^1S
at 20.6158 eV), where the sub-threshold residual is banked as plasma heating
in the crossing cell. Per-ray energy conservation
``Gamma0*E0 = heating + radiated + ionization cost + transmitted`` holds to
accumulated roundoff by construction (the energy decrement and the channel
banks are the same floating-point sums).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ._cathode_solver import _c_log_ei
from ._cross import He_EII_cross_lkup, He_beam_excitation_channel

_ERG_PER_EV = 1.602176634e-12
_ME_CGS = 9.1093837015e-28  # electron mass [g]
_E4_CGS = (4.80320425e-10) ** 4  # e^4 [esu^4]
_OMEGA_PE_COEFF = 5.64e4  # omega_pe = 5.64e4 sqrt(n_e) [rad/s] (NRL)

HE_I_ION_EV = 24.587
HE_E_STOP_EV = 20.6158  # lowest inelastic threshold (2^1S)
HE_OPB_EBAR_EV = 15.8  # Opal-Peterson-Beaty shape parameter for He


def beam_speed_cm_s(E_eV: float) -> float:
    """Nonrelativistic primary speed [cm/s] at kinetic energy E_eV."""
    return math.sqrt(2.0 * E_eV * _ERG_PER_EV / _ME_CGS)


def he_mean_secondary_energy_eV(
    E_eV: float,
    I_ion_eV: float = HE_I_ION_EV,
    ebar_eV: float = HE_OPB_EBAR_EV,
) -> float:
    """Mean secondary-electron energy [eV] per He ionization at E_eV.

    OPB shape over W in [0, (E - I)/2]:
    <W> = Ebar * ln(1 + x^2) / (2 atan(x)), x = W_max / Ebar.
    """
    W_max = 0.5 * (E_eV - I_ion_eV)
    if W_max <= 0.0:
        return 0.0
    x = W_max / ebar_eV
    return ebar_eV * math.log1p(x * x) / (2.0 * math.atan(x))


def coulomb_stopping_eV_per_cm(
    E_eV: float, ne: float, Te: float, model: str = "fast_electron"
) -> float:
    """Coulomb energy-loss rate [eV/cm] of a primary at E_eV."""
    if ne <= 0.0 or E_eV <= 0.0:
        return 0.0
    lnL = _c_log_ei(max(Te, 0.1), ne)
    if model == "fast_electron":
        return 2.0 * math.pi * _E4_CGS * ne * lnL / (E_eV * _ERG_PER_EV) / _ERG_PER_EV
    if model == "legacy_tau_ei":
        tau_ei = 3.44e5 * Te**1.5 / ne / lnL
        return E_eV / (beam_speed_cm_s(E_eV) * tau_ei)
    raise ValueError(
        f"unknown coulomb_model {model!r}; "
        "expected 'fast_electron' or 'legacy_tau_ei'"
    )


def quasilinear_relaxation_length_cm(
    E_eV: float, ne: float, n_b: float
) -> float:
    """Quasilinear beam relaxation length [cm], l_QL = (ne/nb)(v_b/w_pe)ln(ne/nb).

    Valid for a weak beam (n_b << n_e); returns inf outside that domain
    (n_b >= n_e/10) rather than extrapolating the theory into the strong-beam
    regime, and inf for an absent beam or plasma.
    """
    if ne <= 0.0 or n_b <= 0.0 or n_b >= 0.1 * ne:
        return math.inf
    omega_pe = _OMEGA_PE_COEFF * math.sqrt(ne)
    ratio = ne / n_b
    return ratio * (beam_speed_cm_s(E_eV) / omega_pe) * math.log(ratio)


@dataclass(frozen=True)
class BeamDepositionResult:
    """Per-cell deposition of one beam ray; arrays have shape (cells,).

    ionization_events   : ionization events [1/s]
    excitation_events   : excitation events [1/s]
    plasma_heating_erg_s: Coulomb + anomalous drag + secondary-electron
                          energy + sub-threshold residual [erg/s]
    radiated_erg_s      : excitation line radiation [erg/s]
    ionization_cost_erg_s: I_ion * ionization events [erg/s] (kept separate
                          to map onto the solver's beam_ionization_cost term)
    transmitted_flux    : primary flux leaving the far end [1/s]
    transmitted_energy_eV: mean primary energy at exit [eV] (0 if absorbed)
    E_entry_eV          : diagnostic: primary energy entering each cell [eV]
                          (0 for cells the ray never reaches)
    """

    ionization_events: np.ndarray
    excitation_events: np.ndarray
    plasma_heating_erg_s: np.ndarray
    radiated_erg_s: np.ndarray
    ionization_cost_erg_s: np.ndarray
    transmitted_flux: float
    transmitted_energy_eV: float
    E_entry_eV: np.ndarray


def deposit_beam(
    E0_eV: float,
    Gamma0_per_s: float,
    nn: np.ndarray,
    ne: np.ndarray,
    Te: np.ndarray,
    launch: int,
    direction: int,
    dz_cm: np.ndarray,
    *,
    I_ion_eV: float = HE_I_ION_EV,
    E_stop_eV: float = HE_E_STOP_EV,
    coulomb_model: str = "fast_electron",
    anomalous_model: str = "none",
    beam_area_cm2: np.ndarray | float | None = None,
    max_energy_fraction_per_substep: float = 0.02,
) -> BeamDepositionResult:
    """Deposit one monoenergetic beam ray through the column (He only).

    Parameters mirror plan §2: ``E0_eV`` is the accelerating sheath drop,
    ``Gamma0_per_s`` the accepted emitted electron flux, ``nn/ne/Te`` the
    per-cell column state, and the ray is ``(launch, direction, dz_cm)``
    with ``direction`` +1 (toward increasing index) or -1. Cells behind the
    launch point receive nothing.

    ``anomalous_model``: ``"none"`` (default) or ``"quasilinear"``
    (requires ``beam_area_cm2``, scalar or per-cell, to form n_b).
    """
    nn = np.asarray(nn, dtype=float)
    ne = np.asarray(ne, dtype=float)
    Te = np.asarray(Te, dtype=float)
    dz_cm = np.asarray(dz_cm, dtype=float)
    cells = dz_cm.size
    if nn.shape != (cells,) or ne.shape != (cells,) or Te.shape != (cells,):
        raise ValueError("nn, ne, Te, dz_cm must share one shape (cells,)")
    if direction not in (-1, 1):
        raise ValueError(f"direction must be +1 or -1 (got {direction})")
    if anomalous_model not in ("none", "quasilinear"):
        raise ValueError(
            f"unknown anomalous_model {anomalous_model!r}; "
            "expected 'none' or 'quasilinear'"
        )
    if anomalous_model == "quasilinear":
        if beam_area_cm2 is None:
            raise ValueError("anomalous_model='quasilinear' needs beam_area_cm2")
        area = np.broadcast_to(
            np.asarray(beam_area_cm2, dtype=float), (cells,)
        )
    frac = float(max_energy_fraction_per_substep)
    if not 0.0 < frac < 1.0:
        raise ValueError(
            "max_energy_fraction_per_substep must be in (0, 1), got "
            f"{max_energy_fraction_per_substep}"
        )

    ionization_events = np.zeros(cells)
    excitation_events = np.zeros(cells)
    heating = np.zeros(cells)  # erg/s
    radiated = np.zeros(cells)
    ionization_cost = np.zeros(cells)
    E_entry = np.zeros(cells)

    order = range(launch, cells) if direction > 0 else range(launch, -1, -1)
    E = float(E0_eV)
    absorbed = False

    if E <= E_stop_eV:
        # Sub-threshold source: nothing inelastic can happen; the module's
        # domain is beam energies, so pass it through untouched.
        return BeamDepositionResult(
            ionization_events=ionization_events,
            excitation_events=excitation_events,
            plasma_heating_erg_s=heating,
            radiated_erg_s=radiated,
            ionization_cost_erg_s=ionization_cost,
            transmitted_flux=float(Gamma0_per_s),
            transmitted_energy_eV=E,
            E_entry_eV=E_entry,
        )

    for cell in order:
        E_entry[cell] = E
        remaining = float(dz_cm[cell])
        nn_c = float(nn[cell])
        ne_c = float(ne[cell])
        Te_c = float(Te[cell])
        while remaining > 0.0:
            sigma_i = (
                He_EII_cross_lkup(E / I_ion_eV) if E > I_ion_eV else 0.0
            )
            sigma_x, E_rad = He_beam_excitation_channel(E)
            W_sec = he_mean_secondary_energy_eV(E, I_ion_eV=I_ion_eV)
            # channel loss rates [eV/cm]
            L_pot = nn_c * sigma_i * I_ion_eV
            L_sec = nn_c * sigma_i * W_sec
            L_exc = nn_c * sigma_x * E_rad
            L_coul = coulomb_stopping_eV_per_cm(
                E, ne_c, Te_c, model=coulomb_model
            )
            L_anom = 0.0
            if anomalous_model == "quasilinear":
                n_b = Gamma0_per_s / (
                    float(area[cell]) * beam_speed_cm_s(E)
                )
                l_ql = quasilinear_relaxation_length_cm(E, ne_c, n_b)
                if math.isfinite(l_ql) and l_ql > 0.0:
                    L_anom = E / l_ql
            L_tot = L_pot + L_sec + L_exc + L_coul + L_anom
            if L_tot <= 0.0:
                break  # vacuum cell: free streaming
            dz_sub = min(remaining, frac * E / L_tot)
            # Land exactly on E_stop rather than overshooting through it.
            if E - L_tot * dz_sub <= E_stop_eV:
                dz_sub = (E - E_stop_eV) / L_tot
            if dz_sub <= 0.0:
                # E sits at E_stop to roundoff: absorb the residual here.
                heating[cell] += Gamma0_per_s * E * _ERG_PER_EV
                E = 0.0
                absorbed = True
                break
            # Bank each channel with the identical products the energy
            # decrement uses, so conservation closes to roundoff.
            d_pot = L_pot * dz_sub
            d_sec = L_sec * dz_sub
            d_exc = L_exc * dz_sub
            d_coul = L_coul * dz_sub
            d_anom = L_anom * dz_sub
            ionization_cost[cell] += Gamma0_per_s * d_pot * _ERG_PER_EV
            heating[cell] += Gamma0_per_s * (d_sec + d_coul + d_anom) * _ERG_PER_EV
            radiated[cell] += Gamma0_per_s * d_exc * _ERG_PER_EV
            ionization_events[cell] += Gamma0_per_s * nn_c * sigma_i * dz_sub
            excitation_events[cell] += Gamma0_per_s * nn_c * sigma_x * dz_sub
            E -= d_pot + d_sec + d_exc + d_coul + d_anom
            remaining -= dz_sub
            if E <= E_stop_eV:
                # Sub-threshold residual: the primary can only Coulomb-drag
                # from here; bank the remainder as local plasma heating
                # (plan B1's stated closure) and end the ray.
                heating[cell] += Gamma0_per_s * E * _ERG_PER_EV
                E = 0.0
                absorbed = True
                break
        if absorbed:
            break

    return BeamDepositionResult(
        ionization_events=ionization_events,
        excitation_events=excitation_events,
        plasma_heating_erg_s=heating,
        radiated_erg_s=radiated,
        ionization_cost_erg_s=ionization_cost,
        transmitted_flux=0.0 if absorbed else float(Gamma0_per_s),
        transmitted_energy_eV=0.0 if absorbed else E,
        E_entry_eV=E_entry,
    )
