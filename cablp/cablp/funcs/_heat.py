import numpy as np
from ..vars._cons import H_e_mass_ratio
from ._fits import IAEA_exp1, IAEA_exp4, IAEA_exp6
from ..vars._coeff import aHeI, aHeII, aHI, aHII
from ._cross import charge_ex_react
from ._plasmaparams import (
    time_elec_coll,
    time_ion_coll,
    v_thm_e,
    v_ion_speed,
    elec_gyro_freq,
    ion_gyro_freq,
)


def _resolve_per_particle(per_particle, rk):
    """Resolve the legacy ``rk`` keyword to the explicit unit-mode flag."""
    if rk is not None:
        return rk
    return per_particle


def kappa_par_elec(Te, ne, lnlambda, per_particle=True, *, rk=None):
    """
    Electron parallel thermal conductivity (Braginskii) [eV·cm²/s or eV·cm⁻¹/s].

    kappa = 3.16 * tau_e * v_the²

    Parameters
    ----------
    Te : float or array
        Electron temperature [eV].
    ne : float or array
        Electron density [cm⁻³].
    lnlambda : float or array
        Coulomb logarithm.
    per_particle : bool
        If True return the per-particle conductivity [eV·cm²/s].
        If False return the volumetric conductivity kappa * ne [eV·cm⁻¹/s].

    Returns
    -------
    float or array
        Parallel electron thermal conductivity.
    """
    per_particle = _resolve_per_particle(per_particle, rk)
    k = 3.16 * time_elec_coll(Te, ne, lnlambda) * v_thm_e(Te) ** 2
    if per_particle:
        return k
    else:
        return k * ne


def kappa_par_ion(Ti, ni, mu, lnlambda, per_particle=True, *, rk=None):
    """
    Ion parallel thermal conductivity (Braginskii) [eV·cm²/s or eV·cm⁻¹/s].

    kappa = 3.9 * tau_i * v_thi²

    Parameters
    ----------
    Ti : float or array
        Ion temperature [eV].
    ni : float or array
        Ion density [cm⁻³].
    mu : float
        Ion mass number (m_ion / m_proton).
    lnlambda : float or array
        Coulomb logarithm.
    per_particle : bool
        If True return per-particle [eV·cm²/s]; if False return volumetric [eV·cm⁻¹/s].

    Returns
    -------
    float or array
        Parallel ion thermal conductivity.
    """
    per_particle = _resolve_per_particle(per_particle, rk)
    k = 3.9 * time_ion_coll(Ti, ni, mu, lnlambda) * v_ion_speed(Ti, mu) ** 2
    if per_particle:
        return k
    else:
        return k * ni


def elec_par_heat_loss(Te, ne, L_p, L_hf, lnlambda, per_particle=True, *, rk=None):
    """
    Electron parallel heat loss rate per unit volume [eV/s] or [eV·cm⁻³/s].

    Q = kappa_par_elec * Te / L_p / L_hf

    Parameters
    ----------
    Te : float or array
        Electron temperature [eV].
    ne : float or array
        Electron density [cm⁻³].
    L_p : float or array
        Plasma half-length [cm].
    L_hf : float or array
        Heat-flux scale length [cm].
    lnlambda : float or array
        Coulomb logarithm.
    per_particle : bool
        Passed through to kappa_par_elec.

    Returns
    -------
    float or array
        Electron parallel heat loss [eV/s (per_particle=True) or eV·cm⁻³/s (per_particle=False)].
    """
    per_particle = _resolve_per_particle(per_particle, rk)
    return kappa_par_elec(Te, ne, lnlambda, per_particle=per_particle) * Te / L_p / L_hf


def ion_par_heat_loss(Ti, ni, L_p, L_hf, mu, lnlambda, per_particle=True, *, rk=None):
    """
    Ion parallel heat loss rate per unit volume [eV/s] or [eV·cm⁻³/s].

    Q = kappa_par_ion * Ti / L_p / L_hf

    Parameters
    ----------
    Ti : float or array
        Ion temperature [eV].
    ni : float or array
        Ion density [cm⁻³].
    L_p : float or array
        Plasma half-length [cm].
    L_hf : float or array
        Heat-flux scale length [cm].
    mu : float
        Ion mass number.
    lnlambda : float or array
        Coulomb logarithm.
    per_particle : bool
        Passed through to kappa_par_ion.

    Returns
    -------
    float or array
        Ion parallel heat loss [eV/s or eV·cm⁻³/s].
    """
    per_particle = _resolve_per_particle(per_particle, rk)
    return kappa_par_ion(Ti, ni, mu, lnlambda, per_particle=per_particle) * Ti / L_p / L_hf


def elec_par_heat_div(Te, ne, L_plasma, lnlambda):
    """
    Inter-cell electron heating rate from parallel conduction: div(kappa_par * grad(Te)).

    Conservative finite-volume discretization. Boundary cells receive a
    contribution from their single interior face only; wall heat loss is
    handled separately via elec_par_heat_loss.

    Parameters
    ----------
    Te : array, shape (N,)
        Electron temperature [eV].
    ne : array, shape (N,)
        Electron density [cm⁻³].
    L_plasma : array, shape (N,)
        Cell lengths [cm].
    lnlambda : array, shape (N,)
        Coulomb logarithm.

    Returns
    -------
    array, shape (N,)
        dTe/dt contribution [eV s⁻¹] per particle.
    """
    Q_face = -elec_par_heat_face_flux(Te, ne, L_plasma, lnlambda)
    result = np.zeros_like(Te)
    result[:-1] += Q_face / L_plasma[:-1]
    result[1:]  -= Q_face / L_plasma[1:]
    return result


def elec_par_heat_face_flux(Te, ne, L_plasma, lnlambda):
    """
    Signed electron conductive heat flux at interior cell faces.

    Positive values carry heat from cell i to cell i+1; negative values carry
    heat from cell i+1 to cell i.
    """
    kappa = kappa_par_elec(Te, ne, lnlambda, per_particle=True)
    kappa_face = (kappa[:-1] + kappa[1:]) / 2
    d_face = (L_plasma[:-1] + L_plasma[1:]) / 2
    return -kappa_face * (Te[1:] - Te[:-1]) / d_face


def ion_par_heat_div(Ti, ni, L_plasma, mu, lnlambda):
    """
    Inter-cell ion heating rate from parallel conduction: div(kappa_par * grad(Ti)).

    Parameters
    ----------
    Ti : array, shape (N,)
        Ion temperature [eV].
    ni : array, shape (N,)
        Ion density [cm⁻³].
    L_plasma : array, shape (N,)
        Cell lengths [cm].
    mu : float
        Ion mass number.
    lnlambda : array, shape (N,)
        Coulomb logarithm.

    Returns
    -------
    array, shape (N,)
        dTi/dt contribution [eV s⁻¹] per particle.
    """
    Q_face = -ion_par_heat_face_flux(Ti, ni, L_plasma, mu, lnlambda)
    result = np.zeros_like(Ti)
    result[:-1] += Q_face / L_plasma[:-1]
    result[1:]  -= Q_face / L_plasma[1:]
    return result


def ion_par_heat_face_flux(Ti, ni, L_plasma, mu, lnlambda):
    """
    Signed ion conductive heat flux at interior cell faces.

    Positive values carry heat from cell i to cell i+1; negative values carry
    heat from cell i+1 to cell i.
    """
    kappa = kappa_par_ion(Ti, ni, mu, lnlambda, per_particle=True)
    kappa_face = (kappa[:-1] + kappa[1:]) / 2
    d_face = (L_plasma[:-1] + L_plasma[1:]) / 2
    return -kappa_face * (Ti[1:] - Ti[:-1]) / d_face


def kappa_perp_elec(Te, ne, B, lnlambda, per_particle=True, *, rk=None):
    """
    Electron perpendicular thermal conductivity (classical) [eV·cm²/s or eV·cm⁻¹/s].

    kappa_perp = 4.7 * v_the² / Omega_e² / tau_e

    Parameters
    ----------
    Te : float or array
        Electron temperature [eV].
    ne : float or array
        Electron density [cm⁻³].
    B : float or array
        Magnetic field [Gauss].
    lnlambda : float or array
        Coulomb logarithm.
    per_particle : bool
        If True return per-particle; if False return volumetric (× ne).

    Returns
    -------
    float or array
        Perpendicular electron thermal conductivity.
    """
    per_particle = _resolve_per_particle(per_particle, rk)
    k = (
        4.7
        * v_thm_e(Te) ** 2
        / elec_gyro_freq(B) ** 2
        / time_elec_coll(Te, ne, lnlambda)
    )
    if per_particle:
        return k
    else:
        return k * ne


def kappa_perp_ion(Ti, ni, B, mu, lnlambda, per_particle=True, *, rk=None):
    """
    Ion perpendicular thermal conductivity (classical) [eV·cm²/s or eV·cm⁻¹/s].

    kappa_perp = 2 * v_thi² / Omega_i² / tau_i

    Parameters
    ----------
    Ti : float or array
        Ion temperature [eV].
    ni : float or array
        Ion density [cm⁻³].
    B : float or array
        Magnetic field [Gauss].
    mu : float
        Ion mass number.
    lnlambda : float or array
        Coulomb logarithm.
    per_particle : bool
        If True return per-particle; if False return volumetric (× ni).

    Returns
    -------
    float or array
        Perpendicular ion thermal conductivity.
    """
    per_particle = _resolve_per_particle(per_particle, rk)
    k = (
        2
        * v_ion_speed(Ti, mu) ** 2
        / ion_gyro_freq(B, mu) ** 2
        / time_ion_coll(Ti, ni, mu, lnlambda)
    )
    if per_particle:
        return k
    else:
        return k * ni


def elec_perp_heat_loss(Te, ne, R_p, R_hf, B, lnlambda, per_particle=True, *, rk=None):
    """
    Electron perpendicular heat loss rate [eV/s or eV·cm⁻³/s].

    Q = kappa_perp_elec * Te / R_p / R_hf

    Parameters
    ----------
    Te : float or array
        Electron temperature [eV].
    ne : float or array
        Electron density [cm⁻³].
    R_p : float or array
        Plasma radius [cm].
    R_hf : float or array
        Radial heat-flux scale length [cm].
    B : float or array
        Magnetic field [Gauss].
    lnlambda : float or array
        Coulomb logarithm.
    per_particle : bool
        Passed through to kappa_perp_elec.

    Returns
    -------
    float or array
        Electron perpendicular heat loss.
    """
    per_particle = _resolve_per_particle(per_particle, rk)
    return kappa_perp_elec(Te, ne, B, lnlambda, per_particle=per_particle) * Te / R_p / R_hf


def ion_perp_heat_loss(Ti, ni, R_p, R_hf, B, mu, lnlambda, per_particle=True, *, rk=None):
    """
    Ion perpendicular heat loss rate [eV/s or eV·cm⁻³/s].

    Q = kappa_perp_ion * Ti / R_p / R_hf

    Parameters
    ----------
    Ti : float or array
        Ion temperature [eV].
    ni : float or array
        Ion density [cm⁻³].
    R_p : float or array
        Plasma radius [cm].
    R_hf : float or array
        Radial heat-flux scale length [cm].
    B : float or array
        Magnetic field [Gauss].
    mu : float
        Ion mass number.
    lnlambda : float or array
        Coulomb logarithm.
    per_particle : bool
        Passed through to kappa_perp_ion.

    Returns
    -------
    float or array
        Ion perpendicular heat loss.
    """
    per_particle = _resolve_per_particle(per_particle, rk)
    return kappa_perp_ion(Ti, ni, B, mu, lnlambda, per_particle=per_particle) * Ti / R_p / R_hf


def Q_ie(Te, Ti, ne, mu, lnlambda, per_particle=True, *, rk=None):
    """
    Electron-ion energy exchange rate (Braginskii) [eV/s or eV·cm⁻³/s].

    Positive when Te > Ti (electrons lose energy to ions).

    Q = 3 * (Te - Ti) / tau_e / (m_i/m_e)

    Parameters
    ----------
    Te : float or array
        Electron temperature [eV].
    Ti : float or array
        Ion temperature [eV].
    ne : float or array
        Electron density [cm⁻³].
    mu : float
        Ion mass number.
    lnlambda : float or array
        Coulomb logarithm.
    per_particle : bool
        If True return per-particle [eV/s]; if False return volumetric (× ne).

    Returns
    -------
    float or array
        Electron-to-ion energy transfer rate.
    """
    per_particle = _resolve_per_particle(per_particle, rk)
    Q = 3 * (Te - Ti) / time_elec_coll(Te, ne, lnlambda) / H_e_mass_ratio / mu
    if per_particle:
        return Q
    else:
        return Q * ne


def Q_ei_in(ne, Te, per_particle=True, *, rk=None):
    """
    Electron cooling rate due to inelastic collisions with He II ions [eV/s or eV·cm⁻³/s].

    Uses IAEA expression 4 with He II cooling coefficients.

    Parameters
    ----------
    ne : float or array
        Electron density [cm⁻³].
    Te : float or array
        Electron temperature [eV].
    per_particle : bool
        If True return Q*ne [eV/s]; if False return Q*ne² [eV·cm⁻³/s].

    Returns
    -------
    float or array
        Electron cooling rate from He II inelastic collisions.
    """
    per_particle = _resolve_per_particle(per_particle, rk)
    Q = IAEA_exp4(Te, aHeII) * ne
    if per_particle:
        return Q
    else:
        return Q * ne


def Q_en(ne, nn, Te, per_particle=True, *, rk=None):
    """
    Electron cooling rate due to inelastic collisions with He I neutrals [eV/s or eV·cm⁻³/s].

    Uses IAEA expression 1 with He I cooling coefficients.

    Parameters
    ----------
    ne : float or array
        Electron density [cm⁻³].
    nn : float or array
        Neutral density [cm⁻³].
    Te : float or array
        Electron temperature [eV].
    per_particle : bool
        If True return Q*nn [eV/s]; if False return Q*nn*ne [eV·cm⁻³/s].

    Returns
    -------
    float or array
        Electron cooling rate from He I neutral collisions.
    """
    per_particle = _resolve_per_particle(per_particle, rk)
    Q = IAEA_exp1(Te, aHeI) * nn
    if per_particle:
        return Q
    else:
        return Q * ne


def Q_cx_He(ne, nn, Ti, Tn, gas_type="He", per_particle=True, *, rk=None):
    """
    Ion cooling rate due to charge exchange with neutrals [eV/s or eV·cm⁻³/s].

    Q = nn * <sigma*v>_cx * (Ti - Tn)

    Parameters
    ----------
    ne : float or array
        Electron (= ion) density [cm⁻³].
    nn : float or array
        Neutral density [cm⁻³].
    Ti : float or array
        Ion temperature [eV].
    Tn : float or array
        Neutral temperature [eV].
    gas_type : str
        Gas species; "He" or "H".
    per_particle : bool
        If True return per-ion rate [eV/s]; if False return volumetric (× ne).

    Returns
    -------
    float or array
        Ion charge-exchange cooling rate.
    """
    per_particle = _resolve_per_particle(per_particle, rk)
    Q = nn * charge_ex_react(Ti, gas_type) * (Ti - Tn)
    if per_particle:
        return Q
    else:
        return Q * ne


def tau_scale(ne, T, Q):
    """
    Energy confinement time estimate tau = 1.5 * ne * T / Q [s].

    Parameters
    ----------
    ne : float or array
        Density [cm⁻³].
    T : float or array
        Temperature [eV].
    Q : float or array
        Volumetric power loss [eV·cm⁻³/s].

    Returns
    -------
    float or array
        Energy confinement time [s].
    """
    return 1.5 * ne * T / Q


def Q_ie_H(Te, Ti, ne, lnlambda=12):
    """
    Electron-ion energy exchange rate for hydrogen [eV·cm⁻³/s].

    Parameters
    ----------
    Te : float or array
        Electron temperature [eV].
    Ti : float or array
        Ion temperature [eV].
    ne : float or array
        Electron density [cm⁻³].
    lnlambda : float
        Coulomb logarithm; default 12.

    Returns
    -------
    float or array
        Volumetric electron-to-ion energy transfer rate [eV·cm⁻³/s].
    """
    return 3 * ne * (Te - Ti) / time_elec_coll(Te, ne, lnlambda) / H_e_mass_ratio


def Q_ei_in_H(ne, T):
    """
    Electron cooling rate due to inelastic collisions with H II ions [eV·cm⁻⁶/s].

    Uses IAEA expression 6 with H II cooling coefficients.

    Parameters
    ----------
    ne : float or array
        Electron density [cm⁻³].
    T : float or array
        Electron temperature [eV].

    Returns
    -------
    float or array
        Electron cooling rate [eV·cm⁻⁶/s] (proportional to ne²).
    """
    return IAEA_exp6(T, aHII) * ne * ne


def Q_en_H(ne, nn, T):
    """
    Electron cooling rate due to inelastic collisions with H I neutrals [eV·cm⁻⁶/s].

    Uses IAEA expression 1 with H I cooling coefficients.

    Parameters
    ----------
    ne : float or array
        Electron density [cm⁻³].
    nn : float or array
        Neutral density [cm⁻³].
    T : float or array
        Electron temperature [eV].

    Returns
    -------
    float or array
        Electron-neutral cooling rate [eV·cm⁻⁶/s].
    """
    return IAEA_exp1(T, aHI) * ne * nn
