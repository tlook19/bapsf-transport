from ..constants import H_e_mass_ratio
from ..atomic.cross_sections import charge_ex_react
from .params import (
    time_elec_coll,
    time_ion_coll,
    v_thm_e,
    v_ion_speed,
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
