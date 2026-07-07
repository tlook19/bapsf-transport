import numpy as np

from cablp.funcs._fits import IAEA_exp1, IAEA_exp4, IAEA_exp6
from cablp.funcs._heat import Q_ie
from cablp.funcs._plasmaparams import c_log
from cablp.vars._coeff import aHII, aHI, aHeI, aHeII
from cablp.vars._cons import ev_to_erg

from .reactions import reaction_rates
from .state import ConservativeState1D, derive_state


def electron_ion_exchange_rhs(
    state,
    floors,
    ion_mass_g,
    mu,
    b_Qie=1.0,
    ln_lambda_min=1.0,
):
    """Return conservative electron-ion thermal exchange sources.

    ``Q_ie`` is positive when electrons transfer energy to ions. The helper
    returns eV cm^-3 s^-1 with ``per_particle=False``; conservative energies are
    stored as erg cm^-3.
    """
    zeros = np.zeros_like(state.n, dtype=float)
    if b_Qie == 0.0:
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
    q_e_to_i = (
        float(b_Qie)
        * Q_ie(
            derived.Te,
            derived.Ti,
            n,
            mu,
            ln_lambda,
            per_particle=False,
        )
        * ev_to_erg
    )
    return ConservativeState1D(
        n=zeros,
        nn=zeros.copy(),
        M=zeros.copy(),
        Ee=-q_e_to_i,
        Ei=q_e_to_i,
    )


def electron_cooling_rhs(
    state,
    floors,
    ion_mass_g,
    gas_type,
    I_ion,
    b_ioniz=1.0,
    b_rec_rad=1.0,
    b_rec_3b=1.0,
    b_ionization_energy_cost=1.0,
    b_Qei=1.0,
    b_Qen=1.0,
    ionization_energy_cost=True,
    icool=True,
    ncool=True,
    icool_recomb=False,
):
    """Return conservative electron inelastic/radiative cooling sources.

    Cooling terms are volumetric electron-energy sinks. The IAEA fit helpers
    return eV-rate coefficients, so the accumulated loss is converted to
    conservative ``erg cm^-3 s^-1`` before being applied to ``Ee``.
    """
    zeros = np.zeros_like(state.n, dtype=float)
    cooling_eV = zeros.copy()
    derived = derive_state(state, floors=floors, ion_mass_g=ion_mass_g)

    if ionization_energy_cost and b_ionization_energy_cost != 0.0:
        S_ion, _, _ = reaction_rates(
            state=state,
            floors=floors,
            ion_mass_g=ion_mass_g,
            gas_type=gas_type,
            I_ion=I_ion,
            b_ioniz=b_ioniz,
            b_rec_rad=b_rec_rad,
            b_rec_3b=b_rec_3b,
        )
        cooling_eV = cooling_eV + float(b_ionization_energy_cost) * I_ion * S_ion

    if icool and b_Qei != 0.0:
        cooling_eV = cooling_eV + float(b_Qei) * _ion_inelastic_cooling_eV(
            derived.Te,
            state.n,
            gas_type=gas_type,
            recomb=icool_recomb,
        )

    if ncool and b_Qen != 0.0:
        cooling_eV = cooling_eV + float(b_Qen) * _neutral_inelastic_cooling_eV(
            derived.Te,
            state.n,
            state.nn,
            gas_type=gas_type,
        )

    return ConservativeState1D(
        n=zeros,
        nn=zeros.copy(),
        M=zeros.copy(),
        Ee=-cooling_eV * ev_to_erg,
        Ei=zeros.copy(),
    )


def _ion_inelastic_cooling_eV(Te, n, gas_type, recomb=False):
    """Return electron-ion inelastic/radiative cooling [eV cm^-3 s^-1]."""
    if gas_type == "He":
        return IAEA_exp4(Te, aHeII, recomb=recomb) * n * n
    if gas_type == "H":
        return IAEA_exp6(Te, aHII) * n * n
    raise ValueError(f"unsupported gas_type {gas_type!r}; expected 'He' or 'H'")


def _neutral_inelastic_cooling_eV(Te, n, nn, gas_type):
    """Return electron-neutral inelastic cooling [eV cm^-3 s^-1]."""
    if gas_type == "He":
        return IAEA_exp1(Te, aHeI) * n * nn
    if gas_type == "H":
        return IAEA_exp1(Te, aHI) * n * nn
    raise ValueError(f"unsupported gas_type {gas_type!r}; expected 'He' or 'H'")
