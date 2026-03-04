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


def kappa_par_elec(Te, ne, lnlambda, rk=True):
    k = 3.16 * time_elec_coll(Te, ne, lnlambda) * v_thm_e(Te) ** 2
    if rk:
        return k
    else:
        return k * ne


def kappa_par_ion(Ti, ni, mu, lnlambda, rk=True):
    k = 3.9 * time_ion_coll(Ti, ni, mu, lnlambda) * v_ion_speed(Ti, mu) ** 2
    if rk:
        return k
    else:
        return k * ni


def elec_par_heat_loss(Te, ne, L_p, L_hf, lnlambda, rk=True):
    return kappa_par_elec(Te, ne, lnlambda, rk) * Te / L_p / L_hf


def ion_par_heat_loss(Ti, ni, L_p, L_hf, mu, lnlambda, rk=True):
    return kappa_par_ion(Ti, ni, mu, lnlambda, rk) * Ti / L_p / L_hf


def kappa_perp_elec(Te, ne, B, lnlambda, rk=True):
    k = (
        4.7
        * v_thm_e(Te) ** 2
        / elec_gyro_freq(B) ** 2
        / time_elec_coll(Te, ne, lnlambda)
    )
    if rk:
        return k
    else:
        return k * ne


def kappa_perp_ion(Ti, ni, B, mu, lnlambda, rk=True):
    k = (
        2
        * v_ion_speed(Ti, mu) ** 2
        / ion_gyro_freq(B, mu) ** 2
        / time_ion_coll(Ti, ni, mu, lnlambda)
    )
    if rk:
        return k
    else:
        return k * ni


def elec_perp_heat_loss(Te, ne, R_p, R_hf, B, lnlambda, rk=True):
    return kappa_perp_elec(Te, ne, B, lnlambda, rk) * Te / R_p / R_hf


def ion_perp_heat_loss(Ti, ni, R_p, R_hf, B, mu, lnlambda, rk=True):
    return kappa_perp_ion(Ti, ni, B, mu, lnlambda, rk) * Ti / R_p / R_hf


def Q_ie(Te, Ti, ne, mu, lnlambda, rk=True):
    # energy transfer rate between electrons and ions via elastic collisions
    Q = 3 * (Te - Ti) / time_elec_coll(Te, ne, lnlambda) / H_e_mass_ratio / mu
    if rk:
        return Q
    else:
        return Q * ne


def Q_ei_in(ne, Te, rk=True):
    # electron cooling rate due to inelastic collisions with ions
    Q = IAEA_exp4(Te, aHeII) * ne
    if rk:
        return Q
    else:
        return Q * ne


def Q_en(ne, nn, Te, rk=True):
    # electron cooling rate due to inelastic collisions with neutrals
    Q = IAEA_exp1(Te, aHeI) * nn
    if rk:
        return Q
    else:
        return Q * ne


def Q_cx_He(ne, nn, Ti, Tn, gas_type="He", rk=True):
    Q = nn * charge_ex_react(Ti, gas_type) * (Ti - Tn)
    if rk:
        return Q
    else:
        return Q * ne


def tau_scale(ne, T, Q):
    return 1.5 * ne * T / Q


def Q_ie_H(Te, Ti, ne, lnlambda=12):
    return 3 * ne * (Te - Ti) / time_elec_coll(Te, ne, lnlambda) / H_e_mass_ratio


def Q_ei_in_H(ne, T):
    return IAEA_exp6(T, aHII) * ne * ne


def Q_en_H(ne, nn, T):
    return IAEA_exp1(T, aHI) * ne * nn
