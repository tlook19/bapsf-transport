import numpy as np
from cablp.funcs._heat import IAEA_exp1, IAEA_exp4
from cablp.funcs._plasmaparams import (
    v_ion_speed,
    v_thm_e,
    time_elec_coll,
    time_ion_coll,
)
from cablp.vars._cons import qe_SI, I_ion, He_e_mass_ratio
from cablp.funcs._fits import rate_coeff
from cablp.funcs._cross import alpha_3, alpha_r, charge_ex_react
from cablp.funcs._heat import (
    Q_cx_He,
    Q_ie,
    elec_par_heat_loss,
    ion_par_heat_loss,
    elec_perp_heat_loss,
    ion_perp_heat_loss,
)
from cablp.vars._coeff import aHeI, aHeII

fit_coeff = [1.3950030050791237e-05, 13.62996440158007]
en_factor = 2 / 3


def calc_heat_terms(
    a,
    L,
    Q_B,
    off=False,
):
    ne, nn, Te, Ti = a
    if Ti < 0.1:
        Ti = 0.1
    if Te < 0.1:
        Te = 0.1
    e_par_hl = en_factor * 2 * elec_par_heat_loss(Te, ne, L, L)
    i_par_hl = en_factor * 2 * ion_par_heat_loss(Ti, ne, L, L, mu=4)
    Qie = en_factor * Q_ie(Te, Ti, ne)
    Qei = en_factor * IAEA_exp4(Te, aHeII) * ne
    Qen = en_factor * IAEA_exp1(Te, aHeI) * nn
    Qcx = en_factor * Q_cx_He(ne, nn, Ti, 0.1)
    if off:
        Q_eb = 0
    else:
        Q_eb = en_factor * Q_B / ne
    return np.array([e_par_hl, Qie, Qei, Qen, Q_eb, i_par_hl, Qcx])


def dstep(a, L, Q_B, S_gp, S_pump, off):
    """
    Calculate the rate of change of electron density.

    Parameters
    ----------
    ne : float
        Electron density (cm^-3).
    nn : float
        Neutral density (cm^-3).
    T : float
        Temperature (eV).

    Returns
    -------
    float
        Rate of change of electron density (m^-3 s^-1).
    """
    ne, nn, Te, Ti = a
    if Ti < 0.1:
        Ti = 0.1
    if Te < 0.1:
        Te = 0.1
    end_loss = -2 * ne * v_ion_speed(1, 4) / 1800
    S_ionization = ne * nn * rate_coeff(Te, I_ion, *fit_coeff)
    S_rec_rad = ne * nn * alpha_r(Te)
    S_rec_3b = ne * ne * nn * alpha_3(Te)
    print(f"endloss: {end_loss:.2e}")
    thm_flux, Q_ie, Q_ei, Q_en, Q_eb, thm_flux_ion, Q_cx = calc_heat_terms(
        a, L, Q_B, off
    )
    d_ne = end_loss + S_ionization - S_rec_rad - S_rec_3b
    d_nn = -d_ne + (S_gp) - (S_pump * nn)
    d_Te = Q_eb - Q_ei - Q_en - Q_ie - thm_flux
    d_Ti = Q_ie - Q_cx - thm_flux_ion
    print(
        f"Q_eb: {Q_eb:.2e}, Q_ie: {Q_ie:.2e}, Q_ei: {Q_ei:.2e}, Q_en: {Q_en:.2e}, thm_flux: {thm_flux:.2e}, Q_cx: {Q_cx:.2e}, thm_flux_ion: {thm_flux_ion:.2e}"
    )
    # print(end_loss, S_ionization, S_rec_rad, S_rec_3b, thm_flux, Q_ie, Q_ei, Q_en, Q)
    print(f"{d_ne:.2e}, {d_nn:.2e}, {d_Te:.2e}, {d_Ti:.2e}")
    return np.array([d_ne, d_nn, d_Te, d_Ti])


def rk4_step(a, L, h, Q_B, S_gp, S_pump, off):
    k1 = dstep(a, L, Q_B, S_gp, S_pump, off)
    k1_a = a + 0.5 * h * k1
    k2 = dstep(k1_a, L, Q_B, S_gp, S_pump, off)
    k2_a = a + 0.5 * h * k2
    k3 = dstep(k2_a, L, Q_B, S_gp, S_pump, off)
    k3_a = a + h * k3
    k4 = dstep(k3_a, L, Q_B, S_gp, S_pump, off)
    # print("k: ", k1, k2, k3, k4)
    b = a + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    # print("b: ", b)
    return b


def stepper(
    ne0, nn0, Te0, Ti0, L, h1, h2, t_off, t_stop, Q_B, S_gp, S_pump, plasma_vol
):
    time = np.arange(0, t_off, h1)
    i_off = time.shape[0]
    time = np.append(time, np.arange(t_off, t_stop, h2))
    params = np.empty((time.shape[0], 12))
    thm_flux, Q_ie, Q_ei, Q_en, Q_eb, thm_flux_ion, Q_cx = (
        calc_heat_terms([ne0, nn0, Te0, Ti0], L, Q_B) * qe_SI * plasma_vol
    )
    params[0] = np.array(
        [0, ne0, nn0, Te0, Ti0, thm_flux, Q_ie, Q_ei, Q_en, Q_eb, thm_flux_ion, Q_cx]
    )
    for i, t in enumerate(time):
        if i == 0:
            continue
        ne, nn, Te, Ti = params[i - 1, 1:5]
        if i < i_off:
            h = h1
            off = False
        else:
            h = h2
            off = True
            Q_B = 0
            S_gp = 0
        ne, nn, Te, Ti = rk4_step(
            np.array([ne, nn, Te, Ti]), L, h, Q_B, S_gp, S_pump, off
        )
        if Te < 0.1:
            Te = 0.1
        if Ti < 0.1:
            Ti = 0.1
        print(f"{i}, {nn:.2e}, {nn:2e}, {Te:2e}, {Ti:2e}")
        thm_flux, Q_ie, Q_ei, Q_en, Q_eb, thm_flux_ion, Q_cx = (
            calc_heat_terms([ne, nn, Te, Ti], L, Q_B, off) * qe_SI * ne * plasma_vol
        )
        params[i] = [
            t,
            ne,
            nn,
            Te,
            Ti,
            thm_flux,
            Q_ie,
            Q_ei,
            Q_en,
            Q_eb,
            thm_flux_ion,
            Q_cx,
        ]
    return params
