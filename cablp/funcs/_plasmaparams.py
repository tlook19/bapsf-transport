import numpy as np


def time_elec_coll(T, ne, lnlambda):
    return 3.44e5 * (T ** (3 / 2)) / ne / lnlambda


def time_ion_coll(T, n, mu, lnlambda):
    return 2.09e7 * (T ** (3 / 2)) / n / lnlambda * np.sqrt(mu)


def v_thm_e(T):
    return 4.19e7 * np.sqrt(T)


def v_ion_speed(T, mu, Z=1, gamma=1):
    # if T = Ti then this gives the ion thermal velocity. Z and Gamma should be 1
    # if T = Te then this gives the ion sound speed. Z and Gamma can be different from 1
    return 9.79e5 * np.sqrt(gamma * Z * T / mu)


def elec_gyro_freq(B):
    return 1.76e7 * B


def ion_gyro_freq(B, mu, Z=1):
    return 9.58e3 * Z * B / mu


def c_log(Te, n):
    return 23.4 - 1.15 * np.log10(n) + 3.45 * np.log10(Te)
