from numpy import exp as npexp, sqrt as npsqrt
from math import exp as mexp, sqrt as msqrt
from ..vars._cons import qe_SI

# note, revist this when it is determined whether these functions are vectorized in the simulation


def IAEA_exp1(t, a):
    T = t * 1e-3  # convert to keV
    P = a[0] * npexp(-a[1] / (T ** a[2])) / (T ** a[3] + a[4] * T ** a[5])
    return P * 1e-27 / qe_SI


def IAEA_exp4(t, a, recomb=True):
    T = t * 1e-3  # convert to keV
    b = a[0] * npexp(-a[1] / T ** a[2]) / (T ** a[3] + a[4] * T ** a[5])
    if recomb == True:
        b += a[6] * T ** a[7]
    return b * 1e-27 / qe_SI


def IAEA_exp6(t, a):
    T = t * 1e-3  # convert to keV
    P = a[0] * T ** a[1] + a[2] * T ** a[3] + a[4] * T ** a[5]
    return P * 1e-27 / qe_SI


def rate_coeff(T, I, a, b):
    """
    Fit function for the reaction rate coefficient <sigma*v> averaged over a Maxwellian
    at temp T for a process with threshold energy I.

    Parameters
    ----------
    T : float, array of floats
        Temperature in eV
    I : float
        Threshold energy in eV
    a : float
        Fitting parameter
    b : float
        Fitting parameter

    Returns
    -------
    float, array of floats
        Reaction rate coefficient <sigma*v> averaged over a Maxwellian in units of cm^3/s
    """
    return a * npsqrt(T / I) / (I ** (1.5) * (b + T / I)) * npexp(-I / T)


def rate_lambda(I):
    return lambda T, a, b: rate_coeff(T, I, a, b)
