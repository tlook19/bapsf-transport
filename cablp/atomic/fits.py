from numpy import exp as npexp, sqrt as npsqrt
from math import exp as mexp, sqrt as msqrt
from ..constants import qe_SI

# note, revisit this when it is determined whether these functions are vectorized in the simulation


def IAEA_exp1(t, a):
    """
    IAEA fit expression 1 for electron cooling / excitation rate coefficients [cm³/s].

    R(T) = a[0] * exp(-a[1] / T^a[2]) / (T^a[3] + a[4] * T^a[5])

    where T is converted internally to keV. Used for He I and H I neutral cooling.

    Parameters
    ----------
    t : float or array
        Electron temperature [eV].
    a : list or array
        Six IAEA fit coefficients [a0 … a5].

    Returns
    -------
    float or array
        Rate coefficient [cm³/s].
    """
    T = t * 1e-3  # convert to keV
    P = a[0] * npexp(-a[1] / (T ** a[2])) / (T ** a[3] + a[4] * T ** a[5])
    return P * 1e-27 / qe_SI


def IAEA_exp4(t, a, recomb=True):
    """
    IAEA fit expression 4 for electron cooling rate coefficients [cm³/s].

    R(T) = a[0] * exp(-a[1] / T^a[2]) / (T^a[3] + a[4] * T^a[5]) [+ a[6] * T^a[7]]

    The optional recombination term a[6]*T^a[7] is included when recomb=True.
    Used for He II ion cooling.

    Parameters
    ----------
    t : float or array
        Electron temperature [eV].
    a : list or array
        Eight IAEA fit coefficients [a0 … a7].
    recomb : bool
        If True, include the recombination contribution a[6]*T^a[7]; default True.

    Returns
    -------
    float or array
        Rate coefficient [cm³/s].
    """
    T = t * 1e-3  # convert to keV
    b = a[0] * npexp(-a[1] / T ** a[2]) / (T ** a[3] + a[4] * T ** a[5])
    if recomb:
        b += a[6] * T ** a[7]
    return b * 1e-27 / qe_SI


def IAEA_exp6(t, a):
    """
    IAEA fit expression 6 for electron cooling rate coefficients [cm³/s].

    R(T) = a[0] * T^a[1] + a[2] * T^a[3] + a[4] * T^a[5]

    Used for H II ion cooling.

    Parameters
    ----------
    t : float or array
        Electron temperature [eV].
    a : list or array
        Six IAEA fit coefficients [a0 … a5].

    Returns
    -------
    float or array
        Rate coefficient [cm³/s].
    """
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
