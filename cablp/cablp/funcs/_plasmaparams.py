import numpy as np


def time_elec_coll(T, ne, lnlambda):
    """
    Electron collision time [s] (Braginskii).

    Parameters
    ----------
    T : float or array
        Electron temperature [eV].
    ne : float or array
        Electron density [cm⁻³].
    lnlambda : float or array
        Coulomb logarithm (dimensionless).

    Returns
    -------
    float or array
        Electron-electron collision time [s].
    """
    return 3.44e5 * (T ** (3 / 2)) / ne / lnlambda


def time_ion_coll(T, n, mu, lnlambda):
    """
    Ion collision time [s] (Braginskii).

    Parameters
    ----------
    T : float or array
        Ion temperature [eV].
    n : float or array
        Ion density [cm⁻³].
    mu : float
        Ion mass number (A = m_ion / m_proton).
    lnlambda : float or array
        Coulomb logarithm (dimensionless).

    Returns
    -------
    float or array
        Ion-ion collision time [s].
    """
    return 2.09e7 * (T ** (3 / 2)) / n / lnlambda * np.sqrt(mu)


def v_thm_e(T):
    """
    Electron thermal velocity [cm/s].

    v_th = sqrt(2 * T / m_e) in CGS, evaluated as 4.19e7 * sqrt(T[eV]).

    Parameters
    ----------
    T : float or array
        Electron temperature [eV].

    Returns
    -------
    float or array
        Electron thermal velocity [cm/s].
    """
    return 4.19e7 * np.sqrt(T)


def v_ion_speed(T, mu, Z=1, gamma=1):
    """
    Ion characteristic speed [cm/s].

    With default Z=1, gamma=1 and T = Ti, returns the ion thermal velocity.
    With T = Te, returns the Bohm (ion sound) speed C_s = sqrt(gamma*Z*Te/m_i).

    Parameters
    ----------
    T : float or array
        Temperature [eV] (ion thermal velocity if T=Ti; sound speed if T=Te).
    mu : float
        Ion mass number (m_ion / m_proton).
    Z : float
        Ion charge number; default 1.
    gamma : float
        Adiabatic index; default 1.

    Returns
    -------
    float or array
        Ion speed [cm/s].
    """
    # if T = Ti then this gives the ion thermal velocity. Z and Gamma should be 1
    # if T = Te then this gives the ion sound speed. Z and Gamma can be different from 1
    return 9.79e5 * np.sqrt(gamma * Z * T / mu)


def elec_gyro_freq(B):
    """
    Electron cyclotron (gyro) frequency [rad/s].

    Parameters
    ----------
    B : float or array
        Magnetic field [Gauss].

    Returns
    -------
    float or array
        Electron gyrofrequency [rad/s].
    """
    return 1.76e7 * B


def ion_gyro_freq(B, mu, Z=1):
    """
    Ion cyclotron (gyro) frequency [rad/s].

    Parameters
    ----------
    B : float or array
        Magnetic field [Gauss].
    mu : float
        Ion mass number (m_ion / m_proton).
    Z : float
        Ion charge number; default 1.

    Returns
    -------
    float or array
        Ion gyrofrequency [rad/s].
    """
    return 9.58e3 * Z * B / mu


def c_log(Te, n, kind="ee"):
    """
    Coulomb logarithm ln(Lambda) for electron-electron or electron-ion collisions.

    Parameters
    ----------
    Te : float or array
        Electron temperature [eV].
    n : float or array
        Electron density [cm⁻³].
    kind : str
        Collision type:
        - ``"ee"`` : electron-electron (NRL 2019, eq. 2-5); valid for T > 0.1 eV.
        - ``"ei"`` : electron-ion (NRL 2019, eq. 2-3/2-4); switches formula at Te = 10 eV.
        - anything else : simplified formula 23.4 - 1.15*log10(n) + 3.45*log10(Te).

    Returns
    -------
    float or array
        Coulomb logarithm (dimensionless).
    """
    if kind == "ee":
        return (
            23.5
            - np.log(np.sqrt(n) * Te ** (-5 / 4))
            - np.sqrt(1e-5 + ((np.log(Te) - 2) ** 2) / 16)
        )
    elif kind == "ei":
        result = np.where(
            Te > 10,
            24 - np.log(np.sqrt(n) / Te),
            23 - np.log(np.sqrt(n) * Te**-1.5),
        )
        return result
    else:
        return 23.4 - 1.15 * np.log10(n) + 3.45 * np.log10(Te)
