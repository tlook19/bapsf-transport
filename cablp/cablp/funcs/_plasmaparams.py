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

    v_th = sqrt(T / m_e) in CGS, evaluated as 4.19e7 * sqrt(T[eV]).

    This is the NRL Formulary convention, with NO factor of 2 under the root;
    the sqrt(2 T / m_e) spelling would run 5.93e7. The distinction is
    load-bearing rather than cosmetic: the 4.19e7 constant is what makes the
    parallel conductivity built on this speed reproduce Braginskii's 3.16
    coefficient exactly.

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


#: Floor [dimensionless] applied to :func:`c_log` by the transport and
#: exchange estimates that consume it. The fit expression goes negative in
#: the cold, tenuous corner of the discharge, where ln(Lambda) is outside the
#: weak-coupling regime it was derived in; clamping at 1 keeps the collision
#: rates finite and positive there.
LN_LAMBDA_MIN = 1.0


def c_log(Te, n, kind="ei"):
    """
    Coulomb logarithm ln(Lambda) for electron-ion collisions.

    Parameters
    ----------
    Te : float or array
        Electron temperature [eV].
    n : float or array
        Electron density [cm\u207b\u00b3].
    kind : str
        Collision type. ``"ei"`` -- electron-ion (NRL 2019, p. 34, case (b));
        switches formula at Te = 10 eV -- is the only accepted value, and
        anything else raises ``ValueError``. The NRL Formulary prints its
        Coulomb-logarithm cases as a LETTERED list, not as numbered
        equations, so the case is cited here by page and case rather than by
        an equation number.

        The parameter survives the collapse to a single case because every
        call site names it: passing it is how a caller states which Coulomb
        log it means, and a wrong name must fail loudly rather than return
        the electron-ion one under another label.

    Returns
    -------
    float or array
        Coulomb logarithm (dimensionless).

    Raises
    ------
    ValueError
        If ``kind`` is anything other than ``"ei"``.
    """
    if kind != "ei":
        raise ValueError(
            f"c_log(kind={kind!r}) is not available: the only collision type "
            "this helper computes is 'ei' (electron-ion, NRL 2019 p. 34 case "
            "(b))."
        )
    return np.where(
        Te > 10,
        24 - np.log(np.sqrt(n) / Te),
        23 - np.log(np.sqrt(n) * Te**-1.5),
    )
