import mpmath as mp
import numpy as np
from pathlib import Path
from ..vars._cons import (
    M_e_eV,
    qe_cgs,
    atm_cross_cgs,
    Ry_eV,
    E_ion,
    E_21p,
    I_ion,
    I_21p,
    I_double,
    c_cgs,
)

a215 = [
    -7.7782130e2,
    9.5401909e2,
    -5.2277670e2,
    1.5927011e2,
    -2.9525572e1,
    3.4130241e0,
    -2.4055208e-1,
    9.4651813e-3,
    -1.5943253e-4,
]

A_HEII_11s = [5.857e-1, -4.457e-1, 7.680e-1, -2.521, 3.317, 0.0]

# ── EII cross section lookup tables (loaded at import time) ──────────────────
_VARS_DIR = Path(__file__).parent.parent / "vars"

_h_data = np.loadtxt(_VARS_DIR / "h_eii_cross.csv", delimiter=",", comments="#")
_H_LOG_E = np.log(_h_data[:, 0])
_H_LOG_SIGMA = np.log(_h_data[:, 1])

_he_data = np.loadtxt(_VARS_DIR / "he_eii_cross.csv", delimiter=",", comments="#")
_HE_LOG_EPS = np.log(_he_data[:, 0])
_HE_LOG_SIGMA = np.log(_he_data[:, 1])

_HE_ION_RATE_PATH = _VARS_DIR / "he_ion_rate.csv"
if _HE_ION_RATE_PATH.exists():
    _he_ion_rate_data = np.loadtxt(
        _HE_ION_RATE_PATH, delimiter=",", comments="#"
    )
    _HE_ION_LOG_T = np.log(_he_ion_rate_data[:, 0])
    _HE_ION_LOG_RATE = np.log(_he_ion_rate_data[:, 1])
else:
    _HE_ION_LOG_T = None
    _HE_ION_LOG_RATE = None


def H_EII_cross(E, A=a215):
    """
    Hydrogen electron impact ionization cross section [cm^2].

    Parameters
    ----------
    E : float
        Beam energy [eV].
    A : list
        Janev polynomial coefficients (default: a215).
    """
    return np.exp(np.sum([a * np.log(E) ** i for i, a in enumerate(A)]))


def He_EII_cross(eps, A):
    """
    Helium electron impact ionization cross section [cm^2].

    Parameters
    ----------
    eps : float
        Beam energy scaled by reaction threshold energy (E / IE_He).
    A : list of floats
        Coefficients for the cross section calculation.
    """
    a = mp.fdiv(mp.mpf(1e-13), mp.fmul(eps, mp.power(E_ion, 2)))
    b = mp.fdiv(1, eps)
    term1 = mp.fmul(A[0], mp.log(eps))
    term2 = mp.fsum([mp.fmul(A[i], mp.power(mp.fsub(1, b), i)) for i in range(1, 6)])
    return mp.fmul(a, mp.fadd(term1, term2))


def H_EII_cross_lkup(E):
    """
    Hydrogen electron impact ionization cross section [cm^2] via lookup table.

    Log-linear interpolation over h_eii_cross.csv (13.6–1000 eV).

    Parameters
    ----------
    E : float
        Beam energy [eV].
    """
    return float(np.exp(np.interp(np.log(E), _H_LOG_E, _H_LOG_SIGMA,
                                  left=_H_LOG_SIGMA[0], right=_H_LOG_SIGMA[-1])))


def He_EII_cross_lkup(eps):
    """
    Helium electron impact ionization cross section [cm^2] via lookup table.

    Log-linear interpolation over he_eii_cross.csv (eps = 1.001–40.67),
    generated with a_11s coefficients.

    Parameters
    ----------
    eps : float
        Beam energy scaled by He ionization threshold (E / IE_He).
    """
    return float(np.exp(np.interp(np.log(eps), _HE_LOG_EPS, _HE_LOG_SIGMA,
                                   left=_HE_LOG_SIGMA[0], right=_HE_LOG_SIGMA[-1])))


def He_ion_rate_lkup(T):
    """Helium Maxwellian ionization rate coefficient [cm^3/s].

    Interpolates ``he_ion_rate.csv`` in log-temperature/log-rate space.  Values
    outside the tabulated 0.1--100 eV interval use the nearest endpoint.

    Parameters
    ----------
    T : float or array-like
        Electron temperature [eV]. Must be positive.
    """
    if _HE_ION_LOG_T is None:
        raise FileNotFoundError(
            f"Helium ionization-rate table not found: {_HE_ION_RATE_PATH}"
        )

    temperature = np.asarray(T, dtype=float)
    if np.any(temperature <= 0):
        raise ValueError("Electron temperature must be positive")

    log_rate = np.interp(
        np.log(temperature),
        _HE_ION_LOG_T,
        _HE_ION_LOG_RATE,
        left=_HE_ION_LOG_RATE[0],
        right=_HE_ION_LOG_RATE[-1],
    )
    rate = np.exp(log_rate)
    return float(rate) if rate.ndim == 0 else rate


def He_EIE_cross_DA(eps, A):
    """
    Helium electron impact excitation cross section.

    Parameters
    ----------
    eps : float
        Energy scaled by threshold energy
    A : list
        Coefficients for the cross section calculation

    Returns
    -------
    mpf
        Helium electron impact excitation cross section in cm^2
    """
    factor1 = mp.fdiv(mp.fmul(atm_cross_cgs, Ry_eV), mp.fmul(eps, E_21p))
    factor2 = mp.fadd(
        mp.fmul(A[0], mp.log(eps)),
        mp.fsum([A[i] * (eps ** (1 - i)) for i in range(1, 5)]),
    )
    factor3 = mp.fdiv(mp.fadd(eps, 1), mp.fadd(eps, A[5]))
    return mp.fprod([factor1, factor2, factor3])


def int_factor(I):
    return mp.fprod(
        [mp.power(I, 2), mp.sqrt(mp.fdiv(8, mp.fmul(M_e_eV, mp.pi))), c_cgs]
    )


def rate_kern(cross_sec_func, eps, a, T, I):
    # I is the minimum reaction energy threshold
    x = mp.fmul(eps, mp.fdiv(I, T))
    int_kern = mp.fprod([cross_sec_func(eps, a), eps, mp.exp(-x)])
    return int_kern


def kern_lambda(cross_sec_func, a, T, I):
    return lambda eps: rate_kern(cross_sec_func, eps, a, T, I)


def H_ion_rate(E, T):
    return 10e-5 * np.sqrt(T / E) / (E**1.5 * (6.0 + (T / E))) * np.exp(-E / T)


def alpha_r(T, I=None):
    """
    Radiative recombination rate coefficient [cm³/s].

    If ionization energy I is provided, uses the Seaton (1959) hydrogenic formula:
        alpha_r = 5.2e-14 * (I/T)^(1/2) * (0.43 + 0.5*ln(I/T) + 0.496*(I/T)^(-1/3))

    Otherwise falls back to the approximate power-law:
        alpha_r ≈ 2.71e-13 * T^(-0.5)   [valid for T ~ 1–20 eV]

    Parameters
    ----------
    T : float or array
        Electron temperature [eV].
    I : float or None
        Ionization energy [eV]. If provided, uses the Seaton formula.

    Returns
    -------
    float or array
        Radiative recombination rate coefficient [cm³/s].
    """
    if I is not None:
        x = I / T
        return 5.2e-14 * x**0.5 * (0.43 + 0.5 * np.log(x) + 0.496 * x ** (-1 / 3))
    return 2.71e-13 * T ** (-0.5)


def alpha_3(T):
    """
    Three-body recombination rate coefficient [cm⁶/s] — approximate power-law fit.

    alpha_3(T) ≈ 8.75e-27 * T^(-4.5)

    NOTE: This is a rough power-law approximation. For helium, species-specific
    rates should be used when available.

    Parameters
    ----------
    T : float or array
        Electron temperature [eV].

    Returns
    -------
    float or array
        Three-body recombination rate coefficient [cm⁶/s].
    """
    return 8.75e-27 * T ** (-4.5)


def integrate_kern(cross_sec_func, a, T, I):
    """Maxwellian-average a threshold cross section over electron energy.

    The cross section receives ``eps = E / I``.  A shifted integration variable
    ``z = (E - I) / T`` keeps the near-threshold peak resolved when ``T << I``.
    """
    temperatures = np.asarray(T, dtype=float)
    if temperatures.ndim != 1:
        raise ValueError("T must be a one-dimensional array of temperatures")
    if np.any(temperatures <= 0):
        raise ValueError("Temperatures must be positive")

    scale_factor = int_factor(I)
    rate_coeff = np.empty(temperatures.shape)
    for i, temp in enumerate(temperatures):
        temp_mp = mp.mpf(temp)
        ratio = mp.fdiv(temp_mp, I)

        def shifted_integrand(z):
            eps = mp.fadd(1, mp.fmul(ratio, z))
            return mp.fprod([cross_sec_func(eps, a), eps, mp.exp(-z)])

        integral = mp.fprod(
            [
                mp.exp(mp.fneg(mp.fdiv(I, temp_mp))),
                ratio,
                mp.quad(shifted_integrand, [0, mp.inf]),
            ]
        )
        rate_coeff[i] = mp.fprod(
            [integral, scale_factor, mp.power(temp_mp, mp.fneg(1.5))]
        )
    return rate_coeff


A_R318 = [
    [
        -1.831670498376e01,
        2.143624996483e-01,
        5.139117192662e-02,
        -9.896180369559e-04,
        -2.495327546080e-03,
        -2.417046684097e-05,
        1.177406072793e-04,
        -1.483036457978e-05,
        5.351909441226e-07,
    ],
    [
        1.650239332070e-01,
        -1.067658289373e-01,
        9.536923957409e-03,
        9.536923957409e-03,
        6.315097684976e-03,
        -1.265503371044e-03,
        -6.945512319613e-05,
        3.698501620365e-05,
        -3.348172574417e-06,
        9.728230870242e-08,
    ],
    [
        5.025740610454e-02,
        -5.304993033743e-03,
        -1.306075129405e-02,
        2.655464630308e-03,
        7.569269700468e-04,
        -2.956984088728e-04,
        3.424317896619e-05,
        -1.527018819072e-06,
        1.676354786072e-08,
    ],
    [
        5.288358515136e-03,
        8.289383645942e-03,
        -1.033166370333e-03,
        -1.365781346175e-03,
        2.756946036257e-04,
        2.318277483195e-05,
        -9.815693511794e-06,
        8.362050692462e-07,
        -2.237567830699e-08,
    ],
    [
        -2.437122342843e-03,
        -9.698773663345e-05,
        1.280464204775e-03,
        -1.859939123743e-04,
        -1.107375149384e-04,
        3.704494397140e-05,
        -4.285719813022e-06,
        2.058392726953e-07,
        -3.081685803820e-09,
    ],
    [
        -4.461891214720e-04,
        -4.470180279338e-04,
        -8.453294908907e-05,
        1.237942304972e-04,
        -7.217379426085e-06,
        -6.066558692480e-06,
        1.169257650609e-06,
        -7.463594884928e-08,
        1.450862501121e-09,
    ],
    [
        1.731631548110e-04,
        7.944326905066e-05,
        -3.040874906105e-05,
        -1.588253432932e-05,
        5.769971321188e-06,
        -4.951573401626e-07,
        -4.968953461875e-10,
        5.924370389093e-10,
        4.434231893204e-11,
    ],
    [
        -1.588434781959e-05,
        -5.303688417551e-06,
        4.747888095498e-06,
        6.603560345800e-07,
        -6.717311113584e-07,
        1.437520597154e-07,
        -1.618948982477e-08,
        1.078208689229e-09,
        -3.324377862622e-11,
    ],
    [
        4.482291414386e-07,
        1.235167254501e-07,
        -1.923953750574e-07,
        -1.970606344918e-09,
        2.440961351104e-08,
        -6.998724470004e-09,
        9.440094842562e-10,
        -6.619767848464e-11,
        1.935019679501e-12,
    ],
]

A_R531 = [
    [
        -1.992795874184e01,
        2.342319832717e-01,
        5.150488618567e-02,
        -4.457831664145e-03,
        -1.543592188979e-03,
        3.127935819690e-04,
        -1.478649318411e-05,
        -4.796924334410e-07,
        3.623344342191e-08,
    ],
    [
        1.866121633782e-01,
        -1.085479286023e-01,
        5.502643799842e-03,
        6.751016280248e-03,
        -9.368501420643e-04,
        -1.564547327374e-04,
        4.454044051200e-05,
        -3.559977035839e-06,
        9.673665606073e-08,
    ],
    [
        5.632774905403e-02,
        -5.796164637185e-03,
        -1.070448355458e-02,
        1.348104812381e-03,
        6.678034019800e-04,
        -1.638591652038e-04,
        1.091423551219e-05,
        1.429006251276e-08,
        -1.626046817162e-08,
    ],
    [
        -1.523524839309e-03,
        7.964340512260e-03,
        -2.811049856343e-04,
        -1.009960297392e-03,
        1.271484361215e-04,
        2.743635453507e-05,
        -6.757942983709e-06,
        4.890540879010e-07,
        -1.188613673713e-08,
    ],
    [
        -2.153750537851e-03,
        -2.259674261582e-04,
        8.802921038196e-04,
        -2.397230757181e-05,
        -7.802314691135e-05,
        1.364264736037e-05,
        -3.524476702306e-07,
        -6.293604516614e-08,
        3.383446775161e-09,
    ],
    [
        3.308881419986e-04,
        -3.444207072047e-04,
        -1.198248793959e-04,
        5.675989835329e-05,
        5.096915155186e-06,
        -3.146182664420e-06,
        2.858958575343e-07,
        -2.702231414235e-09,
        -3.684509743141e-10,
    ],
    [
        -1.293912998397e-05,
        6.164032099379e-05,
        -7.766013174537e-07,
        -8.581799112319e-06,
        1.122093772403e-06,
        9.721160837100e-08,
        -1.739320039203e-08,
        5.123073994873e-11,
        3.976640871840e-11,
    ],
    [
        -4.067520041201e-07,
        -4.156975252360e-06,
        9.213287306897e-07,
        4.932159877322e-07,
        -1.600545758983e-07,
        1.677569938034e-08,
        -1.179041806243e-09,
        9.702183602739e-11,
        -4.096335085504e-12,
    ],
    [
        2.451185017055e-08,
        1.009896955766e-07,
        -4.014244306169e-08,
        -9.746139175914e-09,
        5.662679103625e-09,
        -9.661966135256e-10,
        9.352212060009e-11,
        -5.755013542991e-12,
        1.667878217006e-13,
    ],
]


def heavy_reaction(T, E, A):
    """
    Heavy-particle reaction rate coefficient from a 2-D polynomial fit in log-log space.

    ln(<sigma*v>) = sum_{i,j} A[i][j] * ln(E)^i * ln(T)^j

    Coefficient tables A_R318 (H + H⁺ charge exchange) and A_R531 (He + He⁺ charge
    exchange) follow the IAEA heavy-particle reaction data format.

    Parameters
    ----------
    T : float or array
        Ion temperature [eV].
    E : float
        Reaction energy scaling parameter [eV] (typically 0.1 eV for cx tables).
    A : list of lists
        2-D array of polynomial coefficients A[i][j].

    Returns
    -------
    float or array
        Reaction rate coefficient [cm³/s].
    """
    ln_sigv = 0
    for i in range(len(A)):
        for j in range(len(A[i])):
            ln_sigv += A[i][j] * (np.log(E) ** i) * (np.log(T) ** j)
    return np.exp(ln_sigv)


temps = np.logspace(-1, 4, 1000)
_cx_He = heavy_reaction(temps, 0.1, A_R531)
_cx_H = heavy_reaction(temps, 0.1, A_R318)


def charge_ex_react(T, gas_type="He"):
    """
    Charge-exchange reaction rate coefficient [cm³/s] via table interpolation.

    Pre-computed tables (_cx_He, _cx_H) are built at import time from
    heavy_reaction() over T = 0.1–10,000 eV. Linear interpolation is used.

    Parameters
    ----------
    T : float or array
        Ion temperature [eV].
    gas_type : str
        Gas species: "He" for helium (A_R531 table) or "H" for hydrogen (A_R318 table).

    Returns
    -------
    float or array
        Charge-exchange rate coefficient [cm³/s].
    """
    if gas_type == "He":
        table = _cx_He
    elif gas_type == "H":
        table = _cx_H
    else:
        raise ValueError(f"unsupported gas_type {gas_type!r}; expected 'He' or 'H'")
    return np.interp(T, temps, table)
