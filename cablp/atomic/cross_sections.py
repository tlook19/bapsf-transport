import math

import mpmath as mp
import numpy as np
from pathlib import Path
from ..numerics.interp import interp_scalar_fused as _interp_scalar_fused
from ..constants import (
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
    ev_to_erg,
    m_He_cgs,
    m_e_cgs,
)

# ── THE HYDROGEN QUARANTINE (ruled 2026-08-27) ───────────────────────────────
#
# This module's hydrogen arms RAISE. They are quarantined, not removed: the
# code and every coefficient table stay exactly where they are, so a validated
# re-opening deletes a guard rather than rewriting an arm.
#
# Two independent reasons, and either alone is sufficient:
#
#   1. UNTESTED DOMAIN. The solver is hard helium-only (D3, 2026-08-21):
#      LAPDSim1D refuses gas_type != "He" at construction, so no hydrogen arm
#      here has a solver-path consumer and none is covered by the golden, the
#      digest gate or the smoke suite. Every H result this module can produce
#      is therefore unexercised by any gate in the repository.
#
#   2. ONE CORRUPT TABLE -- SINCE REPAIRED, 2026-08-27 ([sbq:L295]).
#      ``A_R318`` -- the H + H+ charge-exchange fit -- CARRIED a DUPLICATED
#      coefficient: row 1 repeated 9.536923957409e-03 and so had 10 entries
#      where every other row of A_R318, and every row of the helium A_R531,
#      has 9. ``heavy_reaction`` iterates ``range(len(A[i]))``, so the ragged
#      row silently contributed an extra polynomial term rather than failing;
#      the import-time ``_cx_H`` table was built from it without complaint.
#      The duplicate has been deleted and all 81 coefficients digit-proofed
#      against a re-fetched IAEA HYDHEL 3.1.8 (see the provenance comment at
#      the table). A_R531 was never affected -- the helium arm was untouched
#      by the defect and by the repair.
#
#      THE QUARANTINE STANDS REGARDLESS: reason 1 is sufficient on its own,
#      and repairing the table did not give the hydrogen arms a gate.
#
# Guarded entry points, the narrowest set covering every H route into this
# module: H_EII_cross, H_EII_cross_lkup, and charge_ex_react's gas_type == "H"
# branch.
#
# NOT guarded, deliberately: fits.py's IAEA_exp1/exp4/exp6 and rate_coeff are
# gas-AGNOSTIC fit forms -- IAEA_exp1 is evaluated on the helium aHeI and the
# hydrogen aHI alike, so the species lives in the caller's coefficient table,
# not in the function. Likewise alpha_r/alpha_3 take the ionization potential
# as an argument and are used on the helium path. Guarding any of those would
# refuse helium.

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
_DATA_DIR = Path(__file__).parent / "data"

_h_data = np.loadtxt(_DATA_DIR / "h_eii_cross.csv", delimiter=",", comments="#")
_H_LOG_E = np.log(_h_data[:, 0])
_H_LOG_SIGMA = np.log(_h_data[:, 1])

_he_data = np.loadtxt(_DATA_DIR / "he_eii_cross.csv", delimiter=",", comments="#")
_HE_LOG_EPS = np.log(_he_data[:, 0])
_HE_LOG_SIGMA = np.log(_he_data[:, 1])

# Python-list twins of the He EII table, for the SCALAR lookup below.
# ``interp_scalar_fused`` indexes its table inside a binary search; indexing a
# list hands back a float, where indexing an ndarray builds a numpy scalar per
# probe -- and that dominates the CSDA march, which takes ~200k lookups per
# solver step. Same float64 values, so the interpolated result is bit-identical.
# The ndarrays above stay: the compiled kernel's table view needs buffers, and
# ``scripts/gates/interp_fused_reference.py`` reads them by name.
_HE_LOG_EPS_SEQ = _HE_LOG_EPS.tolist()
_HE_LOG_SIGMA_SEQ = _HE_LOG_SIGMA.tolist()

_HE_ION_RATE_PATH = _DATA_DIR / "he_ion_rate.csv"
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

    Raises
    ------
    ValueError
        Always -- this is a quarantined hydrogen entry point. See the hydrogen
        quarantine note at the top of this module.
    """
    raise ValueError(
        "H_EII_cross is not available: the hydrogen arms of cablp.atomic are "
        "QUARANTINED (untested domain -- no solver-path consumer and no gate "
        "coverage), ruled 2026-08-27. The quarantine's second ground, the "
        "corrupt A_R318 table, was repaired the same day and no longer "
        "applies; the untested domain alone is sufficient. The solver is "
        "helium-only (D3, 2026-08-21). Accepted: He -- use He_EII_cross."
    )
    # RETAINED, not removed: the quarantine is reversible by construction, so
    # a validated re-opening deletes the raise above and this line stands.
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

    Raises
    ------
    ValueError
        Always -- this is a quarantined hydrogen entry point. See the hydrogen
        quarantine note at the top of this module.
    """
    raise ValueError(
        "H_EII_cross_lkup is not available: the hydrogen arms of cablp.atomic "
        "are QUARANTINED (untested domain -- no solver-path consumer and no "
        "gate coverage), ruled 2026-08-27. The quarantine's second ground, "
        "the corrupt A_R318 table, was repaired the same day and no longer "
        "applies; the untested domain alone is sufficient. The solver is "
        "helium-only (D3, 2026-08-21). Accepted: He -- use He_EII_cross_lkup."
    )
    # RETAINED, not removed: the quarantine is reversible by construction, so
    # a validated re-opening deletes the raise above and this line stands.
    return float(np.exp(_interp_scalar_fused(np.log(E), _H_LOG_E, _H_LOG_SIGMA,
                                             left=_H_LOG_SIGMA[0],
                                             right=_H_LOG_SIGMA[-1])))


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
    # math.log/math.exp rather than the numpy scalars: same float64 result
    # (numpy dispatches to the same libm for a scalar argument), ~2x cheaper
    # per call, and this is the march's innermost lookup.
    return math.exp(_interp_scalar_fused(math.log(eps), _HE_LOG_EPS_SEQ,
                                         _HE_LOG_SIGMA_SEQ,
                                         left=_HE_LOG_SIGMA_SEQ[0],
                                         right=_HE_LOG_SIGMA_SEQ[-1]))


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


_ATM_CROSS_CGS_F = float(atm_cross_cgs)
_RY_EV_F = float(Ry_eV)


def He_EIE_omega_allowed(eps, A):
    """Dipole-allowed collision strength Omega(x) [dimensionless], float.

    Eq. (2) of Ralchenko et al., ADNDT 94 (2008) 603: 6-coefficient fit,
    eps = E / E_threshold > 1. Same formula as the mpmath ``He_EIE_cross_DA``
    without the cross-section conversion factor (which that function ties to
    the 2^1P threshold).
    """
    return (
        A[0] * math.log(eps) + sum(A[i] * eps ** (1 - i) for i in range(1, 5))
    ) * (eps + 1.0) / (eps + A[5])


def He_EIE_omega_forbidden(eps, A):
    """Dipole-forbidden collision strength Omega(x) [dimensionless], float.

    Eq. (3) of Ralchenko et al., ADNDT 94 (2008) 603: 5-coefficient fit,
    eps = E / E_threshold > 1.
    """
    return sum(A[i] * eps ** (-i) for i in range(4)) * eps**2 / (eps**2 + A[4])


def He_EIE_cross_manifold(E_eV, entry):
    """Excitation cross section [cm^2] from He 1^1S at electron energy E_eV.

    ``entry`` is a value of ``atomic.coefficients.He_singlet_manifold``:
    ``{"E_eV": threshold, "form": "allowed"|"forbidden", "A": [...]}``.
    Conversion per the paper's Eq. (1) with g_i = 1:
    sigma = pi*a0^2 * Ry / E * Omega(E/E_th). Returns 0.0 at or below
    threshold; any near-threshold negative wiggle of the fitted Omega is
    clamped to zero.
    """
    E_th = entry["E_eV"]
    if E_eV <= E_th:
        return 0.0
    eps = E_eV / E_th
    if entry["form"] == "allowed":
        omega = He_EIE_omega_allowed(eps, entry["A"])
    elif entry["form"] == "forbidden":
        omega = He_EIE_omega_forbidden(eps, entry["A"])
    else:
        raise ValueError(f"unknown manifold fit form {entry['form']!r}")
    return _ATM_CROSS_CGS_F * _RY_EV_F / E_eV * max(omega, 0.0)


def He_singlet_tail_levels(n_max=20):
    """The n >= 5 singlet Rydberg levels for the Eq. (5) tail, per series.

    Yields ``(series, n, E_th_eV, scale)`` where ``scale = (4/n)^3`` and
    E_th = E_lim - Ry/(n - delta)^2 with the per-series quantum defect
    (``atomic.coefficients.He_singlet_quantum_defect``). The n^-3 sum truncated at
    ``n_max = 20`` leaves < 0.1% of the summed tail uncounted.
    """
    from .coefficients import He_ionization_limit_eV, He_singlet_quantum_defect

    for series in ("S", "P", "D", "F"):
        delta = He_singlet_quantum_defect[series]
        for n in range(5, n_max + 1):
            E_th = He_ionization_limit_eV - _RY_EV_F / (n - delta) ** 2
            yield series, n, E_th, (4.0 / n) ** 3


def He_singlet_tail_cross(E_eV, n_max=20):
    """Summed n >= 5 singlet-excitation tail from He 1^1S: (sigma, sigma*E).

    Applies the paper's Eq. (5) scaling to the n = 4 rows of the manifold:
    sigma(1^1S -> n^1L, E) = (4/n)^3 * sigma(1^1S -> 4^1L, E/eps_tilde) with
    eps_tilde = E_th(n)/E_th(4). Born-derived for the allowed series,
    classical for the forbidden ones — a stated remainder estimate, not a
    fitted cross section. Returns ``(sigma_tot_cm2, sigma_E_tot_cm2_eV)``
    where the second entry books each level's threshold as radiated energy.
    """
    from .coefficients import He_singlet_manifold

    base = {s: He_singlet_manifold[f"41{s}"] for s in ("S", "P", "D", "F")}
    sigma_tot = 0.0
    sigma_E_tot = 0.0
    for series, _n, E_th, scale in He_singlet_tail_levels(n_max):
        b = base[series]
        eps_tilde = E_th / b["E_eV"]
        sigma = scale * He_EIE_cross_manifold(E_eV / eps_tilde, b)
        sigma_tot += sigma
        sigma_E_tot += sigma * E_th
    return sigma_tot, sigma_E_tot


def He_beam_excitation_channel(E_eV, n_max=20):
    """Summed He singlet excitation channel for a monoenergetic beam.

    Returns ``(sigma_tot_cm2, E_rad_mean_eV)``: the total 1^1S excitation
    cross section over the fitted n <= 4 manifold plus the Eq. (5) n >= 5
    tail, and the energy-weighted mean radiated energy per event
    (each event books its threshold E_k as prompt line radiation; the 2^1S
    metastable caveat is documented on ``atomic.coefficients.He_singlet_manifold``).
    ``(0.0, 0.0)`` below the lowest threshold (2^1S, 20.6158 eV).

    This is the measured replacement for the historical
    ``b_beam_excitation = 1.4`` estimate (WP-A):
    sigma_tot / sigma_2P = 1.65-1.75 and E_rad_mean = 21.95-21.98 eV over
    the 60-180 eV beam range (``scripts/atomic/measure_beam_manifold.py``).
    """
    from .coefficients import He_singlet_manifold

    sigma_tot = 0.0
    sigma_E_tot = 0.0
    for entry in He_singlet_manifold.values():
        sigma = He_EIE_cross_manifold(E_eV, entry)
        sigma_tot += sigma
        sigma_E_tot += sigma * entry["E_eV"]
    tail_sigma, tail_sigma_E = He_singlet_tail_cross(E_eV, n_max=n_max)
    sigma_tot += tail_sigma
    sigma_E_tot += tail_sigma_E
    if sigma_tot <= 0.0:
        return 0.0, 0.0
    return sigma_tot, sigma_E_tot / sigma_tot


# Lookup-table front end for the summed singlet channel. Profiling
# (2026-07-21, nx=120 csda_ql production config) put the scalar manifold
# sums at ~80% of total step time via deposit_beam's per-substep calls
# (~240 channel calls x 73 scalar level evaluations per step). The table
# is built lazily ONCE from the exact function above, so its nodes are
# exactly the reference values; between nodes it is linear interpolation
# on (sigma, sigma*E) -- interpolating the pair keeps the recovered
# E_rad = sigmaE/sigma inside the physical threshold range. Grid: 2 meV
# spacing across the threshold cluster (20-25 eV, where the curve kinks
# at each level), log-spaced (0.15% steps) on the smooth 25-2000 eV
# decay; measured relative error ~1e-6 away from thresholds. Callers
# above the table span fall back to the exact function. The frozen
# voltage-driven solver keeps calling the exact function directly --
# only the deposition hot loop opts in (deliberate: the frozen path
# stays bit-stable).
_HE_BEAM_EXC_TABLE = None
_HE_BEAM_EXC_SEQ = None


def _he_beam_excitation_table(n_max=20):
    global _HE_BEAM_EXC_TABLE
    if _HE_BEAM_EXC_TABLE is None or _HE_BEAM_EXC_TABLE[0] != n_max:
        E_grid = np.concatenate(
            [
                np.linspace(20.0, 25.0, 2501),
                np.geomspace(25.0, 2000.0, 3000)[1:],
            ]
        )
        sigma = np.empty_like(E_grid)
        sigma_E = np.empty_like(E_grid)
        for k, e in enumerate(E_grid):
            s, e_rad = He_beam_excitation_channel(float(e), n_max=n_max)
            sigma[k] = s
            sigma_E[k] = s * e_rad
        _HE_BEAM_EXC_TABLE = (int(n_max), E_grid, sigma, sigma_E)
    return _HE_BEAM_EXC_TABLE


def _he_beam_excitation_seq(n_max=20):
    """``(E_grid, sigma, sigma_E)`` of the table above, as Python lists.

    Same float64 values -- see the note on ``_HE_LOG_EPS_SEQ`` for why the
    scalar lookup indexes lists while the ndarrays stay for the compiled
    kernel's table view. Keyed on the cached tuple's IDENTITY, the same test
    ``_beam_deposition._csda_tables`` uses, so a rebuilt table (a different
    ``n_max``) rebuilds these too.
    """
    global _HE_BEAM_EXC_SEQ
    table = _he_beam_excitation_table(n_max)
    if _HE_BEAM_EXC_SEQ is None or _HE_BEAM_EXC_SEQ[0] is not table:
        _HE_BEAM_EXC_SEQ = (
            table, table[1].tolist(), table[2].tolist(), table[3].tolist()
        )
    return _HE_BEAM_EXC_SEQ[1:]


def He_beam_excitation_channel_lkup(E_eV, n_max=20):
    """Interpolated ``He_beam_excitation_channel``: same contract, ~100x faster.

    ``(0.0, 0.0)`` at or below the 20 eV table floor (below every singlet
    threshold); exact-function fallback above the 2000 eV table ceiling.
    """
    E_grid, sigma, sigma_E = _he_beam_excitation_seq(n_max)
    E = float(E_eV)
    if E <= E_grid[0]:
        return 0.0, 0.0
    if E >= E_grid[-1]:
        return He_beam_excitation_channel(E, n_max=n_max)
    s = _interp_scalar_fused(E, E_grid, sigma)
    if s <= 0.0:
        return 0.0, 0.0
    return s, _interp_scalar_fused(E, E_grid, sigma_E) / s


def int_factor(I):
    return mp.fprod(
        [mp.power(I, 2), mp.sqrt(mp.fdiv(8, mp.fmul(M_e_eV, mp.pi))), c_cgs]
    )


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


# H + H+ charge exchange, p + H(1s) -> H(1s) + p.
# Source of record: IAEA HYDHEL, D. Reiter, FZJ, version 2020-01-13,
# https://www.eirene.de/Documentation/hydhel.pdf, Sec. 3 H.3, Reaction 3.1.8,
# printed p.165 (PDF p.176). Stored as A_R318[e][t], i.e. one list per source
# E-Index column, indexed by T-Index 0..8.
#
# CITE HYDHEL, NOT THE SPRINGER BOOK. These are HYDHEL's REPLACEMENT fit, not
# the 1987 Springer book coefficients: HYDHEL replaced the original for
# consistency with the cross section (its own note on that page reads
# "original fit from Springer book replaced by this one, which has better
# consistency with cross-section, hence: better energy conservation"). A future
# check against the Springer coefficients would report spurious mismatches, so
# every re-verification must cite HYDHEL.
#
# CORRUPTION AND FIX. Row e=1 carried a single INSERTED DUPLICATE of
# 9.536923957409e-03 (present once in the source, at T-Index 2), giving that
# row 10 entries where every other row here and every A_R531 row has 9;
# heavy_reaction iterates range(len(A[i])), so the ragged row silently
# contributed an extra polynomial term to _cx_H rather than failing. The
# duplicate was deleted 2026-08-27 and the table is now 9x9 = 81. All 81
# coefficients were verified digit for digit against the re-fetched PDF in the
# same change ([sbq:L295]); the source's max/mean relative fit errors are
# 1.1026 % / 0.3105 %.
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
        Gas species. "He" for helium (A_R531 table) is the only accepted value;
        "H" (A_R318 table) is quarantined and raises.

    Returns
    -------
    float or array
        Charge-exchange rate coefficient [cm³/s].

    Raises
    ------
    ValueError
        For ``gas_type="H"`` -- a quarantined hydrogen entry point, and the one
        that reads the A_R318 table directly. See the hydrogen quarantine note
        at the top of this module.
    """
    if gas_type == "He":
        table = _cx_He
    elif gas_type == "H":
        # RETAINED, not removed: _cx_H above is still built at import, so a
        # validated re-opening deletes this raise and restores `table = _cx_H`.
        raise ValueError(
            "gas_type='H' is not available: the hydrogen arms of cablp.atomic "
            "are QUARANTINED, ruled 2026-08-27, and this is the entry point "
            "that reads A_R318 directly. That table's duplicated coefficient "
            "was repaired the same day and all 81 entries digit-proofed "
            "against IAEA HYDHEL 3.1.8, so the corruption ground no longer "
            "applies; the quarantine stands on the untested domain alone -- "
            "no hydrogen arm here has a solver-path consumer or any gate "
            "coverage. The solver is helium-only (D3, 2026-08-21). Accepted: "
            "'He' (A_R531, unaffected)."
        )
    else:
        raise ValueError(f"unsupported gas_type {gas_type!r}; expected 'He'")
    return np.interp(T, temps, table)


# ── Phelps He+/He ion-neutral scattering cross sections (audit A7, R4.3) ──────
#
# REQUIRED CITATION (LXCat terms of use -- cite ALL of the following):
#   (1) Site + retrieval date (identifies the data via the LXCat Time Machine):
#         Phelps database, www.lxcat.net, retrieved on July 25, 2026.
#   (2) Contributor "How to reference" (Phelps database header):
#         http://jilawww.colorado.edu/~avp/
#   (3) Data-group provenance: He+ in He, A.V. Phelps and collaborators
#         (private communication; no journal reference given in the dataset).
#   (4) LXCat platform: Pitchford et al., Plasma Process. Polym. 14, 1600098 (2017).
# The retrieval date above is present in the header of the archived raw download
# ``atomic/data/he_ion_neutral_phelps_lxcat.txt`` and MUST be reproduced in any
# publication. The database gives two analytic
# center-of-mass differential-scatter components for He+ in He (E = relative
# collision energy in eV, sigma in m^2 in the source; converted to cm^2 here):
#
#   backscatter (180 deg = symmetric charge exchange):
#       Qb = 1e-19 * (E/1000)^-0.15 * (1+E/1000)^-0.25 * (1+5/E)^-0.15   [m^2]
#   isotropic (polarization elastic):
#       Qi = 7.63e-20 * E^-0.5                                          [m^2]
#
# Moment mapping used by the R4.3 reduced ion-neutral collision operator. Both
# rows are the transport (momentum-transfer) moment int (1 - cos th) dsigma of
# the two components above -- nothing is fitted, and no constant enters that is
# not on this page:
#   sigma_cx = Qb                     (charge-exchange cross section)
#   sigma_mt = Qi + 2*Qb              (momentum transfer: isotropic contributes
#                                      int(1-cos th)=Qi, backscatter contributes
#                                      (1-cos 180)=2 per unit -> 2*Qb)

_M2_TO_CM2 = 1.0e4
_PHELPS_MU_G = 0.5 * m_He_cgs  # equal-mass He+/He reduced mass [g]


def phelps_he_backscatter_cm2(E_eV):
    """Phelps He+/He backscatter (charge-exchange) cross section [cm^2].

    ``E_eV`` is the relative collision energy in eV. Analytic form (exact to the
    archived LXCat table, verified in ``verify_sim1d_r4_collision.py``); the
    ``1+5/E`` factor makes it approach a finite ~2.21e-15 cm^2 floor as E->0, so
    it is well defined to thermal energies (no low-energy clamp).
    """
    E = np.asarray(E_eV, dtype=float)
    Qb_m2 = (
        1.0e-19
        * (E / 1000.0) ** -0.15
        * (1.0 + E / 1000.0) ** -0.25
        * (1.0 + 5.0 / E) ** -0.15
    )
    return Qb_m2 * _M2_TO_CM2


def phelps_he_isotropic_cm2(E_eV):
    """Phelps He+/He isotropic (polarization-elastic) cross section [cm^2].

    ``E_eV`` is the relative collision energy in eV. ``Qi = 7.63e-20 * E^-0.5``
    m^2 -- the ``sigma ~ 1/v`` polarization law, so ``sigma*v`` is
    velocity-independent and its Maxwellian rate coefficient is constant.
    """
    E = np.asarray(E_eV, dtype=float)
    Qi_m2 = 7.63e-20 * E ** -0.5
    return Qi_m2 * _M2_TO_CM2


def _maxwellian_rate_cm3_s(sigma_cm2_of_eV, T_eV, mu_g, n_energy=2000, e_max_kT=60.0):
    """Return ``<sigma v_rel>`` [cm^3/s] over a relative-velocity Maxwellian.

    ``k(T) = sqrt(8/(pi mu)) (kT)^-3/2 \\int_0^inf sigma(E) E exp(-E/kT) dE`` with
    ``E`` the relative collision energy and ``mu`` the reduced mass. ``sigma`` is
    supplied as a function of ``E`` in eV returning cm^2; the integral is done in
    erg so the returned rate is in cm^3/s.

    Quadrature is on ``x = sqrt(E)``, not on ``E``. Both Phelps cross sections
    carry a ``sigma ~ E^-0.5`` factor, so on a linear ``E`` grid the integrand
    behaves as ``sqrt(E)`` near the origin -- its derivative is unbounded
    there, and the trapezoid rule converges at a rate set by that singularity
    rather than by its usual second order. Substituting ``E = x^2``
    (``dE = 2x dx``) cancels the ``E^-0.5`` exactly and leaves a smooth
    integrand, which is what the trapezoid rule is entitled to assume.

    Measured on the isotropic channel, whose Maxwellian average has a closed
    form (``sigma_iso*v`` is velocity-independent, so the rate is the same
    constant at every temperature): the linear-``E`` grid returned
    ``-1.21e-3`` relative to that closed form at every temperature sampled,
    and this form returns ``+2.9e-7``. Same node count, same cost.
    """
    T_eV = float(T_eV)
    x = np.linspace(0.0, np.sqrt(e_max_kT * T_eV), int(n_energy))
    E_eV = np.maximum(x * x, 1.0e-300)
    sigma = np.asarray(sigma_cm2_of_eV(E_eV), dtype=float)
    E_erg = E_eV * ev_to_erg
    kT = T_eV * ev_to_erg
    prefactor = np.sqrt(8.0 / (np.pi * mu_g)) * kT ** -1.5
    integrand = sigma * E_erg * np.exp(-E_erg / kT)
    # dE = 2x dx, carried in erg to keep the integral's units unchanged.
    return prefactor * np.trapezoid(integrand * 2.0 * x * ev_to_erg, x)


# Pre-computed rate-coefficient tables vs the effective (relative-velocity)
# temperature T_eff, built once at import over 1e-3..1e4 eV (300 K ~ 0.0259 eV
# is well inside the grid). Interpolated by ``phelps_*_rate_cm3_s`` below.
_phelps_Teff = np.logspace(-3, 4, 400)
_phelps_kb = np.array(
    [_maxwellian_rate_cm3_s(phelps_he_backscatter_cm2, T, _PHELPS_MU_G)
     for T in _phelps_Teff]
)
_phelps_kiso = np.array(
    [_maxwellian_rate_cm3_s(phelps_he_isotropic_cm2, T, _PHELPS_MU_G)
     for T in _phelps_Teff]
)


def phelps_cx_rate_cm3_s(T_eff, gas_type="He"):
    """He+/He charge-exchange rate coefficient ``<Qb v_rel>`` [cm^3/s].

    ``T_eff`` is the effective relative-velocity temperature ``(Ti+Tn)/2`` in eV.
    Grows ~sqrt(T_eff) at low temperature (no flat clamp), unlike the IAEA
    ``charge_ex_react`` table whose 0.1 eV floor holds the rate constant below it.
    """
    if gas_type != "He":
        raise ValueError(
            f"Phelps ion-neutral cross sections are He-only (got {gas_type!r})"
        )
    return np.interp(T_eff, _phelps_Teff, _phelps_kb)


def phelps_iso_rate_cm3_s(T_eff, gas_type="He"):
    """He+/He isotropic-elastic rate coefficient ``<Qi v_rel>`` [cm^3/s].

    ``T_eff`` in eV. Velocity-independent (``Qi ~ 1/v``), so this is essentially
    constant (~7.49e-10 cm^3/s), matching the classic Langevin capture rate.
    """
    if gas_type != "He":
        raise ValueError(
            f"Phelps ion-neutral cross sections are He-only (got {gas_type!r})"
        )
    return np.interp(T_eff, _phelps_Teff, _phelps_kiso)


def phelps_momentum_transfer_rate_cm3_s(T_eff, gas_type="He"):
    """He+/He total momentum-transfer rate ``<sigma_mt v_rel>`` scaled by the
    equal-mass reduced-mass factor: ``k_b + 0.5*k_iso`` [cm^3/s].

    This is ``nu_mt / nn`` for the R4.3 reduced drag operator
    ``dM/dt = -m n nu_mt (u - u_n)``. The ``0.5`` on ``k_iso`` and the implicit
    ``2*Qb -> k_b`` are the equal-mass ``mu/m_i = 1/2`` lab-frame factors.
    """
    return (
        phelps_cx_rate_cm3_s(T_eff, gas_type)
        + 0.5 * phelps_iso_rate_cm3_s(T_eff, gas_type)
    )


# ── He ELECTRON-neutral momentum transfer (the QL-relaxation damping side) ───
# A two-node boxed table of the He e-n momentum-transfer cross section, and the
# Maxwellian-scale rate coefficient K_m(Te) built from it. Distinct from the
# `phelps_*` block above in both species and role: those are He+/He ION-neutral
# channels feeding the R4.3 drag operator, this is the ELECTRON-neutral channel
# that collisionally damps a Langmuir wave, and it is read only by the
# `ql_relaxation` beam-anomalous closure's onset gate.
#
# The table is two nodes and nothing is smuggled in between them.
#
# Both nodes are DERIVED from the same three published He elastic
# momentum-transfer sets -- Biagi, IST-Lisbon and Morgan (LXCat, retrieved
# 2026-08-13) -- read at the node energy by linear interpolation, which is the
# convention those tables state for themselves. The NODE is the three-set
# arithmetic centre and the BRACKET is [min, max] over the three sets, so the
# bracket is the measured set-to-set disagreement rather than an assumed bar.
# The endpoints are rounded OUTWARD (low end down, high end up) so that every
# set stays inside the bracket the constant declares: rounding the 5 eV minimum
# 6.209760e-16 to 6.210e-16 would have put Biagi 0.004 % outside it.
# Both endpoints are published here as data so a reported result can quote them
# instead of re-deriving them; `scripts/atomic/kmpull_threeset_check.py` re-runs the
# derivation against the pull of record.
HE_EN_MT_NODE_EV = (5.0, 25.0)
HE_EN_MT_SIGMA_CM2 = (6.280e-16, 1.992e-16)
HE_EN_MT_SIGMA_BRACKET_CM2 = ((6.209e-16, 6.320e-16), (1.950e-16, 2.067e-16))

_HE_EN_MT_LOG_E = np.log(np.asarray(HE_EN_MT_NODE_EV, dtype=float))
_HE_EN_MT_LOG_SIGMA = np.log(np.asarray(HE_EN_MT_SIGMA_CM2, dtype=float))


def he_electron_momentum_transfer_cm2(E_eV):
    """He electron-neutral momentum-transfer cross section [cm^2] at E_eV.

    Log-log interpolation between the two tabulated nodes, CLAMPED flat outside
    them: with two nodes the slope is a chord and not a measurement, so
    extrapolating it past either end would manufacture structure the table does
    not contain. Inside the span the chord is a power law
    ``sigma ~ E^-0.652``.

    Accepts a scalar or an array and returns the same shape.
    """
    E = np.asarray(E_eV, dtype=float)
    logE = np.log(np.maximum(E, 1.0e-300))
    out = np.exp(
        np.interp(logE, _HE_EN_MT_LOG_E, _HE_EN_MT_LOG_SIGMA)
    )
    return float(out) if np.isscalar(E_eV) or out.ndim == 0 else out


def he_electron_momentum_transfer_rate_cm3_s(Te_eV):
    """He e-n momentum-transfer rate coefficient ``K_m(Te)`` [cm^3/s].

    ``nu_en = nn * K_m(Te)`` is the electron-neutral momentum-transfer
    collision frequency [1/s]; half of it is the AMPLITUDE damping rate of a
    Langmuir wave, which is the quantity the QL-relaxation onset gate weighs the
    beam-plasma growth rate against.

    Formed as ``sigma_m(<E>) * <v>`` with both factors the Maxwellian means at
    ``Te`` -- mean energy ``<E> = 1.5 Te``, mean speed
    ``<v> = sqrt(8 kTe / (pi m_e))``. This is the ``<sigma v> ~ sigma(<E>)<v>``
    estimate, not a quadrature: the underlying table is two nodes wide, so a
    Maxwellian average over it would report a precision the data does not have.
    The order-of-magnitude standing is stated with the values in the provenance
    note and travels with every number this gate produces.

    Accepts a scalar or an array and returns the same shape.
    """
    Te = np.asarray(Te_eV, dtype=float)
    Te_pos = np.maximum(Te, 1.0e-300)
    v_mean = np.sqrt(8.0 * Te_pos * ev_to_erg / (np.pi * m_e_cgs))
    out = he_electron_momentum_transfer_cm2(1.5 * Te_pos) * v_mean
    return float(out) if np.isscalar(Te_eV) or np.ndim(out) == 0 else out
