"""
cathode_solver.py
-----------------
Steady-state sheath potential and current solver for a thermionic cathode
system, intended to be called at each step of an outer Runge-Kutta integrator.

Units
-----
- Geometry        : CGS (cm)
- Mass            : CGS (g)
- Temperature     : T_e in eV, T_s in K
- Current         : Amperes [A]
- Resistance      : Ohms [Ω]
- Potential       : Volts [V]
- Power           : Watts [W]

Scaled (dimensionless) quantities used internally:
  psi = phi / T_e        (potential scaled by electron temperature in V)
  J   = I * R_p / T_e   (current scaled by plasma resistance and T_e)
  delta = kB_SI * T_s / (e_SI * T_e)   (temperature ratio T_s / T_e in same units)

References
----------
See module docstring of calling RK solver for physics references.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.optimize import brentq

from cablp.funcs._cross import (
    He_EII_cross_lkup,
    H_EII_cross_lkup,
    He_beam_excitation_channel,
)
from cablp.vars._coeff import b_11s_21p
from cablp.vars._cons import E_21p as _E_21p_eV, Ry_eV as _Ry_eV, atm_cross_cgs as _atm_cross_cgs

# ---------------------------------------------------------------------------
# Universal physical constants
# ---------------------------------------------------------------------------

_e_SI: float = 1.602176634e-19  # Electron charge [C] = [J/eV]
_kB_SI: float = 1.380649e-23  # Boltzmann constant [J/K]
_me_cgs: float = 9.1093837015e-28  # Electron mass [g]
_mp_cgs: float = 1.67262192369e-24  # Proton mass [g]
_pemr: float = _mp_cgs / _me_cgs  # Proton-to-electron mass ratio ≈ 1836.15
_erg_per_eV: float = _e_SI * 1.0e7  # eV → erg conversion


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DeviceConfig:
    """Static device configuration; does not change between RK steps.

    Parameters
    ----------
    A_c     : Cathode area [cm²]
    mu      : Ion mass / proton mass [dimensionless]
    V_bank  : Power supply (cathode bank) voltage [V]
    T_s     : Cathode surface temperature [K]
    phi_wf  : Work function [eV]; default 3.0 (LaB6)
    C_R     : Richardson constant [A cm⁻² K⁻²]; default 29 (LaB6)
    R_comp  : Compliance resistor [Ω]; default 0.004
    eta     : Anode area / cathode area; default 0.5
    Twin    : Second-cathode flag; when True, solve_beam_system also solves the twin cathode at the far end of the plasma arrays
    L_cath  : Cathode-to-anode distance [cm]; default 50
    R_cath  : Cathode radius [cm]; default 19
    """

    A_c: float
    mu: float
    V_bank: float
    T_s: float
    phi_wf: float = 3.0
    C_R: float = 29.0
    R_comp: float = 0.004
    eta: float = 0.358
    Twin: bool = False
    L_cath: float = 50.0
    R_cath: float = 18.0
    # Physical ceiling on the sheath drop phi_c [V]; None = no ceiling (the
    # historical behaviour, where the root bracket scales with V_bank). Set by
    # the inductive-circuit wrapper: the backward-Euler effective V_bank can be
    # orders of magnitude above the physical EMF, and letting the root bracket
    # scale with it exposes pathological far roots (phi_c ~ V_eff at ~zero
    # current) that flap the solve and can run the coupled circuit away.
    phi_sheath_max: float | None = None
    # Annular emission profile (empty tuples = the historical uniform disc).
    # A real cathode's surface temperature falls off radially, so its emission
    # ceiling is a soft ramp, not a razor wall: as the sheath pulls harder the
    # virtual-cathode clamp releases progressively cooler annuli. Each annulus
    # carries its own Richardson emission (T_k over area_k), its own share of
    # the ion current (its overlap with the plasma footprint), and its own
    # space-charge clamp; all share one equipotential psi_c_plus.
    emission_Ts_K: tuple = ()
    emission_area_cm2: tuple = ()
    emission_plasma_frac: tuple = ()

    # Derived constants computed once at construction
    # (stored as slots; frozen prevents reassignment)
    Lambda: float = field(init=False)
    I_eth: float = field(init=False)

    def __post_init__(self) -> None:
        # Lambda = sheath floating-potential parameter
        # Lambda = -ln( sqrt(2*pi / (mu * _pemr)) )
        lam = -math.log(math.sqrt(2.0 * math.pi / (self.mu * _pemr)))
        object.__setattr__(self, "Lambda", lam)

        # I_eth = thermionic emission current [A] (static; depends only on T_s)
        if self.emission_Ts_K:
            i_eth = sum(
                area
                * self.C_R
                * T_k**2
                * math.exp(-_e_SI * self.phi_wf / (_kB_SI * T_k))
                for T_k, area in zip(self.emission_Ts_K, self.emission_area_cm2)
            )
        else:
            i_eth = (
                self.A_c
                * self.C_R
                * self.T_s**2
                * math.exp(-_e_SI * self.phi_wf / (_kB_SI * self.T_s))
            )
        object.__setattr__(self, "I_eth", i_eth)


@dataclass(slots=True)
class PlasmaState:
    """Plasma state provided by the Runge-Kutta solver at each step.

    Parameters
    ----------
    T_e    : Electron temperature [eV]
    n_e    : Electron density [cm⁻³]
    n_n    : Neutral density [cm⁻³]; used for beam MFP (default 0)
    sigma_b: Beam ionization cross-section [cm²]; used for beam MFP (default 0)
    """

    T_e: float
    n_e: float
    n_n: float = 0.0
    sigma_b: float = 0.0


@dataclass(slots=True)
class SolverResult:
    """All quantities returned by the sheath solver.

    Potentials [V]
    --------------
    phi_c_plus  : Classical (positive) part of cathode sheath drop
    phi_c_minus : Inverted (virtual-cathode) part of cathode sheath drop
    phi_c       : Total cathode sheath drop  = phi_c_plus - phi_c_minus
    phi_a       : Anode sheath potential (may be negative above plasma potential)
    V_p         : Plasma ohmic voltage drop
    V_b         : Bias voltage across anode and cathode

    Resistance [Ω]
    --------------
    R_p         : Parallel plasma resistance

    Currents [A]
    ------------
    I_i         : Ion saturation current
    I_e         : Electron saturation current
    I_eth       : Total thermionic emission current (config constant)
    I_eth_star  : Allowed thermionic current (clamped by virtual-cathode limit)
    I_tot       : Net circuit current

    Power [W]
    ---------
    P_wall      : Total power demanded from supply  = I_tot * V_bank
    P_load      : Power delivered to plasma load    = I_tot * V_b
    P_comp      : Compliance resistor dissipation   = I_tot² * R_comp
    P_prim      : Primary-electron power into plasma = I_eth_star * phi_c
    P_ohmic     : Plasma ohmic heating              = I_tot * V_p
    P_loss      : Sheath power loss = P_cathode_e + P_cathode_i_pl + P_anode_e + P_anode_i_pl

    Metadata
    --------
    regime      : 'classical' or 'virtual_cathode'
    """

    # Potentials [V]
    phi_c_plus: float
    phi_c_minus: float
    phi_c: float
    phi_a: float
    V_p: float
    V_b: float
    # Resistance [Ω]
    R_p: float
    # Currents [A]
    I_i: float
    I_i_a: float
    I_e: float
    I_eth: float
    I_eth_star: float
    I_tot: float
    # Power [W]
    P_wall: float
    P_load: float
    P_comp: float
    P_prim: float
    P_ohmic: float
    P_cathode_e: float
    P_cathode_i: float
    P_cathode_i_pl: float
    P_anode_e: float
    P_anode_i: float
    P_anode_i_pl: float
    P_net: float
    P_net2: float
    P_loss: float
    # Regime
    regime: Literal["classical", "virtual_cathode"] = "classical"
    long_mfp: bool = False
    beam_bypass_fraction: float = 0.0
    # Beam mean free path [cm]; 0.0 if beam parameters not provided
    l_b: float = 0.0
    # R3.2 (SIM1D_MODEL_AUDIT_PLAN A16) one-control-surface split. Each electrode
    # power splits into a PLASMA-THERMAL part (Te/2 per ion, 2Te per electron --
    # sourced from the plasma thermal store) and a SHEATH-FALL phi part (sourced
    # from the sheath field / circuit, deposited on the electrode, never through
    # the plasma thermal store). ``*_thermal + *_phi == P_*_e/_i`` by construction.
    # These are the values the repaired fluid boundary reads so fluid == circuit.
    # P_net/P_net2/P_loss above are the DEPRECATED unclosed scalars, kept only so
    # the R1-R4 golden stays bit-exact; the useful closed audit is below.
    P_cathode_e_thermal: float = 0.0
    P_cathode_e_phi: float = 0.0
    P_cathode_i_thermal: float = 0.0
    P_cathode_i_phi: float = 0.0
    P_anode_e_thermal: float = 0.0
    P_anode_e_phi: float = 0.0
    P_anode_i_thermal: float = 0.0
    P_anode_i_phi: float = 0.0
    # Closed surface-resolved audit [W] (Tom 2026-07-24, replaces P_net/P_net2):
    # net power heating the plasma vs power onto each electrode surface.
    P_plasma_thermal_loss: float = 0.0   # total plasma-thermal loss to electrodes
    P_into_plasma: float = 0.0           # P_prim + P_ohmic - plasma-thermal loss
    P_cathode_surface: float = 0.0       # plasma power onto the cathode (thermal+phi)
    P_anode_surface: float = 0.0         # plasma power onto the anode (thermal+phi)
    # Measurement-plane bookkeeping aliases (Tom 2026-07-24). The Poulos names
    # I_tot / V_b are the MODEL LOAD quantities and are kept as-is; these alias to
    # the three-plane convention so a future effective-load change diverges
    # predictably without renaming. All divergences are identically zero until
    # THESIS_NOTES items 24 (parallel branch) / 25 (series impedance) land:
    #   V_dis = V_b + V_series     (measured terminal/discharge voltage)
    #   I_bank = I_plasma + I_parallel   (measured bank/terminal current, I_dis)
    #   I_plasma == I_tot now      (current conducted through the plasma load)
    # so today V_dis == V_b, I_bank == I_plasma == I_tot, and every V*I product
    # coincides: P_load = V_b*I_tot = V_dis*I_bank. The closure is referenced to
    # P_load (power across the load), NOT the terminal product.
    V_series: float = 0.0      # hidden series-impedance drop [V] (item 25); 0 now
    I_parallel: float = 0.0    # stray/parallel branch current [A] (item 24); 0 now
    V_dis: float = 0.0         # measured terminal voltage [V] = V_b + V_series
    I_plasma: float = 0.0      # plasma-conducted current [A] = I_tot now
    I_bank: float = 0.0        # measured bank current [A] = I_plasma + I_parallel
    # Load-power closure diagnostic (Tom 2026-07-24). The circuit does I_tot*V_b
    # of NET field work across the load; by the potential ladder V_b = phi_c + V_p
    # - phi_a and Kirchhoff (the same net loop current threads each region), that
    # is the per-region field work, and each region decomposes per species with
    # the current DIRECTIONS respected -- at the cathode the emission and ion
    # collection deliver field energy while the RETURNING plasma-electron current
    # recovers it: I_tot = I_eth_star + I_i - I_e_ret. The sheath phi work is drawn
    # from the CIRCUIT (it circulates: the cathode sheath accelerates carriers, the
    # anode recovers), NOT from the plasma thermal store -- which is exactly why
    # R3.2 routes phi to the electrode book and 2Te/Te/2 to the plasma book. The
    # beam plasma deposition P_prim and the bulk-current ohmic are a SEPARATE
    # plasma-heating book, not the circuit field work.
    I_e_ret: float = 0.0           # returning plasma-electron current to cathode [A]
    P_load_ledger: float = 0.0     # per-region net-current field work [W]
    P_load_residual: float = 0.0   # P_load - P_load_ledger [W] (~0 by closure)
    I_cathode_kirchhoff_residual: float = 0.0  # (I_eth_star+I_i-I_e_ret) - I_tot [A]


@dataclass(slots=True)
class BeamResult:
    """Beam quantities returned by solve_beam_system for one or two cathodes.

    Arrays have shape (cells,); non-source cells are zero.

    Fields
    ------
    result          : SolverResult for the primary cathode (index 0)
    result_twin     : SolverResult for the twin cathode (index -1), or None
    v_beam          : beam electron velocity [cm/s]
    n_beam          : beam electron density [cm⁻³]
    beam_cross      : EII cross section at beam energy [cm²]
    beam_exc_cross  : neutral-excitation cross section at beam energy [cm²]
                      (zero unless b_beam_excitation is set)
    beam_atten_cross: total inelastic attenuation cross section
                      = beam_cross + beam_exc_cross [cm²]; feed this back as
                      ``beam_cross_prev`` so the sheath solve's MFP sees both
                      channels
    n_beam_ion      : n_beam * beam_cross * v_beam  [s⁻¹]
    A_ion_beam      : n_beam_ion * nn  [cm⁻³ s⁻¹]
    l_b             : beam mean free path per cathode cell [cm]; 0 elsewhere
    p_beam          : neutral ionization probability = l_b * beam_cross * nn [dimensionless]
    l_b_profile     : per-cell MFP for the primary beam [cm]; zeros if beam_cross[0]==0
    l_b_profile_twin: per-cell MFP for the twin beam [cm]; zeros if no twin or beam_cross[-1]==0
    x0_next         : warm-start hint for the next primary solve [V]
    x0_twin_next    : warm-start hint for the next twin solve [V], or None
    beam_exc_energy_eV: radiated energy per excitation event [eV] at the
                      cathode cells (constant threshold under "2p_scalar",
                      energy-weighted manifold mean under "manifold"), or
                      None from a builder that predates the manifold model
                      (consumers fall back to ``beam_excitation_energy_eV``)
    """

    result: SolverResult
    result_twin: SolverResult | None
    v_beam: np.ndarray
    n_beam: np.ndarray
    beam_cross: np.ndarray
    beam_exc_cross: np.ndarray
    beam_atten_cross: np.ndarray
    n_beam_ion: np.ndarray
    A_ion_beam: np.ndarray
    l_b: np.ndarray
    p_beam: np.ndarray
    l_b_profile: np.ndarray
    l_b_profile_twin: np.ndarray
    x0_next: float
    x0_twin_next: float | None
    beam_exc_energy_eV: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _exp_clamped(x: float) -> float:
    """``math.exp`` with the argument capped at 700 (just under overflow).

    Bit-exact for any argument below the cap; above it, returns a finite
    ~1e304 so extreme root-bracket probes (seen under transient effective
    circuit EMFs) yield a huge finite residual the solver can reject,
    instead of an OverflowError that kills the whole run.
    """
    return math.exp(min(x, 700.0))


def _c_log_ei(T_e: float, n_e: float) -> float:
    """Electron-ion Coulomb logarithm (NRL 2019, eqs. 2-3/2-4)."""
    if T_e > 10.0:
        return 24.0 - math.log(math.sqrt(n_e) / T_e)
    return 23.0 - math.log(math.sqrt(n_e) * T_e**-1.5)


def _psi_c_minus(psi_c_plus: float, J_i: float, J_eth: float, mu: float, delta: float) -> float:
    """Virtual-cathode sheath drop (0 in classical regime)."""
    J_crit = _j_eth_crit(psi_c_plus, J_i, mu)
    if J_eth <= 0.0 or J_eth <= J_crit:
        return 0.0
    return delta * math.log(J_eth / J_crit)


def _annular_emission_state(
    psi_c_plus: float,
    J_i: float,
    mu: float,
    J_eth_k: tuple,
    delta_k: tuple,
    ion_frac_k: tuple,
) -> tuple[float, float, bool]:
    """Return ``(J_star_total, psi_minus_eff, any_clamped)`` for the annuli.

    Each annulus sees its own space-charge limit from its share of the ion
    current (``_j_eth_crit`` is linear in ``J_i``); an annulus outside the
    plasma footprint (``ion_frac 0``) is fully choked. All annuli share the
    equipotential ``psi_c_plus``; the loop equation's effective
    virtual-cathode drop is the emission-weighted mean of the local barriers,
    which reduces exactly to the uniform expression for a single annulus.
    """
    J_star_total = 0.0
    weighted_pm = 0.0
    any_clamped = False
    for J_eth_a, delta_a, frac_a in zip(J_eth_k, delta_k, ion_frac_k):
        if J_eth_a <= 0.0:
            continue
        J_crit_a = _j_eth_crit(psi_c_plus, J_i * frac_a, mu) if frac_a > 0.0 else 0.0
        if J_eth_a <= J_crit_a:
            J_star_total += J_eth_a
            continue
        any_clamped = True
        if J_crit_a <= 0.0:
            continue  # fully choked annulus emits nothing
        pm_a = delta_a * math.log(J_eth_a / J_crit_a)
        J_star_total += J_crit_a
        weighted_pm += J_crit_a * pm_a
    psi_minus_eff = weighted_pm / J_star_total if J_star_total > 0.0 else 0.0
    return J_star_total, psi_minus_eff, any_clamped


def _residual_annular(
    psi_c_plus: float,
    J_i: float,
    Lambda: float,
    gamma: float,
    eta: float,
    psi_bank: float,
    mu: float,
    J_i_a: float,
    J_eth_k: tuple,
    delta_k: tuple,
    ion_frac_k: tuple,
    beam_bypass_fraction: float = 0.0,
    tau_a: float = 1.0,
) -> float:
    """Annular-emission variant of ``_residual`` (same loop equation)."""
    J_star, psi_minus, _ = _annular_emission_state(
        psi_c_plus, J_i, mu, J_eth_k, delta_k, ion_frac_k
    )
    J_tot = J_i * (1.0 - _exp_clamped(Lambda - psi_c_plus)) + J_star
    J_anode = J_tot - eta * beam_bypass_fraction * J_star
    anode_arg = max(1.0 + J_anode / J_i_a, 1e-300)
    return (
        psi_c_plus
        - psi_minus
        + (1.0 + gamma) * J_tot
        - tau_a * Lambda
        + tau_a * math.log(anode_arg)
        - psi_bank
    )


def _compute_l_b(phi_c: float, T_e: float, n_e: float, n_n: float, sigma_b: float) -> float:
    """Beam mean free path [cm] for primary electrons accelerated through phi_c [V]."""
    if phi_c <= 0.0:
        return 0.0
    v_beam = math.sqrt(2.0 * phi_c * _erg_per_eV / _me_cgs)
    tau_ei = 3.44e5 * T_e**1.5 / n_e / _c_log_ei(T_e, n_e)
    l_bi = v_beam * tau_ei
    if sigma_b > 0.0 and n_n > 0.0:
        l_bn = 1.0 / (sigma_b * n_n)
        return 1.0 / (1.0 / l_bi + 1.0 / l_bn)
    return l_bi


def _compute_beam_bypass_fraction(l_b: float, L_cath: float) -> float:
    """Fraction of anode-directed thermionic beam that survives to the anode."""
    if l_b <= 0.0 or L_cath <= 0.0:
        return 0.0
    return math.exp(-L_cath / l_b)


_ATM_CROSS_CGS = float(_atm_cross_cgs)
_RY_EV = float(_Ry_eV)
_E21P_EV = float(_E_21p_eV)


def _he_2p_excitation_cross_cm2(eps: float) -> float:
    """Float port of ``He_EIE_cross_DA(eps, b_11s_21p)`` [cm^2].

    The mpmath original costs ~20 us per scalar call and sits in the per-step
    cathode solve; this is the same dipole-allowed formula in plain floats
    (agreement asserted to 1e-12 in the smoke test).
    """
    a = b_11s_21p
    factor1 = _ATM_CROSS_CGS * _RY_EV / (eps * _E21P_EV)
    factor2 = a[0] * math.log(eps) + sum(
        a[i] * eps ** (1 - i) for i in range(1, 5)
    )
    factor3 = (eps + 1.0) / (eps + a[5])
    return factor1 * factor2 * factor3


def beam_excitation_cross(
    phi_c_eV: float,
    b_beam_excitation: float,
    gas_type: str,
    threshold_eV: float = 21.218,
) -> float:
    """Neutral-excitation cross section [cm^2] for the primary beam.

    The 1^1S -> 2^1P dipole-allowed cross section at the beam energy, scaled
    by ``b_beam_excitation``. 2^1P is the dominant singlet channel at beam
    energies (60-180 eV); the rest of the singlet manifold (2^1S, 3^1P, ...)
    adds roughly another 30-50%, which the scale factor is meant to absorb
    (``~1.4`` approximates the full manifold, ``1.0`` is 2^1P alone). Triplet
    (metastable) excitation is exchange-driven and collapses above ~50 eV, so
    it is deliberately not modeled. ``b_beam_excitation = 0`` (default)
    disables the channel entirely and reproduces the historical beam.
    """
    if b_beam_excitation == 0.0 or phi_c_eV <= threshold_eV:
        return 0.0
    if gas_type != "He":
        raise ValueError(
            "b_beam_excitation != 0 is only wired for gas_type 'He' "
            f"(got {gas_type!r}); the excitation cross section is helium's"
        )
    return float(b_beam_excitation) * _he_2p_excitation_cross_cm2(
        phi_c_eV / threshold_eV
    )


def beam_excitation_channel(
    phi_c_eV: float,
    b_beam_excitation: float,
    gas_type: str,
    model: str = "2p_scalar",
    threshold_eV: float = 21.218,
) -> tuple[float, float]:
    """Beam excitation channel: ``(sigma_cm2, E_rad_eV_per_event)``.

    ``model = "2p_scalar"`` (historical): sigma from
    ``beam_excitation_cross`` (b x the 2^1P cross section) and the constant
    ``threshold_eV`` radiated per event — the pre-manifold booking, byte-for-
    byte. ``model = "manifold"``: the measured Ralchenko singlet manifold sum
    (``He_beam_excitation_channel``: fitted n <= 4 levels + Eq. (5) tail)
    with the energy-weighted mean radiated energy; ``b_beam_excitation``
    survives as a pure sensitivity multiplier on the cross section (benchmark
    value 1.0), and ``threshold_eV`` is ignored — thresholds live in the
    manifold registry. He-only, like the scalar path.
    """
    if model == "2p_scalar":
        return (
            beam_excitation_cross(
                phi_c_eV, b_beam_excitation, gas_type, threshold_eV=threshold_eV
            ),
            threshold_eV,
        )
    if model == "manifold":
        if b_beam_excitation == 0.0:
            return 0.0, 0.0
        if gas_type != "He":
            raise ValueError(
                "b_beam_excitation != 0 is only wired for gas_type 'He' "
                f"(got {gas_type!r}); the excitation manifold is helium's"
            )
        sigma, E_rad = He_beam_excitation_channel(phi_c_eV)
        return float(b_beam_excitation) * sigma, E_rad
    raise ValueError(
        f"unknown beam_excitation_model {model!r}; "
        "expected '2p_scalar' or 'manifold'"
    )


def _j_eth_crit(psi: float, J_i: float, mu: float) -> float:
    """Scaled critical thermionic current J_eth_crit(psi_c_plus).

    J_eth_crit = J_i * sqrt(mu * _pemr) * (exp(-psi) + sqrt(1 + 2*psi) - 2)
                 / sqrt(2 * psi)

    Numerically safe near psi = 0 via a Taylor expansion:
        numerator  ~ (1/3) * psi^3   as psi → 0
        denominator ~ sqrt(2) * psi^(1/2)
        → J_eth_crit ~ J_i * sqrt(mu * _pemr) / (3 * sqrt(2)) * psi^(5/2)
    """
    prefactor = J_i * math.sqrt(mu * _pemr)
    if psi <= 0.0:
        return 0.0
    if psi < 1e-3:
        # Taylor expansion to avoid catastrophic cancellation.
        # Numerator = exp(-psi) + sqrt(1+2*psi) - 2
        #           = psi^3/3 - 7*psi^4/12 + 31*psi^5/60 - ...
        # Denominator = sqrt(2*psi)
        # → J_eth_crit / (J_i*sqrt(mu*pemr)) = psi^(5/2)*(1/3 - 7*psi/12 + ...) / sqrt(2)
        numer = psi**3 * (1.0 / 3.0 - 7.0 * psi / 12.0)
        return prefactor * numer / math.sqrt(2.0 * psi)
    return (
        prefactor
        * (math.exp(-psi) + math.sqrt(1.0 + 2.0 * psi) - 2.0)
        / math.sqrt(2.0 * psi)
    )


def _residual(
    psi_c_plus: float,
    J_i: float,
    J_eth: float,
    Lambda: float,
    gamma: float,
    eta: float,
    delta: float,
    psi_bank: float,
    mu: float,
    J_i_a: float,
    beam_bypass_fraction: float = 0.0,
    tau_a: float = 1.0,
) -> float:
    """Root equation for psi_c_plus (scaled positive cathode sheath drop).

    0 = psi_c_plus
        - psi_c_minus(psi_c_plus)
        + (1 + gamma) * J_tot(psi_c_plus)
        - Lambda
        + ln(1 + J_anode / (2 * eta * J_i))
        - psi_bank

    where
        J_tot      = J_i*(1 - exp(Lambda - psi_c_plus)) + J_eth_star
        J_eth_star = min(J_eth, J_eth_crit(psi_c_plus))
        psi_c_minus = delta * ln(J_eth / J_eth_star)  [virtual cathode]
                    = 0                                 [classical]
        J_anode    = J_tot - eta * f_bypass * J_eth_star
    """
    J_crit = _j_eth_crit(psi_c_plus, J_i, mu)

    if J_eth <= 0.0:
        # No thermionic emission → always classical
        J_star = 0.0
        psi_minus = 0.0
    elif J_eth <= J_crit:
        # Classical regime: thermionic current fully admitted
        J_star = J_eth
        psi_minus = 0.0
    else:
        # Virtual-cathode regime: current limited by space charge
        J_star = J_crit
        psi_minus = delta * math.log(J_eth / J_crit)

    J_tot = J_i * (1.0 - _exp_clamped(Lambda - psi_c_plus)) + J_star

    # A fraction of the thermionic beam can reach the anode without a
    # plasma collision/ionization event.
    J_anode = J_tot - eta * beam_bypass_fraction * J_star

    # Anode sheath argument; clamp to avoid log(≤0) in extreme bracketing
    anode_arg = max(1.0 + J_anode / J_i_a, 1e-300)

    # The anode sheath drop is a real voltage, phi_a = Te_anode * psi_a, but this
    # residual is nondimensionalized in units of the *cathode* Te, so it enters
    # scaled by tau_a = Te_anode / Te_cathode. tau_a = 1 (the default, and the
    # case where both electrodes see the same plasma) reduces this exactly to the
    # single-temperature form.
    return (
        psi_c_plus
        - psi_minus
        + (1.0 + gamma) * J_tot
        - tau_a * Lambda
        + tau_a * math.log(anode_arg)
        - psi_bank
    )


class ConvergenceError(RuntimeError):
    """Raised when the root-finding bracket cannot be established."""


def _find_bracket(
    f, a: float, b: float, max_doublings: int = 15, b_max: float | None = None
) -> tuple[float, float]:
    """Return (a, b) such that f(a) and f(b) have opposite signs.

    Doubles `b` up to `max_doublings` times if the initial bracket does not
    contain a sign change. ``b_max`` (when given) caps the expansion so the
    search cannot wander past a physical ceiling.
    """
    if b_max is not None:
        b = min(b, b_max)
    fa = f(a)
    fb = f(b)
    if fa * fb < 0.0:
        return a, b
    for _ in range(max_doublings):
        if b_max is not None and b >= b_max:
            break
        b = b * 2.0 if b_max is None else min(b * 2.0, b_max)
        fb = f(b)
        if fa * fb < 0.0:
            return a, b
    raise ConvergenceError(
        f"Could not bracket root: f({a:.3g})={fa:.3g}, f({b:.3g})={fb:.3g}"
    )


def _P_ion(phi: float, T_e: float, I_i: float, pl: bool = False) -> float:
    """
    Ion power delivered to an electrode [W].

    For a Bohm-sheath ion current I_i, ions arrive with kinetic energy T_e/2
    plus the sheath acceleration energy phi (in eV, numerically equal to V).
    When pl=True the sheath drop is excluded (plasma-side boundary condition).

    Parameters
    ----------
    phi : float
        Electrode sheath potential [V].
    T_e : float
        Electron temperature [eV].
    I_i : float
        Ion saturation current to the electrode [A].
    pl : bool
        If True, return only the thermal contribution I_i * T_e / 2 (no sheath
        acceleration term); used for the plasma-side power balance.

    Returns
    -------
    float
        Ion power to the electrode [W].
    """
    if pl:
        return I_i * T_e / 2
    else:
        return I_i * (T_e / 2 + phi)


def _P_elec(phi: float, T_e: float, I_i: float, Lambda: float) -> float:
    """
    Electron power delivered to an electrode [W].

    The returning electron current is the saturation current I_i*exp(Lambda)
    reduced by the repelling sheath factor exp(-phi/T_e). Each electron carries
    energy 2*T_e (thermal) plus the sheath potential phi.

    Parameters
    ----------
    phi : float
        Electrode sheath potential [V] (positive repels electrons).
    T_e : float
        Electron temperature [eV].
    I_i : float
        Ion saturation current [A] (electron saturation = I_i * exp(Lambda)).
    Lambda : float
        Floating-potential parameter = -ln(sqrt(2*pi/(mu*pemr))).

    Returns
    -------
    float
        Electron power to the electrode [W].
    """
    return I_i * (2 * T_e + phi) * math.exp(Lambda - phi / T_e)


# ---------------------------------------------------------------------------
# Public solver
# ---------------------------------------------------------------------------


def solve(
    config: DeviceConfig,
    plasma: PlasmaState,
    x0: float | None = None,
    floating: bool = False,
    cathode_current_A: float | None = None,
    anode_current_A: float | None = None,
    anode_T_e: float | None = None,
) -> SolverResult:
    """Solve for all sheath potentials and currents given device config and plasma state.

    Parameters
    ----------
    config : DeviceConfig
        Static device parameters (does not change between RK steps).
    plasma : PlasmaState
        Current electron temperature [eV] and density [cm⁻³].
    x0 : float or None
        Warm-start hint for the cathode sheath drop [V]. If None, a cold start
        bracket is used.
    floating : bool
        If True, override V_bank = 0, forcing the floating-potential solution
        where I_tot = V_b = 0 and phi_c = phi_a = T_e * Lambda (in the absence
        of thermionic emission).

    Returns
    -------
    SolverResult
        All potential drops, currents, and power quantities.

    Raises
    ------
    ConvergenceError
        If the root-finding bracket cannot be established.
    """

    T_e = plasma.T_e  # eV  (= V numerically for potential scaling)
    n_e = plasma.n_e  # cm⁻³

    # ------------------------------------------------------------------
    # Plasma-dependent derived quantities
    # ------------------------------------------------------------------

    # Parallel plasma conductivity [Ω⁻¹ cm⁻¹]
    sigma_par = 14.6 * T_e**1.5

    # Parallel plasma resistance [Ω]
    R_p = config.L_cath / (math.pi * config.R_cath**2 * sigma_par)

    # Ion sound speed [cm/s]:  C_s = sqrt(T_e [erg] / (mu * m_p [g]))
    # T_e [eV] * _e_SI [J/eV] * 1e7 [erg/J] = T_e [erg]
    C_s = math.sqrt(T_e * _e_SI * 1.0e7 / (config.mu * _mp_cgs))

    # Ion saturation current [A]:  A_c [cm²] * e [C] * n_e [cm⁻³] * C_s [cm/s]
    I_i = config.A_c * _e_SI * n_e * C_s * math.exp(-0.5)
    if cathode_current_A is not None:
        # The fluid already computed this Bohm current, with presheath-correct
        # local sampling; take it verbatim so the circuit and the fluid cannot
        # disagree about how much current leaves the cathode.
        I_i = float(cathode_current_A)
    # Anode ion current. The historical form scales the cathode current by the
    # mesh opacity on both faces, which assumes the anode sees the same plasma as
    # the cathode -- exactly the assumption a resolved gap breaks (§7).
    I_i_a = 2 * config.eta * I_i
    if anode_current_A is not None:
        I_i_a = float(anode_current_A)
    T_e_anode = T_e if anode_T_e is None else float(anode_T_e)
    tau_a = T_e_anode / T_e

    if not I_i_a > 0.0:
        # The anode sheath relation is psi_a = Lambda - ln(1 + J_anode/J_i_a),
        # which is singular at zero anode current: an anode that collects nothing
        # cannot close the discharge circuit. This is a physical limit, not a
        # numerical one, so it is reported rather than clamped. eta = 0 is a valid
        # legacy limit for the anode's neutral and heat throttles, but not for the
        # circuit -- turn off cathode_coupling instead.
        raise ValueError(
            "anode ion current is zero, so the discharge circuit cannot close "
            f"(eta={config.eta}, I_i={I_i:.6g} A). eta = 0 is a valid limit for "
            "the anode's neutral/heat throttles but not for the circuit; disable "
            "cathode_coupling to model a machine with no anode collection."
        )

    # Electron saturation current [A]
    I_e = I_i * math.exp(config.Lambda + 0.5)

    # Thermionic emission current [A] (precomputed in config)
    I_eth = config.I_eth

    # Temperature ratio (dimensionless): T_s [K] → eV via kB/e
    delta = _kB_SI * config.T_s / (_e_SI * T_e)

    # Circuit resistance ratio
    gamma = config.R_comp / R_p

    psi_bank = config.V_bank / T_e

    # Scaled currents
    J_i = I_i * R_p / T_e
    J_i_a = I_i_a * R_p / T_e
    J_eth = I_eth * R_p / T_e

    Lambda = config.Lambda + 0.5
    eta = config.eta
    mu = config.mu

    # Annular emission profile (empty = historical uniform disc).
    annular = bool(config.emission_Ts_K)
    if annular:
        I_eth_k = tuple(
            area
            * config.C_R
            * T_k**2
            * math.exp(-_e_SI * config.phi_wf / (_kB_SI * T_k))
            for T_k, area in zip(config.emission_Ts_K, config.emission_area_cm2)
        )
        J_eth_k = tuple(i * R_p / T_e for i in I_eth_k)
        delta_k = tuple(
            _kB_SI * T_k / (_e_SI * T_e) for T_k in config.emission_Ts_K
        )
        wetted = sum(
            a * f
            for a, f in zip(config.emission_area_cm2, config.emission_plasma_frac)
        )
        ion_frac_k = tuple(
            (a * f / wetted) if wetted > 0.0 else 0.0
            for a, f in zip(config.emission_area_cm2, config.emission_plasma_frac)
        )

    # ------------------------------------------------------------------
    # Root-find psi_c_plus
    # ------------------------------------------------------------------

    if floating:
        # ------------------------------------------------------------------
        # Floating (open-circuit) solution: I_tot = V_b = 0
        #
        # Two constraints must hold simultaneously:
        #   (1) psi_c_plus - psi_c_minus = Lambda  (net sheath = floating potential)
        #   (2) J_i*(1 - exp(Lambda - psi_c_plus)) = J_eth*exp(-psi_c_minus/delta)
        #       (current balance at cathode, I_tot = 0)
        #
        # Substituting (1) into (2) gives a single residual in psi_c_plus:
        #   f(x) = J_i*(1 - exp(Lambda-x)) - J_eth*exp(-(x-Lambda)/delta) = 0
        #
        # For J_eth = 0: f(Lambda) = 0 directly.
        # For J_eth > 0: virtual cathode always forms; f(Lambda) = -J_eth < 0,
        #   f(inf) → J_i > 0, so root exists in (Lambda, inf).
        # ------------------------------------------------------------------
        if J_eth <= 0.0:
            psi_c_plus = Lambda
            psi_c_minus = 0.0
            J_star = 0.0
            regime: Literal["classical", "virtual_cathode"] = "classical"
        else:
            regime = "virtual_cathode"

            def f_float(x: float) -> float:
                return J_i * (1.0 - _exp_clamped(Lambda - x)) - J_eth * _exp_clamped(
                    -(x - Lambda) / delta
                )

            a_f = Lambda
            b_f = Lambda + 10
            a_f, b_f = _find_bracket(f_float, a_f, b_f)
            psi_c_plus = brentq(
                f_float, a_f, b_f, xtol=1.0e-8, rtol=1.0e-6, full_output=False
            )
            psi_c_minus = psi_c_plus - Lambda
            J_star = J_eth * math.exp(-psi_c_minus / delta)

        J_tot = 0.0
        psi_a = Lambda  # J_tot = 0  →  psi_a = Lambda - ln(1) = Lambda

        phi_c_plus = psi_c_plus * T_e
        phi_c_minus = psi_c_minus * T_e
        phi_c = phi_c_plus - phi_c_minus  # = T_e * Lambda by construction
        phi_a = psi_a * T_e_anode  # = T_e_anode * Lambda

        I_tot = 0.0
        I_eth_star = J_star * T_e / R_p

        V_p = 0.0
        V_b = 0.0

        P_wall = 0.0
        P_load = 0.0
        l_b = 0.0
        long_mfp = False
        beam_bypass_fraction = 0.0

    else:

        def _make_f(beam_bypass_fraction: float = 0.0):
            if annular:
                def f(x: float) -> float:
                    return _residual_annular(
                        x,
                        J_i,
                        Lambda,
                        gamma,
                        eta,
                        psi_bank,
                        mu,
                        J_i_a,
                        J_eth_k,
                        delta_k,
                        ion_frac_k,
                        beam_bypass_fraction,
                        tau_a,
                    )
                return f

            def f(x: float) -> float:
                return _residual(
                    x,
                    J_i,
                    J_eth,
                    Lambda,
                    gamma,
                    eta,
                    delta,
                    psi_bank,
                    mu,
                    J_i_a,
                    beam_bypass_fraction,
                    tau_a,
                )
            return f

        def _psi_minus_at(x: float) -> float:
            if annular:
                return _annular_emission_state(
                    x, J_i, mu, J_eth_k, delta_k, ion_frac_k
                )[1]
            return _psi_c_minus(x, J_i, J_eth, mu, delta)

        def _do_solve(f) -> float:
            nonlocal x0
            psi_cap = None
            if config.phi_sheath_max is not None:
                psi_cap = config.phi_sheath_max / T_e + Lambda + 2.0
            if x0 is not None:
                x0_psi = x0 / T_e
                # Two-stage warm window around the previous root: the narrow
                # window first, then a wider one, so a root that drifted
                # moderately is still tracked by continuity instead of the
                # solve jumping to a far root through the full bracket (the
                # jump is what flaps phi_c between branches step to step).
                for lo, hi in ((0.5, 2.0), (0.125, 8.0)):
                    a_w = max(1.0e-8, x0_psi * lo)
                    b_w = x0_psi * hi
                    try:
                        a_w, b_w = _find_bracket(
                            f, a_w, b_w, max_doublings=2, b_max=psi_cap
                        )
                        return brentq(
                            f, a_w, b_w, xtol=1.0e-8, rtol=1.0e-6, full_output=False
                        )
                    except ConvergenceError:
                        continue
                x0 = None
            a = 1.0e-8
            b = psi_bank + Lambda + 2.0
            if psi_cap is not None:
                b = min(b, psi_cap)
            try:
                a, b = _find_bracket(f, a, b, b_max=psi_cap)
            except ConvergenceError:
                if psi_cap is None:
                    raise
                # No root under the physical ceiling: the load cannot carry
                # the loop current at physical voltages, i.e. a genuine
                # inductive kick. Admit it through the unconstrained bracket;
                # the circuit-state guards bound how far it can run.
                a, b = _find_bracket(f, 1.0e-8, psi_bank + Lambda + 2.0)
            return brentq(f, a, b, xtol=1.0e-8, rtol=1.0e-6, full_output=False)

        # Solve self-consistently for the sheath with a continuous beam-bypass
        # fraction. The bypass fraction is the survival probability over the
        # cathode-anode distance, exp(-L_cath / l_b), and replaces the old hard
        # branch at l_b > L_cath.
        beam_bypass_fraction = 0.0
        l_b = 0.0
        psi_c_plus = _do_solve(_make_f(beam_bypass_fraction))
        for _ in range(4):
            _pm = _psi_minus_at(psi_c_plus)
            l_b = _compute_l_b((psi_c_plus - _pm) * T_e, T_e, n_e, plasma.n_n, plasma.sigma_b)
            next_bypass = _compute_beam_bypass_fraction(l_b, config.L_cath)
            if abs(next_bypass - beam_bypass_fraction) < 1e-4:
                beam_bypass_fraction = next_bypass
                break
            beam_bypass_fraction = next_bypass
            psi_c_plus = _do_solve(_make_f(beam_bypass_fraction))
        _pm = _psi_minus_at(psi_c_plus)
        l_b = _compute_l_b((psi_c_plus - _pm) * T_e, T_e, n_e, plasma.n_n, plasma.sigma_b)
        beam_bypass_fraction = _compute_beam_bypass_fraction(l_b, config.L_cath)
        long_mfp = l_b > 0.0 and l_b > config.L_cath

        # ------------------------------------------------------------------
        # Recover all quantities at the solution
        # ------------------------------------------------------------------

        if annular:
            J_star, psi_c_minus, _clamped = _annular_emission_state(
                psi_c_plus, J_i, mu, J_eth_k, delta_k, ion_frac_k
            )
            regime: Literal["classical", "virtual_cathode"] = (
                "virtual_cathode" if _clamped else "classical"
            )
        else:
            J_crit = _j_eth_crit(psi_c_plus, J_i, mu)

            if J_eth <= 0.0 or J_eth <= J_crit:
                regime = "classical"
                J_star = J_eth if J_eth > 0.0 else 0.0
                psi_c_minus = 0.0
            else:
                regime = "virtual_cathode"
                J_star = J_crit
                psi_c_minus = delta * math.log(J_eth / J_crit)

        J_tot = J_i * (1.0 - _exp_clamped(Lambda - psi_c_plus)) + J_star

        J_anode = J_tot - eta * beam_bypass_fraction * J_star
        # Scaled anode sheath potential; clamp mirrors the residual's guard
        # (extreme-emission roots can land at a marginally negative argument).
        psi_a = Lambda - math.log(max(1.0 + J_anode / J_i_a, 1e-300))

        # Physical potentials [V]
        phi_c_plus = psi_c_plus * T_e
        phi_c_minus = psi_c_minus * T_e
        phi_c = phi_c_plus - phi_c_minus
        phi_a = psi_a * T_e_anode

        # Currents [A]
        I_tot = J_tot * T_e / R_p
        I_eth_star = J_star * T_e / R_p

        # Voltages [V]
        V_p = I_tot * R_p
        V_b = config.V_bank - I_tot * config.R_comp

        # Power [W]
        P_wall = I_tot * config.V_bank
        P_load = I_tot * V_b
    P_comp = I_tot**2 * config.R_comp
    P_prim = (1.0 - eta * beam_bypass_fraction) * I_eth_star * phi_c
    P_ohmic = I_tot * V_p
    P_cathode_e = _P_elec(phi_c, T_e, I_i, Lambda)
    P_cathode_i = _P_ion(phi_c, T_e, I_i)
    P_cathode_i_pl = _P_ion(phi_c, T_e, I_i_a, pl=True)
    P_anode_e = _P_elec(phi_a, T_e_anode, I_i_a, Lambda)
    P_anode_i = _P_ion(phi_a, T_e_anode, I_i_a)
    P_anode_i_pl = _P_ion(phi_a, T_e_anode, I_i_a, pl=True)
    _P_beam_bypass = eta * beam_bypass_fraction * I_eth_star * V_b
    P_net = P_load - P_cathode_e - P_cathode_i - P_anode_e - P_anode_i - _P_beam_bypass
    P_net2 = P_prim + P_ohmic - P_cathode_e - P_cathode_i_pl - P_anode_e - P_anode_i_pl
    P_loss = P_cathode_e + P_cathode_i_pl + P_anode_e + P_anode_i_pl

    return SolverResult(
        phi_c_plus=phi_c_plus,
        phi_c_minus=phi_c_minus,
        phi_c=phi_c,
        phi_a=phi_a,
        V_p=V_p,
        V_b=V_b,
        R_p=R_p,
        I_i=I_i,
        I_i_a=I_i_a,
        I_e=I_e,
        I_eth=I_eth,
        I_eth_star=I_eth_star,
        I_tot=I_tot,
        P_wall=P_wall,
        P_load=P_load,
        P_comp=P_comp,
        P_prim=P_prim,
        P_ohmic=P_ohmic,
        P_cathode_e=P_cathode_e,
        P_cathode_i=P_cathode_i,
        P_cathode_i_pl=P_cathode_i_pl,
        P_anode_e=P_anode_e,
        P_anode_i=P_anode_i,
        P_anode_i_pl=P_anode_i_pl,
        P_net=P_net,
        P_net2=P_net2,
        P_loss=P_loss,
        regime=regime,
        long_mfp=long_mfp,
        beam_bypass_fraction=beam_bypass_fraction,
        l_b=l_b,
    )


# ---------------------------------------------------------------------------
# Beam system solver
# ---------------------------------------------------------------------------


def solve_beam_system(
    config: DeviceConfig,
    Te: np.ndarray,
    ne: np.ndarray,
    nn: np.ndarray,
    beam_cross_prev: np.ndarray,
    plasma_cross: np.ndarray,
    I_ion: float,
    gas_type: str,
    x0: float | None = None,
    x0_twin: float | None = None,
    floating: bool = False,
    cathode_index: int = 0,
    twin_index: int = -1,
    anode_current_A: float | None = None,
    anode_T_e: float | None = None,
    anode_current_twin_A: float | None = None,
    anode_T_e_twin: float | None = None,
    b_beam_excitation: float = 0.0,
    beam_excitation_energy_eV: float = 21.218,
    beam_excitation_model: str = "2p_scalar",
) -> BeamResult:
    """Solve cathode sheath(s) and compute beam quantities for all cells.

    Calls solve() for the primary cathode at ``cathode_index`` and, when
    config.Twin is True, also for the twin at ``twin_index``. Those default to
    0 and -1, the end cells, which is where a lumped source/end geometry puts the
    cathodes. A resolved geometry must pass the cell adjacent to each cathode
    surface instead: cell 0 there is the plasma-dead plenum, and the beam
    quantities written here have to land on the same cell the caller launches the
    beam from.  Beam velocity, density, and
    EII cross section are computed from the sheath potential and stored in
    per-cell arrays (zero everywhere except at active cathode indices).

    Parameters
    ----------
    config        : DeviceConfig with Twin flag controlling twin-cathode logic
    Te, ne, nn    : plasma state arrays [cells]
    beam_cross_prev : EII cross section from previous step (feeds sigma_b)
    plasma_cross  : plasma cross-sectional area per cell [cm²]
    I_ion         : ionization potential [eV]
    gas_type      : "He" or "H"
    x0, x0_twin   : warm-start sheath drop hints [V]
    floating      : override V_bank=0 (open-circuit)

    Returns
    -------
    BeamResult
    """
    cells = len(Te)
    v_beam = np.zeros(cells)
    n_beam = np.zeros(cells)
    beam_cross = np.zeros(cells)
    beam_exc_cross = np.zeros(cells)
    beam_exc_energy = np.zeros(cells)

    result = solve(
        config,
        PlasmaState(
            T_e=Te[cathode_index],
            n_e=ne[cathode_index],
            n_n=nn[cathode_index],
            sigma_b=beam_cross_prev[cathode_index],
        ),
        x0=x0,
        floating=floating,
        anode_current_A=anode_current_A,
        anode_T_e=anode_T_e,
    )
    x0_next = result.phi_c_plus
    phi_c_0 = result.phi_c
    if phi_c_0 > I_ion:
        v_beam[cathode_index] = math.sqrt(2.0 * phi_c_0 * _erg_per_eV / _me_cgs)
        _I_beam_0 = result.I_eth_star * (1.0 - config.eta * result.beam_bypass_fraction)
        n_beam[cathode_index] = _I_beam_0 / (_e_SI * plasma_cross[cathode_index] * v_beam[cathode_index])
        if gas_type == "He":
            beam_cross[cathode_index] = He_EII_cross_lkup(phi_c_0 / I_ion)
        elif gas_type == "H":
            beam_cross[cathode_index] = H_EII_cross_lkup(phi_c_0)
        (
            beam_exc_cross[cathode_index],
            beam_exc_energy[cathode_index],
        ) = beam_excitation_channel(
            phi_c_0,
            b_beam_excitation,
            gas_type,
            model=beam_excitation_model,
            threshold_eV=beam_excitation_energy_eV,
        )

    result_twin = None
    x0_twin_next = None
    if config.Twin:
        result_twin = solve(
            config,
            PlasmaState(
                T_e=Te[twin_index],
                n_e=ne[twin_index],
                n_n=nn[twin_index],
                sigma_b=beam_cross_prev[twin_index],
            ),
            x0=x0_twin,
            floating=floating,
            anode_current_A=anode_current_twin_A,
            anode_T_e=anode_T_e_twin,
        )
        x0_twin_next = result_twin.phi_c_plus
        phi_c_1 = result_twin.phi_c
        if phi_c_1 > I_ion:
            v_beam[twin_index] = math.sqrt(2.0 * phi_c_1 * _erg_per_eV / _me_cgs)
            _I_beam_1 = result_twin.I_eth_star * (
                1.0 - config.eta * result_twin.beam_bypass_fraction
            )
            n_beam[twin_index] = _I_beam_1 / (_e_SI * plasma_cross[twin_index] * v_beam[twin_index])
            if gas_type == "He":
                beam_cross[twin_index] = He_EII_cross_lkup(phi_c_1 / I_ion)
            elif gas_type == "H":
                beam_cross[twin_index] = H_EII_cross_lkup(phi_c_1)
            (
                beam_exc_cross[twin_index],
                beam_exc_energy[twin_index],
            ) = beam_excitation_channel(
                phi_c_1,
                b_beam_excitation,
                gas_type,
                model=beam_excitation_model,
                threshold_eV=beam_excitation_energy_eV,
            )

    beam_atten_cross = beam_cross + beam_exc_cross
    n_beam_ion = n_beam * beam_cross * v_beam
    A_ion_beam = n_beam_ion * nn

    l_b = np.zeros(cells)
    p_beam = np.zeros(cells)
    if beam_cross[cathode_index] != 0.0:
        l_b[cathode_index] = result.l_b
        p_beam[cathode_index] = (
            l_b[cathode_index] * beam_cross[cathode_index] * nn[cathode_index]
        )
    if config.Twin and result_twin is not None and beam_cross[twin_index] != 0.0:
        l_b[twin_index] = result_twin.l_b
        p_beam[twin_index] = (
            l_b[twin_index] * beam_cross[twin_index] * nn[twin_index]
        )

    # Attenuation along the column uses the total inelastic cross section:
    # an excitation collision degrades the primary as surely as an ionizing
    # one, so with the excitation channel on the deposition length shortens.
    l_b_profile = np.zeros(cells)
    if beam_cross[cathode_index] != 0.0:
        for j in range(cells):
            l_b_profile[j] = _compute_l_b(result.phi_c, Te[j], ne[j], nn[j], beam_atten_cross[cathode_index])

    l_b_profile_twin = np.zeros(cells)
    if config.Twin and result_twin is not None and beam_cross[twin_index] != 0.0:
        for j in range(cells):
            l_b_profile_twin[j] = _compute_l_b(result_twin.phi_c, Te[j], ne[j], nn[j], beam_atten_cross[twin_index])

    return BeamResult(
        result=result,
        result_twin=result_twin,
        v_beam=v_beam,
        n_beam=n_beam,
        beam_cross=beam_cross,
        beam_exc_cross=beam_exc_cross,
        beam_atten_cross=beam_atten_cross,
        n_beam_ion=n_beam_ion,
        A_ion_beam=A_ion_beam,
        l_b=l_b,
        p_beam=p_beam,
        l_b_profile=l_b_profile,
        l_b_profile_twin=l_b_profile_twin,
        x0_next=x0_next,
        x0_twin_next=x0_twin_next,
        beam_exc_energy_eV=beam_exc_energy,
    )
