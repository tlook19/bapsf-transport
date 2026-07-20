"""Current-driven cathode sheath solve (CATHODE_IDRIVEN_PLAN.md M2).

The voltage-driven solver (``_cathode_solver.solve``) finds the intersection
of the device curve with a Thevenin load line; under an inductive circuit
fold both curves are near-vertical and the intersection is ill-conditioned
(measured: a <=0.1 V physical perturbation moving the operating point by
800 A, and plateau per-solve V_b flapping over p5-p95 ~ 92-503 V while
I_tot moves 3 A). This module inverts the formulation: given the loop
current -- a smooth, inductor-integrated state -- find the sheath. The
device relation

    J_tot(psi) = J_i * (1 - exp(Lambda - psi)) + J_star(psi)

is monotone increasing in psi (the electron-repelling term and the
space-charge release both grow with sheath depth; the annular sum preserves
this), so the root is unique on a fixed physical bracket and brentq needs
no warm windows, no expanding-bracket ladder, and no ``phi_sheath_max``
heuristics. The anode sheath and the beam-bypass fraction follow
*explicitly* from the solved psi, so the voltage-driven path's bypass
fixed-point iteration disappears as well: one bracketed root-find per
solve, unconditionally.

Everything physical is **imported** from ``_cathode_solver`` -- Richardson
emission via ``DeviceConfig``, the space-charge release ``_j_eth_crit``,
the annular emission state, the sheath power bookkeeping, and the beam
pieces -- so the two solvers cannot drift apart. ``_cathode_solver`` itself
is not modified (hard constraint, plan section 3): its solve paths remain
the historical voltage-driven ones.

``SolverResult`` is returned field-for-field compatible, with two contract
notes:

- ``V_b`` is assembled as the device voltage ``phi_c + V_p - phi_a``. At a
  voltage-driven operating point this equals that solver's
  ``V_bank_eff - I_tot*R_comp_eff`` up to the *voltage-driven* root
  residual (~1e-4 V); the equivalence gate therefore compares device
  voltages on both sides.
- ``regime`` may additionally be ``"capability_limited"``: the imposed
  current exceeds what the sheath can carry at the bracket ceiling
  (``phi_c_cap_V``), i.e. a genuine inductive kick. The bracket-top
  solution is returned with its correspondingly large ``V_b`` and the
  circuit is expected to ramp the current down at ~V/L per step. No
  exception, no fallback ladder.

Floating (open-circuit) solves keep using ``_cathode_solver.solve`` -- its
floating branch models Boltzmann-suppressed emission over the virtual
barrier, which is not the same limit as ``I_tot = 0`` through the hard
space-charge clamp here; routing is the M3 dispatcher's job.

Schottky barrier lowering (opt-in, plan section 2b): the extracting sheath
field lowers the effective work function,
``dphi = sqrt(e E_s / 4 pi eps0)``, tilting the vertical emission ceiling
into a sloped line -- physical conditioning of the knee. Closure (stated,
per plan): the surface field is the Child-Langmuir diode field of the
classical sheath, ``E_s = (4/3) phi_c / s_CL`` with
``s_CL = (sqrt(2)/3) lambda_D (2 psi)^(3/4)``; it applies on the
temperature-limited branch only. The per-annulus/per-disc branches are:

    J_eth_raw > J_crit          virtual cathode: J_star = J_crit,
                                psi_minus = delta*ln(J_eth_raw/J_crit)
                                (space-charge barrier; surface field ~ 0,
                                no enhancement -- it vanishes
                                self-consistently)
    J_eth_eff > J_crit >= raw   marginally choked: J_star = J_crit,
                                psi_minus = 0 (the enhancement is exactly
                                eaten by space charge)
    otherwise                   classical: J_star = J_eth_eff

which is continuous in psi, reduces bit-for-bit to the historical branches
with the term off, and preserves monotonicity. Any phi_wf fit must state
this term's on/off status (plan section 3b): the lowering is ~0.05-0.1 eV,
the same order as the fit resolution.
"""

import math
import sys

import numpy as np
from scipy.optimize import brentq

from cablp.funcs._cathode_solver import (
    BeamResult,
    DeviceConfig,
    PlasmaState,
    SolverResult,
    _P_elec,
    _P_ion,
    _annular_emission_state,
    _compute_beam_bypass_fraction,
    _compute_l_b,
    _e_SI,
    _erg_per_eV,
    _exp_clamped,
    _j_eth_crit,
    _kB_SI,
    _me_cgs,
    _mp_cgs,
    beam_excitation_cross,
)
from cablp.funcs._cross import H_EII_cross_lkup, He_EII_cross_lkup

__all__ = ["solve_idriven", "solve_beam_system_idriven"]

# Float-degeneracy margin for the emission-exhausted plateau. Where every
# emission channel is released and the electron-repelling tail has
# underflowed, J_tot(psi) is numerically *constant* over a wide psi range
# (measured: slope ~ e^-50 at a Te = 3 eV deep-virtual-cathode corner), so
# an imposed current within float noise of the plateau cannot select a
# unique psi -- the device is a current source there and psi is genuinely
# not recoverable from I alone. The margin makes the selection
# deterministic: the solve targets J_imposed minus a few ulps and lands on
# the plateau's *leading edge* (the minimal sheath that carries the
# current). At well-conditioned operating points the shift is
# tol/(dJ/dpsi) ~ 1e-7 V -- far below the 1e-8-relative equivalence gate.
# The Schottky term removes the degeneracy physically (the ceiling gains
# dJ/dpsi > 0 everywhere); M3 should watch breakdown-adjacent states for
# leading-edge/capability chatter if it runs with Schottky off.
_J_PLATEAU_TOL_REL = 64.0 * sys.float_info.epsilon

# Schottky constant: dphi[eV] = sqrt(e * E / (4 pi eps0)) with E in V/m.
_SCHOTTKY_EV_PER_SQRT_V_M = 3.7946865e-5


def _schottky_lowering_eV(phi_c_V: float, T_e: float, n_e: float) -> float:
    """Work-function lowering [eV] from the classical sheath's surface field.

    Child-Langmuir diode field at the emitter for a sheath drop ``phi_c_V``
    over the CL sheath width; zero for a non-extracting (<=0) drop.
    """
    if phi_c_V <= 0.0 or T_e <= 0.0 or n_e <= 0.0:
        return 0.0
    lambda_D_cm = 743.0 * math.sqrt(T_e / n_e)
    psi = phi_c_V / T_e
    s_cl_cm = (math.sqrt(2.0) / 3.0) * lambda_D_cm * (2.0 * psi) ** 0.75
    if s_cl_cm <= 0.0:
        return 0.0
    E_V_per_m = (4.0 / 3.0) * phi_c_V / s_cl_cm * 100.0
    return _SCHOTTKY_EV_PER_SQRT_V_M * math.sqrt(E_V_per_m)


def _uniform_state_schottky(
    psi: float,
    J_i: float,
    J_eth: float,
    mu: float,
    delta: float,
    T_e: float,
    n_e: float,
) -> tuple[float, float, bool]:
    """Uniform-disc ``(J_star, psi_minus, clamped)`` with Schottky lowering."""
    if J_eth <= 0.0:
        return 0.0, 0.0, False
    J_crit = _j_eth_crit(psi, J_i, mu)
    if J_eth > J_crit:
        # Deep space-charge clamp: surface field ~ 0, no enhancement.
        if J_crit <= 0.0:
            return 0.0, 0.0, True
        return J_crit, delta * math.log(J_eth / J_crit), True
    dphi = _schottky_lowering_eV(psi * T_e, T_e, n_e)
    J_eff = J_eth * math.exp(dphi / (delta * T_e))
    if J_eff > J_crit:
        # Enhancement exactly eaten by space charge: choked, no barrier.
        return J_crit, 0.0, True
    return J_eff, 0.0, False


def _annular_state_schottky(
    psi: float,
    J_i: float,
    mu: float,
    J_eth_k: tuple,
    delta_k: tuple,
    ion_frac_k: tuple,
    T_e: float,
    n_e: float,
) -> tuple[float, float, bool]:
    """Annular ``(J_star, psi_minus_eff, any_clamped)`` with Schottky lowering.

    Mirrors ``_cathode_solver._annular_emission_state`` (all annuli share the
    equipotential ``psi``; the effective barrier is the emission-weighted
    mean of the local ones) with the three-branch Schottky rule per annulus.
    """
    dphi = _schottky_lowering_eV(psi * T_e, T_e, n_e)
    J_star_total = 0.0
    weighted_pm = 0.0
    any_clamped = False
    for J_eth_a, delta_a, frac_a in zip(J_eth_k, delta_k, ion_frac_k):
        if J_eth_a <= 0.0:
            continue
        J_crit_a = _j_eth_crit(psi, J_i * frac_a, mu) if frac_a > 0.0 else 0.0
        if J_eth_a > J_crit_a:
            any_clamped = True
            if J_crit_a <= 0.0:
                continue
            J_star_total += J_crit_a
            weighted_pm += J_crit_a * delta_a * math.log(J_eth_a / J_crit_a)
            continue
        J_eff_a = J_eth_a * math.exp(dphi / (delta_a * T_e))
        if J_eff_a > J_crit_a:
            any_clamped = True
            J_star_total += J_crit_a
            continue
        J_star_total += J_eff_a
    psi_minus_eff = weighted_pm / J_star_total if J_star_total > 0.0 else 0.0
    return J_star_total, psi_minus_eff, any_clamped


def solve_idriven(
    config: DeviceConfig,
    plasma: PlasmaState,
    I_tot_A: float,
    cathode_current_A: float | None = None,
    anode_current_A: float | None = None,
    anode_T_e: float | None = None,
    schottky: bool = False,
    phi_c_cap_V: float = 1000.0,
) -> SolverResult:
    """Solve the cathode sheath for an *imposed* loop current.

    Parameters mirror ``_cathode_solver.solve`` where shared:
    ``cathode_current_A``/``anode_current_A``/``anode_T_e`` are the same
    "the fluid already computed this" overrides. ``I_tot_A`` is the loop
    current the external circuit is driving through the device this step
    (must be >= 0; the circuit's transistor/diode clamp owns the sign).
    ``phi_c_cap_V`` is the fixed physical bracket ceiling on the classical
    sheath drop; a current the sheath cannot carry below it returns the
    bracket-top solution tagged ``regime="capability_limited"``.

    Returns a ``SolverResult`` field-for-field compatible with the
    voltage-driven solver's (see the module docstring for the ``V_b`` and
    ``regime`` contract notes).
    """
    if I_tot_A < 0.0:
        raise ValueError(
            f"I_tot_A must be >= 0 (got {I_tot_A}); the circuit's diode "
            "clamp owns the current sign"
        )
    if phi_c_cap_V <= 0.0:
        raise ValueError(f"phi_c_cap_V must be positive (got {phi_c_cap_V})")

    T_e = plasma.T_e
    n_e = plasma.n_e

    # ------------------------------------------------------------------
    # Plasma-derived quantities: identical formulas to the voltage-driven
    # solve, so the two solvers agree bit-for-bit on the operating map.
    # ------------------------------------------------------------------
    sigma_par = 14.6 * T_e**1.5
    R_p = config.L_cath / (math.pi * config.R_cath**2 * sigma_par)
    C_s = math.sqrt(T_e * _e_SI * 1.0e7 / (config.mu * _mp_cgs))
    I_i = config.A_c * _e_SI * n_e * C_s * math.exp(-0.5)
    if cathode_current_A is not None:
        I_i = float(cathode_current_A)
    I_i_a = 2 * config.eta * I_i
    if anode_current_A is not None:
        I_i_a = float(anode_current_A)
    T_e_anode = T_e if anode_T_e is None else float(anode_T_e)
    if not I_i_a > 0.0:
        raise ValueError(
            "anode ion current is zero, so the discharge circuit cannot "
            f"close (eta={config.eta}, I_i={I_i:.6g} A); disable "
            "cathode_coupling to model a machine with no anode collection."
        )

    I_e = I_i * math.exp(config.Lambda + 0.5)
    I_eth = config.I_eth
    delta = _kB_SI * config.T_s / (_e_SI * T_e)
    Lambda = config.Lambda + 0.5
    eta = config.eta
    mu = config.mu

    J_i = I_i * R_p / T_e
    J_i_a = I_i_a * R_p / T_e
    J_eth = I_eth * R_p / T_e
    J_imposed = float(I_tot_A) * R_p / T_e

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
    # The monotone device relation and its single bracketed root
    # ------------------------------------------------------------------
    def _emission_state(psi: float) -> tuple[float, float, bool]:
        if annular:
            if schottky:
                return _annular_state_schottky(
                    psi, J_i, mu, J_eth_k, delta_k, ion_frac_k, T_e, n_e
                )
            return _annular_emission_state(
                psi, J_i, mu, J_eth_k, delta_k, ion_frac_k
            )
        if schottky:
            return _uniform_state_schottky(
                psi, J_i, J_eth, mu, delta, T_e, n_e
            )
        # Historical uniform-disc branches (bit-identical to the voltage-
        # driven solve's recovery step).
        if J_eth <= 0.0:
            return 0.0, 0.0, False
        J_crit = _j_eth_crit(psi, J_i, mu)
        if J_eth <= J_crit:
            return J_eth, 0.0, False
        return J_crit, delta * math.log(J_eth / J_crit), True

    def _J_tot(psi: float) -> float:
        return (
            J_i * (1.0 - _exp_clamped(Lambda - psi))
            + _emission_state(psi)[0]
        )

    def _net_phi_c(psi: float) -> float:
        return (psi - _emission_state(psi)[1]) * T_e

    # The physical ceiling applies to the *net* sheath drop phi_c: in a deep
    # virtual cathode psi_c_plus legitimately exceeds any voltage-scale cap
    # while phi_c = (psi_plus - psi_minus)*T_e stays at bank scale (the
    # barrier eats the difference), and both psi_plus -> phi_c and
    # psi_plus -> J_tot are monotone increasing. So: extend the bracket top
    # geometrically until either the root is inside (f >= 0) or the net
    # sheath exceeds the cap. This is deterministic range extension on a
    # monotone function -- there is exactly one root and no branch to
    # mis-select -- not the voltage-driven path's root-hunting ladder.
    psi_lo = 1.0e-8
    psi_top = max(phi_c_cap_V / T_e, Lambda + 2.0)
    capability_limited = False
    # Stage 1: the exact target. Well-conditioned operating points resolve
    # here and the equivalence with the voltage-driven solve is pristine.
    J_target = J_imposed
    for _ in range(200):
        if _J_tot(psi_top) >= J_target:
            break
        if _net_phi_c(psi_top) >= phi_c_cap_V:
            capability_limited = True
            break
        psi_top *= 2.0
    else:
        capability_limited = True

    if capability_limited and _J_tot(psi_top) >= J_imposed - (
        _J_PLATEAU_TOL_REL * abs(J_imposed)
    ):
        # Stage 2: the imposed current is carriable to within float noise
        # but the exact target is unreachable -- the sub-ulp-flat plateau
        # (see _J_PLATEAU_TOL_REL). Deterministic leading-edge selection
        # against the margined target.
        capability_limited = False
        J_target = J_imposed - _J_PLATEAU_TOL_REL * abs(J_imposed)

    if capability_limited:
        # A genuine inductive kick: the sheath cannot carry the imposed
        # current at physical net voltages. Return the solution *at* the
        # ceiling -- net phi_c = phi_c_cap_V, located by a bracketed solve
        # on the monotone net-sheath map so the reported kick voltage does
        # not depend on where the doubling happened to land -- and let the
        # circuit ramp I down at ~V/L per step.
        if _net_phi_c(psi_top) > phi_c_cap_V:
            psi_c_plus = brentq(
                lambda x: _net_phi_c(x) - phi_c_cap_V,
                psi_lo,
                psi_top,
                xtol=1.0e-12,
                rtol=1.0e-14,
                full_output=False,
            )
        else:
            psi_c_plus = psi_top
    else:
        # f(psi_lo) ~ -J_i*exp(Lambda) < 0 <= J_target, so the bracket is
        # valid by construction; brentq is run tight because it is the only
        # root-find in the module and costs microseconds.
        psi_c_plus = brentq(
            lambda x: _J_tot(x) - J_target,
            psi_lo,
            psi_top,
            xtol=1.0e-12,
            rtol=1.0e-14,
            full_output=False,
        )

    # ------------------------------------------------------------------
    # Everything else follows explicitly from the solved psi
    # ------------------------------------------------------------------
    J_star, psi_c_minus, clamped = _emission_state(psi_c_plus)
    J_tot = J_i * (1.0 - _exp_clamped(Lambda - psi_c_plus)) + J_star
    regime = (
        "capability_limited"
        if capability_limited
        else ("virtual_cathode" if clamped else "classical")
    )

    phi_c_plus = psi_c_plus * T_e
    phi_c_minus = psi_c_minus * T_e
    phi_c = phi_c_plus - phi_c_minus

    # Beam MFP and bypass: explicit evaluation at the solved sheath (the
    # voltage-driven path needs a fixed-point loop here only because its
    # residual feeds bypass back into the root equation).
    l_b = _compute_l_b(phi_c, T_e, n_e, plasma.n_n, plasma.sigma_b)
    beam_bypass_fraction = _compute_beam_bypass_fraction(l_b, config.L_cath)
    long_mfp = l_b > 0.0 and l_b > config.L_cath

    J_anode = J_tot - eta * beam_bypass_fraction * J_star
    psi_a = Lambda - math.log(max(1.0 + J_anode / J_i_a, 1e-300))
    phi_a = psi_a * T_e_anode

    I_tot = J_tot * T_e / R_p
    I_eth_star = J_star * T_e / R_p

    V_p = I_tot * R_p
    # Device voltage from the loop bookkeeping (see module docstring).
    V_b = phi_c + V_p - phi_a
    if capability_limited:
        # The kick MUST be a back-EMF at least as large as the sheath
        # ceiling, monotone-nondecreasing in the imposed current. Without
        # this clamp the frozen ceiling solution can report a *negative*
        # device voltage (beam bypass drives J_anode below -J_i_a, the
        # anode-log clamp fires, phi_a explodes positive) which the
        # circuit reads as a huge forward EMF that no longer grows with I
        # -- the diode backstop never engages and the loop current runs
        # away (measured: I_loop -> 8e8 A before the fluid went
        # non-finite, first full-physics run 2026-07-20). The carried
        # current is likewise floored at zero: past the ceiling the
        # sheath delivers what it can, never a backwards current.
        I_tot = max(I_tot, 0.0)
        V_p = I_tot * R_p
        V_b = max(V_b, float(phi_c_cap_V))

    P_wall = I_tot * (V_b + I_tot * config.R_comp)
    P_load = I_tot * V_b
    P_comp = I_tot**2 * config.R_comp
    P_prim = (1.0 - eta * beam_bypass_fraction) * I_eth_star * phi_c
    P_ohmic = I_tot * V_p
    # Electron sheath powers with *physical flux barriers* -- a deliberate,
    # documented divergence from the frozen module's `_P_elec(phi_net,...)`:
    # (i) plasma electrons reaching the cathode surface climb the classical
    # peak phi_c_plus, not the net phi_c -- in a deep virtual cathode the
    # net can go slightly NEGATIVE while the barrier stays high, and the
    # historical exp(Lambda - phi_net/T) then explodes as exp(|phi_net|/T)
    # (measured: -180 kW of spurious heating into a 0.1 eV floor cell at
    # the first I~0 drive solve, detonating the fluid in one step);
    # (ii) an attracting electrode (phi < 0) collects at most electron
    # *saturation* -- the flux factor is capped at exp(Lambda). Both reduce
    # bit-for-bit to the historical formulas in the classical repelling
    # regime (phi_minus = 0, phi_a >= 0), which is where the M2 equivalence
    # gate lives.
    P_cathode_e = (
        I_i
        * (2.0 * T_e + phi_c)
        * _exp_clamped(Lambda - max(phi_c_plus, 0.0) / T_e)
    )
    P_cathode_i = _P_ion(phi_c, T_e, I_i)
    P_cathode_i_pl = _P_ion(phi_c, T_e, I_i_a, pl=True)
    P_anode_e = (
        I_i_a
        * (2.0 * T_e_anode + phi_a)
        * _exp_clamped(Lambda - max(phi_a, 0.0) / T_e_anode)
    )
    P_anode_i = _P_ion(phi_a, T_e_anode, I_i_a)
    P_anode_i_pl = _P_ion(phi_a, T_e_anode, I_i_a, pl=True)
    _P_beam_bypass = eta * beam_bypass_fraction * I_eth_star * V_b
    P_net = (
        P_load - P_cathode_e - P_cathode_i - P_anode_e - P_anode_i
        - _P_beam_bypass
    )
    P_net2 = (
        P_prim + P_ohmic - P_cathode_e - P_cathode_i_pl - P_anode_e
        - P_anode_i_pl
    )
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


def solve_beam_system_idriven(
    config: DeviceConfig,
    Te: np.ndarray,
    ne: np.ndarray,
    nn: np.ndarray,
    beam_cross_prev: np.ndarray,
    plasma_cross: np.ndarray,
    I_ion: float,
    gas_type: str,
    I_tot_A: float,
    cathode_index: int = 0,
    anode_current_A: float | None = None,
    anode_T_e: float | None = None,
    b_beam_excitation: float = 0.0,
    beam_excitation_energy_eV: float = 21.218,
    schottky: bool = False,
    phi_c_cap_V: float = 1000.0,
) -> BeamResult:
    """Current-driven, single-cathode counterpart of ``solve_beam_system``.

    Calls ``solve_idriven`` for the primary cathode and assembles the same
    per-cell beam arrays with the same formulas (mirrored from the frozen
    voltage-driven module because that assembly lives inside its
    ``solve_beam_system``, which hardwires ``solve``). Twin cathodes are
    out of scope for the current-driven path (the dispatcher raises before
    this is reached), so ``result_twin`` is always ``None`` and the twin
    arrays stay zero. ``x0_next`` is kept for contract compatibility --
    the current-driven solve needs no warm start.
    """
    cells = len(Te)
    v_beam = np.zeros(cells)
    n_beam = np.zeros(cells)
    beam_cross = np.zeros(cells)
    beam_exc_cross = np.zeros(cells)

    result = solve_idriven(
        config,
        PlasmaState(
            T_e=Te[cathode_index],
            n_e=ne[cathode_index],
            n_n=nn[cathode_index],
            sigma_b=beam_cross_prev[cathode_index],
        ),
        I_tot_A=I_tot_A,
        anode_current_A=anode_current_A,
        anode_T_e=anode_T_e,
        schottky=schottky,
        phi_c_cap_V=phi_c_cap_V,
    )
    phi_c_0 = result.phi_c
    if phi_c_0 > I_ion:
        v_beam[cathode_index] = math.sqrt(2.0 * phi_c_0 * _erg_per_eV / _me_cgs)
        _I_beam_0 = result.I_eth_star * (
            1.0 - config.eta * result.beam_bypass_fraction
        )
        n_beam[cathode_index] = _I_beam_0 / (
            _e_SI * plasma_cross[cathode_index] * v_beam[cathode_index]
        )
        if gas_type == "He":
            beam_cross[cathode_index] = He_EII_cross_lkup(phi_c_0 / I_ion)
        elif gas_type == "H":
            beam_cross[cathode_index] = H_EII_cross_lkup(phi_c_0)
        beam_exc_cross[cathode_index] = beam_excitation_cross(
            phi_c_0,
            b_beam_excitation,
            gas_type,
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

    l_b_profile = np.zeros(cells)
    if beam_cross[cathode_index] != 0.0:
        for j in range(cells):
            l_b_profile[j] = _compute_l_b(
                result.phi_c, Te[j], ne[j], nn[j],
                beam_atten_cross[cathode_index],
            )

    return BeamResult(
        result=result,
        result_twin=None,
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
        l_b_profile_twin=np.zeros(cells),
        x0_next=result.phi_c_plus,
        x0_twin_next=None,
    )
