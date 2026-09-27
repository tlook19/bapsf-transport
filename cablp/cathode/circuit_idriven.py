"""Current-driven cathode sheath solve.

Given the loop current -- a smooth, inductor-integrated state -- find the
sheath. The device relation

    J_tot(psi) = J_i * (1 - exp(Lambda - psi)) + J_star(psi)

is monotone increasing in psi (the electron-repelling term and the
space-charge release both grow with sheath depth; the annular sum preserves
this), so the root is unique on a fixed physical bracket and one bracketed
brentq finds it. The anode sheath and the beam-bypass fraction follow
*explicitly* from the solved psi: one bracketed root-find per solve,
unconditionally.

The shared physics is **imported** from :mod:`cablp.cathode.circuit_common`
-- Richardson emission via ``DeviceConfig``, the space-charge release
``j_eth_crit``, the annular emission state, the sheath power bookkeeping,
and the beam pieces -- so this solve and the prescribed-measured one
(:mod:`cablp.cathode.circuit_prescribed`) cannot drift apart.

Contract notes on the returned ``SolverResult``:

- ``V_b`` is assembled as the device voltage ``phi_c + V_p - phi_a``.
- ``regime`` may be ``"capability_limited"``: the imposed current exceeds
  what the sheath can carry at the bracket ceiling ``phi_c_cap_V``, i.e. a
  genuine inductive kick. The bracket-top solution is returned with its
  correspondingly large ``V_b`` and the circuit is expected to ramp the
  current down at ~V/L per step. No exception, no fallback ladder.

Floating (open-circuit) solves come HERE, at ``I_tot_A = 0``. An open
circuit is the zero-current member of this same family: the surface finds
the potential at which its space-charge-limited emission plus the ion
current is exactly returned by collected plasma electrons, which at a hot
emitter is an electron-COLLECTING sheath a couple of ``T_e`` deep, not the
non-emitting floating drop. The reported ``I_tot`` is RECONSTRUCTED from
the sheath at the root, so it recovers the imposed zero only to root-finder
roundoff at the scale of the currents it is built from (order 1e-12 of
``I_eth_star``), not to an exact literal zero;
``I_cathode_kirchhoff_residual`` is the quantity to assert on.

Schottky barrier lowering (opt-in): the extracting sheath
field lowers the effective work function,
``dphi = sqrt(e E_s / 4 pi eps0)``, tilting the vertical emission ceiling
into a sloped line -- physical conditioning of the knee. Closure (stated
explicitly because it is a modelling choice): the surface field is the
Child-Langmuir diode field of the
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
this term's on/off status: the lowering is ~0.05-0.1 eV,
the same order as the fit resolution.
"""

import math
import sys

import numpy as np
from scipy.optimize import brentq

from cablp.cathode.beam_deposition import (
    HE_EII_EDGE_REL_TOL,
    HE_EII_EPS_TOP,
)
from cablp.cathode.circuit_common import (
    E_SI,
    ERG_PER_EV,
    KB_SI,
    ME_CGS,
    BeamResult,
    DeviceConfig,
    P_ion,
    PlasmaState,
    SolverResult,
    annular_emission_state,
    beam_launched_current_A,
    c_log_ei,
    compute_beam_bypass_fraction,
    compute_l_b,
    exp_clamped,
    j_eth_crit,
)
from cablp.plasma.params import (
    LN_LAMBDA_MIN,
    bohm_sound_speed as _bohm_sound_speed,
)
from cablp.atomic.cross_sections import He_EII_cross_lkup
from cablp.cathode.kernels import COMPILED_KERNELS as _COMPILED_KERNELS

__all__ = [
    "assemble_beam_arrays",
    "solve_idriven",
    "solve_beam_system_idriven",
]

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
# tol/(dJ/dpsi) ~ 1e-7 V. The Schottky term removes the degeneracy
# physically (the ceiling gains dJ/dpsi > 0 everywhere).
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
    J_crit = j_eth_crit(psi, J_i, mu)
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

    Mirrors ``circuit_common.annular_emission_state`` (all annuli share the
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
        J_crit_a = j_eth_crit(psi, J_i * frac_a, mu) if frac_a > 0.0 else 0.0
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


# The pure-Python kernels stay reachable under their own names so the compiled
# path can be compared against them (equivalence sweeps, microbenchmarks)
# inside a process that has opted in.
_schottky_lowering_eV_pure = _schottky_lowering_eV
_annular_state_schottky_pure = _annular_state_schottky

# Compiled-kernel selection. Same contract as ``circuit_common``'s block: one
# rebinding site per name, at module scope, before any caller resolves it, so
# the hot path is a plain function object with no per-call branch.
# ``_COMPILED_ROOT`` is the whole root find for the annular + Schottky branch
# -- the bracket ladder plus brentq with the residual evaluated in C. It is
# ``None`` on the default pure path, and ``solve_idriven`` then runs the
# Python ladder, which is its reference transcription.
_COMPILED_ROOT = None
if _COMPILED_KERNELS is not None:
    _COMPILED_KERNELS.check_constants_idriven(_SCHOTTKY_EV_PER_SQRT_V_M)
    _schottky_lowering_eV = _COMPILED_KERNELS.schottky_lowering_eV
    _annular_state_schottky = _COMPILED_KERNELS.annular_state_schottky
    _COMPILED_ROOT = _COMPILED_KERNELS.solve_psi_annular_schottky


def solve_idriven(
    config: DeviceConfig,
    plasma: PlasmaState,
    I_tot_A: float,
    cathode_current_A: float | None = None,
    anode_current_A: float | None = None,
    anode_T_e: float | None = None,
    schottky: bool = False,
    phi_c_cap_V: float = 1000.0,
    alpha_sheath: float | None = None,
    alpha_sheath_anode: float | None = None,
    anode_electron_saturation_A: float | None = None,
    tail_anode_current_A: float = 0.0,
) -> SolverResult:
    """Solve the cathode sheath for an *imposed* loop current.

    ``cathode_current_A``/``anode_current_A``/``anode_T_e`` are "the fluid
    already computed this" overrides of the cathode ion current, the anode
    ion current and the anode electron temperature. ``I_tot_A`` is the loop
    current the external circuit is driving through the device this step
    (must be >= 0; the circuit's transistor/diode clamp owns the sign).
    ``schottky`` arms Schottky barrier lowering of the emission (see the
    module docstring).
    ``phi_c_cap_V`` is the fixed physical bracket ceiling on the net
    sheath drop; a current the sheath cannot carry below it returns the
    bracket-top solution tagged ``regime="capability_limited"``.
    ``alpha_sheath`` / ``alpha_sheath_anode`` are the two electrodes' own
    presheath factors (the sheath-edge density ``n_se = alpha * n_e``);
    ``None`` is the flat ``exp(-1/2)``.
    ``tail_anode_current_A`` is electron current [A] the anode collects
    DIRECTLY from the QL tail walkers the mesh intercepts, i.e. current that
    never crossed the anode sheath. It enters ``J_anode`` with the beam
    bypass's sign and for the same reason -- the sheath has that much less
    plasma-borne current to pass -- so it raises ``phi_a`` logarithmically.
    Its caller supplies the value the LAST accepted step measured: the
    deposition that produces it is solved after this, so the coupling is
    lagged one step rather than iterated. 0.0 (the default) is an exact
    identity on every float here.
    ``anode_electron_saturation_A`` is the EXPLICIT electron saturation current
    the mesh wires can draw -- the electron random flux ``n <v_e> / 4`` on the
    wire area the two faces present, ``2 eta A``, at the anode sample's own
    ``n`` and ``T_e``. The sheath relation caps the collected electron current
    at it: ``I_e,a = I_e,sat exp(-max(phi_a, 0) / T_e,a)`` and
    ``phi_a = T_e,a ln(I_e,sat / (I_i,a + I_anode))``. ``None`` (the default)
    rebuilds it as ``I_i_a * exp(Lambda_a)``, the implicit form -- the same
    number to roundoff wherever ``I_i_a`` is the analytic ``e^(-1/2) n c_s``
    Bohm collection on that same area, which is what the solver's anode sample
    hands over, so the two forms agree at the live configuration. The explicit
    member is carried because it states the cap as what it is, an electron
    random flux on the wire area, and because a caller whose ``I_i_a`` is not
    that analytic form has no other way to say so.

    Returns a ``SolverResult`` (see the module docstring for the ``V_b`` and
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
    # Plasma-derived quantities
    # ------------------------------------------------------------------
    # Parallel plasma conductivity [Ω⁻¹ cm⁻¹]. Spitzer, with the Coulomb
    # logarithm evaluated at the solve's own state rather than frozen: NRL
    # Formulary 2004 p.30 gives the TRANSVERSE resistivity
    # eta_perp = 1.03e-2 Z lnLambda T_e^-3/2 [Ohm cm], and p.38 gives
    # sigma_par = 1.96 sigma_perp at Z = 1 (Braginskii). The two literature
    # factors are left un-collapsed so the lineage stays readable. lnLambda is
    # floored at LN_LAMBDA_MIN, the same floor the transport terms use -- it
    # is a positivity guard for the cold, tenuous corner and does not bind at
    # any physical discharge state.
    ln_lambda = max(c_log_ei(T_e, n_e), LN_LAMBDA_MIN)
    sigma_par = (1.96 / (1.03e-2 * ln_lambda)) * T_e**1.5
    R_p = config.L_cath / (math.pi * config.R_cath**2 * sigma_par)
    C_s = float(_bohm_sound_speed(T_e, config.ion_mass_g))
    # Sheath-edge sampling (R3.2 / A16): the ion Bohm
    # current is drawn at the sheath-edge density n_se = alpha_sheath * n_e. The
    # historical flat exp(-1/2) is the Boltzmann drop across a presheath that
    # fits inside the cell; the fluid boundary instead uses the mesh-independent
    # ``sources.presheath_alpha``. R3.2 lets the caller pass that SAME factor so
    # the circuit current and the fluid sink read one n_se. Electron saturation
    # stays at the bulk density (the ``lam_shift`` that lifts electrons back to
    # n_e is -ln(alpha_sheath), = +0.5 for the flat default). ``None`` keeps the
    # exact flat +0.5.
    #
    # The cathode and the anode are DISTINCT sheaths sampled on different
    # presheaths (the cathode's long collisional presheath vs the anode mesh's
    # short geometric one), so each carries its own factor: ``alpha_sheath`` /
    # ``lam_shift`` for the cathode ion current, the cathode floating balance,
    # and P_cathode_e; ``alpha_sheath_anode`` / ``lam_shift_anode`` for the anode
    # floating potential (``psi_a``) and P_anode_e. The anode ION current
    # ``I_i_a`` is sampled on its own side by ``anode_circuit_sample`` and passed
    # in via ``anode_current_A`` -- never let one electrode's presheath leak into
    # the other's Lambda.
    def _sheath_factors(alpha):
        if alpha is None:
            return math.exp(-0.5), 0.5
        alpha = float(alpha)
        if not alpha > 0.0:
            raise ValueError(f"alpha_sheath must be positive (got {alpha})")
        return alpha, -math.log(alpha)

    alpha_eff, lam_shift = _sheath_factors(alpha_sheath)
    _alpha_eff_anode, lam_shift_anode = _sheath_factors(alpha_sheath_anode)
    I_i = config.A_c * E_SI * n_e * C_s * alpha_eff
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

    I_e = I_i * math.exp(config.Lambda + lam_shift)
    I_eth = config.I_eth
    delta = KB_SI * config.T_s / (E_SI * T_e)
    Lambda = config.Lambda + lam_shift
    Lambda_anode = config.Lambda + lam_shift_anode
    eta = config.eta
    mu = config.mu

    J_i = I_i * R_p / T_e
    J_i_a = I_i_a * R_p / T_e
    J_eth = I_eth * R_p / T_e
    J_imposed = float(I_tot_A) * R_p / T_e
    # A2a two-population split: current the anode collects from the QL TAIL
    # walkers rather than through its own sheath, scaled like every other
    # current here. It enters J_anode with the beam bypass's sign and for the
    # beam bypass's reason -- both are electrons reaching the electrode
    # without the sheath having to pass them, so the plasma-borne share the
    # sheath does have to pass is smaller by that much, and the anode sits a
    # few volts higher (logarithmically). 0.0 (the default) leaves every float
    # below exactly as it was.
    J_tail_a = float(tail_anode_current_A) * R_p / T_e
    # THE ELECTRON SATURATION the wires can draw, stated EXPLICITLY as the
    # random flux on the mesh area where the caller sampled it. The relation
    # below used to reach it implicitly, as ``I_i_a * exp(Lambda_a)``, which
    # is the same number only where ``I_i_a`` is the analytic Bohm collection
    # on that same area; ``None`` keeps that implicit form for callers with
    # no sample of their own.
    I_e_sat_a = (
        I_i_a * math.exp(Lambda_anode)
        if anode_electron_saturation_A is None
        else float(anode_electron_saturation_A)
    )
    if not I_e_sat_a > 0.0:
        raise ValueError(
            "anode electron saturation current must be positive (got "
            f"{anode_electron_saturation_A!r} A): it is the cap the anode "
            "sheath relation divides by, and a non-positive cap has no "
            "sheath solution"
        )

    annular = bool(config.emission_Ts_K)
    if annular:
        I_eth_k = tuple(
            area
            * config.C_R
            * T_k**2
            * math.exp(-E_SI * config.phi_wf / (KB_SI * T_k))
            for T_k, area in zip(config.emission_Ts_K, config.emission_area_cm2)
        )
        J_eth_k = tuple(i * R_p / T_e for i in I_eth_k)
        delta_k = tuple(
            KB_SI * T_k / (E_SI * T_e) for T_k in config.emission_Ts_K
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
            return annular_emission_state(
                psi, J_i, mu, J_eth_k, delta_k, ion_frac_k
            )
        if schottky:
            return _uniform_state_schottky(
                psi, J_i, J_eth, mu, delta, T_e, n_e
            )
        # Uniform-disc hard branches.
        if J_eth <= 0.0:
            return 0.0, 0.0, False
        J_crit = j_eth_crit(psi, J_i, mu)
        if J_eth <= J_crit:
            return J_eth, 0.0, False
        return J_crit, delta * math.log(J_eth / J_crit), True

    def _J_tot(psi: float) -> float:
        return (
            J_i * (1.0 - exp_clamped(Lambda - psi))
            + _emission_state(psi)[0]
        )

    def _net_phi_c(psi: float) -> float:
        return (psi - _emission_state(psi)[1]) * T_e

    def _reported_phi_c(psi: float) -> float:
        # The ceiling test on the located root uses the arithmetic the result
        # is ASSEMBLED with below (phi_c_plus - phi_c_minus), not the ladder's
        # algebraically-equal (psi - psi_minus)*T_e: the two differ in the last
        # bit, and only this form makes "a returned phi_c above the cap is
        # always tagged capability_limited" exactly true rather than true to
        # within a ULP.
        return psi * T_e - _emission_state(psi)[1] * T_e

    _PSI_LO = 1.0e-8
    phi_c_ceiling_V = phi_c_cap_V

    # The physical ceiling applies to the *net* sheath drop phi_c: in a deep
    # virtual cathode psi_c_plus legitimately exceeds any voltage-scale cap
    # while phi_c = (psi_plus - psi_minus)*T_e stays at bank scale (the
    # barrier eats the difference), and both psi_plus -> phi_c and
    # psi_plus -> J_tot are monotone increasing. So: extend the bracket top
    # geometrically until either the root is inside (f >= 0) or the net
    # sheath exceeds the cap. This is deterministic range extension on a
    # monotone function -- there is exactly one root and no branch to
    # mis-select.
    #
    # The ceiling is enforced on the RETURNED ROOT, not merely on the ladder's
    # grid points (fix 2026-08-09). The doubling grid only SAMPLES the cap
    # test, and in the virtual-cathode regime psi_minus > 0 puts the first grid
    # point psi = cap/T_e strictly below the cap in NET phi_c, so the ladder
    # always doubles at least once; the J-test is checked first at the doubled
    # point, so an imposed current reachable within that one doubling returned
    # a J-root whose net phi_c could be anything up to ~2x the cap -- above the
    # ceiling, tagged virtual_cathode, and (worst) INDEPENDENT of the cap, with
    # phi_c(I) non-monotone across the escape window. That non-monotonicity
    # violates the premise of the circuit's own brentq on V_dis(I). So the
    # J-root is tested against the cap after it is located, and a root at or
    # above the ceiling falls through to the ceiling branch below.
    psi_lo = _PSI_LO
    psi_top = max(phi_c_ceiling_V / T_e, Lambda + 2.0)
    # Compiled root find (Tier A, 2026-08-02). The ladder and brentq below
    # evaluate `_J_tot` / `_net_phi_c` ~50-100 times per solve, and each one is
    # a Python round-trip through `_emission_state` and its per-annulus loop.
    # On the annular branch with Schottky lowering the compiled unit runs the
    # identical ladder with the identical residual in C, using SciPy's own C
    # brentq (the same Zeros/brentq.c the Python `brentq` wraps, at the same
    # xtol/rtol/maxiter), and hands back the same `psi_c_plus`. Every other
    # branch, and the default pure path, runs the Python ladder.
    if _COMPILED_ROOT is not None and annular and schottky:
        psi_c_plus, capability_limited, _ = _COMPILED_ROOT(
            J_i, mu, Lambda, T_e, n_e,
            J_eth_k, delta_k, ion_frac_k,
            J_imposed, phi_c_ceiling_V, psi_lo, psi_top, _J_PLATEAU_TOL_REL,
        )
    else:
        capability_limited = False
        # Stage 1: the exact target. Well-conditioned operating points resolve
        # here.
        J_target = J_imposed
        for _ in range(200):
            if _J_tot(psi_top) >= J_target:
                break
            if _net_phi_c(psi_top) >= phi_c_ceiling_V:
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

        if not capability_limited:
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
            # The located root is where the ceiling is actually enforced. The
            # test is made AFTER the J-solve, on the SAME bracket the J-solve
            # used, precisely so that a root below the ceiling is returned bit
            # for bit as before: narrowing psi_top to the cap crossing up front
            # would have moved brentq's last bits on every virtual-cathode
            # solve, ceiling-bound or not.
            if _reported_phi_c(psi_c_plus) >= phi_c_ceiling_V:
                capability_limited = True

        if capability_limited:
            # A genuine inductive kick: the sheath cannot carry the imposed
            # current at physical net voltages. Return the solution *at* the
            # ceiling -- net phi_c = phi_c_ceiling_V, located by a bracketed
            # solve on the monotone net-sheath map so the reported kick
            # voltage does not depend on where the doubling happened to
            # land -- and let the circuit ramp I down at ~V/L per step.
            if _net_phi_c(psi_top) > phi_c_ceiling_V:
                psi_c_plus = brentq(
                    lambda x: _net_phi_c(x) - phi_c_ceiling_V,
                    psi_lo,
                    psi_top,
                    xtol=1.0e-12,
                    rtol=1.0e-14,
                    full_output=False,
                )
            else:
                psi_c_plus = psi_top

    # ------------------------------------------------------------------
    # Everything else follows explicitly from the solved psi
    # ------------------------------------------------------------------
    J_star, psi_c_minus, clamped = _emission_state(psi_c_plus)
    # The device carries the thermionic release and the net
    # ion/returning-electron current; this sum is the imposed loop current to
    # root-finder roundoff.
    J_tot = J_i * (1.0 - exp_clamped(Lambda - psi_c_plus)) + J_star
    regime = (
        "capability_limited"
        if capability_limited
        else ("virtual_cathode" if clamped else "classical")
    )

    phi_c_plus = psi_c_plus * T_e
    phi_c_minus = psi_c_minus * T_e
    phi_c = phi_c_plus - phi_c_minus
    # The one signature the ceiling forbids, on BOTH the pure and the compiled
    # root: a net sheath above the ceiling that is not tagged as sitting on it.
    # One comparison, and it covers the compiled path too because phi_c is
    # re-derived here from whichever root came back.
    if phi_c > phi_c_ceiling_V and regime != "capability_limited":
        raise RuntimeError(
            f"net phi_c={phi_c!r} V escaped the ceiling phi_c_ceiling_V="
            f"{phi_c_ceiling_V!r} V (phi_c_cap_V={phi_c_cap_V!r}) in regime "
            f"{regime!r} (psi_c_plus={psi_c_plus!r}, T_e={T_e!r}, "
            f"I_tot_A={I_tot_A!r})"
        )

    # Beam MFP and bypass: explicit evaluation at the solved sheath.
    l_b = compute_l_b(phi_c, T_e, n_e, plasma.n_n, plasma.sigma_b)
    beam_bypass_fraction = compute_beam_bypass_fraction(l_b, config.L_cath)
    long_mfp = l_b > 0.0 and l_b > config.L_cath

    J_anode = J_tot - eta * beam_bypass_fraction * J_star - J_tail_a
    # Anode floating potential: the anode's own sheath, on its own presheath.
    # phi_a = T_e,a ln(I_e,sat / I_e,a) with I_e,a = I_i,a + I_anode: the same
    # relation, with the saturation cap named instead of reached through
    # ``I_i_a e^Lambda_a``.
    psi_a = math.log(I_e_sat_a / max(I_i_a * (1.0 + J_anode / J_i_a), 1e-300))
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
        V_b = max(V_b, float(phi_c_ceiling_V))

    P_wall = I_tot * (V_b + I_tot * config.R_comp)
    P_load = I_tot * V_b
    P_comp = I_tot**2 * config.R_comp
    gap_survival = 1.0 - eta * beam_bypass_fraction
    P_prim = gap_survival * I_eth_star * phi_c
    P_ohmic = I_tot * V_p
    # Electron sheath powers with *physical flux barriers*:
    # (i) plasma electrons reaching the cathode surface climb the classical
    # peak phi_c_plus, not the net phi_c -- in a deep virtual cathode the
    # net can go slightly NEGATIVE while the barrier stays high, and the
    # historical exp(Lambda - phi_net/T) then explodes as exp(|phi_net|/T)
    # (measured: -180 kW of spurious heating into a 0.1 eV floor cell at
    # the first I~0 drive solve, detonating the fluid in one step);
    # (ii) an attracting electrode (phi < 0) collects at most electron
    # *saturation* -- the flux factor is capped at exp(Lambda). Both reduce
    # to ``I_i (2 T_e + phi) exp(Lambda - phi/T_e)`` in the classical
    # repelling regime (phi_minus = 0, phi_a >= 0).
    # Electron flux factors: the fraction of electron saturation actually
    # reaching each electrode across its repelling sheath. The plasma-thermal
    # (2Te) and sheath-fall (phi) parts ride the SAME flux, so each electrode
    # power splits cleanly into ``_thermal + _phi`` (R3.2 / A16 routing).
    fe_c = exp_clamped(Lambda - max(phi_c_plus, 0.0) / T_e)
    # The anode's collected electron current as a fraction of ``I_i_a``, so
    # the powers below keep their historical ``I_i_a * (...) * fe_a`` shape.
    # The CAP is the explicit saturation, not ``I_i_a e^Lambda_a``.
    fe_a = (
        (I_e_sat_a / I_i_a)
        * exp_clamped(-max(phi_a, 0.0) / T_e_anode)
    )
    # P_*_e / P_*_i keep their EXACT historical expressions (they feed the golden
    # via the fluid deposit and the cathode warming); the split derives the phi
    # part as the remainder so ``_thermal + _phi == P_*`` holds to machine zero.
    P_cathode_e = (
        I_i
        * (2.0 * T_e + phi_c)
        * fe_c
    )
    P_cathode_e_thermal = I_i * (2.0 * T_e) * fe_c
    P_cathode_e_phi = P_cathode_e - P_cathode_e_thermal
    P_cathode_i = P_ion(phi_c, T_e, I_i)
    P_cathode_i_thermal = I_i * (T_e / 2.0)
    P_cathode_i_phi = P_cathode_i - P_cathode_i_thermal
    P_cathode_i_pl = P_ion(phi_c, T_e, I_i_a, pl=True)
    P_anode_e = (
        I_i_a
        * (2.0 * T_e_anode + phi_a)
        * fe_a
    )
    P_anode_e_thermal = I_i_a * (2.0 * T_e_anode) * fe_a
    P_anode_e_phi = P_anode_e - P_anode_e_thermal
    # The QL tail's sheath-fall moment at the anode: the circuit pays
    # ``I_tail_a * phi_a`` for the current the wires take out of the walked
    # tail, exactly as the primary's bypass convention already pays for the
    # flux that streams through the mesh. Zero at a non-positive ``phi_a``.
    # ``I_tail_a`` is LAGGED -- the deposition is solved after the circuit
    # within a step, so this reads the previous accepted step's cull.
    P_tail_phi = max(phi_a, 0.0) * float(tail_anode_current_A)
    P_anode_i = P_ion(phi_a, T_e_anode, I_i_a)
    P_anode_i_thermal = I_i_a * (T_e_anode / 2.0)
    P_anode_i_phi = P_anode_i - P_anode_i_thermal
    P_anode_i_pl = P_ion(phi_a, T_e_anode, I_i_a, pl=True)
    _P_beam_bypass = eta * beam_bypass_fraction * I_eth_star * V_b
    # DEPRECATED unclosed scalars (kept bit-exact for the R1-R4 golden only).
    # NO LONGER EXPORTED to the HDF5; the successors are the closed audit built
    # just below. P_cathode_i_pl -> P_cathode_i_thermal (it is the ANODE ion
    # current under a cathode name), P_anode_i_pl -> P_anode_i_thermal,
    # P_loss -> P_plasma_thermal_loss (P_loss mixes phi-inclusive electron
    # powers with thermal-only ion powers), P_net -> P_into_plasma or
    # P_load_residual (P_net books each sheath fall twice), P_net2 ->
    # P_into_plasma, P_comp -> nothing (it is I_tot**2 * R_comp, never read).
    # See the SolverResult field block in circuit_common.py for the full
    # statement.
    P_net = (
        P_load - P_cathode_e - P_cathode_i - P_anode_e - P_anode_i
        - _P_beam_bypass
    )
    P_net2 = (
        P_prim + P_ohmic - P_cathode_e - P_cathode_i_pl - P_anode_e
        - P_anode_i_pl
    )
    P_loss = P_cathode_e + P_cathode_i_pl + P_anode_e + P_anode_i_pl
    # Closed surface-resolved audit (replaces P_net/P_net2). Only the PLASMA-
    # THERMAL parts leave the plasma thermal store; the phi parts are sheath-field
    # energy deposited on the electrodes. The end wall (floating) exhaust is
    # booked separately on the fluid side (no circuit branch).
    P_plasma_thermal_loss = (
        P_cathode_e_thermal + P_cathode_i_thermal
        + P_anode_e_thermal + P_anode_i_thermal
    )
    P_into_plasma = P_prim + P_ohmic - P_plasma_thermal_loss
    P_cathode_surface = P_cathode_e + P_cathode_i
    P_anode_surface = P_anode_e + P_anode_i
    # Measurement-plane aliases (see SolverResult): keep I_tot / V_b (Poulos), and
    # alias to the three-plane convention with the item-24/25 divergences pinned
    # to zero so P_load = V_b*I_tot = V_dis*I_bank today.
    # Internal series drop on the plasma side of the V_dis probe (R5 ES1 tuning
    # pass, 2026-07-26): R_internal = (1-x)*R_comp plus the separate anode mesh
    # R_mesh_ohm. The device voltage the circuit integrates is V_b + I*R_internal;
    # the measured V_dis = V_bank - I*(x*R_comp) uses the external part only, so
    # the internal drop is invisible to the V_dis formula but lowers the current
    # (raising V_dis). Defaults (x=1, R_mesh=0) -> V_series = 0, bit-exact.
    V_series = I_tot * (
        (1.0 - config.R_comp_partition) * config.R_comp + config.R_mesh_ohm
    )
    I_parallel = 0.0
    I_plasma = I_tot
    I_bank = I_plasma + I_parallel
    V_dis = V_b + V_series
    # Load-power closure (current-resolved; see SolverResult). Per-region net
    # field work I_tot*drop, with the cathode decomposed per species so the
    # returning plasma-electron current recovers energy (minus sign):
    #   cathode:  I_eth_star*phi_c + P_cathode_i_phi - P_cathode_e_phi  (= I_tot*phi_c)
    #   gap:      P_ohmic  (= I_tot*V_p)
    #   anode:    I_tot*phi_a  (net; the model's per-species anode terms carry only
    #             the Bohm ion current, not the full loop current -- A15 anode
    #             interception is R4 -- so the anode region uses the ladder value).
    # I_e_ret = P_cathode_e_phi / phi_c is the returning-electron current; the
    # cathode Kirchhoff (I_eth_star + I_i - I_e_ret == I_tot) is the real check.
    I_e_ret = P_cathode_e_phi / phi_c if phi_c != 0.0 else 0.0
    cathode_field_work = I_eth_star * phi_c + P_cathode_i_phi - P_cathode_e_phi
    P_load_ledger = cathode_field_work + P_ohmic - I_tot * phi_a
    P_load_residual = P_load - P_load_ledger
    I_cathode_kirchhoff_residual = (
        I_eth_star + I_i - I_e_ret
    ) - I_tot

    # Active-bound census (see SolverResult): whether this solve ended up
    # sitting on the ceiling, derived from the regime tag.
    bound_active = 1.0 if capability_limited else 0.0

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
        P_cathode_e_thermal=P_cathode_e_thermal,
        P_cathode_e_phi=P_cathode_e_phi,
        P_cathode_i_thermal=P_cathode_i_thermal,
        P_cathode_i_phi=P_cathode_i_phi,
        P_anode_e_thermal=P_anode_e_thermal,
        P_anode_e_phi=P_anode_e_phi,
        P_tail_phi=P_tail_phi,
        P_anode_i_thermal=P_anode_i_thermal,
        P_anode_i_phi=P_anode_i_phi,
        P_plasma_thermal_loss=P_plasma_thermal_loss,
        P_into_plasma=P_into_plasma,
        P_cathode_surface=P_cathode_surface,
        P_anode_surface=P_anode_surface,
        V_series=V_series,
        I_parallel=I_parallel,
        V_dis=V_dis,
        I_plasma=I_plasma,
        I_bank=I_bank,
        I_e_ret=I_e_ret,
        P_load_ledger=P_load_ledger,
        P_load_residual=P_load_residual,
        I_cathode_kirchhoff_residual=I_cathode_kirchhoff_residual,
        phi_c_ceiling_V=phi_c_ceiling_V,
        bound_active=bound_active,
        regime=regime,
        long_mfp=long_mfp,
        beam_bypass_fraction=beam_bypass_fraction,
        l_b=l_b,
        T_e_anode=T_e_anode,
    )


def solve_beam_system_idriven(
    config: DeviceConfig,
    Te: np.ndarray,
    ne: np.ndarray,
    nn: np.ndarray,
    beam_cross_prev: np.ndarray,
    plasma_cross: np.ndarray,
    I_ion: float,
    I_tot_A: float,
    cathode_index: int = 0,
    anode_current_A: float | None = None,
    anode_T_e: float | None = None,
    schottky: bool = False,
    phi_c_cap_V: float = 1000.0,
    alpha_sheath: float | None = None,
    alpha_sheath_anode: float | None = None,
    anode_electron_saturation_A: float | None = None,
    tail_anode_current_A: float = 0.0,
) -> BeamResult:
    """Current-driven, single-cathode sheath solve plus its beam arrays.

    Calls ``solve_idriven`` for the primary cathode and hands the result to
    :func:`assemble_beam_arrays`. Twin cathodes are out of scope for the
    current-driven path (the dispatcher raises before this is reached), so
    ``result_twin`` is always ``None`` and the twin arrays stay zero.
    """
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
        alpha_sheath=alpha_sheath,
        alpha_sheath_anode=alpha_sheath_anode,
        anode_electron_saturation_A=anode_electron_saturation_A,
        tail_anode_current_A=tail_anode_current_A,
    )
    return assemble_beam_arrays(
        result=result,
        config=config,
        Te=Te,
        ne=ne,
        nn=nn,
        plasma_cross=plasma_cross,
        I_ion=I_ion,
        cathode_index=cathode_index,
    )


def assemble_beam_arrays(
    result,
    config: DeviceConfig,
    Te: np.ndarray,
    ne: np.ndarray,
    nn: np.ndarray,
    plasma_cross: np.ndarray,
    I_ion: float,
    cathode_index: int = 0,
) -> BeamResult:
    """Wrap a solved single-cathode sheath in the per-cell beam arrays.

    The second half of every single-cathode beam system: given the sheath
    ``result``, launch the thermionic beam at its ``phi_c`` and fill the
    per-cell velocity / density / cross-section arrays the fluid reads.
    Extracted so the CURRENT-DRIVEN and the PRESCRIBED-MEASURED beam systems
    share one assembly rather than two copies of it -- the two differ only in
    how the sheath was solved, and a second copy of this would be free to
    drift away from the first.

    Twin cathodes are out of scope for both callers, so ``result_twin`` is
    always ``None`` and the twin arrays stay zero. ``x0_next`` carries the
    solved ``phi_c_plus``; neither caller needs a warm start.
    """
    cells = len(Te)
    v_beam = np.zeros(cells)
    n_beam = np.zeros(cells)
    beam_cross = np.zeros(cells)

    phi_c_0 = result.phi_c
    if phi_c_0 > I_ion:
        v_beam[cathode_index] = math.sqrt(2.0 * phi_c_0 * ERG_PER_EV / ME_CGS)
        _I_beam_0 = beam_launched_current_A(result) * (
            1.0 - config.eta * result.beam_bypass_fraction
        )
        n_beam[cathode_index] = _I_beam_0 / (
            E_SI * plasma_cross[cathode_index] * v_beam[cathode_index]
        )
        # The tabulated He EII cross section ends at eps = E/I_ion =
        # HE_EII_EPS_TOP, and the lookup CLAMPS to its last node above
        # that. On a capability-limited step the beam energy is the sheath
        # ceiling, which at the shipped cap (``cathode_phi_c_cap_V``) sits
        # on the table's last node to within a ULP -- so the edge is
        # INCLUSIVE within HE_EII_EDGE_REL_TOL, exactly as the tail walk's
        # guard has it (K7c): at the edge the clamped value IS the
        # endpoint node and nothing is extrapolated. A larger excess is
        # refused rather than silently clamped, which is what this call
        # did before. Since the sheath root is now capped, reaching the
        # refusal requires a cap configured above the table top.
        _beam_eps = phi_c_0 / I_ion
        _beam_edge_excess = (
            _beam_eps - HE_EII_EPS_TOP
        ) / HE_EII_EPS_TOP
        if _beam_edge_excess > HE_EII_EDGE_REL_TOL:
            raise ValueError(
                "the beam ionization cross section is read from the "
                "tabulated He EII data, which ends at eps = E/I_ion = "
                f"{HE_EII_EPS_TOP:.6f} (i.e. "
                f"{HE_EII_EPS_TOP * I_ion:.2f} eV at I_ion={I_ion}); at "
                f"phi_c={phi_c_0} V the lookup would clamp to its last "
                "node and the beam would deposit on an extrapolated cross "
                "section. This is refused, not approximated (relative "
                f"excess {_beam_edge_excess:.3e}, tolerated "
                f"{HE_EII_EDGE_REL_TOL:.1e}); lower "
                "cathode_phi_c_cap_V to the table top or below"
            )
        beam_cross[cathode_index] = He_EII_cross_lkup(_beam_eps)

    # A separate array: the caller overwrites its launch cell with the
    # effective attenuation cross section it feeds back to the next solve,
    # and ``beam_cross`` must keep the ionization cross section.
    beam_atten_cross = beam_cross.copy()
    n_beam_ion = n_beam * beam_cross * v_beam
    A_ion_beam = n_beam_ion * nn

    l_b = np.zeros(cells)
    p_beam = np.zeros(cells)
    if beam_cross[cathode_index] != 0.0:
        l_b[cathode_index] = result.l_b
        p_beam[cathode_index] = (
            l_b[cathode_index] * beam_cross[cathode_index] * nn[cathode_index]
        )

    return BeamResult(
        result=result,
        result_twin=None,
        v_beam=v_beam,
        n_beam=n_beam,
        beam_cross=beam_cross,
        beam_atten_cross=beam_atten_cross,
        n_beam_ion=n_beam_ion,
        A_ion_beam=A_ion_beam,
        l_b=l_b,
        p_beam=p_beam,
        l_b_profile=np.zeros(cells),
        l_b_profile_twin=np.zeros(cells),
        x0_next=result.phi_c_plus,
        x0_twin_next=None,
    )
