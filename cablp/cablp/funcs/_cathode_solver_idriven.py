"""Current-driven cathode sheath solve (M2).

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
is not modified (a hard constraint of this design): its solve paths remain
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
  (``phi_c_cap_V``, optionally composed with ``circuit_V_avail_V``), i.e.
  a genuine inductive kick. The bracket-top
  solution is returned with its correspondingly large ``V_b`` and the
  circuit is expected to ramp the current down at ~V/L per step. No
  exception, no fallback ladder. With the circuit bound in force the
  REPORTED kick is limited to the loop's available voltage, but the ramp is
  unaffected: the circuit integrates the unbounded demand, so the current
  still ramps down rather than freezing. See ``circuit_V_avail_V`` in
  ``solve_idriven`` and ``cathode.idriven_vdis_evaluator``.

Floating (open-circuit) solves keep using ``_cathode_solver.solve`` -- its
floating branch models Boltzmann-suppressed emission over the virtual
barrier, which is not the same limit as ``I_tot = 0`` through the hard
space-charge clamp here; routing is the M3 dispatcher's job.

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

from cablp.funcs._beam_deposition import (
    HE_EII_EDGE_REL_TOL,
    HE_EII_EPS_TOP,
)
from cablp.funcs._cathode_solver import (
    CATHODE_LNL_MODELS,
    BeamResult,
    DeviceConfig,
    PlasmaState,
    SolverResult,
    _LN_LAMBDA_MIN,
    _P_elec,
    _P_ion,
    _annular_emission_state,
    _c_log_ei,
    _compute_beam_bypass_fraction,
    _compute_l_b,
    _e_SI,
    _erg_per_eV,
    _exp_clamped,
    _j_eth_crit,
    _kB_SI,
    _me_cgs,
    _mp_cgs,
    beam_excitation_channel,
)
from cablp.funcs._cross import H_EII_cross_lkup, He_EII_cross_lkup
from cablp.funcs._kernels import COMPILED_KERNELS as _COMPILED_KERNELS

__all__ = [
    "beam_launch_energy_eV",
    "solve_idriven",
    "solve_beam_system_idriven",
]

#: Quantities ``circuit_bound_object`` may name as the circuit bound's object.
#: ``"phi_c"`` bounds the net cathode drop (the historical composition);
#: ``"device_voltage"`` bounds ``V_b = phi_c - phi_a + V_p``, the quantity the
#: loop equation contains.
_CIRCUIT_BOUND_OBJECTS = ("phi_c", "device_voltage")

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

# Thermal-bridge half-width, in units of kT_s of would-be barrier depth
# (chatter diagnosis, 2026-07-21). The emitted
# Maxwellian has energy spread kT_s, so the SCL<->classical release corner
# is physically smooth over ~kT_s of barrier -- the transition variable is
# x = ln(J_eff/J_crit) = (would-be barrier)/kT_s, and the corner max(0, x)
# is blended over |x| < w. Fixed at 1 (the physical scale), deliberately
# not exposed as a tuning key.
_BRIDGE_HALF_WIDTH = 1.0


def _bridge_release(J_eff: float, J_crit: float) -> tuple[float, float]:
    """Smooth SCL<->classical release: ``(J_star, barrier/kT_s)``.

    Replaces the razor corner ``J_star = min(J_eff, J_crit)`` /
    ``psi_minus = delta * max(0, ln(J_eff/J_crit))`` with a C1 blend of
    ``max(0, x)`` over the window ``|x| < w`` in ``x = ln(J_eff/J_crit)``:

        b(x) = 0            (x <= -w, exact classical branch)
             = (x+w)^2/4w   (|x| < w, quadratic bridge)
             = x            (x >= +w, exact space-charge branch)

    with ``J_star = J_eff * exp(-b)``. Properties (asserted in smoke):
    exact hard-branch reduction outside the window, so calibrated
    operating points away from the knee are untouched; ``J_star <=
    min(J_eff, J_crit)`` everywhere (the blend only ever *adds* barrier);
    and monotonicity of J_tot(psi) is preserved analytically -- writing
    a = dlnJ_eff/dpsi >= 0, c = dlnJ_crit/dpsi >= 0, the blended slope is
    dlnJ_star/dpsi = (1-s)*a + s*c with s = b'(x) in [0, 1], a convex
    combination of the two (nonnegative) branch slopes. The architecture
    rests on that monotonicity.

    Returns the barrier in kT_s units (``psi_minus = delta * b``); the
    caller owns the delta scaling because annuli carry per-annulus deltas.
    """
    if J_eff <= 0.0:
        return 0.0, 0.0
    if J_crit <= 0.0:
        # Fully choked channel (mirrors the hard branches' J_crit <= 0
        # handling: no current, no reported barrier).
        return 0.0, 0.0
    w = _BRIDGE_HALF_WIDTH
    x = math.log(J_eff / J_crit)
    if x <= -w:
        return J_eff, 0.0
    if x >= w:
        return J_crit, x
    b = (x + w) ** 2 / (4.0 * w)
    return J_eff * math.exp(-b), b


def _uniform_state_bridge(
    psi: float,
    J_i: float,
    J_eth: float,
    mu: float,
    delta: float,
    T_e: float,
    n_e: float,
    schottky: bool,
) -> tuple[float, float, bool]:
    """Uniform-disc ``(J_star, psi_minus, clamped)`` with the thermal bridge.

    With Schottky on, the enhanced ``J_eff`` feeds the bridge directly and
    the three-branch choke rule collapses into the smooth kernel: deep in
    the clamp ``J_star -> J_crit`` becomes insensitive to ``J_eff``, so the
    field enhancement self-attenuates without an explicit surface-field
    cutoff. (Divergence from the hard Schottky branches, documented: the
    deep-clamp barrier is referenced to ``J_eff`` rather than ``J_eth`` --
    a ~dphi/T_e difference in the reported psi_minus, order 0.1 V.)
    """
    if J_eth <= 0.0:
        return 0.0, 0.0, False
    J_eff = J_eth
    if schottky:
        dphi = _schottky_lowering_eV(psi * T_e, T_e, n_e)
        if dphi > 0.0:
            J_eff = J_eth * math.exp(dphi / (delta * T_e))
    J_crit = _j_eth_crit(psi, J_i, mu)
    if J_crit <= 0.0:
        return 0.0, 0.0, True
    J_star, b = _bridge_release(J_eff, J_crit)
    return J_star, delta * b, J_eff > J_crit


def _annular_state_bridge(
    psi: float,
    J_i: float,
    mu: float,
    J_eth_k: tuple,
    delta_k: tuple,
    ion_frac_k: tuple,
    T_e: float,
    n_e: float,
    schottky: bool,
) -> tuple[float, float, bool]:
    """Annular ``(J_star, psi_minus_eff, any_clamped)`` with the bridge.

    Same equipotential-annuli structure as ``_annular_emission_state``;
    each annulus passes through the smooth kernel with its own delta, and
    the effective barrier is the emission-weighted mean over *all* annuli
    carrying a (possibly partial, in-window) barrier -- the hard model's
    weighting with the clamped-only restriction lifted, to which it
    reduces when every annulus sits outside its bridge window.
    """
    dphi = _schottky_lowering_eV(psi * T_e, T_e, n_e) if schottky else 0.0
    J_star_total = 0.0
    weighted_pm = 0.0
    any_clamped = False
    for J_eth_a, delta_a, frac_a in zip(J_eth_k, delta_k, ion_frac_k):
        if J_eth_a <= 0.0:
            continue
        J_crit_a = _j_eth_crit(psi, J_i * frac_a, mu) if frac_a > 0.0 else 0.0
        J_eff_a = J_eth_a
        if dphi > 0.0:
            J_eff_a = J_eth_a * math.exp(dphi / (delta_a * T_e))
        if J_crit_a <= 0.0:
            any_clamped = True
            continue
        if J_eff_a > J_crit_a:
            any_clamped = True
        J_star_a, b_a = _bridge_release(J_eff_a, J_crit_a)
        J_star_total += J_star_a
        weighted_pm += J_star_a * delta_a * b_a
    psi_minus_eff = weighted_pm / J_star_total if J_star_total > 0.0 else 0.0
    return J_star_total, psi_minus_eff, any_clamped


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


# The pure-Python kernels stay reachable under their own names so the compiled
# path can be compared against them (equivalence sweeps, microbenchmarks)
# inside a process that has opted in.
_schottky_lowering_eV_pure = _schottky_lowering_eV
_annular_state_schottky_pure = _annular_state_schottky

# Compiled-kernel selection (Tier A, 2026-08-02). Same contract as
# ``_cathode_solver``'s block: one rebinding site per name, at module scope,
# before any caller resolves it, so the hot path is a plain function object
# with no per-call branch. ``_COMPILED_ROOT`` is the whole root find for the
# PRODUCTION branch (annular + Schottky, no thermal bridge) -- the bracket
# ladder plus brentq with the residual evaluated in C, which is what removes
# the ~50-100 Python round-trips per solve. It is ``None`` on every other
# branch and on the default pure path, and ``solve_idriven`` then runs the
# historical Python ladder verbatim.
_COMPILED_ROOT = None
if _COMPILED_KERNELS is not None:
    _COMPILED_KERNELS.check_constants_idriven(_SCHOTTKY_EV_PER_SQRT_V_M)
    _schottky_lowering_eV = _COMPILED_KERNELS.schottky_lowering_eV
    _annular_state_schottky = _COMPILED_KERNELS.annular_state_schottky
    _COMPILED_ROOT = _COMPILED_KERNELS.solve_psi_annular_schottky


def beam_launch_energy_eV(phi_c, climb_V):
    """Return the energy [eV] the beam carries INTO the column.

    ``phi_c`` is the solved net cathode drop -- under the circuit voltage
    bound, the bounded one. ``climb_V`` is the potential [V] the transmitted
    beam must climb between the anode mesh and the column; ``None`` means no
    such step exists and the launch energy IS ``phi_c``, returned as the same
    object so every downstream consumer is bit-for-bit unchanged. A climb is
    only ever a decelerating step here, so a negative value contributes
    nothing, and the result is floored at zero (a fully choked beam).

    The single definition both beam readers use -- the Beer-Lambert beam-array
    assembly in :func:`solve_beam_system_idriven` and the CSDA deposition
    rays -- so a build cannot end up with two launch energies.
    """
    if climb_V is None:
        return phi_c
    return max(float(phi_c) - max(float(climb_V), 0.0), 0.0)


def solve_idriven(
    config: DeviceConfig,
    plasma: PlasmaState,
    I_tot_A: float,
    cathode_current_A: float | None = None,
    anode_current_A: float | None = None,
    anode_T_e: float | None = None,
    schottky: bool = False,
    bridge: bool = False,
    phi_c_cap_V: float = 1000.0,
    alpha_sheath: float | None = None,
    alpha_sheath_anode: float | None = None,
    circuit_V_avail_V: float | None = None,
    circuit_bound_object: str = "phi_c",
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
    ``circuit_V_avail_V`` is the optional CIRCUIT-AVAILABLE device voltage
    [V] -- the largest device voltage the external loop can sustain at this
    current, ``V_src - I*(R_comp + R_mesh)``, which is the loop equation
    ``L dI/dt = V_src - I*(R_comp + R_mesh) - V_b`` read at ``dI/dt = 0``.
    ``None`` (the default) leaves the solve bit-for-bit as it was. Given, it
    composes with ``phi_c_cap_V`` as an upper bound: the ceiling the sheath
    root is solved against becomes ``min(phi_c_cap_V, circuit_V_avail_V)``,
    so the returned ``phi_c`` -- and every consumer keyed to it, notably the
    beam birth energy through ``_compute_l_b`` -- cannot exceed what the
    circuit supplies, and the capability-limited device voltage ``V_b`` is
    clamped to that same available voltage. Must be positive: a loop with no
    available voltage has no ceiling to offer and the caller passes ``None``
    there instead. NB the inductor's back-EMF is deliberately NOT counted as
    available voltage: it is stored energy, not supply.

    That exclusion no longer freezes the loop current (corrected 2026-08-12).
    It once did, because the circuit integrated this same BOUNDED ``V_b``:
    the loop residual was then identically zero wherever the bound bound, so
    ``dI/dt >= 0`` everywhere and the current could only ratchet upward on
    numerical overshoot. The circuit now integrates the sheath's UNBOUNDED
    demand (``cathode.idriven_vdis_evaluator``), so the restoring force is
    present on both sides of the capability wall and the current is free to
    fall while the bound binds. The bound constrains the sheath and beam
    objects only.

    ``circuit_bound_object`` selects WHICH quantity the available voltage
    bounds, and is read only when ``circuit_V_avail_V`` is given:

    - ``"device_voltage"`` -- the bound's object is the DEVICE voltage
      ``V_b = phi_c - phi_a + V_p``, the quantity the loop equation actually
      contains. The circuit member of the composed ceiling is the *net sheath
      drop at which* ``V_b(psi) = circuit_V_avail_V``, located by a bracketed
      solve on the same monotone device relation the current root uses, with
      ``phi_a`` and ``V_p`` evaluated by the identical expressions that
      assemble the returned result. The composition, the ladder, the escape
      invariant and the ``bound_active`` census are unchanged -- only the
      number the circuit contributes to ``min`` changes -- so the compiled
      root path is used exactly as before.
    - ``"phi_c"`` -- the circuit member IS ``circuit_V_avail_V``, i.e. the
      bound's object is the net cathode drop. Bit-for-bit the historical
      composition.

    The two coincide only where ``phi_a`` and ``V_p`` are negligible -- the
    capability-limited / near-vacuum regime of the pre-breakdown build leg. At
    the main-discharge plateau ``phi_a`` is not negligible, ``phi_c``
    legitimately exceeds the available voltage while ``V_b`` does not, and
    ``"phi_c"`` there clamps a correct solve and tags it
    ``capability_limited`` with no error raised (only ``bound_active`` records
    it). ``"device_voltage"`` cannot make that error. NB the object is
    independent of the back-EMF exclusion above, which no longer costs the
    falling leg anything: the bound holds the exported ``V_b`` at the
    available voltage, but the circuit integrates the unbounded demand, so a
    bound solve on a decaying current reports a clamped ``V_b`` while
    ``dI/dt`` stays free.
    ``bridge`` enables the kT_s-width thermal bridge across the
    SCL<->classical release corner (``_bridge_release``); off reproduces
    the hard branches bit-for-bit (the M2 equivalence gate's condition).

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
    # Validate the bound's inputs here, at the top, so a misconfigured call
    # fails before it spends a solve. The COMPOSITION itself happens further
    # down, once the device relation the "device_voltage" object is located on
    # exists; off (``None``) the composed ceiling is ``phi_c_cap_V``, the same
    # float object, and every comparison and bracket below is bit-for-bit the
    # historical one.
    if circuit_V_avail_V is not None:
        circuit_V_avail_V = float(circuit_V_avail_V)
        if not (circuit_V_avail_V > 0.0) or not math.isfinite(
            circuit_V_avail_V
        ):
            raise ValueError(
                "circuit_V_avail_V must be finite and positive when the "
                f"circuit voltage bound is in force (got {circuit_V_avail_V})"
            )
        if circuit_bound_object not in _CIRCUIT_BOUND_OBJECTS:
            raise ValueError(
                "circuit_bound_object must be one of "
                f"{sorted(_CIRCUIT_BOUND_OBJECTS)} when the circuit voltage "
                f"bound is in force (got {circuit_bound_object!r})"
            )

    T_e = plasma.T_e
    n_e = plasma.n_e

    # ------------------------------------------------------------------
    # Plasma-derived quantities: identical formulas to the voltage-driven
    # solve, so the two solvers agree bit-for-bit on the operating map.
    # ------------------------------------------------------------------
    # Parallel plasma conductivity [Ω⁻¹ cm⁻¹]. Spitzer, with the Coulomb
    # logarithm evaluated at the solve's own state rather than frozen: NRL
    # Formulary 2004 p.30 gives the TRANSVERSE resistivity
    # eta_perp = 1.03e-2 Z lnLambda T_e^-3/2 [Ohm cm], and p.38 gives
    # sigma_par = 1.96 sigma_perp at Z = 1 (Braginskii). The two literature
    # factors are left un-collapsed so the lineage stays readable. lnLambda is
    # floored at _LN_LAMBDA_MIN, the same floor the transport terms use -- it
    # is a positivity guard for the cold, tenuous corner and does not bind at
    # any physical discharge state.
    ln_lambda = max(_c_log_ei(T_e, n_e), _LN_LAMBDA_MIN)
    if config.lnL_model == "nrl_ei":
        sigma_par = (1.96 / (1.03e-2 * ln_lambda)) * T_e**1.5
    elif config.lnL_model == "fixed_14p6":
        sigma_par = 14.6 * T_e**1.5
    else:
        raise ValueError(
            "lnL_model must be one of "
            f"{CATHODE_LNL_MODELS} (got {config.lnL_model!r})"
        )
    R_p = config.L_cath / (math.pi * config.R_cath**2 * sigma_par)
    C_s = math.sqrt(T_e * _e_SI * 1.0e7 / (config.mu * _mp_cgs))
    # Sheath-edge sampling (R3.2 / A16): the ion Bohm
    # current is drawn at the sheath-edge density n_se = alpha_sheath * n_e. The
    # historical flat exp(-1/2) is the Boltzmann drop across a presheath that
    # fits inside the cell; the fluid boundary instead uses the mesh-independent
    # ``sources.presheath_alpha``. R3.2 lets the caller pass that SAME factor so
    # the circuit current and the fluid sink read one n_se. Electron saturation
    # stays at the bulk density (the ``lam_shift`` that lifts electrons back to
    # n_e is -ln(alpha_sheath), = +0.5 for the flat default). ``None`` keeps the
    # exact +0.5 so the golden and the M2 equivalence gate are bit-exact.
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
    I_i = config.A_c * _e_SI * n_e * C_s * alpha_eff
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
    delta = _kB_SI * config.T_s / (_e_SI * T_e)
    Lambda = config.Lambda + lam_shift
    Lambda_anode = config.Lambda + lam_shift_anode
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
        if config.emission_area_fraction != 1.0:
            # See the same site in ``_cathode_solver.solve``: the lit patches
            # collect the Bohm flux over their own share of the face, so the
            # attribution sums to f_em, and this division is what keeps the
            # scaled plasma fractions from being normalized straight back out.
            wetted /= config.emission_area_fraction
        ion_frac_k = tuple(
            (a * f / wetted) if wetted > 0.0 else 0.0
            for a, f in zip(config.emission_area_cm2, config.emission_plasma_frac)
        )

    # ------------------------------------------------------------------
    # The monotone device relation and its single bracketed root
    # ------------------------------------------------------------------
    def _emission_state(psi: float) -> tuple[float, float, bool]:
        if bridge:
            if annular:
                return _annular_state_bridge(
                    psi, J_i, mu, J_eth_k, delta_k, ion_frac_k,
                    T_e, n_e, schottky,
                )
            return _uniform_state_bridge(
                psi, J_i, J_eth, mu, delta, T_e, n_e, schottky
            )
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

    def _reported_phi_c(psi: float) -> float:
        # The ceiling test on the located root uses the arithmetic the result
        # is ASSEMBLED with below (phi_c_plus - phi_c_minus), not the ladder's
        # algebraically-equal (psi - psi_minus)*T_e: the two differ in the last
        # bit, and only this form makes "a returned phi_c above the cap is
        # always tagged capability_limited" exactly true rather than true to
        # within a ULP.
        return psi * T_e - _emission_state(psi)[1] * T_e

    def _device_voltage(psi: float) -> float:
        # V_b(psi) = phi_c + V_p - phi_a, assembled by the SAME expressions the
        # returned result is built from below (see the "Everything else follows
        # explicitly from the solved psi" block) so the bound's object and the
        # reported object cannot drift apart. Monotone increasing in psi on the
        # same premises the current root already rests on: phi_c and J_tot both
        # rise with psi, and phi_a falls as the anode collects more.
        J_star_p, psi_minus_p, _ = _emission_state(psi)
        J_tot_p = J_i * (1.0 - _exp_clamped(Lambda - psi)) + J_star_p
        phi_c_p = psi * T_e - psi_minus_p * T_e
        l_b_p = _compute_l_b(phi_c_p, T_e, n_e, plasma.n_n, plasma.sigma_b)
        bypass_p = _compute_beam_bypass_fraction(l_b_p, config.L_cath)
        J_anode_p = J_tot_p - eta * bypass_p * J_star_p
        psi_a_p = Lambda_anode - math.log(
            max(1.0 + J_anode_p / J_i_a, 1e-300)
        )
        I_tot_p = J_tot_p * T_e / R_p
        return phi_c_p + I_tot_p * R_p - psi_a_p * T_e_anode

    # ------------------------------------------------------------------
    # Composed ceiling (see the circuit_V_avail_V / circuit_bound_object
    # contract in the docstring). With no circuit bound this IS
    # ``phi_c_cap_V``, the same float object.
    # ------------------------------------------------------------------
    _PSI_LO = 1.0e-8
    if circuit_V_avail_V is None:
        phi_c_ceiling_V = phi_c_cap_V
        _ceiling_is_circuit = False
    else:
        if circuit_bound_object == "phi_c":
            _circuit_phi_c_ceiling_V = circuit_V_avail_V
        else:
            # The net sheath drop at which the DEVICE voltage reaches the
            # available voltage. Same deterministic range extension on a
            # monotone function the current root uses: grow the bracket top
            # until either V_b has reached the target or the net sheath has
            # already passed the data cap -- past which the cap is the binding
            # member and what the circuit would have contributed is moot.
            _psi_hi = max(phi_c_cap_V / T_e, Lambda + 2.0)
            _reached = _device_voltage(_psi_hi) >= circuit_V_avail_V
            for _ in range(200):
                if _reached or _net_phi_c(_psi_hi) >= phi_c_cap_V:
                    break
                _psi_hi *= 2.0
                _reached = _device_voltage(_psi_hi) >= circuit_V_avail_V
            if not _reached:
                # The device cannot produce the available voltage below the
                # data cap, so the circuit offers no ceiling at all.
                _circuit_phi_c_ceiling_V = math.inf
            elif _device_voltage(_PSI_LO) >= circuit_V_avail_V:
                # Degenerate: the loop cannot sustain even the smallest sheath.
                # The bracket bottom IS the ceiling; keep it strictly positive
                # so the escape invariant's comparison stays well-defined.
                _circuit_phi_c_ceiling_V = _reported_phi_c(_PSI_LO)
            else:
                _circuit_phi_c_ceiling_V = _reported_phi_c(
                    brentq(
                        lambda x: _device_voltage(x) - circuit_V_avail_V,
                        _PSI_LO,
                        _psi_hi,
                        xtol=1.0e-12,
                        rtol=1.0e-14,
                        full_output=False,
                    )
                )
        _ceiling_is_circuit = _circuit_phi_c_ceiling_V < phi_c_cap_V
        phi_c_ceiling_V = (
            _circuit_phi_c_ceiling_V if _ceiling_is_circuit else phi_c_cap_V
        )

    # The physical ceiling applies to the *net* sheath drop phi_c: in a deep
    # virtual cathode psi_c_plus legitimately exceeds any voltage-scale cap
    # while phi_c = (psi_plus - psi_minus)*T_e stays at bank scale (the
    # barrier eats the difference), and both psi_plus -> phi_c and
    # psi_plus -> J_tot are monotone increasing. So: extend the bracket top
    # geometrically until either the root is inside (f >= 0) or the net
    # sheath exceeds the cap. This is deterministic range extension on a
    # monotone function -- there is exactly one root and no branch to
    # mis-select -- not the voltage-driven path's root-hunting ladder.
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
    # On the PRODUCTION branch -- annular emission with Schottky lowering and
    # no thermal bridge -- the compiled unit runs the identical ladder with the
    # identical residual in C, using SciPy's own C brentq (the same
    # Zeros/brentq.c the Python `brentq` wraps, at the same xtol/rtol/maxiter),
    # and hands back the same `psi_c_plus`. Every other branch, and the default
    # pure path, falls through to the Python ladder unchanged.
    if _COMPILED_ROOT is not None and annular and schottky and not bridge:
        psi_c_plus, capability_limited, _ = _COMPILED_ROOT(
            J_i, mu, Lambda, T_e, n_e,
            J_eth_k, delta_k, ion_frac_k,
            J_imposed, phi_c_ceiling_V, psi_lo, psi_top, _J_PLATEAU_TOL_REL,
        )
    else:
        capability_limited = False
        # Stage 1: the exact target. Well-conditioned operating points resolve
        # here and the equivalence with the voltage-driven solve is pristine.
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
    J_tot = J_i * (1.0 - _exp_clamped(Lambda - psi_c_plus)) + J_star
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
    # re-derived here from whichever root came back. The ceiling tested is the
    # COMPOSED one, so the invariant covers the circuit bound as well as the
    # data cap the moment that bound is in force.
    if phi_c > phi_c_ceiling_V and regime != "capability_limited":
        raise RuntimeError(
            f"net phi_c={phi_c!r} V escaped the ceiling phi_c_ceiling_V="
            f"{phi_c_ceiling_V!r} V (phi_c_cap_V={phi_c_cap_V!r}, "
            f"circuit_V_avail_V={circuit_V_avail_V!r}) in regime {regime!r} "
            f"(psi_c_plus={psi_c_plus!r}, T_e={T_e!r}, I_tot_A={I_tot_A!r})"
        )

    # Beam MFP and bypass: explicit evaluation at the solved sheath (the
    # voltage-driven path needs a fixed-point loop here only because its
    # residual feeds bypass back into the root equation).
    l_b = _compute_l_b(phi_c, T_e, n_e, plasma.n_n, plasma.sigma_b)
    beam_bypass_fraction = _compute_beam_bypass_fraction(l_b, config.L_cath)
    long_mfp = l_b > 0.0 and l_b > config.L_cath

    J_anode = J_tot - eta * beam_bypass_fraction * J_star
    # Anode floating potential: the anode's own sheath, on its own presheath.
    psi_a = Lambda_anode - math.log(max(1.0 + J_anode / J_i_a, 1e-300))
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
        if circuit_V_avail_V is not None:
            # ...and no larger than what the loop can supply. Without this the
            # kick above is a device voltage the circuit never sourced: with
            # the historical cap as the floor the build leg reports ~1000 V
            # against a bank supplying ~178 V (measured V_b/V_dis ~ 5.1). The
            # floor and this clamp compose without fighting, because the floor
            # value phi_c_ceiling_V is itself <= circuit_V_avail_V whenever the
            # circuit bound is the binding member of the composition.
            #
            # THIS CLAMPED V_b IS NOT WHAT THE CIRCUIT INTEGRATES, and that
            # separation is load-bearing (2026-08-12). Were it, the clamped
            # branch would give vdis_of_I(I) = (V_src - I*(R_comp + R_mesh))
            # + I*R_internal, hence a loop residual f(I) identically zero and
            # a stage derivative g'(I) = 1 exactly: monotone and bracketed,
            # but with NO restoring force, so dI/dt >= 0 everywhere and the
            # loop current could only ratchet upward on whatever the TR
            # stage's explicit kick overshot to (measured 156.7 A vs a
            # converged 0.9 A). The circuit is handed the UNBOUNDED demand
            # instead -- see cathode.idriven_vdis_evaluator -- which keeps
            # g'(I) > 1 strictly and leaves the current free to fall. The
            # runaway this floor was added to stop (I_loop -> 8e8 A,
            # 2026-07-20) is still closed: past the ceiling the unbounded
            # demand rises to the data cap, far above anything the loop can
            # source, so f goes sharply negative there.
            V_b = min(V_b, circuit_V_avail_V)

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
    # Electron flux factors: the fraction of electron saturation actually
    # reaching each electrode across its repelling sheath. The plasma-thermal
    # (2Te) and sheath-fall (phi) parts ride the SAME flux, so each electrode
    # power splits cleanly into ``_thermal + _phi`` (R3.2 / A16 routing).
    fe_c = _exp_clamped(Lambda - max(phi_c_plus, 0.0) / T_e)
    fe_a = _exp_clamped(Lambda_anode - max(phi_a, 0.0) / T_e_anode)
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
    P_cathode_i = _P_ion(phi_c, T_e, I_i)
    P_cathode_i_thermal = I_i * (T_e / 2.0)
    P_cathode_i_phi = P_cathode_i - P_cathode_i_thermal
    P_cathode_i_pl = _P_ion(phi_c, T_e, I_i_a, pl=True)
    P_anode_e = (
        I_i_a
        * (2.0 * T_e_anode + phi_a)
        * fe_a
    )
    P_anode_e_thermal = I_i_a * (2.0 * T_e_anode) * fe_a
    P_anode_e_phi = P_anode_e - P_anode_e_thermal
    P_anode_i = _P_ion(phi_a, T_e_anode, I_i_a)
    P_anode_i_thermal = I_i_a * (T_e_anode / 2.0)
    P_anode_i_phi = P_anode_i - P_anode_i_thermal
    P_anode_i_pl = _P_ion(phi_a, T_e_anode, I_i_a, pl=True)
    _P_beam_bypass = eta * beam_bypass_fraction * I_eth_star * V_b
    # DEPRECATED unclosed scalars (kept bit-exact for the R1-R4 golden only).
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
    # energy deposited on the electrodes. The collector (floating) exhaust is
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
    I_cathode_kirchhoff_residual = (I_eth_star + I_i - I_e_ret) - I_tot

    # Active-bound census (see SolverResult): which member of the composed
    # ceiling this solve ended up sitting on. Derived from the regime tag and
    # the composition already decided above -- no recomputation, no extra
    # evaluation of anything.
    if not capability_limited:
        bound_active = 0.0
    elif _ceiling_is_circuit:
        bound_active = 2.0
    else:
        bound_active = 1.0

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
        circuit_V_avail_V=(
            float("nan") if circuit_V_avail_V is None else circuit_V_avail_V
        ),
        bound_active=bound_active,
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
    beam_excitation_model: str = "2p_scalar",
    schottky: bool = False,
    bridge: bool = False,
    phi_c_cap_V: float = 1000.0,
    alpha_sheath: float | None = None,
    alpha_sheath_anode: float | None = None,
    circuit_V_avail_V: float | None = None,
    circuit_bound_object: str = "phi_c",
    beam_climb_V: float | None = None,
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

    ``beam_climb_V`` is the potential [V] a transmitted beam electron must
    CLIMB between the anode mesh and the column, i.e. the anode-to-wall
    common-mode offset when the vessel node is armed. ``None`` (the default)
    means no node: the launch energy is ``result.phi_c`` itself, the same
    float object, and every beam array below is bit-for-bit the historical
    one. Given, the energy the beam carries into the column is
    ``max(phi_c - max(beam_climb_V, 0), 0)`` -- the FLUX is untouched, because
    the same electrons arrive, decelerated. A climb that takes the launch
    energy to or below ``I_ion`` launches no beam at all, which is the fully
    choked limit and not an error. The sheath solve's own ``phi_c``, ``V_b``
    and gap bypass are NOT shifted: the climb sits downstream of the mesh, and
    the common-mode node moves the whole cathode/anode system together and so
    cannot change the anode-to-cathode differential the circuit integrates.
    """
    cells = len(Te)
    v_beam = np.zeros(cells)
    n_beam = np.zeros(cells)
    beam_cross = np.zeros(cells)
    beam_exc_cross = np.zeros(cells)
    beam_exc_energy = np.zeros(cells)

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
        bridge=bridge,
        phi_c_cap_V=phi_c_cap_V,
        alpha_sheath=alpha_sheath,
        alpha_sheath_anode=alpha_sheath_anode,
        circuit_V_avail_V=circuit_V_avail_V,
        circuit_bound_object=circuit_bound_object,
    )
    phi_c_0 = beam_launch_energy_eV(result.phi_c, beam_climb_V)
    if phi_c_0 > I_ion:
        v_beam[cathode_index] = math.sqrt(2.0 * phi_c_0 * _erg_per_eV / _me_cgs)
        _I_beam_0 = result.I_eth_star * (
            1.0 - config.eta * result.beam_bypass_fraction
        )
        n_beam[cathode_index] = _I_beam_0 / (
            _e_SI * plasma_cross[cathode_index] * v_beam[cathode_index]
        )
        if gas_type == "He":
            # The tabulated He EII cross section ends at eps = E/I_ion =
            # HE_EII_EPS_TOP, and the lookup CLAMPS to its last node above
            # that. On a capability-limited step the beam energy is the sheath
            # ceiling, which at the shipped cap (1000 V) sits on the table's
            # last node to within a ULP -- so the edge is INCLUSIVE within
            # HE_EII_EDGE_REL_TOL, exactly as the tail walk's guard has it
            # (K7c): at the edge the clamped value IS the endpoint node and
            # nothing is extrapolated. A larger excess is refused rather than
            # silently clamped, which is what this call did before. Since the
            # sheath root is now capped, reaching the refusal requires a cap
            # configured above the table top.
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
        beam_exc_energy_eV=beam_exc_energy,
    )
