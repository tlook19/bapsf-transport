"""Prescribed-measured cathode sheath solve.

``circuit_idriven`` inverts the voltage-driven formulation: given the loop
CURRENT, find the sheath the cathode's Richardson emission must sit at to carry
it. This module goes one step further and imposes BOTH measured drive
quantities -- the discharge current ``I(t)`` and the discharge voltage
``V_dis(t)`` from the rung's own overlay trace -- so the cathode's emission
model, its surface temperature and the bank loop are not consulted at all.

**Why.** The operator sets each rung's discharge current by hand: the heater is
raised until the discharge reaches a target power and then held there. The
drive LEVEL is therefore a machine INPUT, not a prediction, and scoring the
column at the measured drive removes every drive-side calibration from the
comparison. What remains under test is the column: transport, fuelling and the
neutral closure, at a drive that is known rather than fitted.

**What is solved.** Everything follows from the loop bookkeeping the
current-driven result is already assembled with,

    V_dis = V_b + V_series,        V_b = phi_c + V_p - phi_a

read for ``phi_c`` instead of for ``V_dis``:

    phi_c = (V_dis - V_series) + phi_a - V_p

with ``V_p = I*R_p`` the resolved gap drop the model carries (the plasma column
between cathode and anode mesh, Spitzer at the boundary cell's own state), and
``V_series = I*((1 - R_comp_partition)*R_comp + R_mesh_ohm)`` the internal
series drop on the plasma side of the V_dis probe. At the shipped defaults
(``R_comp_partition = 1``, ``R_mesh_ohm = 0``) ``V_series`` is identically zero
and the relation is exactly ``phi_c = V_dis + phi_a - V_p``.

The relation is IMPLICIT, but only weakly and only through one term: the anode
fall ``phi_a`` depends on how much of the emitted beam the gap lets through to
the mesh, and that bypass fraction is a function of ``phi_c``. So ``phi_c`` is
located by a bracketed ``brentq`` on ``phi_c - phi_a(phi_c) + V_p - V_b``,
which is the same construction -- one bracketed root of a monotone relation --
that ``solve_idriven`` uses for its current root, and for the same reason:
``dphi_a/dphi_c`` is positive and small (the bypass fraction varies over a
hundred-volt scale and enters ``phi_a`` logarithmically), so the residual is
strictly increasing and the root is unique. Where the bypass is zero the
root-find returns the explicit relation exactly.

**The emitted current.** The cathode's own emission ceiling is not evaluated:
``T_s``, ``C_R`` and the work function belong to the calibrated model this mode
replaces. The emitted electron current is instead read off the measured loop
current and the plasma's own Bohm ion current at the cathode cell,

    I_eth* = max(I - I_i, 0)

i.e. the loop current the ions do not supply is the current the surface emitted.
This is the deep-repelling-sheath limit of the current-driven Kirchhoff
``I = I_i*(1 - exp(Lambda - psi)) + I_eth*``: the returning plasma-electron
current is not subtracted, because with the emission model withdrawn there is
no ``psi`` to evaluate it at. The floor at zero is the statement that a
measured current below the Bohm ion current is a cathode that emitted nothing,
never a surface that absorbed a net electron current.

**What is NOT modelled here.** ``I_eth`` (the Richardson capability) is reported
as zero rather than as a number computed from an unused surface temperature,
and ``regime`` is ``"prescribed"`` rather than ``"classical"`` -- a reader must
be able to tell a sheath the emission model produced from one the trace did.
There is no virtual cathode: ``phi_c_minus`` is zero because the space-charge
barrier is a property of the emission solve, so ``phi_c_plus == phi_c``.

**The one ceiling that remains.** ``phi_c_cap_V`` still bounds the net sheath
drop, because it is a DOMAIN GUARD on the tabulated He ionization cross section
the beam is deposited with, not a statement about the circuit. A measured
``V_dis`` that would demand a deeper sheath than the table covers returns the
ceiling solution tagged ``"capability_limited"``, exactly as the current-driven
path does.
"""

import math

import numpy as np
from scipy.optimize import brentq

from cablp.cathode.circuit import (
    CATHODE_LNL_MODELS,
    BeamResult,
    DeviceConfig,
    PlasmaState,
    SolverResult,
    _LN_LAMBDA_MIN,
    _P_ion,
    _c_log_ei,
    _compute_beam_bypass_fraction,
    _compute_l_b,
    _e_SI,
    _exp_clamped,
    _mp_cgs,
)
from cablp.cathode.circuit_idriven import assemble_beam_arrays

__all__ = [
    "solve_prescribed",
    "solve_beam_system_prescribed",
]

#: Lower end of the phi_c bracket [V]. Strictly positive so the anode-log
#: clamp and the ceiling comparison stay well defined, and small enough that a
#: measured drive demanding essentially no cathode fall still brackets.
_PHI_C_LO_V = 1.0e-8


def solve_prescribed(
    config: DeviceConfig,
    plasma: PlasmaState,
    I_tot_A: float,
    V_dis_V: float,
    cathode_current_A: float | None = None,
    anode_current_A: float | None = None,
    anode_T_e: float | None = None,
    phi_c_cap_V: float = 1000.0,
    alpha_sheath: float | None = None,
    alpha_sheath_anode: float | None = None,
    tail_anode_current_A: float = 0.0,
) -> SolverResult:
    """Solve the cathode sheath for a MEASURED current and device voltage.

    ``I_tot_A`` and ``V_dis_V`` are the rung's own measured discharge current
    [A] and discharge voltage [V], interpolated onto this model time by
    :mod:`cablp.solvers._sim1d.core.prescribed_drive`. Every other parameter
    mirrors ``solve_idriven``'s: ``cathode_current_A`` / ``anode_current_A`` /
    ``anode_T_e`` are the "the fluid already computed this" overrides,
    ``alpha_sheath`` / ``alpha_sheath_anode`` are the two electrodes' own
    presheath factors, and ``tail_anode_current_A`` is the lagged QL-tail
    current the anode mesh collected without it crossing the anode sheath.

    Returns a ``SolverResult`` field-for-field compatible with the other two
    solvers'; see the module docstring for the ``I_eth``, ``regime`` and
    ``phi_c_minus`` contract notes.
    """
    if I_tot_A < 0.0:
        raise ValueError(
            f"I_tot_A must be >= 0 (got {I_tot_A}); a measured trace read "
            "backwards is a configuration error, not a reverse discharge"
        )
    if phi_c_cap_V <= 0.0:
        raise ValueError(f"phi_c_cap_V must be positive (got {phi_c_cap_V})")
    V_dis_V = float(V_dis_V)
    if not math.isfinite(V_dis_V):
        raise ValueError(f"V_dis_V must be finite (got {V_dis_V})")

    T_e = plasma.T_e
    n_e = plasma.n_e

    # ------------------------------------------------------------------
    # Plasma-derived quantities: the SAME formulas the current-driven solve
    # uses, so the two agree bit-for-bit about the gap the loop crosses.
    # ------------------------------------------------------------------
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

    Lambda = config.Lambda + lam_shift
    Lambda_anode = config.Lambda + lam_shift_anode
    eta = config.eta
    I_e = I_i * math.exp(Lambda)

    # THE MEASURED DRIVE. The loop current is the trace's, and the emitted
    # current is what the ions did not supply (see the module docstring).
    I_tot = float(I_tot_A)
    I_eth_star = max(I_tot - I_i, 0.0)

    V_p = I_tot * R_p
    # The internal series drop the V_dis probe does not see. Zero at the
    # shipped defaults, where V_dis IS the device voltage.
    V_series = I_tot * (
        (1.0 - config.R_comp_partition) * config.R_comp + config.R_mesh_ohm
    )
    V_b_target = V_dis_V - V_series

    # ------------------------------------------------------------------
    # The anode fall as a function of the cathode fall, and the single
    # bracketed root of the loop relation (see the module docstring).
    # ------------------------------------------------------------------
    def _anode_state(phi_c):
        l_b = _compute_l_b(phi_c, T_e, n_e, plasma.n_n, plasma.sigma_b)
        bypass = _compute_beam_bypass_fraction(l_b, config.L_cath)
        I_anode = I_tot - eta * bypass * I_eth_star - float(
            tail_anode_current_A
        )
        psi_a = Lambda_anode - math.log(
            max(1.0 + I_anode / I_i_a, 1e-300)
        )
        return psi_a * T_e_anode, l_b, bypass

    def _residual(phi_c):
        return phi_c + V_p - _anode_state(phi_c)[0] - V_b_target

    capability_limited = False
    if _residual(_PHI_C_LO_V) >= 0.0:
        # The measured device voltage does not sustain even the smallest
        # cathode fall at this gap and anode state. The bracket bottom IS the
        # answer; it is kept strictly positive so the anode log and the
        # ceiling comparison stay well defined.
        phi_c = _PHI_C_LO_V
    elif _residual(phi_c_cap_V) <= 0.0:
        # The measured drive demands a deeper sheath than the tabulated beam
        # cross section covers. Return the ceiling solution and TAG it, the
        # same contract the current-driven path carries at its own ceiling.
        phi_c = float(phi_c_cap_V)
        capability_limited = True
    else:
        phi_c = brentq(
            _residual,
            _PHI_C_LO_V,
            phi_c_cap_V,
            xtol=1.0e-12,
            rtol=1.0e-14,
            full_output=False,
        )

    phi_a, l_b, beam_bypass_fraction = _anode_state(phi_c)
    long_mfp = l_b > 0.0 and l_b > config.L_cath
    # No emission solve, hence no space-charge barrier to report: the whole
    # cathode fall is classical.
    phi_c_plus = phi_c
    phi_c_minus = 0.0
    regime = "capability_limited" if capability_limited else "prescribed"

    # V_b is REASSEMBLED from the solved sheath rather than taken as the
    # target, so the exported device voltage is the one the returned potentials
    # actually make. At an interior root the two agree to the root tolerance;
    # at the ceiling they deliberately do not, and the difference IS the
    # statement that the measured drive was not reproducible below the cap.
    V_b = phi_c + V_p - phi_a
    V_dis = V_b + V_series

    P_wall = I_tot * (V_b + I_tot * config.R_comp)
    P_load = I_tot * V_b
    P_comp = I_tot**2 * config.R_comp
    P_prim = (1.0 - eta * beam_bypass_fraction) * I_eth_star * phi_c
    P_ohmic = I_tot * V_p
    # Electron/ion sheath powers: the SAME expressions the current-driven
    # solve assembles, with the same physical flux barriers (an attracting
    # electrode collects at most electron saturation, and the barrier the
    # plasma electrons climb is phi_c_plus).
    fe_c = _exp_clamped(Lambda - max(phi_c_plus, 0.0) / T_e)
    fe_a = _exp_clamped(Lambda_anode - max(phi_a, 0.0) / T_e_anode)
    P_cathode_e = I_i * (2.0 * T_e + phi_c) * fe_c
    P_cathode_e_thermal = I_i * (2.0 * T_e) * fe_c
    P_cathode_e_phi = P_cathode_e - P_cathode_e_thermal
    P_cathode_i = _P_ion(phi_c, T_e, I_i)
    P_cathode_i_thermal = I_i * (T_e / 2.0)
    P_cathode_i_phi = P_cathode_i - P_cathode_i_thermal
    P_cathode_i_pl = _P_ion(phi_c, T_e, I_i_a, pl=True)
    P_anode_e = I_i_a * (2.0 * T_e_anode + phi_a) * fe_a
    P_anode_e_thermal = I_i_a * (2.0 * T_e_anode) * fe_a
    P_anode_e_phi = P_anode_e - P_anode_e_thermal
    P_anode_i = _P_ion(phi_a, T_e_anode, I_i_a)
    P_anode_i_thermal = I_i_a * (T_e_anode / 2.0)
    P_anode_i_phi = P_anode_i - P_anode_i_thermal
    P_anode_i_pl = _P_ion(phi_a, T_e_anode, I_i_a, pl=True)
    _P_beam_bypass = eta * beam_bypass_fraction * I_eth_star * V_b
    # DEPRECATED unclosed scalars, kept only because SolverResult carries the
    # fields; they are not exported to the saved trajectory (see the
    # SolverResult field block in circuit.py).
    P_net = (
        P_load - P_cathode_e - P_cathode_i - P_anode_e - P_anode_i
        - _P_beam_bypass
    )
    P_net2 = (
        P_prim + P_ohmic - P_cathode_e - P_cathode_i_pl - P_anode_e
        - P_anode_i_pl
    )
    P_loss = P_cathode_e + P_cathode_i_pl + P_anode_e + P_anode_i_pl
    P_plasma_thermal_loss = (
        P_cathode_e_thermal + P_cathode_i_thermal
        + P_anode_e_thermal + P_anode_i_thermal
    )
    P_into_plasma = P_prim + P_ohmic - P_plasma_thermal_loss
    P_cathode_surface = P_cathode_e + P_cathode_i
    P_anode_surface = P_anode_e + P_anode_i

    I_parallel = 0.0
    I_plasma = I_tot
    I_bank = I_plasma + I_parallel
    # Load-power closure, the same current-resolved ledger the current-driven
    # result carries. The cathode Kirchhoff residual is NOT identically zero
    # here and is not meant to be: this mode books the emitted current as
    # I - I_i with no returning-electron term, so the residual MEASURES the
    # returning current the deep-sheath limit dropped instead of hiding it.
    I_e_ret = P_cathode_e_phi / phi_c if phi_c != 0.0 else 0.0
    cathode_field_work = I_eth_star * phi_c + P_cathode_i_phi - P_cathode_e_phi
    P_load_ledger = cathode_field_work + P_ohmic - I_tot * phi_a
    P_load_residual = P_load - P_load_ledger
    I_cathode_kirchhoff_residual = (I_eth_star + I_i - I_e_ret) - I_tot

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
        # The Richardson capability is NOT evaluated in this mode -- it is a
        # property of the surface model the measured drive replaces -- so it
        # is reported as zero rather than as a number read off an unused T_s.
        I_eth=0.0,
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
        # The data cap is the only ceiling in this mode: there is no loop to
        # supply a circuit member, so the census can only ever read 0 or 1.
        phi_c_ceiling_V=float(phi_c_cap_V),
        circuit_V_avail_V=float("nan"),
        bound_active=1.0 if capability_limited else 0.0,
        regime=regime,
        long_mfp=long_mfp,
        beam_bypass_fraction=beam_bypass_fraction,
        l_b=l_b,
    )


def solve_beam_system_prescribed(
    config: DeviceConfig,
    Te: np.ndarray,
    ne: np.ndarray,
    nn: np.ndarray,
    beam_cross_prev: np.ndarray,
    plasma_cross: np.ndarray,
    I_ion: float,
    gas_type: str,
    I_tot_A: float,
    V_dis_V: float,
    cathode_index: int = 0,
    anode_current_A: float | None = None,
    anode_T_e: float | None = None,
    b_beam_excitation: float = 0.0,
    beam_excitation_energy_eV: float = 21.218,
    beam_excitation_model: str = "2p_scalar",
    phi_c_cap_V: float = 1000.0,
    alpha_sheath: float | None = None,
    alpha_sheath_anode: float | None = None,
    beam_climb_V: float | None = None,
    tail_anode_current_A: float = 0.0,
) -> BeamResult:
    """Prescribed-measured counterpart of ``solve_beam_system_idriven``.

    Solves the sheath from the measured ``(I, V_dis)`` pair and hands the
    result to the SHARED per-cell beam assembly
    (:func:`cablp.cathode.circuit_idriven.assemble_beam_arrays`), so the beam
    the column sees is built by exactly the same code on both drive routes and
    only the sheath underneath it differs.
    """
    result = solve_prescribed(
        config,
        PlasmaState(
            T_e=Te[cathode_index],
            n_e=ne[cathode_index],
            n_n=nn[cathode_index],
            sigma_b=beam_cross_prev[cathode_index],
        ),
        I_tot_A=I_tot_A,
        V_dis_V=V_dis_V,
        anode_current_A=anode_current_A,
        anode_T_e=anode_T_e,
        phi_c_cap_V=phi_c_cap_V,
        alpha_sheath=alpha_sheath,
        alpha_sheath_anode=alpha_sheath_anode,
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
        gas_type=gas_type,
        cathode_index=cathode_index,
        b_beam_excitation=b_beam_excitation,
        beam_excitation_energy_eV=beam_excitation_energy_eV,
        beam_excitation_model=beam_excitation_model,
        beam_climb_V=beam_climb_V,
    )
