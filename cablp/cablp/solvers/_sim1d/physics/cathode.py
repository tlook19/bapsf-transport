import dataclasses
from dataclasses import dataclass
import math

import numpy as np

from scipy.optimize import brentq

from cablp.funcs._beam_deposition import deposit_beam
from cablp.funcs._cathode_solver import (
    DeviceConfig,
    PlasmaState,
    _compute_l_b,
    solve_beam_system,
)
from cablp.funcs._cathode_solver_idriven import (
    solve_beam_system_idriven,
    solve_idriven,
)
from cablp.vars._cons import ev_to_erg, qe_SI

from ..core.geometry import (
    anode_flanking_cells,
    cathode_adjacent_cells,
    gap_cell_indices,
)
from ..core.state import ConservativeState1D, derive_state
from .flux import ion_sound_speed
from .reactions import _birth_temperature


@dataclass(frozen=True)
class CathodeCellState1D:
    """Primitive/source-cell state passed toward a cathode adapter."""

    index: int
    role: str
    n: float
    nn: float
    Te: float
    Ti: float
    u: float
    plasma_volume_cm3: float
    neutral_volume_cm3: float
    plasma_area_cm2: float
    neutral_area_cm2: float
    length_cm: float
    Rp_cm: float
    Rm_cm: float


@dataclass(frozen=True)
class CathodeBoundaryState1D:
    """Source/end boundary state and circuit placeholders for cathode coupling."""

    source: CathodeCellState1D
    end: CathodeCellState1D
    enabled: bool
    mode: str
    end_mode: str
    twin_cathode: bool
    circuit: dict


@dataclass(frozen=True)
class CathodeSourceTerms1D:
    """Conservative cathode source placeholders and raw metadata."""

    rhs: ConservativeState1D
    enabled: bool
    mode: str
    metadata: dict


@dataclass(frozen=True)
class CathodeSolve1D:
    """Opt-in cathode solve result without conservative RHS coupling."""

    boundary: CathodeBoundaryState1D
    beam_result: object | None
    device_config: DeviceConfig | None
    x0_next: float | None
    x0_twin_next: float | None
    metadata: dict
    # Per-end CSDA deposition results ({0: primary, -1: twin}), present only
    # under beam_deposition_model = "csda"; None keys mean no active beam.
    beam_deposition: dict | None = None


def anode_circuit_sample(state, derived, geometry, mu, input_dict, end=0):
    """Return ``(I_i_a [A], Te_anode [eV])`` for one anode, or ``(None, None)``.

    §7: the historical circuit takes ``I_i_a = 2*eta*I_i``, scaling the anode
    current straight off the *cathode* cell, which assumes both electrodes see the
    same plasma -- precisely what a resolved cathode-anode gap breaks.

    The current handed back is the same Bohm collection
    ``sources.anode_collection_rhs`` removes from the fluid, summed over both mesh
    faces with each face sampled on its own side. Computing it once and sharing it
    means the circuit and the fluid cannot disagree about the anode current, and it
    is why M5 must not add a second anode particle sink.

    The sheath temperature is collection-weighted across the two faces, matching
    how ``P_anode_e`` is apportioned. Resolving a *separate* sheath per face is
    §11 #6 and remains open.
    """
    anode_faces = np.asarray(getattr(geometry, "anode_face_indices", ()), dtype=int)
    eta = float(input_dict.get("eta", 0.0))
    if anode_faces.size == 0 or eta <= 0.0:
        return None, None
    face = int(anode_faces[0] if end == 0 else anode_faces[-1])
    total = 0.0
    weighted_Te = 0.0
    for cell in (face - 1, face):
        collected = (
            np.exp(-0.5)
            * state.n[cell]
            * ion_sound_speed(derived.Te[cell], mu)
            * eta
            * float(geometry.plasma_area_cm2[cell])
        )
        total += collected
        weighted_Te += collected * float(derived.Te[cell])
    if total <= 0.0:
        return None, None
    return total * qe_SI, weighted_Te / total


def cathode_sample_indices(geometry):
    """Return the ``(source, end)`` cells the cathode circuit samples.

    The cathode solve builds its ion current from the plasma against the cathode
    surface, so in resolved geometry it must read the *cathode-adjacent* cell --
    cell ``[0]`` there is the plasma-dead plenum, whose floor density and
    temperature would drive the circuit with garbage.

    A twin machine samples both cathodes; otherwise the ``end`` slot is the
    collector, which is what ``end_mode`` describes.
    """
    cathode_cells = cathode_adjacent_cells(geometry)
    if not cathode_cells:
        raise ValueError("resolved geometry must define cathode-adjacent cells")
    source_index = int(cathode_cells[0])
    if len(cathode_cells) > 1:
        return source_index, int(cathode_cells[-1])
    return source_index, geometry.cells - 1


def cathode_boundary_state(
    state,
    floors,
    ion_mass_g,
    geometry,
    input_dict,
    input_flags,
):
    """Return finite source/end quantities for a future cathode solver adapter."""
    derived = derive_state(state, floors=floors, ion_mass_g=ion_mass_g)
    source_index, end_index = cathode_sample_indices(geometry)
    return CathodeBoundaryState1D(
        source=_cell_state(source_index, state, derived, geometry),
        end=_cell_state(end_index, state, derived, geometry),
        enabled=bool(input_flags.get("cathode_coupling", False)),
        mode=input_dict.get("cathode_model", "disabled"),
        end_mode=input_dict.get("end_mode", "collector"),
        twin_cathode=bool(input_flags.get("TwinCathode", False)),
        circuit=_circuit_placeholders(input_dict),
    )


def cathode_emission_annuli(input_dict, n_annuli=10):
    """Return ``(Ts_K, area_cm2, plasma_frac)`` tuples for the emission profile.

    The measured plasma Te(x) footprint (FWHM ~ 28-29 cm at the first ES1
    port, slightly broadened en route from the cathode) is read as the
    field-line-mapped emission-current footprint of the real cathode disc:
    ``j(r) = j0 * exp(-4 ln2 r^2 / FWHM^2)``. Inverting Richardson gives the
    local surface temperature ``1/T(r) = 1/T_s - (kB/phi_wf) ln(j/j0)`` with
    ``T_s`` the peak. The center-to-edge drop this implies (~150-200 K) is
    what softens the emission knee from a razor wall into a stable ramp.

    ``plasma_frac`` is each annulus's overlap with the plasma footprint
    (``r < Rp``): annuli beyond it collect no ion current and are
    space-charge choked, which the solver handles naturally.
    """
    R_cath = float(input_dict["R_cath"])
    Rp = float(input_dict.get("Rp", R_cath))
    T_s = float(input_dict["T_s"])
    phi_wf = float(input_dict["phi_wf"])
    fwhm = float(input_dict.get("cathode_Ts_fwhm_cm", 28.0))
    if fwhm <= 0.0:
        raise ValueError(f"cathode_Ts_fwhm_cm must be positive (got {fwhm})")
    kB_over_e = 8.617333262e-5  # eV/K
    edges = np.linspace(0.0, R_cath, int(n_annuli) + 1)
    Ts_k, area_k, frac_k = [], [], []
    for r0, r1 in zip(edges[:-1], edges[1:]):
        r_mid = 0.5 * (r0 + r1)
        ln_j = -4.0 * math.log(2.0) * r_mid**2 / fwhm**2
        inv_T = 1.0 / T_s - (kB_over_e / phi_wf) * ln_j
        Ts_k.append(1.0 / inv_T)
        area_k.append(math.pi * (r1**2 - r0**2))
        if r1 <= Rp:
            frac_k.append(1.0)
        elif r0 >= Rp:
            frac_k.append(0.0)
        else:
            frac_k.append((Rp**2 - r0**2) / (r1**2 - r0**2))
    return tuple(Ts_k), tuple(area_k), tuple(frac_k)


def cathode_device_config(input_dict, input_flags, mu):
    """Build the existing cathode solver's static device configuration."""
    R_cath = float(input_dict["R_cath"])
    profile = str(input_dict.get("cathode_emission_profile", "uniform"))
    if profile == "uniform":
        annuli = ((), (), ())
    elif profile == "gaussian":
        annuli = cathode_emission_annuli(
            input_dict,
            n_annuli=int(input_dict.get("cathode_emission_annuli", 10)),
        )
    else:
        raise ValueError(
            "cathode_emission_profile must be 'uniform' or 'gaussian' "
            f"(got {profile!r})"
        )
    return DeviceConfig(
        A_c=math.pi * R_cath**2,
        mu=mu,
        V_bank=float(input_dict["V_bank"]),
        T_s=float(input_dict["T_s"]),
        phi_wf=float(input_dict["phi_wf"]),
        C_R=float(input_dict["C_R"]),
        R_comp=float(input_dict["R_comp"]),
        eta=float(input_dict["eta"]),
        Twin=bool(input_flags.get("TwinCathode", False)),
        L_cath=float(input_dict["L_cath"]),
        R_cath=R_cath,
        emission_Ts_K=annuli[0],
        emission_area_cm2=annuli[1],
        emission_plasma_frac=annuli[2],
    )


_SIGMA_SB_W_CM2_K4 = 5.670374419e-12
_KB_EV_PER_K = 8.617333262e-5


def cathode_power_balance_terms_W(T_s_K, P_ion_W, I_eth_star_A, input_dict):
    """Return ``(P_heater, P_ion, P_rad, P_emis, P_cond)`` [W] for warming.

    The ``cathode_warming_model = "power_balance"`` surface energy budget
    (CATHODE_IDRIVEN_PLAN.md M1b):

    - ``P_heater`` is pinned by the standby equilibrium at
      ``cathode_Ts_base_K`` -- open circuit means no net emission and no
      substrate gradient, so the heater exactly balances radiation there
      and is not a free parameter.
    - ``P_ion`` is the accepted solve's ion bombardment power.
    - ``P_rad`` is gray-body radiation from the emitting face.
    - ``P_emis`` is evaporative emission cooling: each *actually emitted*
      electron removes ``phi_wf + 2 k_B T_s`` (work function plus the mean
      thermal energy over the barrier). Pass the accepted solve's
      ``I_eth_star`` -- the space-charge-released current, not the
      Richardson ceiling -- and 0 for floating phases, where emitted
      electrons return to the surface and the net cooling vanishes.
    - ``P_cond`` is conduction from the emitting skin layer into the
      heater-held substrate, ``G_cond * (T_s - T_base)`` -- the
      "heater maintains the lower end" restoring term. It vanishes at
      standby by construction, so the heater pinning is unchanged.
      **Without it the balance is unstable at the LAPD operating point**
      (measured 2026-07-20, `es1_nx120_pb_demo.h5`): the bombardment
      feedback gain d(P_ion)/dT through the emission loop exceeds the
      ~230 W/K radiation+emission stiffness, and the current runs to
      12.9 kA before the sheath saturates the loop.

    The net rate is ``(P_heater + P_ion - P_rad - P_emis - P_cond) /
    C_th``; the caller owns the time discretization.
    """
    area = input_dict.get("cathode_rad_area_cm2")
    if area is None:
        area = math.pi * float(input_dict["R_cath"]) ** 2
    eps = float(input_dict.get("cathode_emissivity", 0.7))
    T_env = float(input_dict.get("cathode_env_T_K", 300.0))
    T_base = float(input_dict["cathode_Ts_base_K"])

    def _rad(T):
        return eps * _SIGMA_SB_W_CM2_K4 * float(area) * (T**4 - T_env**4)

    P_emis = max(float(I_eth_star_A), 0.0) * (
        float(input_dict["phi_wf"]) + 2.0 * _KB_EV_PER_K * float(T_s_K)
    )
    P_cond = float(input_dict.get("cathode_conduction_W_per_K", 0.0)) * (
        float(T_s_K) - T_base
    )
    return (
        _rad(T_base),
        max(float(P_ion_W), 0.0),
        _rad(float(T_s_K)),
        P_emis,
        P_cond,
    )


def spitzer_sigma_par_ohm_cm(Te_eV):
    """Parallel Spitzer conductivity [Ohm^-1 cm^-1], as the cathode solver's.

    Must match the internal ``sigma_par = 14.6 * T_e**1.5`` (fixed Coulomb
    logarithm) in ``_cathode_solver.solve`` so the ``"resolved_gap"`` R_p
    model reduces exactly to ``"sample"`` over a uniform gap. Follow-up on
    record (Tom, 2026-07-20): derive sigma_par from the electron-ion
    collision frequency nu_ei -- lnLambda(Te, n) via ``funcs._plasmaparams``
    (``c_log``, ``time_elec_coll``) -- instead of this fixed-lnLambda fit.
    That has to move in lockstep with the voltage-driven solver's internal
    formula (or land only on the current-driven path), so it is a deliberate
    separate change, not a drive-by here.
    """
    return 14.6 * np.asarray(Te_eV, dtype=float) ** 1.5


def resolved_gap_resistance_ohm(Te, geometry):
    """Return the profile-integrated cathode-anode gap resistance [Ohm].

    ``R_p = sum_k dz_k / (sigma_par(Te_k) * A_k)`` over the resolved gap
    cells, with each cell's own plasma-channel area -- the series resistance
    of the actual column the discharge current crosses, and the same
    per-cell Spitzer weighting ``_ohmic_gap_weights`` deposits P_ohmic with.
    The historical single-sample formula spreads the hot cathode-adjacent
    conductivity over the whole gap and so underestimates a colder gap
    (eta_Spitzer ~ Te^-3/2).
    """
    gap = np.asarray(gap_cell_indices(geometry, end=0), dtype=int)
    dz = np.asarray(geometry.length_cm, dtype=float)[gap]
    area = np.asarray(geometry.plasma_area_cm2, dtype=float)[gap]
    sigma = spitzer_sigma_par_ohm_cm(np.asarray(Te, dtype=float)[gap])
    return float(np.sum(dz / (sigma * area)))


def validate_cathode_solver_model(input_dict, input_flags):
    """Validate and return the ``cathode_solver_model`` selection."""
    model = str(input_dict.get("cathode_solver_model", "current_driven"))
    if model != "current_driven":
        raise ValueError(
            "cathode_solver_model='voltage_driven' was removed at "
            "DEPRECATION_PLAN D2; use 'current_driven' or reproduce the "
            "historical path at tag legacy-final-2026-07-22 "
            f"(got {model!r})"
        )
    coupling = bool(input_flags.get("cathode_coupling", False))
    if coupling and bool(input_flags.get("TwinCathode", False)):
        raise ValueError(
            "cathode_solver_model='current_driven' does not support "
            "TwinCathode"
        )
    if coupling and float(input_dict.get("L_parasitic_H", 0.0)) <= 0.0:
        raise ValueError(
            "cathode_solver_model='current_driven' requires "
            "L_parasitic_H > 0"
        )
    return model


def apply_cathode_Rp_model(device_config, derived, geometry, input_dict, input_flags):
    """Apply ``cathode_Rp_model`` to the device config (M1 feed-in).

    Returns ``(device_config, applied_model, R_p_gap_ohm)``; see
    ``resolved_gap_resistance_ohm`` and the ``cathode_Rp_model`` config
    docs. Shared by the per-step solve dispatch and the current-driven
    circuit's V_dis(I) evaluator so the two cannot disagree about R_p.
    """
    Rp_model = validate_cathode_Rp_model(input_dict, input_flags)
    R_p_gap_ohm = None
    if Rp_model == "resolved_gap":
        R_p_gap_ohm = resolved_gap_resistance_ohm(derived.Te, geometry)
        # DeviceConfig.R_cath is an effective value on this path: invert the
        # cathode solver's sampled Spitzer formula so it carries the resolved
        # profile-integrated resistance exactly.
        Te_sample = float(derived.Te[beam_launch(geometry, end=0)[0]])
        sigma_sample = float(spitzer_sigma_par_ohm_cm(Te_sample))
        device_config = dataclasses.replace(
            device_config,
            R_cath=math.sqrt(
                device_config.L_cath / (math.pi * sigma_sample * R_p_gap_ohm)
            ),
        )
    return device_config, Rp_model, R_p_gap_ohm


def idriven_result_evaluator(
    state,
    floors,
    ion_mass_g,
    mu,
    geometry,
    input_dict,
    input_flags,
    beam_cross_prev,
    T_s_override_K=None,
    phi_wf_override_eV=None,
):
    """Return an ``I [A] -> SolverResult`` evaluator at this frozen state.

    Builds the same device config (T_s substitution, ``cathode_Rp_model``
    feed-in, anode sample) as the per-step dispatch, via the same helpers,
    so its consumers and the dispatched solve cannot disagree. Two
    consumers: the circuit advance (through ``idriven_vdis_evaluator``)
    and the power-balance warming update, which needs *accepted-state*
    P_cathode_i / I_eth_star -- the RHS cache ``_cathode_solve`` holds the
    last internal-stage solve of the step, whose P_cathode_i was measured
    at 4.6-7.5x the accepted-state value at the same frozen current
    (2026-07-21; the stage state sits on the other side of the knee).
    """
    derived = derive_state(state, floors=floors, ion_mass_g=ion_mass_g)
    anode_A, anode_Te = anode_circuit_sample(
        state, derived, geometry, mu, input_dict, end=0
    )
    if T_s_override_K is not None:
        input_dict = {**input_dict, "T_s": float(T_s_override_K)}
    if phi_wf_override_eV is not None:
        input_dict = {**input_dict, "phi_wf": float(phi_wf_override_eV)}
    device_config = cathode_device_config(input_dict, input_flags, mu)
    device_config, _, _ = apply_cathode_Rp_model(
        device_config, derived, geometry, input_dict, input_flags
    )
    idx = beam_launch(geometry, end=0)[0]
    beam_cross_prev = np.asarray(beam_cross_prev, dtype=float)
    plasma = PlasmaState(
        T_e=float(derived.Te[idx]),
        n_e=float(state.n[idx]),
        n_n=float(state.nn[idx]),
        sigma_b=float(beam_cross_prev[idx]),
    )
    schottky = bool(input_flags.get("cathode_schottky", False))
    bridge = bool(input_flags.get("cathode_emission_bridge", False))
    cap = float(input_dict.get("cathode_phi_c_cap_V", 1000.0))

    def solve_at(I_A):
        return solve_idriven(
            device_config,
            plasma,
            I_tot_A=max(float(I_A), 0.0),
            anode_current_A=anode_A,
            anode_T_e=anode_Te,
            schottky=schottky,
            bridge=bridge,
            phi_c_cap_V=cap,
        )

    return solve_at


def idriven_vdis_evaluator(
    state,
    floors,
    ion_mass_g,
    mu,
    geometry,
    input_dict,
    input_flags,
    beam_cross_prev,
    T_s_override_K=None,
    phi_wf_override_eV=None,
):
    """Return a ``V_dis(I) [V]`` evaluator at this frozen plasma state.

    Used by the current-driven circuit advance: each implicit stage
    root-finds the loop current against the monotone device voltage, so it
    needs many cheap sheath evaluations at the *accepted* end-of-step state
    with only I varying. Thin wrapper over ``idriven_result_evaluator``.
    """
    solve_at = idriven_result_evaluator(
        state=state,
        floors=floors,
        ion_mass_g=ion_mass_g,
        mu=mu,
        geometry=geometry,
        input_dict=input_dict,
        input_flags=input_flags,
        beam_cross_prev=beam_cross_prev,
        T_s_override_K=T_s_override_K,
        phi_wf_override_eV=phi_wf_override_eV,
    )

    def vdis(I_A):
        return solve_at(I_A).V_b

    return vdis


def advance_circuit_current_driven(
    I_prev_A,
    dt_s,
    V_src_V,
    R_comp_ohm,
    L_H,
    vdis_of_I,
    C_bank_F=None,
    V_cap_prev_V=None,
):
    """TR-BDF2 advance of the loop current against a monotone V_dis(I).

    Integrates ``dI/dt = (V_src - I*R - V_dis(I)) / L`` over one accepted
    step. The stage residual ``g(I) = I - rhs - a*f(I)`` has
    ``g' = 1 + a*(R + dV_dis/dI)/L >= 1`` because the current-driven device
    voltage is monotone in I, so each stage is a bracketed scalar brentq --
    unconditionally well-posed however steep V_dis(I) gets. This is the
    load-bearing design decision (plan §2c, revised 2026-07-20): a
    frozen-V_dis explicit step needs ``dV/dI < 2L/dt ~ 22 mOhm`` at
    production dt, and the measured device slope near the emission ceiling
    is 0.2 Ohm-0.75 MOhm -- explicit would sawtooth exactly where this
    machine operates. TR-BDF2 because the RLC gate demands 2nd order and
    TR alone would ring against the near-vertical branch (L-stability, the
    same argument as the heat-conduction scheme choice).

    ``I >= 0`` is enforced per stage (the plasma-diode stand-in, plan
    §2c): a stage whose unconstrained root is negative clamps to 0.
    ``V_src_V`` is held constant over the step (drive: bank/capacitor
    voltage; tail: 0); the capacitor, when present, is frozen for the I
    stages (droop ~2e-4 V/step) and then advanced trapezoidally. Returns
    ``(I_new_A, V_cap_new_V_or_None, V_dis_step_V)``.

    ``V_dis_step_V`` is the step-integrated discharge voltage -- the
    *inductor's view*, from the integrated loop equation over the step:
    ``<V_dis> = V_src - R*<I> - L*(I_new - I_prev)/dt`` with ``<I>`` the
    piecewise-trapezoidal average through the TR stage. This is the honest
    smooth V_dis trace (chatter diagnosis, 2026-07-21): the per-solve
    ``V_b`` inherits the boundary cell's per-step Te wobble through the
    knee, but the circuit only ever integrates V_dis against L, so the
    step average is the physically meaningful instantaneous voltage.
    """
    L = float(L_H)
    if L <= 0.0:
        raise ValueError(f"L_H must be positive (got {L_H})")
    dt = float(dt_s)
    I_n = max(float(I_prev_A), 0.0)

    def f(I):
        return (float(V_src_V) - I * float(R_comp_ohm) - vdis_of_I(I)) / L

    def stage_solve(rhs_const, a_coef):
        def g(I):
            return I - rhs_const - a_coef * f(I)

        if g(0.0) >= 0.0:
            return 0.0
        hi = max(I_n, 1.0)
        for _ in range(200):
            hi *= 2.0
            if g(hi) > 0.0:
                break
        else:
            raise RuntimeError(
                "circuit stage bracket did not close "
                f"(I_n={I_n:.6g} A, rhs={rhs_const:.6g})"
            )
        return brentq(g, 0.0, hi, xtol=1e-10, rtol=1e-12, full_output=False)

    gamma = 2.0 - math.sqrt(2.0)
    f_n = f(I_n)
    # TR stage to t + gamma*dt: I_g = I_n + (gamma*dt/2)*(f_n + f(I_g))
    a1 = 0.5 * gamma * dt
    I_g = stage_solve(I_n + a1 * f_n, a1)
    # BDF2 stage to t + dt:
    #   I_1 = I_g/(gamma*(2-gamma)) - I_n*(1-gamma)^2/(gamma*(2-gamma))
    #         + dt*(1-gamma)/(2-gamma) * f(I_1)
    denom = gamma * (2.0 - gamma)
    a2 = dt * (1.0 - gamma) / (2.0 - gamma)
    I_new = stage_solve(
        I_g / denom - I_n * (1.0 - gamma) ** 2 / denom, a2
    )

    # Step-integrated V_dis from the loop identity (see docstring). <I> is
    # the trapezoidal average through the internal TR stage -- second-order
    # consistent with the current trajectory the scheme just committed to.
    I_avg = 0.5 * (gamma * (I_n + I_g) + (1.0 - gamma) * (I_g + I_new))
    V_dis_step = (
        float(V_src_V)
        - float(R_comp_ohm) * I_avg
        - L * (I_new - I_n) / dt
    )

    V_cap_new = None
    if C_bank_F is not None and float(C_bank_F) > 0.0:
        V_cap_prev = (
            float(V_cap_prev_V) if V_cap_prev_V is not None else 0.0
        )
        V_cap_new = max(
            V_cap_prev - dt * 0.5 * (I_n + I_new) / float(C_bank_F), 0.0
        )
    return I_new, V_cap_new, V_dis_step


def validate_cathode_Rp_model(input_dict, input_flags):
    """Validate and return the ``cathode_Rp_model`` selection."""
    model = str(input_dict.get("cathode_Rp_model", "sample"))
    if model not in ("sample", "resolved_gap"):
        raise ValueError(
            "cathode_Rp_model must be 'sample' or 'resolved_gap' "
            f"(got {model!r})"
        )
    if model == "resolved_gap" and bool(input_flags.get("TwinCathode", False)):
        raise ValueError(
            "cathode_Rp_model='resolved_gap' does not support TwinCathode: "
            "both cathodes share one DeviceConfig, so a single effective "
            "R_cath cannot carry two gaps sampled at different Te. Use "
            "cathode_Rp_model='sample' for twin configurations."
        )
    return model


def solve_cathode_boundary(
    state,
    floors,
    ion_mass_g,
    mu,
    geometry,
    input_dict,
    input_flags,
    beam_cross_prev,
    I_ion,
    gas_type,
    x0=None,
    x0_twin=None,
    floating=False,
    T_s_override_K=None,
    phi_wf_override_eV=None,
    circuit_I_loop_A=0.0,
):
    """Call the cathode/beam solver and return raw diagnostics only."""
    boundary = cathode_boundary_state(
        state=state,
        floors=floors,
        ion_mass_g=ion_mass_g,
        geometry=geometry,
        input_dict=input_dict,
        input_flags=input_flags,
    )
    if not boundary.enabled:
        return CathodeSolve1D(
            boundary=boundary,
            beam_result=None,
            device_config=None,
            x0_next=x0,
            x0_twin_next=x0_twin,
            metadata={
                "enabled": False,
                "mode": boundary.mode,
                "floating": bool(floating),
                "source_index": boundary.source.index,
                "end_index": boundary.end.index,
                "end_mode": boundary.end_mode,
                "twin_cathode": boundary.twin_cathode,
                "circuit": dict(boundary.circuit),
            },
        )

    derived = derive_state(state, floors=floors, ion_mass_g=ion_mass_g)
    anode_source = anode_circuit_sample(
        state, derived, geometry, mu, input_dict, end=0
    )
    anode_twin = anode_circuit_sample(
        state, derived, geometry, mu, input_dict, end=-1
    )
    if T_s_override_K is not None:
        # cathode_warming_model: substitute the evolving surface temperature
        # at the single point every emission path (uniform Richardson and the
        # annular profile, whose peak re-anchors) reads T_s from.
        input_dict = {**input_dict, "T_s": float(T_s_override_K)}
    if phi_wf_override_eV is not None:
        # cathode_surface_model: substitute the evolving effective work
        # function at the single point every phi_wf consumer reads from --
        # Richardson/DeviceConfig, the Schottky reference barrier, the
        # gaussian profile's Richardson inversion, and (via the same dict
        # in the solver) the power-balance emission-cooling term. One
        # shared constant, changed in one place (plan §3b).
        input_dict = {**input_dict, "phi_wf": float(phi_wf_override_eV)}
    device_config = cathode_device_config(input_dict, input_flags, mu)
    device_config, Rp_model, R_p_gap_ohm = apply_cathode_Rp_model(
        device_config, derived, geometry, input_dict, input_flags
    )
    solver_model = validate_cathode_solver_model(input_dict, input_flags)
    beam_cross_prev = np.asarray(beam_cross_prev, dtype=float)
    if beam_cross_prev.shape != (geometry.cells,):
        raise ValueError(
            "beam_cross_prev must have shape "
            f"({geometry.cells},), got {beam_cross_prev.shape}"
        )
    if not floating:
        # The circuit is explicit solver state: no inductive fold, no
        # warm start -- the solve is a well-posed evaluation at the frozen
        # loop current. Floating phases fall through to the historical
        # open-circuit solve below (its Boltzmann-suppressed emission
        # branch is not the same limit as I_tot = 0 here).
        beam_result = solve_beam_system_idriven(
            config=device_config,
            Te=derived.Te,
            ne=state.n,
            nn=state.nn,
            beam_cross_prev=beam_cross_prev,
            plasma_cross=geometry.plasma_area_cm2,
            I_ion=I_ion,
            gas_type=gas_type,
            I_tot_A=max(float(circuit_I_loop_A), 0.0),
            cathode_index=beam_launch(geometry, end=0)[0],
            anode_current_A=anode_source[0],
            anode_T_e=anode_source[1],
            b_beam_excitation=float(
                input_dict.get("b_beam_excitation", 0.0)
            ),
            beam_excitation_energy_eV=float(
                input_dict.get("beam_excitation_energy_eV", 21.218)
            ),
            beam_excitation_model=str(
                input_dict.get("beam_excitation_model", "2p_scalar")
            ),
            schottky=bool(input_flags.get("cathode_schottky", False)),
            bridge=bool(input_flags.get("cathode_emission_bridge", False)),
            phi_c_cap_V=float(input_dict.get("cathode_phi_c_cap_V", 1000.0)),
        )
    else:
        beam_result = solve_beam_system(
            config=device_config,
            Te=derived.Te,
            ne=state.n,
            nn=state.nn,
            beam_cross_prev=beam_cross_prev,
            plasma_cross=geometry.plasma_area_cm2,
            I_ion=I_ion,
            gas_type=gas_type,
            x0=x0,
            x0_twin=x0_twin,
            floating=bool(floating),
            # The solver samples the plasma at these cells *and* writes its
            # beam quantities there, so they must be the same cells
            # `beam_launch` reads from -- otherwise the beam arrays are all
            # zero where the caller looks.
            cathode_index=beam_launch(geometry, end=0)[0],
            twin_index=beam_launch(geometry, end=-1)[0],
            # Hand the circuit the same anode Bohm current the fluid
            # removes, so the two cannot disagree (§7).
            anode_current_A=anode_source[0],
            anode_T_e=anode_source[1],
            anode_current_twin_A=anode_twin[0],
            anode_T_e_twin=anode_twin[1],
            b_beam_excitation=float(input_dict.get("b_beam_excitation", 0.0)),
            beam_excitation_energy_eV=float(
                input_dict.get("beam_excitation_energy_eV", 21.218)
            ),
            beam_excitation_model=str(
                input_dict.get("beam_excitation_model", "2p_scalar")
            ),
        )
    beam_deposition = None
    if str(input_dict.get("beam_deposition_model", "beer_lambert")) == "csda":
        beam_deposition = _csda_beam_deposition(
            beam_result=beam_result,
            state=state,
            derived=derived,
            geometry=geometry,
            device_config=device_config,
            input_dict=input_dict,
            I_ion=I_ion,
            twin=boundary.twin_cathode,
        )
    return CathodeSolve1D(
        boundary=boundary,
        beam_result=beam_result,
        device_config=device_config,
        x0_next=beam_result.x0_next,
        x0_twin_next=beam_result.x0_twin_next,
        metadata={
            "enabled": True,
            "mode": boundary.mode,
            "floating": bool(floating),
            "source_index": boundary.source.index,
            "end_index": boundary.end.index,
            "end_mode": boundary.end_mode,
            "twin_cathode": boundary.twin_cathode,
            "circuit": dict(boundary.circuit),
            "cathode_Rp_model": Rp_model,
            "R_p_gap_ohm": R_p_gap_ohm,
            "cathode_solver_model": solver_model,
            "result": _solver_result_metadata(beam_result.result),
            "result_twin": _solver_result_metadata(beam_result.result_twin),
        },
        beam_deposition=beam_deposition,
    )


def _csda_beam_deposition(
    beam_result,
    state,
    derived,
    geometry,
    device_config,
    input_dict,
    I_ion,
    twin=False,
):
    """Run the CSDA module for each active cathode ray (B2 wiring).

    Returns ``{0: BeamDepositionResult | None, -1: ...}`` and rewrites
    ``beam_result.beam_atten_cross`` at each launch cell with the effective
    attenuation cross section that makes the frozen sheath solve's
    Beer-Lambert bypass reproduce the module's cathode-anode gap
    transmission on the *next* solve (the same one-step lag the historical
    ``sigma_b`` feedback has). The frozen solve computes
    ``bypass = exp(-L_cath / l_b)`` with ``1/l_b = 1/l_bi + sigma*nn``, so
    the adapter solves for sigma and clamps at 0 — transmissions above the
    solve's Coulomb-only ceiling ``exp(-L_cath/l_bi)`` saturate there
    (stated limitation; exact for transmissions at or below the ceiling,
    including the quasilinear closure's ~0).
    """
    coulomb_model = str(input_dict.get("beam_coulomb_model", "fast_electron"))
    anomalous_model = str(input_dict.get("beam_anomalous_model", "none"))
    L_cath = float(device_config.L_cath)
    deposition = {}
    ends = (0, -1) if twin else (0,)
    for end in ends:
        result = beam_result.result if end == 0 else beam_result.result_twin
        if result is None or result.phi_c <= I_ion:
            deposition[end] = None
            continue
        launch, direction = beam_launch(geometry, end=end)
        Gamma0 = result.I_eth_star / qe_SI
        ray_kwargs = dict(
            nn=state.nn,
            ne=state.n,
            Te=derived.Te,
            launch=launch,
            direction=direction,
            I_ion_eV=float(I_ion),
            coulomb_model=coulomb_model,
            anomalous_model=anomalous_model,
        )
        if anomalous_model != "none":
            ray_kwargs["beam_area_cm2"] = geometry.plasma_area_cm2
        dep = deposit_beam(
            result.phi_c, Gamma0, dz_cm=geometry.length_cm, **ray_kwargs
        )
        deposition[end] = dep
        # Gap transmission: a second, gap-clipped ray (unit flux). CSDA
        # conserves flux until the stop point, so this is 1 if the range
        # exceeds L_cath and 0 otherwise.
        gap_dz = _clip_ray_length(
            geometry.length_cm, launch, direction, L_cath
        )
        gap = deposit_beam(result.phi_c, 1.0, dz_cm=gap_dz, **ray_kwargs)
        transmission = min(max(gap.transmitted_flux, 1.0e-6), 1.0)
        nn_launch = float(state.nn[launch])
        l_bi = _compute_l_b(
            result.phi_c,
            float(derived.Te[launch]),
            float(state.n[launch]),
            0.0,
            0.0,
        )
        sigma_eff = 0.0
        if nn_launch > 0.0 and L_cath > 0.0 and l_bi > 0.0:
            sigma_eff = max(
                0.0,
                (-math.log(transmission) / L_cath - 1.0 / l_bi) / nn_launch,
            )
        beam_result.beam_atten_cross[launch] = sigma_eff
    return deposition


def _clip_ray_length(length_cm, launch, direction, L_cath):
    """dz array truncated so the ray's total path is at most ``L_cath``."""
    dz = np.zeros_like(np.asarray(length_cm, dtype=float))
    remaining = float(L_cath)
    cells = dz.size
    order = range(launch, cells) if direction > 0 else range(launch, -1, -1)
    for cell in order:
        if remaining <= 0.0:
            break
        step = min(float(length_cm[cell]), remaining)
        dz[cell] = step
        remaining -= step
    return dz


def cathode_source_terms(
    state,
    floors,
    ion_mass_g,
    geometry,
    input_dict,
    input_flags,
    cathode_solve=None,
):
    """Return cathode surface particle and electron-power losses."""
    boundary = cathode_boundary_state(
        state=state,
        floors=floors,
        ion_mass_g=ion_mass_g,
        geometry=geometry,
        input_dict=input_dict,
        input_flags=input_flags,
    )
    zeros = np.zeros(geometry.cells, dtype=float)
    if (
        not boundary.enabled
        or cathode_solve is None
        or cathode_solve.beam_result is None
    ):
        return CathodeSourceTerms1D(
            rhs=ConservativeState1D(
                n=zeros,
                nn=zeros.copy(),
                M=zeros.copy(),
                Ee=zeros.copy(),
                Ei=zeros.copy(),
            ),
            enabled=boundary.enabled,
            mode=boundary.mode,
            metadata={
                "source_index": boundary.source.index,
                "end_index": boundary.end.index,
                "end_mode": boundary.end_mode,
                "twin_cathode": boundary.twin_cathode,
                "circuit": dict(boundary.circuit),
                "surface_particle_loss_s_inv": zeros.copy(),
            },
        )

    derived = derive_state(state, floors=floors, ion_mass_g=ion_mass_g)
    dN_loss = zeros.copy()
    # An absorbing cathode face already drains the plasma at the Bohm flux, which
    # is the same criterion the circuit's I_i is built from (A_c*e*n*c_s*exp(-0.5)
    # on this cell's n and Te), so the face and the circuit agree on the current.
    # Applying this volumetric loss as well would remove it twice. The electron
    # power loss below is a separate channel and still applies.
    face_absorbs = bool(
        np.any(np.asarray(getattr(geometry, "plasma_absorbing", ()), dtype=bool))
    )
    if not face_absorbs:
        dN_loss[0] = _cathode_particle_loss_rate(
            cathode_solve.beam_result.result,
            eta=input_dict["eta"],
        )
        if (
            boundary.twin_cathode
            and cathode_solve.beam_result.result_twin is not None
        ):
            dN_loss[-1] = _cathode_particle_loss_rate(
                cathode_solve.beam_result.result_twin,
                eta=input_dict["eta"],
            )

    plasma_loss_rate = dN_loss / geometry.plasma_volume_cm3
    # Recycle at the cathode surface feeds the COLUMN on a two-zone state
    # (recycle faces feed the column, NEUTRAL_TWOZONE_PLAN.md).
    neutral_gain_rate = dN_loss / (
        geometry.plasma_volume_cm3
        if state.nn_a is not None
        else geometry.neutral_volume_cm3
    )
    # Sheath electron power: P_cathode_e is lost at the cathode surface and
    # P_anode_e at the anode mesh (§8). Legacy has neither resolved, so both stay
    # colocated in its source cell exactly as before; resolved geometry lands each
    # at its own electrode.
    electron_power_loss_W = zeros.copy()
    cathode_cells = cathode_adjacent_cells(geometry)
    anode_pairs = anode_flanking_cells(geometry)
    if cathode_cells:
        _deposit_electrode_power(
            electron_power_loss_W,
            result=cathode_solve.beam_result.result,
            cathode_cell=int(cathode_cells[0]),
            anode_pair=anode_pairs[0] if anode_pairs else None,
            state=state,
            derived=derived,
        )
        if (
            boundary.twin_cathode
            and cathode_solve.beam_result.result_twin is not None
        ):
            _deposit_electrode_power(
                electron_power_loss_W,
                result=cathode_solve.beam_result.result_twin,
                cathode_cell=int(cathode_cells[-1]),
                anode_pair=anode_pairs[-1] if len(anode_pairs) > 1 else None,
                state=state,
                derived=derived,
            )
    else:
        electron_power_loss_W[0] = _electron_power_loss_W(
            cathode_solve.beam_result.result
        )
        if (
            boundary.twin_cathode
            and cathode_solve.beam_result.result_twin is not None
        ):
            electron_power_loss_W[-1] = _electron_power_loss_W(
                cathode_solve.beam_result.result_twin
            )
    electron_power_loss_density = electron_power_loss_W * 1.0e7 / (
        geometry.plasma_volume_cm3
    )
    return CathodeSourceTerms1D(
        rhs=ConservativeState1D(
            n=-plasma_loss_rate,
            nn=neutral_gain_rate,
            M=-ion_mass_g * derived.u * plasma_loss_rate,
            Ee=-electron_power_loss_density,
            Ei=-1.5 * ev_to_erg * derived.Ti * plasma_loss_rate,
        ),
        enabled=boundary.enabled,
        mode=boundary.mode,
        metadata={
            "source_index": boundary.source.index,
            "end_index": boundary.end.index,
            "end_mode": boundary.end_mode,
            "twin_cathode": boundary.twin_cathode,
            "circuit": dict(boundary.circuit),
            "surface_particle_loss_s_inv": dN_loss,
            "source_surface_particle_loss_s_inv": float(dN_loss[0]),
            "end_surface_particle_loss_s_inv": float(dN_loss[-1]),
            "electron_power_loss_W": electron_power_loss_W,
            "source_electron_power_loss_W": float(electron_power_loss_W[0]),
            "end_electron_power_loss_W": float(electron_power_loss_W[-1]),
        },
    )


def beam_launch(geometry, end=0):
    """Return the ``(cell, direction)`` a cathode's beam is launched from.

    The beam starts at the plasma cell against the cathode surface and travels
    into the machine, so in resolved geometry it must not begin at cell ``[0]``
    (the plenum) nor deposit into the cells behind the cathode.
    """
    cathode_cells = cathode_adjacent_cells(geometry)
    if not cathode_cells:
        return (0, 1) if end == 0 else (geometry.cells - 1, -1)
    if end == 0:
        return int(cathode_cells[0]), 1
    return int(cathode_cells[-1]), -1


def beam_absorption_weights(length_cm, l_b_profile, cathode_index, direction=None):
    """Return Beer-Lambert absorbed beam fractions for one cathode.

    The beam is launched from ``cathode_index`` and traverses away from it.
    ``direction`` is +1 for a beam heading toward increasing z and -1 for the
    other way; it is inferred for the legacy end cells. Cells *behind* the launch
    point get zero weight -- in resolved geometry those are the plenum and the
    obstruction, which the beam never enters (§5).
    """
    length_cm = np.asarray(length_cm, dtype=float)
    l_b_profile = np.asarray(l_b_profile, dtype=float)
    cells = length_cm.size
    if l_b_profile.shape != (cells,):
        raise ValueError(
            f"l_b_profile must have shape ({cells},), got {l_b_profile.shape}"
        )
    launch = int(cathode_index) % cells
    if direction is None:
        if launch == 0:
            direction = 1
        elif launch == cells - 1:
            direction = -1
        else:
            raise ValueError(
                "direction is required when the beam is not launched from an "
                f"end cell (got cathode_index={cathode_index})"
            )
    if direction > 0:
        order = np.arange(launch, cells)
    else:
        order = np.arange(launch, -1, -1)

    l_b_ordered = l_b_profile[order]
    dx_ordered = length_cm[order]
    safe_l_b = np.where(l_b_ordered > 0.0, l_b_ordered, np.inf)
    tau = np.cumsum(dx_ordered / safe_l_b)
    tau_in = np.concatenate([[0.0], tau[:-1]])
    exp_neg_tau_in = np.exp(-tau_in)
    absorbed_ordered = exp_neg_tau_in * (1.0 - np.exp(-dx_ordered / safe_l_b))

    weights = np.zeros(cells, dtype=float)
    weights[order] = absorbed_ordered
    return weights


def beam_ionization_rhs(
    state,
    floors,
    ion_mass_g,
    geometry,
    input_dict,
    input_flags,
    I_ion,
    cathode_solve=None,
):
    """Return conservative beam ionization and beam electron energy terms."""
    terms = beam_ionization_rhs_terms(
        state=state,
        floors=floors,
        ion_mass_g=ion_mass_g,
        geometry=geometry,
        input_dict=input_dict,
        input_flags=input_flags,
        I_ion=I_ion,
        cathode_solve=cathode_solve,
    )
    rhs = terms["beam_ionization_birth"]
    for term in (
        terms["beam_power_deposition"],
        terms["beam_ionization_cost"],
    ):
        rhs = ConservativeState1D(
            n=rhs.n + term.n,
            nn=rhs.nn + term.nn,
            M=rhs.M + term.M,
            Ee=rhs.Ee + term.Ee,
            Ei=rhs.Ei + term.Ei,
        )
    return rhs


def beam_ionization_rhs_terms(
    state,
    floors,
    ion_mass_g,
    geometry,
    input_dict,
    input_flags,
    I_ion,
    cathode_solve=None,
):
    """Return split beam ionization particle, power, and cost terms."""
    boundary = cathode_boundary_state(
        state=state,
        floors=floors,
        ion_mass_g=ion_mass_g,
        geometry=geometry,
        input_dict=input_dict,
        input_flags=input_flags,
    )
    zeros = np.zeros(geometry.cells, dtype=float)
    if (
        not boundary.enabled
        or cathode_solve is None
        or cathode_solve.beam_result is None
    ):
        return _zero_beam_terms(zeros)

    beam_derived = derive_state(state, floors=floors, ion_mass_g=ion_mass_g)
    E_exc = float(input_dict.get("beam_excitation_energy_eV", 21.218))
    S_beam, S_exc, S_exc_E, beam_power_density = _beam_ionization_sources(
        state=state,
        geometry=geometry,
        cathode_solve=cathode_solve,
        boundary=boundary,
        Te=beam_derived.Te,
        exc_energy_fallback_eV=E_exc,
    )
    volume_ratio = geometry.plasma_volume_cm3 / geometry.neutral_volume_cm3
    # Two-zone state (NEUTRAL_TWOZONE_PLAN.md): nn is the column density on
    # the plasma volume, so the beam's neutral debit converts by exactly 1
    # (the beam attenuates on column gas by construction).
    if state.nn_a is not None:
        volume_ratio = np.ones_like(volume_ratio)
    # In the kinetic-derived two-momentum reduction, beam ionization removes
    # a column neutral carrying u_c and births the ion with that same directed
    # momentum. Presence-gate on M_n_a so all historical M_n closures remain
    # bit-for-bit unchanged.
    if state.M_n_a is not None:
        u_birth = np.asarray(state.M_n, dtype=float) / (
            ion_mass_g
            * np.maximum(np.asarray(state.nn, dtype=float), floors["nn"])
        )
        beam_M_birth = ion_mass_g * u_birth * S_beam
        beam_Mn_debit = -beam_M_birth
        beam_Mna = np.zeros_like(state.M_n_a)
    else:
        beam_M_birth = zeros.copy()
        beam_Mn_debit = None
        beam_Mna = None
    Ti_birth = _birth_temperature(
        input_dict.get("Ti_birth_ionization", "floor"),
        beam_derived.Ti,
        floors["Ti"],
    )
    exc_model = str(input_dict.get("beam_excitation_model", "2p_scalar"))
    csda_active = getattr(cathode_solve, "beam_deposition", None) is not None
    if exc_model == "2p_scalar" and not csda_active:
        # Historical booking: the constant per-event energy factored out of
        # the summed event profile — kept byte-for-byte.
        exc_Ee = -E_exc * ev_to_erg * S_exc
    else:
        # Manifold booking: each ray radiates its own energy-weighted mean
        # per event, set by that cathode's phi_c at solve time; the CSDA
        # module's radiated bank uses the same channel per E(z), so it
        # always books through this path.
        exc_Ee = -ev_to_erg * S_exc_E
    return {
        "beam_ionization_birth": ConservativeState1D(
            n=S_beam,
            nn=-S_beam * volume_ratio,
            M=beam_M_birth,
            Ee=zeros.copy(),
            Ei=1.5 * ev_to_erg * Ti_birth * S_beam,
            M_n=beam_Mn_debit,
            nn_a=(
                np.zeros_like(state.nn_a)
                if state.nn_a is not None
                else None
            ),
            M_n_a=beam_Mna,
        ),
        "beam_power_deposition": ConservativeState1D(
            n=zeros,
            nn=zeros.copy(),
            M=zeros.copy(),
            Ee=beam_power_density,
            Ei=zeros.copy(),
        ),
        "beam_ionization_cost": ConservativeState1D(
            n=zeros,
            nn=zeros.copy(),
            M=zeros.copy(),
            Ee=-I_ion * ev_to_erg * S_beam,
            Ei=zeros.copy(),
        ),
        # Excited neutrals radiate their ~21-22 eV promptly (2^1P lifetime
        # ~ns; the 2^1S metastable share is booked as radiated too, caveat on
        # the manifold registry), so the excitation channel's energy leaves
        # the plasma as He I light rather than heating it. The particle is
        # unchanged: the neutral returns to ground state.
        "beam_excitation_radiation": ConservativeState1D(
            n=zeros.copy(),
            nn=zeros.copy(),
            M=zeros.copy(),
            Ee=exc_Ee,
            Ei=zeros.copy(),
        ),
    }


def _beam_ionization_sources(
    state,
    geometry,
    cathode_solve,
    boundary,
    Te=None,
    exc_energy_fallback_eV=21.218,
):
    zeros = np.zeros(geometry.cells, dtype=float)
    beam_result = cathode_solve.beam_result
    S_beam = zeros.copy()
    S_exc = zeros.copy()
    S_exc_E = zeros.copy()
    beam_power_density = zeros.copy()

    deposition = getattr(cathode_solve, "beam_deposition", None)
    if deposition is not None:
        # CSDA path (B2): the module already integrated each ray; convert
        # its per-cell totals to densities. ``beam_power_deposition`` carries
        # the whole per-cell beam energy (heating + radiated + cost) so the
        # separate cost and radiation sinks subtract to the module's net
        # heating, keeping the four-term decomposition meaningful. P_ohmic
        # keeps its historical gap-weighted booking.
        Vp = geometry.plasma_volume_cm3
        for end, dep in deposition.items():
            if dep is None:
                continue
            S_beam += dep.ionization_events / Vp
            S_exc += dep.excitation_events / Vp
            S_exc_E += dep.radiated_erg_s / Vp / ev_to_erg
            beam_power_density += (
                dep.plasma_heating_erg_s
                + dep.radiated_erg_s
                + dep.ionization_cost_erg_s
            ) / Vp
            solver_result = (
                beam_result.result if end == 0 else beam_result.result_twin
            )
            gap = np.asarray(gap_cell_indices(geometry, end=end), dtype=int)
            ohmic_weights = _ohmic_gap_weights(geometry, gap, Te)
            beam_power_density[gap] += (
                ohmic_weights * solver_result.P_ohmic * 1.0e7 / Vp[gap]
            )
        return S_beam, S_exc, S_exc_E, beam_power_density

    def _exc_energy_at(launch_index):
        # Per-ray radiated energy per event [eV]: the solve's per-cell value
        # when the builder provides it, else the legacy constant.
        energies = beam_result.beam_exc_energy_eV
        if energies is None:
            return exc_energy_fallback_eV
        return float(energies[launch_index])

    source_profile = _beam_ionization_profile(
        state=state,
        geometry=geometry,
        beam_result=beam_result,
        end=0,
    )
    S_beam += source_profile
    exc_profile = _beam_event_profile(
        state=state,
        geometry=geometry,
        beam_result=beam_result,
        event_cross=beam_result.beam_exc_cross,
        end=0,
    )
    S_exc += exc_profile
    S_exc_E += exc_profile * _exc_energy_at(beam_launch(geometry, end=0)[0])
    beam_power_density += _beam_power_deposition_density(
        geometry=geometry,
        beam_result=beam_result,
        solver_result=beam_result.result,
        end=0,
        Te=Te,
    )
    if boundary.twin_cathode and beam_result.result_twin is not None:
        twin_profile = _beam_ionization_profile(
            state=state,
            geometry=geometry,
            beam_result=beam_result,
            end=-1,
        )
        S_beam += twin_profile
        exc_profile_twin = _beam_event_profile(
            state=state,
            geometry=geometry,
            beam_result=beam_result,
            event_cross=beam_result.beam_exc_cross,
            end=-1,
        )
        S_exc += exc_profile_twin
        S_exc_E += exc_profile_twin * _exc_energy_at(
            beam_launch(geometry, end=-1)[0]
        )
        beam_power_density += _beam_power_deposition_density(
            geometry=geometry,
            beam_result=beam_result,
            solver_result=beam_result.result_twin,
            end=-1,
            Te=Te,
        )

    return S_beam, S_exc, S_exc_E, beam_power_density


def _zero_beam_terms(zeros):
    return {
        "beam_ionization_birth": ConservativeState1D(
            n=zeros,
            nn=zeros.copy(),
            M=zeros.copy(),
            Ee=zeros.copy(),
            Ei=zeros.copy(),
        ),
        "beam_power_deposition": ConservativeState1D(
            n=zeros.copy(),
            nn=zeros.copy(),
            M=zeros.copy(),
            Ee=zeros.copy(),
            Ei=zeros.copy(),
        ),
        "beam_ionization_cost": ConservativeState1D(
            n=zeros.copy(),
            nn=zeros.copy(),
            M=zeros.copy(),
            Ee=zeros.copy(),
            Ei=zeros.copy(),
        ),
        "beam_excitation_radiation": ConservativeState1D(
            n=zeros.copy(),
            nn=zeros.copy(),
            M=zeros.copy(),
            Ee=zeros.copy(),
            Ei=zeros.copy(),
        ),
    }


def _cell_state(index, state, derived, geometry):
    return CathodeCellState1D(
        index=int(index),
        role=str(geometry.cell_role[index]),
        n=float(state.n[index]),
        nn=float(state.nn[index]),
        Te=float(derived.Te[index]),
        Ti=float(derived.Ti[index]),
        u=float(derived.u[index]),
        plasma_volume_cm3=float(geometry.plasma_volume_cm3[index]),
        neutral_volume_cm3=float(geometry.neutral_volume_cm3[index]),
        plasma_area_cm2=float(geometry.plasma_area_cm2[index]),
        neutral_area_cm2=float(geometry.neutral_area_cm2[index]),
        length_cm=float(geometry.length_cm[index]),
        Rp_cm=float(geometry.Rp_cm[index]),
        Rm_cm=float(geometry.Rm_cm[index]),
    )


def _circuit_placeholders(input_dict):
    keys = (
        "V_bank",
        "T_s",
        "phi_wf",
        "C_R",
        "R_comp",
        "eta",
        "L_cath",
        "R_cath",
    )
    return {key: input_dict.get(key) for key in keys if key in input_dict}


def _cathode_particle_loss_rate(result, eta):
    return (1.0 + 2.0 * float(eta)) * result.I_i / qe_SI


def _deposit_electrode_power(
    electron_power_loss_W, result, cathode_cell, anode_pair, state, derived
):
    """Land P_cathode_e at the cathode cell and P_anode_e at the anode mesh.

    The anode collects on both mesh faces, so its sheath power is split between
    the two flanking cells in proportion to each face's Bohm collection -- the
    same weighting ``anode_collection_rhs`` uses, so power and particles are
    removed on the same side. With no resolved anode the whole of P_anode_e falls
    back to the cathode cell, which is where the lumped model puts it.
    """
    electron_power_loss_W[cathode_cell] += result.P_cathode_e
    if anode_pair is None:
        electron_power_loss_W[cathode_cell] += result.P_anode_e
        return
    gap_side, column_side = anode_pair
    # Bohm collection ~ n * c_s, and c_s ~ sqrt(Te/mu) with the same mu on both
    # sides, so mu cancels in the normalized split.
    weights = np.array(
        [
            state.n[gap_side] * np.sqrt(derived.Te[gap_side]),
            state.n[column_side] * np.sqrt(derived.Te[column_side]),
        ],
        dtype=float,
    )
    total = weights.sum()
    if not np.isfinite(total) or total <= 0.0:
        weights = np.full(2, 0.5)
    else:
        weights = weights / total
    electron_power_loss_W[gap_side] += weights[0] * result.P_anode_e
    electron_power_loss_W[column_side] += weights[1] * result.P_anode_e


def _electron_power_loss_W(result):
    return result.P_cathode_e + result.P_anode_e


def _beam_event_profile(state, geometry, beam_result, event_cross, end=0):
    """Per-cell rate density of one beam collision channel [cm^-3 s^-1].

    The beam attenuates along the Beer-Lambert profile set by the *total*
    inelastic mean free path (``l_b_profile``); ``l_b * sigma_event * nn`` is
    the fraction of absorbed primaries whose event is this channel, so the
    channels split the same absorbed flux rather than each attenuating
    independently.
    """
    launch, direction = beam_launch(geometry, end=end)
    cross = event_cross[launch]
    if cross == 0.0 or beam_result.beam_cross[launch] == 0.0:
        return np.zeros(geometry.cells, dtype=float)
    l_b_profile = (
        beam_result.l_b_profile if end == 0 else beam_result.l_b_profile_twin
    )
    p_event = l_b_profile * cross * state.nn
    weights = beam_absorption_weights(
        length_cm=geometry.length_cm,
        l_b_profile=l_b_profile,
        cathode_index=launch,
        direction=direction,
    )
    return (
        weights
        * p_event
        * beam_result.n_beam[launch]
        * beam_result.v_beam[launch]
        / geometry.length_cm
    )


def _beam_ionization_profile(state, geometry, beam_result, end=0):
    return _beam_event_profile(
        state=state,
        geometry=geometry,
        beam_result=beam_result,
        event_cross=beam_result.beam_cross,
        end=end,
    )


def _beam_power_deposition_density(
    geometry,
    beam_result,
    solver_result,
    end=0,
    Te=None,
):
    """Return the beam/ohmic power deposition density [erg cm^-3 s^-1].

    ``P_prim`` is carried into the column by the primary beam and so deposits
    along the Beer-Lambert absorption profile.

    ``P_ohmic = I^2 R_p`` is dissipated in the plasma *between* the cathode and
    the anode, so it is spread over the cathode-anode gap rather than piled into
    one boundary cell. The discharge current density is essentially uniform along
    the gap, so the power per unit length follows the local Spitzer resistivity,
    ``eta_sp ~ Te^-3/2``: dissipation concentrates wherever the gap is coldest.
    Legacy geometry has no resolved gap, so the whole of ``P_ohmic`` still lands
    on the single source/end cell exactly as before.
    """
    launch, direction = beam_launch(geometry, end=end)
    beam_cross = beam_result.beam_cross[launch]
    if beam_cross == 0.0:
        return np.zeros(geometry.cells, dtype=float)
    l_b_profile = (
        beam_result.l_b_profile if end == 0 else beam_result.l_b_profile_twin
    )
    weights = beam_absorption_weights(
        length_cm=geometry.length_cm,
        l_b_profile=l_b_profile,
        cathode_index=launch,
        direction=direction,
    )
    density = (
        weights * solver_result.P_prim * 1.0e7 / geometry.plasma_volume_cm3
    )
    gap = np.asarray(gap_cell_indices(geometry, end=end), dtype=int)
    ohmic_weights = _ohmic_gap_weights(geometry, gap, Te)
    density[gap] += (
        ohmic_weights
        * solver_result.P_ohmic
        * 1.0e7
        / geometry.plasma_volume_cm3[gap]
    )
    return density


def _ohmic_gap_weights(geometry, gap, Te):
    """Return the normalized share of ``P_ohmic`` deposited in each gap cell.

    ``P_cell = j^2 * eta_sp * V_cell``; with the current density uniform along
    the gap this reduces to ``P_cell ~ eta_sp * length``, and Spitzer resistivity
    gives ``eta_sp ~ Te^-3/2``. A single-cell gap normalizes to exactly 1.0, so
    legacy deposition is bit-identical.
    """
    lengths = np.asarray(geometry.length_cm, dtype=float)[gap]
    if Te is None or gap.size == 1:
        weights = lengths
    else:
        Te_gap = np.maximum(np.asarray(Te, dtype=float)[gap], 1e-30)
        weights = lengths * Te_gap**-1.5
    total = weights.sum()
    if not np.isfinite(total) or total <= 0.0:
        return np.full(gap.size, 1.0 / gap.size)
    return weights / total


def _solver_result_metadata(result):
    if result is None:
        return None
    keys = (
        "phi_c",
        "phi_a",
        "V_b",
        "I_i",
        "I_eth_star",
        "I_tot",
        "P_prim",
        "P_ohmic",
        "P_loss",
        "P_cathode_e",
        "P_anode_e",
        "beam_bypass_fraction",
        "l_b",
    )
    metadata = {key: float(getattr(result, key)) for key in keys}
    metadata["regime"] = result.regime
    metadata["long_mfp"] = bool(result.long_mfp)
    return metadata
