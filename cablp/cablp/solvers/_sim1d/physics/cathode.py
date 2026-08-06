import dataclasses
from dataclasses import dataclass
import hashlib
import math

import numpy as np

from scipy.optimize import brentq

from cablp.funcs._beam_deposition import (
    deposit_beam,
    BeamDepositionResult,
    _coulomb_stopping_coefficient,
)
from cablp.funcs._cathode_solver import (
    DeviceConfig,
    PlasmaState,
    _compute_beam_bypass_fraction,
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
from .sources import electrode_sheath_alpha


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
    # Per-end ``(probe, ray, circuit)`` gap survival for the item-35 ledger
    # tripwire; keyed only for ends with an active CSDA ray. See
    # ``beam_gap_ledger_mismatch``.
    beam_gap_ledger: dict | None = None


def anode_circuit_sample(state, derived, geometry, mu, input_dict, end=0):
    """Return ``(I_i_a [A], Te_anode [eV])`` for one anode, or ``(None, None)``.

    The historical circuit takes ``I_i_a = 2*eta*I_i``, scaling the anode
    current straight off the *cathode* cell, which assumes both electrodes see the
    same plasma -- precisely what a resolved cathode-anode gap breaks.

    The current handed back is the same Bohm collection
    ``sources.anode_collection_rhs`` removes from the fluid, summed over both mesh
    faces with each face sampled on its own side. Computing it once and sharing it
    means the circuit and the fluid cannot disagree about the anode current, and it
    is why M5 must not add a second anode particle sink.

    The sheath temperature is collection-weighted across the two faces, matching
    how ``P_anode_e`` is apportioned. Resolving a *separate* sheath per face is
    a known open item.
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


def cathode_circuit_alpha_sheath(
    state, derived, geometry, cathode_index, mu, ion_mass_g, input_dict, input_flags
):
    """Return the cathode sheath-edge factor ``n_se/n`` for the circuit, or None.

    R3.2 (A16): under the unified-sampling stance
    (``characteristic_boundary``), the circuit's cathode ion current must be
    drawn at the SAME sheath-edge density the fluid boundary uses, so both call
    ``sources.electrode_sheath_alpha`` on the same cathode-adjacent cell (verified
    identical: ``beam_launch(geometry)[0]`` == the source cathode's live cell).
    Returns ``None`` off-stance so ``solve_idriven`` keeps its exact flat
    ``exp(-1/2)`` and the golden / M2 equivalence gate stay bit-exact. The anode
    is not sampled here -- its geometric mesh presheath stays flat ``exp(-1/2)``.
    """
    if not bool(input_flags.get("characteristic_boundary", False)):
        return None
    return electrode_sheath_alpha(
        nn=float(state.nn[cathode_index]),
        Te=float(derived.Te[cathode_index]),
        Ti=float(derived.Ti[cathode_index]),
        cell_length_cm=float(geometry.length_cm[cathode_index]),
        mu=mu,
        ion_mass_g=ion_mass_g,
        alpha_isat=float(input_dict.get("alpha_isat", math.exp(-0.5))),
        b_presheath_length=float(input_dict.get("b_presheath_length", 1.0)),
        sigma_in_cm2=float(input_dict.get("sigma_in_cm2", 5.0e-15)),
        sigma_in_model=str(input_dict.get("sigma_in_model", "constant")),
        gas_type=input_dict.get("gas_type", "He"),
    )


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
        R_comp_partition=float(input_dict.get("R_comp_partition", 1.0)),
        R_mesh_ohm=float(input_dict.get("R_mesh_ohm", 0.0)),
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
    (M1b):

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
            f"cathode_solver_model must be 'current_driven' (got {model!r})"
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
    alpha_sheath = cathode_circuit_alpha_sheath(
        state, derived, geometry, idx, mu, ion_mass_g, input_dict, input_flags
    )

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
            alpha_sheath=alpha_sheath,
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

    # Internal series drop on the plasma side of the V_dis probe (R5 ES1 tuning
    # pass, 2026-07-26). R_comp is split by the probe: R_external = x*R_comp
    # (bank side, in V_dis) and R_internal = (1-x)*R_comp (probe->plasma), plus a
    # separate anode-mesh R_mesh_ohm also on the plasma side. The circuit
    # integrates the DEVICE voltage V_b + I*(R_internal + R_mesh), while V_dis =
    # V_bank - I*R_external (see advance_circuit's R_comp_ohm = x*R_comp).
    #
    # CORRECTION (2026-08-03): this comment used to say the internal drop
    # "lowers the current, which RAISES V_dis". That is FALSE for the
    # (1-x)*R_comp part -- x CANCELS IDENTICALLY from the loop equation.
    # advance_circuit_current_driven integrates
    #     f(I) = (V_src - I*x*R_comp - vdis_of_I(I)) / L
    #          = (V_src - I*R_comp - V_b(I) - I*R_mesh) / L,
    # so the current sees only the TOTAL R_comp plus R_mesh; x is gone. What x
    # changes is the REPORTED V_dis, relabelling the same drop between the
    # external and internal books (dV_dis/dx = -I*R_comp). The claim IS true of
    # R_mesh, which is genuinely additional series resistance -- so real
    # internal resistance goes in R_mesh_ohm, never in the partition. Defaults
    # (x=1, R_mesh=0) give R_internal_total = 0 -> device voltage = V_b,
    # bit-exact.
    x = float(input_dict.get("R_comp_partition", 1.0))
    R_comp = float(input_dict.get("R_comp", 0.0))
    R_internal_total = (1.0 - x) * R_comp + float(
        input_dict.get("R_mesh_ohm", 0.0)
    )

    def vdis(I_A):
        return solve_at(I_A).V_b + I_A * R_internal_total

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
    load-bearing design decision (revised 2026-07-20): a
    frozen-V_dis explicit step needs ``dV/dI < 2L/dt ~ 22 mOhm`` at
    production dt, and the measured device slope near the emission ceiling
    is 0.2 Ohm-0.75 MOhm -- explicit would sawtooth exactly where this
    machine operates. TR-BDF2 because the RLC gate demands 2nd order and
    TR alone would ring against the near-vertical branch (L-stability, the
    same argument as the heat-conduction scheme choice).

    ``I >= 0`` is enforced per stage (the plasma-diode stand-in): a stage
    whose unconstrained root is negative clamps to 0.
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
        # shared constant, changed in one place.
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
            alpha_sheath=cathode_circuit_alpha_sheath(
                state, derived, geometry, beam_launch(geometry, end=0)[0],
                mu, ion_mass_g, input_dict, input_flags,
            ),
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
            # removes, so the two cannot disagree.
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
    beam_gap_ledger = None
    if str(input_dict.get("beam_deposition_model", "beer_lambert")) == "csda":
        beam_deposition, beam_gap_ledger = _csda_beam_deposition(
            beam_result=beam_result,
            state=state,
            derived=derived,
            geometry=geometry,
            device_config=device_config,
            input_dict=input_dict,
            I_ion=I_ion,
            twin=boundary.twin_cathode,
            anode_interception=bool(
                input_flags.get("beam_anode_interception", False)
            ),
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
        beam_gap_ledger=beam_gap_ledger,
    )


def _sum_beam_deposition(a, b):
    """Sum two CSDA beam rays (fractional-coverage: gap + clump).

    Per-cell deposition arrays add; the beam is energy-limited per ray so the
    combined totals stay bounded by ``Gamma0*E0``. Scalar exit diagnostics are
    flux-combined (unused downstream, kept coherent).
    """
    tf = float(a.transmitted_flux) + float(b.transmitted_flux)
    if tf > 0.0:
        te = (a.transmitted_flux * a.transmitted_energy_eV
              + b.transmitted_flux * b.transmitted_energy_eV) / tf
    else:
        te = 0.0
    return BeamDepositionResult(
        ionization_events=a.ionization_events + b.ionization_events,
        excitation_events=a.excitation_events + b.excitation_events,
        plasma_heating_erg_s=a.plasma_heating_erg_s + b.plasma_heating_erg_s,
        radiated_erg_s=a.radiated_erg_s + b.radiated_erg_s,
        ionization_cost_erg_s=a.ionization_cost_erg_s + b.ionization_cost_erg_s,
        transmitted_flux=tf,
        transmitted_energy_eV=te,
        anode_intercepted_erg_s=(float(a.anode_intercepted_erg_s)
                                 + float(b.anode_intercepted_erg_s)),
        # End ledger (WP-D): both rays leave through the same two ends, so
        # the escaping powers add like every other per-ray bank.
        end_loss_low_erg_s=(float(a.end_loss_low_erg_s)
                            + float(b.end_loss_low_erg_s)),
        end_loss_high_erg_s=(float(a.end_loss_high_erg_s)
                             + float(b.end_loss_high_erg_s)),
        end_loss_transmitted_erg_s=(float(a.end_loss_transmitted_erg_s)
                                    + float(b.end_loss_transmitted_erg_s)),
        # Tail end ledger (WP-E): same argument as the WP-D pair above -- both
        # rays' QL tails leave through the same two ends, so the escaping
        # powers add.
        end_loss_tail_low_erg_s=(float(a.end_loss_tail_low_erg_s)
                                 + float(b.end_loss_tail_low_erg_s)),
        end_loss_tail_high_erg_s=(float(a.end_loss_tail_high_erg_s)
                                  + float(b.end_loss_tail_high_erg_s)),
        E_entry_eV=np.maximum(a.E_entry_eV, b.E_entry_eV),
        # Diagnostic heating splits add like the lumped bank they partition.
        heating_coulomb_erg_s=(a.heating_coulomb_erg_s
                               + b.heating_coulomb_erg_s),
        heating_anomalous_erg_s=(a.heating_anomalous_erg_s
                                 + b.heating_anomalous_erg_s),
        heating_secondary_erg_s=(a.heating_secondary_erg_s
                                 + b.heating_secondary_erg_s),
        heating_terminal_erg_s=(a.heating_terminal_erg_s
                                + b.heating_terminal_erg_s),
        # K6 tail splits add like the banks they partition.
        ionization_events_tail=(a.ionization_events_tail
                                + b.ionization_events_tail),
        excitation_events_tail=(a.excitation_events_tail
                                + b.excitation_events_tail),
        ionization_cost_tail_erg_s=(a.ionization_cost_tail_erg_s
                                    + b.ionization_cost_tail_erg_s),
        radiated_tail_erg_s=(a.radiated_tail_erg_s + b.radiated_tail_erg_s),
    )


def _plasma_active_window(geometry):
    """Return the inclusive ``(lo, hi)`` cell range the plasma occupies.

    The K6 tail walkers may traverse exactly these cells: outside them the
    solver's active-plasma mask zeroes every row, so a walk there deposits and
    births into nothing. Resolved geometry has one contiguous live run (the
    plenum and obstruction sit behind the cathode at the low end); a
    geometry with the live region split into several runs has no single window
    and is refused rather than silently walked across the gap.
    """
    active = np.asarray(geometry.plasma_active, dtype=bool)
    live = np.flatnonzero(active)
    if live.size == 0:
        raise ValueError(
            "no plasma-active cells: the QL tail walk has nowhere to go"
        )
    lo, hi = int(live[0]), int(live[-1])
    if not active[lo : hi + 1].all():
        raise ValueError(
            "plasma-active cells are not contiguous "
            f"({np.flatnonzero(~active[lo : hi + 1]) + lo}); the tail walk "
            "window is a single inclusive range and cannot describe this "
            "topology"
        )
    return lo, hi


def tail_reflect_face(geometry, end=0):
    """Return which walk-window face the cathode at ``end`` occupies (K7).

    ``-1`` for the window's low-index face, ``+1`` for its high-index face:
    the face BEHIND the ray, since the beam is launched from the cathode into
    the machine. That face is the one
    ``heating_anomalous_tail_cathode_boundary="reflect"`` turns walkers around
    at, so it must be the face the cathode actually sits at. A geometry whose
    ray is launched from somewhere other than the window's face cell is refused
    rather than having its reflection applied at a face that is not a cathode.
    """
    lo, hi = _plasma_active_window(geometry)
    launch, direction = beam_launch(geometry, end=end)
    face = -1 if direction > 0 else 1
    face_cell = lo if face < 0 else hi
    if int(launch) != int(face_cell):
        raise ValueError(
            f"the cathode ray for end {end} is launched from cell {launch}, "
            f"which is not the face cell {face_cell} of the plasma-active "
            f"window {(lo, hi)}; sheath reflection turns tail walkers around "
            "at that face, so it has to be the face the cathode occupies"
        )
    return face


def _csda_beam_deposition(
    beam_result,
    state,
    derived,
    geometry,
    device_config,
    input_dict,
    I_ion,
    twin=False,
    anode_interception=False,
):
    """Run the CSDA module for each active cathode ray (B2 wiring).

    Returns ``(deposition, gap_ledger)``. ``deposition`` is
    ``{0: BeamDepositionResult | None, -1: ...}``; ``gap_ledger`` maps each
    end with an active ray to ``(probe, ray, circuit)`` gap survival for the
    item-35 tripwire (see ``beam_gap_ledger_mismatch``). The call also
    rewrites
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

    The gap-transmission probe is FLUX-FAITHFUL: it launches the same total
    flux as the deposition rays above, split the same way when clumping is
    active, so flux-dependent stopping (the quasilinear closure, whose
    relaxation length runs on n_b ~ Gamma0/(A v_b)) is felt by the probe
    exactly as the deposition ray feels it. Transmission is then the ratio of
    total transmitted to total launched flux. It was historically launched at
    unit flux, which made the quasilinear closure invisible to the circuit:
    transmission read 1, ``sigma_eff`` wrote 0, and the circuit kept booking
    ``eta * f_bypass`` of the emitted beam power as never-coupling while the
    real ray stopped inside the gap (item 35).

    ``anode_interception`` (R4.1, audit A15): when set, the mesh solid fraction
    ``device_config.eta`` of the beam surviving the gap is intercepted at the
    anode-face crossing (``deposit_beam(anode_cross_index=..., anode_eta=...)``),
    so the fluid stops depositing the ~164 kW long-mfp beam the circuit already
    books as never entering the plasma. The gap-transmission probe is
    unaffected (it measures gap survival, which feeds the circuit bypass).

    ``beam_product_transport`` (WP-D), ``heating_anomalous_transport`` (WP-E)
    and ``heating_anomalous_tail_ionization`` (K6) are threaded to the
    DEPOSITION rays only, and only when they are
    not their default ``"local"`` / ``"off"``. The probe rays keep the
    historical argument
    list: they are transmission instruments whose single output is the ratio
    of transmitted to launched PRIMARY flux, which no closure here can change
    (the walks move deposited energy and, under K6, add SECONDARY events --
    never the primary's own flux),
    so walking their products or tails would be pure cost. For the same reason
    ``_ray_gap_breakout`` and the item-35 tripwire are unaffected -- both read
    ``transmitted_flux`` and ``E_entry_eV``, which are primary-flux
    quantities the walks never touch.
    """
    coulomb_model = str(input_dict.get("beam_coulomb_model", "fast_electron"))
    anomalous_model = str(input_dict.get("beam_anomalous_model", "none"))
    # WP-D product transport and WP-E QL heating locality. Presence-gated:
    # only the DEPOSITION rays get the keywords, and only when they are not
    # their defaults, so the off path enters deposit_beam with the identical
    # argument list it always had. The gap-transmission PROBE rays below
    # deliberately never receive them -- they are transmission instruments
    # whose only output is a primary-flux ratio, so walking their products or
    # tails would be wasted work and would not change the number they report.
    transport_kwargs = {}
    product_transport = str(input_dict.get("beam_product_transport", "local"))
    if product_transport != "local":
        transport_kwargs["product_transport"] = product_transport
    anomalous_transport = str(
        input_dict.get("heating_anomalous_transport", "local")
    )
    # K7: whether the tail birth energy is keyed to the live phi_c, and
    # whether the cathode face reflects. Both are per-RAY quantities (phi_c is
    # the ray's own accelerating drop and the reflecting face is its own
    # cathode's), so they are resolved in the loop below; here we only decide
    # whether the loop has to do anything at all, which keeps the legacy arms
    # on the identical dict object and therefore bit-exact.
    tail_keying = "fixed"
    tail_phi_fraction = 0.0
    tail_reflect = False
    if anomalous_transport != "local":
        transport_kwargs["anomalous_transport"] = anomalous_transport
        # Read ONLY under tail_walk (the keys are inert otherwise, by design);
        # the solver validated them at construction time.
        tail_keying = str(
            input_dict.get("heating_anomalous_tail_energy_keying", "phi_c")
        )
        if tail_keying == "fixed":
            transport_kwargs["tail_energy_eV"] = float(
                input_dict.get("heating_anomalous_tail_energy_eV", 75.0)
            )
        else:
            _phi_frac = input_dict.get(
                "heating_anomalous_tail_phi_c_fraction", None
            )
            tail_phi_fraction = 0.25 if _phi_frac is None else float(_phi_frac)
        # K6: presence-gated inside the tail_walk branch, because the module
        # refuses the combination the solver has already refused at
        # construction. Passed only when ON, so a tail_walk run without it
        # enters deposit_beam with the argument list it had before K6.
        tail_ionization = str(
            input_dict.get("heating_anomalous_tail_ionization", "off")
        )
        if tail_ionization != "off":
            transport_kwargs["tail_ionization"] = tail_ionization
            # The walk window the module refuses to default (see its
            # docstring): the maximal contiguous PLASMA-ACTIVE run, which in
            # resolved geometry starts at the cathode cell -- so the cathode
            # disc and the obstruction/plenum behind it are a wall to a tail
            # electron, and no pair is born into a row the RHS mask zeroes.
            # Derived from ``geometry.plasma_active`` rather than from the
            # cathode roles, because "the cells whose plasma rows the solver
            # integrates" is exactly the property that matters here, and it is
            # the same array the mask itself is built from.
            transport_kwargs["tail_walk_window"] = _plasma_active_window(
                geometry
            )
        if str(
            input_dict.get(
                "heating_anomalous_tail_cathode_boundary", "reflect"
            )
        ) != "escape":
            # The cathode reflects, so the walk needs the window whether or not
            # the ionizing channel is on: the reflecting face is one of the
            # window's faces, and under "escape" the energy-only walk has no
            # face there at all (it runs the whole grid).
            tail_reflect = True
            transport_kwargs["tail_walk_window"] = _plasma_active_window(
                geometry
            )
    if transport_kwargs:
        # Hoisted stopping coefficient (cost read 2026-08-02, restructure C).
        # The walks' per-cell A in dE/dx = A W**p is a 262-iteration Python
        # listcomp costing ~100 us -- half the whole WP-E per-call surcharge --
        # and it depends only on (ne, Te, model), which are the SAME for every
        # deposition ray in this call: both cathode ends under TwinCathode,
        # both halves of the clumping split (which varies nn alone), and every
        # energy group a future WP-F build adds. Build it once here rather than
        # once per ray. Bit-exact: it is the module's own function on the same
        # inputs, and it is presence-gated behind an active walk closure, so a
        # default-stance run never reaches this line.
        transport_kwargs["stopping_coefficient"] = (
            _coulomb_stopping_coefficient(
                state.n, derived.Te, coulomb_model
            )
        )
    # Fractional-coverage beam-neutral closure (default off/uniform, bit-exact):
    # split the ray into a clump fraction (short l_b against nn*chi -> local seed)
    # and a gap fraction (background nn -> penetration). See config docstrings.
    f_clump = float(input_dict.get("beam_clump_fraction", 0.0))
    chi_clump = float(input_dict.get("beam_clump_enhancement", 1.0))
    clumping = f_clump > 0.0 and chi_clump > 1.0
    L_cath = float(device_config.L_cath)
    eta = float(device_config.eta)
    anode_faces = np.asarray(
        getattr(geometry, "anode_face_indices", ()), dtype=int
    )
    deposition = {}
    gap_ledger = {}
    ends = (0, -1) if twin else (0,)
    for end in ends:
        result = beam_result.result if end == 0 else beam_result.result_twin
        if result is None or result.phi_c <= I_ion:
            deposition[end] = None
            continue
        launch, direction = beam_launch(geometry, end=end)
        Gamma0 = result.I_eth_star / qe_SI
        # K7, per ray: phi_c is THIS cathode's accelerating drop -- the same
        # quantity the ray is launched at and the same one the sheath repels
        # returning electrons with -- so both the keyed birth energy and the
        # reflection threshold come from it. Left as the shared dict when
        # neither correction is engaged, so the legacy arms pass the identical
        # object they always did.
        ray_transport = transport_kwargs
        if tail_keying != "fixed" or tail_reflect:
            ray_transport = dict(transport_kwargs)
            if tail_keying != "fixed":
                ray_transport["tail_energy_eV"] = (
                    tail_phi_fraction * float(result.phi_c)
                )
            if tail_reflect:
                ray_transport["tail_reflect_face"] = tail_reflect_face(
                    geometry, end=end
                )
                ray_transport["tail_reflect_threshold_eV"] = float(
                    result.phi_c
                )
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
        interception_kwargs = {}
        if anode_interception and eta > 0.0 and anode_faces.size > 0:
            # The ray crosses the anode face between cell ``f-1`` and cell ``f``;
            # the first cell on the far (column) side along the ray is the
            # cross cell (``f`` when heading +z, ``f-1`` when heading -z).
            anode_face = int(anode_faces[0] if end == 0 else anode_faces[-1])
            cross_cell = anode_face if direction > 0 else anode_face - 1
            interception_kwargs = dict(
                anode_cross_index=cross_cell, anode_eta=eta
            )
        clump_kwargs = (
            {**ray_kwargs, "nn": np.asarray(state.nn) * chi_clump}
            if clumping
            else None
        )
        # The gap's per-cell path length, shared by the probe below and by the
        # deposition ray's own breakout test.
        gap_dz = _clip_ray_length(
            geometry.length_cm, launch, direction, L_cath
        )
        if clumping:
            # Gap ray: background nn, penetrates (the fast far-end pedestal).
            gap_ray = deposit_beam(
                result.phi_c, (1.0 - f_clump) * Gamma0,
                dz_cm=geometry.length_cm,
                **ray_kwargs, **interception_kwargs, **ray_transport,
            )
            # Clump ray: enhanced nn -> short l_b -> local deposit (front seed).
            clump_ray = deposit_beam(
                result.phi_c, f_clump * Gamma0,
                dz_cm=geometry.length_cm,
                **clump_kwargs, **interception_kwargs, **ray_transport,
            )
            dep = _sum_beam_deposition(gap_ray, clump_ray)
            # Breakout is per-ray: the clump ray can die in the gap while the
            # gap ray penetrates, so the split's gap survival is the
            # flux-weighted mean. (`dep` cannot answer this -- it carries the
            # elementwise MAX of the two E_entry profiles.)
            ray_survival = (
                (1.0 - f_clump)
                * _ray_gap_breakout(gap_ray, gap_dz, launch, direction)
                + f_clump
                * _ray_gap_breakout(clump_ray, gap_dz, launch, direction)
            )
        else:
            dep = deposit_beam(
                result.phi_c, Gamma0, dz_cm=geometry.length_cm,
                **ray_kwargs, **interception_kwargs, **ray_transport,
            )
            ray_survival = _ray_gap_breakout(dep, gap_dz, launch, direction)
        deposition[end] = dep
        # Gap transmission: gap-clipped probe rays MIRRORING the deposition
        # above -- same launched fluxes, same clump split, same nn per ray --
        # truncated at L_cath. Launching at the real Gamma0 (rather than the
        # historical unit flux) is what makes flux-DEPENDENT stopping visible
        # to the circuit: the quasilinear relaxation length runs on the beam
        # density n_b ~ Gamma0/(A v_b), so a unit-flux probe feels no
        # anomalous drag, reads transmission 1, writes sigma_eff = 0, and
        # leaves the circuit booking a bypass the real ray never enjoys
        # (item 35, root-caused 2026-07-27). Under flux-INDEPENDENT stopping
        # (Coulomb CSDA, anomalous_model="none") the ray is flux-linear, so
        # the ratio below is bit-for-bit the historical unit-flux value.
        #
        # --- Probe skip (cost read 2026-08-02, restructure A) --------------
        # The probe is a SECOND full CSDA march and measures ~50% of the whole
        # deposit_beam subsystem, yet in the main discharge it re-derives an
        # answer ``_ray_gap_breakout`` has already given from the deposition
        # ray's own bookkeeping. When that reads 0.0 the ray was ABSORBED
        # inside the gap, and the probe -- the same ray over the same per-cell
        # path lengths, merely stopped at L_cath -- is absorbed at the same
        # point, so ``BeamDepositionResult.transmitted_flux`` is the literal
        # float ``0.0`` (``0.0 if absorbed else gamma``). ``survival`` is then
        # ``0.0 / launched``, exactly 0.0 for any finite positive launch, and
        # ``transmission`` the 1e-6 clamp below. This is an EXACT-ZERO
        # argument, not a tolerance: the skipped branch writes the same floats
        # the probe would have returned, so every downstream number --
        # sigma_eff, the ledger, the tripwire -- is bit-identical.
        #
        # Three conditions break the identity, and under any of them the probe
        # runs exactly as it always has:
        #
        #   clumping     the split launches TWO probes with two different nn
        #                profiles and sums their transmitted fluxes, while
        #                ``ray_survival`` is the flux-weighted mean of two
        #                per-ray breakouts. A zero mean does imply both rays
        #                died, but the two-ray path is left untouched rather
        #                than re-argued: it is off in production.
        #   partial clip ``_clip_ray_length`` truncates the cell L_cath ends
        #                in when the gap does not end on a cell face. The
        #                deposition ray then has MORE path in that cell than
        #                the probe and can die inside it while the probe runs
        #                out of dz and transmits. Production has
        #                ``5 x 10 cm == L_cath`` exactly, so this never binds
        #                there, but the guard is on the general geometry.
        #   anode in gap anode-mesh interception scales the DEPOSITION ray's
        #                flux at the anode-face crossing and the probe's not
        #                at all, so under flux-DEPENDENT stopping (the
        #                quasilinear closure) the two trajectories would part
        #                company. The anode sits past the gap in every
        #                campaign geometry -- the guard is free there -- but
        #                nothing in this function enforces that.
        #
        # The ``Gamma0 == 0`` unit-flux probe is a DIFFERENT measurement (the
        # flux-independent limit; item 35) with no deposition ray behind it to
        # read, so it sits outside this branch and keeps running verbatim.
        probe_transmits_exact_zero = (
            not clumping
            and ray_survival == 0.0
            and _gap_clip_is_face_aligned(gap_dz, geometry.length_cm)
            and not (
                interception_kwargs
                and float(gap_dz[interception_kwargs["anode_cross_index"]])
                > 0.0
            )
        )
        if Gamma0 > 0.0:
            if clumping:
                gap_launch = (1.0 - f_clump) * Gamma0
                clump_launch = f_clump * Gamma0
                transmitted = (
                    float(deposit_beam(
                        result.phi_c, gap_launch,
                        dz_cm=gap_dz, **ray_kwargs,
                    ).transmitted_flux)
                    + float(deposit_beam(
                        result.phi_c, clump_launch,
                        dz_cm=gap_dz, **clump_kwargs,
                    ).transmitted_flux)
                )
                # Sum the LAUNCHED fluxes the same way the transmitted ones
                # are summed, so a fully-transmitting split lands on exactly
                # 1.0 instead of (1-f)+f rounding a ulp off it.
                launched = gap_launch + clump_launch
            elif probe_transmits_exact_zero:
                # The probe would be absorbed exactly where the deposition ray
                # was; skip the march and take the float it would have
                # returned. (Not an approximation of the probe -- its value.)
                transmitted = 0.0
                launched = Gamma0
            else:
                transmitted = float(
                    deposit_beam(
                        result.phi_c, Gamma0, dz_cm=gap_dz, **ray_kwargs
                    ).transmitted_flux
                )
                launched = Gamma0
            survival = transmitted / launched
        else:
            # No emission this frame: the flux-weighted ratio is 0/0. The
            # Gamma0 -> 0 limit of any flux-dependent stopping is the
            # flux-INDEPENDENT transmission, which is exactly what the
            # historical unit-flux probe measures, so keep it verbatim here.
            survival = float(
                deposit_beam(
                    result.phi_c, 1.0, dz_cm=gap_dz, **ray_kwargs
                ).transmitted_flux
            )
        transmission = min(max(survival, 1.0e-6), 1.0)
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
        # --- Ledger tripwire (item 35) ---------------------------------
        # Three views of ONE number -- the fraction of the emitted beam that
        # crosses the cathode-anode gap -- which must agree, and which nothing
        # else in the model compares:
        #
        #   probe    the gap-clipped probe's transmission (feeds sigma_eff)
        #   ray      the DEPOSITION ray's own breakout, read off its internal
        #            bookkeeping and completely independent of the probe
        #   circuit  what the circuit reconstructs from the sigma_eff just
        #            written, built with the CIRCUIT's own functions rather
        #            than by inverting the adapter's algebra
        #
        # ``probe`` vs ``ray`` catches a defect INSIDE the probe -- the
        # item-35 class, where the probe misreports the ray it is supposed to
        # mirror. ``ray`` vs ``circuit`` catches the circuit failing to
        # represent the ray, whatever the cause: adapter clamp saturation, or
        # a broken probe that the adapter faithfully propagated. Item 35 sat
        # silently in the second: the circuit booked ~97% gap survival while
        # the deposition ray delivered 0.
        gap_ledger[end] = (
            transmission,
            ray_survival,
            _compute_beam_bypass_fraction(
                _compute_l_b(
                    result.phi_c,
                    float(derived.Te[launch]),
                    float(state.n[launch]),
                    nn_launch,
                    sigma_eff,
                ),
                L_cath,
            ),
        )
    return deposition, gap_ledger


def _ray_gap_breakout(dep, gap_dz, launch, direction):
    """Fraction of a CSDA ray's flux that crosses the gap: 1.0 or 0.0.

    Probe-independent: it reads only the deposition ray's own bookkeeping.
    A CSDA ray carries its flux unattenuated until it stops (the anode mesh
    is the one exception, and it sits past the gap), so a single ray either
    crosses the gap whole or dies inside it -- there is no partial survival
    to measure. Fractional survival across the clumping split is handled by
    the caller, which weights the two rays' breakouts by their launched flux.

    ``dep.E_entry_eV`` is written for every cell the ray ENTERS and left at
    zero for cells it never reached, so a positive entry energy in the first
    cell beyond the gap means the ray got out. Reading entry energy (rather
    than deposited energy) is what makes this exact even when the gap ends
    mid-cell: entry energy is sampled before any of that cell's path is
    consumed, so truncating the last gap cell cannot perturb it.
    """
    # Reached the far end of the domain, so it certainly cleared the gap.
    # Also covers the sub-threshold ray, which passes through untouched and
    # leaves E_entry all zeros.
    if float(dep.transmitted_flux) > 0.0:
        return 1.0
    E_entry = np.asarray(dep.E_entry_eV, dtype=float)
    cells = gap_dz.size
    order = range(launch, cells) if direction > 0 else range(launch, -1, -1)
    for cell in order:
        if gap_dz[cell] <= 0.0:
            return 1.0 if E_entry[cell] > 0.0 else 0.0
    # The gap runs to the domain edge, so there is no cell beyond it to test;
    # the ray did not leave the far end either, so it died inside the gap.
    return 0.0


# Tripwire tolerance, as a fraction of EMITTED BEAM POWER (see
# ``beam_gap_ledger_mismatch``): 5%, roughly a decade above the benign
# Coulomb-ceiling floor and a decade below the item-35 break.
BEAM_GAP_LEDGER_POWER_ATOL = 0.05


def beam_gap_ledger_mismatch(gap_ledger, eta, atol=BEAM_GAP_LEDGER_POWER_ATOL):
    """Worst CSDA gap-survival ledger divergence, or ``None`` if all agree.

    ``gap_ledger`` maps each active cathode end to ``(probe, ray, circuit)``
    gap survival (see ``_csda_beam_deposition``). Two comparisons are made:

    ``probe`` vs ``ray``
        The probe must reproduce the deposition ray it mirrors. This is the
        item-35 class: a probe that misreports the ray corrupts ``sigma_eff``
        and therefore the circuit, and every internally-consistent check
        downstream still passes.
    ``ray`` vs ``circuit``
        The circuit must be able to represent the ray. Fails on adapter clamp
        saturation, and again -- independently -- whenever a broken probe has
        been propagated into ``sigma_eff``.

    Returns ``(end, kind, left, right, power_fraction)`` for the worst
    offender, where ``kind`` is ``"probe_vs_ray"`` or ``"ray_vs_circuit"``.

    The tolerance is stated on the quantity that matters rather than on the
    survival fractions themselves. The circuit debits
    ``eta * f_bypass * I_eth_star * V_b``, so ``eta * |left - right|`` is the
    fraction of emitted beam power booked to a bypass the fluid never loses
    (or vice versa) -- the ledger hole itself.

    A small benign floor is unavoidable and must stay below ``atol``: a
    fully-transmitting ray cannot be represented above the Beer-Lambert
    solve's Coulomb-only ceiling ``exp(-L_cath/l_bi)``, so the
    ``sigma_eff >= 0`` clamp leaves ``eta * (1 - exp(-L_cath/l_bi))``
    unbooked. That floor is self-limiting -- saturation needs a transmitting
    ray, which needs a long ``l_bi``, which makes the ceiling shortfall
    small -- and measures 0.3-1.3% of emitted beam power across the
    campaign's long-mfp states. Item 35 reads 35.8% on ``probe_vs_ray`` and
    34.6% on ``ray_vs_circuit``.
    """
    eta = float(eta)
    worst = None
    for end, entry in (gap_ledger or {}).items():
        if entry is None:
            continue
        probe, ray, circuit = (float(v) for v in entry)
        for kind, left, right in (
            ("probe_vs_ray", probe, ray),
            ("ray_vs_circuit", ray, circuit),
        ):
            power = eta * abs(left - right)
            if power > atol and (worst is None or power > worst[4]):
                worst = (int(end), kind, left, right, power)
    return worst


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


def _gap_clip_is_face_aligned(gap_dz, length_cm):
    """True when the ``L_cath`` clip landed on a cell face.

    ``_clip_ray_length`` gives each cell along the ray either its full length,
    zero, or -- in the single cell where ``L_cath`` runs out mid-cell -- a
    partial length. That partial cell is the ONLY place a gap-clipped probe
    has less path available than the deposition ray it mirrors, and therefore
    the only place the two can disagree about where the ray stopped: the
    deposition ray can be absorbed inside it while the probe runs out of dz
    first and transmits. Everywhere else the clip is a prefix of the same
    per-cell path lengths. Used by the probe skip in
    ``_csda_beam_deposition``; see the comment there.
    """
    dz = np.asarray(gap_dz, dtype=float)
    full = np.asarray(length_cm, dtype=float)
    return not bool(np.any((dz > 0.0) & (dz < full)))


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
    # Recycle at the cathode surface feeds the COLUMN on a two-zone state.
    neutral_gain_rate = dN_loss / (
        geometry.plasma_volume_cm3
        if state.nn_a is not None
        else geometry.neutral_volume_cm3
    )
    # Sheath electron power: P_cathode_e is lost at the cathode surface and
    # P_anode_e at the anode mesh. Legacy has neither resolved, so both stay
    # colocated in its source cell exactly as before; resolved geometry lands each
    # at its own electrode.
    electron_power_loss_W = zeros.copy()
    cathode_cells = cathode_adjacent_cells(geometry)
    anode_pairs = anode_flanking_cells(geometry)
    # R3.2/A16: under the repaired boundary stance, route only the plasma-thermal
    # electron power to the plasma; the sheath-fall phi is booked on the electrode.
    thermal_only = bool(input_flags.get("characteristic_boundary", False))
    if cathode_cells:
        _deposit_electrode_power(
            electron_power_loss_W,
            result=cathode_solve.beam_result.result,
            cathode_cell=int(cathode_cells[0]),
            anode_pair=anode_pairs[0] if anode_pairs else None,
            state=state,
            derived=derived,
            thermal_only=thermal_only,
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
                thermal_only=thermal_only,
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
    obstruction, which the beam never enters.
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
        smoothing_cm=float(input_dict.get("beam_deposition_smoothing_cm", 0.0)),
    )
    volume_ratio = geometry.plasma_volume_cm3 / geometry.neutral_volume_cm3
    # Two-zone state: nn is the column density on
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
        # Historical zero-drift beam birth: the ion is born at rest.
        u_birth = zeros.copy()
        beam_M_birth = zeros.copy()
        beam_Mn_debit = None
        beam_Mna = None
    Ti_birth = _birth_temperature(
        input_dict.get("Ti_birth_ionization", "floor"),
        beam_derived.Ti,
        floors["Ti"],
    )
    # A14 (R4.2): the beam electron birth already uses the defensible Ee=0
    # convention; under "conservative" reconcile the ion energy too by booking
    # the mass-loading relative-drift mixing energy to Ei (the beam ion is born
    # at u_birth and joins the bulk flow at u_i), matching the bulk birth.
    birth_energy_model = str(
        input_dict.get("ionization_birth_energy_model", "legacy")
    )
    beam_Ei = 1.5 * ev_to_erg * Ti_birth * S_beam
    if birth_energy_model == "conservative":
        beam_Ei = beam_Ei + 0.5 * ion_mass_g * (
            beam_derived.u - u_birth
        ) ** 2 * S_beam
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
            Ei=beam_Ei,
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


_BEAM_SMOOTH_CACHE = {}


def _array_fingerprint(values, dtype):
    """Content fingerprint of an array, canonicalized to the consumed dtype.

    Returned as ``(shape, digest)``; the digest is taken over the exact bytes
    ``_beam_smoothing_matrix`` reads, so two arrays that differ anywhere the
    kernel looks produce different keys.
    """
    arr = np.ascontiguousarray(values, dtype=dtype)
    digest = hashlib.blake2b(arr.tobytes(), digest_size=16).digest()
    return arr.shape, digest


def _beam_smoothing_key(geometry, sigma_cm):
    """Cache key for :func:`_beam_smoothing_matrix`, by CONTENT not address.

    ``id(geometry)`` is unique only among LIVE objects: CPython reuses an
    address once the old geometry is collected, so a freed geometry followed
    by a differently meshed allocation at the same address would return the
    OLD mesh's matrix. A shape mismatch would raise at the matmul, but two
    geometries with the same cell count and different positions/lengths/roles
    (a ``source_fixed_grid`` A/B, an nx-matched ``source_region_dz_cm`` sweep)
    would silently smooth with the wrong kernel.

    Every geometry input the matrix build reads is in the key: ``z_cm`` and
    ``length_cm`` (centres and the cell-length weighting), ``z_edges_cm`` and
    ``cathode_face_indices`` (the reflecting image sources), and
    ``plasma_active`` (the support -- two meshes agreeing in z/lengths/faces
    but differing in cell ROLES build different matrices).
    """
    return (
        round(float(sigma_cm), 8),
        _array_fingerprint(geometry.z_cm, float),
        _array_fingerprint(geometry.length_cm, float),
        _array_fingerprint(geometry.z_edges_cm, float),
        _array_fingerprint(geometry.plasma_active, bool),
        tuple(int(i) for i in np.asarray(geometry.cathode_face_indices, dtype=int)),
    )


def _beam_smoothing_matrix(geometry, sigma_cm):
    """Conservative Gaussian redistribution matrix over the live plasma cells.

    ``W[i, j]`` is the fraction of cell ``j``'s beam deposition moved to cell
    ``i``; columns sum to 1 over the live cells, so ``W @ ext`` conserves the
    total (extensive) deposition. The width is a fixed length in cm, so the
    smoothed profile is mesh-convergent. The O(cells^2) build is cached on the
    geometry CONTENT and the width (see :func:`_beam_smoothing_key`) -- both
    fixed for a run -- so the matrix is built once per run rather than once per
    RHS evaluation, and two distinct meshes can never share an entry.

    The support is ``geometry.plasma_active``, NOT ``plasma_volume_cm3 > 0``.
    The typed plasma-dead cells behind the cathode face (plenum, obstruction)
    carry a finite plasma volume, so a ``Vp > 0`` support puts weight on rows
    that ``_apply_active_plasma_topology`` then zeroes -- silently deleting
    that share of the deposit (~19% at the cathode cell) from every channel
    this kernel serves.

    The cathode surface is a REFLECTING boundary: the Gaussian tail that would
    fall behind an emitting face is folded forward about that face (image
    source at ``2*z_face - z_j``) instead of being discarded, which is what
    keeps the deposit near the cathode physical rather than merely normalized.
    Both faces reflect under ``TwinCathode``. The far machine end needs no
    special handling -- every cell there is active, and normalization absorbs
    the residual tail past the end.

    Each weight is multiplied by the target cell length (a cell-integrated
    approximation) before the column is normalized, so a refined region is not
    over-weighted per cm and the operator is mesh-independent, not just
    conservative. Normalization remains the exact conservation guarantee.
    """
    key = _beam_smoothing_key(geometry, sigma_cm)
    W = _BEAM_SMOOTH_CACHE.get(key)
    if W is not None:
        return W
    z = np.asarray(geometry.z_cm, dtype=float)
    dz = np.asarray(geometry.length_cm, dtype=float)
    z_edges = np.asarray(geometry.z_edges_cm, dtype=float)
    active = np.asarray(geometry.plasma_active, dtype=bool)
    live = np.flatnonzero(active)
    n = z.size
    W = np.zeros((n, n), dtype=float)
    if live.size:
        sigma = float(sigma_cm)
        z_live = z[live][:, None]
        G = np.exp(-0.5 * ((z_live - z[None, :]) / sigma) ** 2)
        for face in np.asarray(geometry.cathode_face_indices, dtype=int):
            z_face = float(z_edges[face])
            G += np.exp(-0.5 * ((z_live + z[None, :] - 2.0 * z_face) / sigma) ** 2)
        G *= dz[live][:, None]
        colsum = G.sum(axis=0)
        # A column whose weights all underflow keeps no deposit to rescale; it
        # cannot occur for an active column (which always contains itself).
        G /= np.where(colsum > 0.0, colsum, 1.0)
        W[live, :] = G
    _BEAM_SMOOTH_CACHE[key] = W
    return W


def _smooth_beam_density(W, density, Vp):
    """Apply the conservative matrix to a per-cell density (events/power / cm^3).

    Works in extensive units (density * Vp) so the deposited total is preserved
    exactly; dead cells (Vp <= 0) carry nothing and stay zero.
    """
    Vp = np.asarray(Vp, dtype=float)
    ext = np.asarray(density, dtype=float) * Vp
    ext_s = W @ ext
    out = np.zeros_like(ext_s)
    live = Vp > 0.0
    out[live] = ext_s[live] / Vp[live]
    return out


def _beam_ionization_sources(
    state,
    geometry,
    cathode_solve,
    boundary,
    Te=None,
    exc_energy_fallback_eV=21.218,
    smoothing_cm=0.0,
):
    zeros = np.zeros(geometry.cells, dtype=float)
    beam_result = cathode_solve.beam_result
    S_beam = zeros.copy()
    S_exc = zeros.copy()
    S_exc_E = zeros.copy()
    beam_power_density = zeros.copy()

    deposition = getattr(cathode_solve, "beam_deposition", None)
    smoothing_cm = float(smoothing_cm)
    if smoothing_cm < 0.0:
        raise ValueError(
            f"beam_deposition_smoothing_cm must be >= 0 (got {smoothing_cm})"
        )
    if deposition is not None and not (smoothing_cm > 0.0):
        # CSDA path (B2), historical UNSMOOTHED branch (bit-exact): the module
        # already integrated each ray; convert its per-cell totals to densities.
        # ``beam_power_deposition`` carries the whole per-cell beam energy
        # (heating + radiated + cost) so the separate cost and radiation sinks
        # subtract to the module's net heating, keeping the four-term
        # decomposition meaningful. P_ohmic keeps its historical gap-weighted
        # booking.
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
    if deposition is not None:
        # CSDA path with conservative deposition smoothing (default-off; the
        # branch above is bit-exact when smoothing is 0). The beam deposition
        # densities are smoothed over a fixed physical width BEFORE the ohmic
        # gap booking is added, so only the beam-range deposition is spread and
        # the totals are conserved (this removes the mesh-scale sheath kick
        # where the beam range crosses a cell boundary).
        Vp = geometry.plasma_volume_cm3
        beam_dep_power = zeros.copy()
        ohmic_power = zeros.copy()
        for end, dep in deposition.items():
            if dep is None:
                continue
            S_beam += dep.ionization_events / Vp
            S_exc += dep.excitation_events / Vp
            S_exc_E += dep.radiated_erg_s / Vp / ev_to_erg
            beam_dep_power += (
                dep.plasma_heating_erg_s
                + dep.radiated_erg_s
                + dep.ionization_cost_erg_s
            ) / Vp
            solver_result = (
                beam_result.result if end == 0 else beam_result.result_twin
            )
            gap = np.asarray(gap_cell_indices(geometry, end=end), dtype=int)
            ohmic_weights = _ohmic_gap_weights(geometry, gap, Te)
            ohmic_power[gap] += (
                ohmic_weights * solver_result.P_ohmic * 1.0e7 / Vp[gap]
            )
        W = _beam_smoothing_matrix(geometry, smoothing_cm)
        S_beam = _smooth_beam_density(W, S_beam, Vp)
        S_exc = _smooth_beam_density(W, S_exc, Vp)
        S_exc_E = _smooth_beam_density(W, S_exc_E, Vp)
        beam_dep_power = _smooth_beam_density(W, beam_dep_power, Vp)
        beam_power_density = beam_dep_power + ohmic_power
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
    electron_power_loss_W, result, cathode_cell, anode_pair, state, derived,
    thermal_only=False,
):
    """Land P_cathode_e at the cathode cell and P_anode_e at the anode mesh.

    The anode collects on both mesh faces, so its sheath power is split between
    the two flanking cells in proportion to each face's Bohm collection -- the
    same weighting ``anode_collection_rhs`` uses, so power and particles are
    removed on the same side. With no resolved anode the whole of P_anode_e falls
    back to the cathode cell, which is where the lumped model puts it.

    ``thermal_only`` (R3.2/A16 routing, the repaired stance): deposit only the
    PLASMA-THERMAL part (2Te per electron), leaving the sheath-fall ``phi`` on the
    electrode/circuit surface instead of removing it from the plasma thermal
    store. Off (the golden default) deposits the full historical P_*_e.
    """
    p_cathode_e = result.P_cathode_e_thermal if thermal_only else result.P_cathode_e
    p_anode_e = result.P_anode_e_thermal if thermal_only else result.P_anode_e
    electron_power_loss_W[cathode_cell] += p_cathode_e
    if anode_pair is None:
        electron_power_loss_W[cathode_cell] += p_anode_e
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
    electron_power_loss_W[gap_side] += weights[0] * p_anode_e
    electron_power_loss_W[column_side] += weights[1] * p_anode_e


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
