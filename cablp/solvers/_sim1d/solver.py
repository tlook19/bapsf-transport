"""Solver implementation for the conservative axial 1D LAPD model."""

import math
import struct
import warnings
from dataclasses import dataclass, replace
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from scipy.linalg import solve_banded

from .core.config import (
    coverage_closure_defaults,
    emitting_area_defaults,
    neutral_probe_source_defaults,
    default_config,
    input_dict_template_1d,
    load_config,
    resolve_config,
    resolve_nn0,
)
from .core.deprecations import warn_deprecated_config
from .core.model_families import resolve_model_families
from .core.geometry import (
    _anode_neutral_transparency,
    absorbing_live_cells_by_role,
    build_geometry,
    is_plenum_cell,
    puff_cell_indices,
    pump_cell_indices,
)
from .core.ignition import (
    IGNITION_BEAM_END_LOSS_KEYS,
    IGNITION_DIAGNOSTIC_FIELDS,
    IGNITION_POWER_GROUPS,
    IgnitionMonitor,
    _Sample as _IgnitionSample,
    empty_ignition_diagnostics,
)
from .core.integrator import (
    floor_state_vector,
    ssprk2_step,
)
from .core.state import (
    NEUTRAL_ANNULUS_NAME,
    NEUTRAL_ENERGY_FLOOR_T_K,
    STATE_NAMES_1D,
    ConservativeState1D,
    apply_state_floors,
    assert_finite_state,
    conservative_from_primitives,
    derive_state,
    neutral_energy_floor,
    pack_state,
    state_field_names,
    unpack_state,
)
from .core.timestep import apply_dt_global_scale, suggest_timestep
from .core.options import build_solver_options
from .core.validation import (
    OPERATOR_SPLITTINGS,
    _RawStageError,
    _bad_array_summary,
    refuse_anode_backscatter_double_book,
    refuse_cathode_backscatter_double_book,
    refuse_dvm_anode_jet_without_cathode_coupling,
    refuse_dvm_cathode_jet_without_cathode_coupling,
    resolve_coverage_config,
    resolve_electron_drift_transport_config,
    resolve_emitting_area_config,
    resolve_jet_arming_criterion,
    resolve_neutral_jet_config,
    resolve_neutral_probe_config,
    validate_equilibration_gas_puff_on,
    validate_gas_puff_config,
    validate_neutral_seed_cache_config,
    validate_operator_splitting,
    validate_phase_config,
    validate_r1_configuration_presence,
    validate_raw_stage,
)
from .physics.conduction import (
    heat_conduction_rhs,
    implicit_heat_conduction_step,
    validate_implicit_heat_scheme,
)
from .physics.kinetic_dvm import (
    ANNULUS_FLIGHT_MODELS as KINETIC_DVM_ANNULUS_FLIGHT_MODELS,
    ELASTIC_MODELS as KINETIC_DVM_ELASTIC_MODELS,
    EXCHANGE_MODELS as KINETIC_DVM_EXCHANGE_MODELS,
    TRANSFER_HOLDS as KINETIC_DVM_TRANSFER_HOLDS,
    WALL_REFLECTION_MODELS as KINETIC_DVM_WALL_REFLECTION_MODELS,
    TransientDVM,
)
from .physics.kinetic_neutrals import (
    EV as _KIN_EV,
    KB as _KIN_KB,
    M_HE as _KIN_M_HE,
    KN2ZoneJump,
    KineticEngineFast,
    VGrid as _KineticVGrid,
    _inflow as _kinetic_inflow,
)
from .physics.cathode import (
    BEAM_DEPOSITION_MODELS,
    BEAM_GAP_LEDGER_POWER_ATOL,
    CoverageView1D,
    beam_anomalous_power_density,
    beam_gap_ledger_mismatch,
    beam_ionization_rhs,
    beam_ionization_rhs_terms,
    beam_launch,
    cathode_boundary_state,
    cathode_power_balance_terms_W,
    cathode_sample_indices,
    cathode_source_terms,
    solve_cathode_boundary,
    tail_reflect_face,
    validate_cathode_Rp_model,
    validate_cathode_lnL_model,
    validate_cathode_solver_model,
)
from cablp.cathode.circuit_idriven import beam_launch_energy_eV
from .physics.cathode import (
    CATHODE_ENV_T_K,
    advance_circuit_current_driven,
    circuit_bound_object,
    idriven_result_evaluator,
    idriven_vdis_evaluator,
    resolve_vessel_node,
    vessel_beam_climb_V,
    vessel_node_advance,
)
from cablp.cathode.circuit_idriven import _CIRCUIT_BOUND_OBJECTS
from .physics.energy import (
    electron_cooling_rhs,
    electron_cooling_rhs_terms,
    electron_ion_exchange_rhs,
    ion_charge_exchange_rhs,
)
from .physics.flux import ion_sound_speed, plasma_flux_rhs, plasma_flux_rhs_terms
from .physics.neutrals import (
    GAS_PUFF_DIAGNOSTIC_FIELDS,
    NEUTRAL_PROBE_WAVEFORMS,
    gas_puff_rate_profile,
    _effective_pump_speed,
    neutral_exchange_coefficients,
    neutral_exchange_rhs,
    neutral_exchange_two_zone_rhs,
    neutral_fluid_flux_rhs,
    neutral_initial_profile_values,
    neutral_probe_profile_weights,
    neutral_probe_source_rhs,
    neutral_probe_waveform_mean,
    neutral_probe_waveform_table,
    neutral_probe_waveform_value,
    neutral_source_sink_rhs,
    neutral_wind_advection_rhs,
    neutral_zone_exchange_conductance,
    neutral_zone_exchange_rhs,
    neutral_zone_volumes,
    puff_rate,
    pump_rate,
    two_zone_knudsen_coefficients,
)
from .physics.reactions import (
    gas_puff_local_ionization_rhs as _gas_puff_local_ionization_rhs,
    reaction_rhs,
    reaction_rhs_terms,
    _birth_temperature,
    recombination_energy_return_rhs,
)
from .physics.hot_neutrals import (
    HOT_CHANNEL_DIAGNOSTIC_FIELDS,
    ballistic_flight_kernels,
    neutral_hot_channel_rhs,
)
from .physics.jet_carrier import cathode_jet_carrier_rhs
from .physics.sources import (
    CATHODE_JET_ENERGY_CONVENTIONS,
    ION_NEUTRAL_DRAG_MODELS,
    add_state_rhs,
    anode_collection_rhs,
    cathode_jet_backscatter_speed,
    characteristic_boundary_rhs,
    ELECTRON_DRIFT_DIAGNOSTIC_ROWS,
    ELECTRON_DRIFT_DIAGNOSTIC_SCALARS,
    electron_drift_transport_rhs,
    ion_neutral_collision_rhs,
    ionization_birth_neutral_temperature_eV,
    neutral_cx_channel_rhs,
    neutral_energy_transfer_row,
    ion_neutral_drag_rhs,
    ion_neutral_frictional_heating_rhs,
    ion_neutral_thermalization_rhs,
    neutral_energy_wall_rhs,
    neutral_momentum_wall_rhs,
    neutral_momentum_two_zone_rhs,
    IONIZATION_BIRTH_DEFICIT_DIAGNOSTIC_FIELDS,
    IONIZATION_BIRTH_DEFICIT_SITES,
    neutral_temperature_eV,
    neutral_wind_two_zone_factors,
    neutral_wind_velocity,
    flux_tube_geometry_rhs,
    hyperbolic_energy_correction_rhs,
    pressure_work_rhs,
)
from .physics.tracer import (
    CRITERION_NAMES,
    affine_time_integral as tracer_affine_time_integral,
    affine_update as tracer_affine_update,
    beam_plasma_thinness as tracer_beam_plasma_thinness,
    bind_census as tracer_bind_census,
    conducted_current_A as tracer_conducted_current_A,
    growth_rate as tracer_growth_rate,
    passive_anomalous_leak as tracer_passive_anomalous_leak,
    quasistatic_Te_eV,
    relative_drift as tracer_relative_drift,
    resolve_criteria as resolve_tracer_criteria,
    transport_ratio as tracer_transport_ratio,
)
from .results.restart import (
    REFUSED_NEUTRAL_MODELS as RESTART_REFUSED_NEUTRAL_MODELS,
)
from cablp.atomic.adas import he_rate_temperature_range_eV, he_rates
from cablp.cathode.beam_deposition import (
    ANOMALOUS_MODELS,
    HE_EII_EDGE_REL_TOL,
    HE_EII_EPS_TOP,
)
from cablp.atomic.cross_sections import charge_ex_react
from cablp.cathode.kernels import PROVENANCE as KERNEL_PROVENANCE
from cablp.constants import (
    I_ion,
    ev_to_erg,
    kb_cgs,
    m_He_cgs,
    qe_SI,
)


#: Timestep multiplier [dimensionless] applied after a rejected attempt. Must
#: lie in (0, 1) to shrink the step; 1/2 halves it, which reaches any smaller
#: admissible dt in a logarithmic number of retries.
DT_REJECT_FACTOR = 0.5
# How many assembled two-zone backward-Euler matrices one solver keeps. This
# is a cache capacity, not a model parameter: no equation reads it, and it
# bounds memory (one entry is a dense (2*cells)^2 array) against a run that
# never settles onto a repeating dt. The equilibration it exists for visits
# ~19 distinct dt values.
TWO_ZONE_MATRIX_CACHE_ENTRIES = 32

#: Relative threshold [dimensionless] for the floor-aware drain exemption on
#: the "surface_loss" timestep bound, consulted only when the
#: ``surface_loss_floor_exempt`` flag is on. It separates a cell HOVERING at
#: its temperature floor (clip plus one step of re-heating residue) from a
#: healthy drained cell orders of magnitude above it; see the flag's entry in
#: ``core/config.py``. It is the ENTRY threshold only: re-admission uses the
#: wider outer threshold named by the params key
#: ``surface_loss_floor_exempt_exit_rtol``, which must exceed this one and is
#: armed at its default; that key set to 0 makes re-admission use this value
#: too, which is the knife edge.
SURFACE_LOSS_FLOOR_EXEMPT_RTOL = 1e-3

#: Relative slack [dimensionless] on the "the accepted step IS dt_min" test the
#: dt_min lock counts. The accepted dt reaches dt_min through a clamp, so the
#: comparison is against a value that has been through the caps and the retry
#: ladder; the slack keeps a last-bit difference from reading as "above the
#: floor" while staying far below any step a physics bound would set.
_DT_MIN_ACCEPTED_RTOL = 1e-9


#: What energy each named RHS term's neutral-density row carries, under the
#: ``neutral_energy`` flag. This table is the disclosure: every term in the
#: ledger appears in it, and a term that is NOT in it raises rather than moving
#: neutrals with no stated energy. Modes:
#:
#: ``"owns"``   the term computes its own ``En`` row (it needs face-level or
#:              per-channel information a net density row cannot carry back)
#: ``"local"``  a pure SINK; the atoms leave at the local ``En/nn``, so the
#:              temperature of what remains does not move
#: ``"wall"``   a SOURCE of gas at the vessel wall temperature: fresh feed, or
#:              plasma neutralized on a surface and re-emitted from it
#: ``"ion"``    a SOURCE of gas at the local ION temperature: a recombined ion
#:              is born as a neutral at the temperature it had (see the routing
#:              note on ``recombination_rad_loss`` below)
#: ``"none"``   the term never moves neutrals
_NEUTRAL_ENERGY_TERM_BOOKING = {
    # --- terms that own their En row -------------------------------------
    "ion_neutral_collision": "owns",
    "neutral_energy_wall": "owns",
    "neutral_cx_channel": "owns",
    "neutral_hot_channel": "owns",
    "cathode_jet_neutral_energy": "owns",
    "cathode_jet_hot_carrier": "owns",
    "neutral_exchange": "owns",
    "neutral_zone_exchange": "owns",
    "neutral_sources": "owns",
    "neutral_wind_advection": "owns",
    # --- sinks: ionization consumes cold gas at its own energy -----------
    "ionization_birth": "local",
    "beam_ionization_birth": "local",
    "gas_puff_local_ionization": "local",
    # --- surface sources: recycled plasma leaves the surface at T_wall ---
    # ``boundary_absorption`` is a permanently-zero row kept for saved-ledger
    # schema stability (retired; Tom, 2026-08-31); its entry stays so the table
    # still covers every row the ledger emits.
    "boundary_absorption": "wall",
    "characteristic_boundary": "wall",
    "anode_collection": "wall",
    "cathode_surface_loss": "wall",
    # The anode electron sheath row is energy only -- it moves no particles,
    # so it has no neutrals to carry a birth temperature for.
    "anode_e_sheath_loss": "none",
    "surface_loss": "wall",
    "neutral_probe_source": "wall",
    # --- recombination: born at the ion temperature ----------------------
    # ROUTING NOTE. These neutrals are Ti-class and so are physically hot, but
    # they stay in the COLD channel deliberately. The operator hands the
    # recombined ion's DIRECTED momentum to M_n as an exact mirror of the ion
    # loss, so its particle, momentum and energy bookings are one closed unit
    # on the cold fluid; moving the particle to the hot channel while its
    # momentum stayed behind would break that mirror, and moving the momentum
    # too would ask the isotropic ballistic kernel to carry a population born
    # with a bulk drift, which is not what it integrates. What IS fixed here is
    # the leak: the ion loses (3/2) k Ti per recombination and, before this
    # pass, nothing received it.
    "recombination_rad_loss": "ion",
    "recombination_3b_loss": "ion",
    # --- everything else never touches nn --------------------------------
    "plasma_advective_flux": "none",
    "plasma_front_flux": "none",
    "pressure_work": "none",
    # The drift operator writes the ELECTRON row alone; it moves no particles
    # of any species, so it has no neutrals to carry a birth temperature for.
    "electron_drift_transport": "none",
    "hyperbolic_dissipation_heating": "none",
    "flux_tube_geometry": "none",
    "ei_exchange": "none",
    "ionization_energy_cost": "none",
    "electron_ion_cooling": "none",
    "electron_neutral_cooling": "none",
    "ion_charge_exchange": "none",
    "ion_neutral_drag": "none",
    "ion_neutral_frictional_heating": "none",
    "ion_neutral_thermalization": "none",
    "neutral_momentum_wall": "none",
    "neutral_momentum_radial": "none",
    "beam_power_deposition": "none",
    "beam_ionization_cost": "none",
    "beam_excitation_radiation": "none",
    "recombination_energy_return": "none",
    "heat_conduction": "none",
}


#: Name of the RHS row carrying the beam's electron-energy deposition. Bound
#: to a constant because ``beam_deposition_in_heat_substep`` has to name the
#: same row in two places -- the one it removes from the explicit sum and the
#: one it hands to the heat substep -- and a typo in either would be a silent
#: energy leak rather than an error.
BEAM_POWER_DEPOSITION_TERM = "beam_power_deposition"

#: The cathode-circuit scalars written to the HDF5, per solved end.
#:
#: SIX NAMES ARE GONE from this tuple and are not exported any more:
#: ``P_cathode_i_pl``, ``P_anode_i_pl``, ``P_loss``, ``P_net``, ``P_net2`` and
#: ``P_comp``. The first five were computed under a power book that does not
#: close (the anode ion current under a cathode name; phi-inclusive electron
#: powers summed against thermal-only ion powers; electrode powers subtracted
#: from load field work, which books each sheath fall twice), and the last two
#: were exported and never read. Their successors are in the closed audit set
#: below, and ``results/cathode_diagnostics.py`` names each one when a caller
#: reads a retired name off a file that no longer carries it. The dataclass
#: fields stay computed for one release so an old file's numbers can still be
#: reproduced at this tip.
_CATHODE_RESULT_KEYS = (
    "phi_c_plus",
    "phi_c_minus",
    "phi_c",
    "phi_a",
    "V_p",
    "V_b",
    "R_p",
    "I_i",
    "I_e",
    "I_eth",
    "I_eth_star",
    "I_tot",
    "P_wall",
    "P_load",
    "P_prim",
    "P_ohmic",
    "P_cathode_e",
    "P_cathode_i",
    "P_anode_e",
    "P_anode_i",
    "beam_bypass_fraction",
    "l_b",
    # Ceiling census (R1). ``phi_c_ceiling_V`` is the ceiling the sheath root
    # was solved against, ``circuit_V_avail_V`` the circuit-available device
    # voltage the optional bound was formed from (NaN when the bound is not in
    # force), ``bound_active`` which member of the composition the solve sat on
    # (0 none / 1 data cap / 2 circuit). All three are NaN on the voltage-driven
    # (floating) solve, which has no ceiling. Runs saved before this build lack
    # the datasets and readers must default them.
    "phi_c_ceiling_V",
    "circuit_V_avail_V",
    "bound_active",
    # R3.2 CLOSED AUDIT SET. The circuit computes all of these on every
    # current-driven solve; until this build none of them reached the HDF5, so
    # a saved trajectory could be read only through the unclosed scalars above.
    # Appended (never inserted) so the historical dataset order is stable.
    #
    # ``I_i_a`` is the ANODE ion current [A] -- the quantity the anode powers
    # are built from, and the one the retired ``P_cathode_i_pl`` silently
    # carried under a cathode name.
    "I_i_a",
    # The one-control-surface split (A16): each electrode power splits into the
    # PLASMA-THERMAL part (2Te per electron, Te/2 per ion -- drawn from the
    # plasma thermal store) and the SHEATH-FALL phi part (drawn from the
    # circuit and deposited on the electrode). ``*_thermal + *_phi == P_*`` to
    # machine zero, and the thermal members are what the fluid boundary
    # actually removes.
    "P_cathode_e_thermal",
    "P_cathode_e_phi",
    "P_cathode_i_thermal",
    "P_cathode_i_phi",
    "P_anode_e_thermal",
    "P_anode_e_phi",
    "P_anode_i_thermal",
    "P_anode_i_phi",
    # Surface-resolved audit [W]: the total plasma-thermal loss to electrodes,
    # the net power heating the plasma (``P_prim + P_ohmic`` minus that loss),
    # and the plasma power landing on each electrode surface.
    "P_plasma_thermal_loss",
    "P_into_plasma",
    "P_cathode_surface",
    "P_anode_surface",
    # Measurement-plane bookkeeping: the hidden series drop, the stray parallel
    # branch, and the three-plane aliases for the measured terminal voltage and
    # currents.
    "V_series",
    "I_parallel",
    "V_dis",
    "I_plasma",
    "I_bank",
    # Load-power closure. ``I_e_ret`` is the returning plasma-electron current
    # to the cathode, ``P_load_ledger`` the per-region net-current field work,
    # and the two residuals are the closure checks: both are ~0 by
    # construction, so a save where either is not small says the circuit
    # ledger has stopped closing.
    "I_e_ret",
    "P_load_ledger",
    "P_load_residual",
    "I_cathode_kirchhoff_residual",
)

#: The members of :data:`_CATHODE_RESULT_KEYS` that only the CURRENT-DRIVEN
#: circuit solve populates. The voltage-driven (floating) solve in
#: ``cablp.cathode.circuit`` never assigns them, so they sit at their
#: ``SolverResult`` dataclass defaults there -- which are zeros, and a zero in
#: a power column is indistinguishable from a computed zero. They are exported
#: as NaN on a floating save instead, so absence of a value is visible in the
#: file rather than inferred. ``I_i_a`` is NOT here: both solves compute it.
_CURRENT_DRIVEN_ONLY_CATHODE_KEYS = frozenset(
    {
        "P_cathode_e_thermal",
        "P_cathode_e_phi",
        "P_cathode_i_thermal",
        "P_cathode_i_phi",
        "P_anode_e_thermal",
        "P_anode_e_phi",
        "P_anode_i_thermal",
        "P_anode_i_phi",
        "P_plasma_thermal_loss",
        "P_into_plasma",
        "P_cathode_surface",
        "P_anode_surface",
        "V_series",
        "I_parallel",
        "V_dis",
        "I_plasma",
        "I_bank",
        "I_e_ret",
        "P_load_ledger",
        "P_load_residual",
        "I_cathode_kirchhoff_residual",
    }
)


@dataclass(frozen=True)
class StepAttempt1D:
    y: np.ndarray
    dt: float
    operator_split: bool
    solver_cache: SimpleNamespace
    floor_ledger: dict
    raw_rejection_reason: str = ""
    raw_rejection_detail: dict | None = None
    ion_booking: np.ndarray | None = None
    source_booking: dict | None = None
    cathode_jet_energy_booking: np.ndarray | None = None
    anode_jet_energy_booking: np.ndarray | None = None
    coverage_burn: np.ndarray | None = None
    coverage_reservoir_burn: np.ndarray | None = None
    coverage_w: np.ndarray | None = None
    # Regime-R2 tracer coefficients frozen at this attempt's step start, or
    # None whenever the tracer is not engaged. Carried on the attempt for the
    # same reason the coverage and DVM accumulators are: a rejected attempt
    # must move neither the Picard cache nor the passive/active boundary.
    tracer: dict | None = None


def _atomic_rate_domain(result):
    """Return saved active-plasma coverage of the bundled He ADF11 grid."""
    te_min_eV, te_max_eV = he_rate_temperature_range_eV()
    atomic_rate_model = str(
        getattr(result, "params", {}).get("atomic_rate_model", "adas")
    )
    Te = np.asarray(result.Te, dtype=float)
    active = np.asarray(result.plasma_active, dtype=bool)
    time = np.asarray(result.time, dtype=float)
    phase = np.asarray(result.phase, dtype=str)
    if atomic_rate_model != "adas":
        count_fraction = np.full(time.shape, np.nan, dtype=float)
        volume_fraction = np.full(time.shape, np.nan, dtype=float)
        active_min = np.full(time.shape, np.nan, dtype=float)
    elif not np.any(active):
        count_fraction = np.zeros(time.shape, dtype=float)
        volume_fraction = np.zeros(time.shape, dtype=float)
        active_min = np.full(time.shape, np.nan, dtype=float)
    else:
        active_Te = Te[:, active]
        below = active_Te < te_min_eV
        count_fraction = np.mean(below, axis=1)
        volumes = np.asarray(result.plasma_volume_cm3, dtype=float)[active]
        volume_fraction = np.sum(below * volumes[None, :], axis=1) / np.sum(
            volumes
        )
        active_min = np.min(active_Te, axis=1)

    below_any = count_fraction > 0.0
    below_afterglow = below_any & (phase == "afterglow")

    def first_time(mask):
        indices = np.flatnonzero(mask)
        return float(time[indices[0]]) if indices.size else np.nan

    return {
        "table_applies": atomic_rate_model == "adas",
        "table_Te_min_eV": te_min_eV,
        "table_Te_max_eV": te_max_eV,
        "active_cell_fraction_below": count_fraction,
        "active_volume_fraction_below": volume_fraction,
        "active_Te_min_eV": active_min,
        "first_below_time_s": first_time(below_any),
        "first_afterglow_below_time_s": first_time(below_afterglow),
    }


@dataclass(frozen=True)
class SimulationProgress1D:
    """Lightweight accepted-step progress snapshot for long-running simulations."""

    fraction: float
    time: float
    t_end: float
    step: int
    max_steps: int
    accepted_dt: float
    suggested_dt: float
    step_cap: str
    active_constraint: str
    retry_count: int
    rejection_reason: str
    phase: str
    saved_samples: int
    wall_elapsed_s: float
    wall_remaining_s: float
    timestep_limiters: tuple = ()


class ProgressPrinter1D:
    """Simple callable progress reporter for scripts and notebooks."""

    def __init__(self, interval_fraction=0.05, interval_steps=100, stream=None):
        self.interval_fraction = max(float(interval_fraction), 0.0)
        self.interval_steps = max(int(interval_steps), 1)
        self.stream = stream
        self._last_fraction = -np.inf
        self._last_step = -self.interval_steps

    def reset(self):
        self._last_fraction = -np.inf
        self._last_step = -self.interval_steps

    def __call__(self, progress):
        if progress.step <= self._last_step or progress.fraction < self._last_fraction:
            self.reset()
        fraction_due = (
            progress.fraction >= 1.0
            or progress.fraction - self._last_fraction >= self.interval_fraction
        )
        step_due = progress.step - self._last_step >= self.interval_steps
        if not (fraction_due or step_due):
            return
        self._last_fraction = progress.fraction
        self._last_step = progress.step
        stream = self.stream
        message = (
            "sim1d progress: "
            f"{100.0 * progress.fraction:6.2f}% "
            f"step={progress.step} "
            f"t={progress.time:.6e}/{progress.t_end:.6e} s "
            f"dt={progress.accepted_dt:.3e} "
            f"elapsed={_format_duration(progress.wall_elapsed_s)} "
            f"eta={_format_duration(progress.wall_remaining_s)} "
            f"phase={progress.phase} "
            f"cap={progress.step_cap} "
            f"constraint={progress.active_constraint}"
        )
        if progress.timestep_limiters:
            message += " limiters=" + ",".join(
                f"{name}:{dt:.3e}" for name, dt in progress.timestep_limiters
            )
        if progress.retry_count:
            message += (
                f" retries={progress.retry_count}"
                f" reason={progress.rejection_reason}"
            )
        print(message, file=stream, flush=True)


class TimestepRejectionError(RuntimeError):
    """Raised when a timestep cannot be accepted after bounded retries."""

    def __init__(
        self,
        message,
        *,
        time=None,
        attempted_dt=None,
        retry_count=None,
        reason=None,
        dt_min=None,
        phase=None,
        active_constraint=None,
        rejection_detail=None,
    ):
        super().__init__(message)
        self.time = time
        self.attempted_dt = attempted_dt
        self.retry_count = retry_count
        self.reason = reason
        self.dt_min = dt_min
        self.phase = phase
        self.active_constraint = active_constraint
        self.rejection_detail = rejection_detail or {}
        self.details = {
            "time": time,
            "attempted_dt": attempted_dt,
            "retry_count": retry_count,
            "reason": reason,
            "dt_min": dt_min,
            "phase": phase,
            "active_constraint": active_constraint,
            "rejection_detail": self.rejection_detail,
        }


class BreakdownError(RuntimeError):
    """Raised when current-triggered plasma breakdown thresholds are missed."""

    def __init__(
        self,
        message,
        *,
        phase=None,
        time=None,
        I_tot=None,
        threshold=None,
        threshold_name=None,
        tau_prebreakdown=None,
        phase_events=None,
        current_trigger_samples=None,
    ):
        super().__init__(message)
        self.phase = phase
        self.time = time
        self.I_tot = I_tot
        self.threshold = threshold
        self.threshold_name = threshold_name
        self.tau_prebreakdown = tau_prebreakdown
        self.phase_events = phase_events
        self.current_trigger_samples = current_trigger_samples
        self.details = {
            "phase": phase,
            "time": time,
            "I_tot": I_tot,
            "threshold": threshold,
            "threshold_name": threshold_name,
            "tau_prebreakdown": tau_prebreakdown,
        }


def load_result_hdf5(path):
    """Load a saved sim1d HDF5 result without constructing a solver."""
    from .results.io import load_result_hdf5 as _load_result_hdf5

    return _load_result_hdf5(path)


def summarize_result(result):
    """Return lightweight health diagnostics for a sim1d run result."""
    from .results.health import summarize_result as _summarize_result

    return _summarize_result(result)


def _phase_event_arrays(events):
    return {
        "time": np.asarray([event[0] for event in events], dtype=float),
        "phase": np.asarray([event[1] for event in events], dtype=object),
        "reason": np.asarray([event[2] for event in events], dtype=object),
    }


def _current_trigger_sample_arrays(samples):
    return {
        "time": np.asarray([sample[0] for sample in samples], dtype=float),
        "I_tot": np.asarray([sample[1] for sample in samples], dtype=float),
    }


def _timestep_rejection_event_arrays(events):
    return {
        "time": np.asarray([event["time"] for event in events], dtype=float),
        "attempted_dt": np.asarray(
            [event["attempted_dt"] for event in events],
            dtype=float,
        ),
        "retry_index": np.asarray(
            [event["retry_index"] for event in events],
            dtype=float,
        ),
        "reason": np.asarray([event["reason"] for event in events], dtype=object),
        "phase": np.asarray([event["phase"] for event in events], dtype=object),
        "active_constraint": np.asarray(
            [event["active_constraint"] for event in events],
            dtype=object,
        ),
    }


def _copy_cache_value(value):
    if value is None:
        return None
    if hasattr(value, "copy"):
        return value.copy()
    return value


def _memo_key_part(value):
    """Return a hashable, BIT-EXACTLY comparable stand-in for ``value``.

    Used to build the read-only cathode solve memo's key
    (:meth:`LAPDSim1D._cathode_solve_memo_key`). Floats and arrays are
    encoded as their RAW BYTES rather than compared as numbers, so the
    resulting key separates values that ``==`` would conflate: ``-0.0`` from
    ``0.0``, and two states differing in the last ulp. NaN compares EQUAL to
    itself here, which is the correct reading for a memo -- the same bits in
    means the same solve out.

    The whole key is compared with ``==`` rather than hashed, so there is no
    collision channel: a hit means every component matched byte for byte.
    """
    if value is None or isinstance(value, (bool, int, str, bytes)):
        # bool before int matters only for repr; both are exact under ==.
        return (type(value).__name__, value)
    if isinstance(value, (float, np.floating)):
        return ("f", np.float64(value).tobytes())
    if isinstance(value, np.ndarray):
        return ("a", value.dtype.str, value.shape,
                np.ascontiguousarray(value).tobytes())
    if isinstance(value, dict):
        return ("d", tuple(sorted(
            (str(k), _memo_key_part(v)) for k, v in value.items()
        )))
    if isinstance(value, (list, tuple)):
        return ("t", tuple(_memo_key_part(v) for v in value))
    fields = getattr(value, "__dict__", None)
    if fields is not None:
        return ("o", type(value).__name__, tuple(sorted(
            (str(k), _memo_key_part(v)) for k, v in fields.items()
        )))
    # Anything else (an opaque object) is deliberately keyed by IDENTITY, so
    # a different object is a MISS. That errs toward re-solving, never toward
    # serving a stale memo.
    return ("id", id(value))


def _format_duration(seconds):
    if not np.isfinite(seconds):
        return "unknown"
    seconds = max(float(seconds), 0.0)
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, seconds = divmod(seconds, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)}m{seconds:02.0f}s"
    hours, minutes = divmod(minutes, 60.0)
    return f"{int(hours)}h{int(minutes):02d}m"


def _estimate_wall_remaining(elapsed_s, fraction):
    fraction = float(fraction)
    elapsed_s = float(elapsed_s)
    if fraction <= 0.0:
        return np.inf
    if fraction >= 1.0:
        return 0.0
    return elapsed_s * (1.0 - fraction) / fraction


def _timestep_limiters(diag, count=3):
    candidates = (
        ("plasma_cfl", diag.dt_plasma_cfl),
        ("front_density", diag.dt_front_density),
        ("surface_loss", diag.dt_surface_loss),
        ("neutral_exchange", diag.dt_neutral_exchange),
        ("neutral_sources", diag.dt_neutral_sources),
        ("reactions", diag.dt_reactions),
        ("energy_exchange", diag.dt_energy_exchange),
        ("electron_cooling", diag.dt_electron_cooling),
        ("ion_charge_exchange", diag.dt_ion_charge_exchange),
        ("heat_conduction", diag.dt_heat_conduction),
        ("ion_neutral_drag", diag.dt_ion_neutral_drag),
        ("circuit", diag.dt_circuit),
        ("dt_max", diag.dt_max),
    )
    finite = [
        (name, float(dt))
        for name, dt in candidates
        if np.isfinite(dt) and float(dt) >= 0.0
    ]
    finite.sort(key=lambda item: item[1])
    return tuple(finite[: max(int(count), 0)])


def _max_relative_change(before, after, scale_floor):
    before = np.asarray(before, dtype=float)
    after = np.asarray(after, dtype=float)
    scale = np.maximum(np.abs(before), float(scale_floor))
    return float(np.max(np.abs(after - before) / scale)) if before.size else 0.0


def _rejection_detail_text(detail):
    if not detail:
        return ""
    fields = detail.get("fields", {})
    if not fields:
        message = detail.get("message", "")
        return str(message) if message else ""
    parts = []
    for name, summary in fields.items():
        indices = summary.get("indices", [])
        count = summary.get("count", 0)
        parts.append(f"{name}[count={count}, indices={indices}]")
    return "; ".join(parts)


class LAPDSim1D:
    """Conservative axial 1D LAPD solver scaffold.

    This scaffold currently includes configuration, geometry, state handling,
    conservative plasma fluxes, pressure work, neutral terms, bulk reactions,
    and a minimal explicit SSPRK2 step.
    """

    def __init__(
        self,
        input_dict=None,
        input_flags=None,
        input_models=None,
        progress_callback=None,
        progress_tracker=None,
        progress_interval_s=1.0e-4,
        run_id=None,
    ):
        """Build a solver: resolve the config, then arm each subsystem.

        Construction runs as an ORDERED sequence of phases, and the order is
        load-bearing -- later refusals read flags earlier phases cached, and
        the initial condition is built only after everything that shapes it
        has been armed. The phases below are the same statements in the same
        order they have always run in; each ``_init_*`` method is one
        contiguous stretch of them, named.

        ``input_models`` carries DECLARATION BLOCKS -- one model family's
        complete membership per entry, stated without namespaces (see
        ``core/model_declarations.py``). They are projected onto the two flat
        namespaces before anything else happens, so every phase below reads
        the same flat surface it always has, and omitting them leaves
        construction bit-identical to the flat-only form.
        """
        if run_id is not None:
            from .results.phase3_capture import validate_run_id

            run_id = validate_run_id(run_id)
            self._run_id = run_id
        self._init_config_and_early_flags(
            input_dict,
            input_flags,
            input_models,
            progress_callback,
            progress_tracker,
            progress_interval_s,
        )
        validate_r1_configuration_presence(
            self._input_dict,
            self._flags,
            geometry=self._geometry,
            ion_neutral_moment_closure=self._ion_neutral_moment_closure,
            hyperbolic_wave_speed=self._hyperbolic_wave_speed,
            raw_stage_validation=self._raw_stage_validation,
        )
        # The declarative half of the deprecation surface (core/deprecations.py):
        # one DeprecationWarning per deprecated control this config actually
        # uses. It reads the canonical defaults from the config templates, so
        # default_config() construction is warning-free by construction and no
        # value is changed -- a deprecated path stays runnable and bit-identical.
        # The hand-written blocks in validate_r1_configuration_presence above
        # keep their own conditions and are NOT duplicated in that table.
        warn_deprecated_config(self._input_dict, self._flags, stacklevel=2)
        self._init_neutral_closure_selection()
        self._init_numerical_guards()
        self._init_neutral_momentum_and_energy()
        # Every RUN-CONSTANT subsystem bundle, resolved ONCE, here, now that
        # each of its inputs is itself resolved (the wall rate immediately
        # above is the last of them).  The `*_kwargs` accessors below read
        # this record instead of re-interpreting the config per RHS call.
        self._options = build_solver_options(
            self._input_dict,
            self._flags,
            geometry=self._geometry,
            gas_type=self._gas_type,
            I_ion=self._I_ion,
            electron_heat_flux_limit=self._electron_heat_flux_limit,
            heat_flux_limiter_f=self._heat_flux_limiter_f,
            heat_flux_limiter_exponent=self._heat_flux_limiter_exponent,
            neutral_energy=self._neutral_energy,
            neutral_energy_alpha=self._neutral_energy_alpha,
            neutral_energy_wall_Tn_eV=self._neutral_energy_wall_Tn_eV,
            neutral_energy_wall_rate=self._neutral_energy_wall_rate,
            wind_column_factor=self._wind_column_factor,
        )
        self._init_hot_neutral_channel_and_jets()
        self._init_atomic_package_refusals()
        self._init_floors_and_initial_state()
        self._init_run_and_circuit_state()
        self._init_cathode_surface_state()
        self._init_beam_transport_refusals()
        self._init_area_closures()
        self._init_run_machinery()

    def _init_config_and_early_flags(
        self,
        input_dict,
        input_flags,
        input_models,
        progress_callback,
        progress_tracker,
        progress_interval_s,
    ):
        """Resolve both config namespaces, the gas, the grid, and the early flags.

        The flags cached here are read by every phase below (and by the R1
        presence check that follows immediately), which is why they are
        resolved before any of them.

        TWO resolutions run here, in this order, and they answer different
        questions. ``resolve_config`` projects any DECLARATION BLOCK onto the
        flat namespaces and merges the caller's overrides onto the templates:
        that is where a family stated as its complete membership becomes the
        flat surface everything downstream reads. ``resolve_model_families``
        then runs on the merged config and BEFORE every validator in this
        constructor: a top-level model selection sets the member keys it owns
        that the caller left at their config defaults, and refuses ONCE --
        naming the selection, its whole member set and every offending key --
        when the caller explicitly set one of them to something the selection
        cannot carry. A config that arrived as a block has already stated
        every member, so the family resolver finds nothing to resolve and
        nothing to refuse; a flat config still gets its cascade flattened.

        The single-key guards those member sets flatten are all still below
        and are all still reached, by both routes; what changes is that a
        resolvable config no longer walks them one refusal at a time.
        """
        self._input_dict, self._flags = resolve_config(
            input_dict, input_flags, input_models
        )
        resolve_model_families(self._input_dict, self._flags)
        self._progress_callback = progress_callback
        self._progress_tracker = progress_tracker
        self._progress_interval_s = (
            1.0e-4 if progress_interval_s is None else progress_interval_s
        )
        self._gas_type = self._input_dict.get("gas_type")
        (
            self._ion_mass_g,
            self._mu,
            self._mu_neutral,
            self._I_ion,
        ) = self._gas_constants(self._gas_type)
        self._geometry = build_geometry(self._input_dict, self._flags)
        # Whether the plasma flux tube has a varying cross-section, and so
        # whether the quasi-1D p*dA/dz momentum source that pairs with the
        # area-weighted pressure flux must be built. The two ways to configure
        # one refuse each other at geometry construction (core.geometry), so
        # this is an either/or, and it is False on every uniform-column
        # configuration -- including the golden, which pins both flags off.
        self._variable_area_geometry = bool(
            self._flags.get("end_expansion_geometry")
        ) or bool(self._flags.get("prescribed_area_geometry"))
        self._active_plasma_topology = bool(
            self._flags.get("active_plasma_topology")
        )
        self._raw_stage_validation = bool(
            self._flags.get("raw_stage_validation")
        )
        self._hyperbolic_wave_speed = str(
            self._input_dict.get("hyperbolic_wave_speed")
        )
        self._hyperbolic_energy_consistent = bool(
            self._flags.get("hyperbolic_energy_consistent")
        )
        self._beam_anode_interception = bool(
            self._flags.get("beam_anode_interception")
        )
        # R4.3 / audit A7+A8: the moment-closed reduced ion-neutral collision
        # operator (Phelps He+/He). Presence-gated -- when on it replaces the
        # four legacy ion-neutral terms; when off it is a strict no-op. He-only.
        self._ion_neutral_moment_closure = bool(
            self._flags.get("ion_neutral_moment_closure")
        )
        if self._ion_neutral_moment_closure and self._gas_type != "He":
            raise ValueError(
                "ion_neutral_moment_closure uses the Phelps He+/He cross "
                f"sections and requires gas_type='He' (got {self._gas_type!r})"
            )
        # The presheath sigma_in model shares the He-only Phelps cross
        # section. Its two legacy arms ("constant", "cx_derived") were the
        # only non-helium path in the solver and were removed at D3.
        _sigma_in_model = str(self._input_dict.get("sigma_in_model"))
        if _sigma_in_model != "phelps":
            raise ValueError(
                f"sigma_in_model={_sigma_in_model!r} is not available: the "
                "legacy 'constant' and 'cx_derived' arms were removed at D3, "
                "2026-08-21. Accepted: 'phelps'."
            )
        if self._gas_type != "He":
            raise ValueError(
                "sigma_in_model='phelps' uses the Phelps He+/He cross section "
                f"and requires gas_type='He' (got {self._gas_type!r}); the "
                "non-helium arms were removed at D3, 2026-08-21 and the "
                "solver is helium-only"
            )

    def _init_neutral_closure_selection(self):
        """Select and arm the neutral closure: zones, model, and recycle routing.

        Everything here is a refusal or a run-constant topology resolved
        once -- the two-zone conductances, the kinetic bookkeeping record,
        the DVM accumulators -- in the order the refusals depend on.
        """
        validate_neutral_seed_cache_config(self._input_dict, self._flags)
        validate_equilibration_gas_puff_on(self._input_dict)
        _x = float(self._input_dict.get("R_comp_partition"))
        if not (0.0 <= _x <= 1.0):
            raise ValueError(
                "R_comp_partition (the external fraction of R_comp) must be in "
                f"[0, 1] (got {_x})"
            )
        _R_mesh = float(self._input_dict.get("R_mesh_ohm"))
        if _R_mesh < 0.0:
            raise ValueError(
                f"R_mesh_ohm (anode-mesh resistance) must be >= 0 (got {_R_mesh})"
            )
        exchange_model = str(
            self._input_dict.get("neutral_exchange_model")
        )
        if exchange_model not in ("constant", "knudsen"):
            raise ValueError(
                "neutral_exchange_model must be 'constant' or 'knudsen' "
                f"(got {exchange_model!r})"
            )
        validate_phase_config(
            self._phase_transition_mode(), self._prebreakdown_timeout_action()
        )
        validate_gas_puff_config(self._input_dict)
        if str(self._input_dict.get("gas_puff_profile")) == "orifice":
            # Derive the row ONCE, here, for the refusals that need the mesh:
            # a port off the grid and a column that is not inside the vessel
            # wall are configuration errors, and they must not first surface
            # from inside a right-hand side. The row is run-constant and
            # memoised, so this also pays its quadrature before the run.
            gas_puff_rate_profile(
                self._geometry,
                1.0,
                float(self._input_dict.get("gas_puff_valves")),
                profile="orifice",
                z_cm=self._input_dict.get("gas_puff_z_cm"),
                orifice_id_cm=self._input_dict.get("gas_puff_orifice_id_cm"),
                orifice_length_cm=self._input_dict.get(
                    "gas_puff_orifice_length_cm"
                ),
            )
        self._neutral_momentum = bool(self._flags.get("neutral_momentum"))
        if (
            self._neutral_momentum
            and self._input_dict.get("ion_neutral_drag_model")
            == "slip"
        ):
            raise ValueError(
                "neutral_momentum is mutually exclusive with "
                "ion_neutral_drag_model='slip': the slip closure is the "
                "evolved M_n equation's own local steady state"
            )
        self._neutral_two_zone = bool(self._flags.get("neutral_two_zone"))
        if self._neutral_two_zone:
            if (
                str(self._input_dict.get("neutral_exchange_model"))
                != "knudsen"
            ):
                raise ValueError(
                    "neutral_two_zone requires neutral_exchange_model="
                    "'knudsen': the per-zone conductances have no "
                    "constant counterpart"
                )
            # Geometry and Tn are fixed for the run: the zone volumes, the
            # radial exchange conductance, and the per-zone axial Knudsen
            # conductances are computed once here.
            self._zone_volumes = neutral_zone_volumes(self._geometry)
            self._check_annulus_not_collapsed()
            self._zone_exchange_cm3_s = neutral_zone_exchange_conductance(
                geometry=self._geometry,
                Tn_K=float(self._input_dict.get("Tn_K")),
                mu_neutral=self._mu_neutral,
            )
            self._zone_axial_coeffs = two_zone_knudsen_coefficients(
                geometry=self._geometry,
                Tn_K=float(self._input_dict.get("Tn_K")),
                mu_neutral=self._mu_neutral,
                clausing_scale=float(
                    self._input_dict.get("neutral_clausing_scale")
                ),
            )
        else:
            self._zone_volumes = None
            self._zone_exchange_cm3_s = None
            self._zone_axial_coeffs = None
        # Assembled two-zone backward-Euler matrices, keyed on the exact bits
        # of dt and the pump state -- everything else the assembly reads is one
        # of the run constants above. See _two_zone_implicit_matrix.
        self._two_zone_matrix_cache = {}
        # The one shared all-zero RHS bundle, built on first use. See
        # _zero_rhs_state.
        self._zero_rhs_state_shared = None
        self._neutral_model = str(
            self._input_dict.get("neutral_model")
        )
        if self._neutral_model not in ("moment", "kinetic", "kinetic_dvm"):
            raise ValueError(
                "neutral_model must be 'moment', 'kinetic' or 'kinetic_dvm' "
                f"(got {self._neutral_model!r})"
            )
        # Live cells against the plasma-terminating surfaces, by role: where the
        # active boundary term books its removal, and so where the kinetic arms'
        # wall-return channels must be read and deposited. Run-constant
        # topology, resolved once (see absorbing_live_cells_by_role).
        self._recycle_cells = absorbing_live_cells_by_role(self._geometry)
        self._end_recycle_to_annulus = bool(
            self._flags.get("end_recycle_to_annulus")
        )
        if self._end_recycle_to_annulus:
            if not self._neutral_two_zone:
                raise ValueError(
                    "end_recycle_to_annulus requires the neutral_two_zone "
                    "flag: the routed stream is deposited into the annulus "
                    "row nn_a, which only the two-zone closure builds"
                )
            routed_cells = self._recycle_cells.get("collector", ())
            V_ann = self._zone_volumes[1]
            dead = [
                int(cell)
                for cell in routed_cells
                if float(V_ann[int(cell)]) <= 0.0
            ]
            if dead:
                raise ValueError(
                    "end_recycle_to_annulus would route the collector recycle "
                    f"into cells with no annulus volume (V_ann = 0): {dead}. "
                    "The destination zone must exist wherever the routing "
                    "deposits, so a geometry whose plasma fills the vessel at "
                    "the collector face is refused rather than silently "
                    "destroying the routed stream"
                )
        if self._neutral_model == "kinetic":
            if not self._neutral_two_zone:
                raise ValueError(
                    "neutral_model='kinetic' rides on the two-zone state: "
                    "set the neutral_two_zone flag"
                )
            # K4a bookkeeping: targets and
            # per-cell relaxation times are produced by refresh-cadence
            # kinetic solves at step ACCEPTANCE (never inside trial RHS
            # evaluations); before the first refresh the moment terms
            # carry the neutrals (the pre-breakdown fill stays moment).
            self._kinetic = SimpleNamespace(
                engine=None,
                target_col=None,
                target_ann=None,
                tau_col=None,
                tau_ann=None,
                responses=None,       # per-channel unit-rate G_k (K4a-t)
                shapes=None,          # frozen channel shapes at refresh
                nu_ref=None,          # absorption field at last refresh
                grid=None,            # frozen shared velocity grid
                next_refresh_s=0.0,
                next_update_s=0.0,
                refresh_s=float(
                    self._input_dict.get("neutral_kinetic_refresh_s")
                ),
                refresh_tol=float(
                    self._input_dict.get("neutral_kinetic_refresh_tol")
                ),
                update_s=1.0e-5,
                nvz=int(self._input_dict.get("neutral_kinetic_nvz")),
                nvp=int(self._input_dict.get("neutral_kinetic_nvp")),
            )
            if self._kinetic.refresh_s <= 0.0:
                raise ValueError(
                    "neutral_kinetic_refresh_s must be positive"
                )
        else:
            self._kinetic = None
        self._dvm = None
        self._dvm_cadence_s = 0.0
        self._dvm_engaged = False
        self._dvm_next_s = 0.0
        self._dvm_last_s = 0.0
        # How many times the neutral clock has actually ticked. The cadence
        # states what was ASKED for; this is what the run DID, which is not
        # recoverable from the saved trajectory because a tick fires on the
        # neutral clock rather than on the save schedule.
        self._dvm_tick_count = 0
        self._dvm_transfer_relax_fraction = 1.0
        self._dvm_transfer_hold = KINETIC_DVM_TRANSFER_HOLDS[0]
        self._dvm_step_transfer = None
        # Exponential-hold tick state: the fluid momentum and ion energy the
        # tick's booked pair rate was measured against, and the constant
        # repayment rate the outstanding hold debt is retired at over this
        # tick. All three are frozen for the tick, exactly as the booked
        # transfer is.
        self._dvm_hold_M0 = np.zeros(self._geometry.cells, dtype=float)
        self._dvm_hold_Ei0 = np.zeros(self._geometry.cells, dtype=float)
        self._dvm_hold_repay_M = np.zeros(self._geometry.cells, dtype=float)
        self._dvm_hold_repay_Ei = np.zeros(self._geometry.cells, dtype=float)
        # Ionization the plasma has BOOKED since the last neutral tick, as a
        # per-cell density increment [cm^-3] on the plasma (= column) volume.
        # This is the counted quantity the kinetic arm debits; see
        # _accumulate_dvm_ion_booking.
        self._dvm_ion_booked = np.zeros(self._geometry.cells, dtype=float)
        self._dvm_ion_stage_weight = 0.0
        self._dvm_ion_stage_accum = None
        # The boundary-inflow channels the plasma has BOOKED since the last
        # neutral tick, in PARTICLES per cell. Same accumulator discipline
        # and the same stage weight as the ionization tally above; see
        # _accumulate_dvm_source_booking.
        self._dvm_source_booked = self._zero_dvm_source_counts()
        self._dvm_source_stage_accum = None
        self._dvm_source_rows = None
        self._dvm_ion_shortfall_warned = False
        # B5 cathode jet: the resolved spec (None when the channel is off or
        # the arm is not selected at all), the counted INCIDENT ion energy
        # the accepted steps have booked since the last neutral tick [erg per
        # column cell], and the two accumulators that carry it -- the same
        # stage weight, the same attempt lifetime and the same
        # reset-before-update discipline as the counted source rows above.
        self._dvm_cathode_jet = None
        self._dvm_cathode_jet_energy_booked = np.zeros(
            self._geometry.cells, dtype=float
        )
        # The COUNT partner of the accumulator above, carried on exactly the
        # steps that contributed to it. The two are booked over different
        # step sets whenever the arming criterion censors part of a tick --
        # the recycle count accrues on every step, the incident energy only
        # on armed ones -- and the directed share has to be drawn from the
        # armed counts for the per-atom launch energy to be the (R_E/R_N)
        # (phi_c + Ti) the channel asserts. Zero here is what makes a fully
        # censored tick route its whole stream thermally.
        self._dvm_cathode_jet_count_booked = np.zeros(
            self._geometry.cells, dtype=float
        )
        self._dvm_cathode_jet_energy_stage_accum = None
        self._dvm_cathode_jet_incident_row = None
        # B4 anode jet: the same five, anode-side. ``_anode_energy_ledger_J``
        # is the channel's own NAMED surface book and is PRESENCE-GATED --
        # ``None`` unless the channel is armed -- so an off run's saved
        # diagnostics and restart payload are the ones they always were.
        self._dvm_anode_jet = None
        self._dvm_anode_jet_energy_booked = np.zeros(
            self._geometry.cells, dtype=float
        )
        self._dvm_anode_jet_energy_stage_accum = None
        self._dvm_anode_jet_incident_row = None
        self._anode_energy_ledger_J = None
        if self._neutral_model == "kinetic_dvm":
            self._configure_kinetic_dvm()
        else:
            # The bounded-chord annulus is an operator inside the transient
            # DVM; asking for it anywhere else has nothing to act on, and a
            # selector that silently does nothing is the trap this refuses.
            flights = str(
                self._input_dict.get(
                    "neutral_kinetic_dvm_annulus_flights"
                )
            )
            if flights != "rates":
                raise ValueError(
                    "neutral_kinetic_dvm_annulus_flights selects an operator "
                    "of the transient DVM's annulus zone and has no meaning "
                    f"under neutral_model={self._neutral_model!r}. Accepted: "
                    "'rates' anywhere, or 'bounded_chord' with "
                    "neutral_model='kinetic_dvm' and the neutral_two_zone "
                    f"flag (got {flights!r})"
                )
            # Same statement for the cylindrical wall's reflection spectrum:
            # it selects what the transient DVM returns its non-accommodated
            # wall share on, and no other neutral model has such a share.
            #
            # The value accepted off-arm is the SHIPPED TEMPLATE DEFAULT,
            # read from the template itself. Off-arm the key has no meaning
            # under ANY of its values, so what the guard actually refuses is
            # a CONFIGURED choice sitting where it can do nothing -- and the
            # only value that is not a configured choice is the one that
            # arrives unasked. Spelling that as a position in
            # KINETIC_DVM_WALL_REFLECTION_MODELS coupled the guard to a tuple
            # ORDERING that merely happened to coincide with the default, and
            # the coincidence broke the moment the default moved (the
            # diffuse_elastic adoption, 2026-08-30): every moment-model build,
            # the golden included, refused at construction. Reading the
            # template keeps the guard correct through any future flip.
            inert = str(
                input_dict_template_1d["neutral_kinetic_dvm_wall_reflection"]
            )
            reflection = str(
                self._input_dict.get(
                    "neutral_kinetic_dvm_wall_reflection"
                )
            )
            if reflection != inert:
                configured = " / ".join(
                    repr(m)
                    for m in KINETIC_DVM_WALL_REFLECTION_MODELS
                    if m != inert
                )
                raise ValueError(
                    "neutral_kinetic_dvm_wall_reflection selects the "
                    "spectrum the transient DVM returns its non-accommodated "
                    "cylindrical-wall share on and has no meaning under "
                    f"neutral_model={self._neutral_model!r}. Accepted: "
                    f"{inert!r} (the shipped default) anywhere, or "
                    f"{configured} with neutral_model='kinetic_dvm' and "
                    f"the neutral_two_zone flag (got {reflection!r})"
                )
            # Same statement for the transfer hold: it selects how the
            # plasma integrates the DVM's tick-booked coupling term, and
            # there is no such term to integrate under any other model.
            hold = self._input_dict.get("neutral_kinetic_dvm_transfer_hold")
            if hold is not None:
                raise ValueError(
                    "neutral_kinetic_dvm_transfer_hold selects how the "
                    "plasma applies the transient DVM's tick-booked "
                    "coupling term and has no meaning under "
                    f"neutral_model={self._neutral_model!r}, which books no "
                    "such term. Accepted: leave it unset, or set it with "
                    f"neutral_model='kinetic_dvm' (got {hold!r})"
                )
            # Same statement for the cathode-side energetic recycle: it
            # splits the transient DVM's COUNTED cathode recycle stream, and
            # no other neutral model carries one to split. Its three
            # coefficients are refused the same way, so a config that sets
            # them and forgets the arm is loud rather than silently inert.
            if bool(
                self._input_dict.get(
                    "neutral_kinetic_dvm_cathode_jet"
                )
            ):
                raise ValueError(
                    "neutral_kinetic_dvm_cathode_jet splits the transient "
                    "DVM's counted cathode recycle into an energetic "
                    "backscatter share and a thermal remainder, and has no "
                    "meaning under "
                    f"neutral_model={self._neutral_model!r}, which carries no "
                    "such counted stream. Accepted: leave it off, or set it "
                    "with neutral_model='kinetic_dvm' and the "
                    "neutral_two_zone flag"
                )
            for _key in (
                "neutral_kinetic_dvm_cathode_jet_R_N",
                "neutral_kinetic_dvm_cathode_jet_R_E",
                "neutral_kinetic_dvm_cathode_jet_T_launch_eV",
            ):
                _given = self._input_dict.get(_key)
                _default = input_dict_template_1d.get(_key)
                if _given != _default:
                    raise ValueError(
                        f"{_key} parameterizes the transient DVM's cathode "
                        "backscatter channel and is read only under "
                        "neutral_kinetic_dvm_cathode_jet, which has no "
                        "meaning under "
                        f"neutral_model={self._neutral_model!r}. Accepted: "
                        f"leave it at {_default!r}, or arm the channel with "
                        f"neutral_model='kinetic_dvm' (got {_given!r})"
                    )
            # The same statement, anode-side: the channel splits the transient
            # DVM's COUNTED anode-mesh collection, and no other neutral model
            # carries one to split.
            if bool(
                self._input_dict.get("neutral_kinetic_dvm_anode_jet")
            ):
                raise ValueError(
                    "neutral_kinetic_dvm_anode_jet splits the transient DVM's "
                    "counted anode-mesh collection into an energetic "
                    "backscatter share and a thermal remainder, and has no "
                    "meaning under "
                    f"neutral_model={self._neutral_model!r}, which carries no "
                    "such counted stream. Accepted: leave it off, or set it "
                    "with neutral_model='kinetic_dvm' and the "
                    "neutral_two_zone flag"
                )
            for _key in (
                "neutral_kinetic_dvm_anode_jet_R_N",
                "neutral_kinetic_dvm_anode_jet_R_E",
                "neutral_kinetic_dvm_anode_jet_T_launch_eV",
            ):
                _given = self._input_dict.get(_key)
                _default = input_dict_template_1d.get(_key)
                if _given != _default:
                    raise ValueError(
                        f"{_key} parameterizes the transient DVM's anode "
                        "backscatter channel and is read only under "
                        "neutral_kinetic_dvm_anode_jet, which has no "
                        "meaning under "
                        f"neutral_model={self._neutral_model!r}. Accepted: "
                        f"leave it at {_default!r}, or arm the channel with "
                        f"neutral_model='kinetic_dvm' (got {_given!r})"
                    )
            # B6: the same statement for the baffle interception. It makes the
            # geometry's thin annular baffles act on the transient DVM's
            # ANNULUS, and no other neutral model carries such an annulus for
            # them to act on -- the fluid's own baffles ride neutral_baffles,
            # which is a different flag and is untouched here.
            if bool(self._flags.get("neutral_kinetic_dvm_baffles")):
                raise ValueError(
                    "neutral_kinetic_dvm_baffles makes the geometry's thin "
                    "annular baffles intercept the transient DVM's annulus "
                    "flux, and has no meaning under "
                    f"neutral_model={self._neutral_model!r}, which carries no "
                    "such annulus. The FLUID baffles are the separate "
                    "neutral_baffles flag and are unaffected. Accepted: leave "
                    "it off, or set it with neutral_model='kinetic_dvm', the "
                    "neutral_two_zone flag and neutral_baffles"
                )

    def _init_numerical_guards(self):
        """Validate the step-controller and non-ignition guards.

        These are budgets rather than physics: the Picard coupling, the
        heat-flux limiter, the floor-exempt drain, the max-step and dt_min
        locks, the wall-clock and accepted-step ignition caps, and the
        dt_growth recovery. Each is checked HERE so a misconfigured guard
        cannot be discovered hours into the run it exists to catch.
        """
        # R5.1 / audit A11: gated fluid<->circuit Picard coupling (default off).
        self._coupled_circuit_picard = bool(
            self._flags.get("coupled_circuit_picard")
        )
        if self._coupled_circuit_picard:
            if self._kinetic is not None:
                raise ValueError(
                    "coupled_circuit_picard is incompatible with the K4a kinetic "
                    "engine: the Picard re-run would double-trigger a kinetic "
                    "refresh, whose state the step snapshot does not restore"
                )
            if self._dvm is not None:
                raise ValueError(
                    "coupled_circuit_picard is incompatible with "
                    "neutral_model='kinetic_dvm': the Picard re-run would "
                    "advance the transient distribution a second time on the "
                    "same step, and the step snapshot does not restore it. "
                    "Accepted: kinetic_dvm with coupled_circuit_picard off"
                )
            self._circuit_picard_tol_rel = float(
                self._input_dict.get("circuit_picard_tol_rel")
            )
            self._circuit_picard_max_iter = int(
                self._input_dict.get("circuit_picard_max_iter")
            )
            if self._circuit_picard_tol_rel <= 0.0:
                raise ValueError(
                    "circuit_picard_tol_rel must be > 0 "
                    f"(got {self._circuit_picard_tol_rel})"
                )
            if self._circuit_picard_max_iter < 1:
                raise ValueError(
                    "circuit_picard_max_iter must be >= 1 "
                    f"(got {self._circuit_picard_max_iter})"
                )
            # Diagnostics: how many accepted steps triggered a Picard re-run, and
            # the total extra fluid re-solves (a null shows the frozen-current lag
            # is below the trigger at production dt).
            self._picard_triggered_steps = 0
            self._picard_extra_solves = 0
        # R5.2 / audit A9: flux-limited electron heat conduction (default on).
        self._electron_heat_flux_limit = bool(
            self._flags.get("electron_heat_flux_limit")
        )
        self._heat_flux_limiter_f = float(
            self._input_dict.get("heat_flux_limiter_f")
        )
        if self._electron_heat_flux_limit and self._heat_flux_limiter_f <= 0.0:
            raise ValueError(
                "heat_flux_limiter_f must be > 0 when electron_heat_flux_limit "
                f"is on (got {self._heat_flux_limiter_f})"
            )
        self._heat_flux_limiter_exponent = float(
            self._input_dict.get("heat_flux_limiter_exponent")
        )
        if self._electron_heat_flux_limit and self._heat_flux_limiter_exponent <= 0.0:
            raise ValueError(
                "heat_flux_limiter_exponent must be > 0 when "
                f"electron_heat_flux_limit is on (got {self._heat_flux_limiter_exponent})"
            )
        # Floor-aware drain exemption on the "surface_loss" dt bound (default
        # ON; presence-gated: None disables the exemption branch entirely so
        # the off path is bit-exact historical behavior).
        _surface_loss_floor_exempt = bool(
            self._flags.get("surface_loss_floor_exempt")
        )
        self._surface_loss_floor_exempt_rtol = (
            SURFACE_LOSS_FLOOR_EXEMPT_RTOL if _surface_loss_floor_exempt else None
        )
        # Hysteresis band on that exemption (armed at the default width; 0
        # restores the knife edge above). Validated here so a band that could
        # never be a band -- outer at or below the inner threshold, or wide
        # enough to be a permanent exemption -- or one armed with the
        # exemption itself off is refused before any compute.
        _exempt_exit_rtol = self._input_dict.get(
            "surface_loss_floor_exempt_exit_rtol"
        )
        try:
            _exempt_exit_value = float(_exempt_exit_rtol)
        except (TypeError, ValueError):
            _exempt_exit_value = np.nan
        if not np.isfinite(_exempt_exit_value) or _exempt_exit_value < 0.0:
            raise ValueError(
                "surface_loss_floor_exempt_exit_rtol must be a finite "
                "non-negative relative threshold, 0 to disable the hysteresis "
                f"band (got {_exempt_exit_rtol!r})"
            )
        if _exempt_exit_value > 0.0:
            if not _surface_loss_floor_exempt:
                raise ValueError(
                    "surface_loss_floor_exempt_exit_rtol is set "
                    f"({_exempt_exit_rtol!r}) while the surface_loss_floor_exempt "
                    "flag is off; the hysteresis band re-admits cells to an "
                    "exemption that is not running, so it could never do "
                    "anything. The band is armed by DEFAULT, so recovering "
                    "the historical bound takes both keys: pass "
                    "surface_loss_floor_exempt_exit_rtol=0.0 alongside the "
                    "cleared flag"
                )
            if _exempt_exit_value <= SURFACE_LOSS_FLOOR_EXEMPT_RTOL:
                raise ValueError(
                    "surface_loss_floor_exempt_exit_rtol must be strictly "
                    "greater than the entry threshold "
                    f"SURFACE_LOSS_FLOOR_EXEMPT_RTOL={SURFACE_LOSS_FLOOR_EXEMPT_RTOL!r} "
                    f"(got {_exempt_exit_rtol!r}); an outer threshold at or "
                    "below the inner one is not a band"
                )
            if _exempt_exit_value >= 1.0:
                raise ValueError(
                    "surface_loss_floor_exempt_exit_rtol must be strictly "
                    f"less than 1.0 (got {_exempt_exit_rtol!r}); the band is a "
                    "fraction of the per-cell floor energy 3/2 n T_floor, so "
                    "at or above 1.0 re-admission would require a margin "
                    "exceeding the floor energy itself -- that is not a "
                    "hysteresis band around the floor, and in the large limit "
                    "it is a permanent exemption from this bound"
                )
        # Presence gate: None keeps plasma_source_timestep on its
        # single-threshold expression and allocates no latch.
        self._surface_loss_floor_exempt_exit_rtol = (
            _exempt_exit_value if _exempt_exit_value > 0.0 else None
        )
        # Per-cell, per-energy-channel exempt memory, owned here because it is
        # run state rather than configuration: keyed "Ee"/"Ei", advanced by
        # suggest_timestep, and deliberately absent from the restart record.
        self._surface_loss_floor_exempt_latch = (
            {} if self._surface_loss_floor_exempt_exit_rtol is not None else None
        )
        _max_steps_action = str(
            self._input_dict.get("max_steps_action")
        )
        if _max_steps_action not in ("raise", "stop"):
            raise ValueError(
                "max_steps_action must be 'raise' or 'stop' "
                f"(got {_max_steps_action!r})"
            )
        self._max_steps_action = _max_steps_action
        # Consecutive-clamp bound on the dt_min lock. Validated here, at
        # construction, so a misconfigured guard cannot be discovered hours
        # into a run.
        _dt_min_lock_max_steps = self._input_dict.get(
            "dt_min_lock_max_steps"
        )
        try:
            _dt_min_lock_value = float(_dt_min_lock_max_steps)
        except (TypeError, ValueError):
            _dt_min_lock_value = np.nan
        if (
            not np.isfinite(_dt_min_lock_value)
            or _dt_min_lock_value != int(_dt_min_lock_value)
            or _dt_min_lock_value <= 0.0
        ):
            raise ValueError(
                "dt_min_lock_max_steps must be a positive integer number of "
                f"consecutive clamped steps (got {_dt_min_lock_max_steps!r})"
            )
        self._dt_min_lock_max_steps = int(_dt_min_lock_value)
        # Non-ignition guards measured in WALL CLOCK and in WORK, the two
        # budgets the simulated-time guards cannot see. Validated here so a
        # misconfigured guard cannot be discovered hours into the very crawl
        # it exists to catch.
        _wall_cap = self._input_dict.get("ignition_wall_clock_cap_s")
        try:
            _wall_cap_value = float(_wall_cap)
        except (TypeError, ValueError):
            _wall_cap_value = np.nan
        if not np.isfinite(_wall_cap_value) or _wall_cap_value < 0.0:
            raise ValueError(
                "ignition_wall_clock_cap_s must be a finite non-negative "
                f"number of seconds, 0 to disable (got {_wall_cap!r})"
            )
        self._ignition_wall_clock_cap_s = _wall_cap_value
        _step_cap = self._input_dict.get("ignition_accepted_step_cap")
        try:
            _step_cap_value = float(_step_cap)
        except (TypeError, ValueError):
            _step_cap_value = np.nan
        if (
            not np.isfinite(_step_cap_value)
            or _step_cap_value != int(_step_cap_value)
            or _step_cap_value < 0.0
        ):
            raise ValueError(
                "ignition_accepted_step_cap must be a non-negative integer "
                f"number of accepted steps, 0 to disable (got {_step_cap!r})"
            )
        self._ignition_accepted_step_cap = int(_step_cap_value)
        # Accelerated dt_growth re-approach. Validated at construction so a
        # factor that could never engage is refused before any compute.
        _growth_patience = self._input_dict.get("dt_growth_recovery_patience")
        try:
            _growth_patience_value = float(_growth_patience)
        except (TypeError, ValueError):
            _growth_patience_value = np.nan
        if (
            not np.isfinite(_growth_patience_value)
            or _growth_patience_value != int(_growth_patience_value)
            or _growth_patience_value < 0.0
        ):
            raise ValueError(
                "dt_growth_recovery_patience must be a non-negative integer "
                "number of consecutive dt_growth-capped steps, 0 to disable "
                f"(got {_growth_patience!r})"
            )
        self._dt_growth_recovery_patience = int(_growth_patience_value)
        _growth_recovery = self._input_dict.get("dt_growth_recovery_factor")
        try:
            _growth_recovery_value = float(_growth_recovery)
        except (TypeError, ValueError):
            _growth_recovery_value = np.nan
        if self._dt_growth_recovery_patience > 0:
            _growth_base = self._input_dict.get("dt_growth_factor")
            try:
                _growth_base_value = float(_growth_base)
            except (TypeError, ValueError):
                _growth_base_value = np.nan
            if (
                not np.isfinite(_growth_recovery_value)
                or not np.isfinite(_growth_base_value)
                or _growth_recovery_value <= _growth_base_value
            ):
                raise ValueError(
                    "dt_growth_recovery_factor must be finite and greater "
                    "than dt_growth_factor when dt_growth_recovery_patience "
                    f"is set (got recovery={_growth_recovery!r}, "
                    f"base={_growth_base!r}); a recovery factor at or below "
                    "the base factor could never accelerate anything"
                )
        self._dt_growth_recovery_factor = _growth_recovery_value
        # THE FOUR RUN-TIME-FIRST GUARDS, HOISTED (with the
        # declaration-block migration, 2026-08-30). Each of these four
        # domains was checked for the first time only once a run was already
        # moving -- dt_growth_factor at the top of run(), the two scheme
        # selectors on every split step and every heat substep, the drag model
        # inside an RHS term. Every one reads self._input_dict and nothing
        # else, so none of them needed run state to be checked; they were
        # simply never asked at construction. Checking them here is this
        # phase's whole point (see the docstring): a misconfigured guard must
        # not be discovered hours into the run it exists to catch.
        #
        # The per-call checks are DELIBERATELY LEFT IN PLACE, all four of them.
        # They are the authority on what each domain means, two of them stand
        # in modules that instruments call without a solver at all
        # (conduction.py, sources.py), and both scheme selectors are also
        # reachable through an explicit per-call argument that never passes
        # through the config. These hoists add a construction-time refusal;
        # they replace nothing.
        #
        # dt_growth_factor: the sibling immediately above already READS this
        # key at construction, but only to compare it against the recovery
        # factor and only when the recovery patience is armed, so the key's own
        # domain went unchecked until run().
        _dt_growth_enabled = bool(self._input_dict.get("dt_growth_enabled"))
        _dt_growth_factor = self._input_dict.get("dt_growth_factor")
        try:
            _dt_growth_factor_value = float(_dt_growth_factor)
        except (TypeError, ValueError):
            _dt_growth_factor_value = np.nan
        if _dt_growth_enabled and not _dt_growth_factor_value > 1.0:
            raise ValueError(
                "dt_growth_factor must be > 1 when dt growth is enabled (got "
                f"{_dt_growth_factor!r}); a factor at or below 1 could never "
                "grow the step, so the growth controller would be armed and "
                "inert. Accepted: a factor above 1, or "
                "dt_growth_enabled=False"
            )
        # The two time-integration scheme selectors. Their domains live with
        # the operators that implement them, and both validators are imported
        # rather than restated so a scheme added to either set is legal here
        # the moment it is legal there.
        validate_operator_splitting(
            self._input_dict.get("operator_splitting")
        )
        validate_implicit_heat_scheme(
            self._input_dict.get("implicit_heat_scheme")
        )
        # ion_neutral_drag_model. This is the one of the four that was not
        # merely late but UNREACHABLE on the shipped stance: the moment closure
        # returns a zero RHS before the drag term is ever evaluated, so a
        # typo'd model name was accepted at construction, ignored for the whole
        # run, and would have surfaced only on the day somebody turned the
        # closure off. The construction-time mutual-exclusion check against
        # neutral_momentum that already exists reads the key's VALUE without
        # checking that the value is one the solver implements.
        _drag_model = self._input_dict.get("ion_neutral_drag_model")
        if _drag_model not in ION_NEUTRAL_DRAG_MODELS:
            raise ValueError(
                "ion_neutral_drag_model must be one of "
                f"{sorted(ION_NEUTRAL_DRAG_MODELS)} (got {_drag_model!r}). "
                "It is checked here rather than at first use because under "
                "the shipped ion_neutral_moment_closure the drag term returns "
                "before reading it, so an unimplemented name would otherwise "
                "run to completion silently."
            )
        # beam_deposition_model, the FIFTH hoist of this class (2026-08-30).
        # Its failure mode is worse than late: there is no per-call check at
        # all. Every read in the solver and in physics/cathode.py is an
        # EQUALITY TEST against 'csda' with a beer_lambert fallback, so a
        # misspelled name is not refused anywhere -- it silently selects
        # beer_lambert and runs the whole discharge under a deposition closure
        # the caller did not ask for, with no line of output saying so.
        # 'cdsa' constructs and runs at base commit ca444dd; that is the
        # negative control, and declm_block_gate.py carries the recipe.
        _deposition_model = self._input_dict.get(
            "beam_deposition_model"
        )
        if _deposition_model not in BEAM_DEPOSITION_MODELS:
            raise ValueError(
                "beam_deposition_model must be one of "
                f"{sorted(BEAM_DEPOSITION_MODELS)} (got "
                f"{_deposition_model!r}). Every read of this key is an "
                "equality test against 'csda' with a 'beer_lambert' "
                "fallback, so an unrecognised name is not refused at the "
                "dispatch: it selects beer_lambert and runs to completion "
                "silently under a closure nobody selected."
            )
        # Presence gate for the beam_ionization_birth timestep bound. Reading
        # it once here keeps the off path out of the branch entirely.
        self._beam_ionization_birth_timestep_bound = bool(
            self._flags.get("beam_ionization_birth_timestep_bound")
        )
        # Global dt-refinement instrument. Validated here so a factor that
        # would loosen the step (>1) or stop the run dead (0, negative) is
        # refused before any compute, and read once so the unarmed run's
        # arithmetic never enters the multiply.
        _dt_global_scale = self._input_dict.get("dt_global_scale")
        try:
            _dt_global_scale_value = float(_dt_global_scale)
        except (TypeError, ValueError):
            _dt_global_scale_value = np.nan
        if (
            not np.isfinite(_dt_global_scale_value)
            or _dt_global_scale_value <= 0.0
            or _dt_global_scale_value > 1.0
        ):
            raise ValueError(
                "dt_global_scale must be a finite refinement factor in "
                f"(0, 1.0], 1.0 to disable (got {_dt_global_scale!r}); the "
                "instrument may only refine the accepted step, never loosen "
                "a bound"
            )
        self._dt_global_scale = _dt_global_scale_value
        # Rate-freezing instrument (I3). A real bool is required: the flag
        # switches which STATE the explicit operator's reaction terms read,
        # and an int or a string smuggled in there would arm a first-order
        # rate channel while reading like a value.
        _rates_at_accepted_state = self._flags.get(
            "rates_at_accepted_state"
        )
        if not isinstance(_rates_at_accepted_state, bool):
            raise ValueError(
                "rates_at_accepted_state must be a bool (got "
                f"{_rates_at_accepted_state!r})"
            )
        self._rates_at_accepted_state = _rates_at_accepted_state
        # Anode sheath-fall electron debit. A real bool is required for the
        # same reason: the flag arms a ~10^2 kW plasma electron sink, and an
        # int or a string smuggled in there would read like a value.
        _anode_sheath_full_debit = self._flags.get(
            "anode_sheath_full_debit"
        )
        if not isinstance(_anode_sheath_full_debit, bool):
            raise ValueError(
                "anode_sheath_full_debit must be a bool (got "
                f"{_anode_sheath_full_debit!r})"
            )
        # This flag COMPLETES the thermal-only electrode routing by adding the
        # sheath-fall share back onto the plasma electron store. That routing
        # is now unconditional (the stance that switched it off was retired
        # 2026-08-31, Tom), so the pairing this used to police -- arming the
        # debit against the full-P_anode_e routing, which would have debited
        # phi_a twice -- is no longer constructible and needs no refusal.
        self._anode_sheath_full_debit = _anode_sheath_full_debit
        # Beam electron-energy deposition re-homed into the implicit heat
        # substep. A real bool for the same reason as the two flags above: the
        # flag MOVES a ~10^5 W source between operators, and an int or a string
        # there would read like a value.
        _beam_deposition_in_heat_substep = self._flags.get(
            "beam_deposition_in_heat_substep"
        )
        if not isinstance(_beam_deposition_in_heat_substep, bool):
            raise ValueError(
                "beam_deposition_in_heat_substep must be a bool (got "
                f"{_beam_deposition_in_heat_substep!r})"
            )
        if _beam_deposition_in_heat_substep and not bool(
            self._flags.get("implicit_heat_conduction")
        ):
            raise ValueError(
                "beam_deposition_in_heat_substep requires "
                "implicit_heat_conduction (got implicit_heat_conduction="
                f"{self._flags.get('implicit_heat_conduction')!r}): the "
                "flag re-homes the beam electron-energy source from the "
                "explicit operator A into the implicit heat substep B, and "
                "with the split off there is no B to host it -- the source "
                "would leave A with nowhere to land"
            )
        self._beam_deposition_in_heat_substep = _beam_deposition_in_heat_substep
        # Census of the attracting-anode branch, advanced on ACCEPTED steps
        # only (see _accept_step_attempt) and exposed on an armed run's
        # cathode diagnostics. Initialised unconditionally so the attributes
        # always exist; only an armed run ever moves them or saves them.
        self._anode_attracting_steps = 0
        self._anode_attracting_last_time_s = np.nan
        # Census of the multi-group plateau's EDGE CLAMP, on the same
        # discipline: advanced on ACCEPTED steps only and exposed on an armed
        # run's cathode diagnostics. The clamp is what makes an unbracketed
        # plateau-edge solve loud instead of silent -- a run that spends
        # frames on the inelastic floor is running a different spectrum from
        # one that does not. Initialised unconditionally so the attributes
        # always exist; only an armed run ever moves them or saves them.
        self._plateau_edge_clamped_steps = 0
        self._plateau_edge_clamped_last_time_s = np.nan
        self._plateau_multigroup = str(
            self._input_dict.get("heating_anomalous_transport")
        ) == "plateau_multigroup"

    def _init_neutral_momentum_and_energy(self):
        """Arm the evolved neutral wind and its optional energy field.

        The radial closure, the neutral_energy field and its two hot-birth
        modifiers, the wall accommodation, the Knudsen temperature arm, and
        the two-zone wind factors -- each gated on the flag that gives it
        something to act on, so no selector here can be silently inert.
        """
        self._neutral_momentum_radial = str(
            self._input_dict.get("neutral_momentum_radial")
        )
        if self._neutral_momentum_radial not in (
            "uniform",
            "two_zone",
            "kinetic_two_moment",
        ):
            raise ValueError(
                "neutral_momentum_radial must be 'uniform', 'two_zone', "
                "or 'kinetic_two_moment' "
                f"(got {self._neutral_momentum_radial!r})"
            )
        if (
            self._neutral_momentum_radial in ("two_zone", "kinetic_two_moment")
            and not self._neutral_momentum
        ):
            raise ValueError(
                f"neutral_momentum_radial={self._neutral_momentum_radial!r} "
                "requires the "
                "neutral_momentum flag: it closes the radial profile of the "
                "evolved wind"
            )
        self._neutral_two_momentum = (
            self._neutral_momentum_radial == "kinetic_two_moment"
        )
        if self._neutral_two_momentum and not self._neutral_two_zone:
            raise ValueError(
                "neutral_momentum_radial='kinetic_two_moment' requires "
                "neutral_two_zone"
            )
        # Wall-branch momentum partition (default off, bit-exact off). The
        # cross section is presence-gated in BOTH directions so an armed flag
        # can never fall back on a defaulted number and a stray cross section
        # can never be silently inert.
        self._neutral_wall_momentum_partition = bool(
            self._flags.get("neutral_wall_momentum_partition")
        )
        _wall_sigma = self._input_dict.get(
            "neutral_wall_partition_sigma_hehe_cm2"
        )
        if self._neutral_wall_momentum_partition:
            if not self._neutral_two_momentum:
                raise ValueError(
                    "the neutral_wall_momentum_partition flag requires "
                    "neutral_momentum_radial='kinetic_two_moment': it "
                    "partitions the wall branch of the two-zone momentum "
                    "operator, and no other radial closure carries an annulus "
                    "momentum row for that branch to act on. Accepted: "
                    "neutral_wall_momentum_partition with "
                    "neutral_momentum_radial='kinetic_two_moment'"
                )
            if _wall_sigma is None:
                raise ValueError(
                    "the neutral_wall_momentum_partition flag requires "
                    "neutral_wall_partition_sigma_hehe_cm2: the He-He elastic "
                    "cross section sets the mean free path the survival "
                    "weight is built from and has no default, because no "
                    "literature-boxed value is carried in the solver"
                )
            _wall_sigma = float(_wall_sigma)
            if not np.isfinite(_wall_sigma) or _wall_sigma <= 0.0:
                raise ValueError(
                    "neutral_wall_partition_sigma_hehe_cm2 must be finite and "
                    f"strictly positive [cm^2]; got {_wall_sigma!r}"
                )
            self._neutral_wall_partition_sigma = _wall_sigma
        else:
            if _wall_sigma is not None:
                raise ValueError(
                    "neutral_wall_partition_sigma_hehe_cm2 is read only under "
                    "the neutral_wall_momentum_partition flag; setting it "
                    f"with the flag off (got {_wall_sigma!r}) would be an "
                    "inert control. Accepted: set both, or neither"
                )
            self._neutral_wall_partition_sigma = None
        # Evolved neutral thermal energy (default ON; bit-exact when off). The
        # field only means anything alongside the moment-closed collision
        # operator (which is what reads and feeds it) and an evolved wind
        # (which is what the frictional half is booked against), and there is
        # no consistent reading of it under a coverage deficit or a kinetic
        # neutral model -- so each of those is refused here rather than
        # silently resolved.
        self._neutral_energy = bool(self._flags.get("neutral_energy"))
        if self._neutral_energy:
            if not self._ion_neutral_moment_closure:
                raise ValueError(
                    "the neutral_energy flag requires "
                    "ion_neutral_moment_closure: the moment-closed collision "
                    "operator is the only term that reads the per-cell Tn and "
                    "books the neutral side of the exchange into En, so "
                    "without it the field would be evolved by nothing but the "
                    "wall sink. Accepted: ion_neutral_moment_closure with "
                    "neutral_momentum"
                )
            if not self._neutral_momentum:
                raise ValueError(
                    "the neutral_energy flag requires neutral_momentum: the "
                    "frictional half of the collisional energy is booked "
                    "against the relative velocity u - u_n, which has no "
                    "meaning without an evolved neutral wind. Accepted: "
                    "ion_neutral_moment_closure with neutral_momentum"
                )
            if bool(self._flags.get("coverage_closure")):
                raise ValueError(
                    "the neutral_energy flag is incompatible with "
                    "coverage_closure: the coverage deficit partitions nn "
                    "alone, so a single mean En spread over a concentrated "
                    "gas would assert a temperature relation between the "
                    "covered and uncovered fractions that nothing in the "
                    "model states. Accepted: neutral_energy without "
                    "coverage_closure"
                )
            if self._neutral_model != "moment":
                raise ValueError(
                    "the neutral_energy flag is incompatible with "
                    f"neutral_model={self._neutral_model!r}: a kinetic "
                    "neutral model already carries the neutral energy as a "
                    "moment of f, so an evolved En field would be a second, "
                    "unowned copy. Accepted: neutral_model='moment'"
                )
            if self._neutral_two_momentum:
                raise ValueError(
                    "the neutral_energy flag is incompatible with "
                    "neutral_momentum_radial='kinetic_two_moment': that "
                    "reduction gives the annulus its own momentum row while "
                    "nothing gives it an energy row, so the single cold fluid "
                    "the mini-flux transports would be split across two "
                    "momenta and one energy. Accepted: neutral_energy with "
                    "neutral_momentum_radial in ('uniform', 'two_zone')"
                )
        # Directed hot births (default off, bit-exact off). The drift it puts
        # into the flight kinematics is the hot channel's own launch velocity,
        # so the flag is meaningless without the channel that launches.
        self._neutral_hot_birth_drift = bool(
            self._flags.get("neutral_hot_birth_drift")
        )
        if self._neutral_hot_birth_drift and not self._neutral_energy:
            raise ValueError(
                "the neutral_hot_birth_drift flag requires neutral_energy: it "
                "directs the CX-born HOT channel's birth kinematics, and "
                "without neutral_energy there is no hot channel -- no "
                "ballistic kernel is built and the flag would be silently "
                "inert. Accepted: neutral_hot_birth_drift with "
                "neutral_energy=True, or neutral_hot_birth_drift=False"
            )
        # Internal walls for the ballistic flight (default ON; bit-exact when off).
        # Same dependency as the drift flag, for the same reason: it changes
        # where the hot channel's flights stop, and without the channel there
        # is no flight to stop.
        self._neutral_hot_internal_wall = bool(
            self._flags.get("neutral_hot_internal_wall")
        )
        if self._neutral_hot_internal_wall and not self._neutral_energy:
            raise ValueError(
                "the neutral_hot_internal_wall flag requires neutral_energy: "
                "it walls the CX-born HOT channel's ballistic flight at the "
                "closed and absorbing plasma faces, and without neutral_energy "
                "there is no hot channel -- no ballistic kernel is built and "
                "the flag would be silently inert. Accepted: "
                "neutral_hot_internal_wall with neutral_energy=True, or "
                "neutral_hot_internal_wall=False"
            )
        self._neutral_energy_alpha = float(
            self._input_dict.get("neutral_energy_wall_accommodation")
        )
        if not (0.0 <= self._neutral_energy_alpha <= 1.0):
            raise ValueError(
                "neutral_energy_wall_accommodation (the thermal "
                "accommodation coefficient alpha_E) must be in [0, 1] (got "
                f"{self._neutral_energy_alpha})"
            )
        # Thermal transpiration: which temperature the Knudsen conductances
        # read. The v1-primary freezes them at Tn_K; the disclosed arm scales
        # them by sqrt(Tn_local/Tn_K), which only exists where a per-cell Tn
        # does.
        self._neutral_knudsen_temperature = str(
            self._input_dict.get("neutral_knudsen_temperature")
        )
        if self._neutral_knudsen_temperature not in ("frozen", "local"):
            raise ValueError(
                "neutral_knudsen_temperature must be 'frozen' or 'local' "
                f"(got {self._neutral_knudsen_temperature!r})"
            )
        if (
            self._neutral_knudsen_temperature == "local"
            and not self._neutral_energy
        ):
            raise ValueError(
                "neutral_knudsen_temperature='local' (the thermal-"
                "transpiration arm) requires the neutral_energy flag: the "
                "scaling reads the evolved per-cell Tn, and without the En "
                "field there is only the config scalar Tn_K, against which "
                "the scale factor is identically 1. Accepted: "
                "neutral_knudsen_temperature='frozen', or 'local' with "
                "neutral_energy"
            )
        # Geometry and Tn_fit are fixed for the run, so the two-zone factors
        # are computed once here; None selects the uniform (legacy) closure
        # in every consumer.
        if self._neutral_momentum_radial == "two_zone":
            (
                self._wind_column_factor,
                self._wind_wall_rate,
            ) = neutral_wind_two_zone_factors(
                geometry=self._geometry,
                Tn_eV=float(self._input_dict.get("Tn_fit")),
                ion_mass_g=self._ion_mass_g,
            )
        else:
            self._wind_column_factor = None
            self._wind_wall_rate = None
        # The ENERGY channel's wall-visit rate is the same free-molecular
        # geometry evaluated at the wall's own temperature rather than the
        # momentum closure's 0.1 eV Tn_fit: the gas that trades energy with a
        # surface is the near-wall gas, which the v1 cut holds at T_wall. The
        # two-zone factors are linear in vbar, so this is the same closure with
        # the right speed in it, not a different one.
        self._neutral_energy_wall_Tn_eV = (
            NEUTRAL_ENERGY_FLOOR_T_K * kb_cgs / ev_to_erg
        )
        if self._neutral_energy and self._neutral_momentum_radial == "two_zone":
            _, self._neutral_energy_wall_rate = neutral_wind_two_zone_factors(
                geometry=self._geometry,
                Tn_eV=self._neutral_energy_wall_Tn_eV,
                ion_mass_g=self._ion_mass_g,
            )
        else:
            self._neutral_energy_wall_rate = None

    def _init_hot_neutral_channel_and_jets(self):
        """Build the ballistic flight kernels and resolve the recycle jets."""
        # The ballistic redistribution kernels are geometry alone (the flight
        # speed cancels out of the axial hop), so they are built once here and
        # never re-entered.
        if self._neutral_energy:
            self._hot_neutral_kernels = ballistic_flight_kernels(
                self._geometry,
                internal_wall=self._neutral_hot_internal_wall,
            )
        else:
            self._hot_neutral_kernels = None
        self._hot_channel_diagnostics = {}
        self._birth_deficit_diagnostics = {}
        _jet = resolve_neutral_jet_config(
            self._input_dict,
            geometry=self._geometry,
            neutral_momentum=self._neutral_momentum,
            neutral_energy=self._neutral_energy,
        )
        self._cathode_jet_enabled = _jet.cathode_jet_enabled
        self._anode_jet_enabled = _jet.anode_jet_enabled
        self._mesh_accommodation = _jet.mesh_accommodation
        self._cathode_jet_R_N = _jet.cathode_jet_R_N
        self._cathode_jet_R_E = _jet.cathode_jet_R_E
        self._anode_jet_R_N = _jet.anode_jet_R_N
        self._anode_jet_R_E = _jet.anode_jet_R_E
        self._cathode_jet_energy_convention = (
            _jet.cathode_jet_energy_convention
        )
        self._anode_jet_energy_convention = (
            _jet.anode_jet_energy_convention
        )
        self._cathode_surface_ion_retention = (
            _jet.cathode_surface_ion_retention
        )
        self._cathode_jet_carrier = _jet.cathode_jet_carrier
        self._jet_carrier_diagnostics = {}
        self._mesh_faces = _jet.mesh_faces
        self._mesh_blocked_area_cm2 = _jet.mesh_blocked_area_cm2
        # Cathode-jet arming criterion. Validated unconditionally (the keys
        # are always present), but ARMED only by a positive arm threshold:
        # ``_jet_arming_active`` is the presence gate every consumer below
        # tests first, so the inert declaration cannot reach the latch at all.
        (
            self._jet_arm_current_A,
            self._jet_disarm_current_A,
            self._jet_arming_active,
        ) = resolve_jet_arming_criterion(self._input_dict)
        # The latch itself. With no criterion declared it is permanently True
        # and no consumer can distinguish this build from the one before it.
        # With a criterion it starts DISARMED: before the discharge has drawn
        # its arming current there is no bombardment for the jets to model,
        # and the first accepted step at or above the arm threshold is what
        # brings them into existence.
        self._jet_armed = not self._jet_arming_active
        # Census of the criterion's own behaviour, for the run summary: how
        # many accepted steps were censored and where the latch transitioned.
        self._jet_arming_censored_steps = 0
        self._jet_arming_transitions = 0
        self._jet_arming_last_transition_s = float("nan")

    def _init_atomic_package_refusals(self):
        """Refuse atomic-package combinations that would double-book photons."""
        self._recombination_energy_return = bool(
            self._input_dict.get("recombination_energy_return")
        )
        if self._recombination_energy_return:
            if (
                str(self._input_dict.get("atomic_rate_model"))
                != "adas"
            ):
                raise ValueError(
                    "recombination_energy_return requires "
                    "atomic_rate_model='adas' (the PRB radiated-power "
                    "booking has no janev counterpart)"
                )
            if bool(self._flags.get("icool_recomb")):
                raise ValueError(
                    "recombination_energy_return already charges the full "
                    "PRB; combining it with icool_recomb double-charges "
                    "the recombination photons"
                )
        if bool(
            self._input_dict.get("adas_low_te_extension")
        ) and bool(self._flags.get("icool_recomb")):
            raise ValueError(
                "adas_low_te_extension must not be combined with "
                "icool_recomb: icool_recomb charges bare PRB, and "
                "adas_low_te_extension amplifies the sub-edge PRB by "
                "~9,300x, so the electron fluid runs away thermally to the "
                "Te floor and the electron_cooling timestep bound collapses "
                "permanently. The consistent net booking "
                "(I_ion*S_rec - P_PRB) that would make the pair sound is "
                "not built. The deep-afterglow low-Te recipe that paired "
                "them is RETIRED; without that booking the afterglow "
                "validity window is Te > 0.2 eV (the ADF11 edge)"
            )

    def _init_floors_and_initial_state(self):
        """Set the floors, arm the initial-condition features, and build state.

        Order is load-bearing: the tracer and the shaped neutral fill are
        armed BEFORE the initial condition, because the construction floor
        is the thing the tracer has to be exempt from and the fill is the
        only thing the profile touches.
        """
        self._floors = {
            "n": float(self._input_dict["ne_floor"]),
            "nn": float(self._input_dict["nn_floor"]),
            "Te": float(self._input_dict["Te_floor"]),
            "Ti": float(self._input_dict["Ti_floor"]),
        }
        self._floor_ledger = self._empty_floor_ledger()
        # Regime-R2 pre-breakdown passive tracer (default off, bit-exact off).
        # Armed HERE, before the initial condition is floored, because the
        # floor is the thing it has to be exempt from: ``ne0 = 0`` is a
        # legitimate true-vacuum start under the tracer, and the construction
        # floor would otherwise clip it to ``ne_floor`` before any feature
        # existed to object. Everything it validates is already built (the
        # floors, the geometry, both config namespaces, the topology flag).
        self._configure_regime_tracer()
        # Shaped initial neutral fill (default off, bit-exact off). Armed
        # HERE, before the initial condition is built, because the initial
        # condition is the only thing it touches.
        self._configure_neutral_initial_profile()
        initial_raw = self._initial_state()
        self._state = apply_state_floors(
            initial_raw, self._floors, self._ion_mass_g
        )
        if self._tracer is not None:
            self._state = self._tracer_exempt_initial_floor(
                initial_raw, self._state
            )
        self._accumulate_floor_ledger(
            self._floor_additions(initial_raw, self._state)
        )
        self._init_sample_smoothing()
        self._y = pack_state(self._state)
        self._derived = derive_state(self._state, self._floors, self._ion_mass_g)

    def _init_run_and_circuit_state(self):
        """Initialize the run-loop, ignition-monitor and circuit state.

        The cathode model validators run here, not at the first solve: an
        unknown model string or an unsupported combination must fail at
        construction.
        """
        self._time = 0.0
        self._t_prebreakdown_trigger = None
        self._t_breakdown_trigger = None
        self._last_current_trigger_time = None
        self._last_current_trigger_I_tot = None
        self._current_trigger_samples = []
        self._run_start_for_phase_events = 0.0
        # Ignition-failure guards. The monitor is always on and is armed only
        # while the cathode drive is active in pre_breakdown/breakdown, so it
        # is inert on any run that reaches main_discharge (BY CONSTRUCTION,
        # not by tuning). ``_t_ignition_abort`` is the shared switch-open
        # instant used by both the stall trip and the tau_prebreakdown
        # hardware-guard timeout.
        self._ignition_monitor = IgnitionMonitor()
        self._t_ignition_abort = None
        self._ignition_abort_reason = None
        self._ignition_abort_context = None
        self._ignition_abort_threshold_name = None
        self._last_ignition_record = None
        self._cathode_x0 = None
        self._cathode_x0_twin = None
        self._cathode_beam_cross = np.zeros(self._geometry.cells)
        # A2a lag state: the electron current [A] the anode collected directly
        # from the QL tail walkers on the LAST ACCEPTED step. The deposition
        # that measures it runs after the circuit within a step, so the
        # coupling is lagged; this member is written ONLY where the accepted
        # solve writes its caches, so a rejected attempt cannot move it.
        self._cathode_tail_anode_I = 0.0
        # Last accepted sheath solve's current, used only by the measured-tail
        # phase gate. The evolved loop state itself is _circuit_I_loop.
        self._circuit_I_prev = 0.0
        self._circuit_V_cap = None  # lazily V_bank; drains when C_bank_F set
        # Fail at construction, not at the first cathode solve mid-run:
        # unknown model strings and the unsupported TwinCathode combination.
        validate_cathode_Rp_model(self._input_dict, self._flags)
        validate_cathode_lnL_model(self._input_dict)
        self._cathode_solver_model = validate_cathode_solver_model(
            self._input_dict, self._flags
        )
        # Circuit voltage bound (R1): a ceiling on the device voltage set by
        # what the loop can supply. It lives inside the current-driven sheath
        # solve and is formed from the bank/capacitor source, so a
        # configuration that has neither would turn it on and get nothing --
        # exactly the silent no-op the house rules forbid.
        if bool(self._flags.get("cathode_circuit_voltage_bound")):
            if not bool(self._flags.get("cathode_coupling")):
                raise ValueError(
                    "cathode_circuit_voltage_bound requires the "
                    "cathode_coupling flag: with no cathode solve there is "
                    "no device voltage to bound"
                )
            if not float(self._input_dict.get("V_bank")) > 0.0:
                raise ValueError(
                    "cathode_circuit_voltage_bound requires a positive "
                    f"V_bank (got {self._input_dict.get('V_bank')!r}); "
                    "the bound is the source voltage minus the series "
                    "resistive drop, so a zero source would leave it "
                    "permanently inactive"
                )
            # Which quantity the available voltage bounds. Validated here so a
            # typo is a construction error rather than a run that silently
            # bounds nothing; the sheath solve validates it again where it is
            # consumed, because it is reachable from callers that never build
            # a solver.
            _bound_object = circuit_bound_object(self._input_dict)
            if _bound_object not in _CIRCUIT_BOUND_OBJECTS:
                raise ValueError(
                    "cathode_circuit_bound_object must be one of "
                    f"{sorted(_CIRCUIT_BOUND_OBJECTS)} (got "
                    f"{_bound_object!r})"
                )
        # Current-driven circuit state: the loop current, integrated once
        # per accepted step.
        self._circuit_I_loop = 0.0
        # Step-integrated discharge voltage (the inductor's view) from the
        # last accepted circuit advance; 0.0 under open circuit.
        self._circuit_V_dis_step = 0.0
        # Running time integral of V_dis_step [V*s] over accepted circuit
        # steps. The per-step instantaneous value is dt-BIASED as a saved
        # trace: saves land on dt-capped steps, which systematically sample
        # the low state of the knee sawtooth (measured 2026-07-21 on
        # es1_nx120_m6_sq3500: saved plateau mean 126 V vs 151 V dt-weighted
        # average, with V_b and the loop reconstruction both at 151 V).
        # The saved circuit_V_dis_step diagnostic is therefore the
        # save-interval average of this integral, not the raw sample; the
        # (time, integral) pair at the previous trajectory save anchors it.
        self._circuit_V_dis_time_integral = 0.0
        self._circuit_V_dis_prev_save = None

    def _init_cathode_surface_state(self):
        """Arm the vessel node and the evolving cathode surface state.

        The warming model's surface temperature and the ads/des coverage
        are the two pieces of cathode state that evolve over a shot; both
        are ``None`` under their default 'none' selectors, which is the
        presence gate every consumer reads.
        """
        # Vessel / common-mode node (default absent). ``_vessel`` is the
        # resolved constant record or None; ``_vessel_V_cm`` is the node's
        # single state variable and stays None while the node is absent, so
        # every consumer is presence-gated on one object.
        self._configure_regime_vessel_node()
        # Cathode warming state: the evolving emitter surface temperature [K]
        # (config cathode_warming_model). None = static T_s.
        warming_model = str(
            self._input_dict.get("cathode_warming_model")
        )
        if warming_model not in ("none", "power_balance"):
            raise ValueError(
                "cathode_warming_model must be 'none' or 'power_balance' "
                f"(got {warming_model!r})"
            )
        self._cathode_warming_model = warming_model
        self._cathode_Ts_K = None
        if warming_model == "power_balance":
            Ts_base = self._input_dict.get("cathode_Ts_base_K")
            if Ts_base is None:
                raise ValueError(
                    "cathode_warming_model='power_balance' requires "
                    "cathode_Ts_base_K (the heater-maintained standby "
                    "surface temperature)"
                )
            if float(
                self._input_dict.get("cathode_heat_capacity_J_per_K")
            ) <= 0.0:
                raise ValueError(
                    "cathode_heat_capacity_J_per_K must be positive"
                )
            eps = float(self._input_dict.get("cathode_emissivity"))
            if not 0.0 < eps <= 1.0:
                raise ValueError(
                    f"cathode_emissivity must be in (0, 1] (got {eps})"
                )
            if float(
                self._input_dict.get("cathode_conduction_W_per_K")
            ) < 0.0:
                raise ValueError(
                    "cathode_conduction_W_per_K must be non-negative"
                )
            if float(Ts_base) <= CATHODE_ENV_T_K:
                raise ValueError(
                    "cathode_Ts_base_K must exceed the "
                    f"{CATHODE_ENV_T_K:g} K chamber-wall temperature"
                )
            self._cathode_Ts_K = float(Ts_base)
        # Surface-state coverage (cathode_surface_model="ads_des",
        # M5a): theta in [0, 1] is the contaminant
        # coverage raising the effective work function,
        # phi_eff = phi_clean + (phi_wf - phi_clean) * theta, evolving as
        #   dtheta/dt = -sigma Gamma_i theta
        # (ion-stimulated desorption -- M5a, the fluence-cleaning limit --
        # which is the only coverage channel, so theta is monotonically
        # non-increasing). None = static phi_wf (historical).
        surface_model = str(
            self._input_dict.get("cathode_surface_model")
        )
        if surface_model not in ("none", "ads_des"):
            raise ValueError(
                "cathode_surface_model must be 'none' or 'ads_des' "
                f"(got {surface_model!r})"
            )
        self._cathode_theta = None
        if surface_model == "ads_des":
            clean = self._input_dict.get("cathode_phiwf_clean_eV")
            if clean is None:
                raise ValueError(
                    "cathode_surface_model='ads_des' requires "
                    "cathode_phiwf_clean_eV (the per-shot-accessible floor)"
                )
            if not float(clean) < float(self._input_dict["phi_wf"]):
                raise ValueError(
                    "cathode_phiwf_clean_eV must be below phi_wf (the "
                    "contaminated shot-start value)"
                )
            if float(self._input_dict.get("cathode_cleaning_sigma_cm2")) < 0.0:
                raise ValueError(
                    "cathode_cleaning_sigma_cm2 must be non-negative"
                )
            self._cathode_theta = 1.0
        # Per-shot surface energy ledger [J]: running integrals of the
        # balance terms over accepted steps. These FIVE rows are
        # power_balance only and stay zero otherwise (the presence-gated
        # backscatter row added just below is not -- see its own note). The net
        # (heater + ion - rad - emis - cond) is the shot's unreturned
        # energy into the emitting skin; cond is what the heater-held
        # substrate absorbed -- the quantity Tom's open-loop-heater drift
        # hypothesis makes checkable against the ES1 trim cadence
        # (a ~sub-kW net imbalance corresponds to ±8 K per 20-30 min).
        self._cathode_energy_ledger_J = {
            "heater": 0.0, "ion": 0.0, "rad": 0.0, "emis": 0.0, "cond": 0.0,
        }
        if self._dvm_cathode_jet is not None:
            # The DVM cathode jet's own named loss row: the R_E share of the
            # ion bombardment energy that leaves the surface WITH the
            # backscattered atoms. PRESENCE-GATED, like the term keys the
            # arm's other channels carry -- the row exists exactly when
            # something books into it, so an off run's ledger, its saved
            # diagnostics and its restart payload are the ones it always was.
            self._cathode_energy_ledger_J["backscatter"] = 0.0
        if self._dvm_anode_jet is not None:
            # The ANODE's own surface energy book, which did not exist before
            # this channel: the anode mesh has no warming model and so had no
            # balance to carry a row in. Two rows, both cumulative [J] and
            # both booked per ACCEPTED step from the same committed
            # (particles, incident energy) pair the birth is formed from:
            #
            #   ion_incident  what the collected ions delivered to the wires,
            #                 sum over accepted steps of N (phi_a + Ti);
            #   backscatter   what left again with the R_N energetic share,
            #                 exactly R_E of it.
            #
            # Their DIFFERENCE is the statement "the anode absorbs 1 - R_E",
            # which is thereby a subtraction of two measured rows rather than
            # a remembered convention. PRESENCE-GATED for the reason the
            # cathode's backscatter row is: absent the channel there is no
            # book at all (``None``), so an off run's saved diagnostics and
            # its restart payload are byte-unchanged.
            self._anode_energy_ledger_J = {
                "ion_incident": 0.0, "backscatter": 0.0,
            }

    def _init_beam_transport_refusals(self):
        """Refuse incomplete beam-deposition and anomalous-transport configs.

        Every selector below is validated at CONSTRUCTION even though the
        deposition module validates it again where it is consumed: a
        misconfiguration must fail before the first cathode solve, and a
        combination the walk machinery could not act on is an incomplete
        configuration rather than a silent no-op.
        """
        if float(self._input_dict.get("beam_deposition_smoothing_cm")) < 0.0:
            raise ValueError(
                "beam_deposition_smoothing_cm must be >= 0 (got "
                f"{self._input_dict.get('beam_deposition_smoothing_cm')})"
            )
        # WP-D non-local beam-product transport. String selector; the module
        # validates it too, but a bad value must fail at CONSTRUCTION rather
        # than on the first cathode solve. Selecting "nonlocal" under
        # beer_lambert is an INCOMPLETE configuration -- that path never
        # launches the CSDA module, so the requested physics could not act --
        # and raises rather than silently doing nothing.
        _bpt = str(self._input_dict.get("beam_product_transport"))
        if _bpt not in ("local", "nonlocal", "terminal_nonlocal"):
            raise ValueError(
                "beam_product_transport must be 'local', 'nonlocal' or "
                f"'terminal_nonlocal' (got {_bpt!r})"
            )
        if _bpt != "local" and str(
            self._input_dict.get("beam_deposition_model")
        ) != "csda":
            raise ValueError(
                f"beam_product_transport={_bpt!r} requires "
                "beam_deposition_model='csda' (the products it transports "
                "are the CSDA ray's; under beer_lambert it would be a "
                "silent no-op)"
            )
        # The wall-charge leg of "terminal_nonlocal": the walked terminal
        # electrons that reach an end land on a terminating surface, so their
        # CURRENT joins the vessel node's electron channel while their energy
        # leaves through the deposition module's end ledger. Presence-gated on
        # BOTH the selector and an armed node (``_configure_regime_vessel_node``
        # ran above), and resolved once here so no consumer re-reads a
        # selector: with either absent this is False and the node's electron
        # current is the transmitted-primary sum it always was.
        self._beam_terminal_wall_charge = (
            _bpt == "terminal_nonlocal" and self._vessel is not None
        )
        # The anomalous closure family. Same discipline: the module validates
        # too, but a bad selector must fail at CONSTRUCTION rather than on the
        # first cathode solve, and selecting a closure the deposition path
        # never launches is an INCOMPLETE configuration, not a no-op.
        _bam = str(self._input_dict.get("beam_anomalous_model"))
        if _bam not in ANOMALOUS_MODELS:
            raise ValueError(
                "beam_anomalous_model must be one of "
                f"{sorted(ANOMALOUS_MODELS)} (got {_bam!r})"
            )
        if _bam == "ql_relaxation":
            if str(
                self._input_dict.get("beam_deposition_model")
            ) != "csda":
                raise ValueError(
                    "beam_anomalous_model='ql_relaxation' requires "
                    "beam_deposition_model='csda': the anomalous channel "
                    "exists only on the CSDA rays, so under beer_lambert the "
                    "closure could not act and the run would read as though "
                    "the middle leg were live when nothing is booked"
                )
            _qlrc = self._input_dict.get("ql_relaxation_coeff")
            if _qlrc is None:
                raise ValueError(
                    "beam_anomalous_model='ql_relaxation' requires "
                    "ql_relaxation_coeff (the registered O(10-100) "
                    "plateau-formation bracket constant); it is deliberately "
                    "not defaulted at the point of use, because a headline "
                    "under this closure is quoted at the bracket endpoints "
                    "and an unstated arm cannot be reported"
                )
            _qlrc = float(_qlrc)
            if not math.isfinite(_qlrc) or _qlrc <= 0.0:
                raise ValueError(
                    "ql_relaxation_coeff must be finite and > 0 (got "
                    f"{self._input_dict.get('ql_relaxation_coeff')})"
                )
        # WP-E QL heating locality. Same discipline as WP-D above: the module
        # validates too, but a misconfiguration must fail at CONSTRUCTION, and
        # every combination in which the walk machinery could not act is an
        # INCOMPLETE configuration rather than a silent no-op. The walk needs
        # (a) the CSDA module to be the thing depositing, and (b) an anomalous
        # channel actually producing the power it carries.
        _hat = str(self._input_dict.get("heating_anomalous_transport"))
        if _hat not in ("local", "tail_walk", "plateau_multigroup"):
            raise ValueError(
                "heating_anomalous_transport must be 'local', 'tail_walk' or "
                f"'plateau_multigroup' (got {_hat!r})"
            )
        _multigroup = _hat == "plateau_multigroup"
        if _multigroup:
            # The multi-group plateau DERIVES its birth spectrum from the
            # extraction (the plateau edge solved on the launch cell's own
            # Maxwellian, then N equal-power groups up to e*phi_c), so the
            # single-line closure's three birth-energy controls are inert
            # under it. Inert is exactly what this repo refuses to be silent
            # about: each is rejected here, at construction, rather than
            # accepted and ignored. Their DEFAULTS are accepted, so an armed
            # stance states the selector; a block-form stance also states the
            # inert dials at their defaults, which the guard accepts.
            if self._input_dict.get(
                "heating_anomalous_tail_phi_c_fraction"
            ) is not None:
                raise ValueError(
                    "heating_anomalous_tail_phi_c_fraction was supplied with "
                    "heating_anomalous_transport='plateau_multigroup', where "
                    "the birth spectrum runs from the derived plateau edge "
                    "E_1 up to e*phi_c and the f dial is inert (the shipped "
                    "f=0.25 line was the WAVE share's stand-in and f=1.0 the "
                    "streaming share's; this closure carries both). Drop the "
                    "fraction, or select heating_anomalous_transport="
                    "'tail_walk' to launch at one keyed energy"
                )
            if float(
                self._input_dict.get(
                    "heating_anomalous_tail_energy_eV"
                )
            ) != 75.0:
                raise ValueError(
                    "heating_anomalous_tail_energy_eV="
                    f"{self._input_dict.get('heating_anomalous_tail_energy_eV')}"
                    " was supplied with heating_anomalous_transport="
                    "'plateau_multigroup', where the birth energies are the "
                    "derived group midpoints and the fixed rung is inert; "
                    "drop it, or select heating_anomalous_tail_energy_keying="
                    "'fixed' under heating_anomalous_transport='tail_walk'"
                )
            if str(
                self._input_dict.get(
                    "heating_anomalous_tail_energy_keying"
                )
            ) != "phi_c":
                raise ValueError(
                    "heating_anomalous_tail_energy_keying was supplied with "
                    "heating_anomalous_transport='plateau_multigroup', which "
                    "keys the spectrum's TOP to the live e*phi_c and its "
                    "BOTTOM to the solved plateau edge; there is no rung to "
                    "select and the keying selector is inert. Drop it, or "
                    "select heating_anomalous_transport='tail_walk'"
                )
            if bool(self._flags.get("coverage_closure")):
                raise ValueError(
                    "heating_anomalous_transport='plateau_multigroup' does "
                    "not support coverage_closure: the two-stream march "
                    "shares ONE withholding bank between the channel and "
                    "reservoir arms and the reservoir carries the density "
                    "FLOOR against the mean-field Te, so the plateau edge "
                    "solved there would be an artifact of the floor "
                    "convention rather than a measurement of the plasma. The "
                    "coverage arms are deferred until that stance is designed"
                )
        # pd1 branched disposal. It fills and consumes the SAME withholding
        # bank the tail walk does -- it only scales it per cell between the
        # march and the walk -- so every requirement the walk states applies to
        # it verbatim, and the two selectors are checked together below rather
        # than in parallel blocks that could drift apart.
        _disposal = str(
            self._input_dict.get("heating_anomalous_disposal")
        )
        if _disposal not in ("local", "landau_branched"):
            raise ValueError(
                "heating_anomalous_disposal must be 'local' or "
                f"'landau_branched' (got {_disposal!r})"
            )
        _branch = _disposal == "landau_branched"
        if _branch and _hat != "local":
            raise ValueError(
                "heating_anomalous_disposal='landau_branched' cannot be "
                f"combined with heating_anomalous_transport={_hat!r}: the "
                "branch already decides what share of each cell's extracted "
                "power is walked, and 'tail_walk' is its f_Landau == 1 "
                "corner, so naming both states two dispositions for one bank. "
                "Select the branch with heating_anomalous_transport='local'"
            )
        if _branch and bool(self._flags.get("coverage_closure")):
            raise ValueError(
                "heating_anomalous_disposal='landau_branched' does not "
                "support coverage_closure: the two-stream march shares ONE "
                "withholding bank between the channel and reservoir arms, so "
                "the reservoir's extracted power cannot be branched on the "
                "reservoir's own state -- and the reservoir carries "
                "ne = the density FLOOR against the mean-field Te, so any "
                "branching there would be an artifact of the floor convention "
                "rather than a measurement of the plasma. The coverage arms "
                "are deferred until that stance is designed"
            )
        _tail_walking = _hat in ("tail_walk", "plateau_multigroup") or _branch
        if _tail_walking:
            _sel = (
                "heating_anomalous_disposal='landau_branched'" if _branch
                else f"heating_anomalous_transport={_hat!r}"
            )
            if str(
                self._input_dict.get("beam_deposition_model")
            ) != "csda":
                raise ValueError(
                    f"{_sel} requires "
                    "beam_deposition_model='csda' (the anomalous heating it "
                    "transports is the CSDA ray's; under beer_lambert it "
                    "would be a silent no-op)"
                )
            if str(
                self._input_dict.get("beam_anomalous_model")
            ) == "none":
                raise ValueError(
                    f"{_sel} requires an "
                    "active anomalous channel "
                    "(beam_anomalous_model='quasilinear'); with no anomalous "
                    "drag there is no power to carry and the setting would "
                    "be a silent no-op"
                )
            _tail_eV = float(
                self._input_dict.get("heating_anomalous_tail_energy_eV")
            )
            if not math.isfinite(_tail_eV) or _tail_eV <= 0.0:
                raise ValueError(
                    "heating_anomalous_tail_energy_eV must be finite and > 0 "
                    f"(got {self._input_dict.get('heating_anomalous_tail_energy_eV')})"
                )
        # K7 sheath-aware tail closure: birth-energy keying and the cathode
        # boundary. The two string domains are checked unconditionally (a typo
        # is a typo whether or not the walk is engaged); everything that
        # depends on the walk being engaged is checked inside the tail_walk
        # branch below, so both keys keep the "inert under 'local'" contract
        # their siblings have.
        _keying = str(
            self._input_dict.get(
                "heating_anomalous_tail_energy_keying"
            )
        )
        if _keying not in ("phi_c", "fixed"):
            raise ValueError(
                "heating_anomalous_tail_energy_keying must be 'phi_c' or "
                f"'fixed' (got {_keying!r})"
            )
        _cath_bnd = str(
            self._input_dict.get(
                "heating_anomalous_tail_cathode_boundary"
            )
        )
        if _cath_bnd not in ("reflect", "escape"):
            raise ValueError(
                "heating_anomalous_tail_cathode_boundary must be 'reflect' or "
                f"'escape' (got {_cath_bnd!r})"
            )
        # f is a DECLARED BRACKET, never a fitted number, so a value off the
        # bracket is refused everywhere rather than only where it is read.
        _phi_frac = self._input_dict.get(
            "heating_anomalous_tail_phi_c_fraction"
        )
        if _phi_frac is not None and float(_phi_frac) not in (0.25, 0.5, 1.0):
            raise ValueError(
                "heating_anomalous_tail_phi_c_fraction must be one of the "
                "declared bracket arms 0.25, 0.5 or 1.0 (got "
                f"{_phi_frac!r}); it is a bracket the campaign reports across, "
                "not a value to fit"
            )
        if _branch:
            # The registered branched closure keys the birth energy to the LIVE
            # cathode drop. The fixed rung is an ASSUMED constant (75 eV) that
            # this closure's zero-new-constants statement does not cover, so it
            # is refused here rather than silently admitted.
            if _keying != "phi_c":
                raise ValueError(
                    "heating_anomalous_disposal='landau_branched' requires "
                    "heating_anomalous_tail_energy_keying='phi_c' (got "
                    f"{_keying!r}): the branched closure's birth energy is the "
                    "live cathode drop e*phi_c(t), and the fixed rung is an "
                    "assumed constant it does not carry"
                )
            # f is a DECLARED BRACKET. Under the branch there is no shipped
            # arm to fall back on -- the registered central arm is f = 1.0 and
            # the default None would silently select 0.25 -- so the arm must be
            # stated rather than defaulted.
            if _phi_frac is None:
                raise ValueError(
                    "heating_anomalous_disposal='landau_branched' requires "
                    "heating_anomalous_tail_phi_c_fraction to be stated "
                    "explicitly (one of the declared bracket arms 0.25, 0.5 "
                    "or 1.0); leaving it None would silently select 0.25 "
                    "while the registered central arm is 1.0"
                )
        if _tail_walking:
            if _keying == "fixed":
                if _phi_frac is not None:
                    raise ValueError(
                        "heating_anomalous_tail_phi_c_fraction was supplied "
                        "with heating_anomalous_tail_energy_keying='fixed', "
                        "where the tail energy is the constant "
                        "heating_anomalous_tail_energy_eV and the fraction "
                        "would do nothing; select keying='phi_c' or drop the "
                        "fraction"
                    )
            elif _tail_eV != 75.0:
                # The rung key is inert under phi_c keying, and the ONE way to
                # get that wrong is to select a bracket rung and have it
                # quietly ignored.
                raise ValueError(
                    "heating_anomalous_tail_energy_eV="
                    f"{_tail_eV} was supplied with "
                    "heating_anomalous_tail_energy_keying='phi_c', where the "
                    "tail energy is f*e*phi_c(t) and the fixed rung is inert; "
                    "select keying='fixed' to use the rung, or set the arm "
                    "through heating_anomalous_tail_phi_c_fraction"
                )
            if _cath_bnd == "reflect" and bool(
                self._flags.get("TwinCathode")
            ):
                raise ValueError(
                    "heating_anomalous_tail_cathode_boundary='reflect' does "
                    "not support TwinCathode: both faces of the walk window "
                    "would be reflecting cathodes, trapping the tail walkers "
                    "between them, and the walk has no termination convention "
                    "for that. Select 'escape' for twin configurations"
                )
            if _cath_bnd == "reflect":
                # The reflecting face has to be the face the cathode actually
                # occupies; a geometry where it is not is refused here rather
                # than reflecting walkers off an arbitrary window edge on the
                # first cathode solve.
                tail_reflect_face(self._geometry, end=0)
        # K6 tail ionization. Same discipline again, and the same reason to
        # duplicate the module's own guard here: a misconfiguration must fail
        # before the first cathode solve, not hours into a run.
        # Since K7b the two depth-1 bars no longer refuse -- they select a
        # treatment per ray (revert below E_stop, march with the measured
        # <= 2.0% understatement above the <W_sec> crossing), so there is
        # nothing for construction to reject there and the exposure is
        # reported in the tail diagnostics instead. The ONE surviving refusal
        # is the EII table edge, and it is the module's own constant evaluated
        # on this solver's I_ion rather than restated, so the two cannot
        # drift apart. Under K7 phi_c keying the LIVE E_tail is f*phi_c(t),
        # which no construction-time check can see; the check below then binds
        # only the (inert) fixed rung and the module's own copy of it,
        # evaluated on the live value at every solve, is what actually holds
        # the walk inside the tabulated cross section. That runtime copy is
        # REACHED: f = 1.0 with phi_c at cathode_phi_c_cap_V puts E_tail on
        # the edge to the last bit. Hence K7c -- the edge is inclusive within
        # HE_EII_EDGE_REL_TOL (the module owns both the constant and the
        # comparison; here it is imported, not restated).
        _tion = str(
            self._input_dict.get("heating_anomalous_tail_ionization")
        )
        if _tion not in ("off", "on"):
            raise ValueError(
                "heating_anomalous_tail_ionization must be 'off' or 'on' "
                f"(got {_tion!r})"
            )
        if _tion == "on":
            if not _tail_walking:
                raise ValueError(
                    "heating_anomalous_tail_ionization='on' requires walkers "
                    "to give the channel to: "
                    "heating_anomalous_transport='tail_walk', "
                    "heating_anomalous_transport='plateau_multigroup' or "
                    "heating_anomalous_disposal='landau_branched' (without "
                    "them the setting would be a silent no-op). "
                    "heating_anomalous_transport accepts 'local', "
                    "'tail_walk' or 'plateau_multigroup'; "
                    "heating_anomalous_disposal accepts 'local' "
                    "or 'landau_branched'; heating_anomalous_tail_ionization "
                    f"accepts 'off' or 'on' (got {_hat!r}, {_disposal!r} and "
                    f"{_tion!r})"
                )
            _E_table_top = HE_EII_EPS_TOP * float(self._I_ion)
            _edge_excess = (_tail_eV - _E_table_top) / _E_table_top
            if _edge_excess > HE_EII_EDGE_REL_TOL:
                raise ValueError(
                    "heating_anomalous_tail_ionization='on' marches the "
                    "walkers on the tabulated He EII cross section, which "
                    f"ends at {_E_table_top:.2f} eV; at "
                    f"heating_anomalous_tail_energy_eV={_tail_eV} eV the "
                    "lookup would clamp to its last node and the walk would "
                    "attenuate on an extrapolated cross section. This is "
                    "refused, not approximated (relative excess "
                    f"{_edge_excess:.3e}, tolerated "
                    f"{HE_EII_EDGE_REL_TOL:.1e})"
                )
        # --- A2a: the anode-mesh cull of the QL tail, and its rider --------
        # Duplicated from the deposition module's own guards for the standing
        # reason: a misconfiguration must fail at construction, not on the
        # first cathode solve. The module keeps its copies -- it is called
        # directly by instruments that never build a solver.
        _tail_cull = bool(
            self._flags.get("beam_tail_anode_interception")
        )
        _R_e = float(
            self._input_dict.get("beam_tail_anode_reflected_particles")
        )
        _eta_E = float(
            self._input_dict.get("beam_tail_anode_reflected_energy")
        )
        for _name, _val in (
            ("beam_tail_anode_reflected_particles", _R_e),
            ("beam_tail_anode_reflected_energy", _eta_E),
        ):
            if not math.isfinite(_val) or not 0.0 <= _val <= 1.0:
                raise ValueError(f"{_name} must be in [0, 1] (got {_val})")
        if _eta_E > _R_e:
            raise ValueError(
                "beam_tail_anode_reflected_energy must not exceed "
                f"beam_tail_anode_reflected_particles (got {_eta_E} > {_R_e}): "
                "both are PER INCIDENT walker, so their ratio is the mean "
                "energy a returned walker carries in units of the energy it "
                "arrived with, and that cannot exceed one"
            )
        if (_R_e > 0.0 or _eta_E > 0.0) and not _tail_cull:
            raise ValueError(
                "beam_tail_anode_reflected_particles/"
                "beam_tail_anode_reflected_energy ride on the anode tail cull "
                "and are read only under beam_tail_anode_interception=True; "
                "with the cull off nothing is intercepted and the pair would "
                "be a silent no-op"
            )
        if _R_e > 0.0 and _tion != "on":
            raise ValueError(
                "the reversed-walker rider "
                "(beam_tail_anode_reflected_particles > 0) requires "
                "heating_anomalous_tail_ionization='on': the energy-only "
                "closed-form tail walk carries a whole cell sequence in one "
                "telescoping integral and has no per-walker launch to "
                "reverse. The cull itself composes with either walk; only the "
                "return does not, and it is refused rather than approximated"
            )
        if _tail_cull:
            if not _tail_walking:
                raise ValueError(
                    "beam_tail_anode_interception culls the QL TAIL walkers, "
                    "so it needs a walked tail: "
                    "heating_anomalous_transport='tail_walk' or "
                    "'plateau_multigroup', or "
                    "heating_anomalous_disposal='landau_branched'. With none "
                    "of them there are no walkers to cull and the flag would "
                    "do nothing"
                )
            if not bool(self._flags.get("beam_anode_interception")):
                raise ValueError(
                    "beam_tail_anode_interception requires "
                    "beam_anode_interception: the tail cull fires at the same "
                    "anode face, on the same mesh solid fraction eta, and "
                    "books to the same anode_intercepted row as the primary's "
                    "interception. Arming one without the other would leave "
                    "two views of one mesh disagreeing about whether it is "
                    "there"
                )
        _fc = float(self._input_dict.get("beam_clump_fraction"))
        if not 0.0 <= _fc < 1.0:
            raise ValueError(
                f"beam_clump_fraction must be in [0, 1) (got {_fc})"
            )
        if float(self._input_dict.get("beam_clump_enhancement")) < 1.0:
            raise ValueError(
                "beam_clump_enhancement must be >= 1 (got "
                f"{self._input_dict.get('beam_clump_enhancement')})"
            )
        _fli = float(self._input_dict.get("gas_puff_local_ionization_fraction"))
        if not 0.0 <= _fli < 1.0:
            raise ValueError(
                f"gas_puff_local_ionization_fraction must be in [0, 1) (got {_fli})"
            )
        if _fli > 0.0 and self._flags.get("neutral_two_zone"):
            raise ValueError(
                "gas_puff_local_ionization_fraction is not supported with "
                "neutral_two_zone (annulus puff routing); disable one"
            )
        _fgp = float(self._input_dict.get("gas_puff_delivery_fraction"))
        if not np.isfinite(_fgp) or not 0.0 < _fgp <= 1.0:
            raise ValueError(
                "gas_puff_delivery_fraction must be finite and in (0, 1] "
                f"(got {_fgp}). It is the dimensionless share of the "
                "at-the-valve flow S_gp that is delivered into the modelled "
                "volume: 1.0 delivers all of it, 0 would delete the fueling "
                "entirely, and >1 would inject more gas than the valve "
                "supplies"
            )

    def _init_area_closures(self):
        """Validate and arm the three area/coverage closures."""
        # Clumpy-plasma coverage closure v1 (default off, bit-exact off).
        _cov = resolve_coverage_config(
            self._input_dict,
            self._flags,
            geometry=self._geometry,
            neutral_model=self._neutral_model,
        )
        self._coverage = _cov.coverage
        self._coverage_r = _cov.r
        self._coverage_tau_s = _cov.tau_s
        self._coverage_f = _cov.f
        self._coverage_deficit = _cov.deficit
        self._coverage_burn_accum = _cov.burn_accum
        self._coverage_burn_weight = _cov.burn_weight
        self._coverage_w_accum = _cov.w_accum
        self._coverage_reservoir_debit = _cov.reservoir_debit
        self._coverage_reservoir_burn_accum = _cov.reservoir_burn_accum
        # Cathode emitting-area percolation (default off, bit-exact off).
        self._cathode_f_em = resolve_emitting_area_config(
            self._input_dict, self._flags
        )
        # Ad-hoc probe neutral source (default off, bit-exact off).
        self._probe = resolve_neutral_probe_config(
            self._input_dict,
            self._flags,
            geometry=self._geometry,
            neutral_model=self._neutral_model,
            neutral_two_zone=self._neutral_two_zone,
        )
        # Electron drift transport and EMF work (default off, bit-exact off).
        # ``None`` when unarmed IS the presence gate: the RHS reads it to
        # decide whether to evaluate Gamma_d at all, so the off path never
        # enters the operator.
        self._electron_drift = resolve_electron_drift_transport_config(
            self._input_dict,
            self._flags,
            geometry=self._geometry,
            active_plasma_topology=self._active_plasma_topology,
        )
        # The operator's named power rows from the LAST RHS evaluation, for the
        # saved diagnostics. Same discipline as the hot-channel rows: written
        # by the evaluation, read by the snapshot that triggered it, never
        # recomputed from the saved sample.
        self._electron_drift_rows = None

    def _init_run_machinery(self):
        """Initialize the run-loop bookkeeping and apply any restart payload."""
        self._cathode_solve = None
        # Single-entry memo for READ-ONLY (``update_cache=False``) cathode
        # solves: ``(key, result)`` or None. See ``solve_cathode_boundary``.
        self._cathode_solve_memo = None
        # Item-35 ledger tripwire: latched so the warning fires once per run.
        self._beam_gap_ledger_warned = False
        self._last_result = None
        self._last_neutral_equilibration_result = None
        self._last_neutral_equilibration_summary = None
        # Set only while start_simulation drives run(); lets run() tell a direct
        # call from the equilibration-aware entry point (see run()).
        self._run_via_start_simulation = False
        # Run-loop controller state handed to the NEXT run() call by a restart,
        # and consumed there. None on every non-restart run, which is the
        # presence gate for the whole resume branch in run().
        self._restart_run_loop = None
        # Deposited by run() when it returns; read by restart_payload().
        self._last_run_loop_state = None
        # Continuation of a previous run. Last in construction: the payload
        # overwrites the initial condition that everything above just built,
        # so it must run after all of it and after every validator.
        self._load_restart_if_configured()
        if self._flags.get("debug_checks"):
            assert_finite_state(self._state, self._derived)

    def _advance_emitting_area_fraction(self, dt):
        """Advance the lit-area fraction ``f_em`` over one accepted step (ea1).

        The law is the logistic ``df_em/dt = r*f_em*(1 - f_em)`` with ``r`` the
        shared percolation clock. Held constant over the step it is exactly
        integrable,

            f' = 1 / (1 + (1/f - 1) * exp(-r * dt)),

        which is unconditionally positive at any dt, cannot leave ``(0, 1]``,
        is monotone non-decreasing for ``r >= 0`` -- so ``f_em`` never falls
        below its seed -- and reduces to ``f`` identically wherever ``r`` is
        zero and to exactly ``1.0`` wherever ``f`` is already 1 (the
        ``1/f - 1`` factor is then exactly 0.0).

        Called only from the accept path, so a rejected attempt leaves the
        fraction untouched and a re-tried step re-runs against the same value.
        """
        r = float(self._input_dict.get("coverage_growth_rate_per_s"))
        dt = float(dt)
        if r == 0.0 or dt <= 0.0:
            return
        f = self._cathode_f_em
        growth = math.exp(-r * dt)
        self._cathode_f_em = min(
            1.0 / (1.0 + (1.0 / f - 1.0) * growth), 1.0
        )

    #: RHS terms whose neutral row is a COVERED-ONLY debit or return: their
    #: rate is proportional to a plasma or beam density, so the reaction can
    #: only happen where the plasma is. Their summed ``nn`` rows drive the
    #: covered column's depletion. Terms that act uniformly across the
    #: cross-section (the gas puff, the pump, neutral transport and the
    #: zone/kinetic exchanges) and terms that transfer no particles (the
    #: ion-neutral collision operators) are deliberately absent.
    #:
    #: Two of the ABSENT terms do carry a neutral row that a plasma density
    #: sets, and are named here so their absence reads as a decision rather
    #: than an oversight:
    #:
    #: * ``characteristic_boundary`` -- the plasma leaving through the end
    #:   sheath returns as neutrals from the WALL, re-emitted diffusely over
    #:   the whole end face. Those neutrals are not channel-born and do not
    #:   land preferentially inside the patches the plasma left through, so
    #:   crediting them to the covered column would enrich it on a surface
    #:   process that has no azimuthal structure.
    #: * ``anode_collection`` -- the same statement at the anode mesh: ions
    #:   collected on a solid surface come back as diffuse surface re-emission
    #:   across the cross-section, not into the channel they arrived in.
    COVERAGE_BURN_TERMS = (
        "ionization_birth",
        "beam_ionization_birth",
        "recombination_rad_loss",
        "recombination_3b_loss",
        "gas_puff_local_ionization",
    )

    def coverage_fraction_profile(self):
        """Return the per-cell covered fraction ``f_cov(z)``, in ``(0, 1]``.

        All ones whenever the closure is off, so a caller needs no branch.

        This is accepted-step STATE, not a function of time: v2's growth law
        ``df_cov(z)/dt = r0*w(z,t)*f_cov*(1-f_cov)`` is driven by the beam
        ionization that the coverage itself shapes, so the closed form v1 could
        evaluate at any stage time no longer exists (v1's own documentation
        said a feedback v2 would have to co-integrate). The field is advanced
        once per ACCEPTED step from a driver accumulated across the SSPRK2
        stages -- the discipline the neutral deficit already uses -- and is
        therefore frozen within a step's stages and unmoved by a rejected
        attempt. See :meth:`_advance_coverage_fraction`.

        The array is a copy, so a caller cannot reach into the solver's state.
        """
        if self._coverage is None:
            return np.ones(self._geometry.cells, dtype=float)
        return self._coverage_f.copy()

    def coverage_fraction(self):
        """Return the volume-weighted COLUMN MEAN of ``f_cov(z)``.

        ``1.0`` whenever the closure is off. This is the scalar summary of the
        z-resolved field -- the single number v1 carried -- and is what the
        saved ``coverage_fraction`` diagnostic reports. Nothing in the physics
        reads it: every consumer takes the per-cell profile.
        """
        if self._coverage is None:
            return 1.0
        volume = np.asarray(self._geometry.plasma_volume_cm3, dtype=float)
        return float(np.sum(self._coverage_f * volume) / np.sum(volume))

    def _coverage_view(self, state, time=None):
        """Return the ``CoverageView1D`` the beam subsystem propagates in.

        ``None`` when the closure is off, which is what keeps every consumer
        on its historical argument list and the off path bit-exact. ``time`` is
        accepted and ignored: the coverage field is accepted-step state under
        v2, not a function of the stage clock (see
        :meth:`coverage_fraction_profile`).
        """
        if self._coverage is None:
            return None
        nn = np.asarray(state.nn, dtype=float)
        nn_channel = np.maximum(nn - self._coverage_deficit, self._floors["nn"])
        f_cov = self._coverage_f
        nn_reservoir = None
        ne_reservoir = None
        if np.any(f_cov < 1.0):
            # The other medium the beam is split across. Its neutral density
            # is the implicit reservoir's; its PLASMA density is the model's
            # own "no plasma" representation, the density floor, because the
            # closure's premise is that plasma lives in the covered fraction.
            # The floor rather than a literal zero because that is what every
            # other plasma-free cell in this solver carries, so the stopping
            # coefficients are evaluated on a state the model already
            # produces rather than on an untested singular one.
            #
            # Cells at f_cov == 1 have no uncovered region and the expression
            # is 0/0 there, so they carry the mean itself -- the value the
            # partition identity gives in the limit. Their reservoir arm is
            # launched with zero area and zero flux and never reads it.
            nn_reservoir = self._coverage_reservoir_from(nn, nn_channel, f_cov)
            ne_reservoir = np.full_like(nn, self._floors["n"])
        return CoverageView1D(
            f_cov=f_cov.copy(),
            nn_channel=nn_channel,
            nn_reservoir=nn_reservoir,
            ne_reservoir=ne_reservoir,
        )

    @staticmethod
    def _coverage_reservoir_from(nn, nn_channel, f_cov):
        """Return ``nn_r`` from the mean, the covered column and ``f_cov(z)``.

        From ``nn = f*nn_c + (1-f)*nn_r`` this is ``nn + f*D/(1-f)`` with
        ``D = nn - nn_c``. Cells at ``f == 1`` have no uncovered region at all
        and take the mean, which is the limit of the partition identity there.
        """
        uncovered = 1.0 - f_cov
        return np.where(
            uncovered > 0.0,
            nn + f_cov * (nn - nn_channel) / np.where(
                uncovered > 0.0, uncovered, 1.0
            ),
            nn,
        )

    def coverage_reservoir_density(self, state=None):
        """Return the uncovered reservoir's neutral density [cm^-3], per cell.

        Diagnostic only: the reservoir is represented IMPLICITLY, as the
        complement of the covered column inside the conserved mean field, so
        nothing integrates this. From ``nn = f*nn_c + (1-f)*nn_r`` it is
        ``nn + f*D/(1-f)`` with ``D = nn - nn_c`` the carried deficit. Where
        ``f_cov = 1`` there is no uncovered region and the mean itself is
        returned for that cell.
        """
        state = self.state if state is None else state
        nn = np.asarray(state.nn, dtype=float)
        if self._coverage is None:
            return nn.copy()
        nn_channel = np.maximum(nn - self._coverage_deficit, self._floors["nn"])
        return self._coverage_reservoir_from(nn, nn_channel, self._coverage_f)

    def coverage_growth_driver(self, terms):
        """Return ``w(z)``, the normalized beam-ionization growth driver.

        ``w`` is the LOCAL beam ionization rate divided by its own
        VOLUME-WEIGHTED COLUMN MEAN,

            w_i = S_i / (sum_j S_j V_j / sum_j V_j),

        with ``S`` the per-cell beam ion birth density [cm^-3 s^-1] -- the
        ``n`` row of ``beam_ionization_birth``, already carrying the
        active-plasma mask -- and ``V`` the plasma cell volumes. Both sums run
        over the PLASMA-ACTIVE cells: those are the cells whose plasma rows the
        solver integrates, the only ones where ``S`` can be nonzero and the
        only ones where a covered fraction means anything. Including the
        plasma-dead plenum volume in the denominator would dilute ``w``
        everywhere by a geometry ratio that has nothing to do with the beam.

        So normalized, ``<w>_V = 1`` identically over that window, which is the
        whole point: the rescaling is parameter-free and leaves
        ``coverage_growth_rate_per_s`` meaning exactly what it meant in v1, the
        column-mean growth rate. It redistributes growth in z without changing
        how much growth there is on average.

        DEGENERATE CASE, handled explicitly rather than by a guard on the
        divisor: when the beam deposits no ionization anywhere -- no cathode
        solve this evaluation, no emission, or a ray that stops in the gap --
        the mean is zero, ``w`` is 0/0, and the answer is ``w == 0`` in every
        cell, i.e. no growth. That is the physical statement (nothing is
        breaking the column down, so no patch spreads), not a numerical
        fallback. A non-finite mean is treated the same way.
        """
        term = terms.get("beam_ionization_birth")
        cells = self._geometry.cells
        if term is None:
            return np.zeros(cells, dtype=float)
        rate = np.asarray(term.n, dtype=float)
        volume = np.asarray(self._geometry.plasma_volume_cm3, dtype=float)
        window = (
            np.asarray(self._geometry.plasma_active, dtype=bool)
            if self._active_plasma_topology
            else np.ones(cells, dtype=bool)
        )
        weight = np.where(window, volume, 0.0)
        total_volume = float(np.sum(weight))
        mean = (
            float(np.sum(rate * weight)) / total_volume
            if total_volume > 0.0
            else 0.0
        )
        if not math.isfinite(mean) or mean <= 0.0:
            return np.zeros(cells, dtype=float)
        return np.where(window, rate / mean, 0.0)

    def _advance_coverage_fraction(self, dt, w_accum):
        """Advance ``f_cov(z)`` over one accepted step (coverage v2).

        The law is the logistic ``df/dt = r0*w*f*(1-f)`` with ``w`` the
        stage-accumulated growth driver: ``w_accum`` carries
        ``sum_stages (dt/2) * w_stage``, so ``w_bar = w_accum/dt`` is the
        step's stage-averaged driver on exactly the equal SSPRK2 stage weights
        the neutral-deficit burn uses. Holding it fixed over the step makes the
        logistic exactly integrable,

            f' = 1 / (1 + (1/f - 1) * exp(-r0 * w_bar * dt)),

        which is unconditionally positive, cannot leave ``(0, 1]`` at any dt,
        and reduces to ``f`` identically wherever ``w_bar`` or ``r0`` is zero
        and to exactly ``1.0`` wherever ``f`` is already 1 (the ``1/f - 1``
        factor is then exactly 0.0).

        Called only from the accept path, so a rejected attempt leaves the
        field untouched and a re-tried step re-derives its own driver.
        """
        if self._coverage_r == 0.0 or w_accum is None:
            return
        dt = float(dt)
        if dt <= 0.0:
            return
        w_bar = np.asarray(w_accum, dtype=float) / dt
        f = self._coverage_f
        growth = np.exp(-self._coverage_r * w_bar * dt)
        self._coverage_f = np.clip(
            1.0 / (1.0 + (1.0 / f - 1.0) * growth), None, 1.0
        )

    def _accumulate_coverage_burn(self, terms):
        """Tally this RHS stage's share of the step's covered-only neutral debit.

        Mirrors the DVM ionization booking: the weight is set once per attempt
        by :meth:`_attempt_step`, both explicit paths book the plasma rows
        through exactly one ``ssprk2_step`` at the full step dt, and SSPRK2
        weights its two stages equally at ``dt/2``. Nothing accumulates
        outside an attempt, and a rejected attempt's tally dies with it.

        The coverage field's own growth driver rides the SAME accumulator
        discipline and the same stage weight, which is what makes ``f_cov``
        co-integrated with the state rather than bolted on after it.
        """
        if self._coverage_burn_accum is None:
            return
        self._coverage_w_accum += (
            self._coverage_burn_weight * self.coverage_growth_driver(terms)
        )
        total = None
        for name in self.COVERAGE_BURN_TERMS:
            term = terms.get(name)
            if term is None:
                continue
            row = np.asarray(term.nn, dtype=float)
            total = row if total is None else total + row
        reservoir = self._coverage_reservoir_debit
        if reservoir is not None:
            # The beam's neutral row above is the SUM over both media. Only
            # the channel arm's share burnt covered gas, so the reservoir
            # arm's debit is subtracted out here and tallied separately: it
            # lowers the mean without lowering the covered column, which moves
            # the deficit the other way (see _advance_coverage_deficit).
            reservoir = np.asarray(reservoir, dtype=float)
            if total is not None:
                total = total - reservoir
            self._coverage_reservoir_burn_accum += (
                self._coverage_burn_weight * reservoir
            )
        if total is not None:
            self._coverage_burn_accum += self._coverage_burn_weight * total

    def _advance_coverage_deficit(self, dt, burn, reservoir_burn=None):
        """Advance the covered column's neutral deficit over one accepted step.

        The covered column absorbs the COVERED-ONLY neutral debit ``B_cov``
        but holds only the fraction ``f_cov`` of the cell's volume, so its
        local density falls ``1/f_cov`` times as fast as the mean's. The beam's
        reservoir arm (v1.1) debits ``B_res`` from the OTHER medium, lowering
        the mean while leaving the covered column alone, which moves the
        deficit the opposite way. With ``D = nn - nn_c``::

            dD/dt = -B_cov*(1 - f)/f + B_res - D/tau_backfill

        (both debits are negative on a burn, so the first term is positive --
        channels deplete -- and the second is negative). This is the whole
        azimuthal exchange: the reservoir/column relaxation ``f(1-f)(nn_r -
        nn_c)/tau`` reduces ALGEBRAICALLY to ``(nn - nn_c)/tau``, so no
        reservoir density is ever formed and the ``f -> 1`` limit is regular
        rather than a 0/0. The mean field is not touched at any point, so the
        total particle inventory is conserved IDENTICALLY -- this is a
        re-partition of what the mean already holds.

        Integrated with the exact integrating factor for a source held
        constant over the step, which is unconditionally positive and stable
        at any dt/tau.
        """
        dt = float(dt)
        # Per-cell under v2: the covered fraction varies along the column, so
        # every factor below that carried the scalar f now carries the local
        # one. The algebra is unchanged and elementwise.
        f = self._coverage_f
        # Debit is negative on a burn; the deficit grows when neutrals leave.
        # ``None`` is a step that evaluated no plasma RHS at all (the
        # neutral-only pre-drive path), i.e. zero debit -- the reservoir then
        # simply relaxes whatever deficit is outstanding.
        if burn is None:
            source = np.zeros(self._geometry.cells, dtype=float)
        else:
            source = -np.asarray(burn, dtype=float) / dt * (1.0 - f) / f
        if reservoir_burn is not None:
            # The reservoir arm's debit enters at weight ONE, not (1-f)/f: it
            # is removed from the mean and from the reservoir, never from the
            # covered column, so it closes the gap between them.
            source = source + np.asarray(reservoir_burn, dtype=float) / dt
        decay = math.exp(-dt / self._coverage_tau_s)
        deficit = self._coverage_deficit * decay + source * self._coverage_tau_s * (
            1.0 - decay
        )
        # The deficit is SIGNED. It is positive where the plasma burns column
        # gas faster than the reservoir refills it, and negative where the
        # covered region is a net neutral SOURCE -- a recombining cold column
        # returns neutrals into the covered fraction alone, enriching it above
        # the mean, which is the same physics with the sign reversed and is
        # not clipped away.
        #
        # The bounds are the two positivity conditions on the partition:
        # nn_c = nn - D >= 0 and nn_r = nn + f*D/(1-f) >= 0. Both are
        # re-partitions of the conserved mean, so hitting either creates and
        # destroys nothing; the lower bound closes onto 0 as f -> 1, where
        # there is no reservoir left to donate from.
        nn = np.maximum(np.asarray(self.state.nn, dtype=float), 0.0)
        floor = -(1.0 - f) / f * nn
        self._coverage_deficit = np.clip(deficit, floor, nn)

    # ------------------------------------------------------------------
    # Regime-R2 pre-breakdown passive-tracer bridge (default off).
    # Method of record: NUMERICS.md, "Regime-R2 pre-breakdown passive-tracer
    # bridge". Every method below returns immediately unless ``self._tracer``
    # is not None, which happens only under the ``regime_tracer`` flag: that is
    # the presence gate, and it is why the off path is bit-exact.
    # ------------------------------------------------------------------

    def _configure_regime_tracer(self):
        """Validate and arm the R2 passive tracer, or leave it absent.

        Every refusal names what the tracer ACCEPTS and fires at construction,
        never at the first step: a run that cannot legally use the tracer must
        not spend compute discovering that.
        """
        self._tracer = None
        self._tracer_passive = np.zeros(self._geometry.cells, dtype=bool)
        self._tracer_geometry_cache = None
        self._tracer_coefficients = None
        self._tracer_background = None
        self._tracer_depletion = None
        self._tracer_census = None
        self._tracer_refreshes = 0
        self._tracer_first_activation = None
        if not self._flags.get("regime_tracer"):
            return
        if not self._flags.get("Plasma"):
            raise ValueError(
                "regime_tracer describes the PLASMA's pre-breakdown build and "
                "has nothing to integrate with the Plasma flag off; accepted: "
                "Plasma on"
            )
        if not self._flags.get("cathode_coupling"):
            raise ValueError(
                "regime_tracer needs the cathode solve: the affine source S is "
                "the beam-impact ionization birth the cathode/beam solver "
                "produces, and without it the tracer has no source at all and "
                "a true-vacuum start would stay at vacuum forever. Accepted: "
                "cathode_coupling on"
            )
        if not self._active_plasma_topology:
            raise ValueError(
                "regime_tracer needs active_plasma_topology: the passive/"
                "active interface is a typed plasma topology (a closed face "
                "with one live cell), and the legacy all-cells flux path has "
                "no notion of the live cell at a closed face, so it cannot "
                "represent the interface. Accepted: active_plasma_topology on"
            )
        neutral_model = str(self._input_dict.get("neutral_model"))
        if neutral_model in RESTART_REFUSED_NEUTRAL_MODELS:
            raise ValueError(
                f"regime_tracer refuses neutral_model={neutral_model!r}: the "
                "tracer's growth rate is built from MOMENT neutral densities, "
                "and the handoff instrument (results/restart.py) does not "
                "serialise a distribution function either. R2 is fluid-arms "
                "only and does not extend DVM support; accepted: "
                f"{sorted(set(('moment',)) )} and any other moment closure"
            )
        anomalous_model = str(
            self._input_dict.get("beam_anomalous_model")
        )
        deposition_model = str(
            self._input_dict.get("beam_deposition_model")
        )
        if anomalous_model != "none" and deposition_model != "csda":
            raise ValueError(
                f"regime_tracer refuses beam_anomalous_model="
                f"{anomalous_model!r} together with beam_deposition_model="
                f"{deposition_model!r}: the anomalous channel exists only on "
                "the CSDA deposition rays, so under any other deposition model "
                "no quasilinear power is booked at all and the tracer's "
                "passive-cell refusal of it has nothing to refuse. A run "
                "configured this way reads as though the corrected booking is "
                "doing work when neither the channel nor its refusal is live. "
                "Accepted: beam_deposition_model='csda' with any "
                "beam_anomalous_model, or beam_anomalous_model='none' with any "
                "deposition model"
            )
        if self._input_dict.get("restart_from") is not None:
            raise ValueError(
                "regime_tracer cannot be combined with restart_from: the "
                "restart payload carries no tracer mask and no neutral-"
                "depletion accumulator, so a resumed tracer would silently "
                "restart criterion (c) from zero and under-report the burn it "
                "has already done. The intended two-stage shape is the "
                "opposite one -- stage 1 runs the conducting leg WITH the "
                "tracer and exports, stage 2 resumes with the flag off"
            )
        self._tracer = resolve_tracer_criteria(self._input_dict, self._floors)
        # Every cell that carries plasma at all starts passive: the tracer
        # exists precisely because the fluid cannot describe the leg yet.
        self._tracer_passive = np.asarray(
            self._geometry.plasma_active, dtype=bool
        ).copy()
        self._tracer_depletion = np.zeros(self._geometry.cells, dtype=float)
        self._tracer_census = self._empty_tracer_census()

    def _configure_regime_vessel_node(self):
        """Validate and arm the vessel common-mode node, or leave it absent.

        Every refusal names what the node ACCEPTS and fires at construction.
        With the flag off ``self._vessel`` and ``self._vessel_V_cm`` are both
        ``None``, which is what every consumer is presence-gated on, so the
        off path enters no branch this feature added.
        """
        self._vessel = None
        self._vessel_V_cm = None
        self._vessel_wall_currents_A = (0.0, 0.0, 0.0)
        self._vessel_charge_ledger_C = {
            "electron": 0.0,
            "ion": 0.0,
            "leak": 0.0,
            "node": 0.0,
            "abs": 0.0,
        }
        if not self._flags.get("regime_vessel_node"):
            return
        if not self._flags.get("cathode_coupling"):
            raise ValueError(
                "regime_vessel_node needs the cathode solve: the electron "
                "current landing on the wall IS the transmitted beam, and "
                "without a beam the node has nothing to charge it. Accepted: "
                "cathode_coupling on"
            )
        if not self._flags.get("Plasma"):
            raise ValueError(
                "regime_vessel_node needs the plasma: the ion wall flux that "
                "balances the beam leakage -- the bootstrap the node exists "
                "to describe -- is a column loss channel, and with the Plasma "
                "flag off the node would charge in one direction forever. "
                "Accepted: Plasma on"
            )
        if not self._flags.get("cathode_circuit_voltage_bound"):
            raise ValueError(
                "regime_vessel_node requires cathode_circuit_voltage_bound: "
                "the climb V_cm is subtracted from the beam's birth energy, "
                "and that energy must be the CIRCUIT-BOUNDED sheath drop. "
                "Without the bound it is the raw cathode_phi_c_cap_V atomic-"
                "data cap (~1000 V against a bank supplying ~178 V), so the "
                "choke would be a small correction on a wrong number. "
                "Accepted: cathode_circuit_voltage_bound on"
            )
        deposition_model = str(
            self._input_dict.get("beam_deposition_model")
        )
        if deposition_model != "csda":
            raise ValueError(
                f"regime_vessel_node refuses beam_deposition_model="
                f"{deposition_model!r}: the electron current the node books "
                "is the beam flux that reaches the TERMINATING SURFACE, and "
                "only the CSDA rays carry that flux "
                "(BeamDepositionResult.transmitted_flux). Under any other "
                "deposition model the wall electron channel would be "
                "identically zero and the node would charge on the ion flux "
                "alone -- a run that reads as though the bootstrap is live "
                "when half of it is missing. Accepted: "
                "beam_deposition_model='csda'"
            )
        self._vessel = resolve_vessel_node(self._input_dict, self._geometry)
        # Stage 1 is WALL-REFERENCED: at the seed currents the build opens
        # with, charging C_total to the bank scale takes far longer than the
        # cycle, so the float cannot engage and the node starts at zero. That
        # is an initial condition, not an assumption -- the ODE is free to
        # leave it whenever the currents make it.
        self._vessel_V_cm = 0.0

    def _vessel_ion_wall_current_A(self):
        """Return the column's ion current [A] onto the vessel, >= 0.

        Reads the LIVE plasma-terminating boundary term -- whichever of the
        characteristic ghost-cell outflow and the volumetric absorption the
        run configured -- on the accepted state, and integrates its ``n`` row
        over the collector cells' plasma volume. The loss channel is not
        re-derived here: this is the same term the fluid itself subtracts, so
        the node cannot book an ion flux the column did not lose.

        Zero when the step carries no cached cathode solve. That is a guard
        with teeth rather than a convenience: the boundary term's cathode-jet
        path re-solves the cathode with ``update_cache=True`` when it is handed
        no solve, so reading the term here without one would let a DIAGNOSTIC
        read mutate the continuation cache and move the trajectory. There is no
        driven plasma to lose ions in that window anyway.
        """
        node = self._vessel
        state = self.state
        cathode_solve = self._cathode_solve
        if cathode_solve is None:
            return 0.0
        term = self.characteristic_boundary_rhs(
            state=state, cathode_solve=cathode_solve, time=self._time
        )
        row = np.asarray(term.n, dtype=float)
        Vp = np.asarray(self._geometry.plasma_volume_cm3, dtype=float)
        cells = node.collector_cells
        # The row is a SINK there (negative); a positive current onto the wall
        # is its negation. Clamped at zero so a cell that is momentarily a net
        # source cannot book a backwards wall current.
        return qe_SI * max(-float(np.sum(row[cells] * Vp[cells])), 0.0)

    def _vessel_electron_wall_current_A(self):
        """Return the beam electron current [A] landing on the vessel, >= 0.

        The CSDA rays' transmitted PRIMARY flux, summed over cathode ends. The
        far end is the vessel, so the flux that leaves the domain there is
        exactly the electron current the wall conductor collects; the flux the
        anode mesh intercepts and the flux that stops in the column are
        system-side and plasma-side respectively and are not booked here.

        Under ``beam_product_transport="terminal_nonlocal"`` the walked
        terminal residual that reaches an end is a SECOND population landing on
        that same surface, and its flux is added on the identical convention
        (summed over rays, whichever end each ray was heading for). Its energy
        is not double-counted: the walk already booked that to the deposition
        module's end ledger, which is an energy ledger and never a charge one.
        With the selector at any other value the term is absent, not zero-by-
        arithmetic -- ``_beam_terminal_wall_charge`` is resolved at
        construction.
        """
        solve = self._cathode_solve
        deposition = None if solve is None else getattr(
            solve, "beam_deposition", None
        )
        if not deposition:
            return 0.0
        flux = 0.0
        for dep in deposition.values():
            if dep is not None:
                flux += float(dep.transmitted_flux)
                if self._beam_terminal_wall_charge:
                    flux += float(dep.terminal_escape_flux_per_s)
        return qe_SI * max(flux, 0.0)

    def _vessel_advance(self, dt):
        """Advance ``V_cm`` and its charge ledger over one ACCEPTED step.

        Accepted steps only, exactly like the loop current: a rejected attempt
        must not move the node. The two wall currents are read at the accepted
        state and frozen across the step, so the choke they produce reaches
        the beam at the next solve -- the same explicit coupling the circuit
        and the cathode thermal state already use.
        """
        node = self._vessel
        if node is None:
            return
        I_e = self._vessel_electron_wall_current_A()
        I_i = self._vessel_ion_wall_current_A()
        V_new, dV, dQ_e, dQ_i, dQ_leak = vessel_node_advance(
            node, self._vessel_V_cm, I_e, I_i, dt
        )
        self._vessel_V_cm = V_new
        ledger = self._vessel_charge_ledger_C
        ledger["electron"] += dQ_e
        ledger["ion"] += dQ_i
        ledger["leak"] += dQ_leak
        ledger["node"] += node.C_total_F * dV
        # Scale for the ledger's closure test: the total charge MOVED, so a
        # residual is judged against the traffic and not against a cancelling
        # net that can pass through zero.
        ledger["abs"] += abs(dQ_e) + abs(dQ_i) + abs(dQ_leak)
        self._vessel_wall_currents_A = (
            I_e,
            I_i,
            0.0 if node.R_leak_ohm is None else V_new / node.R_leak_ohm,
        )

    def vessel_charge_residual(self):
        """Return the node's charge-ledger residual ``(absolute, relative)``.

        ``C_total * sum(dV)`` against ``Q_electron - Q_ion - Q_leak``: the
        auditable form of the node's conservation. Zero traffic returns a zero
        pair rather than dividing by it.
        """
        ledger = self._vessel_charge_ledger_C
        residual = ledger["node"] - (
            ledger["electron"] - ledger["ion"] - ledger["leak"]
        )
        scale = ledger["abs"]
        return residual, (0.0 if scale <= 0.0 else abs(residual) / scale)

    def _tracer_exempt_initial_floor(self, raw, floored):
        """Restore the RAW plasma rows on tracer cells in the initial condition.

        Same exemption :meth:`floor_state_vector` applies every step, applied
        once to the initial condition so ``ne0 = 0`` is a true-vacuum start
        rather than a silent ``ne_floor`` fill. The NEUTRAL rows keep their
        floor: the tracer owns the plasma, not the background.
        """
        passive = self._tracer_passive
        return ConservativeState1D(
            n=np.where(passive, raw.n, floored.n),
            nn=floored.nn,
            M=np.where(passive, raw.M, floored.M),
            Ee=np.where(passive, raw.Ee, floored.Ee),
            Ei=np.where(passive, raw.Ei, floored.Ei),
            M_n=floored.M_n,
            nn_a=floored.nn_a,
            M_n_a=floored.M_n_a,
            En=floored.En,
        )

    @property
    def _tracer_engaged(self):
        """True while any cell is still owned by the tracer."""
        return self._tracer is not None and bool(np.any(self._tracer_passive))

    def _plasma_active_mask(self):
        """Return the cells the FLUID owns: typed-active minus tracer-passive."""
        active = np.asarray(self._geometry.plasma_active, dtype=bool)
        if self._tracer is None:
            return active
        return active & ~self._tracer_passive

    def _plasma_geometry(self):
        """Return the geometry the PLASMA operators see this step.

        Identical object to ``self._geometry`` unless the tracer owns cells, so
        the flag-off path cannot even observe that this method exists. When it
        does own cells, the returned view closes every passive/active interface
        face by exactly the rule ``build_geometry`` uses for a typed
        plasma-dead boundary (``dead[f-1] != dead[f]``), and recomputes
        ``plasma_face_live_cell`` from the composed mask. A closed face carries
        no particle, advective-momentum or thermal-energy flux and no
        conduction, with the active cell's pressure acting on it -- which is
        the interface treatment NUMERICS.md defines, reached by reusing the
        operator that already implements it rather than by adding a branch to
        the flux.
        """
        if not self._tracer_engaged:
            return self._geometry
        key = self._tracer_passive.tobytes()
        cached = self._tracer_geometry_cache
        if cached is not None and cached[0] == key:
            return cached[1]
        base = self._geometry
        cells = int(base.cells)
        active = self._plasma_active_mask()
        dead = ~active
        plasma_open = np.asarray(base.plasma_open, dtype=bool).copy()
        plasma_transmission = np.asarray(
            base.plasma_transmission, dtype=float
        ).copy()
        heat_transmission = np.asarray(
            base.heat_transmission, dtype=float
        ).copy()
        for face in range(1, cells):
            if dead[face - 1] != dead[face]:
                plasma_open[face] = False
                plasma_transmission[face] = 0.0
                heat_transmission[face] = 0.0
        live = np.full(cells + 1, -1, dtype=int)
        for face in np.flatnonzero(~plasma_open):
            face = int(face)
            adjacent = []
            if face > 0 and active[face - 1]:
                adjacent.append(face - 1)
            if face < cells and active[face]:
                adjacent.append(face)
            if len(adjacent) > 1:
                raise ValueError(
                    f"closed plasma face {face} has active cells on both "
                    "sides under the tracer's passive mask"
                )
            if adjacent:
                live[face] = int(adjacent[0])
        view = replace(
            base,
            plasma_active=active,
            plasma_open=plasma_open,
            plasma_transmission=plasma_transmission,
            heat_transmission=heat_transmission,
            plasma_face_live_cell=live,
        )
        self._tracer_geometry_cache = (key, view)
        return view

    @staticmethod
    def _empty_tracer_census():
        return {
            "criterion": np.zeros(0, dtype=int),
            "worst_ratio": np.zeros(0, dtype=float),
            "ratios": {},
            "passive": np.zeros(0, dtype=bool),
            "Te_qs_eV": np.zeros(0, dtype=float),
            "gamma_per_s": np.zeros(0, dtype=float),
            "S_cm3_s": np.zeros(0, dtype=float),
            "refreshes": 0,
        }

    def _tracer_device_voltage_V(self, cathode_solve):
        """Return the R1-bounded device voltage [V] criterion (a) is driven by.

        The sheath solve's own ``V_b``, which under
        ``cathode_circuit_voltage_bound`` is held at or below
        ``min(cathode_phi_c_cap_V, V_avail(I))``. Reading the raw atomic-data
        cap instead would inflate the conducted current by exactly the factor
        the R1 pass removed from the pre-breakdown build leg. Zero when there
        is no solve, which makes criterion (a) inactive rather than undefined.

        The vessel node does NOT shift this value, and that is deliberate:
        ``V_cm`` moves the whole cathode/anode system against the wall, so it
        cannot change the anode-to-cathode differential the column conducts
        under. The node's offset enters the BEAM energy instead --
        :meth:`_tracer_beam_energy_eV` is that seam.
        """
        beam_result = getattr(cathode_solve, "beam_result", None)
        if beam_result is None:
            return 0.0
        voltages = [
            abs(float(getattr(result, "V_b", 0.0)))
            for result in (beam_result.result, beam_result.result_twin)
            if result is not None
        ]
        return max(voltages) if voltages else 0.0

    def _tracer_beam_energy_eV(self, cathode_solve):
        """Return the beam energy [eV] reaching the column, or 0 with no beam.

        Keyed to the BOUNDED sheath drop, never to ``cathode_phi_c_cap_V``:
        the cap is the He EII table top, an atomic-data domain guard, and using
        it as a beam energy is the defect R1 removed. With the vessel node
        armed the mesh-to-column climb is subtracted from that bounded drop by
        the same function the beam readers use, so criterion (b) measures the
        thinness of the plasma to the beam that actually arrives.
        """
        return beam_launch_energy_eV(
            self._tracer_device_voltage_V(cathode_solve),
            vessel_beam_climb_V(self._flags, self._vessel_V_cm),
        )

    def _tracer_launch_cells(self):
        """Return ``{end: launch cell index}`` for the beams criterion (b) sums."""
        launches = {0: int(beam_launch(self._geometry, end=0)[0])}
        if self._flags.get("TwinCathode"):
            launches[-1] = int(beam_launch(self._geometry, end=-1)[0])
        return launches

    def _tracer_reaction_kwargs(self):
        """Return the subset of the reaction kwargs ``reaction_rates`` accepts."""
        full = self._reaction_kwargs()
        return {
            name: full[name]
            for name in (
                "gas_type",
                "I_ion",
                "atomic_rate_model",
                "adas_low_te_extension",
            )
        }

    def _tracer_surface_kwargs(self):
        """Return the surface-absorption kwargs, jet and DVM channels absent.

        The jet moves only the ``M_n`` row and the tracer reads the ``n`` row;
        ``Tn_presheath_eV`` comes from the kinetic arms, which the tracer
        refuses at construction.
        """
        surface = self._surface_loss_kwargs()
        return {
            "alpha_isat": surface["alpha_isat"],
            "b_surface_loss": surface["b_surface_loss"],
            "b_presheath_length": float(
                self._input_dict.get("b_presheath_length")
            ),
            "gas_type": self._gas_type,
        }

    def _tracer_boundary_rhs(self, cathode_solve, time):
        """Return ``probe_state -> plasma-end loss term``.

        The tracer must consume the SAME boundary operator the fluid does, or
        ``gamma`` disagrees with the fluid's own ``n`` row -- which is exactly
        what the smoke's identity assertion catches. Since the legacy
        volumetric absorber was retired 2026-08-31 (Tom) there is one operator,
        so there is nothing left to select between.
        """
        surface_kwargs = self._tracer_surface_kwargs()

        def boundary(probe):
            return characteristic_boundary_rhs(
                state=probe,
                floors=self._floors,
                ion_mass_g=self._ion_mass_g,
                mu=self._mu,
                geometry=self._plasma_geometry(),
                cathode_jet=None,
                wave_speed=self._hyperbolic_wave_speed,
                energy_consistent=self._hyperbolic_energy_consistent,
                **surface_kwargs,
            )

        return boundary

    def _tracer_exchange_kwargs(self):
        return {}

    def _tracer_beam_kwargs(self, state, cathode_solve, time):
        """Return the argument set BOTH beam readers must be built from.

        One object, passed to ``beam_ionization_rhs_terms`` and to
        ``beam_anomalous_power_density`` alike, so the row and the anomalous
        share it is asked to give back cannot be built from different flags,
        a different coverage view or a different smoothing width.
        """
        return {
            "state": state,
            "floors": self._floors,
            "ion_mass_g": self._ion_mass_g,
            "geometry": self._geometry,
            "input_dict": self._input_dict,
            "input_flags": self._effective_cathode_flags(
                time=time, active_only=True
            ),
            "cathode_solve": cathode_solve,
        }

    def _tracer_beam_rows(self, state, cathode_solve, time):
        """Return ``(S, P_net, P_full)``: the beam rows the balance consumes.

        ``S`` is the ``n`` row of ``beam_ionization_birth`` -- the affine
        source, independent of ``n`` to the accuracy criterion (b) enforces.
        ``P_full`` is the sum of the three beam rows the fluid books on ``Ee``
        (deposition, ionization cost, excitation radiation), i.e. the net beam
        heating of the electron fluid.

        Under ``beam_anomalous_model="quasilinear"`` ONLY, ``P_net`` is
        ``P_full`` with the anomalous share removed on every cell the tracer
        owns, and it is ``P_net`` -- not ``P_full`` -- that the quasi-static
        balance absorbs. That closure books near-total absorption by fiat, and
        it is a beam-PLASMA instability: it needs a wave medium, and a passive
        cell by its own definition has no plasma to be that medium, so on such
        a cell the channel does not exist and booking its power there was
        describing an interaction with a plasma that is not there. The channels
        that survive on a passive cell are the ones that do not need one:
        collisional beam drag on plasma electrons (proportional to ``n``, and
        at vacuum-class density accordingly tiny) and the ionization-birth
        bookkeeping, both of which stay exactly as they were.

        The refusal is MODEL-KEYED, and that is the whole of the difference
        between the closure legs. ``"ql_relaxation"`` carries its own onset
        gate and its own density-dependent extracted fraction, so it already
        books what a low-density cell can actually absorb rather than a fiat
        total; refusing it wholesale on the passive set would delete the
        physics the closure exists to supply. A passive cell therefore BOOKS
        ``ql_relaxation``'s power in full, exactly as an active one does, and
        ``P_net == P_full`` everywhere. Under ``"none"`` there is no anomalous
        power to move either way.

        Where the subtraction does apply it is gated on the PASSIVE MASK and on
        nothing else. No density threshold is introduced: the tracer-to-fluid
        handoff and the onset of fiat quasilinear absorption are made the same
        event by construction, so a cell that has become active books the
        anomalous channel in full, unchanged. NUMERICS.md, "Corrected beam
        power booking on passive cells" and "The anomalous closure bracket",
        are the statements of record, and
        :meth:`tracer_passive_anomalous_leak` is the auditable invariant.
        """
        zeros = np.zeros(self._geometry.cells, dtype=float)
        if cathode_solve is None:
            return zeros, zeros.copy(), zeros.copy()
        beam_kwargs = self._tracer_beam_kwargs(state, cathode_solve, time)
        terms = beam_ionization_rhs_terms(
            I_ion=self._I_ion,
            coverage=self._coverage_view(state, time),
            **beam_kwargs,
        )
        S = np.asarray(terms["beam_ionization_birth"].n, dtype=float)
        P_full = (
            np.asarray(terms["beam_power_deposition"].Ee, dtype=float)
            + np.asarray(terms["beam_ionization_cost"].Ee, dtype=float)
            + np.asarray(terms["beam_excitation_radiation"].Ee, dtype=float)
        )
        if str(
            self._input_dict.get("beam_anomalous_model")
        ) != "quasilinear":
            return S, P_full.copy(), P_full
        P_ql = beam_anomalous_power_density(**beam_kwargs)
        P_net = P_full - np.where(self._tracer_passive, P_ql, 0.0)
        return S, P_net, P_full

    def tracer_passive_anomalous_leak(self, state=None, time=None):
        """Return each PASSIVE cell's departure from its closure's policy; 0.

        The audit form of the model-keyed booking in :meth:`_tracer_beam_rows`.
        It re-reads both the anomalous share and the model key from the
        deposition objects and the config through ``physics.tracer``'s own
        references, so it does not travel through the code path it is checking
        and a removed, neutered or mis-keyed refusal is still caught. Returns
        zeros when the tracer is not engaged: there are no passive cells to
        audit.
        """
        cells = int(self._geometry.cells)
        if not self._tracer_engaged:
            return np.zeros(cells, dtype=float)
        if state is None:
            state = self.state
        if time is None:
            time = self._time
        cathode_solve = self._cathode_solve
        if cathode_solve is None:
            cathode_solve = self.solve_cathode_boundary(
                state=state, time=time, update_cache=False
            )
        _S, P_net, P_full = self._tracer_beam_rows(state, cathode_solve, time)
        return tracer_passive_anomalous_leak(
            P_beam_net_consumed=P_net,
            P_beam_net_full=P_full,
            passive=self._tracer_passive,
            beam_kwargs=self._tracer_beam_kwargs(state, cathode_solve, time),
        )

    def _tracer_prepare(self, dt):
        """Return this attempt's frozen tracer coefficients, mutating nothing.

        Called from ``_attempt_step`` so the coefficients are the STEP-START
        ones (the Picard convention) and so a rejected attempt leaves no trace:
        everything here is returned on the attempt and committed only by
        ``_tracer_apply``.
        """
        if not self._tracer_engaged:
            return None
        state = self.state
        time = self._time
        cathode_solve = self._cathode_solve
        if cathode_solve is None and self._flags.get("cathode_coupling"):
            cathode_solve = self.solve_cathode_boundary(
                state=state, time=time, update_cache=False
            )
        S, P_net, _P_full = self._tracer_beam_rows(state, cathode_solve, time)
        n_true = np.maximum(np.asarray(state.n, dtype=float), 0.0)
        n_probe = np.maximum(n_true, float(self._floors["n"]))
        nn = np.asarray(state.nn, dtype=float)
        background = {"n": n_true, "nn": nn, "S": S}
        drift = tracer_relative_drift(self._tracer_background or {}, background)
        cached = self._tracer_coefficients
        refreshed = cached is None or drift > self._tracer["refresh_tol"]
        if refreshed:
            boundary_rhs = self._tracer_boundary_rhs(cathode_solve, time)
            Ti = np.full(self._geometry.cells, float(self._floors["Ti"]))
            Te, sign_changes = quasistatic_Te_eV(
                state=state,
                n_true=n_true,
                n_probe=n_probe,
                Ti_eV=Ti,
                S_beam=S,
                P_beam_net=P_net,
                floors=self._floors,
                ion_mass_g=self._ion_mass_g,
                mu=self._mu,
                cooling_kwargs=self._electron_cooling_kwargs(),
                exchange_kwargs=self._tracer_exchange_kwargs(),
                boundary_rhs=boundary_rhs,
                # THE PASSIVE SET, and only it. A cell the fluid owns has its
                # own electron energy equation, integrated with conduction and
                # the boundary terms in it; asking the local quasi-static
                # balance about that cell is asking a description that was
                # never valid there, and the answer -- or the refusal -- would
                # be about the wrong object. It is also where the anomalous
                # booking is legitimately restored, so the balance would see
                # the whole beam power and refuse for exactly the reason the
                # amendment above removes on passive cells. Whatever reads a
                # temperature on an active cell reads the FLUID's own Te (see
                # :meth:`_tracer_criteria_Te_eV`).
                active=self._tracer_passive & ((n_true > 0.0) | (S > 0.0)),
                Te_ceiling_eV=self._tracer_beam_energy_eV(cathode_solve),
            )
            gamma = tracer_growth_rate(
                state=state,
                n_true=n_true,
                n_probe=n_probe,
                Te_eV=Te,
                Ti_eV=Ti,
                floors=self._floors,
                ion_mass_g=self._ion_mass_g,
                reaction_kwargs=self._tracer_reaction_kwargs(),
                boundary_rhs=boundary_rhs,
            )
            coefficients = {
                "gamma": gamma,
                "Te": Te,
                "Ti": Ti,
                "sign_changes": sign_changes,
            }
        else:
            coefficients = cached
        return {
            "dt": float(dt),
            "refreshed": bool(refreshed),
            "background": background,
            "coefficients": coefficients,
            "S": S,
            "n_start": n_true,
            "V_dev_V": self._tracer_device_voltage_V(cathode_solve),
            "E_beam_eV": self._tracer_beam_energy_eV(cathode_solve),
        }

    def _tracer_criteria_Te_eV(self, Te_qs):
        """Return the temperature the criteria and census read, per cell.

        ``Te_qs`` on a cell the tracer owns; the FLUID's own ``Te`` on every
        other cell. The quasi-static balance is solved on the passive set only,
        so ``Te_qs`` off that set is the floor-by-convention filler and means
        nothing -- reading it would have made criterion (a)'s Spitzer
        conductivity, criterion (b)'s stopping power and the census all describe
        a cold cell wherever the fluid was in fact running hot, and it is the
        re-entry branch of the hysteresis that reads them there.

        Called after the state vector for the step is installed, so the fluid
        rows are this step's, not the previous one's.
        """
        passive = self._tracer_passive
        Te_fluid = derive_state(
            self.state, self._floors, self._ion_mass_g
        ).Te
        return np.where(
            passive,
            np.asarray(Te_qs, dtype=float),
            np.asarray(Te_fluid, dtype=float),
        )

    def _tracer_criteria_n_cm3(self, n_next):
        """Return the density the criteria read, per cell.

        The density analogue of :meth:`_tracer_criteria_Te_eV`, and the same
        principle: the criteria describe the STATE of a cell, and on a cell the
        fluid owns the state is the fluid's. ``n_next`` there is that cell's
        step-START density advanced by one step of the tracer's affine ODE --
        an extrapolation by a description that does not own the cell, and one
        that ignores everything the fluid actually did to it this step
        (advection across its open faces, the flux divergence, the floor).

        On a passive cell this is ``n_next`` by construction, which is also
        exactly what the installed state carries there: ``_tracer_apply`` wrote
        it and ``floor_state_vector`` exempts those cells, so no clip stands
        between the two.
        """
        return np.where(
            self._tracer_passive,
            np.asarray(n_next, dtype=float),
            np.asarray(self.state.n, dtype=float),
        )

    def _tracer_apply(self, prepared):
        """Commit an accepted step's tracer update, mask move and census."""
        if prepared is None:
            return
        dt = float(prepared["dt"])
        gamma = prepared["coefficients"]["gamma"]
        Te = prepared["coefficients"]["Te"]
        Ti = prepared["coefficients"]["Ti"]
        S = prepared["S"]
        passive = self._tracer_passive
        n_start = prepared["n_start"]
        n_next = tracer_affine_update(n_start, gamma, S, dt)
        n_integral = tracer_affine_time_integral(n_start, gamma, S, dt)

        state = self.state
        n = np.asarray(state.n, dtype=float).copy()
        Ee = np.asarray(state.Ee, dtype=float).copy()
        Ei = np.asarray(state.Ei, dtype=float).copy()
        M = np.asarray(state.M, dtype=float).copy()
        n[passive] = n_next[passive]
        Ee[passive] = 1.5 * n_next[passive] * Te[passive] * ev_to_erg
        Ei[passive] = 1.5 * n_next[passive] * Ti[passive] * ev_to_erg
        # A passive cell exchanges no momentum: its interface faces are closed
        # and its own ODE carries none.
        M[passive] = 0.0
        self._set_state_vector(
            pack_state(
                ConservativeState1D(
                    n=n, nn=state.nn, M=M, Ee=Ee, Ei=Ei,
                    M_n=state.M_n, nn_a=state.nn_a, M_n_a=state.M_n_a,
                    En=state.En,
                )
            )
        )

        # Criterion (c) accumulator: the neutrals the PLASMA's own bulk
        # ionization burnt, exactly integrated. The beam's debit is background
        # and deliberately absent -- (c) measures the plasma's back-reaction on
        # the neutrals, not the discharge's.
        gamma_ion = np.maximum(gamma, 0.0)
        self._tracer_depletion = self._tracer_depletion + np.where(
            passive, gamma_ion * n_integral, 0.0
        )
        if prepared["refreshed"]:
            self._tracer_coefficients = prepared["coefficients"]
            self._tracer_background = prepared["background"]
            self._tracer_refreshes += 1
        self._tracer_update_mask(
            prepared,
            self._tracer_criteria_n_cm3(n_next),
            self._tracer_criteria_Te_eV(Te),
            gamma,
        )

    def _tracer_update_mask(self, prepared, n_next, Te, gamma):
        """Move the passive/active boundary, with hysteresis, and census it.

        ``n_next`` and ``Te`` here are the COMPOSED state from
        :meth:`_tracer_criteria_n_cm3` and :meth:`_tracer_criteria_Te_eV` --
        the tracer's on the cells it owns, the FLUID's own everywhere else --
        not the raw affine update and not the raw balance output. The criteria
        judge a cell by the state of that cell, and which description that
        comes from is settled by who owns the cell.
        """
        state = self.state
        criteria = self._tracer["criteria"]
        hysteresis = self._tracer["hysteresis"]
        geometry = self._geometry
        nn = np.maximum(np.asarray(state.nn, dtype=float), 0.0)
        L_plasma_cm = float(
            np.sum(
                np.asarray(geometry.length_cm, dtype=float)[
                    np.asarray(geometry.plasma_active, dtype=bool)
                ]
            )
        )
        I_loop = abs(float(self._cathode_total_current_A()))
        I_cond = tracer_conducted_current_A(
            n_cm3=n_next,
            Te_eV=Te,
            geometry=geometry,
            V_dev_V=prepared["V_dev_V"],
            L_plasma_cm=L_plasma_cm,
        )
        ratios = {
            "current": (
                I_cond / (I_loop * criteria["current"])
                if I_loop > 0.0
                else np.zeros_like(I_cond)
            ),
            "thinness": tracer_beam_plasma_thinness(
                n_cm3=n_next,
                Te_eV=Te,
                geometry=geometry,
                E_beam_eV=prepared["E_beam_eV"],
                launch_cells=self._tracer_launch_cells(),
                coulomb_model=str(
                    self._input_dict.get("beam_coulomb_model")
                ),
            ) / criteria["thinness"],
            # An empty cell (nn == 0) has no neutrals left to burn, so its
            # depletion is total and the ratio is inf -- the cell activates.
            # NOTE the divisor is max(nn, 1): between 0 and 1 cm^-3 the ratio
            # is UNDERSTATED (divided by 1 instead of by nn), so criterion (c)
            # under-reports in a band it cannot physically reach -- 1 cm^-3 is
            # five decades below nn_floor and twelve below any real fill, and
            # the nn == 0 branch above already covers true vacuum. The clamp is
            # there so the divide cannot produce a subnormal or overflow on a
            # nonsense input, not to model anything.
            "depletion": np.where(
                nn > 0.0,
                self._tracer_depletion / np.maximum(nn, 1.0),
                np.inf,
            ) / criteria["depletion"],
        }
        worst, binding = tracer_bind_census(ratios)
        passive = self._tracer_passive
        # Enter/exit hysteresis: a cell leaves passivity above 1 and can only
        # return below 1/h. Monotone criteria never exercise the return branch;
        # it exists so a cell sitting on a threshold cannot chatter.
        activated = passive & (worst > 1.0) & (
            n_next >= self._tracer["activation_ne"]
        )
        returning = (~passive) & (worst < 1.0 / hysteresis) & np.asarray(
            geometry.plasma_active, dtype=bool
        )
        if np.any(activated) and self._tracer_first_activation is None:
            self._tracer_first_activation = (
                float(self._time),
                int(np.flatnonzero(activated)[0]),
                CRITERION_NAMES[int(binding[np.flatnonzero(activated)[0]])],
            )
        new_passive = (passive & ~activated) | returning
        if not np.array_equal(new_passive, passive):
            self._tracer_geometry_cache = None
        self._tracer_passive = new_passive
        self._tracer_census = {
            "criterion": np.asarray(binding, dtype=int),
            "worst_ratio": np.asarray(worst, dtype=float),
            "ratios": {
                name: np.asarray(value, dtype=float)
                for name, value in ratios.items()
            },
            "transport_ratio": tracer_transport_ratio(
                gamma=gamma,
                Te_eV=Te,
                mu=self._mu,
                L_n_cm=0.5 * L_plasma_cm,
            ),
            "passive": new_passive.copy(),
            # The temperature the criteria above actually read: quasi-static on
            # the tracer's cells, the fluid's own on every other. Keeping the
            # raw balance output here instead would publish the floor filler on
            # active cells and disagree with the ratios beside it.
            "Te_qs_eV": np.asarray(Te, dtype=float),
            "gamma_per_s": np.asarray(gamma, dtype=float),
            "S_cm3_s": np.asarray(prepared["S"], dtype=float),
            "refreshes": int(self._tracer_refreshes),
        }

    def _tracer_census_line(self):
        """Return the one-line end-of-run census, or ``None`` off the flag.

        Names which criterion bound most often, where and when the first cell
        activated, and whether the term the description DROPS (parallel
        transport) stayed small. Printed by ``run()`` on every tracer run.
        """
        if self._tracer is None or not self._tracer_census:
            return None
        census = self._tracer_census
        binding = np.asarray(census["criterion"], dtype=int)
        worst = np.asarray(census["worst_ratio"], dtype=float)
        ranked = np.bincount(
            binding[np.isfinite(worst)], minlength=len(CRITERION_NAMES)
        )
        dominant = CRITERION_NAMES[int(np.argmax(ranked))] if ranked.size else "none"
        transport = np.asarray(
            census.get("transport_ratio", np.zeros(0)), dtype=float
        )
        finite_transport = transport[np.isfinite(transport)]
        transport_text = (
            f"{float(np.max(finite_transport)):.3g}"
            if finite_transport.size
            else "n/a (gamma <= 0 everywhere: nothing is growing for "
                 "transport to be small against)"
        )
        first = self._tracer_first_activation
        first_text = (
            "no cell activated"
            if first is None
            else f"first activation t={first[0]:.6g} s cell {first[1]} on {first[2]}"
        )
        return (
            f"regime_r2 tracer census: binding criterion {dominant!r} "
            f"({int(np.max(ranked)) if ranked.size else 0} of {binding.size} "
            f"cells); {int(np.count_nonzero(census['passive']))} cells still "
            f"passive; {first_text}; refreshes={int(census['refreshes'])}; "
            f"worst dropped-transport ratio c_s/(L_n gamma)={transport_text} "
            "(NUMERICS.md tabulates where that stops being small)"
        )

    @property
    def geometry(self):
        return self._geometry

    def _unpack(self, y):
        """Unpack a packed vector with this run's declared optional fields.

        The explicit hints resolve the 6-field width ambiguity (``M_n`` vs
        ``nn_a``) and make a layout/flag desync a loud error instead of a
        silent misread.
        """
        return unpack_state(
            y,
            self._geometry.cells,
            neutral_momentum=self._neutral_momentum,
            neutral_two_zone=self._neutral_two_zone,
            neutral_annulus_momentum=self._neutral_two_momentum,
            neutral_energy=self._neutral_energy,
        )

    @property
    def state(self):
        return self._unpack(self._y)

    @property
    def derived(self):
        return derive_state(self.state, self._floors, self._ion_mass_g)

    @property
    def ion_mass_g(self):
        return self._ion_mass_g

    @property
    def mu(self):
        return self._mu

    @property
    def I_ion(self):
        return self._I_ion

    def _configure_kinetic_dvm(self):
        """Validate and build the K2a transient DVM arm (default-off).

        Every refusal below names what the arm DOES accept. The arm owns
        the whole neutral field once it engages, so a configuration that
        also asks a fluid term to own part of it is a contradiction, not a
        preference: those combinations raise here rather than silently
        letting one of the two win.
        """
        if not self._neutral_two_zone:
            raise ValueError(
                "neutral_model='kinetic_dvm' carries column AND annulus "
                "distributions and stores their moments in nn / nn_a: set "
                "the neutral_two_zone flag"
            )
        if self._neutral_momentum:
            raise ValueError(
                "neutral_model='kinetic_dvm' is incompatible with the "
                "neutral_momentum flag: the kinetic state already carries "
                "the neutral momentum as the first moment of f, so an "
                "evolved M_n field would be a second, unowned copy. "
                "Accepted: neutral_two_zone alone"
            )
        if self._gas_type != "He":
            raise ValueError(
                "neutral_model='kinetic_dvm' is wired for gas_type='He' "
                "only (the Phelps He+/He cross sections and the helium "
                f"velocity grid); got {self._gas_type!r}"
            )
        if float(
            self._input_dict.get("gas_puff_local_ionization_fraction")
        ) > 0.0:
            raise ValueError(
                "neutral_model='kinetic_dvm' is incompatible with "
                "gas_puff_local_ionization_fraction > 0: that channel "
                "removes puff neutrals on the fluid book, which the "
                "kinetic particle ledger would not see. Accepted: "
                "gas_puff_local_ionization_fraction = 0"
            )
        cadence = float(
            self._input_dict.get("neutral_kinetic_dvm_cadence_s")
        )
        if not (cadence > 0.0):
            raise ValueError(
                "neutral_kinetic_dvm_cadence_s must be positive "
                f"(got {cadence})"
            )
        accommodation = float(
            self._input_dict.get("neutral_kinetic_dvm_accommodation")
        )
        if not 0.0 <= accommodation <= 1.0:
            raise ValueError(
                "neutral_kinetic_dvm_accommodation is a surface property in "
                f"[0, 1] (got {accommodation})"
            )
        reflection = str(
            self._input_dict.get(
                "neutral_kinetic_dvm_wall_reflection"
            )
        )
        if reflection not in KINETIC_DVM_WALL_REFLECTION_MODELS:
            raise ValueError(
                "neutral_kinetic_dvm_wall_reflection must be one of "
                f"{KINETIC_DVM_WALL_REFLECTION_MODELS} (got {reflection!r})"
            )
        elastic = str(
            self._input_dict.get("neutral_kinetic_dvm_elastic")
        )
        if elastic not in KINETIC_DVM_ELASTIC_MODELS:
            raise ValueError(
                "neutral_kinetic_dvm_elastic must be one of "
                f"{KINETIC_DVM_ELASTIC_MODELS} (got {elastic!r})"
            )
        exchange = str(
            self._input_dict.get("neutral_kinetic_dvm_exchange")
        )
        if exchange not in KINETIC_DVM_EXCHANGE_MODELS:
            raise ValueError(
                "neutral_kinetic_dvm_exchange must be one of "
                f"{KINETIC_DVM_EXCHANGE_MODELS} (got {exchange!r})"
            )
        flights = str(
            self._input_dict.get(
                "neutral_kinetic_dvm_annulus_flights"
            )
        )
        if flights not in KINETIC_DVM_ANNULUS_FLIGHT_MODELS:
            raise ValueError(
                "neutral_kinetic_dvm_annulus_flights must be one of "
                f"{KINETIC_DVM_ANNULUS_FLIGHT_MODELS} (got {flights!r})"
            )
        relax_fraction = float(
            self._input_dict.get(
                "neutral_kinetic_dvm_transfer_relax_fraction"
            )
        )
        if not 0.0 < relax_fraction <= 1.0:
            raise ValueError(
                "neutral_kinetic_dvm_transfer_relax_fraction is the share of "
                "a cell's ion-energy margin the tick-frozen coupling drain "
                "may consume in one step and must lie in (0, 1] "
                f"(got {relax_fraction})"
            )
        transfer_hold = self._input_dict.get(
            "neutral_kinetic_dvm_transfer_hold"
        )
        transfer_hold = (
            KINETIC_DVM_TRANSFER_HOLDS[0]
            if transfer_hold is None
            else str(transfer_hold)
        )
        if transfer_hold not in KINETIC_DVM_TRANSFER_HOLDS:
            raise ValueError(
                "neutral_kinetic_dvm_transfer_hold must be one of "
                f"{KINETIC_DVM_TRANSFER_HOLDS} (got {transfer_hold!r})"
            )
        if bool(self._input_dict.get("gas_puff_enabled")):
            # The arm births the puff into the ANNULUS distribution, so a
            # puff cell with no annulus volume has nowhere to put it: the
            # 1/V_ann guard would drop the atoms while the particle ledger
            # still booked them as a birth. The profile SHAPE is
            # time-independent (the waveform only scales it), so the cells
            # it lands in are decidable here, at construction.
            share = gas_puff_rate_profile(
                self._geometry,
                1.0,
                float(self._input_dict.get("gas_puff_valves")),
                profile=str(self._input_dict.get("gas_puff_profile")),
                z_cm=self._input_dict.get("gas_puff_z_cm"),
                sigma_cm=float(
                    self._input_dict.get("gas_puff_sigma_cm")
                ),
                throw_cm=float(
                    self._input_dict.get("gas_puff_throw_cm")
                ),
                orifice_id_cm=self._input_dict.get(
                    "gas_puff_orifice_id_cm"
                ),
                orifice_length_cm=self._input_dict.get(
                    "gas_puff_orifice_length_cm"
                ),
                delivery_fraction=float(
                    self._input_dict.get("gas_puff_delivery_fraction")
                ),
            )
            V_ann = np.asarray(self._zone_volumes[1], dtype=float)
            starved = np.flatnonzero(
                (np.asarray(share, dtype=float) > 0.0) & (V_ann <= 0.0)
            )
            if starved.size:
                named = starved[:8].tolist()
                extra = starved.size - 8
                more = "" if extra <= 0 else f" (+{extra} more)"
                raise ValueError(
                    "neutral_model='kinetic_dvm' births the gas puff into "
                    "the ANNULUS distribution (the 300 K at-rest wall-port "
                    "convention), so every cell the puff profile lands in "
                    "must have an annulus to receive it; "
                    f"{starved.size} puff cell(s) carry V_ann = 0 "
                    f"(Rm == Rp there): {named}{more}. Those cells would "
                    "swallow their share of the fueling while the particle "
                    "ledger still booked it as a birth. Accepted: a "
                    "gas_puff_profile whose support lies in cells with "
                    "Rm > Rp, or gas_puff_enabled = False"
                )
        # B5: the cathode-side energetic recycle. Default off, and the whole
        # channel is absent (the engine takes ``cathode_jet=None``) rather
        # than present at a neutral setting, so nothing about the shipped
        # arm changes with the family in the config.
        cathode_jet = None
        if bool(
            self._input_dict.get("neutral_kinetic_dvm_cathode_jet")
        ):
            refuse_cathode_backscatter_double_book(self._input_dict)
            refuse_dvm_cathode_jet_without_cathode_coupling(
                self._input_dict, self._flags
            )
            cathode_jet = {
                "R_N": float(
                    self._input_dict.get(
                        "neutral_kinetic_dvm_cathode_jet_R_N"
                    )
                ),
                "R_E": float(
                    self._input_dict.get(
                        "neutral_kinetic_dvm_cathode_jet_R_E"
                    )
                ),
                "T_launch_eV": self._input_dict.get(
                    "neutral_kinetic_dvm_cathode_jet_T_launch_eV"
                ),
            }
        self._dvm_cathode_jet = cathode_jet
        self._dvm_cadence_s = cadence
        self._dvm_transfer_relax_fraction = relax_fraction
        self._dvm_transfer_hold = transfer_hold
        anode_faces = np.asarray(
            getattr(self._geometry, "anode_face_indices", ()), dtype=int
        )
        # B4: the anode-side energetic recycle, default off and ABSENT rather
        # than present at a neutral setting, exactly as B5 is. The order of
        # the three refusals below is deliberate: the pair guard and the
        # cathode-solve prerequisite are statements about the CONFIGURATION,
        # the geometry assertion is a statement about the MESH, and the
        # coefficient interval is checked inside the engine -- so a config
        # error names itself rather than tripping the geometry check first.
        anode_jet = None
        if bool(self._input_dict.get("neutral_kinetic_dvm_anode_jet")):
            refuse_anode_backscatter_double_book(self._input_dict)
            refuse_dvm_anode_jet_without_cathode_coupling(
                self._input_dict, self._flags
            )
            # The DVM carries ONE mesh face; the fluid carries a LIST of anode
            # faces. The channel's launch direction is defined against the
            # single face the DVM took, so a geometry with more than one anode
            # face (or none) would have the kinetic arm re-emitting off one
            # mesh while the fluid geometry describes another. Refused rather
            # than resolved: which face the atoms came off is not a choice
            # this code is entitled to make.
            mesh_face = int(anode_faces[0]) if anode_faces.size else -999
            if anode_faces.size != 1 or not 0 < mesh_face < self._geometry.cells:
                raise ValueError(
                    "neutral_kinetic_dvm_anode_jet launches the collected "
                    "stream back AWAY from the anode mesh, on the side it was "
                    "collected from, and the transient DVM carries exactly "
                    "ONE mesh face to define that side against. This geometry "
                    f"carries anode_face_indices={anode_faces.tolist()} on "
                    f"{self._geometry.cells} cells, so the kinetic arm's "
                    f"mesh face ({mesh_face}) is not a single interior face "
                    "the fluid geometry agrees on. Accepted: a geometry with "
                    "exactly one interior anode face, or "
                    "neutral_kinetic_dvm_anode_jet = False"
                )
            anode_jet = {
                "R_N": float(
                    self._input_dict.get(
                        "neutral_kinetic_dvm_anode_jet_R_N"
                    )
                ),
                "R_E": float(
                    self._input_dict.get(
                        "neutral_kinetic_dvm_anode_jet_R_E"
                    )
                ),
                "T_launch_eV": self._input_dict.get(
                    "neutral_kinetic_dvm_anode_jet_T_launch_eV"
                ),
            }
        self._dvm_anode_jet = anode_jet
        # B6: the thin annular baffles, default off and ABSENT rather than
        # present at a neutral setting, exactly as the two jets are. The
        # geometry has already validated and mapped them onto faces (and has
        # already refused a clear radius below the local column radius, and a
        # baffle array supplied without its flag); what this flag decides is
        # whether the KINETIC annulus sees them at all. The two refusals below
        # are the two ways a config can ask for that and mean nothing by it.
        baffle_faces = ()
        baffle_radii = ()
        if bool(self._flags.get("neutral_kinetic_dvm_baffles")):
            if not bool(self._flags.get("neutral_baffles")):
                raise ValueError(
                    "neutral_kinetic_dvm_baffles makes the geometry's thin "
                    "annular baffles act on the kinetic annulus, and this "
                    "geometry has none: the fluid neutral_baffles flag is "
                    "off, so neutral_baffle_positions_cm and "
                    "neutral_baffle_clear_radii_cm are forbidden and unset. "
                    "Accepted: arm neutral_baffles with both arrays, or leave "
                    "neutral_kinetic_dvm_baffles off"
                )
            baffle_faces = np.asarray(
                self._geometry.neutral_baffle_face_indices, dtype=int
            )
            baffle_radii = np.asarray(
                self._geometry.neutral_baffle_clear_radius_cm, dtype=float
            )
            if baffle_faces.size == 0:
                raise ValueError(
                    "neutral_kinetic_dvm_baffles is armed but the resolved "
                    "geometry carries no baffle faces, so the channel would "
                    "be silently inert. Accepted: neutral_baffles with a "
                    "non-empty neutral_baffle_positions_cm and "
                    "neutral_baffle_clear_radii_cm, or "
                    "neutral_kinetic_dvm_baffles off"
                )
        self._dvm = TransientDVM(
            geometry=self._geometry,
            nvz=int(self._input_dict.get("neutral_kinetic_dvm_nvz")),
            nvp=int(self._input_dict.get("neutral_kinetic_dvm_nvp")),
            accommodation=accommodation,
            wall_reflection=reflection,
            elastic_model=elastic,
            exchange_model=exchange,
            annulus_flights=flights,
            cathode_jet=cathode_jet,
            anode_jet=anode_jet,
            transparency=1.0 - float(self._input_dict.get("eta")),
            mesh_face=int(anode_faces[0]) if anode_faces.size else -999,
            baffle_faces=baffle_faces,
            baffle_clear_radius_cm=baffle_radii,
            s_L=self._dvm_end_sticking("S_pump_L"),
            s_R=self._dvm_end_sticking("S_pump_R"),
        )

    def _dvm_end_sticking(self, key):
        """Return the end-plane sticking probability of a pump speed [L/s].

        The TPMC/KN2Zone convention: the pumping speed over the one-way
        300 K thermal flux through the end plane, clipped to a probability.
        ``key`` names the end -- ``"S_pump_L"`` the first cell's outer face,
        ``"S_pump_R"`` the last cell's -- and the flux area is THAT end's own
        open neutral area, which is the area the kinetic march counts its
        end-face outflow through. The realized pumping speed is
        ``sticking * area * vbar / 4``, so an area belonging to the other end
        realizes a different speed than the one configured wherever the vessel
        differs between the two ends.

        The instruments named above carry the per-end form too, as of
        2026-08-26 (``physics/kinetic_neutrals.py``, ``scripts/kinetic/mc_neutrals.py``
        and the two ``neutral_arch_*`` benches took ``pi Rm[-1]**2`` at BOTH
        ends until then). Their end-plane area is ``pi Rm**2`` outright, since
        they model neither the obstruction cross-section nor the support rods
        that ``neutral_area_cm2`` subtracts here; on a machine carrying either,
        that residual difference stands.

        Where the end cell is a plenum, ``pump_elbow_conductance_lps`` folds
        into the speed in series first, the same restriction the fluid pump
        term applies at the same cell; a pump on any other role has no
        unmodeled elbow in front of it.

        ``vbar`` carries ``m_He_cgs``, the DVM/KN2Zone neutral mass: the
        probability returned here is consumed by a march whose end-face
        outflow is counted on a velocity grid built on that same mass, so
        the realized speed ``sticking * area * vbar / 4`` is only the
        configured one when both halves of the identity use it.
        """
        index = {"S_pump_L": 0, "S_pump_R": -1}[key]
        speed = float(self._input_dict.get(key, 0.0))
        if speed <= 0.0:
            return 0.0
        speed = _effective_pump_speed(
            speed,
            self._input_dict.get("pump_elbow_conductance_lps")
            if is_plenum_cell(self._geometry, index)
            else None,
        )
        vbar = math.sqrt(8.0 * kb_cgs * 300.0 / (math.pi * m_He_cgs))
        area = float(np.asarray(self._geometry.neutral_area_cm2)[index])
        return min(speed * 1.0e3 / (area * vbar / 4.0), 1.0)

    @property
    def floors(self):
        return dict(self._floors)

    @property
    def time(self):
        return self._time

    def get_config(self):
        """Return copies of the input dictionary and flags used by this object."""
        return dict(self._input_dict), dict(self._flags)

    def get_initial_snapshot(self):
        """Return geometry, conservative state, and derived fields for inspection."""
        state = self.state
        derived = self.derived
        assert_finite_state(state, derived)
        return SimpleNamespace(
            geometry=self._geometry,
            state=state,
            derived=derived,
            y=pack_state(state),
            time=self._time,
        )

    def rhs(
        self, y=None, include_heat_conduction=True, time=None,
        step_window=None,
    ):
        """Return the packed explicit RHS for the current scaffold physics.

        ``step_window`` is the ``(t0, dt)`` interval an integrator is stepping
        over, passed through to the terms that integrate their own explicit
        time dependence exactly over it. Only the probe source reads it; every
        other term is evaluated pointwise at ``time`` as before, so omitting it
        (the default, and what every diagnostic caller does) is the historical
        behaviour exactly.

        With ``beam_deposition_in_heat_substep`` armed this is operator A's
        RHS with the ``beam_power_deposition`` row LEFT OUT of the sum: that
        row is applied by the implicit heat substep instead
        (:meth:`operator_split_step`). The row is still built and still
        REPORTED by :meth:`rhs_terms` with the same deposited power -- only
        which operator applies it moves.
        """
        state_rhs = self._zero_rhs_state()
        terms = self.rhs_terms(
            y=y,
            include_heat_conduction=include_heat_conduction,
            time=time,
            step_window=step_window,
        )
        if self._beam_deposition_in_heat_substep:
            for name, term in terms.items():
                if name == BEAM_POWER_DEPOSITION_TERM:
                    continue
                state_rhs = add_state_rhs(state_rhs, term)
        else:
            for term in terms.values():
                state_rhs = add_state_rhs(state_rhs, term)
        self._accumulate_dvm_ion_booking(terms)
        self._accumulate_dvm_source_booking()
        self._accumulate_coverage_burn(terms)
        # With optional fields on, the packed RHS must always match the
        # state vector's width, even when no term touched them (pads zeros).
        return pack_state(
            state_rhs,
            neutral_momentum=True if self._neutral_momentum else None,
            neutral_two_zone=True if self._neutral_two_zone else None,
            neutral_annulus_momentum=(
                True if self._neutral_two_momentum else None
            ),
            neutral_energy=True if self._neutral_energy else None,
        )

    def _explicit_stage_rhs(self, dt, include_heat_conduction=True):
        """Return the SSPRK2 stage RHS callable for a step of width ``dt``.

        With no probe source this is exactly the historical
        ``self.rhs(y, stage_time)`` -- the presence gate extends to the CALL
        SIGNATURE, so nothing that wraps, replaces or subclasses ``rhs`` sees
        an argument that did not exist before, and the off path's stage calls
        are unchanged down to their keywords.

        With a probe armed, the step window rides along so the term can
        integrate its own explicit time dependence exactly over it. It is
        passed as an ARGUMENT rather than armed on the solver: it is an input
        to the RHS, and threading it makes it structurally impossible for a
        rejected attempt's ``dt`` to be read by the attempt that replaces it.
        """
        extra = (
            {} if include_heat_conduction
            else {"include_heat_conduction": False}
        )
        if self._probe is not None:
            extra["step_window"] = (self._time, dt)
        return lambda yy, tt: self.rhs(yy, time=tt, **extra)

    def rhs_terms(
        self, y=None, include_heat_conduction=True, time=None,
        step_window=None,
    ):
        """Return named conservative RHS contributions for diagnostics.

        ``step_window`` (see :meth:`rhs`) is read only by the probe source.
        """
        # The coverage closure's reservoir-arm debit belongs to THIS
        # evaluation's beam solve and nothing else. Cleared first so a branch
        # that never reaches the beam terms (the neutral-only pre-drive, or
        # Plasma off) cannot leave the accumulator reading a stale solve.
        self._coverage_reservoir_debit = None
        # Likewise the engaged DVM arm's boundary-inflow rows: they belong to
        # THIS evaluation's terms, and a branch that never reaches the strip
        # must not leave the counted handshake reading an older set. The
        # cathode jet's incident-energy row is the same statement about the
        # energy half of that pair.
        self._dvm_source_rows = None
        self._dvm_cathode_jet_incident_row = None
        self._dvm_anode_jet_incident_row = None
        state = self.state if y is None else self._unpack(y)
        # The zone-exchange term exists only in two-zone runs, so the term
        # ledger (and the saved rhs_terms structure) is unchanged when the
        # flag is off. It is pure free-molecular mixing, so it runs in the
        # neutral-only phases too.
        zone_terms = {}
        if self._neutral_two_zone:
            zone_terms["neutral_zone_exchange"] = self.neutral_zone_exchange_rhs(
                state=state
            )
        # The neutral-energy wall sink exists only under its own flag, so the
        # term ledger (and the saved rhs_terms structure) is unchanged when
        # the flag is off. Present in BOTH branches below and identically zero
        # in the neutral-only one -- that phase runs the backward-Euler
        # neutral matrix, which En passes through untouched -- so the saved
        # structure is stable across the phase change.
        energy_wall_terms = {}
        if self._neutral_energy:
            energy_wall_terms["neutral_energy_wall"] = self._zero_rhs_state()
            energy_wall_terms["neutral_cx_channel"] = self._zero_rhs_state()
            energy_wall_terms["neutral_hot_channel"] = self._zero_rhs_state()
            if self._cathode_jet_enabled:
                energy_wall_terms["cathode_jet_neutral_energy"] = (
                    self._zero_rhs_state()
                )
            if self._cathode_jet_carrier:
                energy_wall_terms["cathode_jet_hot_carrier"] = (
                    self._zero_rhs_state()
                )
        # The ad-hoc probe source exists only under its own flag, so the term
        # ledger (and the saved rhs_terms structure) is unchanged when the flag
        # is off. Present in BOTH branches below and identically zero in the
        # neutral-only one: while the solver is on the implicit neutral-only
        # stepper (Plasma off, or the neutral_prebreakdown phase) the step is a
        # backward-Euler neutral matrix that this term deliberately does not
        # enter, so the probe cannot fuel a pre-shot fill or a cached
        # equilibration seed. Recording the zero rather than dropping the key
        # keeps the saved structure stable across the phase change and makes
        # the gate readable instead of inferable.
        probe_terms = {}
        geometry_terms = {}
        if self._variable_area_geometry:
            geometry_terms["flux_tube_geometry"] = (
                self._zero_rhs_state()
                if not self._flags.get("Plasma")
                or self._neutral_prebreakdown_active(time=time)
                else self.flux_tube_geometry_rhs(state=state)
            )
        if not self._flags.get("Plasma") or self._neutral_prebreakdown_active(
            time=time,
        ):
            kinetic_terms = {}
            if self._kinetic is not None:
                kinetic_terms["neutral_kinetic_relaxation"] = (
                    self.neutral_kinetic_relaxation_rhs(state)
                )
            if self._dvm is not None:
                kinetic_terms["neutral_kinetic_dvm_coupling"] = (
                    self.neutral_kinetic_dvm_coupling_rhs()
                )
            if self._probe is not None:
                probe_terms["neutral_probe_source"] = self._zero_rhs_state()
            terms = {
                **zone_terms,
                **probe_terms,
                **energy_wall_terms,
                **geometry_terms,
                **kinetic_terms,
                "plasma_advective_flux": self._zero_rhs_state(),
                "plasma_front_flux": self._zero_rhs_state(),
                # boundary_absorption is permanently zero everywhere since the
                # legacy absorber was retired 2026-08-31 (Tom); kept for saved-
                # ledger schema stability.
                "boundary_absorption": self._zero_rhs_state(),
                "characteristic_boundary": self._zero_rhs_state(),
                "pressure_work": self._zero_rhs_state(),
                # Present with zero rows whether or not the flag is armed, so
                # the saved term structure is stable across the phase change
                # AND across the flag. There is no drive in this branch, so an
                # armed run books zero here too.
                "electron_drift_transport": self._zero_rhs_state(),
                "hyperbolic_dissipation_heating": self._zero_rhs_state(),
                "ei_exchange": self._zero_rhs_state(),
                "ionization_energy_cost": self._zero_rhs_state(),
                "electron_ion_cooling": self._zero_rhs_state(),
                "electron_neutral_cooling": self._zero_rhs_state(),
                "ion_charge_exchange": self._zero_rhs_state(),
                "ion_neutral_drag": self._zero_rhs_state(),
                "ion_neutral_frictional_heating": self._zero_rhs_state(),
                "ion_neutral_thermalization": self._zero_rhs_state(),
                "ion_neutral_collision": self._zero_rhs_state(),
                "neutral_momentum_wall": self._zero_rhs_state(),
                "neutral_wind_advection": self._zero_rhs_state(),
                "surface_loss": self._zero_rhs_state(),
                "anode_collection": self._zero_rhs_state(),
                "cathode_surface_loss": self._zero_rhs_state(),
                "anode_e_sheath_loss": self._zero_rhs_state(),
                "neutral_exchange": self.neutral_exchange_rhs(state=state),
                "neutral_sources": self.neutral_source_sink_rhs(
                    state=state,
                    time=time,
                ),
                "gas_puff_local_ionization": self._zero_rhs_state(),
                "ionization_birth": self._zero_rhs_state(),
                "beam_ionization_birth": self._zero_rhs_state(),
                "beam_power_deposition": self._zero_rhs_state(),
                "beam_ionization_cost": self._zero_rhs_state(),
                "beam_excitation_radiation": self._zero_rhs_state(),
                "recombination_rad_loss": self._zero_rhs_state(),
                "recombination_3b_loss": self._zero_rhs_state(),
                "heat_conduction": self._zero_rhs_state(),
            }
            return self._attach_neutral_energy_rows(
                self._apply_active_plasma_topology(terms), state
            )
        plasma_terms = self.plasma_flux_rhs_terms(state=state)
        # I3 instrument. Armed, the bulk reaction terms are evaluated at the
        # step-START accepted state -- ``self._y`` is only rewritten when a
        # step is ACCEPTED, so it is exactly that state for every SSPRK2 stage
        # and every rejected attempt -- which freezes the rates across the
        # step and caps it at first order in them. Unarmed, this IS ``state``,
        # object for object, so the evaluations below are unchanged.
        reaction_state = (
            self.state if self._rates_at_accepted_state else state
        )
        reaction_terms = self.reaction_rhs_terms(state=reaction_state)
        electron_cooling_terms = self.electron_cooling_rhs_terms(state=state)
        cathode_phase = self._cathode_phase_options(time=time)
        cathode_solve = None
        if cathode_phase["solve_enabled"]:
            cathode_solve = self.solve_cathode_boundary(
                state=state,
                floating=cathode_phase["floating"],
                time=time,
                update_cache=True,
            )
        beam_terms = self.beam_ionization_rhs_terms(
            state=state,
            cathode_solve=cathode_solve,
            time=time,
        )
        # Side channel, not a term: the reservoir arm's neutral debit. It is
        # read out here and deliberately NOT placed in the ledger below, so
        # the RHS sum and the saved term structure are untouched by it.
        #
        # It must carry the SAME plasma-topology mask the beam term it was
        # split out of will get, or the two stop being a split: the accumulator
        # subtracts this from the (masked) beam row, so an unmasked debit would
        # leave a spurious positive residue on every plasma-dead cell -- the
        # plenum and the obstruction behind the cathode, where no birth is
        # applied at all.
        _reservoir_debit = beam_terms.get("coverage_reservoir_nn_debit")
        if _reservoir_debit is not None and self._active_plasma_topology:
            _reservoir_debit = np.where(
                np.asarray(self._geometry.plasma_active, dtype=bool),
                np.asarray(_reservoir_debit, dtype=float),
                0.0,
            )
        self._coverage_reservoir_debit = _reservoir_debit
        if self._probe is not None:
            probe_terms["neutral_probe_source"] = self.neutral_probe_source_rhs(
                state=state,
                time=time,
                step_window=step_window,
            )
        ionization_rate_per_neutral = None
        if self._neutral_energy:
            energy_wall_terms["neutral_energy_wall"] = (
                self.neutral_energy_wall_rhs(state=state)
            )
            energy_wall_terms["neutral_cx_channel"] = (
                self.neutral_cx_channel_rhs(state=state)
            )
            # The hot channel's in-flight ionization must use the SAME
            # per-neutral ionization frequency the bulk channel is using on
            # this evaluation, so it is read back off the reaction term rather
            # than recomputed from the rate tables. The directed surface
            # carrier's in-beam ionization reads the very same array.
            ionization_rate_per_neutral = np.asarray(
                reaction_terms["ionization_birth"].n, dtype=float
            ) / np.maximum(
                np.asarray(reaction_state.nn, dtype=float), self._floors["nn"]
            )
            energy_wall_terms["neutral_hot_channel"] = (
                self.neutral_hot_channel_rhs(
                    state=state,
                    ionization_rate=ionization_rate_per_neutral,
                )
            )
        # The directed surface carrier's launch is the recycle share the
        # boundary term WITHHOLDS for it on this same evaluation. It travels
        # through this dict rather than being recomputed, so the withdrawal
        # and the launch are one number; ``None`` leaves the boundary term on
        # its historical path, keyword for keyword.
        carrier_out = {} if self._cathode_jet_carrier else None
        # The sheath-resolved electrode solve produces BOTH electrode rows in
        # one call. Bound here rather than inline in the dict so the solve
        # happens exactly once: calling it twice would pay for a second sheath
        # solve and let the two rows come from different evaluations.
        electrode_terms = self.cathode_source_terms(
            state=state,
            cathode_solve=cathode_solve,
            time=time,
        )
        # The KEP correction is booked in TWO places, so it is resolved here
        # rather than inline: its pressure half folds into "pressure_work",
        # which then IS the energy-consistent pressure work, and its Rusanov
        # dissipation half is the "hyperbolic_dissipation_heating" row. Both
        # rows are always present -- unarmed, the pressure row is the
        # uncorrected one and the dissipation row is the zero state, so the
        # saved term structure does not move with the flag.
        pressure_work = self.pressure_work_rhs(state=state)
        if self._hyperbolic_energy_consistent:
            kep_pressure, hyperbolic_dissipation = (
                self.hyperbolic_energy_correction_rhs(state=state)
            )
            pressure_work = add_state_rhs(pressure_work, kep_pressure)
        else:
            hyperbolic_dissipation = self._zero_rhs_state()
        terms = {
            **zone_terms,
            **probe_terms,
            **geometry_terms,
            "plasma_advective_flux": plasma_terms["plasma_advective_flux"],
            "plasma_front_flux": plasma_terms["plasma_front_flux"],
            # Permanently zero since the legacy volumetric absorber was
            # retired 2026-08-31 (Tom). The ROW is kept because it is part of
            # the saved ledger schema that existing artifacts and the
            # committed phase-3 RHS provenance enumerate; dropping it would
            # move the saved term set, which this change deliberately does
            # not. Nothing can write it: read the live boundary from
            # "characteristic_boundary" below.
            "boundary_absorption": self._zero_rhs_state(),
            "characteristic_boundary": self.characteristic_boundary_rhs(
                state=state,
                cathode_solve=cathode_solve,
                time=time,
                carrier_out=carrier_out,
            ),
            "pressure_work": pressure_work,
            # The electron-velocity correction to the row above: pressure_work
            # books with the ION velocity, which is exact only where J = 0.
            # Presence-gated -- unarmed, the zero state is recorded and
            # Gamma_d is never evaluated -- and always present, so the saved
            # term structure does not move with the flag.
            "electron_drift_transport": (
                self.electron_drift_transport_rhs(
                    state=state, cathode_solve=cathode_solve
                )
                if self._electron_drift is not None
                else self._zero_rhs_state()
            ),
            # The Rusanov (n, M) numerical kinetic-energy dissipation, deposited
            # into the ion internal energy. It sits in the slot the combined
            # correction row occupied, so the dissipation booking keeps its
            # place in the fold that advances the state.
            "hyperbolic_dissipation_heating": hyperbolic_dissipation,
            "ei_exchange": self.energy_exchange_rhs(state=state),
            "ionization_energy_cost": electron_cooling_terms[
                "ionization_energy_cost"
            ],
            "electron_ion_cooling": electron_cooling_terms[
                "electron_ion_cooling"
            ],
            "electron_neutral_cooling": electron_cooling_terms[
                "electron_neutral_cooling"
            ],
            "ion_charge_exchange": self.ion_charge_exchange_rhs(state=state),
            "ion_neutral_drag": self.ion_neutral_drag_rhs(state=state),
            "ion_neutral_frictional_heating": (
                self.ion_neutral_frictional_heating_rhs(state=state)
            ),
            "ion_neutral_thermalization": (
                self.ion_neutral_thermalization_rhs(state=state)
            ),
            "ion_neutral_collision": (
                self.ion_neutral_collision_rhs(state=state)
            ),
            "neutral_momentum_wall": self.neutral_momentum_wall_rhs(state=state),
            **energy_wall_terms,
            **(
                {
                    "neutral_momentum_radial": (
                        self.neutral_momentum_two_zone_rhs(state=state)
                    )
                }
                if self._neutral_two_momentum
                else {}
            ),
            "neutral_wind_advection": self.neutral_wind_advection_rhs(state=state),
            "surface_loss": self._zero_rhs_state(),
            "anode_collection": self.anode_collection_rhs(
                state=state, cathode_solve=cathode_solve, time=time
            ),
            # The electrode electron sheath power is booked PER ELECTRODE, in
            # two rows, because one row cannot be named honestly: its Ee field
            # was ~100% ANODE in discharge (the cathode sheath repels plasma
            # electrons, so the cathode share is milliwatts against ~10^5 W)
            # while its particle rows were entirely cathode surface physics.
            #
            # cathode_surface_loss -- the cathode's own members: the face's
            # particle loss and recycle on n/nn/M, those ions' thermal energy
            # on Ei, and the CATHODE electron sheath share on Ee.
            "cathode_surface_loss": electrode_terms.rhs,
            # anode_e_sheath_loss -- the ANODE electron sheath deposit, on Ee
            # alone; every other field is zero. It lands at the anode-flanking
            # cells under the Bohm split weights. WITH NO ANODE RESOLVED in
            # the mesh the deposit falls back to the cathode CELL but is still
            # booked to THIS row: the name says which electrode paid, not
            # which cell received it.
            #
            # SUM INVARIANCE: these two sum to the single combined row they
            # replace, bit-exactly wherever an anode is resolved (their
            # per-cell supports are disjoint there, so every cell of one is an
            # exact zero where the other is not, and adding an exact zero into
            # the running total is exact). The unresolved-anode fallback is the
            # one site where they share a cell and the sum agrees to roundoff.
            "anode_e_sheath_loss": electrode_terms.anode_rhs,
            "neutral_exchange": self.neutral_exchange_rhs(state=state),
            "neutral_sources": self.neutral_source_sink_rhs(
                state=state,
                time=time,
            ),
            "gas_puff_local_ionization": self.gas_puff_local_ionization_rhs(
                state=state,
                time=time,
            ),
            "ionization_birth": reaction_terms["ionization_birth"],
            "beam_ionization_birth": beam_terms["beam_ionization_birth"],
            "beam_power_deposition": beam_terms["beam_power_deposition"],
            "beam_ionization_cost": beam_terms["beam_ionization_cost"],
            "beam_excitation_radiation": beam_terms["beam_excitation_radiation"],
            "recombination_rad_loss": reaction_terms["recombination_rad_loss"],
            "recombination_3b_loss": reaction_terms["recombination_3b_loss"],
            "recombination_energy_return": (
                self.recombination_energy_return_rhs(state=state)
            ),
            "heat_conduction": self._zero_rhs_state(),
        }
        if include_heat_conduction:
            terms["heat_conduction"] = self.heat_conduction_rhs(state=state)
        if self._neutral_energy and self._cathode_jet_enabled:
            # Reads the recycle flux the boundary term just computed, so the
            # jet's energy cannot describe a different flux from the one that
            # actually rebirthed the atoms.
            recycle = terms["characteristic_boundary"].nn
            terms["cathode_jet_neutral_energy"] = (
                self.cathode_jet_neutral_energy_rhs(
                    state=state,
                    cathode_solve=cathode_solve,
                    recycle_nn_row=recycle,
                )
            )
        if self._cathode_jet_carrier:
            terms["cathode_jet_hot_carrier"] = (
                self.cathode_jet_hot_carrier_rhs(
                    state=state,
                    cathode_solve=cathode_solve,
                    launch_per_s=carrier_out.get("launch_per_s"),
                    ionization_rate=ionization_rate_per_neutral,
                )
            )
        if self._kinetic is not None:
            # K4a supersession: once targets exist, every term's neutral
            # rows are carried by the kinetic relaxation instead (plasma
            # rows keep their exact forms). The relaxation key is present
            # from the start (zeros before the first refresh) so the saved
            # rhs_terms structure is stable across the run.
            if self._kinetic.target_col is not None and state.nn_a is not None:
                terms = {
                    name: self._strip_neutral_rows(term)
                    for name, term in terms.items()
                }
            terms["neutral_kinetic_relaxation"] = (
                self.neutral_kinetic_relaxation_rhs(state)
            )
        if self._dvm is not None:
            # K2a supersession: once the arm engages it owns every neutral
            # row, and the ion-side momentum/energy of the channels it
            # models, which the coupling term carries instead. The key is
            # present from the first step so the saved term structure is
            # stable across engagement.
            if self._dvm_engaged:
                # B1 counted handshake: read the boundary-inflow rows the
                # arm's sources are built from BEFORE the strip below zeroes
                # them. Side channel, not a term -- the RHS sum and the saved
                # term structure are untouched by it.
                self._dvm_source_rows = self._dvm_source_channel_rows(terms)
                if self._dvm_cathode_jet is not None:
                    # B5: the INCIDENT ion energy those recycle particles
                    # arrived with, at THIS evaluation's committed sheath and
                    # ion temperature. Taken here, beside the count it
                    # belongs to, so the pair is one reading of one state.
                    self._dvm_cathode_jet_incident_row = (
                        self._dvm_cathode_jet_incident_energy_row(
                            self._dvm_source_rows["cathode_face"],
                            state,
                            cathode_solve,
                        )
                    )
                if self._dvm_anode_jet is not None:
                    # B4: the same reading for the anode-collected ions, off
                    # the same evaluation's solve and state.
                    self._dvm_anode_jet_incident_row = (
                        self._dvm_anode_jet_incident_energy_row(
                            self._dvm_source_rows["anode"],
                            state,
                            cathode_solve,
                        )
                    )
                terms = {
                    name: self._strip_dvm_rows(name, term)
                    for name, term in terms.items()
                }
            terms["neutral_kinetic_dvm_coupling"] = (
                self.neutral_kinetic_dvm_coupling_rhs()
            )
        return self._attach_neutral_energy_rows(
            self._apply_active_plasma_topology(terms), state
        )

    def _attach_neutral_energy_rows(self, terms, state):
        """Give every neutral-moving term the ``En`` row its ``nn`` row implies.

        Terms that own their ``En`` row are left alone; the rest get one built
        from the density row they already computed and the birth energy
        ``_NEUTRAL_ENERGY_TERM_BOOKING`` states for them. Doing it here, once,
        rather than inside each of a dozen physics functions is what makes the
        coverage auditable: a term absent from the table raises, so a new
        neutral source cannot be added without saying what temperature its gas
        arrives at.

        It runs AFTER the plasma-topology mask, so an ``En`` row can never
        appear on a cell whose ``nn`` row was masked away.

        It also records the ionization-birth thermal deficit rows
        (:data:`IONIZATION_BIRTH_DEFICIT_DIAGNOSTIC_FIELDS`) on the side
        channel ``self._birth_deficit_diagnostics``, from the same local
        ``Tn`` the sinks are built with. Those rows are DIAGNOSTIC: nothing
        here or downstream adds them to a state or to the RHS ledger.
        """
        if state.En is None:
            return terms
        Tn = ionization_birth_neutral_temperature_eV(
            state, self._floors, self._input_dict.get("Tn_K")
        )
        wall_energy = NEUTRAL_ENERGY_FLOOR_T_K * kb_cgs
        derived = derive_state(
            state, floors=self._floors, ion_mass_g=self._ion_mass_g
        )
        ion_energy = 1.5 * derived.Ti * ev_to_erg
        self._birth_deficit_diagnostics = self._ionization_birth_deficit_rows(
            terms, derived=derived, Tn=Tn
        )
        attached = {}
        for name, term in terms.items():
            try:
                mode = _NEUTRAL_ENERGY_TERM_BOOKING[name]
            except KeyError:
                raise ValueError(
                    f"RHS term {name!r} has no neutral-energy booking. Every "
                    "term that can move neutrals must state the energy its "
                    "particles carry (add it to "
                    "_NEUTRAL_ENERGY_TERM_BOOKING); a term that never touches "
                    "them is declared 'none'."
                ) from None
            if mode == "owns":
                attached[name] = term
                continue
            birth = {
                "none": None,
                "local": None,
                "wall": 1.5 * wall_energy,
                "ion": ion_energy,
            }[mode]
            attached[name] = ConservativeState1D(
                n=term.n,
                nn=term.nn,
                M=term.M,
                Ee=term.Ee,
                Ei=term.Ei,
                M_n=term.M_n,
                nn_a=term.nn_a,
                M_n_a=term.M_n_a,
                En=neutral_energy_transfer_row(
                    term.nn, Tn, birth_energy_erg=birth
                ),
            )
        return attached

    def _ionization_birth_deficit_rows(self, terms, derived, Tn):
        """Return the per-cell ionization-birth thermal deficit rows [W cm^-3].

        Each ionization channel books ONE event twice: the ``En`` sink removes
        the local ``(3/2) k Tn`` per consumed atom, and the ``Ei`` birth adds
        ``(3/2) k T_birth`` per born ion. The two agree only when the ion is
        born at the neutral temperature; otherwise the difference leaves the
        model without a named home, and these rows are where it is named.
        Signed POSITIVE for energy that leaves, on the PLASMA volume (the
        volume the ``Ei`` row lives on), so a row is directly comparable with
        the ion-energy term powers.

        Diagnostic only. The rows are recorded on a side channel and are never
        summed into a state, an RHS ledger, or a timestep bound.
        """
        Ti_birth_ionization = self._input_dict.get(
            "Ti_birth_ionization"
        )
        Ti_birth = _birth_temperature(
            Ti_birth_ionization,
            derived.Ti,
            self._floors["Ti"],
            neutral_temperature=(
                Tn if Ti_birth_ionization == "neutral" else None
            ),
        )
        # Written exactly as the two sides write them: the sink's per-particle
        # energy is neutral_energy_transfer_row's own ``local``, and the birth's
        # is the reaction/beam ``1.5 * ev_to_erg * Ti_birth`` factor.
        per_particle = (
            1.5 * np.asarray(Tn, dtype=float) * ev_to_erg
            - 1.5 * ev_to_erg * Ti_birth
        )
        rows = {}
        total = np.zeros(self._geometry.cells, dtype=float)
        for term_name, field_name in IONIZATION_BIRTH_DEFICIT_SITES:
            term = terms.get(term_name)
            S = (
                np.zeros(self._geometry.cells, dtype=float)
                if term is None
                else np.asarray(term.n, dtype=float)
            )
            row = per_particle * S * 1.0e-7  # erg/s/cm^3 -> W/cm^3
            rows[field_name] = row
            total = total + row
        rows[IONIZATION_BIRTH_DEFICIT_DIAGNOSTIC_FIELDS[0]] = total
        return rows

    def _apply_active_plasma_topology(self, terms):
        """Mask plasma-coupled terms on typed plasma-dead cells.

        With the R2 tracer engaged the mask also covers the cells the tracer
        owns: their plasma rows are the exact affine update's, so a fluid
        contribution to them would be a second, unowned opinion about the same
        density. The tracer refuses to construct without
        ``active_plasma_topology``, so this method is always reached when it is
        engaged.
        """
        if not self._active_plasma_topology:
            return terms
        neutral_only = {
            "neutral_zone_exchange",
            "neutral_momentum_wall",
            "neutral_wind_advection",
            "neutral_exchange",
            "neutral_sources",
            "neutral_kinetic_relaxation",
            # The probe is a neutral source like the puff: gas arrives where
            # the caller put it, whether or not the plasma reaches that cell.
            "neutral_probe_source",
        }
        return {
            name: (
                term
                if name in neutral_only
                else self._mask_inactive_rhs(term, include_neutral=True)
            )
            for name, term in terms.items()
        }

    def _mask_inactive_rhs(self, term, include_neutral):
        active = self._plasma_active_mask()

        def masked(values):
            if values is None:
                return None
            return np.where(active, np.asarray(values, dtype=float), 0.0)

        def copied(values):
            if values is None:
                return None
            return np.asarray(values, dtype=float).copy()

        return ConservativeState1D(
            n=masked(term.n),
            nn=masked(term.nn) if include_neutral else copied(term.nn),
            M=masked(term.M),
            Ee=masked(term.Ee),
            Ei=masked(term.Ei),
            M_n=masked(term.M_n) if include_neutral else copied(term.M_n),
            nn_a=masked(term.nn_a) if include_neutral else copied(term.nn_a),
            M_n_a=(
                masked(term.M_n_a) if include_neutral else copied(term.M_n_a)
            ),
            En=masked(term.En) if include_neutral else copied(term.En),
        )

    def floor_state_vector(self, y):
        """Apply configured density and temperature floors to a packed vector.

        The cells the R2 tracer owns are EXEMPT: their density is the exact
        integral of an affine ODE for which ``n = 0`` is a regular state, so
        clipping them up to ``ne_floor`` would inject the very particles the
        tracer exists to avoid inventing, and would make a true-vacuum initial
        condition impossible. Their plasma rows are restored from the raw
        vector after the shared floor runs, so the floor function itself is
        untouched and the flag-off path is bit-identical. The neutral rows are
        floored everywhere -- they are the background, and the tracer does not
        own them.
        """
        floored = floor_state_vector(
            y=y,
            cells=self._geometry.cells,
            floors=self._floors,
            ion_mass_g=self._ion_mass_g,
            neutral_momentum=self._neutral_momentum,
            neutral_two_zone=self._neutral_two_zone,
            neutral_annulus_momentum=self._neutral_two_momentum,
            neutral_energy=self._neutral_energy,
        )
        if not self._tracer_engaged:
            return floored
        cells = int(self._geometry.cells)
        floored = np.asarray(floored, dtype=float).copy()
        raw = np.asarray(y, dtype=float)
        passive = self._tracer_passive
        for row, name in enumerate(STATE_NAMES_1D):
            if name == "nn":
                continue
            lo = row * cells
            floored[lo:lo + cells] = np.where(
                passive, raw[lo:lo + cells], floored[lo:lo + cells]
            )
        return floored

    @staticmethod
    def _empty_floor_ledger():
        return {
            "n_particles_added": 0.0,
            "nn_particles_added": 0.0,
            "nn_a_particles_added": 0.0,
            "Ee_energy_added_erg": 0.0,
            "Ei_energy_added_erg": 0.0,
            "En_energy_added_erg": 0.0,
        }

    def _floor_additions(self, raw, floored):
        Vp = np.asarray(self._geometry.plasma_volume_cm3, dtype=float)
        Vm = np.asarray(self._geometry.neutral_volume_cm3, dtype=float)
        n_delta = np.where(
            np.asarray(raw.n) < self._floors["n"],
            np.asarray(floored.n) - np.asarray(raw.n),
            0.0,
        )
        nn_delta = np.where(
            np.asarray(raw.nn) < self._floors["nn"],
            np.asarray(floored.nn) - np.asarray(raw.nn),
            0.0,
        )
        n_safe = np.maximum(np.asarray(raw.n), self._floors["n"])
        Ee_floor = 1.5 * n_safe * self._floors["Te"] * ev_to_erg
        Ei_floor = 1.5 * n_safe * self._floors["Ti"] * ev_to_erg
        Ee_delta = np.where(
            np.asarray(raw.Ee) < Ee_floor,
            np.asarray(floored.Ee) - np.asarray(raw.Ee),
            0.0,
        )
        Ei_delta = np.where(
            np.asarray(raw.Ei) < Ei_floor,
            np.asarray(floored.Ei) - np.asarray(raw.Ei),
            0.0,
        )
        if raw.nn_a is None:
            Vnn = Vm
            nn_a_added = 0.0
        else:
            Vnn = Vp
            Vann = np.maximum(Vm - Vp, 0.0)
            nn_a_added = float(
                np.sum(
                    np.where(
                        np.asarray(raw.nn_a) < self._floors["nn"],
                        np.asarray(floored.nn_a) - np.asarray(raw.nn_a),
                        0.0,
                    )
                    * Vann
                )
            )
        if raw.En is None:
            En_added = 0.0
        else:
            # The En floor is measured against the FLOORED nn -- the same
            # quantity apply_state_floors clips against -- so a cell whose nn
            # was itself floored up books only the energy the En clip added.
            En_floor = neutral_energy_floor(
                np.maximum(np.asarray(raw.nn, dtype=float), self._floors["nn"])
            )
            En_added = float(
                np.sum(
                    np.where(
                        np.asarray(raw.En) < En_floor,
                        np.asarray(floored.En) - np.asarray(raw.En),
                        0.0,
                    )
                    * Vnn
                )
            )
        return {
            "n_particles_added": float(
                np.sum(n_delta * Vp)
            ),
            "nn_particles_added": float(
                np.sum(nn_delta * Vnn)
            ),
            "nn_a_particles_added": nn_a_added,
            "Ee_energy_added_erg": float(
                np.sum(Ee_delta * Vp)
            ),
            "Ei_energy_added_erg": float(
                np.sum(Ei_delta * Vp)
            ),
            "En_energy_added_erg": En_added,
        }

    def _floor_vector_with_ledger(self, y):
        raw = self._unpack(y)
        floored_y = self.floor_state_vector(y)
        floored = self._unpack(floored_y)
        return floored_y, self._floor_additions(raw, floored)

    def _accumulate_floor_ledger(self, additions):
        for name in self._floor_ledger:
            self._floor_ledger[name] += float(additions.get(name, 0.0))

    def _validate_raw_stage(self, y, stage):
        """Reject non-finite/negative raw candidates before floor clipping."""
        return validate_raw_stage(y, stage, self._unpack)

    def _step_cache_snapshot(self):
        return SimpleNamespace(
            cathode_x0=_copy_cache_value(self._cathode_x0),
            cathode_x0_twin=_copy_cache_value(self._cathode_x0_twin),
            cathode_beam_cross=self._cathode_beam_cross.copy(),
            cathode_solve=self._cathode_solve,
            cathode_tail_anode_I=float(self._cathode_tail_anode_I),
        )

    def _restore_step_cache(self, snapshot):
        self._cathode_x0 = _copy_cache_value(snapshot.cathode_x0)
        self._cathode_x0_twin = _copy_cache_value(snapshot.cathode_x0_twin)
        self._cathode_beam_cross = np.asarray(
            snapshot.cathode_beam_cross,
            dtype=float,
        ).copy()
        self._cathode_solve = snapshot.cathode_solve
        self._cathode_tail_anode_I = float(snapshot.cathode_tail_anode_I)

    def _attempt_step(self, dt=None, operator_split=None):
        """Return a candidate step without committing state, time, or caches."""
        if operator_split is None:
            operator_split = self._flags.get("implicit_heat_conduction")
        if self._beam_deposition_in_heat_substep and not operator_split:
            # Construction already refuses the flag without
            # implicit_heat_conduction, so the only way here is a caller
            # passing operator_split=False explicitly (run/advance_one_step).
            # Silently stepping would drop the beam deposition entirely: the
            # flag has already removed it from the explicit sum and there is no
            # substep to receive it.
            raise ValueError(
                "beam_deposition_in_heat_substep is armed but this step was "
                "asked for operator_split=False: the beam electron-energy "
                "source has been removed from the explicit operator and lives "
                "in the implicit heat substep, so a non-split step would "
                "deposit no beam power at all"
            )
        if operator_split:
            if dt is None:
                dt = self.suggest_timestep(include_heat_conduction=False).dt
        elif dt is None:
            dt = self.suggest_timestep().dt
        dt = float(dt)
        if self._dvm_rows_superseded():
            # Scope the applied DVM transfer to THIS dt, from the accepted
            # state, once per attempt: both SSPRK stages and the implicit
            # heat substep then see one frozen, floor-aware rate, and the
            # amount the step declined to carry is a single well-defined
            # number the accept path can book. A rejected attempt simply
            # re-scopes at the smaller dt.
            self._dvm_scope_step_transfer(dt)
            # Arm the booked-ionization tally for this attempt only. Every
            # RHS call the integration below makes carries the same SSPRK2
            # stage weight; see _accumulate_dvm_ion_booking.
            self._dvm_ion_stage_accum = np.zeros(
                self._geometry.cells, dtype=float
            )
            self._dvm_ion_stage_weight = 0.5 * dt
            # The B1 boundary-inflow tally rides that same weight and the
            # same attempt lifetime; see _accumulate_dvm_source_booking.
            self._dvm_source_stage_accum = self._zero_dvm_source_counts()
            if self._dvm_cathode_jet is not None:
                # B5: the cathode channel's incident-ENERGY tally, armed and
                # dropped with its count so a rejected attempt books neither.
                self._dvm_cathode_jet_energy_stage_accum = np.zeros(
                    self._geometry.cells, dtype=float
                )
            if self._dvm_anode_jet is not None:
                # B4: the anode channel's incident-ENERGY tally, on the same
                # stage weight and the same attempt lifetime.
                self._dvm_anode_jet_energy_stage_accum = np.zeros(
                    self._geometry.cells, dtype=float
                )

        if self._coverage is not None:
            # Arm the covered-only neutral-debit tally AND the coverage field's
            # growth driver for THIS attempt only, on the same SSPRK2 stage
            # weighting the DVM booking above uses. Both are dropped in the
            # ``finally`` below and carried on the attempt, so a rejected step
            # advances neither.
            self._coverage_burn_accum = np.zeros(
                self._geometry.cells, dtype=float
            )
            self._coverage_reservoir_burn_accum = np.zeros(
                self._geometry.cells, dtype=float
            )
            self._coverage_w_accum = np.zeros(
                self._geometry.cells, dtype=float
            )
            self._coverage_burn_weight = 0.5 * dt

        # R2 tracer: freeze this attempt's affine coefficients at the STEP
        # START, before any stage runs. Nothing is committed here -- the
        # coefficients ride on the attempt and only ``_tracer_apply`` installs
        # them, so a rejected attempt re-freezes at the smaller dt and leaves
        # neither the Picard cache nor the refresh count moved. Absent (None)
        # whenever the flag is off, which is the presence gate for the accept
        # path below.
        attempt_tracer = self._tracer_prepare(dt)

        starting_cache = self._step_cache_snapshot()
        attempt_floor_ledger = self._empty_floor_ledger()

        def floor_with_ledger(y):
            floored, additions = self._floor_vector_with_ledger(y)
            for name in attempt_floor_ledger:
                attempt_floor_ledger[name] += float(additions[name])
            return floored

        raw_rejection_reason = ""
        raw_rejection_detail = {}
        try:
            try:
                if (
                    not self._flags.get("Plasma")
                    or self._neutral_prebreakdown_active()
                ):
                    if self._raw_stage_validation:
                        raw_next = pack_state(
                            self._implicit_neutral_step(
                                dt=dt, apply_density_floor=False
                            )
                        )
                        self._validate_raw_stage(raw_next, "implicit_neutral")
                        y_next = floor_with_ledger(raw_next)
                    else:
                        y_next = pack_state(self._implicit_neutral_step(dt=dt))
                elif operator_split:
                    y_next = self.operator_split_step(
                        dt=dt,
                        floor_func=floor_with_ledger,
                        raw_stage_func=(
                            self._validate_raw_stage
                            if self._raw_stage_validation
                            else None
                        ),
                    )
                else:
                    y_next = ssprk2_step(
                        y0=self._y,
                        dt=dt,
                        rhs_func=self._explicit_stage_rhs(dt),
                        floor_func=floor_with_ledger,
                        time=self._time,
                        raw_stage_func=(
                            self._validate_raw_stage
                            if self._raw_stage_validation
                            else None
                        ),
                    )
            except _RawStageError as error:
                y_next = error.y
                raw_rejection_reason = error.reason
                raw_rejection_detail = error.detail
            candidate_cache = self._step_cache_snapshot()
        finally:
            self._restore_step_cache(starting_cache)
            attempt_ion_booking = self._dvm_ion_stage_accum
            self._dvm_ion_stage_accum = None
            self._dvm_ion_stage_weight = 0.0
            attempt_source_booking = self._dvm_source_stage_accum
            self._dvm_source_stage_accum = None
            attempt_jet_energy_booking = (
                self._dvm_cathode_jet_energy_stage_accum
            )
            self._dvm_cathode_jet_energy_stage_accum = None
            attempt_anode_jet_energy_booking = (
                self._dvm_anode_jet_energy_stage_accum
            )
            self._dvm_anode_jet_energy_stage_accum = None
            attempt_coverage_burn = self._coverage_burn_accum
            attempt_coverage_reservoir_burn = (
                self._coverage_reservoir_burn_accum
            )
            attempt_coverage_w = self._coverage_w_accum
            self._coverage_burn_accum = None
            self._coverage_reservoir_burn_accum = None
            self._coverage_w_accum = None
            self._coverage_burn_weight = 0.0
        return StepAttempt1D(
            y=np.asarray(y_next, dtype=float),
            dt=dt,
            operator_split=bool(operator_split),
            solver_cache=candidate_cache,
            floor_ledger=attempt_floor_ledger,
            raw_rejection_reason=raw_rejection_reason,
            raw_rejection_detail=raw_rejection_detail,
            ion_booking=attempt_ion_booking,
            source_booking=attempt_source_booking,
            cathode_jet_energy_booking=attempt_jet_energy_booking,
            anode_jet_energy_booking=attempt_anode_jet_energy_booking,
            coverage_burn=attempt_coverage_burn,
            coverage_reservoir_burn=attempt_coverage_reservoir_burn,
            coverage_w=attempt_coverage_w,
            tracer=attempt_tracer,
        )

    def _implicit_neutral_step(
        self, dt, state=None, time=None, apply_density_floor=True
    ):
        """Return a backward-Euler neutral-only state update."""
        if state is None:
            state = self.state
        if time is None:
            time = self._time
        if self._neutral_two_zone and state.nn_a is not None:
            return self._implicit_neutral_step_two_zone(
                dt=dt,
                state=state,
                time=time,
                apply_density_floor=apply_density_floor,
            )
        geometry = self._geometry
        source_kwargs = self._neutral_source_kwargs(time=time)
        # The system is TRIDIAGONAL: an axial face writes (i, i), (i, i+1),
        # (i+1, i+1) and (i+1, i) and nothing else, and the pump writes the
        # diagonal, so the only entries that ever exist lie on three bands.
        # Held in LAPACK banded storage ``ab[u + i - j, j] == a[i, j]`` with
        # (l, u) = (1, 1): assembly and solve are O(N), and the identity is
        # the middle band rather than an N x N array of zeros.
        ab = np.zeros((3, geometry.cells), dtype=float)
        ab[1, :] = 1.0
        rhs = np.asarray(state.nn, dtype=float).copy()

        coeff = np.asarray(self.neutral_exchange_coefficients(), dtype=float)
        for face, conductance in enumerate(coeff):
            left = face
            right = face + 1
            left_rate = float(conductance) / float(geometry.neutral_volume_cm3[left])
            right_rate = float(conductance) / float(
                geometry.neutral_volume_cm3[right]
            )
            ab[1, left] += dt * left_rate
            ab[0, right] -= dt * left_rate
            ab[1, right] += dt * right_rate
            ab[2, left] -= dt * right_rate

        # Anchored by role, matching neutrals.neutral_source_sink_rhs: these two
        # paths must stay consistent or the neutral-equilibration path desyncs
        # from the explicit one.
        puff_index, puff_twin_index = puff_cell_indices(geometry)
        pump_left_index, pump_right_index = pump_cell_indices(geometry)

        if source_kwargs["pump_enabled"]:
            elbow = source_kwargs["pump_elbow_conductance_lps"]
            ab[1, pump_left_index] += dt * pump_rate(
                _effective_pump_speed(
                    source_kwargs["S_pump_L"],
                    elbow if is_plenum_cell(geometry, pump_left_index) else None,
                ),
                geometry.neutral_volume_cm3[pump_left_index],
            )
            ab[1, pump_right_index] += dt * pump_rate(
                _effective_pump_speed(
                    source_kwargs["S_pump_R"],
                    elbow if is_plenum_cell(geometry, pump_right_index) else None,
                ),
                geometry.neutral_volume_cm3[pump_right_index],
            )

        if source_kwargs["gas_puff_enabled"]:
            rhs += dt * gas_puff_rate_profile(
                geometry,
                source_kwargs["S_gp"],
                source_kwargs["gas_puff_valves"],
                profile=source_kwargs["gas_puff_profile"],
                z_cm=source_kwargs["gas_puff_z_cm"],
                sigma_cm=source_kwargs["gas_puff_sigma_cm"],
                throw_cm=source_kwargs["gas_puff_throw_cm"],
                orifice_id_cm=source_kwargs["gas_puff_orifice_id_cm"],
                orifice_length_cm=source_kwargs["gas_puff_orifice_length_cm"],
                end=0,
                delivery_fraction=source_kwargs["gas_puff_delivery_fraction"],
            )
            if source_kwargs["twin_cathode"]:
                rhs += dt * gas_puff_rate_profile(
                    geometry,
                    source_kwargs["Twin_S_gp"],
                    source_kwargs["gas_puff_valves"],
                    profile=source_kwargs["gas_puff_profile"],
                    z_cm=source_kwargs["gas_puff_z_cm"],
                    sigma_cm=source_kwargs["gas_puff_sigma_cm"],
                    throw_cm=source_kwargs["gas_puff_throw_cm"],
                    orifice_id_cm=source_kwargs["gas_puff_orifice_id_cm"],
                    orifice_length_cm=source_kwargs["gas_puff_orifice_length_cm"],
                    end=-1,
                    delivery_fraction=source_kwargs["gas_puff_delivery_fraction"],
                )

        nn_next = solve_banded((1, 1), ab, rhs)
        # M_n and En pass through untouched: this step runs pre-plasma, where
        # there is no drag to drive a wind and no plasma to heat the gas.
        return ConservativeState1D(
            n=state.n.copy(),
            nn=(
                np.maximum(nn_next, self._floors["nn"])
                if apply_density_floor
                else nn_next
            ),
            M=state.M.copy(),
            Ee=state.Ee.copy(),
            Ei=state.Ei.copy(),
            M_n=None if state.M_n is None else state.M_n.copy(),
            nn_a=None if state.nn_a is None else state.nn_a.copy(),
            M_n_a=None if state.M_n_a is None else state.M_n_a.copy(),
            En=None if state.En is None else state.En.copy(),
        )

    def _two_zone_implicit_matrix(self, dt, source_kwargs):
        """Return the assembled backward-Euler matrix for one dt, BANDED.

        The 2N x 2N block system is banded once the unknowns are INTERLEAVED --
        column cell ``c`` at ``2c``, annulus cell ``c`` at ``2c + 1``. Each
        zone's axial exchange couples a cell to its axial neighbours, two
        interleaved places away; the radial zone exchange couples the two zones
        of the SAME cell, one place away; the pump is diagonal. Nothing reaches
        further, so the matrix has (l, u) = (2, 2) and is held in LAPACK banded
        storage ``ab[u + i - j, j] == a[i, j]``, shape ``(5, 2N)``. The
        block-ordered form is the same system under a permutation and is never
        assembled: it is 2N x 2N of which all but five bands are zero.

        Everything this assembly reads other than ``dt`` and the pump state is
        a RUN CONSTANT resolved once at construction -- the geometry, the zone
        volumes, the per-zone axial Knudsen conductances and the radial
        zone-exchange conductance -- so the matrix is a function of ``dt`` and
        the pump state alone, and the equilibration it serves re-derives the
        same handful of them tens of thousands of times.

        BOUND AND EVICTION, unchanged by the storage: at most
        ``TWO_ZONE_MATRIX_CACHE_ENTRIES`` entries, the oldest inserted evicted
        first, each entry now ``(5, 2N)`` floats rather than ``(2N, 2N)``. The
        cache is keyed on the EXACT BITS of ``dt`` and of the three pump
        scalars, so two dt values that differ in the last ulp key apart and
        signed zeros do not collide; the pump scalars are in the key whether or
        not the pump is enabled, which over-keys (more misses) rather than
        risking a stale entry. The assembled array is returned READ-ONLY: it is
        handed straight to ``scipy.linalg.solve_banded``, which copies rather
        than writing to its coefficient argument, and freezing it makes a
        future in-place edit raise instead of poisoning every later step that
        shares the entry.
        """
        geometry = self._geometry
        cells = geometry.cells
        V_col, V_ann = self._zone_volumes

        def bits(value):
            return None if value is None else struct.pack("<d", float(value))

        key = (
            bits(dt),
            bool(source_kwargs["pump_enabled"]),
            bits(source_kwargs["pump_elbow_conductance_lps"]),
            bits(source_kwargs["S_pump_L"]),
            bits(source_kwargs["S_pump_R"]),
        )
        cached = self._two_zone_matrix_cache.get(key)
        if cached is not None:
            return cached

        # ``a[i, j]`` lands at ``ab[2 + i - j, j]``; the identity is band 2.
        ab = np.zeros((5, 2 * cells), dtype=float)
        ab[2, :] = 1.0
        column_coeff, annulus_coeff = self._zone_axial_coeffs
        for offset, coeff, volumes in (
            (0, np.asarray(column_coeff, dtype=float), V_col),
            (1, np.asarray(annulus_coeff, dtype=float), V_ann),
        ):
            for face, conductance in enumerate(coeff):
                if conductance <= 0.0:
                    continue
                left = 2 * face + offset
                right = 2 * (face + 1) + offset
                left_rate = float(conductance) / float(volumes[face])
                right_rate = float(conductance) / float(volumes[face + 1])
                ab[2, left] += dt * left_rate
                ab[0, right] -= dt * left_rate
                ab[2, right] += dt * right_rate
                ab[4, left] -= dt * right_rate

        for cell, conductance in enumerate(
            np.asarray(self._zone_exchange_cm3_s, dtype=float)
        ):
            if conductance <= 0.0:
                continue
            col_rate = float(conductance) / float(V_col[cell])
            ann_rate = float(conductance) / float(V_ann[cell])
            ab[2, 2 * cell] += dt * col_rate
            ab[1, 2 * cell + 1] -= dt * col_rate
            ab[2, 2 * cell + 1] += dt * ann_rate
            ab[3, 2 * cell] -= dt * ann_rate

        pump_left_index, pump_right_index = pump_cell_indices(geometry)
        if source_kwargs["pump_enabled"]:
            # The pump coefficient keeps its chamber-volume normalization
            # (S/Vm) applied to BOTH zone densities: at the well-mixed
            # equilibrium the removed flux equals the single-zone S*n_port
            # exactly.
            elbow = source_kwargs["pump_elbow_conductance_lps"]
            for index, speed in (
                (pump_left_index, source_kwargs["S_pump_L"]),
                (pump_right_index, source_kwargs["S_pump_R"]),
            ):
                rate = pump_rate(
                    _effective_pump_speed(
                        speed,
                        elbow if is_plenum_cell(geometry, index) else None,
                    ),
                    geometry.neutral_volume_cm3[index],
                )
                ab[2, 2 * index] += dt * rate
                ab[2, 2 * index + 1] += dt * rate

        matrix = ab
        matrix.flags.writeable = False
        while len(self._two_zone_matrix_cache) >= TWO_ZONE_MATRIX_CACHE_ENTRIES:
            del self._two_zone_matrix_cache[
                next(iter(self._two_zone_matrix_cache))
            ]
        self._two_zone_matrix_cache[key] = matrix
        return matrix

    def _implicit_neutral_step_two_zone(
        self, dt, state, time, apply_density_floor=True
    ):
        """Backward-Euler neutral-only update on the split (nn, nn_a) system.

        The 2N x 2N block system: per-zone axial Knudsen exchange on the
        diagonal blocks, the radial zone-exchange conductance coupling the
        blocks within each cell. The puff and pumps keep their M2 (column)
        routing here -- the M3 source-routing milestone moves them -- and
        must stay consistent with ``neutral_source_sink_rhs`` exactly as
        the single-zone path's comment demands.

        The unknowns are INTERLEAVED here -- column cell ``c`` at ``2c``,
        annulus cell ``c`` at ``2c + 1`` -- because that is the ordering in
        which the system is banded; see :meth:`_two_zone_implicit_matrix`. The
        state and the returned profiles are unaffected: the interleaving lives
        entirely between this right-hand side and the solved vector.
        """
        geometry = self._geometry
        cells = geometry.cells
        source_kwargs = self._neutral_source_kwargs(time=time)
        V_col, V_ann = self._zone_volumes
        matrix = self._two_zone_implicit_matrix(dt, source_kwargs)
        rhs = np.empty(2 * cells, dtype=float)
        rhs[0::2] = np.asarray(state.nn, dtype=float)
        rhs[1::2] = np.asarray(state.nn_a, dtype=float)

        puff_index, puff_twin_index = puff_cell_indices(geometry)

        if source_kwargs["gas_puff_enabled"]:
            # Same routing as neutral_source_sink_rhs: the puff feeds the
            # annulus (re-normalized from the profile's chamber volume so
            # the inflow is conserved), falling back to the column where no
            # annulus exists.
            puff = np.zeros(cells, dtype=float)
            for end, sccm in (
                (0, source_kwargs["S_gp"]),
                (-1, source_kwargs["Twin_S_gp"]),
            ):
                if end == -1 and not source_kwargs["twin_cathode"]:
                    continue
                puff += gas_puff_rate_profile(
                    geometry,
                    sccm,
                    source_kwargs["gas_puff_valves"],
                    profile=source_kwargs["gas_puff_profile"],
                    z_cm=source_kwargs["gas_puff_z_cm"],
                    sigma_cm=source_kwargs["gas_puff_sigma_cm"],
                    throw_cm=source_kwargs["gas_puff_throw_cm"],
                    orifice_id_cm=source_kwargs["gas_puff_orifice_id_cm"],
                    orifice_length_cm=source_kwargs["gas_puff_orifice_length_cm"],
                    end=end,
                    delivery_fraction=source_kwargs["gas_puff_delivery_fraction"],
                )
            particles = puff * np.asarray(
                geometry.neutral_volume_cm3, dtype=float
            )
            into_annulus = V_ann > 0.0
            rhs[1::2] += dt * np.where(
                into_annulus, particles / np.maximum(V_ann, 1e-300), 0.0
            )
            rhs[0::2] += dt * np.where(
                into_annulus, 0.0, particles / np.maximum(V_col, 1e-300)
            )

        solution = solve_banded((2, 2), matrix, rhs)
        nn_next = np.ascontiguousarray(solution[0::2])
        nn_a_next = np.ascontiguousarray(solution[1::2])
        return ConservativeState1D(
            n=state.n.copy(),
            nn=(
                np.maximum(nn_next, self._floors["nn"])
                if apply_density_floor
                else nn_next
            ),
            M=state.M.copy(),
            Ee=state.Ee.copy(),
            Ei=state.Ei.copy(),
            En=None if state.En is None else state.En.copy(),
            M_n=None if state.M_n is None else state.M_n.copy(),
            nn_a=(
                np.maximum(nn_a_next, self._floors["nn"])
                if apply_density_floor
                else nn_a_next
            ),
            M_n_a=None if state.M_n_a is None else state.M_n_a.copy(),
        )

    def _step_rejection_info(self, attempt, y0=None):
        if y0 is None:
            y0 = self._y
        raw_reason = getattr(attempt, "raw_rejection_reason", "")
        if raw_reason:
            return (
                raw_reason,
                dict(getattr(attempt, "raw_rejection_detail", None) or {}),
            )
        y1 = np.asarray(attempt.y, dtype=float)
        packed_summary = _bad_array_summary(y1)

        try:
            state0 = self._unpack(y0)
            state1 = self._unpack(y1)
            derived1 = derive_state(state1, self._floors, self._ion_mass_g)
        except Exception as exc:
            if packed_summary is not None:
                return "nonfinite_state", {"fields": {"packed_y": packed_summary}}
            return "invalid_state", {"message": repr(exc)}

        # The CONSERVED rows are scanned only when the packed candidate itself
        # already failed the finiteness test above. unpack_state returns
        # .copy() of the rows of y1, so every one of them is a bitwise copy of
        # a value that scan has seen: a clean packed vector cannot yield a
        # non-finite n/nn/M/Ee/Ei/M_n/nn_a, and rescanning it on every ACCEPTED
        # step re-answers a question already answered. The DERIVED rows are not
        # covered by it -- they are quotients of the conserved ones -- so they
        # are scanned unconditionally. Insertion order is unchanged, so a dirty
        # packed candidate still reports exactly the dict it always did.
        fields = {}
        if packed_summary is not None:
            fields["n"] = state1.n
            fields["nn"] = state1.nn
            fields["M"] = state1.M
            fields["Ee"] = state1.Ee
            fields["Ei"] = state1.Ei
        fields["u"] = derived1.u
        fields["Te"] = derived1.Te
        fields["Ti"] = derived1.Ti
        fields["pe"] = derived1.pe
        fields["pi"] = derived1.pi
        fields["p"] = derived1.p
        if packed_summary is not None:
            if state1.M_n is not None:
                fields["M_n"] = state1.M_n
            if state1.nn_a is not None:
                fields["nn_a"] = state1.nn_a
        nonfinite_fields = {}
        for name, values in fields.items():
            summary = _bad_array_summary(values)
            if summary is not None:
                nonfinite_fields[name] = summary
        if nonfinite_fields:
            return "nonfinite_state", {"fields": nonfinite_fields}

        negative_density_fields = {}
        for name, values in (
            ("n", state1.n),
            ("nn", state1.nn),
            ("nn_a", state1.nn_a),
        ):
            if values is None:
                continue
            summary = _bad_array_summary(values, mode="negative")
            if summary is not None:
                negative_density_fields[name] = summary
        if negative_density_fields:
            return "negative_density", {"fields": negative_density_fields}

        negative_energy_fields = {}
        for name, values in (("Ee", state1.Ee), ("Ei", state1.Ei)):
            summary = _bad_array_summary(values, mode="negative")
            if summary is not None:
                negative_energy_fields[name] = summary
        if negative_energy_fields:
            return "negative_energy", {"fields": negative_energy_fields}

        density_limit = float(self._input_dict.get("max_density_step_fraction"))
        if density_limit > 0.0 and _max_relative_change(
            state0.n,
            state1.n,
            self._floors["n"],
        ) > density_limit:
            return "density_step_fraction", {}

        neutral_limit = float(self._input_dict.get("max_neutral_step_fraction"))
        if neutral_limit > 0.0 and _max_relative_change(
            state0.nn,
            state1.nn,
            self._floors["nn"],
        ) > neutral_limit:
            return "neutral_step_fraction", {}

        energy_limit = float(self._input_dict.get("max_energy_step_fraction"))
        energy_floor = (
            1.5
            * self._floors["n"]
            * min(self._floors["Te"], self._floors["Ti"])
            * ev_to_erg
        )
        if energy_limit > 0.0 and max(
            _max_relative_change(state0.Ee, state1.Ee, energy_floor),
            _max_relative_change(state0.Ei, state1.Ei, energy_floor),
        ) > energy_limit:
            return "energy_step_fraction", {}

        return "", {}

    def _step_rejection_reason(self, attempt, y0=None):
        return self._step_rejection_info(attempt, y0=y0)[0]

    def _attempt_step_with_retries(self, dt, operator_split, diag):
        dt_min = float(self._input_dict.get("dt_min"))
        max_retries = int(self._input_dict.get("max_step_retries"))
        reject_factor = DT_REJECT_FACTOR
        retries_enabled = bool(
            self._input_dict.get("adaptive_retries_enabled")
        )
        if max_retries < 0:
            raise ValueError(f"max_step_retries must be non-negative ({max_retries})")

        attempted_dt = float(dt)
        retry_count = 0
        last_reason = ""
        last_detail = {}
        accepted_rejection_reason = ""
        rejection_events = []
        y0 = self._y.copy()
        while True:
            attempt = self._attempt_step(
                dt=attempted_dt,
                operator_split=operator_split,
            )
            last_reason, last_detail = self._step_rejection_info(attempt, y0=y0)
            if not last_reason:
                return (
                    attempt,
                    retry_count,
                    accepted_rejection_reason,
                    rejection_events,
                )
            accepted_rejection_reason = last_reason
            rejection_events.append(
                {
                    "time": float(self._time),
                    "attempted_dt": float(attempted_dt),
                    "retry_index": int(retry_count),
                    "reason": last_reason,
                    "phase": getattr(diag, "phase", ""),
                    "active_constraint": getattr(diag, "active_constraint", ""),
                }
            )
            if not retries_enabled or retry_count >= max_retries:
                self._raise_timestep_rejection(
                    attempted_dt=attempted_dt,
                    retry_count=retry_count,
                    reason=last_reason,
                    rejection_detail=last_detail,
                    dt_min=dt_min,
                    diag=diag,
                )
            next_dt = attempted_dt * reject_factor
            if next_dt < dt_min:
                self._raise_timestep_rejection(
                    attempted_dt=attempted_dt,
                    retry_count=retry_count,
                    reason=last_reason,
                    rejection_detail=last_detail,
                    dt_min=dt_min,
                    diag=diag,
                )
            attempted_dt = next_dt
            retry_count += 1

    def _raise_timestep_rejection(
        self,
        attempted_dt,
        retry_count,
        reason,
        rejection_detail,
        dt_min,
        diag,
    ):
        detail_text = _rejection_detail_text(rejection_detail)
        detail_suffix = f", detail={detail_text}" if detail_text else ""
        raise TimestepRejectionError(
            "failed to accept timestep "
            f"at t={self._time:.9e} s after {retry_count} retries "
            f"(attempted_dt={attempted_dt:.9e} s, reason={reason}, "
            f"dt_min={dt_min:.9e} s{detail_suffix})",
            time=float(self._time),
            attempted_dt=float(attempted_dt),
            retry_count=int(retry_count),
            reason=reason,
            dt_min=float(dt_min),
            phase=getattr(diag, "phase", ""),
            active_constraint=getattr(diag, "active_constraint", ""),
            rejection_detail=rejection_detail,
        )

    def _cathode_phi_wf_eff(self):
        """Effective work function [eV] under cathode_surface_model; None off."""
        if self._cathode_theta is None:
            return None
        clean = float(self._input_dict["cathode_phiwf_clean_eV"])
        dirty = float(self._input_dict["phi_wf"])
        return clean + (dirty - clean) * float(self._cathode_theta)

    def _surface_effective_input_dict(self):
        """input_dict with the evolving phi_wf substituted (shared-constant rule)."""
        eff = self._cathode_phi_wf_eff()
        if eff is None:
            return self._input_dict
        return {**self._input_dict, "phi_wf": eff}

    def _accept_step_attempt(self, attempt):
        # B5: what the cathode surface owes the kinetic gas for THIS step's
        # backscatter [erg]. Zero unless the DVM cathode jet booked
        # something below, which is what keeps the surface power balance
        # untouched with the channel off.
        step_backscatter_erg = 0.0
        # B4: what the anode's collected ions delivered to the wires this
        # step [erg]. Zero unless the DVM anode jet booked something below,
        # which is what keeps the anode book absent-and-empty off the channel.
        step_anode_incident_erg = 0.0
        # Book the DVM deferred-transfer ledger FIRST: it reads the step's
        # start state and the transfer the step was scoped with, both of
        # which the lines below overwrite.
        if self._dvm_rows_superseded():
            self._dvm_book_step_transfer(attempt.dt)
            # Only an ACCEPTED attempt's ionization counts: this is the
            # booking the state vector below actually carries.
            booking = getattr(attempt, "ion_booking", None)
            if booking is not None:
                self._dvm_ion_booked = self._dvm_ion_booked + booking
            source_booking = getattr(attempt, "source_booking", None)
            if source_booking is not None:
                self._dvm_source_booked = {
                    name: self._dvm_source_booked[name] + row
                    for name, row in source_booking.items()
                }
            jet_energy_booking = getattr(
                attempt, "cathode_jet_energy_booking", None
            )
            if jet_energy_booking is not None and not (
                self._cathode_jet_censored()
            ):
                # B5: commit this step's counted incident energy to the tick
                # accumulator FIRST, so the tick that may fire later in this
                # same accept sees it, and hold the increment for the surface
                # debit below. One committed number, both books.
                #
                # ARMING CRITERION, DVM side of the conservation pair: while
                # the latch is disarmed this whole block is skipped, so the
                # tick accumulator never receives the energy (nothing is
                # launched) AND ``step_backscatter_erg`` stays at its 0.0
                # initialisation (the surface is never debited). Both sides of
                # the pair are gated by the SAME latch reading, which is what
                # keeps the cumulative identity
                #     backscatter row == sum of the ticks' birth_cathode_jet
                #                        energy + R_E * the not-yet-ticked
                #                        accumulator
                # true across an arm/disarm/re-arm sequence without a new
                # ledger row: a censored step contributes zero to both sides.
                self._dvm_cathode_jet_energy_booked = (
                    self._dvm_cathode_jet_energy_booked + jet_energy_booking
                )
                # The COUNT this step contributed, on the SAME condition and
                # from the SAME committed booking the energy came from. The
                # tick draws its directed share from this rather than from
                # the full recycle count, so numerator and denominator of the
                # per-atom launch energy cover the same steps: a censored
                # step adds to neither, and a tick that was censored
                # throughout draws a directed share of exactly zero.
                #
                # The count partner cannot be absent here: both tallies are
                # armed under the one ``_dvm_rows_superseded`` guard for the
                # same attempt, and the source tally is armed
                # unconditionally there while this energy tally is the
                # narrower of the two. Booking the energy without its count
                # would split the pair silently, so the impossible case is
                # refused rather than skipped.
                if source_booking is None:
                    raise ValueError(
                        "the cathode jet booked incident energy for this "
                        "step with no source booking to draw its count "
                        "partner from"
                    )
                self._dvm_cathode_jet_count_booked = (
                    self._dvm_cathode_jet_count_booked
                    + source_booking["cathode_face"]
                )
                step_backscatter_erg = float(
                    self._dvm_cathode_jet["R_E"]
                    * np.sum(jet_energy_booking)
                )
            anode_jet_energy_booking = getattr(
                attempt, "anode_jet_energy_booking", None
            )
            if anode_jet_energy_booking is not None:
                # B4: the same discipline anode-side -- commit to the tick
                # accumulator FIRST so a tick firing later in this same accept
                # sees it, and hold the increment for the anode book below.
                self._dvm_anode_jet_energy_booked = (
                    self._dvm_anode_jet_energy_booked
                    + anode_jet_energy_booking
                )
                step_anode_incident_erg = float(
                    np.sum(anode_jet_energy_booking)
                )
        self._restore_step_cache(attempt.solver_cache)
        self._set_state_vector(attempt.y)
        self._accumulate_floor_ledger(
            getattr(attempt, "floor_ledger", self._empty_floor_ledger())
        )
        self._time += float(attempt.dt)
        # R2 tracer: the fluid left the passive cells' plasma rows untouched
        # (their RHS was masked and the floor skipped them), so the state now
        # carries their STEP-START density and the exact affine update installs
        # the end-of-step one. Accepted steps only, and before every consumer
        # below reads the state.
        self._tracer_apply(getattr(attempt, "tracer", None))
        # Coverage closure: re-partition the accepted state's neutrals between
        # the burnt covered column and the reservoir. Accepted steps only, and
        # only ever the auxiliary deficit -- the conserved mean field above is
        # never touched by this.
        if self._coverage is not None:
            # The coverage FIELD first: the deficit equation's (1-f)/f weights
            # and its positivity floor are the end-of-step partition's, exactly
            # as v1's were (v1 read the closed form after the clock had already
            # been advanced).
            self._advance_coverage_fraction(
                attempt.dt, getattr(attempt, "coverage_w", None)
            )
            self._advance_coverage_deficit(
                attempt.dt,
                getattr(attempt, "coverage_burn", None),
                getattr(attempt, "coverage_reservoir_burn", None),
            )
        # Electrode sample smoothing: fold the newly accepted state into the
        # supply-average EMA before any accepted-state consumer reads it.
        self._update_sample_smoothing(attempt.dt)
        # K4a refresh: kinetic target solves run
        # only at ACCEPTED states -- deterministic across step retries --
        # and only once the plasma phase is live (the pre-breakdown fill
        # stays moment).
        if (
            self._kinetic is not None
            and self._flags.get("Plasma")
            and not self._neutral_prebreakdown_active()
        ):
            kin = self._kinetic
            if kin.responses is None or self._time >= kin.next_refresh_s:
                self._kinetic_refresh(self._time)
            elif self._time >= kin.next_update_s:
                # cheap continuous update; escalate to a full refresh when
                # the absorption field has moved past the tolerance (the
                # response functions' only staleness channel)
                state = self.state
                derived = self.derived
                nu_ion, nu_cx = self._kinetic_absorption_fields(
                    state, derived
                )
                scale = np.maximum(np.abs(kin.nu_ref), 1e2)
                if float(
                    np.max(np.abs(nu_ion - kin.nu_ref) / scale)
                ) > kin.refresh_tol:
                    self._kinetic_refresh(self._time)
                else:
                    self._kinetic_update_targets(
                        self._time,
                        state=state,
                        derived=derived,
                        nu_pair=(nu_ion, nu_cx),
                    )
        # K2a: the transient DVM advances on its own neutral clock, at
        # ACCEPTED states only -- never inside a trial RHS or a step retry,
        # so a rejected attempt cannot touch the distribution.
        if (
            self._dvm is not None
            and self._flags.get("Plasma")
            and not self._neutral_prebreakdown_active()
        ):
            if not self._dvm_engaged:
                self._dvm_engage()
            elif self._time >= self._dvm_next_s:
                self._dvm_advance(self._time - self._dvm_last_s)
        # Retain the accepted solve current for the measured-tail phase gate.
        solve = self._cathode_solve
        if (
            self._anode_sheath_full_debit
            and solve is not None
            and solve.beam_result is not None
        ):
            # anode_sheath_full_debit census. The flag's attracting-anode
            # branch (phi_a <= 0: the bank pays the fall, so the plasma debit
            # stays thermal-only) is loud by being COUNTED, not by printing.
            # Counted here and nowhere else, so the census measures ACCEPTED
            # steps: a rejected attempt never reaches this commit, and the
            # per-stage RHS evaluations inside one step are not steps.
            _phi_a = float(solve.beam_result.result.phi_a)
            if np.isfinite(_phi_a) and _phi_a <= 0.0:
                self._anode_attracting_steps += 1
                self._anode_attracting_last_time_s = float(self._time)
        if self._plateau_multigroup and solve is not None:
            # Multi-group plateau EDGE-CLAMP census, counted on ACCEPTED steps
            # for the same reason the anode census is: a rejected attempt
            # never reaches this commit, and the per-stage RHS evaluations
            # inside one step are not steps. Any end whose edge solve hit the
            # inelastic floor counts the step once.
            _edges = getattr(solve, "beam_plateau_edge", None) or {}
            if any(int(entry[1]) != 0 for entry in _edges.values()):
                self._plateau_edge_clamped_steps += 1
                self._plateau_edge_clamped_last_time_s = float(self._time)
        if solve is not None and solve.beam_result is not None:
            # Clamp the loop-current state to a generous physical ceiling
            # (~5x the dead-short bank current): the sheath solve has a
            # pathological huge-phi_c branch under inflated effective EMFs,
            # and an unclamped I_prev lets that branch feed a multiplicative
            # runaway (seen: I_prev -> 5e22 A). Real loop currents sit ~50x
            # below this ceiling, so the clamp is inert in normal operation.
            I_ceiling = 5.0 * float(
                self._input_dict.get("V_bank")
            ) / max(float(self._input_dict.get("R_comp")), 1e-6)
            self._circuit_I_prev = min(
                max(float(solve.beam_result.result.I_tot), 0.0),
                max(I_ceiling, 0.0),
            )
        else:
            self._circuit_I_prev = 0.0
        # Shared honest accepted-state solve for the surface updates
        # (power_balance warming and the coverage model): one re-solve at
        # the accepted state, the step's frozen I_loop, and the CURRENT
        # (pre-update) T_s / phi_wf_eff. The RHS cache is the step's last
        # internal-stage solve and must not feed state evolution (measured
        # 4.6-7.5x P_cathode_i inflation, 2026-07-21).
        honest_result = None
        if (
            solve is not None
            and solve.beam_result is not None
            and not bool(solve.metadata.get("floating", False))
            and (
                self._cathode_warming_model == "power_balance"
                or self._cathode_theta is not None
            )
        ):
            honest_result = idriven_result_evaluator(
                state=self._smoothed_sample_state(self.state),
                floors=self._floors,
                ion_mass_g=self._ion_mass_g,
                mu=self._mu,
                geometry=self._geometry,
                input_dict=self._input_dict,
                input_flags=self._effective_cathode_flags(
                    active_only=False, floating=False
                ),
                beam_cross_prev=self._cathode_beam_cross,
                T_s_override_K=self._cathode_Ts_K,
                phi_wf_override_eV=self._cathode_phi_wf_eff(),
                f_em_override=self._cathode_f_em,
            )(self._circuit_I_loop)
        # B5: the backscatter row of the surface energy ledger, booked on
        # EVERY accepted step the channel counted on -- not only on the ones
        # whose warming branch runs. What the row measures is what left WITH
        # THE ATOMS, and that happens on every such step. Whether the SURFACE
        # gives it up is a separate question with a different answer: only
        # ``power_balance`` evolves T_s, so outside it the cathode is a fixed
        # reservoir and gives up nothing -- there is no temperature for the
        # debit to come out of -- exactly as the fluid channel's retention
        # factor is inert outside ``power_balance``. Booking the row
        # regardless is what makes that gap MEASURED rather than remembered,
        # and it is what carries the cumulative identity
        #     backscatter row == sum of the ticks' birth_cathode_jet energy
        #                        + R_E * the not-yet-ticked accumulator
        # against the kinetic arm's own ledger.
        if "backscatter" in self._cathode_energy_ledger_J:
            self._cathode_energy_ledger_J["backscatter"] += (
                step_backscatter_erg * 1.0e-7
            )
        # B4: the anode's own two rows, on the same accepted-step discipline
        # and from the same committed pair. The anode has NO warming model at
        # all, so there is no branch here to be inside of and no temperature
        # for the debit to come out of: these rows are the MEASUREMENT of what
        # the wires received and what left again, which is what makes "the
        # anode absorbs 1 - R_E" a subtraction of two rows.
        if self._anode_energy_ledger_J is not None:
            self._anode_energy_ledger_J["ion_incident"] += (
                step_anode_incident_erg * 1.0e-7
            )
            self._anode_energy_ledger_J["backscatter"] += (
                float(self._dvm_anode_jet["R_E"])
                * step_anode_incident_erg
                * 1.0e-7
            )
        # Cathode warming, accepted steps only (rejected attempts never move
        # the surface temperature).
        if (
            self._cathode_Ts_K is not None
            and solve is not None
            and solve.beam_result is not None
        ):
            if self._cathode_warming_model == "power_balance":
                result = solve.beam_result.result
                # Floating phases: emitted electrons return to the surface,
                # so net evaporative cooling vanishes with the net current.
                floating = bool(solve.metadata.get("floating", False))
                # Honest accepted-state inputs (see the shared result above).
                if honest_result is not None:
                    result = honest_result
                I_emis = 0.0 if floating else float(result.I_eth_star)
                # Emission cooling books the EVOLVING work function when
                # the surface model is on (one shared constant).
                # The surface keeps (1 - R_E) of the ion power when the jet's
                # reflected-energy debit sensitivity arm is on (retention is
                # 1.0 otherwise -- the M5a' calibration convention).
                # ARMING CRITERION, fluid side of the conservation pair: while
                # the latch is disarmed no backscattered atoms were launched
                # this step, so the surface gives nothing up and keeps ALL of
                # its ion bombardment power. Read from the SAME latch state
                # the launch above consulted, so the surface can never be
                # debited for atoms that were never born.
                _retention = (
                    1.0
                    if self._cathode_jet_censored()
                    else self._cathode_surface_ion_retention
                )
                P_heat, P_ion, P_rad, P_emis, P_cond = (
                    cathode_power_balance_terms_W(
                        T_s_K=self._cathode_Ts_K,
                        P_ion_W=float(result.P_cathode_i) * _retention,
                        I_eth_star_A=I_emis,
                        input_dict=self._surface_effective_input_dict(),
                    )
                )
                C_th = float(
                    self._input_dict.get("cathode_heat_capacity_J_per_K")
                )
                # Semi-implicit in the linearized loss: dT = dt*P_net/(C_th
                # + dt*dP_loss/dT). At production dt/tau ~ 5e-5 this is the
                # explicit update to 4 decimal places; for tiny C_th it
                # cannot overshoot the radiative equilibrium and ring.
                eps = float(self._input_dict.get("cathode_emissivity"))
                area = math.pi * float(self._input_dict["R_cath"]) ** 2
                G_lin = (
                    4.0
                    * eps
                    * 5.670374419e-12
                    * float(area)
                    * self._cathode_Ts_K**3
                    + max(I_emis, 0.0) * 2.0 * 8.617333262e-5
                    + float(
                        self._input_dict.get(
                            "cathode_conduction_W_per_K"
                        )
                    )
                )
                # B5 BACKSCATTER DEBIT: the energy the R_E share left with,
                # formed from the counted (particles, incident energy) pair
                # this same accepted step committed -- NOT from a retention
                # factor on P_cathode_i. The retention form reads the
                # CIRCUIT's ion power while the backscattered atoms are
                # counted off the BOUNDARY operator's recycle row, and the
                # two are different books; taking the debit from the counted
                # pair is what makes the surface give up exactly what the
                # kinetic gas received. Erg here, watts in the balance.
                P_back = (
                    step_backscatter_erg * 1.0e-7 / float(attempt.dt)
                    if attempt.dt > 0.0
                    else 0.0
                )
                dT = (
                    float(attempt.dt)
                    * (P_heat + P_ion - P_rad - P_emis - P_cond - P_back)
                    / (C_th + float(attempt.dt) * G_lin)
                )
                self._cathode_Ts_K = max(
                    self._cathode_Ts_K + dT,
                    CATHODE_ENV_T_K,
                )
                ledger = self._cathode_energy_ledger_J
                ledger["heater"] += float(attempt.dt) * P_heat
                ledger["ion"] += float(attempt.dt) * P_ion
                ledger["rad"] += float(attempt.dt) * P_rad
                ledger["emis"] += float(attempt.dt) * P_emis
                ledger["cond"] += float(attempt.dt) * P_cond
        # Surface-state coverage, accepted steps only. Ion flux from the
        # honest accepted-state solve where available (drive phases); the
        # cached solve's I_i otherwise (floating/afterglow -- low stakes).
        # Backward-Euler is exact-form for this linear ODE: theta stays in
        # [0, 1] and cannot overshoot the ads/des equilibrium.
        if (
            self._cathode_theta is not None
            and solve is not None
            and solve.beam_result is not None
        ):
            surf_res = (
                honest_result
                if honest_result is not None
                else solve.beam_result.result
            )
            I_i_A = max(float(surf_res.I_i), 0.0)
            # 0D pairing with the solver's own I_i = A_c e n c_s: the flux
            # density is Gamma_i = I_i / (e A_c) = n c_s e^{-1/2}.
            area_cm2 = math.pi * float(self._input_dict["R_cath"]) ** 2
            Gamma_i = I_i_A / (qe_SI * area_cm2)
            sigma_cl = float(
                self._input_dict.get("cathode_cleaning_sigma_cm2")
            )
            # Energy-dependent ion-stimulated desorption yield (M5a',
            # Tom-approved with literature backing 2026-07-21): the
            # near-threshold Bohdansky factor
            #   f(E) = (1 - (E_th/E)^(2/3)) * (1 - E_th/E)^2
            # (Bohdansky 1984; Garcia-Rosales 1995 revised formulae) with
            # E the honest mean deposited energy per ion, P_cathode_i/I_i
            # [eV] -- the sheath drop plus its presheath/thermal riders,
            # from the same accepted-state solve driving the flux. He->O
            # kinematics (gamma = 0.64) puts E_th = E_B/(gamma(1-gamma))
            # ~ 18-26 eV for chemisorbed O. None = energy-independent
            # (the M5a fluence limit, bit-exact).
            E_th = self._input_dict.get("cathode_cleaning_E_th_eV")
            if E_th is not None and I_i_A > 0.0:
                E_ion_eV = max(float(surf_res.P_cathode_i), 0.0) / I_i_A
                E_th = float(E_th)
                if E_ion_eV <= E_th:
                    sigma_cl = 0.0
                else:
                    r = E_th / E_ion_eV
                    sigma_cl *= (1.0 - r ** (2.0 / 3.0)) * (1.0 - r) ** 2
            loss = sigma_cl * Gamma_i
            self._cathode_theta = self._cathode_theta / (
                1.0 + float(attempt.dt) * loss
            )
        # Emitting-area percolation, accepted steps only, and at the SAME seam
        # as the other two surface states above: the honest accepted-state
        # re-solve read the pre-update surface (pre-update T_s, phi_wf_eff and
        # f_em together), and the circuit advance below reads the post-update
        # one. The clock is autonomous -- it takes no feedback from the state --
        # so unlike the coverage field it needs no stage accumulator.
        if self._cathode_f_em is not None:
            self._advance_emitting_area_fraction(attempt.dt)
        # Current-driven circuit: advance the loop current by one TR-BDF2
        # step against V_dis(I) evaluated at the accepted end-of-step state
        # (and post-warming T_s). Accepted steps only, exactly like the
        # other circuit state. The capacitor is advanced inside the same
        # call (trapezoidal).
        step_phase = self._cathode_phase_options(
            time=self._time - float(attempt.dt)
        )
        if not step_phase["solve_enabled"] or step_phase["floating"]:
            # Open circuit: no loop; the stored inductor energy is dropped.
            self._circuit_I_loop = 0.0
            self._circuit_V_dis_step = 0.0
        else:
            C_bank_id = self._input_dict.get("C_bank_F")
            bank_off = bool(step_phase.get("inductive_tail", False))
            # Lazy first-step charge of the bank, kept exactly where it was;
            # the VALUE is then read through the one shared expression so the
            # RHS-side solve and this advance cannot disagree about V_src.
            if (
                not bank_off
                and C_bank_id is not None
                and float(C_bank_id) > 0.0
                and self._circuit_V_cap is None
            ):
                self._circuit_V_cap = float(
                    self._input_dict.get("V_bank")
                )
            V_src = self._circuit_source_voltage_V(step_phase)
            vdis = idriven_vdis_evaluator(
                state=self.state,
                floors=self._floors,
                ion_mass_g=self._ion_mass_g,
                mu=self._mu,
                geometry=self._geometry,
                input_dict=self._input_dict,
                input_flags=self._effective_cathode_flags(
                    active_only=False, floating=False
                ),
                beam_cross_prev=self._cathode_beam_cross,
                T_s_override_K=self._cathode_Ts_K,
                phi_wf_override_eV=self._cathode_phi_wf_eff(),
                f_em_override=self._cathode_f_em,
                circuit_V_src_V=V_src,
            )
            I_new, V_cap_new, V_dis_step = advance_circuit_current_driven(
                I_prev_A=self._circuit_I_loop,
                dt_s=float(attempt.dt),
                V_src_V=V_src,
                # R_external = x*R_comp: only the external partition appears in
                # V_dis = V_bank - I*R_external - L*dI/dt. The internal part and
                # R_mesh are folded into the device voltage (vdis_of_I) instead.
                # NB x cancels identically from the loop current (corrected
                # 2026-08-03; see the R_comp_partition docstring in
                # core/config.py): the x*R_comp subtracted here and the
                # (1-x)*R_comp inside vdis_of_I sum back to the total R_comp.
                # Only R_mesh genuinely lowers the current.
                R_comp_ohm=float(self._input_dict.get("R_comp"))
                * float(self._input_dict.get("R_comp_partition")),
                L_H=float(self._input_dict.get("L_parasitic_H")),
                vdis_of_I=vdis,
                C_bank_F=None if bank_off else C_bank_id,
                V_cap_prev_V=self._circuit_V_cap,
            )
            self._circuit_I_loop = I_new
            self._circuit_V_dis_step = float(V_dis_step)
            self._circuit_V_dis_time_integral += float(attempt.dt) * float(
                V_dis_step
            )
            if V_cap_new is not None:
                self._circuit_V_cap = V_cap_new
        # ARMING CRITERION, latch update. LAST in the accept path, and
        # deliberately after every consumer above. This step's jets launched
        # under the latch state the step STARTED with, and its surface debit
        # was taken under that same state; moving the latch here is what keeps
        # that pair together. The new state takes effect from the next step --
        # the first step whose RHS can actually launch under it.
        #
        # Judged on the ion current THIS step's own committed cathode solve
        # booked, which is the same solve every row above read, so the
        # criterion and the books it gates are one reading of one state.
        if self._jet_arming_active:
            if not self._jet_armed:
                self._jet_arming_censored_steps += 1
            self._update_jet_arming_latch(solve)
        # Vessel common-mode node: one closed-form step of
        # C dV_cm/dt = I_wall_net, after the circuit so the wall currents are
        # read at the fully accepted state. Absent unless armed.
        self._vessel_advance(float(attempt.dt))
        return self.get_initial_snapshot()

    def _update_jet_arming_latch(self, cathode_solve):
        """Advance the cathode-jet arming latch on one ACCEPTED step.

        Latched hysteresis on the booked ion current: it ARMS at or above
        ``neutral_jet_arm_current_A`` and DISARMS below
        ``neutral_jet_disarm_current_A``, so a current dwelling between the
        two leaves the latch wherever it already was. That band is the whole
        point -- a bare single-threshold comparison would toggle the jets on
        every step a noisy current re-crossed it.

        A step with no cathode solve, or one whose solve returned a non-finite
        ``I_i``, leaves the latch untouched: there is no current to judge it
        by, and guessing a state would be worse than holding the last one that
        was actually measured.

        Called only while the criterion is declared; the presence gate is the
        caller's.
        """
        if cathode_solve is None or cathode_solve.beam_result is None:
            return
        I_i = float(cathode_solve.beam_result.result.I_i)
        if not np.isfinite(I_i):
            return
        was_armed = self._jet_armed
        if not was_armed:
            if I_i >= self._jet_arm_current_A:
                self._jet_armed = True
        elif I_i < self._jet_disarm_current_A:
            self._jet_armed = False
        if self._jet_armed != was_armed:
            self._jet_arming_transitions += 1
            self._jet_arming_last_transition_s = float(self._time)

    # Every attribute one accepted step mutates (fluid attempt cathode cache +
    # accept), for the R5.1/A11 Picard snapshot. The persistent step cache is
    # exactly the four cathode fields (`_restore_step_cache`), so restoring them
    # restores it.
    _PICARD_DIRECT_ATTRS = (
        "_time",
        "_circuit_I_loop",
        "_circuit_I_prev",
        "_circuit_V_dis_step",
        "_circuit_V_dis_time_integral",
        "_circuit_V_cap",
        "_cathode_Ts_K",
        "_cathode_theta",
        "_cathode_f_em",
        "_cathode_solve",
        # A2a lag: a float, mutated only on the accepted-solve write, so a
        # Picard re-run must start from the value the step started from.
        "_cathode_tail_anode_I",
        # Vessel node: the potential is a float, and the ledger/current
        # records are copied below because they are mutable containers.
        "_vessel_V_cm",
        "_vessel_wall_currents_A",
        # Cathode-jet arming latch and its census: all four are written on the
        # accepted step, so a Picard re-run must start from the values the
        # step started from. Omitting the latch here would let a re-run launch
        # under a state a LATER step established.
        "_jet_armed",
        "_jet_arming_censored_steps",
        "_jet_arming_transitions",
        "_jet_arming_last_transition_s",
    )

    def _picard_snapshot(self):
        """Capture the pre-step coupled state for a Picard re-run (R5.1/A11)."""
        snap = {a: getattr(self, a) for a in self._PICARD_DIRECT_ATTRS}
        snap["_vessel_charge_ledger_C"] = dict(self._vessel_charge_ledger_C)
        snap["_y"] = self._y.copy()
        snap["_cathode_x0"] = _copy_cache_value(self._cathode_x0)
        snap["_cathode_x0_twin"] = _copy_cache_value(self._cathode_x0_twin)
        bc = self._cathode_beam_cross
        snap["_cathode_beam_cross"] = (
            None if bc is None else np.asarray(bc, dtype=float).copy()
        )
        snap["_floor_ledger"] = dict(self._floor_ledger)
        snap["_cathode_energy_ledger_J"] = dict(self._cathode_energy_ledger_J)
        ema = self._sample_ema
        snap["_sample_ema"] = (
            None if ema is None else {c: list(v) for c, v in ema.items()}
        )
        return snap

    def _picard_restore(self, snap):
        """Restore exactly the state ``_picard_snapshot`` captured."""
        for a in self._PICARD_DIRECT_ATTRS:
            setattr(self, a, snap[a])
        self._vessel_charge_ledger_C = dict(snap["_vessel_charge_ledger_C"])
        self._cathode_x0 = _copy_cache_value(snap["_cathode_x0"])
        self._cathode_x0_twin = _copy_cache_value(snap["_cathode_x0_twin"])
        bc = snap["_cathode_beam_cross"]
        self._cathode_beam_cross = (
            None if bc is None else np.asarray(bc, dtype=float).copy()
        )
        self._floor_ledger = dict(snap["_floor_ledger"])
        self._cathode_energy_ledger_J = dict(snap["_cathode_energy_ledger_J"])
        ema = snap["_sample_ema"]
        self._sample_ema = (
            None if ema is None else {c: list(v) for c, v in ema.items()}
        )
        # Restore _y EXACTLY (the snapshot is the accepted, already-floored state);
        # do NOT route through _set_state_vector, whose re-flooring is not
        # guaranteed idempotent and would make a Picard re-run start from a
        # perturbed state.
        self._y = snap["_y"].copy()
        self._state = self._unpack(self._y)
        self._derived = derive_state(
            self._state, self._floors, self._ion_mass_g
        )

    # ---- restart: export and resume -------------------------------------
    #
    # The inventory below is the executable form of _sim1d/RESTART.md, which
    # gives the mutation site of every member and the justification for each
    # deliberate omission. Members NOT here are either derivable from config
    # and geometry at construction, per-attempt scratch that is None at any
    # instant a restart can be taken, or dropped with a reason recorded there.

    #: Instance scalars/arrays carried verbatim, grouped by payload section.
    _RESTART_CATHODE_ATTRS = (
        "_cathode_x0",
        "_cathode_x0_twin",
        "_cathode_beam_cross",
        "_cathode_Ts_K",
        "_cathode_theta",
    )
    #: Cathode-jet arming latch and its census. These ride the ``cathode``
    #: group but NOT the strict loop above, because they are PRESENCE-GATED on
    #: the criterion being declared: with none declared the latch is
    #: permanently armed and the three counters permanently at their seed, so
    #: there is nothing to carry and the payload is byte-unchanged.
    _RESTART_JET_ARMING_ATTRS = (
        "_jet_armed",
        "_jet_arming_censored_steps",
        "_jet_arming_transitions",
        "_jet_arming_last_transition_s",
    )
    _RESTART_CIRCUIT_ATTRS = (
        "_circuit_I_loop",
        "_circuit_I_prev",
        "_circuit_V_cap",
        "_circuit_V_dis_step",
        "_circuit_V_dis_time_integral",
    )
    _RESTART_TRIGGER_ATTRS = (
        "_t_prebreakdown_trigger",
        "_t_breakdown_trigger",
        "_last_current_trigger_time",
        "_last_current_trigger_I_tot",
        "_t_ignition_abort",
        "_ignition_abort_reason",
        "_ignition_abort_threshold_name",
        "_run_start_for_phase_events",
    )

    def restart_payload(self):
        """Return this solver's complete evolving state as a flat mapping.

        The instant described is :attr:`time`. ``results.restart`` serialises
        the mapping; keeping the two separate means the inventory lives with
        the solver that owns the attributes, not with the file format.
        """
        cathode = {
            name: _copy_cache_value(getattr(self, name))
            for name in self._RESTART_CATHODE_ATTRS
        }
        # A2a lag, written on its own key rather than joining the strict
        # inventory above so a payload taken before the cull existed stays
        # readable: it restores to 0.0, which is what an unarmed run carries.
        cathode["_cathode_tail_anode_I"] = float(self._cathode_tail_anode_I)
        cathode["energy_ledger_J"] = None
        # Emitting-area percolation, presence-gated exactly like the vessel
        # node and the coverage section: the lit fraction is written only when
        # the closure is armed, so a payload from a run without it is
        # structurally what it always was and older payloads stay readable.
        # ``cathode_emitting_area`` is a structural flag key, so a resume that
        # changes the arming refuses rather than restoring a fraction nothing
        # reads or leaving an armed closure at its seed.
        if self._cathode_f_em is not None:
            cathode["_cathode_f_em"] = float(self._cathode_f_em)
        # Cathode-jet arming latch, presence-gated on the criterion being
        # DECLARED. ``_jet_armed`` decides whether the next step's jets launch
        # at all and whether the surface is debited for them, so a resume that
        # reset it would re-censor jets the producing run had already brought
        # into existence -- a relocation of the channel, not a perturbation of
        # it. The three census members ride with it so a resumed run reports
        # ONE run's arming history rather than two. Written on their own keys
        # rather than joining the strict inventory loop above, so a payload
        # taken before the latch was carried still loads.
        if self._jet_arming_active:
            for name in self._RESTART_JET_ARMING_ATTRS:
                cathode[name] = getattr(self, name)
        circuit = {
            name: getattr(self, name) for name in self._RESTART_CIRCUIT_ATTRS
        }
        prev_save = self._circuit_V_dis_prev_save
        circuit["V_dis_prev_save_t"] = (
            None if prev_save is None else float(prev_save[0])
        )
        circuit["V_dis_prev_save_integral"] = (
            None if prev_save is None else float(prev_save[1])
        )
        # Vessel node, presence-gated exactly like the coverage section below:
        # the potential and its charge ledger are written only when the node is
        # armed, so a payload from a run without it is structurally what it
        # always was. ``regime_vessel_node`` is a structural flag key, so a
        # resume that changes the arming refuses rather than reading half a
        # node.
        if self._vessel is not None:
            circuit["vessel_V_cm"] = float(self._vessel_V_cm)
            for key, value in self._vessel_charge_ledger_C.items():
                circuit[f"vessel_Q_{key}_C"] = float(value)
        triggers = {
            name: getattr(self, name) for name in self._RESTART_TRIGGER_ATTRS
        }
        samples = self._current_trigger_samples
        triggers["sample_time"] = np.asarray(
            [t for t, _ in samples], dtype=float
        )
        triggers["sample_I_tot"] = np.asarray(
            [I for _, I in samples], dtype=float
        )
        monitor = self._ignition_monitor
        ignition = {
            "stalled": bool(monitor.stalled),
            "sample_time": np.asarray(
                [s.time for s in monitor._samples], dtype=float
            ),
            "sample_N": np.asarray(
                [s.N for s in monitor._samples], dtype=float
            ),
            "sample_N_n": np.asarray(
                [s.N_n for s in monitor._samples], dtype=float
            ),
            "sample_Ee": np.asarray(
                [s.Ee for s in monitor._samples], dtype=float
            ),
            "sample_joint": np.asarray(
                [float(s.joint) for s in monitor._samples], dtype=float
            ),
        }
        coverage = {}
        if self._coverage is not None:
            coverage["f"] = np.asarray(self._coverage_f, dtype=float).copy()
            coverage["deficit"] = np.asarray(
                self._coverage_deficit, dtype=float
            ).copy()
        ledgers = dict(self._floor_ledger)
        ledgers.update(
            {
                f"cathode_energy_{key}_J": value
                for key, value in self._cathode_energy_ledger_J.items()
            }
        )
        if self._anode_energy_ledger_J is not None:
            # PRESENCE-GATED: off the channel these keys are absent and the
            # restart payload is byte-unchanged.
            ledgers.update(
                {
                    f"anode_energy_{key}_J": value
                    for key, value in self._anode_energy_ledger_J.items()
                }
            )
        sample_ema = {}
        if self._sample_ema is not None:
            for cell, (n_ema, Te_ema) in self._sample_ema.items():
                sample_ema[f"cell_{int(cell)}_n"] = float(n_ema)
                sample_ema[f"cell_{int(cell)}_Te"] = float(Te_ema)
        return {
            "time": float(self._time),
            "cells": int(self._geometry.cells),
            "state_fields": state_field_names(self.state),
            "y": np.asarray(self._y, dtype=float).copy(),
            "params": dict(self._input_dict),
            "flags": dict(self._flags),
            "compiled_kernels": str(KERNEL_PROVENANCE),
            "cathode": cathode,
            "circuit": circuit,
            "coverage": coverage,
            "triggers": triggers,
            "ignition": ignition,
            "ledgers": ledgers,
            "sample_ema": sample_ema,
            "run_loop": dict(self._restart_run_loop_state()),
        }

    def _restart_run_loop_state(self):
        """Return the run loop's controller state from the last ``run`` call.

        These values live in locals of :meth:`run`, not on the instance, so
        :meth:`run` deposits them here when it returns. ``previous_accepted_dt``
        anchors the dt-growth ramp and ``t_last_save`` sets the save lattice --
        and a save is not passive, because ``_trajectory_snapshot`` rewrites the
        cathode continuation cache through ``rhs_terms``. Both are therefore
        part of the trajectory, not bookkeeping.
        """
        return getattr(self, "_last_run_loop_state", None) or {
            "previous_accepted_dt": None,
            "t_last_save": None,
            "dt_growth_capped_streak": 0,
            "consecutive_dt_min_clamps": 0,
            "saved_frames": 0,
            "accepted_steps": 0,
        }

    def _load_restart_if_configured(self):
        """Replace the initial condition with a restart payload, if configured.

        Presence-gated on ``restart_from``: unset (the default) returns before
        any import, so a non-restart run neither opens a file nor touches a
        single attribute this method would otherwise overwrite.
        """
        source = self._input_dict.get("restart_from")
        if source is None:
            return
        from .results.restart import (
            check_restart_compatibility,
            load_restart_state,
            REFUSED_NEUTRAL_MODELS,
        )

        if self._flags.get("neutral_equilibration"):
            raise ValueError(
                "restart_from cannot be combined with neutral_equilibration: "
                "start_simulation() would run the puff/off accumulation and "
                "OVERWRITE the restored state. A restart payload IS the "
                "neutral seed. Clear the flag on the resuming run."
            )
        if self._neutral_model in REFUSED_NEUTRAL_MODELS:
            raise ValueError(
                f"restart_from cannot be combined with neutral_model="
                f"{self._neutral_model!r}: that arm evolves a velocity "
                "distribution which the sim1d-restart-v1 payload does not "
                "carry, so resuming would silently reseed the kinetic half "
                "from a Maxwellian and the run would not be a continuation. "
                f"Accepted neutral_model values for a restart: everything "
                f"except {list(REFUSED_NEUTRAL_MODELS)}"
            )
        payload = load_restart_state(source)
        check_restart_compatibility(
            payload,
            cells=self._geometry.cells,
            state_fields=state_field_names(self.state),
            params=self._input_dict,
            flags=self._flags,
        )
        self._apply_restart_payload(payload)

    def _apply_restart_payload(self, payload):
        """Overwrite this solver's whole evolving state with ``payload``."""
        cathode = payload["cathode"]
        for name in self._RESTART_CATHODE_ATTRS:
            setattr(self, name, _copy_cache_value(cathode[name]))
        self._cathode_tail_anode_I = float(
            cathode.get("_cathode_tail_anode_I", 0.0)
        )
        if self._cathode_f_em is not None:
            self._cathode_f_em = float(cathode["_cathode_f_em"])
        # Cathode-jet arming latch, presence-gated on THIS solver's criterion:
        # with none declared the latch is permanently armed and restoring a
        # stored one would only mislabel a run summary nothing else reads.
        #
        # A payload written before the latch was carried has no such rows, and
        # so does one from a run that declared no criterion. Rather than
        # resume an armed discharge as disarmed in silence -- which re-censors
        # the jets and moves the trajectory -- the load keeps the
        # constructor's disarmed seed and SAYS so.
        if self._jet_arming_active:
            if all(
                name in cathode for name in self._RESTART_JET_ARMING_ATTRS
            ):
                self._jet_armed = bool(cathode["_jet_armed"])
                self._jet_arming_censored_steps = int(
                    cathode["_jet_arming_censored_steps"]
                )
                self._jet_arming_transitions = int(
                    cathode["_jet_arming_transitions"]
                )
                self._jet_arming_last_transition_s = float(
                    cathode["_jet_arming_last_transition_s"]
                )
            else:
                warnings.warn(
                    "restart payload carries no cathode-jet arming latch "
                    f"({payload.get('path')}), but this run declares an arming "
                    "criterion. The payload predates the latch's carriage, or "
                    "was written by a run that declared no criterion. This "
                    "resumed run therefore starts DISARMED with an empty "
                    "census: until the booked ion current re-crosses "
                    "neutral_jet_arm_current_A the cathode jets launch nothing "
                    "and the surface is debited nothing, so the continuation "
                    "is NOT bit-identical to the unbroken run. Re-export the "
                    "handoff from a build that carries the latch to resume "
                    "armed.",
                    stacklevel=2,
                )
        circuit = payload["circuit"]
        for name in self._RESTART_CIRCUIT_ATTRS:
            setattr(self, name, circuit[name])
        prev_save_t = circuit["V_dis_prev_save_t"]
        self._circuit_V_dis_prev_save = (
            None
            if prev_save_t is None
            else (float(prev_save_t), float(circuit["V_dis_prev_save_integral"]))
        )
        if self._vessel is not None:
            self._vessel_V_cm = float(circuit["vessel_V_cm"])
            for key in self._vessel_charge_ledger_C:
                self._vessel_charge_ledger_C[key] = float(
                    circuit[f"vessel_Q_{key}_C"]
                )
        triggers = payload["triggers"]
        for name in self._RESTART_TRIGGER_ATTRS:
            setattr(self, name, triggers[name])
        self._current_trigger_samples = [
            (float(t), float(I))
            for t, I in zip(triggers["sample_time"], triggers["sample_I_tot"])
        ]
        # The abort CONTEXT is a diagnostic record of a switch-open that has
        # already happened; the reason, time and threshold name above are what
        # the wind-down reads. It is rebuilt by the next guard evaluation.
        self._ignition_abort_context = None
        self._last_ignition_record = None
        ignition = payload["ignition"]
        monitor = self._ignition_monitor
        monitor._samples = [
            _IgnitionSample(
                float(t), float(N), float(N_n), float(Ee), bool(joint)
            )
            for t, N, N_n, Ee, joint in zip(
                ignition["sample_time"],
                ignition["sample_N"],
                ignition["sample_N_n"],
                ignition["sample_Ee"],
                ignition["sample_joint"],
            )
        ]
        monitor._stalled = bool(ignition["stalled"])
        if self._coverage is not None:
            self._coverage_f = np.asarray(
                payload["coverage"]["f"], dtype=float
            ).copy()
            self._coverage_deficit = np.asarray(
                payload["coverage"]["deficit"], dtype=float
            ).copy()
        ledgers = payload["ledgers"]
        for name in self._floor_ledger:
            self._floor_ledger[name] = float(ledgers[name])
        for key in self._cathode_energy_ledger_J:
            self._cathode_energy_ledger_J[key] = float(
                ledgers[f"cathode_energy_{key}_J"]
            )
        if self._anode_energy_ledger_J is not None:
            for key in self._anode_energy_ledger_J:
                self._anode_energy_ledger_J[key] = float(
                    ledgers[f"anode_energy_{key}_J"]
                )
        if self._sample_ema is not None:
            stored = payload["sample_ema"]
            for cell in self._sample_smooth_cells:
                self._sample_ema[int(cell)] = [
                    float(stored[f"cell_{int(cell)}_n"]),
                    float(stored[f"cell_{int(cell)}_Te"]),
                ]
        # State last, and EXACTLY: the payload holds an already-floored
        # accepted state, and re-flooring it is not guaranteed idempotent
        # (the same reasoning _picard_restore records).
        self._y = np.asarray(payload["y"], dtype=float).copy()
        self._state = self._unpack(self._y)
        self._derived = derive_state(
            self._state, self._floors, self._ion_mass_g
        )
        self._time = float(payload["time"])
        run_loop = dict(payload["run_loop"])
        run_loop["resumed"] = True
        self._restart_run_loop = run_loop

    def _accept_step_with_picard(self, generate_attempt):
        """Accept one step, Picard-iterating the loop current at the knee (A11).

        ``generate_attempt() -> (attempt, extra)``; returns
        ``(accept_result, attempt, extra)``. With ``coupled_circuit_picard`` off
        this is exactly one attempt + accept. On, and in a driven phase, the step
        is re-run (<= ``circuit_picard_max_iter``) with the frozen loop current
        set to the previous iteration's result until the loop current a step
        produces matches the one it was run at (relative ``circuit_picard_tol_rel``),
        so fluid + T_s + circuit share one self-consistent ``I_loop``. Rejected
        iterations restore the snapshot exactly, mutating no accepted-step state.
        """
        if not self._coupled_circuit_picard:
            attempt, extra = generate_attempt()
            return self._accept_step_attempt(attempt), attempt, extra
        snap = self._picard_snapshot()
        t_start = self._time
        phase = self._cathode_phase_options(time=t_start)
        driven = bool(phase["solve_enabled"]) and not bool(phase["floating"])
        I_frozen = float(self._circuit_I_loop)
        last = None
        for iteration in range(self._circuit_picard_max_iter):
            if iteration > 0:
                self._picard_restore(snap)
                self._circuit_I_loop = I_frozen
                self._picard_extra_solves += 1
            attempt, extra = generate_attempt()
            result = self._accept_step_attempt(attempt)
            last = (result, attempt, extra)
            if not driven:
                break
            I_new = float(self._circuit_I_loop)
            if abs(I_new - I_frozen) <= self._circuit_picard_tol_rel * max(
                abs(I_new), 1.0
            ):
                break
            I_frozen = I_new
            if iteration == 0:
                self._picard_triggered_steps += 1
        return last

    def advance_one_step(self, dt=None, operator_split=None):
        """Advance the conservative state by one explicit or split step."""
        def _generate():
            attempt = self._attempt_step(dt=dt, operator_split=operator_split)
            reason, detail = self._step_rejection_info(attempt)
            if reason:
                raise ValueError(
                    f"step candidate rejected before acceptance: {reason}; "
                    f"{_rejection_detail_text(detail)}"
                )
            return attempt, None

        result, _attempt, _extra = self._accept_step_with_picard(_generate)
        return result

    def operator_split_step(
        self,
        y=None,
        dt=None,
        splitting=None,
        floor_func=None,
        raw_stage_func=None,
    ):
        """Return one explicit-nonheat plus implicit-heat split step.

        ``splitting`` selects how the non-heat operator A and the heat operator
        B are composed over the step, defaulting to the ``operator_splitting``
        parameter:

        ``"lie"``
            ``A(dt)`` then ``B(dt)``. First-order in dt regardless of how
            accurate either sub-integrator is, because the splitting error goes
            as dt*[A,B].
        ``"strang"``
            ``B(dt/2)``, ``A(dt)``, ``B(dt/2)``. The symmetry cancels the
            leading commutator term, leaving O(dt^2). B is the halved operator
            because it is the cheap one: two banded solves per species against
            a tridiagonal matrix, versus A's reaction-rate evaluations. So
            Strang costs one extra heat substep, not one extra explicit step.

        Second-order overall additionally needs the conduction substep itself
        to be second-order (``implicit_heat_scheme``) with a non-frozen
        conductivity (``heat_picard_iterations``); Strang alone only removes
        the splitting term.

        ``beam_deposition_in_heat_substep`` MOVES ONE TERM ACROSS THE SPLIT:
        the beam's electron-energy deposition leaves A's explicit sum (see
        :meth:`rhs`) and enters B as a source held constant over each substep
        (:meth:`beam_deposition_ee_source`). Nothing else changes -- the beam's
        particle births, ionization cost and excitation radiation stay in A --
        and the deposited power is booked once either way. Off, this method is
        the historical composition, call for call.
        """
        y0 = self._y if y is None else np.asarray(y, dtype=float)
        if dt is None:
            dt = self.suggest_timestep(
                y=y0,
                include_heat_conduction=False,
            ).dt
        if splitting is None:
            splitting = self._operator_splitting()
        else:
            splitting = validate_operator_splitting(splitting)
        if floor_func is None:
            floor_func = self.floor_state_vector
        if raw_stage_func is None and self._raw_stage_validation:
            raw_stage_func = self._validate_raw_stage

        def heat(y_in, sub_dt, source_time=None):
            if self._beam_deposition_in_heat_substep:
                state = self.implicit_heat_conduction_step(
                    dt=sub_dt,
                    y=y_in,
                    ee_source=self.beam_deposition_ee_source(
                        y=y_in, time=source_time
                    ),
                )
            else:
                state = self.implicit_heat_conduction_step(dt=sub_dt, y=y_in)
            raw = pack_state(state)
            if raw_stage_func is not None:
                raw_stage_func(raw, "implicit_heat")
            return floor_func(raw)

        def explicit(y_in, sub_dt):
            # The explicit operator spans the WHOLE step under both splittings
            # (only the cheap heat operator is halved by Strang), so ``sub_dt``
            # is the one window an explicitly time-dependent term is
            # integrated over.
            return ssprk2_step(
                y0=y_in,
                dt=sub_dt,
                rhs_func=self._explicit_stage_rhs(
                    sub_dt, include_heat_conduction=False
                ),
                floor_func=floor_func,
                time=self._time,
                raw_stage_func=raw_stage_func,
            )

        # The source times below matter only under
        # ``beam_deposition_in_heat_substep``; every other substep is
        # autonomous, which is why the unarmed path computes neither of them.
        # The rule is that a substep's source is evaluated at the state it
        # starts from and at the time THAT STATE represents, which under Strang
        # pairs (y0, t) with (post-A, t + dt) -- the same two instants operator
        # A's own SSPRK2 stages use, so the two half-substeps together form the
        # trapezoidal quadrature of the source over the step rather than a
        # left-endpoint rule.
        source_time_start = source_time_end = None
        if self._beam_deposition_in_heat_substep:
            source_time_start = self._time
            source_time_end = self._time + dt
        if splitting == "strang":
            half = 0.5 * dt
            return heat(
                explicit(heat(y0, half, source_time_start), dt),
                half,
                source_time_end,
            )
        return heat(explicit(y0, dt), dt, source_time_end)

    def _operator_splitting(self):
        return validate_operator_splitting(
            self._input_dict.get("operator_splitting")
        )

    def advance_one_step_operator_split(self, dt=None):
        """Advance by explicit non-heat terms then implicit heat conduction."""
        return self.advance_one_step(dt=dt, operator_split=True)

    def run(
        self,
        t_end=None,
        dt=None,
        operator_split=None,
        max_steps=None,
        progress_callback=None,
        progress_tracker=None,
        progress_interval_s=None,
    ):
        """Advance to ``t_end`` and return sparse saved trajectory arrays."""
        self._beam_gap_ledger_warned = False
        # Restart resume. Claimed (and cleared) once, here, so a second run()
        # on the same solver is an ordinary continuation of this one rather
        # than a second replay of the payload. ``None`` on every non-restart
        # run: that is the presence gate for each resume branch below.
        resume = self._restart_run_loop
        self._restart_run_loop = None
        if (
            self._flags.get("neutral_equilibration")
            and not self._run_via_start_simulation
        ):
            warnings.warn(
                "neutral_equilibration is ON but run() was called directly, so "
                "NO equilibration happens: only start_simulation() runs the "
                "puff/off accumulation and seeds nn from it. This run starts "
                "from the direct nn0 fill "
                f"({resolve_nn0(self._input_dict):.3g} cm^-3) "
                "instead of an equilibrated profile. Call start_simulation() "
                "for the equilibrated result, or clear the flag to silence "
                "this.",
                stacklevel=2,
            )
        explicit_t_end = t_end is not None
        if t_end is None:
            t_end = self.default_t_end()
        t_end = float(t_end)
        if t_end < self._time:
            raise ValueError(f"t_end must be >= current time ({t_end} < {self._time})")
        if max_steps is None:
            max_steps = int(self._input_dict.get("max_steps"))
        max_steps = int(max_steps)
        unlimited_steps = max_steps <= 0

        dt_save = float(self._input_dict.get("dt_save"))
        t_save_start = float(self._input_dict.get("t_save_start"))
        max_output_steps = int(self._input_dict.get("max_output_steps"))
        saved = []
        diagnostics = []
        timestep_rejection_events = []
        t_last_save = -np.inf
        previous_accepted_dt = None
        # Frames and accepted steps this run INHERITS. Zero except on a resume,
        # where the max_output_steps and accepted-step budgets must be measured
        # against the two-stage totals rather than this stage's own.
        saved_frames_before = 0
        steps_before = 0
        time_tol = max(1e-15, 1e-12 * max(abs(t_end), 1.0))
        run_start = float(self._time)
        progress_wall_start = perf_counter()
        if resume is None:
            self._run_start_for_phase_events = run_start
        else:
            # A resumed run reports phase events from the ORIGINAL origin, not
            # from the handoff instant: the carried value is the whole two-stage
            # run's start.
            run_start = float(self._run_start_for_phase_events)
        dt_growth_enabled = bool(self._input_dict.get("dt_growth_enabled"))
        dt_growth_factor = float(self._input_dict.get("dt_growth_factor"))
        if dt_growth_enabled and dt_growth_factor <= 1.0:
            raise ValueError(
                "dt_growth_factor must be > 1 when dt growth is enabled "
                f"(got {dt_growth_factor})"
            )
        dynamic_current_t_end = (
            not explicit_t_end
            and self._flags.get("Plasma")
            and self._phase_transition_mode() == "current"
        )
        # A switch-open abort shortens t_end the same way a breakdown trigger
        # does, in EITHER phase-transition mode, so an aborted run winds down
        # and stops instead of crawling to the configured end time.
        dynamic_t_end = not explicit_t_end and self._flags.get("Plasma")

        def should_save(t):
            if max_output_steps > 0 and saved_frames_before + len(saved) >= max_output_steps:
                return False
            if t + 1e-15 < t_save_start:
                return False
            if dt_save <= 0.0:
                return True
            return t - t_last_save >= dt_save - time_tol or abs(t - t_end) <= time_tol

        def next_save_time_after(t):
            if max_output_steps > 0 and saved_frames_before + len(saved) >= max_output_steps:
                return None
            if dt_save <= 0.0:
                return None
            if t + time_tol < t_save_start:
                return t_save_start
            if np.isfinite(t_last_save):
                next_save = t_last_save + dt_save
            else:
                next_save = max(t_save_start, t)
            if next_save <= t + time_tol:
                next_save = t + dt_save
            if next_save <= t_end + time_tol:
                return next_save
            return None

        def cap_step(step_dt, step_cap, candidate_dt, candidate_cap):
            candidate_dt = float(candidate_dt)
            cap_tol = max(
                1e-15,
                1e-12 * max(abs(step_dt), abs(candidate_dt), 1e-30),
            )
            if candidate_dt < step_dt:
                previous_dt = step_dt
                step_dt = candidate_dt
                if candidate_dt < previous_dt - cap_tol:
                    step_cap = candidate_cap
            return step_dt, step_cap

        # The leading save is SUPPRESSED on a resume: the producing stage
        # already saved this instant, and the save is not passive -- it issues
        # a cache-mutating solve through _trajectory_snapshot -> rhs_terms. So
        # one save and one cache write happen at the handoff instant across the
        # pair, exactly as an unsplit run does at that instant. The resumed
        # stage's trajectory therefore begins AFTER the handoff frame, which
        # stage 1's last frame already is.
        if resume is None and should_save(self._time):
            saved.append(self._trajectory_snapshot(self._time))
            t_last_save = self._time

        progress_callback = (
            self._progress_callback
            if progress_callback is None
            else progress_callback
        )
        progress_tracker = (
            self._progress_tracker if progress_tracker is None else progress_tracker
        )
        progress_interval_s = (
            self._progress_interval_s
            if progress_interval_s is None
            else progress_interval_s
        )
        progress_interval_s = max(float(progress_interval_s), 0.0)
        last_progress_time = -np.inf
        force_progress = False
        steps = 0
        max_steps_stopped = False
        consecutive_dt_min_clamps = 0
        # The floor the accepted-dt clamp signal below is measured against --
        # the same value suggest_timestep clamps its raw candidate minimum to.
        dt_min = float(self._input_dict.get("dt_min"))
        # Presence gate for the wall-clock/step-count non-ignition guards: with
        # both caps off nothing below is evaluated and no clock is read.
        ignition_budget_guards = (
            self._ignition_wall_clock_cap_s > 0.0
            or self._ignition_accepted_step_cap > 0
        )
        ignition_wall_clock_start = (
            perf_counter() if ignition_budget_guards else None
        )
        # Presence gate for the accelerated dt_growth re-approach: patience 0
        # never evaluates the branch, so the ramp stays uniformly
        # dt_growth_factor and the step sequence is unchanged.
        dt_growth_recovery_patience = self._dt_growth_recovery_patience
        dt_growth_recovery_factor = self._dt_growth_recovery_factor
        dt_growth_capped_streak = 0
        # Resume: adopt the producing run's controller state, AFTER every local
        # above has taken its fresh-run value. previous_accepted_dt anchors the
        # dt-growth ramp, t_last_save sets the save lattice (and so the cathode
        # cache writes that ride on it), and the streak is the recovery
        # hysteresis -- none is reconstructible from the state, so all three
        # must come across for the resumed steps to be the same steps.
        # ``saved_frames_before``/``steps_before`` carry the counts the
        # max_output_steps and accepted-step budgets are measured against; the
        # WALL-CLOCK budget deliberately restarts (RESTART.md records why).
        if resume is not None:
            previous_accepted_dt = resume["previous_accepted_dt"]
            if resume["t_last_save"] is not None:
                t_last_save = float(resume["t_last_save"])
            dt_growth_capped_streak = int(resume["dt_growth_capped_streak"])
            consecutive_dt_min_clamps = int(resume["consecutive_dt_min_clamps"])
            saved_frames_before = int(resume["saved_frames"])
            steps_before = int(resume["accepted_steps"])
        while self._time < t_end - time_tol:
            if not unlimited_steps and steps >= max_steps:
                if self._max_steps_action == "stop":
                    # Opt-in graceful stop: keep the partial trajectory and
                    # mark the result instead of raising.
                    max_steps_stopped = True
                    break
                raise RuntimeError(
                    f"max_steps={max_steps} reached before t_end={t_end:g} s"
                )
            diag = self.suggest_timestep(
                include_heat_conduction=not (
                    operator_split
                    if operator_split is not None
                    else self._flags.get("implicit_heat_conduction")
                )
            )
            step_dt = diag.dt if dt is None else float(dt)
            step_cap = diag.active_constraint if dt is None else "fixed_dt"
            if dt is None and dt_growth_enabled and previous_accepted_dt is not None:
                step_growth_factor = dt_growth_factor
                if (
                    dt_growth_recovery_patience > 0
                    and dt_growth_capped_streak >= dt_growth_recovery_patience
                ):
                    # Nothing physical has bound for this many steps running:
                    # the ramp is re-approaching, not tracking. Widen the
                    # ceiling the ramp imposes -- every other candidate is
                    # still in the minimum below.
                    step_growth_factor = dt_growth_recovery_factor
                step_dt, step_cap = cap_step(
                    step_dt,
                    step_cap,
                    previous_accepted_dt * step_growth_factor,
                    "dt_growth",
                )
            step_dt, step_cap = cap_step(
                step_dt,
                step_cap,
                t_end - self._time,
                "t_end",
            )
            next_phase_boundary = self.next_phase_boundary_after(
                self._time,
                t_end=t_end,
                time_tol=time_tol,
            )
            if next_phase_boundary is not None:
                step_dt, step_cap = cap_step(
                    step_dt,
                    step_cap,
                    next_phase_boundary - self._time,
                    "phase_boundary",
                )
            next_save_time = next_save_time_after(self._time)
            if dt is None and next_save_time is not None:
                step_dt, step_cap = cap_step(
                    step_dt,
                    step_cap,
                    next_save_time - self._time,
                    "save_time",
                )
            if step_dt <= 0.0:
                raise RuntimeError(f"non-positive timestep selected ({step_dt})")
            def _generate_run_attempt():
                a, rc, rr, ev = self._attempt_step_with_retries(
                    dt=step_dt,
                    operator_split=operator_split,
                    diag=diag,
                )
                return a, (rc, rr, ev)

            _result, attempt, _extra = self._accept_step_with_picard(
                _generate_run_attempt
            )
            retry_count, rejection_reason, step_rejection_events = _extra
            timestep_rejection_events.extend(step_rejection_events)
            self._update_current_phase_triggers()
            if ignition_budget_guards:
                self._check_ignition_budget_guards(
                    accepted_steps=steps_before + steps + 1,
                    wall_clock_start=ignition_wall_clock_start,
                )
            if dynamic_t_end:
                current_t_end = self._dynamic_t_end(dynamic_current_t_end)
                if current_t_end is not None and current_t_end < t_end:
                    t_end = float(current_t_end)
                    time_tol = max(1e-15, 1e-12 * max(abs(t_end), 1.0))
                    self._reset_progress_tracker(progress_tracker)
                    last_progress_time = -np.inf
                    force_progress = True
            if retry_count:
                step_cap = "retry"
            if dt_growth_recovery_patience > 0:
                # Asymmetric by design -- this IS the hysteresis. The streak
                # must be rebuilt from scratch after a single step capped by
                # anything else (a physics bound, an output cadence, or a
                # retry after a rejection), so acceleration ends the instant
                # something real binds and has to re-earn its evidence.
                if step_cap == "dt_growth":
                    dt_growth_capped_streak += 1
                else:
                    dt_growth_capped_streak = 0
            previous_accepted_dt = float(attempt.dt)
            # dt_min lock guard. Only an ADAPTIVE step can be locked: with a
            # caller-supplied dt the clamp does not set the step, so a fixed-dt
            # run cannot crawl and is not counted. Consecutiveness is the
            # discriminator -- self-releasing clamp episodes are a known-good
            # family (see scripts/dtmin_census_runlengths.txt) and must not
            # abort, while a genuine lock never releases.
            #
            # The signal is the ACCEPTED dt, not the raw candidate minimum
            # alone. ``suggest_timestep`` computes ``clamped_to_dt_min`` before
            # the output-cadence, phase-boundary, t_end and dt_growth caps and
            # before the retry ladder, any of which can carry the accepted step
            # below dt_min while no candidate asked for less than dt_min -- and
            # that accepted sub-dt_min step then anchors the growth ramp, so
            # the ramp (not a physics bound) sets every step of the grind that
            # follows and the raw flag never fires. Counting the accepted dt
            # closes that hole; the raw flag is still recorded, unchanged, as
            # its own diagnostic field.
            #
            # The second half of the test is what keeps it a CLAMP rather than
            # a step count: the raw minimum must be strictly above dt_min, so
            # the step was pushed under the floor by a CAP and by nothing else.
            # A run configured with dt_max at dt_min accepts dt_min on every
            # step by construction and is not grinding; where the raw minimum
            # is itself at or below dt_min the raw flag already owns the step.
            accepted_clamped = bool(
                dt is None
                and float(attempt.dt) <= dt_min * (1.0 + _DT_MIN_ACCEPTED_RTOL)
                and diag.dt_raw > dt_min * (1.0 + _DT_MIN_ACCEPTED_RTOL)
            )
            if dt is None and (diag.clamped_to_dt_min or accepted_clamped):
                consecutive_dt_min_clamps += 1
            else:
                consecutive_dt_min_clamps = 0
            step_diag = replace(
                diag,
                accepted_dt=float(attempt.dt),
                clamped_to_dt_min_accepted=float(accepted_clamped),
                step_cap=step_cap,
                retry_count=int(retry_count),
                rejection_reason=rejection_reason,
            )
            diagnostics.append(step_diag)
            steps += 1
            if consecutive_dt_min_clamps > self._dt_min_lock_max_steps:
                # After the step is recorded, so the diagnostics carry the
                # step whose acceptance tripped the lock.
                raise self._dt_min_lock_error(
                    diag=step_diag,
                    consecutive=consecutive_dt_min_clamps,
                )
            if should_save(self._time):
                saved.append(self._trajectory_snapshot(self._time))
                t_last_save = self._time
            progress_due = (
                force_progress
                or self._time >= t_end - time_tol
                or progress_interval_s == 0.0
                or self._time - last_progress_time >= progress_interval_s
            )
            if progress_due:
                self._emit_progress(
                    callback=progress_callback,
                    tracker=progress_tracker,
                    diag=step_diag,
                    t_end=t_end,
                    step=steps,
                    max_steps=max_steps,
                    saved_samples=len(saved),
                    wall_elapsed_s=perf_counter() - progress_wall_start,
                )
                last_progress_time = float(self._time)
                force_progress = False

        # Deposit the run loop's controller state where restart_payload() can
        # reach it. These are locals, so without this an exported end state
        # would be missing the dt-growth anchor and the save lattice entirely
        # -- the two members a naive restart cannot even find to drop.
        self._last_run_loop_state = {
            "previous_accepted_dt": previous_accepted_dt,
            "t_last_save": None if not np.isfinite(t_last_save) else float(t_last_save),
            "dt_growth_capped_streak": int(dt_growth_capped_streak),
            "consecutive_dt_min_clamps": int(consecutive_dt_min_clamps),
            "saved_frames": int(saved_frames_before + len(saved)),
            "accepted_steps": int(steps_before + steps),
        }
        result = self._trajectory_result(
            saved=saved,
            diagnostics=diagnostics,
            steps=steps,
            run_start=run_start,
            timestep_rejection_events=timestep_rejection_events,
        )
        if self._max_steps_action == "stop":
            # Only the opt-in path carries the attribute so default results
            # (and their saved HDF5 files) are byte-identical to before.
            result.run_status = (
                "max_steps_reached" if max_steps_stopped else "completed"
            )
        # R2 tracer census, from day one and on every tracer run: which
        # criterion bound, where the interface got to, and how big the term the
        # description DROPS became. Presence-gated -- a run without the flag
        # prints nothing and carries no extra result field.
        census_line = self._tracer_census_line()
        if census_line is not None:
            print(census_line)
            result.tracer_criterion_census = self._tracer_census
        self._last_result = result
        return result

    def _dt_min_lock_error(self, diag, consecutive):
        """Return the RuntimeError for a run stuck at dt_min.

        Names the true minimizing bound (not "dt_min"), WHICH of the two clamp
        signals counted (the raw candidate minimum, or an accepted step at the
        floor that no candidate asked for -- a ramp-capped grind), what the
        bound asked for, and the cell closest to the density floor: the drained
        floor-pinned cell that makes ``_negative_margin_timestep`` return
        exactly zero.
        """
        n = np.asarray(self.state.n, dtype=float)
        n_floor = float(self._floors["n"])
        cell = int(np.argmin(n - n_floor))
        signals = []
        if diag.clamped_to_dt_min:
            signals.append("raw candidate minimum below dt_min")
        if diag.clamped_to_dt_min_accepted:
            signals.append(
                "ACCEPTED step at dt_min after the caps/retries "
                f"(accepted_dt={diag.accepted_dt:.9e} s, "
                f"step_cap={diag.step_cap!r})"
            )
        return RuntimeError(
            f"dt_min lock: the timestep was clamped up to dt_min on "
            f"{consecutive} consecutive steps, exceeding "
            f"dt_min_lock_max_steps={self._dt_min_lock_max_steps}, at "
            f"t={self._time:.9e} s (phase={diag.phase!r}). Clamp signal on "
            f"the tripping step: {'; '.join(signals) or 'none'}. "
            f"The true active "
            f"constraint is {diag.active_constraint!r}, asking for "
            f"dt_raw={diag.dt_raw:.9e} s against dt_min={diag.dt:.9e} s. "
            f"Cell closest to the density floor: index {cell}, "
            f"n={n[cell]:.9e} cm^-3 against n_floor={n_floor:.9e} cm^-3. "
            "A cell sitting ON a floor while a term still drains it requests "
            "dt=0, which is a modelling breakdown and not a timestep request: "
            "fix the drain or the floor rather than shrinking dt, and raise "
            "dt_min_lock_max_steps only deliberately."
        )

    @staticmethod
    def _reset_progress_tracker(tracker):
        reset = getattr(tracker, "reset", None)
        if reset is not None:
            reset()

    def _emit_progress(
        self,
        callback,
        tracker,
        diag,
        t_end,
        step,
        max_steps,
        saved_samples,
        wall_elapsed_s,
    ):
        if callback is None and tracker is None:
            return
        fraction = 1.0 if t_end <= 0.0 else min(max(self._time / t_end, 0.0), 1.0)
        progress = SimulationProgress1D(
            fraction=float(fraction),
            time=float(self._time),
            t_end=float(t_end),
            step=int(step),
            max_steps=int(max_steps),
            accepted_dt=float(diag.accepted_dt),
            suggested_dt=float(diag.dt),
            step_cap=str(diag.step_cap),
            active_constraint=str(diag.active_constraint),
            retry_count=int(diag.retry_count),
            rejection_reason=str(diag.rejection_reason),
            phase=str(diag.phase),
            saved_samples=int(saved_samples),
            wall_elapsed_s=float(wall_elapsed_s),
            wall_remaining_s=_estimate_wall_remaining(wall_elapsed_s, fraction),
            timestep_limiters=_timestep_limiters(diag),
        )
        if tracker is not None:
            update = getattr(tracker, "update", None)
            if update is not None:
                update(progress)
            else:
                tracker(progress)
        if callback is not None:
            callback(progress.fraction)

    def run_neutral_equilibration(
        self,
        cycles=None,
        t_end=None,
        dt=None,
        max_steps=None,
        progress_callback=None,
        progress_tracker=None,
        progress_interval_s=None,
    ):
        """Run a neutral-only puff/off equilibration and return its result."""
        params, flags = self.get_config()
        flags["Plasma"] = False
        flags["cathode_coupling"] = False
        flags["neutral_equilibration"] = False
        flags["launch_plasma_after_equilibration"] = False
        # The inner sim IS the equilibration -- it must never consult the seed
        # database itself. Leaving this ON contradicts the two flags just
        # cleared, so validate_neutral_seed_cache_config would reject the inner
        # config and a database MISS would raise instead of equilibrating and
        # populating the database (the caller stores the result). The cache-
        # control flags are inert to the seed signature, so clearing this here
        # cannot change the stored entry's key or content.
        flags["use_cached_neutral_seed"] = False
        # The two DVM directed-recycle jets, cleared for the SAME reason as
        # cathode_coupling above: this pre-solve has no plasma and no cathode
        # solve, so there is no collected ion flux for either jet to split and
        # no phi_c / phi_a for it to launch against. They are additionally
        # latched to the arm current (neutral_jet_arm_current_A), which a
        # Plasma=False run never reaches, so the channels are inert here twice
        # over. Without this the construction guards that require a cathode
        # solve behind each jet refuse the INNER sim -- a guard firing on a
        # state where the thing it protects cannot happen. Clearing them
        # changes no configuration that constructed before: every config this
        # touches is one that raised, so the equilibrated seed, its cache
        # signature and every existing trajectory are bit-identical. The OUTER
        # run is untouched: both are input_dict keys, and `params` is the COPY
        # `get_config()` returned, so these two lines reach the inner sim only.
        params["neutral_kinetic_dvm_cathode_jet"] = False
        params["neutral_kinetic_dvm_anode_jet"] = False
        # The equilibration OWNS its neutral start; it must not inherit the
        # outer run's nn0. nn0 is the direct-run fill (a realistic pre-shot
        # background), whereas this inner sim accumulates the fill from
        # near-vacuum over `cycles` puff/off cycles. The 1e8 is inherited from
        # the generator of the frozen gas-puff nn0 table (retired 2026-08-27),
        # which accumulated its equilibrated values the same way from the same
        # start. Pinning it here decouples the two paths, so the direct-run
        # default can move
        # without perturbing any equilibrated run. (The 100-cycle accumulation
        # forgets its start entirely: pumping decays the initial inventory
        # below the last bit of the equilibrated nn, so 1e8 and the former
        # inherited 1e9 seed BIT-IDENTICAL profiles at the production config.)
        params["nn0"] = 1e8
        if cycles is None:
            cycles = int(params.get("neutral_equilibration_cycles", params["cycles"]))
        cycles = int(cycles)
        if cycles <= 0:
            raise ValueError(f"neutral equilibration cycles must be positive ({cycles})")
        params["cycles"] = cycles
        tau_cycle = max(float(params.get("tau_cycle", 0.0)), 0.0)
        if tau_cycle <= 0.0:
            tau_cycle = max(float(params.get("tau_discharge", 0.0)), 0.0)
        if t_end is None:
            t_end = cycles * tau_cycle
        t_end = float(t_end)
        if dt is None:
            dt = params.get("neutral_equilibration_dt", None)
        if dt is not None:
            dt = float(dt)
        params["dt_save"] = max(t_end, 0.0)
        params["t_save_start"] = 0.0
        params["max_output_steps"] = 0
        sim = LAPDSim1D(
            params,
            flags,
            progress_callback=progress_callback,
            progress_tracker=progress_tracker,
            progress_interval_s=progress_interval_s,
        )
        result = sim.run(t_end=t_end, dt=dt, max_steps=max_steps)
        summary = self._neutral_equilibration_summary(result, cycles=cycles)
        result.neutral_equilibration_summary = summary
        self._last_neutral_equilibration_result = result
        self._last_neutral_equilibration_summary = summary
        return result

    def _neutral_equilibration_summary(self, result, cycles):
        final_nn = np.asarray(result.nn[-1], dtype=float)
        return SimpleNamespace(
            cycles=int(cycles),
            tau_cycle=float(self._input_dict.get("tau_cycle")),
            final_time=float(result.final_time),
            mean_nn=float(np.mean(final_nn)),
            std_nn=float(np.std(final_nn)),
            min_nn=float(np.min(final_nn)),
            max_nn=float(np.max(final_nn)),
        )

    def _apply_neutral_equilibration_result(self, result):
        final_nn = np.asarray(result.nn[-1], dtype=float)
        state = self.state
        final_nn_a = None
        if state.nn_a is not None:
            saved_nn_a = getattr(result, "nn_a", None)
            final_nn_a = (
                np.asarray(saved_nn_a[-1], dtype=float).copy()
                if saved_nn_a is not None
                else state.nn_a.copy()
            )
        self._seed_neutral_state(final_nn, final_nn_a)

    def _seed_neutral_state(self, nn, nn_a):
        """Inject an equilibrated neutral profile into the fresh plasma IC.

        Only nn (and the annulus nn_a) come from the equilibration; n, M, Ee, Ei,
        and the neutral momenta stay at the fresh initial condition. Shared by the
        live-equilibration and cached-seed paths so both seed identically.
        """
        state = self.state
        final_nn = np.asarray(nn, dtype=float)
        if final_nn.shape[0] != state.nn.shape[0]:
            raise ValueError(
                f"neutral seed has {final_nn.shape[0]} cells, state has "
                f"{state.nn.shape[0]}"
            )
        final_nn_a = None
        if state.nn_a is not None:
            final_nn_a = (
                state.nn_a.copy() if nn_a is None
                else np.asarray(nn_a, dtype=float).copy()
            )
        seeded = ConservativeState1D(
            n=state.n.copy(),
            nn=final_nn.copy(),
            M=state.M.copy(),
            Ee=state.Ee.copy(),
            Ei=state.Ei.copy(),
            M_n=None if state.M_n is None else state.M_n.copy(),
            nn_a=final_nn_a,
            M_n_a=None if state.M_n_a is None else state.M_n_a.copy(),
            # En is REDERIVED from the seeded nn rather than carried: the
            # equilibrated gas is pre-plasma gas at the wall temperature, and
            # the incoming En belongs to the fresh IC's different nn.
            En=None if state.En is None else neutral_energy_floor(final_nn),
        )
        self._set_state_vector(pack_state(seeded))
        self._time = 0.0

    def _lookup_cached_neutral_seed(self):
        """Return ``(nn, nn_a)`` from the seed database for this config, else None.

        Non-raising database lookup keyed by the neutral-flow signature. A miss
        (new neutral-flow config) returns None so the caller equilibrates and
        stores.
        """
        from .core.neutral_seed_cache import seed_db_path, try_load_neutral_seed

        cache_dir = self._input_dict.get("neutral_seed_cache_dir")
        params, flags = self.get_config()
        path = seed_db_path(cache_dir, params, flags)
        return try_load_neutral_seed(
            path, params, flags, expected_cells=self._geometry.cells
        )

    def _store_cached_neutral_seed(self, neutral_result):
        """Store an equilibration result in the seed database (keyed by signature)."""
        import os

        from .core.neutral_seed_cache import (
            fill_rate_meta,
            save_neutral_seed,
            seed_db_path,
        )

        cache_dir = self._input_dict.get("neutral_seed_cache_dir")
        os.makedirs(str(cache_dir), exist_ok=True)
        params, flags = self.get_config()
        final_nn = np.asarray(neutral_result.nn[-1], dtype=float)
        saved_nn_a = getattr(neutral_result, "nn_a", None)
        final_nn_a = None if saved_nn_a is None else np.asarray(
            saved_nn_a[-1], dtype=float
        )
        path = seed_db_path(cache_dir, params, flags)
        save_neutral_seed(
            path, final_nn, final_nn_a, params, flags,
            meta=fill_rate_meta(params, final_nn),
        )

    def default_t_end(self):
        """Return the configured end time used by ``start_simulation`` [s]."""
        if not self._flags.get("Plasma"):
            cycles = int(self._input_dict.get("cycles"))
            if cycles <= 0:
                raise ValueError(f"cycles must be positive (got {cycles})")
            tau_cycle = max(float(self._input_dict.get("tau_cycle")), 0.0)
            if tau_cycle <= 0.0:
                tau_cycle = max(
                    float(self._input_dict.get("tau_discharge")),
                    0.0,
                )
            return float(cycles) * tau_cycle

        tau_prebreakdown = max(
            float(self._input_dict.get("tau_prebreakdown")),
            0.0,
        )
        tau_breakdown = max(float(self._input_dict.get("tau_breakdown")), 0.0)
        tau_discharge = max(float(self._input_dict.get("tau_discharge")), 0.0)
        tau_afterglow = max(float(self._input_dict.get("tau_afterglow")), 0.0)
        return (
            self._neutral_prebreakdown_duration()
            + tau_prebreakdown
            + tau_breakdown
            + tau_discharge
            + tau_afterglow
        )

    def start_simulation(
        self,
        t_end=None,
        dt=None,
        operator_split=None,
        max_steps=None,
        progress_callback=None,
        progress_tracker=None,
        progress_interval_s=None,
    ):
        """Run the solver and store the result for ``get_results``.

        This mirrors the _sim3 entry-point style while preserving ``run(...)`` as
        the direct result-returning API.
        """
        if self._flags.get("neutral_equilibration"):
            use_db = self._flags.get("use_cached_neutral_seed")
            seed = self._lookup_cached_neutral_seed() if use_db else None
            if seed is not None:
                # Database HIT: reuse the equilibrated neutral seed (bit-identical
                # to running the 100-cycle equilibration for this config).
                self._seed_neutral_state(seed[0], seed[1])
            else:
                # Live equilibration; on a database MISS (a new neutral-flow
                # config = a new fill rate), store the result so the next run at
                # this config reuses it.
                neutral_result = self.run_neutral_equilibration(
                    progress_callback=progress_callback,
                    progress_tracker=progress_tracker,
                    progress_interval_s=progress_interval_s,
                )
                if use_db:
                    self._store_cached_neutral_seed(neutral_result)
                if not self._flags.get("launch_plasma_after_equilibration"):
                    self._last_result = neutral_result
                    return
                self._apply_neutral_equilibration_result(neutral_result)

        self._run_via_start_simulation = True
        try:
            self._last_result = self.run(
                t_end=t_end,
                dt=dt,
                operator_split=operator_split,
                max_steps=max_steps,
                progress_callback=progress_callback,
                progress_tracker=progress_tracker,
                progress_interval_s=progress_interval_s,
            )
        finally:
            self._run_via_start_simulation = False
        if self._last_neutral_equilibration_result is not None:
            self._last_result.neutral_equilibration = (
                self._last_neutral_equilibration_result
            )
            self._last_result.neutral_equilibration_summary = (
                self._last_neutral_equilibration_summary
            )

    def get_results(self):
        """Return the most recent ``start_simulation``/``run`` result."""
        if self._last_result is None:
            raise RuntimeError("simulation has not been run yet")
        return self._last_result

    def get_neutral_equilibration_results(self):
        """Return the most recent optional neutral pre-equilibration result."""
        if self._last_neutral_equilibration_result is None:
            raise RuntimeError("neutral equilibration has not been run yet")
        return self._last_neutral_equilibration_result

    def get_neutral_equilibration_summary(self):
        """Return final neutral-density summary for the latest equilibration."""
        if self._last_neutral_equilibration_summary is None:
            raise RuntimeError("neutral equilibration has not been run yet")
        return self._last_neutral_equilibration_summary

    def save_result(self, path, result, params=None, flags=None):
        """Write a run result to HDF5 with this solver's config metadata."""
        from .results.io import save_result_hdf5

        if params is None or flags is None:
            config_params, config_flags = self.get_config()
            if params is None:
                params = config_params
            if flags is None:
                flags = config_flags
        return save_result_hdf5(path, result, params=params, flags=flags)

    @staticmethod
    def load_result(path):
        """Load a saved sim1d HDF5 result."""
        return load_result_hdf5(path)

    @staticmethod
    def summarize_result(result):
        """Return lightweight health diagnostics for a sim1d run result."""
        return summarize_result(result)

    def suggest_timestep(self, y=None, include_heat_conduction=None, time=None):
        """Return an explicit timestep suggestion and diagnostics.

        A pure query, with one armed exception: while
        ``surface_loss_floor_exempt_exit_rtol`` is set this call ADVANCES the
        instance's ``surface_loss`` floor-exemption latch, so an out-of-loop
        probe call moves the same run state an in-loop call does.
        """
        state = self.state if y is None else self._unpack(y)
        if time is None:
            time = self._time
        plasma_enabled = self._flags.get(
            "Plasma"
        ) and not self._neutral_prebreakdown_active(time=time)
        if include_heat_conduction is None:
            include_heat_conduction = (
                plasma_enabled
                and not self._flags.get("implicit_heat_conduction")
            )
        dt_min = float(self._input_dict.get("dt_min"))
        dt_max = float(self._input_dict.get("dt_max"))
        dvm_superseded = plasma_enabled and self._dvm_rows_superseded()
        plasma_source_rhs = None
        # The bundle's historical trigger is the raw-stage stance. An engaged
        # DVM arm needs it unconditionally: its coupling term is the largest
        # unbounded drain in the ledger, and whether it is bounded must not
        # depend on a validation switch that has nothing to do with it. The
        # widening reaches DVM-engaged runs only, so no other path moves.
        if plasma_enabled and (self._raw_stage_validation or dvm_superseded):
            plasma_source_rhs = self._plasma_source_timestep_rhs(
                state=state,
                time=time,
            )
        circuit_kwargs = self._circuit_timestep_kwargs(state=state, time=time)
        diag = suggest_timestep(
            state=state,
            floors=self._floors,
            ion_mass_g=self._ion_mass_g,
            mu=self._mu,
            geometry=self._plasma_geometry(),
            neutral_exchange_coeff_cm3_s=self.neutral_exchange_coefficients(),
            neutral_source_kwargs=self._neutral_source_kwargs(time=time),
            reaction_kwargs=self._reaction_kwargs() if plasma_enabled else None,
            energy_exchange_kwargs=(
                self._energy_exchange_kwargs() if plasma_enabled else None
            ),
            electron_cooling_kwargs=(
                self._electron_cooling_kwargs() if plasma_enabled else None
            ),
            # The engaged DVM arm zeroes these two terms' applied rows and
            # carries them in its coupling term, so their unstripped bounds
            # are phantoms and are withdrawn (K2d). The replacement is
            # bounded through plasma_source_rhs above.
            ion_charge_exchange_kwargs=(
                self._ion_charge_exchange_kwargs()
                if plasma_enabled and not dvm_superseded
                else None
            ),
            heat_conduction_kwargs=(
                self._heat_conduction_kwargs()
                if plasma_enabled and include_heat_conduction
                else None
            ),
            ion_neutral_drag_kwargs=(
                self._ion_neutral_drag_kwargs()
                if plasma_enabled and not dvm_superseded
                else None
            ),
            # Withdrawn on a neutral-only phase for the reason every plasma
            # bound is: the collision operator that drives En is not applied
            # there, and the wall sink is zeroed with it.
            neutral_energy_kwargs=(
                self._neutral_energy_timestep_kwargs() if plasma_enabled else None
            ),
            circuit_kwargs=circuit_kwargs,
            plasma_source_rhs=plasma_source_rhs,
            source_floor_exempt_rtol=self._surface_loss_floor_exempt_rtol,
            source_floor_exempt_exit_rtol=(
                self._surface_loss_floor_exempt_exit_rtol
            ),
            source_floor_exempt_latch=self._surface_loss_floor_exempt_latch,
            neutral_rows_superseded=dvm_superseded,
            cfl=float(self._input_dict.get("cfl")),
            density_dt_fraction=float(
                self._input_dict.get("density_dt_fraction")
            ),
            neutral_dt_fraction=float(
                self._input_dict.get("neutral_dt_fraction")
            ),
            circuit_dt_fraction=float(
                self._input_dict.get("circuit_dt_fraction")
            ),
            dt_min=dt_min,
            dt_max=dt_max,
            dt_global_scale=self._dt_global_scale,
            include_front=plasma_enabled and self._flags.get("front_flux"),
            alpha_front=float(self._input_dict.get("alpha_front")),
            # With the R2 tracer engaged this mask also excludes the cells the
            # tracer owns. That is the whole point of the bridge: their update
            # has no stability limit, so the floor-poisoned fractional bounds
            # they would otherwise contribute must not set the step. The
            # background is left to choose it.
            plasma_active=(
                self._plasma_active_mask()
                if self._active_plasma_topology
                else None
            ),
            active_plasma_topology=self._active_plasma_topology,
            wave_speed=self._hyperbolic_wave_speed,
        )
        if not plasma_enabled:
            neutral_candidates = {
                "neutral_exchange": diag.dt_neutral_exchange,
                "neutral_sources": diag.dt_neutral_sources,
                # The loop is not a fluid row: a neutral-only phase can still
                # carry a live current-driven circuit, so its bound survives
                # the rebuild rather than being silently dropped. It is inf
                # unless the bundle was built, so no unarmed run moves.
                "circuit": diag.dt_circuit,
                "dt_max": diag.dt_max,
            }
            active_constraint, raw_dt = min(
                neutral_candidates.items(),
                key=lambda item: item[1],
            )
            neutral_dt = min(max(raw_dt, dt_min), dt_max)
            # Same honest-labeling rule as suggest_timestep(): the constraint
            # name stays the bound that minimized; the clamp is its own fact.
            neutral_clamped = neutral_dt == dt_min and raw_dt < dt_min
            # This branch REBUILDS the final dt from the surviving candidates,
            # discarding the one suggest_timestep() already scaled, so the
            # global refinement factor is re-applied at this second final-dt
            # site too. Without it a neutral-only phase would silently run
            # unrefined and the instrument would not be global.
            neutral_dt = apply_dt_global_scale(neutral_dt, self._dt_global_scale)
            diag = replace(
                diag,
                dt=float(neutral_dt),
                dt_plasma_cfl=np.inf,
                dt_front_density=np.inf,
                dt_surface_loss=np.inf,
                dt_reactions=np.inf,
                dt_energy_exchange=np.inf,
                dt_electron_cooling=np.inf,
                dt_ion_charge_exchange=np.inf,
                dt_heat_conduction=np.inf,
                dt_ion_neutral_drag=np.inf,
                dt_neutral_energy=np.inf,
                active_constraint=active_constraint,
                clamped_to_dt_min=float(neutral_clamped),
                dt_raw=float(raw_dt),
            )
        phase, _ = self._phase_info(time)
        switches = self._phase_switches(phase)
        return replace(
            diag,
            time=float(time),
            phase=phase,
            phase_cathode_enabled=float(switches["cathode_enabled"]),
            phase_gas_puff_enabled=float(switches["gas_puff_enabled"]),
            phase_floating=float(switches["floating"]),
        )

    def _plasma_source_timestep_rhs(self, state, time):
        """Return the resolved electrode/source bundle used by its dt bound.

        The bundle is exactly the set of non-flux terms that can drive a cell
        into a floor within one step, and it must be the set THIS STANCE
        RUNS -- the characteristic ghost-cell flux, the only plasma-terminating
        operator since the legacy volumetric absorber was retired 2026-08-31
        (Tom). Reading the wrong operator would bound a term the step never
        applies while leaving the applied one unbounded (the same
        wrong-operator class the recycle channel was fixed for).

        The engaged DVM arm's coupling term joins the bundle for the same
        reason: it is a volumetric ion momentum/energy source of unbounded
        magnitude, frozen between neutral ticks, and until K2d it sat outside
        every timestep bound -- injecting a 1e12 erg/cm^3/s drain changed the
        suggested dt by exactly zero. The rate bounded here is the tick's
        BOOKED transfer, not the step's floor-limited application: this bound
        is the honest question "how big a step can carry what the kinetic
        side booked", and the limiter is the separate backstop for when the
        answer is below ``dt_min``.

        The directed hot surface carrier joins the bundle on the same ground,
        and it joins TOGETHER WITH its withholding: the boundary term is
        evaluated with the carrier's launch channel armed, so the rows bounded
        here are the rows the step applies rather than the v1 rows the step no
        longer books. Its own bound is not circumstantial. The beam's
        deposition frequency is a FLUX times a cross section,
        ``nu ~ (F/A) sigma_cx (g/v_fast)``, with no plasma density in it, so it
        does not fall away with the plasma; on a long arm the afterglow
        stretches ``dt`` at the same time as the launch collapses, and an
        unbounded ion momentum/energy source in that window is exactly the
        configuration that has bitten this bundle before.
        """
        cathode_phase = self._cathode_phase_options(time=time)
        cathode_solve = None
        if cathode_phase["solve_enabled"]:
            cathode_solve = self.solve_cathode_boundary(
                state=state,
                floating=cathode_phase["floating"],
                time=time,
                update_cache=False,
            )
        boundary = self.characteristic_boundary_rhs
        carrier_out = {} if self._cathode_jet_carrier else None
        rhs = boundary(
            state=state,
            cathode_solve=cathode_solve,
            time=time,
            carrier_out=carrier_out,
        )
        if self._cathode_jet_carrier:
            # The beam reads the SAME per-neutral ionization frequency the
            # bulk reaction term uses, exactly as the applied step does, so
            # the bounded rows and the applied rows cannot describe different
            # attenuation.
            reaction = self.reaction_rhs_terms(state=state)
            rhs = add_state_rhs(
                rhs,
                self.cathode_jet_hot_carrier_rhs(
                    state=state,
                    cathode_solve=cathode_solve,
                    launch_per_s=carrier_out.get("launch_per_s"),
                    ionization_rate=np.asarray(
                        reaction["ionization_birth"].n, dtype=float
                    )
                    / np.maximum(
                        np.asarray(state.nn, dtype=float), self._floors["nn"]
                    ),
                    cache_diagnostics=False,
                ),
            )
        rhs = add_state_rhs(
            rhs,
            self.anode_collection_rhs(
                state=state,
                cathode_solve=cathode_solve,
                time=time,
            ),
        )
        # Both electrode rows, from ONE solve: the bound is over the applied
        # row, and after the per-electrode split the applied electrode row is
        # the pair. Their supports are disjoint wherever an anode is resolved,
        # so adding them separately reproduces the pre-split single add exactly.
        _electrode_terms = self.cathode_source_terms(
            state=state,
            cathode_solve=cathode_solve,
            time=time,
        )
        rhs = add_state_rhs(rhs, _electrode_terms.rhs)
        rhs = add_state_rhs(rhs, _electrode_terms.anode_rhs)
        if self._beam_ionization_birth_timestep_bound:
            # The WHOLE applied row, per the applied-row convention: a bound
            # computed from a fraction of a row describes a term the step does
            # not apply, and leaves the remainder unbounded -- the same
            # wrong-operator class as reading the wrong boundary operator
            # above. beam_ionization_birth is a volumetric plasma source that
            # can drive a cell into a floor within one step and has never been
            # in any bound.
            rhs = add_state_rhs(
                rhs,
                self.beam_ionization_rhs_terms(
                    state=state,
                    cathode_solve=cathode_solve,
                    time=time,
                )["beam_ionization_birth"],
            )
        if self._dvm_rows_superseded():
            rhs = add_state_rhs(rhs, self._dvm_booked_transfer_rhs())
        return rhs

    def phase_at_time(self, time):
        """Return the diagnostic runtime phase label for an absolute time [s]."""
        return self._phase_info(float(time))[0]

    def phase_switches_at_time(self, time):
        """Return diagnostic phase switches without changing RHS behavior."""
        phase = self.phase_at_time(time)
        return self._phase_switches(phase)

    def _neutral_prebreakdown_duration(self):
        if not (
            self._flags.get("Plasma")
            and self._flags.get("neutral_prebreakdown")
        ):
            return 0.0
        return max(float(self._input_dict.get("tau_neutral_prebreakdown")), 0.0)

    def _neutral_prebreakdown_active(self, time=None):
        if time is None:
            time = self._time
        duration = self._neutral_prebreakdown_duration()
        return duration > 0.0 and float(time) < duration

    def _plasma_phase_time_origin(self):
        return self._neutral_prebreakdown_duration()

    def _equilibration_puff_on_duration(self):
        """Per-cycle gas-puff ON window [s] of the neutral-equilibration sim.

        Only the ``Plasma=False`` equilibration inner sim reads this; the main
        run's puff is closed by its own waveform envelope, not by this window.

        ``equilibration_gas_puff_on_s`` unset (``None``) falls back to
        ``tau_discharge`` -- the historical double duty, kept bit-exact.
        """
        override = self._input_dict.get("equilibration_gas_puff_on_s")
        if override is None:
            return max(float(self._input_dict.get("tau_discharge")), 0.0)
        return float(override)

    def _equilibration_puff_on_reason(self):
        """Name of the quantity that closes the equilibration puff window."""
        if self._input_dict.get("equilibration_gas_puff_on_s") is None:
            return "tau_discharge"
        return "equilibration_gas_puff_on_s"

    def _equilibration_cycle_position(self, time):
        """Return ``(cycle_index, cycle_time)`` on the equilibration puff lattice.

        The SINGLE source of truth for where an absolute time sits inside the
        neutral-equilibration puff/off cycle. Both ``_phase_info`` (which phase
        is this?) and ``next_phase_boundary_after`` (which instant does the run
        loop step to?) read it, so the two cannot disagree about the puff-off
        instant.

        A ``time`` that sits a hair BELOW a lattice point -- the ordinary result
        of stepping exactly onto a boundary in floating point -- is snapped UP
        onto it.

        Item 37 (nn-IC diagnostician, 2026-07-27): before this was shared,
        ``next_phase_boundary_after`` dropped the puff-off boundary whenever it
        fell inside the run loop's ``time_tol``, while the untolerated modulo in
        ``_phase_info`` still read "puff" -- so the puff stayed ON for one whole
        extra step and the equilibration over-fuelled relative to its configured
        duty (measured at the default schedule, dt=1e-2, 100 cycles: +12.0% for
        tau_discharge=20 ms, +41.0% for 0.0195, +105.3% for 0.0075). The
        tolerance below is relative to the lattice and the elapsed time, and is
        deliberately NOT the run loop's ``t_end``-scaled ``time_tol``: both
        readers must apply the SAME rule, and that rule must not depend on how
        long the run happens to be.
        """
        time = max(float(time), 0.0)
        tau_cycle = max(float(self._input_dict.get("tau_cycle")), 0.0)
        puff_on = self._equilibration_puff_on_duration()
        tol = 1e-12 * max(time, tau_cycle, puff_on)
        if tau_cycle <= 0.0:
            cycle_index = 0.0
            cycle_time = time
        else:
            cycle_index = float(np.floor(time / tau_cycle))
            cycle_time = time - cycle_index * tau_cycle
            if cycle_time < 0.0:
                # the division rounded up onto the next cycle
                cycle_index -= 1.0
                cycle_time = time - cycle_index * tau_cycle
            if cycle_time >= tau_cycle - tol:
                # a hair below the cycle end IS the next cycle's start
                cycle_index += 1.0
                cycle_time = 0.0
        if 0.0 < puff_on and cycle_time < puff_on <= cycle_time + tol:
            cycle_time = puff_on
        return cycle_index, cycle_time

    def next_phase_boundary_after(self, time, t_end=None, time_tol=0.0):
        """Return the next diagnostic phase boundary after ``time`` [s]."""
        time = max(float(time), 0.0)
        time_tol = max(float(time_tol), 0.0)
        t_end = None if t_end is None else float(t_end)

        def in_run_window(boundary):
            if boundary <= time + time_tol:
                return False
            return t_end is None or boundary <= t_end + time_tol

        if not self._flags.get("Plasma"):
            # Equilibration lattice: read the cycle position from the SHARED
            # helper and compare boundaries against that snapped position. The
            # run loop's t_end-scaled ``time_tol`` governs only the t_end
            # window here -- it must NOT decide the puff-off instant, because
            # ``_phase_info`` cannot see it (the puff-duty defect fixed
            # 2026-07-29; see ``_equilibration_cycle_position``).
            def in_end_window(boundary):
                return t_end is None or boundary <= t_end + time_tol

            puff_on = self._equilibration_puff_on_duration()
            tau_cycle = max(float(self._input_dict.get("tau_cycle")), 0.0)
            cycle_index, cycle_time = self._equilibration_cycle_position(time)
            if tau_cycle <= 0.0:
                # One puff window, never repeated: its close is the only
                # boundary, and only while the puff is still open.
                if cycle_time < puff_on and in_end_window(puff_on):
                    return float(puff_on)
                return None

            cycle_start = cycle_index * tau_cycle
            snapped = cycle_start + cycle_time
            cycle_end = cycle_start + tau_cycle
            boundaries = []
            if 0.0 < puff_on < tau_cycle:
                boundaries.append(cycle_start + puff_on)
            boundaries.append(cycle_end)
            if 0.0 < puff_on < tau_cycle:
                boundaries.append(cycle_end + puff_on)
            boundaries.append(cycle_end + tau_cycle)
            for boundary in boundaries:
                if boundary > snapped and in_end_window(boundary):
                    return float(boundary)
            return None

        tau_prebreakdown = max(
            float(self._input_dict.get("tau_prebreakdown")),
            0.0,
        )
        tau_breakdown = max(float(self._input_dict.get("tau_breakdown")), 0.0)
        tau_discharge = max(float(self._input_dict.get("tau_discharge")), 0.0)
        tau_afterglow = max(float(self._input_dict.get("tau_afterglow")), 0.0)
        plasma_origin = self._plasma_phase_time_origin()
        boundaries = []
        if plasma_origin > 0.0:
            boundaries.append(plasma_origin)
        if self._t_ignition_abort is not None:
            # After a switch-open abort the scheduled/current boundaries no
            # longer apply: only the wind-down remains.
            boundaries.extend(
                [
                    float(self._t_ignition_abort),
                    float(self._t_ignition_abort) + tau_afterglow,
                ]
            )
            for boundary in sorted(boundaries):
                if in_run_window(boundary):
                    return float(boundary)
            return None
        if self._phase_transition_mode() == "current":
            main_start = self._t_breakdown_trigger
            if main_start is None:
                boundaries.append(plasma_origin + tau_prebreakdown)
            else:
                boundaries.extend(
                    [
                        main_start + tau_discharge,
                        main_start + tau_discharge + tau_afterglow,
                    ]
                )
        else:
            breakdown_start = plasma_origin + tau_prebreakdown
            main_start = breakdown_start + tau_breakdown
            boundaries.extend(
                [
                    breakdown_start,
                    main_start,
                    main_start + tau_discharge,
                    main_start + tau_discharge + tau_afterglow,
                ]
            )
        gas_event = self._gas_puff_event_time()
        if gas_event is not None:
            boundaries.append(gas_event)
        # A square probe waveform has two hard edges and nothing smooths them,
        # so they are captured the same way the puff's are: land a step
        # boundary on each, and no accepted step straddles one.
        #
        # This is NOT what makes the delivered inventory exact -- the stages
        # consume the waveform's exact step average, so the inventory is right
        # whether or not a step straddles an edge, on any lattice. What the
        # capture buys is the applied RATE: a step that straddles an edge
        # applies a partial-window average across its whole width, which
        # smears the edge in the plasma's RESPONSE (never in the delivered
        # total). Landing on the edges keeps the applied waveform the square
        # that was asked for, and it costs nothing.
        boundaries.extend(self._neutral_probe_event_times())
        for boundary in sorted(boundaries):
            if in_run_window(boundary):
                return float(boundary)
        return None

    def plasma_flux_rhs(self, y=None, include_front=None):
        """Return the conservative plasma flux RHS for inspection/testing."""
        state = self.state if y is None else self._unpack(y)
        use_front = self._flags.get("front_flux")
        if include_front is not None:
            use_front = include_front
        return plasma_flux_rhs(
            state=state,
            floors=self._floors,
            ion_mass_g=self._ion_mass_g,
            mu=self._mu,
            geometry=self._plasma_geometry(),
            include_front=use_front,
            alpha_front=float(self._input_dict.get("alpha_front")),
            active_plasma_topology=self._active_plasma_topology,
            wave_speed=self._hyperbolic_wave_speed,
            energy_consistent=self._hyperbolic_energy_consistent,
        )

    def plasma_flux_rhs_terms(self, y=None, state=None, include_front=None):
        """Return split conservative plasma face-flux RHS terms."""
        if state is None:
            state = self.state if y is None else self._unpack(y)
        use_front = self._flags.get("front_flux")
        if include_front is not None:
            use_front = include_front
        return plasma_flux_rhs_terms(
            state=state,
            floors=self._floors,
            ion_mass_g=self._ion_mass_g,
            mu=self._mu,
            geometry=self._plasma_geometry(),
            include_front=use_front,
            alpha_front=float(self._input_dict.get("alpha_front")),
            active_plasma_topology=self._active_plasma_topology,
            wave_speed=self._hyperbolic_wave_speed,
            energy_consistent=self._hyperbolic_energy_consistent,
        )

    def cathode_jet_neutral_energy_rhs(
        self, state, cathode_solve, recycle_nn_row
    ):
        """Return the ``En`` the cathode jet's recycled atoms actually carry.

        THE COMPOSITION, stated. The jet's MOMENTUM booking is untouched: the
        directed ``v_mix = R_N v_back + (1 - R_N) v_eff`` per particle stays
        inside ``characteristic_boundary_rhs``, where it rides the same term that
        rebirths the particles. Nothing about it is repeated here, so no
        momentum is booked twice. What this term adds is the ENERGY that
        booking has always dropped -- ``characteristic_boundary_rhs`` says so in as
        many words ("the reflected atoms' kinetic energy beyond the mean-flow
        momentum is NOT booked -- neutrals carry no energy field") -- and that
        the ``En`` field now has somewhere to put:

            e_jet = R_N (1/2) m v_back^2 + (1 - R_N) (3/2) k T_s

        with ``v_back`` from
        :func:`~cablp.solvers._sim1d.physics.sources.cathode_jet_backscatter_speed`
        -- the same one spec the momentum booking reads, so the energy here and
        the momentum there describe atoms moving at one speed.

        The generic surface booking has already credited every recycled atom
        the wall energy ``(3/2) k T_wall``, so this term supplies only the
        EXCESS ``e_jet - (3/2) k T_wall``, on the cathode cells alone.

        Because that energy comes off the surface, the surface must stop
        keeping it: the ``cathode_jet_surface_debit`` arm is what removes
        ``R_E`` of the ion bombardment power from the cathode's balance, and
        construction REFUSES this combination without it rather than booking
        the same ``R_E`` on both sides.

        WHICH ``R_E`` LEAVES depends on ``cathode_jet_energy_convention``, and
        only one of the two settings matches that debit. Under
        ``"total_reflected"`` the backscatter carries ``R_E (phi_c + Ti)`` per
        RECYCLED particle, exactly the per-particle share the debit takes off
        the surface. Under ``"legacy"`` it carries ``R_N R_E (phi_c + Ti)``,
        so the ``(1 - R_N) R_E`` remainder is debited from the surface and
        received by nobody.

        The atoms are NOT routed through the hot channel's ballistic kernel.
        That kernel integrates an isotropic volume birth; the backscattered
        flux is a directed surface jet, and its premise does not hold.

        DISCLOSED CONSEQUENCE, and it is not small. Those atoms are hot-class
        -- ``(1/2) m v_back^2`` is tens of eV per particle -- but with no
        kernel to carry them they are booked into the COLD channel at the one
        cathode-adjacent cell. On a short flag-on run with the jet armed
        (R_N = R_E = 0.5) that cell's ``Tn`` reaches ~11 eV, against ~0.5 eV
        everywhere else. The energy is conserved and the cell is the right
        one, but a cold-channel temperature that large is a bookkeeping
        location, not a physical bulk temperature, and any pressure or rate the
        cold fluid derives from it there should be read with that in mind.

        THE WAY OUT IS ``cathode_jet_hot_carrier``, and when it is armed this
        term stops describing the backscatter share at all. The ``R_N`` atoms
        and their ``(1/2) m v_back^2`` leave with the carrier -- the boundary
        term has already withheld them from ``recycle_nn_row`` -- so what
        remains here is the implanted ``1 - R_N`` share alone, desorbing at
        the surface temperature: ``e_jet = (3/2) k T_s`` per REBIRTHED atom.
        The row is still the excess over the wall credit the generic surface
        booking granted, on the same cathode cells; it is only the population
        it describes that shrinks.
        """
        zeros = self._zero_rhs_state()
        if state.En is None or cathode_solve is None:
            return zeros
        spec = self._cathode_jet_spec(cathode_solve)
        if spec is None:
            return zeros
        derived = derive_state(
            state, floors=self._floors, ion_mass_g=self._ion_mass_g
        )
        roles = np.asarray(self._geometry.cell_role)
        cathode = roles == "cathode"
        if not np.any(cathode):
            return zeros
        R_N = float(spec["R_N"])
        if self._cathode_jet_carrier:
            # The carrier owns the R_N backscatter share; ``recycle_nn_row``
            # already counts the implanted remainder alone, so each atom it
            # carries desorbs at the surface temperature.
            e_jet = 1.5 * kb_cgs * max(float(spec["T_s_K"]), 0.0)
        else:
            v_back = cathode_jet_backscatter_speed(
                spec, derived.Ti, self._ion_mass_g
            )
            e_jet = R_N * 0.5 * self._ion_mass_g * v_back**2 + (1.0 - R_N) * (
                1.5 * kb_cgs * max(float(spec["T_s_K"]), 0.0)
            )
        excess = e_jet - 1.5 * kb_cgs * NEUTRAL_ENERGY_FLOOR_T_K
        recycle = np.asarray(recycle_nn_row, dtype=float)
        return ConservativeState1D(
            n=zeros.n,
            nn=zeros.nn,
            M=zeros.M,
            Ee=zeros.Ee,
            Ei=zeros.Ei,
            M_n=zeros.M_n,
            nn_a=zeros.nn_a,
            M_n_a=zeros.M_n_a,
            En=np.where(cathode, np.maximum(recycle, 0.0) * excess, 0.0),
        )

    def pressure_work_rhs(self, y=None, state=None):
        """Return conservative pressure-work energy sources."""
        if state is None:
            state = self.state if y is None else self._unpack(y)
        return pressure_work_rhs(
            state=state,
            floors=self._floors,
            ion_mass_g=self._ion_mass_g,
            geometry=self._plasma_geometry(),
            # b_pressure_work_elec/ions removed as config knobs (R5 stance flip):
            # must be 1 for conservative pressure-work booking (hardwired).
            electron_scale=1.0,
            ion_scale=1.0,
            active_plasma_topology=self._active_plasma_topology,
        )

    def hyperbolic_energy_correction_rhs(self, y=None, state=None):
        """Return the R2 KEP correction as ``(pressure, dissipation)``.

        The first folds into the ``pressure_work`` ledger row and the second
        IS the ``hyperbolic_dissipation_heating`` row; see
        :func:`sources.hyperbolic_energy_correction_rhs`.
        """
        if state is None:
            state = self.state if y is None else self._unpack(y)
        return hyperbolic_energy_correction_rhs(
            state=state,
            floors=self._floors,
            ion_mass_g=self._ion_mass_g,
            mu=self._mu,
            geometry=self._plasma_geometry(),
            wave_speed=self._hyperbolic_wave_speed,
            active_plasma_topology=self._active_plasma_topology,
            electron_scale=1.0,  # b_pressure_work_elec removed (hardwired 1.0)
            ion_scale=1.0,       # b_pressure_work_ions removed (hardwired 1.0)
        )

    def flux_tube_geometry_rhs(self, y=None, state=None):
        """Return the variable-area quasi-1D momentum-pressure source."""
        if state is None:
            state = self.state if y is None else self._unpack(y)
        return flux_tube_geometry_rhs(
            state=state,
            floors=self._floors,
            ion_mass_g=self._ion_mass_g,
            geometry=self._geometry,
        )

    def _jet_cathode_solve(self, cathode_solve, jet_enabled, time):
        """Return the solve the jet terms ride, re-solving like the other
        cathode consumers only when a jet needs it and none was passed."""
        if cathode_solve is not None or not jet_enabled:
            return cathode_solve
        cathode_flags = self._effective_cathode_flags(time=time, active_only=True)
        if not cathode_flags.get("cathode_coupling", False):
            return None
        return self.solve_cathode_boundary(
            state=self.state,
            time=time,
            update_cache=True,
        )

    def electron_drift_transport_rhs(self, state, cathode_solve):
        """Return the electron drift-transport and EMF-work term.

        Presence-gated on ``self._electron_drift``: unarmed, this method is
        never called and ``Gamma_d`` is never evaluated. Armed but with no
        cathode solve carrying a current -- the pre-drive phases, and any
        afterglow frame whose solve did not converge -- the operator books
        exactly zero, because with no current there is no drift.

        The two currents come from the SAME solve, and therefore carry the
        same lag as every other row built on it: the cathode solve is
        evaluated against the loop current the last ACCEPTED step committed
        (``_circuit_I_loop``), frozen across the step and part of the solve's
        own memo key. Reading both from one object is what keeps ``I_beam``
        from being subtracted off a different ``I_tot`` than the one the
        electrode rows are proportional to.
        """
        if cathode_solve is None or cathode_solve.beam_result is None:
            self._electron_drift_rows = None
            return self._zero_rhs_state()
        result = cathode_solve.beam_result.result
        I_tot_A = float(result.I_tot)
        I_beam_A = float(result.I_eth_star)
        if not (np.isfinite(I_tot_A) and np.isfinite(I_beam_A)):
            # A solve that did not resolve its currents cannot say what the
            # drift is; booking a guess would be worse than booking nothing.
            self._electron_drift_rows = None
            return self._zero_rhs_state()
        rhs, rows = electron_drift_transport_rhs(
            state=state,
            floors=self._floors,
            ion_mass_g=self._ion_mass_g,
            geometry=self._plasma_geometry(),
            spec=self._electron_drift,
            I_tot_A=I_tot_A,
            I_beam_A=I_beam_A,
        )
        self._electron_drift_rows = rows
        return rhs

    def _cathode_jet_censored(self):
        """True when the arming criterion is declared and currently DISARMED.

        The ONE reading of the latch that every consumer takes: both cathode
        launch channels (fluid and DVM) and the cathode surface debit. Reading
        it in one place is what makes it impossible for a launch and its debit
        to disagree about whether the jet existed on a given step.

        Always ``False`` when no criterion is declared, which is the presence
        gate: the inert default cannot reach the latch.
        """
        return self._jet_arming_active and not self._jet_armed

    def _cathode_jet_spec(self, cathode_solve):
        """Return the cathode-jet parameters from the current solve, or None.

        ``None`` while the arming criterion is declared and disarmed: below
        the arming current the jet is booked as NOT YET IN EXISTENCE rather
        than launched and then censored downstream.
        """
        if (
            not self._cathode_jet_enabled
            or cathode_solve is None
            or cathode_solve.beam_result is None
            or self._cathode_jet_censored()
        ):
            return None
        phi_c = float(cathode_solve.beam_result.result.phi_c)
        if not np.isfinite(phi_c):
            return None
        T_s = float(
            self._cathode_Ts_K
            if self._cathode_Ts_K is not None
            else float(self._input_dict.get("T_s"))
        )
        return {
            "R_N": self._cathode_jet_R_N,
            "R_E": self._cathode_jet_R_E,
            "phi_c_V": max(phi_c, 0.0),
            "T_s_K": max(T_s, 0.0),
            "energy_convention": self._cathode_jet_energy_convention,
        }

    def _anode_jet_spec(self, cathode_solve):
        """Return the anode-jet parameters from the current solve, or None."""
        if (
            not self._anode_jet_enabled
            or cathode_solve is None
            or cathode_solve.beam_result is None
        ):
            return None
        phi_a = float(cathode_solve.beam_result.result.phi_a)
        if not np.isfinite(phi_a):
            return None
        return {
            "R_N": self._anode_jet_R_N,
            "R_E": self._anode_jet_R_E,
            "phi_a_V": phi_a,
            "energy_convention": self._anode_jet_energy_convention,
        }

    def _end_recycle_annulus_volume(self):
        """Return the annulus volume the end recycle routes into, or ``None``.

        ``None`` whenever ``end_recycle_to_annulus`` is off, which is what
        keeps the boundary terms on their historical column-return path.
        """
        if not self._end_recycle_to_annulus:
            return None
        return self._zone_volumes[1]

    def cathode_jet_hot_carrier_rhs(
        self,
        state,
        cathode_solve,
        launch_per_s,
        ionization_rate=None,
        cache_diagnostics=True,
    ):
        """Return the directed hot surface carrier's flows, caching its ledger.

        The backscatter share of the cathode recycle, carried down the column
        as an algebraic attenuation profile instead of being dumped cold into
        the cathode cell -- the defect ``cathode_jet_neutral_energy_rhs``'s
        docstring names. See
        :mod:`~cablp.solvers._sim1d.physics.jet_carrier`.

        ``launch_per_s`` is the per-cell rate the boundary term withheld for
        this carrier on this same evaluation; ``ionization_rate`` is the
        per-neutral ionization frequency the bulk reaction term is using, so
        the beam and the bulk cannot disagree about it. The named ledger rows
        land on ``self._jet_carrier_diagnostics`` as a side channel, exactly
        as the hot channel's diagnostics do -- they are a reading of the term,
        not a row of it, and nothing about them is saved.

        ``cache_diagnostics=False`` suppresses that side channel, for the same
        reason the timestep bundle re-solves the cathode with
        ``update_cache=False``: a dt PROBE must not leave the ledger reading
        an evaluation the step never accepted.
        """
        zeros = self._zero_rhs_state()
        if state.En is None or launch_per_s is None:
            return zeros
        spec = self._cathode_jet_spec(cathode_solve)
        if spec is None:
            return zeros
        rhs, diagnostics = cathode_jet_carrier_rhs(
            state=state,
            floors=self._floors,
            ion_mass_g=self._ion_mass_g,
            geometry=self._geometry,
            cathode_jet=spec,
            launch_per_s=launch_per_s,
            ionization_rate_per_neutral=(
                np.zeros_like(np.asarray(state.nn, dtype=float))
                if ionization_rate is None
                else ionization_rate
            ),
            I_ion=self._I_ion,
            eta=float(self._input_dict.get("eta")),
        )
        if cache_diagnostics:
            self._jet_carrier_diagnostics = diagnostics
        return rhs

    def characteristic_boundary_rhs(
        self, y=None, state=None, cathode_solve=None, time=None,
        carrier_out=None,
    ):
        """Return the characteristic ghost-cell Bohm outflow (audit A1/A16).

        THE plasma-terminating boundary operator, and the only one since the
        legacy volumetric absorber was retired 2026-08-31 (Tom): a one-sided
        ghost-cell KEP/Rusanov flux against the Bohm outflow state at each
        absorbing face. Reads the surface kwargs and cathode jet, and follows
        the interior's momentum-flux form and wave speed so the boundary and
        the interior stay consistent.

        ``carrier_out`` is the directed hot surface carrier's launch channel;
        ``None`` is the historical call and is unchanged bit for bit.
        """
        if state is None:
            state = self.state if y is None else self._unpack(y)
        surface_kwargs = self._surface_loss_kwargs()
        cathode_solve = self._jet_cathode_solve(
            cathode_solve, self._cathode_jet_enabled, time
        )
        return characteristic_boundary_rhs(
            state=state,
            floors=self._floors,
            ion_mass_g=self._ion_mass_g,
            mu=self._mu,
            geometry=self._plasma_geometry(),
            alpha_isat=surface_kwargs["alpha_isat"],
            b_surface_loss=surface_kwargs["b_surface_loss"],
            b_presheath_length=float(
                self._input_dict.get("b_presheath_length")
            ),
            gas_type=self._gas_type,
            cathode_jet=self._cathode_jet_spec(cathode_solve),
            wave_speed=self._hyperbolic_wave_speed,
            energy_consistent=self._hyperbolic_energy_consistent,
            end_recycle_annulus_volume_cm3=(
                self._end_recycle_annulus_volume()
            ),
            cathode_carrier_out=carrier_out,
        )

    def anode_collection_rhs(
        self, y=None, state=None, cathode_solve=None, time=None
    ):
        """Return the Bohm-flux plasma collection at the anode mesh."""
        if state is None:
            state = self.state if y is None else self._unpack(y)
        cathode_solve = self._jet_cathode_solve(
            cathode_solve, self._anode_jet_enabled, time
        )
        return anode_collection_rhs(
            state=state,
            floors=self._floors,
            ion_mass_g=self._ion_mass_g,
            mu=self._mu,
            geometry=self._plasma_geometry(),
            eta=float(self._input_dict.get("eta")),
            anode_jet=self._anode_jet_spec(cathode_solve),
            sheath_edge_energy=self._anode_sheath_full_debit,
        )

    def ion_neutral_drag_rhs(self, y=None, state=None):
        """Return the conservative ion-neutral drag momentum exchange."""
        if state is None:
            state = self.state if y is None else self._unpack(y)
        if self._ion_neutral_moment_closure:
            # Replaced by the moment-closed ion_neutral_collision term.
            return self._zero_rhs_state()
        return ion_neutral_drag_rhs(
            state=state,
            floors=self._floors,
            ion_mass_g=self._ion_mass_g,
            geometry=self._geometry,
            wind_column_factor=self._wind_column_factor,
            **self._ion_neutral_drag_kwargs(),
            **self._slip_closure_kwargs(),
        )

    def neutral_momentum_wall_rhs(self, y=None, state=None):
        """Return the neutral-wind wall-accommodation momentum sink."""
        if state is None:
            state = self.state if y is None else self._unpack(y)
        return neutral_momentum_wall_rhs(
            state=state,
            floors=self._floors,
            ion_mass_g=self._ion_mass_g,
            Rm_cm=self._geometry.Rm_cm,
            Tn_fit=float(self._input_dict.get("Tn_fit")),
            wall_rate_1_s=self._wind_wall_rate,
        )

    def neutral_energy_wall_rhs(self, y=None, state=None):
        """Return the neutral-energy wall-accommodation sink.

        Active only where the state carries ``En`` (the ``neutral_energy``
        flag); a strict zero otherwise. It shares the momentum wall sink's
        free-molecular GEOMETRY but not its temperature: the visit rate here is
        built from the wall's own 300 K thermal speed rather than the momentum
        closure's 0.1 eV ``Tn_fit``, because the gas that trades energy with a
        surface is the near-wall gas the v1 cut holds at ``T_wall``. The
        two-zone effective rate is the same closure re-evaluated at that speed,
        which it is linear in.
        """
        if state is None:
            state = self.state if y is None else self._unpack(y)
        return neutral_energy_wall_rhs(
            state=state,
            floors=self._floors,
            ion_mass_g=self._ion_mass_g,
            geometry=self._geometry,
            Rm_cm=self._geometry.Rm_cm,
            alpha_E=self._neutral_energy_alpha,
            Tn_fit=self._neutral_energy_wall_Tn_eV,
            wall_rate_1_s=self._neutral_energy_wall_rate,
        )

    def neutral_momentum_two_zone_rhs(self, y=None, state=None):
        """Return kinetic-derived column/annulus momentum coupling."""
        if state is None:
            state = self.state if y is None else self._unpack(y)
        return neutral_momentum_two_zone_rhs(
            state=state,
            floors=self._floors,
            ion_mass_g=self._ion_mass_g,
            geometry=self._geometry,
            Tn_K=float(self._input_dict.get("Tn_K")),
            sigma_hehe_cm2=self._neutral_wall_partition_sigma,
        )

    def neutral_wind_advection_rhs(self, y=None, state=None):
        """Return the neutral gas's axial transport under the wind.

        EXACTLY ONE advection operator runs, and this method is the only place
        that chooses which. Without ``En`` it is the historical first-order
        donor-cell upwind of ``nn`` and ``M_n``. With ``En`` the Rusanov
        mini-flux SUPERSEDES it and carries all three rows plus the cold gas's
        pressure -- the donor-cell term is not called at all, so nothing is
        advected twice. The ledger name is unchanged either way, so the saved
        ``rhs_terms`` structure does not move with the flag.
        """
        if state is None:
            state = self.state if y is None else self._unpack(y)
        if state.En is not None:
            return neutral_fluid_flux_rhs(
                state=state,
                floors=self._floors,
                ion_mass_g=self._ion_mass_g,
                geometry=self._geometry,
                mesh_faces=self._mesh_faces,
                mesh_blocked_area_cm2=self._mesh_blocked_area_cm2,
            )
        return neutral_wind_advection_rhs(
            state=state,
            floors=self._floors,
            ion_mass_g=self._ion_mass_g,
            geometry=self._geometry,
            mesh_faces=self._mesh_faces,
            mesh_blocked_area_cm2=self._mesh_blocked_area_cm2,
        )

    def neutral_cx_channel_rhs(self, y=None, state=None):
        """Return the charge-exchange decoupling correction on the cold gas."""
        if state is None:
            state = self.state if y is None else self._unpack(y)
        return neutral_cx_channel_rhs(
            state=state,
            floors=self._floors,
            ion_mass_g=self._ion_mass_g,
            geometry=self._geometry,
            wind_column_factor=self._wind_column_factor,
            **self._collision_operator_kwargs(),
        )

    def neutral_hot_channel_rhs(self, y=None, state=None, ionization_rate=None):
        """Return the hot channel's ballistic flows, caching its diagnostics.

        ``ionization_rate`` is the per-neutral ionization frequency the bulk
        reaction term is using on this same evaluation, threaded in so the
        in-flight and bulk channels cannot disagree about the rate. The
        per-cell diagnostics (``nn_hot``, ``f_hot``, ``tau_hot``) land on
        ``self._hot_channel_diagnostics`` as a side channel, exactly as the
        coverage reservoir debit does -- they are a reading of the term, not a
        row of it.
        """
        if state is None:
            state = self.state if y is None else self._unpack(y)
        if state.En is None:
            return self._zero_rhs_state()
        rhs, diagnostics = neutral_hot_channel_rhs(
            state=state,
            floors=self._floors,
            ion_mass_g=self._ion_mass_g,
            geometry=self._geometry,
            kernels=self._hot_neutral_kernels,
            I_ion=self._I_ion,
            ionization_rate_per_neutral=(
                np.zeros_like(state.nn)
                if ionization_rate is None
                else ionization_rate
            ),
            wind_column_factor=self._wind_column_factor,
            birth_drift=self._neutral_hot_birth_drift,
            internal_wall=self._neutral_hot_internal_wall,
            **self._collision_operator_kwargs(),
        )
        self._hot_channel_diagnostics = diagnostics
        return rhs

    def ion_neutral_frictional_heating_rhs(self, y=None, state=None):
        """Return the elastic ion-neutral frictional-heating energy source."""
        if state is None:
            state = self.state if y is None else self._unpack(y)
        if self._ion_neutral_moment_closure:
            # Replaced by the moment-closed ion_neutral_collision term.
            return self._zero_rhs_state()
        return ion_neutral_frictional_heating_rhs(
            state=state,
            floors=self._floors,
            ion_mass_g=self._ion_mass_g,
            wind_column_factor=self._wind_column_factor,
            geometry=self._geometry,
            **self._ion_neutral_drag_kwargs(),
            **self._slip_closure_kwargs(),
        )

    def ion_neutral_thermalization_rhs(self, y=None, state=None):
        """Return the elastic ion-neutral thermal-equilibration energy source."""
        if state is None:
            state = self.state if y is None else self._unpack(y)
        if self._ion_neutral_moment_closure:
            # Replaced by the moment-closed ion_neutral_collision term.
            return self._zero_rhs_state()
        if not self._flags.get("ion_neutral_thermalization"):
            return self._zero_rhs_state()
        b_thermalization = self._input_dict.get("b_ion_neutral_thermalization")
        return ion_neutral_thermalization_rhs(
            state=state,
            floors=self._floors,
            ion_mass_g=self._ion_mass_g,
            Tn_fit=float(self._input_dict.get("Tn_fit")),
            b_ion_neutral_thermalization=(
                None if b_thermalization is None else float(b_thermalization)
            ),
            **self._ion_neutral_drag_kwargs(),
        )

    def ion_neutral_collision_rhs(self, y=None, state=None):
        """Return the R4.3 moment-closed reduced ion-neutral collision operator.

        Active only under the ``ion_neutral_moment_closure`` flag; a strict no-op
        otherwise (the four legacy ion-neutral terms carry the physics then). A8:
        the single cold-gas neutral temperature is ``Tn_K`` (300 K feed/wall),
        converted to eV, used for both ``(Tn-Ti)`` and ``T_eff=(Ti+Tn)/2`` -- the
        legacy ``Tn_fit`` is not consulted on this path.
        """
        if state is None:
            state = self.state if y is None else self._unpack(y)
        if not self._ion_neutral_moment_closure:
            return self._zero_rhs_state()
        return ion_neutral_collision_rhs(
            state=state,
            floors=self._floors,
            ion_mass_g=self._ion_mass_g,
            geometry=self._geometry,
            wind_column_factor=self._wind_column_factor,
            **self._collision_operator_kwargs(),
        )

    def _collision_operator_kwargs(self):
        """Return the rate bundle every ion-neutral collision channel shares."""
        return dict(self._options.collision_operator)

    def energy_exchange_rhs(self, y=None, state=None):
        """Return conservative electron-ion thermal exchange sources."""
        if state is None:
            state = self.state if y is None else self._unpack(y)
        return electron_ion_exchange_rhs(
            state=state,
            floors=self._floors,
            ion_mass_g=self._ion_mass_g,
            mu=self._mu,
            **self._energy_exchange_kwargs(),
        )

    def electron_cooling_rhs(self, y=None, state=None):
        """Return conservative electron inelastic/radiative cooling sources."""
        if state is None:
            state = self.state if y is None else self._unpack(y)
        return electron_cooling_rhs(
            state=state,
            floors=self._floors,
            ion_mass_g=self._ion_mass_g,
            **self._electron_cooling_kwargs(),
        )

    def electron_cooling_rhs_terms(self, y=None, state=None):
        """Return split conservative electron cooling source terms."""
        if state is None:
            state = self.state if y is None else self._unpack(y)
        return electron_cooling_rhs_terms(
            state=state,
            floors=self._floors,
            ion_mass_g=self._ion_mass_g,
            **self._electron_cooling_kwargs(),
        )

    def ion_charge_exchange_rhs(self, y=None, state=None):
        """Return conservative ion charge-exchange cooling sources."""
        if state is None:
            state = self.state if y is None else self._unpack(y)
        if self._ion_neutral_moment_closure:
            # CX cooling is folded into the moment-closed ion_neutral_collision
            # term (its (3/2) n nu_mt (Tn-Ti) thermal channel).
            return self._zero_rhs_state()
        return ion_charge_exchange_rhs(
            state=state,
            floors=self._floors,
            ion_mass_g=self._ion_mass_g,
            **self._ion_charge_exchange_kwargs(),
        )

    def heat_conduction_rhs(self, y=None, state=None):
        """Return conservative axial heat-conduction energy sources."""
        if state is None:
            state = self.state if y is None else self._unpack(y)
        return heat_conduction_rhs(
            state=state,
            floors=self._floors,
            ion_mass_g=self._ion_mass_g,
            mu=self._mu,
            geometry=self._plasma_geometry(),
            **self._heat_conduction_kwargs(),
        )

    def cathode_boundary_state(self, y=None, state=None):
        """Return source/end primitive state for future cathode coupling."""
        if state is None:
            state = self.state if y is None else self._unpack(y)
        return cathode_boundary_state(
            state=state,
            floors=self._floors,
            ion_mass_g=self._ion_mass_g,
            geometry=self._geometry,
            input_dict=self._input_dict,
            input_flags=self._flags,
        )

    def cathode_source_terms(self, y=None, state=None, cathode_solve=None, time=None):
        """Return opt-in cathode conservative source placeholders/terms."""
        if state is None:
            state = self.state if y is None else self._unpack(y)
        cathode_flags = self._effective_cathode_flags(time=time, active_only=True)
        if cathode_solve is None and cathode_flags.get("cathode_coupling", False):
            cathode_solve = self.solve_cathode_boundary(
                state=state,
                time=time,
                update_cache=True,
            )
        return cathode_source_terms(
            state=state,
            floors=self._floors,
            ion_mass_g=self._ion_mass_g,
            geometry=self._geometry,
            input_dict=self._input_dict,
            input_flags=cathode_flags,
            cathode_solve=cathode_solve,
        )

    def beam_ionization_rhs(self, y=None, state=None, cathode_solve=None, time=None):
        """Return conservative beam ionization birth terms."""
        if state is None:
            state = self.state if y is None else self._unpack(y)
        cathode_flags = self._effective_cathode_flags(time=time, active_only=True)
        if cathode_solve is None and cathode_flags.get("cathode_coupling", False):
            cathode_solve = self.solve_cathode_boundary(
                state=state,
                time=time,
                update_cache=True,
            )
        return beam_ionization_rhs(
            state=state,
            floors=self._floors,
            ion_mass_g=self._ion_mass_g,
            geometry=self._geometry,
            input_dict=self._input_dict,
            input_flags=cathode_flags,
            I_ion=self._I_ion,
            cathode_solve=cathode_solve,
            coverage=self._coverage_view(state, time),
        )

    def beam_ionization_rhs_terms(
        self,
        y=None,
        state=None,
        cathode_solve=None,
        time=None,
    ):
        """Return split beam particle birth, deposited power, and ionization cost."""
        if state is None:
            state = self.state if y is None else self._unpack(y)
        cathode_flags = self._effective_cathode_flags(time=time, active_only=True)
        if cathode_solve is None and cathode_flags.get("cathode_coupling", False):
            cathode_solve = self.solve_cathode_boundary(
                state=state,
                time=time,
                update_cache=True,
            )
        return beam_ionization_rhs_terms(
            state=state,
            floors=self._floors,
            ion_mass_g=self._ion_mass_g,
            geometry=self._geometry,
            input_dict=self._input_dict,
            input_flags=cathode_flags,
            I_ion=self._I_ion,
            cathode_solve=cathode_solve,
            coverage=self._coverage_view(state, time),
        )

    def beam_deposition_ee_source(self, y=None, state=None, time=None):
        """Return the beam ``Ee`` deposition row [erg cm^-3 s^-1] at a state.

        The source the implicit heat substep integrates when
        ``beam_deposition_in_heat_substep`` is armed, and the exact row
        :meth:`rhs` then leaves out of operator A's sum. It is built from the
        SAME beam path :meth:`rhs_terms` uses and carries the same two gates:
        the phase gate (no plasma, or neutral prebreakdown, means no beam --
        those phases do not run this stepper at all, but a direct caller can
        reach them) and the active-plasma-topology mask. The other two
        post-processing passes ``rhs_terms`` applies leave this row alone by
        construction: both DVM and kinetic stripping preserve ``Ee``, and the
        neutral-energy attachment only adds ``En`` rows.

        The cathode solve here runs with ``update_cache=False``: the solve of
        record -- the one the step's diagnostics and the sheath warm-start
        continuation come from -- stays operator A's, so arming the flag moves
        no diagnostic.
        """
        if state is None:
            state = self.state if y is None else self._unpack(y)
        zeros = np.zeros(self._geometry.cells, dtype=float)
        if not self._flags.get("Plasma") or self._neutral_prebreakdown_active(
            time=time,
        ):
            return zeros
        cathode_phase = self._cathode_phase_options(time=time)
        cathode_solve = None
        if cathode_phase["solve_enabled"]:
            cathode_solve = self.solve_cathode_boundary(
                state=state,
                floating=cathode_phase["floating"],
                time=time,
                update_cache=False,
            )
        beam_terms = self.beam_ionization_rhs_terms(
            state=state,
            cathode_solve=cathode_solve,
            time=time,
        )
        row = np.asarray(
            beam_terms[BEAM_POWER_DEPOSITION_TERM].Ee, dtype=float
        )
        if self._active_plasma_topology:
            row = np.where(self._plasma_active_mask(), row, 0.0)
        return row

    def _cathode_solve_memo_key(
        self,
        state,
        time,
        input_flags,
        floating,
        phi_wf_override_eV,
        circuit_V_src_V,
        coverage,
    ):
        """Return the read-only cathode solve memo's key.

        One entry per argument of the module-level ``solve_cathode_boundary``
        that can move within a run; the run-constant arguments are listed in
        :meth:`solve_cathode_boundary` with the reason they are absent. The
        caller passes the already-resolved values so the key cannot describe a
        different solve from the one it guards.
        """
        return (
            _memo_key_part(state),
            _memo_key_part(time),
            _memo_key_part(input_flags),
            _memo_key_part(bool(floating)),
            _memo_key_part(self._cathode_beam_cross),
            # A2a: a lagged INPUT to this solve, exactly like sigma_b above.
            # Two calls at one (y, t) with different lag values are different
            # solves and must not be served for one another.
            _memo_key_part(self._cathode_tail_anode_I),
            _memo_key_part(self._cathode_x0),
            _memo_key_part(self._cathode_x0_twin),
            _memo_key_part(self._cathode_Ts_K),
            _memo_key_part(phi_wf_override_eV),
            _memo_key_part(self._cathode_f_em),
            _memo_key_part(self._circuit_I_loop),
            _memo_key_part(circuit_V_src_V),
            _memo_key_part(coverage),
            _memo_key_part(self._vessel_V_cm),
        )

    def solve_cathode_boundary(
        self,
        y=None,
        state=None,
        floating=None,
        time=None,
        update_cache=True,
    ):
        """Run the opt-in cathode solver adapter without changing the RHS."""
        if state is None:
            state = self.state if y is None else self._unpack(y)
        state = self._smoothed_sample_state(state)
        cathode_phase = self._cathode_phase_options(time=time)
        if floating is None:
            floating = cathode_phase["floating"]
        input_flags = self._effective_cathode_flags(
            time=time,
            active_only=False,
            floating=bool(floating),
        )
        phi_wf_override_eV = self._cathode_phi_wf_eff()
        circuit_V_src_V = self._circuit_source_voltage_V(cathode_phase)
        coverage = self._coverage_view(state, time)
        # READ-ONLY solve memo (R3 [cathode-accepted-solve-unify]). Several
        # consumers solve at the SAME resolved inputs within one accepted step
        # -- the shared converged accepted-state solve, the dt bound's bundle
        # and the first Strang heat half -- and this serves the second and
        # third of those from the first instead of repeating the sheath/beam
        # solve. Only ``update_cache=False`` calls participate: a cache-writing
        # solve must actually run, because its job is the write (sigma_b, x0,
        # the solve of record), not the value.
        #
        # The key is the FULL mutable input set, not a state/time pair. sigma_b
        # (``beam_cross_prev``) is an INPUT to this solve, so two calls at one
        # (y, t) with different sigma_b are different solves -- that is exactly
        # the lag this member exists to resolve, and keying on state alone
        # would paper over it by serving one for the other. Everything else
        # this passes and that can move within a run joins for the same reason:
        # the warm start (x0/x0_twin), the surface state (T_s, phi_wf, f_em),
        # the circuit (I_loop, V_src, vessel node), the phase-derived flags and
        # ``floating``, and the coverage view. The remaining arguments
        # (``floors``, ``ion_mass_g``, ``mu``, ``geometry``, ``input_dict``,
        # ``I_ion``, ``gas_type``) are assigned once in __init__ and never
        # mutated, so they cannot separate two calls in one run. A miss on any
        # component solves fresh; a stale memo is never served.
        memo_key = None
        if not update_cache:
            memo_key = self._cathode_solve_memo_key(
                state=state,
                time=time,
                input_flags=input_flags,
                floating=floating,
                phi_wf_override_eV=phi_wf_override_eV,
                circuit_V_src_V=circuit_V_src_V,
                coverage=coverage,
            )
            memo = self._cathode_solve_memo
            if memo is not None and memo[0] == memo_key:
                return memo[1]
        result = solve_cathode_boundary(
            state=state,
            floors=self._floors,
            ion_mass_g=self._ion_mass_g,
            mu=self._mu,
            geometry=self._geometry,
            input_dict=self._input_dict,
            input_flags=input_flags,
            beam_cross_prev=self._cathode_beam_cross,
            tail_anode_current_prev_A=self._cathode_tail_anode_I,
            I_ion=self._I_ion,
            gas_type=self._gas_type,
            x0=self._cathode_x0,
            x0_twin=self._cathode_x0_twin,
            floating=floating,
            T_s_override_K=self._cathode_Ts_K,
            phi_wf_override_eV=phi_wf_override_eV,
            f_em_override=self._cathode_f_em,
            circuit_I_loop_A=self._circuit_I_loop,
            circuit_V_src_V=circuit_V_src_V,
            coverage=coverage,
            vessel_V_cm_V=self._vessel_V_cm,
        )
        if memo_key is not None:
            self._cathode_solve_memo = (memo_key, result)
        if update_cache:
            self._warn_beam_gap_ledger(result)
            self._cathode_solve = result
            self._cathode_x0 = result.x0_next
            self._cathode_x0_twin = result.x0_twin_next
            if result.beam_result is not None:
                self._cathode_beam_cross = (
                    result.beam_result.beam_atten_cross.copy()
                )
            # A2a: the lag advances on the SAME acceptance-committed write as
            # sigma_b above. Rejected attempts never reach this branch.
            self._cathode_tail_anode_I = float(result.tail_anode_current_A)
        return result

    def _warn_beam_gap_ledger(self, cathode_solve):
        """Warn ONCE per run if the CSDA beam gap ledger stops closing.

        The circuit charges ``eta * f_bypass`` of the emitted beam power to a
        beam it believes crosses the cathode-anode gap without coupling; the
        fluid deposits whatever the CSDA ray actually delivers. Those views of
        the same gap must agree, and they disagree silently -- no conservation
        check sees it, because each side is internally consistent. That is the
        state the unit-flux probe defect (root-caused 2026-07-27) sat in.

        The third comparison ``ray_vs_ceiling`` is ARMED here: this table
        carries its explanation, which is the condition
        ``beam_gap_ledger_mismatch`` states for the case being anything but
        opt-in. Arming it cannot change WHETHER this warns -- a broken-out
        ray's ceiling shortfall is never larger than its circuit shortfall,
        because the clamp keeps ``circuit <= ceiling`` -- only which of the
        two labels the same excursion is reported under.
        """
        if self._beam_gap_ledger_warned:
            return
        device_config = getattr(cathode_solve, "device_config", None)
        if device_config is None:
            return
        worst = beam_gap_ledger_mismatch(
            getattr(cathode_solve, "beam_gap_ledger", None),
            device_config.eta,
            separate_representability=True,
        )
        if worst is None:
            return
        end, kind, left, right, power = worst
        cause = {
            "probe_vs_ray": (
                "the gap probe reads {left:.6g} while the deposition ray it "
                "is supposed to mirror delivers {right:.6g}. The probe is "
                "misreporting the ray, so the sigma_eff it writes -- and "
                "every circuit quantity downstream of it -- is wrong. This is "
                "the item-35 signature; suspect a launched flux, clump split "
                "or stopping model that the probe and the deposition ray no "
                "longer share"
            ),
            "ray_vs_ceiling": (
                "the deposition ray crossed the gap WHOLE ({left:.6g}) while "
                "the highest survival the Beer-Lambert solve can represent -- "
                "its Coulomb-only ceiling exp(-L_cath/l_bi) -- is {right:.6g}. "
                "No two views of the gap disagree here: a fully transmitting "
                "ray cannot be booked above that ceiling, so the "
                "sigma_eff >= 0 clamp leaves the shortfall unbooked. This is "
                "a REPRESENTABILITY gap, not a defect report; suspect a hot "
                "beam in a dense gap, where l_bi and the range-set "
                "transmission stop tracking each other"
            ),
            "ray_vs_circuit": (
                "the deposition ray delivers gap survival {left:.6g} while "
                "the circuit books a beam-bypass fraction of {right:.6g}. The "
                "usual cause is a gap transmission above the Beer-Lambert "
                "solve's Coulomb-only ceiling exp(-L_cath/l_bi), where the "
                "sigma_eff >= 0 clamp saturates and the adapter cannot "
                "represent the ray"
            ),
        }[kind].format(left=left, right=right)
        self._beam_gap_ledger_warned = True
        warnings.warn(
            f"CSDA beam gap ledger does not close at cathode end {end} "
            f"({kind}): {cause}. That mis-books {100.0 * power:.3g}% of the "
            f"emitted beam power (tolerance "
            f"{100.0 * BEAM_GAP_LEDGER_POWER_ATOL:g}%) -- the circuit is "
            "debiting beam power the fluid does not lose, or vice versa, and "
            "no conservation check sees it because each side is internally "
            "consistent. Warned once per run.",
            stacklevel=2,
        )

    def implicit_heat_conduction_step(self, dt, y=None, state=None, ee_source=None):
        """Return state after one frozen-conductivity implicit heat substep.

        ``ee_source`` (default ``None``, the historical path) is an electron-
        energy source density held constant over the substep and solved with
        the conduction operator; see the module function of the same name.
        """
        if state is None:
            state = self.state if y is None else self._unpack(y)
        return implicit_heat_conduction_step(
            state=state,
            ee_source=ee_source,
            floors=self._floors,
            ion_mass_g=self._ion_mass_g,
            mu=self._mu,
            geometry=self._plasma_geometry(),
            dt=dt,
            implicit_heat_scheme=self._input_dict.get(
                "implicit_heat_scheme"
            ),
            heat_picard_iterations=int(
                self._input_dict.get("heat_picard_iterations")
            ),
            heat_picard_tol=float(self._input_dict.get("heat_picard_tol")),
            **self._heat_conduction_kwargs(),
        )

    def neutral_exchange_rhs(self, y=None, state=None):
        """Return conservative pairwise neutral-exchange sources.

        In two-zone mode the axial exchange runs per zone on the
        precomputed column/annulus Knudsen conductances; the radial
        column/annulus mixing is the separate named
        ``neutral_zone_exchange`` term.
        """
        if state is None:
            state = self.state if y is None else self._unpack(y)
        if self._neutral_two_zone and state.nn_a is not None:
            column_coeff, annulus_coeff = self._zone_axial_coeffs
            return neutral_exchange_two_zone_rhs(
                state=state,
                geometry=self._geometry,
                column_coeff_cm3_s=column_coeff,
                annulus_coeff_cm3_s=annulus_coeff,
                floors=self._floors,
                temperature_scale=self._transpiration_face_scale(state),
            )
        return neutral_exchange_rhs(
            state=state,
            geometry=self._geometry,
            exchange_coeff_cm3_s=self.neutral_exchange_coefficients(),
            floors=self._floors,
            temperature_scale=self._transpiration_face_scale(state),
        )

    def _transpiration_temperature_ratio(self, state):
        """Return the per-cell ``Tn_local / Tn_K`` the transpiration arm reads.

        ``None`` on the frozen (v1-primary) closure, which is what leaves every
        conductance at its construction-time value.
        """
        if self._neutral_knudsen_temperature != "local" or state.En is None:
            return None
        Tn_ref = float(self._input_dict.get("Tn_K")) * kb_cgs / ev_to_erg
        Tn = neutral_temperature_eV(
            state, floors=self._floors, Tn_eV=Tn_ref
        )
        return np.maximum(Tn, 0.0) / Tn_ref

    def _transpiration_face_scale(self, state):
        """Return the internal-face conductance scale, or ``None`` when frozen.

        A conductance is proportional to the thermal speed, so the scale is the
        square root of the temperature ratio, taken on the face average of the
        two adjacent cells.
        """
        ratio = self._transpiration_temperature_ratio(state)
        if ratio is None:
            return None
        return np.sqrt(0.5 * (ratio[:-1] + ratio[1:]))

    def _transpiration_cell_scale(self, state):
        """Return the radial (per-cell) conductance scale, or ``None``."""
        ratio = self._transpiration_temperature_ratio(state)
        if ratio is None:
            return None
        return np.sqrt(ratio)

    def neutral_zone_exchange_rhs(self, y=None, state=None):
        """Return the conservative column/annulus free-molecular exchange."""
        if state is None:
            state = self.state if y is None else self._unpack(y)
        return neutral_zone_exchange_rhs(
            state=state,
            geometry=self._geometry,
            conductance_cm3_s=self._zone_exchange_cm3_s,
            floors=self._floors,
            temperature_scale=self._transpiration_cell_scale(state),
        )

    def neutral_exchange_coefficients(self):
        """Return internal-face neutral exchange coefficients [cm^3/s]."""
        return neutral_exchange_coefficients(
            geometry=self._geometry,
            model=self._input_dict.get("neutral_exchange_model"),
            constant_coeff_cm3_s=float(
                self._input_dict.get("neutral_exchange_coeff_cm3_s")
            ),
            Tn_K=float(self._input_dict.get("Tn_K")),
            mu_neutral=self._mu_neutral,
            clausing_scale=float(self._input_dict.get("neutral_clausing_scale")),
        )

    def neutral_source_sink_rhs(self, y=None, state=None, time=None):
        """Return conservative neutral gas puff and pump sources."""
        if state is None:
            state = self.state if y is None else self._unpack(y)
        return neutral_source_sink_rhs(
            state=state,
            geometry=self._geometry,
            **self._neutral_source_kwargs(time=time),
        )

    def neutral_probe_waveform_value(self, time=None):
        """Return the probe source's INSTANTANEOUS ``w(t)``, or 0.0 when off.

        ``time`` defaults to the solver clock. Zero with the flag off, so a
        caller needs no branch. This is the diagnostic read; what an
        integration step consumes is
        :meth:`neutral_probe_waveform_mean` over the step it is taking.
        """
        if self._probe is None:
            return 0.0
        return neutral_probe_waveform_value(
            self._time if time is None else time,
            self._probe.waveform,
            t_on_s=self._probe.t_on_s,
            t_off_s=self._probe.t_off_s,
            table=self._probe.table,
        )

    def neutral_probe_waveform_mean(self, t0, dt):
        """Return the probe waveform's exact average over ``[t0, t0 + dt]``.

        Zero with the flag off. This is the quantity the integration stages
        consume; ``sum_k dt_k * mean(t_k, dt_k)`` over a run's accepted steps
        is the waveform's exact integral, whatever the step lattice.
        """
        if self._probe is None:
            return 0.0
        return neutral_probe_waveform_mean(
            t0,
            dt,
            self._probe.waveform,
            t_on_s=self._probe.t_on_s,
            t_off_s=self._probe.t_off_s,
            table=self._probe.table,
            table_cumulative=self._probe.table_cumulative,
        )

    def neutral_probe_source_rhs(
        self, y=None, state=None, time=None, step_window=None
    ):
        """Return the ad-hoc probe neutral source term (zeros when off).

        ``step_window`` is the ``(t0, dt)`` interval the caller is integrating
        over. Given, the term carries the waveform's EXACT AVERAGE across it,
        which is what makes the delivered inventory the stated hypothesis
        rather than the trapezoid rule's reading of it (see
        ``physics.neutrals.neutral_probe_waveform_mean``). Omitted -- every
        diagnostic read -- the term carries the instantaneous rate at ``time``,
        which is the same quantity for a window of zero width.
        """
        if self._probe is None:
            return self._zero_rhs_state()
        if state is None:
            state = self.state if y is None else self._unpack(y)
        if step_window is None:
            waveform_value = self.neutral_probe_waveform_value(time=time)
        else:
            waveform_value = self.neutral_probe_waveform_mean(*step_window)
        return neutral_probe_source_rhs(
            state=state,
            geometry=self._geometry,
            amplitude_cm3_s=self._probe.amplitude_cm3_s,
            weights=self._probe.weights,
            waveform_value=waveform_value,
            zone=self._probe.zone,
        )

    def neutral_probe_profile(self):
        """Return the probe source's normalized axial weights ``p(z)`` [1].

        ``None`` whenever the instrument is off. The array is a copy, so a
        caller cannot reach into solver state.
        """
        if self._probe is None:
            return None
        return np.asarray(self._probe.weights, dtype=float).copy()

    def gas_puff_local_ionization_rhs(self, y=None, state=None, time=None):
        """Return the fresh-puff clump local-ionization source (default off)."""
        if state is None:
            state = self.state if y is None else self._unpack(y)
        f = float(self._input_dict.get("gas_puff_local_ionization_fraction"))
        if f <= 0.0:
            return self._zero_rhs_state()
        nk = self._neutral_source_kwargs(time=time)
        if not nk["gas_puff_enabled"]:
            return self._zero_rhs_state()
        puff = gas_puff_rate_profile(
            self._geometry, nk["S_gp"], nk["gas_puff_valves"],
            profile=nk["gas_puff_profile"], z_cm=nk["gas_puff_z_cm"],
            sigma_cm=nk["gas_puff_sigma_cm"], throw_cm=nk["gas_puff_throw_cm"],
            orifice_id_cm=nk["gas_puff_orifice_id_cm"],
            orifice_length_cm=nk["gas_puff_orifice_length_cm"],
            end=0, delivery_fraction=nk["gas_puff_delivery_fraction"],
        )
        if nk["twin_cathode"]:
            puff = puff + gas_puff_rate_profile(
                self._geometry, nk["Twin_S_gp"], nk["gas_puff_valves"],
                profile=nk["gas_puff_profile"], z_cm=nk["gas_puff_z_cm"],
                sigma_cm=nk["gas_puff_sigma_cm"], throw_cm=nk["gas_puff_throw_cm"],
                orifice_id_cm=nk["gas_puff_orifice_id_cm"],
                orifice_length_cm=nk["gas_puff_orifice_length_cm"],
                end=-1, delivery_fraction=nk["gas_puff_delivery_fraction"],
            )
        return _gas_puff_local_ionization_rhs(
            state=state,
            floors=self._floors,
            ion_mass_g=self._ion_mass_g,
            geometry=self._geometry,
            puff_profile=puff,
            fraction=f,
            I_ion=self._I_ion,
            Te_birth_ionization=self._input_dict.get(
                "Te_birth_ionization"
            ),
            Ti_birth_ionization=self._input_dict.get(
                "Ti_birth_ionization"
            ),
            ionization_birth_energy_model=str(
                self._input_dict.get("ionization_birth_energy_model")
            ),
            Tn_K=float(self._input_dict.get("Tn_K")),
        )

    def reaction_rhs(self, y=None, state=None):
        """Return conservative bulk reaction sources."""
        if state is None:
            state = self.state if y is None else self._unpack(y)
        return reaction_rhs(
            state=state,
            floors=self._floors,
            ion_mass_g=self._ion_mass_g,
            geometry=self._geometry,
            **self._reaction_kwargs(),
        )

    def recombination_energy_return_rhs(self, y=None, state=None):
        """Return the GCR-consistent recombination energy pair (Ee only)."""
        if state is None:
            state = self.state if y is None else self._unpack(y)
        return recombination_energy_return_rhs(
            state=state,
            floors=self._floors,
            ion_mass_g=self._ion_mass_g,
            gas_type=self._gas_type,
            I_ion=self._I_ion,
            atomic_rate_model=str(
                self._input_dict.get("atomic_rate_model")
            ),
            enabled=self._recombination_energy_return,
            adas_low_te_extension=bool(
                self._input_dict.get("adas_low_te_extension")
            ),
        )

    def reaction_rhs_terms(self, y=None, state=None):
        """Return split ionization and recombination conservative sources."""
        if state is None:
            state = self.state if y is None else self._unpack(y)
        return reaction_rhs_terms(
            state=state,
            floors=self._floors,
            ion_mass_g=self._ion_mass_g,
            geometry=self._geometry,
            **self._reaction_kwargs(),
        )

    def _cathode_phase_options(self, time=None):
        if time is None:
            time = self._time
        switches = self.phase_switches_at_time(time)
        configured = bool(self._flags.get("cathode_coupling"))
        cathode_enabled = configured and bool(switches["cathode_enabled"])
        floating = configured and bool(switches["floating"])
        # Inductive tail: after the bank transistors open (the "floating"
        # afterglow phase), a nonzero parasitic inductance keeps the loop
        # driven at zero bank volts until its current has decayed -- the
        # measured ~0.5 ms discharge-current tail. Below 1 A (~0.03% of
        # peak, negligible stored energy) the historical floating solution
        # resumes so the late-afterglow sheath physics is unchanged.
        inductive_tail = (
            configured
            and floating
            and float(self._input_dict.get("L_parasitic_H")) > 0.0
            and self._circuit_I_prev > 1.0
        )
        if inductive_tail:
            floating = False
        return {
            "configured": configured,
            "cathode_enabled": cathode_enabled,
            "floating": floating,
            "inductive_tail": inductive_tail,
            "solve_enabled": cathode_enabled or floating or inductive_tail,
        }

    def _circuit_source_voltage_V(self, step_phase):
        """Return the loop's source voltage [V] for this phase.

        Zero once the bank transistors have opened (the inductive tail drives
        the loop at zero bank volts), the live capacitor voltage where a bank
        capacitance is configured, and the fixed ``V_bank`` otherwise. Pure
        read: the lazy first-step charge of ``_circuit_V_cap`` stays in the
        circuit advance, so calling this from the RHS-side solve cannot move
        any solver state.
        """
        if bool(step_phase.get("inductive_tail", False)):
            return 0.0
        C_bank = self._input_dict.get("C_bank_F")
        if (
            C_bank is not None
            and float(C_bank) > 0.0
            and self._circuit_V_cap is not None
        ):
            return float(self._circuit_V_cap)
        return float(self._input_dict.get("V_bank"))

    def _circuit_timestep_kwargs(self, state=None, time=None):
        """Bundle for the loop-relaxation timestep bound, or ``None``.

        ``None`` withdraws the candidate. PRESENCE-GATED on
        ``cathode_circuit_voltage_bound``: without the flag the sheath's
        capability wall never clamps the device voltage, the loop equation
        already carried its restoring force, and the historical adaptive
        controller stands -- so an unarmed run never builds this bundle,
        never spends its two probe solves, and keeps its dt sequence to the
        bit. Also withdrawn where there is no loop to bound: an open circuit
        (no solve, or the floating afterglow), which is the same phase gate
        the circuit advance itself uses.

        The bundle carries the SAME ``vdis_of_I`` the advance integrates,
        built at the state the step starts from, so the bound and the step
        cannot disagree about the device relation.
        """
        if not bool(self._flags.get("cathode_circuit_voltage_bound")):
            return None
        if time is None:
            time = self._time
        step_phase = self._cathode_phase_options(time=time)
        if not step_phase["solve_enabled"] or step_phase["floating"]:
            return None
        vdis = idriven_vdis_evaluator(
            state=self.state if state is None else state,
            floors=self._floors,
            ion_mass_g=self._ion_mass_g,
            mu=self._mu,
            geometry=self._geometry,
            input_dict=self._input_dict,
            input_flags=self._effective_cathode_flags(
                active_only=False, floating=False
            ),
            beam_cross_prev=self._cathode_beam_cross,
            T_s_override_K=self._cathode_Ts_K,
            phi_wf_override_eV=self._cathode_phi_wf_eff(),
            f_em_override=self._cathode_f_em,
            circuit_V_src_V=self._circuit_source_voltage_V(step_phase),
        )
        return {
            "vdis_of_I": vdis,
            "I_A": float(self._circuit_I_loop),
            "L_H": float(self._input_dict.get("L_parasitic_H")),
            "V_src_V": self._circuit_source_voltage_V(step_phase),
            # The external partition, exactly as the advance passes it: the
            # internal partition and R_mesh are already inside vdis_of_I, and
            # the two sum back to the total loop resistance.
            "R_series_ohm": float(self._input_dict.get("R_comp"))
            * float(self._input_dict.get("R_comp_partition")),
        }

    def _effective_cathode_flags(self, time=None, active_only=True, floating=None):
        options = self._cathode_phase_options(time=time)
        # The inductive tail keeps the loop electrically active after the bank
        # opens: its solve is a driven (V=0) circuit, so it counts as enabled
        # for both the solve and the source terms it feeds.
        enabled = options["cathode_enabled"] or options.get(
            "inductive_tail", False
        )
        if not active_only:
            use_floating = options["floating"] if floating is None else bool(floating)
            enabled = enabled or (options["configured"] and use_floating)
        flags = dict(self._flags)
        flags["cathode_coupling"] = bool(enabled)
        return flags

    def _neutral_source_kwargs(self, time=None):
        if time is None:
            time = self._time
        phase_switches = self.phase_switches_at_time(time)
        S_gp, Twin_S_gp = self._effective_gas_puff_sccm(time=time)
        return {
            "S_gp": S_gp,
            "Twin_S_gp": Twin_S_gp,
            "S_pump_L": float(self._input_dict.get("S_pump_L")),
            "S_pump_R": float(self._input_dict.get("S_pump_R")),
            "twin_cathode": self._flags.get("TwinCathode"),
            "gas_puff_enabled": bool(phase_switches["gas_puff_enabled"]),
            "pump_enabled": bool(self._input_dict.get("pump_enabled")),
            "gas_puff_valves": float(self._input_dict.get("gas_puff_valves")),
            "pump_elbow_conductance_lps": self._input_dict.get(
                "pump_elbow_conductance_lps"
            ),
            "gas_puff_profile": str(
                self._input_dict.get("gas_puff_profile")
            ),
            "gas_puff_z_cm": self._input_dict.get("gas_puff_z_cm"),
            "gas_puff_sigma_cm": float(
                self._input_dict.get("gas_puff_sigma_cm")
            ),
            "gas_puff_throw_cm": float(
                self._input_dict.get("gas_puff_throw_cm")
            ),
            "gas_puff_orifice_id_cm": self._input_dict.get(
                "gas_puff_orifice_id_cm"
            ),
            "gas_puff_orifice_length_cm": self._input_dict.get(
                "gas_puff_orifice_length_cm"
            ),
            "gas_puff_delivery_fraction": float(
                self._input_dict.get("gas_puff_delivery_fraction")
            ),
        }

    def _init_sample_smoothing(self):
        """Parse and seed the electrode sample-smoothing EMA (config.py).

        Tracked cells: the cathode sample cell and the first anode face's
        two flanking cells -- every (n, Te) a sheath solve samples. The EMA
        is seeded from the initial state (deterministic) and updated on
        accepted steps only, so dt-retries never move it.

        KNOWN DEVIATION, deliberate and documented rather than unified: the
        smoothed sample does NOT reach every consumer of those cells. The
        substitution happens in ``_smoothed_sample_state``, and only its
        callers see it --

        - the RHS/beam-side sheath solve (``solve_cathode_boundary``, which
          smooths the state it was handed before dispatching), and
        - the accepted-state re-solve that feeds the surface updates
          (cathode warming and the coverage model),

        both consume the EMA sample. The current-driven CIRCUIT advance does
        not: it builds its ``V_dis(I)`` evaluator on ``self.state``
        directly, so the loop-current root-find reads the RAW accepted
        (n, Te) in the same cells. With smoothing on, the two sides can
        therefore evaluate the sheath from different (n, Te) within one
        accepted step. Smoothing is off by default (``None``), where
        ``_smoothed_sample_state`` returns its argument unchanged and the
        two sides read the identical state bit for bit.
        """
        raw = self._input_dict.get("cathode_sample_smoothing")
        self._sample_smoothing = None
        self._sample_ema = None
        self._sample_smooth_cells = ()
        if raw is None:
            return
        if isinstance(raw, str):
            if raw != "presheath":
                raise ValueError(
                    "cathode_sample_smoothing must be None, 'presheath', or "
                    f"a positive time constant in seconds (got {raw!r})"
                )
            self._sample_smoothing = "presheath"
        else:
            tau = float(raw)
            if tau <= 0.0:
                raise ValueError(
                    "cathode_sample_smoothing time constant must be positive "
                    f"(got {tau})"
                )
            self._sample_smoothing = tau
        cells = [cathode_sample_indices(self._geometry)[0]]
        anode_faces = np.asarray(
            getattr(self._geometry, "anode_face_indices", ()), dtype=int
        )
        if anode_faces.size:
            face = int(anode_faces[0])
            cells += [face - 1, face]
        self._sample_smooth_cells = tuple(dict.fromkeys(int(c) for c in cells))
        derived = derive_state(self._state, self._floors, self._ion_mass_g)
        self._sample_ema = {
            c: [float(self._state.n[c]), float(derived.Te[c])]
            for c in self._sample_smooth_cells
        }

    def _update_sample_smoothing(self, dt):
        """Blend the accepted state into the electrode sample EMA."""
        if self._sample_ema is None:
            return
        derived = derive_state(self._state, self._floors, self._ion_mass_g)
        for c in self._sample_smooth_cells:
            n_ema, Te_ema = self._sample_ema[c]
            if self._sample_smoothing == "presheath":
                # Ion transit across the sampled cell at the (smoothed)
                # sound speed: the physical supply-averaging time.
                cs = ion_sound_speed(max(Te_ema, self._floors["Te"]), self._mu)
                tau = float(self._geometry.length_cm[c]) / max(cs, 1.0)
            else:
                tau = float(self._sample_smoothing)
            alpha = 1.0 - math.exp(-float(dt) / tau)
            self._sample_ema[c] = [
                n_ema + alpha * (float(self._state.n[c]) - n_ema),
                Te_ema + alpha * (float(derived.Te[c]) - Te_ema),
            ]

    def _smoothed_sample_state(self, state):
        """Return ``state`` with the sampled electrode cells' (n, Te)
        replaced by their supply-averaged EMA values (config.py:
        cathode_sample_smoothing). Off (the default) returns ``state``
        unchanged -- the golden path is bit-exact.

        This is the ONLY substitution site, so the smoothed sample reaches
        exactly its callers: the RHS/beam-side sheath solve and the
        accepted-state surface-update re-solve. The current-driven circuit
        advance bypasses it and reads the raw accepted state -- a known
        deviation from "every sheath consumer sees the EMA", described in
        full on ``_init_sample_smoothing``."""
        if self._sample_ema is None:
            return state
        n = np.asarray(state.n, dtype=float).copy()
        Ee = np.asarray(state.Ee, dtype=float).copy()
        for c in self._sample_smooth_cells:
            n_ema, Te_ema = self._sample_ema[c]
            n[c] = n_ema
            Ee[c] = 1.5 * n_ema * Te_ema * ev_to_erg
        return ConservativeState1D(
            n=n,
            nn=state.nn,
            M=state.M,
            Ee=Ee,
            Ei=state.Ei,
            M_n=state.M_n,
            nn_a=state.nn_a,
            M_n_a=state.M_n_a,
            En=state.En,
        )

    def _phase_transition_mode(self):
        return self._input_dict.get("phase_transition_mode")

    def _prebreakdown_timeout_action(self):
        return self._input_dict.get("prebreakdown_timeout_action")

    def _main_discharge_start_time(self):
        if self._phase_transition_mode() == "current":
            return self._t_breakdown_trigger
        plasma_origin = self._plasma_phase_time_origin()
        tau_prebreakdown = max(
            float(self._input_dict.get("tau_prebreakdown")),
            0.0,
        )
        tau_breakdown = max(float(self._input_dict.get("tau_breakdown")), 0.0)
        return plasma_origin + tau_prebreakdown + tau_breakdown

    def _current_trigger_t_end(self):
        if (
            not self._flags.get("Plasma")
            or self._phase_transition_mode() != "current"
            or self._t_breakdown_trigger is None
        ):
            return None
        tau_discharge = max(float(self._input_dict.get("tau_discharge")), 0.0)
        tau_afterglow = max(float(self._input_dict.get("tau_afterglow")), 0.0)
        return float(self._t_breakdown_trigger) + tau_discharge + tau_afterglow

    def _effective_gas_puff_sccm(self, time=None):
        if time is None:
            time = self._time
        phase, phase_elapsed = self._phase_info(time)
        S_gp = float(self._input_dict.get("S_gp"))
        Twin_S_gp = float(self._input_dict.get("Twin_S_gp"))
        if not self._flags.get("Plasma"):
            return S_gp, Twin_S_gp
        mode = self._input_dict.get("gas_puff_mode")
        if mode == "square":
            # Measured valve behaviour (Tom, 2026-07-21): the piezo is driven
            # by a SQUARE voltage pulse fired by the SAME trigger that closes
            # the cathode circuit, held for the discharge duration. The
            # supply side is hydraulically stiff (1/4" line at 45 PSI has
            # ~1e6-sccm-class conductance and ~270 shots of stored inventory,
            # so no sag can develop at ~6e3 sccm delivery; the downstream
            # 10 cm stub is at chamber vacuum, so no burst either) -- the
            # delivery is therefore FLAT at S_gp, with erf rise/close set by
            # the piezo opening + entry transit (~0.5-1 ms, boxed by
            # hardware, not fit). Rise anchors on circuit-on (the end of the
            # neutral-prebreakdown phase = the model's trigger instant);
            # close anchors on drive end + the same lag, and its tail runs
            # into the afterglow (a real valve keeps delivering while it
            # closes). No prebreakdown full-rate flow: breakdown rides the
            # inter-shot residual fill, as in the machine.
            rise_center = float(
                self._input_dict.get("gas_puff_rise_center_s")
            )
            rise_width = float(
                self._input_dict.get("gas_puff_rise_width_s")
            )
            close_lag = float(
                self._input_dict.get("gas_puff_close_lag_s")
            )
            t_on = self._plasma_phase_time_origin() + rise_center
            rise = 0.5 * (1.0 + math.erf((float(time) - t_on) / rise_width))
            tau_discharge = max(
                float(self._input_dict.get("tau_discharge")), 0.0
            )
            if self._phase_transition_mode() == "current":
                main_start = self._t_breakdown_trigger
            else:
                main_start = (
                    self._plasma_phase_time_origin()
                    + max(float(self._input_dict.get("tau_prebreakdown")), 0.0)
                    + max(float(self._input_dict.get("tau_breakdown")), 0.0)
                )
            fall = 0.0
            if main_start is not None:
                t_close = float(main_start) + tau_discharge + close_lag
                fall = 0.5 * (
                    1.0 + math.erf((float(time) - t_close) / rise_width)
                )
            envelope = max(rise - fall, 0.0)
            return S_gp * envelope, Twin_S_gp * envelope
        if mode == "double_erf":
            # Valve-like waveform: an erf rise 0 -> S_gp and an erf drop
            # S_gp -> S_gp_decay_target, both on the *scheduled*
            # main-discharge clock so the rise can sit at negative times
            # (a real valve opens before breakdown). Smooth everywhere, so
            # no event capture is needed. The other modes' full-rate
            # prebreakdown behaviour is replaced by the waveform itself.
            if phase not in {
                "neutral_prebreakdown",
                "pre_breakdown",
                "breakdown",
                "main_discharge",
            }:
                return 0.0, 0.0
            plasma_origin = self._plasma_phase_time_origin()
            main_start = (
                plasma_origin
                + max(float(self._input_dict.get("tau_prebreakdown")), 0.0)
                + max(float(self._input_dict.get("tau_breakdown")), 0.0)
            )
            t_rel = float(time) - main_start
            rise = 0.5 * (
                1.0
                + math.erf(
                    (t_rel - float(self._input_dict.get("tau_gp_rise_center")))
                    / float(self._input_dict.get("tau_gp_rise_width"))
                )
            )
            drop = 0.5 * (
                1.0
                + math.erf(
                    (t_rel - float(self._input_dict.get("tau_gp_drop_center")))
                    / float(self._input_dict.get("tau_gp_drop_width"))
                )
            )
            target = float(self._input_dict.get("S_gp_decay_target"))
            twin_target = float(self._input_dict.get("Twin_S_gp_decay_target"))
            return (
                max(S_gp * rise - (S_gp - target) * drop, 0.0),
                max(Twin_S_gp * rise - (Twin_S_gp - twin_target) * drop, 0.0),
            )
        if phase == "neutral_prebreakdown":
            return S_gp, Twin_S_gp
        if phase in {"pre_breakdown", "breakdown"}:
            return S_gp, Twin_S_gp
        if phase != "main_discharge":
            return 0.0, 0.0

        if mode == "pulse_decay_to_level":
            pulse_duration = float(self._input_dict.get("tau_gp_pulse_duration"))
            if phase_elapsed <= pulse_duration:
                return S_gp, Twin_S_gp
            decay_elapsed = phase_elapsed - pulse_duration
            tau_decay = float(self._input_dict.get("tau_gp_decay_duration"))
            decay = float(np.exp(-decay_elapsed / tau_decay))
            target = float(self._input_dict.get("S_gp_decay_target"))
            twin_target = float(self._input_dict.get("Twin_S_gp_decay_target"))
            return (
                target + (S_gp - target) * decay,
                twin_target + (Twin_S_gp - twin_target) * decay,
            )

        tau_after_breakdown = self._input_dict.get("tau_gp_after_breakdown")
        if tau_after_breakdown is None:
            return S_gp, Twin_S_gp
        tau_after_breakdown = float(tau_after_breakdown)
        if phase_elapsed <= tau_after_breakdown:
            return S_gp, Twin_S_gp
        tau_discharge = max(float(self._input_dict.get("tau_discharge")), 0.0)
        tau_decay = (tau_discharge - tau_after_breakdown) * float(
            self._input_dict.get("tau_gp_decay_factor")
        )
        if tau_decay <= 0.0:
            return S_gp, Twin_S_gp
        decay = float(np.exp(-(phase_elapsed - tau_after_breakdown) / tau_decay))
        return decay * S_gp, decay * Twin_S_gp

    def _neutral_probe_event_times(self):
        """Return the probe waveform's hard edges [s], on the absolute clock.

        These sharpen the applied rate, not the delivered inventory, which is
        exact on any lattice -- see the note at the caller.

        Empty with the instrument off, and for ``"const"``, which never
        changes. A ``"square"`` contributes both its edges. A ``"table"``
        contributes the two ends of its tabulated span, where ``w`` drops to
        zero outside: it is piecewise linear and continuous WITHIN the span,
        so its interior nodes need no capture, but each end is a jump whenever
        the caller tabulated a non-zero ``w`` there.
        """
        if self._probe is None:
            return []
        if self._probe.waveform == "square":
            return [float(self._probe.t_on_s), float(self._probe.t_off_s)]
        if self._probe.waveform == "table":
            nodes = np.asarray(self._probe.table, dtype=float)
            return [float(nodes[0, 0]), float(nodes[-1, 0])]
        return []

    def _gas_puff_event_time(self):
        if not (
            self._flags.get("Plasma")
            and bool(self._input_dict.get("gas_puff_enabled"))
        ):
            return None
        main_start = self._main_discharge_start_time()
        if main_start is None:
            return None
        tau_discharge = max(float(self._input_dict.get("tau_discharge")), 0.0)
        mode = self._input_dict.get("gas_puff_mode")
        if mode == "pulse_decay_to_level":
            event = main_start + float(
                self._input_dict.get("tau_gp_pulse_duration")
            )
        else:
            tau_after_breakdown = self._input_dict.get("tau_gp_after_breakdown")
            if tau_after_breakdown is None:
                return None
            event = main_start + float(tau_after_breakdown)
        if event >= main_start + tau_discharge:
            return None
        return event

    def _cathode_total_current_A(self, cathode_solve=None):
        """Return the cathode solve's total current [A], 0.0 when absent.

        ``cathode_solve`` names WHICH solve to read; ``None`` falls back to
        the cached solve of record, which is every historical caller's
        reading. The phase trigger passes its own converged accepted-state
        solve instead (see :meth:`_update_current_phase_triggers`).
        """
        if cathode_solve is None:
            cathode_solve = self._cathode_solve
        if cathode_solve is None or cathode_solve.beam_result is None:
            return 0.0
        return float(cathode_solve.beam_result.result.I_tot)

    def _current_threshold_time(self, threshold, I_now):
        previous_time = self._last_current_trigger_time
        previous_I = self._last_current_trigger_I_tot
        current_time = float(self._time)
        threshold = float(threshold)
        I_now = float(I_now)
        if previous_time is None or previous_I is None:
            return current_time
        previous_time = float(previous_time)
        previous_I = float(previous_I)
        if not np.all(np.isfinite([previous_time, previous_I, current_time, I_now])):
            return current_time
        if (
            previous_I < threshold <= I_now
            and I_now > previous_I
            and current_time > previous_time
        ):
            fraction = (threshold - previous_I) / (I_now - previous_I)
            return previous_time + fraction * (current_time - previous_time)
        return current_time

    def _record_current_trigger_sample(self, I_now):
        self._last_current_trigger_time = float(self._time)
        self._last_current_trigger_I_tot = float(I_now)
        self._current_trigger_samples.append(
            (self._last_current_trigger_time, self._last_current_trigger_I_tot)
        )

    def _open_ignition_switch(self, time, reason, context=None):
        """Open the cathode switch and route the run into the afterglow.

        The single abort action shared by the ignition-stall trip and the
        ``tau_prebreakdown`` hardware-guard timeout. It mirrors what the
        machine does: the switch opens, the drive stops, and what plasma there
        is decays. Mechanically it re-anchors the phase scheduler -- from
        ``time`` on, ``_phase_info`` reports ``afterglow`` (cathode drive off,
        cathode floating, the square valve's closing tail still delivering)
        and then ``post_afterglow``, and ``run`` shortens ``t_end`` to
        ``time + tau_afterglow`` so the run winds down physically and STOPS
        instead of crawling to ``tau_prebreakdown`` at a collapsed timestep.
        No state vector is touched; the abort is a phase transition.
        """
        if self._t_ignition_abort is not None:
            return False
        self._t_ignition_abort = float(time)
        self._ignition_abort_reason = str(reason)
        self._ignition_abort_context = {
            key: float(value) for key, value in dict(context or {}).items()
        }
        detail = ""
        if self._ignition_abort_context:
            detail = " | " + " ".join(
                f"{key}={self._ignition_abort_context[key]:.4g}"
                for key in (
                    "gamma_N_per_s",
                    "gamma_nn_per_s",
                    "dEe_total_W",
                    "P_beam_W",
                    "P_conduction_W",
                    "P_cooling_W",
                    "P_ionization_W",
                    "P_transport_W",
                    "P_beam_end_loss_W",
                )
                if key in self._ignition_abort_context
            )
        warnings.warn(
            f"ignition aborted at t={self._t_ignition_abort:.6e} s "
            f"(reason={self._ignition_abort_reason}): the cathode switch is "
            "OPEN and the run winds down through the afterglow. This run did "
            "NOT ignite -- it has no main_discharge phase and must not be "
            f"scored{detail}",
            stacklevel=2,
        )
        return True

    def _prebreakdown_timeout_switch_open(
        self,
        threshold,
        threshold_name,
        I_now,
        tau_prebreakdown,
    ):
        """Open the switch at the ``tau_prebreakdown`` hardware guard.

        The real LAPD opens the cathode switch if the discharge has not broken
        down by ``tau_prebreakdown`` (0.05 s, hardware-boxed). This mirrors it
        as a real phase transition rather than an exception, so the run leaves
        an artifact that states what the drive was doing when the guard fired.
        """
        context = dict(self._ignition_abort_context or {})
        context.update(
            {
                "I_tot_A": float(I_now),
                "threshold_A": float(threshold),
                "tau_prebreakdown_s": float(tau_prebreakdown),
            }
        )
        last = self._last_ignition_record or {}
        for key in (
            "gamma_N_per_s",
            "gamma_nn_per_s",
            "dEe_total_W",
            "P_beam_W",
            "P_conduction_W",
            "P_cooling_W",
            "P_ionization_W",
            "P_transport_W",
            "P_beam_end_loss_W",
        ):
            if key in last:
                context[key] = float(last[key])
        self._open_ignition_switch(
            time=float(self._time),
            reason="prebreakdown_timeout",
            context=context,
        )
        self._ignition_abort_threshold_name = str(threshold_name)
        self._record_current_trigger_sample(I_now)

    def _check_ignition_budget_guards(self, accepted_steps, wall_clock_start):
        """Trip the switch-open abort on a wall-clock or accepted-step budget.

        The non-simulated-time arm of the non-ignition guards. The stall
        detector and the ``tau_prebreakdown`` timeout both measure SIMULATED
        time, so both are blind to the failure mode where a non-igniting run
        stops producing simulated time at all: the timestep collapses and the
        arm burns hours of wall clock without ever reaching the simulated
        instant either guard watches. These two budgets bound that directly --
        one in wall clock, one in work done -- and route into the same
        ``_open_ignition_switch`` wind-down, so a tripped run leaves the same
        kind of artifact and is refused scoring for the same reason.

        Inert once the run has broken down: a discharge that ignited is not a
        non-ignition failure however long it subsequently takes.
        """
        if (
            self._t_breakdown_trigger is not None
            or self._t_ignition_abort is not None
        ):
            return False
        step_cap = self._ignition_accepted_step_cap
        wall_cap = self._ignition_wall_clock_cap_s
        elapsed = (
            perf_counter() - wall_clock_start
            if wall_clock_start is not None
            else 0.0
        )
        if step_cap > 0 and accepted_steps >= step_cap:
            reason = "accepted_step_cap"
        elif wall_cap > 0.0 and elapsed >= wall_cap:
            reason = "wall_clock_cap"
        else:
            return False
        context = dict(self._last_ignition_record or {})
        context.update(
            {
                "accepted_steps": float(accepted_steps),
                "wall_clock_s": float(elapsed),
                "ignition_accepted_step_cap": float(step_cap),
                "ignition_wall_clock_cap_s": float(wall_cap),
            }
        )
        return self._open_ignition_switch(
            time=float(self._time),
            reason=reason,
            context=context,
        )

    def _ignition_abort_t_end(self):
        """Return the wind-down end time [s] after a switch-open abort."""
        if self._t_ignition_abort is None:
            return None
        tau_afterglow = max(float(self._input_dict.get("tau_afterglow")), 0.0)
        return float(self._t_ignition_abort) + tau_afterglow

    def _dynamic_t_end(self, current_trigger_enabled):
        """Return the shortest dynamically-determined end time [s], or None."""
        candidates = []
        if current_trigger_enabled:
            candidates.append(self._current_trigger_t_end())
        candidates.append(self._ignition_abort_t_end())
        finite = [value for value in candidates if value is not None]
        return min(finite) if finite else None

    def _update_current_phase_triggers(self):
        if (
            self._phase_transition_mode() != "current"
            or not self._flags.get("Plasma")
            or self._neutral_prebreakdown_active()
        ):
            return
        cathode_phase = self._cathode_phase_options(time=self._time)
        converged_solve = None
        if cathode_phase["solve_enabled"]:
            # Two solves, and they are not the same solve (R3
            # [cathode-accepted-solve-unify]). The FIRST is the sigma_b
            # refresher and keeps its historical job: it consumes the sigma_b
            # the accepted step's last internal stage left -- a STAGE-state
            # value -- and writes the accepted-state one, along with the warm
            # start and the solve of record every existing diagnostic reads.
            #
            # The SECOND re-solves at the same accepted state on the sigma_b
            # the first just wrote, which is the converged one: the lag map
            # reaches its fixed point in a single iteration, so this solve is
            # what the dt bound and the first Strang heat half go on to see.
            # The trigger's threshold test reads THIS, not the lagged first
            # solve, and the memo hands the same object to those two consumers
            # instead of letting them repeat it -- which is why the step's
            # solve count falls by one even though a solve was added here.
            self.solve_cathode_boundary(time=self._time, update_cache=True)
            converged_solve = self.solve_cathode_boundary(
                time=self._time, update_cache=False
            )

        I_now = self._cathode_total_current_A(cathode_solve=converged_solve)
        tau_prebreakdown = max(
            float(self._input_dict.get("tau_prebreakdown")),
            0.0,
        )
        I_prebreakdown = float(self._input_dict.get("I_prebreakdown"))
        I_breakdown = float(self._input_dict.get("I_breakdown"))
        time_tol = max(1e-15, 1e-12 * max(abs(tau_prebreakdown), 1.0))
        plasma_origin = self._plasma_phase_time_origin()
        current_phase_elapsed = max(self._time - plasma_origin, 0.0)

        if self._t_breakdown_trigger is not None:
            return
        if self._t_prebreakdown_trigger is None:
            first_threshold = I_prebreakdown if I_prebreakdown > 0.0 else I_breakdown
            if I_now >= first_threshold:
                trigger_time = self._current_threshold_time(first_threshold, I_now)
                if I_prebreakdown > 0.0:
                    self._t_prebreakdown_trigger = trigger_time
                else:
                    self._t_breakdown_trigger = trigger_time
                self._record_current_trigger_sample(I_now)
                return
            if current_phase_elapsed >= tau_prebreakdown - time_tol:
                if self._prebreakdown_timeout_action() == "switch_open":
                    self._prebreakdown_timeout_switch_open(
                        threshold=first_threshold,
                        threshold_name=(
                            "I_prebreakdown"
                            if I_prebreakdown > 0.0
                            else "I_breakdown"
                        ),
                        I_now=I_now,
                        tau_prebreakdown=tau_prebreakdown,
                    )
                    return
                raise BreakdownError(
                    "plasma failed to break down within "
                    f"tau_prebreakdown={tau_prebreakdown:.9e} s "
                    f"(I_tot={I_now:.6g} A < threshold={first_threshold:.6g} A)",
                    phase="pre_breakdown",
                    time=float(self._time),
                    I_tot=float(I_now),
                    threshold=float(first_threshold),
                    threshold_name=(
                        "I_prebreakdown"
                        if I_prebreakdown > 0.0
                        else "I_breakdown"
                    ),
                    tau_prebreakdown=float(tau_prebreakdown),
                    phase_events=self._phase_events(
                        run_start=self._run_start_for_phase_events,
                        final_time=float(self._time),
                    ),
                    current_trigger_samples=_current_trigger_sample_arrays(
                        self._current_trigger_samples + [(self._time, I_now)]
                    ),
                )
            self._record_current_trigger_sample(I_now)
            return

        if I_now >= I_breakdown:
            self._t_breakdown_trigger = self._current_threshold_time(
                I_breakdown,
                I_now,
            )
            self._record_current_trigger_sample(I_now)
            return
        if current_phase_elapsed >= tau_prebreakdown - time_tol:
            if self._prebreakdown_timeout_action() == "switch_open":
                self._prebreakdown_timeout_switch_open(
                    threshold=I_breakdown,
                    threshold_name="I_breakdown",
                    I_now=I_now,
                    tau_prebreakdown=tau_prebreakdown,
                )
                return
            raise BreakdownError(
                "plasma failed to reach breakdown current within "
                f"tau_prebreakdown={tau_prebreakdown:.9e} s "
                f"(I_tot={I_now:.6g} A < I_breakdown={I_breakdown:.6g} A)",
                phase="breakdown",
                time=float(self._time),
                I_tot=float(I_now),
                threshold=float(I_breakdown),
                threshold_name="I_breakdown",
                tau_prebreakdown=float(tau_prebreakdown),
                phase_events=self._phase_events(
                    run_start=self._run_start_for_phase_events,
                    final_time=float(self._time),
                ),
                current_trigger_samples=_current_trigger_sample_arrays(
                    self._current_trigger_samples + [(self._time, I_now)]
                ),
            )
        self._record_current_trigger_sample(I_now)

    def _energy_exchange_kwargs(self):
        return dict(self._options.energy_exchange)

    def _surface_loss_kwargs(self):
        return dict(self._options.surface_loss)

    def _ion_neutral_drag_kwargs(self):
        return dict(self._options.ion_neutral_drag)

    def _slip_closure_kwargs(self):
        """Extra kwargs for the drag/frictional-heating slip closure."""
        return dict(self._options.slip_closure)

    def _electron_cooling_kwargs(self):
        return dict(self._options.electron_cooling)

    def _ion_charge_exchange_kwargs(self):
        return dict(self._options.ion_charge_exchange)

    def _heat_conduction_kwargs(self):
        return dict(self._options.heat_conduction)

    def _neutral_energy_timestep_kwargs(self):
        """Return the bundle the En relaxation bound reads, or None."""
        bundle = self._options.neutral_energy_timestep
        return None if bundle is None else dict(bundle)

    def _reaction_kwargs(self):
        return dict(self._options.reaction)

    def _trajectory_snapshot(self, time):
        state = self.state
        derived = self.derived
        assert_finite_state(state, derived)
        rhs_terms = self.rhs_terms(include_heat_conduction=True, time=time)
        packed_fields = state_field_names(state)
        phase, phase_elapsed = self._phase_info(time)
        phase_switches = self._phase_switches(phase)
        wind = {}
        if state.M_n is not None:
            wind = {
                "M_n": state.M_n.copy(),
                "u_n": neutral_wind_velocity(
                    state,
                    floors=self._floors,
                    ion_mass_g=self._ion_mass_g,
                    geometry=self._geometry,
                ),
            }
        if state.En is not None:
            wind["En"] = state.En.copy()
            wind["Tn"] = neutral_temperature_eV(
                state, floors=self._floors, Tn_eV=np.nan
            )
            # The hot channel is algebraic, so its standing population is a
            # DIAGNOSTIC row rather than packed state: saved from the last RHS
            # evaluation's own rates, not recomputed from the saved sample.
            for name in HOT_CHANNEL_DIAGNOSTIC_FIELDS:
                value = self._hot_channel_diagnostics.get(name)
                wind[name] = (
                    np.zeros_like(state.nn)
                    if value is None
                    else np.asarray(value, dtype=float).copy()
                )
            # The ionization-birth thermal deficit is a DIAGNOSTIC row on the
            # same side channel: what the En sink debited per ionized atom
            # minus what the Ei birth booked, times the birth rate. Zero to
            # roundoff under Ti_birth_ionization="neutral"; the size of the
            # leak otherwise.
            for name in IONIZATION_BIRTH_DEFICIT_DIAGNOSTIC_FIELDS:
                value = self._birth_deficit_diagnostics.get(name)
                wind[name] = (
                    np.zeros_like(state.nn)
                    if value is None
                    else np.asarray(value, dtype=float).copy()
                )
        if state.nn_a is not None:
            wind["nn_a"] = state.nn_a.copy()
        if state.M_n_a is not None:
            wind["M_n_a"] = state.M_n_a.copy()
            wind["u_n_a"] = state.M_n_a / (
                self._ion_mass_g
                * np.maximum(state.nn_a, self._floors["nn"])
            )
        cathode_diagnostics = {
            **self._cathode_diagnostic_snapshot(time=time),
            # R5.4: collector surface-power ledger line (diagnostic-only).
            "collector_surface_power_W": self._collector_surface_power_W(
                rhs_terms, derived
            ),
        }
        ignition_diagnostics = self._ignition_diagnostic_snapshot(
            time=time,
            phase=phase,
            state=state,
            rhs_terms=rhs_terms,
            cathode_diagnostics=cathode_diagnostics,
        )
        snapshot = {
            **wind,
            "time": float(time),
            "phase": phase,
            "phase_elapsed": float(phase_elapsed),
            "phase_cathode_enabled": float(phase_switches["cathode_enabled"]),
            "phase_gas_puff_enabled": float(phase_switches["gas_puff_enabled"]),
            "phase_floating": float(phase_switches["floating"]),
            "n": state.n.copy(),
            "nn": state.nn.copy(),
            "M": state.M.copy(),
            "Ee": state.Ee.copy(),
            "Ei": state.Ei.copy(),
            "u": derived.u.copy(),
            "Te": derived.Te.copy(),
            "Ti": derived.Ti.copy(),
            "pe": derived.pe.copy(),
            "pi": derived.pi.copy(),
            "p": derived.p.copy(),
            "y": pack_state(state),
            "rhs_terms": {
                term_name: {
                    field_name: (
                        np.zeros(self._geometry.cells, dtype=float)
                        if getattr(term_rhs, field_name) is None
                        else np.asarray(
                            getattr(term_rhs, field_name), dtype=float
                        ).copy()
                    )
                    for field_name in packed_fields
                }
                for term_name, term_rhs in rhs_terms.items()
            },
            "cathode_diagnostics": cathode_diagnostics,
            "ignition_diagnostics": ignition_diagnostics,
            "gas_puff_diagnostics": self._gas_puff_diagnostic_snapshot(time=time),
        }
        # Only the DVM arm carries a transfer ledger, so only its runs carry
        # the per-save census record; a moment-model snapshot is unchanged.
        if self._dvm is not None:
            snapshot["dvm_ledger"] = self._dvm_ledger_sample(time=time)
        return snapshot

    def _gas_puff_diagnostic_snapshot(self, time):
        """Return the per-save EFFECTIVE gas-puff waveform record.

        Pure recording (see ``GAS_PUFF_DIAGNOSTIC_FIELDS``): it reads the very
        kwargs ``neutral_source_sink_rhs`` is handed at this instant and folds
        them through the same ``gas_puff_rate_profile``, so the recorded rate
        is the applied one by construction rather than a parallel formula that
        could drift. Nothing here mutates state or feeds an RHS row.
        """
        nk = self._neutral_source_kwargs(time=time)
        gated = bool(nk["gas_puff_enabled"])
        twin = bool(nk["twin_cathode"])
        if not gated:
            # The gate zeroes the whole puff term, so the applied rate is zero
            # regardless of where the waveform sits.
            return {name: 0.0 for name in GAS_PUFF_DIAGNOSTIC_FIELDS}
        puff = gas_puff_rate_profile(
            self._geometry, nk["S_gp"], nk["gas_puff_valves"],
            profile=nk["gas_puff_profile"], z_cm=nk["gas_puff_z_cm"],
            sigma_cm=nk["gas_puff_sigma_cm"], throw_cm=nk["gas_puff_throw_cm"],
            orifice_id_cm=nk["gas_puff_orifice_id_cm"],
            orifice_length_cm=nk["gas_puff_orifice_length_cm"],
            end=0, delivery_fraction=nk["gas_puff_delivery_fraction"],
        )
        if twin:
            puff = puff + gas_puff_rate_profile(
                self._geometry, nk["Twin_S_gp"], nk["gas_puff_valves"],
                profile=nk["gas_puff_profile"], z_cm=nk["gas_puff_z_cm"],
                sigma_cm=nk["gas_puff_sigma_cm"],
                throw_cm=nk["gas_puff_throw_cm"],
                orifice_id_cm=nk["gas_puff_orifice_id_cm"],
                orifice_length_cm=nk["gas_puff_orifice_length_cm"],
                end=-1, delivery_fraction=nk["gas_puff_delivery_fraction"],
            )
        return {
            "S_gp_sccm": float(nk["S_gp"]),
            "Twin_S_gp_sccm": float(nk["Twin_S_gp"]) if twin else 0.0,
            "puff_particles_per_s": float(
                np.sum(
                    puff
                    * np.asarray(self._geometry.neutral_volume_cm3, dtype=float)
                )
            ),
        }

    def _ignition_armed(self, time, phase):
        """Return whether breakdown-progress diagnostics are live right now.

        Structural, not clock-based: the cathode drive must actually be on
        (``cathode_coupling`` configured AND the phase switches enabling it --
        this is "beam-on") and the run must still be pre-ignition. Once a run
        reaches ``main_discharge`` -- or is already winding down after an abort
        -- the monitor is disarmed and its buffer cleared, so the stall
        detector cannot fire on a run that ignited.
        """
        if phase not in {"pre_breakdown", "breakdown"}:
            return False
        if self._t_ignition_abort is not None:
            return False
        return bool(self._cathode_phase_options(time=time)["cathode_enabled"])

    def _ignition_diagnostic_snapshot(
        self,
        time,
        phase,
        state,
        rhs_terms,
        cathode_diagnostics,
    ):
        """Return the per-save breakdown-progress record (see core/ignition).

        Pure reads: inventories from the accepted state, the electron power
        split from the RHS rows this snapshot already built, and the WP-D end
        ledger from the cathode diagnostics. Nothing here feeds an RHS row.
        Outside the armed phases every rate/power field is NaN-defaulted.
        """
        record = empty_ignition_diagnostics()
        armed = self._ignition_armed(time=time, phase=phase)
        record["armed"] = 1.0 if armed else 0.0

        plasma_active = np.asarray(self._geometry.plasma_active, dtype=float)
        plasma_volume = (
            np.asarray(self._geometry.plasma_volume_cm3, dtype=float)
            * plasma_active
        )
        neutral_volume = np.asarray(
            self._geometry.neutral_volume_cm3, dtype=float
        )
        N_plasma = float(np.sum(np.asarray(state.n, dtype=float) * plasma_volume))
        if state.nn_a is None:
            N_neutral = float(
                np.sum(np.asarray(state.nn, dtype=float) * neutral_volume)
            )
        else:
            # Two-zone: nn is the column density, nn_a the annulus (health.py
            # convention).
            column_volume = np.asarray(
                self._geometry.plasma_volume_cm3, dtype=float
            )
            N_neutral = float(
                np.sum(np.asarray(state.nn, dtype=float) * column_volume)
                + np.sum(
                    np.asarray(state.nn_a, dtype=float)
                    * (neutral_volume - column_volume)
                )
            )
        Ee_total = float(
            np.sum(np.asarray(state.Ee, dtype=float) * plasma_volume)
        )
        record["N_plasma"] = N_plasma
        record["N_neutral"] = N_neutral
        record["Ee_total_erg"] = Ee_total

        rates = self._ignition_monitor.record(
            time=float(time),
            N_plasma=N_plasma,
            N_neutral=N_neutral,
            Ee_total=Ee_total,
            armed=armed,
        )
        record["gamma_N_per_s"] = float(rates["gamma_N_per_s"])
        record["gamma_nn_per_s"] = float(rates["gamma_nn_per_s"])
        record["dEe_total_W"] = float(rates["dEe_total_erg_per_s"]) * 1.0e-7
        record["joint_negative"] = 1.0 if rates["joint_negative"] else 0.0

        if armed:
            total_W = 0.0
            for group, term_names in IGNITION_POWER_GROUPS.items():
                power = 0.0
                for term_name in term_names:
                    power += self._electron_term_power_W(
                        rhs_terms, term_name, plasma_volume
                    )
                record[group] = power
            for term_name in rhs_terms:
                total_W += self._electron_term_power_W(
                    rhs_terms, term_name, plasma_volume
                )
            record["P_transport_W"] = total_W - sum(
                record[group] for group in IGNITION_POWER_GROUPS
            )
            record["P_beam_end_loss_W"] = float(
                sum(
                    float(cathode_diagnostics.get(key, 0.0))
                    for key in IGNITION_BEAM_END_LOSS_KEYS
                )
            )

        stalled = bool(rates["stalled"]) and self._t_ignition_abort is None
        record["stalled"] = 1.0 if stalled else 0.0
        self._last_ignition_record = dict(record)
        if stalled:
            self._open_ignition_switch(
                time=float(time),
                reason="ignition_stalled",
                context=record,
            )
        return record

    @staticmethod
    def _electron_term_power_W(rhs_terms, term_name, plasma_volume):
        """Volume-integrated electron power [W] of one named RHS term."""
        term = rhs_terms.get(term_name)
        if term is None:
            return 0.0
        Ee = getattr(term, "Ee", None)
        if Ee is None:
            return 0.0
        return float(np.sum(np.asarray(Ee, dtype=float) * plasma_volume)) * 1.0e-7

    def _trajectory_result(
        self,
        saved,
        diagnostics,
        steps,
        run_start,
        timestep_rejection_events=None,
    ):
        cells = self._geometry.cells

        def stack(name):
            if not saved:
                return np.empty((0, cells), dtype=float)
            return np.stack([snapshot[name] for snapshot in saved])

        def stack_y():
            if not saved:
                return np.empty((0, len(self._y)), dtype=float)
            return np.stack([snapshot["y"] for snapshot in saved])

        rhs_terms = self._stack_trajectory_rhs_terms(saved=saved, cells=cells)
        cathode_diagnostics = self._stack_trajectory_cathode_diagnostics(
            saved=saved,
        )
        total_rhs = {
            field_name: sum(
                (
                    term_fields[field_name]
                    for term_fields in rhs_terms.values()
                ),
                np.zeros((len(saved), cells), dtype=float),
            )
            for field_name in (
                tuple(saved[0]["rhs_terms"][next(iter(rhs_terms))])
                if saved and rhs_terms
                else STATE_NAMES_1D
            )
        }
        electron_energy_terms_W_cm3 = {
            term_name: term_fields["Ee"] * 1.0e-7
            for term_name, term_fields in rhs_terms.items()
        }
        ion_energy_terms_W_cm3 = {
            term_name: term_fields["Ei"] * 1.0e-7
            for term_name, term_fields in rhs_terms.items()
        }

        result = SimpleNamespace(
            params=dict(self._input_dict),
            flags=dict(self._flags),
            # Which arithmetic produced this trajectory (D3/D4). Deliberately
            # NOT a params entry: the kernel path is a fact about the process,
            # not a physics parameter, and params_json is round-trip checked
            # against the constructed config. "pure" is the default and the
            # historical behaviour; an artifact with no compiled_kernels
            # attribute at all predates the selector and is also pure.
            compiled_kernels=KERNEL_PROVENANCE,
            time=np.asarray([snapshot["time"] for snapshot in saved], dtype=float),
            phase=np.asarray([snapshot["phase"] for snapshot in saved], dtype=object),
            phase_elapsed=np.asarray(
                [snapshot["phase_elapsed"] for snapshot in saved],
                dtype=float,
            ),
            phase_cathode_enabled=np.asarray(
                [snapshot["phase_cathode_enabled"] for snapshot in saved],
                dtype=float,
            ),
            phase_gas_puff_enabled=np.asarray(
                [snapshot["phase_gas_puff_enabled"] for snapshot in saved],
                dtype=float,
            ),
            phase_floating=np.asarray(
                [snapshot["phase_floating"] for snapshot in saved],
                dtype=float,
            ),
            y=stack_y(),
            n=stack("n"),
            nn=stack("nn"),
            M=stack("M"),
            momentum=stack("M"),
            Ee=stack("Ee"),
            Ei=stack("Ei"),
            u=stack("u"),
            Te=stack("Te"),
            Ti=stack("Ti"),
            pe=stack("pe"),
            pi=stack("pi"),
            p=stack("p"),
            z_cm=self._geometry.z_cm.copy(),
            cell_role=self._geometry.cell_role.copy(),
            length_cm=self._geometry.length_cm.copy(),
            Rp_cm=self._geometry.Rp_cm.copy(),
            Rm_cm=self._geometry.Rm_cm.copy(),
            plasma_volume_cm3=self._geometry.plasma_volume_cm3.copy(),
            neutral_volume_cm3=self._geometry.neutral_volume_cm3.copy(),
            volume_ratio=self._geometry.volume_ratio.copy(),
            plasma_active=self._geometry.plasma_active.copy(),
            rhs_terms=rhs_terms,
            cathode_diagnostics=cathode_diagnostics,
            ignition_diagnostics={
                name: np.asarray(
                    [snapshot["ignition_diagnostics"][name] for snapshot in saved],
                    dtype=float,
                )
                for name in IGNITION_DIAGNOSTIC_FIELDS
            },
            gas_puff_diagnostics={
                name: np.asarray(
                    [snapshot["gas_puff_diagnostics"][name] for snapshot in saved],
                    dtype=float,
                )
                for name in GAS_PUFF_DIAGNOSTIC_FIELDS
            },
            phase_events=self._phase_events(
                run_start=run_start,
                final_time=float(self._time),
            ),
            timestep_rejection_events=_timestep_rejection_event_arrays(
                timestep_rejection_events or []
            ),
            current_trigger_samples=_current_trigger_sample_arrays(
                self._current_trigger_samples
            ),
            total_rhs=total_rhs,
            floor_ledger=dict(self._floor_ledger),
            electron_energy_terms_W_cm3=electron_energy_terms_W_cm3,
            ion_energy_terms_W_cm3=ion_energy_terms_W_cm3,
            diagnostics=list(diagnostics),
            steps=int(steps),
            final_time=float(self._time),
            t_prebreakdown_trigger=(
                np.nan
                if self._t_prebreakdown_trigger is None
                else float(self._t_prebreakdown_trigger)
            ),
            t_breakdown_trigger=(
                np.nan
                if self._t_breakdown_trigger is None
                else float(self._t_breakdown_trigger)
            ),
        )
        if getattr(self, "_run_id", None) is not None:
            result.run_id = self._run_id
        if self._t_ignition_abort is not None:
            # Present ONLY on a run that aborted, so normal results (and their
            # saved HDF5 files) are unchanged. This is the event context the
            # artifact must carry: why the switch opened, in the electron
            # power ledger's own terms.
            result.ignition_abort = {
                "reason": str(self._ignition_abort_reason),
                "time_s": float(self._t_ignition_abort),
                "window_s": float(self._ignition_monitor.window_s),
                "rate_window_s": float(self._ignition_monitor.rate_window_s),
                "threshold_name": str(self._ignition_abort_threshold_name or ""),
                **{
                    key: float(value)
                    for key, value in (self._ignition_abort_context or {}).items()
                },
            }
        if saved and "M_n" in saved[0]:
            result.M_n = stack("M_n")
            result.u_n = stack("u_n")
        if saved and "En" in saved[0]:
            result.En = stack("En")
            result.Tn = stack("Tn")
            for name in HOT_CHANNEL_DIAGNOSTIC_FIELDS:
                setattr(result, name, stack(name))
            for name in IONIZATION_BIRTH_DEFICIT_DIAGNOSTIC_FIELDS:
                setattr(result, name, stack(name))
        if saved and "nn_a" in saved[0]:
            result.nn_a = stack("nn_a")
        if saved and "M_n_a" in saved[0]:
            result.M_n_a = stack("M_n_a")
            result.u_n_a = stack("u_n_a")
        # Present ONLY on a run that built the DVM arm. A moment-model result
        # carries no such attribute and its saved file no such group, so the
        # census is ABSENT rather than zero wherever it was never kept.
        if self._dvm is not None:
            result.dvm_transfer_ledger = self._dvm_ledger_census(saved)
            result.dvm_tick_count = int(self._dvm_tick_count)
        # Cathode-jet arming census, presence-gated on the CRITERION rather
        # than on either jet channel: a run that declared no criterion carries
        # no such attribute and its saved file no such group, so an
        # arm = 0 file is byte-unchanged by this existing. What it records
        # cannot be recovered from the trajectory -- the runner of the first
        # armed arms had to INFER the transitions from the En row -- because
        # the latch is solver state that no saved field is a function of.
        if self._jet_arming_active:
            result.jet_arming = {
                "arm_current_A": float(self._jet_arm_current_A),
                "disarm_current_A": float(self._jet_disarm_current_A),
                "censored_steps": int(self._jet_arming_censored_steps),
                "transitions": int(self._jet_arming_transitions),
                "last_transition_s": float(
                    self._jet_arming_last_transition_s
                ),
                "armed_at_end": bool(self._jet_armed),
            }
        result.atomic_rate_domain = _atomic_rate_domain(result)
        return result

    def _phase_events(self, run_start, final_time):
        run_start = float(run_start)
        final_time = float(final_time)
        events = []

        def append_event(time, phase, reason):
            time = float(time)
            if time < run_start - 1e-15 or time > final_time + 1e-15:
                return
            if events and abs(events[-1][0] - time) <= 1e-15:
                if events[-1][2] == "initial":
                    return
                events[-1] = (time, phase, reason)
                return
            events.append((time, phase, reason))

        append_event(run_start, self.phase_at_time(run_start), "initial")
        if not self._flags.get("Plasma"):
            # Same window the run loop actually delivers (the puff-duty fix).
            puff_on = self._equilibration_puff_on_duration()
            puff_reason = self._equilibration_puff_on_reason()
            tau_cycle = max(float(self._input_dict.get("tau_cycle")), 0.0)
            if tau_cycle <= 0.0:
                append_event(puff_on, "equilibrium_off", puff_reason)
                return _phase_event_arrays(events)

            cycle_index, _ = self._equilibration_cycle_position(run_start)
            cycle_start = cycle_index * tau_cycle
            while cycle_start <= final_time + 1e-15:
                cycle_end = cycle_start + tau_cycle
                if 0.0 < puff_on < tau_cycle:
                    append_event(
                        cycle_start + puff_on,
                        "equilibrium_off",
                        puff_reason,
                    )
                append_event(cycle_end, "equilibrium_puff", "tau_cycle")
                cycle_start = cycle_end
            return _phase_event_arrays(events)

        tau_prebreakdown = max(
            float(self._input_dict.get("tau_prebreakdown")),
            0.0,
        )
        tau_breakdown = max(float(self._input_dict.get("tau_breakdown")), 0.0)
        tau_discharge = max(float(self._input_dict.get("tau_discharge")), 0.0)
        tau_afterglow = max(float(self._input_dict.get("tau_afterglow")), 0.0)
        plasma_origin = self._plasma_phase_time_origin()
        if plasma_origin > 0.0:
            append_event(plasma_origin, "pre_breakdown", "tau_neutral_prebreakdown")

        abort = self._t_ignition_abort
        if self._phase_transition_mode() == "current":
            if self._t_prebreakdown_trigger is not None:
                append_event(
                    self._t_prebreakdown_trigger,
                    "breakdown",
                    "I_prebreakdown",
                )
            if abort is not None:
                append_event(abort, "afterglow", self._ignition_abort_reason)
                append_event(
                    abort + tau_afterglow,
                    "post_afterglow",
                    "tau_afterglow",
                )
                return _phase_event_arrays(events)
            if self._t_breakdown_trigger is not None:
                main_start = float(self._t_breakdown_trigger)
                append_event(main_start, "main_discharge", "I_breakdown")
                append_event(
                    main_start + tau_discharge,
                    "afterglow",
                    "tau_discharge",
                )
                append_event(
                    main_start + tau_discharge + tau_afterglow,
                    "post_afterglow",
                    "tau_afterglow",
                )
            return _phase_event_arrays(events)

        breakdown_start = plasma_origin + tau_prebreakdown
        main_start = breakdown_start + tau_breakdown
        if abort is not None:
            if tau_breakdown > 0.0 and breakdown_start < abort:
                append_event(breakdown_start, "breakdown", "tau_prebreakdown")
            append_event(abort, "afterglow", self._ignition_abort_reason)
            append_event(
                abort + tau_afterglow,
                "post_afterglow",
                "tau_afterglow",
            )
            return _phase_event_arrays(events)
        if tau_breakdown > 0.0:
            append_event(breakdown_start, "breakdown", "tau_prebreakdown")
            append_event(main_start, "main_discharge", "tau_breakdown")
        else:
            append_event(main_start, "main_discharge", "tau_prebreakdown")
        append_event(main_start + tau_discharge, "afterglow", "tau_discharge")
        append_event(
            main_start + tau_discharge + tau_afterglow,
            "post_afterglow",
            "tau_afterglow",
        )
        return _phase_event_arrays(events)

    def _stack_trajectory_rhs_terms(self, saved, cells):
        if not saved:
            return {}
        term_names = saved[0]["rhs_terms"].keys()
        field_names = tuple(
            saved[0]["rhs_terms"][next(iter(term_names))].keys()
        )
        return {
            term_name: {
                field_name: np.stack(
                    [
                        snapshot["rhs_terms"][term_name][field_name]
                        for snapshot in saved
                    ]
                )
                for field_name in field_names
            }
            for term_name in term_names
        }

    def _phase_info(self, time):
        time = max(float(time), 0.0)
        tau_discharge = max(float(self._input_dict.get("tau_discharge")), 0.0)
        if not self._flags.get("Plasma"):
            puff_on = self._equilibration_puff_on_duration()
            _, cycle_time = self._equilibration_cycle_position(time)
            if cycle_time < puff_on:
                return "equilibrium_puff", cycle_time
            return "equilibrium_off", cycle_time - puff_on

        tau_prebreakdown = max(
            float(self._input_dict.get("tau_prebreakdown")),
            0.0,
        )
        tau_breakdown = max(float(self._input_dict.get("tau_breakdown")), 0.0)
        tau_afterglow = max(float(self._input_dict.get("tau_afterglow")), 0.0)
        plasma_origin = self._plasma_phase_time_origin()
        if plasma_origin > 0.0 and time < plasma_origin:
            return "neutral_prebreakdown", time
        # Switch-open abort (ignition_stalled / prebreakdown_timeout): from the
        # abort instant the run is in the ordinary afterglow -- drive off,
        # cathode floating -- and then post_afterglow. Inert (None) on every
        # run that ignites, which is what keeps the golden bit-exact.
        abort = self._t_ignition_abort
        if abort is not None and time >= abort:
            post_afterglow_start = abort + tau_afterglow
            if time < post_afterglow_start:
                return "afterglow", time - abort
            return "post_afterglow", time - post_afterglow_start
        if self._phase_transition_mode() == "current":
            if self._t_breakdown_trigger is not None:
                main_start = self._t_breakdown_trigger
                afterglow_start = main_start + tau_discharge
                post_afterglow_start = afterglow_start + tau_afterglow
                if time < main_start:
                    if (
                        self._t_prebreakdown_trigger is not None
                        and time >= self._t_prebreakdown_trigger
                    ):
                        return "breakdown", time - self._t_prebreakdown_trigger
                    return "pre_breakdown", time - plasma_origin
                if time < afterglow_start:
                    return "main_discharge", time - main_start
                if time < post_afterglow_start:
                    return "afterglow", time - afterglow_start
                return "post_afterglow", time - post_afterglow_start
            if (
                self._t_prebreakdown_trigger is not None
                and time >= self._t_prebreakdown_trigger
            ):
                return "breakdown", time - self._t_prebreakdown_trigger
            return "pre_breakdown", time - plasma_origin

        breakdown_start = plasma_origin + tau_prebreakdown
        main_start = breakdown_start + tau_breakdown
        afterglow_start = main_start + tau_discharge
        post_afterglow_start = afterglow_start + tau_afterglow
        if time < breakdown_start:
            return "pre_breakdown", time
        if time < main_start:
            return "breakdown", time - breakdown_start
        if time < afterglow_start:
            return "main_discharge", time - main_start
        if time < post_afterglow_start:
            return "afterglow", time - afterglow_start
        return "post_afterglow", time - post_afterglow_start

    def _phase_switches(self, phase):
        discharge_phases = {"pre_breakdown", "breakdown", "main_discharge"}
        if phase == "neutral_prebreakdown":
            return {
                "cathode_enabled": False,
                "gas_puff_enabled": bool(
                    self._input_dict.get("gas_puff_enabled")
                ),
                "floating": False,
            }
        if phase == "equilibrium_puff":
            return {
                "cathode_enabled": False,
                "gas_puff_enabled": bool(
                    self._input_dict.get("gas_puff_enabled")
                ),
                "floating": False,
            }
        if phase == "equilibrium_off":
            return {
                "cathode_enabled": False,
                "gas_puff_enabled": False,
                "floating": False,
            }
        return {
            "cathode_enabled": (
                bool(self._flags.get("cathode_coupling"))
                and phase in discharge_phases
            ),
            "gas_puff_enabled": (
                bool(self._input_dict.get("gas_puff_enabled"))
                and (
                    phase in discharge_phases
                    # The square valve keeps delivering through its erf
                    # closing tail after the drive ends (~ms); the waveform
                    # envelope, not the phase switch, closes the flow.
                    or (
                        phase == "afterglow"
                        and self._input_dict.get("gas_puff_mode") == "square"
                    )
                )
            ),
            "floating": phase == "afterglow",
        }

    def _collector_surface_power_W(self, rhs_terms, derived):
        """Return the power [W] the plasma deposits on the floating collector.

        R5.4 (R3 tail): completes the power ledger with the collector
        surface-power line. The plasma-terminating boundary term
        (`characteristic_boundary`) removes
        electron (2Te sheath), ion internal, and reconstructed kinetic energy at
        the collector cell; the negative of that removal is the surface power the
        collector receives. Diagnostic-only (no state change). It is an ambient
        plasma Bohm-outflow loss -- regime-dependent (small in the high-density/
        detached ES runs, significant in low-puff/attached runs), independent of
        whether any beam survives downstream.
        """
        roles = np.asarray(self._geometry.cell_role)
        collector = roles == "collector"
        if not np.any(collector):
            return 0.0
        Vp = np.asarray(self._geometry.plasma_volume_cm3, dtype=float)
        u = np.asarray(derived.u, dtype=float)
        m = self._ion_mass_g
        total = 0.0
        for name in ("characteristic_boundary", "boundary_absorption"):
            term = rhs_terms.get(name)
            if term is None:
                continue
            Ee = np.asarray(term.Ee, dtype=float)
            Ei = np.asarray(term.Ei, dtype=float)
            M = np.zeros_like(Ee) if term.M is None else np.asarray(term.M, dtype=float)
            n = np.zeros_like(Ee) if term.n is None else np.asarray(term.n, dtype=float)
            dK = u * M - 0.5 * m * u**2 * n  # reconstructed kinetic removal
            p_cell = -(Ee + Ei + dK) * Vp * 1.0e-7  # erg/s -> W; sink -> surface gain
            total += float(np.sum(p_cell[collector]))
        return total

    def _cathode_diagnostic_snapshot(self, time=None):
        cells = self._geometry.cells
        cathode_phase = self._cathode_phase_options(time=time)
        # Discharge voltage over the save interval: the dt-weighted average
        # of the running \int V_dis dt since the previous trajectory save.
        # The raw last-step sample is dt-biased (saves land on dt-capped
        # steps that sample the knee sawtooth's low state; measured
        # 2026-07-21: 126 V saved plateau mean vs 151 V true average, with
        # V_b and the loop reconstruction both at 151 V), so the saved
        # trace averages the integral instead. The first save and repeated
        # save times fall back to the last-step value (0.0 at t=0).
        # Mutates the prev-save anchor: call once per trajectory save only.
        t_now = float(self._time if time is None else time)
        V_dis_save = float(self._circuit_V_dis_step)
        if self._circuit_V_dis_prev_save is not None:
            t_prev, int_prev = self._circuit_V_dis_prev_save
            if t_now > t_prev:
                V_dis_save = (
                    self._circuit_V_dis_time_integral - int_prev
                ) / (t_now - t_prev)
        self._circuit_V_dis_prev_save = (
            t_now,
            float(self._circuit_V_dis_time_integral),
        )
        diag = {
            "enabled": float(bool(self._flags.get("cathode_coupling"))),
            # Instantaneous emitter surface temperature [K]: the configured
            # T_s, or the evolving value under cathode_warming_model.
            "T_s_surface": float(
                self._cathode_Ts_K
                if self._cathode_Ts_K is not None
                else float(self._input_dict.get("T_s"))
            ),
            "configured": float(cathode_phase["configured"]),
            # Current-driven circuit state (0.0 under the voltage-driven
            # solver, whose loop current lives in source_I_tot).
            "circuit_I_loop": float(self._circuit_I_loop),
            # Discharge voltage [V] (the inductor's view): dt-weighted
            # average of the step-integrated V_dis over the save interval
            # (see above) -- the honest discharge-voltage trace, agreeing
            # with per-solve source_V_b and the loop reconstruction on the
            # plateau. 0.0 under voltage-driven. Runs saved before
            # 2026-07-21 store the biased last-step sample under this key
            # (~25 V low on the ES1 plateau).
            "circuit_V_dis_step": V_dis_save,
            # Running \int V_dis dt [V*s]: difference between saves
            # reproduces the dt-weighted average independently of the
            # snapshot cadence (and on pre-fix files).
            "circuit_V_dis_dt_integral": float(
                self._circuit_V_dis_time_integral
            ),
            # Vessel common-mode node. Present only when armed, so a run
            # without it saves exactly the diagnostic set it always did. This
            # is the PREDICTION CHANNEL: V_cm(t) is written, never scored.
            **(
                {}
                if self._vessel is None
                else {
                    "vessel_V_cm_V": float(self._vessel_V_cm),
                    "vessel_beam_climb_V": max(
                        float(self._vessel_V_cm), 0.0
                    ),
                    "vessel_I_e_wall_A": float(
                        self._vessel_wall_currents_A[0]
                    ),
                    "vessel_I_i_wall_A": float(
                        self._vessel_wall_currents_A[1]
                    ),
                    "vessel_I_leak_A": float(
                        self._vessel_wall_currents_A[2]
                    ),
                    "vessel_I_wall_net_A": float(
                        self._vessel_wall_currents_A[0]
                        - self._vessel_wall_currents_A[1]
                        - self._vessel_wall_currents_A[2]
                    ),
                    "vessel_Q_node_C": float(
                        self._vessel_charge_ledger_C["node"]
                    ),
                    "vessel_charge_residual_C": float(
                        self.vessel_charge_residual()[0]
                    ),
                }
            ),
            # Cumulative surface energy ledger [J]. The heater/ion/rad/emis/
            # cond rows are booked inside the power_balance warming branch
            # and stay zero otherwise; the presence-gated backscatter row is
            # booked on every accepted step the DVM cathode jet counted on,
            # warming model or not. See _cathode_energy_ledger_J.
            **{
                f"warming_E_{k}_J": float(v)
                for k, v in self._cathode_energy_ledger_J.items()
            },
            # The ANODE's cumulative surface energy book [J], PRESENCE-GATED on
            # the B4 anode jet: absent the channel these keys do not exist and
            # the saved diagnostic set is the one it always was. The anode has
            # no warming model, so unlike the cathode rows these are a pure
            # measurement -- what the wires received, and what left with the
            # backscatter.
            **(
                {}
                if self._anode_energy_ledger_J is None
                else {
                    f"anode_E_{k}_J": float(v)
                    for k, v in self._anode_energy_ledger_J.items()
                }
            ),
            # Surface-state coverage and the effective work function
            # (cathode_surface_model; 1.0 / static phi_wf when off).
            "surface_theta": float(
                self._cathode_theta
                if self._cathode_theta is not None
                else 1.0
            ),
            "phi_wf_eff": float(
                self._cathode_phi_wf_eff()
                if self._cathode_theta is not None
                else float(self._input_dict.get("phi_wf"))
            ),
            "phase_enabled": float(cathode_phase["cathode_enabled"]),
            "rhs_enabled": float(cathode_phase["cathode_enabled"]),
            "solve_enabled": float(cathode_phase["solve_enabled"]),
            "floating": float(cathode_phase["floating"]),
            "twin_cathode": float(bool(self._flags.get("TwinCathode"))),
            "has_solution": 0.0,
            "has_twin_solution": 0.0,
            "x0_next": np.nan,
            "x0_twin_next": np.nan,
            "v_beam": np.zeros(cells, dtype=float),
            "n_beam": np.zeros(cells, dtype=float),
            "beam_cross": np.zeros(cells, dtype=float),
            "n_beam_ion": np.zeros(cells, dtype=float),
            "A_ion_beam": np.zeros(cells, dtype=float),
            "l_b": np.zeros(cells, dtype=float),
            "p_beam": np.zeros(cells, dtype=float),
            "l_b_profile": np.zeros(cells, dtype=float),
            "l_b_profile_twin": np.zeros(cells, dtype=float),
            # --- CSDA deposition channel breakout (diagnostic only) ---------
            # Per-cell power [W] delivered to the electron pool by each of the
            # four physically distinct channels the module's lumped
            # ``plasma_heating_erg_s`` bank contains, summed over the active
            # rays exactly as ``beam_power_deposition`` sums them. Present
            # (all zero) on every run; ``beam_csda_active`` is the marker that
            # says the numbers are real -- a Beer-Lambert run books its
            # deposition through a different path and leaves these at zero.
            # Under ``beam_deposition_smoothing_cm > 0`` the RHS profile is
            # redistributed but its total is conserved, so the COLUMN SUM of
            # these arrays still matches the smoothed term; the per-cell
            # profile is the unsmoothed one.
            "beam_csda_active": 0.0,
            "beam_heat_coulomb_W": np.zeros(cells, dtype=float),
            "beam_heat_anomalous_W": np.zeros(cells, dtype=float),
            "beam_heat_secondary_W": np.zeros(cells, dtype=float),
            "beam_heat_terminal_W": np.zeros(cells, dtype=float),
            # K6: the QL tail walkers' own share of the three shared banks
            # they write into under
            # ``heating_anomalous_tail_ionization="on"`` -- the pairs they
            # birth [1/s], the potential they invest and the line power they
            # radiate [W]. Diagnostic splits, not extra sources: the events
            # are already inside ``beam_ionization_birth`` and the energies
            # inside the cost and radiation sinks. Identically zero under the
            # default ``"off"``, so every run to date reads zero; runs saved
            # before 2026-08-06 lack the datasets and readers must default
            # them. With ``beam_heat_anomalous_W`` (which under ``"on"``
            # carries the walkers' whole heat delivery) and the tail end
            # ledger, these close the tail channel's branching from a saved
            # file alone.
            "beam_tail_ionization_events_per_s": np.zeros(cells, dtype=float),
            "beam_tail_ionization_cost_W": np.zeros(cells, dtype=float),
            "beam_tail_radiated_W": np.zeros(cells, dtype=float),
            # K7b band exposure [W], summed over the active rays. Under phi_c
            # keying E_tail follows the live cathode drop, so one run visits
            # all three bands: below the lowest inelastic threshold the
            # ionizing march REVERTS to the energy-only walk (exact -- no
            # channel is open there), above the <W_sec> crossing it runs under
            # the measured <= 2.0% depth-1 understatement, and in between
            # nothing is out of band. These say which, per frame, so the foot
            # reversion and the above-bar exposure are readable from a saved
            # trajectory instead of having to be inferred from phi_c.
            # ``_power_W`` are the tail power in each regime and
            # ``_fraction`` the sub-threshold share of the total (NaN when no
            # tail power was launched). Identically zero under
            # ``heating_anomalous_transport="local"`` (the default); runs
            # saved before 2026-08-06 lack the datasets and readers must
            # default them.
            "beam_tail_power_W": 0.0,
            "beam_tail_sub_threshold_power_W": 0.0,
            "beam_tail_above_bar_power_W": 0.0,
            "beam_tail_sub_threshold_fraction": np.nan,
        }
        if self._cathode_f_em is not None:
            # Emitting-area percolation: PRESENCE-GATED, so an unarmed run's
            # saved diagnostic structure is byte-identical to before the
            # closure existed. Saved because the closure's own state is what
            # says whether an arm's f_em actually advanced -- a frozen
            # fraction makes every number the arm produces a mean-field
            # number under another name, and that has to be checkable from a
            # saved trajectory rather than only from a live solver.
            diag["cathode_emitting_area_fraction"] = float(self._cathode_f_em)
        if self._coverage is not None:
            # Clumpy-plasma coverage closure: PRESENCE-GATED so the saved
            # diagnostic structure of every mean-field run -- the golden
            # included -- is byte-identical to before the closure existed.
            # The scalar summary keeps its name and its meaning as "the
            # coverage of the column", now as the volume-weighted mean of the
            # z-resolved field; the profile itself is saved beside it so the
            # axial structure the closure exists to carry is readable from a
            # saved trajectory rather than only from a live solver.
            diag["coverage_fraction"] = float(self.coverage_fraction())
            diag["coverage_fraction_profile"] = self._coverage_f.copy()
            diag["coverage_fraction_min"] = float(np.min(self._coverage_f))
            diag["coverage_fraction_max"] = float(np.max(self._coverage_f))
            diag["coverage_nn_deficit_max"] = float(
                np.max(self._coverage_deficit)
            )
            diag["coverage_nn_column_min"] = float(
                np.min(
                    np.maximum(
                        np.asarray(self.state.nn, dtype=float)
                        - self._coverage_deficit,
                        self._floors["nn"],
                    )
                )
            )
        if self._electron_drift is not None:
            # The drift operator's named rows, PRESENCE-GATED so an unarmed
            # run's saved diagnostic structure -- the golden included -- is
            # byte-identical to before the operator existed. They ride the
            # electrode-diagnostics channel because that is where the current
            # they are built from already lives (``circuit_I_loop``), so a
            # reader can check a row against its own input in one group.
            #
            # The four per-cell power rows [W] are the operator's own four
            # terms and sum to its total, so the volume identity is checkable
            # from a saved trajectory rather than only from a live solver.
            # ``edt_inplasma_emf_V`` is the V_dis partition member: the
            # in-plasma EMF the drift works against, W_EMF per ampere.
            # ``edt_total_W`` is the per-step ledger total over the operator's
            # support.
            #
            # Zeros, not absence, on a save whose RHS evaluation found no
            # solve to read a current from: the row structure must not move
            # between saves of one run, and "no current" is a measurement.
            rows = self._electron_drift_rows
            zeros = np.zeros(self._geometry.cells, dtype=float)
            for name in ELECTRON_DRIFT_DIAGNOSTIC_ROWS:
                diag[name] = (
                    zeros.copy()
                    if rows is None
                    else np.asarray(rows[name], dtype=float).copy()
                )
            for name in ELECTRON_DRIFT_DIAGNOSTIC_SCALARS:
                diag[name] = 0.0 if rows is None else float(rows[name])
        if self._anode_sheath_full_debit:
            # anode_sheath_full_debit census, PRESENCE-GATED so an unarmed
            # run's saved diagnostic structure -- the golden included -- is
            # byte-identical to before the flag existed. Cumulative count of
            # ACCEPTED steps whose sheath solve returned an electron-ATTRACTING
            # anode (phi_a <= 0), where the flag books the thermal-only debit
            # and the bank pays the fall, plus the solver time of the last such
            # step (NaN while none has occurred). Both are as-of-this-save
            # values, so an armed run's transcript can state how much of it
            # took that branch and when it stopped.
            diag["anode_attracting_steps"] = float(self._anode_attracting_steps)
            diag["anode_attracting_last_time_s"] = float(
                self._anode_attracting_last_time_s
            )
        if self._plateau_multigroup:
            # Multi-group plateau closure, PRESENCE-GATED so an unarmed run's
            # saved diagnostic structure -- the golden included -- is
            # byte-identical to before the closure existed.
            #
            # ``beam_plateau_wave_power_W`` is the WAVE/BULK heir of the
            # withheld QL bank, banked as local bulk heat in the extraction
            # cells. It is ALREADY inside beam_heat_anomalous_W (it is the
            # anomalous channel's local delivery); reported apart so the
            # derived split of the bank -- this plus the streaming
            # beam_tail_power_W -- is readable per frame rather than inferred.
            #
            # ``plateau_edge_clamped_*`` are the clamp census: the cumulative
            # count of ACCEPTED steps on which some end's plateau-edge solve
            # hit the inelastic floor and was clamped to it, and the solver
            # time of the last such step (NaN while none has occurred). The
            # edge is a state-dependent solve, so a clamped frame is running a
            # different spectrum -- never silent.
            diag["beam_plateau_wave_power_W"] = 0.0
            diag["plateau_edge_clamped_steps"] = float(
                self._plateau_edge_clamped_steps
            )
            diag["plateau_edge_clamped_last_time_s"] = float(
                self._plateau_edge_clamped_last_time_s
            )
        for prefix in ("source", "end"):
            # Per-ray exit ledger [W]: power the anode mesh intercepts at the
            # anode-face crossing, and power streaming out of the far end.
            # Both are computed by the CSDA ray today and had no consumer and
            # no saved record; they are the unbooked bypass termination.
            diag[f"{prefix}_beam_anode_intercepted_W"] = 0.0
            diag[f"{prefix}_beam_transmitted_W"] = 0.0
            diag[f"{prefix}_beam_transmitted_flux_per_s"] = 0.0
            # WP-D end ledger [W]: beam energy that LEAVES the column through
            # each axial end -- product walks that escape without
            # thermalizing, plus the transmitted primary's Gamma_t*E_t (which
            # the module has computed since B1 and nothing ever banked). This
            # is a loss channel, not a heating one: it is never added to
            # plasma_heating_erg_s and never enters an RHS row. Identically
            # zero under beam_product_transport="local" (the default), so on
            # every run to date these read 0.0; runs saved before 2026-07-28
            # lack the datasets entirely and readers must default them.
            diag[f"{prefix}_beam_end_loss_low_W"] = 0.0
            diag[f"{prefix}_beam_end_loss_high_W"] = 0.0
            # WP-E tail end ledger [W]: QL heating that leaves the column as
            # fast tail electrons through each axial end without thermalizing.
            # A SIBLING of the WP-D pair above, kept separate so the product
            # ledger keeps its measured meaning while the two closures switch
            # independently. Same status: a loss channel, never in an RHS row.
            # Identically zero under heating_anomalous_transport="local" (the
            # default), so on every run to date these read 0.0; runs saved
            # before 2026-08-02 lack the datasets and readers must default.
            #
            # THE TOTAL TAIL ESCAPE IS THE SUM OF ALL FOUR ROWS (both prefixes,
            # both faces) -- that is the quantity the per-ray identity
            #   P_QL = heating_anomalous + ionization_cost_tail + radiated_tail
            #          + end_loss_tail_low + end_loss_tail_high
            # complements, and the ONLY form that is correct for every keying.
            # The high row alone is the total only while the low row is zero,
            # which is a property of the KEYING and not an invariant of the
            # closure: under heating_anomalous_tail_energy_keying="phi_c" a
            # walker is born at E_tail = f*e*phi_c with f <= 1 against a
            # cathode-face reflection threshold of e*phi_c and only ever loses
            # energy, so it can never reach that face at or above the threshold
            # and the low row is exactly 0.0; under "fixed" the rung is
            # decoupled from phi_c, and on any frame whose drive sits below the
            # rung the sheath no longer repels the walkers and the low row
            # fills. Measured: covdisc_fixed (75 eV rung) books up to 0.486 of
            # the launched tail power through the low row on the 43 frames
            # where phi_c fell to 44.6-71.7 V, while the identity itself still
            # closes to 8.1e-16 on every frame. A reader that names only the
            # high row understates the escape there by that much.
            diag[f"{prefix}_beam_end_loss_tail_low_W"] = 0.0
            diag[f"{prefix}_beam_end_loss_tail_high_W"] = 0.0
            # Item-35 gap-survival ledger: three views of the fraction of the
            # emitted beam that crosses the cathode-anode gap, which must
            # agree. ``_probe`` is the gap-clipped probe that feeds sigma_eff,
            # ``_ray`` is the deposition ray's own breakout (independent of
            # the probe), ``_circuit`` is what the Beer-Lambert bypass books
            # from the sigma_eff written for the next solve. NaN where no CSDA
            # ray ran; runs saved before 2026-07-28 have none of the three
            # datasets (readers must default).
            diag[f"{prefix}_beam_gap_survival_probe"] = np.nan
            diag[f"{prefix}_beam_gap_survival_ray"] = np.nan
            diag[f"{prefix}_beam_gap_survival_circuit"] = np.nan
            if self._plateau_multigroup:
                # The plateau edge E_1 [eV] THIS end's extraction solved, and
                # -1.0 on a frame where it was clamped to the inelastic floor
                # (0.0 otherwise). Per end, because the edge is a property of
                # the extraction: both ends solve their own. NaN where no CSDA
                # ray ran. Presence-gated with the closure.
                diag[f"{prefix}_beam_plateau_edge_eV"] = np.nan
                diag[f"{prefix}_beam_plateau_edge_clamped"] = np.nan
        for prefix in ("source", "end"):
            diag[f"{prefix}_regime"] = "none"
            for key in _CATHODE_RESULT_KEYS:
                diag[f"{prefix}_{key}"] = np.nan
            diag[f"{prefix}_long_mfp"] = np.nan
            # At-cap regime flag: 1.0 where the exported ``phi_c`` is the
            # ``cathode_phi_c_cap_V`` ceiling BOUND rather than a free root
            # of the current-matching solve, 0.0 where it is a free root,
            # NaN where no solve ran. Both routes into the ceiling land
            # here -- the bracket ladder reaching the cap before it can
            # carry the imposed current, and the returned-root clip
            # (2026-08-09) that re-tests the located J-root against the cap
            # -- because both leave the solve in
            # ``regime = "capability_limited"``, which is exactly the tag
            # the module's own escape invariant keys on. Read it with any
            # ``phi_c``-derived quantity (notably ``E_tail`` under
            # ``heating_anomalous_tail_energy_keying = "phi_c"``): on a
            # flagged frame that quantity is riding a numerical regime
            # guard, not a device-sustained drop. Diagnostic only -- it is
            # derived from the regime string the solve already returns, so
            # nothing here is recomputed and no exported value moves. Runs
            # saved before 2026-08-09 lack the datasets and readers must
            # default them.
            diag[f"{prefix}_phi_c_at_cap"] = np.nan

        cathode_solve = self._cathode_solve
        if (
            not cathode_phase["solve_enabled"]
            or cathode_solve is None
            or cathode_solve.beam_result is None
        ):
            return diag

        beam_result = cathode_solve.beam_result
        diag["has_solution"] = 1.0
        diag["x0_next"] = _finite_or_nan(cathode_solve.x0_next)
        diag["x0_twin_next"] = _finite_or_nan(cathode_solve.x0_twin_next)
        for name in (
            "v_beam",
            "n_beam",
            "beam_cross",
            "n_beam_ion",
            "A_ion_beam",
            "l_b",
            "p_beam",
            "l_b_profile",
            "l_b_profile_twin",
        ):
            diag[name] = np.asarray(getattr(beam_result, name), dtype=float).copy()
        # Which circuit solve produced these results. The dispatch in
        # ``physics/cathode.py`` keys on exactly this flag: a floating phase
        # takes the voltage-driven ``circuit.solve``, everything else the
        # current-driven one, and the R3.2 audit set is populated only by the
        # latter. Indexed, not ``.get``-defaulted: a snapshot that reaches
        # here has run a solve, so a missing key is a bug to hear about.
        current_driven = not bool(cathode_solve.metadata["floating"])
        self._copy_cathode_result_diagnostics(
            diag=diag,
            prefix="source",
            result=beam_result.result,
            current_driven=current_driven,
        )
        if beam_result.result_twin is not None:
            diag["has_twin_solution"] = 1.0
            self._copy_cathode_result_diagnostics(
                diag=diag,
                prefix="end",
                result=beam_result.result_twin,
                current_driven=current_driven,
            )
        self._copy_beam_deposition_diagnostics(diag, cathode_solve)
        return diag

    @staticmethod
    def _copy_beam_deposition_diagnostics(diag, cathode_solve):
        """Record the CSDA ray's internal channels (diagnostic only).

        Reads the deposition results the cathode solve already produced --
        nothing here is recomputed and nothing feeds the RHS, so the saved
        trajectory is unchanged apart from the added datasets. Erg/s -> W.
        """
        prefixes = {0: "source", -1: "end"}
        for end, entry in (
            getattr(cathode_solve, "beam_gap_ledger", None) or {}
        ).items():
            if entry is None:
                continue
            prefix = prefixes[int(end)]
            diag[f"{prefix}_beam_gap_survival_probe"] = float(entry[0])
            diag[f"{prefix}_beam_gap_survival_ray"] = float(entry[1])
            diag[f"{prefix}_beam_gap_survival_circuit"] = float(entry[2])
        deposition = getattr(cathode_solve, "beam_deposition", None)
        if not deposition:
            return
        for end, dep in deposition.items():
            if dep is None:
                continue
            diag["beam_csda_active"] = 1.0
            diag["beam_heat_coulomb_W"] += dep.heating_coulomb_erg_s * 1.0e-7
            diag["beam_heat_anomalous_W"] += (
                dep.heating_anomalous_erg_s * 1.0e-7
            )
            diag["beam_heat_secondary_W"] += (
                dep.heating_secondary_erg_s * 1.0e-7
            )
            diag["beam_heat_terminal_W"] += dep.heating_terminal_erg_s * 1.0e-7
            diag["beam_tail_ionization_events_per_s"] += (
                dep.ionization_events_tail
            )
            diag["beam_tail_ionization_cost_W"] += (
                dep.ionization_cost_tail_erg_s * 1.0e-7
            )
            diag["beam_tail_radiated_W"] += dep.radiated_tail_erg_s * 1.0e-7
            diag["beam_tail_power_W"] += (
                float(dep.tail_power_erg_s) * 1.0e-7
            )
            diag["beam_tail_sub_threshold_power_W"] += (
                float(dep.tail_sub_threshold_power_erg_s) * 1.0e-7
            )
            diag["beam_tail_above_bar_power_W"] += (
                float(dep.tail_above_bar_power_erg_s) * 1.0e-7
            )
            if "beam_plateau_wave_power_W" in diag:
                # Presence-gated with the closure that fills it (the key is
                # seeded only on an armed run), so this line cannot add a
                # dataset to an unarmed run's saved structure.
                diag["beam_plateau_wave_power_W"] += (
                    float(dep.plateau_wave_power_erg_s) * 1.0e-7
                )
            prefix = prefixes[int(end)]
            diag[f"{prefix}_beam_anode_intercepted_W"] = (
                float(dep.anode_intercepted_erg_s) * 1.0e-7
            )
            diag[f"{prefix}_beam_transmitted_flux_per_s"] = float(
                dep.transmitted_flux
            )
            diag[f"{prefix}_beam_transmitted_W"] = (
                float(dep.transmitted_flux)
                * float(dep.transmitted_energy_eV)
                * ev_to_erg
                * 1.0e-7
            )
            diag[f"{prefix}_beam_end_loss_low_W"] = (
                float(dep.end_loss_low_erg_s) * 1.0e-7
            )
            diag[f"{prefix}_beam_end_loss_high_W"] = (
                float(dep.end_loss_high_erg_s) * 1.0e-7
            )
            diag[f"{prefix}_beam_end_loss_tail_low_W"] = (
                float(dep.end_loss_tail_low_erg_s) * 1.0e-7
            )
            diag[f"{prefix}_beam_end_loss_tail_high_W"] = (
                float(dep.end_loss_tail_high_erg_s) * 1.0e-7
            )
        for end, entry in (
            getattr(cathode_solve, "beam_plateau_edge", None) or {}
        ).items():
            # The multi-group plateau edge this end's extraction solved, and
            # its clamp verdict. Read off the cathode solve rather than the
            # deposition result because the edge belongs to the EXTRACTION --
            # one solve, shared by that end's ray and both halves of a
            # clumping split.
            prefix = prefixes[int(end)]
            diag[f"{prefix}_beam_plateau_edge_eV"] = float(entry[0])
            diag[f"{prefix}_beam_plateau_edge_clamped"] = float(entry[1])
        # K7b: the sub-threshold SHARE, formed once the per-end powers are
        # summed. NaN rather than 0 where no tail power was launched at all --
        # a frame with no QL drive has no fraction, and 0.0 there would read
        # as "fully in band".
        if diag["beam_tail_power_W"] > 0.0:
            diag["beam_tail_sub_threshold_fraction"] = (
                diag["beam_tail_sub_threshold_power_W"]
                / diag["beam_tail_power_W"]
            )

    def _copy_cathode_result_diagnostics(
        self, diag, prefix, result, current_driven
    ):
        """Copy one ``SolverResult`` onto the cathode-diagnostics snapshot.

        ``current_driven`` says which solve produced ``result``. On the
        voltage-driven (floating) solve the members of
        :data:`_CURRENT_DRIVEN_ONLY_CATHODE_KEYS` are never assigned, so they
        are exported as NaN rather than as the dataclass default zero: a
        reader must be able to tell "this path does not compute the quantity"
        from "the quantity is zero here".
        """
        if result is None:
            return
        diag[f"{prefix}_regime"] = str(result.regime)
        for key in _CATHODE_RESULT_KEYS:
            if not current_driven and key in _CURRENT_DRIVEN_ONLY_CATHODE_KEYS:
                diag[f"{prefix}_{key}"] = np.nan
                continue
            diag[f"{prefix}_{key}"] = float(getattr(result, key))
        diag[f"{prefix}_long_mfp"] = float(bool(result.long_mfp))
        # See the default-seeding block in _cathode_diagnostic_snapshot.
        diag[f"{prefix}_phi_c_at_cap"] = float(
            str(result.regime) == "capability_limited"
        )

    def _stack_trajectory_cathode_diagnostics(self, saved):
        if not saved:
            return {}
        names = saved[0]["cathode_diagnostics"].keys()
        return {
            name: np.stack(
                [snapshot["cathode_diagnostics"][name] for snapshot in saved]
            )
            for name in names
        }

    def _kinetic_source_channel_rows(self, boundary, reaction_terms, anode):
        """Return the boundary-inflow source rows in atoms/s per cell.

        The four channels a kinetic arm sources from a PLASMA-side booking:
        the two wall-return faces, read from whichever boundary operator is
        terminating the plasma and placed in the cells that operator drained,
        resolved by role; the volumetric recombination birth; and the anode
        return, which the fluid books into the annulus wherever a resolved
        mesh gives it one. Each row is the term's own neutral row times the
        zone volume that row was written against, so the row IS a particle
        rate and its integral over an interval is particles.

        One expression, two consumers: the tick-time snapshot
        (:meth:`_kinetic_channel_rates`) and the transient arm's counted
        handshake (:meth:`_dvm_source_channel_rows`). The counted path is
        therefore the time integral of exactly the quantity the snapshot
        samples, rather than a second formula for the same channel.
        """
        V_col, V_ann = self._zone_volumes
        recycle = np.clip(boundary.nn, 0.0, None) * V_col
        cath_cells = np.zeros(self._geometry.cells)
        coll_cells = np.zeros(self._geometry.cells)
        for role, target in (
            ("cathode", cath_cells),
            ("collector", coll_cells),
        ):
            for cell in self._recycle_cells.get(role, ()):
                target[cell] = recycle[cell]
        rec_cells = np.clip(
            reaction_terms["recombination_rad_loss"].nn
            + reaction_terms["recombination_3b_loss"].nn,
            0.0,
            None,
        ) * V_col
        an_gain = np.clip(anode.nn, 0.0, None) * V_col
        if anode.nn_a is not None:
            an_gain = an_gain + np.clip(anode.nn_a, 0.0, None) * V_ann
        return {
            "cathode_face": cath_cells,
            "collector_face": coll_cells,
            "recombination": rec_cells,
            "anode": an_gain,
        }

    def _kinetic_channel_rates(self, state, derived, time):
        """Return the current source-channel rates and shapes (K4a-t).

        Cheap (pure numpy on the live fields); called every target update
        so the kinetic targets track source transients continuously --
        the recycle channels collapse WITH the plasma, which is what
        removes the afterglow flood.

        The wall-return channels are read from the boundary term that is
        ACTUALLY removing the plasma -- the characteristic ghost-cell flux --
        and from its own live cells, resolved by role. The recycled quantity
        IS the ``nn`` row the fluid path would have applied itself had the arm
        not superseded it, so what the arm re-injects equals what the boundary
        removed, per face, to roundoff.

        ``cath``/``coll`` are the per-surface totals (the kinetic steady arm
        needs a magnitude per channel); ``cath_cells``/``coll_cells`` place
        the same particles on the grid, which is what the transient arm
        deposits into.
        """
        geometry = self._geometry
        boundary = self.characteristic_boundary_rhs(state=state)
        reaction_terms = self.reaction_rhs_terms(state=state)
        an = self.anode_collection_rhs(state=state)
        rows = self._kinetic_source_channel_rows(boundary, reaction_terms, an)
        cath_cells = rows["cathode_face"]
        coll_cells = rows["collector_face"]
        rec_cells = rows["recombination"]
        an_gain = rows["anode"]
        src_kwargs = self._neutral_source_kwargs(time=time)
        puff_cells = np.zeros(geometry.cells)
        if src_kwargs["gas_puff_enabled"]:
            puff_cells = gas_puff_rate_profile(
                geometry,
                src_kwargs["S_gp"],
                src_kwargs["gas_puff_valves"],
                profile=src_kwargs["gas_puff_profile"],
                z_cm=src_kwargs["gas_puff_z_cm"],
                sigma_cm=src_kwargs["gas_puff_sigma_cm"],
                throw_cm=src_kwargs["gas_puff_throw_cm"],
                orifice_id_cm=src_kwargs["gas_puff_orifice_id_cm"],
                orifice_length_cm=src_kwargs["gas_puff_orifice_length_cm"],
                delivery_fraction=src_kwargs["gas_puff_delivery_fraction"],
            ) * np.asarray(geometry.neutral_volume_cm3, dtype=float)
        return {
            "cath": float(np.sum(cath_cells)),
            "coll": float(np.sum(coll_cells)),
            "cath_cells": cath_cells,
            "coll_cells": coll_cells,
            "puff": puff_cells,
            "rec": rec_cells,
            "anode": an_gain,
        }

    def _kinetic_absorption_fields(self, state, derived):
        n_safe = np.maximum(state.n, 1e6)
        Te_safe = np.maximum(derived.Te, 0.2)
        rates = he_rates(n_safe, Te_safe, ("scd",))
        nu_ion = state.n * rates["scd"]
        nu_cx = state.n * charge_ex_react(
            np.maximum(derived.Ti, 0.05), "He"
        )
        return np.asarray(nu_ion, dtype=float), np.asarray(nu_cx, dtype=float)

    def _kinetic_refresh(self, time):
        """Recompute the per-channel unit-rate responses (K4a-t).

        One engine solve per source channel (the kinetic steady solve is
        LINEAR in the sources at frozen plasma), so between refreshes the
        targets are formed as sum(rate_k * G_k) with the LIVE rates. The
        channel shapes (puff profile, recombination profile, anode split)
        are frozen here and rescaled by magnitude in between.
        """
        state = self.state
        derived = self.derived
        geometry = self._geometry
        kin = self._kinetic
        nu_ion, nu_cx = self._kinetic_absorption_fields(state, derived)
        T_s = float(self._input_dict.get("T_s"))
        anode_faces = np.asarray(
            getattr(geometry, "anode_face_indices", ()), dtype=int
        )
        bg = {
            "z_edges": np.concatenate(
                ([0.0], np.cumsum(geometry.length_cm))
            ),
            "Rp": np.asarray(geometry.Rp_cm, dtype=float),
            "Rm": np.asarray(geometry.Rm_cm, dtype=float),
            "nu_ion": nu_ion,
            "nu_cx": nu_cx,
            "Ti": np.asarray(derived.Ti, dtype=float),
            "u": np.asarray(derived.u, dtype=float),
            "T_s": T_s,
            "S_pump_L": float(self._input_dict.get("S_pump_L")),
            "S_pump_R": float(self._input_dict.get("S_pump_R")),
            "eta": float(self._input_dict.get("eta")),
            "mesh_edge": int(anode_faces[0]) if anode_faces.size else -999,
            "sources": {},
        }
        if getattr(kin, "grid", None) is None:
            # freeze one generous shared grid for the whole run: the
            # compiled kernels bind to it, and it must hold any discharge
            # state (10 eV CX tails + 2e6 cm/s drifts)
            Ti_cap = max(float(np.max(derived.Ti)), 10.0)
            u_cap = max(float(np.max(np.abs(derived.u))), 2.0e6)
            vmax = 4.0 * np.sqrt(Ti_cap * _KIN_EV / _KIN_M_HE) + 1.5 * u_cap
            v_fine = 0.25 * np.sqrt(_KIN_KB * 300.0 / _KIN_M_HE)
            kin.grid = _KineticVGrid(vmax, vmax, kin.nvz, kin.nvp, v_fine)
        jump = KN2ZoneJump(
            bg, nvz=kin.nvz, nvp=kin.nvp, verbose=False, max_gen=600,
            grid=kin.grid,
        )
        if kin.engine is None:
            kin.engine = KineticEngineFast(jump)
        else:
            kin.engine.j = jump
        g = jump.g
        nz = jump.nz
        rates_now = self._kinetic_channel_rates(state, derived, time)
        zero_Sc = np.zeros((nz, g.nvz, g.nvp))
        zero_in = np.zeros((g.nvz, g.nvp))
        zero_wall = np.zeros(nz)

        def shape_of(cells):
            total = float(np.sum(cells))
            if total <= 0:
                return None, 0.0
            return np.asarray(cells, dtype=float) / total, total

        puff_shape, _ = shape_of(rates_now["puff"])
        rec_shape, _ = shape_of(rates_now["rec"])
        anode_shape, _ = shape_of(rates_now["anode"])
        responses = {}
        shapes = {"puff": puff_shape, "rec": rec_shape, "anode": anode_shape}
        for name in ("cath", "coll", "puff", "rec", "anode"):
            Sc = zero_Sc.copy()
            Fc_in_L = zero_in
            Fc_in_R = zero_in
            wall0 = zero_wall.copy()
            if name == "cath":
                Fc_in_L = _kinetic_inflow(
                    1.0, g.half_flux_spectrum(T_s, +1), jump.A_col[0], g
                )
            elif name == "coll":
                Fc_in_R = _kinetic_inflow(
                    1.0, g.half_flux_spectrum(300.0, -1), jump.A_col[-1], g
                )
            elif name == "puff":
                if puff_shape is None:
                    continue
                wall0 = puff_shape.copy()
            elif name == "rec":
                if rec_shape is None:
                    continue
                for i in np.flatnonzero(rec_shape > 0):
                    Sc[i] = Sc[i] + (
                        rec_shape[i] / jump.V_col[i]
                    ) * jump.M_cx[i]
            elif name == "anode":
                if anode_shape is None or not anode_faces.size:
                    continue
                for i in np.flatnonzero(anode_shape > 0):
                    sign = -1 if i < bg["mesh_edge"] else +1
                    Sc[i] = Sc[i] + (
                        anode_shape[i] / jump.V_col[i]
                    ) * g.half_flux_spectrum(300.0, sign)
            res = kin.engine.solve(Sc, Fc_in_L, Fc_in_R, wall0)
            responses[name] = {
                "col": res["nn_col"],
                "ann": res["nn_ann"],
                "arr": res["ann_arrival"],
            }
        kin.responses = responses
        kin.shapes = shapes
        kin.nu_ref = nu_ion
        kin.next_refresh_s = float(time) + kin.refresh_s
        self._kinetic_update_targets(time, state=state, derived=derived,
                                     rates=rates_now, nu_pair=(nu_ion, nu_cx))

    def _kinetic_update_targets(self, time, state=None, derived=None,
                                rates=None, nu_pair=None):
        """Combine the unit responses with the LIVE channel rates (K4a-t)."""
        kin = self._kinetic
        if kin.responses is None:
            return
        if state is None:
            state = self.state
        if derived is None:
            derived = self.derived
        if rates is None:
            rates = self._kinetic_channel_rates(state, derived, time)
        if nu_pair is None:
            nu_pair = self._kinetic_absorption_fields(state, derived)
        nu_ion, nu_cx = nu_pair
        V_col, V_ann = self._zone_volumes
        scalars = {
            "cath": rates["cath"],
            "coll": rates["coll"],
            "puff": float(np.sum(rates["puff"])),
            "rec": float(np.sum(rates["rec"])),
            "anode": float(np.sum(rates["anode"])),
        }
        target_col = np.zeros(self._geometry.cells)
        target_ann = np.zeros(self._geometry.cells)
        arr = np.zeros(self._geometry.cells)
        for name, resp in kin.responses.items():
            s = scalars.get(name, 0.0)
            if s <= 0:
                continue
            target_col = target_col + s * resp["col"]
            target_ann = target_ann + s * resp["ann"]
            arr = arr + s * resp["arr"]
        kin.target_col = np.maximum(target_col, self._floors["nn"])
        kin.target_ann = np.maximum(target_ann, self._floors["nn"])
        # vbar carries m_He_cgs -- the mass this arm's OWN operators are built
        # on (KN2Zone/KN2ZoneJump: kinetic_neutrals.py M_HE, the velocity grid,
        # the wall spectra, the end sticking). The escape rate below is the
        # column's free-streaming loss for the very population the engine
        # marches, so it must be that population's thermal speed; the integer
        # mass number 4 x m_p is a less accurate value of the same quantity,
        # not an alternative convention.
        vbar = np.sqrt(8.0 * kb_cgs * 300.0 / (np.pi * m_He_cgs))
        esc = vbar / (
            2.0 * np.maximum(np.asarray(self._geometry.Rp_cm), 1e-6)
        )
        kin.tau_col = np.clip(
            1.0 / np.maximum(nu_ion + nu_cx + esc, 1e-6), 1e-5, 0.1
        )
        inv_ann = kin.target_ann * np.maximum(V_ann, 1e-6)
        kin.tau_ann = np.clip(
            inv_ann / np.maximum(arr, 1e-30), 1e-4, 0.05
        )
        kin.next_update_s = float(time) + kin.update_s

    # ------------------------------------------------------------ K2a DVM
    #
    # Terms whose ion-side momentum and energy rows the transient DVM
    # supersedes. The first group is pure ion-neutral transfer and is
    # replaced whole; the second group keeps its particle and
    # electron-energy rows (the plasma still books its own ionization,
    # recombination and the electron-side costs) and hands only the ion
    # momentum/energy booking to the measured kinetic moments.
    _DVM_TRANSFER_TERMS = frozenset(
        {
            "ion_charge_exchange",
            "ion_neutral_drag",
            "ion_neutral_frictional_heating",
            "ion_neutral_thermalization",
            "ion_neutral_collision",
        }
    )
    _DVM_BIRTH_TERMS = frozenset(
        {
            "ionization_birth",
            "beam_ionization_birth",
            "recombination_rad_loss",
            "recombination_3b_loss",
        }
    )
    #: The arm's source channels that are settled in COUNTED PARTICLES (B1),
    #: as the keys ``TransientDVM.update`` books them under. Each has a
    #: plasma-side row to count -- the boundary operator's wall return, the
    #: recombination birth, the anode return -- so the arm receives what the
    #: tick actually delivered rather than one snapshot of its rate. The puff
    #: is deliberately absent: it has no plasma-side row of its own (the
    #: fluid term carrying it is the puff NET of the pump sink, and the pump
    #: is the arm's own end-sticking channel), and being a function of time
    #: and configuration alone it carries no plasma-state sampling error.
    _DVM_COUNTED_SOURCES = (
        "cathode_face",
        "collector_face",
        "recombination",
        "anode",
    )

    def _zero_dvm_source_counts(self):
        """Return a fresh zeroed per-channel particle tally."""
        return {
            name: np.zeros(self._geometry.cells, dtype=float)
            for name in self._DVM_COUNTED_SOURCES
        }

    def _dvm_source_channel_rows(self, terms):
        """Return this RHS evaluation's boundary-inflow source rows.

        Read from the terms the step is about to apply, BEFORE
        :meth:`_strip_dvm_rows` zeroes their neutral rows, and through the
        same expression the tick-time snapshot uses, so a channel cannot
        drift between the counted and the sampled path.

        Unmasked by the plasma topology, exactly as the snapshot is: these
        rows are the arm's sources rather than fluid rows the mask decides
        the fate of, and masking one path but not the other would make the
        counted channel the integral of something the snapshot never sampled.
        """
        boundary = terms["characteristic_boundary"]
        return self._kinetic_source_channel_rows(
            boundary, terms, terms["anode_collection"]
        )

    def _strip_dvm_rows(self, name, term):
        """Return ``term`` with the rows the engaged DVM arm owns zeroed."""
        zeros = np.zeros_like(np.asarray(term.nn, dtype=float))
        superseded = (
            name in self._DVM_TRANSFER_TERMS or name in self._DVM_BIRTH_TERMS
        )
        return ConservativeState1D(
            n=term.n,
            nn=zeros,
            M=zeros.copy() if superseded else term.M,
            Ee=term.Ee,
            Ei=zeros.copy() if superseded else term.Ei,
            M_n=None if term.M_n is None else zeros.copy(),
            nn_a=None if term.nn_a is None else zeros.copy(),
            M_n_a=None if term.M_n_a is None else zeros.copy(),
        )

    def _dvm_rows_superseded(self):
        """Whether the DVM arm has taken the fluid rows over (K2a handover)."""
        return self._dvm is not None and self._dvm_engaged

    def _accumulate_dvm_ion_booking(self, terms):
        """Tally this RHS stage's share of the step's booked ionization.

        The engaged arm owns the fluid ``nn`` rows, so the neutrals the
        plasma consumes leave the fluid state through the ``n`` rows of
        ``ionization_birth`` and ``beam_ionization_birth`` alone (their
        ``nn`` counterparts are stripped, and are exactly minus the same
        numbers). Tallying those rows here, at the terms the step actually
        applies -- after the DVM stripping and the plasma-topology mask --
        is what makes the count handed to the arm the count the plasma
        booked, rather than a rate re-derived from a frozen state.

        The weight is set by :meth:`_attempt_step` for the whole attempt.
        Both explicit paths book the plasma rows through exactly one
        ``ssprk2_step`` at the full step dt (the implicit heat substep
        touches no particle row), and SSPRK2 weights its two stages
        EQUALLY at ``dt/2``, so one weight covers every call. Nothing
        accumulates outside an attempt (bare diagnostic reads) and a
        rejected attempt's tally is discarded with the attempt.
        """
        if self._dvm_ion_stage_accum is None:
            return
        self._dvm_ion_stage_accum += self._dvm_ion_stage_weight * (
            np.asarray(terms["ionization_birth"].n, dtype=float)
            + np.asarray(terms["beam_ionization_birth"].n, dtype=float)
        )

    def _dvm_cathode_jet_incident_energy_row(
        self, cathode_row, state, cathode_solve
    ):
        """Return the cathode recycle's INCIDENT ion-energy row [erg/s].

        ``phi_c + Ti`` per collected ion, clamped at zero, times the counted
        recycle rate -- the same per-particle incident energy the fluid
        channel's
        :func:`~cablp.solvers._sim1d.physics.sources.cathode_jet_backscatter_speed`
        reads, so the two arms describe ions arriving with one energy.

        ``phi_c`` comes from the solve THIS evaluation built, clamped at
        zero exactly as :meth:`_cathode_jet_spec` clamps it. The ``Ti``-only
        branch is reachable only where there is no solve to read: a
        configuration with cathode coupling unconfigured -- which
        :func:`~cablp.solvers._sim1d.core.validation.refuse_dvm_cathode_jet_without_cathode_coupling`
        now refuses at construction, so it cannot hold for a whole run -- or
        a driven solve that returned a non-finite sheath potential.

        A CONFIGURED run's afterglow does NOT take that branch: it rides the
        floating cathode solve like every other phase. What that solve
        returns there is measured, not assumed -- at the 1910 K emitting
        surface the Richardson current dwarfs the afterglow Bohm current, so
        the floating output is the EMISSION-DOMINATED one, ``phi_c`` of order
        zero to slightly inverted, rather than the classical non-emitting
        ``Lambda * Te``. The channel therefore self-extinguishes in afterglow
        twice over: by FLUX, since the counted recycle rate follows
        ``n * Te^1.5``, and by LAUNCH ENERGY, since ``phi_c + Ti`` falls to
        the ``Ti`` scale on its own. Both limits are the physics, not a
        fallback path.

        The row is a RATE because its partner (the counted source row) is,
        and the stage accumulator integrates both over the step at one
        weight; multiplying by the step's own dt here would be a second,
        differently-sampled integral of the same interval.
        """
        phi_c = 0.0
        if cathode_solve is not None and cathode_solve.beam_result is not None:
            candidate = float(cathode_solve.beam_result.result.phi_c)
            if np.isfinite(candidate):
                phi_c = max(candidate, 0.0)
        Ti = derive_state(
            state, floors=self._floors, ion_mass_g=self._ion_mass_g
        ).Ti
        per_ion_erg = np.maximum(phi_c + np.asarray(Ti, dtype=float), 0.0) * (
            ev_to_erg
        )
        return np.asarray(cathode_row, dtype=float) * per_ion_erg

    def _dvm_anode_jet_incident_energy_row(
        self, anode_row, state, cathode_solve
    ):
        """Return the anode collection's INCIDENT ion-energy row [erg/s].

        ``phi_a + Ti`` per collected ion, clamped at zero, times the counted
        collection rate -- the same per-particle incident energy the fluid
        channel's
        :func:`~cablp.solvers._sim1d.physics.sources.anode_jet_backscatter_speed`
        reads, so the two arms describe ions arriving with one energy. The
        ions fall through the ion-attracting anode sheath before striking the
        wires, which is what ``phi_a`` is.

        ``phi_a`` comes from the solve THIS evaluation built -- the SAME solve
        :meth:`_anode_jet_spec` reads it from, since the cathode, anode and
        bank are one system and one solve, and it is the A2a-booked ``phi_a``
        of the two-population anode-current split. It is taken unclamped and
        the SUM is clamped, exactly as ``_anode_jet_spec`` and
        ``anode_jet_backscatter_speed`` do it between them: an anode sheath
        that has gone slightly negative subtracts from the thermal energy the
        ions arrive with rather than being read as zero.

        The ``Ti``-only branch is reachable only where there is no solve to
        read -- a configuration with cathode coupling unconfigured, which
        :func:`~cablp.solvers._sim1d.core.validation.refuse_dvm_anode_jet_without_cathode_coupling`
        refuses at construction, or a solve that returned a non-finite
        ``phi_a``.

        NO AFTERGLOW SPECIAL CASE, and none is needed: the channel
        self-extinguishes there twice over, by FLUX -- the counted anode
        collection follows the Bohm flux ``n c_s`` and collapses with the
        plasma -- and by LAUNCH ENERGY, since ``phi_a`` measured on this model
        is a few volts at the discharge operating point and falls to the
        ``Ti`` scale on its own as the plasma goes. Both limits are the
        physics of the same two measured quantities, not a fallback path.

        The row is a RATE because its partner (the counted source row) is, and
        the stage accumulator integrates both over the step at one weight.
        """
        phi_a = 0.0
        if cathode_solve is not None and cathode_solve.beam_result is not None:
            candidate = float(cathode_solve.beam_result.result.phi_a)
            if np.isfinite(candidate):
                phi_a = candidate
        Ti = derive_state(
            state, floors=self._floors, ion_mass_g=self._ion_mass_g
        ).Ti
        per_ion_erg = np.maximum(phi_a + np.asarray(Ti, dtype=float), 0.0) * (
            ev_to_erg
        )
        return np.asarray(anode_row, dtype=float) * per_ion_erg

    def _accumulate_dvm_source_booking(self):
        """Tally this RHS stage's share of the tick's booked source inflow.

        The B1 counterpart of :meth:`_accumulate_dvm_ion_booking`, on the
        same stage weight and the same attempt lifetime: the rows were taken
        from the terms this evaluation built, before the strip zeroed them
        (:meth:`_dvm_source_channel_rows`), so summing them over the tick's
        accepted steps gives the particles each channel actually delivered
        rather than one tick-time reading of its rate. The rows are already
        particle rates, so no volume factor enters here -- unlike the
        ionization tally, which accumulates a density rate.
        """
        if self._dvm_source_stage_accum is None:
            return
        for name, accum in self._dvm_source_stage_accum.items():
            accum += self._dvm_ion_stage_weight * self._dvm_source_rows[name]
        if self._dvm_cathode_jet_energy_stage_accum is not None:
            # B5: the ENERGY half of the cathode channel's counted pair, on
            # the same stage weight and from the same evaluation's rows, so
            # the count and the energy the jet splits are integrals of one
            # state history rather than of two.
            self._dvm_cathode_jet_energy_stage_accum += (
                self._dvm_ion_stage_weight
                * self._dvm_cathode_jet_incident_row
            )
        if self._dvm_anode_jet_energy_stage_accum is not None:
            # B4: the ENERGY half of the anode channel's counted pair, on the
            # same stage weight and from the same evaluation's rows.
            self._dvm_anode_jet_energy_stage_accum += (
                self._dvm_ion_stage_weight
                * self._dvm_anode_jet_incident_row
            )

    def _dvm_booked_transfer_rhs(self):
        """Return the tick's BOOKED transfer as an RHS term, unlimited.

        What the kinetic side measured, before the step's floor-aware relax
        touches it. This is what the timestep bound must see: the limiter is
        the answer to a bound that could not be met, so bounding the limited
        rate would be circular.
        """
        zeros = np.zeros(self._geometry.cells, dtype=float)
        return ConservativeState1D(
            n=zeros,
            nn=zeros.copy(),
            M=self._dvm.M_transfer.copy(),
            Ee=zeros.copy(),
            Ei=self._dvm.Ei_transfer.copy(),
        )

    def neutral_kinetic_dvm_coupling_rhs(self):
        """Return the K2a plasma-side transfer term.

        Momentum and ion energy only, frozen between neutral-clock ticks
        and equal to MINUS the measured moments of the kinetic ionization,
        charge-exchange, elastic and recombination operators. Zeros before
        the arm engages, so the term key is present from the first step and
        the saved term structure is stable across the run.

        Once a step has been scoped (:meth:`_dvm_scope_step_transfer`) this
        returns that step's APPLIED rate: the booked transfer plus any
        outstanding debt, relaxed at cells the drain would otherwise carry
        below their ion-energy floor within the step. Outside a scoped step
        -- before the first attempt, and at a bare diagnostic read -- it
        returns the booked transfer itself, which is what the arm has to
        offer when no step is asking.
        """
        zeros = np.zeros(self._geometry.cells, dtype=float)
        if not self._dvm_rows_superseded():
            return ConservativeState1D(
                n=zeros,
                nn=zeros.copy(),
                M=zeros.copy(),
                Ee=zeros.copy(),
                Ei=zeros.copy(),
            )
        scope = self._dvm_step_transfer
        if scope is None:
            return self._dvm_booked_transfer_rhs()
        return ConservativeState1D(
            n=zeros,
            nn=zeros.copy(),
            M=scope.applied_M.copy(),
            Ee=zeros.copy(),
            Ei=scope.applied_Ei.copy(),
        )

    def _dvm_transfer_apply_mask(self):
        """Cells the coupling term is actually applied on."""
        if self._active_plasma_topology:
            return np.asarray(self._geometry.plasma_active, dtype=bool)
        return np.ones(self._geometry.cells, dtype=bool)

    def _dvm_scope_step_transfer(self, dt):
        """Scope one step's applied DVM transfer (the K2d floor-aware relax).

        The tick-frozen transfer is held constant for a whole neutral clock
        interval while the plasma steps a thousand times inside it, and it
        can flip sign across one tick. At a cell whose ion-energy margin the
        frozen drain would consume within a single step there is no
        admissible timestep -- the explicit e-fold falls below ``dt_min`` --
        and the step dies with a negative ``Ei``. The near-cancelling partner
        (heat conduction) is on the implicit side of the split, so shrinking
        dt does not help: the drain is explicit and the refill is not.

        So the APPLIED drain is capped per cell at
        ``relax_fraction * (Ei - Ei_floor) / dt``, which by construction
        cannot reach the floor inside the step, and the SAME per-cell factor
        is applied to the momentum row -- one physical exchange, one relax
        factor, rather than a rescaled energy against a full momentum.

        Nothing is discarded. The withheld amount is added to a per-cell
        DEBT and re-offered as ``debt / dt`` on every later step, so a cell
        that regains margin pays back what it owes and the ledger identity

            applied_cum + debt + hold_debt == booked_cum

        holds per cell to roundoff (:meth:`_dvm_book_step_transfer` closes
        it). The ``hold_debt`` row is zero throughout under
        ``transfer_hold = "zoh"`` and is the exponential hold's own,
        disjoint, deferral; see :meth:`_dvm_transfer_hold_offer`.
        A cell that never regains margin carries the debt to the end of
        the run, which is the honest statement that the plasma could not
        absorb what the kinetic side booked -- reported, never silently
        dropped.

        Heating (a non-negative ``Ei`` rate) is never limited: it cannot
        threaten a floor.

        Under ``neutral_kinetic_dvm_transfer_hold = "exponential"`` the
        booked rate is first corrected to the EXPONENTIAL HOLD of the
        CX/elastic pair (:meth:`_dvm_transfer_hold_offer`) before any of the
        above; the floor relax then acts on that, so the two mechanisms
        compose in one direction only and each keeps its own debt row.
        """
        dvm = self._dvm
        state = self.state
        apply_mask = self._dvm_transfer_apply_mask()
        booked_M = np.asarray(dvm.M_transfer, dtype=float)
        booked_Ei = np.asarray(dvm.Ei_transfer, dtype=float)
        offer_M, offer_Ei = self._dvm_transfer_hold_offer(dt, state)
        if offer_Ei is None:
            desired_M = booked_M + dvm.M_debt / dt
            desired_Ei = booked_Ei + dvm.Ei_debt / dt
        else:
            desired_M = booked_M + offer_M + dvm.M_debt / dt
            desired_Ei = booked_Ei + offer_Ei + dvm.Ei_debt / dt
        floor_energy = (
            1.5
            * float(self._floors["Ti"])
            * ev_to_erg
            * np.asarray(state.n, dtype=float)
        )
        margin = np.maximum(
            np.asarray(state.Ei, dtype=float) - floor_energy, 0.0
        )
        budget = self._dvm_transfer_relax_fraction * margin / dt
        drain = -desired_Ei
        limited = apply_mask & (drain > budget) & (drain > 0.0)
        scale = np.where(limited, budget / np.where(limited, drain, 1.0), 1.0)
        applied_M = np.where(apply_mask, scale * desired_M, 0.0)
        applied_Ei = np.where(apply_mask, scale * desired_Ei, 0.0)
        self._dvm_step_transfer = SimpleNamespace(
            dt=float(dt),
            apply_mask=apply_mask,
            booked_M=booked_M,
            booked_Ei=booked_Ei,
            offer_M=offer_M,
            offer_Ei=offer_Ei,
            desired_M=desired_M,
            desired_Ei=desired_Ei,
            applied_M=applied_M,
            applied_Ei=applied_Ei,
            limited=limited,
        )
        return self._dvm_step_transfer

    def _dvm_transfer_hold_offer(self, dt, state):
        """Return the exponential hold's correction to the booked pair rate.

        ``(None, None)`` under ``transfer_hold = "zoh"``, where the booked
        rate IS what the step offers and the caller's arithmetic is the
        pre-hold one, unchanged.

        Otherwise the CX/elastic pair is the linear relaxation

            d(Ei)/dt = -nu (Ei - Ei_eq),   d(M)/dt = -nu (M - M_eq)

        at the tick's frozen ``nu`` and targets, and the exact solution over
        a step of length ``dt`` from the step's OWN state is applied as a
        constant rate::

            rate = (X_eq - X) (1 - e^{-nu dt}) / dt
                 = booked_pair * phi(nu dt) - (X - X_tick) (1 - e^{-nu dt}) / dt

        with ``phi(x) = (1 - e^{-x}) / x`` and ``X_tick`` the fluid row the
        tick's rate was measured against, which is what makes the targets
        ``X_eq = X_tick + booked_pair / nu`` -- i.e. the targets the BOOKED
        moments define, so the pair stays antisymmetric with the kinetic
        side rather than being rebuilt from a second formula. The returned
        correction is that rate MINUS the booked pair rate, plus the
        repayment of the outstanding hold debt; the caller adds it to the
        booked total, so the ionization and recombination rows pass through
        untouched as the sources they are.

        THE REPAYMENT ENTERS THROUGH THE TARGET, not as a raw constant
        source added beside the relaxation. Spreading the tick's outstanding
        debt ``D`` over the following tick as a flat ``D / dt_tick`` re-injects
        the very zero-order increment the hold removed, and the coupled
        (state, debt) map is then unstable again -- measured on the synthetic
        cell at four plasma steps per tick: growth at ``nu dt_tick`` of 4 and
        20, i.e. the same threshold as the scheme being replaced, with the
        margin depending on how many plasma steps happen to fall inside a
        tick. Offering it as ``D phi(nu dt) / dt_tick`` instead delivers
        exactly ``D / dt_tick`` in the resolved limit (``phi -> 1``) and damps
        it by the same exponential the relaxation carries when the tick is
        coarse, which makes the map

            g <- e^{-nu dt_tick} g + a D,   D <- -(nu dt_tick - 1 + e^{-...}) g
                                                 + (1 - a) D
            a = (1 - e^{-nu dt_tick}) / (nu dt_tick)

        whose determinant is exactly ``1 - a`` and whose trace is
        ``e^{-nu dt_tick} + 1 - a``. Both eigenvalues therefore lie strictly
        inside the unit circle for every ``nu dt_tick > 0``, independently of
        how the tick is subdivided, and the only fixed point is ``g = D = 0``:
        the debt is driven to zero rather than merely bounded.

        The two limits are the ones that matter: ``nu dt -> 0`` returns the
        repayment alone plus ``-nu (X - X_tick)``, i.e. the zero-order hold
        evaluated at the current state; and no value of ``nu dt`` can carry
        ``X`` past its (debt-shifted) target within a step, which is the
        instability this replaces.
        """
        if self._dvm_transfer_hold == "zoh":
            return None, None
        dvm = self._dvm
        nu = np.asarray(dvm.nu_pair, dtype=float)
        x = nu * dt
        # 1 - e^{-x} and (1 - e^{-x})/x, both accurate as x -> 0.
        psi = -np.expm1(-x)
        positive = x > 0.0
        phi = np.where(positive, psi / np.where(positive, x, 1.0), 1.0)
        dM = np.asarray(state.M, dtype=float) - self._dvm_hold_M0
        dEi = np.asarray(state.Ei, dtype=float) - self._dvm_hold_Ei0
        offer_M = (
            np.asarray(dvm.M_transfer_pair, dtype=float) * (phi - 1.0)
            - dM * psi / dt
            + self._dvm_hold_repay_M * phi
        )
        offer_Ei = (
            np.asarray(dvm.Ei_transfer_pair, dtype=float) * (phi - 1.0)
            - dEi * psi / dt
            + self._dvm_hold_repay_Ei * phi
        )
        return offer_M, offer_Ei

    def _dvm_arm_transfer_hold(self, state):
        """Freeze the exponential hold's tick state and repayment rate.

        Called at every neutral tick (and at engagement), from the ACCEPTED
        state the tick's transfer was booked against. The repayment rate
        retires whatever hold debt stands at the tick over the cadence that
        follows -- never as ``debt / dt`` per plasma step, which would
        re-offer the whole debt on every step and put the truncation back
        into the drain path the hold exists to calm. It is offered THROUGH
        the relaxation rather than beside it; see
        :meth:`_dvm_transfer_hold_offer` for why, and for the stability
        statement that buys.
        """
        self._dvm_hold_M0 = np.asarray(state.M, dtype=float).copy()
        self._dvm_hold_Ei0 = np.asarray(state.Ei, dtype=float).copy()
        if self._dvm_transfer_hold == "zoh":
            return
        cadence = float(self._dvm_cadence_s)
        self._dvm_hold_repay_M = self._dvm.M_hold_debt / cadence
        self._dvm_hold_repay_Ei = self._dvm.Ei_hold_debt / cadence

    def _dvm_book_step_transfer(self, dt):
        """Commit the scoped step's transfer to the deferred-transfer ledger.

        Called at ACCEPT, before the state advances, so the debt a step
        leaves behind is exactly the transfer that step declined to carry.
        Rejected attempts never reach here.
        """
        scope = self._dvm_step_transfer
        if scope is None or scope.dt != float(dt):
            scope = self._dvm_scope_step_transfer(dt)
        dvm = self._dvm
        mask = scope.apply_mask
        dvm.M_debt = np.where(
            mask, (scope.desired_M - scope.applied_M) * dt, 0.0
        )
        dvm.Ei_debt = np.where(
            mask, (scope.desired_Ei - scope.applied_Ei) * dt, 0.0
        )
        if scope.offer_Ei is not None:
            # What the hold offered over and above the booked rate is exactly
            # what it no longer owes; the identity below closes because the
            # offer entered ``desired`` and so reached ``applied_cum`` (or,
            # where the floor relax withheld it, the debt row above).
            dvm.M_hold_debt = np.where(
                mask, dvm.M_hold_debt - scope.offer_M * dt, 0.0
            )
            dvm.Ei_hold_debt = np.where(
                mask, dvm.Ei_hold_debt - scope.offer_Ei * dt, 0.0
            )
        dvm.M_booked_cum = dvm.M_booked_cum + np.where(
            mask, scope.booked_M * dt, 0.0
        )
        dvm.Ei_booked_cum = dvm.Ei_booked_cum + np.where(
            mask, scope.booked_Ei * dt, 0.0
        )
        dvm.M_applied_cum = dvm.M_applied_cum + scope.applied_M * dt
        dvm.Ei_applied_cum = dvm.Ei_applied_cum + scope.applied_Ei * dt
        dvm.relax_steps += 1
        if np.any(scope.limited):
            dvm.relax_limited_steps += 1
            dvm.relax_cell_steps = dvm.relax_cell_steps + scope.limited
        # The scope belonged to the step just booked. Dropping it here keeps
        # "outside a step" a single well-defined reading of the coupling term
        # -- the tick's booked transfer -- rather than the last step's
        # application surviving across a neutral tick that already replaced it.
        self._dvm_step_transfer = None

    def dvm_transfer_ledger(self):
        """Return the deferred-transfer ledger's closure, or None when off.

        ``residual`` is ``applied_cum + debt + hold_debt - booked_cum`` per
        cell, the statement that the relax and the exponential hold DEFERRED
        energy and momentum rather than destroying them; ``*_rel``
        normalizes it by the accumulated throughput plus both outstanding
        debts.

        The ``N`` channel is the same statement for PARTICLES across the
        fluid/kinetic ionization handshake: ``removed_cum + debt -
        booked_cum``, i.e. every neutral the plasma turned into an ion has
        either left the kinetic state or is still owed by it. Unlike the
        two transfer channels this one is a conservation law, not a
        scheduling record -- a nonzero relative residual here is particle
        creation in the coupled system, which is exactly the leak the
        counted handshake removed.
        """
        if self._dvm is None:
            return None
        dvm = self._dvm
        out = {}
        zero_hold = np.zeros(self._geometry.cells, dtype=float)
        for name, applied, booked, debt, hold in (
            ("M", dvm.M_applied_cum, dvm.M_booked_cum, dvm.M_debt,
             dvm.M_hold_debt),
            ("Ei", dvm.Ei_applied_cum, dvm.Ei_booked_cum, dvm.Ei_debt,
             dvm.Ei_hold_debt),
            ("N", dvm.ion_removed_cum, dvm.ion_booked_cum, dvm.ion_debt,
             zero_hold),
        ):
            residual = applied + debt + hold - booked
            scale = (
                np.max(np.abs(booked))
                + np.max(np.abs(debt))
                + np.max(np.abs(hold))
            )
            out[name] = {
                "residual": residual,
                "scale": float(scale),
                "rel": float(
                    np.max(np.abs(residual)) / max(float(scale), 1e-300)
                ),
            }
        out["relax_steps"] = int(dvm.relax_steps)
        out["relax_limited_steps"] = int(dvm.relax_limited_steps)
        out["relax_cell_steps"] = dvm.relax_cell_steps.copy()
        out["ion_shortfall_updates"] = int(dvm.ion_shortfall_updates)
        out["ion_shortfall_cell_updates"] = dvm.ion_shortfall_cell_updates.copy()
        return out

    def _dvm_ledger_sample(self, time):
        """Return the per-save-frame census record of the transfer ledger.

        Three cumulative counters and the frame time -- enough to place WHEN
        the limiter engaged (difference consecutive frames) without carrying
        the per-cell arrays at every save. ``limited_cells`` counts cells
        limited at least once so far, not cells limited at this frame.

        The three ``ion_*`` totals are the particle handshake's running
        domain sums [particles]. They are the per-frame record of the
        neutral sink the engaged arm owns, which the saved ``rhs_terms``
        cannot show -- the fluid ``nn`` rows are stripped to zero once the
        arm supersedes them, so the ionization drain is invisible in the
        term ledger from the first tick onward. Differencing consecutive
        frames gives the ionization booked and debited over that interval.
        """
        dvm = self._dvm
        volume = np.asarray(self._geometry.plasma_volume_cm3, dtype=float)
        return {
            "time": float(time),
            "relax_steps": float(dvm.relax_steps),
            "relax_limited_steps": float(dvm.relax_limited_steps),
            "limited_cells": float(np.count_nonzero(dvm.relax_cell_steps)),
            "Ei_hold_debt_total": float(np.sum(dvm.Ei_hold_debt * volume)),
            "M_hold_debt_total": float(np.sum(dvm.M_hold_debt * volume)),
            "Ei_hold_debt_max_abs": float(np.max(np.abs(dvm.Ei_hold_debt))),
            "ion_booked_total": float(np.sum(dvm.ion_booked_cum)),
            "ion_removed_total": float(np.sum(dvm.ion_removed_cum)),
            "ion_debt_total": float(np.sum(dvm.ion_debt)),
            "ion_shortfall_updates": float(dvm.ion_shortfall_updates),
        }

    def _dvm_ledger_census(self, saved):
        """Return the end-of-run transfer-ledger census for the result.

        The quotable form of the standing DVM report condition: how many
        steps the floor-aware relax limited, what each channel still owes the
        plasma, what the exponential hold still owes as HOLD DEBT (zero
        throughout under ``transfer_hold = "zoh"``), and the cumulative
        booked-vs-applied totals that carry the identity
        ``applied_cum + debt + hold_debt == booked_cum``. Per-cell arrays are
        kept (they are one row of ``cells`` each) so the identity is
        checkable from the artifact and the debt can be localized; the
        scalars are the numbers a report quotes.

        Units are the ledger's own, integrated over the step: ``Ei_*`` in
        erg/cm^3 and ``M_*`` in g/(cm^2 s) per cell, and the ``*_total``
        scalars are those densities integrated over the plasma (= column)
        volume, i.e. erg and g cm/s. The ``ion_*`` block is the particle
        handshake's own ledger and is already in PARTICLES per cell, so it
        takes no volume conversion; ``ion_residual_rel`` is the coupled
        system's particle-conservation residual and is the number a report
        quotes to say the handshake creates nothing.
        """
        dvm = self._dvm
        volume = np.asarray(self._geometry.plasma_volume_cm3, dtype=float)
        census = {
            "engaged": int(bool(self._dvm_engaged)),
            "transfer_hold": self._dvm_transfer_hold,
            "relax_steps": int(dvm.relax_steps),
            "relax_limited_steps": int(dvm.relax_limited_steps),
            "limited_cells": int(np.count_nonzero(dvm.relax_cell_steps)),
            "relax_cell_steps": dvm.relax_cell_steps.copy(),
        }
        for name in ("Ei", "M"):
            debt = np.asarray(getattr(dvm, f"{name}_debt"), dtype=float)
            hold = np.asarray(getattr(dvm, f"{name}_hold_debt"), dtype=float)
            booked = np.asarray(getattr(dvm, f"{name}_booked_cum"), dtype=float)
            applied = np.asarray(getattr(dvm, f"{name}_applied_cum"), dtype=float)
            residual = applied + debt + hold - booked
            scale = float(
                np.max(np.abs(booked))
                + np.max(np.abs(debt))
                + np.max(np.abs(hold))
            )
            census.update(
                {
                    f"{name}_debt": debt.copy(),
                    f"{name}_hold_debt": hold.copy(),
                    f"{name}_booked_cum": booked.copy(),
                    f"{name}_applied_cum": applied.copy(),
                    f"{name}_debt_total": float(np.sum(debt * volume)),
                    f"{name}_debt_max_abs": float(np.max(np.abs(debt))),
                    f"{name}_hold_debt_total": float(np.sum(hold * volume)),
                    f"{name}_hold_debt_max_abs": float(np.max(np.abs(hold))),
                    f"{name}_booked_total": float(np.sum(booked * volume)),
                    f"{name}_applied_total": float(np.sum(applied * volume)),
                    f"{name}_residual_rel": float(
                        np.max(np.abs(residual)) / max(scale, 1e-300)
                    ),
                }
            )
        ion_debt = np.asarray(dvm.ion_debt, dtype=float)
        ion_booked = np.asarray(dvm.ion_booked_cum, dtype=float)
        ion_removed = np.asarray(dvm.ion_removed_cum, dtype=float)
        ion_residual = ion_removed + ion_debt - ion_booked
        ion_scale = float(np.max(np.abs(ion_booked)) + np.max(np.abs(ion_debt)))
        census.update(
            {
                "ion_debt": ion_debt.copy(),
                "ion_booked_cum": ion_booked.copy(),
                "ion_removed_cum": ion_removed.copy(),
                "ion_debt_total": float(np.sum(ion_debt)),
                "ion_debt_max_abs": float(np.max(np.abs(ion_debt))),
                "ion_booked_total": float(np.sum(ion_booked)),
                "ion_removed_total": float(np.sum(ion_removed)),
                "ion_residual_rel": float(
                    np.max(np.abs(ion_residual)) / max(ion_scale, 1e-300)
                ),
                "ion_shortfall_updates": int(dvm.ion_shortfall_updates),
                "ion_shortfall_cell_updates": (
                    dvm.ion_shortfall_cell_updates.copy()
                ),
            }
        )
        for field in (
            "time",
            "relax_steps",
            "relax_limited_steps",
            "limited_cells",
            "Ei_hold_debt_total",
            "M_hold_debt_total",
            "Ei_hold_debt_max_abs",
            "ion_booked_total",
            "ion_removed_total",
            "ion_debt_total",
            "ion_shortfall_updates",
        ):
            census[f"sample_{field}"] = np.asarray(
                [snapshot["dvm_ledger"][field] for snapshot in saved],
                dtype=float,
            )
        return census

    def _dvm_engage(self):
        """Seed the transient distributions from the live fluid neutrals.

        Until this fires the moment terms carry the neutrals exactly as
        they always have -- the pre-breakdown fill and the neutral
        equilibration are untouched by the arm. The seed is a Maxwellian
        at the wall temperature carrying the fluid's own column and
        annulus densities, so the handover conserves particles exactly.
        """
        state = self.state
        self._dvm.seed_from_density(
            np.maximum(np.asarray(state.nn, dtype=float), 0.0),
            np.maximum(np.asarray(state.nn_a, dtype=float), 0.0),
        )
        self._dvm_engaged = True
        self._dvm_last_s = self._time
        self._dvm_next_s = self._time + self._dvm_cadence_s
        self._dvm_arm_transfer_hold(state)
        # The step that engaged the arm booked its ionization and its
        # boundary inflow against the LIVE fluid nn rows, which were not yet
        # stripped, so both are already settled and must not be handed to the
        # arm as well.
        self._dvm_ion_booked = np.zeros(self._geometry.cells, dtype=float)
        self._dvm_source_booked = self._zero_dvm_source_counts()
        self._dvm_cathode_jet_energy_booked = np.zeros(
            self._geometry.cells, dtype=float
        )
        self._dvm_cathode_jet_count_booked = np.zeros(
            self._geometry.cells, dtype=float
        )
        self._dvm_anode_jet_energy_booked = np.zeros(
            self._geometry.cells, dtype=float
        )

    def _dvm_advance(self, dt_neutral):
        """Run one transient DVM update and republish the neutral moments.

        The republish is ONE-SIDED at the density floor. The fluid ``nn``
        and ``nn_a`` rows are written as ``max(moment, nn_floor)``, so a
        cell whose kinetic moment sits below the floor publishes the floor
        to the plasma while the kinetic state keeps its own lower moment,
        and no cell is ever published below its moment. Nothing raises
        ``f`` to match: the two inventories can therefore differ, in that
        one direction only, by at most the floor times the cell volume.
        The kinetic ledger is the inventory of record; the published rows
        are what the plasma-side rate evaluations consume.

        DISCLOSED ORDERING. This tick fires EARLIER in the accept path than
        the cathode warming update, so every surface spectrum it launches --
        the thermal recycle inflow, the closed-face re-emission, the left end
        wall, and with the cathode jet armed the ``1 - R_N`` thermal share --
        reads the surface temperature as it stood BEFORE this step's warming
        increment. It is one step stale by construction. The order is not
        reversed here: the warming update reads the accepted state that this
        tick republishes into, so swapping them would make the surface
        temperature a function of the neutral field of the same instant it is
        driving, and the staleness is a step's worth of ``dT`` on a surface
        whose thermal time constant is many orders above the step.

        DISCLOSED CONVENTION, cathode jet armed. Only the ``R_E`` energy the
        BACKSCATTER carries is debited from the surface. The ``1 - R_N``
        implanted share desorbs at the surface temperature and its
        ``(3/2) k T_s`` per atom is NOT taken off the cathode's balance --
        the same convention the fluid channel ships, and at the production
        recycle rate it is tens of watts against a kilowatt-class
        ``P_cathode_i``. It is a convention, not a measurement: the surface
        energy ledger's ``backscatter`` row is what the surface actually gave
        up, and this share is deliberately outside it.
        """
        state = self.state
        derived = self.derived
        geometry = self._geometry
        nu_ion = self._dvm_ionization_frequency(state, derived)
        rates = self._kinetic_channel_rates(state, derived, self._time)
        # Only the puff is still handed as a rate: it is a configured drive,
        # a function of time and configuration alone with no plasma-side row
        # of its own to count, so a tick-time reading of it carries no
        # plasma-state sampling error. The other four channels are counted
        # below; see _DVM_COUNTED_SOURCES. The per-cell wall returns are not
        # scalars: the return belongs in the cell the boundary term drained,
        # which is not an end cell once an obstruction sits behind the
        # cathode.
        sources = {"puff": np.asarray(rates["puff"], dtype=float)}
        # The counted handshake: what the plasma booked as ionization, and as
        # boundary inflow per channel, over this tick -- in particles, so the
        # arm debits and injects those and not a rate. Reset before the
        # update, not after, so a raise cannot leave a tally to be counted
        # twice.
        ion_counts = self._dvm_ion_booked * np.asarray(
            self._geometry.plasma_volume_cm3, dtype=float
        )
        source_counts = self._dvm_source_booked
        # B5: the counted INCIDENT energy that rides the cathode_face count.
        # None with the channel off, which is what the engine refuses to be
        # handed; reset on the same line as its partner so a raise inside the
        # update can neither double-birth nor double-debit.
        jet_incident = None
        jet_counts = None
        if self._dvm_cathode_jet is not None:
            jet_incident = self._dvm_cathode_jet_energy_booked
            self._dvm_cathode_jet_energy_booked = np.zeros(
                self._geometry.cells, dtype=float
            )
            # Handed and reset on the same lines as its energy partner, for
            # the same reason: a raise inside the update must not leave one
            # of the pair counted twice while the other is spent.
            jet_counts = self._dvm_cathode_jet_count_booked
            self._dvm_cathode_jet_count_booked = np.zeros(
                self._geometry.cells, dtype=float
            )
        anode_jet_incident = None
        if self._dvm_anode_jet is not None:
            anode_jet_incident = self._dvm_anode_jet_energy_booked
            self._dvm_anode_jet_energy_booked = np.zeros(
                self._geometry.cells, dtype=float
            )
        self._dvm_ion_booked = np.zeros(self._geometry.cells, dtype=float)
        self._dvm_source_booked = self._zero_dvm_source_counts()
        self._dvm.update(
            float(dt_neutral),
            n_i=np.asarray(state.n, dtype=float),
            Ti_eV=np.asarray(derived.Ti, dtype=float),
            u_i=np.asarray(derived.u, dtype=float),
            nu_ion=nu_ion,
            ion_counts=ion_counts,
            sources=sources,
            source_counts=source_counts,
            cathode_jet_incident_erg=jet_incident,
            cathode_jet_counts=jet_counts,
            anode_jet_incident_erg=anode_jet_incident,
            T_s_K=(
                float(self._cathode_Ts_K)
                if self._cathode_Ts_K is not None
                else float(self._input_dict.get("T_s"))
            ),
        )
        if (
            not self._dvm_ion_shortfall_warned
            and self._dvm.ion_shortfall_updates
        ):
            self._dvm_ion_shortfall_warned = True
            warnings.warn(
                "the kinetic_dvm arm could not debit the whole ionization "
                "count the plasma booked at "
                f"{int(np.count_nonzero(self._dvm.ion_shortfall_cell_updates))} "
                f"cell(s) by t={self._time:.6g} s: those cells ran out of "
                "column neutrals inside one neutral tick. The shortfall is "
                "held as ion_debt and re-offered on later ticks, so nothing "
                "is lost, but a run that carries debt persistently is "
                "asking for a shorter neutral cadence "
                "(neutral_kinetic_dvm_cadence_s). The per-cell debt and the "
                "shortfall counts are in the saved dvm_transfer_ledger.",
                stacklevel=2,
            )
        self._dvm_last_s = self._time
        self._dvm_next_s = self._time + self._dvm_cadence_s
        self._dvm_tick_count += 1
        # Freeze the hold's tick state against the SAME accepted rows the
        # transfer above was booked against, before the republish below
        # rewrites anything.
        self._dvm_arm_transfer_hold(state)
        # Republish: the fluid neutral density consumed by the plasma
        # physics IS the zeroth moment of f. Written straight into the
        # packed rows rather than through the flooring path, so the plasma
        # rows the step just accepted are not re-rounded.
        cells = geometry.cells
        names = state_field_names(state)
        y = self._y.copy()
        floor = self._floors["nn"]
        y[cells : 2 * cells] = np.maximum(self._dvm.column_density(), floor)
        row = names.index(NEUTRAL_ANNULUS_NAME)
        y[row * cells : (row + 1) * cells] = np.maximum(
            self._dvm.annulus_density(), floor
        )
        self._y = y
        self._state = self._unpack(y)
        self._derived = derive_state(
            self._state, self._floors, self._ion_mass_g
        )

    def _dvm_ionization_frequency(self, state, derived):
        """Return the velocity-blind ionization frequency [1/s] per cell.

        Derived from the ionization the PLASMA actually books -- bulk plus
        beam -- divided by the column density those bookings consumed,
        rather than re-deriving a rate that could drift from it.

        This is the rate the implicit march carries: it sets how the
        surviving neutrals are attenuated and transported WITHIN the tick.
        It does not decide how many neutrals the tick destroys. That is the
        counted ``ion_counts`` channel, which is the accumulated booking
        itself and needs no rate; see :meth:`_dvm_advance`.
        """
        S = np.asarray(
            self.reaction_rhs_terms(state=state)["ionization_birth"].n,
            dtype=float,
        )
        cathode_phase = self._cathode_phase_options(time=self._time)
        cathode_solve = None
        if cathode_phase["solve_enabled"]:
            # Accepted-state solve that does NOT write the warm-start cache:
            # this is a neutral-clock read, not part of the step's own solve
            # sequence.
            cathode_solve = self.solve_cathode_boundary(
                state=state,
                floating=cathode_phase["floating"],
                time=self._time,
                update_cache=False,
            )
        if cathode_solve is not None:
            S = S + np.asarray(
                self.beam_ionization_rhs_terms(
                    state=state,
                    cathode_solve=cathode_solve,
                    time=self._time,
                )["beam_ionization_birth"].n,
                dtype=float,
            )
        nu = S / np.maximum(np.asarray(state.nn, dtype=float), self._floors["nn"])
        if self._active_plasma_topology:
            nu = np.where(
                np.asarray(self._geometry.plasma_active, dtype=bool), nu, 0.0
            )
        return np.maximum(nu, 0.0)

    @staticmethod
    def _strip_neutral_rows(term):
        """Return the term with its nn / nn_a rows zeroed (K4a supersession).

        The plasma-side rows keep their exact forms -- the neutral rows are
        carried by the kinetic relaxation instead. M_n rows pass through
        (the wind stays a moment field in K4a).
        """
        zeros = np.zeros_like(np.asarray(term.nn, dtype=float))
        return ConservativeState1D(
            n=term.n,
            nn=zeros,
            M=term.M,
            Ee=term.Ee,
            Ei=term.Ei,
            M_n=term.M_n,
            nn_a=None if term.nn_a is None else zeros.copy(),
            M_n_a=term.M_n_a,
        )

    def neutral_kinetic_relaxation_rhs(self, state):
        """Return the K4a relaxation term: (nn* - nn)/tau per zone."""
        kin = self._kinetic
        zeros = np.zeros(self._geometry.cells, dtype=float)
        if kin is None or kin.target_col is None or state.nn_a is None:
            return ConservativeState1D(
                n=zeros,
                nn=zeros.copy(),
                M=zeros.copy(),
                Ee=zeros.copy(),
                Ei=zeros.copy(),
            )
        return ConservativeState1D(
            n=zeros,
            nn=(kin.target_col - state.nn) / kin.tau_col,
            M=zeros.copy(),
            Ee=zeros.copy(),
            Ei=zeros.copy(),
            nn_a=(kin.target_ann - state.nn_a) / kin.tau_ann,
        )

    def _zero_rhs_state(self):
        """Return the run's ONE all-zero conservative RHS bundle.

        A structurally absent term is the same object every time it is asked
        for: the rows are all zeros, nothing distinguishes one instance from
        another, and a single RHS evaluation asks for tens of them. The five
        rows are READ-ONLY, so the sharing cannot become a channel between two
        terms -- an in-place write raises where before it would have edited one
        caller's private copy. (The mutation census over cablp/ and scripts/ is
        empty today; the guard is what keeps it so.)
        """
        shared = self._zero_rhs_state_shared
        if shared is None:
            rows = []
            for _ in STATE_NAMES_1D:
                row = np.zeros(self._geometry.cells, dtype=float)
                row.flags.writeable = False
                rows.append(row)
            shared = ConservativeState1D(
                n=rows[0],
                nn=rows[1],
                M=rows[2],
                Ee=rows[3],
                Ei=rows[4],
            )
            self._zero_rhs_state_shared = shared
        return shared

    def _set_state_vector(self, y):
        self._y = self.floor_state_vector(y)
        self._state = self._unpack(self._y)
        self._derived = derive_state(self._state, self._floors, self._ion_mass_g)
        if self._flags.get("debug_checks"):
            assert_finite_state(self._state, self._derived)

    def _check_annulus_not_collapsed(self):
        """Refuse a two-zone geometry whose annulus has collapsed to a sliver.

        Reads ``neutral_annulus_volume_fraction_min`` and raises ``ValueError``
        for any cell whose annulus EXISTS but holds less than that fraction of
        the cell's neutral volume. ``0.0`` disables the check.

        Cells with no annulus at all (``V_ann = 0`` exactly) are deliberately
        exempt: every annulus consumer gates on ``V_ann > 0``, so an absent
        zone is inert. The hazard is the zone that exists and is tiny, because
        it is a DIVISOR and nothing gates on it -- the zone exchange and the
        hot-channel deposit ``landed * Vp / V_ann`` both scale as ``1/V_ann``,
        so a collapsing annulus does not switch off, it stiffens the step.
        """
        threshold = float(
            self._input_dict.get("neutral_annulus_volume_fraction_min")
        )
        if not np.isfinite(threshold) or not 0.0 <= threshold < 1.0:
            raise ValueError(
                "neutral_annulus_volume_fraction_min must be finite and in "
                f"[0, 1) (got {threshold}); it is the minimum share of a "
                "cell's neutral volume the annulus may hold, and 0 disables "
                "the check"
            )
        if threshold <= 0.0 or self._zone_volumes is None:
            return
        V_ann = np.asarray(self._zone_volumes[1], dtype=float)
        Vm = np.asarray(self._geometry.neutral_volume_cm3, dtype=float)
        fraction = V_ann / Vm
        collapsed = np.flatnonzero((V_ann > 0.0) & (fraction < threshold))
        if collapsed.size:
            worst = int(collapsed[np.argmin(fraction[collapsed])])
            raise ValueError(
                "the two-zone annulus has collapsed to a sliver in "
                f"{collapsed.size} cell(s) {collapsed.tolist()}: the worst is "
                f"cell {worst} at V_ann/V_neutral = {float(fraction[worst]):.3e}"
                f" against neutral_annulus_volume_fraction_min={threshold:g} "
                f"(V_ann = {float(V_ann[worst]):.6g} cm^3 inside "
                f"{float(Vm[worst]):.6g} cm^3). The annulus row's sources "
                "divide by that volume, so a vanishing zone stiffens the step "
                "rather than switching off. Widen the vessel there, cap the "
                "plasma area with plasma_area_max_vessel_fraction, or lower "
                "the threshold deliberately"
            )

    def _configure_neutral_initial_profile(self):
        """Build the shaped initial neutral fill, or ``None`` when off.

        Sets ``_nn0_profile`` (the column, or the single neutral field without
        the two-zone closure) and ``_nn0_annulus_profile``. Both are ``None``
        with the ``neutral_initial_profile`` flag off, which is the presence
        gate :meth:`_initial_state` reads: the off path resolves the scalar
        fill exactly as it always has and never touches an array from here.
        """
        enabled = bool(self._flags.get("neutral_initial_profile"))
        column = self._input_dict.get("nn0_profile")
        annulus = self._input_dict.get("nn0_annulus_profile")
        if not enabled:
            configured = [
                name
                for name, value in (
                    ("nn0_profile", column),
                    ("nn0_annulus_profile", annulus),
                )
                if value is not None
            ]
            if configured:
                raise ValueError(
                    f"the shaped-initial-fill parameters {configured} were "
                    "configured without the neutral_initial_profile flag, "
                    "where they are inert (the run would start from the "
                    "uniform scalar nn0 and the profile would never be read); "
                    "set the flag or drop the parameters"
                )
            self._nn0_profile = None
            self._nn0_annulus_profile = None
            return
        if annulus is not None and not self._neutral_two_zone:
            raise ValueError(
                "nn0_annulus_profile requires the neutral_two_zone flag: "
                "without that closure there is one chamber-mean neutral "
                "field and no annulus density for the profile to initialize. "
                "Set neutral_two_zone, or fold the annulus inventory into "
                "nn0_profile"
            )
        if column is None:
            raise ValueError(
                "the neutral_initial_profile flag requires nn0_profile (a "
                f"per-cell sequence of length nx={int(self._geometry.cells)} "
                "of absolute neutral densities [cm^-3]). There is no default: "
                "the flag's whole content is the profile the caller computed"
            )
        if self._input_dict.get("nn0") is not None:
            raise ValueError(
                "neutral_initial_profile supersedes the scalar nn0 for BOTH "
                f"zones, but nn0={self._input_dict['nn0']!r} was supplied as "
                "well. There is no precedence rule to apply and a silent one "
                "would hide which fill the run actually started from: set "
                "nn0=None on a shaped run, and put the uniform level into "
                "nn0_profile if that is what is wanted"
            )
        if self._flags.get("neutral_equilibration"):
            raise ValueError(
                "neutral_initial_profile cannot be combined with "
                "neutral_equilibration: start_simulation() seeds nn (and nn_a) "
                "from the equilibration result AFTER construction, so the "
                "shaped fill would be built and then OVERWRITTEN without a "
                "trace. The two are alternative ways to state the same "
                "initial condition -- clear the flag on a shaped run"
            )
        if self._input_dict.get("restart_from") is not None:
            raise ValueError(
                "neutral_initial_profile cannot be combined with restart_from "
                "for the same reason neutral_equilibration cannot: the restart "
                "payload replaces the whole initial condition after "
                "construction, so the shaped fill would be silently discarded. "
                "A restart payload IS the initial neutral profile"
            )
        self._nn0_profile = neutral_initial_profile_values(
            self._geometry, column, "nn0_profile"
        )
        self._nn0_annulus_profile = (
            None
            if annulus is None
            else neutral_initial_profile_values(
                self._geometry, annulus, "nn0_annulus_profile"
            )
        )

    def _initial_state(self):
        cells = self._geometry.cells
        n0 = np.full(cells, float(self._input_dict["ne0"]))
        nn0 = (
            np.full(cells, float(resolve_nn0(self._input_dict)))
            if self._nn0_profile is None
            else self._nn0_profile.copy()
        )
        u0 = np.full(cells, float(self._input_dict.get("u0")))
        Te0 = np.full(cells, float(self._input_dict["Te0"]))
        Ti0 = np.full(cells, float(self._input_dict["Ti0"]))
        return conservative_from_primitives(
            n=n0,
            nn=nn0,
            u=u0,
            Te=Te0,
            Ti=Ti0,
            ion_mass_g=self._ion_mass_g,
            un=np.zeros(cells) if self._neutral_momentum else None,
            # Both zones start at the same fill density -- the free-molecular
            # equilibrium of the zone exchange; annulus-free cells carry the
            # value inertly. A shaped run may address the annulus separately,
            # in which case its own profile stands in for that convention.
            nn_a=(
                None
                if not self._neutral_two_zone
                else (
                    nn0.copy()
                    if self._nn0_annulus_profile is None
                    else self._nn0_annulus_profile.copy()
                )
            ),
            un_a=np.zeros(cells) if self._neutral_two_momentum else None,
            # Pre-plasma the gas IS at the wall temperature, so this initial
            # condition is exact rather than a convention.
            Tn_K=(
                NEUTRAL_ENERGY_FLOOR_T_K if self._neutral_energy else None
            ),
        )

    @staticmethod
    def _gas_constants(gas_type):
        if gas_type == "He":
            return m_He_cgs, 4, 4, I_ion
        # Ruled 2026-08-27, the [gas-constants-h-arm] successor row of the
        # h-quarantine: this arm was the last m_p_cgs consumer in cablp/.
        if gas_type == "H":
            raise ValueError(
                "gas_type='H' is not available: the hydrogen arm of "
                "_gas_constants was retired 2026-08-27 as dead code. It "
                "returned proton constants that no construction could ever "
                "carry into physics -- the solver is helium-only (D3, "
                "2026-08-21) and refuses gas_type != 'He' a few lines later "
                "in __init__, at the Phelps He+/He sigma_in_model gate. "
                "Accepted: 'He'."
            )
        raise ValueError(f"unsupported gas_type {gas_type!r}; expected 'He'")


def _finite_or_nan(value):
    if value is None:
        return np.nan
    return float(value)
