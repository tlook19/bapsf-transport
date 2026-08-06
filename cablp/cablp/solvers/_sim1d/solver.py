"""Solver implementation for the conservative axial 1D LAPD model."""

import math
import warnings
from dataclasses import dataclass, replace
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from .core.config import (
    default_config,
    load_config,
    resolve_config,
    resolve_nn0,
)
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
    empty_ignition_diagnostics,
)
from .core.integrator import (
    floor_state_vector,
    ssprk2_step,
)
from .core.state import (
    NEUTRAL_ANNULUS_NAME,
    STATE_NAMES_1D,
    ConservativeState1D,
    apply_state_floors,
    assert_finite_state,
    conservative_from_primitives,
    derive_state,
    pack_state,
    state_field_names,
    unpack_state,
)
from .core.timestep import suggest_timestep
from .physics.conduction import heat_conduction_rhs, implicit_heat_conduction_step
from .physics.kinetic_dvm import (
    ANNULUS_FLIGHT_MODELS as KINETIC_DVM_ANNULUS_FLIGHT_MODELS,
    ELASTIC_MODELS as KINETIC_DVM_ELASTIC_MODELS,
    EXCHANGE_MODELS as KINETIC_DVM_EXCHANGE_MODELS,
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
    BEAM_GAP_LEDGER_POWER_ATOL,
    beam_gap_ledger_mismatch,
    beam_ionization_rhs,
    beam_ionization_rhs_terms,
    cathode_boundary_state,
    cathode_power_balance_terms_W,
    cathode_sample_indices,
    cathode_source_terms,
    solve_cathode_boundary,
    tail_reflect_face,
    validate_cathode_Rp_model,
    validate_cathode_solver_model,
)
from .physics.cathode import (
    advance_circuit_current_driven,
    idriven_result_evaluator,
    idriven_vdis_evaluator,
)
from .physics.energy import (
    electron_cooling_rhs,
    electron_cooling_rhs_terms,
    electron_ion_exchange_rhs,
    ion_charge_exchange_rhs,
)
from .physics.flux import ion_sound_speed, plasma_flux_rhs, plasma_flux_rhs_terms
from .physics.neutrals import (
    GAS_PUFF_DIAGNOSTIC_FIELDS,
    gas_puff_rate_profile,
    _effective_pump_speed,
    neutral_exchange_coefficients,
    neutral_exchange_rhs,
    neutral_exchange_two_zone_rhs,
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
    recombination_energy_return_rhs,
)
from .physics.sources import (
    add_state_rhs,
    anode_collection_rhs,
    boundary_absorption_rhs,
    characteristic_boundary_rhs,
    ion_neutral_collision_rhs,
    ion_neutral_drag_rhs,
    ion_neutral_frictional_heating_rhs,
    ion_neutral_thermalization_rhs,
    neutral_momentum_wall_rhs,
    neutral_momentum_two_zone_rhs,
    neutral_wind_two_zone_factors,
    neutral_wind_velocity,
    flux_tube_geometry_rhs,
    hyperbolic_energy_correction_rhs,
    pressure_work_rhs,
)
from .results.compat import add_sim3_compat_aliases
from cablp.funcs._adas import he_rate_temperature_range_eV, he_rates
from cablp.funcs._beam_deposition import (
    HE_E_STOP_EV,
    he_mean_secondary_energy_eV,
)
from cablp.funcs._cross import charge_ex_react
from cablp.funcs._kernels import PROVENANCE as KERNEL_PROVENANCE
from cablp.vars._cons import I_Ry, I_ion, ev_to_erg, kb_cgs, m_He_cgs, m_p_cgs


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
    "P_comp",
    "P_prim",
    "P_ohmic",
    "P_cathode_e",
    "P_cathode_i",
    "P_cathode_i_pl",
    "P_anode_e",
    "P_anode_i",
    "P_anode_i_pl",
    "P_net",
    "P_net2",
    "P_loss",
    "beam_bypass_fraction",
    "l_b",
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


class _RawStageError(ValueError):
    def __init__(self, y, stage, reason, detail):
        super().__init__(f"{stage}: {reason}")
        self.y = np.asarray(y, dtype=float).copy()
        self.stage = str(stage)
        self.reason = str(reason)
        self.detail = dict(detail)


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


OPERATOR_SPLITTINGS = ("lie", "strang")


def _validate_operator_splitting(splitting):
    """Return ``splitting`` unchanged if it names an implemented composition."""
    if splitting not in OPERATOR_SPLITTINGS:
        raise ValueError(
            "operator_splitting must be one of "
            f"{sorted(OPERATOR_SPLITTINGS)} (got {splitting!r})"
        )
    return splitting


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


def _bad_array_summary(values, *, mode="nonfinite", max_indices=8):
    values = np.asarray(values, dtype=float)
    if mode == "negative":
        mask = values < 0.0
    else:
        mask = ~np.isfinite(values)
    bad = np.flatnonzero(mask)
    if bad.size == 0:
        return None
    finite = values[np.isfinite(values)]
    return {
        "count": int(bad.size),
        "indices": bad[:max_indices].astype(int).tolist(),
        "values": values[bad[:max_indices]].astype(float).tolist(),
        "nan_count": int(np.count_nonzero(np.isnan(values))),
        "posinf_count": int(np.count_nonzero(np.isposinf(values))),
        "neginf_count": int(np.count_nonzero(np.isneginf(values))),
        "finite_min": float(np.min(finite)) if finite.size else np.nan,
        "finite_max": float(np.max(finite)) if finite.size else np.nan,
    }


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
        progress_callback=None,
        progress_tracker=None,
        progress_interval_s=1.0e-4,
    ):
        self._input_dict, self._flags = resolve_config(input_dict, input_flags)
        self._progress_callback = progress_callback
        self._progress_tracker = progress_tracker
        self._progress_interval_s = (
            1.0e-4 if progress_interval_s is None else progress_interval_s
        )
        self._gas_type = self._input_dict.get("gas_type", "He")
        (
            self._ion_mass_g,
            self._mu,
            self._mu_neutral,
            self._I_ion,
        ) = self._gas_constants(self._gas_type)
        self._geometry = build_geometry(self._input_dict, self._flags)
        self._active_plasma_topology = bool(
            self._flags.get("active_plasma_topology", False)
        )
        self._raw_stage_validation = bool(
            self._flags.get("raw_stage_validation", False)
        )
        self._hyperbolic_wave_speed = str(
            self._input_dict.get("hyperbolic_wave_speed", "isothermal")
        )
        self._hyperbolic_energy_consistent = bool(
            self._flags.get("hyperbolic_energy_consistent", False)
        )
        self._characteristic_boundary = bool(
            self._flags.get("characteristic_boundary", False)
        )
        self._beam_anode_interception = bool(
            self._flags.get("beam_anode_interception", True)
        )
        # R4.3 / audit A7+A8: the moment-closed reduced ion-neutral collision
        # operator (Phelps He+/He). Presence-gated -- when on it replaces the
        # four legacy ion-neutral terms; when off it is a strict no-op. He-only.
        self._ion_neutral_moment_closure = bool(
            self._flags.get("ion_neutral_moment_closure", False)
        )
        if self._ion_neutral_moment_closure and self._gas_type != "He":
            raise ValueError(
                "ion_neutral_moment_closure uses the Phelps He+/He cross "
                f"sections and requires gas_type='He' (got {self._gas_type!r})"
            )
        # R5 stance flip: the "phelps" presheath sigma_in model (default) shares
        # the He-only Phelps cross section; hydrogen configs must select
        # "constant" or "cx_derived".
        _sigma_in_model = str(self._input_dict.get("sigma_in_model", "phelps"))
        if _sigma_in_model == "phelps" and self._gas_type != "He":
            raise ValueError(
                "sigma_in_model='phelps' uses the Phelps He+/He cross section "
                f"and requires gas_type='He' (got {self._gas_type!r}); select "
                "'constant' or 'cx_derived' for other gases"
            )
        self._validate_r1_configuration_presence()
        self._validate_neutral_seed_cache_config()
        self._validate_equilibration_gas_puff_on()
        _x = float(self._input_dict.get("R_comp_partition", 1.0))
        if not (0.0 <= _x <= 1.0):
            raise ValueError(
                "R_comp_partition (the external fraction of R_comp) must be in "
                f"[0, 1] (got {_x})"
            )
        _R_mesh = float(self._input_dict.get("R_mesh_ohm", 0.0))
        if _R_mesh < 0.0:
            raise ValueError(
                f"R_mesh_ohm (anode-mesh resistance) must be >= 0 (got {_R_mesh})"
            )
        exchange_model = str(
            self._input_dict.get("neutral_exchange_model", "knudsen")
        )
        if exchange_model not in ("constant", "knudsen"):
            raise ValueError(
                "neutral_exchange_model must be 'constant' or 'knudsen' "
                f"(got {exchange_model!r})"
            )
        self._validate_phase_config()
        self._validate_gas_puff_config()
        self._neutral_momentum = bool(self._flags.get("neutral_momentum", False))
        if (
            self._neutral_momentum
            and self._input_dict.get("ion_neutral_drag_model", "constant")
            == "slip"
        ):
            raise ValueError(
                "neutral_momentum is mutually exclusive with "
                "ion_neutral_drag_model='slip': the slip closure is the "
                "evolved M_n equation's own local steady state"
            )
        self._neutral_two_zone = bool(self._flags.get("neutral_two_zone", False))
        if self._neutral_two_zone:
            if (
                str(self._input_dict.get("neutral_exchange_model", "knudsen"))
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
            self._zone_exchange_cm3_s = neutral_zone_exchange_conductance(
                geometry=self._geometry,
                Tn_K=float(self._input_dict.get("Tn_K", 300.0)),
                mu_neutral=self._mu_neutral,
            )
            self._zone_axial_coeffs = two_zone_knudsen_coefficients(
                geometry=self._geometry,
                Tn_K=float(self._input_dict.get("Tn_K", 300.0)),
                mu_neutral=self._mu_neutral,
                clausing_scale=float(
                    self._input_dict.get("neutral_clausing_scale", 1.0)
                ),
            )
        else:
            self._zone_volumes = None
            self._zone_exchange_cm3_s = None
            self._zone_axial_coeffs = None
        self._neutral_model = str(
            self._input_dict.get("neutral_model", "moment")
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
                    self._input_dict.get("neutral_kinetic_refresh_s", 5e-4)
                ),
                refresh_tol=float(
                    self._input_dict.get("neutral_kinetic_refresh_tol", 0.2)
                ),
                update_s=1.0e-5,
                nvz=int(self._input_dict.get("neutral_kinetic_nvz", 48)),
                nvp=int(self._input_dict.get("neutral_kinetic_nvp", 12)),
            )
            if self._kinetic.refresh_s <= 0.0:
                raise ValueError(
                    "neutral_kinetic_refresh_s must be positive"
                )
        else:
            self._kinetic = None
        self._dvm = None
        self._dvm_cadence_s = 0.0
        self._dvm_tn_feedback = False
        self._dvm_engaged = False
        self._dvm_next_s = 0.0
        self._dvm_last_s = 0.0
        self._dvm_transfer_relax_fraction = 1.0
        self._dvm_step_transfer = None
        if self._neutral_model == "kinetic_dvm":
            self._configure_kinetic_dvm()
        else:
            # The bounded-chord annulus is an operator inside the transient
            # DVM; asking for it anywhere else has nothing to act on, and a
            # selector that silently does nothing is the trap this refuses.
            flights = str(
                self._input_dict.get(
                    "neutral_kinetic_dvm_annulus_flights", "rates"
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
        # R5.1 / audit A11: gated fluid<->circuit Picard coupling (default off).
        self._coupled_circuit_picard = bool(
            self._flags.get("coupled_circuit_picard", False)
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
                self._input_dict.get("circuit_picard_tol_rel", 1.0e-2)
            )
            self._circuit_picard_max_iter = int(
                self._input_dict.get("circuit_picard_max_iter", 3)
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
        # R5.2 / audit A9: flux-limited electron heat conduction (default off).
        self._electron_heat_flux_limit = bool(
            self._flags.get("electron_heat_flux_limit", False)
        )
        self._heat_flux_limiter_f = float(
            self._input_dict.get("heat_flux_limiter_f", 0.3)
        )
        if self._electron_heat_flux_limit and self._heat_flux_limiter_f <= 0.0:
            raise ValueError(
                "heat_flux_limiter_f must be > 0 when electron_heat_flux_limit "
                f"is on (got {self._heat_flux_limiter_f})"
            )
        self._heat_flux_limiter_exponent = float(
            self._input_dict.get("heat_flux_limiter_exponent", 1.0)
        )
        if self._electron_heat_flux_limit and self._heat_flux_limiter_exponent <= 0.0:
            raise ValueError(
                "heat_flux_limiter_exponent must be > 0 when "
                f"electron_heat_flux_limit is on (got {self._heat_flux_limiter_exponent})"
            )
        # Floor-aware drain exemption on the "surface_loss" dt bound (default
        # off; presence-gated: None disables the exemption branch entirely so
        # the off path is bit-exact historical behavior).
        _surface_loss_floor_exempt = bool(
            self._flags.get("surface_loss_floor_exempt", False)
        )
        _surface_loss_floor_exempt_rtol = float(
            self._input_dict.get("surface_loss_floor_exempt_rtol", 1e-3)
        )
        if _surface_loss_floor_exempt and not (
            0.0 < _surface_loss_floor_exempt_rtol < 1.0
        ):
            raise ValueError(
                "surface_loss_floor_exempt_rtol must be in (0, 1) when "
                "surface_loss_floor_exempt is on "
                f"(got {_surface_loss_floor_exempt_rtol})"
            )
        self._surface_loss_floor_exempt_rtol = (
            _surface_loss_floor_exempt_rtol
            if _surface_loss_floor_exempt
            else None
        )
        _max_steps_action = str(
            self._input_dict.get("max_steps_action", "raise")
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
            "dt_min_lock_max_steps", 250000
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
        self._neutral_momentum_radial = str(
            self._input_dict.get("neutral_momentum_radial", "uniform")
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
        # Geometry and Tn_fit are fixed for the run, so the two-zone factors
        # are computed once here; None selects the uniform (legacy) closure
        # in every consumer.
        if self._neutral_momentum_radial == "two_zone":
            (
                self._wind_column_factor,
                self._wind_wall_rate,
            ) = neutral_wind_two_zone_factors(
                geometry=self._geometry,
                Tn_eV=float(self._input_dict.get("Tn_fit", 0.1)),
                ion_mass_g=self._ion_mass_g,
            )
        else:
            self._wind_column_factor = None
            self._wind_wall_rate = None
        self._validate_neutral_jet_config()
        self._recombination_energy_return = bool(
            self._input_dict.get("recombination_energy_return", False)
        )
        if self._recombination_energy_return:
            if (
                str(self._input_dict.get("atomic_rate_model", "adas"))
                != "adas"
            ):
                raise ValueError(
                    "recombination_energy_return requires "
                    "atomic_rate_model='adas' (the PRB radiated-power "
                    "booking has no janev counterpart)"
                )
            if bool(self._flags.get("icool_recomb", False)):
                raise ValueError(
                    "recombination_energy_return already charges the full "
                    "PRB; combining it with icool_recomb double-charges "
                    "the recombination photons"
                )
        if bool(
            self._input_dict.get("adas_low_te_extension", False)
        ) and bool(self._flags.get("icool_recomb", False)):
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
        self._floors = {
            "n": float(self._input_dict["ne_floor"]),
            "nn": float(self._input_dict["nn_floor"]),
            "Te": float(self._input_dict["Te_floor"]),
            "Ti": float(self._input_dict["Ti_floor"]),
        }
        self._floor_ledger = self._empty_floor_ledger()
        initial_raw = self._initial_state()
        self._state = apply_state_floors(
            initial_raw, self._floors, self._ion_mass_g
        )
        self._accumulate_floor_ledger(
            self._floor_additions(initial_raw, self._state)
        )
        self._init_sample_smoothing()
        self._y = pack_state(self._state)
        self._derived = derive_state(self._state, self._floors, self._ion_mass_g)
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
        # Last accepted sheath solve's current, used only by the measured-tail
        # phase gate. The evolved loop state itself is _circuit_I_loop.
        self._circuit_I_prev = 0.0
        self._circuit_V_cap = None  # lazily V_bank; drains when C_bank_F set
        # Fail at construction, not at the first cathode solve mid-run:
        # unknown model strings and the unsupported TwinCathode combination.
        validate_cathode_Rp_model(self._input_dict, self._flags)
        self._cathode_solver_model = validate_cathode_solver_model(
            self._input_dict, self._flags
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
        # Cathode warming state: the evolving emitter surface temperature [K]
        # (config cathode_warming_model). None = static T_s.
        warming_model = str(
            self._input_dict.get("cathode_warming_model", "none")
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
                self._input_dict.get("cathode_heat_capacity_J_per_K", 3.0)
            ) <= 0.0:
                raise ValueError(
                    "cathode_heat_capacity_J_per_K must be positive"
                )
            eps = float(self._input_dict.get("cathode_emissivity", 0.7))
            if not 0.0 < eps <= 1.0:
                raise ValueError(
                    f"cathode_emissivity must be in (0, 1] (got {eps})"
                )
            if float(
                self._input_dict.get("cathode_conduction_W_per_K", 0.0)
            ) < 0.0:
                raise ValueError(
                    "cathode_conduction_W_per_K must be non-negative"
                )
            if float(Ts_base) <= float(
                self._input_dict.get("cathode_env_T_K", 300.0)
            ):
                raise ValueError(
                    "cathode_Ts_base_K must exceed cathode_env_T_K"
                )
            self._cathode_Ts_K = float(Ts_base)
        # Surface-state coverage (cathode_surface_model="ads_des",
        # M5a): theta in [0, 1] is the contaminant
        # coverage raising the effective work function,
        # phi_eff = phi_clean + (phi_wf - phi_clean) * theta, evolving as
        #   dtheta/dt = k_ads (1-theta) - [nu0 e^(-E_des/kT_s) + sigma Gamma_i] theta
        # (adsorption / thermal desorption / ion-stimulated desorption).
        # In-shot the ion term dominates (M5a: the fluence-cleaning limit);
        # the other channels are carried for the M5b cycle map and default
        # to zero. None = static phi_wf (historical).
        surface_model = str(
            self._input_dict.get("cathode_surface_model", "none")
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
            for _sk in (
                "cathode_cleaning_sigma_cm2",
                "cathode_ads_rate_per_s",
                "cathode_desorption_prefactor_per_s",
            ):
                if float(self._input_dict.get(_sk, 0.0)) < 0.0:
                    raise ValueError(f"{_sk} must be non-negative")
            self._cathode_theta = 1.0
        # Per-shot surface energy ledger [J] (power_balance only): running
        # integrals of the balance terms over accepted steps. The net
        # (heater + ion - rad - emis - cond) is the shot's unreturned
        # energy into the emitting skin; cond is what the heater-held
        # substrate absorbed -- the quantity Tom's open-loop-heater drift
        # hypothesis makes checkable against the ES1 trim cadence
        # (a ~sub-kW net imbalance corresponds to ±8 K per 20-30 min).
        self._cathode_energy_ledger_J = {
            "heater": 0.0, "ion": 0.0, "rad": 0.0, "emis": 0.0, "cond": 0.0,
        }
        if float(self._input_dict.get("beam_deposition_smoothing_cm", 0.0)) < 0.0:
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
        _bpt = str(self._input_dict.get("beam_product_transport", "local"))
        if _bpt not in ("local", "nonlocal"):
            raise ValueError(
                "beam_product_transport must be 'local' or 'nonlocal' "
                f"(got {_bpt!r})"
            )
        if _bpt == "nonlocal" and str(
            self._input_dict.get("beam_deposition_model", "beer_lambert")
        ) != "csda":
            raise ValueError(
                "beam_product_transport='nonlocal' requires "
                "beam_deposition_model='csda' (the products it transports "
                "are the CSDA ray's; under beer_lambert it would be a "
                "silent no-op)"
            )
        # WP-E QL heating locality. Same discipline as WP-D above: the module
        # validates too, but a misconfiguration must fail at CONSTRUCTION, and
        # every combination in which the walk machinery could not act is an
        # INCOMPLETE configuration rather than a silent no-op. The walk needs
        # (a) the CSDA module to be the thing depositing, and (b) an anomalous
        # channel actually producing the power it carries.
        _hat = str(self._input_dict.get("heating_anomalous_transport", "local"))
        if _hat not in ("local", "tail_walk"):
            raise ValueError(
                "heating_anomalous_transport must be 'local' or 'tail_walk' "
                f"(got {_hat!r})"
            )
        if _hat == "tail_walk":
            if str(
                self._input_dict.get("beam_deposition_model", "beer_lambert")
            ) != "csda":
                raise ValueError(
                    "heating_anomalous_transport='tail_walk' requires "
                    "beam_deposition_model='csda' (the anomalous heating it "
                    "transports is the CSDA ray's; under beer_lambert it "
                    "would be a silent no-op)"
                )
            if str(
                self._input_dict.get("beam_anomalous_model", "none")
            ) == "none":
                raise ValueError(
                    "heating_anomalous_transport='tail_walk' requires an "
                    "active anomalous channel "
                    "(beam_anomalous_model='quasilinear'); with no anomalous "
                    "drag there is no power to carry and the setting would "
                    "be a silent no-op"
                )
            _tail_eV = float(
                self._input_dict.get("heating_anomalous_tail_energy_eV", 75.0)
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
                "heating_anomalous_tail_energy_keying", "phi_c"
            )
        )
        if _keying not in ("phi_c", "fixed"):
            raise ValueError(
                "heating_anomalous_tail_energy_keying must be 'phi_c' or "
                f"'fixed' (got {_keying!r})"
            )
        _cath_bnd = str(
            self._input_dict.get(
                "heating_anomalous_tail_cathode_boundary", "reflect"
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
            "heating_anomalous_tail_phi_c_fraction", None
        )
        if _phi_frac is not None and float(_phi_frac) not in (0.25, 0.5, 1.0):
            raise ValueError(
                "heating_anomalous_tail_phi_c_fraction must be one of the "
                "declared bracket arms 0.25, 0.5 or 1.0 (got "
                f"{_phi_frac!r}); it is a bracket the campaign reports across, "
                "not a value to fit"
            )
        if _hat == "tail_walk":
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
                self._flags.get("TwinCathode", False)
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
        # duplicate the module's own guards here: a misconfiguration must fail
        # before the first cathode solve, not hours into a run. The two
        # tail-energy bars are the module's, evaluated on the module's own
        # threshold constants and <W_sec> convention rather than restated, so
        # they cannot drift apart from the arithmetic they protect.
        # Under K7 phi_c keying the LIVE E_tail is f*phi_c(t), which no
        # construction-time check can see; the bars below then bind only the
        # (inert) fixed rung and the module's own copies of them, evaluated on
        # the live value at every solve, are what actually protect the depth-1
        # truncation. That is stated in the config docstring for the keying.
        _tion = str(
            self._input_dict.get("heating_anomalous_tail_ionization", "off")
        )
        if _tion not in ("off", "on"):
            raise ValueError(
                "heating_anomalous_tail_ionization must be 'off' or 'on' "
                f"(got {_tion!r})"
            )
        if _tion == "on":
            if _hat != "tail_walk":
                raise ValueError(
                    "heating_anomalous_tail_ionization='on' requires "
                    "heating_anomalous_transport='tail_walk' (the ionizing "
                    "channel belongs to the QL tail walkers; without them "
                    "the setting would be a silent no-op). "
                    "heating_anomalous_transport accepts 'local' or "
                    "'tail_walk'; heating_anomalous_tail_ionization accepts "
                    f"'off' or 'on' (got {_hat!r} and {_tion!r})"
                )
            if _tail_eV <= HE_E_STOP_EV:
                raise ValueError(
                    "heating_anomalous_tail_ionization='on' needs "
                    "heating_anomalous_tail_energy_eV above the lowest He "
                    f"inelastic threshold ({HE_E_STOP_EV} eV); at {_tail_eV} "
                    "eV a tail walker cannot ionize or excite at all and the "
                    "setting would be a silent no-op"
                )
            _W_sec_launch = he_mean_secondary_energy_eV(
                _tail_eV, I_ion_eV=float(self._I_ion)
            )
            if _W_sec_launch >= HE_E_STOP_EV:
                raise ValueError(
                    "heating_anomalous_tail_ionization='on' banks each "
                    "secondary as local heat, which is the correct depth-1 "
                    "truncation only while the mean secondary energy stays "
                    "below the lowest inelastic threshold; at "
                    f"heating_anomalous_tail_energy_eV={_tail_eV} eV it is "
                    f"{_W_sec_launch:.4f} eV against {HE_E_STOP_EV} eV, so "
                    "the cascade the walk does not follow would be mis-banked "
                    "as thermalized"
                )
        _fc = float(self._input_dict.get("beam_clump_fraction", 0.0))
        if not 0.0 <= _fc < 1.0:
            raise ValueError(
                f"beam_clump_fraction must be in [0, 1) (got {_fc})"
            )
        if float(self._input_dict.get("beam_clump_enhancement", 1.0)) < 1.0:
            raise ValueError(
                "beam_clump_enhancement must be >= 1 (got "
                f"{self._input_dict.get('beam_clump_enhancement')})"
            )
        _fli = float(self._input_dict.get("gas_puff_local_ionization_fraction", 0.0))
        if not 0.0 <= _fli < 1.0:
            raise ValueError(
                f"gas_puff_local_ionization_fraction must be in [0, 1) (got {_fli})"
            )
        if _fli > 0.0 and self._flags.get("neutral_two_zone", False):
            raise ValueError(
                "gas_puff_local_ionization_fraction is not supported with "
                "neutral_two_zone (annulus puff routing); disable one"
            )
        self._cathode_solve = None
        # Item-35 ledger tripwire: latched so the warning fires once per run.
        self._beam_gap_ledger_warned = False
        self._last_result = None
        self._last_neutral_equilibration_result = None
        self._last_neutral_equilibration_summary = None
        # Set only while start_simulation drives run(); lets run() tell a direct
        # call from the equilibration-aware entry point (see run()).
        self._run_via_start_simulation = False
        if self._flags.get("debug_checks", False):
            assert_finite_state(self._state, self._derived)

    def _validate_r1_configuration_presence(self):
        """Reject R1-audited controls that would otherwise be silent no-ops."""
        frozen_controls = {
            "front_flux_model": (
                str(self._input_dict.get("front_flux_model")),
                "sonic_relaxation",
            ),
            "D_amb_model": (
                str(self._input_dict.get("D_amb_model")),
                "cs_dz",
            ),
            "D_amb": (
                float(self._input_dict.get("D_amb")),
                0.0,
            ),
            "cathode_model": (
                str(self._input_dict.get("cathode_model")),
                "disabled",
            ),
        }
        changed = [
            name
            for name, (actual, canonical) in frozen_controls.items()
            if actual != canonical
        ]
        if changed:
            raise ValueError(
                "R1-audited compatibility/boundary controls are frozen at "
                "their checkpoint values until their owning repair supplies "
                "a replacement operator; noncanonical values would be silent "
                "no-ops: "
                + ", ".join(changed)
            )
        # A13 (R3.3, 2026-07-24): the resolved-boundary surface-loss controls are
        # DEPRECATED 0D artifacts. In the lumped model they stood in for I_sat
        # that could not be separated between the cathode and anode; the resolved
        # geometry measures the Bohm I_sat to each electrode face directly (the
        # characteristic ghost-cell boundary / anode collection), so the area
        # scales and enables have NO operator to control and are never consumed.
        # Their owning repair (R3.3) retires them rather than wiring a 0D fudge
        # into the resolved boundary. Loud on non-default use (no silent no-op).
        deprecated_surface_controls = {
            "source_surface_loss": (
                bool(self._flags.get("source_surface_loss", True)), True,
            ),
            "end_surface_loss": (
                bool(self._flags.get("end_surface_loss", True)), True,
            ),
            "source_surface_area_scale": (
                float(self._input_dict.get("source_surface_area_scale", 1.8)),
                1.8,
            ),
            "end_surface_area_scale": (
                float(self._input_dict.get("end_surface_area_scale", 1.0)),
                1.0,
            ),
        }
        deprecated = [
            name
            for name, (actual, default) in deprecated_surface_controls.items()
            if actual != default
        ]
        if deprecated:
            warnings.warn(
                "resolved-boundary surface-loss controls "
                + ", ".join(deprecated)
                + " are DEPRECATED 0D artifacts (they stood in for un-separated "
                "cathode/anode I_sat); the resolved geometry measures the Bohm "
                "I_sat to each electrode face directly, so they have no effect. "
                "Remove them; reproduce 0D-scaled runs at tag "
                "legacy-final-2026-07-22.",
                DeprecationWarning,
                stacklevel=2,
            )
        # R5 stance flip (2026-07-25) deprecations. These paths remain runnable
        # (A/B arms + tag reproducibility) but are superseded by the repaired
        # production baseline; a non-default/active use warns.
        if not self._ion_neutral_moment_closure:
            warnings.warn(
                "the legacy ion-neutral drag/CX/thermalization path "
                "(ion_neutral_moment_closure=False, with sigma_in_model "
                "'constant'/'cx_derived', b_ion_neutral_drag, "
                "ion_neutral_drag_model, b_ion_neutral_thermalization, and the "
                "Tn_fit collision temperature) is DEPRECATED: the Phelps "
                "moment-closed operator (ion_neutral_moment_closure) is the "
                "production drag baseline. Still runnable as an A/B arm and for "
                "reproducing old results at tag legacy-final-2026-07-22.",
                DeprecationWarning,
                stacklevel=2,
            )
        _gp_mode = str(self._input_dict.get("gas_puff_mode", "square"))
        if _gp_mode in ("pulse_decay_to_level", "decay_after_breakdown", "double_erf"):
            warnings.warn(
                f"gas_puff_mode={_gp_mode!r} is DEPRECATED (the measured "
                "waveform is 'square'); retained runnable only for the frozen "
                "waveform-comparison figures.",
                DeprecationWarning,
                stacklevel=2,
            )
        _deprecated_selectors = {
            "D_amb_model": (str(self._input_dict.get("D_amb_model", "cs_dz")), "cs_dz"),
            "cathode_model": (
                str(self._input_dict.get("cathode_model", "disabled")), "disabled",
            ),
        }
        _sel = [n for n, (a, d) in _deprecated_selectors.items() if a != d]
        if _sel:
            warnings.warn(
                "legacy-compat selectors " + ", ".join(_sel) + " are DEPRECATED "
                "and never consumed by the conservative solver (D_amb_model was "
                "a _sim3-compat knob; cathode_model is superseded by the "
                "cathode_coupling flag).",
                DeprecationWarning,
                stacklevel=2,
            )
        for name in ("Te_birth_ionization", "Ti_birth_ionization"):
            value = self._input_dict.get(name)
            if isinstance(value, str):
                if value not in {"local", "floor"}:
                    raise ValueError(
                        f"{name} must be 'local', 'floor', or a finite "
                        f"non-negative numeric eV value (got {value!r})"
                    )
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                numeric = np.nan
            if not np.isfinite(numeric) or numeric < 0.0:
                raise ValueError(
                    f"{name} must be 'local', 'floor', or a finite "
                    f"non-negative numeric eV value (got {value!r})"
                )
        birth_energy_model = str(
            self._input_dict.get("ionization_birth_energy_model", "legacy")
        )
        if birth_energy_model not in {"legacy", "conservative"}:
            raise ValueError(
                "ionization_birth_energy_model must be 'legacy' or "
                f"'conservative' (got {birth_energy_model!r})"
            )
        if self._hyperbolic_wave_speed not in {"isothermal", "adiabatic"}:
            raise ValueError(
                "hyperbolic_wave_speed must be 'isothermal' or 'adiabatic' "
                f"(got {self._hyperbolic_wave_speed!r})"
            )
        if self._characteristic_boundary:
            # R3.1 characteristic ghost-cell Bohm outflow (audit A1/A16). It acts
            # only on plasma-terminating (absorbing) faces, which exist only in
            # the resolved geometry; without them the flag would be a silent
            # no-op (R1d discipline).
            absorbing = np.asarray(
                getattr(self._geometry, "plasma_absorbing", np.zeros(0)),
                dtype=bool,
            )
            if not np.any(absorbing):
                raise ValueError(
                    "characteristic_boundary requires plasma-terminating "
                    "(absorbing) faces, which exist only in the resolved "
                    "geometry (resolved_boundaries=True); it would otherwise be "
                    "a silent no-op"
                )
        # R4.1 anode-mesh beam interception (audit A15) is the production default
        # (correct csda physics). Like beam_coulomb_model / beam_anomalous_model it
        # is a csda control: it perturbs the operator under beam_deposition_model=
        # "csda" with resolved anode faces, and is inert under beer_lambert (which
        # never launches the CSDA module) or where no anode faces exist. The
        # _csda_beam_deposition wiring applies it only when eta>0 and anode faces
        # are present, so no construction rejection is needed.
        if self._raw_stage_validation and self._flags.get("Plasma", True):
            for initial_name, floor_name in (
                ("Te0", "Te_floor"),
                ("Ti0", "Ti_floor"),
            ):
                initial = float(self._input_dict[initial_name])
                floor = float(self._input_dict[floor_name])
                if not initial > floor:
                    raise ValueError(
                        f"{initial_name} must be strictly greater than "
                        f"{floor_name} when raw_stage_validation=True "
                        f"(got {initial} <= {floor})"
                    )

    def _validate_equilibration_gas_puff_on(self):
        """Reject a nonsense equilibration puff width (loud, at construction).

        ``equilibration_gas_puff_on_s`` overrides the neutral-equilibration
        inner sim's per-cycle puff-ON window. ``None`` means "unset" (fall back
        to ``tau_discharge``); anything else must be a real, finite, positive
        duration that fits inside one puff/off cycle. A zero, negative, or
        longer-than-the-cycle value would silently produce a 0% or >100% duty
        instead of the measured window.
        """
        raw = self._input_dict.get("equilibration_gas_puff_on_s", None)
        if raw is None:
            return
        try:
            puff_on = float(raw)
        except (TypeError, ValueError):
            raise ValueError(
                "equilibration_gas_puff_on_s (the equilibration puff-ON window "
                f"[s]) must be a number or None (got {raw!r})"
            ) from None
        if not np.isfinite(puff_on) or puff_on <= 0.0:
            raise ValueError(
                "equilibration_gas_puff_on_s (the equilibration puff-ON window "
                f"[s]) must be finite and > 0 (got {puff_on!r}); use None to "
                "fall back to tau_discharge"
            )
        tau_cycle = float(self._input_dict.get("tau_cycle", 0.0))
        if tau_cycle > 0.0 and puff_on > tau_cycle:
            raise ValueError(
                "equilibration_gas_puff_on_s (the equilibration puff-ON window "
                f"[s]) must fit inside one puff/off cycle: got {puff_on!r} > "
                f"tau_cycle={tau_cycle!r}"
            )

    def _validate_neutral_seed_cache_config(self):
        """Reject an incoherent cached-neutral-seed configuration (loud, at build).

        ``use_cached_neutral_seed`` replaces the live neutral equilibration with a
        cached seed, so it requires the equilibration pipeline to be selected
        (``neutral_equilibration`` + ``launch_plasma_after_equilibration``) and a
        cache path. A missing path or a contradictory flag would otherwise be a
        silent no-op.
        """
        if not self._flags.get("use_cached_neutral_seed", False):
            return
        problems = []
        if not self._flags.get("neutral_equilibration", False):
            problems.append(
                "neutral_equilibration must be ON (the cache seeds that pipeline)"
            )
        if not self._flags.get("launch_plasma_after_equilibration", False):
            problems.append(
                "launch_plasma_after_equilibration must be ON (nothing to seed "
                "otherwise)"
            )
        if not self._input_dict.get("neutral_seed_cache_dir"):
            problems.append(
                "neutral_seed_cache_dir must be set to the seed-database directory"
            )
        if problems:
            raise ValueError(
                "use_cached_neutral_seed is ON but the configuration is "
                "incoherent: " + "; ".join(problems)
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
            self._input_dict.get("gas_puff_local_ionization_fraction", 0.0)
        ) > 0.0:
            raise ValueError(
                "neutral_model='kinetic_dvm' is incompatible with "
                "gas_puff_local_ionization_fraction > 0: that channel "
                "removes puff neutrals on the fluid book, which the "
                "kinetic particle ledger would not see. Accepted: "
                "gas_puff_local_ionization_fraction = 0"
            )
        cadence = float(
            self._input_dict.get("neutral_kinetic_dvm_cadence_s", 2.5e-5)
        )
        if not (cadence > 0.0):
            raise ValueError(
                "neutral_kinetic_dvm_cadence_s must be positive "
                f"(got {cadence})"
            )
        accommodation = float(
            self._input_dict.get("neutral_kinetic_dvm_accommodation", 1.0)
        )
        if not 0.0 <= accommodation <= 1.0:
            raise ValueError(
                "neutral_kinetic_dvm_accommodation is a surface property in "
                f"[0, 1] (got {accommodation})"
            )
        elastic = str(
            self._input_dict.get("neutral_kinetic_dvm_elastic", "phelps_iso")
        )
        if elastic not in KINETIC_DVM_ELASTIC_MODELS:
            raise ValueError(
                "neutral_kinetic_dvm_elastic must be one of "
                f"{KINETIC_DVM_ELASTIC_MODELS} (got {elastic!r})"
            )
        exchange = str(
            self._input_dict.get("neutral_kinetic_dvm_exchange", "cauchy_chord")
        )
        if exchange not in KINETIC_DVM_EXCHANGE_MODELS:
            raise ValueError(
                "neutral_kinetic_dvm_exchange must be one of "
                f"{KINETIC_DVM_EXCHANGE_MODELS} (got {exchange!r})"
            )
        flights = str(
            self._input_dict.get(
                "neutral_kinetic_dvm_annulus_flights", "rates"
            )
        )
        if flights not in KINETIC_DVM_ANNULUS_FLIGHT_MODELS:
            raise ValueError(
                "neutral_kinetic_dvm_annulus_flights must be one of "
                f"{KINETIC_DVM_ANNULUS_FLIGHT_MODELS} (got {flights!r})"
            )
        tn_feedback = bool(
            self._input_dict.get("neutral_kinetic_dvm_tn_feedback", False)
        )
        if tn_feedback and self._characteristic_boundary:
            raise ValueError(
                "neutral_kinetic_dvm_tn_feedback is incompatible with the "
                "characteristic_boundary stance: there the circuit's "
                "cathode sheath factor samples the same presheath through a "
                "path that carries no Tn, so feeding the measured Tn to only "
                "the fluid half would break the shared sheath-edge density. "
                "Accepted: tn_feedback with characteristic_boundary off"
            )
        relax_fraction = float(
            self._input_dict.get(
                "neutral_kinetic_dvm_transfer_relax_fraction", 0.5
            )
        )
        if not 0.0 < relax_fraction <= 1.0:
            raise ValueError(
                "neutral_kinetic_dvm_transfer_relax_fraction is the share of "
                "a cell's ion-energy margin the tick-frozen coupling drain "
                "may consume in one step and must lie in (0, 1] "
                f"(got {relax_fraction})"
            )
        self._dvm_cadence_s = cadence
        self._dvm_tn_feedback = tn_feedback
        self._dvm_transfer_relax_fraction = relax_fraction
        anode_faces = np.asarray(
            getattr(self._geometry, "anode_face_indices", ()), dtype=int
        )
        self._dvm = TransientDVM(
            geometry=self._geometry,
            nvz=int(self._input_dict.get("neutral_kinetic_dvm_nvz", 48)),
            nvp=int(self._input_dict.get("neutral_kinetic_dvm_nvp", 12)),
            accommodation=accommodation,
            elastic_model=elastic,
            exchange_model=exchange,
            annulus_flights=flights,
            transparency=1.0 - float(self._input_dict.get("eta", 0.358)),
            mesh_face=int(anode_faces[0]) if anode_faces.size else -999,
            s_L=self._dvm_end_sticking("S_pump_L"),
            s_R=self._dvm_end_sticking("S_pump_R"),
        )

    def _dvm_end_sticking(self, key):
        """Return the end-plane sticking probability of a pump speed [L/s].

        The TPMC/KN2Zone convention: the pumping speed over the one-way
        300 K thermal flux through the end plane, clipped to a probability.
        """
        speed = float(self._input_dict.get(key, 0.0))
        if speed <= 0.0:
            return 0.0
        vbar = math.sqrt(
            8.0 * kb_cgs * 300.0 / (math.pi * self._mu_neutral * m_p_cgs)
        )
        area = math.pi * float(np.asarray(self._geometry.Rm_cm)[-1]) ** 2
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

    def rhs(self, y=None, include_heat_conduction=True, time=None):
        """Return the packed explicit RHS for the current scaffold physics."""
        state_rhs = self._zero_rhs_state()
        for term in self.rhs_terms(
            y=y,
            include_heat_conduction=include_heat_conduction,
            time=time,
        ).values():
            state_rhs = add_state_rhs(state_rhs, term)
        # With optional fields on, the packed RHS must always match the
        # state vector's width, even when no term touched them (pads zeros).
        return pack_state(
            state_rhs,
            neutral_momentum=True if self._neutral_momentum else None,
            neutral_two_zone=True if self._neutral_two_zone else None,
            neutral_annulus_momentum=(
                True if self._neutral_two_momentum else None
            ),
        )

    def rhs_terms(self, y=None, include_heat_conduction=True, time=None):
        """Return named conservative RHS contributions for diagnostics."""
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
        geometry_terms = {}
        if self._flags.get("end_expansion_geometry", False):
            geometry_terms["flux_tube_geometry"] = (
                self._zero_rhs_state()
                if not self._flags.get("Plasma", True)
                or self._neutral_prebreakdown_active(time=time)
                else self.flux_tube_geometry_rhs(state=state)
            )
        if not self._flags.get("Plasma", True) or self._neutral_prebreakdown_active(
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
            terms = {
                **zone_terms,
                **geometry_terms,
                **kinetic_terms,
                "plasma_advective_flux": self._zero_rhs_state(),
                "plasma_front_flux": self._zero_rhs_state(),
                "boundary_absorption": self._zero_rhs_state(),
                "characteristic_boundary": self._zero_rhs_state(),
                "pressure_work": self._zero_rhs_state(),
                "hyperbolic_energy_correction": self._zero_rhs_state(),
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
            return self._apply_active_plasma_topology(terms)
        plasma_terms = self.plasma_flux_rhs_terms(state=state)
        reaction_terms = self.reaction_rhs_terms(state=state)
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
        terms = {
            **zone_terms,
            **geometry_terms,
            "plasma_advective_flux": plasma_terms["plasma_advective_flux"],
            "plasma_front_flux": plasma_terms["plasma_front_flux"],
            "boundary_absorption": (
                self._zero_rhs_state()
                if self._characteristic_boundary
                else self.boundary_absorption_rhs(
                    state=state, cathode_solve=cathode_solve, time=time
                )
            ),
            "characteristic_boundary": (
                self.characteristic_boundary_rhs(
                    state=state, cathode_solve=cathode_solve, time=time
                )
                if self._characteristic_boundary
                else self._zero_rhs_state()
            ),
            "pressure_work": self.pressure_work_rhs(state=state),
            "hyperbolic_energy_correction": (
                self.hyperbolic_energy_correction_rhs(state=state)
                if self._hyperbolic_energy_consistent
                else self._zero_rhs_state()
            ),
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
            "cathode_surface_loss": self.cathode_source_terms(
                state=state,
                cathode_solve=cathode_solve,
                time=time,
            ).rhs,
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
                terms = {
                    name: self._strip_dvm_rows(name, term)
                    for name, term in terms.items()
                }
            terms["neutral_kinetic_dvm_coupling"] = (
                self.neutral_kinetic_dvm_coupling_rhs()
            )
        return self._apply_active_plasma_topology(terms)

    def _apply_active_plasma_topology(self, terms):
        """Mask plasma-coupled terms on typed plasma-dead cells."""
        if not self._active_plasma_topology:
            return terms
        neutral_only = {
            "neutral_zone_exchange",
            "neutral_momentum_wall",
            "neutral_wind_advection",
            "neutral_exchange",
            "neutral_sources",
            "neutral_kinetic_relaxation",
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
        active = np.asarray(self._geometry.plasma_active, dtype=bool)

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
        )

    def floor_state_vector(self, y):
        """Apply configured density and temperature floors to a packed vector."""
        return floor_state_vector(
            y=y,
            cells=self._geometry.cells,
            floors=self._floors,
            ion_mass_g=self._ion_mass_g,
            neutral_momentum=self._neutral_momentum,
            neutral_two_zone=self._neutral_two_zone,
            neutral_annulus_momentum=self._neutral_two_momentum,
        )

    @staticmethod
    def _empty_floor_ledger():
        return {
            "n_particles_added": 0.0,
            "nn_particles_added": 0.0,
            "nn_a_particles_added": 0.0,
            "Ee_energy_added_erg": 0.0,
            "Ei_energy_added_erg": 0.0,
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
        packed_summary = _bad_array_summary(y)
        if packed_summary is not None:
            raise _RawStageError(
                y,
                stage,
                "nonfinite_state",
                {"stage": stage, "fields": {"packed_y": packed_summary}},
            )
        state = self._unpack(y)
        fields = {
            "n": state.n,
            "nn": state.nn,
            "M": state.M,
            "Ee": state.Ee,
            "Ei": state.Ei,
        }
        if state.M_n is not None:
            fields["M_n"] = state.M_n
        if state.nn_a is not None:
            fields["nn_a"] = state.nn_a
        if state.M_n_a is not None:
            fields["M_n_a"] = state.M_n_a
        nonfinite = {
            name: summary
            for name, values in fields.items()
            if (summary := _bad_array_summary(values)) is not None
        }
        if nonfinite:
            raise _RawStageError(
                y,
                stage,
                "nonfinite_state",
                {"stage": stage, "fields": nonfinite},
            )
        negative_density = {
            name: summary
            for name, values in (
                ("n", state.n),
                ("nn", state.nn),
                ("nn_a", state.nn_a),
            )
            if values is not None
            and (
                summary := _bad_array_summary(values, mode="negative")
            )
            is not None
        }
        if negative_density:
            raise _RawStageError(
                y,
                stage,
                "negative_density",
                {"stage": stage, "fields": negative_density},
            )
        negative_energy = {
            name: summary
            for name, values in (("Ee", state.Ee), ("Ei", state.Ei))
            if (
                summary := _bad_array_summary(values, mode="negative")
            )
            is not None
        }
        if negative_energy:
            raise _RawStageError(
                y,
                stage,
                "negative_energy",
                {"stage": stage, "fields": negative_energy},
            )

    def _step_cache_snapshot(self):
        return SimpleNamespace(
            cathode_x0=_copy_cache_value(self._cathode_x0),
            cathode_x0_twin=_copy_cache_value(self._cathode_x0_twin),
            cathode_beam_cross=self._cathode_beam_cross.copy(),
            cathode_solve=self._cathode_solve,
        )

    def _restore_step_cache(self, snapshot):
        self._cathode_x0 = _copy_cache_value(snapshot.cathode_x0)
        self._cathode_x0_twin = _copy_cache_value(snapshot.cathode_x0_twin)
        self._cathode_beam_cross = np.asarray(
            snapshot.cathode_beam_cross,
            dtype=float,
        ).copy()
        self._cathode_solve = snapshot.cathode_solve

    def _attempt_step(self, dt=None, operator_split=None):
        """Return a candidate step without committing state, time, or caches."""
        if operator_split is None:
            operator_split = self._flags.get("implicit_heat_conduction", False)
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
                    not self._flags.get("Plasma", True)
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
                        rhs_func=lambda yy, tt: self.rhs(yy, time=tt),
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
        return StepAttempt1D(
            y=np.asarray(y_next, dtype=float),
            dt=dt,
            operator_split=bool(operator_split),
            solver_cache=candidate_cache,
            floor_ledger=attempt_floor_ledger,
            raw_rejection_reason=raw_rejection_reason,
            raw_rejection_detail=raw_rejection_detail,
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
        matrix = np.eye(geometry.cells, dtype=float)
        rhs = np.asarray(state.nn, dtype=float).copy()

        coeff = np.asarray(self.neutral_exchange_coefficients(), dtype=float)
        for face, conductance in enumerate(coeff):
            left = face
            right = face + 1
            left_rate = float(conductance) / float(geometry.neutral_volume_cm3[left])
            right_rate = float(conductance) / float(
                geometry.neutral_volume_cm3[right]
            )
            matrix[left, left] += dt * left_rate
            matrix[left, right] -= dt * left_rate
            matrix[right, right] += dt * right_rate
            matrix[right, left] -= dt * right_rate

        # Anchored by role, matching neutrals.neutral_source_sink_rhs: these two
        # paths must stay consistent or the neutral-equilibration path desyncs
        # from the explicit one.
        puff_index, puff_twin_index = puff_cell_indices(geometry)
        pump_left_index, pump_right_index = pump_cell_indices(geometry)

        if source_kwargs["pump_enabled"]:
            elbow = source_kwargs["pump_elbow_conductance_lps"]
            matrix[pump_left_index, pump_left_index] += dt * pump_rate(
                _effective_pump_speed(
                    source_kwargs["S_pump_L"],
                    elbow if is_plenum_cell(geometry, pump_left_index) else None,
                ),
                geometry.neutral_volume_cm3[pump_left_index],
            )
            matrix[pump_right_index, pump_right_index] += dt * pump_rate(
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
                end=0,
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
                    end=-1,
                )

        nn_next = np.linalg.solve(matrix, rhs)
        # M_n passes through untouched: this step runs pre-plasma, where
        # there is no drag to drive a wind.
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
        )

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
        """
        geometry = self._geometry
        cells = geometry.cells
        source_kwargs = self._neutral_source_kwargs(time=time)
        V_col, V_ann = self._zone_volumes
        matrix = np.eye(2 * cells, dtype=float)
        rhs = np.concatenate(
            (
                np.asarray(state.nn, dtype=float),
                np.asarray(state.nn_a, dtype=float),
            )
        )

        column_coeff, annulus_coeff = self._zone_axial_coeffs
        for offset, coeff, volumes in (
            (0, np.asarray(column_coeff, dtype=float), V_col),
            (cells, np.asarray(annulus_coeff, dtype=float), V_ann),
        ):
            for face, conductance in enumerate(coeff):
                if conductance <= 0.0:
                    continue
                left = offset + face
                right = offset + face + 1
                left_rate = float(conductance) / float(volumes[face])
                right_rate = float(conductance) / float(volumes[face + 1])
                matrix[left, left] += dt * left_rate
                matrix[left, right] -= dt * left_rate
                matrix[right, right] += dt * right_rate
                matrix[right, left] -= dt * right_rate

        for cell, conductance in enumerate(
            np.asarray(self._zone_exchange_cm3_s, dtype=float)
        ):
            if conductance <= 0.0:
                continue
            col_rate = float(conductance) / float(V_col[cell])
            ann_rate = float(conductance) / float(V_ann[cell])
            matrix[cell, cell] += dt * col_rate
            matrix[cell, cells + cell] -= dt * col_rate
            matrix[cells + cell, cells + cell] += dt * ann_rate
            matrix[cells + cell, cell] -= dt * ann_rate

        puff_index, puff_twin_index = puff_cell_indices(geometry)
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
                matrix[index, index] += dt * rate
                matrix[cells + index, cells + index] += dt * rate

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
                    end=end,
                )
            particles = puff * np.asarray(
                geometry.neutral_volume_cm3, dtype=float
            )
            into_annulus = V_ann > 0.0
            rhs[cells:] += dt * np.where(
                into_annulus, particles / np.maximum(V_ann, 1e-300), 0.0
            )
            rhs[:cells] += dt * np.where(
                into_annulus, 0.0, particles / np.maximum(V_col, 1e-300)
            )

        solution = np.linalg.solve(matrix, rhs)
        return ConservativeState1D(
            n=state.n.copy(),
            nn=(
                np.maximum(solution[:cells], self._floors["nn"])
                if apply_density_floor
                else solution[:cells]
            ),
            M=state.M.copy(),
            Ee=state.Ee.copy(),
            Ei=state.Ei.copy(),
            M_n=None if state.M_n is None else state.M_n.copy(),
            nn_a=(
                np.maximum(solution[cells:], self._floors["nn"])
                if apply_density_floor
                else solution[cells:]
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

        fields = {
            "n": state1.n,
            "nn": state1.nn,
            "M": state1.M,
            "Ee": state1.Ee,
            "Ei": state1.Ei,
            "u": derived1.u,
            "Te": derived1.Te,
            "Ti": derived1.Ti,
            "pe": derived1.pe,
            "pi": derived1.pi,
            "p": derived1.p,
        }
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

        density_limit = float(self._input_dict.get("max_density_step_fraction", 0.0))
        if density_limit > 0.0 and _max_relative_change(
            state0.n,
            state1.n,
            self._floors["n"],
        ) > density_limit:
            return "density_step_fraction", {}

        neutral_limit = float(self._input_dict.get("max_neutral_step_fraction", 0.0))
        if neutral_limit > 0.0 and _max_relative_change(
            state0.nn,
            state1.nn,
            self._floors["nn"],
        ) > neutral_limit:
            return "neutral_step_fraction", {}

        energy_limit = float(self._input_dict.get("max_energy_step_fraction", 0.0))
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
        dt_min = float(self._input_dict.get("dt_min", 1e-12))
        max_retries = int(self._input_dict.get("max_step_retries", 8))
        reject_factor = float(self._input_dict.get("dt_reject_factor", 0.5))
        retries_enabled = bool(
            self._input_dict.get("adaptive_retries_enabled", True)
        )
        if max_retries < 0:
            raise ValueError(f"max_step_retries must be non-negative ({max_retries})")
        if not 0.0 < reject_factor < 1.0:
            raise ValueError(
                "dt_reject_factor must be between 0 and 1 "
                f"(got {reject_factor})"
            )

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
        # Book the DVM deferred-transfer ledger FIRST: it reads the step's
        # start state and the transfer the step was scoped with, both of
        # which the lines below overwrite.
        if self._dvm_rows_superseded():
            self._dvm_book_step_transfer(attempt.dt)
        self._restore_step_cache(attempt.solver_cache)
        self._set_state_vector(attempt.y)
        self._accumulate_floor_ledger(
            getattr(attempt, "floor_ledger", self._empty_floor_ledger())
        )
        self._time += float(attempt.dt)
        # Electrode sample smoothing: fold the newly accepted state into the
        # supply-average EMA before any accepted-state consumer reads it.
        self._update_sample_smoothing(attempt.dt)
        # K4a refresh: kinetic target solves run
        # only at ACCEPTED states -- deterministic across step retries --
        # and only once the plasma phase is live (the pre-breakdown fill
        # stays moment).
        if (
            self._kinetic is not None
            and self._flags.get("Plasma", True)
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
            and self._flags.get("Plasma", True)
            and not self._neutral_prebreakdown_active()
        ):
            if not self._dvm_engaged:
                self._dvm_engage()
            elif self._time >= self._dvm_next_s:
                self._dvm_advance(self._time - self._dvm_last_s)
        # Retain the accepted solve current for the measured-tail phase gate.
        solve = self._cathode_solve
        if solve is not None and solve.beam_result is not None:
            # Clamp the loop-current state to a generous physical ceiling
            # (~5x the dead-short bank current): the sheath solve has a
            # pathological huge-phi_c branch under inflated effective EMFs,
            # and an unclamped I_prev lets that branch feed a multiplicative
            # runaway (seen: I_prev -> 5e22 A). Real loop currents sit ~50x
            # below this ceiling, so the clamp is inert in normal operation.
            I_ceiling = 5.0 * float(
                self._input_dict.get("V_bank", 0.0)
            ) / max(float(self._input_dict.get("R_comp", 1.0)), 1e-6)
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
            )(self._circuit_I_loop)
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
                P_heat, P_ion, P_rad, P_emis, P_cond = (
                    cathode_power_balance_terms_W(
                        T_s_K=self._cathode_Ts_K,
                        P_ion_W=float(result.P_cathode_i)
                        * self._cathode_surface_ion_retention,
                        I_eth_star_A=I_emis,
                        input_dict=self._surface_effective_input_dict(),
                    )
                )
                C_th = float(
                    self._input_dict.get("cathode_heat_capacity_J_per_K", 3.0)
                )
                # Semi-implicit in the linearized loss: dT = dt*P_net/(C_th
                # + dt*dP_loss/dT). At production dt/tau ~ 5e-5 this is the
                # explicit update to 4 decimal places; for tiny C_th it
                # cannot overshoot the radiative equilibrium and ring.
                eps = float(self._input_dict.get("cathode_emissivity", 0.7))
                area = self._input_dict.get("cathode_rad_area_cm2")
                if area is None:
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
                            "cathode_conduction_W_per_K", 0.0
                        )
                    )
                )
                dT = (
                    float(attempt.dt)
                    * (P_heat + P_ion - P_rad - P_emis - P_cond)
                    / (C_th + float(attempt.dt) * G_lin)
                )
                self._cathode_Ts_K = max(
                    self._cathode_Ts_K + dT,
                    float(self._input_dict.get("cathode_env_T_K", 300.0)),
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
            Gamma_i = I_i_A / (1.602176634e-19 * area_cm2)
            sigma_cl = float(
                self._input_dict.get("cathode_cleaning_sigma_cm2", 0.0)
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
            k_ads = float(
                self._input_dict.get("cathode_ads_rate_per_s", 0.0)
            )
            nu0 = float(
                self._input_dict.get(
                    "cathode_desorption_prefactor_per_s", 0.0
                )
            )
            nu_th = 0.0
            if nu0 > 0.0:
                T_des = (
                    float(self._cathode_Ts_K)
                    if self._cathode_Ts_K is not None
                    else float(self._input_dict.get("T_s", 300.0))
                )
                nu_th = nu0 * math.exp(
                    -float(
                        self._input_dict.get(
                            "cathode_desorption_energy_eV", 3.0
                        )
                    )
                    / (8.617333262e-5 * max(T_des, 1.0))
                )
            loss = nu_th + sigma_cl * Gamma_i
            self._cathode_theta = (
                self._cathode_theta + float(attempt.dt) * k_ads
            ) / (1.0 + float(attempt.dt) * (k_ads + loss))
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
            if bank_off:
                V_src = 0.0
            elif C_bank_id is not None and float(C_bank_id) > 0.0:
                if self._circuit_V_cap is None:
                    self._circuit_V_cap = float(
                        self._input_dict.get("V_bank", 0.0)
                    )
                V_src = float(self._circuit_V_cap)
            else:
                V_src = float(self._input_dict.get("V_bank", 0.0))
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
                R_comp_ohm=float(self._input_dict.get("R_comp", 0.0))
                * float(self._input_dict.get("R_comp_partition", 1.0)),
                L_H=float(self._input_dict.get("L_parasitic_H", 0.0)),
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
        return self.get_initial_snapshot()

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
        "_cathode_solve",
    )

    def _picard_snapshot(self):
        """Capture the pre-step coupled state for a Picard re-run (R5.1/A11)."""
        snap = {a: getattr(self, a) for a in self._PICARD_DIRECT_ATTRS}
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
            splitting = _validate_operator_splitting(splitting)
        if floor_func is None:
            floor_func = self.floor_state_vector
        if raw_stage_func is None and self._raw_stage_validation:
            raw_stage_func = self._validate_raw_stage

        def heat(y_in, sub_dt):
            state = self.implicit_heat_conduction_step(dt=sub_dt, y=y_in)
            raw = pack_state(state)
            if raw_stage_func is not None:
                raw_stage_func(raw, "implicit_heat")
            return floor_func(raw)

        def explicit(y_in, sub_dt):
            return ssprk2_step(
                y0=y_in,
                dt=sub_dt,
                rhs_func=lambda yy, tt: self.rhs(
                    yy,
                    include_heat_conduction=False,
                    time=tt,
                ),
                floor_func=floor_func,
                time=self._time,
                raw_stage_func=raw_stage_func,
            )

        if splitting == "strang":
            half = 0.5 * dt
            return heat(explicit(heat(y0, half), dt), half)
        return heat(explicit(y0, dt), dt)

    def _operator_splitting(self):
        return _validate_operator_splitting(
            self._input_dict.get("operator_splitting", "lie")
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
        if (
            self._flags.get("neutral_equilibration", False)
            and not self._run_via_start_simulation
        ):
            warnings.warn(
                "neutral_equilibration is ON but run() was called directly, so "
                "NO equilibration happens: only start_simulation() runs the "
                "puff/off accumulation and seeds nn from it. This run starts "
                "from the direct nn0 fill "
                f"({resolve_nn0(self._input_dict, self._flags):.3g} cm^-3) "
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
            max_steps = int(self._input_dict.get("max_steps", 0))
        max_steps = int(max_steps)
        unlimited_steps = max_steps <= 0

        dt_save = float(self._input_dict.get("dt_save", 1e-5))
        t_save_start = float(self._input_dict.get("t_save_start", 0.0))
        max_output_steps = int(self._input_dict.get("max_output_steps", 0))
        saved = []
        diagnostics = []
        timestep_rejection_events = []
        t_last_save = -np.inf
        previous_accepted_dt = None
        time_tol = max(1e-15, 1e-12 * max(abs(t_end), 1.0))
        run_start = float(self._time)
        progress_wall_start = perf_counter()
        self._run_start_for_phase_events = run_start
        dt_growth_enabled = bool(self._input_dict.get("dt_growth_enabled", True))
        dt_growth_factor = float(self._input_dict.get("dt_growth_factor", 1.25))
        if dt_growth_enabled and dt_growth_factor <= 1.0:
            raise ValueError(
                "dt_growth_factor must be > 1 when dt growth is enabled "
                f"(got {dt_growth_factor})"
            )
        dynamic_current_t_end = (
            not explicit_t_end
            and self._flags.get("Plasma", True)
            and self._phase_transition_mode() == "current"
        )
        # A switch-open abort shortens t_end the same way a breakdown trigger
        # does, in EITHER phase-transition mode, so an aborted run winds down
        # and stops instead of crawling to the configured end time.
        dynamic_t_end = not explicit_t_end and self._flags.get("Plasma", True)

        def should_save(t):
            if max_output_steps > 0 and len(saved) >= max_output_steps:
                return False
            if t + 1e-15 < t_save_start:
                return False
            if dt_save <= 0.0:
                return True
            return t - t_last_save >= dt_save - time_tol or abs(t - t_end) <= time_tol

        def next_save_time_after(t):
            if max_output_steps > 0 and len(saved) >= max_output_steps:
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

        if should_save(self._time):
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
                    else self._flags.get("implicit_heat_conduction", False)
                )
            )
            # dt_min lock guard. Only an ADAPTIVE step can be locked: with a
            # caller-supplied dt the clamp does not set the step, so a fixed-dt
            # run cannot crawl and is not counted. Consecutiveness is the
            # discriminator -- self-releasing clamp episodes are a known-good
            # family (see scripts/dtmin_census_runlengths.txt) and must not
            # abort, while a genuine lock never releases.
            if dt is None and diag.clamped_to_dt_min:
                consecutive_dt_min_clamps += 1
                if consecutive_dt_min_clamps > self._dt_min_lock_max_steps:
                    raise self._dt_min_lock_error(
                        diag=diag,
                        consecutive=consecutive_dt_min_clamps,
                    )
            else:
                consecutive_dt_min_clamps = 0
            step_dt = diag.dt if dt is None else float(dt)
            step_cap = diag.active_constraint if dt is None else "fixed_dt"
            if dt is None and dt_growth_enabled and previous_accepted_dt is not None:
                step_dt, step_cap = cap_step(
                    step_dt,
                    step_cap,
                    previous_accepted_dt * dt_growth_factor,
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
            previous_accepted_dt = float(attempt.dt)
            step_diag = replace(
                diag,
                accepted_dt=float(attempt.dt),
                step_cap=step_cap,
                retry_count=int(retry_count),
                rejection_reason=rejection_reason,
            )
            diagnostics.append(step_diag)
            steps += 1
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
        self._last_result = result
        return result

    def _dt_min_lock_error(self, diag, consecutive):
        """Return the RuntimeError for a run stuck at dt_min.

        Names the true minimizing bound (not "dt_min"), what it asked for, and
        the cell closest to the density floor -- the drained floor-pinned cell
        that makes ``_negative_margin_timestep`` return exactly zero.
        """
        n = np.asarray(self.state.n, dtype=float)
        n_floor = float(self._floors["n"])
        cell = int(np.argmin(n - n_floor))
        return RuntimeError(
            f"dt_min lock: the timestep was clamped up to dt_min on "
            f"{consecutive} consecutive steps, exceeding "
            f"dt_min_lock_max_steps={self._dt_min_lock_max_steps}, at "
            f"t={self._time:.9e} s (phase={diag.phase!r}). The true active "
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
        # cleared, so _validate_neutral_seed_cache_config would reject the inner
        # config and a database MISS would raise instead of equilibrating and
        # populating the database (the caller stores the result). The cache-
        # control flags are inert to the seed signature, so clearing this here
        # cannot change the stored entry's key or content.
        flags["use_cached_neutral_seed"] = False
        # The equilibration OWNS its neutral start; it must not inherit the
        # outer run's nn0. nn0 is the direct-run fill (a realistic pre-shot
        # background), whereas this inner sim accumulates the fill from
        # near-vacuum over `cycles` puff/off cycles -- exactly what the
        # cablp/vars/_nn_table.py generator did, at nn0_init = 1e8. Pinning it
        # here decouples the two paths, so the direct-run default can move
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
            tau_cycle=float(self._input_dict.get("tau_cycle", 0.0)),
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
        if not self._flags.get("Plasma", True):
            cycles = int(self._input_dict.get("cycles", 1))
            if cycles <= 0:
                raise ValueError(f"cycles must be positive (got {cycles})")
            tau_cycle = max(float(self._input_dict.get("tau_cycle", 0.0)), 0.0)
            if tau_cycle <= 0.0:
                tau_cycle = max(
                    float(self._input_dict.get("tau_discharge", 0.0)),
                    0.0,
                )
            return float(cycles) * tau_cycle

        tau_prebreakdown = max(
            float(self._input_dict.get("tau_prebreakdown", 0.0)),
            0.0,
        )
        tau_breakdown = max(float(self._input_dict.get("tau_breakdown", 0.0)), 0.0)
        tau_discharge = max(float(self._input_dict.get("tau_discharge", 0.0)), 0.0)
        tau_afterglow = max(float(self._input_dict.get("tau_afterglow", 0.0)), 0.0)
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
        if self._flags.get("neutral_equilibration", False):
            use_db = self._flags.get("use_cached_neutral_seed", False)
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
                if not self._flags.get("launch_plasma_after_equilibration", False):
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
        """Return an explicit timestep suggestion and diagnostics."""
        state = self.state if y is None else self._unpack(y)
        if time is None:
            time = self._time
        plasma_enabled = self._flags.get(
            "Plasma",
            True,
        ) and not self._neutral_prebreakdown_active(time=time)
        if include_heat_conduction is None:
            include_heat_conduction = (
                plasma_enabled
                and not self._flags.get("implicit_heat_conduction", False)
            )
        dt_min = float(self._input_dict.get("dt_min", 1e-12))
        dt_max = float(self._input_dict.get("dt_max", 1e-6))
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
        diag = suggest_timestep(
            state=state,
            floors=self._floors,
            ion_mass_g=self._ion_mass_g,
            mu=self._mu,
            geometry=self._geometry,
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
            plasma_source_rhs=plasma_source_rhs,
            source_floor_exempt_rtol=self._surface_loss_floor_exempt_rtol,
            neutral_rows_superseded=dvm_superseded,
            cfl=float(self._input_dict.get("cfl", 0.4)),
            density_dt_fraction=float(
                self._input_dict.get("density_dt_fraction", 0.25)
            ),
            neutral_dt_fraction=float(
                self._input_dict.get("neutral_dt_fraction", 0.25)
            ),
            heat_dt_fraction=float(self._input_dict.get("heat_dt_fraction", 0.25)),
            drag_dt_fraction=float(self._input_dict.get("drag_dt_fraction", 0.5)),
            dt_min=dt_min,
            dt_max=dt_max,
            include_front=plasma_enabled and self._flags.get("front_flux", True),
            alpha_front=float(self._input_dict.get("alpha_front", 1.0)),
            plasma_active=(
                self._geometry.plasma_active
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
        RUNS: the R3.1 characteristic ghost-cell flux when
        ``characteristic_boundary`` is on, the legacy volumetric absorber
        when it is off. The two disagree face by face -- reading the wrong
        one bounds a term the step never applies while leaving the applied
        one unbounded (the same wrong-operator class the recycle channel was
        fixed for).

        The engaged DVM arm's coupling term joins the bundle for the same
        reason: it is a volumetric ion momentum/energy source of unbounded
        magnitude, frozen between neutral ticks, and until K2d it sat outside
        every timestep bound -- injecting a 1e12 erg/cm^3/s drain changed the
        suggested dt by exactly zero. The rate bounded here is the tick's
        BOOKED transfer, not the step's floor-limited application: this bound
        is the honest question "how big a step can carry what the kinetic
        side booked", and the limiter is the separate backstop for when the
        answer is below ``dt_min``.
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
        boundary = (
            self.characteristic_boundary_rhs
            if self._characteristic_boundary
            else self.boundary_absorption_rhs
        )
        rhs = boundary(
            state=state,
            cathode_solve=cathode_solve,
            time=time,
        )
        rhs = add_state_rhs(
            rhs,
            self.anode_collection_rhs(
                state=state,
                cathode_solve=cathode_solve,
                time=time,
            ),
        )
        rhs = add_state_rhs(
            rhs,
            self.cathode_source_terms(
                state=state,
                cathode_solve=cathode_solve,
                time=time,
            ).rhs,
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
            self._flags.get("Plasma", True)
            and self._flags.get("neutral_prebreakdown", False)
        ):
            return 0.0
        return max(float(self._input_dict.get("tau_neutral_prebreakdown", 0.0)), 0.0)

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
        override = self._input_dict.get("equilibration_gas_puff_on_s", None)
        if override is None:
            return max(float(self._input_dict.get("tau_discharge", 0.0)), 0.0)
        return float(override)

    def _equilibration_puff_on_reason(self):
        """Name of the quantity that closes the equilibration puff window."""
        if self._input_dict.get("equilibration_gas_puff_on_s", None) is None:
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
        tau_cycle = max(float(self._input_dict.get("tau_cycle", 0.0)), 0.0)
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

        if not self._flags.get("Plasma", True):
            # Equilibration lattice: read the cycle position from the SHARED
            # helper and compare boundaries against that snapped position. The
            # run loop's t_end-scaled ``time_tol`` governs only the t_end
            # window here -- it must NOT decide the puff-off instant, because
            # ``_phase_info`` cannot see it (item 37, see
            # ``_equilibration_cycle_position``).
            def in_end_window(boundary):
                return t_end is None or boundary <= t_end + time_tol

            puff_on = self._equilibration_puff_on_duration()
            tau_cycle = max(float(self._input_dict.get("tau_cycle", 0.0)), 0.0)
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
            float(self._input_dict.get("tau_prebreakdown", 0.0)),
            0.0,
        )
        tau_breakdown = max(float(self._input_dict.get("tau_breakdown", 0.0)), 0.0)
        tau_discharge = max(float(self._input_dict.get("tau_discharge", 0.0)), 0.0)
        tau_afterglow = max(float(self._input_dict.get("tau_afterglow", 0.0)), 0.0)
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
        for boundary in sorted(boundaries):
            if in_run_window(boundary):
                return float(boundary)
        return None

    def plasma_flux_rhs(self, y=None, include_front=None):
        """Return the conservative plasma flux RHS for inspection/testing."""
        state = self.state if y is None else self._unpack(y)
        use_front = self._flags.get("front_flux", True)
        if include_front is not None:
            use_front = include_front
        return plasma_flux_rhs(
            state=state,
            floors=self._floors,
            ion_mass_g=self._ion_mass_g,
            mu=self._mu,
            geometry=self._geometry,
            include_front=use_front,
            alpha_front=float(self._input_dict.get("alpha_front", 1.0)),
            active_plasma_topology=self._active_plasma_topology,
            wave_speed=self._hyperbolic_wave_speed,
            energy_consistent=self._hyperbolic_energy_consistent,
            characteristic_boundary=self._characteristic_boundary,
        )

    def plasma_flux_rhs_terms(self, y=None, state=None, include_front=None):
        """Return split conservative plasma face-flux RHS terms."""
        if state is None:
            state = self.state if y is None else self._unpack(y)
        use_front = self._flags.get("front_flux", True)
        if include_front is not None:
            use_front = include_front
        return plasma_flux_rhs_terms(
            state=state,
            floors=self._floors,
            ion_mass_g=self._ion_mass_g,
            mu=self._mu,
            geometry=self._geometry,
            include_front=use_front,
            alpha_front=float(self._input_dict.get("alpha_front", 1.0)),
            active_plasma_topology=self._active_plasma_topology,
            wave_speed=self._hyperbolic_wave_speed,
            energy_consistent=self._hyperbolic_energy_consistent,
            characteristic_boundary=self._characteristic_boundary,
        )

    def pressure_work_rhs(self, y=None, state=None):
        """Return conservative pressure-work energy sources."""
        if state is None:
            state = self.state if y is None else self._unpack(y)
        return pressure_work_rhs(
            state=state,
            floors=self._floors,
            ion_mass_g=self._ion_mass_g,
            geometry=self._geometry,
            # b_pressure_work_elec/ions removed as config knobs (R5 stance flip):
            # must be 1 for conservative pressure-work booking (hardwired).
            electron_scale=1.0,
            ion_scale=1.0,
            active_plasma_topology=self._active_plasma_topology,
        )

    def hyperbolic_energy_correction_rhs(self, y=None, state=None):
        """Return the R2 KEP energy-consistency correction (Ee, Ei sources)."""
        if state is None:
            state = self.state if y is None else self._unpack(y)
        return hyperbolic_energy_correction_rhs(
            state=state,
            floors=self._floors,
            ion_mass_g=self._ion_mass_g,
            mu=self._mu,
            geometry=self._geometry,
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

    def _cathode_jet_spec(self, cathode_solve):
        """Return the cathode-jet parameters from the current solve, or None."""
        if (
            not self._cathode_jet_enabled
            or cathode_solve is None
            or cathode_solve.beam_result is None
        ):
            return None
        phi_c = float(cathode_solve.beam_result.result.phi_c)
        if not np.isfinite(phi_c):
            return None
        T_s = float(
            self._cathode_Ts_K
            if self._cathode_Ts_K is not None
            else float(self._input_dict.get("T_s", 0.0))
        )
        return {
            "R_N": self._cathode_jet_R_N,
            "R_E": self._cathode_jet_R_E,
            "phi_c_V": max(phi_c, 0.0),
            "T_s_K": max(T_s, 0.0),
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
        }

    def boundary_absorption_rhs(
        self, y=None, state=None, cathode_solve=None, time=None
    ):
        """Return the Bohm absorption at the plasma-terminating surfaces."""
        if state is None:
            state = self.state if y is None else self._unpack(y)
        surface_kwargs = self._surface_loss_kwargs()
        cathode_solve = self._jet_cathode_solve(
            cathode_solve, self._cathode_jet_enabled, time
        )
        return boundary_absorption_rhs(
            state=state,
            floors=self._floors,
            ion_mass_g=self._ion_mass_g,
            mu=self._mu,
            geometry=self._geometry,
            alpha_isat=surface_kwargs["alpha_isat"],
            b_surface_loss=surface_kwargs["b_surface_loss"],
            sigma_in_cm2=float(self._input_dict.get("sigma_in_cm2", 5.0e-15)),
            b_presheath_length=float(
                self._input_dict.get("b_presheath_length", 1.0)
            ),
            sigma_in_model=str(
                self._input_dict.get("sigma_in_model", "constant")
            ),
            gas_type=self._gas_type,
            cathode_jet=self._cathode_jet_spec(cathode_solve),
            Tn_presheath_eV=self._dvm_presheath_Tn_eV(),
        )

    def characteristic_boundary_rhs(
        self, y=None, state=None, cathode_solve=None, time=None
    ):
        """Return the R3.1 characteristic ghost-cell Bohm outflow (audit A1/A16).

        Replaces ``boundary_absorption_rhs`` when the ``characteristic_boundary``
        flag is on: a one-sided ghost-cell KEP/Rusanov flux against the Bohm
        outflow state at each absorbing face. Reads the same surface kwargs and
        cathode jet, and follows the interior's momentum-flux form and wave speed
        so a repaired stance stays consistent.
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
            geometry=self._geometry,
            alpha_isat=surface_kwargs["alpha_isat"],
            b_surface_loss=surface_kwargs["b_surface_loss"],
            sigma_in_cm2=float(self._input_dict.get("sigma_in_cm2", 5.0e-15)),
            b_presheath_length=float(
                self._input_dict.get("b_presheath_length", 1.0)
            ),
            sigma_in_model=str(
                self._input_dict.get("sigma_in_model", "constant")
            ),
            gas_type=self._gas_type,
            cathode_jet=self._cathode_jet_spec(cathode_solve),
            wave_speed=self._hyperbolic_wave_speed,
            energy_consistent=self._hyperbolic_energy_consistent,
            # R3.2/A16: this term runs only in the repaired stance, so it always
            # routes electrode energy through the one control surface (electron
            # sheath transmission; driven electrodes owned by the circuit,
            # collector floating).
            sheath_energy_routing=True,
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
            geometry=self._geometry,
            eta=float(self._input_dict.get("eta", 0.0)),
            b_anode_collection=float(
                self._input_dict.get("b_anode_collection", 1.0)
            ),
            anode_jet=self._anode_jet_spec(cathode_solve),
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
            Tn_fit=float(self._input_dict.get("Tn_fit", 0.1)),
            wall_rate_1_s=self._wind_wall_rate,
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
            Tn_K=float(self._input_dict.get("Tn_K", 300.0)),
        )

    def neutral_wind_advection_rhs(self, y=None, state=None):
        """Return upwind advection of nn and M_n by the neutral wind."""
        if state is None:
            state = self.state if y is None else self._unpack(y)
        return neutral_wind_advection_rhs(
            state=state,
            floors=self._floors,
            ion_mass_g=self._ion_mass_g,
            geometry=self._geometry,
            mesh_faces=self._mesh_faces,
            mesh_blocked_area_cm2=self._mesh_blocked_area_cm2,
        )

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
        if not self._flags.get("ion_neutral_thermalization", False):
            return self._zero_rhs_state()
        b_thermalization = self._input_dict.get("b_ion_neutral_thermalization")
        return ion_neutral_thermalization_rhs(
            state=state,
            floors=self._floors,
            ion_mass_g=self._ion_mass_g,
            Tn_fit=float(self._input_dict.get("Tn_fit", 0.1)),
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
        drag_enabled = bool(self._flags.get("ion_neutral_drag", True))
        Tn_K = float(self._input_dict.get("Tn_K", 300.0))
        Tn_eV = Tn_K * kb_cgs / ev_to_erg
        return ion_neutral_collision_rhs(
            state=state,
            floors=self._floors,
            ion_mass_g=self._ion_mass_g,
            gas_type=self._gas_type,
            Tn_eV=Tn_eV,
            b_ion_neutral_drag=(
                float(self._input_dict.get("b_ion_neutral_drag", 1.0))
                if drag_enabled
                else 0.0
            ),
            geometry=self._geometry,
            wind_column_factor=self._wind_column_factor,
        )

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
            geometry=self._geometry,
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
        result = solve_cathode_boundary(
            state=state,
            floors=self._floors,
            ion_mass_g=self._ion_mass_g,
            mu=self._mu,
            geometry=self._geometry,
            input_dict=self._input_dict,
            input_flags=input_flags,
            beam_cross_prev=self._cathode_beam_cross,
            I_ion=self._I_ion,
            gas_type=self._gas_type,
            x0=self._cathode_x0,
            x0_twin=self._cathode_x0_twin,
            floating=floating,
            T_s_override_K=self._cathode_Ts_K,
            phi_wf_override_eV=self._cathode_phi_wf_eff(),
            circuit_I_loop_A=self._circuit_I_loop,
        )
        if update_cache:
            self._warn_beam_gap_ledger(result)
            self._cathode_solve = result
            self._cathode_x0 = result.x0_next
            self._cathode_x0_twin = result.x0_twin_next
            if result.beam_result is not None:
                self._cathode_beam_cross = (
                    result.beam_result.beam_atten_cross.copy()
                )
        return result

    def _warn_beam_gap_ledger(self, cathode_solve):
        """Warn ONCE per run if the CSDA beam gap ledger stops closing.

        The circuit charges ``eta * f_bypass`` of the emitted beam power to a
        beam it believes crosses the cathode-anode gap without coupling; the
        fluid deposits whatever the CSDA ray actually delivers. Those views of
        the same gap must agree, and they disagree silently -- no conservation
        check sees it, because each side is internally consistent. This is the
        state item 35 sat in.
        """
        if self._beam_gap_ledger_warned:
            return
        device_config = getattr(cathode_solve, "device_config", None)
        if device_config is None:
            return
        worst = beam_gap_ledger_mismatch(
            getattr(cathode_solve, "beam_gap_ledger", None),
            device_config.eta,
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

    def implicit_heat_conduction_step(self, dt, y=None, state=None):
        """Return state after one frozen-conductivity implicit heat substep."""
        if state is None:
            state = self.state if y is None else self._unpack(y)
        return implicit_heat_conduction_step(
            state=state,
            floors=self._floors,
            ion_mass_g=self._ion_mass_g,
            mu=self._mu,
            geometry=self._geometry,
            dt=dt,
            implicit_heat_scheme=self._input_dict.get(
                "implicit_heat_scheme",
                "backward_euler",
            ),
            heat_picard_iterations=int(
                self._input_dict.get("heat_picard_iterations", 0)
            ),
            heat_picard_tol=float(self._input_dict.get("heat_picard_tol", 1e-10)),
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
            )
        return neutral_exchange_rhs(
            state=state,
            geometry=self._geometry,
            exchange_coeff_cm3_s=self.neutral_exchange_coefficients(),
        )

    def neutral_zone_exchange_rhs(self, y=None, state=None):
        """Return the conservative column/annulus free-molecular exchange."""
        if state is None:
            state = self.state if y is None else self._unpack(y)
        return neutral_zone_exchange_rhs(
            state=state,
            geometry=self._geometry,
            conductance_cm3_s=self._zone_exchange_cm3_s,
        )

    def neutral_exchange_coefficients(self):
        """Return internal-face neutral exchange coefficients [cm^3/s]."""
        return neutral_exchange_coefficients(
            geometry=self._geometry,
            model=self._input_dict.get("neutral_exchange_model", "knudsen"),
            constant_coeff_cm3_s=float(
                self._input_dict.get("neutral_exchange_coeff_cm3_s", 1.0e5)
            ),
            Tn_K=float(self._input_dict.get("Tn_K", 300.0)),
            mu_neutral=self._mu_neutral,
            clausing_scale=float(self._input_dict.get("neutral_clausing_scale", 1.0)),
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

    def gas_puff_local_ionization_rhs(self, y=None, state=None, time=None):
        """Return the fresh-puff clump local-ionization source (default off)."""
        if state is None:
            state = self.state if y is None else self._unpack(y)
        f = float(self._input_dict.get("gas_puff_local_ionization_fraction", 0.0))
        if f <= 0.0:
            return self._zero_rhs_state()
        nk = self._neutral_source_kwargs(time=time)
        if not nk["gas_puff_enabled"]:
            return self._zero_rhs_state()
        puff = gas_puff_rate_profile(
            self._geometry, nk["S_gp"], nk["gas_puff_valves"],
            profile=nk["gas_puff_profile"], z_cm=nk["gas_puff_z_cm"],
            sigma_cm=nk["gas_puff_sigma_cm"], throw_cm=nk["gas_puff_throw_cm"],
            end=0,
        )
        if nk["twin_cathode"]:
            puff = puff + gas_puff_rate_profile(
                self._geometry, nk["Twin_S_gp"], nk["gas_puff_valves"],
                profile=nk["gas_puff_profile"], z_cm=nk["gas_puff_z_cm"],
                sigma_cm=nk["gas_puff_sigma_cm"], throw_cm=nk["gas_puff_throw_cm"],
                end=-1,
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
                "Te_birth_ionization", "local"
            ),
            Ti_birth_ionization=self._input_dict.get(
                "Ti_birth_ionization", "floor"
            ),
            ionization_birth_energy_model=str(
                self._input_dict.get("ionization_birth_energy_model", "legacy")
            ),
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
            b_rec_rad=float(self._input_dict.get("b_rec_rad", 1.0)),
            atomic_rate_model=str(
                self._input_dict.get("atomic_rate_model", "adas")
            ),
            enabled=self._recombination_energy_return,
            adas_low_te_extension=bool(
                self._input_dict.get("adas_low_te_extension", False)
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
        configured = bool(self._flags.get("cathode_coupling", False))
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
            and float(self._input_dict.get("L_parasitic_H", 0.0)) > 0.0
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
            "S_pump_L": float(self._input_dict.get("S_pump_L", 0.0)),
            "S_pump_R": float(self._input_dict.get("S_pump_R", 0.0)),
            "twin_cathode": self._flags.get("TwinCathode", False),
            "gas_puff_enabled": bool(phase_switches["gas_puff_enabled"]),
            "pump_enabled": bool(self._input_dict.get("pump_enabled", True)),
            "gas_puff_valves": float(self._input_dict.get("gas_puff_valves", 2)),
            "pump_elbow_conductance_lps": self._input_dict.get(
                "pump_elbow_conductance_lps"
            ),
            "gas_puff_profile": str(
                self._input_dict.get("gas_puff_profile", "cell")
            ),
            "gas_puff_z_cm": self._input_dict.get("gas_puff_z_cm"),
            "gas_puff_sigma_cm": float(
                self._input_dict.get("gas_puff_sigma_cm", 50.0)
            ),
            "gas_puff_throw_cm": float(
                self._input_dict.get("gas_puff_throw_cm", 100.0)
            ),
        }

    def _validate_gas_puff_config(self):
        mode = self._input_dict.get("gas_puff_mode", "decay_after_breakdown")
        if mode not in {
            "decay_after_breakdown",
            "pulse_decay_to_level",
            "double_erf",
            "square",
        }:
            raise ValueError(
                "gas_puff_mode must be 'decay_after_breakdown', "
                "'pulse_decay_to_level', 'double_erf', or 'square' "
                f"(got {mode!r})"
            )
        if mode == "square":
            for key in ("gas_puff_rise_width_s",):
                width = float(self._input_dict.get(key, 5.0e-4))
                if width <= 0.0:
                    raise ValueError(f"{key} must be positive (got {width})")
            for key in ("gas_puff_rise_center_s", "gas_puff_close_lag_s"):
                value = float(self._input_dict.get(key, 5.0e-4))
                if value < 0.0:
                    raise ValueError(f"{key} must be >= 0 (got {value})")
        if mode == "double_erf":
            for key in ("tau_gp_rise_width", "tau_gp_drop_width"):
                width = float(self._input_dict.get(key, 1e-3))
                if width <= 0.0:
                    raise ValueError(f"{key} must be positive (got {width})")
        tau_after_breakdown = self._input_dict.get("tau_gp_after_breakdown", None)
        if tau_after_breakdown is not None and float(tau_after_breakdown) < 0.0:
            raise ValueError(
                "tau_gp_after_breakdown must be >= 0 s, or None to keep S_gp "
                f"steady (got {tau_after_breakdown})"
            )
        tau_decay_factor = float(self._input_dict.get("tau_gp_decay_factor", 1.0))
        if tau_decay_factor <= 0.0:
            raise ValueError(
                f"tau_gp_decay_factor must be > 0 (got {tau_decay_factor})"
            )
        tau_pulse_duration = float(self._input_dict.get("tau_gp_pulse_duration", 0.0))
        if tau_pulse_duration < 0.0:
            raise ValueError(
                f"tau_gp_pulse_duration must be >= 0 (got {tau_pulse_duration})"
            )
        tau_decay_duration = float(self._input_dict.get("tau_gp_decay_duration", 1e-3))
        if tau_decay_duration <= 0.0:
            raise ValueError(
                f"tau_gp_decay_duration must be > 0 (got {tau_decay_duration})"
            )

    def _init_sample_smoothing(self):
        """Parse and seed the electrode sample-smoothing EMA (config.py).

        Tracked cells: the cathode sample cell and the first anode face's
        two flanking cells -- every (n, Te) the sheath solve reads. The EMA
        is seeded from the initial state (deterministic) and updated on
        accepted steps only, so dt-retries never move it.
        """
        raw = self._input_dict.get("cathode_sample_smoothing", None)
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
        unchanged -- the golden path is bit-exact."""
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
        )

    def _validate_neutral_jet_config(self):
        """Validate and cache the directed-recycle-jet configuration.

        The jets and the mesh accommodation are M_n physics: they require the
        neutral_momentum flag, and each channel requires the geometry feature
        it rides on (an absorbing cathode face; anode faces with eta > 0), so
        a misconfigured jet fails loudly instead of silently never firing.
        """
        p = self._input_dict
        self._cathode_jet_enabled = bool(p.get("cathode_neutral_jet", False))
        self._anode_jet_enabled = bool(p.get("anode_neutral_jet", False))
        self._mesh_accommodation = bool(
            p.get("neutral_mesh_accommodation", False)
        )
        surface_debit = bool(p.get("cathode_jet_surface_debit", False))
        for prefix, enabled in (
            ("cathode_jet", self._cathode_jet_enabled),
            ("anode_jet", self._anode_jet_enabled),
        ):
            R_N = float(p.get(f"{prefix}_R_N", 0.0))
            R_E = float(p.get(f"{prefix}_R_E", 0.0))
            if enabled and not (0.0 <= R_N <= 1.0 and 0.0 <= R_E <= 1.0):
                raise ValueError(
                    f"{prefix}_R_N and {prefix}_R_E are particle/energy "
                    "reflection coefficients and must lie in [0, 1] "
                    f"(got R_N={R_N}, R_E={R_E})"
                )
            setattr(self, f"_{prefix}_R_N", R_N)
            setattr(self, f"_{prefix}_R_E", R_E)
        needs_mn = (
            self._cathode_jet_enabled
            or self._anode_jet_enabled
            or self._mesh_accommodation
        )
        if needs_mn and not self._neutral_momentum:
            raise ValueError(
                "cathode_neutral_jet / anode_neutral_jet / "
                "neutral_mesh_accommodation are M_n momentum physics and "
                "require the neutral_momentum flag"
            )
        roles = np.asarray(self._geometry.cell_role)
        absorbing = np.asarray(
            getattr(self._geometry, "plasma_absorbing", np.zeros(0)),
            dtype=bool,
        )
        if self._cathode_jet_enabled and not (
            np.any(absorbing) and np.any(roles == "cathode")
        ):
            raise ValueError(
                "cathode_neutral_jet requires an absorbing cathode face "
                "(resolved_boundaries geometry): the jet rides the "
                "boundary-absorption recycle flux"
            )
        anode_faces = np.asarray(
            getattr(self._geometry, "anode_face_indices", ()), dtype=int
        )
        eta = float(p.get("eta", 0.0))
        if (self._anode_jet_enabled or self._mesh_accommodation) and (
            anode_faces.size == 0 or eta <= 0.0
        ):
            raise ValueError(
                "anode_neutral_jet / neutral_mesh_accommodation require "
                "anode faces with eta > 0 (resolved geometry with a mesh)"
            )
        if surface_debit and not self._cathode_jet_enabled:
            raise ValueError(
                "cathode_jet_surface_debit reads the cathode jet's R_E and "
                "requires cathode_neutral_jet"
            )
        # Reflected-energy retention for the surface power balance:
        # (1 - R_E) of the ion bombardment power stays in the surface when
        # the debit sensitivity arm is on; 1.0 (the M5a' calibration
        # convention) otherwise.
        self._cathode_surface_ion_retention = (
            1.0 - self._cathode_jet_R_E if surface_debit else 1.0
        )
        # Blocked mesh area for the wind's momentum accommodation: the open
        # fraction T = 1 - eta*(Ra/Rm)^2 already lives in the face area, so
        # A_blocked = A_open * (1 - T) / T.
        if self._mesh_accommodation:
            transparency = _anode_neutral_transparency(p)
            if transparency <= 0.0:
                raise ValueError(
                    "neutral_mesh_accommodation requires a mesh with open "
                    f"neutral area (transparency {transparency})"
                )
            open_area = np.asarray(
                self._geometry.neutral_face_area_cm2, dtype=float
            )[anode_faces]
            self._mesh_faces = anode_faces
            self._mesh_blocked_area_cm2 = (
                open_area * (1.0 - transparency) / transparency
            )
        else:
            self._mesh_faces = None
            self._mesh_blocked_area_cm2 = None

    def _validate_phase_config(self):
        mode = self._phase_transition_mode()
        if mode not in {"scheduled", "current"}:
            raise ValueError(
                "phase_transition_mode must be 'scheduled' or 'current' "
                f"(got {mode!r})"
            )
        action = self._prebreakdown_timeout_action()
        if action not in {"switch_open", "raise"}:
            raise ValueError(
                "prebreakdown_timeout_action must be 'switch_open' or "
                f"'raise' (got {action!r})"
            )

    def _phase_transition_mode(self):
        return self._input_dict.get("phase_transition_mode", "scheduled")

    def _prebreakdown_timeout_action(self):
        return self._input_dict.get("prebreakdown_timeout_action", "switch_open")

    def _main_discharge_start_time(self):
        if self._phase_transition_mode() == "current":
            return self._t_breakdown_trigger
        plasma_origin = self._plasma_phase_time_origin()
        tau_prebreakdown = max(
            float(self._input_dict.get("tau_prebreakdown", 0.0)),
            0.0,
        )
        tau_breakdown = max(float(self._input_dict.get("tau_breakdown", 0.0)), 0.0)
        return plasma_origin + tau_prebreakdown + tau_breakdown

    def _current_trigger_t_end(self):
        if (
            not self._flags.get("Plasma", True)
            or self._phase_transition_mode() != "current"
            or self._t_breakdown_trigger is None
        ):
            return None
        tau_discharge = max(float(self._input_dict.get("tau_discharge", 0.0)), 0.0)
        tau_afterglow = max(float(self._input_dict.get("tau_afterglow", 0.0)), 0.0)
        return float(self._t_breakdown_trigger) + tau_discharge + tau_afterglow

    def _effective_gas_puff_sccm(self, time=None):
        if time is None:
            time = self._time
        phase, phase_elapsed = self._phase_info(time)
        S_gp = float(self._input_dict.get("S_gp", 0.0))
        Twin_S_gp = float(self._input_dict.get("Twin_S_gp", 0.0))
        if not self._flags.get("Plasma", True):
            return S_gp, Twin_S_gp
        mode = self._input_dict.get("gas_puff_mode", "decay_after_breakdown")
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
                self._input_dict.get("gas_puff_rise_center_s", 5.0e-4)
            )
            rise_width = float(
                self._input_dict.get("gas_puff_rise_width_s", 5.0e-4)
            )
            close_lag = float(
                self._input_dict.get("gas_puff_close_lag_s", 5.0e-4)
            )
            t_on = self._plasma_phase_time_origin() + rise_center
            rise = 0.5 * (1.0 + math.erf((float(time) - t_on) / rise_width))
            tau_discharge = max(
                float(self._input_dict.get("tau_discharge", 0.0)), 0.0
            )
            if self._phase_transition_mode() == "current":
                main_start = self._t_breakdown_trigger
            else:
                main_start = (
                    self._plasma_phase_time_origin()
                    + max(float(self._input_dict.get("tau_prebreakdown", 0.0)), 0.0)
                    + max(float(self._input_dict.get("tau_breakdown", 0.0)), 0.0)
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
                + max(float(self._input_dict.get("tau_prebreakdown", 0.0)), 0.0)
                + max(float(self._input_dict.get("tau_breakdown", 0.0)), 0.0)
            )
            t_rel = float(time) - main_start
            rise = 0.5 * (
                1.0
                + math.erf(
                    (t_rel - float(self._input_dict.get("tau_gp_rise_center", -5e-3)))
                    / float(self._input_dict.get("tau_gp_rise_width", 1e-3))
                )
            )
            drop = 0.5 * (
                1.0
                + math.erf(
                    (t_rel - float(self._input_dict.get("tau_gp_drop_center", 1e-3)))
                    / float(self._input_dict.get("tau_gp_drop_width", 1e-3))
                )
            )
            target = float(self._input_dict.get("S_gp_decay_target", 0.0))
            twin_target = float(self._input_dict.get("Twin_S_gp_decay_target", 0.0))
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
            pulse_duration = float(self._input_dict.get("tau_gp_pulse_duration", 0.0))
            if phase_elapsed <= pulse_duration:
                return S_gp, Twin_S_gp
            decay_elapsed = phase_elapsed - pulse_duration
            tau_decay = float(self._input_dict.get("tau_gp_decay_duration", 1e-3))
            decay = float(np.exp(-decay_elapsed / tau_decay))
            target = float(self._input_dict.get("S_gp_decay_target", 0.0))
            twin_target = float(self._input_dict.get("Twin_S_gp_decay_target", 0.0))
            return (
                target + (S_gp - target) * decay,
                twin_target + (Twin_S_gp - twin_target) * decay,
            )

        tau_after_breakdown = self._input_dict.get("tau_gp_after_breakdown", None)
        if tau_after_breakdown is None:
            return S_gp, Twin_S_gp
        tau_after_breakdown = float(tau_after_breakdown)
        if phase_elapsed <= tau_after_breakdown:
            return S_gp, Twin_S_gp
        tau_discharge = max(float(self._input_dict.get("tau_discharge", 0.0)), 0.0)
        tau_decay = (tau_discharge - tau_after_breakdown) * float(
            self._input_dict.get("tau_gp_decay_factor", 1.0)
        )
        if tau_decay <= 0.0:
            return S_gp, Twin_S_gp
        decay = float(np.exp(-(phase_elapsed - tau_after_breakdown) / tau_decay))
        return decay * S_gp, decay * Twin_S_gp

    def _gas_puff_event_time(self):
        if not (
            self._flags.get("Plasma", True)
            and bool(self._input_dict.get("gas_puff_enabled", True))
        ):
            return None
        main_start = self._main_discharge_start_time()
        if main_start is None:
            return None
        tau_discharge = max(float(self._input_dict.get("tau_discharge", 0.0)), 0.0)
        mode = self._input_dict.get("gas_puff_mode", "decay_after_breakdown")
        if mode == "pulse_decay_to_level":
            event = main_start + float(
                self._input_dict.get("tau_gp_pulse_duration", 0.0)
            )
        else:
            tau_after_breakdown = self._input_dict.get("tau_gp_after_breakdown", None)
            if tau_after_breakdown is None:
                return None
            event = main_start + float(tau_after_breakdown)
        if event >= main_start + tau_discharge:
            return None
        return event

    def _cathode_total_current_A(self):
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

    def _ignition_abort_t_end(self):
        """Return the wind-down end time [s] after a switch-open abort."""
        if self._t_ignition_abort is None:
            return None
        tau_afterglow = max(float(self._input_dict.get("tau_afterglow", 0.0)), 0.0)
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
            or not self._flags.get("Plasma", True)
            or self._neutral_prebreakdown_active()
        ):
            return
        cathode_phase = self._cathode_phase_options(time=self._time)
        if cathode_phase["solve_enabled"]:
            self.solve_cathode_boundary(time=self._time, update_cache=True)

        I_now = self._cathode_total_current_A()
        tau_prebreakdown = max(
            float(self._input_dict.get("tau_prebreakdown", 0.0)),
            0.0,
        )
        I_prebreakdown = float(self._input_dict.get("I_prebreakdown", 0.0))
        I_breakdown = float(self._input_dict.get("I_breakdown", 0.0))
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
        return {
            "b_Qie": float(self._input_dict.get("b_Qie", 1.0)),
            "ln_lambda_min": float(self._input_dict.get("ln_lambda_min", 1.0)),
        }

    def _surface_loss_kwargs(self):
        # The resolved boundary terms read only ``alpha_isat`` and
        # ``b_surface_loss``. The former per-face source/end enables and area
        # scales were A13 no-ops (never consumed) and are DEPRECATED 0D artifacts
        # (R3.3): the resolved geometry measures the Bohm I_sat to each electrode
        # face directly. ``_validate_r1_configuration_presence`` warns on their
        # non-default use.
        return {
            "alpha_isat": float(self._input_dict.get("alpha_isat", np.exp(-0.5))),
            "end_mode": self._input_dict.get("end_mode", "collector"),
            "b_surface_loss": float(self._input_dict.get("b_surface_loss", 1.0)),
        }

    def _ion_neutral_drag_kwargs(self):
        drag_enabled = bool(self._flags.get("ion_neutral_drag", True))
        return {
            "gas_type": self._gas_type,
            "sigma_in_cm2": float(self._input_dict.get("sigma_in_cm2", 5.0e-15)),
            "sigma_in_model": str(
                self._input_dict.get("sigma_in_model", "constant")
            ),
            "b_ion_neutral_drag": (
                float(self._input_dict.get("b_ion_neutral_drag", 1.0))
                if drag_enabled
                else 0.0
            ),
            "cx_only": bool(self._flags.get("ion_neutral_drag_cx_only", False)),
        }

    def _slip_closure_kwargs(self):
        """Extra kwargs for the drag/frictional-heating slip closure."""
        return {
            "drag_model": str(
                self._input_dict.get("ion_neutral_drag_model", "constant")
            ),
            "b_slip_entrainment": float(
                self._input_dict.get("b_slip_entrainment", 1.0)
            ),
            "Rm_cm": self._geometry.Rm_cm,
            "Tn_fit": float(self._input_dict.get("Tn_fit", 0.1)),
        }

    def _electron_cooling_kwargs(self):
        return {
            "gas_type": self._gas_type,
            "I_ion": self._I_ion,
            "b_ioniz": float(self._input_dict.get("b_ioniz", 1.0)),
            "b_rec_rad": float(self._input_dict.get("b_rec_rad", 1.0)),
            "b_rec_3b": float(self._input_dict.get("b_rec_3b", 1.0)),
            # b_ionization_energy_cost removed as a config knob (R5 stance flip):
            # must be 1 for conservative energy booking, and the on/off is the
            # ionization_energy_cost flag. Hardwired 1.0.
            "b_ionization_energy_cost": 1.0,
            "b_Qei": float(self._input_dict.get("b_Qei", 1.0)),
            "b_Qen": float(self._input_dict.get("b_Qen", 1.0)),
            "b_Qei_Te_exp": float(self._input_dict.get("b_Qei_Te_exp", 0.0)),
            "b_Qen_Te_exp": float(self._input_dict.get("b_Qen_Te_exp", 0.0)),
            "b_Q_Te_ref_eV": float(self._input_dict.get("b_Q_Te_ref_eV", 5.0)),
            "atomic_rate_model": str(
                self._input_dict.get("atomic_rate_model", "adas")
            ),
            "ionization_energy_cost": bool(
                self._flags.get("ionization_energy_cost", True)
            ),
            "icool": bool(self._flags.get("icool", True)),
            "ncool": bool(self._flags.get("ncool", True)),
            "icool_recomb": bool(self._flags.get("icool_recomb", False)),
            # A18/R5.3: the low-Te extension defines ONE consistent atomic
            # package -- the electron-cooling prb1 honors it just like the
            # particle-rate acd. Default off => golden bit-exact.
            "adas_low_te_extension": bool(
                self._input_dict.get("adas_low_te_extension", False)
            ),
        }

    def _ion_charge_exchange_kwargs(self):
        return {
            "gas_type": self._gas_type,
            "Tn_fit": float(self._input_dict.get("Tn_fit", 0.1)),
            "b_Qcx": float(self._input_dict.get("b_Qcx", 1.0)),
            "cx": bool(self._flags.get("cx", True)),
        }

    def _heat_conduction_kwargs(self):
        return {
            "b_epara": float(self._input_dict.get("b_epara", 1.0)),
            "b_ipara": float(self._input_dict.get("b_ipara", 1.0)),
            "heat_conduction": bool(self._flags.get("heat_conduction", True)),
            "ln_lambda_min": float(self._input_dict.get("ln_lambda_min", 1.0)),
            "electron_heat_flux_limit": self._electron_heat_flux_limit,
            "heat_flux_limiter_f": self._heat_flux_limiter_f,
            "heat_flux_limiter_exponent": self._heat_flux_limiter_exponent,
        }

    def _reaction_kwargs(self):
        return {
            "gas_type": self._gas_type,
            "I_ion": self._I_ion,
            "b_ioniz": float(self._input_dict.get("b_ioniz", 1.0)),
            "b_rec_rad": float(self._input_dict.get("b_rec_rad", 1.0)),
            "b_rec_3b": float(self._input_dict.get("b_rec_3b", 1.0)),
            "atomic_rate_model": str(
                self._input_dict.get("atomic_rate_model", "adas")
            ),
            "adas_low_te_extension": bool(
                self._input_dict.get("adas_low_te_extension", False)
            ),
            "Te_birth_ionization": self._input_dict.get(
                "Te_birth_ionization", "local"
            ),
            "Ti_birth_ionization": self._input_dict.get(
                "Ti_birth_ionization", "floor"
            ),
            "ionization_birth_energy_model": str(
                self._input_dict.get("ionization_birth_energy_model", "legacy")
            ),
            "wind_column_factor": self._wind_column_factor,
        }

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
            end=0,
        )
        if twin:
            puff = puff + gas_puff_rate_profile(
                self._geometry, nk["Twin_S_gp"], nk["gas_puff_valves"],
                profile=nk["gas_puff_profile"], z_cm=nk["gas_puff_z_cm"],
                sigma_cm=nk["gas_puff_sigma_cm"],
                throw_cm=nk["gas_puff_throw_cm"],
                end=-1,
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
        result.atomic_rate_domain = _atomic_rate_domain(result)
        return add_sim3_compat_aliases(result)

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
        if not self._flags.get("Plasma", True):
            # Same window the run loop actually delivers (item 37).
            puff_on = self._equilibration_puff_on_duration()
            puff_reason = self._equilibration_puff_on_reason()
            tau_cycle = max(float(self._input_dict.get("tau_cycle", 0.0)), 0.0)
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
            float(self._input_dict.get("tau_prebreakdown", 0.0)),
            0.0,
        )
        tau_breakdown = max(float(self._input_dict.get("tau_breakdown", 0.0)), 0.0)
        tau_discharge = max(float(self._input_dict.get("tau_discharge", 0.0)), 0.0)
        tau_afterglow = max(float(self._input_dict.get("tau_afterglow", 0.0)), 0.0)
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
        tau_discharge = max(float(self._input_dict.get("tau_discharge", 0.0)), 0.0)
        if not self._flags.get("Plasma", True):
            puff_on = self._equilibration_puff_on_duration()
            _, cycle_time = self._equilibration_cycle_position(time)
            if cycle_time < puff_on:
                return "equilibrium_puff", cycle_time
            return "equilibrium_off", cycle_time - puff_on

        tau_prebreakdown = max(
            float(self._input_dict.get("tau_prebreakdown", 0.0)),
            0.0,
        )
        tau_breakdown = max(float(self._input_dict.get("tau_breakdown", 0.0)), 0.0)
        tau_afterglow = max(float(self._input_dict.get("tau_afterglow", 0.0)), 0.0)
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
                    self._input_dict.get("gas_puff_enabled", True)
                ),
                "floating": False,
            }
        if phase == "equilibrium_puff":
            return {
                "cathode_enabled": False,
                "gas_puff_enabled": bool(
                    self._input_dict.get("gas_puff_enabled", True)
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
                bool(self._flags.get("cathode_coupling", False))
                and phase in discharge_phases
            ),
            "gas_puff_enabled": (
                bool(self._input_dict.get("gas_puff_enabled", True))
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
        surface-power line. The plasma-terminating boundary term (the active
        `characteristic_boundary` or legacy `boundary_absorption`) removes
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
            "enabled": float(bool(self._flags.get("cathode_coupling", False))),
            # Instantaneous emitter surface temperature [K]: the configured
            # T_s, or the evolving value under cathode_warming_model.
            "T_s_surface": float(
                self._cathode_Ts_K
                if self._cathode_Ts_K is not None
                else float(self._input_dict.get("T_s", 0.0))
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
            # Cumulative surface energy ledger [J] (power_balance warming
            # only; zeros otherwise). See _cathode_energy_ledger_J.
            **{
                f"warming_E_{k}_J": float(v)
                for k, v in self._cathode_energy_ledger_J.items()
            },
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
                else float(self._input_dict.get("phi_wf", 0.0))
            ),
            "phase_enabled": float(cathode_phase["cathode_enabled"]),
            "rhs_enabled": float(cathode_phase["cathode_enabled"]),
            "solve_enabled": float(cathode_phase["solve_enabled"]),
            "floating": float(cathode_phase["floating"]),
            "twin_cathode": float(bool(self._flags.get("TwinCathode", False))),
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
        }
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
        for prefix in ("source", "end"):
            diag[f"{prefix}_regime"] = "none"
            for key in _CATHODE_RESULT_KEYS:
                diag[f"{prefix}_{key}"] = np.nan
            diag[f"{prefix}_long_mfp"] = np.nan

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
        self._copy_cathode_result_diagnostics(
            diag=diag,
            prefix="source",
            result=beam_result.result,
        )
        if beam_result.result_twin is not None:
            diag["has_twin_solution"] = 1.0
            self._copy_cathode_result_diagnostics(
                diag=diag,
                prefix="end",
                result=beam_result.result_twin,
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

    def _copy_cathode_result_diagnostics(self, diag, prefix, result):
        if result is None:
            return
        diag[f"{prefix}_regime"] = str(result.regime)
        for key in _CATHODE_RESULT_KEYS:
            diag[f"{prefix}_{key}"] = float(getattr(result, key))
        diag[f"{prefix}_long_mfp"] = float(bool(result.long_mfp))

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

    def _kinetic_channel_rates(self, state, derived, time):
        """Return the current source-channel rates and shapes (K4a-t).

        Cheap (pure numpy on the live fields); called every target update
        so the kinetic targets track source transients continuously --
        the recycle channels collapse WITH the plasma, which is what
        removes the afterglow flood.

        The wall-return channels are read from the boundary term that is
        ACTUALLY removing the plasma in this stance -- the R3.1 characteristic
        ghost-cell flux when ``characteristic_boundary`` is on, the legacy
        volumetric absorber when it is off -- and from its own live cells,
        resolved by role. Both halves matter: the two operators disagree face
        by face (they are different discretizations of the same surface), and
        the recycled quantity IS the ``nn`` row the fluid path would have
        applied itself had the arm not superseded it, so what the arm re-injects
        equals what the boundary removed, per face, to roundoff.

        ``cath``/``coll`` are the per-surface totals (the kinetic steady arm
        needs a magnitude per channel); ``cath_cells``/``coll_cells`` place
        the same particles on the grid, which is what the transient arm
        deposits into.
        """
        geometry = self._geometry
        V_col, V_ann = self._zone_volumes
        boundary = (
            self.characteristic_boundary_rhs(state=state)
            if self._characteristic_boundary
            else self.boundary_absorption_rhs(state=state)
        )
        recycle = np.clip(boundary.nn, 0.0, None) * V_col
        cath_cells = np.zeros(geometry.cells)
        coll_cells = np.zeros(geometry.cells)
        for role, target in (
            ("cathode", cath_cells),
            ("collector", coll_cells),
        ):
            for cell in self._recycle_cells.get(role, ()):
                target[cell] = recycle[cell]
        reaction_terms = self.reaction_rhs_terms(state=state)
        rec_cells = np.clip(
            reaction_terms["recombination_rad_loss"].nn
            + reaction_terms["recombination_3b_loss"].nn,
            0.0,
            None,
        ) * V_col
        an = self.anode_collection_rhs(state=state)
        an_gain = np.clip(an.nn, 0.0, None) * V_col
        if an.nn_a is not None:
            an_gain = an_gain + np.clip(an.nn_a, 0.0, None) * V_ann
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
        T_s = float(self._input_dict.get("T_s", 1910.0))
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
            "S_pump_L": float(self._input_dict.get("S_pump_L", 0.0)),
            "S_pump_R": float(self._input_dict.get("S_pump_R", 0.0)),
            "eta": float(self._input_dict.get("eta", 0.358)),
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
        vbar = np.sqrt(
            8.0 * kb_cgs * 300.0 / (np.pi * self._mu_neutral * m_p_cgs)
        )
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

            applied_cum + debt == booked_cum

        holds per cell to roundoff (:meth:`_dvm_book_step_transfer` closes
        it). A cell that never regains margin carries the debt to the end of
        the run, which is the honest statement that the plasma could not
        absorb what the kinetic side booked -- reported, never silently
        dropped.

        Heating (a non-negative ``Ei`` rate) is never limited: it cannot
        threaten a floor.
        """
        dvm = self._dvm
        state = self.state
        apply_mask = self._dvm_transfer_apply_mask()
        booked_M = np.asarray(dvm.M_transfer, dtype=float)
        booked_Ei = np.asarray(dvm.Ei_transfer, dtype=float)
        desired_M = booked_M + dvm.M_debt / dt
        desired_Ei = booked_Ei + dvm.Ei_debt / dt
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
            desired_M=desired_M,
            desired_Ei=desired_Ei,
            applied_M=applied_M,
            applied_Ei=applied_Ei,
            limited=limited,
        )
        return self._dvm_step_transfer

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

        ``residual`` is ``applied_cum + debt - booked_cum`` per cell, the
        statement that the relax deferred energy and momentum rather than
        destroying them; ``*_rel`` normalizes it by the accumulated
        throughput plus the outstanding debt.
        """
        if self._dvm is None:
            return None
        dvm = self._dvm
        out = {}
        for name in ("M", "Ei"):
            applied = getattr(dvm, f"{name}_applied_cum")
            booked = getattr(dvm, f"{name}_booked_cum")
            debt = getattr(dvm, f"{name}_debt")
            residual = applied + debt - booked
            scale = np.max(np.abs(booked)) + np.max(np.abs(debt))
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
        return out

    def _dvm_ledger_sample(self, time):
        """Return the per-save-frame census record of the transfer ledger.

        Three cumulative counters and the frame time -- enough to place WHEN
        the limiter engaged (difference consecutive frames) without carrying
        the per-cell arrays at every save. ``limited_cells`` counts cells
        limited at least once so far, not cells limited at this frame.
        """
        dvm = self._dvm
        return {
            "time": float(time),
            "relax_steps": float(dvm.relax_steps),
            "relax_limited_steps": float(dvm.relax_limited_steps),
            "limited_cells": float(np.count_nonzero(dvm.relax_cell_steps)),
        }

    def _dvm_ledger_census(self, saved):
        """Return the end-of-run transfer-ledger census for the result.

        The quotable form of the standing DVM report condition: how many
        steps the floor-aware relax limited, what each channel still owes the
        plasma, and the cumulative booked-vs-applied totals that carry the
        identity ``applied_cum + debt == booked_cum``. Per-cell arrays are
        kept (they are one row of ``cells`` each) so the identity is
        checkable from the artifact and the debt can be localized; the
        scalars are the numbers a report quotes.

        Units are the ledger's own, integrated over the step: ``Ei_*`` in
        erg/cm^3 and ``M_*`` in g/(cm^2 s) per cell, and the ``*_total``
        scalars are those densities integrated over the plasma (= column)
        volume, i.e. erg and g cm/s.
        """
        dvm = self._dvm
        volume = np.asarray(self._geometry.plasma_volume_cm3, dtype=float)
        census = {
            "engaged": int(bool(self._dvm_engaged)),
            "relax_steps": int(dvm.relax_steps),
            "relax_limited_steps": int(dvm.relax_limited_steps),
            "limited_cells": int(np.count_nonzero(dvm.relax_cell_steps)),
            "relax_cell_steps": dvm.relax_cell_steps.copy(),
        }
        for name in ("Ei", "M"):
            debt = np.asarray(getattr(dvm, f"{name}_debt"), dtype=float)
            booked = np.asarray(getattr(dvm, f"{name}_booked_cum"), dtype=float)
            applied = np.asarray(getattr(dvm, f"{name}_applied_cum"), dtype=float)
            residual = applied + debt - booked
            scale = float(np.max(np.abs(booked)) + np.max(np.abs(debt)))
            census.update(
                {
                    f"{name}_debt": debt.copy(),
                    f"{name}_booked_cum": booked.copy(),
                    f"{name}_applied_cum": applied.copy(),
                    f"{name}_debt_total": float(np.sum(debt * volume)),
                    f"{name}_debt_max_abs": float(np.max(np.abs(debt))),
                    f"{name}_booked_total": float(np.sum(booked * volume)),
                    f"{name}_applied_total": float(np.sum(applied * volume)),
                    f"{name}_residual_rel": float(
                        np.max(np.abs(residual)) / max(scale, 1e-300)
                    ),
                }
            )
        for field in ("time", "relax_steps", "relax_limited_steps", "limited_cells"):
            census[f"sample_{field}"] = np.asarray(
                [snapshot["dvm_ledger"][field] for snapshot in saved],
                dtype=float,
            )
        return census

    def _dvm_presheath_Tn_eV(self):
        """Return the per-cell Tn [eV] the presheath should consume, or None."""
        if self._dvm is None or not self._dvm_engaged:
            return None
        if not self._dvm_tn_feedback:
            return None
        return self._dvm.Tn_col_eV

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

    def _dvm_advance(self, dt_neutral):
        """Run one transient DVM update and republish the neutral moments."""
        state = self.state
        derived = self.derived
        geometry = self._geometry
        nu_ion = self._dvm_ionization_frequency(state, derived)
        rates = self._kinetic_channel_rates(state, derived, self._time)
        sources = {
            "puff": np.asarray(rates["puff"], dtype=float),
            "recombination": np.asarray(rates["rec"], dtype=float),
            "anode": np.asarray(rates["anode"], dtype=float),
            # Per-cell, not a scalar: the wall return belongs in the cell the
            # boundary term drained, which is not an end cell once an
            # obstruction sits behind the cathode.
            "cathode_face": np.asarray(rates["cath_cells"], dtype=float),
            "collector_face": np.asarray(rates["coll_cells"], dtype=float),
        }
        self._dvm.update(
            float(dt_neutral),
            n_i=np.asarray(state.n, dtype=float),
            Ti_eV=np.asarray(derived.Ti, dtype=float),
            u_i=np.asarray(derived.u, dtype=float),
            nu_ion=nu_ion,
            sources=sources,
            T_s_K=(
                float(self._cathode_Ts_K)
                if self._cathode_Ts_K is not None
                else float(self._input_dict.get("T_s", 1910.0))
            ),
        )
        self._dvm_last_s = self._time
        self._dvm_next_s = self._time + self._dvm_cadence_s
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
        beam -- divided by the column density those bookings consumed, so
        the kinetic side removes the same neutrals the plasma turns into
        ions rather than re-deriving a rate that could drift from it.
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
        zeros = np.zeros(self._geometry.cells, dtype=float)
        return ConservativeState1D(
            n=zeros,
            nn=zeros.copy(),
            M=zeros.copy(),
            Ee=zeros.copy(),
            Ei=zeros.copy(),
        )

    def _set_state_vector(self, y):
        self._y = self.floor_state_vector(y)
        self._state = self._unpack(self._y)
        self._derived = derive_state(self._state, self._floors, self._ion_mass_g)
        if self._flags.get("debug_checks", False):
            assert_finite_state(self._state, self._derived)

    def _initial_state(self):
        cells = self._geometry.cells
        n0 = np.full(cells, float(self._input_dict["ne0"]))
        nn0 = np.full(cells, float(resolve_nn0(self._input_dict, self._flags)))
        u0 = np.full(cells, float(self._input_dict.get("u0", 0.0)))
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
            # value inertly.
            nn_a=nn0.copy() if self._neutral_two_zone else None,
            un_a=np.zeros(cells) if self._neutral_two_momentum else None,
        )

    @staticmethod
    def _gas_constants(gas_type):
        if gas_type == "He":
            return m_He_cgs, 4, 4, I_ion
        if gas_type == "H":
            return m_p_cgs, 1, 2, I_Ry
        raise ValueError(f"unsupported gas_type {gas_type!r}; expected 'He' or 'H'")


def _finite_or_nan(value):
    if value is None:
        return np.nan
    return float(value)
