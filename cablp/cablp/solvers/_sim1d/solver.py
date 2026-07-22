"""Solver implementation for the conservative axial 1D LAPD model."""

import math
from dataclasses import dataclass, replace
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from .core.config import (
    default_config,
    input_dict_template_1d,
    input_flags_template_1d,
    load_config,
    resolve_nn0,
)
from .core.geometry import (
    _anode_neutral_transparency,
    build_geometry,
    is_plenum_cell,
    puff_cell_indices,
    pump_cell_indices,
)
from .core.integrator import (
    floor_state_vector,
    ssprk2_step,
)
from .core.state import (
    STATE_NAMES_1D,
    ConservativeState1D,
    apply_state_floors,
    assert_finite_state,
    conservative_from_primitives,
    derive_state,
    pack_state,
    unpack_state,
)
from .core.timestep import suggest_timestep
from .physics.conduction import heat_conduction_rhs, implicit_heat_conduction_step
from .physics.cathode import (
    beam_ionization_rhs,
    beam_ionization_rhs_terms,
    cathode_boundary_state,
    cathode_power_balance_terms_W,
    cathode_sample_indices,
    cathode_source_terms,
    solve_cathode_boundary,
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
    reaction_rhs,
    reaction_rhs_terms,
    recombination_energy_return_rhs,
)
from .physics.sources import (
    add_state_rhs,
    anode_collection_rhs,
    boundary_absorption_rhs,
    ion_neutral_drag_rhs,
    ion_neutral_frictional_heating_rhs,
    ion_neutral_thermalization_rhs,
    neutral_momentum_wall_rhs,
    neutral_wind_two_zone_factors,
    neutral_wind_velocity,
    pressure_work_rhs,
)
from .results.compat import add_sim3_compat_aliases
from cablp.vars._cons import I_Ry, I_ion, ev_to_erg, m_He_cgs, m_p_cgs


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
        input_dict=input_dict_template_1d,
        input_flags=input_flags_template_1d,
        progress_callback=None,
        progress_tracker=None,
        progress_interval_s=1.0e-4,
    ):
        self._input_dict = dict(input_dict)
        self._flags = dict(input_flags)
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
        exchange_model = str(
            self._input_dict.get("neutral_exchange_model", "knudsen")
        )
        if exchange_model not in ("constant", "knudsen"):
            raise ValueError(
                "neutral_exchange_model='molecular_flow' was removed at "
                "DEPRECATION_PLAN D2; use 'constant' or 'knudsen', or "
                "reproduce it at tag legacy-final-2026-07-22 "
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
        self._neutral_momentum_radial = str(
            self._input_dict.get("neutral_momentum_radial", "uniform")
        )
        if self._neutral_momentum_radial not in ("uniform", "two_zone"):
            raise ValueError(
                "neutral_momentum_radial must be 'uniform' or 'two_zone' "
                f"(got {self._neutral_momentum_radial!r})"
            )
        if self._neutral_momentum_radial == "two_zone" and not self._neutral_momentum:
            raise ValueError(
                "neutral_momentum_radial='two_zone' requires the "
                "neutral_momentum flag: it closes the radial profile of the "
                "evolved wind"
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
                str(self._input_dict.get("atomic_rate_model", "janev"))
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
        self._floors = {
            "n": float(self._input_dict["ne_floor"]),
            "nn": float(self._input_dict["nn_floor"]),
            "Te": float(self._input_dict["Te_floor"]),
            "Ti": float(self._input_dict["Ti_floor"]),
        }
        self._state = self._initial_state()
        self._state = apply_state_floors(self._state, self._floors, self._ion_mass_g)
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
        # per accepted step (plan §2c).
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
                "cathode_warming_model='ion_bombardment' was removed at "
                "DEPRECATION_PLAN D2; use 'none' or 'power_balance', or "
                "reproduce it at tag legacy-final-2026-07-22 "
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
        # CATHODE_IDRIVEN_PLAN.md M5a): theta in [0, 1] is the contaminant
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
        # (THESIS_NOTES §2: ~sub-kW net imbalance <-> ±8 K per 20-30 min).
        self._cathode_energy_ledger_J = {
            "heater": 0.0, "ion": 0.0, "rad": 0.0, "emis": 0.0, "cond": 0.0,
        }
        self._cathode_solve = None
        self._last_result = None
        self._last_neutral_equilibration_result = None
        self._last_neutral_equilibration_summary = None
        if self._flags.get("debug_checks", False):
            assert_finite_state(self._state, self._derived)

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
        if not self._flags.get("Plasma", True) or self._neutral_prebreakdown_active(
            time=time,
        ):
            return {
                **zone_terms,
                "plasma_advective_flux": self._zero_rhs_state(),
                "plasma_front_flux": self._zero_rhs_state(),
                "boundary_absorption": self._zero_rhs_state(),
                "pressure_work": self._zero_rhs_state(),
                "ei_exchange": self._zero_rhs_state(),
                "ionization_energy_cost": self._zero_rhs_state(),
                "electron_ion_cooling": self._zero_rhs_state(),
                "electron_neutral_cooling": self._zero_rhs_state(),
                "ion_charge_exchange": self._zero_rhs_state(),
                "ion_neutral_drag": self._zero_rhs_state(),
                "ion_neutral_frictional_heating": self._zero_rhs_state(),
                "ion_neutral_thermalization": self._zero_rhs_state(),
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
                "ionization_birth": self._zero_rhs_state(),
                "beam_ionization_birth": self._zero_rhs_state(),
                "beam_power_deposition": self._zero_rhs_state(),
                "beam_ionization_cost": self._zero_rhs_state(),
                "beam_excitation_radiation": self._zero_rhs_state(),
                "recombination_rad_loss": self._zero_rhs_state(),
                "recombination_3b_loss": self._zero_rhs_state(),
                "heat_conduction": self._zero_rhs_state(),
            }
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
            "plasma_advective_flux": plasma_terms["plasma_advective_flux"],
            "plasma_front_flux": plasma_terms["plasma_front_flux"],
            "boundary_absorption": self.boundary_absorption_rhs(
                state=state, cathode_solve=cathode_solve, time=time
            ),
            "pressure_work": self.pressure_work_rhs(state=state),
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
            "neutral_momentum_wall": self.neutral_momentum_wall_rhs(state=state),
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
        return terms

    def floor_state_vector(self, y):
        """Apply configured density and temperature floors to a packed vector."""
        return floor_state_vector(
            y=y,
            cells=self._geometry.cells,
            floors=self._floors,
            ion_mass_g=self._ion_mass_g,
            neutral_momentum=self._neutral_momentum,
            neutral_two_zone=self._neutral_two_zone,
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

        starting_cache = self._step_cache_snapshot()
        try:
            if not self._flags.get("Plasma", True) or self._neutral_prebreakdown_active():
                y_next = pack_state(self._implicit_neutral_step(dt=dt))
            elif operator_split:
                y_next = self.operator_split_step(dt=dt)
            else:
                y_next = ssprk2_step(
                    y0=self._y,
                    dt=dt,
                    rhs_func=lambda yy, tt: self.rhs(yy, time=tt),
                    floor_func=self.floor_state_vector,
                    time=self._time,
                )
            candidate_cache = self._step_cache_snapshot()
        finally:
            self._restore_step_cache(starting_cache)
        return StepAttempt1D(
            y=np.asarray(y_next, dtype=float),
            dt=dt,
            operator_split=bool(operator_split),
            solver_cache=candidate_cache,
        )

    def _implicit_neutral_step(self, dt, state=None, time=None):
        """Return a backward-Euler neutral-only state update."""
        if state is None:
            state = self.state
        if time is None:
            time = self._time
        if self._neutral_two_zone and state.nn_a is not None:
            return self._implicit_neutral_step_two_zone(
                dt=dt, state=state, time=time
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
        # there is no drag to drive a wind (NEUTRAL_MOMENTUM_PLAN.md).
        return ConservativeState1D(
            n=state.n.copy(),
            nn=np.maximum(nn_next, self._floors["nn"]),
            M=state.M.copy(),
            Ee=state.Ee.copy(),
            Ei=state.Ei.copy(),
            M_n=None if state.M_n is None else state.M_n.copy(),
        )

    def _implicit_neutral_step_two_zone(self, dt, state, time):
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
            nn=np.maximum(solution[:cells], self._floors["nn"]),
            M=state.M.copy(),
            Ee=state.Ee.copy(),
            Ei=state.Ei.copy(),
            M_n=None if state.M_n is None else state.M_n.copy(),
            nn_a=np.maximum(solution[cells:], self._floors["nn"]),
        )

    def _step_rejection_info(self, attempt, y0=None):
        if y0 is None:
            y0 = self._y
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
        for name, values in (("n", state1.n), ("nn", state1.nn)):
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
        self._restore_step_cache(attempt.solver_cache)
        self._set_state_vector(attempt.y)
        self._time += float(attempt.dt)
        # Electrode sample smoothing: fold the newly accepted state into the
        # supply-average EMA before any accepted-state consumer reads it.
        self._update_sample_smoothing(attempt.dt)
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
                # the surface model is on (one shared constant, §3b).
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
                R_comp_ohm=float(self._input_dict.get("R_comp", 0.0)),
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

    def advance_one_step(self, dt=None, operator_split=None):
        """Advance the conservative state by one explicit or split step."""
        return self._accept_step_attempt(
            self._attempt_step(dt=dt, operator_split=operator_split)
        )

    def operator_split_step(self, y=None, dt=None, splitting=None):
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

        def heat(y_in, sub_dt):
            state = self.implicit_heat_conduction_step(dt=sub_dt, y=y_in)
            return self.floor_state_vector(pack_state(state))

        def explicit(y_in, sub_dt):
            return ssprk2_step(
                y0=y_in,
                dt=sub_dt,
                rhs_func=lambda yy, tt: self.rhs(
                    yy,
                    include_heat_conduction=False,
                    time=tt,
                ),
                floor_func=self.floor_state_vector,
                time=self._time,
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
        while self._time < t_end - time_tol:
            if not unlimited_steps and steps >= max_steps:
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
            (
                attempt,
                retry_count,
                rejection_reason,
                step_rejection_events,
            ) = self._attempt_step_with_retries(
                dt=step_dt,
                operator_split=operator_split,
                diag=diag,
            )
            timestep_rejection_events.extend(step_rejection_events)
            self._accept_step_attempt(attempt)
            self._update_current_phase_triggers()
            if dynamic_current_t_end:
                current_t_end = self._current_trigger_t_end()
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
        self._last_result = result
        return result

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
        seeded = ConservativeState1D(
            n=state.n.copy(),
            nn=final_nn.copy(),
            M=state.M.copy(),
            Ee=state.Ee.copy(),
            Ei=state.Ei.copy(),
            M_n=None if state.M_n is None else state.M_n.copy(),
            nn_a=final_nn_a,
        )
        self._set_state_vector(pack_state(seeded))
        self._time = 0.0

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
            neutral_result = self.run_neutral_equilibration(
                progress_callback=progress_callback,
                progress_tracker=progress_tracker,
                progress_interval_s=progress_interval_s,
            )
            if not self._flags.get("launch_plasma_after_equilibration", False):
                self._last_result = neutral_result
                return
            self._apply_neutral_equilibration_result(neutral_result)

        self._last_result = self.run(
            t_end=t_end,
            dt=dt,
            operator_split=operator_split,
            max_steps=max_steps,
            progress_callback=progress_callback,
            progress_tracker=progress_tracker,
            progress_interval_s=progress_interval_s,
        )
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
            ion_charge_exchange_kwargs=(
                self._ion_charge_exchange_kwargs() if plasma_enabled else None
            ),
            heat_conduction_kwargs=(
                self._heat_conduction_kwargs()
                if plasma_enabled and include_heat_conduction
                else None
            ),
            ion_neutral_drag_kwargs=(
                self._ion_neutral_drag_kwargs() if plasma_enabled else None
            ),
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
            if neutral_dt == dt_min and raw_dt < dt_min:
                active_constraint = "dt_min"
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
            tau_discharge = max(
                float(self._input_dict.get("tau_discharge", 0.0)),
                0.0,
            )
            tau_cycle = max(float(self._input_dict.get("tau_cycle", 0.0)), 0.0)
            if tau_cycle <= 0.0:
                if in_run_window(tau_discharge):
                    return tau_discharge
                return None

            cycle_index = np.floor(time / tau_cycle)
            cycle_start = cycle_index * tau_cycle
            cycle_end = cycle_start + tau_cycle
            boundaries = []
            if 0.0 < tau_discharge < tau_cycle:
                boundaries.append(cycle_start + tau_discharge)
            boundaries.append(cycle_end)
            if 0.0 < tau_discharge < tau_cycle:
                boundaries.append(cycle_end + tau_discharge)
            boundaries.append(cycle_end + tau_cycle)
            for boundary in boundaries:
                if in_run_window(boundary):
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
            electron_scale=float(self._input_dict.get("b_pressure_work_elec", 1.0)),
            ion_scale=float(self._input_dict.get("b_pressure_work_ions", 1.0)),
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
            self._cathode_solve = result
            self._cathode_x0 = result.x0_next
            self._cathode_x0_twin = result.x0_twin_next
            if result.beam_result is not None:
                self._cathode_beam_cross = (
                    result.beam_result.beam_atten_cross.copy()
                )
        return result

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
                self._input_dict.get("atomic_rate_model", "janev")
            ),
            enabled=self._recombination_energy_return,
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
        )

    def _validate_neutral_jet_config(self):
        """Validate and cache the directed-recycle-jet configuration (§8).

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

    def _phase_transition_mode(self):
        return self._input_dict.get("phase_transition_mode", "scheduled")

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
        cathode_coupled = bool(self._flags.get("cathode_coupling", False))
        source_surface_loss_enabled = bool(
            self._flags.get("source_surface_loss", True)
        )
        if cathode_coupled:
            # The cathode solver returns the combined cathode/anode collection
            # current used by _sim3.py. A future 1D source model will need the
            # cathode solver to expose separate cathode and anode currents.
            source_surface_loss_enabled = False
        return {
            "alpha_isat": float(self._input_dict.get("alpha_isat", np.exp(-0.5))),
            "source_surface_area_scale": float(
                self._input_dict.get("source_surface_area_scale", 2.0)
            ),
            "end_surface_area_scale": float(
                self._input_dict.get("end_surface_area_scale", 1.0)
            ),
            "source_surface_loss_enabled": source_surface_loss_enabled,
            "end_surface_loss_enabled": bool(
                self._flags.get("end_surface_loss", True)
            ),
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
            "b_ionization_energy_cost": float(
                self._input_dict.get("b_ionization_energy_cost", 1.0)
            ),
            "b_Qei": float(self._input_dict.get("b_Qei", 1.0)),
            "b_Qen": float(self._input_dict.get("b_Qen", 1.0)),
            "b_Qei_Te_exp": float(self._input_dict.get("b_Qei_Te_exp", 0.0)),
            "b_Qen_Te_exp": float(self._input_dict.get("b_Qen_Te_exp", 0.0)),
            "b_Q_Te_ref_eV": float(self._input_dict.get("b_Q_Te_ref_eV", 5.0)),
            "atomic_rate_model": str(
                self._input_dict.get("atomic_rate_model", "janev")
            ),
            "ionization_energy_cost": bool(
                self._flags.get("ionization_energy_cost", True)
            ),
            "icool": bool(self._flags.get("icool", True)),
            "ncool": bool(self._flags.get("ncool", True)),
            "icool_recomb": bool(self._flags.get("icool_recomb", False)),
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
        }

    def _reaction_kwargs(self):
        return {
            "gas_type": self._gas_type,
            "I_ion": self._I_ion,
            "b_ioniz": float(self._input_dict.get("b_ioniz", 1.0)),
            "b_rec_rad": float(self._input_dict.get("b_rec_rad", 1.0)),
            "b_rec_3b": float(self._input_dict.get("b_rec_3b", 1.0)),
            "atomic_rate_model": str(
                self._input_dict.get("atomic_rate_model", "janev")
            ),
            "Te_birth_ionization": self._input_dict.get(
                "Te_birth_ionization", "local"
            ),
            "Ti_birth_ionization": self._input_dict.get(
                "Ti_birth_ionization", "floor"
            ),
            "wind_column_factor": self._wind_column_factor,
        }

    def _trajectory_snapshot(self, time):
        state = self.state
        derived = self.derived
        assert_finite_state(state, derived)
        rhs_terms = self.rhs_terms(include_heat_conduction=True, time=time)
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
        return {
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
                    field_name: getattr(term_rhs, field_name).copy()
                    for field_name in STATE_NAMES_1D
                }
                for term_name, term_rhs in rhs_terms.items()
            },
            "cathode_diagnostics": self._cathode_diagnostic_snapshot(time=time),
        }

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
            for field_name in STATE_NAMES_1D
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
            rhs_terms=rhs_terms,
            cathode_diagnostics=cathode_diagnostics,
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
        if saved and "M_n" in saved[0]:
            result.M_n = stack("M_n")
            result.u_n = stack("u_n")
        if saved and "nn_a" in saved[0]:
            result.nn_a = stack("nn_a")
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
            tau_discharge = max(
                float(self._input_dict.get("tau_discharge", 0.0)),
                0.0,
            )
            tau_cycle = max(float(self._input_dict.get("tau_cycle", 0.0)), 0.0)
            if tau_cycle <= 0.0:
                append_event(tau_discharge, "equilibrium_off", "tau_discharge")
                return _phase_event_arrays(events)

            cycle_index = np.floor(run_start / tau_cycle)
            cycle_start = cycle_index * tau_cycle
            while cycle_start <= final_time + 1e-15:
                cycle_end = cycle_start + tau_cycle
                if 0.0 < tau_discharge < tau_cycle:
                    append_event(
                        cycle_start + tau_discharge,
                        "equilibrium_off",
                        "tau_discharge",
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

        if self._phase_transition_mode() == "current":
            if self._t_prebreakdown_trigger is not None:
                append_event(
                    self._t_prebreakdown_trigger,
                    "breakdown",
                    "I_prebreakdown",
                )
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
        return {
            term_name: {
                field_name: np.stack(
                    [
                        snapshot["rhs_terms"][term_name][field_name]
                        for snapshot in saved
                    ]
                )
                for field_name in STATE_NAMES_1D
            }
            for term_name in term_names
        }

    def _phase_info(self, time):
        time = max(float(time), 0.0)
        tau_discharge = max(float(self._input_dict.get("tau_discharge", 0.0)), 0.0)
        if not self._flags.get("Plasma", True):
            tau_cycle = max(float(self._input_dict.get("tau_cycle", 0.0)), 0.0)
            if tau_cycle <= 0.0:
                cycle_time = time
            else:
                cycle_time = time % tau_cycle
            if cycle_time < tau_discharge:
                return "equilibrium_puff", cycle_time
            return "equilibrium_off", cycle_time - tau_discharge

        tau_prebreakdown = max(
            float(self._input_dict.get("tau_prebreakdown", 0.0)),
            0.0,
        )
        tau_breakdown = max(float(self._input_dict.get("tau_breakdown", 0.0)), 0.0)
        tau_afterglow = max(float(self._input_dict.get("tau_afterglow", 0.0)), 0.0)
        plasma_origin = self._plasma_phase_time_origin()
        if plasma_origin > 0.0 and time < plasma_origin:
            return "neutral_prebreakdown", time
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
        }
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
        return diag

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
