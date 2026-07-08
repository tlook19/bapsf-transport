from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from .core.config import (
    default_config,
    input_dict_template_1d,
    input_flags_template_1d,
    load_config,
    resolve_nn0,
)
from .core.geometry import build_geometry
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
    cathode_source_terms,
    solve_cathode_boundary,
)
from .physics.energy import (
    electron_cooling_rhs,
    electron_ion_exchange_rhs,
    ion_charge_exchange_rhs,
)
from .physics.flux import plasma_flux_rhs, plasma_flux_rhs_terms
from .physics.neutrals import (
    neutral_exchange_coefficients,
    neutral_exchange_rhs,
    neutral_source_sink_rhs,
)
from .physics.reactions import reaction_rhs, reaction_rhs_terms
from .physics.sources import (
    add_state_rhs,
    pressure_work_rhs,
    surface_neutralization_rhs,
)
from cablp.vars._cons import I_Ry, I_ion, m_He_cgs, m_p_cgs


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


class BreakdownError(RuntimeError):
    """Raised when current-triggered plasma breakdown thresholds are missed."""


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
    ):
        self._input_dict = dict(input_dict)
        self._flags = dict(input_flags)
        self._progress_callback = progress_callback
        self._gas_type = self._input_dict.get("gas_type", "He")
        (
            self._ion_mass_g,
            self._mu,
            self._mu_neutral,
            self._I_ion,
        ) = self._gas_constants(self._gas_type)
        self._geometry = build_geometry(self._input_dict)
        self._validate_phase_config()
        self._validate_gas_puff_config()
        self._floors = {
            "n": float(self._input_dict["ne_floor"]),
            "nn": float(self._input_dict["nn_floor"]),
            "Te": float(self._input_dict["Te_floor"]),
            "Ti": float(self._input_dict["Ti_floor"]),
        }
        self._state = self._initial_state()
        self._state = apply_state_floors(self._state, self._floors, self._ion_mass_g)
        self._y = pack_state(self._state)
        self._derived = derive_state(self._state, self._floors, self._ion_mass_g)
        self._time = 0.0
        self._t_prebreakdown_trigger = None
        self._t_breakdown_trigger = None
        self._cathode_x0 = None
        self._cathode_x0_twin = None
        self._cathode_beam_cross = np.zeros(self._geometry.cells)
        self._cathode_solve = None
        if self._flags.get("debug_checks", False):
            assert_finite_state(self._state, self._derived)

    @property
    def geometry(self):
        return self._geometry

    @property
    def state(self):
        return unpack_state(self._y, self._geometry.cells)

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
        return pack_state(state_rhs)

    def rhs_terms(self, y=None, include_heat_conduction=True, time=None):
        """Return named conservative RHS contributions for diagnostics."""
        state = self.state if y is None else unpack_state(y, self._geometry.cells)
        plasma_terms = self.plasma_flux_rhs_terms(state=state)
        reaction_terms = self.reaction_rhs_terms(state=state)
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
            "plasma_advective_flux": plasma_terms["plasma_advective_flux"],
            "plasma_front_flux": plasma_terms["plasma_front_flux"],
            "pressure_work": self.pressure_work_rhs(state=state),
            "ei_exchange": self.energy_exchange_rhs(state=state),
            "electron_cooling": self.electron_cooling_rhs(state=state),
            "ion_charge_exchange": self.ion_charge_exchange_rhs(state=state),
            "surface_loss": self.surface_neutralization_rhs(state=state),
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
            "recombination_loss": reaction_terms["recombination_loss"],
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
        )

    def advance_one_step(self, dt=None, operator_split=None):
        """Advance the conservative state by one explicit or split step."""
        if operator_split is None:
            operator_split = self._flags.get("implicit_heat_conduction", False)
        if operator_split:
            return self.advance_one_step_operator_split(dt=dt)
        if dt is None:
            dt = self.suggest_timestep().dt
        self._set_state_vector(
            ssprk2_step(
                y0=self._y,
                dt=dt,
                rhs_func=self.rhs,
                floor_func=self.floor_state_vector,
            )
        )
        self._time += dt
        return self.get_initial_snapshot()

    def operator_split_step(self, y=None, dt=None):
        """Return one explicit-nonheat plus implicit-heat split step."""
        y0 = self._y if y is None else np.asarray(y, dtype=float)
        if dt is None:
            dt = self.suggest_timestep(
                y=y0,
                include_heat_conduction=False,
            ).dt
        explicit_y = ssprk2_step(
            y0=y0,
            dt=dt,
            rhs_func=lambda yy: self.rhs(yy, include_heat_conduction=False),
            floor_func=self.floor_state_vector,
        )
        heat_state = self.implicit_heat_conduction_step(dt=dt, y=explicit_y)
        return self.floor_state_vector(pack_state(heat_state))

    def advance_one_step_operator_split(self, dt=None):
        """Advance by explicit non-heat terms then implicit heat conduction."""
        if dt is None:
            dt = self.suggest_timestep(include_heat_conduction=False).dt
        self._set_state_vector(self.operator_split_step(dt=dt))
        self._time += dt
        return self.get_initial_snapshot()

    def run(self, t_end, dt=None, operator_split=None, max_steps=100000):
        """Advance to ``t_end`` and return sparse saved trajectory arrays."""
        t_end = float(t_end)
        if t_end < self._time:
            raise ValueError(f"t_end must be >= current time ({t_end} < {self._time})")
        if max_steps <= 0:
            raise ValueError(f"max_steps must be positive (got {max_steps})")

        dt_save = float(self._input_dict.get("dt_save", 1e-5))
        t_save_start = float(self._input_dict.get("t_save_start", 0.0))
        max_output_steps = int(self._input_dict.get("max_output_steps", 0))
        saved = []
        diagnostics = []
        t_last_save = -np.inf
        time_tol = max(1e-15, 1e-12 * max(abs(t_end), 1.0))
        run_start = float(self._time)

        def should_save(t):
            if max_output_steps > 0 and len(saved) >= max_output_steps:
                return False
            if t + 1e-15 < t_save_start:
                return False
            if dt_save <= 0.0:
                return True
            return t - t_last_save >= dt_save - time_tol or abs(t - t_end) <= time_tol

        if should_save(self._time):
            saved.append(self._trajectory_snapshot(self._time))
            t_last_save = self._time

        steps = 0
        while self._time < t_end - time_tol:
            if steps >= max_steps:
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
            step_dt = min(step_dt, t_end - self._time)
            next_phase_boundary = self.next_phase_boundary_after(
                self._time,
                t_end=t_end,
                time_tol=time_tol,
            )
            if next_phase_boundary is not None:
                step_dt = min(step_dt, next_phase_boundary - self._time)
            if step_dt <= 0.0:
                raise RuntimeError(f"non-positive timestep selected ({step_dt})")
            self.advance_one_step(dt=step_dt, operator_split=operator_split)
            self._update_current_phase_triggers()
            diagnostics.append(diag)
            steps += 1
            if self._progress_callback is not None and t_end > 0.0:
                self._progress_callback(min(self._time / t_end, 1.0))
            if should_save(self._time):
                saved.append(self._trajectory_snapshot(self._time))
                t_last_save = self._time

        return self._trajectory_result(
            saved=saved,
            diagnostics=diagnostics,
            steps=steps,
            run_start=run_start,
        )

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
        state = self.state if y is None else unpack_state(y, self._geometry.cells)
        if include_heat_conduction is None:
            include_heat_conduction = not self._flags.get(
                "implicit_heat_conduction", False
            )
        diag = suggest_timestep(
            state=state,
            floors=self._floors,
            ion_mass_g=self._ion_mass_g,
            mu=self._mu,
            geometry=self._geometry,
            neutral_exchange_coeff_cm3_s=self.neutral_exchange_coefficients(),
            neutral_source_kwargs=self._neutral_source_kwargs(time=time),
            reaction_kwargs=self._reaction_kwargs(),
            energy_exchange_kwargs=self._energy_exchange_kwargs(),
            electron_cooling_kwargs=self._electron_cooling_kwargs(),
            ion_charge_exchange_kwargs=self._ion_charge_exchange_kwargs(),
            heat_conduction_kwargs=(
                self._heat_conduction_kwargs() if include_heat_conduction else None
            ),
            surface_loss_kwargs=self._surface_loss_kwargs(),
            cfl=float(self._input_dict.get("cfl", 0.4)),
            density_dt_fraction=float(
                self._input_dict.get("density_dt_fraction", 0.25)
            ),
            neutral_dt_fraction=float(
                self._input_dict.get("neutral_dt_fraction", 0.25)
            ),
            heat_dt_fraction=float(self._input_dict.get("heat_dt_fraction", 0.25)),
            dt_min=float(self._input_dict.get("dt_min", 1e-12)),
            dt_max=float(self._input_dict.get("dt_max", 1e-6)),
            include_front=self._flags.get("front_flux", True),
            alpha_front=float(self._input_dict.get("alpha_front", 1.0)),
        )
        if time is None:
            time = self._time
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
        if self._phase_transition_mode() == "current":
            boundaries = []
            main_start = self._t_breakdown_trigger
            if main_start is None:
                boundaries.append(tau_prebreakdown)
            else:
                boundaries.extend(
                    [
                        main_start + tau_discharge,
                        main_start + tau_discharge + tau_afterglow,
                    ]
                )
        else:
            main_start = tau_prebreakdown + tau_breakdown
            boundaries = [
                tau_prebreakdown,
                main_start,
                main_start + tau_discharge,
                main_start + tau_discharge + tau_afterglow,
            ]
        gas_event = self._gas_puff_event_time()
        if gas_event is not None:
            boundaries.append(gas_event)
        for boundary in sorted(boundaries):
            if in_run_window(boundary):
                return float(boundary)
        return None

    def plasma_flux_rhs(self, y=None, include_front=None):
        """Return the conservative plasma flux RHS for inspection/testing."""
        state = self.state if y is None else unpack_state(y, self._geometry.cells)
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
            state = self.state if y is None else unpack_state(y, self._geometry.cells)
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
            state = self.state if y is None else unpack_state(y, self._geometry.cells)
        return pressure_work_rhs(
            state=state,
            floors=self._floors,
            ion_mass_g=self._ion_mass_g,
            geometry=self._geometry,
            electron_scale=float(self._input_dict.get("b_pressure_work_elec", 1.0)),
            ion_scale=float(self._input_dict.get("b_pressure_work_ions", 1.0)),
        )

    def surface_neutralization_rhs(self, y=None, state=None):
        """Return conservative source/end surface neutralization terms."""
        if state is None:
            state = self.state if y is None else unpack_state(y, self._geometry.cells)
        return surface_neutralization_rhs(
            state=state,
            floors=self._floors,
            ion_mass_g=self._ion_mass_g,
            mu=self._mu,
            geometry=self._geometry,
            **self._surface_loss_kwargs(),
        )

    def energy_exchange_rhs(self, y=None, state=None):
        """Return conservative electron-ion thermal exchange sources."""
        if state is None:
            state = self.state if y is None else unpack_state(y, self._geometry.cells)
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
            state = self.state if y is None else unpack_state(y, self._geometry.cells)
        return electron_cooling_rhs(
            state=state,
            floors=self._floors,
            ion_mass_g=self._ion_mass_g,
            **self._electron_cooling_kwargs(),
        )

    def ion_charge_exchange_rhs(self, y=None, state=None):
        """Return conservative ion charge-exchange cooling sources."""
        if state is None:
            state = self.state if y is None else unpack_state(y, self._geometry.cells)
        return ion_charge_exchange_rhs(
            state=state,
            floors=self._floors,
            ion_mass_g=self._ion_mass_g,
            **self._ion_charge_exchange_kwargs(),
        )

    def heat_conduction_rhs(self, y=None, state=None):
        """Return conservative axial heat-conduction energy sources."""
        if state is None:
            state = self.state if y is None else unpack_state(y, self._geometry.cells)
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
            state = self.state if y is None else unpack_state(y, self._geometry.cells)
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
            state = self.state if y is None else unpack_state(y, self._geometry.cells)
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
            state = self.state if y is None else unpack_state(y, self._geometry.cells)
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
            state = self.state if y is None else unpack_state(y, self._geometry.cells)
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
            state = self.state if y is None else unpack_state(y, self._geometry.cells)
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
        )
        if update_cache:
            self._cathode_solve = result
            self._cathode_x0 = result.x0_next
            self._cathode_x0_twin = result.x0_twin_next
            if result.beam_result is not None:
                self._cathode_beam_cross = result.beam_result.beam_cross.copy()
        return result

    def implicit_heat_conduction_step(self, dt, y=None, state=None):
        """Return state after one frozen-conductivity implicit heat substep."""
        if state is None:
            state = self.state if y is None else unpack_state(y, self._geometry.cells)
        return implicit_heat_conduction_step(
            state=state,
            floors=self._floors,
            ion_mass_g=self._ion_mass_g,
            mu=self._mu,
            geometry=self._geometry,
            dt=dt,
            **self._heat_conduction_kwargs(),
        )

    def neutral_exchange_rhs(self, y=None, state=None):
        """Return conservative pairwise neutral-exchange sources."""
        if state is None:
            state = self.state if y is None else unpack_state(y, self._geometry.cells)
        return neutral_exchange_rhs(
            state=state,
            geometry=self._geometry,
            exchange_coeff_cm3_s=self.neutral_exchange_coefficients(),
        )

    def neutral_exchange_coefficients(self):
        """Return internal-face neutral exchange coefficients [cm^3/s]."""
        return neutral_exchange_coefficients(
            geometry=self._geometry,
            model=self._input_dict.get("neutral_exchange_model", "molecular_flow"),
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
            state = self.state if y is None else unpack_state(y, self._geometry.cells)
        return neutral_source_sink_rhs(
            state=state,
            geometry=self._geometry,
            **self._neutral_source_kwargs(time=time),
        )

    def reaction_rhs(self, y=None, state=None):
        """Return conservative bulk reaction sources."""
        if state is None:
            state = self.state if y is None else unpack_state(y, self._geometry.cells)
        return reaction_rhs(
            state=state,
            floors=self._floors,
            ion_mass_g=self._ion_mass_g,
            geometry=self._geometry,
            **self._reaction_kwargs(),
        )

    def reaction_rhs_terms(self, y=None, state=None):
        """Return split ionization and recombination conservative sources."""
        if state is None:
            state = self.state if y is None else unpack_state(y, self._geometry.cells)
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
        return {
            "configured": configured,
            "cathode_enabled": cathode_enabled,
            "floating": floating,
            "solve_enabled": cathode_enabled or floating,
        }

    def _effective_cathode_flags(self, time=None, active_only=True, floating=None):
        options = self._cathode_phase_options(time=time)
        enabled = options["cathode_enabled"]
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
        }

    def _validate_gas_puff_config(self):
        mode = self._input_dict.get("gas_puff_mode", "decay_after_breakdown")
        if mode not in {"decay_after_breakdown", "pulse_decay_to_level"}:
            raise ValueError(
                "gas_puff_mode must be 'decay_after_breakdown' or "
                f"'pulse_decay_to_level' (got {mode!r})"
            )
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
        tau_prebreakdown = max(
            float(self._input_dict.get("tau_prebreakdown", 0.0)),
            0.0,
        )
        tau_breakdown = max(float(self._input_dict.get("tau_breakdown", 0.0)), 0.0)
        return tau_prebreakdown + tau_breakdown

    def _effective_gas_puff_sccm(self, time=None):
        if time is None:
            time = self._time
        phase, phase_elapsed = self._phase_info(time)
        S_gp = float(self._input_dict.get("S_gp", 0.0))
        Twin_S_gp = float(self._input_dict.get("Twin_S_gp", 0.0))
        if not self._flags.get("Plasma", True):
            return S_gp, Twin_S_gp
        if phase in {"pre_breakdown", "breakdown"}:
            return S_gp, Twin_S_gp
        if phase != "main_discharge":
            return 0.0, 0.0

        mode = self._input_dict.get("gas_puff_mode", "decay_after_breakdown")
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

    def _update_current_phase_triggers(self):
        if (
            self._phase_transition_mode() != "current"
            or not self._flags.get("Plasma", True)
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

        if self._t_breakdown_trigger is not None:
            return
        if self._t_prebreakdown_trigger is None:
            first_threshold = I_prebreakdown if I_prebreakdown > 0.0 else I_breakdown
            if I_now >= first_threshold:
                if I_prebreakdown > 0.0:
                    self._t_prebreakdown_trigger = self._time
                else:
                    self._t_breakdown_trigger = self._time
                return
            if self._time >= tau_prebreakdown - time_tol:
                raise BreakdownError(
                    "plasma failed to break down within "
                    f"tau_prebreakdown={tau_prebreakdown:.9e} s "
                    f"(I_tot={I_now:.6g} A < threshold={first_threshold:.6g} A)"
                )
            return

        if I_now >= I_breakdown:
            self._t_breakdown_trigger = self._time
            return
        if self._time >= tau_prebreakdown - time_tol:
            raise BreakdownError(
                "plasma failed to reach breakdown current within "
                f"tau_prebreakdown={tau_prebreakdown:.9e} s "
                f"(I_tot={I_now:.6g} A < I_breakdown={I_breakdown:.6g} A)"
            )

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
            "Te_birth_ionization": self._input_dict.get(
                "Te_birth_ionization", "local"
            ),
            "Ti_birth_ionization": self._input_dict.get(
                "Ti_birth_ionization", "floor"
            ),
        }

    def _trajectory_snapshot(self, time):
        state = self.state
        derived = self.derived
        assert_finite_state(state, derived)
        rhs_terms = self.rhs_terms(include_heat_conduction=True, time=time)
        phase, phase_elapsed = self._phase_info(time)
        phase_switches = self._phase_switches(phase)
        return {
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

    def _trajectory_result(self, saved, diagnostics, steps, run_start):
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

        return SimpleNamespace(
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

    def _phase_events(self, run_start, final_time):
        run_start = float(run_start)
        final_time = float(final_time)
        events = []

        def append_event(time, phase, reason):
            time = float(time)
            if time < run_start - 1e-15 or time > final_time + 1e-15:
                return
            if events and abs(events[-1][0] - time) <= 1e-15:
                events[-1] = (time, phase, reason)
                return
            events.append((time, phase, reason))

        append_event(run_start, self.phase_at_time(run_start), "initial")
        if not self._flags.get("Plasma", True):
            return _phase_event_arrays(events)

        tau_prebreakdown = max(
            float(self._input_dict.get("tau_prebreakdown", 0.0)),
            0.0,
        )
        tau_breakdown = max(float(self._input_dict.get("tau_breakdown", 0.0)), 0.0)
        tau_discharge = max(float(self._input_dict.get("tau_discharge", 0.0)), 0.0)
        tau_afterglow = max(float(self._input_dict.get("tau_afterglow", 0.0)), 0.0)

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

        breakdown_start = tau_prebreakdown
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
                    return "pre_breakdown", time
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
            return "pre_breakdown", time

        breakdown_start = tau_prebreakdown
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
                and phase in discharge_phases
            ),
            "floating": phase == "afterglow",
        }

    def _cathode_diagnostic_snapshot(self, time=None):
        cells = self._geometry.cells
        cathode_phase = self._cathode_phase_options(time=time)
        diag = {
            "enabled": float(bool(self._flags.get("cathode_coupling", False))),
            "configured": float(cathode_phase["configured"]),
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
        self._state = unpack_state(self._y, self._geometry.cells)
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
