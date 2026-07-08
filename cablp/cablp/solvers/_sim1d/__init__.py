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


def load_result_hdf5(path):
    """Load a saved sim1d HDF5 result without constructing a solver."""
    from .results.io import load_result_hdf5 as _load_result_hdf5

    return _load_result_hdf5(path)


def summarize_result(result):
    """Return lightweight health diagnostics for a sim1d run result."""
    from .results.health import summarize_result as _summarize_result

    return _summarize_result(result)


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

    def rhs(self, y=None, include_heat_conduction=True):
        """Return the packed explicit RHS for the current scaffold physics."""
        state_rhs = self._zero_rhs_state()
        for term in self.rhs_terms(
            y=y,
            include_heat_conduction=include_heat_conduction,
        ).values():
            state_rhs = add_state_rhs(state_rhs, term)
        return pack_state(state_rhs)

    def rhs_terms(self, y=None, include_heat_conduction=True):
        """Return named conservative RHS contributions for diagnostics."""
        state = self.state if y is None else unpack_state(y, self._geometry.cells)
        plasma_terms = self.plasma_flux_rhs_terms(state=state)
        reaction_terms = self.reaction_rhs_terms(state=state)
        cathode_solve = None
        if self._flags.get("cathode_coupling", False):
            cathode_solve = self.solve_cathode_boundary(
                state=state,
                update_cache=True,
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
            ).rhs,
            "neutral_exchange": self.neutral_exchange_rhs(state=state),
            "neutral_sources": self.neutral_source_sink_rhs(state=state),
            "ionization_birth": reaction_terms["ionization_birth"],
            "beam_ionization_birth": self.beam_ionization_rhs(
                state=state,
                cathode_solve=cathode_solve,
            ),
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
            if step_dt <= 0.0:
                raise RuntimeError(f"non-positive timestep selected ({step_dt})")
            self.advance_one_step(dt=step_dt, operator_split=operator_split)
            diagnostics.append(diag)
            steps += 1
            if self._progress_callback is not None and t_end > 0.0:
                self._progress_callback(min(self._time / t_end, 1.0))
            if should_save(self._time):
                saved.append(self._trajectory_snapshot(self._time))
                t_last_save = self._time

        return self._trajectory_result(saved=saved, diagnostics=diagnostics, steps=steps)

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

    def suggest_timestep(self, y=None, include_heat_conduction=None):
        """Return an explicit timestep suggestion and diagnostics."""
        state = self.state if y is None else unpack_state(y, self._geometry.cells)
        if include_heat_conduction is None:
            include_heat_conduction = not self._flags.get(
                "implicit_heat_conduction", False
            )
        return suggest_timestep(
            state=state,
            floors=self._floors,
            ion_mass_g=self._ion_mass_g,
            mu=self._mu,
            geometry=self._geometry,
            neutral_exchange_coeff_cm3_s=self.neutral_exchange_coefficients(),
            neutral_source_kwargs=self._neutral_source_kwargs(),
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

    def cathode_source_terms(self, y=None, state=None, cathode_solve=None):
        """Return opt-in cathode conservative source placeholders/terms."""
        if state is None:
            state = self.state if y is None else unpack_state(y, self._geometry.cells)
        if cathode_solve is None and self._flags.get("cathode_coupling", False):
            cathode_solve = self.solve_cathode_boundary(
                state=state,
                update_cache=True,
            )
        return cathode_source_terms(
            state=state,
            floors=self._floors,
            ion_mass_g=self._ion_mass_g,
            geometry=self._geometry,
            input_dict=self._input_dict,
            input_flags=self._flags,
            cathode_solve=cathode_solve,
        )

    def beam_ionization_rhs(self, y=None, state=None, cathode_solve=None):
        """Return conservative beam ionization birth terms."""
        if state is None:
            state = self.state if y is None else unpack_state(y, self._geometry.cells)
        if cathode_solve is None and self._flags.get("cathode_coupling", False):
            cathode_solve = self.solve_cathode_boundary(
                state=state,
                update_cache=True,
            )
        return beam_ionization_rhs(
            state=state,
            floors=self._floors,
            ion_mass_g=self._ion_mass_g,
            geometry=self._geometry,
            input_dict=self._input_dict,
            input_flags=self._flags,
            I_ion=self._I_ion,
            cathode_solve=cathode_solve,
        )

    def solve_cathode_boundary(
        self,
        y=None,
        state=None,
        floating=False,
        update_cache=True,
    ):
        """Run the opt-in cathode solver adapter without changing the RHS."""
        if state is None:
            state = self.state if y is None else unpack_state(y, self._geometry.cells)
        result = solve_cathode_boundary(
            state=state,
            floors=self._floors,
            ion_mass_g=self._ion_mass_g,
            mu=self._mu,
            geometry=self._geometry,
            input_dict=self._input_dict,
            input_flags=self._flags,
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

    def neutral_source_sink_rhs(self, y=None, state=None):
        """Return conservative neutral gas puff and pump sources."""
        if state is None:
            state = self.state if y is None else unpack_state(y, self._geometry.cells)
        return neutral_source_sink_rhs(
            state=state,
            geometry=self._geometry,
            **self._neutral_source_kwargs(),
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

    def _neutral_source_kwargs(self):
        return {
            "S_gp": float(self._input_dict.get("S_gp", 0.0)),
            "Twin_S_gp": float(self._input_dict.get("Twin_S_gp", 0.0)),
            "S_pump_L": float(self._input_dict.get("S_pump_L", 0.0)),
            "S_pump_R": float(self._input_dict.get("S_pump_R", 0.0)),
            "twin_cathode": self._flags.get("TwinCathode", False),
            "gas_puff_enabled": bool(self._input_dict.get("gas_puff_enabled", True)),
            "pump_enabled": bool(self._input_dict.get("pump_enabled", True)),
            "gas_puff_valves": float(self._input_dict.get("gas_puff_valves", 2)),
        }

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
        rhs_terms = self.rhs_terms(include_heat_conduction=True)
        return {
            "time": float(time),
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
        }

    def _trajectory_result(self, saved, diagnostics, steps):
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
            total_rhs=total_rhs,
            electron_energy_terms_W_cm3=electron_energy_terms_W_cm3,
            ion_energy_terms_W_cm3=ion_energy_terms_W_cm3,
            diagnostics=list(diagnostics),
            steps=int(steps),
            final_time=float(self._time),
        )

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
