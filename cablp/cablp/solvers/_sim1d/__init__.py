from types import SimpleNamespace

import numpy as np

from .config import (
    default_config,
    input_dict_template_1d,
    input_flags_template_1d,
    load_config,
    resolve_nn0,
)
from .geometry import build_geometry
from .flux import plasma_flux_rhs
from .integrator import (
    floor_state_vector,
    ssprk2_step,
)
from .neutrals import (
    neutral_exchange_coefficients,
    neutral_exchange_rhs,
    neutral_source_sink_rhs,
)
from .sources import add_state_rhs, pressure_work_rhs
from .state import (
    apply_state_floors,
    assert_finite_state,
    conservative_from_primitives,
    derive_state,
    pack_state,
    unpack_state,
)
from .timestep import suggest_timestep
from cablp.vars._cons import I_Ry, I_ion, m_He_cgs, m_p_cgs


class LAPDSim1D:
    """Conservative axial 1D LAPD solver scaffold.

    This scaffold currently includes configuration, geometry, state handling,
    conservative plasma fluxes, pressure work, neutral terms, and a minimal
    explicit SSPRK2 step.
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
    def floors(self):
        return dict(self._floors)

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
        )

    def rhs(self, y=None):
        """Return the packed explicit RHS for the current scaffold physics."""
        state = self.state if y is None else unpack_state(y, self._geometry.cells)
        flux_rhs = self.plasma_flux_rhs(y=pack_state(state))
        pressure_rhs = self.pressure_work_rhs(state=state)
        neutral_rhs = self.neutral_exchange_rhs(state=state)
        source_rhs = self.neutral_source_sink_rhs(state=state)
        state_rhs = add_state_rhs(
            add_state_rhs(add_state_rhs(flux_rhs, pressure_rhs), neutral_rhs),
            source_rhs,
        )
        return pack_state(state_rhs)

    def floor_state_vector(self, y):
        """Apply configured density and temperature floors to a packed vector."""
        return floor_state_vector(
            y=y,
            cells=self._geometry.cells,
            floors=self._floors,
            ion_mass_g=self._ion_mass_g,
        )

    def advance_one_step(self, dt=None):
        """Advance the conservative state by one explicit SSPRK2 step."""
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
        return self.get_initial_snapshot()

    def suggest_timestep(self, y=None):
        """Return an explicit timestep suggestion and diagnostics."""
        state = self.state if y is None else unpack_state(y, self._geometry.cells)
        return suggest_timestep(
            state=state,
            floors=self._floors,
            ion_mass_g=self._ion_mass_g,
            mu=self._mu,
            geometry=self._geometry,
            neutral_exchange_coeff_cm3_s=self.neutral_exchange_coefficients(),
            neutral_source_kwargs=self._neutral_source_kwargs(),
            cfl=float(self._input_dict.get("cfl", 0.4)),
            density_dt_fraction=float(
                self._input_dict.get("density_dt_fraction", 0.25)
            ),
            neutral_dt_fraction=float(
                self._input_dict.get("neutral_dt_fraction", 0.25)
            ),
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
