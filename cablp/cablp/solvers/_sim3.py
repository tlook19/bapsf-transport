import math
import time
import tomllib
import numpy as np
from types import SimpleNamespace
from cablp.funcs._cathode_solver import (
    DeviceConfig,
    solve_beam_system,
)
from cablp.funcs._heat import (
    elec_par_heat_loss,
    ion_par_heat_loss,
    elec_par_heat_div,
    ion_par_heat_div,
    elec_par_heat_face_flux,
    ion_par_heat_face_flux,
    Q_cx_He,
    Q_ie,
)
from cablp.vars._cons import (
    qe_SI,
    en_factor,
    drag_factor,
    ev_to_erg,
    m_He_cgs,
    m_p_cgs,
    kb_cgs,
)
from cablp.funcs._fits import rate_coeff, IAEA_exp1, IAEA_exp4, IAEA_exp6
from cablp.funcs._cross import alpha_3, alpha_r
from cablp.vars._coeff import aHeI, aHeII, aHI, aHII, He_ion_coeff
from cablp.vars._cons import I_ion as IE_Helium, I_Ry as IE_Hydrogen
from cablp.funcs._plasmaparams import v_ion_speed, v_thm_e, time_elec_coll, time_ion_coll, c_log
from cablp.vars._nn_table import lookup_nn0

H_ion_coeff = [1e-5, 6.0]


def _to_odd(n):
    """Round n up to the nearest odd integer."""
    return n if n % 2 == 1 else n + 1


class BreakdownError(RuntimeError):
    """Raised when the plasma fails to reach the breakdown current within tau_prebreakdown."""


class NumericalInstabilityError(RuntimeError):
    """Raised by debug checks when an accepted step creates an implausible state jump."""


_STATE_NAMES = ("ne", "nn", "Te", "Ti", "v_plasma")


_CATHODE_FIELDS = (
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
    "P_anode_e",
    "P_anode_i",
    "P_net",
    "P_net2",
    "P_loss",
    "long_mfp",
    "beam_bypass_fraction",
    "l_b",
)
_CATHODE_NAN = np.full(len(_CATHODE_FIELDS), np.nan)


def _cathode_to_array(result):
    if result is None:
        return _CATHODE_NAN.copy()
    return np.array([getattr(result, f) for f in _CATHODE_FIELDS])


input_dict_template = {
    "gas_type": "He",
    "ne0": 1e10,
    "Tn_fit": 0.1,  # Neutral temperature for reaction rate fits
    "Te0": 0.1,
    "Ti0": 0.1,
    "Lm": 1800,  # Length of machine
    "Rm": 50,  # Machine radius
    "Lp": 1800,  # Length of plasma
    "Rp": 18,  # Plasma radius
    # Cathode device parameters (used by cathode solver)
    "V_bank": 100,  # Power supply voltage [V]
    "T_s": 1973.15,  # Cathode surface temperature [K]
    "phi_wf": 3.0,  # Work function [eV] (LaB6 default)
    "C_R": 29.0,  # Richardson constant [A cm⁻² K⁻²]
    "R_comp": 0.004,  # Compliance resistor [Ω]
    "eta": 0.358,  # Anode area / cathode area
    "L_cath": 50.0,  # Cathode-to-anode distance [cm]
    "R_cath": 18.0,  # Cathode radius [cm]; A_c = π * R_cath²
    "S_gp": 8000,  # Gas puff source rate
    "Twin_S_gp": 8000,
    "S_pump_L": 4000,  # Vacuum pump sink rate
    "S_pump_R": 4000,
    "b_epara": 1.0,  # Scaling factor for e_para transport
    "b_ipara": 1.0,  # Scaling factor for i_para transport
    "b_ioniz": 1.0,  # Scaling factor for ionization
    "b_rec_rad": 1.0,  # Scaling factor for radiative recombination
    "b_rec_3b": 1.0,  # Scaling factor for three-body recombination
    "b_Qcx": 1.0,  # Scaling factor for charge exchange
    "b_source": 1.0,  # Scaling factor for source heating
    "b_Qie": 1.0,  # Scaling factor for ion-electron heating
    "b_Qei": 1.0,  # Scaling factor for electron-ion cooling
    "b_Qen": 1.0,  # Scaling factor for electron-neutral cooling
    "cycles": 1,
    "tau_prebreakdown": 0.05,   # max pre-breakdown phase duration [s]
    "tau_discharge": 20e-3,    # main discharge duration after breakdown [s]
    "tau_gp_after_breakdown": None,  # optional S_gp exponential-decay start time after breakdown/main-discharge start [s]
    "tau_afterglow": 5e-3,    # afterglow duration after discharge [s]
    "tau_cycle": 3.0,          # total cycle length for Plasma=False [s]
    "I_prebreakdown": 100.0,     # I_tot threshold to exit floor-dominated pre-breakdown (0 = skip)
    "I_breakdown": 1000.0,     # I_tot threshold marking experimental t=0 (fixed at 1 kA for timing)
    "h0": 1e-6,                # initial adaptive step size [s]
    "h_max_discharge": 1e-4,   # max step size during active discharge phases [s]
    "h_max_afterglow": 1e-4,   # max step size during afterglow/off phases [s]
    "dt_save": 1e-5,            # min simulation time between saved output steps [s]; 0 = save every step
    "max_output_steps": 0,     # hard cap on total saved timesteps; 0 = unlimited
    "cells": 3,
    "rtol": 1e-3,  # relative tolerance for adaptive stepping
    "h_min": 1e-10,            # minimum allowed step size [s]
    "ne_cfl_floor": None,      # minimum ne [cm^-3] used when computing the electron-conduction CFL (interior cells only); None = use actual ne
    "ne_floor": 1e8,
    "nn_floor": 1e8,
    "Te_floor": 0.1,
    "Ti_floor": 0.1,
    "Te_reject_floor": 0.0,
    "Ti_reject_floor": 0.0,
    "ne_reject_floor": 0.0,
    "nn_reject_floor": 0.0,
    "debug_max_rel_step_change": np.inf,  # max per-step relative change when debug_checks is enabled
    "debug_max_neighbor_ratio": np.inf,  # max adjacent-cell ratio when debug_checks is enabled
    "v_atol_cs_fraction": 0.01,  # v_plasma abs tolerance as fraction of max(c_s); updated each accepted step
    "debug_step_atol": None,  # optional dict/list of per-component scales for accepted-step jump checks
    "debug_check_start_time": 0.005,  # skip debug jump checks before this absolute simulation time [s]
    "debug_ignore_floor_neighbors": True,  # ignore neighbor ratios involving floor-clamped values
    "alpha_ne_sonic_flux": 1.0,  # multiplier for hybrid_ne sound-speed pressure relaxation
    "beta_ne_sonic_flux": 1.0,  # max hybrid_ne correction as a fraction of ne_face*c_s_face
    "hybrid_ne_taper_dn0": 0.2,  # density contrast where hybrid_ne sonic correction is half strength
    "hybrid_ne_taper_power": 0,  # exponent applied to the hybrid_ne density-contrast taper
    "hybrid_ne_taper_delay": 5e-3,  # time after breakdown before hybrid_ne density taper is applied [s]
    "ion_pressure_weight": 1.0,  # ion contribution to pressure-gradient acceleration: Te + w_i*Ti
}

input_flags_template = {
    "icool": True,
    "ncool": True,
    "cx": True,
    "icool_recomb": False,
    "Plasma": True,
    "TwinCathode": False,
    "Velocity": True,
    "advection": True,  # Include v·∇v convective acceleration in velocity equation
    "adaptive_mesh": False,  # Dynamically refine/coarsen spatial cells based on MFP criterion
    "hybrid_ne": True,      # Interior face flux: velocity advection plus limited sonic pressure correction
    "debug_checks": False,   # Raise early on non-finite states or configured jump/gradient thresholds
    "debug_raise_on_guard": False,  # Raise when accepted endpoints require clipping/flooring
    "reject_floor_violations": True,  # Reject RK steps whose accepted endpoint crosses state floors
    "reject_large_step_changes": False,  # Reject RK steps whose endpoint exceeds debug_max_rel_step_change
}


def load_config(path):
    """
    Load simulation parameters and flags from a TOML file.

    The file should have two optional sections: ``[params]`` and ``[flags]``.
    Any key not present in the file falls back to the value in
    ``input_dict_template`` / ``input_flags_template``.

    Parameters
    ----------
    path : str or path-like
        Path to the ``.toml`` configuration file.

    Returns
    -------
    input_dict : dict
    input_flags : dict

    Example TOML
    ------------
    .. code-block:: toml

        [params]
        cells = 5
        Id    = 3000

        [flags]
        Velocity = true
    """
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    input_dict = {**input_dict_template, **raw.get("params", {})}
    input_flags = {**input_flags_template, **raw.get("flags", {})}
    return input_dict, input_flags


def default_config():
    """Return copies of the default input dictionary and flags."""
    return dict(input_dict_template), dict(input_flags_template)


class LAPDSim:
    def __init__(
        self,
        input_dict=input_dict_template,
        input_flags=input_flags_template,
        progress_callback=None,
    ):
        self._flags = input_flags
        self._input_dict = dict(input_dict)
        self._cathode_cell_len = 100.0  # fixed cathode cell length [cm]
        self._n_interior = _to_odd(input_dict.get("cells", 3))
        if self._n_interior > 19:
            raise ValueError(f"Interior cells must be <= 19 (got {self._n_interior})")
        self._n_cathode_cells = 2  # both ends always have a fixed 100 cm boundary cell
        self._cells = self._n_interior + self._n_cathode_cells
        self._gas_type = input_dict.get("gas_type", "He")
        if self._gas_type == "He":
            self._m_gas = m_He_cgs
            self._mu = 4
            self._mu_neutral = 4  # He atom
            self._I_ion = IE_Helium
            self._ion_fit_coeff = He_ion_coeff
        elif self._gas_type == "H":
            self._m_gas = m_p_cgs
            self._mu = 1
            self._mu_neutral = 2  # H2 molecule
            self._I_ion = IE_Hydrogen
            self._ion_fit_coeff = H_ion_coeff
        self._ne = np.ones(self._cells) * input_dict["ne0"]
        _nn0 = input_dict.get("nn0") or lookup_nn0(
            input_dict["S_gp"], twin=self._flags["TwinCathode"]
        )
        self._nn = np.ones(self._cells) * _nn0
        self._Te = np.ones(self._cells) * input_dict["Te0"]
        self._Ti = np.ones(self._cells) * input_dict["Ti0"]
        self._Tn_fit = input_dict.get("Tn_fit", 0.1)
        self._v_plasma = np.zeros(self._cells)
        self._L_machine = input_dict["Lm"]
        self._L_cell = self._build_L_cell(self._n_interior)
        self._R_machine = np.ones(self._cells) * input_dict["Rm"]
        self._L_plasma = self._build_L_plasma(self._n_interior)
        self.lam_min = 0.3 * np.min(self._L_plasma)
        self._R_plasma = np.ones(self._cells) * input_dict["Rp"]
        self._Rsq_ratio = (self._R_plasma / self._R_machine) ** 2
        self._L_heatflux = self._L_plasma / 2
        self._Tn = input_dict.get("Tn", 0.025)
        self._v_th_n = np.sqrt(8 * kb_cgs * 300.0 / (np.pi * self._mu_neutral * m_p_cgs))
        self._b_epara = input_dict.get("b_epara", 1.0)
        self._b_ipara = input_dict.get("b_ipara", 1.0)
        self._b_ioniz = input_dict.get("b_ioniz", 1.0)
        self._b_rec_rad = input_dict.get("b_rec_rad", 1.0)
        self._b_rec_3b = input_dict.get("b_rec_3b", 1.0)
        self._b_Qcx = input_dict.get("b_Qcx", 1.0)
        self._b_Qie = input_dict.get("b_Qie", 1.0)
        self._b_Qei = input_dict.get("b_Qei", 1.0)
        self._b_Qen = input_dict.get("b_Qen", 1.0)
        self._b_source = input_dict.get("b_source", 1.0)
        self._alpha_ne_sonic_flux = input_dict.get("alpha_ne_sonic_flux", 1.0)
        self._beta_ne_sonic_flux = input_dict.get("beta_ne_sonic_flux", 1.0)
        self._hybrid_ne_taper_dn0 = input_dict.get("hybrid_ne_taper_dn0", 0.2)
        self._hybrid_ne_taper_power = input_dict.get("hybrid_ne_taper_power", 1.0)
        self._hybrid_ne_taper_delay = input_dict.get("hybrid_ne_taper_delay", 5e-3)
        self._ion_pressure_weight = input_dict.get("ion_pressure_weight", 1.0)
        self._plasma_cross = np.pi * self._R_plasma**2
        self._plasma_vol = self._plasma_cross * self._L_plasma
        self._cell_vol = np.pi * self._R_machine**2 * self._L_cell
        self._S_gp = np.zeros(self._cells)
        self._S_pump = np.zeros(self._cells)
        self._S_gp[0] = self.puff_rate(input_dict["S_gp"], 2, self._cell_vol[0])
        self._S_pump[0] = self.pump_rate(input_dict["S_pump_L"], self._cell_vol[0])
        self._S_pump[-1] = self.pump_rate(input_dict["S_pump_R"], self._cell_vol[-1])
        if self._flags["TwinCathode"]:
            self._S_gp[-1] = self.puff_rate(
                input_dict["Twin_S_gp"], 2, self._cell_vol[-1]
            )
        # self._active_discharge = self._I_discharge > 0 : need to calculate later with cathode solver
        # size-2 arrays: index 0 = primary cathode, index 1 = twin cathode
        # updated each _dstep from cathode solver phi_c
        self._cycles = input_dict.get("cycles", 1)
        self._tau_prebreakdown = input_dict.get("tau_prebreakdown", 0.1)
        self._tau_discharge = input_dict.get("tau_discharge", 15e-3)
        self._tau_gp_after_breakdown = input_dict.get("tau_gp_after_breakdown", None)
        if self._tau_gp_after_breakdown is not None and self._tau_gp_after_breakdown < 0:
            raise ValueError(
                "tau_gp_after_breakdown must be >= 0 s, or None to keep S_gp steady "
                "through the main discharge"
            )
        self._tau_afterglow = input_dict.get("tau_afterglow", 10e-3)
        self._t_total = self._tau_prebreakdown + self._tau_discharge + self._tau_afterglow
        self._progress_callback = progress_callback
        self._last_cb_wall_t = None  # set on first callback tick
        self._rate_ema = 0.0         # wall seconds per 1ms sim; 0 = no data yet
        self._tau_cycle = input_dict.get("tau_cycle", 3.0)
        self._I_prebreakdown = input_dict.get("I_prebreakdown", 0.0)
        self._I_breakdown = input_dict.get("I_breakdown", 1000.0)
        self._h0 = input_dict.get("h0", 1e-6)
        self._h_max_discharge = input_dict.get("h_max_discharge", 1e-5)
        self._h_max_afterglow = input_dict.get("h_max_afterglow", 1e-4)
        self._div_v_elec = np.zeros(self._cells)
        self._div_v_ions = np.zeros(self._cells)
        self._Te_conv = np.zeros(self._cells)
        self._Ti_conv = np.zeros(self._cells)
        self._v_face = np.zeros(self._cells - 1)
        # self._ln_lambda = c_log(self._Te, self._ne)
        _R_cath = input_dict.get("R_cath", 19.0)
        self._eta = input_dict.get("eta", 0.358)
        self._device_config = DeviceConfig(
            A_c=math.pi * _R_cath**2,
            mu=self._mu,
            V_bank=input_dict.get("V_bank", 100.0),
            T_s=input_dict.get("T_s", 1900.0),
            phi_wf=input_dict.get("phi_wf", 3.0),
            C_R=input_dict.get("C_R", 29.0),
            R_comp=input_dict.get("R_comp", 0.004),
            eta=self._eta,
            Twin=self._flags["TwinCathode"],
            L_cath=input_dict.get("L_cath", 50.0),
            R_cath=_R_cath,
        )
        self._cathode_x0 = None
        self._cathode_x0_twin = None
        self._cathode_result = None
        self._cathode_result_twin = None
        self._gas_puff_on = True
        self._gas_puff_shutoff_reported = False
        self._t_current = 0.0
        self._rtol = input_dict.get("rtol", 1e-3)
        self._h_min = input_dict.get("h_min", 1e-12)
        self._ne_cfl_floor = input_dict.get("ne_cfl_floor", None)
        self._dt_save = input_dict.get("dt_save", 0.0)
        self._max_output_steps = input_dict.get("max_output_steps", 0)
        self._state_floor = np.array(
            [
                [input_dict.get("ne_floor", 1e8)],
                [input_dict.get("nn_floor", 1e8)],
                [input_dict.get("Te_floor", 0.1)],
                [input_dict.get("Ti_floor", 0.1)],
                [-np.inf],
            ]
        )
        self._reject_floor = np.array(
            [
                [input_dict.get("ne_reject_floor", 0.0)],
                [input_dict.get("nn_reject_floor", 0.0)],
                [input_dict.get("Te_reject_floor", 0.0)],
                [input_dict.get("Ti_reject_floor", 0.0)],
                [-np.inf],
            ]
        )
        # Per-component absolute tolerance matched to existing floor values.
        # Shape (5, 1) broadcasts with state shape (5, cells).
        self._v_atol_cs_fraction = input_dict.get("v_atol_cs_fraction", 0.01)
        _c_s0 = np.max(v_ion_speed(input_dict["Te0"], self._mu))
        self._atol = np.array(
            [
                [self._state_floor[0, 0]],
                [self._state_floor[1, 0]],
                [self._state_floor[2, 0]],
                [self._state_floor[3, 0]],
                [self._v_atol_cs_fraction * _c_s0],
            ]
        )
        self._debug_events = []
        self._debug_max_rel_step_change = input_dict.get(
            "debug_max_rel_step_change", np.inf
        )
        self._debug_max_neighbor_ratio = input_dict.get(
            "debug_max_neighbor_ratio", np.inf
        )
        self._debug_step_atol = self._parse_debug_step_atol(
            input_dict.get("debug_step_atol")
        )
        self._debug_check_start_time = input_dict.get("debug_check_start_time", 0.0)
        self._debug_ignore_floor_neighbors = input_dict.get(
            "debug_ignore_floor_neighbors", True
        )
        _max_interior = max(input_dict.get("max_cells", 19), self._n_interior)
        self._max_cells = _max_interior + self._n_cathode_cells
        self._min_cells = _to_odd(min(input_dict.get("min_cells", 3), self._n_interior))
        self._mfp_refine_thresh = input_dict.get("mfp_refine_threshold", 0.5)
        self._mfp_coarsen_thresh = input_dict.get("mfp_coarsen_threshold", 2.0)
        self._print_init_summary()

    def _parse_debug_step_atol(self, value):
        """Return per-component scales for debug accepted-step jump checks."""
        if value is None:
            return np.array([[1e8], [1e8], [0.05], [0.05], [1e5]])
        if isinstance(value, dict):
            return np.array([[value.get(name, default[0])] for name, default in zip(_STATE_NAMES, self._atol)])
        arr = np.asarray(value, dtype=float)
        if arr.shape != (len(_STATE_NAMES),):
            raise ValueError(
                f"debug_step_atol must have {len(_STATE_NAMES)} entries: {_STATE_NAMES}"
            )
        return arr.reshape(len(_STATE_NAMES), 1)

    def _build_L_plasma(self, n_interior):
        """Build plasma cell length array: fixed 100 cm cathode cells + uniform interior."""
        Lp = self._input_dict["Lp"]
        interior_L = (Lp - self._n_cathode_cells * self._cathode_cell_len) / n_interior
        arr = np.full(n_interior + self._n_cathode_cells, interior_L)
        arr[0] = self._cathode_cell_len
        if self._n_cathode_cells == 2:
            arr[-1] = self._cathode_cell_len
        return arr

    def _build_L_cell(self, n_interior):
        """Build machine cell length array: fixed 100 cm cathode cells + uniform interior."""
        interior_L = (self._L_machine - self._n_cathode_cells * self._cathode_cell_len) / n_interior
        arr = np.full(n_interior + self._n_cathode_cells, interior_L)
        arr[0] = self._cathode_cell_len
        if self._n_cathode_cells == 2:
            arr[-1] = self._cathode_cell_len
        return arr

    def _print_init_summary(self):
        """Print key derived quantities computed during __init__ for sanity checking."""
        print(f"=== LAPDSim init summary ===")
        print(f"  gas_type={self._gas_type}  mu={self._mu}  I_ion={self._I_ion:.4f} eV")
        print(f"  ion_fit_coeff={self._ion_fit_coeff}")
        print(
            f"  V_discharge={self._device_config.V_bank}  R_comp={self._device_config.R_comp}"
        )
        print(f"  T_s={self._device_config.T_s} K")
        print(f"  cells: {self._n_cathode_cells} cathode + {self._n_interior} interior = {self._cells} total")
        print(f"  L_plasma={self._L_plasma} cm")
        print(f"  plasma_vol={self._plasma_vol} cm^3")
        print(f"  Rsq_ratio={self._Rsq_ratio}")
        print(f"  S_gp={self._S_gp} cm^-3 s^-1")
        print(f"  S_pump={self._S_pump} s^-1")
        if self._flags["Plasma"]:
            if self._tau_gp_after_breakdown is not None:
                print(
                    f"  S_gp begins exponential decay {self._tau_gp_after_breakdown*1e3:.3f} ms "
                    "after breakdown/main_discharge start"
                )
            if self._I_prebreakdown > 0:
                print(f"  Phases: pre_breakdown(<{self._tau_prebreakdown*1e3:.1f} ms, I<{self._I_prebreakdown:.0f} A) "
                      f"→ breakdown(I<{self._I_breakdown:.0f} A) "
                      f"→ main_discharge({self._tau_discharge*1e3:.1f} ms) "
                      f"→ afterglow({self._tau_afterglow*1e3:.1f} ms)")
            else:
                print(f"  Phases: pre_breakdown(<{self._tau_prebreakdown*1e3:.1f} ms) "
                      f"→ main_discharge({self._tau_discharge*1e3:.1f} ms) "
                      f"→ afterglow({self._tau_afterglow*1e3:.1f} ms)  I_breakdown={self._I_breakdown:.0f} A")
        else:
            print(f"  Equilibrium: puffing({self._tau_discharge*1e3:.1f} ms) "
                  f"→ off({(self._tau_cycle - self._tau_discharge)*1e3:.1f} ms)  "
                  f"cycles={self._cycles}")
        print(f"============================")

    def set_time_steps(self):
        pass  # time stepping is now fully driven by the adaptive integrator

    def initialize_results(self):
        self._debug_events = []
        self._t_breakdown = None
        self._gas_puff_on = True
        self._gas_puff_shutoff_reported = False
        self._n_beam = np.zeros(self._cells)
        self._beam_cross = np.zeros(self._cells)
        self._l_b_profile = np.zeros(self._cells)
        self._l_b_profile_twin = np.zeros(self._cells)
        self._Qie = np.zeros(self._cells)  # Energy exchange between electrons and ions
        self._Qei = np.zeros(
            self._cells
        )  # Electron cooling due to inelastic collisions with ions
        self._Qen = np.zeros(
            self._cells
        )  # Electron cooling due to inelastic collisions with neutrals
        self._Qcx = np.zeros(
            self._cells
        )  # Ion cooling due to charge exchange with neutrals
        self._Qeb = np.zeros(self._cells)  # Electron heating due to discharge
        self._Qib = np.zeros(self._cells)  # Ion sheath power loss at boundaries
        self._e_par_hl = np.zeros(self._cells)  # Electron parallel heat loss
        self._i_par_hl = np.zeros(self._cells)  # Ion parallel heat loss
        self._e_par_flux = np.zeros(self._cells)  # Net electron parallel heat flux
        self._i_par_flux = np.zeros(self._cells)  # Net ion parallel heat flux
        self._e_par_face_flux = np.zeros(self._cells - 1)
        self._i_par_face_flux = np.zeros(self._cells - 1)
        self._Ne_face_flux = np.zeros(self._cells - 1)
        self._Nn_face_flux = np.zeros(self._cells - 1)
        self._ne_flux = np.zeros(
            self._cells
        )  # boundary outgoing electron particle flux
        self._nn_flux = np.zeros(self._cells)  # boundary outgoing neutral particle flux
        self._Ne_flux = np.zeros(self._cells)  # net electron particle flux
        self._Nn_flux = np.zeros(self._cells)  # net neutral particle flux
        self._S_ion_bulk = np.zeros(self._cells)  # Bulk ionization source
        self._S_rec_rad = np.zeros(self._cells)  # Radiative recombination sink
        self._S_rec_3b = np.zeros(self._cells)  # Three-body recombination sink
        self._S_ion_beam = np.zeros(self._cells)  # Beam ionization source
        self._t_last_save = -np.inf
        # Dynamic lists — converted to arrays by _finalize_results()
        self._time_list = []
        self._densities_list = []
        self._temperatures_list = []
        self._density_terms_list = []
        self._heat_terms_list = []
        self._face_fluxes_list = []
        self._velocities_list = []
        self._synthetic_list = []
        self._primary_mfp_list = []
        self._bulk_mfp_list = []
        self._ln_lambda_list = []
        self._cells_list = []
        self._refinement_events = []
        self._cathode_list = []
        self._cathode_twin_list = []

    def _pad(self, arr):
        """Pad a 1-D per-cell array to max_cells with NaN for uniform output shape."""
        out = np.full(self._max_cells, np.nan)
        out[: self._cells] = arr
        return out

    def _pad_face(self, arr):
        """Pad a 1-D interior-face array to max_cells - 1 with NaN."""
        out = np.full(self._max_cells - 1, np.nan)
        out[: self._cells - 1] = arr
        return out

    def update_results(self, t):
        if self._dt_save > 0 and t - self._t_last_save < self._dt_save:
            return
        if self._max_output_steps > 0 and len(self._time_list) >= self._max_output_steps:
            if len(self._time_list) == self._max_output_steps:
                print(f"  [output] max_output_steps={self._max_output_steps} reached at t={t:.4e} s; further steps will not be saved.")
            return
        self._t_last_save = t
        self._time_list.append(t)
        self._densities_list.append(
            np.array(
                [self._pad(self._ne), self._pad(self._nn), self._pad(self._n_beam)]
            )
        )
        self._temperatures_list.append(
            np.array([self._pad(self._Te), self._pad(self._Ti)])
        )
        self._density_terms_list.append(
            np.array(
                [
                    self._pad(self._Ne_flux),
                    self._pad(self._Nn_flux),
                    self._pad(self._S_ion_bulk),
                    self._pad(self._S_rec_rad),
                    self._pad(self._S_rec_3b),
                    self._pad(self._S_ion_beam),
                ]
            )
        )
        self._heat_terms_list.append(
            np.array(
                [
                    self._pad(self._e_par_flux),
                    self._pad(self._i_par_flux),
                    self._pad(self._Qie),
                    self._pad(self._Qei),
                    self._pad(self._Qen),
                    self._pad(self._Qcx),
                    self._pad(self._Qeb),
                    self._pad(self._div_v_elec),
                    self._pad(self._div_v_ions),
                    self._pad(self._Qib),
                    self._pad(self._Te_conv),
                    self._pad(self._Ti_conv),
                ]
            )
        )
        self._face_fluxes_list.append(
            np.array(
                [
                    self._pad_face(self._Ne_face_flux),
                    self._pad_face(self._Nn_face_flux),
                    self._pad_face(self._e_par_face_flux),
                    self._pad_face(self._i_par_face_flux),
                ]
            )
        )
        self._velocities_list.append(np.array([self._pad(self._v_plasma)]))
        self._synthetic_list.append(self._pad(self._ne * np.sqrt(self._Te)))
        self._cells_list.append(self._cells)
        self._cathode_list.append(
            _cathode_to_array(self._cathode_result if self._flags["Plasma"] else None)
        )
        self._cathode_twin_list.append(
            _cathode_to_array(
                self._cathode_result_twin
                if self._flags["Plasma"] and self._flags["TwinCathode"]
                else None
            )
        )
        if self._flags["Plasma"]:
            tau_e = time_elec_coll(self._Te, self._ne, self._ln_lambda)
            primary_mfp = self._l_b
            bulk_mfp = v_thm_e(self._Te) * tau_e
            ln_lambda = self._ln_lambda.copy()
        else:
            primary_mfp = np.zeros(self._cells)
            bulk_mfp = np.zeros(self._cells)
            ln_lambda = np.zeros(self._cells)
        self._primary_mfp_list.append(self._pad(primary_mfp))
        self._bulk_mfp_list.append(self._pad(bulk_mfp))
        self._ln_lambda_list.append(self._pad(ln_lambda))

    def _finalize_results(self):
        """Convert accumulated result lists to NumPy arrays after simulation."""
        self._time = np.array(self._time_list)
        if self._t_breakdown is not None:
            self._time -= self._t_breakdown
            self._refinement_events = [
                (t - self._t_breakdown, c1, c2) for t, c1, c2 in self._refinement_events
            ]
        self._densities = np.array(self._densities_list)  # (n, 3, max_cells)
        self._temperatures = np.array(self._temperatures_list)  # (n, 2, max_cells)
        self._density_terms = np.array(self._density_terms_list)  # (n, 6, max_cells)
        self._heat_terms = np.array(self._heat_terms_list)  # (n, 12, max_cells)
        self._face_fluxes = np.array(self._face_fluxes_list)  # (n, 4, max_cells - 1)
        self._velocities = np.array(self._velocities_list)  # (n, 1, max_cells)
        self._synthetic = np.array(self._synthetic_list)  # (n, max_cells)
        self._primary_mfp = np.array(self._primary_mfp_list)  # (n, max_cells)
        self._bulk_mfp = np.array(self._bulk_mfp_list)  # (n, max_cells)
        self._ln_lambda = np.array(self._ln_lambda_list)  # (n, max_cells)
        self._cells_at_time = np.array(self._cells_list)  # (n,)
        self._cathode = np.array(self._cathode_list)  # (n, n_fields)
        self._cathode_twin = np.array(self._cathode_twin_list)  # (n, n_fields)

    def calc_density_terms(
        self,
        ne,
        nn,
        Te,
        v_plasma,
        c_s=None,
    ):
        self._Ne_flux, self._Nn_flux = self._calc_n_flux(Te, ne, nn, v_plasma, c_s=c_s)
        self._S_ion_beam = np.zeros(self._cells)
        self._S_ion_bulk = np.zeros(self._cells)
        self._S_rec_rad = np.zeros(self._cells)
        self._S_rec_3b = np.zeros(self._cells)
        if self._flags["Plasma"]:
            if self._gas_type in ("He", "H"):
                self._S_ion_bulk = (
                    self._b_ioniz
                    * ne
                    * nn
                    * rate_coeff(Te, self._I_ion, *self._ion_fit_coeff)
                )
                for i in [0, -1] if self._flags["TwinCathode"] else [0]:
                    if self._beam_cross[i] == 0.0:
                        continue
                    l_b_prof_i = self._l_b_profile if i == 0 else self._l_b_profile_twin
                    p_beam_arr = l_b_prof_i * self._beam_cross[i] * nn
                    weights, _ = self._calc_beam_weights(i)
                    self._S_ion_beam += weights * p_beam_arr * self._n_beam[i] * self._v_beam[i] / self._L_plasma
            # NOTE: alpha_r and alpha_3 are approximate power-law fits used for both species.
            # For helium, replace with better species-specific recombination rates when available.
            self._S_rec_rad = self._b_rec_rad * ne * ne * alpha_r(Te, I=self._I_ion)
            self._S_rec_3b = self._b_rec_3b * ne * ne * ne * alpha_3(Te)

    def _nn_clausing_flux(self, nn):
        """
        Neutral diffusion via molecular flow conductance with Clausing factor.

        For each face between adjacent cells i and i+1:
          L_eff = (L_i + L_{i+1}) / 2
          k(L_eff/R) = 1 / (1 + 1.33 * L_eff / R)
          C_{i,j} = (1/4) * v_th_n * A * k(L_eff/R)

        Net volumetric rate for cell i:
          dn_i/dt = sum_j C_{i,j} * (n_j - n_i) / V_i
                  = (v_th_n / 4) * sum_j k_{i,j} * (n_j - n_i) / L_i

        The machine cross-section A cancels with V_i = A * L_i.
        """
        L = self._L_plasma
        R = self._R_machine[0]
        L_eff = (L[:-1] + L[1:]) / 2
        k_face = 1.0 / (1.0 + (3.0 / 8.0) * L_eff / R)
        kappa = 0.25 * self._v_th_n * k_face  # (cells-1,)
        delta_n = nn[1:] - nn[:-1]
        self._Nn_face_flux = -kappa * delta_n
        Nn_flux = np.zeros(self._cells)
        Nn_flux[:-1] -= self._Nn_face_flux / L[:-1]
        Nn_flux[1:]  += self._Nn_face_flux / L[1:]
        return Nn_flux

    def _dstep_nn(self, nn):
        """Derivative for nn-only integration when Plasma=False."""
        Nn_flux = self._nn_clausing_flux(nn)
        return Nn_flux + self._gas_puff_source() - self._S_pump * nn

    def _gas_puff_source(self, t=None):
        """Return the active gas-puff source vector for the current phase."""
        if self._flags["Plasma"]:
            scale = self._plasma_gas_puff_scale(self._t_current if t is None else t)
            if scale > 0.0:
                return scale * self._S_gp
            return np.zeros(self._cells)
        if getattr(self, "_gas_puff_on", getattr(self, "_discharge_on", False)):
            return self._S_gp
        return np.zeros(self._cells)

    def _plasma_gas_puff_on(self, phase, t):
        return self._plasma_gas_puff_scale(t, phase=phase) > 0.0

    def _plasma_gas_puff_scale(self, t, phase=None):
        phase = phase or getattr(self, "_phase", None)
        if phase in ("pre_breakdown", "breakdown"):
            return 1.0
        if phase != "main_discharge" or self._t_breakdown is None:
            return 0.0
        if self._tau_gp_after_breakdown is None:
            return 1.0

        t_rel = t - self._t_breakdown
        if t_rel <= self._tau_gp_after_breakdown:
            return 1.0

        tau_decay = self._tau_discharge - self._tau_gp_after_breakdown
        if tau_decay <= 0.0:
            return 1.0
        return float(np.exp(-(t_rel - self._tau_gp_after_breakdown) / tau_decay))

    def _sync_plasma_phase_switches(self, phase, t):
        self._phase = phase
        self._discharge_on = phase in ("pre_breakdown", "breakdown", "main_discharge")
        self._floating = phase == "afterglow"
        self._gas_puff_on = self._plasma_gas_puff_on(phase, t)

    def _gas_puff_event_time(self, phase):
        if (
            phase != "main_discharge"
            or self._t_breakdown is None
            or self._tau_gp_after_breakdown is None
        ):
            return None
        return self._t_breakdown + self._tau_gp_after_breakdown

    def _ne_flux_hybrid(self, ne, c_s, v_plasma):
        """
        Interior density flux from resolved flow plus directional acoustic relaxation.

        Positive Gamma_face[f] moves plasma from cell f to cell f+1.
        Sonic emission is limited by the stronger source cell, not the receiving cell.
        The sonic correction fades as adjacent cell densities equilibrate.
        """
        ne_face = 0.5 * (ne[:-1] + ne[1:])
        v_face = 0.5 * (v_plasma[:-1] + v_plasma[1:])

        Gamma_adv = ne_face * v_face
        Gamma_r = ne[:-1] * c_s[:-1]
        Gamma_l = ne[1:] * c_s[1:]
        Gamma_sonic_raw = Gamma_r - Gamma_l
        Gamma_sonic_max = self._beta_ne_sonic_flux * np.maximum(Gamma_r, Gamma_l)
        Gamma_sonic_corr = np.clip(
            Gamma_sonic_raw,
            -Gamma_sonic_max,
            Gamma_sonic_max,
        )

        taper = self._hybrid_ne_density_taper(ne, ne_face)

        Gamma_face = Gamma_adv + self._alpha_ne_sonic_flux * taper * Gamma_sonic_corr

        Ne_flux = np.zeros(self._cells)
        Ne_flux[:-1] -= Gamma_face / self._L_plasma[:-1]
        Ne_flux[1:] += Gamma_face / self._L_plasma[1:]
        return Ne_flux, Gamma_face

    def _hybrid_ne_density_taper(self, ne, ne_face):
        t_breakdown = getattr(self, "_t_breakdown", None)
        t_current = getattr(self, "_t_current", 0.0)
        if t_breakdown is None:
            t_since_breakdown = -np.inf
        else:
            t_since_breakdown = t_current - t_breakdown
        if t_since_breakdown < self._hybrid_ne_taper_delay:
            return 1.0

        rel_dn = np.abs(ne[:-1] - ne[1:]) / np.maximum(ne_face, 1e-300)
        dn0 = max(self._hybrid_ne_taper_dn0, 1e-300)
        taper = rel_dn / (rel_dn + dn0)
        return taper**self._hybrid_ne_taper_power

    def _calc_n_flux(self, Te, ne, nn, v_plasma, c_s=None):
        # TODO: use the cathode solver results to determine plasma loss in cathode cells
        Ne_flux = np.zeros(self._cells)
        Nn_flux = np.zeros(self._cells)
        self._Ne_face_flux = np.zeros(self._cells - 1)
        if self._flags["Plasma"]:
            # Boundary losses: cathode cells use sheath/cathode physics; right wall uses Bohm speed
            Ne_flux[0] = -(1 + 2 * self._eta) * self._cathode_result.I_i / qe_SI / self._plasma_vol[0]
            if self._flags["TwinCathode"]:
                Ne_flux[-1] = -(1 + 2 * self._eta) * self._cathode_result_twin.I_i / qe_SI / self._plasma_vol[-1]
            else:
                Ne_flux[-1] = -ne[-1] * c_s[-1] / self._L_plasma[-1]
            Nn_flux[0] = -Ne_flux[0] * self._Rsq_ratio[0]
            Nn_flux[-1] = -Ne_flux[-1] * self._Rsq_ratio[-1]
            # Interior face transport
            if self._flags.get("hybrid_ne"):
                Ne_flux_interior, self._Ne_face_flux = self._ne_flux_hybrid(
                    ne, c_s, v_plasma
                )
                Ne_flux += Ne_flux_interior
            elif not self._flags["Velocity"]:
                # Symmetric sound-speed flux when velocity equation is off
                Gamma_r = ne[:-1] * c_s[:-1]
                Gamma_l = ne[1:]  * c_s[1:]
                self._Ne_face_flux = Gamma_r - Gamma_l
                Ne_flux[:-1] -= self._Ne_face_flux / self._L_plasma[:-1]
                Ne_flux[1:]  += self._Ne_face_flux / self._L_plasma[1:]
            else:
                ne_face = (ne[:-1] + ne[1:]) / 2
                v_face = (v_plasma[:-1] + v_plasma[1:]) / 2
                self._Ne_face_flux = ne_face * v_face
                Ne_flux[:-1] -= self._Ne_face_flux / self._L_plasma[:-1]
                Ne_flux[1:]  += self._Ne_face_flux / self._L_plasma[1:]
        Nn_flux += self._nn_clausing_flux(nn)
        return Ne_flux, Nn_flux

    def _calc_pres_acc(self, ne, Te, Ti):
        """
        Plasma pressure gradient acceleration -(1/ρ) ∇P [cm/s²].

        Finite-volume approach: face pressures P_face = (P[i] + P[i+1]) / 2
        give a net pressure drop across each cell, consistent with the divergence
        discretization used in elec_par_heat_div and _calc_T_convection.
        Boundary cells use a zero-gradient (Neumann) condition at the wall face,
        so P_face_wall = P_boundary, leaving only the interior face contribution.
        """
        P = ne * (Te + self._ion_pressure_weight * Ti) * ev_to_erg  # erg/cm³
        L = self._L_plasma
        P_face = (P[:-1] + P[1:]) / 2  # face pressures, shape (cells-1,)
        pres_acc = np.zeros(self._cells)
        pres_acc[0]    = -(P_face[0]  - P[0])      / (L[0]    * self._m_gas * ne[0])
        pres_acc[-1]   = -(P[-1]      - P_face[-1]) / (L[-1]   * self._m_gas * ne[-1])
        pres_acc[1:-1] = -(P_face[1:] - P_face[:-1]) / (L[1:-1] * self._m_gas * ne[1:-1])
        return pres_acc

    def _calc_drag_in(self, Ti, nn, v_plasma, v_thm_i=None):
        """
        Calculate the ion-neutral drag force on the plasma per unit volume.
        This term is subtracted from the plasma velocity time derivative.
        """
        if v_thm_i is None:
            v_thm_i = v_ion_speed(Ti, self._mu)
        drag_in = drag_factor * v_thm_i * v_plasma
        return drag_in * nn

    def calc_heat_terms(
        self,
        ne,
        nn,
        Te,
        Ti,
        v_plasma,
        c_s=None,
    ):
        self._e_par_flux, self._i_par_flux = self._calc_cond_heat_flux(Te, Ti, ne)
        self._Qei = np.zeros(self._cells)
        self._Qen = np.zeros(self._cells)
        self._Qcx = np.zeros(self._cells)
        self._Qeb = np.zeros(self._cells)
        self._Qib = np.zeros(self._cells)
        self._div_v_elec = np.zeros(self._cells)
        self._div_v_ions = np.zeros(self._cells)
        self._Qie = (
            self._b_Qie * en_factor * Q_ie(Te, Ti, ne, self._mu, self._ln_lambda)
        )
        if self._flags["Velocity"]:
            self._v_face = (v_plasma[:-1] + v_plasma[1:]) / 2
            # NOTE: physical form of div(v) term should be revisited.
            div_v = self._calc_div_v(Te, c_s=c_s)
            self._div_v_elec = -en_factor * Te * div_v
            self._div_v_ions = -en_factor * Ti * div_v
            self._Te_conv = self._calc_T_convection(Te, v_plasma)
            self._Ti_conv = self._calc_T_convection(Ti, v_plasma)
        if self._flags["icool"]:
            if self._gas_type == "He":
                self._Qei = (
                    self._b_Qei
                    * en_factor
                    * IAEA_exp4(Te, aHeII, recomb=self._flags["icool_recomb"])
                    * ne
                )
            else:
                self._Qei = self._b_Qei * en_factor * IAEA_exp6(Te, aHII) * ne
        if self._flags["ncool"]:
            if self._gas_type == "He":
                self._Qen = self._b_Qen * en_factor * IAEA_exp1(Te, aHeI) * nn
            else:  # H
                self._Qen = self._b_Qen * en_factor * IAEA_exp1(Te, aHI) * nn
        if self._flags["cx"]:
            self._Qcx = (
                self._b_Qcx
                * en_factor
                * Q_cx_He(ne, nn, Ti, self._Tn_fit, gas_type=self._gas_type)
            )
        if self._discharge_on:
            cathode_results = {0: self._cathode_result}
            if self._flags["TwinCathode"]:
                cathode_results[-1] = self._cathode_result_twin
            for i in [0, -1] if self._flags["TwinCathode"] else [0]:
                cr = cathode_results[i]
                if self._beam_cross[i] == 0.0:
                    continue
                l_b_prof_i = self._l_b_profile if i == 0 else self._l_b_profile_twin
                p_beam_arr = l_b_prof_i * self._beam_cross[i] * nn
                weights, _ = self._calc_beam_weights(i)
                P_depo = cr.P_prim + cr.P_ohmic
                self._Qeb += weights * en_factor * (
                    P_depo / (self._plasma_vol * qe_SI * ne)
                    - p_beam_arr * self._n_beam[i] * self._v_beam[i] / self._L_plasma * self._I_ion / ne
                )
                P_loss_e = cr.P_cathode_e + cr.P_anode_e
                P_loss_i = cr.P_loss - P_loss_e
                self._Qeb[i] -= en_factor * P_loss_e / (self._plasma_vol[i] * qe_SI * ne[i])
                self._Qib[i] -= (1+2*self._eta)*Ti[i]*c_s[i]/self._L_plasma[i]

    def _calc_cond_heat_flux(self, Te, Ti, ne):
        e_par_hl = (
            self._b_epara
            * en_factor
            * elec_par_heat_loss(
                Te,
                ne,
                self._L_plasma,
                self._L_heatflux,
                self._ln_lambda,
            )
        )
        i_par_hl = (
            self._b_ipara
            * en_factor
            * ion_par_heat_loss(
                Ti,
                ne,
                self._L_plasma,
                self._L_heatflux,
                self._mu,
                self._ln_lambda,
            )
        )
        e_par_flux = np.zeros(self._cells)
        i_par_flux = np.zeros(self._cells)
        # if there is a cathode, then electron heat-loss is treated by the cathode solver
        # otherwise the heat loss is treated as normal endloss
        # TODO: ion-heat loss in cathode/no-cathode cases?
        if not self._flags["TwinCathode"]:
            e_par_flux[-1] -= e_par_hl[-1]
            i_par_flux[-1] -= i_par_hl[-1]
        self._e_par_face_flux = self._b_epara * en_factor * elec_par_heat_face_flux(
            Te, ne, self._L_plasma, self._ln_lambda
        )
        self._i_par_face_flux = self._b_ipara * en_factor * ion_par_heat_face_flux(
            Ti, ne, self._L_plasma, self._mu, self._ln_lambda
        )
        e_par_flux += self._b_epara * en_factor * elec_par_heat_div(
            Te, ne, self._L_plasma, self._ln_lambda
        )
        i_par_flux += self._b_ipara * en_factor * ion_par_heat_div(
            Ti, ne, self._L_plasma, self._mu, self._ln_lambda
        )
        return e_par_flux, i_par_flux

    def _calc_div_v(self, Te, c_s=None):
        """
        Calculate the divergence of the plasma velocity field (dv/dx) for each cell.
        Uses one-sided differences for end cells and a central difference for interior cells.
        NOTE: the correct discretization scheme should be revisited in a future pass.
        """
        div_v = np.zeros(self._cells)
        v_face = self._v_face
        div_v[0] = (v_face[0] + c_s[0]) / self._L_plasma[0]
        div_v[-1] = (c_s[-1] - v_face[-1]) / self._L_plasma[-1]
        # Interior cells: central difference of face velocities (vectorized)
        div_v[1:-1] = (v_face[1:] - v_face[:-1]) / self._L_plasma[1:-1]
        return div_v

    def _calc_advection(self, v_plasma, c_s):
        """
        v·∇v convective acceleration (1D, cell-centered).

        Boundary conditions mirror _calc_div_v, with Bohm-speed end flows at
        both outer faces:
          cell 0:  v_face_left = -c_s[0]
          cell -1: v_face_right = c_s[-1]
        """
        L = self._L_plasma
        v_face = self._v_face
        adv = np.empty(self._cells)

        # Interior: central difference of face velocities
        adv[1:-1] = v_plasma[1:-1] * (v_face[1:] - v_face[:-1]) / L[1:-1]

        # Cell 0
        adv[0] = v_plasma[0] * (v_face[0] + c_s[0]) / L[0]             # Bohm inflow
        adv[-1] = v_plasma[-1] * (c_s[-1] - v_face[-1]) / L[-1]       # Bohm outflow
        return adv

    def _calc_T_convection(self, T, v_plasma):
        """
        -v·∇T temperature convection term (Eulerian frame correction).
        Uses central-difference face interpolation with zero-gradient BCs at both ends.
        """
        L = self._L_plasma
        T_face = (T[:-1] + T[1:]) / 2  # length cells-1
        conv = np.empty(self._cells)
        conv[1:-1] = -v_plasma[1:-1] * (T_face[1:] - T_face[:-1]) / L[1:-1]
        conv[0] = -v_plasma[0] * (T_face[0] - T[0]) / L[0]
        conv[-1] = -v_plasma[-1] * (T[-1] - T_face[-1]) / L[-1]
        return conv

    def _raise_numerical_instability(self, event):
        self._debug_events.append(event)
        detail = ", ".join(f"{k}={v}" for k, v in event.items())
        raise NumericalInstabilityError(detail)

    def _check_finite_state(self, state, context):
        if not self._flags.get("debug_checks"):
            return
        bad = np.argwhere(~np.isfinite(state))
        if bad.size == 0:
            return
        comp, cell = bad[0]
        self._raise_numerical_instability(
            {
                "kind": "nonfinite_state",
                "context": context,
                "time_s": getattr(self, "_t_current", None),
                "h_s": getattr(self, "_h", None),
                "component": _STATE_NAMES[comp],
                "cell": int(cell),
                "value": state[comp, cell],
            }
        )

    def _check_accepted_step_jump(self, old_state, new_state):
        if getattr(self, "_t_current", 0.0) < self._debug_check_start_time:
            return
        if not self._flags.get("debug_checks") or not np.isfinite(
            self._debug_max_rel_step_change
        ):
            return
        max_rel, comp, cell = self._max_step_change(old_state, new_state)
        if max_rel > self._debug_max_rel_step_change:
            event = {
                "kind": "accepted_step_jump",
                "time_s": getattr(self, "_t_current", None),
                "h_s": getattr(self, "_h", None),
                "component": _STATE_NAMES[comp],
                "cell": int(cell),
                "relative_change": float(max_rel),
                "old": float(old_state[comp, cell]),
                "new": float(new_state[comp, cell]),
            }
            event.update(self._debug_rhs_terms(_STATE_NAMES[comp], int(cell)))
            self._raise_numerical_instability(
                event
            )

    def _max_step_change(self, old_state, new_state):
        scale = self._debug_step_atol + self._rtol * np.maximum(
            np.abs(old_state), np.abs(new_state)
        )
        rel_change = np.abs(new_state - old_state) / scale
        comp, cell = np.unravel_index(np.argmax(rel_change), rel_change.shape)
        return rel_change[comp, cell], comp, cell

    def _large_step_change_limited_h_next(self, h, old_state, new_state):
        if getattr(self, "_t_current", 0.0) < self._debug_check_start_time:
            return None
        if not np.isfinite(self._debug_max_rel_step_change):
            return None
        max_rel, _, _ = self._max_step_change(old_state, new_state)
        if max_rel <= self._debug_max_rel_step_change:
            return None
        shrink = min(0.5, max(0.05, 0.8 * self._debug_max_rel_step_change / max_rel))
        return max(self._h_min, h * shrink)

    def _debug_rhs_terms(self, component, cell):
        """Return the current RHS term decomposition for a debug event."""
        def arr(name):
            return getattr(self, name, np.zeros(self._cells))

        terms_by_component = {
            "ne": {
                "S_ion_bulk": arr("_S_ion_bulk"),
                "S_ion_beam": arr("_S_ion_beam"),
                "-S_rec_rad": -arr("_S_rec_rad"),
                "-S_rec_3b": -arr("_S_rec_3b"),
                "Ne_flux": arr("_Ne_flux"),
            },
            "nn": {
                "Nn_flux": arr("_Nn_flux"),
                "-S_ion_net*Rsq": -(
                    arr("_S_ion_bulk")
                    + arr("_S_ion_beam")
                    - arr("_S_rec_rad")
                    - arr("_S_rec_3b")
                )
                * self._Rsq_ratio,
                "S_gp": self._gas_puff_source(),
                "-S_pump*nn": -self._S_pump * self._nn,
            },
            "Te": {
                "Qeb": arr("_Qeb"),
                "-Qie": -arr("_Qie"),
                "-Qei": -arr("_Qei"),
                "-Qen": -arr("_Qen"),
                "e_par_flux": arr("_e_par_flux"),
                "div_v_elec": arr("_div_v_elec"),
                "Te_conv": arr("_Te_conv"),
            },
            "Ti": {
                "Qie": arr("_Qie"),
                "i_par_flux": arr("_i_par_flux"),
                "-Qcx": -arr("_Qcx"),
                "div_v_ions": arr("_div_v_ions"),
                "Qib": arr("_Qib"),
                "Ti_conv": arr("_Ti_conv"),
            },
        }
        terms = terms_by_component.get(component)
        if terms is None:
            return {}
        values = {name: float(np.asarray(arr)[cell]) for name, arr in terms.items()}
        dominant = max(values, key=lambda name: abs(values[name]))
        return {
            "rhs_sum": float(sum(values.values())),
            "dominant_term": dominant,
            "dominant_value": values[dominant],
            "rhs_terms": values,
        }

    def _check_neighbor_ratios(self, state):
        if getattr(self, "_t_current", 0.0) < self._debug_check_start_time:
            return
        if not self._flags.get("debug_checks") or not np.isfinite(
            self._debug_max_neighbor_ratio
        ):
            return
        for comp, name in enumerate(_STATE_NAMES[:4]):
            values = np.abs(state[comp])
            floor = self._atol[comp, 0]
            denom = np.maximum(np.minimum(values[:-1], values[1:]), floor)
            ratio = np.maximum(values[:-1], values[1:]) / denom
            if self._debug_ignore_floor_neighbors:
                near_floor = (values[:-1] <= floor) | (values[1:] <= floor)
                ratio = np.where(near_floor, 1.0, ratio)
            face = int(np.argmax(ratio))
            max_ratio = ratio[face]
            if max_ratio > self._debug_max_neighbor_ratio:
                self._raise_numerical_instability(
                    {
                        "kind": "neighbor_jump",
                        "time_s": getattr(self, "_t_current", None),
                        "h_s": getattr(self, "_h", None),
                        "component": name,
                        "left_cell": face,
                        "right_cell": face + 1,
                        "ratio": float(max_ratio),
                        "left": float(state[comp, face]),
                        "right": float(state[comp, face + 1]),
                    }
                )

    def _floor_violation(self, state):
        bad = np.argwhere(state < self._reject_floor)
        if bad.size == 0:
            return None
        comp, cell = bad[0]
        return {
            "component": _STATE_NAMES[comp],
            "cell": int(cell),
            "value": float(state[comp, cell]),
            "floor": float(self._reject_floor[comp, 0]),
        }

    def _floor_limited_h_next(self, h, old_state, new_state):
        violation = self._floor_violation(new_state)
        if violation is None:
            return None
        comp = _STATE_NAMES.index(violation["component"])
        cell = violation["cell"]
        floor = violation["floor"]
        old = old_state[comp, cell]
        new = new_state[comp, cell]
        frac_to_floor = (old - floor) / max(old - new, 1e-300)
        shrink = min(0.5, max(0.05, 0.8 * frac_to_floor))
        return max(self._h_min, h * shrink)

    def _apply_state_guards(self, ne, nn, Te, Ti, v_plasma):
        if self._flags.get("debug_checks") and self._flags.get("debug_raise_on_guard"):
            before = np.array([ne.copy(), nn.copy(), Te.copy(), Ti.copy(), v_plasma.copy()])
        else:
            before = None
        np.clip(Ti, self._state_floor[3, 0], 100, out=Ti)
        np.clip(Te, self._state_floor[2, 0], 100, out=Te)
        np.maximum(ne, self._state_floor[0, 0], out=ne)
        np.maximum(nn, self._state_floor[1, 0], out=nn)
        if self._flags["Velocity"]:
            c_s_max = 10 * np.max(v_ion_speed(Te, self._mu))
            np.clip(v_plasma, -c_s_max, c_s_max, out=v_plasma)
        if before is not None:
            after = np.array([ne, nn, Te, Ti, v_plasma])
            changed = np.argwhere(after != before)
            if changed.size:
                comp, cell = changed[0]
                self._raise_numerical_instability(
                    {
                        "kind": "state_guard_triggered",
                        "time_s": getattr(self, "_t_current", None),
                        "h_s": getattr(self, "_h", None),
                        "component": _STATE_NAMES[comp],
                        "cell": int(cell),
                        "before": float(before[comp, cell]),
                        "after": float(after[comp, cell]),
                    }
                )

    def _calc_cathode(self, Te, ne, nn):
        beam_result = solve_beam_system(
            config=self._device_config,
            Te=Te,
            ne=ne,
            nn=nn,
            beam_cross_prev=self._beam_cross,
            plasma_cross=self._plasma_cross,
            I_ion=self._I_ion,
            gas_type=self._gas_type,
            x0=self._cathode_x0,
            x0_twin=self._cathode_x0_twin,
            floating=self._floating,
        )
        self._cathode_result = beam_result.result
        self._cathode_result_twin = beam_result.result_twin
        self._cathode_x0 = beam_result.x0_next
        self._cathode_x0_twin = beam_result.x0_twin_next
        self._v_beam = beam_result.v_beam
        self._n_beam = beam_result.n_beam
        self._beam_cross = beam_result.beam_cross
        self._n_beam_ion = beam_result.n_beam_ion
        self._A_ion_beam = beam_result.A_ion_beam
        self._l_b = beam_result.l_b
        self._p_beam = beam_result.p_beam
        self._l_b_profile = beam_result.l_b_profile
        self._l_b_profile_twin = beam_result.l_b_profile_twin

    def _calc_beam_weights(self, cathode_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Beer-Lambert weights and beam density profile for the beam from cathode_idx.

        Returns
        -------
        weights : ndarray, shape (cells,)
            Fraction of beam absorbed in cell j.
        n_beam_profile : ndarray, shape (cells,)
            Beam electron density at the entry of each cell [cm⁻³],
            equal to n_beam[cathode_idx] * exp(-cumulative optical depth to cell j).
        """
        if cathode_idx == 0:
            order = np.arange(self._cells)
            l_b_prof = self._l_b_profile
            dx = self._L_plasma
        else:
            order = np.arange(self._cells - 1, -1, -1)
            l_b_prof = self._l_b_profile_twin
            dx = self._L_plasma[::-1]

        l_b_ordered = l_b_prof[order]
        dx_ordered = dx[order]
        safe_l_b = np.where(l_b_ordered > 0, l_b_ordered, np.inf)
        tau = np.cumsum(dx_ordered / safe_l_b)
        tau_in = np.concatenate([[0.0], tau[:-1]])
        exp_neg_tau_in = np.exp(-tau_in)
        f_ordered = exp_neg_tau_in * (1.0 - np.exp(-dx_ordered / safe_l_b))

        weights = np.zeros(self._cells)
        weights[order] = f_ordered
        n_beam_profile = np.zeros(self._cells)
        n_beam_profile[order] = self._n_beam[cathode_idx] * exp_neg_tau_in
        return weights, n_beam_profile

    def _resize_mesh(self, new_n_interior):
        """Interpolate interior cells to new_n_interior count; cathode cells stay fixed."""
        old_n_interior = self._n_interior
        new_total = new_n_interior + self._n_cathode_cells

        x_old = (np.arange(old_n_interior) + 0.5) / old_n_interior
        x_new = (np.arange(new_n_interior) + 0.5) / new_n_interior
        interior_slice = slice(1, -1)  # both ends are always fixed boundary cells

        def _resize_arr(arr):
            interior = np.interp(x_new, x_old, arr[interior_slice])
            return np.concatenate([[arr[0]], interior, [arr[-1]]])

        self._ne = _resize_arr(self._ne)
        self._nn = _resize_arr(self._nn)
        self._Te = _resize_arr(self._Te)
        self._Ti = _resize_arr(self._Ti)
        self._v_plasma = _resize_arr(self._v_plasma)
        self._n_interior = new_n_interior
        self._cells = new_total
        self._L_plasma = self._build_L_plasma(new_n_interior)
        self._L_cell = self._build_L_cell(new_n_interior)
        self._L_heatflux = self._L_plasma / 2
        self._R_machine = np.ones(new_total) * self._input_dict["Rm"]
        self._R_plasma = np.ones(new_total) * self._input_dict["Rp"]
        self._Rsq_ratio = (self._R_plasma / self._R_machine) ** 2
        self._plasma_cross = np.pi * self._R_plasma**2
        self._plasma_vol = self._plasma_cross * self._L_plasma
        self._cell_vol = np.pi * self._R_machine**2 * self._L_cell
        self._S_gp = np.zeros(new_total)
        self._S_pump = np.zeros(new_total)
        self._S_gp[0] = self.puff_rate(self._input_dict["S_gp"], 2, self._cell_vol[0])
        self._S_pump[0] = self.pump_rate(
            self._input_dict["S_pump_L"], self._cell_vol[0]
        )
        self._S_pump[-1] = self.pump_rate(
            self._input_dict["S_pump_R"], self._cell_vol[-1]
        )
        if self._flags["TwinCathode"]:
            self._S_gp[-1] = self.puff_rate(
                self._input_dict["Twin_S_gp"], 2, self._cell_vol[-1]
            )
        self._div_v_elec = np.zeros(new_total)
        self._div_v_ions = np.zeros(new_total)
        self._Te_conv = np.zeros(new_total)
        self._Ti_conv = np.zeros(new_total)
        self._v_face = np.zeros(new_total - 1)
        self._e_par_face_flux = np.zeros(new_total - 1)
        self._i_par_face_flux = np.zeros(new_total - 1)
        self._Ne_face_flux = np.zeros(new_total - 1)
        self._Nn_face_flux = np.zeros(new_total - 1)

    def _check_mesh(self, t):
        """Refine or coarsen the spatial mesh based on the bulk electron MFP criterion."""
        if not self._flags.get("adaptive_mesh", True):
            return False
        tau_e = time_elec_coll(self._Te, self._ne, self._ln_lambda)
        bulk_mfp = v_thm_e(self._Te) * tau_e / self._L_plasma
        _max_interior = self._max_cells - self._n_cathode_cells
        if np.min(bulk_mfp) < self._mfp_refine_thresh and self._n_interior < _max_interior:
            new_n_interior = _to_odd(min(self._n_interior * 2, _max_interior))
            self._refinement_events.append((t, self._cells, new_n_interior + self._n_cathode_cells))
            self._resize_mesh(new_n_interior)
            return True
        if (
            np.min(bulk_mfp) > self._mfp_coarsen_thresh
            and self._n_interior > self._min_cells
        ):
            new_n_interior = _to_odd(max(self._n_interior // 2, self._min_cells))
            self._refinement_events.append((t, self._cells, new_n_interior + self._n_cathode_cells))
            self._resize_mesh(new_n_interior)
            return True
        return False

    def _compute_h_max_physical(self, interior_only=False):
        """Step ceiling from diffusive and advective CFL conditions on electron heat conduction.

        interior_only: exclude cathode boundary cells from the kappa_e CFL. Use during
        pre-breakdown when those cells are permanently floor-clamped and their kappa_e
        is an artifact rather than a dynamics constraint.
        """
        if interior_only:
            nc = self._n_cathode_cells
            sl = slice(nc, -nc if nc else None)
            ne = self._ne[sl]
            Te = self._Te[sl]
            ln_lam = self._ln_lambda[sl] if np.ndim(self._ln_lambda) > 0 else self._ln_lambda
        else:
            ne, Te, ln_lam = self._ne, self._Te, self._ln_lambda
        ne_for_cfl = ne if self._ne_cfl_floor is None else np.maximum(ne, self._ne_cfl_floor)
        kappa_e = 3.16 * time_elec_coll(Te, ne_for_cfl, ln_lam) * v_thm_e(Te) ** 2
        h_cond = 0.5 * np.min(self._L_plasma) ** 2 / np.max(kappa_e)
        c_s = v_ion_speed(self._Te, self._mu)
        v_eff = np.max(c_s + np.abs(self._v_plasma))
        h_advect = 0.5 * np.min(self._L_plasma) / v_eff
        return min(h_cond, h_advect)

    def _compute_h_max_nn(self):
        """Step ceiling from CFL condition on neutral Clausing diffusion."""
        return 0.5 * np.min(self._L_plasma) / (0.25 * self._v_th_n)

    def _dstep(self, a, t=None):
        ne, nn, Te, Ti, v_plasma = a
        zeros = np.zeros(self._cells)
        rhs_t = self._t_current if t is None else t

        if not self._flags["Plasma"]:
            _, self._Nn_flux = self._calc_n_flux(Te, ne, nn, v_plasma)
            d_nn = (
                self._Nn_flux
                + self._gas_puff_source()
                - self._S_pump * nn
            )
            return np.array([zeros, d_nn, zeros, zeros, zeros])

        # Lightweight floor for numerical stability at intermediate RK stages.
        # Creates new arrays (does not mutate the input state vector).
        # Full _apply_state_guards (including velocity clip) runs only at the
        # accepted endpoint in _rkf45_step / _rk4_step.
        Te = np.maximum(Te, self._state_floor[2, 0])
        Ti = np.maximum(Ti, self._state_floor[3, 0])
        ne = np.maximum(ne, self._state_floor[0, 0])
        nn = np.maximum(nn, self._state_floor[1, 0])

        # self._ln_lambda, cathode state, and beam quantities are frozen from
        # the last accepted step and updated once per accepted step.
        c_s = v_ion_speed(Te, self._mu)
        v_thm_i = v_ion_speed(Ti, self._mu)
        self.calc_density_terms(ne, nn, Te, v_plasma, c_s=c_s)
        self.calc_heat_terms(ne, nn, Te, Ti, v_plasma, c_s=c_s)

        d_ne = (
            self._S_ion_bulk
            + self._S_ion_beam
            - self._S_rec_rad
            - self._S_rec_3b
            + self._Ne_flux
        )
        d_Te = (
            self._Qeb
            - self._Qie
            - self._Qei
            - self._Qen
            + self._e_par_flux
            + self._div_v_elec
            + self._Te_conv
        )
        d_Ti = (
            self._Qie
            + self._i_par_flux
            - self._Qcx
            + self._div_v_ions
            + self._Qib
            + self._Ti_conv
        )
        d_nn = (
            self._Nn_flux
            - (self._S_ion_bulk + self._S_ion_beam - self._S_rec_rad - self._S_rec_3b)
            * self._Rsq_ratio
            + self._gas_puff_source(rhs_t)
            - self._S_pump * nn
        )
        if self._flags["Velocity"]:
            d_ve = (
                self._calc_pres_acc(ne, Te, Ti)
                - self._calc_drag_in(Ti, nn, v_plasma, v_thm_i=v_thm_i)
                - (self._calc_advection(v_plasma, c_s) if self._flags["advection"] else 0)
            )
        else:
            d_ve = zeros
        return np.array([d_ne, d_nn, d_Te, d_Ti, d_ve])

    def _rkf45_step(self, a):
        """
        One Dormand-Prince RK45 adaptive step.

        Uses ``self._h`` as the trial step size.  Returns
        ``((ne, nn, Te, Ti, v_plasma), h_next, accepted)``.
        Guardrails and diagnostics are stored only on accept.
        The caller is responsible for retrying with the returned ``h_next``
        when ``accepted`` is False.
        """
        if not self._flags["Plasma"]:
            return self._rkf45_step_nn(a)

        # Initialize cathode/ln_lambda on the very first call (self._cathode_result
        # is None until the first accepted step sets it).
        if self._cathode_result is None:
            ne0, nn0, Te0, Ti0, _ = a
            self._ln_lambda = c_log(Te0, ne0)
            self._calc_cathode(Te0, ne0, nn0)

        h = self._h

        # ── Dormand-Prince stages ──────────────────────────────────────────
        t0 = self._t_current
        k1 = self._dstep(a, t0)
        k2 = self._dstep(a + h * (1 / 5 * k1), t0 + h * (1 / 5))
        k3 = self._dstep(a + h * (3 / 40 * k1 + 9 / 40 * k2), t0 + h * (3 / 10))
        k4 = self._dstep(a + h * (44 / 45 * k1 - 56 / 15 * k2 + 32 / 9 * k3), t0 + h * (4 / 5))
        k5 = self._dstep(
            a
            + h
            * (
                19372 / 6561 * k1
                - 25360 / 2187 * k2
                + 64448 / 6561 * k3
                - 212 / 729 * k4
            ),
            t0 + h * (8 / 9),
        )
        k6 = self._dstep(
            a
            + h
            * (
                9017 / 3168 * k1
                - 355 / 33 * k2
                + 46732 / 5247 * k3
                + 49 / 176 * k4
                - 5103 / 18656 * k5
            ),
            t0 + h,
        )

        # ── 5th-order solution ─────────────────────────────────────────────
        y5 = a + h * (
            35 / 384 * k1
            + 500 / 1113 * k3
            + 125 / 192 * k4
            - 2187 / 6784 * k5
            + 11 / 84 * k6
        )

        # ── k7 for error estimate (FSAL) ───────────────────────────────────
        k7 = self._dstep(y5, t0 + h)

        # ── Error vector (difference between 5th and 4th order) ───────────
        err = h * (
            71 / 57600 * k1
            - 71 / 16695 * k3
            + 71 / 1920 * k4
            - 17253 / 339200 * k5
            + 22 / 525 * k6
            - 1 / 40 * k7
        )

        # ── Mixed-tolerance error norm ─────────────────────────────────────
        scale = self._atol + self._rtol * np.abs(a)
        err_norm = np.sqrt(np.mean((err / scale) ** 2))

        # ── Step size control (I-controller, factor clamped to [0.2, 5]) ──
        if not np.isfinite(err_norm) or err_norm == 0.0:
            # NaN/inf means intermediate stages overflowed; shrink aggressively
            factor = 0.2 if not np.isfinite(err_norm) else 5.0
        else:
            factor = min(5.0, max(0.2, 0.9 * err_norm ** (-0.2)))
        h_next = h * factor

        accepted = (np.isfinite(err_norm) and err_norm <= 1.0) or h <= self._h_min

        if accepted:
            if self._flags.get("reject_floor_violations", True) and h > self._h_min:
                h_floor = self._floor_limited_h_next(h, a, y5)
                if h_floor is not None:
                    return None, h_floor, False
            if self._flags.get("reject_large_step_changes", False) and h > self._h_min:
                h_jump = self._large_step_change_limited_h_next(h, a, y5)
                if h_jump is not None:
                    return None, h_jump, False
            ne, nn, Te, Ti, v_plasma = y5
            self._check_finite_state(y5, "accepted_endpoint_before_guards")
            self._check_accepted_step_jump(a, y5)
            self._apply_state_guards(ne, nn, Te, Ti, v_plasma)
            self._check_neighbor_ratios(np.array([ne, nn, Te, Ti, v_plasma]))
            self._ln_lambda = c_log(Te, ne)
            self._calc_cathode(Te, ne, nn)
            c_s = v_ion_speed(Te, self._mu)
            self._atol[4, 0] = self._v_atol_cs_fraction * np.max(c_s)
            self.calc_density_terms(ne, nn, Te, v_plasma, c_s=c_s)
            self.calc_heat_terms(ne, nn, Te, Ti, v_plasma, c_s=c_s)
            return (ne, nn, Te, Ti, v_plasma), h_next, True
        else:
            return None, h_next, False

    def _rkf45_step_nn(self, a):
        """Dormand-Prince RK45 adaptive step on nn only (Plasma=False)."""
        ne, nn, Te, Ti, v_plasma = a
        h = self._h

        # ── Dormand-Prince stages on nn ────────────────────────────────────
        k1 = self._dstep_nn(nn)
        k2 = self._dstep_nn(nn + h * (1 / 5 * k1))
        k3 = self._dstep_nn(nn + h * (3 / 40 * k1 + 9 / 40 * k2))
        k4 = self._dstep_nn(nn + h * (44 / 45 * k1 - 56 / 15 * k2 + 32 / 9 * k3))
        k5 = self._dstep_nn(
            nn
            + h
            * (
                19372 / 6561 * k1
                - 25360 / 2187 * k2
                + 64448 / 6561 * k3
                - 212 / 729 * k4
            )
        )
        k6 = self._dstep_nn(
            nn
            + h
            * (
                9017 / 3168 * k1
                - 355 / 33 * k2
                + 46732 / 5247 * k3
                + 49 / 176 * k4
                - 5103 / 18656 * k5
            )
        )

        # ── 5th-order solution ─────────────────────────────────────────────
        nn5 = nn + h * (
            35 / 384 * k1
            + 500 / 1113 * k3
            + 125 / 192 * k4
            - 2187 / 6784 * k5
            + 11 / 84 * k6
        )

        # ── k7 for error estimate (FSAL) ───────────────────────────────────
        k7 = self._dstep_nn(nn5)

        # ── Error vector ──────────────────────────────────────────────────
        err = h * (
            71 / 57600 * k1
            - 71 / 16695 * k3
            + 71 / 1920 * k4
            - 17253 / 339200 * k5
            + 22 / 525 * k6
            - 1 / 40 * k7
        )

        # ── Mixed-tolerance error norm (nn atol is index 1) ───────────────
        scale = self._atol[1] + self._rtol * np.abs(nn)
        err_norm = np.sqrt(np.mean((err / scale) ** 2))

        if not np.isfinite(err_norm) or err_norm == 0.0:
            factor = 0.2 if not np.isfinite(err_norm) else 5.0
        else:
            factor = min(5.0, max(0.2, 0.9 * err_norm ** (-0.2)))
        h_next = h * factor
        accepted = (np.isfinite(err_norm) and err_norm <= 1.0) or h <= self._h_min

        if accepted:
            state = np.array([ne, nn5, Te, Ti, v_plasma])
            if self._flags.get("reject_floor_violations", True) and h > self._h_min:
                h_floor = self._floor_limited_h_next(h, a, state)
                if h_floor is not None:
                    return None, h_floor, False
            if self._flags.get("reject_large_step_changes", False) and h > self._h_min:
                h_jump = self._large_step_change_limited_h_next(h, a, state)
                if h_jump is not None:
                    return None, h_jump, False
            if self._flags.get("debug_checks"):
                self._check_finite_state(state, "accepted_nn_endpoint")
                self._check_accepted_step_jump(a, state)
                self._check_neighbor_ratios(state)
            self.calc_density_terms(ne, nn5, Te, v_plasma)
            return (ne, nn5, Te, Ti, v_plasma), h_next, True
        else:
            return None, h_next, False

    def start_simulation(self):
        if self._flags["Plasma"] and self._tau_prebreakdown <= 0:
            raise ValueError(
                f"tau_prebreakdown must be > 0 (got {self._tau_prebreakdown})"
            )
        self.initialize_results()
        print("Starting simulation...")
        if self._flags["Plasma"]:
            self._run_plasma_phases()
        else:
            self._run_equilibrium_cycles()
        self._finalize_results()
        print("Simulation complete.")

    def _run_plasma_phases(self):
        """
        Adaptive integration for Plasma=True.

        pre_breakdown: floor-dominated, relaxed h_min, ends at I_prebreakdown (if set)
        breakdown:     plasma forming, discharge h_min, ends when I_tot >= I_breakdown (1 kA = t=0)
        main_discharge: cathode on, S_gp optionally decays after breakdown
        afterglow:      cathode + S_gp off, lasts tau_afterglow

        If I_prebreakdown=0 (default), pre_breakdown transitions directly to main_discharge
        at I_breakdown, preserving the original three-phase behaviour.
        """
        phase = "pre_breakdown"
        t = 0.0
        self._t_breakdown = None
        self._h = self._h0
        # h_min is a single value now; no per-phase distinction
        self._sync_plasma_phase_switches(phase, t)
        step_count = 0
        _last_print_t = -1e-3

        while True:
            self._sync_plasma_phase_switches(phase, t)
            if phase in ("pre_breakdown", "breakdown"):
                t_phase_end = self._tau_prebreakdown
                h_max_base = self._h_max_discharge
            elif phase == "main_discharge":
                t_phase_end = self._t_breakdown + self._tau_discharge
                h_max_base = self._h_max_discharge
            else:  # afterglow
                t_phase_end = self._t_breakdown + self._tau_discharge + self._tau_afterglow
                h_max_base = self._h_max_afterglow

            remaining = t_phase_end - t
            if remaining <= 0:
                break
            if self._cathode_result is not None:
                h_max_base = min(h_max_base, self._compute_h_max_physical(interior_only=(phase == "pre_breakdown")))
            h_cap = min(h_max_base, remaining)
            gas_puff_event_t = self._gas_puff_event_time(phase)
            if (
                gas_puff_event_t is not None
                and self._gas_puff_on
                and t < gas_puff_event_t < t_phase_end
            ):
                h_cap = min(h_cap, gas_puff_event_t - t)
            self._h = min(self._h, h_cap)

            a = np.array([self._ne, self._nn, self._Te, self._Ti, self._v_plasma])
            self._t_current = t
            result, h_next, accepted = self._rkf45_step(a)

            if accepted:
                t_before = t
                gas_puff_was_on = self._gas_puff_on
                self._ne, self._nn, self._Te, self._Ti, self._v_plasma = result
                t += self._h
                self._sync_plasma_phase_switches(phase, t)
                gas_puff_event_t = self._gas_puff_event_time(phase)
                if (
                    gas_puff_was_on
                    and phase == "main_discharge"
                    and gas_puff_event_t is not None
                    and gas_puff_event_t < self._t_breakdown + self._tau_discharge
                    and t_before <= gas_puff_event_t <= t
                    and not self._gas_puff_shutoff_reported
                ):
                    self._gas_puff_shutoff_reported = True
                    print(
                        f"  S_gp exponential decay started at "
                        f"t={(gas_puff_event_t - self._t_breakdown)*1e3:.3f} ms "
                        "after breakdown."
                    )
                self._h = max(min(self._h_min, h_cap), min(h_next, h_cap))
                step_count += 1

                self.update_results(t)

                if self._flags.get("adaptive_mesh"):
                    if self._check_mesh(t):
                        self._h = min(self._h, self._compute_h_max_physical(interior_only=(phase == "pre_breakdown")))

                if t - _last_print_t >= 1e-3:
                    _last_print_t = t
                    t_rel = t if self._t_breakdown is None else t - self._t_breakdown
                    print(f"  [{phase}] t={t_rel*1e3:.3f} ms  h={self._h:.2e} s  steps={step_count}")
                    print(f"  ne={self._ne}  nn={self._nn}")
                    print(f"  Te={self._Te}  Ti={self._Ti}")
                    if self._progress_callback is not None:
                        wall_now = time.time()
                        if self._last_cb_wall_t is not None:
                            seg_wall = wall_now - self._last_cb_wall_t
                            self._rate_ema = 0.2 * seg_wall + 0.8 * self._rate_ema if self._rate_ema else seg_wall
                        else:
                            seg_wall = 0.0
                        self._last_cb_wall_t = wall_now
                        self._progress_callback(min(t / self._t_total, 0.99), phase, seg_wall, self._rate_ema)

                # Phase transition checks
                if phase == "pre_breakdown":
                    I_now = self._cathode_result.I_tot if self._cathode_result is not None else 0.0
                    first_thresh = self._I_prebreakdown if self._I_prebreakdown > 0 else self._I_breakdown
                    if I_now >= first_thresh:

                        if self._I_prebreakdown > 0:
                            phase = "breakdown"
                            print(f"  Pre-breakdown threshold at t={t*1e3:.3f} ms, "
                                  f"I_tot={I_now:.1f} A — entering breakdown phase")
                        else:
                            self._t_breakdown = t
                            phase = "main_discharge"
                            print(f"  Breakdown at t={t*1e3:.3f} ms, I_tot={I_now:.1f} A")
                    elif t >= self._tau_prebreakdown:
                        raise BreakdownError(
                            f"Plasma failed to break down within tau_prebreakdown="
                            f"{self._tau_prebreakdown * 1e3:.1f} ms "
                            f"(I_tot={I_now:.1f} A < I_breakdown={self._I_breakdown:.1f} A)"
                        )
                elif phase == "breakdown":
                    I_now = self._cathode_result.I_tot if self._cathode_result is not None else 0.0
                    if I_now >= self._I_breakdown:
                        self._t_breakdown = t
                        phase = "main_discharge"
                        print(f"  Breakdown (1 kA) at t={t*1e3:.3f} ms, I_tot={I_now:.1f} A")
                    elif t >= self._tau_prebreakdown:
                        raise BreakdownError(
                            f"Plasma failed to reach {self._I_breakdown:.0f} A within tau_prebreakdown="
                            f"{self._tau_prebreakdown * 1e3:.1f} ms "
                            f"(I_tot={I_now:.1f} A)"
                        )
                elif phase == "main_discharge":
                    if t >= self._t_breakdown + self._tau_discharge:
                        phase = "afterglow"
                        self._sync_plasma_phase_switches(phase, t)
                        self._h = min(self._h0, self._h_max_afterglow)
                        print(f"  Main discharge ended at t={(t - self._t_breakdown)*1e3:.3f} ms. "
                              f"Entering afterglow.")
                else:  # afterglow
                    if t >= self._t_breakdown + self._tau_discharge + self._tau_afterglow:
                        break
            else:
                self._h = max(min(self._h_min, h_cap), min(h_next, h_cap))

    def _run_equilibrium_cycles(self):
        """
        Plasma=False equilibrium integration.

        puffing phase: S_gp on for tau_discharge
        off phase: S_gp off for the remainder of tau_cycle
        Repeats for self._cycles cycles.
        """
        h0_nn = self._compute_h_max_nn()
        for j in range(self._cycles):
            print(f"Starting cycle {j+1}/{self._cycles}...")
            t = 0.0
            self._h = min(self._h0, h0_nn)
            t_offset = j * self._tau_cycle

            while t < self._tau_cycle * (1 - 1e-12):
                in_puffing = t < self._tau_discharge
                self._discharge_on = in_puffing
                self._gas_puff_on = in_puffing
                self._floating = True

                t_phase_end = self._tau_discharge if in_puffing else self._tau_cycle
                h_max_base = self._h_max_discharge if in_puffing else self._h_max_afterglow
                h_cap = min(h_max_base, t_phase_end - t)
                self._h = min(self._h, h_cap)

                a = np.array([self._ne, self._nn, self._Te, self._Ti, self._v_plasma])
                self._t_current = t
                result, h_next, accepted = self._rkf45_step(a)

                if accepted:
                    self._ne, self._nn, self._Te, self._Ti, self._v_plasma = result
                    t += self._h
                    self._h = max(min(self._h_min, h_cap), min(h_next, h_cap))
                    self.update_results(t + t_offset)
                else:
                    self._h = max(min(self._h_min, h_cap), min(h_next, h_cap))

    def get_results(self):
        def _cathode_ns(arr):
            return SimpleNamespace(
                **{f: arr[:, i] for i, f in enumerate(_CATHODE_FIELDS)}
            )

        t_breakdown_ms = (self._t_breakdown * 1e3) if getattr(self, "_t_breakdown", None) is not None else None
        return SimpleNamespace(
            time=self._time * 1e3,
            t_breakdown=t_breakdown_ms,
            ne=self._densities[:, 0],
            nn=self._densities[:, 1],
            n_beam=self._densities[:, 2],
            Te=self._temperatures[:, 0],
            Ti=self._temperatures[:, 1],
            Ne_flux=self._density_terms[:, 0],
            Nn_flux=self._density_terms[:, 1],
            Ne_face_flux=self._face_fluxes[:, 0],
            Nn_face_flux=self._face_fluxes[:, 1],
            S_ion_bulk=self._density_terms[:, 2],
            S_rec_rad=self._density_terms[:, 3],
            S_rec_3b=self._density_terms[:, 4],
            S_ion_beam=self._density_terms[:, 5],
            e_par_flux=self._heat_terms[:, 0],
            i_par_flux=self._heat_terms[:, 1],
            e_par_face_flux=self._face_fluxes[:, 2],
            i_par_face_flux=self._face_fluxes[:, 3],
            Qie=self._heat_terms[:, 2],
            Qei=self._heat_terms[:, 3],
            Qen=self._heat_terms[:, 4],
            Qcx=self._heat_terms[:, 5],
            Qeb=self._heat_terms[:, 6],
            div_v_elec=self._heat_terms[:, 7],
            div_v_ions=self._heat_terms[:, 8],
            Qib=self._heat_terms[:, 9],
            Te_conv=self._heat_terms[:, 10],
            Ti_conv=self._heat_terms[:, 11],
            v_plasma=self._velocities[:, 0],
            isat=self._synthetic,
            primary_mfp=self._primary_mfp,
            bulk_mfp=self._bulk_mfp,
            ln_lambda=self._ln_lambda,
            cells_at_time=self._cells_at_time,
            refinement_events=self._refinement_events,
            debug_events=self._debug_events,
            cathode=_cathode_ns(self._cathode),
            cathode_twin=_cathode_ns(self._cathode_twin),
        )

    def puff_rate(self, sccm, valves, chamber_vol):
        return 4.477962e17 * sccm * valves / chamber_vol

    def pump_rate(self, lps, chamber_vol):
        """
        Calculate the pump rate in s^1 from the given lps and chamber volume.

        Parameters
        ----------
        lps : float
            The pump speed in liters per second (lps).
        chamber_vol : float
            The volume of the chamber in cm^3.

        Returns
        -------
        float
            The calculated pump rate in cm^-3/s.
        """
        return lps * 1e3 / chamber_vol

    def get_config(self):
        """Return copies of the input dictionary and flags used for this simulation."""
        return dict(self._input_dict), dict(self._flags)
