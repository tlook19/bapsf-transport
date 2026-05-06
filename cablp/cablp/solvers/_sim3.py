import math
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
    elec_perp_heat_loss,
    ion_perp_heat_loss,
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
)
_CATHODE_NAN = np.full(len(_CATHODE_FIELDS), np.nan)


def _cathode_to_array(result):
    if result is None:
        return _CATHODE_NAN.copy()
    return np.array([getattr(result, f) for f in _CATHODE_FIELDS])


input_dict_template = {
    "gas_type": "He",
    "ne0": 1e9,
    "Tn_fit": 0.1,  # Neutral temperature for reaction rate fits
    "Te0": 0.1,
    "Ti0": 0.1,
    "Bz0": 1500,  # Magnetic field in gauss
    "Lm": 1800,  # Length of machine
    "Rm": 50,  # Machine radius
    "Lp": 1800,  # Length of plasma
    "Rp": 18,  # Plasma radius
    "Rhf": 50,  # Gradient scale length of radial heat flux
    # Cathode device parameters (used by cathode solver)
    "V_bank": 100,  # Power supply voltage [V]
    "T_s": 1900.0,  # Cathode surface temperature [K]
    "phi_wf": 3.0,  # Work function [eV] (LaB6 default)
    "C_R": 29.0,  # Richardson constant [A cm⁻² K⁻²]
    "R_comp": 0.004,  # Compliance resistor [Ω]
    "eta": 0.358,  # Anode area / cathode area
    "L_cath": 50.0,  # Cathode-to-anode distance [cm]
    "R_cath": 18.0,  # Cathode radius [cm]; A_c = π * R_cath²
    "S_gp": 500,  # Gas puff source rate
    "Twin_S_gp": 500,
    "gp_puff_factor": 1.0,  # Initial gas puff multiplier (S_gp * gp_puff_factor for t < tau_gp_ramp)
    "tau_gp_ramp": 0.0,  # Duration of initial boosted puff [s]; 0 = no boost
    "S_pump_L": 4000,  # Vacuum pump sink rate
    "S_pump_R": 4000,
    "tau_I_on": 0.001,  # Time constant for beam current rise
    "b_epara": 1.0,  # Scaling factor for e_para transport
    "b_ipara": 1.0,  # Scaling factor for i_para transport
    "b_eperp": 1.0,  # Scaling factor for e_perp transport
    "b_iperp": 1.0,  # Scaling factor for i_perp transport
    "b_ioniz": 1.0,  # Scaling factor for ionization
    "b_rec_rad": 1.0,  # Scaling factor for radiative recombination
    "b_rec_3b": 1.0,  # Scaling factor for three-body recombination
    "b_Qcx": 1.0,  # Scaling factor for charge exchange
    "b_source": 1.0,  # Scaling factor for source heating
    "b_Qie": 1.0,  # Scaling factor for ion-electron heating
    "b_Qei": 1.0,  # Scaling factor for electron-ion cooling
    "b_Qen": 1.0,  # Scaling factor for electron-neutral cooling
    "cycles": 1,
    "tau_prebreakdown": 0.1,   # max pre-breakdown phase duration [s]
    "tau_discharge": 15e-3,    # main discharge duration after breakdown [s]
    "tau_afterglow": 10e-3,    # afterglow duration after discharge [s]
    "tau_cycle": 3.0,          # total cycle length for Plasma=False [s]
    "I_breakdown": 1000.0,     # I_tot threshold that marks breakdown [A]
    "h0": 1e-6,                # initial adaptive step size [s]
    "h_max_discharge": 1e-5,   # max step size during active discharge phases [s]
    "h_max_afterglow": 1e-4,   # max step size during afterglow/off phases [s]
    "cells": 3,
    "rtol": 1e-3,  # relative tolerance for adaptive stepping
    "h_min": 1e-12,  # minimum allowed step size [s]
}

input_flags_template = {
    "eperp": False,
    "iperp": False,
    "icool": True,
    "ncool": True,
    "cx": True,
    "mit_el": False,
    "C_imp": False,
    "O_imp": False,
    "icool_recomb": False,
    "Plasma": True,
    "TwinCathode": False,
    "Velocity": True,
    "advection": False,  # Include v·∇v convective acceleration in velocity equation
    "adaptive_mesh": False,  # Dynamically refine/coarsen spatial cells based on MFP criterion
    "mfp_transport": False,  # Use exponential MFP kernel for ne transport; False = nearest-neighbor Laplacian
    "nonlocal_ne": False,    # Non-local density flux correction: Beer-Lambert kernel gated by min(c_s*h, λ_ii) > dx
    "sonic_ne": False,       # Interior face density flux at sound speed (symmetric ne*c_s); ignores v_plasma for transport
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
        Velocity     = true
        mfp_transport = true
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
    ):
        self._flags = input_flags
        self._input_dict = dict(input_dict)
        self._cathode_cell_len = 100.0  # fixed cathode cell length [cm]
        self._n_interior = _to_odd(input_dict.get("cells", 3))
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
        self._Bz0 = np.ones(self._cells) * input_dict["Bz0"]
        self._L_machine = input_dict["Lm"]
        self._L_cell = self._build_L_cell(self._n_interior)
        self._R_machine = np.ones(self._cells) * input_dict["Rm"]
        self._L_plasma = self._build_L_plasma(self._n_interior)
        self.lam_min = 0.3 * np.min(self._L_plasma)
        self._R_plasma = np.ones(self._cells) * input_dict["Rp"]
        self._Rsq_ratio = (self._R_plasma / self._R_machine) ** 2
        self._L_heatflux = self._L_plasma / 2
        self._R_heatflux = np.ones(self._cells) * input_dict["Rhf"]
        self._Tn = input_dict.get("Tn", 0.025)
        self._v_th_n = np.sqrt(8 * kb_cgs * 300.0 / (np.pi * self._mu_neutral * m_p_cgs))
        self._b_epara = input_dict.get("b_epara", 1.0)
        self._b_ipara = input_dict.get("b_ipara", 1.0)
        self._b_eperp = input_dict.get("b_eperp", 1.0)
        self._b_iperp = input_dict.get("b_iperp", 1.0)
        self._b_ioniz = input_dict.get("b_ioniz", 1.0)
        self._b_rec_rad = input_dict.get("b_rec_rad", 1.0)
        self._b_rec_3b = input_dict.get("b_rec_3b", 1.0)
        self._b_Qcx = input_dict.get("b_Qcx", 1.0)
        self._b_Qie = input_dict.get("b_Qie", 1.0)
        self._b_Qei = input_dict.get("b_Qei", 1.0)
        self._b_Qen = input_dict.get("b_Qen", 1.0)
        self._b_source = input_dict.get("b_source", 1.0)
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
        self._tau_afterglow = input_dict.get("tau_afterglow", 10e-3)
        self._tau_cycle = input_dict.get("tau_cycle", 3.0)
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
        self._gp_puff_factor = input_dict.get("gp_puff_factor", 1.0)
        self._tau_gp_ramp = input_dict.get("tau_gp_ramp", 0.0)
        self._t_current = 0.0
        self._rtol = input_dict.get("rtol", 1e-3)
        self._h_min = input_dict.get("h_min", 1e-12)
        # Per-component absolute tolerance matched to existing floor values.
        # Shape (5, 1) broadcasts with state shape (5, cells).
        self._atol = np.array([[1e8], [1e8], [0.01], [0.01], [1.0]])
        _max_interior = _to_odd(max(input_dict.get("max_cells", 18), self._n_interior))
        self._max_cells = _max_interior + self._n_cathode_cells
        self._min_cells = _to_odd(min(input_dict.get("min_cells", 3), self._n_interior))
        self._mfp_refine_thresh = input_dict.get("mfp_refine_threshold", 0.5)
        self._mfp_coarsen_thresh = input_dict.get("mfp_coarsen_threshold", 2.0)
        self._print_init_summary()

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
        self._e_perp_hl = np.zeros(self._cells)  # Electron perpendicular heat loss
        self._i_perp_hl = np.zeros(self._cells)  # Ion perpendicular heat loss
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
        # Dynamic lists — converted to arrays by _finalize_results()
        self._time_list = []
        self._densities_list = []
        self._temperatures_list = []
        self._density_terms_list = []
        self._heat_terms_list = []
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

    def update_results(self, t):
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
                    self._pad(self._e_perp_hl),
                    self._pad(self._i_perp_hl),
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
            primary_mfp = self._l_b / self._L_plasma
            bulk_mfp = v_thm_e(self._Te) * tau_e / self._L_plasma
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
        self._heat_terms = np.array(self._heat_terms_list)  # (n, 14, max_cells)
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
            self._S_rec_rad = self._b_rec_rad * ne * ne * alpha_r(Te)
            self._S_rec_3b = self._b_rec_3b * ne * ne * ne * alpha_3(Te)

    def _effective_S_gp(self):
        """Return S_gp, boosted by gp_puff_factor while t < tau_gp_ramp."""
        if self._tau_gp_ramp <= 0.0 or self._gp_puff_factor == 1.0:
            return self._S_gp
        if self._t_current < self._tau_gp_ramp:
            return self._S_gp * self._gp_puff_factor
        return self._S_gp

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
        Nn_flux = np.zeros(self._cells)
        Nn_flux[:-1] += kappa * delta_n / L[:-1]
        Nn_flux[1:]  -= kappa * delta_n / L[1:]
        return Nn_flux

    def _dstep_nn(self, nn):
        """Derivative for nn-only integration when Plasma=False."""
        Nn_flux = self._nn_clausing_flux(nn)
        return Nn_flux + (self._effective_S_gp() if self._discharge_on else 0) - self._S_pump * nn


    # ── NON-LOCAL FLUX KERNEL (disabled) ─────────────────────────────────────
    # _compute_v_kernel and _ne_flux_mfp_kernel are retained for reference but
    # are not called. v_plasma is always evolved by the velocity ODE.

    def _compute_v_kernel(self, ne, c_s):
        """
        Infer cell-centred bulk velocity from the ambipolar flux pattern.

        Net rightward particle flux at face f is Γ[f] - Γ[f+1] = n[f]*c_s[f] - n[f+1]*c_s[f+1].
        Dividing by the face-averaged density gives the implied bulk velocity at that
        face.  Cell-centred values are the average of the two adjacent face velocities
        (one-sided for boundary cells).

        Used to maintain v_plasma continuity: while a cell is in the kernel regime
        (Kn > 1/3) its velocity is set to this value after each accepted RK45 step,
        so the handoff to fluid advection at Kn ≤ 1/3 is seamless.
        """
        Γ = ne * c_s
        n_face = 0.5 * (ne[:-1] + ne[1:])
        v_face = (Γ[:-1] - Γ[1:]) / n_face     # (cells-1,) net rightward face velocity
        v_cell = np.empty(self._cells)
        v_cell[0] = v_face[0]
        v_cell[-1] = v_face[-1]
        v_cell[1:-1] = 0.5 * (v_face[:-1] + v_face[1:])
        return v_cell

    def _ne_flux_mfp_kernel(self, ne, Te, c_s, v_plasma):
        """
        Per-face Kn-gated density transport.

        At each face f the local Knudsen number Kn_f = λ_ei / Δx_face selects
        the transport scheme:

          Kn_f > 1/3  →  Beer-Lambert exponential kernel.
              Rightward flux Γ[f] = ne[f]*c_s[f] leaves cell f and deposits into
              cells f+1, f+2, … with weights derived from exponential absorption
              (mean-free-path = λ_face).  The end cell absorbs all remaining flux,
              so weights sum exactly to 1 without normalisation.
              Leftward flux Γ[f+1] mirrors this toward cell 0.

          Kn_f ≤ 1/3  →  Velocity advection using v_plasma from the ODE.
              Face flux = ne_face · v_face (nearest-neighbour, fluid limit).

        Returns the transport contribution to dne/dt [cm⁻³ s⁻¹].
        Boundary conditions are NOT included; added by the caller.
        """
        L = self._L_plasma
        Γ = ne * c_s
        λ_ei = v_thm_e(Te) * time_elec_coll(Te, ne, self._ln_lambda)
        λ_face = 0.5 * (λ_ei[:-1] + λ_ei[1:])
        Kn_face = λ_face / (0.5 * (L[:-1] + L[1:]))

        Ne_flux = np.zeros(self._cells)

        for f in range(self._cells - 1):
            if Kn_face[f] > (1.0 / 3.0):
                lam = λ_face[f]

                # ── Rightward: Γ[f] leaves cell f, deposits in cells f+1 … cells-1 ──
                L_r = L[f + 1:]                                         # lengths of target cells
                n_r = len(L_r)
                d_entry_r = np.empty(n_r)
                d_entry_r[0] = 0.0
                if n_r > 1:
                    d_entry_r[1:] = np.cumsum(L_r[:-1])
                w_r = np.empty(n_r)
                if n_r > 1:
                    w_r[:-1] = np.exp(-d_entry_r[:-1] / lam) * (1.0 - np.exp(-L_r[:-1] / lam))
                w_r[-1] = np.exp(-d_entry_r[-1] / lam)     # end cell absorbs remainder
                Ne_flux[f] -= Γ[f] / L[f]
                Ne_flux[f + 1:] += Γ[f] * w_r / L_r

                # ── Leftward: Γ[f+1] leaves cell f+1, deposits in cells f … 0 ──
                L_l = L[f::-1]                                          # [L[f], L[f-1], ..., L[0]]
                n_l = len(L_l)
                d_entry_l = np.empty(n_l)
                d_entry_l[0] = 0.0
                if n_l > 1:
                    d_entry_l[1:] = np.cumsum(L_l[:-1])
                w_l = np.empty(n_l)
                if n_l > 1:
                    w_l[:-1] = np.exp(-d_entry_l[:-1] / lam) * (1.0 - np.exp(-L_l[:-1] / lam))
                w_l[-1] = np.exp(-d_entry_l[-1] / lam)     # end cell absorbs remainder
                # w_l[k] → cell f-k (k=0 is cell f, k=f is cell 0)
                left_idx = np.arange(f, -1, -1)
                Ne_flux[f + 1] -= Γ[f + 1] / L[f + 1]
                Ne_flux[left_idx] += Γ[f + 1] * w_l / L[left_idx]

            else:
                # ── Fluid regime: nearest-neighbour velocity advection ──
                ne_b = 0.5 * (ne[f] + ne[f + 1])
                v_b = 0.5 * (v_plasma[f] + v_plasma[f + 1])
                F = ne_b * v_b
                Ne_flux[f] -= F / L[f]
                Ne_flux[f + 1] += F / L[f + 1]

        return Ne_flux

    def _ne_flux_sonic_nonlocal(self, ne, c_s, v_plasma):
        """
        Density flux with non-local correction for long-mean-free-path transport.

        At each interior face f the effective non-local reach is:
            L_eff = min(c_s_face * h, λ_ii_face)
        where c_s * h is how far sound-speed plasma travels in one time step
        and λ_ii = v_thi * τ_ii is the ion-ion mean free path (collisional cap).

        L_eff > dx_face  →  Beer-Lambert exponential kernel.  Outward flux
        Γ = ne * c_s from each cell is deposited into cells within reach with
        weights exp(-d / L_eff) * (1 − exp(−Δx / L_eff)); the last cell
        absorbs all remaining flux so weights sum to exactly 1.  Rightward
        and leftward fluxes are treated symmetrically.

        L_eff ≤ dx_face  →  local fluid advection: ne_face * v_plasma_face.

        Uses self._Ti and self._ln_lambda frozen from the last accepted step.
        Returns the interior-face contribution to dne/dt [cm⁻³ s⁻¹]; boundary
        conditions are NOT included and are added by _calc_n_flux.
        """
        L = self._L_plasma
        h = self._h
        v_thi = v_ion_speed(self._Ti, self._mu)
        tau_ii = time_ion_coll(self._Ti, ne, self._mu, self._ln_lambda)
        lam_ii = v_thi * tau_ii
        L_eff_cell = np.minimum(c_s * h, lam_ii)

        Ne_flux = np.zeros(self._cells)

        for f in range(self._cells - 1):
            L_eff_f = 0.5 * (L_eff_cell[f] + L_eff_cell[f + 1])
            dx_f = 0.5 * (L[f] + L[f + 1])

            if L_eff_f <= dx_f:
                # Local fluid limit
                ne_b = 0.5 * (ne[f] + ne[f + 1])
                v_b = 0.5 * (v_plasma[f] + v_plasma[f + 1])
                F = ne_b * v_b
                Ne_flux[f] -= F / L[f]
                Ne_flux[f + 1] += F / L[f + 1]
            else:
                lam = L_eff_f

                # ── Rightward: ne[f]*c_s[f] leaves cell f → cells f+1, f+2, … ──
                L_r = L[f + 1:]
                d_r = np.empty(len(L_r))
                d_r[0] = 0.0
                if len(L_r) > 1:
                    d_r[1:] = np.cumsum(L_r[:-1])
                w_r = np.empty(len(L_r))
                if len(L_r) > 1:
                    w_r[:-1] = np.exp(-d_r[:-1] / lam) * (1.0 - np.exp(-L_r[:-1] / lam))
                w_r[-1] = np.exp(-d_r[-1] / lam)  # end cell absorbs remainder
                Ne_flux[f] -= ne[f] * c_s[f] / L[f]
                Ne_flux[f + 1:] += ne[f] * c_s[f] * w_r / L_r

                # ── Leftward: ne[f+1]*c_s[f+1] leaves cell f+1 → cells f, f-1, … ──
                L_l = L[f::-1]
                d_l = np.empty(len(L_l))
                d_l[0] = 0.0
                if len(L_l) > 1:
                    d_l[1:] = np.cumsum(L_l[:-1])
                w_l = np.empty(len(L_l))
                if len(L_l) > 1:
                    w_l[:-1] = np.exp(-d_l[:-1] / lam) * (1.0 - np.exp(-L_l[:-1] / lam))
                w_l[-1] = np.exp(-d_l[-1] / lam)  # end cell absorbs remainder
                left_idx = np.arange(f, -1, -1)
                Ne_flux[f + 1] -= ne[f + 1] * c_s[f + 1] / L[f + 1]
                Ne_flux[left_idx] += ne[f + 1] * c_s[f + 1] * w_l / L[left_idx]

        return Ne_flux

    def _calc_n_flux(self, Te, ne, nn, v_plasma, c_s=None):
        # TODO: use the cathode solver results to determine plasma loss in cathode cells
        Ne_flux = np.zeros(self._cells)
        Nn_flux = np.zeros(self._cells)
        if self._flags["Plasma"]:
            if c_s is None:
                c_s = v_ion_speed(Te, self._mu)
            # Boundary losses: cathode cells use sheath/cathode physics; right wall uses Bohm speed
            Ne_flux[0] = -(1 + 2 * self._eta) * self._cathode_result.I_i / qe_SI / self._plasma_vol[0]
            if self._flags["TwinCathode"]:
                Ne_flux[-1] = -(1 + 2 * self._eta) * self._cathode_result_twin.I_i / qe_SI / self._plasma_vol[-1]
            else:
                Ne_flux[-1] = -ne[-1] * c_s[-1] / self._L_plasma[-1]
            Nn_flux[0] = -Ne_flux[0] * self._Rsq_ratio[0]
            Nn_flux[-1] = -Ne_flux[-1] * self._Rsq_ratio[-1]
            # Interior face transport
            if self._flags.get("nonlocal_ne"):
                Ne_flux += self._ne_flux_sonic_nonlocal(ne, c_s, v_plasma)
            elif self._flags.get("sonic_ne") or not self._flags["Velocity"]:
                # Symmetric sound-speed flux: ne[f]*c_s[f] rightward, ne[f+1]*c_s[f+1] leftward
                Gamma_r = ne[:-1] * c_s[:-1]
                Gamma_l = ne[1:]  * c_s[1:]
                Ne_flux[:-1] += (Gamma_l - Gamma_r) / self._L_plasma[:-1]
                Ne_flux[1:]  += (Gamma_r - Gamma_l) / self._L_plasma[1:]
            else:
                ne_face = (ne[:-1] + ne[1:]) / 2
                v_face = (v_plasma[:-1] + v_plasma[1:]) / 2
                F_face = ne_face * v_face
                Ne_flux[:-1] -= F_face / self._L_plasma[:-1]
                Ne_flux[1:]  += F_face / self._L_plasma[1:]
        Nn_flux += self._nn_clausing_flux(nn)
        return Ne_flux, Nn_flux

    def _calc_pres_acc(self, ne, Te):
        """
        Plasma pressure gradient acceleration -(1/ρ) ∇P [cm/s²].

        Finite-volume approach: face pressures P_face = (P[i] + P[i+1]) / 2
        give a net pressure drop across each cell, consistent with the divergence
        discretization used in elec_par_heat_div and _calc_T_convection.
        Boundary cells use a zero-gradient (Neumann) condition at the wall face,
        so P_face_wall = P_boundary, leaving only the interior face contribution.
        """
        P = ne * Te * ev_to_erg  # erg/cm³
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
        self._e_par_flux, self._i_par_flux, self._e_perp_hl, self._i_perp_hl = (
            self._calc_cond_heat_flux(Te, Ti, ne)
        )
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
                self._Qib[i] -= en_factor * P_loss_i / (self._plasma_vol[i] * qe_SI * ne[i])
        if self._flags["C_imp"]:
            pass  # Placeholder for carbon impurity cooling
        if self._flags["O_imp"]:
            pass  # Placeholder for oxygen impurity cooling

    def _calc_cond_heat_flux(self, Te, Ti, ne):
        e_perp_hl = np.zeros(self._cells)
        i_perp_hl = np.zeros(self._cells)
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
        if self._flags["eperp"]:
            e_perp_hl = (
                self._b_eperp
                * en_factor
                * elec_perp_heat_loss(
                    Te, ne, self._Bz0, self._R_plasma, self._R_heatflux, self._ln_lambda
                )
            )
        if self._flags["iperp"]:
            i_perp_hl = (
                self._b_iperp
                * en_factor
                * ion_perp_heat_loss(
                    Ti,
                    ne,
                    self._R_plasma,
                    self._R_heatflux,
                    self._Bz0,
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
        e_par_flux += self._b_epara * en_factor * elec_par_heat_div(
            Te, ne, self._L_plasma, self._ln_lambda
        )
        i_par_flux += self._b_ipara * en_factor * ion_par_heat_div(
            Ti, ne, self._L_plasma, self._mu, self._ln_lambda
        )
        return e_par_flux, i_par_flux, e_perp_hl, i_perp_hl

    def _calc_div_v(self, Te, c_s=None):
        """
        Calculate the divergence of the plasma velocity field (dv/dx) for each cell.
        Uses one-sided differences for end cells and a central difference for interior cells.
        NOTE: the correct discretization scheme should be revisited in a future pass.
        """
        div_v = np.zeros(self._cells)
        if c_s is None:
            c_s = v_ion_speed(Te, self._mu)
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

    def _apply_state_guards(self, ne, nn, Te, Ti, v_plasma):
        np.clip(Ti, 0.01, 100, out=Ti)
        np.clip(Te, 0.01, 100, out=Te)
        np.maximum(ne, 1e8, out=ne)
        np.maximum(nn, 1e8, out=nn)
        if self._flags["Velocity"]:
            c_s_max = 10 * np.max(v_ion_speed(Te, self._mu))
            np.clip(v_plasma, -c_s_max, c_s_max, out=v_plasma)

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
        self._R_heatflux = np.ones(new_total) * self._input_dict["Rhf"]
        self._Bz0 = np.ones(new_total) * self._input_dict["Bz0"]
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

    def _compute_h_max_physical(self):
        """Step ceiling from diffusive and advective CFL conditions on electron heat conduction."""
        kappa_e = 3.16 * time_elec_coll(self._Te, self._ne, self._ln_lambda) * v_thm_e(self._Te) ** 2
        h_cond = 0.5 * np.min(self._L_plasma) ** 2 / np.max(kappa_e)
        c_s = v_ion_speed(self._Te, self._mu)
        v_eff = np.max(c_s + np.abs(self._v_plasma))
        h_advect = 0.5 * np.min(self._L_plasma) / v_eff
        return min(h_cond, h_advect)

    def _dstep(self, a):
        ne, nn, Te, Ti, v_plasma = a
        zeros = np.zeros(self._cells)

        if not self._flags["Plasma"]:
            _, self._Nn_flux = self._calc_n_flux(Te, ne, nn, v_plasma)
            d_nn = (
                self._Nn_flux
                + (self._effective_S_gp() if self._discharge_on else 0)
                - self._S_pump * nn
            )
            return np.array([zeros, d_nn, zeros, zeros, zeros])

        # Lightweight floor for numerical stability at intermediate RK stages.
        # Creates new arrays (does not mutate the input state vector).
        # Full _apply_state_guards (including velocity clip) runs only at the
        # accepted endpoint in _rkf45_step / _rk4_step.
        Te = np.maximum(Te, 0.01)
        Ti = np.maximum(Ti, 0.01)
        ne = np.maximum(ne, 1e8)
        nn = np.maximum(nn, 1e8)

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
            - self._e_perp_hl
            + self._div_v_elec
            + self._Te_conv
        )
        d_Ti = (
            self._Qie
            + self._i_par_flux
            - self._i_perp_hl
            - self._Qcx
            + self._div_v_ions
            + self._Qib
            + self._Ti_conv
        )
        d_nn = (
            self._Nn_flux
            - (self._S_ion_bulk + self._S_ion_beam - self._S_rec_rad - self._S_rec_3b)
            * self._Rsq_ratio
            + (self._effective_S_gp() if self._discharge_on else 0)
            - self._S_pump * nn
        )
        if self._flags["Velocity"]:
            d_ve = (
                self._calc_pres_acc(ne, Te)
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
        k1 = self._dstep(a)
        k2 = self._dstep(a + h * (1 / 5 * k1))
        k3 = self._dstep(a + h * (3 / 40 * k1 + 9 / 40 * k2))
        k4 = self._dstep(a + h * (44 / 45 * k1 - 56 / 15 * k2 + 32 / 9 * k3))
        k5 = self._dstep(
            a
            + h
            * (
                19372 / 6561 * k1
                - 25360 / 2187 * k2
                + 64448 / 6561 * k3
                - 212 / 729 * k4
            )
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
            )
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
        k7 = self._dstep(y5)

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
        factor = min(5.0, max(0.2, 0.9 * err_norm ** (-0.2))) if err_norm > 0 else 5.0
        h_next = h * factor

        accepted = err_norm <= 1.0 or h <= self._h_min

        if accepted:
            ne, nn, Te, Ti, v_plasma = y5
            self._apply_state_guards(ne, nn, Te, Ti, v_plasma)
            self._ln_lambda = c_log(Te, ne)
            self._calc_cathode(Te, ne, nn)
            self.calc_density_terms(ne, nn, Te, v_plasma)
            self.calc_heat_terms(ne, nn, Te, Ti, v_plasma)
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

        factor = min(5.0, max(0.2, 0.9 * err_norm ** (-0.2))) if err_norm > 0 else 5.0
        h_next = h * factor
        accepted = err_norm <= 1.0 or h <= self._h_min

        if accepted:
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
        Three-phase adaptive integration for Plasma=True.

        pre_breakdown: cathode + S_gp on, ends when I_tot >= I_breakdown (t=0)
        main_discharge: cathode + S_gp on, lasts tau_discharge after breakdown
        afterglow: cathode + S_gp off, lasts tau_afterglow
        """
        phase = "pre_breakdown"
        t = 0.0
        self._t_breakdown = None
        self._h = self._h0
        self._discharge_on = True
        self._floating = False
        step_count = 0
        _last_print_t = -1e-3

        while True:
            if phase == "pre_breakdown":
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
                h_max_base = min(h_max_base, self._compute_h_max_physical())
            h_cap = min(h_max_base, remaining)
            self._h = min(self._h, max(h_cap, self._h_min))

            a = np.array([self._ne, self._nn, self._Te, self._Ti, self._v_plasma])
            self._t_current = t
            result, h_next, accepted = self._rkf45_step(a)

            if accepted:
                self._ne, self._nn, self._Te, self._Ti, self._v_plasma = result
                t += self._h
                self._h = min(h_next, h_cap)
                step_count += 1

                self.update_results(t)

                if self._flags.get("adaptive_mesh"):
                    if self._check_mesh(t):
                        self._h = min(self._h, self._compute_h_max_physical())

                if t - _last_print_t >= 1e-3:
                    _last_print_t = t
                    t_rel = t if self._t_breakdown is None else t - self._t_breakdown
                    print(f"  [{phase}] t={t_rel*1e3:.3f} ms  h={self._h:.2e} s  steps={step_count}")
                    print(f"  ne={self._ne}  nn={self._nn}")
                    print(f"  Te={self._Te}  Ti={self._Ti}")

                # Phase transition checks
                if phase == "pre_breakdown":
                    if (self._cathode_result is not None and
                            self._cathode_result.I_tot >= self._I_breakdown):
                        self._t_breakdown = t
                        phase = "main_discharge"
                        print(f"  Breakdown at t={t*1e3:.3f} ms, "
                              f"I_tot={self._cathode_result.I_tot:.1f} A")
                    elif t >= self._tau_prebreakdown:
                        raise BreakdownError(
                            f"Plasma failed to break down within tau_prebreakdown="
                            f"{self._tau_prebreakdown * 1e3:.1f} ms "
                            f"(I_tot={self._cathode_result.I_tot:.1f} A < "
                            f"I_breakdown={self._I_breakdown:.1f} A)"
                        )
                elif phase == "main_discharge":
                    if t >= self._t_breakdown + self._tau_discharge:
                        phase = "afterglow"
                        self._discharge_on = False
                        self._floating = True
                        self._h = min(self._h0, self._h_max_afterglow)
                        print(f"  Main discharge ended at t={(t - self._t_breakdown)*1e3:.3f} ms. "
                              f"Entering afterglow.")
                else:  # afterglow
                    if t >= self._t_breakdown + self._tau_discharge + self._tau_afterglow:
                        break
            else:
                self._h = h_next

    def _run_equilibrium_cycles(self):
        """
        Plasma=False equilibrium integration.

        puffing phase: S_gp on for tau_discharge
        off phase: S_gp off for the remainder of tau_cycle
        Repeats for self._cycles cycles.
        """
        for j in range(self._cycles):
            print(f"Starting cycle {j+1}/{self._cycles}...")
            t = 0.0
            self._h = self._h0
            t_offset = j * self._tau_cycle

            while t < self._tau_cycle * (1 - 1e-12):
                in_puffing = t < self._tau_discharge
                self._discharge_on = in_puffing
                self._floating = True

                t_phase_end = self._tau_discharge if in_puffing else self._tau_cycle
                h_max_base = self._h_max_discharge if in_puffing else self._h_max_afterglow
                h_cap = min(h_max_base, t_phase_end - t)
                self._h = min(self._h, max(h_cap, self._h_min))

                a = np.array([self._ne, self._nn, self._Te, self._Ti, self._v_plasma])
                self._t_current = t
                result, h_next, accepted = self._rkf45_step(a)

                if accepted:
                    self._ne, self._nn, self._Te, self._Ti, self._v_plasma = result
                    t += self._h
                    self._h = min(h_next, h_cap)
                    self.update_results(t + t_offset)
                else:
                    self._h = h_next

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
            S_ion_bulk=self._density_terms[:, 2],
            S_rec_rad=self._density_terms[:, 3],
            S_rec_3b=self._density_terms[:, 4],
            S_ion_beam=self._density_terms[:, 5],
            e_par_flux=self._heat_terms[:, 0],
            i_par_flux=self._heat_terms[:, 1],
            e_perp_hl=self._heat_terms[:, 2],
            i_perp_hl=self._heat_terms[:, 3],
            Qie=self._heat_terms[:, 4],
            Qei=self._heat_terms[:, 5],
            Qen=self._heat_terms[:, 6],
            Qcx=self._heat_terms[:, 7],
            Qeb=self._heat_terms[:, 8],
            div_v_elec=self._heat_terms[:, 9],
            div_v_ions=self._heat_terms[:, 10],
            Qib=self._heat_terms[:, 11],
            Te_conv=self._heat_terms[:, 12],
            Ti_conv=self._heat_terms[:, 13],
            v_plasma=self._velocities[:, 0],
            isat=self._synthetic,
            primary_mfp=self._primary_mfp,
            bulk_mfp=self._bulk_mfp,
            ln_lambda=self._ln_lambda,
            cells_at_time=self._cells_at_time,
            refinement_events=self._refinement_events,
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
