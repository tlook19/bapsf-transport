import math
import tomllib
import numpy as np
from cablp.funcs._cathode_solver import (
    DeviceConfig,
    PlasmaState,
    solve as cathode_solve,
)
from cablp.funcs._heat import (
    elec_par_heat_loss,
    ion_par_heat_loss,
    elec_perp_heat_loss,
    ion_perp_heat_loss,
    Q_cx_He,
    Q_ie,
)
from cablp.vars._cons import (
    qe_SI,
    en_factor,
    drag_factor,
    m_He_cgs,
    ev_to_erg,
    m_p_cgs,
    m_e_cgs,
)
from cablp.funcs._fits import rate_coeff, IAEA_exp1, IAEA_exp4, IAEA_exp6
from cablp.funcs._cross import alpha_3, alpha_r, He_EII_cross, H_EII_cross
from cablp.vars._coeff import aHeI, aHeII, aHI, aHII, a_11s
from cablp.vars._cons import I_ion as IE_Helium, I_Ry as IE_Hydrogen
from cablp.funcs._plasmaparams import v_ion_speed, v_thm_e, time_elec_coll, c_log

He_ion_coeff = [1.3950030050791237e-05, 13.62996440158007]
H_ion_coeff = [1e-5, 6.0]

input_dict_template = {
    "gas_type": "He",
    "ne0": 1e9,
    "Tn_fit": 0.1,  # Neutral temperature for reaction rate fits
    "nn0": 5e12,
    "Source_nn0": 1.2e13,
    "Twin_nn0": 1.2e13,
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
    "d_off": 20e-3,
    "dt_main": 1e-6,
    "end": 25e-3,
    "dt_after": 1e-5,
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
    "breakdown_vel": True,  # Use diffusive flux during breakdown; set False to test without
    "adaptive": True,  # Use Dormand-Prince RK45 adaptive stepping
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
        Velocity      = true
        breakdown_vel = false
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
        self._cells = input_dict.get("cells", 3)
        self._gas_type = input_dict.get("gas_type", "He")
        if self._gas_type == "He":
            self._m_gas = m_He_cgs
            self._mu = 4
            self._I_ion = IE_Helium
            self._ion_fit_coeff = He_ion_coeff
        elif self._gas_type == "H":
            self._m_gas = m_p_cgs
            self._mu = 1
            self._I_ion = IE_Hydrogen
            self._ion_fit_coeff = H_ion_coeff
        self._ne = np.ones(self._cells) * input_dict["ne0"]
        self._nn = np.ones(self._cells) * input_dict["nn0"]
        self._nn[0] = input_dict.get("Source_nn0", self._nn[0])
        self._Te = np.ones(self._cells) * input_dict["Te0"]
        self._Ti = np.ones(self._cells) * input_dict["Ti0"]
        self._Tn_fit = input_dict.get("Tn_fit", 0.1)
        self._v_plasma = np.zeros(self._cells)
        self._Bz0 = np.ones(self._cells) * input_dict["Bz0"]
        self._L_machine = input_dict["Lm"]
        self._L_cell = np.ones(self._cells) * self._L_machine / self._cells
        self._R_machine = np.ones(self._cells) * input_dict["Rm"]
        self._L_plasma = np.ones(self._cells) * input_dict["Lp"] / self._cells
        self._R_plasma = np.ones(self._cells) * input_dict["Rp"]
        self._Rsq_ratio = (self._R_plasma / self._R_machine) ** 2
        self._L_heatflux = self._L_plasma / 2
        self._L_partflux = self._L_plasma / 2
        self._R_heatflux = np.ones(self._cells) * input_dict["Rhf"]
        self._Tn = input_dict.get("Tn", 0.025)
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
            self._nn[-1] = input_dict["Twin_nn0"]
            self._S_gp[-1] = self.puff_rate(
                input_dict["Twin_S_gp"], 2, self._cell_vol[-1]
            )
        # self._active_discharge = self._I_discharge > 0 : need to calculate later with cathode solver
        # size-2 arrays: index 0 = primary cathode, index 1 = twin cathode
        # updated each _dstep from cathode solver phi_c
        self._cycles = input_dict.get("cycles", 1)
        self._cyclelength = 0
        self._d_off = input_dict.get("d_off", 20e-3)
        self._dt_main = input_dict.get("dt_main", 1e-3)
        self._end = input_dict.get("end", 3)
        self._dt_after = input_dict.get("dt_after", 1e-3)
        self._div_v_elec = np.zeros(self._cells)
        self._div_v_ions = np.zeros(self._cells)
        # self._ln_lambda = c_log(self._Te, self._ne)
        _R_cath = input_dict.get("R_cath", 19.0)
        self._device_config = DeviceConfig(
            A_c=math.pi * _R_cath**2,
            mu=self._mu,
            V_bank=input_dict.get("V_bank", 100.0),
            T_s=input_dict.get("T_s", 1900.0),
            phi_wf=input_dict.get("phi_wf", 3.0),
            C_R=input_dict.get("C_R", 29.0),
            R_comp=input_dict.get("R_comp", 0.004),
            eta=input_dict.get("eta", 0.358),
            L_cath=input_dict.get("L_cath", 50.0),
            R_cath=_R_cath,
        )
        self._cathode_x0 = None  # warm-start sheath potential [primary, twin]
        self._cathode_result = None
        if self._flags["TwinCathode"]:
            self._device_config_twin = DeviceConfig(
                A_c=math.pi * _R_cath**2,
                mu=self._mu,
                V_bank=input_dict.get("V_bank", 100.0),
                T_s=input_dict.get("T_s", 1900.0),
                phi_wf=input_dict.get("phi_wf", 3.0),
                C_R=input_dict.get("C_R", 29.0),
                R_comp=input_dict.get("R_comp", 0.004),
                eta=input_dict.get("eta", 0.358),
                Twin=self._flags["TwinCathode"],
                L_cath=input_dict.get("L_cath", 50.0),
                R_cath=_R_cath,
            )
            self._cathode_x0_twin = None
            self._cathode_result_twin = None
        self._rtol = input_dict.get("rtol", 1e-3)
        self._h_min = input_dict.get("h_min", 1e-12)
        # Per-component absolute tolerance matched to existing floor values.
        # Shape (5, 1) broadcasts with state shape (5, cells).
        self._atol = np.array([[1e8], [1e8], [0.01], [0.01], [1.0]])
        self._print_init_summary()

    def _print_init_summary(self):
        """Print key derived quantities computed during __init__ for sanity checking."""
        print(f"=== LAPDSim init summary ===")
        print(f"  gas_type={self._gas_type}  mu={self._mu}  I_ion={self._I_ion:.4f} eV")
        print(f"  ion_fit_coeff={self._ion_fit_coeff}")
        print(
            f"  V_discharge={self._device_config.V_bank}  R_comp={self._device_config.R_comp}"
        )
        print(f"  T_s={self._device_config.T_s} cm/s")
        print(f"  plasma_vol={self._plasma_vol} cm^3")
        print(f"  Rsq_ratio={self._Rsq_ratio}")
        print(f"  S_gp={self._S_gp} cm^-3 s^-1")
        print(f"  S_pump={self._S_pump} s^-1")
        print(f"============================")

    def set_time_steps(self):
        if self._d_off == 0:
            self._i_d_off = 0
            self._afterglow = True
            self._time = np.arange(0, self._end, self._dt_after)
            self._dt_afterglow = self._dt_after
        else:
            self._dt_discharge = self._dt_main
            self._dt_afterglow = self._dt_after
            time = np.arange(0, self._d_off, self._dt_main)
            self._i_d_off = time.shape[0]
            if self._end is not None and self._dt_after is not None:
                time = np.append(
                    time, np.arange(self._d_off, self._end, self._dt_after)
                )
                self._afterglow = True
                self._dt_afterglow = self._dt_after
            else:
                self._afterglow = False
        self._cyclelength = time.shape[0]
        self._onetime = time
        if self._cycles == 1:
            self._time = time
        elif self._cycles > 1 and not self._flags["Plasma"]:
            self._time = np.zeros(self._cycles * self._cyclelength)
            for i in range(self._cycles):
                tslice = slice(i * self._cyclelength, (i + 1) * self._cyclelength)
                self._time[tslice] = time + i * time[-1]
        self._time_flag = True

    def initialize_results(self):
        self._n_beam = np.zeros(self._cells)
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

    def update_results(self, t):
        self._time_list.append(t)
        self._densities_list.append(np.array([self._ne, self._nn, self._n_beam]))
        self._temperatures_list.append(np.array([self._Te, self._Ti]))
        self._density_terms_list.append(
            np.array(
                [
                    self._Ne_flux,
                    self._Nn_flux,
                    self._S_ion_bulk,
                    self._S_rec_rad,
                    self._S_rec_3b,
                    self._S_ion_beam,
                ]
            )
        )
        self._heat_terms_list.append(
            np.array(
                [
                    self._e_par_flux,
                    self._i_par_flux,
                    self._e_perp_hl,
                    self._i_perp_hl,
                    self._Qie,
                    self._Qei,
                    self._Qen,
                    self._Qcx,
                    self._Qeb,
                    self._div_v_elec,
                    self._div_v_ions,
                ]
            )
        )
        self._velocities_list.append(np.array([self._v_plasma]))
        self._synthetic_list.append(self._ne * np.sqrt(self._Te))
        if self._flags["Plasma"]:
            tau_e = time_elec_coll(self._Te, self._ne, self._ln_lambda)
            primary_mfp = self._l_b / self._L_plasma
            bulk_mfp = v_thm_e(self._Te) * tau_e / self._L_plasma
            ln_lambda = self._ln_lambda.copy()
        else:
            primary_mfp = np.zeros(self._cells)
            bulk_mfp = np.zeros(self._cells)
            ln_lambda = np.zeros(self._cells)
        self._primary_mfp_list.append(primary_mfp)
        self._bulk_mfp_list.append(bulk_mfp)
        self._ln_lambda_list.append(ln_lambda)

    def _finalize_results(self):
        """Convert accumulated result lists to NumPy arrays after simulation."""
        self._time = np.array(self._time_list)
        self._densities = np.array(self._densities_list)  # (n, 3, cells)
        self._temperatures = np.array(self._temperatures_list)  # (n, 2, cells)
        self._density_terms = np.array(self._density_terms_list)  # (n, 6, cells)
        self._heat_terms = np.array(self._heat_terms_list)  # (n, 12, cells)
        self._velocities = np.array(self._velocities_list)  # (n, 1, cells)
        self._synthetic = np.array(self._synthetic_list)  # (n, cells)
        self._primary_mfp = np.array(self._primary_mfp_list)  # (n, cells)
        self._bulk_mfp = np.array(self._bulk_mfp_list)  # (n, cells)
        self._ln_lambda = np.array(self._ln_lambda_list)  # (n, cells)

    def calc_density_terms(
        self,
        ne,
        nn,
        Te,
        v_plasma,
    ):
        Ne_flux, Nn_flux = self._calc_n_flux(Te, ne, nn, v_plasma)
        S_ion_beam = np.zeros(self._cells)
        S_ion_bulk = np.zeros(self._cells)
        S_rec_rad = np.zeros(self._cells)
        S_rec_3b = np.zeros(self._cells)
        if self._flags["Plasma"]:
            if self._gas_type in ("He", "H"):
                S_ion_bulk = (
                    self._b_ioniz
                    * ne
                    * nn
                    * rate_coeff(Te, self._I_ion, *self._ion_fit_coeff)
                )
                for i in [0, -1] if self._flags["TwinCathode"] else [0]:
                    S_ion_beam[i] = self._A_ion_beam[i] * self._p_beam[i]
            # NOTE: alpha_r and alpha_3 are approximate power-law fits used for both species.
            # For helium, replace with better species-specific recombination rates when available.
            S_rec_rad = self._b_rec_rad * ne * ne * alpha_r(Te)
            S_rec_3b = self._b_rec_3b * ne * ne * ne * alpha_3(Te)
        return Ne_flux, Nn_flux, S_ion_bulk, S_rec_rad, S_rec_3b, S_ion_beam

    def _calc_n_flux(self, Te, ne, nn, v_plasma):
        """
        Calculate the particle fluxes for electrons and neutrals.
        To be added to the time derivative of the densities to get the net change in density due to transport.

        Parameters
        ----------
        ne : _type_
            _description_
        nn : _type_
            _description_
        vp : _type_
            _description_
        vn : _type_
            _description_
        v_p_bound : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        Ne_flux = np.zeros(self._cells)
        Nn_flux = np.zeros(self._cells)
        nn_flux = nn * v_ion_speed(self._Tn, self._mu)
        if self._flags["Plasma"]:
            ne_flux = ne * v_ion_speed(Te, self._mu)
            Ne_flux[0] = -ne_flux[0] / (2 * self._L_partflux[0])
            Ne_flux[-1] = -ne_flux[-1] / (2 * self._L_partflux[-1])
            Nn_flux[0] = -Ne_flux[0] * self._Rsq_ratio[0]
            Nn_flux[-1] = -Ne_flux[-1] * self._Rsq_ratio[-1]
        denom = self._L_partflux[:-1] + self._L_partflux[1:]
        if self._flags["Plasma"]:
            if self._flags["Velocity"] and not (
                self._breakdown and self._flags["breakdown_vel"]
            ):
                ne_b = (ne[:-1] + ne[1:]) / 2
                v_b = (v_plasma[:-1] + v_plasma[1:]) / 2
                neflux = 2 * ne_b * v_b / denom
            else:
                neflux = (ne_flux[:-1] - ne_flux[1:]) / denom
            Ne_flux[:-1] -= neflux
            Ne_flux[1:] += neflux
        nnflux = (nn_flux[:-1] - nn_flux[1:]) / denom
        Nn_flux[:-1] -= nnflux
        Nn_flux[1:] += nnflux
        return Ne_flux, Nn_flux

    def _calc_beam_prob(self, Te, ne, nn):
        self._l_bi = np.zeros(self._cells)
        self._l_bn = np.zeros(self._cells)
        self._l_b = np.zeros(self._cells)
        self._p_beam = np.zeros(self._cells)
        for i in [0, -1] if self._flags["TwinCathode"] else [0]:
            ln_lambda_ei = c_log(Te[i], ne[i], type="ei")
            self._l_bi[i] = self._v_beam[i] * time_elec_coll(Te[i], ne[i], ln_lambda_ei)
            self._l_bn[i] = 1.0 / (self._beam_cross[i] * nn[i])
            self._l_b[i] = 1.0 / (1.0 / self._l_bi[i] + 1.0 / self._l_bn[i])
            self._p_beam[i] = self._l_b[i] / self._l_bn[i]

    def _calc_pres_acc(self, ne, Te):
        """
        Calculate the plasma pressure gradient acceleration (dv/dt contribution).
        Uses one-sided differences for end cells and a central-like net difference
        for interior cells. Divides by ion mass (self._m_gas) for CGS units.
        NOTE: discretization scheme should be revisited in a future pass.
        """
        P = ne * Te * ev_to_erg  # pressure in erg/cm^3
        L = self._L_partflux
        pres_acc = np.zeros(self._cells)
        # End cell 0: one-sided difference with adjacent cell
        pres_acc[0] = -(P[1] - P[0]) / (L[0] * self._m_gas * ne[0])
        # End cell -1: one-sided difference with adjacent cell
        pres_acc[-1] = -(P[-1] - P[-2]) / (L[-1] * self._m_gas * ne[-1])
        # Interior cells: net contribution from both neighbors (vectorized)
        pres_acc[1:-1] = -(
            (P[2:] - P[1:-1]) / (L[1:-1] + L[2:])
            + (P[1:-1] - P[:-2]) / (L[:-2] + L[1:-1])
        ) / (self._m_gas * ne[1:-1])
        return pres_acc

    def _calc_drag_in(self, Ti, nn, v_plasma):
        """
        Calculate the ion-neutral drag force on the plasma per unit volume.
        This term is subtracted from the plasma velocity time derivative.
        """
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
    ):
        e_par_flux, i_par_flux, e_perp_hl, i_perp_hl = self._calc_cond_heat_flux(
            Te, Ti, ne
        )
        Qei = np.zeros(self._cells)
        Qen = np.zeros(self._cells)
        Qcx = np.zeros(self._cells)
        Qeb = np.zeros(self._cells)
        div_v_elec = np.zeros(self._cells)
        div_v_ions = np.zeros(self._cells)
        Qie = self._b_Qie * en_factor * Q_ie(Te, Ti, ne, self._mu, self._ln_lambda)
        if self._flags["Velocity"]:
            # NOTE: physical form of div(v) term should be revisited.
            div_v = self._calc_div_v(Te, v_plasma)
            div_v_elec = -en_factor * Te * div_v
            div_v_ions = -en_factor * Ti * div_v
        if self._flags["icool"]:
            if self._gas_type == "He":
                Qei = (
                    self._b_Qei
                    * en_factor
                    * IAEA_exp4(Te, aHeII, recomb=self._flags["icool_recomb"])
                    * ne
                )
            else:
                Qei = self._b_Qei * en_factor * IAEA_exp6(Te, aHII) * ne
        if self._flags["ncool"]:
            if self._gas_type == "He":
                Qen = self._b_Qen * en_factor * IAEA_exp1(Te, aHeI) * nn
            else:  # H
                Qen = self._b_Qen * en_factor * IAEA_exp1(Te, aHI) * nn
        if self._flags["cx"]:
            Qcx = (
                self._b_Qcx
                * en_factor
                * Q_cx_He(ne, nn, Ti, self._Tn_fit, gas_type=self._gas_type)
            )
        if self._discharge_on:
            cathode_results = {0: self._cathode_result, -1: self._cathode_result_twin}
            for i in ([0, -1] if self._flags["TwinCathode"] else [0]):
                Qeb[i] = (
                    en_factor
                    * cathode_results[i].P_net
                    / self._plasma_vol[i]
                    / qe_SI
                    / ne[i]
                ) - en_factor * self._p_beam[i] * self._A_ion_beam[i] * self._I_ion / ne[i]
        if self._flags["C_imp"]:
            pass  # Placeholder for carbon impurity cooling
        if self._flags["O_imp"]:
            pass  # Placeholder for oxygen impurity cooling
        return (
            e_par_flux,
            i_par_flux,
            e_perp_hl,
            i_perp_hl,
            Qie,
            Qei,
            Qen,
            Qeb,
            Qcx,
            div_v_elec,
            div_v_ions,
        )

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
        # if self._discharge_on:  # TODO: cathode off heat loss boundary conditions
        #     e_par_flux[0] -= (
        #         self._I_beam[0] * Te[0] / self._plasma_vol[0] / qe_SI / ne[0]
        #     )
        if not self._flags["TwinCathode"]:
            e_par_flux[-1] -= e_par_hl[-1]
        #     elif self._flags["TwinCathode"]:
        #         e_par_flux[-1] -= (
        #             self._I_beam[-1] * Te[-1] / self._plasma_vol[-1] / qe_SI / ne[-1]
        #         )
        # elif not self._discharge_on:
        #     e_par_flux[0] -= e_par_hl[0]
        #     e_par_flux[-1] -= e_par_hl[-1]
        i_par_flux = np.zeros(self._cells)
        # i_par_flux[0] -= i_par_hl[0]
        # i_par_flux[-1] -= i_par_hl[-1]
        L = self._L_heatflux
        denom = L[:-1] + L[1:]
        # Flux from right neighbor (k < cells-1)
        e_par_flux[:-1] += (e_par_hl[1:] * L[:-1] - e_par_hl[:-1] * L[:-1]) / denom
        i_par_flux[:-1] += (i_par_hl[1:] * L[1:] - i_par_hl[:-1] * L[:-1]) / denom
        # Flux from left neighbor (k > 0)
        e_par_flux[1:] += (e_par_hl[:-1] * L[:-1] - e_par_hl[1:] * L[1:]) / denom
        i_par_flux[1:] += (i_par_hl[:-1] * L[:-1] - i_par_hl[1:] * L[1:]) / denom
        return e_par_flux, i_par_flux, e_perp_hl, i_perp_hl

    def _calc_div_v(self, Te, v_plasma):
        """
        Calculate the divergence of the plasma velocity field (dv/dx) for each cell.
        Uses one-sided differences for end cells and a central difference for interior cells.
        NOTE: the correct discretization scheme should be revisited in a future pass.
        """
        div_v = np.zeros(self._cells)
        c_s = v_ion_speed(Te, self._mu)
        # Face-averaged velocities between adjacent cells (length cells-1)
        v_face = (v_plasma[:-1] + v_plasma[1:]) / 2
        if self._discharge_on:
            div_v[0] = v_face[0] / self._L_plasma[0]
            if self._flags["TwinCathode"]:
                div_v[-1] = -v_face[-1] / self._L_plasma[-1]
            else:
                div_v[-1] = (c_s[-1] - v_face[-1]) / self._L_plasma[-1]
        else:
            div_v[0] = (v_face[0] + c_s[0]) / self._L_plasma[0]
            div_v[-1] = (c_s[-1] - v_face[-1]) / self._L_plasma[-1]
        # Interior cells: central difference of face velocities (vectorized)
        div_v[1:-1] = (v_face[1:] - v_face[:-1]) / self._L_plasma[1:-1]
        return div_v

    def _calc_beam_density(self, I, A, v):
        return I / (qe_SI * A * v)

    def _calc_cathode(self, Te, ne, nn):
        self._cathode_result = cathode_solve(
            self._device_config,
            PlasmaState(T_e=Te[0], n_e=ne[0]),
            x0=self._cathode_x0,
            floating=self._floating,
        )
        self._cathode_x0 = self._cathode_result.phi_c_plus
        phi_c_0 = self._cathode_result.phi_c
        self._v_beam = np.zeros(self._cells)
        self._n_beam = np.zeros(self._cells)
        self._n_beam_ion = np.zeros(self._cells)
        self._A_ion_beam = np.zeros(self._cells)
        self._beam_cross = np.zeros(self._cells)
        self._v_beam[0] = np.sqrt(2 * phi_c_0 * ev_to_erg / m_e_cgs)
        self._n_beam[0] = self._calc_beam_density(
            self._cathode_result.I_eth_star, self._plasma_cross[0], self._v_beam[0]
        )
        if self._gas_type == "He":
            self._beam_cross[0] = (
                float(He_EII_cross(phi_c_0 / self._I_ion, a_11s))
                if phi_c_0 > self._I_ion
                else 0.0
            )
        elif self._gas_type == "H":
            self._beam_cross[0] = (
                float(H_EII_cross(phi_c_0)) if phi_c_0 > self._I_ion else 0.0
            )
        if self._flags["TwinCathode"]:
            self._cathode_result_twin = cathode_solve(
                self._device_config,
                PlasmaState(T_e=Te[-1], n_e=ne[-1]),
                x0=self._cathode_x0_twin,
                floating=self._floating,
            )
            self._cathode_x0_twin = self._cathode_result_twin.phi_c_plus
            phi_c_1 = self._cathode_result_twin.phi_c
            self._v_beam[-1] = np.sqrt(2 * phi_c_1 * ev_to_erg / m_e_cgs)
            self._n_beam[-1] = self._calc_beam_density(
                self._cathode_result_twin.I_eth_star,
                self._plasma_cross[-1],
                self._v_beam[-1],
            )
            if self._gas_type == "He":
                self._beam_cross[-1] = (
                    float(He_EII_cross(phi_c_1 / self._I_ion, a_11s))
                    if phi_c_1 > self._I_ion
                    else 0.0
                )
            elif self._gas_type == "H":
                self._beam_cross[-1] = (
                    float(H_EII_cross(phi_c_1)) if phi_c_1 > self._I_ion else 0.0
                )
        self._n_beam_ion = self._n_beam * self._beam_cross * self._v_beam
        self._A_ion_beam = self._n_beam_ion * nn

    def _dstep(self, a):
        ne, nn, Te, Ti, v_plasma = a
        if self._flags["Plasma"]:
            Ti[Ti < 0.01] = 0.01
            Te[Te < 0.01] = 0.01
            Ti[Ti > 100] = 100
            Te[Te > 100] = 100
            ne[ne < 1e8] = 1e8
            nn[nn < 1e8] = 1e8
        if self._flags["Velocity"]:
            c_s_max = np.max(v_ion_speed(Te, self._mu))
            v_plasma[:] = np.clip(v_plasma, -c_s_max, c_s_max)
        if self._flags["Plasma"]:
            self._ln_lambda = c_log(Te, ne)
            self._calc_cathode(Te, ne, nn)
            self._calc_beam_prob(Te, ne, nn)
        (
            Ne_flux,
            Nn_flux,
            S_ion_bulk,
            S_rec_rad,
            S_rec_3b,
            S_ion_beam,
        ) = self.calc_density_terms(
            ne,
            nn,
            Te,
            v_plasma,
        )
        if self._flags["Velocity"]:
            pres_acc = self._calc_pres_acc(ne, Te)
            drag_in_plasma = self._calc_drag_in(Ti, nn, v_plasma)
        else:
            pres_acc = np.zeros(self._cells)
            drag_in_plasma = np.zeros(self._cells)
        if self._flags["Plasma"]:
            (
                e_par_flux,
                i_par_flux,
                e_perp_hl,
                i_perp_hl,
                Qie,
                Qei,
                Qen,
                Qeb,
                Qcx,
                div_v_elec,
                div_v_ions,
            ) = self.calc_heat_terms(
                ne,
                nn,
                Te,
                Ti,
                v_plasma,
            )
            d_ne = S_ion_bulk + S_ion_beam - S_rec_rad - S_rec_3b + Ne_flux
            d_Te = Qeb - Qie - Qei - Qen + e_par_flux - e_perp_hl + div_v_elec
            d_Ti = Qie + i_par_flux - i_perp_hl - Qcx + div_v_ions
        else:
            d_ne = np.zeros(self._cells)
            d_Te = np.zeros(self._cells)
            d_Ti = np.zeros(self._cells)
        d_nn = (
            Nn_flux
            - S_ion_bulk * self._Rsq_ratio
            - S_ion_beam * self._Rsq_ratio
            + S_rec_rad * self._Rsq_ratio
            + S_rec_3b * self._Rsq_ratio
            + (self._S_gp if self._discharge_on else 0)
            - (self._S_pump * nn)
        )
        d_ve = pres_acc - drag_in_plasma

        return np.array([d_ne, d_nn, d_Te, d_Ti, d_ve])

    def _rk4_step(self, a):
        # print("RK4 step with h =", self._h)
        k1 = self._dstep(a)
        k2 = self._dstep(a + (0.5 * self._h * k1))
        k3 = self._dstep(a + (0.5 * self._h * k2))
        k4 = self._dstep(a + (self._h * k3))
        # print("k: ", k1, k2, k3, k4)
        b = a + (self._h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        # print("b: ", b)
        ne, nn, Te, Ti, v_plasma = b
        if self._flags["Plasma"]:
            Ti[Ti < 0.01] = 0.01
            Te[Te < 0.01] = 0.01
            Ti[Ti > 100] = 100
            Te[Te > 100] = 100
            ne[ne < 1e8] = 1e8
            nn[nn < 1e8] = 1e8
        if self._flags["Velocity"]:
            c_s_max = np.max(v_ion_speed(Te, self._mu))
            v_plasma[:] = np.clip(v_plasma, -c_s_max, c_s_max)
        # Store diagnostics at final state (avoids duplicate calls in start_simulation)
        if self._flags["Plasma"]:
            self._ln_lambda = c_log(Te, ne)
            self._calc_cathode(Te, ne, nn)
            self._calc_beam_prob(Te, ne, nn)
        (
            self._Ne_flux,
            self._Nn_flux,
            self._S_ion_bulk,
            self._S_rec_rad,
            self._S_rec_3b,
            self._S_ion_beam,
        ) = self.calc_density_terms(ne, nn, Te, v_plasma)
        if self._flags["Plasma"]:
            (
                self._e_par_flux,
                self._i_par_flux,
                self._e_perp_hl,
                self._i_perp_hl,
                self._Qie,
                self._Qei,
                self._Qen,
                self._Qeb,
                self._Qcx,
                self._div_v_elec,
                self._div_v_ions,
            ) = self.calc_heat_terms(ne, nn, Te, Ti, v_plasma)
        return ne, nn, Te, Ti, v_plasma

    def _rkf45_step(self, a):
        """
        One Dormand-Prince RK45 adaptive step.

        Uses ``self._h`` as the trial step size.  Returns
        ``((ne, nn, Te, Ti, v_plasma), h_next, accepted)``.
        Guardrails and diagnostics are stored only on accept.
        The caller is responsible for retrying with the returned ``h_next``
        when ``accepted`` is False.
        """
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
            if self._flags["Plasma"]:
                np.clip(Ti, 0.01, 100, out=Ti)
                np.clip(Te, 0.01, 100, out=Te)
                np.maximum(ne, 1e8, out=ne)
                np.maximum(nn, 1e8, out=nn)
            if self._flags["Velocity"]:
                c_s_max = np.max(v_ion_speed(Te, self._mu))
                np.clip(v_plasma, -c_s_max, c_s_max, out=v_plasma)
            if self._flags["Plasma"]:
                self._ln_lambda = c_log(Te, ne)
                self._calc_cathode(Te, ne, nn)
                self._calc_beam_prob(Te, ne, nn)
            (
                self._Ne_flux,
                self._Nn_flux,
                self._S_ion_bulk,
                self._S_rec_rad,
                self._S_rec_3b,
                self._S_ion_beam,
            ) = self.calc_density_terms(ne, nn, Te, v_plasma)
            if self._flags["Plasma"]:
                (
                    self._e_par_flux,
                    self._i_par_flux,
                    self._e_perp_hl,
                    self._i_perp_hl,
                    self._Qie,
                    self._Qei,
                    self._Qen,
                    self._Qeb,
                    self._Qcx,
                    self._Qbeam,
                    self._div_v_elec,
                    self._div_v_ions,
                ) = self.calc_heat_terms(ne, nn, Te, Ti, v_plasma)
            return (ne, nn, Te, Ti, v_plasma), h_next, True
        else:
            return None, h_next, False

    def start_simulation(self):
        self.set_time_steps()
        self.initialize_results()
        print("Starting simulation...")
        for j in range(self._cycles):
            print(f"Starting cycle {j+1}/{self._cycles}...")
            t = 0.0
            self._h = self._dt_discharge if self._d_off > 0 else self._dt_afterglow
            _afterglow_transitioned = False
            step_count = 0

            while t < self._end * (1 - 1e-12):
                in_discharge = (self._d_off > 0) and (t < self._d_off)
                h_max = self._dt_discharge if in_discharge else self._dt_afterglow
                # Cap h to not overshoot phase boundary or end time
                if in_discharge:
                    h_max = min(h_max, self._d_off - t)
                h_max = min(h_max, self._end - t)
                self._h = min(self._h, h_max)
                if self._h < self._h_min:
                    self._h = self._h_min

                # ── Update discharge / beam state ──────────────────────────
                if in_discharge and self._flags["Plasma"]:
                    self._discharge_on = True
                    self._breakdown = True
                    self._floating = False
                    # if condition to say breakdown is over
                    #     self._breakdown = False
                    # TODO: calculate beam density in dstep after cathode solver
                elif in_discharge and not self._flags["Plasma"]:
                    self._discharge_on = True
                    self._floating = True
                elif self._afterglow:
                    self._discharge_on = False
                    self._floating = True
                    if not _afterglow_transitioned:
                        _afterglow_transitioned = True
                        self._h = self._dt_afterglow  # reset step size at transition
                else:
                    break

                a = np.array(
                    [
                        self._ne[:],
                        self._nn[:],
                        self._Te[:],
                        self._Ti[:],
                        self._v_plasma[:],
                    ]
                )

                # ── Integrate one step ─────────────────────────────────────
                if self._flags["adaptive"]:
                    result, h_next, accepted = self._rkf45_step(a)
                else:
                    result = self._rk4_step(a)
                    h_next = self._h
                    accepted = True

                if accepted:
                    self._ne, self._nn, self._Te, self._Ti, self._v_plasma = result
                    t += self._h
                    self._h = min(h_next, h_max)
                    self.update_results(t + j * self._end)
                    step_count += 1
                    if step_count % 5000 == 0:
                        print(
                            f"  t={t * 1e3:.3f} ms  h={self._h:.2e} s  steps={step_count}"
                        )
                        print(f"  ne={self._ne}  nn={self._nn}")
                        print(f"  Te={self._Te}  Ti={self._Ti}")
                        print(
                            f"  primary_mfp/dx={self._primary_mfp_list[-1]}"
                            f"  bulk_mfp/dx={self._bulk_mfp_list[-1]}"
                        )
                        print(f"  ln_lambda={self._ln_lambda_list[-1]}")
                else:
                    self._h = h_next  # rejected: shrink h and retry

        self._finalize_results()
        print("Simulation complete.")

    # def heat_balance(self, ylim=(-1e6, 1e6), discharge=True):
    #     # TODO: revisit — currently broken (undefined attributes, outdated API)
    #     if discharge:
    #         self._discharge_on = True
    #     else:
    #         self._discharge_on = False
    #     Ti = self._Ti0
    #     temps = np.arange(0.1, 20, 0.1)
    #     densities = np.logspace(11, 13, num=5)
    #     curves = np.empty((densities.shape[0], temps.shape[0]))
    #     for i, ne in enumerate(densities):
    #         for j, Te in enumerate(temps):
    #             e_par_hl, e_perp_hl, Qie, Qei, Qen, Qeb, i_par_hl, i_perp_hl, Qcx = (
    #                 self.calc_heat_terms(ne, self._nn0, Te, Ti)
    #             )
    #             total = (
    #                 Qeb
    #                 - Qie
    #                 - Qei
    #                 - Qen
    #                 - e_par_hl
    #                 - e_perp_hl
    #                 - i_par_hl
    #                 - i_perp_hl
    #                 - Qcx
    #             )
    #             curves[i, j] = total
    #     fig, ax = plt.subplots()
    #     for i, ne in enumerate(densities):
    #         ax.plot(temps, curves[i], label=f"ne={ne:.1e} cm^-3")
    #     ax.axhline(0, color="k", linestyle="--")
    #     ax.set_xlabel("Electron Temperature (eV)")
    #     ax.set_ylabel("Net Heating/Cooling (eV/s)")
    #     ax.set_title("Heat Balance Curves @ P = {:.1f} W".format(self._P_discharge[0]))
    #     ax.set_ylim(*ylim)
    #     ax.legend()
    #     fig.show()
    #     return

    def get_results(self):
        return {
            "time": self._time * 1e3,
            "ne": self._densities[:, 0],
            "nn": self._densities[:, 1],
            "n_beam": self._densities[:, 2],
            "Te": self._temperatures[:, 0],
            "Ti": self._temperatures[:, 1],
            "Ne_flux": self._density_terms[:, 0],
            "Nn_flux": self._density_terms[:, 1],
            "S_ion_bulk": self._density_terms[:, 2],
            "S_rec_rad": self._density_terms[:, 3],
            "S_rec_3b": self._density_terms[:, 4],
            "S_ion_beam": self._density_terms[:, 5],
            "e_par_flux": self._heat_terms[:, 0],
            "i_par_flux": self._heat_terms[:, 1],
            "e_perp_hl": self._heat_terms[:, 2],
            "i_perp_hl": self._heat_terms[:, 3],
            "Qie": self._heat_terms[:, 4],
            "Qei": self._heat_terms[:, 5],
            "Qen": self._heat_terms[:, 6],
            "Qcx": self._heat_terms[:, 7],
            "Qeb": self._heat_terms[:, 8],
            "div_v_elec": self._heat_terms[:, 9],
            "div_v_ions": self._heat_terms[:, 10],
            "v_plasma": self._velocities[:, 0],
            "isat": self._synthetic,
            "primary_mfp": self._primary_mfp,
            "bulk_mfp": self._bulk_mfp,
            "ln_lambda": self._ln_lambda,
        }

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
