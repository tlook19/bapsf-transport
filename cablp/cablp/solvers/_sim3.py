import tomllib
import numpy as np
from cablp.funcs._heat import (
    IAEA_exp1,
    IAEA_exp4,
    elec_par_heat_loss,
    ion_par_heat_loss,
    elec_perp_heat_loss,
    ion_perp_heat_loss,
    Q_cx_He,
    Q_ie,
)
from cablp.vars._cons import (
    qe_SI,
    I_ion,
    en_factor,
    drag_factor,
    m_He_cgs,
    ev_to_erg,
    m_p_cgs,
)
from cablp.funcs._fits import rate_coeff
from cablp.funcs._cross import alpha_3, alpha_r, He_EII_cross
from cablp.vars._coeff import aHeI, aHeII, a_11s
from cablp.vars._cons import I_ion, I_Ry
from cablp.funcs._plasmaparams import v_ion_speed, time_elec_coll, c_log

fit_coeff = [1.3950030050791237e-05, 13.62996440158007]

input_dict_template = {
    "gas_type": "He",
    "ne0": 1e9,
    "Tn_fit": 0.1,  # Neutral temperature for reaction rate fits
    "nn0": 5e12,
    "Source_nn0": 2e13,
    "Twin_nn0": 2e13,
    "Te0": 0.1,
    "Ti0": 0.1,
    "Bz0": 1500,  # Magnetic field in gauss
    "Lm": 1800,  # Length of machine
    "Rm": 50,  # Machine radius
    "Lp": 1800,  # Length of plasma
    "Rp": 18,  # Plasma radius
    "Rhf": 50,  # Gradient scale length of radial heat flux
    "Vd": 100,  # Discharge voltage
    "Twin_Vd": 100,
    "Id": 2500,  # Discharge current
    "Twin_Id": 2500,
    "anode_transparency": 1,  # Anode transparency
    "S_gp": 2000,  # Gas puff source rate
    "Twin_S_gp": 2000,
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
    "dt_main": 3e-8,
    "end": 21e-3,
    "dt_after": 1e-7,
    "cells": 3,
}

input_flags_template = {
    "eperp": True,
    "iperp": True,
    "icool": True,
    "ncool": True,
    "cx": True,
    "mit_el": False,
    "C_imp": False,
    "O_imp": False,
    "icool_recomb": False,
    "Plasma": True,
    "TwinCathode": False,
    "Velocity": False,
    "breakdown_vel": True,  # Use diffusive flux during breakdown; set False to test without
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
            self._I_ion = I_ion
        elif self._gas_type == "H":
            self._m_gas = m_p_cgs
            self._mu = 1
            self._I_ion = I_Ry
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
        self._V_discharge = np.zeros(self._cells)
        self._V_discharge[0] = input_dict.get("Vd", 0)
        self._I_discharge = np.zeros(self._cells)
        self._I_discharge[0] = input_dict.get("Id", 0)
        self._tau_I_on = input_dict.get("tau_I_on", 0.0001)
        self._P_discharge = self._V_discharge * self._I_discharge
        self._mit_el_temp = input_dict.get("mit_el_temp", 1)
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
            self._V_discharge[-1] = input_dict["Twin_Vd"]
            self._I_discharge[-1] = input_dict["Twin_Id"]
            self._S_gp[-1] = self.puff_rate(
                input_dict["Twin_S_gp"], 2, self._cell_vol[-1]
            )
        self._active_discharge = self._I_discharge > 0
        self._v_beam = np.sqrt(2 * self._V_discharge * 1.60218e-12 / 6.6464731e-24)
        eps = self._V_discharge / self._I_ion
        self._beam_cross = np.zeros(self._cells)
        if self._gas_type == "He":
            self._beam_cross[0] = float(He_EII_cross(eps[0], a_11s))
            if self._flags["TwinCathode"]:
                self._beam_cross[-1] = float(He_EII_cross(eps[-1], a_11s))
        self._n_beam = np.zeros(self._cells)
        self._I_beam = np.zeros(self._cells)
        self._J_beam = np.zeros(self._cells)
        self._anode_trans = input_dict.get("anode_transparency", 1.0)
        self._cycles = input_dict.get("cycles", 1)
        self._cyclelength = 0
        self._d_off = input_dict.get("d_off", 20e-3)
        self._dt_main = input_dict.get("dt_main", 1e-3)
        self._end = input_dict.get("end", 3)
        self._dt_after = input_dict.get("dt_after", 1e-3)
        self._ion_multiplier = 1  # self._V_discharge % self._I_ion
        self._ion_remainder = self._V_discharge - self._ion_multiplier * self._I_ion
        self._v_p_bound = np.zeros(self._cells + 1)
        self._Te_bound = np.zeros(self._cells + 1)
        self._Ti_bound = np.zeros(self._cells + 1)
        self._ne_bound = np.zeros(self._cells + 1)
        self._nn_bound = np.zeros(self._cells + 1)
        self._div_v_elec = np.zeros(self._cells)
        self._div_v_ions = np.zeros(self._cells)
        # self._ln_lambda = c_log(self._Te, self._ne)
        self._cathode_factor = 0.60653

    def set_time_steps(self):
        if self._d_off == 0:
            self._i_d_off = 0
            self._afterlglow = True
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
                self._afterlglow = True
                self._dt_afterglow = self._dt_after
            else:
                self._afterlglow = False
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
        self._Qbeam = np.zeros(self._cells)  # Electron heating due to beam
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
        self._densities = np.zeros((self._cycles * self._cyclelength, 3, self._cells))
        self._densities[0] = np.array([self._ne[:], self._nn[:], self._n_beam[:]])
        self._temperatures = np.zeros(
            (self._cycles * self._cyclelength, 2, self._cells)
        )
        self._temperatures[0] = np.array([self._Te[:], self._Ti[:]])
        self._density_terms = np.zeros(
            (self._cycles * self._cyclelength, 6, self._cells)
        )
        self._heat_terms = np.zeros((self._cycles * self._cyclelength, 12, self._cells))
        self._velocities = np.zeros((self._cycles * self._cyclelength, 1, self._cells))
        # self._velocity_terms = np.zeros((self._cycles * self._cyclelength, 3))
        self._synthetic = np.zeros(
            (self._cycles * self._cyclelength, 3)
        )  # Placeholder for synthetic diagnostics

    def update_results(self, i, j):
        self._densities[i + j * self._cyclelength] = np.array(
            [self._ne[:], self._nn[:], self._n_beam[:]]
        )
        self._temperatures[i + j * self._cyclelength] = np.array(
            [self._Te[:], self._Ti[:]]
        )
        self._density_terms[i + j * self._cyclelength] = np.array(
            [
                self._Ne_flux[:],
                self._Nn_flux[:],
                self._S_ion_bulk[:],
                self._S_rec_rad[:],
                self._S_rec_3b[:],
                self._S_ion_beam[:],
            ]
        )
        self._heat_terms[i + j * self._cyclelength] = np.array(
            [
                self._e_par_flux[:],
                self._i_par_flux[:],
                self._e_perp_hl[:],
                self._i_perp_hl[:],
                self._Qie[:],
                self._Qei[:],
                self._Qen[:],
                self._Qcx[:],
                self._Qeb[:],
                self._Qbeam[:],
                self._div_v_elec[:],
                self._div_v_ions[:],
            ]
        )
        self._velocities[i + j * self._cyclelength] = np.array([self._v_plasma[:]])
        self._synthetic[i + j * self._cyclelength] = self._ne * np.sqrt(self._Te)
        # Placeholder for synthetic diagnostics

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
            self._p_beam = self._calc_beam_prob(Te, ne, nn)
            if self._gas_type == "He":
                S_ion_bulk = (
                    self._b_ioniz * ne * nn * rate_coeff(Te, self._I_ion, *fit_coeff)
                )
                S_ion_beam[0] = (
                    self._A_ion_beam[0] * self._p_beam[0] * self._ion_multiplier
                )
                if self._flags["TwinCathode"]:
                    S_ion_beam[-1] = (
                        self._A_ion_beam[-1] * self._p_beam[-1] * self._ion_multiplier
                    )
            S_rec_rad = self._b_rec_rad * ne * ne * alpha_r(Te)
            S_rec_3b = self._b_rec_3b * ne * ne * ne * alpha_3(Te)
        elif not self._flags["Plasma"]:
            self._p_beam = np.zeros(self._cells)
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
            if self._discharge_on:
                Ne_flux[0] *= self._cathode_factor
                if self._flags["TwinCathode"]:
                    Ne_flux[-1] *= self._cathode_factor
            Nn_flux[0] = -Ne_flux[0] * self._Rsq_ratio[0]
            Nn_flux[-1] = -Ne_flux[-1] * self._Rsq_ratio[-1]
        denom = self._L_partflux[:-1] + self._L_partflux[1:]
        if self._flags["Plasma"]:
            if self._flags["Velocity"] and not (self._breakdown and self._flags["breakdown_vel"]):
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
        p_beam = np.zeros(self._cells)
        p_beam[0] = (
            self._beam_cross[0]
            * nn[0]
            / (
                self._beam_cross[0] * nn[0]
                + (
                    1
                    / (
                        self._v_beam[0]
                        * time_elec_coll(Te[0], ne[0], self._ln_lambda[0])
                    )
                )
            )
        )
        if self._flags["TwinCathode"]:
            p_beam[-1] = (
                self._beam_cross[-1]
                * nn[-1]
                / (
                    self._beam_cross[-1] * nn[-1]
                    + (
                        1
                        / (
                            self._v_beam[-1]
                            * time_elec_coll(Te[-1], ne[-1], self._ln_lambda[-1])
                        )
                    )
                )
            )
        return p_beam

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
        Qbeam = np.zeros(self._cells)
        div_v_elec = np.zeros(self._cells)
        div_v_ions = np.zeros(self._cells)
        Qie = self._b_Qie * en_factor * Q_ie(Te, Ti, ne, self._mu, self._ln_lambda)
        if self._flags["Velocity"]:
            # NOTE: physical form of div(v) term should be revisited.
            div_v = self._calc_div_v(Te, v_plasma)
            div_v_elec = -en_factor * Te * div_v
            div_v_ions = -en_factor * Ti * div_v
        if self._gas_type == "He":
            if self._flags["icool"]:
                Qei = np.array(
                    [
                        self._b_Qei
                        * en_factor
                        * IAEA_exp4(Te[i], aHeII, recomb=self._flags["icool_recomb"])
                        * ne[i]
                        for i in range(self._cells)
                    ]
                )
            if self._flags["ncool"]:
                Qen = np.array(
                    [
                        self._b_Qen * en_factor * IAEA_exp1(Te[i], aHeI) * nn[i]
                        for i in range(self._cells)
                    ]
                )
            if self._flags["cx"]:
                Qcx = self._b_Qcx * en_factor * Q_cx_He(ne, nn, Ti, self._Tn_fit)
        if self._discharge_on:
            Qeb[0] = (
                en_factor
                * self._I_beam[0]
                * self._anode_trans
                * self._V_discharge[0]
                / self._plasma_vol[0]
                / qe_SI
                / ne[0]
                # - en_factor
                # * self._p_beam[0]
                # * self._A_ion_beam[0]
                # * self._ion_multiplier
                # * self._I_ion
            )
            if self._flags["TwinCathode"]:
                Qeb[-1] = (
                    en_factor
                    * self._I_beam[-1]
                    * self._anode_trans
                    * self._V_discharge[-1]
                    / self._plasma_vol[-1]
                    / qe_SI
                    / ne[-1]
                    # - en_factor
                    # * self._p_beam[-1]
                    # * self._A_ion_beam[-1]
                    # * self._ion_multiplier
                    # * self._I_ion
                )
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
            Qbeam,
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
        if self._discharge_on:
            e_par_flux[0] -= (
                self._I_beam[0] * Te[0] / self._plasma_vol[0] / qe_SI / ne[0]
            )
            if not self._flags["TwinCathode"]:
                e_par_flux[-1] -= e_par_hl[-1]
            elif self._flags["TwinCathode"]:
                e_par_flux[-1] -= (
                    self._I_beam[-1] * Te[-1] / self._plasma_vol[-1] / qe_SI / ne[-1]
                )
        elif not self._discharge_on:
            e_par_flux[0] -= e_par_hl[0]
            e_par_flux[-1] -= e_par_hl[-1]
        i_par_flux = np.zeros(self._cells)
        i_par_flux[0] -= i_par_hl[0]
        i_par_flux[-1] -= i_par_hl[-1]
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
            div_v[0] = (v_face[0] - c_s[0]) / self._L_plasma[0]
            div_v[-1] = (c_s[-1] - v_face[-1]) / self._L_plasma[-1]
        # Interior cells: central difference of face velocities (vectorized)
        div_v[1:-1] = (v_face[1:] - v_face[:-1]) / self._L_plasma[1:-1]
        return div_v

    def calc_boundary_terms(self, ne, nn, Te, Ti, v_plasma):
        # DEPRECATED: no longer called from _dstep. Boundary values are now computed
        # inline in _calc_n_flux and _calc_pres_acc. Kept for reference.
        c_sound = v_ion_speed(Te, self._mu)
        self._v_p_bound[0] = -c_sound[0]
        self._v_p_bound[-1] = c_sound[-1]
        self._Te_bound[0] = Te[0]
        self._Te_bound[-1] = Te[-1]
        self._Ti_bound[0] = Ti[0]
        self._Ti_bound[-1] = Ti[-1]
        self._ne_bound[0] = ne[0]
        self._ne_bound[-1] = ne[-1]
        self._nn_bound[0] = nn[0]
        self._nn_bound[-1] = nn[-1]
        for k in range(1, self._cells):
            self._v_p_bound[k] = (v_plasma[k - 1] + v_plasma[k]) / 2
            self._Te_bound[k] = (Te[k - 1] + Te[k]) / 2
            self._Ti_bound[k] = (Ti[k - 1] + Ti[k]) / 2
            self._ne_bound[k] = (ne[k - 1] + ne[k]) / 2
            self._nn_bound[k] = (nn[k - 1] + nn[k]) / 2

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
            self._A_ion_beam = self._n_beam_ion * nn
            self._ln_lambda = c_log(Te, ne)
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
                Qbeam,
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
            d_Te = Qeb + Qbeam - Qie - Qei - Qen + e_par_flux - e_perp_hl + div_v_elec
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
        k1_a = a + (0.5 * self._h * k1)
        k2 = self._dstep(k1_a)
        k2_a = a + (0.5 * self._h * k2)
        k3 = self._dstep(k2_a)
        k3_a = a + (self._h * k3)
        k4 = self._dstep(k3_a)
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
            self._A_ion_beam = self._n_beam_ion * nn
            self._ln_lambda = c_log(Te, ne)
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
        return ne, nn, Te, Ti, v_plasma

    def start_simulation(self):
        self.set_time_steps()
        self.initialize_results()
        print("Starting simulation...")
        t = self._cyclelength
        for j in range(self._cycles):
            print(f"Starting cycle {j+1}/{self._cycles}...")
            self._I_beam = np.zeros(self._cells)
            self._J_beam = np.zeros(self._cells)
            self._n_beam = np.zeros(self._cells)
            self._n_beam_ion = np.zeros(self._cells)
            for i, time in enumerate(self._onetime):
                if i < self._i_d_off and self._flags["Plasma"]:
                    self._discharge_on = True
                    self._h = self._dt_discharge
                    self._I_beam = self._I_discharge * (
                        1 - np.exp(-time / self._tau_I_on)
                    )
                    self._breakdown = True
                    if self._active_discharge.any() and np.all(
                        self._I_beam[self._active_discharge]
                        >= self._I_discharge[self._active_discharge]
                    ):
                        self._I_beam = self._I_discharge
                        self._breakdown = False
                    self._J_beam[0] = self._I_beam[0] / self._plasma_cross[0] / qe_SI
                    self._n_beam[0] = self._J_beam[0] / self._v_beam[0]
                    self._n_beam_ion[0] = (
                        self._n_beam[0] * self._beam_cross[0] * self._v_beam[0]
                    )
                    if self._flags["TwinCathode"]:
                        self._J_beam[-1] = (
                            self._I_beam[-1] / self._plasma_cross[-1] / qe_SI
                        )
                        self._n_beam[-1] = self._J_beam[-1] / self._v_beam[-1]
                        self._n_beam_ion[-1] = (
                            self._n_beam[-1] * self._beam_cross[-1] * self._v_beam[-1]
                        )
                elif i < self._i_d_off and not self._flags["Plasma"]:
                    self._discharge_on = True
                    self._h = self._dt_discharge
                elif self._afterlglow:
                    self._discharge_on = False
                    self._h = self._dt_afterglow
                    if i == self._i_d_off:  # zero beam quantities on discharge→afterglow transition
                        self._I_beam[:] = 0
                        self._J_beam[:] = 0
                        self._n_beam[:] = 0
                        self._n_beam_ion[:] = 0
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
                (
                    self._ne,
                    self._nn,
                    self._Te,
                    self._Ti,
                    self._v_plasma,
                ) = self._rk4_step(a)
                self.update_results(i, j)
                if i % 5000 == 0 or i == t - 1:
                    print(f"Step {i+1}/{t}:")
                    print(f"ne= {self._ne}, nn = {self._nn}")
                    print(f"Te = {self._Te}, Ti = {self._Ti}")
                    # print(f"v_plasma = {self._v_plasma}")
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
            "Qbeam": self._heat_terms[:, 9],
            "div_v_elec": self._heat_terms[:, 10],
            "div_v_ions": self._heat_terms[:, 11],
            "v_plasma": self._velocities[:, 0],
            "isat": self._synthetic[:, 0],
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
