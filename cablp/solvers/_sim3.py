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
from matplotlib import pyplot as plt

fit_coeff = [1.3950030050791237e-05, 13.62996440158007]

input_dic_template = {
    "ne0": [1e8, 1e8, 1e8],
    "nn0": [3e13, 1e8, 1e8],
    "Te0": [0.1, 0.1, 0.1],
    "Ti0": [0.1, 0.1, 0.1],
    "Bz0": [800, 800, 800],  # Magnetic field in gauss
    "Lm": [100, 1800, 100],  # Length of machine
    "Rm": [50, 50, 50],  # Machine radius
    "Lp": [100, 1800, 100],  # Length of plasma
    "Rp": [18, 18, 18],  # Plasma radius
    "Lhf": [50, 1000, 50],  # Gradient scale length of axial heat flux
    "Lpf": [50, 1000, 50],  # Gradient scale length of axial particle flux
    "Rhf": [50, 50, 50],  # Gradient scale length radial of heat flux
    "Vd": 120,  # Discharge voltage
    "Id": 2500,  # Discharge current
    "anode_transparency": 0.5,  # Anode transparency
    "S_gp": [1200, 0, 0],  # Gas puff source rate
    "S_pump": [4000, 0, 4000],  # Vacuum pump sink rate
    "mit_el_temp": 0.5,  # reduced edge loss temp
    "Tn": 0.1,  # Neutral temperature
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
    "cycles": 1,  # number of discharge cycles to simulate
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
    "NoPlasma": False,
}


class LAPDSim:
    def __init__(
        self,
        input_dict=input_dic_template,
        input_flags=input_flags_template,
    ):
        self._flags = input_flags
        self._cells = input_dict["cells"] if "cells" in input_dict else 3
        self._gas_type = input_dict["GasType"] if "GasType" in input_flags else "He"
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
        self._nn[0] = (
            input_dict["Source_nn0"] if "Source_nn0" in input_dict else self._nn[0]
        )
        self._Te = np.ones(self._cells) * input_dict["Te0"]
        self._Ti = np.ones(self._cells) * input_dict["Ti0"]
        self._Tn_fit = input_dict["Tn_fit"] if "Tn_fit" in input_dict else 0.1
        self._v_plasma = np.zeros(self._cells)
        self._v_neutral = np.zeros(self._cells)
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
        self._V_discharge[0] = input_dict["Vd"] if "Vd" in input_dict else 0
        self._I_discharge = np.zeros(self._cells)
        self._I_discharge[0] = input_dict["Id"] if "Id" in input_dict else 0
        self._tau_I_on = input_dict["tau_I_on"] if "tau_I_on" in input_dict else 0.0001
        self._P_discharge = self._V_discharge * self._I_discharge
        self._mit_el_temp = (
            input_dict["mit_el_temp"] if "mit_el_temp" in input_dict else 1
        )
        self._Tn = input_dict["Tn"] if "Tn" in input_dict else 0.025
        self._b_epara = input_dict["b_epara"] if "b_epara" in input_dict else 1.0
        self._b_ipara = input_dict["b_ipara"] if "b_ipara" in input_dict else 1.0
        self._b_eperp = input_dict["b_eperp"] if "b_eperp" in input_dict else 1.0
        self._b_iperp = input_dict["b_iperp"] if "b_iperp" in input_dict else 1.0
        self._b_ioniz = input_dict["b_ioniz"] if "b_ioniz" in input_dict else 1.0
        self._b_rec_rad = input_dict["b_rec_rad"] if "b_rec_rad" in input_dict else 1.0
        self._b_rec_3b = input_dict["b_rec_3b"] if "b_rec_3b" in input_dict else 1.0
        self._b_Qcx = input_dict["b_Qcx"] if "b_Qcx" in input_dict else 1.0
        self._b_Qie = input_dict["b_Qie"] if "b_Qie" in input_dict else 1.0
        self._b_Qei = input_dict["b_Qei"] if "b_Qei" in input_dict else 1.0
        self._b_Qen = input_dict["b_Qen"] if "b_Qen" in input_dict else 1.0
        self._b_source = input_dict["b_source"] if "b_source" in input_dict else 1.0
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
        self._anode_trans = (
            input_dict["anode_transparency"]
            if "anode_transparency" in input_dict
            else 1.0
        )
        self._cycles = input_dict["cycles"] if "cycles" in input_dict else 1
        self._cyclelength = 0
        self._d_off = input_dict["d_off"] if "d_off" in input_dict else 20e-3
        self._dt_main = input_dict["dt_main"] if "dt_main" in input_dict else 1e-3
        self._end = input_dict["end"] if "end" in input_dict else 3
        self._dt_after = input_dict["dt_after"] if "dt_after" in input_dict else 1e-3
        self._ion_multiplier = 1  # self._V_discharge % self._I_ion
        self._ion_remainder = self._V_discharge - self._ion_multiplier * self._I_ion
        self._v_p_bound = np.zeros(self._cells + 1)
        self._v_n_bound = np.zeros(self._cells + 1)
        self._Te_bound = np.zeros(self._cells + 1)
        self._Ti_bound = np.zeros(self._cells + 1)
        self._ne_bound = np.zeros(self._cells + 1)
        self._nn_bound = np.zeros(self._cells + 1)
        self._v_pres_plasma = np.zeros(self._cells)
        self._v_pres_neutral = np.zeros(self._cells)
        self._v_conv_plasma = np.zeros(self._cells)
        self._v_conv_neutral = np.zeros(self._cells)
        self._drag_in = np.zeros(self._cells)
        self._div_v_elec = np.zeros(self._cells)
        self._div_v_ions = np.zeros(self._cells)
        self._P_flux_elec = np.zeros(self._cells)
        self._P_flux_ions = np.zeros(self._cells)
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
        self._heat_terms = np.zeros((self._cycles * self._cyclelength, 14, self._cells))
        self._velocities = np.zeros((self._cycles * self._cyclelength, 2, self._cells))
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
                self._P_flux_elec[:],
                self._P_flux_ions[:],
            ]
        )
        self._velocities[i + j * self._cyclelength] = np.array(
            [self._v_plasma[:], self._v_neutral[:]]
        )
        self._synthetic[i + j * self._cyclelength] = self._ne * np.sqrt(self._Te)
        # Placeholder for synthetic diagnostics

    def calc_density_terms(
        self,
        ne,
        nn,
        Te,
    ):
        Ne_flux, Nn_flux = self._calc_n_flux(Te, ne, nn)
        if self._flags["Plasma"]:
            self._p_beam = self._calc_beam_prob(Te, ne, nn)
            if self._gas_type == "He":
                S_ion_bulk = (
                    self._b_ioniz * ne * nn * rate_coeff(Te, self._I_ion, *fit_coeff)
                )
                S_ion_beam = self._A_ion_beam * self._p_beam * self._ion_multiplier
            S_rec_rad = self._b_rec_rad * ne * ne * alpha_r(Te)
            S_rec_3b = self._b_rec_3b * ne * ne * ne * alpha_3(Te)
        elif not self._flags["Plasma"]:
            S_ion_bulk = np.zeros(self._cells)
            S_rec_rad = np.zeros(self._cells)
            S_rec_3b = np.zeros(self._cells)
            S_ion_beam = np.zeros(self._cells)
            self._p_beam = np.zeros(self._cells)
        den_terms = np.array(
            [
                Ne_flux,
                Nn_flux,
                S_ion_bulk,
                S_rec_rad,
                S_rec_3b,
                S_ion_beam,
            ]
        )
        return den_terms

    def _calc_n_flux(self, Te, ne, nn):
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
        if self._flags["Velocity"]:
            if self._flags["Plasma"]:
                Ne_flux[0] = (
                    ne[0]
                    * self._v_p_bound[0]
                    * self._cathode_factor
                    / self._L_partflux[0]
                )
                Ne_flux[-1] = -ne[-1] * self._v_p_bound[-1] / self._L_partflux[-1]
                if self._flags["TwinCathode"]:
                    Ne_flux[-1] *= self._cathode_factor
                Nn_flux[0] = -Ne_flux[0] * self._Rsq_ratio[0]
                Nn_flux[-1] = -Ne_flux[-1] * self._Rsq_ratio[-1]
            for k in range(self._cells):
                if (
                    k < self._cells - 1
                ):  # set right flux surface for every cell but last
                    if self._flags["Plasma"]:  # if theres plasma there is ne flux
                        Ne_flux[k] -= (
                            2
                            * (self._ne_bound[k + 1] * self._v_p_bound[k + 1])
                            / (self._L_partflux[k] + self._L_partflux[k + 1])
                        )
                    Nn_flux[k] -= (
                        2
                        * (self._nn_bound[k + 1] * self._v_n_bound[k + 1])
                        / (self._L_partflux[k] + self._L_partflux[k + 1])
                    )
                if k > 0:  # set left flux surface for every cell but first
                    if self._flags["Plasma"]:
                        Ne_flux[k] += (
                            2
                            * (self._ne_bound[k] * self._v_p_bound[k])
                            / (self._L_partflux[k] + self._L_partflux[k - 1])
                        )
                    Nn_flux[k] += (
                        2
                        * (self._nn_bound[k] * self._v_n_bound[k])
                        / (self._L_partflux[k] + self._L_partflux[k - 1])
                    )
        elif self._flags["Velocity"] == False:
            if self._flags["Plasma"]:
                ne_flux = ne * v_ion_speed(Te, self._mu)
                Ne_flux[0] = (
                    -ne_flux[0] * self._cathode_factor / (2 * self._L_partflux[0])
                )
                Ne_flux[-1] = -ne_flux[-1] / (2 * self._L_partflux[-1])
                if self._flags["TwinCathode"]:
                    Ne_flux[-1] *= self._cathode_factor
                Nn_flux[0] = -Ne_flux[0] * self._Rsq_ratio[0]
                Nn_flux[-1] = -Ne_flux[-1] * self._Rsq_ratio[-1]
            nn_flux = nn * v_ion_speed(self._Tn, self._mu)
            for k in range(self._cells):
                if k < (
                    self._cells - 1
                ):  # set right flux surface for every cell but last
                    if self._flags["Plasma"]:  # if theres plasma there is ne flux
                        Ne_flux[k] += (ne_flux[k + 1] - ne_flux[k]) / (
                            self._L_partflux[k] + self._L_partflux[k + 1]
                        )
                    Nn_flux[k] += (nn_flux[k + 1] - nn_flux[k]) / (
                        self._L_partflux[k] + self._L_partflux[k + 1]
                    )
                if k > 0:  # set left flux surface for every cell but first
                    if self._flags["Plasma"]:
                        Ne_flux[k] += (ne_flux[k - 1] - ne_flux[k]) / (
                            self._L_partflux[k] + self._L_partflux[k - 1]
                        )
                    Nn_flux[k] += (nn_flux[k - 1] - nn_flux[k]) / (
                        self._L_partflux[k] + self._L_partflux[k - 1]
                    )
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

    def _calc_velocity_terms(self, ne, nn, Ti, v_plasma, v_neutral):
        v_conv_plasma = self._calc_v_conv(v_plasma, self._v_p_bound)
        v_conv_neutral = self._calc_v_conv(v_neutral, self._v_n_bound)
        drag_in_plasma, drag_in_neutral = self._calc_drag_in(
            Ti, ne, nn, v_plasma, v_neutral
        )
        v_pres_plasma, v_pres_neutral = self._calc_v_pres(
            ne, nn, self._ne_bound, self._nn_bound, self._Te_bound
        )

        return (
            v_conv_plasma,
            v_conv_neutral,
            drag_in_plasma,
            drag_in_neutral,
            v_pres_plasma,
            v_pres_neutral,
        )

    def _calc_v_conv(self, v, v_bound):
        """
        Calculate the convective velocity term.
        This term should be added to time derivative of the velocity to get the net acceleration.
        It includes contributions from the boundary conditions and the spatial variation of the velocity.

        Parameters
        ----------
        v : _type_
            _description_
        v_bound : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        v_conv = np.zeros(self._cells)
        v_conv[0] = v[0] ** 2 / self._L_partflux[0]
        v_conv[-1] = -v[-1] ** 2 / self._L_partflux[-1]
        for k in range(self._cells):
            v_conv[k] += v[0] * (v_bound[k + 1] - v_bound[k]) / self._L_plasma[k]
            if k < self._cells - 1:
                v_conv[k] -= v[k + 1] ** 2 / (
                    self._L_partflux[k] + self._L_partflux[k + 1]
                )
            if k > 0:
                v_conv[k] += v[k - 1] ** 2 / (
                    self._L_partflux[k] + self._L_partflux[k - 1]
                )
        return v_conv

    def _calc_v_pres(self, ne, nn, ne_bound, nn_bound, Te_bound):
        """
        Calculate the pressure gradient velocity term. This term should be added to time derivative of the velocity to get the net acceleration.

        Parameters
        ----------
        ne : _type_
            _description_
        nn : _type_
            _description_
        ne_bound : _type_
            _description_
        nn_bound : _type_
            _description_
        Te_bound : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        v_p_pres = np.zeros(self._cells)
        v_n_pres = np.zeros(self._cells)
        for k in range(self._cells):
            v_p_pres -= (
                (ne_bound[k + 1] * Te_bound[k + 1] - ne_bound[k] * Te_bound[k])
                / self._L_plasma[k]
                / ne[k]
            )
            v_n_pres -= (
                self._Tn * (nn_bound[k + 1] - nn_bound[k]) / self._L_plasma[k] / nn[k]
            )
        return v_p_pres * ev_to_erg / m_He_cgs, v_n_pres * ev_to_erg / m_He_cgs

    def _calc_drag_in(self, Ti, ne, nn, v_plasma, v_neutral):
        """
        Calculate the ion drag force on the plasma. This term should be multiplied by nn and substracted from the plasma velocity time derivative to get the net acceleration.
        It should be multiplied by ne and added to the neutral velocity time derivative to get the net acceleration of the neutrals.

        Parameters
        ----------
        Ti : _type_
            _description_
        v_plasma : _type_
            _description_
        v_neutral : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        v_thm_i = v_ion_speed(Ti, self._mu)
        v_thm_n = v_ion_speed(self._Tn, self._mu)
        drag_in = drag_factor * (
            v_thm_i * v_plasma - v_thm_n * v_neutral
        )  # does it need v_thm_n?
        return drag_in * nn, drag_in * ne

    def calc_heat_terms(
        self,
        ne,
        nn,
        Te,
        Ti,
        v_plasma,
        v_neutral,
    ):
        e_par_flux, i_par_flux, e_perp_hl, i_perp_hl = self._calc_cond_heat_flux(
            Te, Ti, ne
        )
        Qei = np.zeros(self._cells)
        Qen = np.zeros(self._cells)
        Qcx = np.zeros(self._cells)
        Qeb = np.zeros(self._cells)
        Qbeam = np.zeros(self._cells)
        P_flux_elec = np.zeros(self._cells)
        P_flux_ions = np.zeros(self._cells)
        div_v_elec = np.zeros(self._cells)
        div_v_ions = np.zeros(self._cells)
        Qie = self._b_Qie * en_factor * Q_ie(Te, Ti, ne, self._mu, self._ln_lambda)
        if self._flags["Velocity"]:
            P_flux_elec, P_flux_ions, div_v_elec, div_v_ions = (
                self._calc_conv_heat_flux(Te, Ti, ne, nn, v_plasma, v_neutral)
            )
        if self._gas_type == "He":
            if self._flags["icool"]:
                Qei = np.array(
                    [
                        self._b_Qei
                        * en_factor
                        * IAEA_exp4(Te[i], aHeII, recomb=self._flags["icool_recomb"])
                        * ne[i]
                        for i in range(3)
                    ]
                )
            if self._flags["ncool"]:
                Qen = np.array(
                    [
                        self._b_Qen * en_factor * IAEA_exp1(Te[i], aHeI) * nn[i]
                        for i in range(3)
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
        heat_terms = np.array(
            [
                e_par_flux[:],
                i_par_flux[:],
                e_perp_hl[:],
                i_perp_hl[:],
                Qie[:],
                Qei[:],
                Qen[:],
                Qeb[:],
                Qcx[:],
                Qbeam[:],
                div_v_elec[:],
                div_v_ions[:],
                P_flux_elec[:],
                P_flux_ions[:],
            ]
        )
        return heat_terms

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
        e_par_flux[0] -= self._I_beam[0] * Te[0] / self._plasma_vol[0] / qe_SI / ne[0]
        i_par_flux = np.zeros(self._cells)
        # e_par_flux[0] -= e_par_hl[0]
        if not self._flags["TwinCathode"]:
            e_par_flux[-1] -= e_par_hl[-1]
        elif self._flags["TwinCathode"]:
            e_par_flux[-1] -= (
                self._I_beam[-1] * Te[-1] / self._plasma_vol[-1] / qe_SI / ne[-1]
            )
        # print("e_par_flux[0]: ", e_par_flux[0], " e_par_flux[-1]: ", e_par_flux[-1])
        i_par_flux[0] -= i_par_hl[0]
        i_par_flux[-1] -= i_par_hl[-1]
        for k in range(self._cells):
            if k < self._cells - 1:
                e_par_flux[k] += (
                    e_par_hl[k + 1] * self._L_heatflux[k]
                    - e_par_hl[k] * self._L_heatflux[k]
                ) / (self._L_heatflux[k] + self._L_heatflux[k + 1])
                i_par_flux[k] += (
                    i_par_hl[k + 1] * self._L_heatflux[k + 1]
                    - i_par_hl[k] * self._L_heatflux[k]
                ) / (self._L_heatflux[k] + self._L_heatflux[k + 1])
            if k > 0:
                e_par_flux[k] += (
                    e_par_hl[k - 1] * self._L_heatflux[k - 1]
                    - e_par_hl[k] * self._L_heatflux[k]
                ) / (self._L_heatflux[k - 1] + self._L_heatflux[k])
                i_par_flux[k] += (
                    i_par_hl[k - 1] * self._L_heatflux[k - 1]
                    - i_par_hl[k] * self._L_heatflux[k]
                ) / (self._L_heatflux[k - 1] + self._L_heatflux[k])
        return e_par_flux, i_par_flux, e_perp_hl, i_perp_hl

    def _calc_conv_heat_flux(self, Te, Ti, v_plasma, v_neutral):
        P_flux_elec = np.zeros(self._cells)
        P_flux_ions = np.zeros(self._cells)
        div_v_elec = np.zeros(self._cells)
        div_v_ions = np.zeros(self._cells)

        for k in range(self._cells):
            P_flux_elec[k] = self._L_heatflux[k] * (
                v_plasma[k] * Te[k] - v_neutral[k] * Ti[k]
            )
            P_flux_ions[k] = self._L_heatflux[k] * (
                v_plasma[k] * Ti[k] - v_neutral[k] * Te[k]
            )
            div_v_elec[k] = self._L_heatflux[k] * (v_plasma[k] - v_neutral[k])
            div_v_ions[k] = self._L_heatflux[k] * (v_plasma[k] - v_neutral[k])
            # P_flux_elec[0] = self._Te_bound[0] * self._v_p_bound[0] / self._L_heatflux[
            #     0
            # ] - 2 * self._Te_bound[1] * self._v_p_bound[1] / (
            #     self._L_heatflux[0] + self._L_heatflux[1]
            # )
            # P_flux_elec[1] = 2 * self._Te_bound[1] * self._v_p_bound[1] / (
            #     self._L_heatflux[0] + self._L_heatflux[1]
            # ) - 2 * self._Te_bound[2] * self._v_p_bound[2] / (
            #     self._L_heatflux[1] + self._L_heatflux[2]
            # )
            # P_flux_elec[2] = (
            #     2
            #     * self._Te_bound[2]
            #     * self._v_p_bound[2]
            #     / (self._L_heatflux[1] + self._L_heatflux[2])
            #     - self._Te_bound[3] * self._v_p_bound[3] / self._L_heatflux[2]
            # )
            # P_flux_ions[0] = self._Ti_bound[0] * self._v_p_bound[0] / self._L_heatflux[
            #     0
            # ] - 2 * self._Ti_bound[1] * self._v_p_bound[1] / (
            #     self._L_heatflux[0] + self._L_heatflux[1]
            # )
            # P_flux_ions[1] = 2 * self._Ti_bound[1] * self._v_p_bound[1] / (
            #     self._L_heatflux[0] + self._L_heatflux[1]
            # ) - 2 * self._Ti_bound[2] * self._v_p_bound[2] / (
            #     self._L_heatflux[1] + self._L_heatflux[2]
            # )
            # P_flux_ions[2] = (
            #     2
            #     * self._Ti_bound[2]
            #     * self._v_p_bound[2]
            #     / (self._L_heatflux[1] + self._L_heatflux[2])
            #     - self._Ti_bound[3] * self._v_p_bound[3] / self._L_heatflux[2]
            # )
            # div_v_elec[0] = (
            #     en_factor
            #     * Te[0]
            #     * (self._v_p_bound[1] - self._v_p_bound[0])
            #     / self._L_plasma[0]
            # )
            # div_v_elec[1] = (
            #     en_factor
            #     * Te[1]
            #     * (self._v_p_bound[2] - self._v_p_bound[1])
            #     / self._L_plasma[1]
            # )
            # div_v_elec[2] = (
            #     en_factor
            #     * Te[2]
            #     * (self._v_p_bound[3] - self._v_p_bound[2])
            #     / self._L_plasma[2]
            # )
            # div_v_ions[0] = (
            #     en_factor
            #     * Ti[0]
            #     * (self._v_p_bound[1] - self._v_p_bound[0])
            #     / self._L_plasma[0]
            # )
            # div_v_ions[1] = (
            #     en_factor
            #     * Ti[1]
            #     * (self._v_p_bound[2] - self._v_p_bound[1])
            #     / self._L_plasma[1]
            # )
            # div_v_ions[2] = (
            #     en_factor
            #     * Ti[2]
            #     * (self._v_p_bound[3] - self._v_p_bound[2])
            #     / self._L_plasma[2]
            # )
        return P_flux_elec, P_flux_ions, div_v_elec, div_v_ions

    def calc_boundary_terms(self, ne, nn, Te, Ti, v_plasma, v_neutral):
        c_sound = v_ion_speed(Te, self._mu)
        self._v_p_bound[0] = -c_sound[0]
        self._v_p_bound[-1] = c_sound[-1]
        self._Te_bound[0] = Te[0]
        self._Te_bound[-1] = Te[-1]
        self._v_n_bound[0] = -self._v_p_bound[0]
        self._v_n_bound[-1] = -self._v_p_bound[-1]
        self._Ti_bound[0] = Ti[0]
        self._Ti_bound[-1] = Ti[-1]
        self._ne_bound[0] = ne[0]
        self._ne_bound[-1] = ne[-1]
        self._nn_bound[0] = nn[0]
        self._nn_bound[-1] = nn[-1]
        for k in range(1, self._cells):
            self._v_p_bound[k] = (v_plasma[k - 1] + v_plasma[k]) / 2
            self._v_n_bound[k] = (v_neutral[k - 1] + v_neutral[k]) / 2
            self._Te_bound[k] = (Te[k - 1] + Te[k]) / 2
            self._Ti_bound[k] = (Ti[k - 1] + Ti[k]) / 2
            self._ne_bound[k] = (ne[k - 1] + ne[k]) / 2
            self._nn_bound[k] = (nn[k - 1] + nn[k]) / 2

    def _dstep(self, a):
        ne, nn, Te, Ti, v_plasma, v_neutral = a
        if self._flags["Plasma"]:
            self._A_ion_beam = self._n_beam_ion * nn
            self._ln_lambda = c_log(Te, ne)
        self.calc_boundary_terms(ne, nn, Te, Ti, v_plasma, v_neutral)
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
        )
        if self._discharge_on:
            puff = self._S_gp[:]
        else:
            puff = np.zeros(3)
        if self._flags["Velocity"]:
            (
                v_conv_plasma,
                v_conv_neutral,
                drag_in_plasma,
                drag_in_neutral,
                v_pres_plasma,
                v_pres_neutral,
            ) = self._calc_velocity_terms(
                ne,
                nn,
                Te,
                Ti,
                v_plasma,
                v_neutral,
            )
        elif self._flags["Velocity"] == False:
            v_conv_plasma = np.zeros(self._cells)
            v_conv_neutral = np.zeros(self._cells)
            drag_in_plasma = np.zeros(self._cells)
            drag_in_neutral = np.zeros(self._cells)
            v_pres_plasma = np.zeros(self._cells)
            v_pres_neutral = np.zeros(self._cells)
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
                P_flux_elec,
                P_flux_ions,
            ) = self.calc_heat_terms(
                ne,
                nn,
                Te,
                Ti,
                v_plasma,
                v_neutral,
            )
            d_ne = S_ion_bulk + S_ion_beam - S_rec_rad - S_rec_3b + Ne_flux
            d_Te = (
                Qeb
                + Qbeam
                - Qie
                - Qei
                - Qen
                + e_par_flux
                - e_perp_hl
                - div_v_elec
                - P_flux_elec
            )
            d_Ti = Qie + i_par_flux - i_perp_hl - Qcx - div_v_ions - P_flux_ions
        elif self._flags["Plasma"] == False:
            d_ne = np.zeros(self._cells)
            d_Te = np.zeros(self._cells)
            d_Ti = np.zeros(self._cells)
        d_nn = (
            Nn_flux
            - S_ion_bulk * self._Rsq_ratio
            - S_ion_beam * self._Rsq_ratio
            + S_rec_rad * self._Rsq_ratio
            + S_rec_3b * self._Rsq_ratio
            + puff
            - (self._S_pump * nn)
        )
        d_ve = v_conv_plasma + v_pres_plasma - drag_in_plasma
        d_vn = v_conv_neutral + v_pres_neutral + drag_in_neutral

        return np.array([d_ne, d_nn, d_Te, d_Ti, d_ve, d_vn])

    def _rk4_step(self, a):
        # print("RK4 step with h =", self._h)
        k1 = self._dstep(a).copy()
        k1_a = a + (0.5 * self._h * k1)
        k2 = self._dstep(k1_a).copy()
        k2_a = a + (0.5 * self._h * k2)
        k3 = self._dstep(k2_a).copy()
        k3_a = a + (self._h * k3)
        k4 = self._dstep(k3_a).copy()
        # print("k: ", k1, k2, k3, k4)
        b = a + (self._h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        # print("b: ", b)
        ne, nn, Te, Ti, v_plasma, v_neutral = b
        if self._flags["Plasma"]:
            Ti[Ti < 0.01] = 0.01
            Te[Te < 0.01] = 0.01
            Ti[Ti > 100] = 100
            Te[Te > 100] = 100
            ne[ne < 1e8] = 1e8
            nn[nn < 1e8] = 1e8
        return ne, nn, Te, Ti, v_plasma, v_neutral

    def start_simulation(self):
        self.set_time_steps()
        if not self._time_flag:
            raise ValueError("Time steps not set. Use set_time_steps() method.")
        self.initialize_results()
        print("Starting simulation...")
        t = self._cyclelength
        for j in range(self._cycles):
            print(f"Starting cycle {j+1}/{self._cycles}...")
            for i, time in enumerate(self._onetime):
                self._I_beam = np.zeros(self._cells)
                self._J_beam = np.zeros(self._cells)
                self._n_beam = np.zeros(self._cells)
                self._n_beam_ion = np.zeros(self._cells)
                if i < self._i_d_off and self._flags["Plasma"]:
                    self._discharge_on = True
                    self._h = self._dt_discharge
                    if self._I_beam.any() < self._I_discharge.any():
                        self._I_beam = self._I_discharge * (
                            1 - np.exp(-time / self._tau_I_on)
                        )
                    else:
                        self._I_beam = self._I_discharge
                    self._J_beam = self._I_beam / self._plasma_cross / qe_SI
                    self._n_beam[0] = self._J_beam[0] / self._v_beam[0]
                    self._n_beam_ion[0] = (
                        self._n_beam[0] * self._beam_cross[0] * self._v_beam[0]
                    )
                    if self._flags["TwinCathode"]:
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
                else:
                    break

                a = np.array(
                    [
                        self._ne[:],
                        self._nn[:],
                        self._Te[:],
                        self._Ti[:],
                        self._v_plasma[:],
                        self._v_neutral[:],
                    ]
                )
                (
                    self._ne,
                    self._nn,
                    self._Te,
                    self._Ti,
                    self._v_plasma,
                    self._v_neutral,
                ) = self._rk4_step(a)
                (
                    self._Ne_flux[:],
                    self._Nn_flux[:],
                    self._S_ion_bulk[:],
                    self._S_rec_rad[:],
                    self._S_rec_3b[:],
                    self._S_ion_beam[:],
                ) = self.calc_density_terms(
                    self._ne[:],
                    self._nn[:],
                    self._Te[:],
                )
                if self._flags["Plasma"]:
                    (
                        self._e_par_flux[:],
                        self._i_par_flux[:],
                        self._e_perp_hl[:],
                        self._i_perp_hl[:],
                        self._Qie[:],
                        self._Qei[:],
                        self._Qen[:],
                        self._Qeb[:],
                        self._Qcx[:],
                        self._Qbeam[:],
                        self._div_v_elec[:],
                        self._div_v_ions[:],
                        self._P_flux_elec[:],
                        self._P_flux_ions[:],
                    ) = self.calc_heat_terms(
                        self._ne,
                        self._nn,
                        self._Te,
                        self._Ti,
                        self._v_plasma,
                        self._v_neutral,
                    )
                self.update_results(i, j)
                if i % 5000 == 0 or i == t - 1:
                    print(f"Step {i+1}/{t}:")
                    print(f"ne= {self._ne}, nn = {self._nn}")
                    print(f"Te = {self._Te}, Ti = {self._Ti}")
                    # print(f"v_plasma = {self._v_plasma}, v_neutral = {self._v_neutral}")
                    # print(
                    #     f"v_pres_plasma = {self._v_pres_plasma}, v_pres_neutral = {self._v_pres_neutral}"
                    # )
                    # print(
                    #     f"v_conv_plasma = {self._v_conv_plasma}, v_conv_neutral = {self._v_conv_neutral}"
                    # )
                    # print(f"v_drag_in = {self._drag_in}")
        print("Simulation complete.")

    def heat_balance(self, ylim=(-1e6, 1e6), discharge=True):
        if discharge:
            self._discharge_on = True
        else:
            self._discharge_on = False
        Ti = self._Ti0
        temps = np.arange(0.1, 20, 0.1)
        densities = np.logspace(11, 13, num=5)
        curves = np.empty((densities.shape[0], temps.shape[0]))
        for i, ne in enumerate(densities):
            for j, Te in enumerate(temps):
                e_par_hl, e_perp_hl, Qie, Qei, Qen, Qeb, i_par_hl, i_perp_hl, Qcx = (
                    self.calc_heat_terms(ne, self._nn0, Te, Ti)
                )
                total = (
                    Qeb
                    - Qie
                    - Qei
                    - Qen
                    - e_par_hl
                    - e_perp_hl
                    - i_par_hl
                    - i_perp_hl
                    - Qcx
                )
                curves[i, j] = total
        fig, ax = plt.subplots()
        for i, ne in enumerate(densities):
            ax.plot(temps, curves[i], label=f"ne={ne:.1e} cm^-3")
        ax.axhline(0, color="k", linestyle="--")
        ax.set_xlabel("Electron Temperature (eV)")
        ax.set_ylabel("Net Heating/Cooling (eV/s)")
        ax.set_title("Heat Balance Curves @ P = {:.1f} W".format(self._P_discharge))
        ax.set_ylim(*ylim)
        ax.legend()
        fig.show()
        return

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
            "P_flux_elec": self._heat_terms[:, 12],
            "P_flux_ions": self._heat_terms[:, 13],
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
