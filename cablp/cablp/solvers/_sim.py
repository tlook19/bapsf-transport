# DEPRECATED: Superseded by _sim3.py / LAPDSim. Kept for reference only.
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
from cablp.vars._cons import qe_SI, I_ion, en_factor
from cablp.funcs._fits import rate_coeff
from cablp.funcs._cross import alpha_3, alpha_r, He_EII_cross
from cablp.vars._coeff import aHeI, aHeII, a_11s
from cablp.vars._cons import I_ion
from cablp.funcs._plasmaparams import v_ion_speed, time_elec_coll
from matplotlib import pyplot as plt

fit_coeff = [1.3950030050791237e-05, 13.62996440158007]

input_dic_template = {
    "ne0": 1e11,
    "nn0": 2e13,
    "Te0": 0.1,
    "Ti0": 0.1,
    "Bz0": 800,  # Magnetic field in gauss
    "Lm": 1800,  # Length of machine
    "Rm": 50,  # Machine radius
    "Lp": 1800,  # Length of plasma
    "Rp": 18,  # Plasma radius
    "Lhf": 1000,  # Gradient scale length of axial heat flux
    "Lpf": 1800,  # Gradient scale length of axial particle flux
    "Rhf": 50,  # Gradient scale length radial of heat flux
    "Vd": 120,  # Discharge voltage
    "Id": 2500,  # Discharge current
    "S_gp": 0,  # Gas puff source rate
    "S_pump": 0,  # Vacuum pump sink rate
    "Tn": 0.1,  # Neutral temperature
    "tau_I_on": 0.001,  # Time constant for beam current rise
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
}


class LAPDSim:
    def __init__(
        self,
        input_dict=input_dic_template,
        input_flags=input_flags_template,
    ):
        self._ne = input_dict["ne0"]
        self._nn = input_dict["nn0"]
        self._Te = input_dict["Te0"]
        self._Ti = input_dict["Ti0"]
        self._ne0 = input_dict["ne0"]
        self._nn0 = input_dict["nn0"]
        self._Te0 = input_dict["Te0"]
        self._Ti0 = input_dict["Ti0"]
        self._Bz0 = input_dict["Bz0"]
        self._L_machine = input_dict["Lm"]
        self._R_machrad = input_dict["Rm"]
        self._L_plasma = input_dict["Lp"]
        self._R_plasma = input_dict["Rp"]
        self._L_heatflux = input_dict["Lhf"]
        self._L_partflux = input_dict["Lpf"]
        self._R_heatflux = input_dict["Rhf"]
        self._V_discharge = input_dict["Vd"]
        self._I_discharge = input_dict["Id"]
        self._tau_I_on = input_dict["tau_I_on"]
        self._P_discharge = self._V_discharge * self._I_discharge
        self._S_gp = input_dict["S_gp"]
        self._S_pump = input_dict["S_pump"]
        self._mit_el_temp = (
            input_dict["mit_el_temp"] if "mit_el_temp" in input_dict else 1
        )
        self._Tn = input_dict["Tn"] if "Tn" in input_dict else 0.1
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
        self._chamber_vol = np.pi * self._R_machrad**2 * self._L_machine
        self._Q_B = (
            self._b_source * self._P_discharge / self._plasma_vol / qe_SI
        )  # in eV/cm^3/s
        self._flags = input_flags
        self._v_beam = np.sqrt(2 * self._V_discharge * 1.60218e-12 / 6.6464731e-24)
        eps = self._V_discharge / I_ion
        self._beam_cross = float(He_EII_cross(eps, a_11s))
        self._n_beam = 0
        self._I_beam = 0
        self._J_beam = 0
        self._anode_trans = (
            input_dict["anode_transparency"]
            if "anode_transparency" in input_dict
            else 1.0
        )

    def set_machine_params(
        self, Lp=None, Lm=None, Lhf=None, Lpf=None, Rm=None, Rp=None
    ):
        if Lp is not None:
            self._L_plasma = Lp
        if Lm is not None:
            self._L_machine = Lm
        if Lhf is not None:
            self._L_heatflux = Lhf
        if Lpf is not None:
            self._L_partflux = Lpf
        if Rm is not None:
            self._R_machrad = Rm
        if Rp is not None:
            self._R_plasma = Rp
        self._plasma_vol = np.pi * self._R_plasma**2 * self._L_plasma
        self._chamber_vol = np.pi * self._R_machrad**2 * self._L_machine

    def set_discharge_params(self, Vd=None, Id=None):
        if Vd is not None:
            self._V_discharge = Vd
        if Id is not None:
            self._I_discharge = Id
        self._P_discharge = self._V_discharge * self._I_discharge

    def set_initial_conditions(self, ne0=None, nn0=None, Te0=None, Ti0=None, Bz0=None):
        if ne0 is not None:
            self._ne = ne0
            self._ne0 = ne0
        if nn0 is not None:
            self._nn = nn0
            self._nn0 = nn0
        if Te0 is not None:
            self._Te = Te0
            self._Te0 = Te0
        if Ti0 is not None:
            self._Ti = Ti0
            self._Ti0 = Ti0
        if Bz0 is not None:
            self._Bz0 = Bz0

    def set_time_steps(
        self, t_discharge_off, dt_discharge, t_end=None, dt_afterglow=None
    ):
        if t_discharge_off == 0:
            self._i_d_off = 0
            self._afterlglow = True
            self._time = np.arange(0, t_end, dt_afterglow)
            self._dt_afterglow = dt_afterglow
        else:
            self._dt_discharge = dt_discharge
            self._dt_afterglow = dt_afterglow
            time = np.arange(0, t_discharge_off, dt_discharge)
            self._i_d_off = time.shape[0]
            if t_end is not None and dt_afterglow is not None:
                time = np.append(time, np.arange(t_discharge_off, t_end, dt_afterglow))
                self._afterlglow = True
                self._dt_afterglow = dt_afterglow
            else:
                self._afterlglow = False
        self._time = time
        self._time_flag = True

    def initialize_results(self):
        self._Qie = 0  # Energy exchange between electrons and ions
        self._Qei = 0  # Electron cooling due to inelastic collisions with ions
        self._Qen = 0  # Electron cooling due to inelastic collisions with neutrals
        self._Qcx = 0  # Ion cooling due to charge exchange with neutrals
        self._Qeb = 0  # Electron heating due to discharge
        self._Qbeam = 0  # Ion heating due to beam ions
        self._e_par_hl = 0  # Electron parallel heat loss
        self._i_par_hl = 0  # Ion parallel heat loss
        self._e_perp_hl = 0  # Electron perpendicular heat loss
        self._i_perp_hl = 0  # Ion perpendicular heat loss
        self._end_loss = 0  # Particle end loss
        self._S_ion_bulk = 0  # Bulk ionization source
        self._S_rec_rad = 0  # Radiative recombination sink
        self._S_rec_3b = 0  # Three-body recombination sink
        self._S_ion_beam = 0  # Beam ionization source
        self._densities = np.empty((self._time.shape[0], 3))
        self._densities[0] = np.array([self._ne0, self._nn0, self._n_beam])
        self._temperatures = np.empty((self._time.shape[0], 2))
        self._temperatures[0] = np.array([self._Te0, self._Ti0])
        self._density_terms = np.empty((self._time.shape[0], 5))
        self._density_terms[0] = np.array(
            [
                self._end_loss,
                self._S_ion_bulk,
                self._S_rec_rad,
                self._S_rec_3b,
                self._S_ion_beam,
            ]
        )
        self._heat_terms = np.empty((self._time.shape[0], 10))
        self._heat_terms[0] = np.array(
            [
                self._e_par_hl,
                self._e_perp_hl,
                self._Qie,
                self._Qei,
                self._Qen,
                self._i_par_hl,
                self._i_perp_hl,
                self._Qcx,
                self._Qeb,
                self._Qbeam,
            ]
        )
        self._synthetic = np.empty(
            (self._time.shape[0], 3)
        )  # Placeholder for synthetic diagnostics
        isat = self._ne * np.sqrt(self._Te)
        self._synthetic[0] = np.array([isat, 0, 0])  # Initialize with zeros

    def update_results(self, i):
        self._densities[i] = np.array([self._ne, self._nn, self._n_beam])
        self._temperatures[i] = np.array([self._Te, self._Ti])
        self._density_terms[i] = np.array(
            [
                self._end_loss,
                self._S_ion_bulk,
                self._S_rec_rad,
                self._S_rec_3b,
                self._S_ion_beam,
            ]
        )
        self._heat_terms[i] = np.array(
            [
                self._e_par_hl,
                self._e_perp_hl,
                self._Qie,
                self._Qei,
                self._Qen,
                self._i_par_hl,
                self._i_perp_hl,
                self._Qcx,
                self._Qeb,
                self._Qbeam,
            ]
        )
        isat = self._ne * np.sqrt(self._Te)
        self._synthetic[i] = np.array(
            [isat, 0, 0]
        )  # Placeholder for synthetic diagnostics

    def get_results(self):
        return {
            "time": self._time * 1e3,
            "ne": self._densities[:, 0],
            "nn": self._densities[:, 1],
            "n_beam": self._densities[:, 2],
            "Te": self._temperatures[:, 0],
            "Ti": self._temperatures[:, 1],
            "end_loss": self._density_terms[:, 0],
            "S_ion_bulk": self._density_terms[:, 1],
            "S_rec_rad": self._density_terms[:, 2],
            "S_rec_3b": self._density_terms[:, 3],
            "S_ion_beam": self._density_terms[:, 4],
            "e_par_hl": self._heat_terms[:, 0],
            "e_perp_hl": self._heat_terms[:, 1],
            "Qie": self._heat_terms[:, 2],
            "Qei": self._heat_terms[:, 3],
            "Qen": self._heat_terms[:, 4],
            "i_par_hl": self._heat_terms[:, 5],
            "i_perp_hl": self._heat_terms[:, 6],
            "Qcx": self._heat_terms[:, 7],
            "Qeb": self._heat_terms[:, 8],
            "Qbeam": self._heat_terms[:, 9],
            "isat": self._synthetic[:, 0],
        }

    def calc_density_terms(self, ne, nn, Te):
        if Te < 0.01:
            Te = 0.01
        if ne < 1e8:
            ne = 1e8
        if nn < 1e8:
            nn = 1e8
        if Te > self._mit_el_temp and self._flags["mit_el"]:
            end_loss = 2 * ne * v_ion_speed(self._mit_el_temp, mu=4) / self._L_partflux
        else:
            end_loss = 2 * ne * v_ion_speed(Te, mu=4) / self._L_partflux
        self._p_beam = (
            self._beam_cross
            * nn
            / (self._beam_cross * nn + (1 / (self._v_beam * time_elec_coll(Te, ne))))
        )
        S_ion_bulk = self._b_ioniz * ne * nn * rate_coeff(Te, I_ion, *fit_coeff)
        S_ion_beam = self._A_ion_beam * self._p_beam
        S_rec_rad = self._b_rec_rad * ne * ne * alpha_r(Te)
        S_rec_3b = self._b_rec_3b * ne * ne * ne * alpha_3(Te)
        blah = np.array([end_loss, S_ion_bulk, S_rec_rad, S_rec_3b, S_ion_beam])
        # print("Density terms: ", blah)
        return blah

    def calc_heat_terms(self, ne, nn, Te, Ti):
        if Ti < 0.01:
            Ti = 0.01
        if Te < 0.01:
            Te = 0.01
        if ne < 1e8:
            ne = 1e8
        if nn < 1e8:
            nn = 1e8
        e_par_hl = (
            self._b_epara
            * en_factor
            * 2
            * elec_par_heat_loss(Te, ne, self._L_plasma, self._L_heatflux)
        )
        i_par_hl = (
            self._b_ipara
            * en_factor
            * 2
            * ion_par_heat_loss(Ti, ne, self._L_plasma, self._L_heatflux)
        )
        Qei = 0
        Qen = 0
        Qcx = 0
        Qeb = 0
        Qbeam = 0
        e_perp_hl = 0
        i_perp_hl = 0
        if self._flags["eperp"]:
            e_perp_hl = (
                self._b_eperp
                * en_factor
                * 2
                * np.pi
                * elec_perp_heat_loss(
                    Te, ne, self._Bz0, self._L_plasma, self._L_heatflux
                )
            )
        if self._flags["iperp"]:
            i_perp_hl = (
                self._b_iperp
                * en_factor
                * 2
                * np.pi
                * ion_perp_heat_loss(
                    Ti,
                    ne,
                    self._Bz0,
                    self._L_plasma,
                    self._L_heatflux,
                    mu=4,
                )
            )
        Qie = (
            self._b_Qie * en_factor * Q_ie(Te, Ti, ne)
        )  # Qie = self._b_Qie * en_factor * Q_ie(Te, Ti, ne)
        if self._flags["icool"]:
            Qei = (
                self._b_Qei
                * en_factor
                * IAEA_exp4(Te, aHeII, recomb=self._flags["icool_recomb"])
                * ne
            )
        if self._flags["ncool"]:
            Qen = self._b_Qen * en_factor * IAEA_exp1(Te, aHeI) * nn
        if self._flags["cx"]:
            Qcx = self._b_Qcx * en_factor * Q_cx_He(ne, nn, Ti, self._Tn)
        if self._discharge_on:
            Qeb = (1 - self._p_beam) * (
                en_factor
                * self._I_beam
                * self._anode_trans
                * self._V_discharge
                / ne
                / self._plasma_vol
                / qe_SI
            )
            Qbeam = (
                en_factor
                * self._p_beam
                * self._A_ion_beam
                * (self._V_discharge - I_ion)
                / ne
            )
            # print(Qeb, Qbeam)
        if self._flags["C_imp"]:
            pass  # Placeholder for carbon impurity cooling
        if self._flags["O_imp"]:
            pass  # Placeholder for oxygen impurity cooling
        blah = np.array(
            [e_par_hl, e_perp_hl, Qie, Qei, Qen, Qeb, i_par_hl, i_perp_hl, Qcx, Qbeam]
        )
        # print("Heat terms: ", blah)
        return blah

    def _dstep(self, a):
        if self._Ti < 0.01:
            self._Ti = 0.01
        if self._Te < 0.01:
            self._Te = 0.01
        if self._ne < 1e8:
            self._ne = 1e8
        if self._nn < 1e8:
            self._nn = 1e8
        ne, nn, Te, Ti = a
        end_loss, S_ion_bulk, S_rec_rad, S_rec_3b, S_ion_beam = self.calc_density_terms(
            ne, nn, Te
        ).copy()
        d_ne = S_ion_bulk + S_ion_beam - S_rec_rad - S_rec_3b - end_loss
        d_nn = self._S_gp - (self._S_pump * nn) - d_ne
        e_par_hl, e_perp_hl, Qie, Qei, Qen, Qeb, i_par_hl, i_perp_hl, Qcx, Qbeam = (
            self.calc_heat_terms(ne, nn, Te, Ti).copy()
        )
        d_Te = Qeb + Qbeam - Qie - Qei - Qen - e_par_hl - e_perp_hl
        d_Ti = Qie - i_par_hl - i_perp_hl - Qcx
        return np.array([d_ne, d_nn, d_Te, d_Ti])

    def _rk4_step(self, a):
        # print("RK4 step with h =", self._h)
        k1 = self._dstep(a).copy()
        k1_a = a + 0.5 * self._h * k1
        k2 = self._dstep(k1_a).copy()
        k2_a = a + 0.5 * self._h * k2
        k3 = self._dstep(k2_a).copy()
        k3_a = a + self._h * k3
        k4 = self._dstep(k3_a).copy()
        # print("k: ", k1, k2, k3, k4)
        b = a + (self._h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        # print("b: ", b)
        self._ne, self._nn, self._Te, self._Ti = b
        if self._Ti < 0.01:
            self._Ti = 0.01
        if self._Te < 0.01:
            self._Te = 0.01
        if self._ne < 1e8:
            self._ne = 1e8
        if self._nn < 1e8:
            self._nn = 1e8

    def start_simulation(self):
        if not self._time_flag:
            raise ValueError("Time steps not set. Use set_time_steps() method.")
        self.initialize_results()
        print("Starting simulation...")
        t = self._time.shape[0]
        for i, time in enumerate(self._time):
            if i == 0:
                continue
            elif i < self._i_d_off:
                self._discharge_on = True
                self._h = self._dt_discharge
                if self._I_beam < self._I_discharge:
                    self._I_beam = self._I_discharge * (
                        1 - np.exp(-time / self._tau_I_on)
                    )
                else:
                    self._I_beam = self._I_discharge
                self._J_beam = self._I_beam / self._plasma_cross / qe_SI
                self._n_beam = self._J_beam / self._v_beam
                self._n_beam_ion = self._n_beam * self._beam_cross * self._v_beam
            elif self._afterlglow:
                self._discharge_on = False
                self._h = self._dt_afterglow
                self._I_beam = 0
                self._J_beam = 0
                self._n_beam = 0
                self._n_beam_ion = 0
            else:
                break
            if self._Ti < 0.01:
                self._Ti = 0.01
            if self._Te < 0.01:
                self._Te = 0.01
            if self._ne < 1e8:
                self._ne = 1e8
            if self._nn < 1e8:
                self._nn = 1e8
            ne = self._ne
            nn = self._nn
            Te = self._Te
            Ti = self._Ti
            self._A_ion_beam = self._n_beam_ion * nn
            a = np.array([ne, nn, Te, Ti])
            self._rk4_step(a)
            ne = self._ne
            nn = self._nn
            Te = self._Te
            Ti = self._Ti
            (
                self._end_loss,
                self._S_ion_bulk,
                self._S_rec_rad,
                self._S_rec_3b,
                self._S_ion_beam,
            ) = self.calc_density_terms(ne, nn, Te)
            (
                self._e_par_hl,
                self._e_perp_hl,
                self._Qie,
                self._Qei,
                self._Qen,
                self._Qeb,
                self._i_par_hl,
                self._i_perp_hl,
                self._Qcx,
                self._Qbeam,
            ) = self.calc_heat_terms(ne, nn, Te, Ti)
            self.update_results(i)
            print(
                f"Step {i+1}/{t}: ne={self._ne:.2e}, nn={self._nn:.2e}, Te={self._Te:.2f} eV, Ti={self._Ti:.2f} eV"
            )
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

    def operating_point_power(
        self,
        ylim=(-1e6, 1e6),
        eltemp=1,
        pbeam=1,
        g=1,
        ncool=False,
        icool=False,
        iheat=False,
    ):
        self._discharge_on = True
        Temps = np.arange(0.1, 100, 0.01)
        nns = np.logspace(12, 14, num=21)
        nes = nns
        self._mit_el_temp = eltemp
        self._I_beam = self._I_discharge
        self._J_beam = self._I_beam / self._plasma_cross / qe_SI
        self._n_beam = self._J_beam / self._v_beam
        self._n_beam_ion = self._n_beam * self._beam_cross * self._v_beam
        self._p_beam = pbeam
        # self._p_beam = (
        #     self._beam_cross
        #     * nn
        #     / (self._beam_cross * nn + (1 / (self._v_beam * time_elec_coll(Te, ne))))
        # )
        curves = np.empty((nes.shape[0], Temps.shape[0]))
        te_op_pts = np.empty(nes.shape[0])
        Ti = 1
        for i, ne in enumerate(nes):
            nn = nns[i]
            self._A_ion_beam = self._n_beam_ion * nn
            for j, Te in enumerate(Temps):
                end_loss, S_ion_bulk, S_rec_rad, S_rec_3b, S_ion_beam = (
                    self.calc_density_terms(ne, nn, Te)
                )
                total = S_ion_bulk + S_ion_beam - g * end_loss
                curves[i, j] = total
        for i, nn in enumerate(nns):
            te_op_pts[i] = Temps[np.argmin(np.abs(curves[i]))]
        fig, ax = plt.subplots()
        ax.semilogx(nns, te_op_pts)
        ax.set_ylabel("Electron Temperature (eV)")
        ax.set_xlabel("Neutral Density (cm^-3)")
        ax.set_title("Operating Point Electron Temperature vs Neutral Density")
        fig.show()
        Ratios = np.logspace(-1, 1, num=21)
        Powers = np.arange(50000, 2000000, 2000)
        curves_pwr = np.empty((nns.shape[0], Ratios.shape[0], Powers.shape[0]))
        pwr_op_pts = np.empty((nns.shape[0], Ratios.shape[0]))
        for k, ratio in enumerate(Ratios):
            for i, nn in enumerate(nns):
                ne = ratio * nn
                Te = te_op_pts[i]
                for j, P in enumerate(Powers):
                    self._P_discharge = P
                    self._Q_B = (
                        self._b_source
                        * self._P_discharge
                        / self._plasma_vol
                        / qe_SI
                        / ne
                    )
                    (
                        e_par_hl,
                        e_perp_hl,
                        Qie,
                        Qei,
                        Qen,
                        Qeb,
                        i_par_hl,
                        i_perp_hl,
                        Qcx,
                        Qbeam,
                    ) = self.calc_heat_terms(ne, nn, Te, Ti=1)
                    curves_pwr[i, k, j] = (2 / 3) * self._Q_B - e_par_hl
                    if ncool:
                        curves_pwr[i, k, j] -= Qen
                    if icool:
                        curves_pwr[i, k, j] -= Qei
                    if iheat:
                        curves_pwr[i, k, j] -= Qie
        for i, nn in enumerate(nns):
            for k, ratio in enumerate(Ratios):
                pwr_op_pts[i, k] = Powers[np.argmin(np.abs(curves_pwr[i, k]))]
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, layout="tight")
        Y = np.log10(nns)
        X = np.log10(Ratios)
        X, Y = np.meshgrid(X, Y)
        surf = ax.plot_surface(X, Y, np.log10(pwr_op_pts).T, cmap="plasma")
        ax.set_xlabel("log10(ne/nn)")
        ax.set_ylabel("log10(nn) (cm^-3)")
        ax.set_zlabel("log10(Power) (W)")
        fig.colorbar(surf, shrink=0.3, aspect=10)
        fig.savefig("opcon_power.svg")
        plt.show()
