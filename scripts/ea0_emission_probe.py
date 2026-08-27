"""ea0 design read: cheap emission-path arithmetic (no time integration).

Bounded probes only:
  (1) Richardson full-area capability I_eth at the stance (uniform + gaussian
      annular, dirty/clean phi_wf, base/static T_s) -- the f_em normalizer
      candidates.
  (2) ads_des dynamic range exp(dphi_wf / kT_s) -- can cleanliness BE the
      emitting-area throttle at the shipped constants?
  (3) Capability-wall solves: solve_idriven at an imposed current far above
      capability, at foot-bracket plasma states -> the carried current at the
      ceiling (the wall the model rides).
  (4) f_em0 = I_machine_foot / normalizer candidates + subcriticality and
      foot-duration arithmetic against the pd1owner gain numbers.
  (5) power_balance thermal time constant (thermal-spreading keying option).
"""
import math
import numpy as np

from cablp.solvers._sim1d import default_config
from cablp.solvers._sim1d.physics.cathode import (
    cathode_device_config,
    cathode_emission_annuli,
)
from cablp.funcs._cathode_solver_idriven import solve_idriven
from cablp.funcs._cathode_solver import PlasmaState

cfg_d, cfg_f = default_config()
d = dict(cfg_d)
f = dict(cfg_f)
mu = 4.002602  # He
kB_over_e = 8.617333262e-5

print("=== stance keys (config defaults; mirrored by compare_sim1d_es1.PARAM_OVERRIDES) ===")
for k in ("T_s", "cathode_Ts_base_K", "phi_wf", "cathode_phiwf_clean_eV",
          "C_R", "R_cath", "eta", "cathode_emission_profile",
          "cathode_Ts_fwhm_cm", "cathode_emission_annuli",
          "cathode_warming_model", "cathode_surface_model",
          "cathode_phi_c_cap_V", "cathode_heat_capacity_J_per_K",
          "cathode_conduction_W_per_K", "cathode_emissivity",
          "cathode_cleaning_sigma_cm2", "cathode_cleaning_E_th_eV"):
    print(f"  {k} = {d.get(k)!r}")

def richardson_uniform(Ts, phiwf, R_cath, C_R):
    A = math.pi * R_cath**2
    return A * C_R * Ts**2 * math.exp(-phiwf / (kB_over_e * Ts))

def richardson_annular(dd):
    Ts_k, area_k, frac_k = cathode_emission_annuli(dd, n_annuli=int(dd.get("cathode_emission_annuli", 10)))
    tot = sum(a * dd["C_R"] * T**2 * math.exp(-dd["phi_wf"] / (kB_over_e * T))
              for T, a in zip(Ts_k, area_k))
    wet = sum(a * dd["C_R"] * T**2 * math.exp(-dd["phi_wf"] / (kB_over_e * T))
              for T, a, fr in zip(Ts_k, area_k, frac_k) if fr > 0)
    return tot, wet, Ts_k[0], Ts_k[-1]

print("\n=== (1) Richardson full-area capability I_eth [A] ===")
for Ts in (float(d["cathode_Ts_base_K"]), float(d["T_s"])):
    for phiwf, tag in ((float(d["phi_wf"]), "dirty 2.869"), (float(d["cathode_phiwf_clean_eV"]), "clean 2.809")):
        dd = {**d, "T_s": Ts, "phi_wf": phiwf}
        uni = richardson_uniform(Ts, phiwf, float(d["R_cath"]), float(d["C_R"]))
        tot, wet, T0, T9 = richardson_annular(dd)
        print(f"  T_s={Ts:8.2f} K  phi_wf={tag}:  uniform disc {uni:9.2f}  gaussian-annular total {tot:9.2f} (annuli T {T0:.1f}..{T9:.1f} K)")

print("\n=== (2) ads_des dynamic range at shipped constants ===")
for Ts in (float(d["cathode_Ts_base_K"]), float(d["T_s"])):
    dphi = float(d["phi_wf"]) - float(d["cathode_phiwf_clean_eV"])
    ratio = math.exp(dphi / (kB_over_e * Ts))
    print(f"  T_s={Ts:.2f} K: dphi_wf={dphi:.3f} eV -> emission ratio clean/dirty = {ratio:.3f}")

print("\n=== (3) capability-wall solves (imposed I >> capability; carried I_tot at ceiling) ===")
dev = cathode_device_config(d, f, mu)
print(f"  DeviceConfig: A_c={dev.A_c:.1f} cm^2, T_s={dev.T_s}, phi_wf={dev.phi_wf}, eta={dev.eta}, "
      f"annuli={len(dev.emission_Ts_K)}, config I_eth(static)={dev.I_eth:.2f} A")
schottky = bool(f.get("cathode_schottky", False))
print(f"  cathode_schottky flag = {schottky}")
for (Te, ne, nn, tag) in (
    (0.5, 1.0e9,  2.0e13, "seed / t=0-class"),
    (2.0, 1.0e10, 2.0e13, "early foot"),
    (12.0, 1.0e11, 2.0e13, "mid build (pd1owner window)"),
    (13.0, 1.55e11, 2.0e13, "build end (pd1owner t=1e-4)"),
):
    res = solve_idriven(dev, PlasmaState(T_e=Te, n_e=ne, n_n=nn, sigma_b=0.0),
                        I_tot_A=1.0e4, schottky=schottky,
                        phi_c_cap_V=float(d["cathode_phi_c_cap_V"]))
    print(f"  Te={Te:5.1f} eV ne={ne:8.2e}: regime={res.regime:19s} carried I_tot={res.I_tot:9.3f} A "
          f"I_eth_star={res.I_eth_star:9.3f} A phi_c={res.phi_c:8.2f} V")

print("\n=== (4) f_em0 candidates and subcriticality arithmetic ===")
I_foot = (0.34, 0.47)      # machine back-extrapolated window-start current [A]
gamma_primary_full = 1.8685e4   # pd1owner_reads.txt sec (3), f100 leg mean primary gain [1/s]
loss_all_surface   = 4.4532e3   # same leg all-surface loss [1/s]
tau_F2 = 719.0e-6               # F2-calibrated e-fold (coverage_growth_rate 1390/s)
print(f"  pd1owner: primary-alone gain {gamma_primary_full:.4g} /s, all-surface loss {loss_all_surface:.4g} /s, "
      f"ratio {gamma_primary_full/loss_all_surface:.2f}")
for norm, tag in ((54.0, "model capability wall, build end (pd1owner: 1.5->54 A by 1e-5 s)"),
                  (1.5, "model wall-riding current at t=0 (seed-transient)"),
                  (dev.I_eth, f"static Richardson ceiling {dev.I_eth:.1f} A")):
    for I0 in I_foot:
        fe0 = I0 / norm
        g0 = fe0 * gamma_primary_full
        f_crit = loss_all_surface / gamma_primary_full
        t_cross = tau_F2 * math.log(f_crit / fe0) if fe0 < f_crit else 0.0
        print(f"  norm {norm:8.2f} A ({tag[:48]:48s}) I0={I0:.2f}: f_em0={fe0:.4f} "
          f"-> primary gain at f_em0 {g0:8.1f} /s (sub/supercritical x{g0/loss_all_surface:.3f}); "
          f"f_crit={f_crit:.3f}, t_cross={t_cross*1e3:.2f} ms")

print("\n=== (5) power_balance thermal time constant (thermal-spreading keying) ===")
C_th = float(d["cathode_heat_capacity_J_per_K"])
eps = float(d["cathode_emissivity"])
area = math.pi * float(d["R_cath"])**2
Ts = float(d["cathode_Ts_base_K"])
G_rad = 4 * eps * 5.670374419e-12 * area * Ts**3
G_cond = float(d["cathode_conduction_W_per_K"])
print(f"  G_rad(4*eps*sigma*A*T^3)={G_rad:.1f} W/K, G_cond={G_cond:.1f} W/K, C_th={C_th:.1f} J/K")
print(f"  tau_thermal = C_th/(G_rad+G_cond) = {C_th/(G_rad+G_cond)*1e3:.1f} ms  (vs tau_I = 0.719 ms)")
