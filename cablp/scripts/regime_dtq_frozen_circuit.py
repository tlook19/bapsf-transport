"""regime_dtq probe 2: the circuit ODE alone, at the FROZEN t=0 plasma state.

(i)  Maps V_dis(I) and f(I) = (V_src - I*R_comp - V_dis(I))/L over I to show
     the wall structure and the sign of f on each side of it.
(ii) Integrates advance_circuit_current_driven repeatedly at several dt with
     the plasma state, T_s, phi_wf frozen at t=0 -- if the dt divergence
     reproduces here, the mechanism is circuit-side only (no co-evolution).
"""
import math
import os
import sys
import warnings

warnings.simplefilter("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from regime_r2_overlap_gate import build_config  # noqa: E402
from cablp.solvers._sim1d import LAPDSim1D  # noqa: E402
from cablp.solvers._sim1d.physics.cathode import (  # noqa: E402
    advance_circuit_current_driven,
    idriven_result_evaluator,
    idriven_vdis_evaluator,
)

params, flags = build_config(20, True)
sim = LAPDSim1D(params, flags)

step_phase = sim._cathode_phase_options(time=0.0)
print("phase options at t=0:", step_phase)
# mimic the lazy bank charge the accept path performs
C_bank = params.get("C_bank_F")
V_cap = float(params.get("V_bank", 0.0))
V_src = V_cap
R_comp = float(params.get("R_comp", 0.0)) * float(params.get("R_comp_partition", 1.0))
L = float(params.get("L_parasitic_H", 0.0))
print(f"V_src={V_src:.4g} V  R_comp={R_comp:.4g} Ohm  L={L:.4g} H  "
      f"C_bank={C_bank}  R_mesh={params.get('R_mesh_ohm', 0.0)}")
print(f"L/R_comp = {L/R_comp:.4g} s ; R_comp*C = {R_comp*float(C_bank):.4g} s")

common = dict(
    state=sim.state,
    floors=sim._floors,
    ion_mass_g=sim._ion_mass_g,
    mu=sim._mu,
    geometry=sim._geometry,
    input_dict=sim._input_dict,
    input_flags=sim._effective_cathode_flags(active_only=False, floating=False),
    beam_cross_prev=sim._cathode_beam_cross,
    T_s_override_K=sim._cathode_Ts_K,
    phi_wf_override_eV=sim._cathode_phi_wf_eff(),
    circuit_V_src_V=V_src,
)
vdis = idriven_vdis_evaluator(**common)
solve_at = idriven_result_evaluator(**common)

print("\n-- V_dis(I) map at the frozen t=0 state --")
print("   I [A]        V_dis [V]   V_avail [V]  f(I) [A/s]    regime")
Is = [0.0, 0.5, 0.9, 0.92, 0.9219, 0.922, 0.925, 0.93, 1.0, 1.07, 1.5, 2.0,
      5.0, 10.0, 17.52, 50.0, 156.7, 500.0, 2000.0]
for I in Is:
    V = vdis(I)
    res = solve_at(I)
    V_avail = V_src - I * (float(params.get("R_comp", 0.0))
                           + float(params.get("R_mesh_ohm", 0.0)))
    f = (V_src - I * R_comp - V) / L
    print(f"   {I:10.6g}  {V:10.6g}  {V_avail:10.6g}  {f:12.6g}   "
          f"{getattr(res, 'regime', '?')}")

print("\n-- frozen-state circuit integration, I(0)=0, to t=2e-5 --")
for dt in (2.0e-5, 3.0e-6, 1.0e-6, 3.0e-7, 1.0e-7, 1.0e-8):
    n_steps = max(1, int(round(2.0e-5 / dt)))
    I = 0.0
    V_cap_s = V_cap
    for _ in range(n_steps):
        I, V_cap_new, V_dis_step = advance_circuit_current_driven(
            I_prev_A=I,
            dt_s=dt,
            V_src_V=V_cap_s,
            R_comp_ohm=R_comp,
            L_H=L,
            vdis_of_I=vdis,
            C_bank_F=C_bank,
            V_cap_prev_V=V_cap_s,
        )
        if V_cap_new is not None:
            V_cap_s = V_cap_new
    print(f"   dt={dt:8.1e}  steps={n_steps:5d}  I(2e-5)={I:10.5g} A  "
          f"V_dis_last={V_dis_step:8.5g} V")
