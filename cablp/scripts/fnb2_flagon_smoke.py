"""Smoke-grade flag-ON short run for the NBL pass-2 two-channel neutral build.

STILL NOT A SCOREABLE PHYSICS ARM -- nothing here is compared against data --
but unlike pass 1 the En budget is now COMPLETE, so the two watch items pass 1
left open are checkable and are checked here:

  W-A  the puff floor debit. Pass 1 added puff particles with no energy of
       their own and the En floor had to invent it, one clip at a time
       (5.48e4 erg over the window). The puff now arrives at the wall energy,
       so the debit must collapse toward zero.
  W-B  the unbounded Tn. Pass 1 ran Tn up to 8.55 eV (99230 K) because the CX
       channel heated the cold gas with energy that in fact leaves it. The CX
       share is now a population swap, so Tn must stay in a physical band.

Also reports the dt census (the 7x dt_neutral_energy headroom watch) and the
hot-channel diagnostics: f_hot should land in the fts-predicted 5-20% class
wherever the plasma is live.

Run with PYTHONPATH=<worktree>/cablp.
"""
import numpy as np

import cablp
from cablp.solvers._sim1d import LAPDSim1D, default_config, summarize_result
from cablp.solvers._sim1d.core.state import (
    NEUTRAL_ENERGY_FLOOR_T_K,
    neutral_energy_floor,
)
from cablp.vars._cons import ev_to_erg, kb_cgs

print("cablp package file:", cablp.__file__)

params, flags = default_config()
params["nx"] = 24
params["S_gp"] = 9010
params["max_steps_action"] = "stop"
flags["neutral_momentum"] = True
flags["neutral_two_zone"] = True
flags["neutral_energy"] = True

MAX_STEPS = 20000
T_END = 3.0e-3

# The pass-1 numbers this run is measured against, on the SAME configuration
# and the same step budget (fnb1_flagon_smoke.txt).
PASS1_PUFF_DEBIT_ERG = 5.482078e04
PASS1_TN_MAX_EV = 8.551018e00

sim = LAPDSim1D(input_dict=params, input_flags=flags)
print("packed rows:", sim._y.size // sim._geometry.cells)
print("alpha_E:", sim._neutral_energy_alpha)
print("knudsen temperature:", sim._neutral_knudsen_temperature)
landing, residence, end_fraction = sim._hot_neutral_kernels
print(f"ballistic kernel: {landing.shape[0]} cells, "
      f"landing rows close to 1 within "
      f"{float(np.max(np.abs(landing.sum(axis=1) - 1.0))):.2e}, "
      f"end-plane fold-back mean {float(np.mean(end_fraction)):.4f}")

sim.start_simulation(t_end=T_END, dt=None, operator_split=None,
                     max_steps=MAX_STEPS)
result = sim.get_results()

health = summarize_result(result)
print(f"\nstatus={result.run_status} steps={result.steps} "
      f"t_end={result.time[-1]:.6e} s")
print("finite (every saved field):", health.finite)
print("per-field finiteness:", health.finite_fields)

wall_eV = NEUTRAL_ENERGY_FLOOR_T_K * kb_cgs / ev_to_erg
margin = result.En - neutral_energy_floor(result.nn)
Tn_max = float(np.max(result.Tn))
print(f"\nTn [eV]: min={float(np.min(result.Tn)):.6e} "
      f"max={Tn_max:.6e}  wall={wall_eV:.6e}")
print(f"Tn [K]:  min={float(np.min(result.Tn)) * ev_to_erg / kb_cgs:.2f} "
      f"max={Tn_max * ev_to_erg / kb_cgs:.2f}")
print("En >= floor at every saved sample:",
      bool(np.all(result.Tn >= wall_eV * (1.0 - 1e-12))))
print(f"worst saved En margin above the floor: {float(np.min(margin)):.6e} "
      "erg/cm^3")

print("\nfloor ledger:")
for name, value in sorted(result.floor_ledger.items()):
    print(f"  {name:28s} {float(value):.6e}")

puff_debit = float(result.floor_ledger["En_energy_added_erg"])
print("\n--- WATCH ITEM W-A: the puff floor debit ---")
print(f"  pass 1: {PASS1_PUFF_DEBIT_ERG:.6e} erg   "
      f"pass 2: {puff_debit:.6e} erg")
if PASS1_PUFF_DEBIT_ERG > 0.0:
    print(f"  ratio to pass 1: {puff_debit / PASS1_PUFF_DEBIT_ERG:.3e}")
w_a = puff_debit <= 1.0e-3 * PASS1_PUFF_DEBIT_ERG
print(f"  RESOLVED (debit collapsed by >= 1000x): {w_a}")

print("\n--- WATCH ITEM W-B: the neutral temperature band ---")
print(f"  pass 1 max Tn: {PASS1_TN_MAX_EV:.6e} eV "
      f"({PASS1_TN_MAX_EV * ev_to_erg / kb_cgs:.0f} K)")
print(f"  pass 2 max Tn: {Tn_max:.6e} eV "
      f"({Tn_max * ev_to_erg / kb_cgs:.0f} K)")
# The physical expectation: the cold channel is heated only by the ELASTIC
# ion-neutral rate against a few-eV ion population, and the wall pulls it back
# at 300 K. A cold bulk that stays inside ~1 eV is the band; the CX-born atoms
# that used to push it to 8.55 eV are now their own channel.
w_b = Tn_max < 1.0
print(f"  RESOLVED (cold bulk stays below 1 eV): {w_b}")

print("\n--- hot channel ---")
live = result.n > 1.0e10
if np.any(live):
    f_hot = result.f_hot[live]
    print(f"  f_hot where n > 1e10: min={float(np.min(f_hot)):.4f} "
          f"median={float(np.median(f_hot)):.4f} "
          f"max={float(np.max(f_hot)):.4f}")
    print(f"  nn_hot max: {float(np.max(result.nn_hot)):.4e} cm^-3")
    tau = result.tau_hot[result.tau_hot > 0.0]
    if tau.size:
        print(f"  tau_hot [us]: min={float(np.min(tau)) * 1e6:.3f} "
              f"median={float(np.median(tau)) * 1e6:.3f} "
              f"max={float(np.max(tau)) * 1e6:.3f}")
    median_f = float(np.median(f_hot))
    hot_ok = bool(np.all(np.isfinite(result.f_hot))) and 0.0 <= median_f <= 0.60
    print(f"  fts-predicted class is 5-20%; median here {median_f:.4f}")
else:
    hot_ok = False
    print("  no live plasma in the window")

dtn = np.array([d.dt_neutral_energy for d in result.diagnostics])
dt = np.array([d.dt for d in result.diagnostics])
binding = sum(1 for d in result.diagnostics
              if d.active_constraint == "neutral_energy")
print(f"\ndt_neutral_energy: min={float(np.min(dtn)):.6e} "
      f"max={float(np.max(dtn)):.6e} s  (present on all "
      f"{int(np.sum(np.isfinite(dtn)))}/{dtn.size} steps)")
print(f"accepted dt:       min={float(np.min(dt)):.6e} "
      f"max={float(np.max(dt)):.6e} s")
print(f"worst dt_neutral_energy / accepted dt ratio: "
      f"{float(np.min(dtn / dt)):.1f}x headroom")
print(f"steps where neutral_energy was the ACTIVE constraint: {binding}")
print("active constraints seen:",
      sorted({d.active_constraint for d in result.diagnostics}))

ok = (
    health.finite
    and bool(np.all(result.Tn >= wall_eV * (1.0 - 1e-12)))
    and bool(np.all(np.isfinite(dtn)))
    and w_a
    and w_b
    and hot_ok
)
print("\nFLAG-ON SMOKE:", "PASS (watch items resolved; not a scored arm)"
      if ok else "FAIL")
