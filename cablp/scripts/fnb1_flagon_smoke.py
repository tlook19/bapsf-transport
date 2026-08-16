"""Smoke-grade flag-ON short run for the NBL pass-1 En field.

NOT A PHYSICS ARM. The En budget is deliberately incomplete in pass 1 -- there
is no pressure force, no En advection, no Knudsen enthalpy carriage, and no
jet/puff/pump/reaction En bookkeeping -- so the numbers below say only that the
field integrates, stays above its floor, and rides the timestep census without
binding. Nothing here is scoreable.

Reports: finiteness of every saved field, the En floor margin, the floor
ledger's En line, and the dt_neutral_energy census against the accepted dt.

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

sim = LAPDSim1D(input_dict=params, input_flags=flags)
print("packed rows:", sim._y.size // sim._geometry.cells)
print("alpha_E:", sim._neutral_energy_alpha)
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
print(f"\nTn [eV]: min={float(np.min(result.Tn)):.6e} "
      f"max={float(np.max(result.Tn)):.6e}  wall={wall_eV:.6e}")
print(f"Tn [K]:  min={float(np.min(result.Tn)) * ev_to_erg / kb_cgs:.2f} "
      f"max={float(np.max(result.Tn)) * ev_to_erg / kb_cgs:.2f}")
print("En >= floor at every saved sample:",
      bool(np.all(result.Tn >= wall_eV * (1.0 - 1e-12))))
print(f"worst saved En margin above the floor: {float(np.min(margin)):.6e} "
      "erg/cm^3")

print("\nfloor ledger:")
for name, value in sorted(result.floor_ledger.items()):
    print(f"  {name:28s} {float(value):.6e}")

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
    and binding == 0
    and bool(np.all(np.isfinite(dtn)))
)
print("\nFLAG-ON SMOKE:", "PASS (plumbing only, not a physics claim)"
      if ok else "FAIL")
