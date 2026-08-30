"""How many accepted steps of the g1atrim route reach the first tail walk?

The A2a bit-inertness A/B is only evidence about the tail-walk branch if its
window CONTAINS a tail walk. The golden-at-stance config starts pre-breakdown
with the beam off, so the walk engages some way in; this measures where, so the
A/B's step budget is chosen from the run rather than guessed.

Stops at the first entry into the ionizing tail walk and prints the step count,
the physical time and the phase. Run from the worktree root with
PYTHONPATH=<worktree>.
"""

import sys
from pathlib import Path

import numpy as np

import cablp
from cablp.solvers._sim1d import LAPDSim1D
from cablp.cathode import beam_deposition as B

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from baseline_sim1d import build_baseline_config  # noqa: E402

print(f"cablp.__file__ = {cablp.__file__}")

HITS = {"chains": 0, "plans": 0}
_real = B._tail_lane_chains


def _counted(plans, *a, **kw):
    HITS["chains"] += 1
    for _E, hf, _ion in plans:
        if np.any(np.asarray(hf) > 0.0):
            HITS["plans"] += 1
    return _real(plans, *a, **kw)


B._tail_lane_chains = _counted


def step_once(sim):
    split = sim._flags.get("implicit_heat_conduction", False)
    diag = sim.suggest_timestep(include_heat_conduction=not split)

    def generate():
        attempt, retries, reason, events = sim._attempt_step_with_retries(
            dt=diag.dt, operator_split=None, diag=diag,
        )
        return attempt, (retries, reason, events)

    return sim._accept_step_with_picard(generate)


CAP = int(sys.argv[1]) if len(sys.argv) > 1 else 60000
params, flags = build_baseline_config()
params = dict(params)
params["max_steps_action"] = "stop"
sim = LAPDSim1D(input_dict=params, input_flags=dict(flags))
print(f"cells = {sim.geometry.cells}")
for k in range(1, CAP + 1):
    step_once(sim)
    if HITS["plans"]:
        print(f"FIRST TAIL WALK at accepted step {k}")
        print(f"  time_s     = {sim.time!r}")
        print(f"  chain calls= {HITS['chains']}")
        print(f"  populations= {HITS['plans']}")
        break
    if k % 2000 == 0:
        print(f"  step {k}: t = {sim.time:.6e} s, no tail walk yet",
              flush=True)
else:
    print(f"NO TAIL WALK within {CAP} accepted steps (t = {sim.time:.6e} s)")
    sys.exit(2)
