"""Bit-exact A/B: neutral-energy keys ABSENT vs EXPLICITLY at their defaults.

Raw uint64 comparison of every saved float field over a short real run on three
packed layouts -- not a tolerance. Arm A omits every key the NBL passes added;
arm B states them all explicitly at their registered defaults
(neutral_energy=False, neutral_energy_wall_accommodation=0.40,
neutral_knudsen_temperature="frozen").

The hot-channel DIAGNOSTICS pass registers NO new configuration key: the
diagnostics it saves are readings of a term that only exists under
``neutral_energy``, so with the flag off there is nothing new to switch on and
nothing new to write. The key set under test is therefore unchanged from
fnb2_bitexact_ab.py, deliberately -- this run re-establishes the same claim on
the changed build rather than testing a new knob. If the two arms agree at the
bit, the diagnostics pass is inert with the flag off, which the golden then
confirms on the production configuration.

Run with PYTHONPATH=<worktree>/cablp.
"""
import time as _time

import numpy as np

import cablp
from cablp.solvers._sim1d import LAPDSim1D, default_config

print("cablp package file:", cablp.__file__)

# Step-BOUNDED so the arm has a known cost. The budget carries the run past the
# square puff's opening edge so the neutral terms the En field would couple to
# are genuinely live; the reached time and step count are reported per arm so
# the window is auditable rather than asserted.
MAX_STEPS = 20000
T_END = 3.0e-3

NEW_PARAMS = {
    "neutral_energy_wall_accommodation": 0.40,   # pass 1
    "neutral_knudsen_temperature": "frozen",     # pass 2
}
NEW_FLAG = "neutral_energy"


def run(explicit, two_zone=True, momentum=True):
    params, flags = default_config()
    params["nx"] = 24
    params["S_gp"] = 9010
    params["max_steps_action"] = "stop"
    flags["neutral_momentum"] = bool(momentum)
    flags["neutral_two_zone"] = bool(two_zone)
    if explicit:
        params.update(NEW_PARAMS)
        flags[NEW_FLAG] = False
    else:
        for name in NEW_PARAMS:
            params.pop(name, None)
        flags.pop(NEW_FLAG, None)
    for name in NEW_PARAMS:
        assert (name in params) is explicit
    assert (NEW_FLAG in flags) is explicit
    sim = LAPDSim1D(input_dict=params, input_flags=flags)
    sim.start_simulation(t_end=T_END, dt=None, operator_split=None,
                         max_steps=MAX_STEPS)
    return sim.get_results()


def raw(a):
    return np.ascontiguousarray(np.asarray(a, dtype=float)).view(np.uint64)


# The config echoes are EXPECTED to differ: one carries the keys, the other
# does not. That is the input, not an output, so they are excluded by name and
# the exclusion is stated rather than silent.
CONFIG_ECHOES = {"params", "flags"}


def compare(label, a, b):
    fields = [f for f in sorted(vars(a)) if f not in CONFIG_ECHOES]
    checked = 0
    bad = []
    for name in fields:
        try:
            va, vb = getattr(a, name), getattr(b, name)
        except Exception:
            continue
        if callable(va):
            continue
        if isinstance(va, np.ndarray) and va.dtype.kind == "f":
            checked += 1
            if va.shape != vb.shape or not np.array_equal(raw(va), raw(vb)):
                bad.append(name)
        elif isinstance(va, float):
            checked += 1
            if raw(np.array([va]))[0] != raw(np.array([vb]))[0]:
                bad.append(name)
        elif isinstance(va, dict):
            if set(va) != set(vb):
                bad.append(f"{name}: key sets differ")
                continue
            for k in sorted(va):
                x, y = va[k], vb[k]
                if isinstance(x, np.ndarray) and x.dtype.kind == "f":
                    checked += 1
                    if not np.array_equal(raw(x), raw(y)):
                        bad.append(f"{name}[{k}]")
                elif isinstance(x, float):
                    checked += 1
                    if raw(np.array([x]))[0] != raw(np.array([y]))[0]:
                        bad.append(f"{name}[{k}]")
    print(f"  {label}: {checked} float fields/entries compared at raw uint64 "
          f"-> {'BIT-IDENTICAL' if not bad else 'MISMATCH ' + str(bad)}")
    return not bad


ok = True
_t0 = _time.time()
for label, kwargs in (
    ("fa2-class: S_gp 9010, neutral_momentum, two-zone", {}),
    ("single-zone, neutral_momentum", {"two_zone": False}),
    ("no optional neutral fields (historical 5-row layout)",
     {"two_zone": False, "momentum": False}),
):
    a = run(False, **kwargs)
    print(f"  ... arm A done ({_time.time() - _t0:.0f}s)", flush=True)
    b = run(True, **kwargs)
    print(f"  ... arm B done ({_time.time() - _t0:.0f}s)", flush=True)
    print(f"[{label}]  status={a.run_status}/{b.run_status} "
          f"t_end={a.time[-1]:.12e} vs {b.time[-1]:.12e} "
          f"steps={a.steps}/{b.steps} "
          f"packed width={a.y.shape[1] // a.n.shape[1]} rows "
          f"En saved={hasattr(a, 'En')}")
    ok &= compare("absent vs explicit (False, 0.40, 'frozen')", a, b)

print()
print("A/B RESULT:",
      "PASS -- the registered defaults are bit-exact" if ok else "FAIL")
