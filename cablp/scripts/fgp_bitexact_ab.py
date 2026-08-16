"""Bit-exact A/B: key ABSENT vs key EXPLICITLY 1.0, over a short real run.

Raw uint64 comparison of every saved field -- not a tolerance.
Run with PYTHONPATH=<worktree>/cablp.
"""
import numpy as np

import cablp
from cablp.solvers._sim1d import LAPDSim1D, default_config

print("cablp package file:", cablp.__file__)

# Step-BOUNDED so the arm has a known cost. The budget is chosen to carry the
# run past the "square" opening edge (centre 5e-4 s, width 5e-4 s) so the puff
# term is genuinely open and non-trivial, rather than still at the foot of the
# erf; the reached time and the applied puff level are reported per arm so the
# window is auditable rather than asserted.
MAX_STEPS = 20000
T_END = 3.0e-3


def run(with_key, two_zone=False, profile=None):
    params, flags = default_config()
    params["nx"] = 24
    params["max_steps_action"] = "stop"
    if profile is not None:
        params["gas_puff_profile"] = profile
    if two_zone:
        flags["neutral_two_zone"] = True
    if with_key:
        params["gas_puff_delivery_fraction"] = 1.0
    else:
        params.pop("gas_puff_delivery_fraction", None)
    assert ("gas_puff_delivery_fraction" in params) is with_key
    sim = LAPDSim1D(input_dict=params, input_flags=flags)
    sim.start_simulation(t_end=T_END, dt=None, operator_split=None,
                         max_steps=MAX_STEPS)
    return sim.get_results()


def raw(a):
    return np.ascontiguousarray(np.asarray(a, dtype=float)).view(np.uint64)


# The config echoes are EXPECTED to differ: one carries the key, the other
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


import time as _time

ok = True
_t0 = _time.time()
for label, kwargs in (
    ("single-zone, cosine_pipe (shipped default)", {}),
    ("single-zone, cell profile", {"profile": "cell"}),
    ("two-zone (annulus puff routing)", {"two_zone": True}),
):
    a = run(False, **kwargs)
    print(f"  ... arm A done ({_time.time() - _t0:.0f}s)", flush=True)
    b = run(True, **kwargs)
    print(f"  ... arm B done ({_time.time() - _t0:.0f}s)", flush=True)
    pa = a.gas_puff_diagnostics["puff_particles_per_s"]
    print(f"[{label}]  status={a.run_status}/{b.run_status} "
          f"t_end={a.time[-1]:.12e} vs {b.time[-1]:.12e} "
          f"steps={a.steps} peak puff_particles_per_s={float(np.max(pa)):.6e} "
          f"final={float(pa[-1]):.6e}")
    ok &= compare("absent vs explicit 1.0", a, b)

print()
print("A/B RESULT:", "PASS -- default 1.0 is bit-exact" if ok else "FAIL")
