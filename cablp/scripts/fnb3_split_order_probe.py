"""fnb3 -- does splitting neutral_hot_channel into TWO ledger rows move the RHS?

``LAPDSim1D.rhs`` accumulates the term ledger in dict order,
``state_rhs = add_state_rhs(state_rhs, term)`` (solver.py:4114-4115). The hot
channel currently contributes ONE array per field, e.g. ``Ei = dEi_recx +
dEi_ion``. Splitting it into two ledger rows would instead contribute
``dEi_recx`` and ``dEi_ion`` as two separate accumulator additions, turning

    S + (a + b)        into        (S + a) + b

which is the same number in exact arithmetic and NOT the same double. This
probe measures the difference on the real flag-ON state, at raw uint64, before
anything is built on the assumption either way. It mutates nothing.

Run with PYTHONPATH=<worktree>/cablp.
"""
import numpy as np

import cablp
from cablp.solvers._sim1d import LAPDSim1D, default_config
from cablp.solvers._sim1d.core.state import STATE_NAMES_1D
from cablp.solvers._sim1d.physics.sources import add_state_rhs

print("cablp package file:", cablp.__file__)

params, flags = default_config()
params["nx"] = 24
params["S_gp"] = 9010
params["max_steps_action"] = "stop"
flags["neutral_momentum"] = True
flags["neutral_two_zone"] = True
flags["neutral_energy"] = True

sim = LAPDSim1D(input_dict=params, input_flags=flags)
# Step off the initial condition so the hot channel is genuinely live rather
# than sitting on a seed state whose rows are all zero.
sim.start_simulation(t_end=3.0e-4, dt=None, operator_split=None, max_steps=4000)
print(f"stepped to t={sim.time:.6e} s")

terms = sim.rhs_terms(include_heat_conduction=True, time=sim.time)
names = list(terms)
print(f"ledger terms: {len(names)}")
target = "neutral_hot_channel"
print(f"{target} at ledger position {names.index(target)} of {len(names)}")

# The two halves the term's Ei and M rows are the sum of, recomputed exactly
# the way hot_neutrals.py builds them, by re-running the term with the
# in-flight ionization channel switched off and on. Rather than re-derive them,
# the probe uses the WEAKEST possible form of the question: split the term's
# own rows into ANY two arrays a, b with a + b bit-identical to the row, and
# ask whether the accumulation still lands on the same bits. If it does not for
# a representative split, it does not for the physical one either.
hot = terms[target]


def halves(row):
    """Return (a, b) with a + b bit-identical to ``row`` (a genuine split)."""
    a = 0.5 * np.asarray(row, dtype=float)
    b = np.asarray(row, dtype=float) - a
    assert np.array_equal((a + b).view(np.uint64),
                          np.ascontiguousarray(row).view(np.uint64))
    return a, b


def accumulate(term_list):
    acc = sim._zero_rhs_state()
    for term in term_list:
        acc = add_state_rhs(acc, term)
    return acc


from cablp.solvers._sim1d.core.state import ConservativeState1D

split_a = {}
split_b = {}
for field in STATE_NAMES_1D:
    value = getattr(hot, field, None)
    if value is None:
        split_a[field] = None
        split_b[field] = None
        continue
    a, b = halves(value)
    split_a[field] = a
    split_b[field] = b

combined_list = list(terms.values())
split_list = []
for name, term in terms.items():
    if name != target:
        split_list.append(term)
        continue
    split_list.append(ConservativeState1D(**split_a))
    split_list.append(ConservativeState1D(**split_b))

combined = accumulate(combined_list)
split = accumulate(split_list)

print("\nfield-by-field comparison of the accumulated RHS, raw uint64:")
worst_any = False
for field in STATE_NAMES_1D:
    x = getattr(combined, field, None)
    y = getattr(split, field, None)
    if x is None or y is None:
        continue
    x = np.ascontiguousarray(np.asarray(x, dtype=float))
    y = np.ascontiguousarray(np.asarray(y, dtype=float))
    same = np.array_equal(x.view(np.uint64), y.view(np.uint64))
    ndiff = int(np.count_nonzero(x.view(np.uint64) != y.view(np.uint64)))
    worst = float(np.max(np.abs(x - y))) if x.size else 0.0
    scale = max(float(np.max(np.abs(x))), 1e-300) if x.size else 1.0
    worst_any |= not same
    print(f"  {field:5s} bit-identical={str(same):5s} cells differing="
          f"{ndiff:3d}/{x.size:3d}  max|d|={worst:.6e}  rel={worst/scale:.3e}")

print()
print("VERDICT:", "a ledger SPLIT MOVES the accumulated RHS -- an rhs_terms "
      "split cannot be bit-exact" if worst_any else
      "the accumulation is insensitive to the split on this state")
