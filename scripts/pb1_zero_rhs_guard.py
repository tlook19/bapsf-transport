"""Guard and bit-exactness checks for the shared zero RHS bundle.

``_zero_rhs_state`` now returns ONE read-only all-zero bundle per solver.
Four checks:

  (a) the bundle is shared -- repeated calls return the same object, and every
      one of its five rows is read-only, so an in-place write RAISES;
  (b) adding an all-zero bundle to the accumulator is bit-exact on the values
      the solver actually reaches: over N real golden steps, every accumulator
      state inside ``rhs`` satisfies ``acc + zeros == acc`` at raw uint64, on
      every row;
  (c) the induction (b) rests on holds in practice: no accumulator entry is
      ever -0.0 -- the one value for which ``x + (+0.0)`` would not return
      ``x``'s own bit pattern;
  (d) the SKIP OPPORTUNITY, which is why the brief's optional
      "skip flag-structural zero terms in the rhs_terms sum" was NOT taken:
      how many of the terms reaching the sum are the shared object (skippable
      by identity) versus merely all-zero by value (not skippable without an
      O(cells) scan per term). The identity count is the measured ceiling of
      that optimization.
"""

import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from baseline_sim1d import BASELINE_RUN_KWARGS, build_baseline_config  # noqa: E402
from cablp.solvers._sim1d import LAPDSim1D  # noqa: E402
from cablp.solvers._sim1d.core.state import STATE_NAMES_1D  # noqa: E402
from cablp.solvers._sim1d.physics import sources as _sources  # noqa: E402
from cablp.solvers._sim1d import solver as _solver  # noqa: E402

STEPS = 60


def main():
    failures = []
    params, flags = build_baseline_config({"max_steps_action": "stop"})
    sim = LAPDSim1D(params, flags)

    # (a) sharing and the read-only guard.
    zero_a = sim._zero_rhs_state()
    zero_b = sim._zero_rhs_state()
    print(f"(a) repeated calls return the same object: {zero_a is zero_b}")
    if zero_a is not zero_b:
        failures.append("zero bundle is not shared")
    for name in STATE_NAMES_1D:
        row = getattr(zero_a, name)
        try:
            row[0] = 1.0
        except ValueError as exc:
            print(f"(a) write to {name} raised ValueError: {exc}")
        else:
            print(f"(a) write to {name} SUCCEEDED -- the guard is not armed")
            failures.append(f"{name} row is writeable")

    # (d) how many terms reaching the sum could a skip have elided?
    original_rhs_terms = LAPDSim1D.rhs_terms
    opportunity = {"terms": 0, "shared_object": 0, "zero_by_value": 0}

    def counting_rhs_terms(self, *args, **kwargs):
        terms = original_rhs_terms(self, *args, **kwargs)
        shared = self._zero_rhs_state()
        for term in terms.values():
            opportunity["terms"] += 1
            if term is shared:
                opportunity["shared_object"] += 1
            elif not any(
                np.any(np.asarray(getattr(term, name), dtype=float))
                for name in STATE_NAMES_1D
            ):
                opportunity["zero_by_value"] += 1
        return terms

    LAPDSim1D.rhs_terms = counting_rhs_terms

    # (b) and (c): re-do the elidable add over real accumulator states.
    stats = {"adds": 0, "differing": 0, "neg_zero": 0}
    original_add = _sources.add_state_rhs

    def checking_add(left, right):
        # `left` is the accumulator at this point in the sum. Verify that the
        # add the skip elides -- left + zeros -- would have returned left's own
        # bits, and that left carries no -0.0.
        for name in STATE_NAMES_1D:
            values = np.asarray(getattr(left, name), dtype=float)
            summed = values + np.zeros_like(values)
            stats["adds"] += 1
            stats["differing"] += int(
                np.count_nonzero(
                    np.ascontiguousarray(values).view(np.uint64)
                    != np.ascontiguousarray(summed).view(np.uint64)
                )
            )
            stats["neg_zero"] += int(
                np.count_nonzero(np.signbit(values) & (values == 0.0))
            )
        return original_add(left, right)

    _solver.add_state_rhs = checking_add
    try:
        kwargs = dict(BASELINE_RUN_KWARGS)
        kwargs["max_steps"] = STEPS
        sim.start_simulation(**kwargs)
    finally:
        _solver.add_state_rhs = original_add
        LAPDSim1D.rhs_terms = original_rhs_terms

    print(
        f"(b) accumulator rows re-added with zeros: {stats['adds']}; "
        f"differing uint64 words: {stats['differing']}"
    )
    print(f"(c) accumulator entries equal to -0.0: {stats['neg_zero']}")
    total = opportunity["terms"]
    print(
        f"(d) terms reaching the sum: {total}; "
        f"skippable by identity: {opportunity['shared_object']} "
        f"({opportunity['shared_object'] / total:.4%}); "
        f"all-zero by value only: {opportunity['zero_by_value']} "
        f"({opportunity['zero_by_value'] / total:.4%})"
        if total
        else "(d) no terms observed"
    )
    if stats["adds"] == 0:
        failures.append("no accumulator states were observed")
    if stats["differing"]:
        failures.append("skipping the zero add would move a bit")
    if stats["neg_zero"]:
        failures.append("an accumulator entry was -0.0")

    if failures:
        print("ZERO RHS SHARING FAIL: " + "; ".join(failures))
        return 1
    print("ZERO RHS SHARING OK: shared, guarded, and the skip moves no bit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
