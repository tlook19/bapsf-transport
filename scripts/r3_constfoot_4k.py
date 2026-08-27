"""Short-horizon FINAL-STATE footprint instrument for constant-value changes.

The golden (``baseline_sim1d.py --verify``) answers "did the trajectory move";
this answers "by how much, per field". It is the A/B instrument for a
NUMERICS-MOVING change whose whole point is that the golden is expected to
move, so a pass/fail fixture gate has nothing to say about it.

* the configuration is EXACTLY the golden's, built by
  ``baseline_sim1d.build_baseline_config`` (``default_config()`` + the committed
  stance file ``scripts/stances/g1atrim.toml`` minus its mesh-sized package +
  ``BASELINE_PARAM_OVERRIDES``/``BASELINE_FLAG_OVERRIDES``, so ``nx = 60`` and
  the pinned scalar neutral fill travel with it), reused rather than
  re-implemented so it cannot drift away from the golden's own config;
* the solver is advanced ``FOOTPRINT_STEPS`` accepted steps and the FINAL
  accepted state is written out per field -- the five conservative fields
  ``n``/``nn``/``M``/``Ee``/``Ei`` and the two derived temperatures
  ``Te``/``Ti``, taken from the solver's own ``state``/``derived`` properties;
* ``--compare A.npz B.npz`` prints, per field, the max and RMS relative
  deviation of B against A and whether the field is bit-identical.

Like ``golden_digest_gate.py`` this pins ``max_steps_action = "stop"``: the
golden pins ``"raise"`` because there ``max_steps`` is a tripwire, whereas here
the cap IS the run length. It changes what happens AT the cap and nothing
before it.

This writes no committed fixture and gates nothing. It MEASURES: the arm
``.npz`` files are run artifacts, and the reference arm is whatever code state
the operator captured it from, recorded in the file's own ``provenance`` entry.

Usage::

    # capture one arm (the code state currently on disk)
    python scripts/r3_constfoot_4k.py --out scripts/r3const_ref.npz

    # compare two arms, field by field
    python scripts/r3_constfoot_4k.py --compare scripts/r3const_ref.npz \
        scripts/r3const_qe.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from cablp.cathode.kernels import PROVENANCE as KERNEL_PROVENANCE
from cablp.solvers._sim1d import LAPDSim1D

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
# The golden's own config-construction path, reused rather than re-implemented.
from baseline_sim1d import (  # noqa: E402
    BASELINE_RUN_KWARGS,
    build_baseline_config,
)

#: Horizon of the measurement, in ACCEPTED steps. A cost knob, not physics.
#: Matched to ``golden_digest_gate.DIGEST_STEPS`` so the two instruments speak
#: about the same window of the same run.
FOOTPRINT_STEPS = 4000

#: The one departure from the golden pins -- see the module docstring.
FOOTPRINT_PARAM_OVERRIDES = {"max_steps_action": "stop"}

#: Fields written out: the packed conservative state, plus the two derived
#: temperatures (which is where a rate-coefficient change shows up first).
CONSERVATIVE_FIELDS = ("n", "nn", "M", "Ee", "Ei")
DERIVED_FIELDS = ("Te", "Ti")

#: Format tag, so a future layout change is legible rather than a silent
#: mismatch on every field at once.
FOOTPRINT_FORMAT = "sim1d-constfoot-v1"


def _step_counter():
    """Return a progress tracker that just counts accepted steps."""

    class _Counter:
        def __init__(self):
            self.steps = 0

        def update(self, progress):
            self.steps = int(progress.step)

    return _Counter()


def run_footprint(steps=FOOTPRINT_STEPS):
    """Run ``steps`` accepted steps of the golden config; return a field dict.

    Raises ``RuntimeError`` if the run ends before ``steps`` accepted steps: a
    short run's final state is not comparable to a full one and must not be
    reported as a deviation.
    """
    params, flags = build_baseline_config(FOOTPRINT_PARAM_OVERRIDES)
    tracker = _step_counter()
    # Bound at CONSTRUCTION with no tracker passed to start_simulation, for the
    # reason golden_digest_gate.py records: start_simulation forwards its own
    # progress arguments to the neutral equilibration, which builds a SEPARATE
    # inner solver, so a tracker passed there would count its steps as well.
    sim = LAPDSim1D(params, flags, progress_tracker=tracker, progress_interval_s=0.0)

    kwargs = dict(BASELINE_RUN_KWARGS)
    kwargs["max_steps"] = int(steps)
    sim.start_simulation(**kwargs)

    if tracker.steps != int(steps):
        raise RuntimeError(
            f"footprint run took {tracker.steps} accepted steps, expected "
            f"{steps}: the run ended before the horizon (t_end or an abort "
            "reached first), so its final state is not comparable to a full one"
        )

    # sim.state / sim.derived unpack the solver's own accepted packed vector,
    # so this is the state the 4001st step would have started from.
    state = sim.state
    derived = sim.derived
    out = {name: np.asarray(getattr(state, name), dtype=float)
           for name in CONSERVATIVE_FIELDS}
    out.update({name: np.asarray(getattr(derived, name), dtype=float)
                for name in DERIVED_FIELDS})
    out["steps"] = np.asarray(int(steps))
    out["cells"] = np.asarray(int(sim.geometry.cells))
    out["final_time"] = np.asarray(float(sim.get_results().final_time))
    out["provenance"] = np.asarray(
        f"{FOOTPRINT_FORMAT} kernels={KERNEL_PROVENANCE}"
    )
    return out


def capture(out_path, steps=FOOTPRINT_STEPS):
    """Run one arm and write its final state to ``out_path`` as an NPZ."""
    fields = run_footprint(steps=steps)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **fields)
    print(
        f"footprint captured: {out_path}, steps={int(fields['steps'])}, "
        f"cells={int(fields['cells'])}, "
        f"final_time={float(fields['final_time']):.9e}, "
        f"kernels={KERNEL_PROVENANCE}"
    )
    return 0


def _relative_deviation(a, b):
    """Return ``(max_rel, rms_rel, n_differing)`` of ``b`` against ``a``.

    Relative to ``|a|`` elementwise; a cell where ``a`` is exactly zero
    contributes its ABSOLUTE difference rather than being dropped or turned
    into an infinity, so an exactly-zero reference cell that moves is still
    visible and still finite.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    diff = np.abs(b - a)
    scale = np.where(a == 0.0, 1.0, np.abs(a))
    rel = diff / scale
    differing = int(np.count_nonzero(
        a.view(np.uint64) != b.view(np.uint64)
    ))
    return float(np.max(rel)), float(np.sqrt(np.mean(rel * rel))), differing


def compare(ref_path, arm_path):
    """Print the per-field deviation of ``arm_path`` against ``ref_path``."""
    ref = np.load(ref_path, allow_pickle=False)
    arm = np.load(arm_path, allow_pickle=False)

    print(f"reference : {ref_path}  ({ref['provenance']})")
    print(f"arm       : {arm_path}  ({arm['provenance']})")
    if int(ref["steps"]) != int(arm["steps"]) or int(ref["cells"]) != int(arm["cells"]):
        print(
            "  MISMATCHED RUN SHAPE: "
            f"ref steps={int(ref['steps'])} cells={int(ref['cells'])} vs "
            f"arm steps={int(arm['steps'])} cells={int(arm['cells'])} -- "
            "the two arms did not run the same problem"
        )
        return 1

    names = CONSERVATIVE_FIELDS + DERIVED_FIELDS
    print(f"{'field':>6}  {'max_rel':>12}  {'rms_rel':>12}  {'differing':>9}")
    all_identical = True
    for name in names:
        max_rel, rms_rel, differing = _relative_deviation(ref[name], arm[name])
        if differing:
            all_identical = False
        print(f"{name:>6}  {max_rel:12.6e}  {rms_rel:12.6e}  {differing:9d}")
    dt_rel = _relative_deviation(ref["final_time"], arm["final_time"])[0]
    print(f"{'t_end':>6}  {dt_rel:12.6e}")
    print("BIT-IDENTICAL" if all_identical else "MOVED")
    return 0


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description=(
            "Final-state footprint of the sim1d golden configuration after a "
            "short accepted-step horizon, and the A/B comparison of two such "
            "captures."
        )
    )
    parser.add_argument(
        "--out",
        help="Capture one arm and write its final state to this NPZ path.",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("REF", "ARM"),
        help="Compare two captured arms field by field (ARM against REF).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=FOOTPRINT_STEPS,
        help=f"Accepted-step horizon for a capture (default {FOOTPRINT_STEPS}).",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    if args.compare:
        return compare(*args.compare)
    if args.out:
        return capture(args.out, steps=args.steps)
    print("nothing to do: pass --out to capture an arm or --compare to compare two")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
