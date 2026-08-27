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

**Read ``--fixed-dt`` before reading any deviation this prints.** On the
ADAPTIVE grid this instrument does not resolve a smooth footprint at all: a
1-ulp perturbation flips the timestep selector at a near-tie and the arms end
4,000 steps apart in step count but at DIFFERENT physical times, so the
deviation is dominated by that time offset (measured floor ~1e-2, and response
is not monotone in perturbation size). ``--fixed-dt`` pins the grid so every
arm walks the same instants; that is the mode a footprint should be read from.

Usage::

    # capture one arm (the code state currently on disk)
    python scripts/r3_constfoot_4k.py --out scripts/r3const_ref.npz

    # the same, on the matched grid -- what a footprint should be read from
    python scripts/r3_constfoot_4k.py --fixed-dt --out scripts/r3const_v2_ref.npz

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

#: THE MATCHED GRID (``--fixed-dt``). The adaptive-dt battery could not resolve
#: any smooth footprint: a perturbation as small as 1 ulp flips the timestep
#: selector at a near-tie, the arms leave the shared time grid, and after a
#: fixed STEP COUNT they sit at different physical times -- so the comparison
#: measures the time offset, not the constant. Forcing dt removes that channel
#: entirely: every arm then walks the SAME 4,000 instants and lands on the same
#: t_end by construction, so a field difference can only have come from the
#: arithmetic.
#:
#: The value is the adaptive reference arm's own OBSERVED constant step
#: (4000 * 1.175e-7 = 4.7e-4 s exactly), so the matched grid is the grid the
#: golden config actually chose for itself over this horizon rather than a
#: number invented for the instrument.
#:
#: This is a DRIVER-side control: ``run()`` takes ``dt`` and uses it verbatim
#: (``step_dt = diag.dt if dt is None else float(dt)``), bypassing the limiter,
#: the dt-growth ramp and the dt_min clamp. No solver code is involved.
FIXED_DT_S = 1.175e-7


def _step_counter():
    """Return a progress tracker that just counts accepted steps."""

    class _Counter:
        def __init__(self):
            self.steps = 0

        def update(self, progress):
            self.steps = int(progress.step)

    return _Counter()


def run_footprint(steps=FOOTPRINT_STEPS, fixed_dt=None):
    """Run ``steps`` accepted steps of the golden config; return a field dict.

    ``fixed_dt`` forces every step to that size (the matched grid; see
    ``FIXED_DT_S``). ``None`` leaves the adaptive selector in charge.

    Raises ``RuntimeError`` if the run ends before ``steps`` accepted steps: a
    short run's final state is not comparable to a full one and must not be
    reported as a deviation. In fixed-dt mode it additionally raises if the
    grid did not come out matched -- see below.
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
    if fixed_dt is not None:
        kwargs["dt"] = float(fixed_dt)
    sim.start_simulation(**kwargs)

    if tracker.steps != int(steps):
        raise RuntimeError(
            f"footprint run took {tracker.steps} accepted steps, expected "
            f"{steps}: the run ended before the horizon (t_end or an abort "
            "reached first), so its final state is not comparable to a full one"
        )

    result = sim.get_results()
    if fixed_dt is not None:
        # THE MATCHED-GRID ASSERTION, and it is not a formality: a caller-
        # supplied dt fixes the step the loop ATTEMPTS, but a REJECTED attempt
        # still retries at a shortened dt (the loop relabels that step's cap
        # "retry"). One retry anywhere would put this arm on its own time grid
        # and silently turn the comparison back into the time-offset
        # measurement that fixed dt exists to eliminate. Both halves are
        # checked: no rejection was recorded, AND the arithmetic landed exactly
        # where an unretried grid must land.
        rejections = int(
            np.asarray(result.timestep_rejection_events["time"]).size
        )
        # The nominal end time is only a COARSE check, and deliberately a
        # tolerant one: the loop reaches t by accumulating ``t += dt`` 4,000
        # times, which is not float-equal to ``4000 * dt`` (a few hundred ulp
        # of accumulation, ~1e-14 relative). Demanding exact equality here
        # rejects a perfectly matched grid. What it IS sized to catch is a
        # gross divergence -- a shortened or skipped step -- which lands ~1e-3
        # away, eleven orders above the accumulation.
        #
        # The SHARP matched-grid test is not available here at all, because it
        # is a statement about two arms rather than one: every fixed-dt arm
        # accumulates the same t from the same dt in the same order, so their
        # final_times must agree BIT FOR BIT. ``compare`` enforces that.
        expected_t_end = float(steps) * float(fixed_dt)
        drift = abs(float(result.final_time) - expected_t_end) / expected_t_end
        if rejections or drift > 1e-9:
            raise RuntimeError(
                "fixed-dt grid NOT matched: "
                f"{rejections} rejected step(s), final_time="
                f"{float(result.final_time)!r} vs nominal "
                f"{expected_t_end!r} (relative drift {drift:.3e}). This arm "
                "walked a different set of instants from an unretried arm, so "
                "its final state is not comparable and must not be reported "
                "as a footprint."
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
    out["final_time"] = np.asarray(float(result.final_time))
    out["fixed_dt"] = np.asarray(0.0 if fixed_dt is None else float(fixed_dt))
    out["provenance"] = np.asarray(
        f"{FOOTPRINT_FORMAT} kernels={KERNEL_PROVENANCE} "
        f"grid={'fixed-dt' if fixed_dt is not None else 'adaptive'}"
    )
    return out


def capture(out_path, steps=FOOTPRINT_STEPS, fixed_dt=None):
    """Run one arm and write its final state to ``out_path`` as an NPZ."""
    fields = run_footprint(steps=steps, fixed_dt=fixed_dt)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **fields)
    print(
        f"footprint captured: {out_path}, steps={int(fields['steps'])}, "
        f"cells={int(fields['cells'])}, "
        f"final_time={float(fields['final_time']):.9e}, "
        f"grid={'fixed-dt ' + repr(fixed_dt) if fixed_dt is not None else 'adaptive'}, "
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
    # Comparing a fixed-dt arm against an adaptive one would read as a
    # footprint while actually being the grid difference, which is the exact
    # confusion this mode exists to end. Older captures carry no fixed_dt key
    # and are treated as adaptive.
    ref_dt = float(ref["fixed_dt"]) if "fixed_dt" in ref.files else 0.0
    arm_dt = float(arm["fixed_dt"]) if "fixed_dt" in arm.files else 0.0
    if ref_dt != arm_dt:
        print(
            "  MISMATCHED GRID MODE: "
            f"ref fixed_dt={ref_dt!r} vs arm fixed_dt={arm_dt!r} -- one arm "
            "walked the adaptive grid and the other a forced one, so this "
            "comparison would report the grid difference as a footprint"
        )
        return 1

    # THE SHARP MATCHED-GRID TEST. Two fixed-dt arms accumulate the same t from
    # the same dt in the same order, so their end times must agree BIT FOR BIT
    # regardless of what the constants did to the state. Any difference means
    # they did not walk the same instants, and every field deviation below
    # would then be partly a time offset -- the confounder this mode exists to
    # remove. Refused rather than reported.
    if ref_dt and arm_dt:
        ref_t = float(ref["final_time"])
        arm_t = float(arm["final_time"])
        if ref_t != arm_t:
            print(
                "  GRID NOT MATCHED: fixed-dt arms disagree on final_time "
                f"({ref_t!r} vs {arm_t!r}) -- they did not walk the same "
                "instants, so these fields are not comparable as a footprint"
            )
            return 1
        print(f"matched grid confirmed: both arms end at t={ref_t!r} (bit-identical)")

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
    parser.add_argument(
        "--fixed-dt",
        nargs="?",
        type=float,
        const=FIXED_DT_S,
        default=None,
        metavar="SECONDS",
        help=(
            "Force every step to this size (default "
            f"{FIXED_DT_S}, the adaptive reference arm's own observed constant "
            "step), so all arms walk the SAME instants and the comparison "
            "cannot pick up a time offset. Refuses the capture if any step was "
            "rejected and retried, which would break the matched grid."
        ),
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    if args.compare:
        return compare(*args.compare)
    if args.out:
        return capture(args.out, steps=args.steps, fixed_dt=args.fixed_dt)
    print("nothing to do: pass --out to capture an arm or --compare to compare two")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
