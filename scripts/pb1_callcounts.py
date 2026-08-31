"""[perf-batch-1] Exact per-accepted-step call counts for the batch's hot sites.

cProfile gives exact counts but only for what survives its ``--top`` cut, and
it inflates the run 2.2x. This instrument counts a NAMED, SHORT list of
functions by wrapping them, runs the golden-at-stance config un-profiled, and
prints calls and calls-per-accepted-step. Untracked run instrument, not a gate.

    python scripts/pb1_callcounts.py --steps 1000
"""

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from baseline_sim1d import BASELINE_RUN_KWARGS, build_baseline_config  # noqa: E402
from cablp.cathode.kernels import PROVENANCE as KERNEL_PROVENANCE  # noqa: E402
from cablp.solvers._sim1d import LAPDSim1D  # noqa: E402

COUNTS = {}


def _wrap(holder, attr, label):
    """Count calls to ``holder.attr`` through EVERY module that bound it.

    ``from x import f`` copies the function into the importer's globals, so
    patching only the defining module misses every consumer that imported the
    name directly -- which silently reports zero calls.
    """
    original = getattr(holder, attr, None)
    if original is None:
        return None
    COUNTS[label] = 0

    def counted(*args, **kwargs):
        COUNTS[label] += 1
        return original(*args, **kwargs)

    patched = []
    for module in list(sys.modules.values()):
        if module is None or not getattr(module, "__name__", "").startswith(
            ("cablp", "__main__")
        ):
            continue
        if getattr(module, attr, None) is original:
            setattr(module, attr, counted)
            patched.append(module)
    if getattr(holder, attr, None) is original:
        setattr(holder, attr, counted)
        patched.append(holder)
    return (patched, attr, original)


def targets():
    """Return the (holder, attr, label) triples this batch measures."""
    from cablp.atomic import adas
    from cablp.plasma import params as plasma_params
    from cablp.solvers._sim1d.core import state as core_state
    from cablp.solvers._sim1d.core import validation as core_validation
    from cablp.solvers._sim1d.physics import cathode as physics_cathode

    return [
        (adas, "_interp_coords", "adas._interp_coords"),
        (adas, "_interp_blend", "adas._interp_blend"),
        (adas, "he_rates", "adas.he_rates"),
        (physics_cathode, "_array_fingerprint", "cathode._array_fingerprint"),
        (physics_cathode, "_beam_smoothing_key", "cathode._beam_smoothing_key"),
        (core_validation, "_bad_array_summary", "validation._bad_array_summary"),
        (core_validation, "validate_raw_stage", "validation.validate_raw_stage"),
        (core_state, "unpack_state", "state.unpack_state"),
        (plasma_params, "c_log", "params.c_log"),
        (LAPDSim1D, "rhs_terms", "solver.rhs_terms"),
        (LAPDSim1D, "_step_rejection_info", "solver._step_rejection_info"),
        (LAPDSim1D, "_zero_rhs_state", "solver._zero_rhs_state"),
        (
            LAPDSim1D,
            "_implicit_neutral_step_two_zone",
            "solver._implicit_neutral_step_two_zone",
        ),
    ]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1000)
    args = ap.parse_args(argv)

    restore = [t for t in (_wrap(*x) for x in targets()) if t is not None]
    print(f"kernels={KERNEL_PROVENANCE} steps={args.steps}")

    params, flags = build_baseline_config({"max_steps_action": "stop"})
    sim = LAPDSim1D(params, flags)
    kwargs = dict(BASELINE_RUN_KWARGS)
    kwargs["max_steps"] = int(args.steps)
    try:
        sim.start_simulation(**kwargs)
    finally:
        for holders, attr, original in restore:
            for holder in holders:
                setattr(holder, attr, original)

    steps = float(args.steps)
    width = max(len(k) for k in COUNTS)
    for label in sorted(COUNTS, key=lambda k: -COUNTS[k]):
        n = COUNTS[label]
        print(f"{label:<{width}}  calls={n:<10} per_step={n / steps:9.4f}")
    rhs = COUNTS.get("solver.rhs_terms", 0)
    if rhs:
        for label in ("adas._interp_coords", "adas._interp_blend", "adas.he_rates"):
            print(
                f"{label:<{width}}  per_rhs_terms={COUNTS.get(label, 0) / rhs:9.4f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
