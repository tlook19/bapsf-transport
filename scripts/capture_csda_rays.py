"""Capture real ``deposit_beam`` argument tuples from a production run.

The companion to ``spike_csda_march.py``: that script needs
PRODUCTION-REALISTIC ray arguments to compare the compiled CSDA march against
the pure one, and reconstructing them from a saved HDF5 would mean re-deriving
the cathode solve's ``(E0, Gamma0, launch, direction, area)`` outside the
solver. Recording the calls as they happen is exact and needs no
reconstruction.

The run is the golden fixture's own configuration (``baseline_sim1d``), so the
captured states ARE the production stance. One snapshot is kept per requested
time window, the FIRST call at or after each target time, and the captured
arrays are copied because the solver reuses its buffers.

The pre-breakdown long-mfp window (t ~ 2.01 ms) is in the default target list
deliberately: the 2026-08-02 cost read identified it as the outlier the main
discharge never exercises (E0 ~ 218 eV, ~366 substeps, a ray spanning 260
cells, against a main-discharge ray that dies in its launch cell).

Sampling that window by TIME is not enough on its own, though -- the widest
ray is a narrow, ~50 us feature, and which call lands on it depends on the
adaptive step. So the run also keeps a running WIDEST-RAY champion, judged by
how many cells the ray actually reaches (``E_entry_eV > 0``), read straight
off the result the real call already produced. That costs one array reduction
per call and finds the outlier whatever time it happens at.

Usage::

    python scripts/capture_csda_rays.py --output scripts/csda_rays_prod.pkl
    # the long-mfp window alone, in ~1 minute:
    python scripts/capture_csda_rays.py --t-end 2.1e-3 \\
        --output scripts/csda_rays_longmfp.pkl
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import baseline_sim1d as baseline  # noqa: E402
import cablp.solvers._sim1d.physics.cathode as cathode_mod  # noqa: E402

# Target times [s] and the phase label each is meant to sample.
DEFAULT_TARGETS = (
    (2.010e-3, "pre_breakdown_long_mfp"),
    (2.400e-3, "breakdown"),
    (3.000e-3, "early_discharge"),
    (6.000e-3, "discharge"),
    (12.00e-3, "plateau"),
    (20.00e-3, "plateau_end"),
)


def capture(targets, t_end):
    """Run the baseline config, returning one recorded call per target time."""
    real_deposit = cathode_mod.deposit_beam
    pending = list(targets)
    captured = []
    widest = {"span": -1, "state": None}
    counters = {"calls": 0, "steps": 0}

    def snapshot(label, t_target, now, args, kwargs):
        return {
            "label": label,
            "t_target_s": t_target,
            "t_actual_s": now,
            "args": tuple(
                np.array(a, copy=True) if isinstance(a, np.ndarray) else a
                for a in args
            ),
            "kwargs": {
                k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v)
                for k, v in kwargs.items()
            },
        }

    def recording_deposit(*args, **kwargs):
        counters["calls"] += 1
        now = float(sim._time)
        if pending and now >= pending[0][0]:
            t_target, label = pending.pop(0)
            captured.append(snapshot(label, t_target, now, args, kwargs))
        result = real_deposit(*args, **kwargs)
        # Widest-ray champion: how many cells the primary actually reached.
        span = int(np.count_nonzero(result.E_entry_eV > 0.0))
        if span > widest["span"]:
            widest["span"] = span
            widest["state"] = snapshot("widest_ray", float("nan"), now, args, kwargs)
        return result

    params, flags = baseline.build_baseline_config()
    sim = baseline.LAPDSim1D(params, flags)
    cathode_mod.deposit_beam = recording_deposit
    try:
        sim.start_simulation(
            t_end=t_end, dt=None, operator_split=None, max_steps=None
        )
    finally:
        cathode_mod.deposit_beam = real_deposit
    result = sim.get_results()
    counters["steps"] = int(np.asarray(result.time).size)
    counters["final_time_s"] = float(np.asarray(result.time)[-1])
    counters["widest_span_cells"] = widest["span"]
    if widest["state"] is not None:
        captured.append(widest["state"])
    return captured, counters


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="destination .pkl")
    parser.add_argument(
        "--t-end",
        type=float,
        default=None,
        help="stop early (default: the fixture's own dynamic t_end)",
    )
    args = parser.parse_args(argv)

    captured, counters = capture(DEFAULT_TARGETS, args.t_end)
    out = Path(args.output)
    with out.open("wb") as handle:
        pickle.dump({"states": captured, "counters": counters}, handle)
    print(f"captured {len(captured)} ray states -> {out}")
    for state in captured:
        print(
            "  %-24s t_target=%.3f ms  t_actual=%.3f ms"
            % (state["label"], state["t_target_s"] * 1e3, state["t_actual_s"] * 1e3)
        )
    print(
        "deposit_beam calls=%d  saved steps=%d  final_time=%.4f ms  "
        "widest ray=%d cells"
        % (
            counters["calls"],
            counters["steps"],
            counters["final_time_s"] * 1e3,
            counters["widest_span_cells"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
