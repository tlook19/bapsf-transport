"""G3/G4 gate run for the floor-aware surface_loss drain bound.

Replicates the f=0.1 production stack EXACTLY by loading the reference run's
own resolved params_json/flags_json (es1_r5_f01_ag26ms.h5), turns the fix ON
(surface_loss_floor_exempt=True), and runs to an explicit t_end (default
26 ms) so the afterglow segment the reference could not reach is covered.
The reference file is read-only input; the output is a new artifact.

Usage:
    python scripts/run_floorfix_g3g4.py --ref <reference.h5> \
        --out scripts/es1_r5_f01_floorfix_26ms.h5 [--t-end 26e-3]
"""

import argparse
import json
import time as _walltime

import h5py
import numpy as np

from cablp.solvers._sim1d import LAPDSim1D
from cablp.solvers._sim1d.results.io import save_result_hdf5


class WallTracker:
    """Collect (wall_elapsed_s, sim_time, step) samples from run progress."""

    def __init__(self):
        self.samples = []

    def update(self, progress):
        self.samples.append(
            (progress.wall_elapsed_s, progress.time, progress.step)
        )

    def reset(self):
        pass


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", required=True,
                    help="reference h5 whose params/flags are replicated")
    ap.add_argument("--out", required=True)
    ap.add_argument("--t-end", type=float, default=26e-3)
    args = ap.parse_args(argv)

    with h5py.File(args.ref, "r") as h5:
        params = json.loads(h5.attrs["params_json"])
        flags = json.loads(h5.attrs["flags_json"])
        ref_pre = float(h5.attrs["t_prebreakdown_trigger"])
        ref_brk = float(h5.attrs["t_breakdown_trigger"])
    flags["surface_loss_floor_exempt"] = True

    sim = LAPDSim1D(params, flags)
    tracker = WallTracker()
    t0 = _walltime.perf_counter()
    sim.start_simulation(t_end=args.t_end, progress_tracker=tracker)
    wall_total = _walltime.perf_counter() - t0
    result = sim.get_results()

    print(f"# wall_total={wall_total:.1f} s  final_time={result.final_time*1e3:.4f} ms  "
          f"steps={result.steps}")
    print(f"# t_prebreakdown_trigger={result.t_prebreakdown_trigger*1e6:.3f} us "
          f"(ref {ref_pre*1e6:.3f} us, delta "
          f"{(result.t_prebreakdown_trigger-ref_pre)*1e6:.4f} us)")
    print(f"# t_breakdown_trigger={result.t_breakdown_trigger*1e6:.3f} us "
          f"(ref {ref_brk*1e6:.3f} us, delta "
          f"{(result.t_breakdown_trigger-ref_brk)*1e6:.4f} us)")

    # Afterglow wall-rate from the main-run progress samples (the main run's
    # wall clock restarts at 0; equilibration samples carry sim times of
    # seconds, the main run tops out at t_end).
    samples = np.asarray(tracker.samples, dtype=float)
    if samples.size:
        main_mask = samples[:, 1] <= args.t_end + 1e-9
        # keep only the trailing contiguous main-run block
        idx = np.flatnonzero(~main_mask)
        start = idx[-1] + 1 if idx.size else 0
        main_samples = samples[start:]
        drive_end = ref_brk + float(params["tau_discharge"])
        ag = main_samples[main_samples[:, 1] >= drive_end]
        if len(ag) >= 2:
            dwall = ag[-1, 0] - ag[0, 0]
            dsim_ms = (ag[-1, 1] - ag[0, 1]) * 1e3
            print(f"# afterglow segment: {ag[0,1]*1e3:.3f}-{ag[-1,1]*1e3:.3f} ms, "
                  f"wall {dwall:.1f} s, rate {dwall/max(dsim_ms,1e-12):.1f} s/ms")
        run_wall = main_samples[-1, 0] if len(main_samples) else float("nan")
        print(f"# main-run wall={run_wall:.1f} s (equilibration+overhead "
              f"{wall_total-run_wall:.1f} s)")

    for name, total in sorted(result.floor_ledger.items()):
        print(f"# floor_ledger[{name}] = {total:.6e}")

    save_result_hdf5(args.out, result)
    print(f"# saved {args.out}")


if __name__ == "__main__":
    main()
