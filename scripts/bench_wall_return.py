"""Time the ``"diffuse_elastic"`` wall return over the fixture corpus.

The measured quantity is the MIN WALL TIME PER CALL of
``TransientDVM._wall_return_counts`` on the stance velocity grid (64 x 24),
replaying the calls stored in ``scripts/data/wall_return_reference.npz``. The
minimum over repeats is reported rather than a mean because a minimum is the
statistic a contended machine cannot inflate.

One process, no threads, and the corpus is loaded once before timing starts, so
what is timed is the chain and nothing around it. Every arm is reported; the
stance arm is the registered number.

Usage (from the repo root, PYTHONPATH set to the repo root):

    python scripts/bench_wall_return.py
    python scripts/bench_wall_return.py --repeats 9 --label post
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cablp.solvers._sim1d import default_config  # noqa: E402
from cablp.solvers._sim1d.core.geometry import build_geometry  # noqa: E402
from cablp.solvers._sim1d.physics import kinetic_dvm as kd  # noqa: E402
from cablp.solvers._sim1d.physics.kinetic_neutrals import VGrid  # noqa: E402

DEFAULT_REFERENCE = (
    Path(__file__).resolve().parent / "data" / "wall_return_reference.npz"
)


def _arm_calls(data, arm, geometry):
    """Return ``(dvm, calls)`` for one corpus arm; ``calls`` are call arguments."""
    gargs = data[f"{arm}__grid_args"]
    grid = VGrid(gargs[0], gargs[1], int(gargs[2]), int(gargs[3]), gargs[4])
    dvm = kd.TransientDVM(
        geometry=geometry, grid=grid, wall_reflection="diffuse_elastic"
    )
    L_wall = data[f"{arm}__L_wall"]
    N_wall = data[f"{arm}__N_wall"]
    alpha = data[f"{arm}__alpha"]
    group = data[f"{arm}__group"]
    calls = []
    for c in np.unique(group):
        sel = group == c
        if float(alpha[c]) >= 1.0:
            # The degenerate call places nothing and never enters the solve; it
            # is in the corpus for exactness, not for timing.
            continue
        calls.append((L_wall[sel].copy(), N_wall[sel].copy(), float(alpha[c])))
    return dvm, calls, f"{int(gargs[2])}x{int(gargs[3])}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repeats", type=int, default=5,
                        help="timed repeats over the whole corpus (min is reported)")
    parser.add_argument("--label", default="",
                        help="free-text label printed with the result")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    args = parser.parse_args(argv)

    if args.repeats < 5:
        raise SystemExit("at least 5 repeats: a single pass is not a measurement")
    if not args.reference.exists():
        raise SystemExit(
            f"the wall-return fixture corpus is missing: {args.reference}. "
            "Build it with scripts/build_wall_return_reference.py"
        )

    data = np.load(args.reference, allow_pickle=True)
    geom_params, geom_flags = default_config()
    geom_params["nx"] = 12
    geometry = build_geometry(geom_params, geom_flags)

    print(f"bench_wall_return {args.label or '(no label)'} "
          f"repeats={args.repeats} numpy={np.__version__}")
    for arm in [str(name) for name in data["arms"]]:
        dvm, calls, shape = _arm_calls(data, arm, geometry)
        for L, N, alpha in calls:          # warm-up, untimed
            dvm._wall_return_counts(L, N, alpha)
        best = float("inf")
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            for L, N, alpha in calls:
                dvm._wall_return_counts(L, N, alpha)
            best = min(best, time.perf_counter() - t0)
        per_call = best / len(calls)
        print(
            f"arm {arm:>7} ({shape:>5}): {len(calls)} calls x "
            f"{L.shape[0]} rows -- min wall {best * 1e3:.3f} ms, "
            f"per call {per_call * 1e3:.4f} ms"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
