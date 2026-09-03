"""Build a cached neutral-equilibration seed for a sim1d configuration.

Runs the 100-cycle neutral-only equilibration ONCE for a given config and writes
the equilibrated nn/nn_a profile plus a fail-closed neutral-flow signature to an
.npz. A run that sets the ``use_cached_neutral_seed`` flag then reuses this seed
instead of re-running the equilibration, until the neutral-flow configuration
(puffing / pumping / neutral-flow geometry / neutral-momentum / two-zone /
kinetic mode) changes (a change raises a loud ValueError at load time).

The seed .npz is a derived, regenerable artifact and is gitignored (never
committed), like every other run artifact under scripts/.

Usage (ES1 production config, nx=120)::

    python scripts/build_neutral_seed_cache.py --es1 --nx 120 \
        --out scripts/neutral_seed_es1_nx120.npz

The config is assembled by the SAME path the run uses (compare_sim1d_es1) so the
signature is guaranteed to match; only the inert cache-control keys differ.
"""

import argparse
import time as _walltime

import numpy as np

from cablp.solvers._sim1d import LAPDSim1D
from cablp.solvers._sim1d.core.neutral_seed_cache import save_neutral_seed


def es1_config(nx):
    """Return the (params, flags) the ES1 benchmark run uses, minus the run."""
    # scripts/ sibling imports: the seven purpose subdirectories on sys.path.
    import sys as _sys
    from pathlib import Path as _Path
    for _sub in ("atomic", "gates", "kinetic", "run", "score", "stance",
                 "verify"):
        _dir = str(_Path(__file__).resolve().parents[1] / _sub)
        if _dir not in _sys.path:
            _sys.path.insert(0, _dir)

    from compare_sim1d_es1 import PARAM_OVERRIDES, FLAG_OVERRIDES
    from cablp.solvers._sim1d import default_config

    params, flags = default_config()
    params.update(PARAM_OVERRIDES)
    flags.update(FLAG_OVERRIDES)
    params["neutral_exchange_model"] = "knudsen"  # run_model default
    if nx is not None:
        params["nx"] = nx
    return params, flags


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--es1", action="store_true",
                    help="assemble the compare_sim1d_es1 production config")
    ap.add_argument("--nx", type=int, default=120)
    ap.add_argument("--db-dir", required=True,
                    help="seed-database directory (entry keyed by signature)")
    ap.add_argument("--S-gp", type=float, default=None,
                    help="override S_gp (to pre-populate a fill-rate entry)")
    args = ap.parse_args()

    if not args.es1:
        ap.error("only --es1 config assembly is implemented; add a source as needed")

    import os
    from cablp.solvers._sim1d.core.neutral_seed_cache import (
        seed_db_path, fill_rate_meta,
    )

    params, flags = es1_config(args.nx)
    if args.S_gp is not None:
        params["S_gp"] = args.S_gp
    sim = LAPDSim1D(params, flags)
    p_eff, f_eff = sim.get_config()
    os.makedirs(args.db_dir, exist_ok=True)
    out = seed_db_path(args.db_dir, p_eff, f_eff)
    print(f"# building neutral seed: nx={args.nx} S_gp={p_eff.get('S_gp')} -> {out}")
    t0 = _walltime.time()
    result = sim.run_neutral_equilibration()
    wall = _walltime.time() - t0

    nn = np.asarray(result.nn[-1], dtype=float)
    saved_nn_a = getattr(result, "nn_a", None)
    nn_a = None if saved_nn_a is None else np.asarray(saved_nn_a[-1], dtype=float)
    meta = fill_rate_meta(p_eff, nn)
    meta["equilibration_wall_s"] = round(wall, 2)
    sig = save_neutral_seed(out, nn, nn_a, p_eff, f_eff, meta=meta)
    print(f"# equilibration wall={wall:.1f} s  cells={nn.shape[0]}  "
          f"mean_nn={meta['mean_nn']:.3e}  signature={sig[:16]}...")
    print(f"# wrote {out}")


if __name__ == "__main__":
    main()
