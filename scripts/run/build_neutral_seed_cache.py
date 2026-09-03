"""Build a cached neutral-equilibration seed for a sim1d configuration.

Runs the 100-cycle neutral-only equilibration ONCE for a given config and writes
the equilibrated nn/nn_a profile plus a fail-closed neutral-flow signature to an
.npz. A run that sets the ``use_cached_neutral_seed`` flag then reuses this seed
instead of re-running the equilibration, until the neutral-flow configuration
(puffing / pumping / neutral-flow geometry / neutral-momentum / two-zone /
kinetic mode) changes (a change raises a loud ValueError at load time).

The seed .npz is a derived, regenerable artifact and is gitignored (never
committed), like every other run artifact under scripts/.

Usage (the reference configuration, nx=120)::

    python scripts/run/build_neutral_seed_cache.py --stance g1atrim --nx 120 \
        --db-dir scripts/neutral_seed_db

A SEED NAMES ITS CONFIGURATION. It is an initial condition that later runs
stand on, so a seed equilibrated at an unnamed configuration and fed to named
runs is
the unstanced divergence itself, one layer down and invisible in the runs' own
metadata. ``--stance NAME`` (or a path) names it and the name is written into
the cache file's ``meta``, where ``neutral_seed_configuration`` reads it back
presence-gated; a build that genuinely names none says so with ``--no-stance``.

The config is assembled by the SAME path the run uses -- this file's shared
package with the named configuration over it -- so the signature is guaranteed
to match; only the inert cache-control keys differ.
"""

import argparse
import time as _walltime

import numpy as np

from cablp.solvers._sim1d import LAPDSim1D
from cablp.solvers._sim1d.core.neutral_seed_cache import save_neutral_seed


def es1_config(nx, stance=None):
    """Return the ``(params, flags, lineage)`` the run uses, minus the run.

    ``stance`` is the committed configuration's name or a configuration file's
    path, applied over the shared package exactly where a campaign driver
    applies it -- LAST, so a key the configuration owns wins. ``None`` builds
    the shared package alone and returns no lineage, which is the
    ``--no-stance`` route.
    """
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

    from stance_config import load_named_configuration

    params, flags = default_config()
    params.update(PARAM_OVERRIDES)
    flags.update(FLAG_OVERRIDES)
    params["neutral_exchange_model"] = "knudsen"  # run_model default
    if nx is not None:
        params["nx"] = nx
    lineage = None
    if stance is not None:
        named = load_named_configuration(stance)
        params.update(named.params)
        flags.update(named.flags)
        lineage = named.lineage
    return params, flags, lineage


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--es1", action="store_true",
                    help="assemble the compare_sim1d_es1 shared package "
                         "(the base layer beneath the named configuration)")
    ap.add_argument("--nx", type=int, default=120)
    stance_group = ap.add_mutually_exclusive_group()
    stance_group.add_argument(
        "--stance", metavar="NAME", default=None,
        help="committed configuration file (scripts/stances/NAME.toml), or a "
             "configuration file's path, this seed is equilibrated at")
    stance_group.add_argument(
        "--no-stance", action="store_true",
        help="acknowledge that this seed names no configuration and is built "
             "from this script's defaults plus the overrides on this command "
             "line")
    ap.add_argument("--db-dir", required=True,
                    help="seed-database directory (entry keyed by signature)")
    ap.add_argument("--S-gp", type=float, default=None,
                    help="override S_gp (to pre-populate a fill-rate entry)")
    args = ap.parse_args()
    # A seed names the configuration it was equilibrated at. Later runs stand
    # on this initial condition, and a seed that cannot say what it came from
    # cannot be checked against the run that consumes it.
    if args.stance is None and not args.no_stance:
        raise SystemExit(
            "build_neutral_seed_cache: name the configuration package. Pass "
            "--stance <name> to equilibrate at a committed stance file, or "
            "--no-stance to acknowledge that this seed has none and is built "
            "from this script's defaults plus the overrides on this command "
            "line."
        )

    if not args.es1:
        ap.error("only --es1 config assembly is implemented; add a source as needed")

    import os
    from cablp.solvers._sim1d.core.neutral_seed_cache import (
        seed_db_path, fill_rate_meta,
    )

    params, flags, lineage = es1_config(args.nx, stance=args.stance)
    if args.S_gp is not None:
        params["S_gp"] = args.S_gp
    sim = LAPDSim1D(params, flags)
    p_eff, f_eff = sim.get_config()
    os.makedirs(args.db_dir, exist_ok=True)
    out = seed_db_path(args.db_dir, p_eff, f_eff)
    print(
        f"# building neutral seed: configuration="
        f"{'<unnamed>' if lineage is None else lineage.name} "
        f"nx={args.nx} S_gp={p_eff.get('S_gp')} -> {out}"
    )
    t0 = _walltime.time()
    result = sim.run_neutral_equilibration()
    wall = _walltime.time() - t0

    nn = np.asarray(result.nn[-1], dtype=float)
    saved_nn_a = getattr(result, "nn_a", None)
    nn_a = None if saved_nn_a is None else np.asarray(saved_nn_a[-1], dtype=float)
    meta = fill_rate_meta(p_eff, nn)
    meta["equilibration_wall_s"] = round(wall, 2)
    # WHICH configuration this seed is. The identity is restated over the
    # config the builder actually equilibrated -- the named configuration plus
    # this command line's nx and S_gp -- because that, not the file alone, is
    # what produced the profile.
    if lineage is not None:
        stated = lineage.with_identity(p_eff, f_eff)
        meta["configuration_name"] = stated.name
        meta["configuration_identity"] = stated.identity
    sig = save_neutral_seed(out, nn, nn_a, p_eff, f_eff, meta=meta)
    print(f"# equilibration wall={wall:.1f} s  cells={nn.shape[0]}  "
          f"mean_nn={meta['mean_nn']:.3e}  signature={sig[:16]}...")
    print(f"# wrote {out}")


if __name__ == "__main__":
    main()
