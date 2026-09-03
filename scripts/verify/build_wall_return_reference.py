"""Build the wall-return fixture corpus ``scripts/data/wall_return_reference.npz``.

The corpus pins the ``"diffuse_elastic"`` cylindrical-wall return chain --
``TransientDVM._wall_return_counts`` -> ``_solve_wall_return_spectra`` ->
``_cosine_wall_spectra`` -- at raw float64. It is the single source the verifier
``scripts/verify_wall_return_reference.py`` checks an implementation against, so
a transcription of that chain never needs a second implementation of it to be
believed.

WHAT IS IN IT. Three arms, each a velocity grid and a set of CALLS:

``stance``
    the campaign velocity grid (``nvz`` x ``nvp`` = 64 x 24), with landing
    arrays ``L_wall`` taken from a short run of the committed golden
    configuration, cut into small cell groups so the whole corpus fits in a few
    megabytes;
``syn48`` / ``syn96``
    synthetic 48 x 12 and 96 x 32 grids, whose landings are seeded Maxwellian
    projections perturbed deterministically -- they exist so the corpus covers
    grid shapes the campaign does not run.

Every arm carries the degenerate branches the chain must keep exact: one call at
``alpha = 1`` (nothing to place) and one call holding a cell with no landings at
all.

A CALL IS THE UNIT, not a cell. The bisection's iteration count is set by the
slowest cell in the call and every cell keeps bisecting until then, so a cell's
answer depends bit-for-bit on the cells it was solved with. Regrouping the rows
would silently produce a different fixture.

Usage (from the repo root, PYTHONPATH set to the repo root):

    python scripts/build_wall_return_reference.py
    python scripts/build_wall_return_reference.py --steps 600 --out <path>
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cablp.solvers._sim1d import LAPDSim1D, default_config  # noqa: E402
from cablp.solvers._sim1d.core.geometry import build_geometry  # noqa: E402
from cablp.solvers._sim1d.physics import kinetic_dvm as kd  # noqa: E402
from cablp.solvers._sim1d.physics.kinetic_neutrals import VGrid  # noqa: E402

from baseline_sim1d import build_baseline_config  # noqa: E402

DEFAULT_OUT = Path(__file__).resolve().parent / "data" / "wall_return_reference.npz"

# Rows per call, and calls per arm. Sized so the whole corpus stays a few
# megabytes at raw float64 while carrying more than 200 cell rows.
ROWS_PER_CALL = 12
# Thermal speeds sampled per arm for the direct ``_cosine_wall_spectra`` leg,
# each contributing three spectra (the seed and its halved/doubled bracket ends).
COSINE_SEEDS = 16
STANCE_CALLS = 12
SYN48_CALLS = 5
SYN96_CALLS = 3


def _capture_stance_rows(steps):
    """Return ``(grid_args, rows, alpha)`` from a short golden-config run.

    ``rows`` is a list of single-cell ``L_wall`` arrays recorded off the live
    ``_wall_return_counts`` calls; ``grid_args`` are the ``VGrid`` constructor
    arguments the run built its velocity grid with.
    """
    params, flags = build_baseline_config()
    flags["neutral_equilibration"] = False

    grid_args = {}
    orig_grid_init = VGrid.__init__

    def recording_grid_init(self, vmax_z, vmax_p, nvz, nvp, v_fine):
        grid_args.setdefault(
            "args", (float(vmax_z), float(vmax_p), int(nvz), int(nvp), float(v_fine))
        )
        return orig_grid_init(self, vmax_z, vmax_p, nvz, nvp, v_fine)

    seen = []
    orig_counts = kd.TransientDVM._wall_return_counts

    def recording_counts(self, L_wall, N_wall, alpha):
        seen.append((np.array(L_wall, dtype=float), float(alpha)))
        return orig_counts(self, L_wall, N_wall, alpha)

    VGrid.__init__ = recording_grid_init
    kd.TransientDVM._wall_return_counts = recording_counts
    try:
        sim = LAPDSim1D(params, flags)
        try:
            sim.run(t_end=1.0, max_steps=int(steps))
        except RuntimeError as exc:
            if "max_steps" not in str(exc):
                raise
    finally:
        VGrid.__init__ = orig_grid_init
        kd.TransientDVM._wall_return_counts = orig_counts

    if not seen:
        raise SystemExit("the short run reached no diffuse wall return; raise --steps")
    alpha = seen[-1][1]
    rows = []
    for L_wall, _ in seen:
        for cell in range(L_wall.shape[0]):
            row = L_wall[cell]
            if row.sum() > 0.0:
                rows.append(row)
    return grid_args["args"], rows, alpha


def _synthetic_rows(grid, count, seed):
    """Return ``count`` seeded per-cell landing arrays on ``grid``.

    Each row is a drifting-Maxwellian bin-mass projection at a temperature and
    drift drawn from a fixed-seed generator and scaled to a physical count, then
    perturbed multiplicatively so the row is not exactly the projection the
    solve would recover analytically.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(count):
        T_eV = float(rng.uniform(0.02, 0.6))
        u = float(rng.uniform(-3.0e4, 3.0e4))
        base = grid.maxwellian(T_eV, u)
        pert = 1.0 + 0.25 * rng.standard_normal(base.shape)
        row = np.maximum(base * pert, 0.0) * float(rng.uniform(1e6, 1e10))
        if row.sum() <= 0.0:
            row = base.copy()
        rows.append(row)
    return rows


def _build_arm(dvm, rows, n_calls, alpha_default):
    """Assemble one arm's calls and evaluate the reference chain on them.

    The degenerate branches are placed deliberately: the LAST call runs at
    ``alpha = 1`` (nothing to place) and the second-to-last call's first row is
    zeroed (a cell with no landings), so both are covered on every grid.
    """
    need = n_calls * ROWS_PER_CALL
    if len(rows) < need:
        reps = int(np.ceil(need / max(len(rows), 1)))
        rows = (rows * reps)[:need]
    rows = rows[:need]

    L_calls = []
    alphas = []
    for c in range(n_calls):
        block = np.stack(rows[c * ROWS_PER_CALL:(c + 1) * ROWS_PER_CALL]).copy()
        if c == n_calls - 2:
            block[0] = 0.0
        L_calls.append(block)
        alphas.append(1.0 if c == n_calls - 1 else alpha_default)

    L_all, N_all, group, e_bar_all, spectra_all, out_all, live_all = (
        [], [], [], [], [], [], []
    )
    for c, (L, alpha) in enumerate(zip(L_calls, alphas)):
        N = L.sum(axis=(1, 2))
        out = dvm._wall_return_counts(L, N, alpha)
        live = (N > 0.0) & (alpha < 1.0)
        e_bar = np.full(L.shape[0], np.nan)
        spectra = np.zeros_like(L)
        if np.any(live):
            idx = np.flatnonzero(live)
            e_bar[idx] = (L[idx] * dvm.E_bin).sum(axis=(1, 2)) / N[idx]
            spectra[idx] = dvm._solve_wall_return_spectra(e_bar[idx])
        L_all.append(L)
        N_all.append(N)
        group.append(np.full(L.shape[0], c, dtype=np.int32))
        e_bar_all.append(e_bar)
        spectra_all.append(spectra)
        out_all.append(out)
        live_all.append(live)

    # Direct ``_cosine_wall_spectra`` samples, so the innermost function is
    # pinned on its own inputs and not only through the chain above. The
    # thermal speeds are the solve's own seed ``sqrt(e_bar / 2 m)`` at every
    # live row, together with the halved and doubled bracket ends the search
    # actually evaluates. Evenly subsampled to ``COSINE_SEEDS`` rows per arm so
    # the spectra this leg stores stay a small share of the corpus.
    live_e_bar = np.concatenate(e_bar_all)[np.concatenate(live_all)]
    take = np.unique(
        np.linspace(0, live_e_bar.size - 1, COSINE_SEEDS).astype(int)
    )
    seed_s = np.sqrt(live_e_bar[take] / (2.0 * kd.M_HE))
    cw_s = np.concatenate([0.5 * seed_s, seed_s, 2.0 * seed_s])
    cw_f = kd._cosine_wall_spectra(dvm.g, cw_s)

    return {
        "cw_s": cw_s,
        "cw_f": cw_f,
        "L_wall": np.concatenate(L_all),
        "N_wall": np.concatenate(N_all),
        "alpha": np.asarray(alphas, dtype=float),
        "group": np.concatenate(group),
        "e_bar": np.concatenate(e_bar_all),
        "spectra": np.concatenate(spectra_all),
        "out": np.concatenate(out_all),
        "live": np.concatenate(live_all),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=600,
                        help="max solver steps of the short capture run")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    grid_args, stance_rows, alpha = _capture_stance_rows(args.steps)
    vmax_z, vmax_p, nvz, nvp, v_fine = grid_args
    print(f"stance grid: nvz={nvz} nvp={nvp} vmax={vmax_z:.6e} v_fine={v_fine:.6e}")
    print(f"stance rows captured: {len(stance_rows)}  alpha={alpha}")

    # A geometry is needed only to construct a TransientDVM; the three methods
    # this corpus pins read the velocity grid and nothing else, so the small
    # default tube is used for every arm and no stance geometry is baked in.
    geom_params, geom_flags = default_config()
    geom_params["nx"] = 12
    geometry = build_geometry(geom_params, geom_flags)

    specs = [
        ("stance", (vmax_z, vmax_p, nvz, nvp, v_fine), STANCE_CALLS, stance_rows, None),
        ("syn48", (vmax_z, vmax_p, 48, 12, v_fine), SYN48_CALLS, None, 4801),
        ("syn96", (vmax_z, vmax_p, 96, 32, v_fine), SYN96_CALLS, None, 9601),
    ]
    arms = {}
    saved = {}
    for name, gargs, n_calls, rows, seed in specs:
        grid = VGrid(*gargs)
        dvm = kd.TransientDVM(
            geometry=geometry, grid=grid, wall_reflection="diffuse_elastic"
        )
        if rows is None:
            rows = _synthetic_rows(grid, n_calls * ROWS_PER_CALL, seed)
        arm = _build_arm(dvm, rows, n_calls, alpha)
        arms[name] = arm
        saved[f"{name}__grid_args"] = np.asarray(gargs, dtype=float)
        saved[f"{name}__vz"] = grid.vz
        saved[f"{name}__vp"] = grid.vp
        for key, value in arm.items():
            saved[f"{name}__{key}"] = value
        print(f"arm {name}: {arm['L_wall'].shape[0]} rows in {n_calls} calls, "
              f"grid {gargs[2]}x{gargs[3]}, live rows {int(arm['live'].sum())}")

    saved["arms"] = np.asarray(list(arms), dtype=object)
    saved["rows_per_call"] = np.asarray(ROWS_PER_CALL)
    total = sum(a["L_wall"].shape[0] for a in arms.values())
    saved["total_rows"] = np.asarray(total)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **saved)
    size = args.out.stat().st_size
    print(f"wrote {args.out} -- {total} cell rows, {size / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
