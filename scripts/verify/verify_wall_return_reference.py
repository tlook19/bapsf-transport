"""Verify the live wall-return chain against the committed fixture corpus.

The corpus ``scripts/data/wall_return_reference.npz`` (built by
``scripts/build_wall_return_reference.py``) holds raw float64 inputs and
outputs of the ``"diffuse_elastic"`` cylindrical-wall return chain. This script
replays every stored call through the LIVE code and reports the number of
DIFFERING RAW UINT64 values -- the float64 bit patterns compared as integers,
so a one-ulp move is a difference and NaN compares by pattern like anything
else. The bar is zero, on every arm and every leg.

Three legs, one per function in the chain, each covering every arm the corpus
carries (64 x 24, 48 x 12, 96 x 32):

``counts``
    ``TransientDVM._wall_return_counts`` on each stored call, including the
    ``alpha = 1`` degenerate call and the call holding a cell with no landings;
``spectra``
    ``TransientDVM._solve_wall_return_spectra`` on each call's live rows;
``cosine``
    ``_cosine_wall_spectra`` on the stored thermal speeds directly.

A CALL IS THE UNIT. The bisection runs until its slowest cell converges and
every cell keeps bisecting until then, so a cell's answer depends bit-for-bit on
the cells it was solved with; the calls are replayed exactly as stored.

Usage (from the repo root, PYTHONPATH set to the repo root):

    python scripts/verify_wall_return_reference.py --verify
    python scripts/verify_wall_return_reference.py --verify --impl perturbed

``--impl perturbed`` is the NEGATIVE CONTROL: it moves the last returned
cosine-spectrum value by one ulp and nothing else. Every leg must then report a
non-zero differing count, and the script exits 1. A verifier that passes its own
negative control is the only kind worth quoting.

``--tolerance`` is the other question, for a corpus built by a DIFFERENT solve
than the live one: not "are the bits the same" but "is the physics the same to a
stated numerical tolerance". It replays the identical calls and reports, per arm,

* the largest relative mean-energy miss of the LIVE spectra against the
  corpus's own stored targets ``e_bar`` -- the quantity the solve's shipped
  ``WALL_ENERGY_SOLVE_REL_TOL`` bounds -- against a bar of
  ``TOLERANCE_ENERGY_REL``;
* the largest relative difference between the live and stored spectra, and
  between the live and stored placed counts, against a bar of
  ``TOLERANCE_SPECTRUM_REL``; a stored value of exactly zero is required to
  come back exactly zero and is counted separately;
* the ``cosine`` leg at RAW UINT64 still, with a bar of zero:
  :func:`_cosine_wall_spectra` is the spectrum both solves place, so it is
  held bit-exact even where the solved temperature is allowed to move.
"""

import argparse
import sys
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

#: ``--tolerance`` bar on the relative mean-energy miss of the live spectra
#: against the corpus's stored targets. The shipped solve tolerance.
TOLERANCE_ENERGY_REL = 1.0e-12
#: ``--tolerance`` bar on the relative difference between the live and stored
#: spectra and placed counts.
TOLERANCE_SPECTRUM_REL = 1.0e-11


def _differing(got, want):
    """Return how many raw uint64 bit patterns differ between two arrays."""
    got = np.ascontiguousarray(np.asarray(got, dtype=float))
    want = np.ascontiguousarray(np.asarray(want, dtype=float))
    if got.shape != want.shape:
        raise SystemExit(f"shape mismatch: got {got.shape}, reference {want.shape}")
    return int(np.count_nonzero(got.view(np.uint64) != want.view(np.uint64)))


def _perturb_cosine():
    """Install the negative control: one ulp on the last cosine-spectrum value."""
    original = kd._cosine_wall_spectra

    def perturbed(g, s):
        f = original(g, s)
        flat = f.reshape(-1)
        flat[-1] = np.nextafter(flat[-1], np.inf)
        return f

    kd._cosine_wall_spectra = perturbed
    return original


def verify(reference, impl):
    """Replay the corpus and print one line per arm and leg. Return an exit code."""
    data = np.load(reference, allow_pickle=True)
    arms = [str(name) for name in data["arms"]]
    rows_per_call = int(data["rows_per_call"])

    if impl == "perturbed":
        _perturb_cosine()

    # A geometry is needed only to construct a TransientDVM; the three verified
    # methods read the velocity grid and nothing else, so the small default tube
    # stands in for every arm and no stance geometry is baked in.
    geom_params, geom_flags = default_config()
    geom_params["nx"] = 12
    geometry = build_geometry(geom_params, geom_flags)

    total_bad = 0
    total_values = 0
    for arm in arms:
        gargs = data[f"{arm}__grid_args"]
        grid = VGrid(gargs[0], gargs[1], int(gargs[2]), int(gargs[3]), gargs[4])
        bad_grid = _differing(grid.vz, data[f"{arm}__vz"]) + _differing(
            grid.vp, data[f"{arm}__vp"]
        )
        if bad_grid:
            raise SystemExit(
                f"arm {arm}: the rebuilt velocity grid is not the corpus's own "
                f"({bad_grid} differing axis values); the fixture cannot be "
                "compared against a different grid"
            )
        dvm = kd.TransientDVM(
            geometry=geometry, grid=grid, wall_reflection="diffuse_elastic"
        )

        L_wall = data[f"{arm}__L_wall"]
        N_wall = data[f"{arm}__N_wall"]
        alpha = data[f"{arm}__alpha"]
        group = data[f"{arm}__group"]
        e_bar = data[f"{arm}__e_bar"]
        spectra = data[f"{arm}__spectra"]
        out = data[f"{arm}__out"]
        live = data[f"{arm}__live"]

        bad_counts = bad_spectra = 0
        n_counts = n_spectra = 0
        for c in np.unique(group):
            sel = group == c
            got = dvm._wall_return_counts(L_wall[sel], N_wall[sel], float(alpha[c]))
            bad_counts += _differing(got, out[sel])
            n_counts += out[sel].size
            rows = np.flatnonzero(sel & live)
            if rows.size:
                got_s = dvm._solve_wall_return_spectra(e_bar[rows])
                bad_spectra += _differing(got_s, spectra[rows])
                n_spectra += spectra[rows].size

        cw_s = data[f"{arm}__cw_s"]
        cw_f = data[f"{arm}__cw_f"]
        got_f = kd._cosine_wall_spectra(dvm.g, cw_s)
        bad_cosine = _differing(got_f, cw_f)

        n_grid = f"{int(gargs[2])}x{int(gargs[3])}"
        print(
            f"arm {arm:>7} ({n_grid:>5}): "
            f"counts {bad_counts} differing of {n_counts}, "
            f"spectra {bad_spectra} differing of {n_spectra}, "
            f"cosine {bad_cosine} differing of {cw_f.size}"
        )
        total_bad += bad_counts + bad_spectra + bad_cosine
        total_values += n_counts + n_spectra + cw_f.size

    print(
        f"rows {int(data['total_rows'])}, rows per call {rows_per_call}, "
        f"impl {impl}: {total_bad} differing raw uint64 of {total_values}"
    )
    if total_bad:
        print("FAIL: the live wall-return chain does not reproduce the corpus")
        return 1
    print("PASS: 0 differing raw uint64 values")
    return 0


def _relative(got, want):
    """Return ``(max relative difference, exact-zero mismatches, values)``.

    The relative difference is elementwise ``|got - want| / want`` over the
    entries where the stored value is strictly positive. Entries stored as
    exactly zero carry no scale to divide by, so they are held to exact
    equality instead and counted -- a bin that went from empty to occupied is
    a structural change, not a tolerance question, and must not hide inside a
    maximum taken over the other entries.
    """
    got = np.asarray(got, dtype=float)
    want = np.asarray(want, dtype=float)
    if got.shape != want.shape:
        raise SystemExit(f"shape mismatch: got {got.shape}, reference {want.shape}")
    live = want > 0.0
    rel = 0.0
    if np.any(live):
        rel = float(np.max(np.abs(got[live] - want[live]) / want[live]))
    zeros = int(np.count_nonzero(got[~live] != want[~live]))
    return rel, zeros, int(want.size)


def tolerance(reference):
    """Replay the corpus against the LIVE solve at a stated tolerance."""
    data = np.load(reference, allow_pickle=True)
    arms = [str(name) for name in data["arms"]]

    geom_params, geom_flags = default_config()
    geom_params["nx"] = 12
    geometry = build_geometry(geom_params, geom_flags)

    worst_energy = worst_spectrum = 0.0
    bad_zeros = bad_cosine = 0
    for arm in arms:
        gargs = data[f"{arm}__grid_args"]
        grid = VGrid(gargs[0], gargs[1], int(gargs[2]), int(gargs[3]), gargs[4])
        dvm = kd.TransientDVM(
            geometry=geometry, grid=grid, wall_reflection="diffuse_elastic"
        )

        L_wall = data[f"{arm}__L_wall"]
        N_wall = data[f"{arm}__N_wall"]
        alpha = data[f"{arm}__alpha"]
        group = data[f"{arm}__group"]
        e_bar = data[f"{arm}__e_bar"]
        spectra = data[f"{arm}__spectra"]
        out = data[f"{arm}__out"]
        live = data[f"{arm}__live"]

        arm_energy = arm_counts = arm_spectra = 0.0
        arm_zeros = 0
        for c in np.unique(group):
            sel = group == c
            got = dvm._wall_return_counts(L_wall[sel], N_wall[sel], float(alpha[c]))
            rel, zeros, _ = _relative(got, out[sel])
            arm_counts = max(arm_counts, rel)
            arm_zeros += zeros
            rows = np.flatnonzero(sel & live)
            if not rows.size:
                continue
            got_s = dvm._solve_wall_return_spectra(e_bar[rows])
            rel, zeros, _ = _relative(got_s, spectra[rows])
            arm_spectra = max(arm_spectra, rel)
            arm_zeros += zeros
            got_e = (got_s * dvm.E_bin).sum(axis=(1, 2))
            arm_energy = max(
                arm_energy,
                float(np.max(np.abs(got_e - e_bar[rows]) / e_bar[rows])),
            )

        cw_f = data[f"{arm}__cw_f"]
        arm_cosine = _differing(kd._cosine_wall_spectra(dvm.g, data[f"{arm}__cw_s"]),
                                cw_f)

        print(
            f"arm {arm:>7} ({int(gargs[2])}x{int(gargs[3])}): "
            f"energy miss {arm_energy:.3e}, spectra rel {arm_spectra:.3e}, "
            f"counts rel {arm_counts:.3e}, stored-zero mismatches {arm_zeros}, "
            f"cosine {arm_cosine} differing of {cw_f.size}"
        )
        worst_energy = max(worst_energy, arm_energy)
        worst_spectrum = max(worst_spectrum, arm_counts, arm_spectra)
        bad_zeros += arm_zeros
        bad_cosine += arm_cosine

    print(
        f"rows {int(data['total_rows'])}: max relative mean-energy miss "
        f"{worst_energy:.3e} (bar {TOLERANCE_ENERGY_REL:.1e}), max relative "
        f"spectrum/counts difference {worst_spectrum:.3e} (bar "
        f"{TOLERANCE_SPECTRUM_REL:.1e}), stored-zero mismatches {bad_zeros} "
        f"(bar 0), cosine {bad_cosine} differing raw uint64 (bar 0)"
    )
    if (
        worst_energy > TOLERANCE_ENERGY_REL
        or worst_spectrum > TOLERANCE_SPECTRUM_REL
        or bad_zeros
        or bad_cosine
    ):
        print("FAIL: the live wall-return chain is outside the stated tolerance")
        return 1
    print("PASS: the live wall-return chain is inside the stated tolerance")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verify", action="store_true",
                        help="replay the corpus and exit non-zero on any difference")
    parser.add_argument("--tolerance", action="store_true",
                        help="replay the corpus at a stated numerical tolerance "
                             "instead of at raw uint64 (see the module docstring)")
    parser.add_argument("--impl", choices=("live", "perturbed"), default="live",
                        help="'perturbed' is the negative control (see the module "
                             "docstring); it must FAIL")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    args = parser.parse_args(argv)

    if not (args.verify or args.tolerance):
        parser.error("nothing to do without --verify or --tolerance")
    if args.verify and args.tolerance:
        parser.error("--verify and --tolerance ask different questions; pick one")
    if not args.reference.exists():
        raise SystemExit(
            f"the wall-return fixture corpus is missing: {args.reference}. "
            "Build it with scripts/build_wall_return_reference.py"
        )
    if args.tolerance:
        if args.impl != "live":
            parser.error("--impl is the exact mode's negative control")
        return tolerance(args.reference)
    return verify(args.reference, args.impl)


if __name__ == "__main__":
    sys.exit(main())
