"""Capture/verify the FUSED ``np.interp`` reference vector.

Why this exists
---------------
``_cathode_kernels_cy._interp_scalar`` transcribes numpy's ``arr_interp`` and
reproduces its lerp with an explicit ``fma()``, because numpy's C was compiled
WITH FP contraction on the platform the transcription was written against
(macOS/arm64). That is a property of numpy's BUILD, not of numpy's source:
contraction needs a fused-multiply-add instruction in the compilation target,
and it is present in the aarch64 base ISA but absent from the ``x86_64_v2``
baseline that numpy's linux wheels are built for. On 2026-08-16 the campaign
moved to linux-64 and the assumption silently inverted -- the explicit ``fma``
began to MISMATCH numpy rather than match it, and the compiled kernels stopped
being bit-exact against the pure path for any configuration that interpolates.

The golden did not catch it: its configuration never reaches ``arr_interp``.
**A bit-exactness gate only certifies the code it exercises.**

This fixture pins the FUSED answers themselves, so "does this implementation
reproduce contracted numpy?" stops being a question about which machine you
are standing on. Captured on macOS/arm64 while that platform was still
available; from here on it is just data, checkable anywhere, forever.

Usage::

    # on a CONTRACTED platform (macOS/arm64), once:
    python scripts/interp_fused_reference.py --capture

    # anywhere, any time -- the package's own fused helper:
    python scripts/interp_fused_reference.py --verify

    # the platform probe: is THIS numpy's arr_interp contracted?
    python scripts/interp_fused_reference.py --verify --impl numpy

``--verify`` compares at raw uint64, never with a tolerance: a tolerance would
defeat the entire point.

``--impl helper`` (the default) checks
``cablp.funcs._interp.interp_scalar_fused``, which writes the fusion out with
``math.fma`` and must therefore reproduce the fixture on EVERY platform. A
failure there is a bug in the helper.

``--impl numpy`` checks ambient ``np.interp`` and so measures the platform, not
the package: it passes on a contracted build and is EXPECTED to fail on
linux-64. That failure IS the finding, and is what the helper exists to make
irrelevant rather than something to tolerate.

Coverage follows the method recorded in ``_interp_scalar``'s own docstring
(1,235,520 queries): every exact node, both ``nextafter`` neighbours of every
node, midpoints, out-of-range on both sides, and NaN -- over the REAL tables
this package interpolates, plus synthetic tables for the degenerate shapes
(``n == 1``, ``n == 2``) and for wide dynamic range.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_FIXTURE = SCRIPT_DIR / "data" / "interp_fused_reference.npz"

# Distinct sentinels, so the left/right branches are OBSERVABLE in the output
# rather than aliasing onto fp[0]/fp[-1] (numpy's defaults).
LVAL = -1.2345678901234567e5
RVAL = 9.8765432109876543e4


def _real_tables():
    """The (name, xp, fp) triples this package actually interpolates."""
    from cablp.funcs import _cross as C

    names = [
        ("H_beam_ionization", "_H_LOG_E", "_H_LOG_SIGMA"),
        ("He_beam_ionization", "_HE_LOG_EPS", "_HE_LOG_SIGMA"),
        ("He_ionization_rate", "_HE_ION_LOG_T", "_HE_ION_LOG_RATE"),
        ("phelps_backscatter", "_phelps_Teff", "_phelps_kb"),
        ("phelps_isotropic", "_phelps_Teff", "_phelps_kiso"),
        ("He_en_momentum_transfer", "_HE_EN_MT_LOG_E", "_HE_EN_MT_LOG_SIGMA"),
    ]
    out = []
    for label, xp_name, fp_name in names:
        xp = getattr(C, xp_name, None)
        fp = getattr(C, fp_name, None)
        if xp is None or fp is None:
            print(f"  ! skipping {label}: {xp_name}/{fp_name} not found")
            continue
        xp = np.ascontiguousarray(xp, dtype=np.float64)
        fp = np.ascontiguousarray(fp, dtype=np.float64)
        if xp.ndim != 1 or xp.shape != fp.shape or xp.size < 2:
            print(f"  ! skipping {label}: unusable shape {xp.shape}/{fp.shape}")
            continue
        out.append((label, xp, fp))
    return out


def _synthetic_tables():
    """Degenerate shapes and wide dynamic range the real tables do not cover."""
    tables = []
    # n == 1: numpy uses a separate loop; _interp_scalar claims the general
    # path reduces to the same three-way answer. Pin that claim.
    tables.append(("synth_n1", np.array([2.5]), np.array([-7.25])))
    tables.append(("synth_n2", np.array([0.0, 1.0]), np.array([-1.0, 3.0])))
    # Non-uniform spacing with an extreme slope, where a single vs double
    # rounding is most likely to separate.
    xp = np.array([0.0, 1e-8, 1e-4, 1.0, 3.0, 1e3, 1e7])
    fp = np.array([-1e9, 5e-7, 2.0, -3.5, 1e12, -4e-11, 7e8])
    tables.append(("synth_wide_range", xp, fp))
    # Equal consecutive fp values, which is the only route into the
    # fp[j] == fp[j+1] NaN-ladder branch.
    tables.append(
        ("synth_flat_segments",
         np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
         np.array([1.5, 1.5, -2.0, -2.0, -2.0]))
    )
    return tables


def _queries(xp):
    """Exact nodes, both nextafter neighbours, midpoints, out-of-range, NaN."""
    q = [xp]
    q.append(np.nextafter(xp, -np.inf))
    q.append(np.nextafter(xp, np.inf))
    if xp.size >= 2:
        q.append(0.5 * (xp[:-1] + xp[1:]))
        # A few interior points that are not midpoints, to sample generic
        # fractional positions within a cell.
        for frac in (0.1, 0.25, 1.0 / 3.0, 0.7, 0.99):
            q.append(xp[:-1] + frac * (xp[1:] - xp[:-1]))
    span = float(xp[-1] - xp[0]) if xp.size >= 2 else 1.0
    span = span if span > 0 else 1.0
    q.append(np.array([
        xp[0] - span, xp[0] - 1e-9 * span, np.nextafter(xp[0], -np.inf),
        xp[-1] + span, xp[-1] + 1e-9 * span, np.nextafter(xp[-1], np.inf),
        np.nan,
    ]))
    return np.concatenate([np.atleast_1d(a).astype(np.float64) for a in q])


def _pack(tables):
    """Build the fixture arrays: for each table, queries and BOTH result sets."""
    data, manifest, total = {}, [], 0
    for label, xp, fp in tables:
        x = _queries(xp)
        r_def = np.interp(x, xp, fp)                       # numpy's defaults
        r_sen = np.interp(x, xp, fp, left=LVAL, right=RVAL)  # observable branches
        data[f"{label}__xp"] = xp
        data[f"{label}__fp"] = fp
        data[f"{label}__x"] = x
        data[f"{label}__r_default"] = r_def
        data[f"{label}__r_sentinel"] = r_sen
        manifest.append(label)
        total += 2 * x.size
    data["__manifest__"] = np.array(manifest)
    data["__lval__"] = np.array([LVAL])
    data["__rval__"] = np.array([RVAL])
    return data, total


def _contracted_here():
    """Is THIS numpy's arr_interp fused? Measured, not inferred from platform."""
    n, m = 2048, 100000
    xp = np.arange(n, dtype=np.float64) * (100.0 / (n - 1))
    fp = ((np.arange(n) * 7919 % 10007).astype(np.float64) / 10007.0) * 2e3 - 1e3
    x = (np.arange(m, dtype=np.float64) * 99.9) / (m - 1)
    j = np.clip(np.searchsorted(xp, x) - 1, 0, n - 2)
    slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j])
    unfused = slope * (x - xp[j]) + fp[j]
    return not bool((np.interp(x, xp, fp).view(np.uint64)
                     == unfused.view(np.uint64)).all())


def _evaluator(impl):
    """Return ``f(x, xp, fp, **kw) -> array`` for the named implementation.

    ``numpy`` is ambient ``np.interp`` and therefore probes the platform's
    contraction. ``helper`` is the package's explicit-``fma`` scalar helper,
    mapped over the query vector one element at a time -- it is a SCALAR
    function, and evaluating it elementwise is the only faithful way to ask it
    the fixture's questions.
    """
    if impl == "numpy":
        return lambda x, xp, fp, **kw: np.interp(x, xp, fp, **kw)

    from cablp.funcs._interp import interp_scalar_fused

    def _helper(x, xp, fp, **kw):
        return np.array(
            [interp_scalar_fused(v, xp, fp, **kw) for v in np.asarray(x).ravel()],
            dtype=np.float64,
        )

    return _helper


def _bitdiff(a, b):
    """Count differing float64 values, treating NaN payloads as equal-if-both-NaN."""
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    both_nan = np.isnan(a) & np.isnan(b)
    return int((~both_nan & (a.view(np.uint64) != b.view(np.uint64))).sum())


def capture(path):
    fused = _contracted_here()
    print(f"numpy {np.__version__} on {sys.platform}; arr_interp contracted: {fused}")
    if not fused:
        print("REFUSING to capture: this numpy is NOT contracted, so its answers "
              "are not the fused reference. Capture on a contracted platform "
              "(aarch64 numpy, e.g. macOS/arm64).")
        return 1
    tables = _real_tables() + _synthetic_tables()
    data, total = _pack(tables)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    print(f"captured {len(tables)} tables, {total} pinned values -> {path}")
    for label, xp, _ in tables:
        print(f"    {label:26s} n={xp.size}")
    return 0


def verify(path, impl="helper", quiet=False):
    if not path.exists():
        print(f"FAIL: fixture not found: {path}")
        return 1
    z = np.load(path, allow_pickle=False)
    manifest = [str(s) for s in z["__manifest__"]]
    if not manifest:
        print("FAIL: fixture manifest is EMPTY — refusing to report a pass")
        return 1
    evaluate = _evaluator(impl)
    lval, rval = float(z["__lval__"][0]), float(z["__rval__"][0])
    total = bad = checked = 0
    for label in manifest:
        xp, fp, x = z[f"{label}__xp"], z[f"{label}__fp"], z[f"{label}__x"]
        for kind, kw in (("default", {}), ("sentinel", dict(left=lval, right=rval))):
            want = z[f"{label}__r_{kind}"]
            got = evaluate(x, xp, fp, **kw)
            if np.asarray(got).size != want.size:
                print(f"FAIL: {impl} returned {np.asarray(got).size} values for "
                      f"{label} [{kind}], expected {want.size}")
                return 1
            d = _bitdiff(got, want)
            total += want.size
            checked += 1
            bad += d
            if d and not quiet:
                print(f"    MISMATCH {label:26s} [{kind:8s}] {d:6d} / {want.size}")
    if checked == 0 or total == 0:
        print("FAIL: nothing was compared — refusing to report a pass")
        return 1
    print(f"impl={impl}; numpy {np.__version__} on {sys.platform}; "
          f"arr_interp contracted: {_contracted_here()}")
    print(f"compared {total} values across {len(manifest)} tables: "
          f"{bad} differing")
    if bad:
        print(f"VERIFY FAILED — {impl} does not reproduce the fused reference.")
        if impl == "numpy":
            print("  Expected on linux-64 (x86_64_v2 baseline has no FMA). An "
                  "implementation claiming to reproduce contracted numpy must "
                  "make this pass; a tolerance would defeat the purpose.")
        else:
            print("  The helper writes the fusion out with math.fma and must "
                  "pass on EVERY platform, so this is a bug in the helper, not "
                  "a property of this machine.")
        return 1
    print(f"VERIFY OK — {impl} is bit-identical to the fused reference.")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--capture", action="store_true",
                   help="write the fixture (contracted platforms only)")
    g.add_argument("--verify", action="store_true",
                   help="check an implementation against the fixture, at raw "
                        "uint64")
    p.add_argument("--impl", choices=("numpy", "helper"), default="helper",
                   help="what to check: the package's fused helper (default), "
                        "or ambient np.interp as a platform probe")
    p.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args(argv)
    return capture(a.fixture) if a.capture else verify(a.fixture, a.impl, a.quiet)


if __name__ == "__main__":
    raise SystemExit(main())
