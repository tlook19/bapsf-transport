"""D4 spike instrument: compiled vs pure ``_j_eth_crit`` -- equivalence + speed.

Two questions, both empirical:

1. **Equivalence.** Does the Cython transcription reproduce the pure-Python
   kernel bit-for-bit over the operating range? Reported as the ULP
   distribution, not as a tolerance test -- "how far apart are they" is the
   answer the spike needs, and a tolerance would hide it.
2. **Speed.** What does one call cost each way? The cathode solve is ~31 % of
   a production run and is call-overhead bound (2026-07-30 profile), so the
   per-call figure is the one that matters.

The bit-exactness VERDICT for the campaign is the golden baseline run with
``CABLP_COMPILED_KERNELS=1``; this script is the mechanism-level companion
that says *where* any difference comes from.

Requires the opt-in, because it needs both kernels in one process::

    CABLP_COMPILED_KERNELS=1 python scripts/spike_cython_kernels.py \
        --label spike_cython_kernels

Sweep coverage
--------------
``psi`` spans 1e-14 .. 1e4 logarithmically plus a linear fill and the exact
branch boundary 1e-3, so both the Taylor branch (``psi < 1e-3``), the closed
form, the ``psi <= 0`` short circuit and the boundary itself are exercised
densely. ``J_i`` spans 1e-6 .. 1e6 and ``mu`` covers helium (the thesis gas,
``mu = 4``) plus hydrogen and a heavy case. These decades strictly contain the
production arguments rather than being sampled from a production trace.
"""

import argparse
import math
import struct
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent

from cablp.cathode import kernels as _kernels  # noqa: E402
from cablp.cathode.circuit import _j_eth_crit_pure, _pemr  # noqa: E402


def ulp_distance(a, b):
    """Return the number of representable doubles between ``a`` and ``b``.

    Uses the monotone integer ordering of IEEE-754 doubles, so 0 means bit
    identical and 1 means adjacent representable values.
    """
    if a == b:
        return 0
    if math.isnan(a) or math.isnan(b):
        return -1
    ia = struct.unpack("<q", struct.pack("<d", a))[0]
    ib = struct.unpack("<q", struct.pack("<d", b))[0]
    if ia < 0:
        ia = -0x8000000000000000 - ia
    if ib < 0:
        ib = -0x8000000000000000 - ib
    return abs(ia - ib)


def sweep_points():
    """Return the (psi, J_i, mu) grid, broad and branch-aware."""
    psi = np.concatenate(
        [
            np.array([-1.0, -1e-9, 0.0]),
            np.logspace(-14.0, 4.0, 1801),
            np.linspace(1e-6, 5.0e-3, 401),   # dense across the Taylor branch
            np.linspace(5.0e-3, 200.0, 801),  # dense across the closed form
            np.array([1e-3, np.nextafter(1e-3, 0.0), np.nextafter(1e-3, 1.0)]),
        ]
    )
    J_i = np.logspace(-6.0, 6.0, 13)
    mu = np.array([1.0, 4.0, 20.0, 40.0])  # He (thesis gas) is mu = 4
    return psi, J_i, mu


def run_equivalence(compiled, stream):
    psi_grid, J_grid, mu_grid = sweep_points()
    n = 0
    exact = 0
    worst = (0, None)
    hist = {}
    branch_counts = {"zero": 0, "taylor": 0, "closed": 0}
    for mu in mu_grid:
        for J_i in J_grid:
            for psi in psi_grid:
                a = _j_eth_crit_pure(float(psi), float(J_i), float(mu))
                b = compiled.j_eth_crit(float(psi), float(J_i), float(mu))
                d = ulp_distance(a, b)
                n += 1
                if d == 0:
                    exact += 1
                hist[d] = hist.get(d, 0) + 1
                if d > worst[0]:
                    worst = (d, (float(psi), float(J_i), float(mu), a, b))
                if psi <= 0.0:
                    branch_counts["zero"] += 1
                elif psi < 1e-3:
                    branch_counts["taylor"] += 1
                else:
                    branch_counts["closed"] += 1

    print("EQUIVALENCE SWEEP (compiled vs pure)", file=stream)
    print("-" * 72, file=stream)
    print(f"points compared      : {n}", file=stream)
    print(
        f"  branch psi<=0      : {branch_counts['zero']}\n"
        f"  branch psi<1e-3    : {branch_counts['taylor']}\n"
        f"  branch closed form : {branch_counts['closed']}",
        file=stream,
    )
    print(f"bit-identical        : {exact}  ({100.0 * exact / n:.6f} %)", file=stream)
    print(f"max ULP distance     : {worst[0]}", file=stream)
    print("ULP histogram        : " + ", ".join(
        f"{k}:{v}" for k, v in sorted(hist.items())
    ), file=stream)
    if worst[1] is not None:
        psi, J_i, mu, a, b = worst[1]
        rel = abs(a - b) / abs(a) if a != 0.0 else float("nan")
        print(
            f"worst point          : psi={psi!r} J_i={J_i!r} mu={mu!r}\n"
            f"  pure     = {a!r}\n"
            f"  compiled = {b!r}\n"
            f"  rel diff = {rel:.3e}",
            file=stream,
        )
    return worst[0], exact, n


def run_benchmark(compiled, stream, repeats, inner):
    """Microbenchmark both kernels on the closed-form and Taylor branches."""
    print("\nMICROBENCHMARK (_j_eth_crit)", file=stream)
    print("-" * 72, file=stream)
    print(
        f"{'branch':>10}  {'pure calls/s':>14}  {'cy calls/s':>14}  "
        f"{'pure ns':>9}  {'cy ns':>9}  {'speedup':>8}",
        file=stream,
    )
    results = {}
    for name, psi in (("closed", 12.5), ("taylor", 5.0e-4)):
        timings = {}
        for label, fn in (("pure", _j_eth_crit_pure), ("cy", compiled.j_eth_crit)):
            best = float("inf")
            for _ in range(repeats):
                t0 = perf_counter()
                for _ in range(inner):
                    fn(psi, 1.5, 4.0)
                best = min(best, perf_counter() - t0)
            timings[label] = best / inner
        speedup = timings["pure"] / timings["cy"]
        results[name] = (1.0 / timings["pure"], 1.0 / timings["cy"], speedup)
        print(
            f"{name:>10}  {1.0 / timings['pure']:14,.0f}  "
            f"{1.0 / timings['cy']:14,.0f}  "
            f"{timings['pure'] * 1e9:9.1f}  {timings['cy'] * 1e9:9.1f}  "
            f"{speedup:8.2f}x",
            file=stream,
        )
    print(
        "\nLoop overhead is included in both columns identically, so the "
        "speedup ratio is meaningful even though the absolute ns are not a "
        "pure function cost.",
        file=stream,
    )
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="spike_cython_kernels")
    parser.add_argument("--out-dir", default=str(SCRIPT_DIR))
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--inner", type=int, default=200000)
    args = parser.parse_args(argv)

    compiled = _kernels.COMPILED_KERNELS
    if compiled is None:
        raise SystemExit(
            f"this spike needs both kernels in one process; set "
            f"{_kernels.ENV_VAR}=1 (and build the extension with "
            "`python build_ext.py --inplace`)"
        )

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.label}.txt"

    import io

    buf = io.StringIO()
    print("=" * 72, file=buf)
    print("D4 CYTHON KERNEL SPIKE -- _j_eth_crit", file=buf)
    print("=" * 72, file=buf)
    print(f"provenance      : {_kernels.PROVENANCE}", file=buf)
    print(f"compiled module : {compiled.__file__}", file=buf)
    print(f"python          : {sys.version.split()[0]}", file=buf)
    print(f"pemr python     : {_pemr!r}", file=buf)
    print(f"pemr compiled   : {compiled.pemr()!r}", file=buf)
    print(f"pemr identical  : {compiled.pemr() == _pemr}", file=buf)
    print("", file=buf)

    worst_ulp, exact, n = run_equivalence(compiled, buf)
    run_benchmark(compiled, buf, args.repeats, args.inner)

    print("\nVERDICT", file=buf)
    print("-" * 72, file=buf)
    if worst_ulp == 0:
        print(
            f"BIT-EXACT over the swept range: all {n} points identical.",
            file=buf,
        )
    else:
        print(
            f"NOT bit-exact: {n - exact} of {n} points differ, worst "
            f"{worst_ulp} ULP.",
            file=buf,
        )
    text = buf.getvalue()
    path.write_text(text)
    print(text)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
