"""Reproduce the subnormal-product finding and exercise the underflow fence.

``cablp.numerics.interp.fma_array`` rebuilds ``math.fma``'s single rounding out
of Dekker's two-product and Knuth's two-sum. Dekker's transform is error-free
only ABSENT UNDERFLOW: once the product ``a*b`` falls into the subnormal range
the split partial products round, the residual ``e`` no longer captures the
whole error, and the reconstruction quietly stops matching ``math.fma``. The
overflow end of that window was already fenced by ``FMA_ARRAY_MAX_ABS``; the
underflow end is fenced by ``FMA_ARRAY_MIN_ABS``.

This script is the evidence for both halves of that claim. Four blocks, all
reported at raw uint64 against ``math.fma``:

A. **The finding.** Targeted triples whose product lands squarely in the
   subnormal range, run through the UNCHECKED kernel (the fence's whole point
   is that the checked entry no longer reaches them). Expected: a large
   fraction differing.
B. **The fence.** The same operands through the checked ``fma_array``.
   Expected: ``ValueError``, quoted verbatim.
C. **Non-binding controls.** The three neighbouring classes the fence
   deliberately does NOT refuse -- a subnormal ADDEND, normal products that
   CANCEL into a subnormal SUM, and exact-zero factors. Expected: 0 differing,
   no raise.
D. **In-domain sweep.** A large random sweep spanning the whole fenced domain
   (exponents from ``FMA_ARRAY_MIN_ABS`` to ``FMA_ARRAY_MAX_ABS``), plus
   adversarial near-cancellation triples. Expected: 0 differing.

Usage::

    python scripts/r3fma_underflow_fence.py
    python scripts/r3fma_underflow_fence.py --sweep 2000000
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from cablp.numerics.interp import (  # noqa: E402
    FMA_ARRAY_MAX_ABS,
    FMA_ARRAY_MIN_ABS,
    _fma_array_unchecked,
    fma_array,
)

#: Dekker's exactness threshold, ``2**(e_min + p - 1)`` for binary64.
DEKKER_FLOOR = 2.0 ** (-1022 + 53 - 1)
MIN_NORMAL = float(np.finfo(np.float64).tiny)

#: Default size of the in-domain sweep (block D). The brief's bar is >= 1e6.
DEFAULT_SWEEP = 1_000_000


def _bits(x):
    return np.ascontiguousarray(x, dtype=float).view(np.uint64)


def _reference(a, b, c):
    """``math.fma`` elementwise -- the answer every block is measured against."""
    return np.array(
        [math.fma(float(x), float(y), float(z))
         for x, y, z in np.broadcast(a, b, c)],
        dtype=float,
    )


def _differing(got, a, b, c):
    got = np.asarray(got, dtype=float).ravel()
    want = _reference(a, b, c)
    return int(np.count_nonzero(_bits(got) != _bits(want))), want.size


def _decades(rng, lo, hi, n):
    """Random float64 with base-10 exponent uniform in ``[lo, hi]``."""
    e = rng.integers(lo, hi + 1, n)
    m = rng.uniform(1.0, 10.0, n)
    s = rng.choice([-1.0, 1.0], n)
    return s * m * np.float64(10.0) ** e


def block_a_finding(rng, n):
    """Targeted subnormal-product triples through the unchecked kernel."""
    # Exponents chosen so the whole battery violates Dekker's precondition --
    # |a*b| below 2**-970 -- while staying above the flush-to-zero floor, which
    # is where the transform actually breaks. Both factors are ordinary
    # normals; only their product is out of domain.
    a = _decades(rng, -160, -150, n)
    b = _decades(rng, -150, -145, n)
    prod = np.abs(a) * np.abs(b)
    print("A. THE FINDING -- product forced into the subnormal range")
    print(f"   |a*b| range           : [{prod.min():.4e}, {prod.max():.4e}]")
    print(f"   all below 2**-970     : {bool(np.all(prod < DEKKER_FLOOR))}")

    # Two addends over the same factors. ``c = -a*b`` is the sharp form of the
    # class -- the answer IS the product's rounding error, so a residual that
    # lost bits is fully exposed; an independently drawn c dilutes it, because
    # most draws are decided by c long before the tail matters. The rate is
    # therefore a property of the probe, not of the defect: what the defect
    # asserts is that it is NOT ZERO.
    worst = None
    for label, c in (("c = -(a*b), residual fully exposed", -(a * b)),
                     ("c drawn independently", _decades(rng, -320, -300, n))):
        diff, total = _differing(_fma_array_unchecked(a, b, c), a, b, c)
        print(f"   {label:34s} : {diff} differing of {total} "
              f"({100.0 * diff / total:.1f}%)")
        if worst is None or diff > worst[3]:
            worst = (a, b, c, diff, total)
    return worst


def block_b_fence(a, b, c):
    """The same operands through the checked entry point."""
    print("B. THE FENCE -- same operands through fma_array()")
    try:
        fma_array(a, b, c)
    except ValueError as exc:
        print("   ValueError raised, verbatim:")
        for line in str(exc).split(". "):
            print(f"     {line.strip()}")
        return True
    print("   NO RAISE -- the fence did not fire; this is a FAILURE")
    return False


def block_c_controls(rng, n):
    """The classes the fence must NOT refuse."""
    print("C. NON-BINDING CONTROLS -- classes the fence leaves alone")
    ok = True

    # (i) subnormal ADDEND, normal factors: c never multiplies.
    a = _decades(rng, -50, 50, n)
    b = _decades(rng, -50, 50, n)
    c = _decades(rng, -320, -300, n)
    diff, total = _differing(fma_array(a, b, c), a, b, c)
    print(f"   (i)   subnormal addend c, normal a,b : {diff} differing of "
          f"{total} (no raise)")
    ok &= diff == 0

    # (ii) normal product CANCELLING into a subnormal sum. Factors sit just
    # above the floor, so p = a*b is a small NORMAL and Dekker's transform is
    # exact; c = -p annihilates it, leaving the exact sum equal to the product's
    # own rounding error, which is subnormal for a workable fraction of draws.
    a = _decades(rng, -145, -142, n)
    b = _decades(rng, -145, -142, n)
    p = a * b
    finite = np.isfinite(p) & (np.abs(p) >= MIN_NORMAL)
    a, b, p = a[finite], b[finite], p[finite]
    c = -p
    exact_tail = np.abs(np.array([math.fma(float(x), float(y), float(z))
                                  for x, y, z in zip(a, b, c)]))
    sub = (exact_tail > 0.0) & (exact_tail < MIN_NORMAL)
    diff, total = _differing(fma_array(a, b, c), a, b, c)
    d_sub, n_sub = _differing(fma_array(a[sub], b[sub], c[sub]),
                              a[sub], b[sub], c[sub])
    print(f"   (ii)  normal product, subnormal SUM  : {diff} differing of "
          f"{total}; of those {n_sub} have a genuinely subnormal a*b+c "
          f"(max {exact_tail[sub].max():.3e} < {MIN_NORMAL:.3e}) and "
          f"{d_sub} differ")
    ok &= diff == 0 and n_sub > 0

    # (iii) exact-zero factors: 0*x is exactly 0, so the floor exempts them.
    a = np.where(rng.random(n) < 0.5, 0.0, _decades(rng, -50, 50, n))
    b = np.where(rng.random(n) < 0.5, 0.0, _decades(rng, -50, 50, n))
    c = _decades(rng, -50, 50, n)
    diff, total = _differing(fma_array(a, b, c), a, b, c)
    print(f"   (iii) exact-zero factors             : {diff} differing of "
          f"{total} (no raise, {int(np.count_nonzero(a == 0.0))} zero a, "
          f"{int(np.count_nonzero(b == 0.0))} zero b)")
    ok &= diff == 0
    return ok


def block_d_sweep(rng, n):
    """Random operands spanning the whole fenced domain."""
    print("D. IN-DOMAIN SWEEP -- the full fenced domain")
    # ``_decades`` draws a mantissa in [1, 10), so the top exponent is one
    # below the ceiling's: the sweep spans the domain without stepping over it.
    lo = int(round(math.log10(FMA_ARRAY_MIN_ABS)))
    hi = int(round(math.log10(FMA_ARRAY_MAX_ABS))) - 1
    ok = True

    # Split the budget: a uniform sweep, then adversarial near-cancellations
    # (c ~ -a*b) where the closing addition's rounding decision is a tie or
    # nearly one -- the case round-to-odd exists for.
    n_uniform = n - n // 4
    n_cancel = n - n_uniform

    a = _decades(rng, lo, hi, n_uniform)
    b = _decades(rng, lo, hi, n_uniform)
    c = _decades(rng, lo, hi, n_uniform)
    # Keep the ceiling honest: the domain's own bound is on the operands, and
    # the product of two fenced operands stays representable by construction.
    diff, total = _differing(fma_array(a, b, c), a, b, c)
    prod = np.abs(a) * np.abs(b)
    print(f"   uniform   : {diff} differing of {total}   "
          f"|a*b| in [{prod[prod > 0].min():.3e}, {prod.max():.3e}]")
    ok &= diff == 0

    a2 = _decades(rng, -60, 60, n_cancel)
    b2 = _decades(rng, -60, 60, n_cancel)
    p = a2 * b2
    # c just off -p, by 0 to a few ULP, so s = p + c cancels almost totally and
    # the tail decides the result.
    nudge = np.float64(2.0) ** rng.integers(-70, -40, n_cancel)
    c2 = -p * (1.0 + nudge * rng.choice([-1.0, 1.0], n_cancel))
    keep = np.isfinite(c2) & (np.abs(c2) <= FMA_ARRAY_MAX_ABS)
    a2, b2, c2 = a2[keep], b2[keep], c2[keep]
    diff, total = _differing(fma_array(a2, b2, c2), a2, b2, c2)
    print(f"   cancelling: {diff} differing of {total}")
    ok &= diff == 0
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", type=int, default=DEFAULT_SWEEP,
                    help="operand triples in the in-domain sweep (block D)")
    ap.add_argument("--probe", type=int, default=300_000,
                    help="operand triples per targeted/control block")
    ap.add_argument("--seed", type=int, default=20260827)
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    print(f"FMA_ARRAY_MIN_ABS = {FMA_ARRAY_MIN_ABS:g}   "
          f"FMA_ARRAY_MAX_ABS = {FMA_ARRAY_MAX_ABS:g}")
    print(f"Dekker exactness threshold 2**-970 = {DEKKER_FLOOR:.6e}; "
          f"MIN_ABS**2 = {FMA_ARRAY_MIN_ABS ** 2:.3e} "
          f"({FMA_ARRAY_MIN_ABS ** 2 / DEKKER_FLOOR:.1f}x above it)")
    print(f"MAX_ABS**2 = {FMA_ARRAY_MAX_ABS ** 2:.3e} "
          f"(finite: {math.isfinite(FMA_ARRAY_MAX_ABS ** 2)})")
    print()

    a, b, c, diff, total = block_a_finding(rng, args.probe)
    finding = diff > 0
    print()
    fenced = block_b_fence(a, b, c)
    print()
    controls = block_c_controls(rng, args.probe)
    print()
    sweep = block_d_sweep(rng, args.sweep)
    print()

    checks = {
        "A finding reproduced (unchecked kernel differs on subnormal products)":
            finding,
        "B fence raises on that class": fenced,
        "C controls unaffected (0 differing, no raise)": controls,
        "D in-domain sweep bit-identical to math.fma": sweep,
    }
    for label, passed in checks.items():
        print(f"  [{'ok' if passed else 'FAIL'}] {label}")
    if all(checks.values()):
        print("FENCE OK -- the underflow domain raises, everything reachable "
              "is untouched and still bit-identical to math.fma.")
        return 0
    print("FENCE FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
