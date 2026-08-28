"""Measure the REACHABLE operand domain of the lane march's fused multiply-add.

``cablp.numerics.interp._fma_array_unchecked`` is the one site where Dekker's
two-product runs in production. Its error-freeness has a lower precondition as
well as the upper one ``FMA_ARRAY_MAX_ABS`` already fences: the transform is
exact only while the product ``a*b`` stays clear of the subnormal range. This
script measures how far the reachable corpus actually sits from that window, so
the fence can be placed with a documented margin rather than a guess.

It intercepts every ``_fma_array_unchecked`` call made while replaying

* the committed ``deposit_beam`` reference corpus
  (``scripts/deposit_beam_reference.py --verify``), and
* the randomized lane batteries of ``scripts/r3lane_equivalence.py``,

and reports, per operand and for the product, the extreme magnitudes seen.
Zeros are excluded from the minima: ``0*x`` is exactly ``0`` and cannot
underflow, so a zero operand is outside the precondition's scope.

Usage::

    python scripts/r3fma_domain_probe.py --random 2000
"""

import argparse
import runpy
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import cablp.numerics.interp as I  # noqa: E402

#: IEEE-754 float64 smallest positive normal, and Dekker's own threshold
#: ``2**(e_min + p - 1)`` below which the two-product residual stops being
#: exact (Muller et al., Handbook of Floating-Point Arithmetic, 2nd ed. 4.4.1).
MIN_NORMAL = float(np.finfo(np.float64).tiny)
DEKKER_FLOOR = 2.0 ** (-1022 + 53 - 1)


class Census:
    """Running extremes over every operand triple the corpus produces."""

    def __init__(self):
        self.calls = 0
        self.elements = 0
        self.min_abs = {"a": np.inf, "b": np.inf, "c": np.inf, "a*b": np.inf}
        self.max_abs = {"a": 0.0, "b": 0.0, "c": 0.0, "a*b": 0.0}
        self.zeros = {"a": 0, "b": 0, "c": 0}

    def record(self, a, b, c):
        a, b, c = (np.atleast_1d(np.asarray(v, dtype=float))
                   for v in (a, b, c))
        self.calls += 1
        self.elements += int(np.broadcast(a, b, c).size)
        prod = np.abs(np.broadcast_arrays(a, b)[0]
                      * np.broadcast_arrays(a, b)[1])
        for name, arr in (("a", a), ("b", b), ("c", c), ("a*b", prod)):
            mag = np.abs(arr)
            if name != "a*b":
                self.zeros[name] += int(np.count_nonzero(mag == 0.0))
            nz = mag[mag > 0.0]
            if nz.size:
                self.min_abs[name] = min(self.min_abs[name], float(nz.min()))
            if mag.size:
                self.max_abs[name] = max(self.max_abs[name], float(mag.max()))

    def report(self):
        lines = [
            f"fma calls intercepted : {self.calls}",
            f"operand elements      : {self.elements}",
        ]
        for name in ("a", "b", "c", "a*b"):
            lo = self.min_abs[name]
            hi = self.max_abs[name]
            zline = ("" if name == "a*b"
                     else f"   exact zeros {self.zeros[name]}")
            lines.append(
                f"|{name}| nonzero range : [{lo:.6e}, {hi:.6e}]{zline}"
            )
        prod_lo = self.min_abs["a*b"]
        lines.append(
            f"min |a*b| / Dekker floor 2**-970 = {DEKKER_FLOOR:.6e} : "
            f"{prod_lo / DEKKER_FLOOR:.3e}x  "
            f"({np.log10(prod_lo / DEKKER_FLOOR):.1f} decades of margin)"
        )
        lines.append(
            f"min |a*b| / smallest normal {MIN_NORMAL:.6e} : "
            f"{np.log10(prod_lo / MIN_NORMAL):.1f} decades of margin"
        )
        return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--random", type=int, default=2000,
                    help="randomized lane batteries to include (0 to skip)")
    args = ap.parse_args(argv)

    census = Census()
    real = I._fma_array_unchecked

    def spy(a, b, c):
        census.record(a, b, c)
        return real(a, b, c)

    I._fma_array_unchecked = spy
    # The lane march imported the private helper by name at module load.
    import cablp.numerics.interp  # noqa: F401
    try:
        print("[corpus] deposit_beam_reference.py --verify")
        sys.argv = ["deposit_beam_reference.py", "--verify"]
        try:
            runpy.run_path(str(SCRIPT_DIR / "deposit_beam_reference.py"),
                           run_name="__main__")
        except SystemExit as exc:
            print(f"[corpus] exit={exc.code}")
        print(f"[corpus] running total: {census.calls} fma calls")

        print("[lanes] r3lane_equivalence.py --corpus"
              + (f" --random {args.random}" if args.random else ""))
        sys.argv = ["r3lane_equivalence.py", "--corpus"]
        if args.random:
            sys.argv += ["--random", str(args.random)]
        try:
            runpy.run_path(str(SCRIPT_DIR / "r3lane_equivalence.py"),
                           run_name="__main__")
        except SystemExit as exc:
            print(f"[lanes] exit={exc.code}")
    finally:
        I._fma_array_unchecked = real

    print()
    print("=== reachable fma operand domain ===")
    print(census.report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
