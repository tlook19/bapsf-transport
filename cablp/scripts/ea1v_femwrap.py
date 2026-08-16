"""ea1v: f_em CAPTURE WRAPPER -- INSTRUMENTATION ONLY, harness UNMODIFIED.

The named ea1 stage-1 harnesses (scripts/efold1_traj.py, scripts/pd0_endvent_traj.py)
record no emitting-area column: the build exposes f_em as the run-result
diagnostic ``cathode_diagnostics["cathode_emitting_area_fraction"]``
(solver.py:9820, from ``self._cathode_f_em``), and both harnesses discard the
per-step run result by design. The ea1 registration's ANTI-VACUITY condition
("B1/B2 numbers quote only from arms whose f_em actually advanced; no frozen
f_em") therefore needs f_em MEASURED on the scored trajectory, not derived from
the closed-form logistic.

This wrapper adds exactly one thing and changes nothing else:

    LAPDSim1D.run is replaced by a logging wrapper that calls the ORIGINAL
    method and, in a finally: block, appends (sim._time, sim._cathode_f_em)
    to a trace.

No subclass (class identity is untouched), no harness edit, no solver edit, no
config change, no change to what is integrated or in what order -- the same
pattern as the pd0 "single-delta on instrumentation only" probe. The physics
trajectory is byte-identical to the same harness invocation without the
wrapper; that identity is checkable at the harness's own printed summary and,
for the pd0 arms, at its IDENTITY CONTROL against the recorded arm npz.

Usage (from <checkout>/cablp, PYTHONPATH=$PWD:$PWD/scripts):
    python -u scripts/ea1v_femwrap.py <harness-module> <fem-npz> <harness argv...>
e.g.
    python -u scripts/ea1v_femwrap.py efold1_traj scripts/ea1v_seed_fem.npz \
        --label ea1v_seed --t-target 1e-3 --set cathode_emitting_area=true ...

Writes only the named f_em npz (plus whatever the harness itself writes).
"""
import sys

import numpy as np

from cablp.solvers._sim1d import LAPDSim1D

TRACE = []
_ORIG_RUN = LAPDSim1D.run


def _logging_run(self, *args, **kwargs):
    try:
        return _ORIG_RUN(self, *args, **kwargs)
    finally:
        f = getattr(self, "_cathode_f_em", None)
        TRACE.append((float(getattr(self, "_time", float("nan"))),
                      float("nan") if f is None else float(f)))


LAPDSim1D.run = _logging_run


def main(argv):
    if len(argv) < 3:
        raise SystemExit("usage: ea1v_femwrap.py <harness-module> <fem-npz> "
                         "<harness argv...>")
    harness_name, fem_npz = argv[0], argv[1]
    print(f"== ea1v_femwrap: harness={harness_name} fem_npz={fem_npz} "
          f"(LAPDSim1D.run logging wrapper armed; physics untouched)")
    mod = __import__(harness_name)
    rc = mod.main(argv[2:])

    arr = np.asarray(TRACE, dtype=float)
    print(f"\n   -- f_em TRACE (measured on the scored trajectory) --")
    if arr.size == 0:
        print("   NO run() calls recorded")
        return rc
    f = arr[:, 1]
    finite = np.isfinite(f)
    print(f"   recorded run() calls: {arr.shape[0]}  "
          f"(f_em present on {int(finite.sum())})")
    if not finite.any():
        print("   f_em is None on every step -- CLOSURE NOT ARMED")
    else:
        ff = f[finite]
        tt = arr[finite, 0]
        grew = float(ff[-1]) > float(ff[0])
        print(f"   f_em: {ff[0]:.10e} -> {ff[-1]:.10e} over "
              f"t = [{tt[0]:.6e}, {tt[-1]:.6e}] s")
        print(f"   growth factor {ff[-1] / ff[0]:.6f}  "
              f"= {np.log(ff[-1] / ff[0]):.6f} e-folds")
        print(f"   monotone non-decreasing: "
              f"{bool(np.all(np.diff(ff) >= 0.0))}   ADVANCED OFF SEED: {grew}")
        print(f"   ANTI-VACUITY: {'PASS (f_em not frozen)' if grew else 'VACUOUS (f_em frozen)'}")
    np.savez(fem_npz, fem=arr)
    print(f"   saved {fem_npz}")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
