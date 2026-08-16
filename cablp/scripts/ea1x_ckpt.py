"""ea1x: CHECKPOINTING TRACE RECORDER -- INSTRUMENTATION ONLY, harness UNMODIFIED.

WHY THIS EXISTS (measured, not anticipated): the first ea1x ms-class arm
(t-target 5e-3 s) was KILLED AT THE 900 s PRE-REGISTERED WALL CAP, and
efold1_traj.py writes its npz and prints its reads only AFTER the stepping loop
returns -- so the cap-kill left NO artifact at all and no measurement of how far
the arm got. This wrapper adds crash/kill survivability and nothing else.

WHAT IT DOES

  1. imports scripts/ea1v_femwrap.py UNMODIFIED (the stage-1 f_em recorder of
     record, whose inertness is already proven raw-byte-identical), which
     patches LAPDSim1D.run with its logging wrapper;
  2. wraps THAT with a second logging wrapper which, in a finally: block after
     the original run() returns, appends one row transcribed VERBATIM from
     efold1_traj.py:96-108 -- the same expressions, evaluated at the same point
     in the same loop:

        t, dt, I_loop, phi_c, I_eth*, <n>_active, n[2], n[7], n_max

     plus a tenth column, f_em (sim._cathode_f_em, None -> NaN);
  3. every --ckpt-every seconds of wall it dumps the rows so far to
     <ckpt>.npz via a tmp-file + os.replace, so a kill at the cap leaves a
     complete-up-to-that-instant trajectory;
  4. calls ea1v_femwrap.main(argv) -- i.e. the arm runs through the SAME
     harness invocation as stage 1, and if it finishes under the cap the
     standard artifacts (efold1 npz, f_em npz) are written as usual.

No subclass, no harness edit, no solver edit, no config change, no change to
what is integrated or in what order. Reading solver attributes and appending to
a list is inert exactly as the stage-1 wrapper's f_em read is inert; the
INERTNESS IDENTITY CONTROL is run explicitly (a 1e-3 s central-seed arm whose
recorded columns are compared raw-byte against the stage-1 record
scripts/ea1v_seed.npz).

Usage (from <checkout>/cablp, PYTHONPATH=$PWD:$PWD/scripts):
    python -u scripts/ea1x_ckpt.py <ckpt-npz> <every-s> <harness-module> \
        <fem-npz> <harness argv...>
"""
import os
import sys
import time

import numpy as np

import ea1v_femwrap                      # UNMODIFIED; patches LAPDSim1D.run
from cablp.solvers._sim1d import LAPDSim1D

ROWS = []
_STATE = {"t_prev": 0.0, "active": None, "last": 0.0, "path": None,
          "every": 20.0, "dumps": 0}
_WRAPPED_RUN = LAPDSim1D.run             # = ea1v_femwrap._logging_run


def _dump(final=False):
    path = _STATE["path"]
    if path is None:
        return
    arr = np.asarray(ROWS, dtype=float)
    tmp = path + ".tmp.npz"
    np.savez(tmp, trace=arr)
    os.replace(tmp, path)
    _STATE["dumps"] += 1
    if final:
        print(f"   [ckpt] FINAL dump: {arr.shape[0]} rows -> {path} "
              f"({_STATE['dumps']} dumps total)", flush=True)


def _recording_run(self, *args, **kwargs):
    try:
        return _WRAPPED_RUN(self, *args, **kwargs)
    finally:
        try:
            if _STATE["active"] is None:
                _STATE["active"] = np.asarray(self.geometry.plasma_active, bool)
            active = _STATE["active"]
            t = float(self._time)
            E0 = G0 = float("nan")
            solve = self._cathode_solve
            br = getattr(solve, "beam_result", None) if solve is not None else None
            res = getattr(br, "result", None) if br is not None else None
            if res is not None:
                E0 = float(res.phi_c)
                G0 = float(res.I_eth_star)
            n = np.asarray(self.state.n, float)
            f = getattr(self, "_cathode_f_em", None)
            ROWS.append((t, t - _STATE["t_prev"], float(self._circuit_I_loop),
                         E0, G0, float(n[active].mean()), float(n[2]),
                         float(n[7]), float(n.max()),
                         float("nan") if f is None else float(f)))
            _STATE["t_prev"] = t
            now = time.time()
            if now - _STATE["last"] >= _STATE["every"]:
                _STATE["last"] = now
                _dump()
                print(f"   [ckpt] t={t:.6e} s  rows={len(ROWS)}  "
                      f"I_loop={float(self._circuit_I_loop):.6g} A  "
                      f"<n>={float(n[active].mean()):.6g}  "
                      f"wall={now - _STATE['t0']:.0f} s", flush=True)
        except Exception as err:                   # instrument, never fatal
            print(f"   [ckpt] recorder error (ignored): "
                  f"{type(err).__name__}: {err}", flush=True)


LAPDSim1D.run = _recording_run


def main(argv):
    if len(argv) < 4:
        raise SystemExit("usage: ea1x_ckpt.py <ckpt-npz> <every-s> "
                         "<harness-module> <fem-npz> <harness argv...>")
    _STATE["path"] = argv[0]
    _STATE["every"] = float(argv[1])
    _STATE["t0"] = _STATE["last"] = time.time()
    print(f"== ea1x_ckpt: checkpoint={_STATE['path']} every {_STATE['every']:g} s "
          f"(recorder layered OVER the unmodified ea1v_femwrap; physics untouched)",
          flush=True)
    try:
        rc = ea1v_femwrap.main(argv[2:])
    finally:
        _dump(final=True)
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
