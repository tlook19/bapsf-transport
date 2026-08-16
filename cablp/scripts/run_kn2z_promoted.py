"""KN2Z revival run: promoted production stance + kinetic two-zone arm.

Config authority is compare_sim1d_es1.run_model (imported, not reimplemented).
The only local additions are PURE INSTRUMENTATION, injected by shimming
LAPDSim1D.start_simulation / _kinetic_refresh at the wrapper level:
  - a throttled progress printer (heartbeat -> log, for G1 throughput)
  - a wall-clock counter around _kinetic_refresh (refresh cost)
Neither touches physics, config, or accepted-step state.

M0 (re-registered 2026-07-27): promoted production defaults
  + neutral_two_zone=True (flag; kinetic engine prerequisite)
  + neutral_model='kinetic'
  + neutral_kinetic_nvz=48, nvp=12, refresh_s=5e-4, refresh_tol=0.2 (explicit,
    stated even though they equal the current solver defaults, so the
    artifact's params_json is self-documenting)
  + tau_afterglow=0.006 (run cost; no-op vs the reference)
nx = PRODUCTION_NX = 240 (driver default, untouched).
"""

import sys
import time as _time

sys.path.insert(0, "/Users/tlook/bapsf/bapsf-transport/cablp/scripts")

from cablp.solvers._sim1d import LAPDSim1D
from cablp.solvers._sim1d.results.io import save_result_hdf5
import compare_sim1d_es1 as C

OUT = "/Users/tlook/bapsf/bapsf-transport/cablp/scripts/es1_kn2z_promoted_nx240.h5"

KINETIC_EXTRA = {
    "neutral_model": "kinetic",
    "neutral_kinetic_nvz": 48,
    "neutral_kinetic_nvp": 12,
    "neutral_kinetic_refresh_s": 5e-4,
    "neutral_kinetic_refresh_tol": 0.2,
    "tau_afterglow": 0.006,
}
KINETIC_FLAGS = {"neutral_two_zone": True}

# ---- instrumentation shims (no physics) ---------------------------------
_refresh = {"n": 0, "wall": 0.0}
_orig_refresh = LAPDSim1D._kinetic_refresh


def _timed_refresh(self, t):
    t0 = _time.perf_counter()
    try:
        return _orig_refresh(self, t)
    finally:
        _refresh["n"] += 1
        _refresh["wall"] += _time.perf_counter() - t0


LAPDSim1D._kinetic_refresh = _timed_refresh

T0 = _time.perf_counter()
_hb = {"last": 0.0}

# Live G2 recorder: the trajectory accumulates in locals inside run(), so a
# kill-at-cap would leave no artifact at all. This snapshots ONLY the G2
# observable (n at the five port cells) straight off the solver state -- a
# pure read, no mutation -- and dumps it on exit/SIGTERM.
import atexit
import signal
import numpy as _np

PORT_Z = (470.05, 789.55, 1045.15, 1428.55, 1716.10)
PORTS = (11, 21, 29, 41, 50)
REC = {"t": [], "n": [], "cells": None, "z": None}
PORT_NPZ = ("/Users/tlook/bapsf/bapsf-transport/cablp/scripts/"
            "es1_kn2z_promoted_nx240_ports.npz")


def _dump_ports(*_a):
    if not REC["t"]:
        return
    _np.savez(
        PORT_NPZ,
        time_s=_np.asarray(REC["t"]),
        n_cm3=_np.asarray(REC["n"]),
        ports=_np.asarray(PORTS),
        port_z_cm=_np.asarray(PORT_Z),
        cells=_np.asarray(REC["cells"]),
        cell_z_cm=_np.asarray(REC["z"]),
        refresh_n=_refresh["n"],
        refresh_wall_s=_refresh["wall"],
    )
    print(f"[dump] wrote {PORT_NPZ} with {len(REC['t'])} samples", flush=True)


atexit.register(_dump_ports)
for _s in (signal.SIGTERM, signal.SIGINT):
    signal.signal(_s, lambda *a: (_dump_ports(), sys.exit(143)))


def _record(sim):
    if REC["cells"] is None:
        g = sim.geometry
        zc = _np.asarray(g.z_cm, dtype=float)
        act = _np.asarray(getattr(g, "plasma_active", _np.ones(zc.shape, bool)))
        cand = _np.flatnonzero(act)
        REC["cells"] = [int(cand[_np.argmin(_np.abs(zc[cand] - zp))])
                        for zp in PORT_Z]
        REC["z"] = [float(zc[c]) for c in REC["cells"]]
    REC["t"].append(float(sim._time))
    REC["n"].append(_np.asarray(sim.state.n, dtype=float)[REC["cells"]].copy())


def _progress(p):
    now = _time.perf_counter() - T0
    if now - _hb["last"] < 20.0 and p.fraction < 1.0:
        return
    _hb["last"] = now
    print(
        f"[{now:8.1f}s] step={p.step:8d} t={p.time:.6f}s "
        f"frac={p.fraction:6.3f} phase={p.phase:>16s} "
        f"dt={p.accepted_dt:.3e} cap={p.step_cap} "
        f"saved={p.saved_samples} steps/s={p.step / max(now, 1e-9):7.1f} "
        f"refresh_n={_refresh['n']} refresh_s={_refresh['wall']:.1f} "
        f"rec={len(REC['t'])}",
        flush=True,
    )
    _dump_ports()  # periodic, so even a hard kill leaves the G2 observable


_orig_start = LAPDSim1D.start_simulation


def _start_with_progress(self, *a, **kw):
    # NB progress_callback receives a bare float (fraction); progress_tracker
    # receives the full SimulationProgress1D. We want the latter.
    last = {"t": -1.0}

    def tracker(p):
        # record the G2 observable on a 10 us model-time cadence (matches the
        # reference artifact's save cadence); only in the plasma phases
        if p.phase not in ("equilibrium_off",):
            if self._time - last["t"] >= 1e-5:
                last["t"] = self._time
                try:
                    _record(self)
                except Exception as exc:  # never let instrumentation kill a run
                    print(f"[record] skipped: {exc!r}", flush=True)
        _progress(p)

    kw.setdefault("progress_tracker", tracker)
    kw.setdefault("progress_interval_s", 0.0)
    return _orig_start(self, *a, **kw)


LAPDSim1D.start_simulation = _start_with_progress
# -------------------------------------------------------------------------

print(f"nx={C.PRODUCTION_NX} extra={KINETIC_EXTRA} flags_extra={KINETIC_FLAGS}",
      flush=True)

result, geometry, params, flags = C.run_model(
    nx=C.PRODUCTION_NX,
    extra=KINETIC_EXTRA,
    flags_extra=KINETIC_FLAGS,
)
wall = _time.perf_counter() - T0
print(f"SOLVE DONE wall={wall:.1f}s steps={result.steps} "
      f"steps/s={result.steps / wall:.2f} "
      f"refresh_n={_refresh['n']} refresh_wall_s={_refresh['wall']:.1f} "
      f"refresh_frac={_refresh['wall'] / wall:.4f}", flush=True)

save_result_hdf5(OUT, result, params=params, flags=flags)
print(f"saved {OUT}", flush=True)
