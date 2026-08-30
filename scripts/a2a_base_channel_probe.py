"""A2a BASE probe: does the QL tail actually cross the anode plane?

Runs the golden-at-stance config for a short window at BASE (no A2a code) and
reports, per deposition call that walks a tail:

  * the tail walk window ``(lo, hi)``,
  * the primary's anode cross cell (the cell the mesh interception fires in),
  * which cells the anomalous channel launches walkers from, and
  * how much launched tail power sits on each side of the anode plane.

The premise the A2a cull stands on is that some launched tail power lies on the
COLUMN side of the anode cross cell, so a walker heading back toward the
cathode crosses the mesh. A run that reports no such power falsifies the
channel and the build stops.

Run from the worktree root with PYTHONPATH=<worktree>.
"""

import sys
import numpy as np

sys.path.insert(0, "scripts")

import cablp
from cablp.solvers._sim1d import LAPDSim1D
from cablp.cathode import beam_deposition as bd
from cablp.solvers._sim1d.physics import cathode as cath

from baseline_sim1d import build_baseline_config

print(f"cablp.__file__ = {cablp.__file__}")
print(f"KERNEL_ID      = {bd.KERNEL_ID if hasattr(bd, 'KERNEL_ID') else 'n/a'}")

RECORD = []
_real_window = cath._plasma_active_window
_real_deposit = cath.deposit_beam
_state = {"window": None, "cross": None}


def _window(geometry):
    out = _real_window(geometry)
    _state["window"] = out
    return out


def _deposit(*args, **kwargs):
    cross = kwargs.get("anode_cross_index")
    if cross is not None:
        _state["cross"] = int(cross)
    return _real_deposit(*args, **kwargs)


cath._plasma_active_window = _window
cath.deposit_beam = _deposit

# Instrument the tail launch inside deposit_beam by wrapping _tail_lane_chains,
# which every ionizing tail walk routes through; its ``plans`` carry the
# per-birth half fluxes in WINDOW-LOCAL indexing.
_real_chains = bd._tail_lane_chains


def _chains(plans, nn_w, ne_w, Te_w, dz_w, march_kwargs, tail_lo, tail_hi,
            reflect_face, E_reflect, cull=None):
    for E_walk, half_flux, ionizes in plans:
        births = np.flatnonzero(np.asarray(half_flux) > 0.0)
        if births.size:
            RECORD.append(
                (
                    float(E_walk),
                    int(births[0]),
                    int(births[-1]),
                    int(births.size),
                    float(np.sum(np.asarray(half_flux)[births]) * 2.0 * E_walk),
                    tail_lo,
                    tail_hi,
                    reflect_face,
                    bool(ionizes),
                )
            )
    return _real_chains(plans, nn_w, ne_w, Te_w, dz_w, march_kwargs, tail_lo,
                        tail_hi, reflect_face, E_reflect, cull=cull)


bd._tail_lane_chains = _chains

params, flags = build_baseline_config()
sim = LAPDSim1D(params, flags)
sim.start_simulation(t_end=0.0165, dt=None, operator_split=None, max_steps=150000)

win = _state["window"]
cross = _state["cross"]
print(f"tail walk window (lo, hi) = {win}")
print(f"primary anode cross cell  = {cross}")
if win is not None and cross is not None:
    print(f"anode cross, window-local = {cross - win[0]}")
print(f"tail launch records       = {len(RECORD)}")
if not RECORD:
    print("FALSIFIED: no ionizing tail launch recorded in this window")
    raise SystemExit(2)

lo = win[0]
local_cross = cross - lo
below = 0
across = 0
pow_below = 0.0
pow_across = 0.0
first_birth = min(r[1] for r in RECORD)
last_birth = max(r[2] for r in RECORD)
for (E, b0, b1, n, p, tlo, thi, rf, ion) in RECORD:
    if b1 >= local_cross:
        across += 1
        pow_across += p
    else:
        below += 1
        pow_below += p
print(f"reflect_face              = {RECORD[0][7]}")
print(f"ionizing legs             = {RECORD[0][8]}")
print(f"birth cells (window-local): min={first_birth} max={last_birth}")
print(f"records whose birth span reaches/passes the anode cell: {across}")
print(f"records entirely cathode-side of the anode cell:        {below}")
print(f"launched tail power, column side [erg/s]: {pow_across:.6e}")
print(f"launched tail power, gap side    [erg/s]: {pow_below:.6e}")
if across == 0:
    print("FALSIFIED: no tail power is launched on the column side of the "
          "anode plane; the first-crossing cull would have nothing to cull")
    raise SystemExit(2)
print("PREMISE HOLDS: tail walkers are launched on the column side of the "
      "anode plane and their -B half crosses it.")
