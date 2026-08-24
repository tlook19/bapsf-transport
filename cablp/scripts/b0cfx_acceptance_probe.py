"""R11 acceptance probe for the density-preserving Maxwellian projection.

The B0c ladder's ``cad_3.125e-06`` arm violated R11 (the I6 independence
band) on the particle ledger. The diagnosis (banked as
``b0cdiag_probe_maxw.py`` / ``b0cdiag_probe_loop.py``, 2026-08-24) traced it
to ``VGrid.maxwellian(exact_moments=True)``: on a cold near-sonic cell of a
coarse velocity grid the two-basis 2x2 compensation solve goes numerically
singular, the huge coefficients it returns amplify the roundoff-level
residue of the basis functions' moment-free property, and the returned bin
masses sum to 1.0625 instead of 1. The collisional-birth projection then
manufactures particles, which the ledger reports as a distribution residual.

This probe is the diagnostician's method, re-run as an acceptance test. It

  1. builds the failing arm, wraps ``TransientDVM.update`` so the LAST
     pre-tick snapshot and the exact plasma kwargs the solver passed are
     captured, and steps until the first neutral tick completes;
  2. censuses ``M_i.sum() - 1`` over every cell's ion Maxwellian at that
     tick's plasma state -- the direct read of the invariant;
  3. restores the snapshot and REPLAYS that same tick, reporting the
     particle ledger residual it closes to.

PASS means: the census is at roundoff on every cell, and the replayed tick's
particle-distribution residual is at roundoff rather than the 4.98e-10 of
record. Run it before and after the fix; the two transcripts are the
evidence pair.

Usage (from <checkout>/cablp, PYTHONPATH set to that same cablp):
    python scripts/b0cfx_acceptance_probe.py
"""
import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from verify_sim1d_k2_dvm import advance_one_step, make_sim  # noqa: E402

from cablp.solvers._sim1d.physics.kinetic_dvm import ledger_residual  # noqa: E402

# The registered B0c arm that failed R11.
ARM_KNOBS = dict(
    neutral_kinetic_dvm_cadence_s=3.125e-06,
    neutral_kinetic_dvm_nvp=6,
    neutral_kinetic_dvm_nvz=16,
)

# Roundoff bar. ROUNDOFF_REL is the k2_dvm suite's own tolerance; the density
# census is an absolute deviation from 1, so it is compared against the same
# number.
BAR = 1.0e-12


def main():
    print("B0c R11 acceptance probe -- density-preserving Maxwellian "
          "projection")
    print(f"  arm knobs: {ARM_KNOBS}")

    sim = make_sim(**ARM_KNOBS)
    dvm = sim._dvm
    grid = dvm.g
    print(f"  velocity grid: ({grid.nvz},{grid.nvp}), "
          f"vz in [{grid.vz.min():.6g}, {grid.vz.max():.6g}] cm/s")

    update = dvm.update
    captured = {}

    def _capture(dt, **kwargs):
        captured["snapshot"] = dvm.snapshot()
        captured["dt"] = float(dt)
        captured["kwargs"] = kwargs
        return update(dt, **kwargs)

    dvm.update = _capture
    steps = 0
    while dvm.updates < 1 and steps < 20000:
        advance_one_step(sim)
        steps += 1
    dvm.update = update
    if not captured:
        raise SystemExit("REFUSED: no neutral tick completed; nothing to probe")
    print(f"  first neutral tick after {steps} steps, "
          f"dt_n = {captured['dt']:.6g} s")

    kwargs = captured["kwargs"]
    Ti = np.asarray(kwargs["Ti_eV"], dtype=float)
    u_i = np.asarray(kwargs["u_i"], dtype=float)
    print(f"  plasma state at that tick: Ti in [{Ti.min():.6g}, "
          f"{Ti.max():.6g}] eV, u_i in [{u_i.min():.6g}, {u_i.max():.6g}] cm/s")

    # (2) the invariant, read directly off the ion Maxwellian of every cell.
    dev = np.empty(dvm.nz)
    for i in range(dvm.nz):
        M_i = grid.maxwellian(max(float(Ti[i]), 0.02), float(u_i[i]))
        dev[i] = float(M_i.sum()) - 1.0
    worst = int(np.argmax(np.abs(dev)))
    census_ok = float(np.max(np.abs(dev))) <= BAR
    print("")
    print(f"  [census] M_i.sum() - 1 over {dvm.nz} cells: "
          f"min {dev.min():.6e}, max {dev.max():.6e}")
    print(f"           worst cell {worst}: Ti = {Ti[worst]:.6g} eV, "
          f"u_i = {u_i[worst]:.6g} cm/s, sum-1 = {dev[worst]:.6e}")
    for i in np.argsort(np.abs(dev))[-5:][::-1]:
        print(f"             cell {int(i):3d}  Ti={Ti[i]:.6g}  "
              f"u_i={u_i[i]:.6g}  sum-1={dev[i]:.6e}")
    print(f"           fixed spectra: M_wall.sum()-1 = "
          f"{float(dvm.M_wall.sum()) - 1.0:.6e}, M_cold.sum()-1 = "
          f"{float(dvm.M_cold.sum()) - 1.0:.6e}")
    print(f"           census {'PASS' if census_ok else 'FAIL'} "
          f"(bar {BAR:g})")

    # (3) replay the same tick from the same snapshot.
    dvm.restore(captured["snapshot"])
    update(captured["dt"], **kwargs)
    residual = ledger_residual(dvm.last_ledger)
    dist = abs(float(residual["distribution"]))
    dist_rel = abs(float(residual["distribution_rel"]))
    dom_rel = abs(float(residual["domain_rel"]))
    replay_ok = dist_rel <= BAR and dom_rel <= BAR
    print("")
    print(f"  [replay] particle ledger on the replayed tick: "
          f"distribution {dist:.6e}, distribution_rel {dist_rel:.6e}, "
          f"domain_rel {dom_rel:.6e}")
    print(f"           replay {'PASS' if replay_ok else 'FAIL'} "
          f"(bar {BAR:g} on both relative forms)")

    ok = census_ok and replay_ok
    print("")
    print(f"  ACCEPTANCE: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
