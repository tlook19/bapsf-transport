"""fnb3 -- WHERE THE 3% HOT-CHANNEL BRANCHING SHORTFALL COMES FROM.

fa4 read the branching closure off a saved artifact as

    S_cx  ==  wall deaths + in-flight ionizations

    S_cx    = -rhs_terms/neutral_cx_channel/nn   integrated on V_col
    wall    =  rhs_terms/neutral_hot_channel/nn_a integrated on V_ann
    ionized =  rhs_terms/neutral_hot_channel/n    integrated on V_col

and found it closing to 0.970, not 1. This probe asks what the missing 3% is.

THE MODEL'S OWN IDENTITY IS EXACT. Writing the fixed point as extensive rates,
``births_ext = S_cx*Vp + residence^T (b_recx * births_ext)``, and summing over
cells with ``residence`` rows closing to 1 gives
``sum(births_ext (1 - b_recx)) == sum(S_cx Vp)``; since
``b_wall + b_recx + b_ion == 1`` exactly, that IS
``sum(wall_ext) + sum(ionized_ext) == sum(S_cx Vp)``. Part 1 checks that
directly against the newly-saved per-cell diagnostics, which are the PRE-MASK
rates.

THE END-PLANE FOLD IS NOT A LOSS EITHER. ``ballistic_flight_kernels`` clips a
flight that would leave through an end plane back onto the end cell, so the
landing rows still close to 1 and no atom leaves the inventory.

What DOES remove atoms from the saved ledger is
``_apply_active_plasma_topology``: it runs AFTER the ballistic spread and zeros
the hot channel's rows on plasma-dead cells. An atom BORN on a live cell whose
flight ends on a dead one has its birth debited (the cx row is live there) and
its deposit deleted (the hot row is masked there) -- so the two halves of the
identity are masked on DIFFERENT cell sets and the read no longer closes.

Part 1 runs the flag-ON smoke configuration and decomposes the read exactly.
Part 2 rebuilds the fa4 arm's geometry and kernels from its own params_json
with NO solve, and localizes the measured deficit against the plasma-dead
cells. Nothing is fitted, tuned, or written back.

Usage (PYTHONPATH=<worktree>/cablp):
    fnb3_closure.py [LEGACY.h5 ...]
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np

import cablp
from cablp.solvers._sim1d import LAPDSim1D, default_config

print("cablp package file:", cablp.__file__)

HERE = Path(__file__).resolve().parent


def rule(title):
    print("=" * 100)
    print(title)
    print("=" * 100)


# --------------------------------------------------------------------------
# Part 1 -- the exact decomposition on a fresh flag-ON run
# --------------------------------------------------------------------------
rule("PART 1 -- exact decomposition on a fresh flag-ON run (nx=24 smoke config)")

params, flags = default_config()
params["nx"] = 24
params["S_gp"] = 9010
params["max_steps_action"] = "stop"
flags["neutral_momentum"] = True
flags["neutral_two_zone"] = True
flags["neutral_energy"] = True

sim = LAPDSim1D(input_dict=params, input_flags=flags)
sim.start_simulation(t_end=1.0e-3, dt=None, operator_split=None, max_steps=9000)
res = sim.get_results()
print(f"status={res.run_status} steps={res.steps} t_end={res.time[-1]:.6e} s "
      f"samples={res.time.size}")

Vp = np.asarray(res.plasma_volume_cm3, dtype=float)
Vm = np.asarray(res.neutral_volume_cm3, dtype=float)
Vann = Vm - Vp
active = np.asarray(res.plasma_active, dtype=float) > 0.0
dead = ~active
landing, residence, end_fraction = sim._hot_neutral_kernels

print(f"\ncells={Vp.size}  plasma-dead cells={int(np.count_nonzero(dead))} "
      f"at z={np.asarray(res.z_cm)[dead]}")
print(f"cell roles there: "
      f"{[str(r) for r in np.asarray(res.cell_role)[dead]]}")

# Use the LAST saved sample, where the discharge is running.
k = -1
S_cx = np.asarray(res.hot_S_cx[k], dtype=float)
wall = np.asarray(res.hot_wall[k], dtype=float)
recx = np.asarray(res.hot_recx[k], dtype=float)
ion = np.asarray(res.hot_ionized[k], dtype=float)
births = np.asarray(res.hot_births[k], dtype=float)
endf = np.asarray(res.hot_end_fraction[k], dtype=float)

birth_ext = float(np.sum(births * Vp))
S_ext = float(np.sum(S_cx * Vp))
wall_ext = float(np.sum(wall * Vp))
recx_ext = float(np.sum(recx * Vp))
ion_ext = float(np.sum(ion * Vp))

print("\n1a. THE MODEL'S OWN IDENTITY, on the pre-mask per-cell diagnostics")
print(f"    sum(S_cx*Vp)                       = {S_ext:.10e}  1/s")
print(f"    sum(wall*Vp) + sum(ionized*Vp)     = {wall_ext + ion_ext:.10e}  1/s")
id_rel = abs(wall_ext + ion_ext - S_ext) / max(abs(S_ext), 1e-300)
print(f"    relative residual                  = {id_rel:.3e}   "
      f"(exact identity; only linear-solve roundoff)")
print(f"    per-cell fates close on births:    "
      f"max|wall+recx+ion - births|/births = "
      f"{float(np.max(np.abs(wall + recx + ion - births) / np.maximum(births, 1e-300))):.3e}")
print(f"    re-CX replacement traffic          = {recx_ext:.6e} 1/s "
      f"({recx_ext / max(birth_ext, 1e-300) * 100:.2f}% of all births; it "
      f"replaces its own atom and so cancels from the closure)")

print("\n1b. fa4's READ of the same instant, off the MASKED ledger rows")
cx_row = -np.asarray(res.rhs_terms["neutral_cx_channel"]["nn"][k], dtype=float)
wall_row = np.asarray(res.rhs_terms["neutral_hot_channel"]["nn_a"][k], dtype=float)
ion_row = np.asarray(res.rhs_terms["neutral_hot_channel"]["n"][k], dtype=float)
A = float(cx_row @ Vp)
D = float(wall_row @ Vann) + float(ion_row @ Vp)
print(f"    S_cx read  A = -cx.nn  . V_col     = {A:.10e}  1/s")
print(f"    wall+ion   D                       = {D:.10e}  1/s")
print(f"    D / A                              = {D / A:.6f}   "
      f"deficit = {A - D:.6e} 1/s ({100 * (1 - D / A):.3f}%)")

print("\n1c. ATTRIBUTION -- the deposit the topology mask deleted")
# Extensive deposit landing on / residing over each cell, PRE-mask, rebuilt
# from the birth-cell rates and the kernels the solver itself used.
landed_ext = (wall * Vp) @ landing
ionized_ext = (ion * Vp) @ residence
lost_wall = float(np.sum(landed_ext[dead]))
lost_ion = float(np.sum(ionized_ext[dead]))
born_dead = float(np.sum((S_cx * Vp)[dead]))
print(f"    wall deposit landing on plasma-dead cells   = {lost_wall:.6e} 1/s")
print(f"    ionization deposit residing over dead cells = {lost_ion:.6e} 1/s")
print(f"    S_cx births ON dead cells (also masked out) = {born_dead:.6e} 1/s")
predicted = (lost_wall + lost_ion) - born_dead
print(f"    predicted deficit A - D                     = {predicted:.6e} 1/s")
print(f"    measured  deficit A - D                     = {A - D:.6e} 1/s")
print(f"    attribution residual (relative)             = "
      f"{abs(predicted - (A - D)) / max(abs(A - D), 1e-300):.3e}")

print("\n1d. THE END-PLANE FOLD -- is it the shortfall?")
fold = float(np.sum(endf * wall * Vp)) / max(wall_ext, 1e-300)
print(f"    wall-weighted end-plane fold fraction       = {fold:.6f} "
      f"({100 * fold:.3f}%)")
print(f"    fold mean over cells / max over cells       = "
      f"{float(np.mean(endf)):.6f} / {float(np.max(endf)):.6f}")
print(f"    landing rows still close to 1 within        = "
      f"{float(np.max(np.abs(landing.sum(axis=1) - 1.0))):.2e}   "
      f"-> the fold RELOCATES atoms, it does not remove them")
low_end, high_end = 0, Vp.size - 1
print(f"    end cells: z[{low_end}]={float(np.asarray(res.z_cm)[low_end]):.1f} "
      f"active={bool(active[low_end])}   "
      f"z[{high_end}]={float(np.asarray(res.z_cm)[high_end]):.1f} "
      f"active={bool(active[high_end])}")


# --------------------------------------------------------------------------
# Part 2 -- the fa4 arm, from its own artifact
# --------------------------------------------------------------------------
for path in sys.argv[1:]:
    rule(f"PART 2 -- legacy artifact {path}")
    with h5py.File(path, "r") as h5:
        p = json.loads(h5.attrs["params_json"])
        f = json.loads(h5.attrs["flags_json"])
        g = h5["geometry"]
        z = g["z_cm"][:]
        vp = g["plasma_volume_cm3"][:]
        vann = g["neutral_volume_cm3"][:] - vp
        pa = g["plasma_active"][:] > 0.0
        roles = np.asarray(
            [r.decode() if isinstance(r, bytes) else str(r)
             for r in g["cell_role"][:]]
        )
        t_ms = h5["time"][:] * 1e3
        cx = -h5["rhs_terms/neutral_cx_channel/nn"][:]
        wl = h5["rhs_terms/neutral_hot_channel/nn_a"][:]
        io_ = h5["rhs_terms/neutral_hot_channel/n"][:]
        has_diag = "hot_S_cx" in h5
    print(f"    cells={z.size}  saved to {t_ms[-1]:.1f} ms  "
          f"new hot diagnostics present in file: {has_diag}")
    print(f"    plasma-dead cells: z={z[~pa]}  roles={list(roles[~pa])}")
    src = z <= 100.0
    # fa4's own plateau window first, so the number is directly comparable to
    # the 0.96970 it published; then the whole post-15 ms tail.
    for lo, hi in ((15.0, 19.5), (15.0, t_ms[-1])):
        i0 = int(np.searchsorted(t_ms, lo))
        i1 = min(int(np.searchsorted(t_ms, hi)) + 1, t_ms.size)
        sl = slice(i0, i1)
        A = float((cx[sl] @ vp).mean())
        D = float((wl[sl] @ vann).mean() + (io_[sl] @ vp).mean())
        A_s = float((cx[sl][:, src] @ vp[src]).mean())
        D_s = float((wl[sl][:, src] @ vann[src]).mean()
                    + (io_[sl][:, src] @ vp[src]).mean())
        print(f"    window {lo:5.1f}-{hi:5.1f} ms: A={A:.6e}  D={D:.6e}  "
              f"D/A={D / A:.6f}  deficit={A - D:.6e}")
        print(f"        SOURCE z<=100 alone: deficit={A_s - D_s:.6e}  "
              f"= {100 * (A_s - D_s) / max(A - D, 1e-300):.2f}% of the whole "
              f"column's deficit")

    # Rebuild the geometry and kernels the run used -- construction only, no
    # solve -- so the mask hypothesis can be tested on THIS grid.
    sim2 = LAPDSim1D(input_dict=p, input_flags=f)
    land2, res2, end2 = sim2._hot_neutral_kernels
    assert np.array_equal(
        np.asarray(sim2._geometry.plasma_active, dtype=bool), pa
    ), "rebuilt geometry disagrees with the artifact"
    to_dead_land = land2[:, ~pa].sum(axis=1)
    to_dead_res = res2[:, ~pa].sum(axis=1)
    live = np.where(pa & (to_dead_land > 0.0))[0]
    print(f"\n    kernel mass from LIVE birth cells onto DEAD cells:")
    print(f"      live cells with any landing on a dead cell: {live.size}")
    for i in live[:12]:
        print(f"        z={z[i]:8.1f} ({roles[i]:<12s}) landing->dead="
              f"{to_dead_land[i]:.4f}  residence->dead={to_dead_res[i]:.4f}  "
              f"end-fold={end2[i]:.4f}")
    print(f"      end-plane fold fraction: mean={float(np.mean(end2)):.6f} "
          f"max={float(np.max(end2)):.6f} at z={z[int(np.argmax(end2))]:.1f}")
    print(f"      low-z end cell z={z[0]:.1f} role={roles[0]} "
          f"plasma_active={bool(pa[0])}")
    print(f"      high-z end cell z={z[-1]:.1f} role={roles[-1]} "
          f"plasma_active={bool(pa[-1])}")
