"""fnb3 -- the hot diagnostics survive save/load, and old files still open.

Three checks, all on real artifacts:

  R1  a flag-ON run's HDF5 now CARRIES the eleven per-cell hot-channel
      diagnostics, and ``load_result_hdf5`` returns each of them bit-identical
      to the live result's array.
  R2  the split is exact: ``hot_Ei_recx + hot_Ei_ionization`` is bit-identical
      to the saved ``rhs_terms/neutral_hot_channel/Ei`` row wherever the
      plasma-topology mask left that row alone, and the physics function's own
      ``diagnostics`` halves sum bitwise to its ``Ei`` row with no mask in the
      way at all.
  R3  a file written WITHOUT the new datasets (the historical layout, built by
      deleting them from a written file) still loads, with the attributes
      simply absent rather than zero.

Read-only with respect to the model: nothing is tuned and no baseline is
touched. Run with PYTHONPATH=<worktree>/cablp.
"""
import shutil
from pathlib import Path

import h5py
import numpy as np

import cablp
from cablp.solvers._sim1d import LAPDSim1D, default_config, load_result_hdf5
from cablp.solvers._sim1d.results.io import save_result_hdf5
from cablp.solvers._sim1d.core.state import derive_state
from cablp.solvers._sim1d.physics.hot_neutrals import (
    HOT_CHANNEL_DIAGNOSTIC_FIELDS,
    neutral_hot_channel_rhs,
)

print("cablp package file:", cablp.__file__)

HERE = Path(__file__).resolve().parent
RUN_H5 = HERE / "fnb3_roundtrip.h5"
OLD_H5 = HERE / "fnb3_roundtrip_legacy.h5"


def raw(a):
    return np.ascontiguousarray(np.asarray(a, dtype=float)).view(np.uint64)


def bitwise(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return a.shape == b.shape and np.array_equal(raw(a), raw(b))


params, flags = default_config()
params["nx"] = 24
params["S_gp"] = 9010
params["max_steps_action"] = "stop"
flags["neutral_momentum"] = True
flags["neutral_two_zone"] = True
flags["neutral_energy"] = True

sim = LAPDSim1D(input_dict=params, input_flags=flags)
sim.start_simulation(t_end=5.0e-4, dt=None, operator_split=None, max_steps=6000)
result = sim.get_results()
print(f"status={result.run_status} steps={result.steps} "
      f"t_end={result.time[-1]:.6e} s  samples={result.time.size}")

save_result_hdf5(RUN_H5, result)
loaded = load_result_hdf5(RUN_H5)

print("\n--- R1: the diagnostics are written and round-trip bitwise ---")
r1 = True
with h5py.File(RUN_H5, "r") as h5:
    for name in HOT_CHANNEL_DIAGNOSTIC_FIELDS:
        present = name in h5
        live = getattr(result, name, None)
        back = getattr(loaded, name, None)
        same = present and live is not None and back is not None and bitwise(
            live, back
        )
        r1 &= same
        shape = tuple(h5[name].shape) if present else None
        finite = bool(np.all(np.isfinite(live))) if live is not None else False
        print(f"  {name:18s} in file={str(present):5s} shape={str(shape):12s} "
              f"finite={str(finite):5s} round-trip bitwise={same}")
print(f"  R1: {'PASS' if r1 else 'FAIL'}")

print("\n--- R2: the two Ei halves sum bit-identically to the combined row ---")
# (a) At the physics function, with no plasma-topology mask in the way: the
#     machine-exact form of the identity.
state = sim.state
rate = 1.0e3 * np.ones_like(np.asarray(state.nn, dtype=float))
rhs, diag = neutral_hot_channel_rhs(
    state=state, floors=sim._floors, ion_mass_g=sim._ion_mass_g,
    geometry=sim._geometry, gas_type=sim._gas_type,
    Tn_eV=float(params["Tn_K"]) * 1.380649e-16 / 1.602176634e-12,
    ionization_rate_per_neutral=rate, kernels=sim._hot_neutral_kernels,
    I_ion=sim._I_ion, wind_column_factor=sim._wind_column_factor,
)
halves = np.asarray(diag["hot_Ei_recx"]) + np.asarray(diag["hot_Ei_ionization"])
r2a = bitwise(halves, rhs.Ei)
print(f"  (a) function level, unmasked: "
      f"hot_Ei_recx + hot_Ei_ionization == rhs.Ei bitwise = {r2a}")
print(f"      scale: max|Ei| = {float(np.max(np.abs(rhs.Ei))):.6e} erg/cm^3/s  "
      f"max|recx| = {float(np.max(np.abs(diag['hot_Ei_recx']))):.6e}  "
      f"max|ioniz| = {float(np.max(np.abs(diag['hot_Ei_ionization']))):.6e}")

# (b) Against the SAVED ledger row, which the caller masked on plasma-dead
#     cells. The mask writes a hard 0.0, so the identity survives it wherever
#     the mask left the row alone; the masked cells are counted, not hidden.
saved_Ei = np.asarray(loaded.rhs_terms["neutral_hot_channel"]["Ei"], dtype=float)
sum_Ei = (np.asarray(loaded.hot_Ei_recx, dtype=float)
          + np.asarray(loaded.hot_Ei_ionization, dtype=float))
active = np.asarray(loaded.plasma_active, dtype=float) > 0.0
mask = np.broadcast_to(active, saved_Ei.shape)
r2b = bitwise(saved_Ei[mask], sum_Ei[mask])
n_masked = int(np.count_nonzero(~mask))
elsewhere = int(np.count_nonzero(saved_Ei[~mask] != 0.0)) if n_masked else 0
print(f"  (b) saved trajectory, on plasma-ACTIVE cells "
      f"({int(np.count_nonzero(mask))} of {mask.size}): bitwise = {r2b}")
print(f"      plasma-dead entries masked out of the row: {n_masked} "
      f"(nonzero there: {elsewhere})")
r2 = r2a and r2b
print(f"  R2: {'PASS' if r2 else 'FAIL'}")

print("\n--- R3: a file without the new datasets still loads ---")
shutil.copyfile(RUN_H5, OLD_H5)
with h5py.File(OLD_H5, "a") as h5:
    for name in HOT_CHANNEL_DIAGNOSTIC_FIELDS:
        if name in h5:
            del h5[name]
legacy = load_result_hdf5(OLD_H5)
absent = [n for n in HOT_CHANNEL_DIAGNOSTIC_FIELDS if not hasattr(legacy, n)]
r3 = (
    len(absent) == len(HOT_CHANNEL_DIAGNOSTIC_FIELDS)
    and bitwise(legacy.En, loaded.En)
    and bitwise(legacy.y, loaded.y)
)
print(f"  attributes absent (not zero-filled): {len(absent)} of "
      f"{len(HOT_CHANNEL_DIAGNOSTIC_FIELDS)}")
print(f"  the rest of the file is unchanged (En, y bitwise): "
      f"{bitwise(legacy.En, loaded.En) and bitwise(legacy.y, loaded.y)}")
print(f"  R3: {'PASS' if r3 else 'FAIL'}")

print()
print("ROUNDTRIP:", "PASS" if (r1 and r2 and r3) else "FAIL")
