"""L2 arm 4 (free C_R fit): regime / step-cap census on the final artifact.

SCRATCH ANALYSIS SCRIPT (untracked). Read-only over saved artifacts; solves
nothing and writes no h5. Reproduces the same per-arm fields the arm-1/2
report carried (scripts/l2arm_regime_inventory.txt section C), so the arm-4
disclosure reads sit on the identical definitions.

Datasets are read individually with h5py rather than through
load_result_hdf5: these artifacts are ~2.5 GB each.

Usage::

    python scripts/l2a4_census.py LABEL=PATH [LABEL=PATH ...]
"""

import collections
import json
import sys

import h5py
import numpy as np


def _decode(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _decode_arr(values):
    return np.asarray([_decode(v) for v in values])


PARAM_KEYS = (
    "C_R",
    "cathode_circuit_bound_object",
    "cathode_phi_c_cap_V",
    "circuit_picard_max_iter",
    "max_step_retries",
    "dt_min_lock_max_steps",
    "ignition_accepted_step_cap",
    "ignition_wall_clock_cap_s",
)
FLAG_KEYS = (
    "cathode_circuit_voltage_bound",
    "regime_tracer",
    "regime_vessel_node",
    "TwinCathode",
    "electron_heat_flux_limit",
)
SCALAR_KEYS = (
    "source_bound_active",
    "end_bound_active",
    "source_phi_c_at_cap",
    "end_phi_c_at_cap",
    "has_solution",
    "has_twin_solution",
    "twin_cathode",
    "floating",
)
CEILING_KEYS = ("source_phi_c_ceiling_V", "end_phi_c_ceiling_V")


def census(values):
    return dict(collections.Counter(values.tolist()))


def report(label, path):
    with h5py.File(path, "r") as h5:
        params = json.loads(_decode(h5.attrs["params_json"]))
        flags = json.loads(_decode(h5.attrs["flags_json"]))
        diag = h5["cathode_diagnostics"]
        t_save = np.asarray(h5["time"][:], float)
        phase = _decode_arr(h5["phase"][:])
        arrays = {k: np.asarray(diag[k][:], float) for k in SCALAR_KEYS + CEILING_KEYS}
        src_regime = _decode_arr(diag["source_regime"][:])
        end_regime = _decode_arr(diag["end_regime"][:])
        clamped = np.asarray(h5["diagnostics/clamped_to_dt_min"][:], float)
        retry = np.asarray(h5["diagnostics/retry_count"][:], float)
        step_cap = _decode_arr(h5["diagnostics/step_cap"][:])
        active = _decode_arr(h5["diagnostics/active_constraint"][:])
        n_rej = int(h5["timestep_rejection_events/time"].shape[0]) \
            if "timestep_rejection_events" in h5 \
            and "time" in h5["timestep_rejection_events"] else 0
        attrs = {k: h5.attrs[k] for k in ("steps", "run_status", "final_time",
                                          "compiled_kernels") if k in h5.attrs}

    hits = np.flatnonzero(phase == "main_discharge")
    t0 = float(t_save[hits[0]]) if hits.size else float("nan")
    t_ms = (t_save - t0) * 1e3

    print(f"--- {label} ({path}) ---")
    print(f"   run_status={attrs.get('run_status')} steps={attrs.get('steps')} "
          f"final_time={float(attrs.get('final_time', float('nan'))):.9f} s "
          f"compiled_kernels={attrs.get('compiled_kernels')}")
    for k in PARAM_KEYS:
        if k in params:
            print(f"   params  {k} = {params[k]!r}")
    for k in FLAG_KEYS:
        if k in flags:
            print(f"   flags   {k} = {flags[k]}")
    print(f"   source_regime census: {census(src_regime)}")
    print(f"   end_regime census: {census(end_regime)}")
    for k in SCALAR_KEYS:
        a = arrays[k]
        finite = np.isfinite(a)
        nf = int((~finite).sum())
        nz = int(np.count_nonzero(a[finite])) if finite.any() else 0
        lo = np.min(a[finite]) if finite.any() else float("nan")
        hi = np.max(a[finite]) if finite.any() else float("nan")
        print(f"   {k}: n={a.size} nonfinite={nf} nonzero={nz} "
              f"min={lo:g} max={hi:g}")
    for k in CEILING_KEYS:
        a = arrays[k]
        finite = np.isfinite(a)
        lo = np.min(a[finite]) if finite.any() else float("nan")
        hi = np.max(a[finite]) if finite.any() else float("nan")
        print(f"   {k}: n={a.size} nonfinite={int((~finite).sum())} "
              f"min={lo:g} max={hi:g}")
    print(f"   diagnostics/clamped_to_dt_min: n={clamped.size} "
          f"nonzero={int(np.count_nonzero(clamped))}")
    print(f"   diagnostics/retry_count: n={retry.size} "
          f"nonzero={int(np.count_nonzero(retry))} max={retry.max():g}")
    print(f"   diagnostics/step_cap census: {census(step_cap)}")
    print(f"   diagnostics/active_constraint census: {census(active)}")
    print(f"   timestep_rejection_events: n={n_rej}")
    for name in ("capability_limited", "virtual_cathode"):
        idx = np.flatnonzero(src_regime == name)
        if idx.size:
            print(f"   {label:<12s} {name:<19s} n={idx.size:<6d} "
                  f"t_ms range [{t_ms[idx].min():.3f}, {t_ms[idx].max():.3f}]")
        else:
            print(f"   {label:<12s} {name:<19s} n=0")
    print()


def main(argv):
    if not argv:
        raise SystemExit(__doc__)
    print("L2 ARM 4 -- REGIME / STEP-CAP CENSUS "
          "(same fields as scripts/l2arm_regime_inventory.txt section C)")
    print()
    for item in argv:
        label, sep, path = item.partition("=")
        if not sep:
            raise SystemExit(f"expected LABEL=PATH, got {item!r}")
        report(label, path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
