"""Cut one pre-fill time out of an equilibration map into a shaped-nn0 npz.

``scripts/eqmap_make.py`` records nn(z,t) and nn_a(z,t) through a foot-fill
101st cycle -- a map of starting distributions parameterised by pre-fill time.
This script takes that map and a pre-fill time and writes the
shaped-initial-fill npz that the EXISTING ``neutral_initial_profile``
capability consumes::

    python scripts/eqmap_slice.py --map scripts/eqmap_demo_es1_nx240.npz \
        --prefill-s 4.5e-3 --out scripts/eqmap_nn0_es1_t4p5ms.npz

    python scripts/run_m6_point.py --es 1 --nx 240 --two-zone \
        --nn0-profile-npz scripts/eqmap_nn0_es1_t4p5ms.npz \
        --extra-flag neutral_equilibration=false \
        --sgp 5200 --save-h5 scripts/somerun.h5

No second initial-condition mechanism is built here.  The output is an ordinary
sp3 shaped-fill npz -- ``nn0_profile``, optionally ``nn0_annulus_profile``,
``z_cm``, ``provenance`` -- and it reaches the solver through
``run_m6_point.py --nn0-profile-npz`` like any other.

INTERPOLATION.  The slice is LINEAR IN TIME between the two bracketing map
samples, applied per cell and per zone.  A pre-fill time that lands exactly on
a recorded sample is returned as that sample VERBATIM -- no interpolation
arithmetic runs -- so slicing a map at ``t = 0`` reproduces the standard
equilibrated seed bit for bit.  Linear interpolation of positive samples is
positive, so the sp3 positivity validator cannot be tripped by the
interpolation itself.

INTERPOLATION ERROR.  For a linear interpolant on a cadence ``h`` the pointwise
error is bounded by ``h^2 max|d2nn/dt2| / 8``.  This script estimates that
bound from the map's OWN second differences around the requested time and
prints it, absolute and relative, per zone.  It is an estimate of the error the
slice carries relative to the trajectory the map was cut from -- not a solver
error bar.  Cut a finer-cadence map if it is too large for the use.

``--selfcheck`` (on by default) re-reads the written file exactly as
``run_m6_point.py`` does, builds a ``LAPDSim1D`` at the map's own stance, and
asserts the state's neutral fields ARE the written arrays -- so the file is
proved to pass the capability's own construction-time validators before it is
ever handed to a campaign run.

Run from ``<checkout>/cablp`` with ``PYTHONPATH`` set to the same ``cablp/``.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eqmap_make import MAP_FORMAT, stance_config  # noqa: E402


def load_map(path):
    """Load and validate an eqmap npz; return a dict of its arrays + header."""
    with np.load(path, allow_pickle=False) as data:
        keys = set(data.files)
        if "format" not in keys or str(data["format"]) != MAP_FORMAT:
            raise ValueError(
                f"{path} is not a {MAP_FORMAT} map (format="
                f"{str(data['format']) if 'format' in keys else 'absent'}); "
                "produce one with scripts/eqmap_make.py"
            )
        for required in ("t_s", "nn", "z_cm", "provenance"):
            if required not in keys:
                raise ValueError(f"{path} carries no '{required}' array")
        out = {
            "t_s": np.asarray(data["t_s"], dtype=float),
            "nn": np.asarray(data["nn"], dtype=float),
            "z_cm": np.asarray(data["z_cm"], dtype=float),
            "provenance": json.loads(str(data["provenance"])),
            "nn_a": (
                np.asarray(data["nn_a"], dtype=float)
                if "nn_a" in keys else None
            ),
        }
    t_s, nn = out["t_s"], out["nn"]
    if t_s.ndim != 1 or nn.ndim != 2 or nn.shape[0] != t_s.size:
        raise ValueError(
            f"{path}: nn must be (samples, cells) matching t_s; got "
            f"nn{nn.shape} against t_s{t_s.shape}"
        )
    if nn.shape[1] != out["z_cm"].size:
        raise ValueError(
            f"{path}: nn has {nn.shape[1]} cells but z_cm has "
            f"{out['z_cm'].size}"
        )
    if out["nn_a"] is not None and out["nn_a"].shape != nn.shape:
        raise ValueError(
            f"{path}: nn_a{out['nn_a'].shape} does not match nn{nn.shape}"
        )
    if np.any(np.diff(t_s) <= 0.0):
        raise ValueError(f"{path}: the time axis is not strictly increasing")
    return out


def interpolate(t_s, values, t, sample_tol):
    """Return (slice, mode, lo, hi, weight) for ``values`` at time ``t``.

    An exact sample hit (within ``sample_tol``) is returned VERBATIM; otherwise
    the two bracketing samples are blended linearly.
    """
    hit = int(np.argmin(np.abs(t_s - t)))
    if abs(t_s[hit] - t) <= sample_tol:
        return values[hit].copy(), "exact_sample", hit, hit, 0.0
    hi = int(np.searchsorted(t_s, t))
    lo = hi - 1
    span = t_s[hi] - t_s[lo]
    w = float((t - t_s[lo]) / span)
    return (
        (1.0 - w) * values[lo] + w * values[hi],
        "linear",
        lo,
        hi,
        w,
    )


def interpolation_error(t_s, values, lo, hi):
    """Estimate the linear-interpolation error bound h^2 max|f''| / 8.

    ``f''`` is estimated by the map's own second difference across the
    bracketing interval, using the widest stencil available around it.
    """
    n = t_s.size
    if n < 3 or lo == hi:
        return 0.0, 0.0
    i = max(1, min(n - 2, lo))
    h_prev = t_s[i] - t_s[i - 1]
    h_next = t_s[i + 1] - t_s[i]
    second = 2.0 * (
        (values[i + 1] - values[i]) / h_next
        - (values[i] - values[i - 1]) / h_prev
    ) / (h_prev + h_next)
    h = float(t_s[hi] - t_s[lo])
    bound = h * h * np.abs(second) / 8.0
    denom = np.maximum(np.abs(values[i]), 1e-300)
    return float(np.max(bound)), float(np.max(bound / denom))


def validate_profile(values, key, nn_floor):
    """Re-apply the sp3 array rules here, so a bad slice never reaches a run."""
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{key}: {int(np.count_nonzero(~np.isfinite(values)))} "
                         f"non-finite entries of {values.size}")
    if np.any(values <= 0.0):
        raise ValueError(
            f"{key}: every entry must be > 0 (the sp3 validator refuses a "
            f"non-positive initial density); got min {float(values.min()):.6g}"
        )
    if np.any(values < nn_floor):
        raise ValueError(
            f"{key}: {int(np.count_nonzero(values < nn_floor))} entries lie "
            f"below nn_floor={nn_floor:g} and would be silently clipped up to "
            "it on the first state read rather than starting where the map put "
            "them"
        )


def selfcheck(out_path, header):
    """(c) The written file passes the sp3 capability's own validators.

    Re-reads the npz through the SAME lines ``run_m6_point.py`` uses, builds a
    LAPDSim1D at the map's stance with the shaped-fill flag armed, and asserts
    the constructed state's neutral fields ARE the written arrays.  Construction
    is where every sp3 validator lives, so a build that returns at all has
    passed them.
    """
    from cablp.solvers._sim1d import LAPDSim1D

    # run_m6_point.py:123-141, verbatim in substance.
    with np.load(out_path, allow_pickle=False) as data:
        if "nn0_profile" not in data:
            raise AssertionError(f"{out_path} carries no 'nn0_profile' array")
        nn0_profile = np.asarray(data["nn0_profile"], dtype=float).tolist()
        nn0_annulus_profile = (
            np.asarray(data["nn0_annulus_profile"], dtype=float).tolist()
            if "nn0_annulus_profile" in data else None
        )

    params, flags = stance_config(
        header.get("es"),
        header.get("nx"),
        header.get("S_gp_sccm"),
        bool(header.get("two_zone", False)),
        dict(header.get("stance_extra") or {}),
        dict(header.get("stance_extra_flag") or {}),
    )
    params["nn0_profile"] = nn0_profile
    if nn0_annulus_profile is not None:
        params["nn0_annulus_profile"] = nn0_annulus_profile
    params["nn0"] = None
    flags["neutral_initial_profile"] = True
    # The stance ships neutral_equilibration ON and the solver REFUSES it with
    # a shaped IC -- the same delta a shaped campaign arm states on its command
    # line (--extra-flag neutral_equilibration=false).
    flags["neutral_equilibration"] = False

    sim = LAPDSim1D(params, flags)
    ok_nn = bool(np.array_equal(sim.state.nn, np.asarray(nn0_profile)))
    detail = {"cells": int(sim.geometry.cells), "nn_exact": ok_nn}
    if nn0_annulus_profile is not None:
        ok_a = bool(
            np.array_equal(sim.state.nn_a, np.asarray(nn0_annulus_profile))
        )
        detail["nn_a_exact"] = ok_a
        ok_nn = ok_nn and ok_a
    # The floor is applied on every state read; prove it never bound here.
    detail["min_nn"] = float(np.min(sim.state.nn))
    detail["nn_floor"] = float(sim._floors["nn"])
    detail["floor_inert"] = bool(detail["min_nn"] > detail["nn_floor"])
    detail["pass"] = bool(ok_nn and detail["floor_inert"])
    return detail


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--map", required=True, help="map npz from eqmap_make.py")
    ap.add_argument("--prefill-s", type=float, default=None,
                    help="the pre-fill time to cut [s]")
    ap.add_argument("--out", default=None, help="output shaped-nn0 npz")
    ap.add_argument("--list", action="store_true",
                    help="print the map's header and time axis, then exit")
    ap.add_argument("--sample-tol", type=float, default=1e-12,
                    help="a pre-fill time within this of a recorded sample is "
                         "taken VERBATIM rather than interpolated (default 1e-12 s)")
    ap.add_argument("--no-selfcheck", action="store_true",
                    help="skip the construction round-trip check (c)")
    args = ap.parse_args(argv)

    m = load_map(args.map)
    header = m["provenance"]

    if args.list:
        print(json.dumps(header, indent=1, sort_keys=True))
        print(f"time axis [s]: {m['t_s']}")
        return 0
    if args.prefill_s is None or args.out is None:
        ap.error("--prefill-s and --out are required unless --list is given")

    t_s = m["t_s"]
    t = float(args.prefill_s)
    if t < t_s[0] - args.sample_tol or t > t_s[-1] + args.sample_tol:
        raise ValueError(
            f"--prefill-s {t:g} lies outside the map's axis "
            f"[{t_s[0]:g}, {t_s[-1]:g}] s. The map's foot axis is "
            f"foot_s={header.get('foot_s')} s; extrapolation is refused -- "
            "rebuild the map with a longer --foot-s"
        )

    nn_slice, mode, lo, hi, w = interpolate(t_s, m["nn"], t, args.sample_tol)
    err_abs, err_rel = interpolation_error(t_s, m["nn"], lo, hi)
    nn_floor = 1e8
    validate_profile(nn_slice, "nn0_profile", nn_floor)

    nn_a_slice = None
    err_a = (0.0, 0.0)
    if m["nn_a"] is not None:
        nn_a_slice, _, _, _, _ = interpolate(t_s, m["nn_a"], t, args.sample_tol)
        err_a = interpolation_error(t_s, m["nn_a"], lo, hi)
        validate_profile(nn_a_slice, "nn0_annulus_profile", nn_floor)

    cadence = float(header.get("cadence_s", np.min(np.diff(t_s))))
    ledger = {
        "kind": "shaped initial neutral fill, sliced from an equilibration map",
        "producer": "scripts/eqmap_slice.py",
        "map_file": str(Path(args.map).resolve()),
        "map_format": MAP_FORMAT,
        "map_config_sha256": header.get("config_sha256"),
        "map_neutral_seed_signature": header.get("neutral_seed_signature"),
        "prefill_s": t,
        "prefill_axis_note": (
            "THE PRE-FILL TIME IS THE DISCLOSED AXIS this slice was cut at: "
            "the fill a discharge would start from had it broken down this "
            "long after the valve opened. It is not a boxed constant"
        ),
        "slice_mode": mode,
        "bracket_samples": [int(lo), int(hi)],
        "bracket_times_s": [float(t_s[lo]), float(t_s[hi])],
        "bracket_weight": float(w),
        "interpolation": (
            "exact recorded sample, copied verbatim" if mode == "exact_sample"
            else "linear in time, per cell and per zone, between the two "
                 "bracketing map samples"
        ),
        "interp_error_bound_note": (
            "h^2 max|d2nn/dt2| / 8, with the second derivative estimated from "
            "the map's own samples around the bracket; the error of this slice "
            "relative to the trajectory the map was cut from, not a solver "
            "error bar"
        ),
        "interp_error_abs_cm3": err_abs,
        "interp_error_rel": err_rel,
        "interp_error_annulus_abs_cm3": err_a[0],
        "interp_error_annulus_rel": err_a[1],
        "map_cadence_s": cadence,
        "cells": int(nn_slice.size),
        "nx": header.get("nx"),
        "es": header.get("es"),
        "two_zone": bool(header.get("two_zone", False)),
        "S_gp_sccm": header.get("S_gp_sccm"),
        "foot_s": header.get("foot_s"),
        "equilibration_cycles": header.get("equilibration_cycles"),
        "equilibration_puff_on_s": header.get("equilibration_puff_on_s"),
        "equilibration_tau_cycle_s": header.get("equilibration_tau_cycle_s"),
        "base_is_map_t0": (
            "the map's t=0 row is the standard equilibrated seed; a slice at "
            "t=0 reproduces it verbatim"
        ),
        "column_min_cm3": float(nn_slice.min()),
        "column_max_cm3": float(nn_slice.max()),
        "column_mean_cm3": float(nn_slice.mean()),
        "annulus_min_cm3": None if nn_a_slice is None else float(nn_a_slice.min()),
        "annulus_max_cm3": None if nn_a_slice is None else float(nn_a_slice.max()),
        "run_with": (
            "run_m6_point.py --nn0-profile-npz <this file> "
            "--extra-flag neutral_equilibration=false"
            + (" --two-zone" if nn_a_slice is not None else "")
        ),
    }

    payload = {
        "nn0_profile": nn_slice,
        "z_cm": m["z_cm"],
        "provenance": json.dumps(ledger, sort_keys=True),
    }
    if nn_a_slice is not None:
        payload["nn0_annulus_profile"] = nn_a_slice
    np.savez(args.out, **payload)

    print(f"# sliced {args.map} at prefill_s={t:g} ({mode})")
    print(f"#   bracket samples {lo}..{hi} at t={t_s[lo]:g}..{t_s[hi]:g} s, "
          f"weight {w:.6g}")
    print(f"#   column  min/mean/max = {nn_slice.min():.6e} / "
          f"{nn_slice.mean():.6e} / {nn_slice.max():.6e} cm^-3")
    if nn_a_slice is not None:
        print(f"#   annulus min/mean/max = {nn_a_slice.min():.6e} / "
              f"{nn_a_slice.mean():.6e} / {nn_a_slice.max():.6e} cm^-3")
    print(f"#   interpolation error bound: {err_abs:.6e} cm^-3 "
          f"({err_rel:.6e} relative)")
    if nn_a_slice is not None:
        print(f"#   annulus error bound:       {err_a[0]:.6e} cm^-3 "
              f"({err_a[1]:.6e} relative)")
    print(f"# wrote {args.out}")

    if not args.no_selfcheck:
        detail = selfcheck(args.out, header)
        print("# --- check (c) the slice passes the sp3 validators ---")
        print(f"#   {json.dumps(detail, sort_keys=True)}")
        if not detail["pass"]:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
