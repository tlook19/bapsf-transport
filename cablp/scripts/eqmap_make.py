"""Produce an EQUILIBRATION MAP: nn(z,t) through a foot-fill 101st cycle.

The standard neutral equilibration runs its configured cycles (100 by default)
of ``tau_cycle`` each, every cycle opening the valve for
``equilibration_gas_puff_on_s``, and keeps only the final profile as the run's
neutral seed.  This script runs that equilibration and then ONE ADDITIONAL
cycle whose puff window is the FOOT FILL TIME, recording the whole nn(z,t) and
nn_a(z,t) trajectory of that extra cycle at a configurable cadence.

The result is a MAP of starting distributions parameterised by pre-fill time:
row ``k`` is the neutral fill a discharge would start from had it broken down
``t_s[k]`` after the valve opened.  ``scripts/eqmap_slice.py`` cuts a row out of
it and writes the shaped-initial-fill npz that ``run_m6_point.py
--nn0-profile-npz`` consumes, so the map's slices reach a run through the
EXISTING ``neutral_initial_profile`` capability and no other path.

SCRIPT-ONLY.  Nothing here is solver code and no solver code was changed for
it: the equilibration is driven through the public
``LAPDSim1D.run_neutral_equilibration``, and the 101st cycle is an ordinary
``Plasma=False`` inner sim whose neutral initial condition is planted through
the public ``neutral_initial_profile`` keys -- the same capability the map's
own slices are delivered by.

THE FOOT FILL TIME IS AN AXIS, NOT A CONSTANT.  ``--foot-s`` sets how far the
map extends; it is recorded in the header and every slice states the pre-fill
time it was cut at.  This script boxes no physical quantity.

Two consistency checks run on every build and are recorded in the header
(``--strict-checks`` turns a failure into a non-zero exit):

  (a) THE NULL.  Row 0 of the map must equal the standard equilibrated seed
      EXACTLY, in both zones -- a zero-length 101st cycle is the shipped
      convention, so the map must contain the shipped convention verbatim as
      its own t=0.
  (b) INVENTORY BOOKKEEPING.  Plasma-off, the ONLY term that can change the
      total neutral inventory is ``neutral_sources`` (puff minus pump);
      transport and zone exchange move gas without creating it.  So over every
      sample interval, d(total nn + nn_a)/dt must equal the volume-integrated
      ``neutral_sources``, and that term must decompose as the recorded puff
      influx minus a non-negative pump rate.  This is the A0 budget structure
      with the plasma channels absent.

Usage (production ES1 stance, two-zone, a 10 ms foot axis at 0.25 ms cadence)::

    python scripts/eqmap_make.py --es 1 --nx 240 --two-zone \
        --foot-s 10e-3 --cadence-s 0.25e-3 \
        --out scripts/eqmap_demo_es1_nx240.npz

Run from ``<checkout>/cablp`` with ``PYTHONPATH`` set to the same ``cablp/``.
"""

import argparse
import hashlib
import json
import sys
import time as _walltime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cablp.solvers._sim1d import LAPDSim1D, default_config  # noqa: E402
from cablp.solvers._sim1d.core.neutral_seed_cache import (  # noqa: E402
    neutral_seed_signature,
)
from cablp.solvers._sim1d.physics.neutrals import (  # noqa: E402
    neutral_zone_volumes,
)

# npz format tag. eqmap_slice.py refuses anything else.
MAP_FORMAT = "sim1d-eqmap-v1"


def stance_config(es, nx, sgp, two_zone, extra, extra_flag):
    """Return the (params, flags) the map is built at.

    Assembled by the SAME path a campaign run uses -- ``default_config()`` plus
    the ES benchmark overrides -- so the map's equilibrated base is the base
    that stance's runs would have equilibrated to for themselves.
    """
    params, flags = default_config()
    if es is not None:
        from compare_sim1d_es1 import FLAG_OVERRIDES, PARAM_OVERRIDES

        params.update(PARAM_OVERRIDES)
        flags.update(FLAG_OVERRIDES)
        # run_m6_point's own neutral-exchange stance.
        params["neutral_exchange_model"] = "knudsen"
    if nx is not None:
        params["nx"] = int(nx)
    if sgp is not None:
        params["S_gp"] = float(sgp)
    if two_zone:
        flags["neutral_two_zone"] = True
        params["neutral_exchange_model"] = "knudsen"
    params.update(extra)
    flags.update(extra_flag)
    return params, flags


def config_hash(params, flags):
    """Return a sha256 over the FULL resolved configuration."""
    blob = json.dumps(
        {
            "params": {k: repr(v) for k, v in sorted(params.items())},
            "flags": {k: repr(v) for k, v in sorted(flags.items())},
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def equilibrate(params, flags, cycles):
    """Run the standard neutral equilibration; return (sim, nn, nn_a, wall_s)."""
    sim = LAPDSim1D(dict(params), dict(flags))
    t0 = _walltime.time()
    result = sim.run_neutral_equilibration(cycles=cycles)
    wall = _walltime.time() - t0
    nn = np.asarray(result.nn[-1], dtype=float)
    saved_nn_a = getattr(result, "nn_a", None)
    nn_a = None if saved_nn_a is None else np.asarray(saved_nn_a[-1], dtype=float)
    return sim, nn, nn_a, wall, result


def foot_cycle(params, flags, nn, nn_a, foot_s, cadence_s, map_dt):
    """Run the 101st cycle and return its trajectory.

    The inner sim is built exactly as ``run_neutral_equilibration`` builds its
    own (plasma off, cathode off, no nested equilibration, no seed cache), with
    two deltas that ARE the instrument: its neutral initial condition is the
    equilibrated profile planted through ``neutral_initial_profile`` rather than
    a uniform scalar, and its per-cycle puff window is the foot fill time
    instead of the stance's ``equilibration_gas_puff_on_s``.
    """
    p, f = dict(params), dict(flags)
    # The equilibration inner-sim stance, verbatim from
    # LAPDSim1D.run_neutral_equilibration.
    f["Plasma"] = False
    f["cathode_coupling"] = False
    f["neutral_equilibration"] = False
    f["launch_plasma_after_equilibration"] = False
    f["use_cached_neutral_seed"] = False
    # ...and the two deltas that make this the 101st cycle rather than another
    # standard one. The scalar nn0 is superseded for BOTH zones (the solver
    # refuses an armed flag alongside an explicit scalar), so it is cleared.
    f["neutral_initial_profile"] = True
    p["nn0"] = None
    p["nn0_profile"] = np.asarray(nn, dtype=float).tolist()
    if nn_a is not None:
        p["nn0_annulus_profile"] = np.asarray(nn_a, dtype=float).tolist()
    p["equilibration_gas_puff_on_s"] = float(foot_s)
    p["cycles"] = 1
    p["dt_save"] = float(cadence_s)
    p["t_save_start"] = 0.0
    p["max_output_steps"] = 0

    sim = LAPDSim1D(p, f)
    # The map's t=0 is the equilibrated seed by construction; assert it before
    # spending the cycle rather than discovering it in check (a) afterwards.
    if not np.array_equal(sim.state.nn, np.asarray(nn, dtype=float)):
        raise ValueError(
            "the 101st cycle's initial nn is not the equilibrated seed; the "
            "neutral_initial_profile path did not plant the profile verbatim"
        )
    if nn_a is not None and not np.array_equal(
        sim.state.nn_a, np.asarray(nn_a, dtype=float)
    ):
        raise ValueError(
            "the 101st cycle's initial nn_a is not the equilibrated annulus seed"
        )
    result = sim.run(t_end=float(foot_s), dt=float(map_dt))
    return sim, result


def check_null(result, nn, nn_a):
    """(a) THE NULL: row 0 of the map IS the standard equilibrated seed."""
    row0 = np.asarray(result.nn, dtype=float)[0]
    ok = bool(np.array_equal(row0, np.asarray(nn, dtype=float)))
    detail = {"nn_exact": ok}
    if nn_a is not None:
        row0_a = np.asarray(result.nn_a, dtype=float)[0]
        ok_a = bool(np.array_equal(row0_a, np.asarray(nn_a, dtype=float)))
        detail["nn_a_exact"] = ok_a
        ok = ok and ok_a
    detail["pass"] = bool(ok)
    return detail


def check_inventory(sim, result, tol):
    """(b) INVENTORY BOOKKEEPING along the 101st cycle, plasma-off.

    Plasma off, ``neutral_sources`` (puff minus pump) is the only term that can
    change the TOTAL neutral inventory; ``neutral_exchange``,
    ``neutral_zone_exchange`` and the axial transport redistribute it.  So the
    identity under test on every sample interval is

        Delta(inventory) == integral of the volume-integrated neutral_sources

    with the integral taken by the trapezoid rule on the SAMPLE cadence -- so
    the residual reported here is a QUADRATURE error at the map's cadence, not
    a solver error, and it shrinks with the cadence rather than with dt.

    THE CLOSING EDGE IS EXCLUDED AND REPORTED SEPARATELY.  The puff is a square
    wave that shuts at exactly ``foot_s``, so the final sample's
    ``neutral_sources`` is pump-only while the interval before it was fuelled
    throughout.  A trapezoid across that discontinuity loses ~half the closing
    interval's fuel -- a property of quadrature on a square forcing, and the
    same effect ``audit_sim1d_equilibration_duty.py`` documents for the
    integrator.  Intervals with the puff gate on at BOTH endpoints carry the
    check; the closing interval is reported as its own number.
    """
    geometry = sim.geometry
    two_zone = getattr(result, "nn_a", None) is not None
    if two_zone:
        V_col, V_ann = neutral_zone_volumes(geometry)
    else:
        V_col = np.asarray(geometry.neutral_volume_cm3, dtype=float)
        V_ann = None

    t = np.asarray(result.time, dtype=float)
    nn = np.asarray(result.nn, dtype=float)
    inventory = nn @ V_col
    src = result.rhs_terms["neutral_sources"]
    source_rate = np.asarray(src["nn"], dtype=float) @ V_col
    if two_zone:
        nn_a = np.asarray(result.nn_a, dtype=float)
        inventory = inventory + nn_a @ V_ann
        source_rate = source_rate + np.asarray(src["nn_a"], dtype=float) @ V_ann

    puff = np.asarray(
        result.gas_puff_diagnostics["puff_particles_per_s"], dtype=float
    )
    gate = np.asarray(result.phase_gas_puff_enabled, dtype=float)
    # puff - pump == neutral_sources, so the implied pump is what the puff did
    # not deposit. It is a loss, so it must never be negative.
    implied_pump = puff - source_rate

    d_inventory = np.diff(inventory)
    integrated = 0.5 * (source_rate[:-1] + source_rate[1:]) * np.diff(t)
    on_interval = (gate[:-1] > 0.0) & (gate[1:] > 0.0)

    scale = np.maximum(np.abs(d_inventory), 1e-300)
    residual = np.abs(d_inventory - integrated) / scale
    interior = residual[on_interval]
    worst = float(np.max(interior)) if interior.size else float("nan")

    closing = np.flatnonzero(~on_interval)
    closing_residual = (
        [float(residual[i]) for i in closing] if closing.size else []
    )

    # The cumulative statement over the fuelled span, which is what a slice at
    # any pre-fill time actually integrates.
    if interior.size:
        last_on = int(np.flatnonzero(on_interval)[-1]) + 1
        cumulative = abs(
            float(inventory[last_on] - inventory[0])
            - float(integrated[:last_on].sum())
        ) / abs(float(inventory[last_on] - inventory[0]))
    else:
        cumulative = float("nan")

    detail = {
        "intervals_checked": int(interior.size),
        "worst_interval_rel": worst,
        "cumulative_rel_over_fuelled_span": float(cumulative),
        "tol": float(tol),
        "closing_edge_intervals": [int(i) for i in closing],
        "closing_edge_rel": closing_residual,
        "implied_pump_min_per_s": float(np.min(implied_pump)),
        "implied_pump_max_per_s": float(np.max(implied_pump)),
        "implied_pump_frac_of_puff_at_t0": (
            float(implied_pump[0] / puff[0]) if puff[0] > 0 else float("nan")
        ),
        "puff_particles_per_s_while_on": float(puff[0]),
        "inventory_t0": float(inventory[0]),
        "inventory_tend": float(inventory[-1]),
        "inventory_rel_rise": float(inventory[-1] / inventory[0] - 1.0),
        "pump_non_negative": bool(np.all(implied_pump >= 0.0)),
    }
    detail["pass"] = bool(
        interior.size > 0
        and worst < tol
        and cumulative < tol
        and detail["pump_non_negative"]
    )
    return detail


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--es", type=int, default=1,
                    help="ES benchmark stance to build at (default 1). "
                         "--es 0 uses bare default_config()")
    ap.add_argument("--nx", type=int, default=None)
    ap.add_argument("--sgp", type=float, default=None, help="override S_gp [sccm]")
    ap.add_argument("--two-zone", action="store_true",
                    help="neutral_two_zone: nn is the column density, nn_a the "
                         "annulus, and the map carries both")
    ap.add_argument("--cycles", type=int, default=None,
                    help="standard equilibration cycles before the 101st "
                         "(default: the stance's neutral_equilibration_cycles)")
    ap.add_argument("--foot-s", type=float, default=10e-3,
                    help="THE AXIS: the 101st cycle's puff window and the map's "
                         "time extent [s] (default 10e-3)")
    ap.add_argument("--cadence-s", type=float, default=0.25e-3,
                    help="map sampling cadence [s] (default 0.25e-3)")
    ap.add_argument("--map-dt", type=float, default=None,
                    help="fixed timestep of the 101st cycle [s] (default: "
                         "cadence/10). NOT the equilibration's own dt, which is "
                         "far coarser than a foot window")
    ap.add_argument("--tol", type=float, default=1e-3,
                    help="check (b) quadrature tolerance (default 1e-3)")
    ap.add_argument("--strict-checks", action="store_true",
                    help="exit non-zero if a consistency check fails "
                         "(default: record the failure in the header and exit 0)")
    ap.add_argument("--extra", nargs="*", default=(),
                    help="additional k=v input_dict overrides (JSON-parsed)")
    ap.add_argument("--extra-flag", nargs="*", default=(),
                    help="additional k=v input_flags overrides (JSON-parsed)")
    ap.add_argument("--out", required=True, help="output map .npz")
    args = ap.parse_args(argv)

    def parse_kv(items):
        out = {}
        for kv in items:
            k, v = kv.split("=", 1)
            try:
                out[k] = json.loads(v)
            except json.JSONDecodeError:
                out[k] = v
        return out

    if args.foot_s <= 0.0:
        ap.error("--foot-s must be positive")
    if args.cadence_s <= 0.0:
        ap.error("--cadence-s must be positive")
    if args.cadence_s > args.foot_s:
        ap.error("--cadence-s must not exceed --foot-s")
    map_dt = args.map_dt if args.map_dt is not None else args.cadence_s / 10.0
    if map_dt <= 0.0:
        ap.error("--map-dt must be positive")

    extra = parse_kv(args.extra)
    extra_flag = parse_kv(args.extra_flag)
    params, flags = stance_config(
        None if args.es == 0 else args.es,
        args.nx,
        args.sgp,
        args.two_zone,
        extra,
        extra_flag,
    )

    print(f"# eqmap: equilibrating (es={args.es} nx={params.get('nx')} "
          f"two_zone={flags.get('neutral_two_zone')})")
    sim, nn, nn_a, wall, eq_result = equilibrate(params, flags, args.cycles)
    p_eff, f_eff = sim.get_config()
    cycles = int(eq_result.neutral_equilibration_summary.cycles)
    print(f"# equilibrated: {cycles} cycles, wall={wall:.1f} s, "
          f"cells={nn.size}, mean_nn={nn.mean():.4e}")

    print(f"# eqmap: 101st cycle, puff on for foot_s={args.foot_s:g} s, "
          f"cadence={args.cadence_s:g} s, dt={map_dt:g} s")
    t1 = _walltime.time()
    foot_sim, result = foot_cycle(
        p_eff, f_eff, nn, nn_a, args.foot_s, args.cadence_s, map_dt
    )
    foot_wall = _walltime.time() - t1

    t_s = np.asarray(result.time, dtype=float)
    map_nn = np.asarray(result.nn, dtype=float)
    map_nn_a = (
        None if getattr(result, "nn_a", None) is None
        else np.asarray(result.nn_a, dtype=float)
    )
    print(f"# 101st cycle: {t_s.size} samples over "
          f"[{t_s[0]:g}, {t_s[-1]:g}] s, wall={foot_wall:.1f} s, "
          f"steps={result.steps}")

    null_check = check_null(result, nn, nn_a)
    budget_check = check_inventory(foot_sim, result, args.tol)

    geometry = foot_sim.geometry
    if map_nn_a is not None:
        V_col, V_ann = neutral_zone_volumes(geometry)
    else:
        V_col = np.asarray(geometry.neutral_volume_cm3, dtype=float)
        V_ann = np.zeros_like(V_col)

    ledger = {
        "format": MAP_FORMAT,
        "kind": "equilibration map: nn(z,t) through a foot-fill 101st cycle",
        "producer": "scripts/eqmap_make.py",
        # --- the stance the map was built at ---
        "es": None if args.es == 0 else int(args.es),
        "nx": int(p_eff["nx"]),
        "cells": int(geometry.cells),
        "two_zone": bool(f_eff.get("neutral_two_zone", False)),
        "neutral_exchange_model": p_eff.get("neutral_exchange_model"),
        # The build-time overrides, recorded so a consumer can rebuild the
        # SAME stance -- eqmap_slice.py's construction check replays them.
        "stance_extra": extra,
        "stance_extra_flag": extra_flag,
        "config_sha256": config_hash(p_eff, f_eff),
        "neutral_seed_signature": neutral_seed_signature(p_eff, f_eff),
        # --- the fuelling configuration ---
        "S_gp_sccm": float(p_eff["S_gp"]),
        "gas_puff_valves": int(p_eff.get("gas_puff_valves", 2)),
        "gas_puff_profile": p_eff.get("gas_puff_profile"),
        "gas_puff_z_cm": p_eff.get("gas_puff_z_cm"),
        "gas_puff_throw_cm": p_eff.get("gas_puff_throw_cm"),
        "gas_puff_mode": p_eff.get("gas_puff_mode"),
        "S_pump_L_Lps": p_eff.get("S_pump_L"),
        "S_pump_R_Lps": p_eff.get("S_pump_R"),
        # --- the standard equilibration that produced t=0 ---
        "equilibration_cycles": cycles,
        "equilibration_tau_cycle_s": float(p_eff.get("tau_cycle", 0.0)),
        "equilibration_puff_on_s": p_eff.get("equilibration_gas_puff_on_s"),
        "equilibration_dt_s": p_eff.get("neutral_equilibration_dt"),
        "equilibration_wall_s": round(wall, 2),
        "base_mean_density_cm3": float(nn.mean()),
        "base_column_min_cm3": float(nn.min()),
        "base_column_max_cm3": float(nn.max()),
        "base_annulus_min_cm3": None if nn_a is None else float(nn_a.min()),
        "base_annulus_max_cm3": None if nn_a is None else float(nn_a.max()),
        # --- the 101st cycle: the AXIS and its numerics ---
        "foot_s": float(args.foot_s),
        "foot_axis_note": (
            "the foot fill time is a DISCLOSED AXIS of this map, not a boxed "
            "constant: every slice states the pre-fill time it was cut at"
        ),
        "cadence_s": float(args.cadence_s),
        "map_dt_s": float(map_dt),
        "map_dt_note": (
            "the 101st cycle's fixed timestep, chosen to resolve the foot "
            "window; the standard equilibration's own dt "
            f"({p_eff.get('neutral_equilibration_dt')} s) is far coarser than "
            "the whole foot and cannot resolve it"
        ),
        "samples": int(t_s.size),
        "foot_cycle_steps": int(result.steps),
        "foot_cycle_wall_s": round(foot_wall, 2),
        "t_min_s": float(t_s[0]),
        "t_max_s": float(t_s[-1]),
        # --- the checks ---
        "check_a_null": null_check,
        "check_b_inventory": budget_check,
    }

    payload = {
        "format": MAP_FORMAT,
        "t_s": t_s,
        "nn": map_nn,
        "nn_base": nn,
        "z_cm": np.asarray(geometry.z_cm, dtype=float),
        "length_cm": np.asarray(geometry.length_cm, dtype=float),
        "cell_role": np.asarray(geometry.cell_role, dtype=str),
        "neutral_volume_cm3": np.asarray(
            geometry.neutral_volume_cm3, dtype=float
        ),
        "V_col_cm3": V_col,
        "V_ann_cm3": V_ann,
        "puff_particles_per_s": np.asarray(
            result.gas_puff_diagnostics["puff_particles_per_s"], dtype=float
        ),
        "phase_gas_puff_enabled": np.asarray(
            result.phase_gas_puff_enabled, dtype=float
        ),
        "provenance": json.dumps(ledger, sort_keys=True),
    }
    if map_nn_a is not None:
        payload["nn_a"] = map_nn_a
        payload["nn_a_base"] = nn_a
    np.savez(args.out, **payload)

    print("# --- check (a) NULL: map row 0 == standard equilibrated seed ---")
    print(f"#   {json.dumps(null_check, sort_keys=True)}")
    print("# --- check (b) INVENTORY: d(nn+nn_a)/dt == puff - pump ---")
    for key in (
        "intervals_checked", "worst_interval_rel",
        "cumulative_rel_over_fuelled_span", "tol", "pump_non_negative",
        "implied_pump_frac_of_puff_at_t0", "puff_particles_per_s_while_on",
        "inventory_t0", "inventory_tend", "inventory_rel_rise",
        "closing_edge_intervals", "closing_edge_rel", "pass",
    ):
        print(f"#   {key} = {budget_check[key]}")
    print(f"# wrote {args.out}")

    failed = [
        name for name, detail in (("a", null_check), ("b", budget_check))
        if not detail["pass"]
    ]
    if failed:
        print(f"# CHECK FAILURE: {', '.join(failed)}")
        if args.strict_checks:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
