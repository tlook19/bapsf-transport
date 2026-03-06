"""
Grid parameter sweep runner for LAPDSim.

Usage
-----
::

    from cablp.analysis import grid_sweep

    grid_sweep(
        param_ranges={"Vd": [100, 200], "Id": [2500, 5000]},
        flag_ranges={"eperp": [True, False]},
        fixed_params={"gas_type": "He", "cells": 3},
        fixed_flags={"Plasma": True, "icool": True},
        db_path="sweep.h5",
        t_window=(10.0, 20.0),
    )
"""
import itertools
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

from ..solvers._sim3 import LAPDSim, input_dict_template, input_flags_template
from .database import open_db, save_run, update_index, list_runs
from .stats import compute_window_stats


_EQUIL_DT_MAX = 1e-2  # fixed max step for neutral equilibration (never inherits main sim value)


def _equil_cache_key(params, flags):
    """
    Return a hashable key summarising the parameters that drive neutral-density
    equilibration.  Runs that share this key can reuse the same ``nn_eq`` result.

    The key covers: S_gp, S_pump_L/R, Source_nn0, cells, gas_type, Lm, Lp,
    TwinCathode flag, and Twin_S_gp (when TwinCathode is active).
    Parameters that only affect plasma dynamics (Vd, Id, Bz0, …) are excluded.
    """
    twin_active = bool(flags.get("TwinCathode", False))
    return (
        float(params.get("S_gp", 0.0)),
        float(params.get("S_pump_L", 0.0)),
        float(params.get("S_pump_R", 0.0)),
        float(params.get("Source_nn0", 0.0)),
        int(params.get("cells", 3)),
        str(params.get("gas_type", "He")),
        float(params.get("Lm", 1800.0)),
        float(params.get("Lp", 1800.0)),
        twin_active,
        float(params.get("Twin_S_gp", 0.0)) if twin_active else 0.0,
    )


def equilibrate_neutrals(
    params,
    flags=None,
    cycles=100,
    t_per_cycle=3.0,
    nn0_init=1e8,
    verbose=True,
):
    """
    Run a plasma-off simulation to find the equilibrium background neutral density.

    Starts all cells at ``nn0_init`` and advances ``cycles`` cycles of
    ``t_per_cycle`` seconds each with plasma off and adaptive time stepping
    (maximum step 1e-2 s).  The neutral density at the end of the final
    cycle is returned and can be used as the initial condition for a
    plasma-on simulation.

    Gas puff (``S_gp``) and pumping (``S_pump_L/R``) rates are taken from
    ``params`` so the result is consistent with the cathode configuration.

    Parameters
    ----------
    params : dict
        Simulation parameters.  ``S_gp``, ``S_pump_L``, ``S_pump_R``,
        ``cells``, ``gas_type``, and cathode geometry keys are used.
    flags : dict or None
        Simulation flags.  ``Plasma`` is forced to ``False``;
        ``Velocity`` is forced to ``False``; ``adaptive`` is forced to
        ``True``.
    cycles : int
        Number of gas-fill cycles to run.  Default 100.
    t_per_cycle : float
        Duration of each cycle in seconds.  Default 3.0.
    nn0_init : float
        Seed neutral density (cm⁻³) for all cells.  Default 1e8.
    verbose : bool
        Print start/end summary lines.

    Returns
    -------
    nn_eq : np.ndarray, shape (cells,)
        Equilibrium neutral density in each cell at the end of the final cycle.
    """
    eq_params = {
        **input_dict_template,
        **params,
        "nn0": nn0_init,
        # Source_nn0 is kept from params (user-controlled; used to emulate gas puff profile)
        # S_gp is active only during the discharge phase (first 20 ms of each cycle)
        "cycles": cycles,
        "end": t_per_cycle,
        "d_off": 20e-3,     # gas puff on for first 20 ms, then neutral diffusion/pumping
        # Always use the fixed equilibration step size, never the main sim's dt_main/dt_after
        "dt_main": _EQUIL_DT_MAX,
        "dt_after": _EQUIL_DT_MAX,
    }
    # Twin_nn0 is kept from params (user-controlled, same rationale as Source_nn0)

    eq_flags = {
        **input_flags_template,
        **(flags or {}),
        "Plasma": False,    # no ionisation / recombination
        "Velocity": False,  # velocity meaningless without plasma
        "adaptive": True,   # adaptive stepping for efficiency
    }

    if verbose:
        print(
            f"  [nn equil] {cycles} cycles × {t_per_cycle} s  "
            f"(dt_max={_EQUIL_DT_MAX} s, nn0_init={nn0_init:.1e})"
        )

    sim = LAPDSim(eq_params, eq_flags)
    sim.start_simulation()

    nn_eq = sim.get_results()["nn"][-1]  # shape (cells,)

    if verbose:
        print(f"  [nn equil] nn_eq = {nn_eq} cm⁻³")

    return nn_eq


def _apply_equilibrated_nn(params, flags, nn_eq):
    """
    Return a copy of ``params`` with ``nn0`` set from the equilibrated neutral
    density array.

    ``nn0`` is set to the mean of the interior cells so it remains a scalar.
    ``Source_nn0`` and ``Twin_nn0`` are left unchanged — both are user-controlled
    initial conditions analogous to a gas puff profile.
    """
    n_cells = int(params.get("cells", 3))
    patched = dict(params)
    # nn0 sets the background for all non-overridden cells
    if n_cells > 2:
        patched["nn0"] = float(nn_eq[1:-1].mean())
    else:
        patched["nn0"] = float(nn_eq.mean())
    # Source_nn0 and Twin_nn0 are left as-is — the user sets them manually
    return patched


def param_combinations(param_ranges, flag_ranges):
    """
    Generate all (params_dict, flags_dict) combinations.

    Parameters
    ----------
    param_ranges : dict
        ``{param_name: [val1, val2, ...]}``.
    flag_ranges : dict
        ``{flag_name: [True, False, ...]}``.

    Returns
    -------
    list of (params_patch, flags_patch)
        Each element is a pair of dicts containing only the varied keys.
        Keys are sorted for reproducibility.
    """
    param_keys = sorted(param_ranges.keys())
    flag_keys = sorted(flag_ranges.keys())

    param_vals = [param_ranges[k] for k in param_keys]
    flag_vals = [flag_ranges[k] for k in flag_keys]

    all_vals = param_vals + flag_vals
    all_keys = param_keys + flag_keys

    if not all_keys:
        return [({}, {})]

    combos = []
    for combo in itertools.product(*all_vals):
        combined = dict(zip(all_keys, combo))
        p_patch = {k: combined[k] for k in param_keys}
        f_patch = {k: combined[k] for k in flag_keys}
        combos.append((p_patch, f_patch))

    return combos


def grid_sweep(
    param_ranges,
    flag_ranges=None,
    fixed_params=None,
    fixed_flags=None,
    db_path="sweep.h5",
    t_window=(10.0, 20.0),
    param_aliases=None,
    param_transforms=None,
    equilibrate_nn=False,
    verbose=True,
    verbose_equil=None,
):
    """
    Run all combinations of ``param_ranges × flag_ranges`` and save to an HDF5 database.

    Parameters
    ----------
    param_ranges : dict
        ``{param_name: [val1, val2, ...]}``.  Keys must match ``input_dict_template``.
    flag_ranges : dict or None
        ``{flag_name: [True, False, ...]}``.  Keys must match ``input_flags_template``.
    fixed_params : dict or None
        Parameters held constant (merged over ``input_dict_template``).
    fixed_flags : dict or None
        Flags held constant (merged over ``input_flags_template``).
    db_path : str or path-like
        Path to the HDF5 database.  Created if it does not exist.
    t_window : tuple of float
        (t_start, t_end) in ms for window statistics.
    param_aliases : dict or None
        ``{alias_key: source_key}`` pairs applied **after** building each run's
        ``params`` dict.  E.g. ``{"Twin_Vd": "Vd"}`` ensures ``Twin_Vd`` always
        equals the current ``Vd`` value, even when ``Vd`` is swept.
    param_transforms : callable or None
        ``(params, flags) -> params`` applied after ``param_aliases``.  Used to
        derive computed parameters, e.g. ``Id = P_in / Vd``.  The callable
        may modify ``params`` in-place and must return the updated dict.
    equilibrate_nn : bool
        If ``True``, run a 100-cycle plasma-off pre-simulation before each
        run to find the equilibrium neutral density.  Results are cached by
        neutral-dynamics key (S_gp, pumping, cells, gas_type, TwinCathode,
        Twin_S_gp) and reused for runs that share the same neutral equilibrium.
    verbose : bool
        Print progress messages including per-run timing.
    verbose_equil : bool or None
        Print equilibration detail messages.  ``None`` (default) inherits
        from ``verbose``.  Set to ``False`` to suppress equilibration inner
        prints while keeping sweep progress prints.

    Returns
    -------
    list of str
        Run IDs that completed successfully.

    Notes
    -----
    - Run IDs are assigned sequentially as ``run_0000``, ``run_0001``, …
    - Runs already present in the database are skipped, so an interrupted sweep
      can be resumed by calling this function again with the same arguments.
    - If a simulation raises an exception the run is marked ``'failed'`` in the
      index and execution continues with the next combination.
    """
    if flag_ranges is None:
        flag_ranges = {}

    _verbose_equil = verbose if verbose_equil is None else verbose_equil

    combos = param_combinations(param_ranges, flag_ranges)
    n_total = len(combos)

    if verbose:
        print(f"Grid sweep: {n_total} combinations → '{db_path}'")

    _nn_cache = {}  # cache_key → (nn_eq, equil_time_s)
    t_sweep_start = time.time()

    with open_db(db_path, mode="a") as db:
        existing = set(list_runs(db))
        successful = []

        for i, (p_patch, f_patch) in enumerate(combos):
            run_id = f"run_{i:04d}"

            if run_id in existing:
                if verbose:
                    print(f"  [{i+1}/{n_total}] {run_id} already in database — skipping.")
                successful.append(run_id)
                continue

            # Build full params and flags dicts
            params = {**input_dict_template, **(fixed_params or {}), **p_patch}
            flags = {**input_flags_template, **(fixed_flags or {}), **f_patch}

            # Apply param aliases (e.g. symmetric twin mirroring: Twin_Vd = Vd)
            if param_aliases:
                for alias, source in param_aliases.items():
                    if source in params:
                        params[alias] = params[source]

            # Apply param transforms (derive computed params, e.g. Id = P_in / Vd)
            if param_transforms is not None:
                params = param_transforms(params, flags)

            if verbose:
                varied = {**p_patch, **f_patch}
                print(f"  [{i+1}/{n_total}] {run_id}  {varied}")

            equil_time = 0.0
            cache_hit = False

            # Pre-equilibrate neutral density if requested
            if equilibrate_nn:
                twin_active = bool(flags.get("TwinCathode", False))
                twin_str = (
                    f"  Twin_S_gp={params.get('Twin_S_gp', 0):.0f}" if twin_active else ""
                )
                cache_key = _equil_cache_key(params, flags)
                if cache_key in _nn_cache:
                    nn_eq, equil_time = _nn_cache[cache_key]
                    cache_hit = True
                    if verbose:
                        print(
                            f"    [nn equil] cache hit"
                            f"  S_gp={params.get('S_gp', 0):.0f}"
                            f"  twin={'on' if twin_active else 'off'}{twin_str}"
                        )
                else:
                    t0_equil = time.time()
                    nn_eq = equilibrate_neutrals(params, flags, verbose=_verbose_equil)
                    equil_time = time.time() - t0_equil
                    _nn_cache[cache_key] = (nn_eq, equil_time)
                    if verbose:
                        print(
                            f"    [nn equil] computed"
                            f"  S_gp={params.get('S_gp', 0):.0f}"
                            f"  twin={'on' if twin_active else 'off'}{twin_str}"
                            f"  time={equil_time:.1f}s"
                        )
                params = _apply_equilibrated_nn(params, flags, nn_eq)

            t0_run = time.time()
            try:
                sim = LAPDSim(params, flags)
                sim.start_simulation()
                results = sim.get_results()
                run_time = time.time() - t0_run
                stats = compute_window_stats(results, t_window)
                n_cells = int(params.get("cells", 3))

                save_run(db, run_id, params, flags, results, stats)
                update_index(db, run_id, params, flags, stats, n_cells, status="ok")
                successful.append(run_id)

                if verbose:
                    twin_active = bool(flags.get("TwinCathode", False))
                    equil_str = ""
                    if equilibrate_nn:
                        twin_str = (
                            f"  Twin_S_gp={params.get('Twin_S_gp', 0):.0f}" if twin_active else ""
                        )
                        cache_str = "cached" if cache_hit else f"{equil_time:.1f}s"
                        equil_str = (
                            f"  equil={cache_str}"
                            f"  S_gp={params.get('S_gp', 0):.0f}"
                            f"  twin={'on' if twin_active else 'off'}{twin_str}"
                        )
                    print(
                        f"    ne_var={stats['ne_var']:.3e}  Te_var={stats['Te_var']:.3e}"
                        f"  ne_mean={stats['ne_mean']:.3e}  Te_mean={stats['Te_mean']:.3f} eV"
                        f"  run={run_time:.1f}s{equil_str}"
                    )

            except Exception:
                run_time = time.time() - t0_run
                tb = traceback.format_exc()
                print(f"  [{i+1}/{n_total}] {run_id} FAILED (run={run_time:.1f}s):\n{tb}")

                # Save a minimal failure record so the index stays consistent
                with open_db(db_path, mode="a") as db2:
                    grp = db2.require_group("runs").require_group(run_id)
                    grp.attrs["status"] = "failed"
                    grp.attrs["error"] = tb[:2000]  # truncate for storage
                    for k, v in params.items():
                        try:
                            grp.attrs[f"param_{k}"] = v
                        except TypeError:
                            grp.attrs[f"param_{k}"] = str(v)
                    for k, v in flags.items():
                        grp.attrs[f"flag_{k}"] = bool(v)
                    update_index(db2, run_id, params, flags, {}, 0, status="failed")

    total_time = time.time() - t_sweep_start
    if verbose:
        print(f"Sweep complete: {len(successful)}/{n_total} runs succeeded.  Total: {total_time:.1f}s")

    return successful


# ── Parallel sweep ─────────────────────────────────────────────────────────────

def _run_single_worker(args):
    """
    Module-level worker function for ProcessPoolExecutor (must be picklable).

    Parameters
    ----------
    args : tuple of (run_id, params, flags)

    Returns
    -------
    tuple of (run_id, params, flags, results, run_time_s)
    """
    run_id, params, flags = args
    t0 = time.time()
    sim = LAPDSim(params, flags)
    sim.start_simulation()
    results = sim.get_results()
    run_time = time.time() - t0
    return run_id, params, flags, results, run_time


def grid_sweep_parallel(
    param_ranges,
    flag_ranges=None,
    fixed_params=None,
    fixed_flags=None,
    db_path="sweep.h5",
    t_window=(10.0, 20.0),
    n_workers=1,
    progress_callback=None,
    param_aliases=None,
    param_transforms=None,
    equilibrate_nn=False,
    verbose=True,
    verbose_equil=None,
    stop_event=None,
):
    """
    Run all combinations of ``param_ranges × flag_ranges`` in parallel and save to HDF5.

    Simulations are executed in a ``ProcessPoolExecutor`` with ``n_workers`` workers.
    HDF5 writes are performed on the calling thread to avoid concurrent write conflicts.

    Parameters
    ----------
    param_ranges : dict
        ``{param_name: [val1, val2, ...]}``.
    flag_ranges : dict or None
        ``{flag_name: [True, False, ...]}``.
    fixed_params : dict or None
        Parameters held constant.
    fixed_flags : dict or None
        Flags held constant.
    db_path : str or path-like
        Path to the HDF5 database.
    t_window : tuple of float
        (t_start, t_end) in ms for window statistics.
    n_workers : int
        Number of parallel worker processes.
    progress_callback : callable or None
        Called after each run completes:
        ``progress_callback(i, total, run_id, status, stats)``
        where ``stats`` is the window-stats dict (empty on failure) augmented
        with internal timing keys (``_run_time_s``, ``_equil_time_s``,
        ``_equil_cache_hit``, ``_equil_S_gp``, ``_equil_twin``,
        ``_equil_Twin_S_gp``, ``_equilibrate_nn``).  These ``_``-prefixed
        keys are **not** stored in the HDF5 database.
    param_aliases : dict or None
        ``{alias_key: source_key}`` pairs applied after building each run's
        params dict.  E.g. ``{"Twin_Vd": "Vd"}`` ensures symmetric twin mode
        tracks the primary even when ``Vd`` is swept.
    param_transforms : callable or None
        ``(params, flags) -> params`` applied after ``param_aliases``.  Used to
        derive computed parameters, e.g. ``Id = P_in / Vd``.  Applied before
        dispatching to workers, so workers always receive fully-resolved params.
    equilibrate_nn : bool
        If ``True``, run a 100-cycle plasma-off pre-simulation for each
        combination (serially, before dispatch) to find the equilibrium
        neutral density.  Results are cached by neutral-dynamics key so that
        runs sharing the same S_gp / pumping / cells / TwinCathode config
        only equilibrate once.
    verbose : bool
        Print progress messages.
    verbose_equil : bool or None
        Print equilibration detail messages.  ``None`` (default) inherits
        from ``verbose``.  Set to ``False`` to suppress equilibration inner
        prints while keeping sweep progress prints.

    Returns
    -------
    list of str
        Run IDs that completed successfully.
    """
    if flag_ranges is None:
        flag_ranges = {}

    _verbose_equil = verbose if verbose_equil is None else verbose_equil

    combos = param_combinations(param_ranges, flag_ranges)
    n_total = len(combos)

    if verbose:
        print(f"Parallel sweep ({n_workers} workers): {n_total} combinations → '{db_path}'")

    with open_db(db_path, mode="a") as db:
        existing = set(list_runs(db))

    _nn_cache = {}  # cache_key → (nn_eq, equil_time_s)

    # Build pending list (skip already-done runs; pre-equilibrate if requested)
    pending = []
    pending_equil_info = []  # parallel list of equil metadata per pending run
    successful = []
    for i, (p_patch, f_patch) in enumerate(combos):
        run_id = f"run_{i:04d}"
        if run_id in existing:
            if verbose:
                print(f"  {run_id} already in database — skipping.")
            successful.append(run_id)
            continue
        params = {**input_dict_template, **(fixed_params or {}), **p_patch}
        flags = {**input_flags_template, **(fixed_flags or {}), **f_patch}

        # Apply param aliases (e.g. symmetric twin mirroring: Twin_Vd = Vd)
        if param_aliases:
            for alias, source in param_aliases.items():
                if source in params:
                    params[alias] = params[source]

        # Apply param transforms (derive computed params, e.g. Id = P_in / Vd)
        if param_transforms is not None:
            params = param_transforms(params, flags)

        equil_time = 0.0
        cache_hit = False
        twin_active = bool(flags.get("TwinCathode", False))

        if equilibrate_nn:
            twin_str = (
                f"  Twin_S_gp={params.get('Twin_S_gp', 0):.0f}" if twin_active else ""
            )
            cache_key = _equil_cache_key(params, flags)
            if cache_key in _nn_cache:
                nn_eq, equil_time = _nn_cache[cache_key]
                cache_hit = True
                if verbose:
                    print(
                        f"  [{i+1}/{n_total}] {run_id}: nn equil cache hit"
                        f"  S_gp={params.get('S_gp', 0):.0f}"
                        f"  twin={'on' if twin_active else 'off'}{twin_str}"
                    )
            else:
                if verbose:
                    print(f"  [{i+1}/{n_total}] {run_id}: equilibrating nn0 …")
                t0_equil = time.time()
                nn_eq = equilibrate_neutrals(params, flags, verbose=_verbose_equil)
                equil_time = time.time() - t0_equil
                _nn_cache[cache_key] = (nn_eq, equil_time)
                if verbose:
                    print(
                        f"    [nn equil] done"
                        f"  S_gp={params.get('S_gp', 0):.0f}"
                        f"  twin={'on' if twin_active else 'off'}{twin_str}"
                        f"  time={equil_time:.1f}s"
                    )
            params = _apply_equilibrated_nn(params, flags, nn_eq)

        pending.append((run_id, params, flags))
        pending_equil_info.append({
            "equil_time_s": equil_time,
            "cache_hit": cache_hit,
            "S_gp": float(params.get("S_gp", 0.0)),
            "twin": twin_active,
            "Twin_S_gp": float(params.get("Twin_S_gp", 0.0)) if twin_active else 0.0,
        })

    if not pending:
        if verbose:
            print("All runs already complete.")
        return successful

    completed_count = len(successful)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_run = {}
        for (run_id, params, flags), equil_info in zip(pending, pending_equil_info):
            fut = executor.submit(_run_single_worker, (run_id, params, flags))
            future_to_run[fut] = (run_id, params, flags, equil_info)

        for future in as_completed(future_to_run):
            run_id, params, flags, equil_info = future_to_run[future]
            completed_count += 1

            try:
                _, _, _, results, run_time = future.result()
                stats = compute_window_stats(results, t_window)
                n_cells = int(params.get("cells", 3))

                with open_db(db_path, mode="a") as db:
                    save_run(db, run_id, params, flags, results, stats)
                    update_index(db, run_id, params, flags, stats, n_cells, status="ok")

                successful.append(run_id)

                if verbose:
                    twin_str = (
                        f"  Twin_S_gp={equil_info['Twin_S_gp']:.0f}"
                        if equil_info["twin"] else ""
                    )
                    equil_str = ""
                    if equilibrate_nn:
                        cache_str = "cached" if equil_info["cache_hit"] else f"{equil_info['equil_time_s']:.1f}s"
                        equil_str = (
                            f"  equil={cache_str}"
                            f"  S_gp={equil_info['S_gp']:.0f}"
                            f"  twin={'on' if equil_info['twin'] else 'off'}{twin_str}"
                        )
                    print(
                        f"  [{completed_count}/{n_total}] {run_id} ok — "
                        f"ne_var={stats['ne_var']:.3e}  run={run_time:.1f}s{equil_str}"
                    )

                if progress_callback is not None:
                    callback_stats = {
                        **stats,
                        "_run_time_s": run_time,
                        "_equil_time_s": equil_info["equil_time_s"],
                        "_equil_cache_hit": equil_info["cache_hit"],
                        "_equil_S_gp": equil_info["S_gp"],
                        "_equil_twin": equil_info["twin"],
                        "_equil_Twin_S_gp": equil_info["Twin_S_gp"],
                        "_equilibrate_nn": equilibrate_nn,
                    }
                    progress_callback(completed_count, n_total, run_id, "ok", callback_stats)

            except Exception:
                tb = traceback.format_exc()
                print(f"  [{completed_count}/{n_total}] {run_id} FAILED:\n{tb}")

                with open_db(db_path, mode="a") as db2:
                    grp = db2.require_group("runs").require_group(run_id)
                    grp.attrs["status"] = "failed"
                    grp.attrs["error"] = tb[:2000]
                    for k, v in params.items():
                        try:
                            grp.attrs[f"param_{k}"] = v
                        except TypeError:
                            grp.attrs[f"param_{k}"] = str(v)
                    for k, v in flags.items():
                        grp.attrs[f"flag_{k}"] = bool(v)
                    update_index(db2, run_id, params, flags, {}, 0, status="failed")

                if progress_callback is not None:
                    callback_stats = {
                        "_run_time_s": 0.0,
                        "_equil_time_s": equil_info["equil_time_s"],
                        "_equil_cache_hit": equil_info["cache_hit"],
                        "_equil_S_gp": equil_info["S_gp"],
                        "_equil_twin": equil_info["twin"],
                        "_equil_Twin_S_gp": equil_info["Twin_S_gp"],
                        "_equilibrate_nn": equilibrate_nn,
                    }
                    progress_callback(completed_count, n_total, run_id, "failed", callback_stats)

            # After each run, check for a user-requested stop.  Cancel any not-yet-started
            # futures (best-effort; running processes will complete naturally).
            if stop_event is not None and stop_event.is_set():
                for remaining_fut in future_to_run:
                    if not remaining_fut.done():
                        remaining_fut.cancel()
                if verbose:
                    print(f"  Sweep aborted by user after {completed_count}/{n_total} runs.")
                break

    if verbose:
        print(f"Parallel sweep complete: {len(successful)}/{n_total} runs succeeded.")

    return successful
