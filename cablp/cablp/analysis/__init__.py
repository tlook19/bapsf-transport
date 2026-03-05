"""
cablp.analysis — sweep, database and visualisation tools for LAPDSim.

Quick-start
-----------
**Run a parameter sweep and archive results:**

::

    from cablp.analysis import grid_sweep

    grid_sweep(
        param_ranges={"Vd": [100, 200], "Id": [2500, 5000]},
        flag_ranges={"eperp": [True, False]},
        fixed_params={"gas_type": "He", "cells": 3, "d_off": 20e-3, "end": 25e-3},
        db_path="sweep.h5",
        t_window=(10.0, 20.0),
    )

**Load and query the database:**

::

    from cablp.analysis import open_db, load_index

    with open_db("sweep.h5") as db:
        idx = load_index(db)

    ne_var = idx["stats_10_20ms"]["ne_var"]
    Vd_vals = idx["params"]["Vd"]

**Visualise one run:**

::

    from cablp.analysis import open_db, load_run, plot_run

    with open_db("sweep.h5") as db:
        params, flags, results = load_run(db, "run_0000")

    figs = plot_run(results, params, flags, z_convention="sim")
    figs["ne"].show()

**Variance scatter plot:**

::

    from cablp.analysis import open_db, load_index, plot_sweep_variance

    with open_db("sweep.h5") as db:
        idx = load_index(db)

    plot_sweep_variance(idx, x_param="Vd", hue_param="Id", quantity="ne")
"""

from .database import (
    open_db,
    save_run,
    load_run,
    load_run_stats,
    list_runs,
    update_index,
    load_index,
    rebuild_index,
)
from .sweep import grid_sweep, grid_sweep_parallel, param_combinations, equilibrate_neutrals
from .stats import cell_centers, compute_window_stats
from .plot import (
    plot_run,
    plot_run_comparison,
    plot_sweep_variance,
    plot_sweep_heatmap,
    position_labels,
)

__all__ = [
    # database
    "open_db",
    "save_run",
    "load_run",
    "load_run_stats",
    "list_runs",
    "update_index",
    "load_index",
    "rebuild_index",
    # sweep
    "grid_sweep",
    "grid_sweep_parallel",
    "param_combinations",
    "equilibrate_neutrals",
    # stats
    "cell_centers",
    "compute_window_stats",
    # plot
    "plot_run",
    "plot_run_comparison",
    "plot_sweep_variance",
    "plot_sweep_heatmap",
    "position_labels",
]
