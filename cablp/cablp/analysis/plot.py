"""
Visualization routines for LAPDSim results.

Single-run plots
----------------
``plot_run(results, params, flags)`` is the main entry point.  It routes to
2-D line plots (≤5 cells) or 3-D surface plots (>5 cells).

Sweep analysis plots
--------------------
``plot_sweep_variance`` and ``plot_sweep_heatmap`` operate on the index dict
returned by ``database.load_index()``.

Run comparison
--------------
``plot_run_comparison(db_path, run_ids, quantity)`` overlays one quantity from
multiple archived runs on a single axes.
"""
import math

import matplotlib.pyplot as plt
import numpy as np

from .stats import cell_centers

_qe_SI = 1.602176634e-19  # J per eV

# ── Position helpers ──────────────────────────────────────────────────────────


def position_labels(z_positions, convention="sim"):
    """
    Build legend/axis labels from cell-center positions.

    Parameters
    ----------
    z_positions : array-like
        Cell-center positions in cm (from ``cell_centers()``).
    convention : str
        'sim' or 'exp' — used only to annotate the label if desired.

    Returns
    -------
    list of str, e.g. ['z=300 cm', 'z=900 cm', 'z=1500 cm']
    """
    return [f"z={z:.0f} cm" for z in z_positions]


def _z_axis_label(convention):
    if convention == "sim":
        return "Simulation z [cm]  (z=0 at source end)"
    return "Experimental z [cm]  (z=0 at far end)"


# ── Title / subtitle helper ───────────────────────────────────────────────────


def _run_title(params, flags):
    """One-line parameter summary for plot titles."""
    gas = params.get("gas_type", "?")
    Vd = params.get("Vd", 0)
    Id = params.get("Id", 0)
    cells = params.get("cells", "?")
    twin_active = flags.get("TwinCathode", False)
    Twin_Id = params.get("Twin_Id", 0.0) if twin_active else 0.0
    S_gp = params.get("S_gp", 0.0)
    Twin_S_gp = params.get("Twin_S_gp", 0.0) if twin_active else 0.0
    P_MW = (Id + Twin_Id) * Vd / 1e6
    S_gp_total = S_gp + Twin_S_gp
    twin = "twin" if twin_active else "single"
    return f"{gas}  Vd={Vd:.0f} V  P={P_MW:.2f} MW  S_gp={S_gp_total:.0f}  {cells} cells  [{twin}]"


# ── Main entry point ──────────────────────────────────────────────────────────


def plot_run(results, params, flags, z_convention="sim", save_dir=None):
    """
    Plot one simulation run.

    Routes to :func:`_plot_2d` (≤5 cells) or :func:`_plot_3d` (>5 cells).

    Parameters
    ----------
    results : dict
        Output of ``sim.get_results()``.
    params : dict
        Input parameter dict used for the run.
    flags : dict
        Input flags dict used for the run.
    z_convention : str
        'sim'  — z=0 at source/left end.
        'exp'  — z=0 at far/right end (experimental convention).
    save_dir : str or path-like or None
        If given, PNGs are saved here with auto-generated filenames.

    Returns
    -------
    dict of {figure_name: matplotlib.Figure}
    """
    n_cells = results["ne"].shape[1]
    L_plasma = params.get("Lp", params.get("L_plasma", 1800))
    z_pos = cell_centers(n_cells, L_plasma, convention=z_convention)

    if n_cells <= 5:
        return _plot_2d(results, params, flags, z_pos, z_convention, save_dir)
    else:
        return _plot_3d(results, params, flags, z_pos, z_convention, save_dir)


# ── 2-D plots (≤5 cells) ─────────────────────────────────────────────────────


def _save(fig, name, save_dir):
    if save_dir is not None:
        import pathlib

        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(pathlib.Path(save_dir) / f"{name}.png", dpi=150, bbox_inches="tight")


def _plot_2d(results, params, flags, z_pos, z_convention, save_dir):
    """2-D time-series plots for ≤5 cells."""
    t = results["time"]
    labels = position_labels(z_pos, z_convention)
    title_base = _run_title(params, flags)

    n_cells = results["ne"].shape[1]
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(n_cells)]

    figs = {}

    # Helper: one figure with one axes
    def _fig(title, ylabel, yscale="linear"):
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        ax.set_xlabel("Time [ms]")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}\n{title_base}", fontsize=10)
        ax.set_yscale(yscale)
        return fig, ax

    _LEGEND_KW = dict(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)

    def _lines(ax, arr, legend=True):
        for ci in range(n_cells):
            ax.plot(t, arr[:, ci], color=colors[ci], label=labels[ci])
        if legend:
            ax.legend(**_LEGEND_KW)

    def _autolim(ax, *arrs, t_cut=0.5):
        """Set ylim from non-transient data (t >= t_cut ms), ignoring early spikes."""
        mask = t >= t_cut
        if not mask.any():
            return
        vals = []
        for arr in arrs:
            a = np.asarray(arr)
            sub = a[mask, :].ravel() if a.ndim == 2 else a[mask].ravel()
            finite = sub[np.isfinite(sub)]
            if ax.get_yscale() == "log":
                finite = finite[finite > 0]
            if finite.size:
                vals.append(finite)
        if not vals:
            return
        all_v = np.concatenate(vals)
        if not all_v.size:
            return
        if ax.get_yscale() == "log":
            lv = np.log10(all_v)
            span = lv.max() - lv.min()
            margin = max(0.05 * span, 0.3)
            ax.set_ylim(10 ** (lv.min() - margin), 10 ** (lv.max() + margin))
        else:
            span = all_v.max() - all_v.min()
            margin = max(0.05 * span, abs(np.median(all_v)) * 0.05, 1e-30)
            ax.set_ylim(all_v.min() - margin, all_v.max() + margin)

    # ── Electron density ──────────────────────────────────────────────────────
    fig, ax = _fig("Electron Density", r"$n_e$ [cm$^{-3}$]")
    _lines(ax, results["ne"])
    _autolim(ax, results["ne"])
    figs["ne"] = fig
    _save(fig, "ne", save_dir)

    # ── Neutral density ───────────────────────────────────────────────────────
    fig, ax = _fig("Neutral Density", r"$n_n$ [cm$^{-3}$]")
    _lines(ax, results["nn"])
    _autolim(ax, results["nn"])
    figs["nn"] = fig
    _save(fig, "nn", save_dir)

    # ── Ionisation fraction (ne/nn) ───────────────────────────────────────────
    with np.errstate(divide="ignore", invalid="ignore"):
        ion_frac = np.where(results["nn"] > 0, results["ne"] / results["nn"], np.nan)
    fig, ax = _fig("Ionisation Fraction", r"$n_e / n_n$", yscale="log")
    _lines(ax, ion_frac)
    ax.set_ylim(1e-3, 1e3)           # fixed symmetric bounds (overrides autolim)
    ax.axhline(1.0, color="lightgray", lw=1.0, zorder=0)
    figs["ion_ratio"] = fig
    _save(fig, "ion_ratio", save_dir)

    # ── Electron temperature ──────────────────────────────────────────────────
    fig, ax = _fig("Electron Temperature", r"$T_e$ [eV]")
    _lines(ax, results["Te"])
    _autolim(ax, results["Te"])
    figs["Te"] = fig
    _save(fig, "Te", save_dir)

    # ── Ion temperature ───────────────────────────────────────────────────────
    fig, ax = _fig("Ion Temperature", r"$T_i$ [eV]")
    _lines(ax, results["Ti"])
    _autolim(ax, results["Ti"])
    figs["Ti"] = fig
    _save(fig, "Ti", save_dir)

    # ── Neutral cooling (Qen) — power per cell ────────────────────────────────
    # Convert from temperature-rate units to watts per cell:
    #   P [W] = (3/2) * ne * Qen * cell_vol * qe_SI
    # en_factor = 2/3 is already folded in, so (3/2) * Qen/en_factor = (3/2)*(3/2)*Qen
    # For consistency with existing notebooks we approximate:
    #   P ≈ ne * Qen * cell_vol * qe_SI  (drops the 3/2 pre-factor)
    L_plasma = params.get("Lp", 1800)
    Rp = params.get("Rp", 18)
    cell_vol = math.pi * Rp**2 * (L_plasma / n_cells)  # cm³ per cell

    Qen_W = results["Qen"] * results["ne"] * cell_vol * _qe_SI
    Qen_W = np.where(Qen_W > 0, Qen_W, np.nan)
    Qen_total = np.nansum(Qen_W, axis=1)
    Qen_total = np.where(Qen_total > 0, Qen_total, np.nan)

    fig, ax = _fig("Electron Cooling by Neutral Radiation", "Power [W]", yscale="log")
    _lines(ax, Qen_W, legend=False)
    ax.plot(t, Qen_total, color="black", lw=2.0, ls="--", label="Total")
    ax.legend(**_LEGEND_KW)
    _autolim(ax, Qen_W, Qen_total)
    ax.set_ylim(bottom=1e3)
    figs["Qen_power"] = fig
    _save(fig, "Qen_power", save_dir)

    # ── Ion power loss to charge exchange (Qcx) ───────────────────────────────
    Qcx_W = np.abs(results["Qcx"]) * results["ne"] * cell_vol * _qe_SI
    Qcx_W = np.where(Qcx_W > 0, Qcx_W, np.nan)
    Qcx_total = np.nansum(Qcx_W, axis=1)
    Qcx_total = np.where(Qcx_total > 0, Qcx_total, np.nan)

    fig, ax = _fig("Ion Power Loss to Charge Exchange", "Power [W]", yscale="log")
    _lines(ax, Qcx_W, legend=False)
    ax.plot(t, Qcx_total, color="black", lw=2.0, ls="--", label="Total")
    ax.legend(**_LEGEND_KW)
    _autolim(ax, Qcx_W, Qcx_total)
    figs["Qcx_power"] = fig
    _save(fig, "Qcx_power", save_dir)

    # ── Power balance ─────────────────────────────────────────────────────────
    # Normalise each term (summed over cells) to input power.
    Vd = params.get("Vd", 0)
    Id = params.get("Id", 0)
    input_power = Vd * Id
    if flags.get("TwinCathode", False):
        input_power += params.get("Twin_Vd", Vd) * params.get("Twin_Id", Id)
    if input_power == 0:
        input_power = 1.0  # avoid divide-by-zero for passive runs

    def _frac(key):
        total = np.abs(results[key]).sum(axis=1)
        return np.where(total > 0, total / input_power, np.nan)

    _pb_fracs = [_frac(k) for k in ("e_par_flux", "Qie", "Qei", "Qen", "Qeb", "Qcx")]
    fig, ax = _fig("Power Balance (fraction of input)", "Fraction of Input Power", yscale="log")
    ax.plot(t, _pb_fracs[0], label=r"$e$-par flux", lw=1.5)
    ax.plot(t, _pb_fracs[1], label=r"$Q_{ie}$", lw=1.5)
    ax.plot(t, _pb_fracs[2], label=r"$Q_{ei}$", lw=1.5)
    ax.plot(t, _pb_fracs[3], label=r"$Q_{en}$", lw=1.5)
    ax.plot(t, _pb_fracs[4], label=r"$Q_{eb}$", lw=1.5)
    ax.plot(t, _pb_fracs[5], label=r"$Q_{cx}$", lw=1.5)
    ax.legend(**_LEGEND_KW)
    _autolim(ax, *_pb_fracs)
    ax.set_ylim(bottom=1e-2)
    figs["power_balance"] = fig
    _save(fig, "power_balance", save_dir)

    # ── Ion power balance ─────────────────────────────────────────────────────
    _ion_fracs = [_frac(k) for k in ("Qie", "Qcx", "i_par_flux")]
    fig, ax = _fig("Ion Power Balance (fraction of input)", "Fraction of Input Power", yscale="log")
    ax.plot(t, _ion_fracs[0], label=r"$Q_{io}$ (e-i exchange)", lw=1.5)
    ax.plot(t, _ion_fracs[1], label=r"$Q_{cx}$ (charge exchange)", lw=1.5)
    ax.plot(t, _ion_fracs[2], label=r"$i$-par flux (conductive)", lw=1.5)
    ax.legend(**_LEGEND_KW)
    _autolim(ax, *_ion_fracs)
    figs["ion_power_balance"] = fig
    _save(fig, "ion_power_balance", save_dir)

    # ── Isat synthetic diagnostic (normalised to t=d_off) ────────────────────
    if "isat" in results:
        isat_raw = results["isat"]
        if isat_raw.ndim == 1:          # old run: only first cell was stored
            isat_raw = isat_raw[:, np.newaxis]

        # Full-run plot: normalise to value at d_off
        t_norm_ms = params.get("d_off", 20e-3) * 1e3  # d_off seconds → ms
        norm_idx = int(np.argmin(np.abs(t - t_norm_ms)))
        norm_vals = isat_raw[norm_idx, :]  # (n_cells,)
        with np.errstate(invalid="ignore", divide="ignore"):
            isat_norm = np.where(norm_vals != 0,
                                 isat_raw / norm_vals[np.newaxis, :],
                                 np.nan)
        fig, ax = _fig(
            rf"Normalised $I_{{sat}}$  ($n_e\sqrt{{T_e}}$, norm. at t={t_norm_ms:.0f} ms)",
            r"$I_{sat}$ [norm.]",
        )
        _lines(ax, isat_norm)
        _autolim(ax, isat_norm)
        ax.axhline(1, color="k", lw=0.8, ls="--")
        figs["isat"] = fig
        _save(fig, "isat", save_dir)

        # Afterglow-only plot: t >= 20 ms, normalise to first point in window
        ag_mask = t >= 20.0
        if ag_mask.any():
            t_ag = t[ag_mask]
            isat_ag = isat_raw[ag_mask, :]
            norm_vals_ag = isat_ag[0, :]
            with np.errstate(invalid="ignore", divide="ignore"):
                isat_ag_norm = np.where(norm_vals_ag != 0,
                                        isat_ag / norm_vals_ag[np.newaxis, :],
                                        np.nan)
            fig, ax = _fig(
                r"Afterglow $I_{sat}$ (t$\geq$20 ms, norm. at t=20 ms)",
                r"$I_{sat}$ [norm.]",
            )
            for ci in range(n_cells):
                ax.plot(t_ag, isat_ag_norm[:, ci], color=colors[ci], label=labels[ci])
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
            ax.axhline(1, color="k", lw=0.8, ls="--")
            figs["isat_afterglow"] = fig
            _save(fig, "isat_afterglow", save_dir)

    # ── Parallel velocity ─────────────────────────────────────────────────────
    if "v_plasma" in results:
        _v_plot = results["v_plasma"] / 100.0
        fig, ax = _fig("Parallel Plasma Velocity", r"$v_\parallel$ [m/s]")
        _lines(ax, _v_plot)
        _autolim(ax, _v_plot)
        ax.axhline(0, color="k", lw=0.8, ls="--")
        figs["v_plasma"] = fig
        _save(fig, "v_plasma", save_dir)

    # ── Parallel Mach number ──────────────────────────────────────────────────
    if "v_plasma" in results and "Te" in results:
        _gas = str(params.get("gas_type", "He")).strip().lower()
        _mu = 4.0 if _gas in ("he", "helium") else 1.0  # He=4, H=1
        c_s = 9.79e5 * np.sqrt(results["Te"] / _mu)  # cm/s, same units as v_plasma
        with np.errstate(invalid="ignore", divide="ignore"):
            mach = np.where(c_s > 0, results["v_plasma"] / c_s, np.nan)
        fig, ax = _fig("Parallel Mach Number", r"$M_\parallel = v_\parallel / c_s(T_e)$")
        _lines(ax, mach)
        _autolim(ax, mach)
        ax.axhline(0, color="k", lw=0.8, ls="--")
        figs["mach"] = fig
        _save(fig, "mach", save_dir)

    # ── Mean free paths ───────────────────────────────────────────────────────
    if "primary_mfp" in results and "bulk_mfp" in results:
        fig, ax = _fig("Electron Mean Free Paths / Cell Length", "MFP / cell length", yscale="log")
        for ci in range(n_cells):
            ax.plot(
                t,
                results["primary_mfp"][:, ci],
                color=colors[ci],
                ls="-",
                label=f"primary {labels[ci]}",
            )
            ax.plot(
                t,
                results["bulk_mfp"][:, ci],
                color=colors[ci],
                ls="--",
                label=f"bulk {labels[ci]}",
            )
        ax.axhline(1, color="k", lw=0.8, ls=":", label="MFP = cell length")
        _autolim(ax, results["primary_mfp"], results["bulk_mfp"])
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=8)
        figs["mfp"] = fig
        _save(fig, "mfp", save_dir)

    # ── Coulomb logarithm ─────────────────────────────────────────────────────
    if "ln_lambda" in results:
        fig, ax = _fig("Coulomb Logarithm", r"$\ln \Lambda$")
        _lines(ax, results["ln_lambda"])
        _autolim(ax, results["ln_lambda"])
        figs["ln_lambda"] = fig
        _save(fig, "ln_lambda", save_dir)

    return figs


# ── 3-D plots (>5 cells) ──────────────────────────────────────────────────────


def _plot_3d(results, params, flags, z_pos, z_convention, save_dir):
    """3-D surface plots for >5 cells."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # registers 3D projection

    t = results["time"]
    title_base = _run_title(params, flags)
    z_label = _z_axis_label(z_convention)
    figs = {}

    T, Z = np.meshgrid(t, z_pos)  # both (n_cells, n_t)

    def _surf(key, label, title, log=False, scale=1.0):
        vals = results[key].T * scale  # (n_cells, n_t)
        if log:
            vals = np.where(vals > 0, np.log10(vals), np.nan)
            label = f"log₁₀({label})"

        fig = plt.figure(figsize=(10, 6), constrained_layout=True)
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(T, Z, vals, cmap="viridis", linewidth=0, antialiased=True)
        ax.set_xlabel("Time [ms]")
        ax.set_ylabel(z_label)
        ax.set_zlabel(label)
        ax.set_title(f"{title}\n{title_base}", fontsize=10)
        fig.colorbar(surf, ax=ax, shrink=0.5, label=label)
        return fig

    figs["ne_3d"] = _surf("ne", r"$n_e$ [cm⁻³]", "Electron Density", log=True)
    _save(figs["ne_3d"], "ne_3d", save_dir)

    figs["nn_3d"] = _surf("nn", r"$n_n$ [cm⁻³]", "Neutral Density", log=True)
    _save(figs["nn_3d"], "nn_3d", save_dir)

    figs["Te_3d"] = _surf("Te", r"$T_e$ [eV]", "Electron Temperature")
    _save(figs["Te_3d"], "Te_3d", save_dir)

    figs["Ti_3d"] = _surf("Ti", r"$T_i$ [eV]", "Ion Temperature")
    _save(figs["Ti_3d"], "Ti_3d", save_dir)

    if "v_plasma" in results:
        figs["v_plasma_3d"] = _surf("v_plasma", r"$v_\parallel$ [m/s]", "Parallel Velocity", scale=1/100.0)
        _save(figs["v_plasma_3d"], "v_plasma_3d", save_dir)

    return figs


# ── Sweep analysis plots ──────────────────────────────────────────────────────


def plot_sweep_variance(
    index,
    x_param,
    hue_param=None,
    quantity="ne",
    metric="var",
    t_label="10–20 ms",
    save_dir=None,
):
    """
    Scatter plot: varied parameter vs. cell-to-cell variance (or other metric).

    Parameters
    ----------
    index : dict
        Output of ``load_index()``.
    x_param : str
        Parameter name to plot on the x-axis.  Must be in ``index['params']``.
    hue_param : str or None
        If given, color points by this second parameter or flag.
    quantity : str
        'ne' or 'Te'.
    metric : str
        'var'  — variance of per-cell time-means (spatial variance).
        'cov'  — coefficient of variation (std/mean).
        'min' / 'max' / 'mean'.
    t_label : str
        Time window description shown in title.
    save_dir : str or None

    Returns
    -------
    matplotlib.Figure
    """
    stats = index["stats_10_20ms"]
    params = index["params"]
    flags = index["flags"]

    y_key = f"{quantity}_{metric}"
    if y_key not in stats:
        raise KeyError(
            f"'{y_key}' not found in index stats.  "
            f"Available: {list(stats.keys())}"
        )

    # Retrieve x values
    def _to_float_array(arr, name):
        try:
            return np.asarray(arr, dtype=float)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Parameter '{name}' contains non-numeric values and cannot be "
                f"used as an axis.  Values: {list(arr)[:5]}"
            ) from exc

    if x_param in params:
        x = _to_float_array(params[x_param], x_param)
    elif x_param in flags:
        x = _to_float_array(flags[x_param], x_param)
    else:
        raise KeyError(f"'{x_param}' not found in params or flags.")

    y = np.asarray(stats[y_key], dtype=float)

    # Filter to successful runs
    ok = np.array(index["status"]) == "ok"
    x, y = x[ok], y[ok]

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)

    if hue_param is not None:
        if hue_param in params:
            hue = _to_float_array(params[hue_param], hue_param)[ok]
        elif hue_param in flags:
            hue = _to_float_array(flags[hue_param], hue_param)[ok]
        else:
            raise KeyError(f"hue_param '{hue_param}' not found in params or flags.")

        sc = ax.scatter(x, y, c=hue, cmap="viridis", s=60, zorder=3)
        fig.colorbar(sc, ax=ax, label=hue_param)
    else:
        ax.scatter(x, y, s=60, zorder=3)

    ax.set_xlabel(x_param)
    metric_labels = {
        "var": "variance",
        "cov": "coeff. of variation (std/mean)",
        "min": "minimum",
        "max": "maximum",
        "mean": "mean",
    }
    ax.set_ylabel(
        f"{quantity} cell-to-cell {metric_labels.get(metric, metric)}"
        f" [{t_label}]"
    )
    ax.set_title(f"Effect of {x_param} on {quantity} uniformity  [{t_label}]")
    ax.grid(True, alpha=0.3)

    if save_dir is not None:
        _save(fig, f"sweep_var_{quantity}_{x_param}", save_dir)

    return fig


def plot_sweep_heatmap(
    index,
    x_param,
    y_param,
    quantity="ne_var",
    t_label="10–20 ms",
    save_dir=None,
):
    """
    2-D heatmap of a sweep statistic as a function of two varied parameters.

    Parameters
    ----------
    index : dict
        Output of ``load_index()``.
    x_param, y_param : str
        Parameter names for the two axes.
    quantity : str
        Key in ``index['stats_10_20ms']``, e.g. ``'ne_var'``, ``'Te_mean'``.
    t_label : str
    save_dir : str or None

    Returns
    -------
    matplotlib.Figure
    """
    stats = index["stats_10_20ms"]
    params = index["params"]
    flags = index["flags"]

    if quantity not in stats:
        raise KeyError(f"'{quantity}' not found in index stats.")

    def _get(name):
        try:
            if name in params:
                return np.asarray(params[name], dtype=float)
            if name in flags:
                return np.asarray(flags[name], dtype=float)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Parameter '{name}' contains non-numeric values and cannot be used as an axis."
            ) from exc
        raise KeyError(f"'{name}' not found in params or flags.")

    ok = np.array(index["status"]) == "ok"
    x_all = _get(x_param)[ok]
    y_all = _get(y_param)[ok]
    z_all = np.asarray(stats[quantity], dtype=float)[ok]

    x_vals = np.unique(x_all)
    y_vals = np.unique(y_all)

    grid = np.full((len(y_vals), len(x_vals)), np.nan)
    for xi, xv in enumerate(x_vals):
        for yi, yv in enumerate(y_vals):
            mask = (x_all == xv) & (y_all == yv)
            if mask.any():
                grid[yi, xi] = z_all[mask].mean()

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    im = ax.pcolormesh(x_vals, y_vals, grid, cmap="viridis", shading="auto")
    fig.colorbar(im, ax=ax, label=quantity)
    ax.set_xlabel(x_param)
    ax.set_ylabel(y_param)
    ax.set_title(f"{quantity}  [{t_label}]")

    if save_dir is not None:
        _save(fig, f"sweep_heatmap_{quantity}_{x_param}_{y_param}", save_dir)

    return fig


# ── Run comparison ────────────────────────────────────────────────────────────

_COMPARISON_YLABELS = {
    "ne": r"$n_e$ [cm$^{-3}$]",
    "nn": r"$n_n$ [cm$^{-3}$]",
    "Te": r"$T_e$ [eV]",
    "Ti": r"$T_i$ [eV]",
    "v_plasma": r"$v_\parallel$ [m/s]",
    "isat": r"$I_{sat}$ [norm.]",
    "ln_lambda": r"$\ln\Lambda$",
    "primary_mfp": "Primary MFP / cell length",
    "bulk_mfp": "Bulk MFP / cell length",
}

_RUN_LINESTYLES = ["-", "--", ":", "-."]


def plot_run_comparison(db_path, run_ids, quantity, cell_idx=-1):
    """
    Load multiple archived runs and overlay one quantity on a single axes.

    Parameters
    ----------
    db_path : str or path-like
        Path to the HDF5 database.
    run_ids : list of str
        Run IDs to compare (typically 2–4).
    quantity : str
        Key in the results dict, e.g. ``'ne'``, ``'Te'``, ``'v_plasma'``.
    cell_idx : int
        Which cell to plot.  ``-1`` (default) plots every cell individually,
        using color to distinguish cells and linestyle to distinguish runs.

    Returns
    -------
    matplotlib.Figure
    """
    from matplotlib.lines import Line2D
    from .database import open_db, load_run

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    cmap = plt.get_cmap("tab10")

    run_labels = []   # (run_i, label_str) for legend
    cells_seen = set()

    with open_db(db_path) as db:
        for run_i, run_id in enumerate(run_ids):
            params, flags, results = load_run(db, run_id, keys=["time", quantity])
            t = results["time"]
            data = results.get(quantity)
            if data is None:
                continue

            gas = params.get("gas_type", "?")
            Vd = params.get("Vd", 0)
            Id = params.get("Id", 0)
            twin_active = flags.get("TwinCathode", False)
            Twin_Id = params.get("Twin_Id", 0.0) if twin_active else 0.0
            S_gp = params.get("S_gp", 0.0)
            Twin_S_gp = params.get("Twin_S_gp", 0.0) if twin_active else 0.0
            P_total_MW = (Id + Twin_Id) * Vd / 1e6
            S_gp_total = S_gp + Twin_S_gp
            twin_str = "twin" if twin_active else "single"
            run_label = f"{run_id}  {gas}  P={P_total_MW:.2f}MW  S_gp={S_gp_total:.0f}  [{twin_str}]"
            run_labels.append((run_i, run_label))

            ls = _RUN_LINESTYLES[run_i % len(_RUN_LINESTYLES)]

            # Scale velocity from cm/s to m/s
            if quantity == "v_plasma":
                data = data / 100.0

            # Normalise isat per-cell to value at t=20 ms
            if quantity == "isat":
                if data.ndim == 1:      # old run: only first cell was stored
                    data = data[:, np.newaxis]
                norm_idx = int(np.argmin(np.abs(t - 20.0)))
                norm_vals = data[norm_idx, :]
                with np.errstate(invalid="ignore", divide="ignore"):
                    data = np.where(norm_vals != 0,
                                    data / norm_vals[np.newaxis, :],
                                    np.nan)

            if cell_idx == -1 and data.ndim == 2:
                # One line per cell: color = cell, linestyle = run
                for c in range(data.shape[1]):
                    ax.plot(t, data[:, c], color=cmap(c % 10), ls=ls, lw=1.5)
                    cells_seen.add(c)
            elif data.ndim == 2:
                ax.plot(t, data[:, int(cell_idx)], color=cmap(run_i % 10), ls="-", lw=1.5)
            else:
                ax.plot(t, data, color=cmap(run_i % 10), ls="-", lw=1.5)

    ylabel = _COMPARISON_YLABELS.get(quantity, quantity)
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel(ylabel)

    cell_desc = "all cells" if cell_idx == -1 else f"cell {cell_idx}"
    if quantity == "isat":
        ax.set_title(rf"Normalised $I_{{sat}}$ comparison  ({cell_desc}, norm. at t=20 ms)")
        ax.axhline(1, color="k", lw=0.8, ls="--")
    else:
        ax.set_title(f"{quantity} comparison  ({cell_desc})")

    if cell_idx == -1 and cells_seen:
        # Two-part legend: linestyle = run, color = cell
        run_handles = [
            Line2D([0], [0], color="dimgray",
                   ls=_RUN_LINESTYLES[i % len(_RUN_LINESTYLES)], lw=1.5, label=lbl)
            for i, lbl in run_labels
        ]
        cell_handles = [
            Line2D([0], [0], color=cmap(c % 10), ls="-", lw=2.5, label=f"cell {c}")
            for c in sorted(cells_seen)
        ]
        ax.legend(handles=run_handles + cell_handles, fontsize=7,
                  loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
    else:
        handles = [
            Line2D([0], [0], color=cmap(i % 10), ls="-", lw=1.5, label=lbl)
            for i, lbl in run_labels
        ]
        ax.legend(handles=handles, fontsize=8,
                  loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)

    ax.grid(True, alpha=0.3)
    return fig
