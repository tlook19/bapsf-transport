import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from cablp.solvers._sim1d import load_result_hdf5


_DENSE_MANTISSAS = np.array(
    [
        1.0,
        1.125,
        1.25,
        1.375,
        1.5,
        1.75,
        2.0,
        2.25,
        2.5,
        2.75,
        3.0,
        3.5,
        4.0,
        4.5,
        5.0,
        6.25,
        7.5,
        8.75,
    ]
)


def main(argv=None):
    args = _parse_args(argv)
    result = load_result_hdf5(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    time_origin = _time_origin(result, mode=args.time_origin)
    time_scale, time_label = _time_unit(args.time_unit)
    shifted_time_s = np.asarray(result.time, dtype=float) - time_origin
    t_plot = shifted_time_s * time_scale
    t_slice_ms = shifted_time_s * 1.0e3
    z_cm = np.asarray(result.z_cm, dtype=float)
    phase_events = _shifted_phase_events(result, time_origin, time_scale)
    phase_events_ms = _shifted_phase_events(result, time_origin, 1.0e3)
    prefix = args.prefix or Path(args.input).stem

    figures = {
        "summary": _plot_summary(result, t_plot, time_label, phase_events),
        "densities": _plot_densities(result, z_cm, t_plot, time_label, phase_events),
        "temperatures": _plot_temperatures(
            result,
            z_cm,
            t_plot,
            time_label,
            phase_events,
        ),
        "velocity": _plot_velocity(result, z_cm, t_plot, time_label, phase_events),
        "energy_terms": _plot_energy_terms(result, t_plot, time_label, phase_events),
        "cathode": _plot_cathode(result, t_plot, time_label, phase_events),
        "phase": _plot_phase(result, t_plot, time_label, phase_events),
    }

    written = []
    for name, fig in figures.items():
        if fig is None:
            continue
        path = output_dir / f"{prefix}_{name}.{args.format}"
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        written.append(path)
        plt.close(fig)

    slice_written = []
    if not args.no_time_slices:
        slice_written = _write_main_discharge_slices(
            result=result,
            z_cm=z_cm,
            t_ms=t_slice_ms,
            phase_events_ms=phase_events_ms,
            output_dir=output_dir / "time_slices",
            prefix=prefix,
            image_format=args.format,
            dpi=args.dpi,
            interval_ms=args.slice_interval_ms,
        )
        written.extend(slice_written)

    print(
        "sim1d plots written: "
        f"count={len(written)}, "
        f"time_slices={len(slice_written)}, "
        f"time_origin={time_origin:.9e} s, "
        f"output_dir={output_dir}"
    )
    for path in written:
        print(path)
    return 0


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description=(
            "Generate run-inspector style plots for a saved LAPDSim1D HDF5 result."
        )
    )
    parser.add_argument("input", help="Input sim1d HDF5 result path.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for plot files. Defaults to '<input stem>_plots'.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Output filename prefix. Defaults to the input file stem.",
    )
    parser.add_argument(
        "--format",
        default="png",
        choices=("png", "pdf", "svg"),
        help="Output image format.",
    )
    parser.add_argument("--dpi", type=int, default=150, help="Raster output DPI.")
    parser.add_argument(
        "--time-origin",
        default="main_discharge",
        choices=("main_discharge", "breakdown", "start"),
        help=(
            "Time origin for plots. The default shifts t=0 to the saved "
            "main_discharge phase event."
        ),
    )
    parser.add_argument(
        "--time-unit",
        default="ms",
        choices=("s", "ms"),
        help="Displayed time unit after applying the origin shift.",
    )
    parser.add_argument(
        "--slice-interval-ms",
        type=float,
        default=1.0,
        help="Main-discharge time-slice spacing [ms].",
    )
    parser.add_argument(
        "--no-time-slices",
        action="store_true",
        help="Disable automatic 1D profile plots during main discharge.",
    )
    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = str(Path(args.input).with_suffix("")) + "_plots"
    return args


def _time_unit(unit):
    if unit == "s":
        return 1.0, "Time since main discharge start [s]"
    return 1.0e3, "Time since main discharge start [ms]"


def _time_origin(result, mode="main_discharge"):
    if mode == "start":
        return float(np.asarray(result.time, dtype=float)[0])
    if mode == "breakdown":
        trigger = float(getattr(result, "t_breakdown_trigger", np.nan))
        if np.isfinite(trigger):
            return trigger
    if mode == "main_discharge":
        phase_events = getattr(result, "phase_events", {})
        times = np.asarray(phase_events.get("time", ()), dtype=float)
        phases = np.asarray(phase_events.get("phase", ()), dtype=object)
        matches = np.flatnonzero(phases == "main_discharge")
        if matches.size:
            return float(times[matches[0]])
        trigger = float(getattr(result, "t_breakdown_trigger", np.nan))
        if np.isfinite(trigger):
            return trigger
    return float(np.asarray(result.time, dtype=float)[0])


def _shifted_phase_events(result, time_origin, time_scale):
    phase_events = getattr(result, "phase_events", {})
    times = np.asarray(phase_events.get("time", ()), dtype=float)
    phases = np.asarray(phase_events.get("phase", ()), dtype=object)
    return [
        (float((time - time_origin) * time_scale), str(phase))
        for time, phase in zip(times, phases)
    ]


def _plot_title(result):
    params = getattr(result, "params", {}) or {}
    flags = getattr(result, "flags", {}) or {}
    gas = params.get("gas_type", "?")
    nx = params.get("nx", getattr(result, "n", np.empty((0, 0))).shape[1])
    s_gp = params.get("S_gp", np.nan)
    v_bank = params.get("V_bank", np.nan)
    implicit_heat = flags.get("implicit_heat_conduction", False)
    return (
        f"sim1d {gas}  nx={nx}  S_gp={_fmt_scalar(s_gp)}  "
        f"V_bank={_fmt_scalar(v_bank)}  implicit_heat={implicit_heat}"
    )


def _fmt_scalar(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(value):
        return "nan"
    return f"{value:g}"


def _finite_2d(values):
    arr = np.asarray(values, dtype=float)
    return np.where(np.isfinite(arr), arr, np.nan)


def _positive_log10(values):
    arr = _finite_2d(values)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(arr > 0.0, np.log10(arr), np.nan)


def _cell_average(result, values, weight_name="plasma_volume_cm3"):
    arr = _finite_2d(values)
    weights = np.asarray(getattr(result, weight_name), dtype=float)
    if arr.ndim != 2 or weights.ndim != 1 or weights.size != arr.shape[1]:
        return np.nanmean(arr, axis=1)
    valid = np.isfinite(arr)
    weighted = np.where(valid, arr * weights[np.newaxis, :], 0.0)
    denom = np.where(valid, weights[np.newaxis, :], 0.0).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 0.0, weighted.sum(axis=1) / denom, np.nan)


def _sum_power(result, terms):
    volume = np.asarray(result.plasma_volume_cm3, dtype=float)
    series = {}
    for name, arr in terms.items():
        values = _finite_2d(arr)
        if values.ndim != 2 or values.shape[1] != volume.size:
            continue
        series[name] = np.nansum(values * volume[np.newaxis, :], axis=1)
    return series


def _write_main_discharge_slices(
    *,
    result,
    z_cm,
    t_ms,
    phase_events_ms,
    output_dir,
    prefix,
    image_format,
    dpi,
    interval_ms,
):
    slice_times = _main_discharge_slice_times(
        result,
        phase_events_ms=phase_events_ms,
        interval_ms=interval_ms,
    )
    if not slice_times:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for slice_time_ms in slice_times:
        fig = _plot_time_slice_summary(
            result=result,
            z_cm=z_cm,
            t_ms=t_ms,
            slice_time_ms=slice_time_ms,
        )
        label_ms = int(round(slice_time_ms))
        path = output_dir / f"{prefix}_slice_{label_ms:03d}ms.{image_format}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        written.append(path)
        plt.close(fig)
    return written


def _main_discharge_slice_times(result, *, phase_events_ms, interval_ms):
    interval_ms = float(interval_ms)
    if interval_ms <= 0.0:
        raise ValueError(f"slice interval must be positive ({interval_ms})")

    main_start = None
    afterglow_start = None
    for time_ms, phase in phase_events_ms:
        if phase == "main_discharge" and main_start is None:
            main_start = float(time_ms)
        elif phase == "afterglow" and afterglow_start is None:
            afterglow_start = float(time_ms)

    if main_start is None:
        return []
    if afterglow_start is None:
        params = getattr(result, "params", {}) or {}
        afterglow_start = main_start + 1.0e3 * float(
            params.get("tau_discharge", 0.0)
        )
    duration_ms = afterglow_start - main_start
    if not np.isfinite(duration_ms) or duration_ms < 0.0:
        return []

    count = int(np.floor(duration_ms / interval_ms + 1.0e-9))
    times = [main_start + index * interval_ms for index in range(count + 1)]
    if times and times[-1] < afterglow_start - 1.0e-9:
        times.append(afterglow_start)
    elif not times:
        times = [main_start]
    return times


def _plot_time_slice_summary(result, z_cm, t_ms, slice_time_ms):
    idx = int(np.argmin(np.abs(t_ms - slice_time_ms)))
    actual_ms = float(t_ms[idx])
    title = _plot_title(result)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)

    ne = _finite_2d(result.n)[idx]
    nn = _finite_2d(result.nn)[idx]
    axes[0, 0].plot(z_cm, ne, marker="o", markersize=3, label="ne")
    axes[0, 0].plot(z_cm, nn, marker="o", markersize=3, label="nn")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylabel("Density [cm^-3]")
    axes[0, 0].legend(loc="best")

    axes[0, 1].plot(z_cm, _finite_2d(result.Te)[idx], marker="o", markersize=3, label="Te")
    axes[0, 1].plot(z_cm, _finite_2d(result.Ti)[idx], marker="o", markersize=3, label="Ti")
    axes[0, 1].set_ylabel("Temperature [eV]")
    axes[0, 1].legend(loc="best")

    if hasattr(result, "u"):
        axes[1, 0].plot(z_cm, _finite_2d(result.u)[idx] / 100.0, marker="o", markersize=3)
    axes[1, 0].axhline(0.0, color="k", lw=0.8)
    axes[1, 0].set_ylabel("u [m/s]")

    with np.errstate(divide="ignore", invalid="ignore"):
        ion_ratio = np.where(result.nn > 0.0, result.n / result.nn, np.nan)
    axes[1, 1].plot(
        z_cm,
        _finite_2d(ion_ratio)[idx],
        marker="o",
        markersize=3,
    )
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_ylabel("ne/nn")

    for ax in axes.ravel():
        ax.set_xlabel("z [cm]")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Main Discharge Slice at {slice_time_ms:.3f} ms "
        f"(nearest saved {actual_ms:.3f} ms)\n{title}",
        fontsize=11,
    )
    return fig


def _log_tick_label(v, pos=None):
    value = 10**v
    if not np.isfinite(value) or value <= 0.0:
        return ""
    exp = int(np.floor(np.log10(value)))
    pref = value / (10**exp)
    if np.isclose(pref, 1.0):
        return rf"$10^{{{exp}}}$"
    return rf"${pref:.1f}\times10^{{{exp}}}$"


def _ratio_tick_label(v, pos=None):
    value = 10**v
    if not np.isfinite(value):
        return ""
    return f"{value:g}" if value >= 1.0 else f"{value:.3g}"


def _linear_tick_label(v, pos=None):
    return f"{v:g}"


def _nice_log_levels(vmin_log, vmax_log, mantissas=(1, 2, 5)):
    emin = int(np.floor(vmin_log))
    emax = int(np.ceil(vmax_log))
    vals = []
    for exp in range(emin, emax + 1):
        for mantissa in mantissas:
            value = np.log10(float(mantissa)) + exp
            if vmin_log - 1.0e-9 <= value <= vmax_log + 1.0e-9:
                vals.append(value)
    return np.array(vals) if vals else np.linspace(vmin_log, vmax_log, 5)


def _nice_linear_levels(vmin, vmax, target=16):
    if vmax <= vmin:
        return np.array([vmin, vmax])
    raw_step = (vmax - vmin) / max(target - 1, 1)
    exp = np.floor(np.log10(max(abs(raw_step), 1.0e-30)))
    frac = raw_step / 10**exp
    if frac <= 1.0:
        nice_frac = 1.0
    elif frac <= 2.0:
        nice_frac = 2.0
    elif frac <= 2.5:
        nice_frac = 2.5
    elif frac <= 5.0:
        nice_frac = 5.0
    else:
        nice_frac = 10.0
    step = nice_frac * 10**exp
    start = np.floor(vmin / step) * step
    stop = np.ceil(vmax / step) * step
    return np.arange(start, stop + 0.5 * step, step)


def _add_phase_lines(ax, phase_events, orientation="vertical"):
    for time, phase in phase_events:
        if phase not in {"main_discharge", "afterglow"}:
            continue
        if orientation == "horizontal":
            ax.axhline(time, color="0.55", lw=0.8, ls="--", alpha=0.65)
            ax.text(
                0.01,
                time,
                phase,
                transform=ax.get_yaxis_transform(),
                fontsize=7,
                color="0.35",
                va="bottom",
            )
        else:
            ax.axvline(time, color="0.55", lw=0.8, ls="--", alpha=0.65)
            ax.text(
                time,
                0.99,
                phase,
                transform=ax.get_xaxis_transform(),
                rotation=90,
                fontsize=7,
                color="0.35",
                va="top",
                ha="right",
            )


def _finish_time_axis(ax, time_label, phase_events):
    ax.set_xlabel(time_label)
    ax.grid(True, alpha=0.25)
    _add_phase_lines(ax, phase_events)


def _plot_summary(result, t_plot, time_label, phase_events):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    title = _plot_title(result)

    axes[0, 0].plot(t_plot, _cell_average(result, result.n), label="ne")
    axes[0, 0].plot(t_plot, _cell_average(result, result.nn, "neutral_volume_cm3"), label="nn")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylabel("Volume average density [cm^-3]")
    axes[0, 0].legend(loc="best")
    _finish_time_axis(axes[0, 0], time_label, phase_events)

    axes[0, 1].plot(t_plot, _cell_average(result, result.Te), label="Te")
    axes[0, 1].plot(t_plot, _cell_average(result, result.Ti), label="Ti")
    axes[0, 1].set_ylabel("Volume average temperature [eV]")
    axes[0, 1].legend(loc="best")
    _finish_time_axis(axes[0, 1], time_label, phase_events)

    if hasattr(result, "u"):
        axes[1, 0].plot(t_plot, _cell_average(result, result.u) / 100.0)
        axes[1, 0].axhline(0.0, color="k", lw=0.8)
    axes[1, 0].set_ylabel("Volume average u [m/s]")
    _finish_time_axis(axes[1, 0], time_label, phase_events)

    with np.errstate(divide="ignore", invalid="ignore"):
        ion_ratio = np.where(result.nn > 0.0, result.n / result.nn, np.nan)
    axes[1, 1].plot(t_plot, _cell_average(result, ion_ratio, "neutral_volume_cm3"))
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_ylabel("Volume average ne/nn")
    _finish_time_axis(axes[1, 1], time_label, phase_events)

    fig.suptitle(f"Run Summary\n{title}", fontsize=11)
    return fig


def _plot_densities(result, z_cm, t_plot, time_label, phase_events):
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(result.nn > 0.0, result.n / result.nn, np.nan)
    panels = (
        {
            "title": "Electron density",
            "values": _positive_log10(result.n),
            "cbar": r"$n_e$ [cm$^{-3}$]",
            "is_log": True,
        },
        {
            "title": "Neutral density",
            "values": _positive_log10(result.nn),
            "cbar": r"$n_n$ [cm$^{-3}$]",
            "is_log": True,
        },
        {
            "title": "Ionization fraction",
            "values": _positive_log10(ratio),
            "cbar": r"$n_e/n_n$",
            "is_log": True,
            "is_ratio": True,
            "vmin": -2.0,
            "vmax": 2.0,
        },
    )
    return _plot_contour_panels(
        panels,
        z_cm,
        t_plot,
        time_label,
        phase_events,
        f"Densities\n{_plot_title(result)}",
    )


def _plot_temperatures(result, z_cm, t_plot, time_label, phase_events):
    panels = (
        {
            "title": "Electron temperature",
            "values": _finite_2d(result.Te),
            "cbar": "Te [eV]",
            "is_log": False,
        },
        {
            "title": "Ion temperature",
            "values": _finite_2d(result.Ti),
            "cbar": "Ti [eV]",
            "is_log": False,
        },
    )
    return _plot_contour_panels(
        panels,
        z_cm,
        t_plot,
        time_label,
        phase_events,
        f"Temperatures\n{_plot_title(result)}",
    )


def _plot_velocity(result, z_cm, t_plot, time_label, phase_events):
    if not hasattr(result, "u"):
        return None
    panels = (
        {
            "title": "Parallel velocity",
            "values": _finite_2d(result.u) / 100.0,
            "cbar": "u [m/s]",
            "is_log": False,
        },
    )
    return _plot_contour_panels(
        panels,
        z_cm,
        t_plot,
        time_label,
        phase_events,
        f"Parallel Velocity\n{_plot_title(result)}",
    )


def _plot_contour_panels(panels, z_cm, t_plot, time_label, phase_events, title):
    if not panels:
        return None
    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(5.2 * len(panels), 4.2),
        constrained_layout=True,
        squeeze=False,
    )
    Z, T = np.meshgrid(z_cm, t_plot)
    for ax, panel in zip(axes[0], panels):
        _contour_panel(
            ax,
            fig,
            Z,
            T,
            panel["values"],
            panel["title"],
            panel["cbar"],
            is_log=panel.get("is_log", True),
            is_ratio=panel.get("is_ratio", False),
            vmin=panel.get("vmin"),
            vmax=panel.get("vmax"),
            xlabel="z [cm]",
            ylabel=time_label,
        )
        _add_phase_lines(ax, phase_events, orientation="horizontal")
    fig.suptitle(title, fontsize=11)
    return fig


def _contour_panel(
    ax,
    fig,
    Z_mesh,
    T_mesh,
    data,
    title,
    cbar_label,
    is_log=True,
    is_ratio=False,
    vmin=None,
    vmax=None,
    xlabel="z [cm]",
    ylabel="Time [ms]",
):
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        return

    vmin_plot = float(np.floor(np.nanmin(finite))) if vmin is None else float(vmin)
    vmax_plot = float(np.ceil(np.nanmax(finite))) if vmax is None else float(vmax)
    if vmin_plot >= vmax_plot:
        vmax_plot = vmin_plot + 1.0

    levels = np.linspace(vmin_plot, vmax_plot, 100)

    if is_ratio:
        line_values = np.array(
            [
                0.01,
                0.015,
                0.02,
                0.03,
                0.05,
                0.07,
                0.1,
                0.15,
                0.2,
                0.3,
                0.5,
                0.7,
                1.0,
                1.5,
                2.0,
                3.0,
                5.0,
                7.0,
                10.0,
                15.0,
                20.0,
                30.0,
                50.0,
                70.0,
                100.0,
            ]
        )
        label_values = np.array(
            [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
        )
        line_levels = np.log10(line_values)
        label_levels = np.log10(label_values)
        cbar_ticks = label_levels
        tick_fmt = FuncFormatter(_ratio_tick_label)
    elif is_log:
        line_levels = _nice_log_levels(vmin_plot, vmax_plot, _DENSE_MANTISSAS)
        label_levels = _nice_log_levels(
            vmin_plot,
            vmax_plot,
            (1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 7.5),
        )
        cbar_ticks = _nice_log_levels(vmin_plot, vmax_plot, (1, 1.5, 2, 3, 5, 7))
        tick_fmt = FuncFormatter(_log_tick_label)
    else:
        line_levels = _nice_linear_levels(vmin_plot, vmax_plot, target=16)
        label_levels = _nice_linear_levels(vmin_plot, vmax_plot, target=8)
        # Keep the matching LINE level, not the label one. clabel() requires
        # exact membership in the contour set's levels, while the two
        # _nice_linear_levels calls above reach the same level by different
        # step arithmetic -- a symmetric range renders zero as 0.0 at target=8
        # but as -2.2e-16 at target=16. Selecting approximately and then
        # passing the label value back raised "Specified levels don't match
        # available levels" on any panel whose data straddled zero evenly
        # (e.g. an all-but-zero velocity field).
        label_levels = np.array(
            [
                line_levels[int(np.argmin(np.abs(level - line_levels)))]
                for level in label_levels
                if np.any(np.abs(level - line_levels) < 1.0e-10)
            ]
        )
        cbar_ticks = label_levels
        tick_fmt = FuncFormatter(_linear_tick_label)

    line_levels = line_levels[
        (line_levels >= vmin_plot) & (line_levels <= vmax_plot)
    ]
    label_levels = label_levels[
        (label_levels >= vmin_plot) & (label_levels <= vmax_plot)
    ]
    cbar_ticks = cbar_ticks[(cbar_ticks >= vmin_plot) & (cbar_ticks <= vmax_plot)]

    cf = ax.contourf(
        Z_mesh,
        T_mesh,
        data,
        levels=levels,
        cmap="plasma",
        extend="both",
    )
    if line_levels.size >= 2:
        cs = ax.contour(
            Z_mesh,
            T_mesh,
            data,
            levels=line_levels,
            colors="black",
            linewidths=0.8,
            alpha=0.9,
        )
        if label_levels.size:
            ax.clabel(cs, levels=label_levels, inline=True, fontsize=6, fmt=tick_fmt)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label(cbar_label)
    if cbar_ticks.size:
        cbar.set_ticks(cbar_ticks)
    cbar.ax.yaxis.set_major_formatter(tick_fmt)


def _plot_energy_terms(result, t_plot, time_label, phase_events):
    electron = _sum_power(result, getattr(result, "electron_energy_terms_W_cm3", {}))
    ion = _sum_power(result, getattr(result, "ion_energy_terms_W_cm3", {}))
    if not electron and not ion:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    _plot_power_series(axes[0], t_plot, electron, "Electron energy terms")
    _plot_power_series(axes[1], t_plot, ion, "Ion energy terms")
    for ax in axes:
        _finish_time_axis(ax, time_label, phase_events)
    fig.suptitle(f"Integrated Energy Source Terms\n{_plot_title(result)}", fontsize=11)
    return fig


def _plot_power_series(ax, t_plot, series, title):
    for name, values in sorted(series.items()):
        if np.any(np.isfinite(values)):
            ax.plot(t_plot, values, label=name)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_ylabel("Integrated power [W]")
    ax.set_title(title)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7)


def _plot_cathode(result, t_plot, time_label, phase_events):
    cathode = getattr(result, "cathode_diagnostics", {})
    i_tot = _field(cathode, "source_I_tot")
    v_b = _field(cathode, "source_V_b")
    p_net = _field(cathode, "source_P_net")
    if i_tot is None and v_b is None and p_net is None:
        return None

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6), constrained_layout=True, sharex=True)
    if i_tot is not None:
        axes[0].plot(t_plot[: len(i_tot)], i_tot, label="source I_tot [A]", color="tab:blue")
    # Discharge voltage. Primary trace: per-solve V_b -- with electrode
    # sample smoothing on (cathode_sample_smoothing) this IS the clean
    # physical operating point (measured 2026-07-21: sigma 2.5 V, tracking
    # the ES1 plateau decline at the right absolute level); unsmoothed
    # M_n-closure runs show its sampling hash instead. circuit_V_dis_step
    # (since 2026-07-21 the save-interval dt-weighted average, agreeing
    # with V_b and the loop reconstruction on the plateau) is shown
    # 0.3-ms-averaged and faint to tame its per-save chatter. Runs saved
    # BEFORE 2026-07-21 store the biased last-step sample under the same
    # key (~25 V low on the ES1 plateau) -- do not read absolute levels
    # from the faint trace on old files.
    v_step = _field(cathode, "circuit_V_dis_step")
    if v_step is not None and not np.any(np.isfinite(v_step) & (v_step != 0.0)):
        v_step = None
    if v_b is not None or v_step is not None:
        ax_v = axes[0].twinx()
        width = 31
        # A "0.3 ms average" needs at least a kernel's worth of samples. Below
        # that np.convolve(mode="same") returns the KERNEL's length, not the
        # trace's, and the plot call fails on mismatched x/y -- reachable only
        # on very short debug runs (a production trace is orders of magnitude
        # longer than 31 saves), and only since the drive became live from
        # t = 0, which is what first made this trace nonzero there.
        if v_step is not None and len(v_step) >= width:
            kernel = np.ones(width) / width
            v_avg = np.convolve(
                np.nan_to_num(v_step, nan=float(np.nanmedian(v_step))),
                kernel,
                mode="same",
            )
            ax_v.plot(
                t_plot[: len(v_avg)],
                v_avg,
                label="circuit V_dis, 0.3 ms avg",
                color="tab:red",
                alpha=0.4,
                lw=0.9,
            )
        if v_b is not None:
            ax_v.plot(
                t_plot[: len(v_b)],
                v_b,
                label="per-solve V_b [V]",
                color="tab:orange",
                lw=1.2,
            )
        ax_v.set_ylabel("V [V]", color="tab:orange")
        ax_v.tick_params(axis="y", labelcolor="tab:orange")
        ax_v.legend(loc="lower right", fontsize=7)
    axes[0].set_ylabel("I_tot [A]", color="tab:blue")
    axes[0].tick_params(axis="y", labelcolor="tab:blue")
    axes[0].set_title("Cathode current and discharge voltage")
    _add_phase_lines(axes[0], phase_events)
    axes[0].grid(True, alpha=0.25)

    power_fields = {
        "P_net": _field(cathode, "source_P_net"),
        "P_loss": _field(cathode, "source_P_loss"),
        "P_wall": _field(cathode, "source_P_wall"),
        "P_prim": _field(cathode, "source_P_prim"),
    }
    for name, values in power_fields.items():
        if values is not None:
            axes[1].plot(t_plot[: len(values)], values, label=name)
    axes[1].set_ylabel("Power [W]")
    axes[1].set_title("Cathode powers")
    axes[1].legend(loc="best", fontsize=8)
    _finish_time_axis(axes[1], time_label, phase_events)

    fig.suptitle(f"Cathode Diagnostics\n{_plot_title(result)}", fontsize=11)
    return fig


def _field(fields, name):
    if not isinstance(fields, dict) or name not in fields:
        return None
    arr = np.asarray(fields[name], dtype=float)
    if arr.ndim != 1 or not np.any(np.isfinite(arr)):
        return None
    return arr


def _plot_phase(result, t_plot, time_label, phase_events):
    if not hasattr(result, "phase"):
        return None
    phase_names = [
        "neutral_prebreakdown",
        "pre_breakdown",
        "breakdown",
        "main_discharge",
        "afterglow",
        "post_afterglow",
        "equilibrium_puff",
        "equilibrium_off",
    ]
    phase_to_code = {phase: index for index, phase in enumerate(phase_names)}
    codes = np.asarray([phase_to_code.get(str(phase), np.nan) for phase in result.phase], dtype=float)
    fig, ax = plt.subplots(figsize=(8.5, 2.8), constrained_layout=True)
    ax.step(t_plot, codes, where="post")
    ax.set_yticks(list(phase_to_code.values()))
    ax.set_yticklabels(phase_names)
    ax.set_ylabel("Phase")
    ax.set_title("Phase Timeline")
    _finish_time_axis(ax, time_label, phase_events)
    return fig


if __name__ == "__main__":
    raise SystemExit(main())
