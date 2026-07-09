import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from cablp.solvers._sim1d import load_result_hdf5


def main(argv=None):
    args = _parse_args(argv)
    result = load_result_hdf5(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    time_origin = _time_origin(result, mode=args.time_origin)
    time_scale, time_label = _time_unit(args.time_unit)
    t_plot = (np.asarray(result.time, dtype=float) - time_origin) * time_scale
    z_cm = np.asarray(result.z_cm, dtype=float)
    phase_events = _shifted_phase_events(result, time_origin, time_scale)
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

    print(
        "sim1d plots written: "
        f"count={len(written)}, "
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
    return [(float((time - time_origin) * time_scale), str(phase)) for time, phase in zip(times, phases)]


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


def _add_phase_lines(ax, phase_events, orientation="vertical"):
    for time, phase in phase_events:
        if phase == "post_afterglow":
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
        ("Electron density", _positive_log10(result.n), "log10 ne [cm^-3]"),
        ("Neutral density", _positive_log10(result.nn), "log10 nn [cm^-3]"),
        ("Ionization fraction", _positive_log10(ratio), "log10 ne/nn"),
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
        ("Electron temperature", _finite_2d(result.Te), "Te [eV]"),
        ("Ion temperature", _finite_2d(result.Ti), "Ti [eV]"),
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
    panels = (("Parallel velocity", _finite_2d(result.u) / 100.0, "u [m/s]"),)
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
    for ax, (panel_title, values, cbar_label) in zip(axes[0], panels):
        finite = values[np.isfinite(values)]
        if not finite.size:
            ax.text(0.5, 0.5, "no finite data", transform=ax.transAxes, ha="center")
            continue
        mesh = ax.pcolormesh(Z, T, values, shading="auto")
        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label(cbar_label)
        ax.set_title(panel_title)
        ax.set_xlabel("z [cm]")
        ax.set_ylabel(time_label)
        _add_phase_lines(ax, phase_events, orientation="horizontal")
    fig.suptitle(title, fontsize=11)
    return fig


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
    if v_b is not None:
        ax_v = axes[0].twinx()
        ax_v.plot(t_plot[: len(v_b)], v_b, label="source V_b [V]", color="tab:orange")
        ax_v.set_ylabel("V_b [V]", color="tab:orange")
        ax_v.tick_params(axis="y", labelcolor="tab:orange")
    axes[0].set_ylabel("I_tot [A]", color="tab:blue")
    axes[0].tick_params(axis="y", labelcolor="tab:blue")
    axes[0].set_title("Cathode current and beam voltage")
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
