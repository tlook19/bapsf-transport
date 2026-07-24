"""Render visual-scoring figures for the matched ES1 two-momentum arm."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from plot_es1_end_geometry import (
    PORTS,
    Z_PORT_CM,
    density_sigma,
    load_run,
    shape_rms_ln,
    te_sigma,
    window,
)


HERE = Path(__file__).resolve().parent
FIGSTYLE_DIR = Path.home() / "bapsf" / "docs" / "figures"
PLATEAU = (15.0, 19.5)


def port_index(run, p):
    return int(np.argmin(np.abs(run.z_cm - Z_PORT_CM[p])))


def interval_voltage(run):
    return run.voltage_V


def measured_profile(overlay, field):
    if field == "n":
        values = overlay["density_mean_cm3"]
        time_ms = overlay["density_time_ms"]
    else:
        values = overlay["te_mean_ev"]
        time_ms = overlay["te_time_ms"]
    return np.nanmean(values[:, window(time_ms, *PLATEAU)], axis=1)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--algebraic", type=Path, required=True)
    parser.add_argument("--two-momentum", type=Path, required=True)
    parser.add_argument(
        "--overlay",
        type=Path,
        default=HERE / "data" / "es1_sim1d_overlay.npz",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(FIGSTYLE_DIR))
    import figstyle as fs

    runs = [
        load_run("algebraic $M_n$", args.algebraic),
        load_run("two momentum $M_c,M_a$", args.two_momentum),
    ]
    overlay = np.load(args.overlay)
    colors = fs.palette("slide")
    model_colors = colors[:2]
    model_styles = ("-", "--")
    measured = "#1A1A1A"

    # All five ports, with the same ES1 bands used by the scorer.
    fig, axes = fs.new_figure(
        "slide", figsize=(10.28, 13.2), nrows=5, ncols=2, sharex=True
    )
    for p, port in enumerate(PORTS):
        for run, color, style in zip(runs, model_colors, model_styles):
            iz = port_index(run, p)
            keep = window(run.time_ms, 0.0, 20.0)
            axes[p, 0].plot(
                run.time_ms[keep],
                run.n[keep, iz] / 1e13,
                color=color,
                ls=style,
                label=run.label,
            )
            axes[p, 1].plot(
                run.time_ms[keep],
                run.Te[keep, iz],
                color=color,
                ls=style,
                label=run.label,
            )
        tn = overlay["density_time_ms"]
        en = overlay["density_mean_cm3"][p] / 1e13
        sn = density_sigma(
            en,
            overlay["density_total_sem_cm3"][p] / 1e13,
            overlay["te_mean_ev"][p],
        )
        axes[p, 0].fill_between(
            tn, np.maximum(en - sn, 0.0), en + sn, color=measured, alpha=0.10
        )
        axes[p, 0].plot(tn, en, color=measured, ls=":", label="measured ES1")
        tt = overlay["te_time_ms"]
        et = overlay["te_mean_ev"][p]
        st = te_sigma(et, overlay["te_sem_ev"][p])
        axes[p, 1].fill_between(
            tt, np.maximum(et - st, 0.0), et + st, color=measured, alpha=0.10
        )
        axes[p, 1].plot(tt, et, color=measured, ls=":", label="measured ES1")
        for ax in axes[p]:
            ax.set_xlim(0.0, 20.0)
            ax.set_ylim(bottom=0.0)
            ax.text(
                0.03, 0.84, f"port {port}", transform=ax.transAxes,
                weight="bold", fontsize=12,
            )
        axes[p, 0].set_ylabel(r"$n_e$ [$10^{13}$ cm$^{-3}$]")
        axes[p, 1].set_ylabel(r"$T_e$ [eV]")
    for ax in axes[-1]:
        ax.set_xlabel("time after breakdown [ms]")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3, fontsize=11)
    port_path = args.output_dir / "es1_mn2mom_port_timeseries.png"
    fs.save(fig, port_path, "slide")

    # Compact scoring audit: profiles, I/V, afterglow shape, and current tail.
    fig, axes = fs.new_figure(
        "slide", figsize=(13.2, 8.5), nrows=2, ncols=3
    )
    for field, scale, ax, ylabel in (
        ("n", 1e13, axes[0, 0], r"$n_e$ [$10^{13}$ cm$^{-3}$]"),
        ("Te", 1.0, axes[0, 1], r"$T_e$ [eV]"),
    ):
        model_ports = []
        for run, color, style in zip(runs, model_colors, model_styles):
            keep = window(run.time_ms, *PLATEAU)
            values = getattr(run, field)[keep] / scale
            q10, q50, q90 = np.quantile(values, [0.10, 0.50, 0.90], axis=0)
            zkeep = (run.z_cm >= 400.0) & (run.z_cm <= 1800.0)
            ax.fill_between(
                run.z_cm[zkeep], q10[zkeep], q90[zkeep],
                color=color, alpha=0.10,
            )
            ax.plot(
                run.z_cm[zkeep], q50[zkeep],
                color=color, ls=style, label=run.label,
            )
            model_ports.append(
                np.asarray([q50[port_index(run, p)] for p in range(5)])
            )
        exp = measured_profile(overlay, field) / scale
        if field == "n":
            sig = density_sigma(
                overlay["density_mean_cm3"] / scale,
                overlay["density_total_sem_cm3"] / scale,
                overlay["te_mean_ev"],
            )
            t_exp = overlay["density_time_ms"]
        else:
            sig = te_sigma(overlay["te_mean_ev"], overlay["te_sem_ev"])
            t_exp = overlay["te_time_ms"]
        exp_sig = np.sqrt(
            np.nanmean(sig[:, window(t_exp, *PLATEAU)] ** 2, axis=1)
        )
        ax.errorbar(
            Z_PORT_CM, exp, yerr=exp_sig, fmt="o", color=measured,
            capsize=4, label="measured ES1",
        )
        scores = [shape_rms_ln(v, exp) for v in model_ports]
        ax.text(
            0.03, 0.97,
            "shape RMS$_{ln}$: " + " / ".join(f"{x:.3f}" for x in scores),
            transform=ax.transAxes, va="top", fontsize=11,
        )
        ax.set_xlim(400.0, 1800.0)
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel("axial position [cm]")
        ax.set_ylabel(ylabel)

    for run, color, style in zip(runs, model_colors, model_styles):
        keep = window(run.time_ms, 0.0, 20.0)
        axes[0, 2].plot(
            run.time_ms[keep], run.current_A[keep] / 1e3,
            color=color, ls=style, label=run.label,
        )
        axes[1, 0].plot(
            run.time_ms[keep], interval_voltage(run)[keep],
            color=color, ls=style, label=run.label,
        )
    td = overlay["discharge_time_ms"]
    kd = window(td, 0.0, 20.0)
    axes[0, 2].plot(
        td[kd], overlay["discharge_current_mean_a"][kd] / 1e3,
        color=measured, ls=":", label="measured ES1",
    )
    axes[1, 0].plot(
        td[kd], overlay["discharge_voltage_positive_mean_v"][kd],
        color=measured, ls=":", label="measured ES1",
    )
    axes[0, 2].set_ylabel("discharge current [kA]")
    axes[1, 0].set_ylabel("discharge voltage [V]")
    for ax in (axes[0, 2], axes[1, 0]):
        ax.set_xlim(0.0, 20.0)
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel("time after breakdown [ms]")

    port_colors = colors[:5]
    for p, (port, color) in enumerate(zip(PORTS, port_colors)):
        for run, style in zip(runs, model_styles):
            iz = port_index(run, p)
            proxy = run.n[:, iz] * np.sqrt(np.maximum(run.Te[:, iz], 0.0))
            ref = float(np.interp(20.0, run.time_ms, proxy))
            keep = window(run.time_ms, 20.0, 25.0)
            axes[1, 1].semilogy(
                run.time_ms[keep], np.maximum(proxy[keep] / ref, 1e-4),
                color=color, ls=style,
                label=f"p{port} {run.label}" if p == 0 else None,
            )
        te = overlay["isat_decay_time_ms"]
        ie = overlay["isat_decay_mean_a"][p]
        good = np.isfinite(ie) & (ie > 0.0)
        if np.any(good):
            axes[1, 1].semilogy(
                te[good], ie[good] / ie[good][0],
                color=color, ls=":", alpha=0.75,
            )
    axes[1, 1].set_xlim(20.0, 25.0)
    axes[1, 1].set_ylim(1e-3, 1.3)
    axes[1, 1].set_xlabel("time after breakdown [ms]")
    axes[1, 1].set_ylabel(r"normalized $I_{sat}$ proxy")
    axes[1, 1].text(
        0.03, 0.08, "color = port; dotted = ES1",
        transform=axes[1, 1].transAxes, fontsize=10,
    )

    for run, color, style in zip(runs, model_colors, model_styles):
        keep = window(run.time_ms, 19.5, 22.2)
        axes[1, 2].semilogy(
            run.time_ms[keep], np.maximum(run.current_A[keep], 1.0),
            color=color, ls=style, label=run.label,
        )
    kt = window(td, 19.5, 22.2)
    axes[1, 2].semilogy(
        td[kt], np.maximum(overlay["discharge_current_mean_a"][kt], 1.0),
        color=measured, ls=":", label="measured ES1",
    )
    axes[1, 2].set_xlim(19.5, 22.2)
    axes[1, 2].set_ylim(1.0, 5000.0)
    axes[1, 2].set_xlabel("time after breakdown [ms]")
    axes[1, 2].set_ylabel("discharge current [A]")

    axes[0, 2].legend(loc="lower right", fontsize=9)
    summary_path = args.output_dir / "es1_mn2mom_summary.png"
    fs.save(fig, summary_path, "slide")
    print(port_path)
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
