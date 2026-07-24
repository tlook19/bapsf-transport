"""Render advisor-facing ES1 comparison figures from existing H5 runs.

This is a plotting-only campaign instrument.  It does not run LAPDSim1D or
modify the shared scorer.  The model-line bracket is deliberately explicit:
moment-2z is retained as-is, while the kinetic arms carry the low-Te ADAS
extension.  Plateau profiles show temporal bands so the kinetic limit cycle
cannot be hidden by a window average.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OVERLAY = SCRIPT_DIR / "data" / "es1_sim1d_overlay.npz"
DEFAULT_RUNS = {
    "moment": SCRIPT_DIR / "es1_nx120_m6_sq3400_2z_es1.h5",
    "kinetic": SCRIPT_DIR / "es1_nx120_m6_sq3400_k4t_lowte_es1.h5",
    "cad2": SCRIPT_DIR / "es1_nx120_m6_sq3400_k4t_lowte_cad2_es1.h5",
}
FIGSTYLE_DIR = Path.home() / "bapsf" / "docs" / "figures"
DEFAULT_OUTPUT = FIGSTYLE_DIR / "slides"
PORTS = np.array([11, 21, 29, 41, 50])
Z_PORT_CM = np.array([470.05, 789.55, 1045.15, 1428.55, 1716.10])
PLATEAU = (10.0, 19.5)


@dataclass
class Run:
    label: str
    path: Path
    time_ms: np.ndarray
    z_cm: np.ndarray
    n: np.ndarray
    Te: np.ndarray
    current_A: np.ndarray
    voltage_V: np.ndarray
    surface_T_K: np.ndarray
    cathode_ion_power_W: np.ndarray


def load_run(label: str, path: Path) -> Run:
    with h5py.File(path, "r") as h5:
        t0 = float(h5.attrs["t_breakdown_trigger"])
        time_s = h5["time"][:]
        diag = h5["cathode_diagnostics"]
        # The saved quantity is the running integral of V_dis.  Differentiate
        # across save intervals to recover the cadence-unbiased interval
        # average used by fingerprints_sim1d.py (the inductor's view).
        voltage_integral = diag["circuit_V_dis_dt_integral"][:]
        voltage_mid = np.diff(voltage_integral) / np.diff(time_s)
        voltage = np.concatenate(
            [[voltage_mid[0] if voltage_mid.size else 0.0], voltage_mid]
        )
        return Run(
            label=label,
            path=path,
            time_ms=(time_s - t0) * 1e3,
            z_cm=h5["geometry/z_cm"][:],
            n=h5["n"][:],
            Te=h5["Te"][:],
            current_A=diag["circuit_I_loop"][:],
            voltage_V=voltage,
            surface_T_K=diag["T_s_surface"][:],
            cathode_ion_power_W=diag["source_P_cathode_i"][:],
        )


def port_index(run: Run, port_index_: int) -> int:
    return int(np.argmin(np.abs(run.z_cm - Z_PORT_CM[port_index_])))


def window_mask(time_ms: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (time_ms >= lo) & (time_ms <= hi)


def te_total_sigma(te: np.ndarray, sem: np.ndarray) -> np.ndarray:
    sys_ = 0.25 * np.abs(te) + 0.20
    return np.sqrt(sem**2 + sys_**2)


def density_total_sigma(
    density: np.ndarray, sem: np.ndarray, te: np.ndarray
) -> np.ndarray:
    te_safe = np.maximum(np.abs(te), 1e-3)
    sig_te = 0.25 * te_safe + 0.20
    sys_ = np.abs(density) * np.sqrt((0.5 * sig_te / te_safe) ** 2 + 0.10**2)
    return np.sqrt(sem**2 + sys_**2)


def shape_rms_ln(model: np.ndarray, measured: np.ndarray) -> float:
    """RMS log-profile residual after removing one overall magnitude offset."""
    good = np.isfinite(model) & np.isfinite(measured) & (model > 0) & (measured > 0)
    resid = np.log(model[good] / measured[good])
    resid -= np.mean(resid)
    return float(np.sqrt(np.mean(resid**2)))


def save_figure(fs, fig, output: Path, name: str) -> Path:
    path = output / name
    fs.save(fig, path, "slide")
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--moment", type=Path, default=DEFAULT_RUNS["moment"])
    parser.add_argument("--kinetic", type=Path, default=DEFAULT_RUNS["kinetic"])
    parser.add_argument("--cad2", type=Path, default=DEFAULT_RUNS["cad2"])
    parser.add_argument("--overlay", type=Path, default=DEFAULT_OVERLAY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    for path in (args.moment, args.kinetic, args.cad2, args.overlay):
        if not path.exists():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(FIGSTYLE_DIR))
    import figstyle as fs

    runs = [
        load_run("moment-2z", args.moment),
        load_run("K4t + low-Te", args.kinetic),
        load_run("K4t, cadence/2", args.cad2),
    ]
    overlay = np.load(args.overlay)
    colors = fs.palette("slide")
    model_colors = [colors[0], colors[1], colors[2]]
    model_styles = ["-", "-", "--"]
    measured_color = "#1A1A1A"
    generated = []

    # Representative density dynamics: mid-column and far-column ports.
    fig, axes = fs.new_figure(
        "slide", figsize=(10.28, 4.55), nrows=1, ncols=2, sharex=True
    )
    for ax, port in zip(axes, (21, 41)):
        p = int(np.flatnonzero(PORTS == port)[0])
        for run, color, ls in zip(runs, model_colors, model_styles):
            iz = port_index(run, p)
            mask = window_mask(run.time_ms, 0.0, 20.0)
            ax.plot(
                run.time_ms[mask], run.n[mask, iz] / 1e13,
                color=color, ls=ls, label=run.label,
            )
        t_exp = overlay["density_time_ms"]
        mean = overlay["density_mean_cm3"][p] / 1e13
        sem = overlay["density_total_sem_cm3"][p] / 1e13
        te_exp = overlay["te_mean_ev"][p]
        sigma = density_total_sigma(mean, sem, te_exp)
        ax.fill_between(t_exp, mean - sigma, mean + sigma, color=measured_color, alpha=0.10)
        ax.plot(t_exp, mean, color=measured_color, ls=":", label="measured ES1")
        ax.set_xlim(0, 20)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("time after breakdown [ms]")
        ax.set_ylabel(r"$n_e$ [$10^{13}$ cm$^{-3}$]")
        ax.text(0.03, 0.92, f"port {port}", transform=ax.transAxes, weight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=4, fontsize=14)
    generated.append(save_figure(fs, fig, args.output_dir, "es1-advisor-bracket-density.png"))

    # Representative temperature dynamics.
    fig, axes = fs.new_figure(
        "slide", figsize=(10.28, 4.55), nrows=1, ncols=2, sharex=True
    )
    for ax, port in zip(axes, (21, 41)):
        p = int(np.flatnonzero(PORTS == port)[0])
        for run, color, ls in zip(runs, model_colors, model_styles):
            iz = port_index(run, p)
            mask = window_mask(run.time_ms, 0.0, 20.0)
            ax.plot(
                run.time_ms[mask], run.Te[mask, iz],
                color=color, ls=ls, label=run.label,
            )
        t_exp = overlay["te_time_ms"]
        mean = overlay["te_mean_ev"][p]
        sigma = te_total_sigma(mean, overlay["te_sem_ev"][p])
        ax.fill_between(t_exp, mean - sigma, mean + sigma, color=measured_color, alpha=0.10)
        ax.plot(t_exp, mean, color=measured_color, ls=":", label="measured ES1")
        ax.set_xlim(0, 20)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("time after breakdown [ms]")
        ax.set_ylabel(r"$T_e$ [eV]")
        ax.text(0.03, 0.92, f"port {port}", transform=ax.transAxes, weight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=4, fontsize=14)
    generated.append(save_figure(fs, fig, args.output_dir, "es1-advisor-bracket-temperature.png"))

    # Plateau profiles.  Median + 10--90 % temporal bands expose the cycle.
    fig, axes = fs.new_figure("slide", figsize=(10.28, 4.65), nrows=1, ncols=2)
    measured_profile = {}
    for field, ax, scale, ylabel in (
        ("n", axes[0], 1e13, r"$n_e$ [$10^{13}$ cm$^{-3}$]"),
        ("Te", axes[1], 1.0, r"$T_e$ [eV]"),
    ):
        model_at_ports = []
        for ri, (run, color, ls) in enumerate(zip(runs, model_colors, model_styles)):
            mask = window_mask(run.time_ms, *PLATEAU)
            values = getattr(run, field)[mask] / scale
            q10, q50, q90 = np.quantile(values, [0.10, 0.50, 0.90], axis=0)
            zmask = (run.z_cm >= 400) & (run.z_cm <= 1800)
            if ri < 2:
                ax.fill_between(run.z_cm[zmask], q10[zmask], q90[zmask], color=color, alpha=0.12)
            ax.plot(run.z_cm[zmask], q50[zmask], color=color, ls=ls, label=run.label)
            model_at_ports.append(
                np.array([q50[port_index(run, p)] for p in range(len(PORTS))])
            )
        if field == "n":
            t_exp = overlay["density_time_ms"]
            exp = overlay["density_mean_cm3"] / scale
            sem = overlay["density_total_sem_cm3"] / scale
            te = overlay["te_mean_ev"]
            sigma = density_total_sigma(exp, sem, te)
        else:
            t_exp = overlay["te_time_ms"]
            exp = overlay["te_mean_ev"]
            sigma = te_total_sigma(exp, overlay["te_sem_ev"])
        emask = window_mask(t_exp, *PLATEAU)
        exp_profile = np.nanmean(exp[:, emask], axis=1)
        exp_sigma = np.sqrt(np.nanmean(sigma[:, emask] ** 2, axis=1))
        measured_profile[field] = exp_profile
        ax.errorbar(
            Z_PORT_CM, exp_profile, yerr=exp_sigma, fmt="o", ms=8,
            color=measured_color, ecolor=measured_color, capsize=4,
            label="measured ES1",
        )
        shape = [shape_rms_ln(v, exp_profile) for v in model_at_ports]
        ax.text(
            0.03, 0.97,
            "shape RMS$_{ln}$\nM / K4t / cadence = "
            + " / ".join(f"{v:.2f}" for v in shape),
            transform=ax.transAxes, va="top", fontsize=14,
        )
        ax.set_xlim(400, 1800)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("axial position [cm]")
        ax.set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=4, fontsize=14)
    generated.append(save_figure(fs, fig, args.output_dir, "es1-advisor-bracket-profiles.png"))

    # Discharge current and voltage.
    fig, axes = fs.new_figure(
        "slide", figsize=(10.28, 4.55), nrows=1, ncols=2, sharex=True
    )
    for run, color, ls in zip(runs, model_colors, model_styles):
        mask = window_mask(run.time_ms, 0.0, 20.0)
        axes[0].plot(run.time_ms[mask], run.current_A[mask] / 1e3, color=color, ls=ls, label=run.label)
        axes[1].plot(run.time_ms[mask], run.voltage_V[mask], color=color, ls=ls, label=run.label)
    t_exp = overlay["discharge_time_ms"]
    emask = window_mask(t_exp, 0.0, 20.0)
    current = overlay["discharge_current_mean_a"] / 1e3
    current_sem = overlay["discharge_current_sem_a"] / 1e3
    voltage = overlay["discharge_voltage_positive_mean_v"]
    voltage_sem = overlay["discharge_voltage_sem_v"]
    axes[0].fill_between(t_exp[emask], (current-current_sem)[emask], (current+current_sem)[emask], color=measured_color, alpha=0.10)
    axes[0].plot(t_exp[emask], current[emask], color=measured_color, ls=":", label="measured ES1")
    axes[1].fill_between(t_exp[emask], (voltage-voltage_sem)[emask], (voltage+voltage_sem)[emask], color=measured_color, alpha=0.10)
    axes[1].plot(t_exp[emask], voltage[emask], color=measured_color, ls=":", label="measured ES1")
    axes[0].set_ylabel("discharge current [kA]")
    axes[1].set_ylabel("discharge voltage [V]")
    for ax in axes:
        ax.set_xlim(0, 20)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("time after breakdown [ms]")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=4, fontsize=14)
    generated.append(save_figure(fs, fig, args.output_dir, "es1-advisor-bracket-iv.png"))

    # Cathode response: surface temperature and accepted-state ion power.
    fig, axes = fs.new_figure(
        "slide", figsize=(10.28, 4.55), nrows=1, ncols=2, sharex=True
    )
    for run, color, ls in zip(runs, model_colors, model_styles):
        mask = window_mask(run.time_ms, 0.0, 20.0)
        axes[0].plot(run.time_ms[mask], run.surface_T_K[mask], color=color, ls=ls, label=run.label)
        axes[1].plot(run.time_ms[mask], run.cathode_ion_power_W[mask] / 1e3, color=color, ls=ls, label=run.label)
    axes[0].set_ylabel("surface temperature [K]")
    axes[1].set_ylabel("cathode ion power [kW]")
    for ax in axes:
        ax.set_xlim(0, 20)
        ax.set_xlabel("time after breakdown [ms]")
    surface_values = np.concatenate(
        [run.surface_T_K[window_mask(run.time_ms, 0.0, 20.0)] for run in runs]
    )
    axes[0].set_ylim(np.nanmin(surface_values) - 0.4, np.nanmax(surface_values) + 0.4)
    axes[1].set_ylim(bottom=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3, fontsize=14)
    generated.append(save_figure(fs, fig, args.output_dir, "es1-advisor-bracket-cathode.png"))

    # Afterglow Isat-shape proxy at a representative mid and far port.
    fig, axes = fs.new_figure(
        "slide", figsize=(10.28, 4.55), nrows=1, ncols=2, sharex=True, sharey=True
    )
    for ax, port in zip(axes, (21, 50)):
        p = int(np.flatnonzero(PORTS == port)[0])
        for run, color, ls in zip(runs, model_colors, model_styles):
            iz = port_index(run, p)
            proxy = run.n[:, iz] * np.sqrt(np.maximum(run.Te[:, iz], 0.0))
            ref = float(np.interp(20.0, run.time_ms, proxy))
            mask = window_mask(run.time_ms, 20.0, 25.0)
            ax.semilogy(run.time_ms[mask], np.maximum(proxy[mask] / ref, 1e-4), color=color, ls=ls, label=run.label)
        t_exp = overlay["isat_decay_time_ms"]
        exp = overlay["isat_decay_mean_a"][p]
        sem = overlay["isat_decay_sem_a"][p]
        good = np.isfinite(exp) & (exp > 0)
        ref_exp = float(exp[np.flatnonzero(good)[0]])
        ax.fill_between(t_exp[good], np.maximum((exp-sem)[good]/ref_exp, 1e-4), (exp+sem)[good]/ref_exp, color=measured_color, alpha=0.10)
        ax.semilogy(t_exp[good], exp[good]/ref_exp, color=measured_color, ls=":", label="measured ES1")
        ax.set_xlim(20, 25)
        ax.set_ylim(1e-3, 1.3)
        ax.set_xlabel("time after breakdown [ms]")
        ax.set_ylabel(r"normalized $I_{sat}$ proxy")
        ax.text(0.04, 0.10, f"port {port}", transform=ax.transAxes, weight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=4, fontsize=14)
    generated.append(save_figure(fs, fig, args.output_dir, "es1-advisor-bracket-afterglow.png"))

    # Inductive current tail; logarithmic scale makes creep/floor behavior visible.
    fig, ax = fs.new_figure("slide", figsize=(6.88, 4.65))
    for run, color, ls in zip(runs, model_colors, model_styles):
        mask = window_mask(run.time_ms, 19.5, 22.2)
        ax.semilogy(run.time_ms[mask], np.maximum(run.current_A[mask], 1.0), color=color, ls=ls, label=run.label)
    tail_t_exp = overlay["discharge_time_ms"]
    emask = window_mask(tail_t_exp, 19.5, 22.2)
    ax.semilogy(tail_t_exp[emask], np.maximum(overlay["discharge_current_mean_a"][emask], 1.0), color=measured_color, ls=":", label="measured ES1")
    ax.set_xlim(19.5, 22.2)
    ax.set_ylim(1, 5000)
    ax.set_xlabel("time after breakdown [ms]")
    ax.set_ylabel("discharge current [A]")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2, fontsize=14)
    generated.append(save_figure(fs, fig, args.output_dir, "es1-advisor-bracket-tail.png"))

    # Cadence discriminator scalars, as the numerical-validation slide.
    period = np.array([[2.935, 2.940, 2.930, 2.930, 2.920], [3.020, 3.000, 3.010, 3.020, 3.020]])
    amplitude = np.array([[0.290, 0.611, 1.147, 2.009, 1.642], [0.301, 0.639, 1.174, 1.968, 1.659]])
    fig, axes = fs.new_figure("slide", figsize=(10.28, 4.55), nrows=1, ncols=2)
    for vals, color, ls, label in zip(period, model_colors[1:], model_styles[1:], [runs[1].label, runs[2].label]):
        axes[0].plot(PORTS, vals, marker="o", ms=9, color=color, ls=ls, label=label)
    for vals, color, ls, label in zip(amplitude, model_colors[1:], model_styles[1:], [runs[1].label, runs[2].label]):
        axes[1].plot(PORTS, vals, marker="o", ms=9, color=color, ls=ls, label=label)
    axes[0].set_ylabel("cycle period [ms]")
    axes[0].set_ylim(2.85, 3.08)
    axes[1].set_ylabel("density amplitude\n" + r"[$10^{13}$ cm$^{-3}$]")
    axes[1].set_ylim(bottom=0)
    for ax in axes:
        ax.set_xlabel("port")
        ax.set_xticks(PORTS)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2, fontsize=14)
    generated.append(save_figure(fs, fig, args.output_dir, "es1-advisor-bracket-cadence.png"))

    for path in generated:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
