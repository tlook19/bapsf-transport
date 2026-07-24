"""Render the fixed-drive ES1 expanded-end geometry discriminator.

Plotting only: consumes the existing moment-2z baseline plus the provisional
mild-flare and moderate-flare H5 artifacts.  No scorer or simulation driver is
modified.  The expanded-end dimensions are provisional until CAD/B(z) arrive.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
FIGSTYLE_DIR = Path.home() / "bapsf" / "docs" / "figures"
DEFAULT_OUTPUT = FIGSTYLE_DIR / "slides"
DEFAULT_OVERLAY = SCRIPT_DIR / "data" / "es1_sim1d_overlay.npz"
DEFAULT_RUNS = {
    "base": SCRIPT_DIR / "es1_nx120_m6_sq3400_2z_es1.h5",
    "vessel": SCRIPT_DIR / "es1_nx120_m6_sq3400_2z_g1_endvessel_es1.h5",
    "flare": SCRIPT_DIR / "es1_nx120_m6_sq3400_2z_g2_endflare50_es1.h5",
}
PORTS = np.array([11, 21, 29, 41, 50])
Z_PORT_CM = np.array([470.05, 789.55, 1045.15, 1428.55, 1716.10])
PLATEAU = (15.0, 19.5)
END_START_CM = 1975.85


@dataclass
class Run:
    label: str
    path: Path
    time_ms: np.ndarray
    z_cm: np.ndarray
    roles: np.ndarray
    n: np.ndarray
    nn: np.ndarray
    nn_a: np.ndarray
    Te: np.ndarray
    u: np.ndarray
    Rp_cm: np.ndarray
    Rm_cm: np.ndarray
    Vp_cm3: np.ndarray
    Vn_cm3: np.ndarray
    current_A: np.ndarray
    voltage_V: np.ndarray


def load_run(label: str, path: Path) -> Run:
    with h5py.File(path, "r") as h5:
        time_s = h5["time"][:]
        t0 = float(h5.attrs["t_breakdown_trigger"])
        diag = h5["cathode_diagnostics"]
        integral = diag["circuit_V_dis_dt_integral"][:]
        mid = np.diff(integral) / np.diff(time_s)
        voltage = np.concatenate([[mid[0] if mid.size else 0.0], mid])
        nn = h5["nn"][:]
        nn_a = h5["nn_a"][:] if "nn_a" in h5 else nn.copy()
        geometry = h5["geometry"]
        return Run(
            label=label,
            path=path,
            time_ms=(time_s - t0) * 1e3,
            z_cm=geometry["z_cm"][:],
            roles=np.asarray([x.decode() for x in geometry["cell_role"][:]]),
            n=h5["n"][:],
            nn=nn,
            nn_a=nn_a,
            Te=h5["Te"][:],
            u=h5["u"][:],
            Rp_cm=geometry["Rp_cm"][:],
            Rm_cm=geometry["Rm_cm"][:],
            Vp_cm3=geometry["plasma_volume_cm3"][:],
            Vn_cm3=geometry["neutral_volume_cm3"][:],
            current_A=diag["circuit_I_loop"][:],
            voltage_V=voltage,
        )


def window(t, lo, hi):
    return (t >= lo) & (t <= hi)


def port_index(run, p):
    return int(np.argmin(np.abs(run.z_cm - Z_PORT_CM[p])))


def te_sigma(te, sem):
    return np.sqrt(sem**2 + (0.25 * np.abs(te) + 0.20) ** 2)


def density_sigma(n, sem, te):
    te_safe = np.maximum(np.abs(te), 1e-3)
    sig_te = 0.25 * te_safe + 0.20
    systematic = np.abs(n) * np.sqrt((0.5 * sig_te / te_safe) ** 2 + 0.10**2)
    return np.sqrt(sem**2 + systematic**2)


def shape_rms_ln(model, measured):
    good = (
        np.isfinite(model)
        & np.isfinite(measured)
        & (model > 0.0)
        & (measured > 0.0)
    )
    residual = np.log(model[good] / measured[good])
    residual -= np.mean(residual)
    return float(np.sqrt(np.mean(residual**2)))


def binned_median(time_ms, values, width_ms=0.20):
    edges = np.arange(0.0, 20.0 + width_ms, width_ms)
    x, y = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (time_ms >= lo) & (time_ms < hi) & np.isfinite(values)
        if np.any(mask):
            x.append(0.5 * (lo + hi))
            y.append(float(np.median(values[mask])))
    return np.asarray(x), np.asarray(y)


def save(fs, fig, output, name):
    path = output / name
    fs.save(fig, path, "slide")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_RUNS["base"])
    parser.add_argument("--vessel", type=Path, default=DEFAULT_RUNS["vessel"])
    parser.add_argument("--flare", type=Path, default=DEFAULT_RUNS["flare"])
    parser.add_argument("--base-label", default="moment-2z (inherited)")
    parser.add_argument(
        "--vessel-label", default="vessel + 15→18 cm (unrecal.)"
    )
    parser.add_argument("--flare-label", default="50 cm flare (unrecal.)")
    parser.add_argument(
        "--layout-labels",
        nargs=3,
        default=(
            r"moment-2z: $R_p$",
            r"15→18 cm (unrecal.): $R_p$",
            r"50 cm flare (unrecal.): $R_p$",
        ),
    )
    parser.add_argument(
        "--shape-labels",
        nargs=3,
        default=("base", "15→18", "50 cm"),
    )
    parser.add_argument("--overlay", type=Path, default=DEFAULT_OVERLAY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    for path in (args.base, args.vessel, args.flare, args.overlay):
        if not path.exists():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(FIGSTYLE_DIR))
    import figstyle as fs

    runs = [
        load_run(args.base_label, args.base),
        load_run(args.vessel_label, args.vessel),
        load_run(args.flare_label, args.flare),
    ]
    overlay = np.load(args.overlay)
    colors = fs.palette("slide")
    line_colors = [colors[0], colors[1], colors[2]]
    line_styles = ["-", "-", "--"]
    measured = "#1A1A1A"
    generated = []

    # Full per-port density and temperature traces, following the campaign's
    # visual-inspection convention: every ES1 port and its measured overlay.
    fig, axes = fs.new_figure(
        "slide",
        figsize=(10.28, 13.2),
        nrows=len(PORTS),
        ncols=2,
        sharex=True,
    )
    for p, port in enumerate(PORTS):
        density_ax, te_ax = axes[p]
        for run, color, style in zip(runs, line_colors, line_styles):
            iz = port_index(run, p)
            mask = window(run.time_ms, 0.0, 20.0)
            density_ax.plot(
                run.time_ms[mask],
                run.n[mask, iz] / 1e13,
                color=color,
                ls=style,
                label=run.label,
            )
            te_ax.plot(
                run.time_ms[mask],
                run.Te[mask, iz],
                color=color,
                ls=style,
                label=run.label,
            )
        exp_t = overlay["density_time_ms"]
        exp_n = overlay["density_mean_cm3"][p] / 1e13
        exp_sem = overlay["density_total_sem_cm3"][p] / 1e13
        exp_te = overlay["te_mean_ev"][p]
        density_unc = density_sigma(exp_n, exp_sem, exp_te)
        density_ax.fill_between(
            exp_t,
            np.maximum(exp_n - density_unc, 0.0),
            exp_n + density_unc,
            color=measured,
            alpha=0.10,
        )
        density_ax.plot(
            exp_t, exp_n, color=measured, ls=":", label="measured ES1"
        )

        te_t = overlay["te_time_ms"]
        te_sem = overlay["te_sem_ev"][p]
        te_unc = te_sigma(exp_te, te_sem)
        te_ax.fill_between(
            te_t,
            np.maximum(exp_te - te_unc, 0.0),
            exp_te + te_unc,
            color=measured,
            alpha=0.10,
        )
        te_ax.plot(te_t, exp_te, color=measured, ls=":", label="measured ES1")

        for ax in (density_ax, te_ax):
            ax.set_xlim(0.0, 20.0)
            ax.set_ylim(bottom=0.0)
            ax.text(
                0.03,
                0.84,
                f"port {port}",
                transform=ax.transAxes,
                weight="bold",
                fontsize=12,
            )
        density_ax.set_ylabel(r"$n_e$ [$10^{13}$ cm$^{-3}$]")
        te_ax.set_ylabel(r"$T_e$ [eV]")
    for ax in axes[-1]:
        ax.set_xlabel("time after breakdown [ms]")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2, fontsize=11)
    generated.append(
        save(
            fs,
            fig,
            args.output_dir,
            "es1-end-geometry-port-timeseries.png",
        )
    )

    # Plateau profiles with temporal bands and shape-only RMS.
    fig, axes = fs.new_figure("slide", figsize=(10.28, 4.65), nrows=1, ncols=2)
    for field, scale, ax, ylabel in (
        ("n", 1e13, axes[0], r"$n_e$ [$10^{13}$ cm$^{-3}$]"),
        ("Te", 1.0, axes[1], r"$T_e$ [eV]"),
    ):
        at_ports = []
        for run, color, style in zip(runs, line_colors, line_styles):
            mask = window(run.time_ms, *PLATEAU)
            values = getattr(run, field)[mask] / scale
            q10, q50, q90 = np.quantile(values, [0.10, 0.50, 0.90], axis=0)
            ax.fill_between(run.z_cm, q10, q90, color=color, alpha=0.09)
            ax.plot(run.z_cm, q50, color=color, ls=style, label=run.label)
            at_ports.append(
                np.asarray([q50[port_index(run, p)] for p in range(len(PORTS))])
            )
        if field == "n":
            exp_t = overlay["density_time_ms"]
            exp = overlay["density_mean_cm3"] / scale
            sem = overlay["density_total_sem_cm3"] / scale
            sigma = density_sigma(exp, sem, overlay["te_mean_ev"])
        else:
            exp_t = overlay["te_time_ms"]
            exp = overlay["te_mean_ev"]
            sigma = te_sigma(exp, overlay["te_sem_ev"])
        exp_mask = window(exp_t, *PLATEAU)
        exp_profile = np.nanmean(exp[:, exp_mask], axis=1)
        exp_sigma = np.sqrt(np.nanmean(sigma[:, exp_mask] ** 2, axis=1))
        ax.errorbar(
            Z_PORT_CM,
            exp_profile,
            yerr=exp_sigma,
            fmt="o",
            ms=8,
            capsize=4,
            color=measured,
            ecolor=measured,
            label="measured ES1",
        )
        shape = [shape_rms_ln(values, exp_profile) for values in at_ports]
        ax.text(
            0.03,
            0.97,
            "shape RMS$_{ln}$\n"
            + "\n".join(
                f"{label}: {value:.2f}"
                for label, value in zip(args.shape_labels, shape)
            ),
            transform=ax.transAxes,
            va="top",
            fontsize=12,
        )
        ax.axvspan(END_START_CM, 2130.0, color=colors[3], alpha=0.06)
        ax.axvline(END_START_CM, color=colors[3], ls=":", lw=2)
        ax.set_xlim(400.0, 2130.0)
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel("axial position [cm]")
        ax.set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2, fontsize=11)
    generated.append(save(fs, fig, args.output_dir, "es1-end-geometry-profiles.png"))

    # Hardware/closure geometry: vessel and flux-tube radii.
    fig, ax = fs.new_figure("slide", figsize=(6.88, 4.65))
    for run, color, style, label in zip(
        runs, line_colors, line_styles, args.layout_labels
    ):
        ax.plot(run.z_cm, run.Rp_cm, color=color, ls=style, label=label)
    ax.plot(
        runs[-1].z_cm,
        runs[-1].Rm_cm,
        color=measured,
        ls=":",
        label=r"expanded vessel: $R_m$",
    )
    ax.axvline(END_START_CM, color=colors[3], ls=":", lw=2)
    ax.set_xlim(1750.0, 2130.0)
    ax.set_ylim(0.0, 110.0)
    ax.set_xlabel("axial position [cm]")
    ax.set_ylabel("radius [cm]")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2, fontsize=11)
    generated.append(save(fs, fig, args.output_dir, "es1-end-geometry-layout.png"))

    # Direct test of the user's flare expectation: density and parallel speed.
    fig, axes = fs.new_figure(
        "slide", figsize=(10.28, 4.55), nrows=1, ncols=2, sharex=True
    )
    for run, color, style in zip(runs, line_colors, line_styles):
        mask = window(run.time_ms, *PLATEAU)
        n50 = np.median(run.n[mask], axis=0) / 1e13
        u50 = np.median(run.u[mask], axis=0) / 1e5
        axes[0].plot(run.z_cm, n50, color=color, ls=style, label=run.label)
        axes[1].plot(run.z_cm, u50, color=color, ls=style, label=run.label)
    for ax in axes:
        ax.axvline(END_START_CM, color=colors[3], ls=":", lw=2)
        ax.axhline(0.0, color=measured, lw=1)
        ax.set_xlim(1750.0, 2130.0)
        ax.set_xlabel("axial position [cm]")
    axes[0].set_ylim(bottom=0.0)
    axes[0].set_ylabel(r"plateau $n_e$ [$10^{13}$ cm$^{-3}$]")
    axes[1].set_ylabel(r"plateau $u_\parallel$ [km s$^{-1}$]")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2, fontsize=11)
    generated.append(save(fs, fig, args.output_dir, "es1-end-geometry-flow.png"))

    # Neutral reservoir: end-region total inventory and annulus profile.
    fig, axes = fs.new_figure("slide", figsize=(10.28, 4.55), nrows=1, ncols=2)
    for run, color, style in zip(runs, line_colors, line_styles):
        end = np.isin(run.roles, np.asarray(["end", "collector"]))
        ann_volume = np.maximum(run.Vn_cm3 - run.Vp_cm3, 0.0)
        inventory = np.sum(
            run.nn[:, end] * run.Vp_cm3[end]
            + run.nn_a[:, end] * ann_volume[end],
            axis=1,
        )
        mask = window(run.time_ms, 0.0, 20.0)
        axes[0].plot(
            run.time_ms[mask],
            inventory[mask] / 1e19,
            color=color,
            ls=style,
            label=run.label,
        )
        plateau = window(run.time_ms, *PLATEAU)
        profile = np.median(run.nn_a[plateau], axis=0)
        axes[1].semilogy(
            run.z_cm,
            np.maximum(profile, 1e7),
            color=color,
            ls=style,
            label=run.label,
        )
    axes[0].set_xlim(0.0, 20.0)
    axes[0].set_ylim(bottom=0.0)
    axes[0].set_xlabel("time after breakdown [ms]")
    axes[0].set_ylabel(r"end neutral inventory [$10^{19}$]")
    axes[1].axvline(END_START_CM, color=colors[3], ls=":", lw=2)
    axes[1].set_xlim(400.0, 2130.0)
    axes[1].set_xlabel("axial position [cm]")
    axes[1].set_ylabel(r"plateau annulus $n_n$ [cm$^{-3}$]")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3, fontsize=14)
    generated.append(save(fs, fig, args.output_dir, "es1-end-geometry-neutrals.png"))

    # Drive traces; voltage uses a 0.2 ms median only for legibility.
    fig, axes = fs.new_figure(
        "slide", figsize=(10.28, 4.55), nrows=1, ncols=2, sharex=True
    )
    for run, color, style in zip(runs, line_colors, line_styles):
        mask = window(run.time_ms, 0.0, 20.0)
        axes[0].plot(
            run.time_ms[mask],
            run.current_A[mask] / 1e3,
            color=color,
            ls=style,
            label=run.label,
        )
        tv, vv = binned_median(run.time_ms, run.voltage_V)
        axes[1].plot(tv, vv, color=color, ls=style, label=run.label)
    exp_t = overlay["discharge_time_ms"]
    exp_mask = window(exp_t, 0.0, 20.0)
    exp_i = overlay["discharge_current_mean_a"] / 1e3
    exp_i_sem = overlay["discharge_current_sem_a"] / 1e3
    exp_v = overlay["discharge_voltage_positive_mean_v"]
    exp_v_sem = overlay["discharge_voltage_sem_v"]
    axes[0].fill_between(
        exp_t[exp_mask],
        (exp_i - exp_i_sem)[exp_mask],
        (exp_i + exp_i_sem)[exp_mask],
        color=measured,
        alpha=0.10,
    )
    axes[0].plot(exp_t[exp_mask], exp_i[exp_mask], color=measured, ls=":", label="measured ES1")
    axes[1].fill_between(
        exp_t[exp_mask],
        (exp_v - exp_v_sem)[exp_mask],
        (exp_v + exp_v_sem)[exp_mask],
        color=measured,
        alpha=0.10,
    )
    axes[1].plot(exp_t[exp_mask], exp_v[exp_mask], color=measured, ls=":", label="measured ES1")
    axes[0].set_ylabel("discharge current [kA]")
    axes[1].set_ylabel("discharge voltage [V]")
    for ax in axes:
        ax.set_xlim(0.0, 20.0)
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel("time after breakdown [ms]")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2, fontsize=11)
    generated.append(save(fs, fig, args.output_dir, "es1-end-geometry-iv.png"))

    # All five measured afterglow traces plus the current-tail impedance
    # observable.  Keeping every port visible prevents a one-port tail verdict.
    fig, axes = fs.new_figure(
        "slide", figsize=(10.28, 9.0), nrows=3, ncols=2
    )
    decay_t = overlay["isat_decay_time_ms"]
    for p, (port, ax) in enumerate(zip(PORTS, axes.flat[:5])):
        for run, color, style in zip(runs, line_colors, line_styles):
            iz = port_index(run, p)
            proxy = run.n[:, iz] * np.sqrt(np.maximum(run.Te[:, iz], 0.0))
            reference = float(np.interp(20.0, run.time_ms, proxy))
            mask = window(run.time_ms, 20.0, 25.0)
            ax.semilogy(
                run.time_ms[mask],
                np.maximum(proxy[mask] / reference, 1e-4),
                color=color,
                ls=style,
                label=run.label,
            )
        exp = overlay["isat_decay_mean_a"][p]
        sem = overlay["isat_decay_sem_a"][p]
        good = np.isfinite(exp) & (exp > 0.0)
        reference = float(exp[np.flatnonzero(good)[0]])
        ax.fill_between(
            decay_t[good],
            np.maximum((exp - sem)[good] / reference, 1e-4),
            (exp + sem)[good] / reference,
            color=measured,
            alpha=0.10,
        )
        ax.semilogy(
            decay_t[good],
            exp[good] / reference,
            color=measured,
            ls=":",
            label="measured ES1",
        )
        ax.set_xlim(20.0, 25.0)
        ax.set_ylim(1e-3, 1.3)
        ax.set_xlabel("time after breakdown [ms]")
        ax.set_ylabel(r"normalized $I_{sat}$ proxy")
        ax.text(
            0.03,
            0.84,
            f"port {port}",
            transform=ax.transAxes,
            weight="bold",
            fontsize=12,
        )

    current_ax = axes.flat[5]
    for run, color, style in zip(runs, line_colors, line_styles):
        tail = window(run.time_ms, 19.5, 22.2)
        current_ax.semilogy(
            run.time_ms[tail],
            np.maximum(run.current_A[tail], 1.0),
            color=color,
            ls=style,
            label=run.label,
        )
    drive_t = overlay["discharge_time_ms"]
    tail = window(drive_t, 19.5, 22.2)
    drive_current = overlay["discharge_current_mean_a"][tail]
    drive_sem = overlay["discharge_current_sem_a"][tail]
    current_ax.fill_between(
        drive_t[tail],
        np.maximum(drive_current - drive_sem, 1.0),
        drive_current + drive_sem,
        color=measured,
        alpha=0.10,
    )
    current_ax.semilogy(
        drive_t[tail],
        np.maximum(drive_current, 1.0),
        color=measured,
        ls=":",
        label="measured ES1",
    )
    current_ax.set_xlim(19.5, 22.2)
    current_ax.set_ylim(1.0, 5000.0)
    current_ax.set_xlabel("time after breakdown [ms]")
    current_ax.set_ylabel("discharge current [A]")
    current_ax.text(
        0.03,
        0.84,
        "drive tail",
        transform=current_ax.transAxes,
        weight="bold",
        fontsize=12,
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2, fontsize=11)
    generated.append(save(fs, fig, args.output_dir, "es1-end-geometry-afterglow-tail.png"))

    for path in generated:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
