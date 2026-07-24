#!/usr/bin/env python3
"""Plot the exact committed production golden trajectory without rerunning it.

The golden NPZ intentionally stores only the packed conservative plasma state,
time, and phase.  Circuit, cathode, and per-term diagnostic ledgers are not
present, so this instrument decodes the available n/nn/u/Te/Ti fields and does
not reconstruct diagnostics that were never saved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from baseline_sim1d import build_baseline_config
from cablp.funcs._adas import he_rate_temperature_range_eV
from cablp.solvers._sim1d import LAPDSim1D
from cablp.solvers._sim1d.core.geometry import PLASMA_DEAD_ROLES
from cablp.vars._cons import ev_to_erg


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BASELINE = SCRIPT_DIR / "baselines" / "production_discharge.npz"
PORTS = np.array([11, 21, 29, 41, 50])
Z_PORT_CM = np.array([470.05, 789.55, 1045.15, 1428.55, 1716.10])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=180)
    return parser


def _decode_golden(path: Path):
    sidecar = path.with_suffix(".json")
    metadata = json.loads(sidecar.read_text())
    golden = np.load(path, allow_pickle=False)
    time_s = np.asarray(golden["time"], dtype=float)
    phase = np.asarray(golden["phase"], dtype=str)
    y = np.asarray(golden["y"], dtype=float)

    params, flags = build_baseline_config()
    sim = LAPDSim1D(params, flags)
    geometry = sim.geometry
    cells = geometry.cells
    if y.shape[1] != 5 * cells:
        raise ValueError(
            f"golden width {y.shape[1]} does not match 5*{cells} cells"
        )
    packed = y.reshape((y.shape[0], 5, cells))
    n = packed[:, 0]
    nn = packed[:, 1]
    M = packed[:, 2]
    Ee = packed[:, 3]
    Ei = packed[:, 4]
    n_safe = np.maximum(n, sim.floors["n"])
    u = M / (sim.ion_mass_g * n_safe)
    Te = np.maximum(
        (2.0 / 3.0) * Ee / (n_safe * ev_to_erg), sim.floors["Te"]
    )
    Ti = np.maximum(
        (2.0 / 3.0) * Ei / (n_safe * ev_to_erg), sim.floors["Ti"]
    )
    main = np.flatnonzero(phase == "main_discharge")
    if not main.size:
        raise ValueError("golden contains no main_discharge samples")
    t0_s = float(time_s[main[0]])
    return {
        "metadata": metadata,
        "geometry": geometry,
        "time_ms": (time_s - t0_s) * 1.0e3,
        "phase": phase,
        "n": n,
        "nn": nn,
        "u": u,
        "Te": Te,
        "Ti": Ti,
    }


def _port_indices(z_cm: np.ndarray) -> np.ndarray:
    return np.array([np.argmin(np.abs(z_cm - z)) for z in Z_PORT_CM], dtype=int)


def _phase_lines(axes) -> None:
    for ax in np.ravel(axes):
        ax.axvline(0.0, color="0.45", lw=1.0, ls=":")
        ax.axvline(20.0, color="0.35", lw=1.0, ls="--")
        ax.grid(alpha=0.2)


def _first_afterglow_low_te_ms(data, indices) -> float:
    te_min_eV, _ = he_rate_temperature_range_eV()
    afterglow = np.asarray(data["phase"]) == "afterglow"
    below = np.any(np.asarray(data["Te"])[:, indices] < te_min_eV, axis=1)
    crossing = np.flatnonzero(afterglow & below)
    return float(data["time_ms"][crossing[0]]) if crossing.size else np.nan


def _plot_timeseries(data):
    t = data["time_ms"]
    z = np.asarray(data["geometry"].z_cm, dtype=float)
    idx = _port_indices(z)
    selected = (1, 4)
    colors = ("C0", "C1")
    fig, axes = plt.subplots(
        2, 2, figsize=(10.8, 6.8), sharex=True, constrained_layout=True
    )
    for p, color in zip(selected, colors):
        label = f"port {PORTS[p]} ({z[idx[p]]:.0f} cm cell)"
        axes[0, 0].plot(t, data["n"][:, idx[p]] / 1e12, color=color, label=label)
        axes[0, 1].plot(t, data["Te"][:, idx[p]], color=color, label=label)
        proxy = data["n"][:, idx[p]] * np.sqrt(np.maximum(data["Te"][:, idx[p]], 0))
        axes[1, 0].plot(
            t,
            proxy / 1e12,
            color=color,
            label=label,
        )
    live = np.array(
        [role not in PLASMA_DEAD_ROLES for role in data["geometry"].cell_role]
    )
    neutral_volume = np.asarray(data["geometry"].neutral_volume_cm3, dtype=float)
    nn_mean = np.sum(data["nn"] * neutral_volume[None, :], axis=1) / np.sum(
        neutral_volume
    )
    n_volume = np.asarray(data["geometry"].plasma_volume_cm3, dtype=float)
    n_mean = np.sum(data["n"][:, live] * n_volume[live][None, :], axis=1) / np.sum(
        n_volume[live]
    )
    axes[1, 1].plot(t, nn_mean / 1e12, color="C2", label="chamber-mean neutral")
    axes[1, 1].plot(t, n_mean / 1e12, color="C3", label="live plasma mean")

    axes[0, 0].set_ylabel(r"$n_e$ [$10^{12}$ cm$^{-3}$]")
    axes[0, 1].set_ylabel(r"$T_e$ [eV]")
    axes[1, 0].set_ylabel(r"$n_e\sqrt{T_e}$ proxy")
    axes[1, 1].set_ylabel(r"volume mean [$10^{12}$ cm$^{-3}$]")
    axes[1, 0].set_xlabel("time from main-discharge start [ms]")
    axes[1, 1].set_xlabel("time from main-discharge start [ms]")
    for ax in axes.ravel():
        ax.legend(frameon=False, fontsize=8)
    _phase_lines(axes)
    low_te_time = _first_afterglow_low_te_ms(data, idx)
    if np.isfinite(low_te_time):
        for ax in axes.ravel():
            ax.axvline(low_te_time, color="C3", lw=1.1, ls="--", alpha=0.8)
        axes[0, 1].text(
            low_te_time,
            0.98,
            r"$T_e$ below ADF11 grid",
            transform=axes[0, 1].get_xaxis_transform(),
            rotation=90,
            ha="right",
            va="top",
            color="C3",
            fontsize=8,
        )
    axes[0, 0].text(
        0.01,
        0.97,
        "main discharge",
        transform=axes[0, 0].transAxes,
        ha="left",
        va="top",
        color="0.35",
        fontsize=8,
    )
    axes[0, 0].text(
        0.79,
        0.97,
        "afterglow",
        transform=axes[0, 0].transAxes,
        ha="left",
        va="top",
        color="0.35",
        fontsize=8,
    )
    fig.suptitle(
        "Committed production golden: decoded time traces\n"
        "single-zone moment / M6 candidate; no experimental overlay",
        fontsize=12,
    )
    return fig


def _nearest_time_index(time_ms: np.ndarray, target_ms: float) -> int:
    return int(np.argmin(np.abs(time_ms - target_ms)))


def _plot_profiles(data):
    t = data["time_ms"]
    z = np.asarray(data["geometry"].z_cm, dtype=float)
    roles = np.asarray(data["geometry"].cell_role)
    show = np.array(
        [
            role not in PLASMA_DEAD_ROLES and z_value >= 0.0
            for role, z_value in zip(roles, z)
        ]
    )
    targets = (5.0, 10.0, 15.0, 19.5, 20.45)
    labels = ("5 ms", "10 ms", "15 ms", "19.5 ms", "afterglow +0.45 ms")
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(targets)))
    fig, axes = plt.subplots(
        2, 2, figsize=(10.8, 6.8), sharex=True, constrained_layout=True
    )
    for target, label, color in zip(targets, labels, colors):
        it = _nearest_time_index(t, target)
        axes[0, 0].plot(z[show] / 100.0, data["n"][it, show] / 1e12, color=color, label=label)
        axes[0, 1].plot(z[show] / 100.0, data["Te"][it, show], color=color, label=label)
        axes[1, 0].plot(z[show] / 100.0, data["u"][it, show] / 1e5, color=color, label=label)
        axes[1, 1].plot(z / 100.0, data["nn"][it] / 1e12, color=color, label=label)

    axes[0, 0].set_ylabel(r"$n_e$ [$10^{12}$ cm$^{-3}$]")
    axes[0, 1].set_ylabel(r"$T_e$ [eV]")
    axes[1, 0].set_ylabel(r"$u_\parallel$ [km/s]")
    axes[1, 1].set_ylabel(r"$n_n$ [$10^{12}$ cm$^{-3}$]")
    axes[1, 0].set_xlabel("z from cathode [m]")
    axes[1, 1].set_xlabel("z from cathode [m]")
    for ax in axes.ravel():
        ax.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle(
        "Committed production golden: axial profiles\n"
        "the 20.45 ms curve is 0.45 ms after current shutoff",
        fontsize=12,
    )
    return fig


def main() -> int:
    args = _parser().parse_args()
    data = _decode_golden(args.baseline)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "production-golden-timeseries.png": _plot_timeseries(data),
        "production-golden-profiles.png": _plot_profiles(data),
    }
    for name, fig in figures.items():
        path = args.output_dir / name
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
