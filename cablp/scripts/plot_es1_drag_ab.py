"""Plot the registered fixed-end/two-baffle b_drag=0.5 versus 1.0 A/B."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np


FIGSTYLE_DIR = Path.home() / "bapsf" / "docs" / "figures"
PORT_Z_CM = np.asarray([470.05, 789.55, 1045.15, 1428.55, 1716.10])
PLATEAU = (15.0, 19.5)
BAFFLE_Z_CM = (146.2925, 980.8275)
END_START_CM = 1975.85


def load(path):
    with h5py.File(path, "r") as h5:
        time_ms = (
            h5["time"][:] - float(h5.attrs["t_breakdown_trigger"])
        ) * 1e3
        return {
            "time_ms": time_ms,
            "z_cm": h5["geometry/z_cm"][:],
            "u": h5["u"][:],
            "M": h5["M"][:],
            "drag_M": h5["rhs_terms/ion_neutral_drag/M"][:],
        }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(FIGSTYLE_DIR))
    import figstyle as fs

    runs = [
        ("A: drag 0.5", load(args.a)),
        ("B: drag 1.0", load(args.b)),
    ]
    colors = fs.palette("slide")
    fig, axes = fs.new_figure(
        "slide", figsize=(10.28, 4.65), nrows=1, ncols=3
    )

    for (label, run), color, style in zip(
        runs, colors[:2], ("-", "--")
    ):
        mask = (
            (run["time_ms"] >= PLATEAU[0])
            & (run["time_ms"] <= PLATEAU[1])
        )
        velocity = np.median(run["u"][mask], axis=0)
        tau_s = np.median(
            np.abs(run["M"][mask])
            / np.maximum(np.abs(run["drag_M"][mask]), 1e-300),
            axis=0,
        )
        length_cm = np.median(np.abs(run["u"][mask]), axis=0) * tau_s
        axes[0].plot(
            run["z_cm"], velocity / 1e5, color=color, ls=style, label=label
        )
        axes[1].semilogy(
            run["z_cm"], np.maximum(tau_s * 1e3, 1e-4),
            color=color, ls=style, label=label
        )
        axes[2].semilogy(
            run["z_cm"], np.maximum(length_cm / 100.0, 1e-4),
            color=color, ls=style, label=label
        )
        port_indices = [
            int(np.argmin(np.abs(run["z_cm"] - z))) for z in PORT_Z_CM
        ]
        axes[0].scatter(
            run["z_cm"][port_indices],
            velocity[port_indices] / 1e5,
            color=color,
            s=22,
            zorder=3,
        )

    axes[0].axhline(0.0, color=colors[3], lw=1)
    axes[0].set_ylabel(r"plateau $u_i$ [km s$^{-1}$]", fontsize=12)
    axes[1].set_ylabel("local drag time [ms]", fontsize=12)
    axes[2].set_ylabel("local stopping length [m]", fontsize=12)
    for ax in axes:
        for z in BAFFLE_Z_CM:
            ax.axvline(z, color=colors[3], ls=":", lw=1)
        ax.axvline(END_START_CM, color=colors[3], ls=":", lw=2)
        ax.set_xlim(0.0, 2130.0)
        ax.set_xlabel("axial position [cm]", fontsize=12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2, fontsize=11)

    path = args.output_dir / "es1-drag-ab-velocity-timescale.png"
    fs.save(fig, path, "slide")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
