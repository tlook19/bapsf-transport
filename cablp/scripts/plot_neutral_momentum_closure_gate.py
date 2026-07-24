"""Plot the KN2Zone and local two-momentum frozen-background gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

from cablp.solvers._sim1d.core.geometry import build_geometry
from cablp.solvers._sim1d.physics.kinetic_neutrals import M_HE
from cablp.solvers._sim1d.physics.sources import (
    neutral_wind_two_zone_factors,
)


FIGSTYLE_DIR = Path.home() / "bapsf" / "docs" / "figures"
PORT_Z_CM = np.asarray([470.05, 789.55, 1045.15, 1428.55, 1716.10])


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--kinetic", type=Path, required=True)
    parser.add_argument("--prototype", type=Path, required=True)
    parser.add_argument("--window", nargs=2, type=float, default=(15.0, 19.5))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    with h5py.File(args.run, "r") as h5:
        time_ms = (
            h5["time"][:] - float(h5.attrs["t_breakdown_trigger"])
        ) * 1e3
        mask = (time_ms >= args.window[0]) & (time_ms <= args.window[1])
        z_cm = h5["geometry/z_cm"][:]
        ui = np.median(h5["u"][mask], axis=0)
        un_mean = np.median(h5["u_n"][mask], axis=0)
        params = json.loads(h5.attrs["params_json"])
        flags = json.loads(h5.attrs["flags_json"])
    geometry = build_geometry(params, flags)
    column_factor, _ = neutral_wind_two_zone_factors(
        geometry=geometry,
        Tn_eV=float(params["Tn_fit"]),
        ion_mass_g=M_HE,
    )
    current_ratio = un_mean * column_factor / np.maximum(np.abs(ui), 1e-300)

    kinetic = np.load(args.kinetic)
    prototype = np.load(args.prototype)
    branch = "wall300K_diffuse"
    local_ratio = prototype[f"{branch}_uc"] / np.maximum(np.abs(ui), 1e-300)

    sys.path.insert(0, str(FIGSTYLE_DIR))
    import figstyle as fs

    colors = fs.palette("slide")
    fig, axes = fs.new_figure(
        "slide", figsize=(10.28, 7.6), nrows=2, ncols=1, sharex=True
    )
    axes[0].plot(
        z_cm,
        current_ratio,
        color=colors[0],
        label="current algebraic $M_n$",
    )
    axes[0].plot(
        z_cm,
        local_ratio,
        color=colors[1],
        ls="--",
        label="local two-momentum upper-survival",
    )
    kinetic_ui = np.interp(kinetic["z_cm"], z_cm, ui)
    axes[0].plot(
        kinetic["z_cm"],
        kinetic["un_col"] / np.maximum(np.abs(kinetic_ui), 1e-300),
        color=colors[2],
        label="KN2Zone first moment",
    )
    axes[0].axhline(0.5, color=colors[3], ls=":", lw=1)
    for z_port in PORT_Z_CM:
        axes[0].plot(
            z_port,
            np.interp(z_port, kinetic["z_cm"], kinetic["un_col"])
            / max(abs(np.interp(z_port, z_cm, ui)), 1e-300),
            marker="o",
            color=colors[2],
            ms=4,
        )
    axes[0].set_ylabel(r"column $u_n/u_i$")
    axes[0].set_ylim(-0.15, 1.05)
    axes[0].legend(fontsize=10)

    axes[1].semilogy(
        z_cm,
        np.maximum(prototype[f"{branch}_entrainment"], 1.0),
        color=colors[0],
        label="ion→neutral entrainment",
    )
    axes[1].semilogy(
        z_cm,
        np.maximum(prototype[f"{branch}_nu_ca"], 1.0),
        color=colors[1],
        ls="--",
        label="fast CX column→annulus",
    )
    axes[1].semilogy(
        z_cm,
        np.maximum(prototype[f"{branch}_nu_wall"], 1.0),
        color=colors[2],
        label="annulus cylindrical wall",
    )
    baffle_rate = prototype[f"{branch}_nu_baffle"]
    active = baffle_rate > 0.0
    axes[1].scatter(
        z_cm[active],
        baffle_rate[active],
        marker="s",
        color=colors[3],
        label="baffle-face damping",
        zorder=5,
    )
    axes[1].set_ylabel(r"momentum rate [s$^{-1}$]")
    axes[1].set_xlabel("axial position [cm]")
    axes[1].legend(fontsize=10)

    for ax in axes:
        for face in geometry.neutral_baffle_face_indices:
            ax.axvline(
                geometry.z_edges_cm[int(face)],
                color=colors[3],
                ls=":",
                lw=1,
            )
        ax.set_xlim(0.0, z_cm[-1] + 0.5 * geometry.length_cm[-1])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fs.save(fig, args.output, "slide")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
