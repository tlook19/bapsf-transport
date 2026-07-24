"""Plot the fixed-end/two-baffle evolved-neutral-momentum closure A/B."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

from cablp.solvers._sim1d import LAPDSim1D
from cablp.solvers._sim1d.core.geometry import build_geometry
from cablp.solvers._sim1d.physics.sources import neutral_wind_two_zone_factors


FIGSTYLE_DIR = Path.home() / "bapsf" / "docs" / "figures"
PLATEAU = (15.0, 19.5)
BAFFLE_Z_CM = (146.2925, 980.8275)
END_START_CM = 1975.85


def load(path):
    with h5py.File(path, "r") as h5:
        t = (h5["time"][:] - float(h5.attrs["t_breakdown_trigger"])) * 1e3
        run = {
            "time_ms": t,
            "z_cm": h5["geometry/z_cm"][:],
            "u": h5["u"][:],
            "M": h5["M"][:],
            "drag_M": h5["rhs_terms/ion_neutral_drag/M"][:],
            "nn": h5["nn"][:],
            "params": json.loads(h5.attrs["params_json"]),
            "flags": json.loads(h5.attrs["flags_json"]),
        }
        if "u_n" in h5:
            run["u_n"] = h5["u_n"][:]
        if "M_n_a" in h5:
            run["M_n_a"] = h5["M_n_a"][:]
            run["u_n_a"] = h5["u_n_a"][:]
            run["nn_a"] = h5["nn_a"][:]
        return run


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--mn", type=Path, required=True)
    parser.add_argument("--a-label", default="ion: no $M_n$")
    parser.add_argument("--mn-label", default="ion: evolved $M_n$")
    parser.add_argument("--neutral-label", default="column neutral wind")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(FIGSTYLE_DIR))
    import figstyle as fs

    a = load(args.a)
    mn = load(args.mn)
    geometry = build_geometry(mn["params"], mn["flags"])
    ion_mass_g = LAPDSim1D._gas_constants(mn["params"]["gas_type"])[0]
    two_momentum = "M_n_a" in mn
    colors = fs.palette("slide")
    fig, axes = fs.new_figure(
        "slide", figsize=(10.28, 8.5), nrows=2, ncols=2
    )

    masks = {
        "a": (a["time_ms"] >= PLATEAU[0]) & (a["time_ms"] <= PLATEAU[1]),
        "mn": (mn["time_ms"] >= PLATEAU[0]) & (mn["time_ms"] <= PLATEAU[1]),
    }
    ui_a = np.median(a["u"][masks["a"]], axis=0)
    ui_mn = np.median(mn["u"][masks["mn"]], axis=0)
    if two_momentum:
        un_col = np.median(mn["u_n"][masks["mn"]], axis=0)
        un_ann = np.median(mn["u_n_a"][masks["mn"]], axis=0)
        Vc = geometry.plasma_volume_cm3
        Va = geometry.neutral_volume_cm3 - Vc
        nn = np.median(mn["nn"][masks["mn"]], axis=0)
        nn_a = np.median(mn["nn_a"][masks["mn"]], axis=0)
        un_mean = (
            nn * Vc * un_col + nn_a * Va * un_ann
        ) / np.maximum(nn * Vc + nn_a * Va, 1e-300)
    else:
        column_factor, _ = neutral_wind_two_zone_factors(
            geometry, mn["params"]["Tn_fit"], ion_mass_g
        )
        un_mean = np.median(mn["u_n"][masks["mn"]], axis=0)
        un_col = un_mean * column_factor
        un_ann = None

    axes[0, 0].plot(
        a["z_cm"], ui_a / 1e5, color=colors[0], label=args.a_label
    )
    axes[0, 0].plot(
        mn["z_cm"],
        ui_mn / 1e5,
        color=colors[1],
        ls="--",
        label=args.mn_label,
    )
    axes[0, 0].plot(
        mn["z_cm"],
        un_col / 1e5,
        color=colors[2],
        label=args.neutral_label,
    )
    axes[0, 0].plot(
        mn["z_cm"],
        un_mean / 1e5,
        color=colors[3],
        ls=":",
        label="chamber-mean wind",
    )
    if un_ann is not None:
        axes[0, 0].plot(
            mn["z_cm"],
            un_ann / 1e5,
            color=colors[4],
            ls="-.",
            label="annulus neutral wind",
        )
    axes[0, 0].axhline(0.0, color=colors[3], lw=1)
    axes[0, 0].set_ylabel(r"plateau velocity [km s$^{-1}$]", fontsize=12)
    axes[0, 0].set_ylim(-5.0, 15.0)

    ratio = np.divide(
        un_col,
        ui_mn,
        out=np.full_like(un_col, np.nan),
        where=np.abs(ui_mn) > 1e3,
    )
    axes[0, 1].plot(mn["z_cm"], ratio, color=colors[2])
    axes[0, 1].axhline(1.0, color=colors[3], ls=":", lw=1)
    axes[0, 1].set_ylim(-0.2, 1.2)
    axes[0, 1].set_ylabel(r"column $u_n/u_i$", fontsize=12)

    for label, run, mask, color, style in (
        (args.a_label, a, masks["a"], colors[0], "-"),
        (args.mn_label, mn, masks["mn"], colors[1], "--"),
    ):
        tau_s = np.median(
            np.abs(run["M"][mask])
            / np.maximum(np.abs(run["drag_M"][mask]), 1e-300),
            axis=0,
        )
        axes[1, 0].semilogy(
            run["z_cm"],
            np.maximum(tau_s * 1e3, 1e-4),
            color=color,
            ls=style,
            label=label,
        )
    axes[1, 0].set_ylabel("local drag time [ms]", fontsize=12)

    for face, color, label in zip(
        geometry.neutral_baffle_face_indices,
        colors[:2],
        ("source baffle", "port-27 baffle"),
    ):
        face = int(face)
        left, right = face - 1, face
        u_col_field = mn["u_n"] if two_momentum else (
            mn["u_n"] * column_factor[None, :]
        )
        u_face = 0.5 * (u_col_field[:, left] + u_col_field[:, right])
        donor = np.where(u_face > 0.0, mn["nn"][:, left], mn["nn"][:, right])
        flux = u_face * donor * geometry.plasma_face_area_cm2[face]
        if two_momentum:
            ua_face = 0.5 * (
                mn["u_n_a"][:, left] + mn["u_n_a"][:, right]
            )
            donor_a = np.where(
                ua_face > 0.0,
                mn["nn_a"][:, left],
                mn["nn_a"][:, right],
            )
            area_a = max(
                geometry.neutral_face_area_cm2[face]
                - geometry.plasma_face_area_cm2[face],
                0.0,
            )
            flux = flux + ua_face * donor_a * area_a
        mask = (mn["time_ms"] >= 0.0) & (mn["time_ms"] <= 20.0)
        axes[1, 1].plot(
            mn["time_ms"][mask],
            flux[mask] / 1e19,
            color=color,
            label=label,
        )
    axes[1, 1].axhline(0.0, color=colors[3], lw=1)
    axes[1, 1].set_xlim(0.0, 20.0)
    axes[1, 1].set_ylabel(r"directed neutral flux [$10^{19}$ s$^{-1}$]", fontsize=12)
    axes[1, 1].set_xlabel("time after breakdown [ms]", fontsize=12)

    for ax in axes.flat[:3]:
        for z in BAFFLE_Z_CM:
            ax.axvline(z, color=colors[3], ls=":", lw=1)
        ax.axvline(END_START_CM, color=colors[3], ls=":", lw=2)
        ax.set_xlim(0.0, 2130.0)
        ax.set_xlabel("axial position [cm]", fontsize=12)
    for ax in (axes[0, 0], axes[1, 0], axes[1, 1]):
        ax.legend(fontsize=9)

    path = args.output_dir / "es1-mn-ab-wind-drag.png"
    fs.save(fig, path, "slide")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
