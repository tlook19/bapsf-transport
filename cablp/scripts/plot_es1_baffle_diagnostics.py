"""Plot the registered ES1 first-baffle neutral and refill diagnostics."""

from pathlib import Path
import argparse
import json
import sys

import h5py
import numpy as np

from cablp.solvers._sim1d.core.geometry import build_geometry
from cablp.solvers._sim1d.physics.neutrals import two_zone_knudsen_coefficients


HERE = Path(__file__).resolve().parent
FIGSTYLE_DIR = Path.home() / "bapsf" / "docs" / "figures"
PORTS = np.array([11, 21, 29, 41, 50])
Z_PORT_CM = np.array([470.05, 789.55, 1045.15, 1428.55, 1716.10])


def load(path):
    with h5py.File(path, "r") as h5:
        params = json.loads(h5.attrs["params_json"])
        flags = json.loads(h5.attrs["flags_json"])
        return {
            "params": params,
            "flags": flags,
            "time_ms": (
                h5["time"][:] - float(h5.attrs["t_breakdown_trigger"])
            )
            * 1e3,
            "z_cm": h5["geometry/z_cm"][:],
            "nn": h5["nn"][:],
            "nn_a": h5["nn_a"][:],
            "phase": np.asarray([value.decode() for value in h5["phase"][:]]),
        }


def save(fs, fig, output, name):
    path = output / name
    fs.save(fig, path, "slide")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--baffle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(FIGSTYLE_DIR))
    import figstyle as fs

    control = load(args.control)
    baffle = load(args.baffle)
    geometry = build_geometry(baffle["params"], baffle["flags"])
    (face,) = geometry.neutral_baffle_face_indices
    face = int(face)
    baffle_z = float(geometry.z_edges_cm[face])
    _, ann_coeff = two_zone_knudsen_coefficients(
        geometry,
        Tn_K=baffle["params"]["Tn_K"],
        mu_neutral=4.0,
        clausing_scale=baffle["params"]["neutral_clausing_scale"],
    )

    colors = fs.palette("slide")
    generated = []

    # Plateau neutral profile: expose the reservoir split and the pressure jump.
    fig, axes = fs.new_figure("slide", figsize=(10.28, 4.65), nrows=1, ncols=2)
    for run, color, label in (
        (control, colors[0], "unbaffled / L6.6"),
        (baffle, colors[2], "first baffle / L8.1"),
    ):
        mask = (run["time_ms"] >= 15.0) & (run["time_ms"] <= 19.5)
        ann = np.median(run["nn_a"][mask], axis=0)
        col = np.median(run["nn"][mask], axis=0)
        axes[0].semilogy(run["z_cm"], ann, color=color, label=f"{label}: annulus")
        axes[0].semilogy(
            run["z_cm"], col, color=color, ls="--", label=f"{label}: column"
        )
        zoom = (run["z_cm"] >= 40.0) & (run["z_cm"] <= 300.0)
        axes[1].semilogy(run["z_cm"][zoom], ann[zoom], color=color)
        axes[1].semilogy(run["z_cm"][zoom], col[zoom], color=color, ls="--")
    for ax in axes:
        ax.axvline(baffle_z, color=colors[3], ls=":", lw=2)
        ax.set_xlabel("axial position [cm]")
        ax.set_ylabel(r"neutral density [cm$^{-3}$]")
        ax.set_ylim(bottom=1e8)
    axes[1].text(
        0.50,
        0.96,
        f"baffle face {baffle_z:.2f} cm",
        transform=axes[1].transAxes,
        ha="center",
        va="top",
        fontsize=12,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2, fontsize=10)
    generated.append(save(fs, fig, args.output_dir, "es1-first-baffle-neutrals.png"))

    # Annulus flux at the baffle plus five-port radial refill after switch-off.
    fig, axes = fs.new_figure("slide", figsize=(10.28, 9.5), nrows=3, ncols=2)
    flux_ax = axes[0, 0]
    for run, color, label in (
        (control, colors[0], "unbaffled / L6.6"),
        (baffle, colors[2], "first baffle / L8.1"),
    ):
        run_geometry = build_geometry(run["params"], run["flags"])
        _, run_ann_coeff = two_zone_knudsen_coefficients(
            run_geometry,
            Tn_K=run["params"]["Tn_K"],
            mu_neutral=4.0,
            clausing_scale=run["params"]["neutral_clausing_scale"],
        )
        run_face = int(np.argmin(np.abs(run_geometry.z_edges_cm - baffle_z)))
        interior = run_face - 1
        flux = run_ann_coeff[interior] * (
            run["nn_a"][:, interior] - run["nn_a"][:, interior + 1]
        )
        mask = (run["time_ms"] >= 0.0) & (run["time_ms"] <= 20.0)
        flux_ax.plot(
            run["time_ms"][mask],
            flux[mask] / 1e21,
            color=color,
            label=label,
        )
    flux_ax.set_xlabel("time after breakdown [ms]")
    flux_ax.set_ylabel(r"downstream annulus flux [$10^{21}$ s$^{-1}$]")
    flux_ax.axhline(0.0, color=colors[3], lw=1)
    flux_ax.legend(fontsize=9)

    afterglow = np.flatnonzero(baffle["phase"] == "afterglow")
    switch = int(afterglow[0])
    t0 = baffle["time_ms"][switch]
    refill_axes = list(axes.flat[1:])
    for panel, (ax, port, z_port) in enumerate(
        zip(refill_axes, PORTS, Z_PORT_CM)
    ):
        iz = int(np.argmin(np.abs(baffle["z_cm"] - z_port)))
        for run, color, style, label in (
            (control, colors[0], "-", "unbaffled"),
            (baffle, colors[2], "--", "baffle"),
        ):
            gap = run["nn_a"][:, iz] - run["nn"][:, iz]
            run_switch = int(np.flatnonzero(run["phase"] == "afterglow")[0])
            run_t0 = run["time_ms"][run_switch]
            initial = gap[run_switch]
            mask = (
                (run["time_ms"] >= run_t0)
                & (run["time_ms"] <= run_t0 + 1.0)
                & (initial > 0.0)
            )
            ax.plot(
                run["time_ms"][mask] - run_t0,
                gap[mask] / initial,
                color=color,
                ls=style,
                label=label,
            )
        ax.axhline(np.exp(-1.0), color=colors[3], ls=":", lw=1)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-0.1, 1.05)
        ax.text(
            0.05,
            0.88,
            f"port {port}",
            transform=ax.transAxes,
            fontsize=11,
        )
        ax.set_xlabel("time after switch-off [ms]")
    handles, labels = refill_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2, fontsize=10)
    generated.append(save(fs, fig, args.output_dir, "es1-first-baffle-refill.png"))

    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
