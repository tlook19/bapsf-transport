"""Plot neutral diagnostics for the registered ES1 fixed-end/two-baffle arm."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from cablp.solvers._sim1d.core.geometry import build_geometry
from cablp.solvers._sim1d.physics.neutrals import two_zone_knudsen_coefficients
from score_es1_two_baffles import PORTS, Z_PORT_CM, load, plateau_mask


FIGSTYLE_DIR = Path.home() / "bapsf" / "docs" / "figures"


def save(fs, fig, output, name):
    path = output / name
    fs.save(fig, path, "slide")
    return path


def face_flux(run, z_face):
    geometry = build_geometry(run["params"], run["flags"])
    _, annulus_coeff = two_zone_knudsen_coefficients(
        geometry,
        Tn_K=run["params"]["Tn_K"],
        mu_neutral=4.0,
        clausing_scale=run["params"]["neutral_clausing_scale"],
    )
    face = int(np.argmin(np.abs(geometry.z_edges_cm - z_face)))
    interior = face - 1
    return annulus_coeff[interior] * (
        run["nn_a"][:, interior] - run["nn_a"][:, interior + 1]
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--first-baffle", type=Path, required=True)
    parser.add_argument("--combined", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(FIGSTYLE_DIR))
    import figstyle as fs

    runs = [
        ("unbaffled / L6.6", load(args.control)),
        ("one R40 baffle / L8.1", load(args.first_baffle)),
        ("two R30 baffles + fixed end / L8.1", load(args.combined)),
    ]
    combined = runs[-1][1]
    geometry = build_geometry(combined["params"], combined["flags"])
    faces = np.asarray(geometry.neutral_baffle_face_indices, dtype=int)
    face_z = geometry.z_edges_cm[faces]
    colors = fs.palette("slide")
    styles = ["-", "-", "--"]
    generated = []

    fig, axes = fs.new_figure(
        "slide", figsize=(10.28, 8.7), nrows=2, ncols=2
    )
    for (label, run), color, style in zip(runs, colors[:3], styles):
        mask = plateau_mask(run)
        axes[0, 0].semilogy(
            run["z_cm"],
            np.median(run["nn_a"][mask], axis=0),
            color=color,
            ls=style,
            label=label,
        )
        axes[0, 1].semilogy(
            run["z_cm"],
            np.median(run["nn"][mask], axis=0),
            color=color,
            ls=style,
            label=label,
        )
    axes[0, 0].set_ylabel(r"annulus $n_n$ [cm$^{-3}$]")
    axes[0, 1].set_ylabel(r"column $n_n$ [cm$^{-3}$]")
    for ax in axes[0]:
        for z in face_z:
            ax.axvline(z, color=colors[3], ls=":", lw=2)
        ax.set_xlim(0.0, 2130.0)
        ax.set_ylim(bottom=1e7)
        ax.set_xlabel("axial position [cm]")

    for panel, z in enumerate(face_z):
        ax = axes[1, panel]
        for (label, run), color, style in zip(runs, colors[:3], styles):
            flux = face_flux(run, z)
            mask = (run["time_ms"] >= 0.0) & (run["time_ms"] <= 20.0)
            ax.plot(
                run["time_ms"][mask],
                flux[mask] / 1e21,
                color=color,
                ls=style,
                label=label,
            )
        ax.axhline(0.0, color=colors[3], lw=1)
        ax.set_xlim(0.0, 20.0)
        ax.set_xlabel("time after breakdown [ms]")
        ax.set_ylabel(r"annulus flux [$10^{21}$ s$^{-1}$]")
        ax.text(
            0.04,
            0.92,
            f"face z = {z:.2f} cm",
            transform=ax.transAxes,
            va="top",
            fontsize=12,
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2, fontsize=10)
    generated.append(
        save(fs, fig, args.output_dir, "es1-two-baffle-neutrals-flux.png")
    )

    fig, axes = fs.new_figure(
        "slide", figsize=(10.28, 9.4), nrows=3, ncols=2
    )
    inventory_ax = axes.flat[0]
    bounds = np.concatenate(([0], faces, [geometry.cells]))
    annulus_volume = np.maximum(
        combined["neutral_volume_cm3"] - combined["plasma_volume_cm3"], 0.0
    )
    region_labels = ("source reservoir", "central reservoir", "downstream + end")
    mask = (combined["time_ms"] >= 0.0) & (combined["time_ms"] <= 20.0)
    for start, stop, label, color in zip(
        bounds[:-1], bounds[1:], region_labels, colors[:3]
    ):
        inventory = np.sum(
            combined["nn_a"][:, start:stop] * annulus_volume[start:stop],
            axis=1,
        )
        inventory_ax.plot(
            combined["time_ms"][mask],
            inventory[mask] / 1e19,
            color=color,
            label=label,
        )
    inventory_ax.set_xlim(0.0, 20.0)
    inventory_ax.set_ylim(bottom=0.0)
    inventory_ax.set_xlabel("time after breakdown [ms]")
    inventory_ax.set_ylabel(r"inventory [$10^{19}$ particles]", fontsize=12)
    inventory_ax.legend(fontsize=9)

    for ax, port, z in zip(axes.flat[1:], PORTS, Z_PORT_CM):
        for (label, run), color, style in zip(runs, colors[:3], styles):
            index = int(np.argmin(np.abs(run["z_cm"] - z)))
            switch = int(np.flatnonzero(run["phase"] == "afterglow")[0])
            elapsed = run["time_ms"] - run["time_ms"][switch]
            gap = run["nn_a"][:, index] - run["nn"][:, index]
            initial = float(gap[switch])
            valid = (
                (elapsed >= 0.0)
                & (elapsed <= 1.0)
                & np.isfinite(gap)
                & (initial > 0.0)
            )
            ax.plot(
                elapsed[valid],
                gap[valid] / initial,
                color=color,
                ls=style,
                label=label,
            )
        ax.axhline(np.exp(-1.0), color=colors[3], ls=":", lw=1)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-0.1, 1.05)
        ax.set_xlabel("time after switch-off [ms]")
        ax.set_ylabel(r"gap / gap$_0$", fontsize=12)
        ax.text(
            0.05,
            0.88,
            f"port {port}",
            transform=ax.transAxes,
            fontsize=11,
        )
    handles, labels = axes.flat[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2, fontsize=10)
    generated.append(
        save(fs, fig, args.output_dir, "es1-two-baffle-reservoir-refill.png")
    )

    for path in generated:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
