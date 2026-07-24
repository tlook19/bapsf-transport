"""Score the registered fixed-end/two-baffle evolved-neutral-momentum arm."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from cablp.solvers._sim1d import LAPDSim1D
from cablp.solvers._sim1d.core.geometry import build_geometry
from cablp.solvers._sim1d.physics.sources import neutral_wind_two_zone_factors
from score_es1_two_baffles import (
    Z_PORT_CM,
    crossing_time_ms,
    load,
    measured_profile,
    plateau_mask,
    port_values,
    shape_rms_ln,
)


HERE = Path(__file__).resolve().parent


def add_momentum_fields(run):
    with h5py.File(run["path"], "r") as h5:
        run["u"] = h5["u"][:]
        run["M"] = h5["M"][:]
        run["drag_M"] = h5["rhs_terms/ion_neutral_drag/M"][:]
        if "M_n" in h5:
            run["M_n"] = h5["M_n"][:]
            run["u_n"] = h5["u_n"][:]
        if "M_n_a" in h5:
            run["M_n_a"] = h5["M_n_a"][:]
            run["u_n_a"] = h5["u_n_a"][:]
            run["nn_a"] = h5["nn_a"][:]
    return run


def port_vector(run, values):
    median = np.median(values[plateau_mask(run)], axis=0)
    return np.asarray(
        [median[np.argmin(np.abs(run["z_cm"] - z))] for z in Z_PORT_CM]
    )


def drag_metrics(run):
    mask = plateau_mask(run)
    tau_s = np.median(
        np.abs(run["M"][mask])
        / np.maximum(np.abs(run["drag_M"][mask]), 1e-300),
        axis=0,
    )
    speed = np.median(np.abs(run["u"][mask]), axis=0)
    return (
        port_vector(run, np.broadcast_to(tau_s, run["u"].shape)) * 1e3,
        port_vector(run, np.broadcast_to(speed * tau_s, run["u"].shape)),
    )


def wind_diagnostics(run):
    geometry = build_geometry(run["params"], run["flags"])
    ion_mass_g = LAPDSim1D._gas_constants(run["params"]["gas_type"])[0]
    two_momentum = "M_n_a" in run
    if two_momentum:
        u_column = run["u_n"]
        u_annulus = run["u_n_a"]
        Vc = geometry.plasma_volume_cm3
        Va = geometry.neutral_volume_cm3 - Vc
        u_mean_field = (
            run["nn"] * Vc[None, :] * u_column
            + run["nn_a"] * Va[None, :] * u_annulus
        ) / np.maximum(
            run["nn"] * Vc[None, :] + run["nn_a"] * Va[None, :],
            1e-300,
        )
    else:
        column_factor, _ = neutral_wind_two_zone_factors(
            geometry,
            Tn_eV=run["params"]["Tn_fit"],
            ion_mass_g=ion_mass_g,
        )
        u_column = run["u_n"] * column_factor[None, :]
        u_annulus = None
        u_mean_field = run["u_n"]
    ui = port_vector(run, run["u"])
    u_mean = port_vector(run, u_mean_field)
    u_col = port_vector(run, u_column)
    u_ann = None if u_annulus is None else port_vector(run, u_annulus)

    mask = plateau_mask(run)
    fluxes = []
    for face in geometry.neutral_baffle_face_indices:
        face = int(face)
        left, right = face - 1, face
        u_face = 0.5 * (u_column[:, left] + u_column[:, right])
        donor = np.where(u_face > 0.0, run["nn"][:, left], run["nn"][:, right])
        column_flux = (
            u_face * donor * geometry.plasma_face_area_cm2[face]
        )
        annulus_flux = np.zeros_like(column_flux)
        if u_annulus is not None:
            ua_face = 0.5 * (u_annulus[:, left] + u_annulus[:, right])
            donor_a = np.where(
                ua_face > 0.0,
                run["nn_a"][:, left],
                run["nn_a"][:, right],
            )
            area_a = max(
                geometry.neutral_face_area_cm2[face]
                - geometry.plasma_face_area_cm2[face],
                0.0,
            )
            annulus_flux = ua_face * donor_a * area_a
        fluxes.append(
            (
                float(geometry.z_edges_cm[face]),
                float(np.median(column_flux[mask])),
                float(np.median(annulus_flux[mask])),
                float(geometry.plasma_face_area_cm2[face]),
                float(geometry.neutral_face_area_cm2[face]),
            )
        )
    return ui, u_mean, u_col, u_ann, fluxes


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--mn", type=Path, required=True)
    parser.add_argument(
        "--overlay", type=Path, default=HERE / "data" / "es1_sim1d_overlay.npz"
    )
    args = parser.parse_args(argv)
    overlay = np.load(args.overlay)
    measured_n = measured_profile(overlay, "n")
    measured_te = measured_profile(overlay, "Te")
    results = {}

    for label, path in (("A_no_Mn", args.a), ("B_evolved_Mn", args.mn)):
        run = add_momentum_fields(load(path))
        n = port_values(run, "n")
        te = port_values(run, "Te")
        speed = port_vector(run, np.abs(run["u"])) / 1e5
        tau_ms, length_cm = drag_metrics(run)
        results[label] = {
            "n_shape": shape_rms_ln(n, measured_n),
            "te_shape": shape_rms_ln(te, measured_te),
            "n_median": float(np.median(n / measured_n)),
        }
        print(f"\n[{label}] {path}")
        print(
            "plateau_current_A="
            f"{np.mean(run['current_A'][plateau_mask(run)]):.6f}"
        )
        print("density_ratios=" + ",".join(f"{x:.6f}" for x in n / measured_n))
        print(f"density_shape_rms_ln={results[label]['n_shape']:.9f}")
        print(f"density_median_ratio={results[label]['n_median']:.9f}")
        print("Te_ratios=" + ",".join(f"{x:.6f}" for x in te / measured_te))
        print(f"Te_shape_rms_ln={results[label]['te_shape']:.9f}")
        print("ion_speed_km_s=" + ",".join(f"{x:.6f}" for x in speed))
        print("drag_tau_ms=" + ",".join(f"{x:.6f}" for x in tau_ms))
        print("drag_length_cm=" + ",".join(f"{x:.6f}" for x in length_cm))
        print(
            "tail_ms_1e_10pct_1pct="
            + ",".join(
                f"{crossing_time_ms(run, fraction):.9f}"
                for fraction in (np.exp(-1.0), 0.1, 0.01)
            )
        )
        if label == "B_evolved_Mn":
            ui, u_mean, u_col, u_ann, fluxes = wind_diagnostics(run)
            print(
                "neutral_wind_mean_km_s="
                + ",".join(f"{x / 1e5:.6f}" for x in u_mean)
            )
            print(
                "neutral_wind_column_km_s="
                + ",".join(f"{x / 1e5:.6f}" for x in u_col)
            )
            if u_ann is not None:
                print(
                    "neutral_wind_annulus_km_s="
                    + ",".join(f"{x / 1e5:.6f}" for x in u_ann)
                )
            print(
                "column_wind_over_ion="
                + ",".join(
                    f"{x:.6f}" for x in u_col / np.maximum(np.abs(ui), 1e-300)
                )
            )
            for z, flux_c, flux_a, area_col, area_total in fluxes:
                print(
                    f"baffle_wind_flux z_cm={z:.6f} "
                    f"column_particles_s={flux_c:.9e} "
                    f"annulus_particles_s={flux_a:.9e} "
                    f"total_particles_s={flux_c + flux_a:.9e} "
                    f"plasma_face_area_cm2={area_col:.9f} "
                    f"neutral_face_area_cm2={area_total:.9f}"
                )

    a = results["A_no_Mn"]
    b = results["B_evolved_Mn"]
    print("\n[A/B registered discriminators]")
    print(f"density_shape_improvement={a['n_shape'] - b['n_shape']:.9f}")
    print(f"Te_shape_degradation={b['te_shape'] - a['te_shape']:.9f}")
    print(
        "density_median_move_toward_one="
        f"{abs(1-a['n_median']) - abs(1-b['n_median']):.9f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
