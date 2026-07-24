"""Score the registered ES1 fixed-end/two-baffle geometry arm.

This is a read-only campaign instrument.  It reports the pre-registered
five-port shape metric, current tail, aperture jumps/fluxes, reservoir
inventories, and local annulus-column refill without modifying a shared scorer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from cablp.solvers._sim1d.core.geometry import build_geometry
from cablp.solvers._sim1d.physics.neutrals import two_zone_knudsen_coefficients


HERE = Path(__file__).resolve().parent
PORTS = np.asarray([11, 21, 29, 41, 50])
Z_PORT_CM = np.asarray([470.05, 789.55, 1045.15, 1428.55, 1716.10])
PLATEAU = (15.0, 19.5)


def load(path: Path):
    with h5py.File(path, "r") as h5:
        phase = np.asarray([value.decode() for value in h5["phase"][:]])
        return {
            "path": path,
            "params": json.loads(h5.attrs["params_json"]),
            "flags": json.loads(h5.attrs["flags_json"]),
            "time_ms": (
                h5["time"][:] - float(h5.attrs["t_breakdown_trigger"])
            )
            * 1e3,
            "z_cm": h5["geometry/z_cm"][:],
            "plasma_volume_cm3": h5["geometry/plasma_volume_cm3"][:],
            "neutral_volume_cm3": h5["geometry/neutral_volume_cm3"][:],
            "n": h5["n"][:],
            "Te": h5["Te"][:],
            "nn": h5["nn"][:],
            "nn_a": h5["nn_a"][:],
            "current_A": h5["cathode_diagnostics/circuit_I_loop"][:],
            "phase": phase,
        }


def plateau_mask(run):
    return (run["time_ms"] >= PLATEAU[0]) & (run["time_ms"] <= PLATEAU[1])


def port_values(run, field):
    median = np.median(run[field][plateau_mask(run)], axis=0)
    return np.asarray(
        [median[np.argmin(np.abs(run["z_cm"] - z))] for z in Z_PORT_CM]
    )


def measured_profile(overlay, field):
    if field == "n":
        time = overlay["density_time_ms"]
        values = overlay["density_mean_cm3"]
    else:
        time = overlay["te_time_ms"]
        values = overlay["te_mean_ev"]
    mask = (time >= PLATEAU[0]) & (time <= PLATEAU[1])
    return np.nanmean(values[:, mask], axis=1)


def shape_rms_ln(model, measured):
    residual = np.log(model / measured)
    residual -= np.mean(residual)
    return float(np.sqrt(np.mean(residual**2)))


def crossing_time_ms(run, fraction):
    t = run["time_ms"]
    current = run["current_A"]
    reference = float(np.interp(20.0, t, current))
    after = np.flatnonzero((t >= 20.0) & (current <= fraction * reference))
    if not after.size:
        return np.nan
    index = int(after[0])
    if index == 0:
        return float(t[index] - 20.0)
    t0, t1 = t[index - 1 : index + 1]
    i0, i1 = current[index - 1 : index + 1]
    target = fraction * reference
    if i1 == i0:
        cross = t1
    else:
        cross = t0 + (target - i0) * (t1 - t0) / (i1 - i0)
    return float(cross - 20.0)


def refill_tau_ms(run):
    switch = int(np.flatnonzero(run["phase"] == "afterglow")[0])
    elapsed = run["time_ms"] - run["time_ms"][switch]
    values = []
    for z in Z_PORT_CM:
        index = int(np.argmin(np.abs(run["z_cm"] - z)))
        gap = run["nn_a"][:, index] - run["nn"][:, index]
        initial = float(gap[switch])
        mask = (
            (elapsed >= 0.0)
            & (elapsed <= 0.6)
            & np.isfinite(gap)
            & (gap > 0.0)
            & (initial > 0.0)
        )
        slope = np.polyfit(elapsed[mask], np.log(gap[mask] / initial), 1)[0]
        values.append(float(-1.0 / slope) if slope < 0.0 else np.nan)
    return np.asarray(values)


def aperture_diagnostics(run):
    geometry = build_geometry(run["params"], run["flags"])
    _, annulus_coeff = two_zone_knudsen_coefficients(
        geometry,
        Tn_K=run["params"]["Tn_K"],
        mu_neutral=4.0,
        clausing_scale=run["params"]["neutral_clausing_scale"],
    )
    mask = plateau_mask(run)
    median_annulus = np.median(run["nn_a"][mask], axis=0)
    diagnostics = []
    for face, clear in zip(
        geometry.neutral_baffle_face_indices,
        geometry.neutral_baffle_clear_radius_cm,
    ):
        face = int(face)
        interior = face - 1
        flux = annulus_coeff[interior] * (
            run["nn_a"][:, interior] - run["nn_a"][:, interior + 1]
        )
        diagnostics.append(
            {
                "face": face,
                "z_cm": float(geometry.z_edges_cm[face]),
                "clear_radius_cm": float(clear),
                "upstream_over_downstream": float(
                    median_annulus[interior] / median_annulus[interior + 1]
                ),
                "downstream_flux_per_s": float(np.median(flux[mask])),
                "annulus_coeff_cm3_s": float(annulus_coeff[interior]),
            }
        )

    faces = np.asarray(geometry.neutral_baffle_face_indices, dtype=int)
    bounds = np.concatenate(([0], faces, [geometry.cells]))
    annulus_volume = np.maximum(
        run["neutral_volume_cm3"] - run["plasma_volume_cm3"], 0.0
    )
    inventories = []
    for start, stop in zip(bounds[:-1], bounds[1:]):
        inventory = np.sum(
            run["nn_a"][:, start:stop] * annulus_volume[start:stop], axis=1
        )
        inventories.append(float(np.median(inventory[mask])))
    return diagnostics, np.asarray(inventories)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--first-baffle", type=Path, required=True)
    parser.add_argument("--combined", type=Path, required=True)
    parser.add_argument(
        "--overlay", type=Path, default=HERE / "data" / "es1_sim1d_overlay.npz"
    )
    args = parser.parse_args(argv)

    runs = [
        ("unbaffled", load(args.control)),
        ("first-baffle", load(args.first_baffle)),
        ("fixed-end-two-baffle", load(args.combined)),
    ]
    overlay = np.load(args.overlay)
    measured_n = measured_profile(overlay, "n")
    measured_te = measured_profile(overlay, "Te")

    for label, run in runs:
        n_values = port_values(run, "n")
        te_values = port_values(run, "Te")
        current = np.mean(run["current_A"][plateau_mask(run)])
        print(f"\n[{label}] {run['path']}")
        print(f"plateau_current_A={current:.6f} ratio_to_2991={current / 2991.0:.9f}")
        print("density_ratios=" + ",".join(f"{x:.6f}" for x in n_values / measured_n))
        print(f"density_shape_rms_ln={shape_rms_ln(n_values, measured_n):.9f}")
        print("Te_ratios=" + ",".join(f"{x:.6f}" for x in te_values / measured_te))
        print(f"Te_shape_rms_ln={shape_rms_ln(te_values, measured_te):.9f}")
        print(
            "tail_ms_1e_10pct_1pct="
            + ",".join(
                f"{crossing_time_ms(run, fraction):.9f}"
                for fraction in (np.exp(-1.0), 0.1, 0.01)
            )
        )
        print(
            "refill_tau_ms="
            + ",".join(f"{value:.9f}" for value in refill_tau_ms(run))
        )

    apertures, inventories = aperture_diagnostics(runs[-1][1])
    print("\n[combined apertures]")
    for item in apertures:
        print(
            "face={face} z_cm={z_cm:.6f} clear_cm={clear_radius_cm:.1f} "
            "jump={upstream_over_downstream:.9f} "
            "flux_per_s={downstream_flux_per_s:.9e} "
            "coeff_cm3_s={annulus_coeff_cm3_s:.9e}".format(**item)
        )
    print(
        "annulus_reservoir_inventories_particles="
        + ",".join(f"{value:.9e}" for value in inventories)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
