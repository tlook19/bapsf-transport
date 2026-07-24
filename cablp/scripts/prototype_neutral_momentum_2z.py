"""Frozen-field prototype of a conservative two-zone momentum reduction.

This is a scoping model, not LAPDSim1D physics.  It solves the local steady
column/annulus momentum balance on a saved plasma background.  The outward
column momentum carrier uses the ion-temperature CX flight rate; annulus
return and cylindrical-wall accommodation use either the 300 K wall spectrum
or the moment model's Tn_fit.  Annular baffles add their geometry-derived
diffuse or specular momentum-accommodation rates.

Axial momentum transport is deliberately omitted, so this is the local
upper-survival member to compare with the nonlocal KN2Zone first moments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from cablp.solvers._sim1d.core.geometry import build_geometry
from cablp.solvers._sim1d.physics.kinetic_neutrals import (
    EV,
    KB,
    M_HE,
    T_WALL_K,
)
from cablp.solvers._sim1d.physics.sources import (
    ion_neutral_momentum_frequency,
)


PORT_Z_CM = np.asarray([470.05, 789.55, 1045.15, 1428.55, 1716.10])


def vbar_eV(temperature_eV):
    return np.sqrt(
        8.0 * np.asarray(temperature_eV, dtype=float) * EV / (np.pi * M_HE)
    )


def solve_local(
    *,
    n,
    nn_col,
    nn_ann,
    Ti,
    ui,
    geometry,
    sigma_in_cm2,
    sigma_in_model,
    return_temperature_eV,
    baffle_reflection,
):
    nu_in = ion_neutral_momentum_frequency(
        nn=nn_col,
        Ti=Ti,
        ion_mass_g=M_HE,
        gas_type="He",
        sigma_in_cm2=sigma_in_cm2,
        cx_only=False,
        sigma_in_model=sigma_in_model,
    )
    entrainment = nu_in * n / np.maximum(nn_col, 1e-300)
    Rp = np.asarray(geometry.Rp_cm, dtype=float)
    Rm = np.asarray(geometry.Rm_cm, dtype=float)
    shell = np.maximum(Rm**2 - Rp**2, 1e-300)
    nu_ca = vbar_eV(Ti) / (2.0 * Rp)
    v_return = float(vbar_eV(return_temperature_eV))
    nu_ac = v_return * Rp / (2.0 * shell)
    nu_wall = v_return * Rm / (2.0 * shell)

    baffle_rate = np.zeros(geometry.cells)
    if baffle_reflection not in ("diffuse", "specular"):
        raise ValueError("baffle_reflection must be diffuse or specular")
    reflection_factor = 1.0 if baffle_reflection == "diffuse" else 2.0
    V_ann = np.maximum(
        geometry.neutral_volume_cm3 - geometry.plasma_volume_cm3, 1e-300
    )
    v_wall = np.sqrt(8.0 * KB * T_WALL_K / (np.pi * M_HE))
    for face, clear in zip(
        geometry.neutral_baffle_face_indices,
        geometry.neutral_baffle_clear_radius_cm,
    ):
        face = int(face)
        left, right = face - 1, face
        Rm_face = 0.5 * (Rm[left] + Rm[right])
        blocked_area = np.pi * max(Rm_face**2 - float(clear) ** 2, 0.0)
        for cell in (left, right):
            baffle_rate[cell] += (
                reflection_factor
                * 0.25
                * v_wall
                * blocked_area
                / V_ann[cell]
            )

    volume_ratio = V_ann / np.maximum(geometry.plasma_volume_cm3, 1e-300)
    uc = np.zeros_like(ui)
    ua = np.zeros_like(ui)
    for i in range(ui.size):
        nc = max(nn_col[i], 1e-300)
        na = max(nn_ann[i], 1e-300)
        r = volume_ratio[i]
        matrix = np.asarray(
            [
                [
                    entrainment[i] + nu_ca[i],
                    -r * nu_ac[i] * na / nc,
                ],
                [
                    -(nu_ca[i] / r) * nc / na,
                    nu_ac[i] + nu_wall[i] + baffle_rate[i],
                ],
            ]
        )
        uc[i], ua[i] = np.linalg.solve(
            matrix, np.asarray([entrainment[i] * ui[i], 0.0])
        )
    return {
        "uc": uc,
        "ua": ua,
        "entrainment": entrainment,
        "nu_ca": nu_ca,
        "nu_ac": nu_ac,
        "nu_wall": nu_wall,
        "nu_baffle": baffle_rate,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--kinetic", type=Path, required=True)
    parser.add_argument("--window", nargs=2, type=float, default=(15.0, 19.5))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    with h5py.File(args.run, "r") as h5:
        time_ms = (
            h5["time"][:] - float(h5.attrs["t_breakdown_trigger"])
        ) * 1e3
        mask = (time_ms >= args.window[0]) & (time_ms <= args.window[1])
        params = json.loads(h5.attrs["params_json"])
        flags = json.loads(h5.attrs["flags_json"])
        fields = {
            name: np.median(h5[name][mask], axis=0)
            for name in ("n", "nn", "nn_a", "Ti", "u")
        }
        z_cm = h5["geometry/z_cm"][:]
    geometry = build_geometry(params, flags)
    kinetic = np.load(args.kinetic)

    branches = {}
    for thermal_name, return_T in (
        ("wall300K", KB * T_WALL_K / EV),
        ("Tn_fit", float(params["Tn_fit"])),
    ):
        for reflection in ("diffuse", "specular"):
            name = f"{thermal_name}_{reflection}"
            branches[name] = solve_local(
                n=fields["n"],
                nn_col=fields["nn"],
                nn_ann=fields["nn_a"],
                Ti=fields["Ti"],
                ui=fields["u"],
                geometry=geometry,
                sigma_in_cm2=float(params["sigma_in_cm2"]),
                sigma_in_model=str(params["sigma_in_model"]),
                return_temperature_eV=return_T,
                baffle_reflection=reflection,
            )

    print(
        "port ui_km_s kinetic_uc/ui "
        + " ".join(f"{name}_uc/ui" for name in branches)
    )
    port_rows = []
    for port, z_port in zip((11, 21, 29, 41, 50), PORT_Z_CM):
        ih = int(np.argmin(np.abs(z_cm - z_port)))
        ik = int(np.argmin(np.abs(kinetic["z_cm"] - z_port)))
        row = [
            float(port),
            float(z_cm[ih]),
            float(fields["u"][ih]),
            float(kinetic["un_col"][ik] / fields["u"][ih]),
        ]
        row.extend(
            float(branch["uc"][ih] / fields["u"][ih])
            for branch in branches.values()
        )
        port_rows.append(row)
        print(
            f"{port:2d} {fields['u'][ih] / 1e5:9.4f} "
            + " ".join(f"{value:11.6f}" for value in row[3:])
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "z_cm": z_cm,
        "ui": fields["u"],
        "kinetic_z_cm": kinetic["z_cm"],
        "kinetic_uc": kinetic["un_col"],
        "kinetic_ua": kinetic["un_ann"],
        "port_rows": np.asarray(port_rows),
        "branch_names": np.asarray(list(branches), dtype="S"),
        "source_run": str(args.run),
        "kinetic_source": str(args.kinetic),
    }
    for name, branch in branches.items():
        for field, values in branch.items():
            payload[f"{name}_{field}"] = values
    np.savez(args.out, **payload)
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
