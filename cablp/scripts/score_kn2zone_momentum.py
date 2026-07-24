"""Score KN2Zone's first-moment radial-momentum gate on a saved run.

This is an offline validation instrument.  It does not modify LAPDSim1D or
fit a damping coefficient.  The K1b jump engine supplies the column
distribution and annulus residence moments; this script extracts the axial
first moments and the bin-resolved column-to-annulus momentum ledger.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from cablp.solvers._sim1d.core.geometry import build_geometry
from cablp.solvers._sim1d.physics.kinetic_neutrals import (
    KN2ZoneJump,
    M_HE,
)
from cablp.solvers._sim1d.physics.sources import (
    neutral_wind_two_zone_factors,
)
from mc_neutrals import load_background


PORT_Z_CM = np.asarray([470.05, 789.55, 1045.15, 1428.55, 1716.10])


def _relative_error(got, expected):
    scale = max(float(np.max(np.abs(expected))), 1e-300)
    return float(np.max(np.abs(got - expected)) / scale)


def _saved_plasma_moments(path, window_ms):
    with h5py.File(path, "r") as h5:
        time_ms = (
            h5["time"][:] - float(h5.attrs["t_breakdown_trigger"])
        ) * 1e3
        mask = (time_ms >= window_ms[0]) & (time_ms <= window_ms[1])
        if not np.any(mask):
            raise ValueError(f"empty scoring window {window_ms}")
        z_cm = h5["geometry/z_cm"][:]
        ion_u = np.median(h5["u"][mask], axis=0)
        chamber_un = np.median(h5["u_n"][mask], axis=0)
        params = json.loads(h5.attrs["params_json"])
        flags = json.loads(h5.attrs["flags_json"])
    geometry = build_geometry(params, flags)
    column_factor, _ = neutral_wind_two_zone_factors(
        geometry=geometry,
        Tn_eV=float(params["Tn_fit"]),
        ion_mass_g=M_HE,
    )
    return z_cm, ion_u, chamber_un * column_factor


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--window", nargs=2, type=float, default=(15.0, 19.5))
    parser.add_argument("--nvz", type=int, default=80)
    parser.add_argument("--nvp", type=int, default=24)
    parser.add_argument("--truncate", type=float, default=1e-3)
    parser.add_argument("--max-gen", type=int, default=400)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    window = tuple(args.window)
    background = load_background(args.run, window)
    background["zone_rates"] = "flux"
    engine = KN2ZoneJump(
        background,
        nvz=args.nvz,
        nvp=args.nvp,
        truncate=args.truncate,
        max_gen=args.max_gen,
        verbose=True,
    )
    result = engine.solve()
    Fc = np.asarray(result["Fc"], dtype=float)
    vz = engine.g.VZ[None, :, :]

    nn_col_direct = Fc.sum(axis=(1, 2))
    first_col_direct = (Fc * vz).sum(axis=(1, 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        un_col_direct = np.where(
            nn_col_direct > 0.0,
            first_col_direct / nn_col_direct,
            0.0,
        )
    nn_col_error = _relative_error(nn_col_direct, result["nn_col"])
    un_col_error = _relative_error(un_col_direct, result["un_col"])

    M_col = M_HE * first_col_direct
    M_ann = M_HE * np.asarray(result["nn_ann"]) * np.asarray(result["un_ann"])

    # Each bin escapes the column at nux(v_perp).  Multiplying its particle
    # rate by m*v_z gives the signed axial momentum transferred into the
    # annulus.  The volume-integrated form is the exact column/annulus ledger.
    radial_M_density_rate = M_HE * (
        Fc * vz * engine.nux[:, None, :]
    ).sum(axis=(1, 2))
    radial_M_total_rate = radial_M_density_rate * engine.V_col
    radial_M_loop = np.asarray(
        [
            M_HE
            * engine.V_col[i]
            * sum(
                Fc[i, j, k]
                * engine.g.vz[j]
                * engine.nux[i, k]
                for j in range(engine.g.nvz)
                for k in range(engine.g.nvp)
            )
            for i in range(engine.nz)
        ]
    )
    radial_ledger_error = _relative_error(
        radial_M_total_rate, radial_M_loop
    )

    z_cm = 0.5 * (
        np.asarray(background["z_edges"][:-1])
        + np.asarray(background["z_edges"][1:])
    )
    saved_z, ion_u, current_Mn_u_col = _saved_plasma_moments(args.run, window)
    rows = []
    for port, z_port in zip((11, 21, 29, 41, 50), PORT_Z_CM):
        ik = int(np.argmin(np.abs(z_cm - z_port)))
        ih = int(np.argmin(np.abs(saved_z - z_port)))
        ratio = result["un_col"][ik] / max(abs(ion_u[ih]), 1e-300)
        rows.append(
            (
                port,
                z_cm[ik],
                ion_u[ih],
                result["un_col"][ik],
                result["un_ann"][ik],
                current_Mn_u_col[ih],
                ratio,
                result["nn_col"][ik],
                result["nn_ann"][ik],
                radial_M_total_rate[ik],
            )
        )

    finite = all(
        np.all(np.isfinite(values))
        for values in (
            M_col,
            M_ann,
            radial_M_density_rate,
            radial_M_total_rate,
            result["un_col"],
            result["un_ann"],
        )
    )
    identity_pass = (
        nn_col_error <= 1e-12
        and un_col_error <= 1e-12
        and radial_ledger_error <= 1e-12
        and finite
    )
    ratios = np.asarray([row[6] for row in rows])
    direction_pass = bool(
        np.all(ratios[:3] <= 0.55)
        and np.all(ratios[3:] < 0.90)
    )

    print("\n=== KN2Zone first-moment identities ===")
    print(f"nn_col_direct_error={nn_col_error:.3e}")
    print(f"un_col_direct_error={un_col_error:.3e}")
    print(f"radial_momentum_ledger_error={radial_ledger_error:.3e}")
    print(f"all_finite={finite}")
    print(f"identity_gate={'PASS' if identity_pass else 'FAIL'}")
    print("\n=== Five-port kinetic momentum gate ===")
    print(
        "port z_cm ui_km_s ucol_kn_km_s uann_kn_km_s "
        "ucol_current_Mn_km_s ucol_over_ui nn_col nn_ann "
        "radial_M_flux_dyn"
    )
    for row in rows:
        print(
            f"{row[0]:2d} {row[1]:7.2f} "
            f"{row[2] / 1e5:9.4f} {row[3] / 1e5:12.4f} "
            f"{row[4] / 1e5:12.4f} {row[5] / 1e5:18.4f} "
            f"{row[6]:11.6f} {row[7]:.6e} {row[8]:.6e} "
            f"{row[9]:+.6e}"
        )
    print(f"direction_gate={'PASS' if direction_pass else 'FAIL'}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        z_cm=z_cm,
        nn_col=result["nn_col"],
        nn_ann=result["nn_ann"],
        un_col=result["un_col"],
        un_ann=result["un_ann"],
        M_col=M_col,
        M_ann=M_ann,
        radial_M_density_rate=radial_M_density_rate,
        radial_M_total_rate=radial_M_total_rate,
        port_rows=np.asarray(rows, dtype=float),
        identity_pass=identity_pass,
        direction_pass=direction_pass,
        generations=result["generations"],
        source_run=str(args.run),
        baffles_resolved=False,
    )
    print(f"saved {args.out}")
    return 0 if identity_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
