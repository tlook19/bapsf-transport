"""Score the registered fixed-end/two-baffle b_drag=0.5 versus 1.0 A/B."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from score_es1_two_baffles import (
    PORTS,
    Z_PORT_CM,
    crossing_time_ms,
    load,
    measured_profile,
    plateau_mask,
    port_values,
    shape_rms_ln,
)


HERE = Path(__file__).resolve().parent


def momentum_fields(run):
    with h5py.File(run["path"], "r") as h5:
        return h5["M"][:], h5["rhs_terms/ion_neutral_drag/M"][:]


def port_speed_and_drag(run):
    momentum, drag_rhs = momentum_fields(run)
    mask = plateau_mask(run)
    speed_km_s = []
    tau_ms = []
    length_cm = []
    for z in Z_PORT_CM:
        index = int(np.argmin(np.abs(run["z_cm"] - z)))
        speed = float(np.median(np.abs(run["u"][mask, index])))
        tau = float(
            np.median(
                np.abs(momentum[mask, index])
                / np.maximum(np.abs(drag_rhs[mask, index]), 1e-300)
            )
        )
        speed_km_s.append(speed / 1e5)
        tau_ms.append(tau * 1e3)
        length_cm.append(speed * tau)
    return np.asarray(speed_km_s), np.asarray(tau_ms), np.asarray(length_cm)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument(
        "--overlay", type=Path, default=HERE / "data" / "es1_sim1d_overlay.npz"
    )
    args = parser.parse_args(argv)
    overlay = np.load(args.overlay)
    measured_n = measured_profile(overlay, "n")
    measured_te = measured_profile(overlay, "Te")

    results = {}
    for label, path in (("A_drag0p5", args.a), ("B_drag1", args.b)):
        run = load(path)
        with h5py.File(path, "r") as h5:
            run["u"] = h5["u"][:]
        n = port_values(run, "n")
        te = port_values(run, "Te")
        speed, tau, length = port_speed_and_drag(run)
        results[label] = {
            "speed": speed,
            "n_shape": shape_rms_ln(n, measured_n),
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
        print(f"Te_shape_rms_ln={shape_rms_ln(te, measured_te):.9f}")
        print("speed_km_s=" + ",".join(f"{x:.6f}" for x in speed))
        print("drag_tau_ms=" + ",".join(f"{x:.6f}" for x in tau))
        print("drag_length_cm=" + ",".join(f"{x:.6f}" for x in length))
        print(
            "tail_ms_1e_10pct_1pct="
            + ",".join(
                f"{crossing_time_ms(run, fraction):.9f}"
                for fraction in (np.exp(-1.0), 0.1, 0.01)
            )
        )

    a = results["A_drag0p5"]
    b = results["B_drag1"]
    a_speed = float(np.mean(a["speed"][:3]))
    b_speed = float(np.mean(b["speed"][:3]))
    print("\n[A/B registered discriminators]")
    print(f"mean_speed_ports_11_21_29_A_km_s={a_speed:.9f}")
    print(f"mean_speed_ports_11_21_29_B_km_s={b_speed:.9f}")
    print(f"speed_reduction_fraction={1.0 - b_speed / a_speed:.9f}")
    print(f"density_shape_improvement={a['n_shape'] - b['n_shape']:.9f}")
    print(f"density_median_move_toward_one={abs(1-a['n_median']) - abs(1-b['n_median']):.9f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
