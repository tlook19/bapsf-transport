#!/usr/bin/env python3
"""Axisymmetric LAPD end-coil field and flux-surface instrument.

The coil dimensions come from Gekelman et al., Rev. Sci. Instrum. 62,
2875 (1991), Sec. III:

* 14 turns per coil (two seven-turn sections),
* 1.22 m inner diameter and 1.53 m outer diameter,
* 15.3 cm axial coil spacing, and
* measured/calculated uniform-stack figure of merit 1.29 G/A.

The present-day axial coil coordinates are not documented by that paper.
This script therefore represents the downstream end as a configurable
uniform stack ending at z=0.  It is a geometry/sensitivity instrument, not
an as-built current-LAPD field map.  The field topology is computed from
the exact circular-filament Biot-Savart field (complete elliptic
integrals); the current is normalized to the requested interior field.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import ellipe, ellipk


MU0 = 4.0e-7 * np.pi
GAUSS_PER_TESLA = 1.0e4


@dataclass(frozen=True)
class CoilStack:
    """A uniform upstream stack of circular filament coils."""

    n_coils: int = 80
    pitch_m: float = 0.153
    radius_m: float = 0.5 * (1.22 / 2.0 + 1.53 / 2.0)
    turns: int = 14
    last_coil_z_m: float = 0.0

    @property
    def z_m(self) -> np.ndarray:
        return self.last_coil_z_m - self.pitch_m * np.arange(
            self.n_coils - 1, -1, -1, dtype=float
        )


def _validate_stack(stack: CoilStack) -> None:
    if stack.n_coils < 2:
        raise ValueError("n_coils must be >= 2")
    if stack.pitch_m <= 0.0 or stack.radius_m <= 0.0:
        raise ValueError("coil pitch and radius must be positive")
    if stack.turns < 1:
        raise ValueError("turns must be >= 1")


def axis_bz_per_amp(z_m: np.ndarray | float, stack: CoilStack) -> np.ndarray:
    """Return on-axis Bz [T/A] for the complete stack."""

    _validate_stack(stack)
    z = np.asarray(z_m, dtype=float)
    dz = z[..., None] - stack.z_m
    a2 = stack.radius_m**2
    return np.sum(
        MU0 * stack.turns * a2 / (2.0 * (a2 + dz * dz) ** 1.5),
        axis=-1,
    )


def loop_field_per_amp(
    r_m: float, z_m: float, stack: CoilStack
) -> tuple[float, float]:
    """Return (Br, Bz) [T/A] from the complete circular-filament stack."""

    _validate_stack(stack)
    r = float(r_m)
    dz = float(z_m) - stack.z_m
    a = stack.radius_m

    if r < 1.0e-10:
        return 0.0, float(axis_bz_per_amp(z_m, stack))

    alpha2 = (a - r) ** 2 + dz * dz
    beta2 = (a + r) ** 2 + dz * dz
    if np.any(alpha2 <= 1.0e-16):
        raise ValueError("field requested on the idealized filament")

    beta = np.sqrt(beta2)
    m = np.clip(4.0 * a * r / beta2, 0.0, 1.0 - 1.0e-14)
    K = ellipk(m)
    E = ellipe(m)
    pref = MU0 * stack.turns / (2.0 * np.pi * beta)

    br = pref * dz / r * (-K + (a * a + r * r + dz * dz) / alpha2 * E)
    bz = pref * (K + (a * a - r * r - dz * dz) / alpha2 * E)
    return float(np.sum(br)), float(np.sum(bz))


def calibrated_current(
    stack: CoilStack,
    bulk_field_gauss: float,
    reference_z_m: float,
    paper_figure_of_merit_gauss_per_amp: float,
) -> tuple[float, float, float]:
    """Return current, raw model merit, and field calibration factor.

    The single-filament radial proxy slightly underpredicts the paper's
    finite-cross-section stack merit.  The multiplicative calibration changes
    field magnitude but not field-line topology.
    """

    if bulk_field_gauss <= 0.0:
        raise ValueError("bulk_field_gauss must be positive")
    if paper_figure_of_merit_gauss_per_amp <= 0.0:
        raise ValueError("paper figure of merit must be positive")
    model_merit = float(axis_bz_per_amp(reference_z_m, stack) * GAUSS_PER_TESLA)
    if model_merit <= 0.0:
        raise ValueError("reference point has non-positive Bz")
    current = bulk_field_gauss / paper_figure_of_merit_gauss_per_amp
    calibration = paper_figure_of_merit_gauss_per_amp / model_merit
    return current, model_merit, calibration


def trace_flux_surface(
    stack: CoilStack,
    *,
    z_start_m: float,
    z_stop_m: float,
    radius_start_m: float,
    vessel_radius_m: float,
    n_points: int,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Trace dr/dz=Br/Bz until the selected flux surface reaches the wall."""

    if not (z_stop_m > z_start_m):
        raise ValueError("z_stop_m must exceed z_start_m")
    if not (0.0 < radius_start_m < vessel_radius_m):
        raise ValueError("radius_start_m must lie inside the vessel")
    if n_points < 3:
        raise ValueError("n_points must be >= 3")

    def rhs(z: float, radius: np.ndarray) -> np.ndarray:
        br, bz = loop_field_per_amp(float(radius[0]), z, stack)
        if abs(bz) < 1.0e-14:
            raise RuntimeError("Bz approached zero while tracing the flux surface")
        return np.array([br / bz])

    def wall_event(_z: float, radius: np.ndarray) -> float:
        return vessel_radius_m - float(radius[0])

    wall_event.terminal = True
    wall_event.direction = -1.0

    sol = solve_ivp(
        rhs,
        (z_start_m, z_stop_m),
        np.array([radius_start_m]),
        events=wall_event,
        dense_output=True,
        rtol=2.0e-9,
        atol=2.0e-11,
        max_step=min(stack.pitch_m / 5.0, (z_stop_m - z_start_m) / 100.0),
    )
    if not sol.success:
        raise RuntimeError(sol.message)

    hit_wall = bool(sol.t_events[0].size)
    z_end = float(sol.t_events[0][0]) if hit_wall else z_stop_m
    z = np.linspace(z_start_m, z_end, n_points)
    radius = sol.sol(z)[0]
    return z, radius, hit_wall


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-coils", type=int, default=80)
    parser.add_argument("--pitch-m", type=float, default=0.153)
    parser.add_argument("--coil-radius-m", type=float, default=0.6875)
    parser.add_argument("--turns", type=int, default=14)
    parser.add_argument("--last-coil-z-m", type=float, default=0.0)
    parser.add_argument("--bulk-field-gauss", type=float, default=1400.0)
    parser.add_argument("--reference-z-m", type=float, default=-3.0)
    parser.add_argument("--paper-figure-of-merit-gauss-per-amp", type=float, default=1.29)
    parser.add_argument("--z-start-m", type=float, default=-1.0)
    parser.add_argument("--z-stop-m", type=float, default=1.5)
    parser.add_argument("--plasma-radius-m", type=float, default=0.18)
    parser.add_argument("--main-vessel-radius-m", type=float, default=0.5)
    parser.add_argument("--end-vessel-radius-m", type=float, default=1.0)
    parser.add_argument("--samples", type=int, default=501)
    parser.add_argument("--output-png", type=Path)
    parser.add_argument("--output-npz", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    stack = CoilStack(
        n_coils=args.n_coils,
        pitch_m=args.pitch_m,
        radius_m=args.coil_radius_m,
        turns=args.turns,
        last_coil_z_m=args.last_coil_z_m,
    )
    current_a, model_merit_ga, field_calibration = calibrated_current(
        stack,
        args.bulk_field_gauss,
        args.reference_z_m,
        args.paper_figure_of_merit_gauss_per_amp,
    )

    z_axis = np.linspace(args.z_start_m, args.z_stop_m, args.samples)
    bz_axis_g = (
        axis_bz_per_amp(z_axis, stack)
        * field_calibration
        * current_a
        * GAUSS_PER_TESLA
    )
    z_flux, r_flux, hit_wall = trace_flux_surface(
        stack,
        z_start_m=args.z_start_m,
        z_stop_m=args.z_stop_m,
        radius_start_m=args.plasma_radius_m,
        vessel_radius_m=args.end_vessel_radius_m,
        n_points=args.samples,
    )
    wall_z = float(z_flux[-1]) if hit_wall else np.nan
    sensitivity_radii = (1.22 / 2.0, stack.radius_m, 1.53 / 2.0)
    sensitivity_offsets = (
        stack.last_coil_z_m - 0.5 * stack.pitch_m,
        stack.last_coil_z_m,
        stack.last_coil_z_m + 0.5 * stack.pitch_m,
    )
    radius_sensitivity = []
    for coil_radius_m in sensitivity_radii:
        for last_coil_z_m in sensitivity_offsets:
            sensitivity_stack = CoilStack(
                n_coils=stack.n_coils,
                pitch_m=stack.pitch_m,
                radius_m=coil_radius_m,
                turns=stack.turns,
                last_coil_z_m=last_coil_z_m,
            )
            z_case, r_case, _ = trace_flux_surface(
                sensitivity_stack,
                z_start_m=args.z_start_m,
                z_stop_m=args.z_stop_m,
                radius_start_m=args.plasma_radius_m,
                vessel_radius_m=args.end_vessel_radius_m,
                n_points=args.samples,
            )
            radius_sensitivity.append(
                np.interp(
                    z_axis,
                    z_case,
                    r_case,
                    left=r_case[0],
                    right=args.end_vessel_radius_m,
                )
            )
    radius_sensitivity = np.asarray(radius_sensitivity)
    radius_low = np.min(radius_sensitivity, axis=0)
    radius_high = np.max(radius_sensitivity, axis=0)

    print(f"stack coils: {stack.n_coils}")
    print(f"documented coil radius proxy: {stack.radius_m:.4f} m")
    print(f"raw filament merit at reference: {model_merit_ga:.4f} G/A")
    print(
        "paper uniform-stack figure of merit: "
        f"{args.paper_figure_of_merit_gauss_per_amp:.4f} G/A"
    )
    print(f"magnitude calibration factor: {field_calibration:.5f}")
    print(f"current for {args.bulk_field_gauss:.1f} G: {current_a:.2f} A")
    print(
        "axis B at last-coil plane: "
        f"{np.interp(args.last_coil_z_m, z_axis, bz_axis_g):.1f} G"
    )
    if hit_wall:
        print(f"r={args.end_vessel_radius_m:.3f} m wall reached at z={wall_z:.4f} m")
    else:
        print(
            f"flux-surface radius at z={z_flux[-1]:.3f} m: "
            f"{r_flux[-1]:.4f} m (wall not reached)"
        )
    for z_report in (args.last_coil_z_m, 0.5, 1.0, 1.5):
        if z_flux[0] <= z_report <= z_flux[-1]:
            print(
                f"flux-surface radius at z={z_report:.3f} m: "
                f"{np.interp(z_report, z_flux, r_flux):.4f} m "
                f"(geometry sensitivity "
                f"{np.interp(z_report, z_axis, radius_low):.4f}-"
                f"{np.interp(z_report, z_axis, radius_high):.4f} m)"
            )

    fig, (ax_b, ax_r) = plt.subplots(
        2, 1, figsize=(8.2, 6.6), sharex=True, constrained_layout=True
    )
    ax_b.plot(z_axis, bz_axis_g / args.bulk_field_gauss, color="C0", lw=2.2)
    ax_b.axvline(args.last_coil_z_m, color="0.3", ls="--", lw=1.0)
    ax_b.axhline(1.0, color="0.55", ls=":", lw=1.0)
    ax_b.set_ylabel(r"$B_z(0,z)/B_{\rm bulk}$")
    ax_b.set_title("LAPD downstream coil-termination field (provisional)")
    ax_b.grid(alpha=0.2)

    ax_r.fill_between(
        z_axis,
        radius_low,
        radius_high,
        color="C1",
        alpha=0.18,
        label="coil-radius / last-position sensitivity",
    )
    ax_r.plot(z_flux, r_flux, color="C1", lw=2.2, label="mean-radius trace")
    ax_r.axvline(args.last_coil_z_m, color="0.3", ls="--", lw=1.0)
    ax_r.fill_between(
        [args.z_start_m, args.last_coil_z_m],
        args.main_vessel_radius_m,
        args.end_vessel_radius_m,
        color="0.75",
        alpha=0.25,
        label="outside main vessel",
    )
    ax_r.axhline(args.end_vessel_radius_m, color="0.45", lw=1.0)
    ax_r.set_xlabel("z relative to last-coil/end-cell plane [m]")
    ax_r.set_ylabel("flux-surface radius [m]")
    ax_r.set_ylim(0.0, 1.05 * args.end_vessel_radius_m)
    ax_r.grid(alpha=0.2)
    ax_r.legend(loc="upper left", frameon=False)

    if args.output_png is not None:
        args.output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output_png, dpi=180)
    if args.output_npz is not None:
        args.output_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output_npz,
            z_axis_m=z_axis,
            bz_axis_gauss=bz_axis_g,
            z_flux_m=z_flux,
            flux_radius_m=r_flux,
            flux_radius_sensitivity_low_m=radius_low,
            flux_radius_sensitivity_high_m=radius_high,
            wall_hit=np.array(hit_wall),
            wall_z_m=np.array(wall_z),
            current_a=np.array(current_a),
            raw_model_figure_of_merit_gauss_per_amp=np.array(model_merit_ga),
            paper_figure_of_merit_gauss_per_amp=np.array(
                args.paper_figure_of_merit_gauss_per_amp
            ),
            field_magnitude_calibration=np.array(field_calibration),
            coil_z_m=stack.z_m,
            coil_radius_m=np.array(stack.radius_m),
            coil_pitch_m=np.array(stack.pitch_m),
            coil_turns=np.array(stack.turns),
        )
    if args.output_png is None:
        plt.show()


if __name__ == "__main__":
    main()
