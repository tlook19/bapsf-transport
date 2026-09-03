#!/usr/bin/env python3
"""Axisymmetric LAPD end-field solve on the MEASURED coil census.

This is the census-based successor to ``solve_lapd_coil_field.py`` (at commit
48be9a4, retired 2026-09-03).  That
script represented the machine as a single uniform 80-coil stack (0.153 m
pitch, 0.6875 m mean radius) terminating at z=0; the real magnet layout is
segmented, has a 0.83 m bridged gap after the last main coil, and carries a
concentrated pair of far-source ("end") coils beyond the gap.  A uniform
stack is not a faithful surrogate for that fringe, so every termination
number is recomputed here.

Census provenance
-----------------
DESIGN-class SolidWorks export ``LAPD_TomLook-Magnets.STEP``
(engineer-supplied, received 2026-08-18), reduced to per-part global
bounding boxes in ``tree_bbox_mag.txt``.  Model coordinate:
``z_model = (-4560 - z_CAD_mm)/1000`` (z=0 at the LaB6 emitting face, +z
toward the far end).  The per-coil axial centres below are the bbox
midpoints of the ``Magnet-yellow`` (22), ``Magnet-pink`` (34) and
``2nd-Cathode_coil`` (2) parts; they are embedded as literals so this
script stands alone, and ``--verify-census`` re-parses the CAD dump and
asserts the literals still match.  Winding-pack radii are the analytic
cylindrical-face radii quoted in the CAD brief.  As-built deviations are
unquantified.

Source end
----------
The 16 source-coil frames at z_model -0.675...+0.747 m are RECTANGULAR
(1327 x 1473 mm), hence not representable as coaxial circular filaments,
and they sit >13 m from the region of interest.  They are OMITTED, exactly
as the predecessor script omitted them; ``--source-coil-bound`` prints the
equal-area circular-loop upper bound on what that omission costs at the far
end.

Currents
--------
One base per-coil ampere-turn value ``I0`` for every main-stack coil
(yellow + pink alike -- "uniform per-coil current"), scaled so that the
on-axis field at the mid-point of the pink span equals ``--bulk-field-gauss``
(1400 G, machine-ruled).  Turns-per-coil is absorbed into ``I0``, so no
figure-of-merit calibration is needed (the predecessor's 1.29 G/A factor
existed only because it normalised per AMP rather than per coil).  The end
pair carries ``f_end * I0`` with ``f_end`` a free parameter; two named cases
are solved:

* ``droop-min`` -- ``f_end`` minimising ``max |Bz(0,z) - B_bulk|`` over the
  bridged gap span ``[18.55, 19.56] m`` (minimax over that span, on axis);
* ``off`` -- ``f_end = 0``.

Fields come from the exact circular-filament Biot-Savart expressions
(complete elliptic integrals); the flux function ``Psi(r,z) = 2*pi*r*A_phi``
is evaluated from the same filaments and is used both to anchor the traced
surface and as an independent conservation check on the ODE trace.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.integrate import solve_ivp  # noqa: E402
from scipy.optimize import brentq, minimize_scalar  # noqa: E402
from scipy.special import ellipe, ellipk  # noqa: E402


MU0 = 4.0e-7 * np.pi
GAUSS_PER_TESLA = 1.0e4

CAD_Z_OFFSET_MM = -4560.0

# --- measured census (CAD 2026-08-18, design class) -------------------------
# Magnet-yellow, cathode side: 11 coils, first two a close pair.
YELLOW_NEAR_CENTERS_M = (
    1.191705, 1.346705, 1.666705, 1.986705, 2.306705, 2.626705,
    2.946705, 3.266705, 3.586705, 3.906705, 4.226705,
)
# Magnet-pink, main stack: 34 coils at exactly 0.32 m pitch.
PINK_CENTERS_M = tuple(round(4.546700 + 0.32 * i, 6) for i in range(34))
# Magnet-yellow, far side: 11 coils, last two a close pair.
YELLOW_FAR_CENTERS_M = (
    15.426705, 15.746705, 16.066705, 16.386705, 16.706705, 17.026705,
    17.346705, 17.666705, 17.986705, 18.306705, 18.474975,
)
# 2nd-Cathode_coil: the far-source ("end") pair, adjacent, 88.9 mm each.
END_PAIR_CENTERS_M = (19.425500, 19.514400)

YELLOW_R_IN_M, YELLOW_R_OUT_M, YELLOW_THICK_M = 0.5651, 0.7207, 0.1588
PINK_R_IN_M, PINK_R_OUT_M, PINK_THICK_M = 0.6205, 0.7500, 0.1698
END_R_IN_M, END_R_OUT_M, END_THICK_M = 0.5842, 0.7620, 0.0889

# Source coils (rectangular frames) -- omitted; used only for the bound.
SOURCE_COIL_CENTERS_M = tuple(-0.63060 + 0.08863 * i for i in range(16))
SOURCE_FRAME_X_M, SOURCE_FRAME_Y_M = 1.3272, 1.4732

PINK_SPAN_M = (PINK_CENTERS_M[0] - 0.5 * PINK_THICK_M,
               PINK_CENTERS_M[-1] + 0.5 * PINK_THICK_M)
GAP_SPAN_M = (18.55, 19.56)


@dataclass(frozen=True)
class CoilGroup:
    """A set of identical coaxial winding packs sharing one current."""

    name: str
    centers_m: tuple[float, ...]
    r_in_m: float
    r_out_m: float
    thickness_m: float

    def filaments(self, n_r: int, n_z: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (radius, z, weight) filament arrays for this group.

        The winding pack is discretised by the midpoint rule over its
        rectangular cross-section; ``weight`` is the fraction of the coil's
        ampere-turns carried by each filament and sums to ``len(centers)``.
        """

        if n_r < 1 or n_z < 1:
            raise ValueError("n_r and n_z must be >= 1")
        frac_r = (np.arange(n_r, dtype=float) + 0.5) / n_r
        frac_z = (np.arange(n_z, dtype=float) + 0.5) / n_z - 0.5
        radii = self.r_in_m + (self.r_out_m - self.r_in_m) * frac_r
        offsets = self.thickness_m * frac_z
        centers = np.asarray(self.centers_m, dtype=float)
        r = np.broadcast_to(radii[None, :, None], (centers.size, n_r, n_z))
        z = centers[:, None, None] + offsets[None, None, :]
        z = np.broadcast_to(z, (centers.size, n_r, n_z))
        w = np.full(centers.size * n_r * n_z, 1.0 / (n_r * n_z))
        return r.reshape(-1).copy(), z.reshape(-1).copy(), w


@dataclass(frozen=True)
class Census:
    """The measured census split into its two independently-fed parts."""

    main: tuple[CoilGroup, ...]
    end: tuple[CoilGroup, ...]


def measured_census() -> Census:
    """Return the 2026-08-18 CAD census."""

    return Census(
        main=(
            CoilGroup("yellow-near", YELLOW_NEAR_CENTERS_M,
                      YELLOW_R_IN_M, YELLOW_R_OUT_M, YELLOW_THICK_M),
            CoilGroup("pink", PINK_CENTERS_M,
                      PINK_R_IN_M, PINK_R_OUT_M, PINK_THICK_M),
            CoilGroup("yellow-far", YELLOW_FAR_CENTERS_M,
                      YELLOW_R_IN_M, YELLOW_R_OUT_M, YELLOW_THICK_M),
        ),
        end=(
            CoilGroup("end-pair", END_PAIR_CENTERS_M,
                      END_R_IN_M, END_R_OUT_M, END_THICK_M),
        ),
    )


def uniform_stack_census(
    *,
    n_coils: int = 80,
    pitch_m: float = 0.153,
    radius_m: float = 0.6875,
    last_coil_z_m: float = 0.0,
) -> Census:
    """Return the predecessor script's uniform stack as a Census.

    Zero-thickness, zero-radial-width groups reproduce its single-filament
    idealisation exactly, which is what the limit gate needs.
    """

    centers = tuple(
        last_coil_z_m - pitch_m * k for k in range(n_coils - 1, -1, -1)
    )
    return Census(
        main=(CoilGroup("uniform", centers, radius_m, radius_m, 0.0),),
        end=(),
    )


class FilamentSet:
    """Flattened filament arrays plus the field/flux kernels."""

    def __init__(self, groups: tuple[CoilGroup, ...], n_r: int, n_z: int) -> None:
        if not groups:
            self.a = np.zeros(0)
            self.zc = np.zeros(0)
            self.w = np.zeros(0)
            return
        parts = [g.filaments(n_r, n_z) for g in groups]
        self.a = np.concatenate([p[0] for p in parts])
        self.zc = np.concatenate([p[1] for p in parts])
        self.w = np.concatenate([p[2] for p in parts])

    @property
    def n_filaments(self) -> int:
        return int(self.a.size)

    def axis_bz(self, z_m: np.ndarray | float) -> np.ndarray:
        """On-axis Bz [T per unit coil ampere-turn]."""

        if self.a.size == 0:
            return np.zeros_like(np.asarray(z_m, dtype=float))
        z = np.asarray(z_m, dtype=float)
        dz = z[..., None] - self.zc
        a2 = self.a**2
        return np.sum(MU0 * self.w * a2 / (2.0 * (a2 + dz * dz) ** 1.5), axis=-1)

    def field(self, r_m: float, z_m: float) -> tuple[float, float]:
        """(Br, Bz) [T per unit coil ampere-turn] at (r, z)."""

        if self.a.size == 0:
            return 0.0, 0.0
        r = float(r_m)
        if r < 1.0e-10:
            return 0.0, float(self.axis_bz(z_m))
        dz = float(z_m) - self.zc
        a = self.a
        alpha2 = (a - r) ** 2 + dz * dz
        if np.any(alpha2 <= 1.0e-16):
            raise ValueError("field requested on an idealized filament")
        beta2 = (a + r) ** 2 + dz * dz
        beta = np.sqrt(beta2)
        m = np.clip(4.0 * a * r / beta2, 0.0, 1.0 - 1.0e-14)
        kk = ellipk(m)
        ee = ellipe(m)
        pref = MU0 * self.w / (2.0 * np.pi * beta)
        br = pref * dz / r * (-kk + (a * a + r * r + dz * dz) / alpha2 * ee)
        bz = pref * (kk + (a * a - r * r - dz * dz) / alpha2 * ee)
        return float(np.sum(br)), float(np.sum(bz))

    def flux(self, r_m: float, z_m: float) -> float:
        """Magnetic flux [Wb per unit coil ampere-turn] through radius r."""

        if self.a.size == 0:
            return 0.0
        r = float(r_m)
        if r < 1.0e-10:
            return float(np.pi * r * r * self.axis_bz(z_m))
        dz = float(z_m) - self.zc
        a = self.a
        beta2 = (a + r) ** 2 + dz * dz
        m = np.clip(4.0 * a * r / beta2, 1.0e-300, 1.0 - 1.0e-14)
        kk = ellipk(m)
        ee = ellipe(m)
        # A_phi = mu0*I/(pi*k) * sqrt(a/r) * [(1 - m/2) K - E]
        a_phi = (
            MU0 * self.w / (np.pi * np.sqrt(m))
            * np.sqrt(a / r)
            * ((1.0 - 0.5 * m) * kk - ee)
        )
        return float(2.0 * np.pi * r * np.sum(a_phi))


class CensusField:
    """Main stack + end pair with an independent end-pair current fraction."""

    def __init__(self, census: Census, n_r: int, n_z: int) -> None:
        self.main = FilamentSet(census.main, n_r, n_z)
        self.end = FilamentSet(census.end, n_r, n_z)
        self.n_r = int(n_r)
        self.n_z = int(n_z)

    @property
    def n_filaments(self) -> int:
        return self.main.n_filaments + self.end.n_filaments

    def base_current(
        self, end_fraction: float, bulk_field_gauss: float, reference_z_m: float
    ) -> float:
        """Per-coil ampere-turns giving ``bulk_field_gauss`` at the reference."""

        unit = float(
            self.main.axis_bz(reference_z_m)
            + end_fraction * self.end.axis_bz(reference_z_m)
        )
        if unit <= 0.0:
            raise ValueError("reference plane has non-positive unit Bz")
        return bulk_field_gauss / (unit * GAUSS_PER_TESLA)

    def axis_bz_gauss(
        self, z_m: np.ndarray, end_fraction: float, current: float
    ) -> np.ndarray:
        return (
            (self.main.axis_bz(z_m) + end_fraction * self.end.axis_bz(z_m))
            * current
            * GAUSS_PER_TESLA
        )

    def field(
        self, r_m: float, z_m: float, end_fraction: float, current: float
    ) -> tuple[float, float]:
        br_m, bz_m = self.main.field(r_m, z_m)
        br_e, bz_e = self.end.field(r_m, z_m)
        return (
            current * (br_m + end_fraction * br_e),
            current * (bz_m + end_fraction * bz_e),
        )

    def flux(
        self, r_m: float, z_m: float, end_fraction: float, current: float
    ) -> float:
        return current * (
            self.main.flux(r_m, z_m) + end_fraction * self.end.flux(r_m, z_m)
        )


def solve_droop_min_fraction(
    field: CensusField,
    *,
    bulk_field_gauss: float,
    reference_z_m: float,
    gap_span_m: tuple[float, float],
    samples: int = 401,
) -> tuple[float, float]:
    """Return (f_end, max|Bz - B_bulk| [G]) minimising droop over the gap."""

    z_gap = np.linspace(gap_span_m[0], gap_span_m[1], samples)
    unit_main = field.main.axis_bz(z_gap)
    unit_end = field.end.axis_bz(z_gap)
    ref_main = float(field.main.axis_bz(reference_z_m))
    ref_end = float(field.end.axis_bz(reference_z_m))

    def objective(f: float) -> float:
        denom = ref_main + f * ref_end
        if denom <= 0.0:
            return np.inf
        bz = (unit_main + f * unit_end) / denom * bulk_field_gauss
        return float(np.max(np.abs(bz - bulk_field_gauss)))

    res = minimize_scalar(objective, bounds=(0.0, 20.0), method="bounded",
                          options={"xatol": 1.0e-8})
    if not res.success:
        raise RuntimeError(f"droop-min search failed: {res.message}")
    return float(res.x), float(res.fun)


def anchor_radius(
    field: CensusField,
    *,
    z_m: float,
    plasma_radius_m: float,
    bulk_field_gauss: float,
    end_fraction: float,
    current: float,
) -> tuple[float, float]:
    """Radius at ``z_m`` carrying the plasma column's uniform-field flux.

    The Rp surface is defined by the flux ``pi*Rp^2*B_bulk`` that the column
    carries where the field is exactly ``B_bulk``; solving for the radius at
    the trace start makes the start point flux-consistent rather than
    assuming the local field is exactly uniform.
    """

    psi_target = np.pi * plasma_radius_m**2 * bulk_field_gauss / GAUSS_PER_TESLA

    def residual(r: float) -> float:
        return field.flux(r, z_m, end_fraction, current) - psi_target

    r0 = brentq(residual, 0.5 * plasma_radius_m, 2.0 * plasma_radius_m,
                xtol=1.0e-14, rtol=1.0e-14)
    return float(r0), float(psi_target)


def trace_flux_surface(
    field: CensusField,
    *,
    z_start_m: float,
    z_stop_m: float,
    radius_start_m: float,
    end_fraction: float,
    current: float,
    radius_cap_m: float,
    n_points: int,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Integrate dr/dz = Br/Bz from ``z_start_m`` to ``z_stop_m``."""

    if not (z_stop_m > z_start_m):
        raise ValueError("z_stop_m must exceed z_start_m")
    if not (0.0 < radius_start_m < radius_cap_m):
        raise ValueError("radius_start_m must lie inside the radius cap")
    if n_points < 3:
        raise ValueError("n_points must be >= 3")

    def rhs(z: float, radius: np.ndarray) -> np.ndarray:
        br, bz = field.field(float(radius[0]), z, end_fraction, current)
        if abs(bz) < 1.0e-12:
            raise RuntimeError("Bz approached zero while tracing")
        return np.array([br / bz])

    def cap_event(_z: float, radius: np.ndarray) -> float:
        return radius_cap_m - float(radius[0])

    cap_event.terminal = True
    cap_event.direction = -1.0

    sol = solve_ivp(
        rhs,
        (z_start_m, z_stop_m),
        np.array([radius_start_m]),
        events=cap_event,
        dense_output=True,
        rtol=2.0e-10,
        atol=2.0e-12,
        max_step=0.05,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    capped = bool(sol.t_events[0].size)
    z_end = float(sol.t_events[0][0]) if capped else z_stop_m
    z = np.linspace(z_start_m, z_end, n_points)
    return z, sol.sol(z)[0], capped


def first_crossing(
    z: np.ndarray, r: np.ndarray, target_m: float
) -> tuple[float, float]:
    """Return (z, dr/dz) at the first upward crossing of ``target_m``."""

    idx = np.nonzero(r >= target_m)[0]
    if idx.size == 0 or idx[0] == 0:
        return float("nan"), float("nan")
    i = int(idx[0])
    z_cross = float(
        np.interp(target_m, [r[i - 1], r[i]], [z[i - 1], z[i]])
    )
    slope = float((r[i] - r[i - 1]) / (z[i] - z[i - 1]))
    return z_cross, slope


def source_coil_bound(
    *, probe_z_m: float, bulk_field_gauss: float, reference_z_m: float,
    field: CensusField, end_fraction: float,
) -> float:
    """Equal-area circular-loop upper bound on the omitted source coils [G].

    The 16 rectangular frames are replaced by circular loops of equal
    enclosed area (which maximises the on-axis dipole moment for a given
    area) carrying the same per-coil ampere-turns as the main stack.
    """

    current = field.base_current(end_fraction, bulk_field_gauss, reference_z_m)
    a = np.sqrt(SOURCE_FRAME_X_M * SOURCE_FRAME_Y_M / np.pi)
    zc = np.asarray(SOURCE_COIL_CENTERS_M, dtype=float)
    dz = probe_z_m - zc
    bz = np.sum(MU0 * a**2 / (2.0 * (a**2 + dz * dz) ** 1.5))
    return float(bz * current * GAUSS_PER_TESLA)


def verify_census(cad_bbox_path: Path) -> list[str]:
    """Re-parse the CAD bbox dump and assert the embedded literals match."""

    groups: dict[str, list[float]] = {
        "Magnet-yellow": [], "Magnet-pink": [], "2nd-Cathode_coil": [],
    }
    pattern = re.compile(r"z\[([-\d.]+),([-\d.]+)\]")
    for line in cad_bbox_path.read_text().splitlines():
        name = line.split("|")[0].strip()
        if name not in groups:
            continue
        match = pattern.search(line)
        if match is None:
            continue
        z_lo = (CAD_Z_OFFSET_MM - float(match.group(2))) / 1000.0
        z_hi = (CAD_Z_OFFSET_MM - float(match.group(1))) / 1000.0
        groups[name].append(0.5 * (z_lo + z_hi))

    report: list[str] = [f"census check against {cad_bbox_path}:"]
    yellow = np.sort(np.asarray(groups["Magnet-yellow"]))
    pink = np.sort(np.asarray(groups["Magnet-pink"]))
    end = np.sort(np.asarray(groups["2nd-Cathode_coil"]))
    embedded_yellow = np.sort(
        np.asarray(YELLOW_NEAR_CENTERS_M + YELLOW_FAR_CENTERS_M)
    )
    for label, parsed, embedded in (
        ("Magnet-yellow", yellow, embedded_yellow),
        ("Magnet-pink", pink, np.asarray(PINK_CENTERS_M)),
        ("2nd-Cathode_coil", end, np.asarray(END_PAIR_CENTERS_M)),
    ):
        if parsed.size != embedded.size:
            raise ValueError(
                f"{label}: CAD has {parsed.size} parts, census has {embedded.size}"
            )
        worst = float(np.max(np.abs(parsed - embedded)))
        if worst > 5.0e-5:
            raise ValueError(f"{label}: census centre mismatch {worst:.3e} m")
        report.append(f"  {label:<18s} n={parsed.size:2d} "
                      f"max |dz| = {worst * 1e6:.1f} um")
    return report


def run_convergence(
    census: Census,
    *,
    bulk_field_gauss: float,
    reference_z_m: float,
    plasma_radius_m: float,
    levels: tuple[tuple[int, int], ...],
    probe_z_m: tuple[float, ...],
    z_start_m: float,
    z_stop_m: float,
    radius_cap_m: float,
    samples: int,
) -> list[dict[str, object]]:
    """Refine the per-coil filament grid and report the change in outputs."""

    rows: list[dict[str, object]] = []
    for n_r, n_z in levels:
        field = CensusField(census, n_r, n_z)
        f_end, _ = solve_droop_min_fraction(
            field, bulk_field_gauss=bulk_field_gauss,
            reference_z_m=reference_z_m, gap_span_m=GAP_SPAN_M,
        )
        current = field.base_current(f_end, bulk_field_gauss, reference_z_m)
        bz = field.axis_bz_gauss(np.asarray(probe_z_m), f_end, current)
        r0, _ = anchor_radius(
            field, z_m=z_start_m, plasma_radius_m=plasma_radius_m,
            bulk_field_gauss=bulk_field_gauss, end_fraction=f_end,
            current=current,
        )
        z_tr, r_tr, _ = trace_flux_surface(
            field, z_start_m=z_start_m, z_stop_m=z_stop_m,
            radius_start_m=r0, end_fraction=f_end, current=current,
            radius_cap_m=radius_cap_m, n_points=samples,
        )
        z_762, _ = first_crossing(z_tr, r_tr, 0.762)
        rows.append({
            "n_r": n_r, "n_z": n_z,
            "n_filaments": field.n_filaments,
            "f_end": f_end,
            "current_a": current,
            "bz": bz,
            "r_valve": float(np.interp(19.02, z_tr, r_tr)),
            "z_762": z_762,
        })
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bulk-field-gauss", type=float, default=1400.0)
    parser.add_argument("--plasma-radius-m", type=float, default=0.18415)
    parser.add_argument("--z-start-m", type=float, default=14.0)
    parser.add_argument("--z-stop-m", type=float, default=22.5)
    parser.add_argument("--samples", type=int, default=4251)
    parser.add_argument("--n-r", type=int, default=4,
                        help="radial filaments per coil (converged default)")
    parser.add_argument("--n-z", type=int, default=4,
                        help="axial filaments per coil (converged default)")
    parser.add_argument("--radius-cap-m", type=float, default=1.5,
                        help="trace terminates if the surface reaches this radius")
    parser.add_argument("--main-bore-radius-m", type=float, default=0.5)
    parser.add_argument("--far-chamber-radius-m", type=float, default=0.762)
    parser.add_argument("--valve-blade-z-m", type=float, default=19.02)
    parser.add_argument("--last-main-coil-z-m", type=float, default=18.475)
    parser.add_argument("--end-pair-centroid-z-m", type=float, default=19.47)
    parser.add_argument("--end-wall-z-m", type=float, default=21.178)
    parser.add_argument("--verify-census", type=Path,
                        help="path to tree_bbox_mag.txt; re-checks the literals")
    parser.add_argument("--limit-check-npz", type=Path,
                        help="old uniform-stack npz to reproduce as a limit gate")
    parser.add_argument("--skip-convergence", action="store_true")
    parser.add_argument("--output-npz", type=Path)
    parser.add_argument("--output-txt", type=Path)
    parser.add_argument("--output-png", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    census = measured_census()
    reference_z_m = 0.5 * (PINK_SPAN_M[0] + PINK_SPAN_M[1])

    emit("LAPD end-field solve on the MEASURED coil census")
    emit("census provenance: design-class SolidWorks export "
         "LAPD_TomLook-Magnets.STEP (2026-08-18),")
    emit("  reduced to tree_bbox_mag.txt; z_model = (-4560 - z_CAD_mm)/1000")
    emit("  yellow 11 + 11 (r_in 0.5651 / r_out 0.7207 m, 0.1588 m thick)")
    emit("  pink   34      (r_in 0.6205 / r_out 0.7500 m, 0.1698 m thick)")
    emit("  end    2       (r_in 0.5842 / r_out 0.7620 m, 0.0889 m thick)")
    emit("source end: 16 RECTANGULAR frames at -0.675..+0.747 m are OMITTED "
         "(not coaxial-circular;")
    emit("  >13 m from the far end) -- as in the predecessor script; bound "
         "reported below.")
    emit(f"current scaling: uniform per-coil ampere-turns, on-axis Bz = "
         f"{args.bulk_field_gauss:.1f} G at the pink-span midpoint "
         f"z = {reference_z_m:.4f} m")
    emit(f"filament discretisation: {args.n_r} radial x {args.n_z} axial per coil")
    emit()

    if args.verify_census is not None:
        for line in verify_census(args.verify_census):
            emit(line)
        emit()

    # --- gate (i): uniform-stack limit ------------------------------------
    if args.limit_check_npz is not None:
        old = np.load(args.limit_check_npz)
        limit_field = CensusField(uniform_stack_census(), 1, 1)
        # The predecessor normalised per AMP with a 1.29 G/A merit; its
        # per-coil ampere-turns are turns * current * calibration.
        amp_turns = (
            float(old["coil_turns"])
            * float(old["current_a"])
            * float(old["field_magnitude_calibration"])
        )
        z_old = old["z_axis_m"]
        bz_new = limit_field.axis_bz_gauss(z_old, 0.0, amp_turns)
        bz_old = old["bz_axis_gauss"]
        rel = float(np.max(np.abs(bz_new - bz_old) / np.abs(bz_old)))
        emit("GATE (i) uniform-stack limit vs "
             f"{args.limit_check_npz.name}:")
        emit(f"  max |dBz|/Bz over {z_old.size} samples = {rel:.3e} "
             f"(tolerance 1e-12)")
        emit(f"  VERDICT: {'PASS' if rel < 1.0e-12 else 'FAIL'}")
        emit()

    # --- gate (ii): interior ripple ---------------------------------------
    field = CensusField(census, args.n_r, args.n_z)
    f_probe, _ = solve_droop_min_fraction(
        field, bulk_field_gauss=args.bulk_field_gauss,
        reference_z_m=reference_z_m, gap_span_m=GAP_SPAN_M,
    )
    i_probe = field.base_current(f_probe, args.bulk_field_gauss, reference_z_m)
    z_pink = np.linspace(PINK_SPAN_M[0], PINK_SPAN_M[1], 20001)
    bz_pink = field.axis_bz_gauss(z_pink, f_probe, i_probe)
    # The 0.32 m periodic ripple and the slow semi-infinite-stack end-effect
    # sag are different quantities; the ripple is measured over ONE central
    # pitch period so the sag cannot contaminate it.
    z_period = np.linspace(reference_z_m - 0.32, reference_z_m, 3201)
    bz_period = field.axis_bz_gauss(z_period, f_probe, i_probe)
    ripple_pp = float(np.ptp(bz_period) / np.mean(bz_period))
    sag_pp = float(np.ptp(bz_pink) / np.max(bz_pink))
    emit("GATE (ii) interior uniformity on axis:")
    emit(f"  periodic ripple over one 0.32 m pitch period at the reference: "
         f"peak-to-peak {np.ptp(bz_period):.4e} G "
         f"({ripple_pp:.3e} relative)")
    emit(f"  pitch 0.32 m / pink coil mean radius "
         f"{0.5 * (PINK_R_IN_M + PINK_R_OUT_M):.4f} m = "
         f"{0.32 / (0.5 * (PINK_R_IN_M + PINK_R_OUT_M)):.4f}")
    emit(f"  separately, the slow sag across the whole pink span "
         f"({PINK_SPAN_M[0]:.3f}..{PINK_SPAN_M[1]:.3f} m) is "
         f"{np.ptp(bz_pink):.3f} G ({sag_pp:.3e} of the peak) -- this is the")
    emit("  semi-infinite-stack end effect of the finite yellow+pink stack, "
         "not coil ripple.")
    emit()

    bound_g = source_coil_bound(
        probe_z_m=args.valve_blade_z_m, bulk_field_gauss=args.bulk_field_gauss,
        reference_z_m=reference_z_m, field=field, end_fraction=f_probe,
    )
    emit(f"omitted source coils: equal-area circular-loop bound at "
         f"z = {args.valve_blade_z_m:.3f} m is {bound_g:.4e} G "
         f"({bound_g / args.bulk_field_gauss:.2e} of B_bulk)")
    emit()

    # --- convergence ------------------------------------------------------
    convergence_rows: list[dict[str, object]] = []
    if not args.skip_convergence:
        probes = (args.last_main_coil_z_m, args.valve_blade_z_m,
                  args.end_pair_centroid_z_m)
        convergence_rows = run_convergence(
            census,
            bulk_field_gauss=args.bulk_field_gauss,
            reference_z_m=reference_z_m,
            plasma_radius_m=args.plasma_radius_m,
            levels=((1, 1), (2, 2), (4, 4), (8, 8)),
            probe_z_m=probes,
            z_start_m=args.z_start_m,
            z_stop_m=args.z_stop_m,
            radius_cap_m=args.radius_cap_m,
            samples=args.samples,
        )
        emit("CONVERGENCE (droop-min case; filaments per coil):")
        emit("   nr x nz   nfil    f_end      I0 [A-t]   "
             "Bz(18.475)  Bz(19.020)  Bz(19.470)   r(19.02)   z[r=0.762]")
        for row in convergence_rows:
            bz = row["bz"]
            emit(f"   {row['n_r']:2d} x {row['n_z']:2d} {row['n_filaments']:7d} "
                 f"{row['f_end']:9.6f} {row['current_a']:11.2f} "
                 f"{bz[0]:11.4f} {bz[1]:11.4f} {bz[2]:11.4f} "
                 f"{row['r_valve']:10.6f} {row['z_762']:11.5f}")
        ref = convergence_rows[-1]
        emit("   change from the previous level to the finest:")
        prev = convergence_rows[-2]
        emit(f"     f_end     {abs(prev['f_end'] - ref['f_end']):.3e}")
        emit(f"     Bz probes {np.max(np.abs(np.asarray(prev['bz']) - np.asarray(ref['bz']))):.3e} G")
        emit(f"     r(19.02)  {abs(prev['r_valve'] - ref['r_valve']):.3e} m")
        emit(f"     z[r=0.762]{abs(prev['z_762'] - ref['z_762']):.3e} m")
        emit()

    # --- the two cases ----------------------------------------------------
    z_axis = np.linspace(args.z_start_m, args.z_stop_m, args.samples)
    results: dict[str, dict[str, object]] = {}
    for case in ("droop-min", "off"):
        if case == "off":
            f_end = 0.0
            droop_max = np.nan
        else:
            f_end, droop_max = solve_droop_min_fraction(
                field, bulk_field_gauss=args.bulk_field_gauss,
                reference_z_m=reference_z_m, gap_span_m=GAP_SPAN_M,
            )
        current = field.base_current(f_end, args.bulk_field_gauss, reference_z_m)
        bz_axis = field.axis_bz_gauss(z_axis, f_end, current)
        r0, psi_target = anchor_radius(
            field, z_m=args.z_start_m, plasma_radius_m=args.plasma_radius_m,
            bulk_field_gauss=args.bulk_field_gauss, end_fraction=f_end,
            current=current,
        )
        z_flux, r_flux, capped = trace_flux_surface(
            field, z_start_m=args.z_start_m, z_stop_m=args.z_stop_m,
            radius_start_m=r0, end_fraction=f_end, current=current,
            radius_cap_m=args.radius_cap_m, n_points=args.samples,
        )
        # gate (iii): flux conservation along the traced surface
        probe_idx = np.unique(np.linspace(0, z_flux.size - 1, 401).astype(int))
        psi = np.array([
            field.flux(float(r_flux[k]), float(z_flux[k]), f_end, current)
            for k in probe_idx
        ])
        psi_rel = float(np.max(np.abs(psi - psi_target) / psi_target))

        z_bore, slope_bore = first_crossing(z_flux, r_flux, args.main_bore_radius_m)
        z_far, slope_far = first_crossing(z_flux, r_flux, args.far_chamber_radius_m)
        r_valve = float(np.interp(args.valve_blade_z_m, z_flux, r_flux))
        bz_valve = float(np.interp(args.valve_blade_z_m, z_axis, bz_axis))
        bz_lastcoil = float(np.interp(args.last_main_coil_z_m, z_axis, bz_axis))
        bz_endpair = float(np.interp(args.end_pair_centroid_z_m, z_axis, bz_axis))
        if z_flux[-1] >= args.end_wall_z_m:
            r_wall = float(np.interp(args.end_wall_z_m, z_flux, r_flux))
            br_w, bz_w = field.field(r_wall, args.end_wall_z_m, f_end, current)
            slope_wall = br_w / bz_w
        else:
            r_wall = float("nan")
            slope_wall = float("nan")
        gap_mask = (z_axis >= GAP_SPAN_M[0]) & (z_axis <= GAP_SPAN_M[1])
        gap_max_dev = float(np.max(np.abs(bz_axis[gap_mask] - args.bulk_field_gauss)))
        gap_min_bz = float(np.min(bz_axis[gap_mask]))

        results[case] = {
            "f_end": f_end, "current_a": current, "bz_axis": bz_axis,
            "z_flux": z_flux, "r_flux": r_flux, "capped": capped,
            "r0": r0, "psi_target": psi_target, "psi_rel": psi_rel,
            "z_bore": z_bore, "slope_bore": slope_bore,
            "z_far": z_far, "slope_far": slope_far,
            "r_valve": r_valve, "bz_valve": bz_valve,
            "bz_lastcoil": bz_lastcoil, "bz_endpair": bz_endpair,
            "r_wall": r_wall, "slope_wall": slope_wall,
            "gap_max_dev": gap_max_dev, "gap_min_bz": gap_min_bz,
            "droop_max": droop_max,
            "trace_end_z": float(z_flux[-1]),
        }

    emit("GATE (iii) flux conservation of the traced surface "
         "(Psi from the same filaments):")
    for case, res in results.items():
        emit(f"  {case:<9s} max |dPsi|/Psi along the trace = {res['psi_rel']:.3e}")
    emit()

    # --- headline table ---------------------------------------------------
    emit("HEADLINE TABLE")
    emit(f"  end-pair current fraction f_end (droop-min) = "
         f"{results['droop-min']['f_end']:.6f} of the per-coil main current")
    emit(f"  droop-min criterion: minimise max |Bz(0,z) - {args.bulk_field_gauss:.0f} G| "
         f"over z in [{GAP_SPAN_M[0]:.2f}, {GAP_SPAN_M[1]:.2f}] m")
    emit()
    emit(f"  {'quantity':<52s} {'droop-min':>14s} {'off':>14s}")
    rows = (
        (f"Bz at valve blade z = {args.valve_blade_z_m:.3f} m [G]",
         "bz_valve", "{:14.3f}"),
        (f"flux radius at valve blade z = {args.valve_blade_z_m:.3f} m [m]",
         "r_valve", "{:14.6f}"),
        (f"Bz at last main coil centre z = {args.last_main_coil_z_m:.3f} m [G]",
         "bz_lastcoil", "{:14.3f}"),
        (f"Bz at end-pair centroid z = {args.end_pair_centroid_z_m:.3f} m [G]",
         "bz_endpair", "{:14.3f}"),
        ("first r = 0.762 m side-wall contact z [m] (open scenario)",
         "z_far", "{:14.4f}"),
        ("dr/dz at that contact [-]", "slope_far", "{:14.5f}"),
        (f"r at end wall z = {args.end_wall_z_m:.3f} m [m] (open scenario)",
         "r_wall", "{:14.6f}"),
        (f"dr/dz at end wall z = {args.end_wall_z_m:.3f} m [-]",
         "slope_wall", "{:14.5f}"),
        ("first r = 0.500 m main-bore contact z [m]", "z_bore", "{:14.4f}"),
        ("dr/dz at main-bore contact [-]", "slope_bore", "{:14.5f}"),
        (f"trace end z [m] (radius cap {args.radius_cap_m:.2f} m)",
         "trace_end_z", "{:14.4f}"),
        (f"max |Bz - B_bulk| over [{GAP_SPAN_M[0]:.2f}, {GAP_SPAN_M[1]:.2f}] m [G]",
         "gap_max_dev", "{:14.3f}"),
        (f"min Bz over [{GAP_SPAN_M[0]:.2f}, {GAP_SPAN_M[1]:.2f}] m [G]",
         "gap_min_bz", "{:14.3f}"),
        ("per-coil current I0 [ampere-turns]", "current_a", "{:14.2f}"),
        ("trace start radius at z = 14.0 m [m]", "r0", "{:14.6f}"),
    )
    for label, key, fmt in rows:
        emit(f"  {label:<52s} "
             f"{fmt.format(results['droop-min'][key])} "
             f"{fmt.format(results['off'][key])}")
    emit()
    emit(f"  max droop over [{GAP_SPAN_M[0]:.2f}, {GAP_SPAN_M[1]:.2f}] m, "
         f"droop-min case: {results['droop-min']['droop_max']:.3f} G "
         f"({results['droop-min']['droop_max'] / args.bulk_field_gauss:.4%} "
         f"of B_bulk)")
    emit()
    emit("  Notes. All flux-surface numbers past the first wall contact are "
         "the VACUUM-FIELD")
    emit("  continuation of the surface, not a plasma statement. A NaN end-wall "
         "entry means the")
    emit("  trace reached the radius cap before the end-wall plane; the trace "
         "end z is tabulated.")
    emit()

    # --- artifacts --------------------------------------------------------
    if args.output_npz is not None:
        payload: dict[str, np.ndarray] = {
            "z_axis_m": z_axis,
            "bulk_field_gauss": np.array(args.bulk_field_gauss),
            "plasma_radius_m": np.array(args.plasma_radius_m),
            "reference_z_m": np.array(reference_z_m),
            "gap_span_m": np.array(GAP_SPAN_M),
            "n_r": np.array(args.n_r),
            "n_z": np.array(args.n_z),
            "coil_centers_yellow_near_m": np.array(YELLOW_NEAR_CENTERS_M),
            "coil_centers_pink_m": np.array(PINK_CENTERS_M),
            "coil_centers_yellow_far_m": np.array(YELLOW_FAR_CENTERS_M),
            "coil_centers_end_pair_m": np.array(END_PAIR_CENTERS_M),
            "coil_radii_yellow_m": np.array([YELLOW_R_IN_M, YELLOW_R_OUT_M]),
            "coil_radii_pink_m": np.array([PINK_R_IN_M, PINK_R_OUT_M]),
            "coil_radii_end_pair_m": np.array([END_R_IN_M, END_R_OUT_M]),
            "coil_thickness_yellow_pink_end_m": np.array(
                [YELLOW_THICK_M, PINK_THICK_M, END_THICK_M]
            ),
            "interior_ripple_relative": np.array(ripple_pp),
            "source_coil_bound_gauss": np.array(bound_g),
        }
        for case, res in results.items():
            tag = case.replace("-", "_")
            payload[f"{tag}_end_current_fraction"] = np.array(res["f_end"])
            payload[f"{tag}_current_ampere_turns"] = np.array(res["current_a"])
            payload[f"{tag}_bz_axis_gauss"] = res["bz_axis"]
            payload[f"{tag}_z_flux_m"] = res["z_flux"]
            payload[f"{tag}_flux_radius_m"] = res["r_flux"]
            payload[f"{tag}_flux_psi_relative_error"] = np.array(res["psi_rel"])
            payload[f"{tag}_crossing_radii_m"] = np.array(
                [args.main_bore_radius_m, args.far_chamber_radius_m]
            )
            payload[f"{tag}_crossing_z_m"] = np.array(
                [res["z_bore"], res["z_far"]]
            )
            payload[f"{tag}_crossing_slope"] = np.array(
                [res["slope_bore"], res["slope_far"]]
            )
            payload[f"{tag}_bz_valve_blade_gauss"] = np.array(res["bz_valve"])
            payload[f"{tag}_flux_radius_valve_blade_m"] = np.array(res["r_valve"])
            payload[f"{tag}_bz_last_main_coil_gauss"] = np.array(res["bz_lastcoil"])
            payload[f"{tag}_bz_end_pair_centroid_gauss"] = np.array(res["bz_endpair"])
            payload[f"{tag}_r_end_wall_m"] = np.array(res["r_wall"])
            payload[f"{tag}_slope_end_wall"] = np.array(res["slope_wall"])
            payload[f"{tag}_gap_max_deviation_gauss"] = np.array(res["gap_max_dev"])
            payload[f"{tag}_gap_min_bz_gauss"] = np.array(res["gap_min_bz"])
            payload[f"{tag}_trace_end_z_m"] = np.array(res["trace_end_z"])
            payload[f"{tag}_trace_hit_radius_cap"] = np.array(res["capped"])
            payload[f"{tag}_radius_cap_m"] = np.array(args.radius_cap_m)
        if convergence_rows:
            payload["convergence_n_r"] = np.array(
                [r["n_r"] for r in convergence_rows]
            )
            payload["convergence_n_z"] = np.array(
                [r["n_z"] for r in convergence_rows]
            )
            payload["convergence_f_end"] = np.array(
                [r["f_end"] for r in convergence_rows]
            )
            payload["convergence_bz_probes_gauss"] = np.array(
                [r["bz"] for r in convergence_rows]
            )
            payload["convergence_probe_z_m"] = np.array(
                [args.last_main_coil_z_m, args.valve_blade_z_m,
                 args.end_pair_centroid_z_m]
            )
            payload["convergence_r_valve_m"] = np.array(
                [r["r_valve"] for r in convergence_rows]
            )
            payload["convergence_z_far_wall_m"] = np.array(
                [r["z_762"] for r in convergence_rows]
            )
        args.output_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output_npz, **payload)
        emit(f"wrote {args.output_npz}")

    if args.output_txt is not None:
        args.output_txt.parent.mkdir(parents=True, exist_ok=True)
        args.output_txt.write_text("\n".join(lines) + "\n")
        print(f"wrote {args.output_txt}")

    if args.output_png is not None:
        fig, (ax_b, ax_r) = plt.subplots(
            2, 1, figsize=(8.6, 7.0), sharex=True, constrained_layout=True
        )
        for case, color in (("droop-min", "C0"), ("off", "C3")):
            res = results[case]
            ax_b.plot(z_axis, res["bz_axis"] / args.bulk_field_gauss,
                      color=color, lw=2.0,
                      label=f"{case} (f_end = {res['f_end']:.3f})")
            ax_r.plot(res["z_flux"], res["r_flux"], color=color, lw=2.0,
                      label=case)
        for ax in (ax_b, ax_r):
            ax.axvspan(*GAP_SPAN_M, color="0.85", alpha=0.5, zorder=0)
            ax.axvline(args.valve_blade_z_m, color="0.3", ls="--", lw=1.0)
            ax.axvline(args.last_main_coil_z_m, color="0.5", ls=":", lw=1.0)
            ax.grid(alpha=0.2)
        ax_b.axhline(1.0, color="0.55", ls=":", lw=1.0)
        ax_b.set_ylabel(r"$B_z(0,z)/B_{\rm bulk}$")
        ax_b.set_title("LAPD far-end field on the measured coil census "
                       "(CAD 2026-08-18)")
        ax_b.legend(loc="lower left", frameon=False)
        ax_r.axhline(args.main_bore_radius_m, color="0.45", lw=1.0)
        ax_r.axhline(args.far_chamber_radius_m, color="0.45", lw=1.0)
        ax_r.set_xlabel("z (model coordinate, from the LaB6 face) [m]")
        ax_r.set_ylabel(r"$R_p = 18.415$ cm flux-surface radius [m]")
        ax_r.legend(loc="upper left", frameon=False)
        args.output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output_png, dpi=180)
        print(f"wrote {args.output_png}")


if __name__ == "__main__":
    main()
