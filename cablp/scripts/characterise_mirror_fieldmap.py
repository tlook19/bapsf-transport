#!/usr/bin/env python3
"""Characterise the solved LAPD axial field map on a LAPDSim1D mesh.

READ-ONLY. This runs no simulation, constructs no ``LAPDSim1D``, and writes
nothing but its own report and figure. It puts ``B_z(z)``, ``dB/dz``, the
mirror ratio and the traced flux-surface radius on the production mesh, and
says where the gradient is large enough to matter and over how much of the
domain.

It evaluates NO force, and there is none to evaluate: the fluid mirror force
is already in the model as the quasi-1D ``p dA/dz`` source
(``physics.sources.flux_tube_geometry_rhs``, armed by
``prescribed_area_geometry``, which the stance of record sets). Flux
conservation gives ``A ∝ 1/B``, so under the isotropic closure that source IS
the distribution average of ``-mu grad_par B``, term for term, and a second
``-mu grad B`` term would double-count it exactly. This script is the
measurement that lets that identity be stated against numbers: the ``A(z)``
the stance runs is traced from the SAME map read here. See
``cablp/solvers/_sim1d/physics/mirror_field.py`` and ``MODEL.md``
("Prescribed flux-tube and vessel geometry -> Mirror force").

It is also the loader's only in-tree consumer -- deliberately, since the
loader ships no solver configuration key.

Meshes
------
Two, because they answer different questions and disagree by more than
rounding at the end fringe:

``stance``
    The stance of record (``scripts/stances/g1atrim.toml``) as production runs
    it -- 280 cells, 7.5 cm through the far column. This is the mesh a
    registration should read.
``golden``
    The same stance at ``nx = 60``, which is what the golden fixture pins. Its
    far-column cells are ~34 cm, so it resolves the end fringe far more
    coarsely; reported so the difference is visible rather than assumed away.

Thresholds
----------
Two scale-free measures of "significant gradient", both reported against
explicit thresholds rather than a felt judgement:

``per-cell relative change``  ``|B(z+) - B(z-)| / B_cell`` across one cell --
    how much the field moves over a mesh cell. This is the number a
    cell-centred term's truncation error scales with.
``inverse gradient length``  ``|dB/dz| / B`` [1/m] -- the reciprocal scale
    length ``1/L_B``, independent of the mesh.

Both are tabulated as the fraction of CELLS and, separately, the fraction of
axial LENGTH above each threshold: the mesh is graded, so the two fractions
are not the same number and quoting only the cell count overstates a fringe
that lives in short cells.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from stance_config import stance_config  # noqa: E402

from cablp.solvers._sim1d import default_config  # noqa: E402
from cablp.solvers._sim1d.core.geometry import build_geometry  # noqa: E402
from cablp.solvers._sim1d.physics.mirror_field import (  # noqa: E402
    MIRROR_FIELD_CASES,
    load_mirror_field,
)

DEFAULT_MAP = _HERE / "lapd_end_field_Rp18p415.npz"

# Fractional change of B across one mesh cell.
CELL_CHANGE_THRESHOLDS = (1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1)
# Inverse gradient length |dB/dz|/B, per metre.
INVERSE_LB_THRESHOLDS_PER_M = (1.0e-3, 1.0e-2, 1.0e-1, 1.0)


def _fractions(mask, lengths_cm):
    """Return (fraction of cells, fraction of axial length) under ``mask``."""
    total_cells = mask.size
    total_length = float(lengths_cm.sum())
    return (
        float(np.count_nonzero(mask)) / total_cells,
        float(lengths_cm[mask].sum()) / total_length,
    )


def _describe_mesh(emit, label, geometry):
    lengths = np.asarray(geometry.length_cm, dtype=float)
    emit(f"  mesh                     {label}")
    emit(f"  cells                    {geometry.cells}  (nx = {geometry.nx})")
    emit(
        f"  axial span               [{geometry.z_edges_cm[0]:.2f}, "
        f"{geometry.z_edges_cm[-1]:.2f}] cm"
    )
    emit(
        f"  cell length min/max      {lengths.min():.3f} / {lengths.max():.3f} cm"
    )


def _characterise(emit, profile, geometry, *, table_from_cm):
    lengths = np.asarray(geometry.length_cm, dtype=float)
    z = profile.z_cm
    b = profile.B_cell_gauss
    dbdz_native = profile.dBdz_native_cell_gauss_per_cm
    dbdz_mesh = profile.dBdz_mesh_cell_gauss_per_cm

    emit("")
    emit("  -- field on the mesh (gauss; z in cm; dB/dz in gauss/cm) --")
    emit(f"  map axial support        [{profile.map_z_min_cm:.1f}, "
         f"{profile.map_z_max_cm:.1f}] cm")
    emit(f"  interior fill            {profile.interior_fill_gauss:.4f} G "
         f"on {int(profile.interior_fill_cell.sum())} of {b.size} cells "
         f"(z < {profile.map_z_min_cm:.1f} cm)")
    emit(f"  interior ripple bound    {profile.interior_ripple_relative:.6e} "
         "of B_bulk (peak, census-measured; the fill discards its gradient)")
    emit(f"  B_bulk (map record)      {profile.B_bulk_gauss:.4f} G")
    emit(f"  B min / max on mesh      {profile.B_min_gauss:.4f} / "
         f"{profile.B_max_gauss:.4f} G")
    emit(f"  argmax B                 z = {z[int(np.argmax(b))]:.2f} cm")
    emit(f"  argmin B                 z = {z[int(np.argmin(b))]:.2f} cm")
    emit(f"  max B / B_bulk           {profile.mirror_ratio_bulk_cell.max():.6f}")
    emit(f"  min B / B_bulk           {profile.mirror_ratio_bulk_cell.min():.6f}")
    emit(f"  mirror ratio max         {profile.mirror_ratio_cell.max():.4f}  "
         "(B_max / B_min over this mesh)")

    diff = np.abs(b - profile.B_cell_average_gauss)
    worst = int(np.argmax(diff))
    emit("")
    emit("  -- sampling: point value vs cell average of the same interpolant --")
    emit(f"  max |point - average|    {diff.max():.4f} G at z = {z[worst]:.2f} cm "
         f"({diff.max() / max(b[worst], 1e-30):.4%} of the local field)")
    emit(f"  L2 over the mesh         {float(np.sqrt((diff ** 2).mean())):.4f} G")

    emit("")
    emit("  -- gradient: native 2 mm differencing vs mesh differencing --")
    emit(f"  max |dB/dz| native       {np.abs(dbdz_native).max():.4f} G/cm "
         f"at z = {z[int(np.argmax(np.abs(dbdz_native)))]:.2f} cm")
    emit(f"  max |dB/dz| mesh         {np.abs(dbdz_mesh).max():.4f} G/cm "
         f"at z = {z[int(np.argmax(np.abs(dbdz_mesh)))]:.2f} cm")
    gap = np.abs(dbdz_native - dbdz_mesh)
    emit(f"  max native - mesh        {gap.max():.4f} G/cm at "
         f"z = {z[int(np.argmax(gap))]:.2f} cm")

    emit("")
    emit("  -- how much of the domain sees a gradient --")
    cell_change = np.abs(dbdz_mesh) * lengths / np.maximum(b, 1e-30)
    emit("  |dB| across one cell, relative to the local field:")
    emit("    threshold      cells        of cells     of axial length")
    for thr in CELL_CHANGE_THRESHOLDS:
        mask = cell_change > thr
        fc, fl = _fractions(mask, lengths)
        emit(f"    > {thr:<10.0e} {int(mask.sum()):>5d}        "
             f"{fc:>8.4%}       {fl:>8.4%}")
    inv_lb_per_m = 100.0 * np.abs(dbdz_native) / np.maximum(b, 1e-30)
    emit("  inverse gradient length |dB/dz|/B [1/m], native differencing:")
    emit("    threshold      cells        of cells     of axial length")
    for thr in INVERSE_LB_THRESHOLDS_PER_M:
        mask = inv_lb_per_m > thr
        fc, fl = _fractions(mask, lengths)
        emit(f"    > {thr:<10.0e} {int(mask.sum()):>5d}        "
             f"{fc:>8.4%}       {fl:>8.4%}")
    significant = inv_lb_per_m > INVERSE_LB_THRESHOLDS_PER_M[1]
    if np.any(significant):
        first = int(np.flatnonzero(significant)[0])
        emit(f"  first cell with |dB/dz|/B > "
             f"{INVERSE_LB_THRESHOLDS_PER_M[1]:.0e} /m: "
             f"z = {z[first]:.2f} cm (cell {first})")

    emit("")
    emit("  -- mirror ratio thresholds (B / B_bulk) --")
    ratio = profile.mirror_ratio_bulk_cell
    for thr in (1.01, 1.05, 1.10):
        mask = ratio > thr
        fc, fl = _fractions(mask, lengths)
        emit(f"    B/B_bulk > {thr:<5.2f}   {int(mask.sum()):>5d} cells   "
             f"{fc:>8.4%} of cells   {fl:>8.4%} of length")
    for thr in (0.9, 0.5, 0.1):
        mask = ratio < thr
        fc, fl = _fractions(mask, lengths)
        emit(f"    B/B_bulk < {thr:<5.2f}   {int(mask.sum()):>5d} cells   "
             f"{fc:>8.4%} of cells   {fl:>8.4%} of length")

    emit("")
    emit("  -- traced flux surface (VACUUM-FIELD CONTINUATION past first wall "
         "contact) --")
    emit(f"  anchored on column       Rp = {100.0 * profile.plasma_radius_m:.4f} cm")
    emit(f"  first wall contact       z = {profile.first_wall_contact_z_cm:.2f} cm")
    emit(f"  bore crossings           " + ", ".join(
        f"r = {r:.2f} cm at z = {zz:.2f} cm"
        for r, zz in zip(profile.wall_crossing_radii_cm, profile.wall_crossing_z_cm)
    ))
    emit(f"  trace terminates         z = {profile.trace_end_z_cm:.2f} cm "
         "(radius cap)")
    emit(f"  cells with a traced r    {int(profile.flux_radius_valid_cell.sum())} "
         f"of {b.size}")
    emit(f"  of those, past contact   "
         f"{int(profile.flux_radius_vacuum_continuation_cell.sum())} "
         "(vacuum continuation -- NOT a plasma statement)")
    nan_cells = int((~profile.flux_radius_valid_cell & (z > profile.map_z_min_cm)).sum())
    if nan_cells:
        emit(f"  mesh cells past the trace {nan_cells} (radius returned NaN)")

    emit("")
    emit(f"  -- per-cell table from z = {table_from_cm:.0f} cm --")
    emit("      z [cm]      B [G]   dB/dz_nat  dB/dz_mesh   B/B_bulk    "
         "L_B [m]    r_flux [cm]")
    for i in np.flatnonzero(z >= table_from_cm):
        lb = b[i] / max(abs(dbdz_native[i]), 1e-30) / 100.0
        emit(f"  {z[i]:10.2f} {b[i]:10.3f} {dbdz_native[i]:11.4f} "
             f"{dbdz_mesh[i]:11.4f} {ratio[i]:10.5f} {lb:10.3f} "
             f"{profile.flux_radius_cell_cm[i]:12.3f}")


def _figure(path, profiles, geometry_label):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(9.0, 10.0), sharex=True)
    for case, profile in profiles.items():
        z_m = profile.z_cm / 100.0
        axes[0].plot(z_m, profile.B_cell_gauss, marker=".", ms=3, label=case)
        axes[1].plot(
            z_m, profile.dBdz_native_cell_gauss_per_cm, marker=".", ms=3, label=case
        )
        axes[2].plot(z_m, profile.mirror_ratio_bulk_cell, marker=".", ms=3, label=case)
    any_profile = next(iter(profiles.values()))
    for ax in axes:
        ax.axvline(any_profile.map_z_min_cm / 100.0, color="0.6", ls=":", lw=1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("B_z on axis [G]")
    axes[0].set_title(
        f"LAPD axial field on the {geometry_label} mesh\n"
        "dotted line: low edge of the solved map; below it, uniform fill "
        "(dB/dz = 0 by construction)",
        fontsize=10,
    )
    axes[1].set_ylabel("dB/dz [G/cm], native differencing")
    axes[2].set_ylabel("B / B_bulk")
    axes[2].set_xlabel("z [m]  (z = 0 at the cathode face)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--map", default=str(DEFAULT_MAP),
        help="field-map .npz (default: scripts/lapd_end_field_Rp18p415.npz)",
    )
    parser.add_argument(
        "--stance", default="g1atrim",
        help="stance whose mesh is characterised (default: g1atrim)",
    )
    parser.add_argument(
        "--output", default=None,
        help="write the report here as well as to stdout",
    )
    parser.add_argument(
        "--figure", default=None,
        help="render B, dB/dz and B/B_bulk to this PNG (stance mesh)",
    )
    parser.add_argument(
        "--table-from-cm", type=float, default=1750.0,
        help="per-cell table starts at this z (default 1750 cm)",
    )
    args = parser.parse_args(argv)

    lines = []

    def emit(text=""):
        lines.append(text)
        print(text)

    emit("LAPD axial field-map characterisation for LAPDSim1D")
    emit("=" * 72)
    emit(f"map            {Path(args.map).resolve()}")
    emit(f"stance         {args.stance}")
    emit("cases          " + ", ".join(MIRROR_FIELD_CASES))
    emit("units          B in gauss, z in cm, dB/dz in gauss/cm (CGS)")
    emit("coordinate     z = 0 at the LaB6 emitting face, +z toward the far end")
    emit("")
    emit("This is a READ-ONLY characterisation. No force term exists yet; its")
    emit("form is a pending registration.")

    stance_params, stance_flags = stance_config(args.stance)
    golden_params, golden_flags = stance_config(args.stance)
    golden_params["nx"] = 60
    # The golden mesh cannot carry the stance's per-cell profiles: they are
    # sized to nx = 268. Drop them and the flag that requires them, exactly as
    # the golden config does.
    for key in (
        "plasma_radius_profile_cm",
        "machine_radius_profile_cm",
        "neutral_baffle_positions_cm",
        "neutral_baffle_clear_radii_cm",
    ):
        golden_params[key] = None
    golden_flags["prescribed_area_geometry"] = False
    golden_flags["neutral_baffles"] = False

    default_params, default_flags = default_config()

    meshes = (
        ("stance g1atrim (nx = 268)", stance_params, stance_flags),
        ("golden-style (stance at nx = 60)", golden_params, golden_flags),
        ("default_config()", default_params, default_flags),
    )

    stance_profiles = {}
    for label, params, flags in meshes:
        geometry = build_geometry(params, flags)
        emit("")
        emit("=" * 72)
        _describe_mesh(emit, label, geometry)
        for case in MIRROR_FIELD_CASES:
            profile = load_mirror_field(
                map_path=args.map,
                case=case,
                geometry=geometry,
                plasma_radius_cm=params["Rp"],
            )
            if label.startswith("stance"):
                stance_profiles[case] = profile
            emit("")
            emit("-" * 72)
            emit(f"  CASE {case}")
            _characterise(
                emit, profile, geometry, table_from_cm=args.table_from_cm
            )

    if args.output:
        Path(args.output).write_text("\n".join(lines) + "\n")
        print(f"\nreport written to {Path(args.output).resolve()}")
    if args.figure:
        _figure(args.figure, stance_profiles, "stance g1atrim")
        print(f"figure written to {Path(args.figure).resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
