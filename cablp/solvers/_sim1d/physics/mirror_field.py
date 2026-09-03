"""Axial magnetic-field map ingestion for the LAPD flux-tube geometry.

This module is a READ-ONLY LOADER. It reads the solved LAPD end-field map,
puts ``B_z`` and its axial derivative on the solver's mesh, and reports the
derived mirror ratio and traced flux-surface radius. **It contains no force
term, and one must never be added here.**

Why there is no force term here
-------------------------------
Because the mirror force is ALREADY IN the model, and has been since
``prescribed_area_geometry`` shipped. The quasi-1D momentum equation the
solver integrates under that flag is

    d_t(A rho u) + d_z[A(rho u^2 + p)] = p dA/dz + A S_M,

and the ``p dA/dz`` source (``physics.sources.flux_tube_geometry_rhs``) IS the
fluid mirror force. Flux conservation along a field line gives ``A B =
const``, so ``A ∝ 1/B`` exactly; under the isotropic closure the state vector
``(n, nn, M, Ee, Ei)`` supports, the distribution average of ``-mu grad_par B``
reduces term for term to that source. The stance's ``A(z)`` is traced from the
SAME field map this loader reads, with the trace's own flux-conservation
residual recorded in the map as ``{case}_flux_psi_relative_error`` (max
``|dPsi|/Psi`` of 3.947e-9 for ``droop_min`` and 3.094e-9 for ``off`` in the
Rp = 18.415 cm census). So the two are not merely the same physics in
principle -- they are the same field, the same surface, the same numbers.

A ``-mu grad B`` term added alongside it would therefore double-count the
mirror force by 100 %, exactly. That is not a modelling preference; it is an
identity. ``MODEL.md`` states it under "Prescribed flux-tube and vessel
geometry -> Mirror force", together with the matching statement for the energy
side: expansion cooling through the flare is carried by
``sources.velocity_divergence`` inside the existing pressure work, so there is
no separate mirror-cooling source either.

The one term this module could ever legitimately feed is the pressure-
ANISOTROPY correction ``(p_perp - p_par)/B dB/dz`` -- the residual the
isotropic reduction discards. It is not built, and it is not merely unbuilt
but unsupported: it needs a ``p_perp``/``p_par`` closure that this state
vector does not carry, and its magnitude is bounded at a few percent of the
isotropic term anyway. Registering it would be a new closure, not a flag.

Consequently this module ships NO solver configuration key and NO flag. Its
consumer is ``scripts/characterise_mirror_fieldmap.py`` (at commit 48be9a4,
retired 2026-09-03), which measures what
the field does on the production mesh so that the identity above can be
checked against numbers rather than asserted. Nothing in the solver imports
it, and nothing in the solver should: the solver does no file I/O, and the
field-to-area conversion happens outside it (``scripts/g1_build_profiles.py``).

What the map is
---------------
``scripts/solve_lapd_coil_field_census.py`` solves the axisymmetric vacuum
field of the measured LAPD coil census (exact circular-filament Biot-Savart,
complete elliptic integrals) and writes an ``.npz``. Two named end-coil cases
are solved and both are carried in the same file, distinguished by an array-
name prefix:

``droop_min``
    The far ("end") coil pair carries the current fraction ``f_end`` that
    minimises ``max |B_z(0,z) - B_bulk|`` over the bridged gap span
    ``[18.55, 19.56] m``.
``off``
    ``f_end = 0`` -- the end pair unpowered.

The two are NOT small perturbations of each other: at the end-pair centroid
they differ by an order of magnitude, so ``case`` is a REQUIRED argument with
no default reading rather than something inferred.

Coordinate and units
--------------------
The map's ``z_axis_m`` is the SAME axial coordinate the solver uses: ``z = 0``
at the LaB6 emitting face, ``+z`` toward the far end (the census reduces the
CAD export by ``z_model = (-4560 - z_CAD_mm)/1000``, and ``core.geometry``
puts the cathode surface at ``z = 0`` with the plenum at negative ``z``).
Everything this module returns is CGS to match the solver: ``B`` in gauss,
``z`` in cm, ``dB/dz`` in gauss/cm. The map itself stores gauss and METRES;
the conversion happens here, once.

Axial coverage, and the interior fill
-------------------------------------
The map is an END-FIELD map: it is solved on ``z in [14.0, 22.5] m`` only.
The solver mesh starts behind the cathode at negative ``z``, so most of the
domain lies below the map. Cells below the map's low edge are filled with a
single uniform value -- by default the map's own recorded ``bulk_field_gauss``
(1400 G, the machine-ruled level the census normalises its per-coil current
to). That fill is a DECLARED approximation, not a solved field: the census
measures the interior coil ripple at ``interior_ripple_relative`` (5.43e-5 of
``B_bulk``, peak, at the 0.32 m pink-stack pitch), so the fill is accurate in
MAGNITUDE to that bound but sets ``dB/dz = 0`` exactly, discarding a ripple
gradient of order ``2*pi*5.43e-5*B_bulk/0.32 m``. Any consumer that cares
about ``dB/dz`` rather than ``B`` must decide whether that discard is
acceptable; the loader states it rather than hiding it. Cells ABOVE the map's
high edge raise -- there is nothing to extend from.

Flux-surface radius, and the vacuum-continuation caveat
-------------------------------------------------------
The map also carries the traced flux surface anchored on the ``Rp`` column,
``{case}_z_flux_m`` / ``{case}_flux_radius_m``. **All flux-surface numbers
past the first wall contact are the VACUUM-FIELD CONTINUATION of the surface,
not a plasma statement** (the census script's own note). The traced surface
crosses the 0.5 m main bore, and then the 0.762 m far-chamber bore, at
``{case}_crossing_z_m``; beyond the first of those the radius is what the
vacuum field would do if the wall were not there. The trace also terminates
at a radius cap, which for the ``off`` case is BELOW the far end of the
production mesh -- the radius is returned as NaN there, with an explicit
validity mask, rather than extrapolated.

``flux_radius_cell_cm`` is returned raw, in the trace's own anchoring. The
in-repo convention for turning it into a flux-tube radius is the RATIO form
``Rp * r_flux(z) / r_flux(z_ref)`` used by ``scripts/g1_build_profiles.py``,
which divides out the traced surface's anchored-sag offset; this module does
not apply it, because which anchor a consumer wants is the consumer's choice.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


# The two end-coil cases the census solves, and the array-name prefix each
# uses inside the .npz. A case outside this mapping is a hard error: there is
# no third case in the file to fall back to.
MIRROR_FIELD_CASES = ("droop_min", "off")


@dataclass(frozen=True)
class MirrorFieldProfile:
    """The end-field map resolved onto one solver mesh. CGS throughout.

    Cell-indexed arrays have ``geometry.cells`` entries; face-indexed arrays
    have ``geometry.cells + 1``.

    case:
        Which end-coil case was read (``"droop_min"`` or ``"off"``).
    map_path:
        Absolute path of the ``.npz`` this was read from.
    z_cm / z_edges_cm:
        The mesh this profile is bound to, copied so the record cannot be
        read against a different geometry by accident.
    B_cell_gauss / B_face_gauss:
        On-axis ``B_z`` POINT-SAMPLED at cell centres / faces by piecewise-
        linear interpolation of the map's native 2 mm axial grid.
    B_cell_average_gauss:
        On-axis ``B_z`` CELL-AVERAGED instead: the exact integral of the same
        piecewise-linear interpolant over each cell, divided by the cell
        length. Supplied alongside the point sample so a consumer can measure
        what the sampling choice costs rather than assume it is negligible.
    dBdz_native_cell_gauss_per_cm:
        ``dB/dz`` differenced on the map's NATIVE 2 mm grid (central
        differences) and then sampled at cell centres. This is the local
        gradient, coil ripple included.
    dBdz_mesh_cell_gauss_per_cm:
        ``dB/dz`` differenced on the MESH: ``(B_face[i+1] - B_face[i])`` over
        the cell length. This is what a finite-volume term built from face
        values sees, and it is not the same number as the native gradient
        wherever the field has structure below the cell scale.
    mirror_ratio_cell:
        ``B_cell / B_min_gauss`` -- the local field normalised to the weakest
        point ON THIS MESH.
    mirror_ratio_bulk_cell:
        ``B_cell / B_bulk_gauss`` -- normalised to the map's recorded bulk
        level instead. Reported separately because the two normalisations
        answer different questions and neither is privileged here.
    B_min_gauss / B_max_gauss:
        Extrema of ``B_cell_gauss`` over the mesh.
    B_bulk_gauss:
        The map's recorded ``bulk_field_gauss``.
    interior_fill_gauss:
        The uniform value written into cells below the map's low edge.
    interior_fill_cell:
        Boolean mask, true where that fill was used instead of the map.
    interior_ripple_relative:
        The census's measured peak interior ripple, as a fraction of
        ``B_bulk``. The accuracy bound on ``interior_fill_gauss``.
    flux_radius_cell_cm:
        Traced vacuum flux-surface radius at each cell centre, NaN outside
        the trace's own axial support. Carries the vacuum-continuation
        caveat in the module docstring.
    flux_radius_valid_cell:
        Boolean mask, true where ``flux_radius_cell_cm`` is a traced value.
    flux_radius_vacuum_continuation_cell:
        Boolean mask, true where the traced radius is past the FIRST wall
        contact and is therefore vacuum continuation rather than a statement
        about a plasma column.
    first_wall_contact_z_cm:
        Axial position of the first traced crossing of a vessel bore, or NaN
        if the trace never reaches one.
    wall_crossing_z_cm / wall_crossing_radii_cm:
        All traced bore crossings, as recorded by the census.
    trace_end_z_cm:
        Where the flux trace terminated (it stops at a radius cap).
    map_z_min_cm / map_z_max_cm:
        Axial support of the solved field map.
    plasma_radius_m:
        The column radius the flux surface was anchored on, from the map.
    """

    case: str
    map_path: str
    z_cm: np.ndarray
    z_edges_cm: np.ndarray
    B_cell_gauss: np.ndarray
    B_face_gauss: np.ndarray
    B_cell_average_gauss: np.ndarray
    dBdz_native_cell_gauss_per_cm: np.ndarray
    dBdz_mesh_cell_gauss_per_cm: np.ndarray
    mirror_ratio_cell: np.ndarray
    mirror_ratio_bulk_cell: np.ndarray
    B_min_gauss: float
    B_max_gauss: float
    B_bulk_gauss: float
    interior_fill_gauss: float
    interior_fill_cell: np.ndarray
    interior_ripple_relative: float
    flux_radius_cell_cm: np.ndarray
    flux_radius_valid_cell: np.ndarray
    flux_radius_vacuum_continuation_cell: np.ndarray
    first_wall_contact_z_cm: float
    wall_crossing_z_cm: np.ndarray
    wall_crossing_radii_cm: np.ndarray
    trace_end_z_cm: float
    map_z_min_cm: float
    map_z_max_cm: float
    plasma_radius_m: float


def _require(payload, key, map_path):
    """Return ``payload[key]``, naming the file when the key is absent."""
    if key not in payload:
        raise ValueError(
            f"field map {map_path} has no array {key!r}; it does not look "
            "like a solve_lapd_coil_field_census.py output "
            f"(arrays present: {', '.join(sorted(payload.files))})"
        )
    return payload[key]


def _cell_average(z_native_cm, b_native_gauss, z_edges_cm, lengths_cm):
    """Return the exact cell mean of the piecewise-linear map interpolant.

    Integrating the interpolant analytically (cumulative trapezoid on the
    native grid, sampled at the cell edges) rather than sub-sampling it keeps
    the average independent of any quadrature choice made here.
    """
    dz = np.diff(z_native_cm)
    integral = np.concatenate(
        ([0.0], np.cumsum(0.5 * dz * (b_native_gauss[1:] + b_native_gauss[:-1])))
    )
    at_edges = np.interp(z_edges_cm, z_native_cm, integral)
    return np.diff(at_edges) / lengths_cm


def load_mirror_field(
    *,
    map_path,
    case,
    geometry,
    interior_fill_gauss=None,
    plasma_radius_cm=None,
):
    """Read the end-field map and resolve it onto ``geometry``.

    A LIBRARY function, called with explicit arguments. There is no solver
    configuration key and no flag behind it: see the module docstring for why
    a solver-side control here would gate a force that is already in the
    model. Callers pass a geometry they built themselves.

    ``map_path`` and ``case`` are required keyword arguments with no defaults,
    so omitting either is a ``TypeError`` at the call site rather than a
    guessed reading.

    map_path:
        Path to a ``solve_lapd_coil_field_census.py`` ``.npz``.
    case:
        One of ``MIRROR_FIELD_CASES``. No default: the two cases differ by an
        order of magnitude at the end pair.
    geometry:
        A ``Sim1DGeometry``; ``z_cm``, ``z_edges_cm`` and ``length_cm`` are
        read.
    interior_fill_gauss:
        Uniform value for mesh cells below the map's axial support. ``None``
        uses the map's recorded ``bulk_field_gauss``.
    plasma_radius_cm:
        If given, the column radius the caller's configuration uses. It is
        checked against the radius the map's flux surface was anchored on and
        a mismatch raises -- a flux trace anchored on a different column is
        not a trace of this one.

    Raises ``ValueError`` on a missing file, an unknown case, a malformed or
    non-monotonic map, a mesh that runs past the map's high edge, or a plasma
    radius that disagrees with the map's anchor.
    """
    if case not in MIRROR_FIELD_CASES:
        raise ValueError(
            f"load_mirror_field(case=...) must be one of "
            f"{MIRROR_FIELD_CASES} (got {case!r}); the .npz carries exactly "
            "these end-coil cases and there is no default reading -- they "
            "differ by an order of magnitude at the end pair"
        )
    path = Path(map_path).expanduser()
    if not path.is_file():
        raise ValueError(
            "load_mirror_field(map_path=...) does not name a readable "
            f"file: {path}"
        )
    with np.load(path) as payload:
        z_native_cm = 100.0 * np.asarray(
            _require(payload, "z_axis_m", path), dtype=float
        )
        b_native_gauss = np.asarray(
            _require(payload, f"{case}_bz_axis_gauss", path), dtype=float
        )
        b_bulk = float(_require(payload, "bulk_field_gauss", path))
        ripple = float(_require(payload, "interior_ripple_relative", path))
        map_radius_m = float(_require(payload, "plasma_radius_m", path))
        z_flux_cm = 100.0 * np.asarray(
            _require(payload, f"{case}_z_flux_m", path), dtype=float
        )
        r_flux_cm = 100.0 * np.asarray(
            _require(payload, f"{case}_flux_radius_m", path), dtype=float
        )
        crossing_z_cm = 100.0 * np.asarray(
            _require(payload, f"{case}_crossing_z_m", path), dtype=float
        )
        crossing_r_cm = 100.0 * np.asarray(
            _require(payload, f"{case}_crossing_radii_m", path), dtype=float
        )
        trace_end_cm = 100.0 * float(
            _require(payload, f"{case}_trace_end_z_m", path)
        )

    if z_native_cm.ndim != 1 or z_native_cm.size < 2:
        raise ValueError(
            f"field map {path}: z_axis_m must be a 1D grid of at least two "
            f"points (got shape {z_native_cm.shape})"
        )
    if b_native_gauss.shape != z_native_cm.shape:
        raise ValueError(
            f"field map {path}: {case}_bz_axis_gauss has shape "
            f"{b_native_gauss.shape} but z_axis_m has {z_native_cm.shape}"
        )
    if not np.all(np.diff(z_native_cm) > 0.0):
        raise ValueError(
            f"field map {path}: z_axis_m is not strictly increasing, so the "
            "interpolation onto the mesh is not defined"
        )
    if not np.all(np.isfinite(b_native_gauss)):
        raise ValueError(
            f"field map {path}: {case}_bz_axis_gauss is not everywhere finite"
        )
    if b_bulk <= 0.0 or not np.isfinite(b_bulk):
        raise ValueError(
            f"field map {path}: bulk_field_gauss must be finite and positive "
            f"(got {b_bulk})"
        )

    if plasma_radius_cm is not None:
        want = float(plasma_radius_cm)
        have = 100.0 * map_radius_m
        if abs(want - have) > 1.0e-6 * max(abs(have), 1.0):
            raise ValueError(
                f"field map {path} traced its flux surface on a column of "
                f"radius {have} cm, but this configuration runs Rp = {want} "
                "cm. The traced surface belongs to the column it was "
                "anchored on; re-solve the census at this Rp instead of "
                "reading the wrong surface"
            )

    if interior_fill_gauss is None:
        fill = b_bulk
    else:
        fill = float(interior_fill_gauss)
        if not np.isfinite(fill) or fill <= 0.0:
            raise ValueError(
                "load_mirror_field(interior_fill_gauss=...) is the "
                "uniform on-axis Bz used below the map's axial support and "
                "must be finite and positive (got "
                f"{interior_fill_gauss!r}); pass None to use the map's own "
                "recorded bulk_field_gauss"
            )

    z_cm = np.asarray(geometry.z_cm, dtype=float)
    z_edges_cm = np.asarray(geometry.z_edges_cm, dtype=float)
    lengths_cm = np.asarray(geometry.length_cm, dtype=float)
    z_lo, z_hi = float(z_native_cm[0]), float(z_native_cm[-1])
    if float(z_edges_cm[-1]) > z_hi:
        raise ValueError(
            f"the mesh extends to z = {float(z_edges_cm[-1])} cm but the "
            f"field map {path} is solved only to z = {z_hi} cm. There is "
            "nothing to extend the field from past the map's high edge; "
            "re-solve the census over the mesh's span"
        )

    # Point sample and cell average, both on the same piecewise-linear
    # interpolant of the native grid; below the map's low edge both take the
    # declared uniform interior fill.
    b_face = np.where(
        z_edges_cm < z_lo, fill, np.interp(z_edges_cm, z_native_cm, b_native_gauss)
    )
    b_cell = np.where(
        z_cm < z_lo, fill, np.interp(z_cm, z_native_cm, b_native_gauss)
    )
    interior_fill_cell = z_cm < z_lo
    b_cell_avg = _cell_average(
        z_native_cm, b_native_gauss, np.clip(z_edges_cm, z_lo, z_hi), lengths_cm
    )
    # A cell straddling the low edge, or lying wholly below it, gets the fill
    # weighted over the part of it the map does not cover.
    below = np.minimum(np.maximum(z_lo - z_edges_cm[:-1], 0.0), lengths_cm)
    b_cell_avg = b_cell_avg + fill * below / lengths_cm

    # Native-resolution gradient (ripple included), then the mesh-resolved
    # one. Both in gauss/cm; the fill region has zero gradient by construction
    # of the fill, which is exactly the discard the module docstring names.
    dbdz_native = np.gradient(b_native_gauss, z_native_cm)
    dbdz_native_cell = np.where(
        z_cm < z_lo, 0.0, np.interp(z_cm, z_native_cm, dbdz_native)
    )
    dbdz_mesh_cell = (b_face[1:] - b_face[:-1]) / lengths_cm

    b_min = float(np.min(b_cell))
    b_max = float(np.max(b_cell))

    # Flux surface: traced values only, NaN outside the trace's support, and
    # a separate mask for the vacuum continuation past first wall contact.
    flux_radius = np.full(z_cm.shape, np.nan, dtype=float)
    flux_valid = (z_cm >= float(z_flux_cm[0])) & (z_cm <= float(z_flux_cm[-1]))
    flux_radius[flux_valid] = np.interp(z_cm[flux_valid], z_flux_cm, r_flux_cm)
    finite_crossings = crossing_z_cm[np.isfinite(crossing_z_cm)]
    first_contact = (
        float(np.min(finite_crossings)) if finite_crossings.size else float("nan")
    )
    vacuum_continuation = flux_valid & (
        z_cm >= first_contact if np.isfinite(first_contact) else False
    )

    return MirrorFieldProfile(
        case=case,
        map_path=str(path.resolve()),
        z_cm=z_cm.copy(),
        z_edges_cm=z_edges_cm.copy(),
        B_cell_gauss=b_cell,
        B_face_gauss=b_face,
        B_cell_average_gauss=b_cell_avg,
        dBdz_native_cell_gauss_per_cm=dbdz_native_cell,
        dBdz_mesh_cell_gauss_per_cm=dbdz_mesh_cell,
        mirror_ratio_cell=b_cell / b_min,
        mirror_ratio_bulk_cell=b_cell / b_bulk,
        B_min_gauss=b_min,
        B_max_gauss=b_max,
        B_bulk_gauss=b_bulk,
        interior_fill_gauss=fill,
        interior_fill_cell=interior_fill_cell,
        interior_ripple_relative=ripple,
        flux_radius_cell_cm=flux_radius,
        flux_radius_valid_cell=flux_valid,
        flux_radius_vacuum_continuation_cell=np.asarray(
            vacuum_continuation, dtype=bool
        ),
        first_wall_contact_z_cm=first_contact,
        wall_crossing_z_cm=crossing_z_cm,
        wall_crossing_radii_cm=crossing_r_cm,
        trace_end_z_cm=trace_end_cm,
        map_z_min_cm=z_lo,
        map_z_max_cm=z_hi,
        plasma_radius_m=map_radius_m,
    )
