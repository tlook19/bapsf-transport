from dataclasses import dataclass

import numpy as np


# Roles that carry no plasma: the machine behind the cathode. The plasma domain
# is bounded inside the neutral domain by a reflecting face wherever a
# plasma-dead cell abuts a live one.
PLASMA_DEAD_ROLES = frozenset({"plenum", "obstruction"})


@dataclass(frozen=True)
class Sim1DGeometry:
    """Axial layout for conservative 1D state arrays.

    The resolved typed-segment machine contains plenum / obstruction / cathode /
    anode / puff / column / collector regions. ``nx`` is the number of column
    cells and ``cells`` is the full typed-segment cell count.

    Neutral face quantities are *restricting* apertures (the minimum of the two
    adjacent cells), not arithmetic means: a conductance between a wide and a
    narrow duct is set by the narrow one.
    """

    nx: int
    length_cm: np.ndarray
    z_edges_cm: np.ndarray
    z_cm: np.ndarray
    cell_role: np.ndarray
    Rp_cm: np.ndarray
    Rm_cm: np.ndarray
    neutral_hydraulic_radius_cm: np.ndarray
    plasma_area_cm2: np.ndarray
    neutral_area_cm2: np.ndarray
    plasma_volume_cm3: np.ndarray
    neutral_volume_cm3: np.ndarray
    volume_ratio: np.ndarray
    plasma_face_area_cm2: np.ndarray
    neutral_face_area_cm2: np.ndarray
    neutral_face_hydraulic_radius_cm: np.ndarray
    # Authoritative plasma topology derived once from typed cell roles.
    # ``plasma_active`` is true only where plasma state/operators are live.
    # ``plasma_face_live_cell`` gives the one active cell adjacent to a
    # closed face, or -1 when that face bounds no active plasma.
    # ``cathode_cell_indices`` is the same kind of derived topology: the plasma
    # cell against each cathode surface, resolved once at construction (see
    # ``cathode_adjacent_cells``).
    plasma_active: np.ndarray
    plasma_face_live_cell: np.ndarray
    cathode_cell_indices: tuple
    plasma_open: np.ndarray
    plasma_absorbing: np.ndarray
    plasma_transmission: np.ndarray
    heat_transmission: np.ndarray
    neutral_face_conductance_cm3_s: np.ndarray
    center_distance_cm: np.ndarray
    # Positions of the cathode and anode *surfaces*: face
    # indices. The cathode faces are plasma walls at z = 0; the anode faces are
    # interior and plasma-open, throttled for heat and neutrals in M3.
    cathode_face_indices: np.ndarray
    anode_face_indices: np.ndarray
    # Optional thin annular baffles. Each clear radius is paired with the face
    # at the same array index; the aperture leaves the plasma channel open and
    # restricts only the surrounding neutral annulus.
    neutral_baffle_face_indices: np.ndarray
    neutral_baffle_clear_radius_cm: np.ndarray

    @property
    def cells(self):
        return self.length_cm.size

    @property
    def dz_cm(self):
        # Reporting convenience only; meaningful for the uniform legacy grid.
        return self.length_cm[1] if self.nx else 0.0


def build_geometry(input_dict, flags=None):
    """Build the resolved typed-segment machine geometry.

    ``resolved_boundaries=False`` and the legacy lumped geometry were removed.
    Keep a construction-time error for stale configurations so they cannot silently
    change geometry; historical results remain reproducible at
    ``legacy-final-2026-07-22``.
    """
    removed_keys = {
        "Lz",
        "source_length_cm",
        "end_length_cm",
        "source_Rm",
        "end_Rm",
        "source_Rp",
        "end_Rp",
    }
    stale = sorted(removed_keys.intersection(input_dict))
    if stale:
        raise ValueError(
            "these legacy lumped-geometry keys have been removed: "
            + ", ".join(stale)
            + "; reproduce that configuration at tag legacy-final-2026-07-22"
        )
    flags = flags or {}
    if not bool(flags.get("resolved_boundaries", True)):
        raise ValueError(
            "resolved_boundaries=False has been removed; "
            "use resolved typed-segment geometry or reproduce the historical "
            "configuration at tag legacy-final-2026-07-22"
        )
    return _build_resolved_geometry(input_dict, flags)


def puff_cell_indices(geometry):
    """Return ``(primary, twin)`` cell indices carrying the gas puff.

    Resolved geometry tags column cells with the ``puff`` role.
    """
    puff = np.flatnonzero(np.asarray(geometry.cell_role) == "puff")
    if not puff.size:
        raise ValueError("resolved geometry has no puff cells")
    return int(puff[0]), int(puff[-1])


def pump_cell_indices(geometry):
    """Return ``(left, right)`` cell indices carrying the pump sinks.

    The pump belongs on the plenum behind a cathode (§4); the non-cathode end
    keeps its own pump on the collector. Resolving by role keeps this correct if
    the layout changes.
    """
    roles = np.asarray(geometry.cell_role)
    left = np.flatnonzero(roles == "plenum")
    right = np.flatnonzero((roles == "plenum") | (roles == "collector"))
    left_index = int(left[0]) if left.size else 0
    right_index = int(right[-1]) if right.size else geometry.cells - 1
    return left_index, right_index


def _derive_cathode_adjacent_cells(cell_role, cathode_face_indices):
    """Resolve the plasma cell against each cathode surface, from scratch.

    The cathode surface is a face, so its surface terms (ion neutralization,
    sheath electron loss, ohmic deposition -- §8) land on the plasma-side cell
    next to it. Plasma is on the high-z side at the source cathode and the low-z
    side at a twin cathode, so pick whichever neighbour is not plasma-dead.

    This is the authoritative definition. It runs ONCE, at geometry
    construction; ``cathode_adjacent_cells`` reads the stored answer. Kept
    public-ish (module-private but importable) so tests can recompute a fresh
    reference and check the stored value against it.
    """
    roles = np.asarray(cell_role)
    dead = np.asarray([role in PLASMA_DEAD_ROLES for role in roles], dtype=bool)
    cells = []
    for face in np.asarray(cathode_face_indices, dtype=int):
        left, right = face - 1, face
        if 0 <= right < roles.size and not dead[right]:
            cells.append(int(right))
        elif 0 <= left < roles.size and not dead[left]:
            cells.append(int(left))
    return tuple(cells)


def cathode_adjacent_cells(geometry):
    """Return the plasma cell against each cathode surface.

    A pure function of run-constant topology (``cell_role`` and
    ``cathode_face_indices``), so it is resolved once by
    ``_derive_cathode_adjacent_cells`` at construction and stored on the
    geometry as ``cathode_cell_indices`` -- the same treatment the sibling
    derived-topology fields ``plasma_active`` and ``plasma_face_live_cell``
    already get. It used to rebuild two arrays and run a Python comprehension
    over ``cell_role`` on every call, and the cathode/beam/circuit paths call
    it ~24x per accepted step (2.28M times in a production run).

    Deliberately NOT an id()- or content-keyed lookaside cache: ``Sim1DGeometry``
    is a frozen dataclass built at exactly one site, so storing the derivation
    as a field makes staleness structurally impossible rather than merely
    unlikely. The value cannot disagree with the geometry it came from because
    there is no second copy to disagree with.
    """
    return geometry.cathode_cell_indices


def anode_flanking_cells(geometry):
    """Return ``(gap_side, column_side)`` cell pairs flanking each anode face.

    The anode neutralizes collected ions on *both* mesh faces, so the resulting
    neutrals are split across the two cells it separates (§7); the mesh throttles
    flow between them, which is why the side matters.
    """
    pairs = []
    for face in np.asarray(geometry.anode_face_indices, dtype=int):
        pairs.append((int(face - 1), int(face)))
    return tuple(pairs)


def gap_cell_indices(geometry, end=0):
    """Return the cathode-anode gap cells at the given machine end.

    Ohmic dissipation is I^2 R_p with R_p the plasma resistance *between* the
    cathode and the anode, so the power is deposited along the gap rather than
    piled into one boundary cell (§8).

    ``end`` selects the machine end: ``0`` for the source cathode, ``-1`` for the
    twin.
    """
    cathode_faces = np.asarray(geometry.cathode_face_indices, dtype=int)
    anode_faces = np.asarray(geometry.anode_face_indices, dtype=int)
    if cathode_faces.size == 0 or anode_faces.size == 0:
        raise ValueError("resolved geometry must define cathode and anode faces")
    which = 0 if end == 0 else -1
    cathode_face = int(cathode_faces[which])
    anode_face = int(anode_faces[which])
    low, high = sorted((cathode_face, anode_face))
    return tuple(range(low, high))


def is_plenum_cell(geometry, index):
    """Return True when ``index`` is a plenum (pump-behind-cathode) cell."""
    return str(np.asarray(geometry.cell_role)[index]) == "plenum"


def _anode_neutral_transparency(input_dict):
    """Return the anode face's open-area fraction for neutrals.

    The anode is a mesh disc of radius ``anode_radius_cm`` in a chamber of
    radius ``Rm``. Neutrals pass through the annulus around the disc freely
    and through the disc itself with the mesh transparency ``1 - eta``, so
    the open fraction is ``1 - eta * (Ra/Rm)^2`` -- exactly ``1 - eta`` when
    the disc spans the chamber (``anode_radius_cm = None``, the historical
    default). Heat transmission and Bohm collection keep the bare ``1 - eta``
    / ``eta``: the disc must still cover the plasma channel (``Ra >= Rp``),
    so the plasma-side physics does not see the annulus.
    """
    Ra = input_dict.get("anode_radius_cm")
    eta = float(input_dict.get("eta", 0.0))
    if Ra is None:
        return 1.0 - eta
    Ra = float(Ra)
    Rm = float(input_dict.get("Rm", 50.0))
    Rp = float(input_dict.get("Rp", 18.0))
    if not Rp <= Ra <= Rm:
        raise ValueError(
            f"anode_radius_cm must satisfy Rp <= Ra <= Rm "
            f"(got Ra={Ra}, Rp={Rp}, Rm={Rm}); an anode smaller than the "
            "plasma channel would invalidate the collection/heat treatment"
        )
    return 1.0 - eta * (Ra / Rm) ** 2


def _build_resolved_geometry(input_dict, flags):
    """Build the resolved typed-segment machine.

    The **cathode surface is the origin**: it sits at ``z = 0`` and the anode at
    ``z = cathode_anode_gap_cm``. Both are *faces*, not cells -- surfaces have
    a position but no length. The plenum and any obstruction
    extend to negative z behind the cathode, so ``Lm`` spans the cathode surface
    to the far machine end and the total mesh is longer than ``Lm``.

    Single-cathode (default), reading left to right::

        [plenum, (obstruction)] |cathode  [cathode..gap x nx_gap]  anode|
        [puff, column x (nx-1)] [collector]

    Twin cathode (``TwinCathode``) mirrors the source end instead of the
    collector, putting its cathode surface at ``z = Lm``. Column cells
    adjacent to an anode face carry the ``puff`` role (gas enters in front of
    the anode); the plasma cell against a cathode surface carries the
    ``cathode`` role so cathode surface terms have somewhere to land.

    The annular cathode-structure obstruction is a *real cell* of length
    ``Lcs``, so it holds gas and its inventory reaches the pump. It is omitted
    entirely when ``Lcs <= 0``, which is the legacy limit.

    The default-off ``source_fixed_grid`` flag replaces the uniform column with a
    fixed-cell source region plus an ``nx``-refined far column, so a refinement
    study no longer moves the near-source cell edges (see
    ``_source_fixed_grid_spec``). With it on, ``nx`` counts the far-column cells
    only and the ``puff`` role follows ``gas_puff_z_cm``.
    """
    nx = int(input_dict.get("nx", 60))
    nx_gap = int(input_dict.get("nx_gap", 5))
    if nx <= 0:
        raise ValueError(f"nx must be positive (got {nx})")
    if nx_gap <= 0:
        raise ValueError(f"nx_gap must be positive (got {nx_gap})")

    total_length = float(input_dict.get("Lm", 2000.0))
    twin = bool(flags.get("TwinCathode", False))
    end_expansion = _end_expansion_spec(input_dict, flags, twin=twin)

    plenum_length = float(input_dict.get("plenum_length_cm", 100.0))
    gap_length = float(input_dict.get("cathode_anode_gap_cm", 50.0))
    collector_length = float(input_dict.get("collector_length_cm", 100.0))
    for name, value in (
        ("plenum_length_cm", plenum_length),
        ("cathode_anode_gap_cm", gap_length),
    ):
        if value <= 0.0:
            raise ValueError(f"{name} must be positive (got {value})")

    Rp = float(input_dict.get("Rp", 18.0))
    Rm = float(input_dict.get("Rm", 50.0))
    Rcs = float(input_dict.get("Rcs", 0.0))
    Lcs = float(input_dict.get("Lcs", 0.0))
    Rsup = float(input_dict.get("Rsup", 0.0))
    if not 0.0 <= Rcs < Rm:
        raise ValueError(f"Rcs must satisfy 0 <= Rcs < Rm (got {Rcs} vs Rm={Rm})")
    if not 0.0 <= Rsup < Rm:
        raise ValueError(f"Rsup must satisfy 0 <= Rsup < Rm (got {Rsup} vs Rm={Rm})")

    has_obstruction = Lcs > 0.0

    # Behind the cathode surface (negative z): plenum, then the optional duct.
    behind_roles = ["plenum"] + (["obstruction"] if has_obstruction else [])
    behind_lengths = [plenum_length] + ([Lcs] if has_obstruction else [])

    # Cathode surface -> anode: the plasma cell against the cathode carries the
    # cathode role; the rest of the gap is plain gap.
    gap_roles = ["cathode"] + ["gap"] * (nx_gap - 1)
    gap_lengths = [gap_length / nx_gap] * nx_gap

    if twin:
        column_length = total_length - 2.0 * gap_length
    else:
        if collector_length <= 0.0:
            raise ValueError(
                f"collector_length_cm must be positive (got {collector_length})"
            )
        column_length = total_length - gap_length - collector_length
    if column_length <= 0.0:
        raise ValueError(
            "resolved boundary regions exceed Lm; no room for the column "
            f"(column_length={column_length} cm, Lm={total_length} cm)"
        )

    source_grid = _source_fixed_grid_spec(
        input_dict,
        flags,
        gap_length=gap_length,
        total_length=total_length,
        collector_length=collector_length,
        twin=twin,
    )
    if source_grid is None:
        column_roles = ["puff"] + ["column"] * (nx - 1)
        if twin:
            column_roles[-1] = "puff"
        column_lengths = [column_length / nx] * nx
    else:
        # Fixed-cell source region: the first ``n_fixed`` column cells have a
        # prescribed length independent of ``nx``, which then meshes only the
        # remaining far column. The puff role follows gas_puff_z_cm rather than
        # the first column cell, so the fueling centre stops moving with nx.
        n_fixed = source_grid["cells"]
        outer_length = column_length - source_grid["span_cm"]
        if outer_length <= 0.0:
            raise ValueError(
                "source_fixed_grid leaves no far column between the source "
                f"region and the collector (outer_length={outer_length} cm)"
            )
        column_roles = ["column"] * (n_fixed + nx)
        column_roles[source_grid["puff_offset"]] = "puff"
        column_lengths = [source_grid["dz_cm"]] * n_fixed + [
            outer_length / nx
        ] * nx

    roles = behind_roles + gap_roles + column_roles
    lengths = behind_lengths + gap_lengths + column_lengths
    if twin:
        roles += list(reversed(gap_roles)) + list(reversed(behind_roles))
        lengths += list(reversed(gap_lengths)) + list(reversed(behind_lengths))
    else:
        end_cells = 1 if end_expansion is None else end_expansion["cells"]
        roles += ["end"] * (end_cells - 1) + ["collector"]
        lengths += [collector_length / end_cells] * end_cells

    length_cm = np.asarray(lengths, dtype=float)
    cell_role = np.asarray(roles, dtype=object)
    cells = length_cm.size

    # Cathode and anode faces. Face f separates cell f-1 from cell f, so the
    # cathode surface is the face just past the cells behind it.
    cathode_face = len(behind_roles)
    anode_face = cathode_face + nx_gap
    cathode_faces = [cathode_face]
    anode_faces = [anode_face]
    if twin:
        far_cathode_face = cells - len(behind_roles)
        cathode_faces.append(far_cathode_face)
        anode_faces.append(far_cathode_face - nx_gap)

    # Shift the origin so the cathode surface lands exactly on z = 0.
    z_edges_cm = np.concatenate(([0.0], np.cumsum(length_cm)))
    z_edges_cm = z_edges_cm - z_edges_cm[cathode_face]
    z_cm = 0.5 * (z_edges_cm[:-1] + z_edges_cm[1:])

    far_end_z = (
        z_edges_cm[cathode_faces[-1]] if twin else z_edges_cm[-1]
    )
    if not np.isclose(far_end_z, total_length):
        raise ValueError(
            "resolved geometry must span Lm from the cathode surface "
            f"(got {far_end_z} cm vs {total_length} cm)"
        )

    Rp_cm = np.full(cells, Rp, dtype=float)
    Rm_cm = np.full(cells, Rm, dtype=float)
    # Plenum cells stay at full plasma radius: they are made plasma-dead by the
    # reflecting cathode face, not by shrinking Rp (§5 warns a near-zero plasma
    # volume blows up the flux divergence).
    plasma_area_cm2 = np.pi * Rp_cm**2
    neutral_area_cm2 = np.pi * Rm_cm**2
    neutral_hydraulic_radius_cm = Rm_cm.copy()
    plasma_face_area_override = None

    if end_expansion is not None:
        end = np.flatnonzero(
            np.isin(cell_role, np.asarray(["end", "collector"], dtype=object))
        )
        n_end = int(end_expansion["cells"])
        if end.size != n_end or not np.array_equal(
            end, np.arange(cells - n_end, cells)
        ):
            raise ValueError("expanded end cells must be one contiguous terminal block")

        # The vessel makes an abrupt step to its larger radius at the end-cell
        # entrance. Neutral faces remain restricting apertures, so that entrance
        # still sees the upstream bore while the downstream volume is enlarged.
        Rm_end = float(end_expansion["machine_radius_cm"])
        Rm_cm[end] = Rm_end
        neutral_area_cm2[end] = np.pi * Rm_end**2
        neutral_hydraulic_radius_cm[end] = Rm_end

        # Resolve a smooth *area* flare. Area is the flux-tube variable
        # (A*B=const); the half-cosine has zero slope at both ends and is an
        # explicit provisional closure until measured B(z) is supplied.
        A0 = np.pi * Rp**2
        A1 = np.pi * float(end_expansion["plasma_radius_cm"]) ** 2
        xi_face = np.linspace(0.0, 1.0, n_end + 1)
        smooth = 0.5 - 0.5 * np.cos(np.pi * xi_face)
        end_face_area = A0 + (A1 - A0) * smooth
        end_cell_area = 0.5 * (end_face_area[:-1] + end_face_area[1:])
        plasma_area_cm2[end] = end_cell_area
        Rp_cm[end] = np.sqrt(end_cell_area / np.pi)

        plasma_face_area_override = _face_area(plasma_area_cm2)
        start_face = int(end[0])
        plasma_face_area_override[
            start_face : start_face + n_end + 1
        ] = end_face_area

    # Obstruction cells are an annular duct: reduced open area AND a reduced
    # hydraulic radius (Rm - Rcs), the two differing independently (§3 keystone).
    obstruction = cell_role == "obstruction"
    if np.any(obstruction):
        neutral_area_cm2[obstruction] = np.pi * (Rm**2 - Rcs**2)
        neutral_hydraulic_radius_cm[obstruction] = Rm - Rcs

    # Support rods block plenum volume only: distributed thin structure, not a
    # duct, so the hydraulic radius is untouched (§3).
    if Rsup > 0.0:
        plenum = cell_role == "plenum"
        neutral_area_cm2[plenum] = np.pi * (Rm**2 - Rsup**2)

    baffle_faces, baffle_radii = _neutral_baffle_spec(
        input_dict=input_dict,
        flags=flags,
        z_edges_cm=z_edges_cm,
        Rp_cm=Rp_cm,
        Rm_cm=Rm_cm,
        cathode_face_indices=cathode_faces,
        anode_face_indices=anode_faces,
    )

    plasma_volume_cm3 = plasma_area_cm2 * length_cm
    neutral_volume_cm3 = neutral_area_cm2 * length_cm
    volume_ratio = plasma_volume_cm3 / neutral_volume_cm3
    center_distance_cm = np.diff(z_cm)

    return _assemble_geometry(
        nx=nx,
        length_cm=length_cm,
        z_edges_cm=z_edges_cm,
        z_cm=z_cm,
        cell_role=cell_role,
        Rp_cm=Rp_cm,
        Rm_cm=Rm_cm,
        neutral_hydraulic_radius_cm=neutral_hydraulic_radius_cm,
        plasma_area_cm2=plasma_area_cm2,
        neutral_area_cm2=neutral_area_cm2,
        plasma_volume_cm3=plasma_volume_cm3,
        neutral_volume_cm3=neutral_volume_cm3,
        volume_ratio=volume_ratio,
        center_distance_cm=center_distance_cm,
        cathode_face_indices=np.asarray(cathode_faces, dtype=int),
        anode_face_indices=np.asarray(anode_faces, dtype=int),
        neutral_baffle_face_indices=baffle_faces,
        neutral_baffle_clear_radius_cm=baffle_radii,
        anode_transparency=1.0 - float(input_dict.get("eta", 0.0)),
        anode_neutral_transparency=_anode_neutral_transparency(input_dict),
        anode_advective_block=float(
            input_dict.get("b_anode_advective_block", 0.0)
        ),
        # Plasma-terminating surfaces: every cathode, plus the collector's outer
        # face when there is one (a twin machine ends in plenums instead, whose
        # back walls are closed and see no plasma).
        absorbing_face_indices=(
            list(cathode_faces) if twin else list(cathode_faces) + [cells]
        ),
        plasma_face_area_override=plasma_face_area_override,
    )


def _source_fixed_grid_spec(
    input_dict, flags, *, gap_length, total_length, collector_length, twin
):
    """Validate and return the optional fixed-cell source-region specification.

    Resolution studies on the default mesh are self-confounding: ``nx`` uniform
    column cells span anode face to collector start, so refining ``nx`` moves
    every cell edge -- including the puff cell, whose centre anchors the default
    cosine puff profile. This default-off mode pins the column between the anode
    face and ``source_region_length_cm`` to cells of exactly
    ``source_region_dz_cm``, leaving ``nx`` to refine only the far column.

    Presence-gated in both directions (the ``input_dict`` / ``input_flags``
    silent-namespace trap): the two parameters are required when the flag is on
    and forbidden when it is off. Returns ``None`` when off, which is the only
    path the production geometry takes.
    """
    keys = ("source_region_length_cm", "source_region_dz_cm")
    raw = {key: input_dict.get(key) for key in keys}
    provided = {key: value is not None for key, value in raw.items()}
    enabled = bool(flags.get("source_fixed_grid", False))
    if not enabled:
        stale = [key for key, present in provided.items() if present]
        if stale:
            raise ValueError(
                "source region parameters require the default-off "
                "source_fixed_grid flag: " + ", ".join(stale)
            )
        return None
    missing = [key for key, present in provided.items() if not present]
    if missing:
        raise ValueError(
            "source_fixed_grid requires all source region parameters; missing "
            + ", ".join(missing)
        )
    if twin:
        raise ValueError(
            "source_fixed_grid is defined only for the single-cathode layout; "
            "mirroring the fixed source region onto a TwinCathode end is not "
            "implemented"
        )

    region_length = float(raw["source_region_length_cm"])
    dz = float(raw["source_region_dz_cm"])
    if not np.isfinite(region_length) or not np.isfinite(dz) or dz <= 0.0:
        raise ValueError(
            "source_region_length_cm and source_region_dz_cm must be finite "
            f"with a positive cell size (got {region_length}, {dz})"
        )
    if region_length <= gap_length:
        raise ValueError(
            "source_region_length_cm must lie strictly beyond the anode face "
            f"(got {region_length} cm vs cathode_anode_gap_cm={gap_length} cm)"
        )
    column_end = total_length - collector_length
    if region_length >= column_end:
        raise ValueError(
            "source_region_length_cm must lie strictly before the collector "
            f"block (got {region_length} cm vs Lm - collector_length_cm = "
            f"{column_end} cm)"
        )

    span = region_length - gap_length
    cells_float = span / dz
    cells = int(round(cells_float))
    if cells < 1 or abs(cells_float - cells) > 1e-9 * max(cells_float, 1.0):
        raise ValueError(
            "the source region must be an integer number of "
            f"source_region_dz_cm cells (got {span} cm / {dz} cm = "
            f"{cells_float})"
        )

    puff_z = input_dict.get("gas_puff_z_cm")
    if puff_z is None:
        raise ValueError(
            "source_fixed_grid requires an explicit gas_puff_z_cm: the puff "
            "role follows the fueling position instead of the first column "
            "cell, so it cannot be left to the mesh"
        )
    puff_z = float(puff_z)
    if not gap_length <= puff_z < region_length:
        raise ValueError(
            "gas_puff_z_cm must lie in [cathode_anode_gap_cm, "
            f"source_region_length_cm) (got {puff_z} cm, region "
            f"[{gap_length}, {region_length}) cm)"
        )
    offset = int(np.floor((puff_z - gap_length) / dz))
    if not 0 <= offset < cells:
        raise ValueError(
            f"gas_puff_z_cm={puff_z} cm maps outside the fixed source region "
            f"cells (offset {offset} of {cells})"
        )
    return {
        "cells": cells,
        "dz_cm": dz,
        "span_cm": span,
        "region_length_cm": region_length,
        "puff_offset": offset,
    }


def _end_expansion_spec(input_dict, flags, *, twin):
    """Validate and return the optional expanded-end geometry specification."""
    keys = (
        "end_expansion_cells",
        "end_expansion_machine_radius_cm",
        "end_expansion_plasma_radius_cm",
    )
    raw = {key: input_dict.get(key) for key in keys}
    provided = {key: value is not None for key, value in raw.items()}
    enabled = bool(flags.get("end_expansion_geometry", False))
    if not enabled:
        stale = [key for key, present in provided.items() if present]
        if stale:
            raise ValueError(
                "end expansion parameters require the default-off "
                "end_expansion_geometry flag: " + ", ".join(stale)
            )
        return None
    missing = [key for key, present in provided.items() if not present]
    if missing:
        raise ValueError(
            "end_expansion_geometry requires all end expansion parameters; "
            "missing " + ", ".join(missing)
        )
    if twin:
        raise ValueError(
            "end_expansion_geometry is defined only for the single-cathode "
            "collector end"
        )

    cells_float = float(raw["end_expansion_cells"])
    cells = int(cells_float)
    if cells_float != cells or cells < 2:
        raise ValueError(
            "end_expansion_cells must be an integer >= 2 "
            f"(got {raw['end_expansion_cells']!r})"
        )
    Rm = float(input_dict.get("Rm", 50.0))
    Rp = float(input_dict.get("Rp", 18.0))
    Rm_end = float(raw["end_expansion_machine_radius_cm"])
    Rp_end = float(raw["end_expansion_plasma_radius_cm"])
    if not np.isfinite(Rm_end) or Rm_end < Rm:
        raise ValueError(
            "end_expansion_machine_radius_cm must be finite and >= Rm "
            f"(got {Rm_end} vs Rm={Rm})"
        )
    if not np.isfinite(Rp_end) or not Rp <= Rp_end <= Rm_end:
        raise ValueError(
            "end_expansion_plasma_radius_cm must satisfy "
            f"Rp <= Rp_end <= Rm_end (got {Rp_end}, Rp={Rp}, Rm_end={Rm_end})"
        )
    return {
        "cells": cells,
        "machine_radius_cm": Rm_end,
        "plasma_radius_cm": Rp_end,
    }


def _neutral_baffle_spec(
    *,
    input_dict,
    flags,
    z_edges_cm,
    Rp_cm,
    Rm_cm,
    cathode_face_indices,
    anode_face_indices,
):
    """Validate and map optional thin annular baffles onto mesh faces."""
    keys = ("neutral_baffle_positions_cm", "neutral_baffle_clear_radii_cm")
    raw = {key: input_dict.get(key) for key in keys}
    provided = {key: value is not None for key, value in raw.items()}
    enabled = bool(flags.get("neutral_baffles", False))
    if not enabled:
        stale = [key for key, present in provided.items() if present]
        if stale:
            raise ValueError(
                "neutral baffle parameters require the default-off "
                "neutral_baffles flag: " + ", ".join(stale)
            )
        return np.empty(0, dtype=int), np.empty(0, dtype=float)
    missing = [key for key, present in provided.items() if not present]
    if missing:
        raise ValueError(
            "neutral_baffles requires positions and clear radii; missing "
            + ", ".join(missing)
        )

    def as_vector(name, value):
        array = np.asarray(value, dtype=float)
        if array.ndim == 0:
            array = array.reshape(1)
        if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be a finite non-empty scalar or sequence")
        return array

    positions = as_vector(keys[0], raw[keys[0]])
    radii = as_vector(keys[1], raw[keys[1]])
    if positions.shape != radii.shape:
        raise ValueError(
            "neutral baffle positions and clear radii must have equal lengths"
        )

    z_edges = np.asarray(z_edges_cm, dtype=float)
    Rp = np.asarray(Rp_cm, dtype=float)
    Rm = np.asarray(Rm_cm, dtype=float)
    faces = np.asarray(
        [int(np.argmin(np.abs(z_edges - position))) for position in positions],
        dtype=int,
    )
    if np.any((faces <= 0) | (faces >= Rp.size)):
        raise ValueError("neutral baffles must map to interior mesh faces")
    if np.unique(faces).size != faces.size:
        raise ValueError("neutral baffle positions must map to distinct mesh faces")
    forbidden = set(np.asarray(cathode_face_indices, dtype=int))
    forbidden.update(np.asarray(anode_face_indices, dtype=int))
    if any(int(face) in forbidden for face in faces):
        raise ValueError("neutral baffles cannot coincide with cathode or anode faces")

    for requested, face, clear in zip(positions, faces, radii):
        Rp_face = max(float(Rp[face - 1]), float(Rp[face]))
        Rm_face = min(float(Rm[face - 1]), float(Rm[face]))
        if not Rp_face <= float(clear) < Rm_face:
            raise ValueError(
                "neutral baffle clear radius must satisfy local "
                f"Rp <= R_clear < Rm (got {clear}, Rp={Rp_face}, Rm={Rm_face})"
            )
        half_spacing = 0.5 * min(
            float(z_edges[face] - z_edges[face - 1]),
            float(z_edges[face + 1] - z_edges[face]),
        )
        if abs(float(z_edges[face]) - float(requested)) > half_spacing + 1e-12:
            raise ValueError(
                "neutral baffle position is too far from its nearest mesh face"
            )
    order = np.argsort(faces)
    return faces[order], radii[order]


def _assemble_geometry(
    *,
    nx,
    length_cm,
    z_edges_cm,
    z_cm,
    cell_role,
    Rp_cm,
    Rm_cm,
    neutral_hydraulic_radius_cm,
    plasma_area_cm2,
    neutral_area_cm2,
    plasma_volume_cm3,
    neutral_volume_cm3,
    volume_ratio,
    center_distance_cm,
    cathode_face_indices=None,
    anode_face_indices=None,
    neutral_baffle_face_indices=None,
    neutral_baffle_clear_radius_cm=None,
    anode_transparency=1.0,
    anode_neutral_transparency=None,
    anode_advective_block=0.0,
    absorbing_face_indices=None,
    plasma_face_area_override=None,
):
    """Derive the face arrays from the cell arrays and pack a ``Sim1DGeometry``.

    Plasma walls are found from the roles: both external ends, plus any interior
    face where a plasma-dead cell (plenum/obstruction) abuts a live one. Legacy
    roles contain no plasma-dead cells, so only the external ends are walls --
    exactly today's behavior.
    """
    cells = length_cm.size
    cathode_faces = (
        np.empty(0, dtype=int)
        if cathode_face_indices is None
        else np.asarray(cathode_face_indices, dtype=int)
    )
    plasma_face_area_cm2 = _face_area(plasma_area_cm2)
    if plasma_face_area_override is not None:
        override = np.asarray(plasma_face_area_override, dtype=float)
        if override.shape != (cells + 1,) or not np.all(
            np.isfinite(override) & (override > 0.0)
        ):
            raise ValueError(
                "plasma_face_area_override must be finite and positive with "
                f"shape {(cells + 1,)}"
            )
        plasma_face_area_cm2 = override.copy()
    # Restricting apertures for the neutral conductance (see class docstring).
    neutral_face_area_cm2 = _face_min(neutral_area_cm2)
    neutral_face_hydraulic_radius_cm = _face_min(neutral_hydraulic_radius_cm)

    dead = np.asarray([role in PLASMA_DEAD_ROLES for role in cell_role], dtype=bool)
    plasma_active = ~dead
    # Absorbing faces are the plasma-terminating surfaces: the whole cross-section
    # ends there, so the plasma goes sonic into the sheath and is neutralized.
    # They are a *refinement* of walls -- nothing passes
    # through to the far side -- and legacy geometry has none, keeping its
    # historical volumetric surface terms.
    plasma_absorbing = np.zeros(cells + 1, dtype=bool)
    for face in np.asarray(
        [] if absorbing_face_indices is None else absorbing_face_indices, dtype=int
    ):
        plasma_absorbing[face] = True
    plasma_open = np.ones(cells + 1, dtype=bool)
    plasma_transmission = np.ones(cells + 1, dtype=float)
    heat_transmission = np.ones(cells + 1, dtype=float)
    walls = [0, cells]
    walls += [face for face in range(1, cells) if dead[face - 1] != dead[face]]
    for face in walls:
        plasma_open[face] = False
        plasma_transmission[face] = 0.0
        heat_transmission[face] = 0.0
    plasma_face_live_cell = np.full(cells + 1, -1, dtype=int)
    for face in np.flatnonzero(~plasma_open):
        adjacent = []
        if face > 0 and plasma_active[face - 1]:
            adjacent.append(face - 1)
        if face < cells and plasma_active[face]:
            adjacent.append(face)
        if len(adjacent) > 1:
            raise ValueError(
                f"closed plasma face {face} has active cells on both sides"
            )
        if adjacent:
            plasma_face_live_cell[face] = int(adjacent[0])

    # The anode mesh is plasma-open but partially blocking, and the three
    # transmissions are independent (§3). eta = 0 recovers a fully transparent
    # anode -- the legacy limit.
    #
    # Heat and neutrals are throttled by the transparency (1-eta). The *advective*
    # plasma flux is NOT: the anode removes plasma through the Bohm sheath flux at
    # its wires (physics/sources.anode_collection_rhs), and shrinking the face as
    # well would remove the same particles twice (§5). Mass that misses a wire
    # simply streams through the holes. `b_anode_advective_block` (default 0)
    # exists only to dial that blocking back in for a sensitivity study; note it
    # *reflects* rather than absorbs, since the absorption is always Bohm.
    transparency = float(anode_transparency)
    if not 0.0 <= transparency <= 1.0:
        raise ValueError(
            f"anode transparency must lie in [0, 1] (got {transparency})"
        )
    block = float(anode_advective_block)
    if not 0.0 <= block <= 1.0:
        raise ValueError(
            f"b_anode_advective_block must lie in [0, 1] (got {block})"
        )
    neutral_transparency = (
        transparency
        if anode_neutral_transparency is None
        else float(anode_neutral_transparency)
    )
    if not 0.0 <= neutral_transparency <= 1.0:
        raise ValueError(
            "anode neutral transparency must lie in [0, 1] "
            f"(got {neutral_transparency})"
        )
    for face in np.asarray(
        [] if anode_face_indices is None else anode_face_indices, dtype=int
    ):
        plasma_transmission[face] = 1.0 - block * (1.0 - transparency)
        heat_transmission[face] = transparency
        neutral_face_area_cm2[face] = (
            neutral_face_area_cm2[face] * neutral_transparency
        )

    baffle_faces = np.asarray(
        [] if neutral_baffle_face_indices is None else neutral_baffle_face_indices,
        dtype=int,
    )
    baffle_radii = np.asarray(
        []
        if neutral_baffle_clear_radius_cm is None
        else neutral_baffle_clear_radius_cm,
        dtype=float,
    )
    if baffle_faces.shape != baffle_radii.shape:
        raise ValueError("neutral baffle face and radius arrays must have equal shape")
    for face, clear in zip(baffle_faces, baffle_radii):
        if not 0 < int(face) < cells:
            raise ValueError("neutral baffle faces must be interior")
        neutral_face_area_cm2[int(face)] = np.pi * float(clear) ** 2

    # No prescribed neutral apertures: NaN => derive the molecular-flow (Clausing)
    # conductance from the face area + hydraulic radius. Kept as an escape hatch
    # for a face whose conductance is known directly rather than geometrically.
    neutral_face_conductance_cm3_s = np.full(cells + 1, np.nan, dtype=float)

    return Sim1DGeometry(
        nx=nx,
        length_cm=length_cm,
        z_edges_cm=z_edges_cm,
        z_cm=z_cm,
        cell_role=cell_role,
        Rp_cm=Rp_cm,
        Rm_cm=Rm_cm,
        neutral_hydraulic_radius_cm=neutral_hydraulic_radius_cm,
        plasma_area_cm2=plasma_area_cm2,
        neutral_area_cm2=neutral_area_cm2,
        plasma_volume_cm3=plasma_volume_cm3,
        neutral_volume_cm3=neutral_volume_cm3,
        volume_ratio=volume_ratio,
        plasma_face_area_cm2=plasma_face_area_cm2,
        neutral_face_area_cm2=neutral_face_area_cm2,
        neutral_face_hydraulic_radius_cm=neutral_face_hydraulic_radius_cm,
        plasma_active=plasma_active,
        plasma_face_live_cell=plasma_face_live_cell,
        # Derived once here, for the same reason plasma_active is: the cathode/
        # beam/circuit paths ask for it ~24x per accepted step.
        cathode_cell_indices=_derive_cathode_adjacent_cells(
            cell_role, cathode_faces
        ),
        plasma_open=plasma_open,
        plasma_absorbing=plasma_absorbing,
        plasma_transmission=plasma_transmission,
        heat_transmission=heat_transmission,
        neutral_face_conductance_cm3_s=neutral_face_conductance_cm3_s,
        center_distance_cm=center_distance_cm,
        cathode_face_indices=cathode_faces,
        anode_face_indices=(
            np.empty(0, dtype=int)
            if anode_face_indices is None
            else np.asarray(anode_face_indices, dtype=int)
        ),
        neutral_baffle_face_indices=baffle_faces,
        neutral_baffle_clear_radius_cm=baffle_radii,
    )


def _face_area(cell_area):
    """Return external plus internal face areas from adjacent cell averages."""
    face_area = np.empty(cell_area.size + 1, dtype=float)
    face_area[0] = cell_area[0]
    face_area[-1] = cell_area[-1]
    face_area[1:-1] = 0.5 * (cell_area[:-1] + cell_area[1:])
    return face_area


def _face_min(cell_values):
    """Return face values as the restricting (minimum) of adjacent cells.

    Externals take the end cell's value. For uniform adjacent cells this equals
    the arithmetic mean bit-for-bit (``0.5*(a+a) == a`` in IEEE-754), so legacy
    geometry is unaffected.
    """
    face_values = np.empty(cell_values.size + 1, dtype=float)
    face_values[0] = cell_values[0]
    face_values[-1] = cell_values[-1]
    face_values[1:-1] = np.minimum(cell_values[:-1], cell_values[1:])
    return face_values
