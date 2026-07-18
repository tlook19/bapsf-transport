from dataclasses import dataclass

import numpy as np


# Roles that carry no plasma: the machine behind the cathode (BOUNDARY_REGIONS_
# PLAN.md §5). The plasma domain is bounded inside the neutral domain by a
# reflecting face wherever a plasma-dead cell abuts a live one.
PLASMA_DEAD_ROLES = frozenset({"plenum", "obstruction"})


@dataclass(frozen=True)
class Sim1DGeometry:
    """Axial layout for conservative 1D state arrays.

    Two modes populate the *same* fields (BOUNDARY_REGIONS_PLAN.md §13):

    - **legacy** (default): the historical 0D-source, 1D-domain, 0D-end lump.
    - **resolved** (``resolved_boundaries`` flag): the typed-segment machine of
      §3 -- plenum / obstruction / cathode / anode / puff / column / collector.

    Operators read the fields, never the mode, so "off" is legacy inputs flowing
    through one code path rather than a parallel branch.

    ``nx`` is the number of column/domain cells; ``cells`` is the true cell count
    (``nx + 2`` in legacy, larger in resolved), so index-agnostic consumers keep
    working in both modes.

    Neutral face quantities are *restricting* apertures (the minimum of the two
    adjacent cells), not arithmetic means: a conductance between a wide and a
    narrow duct is set by the narrow one. This is bit-identical to the historical
    mean whenever adjacent cells share a radius, which is every legacy config that
    does not override ``source_Rm``/``end_Rm``.
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
    plasma_open: np.ndarray
    plasma_transmission: np.ndarray
    heat_transmission: np.ndarray
    neutral_face_conductance_cm3_s: np.ndarray
    center_distance_cm: np.ndarray
    # Positions of the cathode and anode *surfaces* (plan §11 decision 5): face
    # indices, empty in legacy geometry which has no resolved surfaces. The
    # cathode faces are plasma walls at z = 0; the anode faces are interior and
    # plasma-open, throttled for heat and neutrals in M3.
    cathode_face_indices: np.ndarray
    anode_face_indices: np.ndarray

    @property
    def cells(self):
        return self.length_cm.size

    @property
    def dz_cm(self):
        # Reporting convenience only; meaningful for the uniform legacy grid.
        return self.length_cm[1] if self.nx else 0.0


def build_geometry(input_dict, flags=None):
    """Build the axial geometry, dispatching on the ``resolved_boundaries`` flag.

    Off (default) => the legacy source/domain/end lump; on => the resolved
    typed-segment machine. Geometry construction is the *only* place the flag is
    read: everything downstream keys off the returned arrays (§13).
    """
    resolved = bool((flags or {}).get("resolved_boundaries", False))
    if resolved:
        return _build_resolved_geometry(input_dict, flags or {})
    return _build_legacy_geometry(input_dict)


def puff_cell_indices(geometry):
    """Return ``(primary, twin)`` cell indices carrying the gas puff.

    Resolved geometry tags column cells with the ``puff`` role; legacy anchors the
    puff at the source cell (and the end cell for a twin cathode), so this returns
    ``(0, cells-1)`` there -- one code path, legacy limit included (§13).
    """
    puff = np.flatnonzero(np.asarray(geometry.cell_role) == "puff")
    if puff.size:
        return int(puff[0]), int(puff[-1])
    return 0, geometry.cells - 1


def pump_cell_indices(geometry):
    """Return ``(left, right)`` cell indices carrying the pump sinks.

    The pump belongs on the plenum behind a cathode (§4); the non-cathode end
    keeps its own pump on the collector. In every layout built here -- legacy,
    resolved single-cathode, resolved twin -- those are the first and last cells,
    but resolving by role keeps this correct if the layout changes.
    """
    roles = np.asarray(geometry.cell_role)
    left = np.flatnonzero(roles == "plenum")
    right = np.flatnonzero((roles == "plenum") | (roles == "collector"))
    left_index = int(left[0]) if left.size else 0
    right_index = int(right[-1]) if right.size else geometry.cells - 1
    return left_index, right_index


def cathode_adjacent_cells(geometry):
    """Return the plasma cell against each cathode surface.

    The cathode surface is a face, so its surface terms (ion neutralization,
    sheath electron loss, ohmic deposition -- §8) land on the plasma-side cell
    next to it. Plasma is on the high-z side at the source cathode and the low-z
    side at a twin cathode, so pick whichever neighbour is not plasma-dead.
    """
    roles = np.asarray(geometry.cell_role)
    dead = np.asarray([role in PLASMA_DEAD_ROLES for role in roles], dtype=bool)
    cells = []
    for face in np.asarray(geometry.cathode_face_indices, dtype=int):
        left, right = face - 1, face
        if 0 <= right < roles.size and not dead[right]:
            cells.append(int(right))
        elif 0 <= left < roles.size and not dead[left]:
            cells.append(int(left))
    return tuple(cells)


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


def is_plenum_cell(geometry, index):
    """Return True when ``index`` is a plenum (pump-behind-cathode) cell."""
    return str(np.asarray(geometry.cell_role)[index]) == "plenum"


def _build_legacy_geometry(input_dict):
    """Build the default hybrid 0D-source, 1D-domain, 0D-end geometry.

    The numeric lines below are unchanged from the historical ``build_geometry``;
    only the schema arrays are appended, at legacy defaults. Keep them verbatim so
    the golden baseline stays bit-exact with the master switch off.
    """
    nx = int(input_dict.get("nx", 60))
    if nx <= 0:
        raise ValueError(f"nx must be positive (got {nx})")

    source_length = float(input_dict.get("source_length_cm", 100.0))
    end_length = float(input_dict.get("end_length_cm", 100.0))
    total_length = float(input_dict.get("Lm", 2000.0))
    resolved_length = float(
        input_dict.get("Lz", total_length - source_length - end_length)
    )

    if source_length <= 0 or end_length <= 0 or resolved_length <= 0:
        raise ValueError("source, resolved, and end lengths must all be positive")

    tiled_length = source_length + resolved_length + end_length
    if not np.isclose(tiled_length, total_length):
        raise ValueError(
            "source_length_cm + Lz + end_length_cm must equal Lm "
            f"(got {tiled_length} cm vs {total_length} cm)"
        )

    dz = resolved_length / nx
    length_cm = np.concatenate(([source_length], np.full(nx, dz), [end_length]))
    z_edges_cm = np.concatenate(([0.0], np.cumsum(length_cm)))
    z_cm = 0.5 * (z_edges_cm[:-1] + z_edges_cm[1:])

    cell_role = np.empty(nx + 2, dtype=object)
    cell_role[0] = "source"
    cell_role[1:-1] = "domain"
    cell_role[-1] = "end"

    Rp = float(input_dict.get("Rp", 18.0))
    Rm = float(input_dict.get("Rm", 50.0))
    Rp_cm = np.full(nx + 2, Rp, dtype=float)
    Rm_cm = np.full(nx + 2, Rm, dtype=float)
    Rp_cm[0] = float(input_dict.get("source_Rp") or Rp)
    Rp_cm[-1] = float(input_dict.get("end_Rp") or Rp)
    Rm_cm[0] = float(input_dict.get("source_Rm") or Rm)
    Rm_cm[-1] = float(input_dict.get("end_Rm") or Rm)

    plasma_area_cm2 = np.pi * Rp_cm**2
    neutral_area_cm2 = np.pi * Rm_cm**2
    plasma_volume_cm3 = plasma_area_cm2 * length_cm
    neutral_volume_cm3 = neutral_area_cm2 * length_cm
    volume_ratio = plasma_volume_cm3 / neutral_volume_cm3

    center_distance_cm = np.diff(z_cm)

    # Legacy schema defaults: neutral hydraulic radius = Rm (one-radius model);
    # plasma reflecting walls only at the two external ends.
    neutral_hydraulic_radius_cm = Rm_cm.copy()
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
    )


def _build_resolved_geometry(input_dict, flags):
    """Build the resolved typed-segment machine (BOUNDARY_REGIONS_PLAN.md §3).

    The **cathode surface is the origin**: it sits at ``z = 0`` and the anode at
    ``z = cathode_anode_gap_cm``. Both are *faces*, not cells (plan §11 decision
    5) -- surfaces have a position but no length. The plenum and any obstruction
    extend to negative z behind the cathode, so ``Lm`` spans the cathode surface
    to the far machine end and the total mesh is longer than ``Lm``.

    Single-cathode (default), reading left to right::

        [plenum, (obstruction)] |cathode  [cathode..gap x nx_gap]  anode|
        [puff, column x (nx-1)] [collector]

    Twin cathode (``TwinCathode``) mirrors the source end instead of the collector
    (plan §11 decision 4), putting its cathode surface at ``z = Lm``. Column cells
    adjacent to an anode face carry the ``puff`` role (gas enters in front of the
    anode, §2); the plasma cell against a cathode surface carries the ``cathode``
    role so cathode surface terms have somewhere to land (§8).

    The annular cathode-structure obstruction is a *real cell* of length ``Lcs``
    (plan §11 decision 1), so it holds gas and its inventory reaches the pump. It
    is omitted entirely when ``Lcs <= 0``, which is the legacy limit.
    """
    nx = int(input_dict.get("nx", 60))
    nx_gap = int(input_dict.get("nx_gap", 5))
    if nx <= 0:
        raise ValueError(f"nx must be positive (got {nx})")
    if nx_gap <= 0:
        raise ValueError(f"nx_gap must be positive (got {nx_gap})")

    total_length = float(input_dict.get("Lm", 2000.0))
    twin = bool(flags.get("TwinCathode", False))

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

    column_roles = ["puff"] + ["column"] * (nx - 1)
    if twin:
        column_roles[-1] = "puff"
    column_lengths = [column_length / nx] * nx

    roles = behind_roles + gap_roles + column_roles
    lengths = behind_lengths + gap_lengths + column_lengths
    if twin:
        roles += list(reversed(gap_roles)) + list(reversed(behind_roles))
        lengths += list(reversed(gap_lengths)) + list(reversed(behind_lengths))
    else:
        roles += ["collector"]
        lengths += [collector_length]

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
        anode_transparency=1.0 - float(input_dict.get("eta", 0.0)),
    )


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
    anode_transparency=1.0,
):
    """Derive the face arrays from the cell arrays and pack a ``Sim1DGeometry``.

    Plasma walls are found from the roles: both external ends, plus any interior
    face where a plasma-dead cell (plenum/obstruction) abuts a live one. Legacy
    roles contain no plasma-dead cells, so only the external ends are walls --
    exactly today's behavior.
    """
    cells = length_cm.size
    plasma_face_area_cm2 = _face_area(plasma_area_cm2)
    # Restricting apertures for the neutral conductance (see class docstring).
    neutral_face_area_cm2 = _face_min(neutral_area_cm2)
    neutral_face_hydraulic_radius_cm = _face_min(neutral_hydraulic_radius_cm)

    dead = np.asarray([role in PLASMA_DEAD_ROLES for role in cell_role], dtype=bool)
    plasma_open = np.ones(cells + 1, dtype=bool)
    plasma_transmission = np.ones(cells + 1, dtype=float)
    heat_transmission = np.ones(cells + 1, dtype=float)
    walls = [0, cells]
    walls += [face for face in range(1, cells) if dead[face - 1] != dead[face]]
    for face in walls:
        plasma_open[face] = False
        plasma_transmission[face] = 0.0
        heat_transmission[face] = 0.0

    # The anode mesh is plasma-open but partially blocking, and the three
    # transmissions are independent (§3): the solid fraction eta intercepts the
    # directed plasma flux (§11 decision 7), blocks the same fraction of parallel
    # conduction, and the neutral aperture is the transparency times the machine
    # cross-section. eta = 0 recovers a fully transparent anode -- the legacy limit.
    transparency = float(anode_transparency)
    if not 0.0 <= transparency <= 1.0:
        raise ValueError(
            f"anode transparency must lie in [0, 1] (got {transparency})"
        )
    for face in np.asarray(
        [] if anode_face_indices is None else anode_face_indices, dtype=int
    ):
        plasma_transmission[face] = transparency
        heat_transmission[face] = transparency
        neutral_face_area_cm2[face] = neutral_face_area_cm2[face] * transparency

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
        plasma_open=plasma_open,
        plasma_transmission=plasma_transmission,
        heat_transmission=heat_transmission,
        neutral_face_conductance_cm3_s=neutral_face_conductance_cm3_s,
        center_distance_cm=center_distance_cm,
        cathode_face_indices=(
            np.empty(0, dtype=int)
            if cathode_face_indices is None
            else np.asarray(cathode_face_indices, dtype=int)
        ),
        anode_face_indices=(
            np.empty(0, dtype=int)
            if anode_face_indices is None
            else np.asarray(anode_face_indices, dtype=int)
        ),
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
