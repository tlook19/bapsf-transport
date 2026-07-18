from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Sim1DGeometry:
    """Axial layout for conservative 1D state arrays.

    Two modes populate the *same* fields (BOUNDARY_REGIONS_PLAN.md §13):

    - **legacy** (default): the historical 0D-source, 1D-domain, 0D-end lump.
    - **resolved** (``resolved_boundaries`` flag): the typed-segment machine of
      §3 -- plenum / cathode / anode / puff / column / collector cells.

    Operators read the fields, never the mode, so "off" is legacy inputs flowing
    through one code path rather than a parallel branch. The face-property arrays
    (``plasma_open``, ``heat_transmission``, ``neutral_face_conductance_cm3_s``)
    and the decoupled ``*_hydraulic_radius_cm`` are laid down here at their legacy
    defaults so a future array-driven operator reproduces today's behavior
    exactly; later milestones (M2+) rewire the operators to consume them.

    ``nx`` is the number of column/domain cells; ``cells`` is the true cell count
    (``nx + 2`` in legacy, larger in resolved), so index-agnostic consumers keep
    working in both modes.
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
    heat_transmission: np.ndarray
    neutral_face_conductance_cm3_s: np.ndarray
    center_distance_cm: np.ndarray

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

    plasma_face_area_cm2 = _face_area(plasma_area_cm2)
    neutral_face_area_cm2 = _face_area(neutral_area_cm2)
    center_distance_cm = np.diff(z_cm)

    # Legacy schema defaults: neutral hydraulic radius = Rm (one-radius model);
    # plasma reflecting walls only at the two external ends; heat conducts across
    # every interior face and no external face; no prescribed neutral apertures.
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
        plasma_face_area_cm2=plasma_face_area_cm2,
        neutral_face_area_cm2=neutral_face_area_cm2,
        center_distance_cm=center_distance_cm,
        interior_plasma_walls=(),
    )


def _build_resolved_geometry(input_dict, flags):
    """Build the resolved typed-segment machine (BOUNDARY_REGIONS_PLAN.md §3).

    Single-cathode (default): ``[plenum, cathode, anode, <column...>, collector]``
    with the first column cell tagged ``puff``. Twin cathode (``TwinCathode``):
    the symmetric mirror ``[plenum, cathode, anode, <column...>, anode, cathode,
    plenum]`` (plan §11 decision 4).

    This layout and its per-role lengths are provisional: the neutral apertures
    (obstruction/mesh) and the anode/obstruction face-vs-cell choices (§11 #1,#5)
    are wired in M2-M4. M1 only needs a *valid* typed geometry carrying the schema
    arrays, so the operators of later milestones have something to key off.
    """
    nx = int(input_dict.get("nx", 60))
    if nx <= 0:
        raise ValueError(f"nx must be positive (got {nx})")

    total_length = float(input_dict.get("Lm", 2000.0))
    twin = bool(flags.get("TwinCathode", False))

    plenum_length = float(input_dict.get("plenum_length_cm", 100.0))
    cathode_length = float(input_dict.get("cathode_length_cm", 30.0))
    anode_length = float(input_dict.get("anode_length_cm", 10.0))
    collector_length = float(input_dict.get("collector_length_cm", 100.0))
    for name, value in (
        ("plenum_length_cm", plenum_length),
        ("cathode_length_cm", cathode_length),
        ("anode_length_cm", anode_length),
    ):
        if value <= 0.0:
            raise ValueError(f"{name} must be positive (got {value})")

    cathode_end = plenum_length + cathode_length + anode_length
    if twin:
        column_length = total_length - 2.0 * cathode_end
    else:
        if collector_length <= 0.0:
            raise ValueError(
                f"collector_length_cm must be positive (got {collector_length})"
            )
        column_length = total_length - cathode_end - collector_length
    if column_length <= 0.0:
        raise ValueError(
            "resolved boundary cells exceed Lm; no room for the column "
            f"(column_length={column_length} cm, Lm={total_length} cm)"
        )
    dz = column_length / nx

    # Build the cell sequence: (role, length). The first column cell carries the
    # puff role (gas injected in front of the source anode, §2).
    roles = ["plenum", "cathode", "anode"]
    lengths = [plenum_length, cathode_length, anode_length]
    roles += ["puff"] + ["column"] * (nx - 1)
    lengths += [dz] * nx
    if twin:
        roles += ["anode", "cathode", "plenum"]
        lengths += [anode_length, cathode_length, plenum_length]
    else:
        roles += ["collector"]
        lengths += [collector_length]

    length_cm = np.asarray(lengths, dtype=float)
    if not np.isclose(length_cm.sum(), total_length):
        raise ValueError(
            "resolved segment lengths must sum to Lm "
            f"(got {length_cm.sum()} cm vs {total_length} cm)"
        )
    cell_role = np.asarray(roles, dtype=object)
    cells = length_cm.size
    z_edges_cm = np.concatenate(([0.0], np.cumsum(length_cm)))
    z_cm = 0.5 * (z_edges_cm[:-1] + z_edges_cm[1:])

    Rp = float(input_dict.get("Rp", 18.0))
    Rm = float(input_dict.get("Rm", 50.0))
    Rp_cm = np.full(cells, Rp, dtype=float)
    Rm_cm = np.full(cells, Rm, dtype=float)
    # Plenum cells stay at full plasma radius here; they are made plasma-dead by
    # the reflecting cathode face below, not by shrinking Rp (§5 warns a near-zero
    # plasma volume blows up the flux divergence).
    neutral_hydraulic_radius_cm = Rm_cm.copy()

    plasma_area_cm2 = np.pi * Rp_cm**2
    neutral_area_cm2 = np.pi * Rm_cm**2
    plasma_volume_cm3 = plasma_area_cm2 * length_cm
    neutral_volume_cm3 = neutral_area_cm2 * length_cm
    volume_ratio = plasma_volume_cm3 / neutral_volume_cm3

    plasma_face_area_cm2 = _face_area(plasma_area_cm2)
    neutral_face_area_cm2 = _face_area(neutral_area_cm2)
    center_distance_cm = np.diff(z_cm)

    # The plasma domain is bounded inside the neutral domain by reflecting faces
    # at each plenum<->cathode boundary (§5): those interior faces are plasma
    # walls, in addition to the two external ends handled by _assemble_geometry.
    interior_plasma_walls = tuple(
        face
        for face in range(1, cells)
        if {cell_role[face - 1], cell_role[face]} == {"plenum", "cathode"}
    )

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
        plasma_face_area_cm2=plasma_face_area_cm2,
        neutral_face_area_cm2=neutral_face_area_cm2,
        center_distance_cm=center_distance_cm,
        interior_plasma_walls=interior_plasma_walls,
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
    plasma_face_area_cm2,
    neutral_face_area_cm2,
    center_distance_cm,
    interior_plasma_walls,
):
    """Derive the shared face-property arrays and pack a ``Sim1DGeometry``.

    ``interior_plasma_walls`` lists interior face indices that are plasma walls in
    addition to the two external ends (empty in legacy; the plenum<->cathode faces
    in resolved).
    """
    cells = length_cm.size
    neutral_face_hydraulic_radius_cm = _face_average(neutral_hydraulic_radius_cm)

    # Plasma reflecting walls: both external ends, plus any interior walls.
    plasma_open = np.ones(cells + 1, dtype=bool)
    heat_transmission = np.ones(cells + 1, dtype=float)
    for face in (0, cells):
        plasma_open[face] = False
        heat_transmission[face] = 0.0
    for face in interior_plasma_walls:
        plasma_open[face] = False
        heat_transmission[face] = 0.0

    # No prescribed neutral apertures yet: NaN => derive the molecular-flow
    # (Clausing) conductance from area + hydraulic radius (M2 populates this).
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
        heat_transmission=heat_transmission,
        neutral_face_conductance_cm3_s=neutral_face_conductance_cm3_s,
        center_distance_cm=center_distance_cm,
    )


def _face_area(cell_area):
    """Return external plus internal face areas from adjacent cell averages."""
    face_area = np.empty(cell_area.size + 1, dtype=float)
    face_area[0] = cell_area[0]
    face_area[-1] = cell_area[-1]
    face_area[1:-1] = 0.5 * (cell_area[:-1] + cell_area[1:])
    return face_area


def _face_average(cell_values):
    """Return face values from adjacent cell averages (externals = end cell).

    Same interior average as ``_face_area`` but for a per-cell quantity (e.g. the
    neutral hydraulic radius); the interior faces reproduce today's
    ``R_face = 0.5*(Rm[:-1] + Rm[1:])``.
    """
    face_values = np.empty(cell_values.size + 1, dtype=float)
    face_values[0] = cell_values[0]
    face_values[-1] = cell_values[-1]
    face_values[1:-1] = 0.5 * (cell_values[:-1] + cell_values[1:])
    return face_values
