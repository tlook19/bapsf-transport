from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Sim1DGeometry:
    """Axial source-domain-end layout for conservative 1D state arrays."""

    nx: int
    length_cm: np.ndarray
    z_edges_cm: np.ndarray
    z_cm: np.ndarray
    cell_role: np.ndarray
    Rp_cm: np.ndarray
    Rm_cm: np.ndarray
    plasma_area_cm2: np.ndarray
    neutral_area_cm2: np.ndarray
    plasma_volume_cm3: np.ndarray
    neutral_volume_cm3: np.ndarray
    volume_ratio: np.ndarray
    plasma_face_area_cm2: np.ndarray
    neutral_face_area_cm2: np.ndarray
    center_distance_cm: np.ndarray

    @property
    def cells(self):
        return self.nx + 2

    @property
    def dz_cm(self):
        return self.length_cm[1] if self.nx else 0.0


def build_geometry(input_dict):
    """Build the default hybrid 0D-source, 1D-domain, 0D-end geometry."""
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

    return Sim1DGeometry(
        nx=nx,
        length_cm=length_cm,
        z_edges_cm=z_edges_cm,
        z_cm=z_cm,
        cell_role=cell_role,
        Rp_cm=Rp_cm,
        Rm_cm=Rm_cm,
        plasma_area_cm2=plasma_area_cm2,
        neutral_area_cm2=neutral_area_cm2,
        plasma_volume_cm3=plasma_volume_cm3,
        neutral_volume_cm3=neutral_volume_cm3,
        volume_ratio=volume_ratio,
        plasma_face_area_cm2=plasma_face_area_cm2,
        neutral_face_area_cm2=neutral_face_area_cm2,
        center_distance_cm=center_distance_cm,
    )


def _face_area(cell_area):
    """Return external plus internal face areas from adjacent cell averages."""
    face_area = np.empty(cell_area.size + 1, dtype=float)
    face_area[0] = cell_area[0]
    face_area[-1] = cell_area[-1]
    face_area[1:-1] = 0.5 * (cell_area[:-1] + cell_area[1:])
    return face_area
