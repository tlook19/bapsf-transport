from dataclasses import dataclass

import numpy as np

from ..core.state import ConservativeState1D, derive_state


@dataclass(frozen=True)
class CathodeCellState1D:
    """Primitive/source-cell state passed toward a cathode adapter."""

    index: int
    role: str
    n: float
    nn: float
    Te: float
    Ti: float
    u: float
    plasma_volume_cm3: float
    neutral_volume_cm3: float
    plasma_area_cm2: float
    neutral_area_cm2: float
    length_cm: float
    Rp_cm: float
    Rm_cm: float


@dataclass(frozen=True)
class CathodeBoundaryState1D:
    """Source/end boundary state and circuit placeholders for cathode coupling."""

    source: CathodeCellState1D
    end: CathodeCellState1D
    enabled: bool
    mode: str
    end_mode: str
    twin_cathode: bool
    circuit: dict


@dataclass(frozen=True)
class CathodeSourceTerms1D:
    """Conservative cathode source placeholders and raw metadata."""

    rhs: ConservativeState1D
    enabled: bool
    mode: str
    metadata: dict


def cathode_boundary_state(
    state,
    floors,
    ion_mass_g,
    geometry,
    input_dict,
    input_flags,
):
    """Return finite source/end quantities for a future cathode solver adapter."""
    derived = derive_state(state, floors=floors, ion_mass_g=ion_mass_g)
    return CathodeBoundaryState1D(
        source=_cell_state(0, state, derived, geometry),
        end=_cell_state(geometry.cells - 1, state, derived, geometry),
        enabled=bool(input_flags.get("cathode_coupling", False)),
        mode=input_dict.get("cathode_model", "disabled"),
        end_mode=input_dict.get("end_mode", "collector"),
        twin_cathode=bool(input_flags.get("TwinCathode", False)),
        circuit=_circuit_placeholders(input_dict),
    )


def cathode_source_terms(
    state,
    floors,
    ion_mass_g,
    geometry,
    input_dict,
    input_flags,
):
    """Return disabled cathode source placeholders in conservative units."""
    boundary = cathode_boundary_state(
        state=state,
        floors=floors,
        ion_mass_g=ion_mass_g,
        geometry=geometry,
        input_dict=input_dict,
        input_flags=input_flags,
    )
    zeros = np.zeros(geometry.cells, dtype=float)
    return CathodeSourceTerms1D(
        rhs=ConservativeState1D(
            n=zeros,
            nn=zeros.copy(),
            M=zeros.copy(),
            Ee=zeros.copy(),
            Ei=zeros.copy(),
        ),
        enabled=boundary.enabled,
        mode=boundary.mode,
        metadata={
            "source_index": boundary.source.index,
            "end_index": boundary.end.index,
            "end_mode": boundary.end_mode,
            "twin_cathode": boundary.twin_cathode,
            "circuit": dict(boundary.circuit),
        },
    )


def _cell_state(index, state, derived, geometry):
    return CathodeCellState1D(
        index=int(index),
        role=str(geometry.cell_role[index]),
        n=float(state.n[index]),
        nn=float(state.nn[index]),
        Te=float(derived.Te[index]),
        Ti=float(derived.Ti[index]),
        u=float(derived.u[index]),
        plasma_volume_cm3=float(geometry.plasma_volume_cm3[index]),
        neutral_volume_cm3=float(geometry.neutral_volume_cm3[index]),
        plasma_area_cm2=float(geometry.plasma_area_cm2[index]),
        neutral_area_cm2=float(geometry.neutral_area_cm2[index]),
        length_cm=float(geometry.length_cm[index]),
        Rp_cm=float(geometry.Rp_cm[index]),
        Rm_cm=float(geometry.Rm_cm[index]),
    )


def _circuit_placeholders(input_dict):
    keys = (
        "V_bank",
        "T_s",
        "phi_wf",
        "C_R",
        "R_comp",
        "eta",
        "L_cath",
        "R_cath",
    )
    return {key: input_dict.get(key) for key in keys if key in input_dict}
