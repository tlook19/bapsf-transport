from dataclasses import dataclass
import math

import numpy as np

from cablp.funcs._cathode_solver import DeviceConfig, solve_beam_system
from cablp.vars._cons import ev_to_erg, qe_SI

from ..core.geometry import (
    anode_flanking_cells,
    cathode_adjacent_cells,
    gap_cell_indices,
)
from ..core.state import ConservativeState1D, derive_state
from .reactions import _birth_temperature


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


@dataclass(frozen=True)
class CathodeSolve1D:
    """Opt-in cathode solve result without conservative RHS coupling."""

    boundary: CathodeBoundaryState1D
    beam_result: object | None
    device_config: DeviceConfig | None
    x0_next: float | None
    x0_twin_next: float | None
    metadata: dict


def cathode_sample_indices(geometry):
    """Return the ``(source, end)`` cells the cathode circuit samples.

    The cathode solve builds its ion current from the plasma against the cathode
    surface, so in resolved geometry it must read the *cathode-adjacent* cell --
    cell ``[0]`` there is the plasma-dead plenum, whose floor density and
    temperature would drive the circuit with garbage. Legacy geometry has no
    cathode faces and keeps its source/end cells (§8).

    A twin machine samples both cathodes; otherwise the ``end`` slot is the
    collector, which is what ``end_mode`` describes.
    """
    cathode_cells = cathode_adjacent_cells(geometry)
    if not cathode_cells:
        return 0, geometry.cells - 1
    source_index = int(cathode_cells[0])
    if len(cathode_cells) > 1:
        return source_index, int(cathode_cells[-1])
    return source_index, geometry.cells - 1


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
    source_index, end_index = cathode_sample_indices(geometry)
    return CathodeBoundaryState1D(
        source=_cell_state(source_index, state, derived, geometry),
        end=_cell_state(end_index, state, derived, geometry),
        enabled=bool(input_flags.get("cathode_coupling", False)),
        mode=input_dict.get("cathode_model", "disabled"),
        end_mode=input_dict.get("end_mode", "collector"),
        twin_cathode=bool(input_flags.get("TwinCathode", False)),
        circuit=_circuit_placeholders(input_dict),
    )


def cathode_device_config(input_dict, input_flags, mu):
    """Build the existing cathode solver's static device configuration."""
    R_cath = float(input_dict["R_cath"])
    return DeviceConfig(
        A_c=math.pi * R_cath**2,
        mu=mu,
        V_bank=float(input_dict["V_bank"]),
        T_s=float(input_dict["T_s"]),
        phi_wf=float(input_dict["phi_wf"]),
        C_R=float(input_dict["C_R"]),
        R_comp=float(input_dict["R_comp"]),
        eta=float(input_dict["eta"]),
        Twin=bool(input_flags.get("TwinCathode", False)),
        L_cath=float(input_dict["L_cath"]),
        R_cath=R_cath,
    )


def solve_cathode_boundary(
    state,
    floors,
    ion_mass_g,
    mu,
    geometry,
    input_dict,
    input_flags,
    beam_cross_prev,
    I_ion,
    gas_type,
    x0=None,
    x0_twin=None,
    floating=False,
):
    """Call the cathode/beam solver and return raw diagnostics only."""
    boundary = cathode_boundary_state(
        state=state,
        floors=floors,
        ion_mass_g=ion_mass_g,
        geometry=geometry,
        input_dict=input_dict,
        input_flags=input_flags,
    )
    if not boundary.enabled:
        return CathodeSolve1D(
            boundary=boundary,
            beam_result=None,
            device_config=None,
            x0_next=x0,
            x0_twin_next=x0_twin,
            metadata={
                "enabled": False,
                "mode": boundary.mode,
                "floating": bool(floating),
                "source_index": boundary.source.index,
                "end_index": boundary.end.index,
                "end_mode": boundary.end_mode,
                "twin_cathode": boundary.twin_cathode,
                "circuit": dict(boundary.circuit),
            },
        )

    derived = derive_state(state, floors=floors, ion_mass_g=ion_mass_g)
    device_config = cathode_device_config(input_dict, input_flags, mu)
    beam_cross_prev = np.asarray(beam_cross_prev, dtype=float)
    if beam_cross_prev.shape != (geometry.cells,):
        raise ValueError(
            "beam_cross_prev must have shape "
            f"({geometry.cells},), got {beam_cross_prev.shape}"
        )
    beam_result = solve_beam_system(
        config=device_config,
        Te=derived.Te,
        ne=state.n,
        nn=state.nn,
        beam_cross_prev=beam_cross_prev,
        plasma_cross=geometry.plasma_area_cm2,
        I_ion=I_ion,
        gas_type=gas_type,
        x0=x0,
        x0_twin=x0_twin,
        floating=bool(floating),
    )
    return CathodeSolve1D(
        boundary=boundary,
        beam_result=beam_result,
        device_config=device_config,
        x0_next=beam_result.x0_next,
        x0_twin_next=beam_result.x0_twin_next,
        metadata={
            "enabled": True,
            "mode": boundary.mode,
            "floating": bool(floating),
            "source_index": boundary.source.index,
            "end_index": boundary.end.index,
            "end_mode": boundary.end_mode,
            "twin_cathode": boundary.twin_cathode,
            "circuit": dict(boundary.circuit),
            "result": _solver_result_metadata(beam_result.result),
            "result_twin": _solver_result_metadata(beam_result.result_twin),
        },
    )


def cathode_source_terms(
    state,
    floors,
    ion_mass_g,
    geometry,
    input_dict,
    input_flags,
    cathode_solve=None,
):
    """Return cathode surface particle and electron-power losses."""
    boundary = cathode_boundary_state(
        state=state,
        floors=floors,
        ion_mass_g=ion_mass_g,
        geometry=geometry,
        input_dict=input_dict,
        input_flags=input_flags,
    )
    zeros = np.zeros(geometry.cells, dtype=float)
    if (
        not boundary.enabled
        or cathode_solve is None
        or cathode_solve.beam_result is None
    ):
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
                "surface_particle_loss_s_inv": zeros.copy(),
            },
        )

    derived = derive_state(state, floors=floors, ion_mass_g=ion_mass_g)
    dN_loss = zeros.copy()
    # An absorbing cathode face already drains the plasma at the Bohm flux, which
    # is the same criterion the circuit's I_i is built from (A_c*e*n*c_s*exp(-0.5)
    # on this cell's n and Te), so the face and the circuit agree on the current.
    # Applying this volumetric loss as well would remove it twice. The electron
    # power loss below is a separate channel and still applies.
    face_absorbs = bool(
        np.any(np.asarray(getattr(geometry, "plasma_absorbing", ()), dtype=bool))
    )
    if not face_absorbs:
        dN_loss[0] = _cathode_particle_loss_rate(
            cathode_solve.beam_result.result,
            eta=input_dict["eta"],
        )
        if (
            boundary.twin_cathode
            and cathode_solve.beam_result.result_twin is not None
        ):
            dN_loss[-1] = _cathode_particle_loss_rate(
                cathode_solve.beam_result.result_twin,
                eta=input_dict["eta"],
            )

    plasma_loss_rate = dN_loss / geometry.plasma_volume_cm3
    neutral_gain_rate = dN_loss / geometry.neutral_volume_cm3
    # Sheath electron power: P_cathode_e is lost at the cathode surface and
    # P_anode_e at the anode mesh (§8). Legacy has neither resolved, so both stay
    # colocated in its source cell exactly as before; resolved geometry lands each
    # at its own electrode.
    electron_power_loss_W = zeros.copy()
    cathode_cells = cathode_adjacent_cells(geometry)
    anode_pairs = anode_flanking_cells(geometry)
    if cathode_cells:
        _deposit_electrode_power(
            electron_power_loss_W,
            result=cathode_solve.beam_result.result,
            cathode_cell=int(cathode_cells[0]),
            anode_pair=anode_pairs[0] if anode_pairs else None,
            state=state,
            derived=derived,
        )
        if (
            boundary.twin_cathode
            and cathode_solve.beam_result.result_twin is not None
        ):
            _deposit_electrode_power(
                electron_power_loss_W,
                result=cathode_solve.beam_result.result_twin,
                cathode_cell=int(cathode_cells[-1]),
                anode_pair=anode_pairs[-1] if len(anode_pairs) > 1 else None,
                state=state,
                derived=derived,
            )
    else:
        electron_power_loss_W[0] = _electron_power_loss_W(
            cathode_solve.beam_result.result
        )
        if (
            boundary.twin_cathode
            and cathode_solve.beam_result.result_twin is not None
        ):
            electron_power_loss_W[-1] = _electron_power_loss_W(
                cathode_solve.beam_result.result_twin
            )
    electron_power_loss_density = electron_power_loss_W * 1.0e7 / (
        geometry.plasma_volume_cm3
    )
    return CathodeSourceTerms1D(
        rhs=ConservativeState1D(
            n=-plasma_loss_rate,
            nn=neutral_gain_rate,
            M=-ion_mass_g * derived.u * plasma_loss_rate,
            Ee=-electron_power_loss_density,
            Ei=-1.5 * ev_to_erg * derived.Ti * plasma_loss_rate,
        ),
        enabled=boundary.enabled,
        mode=boundary.mode,
        metadata={
            "source_index": boundary.source.index,
            "end_index": boundary.end.index,
            "end_mode": boundary.end_mode,
            "twin_cathode": boundary.twin_cathode,
            "circuit": dict(boundary.circuit),
            "surface_particle_loss_s_inv": dN_loss,
            "source_surface_particle_loss_s_inv": float(dN_loss[0]),
            "end_surface_particle_loss_s_inv": float(dN_loss[-1]),
            "electron_power_loss_W": electron_power_loss_W,
            "source_electron_power_loss_W": float(electron_power_loss_W[0]),
            "end_electron_power_loss_W": float(electron_power_loss_W[-1]),
        },
    )


def beam_launch(geometry, end=0):
    """Return the ``(cell, direction)`` a cathode's beam is launched from.

    The beam starts at the plasma cell against the cathode surface and travels
    into the machine, so in resolved geometry it must not begin at cell ``[0]``
    (the plenum) nor deposit into the cells behind the cathode.
    """
    cathode_cells = cathode_adjacent_cells(geometry)
    if not cathode_cells:
        return (0, 1) if end == 0 else (geometry.cells - 1, -1)
    if end == 0:
        return int(cathode_cells[0]), 1
    return int(cathode_cells[-1]), -1


def beam_absorption_weights(length_cm, l_b_profile, cathode_index, direction=None):
    """Return Beer-Lambert absorbed beam fractions for one cathode.

    The beam is launched from ``cathode_index`` and traverses away from it.
    ``direction`` is +1 for a beam heading toward increasing z and -1 for the
    other way; it is inferred for the legacy end cells. Cells *behind* the launch
    point get zero weight -- in resolved geometry those are the plenum and the
    obstruction, which the beam never enters (§5).
    """
    length_cm = np.asarray(length_cm, dtype=float)
    l_b_profile = np.asarray(l_b_profile, dtype=float)
    cells = length_cm.size
    if l_b_profile.shape != (cells,):
        raise ValueError(
            f"l_b_profile must have shape ({cells},), got {l_b_profile.shape}"
        )
    launch = int(cathode_index) % cells
    if direction is None:
        if launch == 0:
            direction = 1
        elif launch == cells - 1:
            direction = -1
        else:
            raise ValueError(
                "direction is required when the beam is not launched from an "
                f"end cell (got cathode_index={cathode_index})"
            )
    if direction > 0:
        order = np.arange(launch, cells)
    else:
        order = np.arange(launch, -1, -1)

    l_b_ordered = l_b_profile[order]
    dx_ordered = length_cm[order]
    safe_l_b = np.where(l_b_ordered > 0.0, l_b_ordered, np.inf)
    tau = np.cumsum(dx_ordered / safe_l_b)
    tau_in = np.concatenate([[0.0], tau[:-1]])
    exp_neg_tau_in = np.exp(-tau_in)
    absorbed_ordered = exp_neg_tau_in * (1.0 - np.exp(-dx_ordered / safe_l_b))

    weights = np.zeros(cells, dtype=float)
    weights[order] = absorbed_ordered
    return weights


def beam_ionization_rhs(
    state,
    floors,
    ion_mass_g,
    geometry,
    input_dict,
    input_flags,
    I_ion,
    cathode_solve=None,
):
    """Return conservative beam ionization and beam electron energy terms."""
    terms = beam_ionization_rhs_terms(
        state=state,
        floors=floors,
        ion_mass_g=ion_mass_g,
        geometry=geometry,
        input_dict=input_dict,
        input_flags=input_flags,
        I_ion=I_ion,
        cathode_solve=cathode_solve,
    )
    rhs = terms["beam_ionization_birth"]
    for term in (
        terms["beam_power_deposition"],
        terms["beam_ionization_cost"],
    ):
        rhs = ConservativeState1D(
            n=rhs.n + term.n,
            nn=rhs.nn + term.nn,
            M=rhs.M + term.M,
            Ee=rhs.Ee + term.Ee,
            Ei=rhs.Ei + term.Ei,
        )
    return rhs


def beam_ionization_rhs_terms(
    state,
    floors,
    ion_mass_g,
    geometry,
    input_dict,
    input_flags,
    I_ion,
    cathode_solve=None,
):
    """Return split beam ionization particle, power, and cost terms."""
    boundary = cathode_boundary_state(
        state=state,
        floors=floors,
        ion_mass_g=ion_mass_g,
        geometry=geometry,
        input_dict=input_dict,
        input_flags=input_flags,
    )
    zeros = np.zeros(geometry.cells, dtype=float)
    if (
        not boundary.enabled
        or cathode_solve is None
        or cathode_solve.beam_result is None
    ):
        return _zero_beam_terms(zeros)

    beam_derived = derive_state(state, floors=floors, ion_mass_g=ion_mass_g)
    S_beam, beam_power_density = _beam_ionization_sources(
        state=state,
        geometry=geometry,
        cathode_solve=cathode_solve,
        boundary=boundary,
        Te=beam_derived.Te,
    )
    volume_ratio = geometry.plasma_volume_cm3 / geometry.neutral_volume_cm3
    Ti_birth = _birth_temperature(
        input_dict.get("Ti_birth_ionization", "floor"),
        beam_derived.Ti,
        floors["Ti"],
    )
    return {
        "beam_ionization_birth": ConservativeState1D(
            n=S_beam,
            nn=-S_beam * volume_ratio,
            M=zeros.copy(),
            Ee=zeros.copy(),
            Ei=1.5 * ev_to_erg * Ti_birth * S_beam,
        ),
        "beam_power_deposition": ConservativeState1D(
            n=zeros,
            nn=zeros.copy(),
            M=zeros.copy(),
            Ee=beam_power_density,
            Ei=zeros.copy(),
        ),
        "beam_ionization_cost": ConservativeState1D(
            n=zeros,
            nn=zeros.copy(),
            M=zeros.copy(),
            Ee=-I_ion * ev_to_erg * S_beam,
            Ei=zeros.copy(),
        ),
    }


def _beam_ionization_sources(
    state,
    geometry,
    cathode_solve,
    boundary,
    Te=None,
):
    zeros = np.zeros(geometry.cells, dtype=float)
    beam_result = cathode_solve.beam_result
    S_beam = zeros.copy()
    beam_power_density = zeros.copy()
    source_profile = _beam_ionization_profile(
        state=state,
        geometry=geometry,
        beam_result=beam_result,
        end=0,
    )
    S_beam += source_profile
    beam_power_density += _beam_power_deposition_density(
        geometry=geometry,
        beam_result=beam_result,
        solver_result=beam_result.result,
        end=0,
        Te=Te,
    )
    if boundary.twin_cathode and beam_result.result_twin is not None:
        twin_profile = _beam_ionization_profile(
            state=state,
            geometry=geometry,
            beam_result=beam_result,
            end=-1,
        )
        S_beam += twin_profile
        beam_power_density += _beam_power_deposition_density(
            geometry=geometry,
            beam_result=beam_result,
            solver_result=beam_result.result_twin,
            end=-1,
            Te=Te,
        )

    return S_beam, beam_power_density


def _zero_beam_terms(zeros):
    return {
        "beam_ionization_birth": ConservativeState1D(
            n=zeros,
            nn=zeros.copy(),
            M=zeros.copy(),
            Ee=zeros.copy(),
            Ei=zeros.copy(),
        ),
        "beam_power_deposition": ConservativeState1D(
            n=zeros.copy(),
            nn=zeros.copy(),
            M=zeros.copy(),
            Ee=zeros.copy(),
            Ei=zeros.copy(),
        ),
        "beam_ionization_cost": ConservativeState1D(
            n=zeros.copy(),
            nn=zeros.copy(),
            M=zeros.copy(),
            Ee=zeros.copy(),
            Ei=zeros.copy(),
        ),
    }


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


def _cathode_particle_loss_rate(result, eta):
    return (1.0 + 2.0 * float(eta)) * result.I_i / qe_SI


def _deposit_electrode_power(
    electron_power_loss_W, result, cathode_cell, anode_pair, state, derived
):
    """Land P_cathode_e at the cathode cell and P_anode_e at the anode mesh.

    The anode collects on both mesh faces, so its sheath power is split between
    the two flanking cells in proportion to each face's Bohm collection -- the
    same weighting ``anode_collection_rhs`` uses, so power and particles are
    removed on the same side. With no resolved anode the whole of P_anode_e falls
    back to the cathode cell, which is where the lumped model puts it.
    """
    electron_power_loss_W[cathode_cell] += result.P_cathode_e
    if anode_pair is None:
        electron_power_loss_W[cathode_cell] += result.P_anode_e
        return
    gap_side, column_side = anode_pair
    # Bohm collection ~ n * c_s, and c_s ~ sqrt(Te/mu) with the same mu on both
    # sides, so mu cancels in the normalized split.
    weights = np.array(
        [
            state.n[gap_side] * np.sqrt(derived.Te[gap_side]),
            state.n[column_side] * np.sqrt(derived.Te[column_side]),
        ],
        dtype=float,
    )
    total = weights.sum()
    if not np.isfinite(total) or total <= 0.0:
        weights = np.full(2, 0.5)
    else:
        weights = weights / total
    electron_power_loss_W[gap_side] += weights[0] * result.P_anode_e
    electron_power_loss_W[column_side] += weights[1] * result.P_anode_e


def _electron_power_loss_W(result):
    return result.P_cathode_e + result.P_anode_e


def _beam_ionization_profile(state, geometry, beam_result, end=0):
    launch, direction = beam_launch(geometry, end=end)
    beam_cross = beam_result.beam_cross[launch]
    if beam_cross == 0.0:
        return np.zeros(geometry.cells, dtype=float)
    l_b_profile = (
        beam_result.l_b_profile if end == 0 else beam_result.l_b_profile_twin
    )
    p_beam = l_b_profile * beam_cross * state.nn
    weights = beam_absorption_weights(
        length_cm=geometry.length_cm,
        l_b_profile=l_b_profile,
        cathode_index=launch,
        direction=direction,
    )
    return (
        weights
        * p_beam
        * beam_result.n_beam[launch]
        * beam_result.v_beam[launch]
        / geometry.length_cm
    )


def _beam_power_deposition_density(
    geometry,
    beam_result,
    solver_result,
    end=0,
    Te=None,
):
    """Return the beam/ohmic power deposition density [erg cm^-3 s^-1].

    ``P_prim`` is carried into the column by the primary beam and so deposits
    along the Beer-Lambert absorption profile.

    ``P_ohmic = I^2 R_p`` is dissipated in the plasma *between* the cathode and
    the anode, so it is spread over the cathode-anode gap rather than piled into
    one boundary cell. The discharge current density is essentially uniform along
    the gap, so the power per unit length follows the local Spitzer resistivity,
    ``eta_sp ~ Te^-3/2``: dissipation concentrates wherever the gap is coldest.
    Legacy geometry has no resolved gap, so the whole of ``P_ohmic`` still lands
    on the single source/end cell exactly as before.
    """
    launch, direction = beam_launch(geometry, end=end)
    beam_cross = beam_result.beam_cross[launch]
    if beam_cross == 0.0:
        return np.zeros(geometry.cells, dtype=float)
    l_b_profile = (
        beam_result.l_b_profile if end == 0 else beam_result.l_b_profile_twin
    )
    weights = beam_absorption_weights(
        length_cm=geometry.length_cm,
        l_b_profile=l_b_profile,
        cathode_index=launch,
        direction=direction,
    )
    density = (
        weights * solver_result.P_prim * 1.0e7 / geometry.plasma_volume_cm3
    )
    gap = np.asarray(gap_cell_indices(geometry, end=end), dtype=int)
    ohmic_weights = _ohmic_gap_weights(geometry, gap, Te)
    density[gap] += (
        ohmic_weights
        * solver_result.P_ohmic
        * 1.0e7
        / geometry.plasma_volume_cm3[gap]
    )
    return density


def _ohmic_gap_weights(geometry, gap, Te):
    """Return the normalized share of ``P_ohmic`` deposited in each gap cell.

    ``P_cell = j^2 * eta_sp * V_cell``; with the current density uniform along
    the gap this reduces to ``P_cell ~ eta_sp * length``, and Spitzer resistivity
    gives ``eta_sp ~ Te^-3/2``. A single-cell gap normalizes to exactly 1.0, so
    legacy deposition is bit-identical.
    """
    lengths = np.asarray(geometry.length_cm, dtype=float)[gap]
    if Te is None or gap.size == 1:
        weights = lengths
    else:
        Te_gap = np.maximum(np.asarray(Te, dtype=float)[gap], 1e-30)
        weights = lengths * Te_gap**-1.5
    total = weights.sum()
    if not np.isfinite(total) or total <= 0.0:
        return np.full(gap.size, 1.0 / gap.size)
    return weights / total


def _solver_result_metadata(result):
    if result is None:
        return None
    keys = (
        "phi_c",
        "phi_a",
        "V_b",
        "I_i",
        "I_eth_star",
        "I_tot",
        "P_prim",
        "P_ohmic",
        "P_loss",
        "P_cathode_e",
        "P_anode_e",
        "beam_bypass_fraction",
        "l_b",
    )
    metadata = {key: float(getattr(result, key)) for key in keys}
    metadata["regime"] = result.regime
    metadata["long_mfp"] = bool(result.long_mfp)
    return metadata
