from dataclasses import dataclass
import math

import numpy as np

from cablp.funcs._cathode_solver import DeviceConfig, solve_beam_system
from cablp.vars._cons import ev_to_erg, qe_SI

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
    dN_loss[0] = _cathode_particle_loss_rate(
        cathode_solve.beam_result.result,
        eta=input_dict["eta"],
    )
    if boundary.twin_cathode and cathode_solve.beam_result.result_twin is not None:
        dN_loss[-1] = _cathode_particle_loss_rate(
            cathode_solve.beam_result.result_twin,
            eta=input_dict["eta"],
        )

    plasma_loss_rate = dN_loss / geometry.plasma_volume_cm3
    neutral_gain_rate = dN_loss / geometry.neutral_volume_cm3
    electron_power_loss_W = zeros.copy()
    electron_power_loss_W[0] = _electron_power_loss_W(
        cathode_solve.beam_result.result
    )
    if boundary.twin_cathode and cathode_solve.beam_result.result_twin is not None:
        electron_power_loss_W[-1] = _electron_power_loss_W(
            cathode_solve.beam_result.result_twin
        )
    # P_cathode_e and P_anode_e are colocated in the 0D source cell for now.
    # A future resolved 1D source needs these split so each loss lands at the
    # appropriate cathode/anode location.
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


def beam_absorption_weights(length_cm, l_b_profile, cathode_index):
    """Return Beer-Lambert absorbed beam fractions for one cathode."""
    length_cm = np.asarray(length_cm, dtype=float)
    l_b_profile = np.asarray(l_b_profile, dtype=float)
    cells = length_cm.size
    if l_b_profile.shape != (cells,):
        raise ValueError(
            f"l_b_profile must have shape ({cells},), got {l_b_profile.shape}"
        )
    if cathode_index == 0:
        order = np.arange(cells)
    elif cathode_index in {-1, cells - 1}:
        order = np.arange(cells - 1, -1, -1)
    else:
        raise ValueError(
            f"cathode_index must be 0 or -1/{cells - 1}, got {cathode_index}"
        )

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

    S_beam, beam_power_density = _beam_ionization_sources(
        state=state,
        geometry=geometry,
        cathode_solve=cathode_solve,
        boundary=boundary,
    )
    volume_ratio = geometry.plasma_volume_cm3 / geometry.neutral_volume_cm3
    Ti_birth = _birth_temperature(
        input_dict.get("Ti_birth_ionization", "floor"),
        derive_state(state, floors=floors, ion_mass_g=ion_mass_g).Ti,
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
):
    zeros = np.zeros(geometry.cells, dtype=float)
    beam_result = cathode_solve.beam_result
    S_beam = zeros.copy()
    beam_power_density = zeros.copy()
    source_profile = _beam_ionization_profile(
        state=state,
        geometry=geometry,
        beam_result=beam_result,
        cathode_index=0,
    )
    S_beam += source_profile
    beam_power_density += _beam_power_deposition_density(
        geometry=geometry,
        beam_result=beam_result,
        solver_result=beam_result.result,
        cathode_index=0,
    )
    if boundary.twin_cathode and beam_result.result_twin is not None:
        twin_profile = _beam_ionization_profile(
            state=state,
            geometry=geometry,
            beam_result=beam_result,
            cathode_index=-1,
        )
        S_beam += twin_profile
        beam_power_density += _beam_power_deposition_density(
            geometry=geometry,
            beam_result=beam_result,
            solver_result=beam_result.result_twin,
            cathode_index=-1,
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


def _electron_power_loss_W(result):
    return result.P_cathode_e + result.P_anode_e


def _beam_ionization_profile(state, geometry, beam_result, cathode_index):
    beam_cross = beam_result.beam_cross[cathode_index]
    if beam_cross == 0.0:
        return np.zeros(geometry.cells, dtype=float)
    l_b_profile = (
        beam_result.l_b_profile
        if cathode_index == 0
        else beam_result.l_b_profile_twin
    )
    p_beam = l_b_profile * beam_cross * state.nn
    weights = beam_absorption_weights(
        length_cm=geometry.length_cm,
        l_b_profile=l_b_profile,
        cathode_index=cathode_index,
    )
    return (
        weights
        * p_beam
        * beam_result.n_beam[cathode_index]
        * beam_result.v_beam[cathode_index]
        / geometry.length_cm
    )


def _beam_power_deposition_density(
    geometry,
    beam_result,
    solver_result,
    cathode_index,
):
    """Return the beam/ohmic power deposition density [erg cm^-3 s^-1].

    ``P_prim`` is carried into the column by the primary beam and so deposits
    along the Beer-Lambert absorption profile. ``P_ohmic`` is the ohmic
    dissipation at the cathode's own boundary cell and deposits there only,
    rather than being spread along the beam path.
    """
    beam_cross = beam_result.beam_cross[cathode_index]
    if beam_cross == 0.0:
        return np.zeros(geometry.cells, dtype=float)
    l_b_profile = (
        beam_result.l_b_profile
        if cathode_index == 0
        else beam_result.l_b_profile_twin
    )
    weights = beam_absorption_weights(
        length_cm=geometry.length_cm,
        l_b_profile=l_b_profile,
        cathode_index=cathode_index,
    )
    density = (
        weights * solver_result.P_prim * 1.0e7 / geometry.plasma_volume_cm3
    )
    density[cathode_index] += (
        solver_result.P_ohmic
        * 1.0e7
        / geometry.plasma_volume_cm3[cathode_index]
    )
    return density


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
