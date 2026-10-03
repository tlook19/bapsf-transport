from dataclasses import dataclass
import hashlib
import math

import numpy as np

from scipy.optimize import brentq

from cablp.cathode.beam_deposition import (
    deposit_beam,
    plateau_edge_energy_eV,
    BeamDepositionResult,
    _coulomb_stopping_coefficient,
)
from cablp.cathode.circuit_common import (
    ANODE_TAIL_BOOKINGS,
    DeviceConfig,
    PlasmaState,
    compute_beam_bypass_fraction,
    compute_l_b,
    beam_launched_current_A,
)
from cablp.plasma.params import LN_LAMBDA_MIN, c_log, electron_mean_speed
from cablp.cathode.circuit_idriven import (
    solve_beam_system_idriven,
    solve_idriven,
)
from cablp.cathode.circuit_prescribed import solve_beam_system_prescribed
from cablp.constants import ev_to_erg, qe_SI

from ..core.geometry import (
    anode_flanking_cells,
    cathode_adjacent_cells,
    gap_cell_indices,
)
from ..core.state import ConservativeState1D, derive_state
from .flux import ion_sound_speed
from .reactions import _birth_temperature
from .sources import (
    electrode_sheath_alpha,
    ionization_birth_neutral_temperature_eV,
)


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
    """Source/end boundary state and circuit placeholders for cathode coupling.

    ``end`` is ``None`` on a column ending in a mirror face, which has no
    far electrode to sample (see ``cathode_sample_indices``).
    """

    source: CathodeCellState1D
    end: CathodeCellState1D | None
    enabled: bool
    twin_cathode: bool
    circuit: dict


@dataclass(frozen=True)
class CathodeSourceTerms1D:
    """Conservative electrode source rows and raw metadata.

    The electrode electron sheath power is SPLIT BY ELECTRODE across two
    conservative rows, which the solver registers under two names:

    ``rhs`` (``cathode_surface_loss``)
        The cathode's own members: the face's particle loss and its recycle
        on ``n``/``nn``/``M``, those ions' thermal energy on ``Ei``, and the
        CATHODE's electron sheath share on ``Ee`` -- ``P_cathode_e_thermal``
        under the repaired routing, which is milliwatts in discharge because
        the cathode sheath repels plasma electrons. Under a plasma-ABSORBING
        cathode face the first four members are EXACTLY ZERO: the boundary
        operator owns that face's particle, recycle, momentum and ion-thermal
        bookings, and this row is then the ``Ee`` member alone.
    ``anode_rhs`` (``anode_e_sheath_loss``)
        The ANODE electron sheath deposit, on ``Ee`` alone: every other field
        is zero. ``P_anode_e_thermal`` plus, when the anode is
        electron-repelling, the ``phi_a`` share -- landed at the
        anode-flanking cells under the Bohm split weights.

    SUM INVARIANCE: ``rhs + anode_rhs`` is the single combined row this pair
    replaces. Where an anode is resolved the two have DISJOINT per-cell
    support, so the sum reproduces the pre-split row bit-exactly (every cell
    of one row is an exact zero where the other is not). The one site where
    they overlap is the no-resolved-anode fallback, which lands both shares
    in the cathode cell; there the sum agrees to roundoff rather than
    bit-exactly, because the power-to-density scaling is applied to each
    share separately instead of to their sum.
    """

    rhs: ConservativeState1D
    anode_rhs: ConservativeState1D
    enabled: bool
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
    # Per-end CSDA deposition results ({0: primary, -1: twin}), present
    # whenever the solve ran; None keys mean no active beam.
    beam_deposition: dict | None = None
    # Per-end ``(probe, ray, circuit, ceiling)`` gap survival for the item-35
    # ledger tripwire; keyed only for ends with an active CSDA ray. The first
    # three are views of ONE number; ``ceiling`` is the most the circuit can
    # represent at this state, which is what separates a representability gap
    # from a divergence. See ``beam_gap_ledger_mismatch``.
    beam_gap_ledger: dict | None = None
    # Per-end ``(E_1 [eV], clamp)`` plateau edge of the multi-group closure
    # (``heating_anomalous_transport="plateau_multigroup"``), ``None`` under
    # every other value so an unarmed solve carries exactly the fields it
    # always did. ``E_1`` is solved per EXTRACTION -- it is a property of the
    # sheath drop and the launch cell's own Maxwellian, shared by that end's
    # deposition ray and both halves of a clumping split -- and ``clamp`` is
    # ``-1`` on the frames where the solve hit the inelastic floor and was
    # clamped to it (never silent: this is what the solver's clamp census
    # counts).
    beam_plateau_edge: dict | None = None
    # A2a: the electron current [A] this solve's deposition measured the anode
    # collecting DIRECTLY from the QL tail walkers -- the culled flux less
    # whatever the reversed-walker rider sent back, summed over the ends. It is
    # an OUTPUT here and an INPUT to the NEXT accepted step's circuit solve (the
    # deposition runs after the circuit within a step), so the caller commits it
    # only on acceptance. 0.0 whenever the tail cull is off, which is the value
    # the circuit's parameter defaults to.
    tail_anode_current_A: float = 0.0
    # ``anode_tail_booking = "emission_fraction"`` only (0.0 under
    # ``"lagged_current"``): the same collected tail current PER EMITTED
    # ELECTRON, ``c_tail = I_tail / (e G0)``, and the walker flux this solve's
    # deposition launched on the cathode side of the anode plane per emitted
    # electron, ``w_gap``. Both are INPUTS to the next accepted step's circuit
    # solve, committed on acceptance like ``tail_anode_current_A``.
    tail_anode_coefficient: float = 0.0
    anode_gap_walker_fraction: float = 0.0
    # The primary's net interception on its returns to the anode plane per
    # emitted electron, ``c_ret``, on the same terms (``"emission_fraction"``
    # only; 0.0 otherwise).
    primary_return_coefficient: float = 0.0


def anode_circuit_sample(state, derived, geometry, ion_mass_g, input_dict, end=0):
    """Return ``(I_i_a [A], Te_anode [eV], I_e_sat_a [A])`` for one anode.

    ``(None, None, None)`` where the geometry resolves no anode face or the
    mesh is fully open.

    The historical circuit takes ``I_i_a = 2*eta*I_i``, scaling the anode
    current straight off the *cathode* cell, which assumes both electrodes see the
    same plasma -- precisely what a resolved cathode-anode gap breaks.

    The current handed back is the same Bohm collection
    ``sources.anode_collection_rhs`` removes from the fluid, summed over both mesh
    faces with each face sampled on its own side. Computing it once and sharing it
    means the circuit and the fluid cannot disagree about the anode current, and it
    is why M5 must not add a second anode particle sink.

    The sheath temperature is collection-weighted across the two faces, matching
    how ``P_anode_e`` is apportioned. Resolving a *separate* sheath per face is
    a known open item.

    The third member is the ELECTRON SATURATION current the wires can draw:
    the electron random flux ``n * v_e_bar / 4`` on the wire area each face
    presents, ``eta * A_c``, summed over the two faces on their own sides
    exactly as the ion current is. It is the explicit form of what the sheath
    relation reaches implicitly through ``I_i_a * exp(Lambda_a)``, and here the
    two are the SAME number to roundoff: the ``I_i_a`` handed over IS the
    analytic ``e^(-1/2) n c_s`` Bohm collection on that same wire area, and the
    ratio of the two expressions is ``exp(Lambda_a)`` with no ``n`` and no
    ``T_e`` left in it, so the agreement survives the per-face sum even where
    the two faces sample different states. The explicit form is kept because it
    is the physical statement of the cap -- an electron random flux on the wire
    area the mesh presents -- rather than an ion current rescaled by a mass
    ratio. Both are evaluated on the same samples, so the two faces' densities
    and temperatures enter the electron cap the way they enter the ion current.
    """
    anode_faces = np.asarray(getattr(geometry, "anode_face_indices", ()), dtype=int)
    eta = float(input_dict.get("eta", 0.0))
    if anode_faces.size == 0 or eta <= 0.0:
        return None, None, None
    face = int(anode_faces[0] if end == 0 else anode_faces[-1])
    total = 0.0
    weighted_Te = 0.0
    saturation = 0.0
    for cell in (face - 1, face):
        wire_area = eta * float(geometry.plasma_area_cm2[cell])
        collected = (
            np.exp(-0.5)
            * state.n[cell]
            * ion_sound_speed(derived.Te[cell], ion_mass_g)
            * wire_area
        )
        total += collected
        weighted_Te += collected * float(derived.Te[cell])
        saturation += (
            0.25
            * state.n[cell]
            * electron_mean_speed(derived.Te[cell])
            * wire_area
        )
    if total <= 0.0:
        return None, None, None
    return total * qe_SI, weighted_Te / total, saturation * qe_SI


def cathode_sample_indices(geometry):
    """Return the ``(source, end)`` cells the cathode circuit samples.

    The cathode solve builds its ion current from the plasma against the cathode
    surface, so in resolved geometry it must read the *cathode-adjacent* cell --
    cell ``[0]`` there is the plasma-dead plenum, whose floor density and
    temperature would drive the circuit with garbage.

    A twin machine samples both cathodes; an end wall machine's ``end`` slot
    is the end wall cell. A column ending in a mirror face has no far
    electrode: its last cell sits against the symmetry plane, so the ``end``
    slot is ``None`` there.
    """
    cathode_cells = cathode_adjacent_cells(geometry)
    if not cathode_cells:
        raise ValueError("resolved geometry must define cathode-adjacent cells")
    source_index = int(cathode_cells[0])
    if len(cathode_cells) > 1:
        return source_index, int(cathode_cells[-1])
    if np.asarray(getattr(geometry, "mirror_face_indices", ())).size:
        return source_index, None
    return source_index, geometry.cells - 1


def cathode_circuit_alpha_sheath(
    state, derived, geometry, cathode_index, ion_mass_g, input_dict
):
    """Return the cathode sheath-edge factor ``n_se/n`` for the circuit.

    Unified sampling (A16): the circuit's cathode ion current is
    drawn at the SAME sheath-edge density the fluid boundary uses, so both call
    ``sources.electrode_sheath_alpha`` on the same cathode-adjacent cell (verified
    identical: ``beam_launch(geometry)[0]`` == the source cathode's live cell).
    Unconditional since the legacy volumetric-absorber stance, which sampled a
    flat ``exp(-1/2)`` here instead, was retired; see commit 1fc05c9. The
    anode is not sampled here -- its geometric mesh presheath stays flat
    ``exp(-1/2)``.
    """
    return electrode_sheath_alpha(
        nn=float(state.nn[cathode_index]),
        Te=float(derived.Te[cathode_index]),
        Ti=float(derived.Ti[cathode_index]),
        cell_length_cm=float(geometry.length_cm[cathode_index]),
        ion_mass_g=ion_mass_g,
        alpha_isat=float(input_dict.get("alpha_isat", math.exp(-0.5))),
        b_presheath_length=float(input_dict.get("b_presheath_length", 1.0)),
    )


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
        end=(
            None
            if end_index is None
            else _cell_state(end_index, state, derived, geometry)
        ),
        enabled=bool(input_flags.get("cathode_coupling", False)),
        twin_cathode=bool(input_flags.get("TwinCathode", False)),
        circuit=_circuit_placeholders(input_dict),
    )


def cathode_device_config(input_dict, input_flags, mu, ion_mass_g):
    """Build the existing cathode solver's static device configuration.

    The emitting face is the uniform disc of radius ``R_cath``: one surface
    temperature ``cathode_Ts_base_K`` and one work function ``phi_wf`` over
    the whole area, which both emits and collects the ion current.
    """
    R_cath = float(input_dict["R_cath"])
    return DeviceConfig(
        A_c=math.pi * R_cath**2,
        mu=mu,
        ion_mass_g=ion_mass_g,
        T_s=float(input_dict["cathode_Ts_base_K"]),
        phi_wf=float(input_dict["phi_wf"]),
        C_R=float(input_dict["C_R"]),
        R_comp=float(input_dict["R_comp"]),
        R_comp_partition=float(input_dict.get("R_comp_partition", 1.0)),
        R_mesh_ohm=float(input_dict.get("R_mesh_ohm", 0.0)),
        eta=float(input_dict["eta"]),
        L_cath=float(input_dict["L_cath"]),
        R_cath=R_cath,
    )


_SIGMA_SB_W_CM2_K4 = 5.670374419e-12
_KB_EV_PER_K = 8.617333262e-5

#: Chamber-wall temperature [K] the cathode surface radiates against, and the
#: floor of the evolving power-balance surface temperature. Negligible against
#: ``T_s^4``.
CATHODE_ENV_T_K = 300.0


def cathode_power_balance_terms_W(T_s_K, P_ion_W, I_eth_star_A, input_dict):
    """Return ``(P_heater, P_ion, P_rad, P_emis, P_cond)`` [W] for warming.

    The power-balance surface energy budget
    (M1b):

    - ``P_heater`` is pinned by the standby equilibrium at
      ``cathode_Ts_base_K`` -- open circuit means no net emission and no
      substrate gradient, so the heater exactly balances radiation there
      and is not a free parameter.
    - ``P_ion`` is the accepted solve's ion bombardment power.
    - ``P_rad`` is gray-body radiation from the emitting face.
    - ``P_emis`` is evaporative emission cooling: each *actually emitted*
      electron removes ``phi_wf + 2 k_B T_s`` (work function plus the mean
      thermal energy over the barrier). Pass the accepted solve's
      ``I_eth_star`` -- the space-charge-released current, not the
      Richardson ceiling -- in EVERY phase, the open circuit included: zero
      NET current is not zero emission, and the electrons that clear the
      virtual cathode leave whether or not the loop carries their charge
      away. What the surface does NOT get back here is the energy of the
      plasma electrons it collects in return; that deposit is not modelled,
      and while it is off the books this term is the whole of the face's
      electron-channel budget.
    - ``P_cond`` is conduction from the emitting skin layer into the
      heater-held substrate, ``G_cond * (T_s - T_base)`` -- the
      "heater maintains the lower end" restoring term. It vanishes at
      standby by construction, so the heater pinning is unchanged.
      **Without it the balance is unstable at the LAPD operating point**
      (measured 2026-07-20, `es1_nx120_pb_demo.h5`): the bombardment
      feedback gain d(P_ion)/dT through the emission loop exceeds the
      ~230 W/K radiation+emission stiffness, and the current runs to
      12.9 kA before the sheath saturates the loop.

    The net rate is ``(P_heater + P_ion - P_rad - P_emis - P_cond) /
    C_th``; the caller owns the time discretization.
    """
    area = math.pi * float(input_dict["R_cath"]) ** 2
    eps = float(input_dict.get("cathode_emissivity", 0.7))
    T_env = CATHODE_ENV_T_K
    T_base = float(input_dict["cathode_Ts_base_K"])

    def _rad(T):
        return eps * _SIGMA_SB_W_CM2_K4 * float(area) * (T**4 - T_env**4)

    P_emis = max(float(I_eth_star_A), 0.0) * (
        float(input_dict["phi_wf"]) + 2.0 * _KB_EV_PER_K * float(T_s_K)
    )
    P_cond = float(input_dict.get("cathode_conduction_W_per_K", 0.0)) * (
        float(T_s_K) - T_base
    )
    return (
        _rad(T_base),
        max(float(P_ion_W), 0.0),
        _rad(float(T_s_K)),
        P_emis,
        P_cond,
    )


def spitzer_sigma_par_ohm_cm(Te_eV, n_cm3):
    """Parallel Spitzer conductivity [Ohm^-1 cm^-1], as the cathode solver's.

    Matches the internal ``sigma_par`` of the cathode solve under its
    ``"nrl_ei"`` Coulomb logarithm, so the ohmic gap deposition and the
    solve's gap resistance read one conductivity. Elementwise in
    ``(Te_eV, n_cm3)``.

    The Coulomb logarithm is the state-dependent electron-ion log at the local
    ``(Te, n)``, floored at ``LN_LAMBDA_MIN`` -- the SAME
    ``c_log(..., kind="ei")`` convention the conduction and exchange terms
    use, so the solver carries one lnLambda repo-wide.
    """
    Te = np.asarray(Te_eV, dtype=float)
    ln_lambda = np.maximum(
        c_log(Te, np.asarray(n_cm3, dtype=float), kind="ei"), LN_LAMBDA_MIN
    )
    return (1.96 / (1.03e-2 * ln_lambda)) * Te**1.5


#: The drive formulations ``cathode_solver_model`` dispatches on. EXPORTED
#: because the validator's refusal message and the solver's own dispatch read
#: one domain; a domain stated twice is a domain that drifts.
CATHODE_SOLVER_MODELS = ("current_driven", "prescribed_measured")


def validate_cathode_solver_model(input_dict, input_flags):
    """Validate and return the ``cathode_solver_model`` selection.

    ``"prescribed_measured"`` inherits the current-driven path's two
    structural requirements unchanged: it drives ONE cathode (the prescribed
    beam system, like the current-driven one, has no twin), and its FOOT is a
    real current-driven discharge, so the loop inductance the foot integrates
    must still be positive. What it does not inherit is the emission and bank
    calibration's meaning past the hand-off; that is stated at the key, and
    the trace keys themselves are resolved in ``core/prescribed_drive.py``.
    """
    model = str(input_dict.get("cathode_solver_model", "current_driven"))
    if model not in CATHODE_SOLVER_MODELS:
        raise ValueError(
            "cathode_solver_model must be 'current_driven' or "
            f"'prescribed_measured' (got {model!r})"
        )
    coupling = bool(input_flags.get("cathode_coupling", False))
    if coupling and bool(input_flags.get("TwinCathode", False)):
        raise ValueError(
            f"cathode_solver_model={model!r} does not support TwinCathode"
        )
    if coupling and float(input_dict.get("L_parasitic_H", 0.0)) <= 0.0:
        raise ValueError(
            f"cathode_solver_model={model!r} requires L_parasitic_H > 0"
        )
    return model


def anode_tail_booking_coefficients(
    tail_anode_current_A,
    emitted_current_A,
    gap_born_flux_per_s,
    eta,
    beta,
    primary_return_flux_per_s=0.0,
):
    """Return ``(c_tail, w_gap, c_ret)`` for ``anode_tail_booking="emission_fraction"``.

    ``c_tail = I_tail / I_emit`` is the collected tail-walker current per
    emitted electron, ``w_gap = e * gap_born_flux / I_emit`` the walker flux
    launched on the cathode side of the anode plane per emitted electron, and
    ``c_ret = e * primary_return_flux / I_emit`` the primary's net
    interception on its returns to the plane per emitted electron, all from
    one deposition; zeros when nothing is emitted. The circuit books at most
    ``eta * beta * (1 - w_gap) + c_ret + c_tail`` of the emission as
    collected directly by the anode, which cannot exceed the emission: raises
    ``RuntimeError`` when it does, or when ``w_gap`` leaves ``[0, 1]``.
    """
    emitted = float(emitted_current_A)
    if not emitted > 0.0:
        return 0.0, 0.0, 0.0
    c_tail = float(tail_anode_current_A) / emitted
    w_gap = qe_SI * float(gap_born_flux_per_s) / emitted
    c_ret = qe_SI * float(primary_return_flux_per_s) / emitted
    booked = float(eta) * float(beta) * (1.0 - w_gap) + c_ret + c_tail
    if not (0.0 <= w_gap <= 1.0 and booked <= 1.0):
        raise RuntimeError(
            "the anode's direct fast-electron collection exceeds the "
            "emission: eta*beta*(1 - w_gap) + c_ret + c_tail = "
            f"{booked!r} (eta={float(eta)!r}, beta={float(beta)!r}, "
            f"w_gap={w_gap!r}, c_ret={c_ret!r}, c_tail={c_tail!r}; emitted "
            f"{emitted!r} A, collected tail {float(tail_anode_current_A)!r} "
            f"A, gap-born walker flux {float(gap_born_flux_per_s)!r} /s, "
            f"primary net return interception "
            f"{float(primary_return_flux_per_s)!r} /s); the booking requires "
            "w_gap in [0, 1] and a total of at most 1"
        )
    return c_tail, w_gap, c_ret


def idriven_result_evaluator(
    state,
    floors,
    ion_mass_g,
    mu,
    geometry,
    input_dict,
    input_flags,
    beam_cross_prev,
    T_s_override_K=None,
    phi_wf_override_eV=None,
    tail_anode_current_prev_A=0.0,
    tail_anode_coefficient_prev=0.0,
    anode_gap_walker_fraction_prev=0.0,
    primary_return_coefficient_prev=0.0,
    cathode_ion_removal_prev_W=0.0,
):
    """Return an ``I [A] -> SolverResult`` evaluator at this frozen state.

    Builds the same device config (T_s and phi_wf substitution, anode sample)
    as the per-step dispatch, via the same helpers, so its consumers and the
    dispatched solve cannot disagree. Two consumers: the circuit advance
    (through ``idriven_vdis_evaluator``) and the power-balance warming update,
    which needs *accepted-state* P_cathode_i / I_eth_star -- the RHS cache
    ``_cathode_solve`` holds the last internal-stage solve of the step, whose
    P_cathode_i differs from the accepted-state value at the same frozen
    current (the stage state sits on the other side of the knee).

    The anode's fast-electron booking follows ``anode_tail_booking``. Under
    ``"emission_fraction"`` the evaluator books the three lagged coefficients
    the caller passes, exactly as the dispatched solve does. Under
    ``"lagged_current"`` it books ``tail_anode_current_prev_A``, whose
    default is 0.0, and the solver's two callers (the circuit advance and the
    accepted-state re-solve) do not pass it: under that booking both evaluate
    the anode with NO tail current, while the dispatched solve reads the
    lagged one.

    ``cathode_ion_removal_prev_W`` [W] is the energy the fluid removes with
    the ions it delivers to the cathode face, handed to the circuit as
    ``cathode_ion_removal_W`` (see
    :func:`~cablp.cathode.circuit_idriven.solve_idriven`).
    """
    derived = derive_state(state, floors=floors, ion_mass_g=ion_mass_g)
    anode_A, anode_Te, anode_e_sat = anode_circuit_sample(
        state, derived, geometry, ion_mass_g, input_dict, end=0
    )
    if T_s_override_K is not None:
        input_dict = {**input_dict, "cathode_Ts_base_K": float(T_s_override_K)}
    if phi_wf_override_eV is not None:
        input_dict = {**input_dict, "phi_wf": float(phi_wf_override_eV)}
    device_config = cathode_device_config(
        input_dict, input_flags, mu, ion_mass_g
    )
    idx = beam_launch(geometry, end=0)[0]
    beam_cross_prev = np.asarray(beam_cross_prev, dtype=float)
    # MEAN densities, matching the dispatched solve in
    # solve_cathode_boundary: this ONE n_e is spent on both the bilinear beam
    # coupling length and the linear Bohm ion current, and only the former may
    # be concentrated.
    plasma = PlasmaState(
        T_e=float(derived.Te[idx]),
        n_e=float(state.n[idx]),
        n_n=float(state.nn[idx]),
        sigma_b=float(beam_cross_prev[idx]),
    )
    cap = float(input_dict.get("cathode_phi_c_cap_V", 1000.0))
    alpha_sheath = cathode_circuit_alpha_sheath(
        state, derived, geometry, idx, ion_mass_g, input_dict
    )

    if input_dict.get("anode_tail_booking") == "emission_fraction":
        booking_kwargs = dict(
            anode_tail_booking="emission_fraction",
            tail_anode_coefficient=float(tail_anode_coefficient_prev),
            anode_gap_walker_fraction=float(anode_gap_walker_fraction_prev),
            primary_return_coefficient=float(
                primary_return_coefficient_prev
            ),
        )
    else:
        booking_kwargs = dict(
            tail_anode_current_A=float(tail_anode_current_prev_A),
        )

    def solve_at(I_A, anode_balance_probe=False):
        return solve_idriven(
            device_config,
            plasma,
            I_tot_A=max(float(I_A), 0.0),
            anode_current_A=anode_A,
            anode_T_e=anode_Te,
            anode_electron_saturation_A=anode_e_sat,
            schottky=True,
            phi_c_cap_V=cap,
            alpha_sheath=alpha_sheath,
            **booking_kwargs,
            anode_balance_probe=bool(anode_balance_probe),
            cathode_ion_removal_W=float(cathode_ion_removal_prev_W),
        )

    return solve_at


def idriven_vdis_evaluator(
    state,
    floors,
    ion_mass_g,
    mu,
    geometry,
    input_dict,
    input_flags,
    beam_cross_prev,
    T_s_override_K=None,
    phi_wf_override_eV=None,
    tail_anode_current_prev_A=0.0,
    tail_anode_coefficient_prev=0.0,
    anode_gap_walker_fraction_prev=0.0,
    primary_return_coefficient_prev=0.0,
    cathode_ion_removal_prev_W=0.0,
):
    """Return a ``V_dis(I) [V]`` evaluator at this frozen plasma state.

    Used by the current-driven circuit advance: each implicit stage
    root-finds the loop current against the monotone device voltage, so it
    needs many cheap sheath evaluations at the *accepted* end-of-step state
    with only I varying. Thin wrapper over ``idriven_result_evaluator``.
    """
    solve_at = idriven_result_evaluator(
        state=state,
        floors=floors,
        ion_mass_g=ion_mass_g,
        mu=mu,
        geometry=geometry,
        input_dict=input_dict,
        input_flags=input_flags,
        beam_cross_prev=beam_cross_prev,
        T_s_override_K=T_s_override_K,
        phi_wf_override_eV=phi_wf_override_eV,
        tail_anode_current_prev_A=tail_anode_current_prev_A,
        tail_anode_coefficient_prev=tail_anode_coefficient_prev,
        anode_gap_walker_fraction_prev=anode_gap_walker_fraction_prev,
        primary_return_coefficient_prev=primary_return_coefficient_prev,
        cathode_ion_removal_prev_W=cathode_ion_removal_prev_W,
    )

    # Internal series drop on the plasma side of the V_dis probe (R5 ES1 tuning
    # pass, 2026-07-26). R_comp is split by the probe: R_external = x*R_comp
    # (bank side, in V_dis) and R_internal = (1-x)*R_comp (probe->plasma), plus a
    # separate anode-mesh R_mesh_ohm also on the plasma side. The circuit
    # integrates the DEVICE voltage V_b + I*(R_internal + R_mesh), while V_dis =
    # V_bank - I*R_external (see advance_circuit's R_comp_ohm = x*R_comp).
    #
    # CORRECTION (2026-08-03): this comment used to say the internal drop
    # "lowers the current, which RAISES V_dis". That is FALSE for the
    # (1-x)*R_comp part -- x CANCELS IDENTICALLY from the loop equation.
    # advance_circuit_current_driven integrates
    #     f(I) = (V_src - I*x*R_comp - vdis_of_I(I)) / L
    #          = (V_src - I*R_comp - V_b(I) - I*R_mesh) / L,
    # so the current sees only the TOTAL R_comp plus R_mesh; x is gone. What x
    # changes is the REPORTED V_dis, relabelling the same drop between the
    # external and internal books (dV_dis/dx = -I*R_comp). The claim IS true of
    # R_mesh, which is genuinely additional series resistance -- so real
    # internal resistance goes in R_mesh_ohm, never in the partition. Defaults
    # (x=1, R_mesh=0) give R_internal_total = 0 -> device voltage = V_b,
    # bit-exact.
    x = float(input_dict.get("R_comp_partition", 1.0))
    R_comp = float(input_dict.get("R_comp", 0.0))
    R_internal_total = (1.0 - x) * R_comp + float(
        input_dict.get("R_mesh_ohm", 0.0)
    )

    def vdis(I_A, anode_balance_probe=False):
        return (
            solve_at(I_A, anode_balance_probe=anode_balance_probe).V_b
            + I_A * R_internal_total
        )

    return vdis


def advance_circuit_current_driven(
    I_prev_A,
    dt_s,
    V_src_V,
    R_comp_ohm,
    L_H,
    vdis_of_I,
    C_bank_F=None,
    V_cap_prev_V=None,
    vdis_bracket_probe=None,
):
    """TR-BDF2 advance of the loop current against a monotone V_dis(I).

    Integrates ``dI/dt = (V_src - I*R - V_dis(I)) / L`` over one accepted
    step. The stage residual ``g(I) = I - rhs - a*f(I)`` has
    ``g' = 1 + a*(R + dV_dis/dI)/L >= 1`` because the current-driven device
    voltage is monotone in I, so each stage is a bracketed scalar brentq --
    unconditionally well-posed however steep V_dis(I) gets. This is the
    load-bearing design decision (revised 2026-07-20): a
    frozen-V_dis explicit step needs ``dV/dI < 2L/dt ~ 22 mOhm`` at
    production dt, and the measured device slope near the emission ceiling
    is 0.2 Ohm-0.75 MOhm -- explicit would sawtooth exactly where this
    machine operates. TR-BDF2 because the RLC gate demands 2nd order and
    TR alone would ring against the near-vertical branch (L-stability, the
    same argument as the heat-conduction scheme choice).

    ``I >= 0`` is enforced per stage (the plasma-diode stand-in): a stage
    whose unconstrained root is negative clamps to 0. Each stage probes its
    bracket's lower endpoint ``I = 0`` through ``vdis_bracket_probe``
    (``None``: ``vdis_of_I`` itself), the evaluation at which the sheath solve
    keeps a floored anode balance instead of refusing it; every other
    evaluation goes through ``vdis_of_I``.
    ``V_src_V`` is held constant over the step (drive: bank/capacitor
    voltage; tail: 0); the capacitor, when present, is frozen for the I
    stages (droop ~2e-4 V/step) and then advanced trapezoidally. Returns
    ``(I_new_A, V_cap_new_V_or_None, V_dis_step_V)``.

    ``V_dis_step_V`` is the step-integrated discharge voltage -- the
    *inductor's view*, from the integrated loop equation over the step:
    ``<V_dis> = V_src - R*<I> - L*(I_new - I_prev)/dt`` with ``<I>`` the
    piecewise-trapezoidal average through the TR stage. This is the honest
    smooth V_dis trace (chatter diagnosis, 2026-07-21): the per-solve
    ``V_b`` inherits the boundary cell's per-step Te wobble through the
    knee, but the circuit only ever integrates V_dis against L, so the
    step average is the physically meaningful instantaneous voltage.
    """
    L = float(L_H)
    if L <= 0.0:
        raise ValueError(f"L_H must be positive (got {L_H})")
    dt = float(dt_s)
    I_n = max(float(I_prev_A), 0.0)

    vdis_probe = vdis_of_I if vdis_bracket_probe is None else vdis_bracket_probe

    def f(I, vdis=vdis_of_I):
        return (float(V_src_V) - I * float(R_comp_ohm) - vdis(I)) / L

    def stage_solve(rhs_const, a_coef):
        def g(I):
            return I - rhs_const - a_coef * f(I)

        # The bracket's lower endpoint, evaluated once through the probe and
        # handed back to brentq at that endpoint rather than re-solved there.
        g_lo = 0.0 - rhs_const - a_coef * f(0.0, vdis_probe)

        def g_bracket(I):
            return g_lo if I == 0.0 else g(I)

        if g_lo >= 0.0:
            return 0.0
        hi = max(I_n, 1.0)
        for _ in range(200):
            hi *= 2.0
            if g(hi) > 0.0:
                break
        else:
            raise RuntimeError(
                "circuit stage bracket did not close "
                f"(I_n={I_n:.6g} A, rhs={rhs_const:.6g})"
            )
        return brentq(
            g_bracket, 0.0, hi, xtol=1e-10, rtol=1e-12, full_output=False
        )

    gamma = 2.0 - math.sqrt(2.0)
    f_n = f(I_n)
    # TR stage to t + gamma*dt: I_g = I_n + (gamma*dt/2)*(f_n + f(I_g))
    a1 = 0.5 * gamma * dt
    I_g = stage_solve(I_n + a1 * f_n, a1)
    # BDF2 stage to t + dt:
    #   I_1 = I_g/(gamma*(2-gamma)) - I_n*(1-gamma)^2/(gamma*(2-gamma))
    #         + dt*(1-gamma)/(2-gamma) * f(I_1)
    denom = gamma * (2.0 - gamma)
    a2 = dt * (1.0 - gamma) / (2.0 - gamma)
    I_new = stage_solve(
        I_g / denom - I_n * (1.0 - gamma) ** 2 / denom, a2
    )

    # Step-integrated V_dis from the loop identity (see docstring). <I> is
    # the trapezoidal average through the internal TR stage -- second-order
    # consistent with the current trajectory the scheme just committed to.
    I_avg = 0.5 * (gamma * (I_n + I_g) + (1.0 - gamma) * (I_g + I_new))
    V_dis_step = (
        float(V_src_V)
        - float(R_comp_ohm) * I_avg
        - L * (I_new - I_n) / dt
    )

    V_cap_new = None
    if C_bank_F is not None and float(C_bank_F) > 0.0:
        V_cap_prev = (
            float(V_cap_prev_V) if V_cap_prev_V is not None else 0.0
        )
        V_cap_new = max(
            V_cap_prev - dt * 0.5 * (I_n + I_new) / float(C_bank_F), 0.0
        )
    return I_new, V_cap_new, V_dis_step


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
    x0=None,
    x0_twin=None,
    floating=False,
    T_s_override_K=None,
    phi_wf_override_eV=None,
    circuit_I_loop_A=0.0,
    tail_anode_current_prev_A=0.0,
    prescribed_drive=None,
    tail_anode_coefficient_prev=0.0,
    anode_gap_walker_fraction_prev=0.0,
    primary_return_coefficient_prev=0.0,
    cathode_ion_removal_prev_W=0.0,
):
    """Call the cathode/beam solver and return raw diagnostics only.

    ``prescribed_drive`` is the measured ``(I [A], V_dis [V])`` pair when
    ``cathode_solver_model = "prescribed_measured"`` has taken over, and
    ``None`` otherwise -- which is every step of every other configuration,
    and every step of a prescribed run before its hand-off. It is the presence
    gate on the whole prescribed branch: without it this function cannot reach
    ``solve_beam_system_prescribed`` at all, so the off path is the historical
    dispatch bit for bit. The caller resolves it, because the trace lives on
    the model clock and this function is not given a time.

    ``anode_tail_booking`` (read from ``input_dict``) selects which lagged
    inputs the circuit reads: ``tail_anode_current_prev_A`` under
    ``"lagged_current"``, or ``tail_anode_coefficient_prev``,
    ``anode_gap_walker_fraction_prev`` and
    ``primary_return_coefficient_prev`` under ``"emission_fraction"``, whose
    successors this solve's deposition produces (see
    :func:`anode_tail_booking_coefficients`).

    ``cathode_ion_removal_prev_W`` [W] is the energy the fluid removed with
    the ions it delivered to the cathode face over the last accepted step,
    per second; the circuit books it as the cathode ion member of the
    plasma-thermal book and adds it to the surface's ion power.
    """
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
                "floating": bool(floating),
                "source_index": boundary.source.index,
                "end_index": (
                    None if boundary.end is None else boundary.end.index
                ),
                "twin_cathode": boundary.twin_cathode,
                "circuit": dict(boundary.circuit),
            },
        )

    derived = derive_state(state, floors=floors, ion_mass_g=ion_mass_g)
    anode_source = anode_circuit_sample(
        state, derived, geometry, ion_mass_g, input_dict, end=0
    )
    if T_s_override_K is not None:
        # The power balance: substitute the evolving surface temperature at
        # the single point the emission reads the surface temperature from.
        # That point is cathode_Ts_base_K, the initial condition this
        # evolving value started from -- so the substitution reaches every
        # emission consumer and no other. The conduction term's own read of
        # cathode_Ts_base_K is the SUBSTRATE temperature and must NOT see
        # this value; it is served from the solver's own input_dict
        # (_surface_effective_input_dict), never from this local copy.
        input_dict = {**input_dict, "cathode_Ts_base_K": float(T_s_override_K)}
    if phi_wf_override_eV is not None:
        # The ads/des surface state: substitute the evolving effective work
        # function at the single point every phi_wf consumer reads from --
        # Richardson/DeviceConfig, the Schottky reference barrier, and (via
        # the same dict in the solver) the power-balance emission-cooling
        # term. One shared constant, changed in one place.
        input_dict = {**input_dict, "phi_wf": float(phi_wf_override_eV)}
    device_config = cathode_device_config(
        input_dict, input_flags, mu, ion_mass_g
    )
    solver_model = validate_cathode_solver_model(input_dict, input_flags)
    booking = str(input_dict.get("anode_tail_booking", "lagged_current"))
    if booking not in ANODE_TAIL_BOOKINGS:
        raise ValueError(
            "anode_tail_booking must be 'lagged_current' or "
            f"'emission_fraction' (got {booking!r})"
        )
    # The circuit's lagged fast-electron inputs. Presence-gated: under the
    # default only the absolute tail current reaches the circuit, exactly as
    # before the selector existed.
    if booking == "emission_fraction":
        circuit_tail_kwargs = dict(
            tail_anode_current_A=0.0,
            anode_tail_booking=booking,
            tail_anode_coefficient=float(tail_anode_coefficient_prev),
            anode_gap_walker_fraction=float(anode_gap_walker_fraction_prev),
            primary_return_coefficient=float(primary_return_coefficient_prev),
        )
    else:
        circuit_tail_kwargs = dict(
            tail_anode_current_A=float(tail_anode_current_prev_A),
        )
    beam_cross_prev = np.asarray(beam_cross_prev, dtype=float)
    if beam_cross_prev.shape != (geometry.cells,):
        raise ValueError(
            "beam_cross_prev must have shape "
            f"({geometry.cells},), got {beam_cross_prev.shape}"
        )
    if prescribed_drive is not None and not floating:
        # PRESCRIBED MEASURED DRIVE. Both loop quantities are measurements
        # this step, so there is no root to find in the current and no
        # emission ceiling to consult: the sheath follows from the loop
        # relation (see cablp/cathode/circuit_prescribed.py). Floating phases
        # are excluded on purpose -- an open circuit carries no measured
        # discharge, and the caller withdraws the drive there -- so the
        # afterglow keeps the historical open-circuit solve unchanged.
        beam_result = solve_beam_system_prescribed(
            config=device_config,
            Te=derived.Te,
            ne=state.n,
            nn=state.nn,
            beam_cross_prev=beam_cross_prev,
            plasma_cross=geometry.plasma_area_cm2,
            I_ion=I_ion,
            # Floored at zero on the same convention the current-driven
            # branch floors its loop current: the device carries what it can,
            # never a backwards current, and a trace sample below zero is
            # baseline noise rather than a reverse discharge.
            I_tot_A=max(float(prescribed_drive.I_A), 0.0),
            V_dis_V=float(prescribed_drive.V_dis_V),
            cathode_index=beam_launch(geometry, end=0)[0],
            anode_current_A=anode_source[0],
            anode_T_e=anode_source[1],
            anode_electron_saturation_A=anode_source[2],
            alpha_sheath=cathode_circuit_alpha_sheath(
                state, derived, geometry, beam_launch(geometry, end=0)[0],
                ion_mass_g, input_dict,
            ),
            phi_c_cap_V=float(input_dict.get("cathode_phi_c_cap_V", 1000.0)),
            # The prescribed drive books the lagged absolute tail current
            # only; ``"emission_fraction"`` is refused with it at construction.
            tail_anode_current_A=float(tail_anode_current_prev_A),
            cathode_ion_removal_W=float(cathode_ion_removal_prev_W),
        )
    else:
        # The circuit is explicit solver state: no inductive fold, no
        # warm start -- the solve is a well-posed evaluation at the frozen
        # loop current.
        #
        # OPEN CIRCUIT (``floating``) is the SAME solve read at I_tot = 0.
        # That is what an open circuit is: the loop carries no current, and
        # the surface finds the potential at which its released emission plus
        # the ion current is exactly returned by collected plasma electrons.
        # Solving it here rather than through a separate open-circuit root
        # keeps the current balance exact (the root is the same monotone
        # device relation), keeps the emission and virtual-cathode physics,
        # and populates every ``P_*_thermal``/``P_*_phi`` electrode field --
        # so the electrode rows are CONTINUOUS across the hand-off, the same
        # formulas evaluated at zero current.
        I_tot_A = 0.0 if floating else max(float(circuit_I_loop_A), 0.0)
        beam_result = solve_beam_system_idriven(
            config=device_config,
            Te=derived.Te,
            ne=state.n,
            nn=state.nn,
            beam_cross_prev=beam_cross_prev,
            plasma_cross=geometry.plasma_area_cm2,
            I_ion=I_ion,
            I_tot_A=I_tot_A,
            cathode_index=beam_launch(geometry, end=0)[0],
            anode_current_A=anode_source[0],
            anode_T_e=anode_source[1],
            anode_electron_saturation_A=anode_source[2],
            alpha_sheath=cathode_circuit_alpha_sheath(
                state, derived, geometry, beam_launch(geometry, end=0)[0],
                ion_mass_g, input_dict,
            ),
            schottky=True,
            phi_c_cap_V=float(input_dict.get("cathode_phi_c_cap_V", 1000.0)),
            **circuit_tail_kwargs,
            cathode_ion_removal_W=float(cathode_ion_removal_prev_W),
        )
    (
        beam_deposition,
        beam_gap_ledger,
        beam_plateau_edge,
    ) = _csda_beam_deposition(
        beam_result=beam_result,
        state=state,
        derived=derived,
        geometry=geometry,
        device_config=device_config,
        input_dict=input_dict,
        I_ion=I_ion,
        twin=boundary.twin_cathode,
    )
    # The primary beam's per-cell mean free path, a saved diagnostic, at the
    # attenuation cross section the call above has just written into the
    # launch cell -- the one fed back to the next sheath solve -- so the
    # profile reports the quantity the solve uses. At the launch potential
    # ``phi_c``, like the deposition ray, so the two are one beam energy.
    launch_0 = beam_launch(geometry, end=0)[0]
    if beam_result.beam_cross[launch_0] != 0.0:
        phi_c_0 = beam_result.result.phi_c
        sigma_atten = beam_result.beam_atten_cross[launch_0]
        for j in range(geometry.cells):
            beam_result.l_b_profile[j] = compute_l_b(
                phi_c_0, derived.Te[j], state.n[j], state.nn[j], sigma_atten,
            )
    # A2a: what the anode actually COLLECTED from the tail this solve --
    # the culled walkers less the ones the rider sent back, converted from a
    # walker flux to a current. Summed over the ends with an active ray,
    # because the circuit sees one anode current. This is the value the NEXT
    # accepted step's sheath solve reads; nothing in THIS step consumes it.
    tail_anode_current_A = 0.0
    if beam_deposition is not None:
        for _dep in beam_deposition.values():
            if _dep is None:
                continue
            tail_anode_current_A += qe_SI * (
                float(_dep.tail_anode_culled_flux_per_s)
                - float(_dep.tail_anode_returned_flux_per_s)
            )
    # The same collection per emitted electron, with the walker flux born on
    # the cathode side of the anode plane, for the next solve's
    # ``"emission_fraction"`` booking; the coefficient function asserts that
    # the booked direct collection does not exceed the emission.
    tail_anode_coefficient = 0.0
    anode_gap_walker_fraction = 0.0
    primary_return_coefficient = 0.0
    if booking == "emission_fraction" and beam_deposition is not None:
        emitted_A = 0.0
        gap_born = 0.0
        returned = 0.0
        for _end, _dep in beam_deposition.items():
            if _dep is None:
                continue
            _res = (
                beam_result.result if _end == 0 else beam_result.result_twin
            )
            emitted_A += beam_launched_current_A(_res)
            gap_born += float(_dep.tail_gap_born_flux_per_s)
            returned += float(_dep.primary_net_return_flux_per_s)
        (
            tail_anode_coefficient,
            anode_gap_walker_fraction,
            primary_return_coefficient,
        ) = anode_tail_booking_coefficients(
            tail_anode_current_A,
            emitted_A,
            gap_born,
            device_config.eta,
            beam_result.result.beam_bypass_fraction,
            returned,
        )
    return CathodeSolve1D(
        boundary=boundary,
        beam_result=beam_result,
        device_config=device_config,
        x0_next=beam_result.x0_next,
        x0_twin_next=beam_result.x0_twin_next,
        metadata={
            "enabled": True,
            "floating": bool(floating),
            "source_index": boundary.source.index,
            "end_index": (
                None if boundary.end is None else boundary.end.index
            ),
            "twin_cathode": boundary.twin_cathode,
            "circuit": dict(boundary.circuit),
            "cathode_solver_model": solver_model,
            "result": _solver_result_metadata(beam_result.result),
            "result_twin": _solver_result_metadata(beam_result.result_twin),
        },
        beam_deposition=beam_deposition,
        beam_gap_ledger=beam_gap_ledger,
        beam_plateau_edge=beam_plateau_edge,
        tail_anode_current_A=tail_anode_current_A,
        tail_anode_coefficient=tail_anode_coefficient,
        anode_gap_walker_fraction=anode_gap_walker_fraction,
        primary_return_coefficient=primary_return_coefficient,
    )


def _sum_beam_deposition(a, b):
    """Sum two CSDA beam rays (fractional-coverage: gap + clump).

    Per-cell deposition arrays add; the beam is energy-limited per ray so the
    combined totals stay bounded by ``Gamma0*E0``. Scalar exit diagnostics are
    flux-combined (unused downstream, kept coherent).
    """
    tf = float(a.transmitted_flux) + float(b.transmitted_flux)
    if tf > 0.0:
        te = (a.transmitted_flux * a.transmitted_energy_eV
              + b.transmitted_flux * b.transmitted_energy_eV) / tf
    else:
        te = 0.0
    return BeamDepositionResult(
        ionization_events=a.ionization_events + b.ionization_events,
        excitation_events=a.excitation_events + b.excitation_events,
        plasma_heating_erg_s=a.plasma_heating_erg_s + b.plasma_heating_erg_s,
        radiated_erg_s=a.radiated_erg_s + b.radiated_erg_s,
        ionization_cost_erg_s=a.ionization_cost_erg_s + b.ionization_cost_erg_s,
        transmitted_flux=tf,
        transmitted_energy_eV=te,
        anode_intercepted_erg_s=(float(a.anode_intercepted_erg_s)
                                 + float(b.anode_intercepted_erg_s)),
        # End ledger (WP-D): both rays leave through the same two ends, so
        # the escaping powers add like every other per-ray bank.
        end_loss_low_erg_s=(float(a.end_loss_low_erg_s)
                            + float(b.end_loss_low_erg_s)),
        end_loss_high_erg_s=(float(a.end_loss_high_erg_s)
                             + float(b.end_loss_high_erg_s)),
        end_loss_transmitted_erg_s=(float(a.end_loss_transmitted_erg_s)
                                    + float(b.end_loss_transmitted_erg_s)),
        # The walked terminal population that reached an end: both rays'
        # escapes land on the same surfaces, so the fluxes add like the
        # powers above.
        terminal_escape_flux_per_s=(float(a.terminal_escape_flux_per_s)
                                    + float(b.terminal_escape_flux_per_s)),
        # Tail end ledger (WP-E): same argument as the WP-D pair above -- both
        # rays' QL tails leave through the same two ends, so the escaping
        # powers add.
        end_loss_tail_low_erg_s=(float(a.end_loss_tail_low_erg_s)
                                 + float(b.end_loss_tail_low_erg_s)),
        end_loss_tail_high_erg_s=(float(a.end_loss_tail_high_erg_s)
                                  + float(b.end_loss_tail_high_erg_s)),
        E_entry_eV=np.maximum(a.E_entry_eV, b.E_entry_eV),
        # Diagnostic heating splits add like the lumped bank they partition.
        heating_coulomb_erg_s=(a.heating_coulomb_erg_s
                               + b.heating_coulomb_erg_s),
        heating_anomalous_erg_s=(a.heating_anomalous_erg_s
                                 + b.heating_anomalous_erg_s),
        heating_secondary_erg_s=(a.heating_secondary_erg_s
                                 + b.heating_secondary_erg_s),
        heating_terminal_erg_s=(a.heating_terminal_erg_s
                                + b.heating_terminal_erg_s),
        # K6 tail splits add like the banks they partition.
        ionization_events_tail=(a.ionization_events_tail
                                + b.ionization_events_tail),
        excitation_events_tail=(a.excitation_events_tail
                                + b.excitation_events_tail),
        ionization_cost_tail_erg_s=(a.ionization_cost_tail_erg_s
                                    + b.ionization_cost_tail_erg_s),
        radiated_tail_erg_s=(a.radiated_tail_erg_s + b.radiated_tail_erg_s),
        # K7b exposure ledger: both rays launch their own tail power at the
        # SAME E_tail (the split varies nn alone), so they land in the same
        # band and the powers add like every other per-ray bank.
        tail_power_erg_s=(float(a.tail_power_erg_s)
                          + float(b.tail_power_erg_s)),
        tail_sub_threshold_power_erg_s=(
            float(a.tail_sub_threshold_power_erg_s)
            + float(b.tail_sub_threshold_power_erg_s)
        ),
        tail_above_bar_power_erg_s=(
            float(a.tail_above_bar_power_erg_s)
            + float(b.tail_above_bar_power_erg_s)
        ),
        # Multi-group wave/bulk share: both rays split the SAME extraction's
        # plateau at the same edge (the split varies nn alone), so their
        # locally-banked wave powers add like every other per-ray bank.
        plateau_wave_power_erg_s=(
            float(a.plateau_wave_power_erg_s)
            + float(b.plateau_wave_power_erg_s)
        ),
        # A2a: both rays meet the same mesh, so their tail-cull rows add like
        # every other per-ray bank. The clumping split's two rays each cull
        # their own share of one tail, so the sum is the walk's own total.
        tail_anode_culled_flux_per_s=(
            float(a.tail_anode_culled_flux_per_s)
            + float(b.tail_anode_culled_flux_per_s)
        ),
        tail_anode_culled_erg_s=(
            float(a.tail_anode_culled_erg_s)
            + float(b.tail_anode_culled_erg_s)
        ),
        tail_anode_returned_flux_per_s=(
            float(a.tail_anode_returned_flux_per_s)
            + float(b.tail_anode_returned_flux_per_s)
        ),
        tail_anode_returned_erg_s=(
            float(a.tail_anode_returned_erg_s)
            + float(b.tail_anode_returned_erg_s)
        ),
        tail_anode_sheath_reflected_flux_per_s=(
            float(a.tail_anode_sheath_reflected_flux_per_s)
            + float(b.tail_anode_sheath_reflected_flux_per_s)
        ),
        tail_anode_sheath_reflected_erg_s=(
            float(a.tail_anode_sheath_reflected_erg_s)
            + float(b.tail_anode_sheath_reflected_erg_s)
        ),
        tail_gap_born_flux_per_s=(
            float(a.tail_gap_born_flux_per_s)
            + float(b.tail_gap_born_flux_per_s)
        ),
        **{
            name: float(getattr(a, name)) + float(getattr(b, name))
            for name in _NET_LEDGER_FIELDS
        },
        # Mirror far end: both rays meet the same plane and the same leg
        # budget, so their arrivals and residuals add like the end ledger.
        **{
            name: float(getattr(a, name)) + float(getattr(b, name))
            for name in _MIRROR_DEPOSITION_FIELDS
        },
    )


#: The ``BeamDepositionResult`` scalars the primary's net-basis ledger fills
#: (0.0 unless ``primary_net_basis``); flux rows, so a clumping split adds them.
_NET_LEDGER_FIELDS = (
    "primary_births_flux_per_s",
    "primary_net_direct_flux_per_s",
    "primary_net_return_flux_per_s",
    "primary_net_remnant_flux_per_s",
)


#: The ``BeamDepositionResult`` scalars a mirror far end fills (0.0 on every
#: other layout): the primary and walker flux/power arriving at the plane and
#: the two leg-cap residuals.
_MIRROR_DEPOSITION_FIELDS = (
    "primary_mirror_flux_per_s",
    "primary_mirror_erg_s",
    "primary_mirror_residual_flux_per_s",
    "primary_mirror_residual_erg_s",
    "tail_mirror_flux_per_s",
    "tail_mirror_erg_s",
    "tail_leg_cap_residual_flux_per_s",
    "tail_leg_cap_residual_erg_s",
)


def _plasma_active_window(geometry):
    """Return the inclusive ``(lo, hi)`` cell range the plasma occupies.

    The K6 tail walkers may traverse exactly these cells: outside them the
    solver's active-plasma mask zeroes every row, so a walk there deposits and
    births into nothing. Resolved geometry has one contiguous live run (the
    plenum and obstruction sit behind the cathode at the low end); a
    geometry with the live region split into several runs has no single window
    and is refused rather than silently walked across the gap.
    """
    active = np.asarray(geometry.plasma_active, dtype=bool)
    live = np.flatnonzero(active)
    if live.size == 0:
        raise ValueError(
            "no plasma-active cells: the QL tail walk has nowhere to go"
        )
    lo, hi = int(live[0]), int(live[-1])
    if not active[lo : hi + 1].all():
        raise ValueError(
            "plasma-active cells are not contiguous "
            f"({np.flatnonzero(~active[lo : hi + 1]) + lo}); the tail walk "
            "window is a single inclusive range and cannot describe this "
            "topology"
        )
    return lo, hi


def tail_reflect_face(geometry, end=0):
    """Return which walk-window face the cathode at ``end`` occupies (K7).

    ``-1`` for the window's low-index face, ``+1`` for its high-index face:
    the face BEHIND the ray, since the beam is launched from the cathode into
    the machine. That face is the one
    ``heating_anomalous_tail_cathode_boundary="reflect"`` turns walkers around
    at, so it must be the face the cathode actually sits at. A geometry whose
    ray is launched from somewhere other than the window's face cell is refused
    rather than having its reflection applied at a face that is not a cathode.
    """
    lo, hi = _plasma_active_window(geometry)
    launch, direction = beam_launch(geometry, end=end)
    face = -1 if direction > 0 else 1
    face_cell = lo if face < 0 else hi
    if int(launch) != int(face_cell):
        raise ValueError(
            f"the cathode ray for end {end} is launched from cell {launch}, "
            f"which is not the face cell {face_cell} of the plasma-active "
            f"window {(lo, hi)}; sheath reflection turns tail walkers around "
            "at that face, so it has to be the face the cathode occupies"
        )
    return face


def tail_mirror_face(geometry):
    """Return the walk-window face the MIRROR plane occupies, or ``None``.

    ``None`` on every geometry without a mirror face (``far_end =
    "end_wall"``, ``TwinCathode``). Under ``far_end = "mirror"`` the plane
    ends the grid at its high-index end (face ``cells``), so the answer is
    ``+1`` -- after checking that the plasma-active window ends at the last
    cell, because the CSDA module turns the primary and the tail walkers round
    at that window face and walks them back inside it. A mirror face anywhere
    else, or a window stopping short of it, raises.
    """
    faces = np.asarray(getattr(geometry, "mirror_face_indices", ()), dtype=int)
    if faces.size == 0:
        return None
    cells = int(geometry.cells)
    lo, hi = _plasma_active_window(geometry)
    if faces.tolist() != [cells] or hi != cells - 1:
        raise ValueError(
            f"the mirror face(s) {faces.tolist()} and the plasma-active window "
            f"{(lo, hi)} do not end the grid together (cells={cells}); the "
            "beam and the tail walkers turn round at the window face that is "
            "the mirror plane"
        )
    return 1


def _csda_beam_deposition(
    beam_result,
    state,
    derived,
    geometry,
    device_config,
    input_dict,
    I_ion,
    twin=False,
):
    """Run the CSDA module for each active cathode ray (B2 wiring).

    Returns ``(deposition, gap_ledger, plateau_edge)``. ``deposition`` is
    ``{0: BeamDepositionResult | None, -1: ...}``; ``gap_ledger`` maps each
    end with an active ray to ``(probe, ray, circuit, ceiling)`` gap survival
    for the item-35 tripwire, ``ceiling`` being the Coulomb-only bound the
    clamp described below pins the circuit to when the ray breaks out
    (see ``beam_gap_ledger_mismatch``). The call also
    rewrites
    ``beam_result.beam_atten_cross`` at each launch cell with the effective
    attenuation cross section that makes the frozen sheath solve's
    Beer-Lambert bypass reproduce the module's cathode-anode gap
    transmission on the *next* solve (the same one-step lag the historical
    ``sigma_b`` feedback has). The frozen solve computes
    ``bypass = exp(-L_cath / l_b)`` with ``1/l_b = 1/l_bi + sigma*nn``, so
    the adapter solves for sigma and clamps at 0 — transmissions above the
    solve's Coulomb-only ceiling ``exp(-L_cath/l_bi)`` saturate there
    (stated limitation; exact for transmissions at or below the ceiling,
    including the quasilinear closure's ~0).

    The gap-transmission probe is FLUX-FAITHFUL: it launches the same total
    flux as the deposition rays above, split the same way when clumping is
    active, so flux-dependent stopping (the quasilinear closure, whose
    relaxation length runs on n_b ~ Gamma0/(A v_b)) is felt by the probe
    exactly as the deposition ray feels it. Transmission is then the ratio of
    total transmitted to total launched flux. It was historically launched at
    unit flux, which made the quasilinear closure invisible to the circuit:
    transmission read 1, ``sigma_eff`` wrote 0, and the circuit kept booking
    ``eta * f_bypass`` of the emitted beam power as never-coupling while the
    real ray stopped inside the gap.

    Anode-mesh interception (R4.1, audit A15): wherever the geometry resolves
    an anode face and ``device_config.eta > 0``, the mesh solid fraction
    ``eta`` of the beam surviving the gap is intercepted at the anode-face
    crossing (``deposit_beam(anode_cross_index=..., anode_eta=...)``), so the
    fluid does not deposit the long-mfp beam the circuit books as never
    entering the plasma. The gap-transmission probe is unaffected (it measures
    gap survival, which feeds the circuit bypass).

    ``heating_anomalous_transport="plateau_multigroup"`` (WP-E, with the K6
    ionizing walkers) is threaded to the DEPOSITION rays only. The probe rays
    keep the historical argument list: they are transmission instruments
    whose single output is the ratio of transmitted to launched PRIMARY flux,
    which no closure here can change (the walks move deposited energy and add
    SECONDARY events -- never the primary's own flux), so walking their tails
    would be pure cost. For the same reason
    ``_ray_gap_breakout`` and the item-35 tripwire are unaffected -- both read
    ``transmitted_flux`` and ``E_entry_eV``, which are primary-flux
    quantities the walks never touch.
    """
    anomalous_model = str(input_dict.get("beam_anomalous_model", "none"))
    # The multi-group plateau closure is the one walked tail. It derives its
    # birth spectrum, and the per-ray loop solves the plateau edge for it.
    multigroup = str(
        input_dict.get("heating_anomalous_transport", "local")
    ) == "plateau_multigroup"
    # A2a. Read here, threaded onto the DEPOSITION rays only (below): the
    # gap-transmission probes report a primary-flux ratio and never walk a
    # tail, so culling one there would be work with no output. The solver
    # validated the trio at construction.
    # The tail cull is not a choice: a mesh that is opaque to the streaming
    # primary is opaque to the QL tail walkers too, so it is armed wherever
    # the primary's interception is AND the closure actually walks a tail.
    # With no walked tail there are no walkers to cull.
    tail_interception = multigroup
    net_basis = (
        str(input_dict.get("anode_tail_booking", "lagged_current"))
        == "emission_fraction"
    )
    tail_R_e = float(
        input_dict.get("beam_tail_anode_reflected_particles", 0.0)
    )
    tail_eta_E = float(
        input_dict.get("beam_tail_anode_reflected_energy", 0.0)
    )
    # Read unconditionally, threaded conditionally (below). ``None`` is what
    # the module refuses on rather than substituting a default for, so a config
    # that selects the closure without registering a bracket arm still raises
    # -- the solver's construction-time check is the loud copy of the same
    # requirement.
    ql_relaxation_coeff = input_dict.get("ql_relaxation_coeff", None)
    # The medium every ray below marches through, and the cross-section the
    # quasilinear closure forms its beam density on.
    ray_ne, ray_nn = state.n, state.nn
    beam_area_cm2 = geometry.plasma_area_cm2
    # QL heating locality. Presence-gated: only the DEPOSITION rays get the
    # keywords, and only when the tail is walked, so the "local" path enters
    # deposit_beam with the identical argument list it always had. The
    # gap-transmission PROBE rays below deliberately never receive them --
    # they are transmission instruments whose only output is a primary-flux
    # ratio, so walking their tails would be wasted work and would not change
    # the number they report.
    transport_kwargs = {}
    # Whether the cathode face reflects. A per-RAY quantity (the reflecting
    # face is its own cathode's and the threshold its own phi_c), so it is
    # resolved in the loop below.
    tail_reflect = False
    if multigroup:
        transport_kwargs["anomalous_transport"] = "plateau_multigroup"
        # The launch-direction split, passed only when it is not the symmetric
        # default, so a symmetric arm enters deposit_beam with the argument
        # list it had before this key existed.
        tail_forward_fraction = float(
            input_dict.get("heating_anomalous_tail_forward_fraction", 0.5)
        )
        if tail_forward_fraction != 0.5:
            transport_kwargs["tail_forward_fraction"] = tail_forward_fraction
        # K6: the walkers ionize and excite the column gas they cross.
        transport_kwargs["tail_ionization"] = "on"
        # The walk window the module refuses to default (see its docstring):
        # the maximal contiguous PLASMA-ACTIVE run, which in resolved geometry
        # starts at the cathode cell -- so the cathode disc and the
        # obstruction/plenum behind it are a wall to a tail electron, and no
        # pair is born into a row the RHS mask zeroes. Derived from
        # ``geometry.plasma_active`` rather than from the cathode roles,
        # because "the cells whose plasma rows the solver integrates" is
        # exactly the property that matters here, and it is the same array the
        # mask itself is built from.
        transport_kwargs["tail_walk_window"] = _plasma_active_window(geometry)
        if str(
            input_dict.get(
                "heating_anomalous_tail_cathode_boundary", "reflect"
            )
        ) != "escape":
            # The cathode reflects: the reflecting face is one of the walk
            # window's faces.
            tail_reflect = True
    if transport_kwargs:
        # Hoisted stopping coefficient (cost read 2026-08-02, restructure C).
        # The walks' per-cell A in dE/dx = A W**p is a 262-iteration Python
        # listcomp costing ~100 us -- half the whole WP-E per-call surcharge --
        # and it depends only on (ne, Te, model), which are the SAME for every
        # deposition ray in this call: both cathode ends under TwinCathode,
        # both halves of the clumping split (which varies nn alone), and every
        # energy group a future WP-F build adds. Build it once here rather than
        # once per ray.
        #
        # The walk runs on the same mean state the rays march through.
        transport_kwargs["stopping_coefficient"] = (
            _coulomb_stopping_coefficient(
                state.n, derived.Te, "fast_electron"
            )
        )
    # The mirror far end, threaded onto the DEPOSITION rays only, like the
    # walk: the gap probes are clipped at L_cath and never reach the plane.
    # Presence-gated on the geometry's mirror face, so an end wall ray enters
    # deposit_beam with the argument list it always had. The window the
    # module walks the returning populations in is the plasma-active one the
    # walked tail already uses.
    mirror_face = tail_mirror_face(geometry)
    if mirror_face is not None:
        transport_kwargs["mirror_face"] = mirror_face
        transport_kwargs["tail_walk_window"] = _plasma_active_window(geometry)
    # Fractional-coverage beam-neutral closure (default off/uniform, bit-exact):
    # split the ray into a clump fraction (short l_b against nn*chi -> local seed)
    # and a gap fraction (background nn -> penetration). See config docstrings.
    f_clump = float(input_dict.get("beam_clump_fraction", 0.0))
    chi_clump = float(input_dict.get("beam_clump_enhancement", 1.0))
    clumping = f_clump > 0.0 and chi_clump > 1.0
    L_cath = float(device_config.L_cath)
    eta = float(device_config.eta)
    anode_faces = np.asarray(
        getattr(geometry, "anode_face_indices", ()), dtype=int
    )
    deposition = {}
    gap_ledger = {}
    # Presence-gated: ``None`` under every value but the multi-group plateau,
    # so an unarmed solve's result carries exactly the fields it always did.
    plateau_edge = {} if multigroup else None
    ends = (0, -1) if twin else (0,)
    for end in ends:
        result = beam_result.result if end == 0 else beam_result.result_twin
        # The energy THIS ray carries into the column: the solved net cathode
        # drop ``result.phi_c``. One local carries it to the deposition ray,
        # the gap probe, the plateau spectrum and the sigma_eff inversion
        # alike, so the ray and the instruments that measure it cannot be
        # launched at two different energies.
        phi_c_ray = None if result is None else result.phi_c
        if result is None or phi_c_ray <= I_ion:
            deposition[end] = None
            continue
        launch, direction = beam_launch(geometry, end=end)
        Gamma0 = beam_launched_current_A(result) / qe_SI
        # K7, per ray: phi_c is THIS cathode's accelerating drop -- the same
        # quantity the ray is launched at and the same one the sheath repels
        # returning electrons with -- so the top of the plateau spectrum and
        # the reflection threshold both come from it. Left as the shared dict
        # when the tail is not walked, so the "local" path passes the
        # identical object it always did.
        ray_transport = transport_kwargs
        if multigroup:
            ray_transport = dict(transport_kwargs)
            if multigroup:
                # THE PLATEAU EDGE, solved once per EXTRACTION (this ray's own
                # sheath drop, this ray's own launch cell, the whole emitted
                # flux): the flat plateau's level is the launch cell's
                # Maxwellian at the edge and the flat band must carry the
                # emitted beam's number flux, which is one equation with one
                # root. Solved here rather than inside the deposition module
                # so a clumping split -- which varies nn alone and is NOT two
                # extractions -- shares the one edge its one solve produced,
                # and so the clamp census has a single per-solve entry to
                # count.
                _E1, _clamp = plateau_edge_energy_eV(
                    float(phi_c_ray),
                    Gamma0 / float(geometry.plasma_area_cm2[launch]),
                    float(ray_ne[launch]),
                    float(derived.Te[launch]),
                )
                ray_transport["plateau_edge_eV"] = _E1
                plateau_edge[end] = (_E1, _clamp)
            if tail_reflect:
                ray_transport["tail_reflect_face"] = tail_reflect_face(
                    geometry, end=end
                )
                ray_transport["tail_reflect_threshold_eV"] = float(
                    phi_c_ray
                )
        ray_kwargs = dict(
            nn=ray_nn,
            ne=ray_ne,
            Te=derived.Te,
            launch=launch,
            direction=direction,
            I_ion_eV=float(I_ion),
            anomalous_model=anomalous_model,
        )
        if anomalous_model != "none":
            ray_kwargs["beam_area_cm2"] = beam_area_cm2
        # Presence-gated exactly like the area above: the bracket constant only
        # reaches the module when the closure that reads it is selected, so the
        # other two arms enter deposit_beam with the argument list they shipped
        # with.
        if anomalous_model == "ql_relaxation":
            ray_kwargs["ql_relaxation_coeff"] = ql_relaxation_coeff
        interception_kwargs = {}
        if eta > 0.0 and anode_faces.size > 0:
            # The ray crosses the anode face between cell ``f-1`` and cell ``f``;
            # the first cell on the far (column) side along the ray is the
            # cross cell (``f`` when heading +z, ``f-1`` when heading -z).
            anode_face = int(anode_faces[0] if end == 0 else anode_faces[-1])
            cross_cell = anode_face if direction > 0 else anode_face - 1
            interception_kwargs = dict(
                anode_cross_index=cross_cell, anode_eta=eta
            )
            if tail_interception:
                # The SAME cell and the SAME eta, met by the QL tail walkers
                # instead of by the streaming primary. One mesh, one opacity:
                # the two views cannot disagree about whether it is there.
                interception_kwargs.update(
                    tail_anode_cross_index=cross_cell,
                    tail_anode_eta=eta,
                    # The wires' own barrier: this solve's anode drop, the
                    # SAME number the circuit books the tail's return power
                    # on. The mesh floats phi_a below the plasma, so an
                    # intercepted walker below it never reaches a wire and
                    # the sheath turns it back. Read from THIS step's solve
                    # (the circuit is solved before the deposition), so the
                    # barrier is not lagged the way the tail CURRENT it
                    # feeds back is. A non-positive drop reflects nothing.
                    tail_anode_phi_eV=max(float(result.phi_a), 0.0),
                    tail_anode_reflected_particles=tail_R_e,
                    tail_anode_reflected_energy=tail_eta_E,
                )
                if net_basis:
                    # The primary's particle ledger on the net basis and the
                    # wire-sheath test on its returns, for the booking that
                    # reads them.
                    interception_kwargs["primary_net_basis"] = True
                    # The same rule for the outbound primary at the plane:
                    # the share of its eta interception THIS solve's anode
                    # balance collected (1 where the beam at phi_c clears
                    # the sheath, 0 where it cannot, between where phi_a is
                    # pinned at phi_c). Read from the solve that launched
                    # the ray, like the wires' barrier above, so the circuit
                    # and the deposition book one fraction.
                    interception_kwargs[
                        "primary_anode_collected_fraction"
                    ] = float(result.anode_direct_collected_fraction)
        clump_kwargs = (
            {**ray_kwargs, "nn": np.asarray(ray_nn) * chi_clump}
            if clumping
            else None
        )
        # The gap's per-cell path length, shared by the probe below and by the
        # deposition ray's own breakout test.
        gap_dz = _clip_ray_length(
            geometry.length_cm, launch, direction, L_cath
        )
        if clumping:
            # Gap ray: background nn, penetrates (the fast far-end pedestal).
            gap_ray = deposit_beam(
                phi_c_ray, (1.0 - f_clump) * Gamma0,
                dz_cm=geometry.length_cm,
                **ray_kwargs, **interception_kwargs, **ray_transport,
            )
            # Clump ray: enhanced nn -> short l_b -> local deposit (front seed).
            clump_ray = deposit_beam(
                phi_c_ray, f_clump * Gamma0,
                dz_cm=geometry.length_cm,
                **clump_kwargs, **interception_kwargs, **ray_transport,
            )
            dep = _sum_beam_deposition(gap_ray, clump_ray)
            # Breakout is per-ray: the clump ray can die in the gap while the
            # gap ray penetrates, so the split's gap survival is the
            # flux-weighted mean. (`dep` cannot answer this -- it carries the
            # elementwise MAX of the two E_entry profiles.)
            ray_survival = (
                (1.0 - f_clump)
                * _ray_gap_breakout(gap_ray, gap_dz, launch, direction)
                + f_clump
                * _ray_gap_breakout(clump_ray, gap_dz, launch, direction)
            )
        else:
            dep = deposit_beam(
                phi_c_ray, Gamma0, dz_cm=geometry.length_cm,
                **ray_kwargs, **interception_kwargs, **ray_transport,
            )
            ray_survival = _ray_gap_breakout(dep, gap_dz, launch, direction)
        deposition[end] = dep
        # Gap transmission: gap-clipped probe rays MIRRORING the deposition
        # above -- same launched fluxes, same clump split, same nn per ray --
        # truncated at L_cath. Launching at the real Gamma0 (rather than the
        # historical unit flux) is what makes flux-DEPENDENT stopping visible
        # to the circuit: the quasilinear relaxation length runs on the beam
        # density n_b ~ Gamma0/(A v_b), so a unit-flux probe feels no
        # anomalous drag, reads transmission 1, writes sigma_eff = 0, and
        # leaves the circuit booking a bypass the real ray never enjoys
        # (root-caused 2026-07-27). Under flux-INDEPENDENT stopping
        # (Coulomb CSDA, anomalous_model="none") the ray is flux-linear, so
        # the ratio below is bit-for-bit the historical unit-flux value.
        #
        # --- Probe skip (cost read 2026-08-02, restructure A) --------------
        # The probe is a SECOND full CSDA march and measures ~50% of the whole
        # deposit_beam subsystem, yet in the main discharge it re-derives an
        # answer ``_ray_gap_breakout`` has already given from the deposition
        # ray's own bookkeeping. When that reads 0.0 the ray was ABSORBED
        # inside the gap, and the probe -- the same ray over the same per-cell
        # path lengths, merely stopped at L_cath -- is absorbed at the same
        # point, so ``BeamDepositionResult.transmitted_flux`` is the literal
        # float ``0.0`` (``0.0 if absorbed else gamma``). ``survival`` is then
        # ``0.0 / launched``, exactly 0.0 for any finite positive launch, and
        # ``transmission`` the 1e-6 clamp below. This is an EXACT-ZERO
        # argument, not a tolerance: the skipped branch writes the same floats
        # the probe would have returned, so every downstream number --
        # sigma_eff, the ledger, the tripwire -- is bit-identical.
        #
        # Three conditions break the identity, and under any of them the probe
        # runs exactly as it always has:
        #
        #   clumping     the split launches TWO probes with two different nn
        #                profiles and sums their transmitted fluxes, while
        #                ``ray_survival`` is the flux-weighted mean of two
        #                per-ray breakouts. A zero mean does imply both rays
        #                died, but the two-ray path is left untouched rather
        #                than re-argued: it is off in production.
        #   partial clip ``_clip_ray_length`` truncates the cell L_cath ends
        #                in when the gap does not end on a cell face. The
        #                deposition ray then has MORE path in that cell than
        #                the probe and can die inside it while the probe runs
        #                out of dz and transmits. Production does NOT bind
        #                here: the CAD-span gap is ``5 x 10.65 == 53.25`` and
        #                ``L_cath`` is the same distance, so the clip ends on
        #                the anode face and ``_clip_ray_length`` lands on it
        #                EXACTLY (see its invariant -- forward accumulation,
        #                plus a mesh-scale snap for the residual case). The
        #                guard is on the general geometry. Historical note,
        #                because it was a real defect and not a hypothetical:
        #                the clip used to decrement a running remainder, which
        #                left a 3.55e-15 cm sliver at this gap, put non-zero
        #                dz on the anode-crossing cell, and opened the item-35
        #                ledger by 35.8 % of emitted beam power. It survived
        #                the previous 50 cm gap only because ``50.0/5 == 10.0``
        #                is exact in binary.
        #   anode in gap anode-mesh interception scales the DEPOSITION ray's
        #                flux at the anode-face crossing and the probe's not
        #                at all, so under flux-DEPENDENT stopping (the
        #                quasilinear closure) the two trajectories would part
        #                company. The anode sits past the gap in every
        #                campaign geometry -- the guard is free there -- but
        #                nothing in this function enforces that.
        #
        # The ``Gamma0 == 0`` unit-flux probe is a DIFFERENT measurement (the
        # flux-independent limit) with no deposition ray behind it to
        # read, so it sits outside this branch and keeps running verbatim.
        probe_transmits_exact_zero = (
            not clumping
            and ray_survival == 0.0
            and _gap_clip_is_face_aligned(gap_dz, geometry.length_cm)
            and not (
                interception_kwargs
                and float(gap_dz[interception_kwargs["anode_cross_index"]])
                > 0.0
            )
        )
        if Gamma0 > 0.0:
            if clumping:
                gap_launch = (1.0 - f_clump) * Gamma0
                clump_launch = f_clump * Gamma0
                transmitted = (
                    float(deposit_beam(
                        phi_c_ray, gap_launch,
                        dz_cm=gap_dz, **ray_kwargs,
                    ).transmitted_flux)
                    + float(deposit_beam(
                        phi_c_ray, clump_launch,
                        dz_cm=gap_dz, **clump_kwargs,
                    ).transmitted_flux)
                )
                # Sum the LAUNCHED fluxes the same way the transmitted ones
                # are summed, so a fully-transmitting split lands on exactly
                # 1.0 instead of (1-f)+f rounding a ulp off it.
                launched = gap_launch + clump_launch
            elif probe_transmits_exact_zero:
                # The probe would be absorbed exactly where the deposition ray
                # was; skip the march and take the float it would have
                # returned. (Not an approximation of the probe -- its value.)
                transmitted = 0.0
                launched = Gamma0
            else:
                transmitted = float(
                    deposit_beam(
                        phi_c_ray, Gamma0, dz_cm=gap_dz, **ray_kwargs
                    ).transmitted_flux
                )
                launched = Gamma0
            survival = transmitted / launched
        else:
            # No emission this frame: the flux-weighted ratio is 0/0. The
            # Gamma0 -> 0 limit of any flux-dependent stopping is the
            # flux-INDEPENDENT transmission, which is exactly what the
            # historical unit-flux probe measures, so keep it verbatim here.
            survival = float(
                deposit_beam(
                    phi_c_ray, 1.0, dz_cm=gap_dz, **ray_kwargs
                ).transmitted_flux
            )
        transmission = min(max(survival, 1.0e-6), 1.0)
        # sigma_eff is an EFFECTIVE cross section whose entire job is to make
        # the frozen sheath solve's Beer-Lambert bypass reproduce the
        # transmission the module measured, and that frozen solve runs on the
        # mean fields (see solve_cathode_boundary).
        nn_launch = float(state.nn[launch])
        l_bi = compute_l_b(
            phi_c_ray,
            float(derived.Te[launch]),
            float(state.n[launch]),
            0.0,
            0.0,
        )
        sigma_eff = 0.0
        if nn_launch > 0.0 and L_cath > 0.0 and l_bi > 0.0:
            sigma_eff = max(
                0.0,
                (-math.log(transmission) / L_cath - 1.0 / l_bi) / nn_launch,
            )
        beam_result.beam_atten_cross[launch] = sigma_eff
        # --- Beam gap ledger tripwire ----------------------------------
        # Three views of ONE number -- the fraction of the emitted beam that
        # crosses the cathode-anode gap -- which must agree, and which nothing
        # else in the model compares:
        #
        #   probe    the gap-clipped probe's transmission (feeds sigma_eff)
        #   ray      the DEPOSITION ray's own breakout, read off its internal
        #            bookkeeping and completely independent of the probe
        #   circuit  what the circuit reconstructs from the sigma_eff just
        #            written, built with the CIRCUIT's own functions rather
        #            than by inverting the adapter's algebra
        #
        # ``probe`` vs ``ray`` catches a defect INSIDE the probe -- the
        # item-35 class, where the probe misreports the ray it is supposed to
        # mirror. ``ray`` vs ``circuit`` catches the circuit failing to
        # represent the ray, whatever the cause: adapter clamp saturation, or
        # a broken probe that the adapter faithfully propagated. Item 35 sat
        # silently in the second: the circuit booked ~97% gap survival while
        # the deposition ray delivered 0.
        #
        # A FOURTH number is carried for the third case, and it is not a view
        # of the same quantity: ``ceiling`` is the highest survival the
        # circuit's Beer-Lambert solve CAN represent at this state, the
        # Coulomb-only ``exp(-L_cath/l_bi)`` that the ``sigma_eff >= 0`` clamp
        # pins it to. It is what separates a representability gap from a
        # divergence when the ray breaks out (see
        # ``beam_gap_ledger_mismatch``), and it is the same ``l_bi`` the
        # inversion above already computed -- read, not re-derived.
        gap_ledger[end] = (
            transmission,
            ray_survival,
            compute_beam_bypass_fraction(
                compute_l_b(
                    phi_c_ray,
                    float(derived.Te[launch]),
                    float(state.n[launch]),
                    nn_launch,
                    sigma_eff,
                ),
                L_cath,
            ),
            compute_beam_bypass_fraction(l_bi, L_cath),
        )
    return deposition, gap_ledger, plateau_edge


def _ray_gap_breakout(dep, gap_dz, launch, direction):
    """Fraction of a CSDA ray's flux that crosses the gap: 1.0 or 0.0.

    Probe-independent: it reads only the deposition ray's own bookkeeping.
    A CSDA ray carries its flux unattenuated until it stops (the anode mesh
    is the one exception, and it sits past the gap), so a single ray either
    crosses the gap whole or dies inside it -- there is no partial survival
    to measure. Fractional survival across the clumping split is handled by
    the caller, which weights the two rays' breakouts by their launched flux.

    ``dep.E_entry_eV`` is written for every cell the ray ENTERS and left at
    zero for cells it never reached, so a positive entry energy in the first
    cell beyond the gap means the ray got out. Reading entry energy (rather
    than deposited energy) is what makes this exact even when the gap ends
    mid-cell: entry energy is sampled before any of that cell's path is
    consumed, so truncating the last gap cell cannot perturb it.
    """
    # Reached the far end of the domain, so it certainly cleared the gap.
    # Also covers the sub-threshold ray, which passes through untouched and
    # leaves E_entry all zeros.
    if float(dep.transmitted_flux) > 0.0:
        return 1.0
    E_entry = np.asarray(dep.E_entry_eV, dtype=float)
    cells = gap_dz.size
    order = range(launch, cells) if direction > 0 else range(launch, -1, -1)
    for cell in order:
        if gap_dz[cell] <= 0.0:
            return 1.0 if E_entry[cell] > 0.0 else 0.0
    # The gap runs to the domain edge, so there is no cell beyond it to test;
    # the ray did not leave the far end either, so it died inside the gap.
    return 0.0


# Tripwire tolerance, as a fraction of EMITTED BEAM POWER (see
# ``beam_gap_ledger_mismatch``): 5%, roughly a decade above the benign
# Coulomb-ceiling floor and a decade below the item-35 break.
BEAM_GAP_LEDGER_POWER_ATOL = 0.05


def beam_gap_ledger_mismatch(
    gap_ledger,
    eta,
    atol=BEAM_GAP_LEDGER_POWER_ATOL,
    separate_representability=False,
):
    """Worst CSDA gap-survival ledger divergence, or ``None`` if all agree.

    ``gap_ledger`` maps each active cathode end to
    ``(probe, ray, circuit[, ceiling])`` gap survival (see
    ``_csda_beam_deposition``). Three comparisons are made:

    ``probe`` vs ``ray``
        The probe must reproduce the deposition ray it mirrors. This is the
        item-35 class: a probe that misreports the ray corrupts ``sigma_eff``
        and therefore the circuit, and every internally-consistent check
        downstream still passes.
    ``ray`` vs ``ceiling`` (only when ``separate_representability``)
        The MARGINAL-TRANSMISSION case, and the only one that is not a defect
        report. It is evaluated only for a BROKEN-OUT ray -- a ray that
        crossed the gap whole -- and scores it against the highest survival
        the circuit is able to represent. A fully-transmitting ray cannot be
        represented above the Beer-Lambert solve's Coulomb-only ceiling
        ``exp(-L_cath/l_bi)``, so the ``sigma_eff >= 0`` clamp leaves
        ``eta * (1 - ceiling)`` unbooked. That shortfall is a
        REPRESENTABILITY gap, not a disagreement between two views of one
        number, and naming it as one is this case's entire job.
    ``ray`` vs ``circuit``
        The circuit must be able to represent the ray. Fails on adapter clamp
        saturation, and again -- independently -- whenever a broken probe has
        been propagated into ``sigma_eff``.

    Returns ``(end, kind, left, right, power_fraction)`` for the worst
    offender, where ``kind`` is ``"probe_vs_ray"``, ``"ray_vs_ceiling"`` or
    ``"ray_vs_circuit"``.

    The tolerance is stated on the quantity that matters rather than on the
    survival fractions themselves. The circuit debits
    ``eta * f_bypass * beam_launched_current_A * V_b`` -- the launched current
    ``I_eth_star + I_see``, not ``I_eth_star`` alone -- so
    ``eta * |left - right|`` is the fraction of emitted beam power booked to a
    bypass the fluid never loses (or vice versa) -- the ledger hole itself.

    The ceiling shortfall was long assumed to be a small benign floor that
    would stay below ``atol`` on its own -- saturation needs a transmitting
    ray, which needs a long ``l_bi``, which makes the shortfall small -- and
    it measures 0.3-1.3% of emitted beam power across the campaign's
    long-mfp states. That self-limiting argument FAILS for a hot beam in a
    dense gap, where ``l_bi`` and the range-set transmission stop tracking
    each other: the 2026-08-06 diagnosis measured a broken-out ray against a
    ceiling of only ~0.78, and the excursion arrived labelled
    ``ray_vs_circuit`` -- indistinguishable, at the point of use, from a real
    ledger hole. That is what ``ray_vs_ceiling`` exists to tell apart. It is
    ordered BEFORE ``ray_vs_circuit`` below because in exactly that regime
    the clamp pins ``circuit`` to ``ceiling``, the two powers are equal, and
    the strict ``>`` hands the tie to whichever is seen first: the label
    changes, the trip does not. Item 35 -- a genuine hole -- reads 35.8% on
    ``probe_vs_ray`` and 34.6% on ``ray_vs_circuit``, and stays there,
    because a probe that misreports its ray leaves ``sigma_eff`` positive and
    ``circuit`` strictly below ``ceiling``.

    ``separate_representability`` is OFF by default, and with it off this
    function is the two-case instrument it has always been, verbatim -- the
    third case cannot change a returned value, only relabel one. It is a
    parameter rather than the new behaviour because the caller that acts on
    the result, ``LAPDSim1D._warn_beam_gap_ledger``, dispatches ``kind``
    through an exhaustive table and owns the operator-facing text for each;
    a third kind is meaningless until that table carries its explanation, and
    the two must land together. That table now carries it, so that caller
    passes ``separate_representability=True``; the default stays OFF for
    every other caller and for the two-case instrument's own tests.
    """
    eta = float(eta)
    worst = None
    for end, entry in (gap_ledger or {}).items():
        if entry is None:
            continue
        # A three-element entry is the pre-ceiling ledger shape; its third
        # case is simply not evaluated, which is the old behaviour exactly.
        probe, ray, circuit, *rest = (float(v) for v in entry)
        ceiling = rest[0] if rest else None
        comparisons = [("probe_vs_ray", probe, ray)]
        # ``ray`` is ``_ray_gap_breakout``'s binary verdict, so "broke out"
        # is a comparison against 1.0 and not a threshold of any kind.
        if separate_representability and ceiling is not None and ray >= 1.0:
            comparisons.append(("ray_vs_ceiling", ray, ceiling))
        comparisons.append(("ray_vs_circuit", ray, circuit))
        for kind, left, right in comparisons:
            power = eta * abs(left - right)
            if power > atol and (worst is None or power > worst[4]):
                worst = (int(end), kind, left, right, power)
    return worst


#: Relative width below which a clip remainder is snapped to the nearest cell
#: face (see ``_clip_ray_length``). It is a MESH-SCALE bound, not a physics
#: one: at the production 10.65 cm gap cell it is 1.065e-11 cm -- about a
#: tenth of a picometre. That is ~1e4 above the double-rounding scale of these
#: accumulations (~1e-15 relative) and ~1e10 below any length this model
#: resolves, so it can only ever absorb arithmetic noise. A residue that small
#: must never be read as the ray crossing the anode face; a real partial clip
#: is a macroscopic fraction of a cell and is untouched.
_CLIP_FACE_SNAP_REL = 1.0e-12


def _clip_ray_length(length_cm, launch, direction, L_cath):
    """dz array truncated so the ray's total path is at most ``L_cath``.

    **Invariant: a clip that ends ON a cell face lands on it EXACTLY** -- the
    stop cell gets either its full length or zero, never a rounding sliver.
    Two mechanisms hold it, in this order of preference:

    1. **Exact arithmetic.** The distance travelled is ACCUMULATED FORWARD
       along the ray (``travelled + step``) rather than decremented out of a
       running remainder. Forward accumulation reproduces the same
       left-to-right sum the mesh itself uses to place its faces, so where
       ``L_cath`` coincides with a face the comparison is exact with no
       tolerance at all. The previous subtractive form did NOT have this
       property: ``53.25 - 5 x 10.65`` leaves ``3.55e-15`` while
       ``5 x 10.65`` accumulated forward is ``53.25`` to the bit.
    2. **A mesh-scale snap** (``_CLIP_FACE_SNAP_REL``) for the residual case
       where ``L_cath`` and the accumulated face differ by rounding rather
       than coinciding. Exactness is genuinely unattainable there -- the two
       quantities come from different arithmetic -- so a bound is used, and it
       is justified from the mesh scale rather than chosen for convenience.

    Why it matters, and it is not cosmetic: a sliver in the stop cell makes
    ``_gap_clip_is_face_aligned`` false AND puts a non-zero ``dz`` on the
    anode-crossing cell. The second is the damaging one -- anode-mesh
    interception scales the deposition ray's flux there and the probe's not at
    all, so the two part company and the item-35 gap ledger opens. Measured on
    the CAD-span geometry before this fix: a 3.55e-15 cm sliver mis-booked
    35.8 % of the emitted beam power (tolerance 5 %).
    """
    full = np.asarray(length_cm, dtype=float)
    dz = np.zeros_like(full)
    limit = float(L_cath)
    cells = dz.size
    order = range(launch, cells) if direction > 0 else range(launch, -1, -1)
    # Distance from the launch face to the NEAR face of the current cell.
    travelled = 0.0
    for cell in order:
        if travelled >= limit:
            break
        step = float(full[cell])
        far_face = travelled + step
        if far_face <= limit:
            # The whole cell is inside the clip: emit its length verbatim.
            dz[cell] = step
        else:
            partial = limit - travelled
            snap = _CLIP_FACE_SNAP_REL * step
            if partial <= snap:
                # The clip landed on this cell's NEAR face to within
                # arithmetic noise: stop short rather than emit a sliver.
                break
            if partial >= step - snap:
                # ... and on its FAR face: take the whole cell, so the guards
                # below see a prefix of the mesh rather than a near-full cell.
                dz[cell] = step
            else:
                dz[cell] = partial
            break
        travelled = far_face
    return dz


def _gap_clip_is_face_aligned(gap_dz, length_cm):
    """True when the ``L_cath`` clip landed on a cell face.

    ``_clip_ray_length`` gives each cell along the ray either its full length,
    zero, or -- in the single cell where ``L_cath`` runs out GENUINELY
    mid-cell -- a partial length. It guarantees that a clip ending on a face
    produces no partial cell at all (see its invariant), so this test reads
    real geometry and never arithmetic noise. That partial cell is the ONLY
    place a gap-clipped probe
    has less path available than the deposition ray it mirrors, and therefore
    the only place the two can disagree about where the ray stopped: the
    deposition ray can be absorbed inside it while the probe runs out of dz
    first and transmits. Everywhere else the clip is a prefix of the same
    per-cell path lengths. Used by the probe skip in
    ``_csda_beam_deposition``; see the comment there.
    """
    dz = np.asarray(gap_dz, dtype=float)
    full = np.asarray(length_cm, dtype=float)
    return not bool(np.any((dz > 0.0) & (dz < full)))


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
            anode_rhs=ConservativeState1D(
                n=zeros.copy(),
                nn=zeros.copy(),
                M=zeros.copy(),
                Ee=zeros.copy(),
                Ei=zeros.copy(),
            ),
            enabled=boundary.enabled,
            metadata={
                "source_index": boundary.source.index,
                "end_index": (
                    None if boundary.end is None else boundary.end.index
                ),
                "twin_cathode": boundary.twin_cathode,
                "circuit": dict(boundary.circuit),
                "surface_particle_loss_s_inv": zeros.copy(),
            },
        )

    derived = derive_state(state, floors=floors, ion_mass_g=ion_mass_g)
    dN_loss = zeros.copy()
    # An absorbing cathode face already drains the plasma at the Bohm flux, the
    # same expression the circuit's I_i is built from (A*e*n*c_s*alpha_se on
    # the same sheath-edge factor, sound speed and area, the emitting disc
    # being the face), so applying this volumetric loss as well would remove
    # the same population twice. The two numbers differ only by sampling: the
    # face reads the live cell's raw state, the circuit the smoothed electrode
    # sample (MODEL.md, "ONE book for the cathode ion current"). The electron
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
    # Recycle at the cathode surface feeds the COLUMN on a two-zone state.
    neutral_gain_rate = dN_loss / (
        geometry.plasma_volume_cm3
        if state.nn_a is not None
        else geometry.neutral_volume_cm3
    )
    # Sheath electron power: P_cathode_e is lost at the cathode surface and
    # P_anode_e at the anode mesh. Legacy has neither resolved, so both stay
    # colocated in its source cell exactly as before; resolved geometry lands each
    # at its own electrode.
    # Split by ELECTRODE, one accumulator each. Where an anode is resolved
    # these two never touch the same cell, which is what makes their sum
    # bit-exactly the single row they replace.
    cathode_power_loss_W = zeros.copy()
    anode_power_loss_W = zeros.copy()
    # The anode block's own electron temperature, per cell the anode power
    # lands in; 0.0 everywhere else. It is the reference temperature of the
    # exactly-linear sink rate built below.
    anode_Te_ref_eV = zeros.copy()
    cathode_cells = cathode_adjacent_cells(geometry)
    anode_pairs = anode_flanking_cells(geometry)
    if cathode_cells:
        _deposit_electrode_power(
            cathode_power_loss_W,
            anode_power_loss_W,
            result=cathode_solve.beam_result.result,
            cathode_cell=int(cathode_cells[0]),
            anode_pair=anode_pairs[0] if anode_pairs else None,
            state=state,
            derived=derived,
            anode_Te_ref_eV=anode_Te_ref_eV,
        )
        if (
            boundary.twin_cathode
            and cathode_solve.beam_result.result_twin is not None
        ):
            _deposit_electrode_power(
                cathode_power_loss_W,
                anode_power_loss_W,
                result=cathode_solve.beam_result.result_twin,
                cathode_cell=int(cathode_cells[-1]),
                anode_pair=anode_pairs[-1] if len(anode_pairs) > 1 else None,
                state=state,
                derived=derived,
                anode_Te_ref_eV=anode_Te_ref_eV,
            )
    else:
        # Unresolved geometry: the lumped model puts both electrodes in the
        # boundary cell, and this branch keeps the HISTORICAL full P_*_e (it
        # predates the thermal-only routing and is unaffected by it). The two
        # shares are still booked to their own rows, so the split is complete
        # here as well; the caveat is that they share a cell, so the summed
        # density agrees to roundoff rather than bit-exactly.
        result_source = cathode_solve.beam_result.result
        cathode_power_loss_W[0] = result_source.P_cathode_e
        anode_power_loss_W[0] = result_source.P_anode_e
        anode_Te_ref_eV[0] = float(result_source.T_e_anode)
        if (
            boundary.twin_cathode
            and cathode_solve.beam_result.result_twin is not None
        ):
            result_twin = cathode_solve.beam_result.result_twin
            cathode_power_loss_W[-1] = result_twin.P_cathode_e
            anode_power_loss_W[-1] = result_twin.P_anode_e
            anode_Te_ref_eV[-1] = float(result_twin.T_e_anode)
    volume_cm3 = geometry.plasma_volume_cm3
    cathode_power_loss_density = cathode_power_loss_W * 1.0e7 / volume_cm3
    anode_power_loss_density = anode_power_loss_W * 1.0e7 / volume_cm3
    anode_sink_rate_s_inv = _anode_sink_rate_s_inv(
        anode_power_loss_density=anode_power_loss_density,
        anode_Te_ref_eV=anode_Te_ref_eV,
        n=np.maximum(state.n, floors["n"]),
    )
    # Metadata keeps the COMBINED array under its historical name and meaning
    # (both electrodes, per cell) and states each electrode's share beside it.
    # This is a diagnostic, not the booking: the booking is the two rows.
    electron_power_loss_W = cathode_power_loss_W + anode_power_loss_W
    return CathodeSourceTerms1D(
        rhs=ConservativeState1D(
            n=-plasma_loss_rate,
            nn=neutral_gain_rate,
            M=-ion_mass_g * derived.u * plasma_loss_rate,
            Ee=-cathode_power_loss_density,
            Ei=-1.5 * ev_to_erg * derived.Ti * plasma_loss_rate,
        ),
        anode_rhs=ConservativeState1D(
            n=zeros.copy(),
            nn=zeros.copy(),
            M=zeros.copy(),
            Ee=-anode_power_loss_density,
            Ei=zeros.copy(),
        ),
        enabled=boundary.enabled,
        metadata={
            "source_index": boundary.source.index,
            "end_index": (
                None if boundary.end is None else boundary.end.index
            ),
            "twin_cathode": boundary.twin_cathode,
            "circuit": dict(boundary.circuit),
            "surface_particle_loss_s_inv": dN_loss,
            "source_surface_particle_loss_s_inv": float(dN_loss[0]),
            "end_surface_particle_loss_s_inv": float(dN_loss[-1]),
            "electron_power_loss_W": electron_power_loss_W,
            "source_electron_power_loss_W": float(electron_power_loss_W[0]),
            "end_electron_power_loss_W": float(electron_power_loss_W[-1]),
            "cathode_power_loss_W": cathode_power_loss_W,
            "anode_power_loss_W": anode_power_loss_W,
            "anode_Te_ref_eV": anode_Te_ref_eV,
            "anode_sink_rate_s_inv": anode_sink_rate_s_inv,
        },
    )


def _anode_sink_rate_s_inv(anode_power_loss_density, anode_Te_ref_eV, n):
    """Return the anode electron debit as a first-order loss rate [s^-1].

    The booked row is ``P_c`` [erg cm^-3 s^-1] at the circuit's own anode
    sample temperature ``Te_a``, and the debit per collected electron is
    ``(2 + psi_a) Te``, so the row is EXACTLY LINEAR in the temperature at a
    frozen solve. Dividing by the cell's heat capacity at the sample
    temperature,

        nu_c = P_c / (3/2 n_c Te_a e),

    turns it into the rate an implicit substep can carry: applying ``nu_c``
    to the LOCAL temperature reproduces the circuit's booking exactly where
    ``Te_c == Te_a`` and otherwise states the same physics at the
    temperature the electrons actually leave with, ``P_c Te_c / Te_a``.

    Zero power gives exactly zero rate. Power with no usable reference
    temperature has no rate form and raises rather than being dropped.
    """
    power = np.asarray(anode_power_loss_density, dtype=float)
    Te_ref = np.asarray(anode_Te_ref_eV, dtype=float)
    capacity_at_ref = 1.5 * np.asarray(n, dtype=float) * Te_ref * ev_to_erg
    usable = capacity_at_ref > 0.0
    unbookable = (power != 0.0) & ~usable
    if np.any(unbookable):
        bad = int(np.flatnonzero(unbookable)[0])
        raise RuntimeError(
            "anode electron sheath power has no rate form at cell "
            f"{bad}: power={power[bad]!r} erg cm^-3 s^-1 against reference "
            f"T_e_anode={Te_ref[bad]!r} eV and n={np.asarray(n)[bad]!r} "
            "cm^-3; the implicit substep cannot carry a debit whose "
            "reference heat capacity is not positive"
        )
    return np.where(usable, power / np.where(usable, capacity_at_ref, 1.0), 0.0)


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
    (
        S_beam,
        S_exc,
        S_exc_E,
        beam_power_density,
    ) = _beam_ionization_sources(
        state=state,
        geometry=geometry,
        cathode_solve=cathode_solve,
        boundary=boundary,
        Te=beam_derived.Te,
        n=np.maximum(state.n, floors["n"]),
        smoothing_cm=float(input_dict.get("beam_deposition_smoothing_cm", 0.0)),
    )
    volume_ratio = geometry.plasma_volume_cm3 / geometry.neutral_volume_cm3
    # Two-zone state: nn is the column density on
    # the plasma volume, so the beam's neutral debit converts by exactly 1
    # (the beam attenuates on column gas by construction).
    if state.nn_a is not None:
        volume_ratio = np.ones_like(volume_ratio)
    # In the kinetic-derived two-momentum reduction, beam ionization removes
    # a column neutral carrying u_c and births the ion with that same directed
    # momentum. Presence-gate on M_n_a so all historical M_n closures remain
    # bit-for-bit unchanged.
    if state.M_n_a is not None:
        u_birth = np.asarray(state.M_n, dtype=float) / (
            ion_mass_g
            * np.maximum(np.asarray(state.nn, dtype=float), floors["nn"])
        )
        beam_M_birth = ion_mass_g * u_birth * S_beam
        beam_Mn_debit = -beam_M_birth
        beam_Mna = np.zeros_like(state.M_n_a)
    else:
        # Historical zero-drift beam birth: the ion is born at rest.
        u_birth = zeros.copy()
        beam_M_birth = zeros.copy()
        beam_Mn_debit = None
        beam_Mna = None
    Ti_birth_ionization = input_dict.get("Ti_birth_ionization", "neutral")
    Ti_birth = _birth_temperature(
        Ti_birth_ionization,
        beam_derived.Ti,
        neutral_temperature=(
            ionization_birth_neutral_temperature_eV(
                state, floors, input_dict.get("Tn_K", 300.0)
            )
            if Ti_birth_ionization == "neutral"
            else None
        ),
    )
    # The beam electron is born cold (Ee = 0); the ion energy books the
    # mass-loading relative-drift mixing energy to Ei (the beam ion is born at
    # u_birth and joins the bulk flow at u_i), matching the bulk birth.
    beam_Ei = 1.5 * ev_to_erg * Ti_birth * S_beam
    beam_Ei = beam_Ei + 0.5 * ion_mass_g * (
        beam_derived.u - u_birth
    ) ** 2 * S_beam
    # Each ray radiates its own energy per event: the CSDA module's radiated
    # bank books the measured singlet manifold per E(z).
    exc_Ee = -ev_to_erg * S_exc_E
    return {
        "beam_ionization_birth": ConservativeState1D(
            n=S_beam,
            nn=-S_beam * volume_ratio,
            M=beam_M_birth,
            Ee=zeros.copy(),
            Ei=beam_Ei,
            M_n=beam_Mn_debit,
            nn_a=(
                np.zeros_like(state.nn_a)
                if state.nn_a is not None
                else None
            ),
            M_n_a=beam_Mna,
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
        # Excited neutrals radiate their ~21-22 eV promptly (2^1P lifetime
        # ~ns; the 2^1S metastable share is booked as radiated too, caveat on
        # the manifold registry), so the excitation channel's energy leaves
        # the plasma as He I light rather than heating it. The particle is
        # unchanged: the neutral returns to ground state.
        "beam_excitation_radiation": ConservativeState1D(
            n=zeros.copy(),
            nn=zeros.copy(),
            M=zeros.copy(),
            Ee=exc_Ee,
            Ei=zeros.copy(),
        ),
    }


_BEAM_SMOOTH_CACHE = {}
# Memo of the CACHE KEY itself, so the content fingerprints below are taken
# once per geometry rather than once per RHS evaluation. See
# :func:`_beam_smoothing_key` for why keying it on ``id(geometry)`` is sound.
# Each entry holds a STRONG REFERENCE to its geometry -- that is what makes
# the id unique -- so the memo is CAPPED and evicted in insertion order:
# uncapped, it would pin every geometry a process ever built, and the
# fingerprinted arrays with them. A process builds few geometries (one per
# run, a handful across a sweep), so a small cap holds the live ones.
_BEAM_SMOOTH_KEY_CACHE = {}
_BEAM_SMOOTH_KEY_CACHE_ENTRIES = 8


def _array_fingerprint(values, dtype):
    """Content fingerprint of an array, canonicalized to the consumed dtype.

    Returned as ``(shape, digest)``; the digest is taken over the exact bytes
    ``_beam_smoothing_matrix`` reads, so two arrays that differ anywhere the
    kernel looks produce different keys.
    """
    arr = np.ascontiguousarray(values, dtype=dtype)
    digest = hashlib.blake2b(arr.tobytes(), digest_size=16).digest()
    return arr.shape, digest


def _beam_smoothing_key(geometry, sigma_cm):
    """Cache key for :func:`_beam_smoothing_matrix`, by CONTENT not address.

    ``id(geometry)`` is unique only among LIVE objects: CPython reuses an
    address once the old geometry is collected, so a freed geometry followed
    by a differently meshed allocation at the same address would return the
    OLD mesh's matrix. A shape mismatch would raise at the matmul, but two
    geometries with the same cell count and different positions/lengths/roles
    (an nx-matched ``source_region_dz_cm`` sweep)
    would silently smooth with the wrong kernel.

    Every geometry input the matrix build reads is in the key: ``z_cm`` and
    ``length_cm`` (centres and the cell-length weighting), ``z_edges_cm``,
    ``cathode_face_indices`` and ``mirror_face_indices`` (the reflecting image
    sources), and ``plasma_active`` (the support -- two meshes agreeing in
    z/lengths/faces but differing in cell ROLES build different matrices).
    The mirror faces enter the key only on a geometry that has one, so an
    end wall key is the one it always was.

    The key itself is memoized on ``id(geometry)``, which is sound here and
    only here because the memo HOLDS A STRONG REFERENCE to the geometry it
    keyed: an address CPython still has a live reference to cannot be handed
    to a later allocation, so the reuse hazard the paragraph above describes
    is closed structurally rather than by re-fingerprinting. That strong
    reference is also why the memo is capped at
    ``_BEAM_SMOOTH_KEY_CACHE_ENTRIES`` and evicted in insertion order:
    otherwise it would pin every geometry a process ever built. Eviction is
    safe for the same reason it is needed -- an evicted geometry may then be
    collected and its address reused, but the entry that named that address
    is gone, so the next lookup at it is a miss and re-fingerprints.

    The one thing identity keying cannot see is a content edit made IN PLACE
    on a live geometry, so EVERY array the key reads -- ``z_cm``,
    ``length_cm``, ``z_edges_cm``, ``plasma_active``,
    ``cathode_face_indices`` and ``mirror_face_indices`` -- is marked
    read-only on the first key build:
    such an edit now raises instead of silently returning the previous mesh's
    matrix. Leaving any ONE of them writeable reopens the whole hazard, since
    a stale key is served whenever any component of the content the key
    summarises has moved. ``Sim1DGeometry`` is a frozen dataclass built at
    exactly one site and no consumer writes to any of the six.
    """
    memo_key = (id(geometry), round(float(sigma_cm), 8))
    entry = _BEAM_SMOOTH_KEY_CACHE.get(memo_key)
    if entry is not None:
        return entry[1]
    for values in (
        geometry.z_cm,
        geometry.length_cm,
        geometry.z_edges_cm,
        geometry.plasma_active,
        geometry.cathode_face_indices,
        getattr(geometry, "mirror_face_indices", None),
    ):
        if isinstance(values, np.ndarray):
            values.flags.writeable = False
    mirror_faces = tuple(
        int(i)
        for i in np.asarray(
            getattr(geometry, "mirror_face_indices", ()), dtype=int
        )
    )
    key = (
        round(float(sigma_cm), 8),
        _array_fingerprint(geometry.z_cm, float),
        _array_fingerprint(geometry.length_cm, float),
        _array_fingerprint(geometry.z_edges_cm, float),
        _array_fingerprint(geometry.plasma_active, bool),
        tuple(int(i) for i in np.asarray(geometry.cathode_face_indices, dtype=int)),
    ) + ((("mirror", mirror_faces),) if mirror_faces else ())
    # The geometry is stored, not just its id: the strong reference is what
    # makes the id unique for as long as the entry lives -- and what the cap
    # below bounds, so a long-lived process cannot accumulate geometries.
    while len(_BEAM_SMOOTH_KEY_CACHE) >= _BEAM_SMOOTH_KEY_CACHE_ENTRIES:
        del _BEAM_SMOOTH_KEY_CACHE[next(iter(_BEAM_SMOOTH_KEY_CACHE))]
    _BEAM_SMOOTH_KEY_CACHE[memo_key] = (geometry, key)
    return key


def _beam_smoothing_matrix(geometry, sigma_cm):
    """Conservative Gaussian redistribution matrix over the live plasma cells.

    ``W[i, j]`` is the fraction of cell ``j``'s beam deposition moved to cell
    ``i``; columns sum to 1 over the live cells, so ``W @ ext`` conserves the
    total (extensive) deposition. The width is a fixed length in cm, so the
    smoothed profile is mesh-convergent. The O(cells^2) build is cached on the
    geometry CONTENT and the width (see :func:`_beam_smoothing_key`) -- both
    fixed for a run -- so the matrix is built once per run rather than once per
    RHS evaluation, and two distinct meshes can never share an entry.

    The support is ``geometry.plasma_active``, NOT ``plasma_volume_cm3 > 0``.
    The typed plasma-dead cells behind the cathode face (plenum, obstruction)
    carry a finite plasma volume, so a ``Vp > 0`` support puts weight on rows
    that ``_apply_active_plasma_topology`` then zeroes -- silently deleting
    that share of the deposit (~19% at the cathode cell) from every channel
    this kernel serves.

    The cathode surface is a REFLECTING boundary: the Gaussian tail that would
    fall behind an emitting face is folded forward about that face (image
    source at ``2*z_face - z_j``) instead of being discarded, which is what
    keeps the deposit near the cathode physical rather than merely normalized.
    Both faces reflect under ``TwinCathode``. A MIRROR face (``far_end =
    "mirror"``) reflects too, about the mirror plane: the machine is symmetric
    there, so the Gaussian tail that crosses the plane is the image source's
    deposit smoothed back across it, and the fold makes the half column's
    matrix the full two-source machine's restricted to one half. The far end
    wall needs no special handling -- every cell there is active, and
    normalization absorbs the residual tail past the end.

    Each weight is multiplied by the target cell length (a cell-integrated
    approximation) before the column is normalized, so a refined region is not
    over-weighted per cm and the operator is mesh-independent, not just
    conservative. Normalization remains the exact conservation guarantee.
    """
    key = _beam_smoothing_key(geometry, sigma_cm)
    W = _BEAM_SMOOTH_CACHE.get(key)
    if W is not None:
        return W
    z = np.asarray(geometry.z_cm, dtype=float)
    dz = np.asarray(geometry.length_cm, dtype=float)
    z_edges = np.asarray(geometry.z_edges_cm, dtype=float)
    active = np.asarray(geometry.plasma_active, dtype=bool)
    live = np.flatnonzero(active)
    n = z.size
    W = np.zeros((n, n), dtype=float)
    if live.size:
        sigma = float(sigma_cm)
        z_live = z[live][:, None]
        G = np.exp(-0.5 * ((z_live - z[None, :]) / sigma) ** 2)
        for face in np.asarray(geometry.cathode_face_indices, dtype=int):
            z_face = float(z_edges[face])
            G += np.exp(-0.5 * ((z_live + z[None, :] - 2.0 * z_face) / sigma) ** 2)
        for face in np.asarray(
            getattr(geometry, "mirror_face_indices", ()), dtype=int
        ):
            z_face = float(z_edges[face])
            G += np.exp(-0.5 * ((z_live + z[None, :] - 2.0 * z_face) / sigma) ** 2)
        G *= dz[live][:, None]
        colsum = G.sum(axis=0)
        # A column whose weights all underflow keeps no deposit to rescale; it
        # cannot occur for an active column (which always contains itself).
        G /= np.where(colsum > 0.0, colsum, 1.0)
        W[live, :] = G
    _BEAM_SMOOTH_CACHE[key] = W
    return W


def _smooth_beam_density(W, density, Vp):
    """Apply the conservative matrix to a per-cell density (events/power / cm^3).

    Works in extensive units (density * Vp) so the deposited total is preserved
    exactly; dead cells (Vp <= 0) carry nothing and stay zero.
    """
    Vp = np.asarray(Vp, dtype=float)
    ext = np.asarray(density, dtype=float) * Vp
    ext_s = W @ ext
    out = np.zeros_like(ext_s)
    live = Vp > 0.0
    out[live] = ext_s[live] / Vp[live]
    return out


def _beam_ionization_sources(
    state,
    geometry,
    cathode_solve,
    boundary,
    Te=None,
    n=None,
    smoothing_cm=0.0,
):
    """Return ``(S_beam, S_exc, S_exc_E, beam_power_density)``."""
    zeros = np.zeros(geometry.cells, dtype=float)
    beam_result = cathode_solve.beam_result
    S_beam = zeros.copy()
    S_exc = zeros.copy()
    S_exc_E = zeros.copy()
    beam_power_density = zeros.copy()

    deposition = getattr(cathode_solve, "beam_deposition", None)
    smoothing_cm = float(smoothing_cm)
    if smoothing_cm < 0.0:
        raise ValueError(
            f"beam_deposition_smoothing_cm must be >= 0 (got {smoothing_cm})"
        )
    if not (smoothing_cm > 0.0):
        # CSDA path (B2), historical UNSMOOTHED branch (bit-exact): the module
        # already integrated each ray; convert its per-cell totals to densities.
        # ``beam_power_deposition`` carries the whole per-cell beam energy
        # (heating + radiated + cost) so the separate cost and radiation sinks
        # subtract to the module's net heating, keeping the four-term
        # decomposition meaningful. P_ohmic keeps its historical gap-weighted
        # booking.
        Vp = geometry.plasma_volume_cm3
        for end, dep in deposition.items():
            if dep is None:
                continue
            S_beam += dep.ionization_events / Vp
            S_exc += dep.excitation_events / Vp
            S_exc_E += dep.radiated_erg_s / Vp / ev_to_erg
            beam_power_density += (
                dep.plasma_heating_erg_s
                + dep.radiated_erg_s
                + dep.ionization_cost_erg_s
            ) / Vp
            solver_result = (
                beam_result.result if end == 0 else beam_result.result_twin
            )
            gap = np.asarray(gap_cell_indices(geometry, end=end), dtype=int)
            ohmic_weights = _ohmic_gap_weights(geometry, gap, Te, n)
            beam_power_density[gap] += (
                ohmic_weights * solver_result.P_ohmic * 1.0e7 / Vp[gap]
            )
        return S_beam, S_exc, S_exc_E, beam_power_density
    # CSDA path with conservative deposition smoothing (default-off; the
    # branch above is bit-exact when smoothing is 0). The beam deposition
    # densities are smoothed over a fixed physical width BEFORE the ohmic
    # gap booking is added, so only the beam-range deposition is spread and
    # the totals are conserved (this removes the mesh-scale sheath kick
    # where the beam range crosses a cell boundary).
    Vp = geometry.plasma_volume_cm3
    beam_dep_power = zeros.copy()
    ohmic_power = zeros.copy()
    for end, dep in deposition.items():
        if dep is None:
            continue
        S_beam += dep.ionization_events / Vp
        S_exc += dep.excitation_events / Vp
        S_exc_E += dep.radiated_erg_s / Vp / ev_to_erg
        beam_dep_power += (
            dep.plasma_heating_erg_s
            + dep.radiated_erg_s
            + dep.ionization_cost_erg_s
        ) / Vp
        solver_result = (
            beam_result.result if end == 0 else beam_result.result_twin
        )
        gap = np.asarray(gap_cell_indices(geometry, end=end), dtype=int)
        ohmic_weights = _ohmic_gap_weights(geometry, gap, Te, n)
        ohmic_power[gap] += (
            ohmic_weights * solver_result.P_ohmic * 1.0e7 / Vp[gap]
        )
    W = _beam_smoothing_matrix(geometry, smoothing_cm)
    S_beam = _smooth_beam_density(W, S_beam, Vp)
    S_exc = _smooth_beam_density(W, S_exc, Vp)
    S_exc_E = _smooth_beam_density(W, S_exc_E, Vp)
    beam_dep_power = _smooth_beam_density(W, beam_dep_power, Vp)
    beam_power_density = beam_dep_power + ohmic_power
    return S_beam, S_exc, S_exc_E, beam_power_density


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
        "beam_excitation_radiation": ConservativeState1D(
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
        "cathode_Ts_base_K",
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
    cathode_power_loss_W, anode_power_loss_W, result, cathode_cell, anode_pair,
    state, derived, anode_Te_ref_eV=None,
):
    """Land P_cathode_e and P_anode_e in their OWN per-electrode accumulators.

    ``cathode_power_loss_W`` receives the cathode's electron sheath power at
    the cathode cell; ``anode_power_loss_W`` receives the anode's. They become
    the ``Ee`` rows of ``cathode_surface_loss`` and ``anode_e_sheath_loss``
    respectively, and nothing is added to both.

    The anode collects on both mesh faces, so its sheath power is split between
    the two flanking cells in proportion to each face's Bohm collection -- the
    same weighting ``anode_collection_rhs`` uses, so power and particles are
    removed on the same side.

    WITH NO RESOLVED ANODE the whole of P_anode_e falls back to the CATHODE
    CELL, which is where the lumped model puts it -- but it is still booked to
    ``anode_power_loss_W``, so it appears in the ``anode_e_sheath_loss`` row at
    that cell rather than being folded into the cathode's. The row name is
    about which ELECTRODE paid, not about which cell it landed in; that is why
    the anode row can be nonzero at the cathode cell, and it is the one
    configuration in which the two rows share a cell.

    THERMAL-ONLY ROUTING (A16), unconditional since the legacy
    volumetric-absorber stance was retired (see commit 1fc05c9): only the
    PLASMA-THERMAL part (2Te per electron) is deposited, leaving the
    sheath-fall ``phi`` on the electrode/circuit surface instead of removing it
    from the plasma thermal store.

    THE ANODE FULL DEBIT: add the anode's sheath-fall share
    ``phi_a * I_e_coll`` back onto the plasma electron store, so the ANODE
    debit is the sheath-edge ``(2 Te + phi_a)`` per collected electron while
    the cathode side keeps its thermal-only routing -- at the cathode the
    accelerated species is the ion, so the electron fall there is not
    plasma-electron energy IN THIS FUNCTION's own
    booking. The sibling rows ``end_wall_e_sheath_climb`` and
    ``cathode_e_collected_climb`` -- presence-gated by the geometry's end wall
    face and by its emitting cathode face respectively, and independent of
    this function -- do book a collected-electron fall as plasma-electron energy,
    at the end wall and at the emitting cathode face. ``I_e_coll`` is the
    collected electron current ``I_i_a * fe_a``, and its ``phi_a`` moment is
    the result's own ``P_anode_e_phi``, the complementary member of the R3.2
    split, which rides exactly that flux. The increment is deposited under
    the same split weights, so power and particles still leave on the same
    side. Only the plasma store moves: no circuit or load-ledger quantity is
    read or written here.

    TWO REGIMES, both booked. ``phi_a > 0`` is the electron-REPELLING anode:
    the collected electrons climbed the fall, the plasma paid for it, and the
    increment above applies. ``phi_a <= 0`` is the electron-ATTRACTING anode
    -- ``phi_a`` is solved as ``Lambda_anode - log(1 + J_anode / J_i_a)``, so
    that is exactly the statement that the demanded anode electron current
    has reached or passed electron saturation. There the field does work ON
    the electrons and the BANK is the payer, so the plasma-side debit is the
    thermal ``2 Te`` alone: NO increment is applied. That branch is not
    silent -- ``LAPDSim1D`` counts the accepted steps that take it and records
    the last such time, and exposes both on the cathode diagnostics. A
    non-finite ``phi_a`` belongs to neither regime and raises.

    Composition with the thermal-only routing above, which always runs, so
    the repelling-regime anode deposit is
    ``P_anode_e_thermal + P_anode_e_phi``. The unresolved-cathode fallback
    below this function already deposits the full ``P_anode_e`` and so is
    already on the corrected anode convention.

    ``anode_Te_ref_eV``, when given, records per cell the electron
    temperature the ANODE block of the solve that deposited there ran on
    (``SolverResult.T_e_anode``). It is the reference temperature of the
    linear sink rate the caller builds from this row, and is written at
    exactly the cells the anode power landed in.
    """
    p_cathode_e = result.P_cathode_e_thermal
    p_anode_e = anode_plasma_thermal_power_W(result)
    cathode_power_loss_W[cathode_cell] += p_cathode_e
    if anode_pair is None:
        anode_power_loss_W[cathode_cell] += p_anode_e
        if anode_Te_ref_eV is not None:
            anode_Te_ref_eV[cathode_cell] = float(result.T_e_anode)
        return
    gap_side, column_side = anode_pair
    weights = anode_power_split_weights(state, derived, anode_pair)
    anode_power_loss_W[gap_side] += weights[0] * p_anode_e
    anode_power_loss_W[column_side] += weights[1] * p_anode_e
    if anode_Te_ref_eV is not None:
        anode_Te_ref_eV[gap_side] = float(result.T_e_anode)
        anode_Te_ref_eV[column_side] = float(result.T_e_anode)


def anode_power_split_weights(state, derived, anode_pair):
    """Return the two flanking cells' shares of one anode's sheath power.

    Bohm collection ~ n * c_s, and c_s ~ sqrt(Te/mu) with the same mu on both
    sides, so mu cancels in the normalized split. The ONE definition: the
    deposited row and the implicit sink rate built from it read this, so the
    power the circuit books and the rate the substep applies cannot end up
    split differently.
    """
    gap_side, column_side = anode_pair
    weights = np.array(
        [
            state.n[gap_side] * np.sqrt(derived.Te[gap_side]),
            state.n[column_side] * np.sqrt(derived.Te[column_side]),
        ],
        dtype=float,
    )
    total = weights.sum()
    if not np.isfinite(total) or total <= 0.0:
        return np.full(2, 0.5)
    return weights / total


def anode_plasma_thermal_power_W(result):
    """Return the anode electron power [W] charged to the PLASMA store.

    ``P_anode_e_thermal`` always, plus the sheath-fall share
    ``P_anode_e_phi`` in the REPELLING regime. See
    :func:`_deposit_electrode_power` for the two regimes and why the
    attracting one books the thermal part alone.
    """
    p_anode_e = result.P_anode_e_thermal
    phi_a = float(result.phi_a)
    if not np.isfinite(phi_a):
        # Neither regime: a non-finite sheath potential cannot say who
        # paid the fall, so there is nothing to book either way.
        raise RuntimeError(
            "anode sheath debit: non-finite anode sheath potential "
            f"(phi_a={result.phi_a!r} V); neither the repelling nor the "
            "attracting booking is defined there"
        )
    if phi_a > 0.0:
        # REPELLING anode: phi_a * I_e_coll, the sheath-fall moment of the
        # SAME collected electron flux the 2Te part rides.
        p_anode_e = p_anode_e + result.P_anode_e_phi
    # ATTRACTING anode (phi_a <= 0): the bank pays the fall, so the
    # plasma-side debit stays thermal-only. Counted by the caller, never
    # printed.
    return p_anode_e


#: The names of the three cathode-face electron-energy rows
#: :func:`cathode_emission_sheath_power_W` returns, in the order it returns
#: them. Bound to a constant because the solver has to name the same three in
#: three places -- the term dict it seeds, the term dict it fills, and the
#: neutral-energy booking table -- and a typo in any one of them would be a
#: silently missing row rather than an error.
END_SHEATH_CATHODE_ROWS = (
    "cathode_e_emitted_enthalpy",
    "cathode_e_emitted_fall",
    "cathode_e_collected_climb",
)


def cathode_emission_sheath_power_W(result, T_s_K):
    """Return the emitting face's three electron-energy powers [W].

    The emitting cathode face's sheath-edge closure: what the plasma
    ELECTRON store gains and loses at an emitting surface, over and above the
    ``2 Te`` per collected electron ``P_cathode_e_thermal`` already books and
    the net-``phi_c`` beam energy the deposition march already distributes.
    Returned in the order of :data:`END_SHEATH_CATHODE_ROWS`, each signed as
    a contribution to the plasma electron store (positive = heating):

    ``+2 k_B T_s Gamma_em``
        The enthalpy the released electrons carry into the plasma. They leave
        a half-Maxwellian at the surface temperature ``T_s`` [K], so the
        flux-weighted mean energy of the population that clears the virtual
        cathode is ``2 k_B T_s`` at the barrier peak. ``Gamma_em =
        I_eth_star / e`` is the SPACE-CHARGE-RELEASED flux, not the Richardson
        ceiling. Always >= 0.

    ``+e (phi_c_plus - max(phi_c, 0)) Gamma_em``
        The remainder of the fall those same electrons drop through on their
        way from the barrier peak into the plasma. The beam row already
        carries the NET ``phi_c`` and is untouched here, so what is left is
        the part the virtual cathode adds: identically zero while
        ``phi_c_minus = 0`` (there ``phi_c == phi_c_plus``), positive once a
        virtual cathode has formed. Always >= 0.

    ``-e phi_c_plus Gamma_ec``
        The barrier the COLLECTED plasma electrons climbed, taken from their
        own thermal store -- the plasma-pays convention the anode sheath
        debit books, applied to the identical physics at the cathode.
        ``Gamma_ec = I_e_ret / e`` is the returning plasma-electron flux.
        Always <= 0 for a repelling face.

    ``result`` is a cathode circuit ``SolverResult`` and ``T_s_K`` the
    emitter surface temperature [K] the solve was run at (the evolving
    power-balance value). A non-finite potential or current here has no
    booking either way and raises rather than planting a NaN in an energy row.
    """
    I_em = float(result.I_eth_star)
    I_ec = float(result.I_e_ret)
    phi_c_plus = float(result.phi_c_plus)
    phi_c = float(result.phi_c)
    T_s = float(T_s_K)
    for name, value in (
        ("I_eth_star", I_em),
        ("I_e_ret", I_ec),
        ("phi_c_plus", phi_c_plus),
        ("phi_c", phi_c),
        ("T_s_K", T_s),
    ):
        if not np.isfinite(value):
            raise RuntimeError(
                "cathode face sheath debit: the cathode solve returned a "
                f"non-finite {name} ({value!r}); the emitting face's "
                "electron-energy booking is undefined there"
            )
    # k_B T_s as a voltage, so all three rows are one current times one
    # potential and share the elementary charge exactly.
    kT_s_V = _KB_EV_PER_K * T_s
    return (
        2.0 * kT_s_V * I_em,
        (phi_c_plus - max(phi_c, 0.0)) * I_em,
        -phi_c_plus * I_ec,
    )


def _ohmic_gap_weights(geometry, gap, Te, n=None):
    """Return the normalized share of ``P_ohmic`` deposited in each gap cell.

    ``P_cell = j^2 * eta_sp * V_cell``; with the current density uniform along
    the gap this reduces to ``P_cell ~ eta_sp * length``, and Spitzer
    resistivity gives ``eta_sp ~ lnLambda(Te, n) * Te^-3/2``. The weights are
    built from ``spitzer_sigma_par_ohm_cm`` itself, so the deposition profile
    and the gap resistance cannot disagree about the conductivity; the
    normalization divides out its constant prefactor. A single-cell gap
    normalizes to exactly 1.0, so legacy deposition is bit-identical.
    """
    lengths = np.asarray(geometry.length_cm, dtype=float)[gap]
    if Te is None or n is None or gap.size == 1:
        weights = lengths
    else:
        Te_gap = np.maximum(np.asarray(Te, dtype=float)[gap], 1e-30)
        weights = lengths / spitzer_sigma_par_ohm_cm(
            Te_gap, np.asarray(n, dtype=float)[gap]
        )
    total = weights.sum()
    if not np.isfinite(total) or total <= 0.0:
        return np.full(gap.size, 1.0 / gap.size)
    return weights / total


def _solver_result_metadata(result):
    """Return the per-solve scalars carried on the cathode solve's metadata.

    POTENTIALS, CURRENTS AND BEAM GEOMETRY ONLY. The five power entries this
    once carried -- ``P_prim``, ``P_ohmic``, ``P_loss``, ``P_cathode_e``,
    ``P_anode_e`` -- are gone: nothing read them, one of them (``P_loss``) was
    a pre-closure scalar that is no longer exported at all, and a metadata dict
    is the wrong place to keep a second copy of numbers the cathode
    diagnostics already write per save. Read the powers off
    ``cathode_diagnostics`` instead, where the closed audit set sits beside
    them.
    """
    if result is None:
        return None
    keys = (
        "phi_c",
        "phi_a",
        "V_b",
        "I_i",
        "I_eth_star",
        "I_tot",
        "beam_bypass_fraction",
        "l_b",
    )
    metadata = {key: float(getattr(result, key)) for key in keys}
    metadata["regime"] = result.regime
    metadata["long_mfp"] = bool(result.long_mfp)
    return metadata
