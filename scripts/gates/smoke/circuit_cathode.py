"""Smoke cases: the cathode sheath solve, the discharge circuit, the anode and
the electrode sample.
"""

import json
import math
from pathlib import Path
import shutil
import tempfile

import h5py
import numpy as np

from cablp.cathode import (
    beam_deposition as _beam_deposition_mod,
    circuit_idriven as _cathode_solver_idriven_mod,
    circuit_common as _cathode_solver_mod,
)
from cablp.constants import I_ion, ev_to_erg, m_He_cgs, qe_SI
from cablp.solvers._sim1d import (
    KINETIC_DVM_INCOMPATIBLE_DEFAULTS,
    LAPDSim1D,
    default_config,
    load_result_hdf5,
    summarize_result,
)
from cablp.solvers._sim1d.core.geometry import (
    absorbing_live_cells_by_role,
    puff_cell_indices,
)
from cablp.solvers._sim1d.core.integrator import ssprk2_step
from cablp.solvers._sim1d.core.state import (
    ConservativeState1D,
    STATE_NAMES_1D,
    derive_state,
    pack_state,
)
from cablp.solvers._sim1d.physics.cathode import (
    cathode_emission_sheath_power_W,
    cathode_sample_indices,
)
from cablp.solvers._sim1d.physics.conduction import conductive_face_flux
from cablp.solvers._sim1d.solver import END_SHEATH_CATHODE_ROWS
from cablp.solvers._sim1d.physics.neutrals import (
    gas_puff_rate_profile,
    neutral_exchange_coefficients,
    neutral_thermal_speed,
    neutral_zone_volumes,
)
from cablp.solvers._sim1d.physics.sources import (
    add_state_rhs,
    velocity_divergence,
)
from cablp.solvers._sim1d.results import restart as _restart_mod

from ._harness import (
    _CAPFIX_ESCAPE_CONFIG,
    _CAPFIX_ESCAPE_I_A,
    _CAPFIX_ESCAPE_KWARGS,
    _CAPFIX_ESCAPE_PLASMA,
    _anode_sink_config,
    _anode_sink_sim,
    _base_config,
    _base_sim,
    _case,
    _cathode_flags,
    _cathode_unit_config,
    _pin_pre_r2a_neutral_stance,
    _resolved_cathode_flags,
    _resolved_config,
    _resolved_geometry,
)


# The currents that bracket the escape window of the frozen state; see
# ``_CAPFIX_ESCAPE_I_A`` in ``_harness``.
_CAPFIX_BELOW_I_A = (5.0, 5.45, 5.46)
_CAPFIX_WINDOW_I_A = (5.47, 5.5, _CAPFIX_ESCAPE_I_A, 5.57, 5.58, 6.0, 8.0)


# --------------------------------------------------------------------
# twin-cathode-plateau-multigroup
# --------------------------------------------------------------------
@_case(
    "twin-cathode-plateau-multigroup",
    historical_stance=True,
)
def _case_twin_cathode_plateau_multigroup(
    twin_base_flags, twin_base_params
):
    # PRESENCE GATE for the plateau-edge pair, BOTH DIRECTIONS, on the twin
    # layout. The twin fixture in variable-area-well-balancedness resolves
    # heating_anomalous_transport="local", so the equivalence it asserts there
    # can only ever exercise the ABSENT branch: a build that stopped emitting
    # the pair entirely would still satisfy it. This case arms the closure and
    # asserts the pair is PRESENT under BOTH cathode prefixes, then clears it
    # and asserts it is present under NEITHER -- the same two rows, the same
    # two prefixes, one fixture apart.
    #
    # The selector's accepted values are stated by the solver's own validator:
    # "heating_anomalous_transport must be 'local' or 'plateau_multigroup'".
    # Every other dial the armed arm requires is already a config default
    # (beam_anomalous_model="quasilinear") -- the exception is the cathode
    # boundary, which the
    # twin's own refusal decides and which is asserted below rather than
    # assumed.
    twin_flags = dict(twin_base_flags)
    twin_flags["TwinCathode"] = True
    twin_flags["cathode_coupling"] = False
    _mg_rows = ("beam_plateau_edge_eV", "beam_plateau_edge_clamped")

    # 'reflect' -- the shipped default -- is REFUSED with two cathodes, so the
    # armed fixture states 'escape'. Asserted, not assumed: if that refusal
    # ever moved, the fixture below would be selecting a boundary for a reason
    # that no longer exists.
    mg_reflect_params = dict(
        twin_base_params,
        heating_anomalous_transport="plateau_multigroup",
        heating_anomalous_tail_cathode_boundary="reflect",
    )
    try:
        LAPDSim1D(mg_reflect_params, twin_flags)
    except ValueError as exc:
        assert "does not support TwinCathode" in str(exc)
    else:
        raise AssertionError(
            "a reflecting cathode boundary constructed on the twin layout"
        )

    mg_params = dict(
        twin_base_params,
        heating_anomalous_transport="plateau_multigroup",
        heating_anomalous_tail_cathode_boundary="escape",
    )
    mg_sim = LAPDSim1D(mg_params, twin_flags)
    assert mg_sim._plateau_multigroup
    mg_diag = mg_sim._cathode_diagnostic_snapshot()
    for _mg_prefix in ("source", "end"):
        for _mg_row in _mg_rows:
            assert f"{_mg_prefix}_{_mg_row}" in mg_diag, (_mg_prefix, _mg_row)

    # The OFF arm of the same fixture: identical geometry and identical flags,
    # the selector alone cleared, and the pair is gone from both prefixes.
    local_params = dict(
        twin_base_params,
        heating_anomalous_transport="local",
        heating_anomalous_tail_cathode_boundary="escape",
    )
    local_sim = LAPDSim1D(local_params, twin_flags)
    assert not local_sim._plateau_multigroup
    local_diag = local_sim._cathode_diagnostic_snapshot()
    for _mg_prefix in ("source", "end"):
        for _mg_row in _mg_rows:
            assert f"{_mg_prefix}_{_mg_row}" not in local_diag, (
                _mg_prefix, _mg_row
            )
    # The rest of the per-end cathode block is present either way, so the
    # absence just asserted is the multi-group pair's own and not a twin whose
    # end block failed to be seeded at all.
    for _mg_prefix in ("source", "end"):
        assert f"{_mg_prefix}_regime" in local_diag
        assert f"{_mg_prefix}_regime" in mg_diag


# --------------------------------------------------------------------
# cathode-spitzer-and-base-boundary
# --------------------------------------------------------------------
@_case(
    "cathode-spitzer-and-base-boundary",
    historical_stance=True,
    provides=("_warnings", "dt_default", "neutral_phase_params"),
)
def _case_cathode_spitzer_and_base_boundary(cathode_face):
    params, flags = _base_config()
    resolved_params, resolved_flags = _resolved_config()
    sim, snapshot = _base_sim()
    geom = snapshot.geometry
    state = snapshot.state
    derived = snapshot.derived
    resolved_geom = _resolved_geometry()
    resolved_cathode_flags = _resolved_cathode_flags()
    import warnings as _warnings

    from cablp.cathode.circuit_common import c_log_ei
    from cablp.solvers._sim1d.physics.cathode import spitzer_sigma_par_ohm_cm

    # sigma_par carries a state-dependent Coulomb logarithm: the NRL
    # transverse resistivity (p.30) lifted to parallel by the Braginskii 1.96
    # (p.38).
    assert np.isclose(
        spitzer_sigma_par_ohm_cm(4.0, 4.0e12),
        (1.96 / (1.03e-2 * c_log_ei(4.0, 4.0e12))) * 4.0**1.5,
        rtol=0.0,
        atol=0.0,
    )
    # lnLambda LITERAL PIN, against an independent literal, so a
    # transcription error shared by the helper and its consumers cannot
    # cancel out of a comparison built from both.
    assert np.isclose(c_log_ei(3.0, 4.0e12), 10.1392, rtol=1e-5), c_log_ei(
        3.0, 4.0e12
    )
    assert np.isclose(c_log_ei(12.0, 4.0e12), 11.9762, rtol=1e-5), c_log_ei(
        12.0, 4.0e12
    )

    # Knudsen neutral transport is mesh-independent.
    knudsen_D = []
    for knudsen_nx in (60, 185):
        knudsen_params = dict(resolved_params)
        knudsen_params["nx"] = knudsen_nx
        knudsen_geom = LAPDSim1D(
            knudsen_params, resolved_flags
        ).get_initial_snapshot().geometry
        mid = knudsen_geom.cells // 2
        coeff = neutral_exchange_coefficients(
            geometry=knudsen_geom,
                        Tn_K=knudsen_params["Tn_K"],
            mu_neutral=4,
            clausing_scale=1.0,
        )
        knudsen_D.append(
            coeff[mid]
            * knudsen_geom.length_cm[mid]
            / knudsen_geom.neutral_area_cm2[mid]
        )
    # Knudsen: identical diffusivity at 30.8 cm and 10 cm cells, and it equals the
    # physical free-molecular value (2/3)*v_th*R.
    assert np.isclose(knudsen_D[0], knudsen_D[1], rtol=1e-12)
    expected_D = (
        (2.0 / 3.0)
        * neutral_thermal_speed(Tn_K=resolved_params["Tn_K"], mu_neutral=4)
        * resolved_params["Rm"]
    )
    assert np.isclose(knudsen_D[0], expected_D, rtol=1e-12)

    # M3: no parallel heat conduction crosses a cathode surface into the plenum.
    resolved_q = conductive_face_flux(
        temperature=np.linspace(5.0, 1.0, resolved_geom.cells),
        conductivity=np.full(resolved_geom.cells, 1.0e5),
        geometry=resolved_geom,
    )
    assert resolved_q[cathode_face] == 0.0
    assert np.isfinite(resolved_q).all()

    for values in (
        state.n,
        state.nn,
        state.M,
        state.Ee,
        state.Ei,
        derived.u,
        derived.Te,
        derived.Ti,
        derived.pe,
        derived.pi,
        derived.p,
    ):
        assert np.all(np.isfinite(values))

    neutral_coeff = sim.neutral_exchange_coefficients()
    assert neutral_coeff.shape == (geom.cells - 1,)
    assert np.all(np.isfinite(neutral_coeff))
    assert np.all(neutral_coeff >= 0.0)
    assert np.any(neutral_coeff > 0.0)

    dt_default = sim.suggest_timestep()
    assert np.isfinite(dt_default.dt)
    assert dt_default.dt > 0.0
    assert dt_default.dt <= params["dt_max"]
    assert dt_default.dt >= params["dt_min"]
    assert np.isclose(dt_default.time, 0.0)
    assert dt_default.phase == "pre_breakdown"
    assert dt_default.phase_cathode_enabled == 0.0
    assert dt_default.phase_gas_puff_enabled == 1.0
    assert dt_default.phase_floating == 0.0
    assert dt_default.active_constraint in {
        "plasma_cfl",
        "surface_loss",
        "neutral_exchange",
        "neutral_sources",
        "reactions",
        "energy_exchange",
        "electron_cooling",
        "ion_charge_exchange",
        "heat_conduction",
        "dt_max",
    }
    # The clamp is no longer a constraint NAME (2026-08-05): it is carried by
    # clamped_to_dt_min, so "dt_min" is not an admissible label any more.
    assert dt_default.clamped_to_dt_min == 0.0
    assert dt_default.dt_raw >= params["dt_min"]
    assert np.isfinite(dt_default.dt_neutral_sources)
    assert dt_default.dt_surface_loss > 0.0
    assert dt_default.dt_reactions > 0.0
    assert dt_default.dt_energy_exchange > 0.0
    assert dt_default.dt_electron_cooling > 0.0
    assert dt_default.dt_ion_charge_exchange > 0.0
    assert dt_default.dt_heat_conduction > 0.0

    cathode_boundary = sim.cathode_boundary_state()
    assert not cathode_boundary.enabled
    assert cathode_boundary.source.index == cathode_face
    assert cathode_boundary.source.role == "cathode"
    assert cathode_boundary.end.index == geom.cells - 1
    assert cathode_boundary.end.role == "end_wall"
    assert cathode_boundary.twin_cathode == flags["TwinCathode"]
    for key in (
        "V_bank",
        "cathode_Ts_base_K",
        "phi_wf",
        "C_R",
        "R_comp",
        "eta",
        "L_cath",
        "R_cath",
    ):
        assert key in params
        assert key in cathode_boundary.circuit
        assert np.isfinite(cathode_boundary.circuit[key])
        assert np.isclose(cathode_boundary.circuit[key], params[key])
    for cell in (cathode_boundary.source, cathode_boundary.end):
        for value in (
            cell.n,
            cell.nn,
            cell.Te,
            cell.Ti,
            cell.u,
            cell.plasma_volume_cm3,
            cell.neutral_volume_cm3,
            cell.plasma_area_cm2,
            cell.neutral_area_cm2,
            cell.length_cm,
            cell.Rp_cm,
            cell.Rm_cm,
        ):
            assert np.isfinite(value)
    cathode_terms = sim.cathode_source_terms()
    assert not cathode_terms.enabled
    assert cathode_terms.metadata["source_index"] == cathode_face
    assert cathode_terms.metadata["end_index"] == geom.cells - 1
    for key, value in cathode_boundary.circuit.items():
        assert np.isclose(cathode_terms.metadata["circuit"][key], value)
    for values in (
        cathode_terms.rhs.n,
        cathode_terms.rhs.nn,
        cathode_terms.rhs.M,
        cathode_terms.rhs.Ee,
        cathode_terms.rhs.Ei,
    ):
        assert np.allclose(values, 0.0)
    assert np.allclose(pack_state(cathode_terms.rhs), 0.0)
    disabled_cathode_solve = sim.solve_cathode_boundary()
    assert not disabled_cathode_solve.boundary.enabled
    assert disabled_cathode_solve.beam_result is None
    assert disabled_cathode_solve.device_config is None
    assert disabled_cathode_solve.metadata["enabled"] is False
    assert sim.phase_at_time(0.0) == "pre_breakdown"
    assert sim.phase_at_time(params["tau_prebreakdown"]) == "main_discharge"
    assert (
        sim.phase_at_time(params["tau_prebreakdown"] + params["tau_discharge"])
        == "afterglow"
    )
    assert (
        sim.phase_at_time(
            params["tau_prebreakdown"]
            + params["tau_discharge"]
            + params["tau_afterglow"]
        )
        == "post_afterglow"
    )
    assert np.isclose(
        sim.next_phase_boundary_after(0.0),
        params["tau_prebreakdown"],
    )
    assert np.isclose(
        sim.next_phase_boundary_after(params["tau_prebreakdown"]),
        params["tau_prebreakdown"] + params["tau_discharge"],
    )
    neutral_phase_flags = dict(flags)
    neutral_phase_flags["Plasma"] = False
    neutral_phase_params = dict(params)
    neutral_phase_params["tau_discharge"] = 2.0e-10
    neutral_phase_params["tau_cycle"] = 5.0e-10
    # This block checks the temporal puff SCHEDULE (on/off per phase): the
    # on-minus-off source, summed over both zones in particles per second, is
    # the whole unshaped puff row.
    neutral_phase_sim = LAPDSim1D(neutral_phase_params, neutral_phase_flags)
    assert neutral_phase_sim.phase_at_time(0.0) == "equilibrium_puff"
    assert neutral_phase_sim.phase_at_time(3.0e-10) == "equilibrium_off"
    assert np.isclose(neutral_phase_sim.next_phase_boundary_after(0.0), 2.0e-10)
    assert np.isclose(
        neutral_phase_sim.next_phase_boundary_after(2.0e-10),
        5.0e-10,
    )
    neutral_puff_source = neutral_phase_sim.neutral_source_sink_rhs(time=0.0)
    neutral_off_source = neutral_phase_sim.neutral_source_sink_rhs(time=3.0e-10)
    neutral_geom = neutral_phase_sim.get_initial_snapshot().geometry
    neutral_puff_cell, _ = puff_cell_indices(neutral_geom)
    neutral_Vc, neutral_Va = neutral_zone_volumes(neutral_geom)
    neutral_puff_particles = (
        (neutral_puff_source.nn - neutral_off_source.nn) * neutral_Vc
        + (neutral_puff_source.nn_a - neutral_off_source.nn_a) * neutral_Va
    )
    assert neutral_puff_particles[neutral_puff_cell] > 0.0
    assert np.allclose(
        neutral_puff_particles,
        gas_puff_rate_profile(
            neutral_geom,
            neutral_phase_params["S_gp"],
            neutral_phase_params["gas_puff_valves"],
            z_cm=neutral_phase_params["gas_puff_z_cm"],
            orifice_id_cm=neutral_phase_params["gas_puff_orifice_id_cm"],
            orifice_length_cm=neutral_phase_params["gas_puff_orifice_length_cm"],
        )
        * np.asarray(neutral_geom.neutral_volume_cm3, dtype=float),
        rtol=1e-12,
        atol=0.0,
    )
    assert sim.phase_switches_at_time(0.0) == {
        "cathode_enabled": False,
        "gas_puff_enabled": True,
        "floating": False,
    }
    # The square valve keeps delivering through its closing tail, so the
    # afterglow leaves the puff switch open; the envelope closes the flow.
    assert sim.phase_switches_at_time(
        params["tau_prebreakdown"] + params["tau_discharge"]
    ) == {
        "cathode_enabled": False,
        "gas_puff_enabled": True,
        "floating": True,
    }
    assert neutral_phase_sim.phase_switches_at_time(0.0) == {
        "cathode_enabled": False,
        "gas_puff_enabled": True,
        "floating": False,
    }
    assert neutral_phase_sim.phase_switches_at_time(3.0e-10) == {
        "cathode_enabled": False,
        "gas_puff_enabled": False,
        "floating": False,
    }

    cathode_flags = _cathode_flags()
    return locals()


# --------------------------------------------------------------------
# cathode-boundary-beam-terms
# --------------------------------------------------------------------
@_case(
    "cathode-boundary-beam-terms",
    historical_stance=True,
    provides=(
        "beam_birth_terms", "cathode_sim",
        "cathode_solve", "split_beam_terms",
    ),
)
def _case_cathode_boundary_beam_terms(cathode_face):
    # The cathode boundary + beam-ionization bookkeeping on the CSDA beam.
    params, flags = _base_config()
    sim, snapshot = _base_sim()
    geom = snapshot.geometry
    cathode_flags = _cathode_flags()
    cathode_sim = LAPDSim1D(params, cathode_flags)
    cathode_sim._circuit_I_loop = 3000.0
    cathode_solve = cathode_sim.solve_cathode_boundary()
    assert cathode_solve.boundary.enabled
    assert cathode_solve.device_config is not None
    assert cathode_solve.beam_result is not None
    assert cathode_solve.metadata["enabled"] is True
    assert cathode_solve.metadata["floating"] is False
    assert cathode_solve.metadata["result_twin"] is None
    assert np.isclose(cathode_solve.device_config.R_cath, params["R_cath"])
    assert np.isclose(
        cathode_solve.device_config.A_c,
        np.pi * params["R_cath"] ** 2,
    )
    assert np.isfinite(cathode_solve.x0_next)
    assert cathode_solve.x0_twin_next is None
    assert cathode_solve.beam_result.result.I_tot > 0.0
    assert cathode_solve.beam_result.result.phi_c > params["Te0"]
    assert cathode_solve.beam_result.v_beam.shape == (geom.cells,)
    assert cathode_solve.beam_result.n_beam.shape == (geom.cells,)
    assert cathode_solve.beam_result.beam_cross.shape == (geom.cells,)
    assert np.all(np.isfinite(cathode_solve.beam_result.v_beam))
    assert np.all(np.isfinite(cathode_solve.beam_result.n_beam))
    assert np.all(np.isfinite(cathode_solve.beam_result.beam_cross))
    assert cathode_solve.beam_result.v_beam[cathode_face] > 0.0
    assert cathode_solve.beam_result.n_beam[cathode_face] > 0.0
    assert cathode_solve.beam_result.beam_cross[cathode_face] > 0.0
    cached_cathode_solve = cathode_sim.solve_cathode_boundary()
    assert np.isclose(
        cached_cathode_solve.metadata["result"]["I_tot"],
        cathode_solve.metadata["result"]["I_tot"],
    )
    afterglow_time = params["tau_prebreakdown"] + params["tau_discharge"]
    floating_cathode_solve = cathode_sim.solve_cathode_boundary(
        time=afterglow_time,
        update_cache=False,
    )
    assert floating_cathode_solve.boundary.enabled
    assert floating_cathode_solve.metadata["enabled"] is True
    assert floating_cathode_solve.metadata["floating"] is True
    assert floating_cathode_solve.beam_result is not None
    # ROW 1 ENFORCEMENT: overriding floating=False in a phase that IS
    # floating (asserted above) is now a loud refusal at
    # _effective_cathode_flags, not the silent "boundary.enabled=False"
    # reading this case asserted before the enforcement landed -- that
    # silent reading was exactly the class of mis-booking the enforcement
    # closes (this override handed a configuration that does not exist).
    _inactive_afterglow_raised = None
    try:
        cathode_sim.solve_cathode_boundary(
            time=afterglow_time,
            floating=False,
            update_cache=False,
        )
    except ValueError as exc:
        _inactive_afterglow_raised = exc
    assert _inactive_afterglow_raised is not None, (
        "solve_cathode_boundary(floating=False) in a floating phase must "
        "refuse under the _effective_cathode_flags enforcement"
    )
    assert "active_only=False, floating=False" in str(
        _inactive_afterglow_raised
    ), str(_inactive_afterglow_raised)
    post_afterglow_solve = cathode_sim.solve_cathode_boundary(
        time=afterglow_time + params["tau_afterglow"],
        update_cache=False,
    )
    assert not post_afterglow_solve.boundary.enabled
    cathode_loss_terms = cathode_sim.cathode_source_terms(cathode_solve=cathode_solve)
    assert cathode_loss_terms.enabled
    assert np.all(np.isfinite(pack_state(cathode_loss_terms.rhs)))
    # Both electrode rows come out of the one solve, and their supports are
    # disjoint here (the anode is resolved), which is the property that makes
    # their sum bit-exactly the single row they replaced.
    assert np.all(np.isfinite(pack_state(cathode_loss_terms.anode_rhs)))
    _cath_Ee = np.asarray(cathode_loss_terms.rhs.Ee, dtype=float)
    _an_Ee = np.asarray(cathode_loss_terms.anode_rhs.Ee, dtype=float)
    assert not np.any((_cath_Ee != 0.0) & (_an_Ee != 0.0))
    for _zero_field in ("n", "nn", "M", "Ei"):
        assert np.allclose(
            getattr(cathode_loss_terms.anode_rhs, _zero_field), 0.0
        )
    # THE ELECTRODE ROWS ARE CONTINUOUS ACROSS THE HAND-OFF. The open-circuit
    # afterglow is the current-driven solve read at I_tot = 0, so the same
    # formulas are evaluated there rather than the rows dropping to zero
    # because the phase changed name: the terms stay ENABLED, the face still
    # loses its Bohm flux and still books the electron sheath it collects.
    afterglow_cathode_loss_terms = cathode_sim.cathode_source_terms(
        cathode_solve=floating_cathode_solve, time=afterglow_time
    )
    assert afterglow_cathode_loss_terms.enabled
    assert np.all(np.isfinite(pack_state(afterglow_cathode_loss_terms.rhs)))
    assert np.all(
        np.isfinite(pack_state(afterglow_cathode_loss_terms.anode_rhs))
    )
    assert not np.allclose(pack_state(afterglow_cathode_loss_terms.rhs), 0.0)
    # Same row structure as the driven phase: disjoint Ee supports, and the
    # anode row carries Ee alone.
    _ag_cath_Ee = np.asarray(afterglow_cathode_loss_terms.rhs.Ee, dtype=float)
    _ag_an_Ee = np.asarray(
        afterglow_cathode_loss_terms.anode_rhs.Ee, dtype=float
    )
    assert not np.any((_ag_cath_Ee != 0.0) & (_ag_an_Ee != 0.0))
    for _zero_field in ("n", "nn", "M", "Ei"):
        assert np.allclose(
            getattr(afterglow_cathode_loss_terms.anode_rhs, _zero_field), 0.0
        )
    beam_birth_terms = cathode_sim.beam_ionization_rhs(
        cathode_solve=cathode_solve,
    )
    split_beam_terms = cathode_sim.beam_ionization_rhs_terms(
        cathode_solve=cathode_solve,
    )
    assert set(split_beam_terms) == {
        "beam_ionization_birth",
        "beam_power_deposition",
        "beam_ionization_cost",
        "beam_excitation_radiation",
    }
    # beam_ionization_rhs is the birth, power-deposition and cost rows; the
    # excitation radiation is a fourth row of its own.
    # Compared row by row on the five base fields; the split rows also carry
    # the annulus row the two-zone state packs, which the beam never touches
    # (it ionizes column gas), so it must be identically zero there.
    for split_field in STATE_NAMES_1D:
        split_beam_sum = np.zeros_like(
            np.asarray(getattr(beam_birth_terms, split_field), dtype=float)
        )
        for split_name, split_term in split_beam_terms.items():
            if split_name == "beam_excitation_radiation":
                continue
            split_beam_sum = split_beam_sum + getattr(split_term, split_field)
        assert np.allclose(
            split_beam_sum, getattr(beam_birth_terms, split_field)
        ), split_field
    for split_name, split_term in split_beam_terms.items():
        if getattr(split_term, "nn_a", None) is not None:
            assert np.all(np.asarray(split_term.nn_a) == 0.0), split_name
    assert np.all(beam_birth_terms.n >= 0.0)
    assert np.any(beam_birth_terms.n > 0.0)
    assert np.all(beam_birth_terms.nn <= 0.0)
    assert np.allclose(beam_birth_terms.M, 0.0)
    assert np.all(beam_birth_terms.Ei >= 0.0)
    assert np.allclose(
        split_beam_terms["beam_ionization_birth"].n,
        beam_birth_terms.n,
    )
    assert np.allclose(
        split_beam_terms["beam_ionization_birth"].nn,
        beam_birth_terms.nn,
    )
    assert np.allclose(split_beam_terms["beam_ionization_birth"].Ee, 0.0)
    assert np.all(split_beam_terms["beam_power_deposition"].Ee >= 0.0)
    assert np.any(split_beam_terms["beam_power_deposition"].Ee > 0.0)
    assert np.all(split_beam_terms["beam_ionization_cost"].Ee <= 0.0)
    assert np.any(split_beam_terms["beam_ionization_cost"].Ee < 0.0)
    for zero_particle_term in (
        split_beam_terms["beam_power_deposition"],
        split_beam_terms["beam_ionization_cost"],
    ):
        assert np.allclose(zero_particle_term.n, 0.0)
        assert np.allclose(zero_particle_term.nn, 0.0)
        assert np.allclose(zero_particle_term.M, 0.0)
        assert np.allclose(zero_particle_term.Ei, 0.0)
    # Under the two-zone split nn is the COLUMN density, booked on V_col.
    beam_Vc, _beam_Va = neutral_zone_volumes(geom)
    beam_inventory_scale = np.sum(
        np.abs(beam_birth_terms.n * geom.plasma_volume_cm3)
        + np.abs(beam_birth_terms.nn * beam_Vc)
    )
    assert np.isclose(
        math.fsum(
            (
                beam_birth_terms.n * geom.plasma_volume_cm3
                + beam_birth_terms.nn * beam_Vc
            ).tolist()
        ),
        0.0,
        atol=1e-12 * beam_inventory_scale,
    )
    return locals()


# --------------------------------------------------------------------
# cathode-annular-solve-fixtures
# --------------------------------------------------------------------
@_case(
    "cathode-annular-solve-fixtures",
    provides=(
        "PlasmaState", "gauss_cfg", "hot_cfg",
        "one_annulus", "plasma_probe", "uni_cfg",
    ),
)
def _case_cathode_annular_solve_fixtures():
    # --- Device fixtures for the sheath-solve cases below. The model's
    # emitting face is the uniform disc; the annular device configurations
    # here exercise the circuit layer's annular emission path directly. A
    # single warm annulus at the plasma footprint must reproduce the uniform
    # solve.
    sim, snapshot = _base_sim()
    from cablp.cathode.circuit_common import PlasmaState
    from cablp.cathode.circuit_idriven import solve_idriven
    from cablp.solvers._sim1d.physics.cathode import cathode_device_config
    import dataclasses as _dc

    knee_params, knee_flags = default_config()
    knee_params.update({"V_bank": 173.6, "R_comp": 5.72e-3,
                        "cathode_Ts_base_K": 2008.0,
                        "R_cath": 15.0, "Rp": 15.0,
                        "phi_wf": 3.0})
    uni_cfg = cathode_device_config(
        knee_params, knee_flags, sim.mu, sim.ion_mass_g
    )
    plasma_probe = PlasmaState(T_e=8.0, n_e=4e12, n_n=1.5e13, sigma_b=4e-17)
    # single annulus, fully wetted, at T_s: identical emission physics
    one_annulus = _dc.replace(
        uni_cfg,
        emission_Ts_K=(knee_params["cathode_Ts_base_K"],),
        emission_area_cm2=(uni_cfg.A_c,),
        emission_plasma_frac=(1.0,),
    )
    # The uniform-disc branch and the annular emission state
    # (``circuit_common.annular_emission_state``) on one annulus, at the same
    # imposed current: the same sheath.
    r_uni = solve_idriven(uni_cfg, plasma_probe, I_tot_A=2823.3497327720015)
    r_one = solve_idriven(
        one_annulus, plasma_probe, I_tot_A=2823.3497327720015
    )
    assert np.isclose(r_one.I_tot, r_uni.I_tot, rtol=1e-10)
    assert np.isclose(r_one.phi_c, r_uni.phi_c, rtol=1e-10)
    assert np.isclose(one_annulus.I_eth, uni_cfg.I_eth, rtol=1e-12)

    def _annular_cfg(T_s, R_cath=19.0, Rp=15.0, fwhm=28.0, n_annuli=10):
        """A ten-annulus device whose Richardson footprint is a gaussian of
        the given FWHM, each annulus at the surface temperature that
        footprint implies, with its overlap with the plasma radius ``Rp``."""
        kB_over_e = 8.617333262e-5
        edges = np.linspace(0.0, R_cath, n_annuli + 1)
        Ts_k, area_k, frac_k = [], [], []
        for r0, r1 in zip(edges[:-1], edges[1:]):
            r_mid = 0.5 * (r0 + r1)
            ln_j = -4.0 * math.log(2.0) * r_mid**2 / fwhm**2
            Ts_k.append(1.0 / (1.0 / T_s - (kB_over_e / 3.0) * ln_j))
            area_k.append(math.pi * (r1**2 - r0**2))
            if r1 <= Rp:
                frac_k.append(1.0)
            elif r0 >= Rp:
                frac_k.append(0.0)
            else:
                frac_k.append((Rp**2 - r0**2) / (r1**2 - r0**2))
        base = cathode_device_config(
            dict(knee_params, R_cath=R_cath, cathode_Ts_base_K=T_s),
            knee_flags, sim.mu, sim.ion_mass_g,
        )
        return _dc.replace(
            base,
            emission_Ts_K=tuple(Ts_k),
            emission_area_cm2=tuple(area_k),
            emission_plasma_frac=tuple(frac_k),
        )

    gauss_cfg = _annular_cfg(2008.0)
    assert np.all(np.diff(gauss_cfg.emission_Ts_K) < 0.0)
    assert gauss_cfg.I_eth < uni_cfg.I_eth * (np.pi * 19.0**2) / uni_cfg.A_c
    hot_cfg = _annular_cfg(2110.0)
    return locals()


# --------------------------------------------------------------------
# cathode-current-driven-sheath-solve
# --------------------------------------------------------------------
@_case(
    "cathode-current-driven-sheath-solve",
    provides=(
        "_cap_beam", "_cap_cfg", "_cap_pl", "id_ceiling", "id_grid",
        "id_plasmas", "solve_idriven",
    ),
)
def _case_cathode_current_driven_sheath_solve(
    PlasmaState, gauss_cfg, hot_cfg, one_annulus, plasma_probe, uni_cfg
):
    # --- Current-driven sheath solve: over a sweep of device fixtures and
    # plasma states, the imposed current is carried through the monotone
    # device relation -- the reported I_tot recovers it and V_b is the device
    # voltage -- in both the classical and the virtual-cathode regime. The imposed currents are literals: each
    # is the operating point a sheath/Thevenin load-line solve reached at
    # that state, so the sweep visits real operating points in both regimes.
    from cablp.cathode.circuit_idriven import solve_idriven

    id_plasmas = (
        plasma_probe,
        PlasmaState(T_e=3.0, n_e=5.0e11, n_n=2.0e13, sigma_b=0.0),
        PlasmaState(T_e=12.0, n_e=1.0e13, n_n=5.0e12, sigma_b=4e-17),
    )
    # (device, plasma index, imposed I_tot [A], regime). The degenerate
    # gauss_cfg / plasma-1 corner is covered by its own test below, where psi
    # is not recoverable from I within float precision.
    id_sweep = (
        (uni_cfg, 0, 2823.3497327720015, "classical"),
        (uni_cfg, 1, 2058.350165559722, "virtual_cathode"),
        (uni_cfg, 2, 3610.051981749392, "classical"),
        (one_annulus, 0, 2823.3497327720015, "classical"),
        (one_annulus, 1, 2058.350165559722, "virtual_cathode"),
        (one_annulus, 2, 3610.051981749392, "classical"),
        (gauss_cfg, 0, 2262.7823999747234, "virtual_cathode"),
        (gauss_cfg, 2, 3525.011001921395, "virtual_cathode"),
        (hot_cfg, 0, 4817.624297235924, "virtual_cathode"),
        (hot_cfg, 1, 3265.2909611092355, "virtual_cathode"),
        (hot_cfg, 2, 6079.781162732751, "virtual_cathode"),
    )
    id_regimes = set()
    for id_cfg, id_k, id_I, id_regime in id_sweep:
        ri = solve_idriven(id_cfg, id_plasmas[id_k], I_tot_A=id_I)
        id_regimes.add(ri.regime)
        assert ri.regime == id_regime, (id_k, ri.regime, id_regime)
        assert np.isclose(ri.I_tot, id_I, rtol=1e-8), (id_k, ri.I_tot, id_I)
        # V_b contract: the I-driven V_b is the device voltage.
        id_v_dev = ri.phi_c + ri.V_p - ri.phi_a
        assert np.isclose(ri.V_b, id_v_dev, rtol=1e-8, atol=1e-8)
    assert {"classical", "virtual_cathode"} <= id_regimes

    # Degenerate emission-exhausted plateau (the I-driven formulation's weak
    # spot): at this corner every annulus is released and the electron tail
    # has underflowed, so J_tot(psi) is numerically constant -- psi is NOT
    # recoverable from I alone. The solve must stay deterministic
    # (leading-edge selection), reproduce the *currents*, and never raise.
    # The current literals are the load-line operating point at this corner.
    id_deg_I = 1697.455580536431
    id_deg_ri = solve_idriven(gauss_cfg, id_plasmas[1], I_tot_A=id_deg_I)
    assert id_deg_ri.regime in ("virtual_cathode", "capability_limited")
    assert np.isfinite(id_deg_ri.phi_c) and np.isfinite(id_deg_ri.V_b)
    assert np.isclose(id_deg_ri.I_tot, id_deg_I, rtol=1e-8)
    assert np.isclose(id_deg_ri.I_eth_star, 1650.5947226668688, rtol=1e-6)
    id_deg_repeat = solve_idriven(gauss_cfg, id_plasmas[1], I_tot_A=id_deg_I)
    assert id_deg_repeat.phi_c == id_deg_ri.phi_c  # deterministic

    # Monotone by construction: deeper sheath carries more current, so the
    # inverse map I -> phi is single-valued and increasing.
    id_ref = solve_idriven(uni_cfg, plasma_probe, I_tot_A=2823.3497327720015)
    id_ceiling = id_ref.I_i + id_ref.I_eth
    id_grid = np.linspace(10.0, 0.98 * id_ceiling, 25)
    id_phis = [
        solve_idriven(uni_cfg, plasma_probe, I_tot_A=float(I)).phi_c_plus
        for I in id_grid
    ]
    assert np.all(np.diff(id_phis) > 0.0)

    # Capability-limited: an imposed current beyond the sheath's ceiling
    # returns the bracket-top solution, tagged, finite, at a large V_b --
    # no exception, no fallback ladder (the M3 circuit ramps I down ~V/L).
    id_cap = solve_idriven(
        uni_cfg, plasma_probe, I_tot_A=1.05 * id_ceiling
    )
    assert id_cap.regime == "capability_limited"
    assert np.isfinite(id_cap.V_b) and id_cap.V_b > id_ref.V_b
    # The kick is always a back-EMF >= the ceiling and carries a
    # non-negative current -- the clamp that prevents the measured
    # capability-runaway (negative V_b read as forward EMF).
    assert id_cap.V_b >= 1000.0
    assert id_cap.I_tot >= 0.0
    # The kick is reported *at* the net-sheath ceiling, not wherever the
    # bracket expansion happened to land.
    assert np.isclose(id_cap.phi_c, 1000.0, rtol=1e-9), id_cap.phi_c
    try:
        solve_idriven(uni_cfg, plasma_probe, I_tot_A=-1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for negative imposed current")

    # The ceiling binds the RETURNED ROOT, not just the ladder's grid points
    # (2026-08-09). At the frozen escaping state the pre-fix ladder doubled
    # once, found the J-root inside the doubled bracket and returned a net
    # phi_c far above a 1000 V cap -- unclamped, tagged
    # virtual_cathode, and independent of the cap. Post-fix: phi_c(I) rises to
    # the cap and stays there, so the whole window is capability_limited AT
    # the ceiling and the escape signature (phi_c above the cap in any other
    # regime) cannot be produced.
    _cap_cfg = _cathode_solver_mod.DeviceConfig(**_CAPFIX_ESCAPE_CONFIG)
    _cap_pl = PlasmaState(**_CAPFIX_ESCAPE_PLASMA)
    _cap_prev = 0.0
    for _cap_I in _CAPFIX_BELOW_I_A:
        _cap_r = solve_idriven(
            _cap_cfg, _cap_pl, I_tot_A=_cap_I, **_CAPFIX_ESCAPE_KWARGS
        )
        assert _cap_r.regime == "virtual_cathode", (_cap_I, _cap_r.regime)
        assert _cap_r.phi_c < 1000.0, (_cap_I, _cap_r.phi_c)
        assert _cap_r.phi_c > _cap_prev, (_cap_I, _cap_r.phi_c, _cap_prev)
        _cap_prev = _cap_r.phi_c
    for _cap_I in _CAPFIX_WINDOW_I_A:
        _cap_r = solve_idriven(
            _cap_cfg, _cap_pl, I_tot_A=_cap_I, **_CAPFIX_ESCAPE_KWARGS
        )
        assert _cap_r.regime == "capability_limited", (_cap_I, _cap_r.regime)
        # AT the cap, to the ceiling brentq's own rtol (1e-14) -- not the
        # 4 %-to-85 % excursions the escape produced.
        assert np.isclose(_cap_r.phi_c, 1000.0, rtol=1e-12, atol=0.0), (
            _cap_I, _cap_r.phi_c
        )
        assert _cap_r.V_b >= 1000.0, (_cap_I, _cap_r.V_b)
        assert _cap_r.I_tot >= 0.0, (_cap_I, _cap_r.I_tot)
    # Cap-dependence restored: pre-fix the escaped root was the SAME voltage
    # at every cap from 950 V to 1500 V. Post-fix the answer is the cap, until
    # the cap rises above what the sheath would have reached on its own.
    for _cap_V in (500.0, 900.0, 1000.0, 1500.0):
        _cap_r = solve_idriven(
            _cap_cfg, _cap_pl,
            I_tot_A=_CAPFIX_ESCAPE_I_A,
            **{**_CAPFIX_ESCAPE_KWARGS, "phi_c_cap_V": _cap_V},
        )
        assert _cap_r.regime == "capability_limited", (_cap_V, _cap_r.regime)
        assert np.isclose(_cap_r.phi_c, _cap_V, rtol=1e-12, atol=0.0), (
            _cap_V, _cap_r.phi_c
        )
    _cap_free = solve_idriven(
        _cap_cfg, _cap_pl,
        I_tot_A=_CAPFIX_ESCAPE_I_A,
        **{**_CAPFIX_ESCAPE_KWARGS, "phi_c_cap_V": 2000.0},
    )
    assert _cap_free.regime == "virtual_cathode", _cap_free.regime
    # The free root is a property of this frozen state and of the ion current
    # the sheath balance carries, so it moves with the sound speed and the
    # sheath lift; the clause gates that it sits ABOVE every cap above, which
    # is what makes the caps binding rather than incidental.
    assert 1658.0 < _cap_free.phi_c < 1660.0, _cap_free.phi_c
    # The beam-side lookup keeps its no-extrapolation contract at that raised
    # cap: the He EII table ends at 1000 eV, so a sheath past it is refused
    # rather than silently clamped to the last node.
    try:
        _cathode_solver_idriven_mod.solve_beam_system_idriven(
            _cap_cfg,
            np.array([_cap_pl.T_e, _cap_pl.T_e]),
            np.array([_cap_pl.n_e, _cap_pl.n_e]),
            np.array([_cap_pl.n_n, _cap_pl.n_n]),
            np.zeros(2),
            np.array([_cap_cfg.A_c, _cap_cfg.A_c]),
            I_ion,
            _CAPFIX_ESCAPE_I_A,
            cathode_index=0,
            **{**_CAPFIX_ESCAPE_KWARGS, "phi_c_cap_V": 2000.0},
        )
    except ValueError as _cap_err:
        assert "refused, not approximated" in str(_cap_err), _cap_err
    else:
        raise AssertionError(
            "a beam energy past the He EII table edge must be refused"
        )
    # ... and at the shipped cap it is INSIDE the table (the ceiling sits on
    # the last node to within a ULP), so the guard is inert and the lookup
    # returns what it always returned.
    _cap_beam = _cathode_solver_idriven_mod.solve_beam_system_idriven(
        _cap_cfg,
        np.array([_cap_pl.T_e, _cap_pl.T_e]),
        np.array([_cap_pl.n_e, _cap_pl.n_e]),
        np.array([_cap_pl.n_n, _cap_pl.n_n]),
        np.zeros(2),
        np.array([_cap_cfg.A_c, _cap_cfg.A_c]),
        I_ion,
        _CAPFIX_ESCAPE_I_A,
        cathode_index=0,
        **_CAPFIX_ESCAPE_KWARGS,
    )
    assert _cap_beam.result.regime == "capability_limited"
    assert _cap_beam.beam_cross[0] == _beam_deposition_mod.He_EII_cross_lkup(
        _cap_beam.result.phi_c / I_ion
    ), _cap_beam.beam_cross[0]
    return locals()


# --------------------------------------------------------------------
# cathode-clamp-census
# --------------------------------------------------------------------
@_case(
    "cathode-clamp-census",
)
def _case_cathode_clamp_census():
    def _r1_sim_config(**overrides):
        _p, _f = default_config()
        _p.update({
            "nx": 12,
            "cathode_Ts_base_K": 1998.15,
            "cathode_cleaning_E_th_eV": None,
        })
        _p["initial_neutral_state"] = "fill"
        for _k, _v in overrides.items():
            if _k in _f:
                _f[_k] = _v
            else:
                _p[_k] = _v
        return _p, _f

    # THE CLAMP IS COUNTED. A cathode solve whose root sits above the composed
    # ceiling returns the ceiling value tagged ``capability_limited`` and
    # raises nothing, so a run that spent solves there is indistinguishable
    # from one that did not unless the count exists. The solver counts the
    # clamped and the total accepted cathode solves; these are the two
    # directions of that counter.
    #
    # ARMED: an imposed loop current far above what this tiny cathode can
    # emit at the ceiling. The solve clamps, so the clamped count moves and
    # the first-clamp time is stamped.
    _clamp_hi = LAPDSim1D(*_r1_sim_config())
    assert _clamp_hi._cathode_total_solves == 0
    assert _clamp_hi._cathode_clamped_solves == 0
    assert math.isnan(_clamp_hi._cathode_clamp_first_t_s)
    _clamp_hi._circuit_I_loop = 1.0e3
    _clamp_hi_solve = _clamp_hi.solve_cathode_boundary(
        floating=False, update_cache=True
    )
    assert (
        str(_clamp_hi_solve.beam_result.result.regime) == "capability_limited"
    ), _clamp_hi_solve.beam_result.result.regime
    assert _clamp_hi._cathode_total_solves == 1
    assert _clamp_hi._cathode_clamped_solves >= 1
    assert math.isfinite(_clamp_hi._cathode_clamp_first_t_s)
    assert math.isfinite(_clamp_hi._cathode_clamp_last_t_s)

    # NEGATIVE CONTROL: the same fixture and the same call at a current the
    # cathode carries below the ceiling. The solve is counted and the clamped
    # count stays at zero -- so the counter is measuring the clamp and not
    # merely the solve.
    _clamp_lo = LAPDSim1D(*_r1_sim_config())
    _clamp_lo._circuit_I_loop = 1.0
    _clamp_lo_solve = _clamp_lo.solve_cathode_boundary(
        floating=False, update_cache=True
    )
    assert (
        str(_clamp_lo_solve.beam_result.result.regime) != "capability_limited"
    ), _clamp_lo_solve.beam_result.result.regime
    assert _clamp_lo._cathode_total_solves == 1
    assert _clamp_lo._cathode_clamped_solves == 0
    assert math.isnan(_clamp_lo._cathode_clamp_first_t_s)
    assert math.isnan(_clamp_lo._cathode_clamp_last_t_s)

    # A READ-ONLY solve is not a solve this run performed: it does not write
    # the cathode caches and it must not move the census either, or the
    # denominator would count the dt bound's probe solves alongside the
    # accepted ones.
    _clamp_lo.solve_cathode_boundary(floating=False, update_cache=False)
    assert _clamp_lo._cathode_total_solves == 1

    # RESTART. The counters are CARRIED, so a resumed run continues the
    # producing run's census instead of restarting it -- losing a count is a
    # lost measurement, which is the reason the jet-arming counts are carried
    # and it is the same reason here. Both directions: a payload that has the
    # keys continues, and one that predates them restores the seed rather than
    # raising.
    _clamp_params, _clamp_flags = _r1_sim_config()
    _clamp_run = LAPDSim1D(dict(_clamp_params), dict(_clamp_flags))
    for _ in range(6):
        _clamp_run.advance_one_step(dt=2.0e-9)
    _clamp_pre = int(_clamp_run._cathode_total_solves)
    assert _clamp_pre >= 1, _clamp_pre
    with tempfile.TemporaryDirectory() as _clamp_tmp:
        _clamp_payload = Path(_clamp_tmp) / "clamp_census.restart.h5"
        _restart_mod.save_restart_state(_clamp_payload, _clamp_run)
        _clamp_resumed = LAPDSim1D(
            {**_clamp_params, "restart_from": str(_clamp_payload)},
            dict(_clamp_flags),
        )
        assert _clamp_resumed._cathode_total_solves == _clamp_pre, (
            _clamp_resumed._cathode_total_solves, _clamp_pre
        )
        assert (
            _clamp_resumed._cathode_clamped_solves
            == _clamp_run._cathode_clamped_solves
        )
        _clamp_resumed.advance_one_step(dt=2.0e-9)
        assert _clamp_resumed._cathode_total_solves >= _clamp_pre + 1, (
            _clamp_resumed._cathode_total_solves, _clamp_pre
        )

        # A payload written before the keys existed. The counters are stored
        # as attributes of the ``cathode`` group, the two integers under the
        # writer's ``__int`` type tag, so a legacy payload is exactly this
        # file with those four attributes gone.
        _clamp_legacy = Path(_clamp_tmp) / "clamp_census_legacy.restart.h5"
        _clamp_legacy.write_bytes(_clamp_payload.read_bytes())
        with h5py.File(_clamp_legacy, "r+") as _clamp_h5:
            _clamp_grp = _clamp_h5["cathode"]
            for _clamp_key in (
                "_cathode_total_solves__int",
                "_cathode_clamped_solves__int",
                "_cathode_clamp_first_t_s",
                "_cathode_clamp_last_t_s",
            ):
                assert _clamp_key in _clamp_grp.attrs, _clamp_key
                del _clamp_grp.attrs[_clamp_key]
        _clamp_old = LAPDSim1D(
            {**_clamp_params, "restart_from": str(_clamp_legacy)},
            dict(_clamp_flags),
        )
        assert _clamp_old._cathode_total_solves == 0
        assert _clamp_old._cathode_clamped_solves == 0
        assert math.isnan(_clamp_old._cathode_clamp_first_t_s)
        assert math.isnan(_clamp_old._cathode_clamp_last_t_s)


# --------------------------------------------------------------------
# circuit-current-driven-integration
# --------------------------------------------------------------------
@_case(
    "circuit-current-driven-integration",
    historical_stance=True,
    provides=(
        "idriven_vdis_evaluator", "m3_Iloop", "m3_cathode_flags",
        "m3_diag", "m3_params", "m3_run_sim",
    ),
)
def _case_circuit_current_driven_integration():
    # --- Current-driven circuit integration (M3):
    # TR-BDF2 stages as bracketed scalar root-finds against monotone
    # V_dis(I). Gate 1: 2nd order on the analytic RLC decay with a linear
    # V_dis(I) load (halve dt, error / ~4).
    resolved_cathode_flags = _resolved_cathode_flags()
    from cablp.solvers._sim1d.physics.cathode import (
        advance_circuit_current_driven,
        idriven_vdis_evaluator,
        validate_cathode_solver_model,
    )

    m3_L, m3_R, m3_Rd, m3_V0d, m3_Vs = 6.6e-6, 5.72e-3, 5.0e-2, 120.0, 173.6
    m3_lin = lambda I: m3_V0d + m3_Rd * I  # noqa: E731
    m3_tau = m3_L / (m3_R + m3_Rd)
    m3_Iinf = (m3_Vs - m3_V0d) / (m3_R + m3_Rd)
    m3_T = 2.0e-4  # ~1.7 tau

    def m3_integrate(nsteps):
        I = 0.0
        dt = m3_T / nsteps
        for _ in range(nsteps):
            I, _, _ = advance_circuit_current_driven(
                I, dt, m3_Vs, m3_R, m3_L, m3_lin
            )
        return I

    m3_exact = m3_Iinf * (1.0 - np.exp(-m3_T / m3_tau))
    m3_e1 = abs(m3_integrate(40) - m3_exact)
    m3_e2 = abs(m3_integrate(80) - m3_exact)
    m3_order = np.log2(m3_e1 / m3_e2)
    assert 1.8 < m3_order < 2.4, (m3_order, m3_e1, m3_e2)

    # Gate 2: the plasma-diode clamp -- freewheel against a constant
    # positive V_dis decays to exactly 0 and never goes negative.
    m3_I = 500.0
    for _ in range(400):
        m3_I, _, m3_Vstep = advance_circuit_current_driven(
            m3_I, 2.0e-6, 0.0, m3_R, m3_L, lambda I: 50.0
        )
        assert m3_I >= 0.0
        assert np.isfinite(m3_Vstep)
    assert m3_I == 0.0

    # Gate 3: the stiff wall (why the scheme is implicit).
    # A device curve with a 1 MOhm/A branch above I_ceil:
    # explicit/frozen-V_dis needs dV/dI < 2L/dt ~ 22 mOhm and would
    # sawtooth; the implicit stages must approach the wall monotonically,
    # never overshoot it (L-stability), and pin there.
    m3_Icl = 2000.0
    m3_wall = lambda I: 150.0 + 1.0e6 * max(I - m3_Icl, 0.0)  # noqa: E731
    # Equilibrium just above the knee: V_src - I R - 150 = 1e6 (I - Icl).
    m3_Istar = (m3_Vs - 150.0 + 1.0e6 * m3_Icl) / (1.0e6 + m3_R)
    m3_hist = [1800.0]
    for _ in range(400):
        m3_hist.append(
            advance_circuit_current_driven(
                m3_hist[-1], 6.0e-7, m3_Vs, m3_R, m3_L, m3_wall
            )[0]
        )
    m3_hist = np.array(m3_hist)
    assert np.all(np.diff(m3_hist) > -1e-9)  # monotone approach, no sawtooth
    assert np.max(m3_hist) <= m3_Istar + 1e-6  # L-stable: never overshoots
    assert abs(m3_hist[-1] - m3_Istar) < 0.1  # pinned at the wall
    # Capacitor bookkeeping: trapezoidal drain, floored at zero.
    m3_Iv, m3_Vc, _ = advance_circuit_current_driven(
        1000.0, 1.0e-6, 170.0, m3_R, m3_L, m3_lin,
        C_bank_F=8.9, V_cap_prev_V=170.0,
    )
    assert 0.0 < m3_Vc < 170.0

    # Step-integrated V_dis (the inductor's view). At the linear-load
    # equilibrium the current is stationary, so the loop identity closes
    # exactly: <V_dis> = V_src - R*I_inf, and it must equal the device
    # value V_dis(I_inf). Off equilibrium it must sit inside the step's
    # V_dis range (monotone device, monotone I trajectory).
    m3_Ieq, _, m3_Veq = advance_circuit_current_driven(
        m3_Iinf, 6.0e-7, m3_Vs, m3_R, m3_L, m3_lin
    )
    assert abs(m3_Ieq - m3_Iinf) < 1e-6 * m3_Iinf
    assert abs(m3_Veq - (m3_Vs - m3_R * m3_Iinf)) < 1e-6
    assert abs(m3_Veq - m3_lin(m3_Iinf)) < 1e-6
    m3_Ir, _, m3_Vr = advance_circuit_current_driven(
        0.5 * m3_Iinf, 6.0e-7, m3_Vs, m3_R, m3_L, m3_lin
    )
    m3_Vlo = min(m3_lin(0.5 * m3_Iinf), m3_lin(m3_Ir))
    m3_Vhi = max(m3_lin(0.5 * m3_Iinf), m3_lin(m3_Ir))
    assert m3_Vlo - 1e-9 <= m3_Vr <= m3_Vhi + 1e-9, (m3_Vlo, m3_Vr, m3_Vhi)

    # Gate 4: solver dispatch. A current-driven sim's solve is an
    # evaluation at the frozen loop current; floating routes to the
    # historical open-circuit branch; validation fails fast.
    # M3 circuit integration on the simple cathode/fluid stance (isolates the
    # loop-current advance + vdis consistency from the beam/smoothing/repair
    # confounds); the M3 circuit machinery is model-agnostic.
    m3_cu_params, m3_cu_flags = _cathode_unit_config()
    m3_params = dict(m3_cu_params)
    m3_params.update(
        {
            "V_bank": 173.6,
            "R_comp": 5.72e-3,
            "L_parasitic_H": 6.6e-6,
            "cathode_solver_model": "current_driven",
            "dt_save": 0.0,
        }
    )
    m3_cathode_flags = dict(
        m3_cu_flags, cathode_coupling=True,
    )
    m3_sim = LAPDSim1D(m3_params, m3_cathode_flags)
    m3_sim._circuit_I_loop = 800.0
    m3_solve = m3_sim.solve_cathode_boundary(update_cache=False)
    assert m3_solve.metadata["cathode_solver_model"] == "current_driven"
    assert np.isclose(
        m3_solve.beam_result.result.I_tot, 800.0, rtol=1e-6
    ) or m3_solve.beam_result.result.regime == "capability_limited"
    assert m3_solve.beam_result.result_twin is None
    m3_float = m3_sim.solve_cathode_boundary(floating=True, update_cache=False)
    # The open-circuit phase is the current-driven solve at I_tot = 0, so the
    # reported I_tot is that solve's own RECONSTRUCTION of the imposed zero out
    # of currents of order I_eth*: it carries root-finder roundoff at that
    # scale rather than being the exact literal the retired open-circuit root
    # assigned. Kirchhoff is the quantity that must close, and it does.
    _m3_fr = m3_float.beam_result.result
    assert abs(_m3_fr.I_tot) <= 1.0e-12 * max(_m3_fr.I_eth_star, 1.0), (
        _m3_fr.I_tot, _m3_fr.I_eth_star
    )
    assert abs(_m3_fr.I_cathode_kirchhoff_residual) <= 1.0e-12, (
        _m3_fr.I_cathode_kirchhoff_residual
    )
    for m3_bad_params, m3_bad_flags in (
        (dict(m3_params, cathode_solver_model="bogus"), resolved_cathode_flags),
        (dict(m3_params, L_parasitic_H=0.0), resolved_cathode_flags),
        (m3_params, dict(resolved_cathode_flags, TwinCathode=True)),
    ):
        try:
            LAPDSim1D(m3_bad_params, m3_bad_flags)
        except ValueError:
            pass
        else:
            raise AssertionError(
                "expected ValueError for "
                f"{m3_bad_params.get('cathode_solver_model')}"
            )
    assert (
        validate_cathode_solver_model(m3_params, resolved_cathode_flags)
        == "current_driven"
    )
    # B2: the CSDA deposition rides the current-driven dispatch too (the
    # solver-agnostic interface's second consumer).
    m3_csda_sim = LAPDSim1D(dict(m3_params), resolved_cathode_flags)
    m3_csda_sim._circuit_I_loop = 800.0
    m3_csda_solve = m3_csda_sim.solve_cathode_boundary(update_cache=False)
    assert m3_csda_solve.beam_deposition is not None
    m3_csda_dep = m3_csda_solve.beam_deposition[0]
    assert m3_csda_dep is not None
    m3_csda_res = m3_csda_solve.beam_result.result
    m3_csda_budget = m3_csda_res.I_eth_star * m3_csda_res.phi_c * 1.0e7
    m3_csda_total = (
        m3_csda_dep.plasma_heating_erg_s.sum()
        + m3_csda_dep.radiated_erg_s.sum()
        + m3_csda_dep.ionization_cost_erg_s.sum()
        # R4.1 anode interception is the production default, so the anode-removed
        # energy is part of the per-ray budget.
        + float(m3_csda_dep.anode_intercepted_erg_s)
        + m3_csda_dep.transmitted_flux
        * m3_csda_dep.transmitted_energy_eV
        * ev_to_erg
    )
    assert abs(m3_csda_total - m3_csda_budget) / m3_csda_budget < 1e-9

    # Gate 5: drive mini-run. The loop current starts at 0 and rises at
    # ~(V_src - V_dis)/L; the per-step solve reports the *frozen* current
    # (evaluation, not iteration).
    m3_run_sim = LAPDSim1D(m3_params, m3_cathode_flags)
    m3_result = m3_run_sim.run(t_end=3.0e-10, dt=1.0e-10)
    m3_diag = m3_result.cathode_diagnostics
    m3_Iloop = np.asarray(m3_diag["circuit_I_loop"], float)
    assert m3_Iloop[0] == 0.0
    assert np.all(np.isfinite(m3_Iloop))
    assert np.all(np.diff(m3_Iloop) > 0.0)  # rising from 0 under drive
    assert m3_Iloop[-1] < 1.0  # 3e-10 s at ~2.6e7 A/s
    # Discharge-voltage diagnostic: 0.0 before any circuit advance, then
    # the save-interval dt-weighted average of the inductor's-view V_dis
    # (here identical to the per-step value: fixed dt, saves every step)
    # -- reconstructable from the loop identity save-to-save.
    m3_Vstep = np.asarray(m3_diag["circuit_V_dis_step"], float)
    assert m3_Vstep.shape == m3_Iloop.shape
    assert m3_Vstep[0] == 0.0
    assert np.all(np.isfinite(m3_Vstep))
    m3_recon = (
        m3_params["V_bank"]
        - 6.6e-6 * np.diff(m3_Iloop) / 1.0e-10
        - m3_params["R_comp"] * 0.5 * (m3_Iloop[1:] + m3_Iloop[:-1])
    )
    assert np.allclose(m3_Vstep[1:], m3_recon, atol=0.5), (
        m3_Vstep[1:], m3_recon
    )
    return locals()


# --------------------------------------------------------------------
# cathode-power-balance-under-current-drive
# --------------------------------------------------------------------
@_case(
    "cathode-power-balance-under-current-drive",
    historical_stance=True,
)
def _case_cathode_power_balance_under_current_drive(
    m3_Iloop, m3_diag, m3_params, m3_run_sim
):
    from cablp.solvers._sim1d.physics.cathode import idriven_vdis_evaluator

    # Power-balance warming under current_driven must feed on the HONEST
    # accepted-state solve, not the RHS cache: the cache holds the step's
    # last internal-stage solve, measured at 4.6-7.5x the accepted-state
    # P_cathode_i at the same frozen current (2026-07-21). Spy on the
    # evaluator the warming branch uses and require the energy ledger to
    # integrate exactly the honest values it returned.
    resolved_cathode_flags = _resolved_cathode_flags()
    import cablp.solvers._sim1d.solver as _solver_mod

    pbh_calls = []
    _pbh_orig = _solver_mod.idriven_result_evaluator

    def _pbh_spy(**kw):
        f = _pbh_orig(**kw)

        def g(I):
            res = f(I)
            pbh_calls.append((float(I), float(res.P_cathode_i)))
            return res

        return g

    _solver_mod.idriven_result_evaluator = _pbh_spy
    try:
        pbh_sim = LAPDSim1D(
            dict(
                m3_params,
                cathode_Ts_base_K=1910.0,
                cathode_heat_capacity_J_per_K=120.0,
                cathode_conduction_W_per_K=1200.0,
            ),
            resolved_cathode_flags,
        )
        pbh_result = pbh_sim.run(t_end=3.0e-10, dt=1.0e-10)
    finally:
        _solver_mod.idriven_result_evaluator = _pbh_orig
    assert len(pbh_calls) == 3, len(pbh_calls)  # one per accepted step
    pbh_E_ion = float(
        np.asarray(
            pbh_result.cathode_diagnostics["warming_E_ion_J"], float
        )[-1]
    )
    assert np.isclose(
        pbh_E_ion,
        sum(1.0e-10 * max(p, 0.0) for _, p in pbh_calls),
        rtol=1e-12,
        atol=0.0,
    ), (pbh_E_ion, pbh_calls)

    # Surface-state coverage model (ads/des). Validation fails fast; the
    # coverage
    # update must reproduce the backward-Euler form exactly from the spy's
    # honest I_i; phi_eff must actually reach the solve (a cleaner surface
    # emits more at fixed T_s and imposed current => shallower sheath).
    for sf_bad in (
        # missing clean floor (the default supplies one, so clear it):
        {"cathode_phiwf_clean_eV": None},
        {"cathode_phiwf_clean_eV": 99.0},  # floor above phi_wf
        {"cathode_phiwf_clean_eV": 2.75,
         "cathode_cleaning_sigma_cm2": -1.0},
    ):
        try:
            LAPDSim1D(dict(m3_params, **sf_bad), resolved_cathode_flags)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {sf_bad}")
    # ads_des is the subject here; run it on the simple cathode/fluid stance
    # (the theta reproduction spies the exact evaluator call sequence) with the
    # M3 circuit specifics.
    sf_cu_params, sf_cu_flags = _cathode_unit_config()
    sf_params = dict(
        sf_cu_params,
        V_bank=173.6,
        R_comp=5.72e-3,
        L_parasitic_H=6.6e-6,
        cathode_solver_model="current_driven",
        dt_save=0.0,
        cathode_phiwf_clean_eV=2.75,
        cathode_cleaning_sigma_cm2=1.0e-16,
    )
    sf_flags = dict(
        sf_cu_flags, cathode_coupling=True,
    )
    sf_calls = []
    _sf_orig = _solver_mod.idriven_result_evaluator

    def _sf_spy(**kw):
        f = _sf_orig(**kw)

        def g(I):
            res = f(I)
            sf_calls.append(float(res.I_i))
            return res

        return g

    # This coverage test spies the exact evaluator I_i CALL SEQUENCE and replays
    # the backward-Euler update once per call, so the exact match couples to the
    # solver's internal call count; run it on the simple stance (sf_flags) and
    # assert the backward-Euler FORM to 1e-11 rather than 1e-15.
    _solver_mod.idriven_result_evaluator = _sf_spy
    try:
        sf_sim = LAPDSim1D(sf_params, sf_flags)
        assert sf_sim._cathode_theta == 1.0
        sf_sim._circuit_I_loop = 800.0
        sf_result = sf_sim.run(t_end=3.0e-10, dt=1.0e-10)
    finally:
        _solver_mod.idriven_result_evaluator = _sf_orig
    sf_theta = np.asarray(
        sf_result.cathode_diagnostics["surface_theta"], float
    )
    sf_phieff = np.asarray(
        sf_result.cathode_diagnostics["phi_wf_eff"], float
    )
    assert np.all(np.isfinite(sf_theta)) and np.all(sf_theta <= 1.0)
    assert np.all(np.diff(sf_theta) <= 0.0)  # ion-stimulated cleaning only
    # Reproduce the backward-Euler update exactly from the spy's honest
    # I_i sequence (run() starts I_loop at 0, so the accepted honest
    # solves carry the near-floating I_i -- the form is what's tested).
    sf_area = np.pi * float(sf_params["R_cath"]) ** 2
    sf_th = 1.0
    for sf_Ii in sf_calls:
        sf_G = max(sf_Ii, 0.0) / (1.602176634e-19 * sf_area)
        sf_loss = 1.0e-16 * sf_G
        sf_th = sf_th / (1.0 + 1.0e-10 * sf_loss)
    assert np.isclose(sf_theta[-1], sf_th, rtol=0.0, atol=1e-11), (
        sf_theta[-1], sf_th
    )
    assert np.allclose(
        sf_phieff,
        2.75 + (float(sf_params["phi_wf"]) - 2.75) * sf_theta,
        rtol=1e-12,
    )
    # phi_eff reaches the solve: the dispatched device config's Richardson
    # ceiling must grow as the surface cleans (regime-independent -- a
    # deep-SCL solve's phi_c legitimately ignores emission capability, so
    # the ceiling is the right plumbing observable). Ratio check against
    # the Richardson exponent at the config T_s.
    sf_sim2 = LAPDSim1D(sf_params, sf_flags)
    sf_sim2._circuit_I_loop = 800.0
    sf_hi = sf_sim2.solve_cathode_boundary(update_cache=False)
    sf_sim2._cathode_theta = 0.2
    sf_lo = sf_sim2.solve_cathode_boundary(update_cache=False)
    assert sf_lo.device_config.I_eth > sf_hi.device_config.I_eth
    sf_dphi = 0.8 * (float(sf_params["phi_wf"]) - 2.75)
    sf_kT = 8.617333262e-5 * float(sf_params["cathode_Ts_base_K"])
    assert np.isclose(
        sf_lo.device_config.I_eth / sf_hi.device_config.I_eth,
        np.exp(sf_dphi / sf_kT),
        rtol=1e-9,
    )

    # M5a' energy-dependent yield: with cathode_cleaning_E_th_eV set, the
    # coverage update scales sigma by the Bohdansky near-threshold factor
    # at E = P_cathode_i/I_i. Below threshold nothing cleans (theta
    # frozen); with E_th = None the M5a fluence limit is reproduced
    # bit-for-bit (default-compat gate).
    sfE_params = dict(sf_params, cathode_cleaning_E_th_eV=1.0e6)
    sfE_sim = LAPDSim1D(sfE_params, sf_flags)
    sfE_sim._circuit_I_loop = 800.0
    sfE_result = sfE_sim.run(t_end=3.0e-10, dt=1.0e-10)
    sfE_theta = np.asarray(
        sfE_result.cathode_diagnostics["surface_theta"], float
    )
    assert np.all(sfE_theta == 1.0), sfE_theta  # far below threshold
    sfN_params = dict(sf_params, cathode_cleaning_E_th_eV=None)
    sfN_sim = LAPDSim1D(sfN_params, sf_flags)
    sfN_sim._circuit_I_loop = 800.0
    sfN_result = sfN_sim.run(t_end=3.0e-10, dt=1.0e-10)
    assert np.array_equal(
        np.asarray(sfN_result.cathode_diagnostics["surface_theta"], float),
        sf_theta,
    )

    # Saved diagnostics are refreshed post-accept, so the recorded solve
    # is an evaluation at the *accepted* loop current of the same save.
    for m3_k in (1, 2, 3):
        assert np.isclose(
            m3_diag["source_I_tot"][m3_k],
            m3_Iloop[m3_k],
            rtol=1e-6,
            atol=1e-9,
        ), (m3_k, m3_diag["source_I_tot"][m3_k], m3_Iloop[m3_k])
    # The evaluator used by the circuit advance agrees with the dispatched
    # solve's device voltage at the same state and current.
    # On the smoothed sample the dispatched solve reads, so the two are
    # evaluated on one state.
    m3_vdis = idriven_vdis_evaluator(
        state=m3_run_sim._smoothed_sample_state(m3_run_sim.state),
        floors=m3_run_sim._floors,
        ion_mass_g=m3_run_sim._ion_mass_g,
        mu=m3_run_sim._mu,
        geometry=m3_run_sim._geometry,
        input_dict=m3_run_sim._input_dict,
        input_flags=m3_run_sim._effective_cathode_flags(active_only=False),
        beam_cross_prev=m3_run_sim._cathode_beam_cross,
        T_s_override_K=m3_run_sim._cathode_Ts_K,
    )
    m3_direct = m3_run_sim.solve_cathode_boundary(update_cache=False)
    assert np.isclose(
        m3_vdis(m3_run_sim._circuit_I_loop),
        m3_direct.beam_result.result.V_b,
        rtol=1e-10,
    )


# --------------------------------------------------------------------
# cathode-power-balance-warming
# --------------------------------------------------------------------
@_case(
    "cathode-power-balance-warming",
    historical_stance=True,
    provides=("growth_flags", "growth_params"),
)
def _case_cathode_power_balance_warming(
    cathode_diag, cathode_run_flags, cathode_run_params,
    cathode_run_result, cathode_run_sim, expected_rhs_terms,
    no_source_params
):
    # --- Power-balance warming (M1b): the surface energy budget replaces the
    # imposed T_s asymptote. Heater pinned by standby equilibrium; emission
    # cooling uses the actually emitted current.
    params, flags = _base_config()
    sim, snapshot = _base_sim()
    geom = snapshot.geometry
    from cablp.solvers._sim1d.physics.cathode import (
        CATHODE_ENV_T_K,
        cathode_power_balance_terms_W,
    )

    pb_params = dict(cathode_run_params)
    pb_params["cathode_heat_capacity_J_per_K"] = float(
        no_source_params["cathode_heat_capacity_J_per_K"]
    )
    pb_params["cathode_Ts_base_K"] = (
        float(cathode_run_params["cathode_Ts_base_K"]) - 110.0
    )
    # This block probes the power-balance TERMS against the pure-radiation
    # baseline (the substrate conduction is exercised separately via
    # pb_cond_dict below); the production default now sets conduction=1200, so
    # pin the baseline to zero here.
    pb_params["cathode_conduction_W_per_K"] = 0.0
    pb_dict = pb_params
    T_base = pb_dict["cathode_Ts_base_K"]
    _pb_signs = ((0, 1), (1, 1), (2, -1), (3, -1), (4, -1))
    # Standby: no discharge => exact equilibrium at T_base, by construction
    # (conduction also vanishes there, so the heater pinning is unchanged).
    ph, pi_, pr, pe, pc = cathode_power_balance_terms_W(
        T_base, 0.0, 0.0, pb_dict
    )
    assert ph == pr and pi_ == 0.0 and pe == 0.0 and pc == 0.0
    # Radiation restores: net power negative above standby, positive below.
    assert sum(
        cathode_power_balance_terms_W(T_base + 50.0, 0.0, 0.0, pb_dict)[i] * s
        for i, s in _pb_signs
    ) < 0.0
    assert sum(
        cathode_power_balance_terms_W(T_base - 50.0, 0.0, 0.0, pb_dict)[i] * s
        for i, s in _pb_signs
    ) > 0.0
    # Magnitudes at the production point (the M1b design numbers): radiation
    # ~60-70 kW at 2000 K over the disc, emission cooling ~10 kW at 3 kA.
    _, _, pr2000, pe3ka, _ = cathode_power_balance_terms_W(
        2000.0, 0.0, 3000.0, pb_dict
    )
    assert 3.0e4 < pr2000 < 1.2e5, pr2000
    assert 9.0e3 < pe3ka < 1.15e4, pe3ka
    # Substrate conduction is the strong restoring term (the pure-radiation
    # balance measured unstable at the LAPD operating point): G_cond scales
    # the excursion linearly and vanishes at standby.
    pb_cond_dict = dict(pb_dict, cathode_conduction_W_per_K=2000.0)
    assert cathode_power_balance_terms_W(
        T_base + 100.0, 0.0, 0.0, pb_cond_dict
    )[4] == 2000.0 * 100.0
    assert cathode_power_balance_terms_W(
        T_base, 0.0, 0.0, pb_cond_dict
    )[4] == 0.0
    # Emission cooling lowers the equilibrium: with the same drive power a
    # cooling-on balance point sits below the cooling-off one.
    drive_W = 5.0e4
    def _pb_net(T, I_emis):
        h, p, r, e, c = cathode_power_balance_terms_W(
            T, drive_W, I_emis, pb_dict
        )
        return h + p - r - e - c
    T_grid = np.linspace(T_base, T_base + 400.0, 4001)
    eq_off = T_grid[np.argmin(np.abs([_pb_net(T, 0.0) for T in T_grid]))]
    eq_on = T_grid[np.argmin(np.abs([_pb_net(T, 3000.0) for T in T_grid]))]
    assert eq_on < eq_off
    assert eq_off - T_base > 100.0  # ~50 kW drives an O(100 K) rise
    # Mini-run: the saved trajectory must reproduce the semi-implicit update
    # exactly from the saved solve diagnostics — including the emission
    # cooling sign (this near-standby state is net *cooling*: ~1 A of
    # emitted current outweighs ~3 W of bombardment).
    pb_sim = LAPDSim1D(pb_params, cathode_run_flags)
    pb_result = pb_sim.run(t_end=3.0e-10, dt=1.0e-10)
    pb_diag = pb_result.cathode_diagnostics
    pb_Ts = pb_diag["T_s_surface"]
    assert pb_Ts[0] == T_base
    assert np.all(np.isfinite(pb_Ts))
    assert np.any(pb_Ts != T_base)  # the accepted-step update actually runs
    assert np.allclose(pb_Ts, T_base, atol=1e-6)  # near standby, barely moves
    _pb_sb, _pb_kb = 5.670374419e-12, 8.617333262e-5
    _pb_area = np.pi * float(pb_params["R_cath"]) ** 2
    _pb_eps = float(pb_params["cathode_emissivity"])
    _pb_C = float(pb_params["cathode_heat_capacity_J_per_K"])
    pb_T_prev = float(pb_Ts[0])
    for pb_k in range(1, pb_Ts.size):
        pb_I = (
            0.0
            if pb_diag["floating"][pb_k]
            else max(float(pb_diag["source_I_eth_star"][pb_k]), 0.0)
        )
        pb_h, pb_p, pb_r, pb_e, pb_c = cathode_power_balance_terms_W(
            pb_T_prev, pb_diag["source_P_cathode_i"][pb_k], pb_I, pb_dict
        )
        pb_G = (
            4.0 * _pb_eps * _pb_sb * _pb_area * pb_T_prev**3
            + pb_I * 2.0 * _pb_kb
            + float(pb_params.get("cathode_conduction_W_per_K", 0.0))
        )
        pb_T_prev = max(
            pb_T_prev
            + 1.0e-10
            * (pb_h + pb_p - pb_r - pb_e - pb_c)
            / (_pb_C + 1.0e-10 * pb_G),
            CATHODE_ENV_T_K,
        )
        assert np.isclose(pb_Ts[pb_k], pb_T_prev, rtol=0.0, atol=1e-9), (
            pb_k, pb_Ts[pb_k], pb_T_prev,
        )
    # Vanishing heat capacity: the semi-implicit update jumps to the
    # linearized equilibrium instead of overshooting and ringing.
    snap_pb_params = dict(pb_params)
    snap_pb_params["cathode_heat_capacity_J_per_K"] = 1.0e-30
    snap_pb_sim = LAPDSim1D(snap_pb_params, cathode_run_flags)
    snap_pb_result = snap_pb_sim.run(t_end=3.0e-10, dt=1.0e-10)
    snap_pb_Ts = snap_pb_result.cathode_diagnostics["T_s_surface"]
    assert np.all(np.isfinite(snap_pb_Ts))
    assert np.all(np.diff(snap_pb_Ts[1:]) >= -1.0)  # settles, no ringing
    assert snap_pb_Ts[-1] < 4000.0  # bounded by the linearized-loss backstop
    assert np.all(np.isfinite(cathode_diag["beam_cross"]))
    assert np.all(np.isfinite(cathode_diag["n_beam"]))
    assert np.all(np.isfinite(cathode_diag["v_beam"]))
    assert np.all(np.isfinite(cathode_diag["l_b_profile"]))
    assert np.allclose(cathode_diag["l_b_profile_twin"], 0.0)
    assert np.all(
        np.isfinite(cathode_run_result.rhs_terms["cathode_surface_loss"]["n"])
    )
    for _beam_key in (
        "beam_ionization_birth",
        "beam_power_deposition",
        "beam_ionization_cost",
    ):
        assert np.all(np.isfinite(cathode_run_result.rhs_terms[_beam_key]["Ee"]))
    _ns_rt_I_tot = np.asarray(
        cathode_run_result.cathode_diagnostics["source_I_tot"], dtype=float
    )[:4]
    _ns_rt_I_e = np.asarray(
        cathode_run_result.cathode_diagnostics["source_I_e"], dtype=float
    )[:4]
    assert _ns_rt_I_tot[0] >= -1.0e-12 * abs(_ns_rt_I_e[0]), _ns_rt_I_tot[0]
    assert np.all(_ns_rt_I_tot[1:] > 0.0), _ns_rt_I_tot[1:]
    # A claim about the SOLVER, over the terms a retired _sim3 alias summed.
    # It used to read "np.any(sum > 0)" -- these channels deposit net POSITIVE
    # electron power somewhere -- which was true only because this fixture
    # pinned the legacy full-P_*_e electrode routing: the cathode row then
    # deposited the sheath fall phi_c into the plasma electron store and ran
    # +2.56e-05 W/cm^3 POSITIVE. With the A16 thermal-only routing
    # unconditional (see commit 1fc05c9) phi is the ELECTRODE's, the cathode
    # row is a pure sink, and the sum is <= 0 everywhere. So the surviving --
    # and strictly stronger -- statement is asserted instead: the channels are
    # LIVE, and they never deposit net positive electron power into the plasma.
    _pb_sum = sum(
        np.asarray(
            cathode_run_result.electron_energy_terms_W_cm3[_term],
            dtype=float,
        )
        for _term in (
            "beam_power_deposition",
            "beam_ionization_cost",
            "cathode_surface_loss",
            "anode_e_sheath_loss",
        )
    )
    assert np.all(np.isfinite(_pb_sum))
    assert np.any(_pb_sum != 0.0)  # non-vacuous: the channels are live
    assert np.all(_pb_sum <= 0.0)  # the sheath fall is the electrode's
    # The split is honest about which electrode paid. The load-bearing
    # property is DISJOINT SUPPORT: with a resolved anode the cathode share
    # lands at the cathode cell and the anode share at the flanking cells, so
    # no cell receives both -- which is what makes the pair's sum bit-exactly
    # the single row it replaced. (Which of the two is LARGER is a property of
    # the routing, not of the split: under the thermal-only electrode routing
    # -- unconditional since commit 1fc05c9 -- the anode dominates by orders
    # of magnitude.)
    _cathode_Ee = np.asarray(
        cathode_run_result.electron_energy_terms_W_cm3["cathode_surface_loss"],
        dtype=float,
    )
    _anode_Ee = np.asarray(
        cathode_run_result.electron_energy_terms_W_cm3["anode_e_sheath_loss"],
        dtype=float,
    )
    assert np.all(np.isfinite(_cathode_Ee)) and np.all(np.isfinite(_anode_Ee))
    assert not np.any((_cathode_Ee != 0.0) & (_anode_Ee != 0.0))
    assert np.abs(_anode_Ee).max() > 0.0
    assert np.abs(_cathode_Ee).max() > 0.0
    # Summed over the five base rows; the packed y also carries nn_a. The
    # circuit solve arms the emitting cathode face's three sheath rows on top
    # of the circuit-off term set.
    cathode_saved_sum = np.zeros(
        (
            cathode_run_result.y.shape[0],
            len(STATE_NAMES_1D) * np.asarray(cathode_run_result.nn).shape[1],
        )
    )
    assert set(cathode_run_result.rhs_terms) == (
        expected_rhs_terms | set(END_SHEATH_CATHODE_ROWS)
    )
    for term_name in cathode_run_result.rhs_terms:
        term_fields = cathode_run_result.rhs_terms[term_name]
        assert np.allclose(
            cathode_run_result.electron_energy_terms_W_cm3[term_name],
            1.0e-7 * term_fields["Ee"],
        )
        cathode_saved_sum = cathode_saved_sum + np.concatenate(
            [term_fields[field_name] for field_name in STATE_NAMES_1D],
            axis=1,
        )
    cathode_packed_total_rhs = np.concatenate(
        [
            cathode_run_result.total_rhs[field_name]
            for field_name in STATE_NAMES_1D
        ],
        axis=1,
    )
    assert np.allclose(cathode_saved_sum, cathode_packed_total_rhs)
    cathode_run_summary = summarize_result(cathode_run_result)
    assert cathode_run_summary.finite
    assert cathode_run_summary.n_min >= cathode_run_params["ne_floor"]
    assert cathode_run_summary.nn_min >= cathode_run_params["nn_floor"]
    assert cathode_run_summary.Te_min >= cathode_run_params["Te_floor"]
    assert cathode_run_summary.Ti_min >= cathode_run_params["Ti_floor"]
    assert cathode_run_summary.phase_counts == {"pre_breakdown": 4}
    assert cathode_run_summary.diagnostic_phase_counts == {"pre_breakdown": 3}
    assert cathode_run_summary.phase_switch_fractions == {
        "cathode_enabled": 1.0,
        "floating": 0.0,
        "gas_puff_enabled": 0.0,
    }
    assert cathode_run_summary.cathode_diagnostic_fractions["configured"] == 1.0
    assert cathode_run_summary.cathode_diagnostic_fractions["phase_enabled"] == 1.0
    assert cathode_run_summary.cathode_diagnostic_fractions["rhs_enabled"] == 1.0
    assert cathode_run_summary.cathode_diagnostic_fractions["solve_enabled"] == 1.0
    assert cathode_run_summary.cathode_diagnostic_fractions["floating"] == 0.0
    assert cathode_run_summary.cathode_diagnostic_fractions["has_solution"] == 1.0
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = cathode_run_sim.save_result(
            f"{tmpdir}/sim1d_cathode_smoke.h5",
            cathode_run_result,
        )
        with h5py.File(output_path, "r") as h5:
            assert h5.attrs["steps"] == cathode_run_result.steps
            saved_flags = json.loads(h5.attrs["flags_json"])
            assert saved_flags["cathode_coupling"]
            assert h5["rhs_terms/cathode_surface_loss/n"].shape == (4, geom.cells)
            assert h5["rhs_terms/beam_power_deposition/Ee"].shape == (
                4,
                geom.cells,
            )
            assert h5["rhs_terms/beam_ionization_cost/Ee"].shape == (
                4,
                geom.cells,
            )
            assert h5["cathode_diagnostics/source_phi_c"].shape == (4,)
            assert h5["cathode_diagnostics/source_regime"].shape == (4,)
            assert np.all(h5["cathode_diagnostics/phase_enabled"][()] == 1.0)
            assert np.all(h5["cathode_diagnostics/solve_enabled"][()] == 1.0)
            assert np.all(h5["cathode_diagnostics/floating"][()] == 0.0)
            assert h5["cathode_diagnostics/beam_cross"].shape == (
                4,
                geom.cells,
            )
            _ns_h5_I_tot = h5["cathode_diagnostics/source_I_tot"][()]
            _ns_h5_I_e = h5["cathode_diagnostics/source_I_e"][()]
            assert (
                _ns_h5_I_tot[0] >= -1.0e-12 * abs(_ns_h5_I_e[0])
            ), _ns_h5_I_tot[0]
            assert np.all(_ns_h5_I_tot[1:] > 0.0), _ns_h5_I_tot[1:]
            assert all(
                value.decode("utf-8")
                in {"classical", "virtual_cathode", "capability_limited"}
                for value in h5["cathode_diagnostics/source_regime"][()]
            )
            assert np.all(
                np.isfinite(h5["cathode_diagnostics/beam_cross"][()])
            )
        loaded_cathode_result = load_result_hdf5(output_path)
        assert loaded_cathode_result.flags["cathode_coupling"]
        assert np.allclose(
            loaded_cathode_result.phase_cathode_enabled,
            cathode_run_result.phase_cathode_enabled,
        )
        # The circuit solve arms the emitting cathode face's three sheath
        # rows on top of the circuit-off term set.
        assert set(loaded_cathode_result.rhs_terms) == (
            expected_rhs_terms | set(END_SHEATH_CATHODE_ROWS)
        )
        assert np.allclose(
            loaded_cathode_result.rhs_terms["cathode_surface_loss"]["n"],
            cathode_run_result.rhs_terms["cathode_surface_loss"]["n"],
        )
        assert np.allclose(
            loaded_cathode_result.rhs_terms["beam_power_deposition"]["Ee"],
            cathode_run_result.rhs_terms["beam_power_deposition"]["Ee"],
        )
        assert np.allclose(
            loaded_cathode_result.cathode_diagnostics["source_I_tot"],
            cathode_run_result.cathode_diagnostics["source_I_tot"],
        )
        # cathode.I_tot / S_ion_beam / Qeb round-tripped here through the
        # retired _sim3 aliases. Each was a view of a row the two assertions
        # immediately above already round-trip by name -- source_I_tot and
        # the beam Ee terms -- so their removal leaves the read path covered.
        assert np.allclose(
            loaded_cathode_result.cathode_diagnostics["solve_enabled"],
            cathode_run_result.cathode_diagnostics["solve_enabled"],
        )
        assert np.allclose(
            loaded_cathode_result.cathode_diagnostics["floating"],
            cathode_run_result.cathode_diagnostics["floating"],
        )
        assert np.allclose(
            loaded_cathode_result.cathode_diagnostics["beam_cross"],
            cathode_run_result.cathode_diagnostics["beam_cross"],
        )
        assert np.all(
            loaded_cathode_result.cathode_diagnostics["source_regime"]
            == cathode_run_result.cathode_diagnostics["source_regime"]
        )
        assert np.allclose(
            loaded_cathode_result.electron_energy_terms_W_cm3[
                "beam_ionization_cost"
            ],
            cathode_run_result.electron_energy_terms_W_cm3[
                "beam_ionization_cost"
            ],
        )

    sparse_params = dict(no_source_params)
    sparse_params["dt_save"] = 1.0e-10
    sparse_params["t_save_start"] = 1.0e-10
    sparse_params["max_output_steps"] = 2
    sparse_sim = LAPDSim1D(sparse_params, flags)
    sparse_result = sparse_sim.run(t_end=4.0e-10, dt=1.0e-10)
    assert sparse_result.steps == 4
    assert sparse_result.time.shape == (2,)
    assert np.allclose(sparse_result.time, [1.0e-10, 2.0e-10])

    adaptive_params = dict(no_source_params)
    adaptive_params["dt_save"] = 1.0e-10
    adaptive_sim = LAPDSim1D(adaptive_params, flags)
    adaptive_result = adaptive_sim.run(t_end=2.5e-10)
    assert adaptive_result.steps == 3
    assert np.allclose(adaptive_result.time, [0.0, 1.0e-10, 2.0e-10, 2.5e-10])
    assert [diag.active_constraint for diag in adaptive_result.diagnostics] == [
        "heat_conduction",
        "heat_conduction",
        "heat_conduction",
    ]
    assert [diag.step_cap for diag in adaptive_result.diagnostics] == [
        "save_time",
        "save_time",
        "t_end",
    ]
    assert np.allclose(
        [diag.accepted_dt for diag in adaptive_result.diagnostics],
        [1.0e-10, 1.0e-10, 0.5e-10],
    )
    adaptive_summary = summarize_result(adaptive_result)
    assert adaptive_summary.constraint_counts == {"heat_conduction": 3}
    assert adaptive_summary.step_cap_counts == {"save_time": 2, "t_end": 1}
    assert np.isclose(adaptive_summary.accepted_dt_min, 0.5e-10)
    assert np.isclose(adaptive_summary.accepted_dt_max, 1.0e-10)

    growth_params = dict(no_source_params)
    growth_params["dt_save"] = 0.0
    growth_params["dt_max"] = 1.0e-6
    growth_params["dt_growth_factor"] = 1.25
    growth_params["tau_prebreakdown"] = 0.5e-6
    growth_params["tau_discharge"] = 10.0e-6
    growth_flags = dict(flags)
    growth_flags["heat_conduction"] = False
    growth_sim = LAPDSim1D(growth_params, growth_flags)
    growth_result = growth_sim.run(t_end=1.5e-6)
    # The surface_loss drain bound is always evaluated, so it binds the
    # first step and, once the ramp has re-approached it, the last two
    # before t_end; the phase boundary cuts the step that crosses it, and
    # the four steps after it are the dt_growth ramp (each 1.25x the last).
    assert growth_result.steps == 9
    assert np.allclose(
        growth_result.time,
        [
            0.0, 4.39038e-7, 5.0e-7, 5.76202e-7, 6.71455e-7, 7.90522e-7,
            9.39354e-7, 1.18683e-6, 1.38459e-6, 1.5e-6,
        ],
        rtol=1.0e-5, atol=0.0,
    )
    assert [diag.step_cap for diag in growth_result.diagnostics] == [
        "surface_loss",
        "phase_boundary",
        "dt_growth",
        "dt_growth",
        "dt_growth",
        "dt_growth",
        "surface_loss",
        "surface_loss",
        "t_end",
    ]
    growth_dts = [diag.accepted_dt for diag in growth_result.diagnostics]
    assert np.allclose(
        growth_dts,
        [
            4.39038e-7, 6.09619e-8, 7.62024e-8, 9.52530e-8, 1.19066e-7,
            1.48833e-7, 2.47481e-7, 1.97755e-7, 1.15411e-7,
        ],
        rtol=1.0e-5, atol=0.0,
    )
    for growth_step in range(2, 6):
        assert np.isclose(
            growth_dts[growth_step], 1.25 * growth_dts[growth_step - 1],
            rtol=1.0e-12, atol=0.0,
        ), growth_step
    growth_summary = summarize_result(growth_result)
    assert growth_summary.step_cap_counts == {
        "dt_growth": 4,
        "phase_boundary": 1,
        "surface_loss": 3,
        "t_end": 1,
    }
    return locals()


# --------------------------------------------------------------------
# electrode-sample-smoothing
# --------------------------------------------------------------------
@_case(
    "electrode-sample-smoothing",
    historical_stance=True,
    provides=("r1a_flags", "r1a_params"),
)
def _case_electrode_sample_smoothing(m3_params):
    # --- Electrode sample smoothing: EMA of the sampled cathode/anode-flank
    # (n, Te) at the presheath transit time, accepted-steps only; the solve
    # reads the smoothed state.
    resolved_cathode_flags = _resolved_cathode_flags()
    # The I_i-vs-n proportionality asserted below at rtol=1e-9 holds only in
    # the near-vacuum limit: compute_l_b harmonically combines the beam's
    # electron-ion MFP (l_bi ~ 1/n_e) with its electron-NEUTRAL MFP
    # (l_bn = 1/(sigma_b*n_n)). While n_n is negligible l_b is a pure 1/n_e
    # power law and the self-consistent phi_c leaves I_i exactly linear in n;
    # at the realistic direct-run nn0 (2e13) the neutral leg is comparable, so
    # I_i departs from exact linearity (measured ratio 3.00077 instead of 3).
    # That coupling is physical -- pin the low fill this identity is stated in
    # rather than loosening the tolerance.
    ss_sim = LAPDSim1D(
        dict(m3_params, nn0=1.0e9),
        resolved_cathode_flags,
    )
    ss_cath = cathode_sample_indices(ss_sim.geometry)[0]
    ss_aface = int(ss_sim.geometry.anode_face_indices[0])
    assert set(ss_sim._sample_smooth_cells) == {ss_cath, ss_aface - 1, ss_aface}
    # Seeded from the initial state: the patched state is initially identical.
    ss_state0 = ss_sim.state
    ss_patched0 = ss_sim._smoothed_sample_state(ss_state0)
    assert np.allclose(ss_patched0.n, ss_state0.n, rtol=1e-14)
    # Hand-check the EMA blend: perturb the state, accept one step, verify
    # ema' = ema + (1 - exp(-dt/tau)) * (x - ema) with tau = l / c_s(Te_ema).
    ss_n_old, ss_Te_old = ss_sim._sample_ema[ss_cath]
    ss_state_p = ss_sim.state
    ss_n_new = float(ss_state_p.n[ss_cath]) * 2.0
    ss_sim._state.n[ss_cath] = ss_n_new
    ss_dt = 1.0e-6
    ss_sim._update_sample_smoothing(ss_dt)
    from cablp.solvers._sim1d.physics.flux import ion_sound_speed as _ss_cs
    ss_tau = float(ss_sim.geometry.length_cm[ss_cath]) / _ss_cs(
        max(ss_Te_old, ss_sim.floors["Te"]), ss_sim.ion_mass_g
    )
    ss_alpha = 1.0 - np.exp(-ss_dt / ss_tau)
    assert np.isclose(
        ss_sim._sample_ema[ss_cath][0],
        ss_n_old + ss_alpha * (ss_n_new - ss_n_old),
        rtol=1e-12,
    )
    # The solve consumes the smoothed sample: with the EMA pinned at the
    # unperturbed density, doubling the instantaneous cathode-cell density
    # must NOT move the solve, and forcing the EMA must move it.
    ss_sim._circuit_I_loop = 800.0
    ss_sim._sample_ema[ss_cath][0] = ss_n_old  # pin the EMA
    ss_res_b = ss_sim.solve_cathode_boundary(update_cache=False)
    ss_sim._state.n[ss_cath] = ss_n_new * 4.0  # instantaneous state ignored
    ss_res_b2 = ss_sim.solve_cathode_boundary(update_cache=False)
    assert np.isclose(
        ss_res_b2.beam_result.result.I_i,
        ss_res_b.beam_result.result.I_i,
        rtol=1e-12,
    )
    ss_sim._sample_ema[ss_cath][0] = ss_n_old * 3.0  # the EMA moves the solve
    ss_res_c = ss_sim.solve_cathode_boundary(update_cache=False)
    assert np.isclose(
        ss_res_c.beam_result.result.I_i,
        3.0 * ss_res_b.beam_result.result.I_i,
        rtol=1e-9,
    )

    # R1a: one authoritative active-plasma topology. Every closed face has at
    # most one live-side cell, pressure work is invariant to the dead-side
    # velocity, and plasma rows in plenum/obstruction cells are bit-invariant
    # through multiple accepted steps when the (default-on) repair is enabled.
    r1a_params, r1a_flags = default_config()
    r1a_params.update(
        {
            "nx": 8,
            "nx_gap": 2,
            "ne0": 2.0e10,
            "nn0": 2.0e12,
            "Te0": 1.0,
            "Ti0": 0.5,
            "phase_transition_mode": "scheduled",
            "tau_prebreakdown": 0.0,
            "tau_breakdown": 0.0,
            "tau_discharge": 1.0e-6,
            "initial_neutral_state": "fill",
        }
    )
    r1a_flags.update(
        {
            "cathode_coupling": False,
            # This block and the R1b/R1c blocks built on it step with
            # ``operator_split=False`` on purpose -- they are about the
            # explicit operator's own rows and rejection machinery. The
            # split stance moves the anode electron-sheath debit into the
            # implicit substep, and a non-split step then has nowhere to
            # apply it, which the solver refuses; so the stance here is the
            # one these steps actually take.
            "implicit_heat_conduction": False,
        }
    )
    # R1b below walks the five/six/seven/eight-row layouts by ADDING closure
    # flags to this base, so the base has to be the five-row one.
    _pin_pre_r2a_neutral_stance(r1a_params, r1a_flags)
    r1a_sim = LAPDSim1D(r1a_params, r1a_flags)
    r1a_geom = r1a_sim.geometry
    r1a_active = np.asarray(r1a_geom.plasma_active, dtype=bool)
    r1a_dead = ~r1a_active
    assert np.any(r1a_dead)
    for r1a_face in np.flatnonzero(~r1a_geom.plasma_open):
        adjacent = []
        if r1a_face > 0 and r1a_active[r1a_face - 1]:
            adjacent.append(r1a_face - 1)
        if r1a_face < r1a_geom.cells and r1a_active[r1a_face]:
            adjacent.append(r1a_face)
        expected_live = adjacent[0] if adjacent else -1
        assert int(r1a_geom.plasma_face_live_cell[r1a_face]) == expected_live

    r1a_state = r1a_sim.state
    r1a_M_perturbed = r1a_state.M.copy()
    r1a_M_perturbed[r1a_dead] = 1.0e6
    r1a_dead_fast = ConservativeState1D(
        n=r1a_state.n.copy(),
        nn=r1a_state.nn.copy(),
        M=r1a_M_perturbed,
        Ee=r1a_state.Ee.copy(),
        Ei=r1a_state.Ei.copy(),
    )
    div_reference = velocity_divergence(
        r1a_state,
        r1a_sim.floors,
        r1a_sim.ion_mass_g,
        r1a_geom,
    )
    div_dead_fast = velocity_divergence(
        r1a_dead_fast,
        r1a_sim.floors,
        r1a_sim.ion_mass_g,
        r1a_geom,
    )
    assert np.array_equal(div_reference[r1a_active], div_dead_fast[r1a_active])

    r1a_initial = r1a_sim.state
    r1a_dead_initial = {
        name: getattr(r1a_initial, name)[r1a_dead].copy()
        for name in ("n", "M", "Ee", "Ei")
    }
    for _ in range(4):
        r1a_sim.advance_one_step(dt=1.0e-10, operator_split=False)
    r1a_final = r1a_sim.state
    for name, initial_values in r1a_dead_initial.items():
        assert np.array_equal(getattr(r1a_final, name)[r1a_dead], initial_values)
    for term_name, term in r1a_sim.rhs_terms().items():
        if term_name in {
            "neutral_zone_exchange",
            "neutral_momentum_wall",
            "neutral_wind_advection",
            "neutral_exchange",
            "neutral_sources",
        }:
            continue
        for field_name in ("n", "nn", "M", "Ee", "Ei"):
            assert np.array_equal(
                getattr(term, field_name)[r1a_dead],
                np.zeros(np.count_nonzero(r1a_dead)),
            )
    return locals()


# --------------------------------------------------------------------
# cathode-closed-audit-export
# --------------------------------------------------------------------
@_case("cathode-closed-audit-export", historical_stance=True)
def _case_cathode_closed_audit_export():
    # THE CLOSED AUDIT SET REACHES THE FILE, AND THE PRE-CLOSURE SCALARS DO
    # NOT. The circuit has computed a closed, surface-resolved power and
    # current audit on every current-driven solve since R3.2; until the
    # export retirement none of it was saved, and what WAS saved beside it
    # were six scalars from the sim3-era power book that does not close.
    # This case runs a short current-driven march, saves it, reads it back,
    # and asserts both halves of that trade on the FILE -- not on an
    # in-memory result -- because the file is what every consumer reads.
    from cablp.solvers._sim1d.results.io import save_result_hdf5 as _ce_save
    from cablp.solvers._sim1d.results.cathode_diagnostics import (
        RETIRED_CATHODE_DIAGNOSTICS,
        RETIRED_CATHODE_DIAGNOSTIC_KEYS,
        RetiredCathodeDiagnosticError,
    )
    from cablp.solvers._sim1d.solver import (
        _CATHODE_RESULT_KEYS,
        _CURRENT_DRIVEN_ONLY_CATHODE_KEYS,
    )

    ce_cu_params, ce_cu_flags = _cathode_unit_config()
    ce_params = dict(ce_cu_params)
    ce_params.update(
        {
            "V_bank": 173.6,
            "R_comp": 5.72e-3,
            "L_parasitic_H": 6.6e-6,
            "cathode_solver_model": "current_driven",
            "dt_save": 0.0,
        }
    )
    ce_flags = dict(
        ce_cu_flags, cathode_coupling=True
    )
    ce_sim = LAPDSim1D(ce_params, ce_flags)
    # Start the loop at a real discharge current. The residual gate below is
    # ROW-RELATIVE, and a ratio against a row that is itself at roundoff --
    # which is what a loop starting from 0 gives over four 0.1 ns steps --
    # would gate on noise instead of on closure.
    ce_sim._circuit_I_loop = 800.0
    ce_result = ce_sim.run(t_end=3.0e-10, dt=1.0e-10)

    # The 22 fields the retirement added to the export, written out rather
    # than derived from the tuple under test: a gate that reads its
    # expectation out of the thing it is checking checks nothing.
    ce_closed = (
        "I_i_a",
        "P_cathode_e_thermal", "P_cathode_e_phi",
        "P_cathode_i_thermal", "P_cathode_i_phi",
        "P_anode_e_thermal", "P_anode_e_phi",
        "P_anode_i_thermal", "P_anode_i_phi",
        "P_plasma_thermal_loss", "P_into_plasma",
        "P_cathode_surface", "P_anode_surface",
        "V_series", "I_parallel", "V_dis", "I_plasma", "I_bank",
        "I_e_ret", "P_load_ledger", "P_load_residual",
        "I_cathode_kirchhoff_residual",
    )
    assert len(set(ce_closed)) == 22
    assert set(ce_closed) <= set(_CATHODE_RESULT_KEYS)
    # Every member of the current-driven-only set is one of them, and I_i_a
    # is the one member both solves compute.
    assert _CURRENT_DRIVEN_ONLY_CATHODE_KEYS == set(ce_closed) - {"I_i_a"}

    with tempfile.TemporaryDirectory() as ce_dir:
        ce_path = f"{ce_dir}/closed_audit.h5"
        _ce_save(ce_path, ce_result)
        ce_loaded = load_result_hdf5(ce_path)
        ce_dg = ce_loaded.cathode_diagnostics

        # (a) PRESENT AND FINITE. This run has one cathode, so it carries
        # the `source_` datasets and NONE of the `end_` ones: the twin block
        # is presence-gated on TwinCathode, and on a single cathode it would
        # have been NaN in every frame. Absence, not NaN, is the statement.
        for ce_key in ce_closed:
            assert f"source_{ce_key}" in ce_dg, ce_key
            assert f"end_{ce_key}" not in ce_dg, ce_key
            ce_vals = np.asarray(ce_dg[f"source_{ce_key}"], dtype=float)
            assert ce_vals.shape == ce_loaded.time.shape, ce_key
            assert np.all(np.isfinite(ce_vals)), ce_key

        # (a2) THE ANODE'S OWN TEMPERATURE AND ITS ENERGY PAIR. The solve's
        # anode block runs on ``T_e_anode`` and every anode member is
        # referenced to it, so it is exported beside them. The four
        # ``anode_e_sheath_*`` scalars are the implicit substep's
        # reconciliation -- what the circuit charged against what the
        # plasma paid -- and are UNCONDITIONAL, so they are present here
        # even on a run whose window books nothing.
        assert "T_e_anode" in _CATHODE_RESULT_KEYS
        assert "T_e_anode" not in ce_closed
        assert "T_e_anode" not in _CURRENT_DRIVEN_ONLY_CATHODE_KEYS
        ce_Te_a = np.asarray(ce_dg["source_T_e_anode"], dtype=float)
        assert ce_Te_a.shape == ce_loaded.time.shape
        assert np.all(np.isfinite(ce_Te_a))
        ce_has_solution = (
            np.asarray(ce_dg["has_solution"], dtype=float) == 1.0
        )
        assert np.all(ce_Te_a[ce_has_solution] > 0.0), ce_Te_a
        assert "end_T_e_anode" not in ce_dg
        # (a3) THE TAIL'S OWN SHEATH-FALL MOMENT. ``P_tail_phi`` is the
        # partner of ``P_anode_e_phi`` for the current the wires take out
        # of the QL tail rather than out of the thermal return. All three
        # circuit variants compute it, so unlike the 22 above it is NOT
        # current-driven-only; it is exported all the same, and it is
        # identically 0.0 on a run that hands in no tail current -- this
        # one -- because the deposition that measures it is solved after
        # the circuit and so is lagged a step behind.
        assert "P_tail_phi" in _CATHODE_RESULT_KEYS
        assert "P_tail_phi" not in ce_closed
        assert "P_tail_phi" not in _CURRENT_DRIVEN_ONLY_CATHODE_KEYS
        ce_tail_phi = np.asarray(ce_dg["source_P_tail_phi"], dtype=float)
        assert ce_tail_phi.shape == ce_loaded.time.shape
        assert np.all(np.isfinite(ce_tail_phi))
        assert np.all(ce_tail_phi == 0.0), ce_tail_phi
        assert "end_P_tail_phi" not in ce_dg
        for ce_pair_key in (
            "anode_e_sheath_booked_W", "anode_e_sheath_realised_W",
            "anode_e_sheath_booked_J", "anode_e_sheath_realised_J",
        ):
            assert ce_pair_key in ce_dg, ce_pair_key
            ce_pair_vals = np.asarray(ce_dg[ce_pair_key], dtype=float)
            assert ce_pair_vals.shape == ce_loaded.time.shape, ce_pair_key
            assert np.all(np.isfinite(ce_pair_vals)), ce_pair_key
            assert np.all(ce_pair_vals >= 0.0), ce_pair_key
        # The cumulative rows never decrease.
        for ce_pair_key in (
            "anode_e_sheath_booked_J", "anode_e_sheath_realised_J",
        ):
            ce_pair_vals = np.asarray(ce_dg[ce_pair_key], dtype=float)
            assert np.all(np.diff(ce_pair_vals) >= 0.0), ce_pair_key

        # (b) THE RESIDUALS ARE THE CLOSURE NUMBERS. Row-relative against the
        # row each one is a residual OF, and both rows are asserted physical
        # first so the normalization is meaningful.
        ce_solved = np.asarray(ce_dg["has_solution"], dtype=float) == 1.0
        ce_floating = np.asarray(ce_dg["floating"], dtype=float) == 1.0
        ce_cd = ce_solved & ~ce_floating
        assert ce_cd.any()
        ce_P = np.asarray(ce_dg["source_P_load"], dtype=float)[ce_cd]
        ce_I = np.asarray(ce_dg["source_I_tot"], dtype=float)[ce_cd]
        ce_rP = np.asarray(
            ce_dg["source_P_load_residual"], dtype=float
        )[ce_cd]
        ce_rI = np.asarray(
            ce_dg["source_I_cathode_kirchhoff_residual"], dtype=float
        )[ce_cd]
        assert np.all(np.abs(ce_I) > 1.0), ce_I
        assert np.all(np.abs(ce_P) > 1.0), ce_P
        assert np.all(np.abs(ce_rP) <= 1.0e-9 * np.abs(ce_P)), (ce_rP, ce_P)
        assert np.all(np.abs(ce_rI) <= 1.0e-9 * np.abs(ce_I)), (ce_rI, ce_I)

        # (c) THE SIX RETIRED NAMES ARE NOT ON A NEW FILE.
        assert len(RETIRED_CATHODE_DIAGNOSTICS) == 6
        assert len(RETIRED_CATHODE_DIAGNOSTIC_KEYS) == 12
        for ce_name in RETIRED_CATHODE_DIAGNOSTIC_KEYS:
            assert ce_name not in ce_dg, ce_name
        for ce_bare in RETIRED_CATHODE_DIAGNOSTICS:
            assert ce_bare not in _CATHODE_RESULT_KEYS, ce_bare

        # (d) READING ONE RAISES, NAMING THE SUCCESSOR -- through `[]` and
        # through `.get`, because a `.get` default would substitute an
        # invented number for the withdrawn one.
        for ce_bare, ce_advice in RETIRED_CATHODE_DIAGNOSTICS.items():
            ce_successor = ce_advice.split()[0]
            for ce_prefix in ("source", "end"):
                ce_name = f"{ce_prefix}_{ce_bare}"
                for ce_read in (
                    lambda d, k: d[k],
                    lambda d, k: d.get(k),
                    lambda d, k: d.get(k, 0.0),
                ):
                    try:
                        ce_read(ce_dg, ce_name)
                    except RetiredCathodeDiagnosticError as ce_exc:
                        ce_text = str(ce_exc)
                        assert ce_name in ce_text, ce_text
                        assert ce_successor in ce_text, (ce_name, ce_text)
                    else:
                        raise AssertionError(
                            f"reading retired {ce_name!r} did not raise"
                        )
        # The refusal is targeted: an ordinary missing key is still an
        # ordinary KeyError, not a retirement message.
        try:
            ce_dg["source_not_a_diagnostic"]
        except RetiredCathodeDiagnosticError:
            raise AssertionError(
                "an unrelated missing key raised the retirement error"
            )
        except KeyError:
            pass

        # (e) NEGATIVE CONTROL: an OLD-FORMAT file, built here by writing the
        # retired datasets back onto a copy, still reads them. The retirement
        # withdraws a name from the export; it does not rewrite the record of
        # what an earlier build computed. Without this leg the refusal above
        # would be indistinguishable from "these names never read at all".
        ce_old_path = f"{ce_dir}/closed_audit_old_format.h5"
        shutil.copyfile(ce_path, ce_old_path)
        ce_sentinel = 7.5
        with h5py.File(ce_old_path, "r+") as ce_h5:
            for ce_name in sorted(RETIRED_CATHODE_DIAGNOSTIC_KEYS):
                ce_h5["cathode_diagnostics"].create_dataset(
                    ce_name,
                    data=np.full(ce_loaded.time.shape, ce_sentinel),
                )
        ce_old_dg = load_result_hdf5(ce_old_path).cathode_diagnostics
        for ce_name in RETIRED_CATHODE_DIAGNOSTIC_KEYS:
            assert ce_name in ce_old_dg, ce_name
            assert np.allclose(ce_old_dg[ce_name], ce_sentinel), ce_name
            assert np.allclose(ce_old_dg.get(ce_name), ce_sentinel), ce_name

    # (f) THE FLOATING PATH EXPORTS NaN, NOT ZERO. A solve that leaves the
    # audit set at its dataclass defaults would export zeros, and a zero in a
    # power column is indistinguishable from a computed zero. Exercised on the
    # solve already in hand rather than on a second run: the branch under
    # test is the export's, not the circuit's.
    ce_solve = ce_sim.solve_cathode_boundary(update_cache=False)
    ce_diag_nan = {}
    ce_sim._copy_cathode_result_diagnostics(
        diag=ce_diag_nan,
        prefix="source",
        result=ce_solve.beam_result.result,
        current_driven=False,
    )
    for ce_key in _CURRENT_DRIVEN_ONLY_CATHODE_KEYS:
        assert np.isnan(ce_diag_nan[f"source_{ce_key}"]), ce_key
    assert np.isfinite(ce_diag_nan["source_I_i_a"])
    assert np.isfinite(ce_diag_nan["source_I_tot"])


# --------------------------------------------------------------------
# circuit-cathode-retired-keys-refuse
# --------------------------------------------------------------------
@_case("circuit-cathode-retired-keys-refuse")
def _case_circuit_cathode_retired_keys_refuse():
    # The circuit and cathode selectors, flags and parameters removed with the
    # closures they served. Each is gone from its template, is on the retired
    # register of ITS OWN namespace, and a configuration naming it -- at any
    # value, the old default included -- is refused at construction with the
    # key named as RETIRED. A retired name filed in the OTHER namespace reads
    # as the plain unknown key it is there.
    from cablp.solvers._sim1d.core.config import (
        RETIRED_FLAG_KEYS,
        RETIRED_PARAM_KEYS,
        input_dict_template_1d,
        input_flags_template_1d,
    )

    _rk_params = {
        "cathode_model": "disabled",
        "cathode_warming_model": "power_balance",
        "cathode_surface_model": "ads_des",
        "cathode_sample_smoothing": "presheath",
        "cathode_emission_profile": "uniform",
        "cathode_Ts_fwhm_cm": 28.0,
        "cathode_emission_annuli": 10,
        "cathode_emitting_area_initial_fraction": 0.0075,
        "cathode_Rp_model": "sample",
        "cathode_lnL_model": "nrl_ei",
        "cathode_circuit_sample": "raw",
        "cathode_circuit_bound_object": "device_voltage",
        "circuit_dt_fraction": 0.25,
        "circuit_picard_tol_rel": 1.0e-2,
        "circuit_picard_max_iter": 3,
        "cathode_ion_secondary_emission_yield": None,
        "vessel_capacitance_F": 1.3e-6,
        "vessel_leak_resistance_ohm": 1.0e10,
    }
    _rk_flags = {
        "cathode_schottky": True,
        "anode_sheath_full_debit": True,
        "cathode_emission_bridge": False,
        "cathode_emitting_area": False,
        "cathode_enthalpy_on_beam": False,
        "cathode_ion_secondary_emission": False,
        "coupled_circuit_picard": False,
        "cathode_circuit_voltage_bound": False,
        "cathode_circuit_project_over_wall": False,
        "regime_vessel_node": False,
    }
    _rk_base_p, _rk_base_f = default_config()
    for _rk_key, _rk_value in _rk_params.items():
        assert _rk_key not in input_dict_template_1d, _rk_key
        assert _rk_key not in input_flags_template_1d, _rk_key
        assert _rk_key in RETIRED_PARAM_KEYS, _rk_key
        try:
            LAPDSim1D(dict(_rk_base_p, **{_rk_key: _rk_value}), _rk_base_f)
        except ValueError as _rk_exc:
            assert f"{_rk_key} is RETIRED" in str(_rk_exc), str(_rk_exc)
        else:
            raise AssertionError(f"retired params key {_rk_key} ACCEPTED")
    for _rk_key, _rk_value in _rk_flags.items():
        assert _rk_key not in input_dict_template_1d, _rk_key
        assert _rk_key not in input_flags_template_1d, _rk_key
        assert _rk_key in RETIRED_FLAG_KEYS, _rk_key
        try:
            LAPDSim1D(_rk_base_p, dict(_rk_base_f, **{_rk_key: _rk_value}))
        except ValueError as _rk_exc:
            assert f"{_rk_key} is RETIRED" in str(_rk_exc), str(_rk_exc)
        else:
            raise AssertionError(f"retired flags key {_rk_key} ACCEPTED")
    # A retired FLAG name in params is a misfiled key, not a retired one.
    try:
        LAPDSim1D(dict(_rk_base_p, cathode_schottky=True), _rk_base_f)
    except ValueError as _rk_exc:
        assert "unknown LAPDSim1D configuration keys" in str(_rk_exc)
        assert "RETIRED" not in str(_rk_exc), str(_rk_exc)
    else:
        raise AssertionError("a misfiled retired flag name was ACCEPTED")


# --------------------------------------------------------------------
# ts-retirement-successor-key
# --------------------------------------------------------------------
@_case("ts-retirement-successor-key", historical_stance=True)
def _case_ts_retirement_successor_key():
    # --- T_s IS RETIRED (2026-09-03): a sim3-era development artifact that
    # every solver call site already overrode, leaving it inert under the
    # production warming model and live on only two paths. Both now read
    # ``cathode_Ts_base_K``, the standby the power balance evolves from, and
    # the retired name is refused at
    # construction with the successor named -- never silently accepted, and
    # never silently inert.
    from cablp.solvers._sim1d.core.config import (
        RETIRED_PARAM_KEYS,
        input_dict_template_1d,
        input_flags_template_1d,
    )

    # (a) the name is gone from BOTH namespaces and is on the retired register
    assert "T_s" not in input_dict_template_1d
    assert "T_s" not in input_flags_template_1d
    assert "T_s" in RETIRED_PARAM_KEYS
    assert "cathode_Ts_base_K" in input_dict_template_1d

    # (b) naming it in input_dict is refused, and the refusal SAYS WHAT
    # REPLACED IT -- the case a stored pre-removal configuration file hits.
    _ts_p, _ts_f = default_config()
    try:
        LAPDSim1D(dict(_ts_p, T_s=1998.15), _ts_f)
    except ValueError as _ts_exc:
        _ts_msg = str(_ts_exc)
    else:
        raise AssertionError("T_s was ACCEPTED in input_dict")
    assert "unknown LAPDSim1D configuration keys" in _ts_msg, _ts_msg
    assert "T_s is RETIRED" in _ts_msg, _ts_msg
    assert "cathode_Ts_base_K" in _ts_msg, _ts_msg

    # (c) the flags namespace refuses it too: the two are separate namespaces
    # and a key misfiled into the other one still raises rather than going
    # inert.
    try:
        LAPDSim1D(_ts_p, dict(_ts_f, T_s=True))
    except ValueError as _ts_fexc:
        assert "flags=['T_s']" in str(_ts_fexc), str(_ts_fexc)
    else:
        raise AssertionError("T_s was ACCEPTED in input_flags")

    # (d) NEGATIVE CONTROL on the message class. The successor sentence is
    # scoped to the retired name: an ordinary unknown key must still get the
    # bare refusal, with no successor attached. Without the scope this
    # assertion fails, which is what makes (b) evidence rather than decoration.
    try:
        LAPDSim1D(dict(_ts_p, not_a_key=1.0), _ts_f)
    except ValueError as _ts_unk:
        _ts_unk_msg = str(_ts_unk)
    else:
        raise AssertionError("an unknown key was ACCEPTED")
    assert "unknown LAPDSim1D configuration keys" in _ts_unk_msg, _ts_unk_msg
    assert "RETIRED" not in _ts_unk_msg, _ts_unk_msg
    assert "cathode_Ts_base_K" not in _ts_unk_msg, _ts_unk_msg

    # (e) A HELD SURFACE reads the standby. This was T_s's last live read on
    # the fluid path, so the surface the emission solve is built at, and the
    # surface the diagnostics report, must both be the configured
    # cathode_Ts_base_K -- checked at a value that is nobody's default, so a
    # read that fell back elsewhere could not pass by coincidence. The unit
    # stance holds the power-balance surface at its standby.
    _ts_static, _ts_sflags = _cathode_unit_config()
    _ts_static.update({
        "nx": 12,
        "dt_save": 0.0,
        "cathode_Ts_base_K": 1873.0,
        "cathode_solver_model": "current_driven",
        "V_bank": 173.6,
        "R_comp": 5.72e-3,
    })
    _ts_sflags = dict(
        _ts_sflags, cathode_coupling=True,
    )
    _ts_sim = LAPDSim1D(_ts_static, _ts_sflags)
    _ts_sim._circuit_I_loop = 800.0
    _ts_dev = _ts_sim.solve_cathode_boundary(update_cache=False).device_config
    assert _ts_dev.T_s == 1873.0, _ts_dev.T_s
    _ts_res = _ts_sim.run(t_end=3.0e-10, dt=1.0e-10)
    assert np.allclose(_ts_res.cathode_diagnostics["T_s_surface"], 1873.0)

    # (g) THE SUCCESSOR IS REQUIRED, AND REFUSED AT CONSTRUCTION WHEN UNSET:
    # with the retired key gone there is nothing to fall back on, so None
    # would reach the emission solve and die as a TypeError deep inside it.
    # It is refused here instead, where the configuration is still the
    # subject.
    _ts_bare_p, _ts_bare_f = default_config()
    _ts_bare_p.update({"nx": 12, "cathode_Ts_base_K": None})
    for _ts_bad_p, _ts_bad_f in (
        (_ts_bare_p, _ts_bare_f),
    ):
        try:
            LAPDSim1D(_ts_bad_p, _ts_bad_f)
        except ValueError as _ts_exc:
            assert "cathode_Ts_base_K" in str(_ts_exc), str(_ts_exc)
        else:
            raise AssertionError("cathode_Ts_base_K=None was ACCEPTED")

    # Positive control: with the key SET the configuration constructs, so the
    # requirement refuses only the unset case and moves no run.
    LAPDSim1D(dict(_ts_bare_p, cathode_Ts_base_K=1910.0), _ts_bare_f)


# ----------------------------------------------------------------------
# floating-open-circuit-current-balance
# ----------------------------------------------------------------------
@_case("floating-open-circuit-current-balance")
def _case_floating_open_circuit_current_balance():
    """The open-circuit cathode point is the current-driven solve at I = 0.

    UNIT FIXTURE on three states of the reference machine -- the seed, the
    discharge plateau, and the late afterglow -- with the circuit scalars
    pinned here rather than read from a stance, so the fixture measures the
    sheath physics and not a stance value.

    WHAT IS ASSERTED
      1. Kirchhoff. The solve's own residual (I_eth* + I_i - I_e,ret) - I_tot
         is <= 1e-12 A on every state. An open circuit that does not conserve
         current is not an open circuit.
      2. The emissive floating point. At the late state the cathode sits
         1-2 Te BELOW the plasma in the classical part of its sheath
         (phi_c_plus / Te in [1, 2]) -- the space-charge-limited emissive
         value -- with the emission current far above the ion current and the
         collected electron current above both. That is the physical content:
         a hot emitter at zero net current COLLECTS, it does not sit at the
         non-emitting floating drop.
      3. Every electrode power field is assigned. The retired branch left the
         *_thermal / *_phi members at their dataclass default of 0.0, so an
         open-circuit phase booked no electrode power at all; here they are
         finite and the cathode thermal booking is strictly negative-going
         work on the plasma store (a positive P_cathode_e_thermal, which the
         RHS subtracts).
    """
    from cablp.cathode.circuit_common import (
        DeviceConfig as _fb_DeviceConfig,
        PlasmaState as _fb_PlasmaState,
    )
    from cablp.cathode.circuit_idriven import solve_idriven as _fb_solve_idriven

    # Reference-machine circuit scalars (LaB6 disc, He, the production
    # compliance/anode geometry). Pinned, not read from a stance file: this
    # fixture is about the solve, and a stance edit must not move it.
    _fb_cfg_kw = dict(
        A_c=math.pi * 18.415 ** 2,
        mu=4.0026,
        ion_mass_g=m_He_cgs,
        phi_wf=2.869,
        C_R=9.3,
        R_comp=0.0072244,
        eta=0.358,
        L_cath=53.25,
        R_cath=18.415,
    )
    # (label, n_e [cm^-3], T_e [eV], n_n [cm^-3], T_s [K], anode T_e [eV])
    _fb_states = (
        ("seed", 1.0e9, 0.21, 6.2901005643418e12, 1910.0, 0.21),
        ("plateau", 5.13904847258451e12, 8.598381836576038,
         4.662555042487933e13, 1914.9084700456249, 8.342168945781902),
        ("late_afterglow", 3.424966858999728e11, 0.1,
         4.701649450942045e13, 1913.453373340071, 0.1),
    )
    _fb_alpha = math.exp(-0.5)

    for _fb_label, _fb_n, _fb_Te, _fb_nn, _fb_Ts, _fb_Te_a in _fb_states:
        _fb_cfg = _fb_DeviceConfig(T_s=_fb_Ts, **_fb_cfg_kw)
        _fb_pl = _fb_PlasmaState(T_e=_fb_Te, n_e=_fb_n, n_n=_fb_nn)
        _fb_r = _fb_solve_idriven(
            _fb_cfg, _fb_pl, 0.0,
            anode_T_e=_fb_Te_a,
            alpha_sheath=_fb_alpha,
            alpha_sheath_anode=_fb_alpha,
            phi_c_cap_V=1000.0,
        )
        # 1. Kirchhoff, on the solve's own exported residual.
        # The reported I_tot is the sheath's own RECONSTRUCTION of the
        # imposed zero out of currents of order I_eth*, so it carries
        # root-finder roundoff at that scale, not at 1 A.
        assert abs(_fb_r.I_tot) <= 1.0e-12 * max(_fb_r.I_eth_star, 1.0), (
            _fb_label, _fb_r.I_tot, _fb_r.I_eth_star
        )
        assert abs(_fb_r.I_cathode_kirchhoff_residual) <= 1.0e-12, (
            _fb_label, _fb_r.I_cathode_kirchhoff_residual
        )
        # 3. Every electrode field assigned and finite.
        for _fb_key in (
            "P_cathode_e_thermal", "P_cathode_e_phi",
            "P_cathode_i_thermal", "P_cathode_i_phi",
            "P_anode_e_thermal", "P_anode_e_phi",
        ):
            assert np.isfinite(getattr(_fb_r, _fb_key)), (_fb_label, _fb_key)
        assert _fb_r.P_cathode_e_thermal > 0.0, (
            _fb_label, _fb_r.P_cathode_e_thermal
        )
        assert _fb_r.P_anode_e_thermal > 0.0, (
            _fb_label, _fb_r.P_anode_e_thermal
        )

    # 2. The emissive floating point, at the late-afterglow state.
    _fb_cfg = _fb_DeviceConfig(T_s=1913.453373340071, **_fb_cfg_kw)
    _fb_late = _fb_solve_idriven(
        _fb_cfg,
        _fb_PlasmaState(T_e=0.1, n_e=3.424966858999728e11,
                        n_n=4.701649450942045e13),
        0.0,
        anode_T_e=0.1,
        alpha_sheath=_fb_alpha,
        alpha_sheath_anode=_fb_alpha,
        phi_c_cap_V=1000.0,
    )
    _fb_ratio = _fb_late.phi_c_plus / 0.1
    assert 1.0 <= _fb_ratio <= 2.0, _fb_ratio
    _fb_I_ret = _fb_late.I_eth_star + _fb_late.I_i - _fb_late.I_tot
    assert _fb_late.I_eth_star > 10.0 * _fb_late.I_i, (
        _fb_late.I_eth_star, _fb_late.I_i
    )
    assert _fb_I_ret > _fb_late.I_eth_star > 0.0, (
        _fb_I_ret, _fb_late.I_eth_star
    )


# --------------------------------------------------------------------
# cathode-emitted-fall-beam-row-non-overlap
# --------------------------------------------------------------------
@_case("cathode-emitted-fall-beam-row-non-overlap")
def _case_cathode_emitted_fall_beam_row_non_overlap(
    id_plasmas, plasma_probe, solve_idriven, uni_cfg
):
    # THE IDENTITY THE BEAM-ROW NON-OVERLAP RESTS ON.
    # `cathode_e_emitted_fall` books e (phi_c_plus - max(phi_c, 0)) Gamma_em:
    # the part of the cathode fall the released electrons drop through that
    # the beam deposition row does not already carry. The beam row carries the
    # NET phi_c, so the two rows may not overlap -- and the whole reason they
    # do not is that the circuit defines phi_c = phi_c_plus - phi_c_minus, so
    # with no virtual cathode (phi_c_minus = 0) the two potentials are the
    # same number and the fall row is EXACTLY zero, not small.
    #
    # Checked over a DRIVEN DISCHARGE state of the same circuit the solver's
    # cathode solve calls, on both sides of the threshold: a hot, dense plasma
    # is released-classical at every drive current (the emitter delivers its
    # full space-charge-released current and the sheath never inverts), and a
    # cold, thin one is a deep virtual cathode at the same currents. The
    # second is the NEGATIVE CONTROL: it is what says the zero above is the
    # identity holding and not the row being dead.
    # The fall row does not read T_s at all -- it is
    # (phi_c_plus - max(phi_c, 0)) I_eth_star -- so this only has to be the
    # surface temperature the fixture's own device is at, which is what makes
    # the enthalpy row a live comparator beside it.
    T_s_K = float(uni_cfg.T_s)
    assert T_s_K > 0.0, T_s_K

    # (i) DRIVEN, NO VIRTUAL CATHODE -> the fall row is exactly zero.
    zero_points = 0
    for _fb_I in (20.0, 100.0, 300.0, 600.0, 1000.0, 2000.0):
        _fb_r = solve_idriven(uni_cfg, plasma_probe, I_tot_A=_fb_I)
        # The circuit's own definition, first: everything below reads off it.
        assert _fb_r.phi_c == _fb_r.phi_c_plus - _fb_r.phi_c_minus, _fb_I
        assert _fb_r.phi_c_minus == 0.0, (_fb_I, _fb_r.phi_c_minus)
        assert _fb_r.phi_c == _fb_r.phi_c_plus, _fb_I
        assert _fb_r.phi_c > 0.0, (_fb_I, _fb_r.phi_c)
        assert _fb_r.I_eth_star > 0.0, (_fb_I, _fb_r.I_eth_star)
        _fb_enth, _fb_fall, _fb_climb = cathode_emission_sheath_power_W(
            _fb_r, T_s_K
        )
        assert _fb_fall == 0.0, (_fb_I, _fb_fall)
        # ... while the OTHER two rows are live at the same point, so the zero
        # is this row's own property and not a dead solve.
        assert _fb_enth > 0.0, (_fb_I, _fb_enth)
        assert _fb_climb <= 0.0, (_fb_I, _fb_climb)
        zero_points += 1
    assert zero_points == 6, zero_points

    # (ii) NEGATIVE CONTROL -- a virtual cathode has formed, so the row is
    # nonzero and is exactly the inverted part of the drop times the released
    # current, which is the whole of what the beam row does not carry.
    nonzero_points = 0
    for _fb_I in (20.0, 100.0, 300.0):
        _fb_r = solve_idriven(uni_cfg, id_plasmas[1], I_tot_A=_fb_I)
        assert _fb_r.phi_c == _fb_r.phi_c_plus - _fb_r.phi_c_minus, _fb_I
        assert _fb_r.phi_c_minus > 0.0, (_fb_I, _fb_r.phi_c_minus)
        assert _fb_r.phi_c > 0.0, (_fb_I, _fb_r.phi_c)
        _fb_enth, _fb_fall, _fb_climb = cathode_emission_sheath_power_W(
            _fb_r, T_s_K
        )
        assert _fb_fall > 0.0, (_fb_I, _fb_fall)
        assert np.isclose(
            _fb_fall,
            _fb_r.phi_c_minus * _fb_r.I_eth_star,
            rtol=1e-12,
            atol=0.0,
        ), (_fb_I, _fb_fall, _fb_r.phi_c_minus * _fb_r.I_eth_star)
        nonzero_points += 1
    assert nonzero_points == 3, nonzero_points


# ----------------------------------------------------------------------
# effective-cathode-flags-refuses-driven-override-in-floating-phase
# ----------------------------------------------------------------------
@_case(
    "effective-cathode-flags-refuses-driven-override-in-floating-phase",
    provides=(),
)
def _case_effective_cathode_flags_refuses_driven_override_in_floating_phase():
    """``_effective_cathode_flags`` refuses the driven override off-phase.

    ``active_only=False, floating=False`` asks for the DRIVEN mapping
    regardless of phase -- the circuit advance passes exactly this, and
    returns before reaching the call on ``step_phase["floating"]``, so the
    override is inert for it by construction. A caller that CAN reach a
    floating phase and still passes this override is handed a configuration
    that does not exist (a floating phase reported as
    ``cathode_coupling=False``), which is the same class of silent mis-booking
    the hand-off and final-step-boundary fixes closed (2026-09-10) -- so the
    method now refuses it loudly instead.

    The sim is constructed but never run, so ``_circuit_I_prev`` stays at
    its construction-time 0.0 and the inductive-tail exception (which needs
    ``_circuit_I_prev > 1.0``) cannot fire regardless of ``L_parasitic_H``:
    ``options["floating"]`` reads True exactly where the scheduled ladder
    says ``afterglow``.
    """
    _ecf_params, _ecf_flags = default_config()
    _ecf_params = dict(_ecf_params)
    _ecf_flags = dict(_ecf_flags)
    _ecf_params["initial_neutral_state"] = "fill"
    _ecf_params["nx"] = 16
    _ecf_params["phase_transition_mode"] = "scheduled"
    _ecf_params["tau_prebreakdown"] = 1.0e-7
    _ecf_params["tau_breakdown"] = 0.0
    _ecf_params["tau_discharge"] = 1.0e-7
    _ecf_params["tau_afterglow"] = 1.0e-7
    _ecf_sim = LAPDSim1D(_ecf_params, _ecf_flags)
    assert _ecf_sim._circuit_I_prev == 0.0, _ecf_sim._circuit_I_prev

    _ecf_afterglow_time = (
        _ecf_sim._plasma_phase_time_origin()
        + float(_ecf_params["tau_prebreakdown"])
        + float(_ecf_params["tau_breakdown"])
        + float(_ecf_params["tau_discharge"])
        + 0.5 * float(_ecf_params["tau_afterglow"])
    )
    _ecf_options = _ecf_sim._cathode_phase_options(time=_ecf_afterglow_time)
    assert _ecf_options["floating"] is True, _ecf_options

    # (i) active_only=False, floating=False in a floating phase: refused,
    # naming the caller's request and the phase's own reading.
    _ecf_raised = None
    try:
        _ecf_sim._effective_cathode_flags(
            time=_ecf_afterglow_time, active_only=False, floating=False
        )
    except ValueError as exc:
        _ecf_raised = exc
    assert _ecf_raised is not None, (
        "no refusal for the driven override in a floating phase"
    )
    assert "active_only=False, floating=False" in str(_ecf_raised), (
        str(_ecf_raised)
    )
    assert "floating=True" in str(_ecf_raised), str(_ecf_raised)

    # (ii) active_only=True at the same time: the phase's own reading, no
    # refusal -- a floating, configured phase reports cathode_coupling True.
    _ecf_flags_out = _ecf_sim._effective_cathode_flags(
        time=_ecf_afterglow_time, active_only=True
    )
    assert _ecf_flags_out["cathode_coupling"] is True, _ecf_flags_out


# ----------------------------------------------------------------------
# cathode-face-one-ion-current
# ----------------------------------------------------------------------
@_case("cathode-face-one-ion-current", historical_stance=True)
def _case_cathode_face_one_ion_current():
    """The fluid's cathode-face ion loss and the circuit's I_i are ONE number.

    (i) With the electrode sample smoothing off, the current the boundary
    operator removes through the cathode face equals the ``I_i`` both circuit
    adapters return at the same state, to 1e-12 relative.
    (ii) With ``presheath`` smoothing the circuit's ``I_i`` is that SAME
    expression evaluated on the smoothed sample, to roundoff, on every
    sampled step -- the EMA is a sampling of the one formula, not a second
    formula. The raw-vs-smoothed spread is a property of the filter and of
    how fast the sample is moving, so it is REPORTED rather than bounded: the
    EMA's time constant is the ion transit across the sampled cell (tens of
    microseconds here) and a nanosecond-scale breakdown transient moves the
    raw sample straight through it.
    (iii) A short run stays positive and finite.
    """
    from cablp.cathode.circuit_common import PlasmaState as _cf_PlasmaState
    from cablp.cathode.circuit_idriven import solve_idriven as _cf_idriven
    from cablp.cathode.circuit_prescribed import (
        solve_prescribed as _cf_prescribed,
    )
    from cablp.solvers._sim1d.physics import flux as _cf_flux
    from cablp.solvers._sim1d.physics.cathode import (
        cathode_circuit_alpha_sheath,
        cathode_device_config,
    )

    _cf_params, _cf_flags = _base_config()
    _cf_params = dict(_cf_params, max_steps_action="stop")
    _cf_flags = dict(_cf_flags)
    _cf_flags["cathode_coupling"] = True
    _cf_params["initial_neutral_state"] = "fill"

    def _cf_face_current(sim):
        """Return ``(I_fluid_A, cell, n, Te, alpha_eff)`` at the cathode face."""
        geom = sim.geometry
        cell = int(absorbing_live_cells_by_role(geom)["cathode"][0])
        Vp = float(np.asarray(geom.plasma_volume_cm3)[cell])
        bnd = sim.characteristic_boundary_rhs(state=sim.state)
        removed = -float(np.asarray(bnd.n)[cell]) * Vp
        derived = derive_state(
            sim.state, floors=sim.floors, ion_mass_g=sim.ion_mass_g
        )
        alpha = cathode_circuit_alpha_sheath(
            sim.state, derived, geom, cell, sim.ion_mass_g, sim._input_dict
        )
        return (
            removed * qe_SI,
            cell,
            float(sim.state.n[cell]),
            float(derived.Te[cell]),
            alpha,
        )

    # (i) ON THE RAW STATE: one number, to roundoff, on BOTH adapters.
    _cf_raw = LAPDSim1D(dict(_cf_params), dict(_cf_flags))
    _cf_I, _cf_cell, _cf_n, _cf_Te, _cf_alpha = _cf_face_current(_cf_raw)
    # The face area and the emitting area are asserted equal at construction,
    # so the only thing left to check is that the two expressions agree.
    _cf_dev = cathode_device_config(
        _cf_raw._input_dict,
        _cf_raw._effective_cathode_flags(time=None, active_only=False),
        _cf_raw.mu,
        _cf_raw.ion_mass_g,
    )
    _cf_pl = _cf_PlasmaState(T_e=_cf_Te, n_e=_cf_n, n_n=0.0, sigma_b=0.0)
    _cf_id = _cf_idriven(
        _cf_dev, _cf_pl, I_tot_A=1200.0, anode_current_A=200.0,
        anode_T_e=_cf_Te, alpha_sheath=_cf_alpha,
    )
    _cf_pr = _cf_prescribed(
        _cf_dev, _cf_pl, I_tot_A=1200.0, V_dis_V=60.0,
        anode_current_A=200.0, anode_T_e=_cf_Te, alpha_sheath=_cf_alpha,
    )
    for _cf_label, _cf_res in (("idriven", _cf_id), ("prescribed", _cf_pr)):
        assert abs(_cf_res.I_i / _cf_I - 1.0) <= 1.0e-12, (
            _cf_label, _cf_res.I_i, _cf_I
        )
    # ...and it is the analytic Bohm current on the emitting area.
    _cf_want = (
        qe_SI * _cf_dev.A_c * _cf_alpha * _cf_n
        * float(_cf_flux.ion_sound_speed(_cf_Te, _cf_raw.ion_mass_g))
    )
    assert abs(_cf_I / _cf_want - 1.0) <= 1.0e-12, (_cf_I, _cf_want)

    # (ii) ON THE SMOOTHED SAMPLE the solve reads: ONE formula, two samples.
    _cf_sim = LAPDSim1D(dict(_cf_params), dict(_cf_flags))
    _cf_worst = 0.0
    _cf_first = None
    _cf_samples = 0
    for _ in range(12):
        _cf_sim.run(t_end=None, dt=None, max_steps=2)
        _cf_step_I = _cf_face_current(_cf_sim)[0]
        _cf_sim.rhs_terms()
        _cf_solve = _cf_sim._cathode_solve
        if _cf_solve is None or _cf_solve.beam_result is None:
            continue
        _cf_circuit_I = float(_cf_solve.beam_result.result.I_i)
        # The circuit's own number, rebuilt from the SMOOTHED sample by the
        # one expression the face uses on the raw one.
        _cf_sm = _cf_sim._smoothed_sample_state(_cf_sim.state)
        _cf_sm_der = derive_state(
            _cf_sm, floors=_cf_sim.floors, ion_mass_g=_cf_sim.ion_mass_g
        )
        _cf_sm_cell = int(
            absorbing_live_cells_by_role(_cf_sim.geometry)["cathode"][0]
        )
        _cf_sm_alpha = cathode_circuit_alpha_sheath(
            _cf_sm, _cf_sm_der, _cf_sim.geometry, _cf_sm_cell,
            _cf_sim.ion_mass_g, _cf_sim._input_dict,
        )
        _cf_sm_I = (
            qe_SI
            * math.pi * float(_cf_sim._input_dict["R_cath"]) ** 2
            * _cf_sm_alpha
            * float(_cf_sm.n[_cf_sm_cell])
            * float(
                _cf_flux.ion_sound_speed(
                    float(_cf_sm_der.Te[_cf_sm_cell]), _cf_sim.ion_mass_g
                )
            )
        )
        assert abs(_cf_circuit_I / _cf_sm_I - 1.0) <= 1.0e-14, (
            _cf_circuit_I, _cf_sm_I
        )
        _cf_spread = abs(_cf_circuit_I / _cf_step_I - 1.0)
        if _cf_first is None:
            _cf_first = _cf_spread
        _cf_worst = max(_cf_worst, _cf_spread)
        _cf_samples += 1
    assert _cf_samples >= 10, _cf_samples

    # (iii) POSITIVITY AND FINITENESS over the same run, with the wall-side
    # ratio and the Mach number at the cathode-adjacent cell reported.
    _cf_n_arr = np.asarray(_cf_sim.state.n, dtype=float)
    assert np.all(np.isfinite(_cf_n_arr)) and np.all(_cf_n_arr > 0.0)
    _cf_der = derive_state(
        _cf_sim.state, floors=_cf_sim.floors, ion_mass_g=_cf_sim.ion_mass_g
    )
    assert np.all(np.isfinite(np.asarray(_cf_der.Te, dtype=float)))
    _cf_c1 = int(absorbing_live_cells_by_role(_cf_sim.geometry)["cathode"][0])
    _cf_mach = float(_cf_der.u[_cf_c1]) / float(
        _cf_flux.ion_sound_speed(
            float(_cf_der.Te[_cf_c1]), _cf_sim.ion_mass_g
        )
    )
    print(
        "  cathode face: n1/n2 = "
        f"{_cf_n_arr[_cf_c1] / _cf_n_arr[_cf_c1 + 1]:.6f}, "
        f"Mach(cell {_cf_c1}) = {_cf_mach:.6f}; "
        f"raw-vs-smoothed |I_circuit/I_fluid - 1| over {_cf_samples} samples: "
        f"first {_cf_first:.3e}, worst {_cf_worst:.3e}"
    )


# ----------------------------------------------------------------------
# cathode-jet-incident-power-one-book
# ----------------------------------------------------------------------
@_case("cathode-jet-incident-power-one-book", historical_stance=True)
def _case_cathode_jet_incident_power_one_book():
    """The jet's incident power and the surface's ion credit are one book.

    The DVM cathode jet's incident-energy row is the circuit's own per-ion
    incident energy ``phi_c + Te/2`` on the fluid's delivered count, so the
    power the surface is debited (``R_E`` of that row) is a share of the very
    power ``P_cathode_i`` credits it with.
    """
    _cp_params, _cp_flags = _base_config()
    _cp_params = dict(_cp_params, max_steps_action="stop")
    _cp_flags = dict(_cp_flags)
    _cp_flags["cathode_coupling"] = True
    _cp_params["initial_neutral_state"] = "fill"
    _cp_sim = LAPDSim1D(dict(_cp_params), _cp_flags)
    _cp_cell = int(
        absorbing_live_cells_by_role(_cp_sim.geometry)["cathode"][0]
    )
    _cp_row = np.zeros(_cp_sim.geometry.cells, dtype=float)
    _cp_row[_cp_cell] = 1.0

    def _cp_read():
        """Return ``(result, Te, per_ion_erg)`` at the current state."""
        # The electrode sample re-seeded from the current state, so the
        # circuit and the row read ONE state and the only thing the
        # comparison can see is the per-ion ENERGY.
        _cp_sim._init_sample_smoothing()
        _cp_sim.rhs_terms()
        solve = _cp_sim._cathode_solve
        assert solve is not None and solve.beam_result is not None
        der = derive_state(
            _cp_sim.state, floors=_cp_sim.floors,
            ion_mass_g=_cp_sim.ion_mass_g,
        )
        per_ion = float(
            _cp_sim._dvm_cathode_jet_incident_energy_row(
                _cp_row, _cp_sim.state, solve
            )[_cp_cell]
        )
        return solve.beam_result.result, float(der.Te[_cp_cell]), per_ion

    # (a) AN INVERTED SHEATH accelerates no ion into the surface, so the row
    # clamps the fall at zero and carries the presheath half-Te alone. The
    # circuit's own P_cathode_i does NOT clamp -- it is a signed power, and
    # this is the one state at which the two per-ion numbers differ.
    _cp_res, _cp_Te, _cp_per_ion = _cp_read()
    assert _cp_res.phi_c < 0.0, _cp_res.phi_c
    assert abs(
        _cp_per_ion / (0.5 * _cp_Te * ev_to_erg) - 1.0
    ) <= 1.0e-14, _cp_per_ion

    # (b) AT AN ACCELERATING SHEATH the two are ONE number: the row's per-ion
    # incident energy is exactly P_cathode_i per collected ion.
    for _ in range(12):
        _cp_sim.run(t_end=None, dt=None, max_steps=4)
        _cp_res, _cp_Te, _cp_per_ion = _cp_read()
        if _cp_res.phi_c > 0.0:
            break
    assert _cp_res.phi_c > 0.0, _cp_res.phi_c
    _cp_circuit_eV = float(_cp_res.P_cathode_i) / float(_cp_res.I_i)
    assert abs(
        _cp_circuit_eV / (float(_cp_res.phi_c) + 0.5 * _cp_Te) - 1.0
    ) <= 1.0e-12, (_cp_circuit_eV, _cp_res.phi_c, _cp_Te)
    assert abs(
        _cp_per_ion / (_cp_circuit_eV * ev_to_erg) - 1.0
    ) <= 1.0e-12, (_cp_per_ion, _cp_circuit_eV)

    # THE SURFACE DEBIT IS R_E OF THAT BOOKED POWER, exactly. Armed, with a
    # cadence no step of this run reaches, the cathode ledger's backscatter
    # row and the accumulator the next tick has not yet been given are the
    # same booking read twice -- the created-once identity with the tick term
    # at zero.
    _cp_jp, _cp_jf = default_config()
    for _cp_space, _cp_key, _cp_value, _cp_why in (
        KINETIC_DVM_INCOMPATIBLE_DEFAULTS
    ):
        (_cp_jf if _cp_space == "flags" else _cp_jp)[_cp_key] = _cp_value
    _cp_jp["initial_neutral_state"] = "fill"
    _cp_jf["cathode_coupling"] = True
    _cp_jp.update({
        "neutral_model": "kinetic_dvm",
        # A cadence no step of this run reaches, so the accumulator the row
        # is compared against is never handed to a tick and reset.
        "neutral_kinetic_dvm_cadence_s": 1.0,
        # The shipped velocity grid, pinned wide enough to carry the launch
        # band this jet's coefficients can produce.
        "neutral_kinetic_dvm_vmax_cm_s": 3.0e7,
        "neutral_kinetic_dvm_cathode_jet": True,
        "max_steps_action": "stop",
    })
    _cp_jet = LAPDSim1D(_cp_jp, _cp_jf)
    _cp_jet.run(t_end=None, dt=1.0e-9, max_steps=8)
    _cp_R_E = float(_cp_jet._dvm_cathode_jet["R_E"])
    _cp_ledger = float(_cp_jet._cathode_energy_ledger_J["backscatter"])
    _cp_accum = float(np.sum(_cp_jet._dvm_cathode_jet_energy_booked))
    assert _cp_accum > 0.0, _cp_accum
    assert abs(
        _cp_ledger / (_cp_R_E * _cp_accum * 1.0e-7) - 1.0
    ) <= 1.0e-14, (_cp_ledger, _cp_accum, _cp_R_E)


# ----------------------------------------------------------------------
# anode-e-sheath-realised-equals-booked
# ----------------------------------------------------------------------
@_case("anode-e-sheath-realised-equals-booked")
def _case_anode_e_sheath_realised_equals_booked():
    sim, pair = _anode_sink_sim()
    # The rate is built from the SAME split weights and the SAME power the
    # reported row carries, so the row and the rate cannot disagree about
    # where the debit lands.
    nu, booked_W = sim.electrode_ee_sink_rate()
    row = np.asarray(sim.rhs_terms()["anode_e_sheath_loss"].Ee, dtype=float)
    volumes = np.asarray(sim.geometry.plasma_volume_cm3, dtype=float)
    assert np.all(nu >= 0.0)
    assert np.all(nu[[c for c in range(sim.geometry.cells) if c not in pair]]
                  == 0.0)
    assert np.all(nu[pair] > 0.0), nu[pair]
    # booked_W is the row, cell for cell: W vs erg cm^-3 s^-1.
    assert np.allclose(
        -row * volumes * 1.0e-7, booked_W, rtol=1.0e-13, atol=0.0
    )
    # nu is that power over the heat capacity at the circuit's own anode
    # sample temperature.
    Te_ref = np.asarray(
        sim.cathode_source_terms().metadata["anode_Te_ref_eV"], dtype=float
    )
    n_floor = np.maximum(np.asarray(sim.state.n, dtype=float), sim.floors["n"])
    expected = np.zeros_like(nu)
    expected[pair] = (
        -row[pair] / (1.5 * n_floor[pair] * Te_ref[pair] * ev_to_erg)
    )
    assert np.allclose(nu, expected, rtol=1.0e-13, atol=0.0)

    # One accepted step: the attempt's book is the sum of the substeps'
    # ledgers, and the realised debit differs from the circuit's booking by
    # exactly the local <Te_c/Te_a>.
    attempt = sim._attempt_step()
    booking = attempt.electrode_sink_booking
    assert booking is not None
    assert booking["window_s"] > 0.0
    assert booking["circuit_booked"] > 0.0
    assert booking["realised"] > 0.0
    ratio = booking["realised"] / booking["circuit_booked"]
    Te_local = np.asarray(sim.derived.Te, dtype=float)
    bracket = sorted(Te_local[pair] / Te_ref[pair])
    print(
        "  anode e-sheath realised/booked = %.4f; local Te/Te_a bracket "
        "[%.4f, %.4f]" % (ratio, bracket[0], bracket[1])
    )
    assert bracket[0] * 0.9 <= ratio <= bracket[1] * 1.1, (ratio, bracket)
    # The per-cell profile is the same energy, per cell.
    profile_J = (
        float(np.sum(booking["realised_profile"] * volumes)) * 1.0e-7
    )
    assert abs(profile_J / booking["realised"] - 1.0) < 1.0e-12, (
        profile_J, booking["realised"]
    )
    # ... and the saved scalars are the committed attempt, not a recompute.
    before = dict(sim._anode_e_sheath_ledger_J)
    step_dt = sim.suggest_timestep(include_heat_conduction=False).dt
    sim.advance_one_step(dt=step_dt)
    diag = sim._cathode_diagnostic_snapshot()
    step_booked = (
        sim._anode_e_sheath_ledger_J["circuit_booked"] - before["circuit_booked"]
    )
    step_realised = (
        sim._anode_e_sheath_ledger_J["realised"] - before["realised"]
    )
    assert step_booked > 0.0 and step_realised > 0.0
    assert diag["anode_e_sheath_booked_J"] == (
        sim._anode_e_sheath_ledger_J["circuit_booked"]
    )
    assert diag["anode_e_sheath_realised_J"] == (
        sim._anode_e_sheath_ledger_J["realised"]
    )
    # The step value is a difference of the cumulative ledger, so it is held
    # to a few ulps of that ledger -- the same bound the realised half below
    # is held to.
    booked_tol = 8.0 * np.finfo(float).eps * max(
        abs(sim._anode_e_sheath_ledger_J["circuit_booked"]),
        abs(before["circuit_booked"]),
    )
    assert abs(
        diag["anode_e_sheath_booked_W"] * step_dt - step_booked
    ) <= booked_tol, (
        diag["anode_e_sheath_booked_W"], step_dt, step_booked, booked_tol,
    )
    # The step value is a difference of the cumulative ledger, so it cannot
    # be resolved more finely than a few ulps of that ledger.
    realised_tol = 8.0 * np.finfo(float).eps * max(
        abs(sim._anode_e_sheath_ledger_J["realised"]),
        abs(before["realised"]),
    )
    assert abs(
        diag["anode_e_sheath_realised_W"] * step_dt - step_realised
    ) <= realised_tol, (
        diag["anode_e_sheath_realised_W"] * step_dt, step_realised,
        realised_tol,
    )


# ----------------------------------------------------------------------
# cathode-e-climb-realised-equals-booked
# ----------------------------------------------------------------------
@_case("cathode-e-climb-realised-equals-booked")
def _case_cathode_e_climb_realised_equals_booked():
    # The emitting cathode face's collected-electron climb rides the implicit
    # heat substep beside the anode's debit, with its own booked-against-
    # realised pair. The anode case above gates the anode pair; this gates
    # the climb's, on the same fixture.
    sim, pair = _anode_sink_sim()
    assert sim._cathode_climb_in_heat_substep
    nu, booked_W = sim.cathode_climb_ee_sink_rate()
    cells = [int(c) for c in np.flatnonzero(nu)]
    assert cells and not set(cells) & set(pair), (cells, pair)
    row = np.asarray(
        sim.rhs_terms()["cathode_e_collected_climb"].Ee, dtype=float
    )
    volumes = np.asarray(sim.geometry.plasma_volume_cm3, dtype=float)
    # booked_W is the reported row, cell for cell: W vs erg cm^-3 s^-1.
    assert np.allclose(
        -row * volumes * 1.0e-7, booked_W, rtol=1.0e-13, atol=0.0
    )
    # nu is that power over the cell's electron heat capacity at the state
    # the rate is formed at.
    Te0 = np.asarray(sim.derived.Te, dtype=float)
    n_floor = np.maximum(np.asarray(sim.state.n, dtype=float), sim.floors["n"])
    expected = np.zeros_like(nu)
    expected[cells] = -row[cells] / (1.5 * n_floor[cells] * Te0[cells] * ev_to_erg)
    assert np.allclose(nu, expected, rtol=1.0e-13, atol=0.0)

    # One attempt: the climb's share of the attempt's book, apart from the
    # anode's. The rate is referenced to the temperature each substep starts
    # at, so the realised debit is the booked one scaled by the substep's
    # mean Te/Te_start. Against the sink alone that mean is the exponential
    # decay's (1 - exp(-x))/x at x = nu*dt over the whole step (a Strang half
    # re-forms the rate at its own start, which only raises it) and at most
    # 1; the bracket is those two, widened by the anode case's 0.9/1.1. (The
    # end state cannot bracket it: the temperature floor acts on this cell.)
    ledger_before = dict(sim._cathode_e_climb_ledger_J)
    accum_before = sim._cathode_e_climb_realised_accum
    attempt = sim._attempt_step()
    booking = attempt.electrode_sink_booking
    assert booking is not None
    assert booking["cathode_circuit_booked"] > 0.0
    assert booking["cathode_realised"] > 0.0
    ratio = booking["cathode_realised"] / booking["cathode_circuit_booked"]
    x = float(np.max(nu[cells])) * float(attempt.dt)
    bracket = [(1.0 - np.exp(-x)) / x, 1.0]
    print(
        "  cathode climb realised/booked = %.4f; nu*dt = %.4f, decay "
        "bracket [%.4f, %.4f]" % (ratio, x, bracket[0], bracket[1])
    )
    assert bracket[0] * 0.9 <= ratio <= bracket[1] * 1.1, (ratio, bracket)
    assert booking["cathode_realised"] <= booking["cathode_circuit_booked"], (
        booking["cathode_realised"], booking["cathode_circuit_booked"]
    )
    # The per-cell profile is the same energy, on the climb's cells only; the
    # anode's profile carries nothing there.
    profile = np.asarray(booking["cathode_realised_profile"], dtype=float)
    assert set(np.flatnonzero(profile).tolist()) <= set(cells)
    assert np.all(np.asarray(booking["realised_profile"])[cells] == 0.0)
    profile_J = float(np.sum(profile * volumes)) * 1.0e-7
    assert abs(profile_J / booking["cathode_realised"] - 1.0) < 1.0e-12, (
        profile_J, booking["cathode_realised"]
    )
    # An attempt that is not accepted -- what a rejection is -- books
    # nothing: the ledger and the accumulator are untouched, and the
    # attempt-scoped book is dropped.
    assert sim._cathode_e_climb_ledger_J == ledger_before
    assert sim._cathode_e_climb_realised_accum is accum_before
    assert sim._anode_e_sheath_attempt is None

    # An accepted step commits the pair, and the saved scalars are the
    # committed book, not a recompute.
    accum_start = (
        np.zeros(sim.geometry.cells, dtype=float)
        if accum_before is None
        else np.array(accum_before, dtype=float)
    )
    step_dt = sim.suggest_timestep(include_heat_conduction=False).dt
    sim.advance_one_step(dt=step_dt)
    step_booked = (
        sim._cathode_e_climb_ledger_J["circuit_booked"]
        - ledger_before["circuit_booked"]
    )
    step_realised = (
        sim._cathode_e_climb_ledger_J["realised"] - ledger_before["realised"]
    )
    assert step_booked > 0.0 and step_realised > 0.0
    # The accumulator's growth over the step, times the cell volumes, is the
    # ledger's realised step.
    accum_J = float(
        np.sum((sim._cathode_e_climb_realised_accum - accum_start) * volumes)
    ) * 1.0e-7
    realised_tol = 8.0 * np.finfo(float).eps * max(
        abs(sim._cathode_e_climb_ledger_J["realised"]),
        abs(ledger_before["realised"]),
    )
    assert abs(accum_J - step_realised) <= realised_tol + 1.0e-12 * abs(
        step_realised
    ), (accum_J, step_realised)
    diag = sim._cathode_diagnostic_snapshot()
    assert diag["cathode_e_climb_booked_J"] == (
        sim._cathode_e_climb_ledger_J["circuit_booked"]
    )
    assert diag["cathode_e_climb_realised_J"] == (
        sim._cathode_e_climb_ledger_J["realised"]
    )

    # RESTART ROUND-TRIP: the pair travels in the payload and is restored
    # exactly; a payload written before the pair existed restores 0.0.
    payload = sim.restart_payload()
    for key in ("circuit_booked", "realised"):
        assert payload["ledgers"][f"cathode_e_climb_{key}_J"] == (
            sim._cathode_e_climb_ledger_J[key]
        )
    fresh = LAPDSim1D(*_anode_sink_config())
    fresh._apply_restart_payload(payload)
    assert fresh._cathode_e_climb_ledger_J == sim._cathode_e_climb_ledger_J
    old_payload = dict(payload)
    old_payload["ledgers"] = {
        k: v for k, v in payload["ledgers"].items()
        if not k.startswith("cathode_e_climb_")
    }
    older = LAPDSim1D(*_anode_sink_config())
    older._apply_restart_payload(old_payload)
    assert older._cathode_e_climb_ledger_J == {
        "circuit_booked": 0.0, "realised": 0.0,
    }


# ----------------------------------------------------------------------
# anode-cells-no-within-step-sawtooth
# ----------------------------------------------------------------------
@_case("anode-cells-no-within-step-sawtooth")
def _case_anode_cells_no_within_step_sawtooth():
    # The DEFECT this member exists to remove: operator A used to take the
    # whole anode electron debit out of the two flanking cells explicitly
    # and the following heat substep put it back, so every Te-dependent A
    # row in those cells ran at a dipped temperature. The A/B below is the
    # two compositions at ONE dt and ONE state -- the live one, with the row
    # carried as an implicit rate, against the historical one with the row
    # back in A -- so the difference IS the removed sawtooth.
    sim, pair = _anode_sink_sim()
    dt = sim.suggest_timestep(include_heat_conduction=False).dt
    y0 = sim._y.copy()
    nu, _booked_W = sim.electrode_ee_sink_rate()
    capacity = 1.5 * np.maximum(
        np.asarray(sim.state.n, dtype=float), sim.floors["n"]
    ) * ev_to_erg

    def compose(row_in_a):
        # ONE solver, so both compositions run on the same circuit state,
        # the same warm start and the same caches; the step caches are
        # snapshotted around each pass so neither leaves a trace in the
        # other, and nothing is ever accepted.
        cache = sim._step_cache_snapshot()
        live_terms = sim._heat_substep_terms
        live_gate = sim._electrode_sink_in_heat_substep
        if row_in_a:
            sim._heat_substep_terms = frozenset()
            sim._electrode_sink_in_heat_substep = False
        try:
            half = sim.implicit_heat_conduction_step(
                dt=0.5 * dt,
                y=y0,
                ee_sink_rate=None if row_in_a else nu,
            )
            y_half = sim.floor_state_vector(pack_state(half))
            y_after = ssprk2_step(
                y0=y_half,
                dt=dt,
                rhs_func=sim._explicit_stage_rhs(
                    dt, include_heat_conduction=False
                ),
                floor_func=sim.floor_state_vector,
                time=sim._time,
            )
        finally:
            sim._heat_substep_terms = live_terms
            sim._electrode_sink_in_heat_substep = live_gate
            sim._restore_step_cache(cache)
        Te_half = np.asarray(sim._unpack(y_half).Ee, dtype=float) / capacity
        Te_after = np.asarray(sim._unpack(y_after).Ee, dtype=float) / capacity
        return (Te_after - Te_half) / Te_half

    live = compose(row_in_a=False)
    legacy = compose(row_in_a=True)
    # What the EXPLICIT operator removes from these cells in one step: the
    # row's own power over the local electron store. It is the implicit
    # rate's nu*dt scaled by Te_a/Te_local, because the rate is referenced
    # to the circuit's anode sample and the row is a fixed power.
    row = np.asarray(sim.rhs_terms()["anode_e_sheath_loss"].Ee, dtype=float)
    Ee0 = np.asarray(sim.state.Ee, dtype=float)
    removed = dt * np.abs(row[pair]) / Ee0[pair]
    print(
        "  anode within-step A-stage Te change at cells %s: implicit %s, "
        "explicit-row %s (explicit removal %s, nu*dt %s)"
        % (pair, np.round(live[pair], 6), np.round(legacy[pair], 6),
           np.round(removed, 6), np.round(nu[pair] * dt, 6))
    )
    # The explicit composition dips FURTHER at both anode cells. The size
    # of the extra dip is printed above, not gated here: the implicit rate
    # nu itself is gated by anode-e-sheath-realised-equals-booked.
    assert np.all(legacy[pair] < live[pair]), (legacy[pair], live[pair])
    # The two compositions differ AT THE ANODE CELLS. Operator A's second
    # SSPRK2 stage sees the first stage's state, so a flux-coupled neighbour
    # picks up a fraction of it -- what must not happen is the difference
    # living anywhere but the two cells the row lands in.
    difference = np.abs(live - legacy)
    other = [c for c in range(sim.geometry.cells) if c not in pair]
    print(
        "  anode-cell difference %s vs worst elsewhere %.3e (cell %d)"
        % (np.round(difference[pair], 8), np.max(difference[other]),
           other[int(np.argmax(difference[other]))])
    )
    assert np.max(difference[other]) < 0.2 * np.min(difference[pair]), (
        np.max(difference[other]), difference[pair]
    )
    # And the live composition leaves the anode cells within 3 % across A.
    assert np.all(np.abs(live[pair]) < 0.03), live[pair]


# ----------------------------------------------------------------------
# anode-ion-collection-counted-vs-circuit
# ----------------------------------------------------------------------
@_case("anode-ion-collection-counted-vs-circuit")
def _case_anode_ion_collection_counted_vs_circuit():
    # The fluid-applied anode ion current against the circuit's own I_i_a.
    # The two are the SAME book, so the only thing separating them is how
    # the anode cells' state is sampled -- and an anode cell running at a
    # within-step dipped Te samples low, because the Bohm flux goes as
    # sqrt(Te). This is therefore the acceptance instrument for the
    # implicit member, read as an A/B of the two compositions at one
    # fixture and one state.
    #
    # The ABSOLUTE level is NOT this gate's business: at this fixture's
    # operating point (9 us into a scheduled discharge) the circuit's
    # supply-averaged EMA has not caught up with the raw state, and the
    # ratio sits near 0.71 on BOTH compositions. The level belongs to a
    # plateau read, not to a smoke case.
    def counted_over_circuit(legacy):
        sim, _pair = _anode_sink_sim()
        if legacy:
            # The historical composition: the row applied explicitly by A.
            sim._heat_substep_terms = frozenset()
            sim._electrode_sink_in_heat_substep = False
        volumes = np.asarray(sim.geometry.plasma_volume_cm3, dtype=float)
        counted = {"n": np.zeros(sim.geometry.cells, dtype=float), "w": 0.0}
        original_terms = sim.rhs_terms

        def counting_rhs_terms(*args, **kwargs):
            terms = original_terms(*args, **kwargs)
            if counted["w"] > 0.0:
                counted["n"] = counted["n"] + counted["w"] * np.asarray(
                    terms["anode_collection"].n, dtype=float
                )
            return terms

        sim.rhs_terms = counting_rhs_terms
        solved = []
        elapsed = 0.0
        for _ in range(6):
            dt = sim.suggest_timestep(include_heat_conduction=False).dt
            # SSPRK2 weights each stage's RHS by dt/2.
            counted["w"] = 0.5 * dt
            sim.advance_one_step(dt=dt)
            counted["w"] = 0.0
            solved.append(float(sim._cathode_solve.beam_result.result.I_i_a))
            elapsed += dt
        sim.rhs_terms = original_terms
        # The anode collection row is a particle SINK, so its integral is
        # negative; e times the rate is the current.
        counted_A = -float(np.sum(counted["n"] * volumes)) * qe_SI / elapsed
        circuit_A = float(np.mean(solved))
        return counted_A / circuit_A, counted_A, circuit_A

    implicit_ratio, implicit_A, implicit_circuit = counted_over_circuit(False)
    legacy_ratio, legacy_A, legacy_circuit = counted_over_circuit(True)
    print(
        "  anode counted/circuit ion current: implicit %.4f (%.3f / %.3f A), "
        "explicit-row %.4f (%.3f / %.3f A)"
        % (implicit_ratio, implicit_A, implicit_circuit,
           legacy_ratio, legacy_A, legacy_circuit)
    )
    # Taking the debit out of operator A moves the counted current TOWARD
    # the solve's -- the anode cells no longer sample a within-step dipped
    # Te, and the Bohm flux goes as sqrt(Te). The MARGIN is set by nu*dt,
    # which is ~1e-4 at this fixture's step, so the assertion is on the
    # direction and on the margin being above round-off; it is not a claim
    # about the level.
    assert implicit_ratio > legacy_ratio, (implicit_ratio, legacy_ratio)
    assert abs(1.0 - implicit_ratio) < abs(1.0 - legacy_ratio), (
        implicit_ratio, legacy_ratio
    )
    assert implicit_ratio - legacy_ratio > 1.0e-5, (
        implicit_ratio, legacy_ratio
    )


# ----------------------------------------------------------------------
# anode-e-sheath-row-reported-not-applied
# ----------------------------------------------------------------------
@_case("anode-e-sheath-row-reported-not-applied")
def _case_anode_e_sheath_row_reported_not_applied():
    sim, pair = _anode_sink_sim()
    assert sim._heat_substep_terms == frozenset(
        {"anode_e_sheath_loss", "beam_power_deposition"}
    )
    terms = sim.rhs_terms()
    assert "anode_e_sheath_loss" in terms
    row = np.asarray(terms["anode_e_sheath_loss"].Ee, dtype=float)
    assert np.all(np.abs(row[pair]) > 0.0)
    # The rows the implicit substep applies are withdrawn from rhs(): the
    # anode's (above) and, on this single-cathode layout with the circuit
    # solve running, the cathode face's collected-electron climb, which is
    # reported too.
    assert sim._cathode_climb_in_heat_substep
    assert "cathode_e_collected_climb" in terms
    withdrawn = sim._heat_substep_terms | {"cathode_e_collected_climb"}
    # rhs() is the sum of every OTHER row, bit for bit.
    expected = None
    for name, term in terms.items():
        if name in withdrawn:
            continue
        expected = term if expected is None else add_state_rhs(expected, term)
    assert sim.rhs().tobytes() == pack_state(expected).tobytes()
    # ... and the anode row, on its own, is withdrawn and reported but not
    # applied: adding it back to rhs() is the sum with it included.
    with_anode = None
    for name, term in terms.items():
        if name in withdrawn - {"anode_e_sheath_loss"}:
            continue
        with_anode = (
            term if with_anode is None else add_state_rhs(with_anode, term)
        )
    assert sim.rhs().tobytes() != pack_state(with_anode).tobytes()

    # With the split OFF the row is back in A, bit-exactly.
    off_params, off_flags = _anode_sink_config()
    off_flags["implicit_heat_conduction"] = False
    off = LAPDSim1D(off_params, off_flags)
    off._set_state_vector(sim._y.copy())
    off._time = sim._time
    assert off._heat_substep_terms == frozenset()
    off_terms = off.rhs_terms()
    off_expected = None
    for term in off_terms.values():
        off_expected = (
            term if off_expected is None else add_state_rhs(off_expected, term)
        )
    assert off.rhs().tobytes() == pack_state(off_expected).tobytes()
    assert np.all(
        np.abs(np.asarray(off_terms["anode_e_sheath_loss"].Ee)[pair]) > 0.0
    )
    # ... and a split-on run asked for a non-split step REFUSES rather than
    # dropping the debit on the floor.
    try:
        sim.advance_one_step(dt=1.0e-12, operator_split=False)
    except ValueError as error:
        assert "implicit_heat_conduction is on" in str(error), error
        assert "anode electron-sheath" in str(error), error
    else:
        raise AssertionError(
            "a non-split step on a split stance silently dropped the anode "
            "electron-sheath debit"
        )
