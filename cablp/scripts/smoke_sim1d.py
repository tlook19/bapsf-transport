import json
from io import StringIO
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import h5py
import numpy as np

from cablp.funcs._heat import elec_par_heat_div, ion_par_heat_div
from cablp.funcs._plasmaparams import c_log
from cablp.solvers._sim1d import (
    BreakdownError,
    LAPDSim1D,
    ProgressPrinter1D,
    SimulationProgress1D,
    TimestepRejectionError,
    default_config,
    load_result_hdf5,
    summarize_result,
)
from cablp.solvers._sim1d.physics.conduction import (
    heat_conduction_rhs,
    implicit_heat_conduction_step,
)
from cablp.solvers._sim1d.physics.cathode import beam_absorption_weights
from cablp.solvers._sim1d.physics.energy import (
    electron_cooling_rhs,
    electron_ion_exchange_rhs,
    ion_charge_exchange_rhs,
)
from cablp.solvers._sim1d.physics.flux import front_filling_fluxes
from cablp.solvers._sim1d.core.integrator import ssprk2_step
from cablp.solvers._sim1d.physics.neutrals import (
    neutral_exchange_coefficients,
    neutral_inventory_rate,
    puff_rate,
    pump_rate,
)
from cablp.solvers._sim1d.physics.reactions import (
    particle_inventory_rate,
    reaction_rates,
)
from cablp.solvers._sim1d.physics.sources import velocity_divergence
from cablp.solvers._sim1d.core.state import (
    STATE_NAMES_1D,
    conservative_from_primitives,
    derive_state,
    pack_state,
    unpack_state,
)
from cablp.vars._cons import I_Ry, en_factor, ev_to_erg, m_p_cgs, qe_SI


def main():
    params, flags = default_config()
    assert params["cycles"] == 1
    assert params["phase_transition_mode"] == "current"
    assert params["gas_puff_mode"] == "pulse_decay_to_level"
    assert params["tau_neutral_prebreakdown"] > 0.0
    assert flags["neutral_prebreakdown"]
    params["phase_transition_mode"] = "scheduled"
    params["gas_puff_mode"] = "decay_after_breakdown"
    flags["neutral_prebreakdown"] = False
    flags["cathode_coupling"] = False
    flags["implicit_heat_conduction"] = False
    sim = LAPDSim1D(params, flags)
    snapshot = sim.get_initial_snapshot()
    geom = snapshot.geometry
    state = snapshot.state
    derived = snapshot.derived

    assert geom.cells == params["nx"] + 2
    assert geom.length_cm.shape == (geom.cells,)
    assert geom.plasma_volume_cm3.shape == (geom.cells,)
    assert geom.neutral_volume_cm3.shape == (geom.cells,)
    assert geom.plasma_face_area_cm2.shape == (geom.cells + 1,)
    assert geom.neutral_face_area_cm2.shape == (geom.cells + 1,)
    assert geom.center_distance_cm.shape == (geom.cells - 1,)
    assert np.isclose(geom.z_edges_cm[0], 0.0)
    assert np.isclose(geom.z_edges_cm[-1], params["Lm"])
    assert geom.cell_role[0] == "source"
    assert geom.cell_role[-1] == "end"
    assert np.all(geom.cell_role[1:-1] == "domain")
    assert np.all(geom.plasma_volume_cm3 > 0.0)
    assert np.all(geom.neutral_volume_cm3 > geom.plasma_volume_cm3)

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

    constant_coeff = neutral_exchange_coefficients(
        geometry=geom,
        model="constant",
        constant_coeff_cm3_s=params["neutral_exchange_coeff_cm3_s"],
        Tn_K=params["Tn_K"],
        mu_neutral=4,
        clausing_scale=params["neutral_clausing_scale"],
    )
    assert constant_coeff.shape == (geom.cells - 1,)
    assert np.allclose(constant_coeff, params["neutral_exchange_coeff_cm3_s"])

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
        "front_density",
        "surface_loss",
        "neutral_exchange",
        "neutral_sources",
        "reactions",
        "energy_exchange",
        "electron_cooling",
        "ion_charge_exchange",
        "heat_conduction",
        "dt_max",
        "dt_min",
    }
    assert np.isfinite(dt_default.dt_neutral_sources)
    assert dt_default.dt_surface_loss > 0.0
    assert dt_default.dt_reactions > 0.0
    assert dt_default.dt_energy_exchange > 0.0
    assert dt_default.dt_electron_cooling > 0.0
    assert dt_default.dt_ion_charge_exchange > 0.0
    assert dt_default.dt_heat_conduction > 0.0

    cathode_boundary = sim.cathode_boundary_state()
    assert not cathode_boundary.enabled
    assert cathode_boundary.mode == "disabled"
    assert cathode_boundary.source.index == 0
    assert cathode_boundary.source.role == "source"
    assert cathode_boundary.end.index == geom.cells - 1
    assert cathode_boundary.end.role == "end"
    assert cathode_boundary.end_mode == params["end_mode"]
    assert cathode_boundary.twin_cathode == flags["TwinCathode"]
    for key in (
        "V_bank",
        "T_s",
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
    assert cathode_terms.mode == "disabled"
    assert cathode_terms.metadata["source_index"] == 0
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
    assert neutral_puff_source.nn[0] > neutral_off_source.nn[0]
    assert np.isclose(
        neutral_puff_source.nn[0] - neutral_off_source.nn[0],
        puff_rate(
            neutral_phase_params["S_gp"],
            neutral_phase_params["gas_puff_valves"],
            neutral_geom.neutral_volume_cm3[0],
        ),
    )
    assert sim.phase_switches_at_time(0.0) == {
        "cathode_enabled": False,
        "gas_puff_enabled": True,
        "floating": False,
    }
    assert sim.phase_switches_at_time(
        params["tau_prebreakdown"] + params["tau_discharge"]
    ) == {
        "cathode_enabled": False,
        "gas_puff_enabled": False,
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

    cathode_flags = dict(flags)
    cathode_flags["cathode_coupling"] = True
    cathode_sim = LAPDSim1D(params, cathode_flags)
    cathode_solve = cathode_sim.solve_cathode_boundary()
    assert cathode_solve.boundary.enabled
    assert cathode_solve.device_config is not None
    assert cathode_solve.beam_result is not None
    assert cathode_solve.metadata["enabled"] is True
    assert cathode_solve.metadata["floating"] is False
    assert cathode_solve.metadata["result_twin"] is None
    assert cathode_solve.device_config.Twin == cathode_flags["TwinCathode"]
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
    assert cathode_solve.beam_result.v_beam[0] > 0.0
    assert cathode_solve.beam_result.n_beam[0] > 0.0
    assert cathode_solve.beam_result.beam_cross[0] > 0.0
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
    inactive_afterglow_solve = cathode_sim.solve_cathode_boundary(
        time=afterglow_time,
        floating=False,
        update_cache=False,
    )
    assert not inactive_afterglow_solve.boundary.enabled
    post_afterglow_solve = cathode_sim.solve_cathode_boundary(
        time=afterglow_time + params["tau_afterglow"],
        update_cache=False,
    )
    assert not post_afterglow_solve.boundary.enabled
    cathode_loss_terms = cathode_sim.cathode_source_terms(
        cathode_solve=cathode_solve,
    )
    assert cathode_loss_terms.enabled
    assert cathode_loss_terms.mode == "disabled"
    assert cathode_loss_terms.rhs.n[0] < 0.0
    assert cathode_loss_terms.rhs.nn[0] > 0.0
    assert np.allclose(cathode_loss_terms.rhs.n[1:], 0.0)
    assert np.allclose(cathode_loss_terms.rhs.nn[1:], 0.0)
    assert np.allclose(cathode_loss_terms.rhs.M, 0.0)
    assert cathode_loss_terms.rhs.Ee[0] < 0.0
    assert cathode_loss_terms.rhs.Ei[0] < 0.0
    assert np.allclose(cathode_loss_terms.rhs.Ee[1:], 0.0)
    assert np.allclose(cathode_loss_terms.rhs.Ei[1:], 0.0)
    expected_cathode_loss = (
        (1.0 + 2.0 * params["eta"])
        * cathode_solve.beam_result.result.I_i
        / qe_SI
    )
    assert np.isclose(
        cathode_loss_terms.metadata["source_surface_particle_loss_s_inv"],
        expected_cathode_loss,
    )
    assert np.isclose(
        -cathode_loss_terms.rhs.n[0] * geom.plasma_volume_cm3[0],
        expected_cathode_loss,
    )
    assert np.isclose(
        cathode_loss_terms.rhs.nn[0] * geom.neutral_volume_cm3[0],
        expected_cathode_loss,
    )
    expected_electron_power_loss_W = (
        cathode_solve.beam_result.result.P_cathode_e
        + cathode_solve.beam_result.result.P_anode_e
    )
    assert np.isclose(
        cathode_loss_terms.metadata["source_electron_power_loss_W"],
        expected_electron_power_loss_W,
    )
    cathode_inventory_scale = np.sum(
        np.abs(cathode_loss_terms.rhs.n * geom.plasma_volume_cm3)
        + np.abs(cathode_loss_terms.rhs.nn * geom.neutral_volume_cm3)
    )
    assert np.isclose(
        particle_inventory_rate(cathode_loss_terms.rhs, geom),
        0.0,
        atol=1e-12 * cathode_inventory_scale,
    )
    assert np.allclose(
        cathode_loss_terms.rhs.Ee[0],
        -expected_electron_power_loss_W * 1.0e7 / geom.plasma_volume_cm3[0],
    )
    assert np.allclose(
        cathode_loss_terms.rhs.Ei[0],
        1.5 * ev_to_erg * params["Ti0"] * cathode_loss_terms.rhs.n[0],
    )
    afterglow_cathode_loss_terms = cathode_sim.cathode_source_terms(
        cathode_solve=floating_cathode_solve,
        time=afterglow_time,
    )
    assert not afterglow_cathode_loss_terms.enabled
    assert np.allclose(pack_state(afterglow_cathode_loss_terms.rhs), 0.0)
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
    }
    split_beam_sum = np.zeros_like(pack_state(beam_birth_terms))
    for split_term in split_beam_terms.values():
        split_beam_sum = split_beam_sum + pack_state(split_term)
    assert np.allclose(split_beam_sum, pack_state(beam_birth_terms))
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
    beam_inventory_scale = np.sum(
        np.abs(beam_birth_terms.n * geom.plasma_volume_cm3)
        + np.abs(beam_birth_terms.nn * geom.neutral_volume_cm3)
    )
    assert np.isclose(
        particle_inventory_rate(beam_birth_terms, geom),
        0.0,
        atol=1e-12 * beam_inventory_scale,
    )
    beam_weights = beam_absorption_weights(
        length_cm=geom.length_cm,
        l_b_profile=cathode_solve.beam_result.l_b_profile,
        cathode_index=0,
    )
    expected_beam_power_density = (
        beam_weights
        * (
            cathode_solve.beam_result.result.P_prim
            + cathode_solve.beam_result.result.P_ohmic
        )
        * 1.0e7
        / geom.plasma_volume_cm3
    )
    assert np.allclose(
        split_beam_terms["beam_power_deposition"].Ee,
        expected_beam_power_density,
    )
    assert np.allclose(
        split_beam_terms["beam_ionization_cost"].Ee,
        -sim.I_ion * ev_to_erg * beam_birth_terms.n,
    )
    assert np.allclose(
        beam_birth_terms.Ee,
        (
            split_beam_terms["beam_power_deposition"].Ee
            + split_beam_terms["beam_ionization_cost"].Ee
        ),
    )
    assert np.allclose(
        split_beam_terms["beam_ionization_birth"].Ei,
        1.5 * ev_to_erg * params["Ti_floor"] * beam_birth_terms.n,
    )
    afterglow_beam_terms = cathode_sim.beam_ionization_rhs_terms(
        cathode_solve=floating_cathode_solve,
        time=afterglow_time,
    )
    for term in afterglow_beam_terms.values():
        assert np.allclose(pack_state(term), 0.0)
    cathode_rhs_terms = cathode_sim.rhs_terms(include_heat_conduction=False)
    assert "cathode_surface_loss" in cathode_rhs_terms
    assert "beam_ionization_birth" in cathode_rhs_terms
    assert "beam_power_deposition" in cathode_rhs_terms
    assert "beam_ionization_cost" in cathode_rhs_terms
    assert cathode_rhs_terms["cathode_surface_loss"].n[0] < 0.0
    assert cathode_rhs_terms["cathode_surface_loss"].nn[0] > 0.0
    assert np.allclose(cathode_rhs_terms["cathode_surface_loss"].n[1:], 0.0)
    assert np.all(cathode_rhs_terms["beam_ionization_birth"].n >= 0.0)
    assert np.any(cathode_rhs_terms["beam_ionization_birth"].n > 0.0)
    assert np.all(cathode_rhs_terms["beam_power_deposition"].Ee >= 0.0)
    assert np.all(cathode_rhs_terms["beam_ionization_cost"].Ee <= 0.0)
    assert np.allclose(cathode_rhs_terms["surface_loss"].n[0], 0.0)
    assert np.allclose(cathode_rhs_terms["surface_loss"].nn[0], 0.0)
    assert cathode_rhs_terms["surface_loss"].n[-1] < 0.0
    assert cathode_rhs_terms["surface_loss"].nn[-1] > 0.0
    cathode_nonheat_rhs = cathode_sim.rhs(include_heat_conduction=False)
    cathode_term_sum = np.zeros_like(cathode_nonheat_rhs)
    for term in cathode_rhs_terms.values():
        cathode_term_sum = cathode_term_sum + pack_state(term)
    assert np.allclose(cathode_term_sum, cathode_nonheat_rhs)
    afterglow_rhs_terms = cathode_sim.rhs_terms(
        include_heat_conduction=False,
        time=afterglow_time,
    )
    assert np.allclose(
        pack_state(afterglow_rhs_terms["cathode_surface_loss"]),
        0.0,
    )
    assert np.allclose(
        pack_state(afterglow_rhs_terms["beam_ionization_birth"]),
        0.0,
    )
    assert np.allclose(
        pack_state(afterglow_rhs_terms["beam_power_deposition"]),
        0.0,
    )
    assert np.allclose(
        pack_state(afterglow_rhs_terms["beam_ionization_cost"]),
        0.0,
    )
    afterglow_nonheat_rhs = cathode_sim.rhs(
        include_heat_conduction=False,
        time=afterglow_time,
    )
    afterglow_term_sum = np.zeros_like(afterglow_nonheat_rhs)
    for term in afterglow_rhs_terms.values():
        afterglow_term_sum = afterglow_term_sum + pack_state(term)
    assert np.allclose(afterglow_term_sum, afterglow_nonheat_rhs)

    twin_cathode_flags = dict(cathode_flags)
    twin_cathode_flags["TwinCathode"] = True
    twin_cathode_sim = LAPDSim1D(params, twin_cathode_flags)
    twin_cathode_solve = twin_cathode_sim.solve_cathode_boundary()
    assert twin_cathode_solve.boundary.twin_cathode
    assert twin_cathode_solve.device_config.Twin
    assert twin_cathode_solve.beam_result.result_twin is not None
    assert twin_cathode_solve.metadata["result_twin"] is not None
    assert np.isfinite(twin_cathode_solve.x0_twin_next)
    assert twin_cathode_solve.beam_result.beam_cross[0] > 0.0
    assert twin_cathode_solve.beam_result.beam_cross[-1] > 0.0
    assert twin_cathode_solve.beam_result.n_beam[0] > 0.0
    assert twin_cathode_solve.beam_result.n_beam[-1] > 0.0
    twin_cathode_loss_terms = twin_cathode_sim.cathode_source_terms(
        cathode_solve=twin_cathode_solve,
    )
    assert twin_cathode_loss_terms.rhs.n[0] < 0.0
    assert twin_cathode_loss_terms.rhs.n[-1] < 0.0
    assert twin_cathode_loss_terms.rhs.nn[0] > 0.0
    assert twin_cathode_loss_terms.rhs.nn[-1] > 0.0
    assert twin_cathode_loss_terms.rhs.Ee[0] < 0.0
    assert twin_cathode_loss_terms.rhs.Ee[-1] < 0.0
    assert twin_cathode_loss_terms.rhs.Ei[0] < 0.0
    assert twin_cathode_loss_terms.rhs.Ei[-1] < 0.0
    expected_twin_source_loss = (
        (1.0 + 2.0 * params["eta"])
        * twin_cathode_solve.beam_result.result.I_i
        / qe_SI
    )
    expected_twin_end_loss = (
        (1.0 + 2.0 * params["eta"])
        * twin_cathode_solve.beam_result.result_twin.I_i
        / qe_SI
    )
    assert np.isclose(
        twin_cathode_loss_terms.metadata["source_surface_particle_loss_s_inv"],
        expected_twin_source_loss,
    )
    assert np.isclose(
        twin_cathode_loss_terms.metadata["end_surface_particle_loss_s_inv"],
        expected_twin_end_loss,
    )
    twin_inventory_scale = np.sum(
        np.abs(twin_cathode_loss_terms.rhs.n * geom.plasma_volume_cm3)
        + np.abs(twin_cathode_loss_terms.rhs.nn * geom.neutral_volume_cm3)
    )
    assert np.isclose(
        particle_inventory_rate(twin_cathode_loss_terms.rhs, geom),
        0.0,
        atol=1e-12 * twin_inventory_scale,
    )
    twin_beam_combined = twin_cathode_sim.beam_ionization_rhs(
        cathode_solve=twin_cathode_solve,
    )
    twin_beam_terms = twin_cathode_sim.beam_ionization_rhs_terms(
        cathode_solve=twin_cathode_solve,
    )
    twin_beam_sum = np.zeros_like(pack_state(twin_beam_combined))
    for split_term in twin_beam_terms.values():
        twin_beam_sum = twin_beam_sum + pack_state(split_term)
    assert np.allclose(twin_beam_sum, pack_state(twin_beam_combined))
    assert np.all(twin_beam_terms["beam_ionization_birth"].n >= 0.0)
    assert twin_beam_terms["beam_ionization_birth"].n[0] > 0.0
    assert twin_beam_terms["beam_ionization_birth"].n[-1] > 0.0
    assert np.all(twin_beam_terms["beam_power_deposition"].Ee >= 0.0)
    assert twin_beam_terms["beam_power_deposition"].Ee[0] > 0.0
    assert twin_beam_terms["beam_power_deposition"].Ee[-1] > 0.0
    assert np.all(twin_beam_terms["beam_ionization_cost"].Ee <= 0.0)
    assert twin_beam_terms["beam_ionization_cost"].Ee[0] < 0.0
    assert twin_beam_terms["beam_ionization_cost"].Ee[-1] < 0.0
    twin_source_weights = beam_absorption_weights(
        length_cm=geom.length_cm,
        l_b_profile=twin_cathode_solve.beam_result.l_b_profile,
        cathode_index=0,
    )
    twin_end_weights = beam_absorption_weights(
        length_cm=geom.length_cm,
        l_b_profile=twin_cathode_solve.beam_result.l_b_profile_twin,
        cathode_index=-1,
    )
    expected_twin_beam_power_density = (
        twin_source_weights
        * (
            twin_cathode_solve.beam_result.result.P_prim
            + twin_cathode_solve.beam_result.result.P_ohmic
        )
        + twin_end_weights
        * (
            twin_cathode_solve.beam_result.result_twin.P_prim
            + twin_cathode_solve.beam_result.result_twin.P_ohmic
        )
    ) * 1.0e7 / geom.plasma_volume_cm3
    assert np.allclose(
        twin_beam_terms["beam_power_deposition"].Ee,
        expected_twin_beam_power_density,
    )
    assert np.allclose(
        twin_beam_terms["beam_ionization_cost"].Ee,
        -sim.I_ion * ev_to_erg * twin_beam_terms["beam_ionization_birth"].n,
    )

    rhs = sim.plasma_flux_rhs(include_front=False)
    for values in (rhs.n, rhs.nn, rhs.M, rhs.Ee, rhs.Ei):
        assert np.allclose(values, 0.0, atol=1e-20)
    pressure_rhs = sim.pressure_work_rhs()
    for values in (
        pressure_rhs.n,
        pressure_rhs.nn,
        pressure_rhs.M,
        pressure_rhs.Ee,
        pressure_rhs.Ei,
    ):
        assert np.allclose(values, 0.0, atol=1e-20)
    neutral_rhs = sim.neutral_exchange_rhs()
    for values in (
        neutral_rhs.n,
        neutral_rhs.nn,
        neutral_rhs.M,
        neutral_rhs.Ee,
        neutral_rhs.Ei,
    ):
        assert np.allclose(values, 0.0, atol=1e-20)
    source_rhs = sim.neutral_source_sink_rhs()
    assert source_rhs.nn[0] > 0.0
    assert source_rhs.nn[-1] < 0.0
    assert np.isclose(
        source_rhs.nn[0],
        puff_rate(params["S_gp"], params["gas_puff_valves"], geom.neutral_volume_cm3[0])
        - pump_rate(params["S_pump_L"], geom.neutral_volume_cm3[0]) * state.nn[0],
    )
    assert np.isclose(
        source_rhs.nn[-1],
        -pump_rate(params["S_pump_R"], geom.neutral_volume_cm3[-1]) * state.nn[-1],
    )
    afterglow_source = sim.neutral_source_sink_rhs(
        time=params["tau_prebreakdown"] + params["tau_discharge"]
    )
    assert np.isclose(
        afterglow_source.nn[0],
        -pump_rate(params["S_pump_L"], geom.neutral_volume_cm3[0]) * state.nn[0],
    )
    assert np.isclose(
        source_rhs.nn[0] - afterglow_source.nn[0],
        puff_rate(params["S_gp"], params["gas_puff_valves"], geom.neutral_volume_cm3[0]),
    )
    assert np.isclose(afterglow_source.nn[-1], source_rhs.nn[-1])
    afterglow_source_terms = sim.rhs_terms(
        include_heat_conduction=False,
        time=params["tau_prebreakdown"] + params["tau_discharge"],
    )
    afterglow_dt_diag = sim.suggest_timestep(
        time=params["tau_prebreakdown"] + params["tau_discharge"]
    )
    assert np.isclose(
        afterglow_dt_diag.time,
        params["tau_prebreakdown"] + params["tau_discharge"],
    )
    assert afterglow_dt_diag.phase == "afterglow"
    assert afterglow_dt_diag.phase_cathode_enabled == 0.0
    assert afterglow_dt_diag.phase_gas_puff_enabled == 0.0
    assert afterglow_dt_diag.phase_floating == 1.0
    assert np.allclose(
        afterglow_source_terms["neutral_sources"].nn,
        afterglow_source.nn,
    )
    assert np.allclose(afterglow_source_terms["neutral_sources"].n, 0.0)
    assert (
        afterglow_dt_diag.dt_neutral_sources >= sim.suggest_timestep().dt_neutral_sources
    )
    decay_params = dict(params)
    decay_params["gas_puff_mode"] = "decay_after_breakdown"
    decay_params["pump_enabled"] = False
    decay_params["tau_prebreakdown"] = 1.0e-10
    decay_params["tau_discharge"] = 4.0e-10
    decay_params["tau_afterglow"] = 1.0e-10
    decay_params["tau_gp_after_breakdown"] = 1.0e-10
    decay_params["tau_gp_decay_factor"] = 1.0
    decay_sim = LAPDSim1D(decay_params, flags)
    decay_geom = decay_sim.get_initial_snapshot().geometry
    decay_main_start = decay_params["tau_prebreakdown"]
    decay_event = decay_main_start + decay_params["tau_gp_after_breakdown"]
    assert np.isclose(
        decay_sim.next_phase_boundary_after(decay_main_start),
        decay_event,
    )
    decay_on = decay_sim.neutral_source_sink_rhs(time=decay_event)
    assert np.isclose(
        decay_on.nn[0],
        puff_rate(
            decay_params["S_gp"],
            decay_params["gas_puff_valves"],
            decay_geom.neutral_volume_cm3[0],
        ),
    )
    decay_time = decay_main_start + 2.0e-10
    decay_tau = (
        decay_params["tau_discharge"] - decay_params["tau_gp_after_breakdown"]
    ) * decay_params["tau_gp_decay_factor"]
    decay_factor = np.exp(-(decay_time - decay_event) / decay_tau)
    decay_rhs = decay_sim.neutral_source_sink_rhs(time=decay_time)
    assert np.isclose(
        decay_rhs.nn[0],
        puff_rate(
            decay_params["S_gp"] * decay_factor,
            decay_params["gas_puff_valves"],
            decay_geom.neutral_volume_cm3[0],
        ),
    )

    pulse_params = dict(decay_params)
    pulse_params["gas_puff_mode"] = "pulse_decay_to_level"
    pulse_params["S_gp_decay_target"] = 1000.0
    pulse_params["tau_gp_pulse_duration"] = 1.0e-10
    pulse_params["tau_gp_decay_duration"] = 2.0e-10
    pulse_sim = LAPDSim1D(pulse_params, flags)
    pulse_geom = pulse_sim.get_initial_snapshot().geometry
    pulse_event = pulse_params["tau_prebreakdown"] + pulse_params["tau_gp_pulse_duration"]
    assert np.isclose(
        pulse_sim.next_phase_boundary_after(pulse_params["tau_prebreakdown"]),
        pulse_event,
    )
    pulse_on = pulse_sim.neutral_source_sink_rhs(time=pulse_event)
    assert np.isclose(
        pulse_on.nn[0],
        puff_rate(
            pulse_params["S_gp"],
            pulse_params["gas_puff_valves"],
            pulse_geom.neutral_volume_cm3[0],
        ),
    )
    pulse_time = pulse_event + 1.0e-10
    pulse_decay = np.exp(-(pulse_time - pulse_event) / pulse_params["tau_gp_decay_duration"])
    pulse_s_gp = (
        pulse_params["S_gp_decay_target"]
        + (pulse_params["S_gp"] - pulse_params["S_gp_decay_target"]) * pulse_decay
    )
    pulse_rhs = pulse_sim.neutral_source_sink_rhs(time=pulse_time)
    assert np.isclose(
        pulse_rhs.nn[0],
        puff_rate(
            pulse_s_gp,
            pulse_params["gas_puff_valves"],
            pulse_geom.neutral_volume_cm3[0],
        ),
    )
    assert np.allclose(source_rhs.n, 0.0)
    assert np.allclose(source_rhs.M, 0.0)
    assert np.allclose(source_rhs.Ee, 0.0)
    assert np.allclose(source_rhs.Ei, 0.0)

    disabled_params = dict(params)
    disabled_params["gas_puff_enabled"] = False
    disabled_params["pump_enabled"] = False
    disabled_sim = LAPDSim1D(disabled_params, flags)
    disabled_source = disabled_sim.neutral_source_sink_rhs()
    for values in (
        disabled_source.n,
        disabled_source.nn,
        disabled_source.M,
        disabled_source.Ee,
        disabled_source.Ei,
    ):
        assert np.allclose(values, 0.0, atol=1e-20)

    surface_state = conservative_from_primitives(
        n=np.full(geom.cells, params["ne0"]),
        nn=state.nn,
        u=np.full(geom.cells, 1.0e5),
        Te=np.full(geom.cells, 1.0),
        Ti=np.full(geom.cells, 0.5),
        ion_mass_g=sim.ion_mass_g,
    )
    surface_rhs = sim.surface_neutralization_rhs(state=surface_state)
    active_surface = np.zeros(geom.cells, dtype=bool)
    active_surface[[0, -1]] = True
    assert np.all(surface_rhs.n[active_surface] < 0.0)
    assert np.all(surface_rhs.nn[active_surface] > 0.0)
    assert np.all(surface_rhs.M[active_surface] < 0.0)
    assert np.all(surface_rhs.Ee[active_surface] < 0.0)
    assert np.all(surface_rhs.Ei[active_surface] < 0.0)
    assert np.allclose(surface_rhs.n[1:-1], 0.0)
    assert np.allclose(surface_rhs.nn[1:-1], 0.0)
    assert np.allclose(surface_rhs.M[1:-1], 0.0)
    assert np.allclose(surface_rhs.Ee[1:-1], 0.0)
    assert np.allclose(surface_rhs.Ei[1:-1], 0.0)
    surface_inventory_scale = np.sum(
        np.abs(surface_rhs.n * geom.plasma_volume_cm3)
        + np.abs(surface_rhs.nn * geom.neutral_volume_cm3)
    )
    assert np.isclose(
        particle_inventory_rate(surface_rhs, geom),
        0.0,
        atol=1e-12 * surface_inventory_scale,
    )
    assert np.allclose(
        surface_rhs.Ee[active_surface],
        1.5 * ev_to_erg * surface_rhs.n[active_surface],
    )
    assert np.allclose(
        surface_rhs.Ei[active_surface],
        1.5 * 0.5 * ev_to_erg * surface_rhs.n[active_surface],
    )
    surface_dt = sim.suggest_timestep(y=pack_state(surface_state))
    assert np.isfinite(surface_dt.dt_surface_loss)

    disabled_surface_params = dict(params)
    disabled_surface_params["b_surface_loss"] = 0.0
    disabled_surface_sim = LAPDSim1D(disabled_surface_params, flags)
    disabled_surface_rhs = disabled_surface_sim.surface_neutralization_rhs(
        state=surface_state
    )
    for values in (
        disabled_surface_rhs.n,
        disabled_surface_rhs.nn,
        disabled_surface_rhs.M,
        disabled_surface_rhs.Ee,
        disabled_surface_rhs.Ei,
    ):
        assert np.allclose(values, 0.0)

    reaction_rhs = sim.reaction_rhs()
    for values in (
        reaction_rhs.n,
        reaction_rhs.nn,
        reaction_rhs.M,
        reaction_rhs.Ee,
        reaction_rhs.Ei,
    ):
        assert np.all(np.isfinite(values))
    reaction_inventory_scale = np.sum(
        np.abs(reaction_rhs.n * geom.plasma_volume_cm3)
        + np.abs(reaction_rhs.nn * geom.neutral_volume_cm3)
    )
    reaction_inventory_tol = 1e-12 * reaction_inventory_scale
    assert np.isclose(
        particle_inventory_rate(reaction_rhs, geom),
        0.0,
        atol=reaction_inventory_tol,
    )
    reaction_terms = sim.reaction_rhs_terms()
    assert set(reaction_terms) == {
        "ionization_birth",
        "recombination_rad_loss",
        "recombination_3b_loss",
    }
    reaction_term_sum = np.zeros_like(pack_state(reaction_rhs))
    for term in reaction_terms.values():
        for field_name in STATE_NAMES_1D:
            assert np.all(np.isfinite(getattr(term, field_name)))
        assert np.isclose(
            particle_inventory_rate(term, geom),
            0.0,
            atol=reaction_inventory_tol,
        )
        reaction_term_sum = reaction_term_sum + pack_state(term)
    assert np.allclose(reaction_term_sum, pack_state(reaction_rhs))
    S_ion_off, _, _ = reaction_rates(
        state=state,
        floors=sim.floors,
        ion_mass_g=sim.ion_mass_g,
        gas_type=params["gas_type"],
        I_ion=sim.I_ion,
        b_ioniz=0.0,
        b_rec_rad=params["b_rec_rad"],
        b_rec_3b=params["b_rec_3b"],
    )
    assert np.allclose(S_ion_off, 0.0)

    recomb_state = conservative_from_primitives(
        n=np.full(geom.cells, params["ne0"]),
        nn=state.nn,
        u=np.full(geom.cells, 1.0e5),
        Te=np.full(geom.cells, 1.0),
        Ti=np.full(geom.cells, 1.0),
        ion_mass_g=sim.ion_mass_g,
    )
    recomb_params = dict(params)
    recomb_params["b_ioniz"] = 0.0
    recomb_sim = LAPDSim1D(recomb_params, flags)
    recomb_rhs = recomb_sim.reaction_rhs(state=recomb_state)
    assert np.all(recomb_rhs.M < 0.0)
    assert np.all(recomb_rhs.Ee < 0.0)
    assert np.all(recomb_rhs.Ei < 0.0)

    density_ramp = np.linspace(2.0, 1.0, geom.cells) * params["ne0"]
    ramp_state = conservative_from_primitives(
        n=density_ramp,
        nn=state.nn,
        u=np.zeros(geom.cells),
        Te=np.full(geom.cells, params["Te0"]),
        Ti=np.full(geom.cells, params["Ti0"]),
        ion_mass_g=sim.ion_mass_g,
    )
    ramp_front = front_filling_fluxes(
        state=ramp_state,
        floors=sim.floors,
        ion_mass_g=sim.ion_mass_g,
        mu=sim.mu,
        geometry=geom,
        alpha_front=params["alpha_front"],
    )
    assert np.all(ramp_front.n[1:-1] > 0.0)
    ramp_rhs = sim.plasma_flux_rhs(y=pack_state(ramp_state), include_front=True)
    for values in (ramp_rhs.n, ramp_rhs.nn, ramp_rhs.M, ramp_rhs.Ee, ramp_rhs.Ei):
        assert np.all(np.isfinite(values))
    ramp_flux_terms = sim.plasma_flux_rhs_terms(
        state=ramp_state,
        include_front=True,
    )
    assert set(ramp_flux_terms) == {"plasma_advective_flux", "plasma_front_flux"}
    ramp_flux_sum = np.zeros_like(pack_state(ramp_rhs))
    for term in ramp_flux_terms.values():
        for field_name in STATE_NAMES_1D:
            assert np.all(np.isfinite(getattr(term, field_name)))
        ramp_flux_sum = ramp_flux_sum + pack_state(term)
    assert np.allclose(ramp_flux_sum, pack_state(ramp_rhs))
    no_front_terms = sim.plasma_flux_rhs_terms(
        state=ramp_state,
        include_front=False,
    )
    assert np.allclose(pack_state(no_front_terms["plasma_front_flux"]), 0.0)

    nn_ramp_state = conservative_from_primitives(
        n=state.n,
        nn=np.linspace(2.0, 1.0, geom.cells) * state.nn[0],
        u=np.zeros(geom.cells),
        Te=np.full(geom.cells, params["Te0"]),
        Ti=np.full(geom.cells, params["Ti0"]),
        ion_mass_g=sim.ion_mass_g,
    )
    nn_ramp_rhs = sim.neutral_exchange_rhs(state=nn_ramp_state)
    assert nn_ramp_rhs.nn[0] < 0.0
    assert nn_ramp_rhs.nn[-1] > 0.0
    inventory_terms = nn_ramp_rhs.nn * geom.neutral_volume_cm3
    inventory_tol = 1e-12 * np.sum(np.abs(inventory_terms))
    assert np.isclose(
        neutral_inventory_rate(nn_ramp_rhs, geom), 0.0, atol=inventory_tol
    )
    assert np.allclose(nn_ramp_rhs.n, 0.0)
    assert np.allclose(nn_ramp_rhs.M, 0.0)
    assert np.allclose(nn_ramp_rhs.Ee, 0.0)
    assert np.allclose(nn_ramp_rhs.Ei, 0.0)

    nn_ramp_dt = sim.suggest_timestep(y=pack_state(nn_ramp_state))
    assert np.isfinite(nn_ramp_dt.dt_neutral_exchange)
    assert nn_ramp_dt.dt_neutral_exchange < dt_default.dt_neutral_exchange

    expanding_state = conservative_from_primitives(
        n=np.full(geom.cells, params["ne0"]),
        nn=state.nn,
        u=np.linspace(-1.0e4, 1.0e4, geom.cells),
        Te=np.full(geom.cells, params["Te0"]),
        Ti=np.full(geom.cells, params["Ti0"]),
        ion_mass_g=sim.ion_mass_g,
    )
    expanding_div_u = velocity_divergence(
        expanding_state, sim.floors, sim.ion_mass_g, geom
    )
    expanding_pressure = sim.pressure_work_rhs(state=expanding_state)
    assert np.all(expanding_div_u[2:-2] > 0.0)
    assert np.all(expanding_pressure.Ee[2:-2] < 0.0)
    assert np.all(expanding_pressure.Ei[2:-2] < 0.0)

    compressing_state = conservative_from_primitives(
        n=np.full(geom.cells, params["ne0"]),
        nn=state.nn,
        u=np.linspace(1.0e4, -1.0e4, geom.cells),
        Te=np.full(geom.cells, params["Te0"]),
        Ti=np.full(geom.cells, params["Ti0"]),
        ion_mass_g=sim.ion_mass_g,
    )
    compressing_div_u = velocity_divergence(
        compressing_state, sim.floors, sim.ion_mass_g, geom
    )
    compressing_pressure = sim.pressure_work_rhs(state=compressing_state)
    assert np.all(compressing_div_u[2:-2] < 0.0)
    assert np.all(compressing_pressure.Ee[2:-2] > 0.0)
    assert np.all(compressing_pressure.Ei[2:-2] > 0.0)

    hot_e_state = conservative_from_primitives(
        n=np.full(geom.cells, params["ne0"]),
        nn=state.nn,
        u=np.zeros(geom.cells),
        Te=np.full(geom.cells, 2.0),
        Ti=np.full(geom.cells, 0.5),
        ion_mass_g=sim.ion_mass_g,
    )
    hot_e_exchange = sim.energy_exchange_rhs(state=hot_e_state)
    assert np.all(hot_e_exchange.Ee < 0.0)
    assert np.all(hot_e_exchange.Ei > 0.0)
    assert np.allclose(hot_e_exchange.Ee + hot_e_exchange.Ei, 0.0)
    hot_e_dt = sim.suggest_timestep(y=pack_state(hot_e_state))
    assert np.isfinite(hot_e_dt.dt_energy_exchange)

    hot_i_state = conservative_from_primitives(
        n=np.full(geom.cells, params["ne0"]),
        nn=state.nn,
        u=np.zeros(geom.cells),
        Te=np.full(geom.cells, 0.5),
        Ti=np.full(geom.cells, 2.0),
        ion_mass_g=sim.ion_mass_g,
    )
    hot_i_exchange = sim.energy_exchange_rhs(state=hot_i_state)
    assert np.all(hot_i_exchange.Ee > 0.0)
    assert np.all(hot_i_exchange.Ei < 0.0)
    assert np.allclose(hot_i_exchange.Ee + hot_i_exchange.Ei, 0.0)

    equal_temp_exchange = sim.energy_exchange_rhs()
    assert np.allclose(equal_temp_exchange.Ee, 0.0, atol=1e-30)
    assert np.allclose(equal_temp_exchange.Ei, 0.0, atol=1e-30)
    disabled_exchange = electron_ion_exchange_rhs(
        state=hot_e_state,
        floors=sim.floors,
        ion_mass_g=sim.ion_mass_g,
        mu=sim.mu,
        b_Qie=0.0,
        ln_lambda_min=params["ln_lambda_min"],
    )
    assert np.allclose(disabled_exchange.Ee, 0.0)
    assert np.allclose(disabled_exchange.Ei, 0.0)

    cooling_state = conservative_from_primitives(
        n=np.full(geom.cells, 1.0e12),
        nn=np.full(geom.cells, 1.0e12),
        u=np.zeros(geom.cells),
        Te=np.full(geom.cells, 10.0),
        Ti=np.full(geom.cells, 1.0),
        ion_mass_g=sim.ion_mass_g,
    )
    cooling_rhs = sim.electron_cooling_rhs(state=cooling_state)
    cooling_terms = sim.electron_cooling_rhs_terms(state=cooling_state)
    assert set(cooling_terms) == {
        "ionization_energy_cost",
        "electron_ion_cooling",
        "electron_neutral_cooling",
    }
    cooling_term_sum = np.zeros_like(pack_state(cooling_rhs))
    for term in cooling_terms.values():
        cooling_term_sum = cooling_term_sum + pack_state(term)
    assert np.allclose(cooling_term_sum, pack_state(cooling_rhs))
    assert np.any(cooling_terms["ionization_energy_cost"].Ee < 0.0)
    assert np.any(cooling_terms["electron_ion_cooling"].Ee < 0.0)
    assert np.any(cooling_terms["electron_neutral_cooling"].Ee < 0.0)
    assert np.all(cooling_rhs.Ee < 0.0)
    assert np.allclose(cooling_rhs.n, 0.0)
    assert np.allclose(cooling_rhs.nn, 0.0)
    assert np.allclose(cooling_rhs.M, 0.0)
    assert np.allclose(cooling_rhs.Ei, 0.0)
    cooling_dt = sim.suggest_timestep(y=pack_state(cooling_state))
    assert np.isfinite(cooling_dt.dt_electron_cooling)

    ionization_only_cooling = electron_cooling_rhs(
        state=cooling_state,
        floors=sim.floors,
        ion_mass_g=sim.ion_mass_g,
        gas_type=params["gas_type"],
        I_ion=sim.I_ion,
        b_ioniz=params["b_ioniz"],
        b_rec_rad=params["b_rec_rad"],
        b_rec_3b=params["b_rec_3b"],
        b_ionization_energy_cost=params["b_ionization_energy_cost"],
        b_Qei=0.0,
        b_Qen=0.0,
        ionization_energy_cost=True,
        icool=True,
        ncool=True,
        icool_recomb=flags["icool_recomb"],
    )
    assert np.all(ionization_only_cooling.Ee < 0.0)

    disabled_cooling = electron_cooling_rhs(
        state=cooling_state,
        floors=sim.floors,
        ion_mass_g=sim.ion_mass_g,
        gas_type=params["gas_type"],
        I_ion=sim.I_ion,
        b_ioniz=params["b_ioniz"],
        b_rec_rad=params["b_rec_rad"],
        b_rec_3b=params["b_rec_3b"],
        b_ionization_energy_cost=0.0,
        b_Qei=0.0,
        b_Qen=0.0,
        ionization_energy_cost=True,
        icool=True,
        ncool=True,
        icool_recomb=flags["icool_recomb"],
    )
    assert np.allclose(disabled_cooling.Ee, 0.0)
    assert np.allclose(disabled_cooling.Ei, 0.0)

    hot_ion_cx_state = conservative_from_primitives(
        n=np.full(geom.cells, 1.0e12),
        nn=np.full(geom.cells, 1.0e12),
        u=np.zeros(geom.cells),
        Te=np.full(geom.cells, 1.0),
        Ti=np.full(geom.cells, 10.0),
        ion_mass_g=sim.ion_mass_g,
    )
    hot_ion_cx = sim.ion_charge_exchange_rhs(state=hot_ion_cx_state)
    assert np.all(hot_ion_cx.Ei < 0.0)
    assert np.allclose(hot_ion_cx.n, 0.0)
    assert np.allclose(hot_ion_cx.nn, 0.0)
    assert np.allclose(hot_ion_cx.M, 0.0)
    assert np.allclose(hot_ion_cx.Ee, 0.0)
    hot_ion_cx_dt = sim.suggest_timestep(y=pack_state(hot_ion_cx_state))
    assert np.isfinite(hot_ion_cx_dt.dt_ion_charge_exchange)

    warm_neutral_cx = ion_charge_exchange_rhs(
        state=hot_ion_cx_state,
        floors=sim.floors,
        ion_mass_g=sim.ion_mass_g,
        gas_type=params["gas_type"],
        Tn_fit=20.0,
        b_Qcx=params["b_Qcx"],
        cx=True,
    )
    assert np.all(warm_neutral_cx.Ei > 0.0)

    disabled_cx = ion_charge_exchange_rhs(
        state=hot_ion_cx_state,
        floors=sim.floors,
        ion_mass_g=sim.ion_mass_g,
        gas_type=params["gas_type"],
        Tn_fit=params["Tn_fit"],
        b_Qcx=0.0,
        cx=True,
    )
    assert np.allclose(disabled_cx.Ei, 0.0)

    hydrogen_params = dict(params)
    hydrogen_params["gas_type"] = "H"
    hydrogen_params["dt_save"] = 0.0
    hydrogen_sim = LAPDSim1D(hydrogen_params, flags)
    assert np.isclose(hydrogen_sim.ion_mass_g, m_p_cgs)
    assert hydrogen_sim.mu == 1
    assert hydrogen_sim._mu_neutral == 2
    assert np.isclose(hydrogen_sim.I_ion, I_Ry)

    hydrogen_state = conservative_from_primitives(
        n=np.full(geom.cells, 1.0e12),
        nn=np.full(geom.cells, 1.0e12),
        u=np.zeros(geom.cells),
        Te=np.full(geom.cells, 10.0),
        Ti=np.full(geom.cells, 10.0),
        ion_mass_g=hydrogen_sim.ion_mass_g,
    )
    hydrogen_rates = reaction_rates(
        state=hydrogen_state,
        floors=hydrogen_sim.floors,
        ion_mass_g=hydrogen_sim.ion_mass_g,
        gas_type="H",
        I_ion=hydrogen_sim.I_ion,
        b_ioniz=hydrogen_params["b_ioniz"],
        b_rec_rad=hydrogen_params["b_rec_rad"],
        b_rec_3b=hydrogen_params["b_rec_3b"],
    )
    for rate in hydrogen_rates:
        assert np.all(np.isfinite(rate))
        assert np.all(rate > 0.0)
    hydrogen_reaction = hydrogen_sim.reaction_rhs(state=hydrogen_state)
    for values in (
        hydrogen_reaction.n,
        hydrogen_reaction.nn,
        hydrogen_reaction.M,
        hydrogen_reaction.Ee,
        hydrogen_reaction.Ei,
    ):
        assert np.all(np.isfinite(values))
    hydrogen_inventory_scale = np.sum(
        np.abs(hydrogen_reaction.n * geom.plasma_volume_cm3)
        + np.abs(hydrogen_reaction.nn * geom.neutral_volume_cm3)
    )
    assert np.isclose(
        particle_inventory_rate(hydrogen_reaction, geom),
        0.0,
        atol=1e-12 * hydrogen_inventory_scale,
    )
    hydrogen_cooling_terms = hydrogen_sim.electron_cooling_rhs_terms(
        state=hydrogen_state,
    )
    assert np.any(hydrogen_cooling_terms["ionization_energy_cost"].Ee < 0.0)
    assert np.any(hydrogen_cooling_terms["electron_ion_cooling"].Ee < 0.0)
    assert np.any(hydrogen_cooling_terms["electron_neutral_cooling"].Ee < 0.0)
    hydrogen_cx = hydrogen_sim.ion_charge_exchange_rhs(state=hydrogen_state)
    assert np.all(np.isfinite(hydrogen_cx.Ei))
    assert np.all(hydrogen_cx.Ei < 0.0)
    hydrogen_warm_neutral_cx = ion_charge_exchange_rhs(
        state=hydrogen_state,
        floors=hydrogen_sim.floors,
        ion_mass_g=hydrogen_sim.ion_mass_g,
        gas_type="H",
        Tn_fit=20.0,
        b_Qcx=hydrogen_params["b_Qcx"],
        cx=True,
    )
    assert np.all(hydrogen_warm_neutral_cx.Ei > 0.0)
    hydrogen_dt = hydrogen_sim.suggest_timestep(y=pack_state(hydrogen_state))
    assert np.isfinite(hydrogen_dt.dt_reactions)
    assert np.isfinite(hydrogen_dt.dt_electron_cooling)
    assert np.isfinite(hydrogen_dt.dt_ion_charge_exchange)
    hydrogen_result = hydrogen_sim.run(t_end=1.0e-10, dt=1.0e-10)
    assert hydrogen_result.steps == 1
    assert np.all(np.isfinite(hydrogen_result.n))
    assert np.all(np.isfinite(hydrogen_result.Te))
    assert np.all(np.isfinite(hydrogen_result.Ti))

    try:
        ion_charge_exchange_rhs(
            state=hydrogen_state,
            floors=hydrogen_sim.floors,
            ion_mass_g=hydrogen_sim.ion_mass_g,
            gas_type="Ar",
            Tn_fit=hydrogen_params["Tn_fit"],
            b_Qcx=hydrogen_params["b_Qcx"],
            cx=True,
        )
    except ValueError as exc:
        assert "unsupported gas_type" in str(exc)
    else:
        raise AssertionError("expected unsupported gas_type to fail")

    heat_state = conservative_from_primitives(
        n=np.full(geom.cells, 1.0e12),
        nn=state.nn,
        u=np.zeros(geom.cells),
        Te=np.linspace(2.0, 1.0, geom.cells),
        Ti=np.linspace(1.5, 0.5, geom.cells),
        ion_mass_g=sim.ion_mass_g,
    )
    heat_rhs = sim.heat_conduction_rhs(state=heat_state)
    for values in (heat_rhs.n, heat_rhs.nn, heat_rhs.M):
        assert np.allclose(values, 0.0)
    assert np.all(np.isfinite(heat_rhs.Ee))
    assert np.all(np.isfinite(heat_rhs.Ei))
    assert heat_rhs.Ee[0] < 0.0
    assert heat_rhs.Ee[-1] > 0.0
    assert heat_rhs.Ei[0] < 0.0
    assert heat_rhs.Ei[-1] > 0.0
    heat_energy_tol = 1e-12 * np.sum(
        np.abs(heat_rhs.Ee * geom.plasma_volume_cm3)
    )
    assert np.isclose(
        np.sum(heat_rhs.Ee * geom.plasma_volume_cm3),
        0.0,
        atol=heat_energy_tol,
    )
    heat_ion_energy_tol = 1e-12 * np.sum(
        np.abs(heat_rhs.Ei * geom.plasma_volume_cm3)
    )
    assert np.isclose(
        np.sum(heat_rhs.Ei * geom.plasma_volume_cm3),
        0.0,
        atol=heat_ion_energy_tol,
    )
    heat_dt = sim.suggest_timestep(y=pack_state(heat_state))
    assert np.isfinite(heat_dt.dt_heat_conduction)
    heat_derived = derive_state(heat_state, sim.floors, sim.ion_mass_g)
    heat_ln_lambda = np.maximum(
        c_log(heat_derived.Te, heat_state.n, kind="ei"),
        params["ln_lambda_min"],
    )
    dTe_dt = heat_rhs.Ee / (1.5 * heat_state.n * ev_to_erg)
    dTi_dt = heat_rhs.Ei / (1.5 * heat_state.n * ev_to_erg)
    legacy_dTe_dt = en_factor * elec_par_heat_div(
        heat_derived.Te,
        heat_state.n,
        geom.length_cm,
        heat_ln_lambda,
    )
    legacy_dTi_dt = en_factor * ion_par_heat_div(
        heat_derived.Ti,
        heat_state.n,
        geom.length_cm,
        sim.mu,
        heat_ln_lambda,
    )
    assert np.allclose(dTe_dt, legacy_dTe_dt)
    assert np.allclose(dTi_dt, legacy_dTi_dt)

    disabled_heat = heat_conduction_rhs(
        state=heat_state,
        floors=sim.floors,
        ion_mass_g=sim.ion_mass_g,
        mu=sim.mu,
        geometry=geom,
        b_epara=0.0,
        b_ipara=0.0,
        heat_conduction=True,
        ln_lambda_min=params["ln_lambda_min"],
    )
    assert np.allclose(disabled_heat.Ee, 0.0)
    assert np.allclose(disabled_heat.Ei, 0.0)

    implicit_heat_state = sim.implicit_heat_conduction_step(
        dt=heat_dt.dt_heat_conduction,
        state=heat_state,
    )
    implicit_heat_derived = derive_state(
        implicit_heat_state, sim.floors, sim.ion_mass_g
    )
    assert np.all(np.isfinite(implicit_heat_state.Ee))
    assert np.all(np.isfinite(implicit_heat_state.Ei))
    assert np.all(implicit_heat_derived.Te >= params["Te_floor"])
    assert np.all(implicit_heat_derived.Ti >= params["Ti_floor"])
    assert implicit_heat_derived.Te[0] < heat_derived.Te[0]
    assert implicit_heat_derived.Te[-1] > heat_derived.Te[-1]
    assert implicit_heat_derived.Ti[0] < heat_derived.Ti[0]
    assert implicit_heat_derived.Ti[-1] > heat_derived.Ti[-1]
    assert np.isclose(
        np.sum((implicit_heat_state.Ee - heat_state.Ee) * geom.plasma_volume_cm3),
        0.0,
        atol=heat_energy_tol,
    )
    assert np.isclose(
        np.sum((implicit_heat_state.Ei - heat_state.Ei) * geom.plasma_volume_cm3),
        0.0,
        atol=heat_ion_energy_tol,
    )

    small_dt = min(1.0e-11, 1.0e-4 * heat_dt.dt_heat_conduction)
    small_implicit = sim.implicit_heat_conduction_step(dt=small_dt, state=heat_state)
    assert np.allclose(
        (small_implicit.Ee - heat_state.Ee) / small_dt,
        heat_rhs.Ee,
        rtol=5.0e-4,
        atol=1.0e-8,
    )
    assert np.allclose(
        (small_implicit.Ei - heat_state.Ei) / small_dt,
        heat_rhs.Ei,
        rtol=5.0e-4,
        atol=1.0e-8,
    )

    disabled_implicit = implicit_heat_conduction_step(
        state=heat_state,
        floors=sim.floors,
        ion_mass_g=sim.ion_mass_g,
        mu=sim.mu,
        geometry=geom,
        dt=heat_dt.dt_heat_conduction,
        b_epara=0.0,
        b_ipara=0.0,
        heat_conduction=True,
        ln_lambda_min=params["ln_lambda_min"],
    )
    assert np.allclose(disabled_implicit.Ee, heat_state.Ee)
    assert np.allclose(disabled_implicit.Ei, heat_state.Ei)

    nonheat_rhs = sim.rhs(pack_state(heat_state), include_heat_conduction=False)
    full_rhs = sim.rhs(pack_state(heat_state), include_heat_conduction=True)
    heat_rhs_y = pack_state(heat_rhs)
    assert np.allclose(full_rhs - nonheat_rhs, heat_rhs_y)
    rhs_terms = sim.rhs_terms(pack_state(heat_state), include_heat_conduction=True)
    expected_rhs_terms = {
        "plasma_advective_flux",
        "plasma_front_flux",
        "pressure_work",
        "ei_exchange",
        "ionization_energy_cost",
        "electron_ion_cooling",
        "electron_neutral_cooling",
        "ion_charge_exchange",
        "surface_loss",
        "cathode_surface_loss",
        "neutral_exchange",
        "neutral_sources",
        "ionization_birth",
        "beam_ionization_birth",
        "beam_power_deposition",
        "beam_ionization_cost",
        "recombination_rad_loss",
        "recombination_3b_loss",
        "heat_conduction",
    }
    assert set(rhs_terms) == expected_rhs_terms
    term_sum = np.zeros_like(full_rhs)
    for term in rhs_terms.values():
        for field_name in STATE_NAMES_1D:
            assert np.all(np.isfinite(getattr(term, field_name)))
        term_sum = term_sum + pack_state(term)
    assert np.allclose(term_sum, full_rhs)
    nonheat_terms = sim.rhs_terms(
        pack_state(heat_state),
        include_heat_conduction=False,
    )
    assert np.allclose(pack_state(nonheat_terms["heat_conduction"]), 0.0)
    assert np.allclose(
        pack_state(rhs_terms["heat_conduction"]),
        full_rhs - nonheat_rhs,
    )
    assert np.allclose(pack_state(rhs_terms["cathode_surface_loss"]), 0.0)
    assert np.allclose(pack_state(rhs_terms["beam_ionization_birth"]), 0.0)
    assert np.allclose(pack_state(rhs_terms["beam_power_deposition"]), 0.0)
    assert np.allclose(pack_state(rhs_terms["beam_ionization_cost"]), 0.0)

    split_dt = min(1.0e-10, 0.1 * heat_dt.dt_heat_conduction)
    manual_explicit_y = ssprk2_step(
        y0=pack_state(heat_state),
        dt=split_dt,
        rhs_func=lambda yy: sim.rhs(yy, include_heat_conduction=False),
        floor_func=sim.floor_state_vector,
    )
    manual_heat_state = sim.implicit_heat_conduction_step(
        dt=split_dt,
        y=manual_explicit_y,
    )
    manual_split_y = sim.floor_state_vector(pack_state(manual_heat_state))
    split_y = sim.operator_split_step(y=pack_state(heat_state), dt=split_dt)
    assert np.allclose(split_y, manual_split_y)

    no_heat_bound_dt = sim.suggest_timestep(
        y=pack_state(heat_state),
        include_heat_conduction=False,
    )
    assert np.isinf(no_heat_bound_dt.dt_heat_conduction)
    assert no_heat_bound_dt.active_constraint != "heat_conduction"

    fast_state = conservative_from_primitives(
        n=np.full(geom.cells, params["ne0"]),
        nn=state.nn,
        u=np.full(geom.cells, 1.0e7),
        Te=np.full(geom.cells, params["Te0"]),
        Ti=np.full(geom.cells, params["Ti0"]),
        ion_mass_g=sim.ion_mass_g,
    )
    fast_dt = sim.suggest_timestep(y=pack_state(fast_state))
    assert fast_dt.dt_plasma_cfl < dt_default.dt_plasma_cfl

    no_source_params = dict(params)
    no_source_params["gas_puff_enabled"] = False
    no_source_params["pump_enabled"] = False
    no_source_params["b_ioniz"] = 0.0
    no_source_params["b_rec_rad"] = 0.0
    no_source_params["b_rec_3b"] = 0.0
    no_source_params["b_Qei"] = 0.0
    no_source_params["b_Qen"] = 0.0
    no_source_params["b_Qcx"] = 0.0
    no_source_params["b_ionization_energy_cost"] = 0.0
    no_source_params["b_surface_loss"] = 0.0
    no_source_sim = LAPDSim1D(no_source_params, flags)
    y_before = no_source_sim.get_initial_snapshot().y.copy()
    explicit_attempt = no_source_sim._attempt_step(dt=1e-10)
    assert np.isclose(explicit_attempt.dt, 1e-10)
    assert not explicit_attempt.operator_split
    assert np.isclose(no_source_sim.time, 0.0)
    assert np.allclose(no_source_sim.get_initial_snapshot().y, y_before)
    explicit_attempt_after = no_source_sim._accept_step_attempt(explicit_attempt)
    assert np.isclose(no_source_sim.time, 1e-10)
    assert np.allclose(explicit_attempt_after.y, y_before, rtol=0.0, atol=1e-20)

    no_source_sim = LAPDSim1D(no_source_params, flags)
    y_before = no_source_sim.get_initial_snapshot().y.copy()
    stationary_after = no_source_sim.advance_one_step(1e-10)
    assert np.allclose(stationary_after.y, y_before, rtol=0.0, atol=1e-20)

    split_flags = dict(flags)
    split_flags["implicit_heat_conduction"] = True
    no_source_split_sim = LAPDSim1D(no_source_params, split_flags)
    split_before = no_source_split_sim.get_initial_snapshot().y.copy()
    split_attempt = no_source_split_sim._attempt_step(dt=1e-10)
    assert split_attempt.operator_split
    assert np.isclose(no_source_split_sim.time, 0.0)
    assert np.allclose(no_source_split_sim.get_initial_snapshot().y, split_before)
    split_attempt_after = no_source_split_sim._accept_step_attempt(split_attempt)
    assert np.isclose(no_source_split_sim.time, 1e-10)
    assert np.allclose(split_attempt_after.y, split_before, rtol=0.0, atol=1e-18)

    no_source_split_sim = LAPDSim1D(no_source_params, split_flags)
    split_before = no_source_split_sim.get_initial_snapshot().y.copy()
    split_stationary_after = no_source_split_sim.advance_one_step(1e-10)
    assert np.allclose(split_stationary_after.y, split_before, rtol=0.0, atol=1e-18)

    run_params = dict(no_source_params)
    run_params["dt_save"] = 0.0
    run_sim = LAPDSim1D(run_params, flags)
    try:
        run_sim.get_results()
    except RuntimeError as exc:
        assert "simulation has not been run yet" in str(exc)
    else:
        raise AssertionError("expected get_results before a run to fail")
    run_before = run_sim.get_initial_snapshot().y.copy()
    run_result = run_sim.run(t_end=3.0e-10, dt=1.0e-10)
    assert run_sim.get_results() is run_result
    assert run_result.steps == 3
    assert np.isclose(run_result.final_time, 3.0e-10)
    assert run_result.time.shape == (4,)
    assert run_result.phase.shape == (4,)
    assert np.all(run_result.phase == "pre_breakdown")
    assert np.allclose(run_result.phase_elapsed, run_result.time)
    assert np.allclose(run_result.phase_cathode_enabled, 0.0)
    assert np.allclose(run_result.phase_gas_puff_enabled, 0.0)
    assert np.allclose(run_result.phase_floating, 0.0)
    assert np.isnan(run_result.t_prebreakdown_trigger)
    assert np.isnan(run_result.t_breakdown_trigger)
    assert np.isnan(run_result.t_breakdown)
    assert np.isnan(run_result.t_breakdown_ms)
    assert np.allclose(run_result.time_since_breakdown, run_result.time)
    assert np.allclose(run_result.time_ms_since_breakdown, 1.0e3 * run_result.time)
    assert np.allclose(run_result.phase_events["time"], [0.0])
    assert list(run_result.phase_events["phase"]) == ["pre_breakdown"]
    assert list(run_result.phase_events["reason"]) == ["initial"]
    assert np.allclose(run_result.timestep_rejection_events["time"], [])
    assert list(run_result.timestep_rejection_events["reason"]) == []
    assert np.allclose(run_result.current_trigger_samples["time"], [])
    assert np.allclose(run_result.current_trigger_samples["I_tot"], [])
    assert run_result.y.shape == (4, run_before.size)
    assert run_result.n.shape == (4, geom.cells)
    assert len(run_result.diagnostics) == 3
    assert [diag.phase for diag in run_result.diagnostics] == [
        "pre_breakdown",
        "pre_breakdown",
        "pre_breakdown",
    ]
    assert np.allclose([diag.accepted_dt for diag in run_result.diagnostics], 1.0e-10)
    assert [diag.step_cap for diag in run_result.diagnostics] == [
        "fixed_dt",
        "fixed_dt",
        "fixed_dt",
    ]
    assert np.allclose(
        [diag.time for diag in run_result.diagnostics],
        [0.0, 1.0e-10, 2.0e-10],
    )

    capped_params = dict(run_params)
    capped_params["max_steps"] = 2
    capped_sim = LAPDSim1D(capped_params, flags)
    try:
        capped_sim.run(t_end=3.0e-10, dt=1.0e-10)
    except RuntimeError as exc:
        assert "max_steps=2 reached" in str(exc)
    else:
        raise AssertionError("expected configured max_steps cap to abort the run")

    unlimited_params = dict(run_params)
    unlimited_params["max_steps"] = 0
    unlimited_sim = LAPDSim1D(unlimited_params, flags)
    unlimited_result = unlimited_sim.run(t_end=3.0e-10, dt=1.0e-10)
    assert unlimited_result.steps == 3
    assert np.allclose(
        [diag.phase_cathode_enabled for diag in run_result.diagnostics],
        0.0,
    )
    assert np.allclose(
        [diag.phase_gas_puff_enabled for diag in run_result.diagnostics],
        0.0,
    )
    assert np.allclose(
        [diag.phase_floating for diag in run_result.diagnostics],
        0.0,
    )
    assert set(run_result.rhs_terms) == expected_rhs_terms
    assert set(run_result.electron_energy_terms_W_cm3) == expected_rhs_terms
    assert set(run_result.ion_energy_terms_W_cm3) == expected_rhs_terms
    assert run_result.cathode_diagnostics["enabled"].shape == (4,)
    assert np.allclose(run_result.cathode_diagnostics["enabled"], 0.0)
    assert np.allclose(run_result.cathode_diagnostics["configured"], 0.0)
    assert np.allclose(run_result.cathode_diagnostics["phase_enabled"], 0.0)
    assert np.allclose(run_result.cathode_diagnostics["rhs_enabled"], 0.0)
    assert np.allclose(run_result.cathode_diagnostics["solve_enabled"], 0.0)
    assert np.allclose(run_result.cathode_diagnostics["floating"], 0.0)
    assert np.allclose(run_result.cathode_diagnostics["has_solution"], 0.0)
    assert run_result.cathode_diagnostics["beam_cross"].shape == (4, geom.cells)
    assert np.allclose(run_result.cathode_diagnostics["beam_cross"], 0.0)
    assert np.all(np.isnan(run_result.cathode_diagnostics["source_phi_c"]))
    assert np.all(run_result.cathode_diagnostics["source_regime"] == "none")
    assert np.all(run_result.cathode_diagnostics["end_regime"] == "none")
    saved_term_sum = np.zeros_like(run_result.y)
    for term_name in expected_rhs_terms:
        term_fields = run_result.rhs_terms[term_name]
        for field_name in STATE_NAMES_1D:
            assert term_fields[field_name].shape == (4, geom.cells)
            assert np.all(np.isfinite(term_fields[field_name]))
        assert np.allclose(
            run_result.electron_energy_terms_W_cm3[term_name],
            1.0e-7 * term_fields["Ee"],
        )
        assert np.allclose(
            run_result.ion_energy_terms_W_cm3[term_name],
            1.0e-7 * term_fields["Ei"],
        )
        saved_term_sum = saved_term_sum + np.concatenate(
            [term_fields[field_name] for field_name in STATE_NAMES_1D],
            axis=1,
        )
    packed_total_rhs = np.concatenate(
        [run_result.total_rhs[field_name] for field_name in STATE_NAMES_1D],
        axis=1,
    )
    assert np.allclose(saved_term_sum, packed_total_rhs)
    assert np.allclose(run_result.y, run_before[None, :], rtol=0.0, atol=1e-20)
    assert np.allclose(run_result.time, [0.0, 1.0e-10, 2.0e-10, 3.0e-10])
    for field_name in (
        "ne",
        "v_plasma",
        "Ne_flux",
        "Nn_flux",
        "S_ion_bulk",
        "S_ion_beam",
        "S_rec_rad",
        "S_rec_3b",
        "Qie",
        "Qei",
        "Qen",
        "Qcx",
        "Qeb",
        "Qib",
        "e_par_flux",
        "i_par_flux",
        "e_perp_hl",
        "i_perp_hl",
        "cathode",
        "cathode_twin",
    ):
        assert hasattr(run_result, field_name)
    assert np.allclose(run_result.ne, run_result.n)
    assert np.allclose(run_result.v_plasma, run_result.u)
    assert run_result.Ne_flux.shape == run_result.n.shape
    assert run_result.Nn_flux.shape == run_result.n.shape
    assert run_result.S_ion_bulk.shape == run_result.n.shape
    assert run_result.Qie.shape == run_result.n.shape
    assert np.allclose(
        run_result.Ne_flux,
        (
            run_result.rhs_terms["plasma_advective_flux"]["n"]
            + run_result.rhs_terms["plasma_front_flux"]["n"]
            + run_result.rhs_terms["surface_loss"]["n"]
            + run_result.rhs_terms["cathode_surface_loss"]["n"]
        ),
    )
    assert np.allclose(
        run_result.Nn_flux,
        (
            run_result.rhs_terms["neutral_exchange"]["nn"]
            + run_result.rhs_terms["surface_loss"]["nn"]
            + run_result.rhs_terms["cathode_surface_loss"]["nn"]
        ),
    )
    assert np.allclose(
        run_result.S_ion_bulk,
        run_result.rhs_terms["ionization_birth"]["n"],
    )
    assert np.allclose(
        run_result.S_ion_beam,
        run_result.rhs_terms["beam_ionization_birth"]["n"],
    )
    assert np.allclose(
        run_result.S_rec_rad,
        -run_result.rhs_terms["recombination_rad_loss"]["n"],
    )
    assert np.allclose(
        run_result.S_rec_3b,
        -run_result.rhs_terms["recombination_3b_loss"]["n"],
    )
    assert np.allclose(
        run_result.Qei,
        -run_result.electron_energy_terms_W_cm3["electron_ion_cooling"],
    )
    assert np.allclose(
        run_result.Qen,
        -run_result.electron_energy_terms_W_cm3["electron_neutral_cooling"],
    )
    assert np.allclose(
        run_result.e_par_flux,
        run_result.electron_energy_terms_W_cm3["heat_conduction"],
    )
    assert np.allclose(
        run_result.i_par_flux,
        run_result.ion_energy_terms_W_cm3["heat_conduction"],
    )
    assert np.allclose(run_result.e_perp_hl, 0.0)
    assert np.allclose(run_result.i_perp_hl, 0.0)
    assert run_result.sim3_compat_units["energy_terms"] == "W/cm^3"
    assert run_result.sim3_compat_units["time_ms_since_breakdown"] == "ms"
    assert "power density" in run_result.sim3_compat_notes["Qei"]
    assert "breakdown-relative" in run_result.sim3_compat_notes["time_since_breakdown"]
    assert run_result.cathode.I_tot.shape == run_result.time.shape
    assert np.all(np.isnan(run_result.cathode.I_tot))

    entry_flags = dict(flags)
    entry_flags["neutral_equilibration"] = False
    entry_flags["launch_plasma_after_equilibration"] = False
    entry_sim = LAPDSim1D(run_params, entry_flags)
    entry_sim.start_simulation(t_end=3.0e-10, dt=1.0e-10)
    entry_result = entry_sim.get_results()
    assert entry_result.steps == run_result.steps
    assert np.isclose(entry_result.final_time, run_result.final_time)
    assert np.allclose(entry_result.time, run_result.time)
    assert np.allclose(entry_result.y, run_result.y, rtol=0.0, atol=1e-20)

    progress_fractions = []
    progress_snapshots = []
    progress_sim = LAPDSim1D(run_params, flags)
    progress_result = progress_sim.run(
        t_end=3.0e-10,
        dt=1.0e-10,
        progress_callback=progress_fractions.append,
        progress_tracker=progress_snapshots.append,
    )
    assert progress_result.steps == 3
    assert np.allclose(progress_fractions, [1 / 3, 1.0])
    assert len(progress_snapshots) == 2
    assert all(isinstance(progress, SimulationProgress1D) for progress in progress_snapshots)
    assert np.isclose(progress_snapshots[-1].fraction, 1.0)
    assert np.isclose(progress_snapshots[-1].time, progress_result.final_time)
    assert progress_snapshots[-1].step == progress_result.steps
    assert progress_snapshots[-1].saved_samples == len(progress_result.time)
    assert progress_snapshots[-1].step_cap == "fixed_dt"
    assert progress_snapshots[-1].timestep_limiters
    assert len(progress_snapshots[-1].timestep_limiters) <= 3
    assert all(
        isinstance(name, str) and np.isfinite(dt)
        for name, dt in progress_snapshots[-1].timestep_limiters
    )
    printer_stream = StringIO()
    progress_printer = ProgressPrinter1D(
        interval_fraction=0.0,
        interval_steps=100,
        stream=printer_stream,
    )
    progress_printer(progress_snapshots[-1])
    progress_printer(
        SimulationProgress1D(
            fraction=0.25,
            time=1.0e-10,
            t_end=4.0e-10,
            step=1,
            max_steps=0,
            accepted_dt=1.0e-10,
            suggested_dt=1.0e-10,
            step_cap="fixed_dt",
            active_constraint="dt_max",
            retry_count=0,
            rejection_reason="",
            phase="neutral_prebreakdown",
            saved_samples=1,
            wall_elapsed_s=0.0,
            wall_remaining_s=0.0,
        )
    )
    printer_output = printer_stream.getvalue()
    assert printer_output.count("sim1d progress:") == 2
    assert "limiters=" in printer_output
    every_step_progress = []
    LAPDSim1D(run_params, flags).run(
        t_end=3.0e-10,
        dt=1.0e-10,
        progress_callback=every_step_progress.append,
        progress_interval_s=0.0,
    )
    assert np.allclose(every_step_progress, [1 / 3, 2 / 3, 1.0])

    default_end_params = dict(run_params)
    default_end_params["tau_prebreakdown"] = 1.0e-10
    default_end_params["tau_breakdown"] = 1.0e-10
    default_end_params["tau_discharge"] = 1.0e-10
    default_end_params["tau_afterglow"] = 1.0e-10
    default_end_sim = LAPDSim1D(default_end_params, flags)
    assert np.isclose(default_end_sim.default_t_end(), 4.0e-10)
    default_end_result = default_end_sim.run(dt=1.0e-10)
    assert np.isclose(default_end_result.final_time, 4.0e-10)
    assert np.allclose(
        default_end_result.time,
        [0.0, 1.0e-10, 2.0e-10, 3.0e-10, 4.0e-10],
    )
    run_summary = summarize_result(run_result)
    run_summary_from_solver = LAPDSim1D.summarize_result(run_result)
    for summary in (run_summary, run_summary_from_solver):
        assert summary.finite
        assert summary.samples == 4
        assert summary.steps == run_result.steps
        assert np.isclose(summary.final_time, run_result.final_time)
        assert summary.n_min >= no_source_params["ne_floor"]
        assert summary.nn_min >= no_source_params["nn_floor"]
        assert summary.Te_min >= no_source_params["Te_floor"]
        assert summary.Ti_min >= no_source_params["Ti_floor"]
        assert np.isclose(
            summary.total_particle_inventory_relative_drift,
            0.0,
            atol=1e-14,
        )
        assert np.isclose(summary.thermal_energy_relative_drift, 0.0, atol=1e-14)
        assert summary.phase_counts == {"pre_breakdown": 4}
        assert summary.diagnostic_phase_counts == {"pre_breakdown": 3}
        assert summary.phase_event_count == 1
        assert summary.phase_event_phase_counts == {"pre_breakdown": 1}
        assert summary.phase_event_reason_counts == {"initial": 1}
        assert summary.last_phase_event == {
            "time": 0.0,
            "phase": "pre_breakdown",
            "reason": "initial",
        }
        assert summary.current_trigger_sample_count == 0
        assert summary.last_current_trigger_sample is None
        assert summary.phase_switch_fractions == {
            "cathode_enabled": 0.0,
            "floating": 0.0,
            "gas_puff_enabled": 0.0,
        }
        assert summary.cathode_diagnostic_fractions["configured"] == 0.0
        assert summary.cathode_diagnostic_fractions["solve_enabled"] == 0.0
        assert summary.cathode_diagnostic_fractions["has_solution"] == 0.0
        assert summary.constraint_counts == {"heat_conduction": 3}
        assert summary.step_cap_counts == {"fixed_dt": 3}
        assert np.isclose(summary.accepted_dt_min, 1.0e-10)
        assert np.isclose(summary.accepted_dt_max, 1.0e-10)
        assert summary.retrying_step_count == 0
        assert summary.total_retry_count == 0
        assert summary.max_retry_count == 0
        assert summary.rejection_reason_counts == {}
        assert summary.timestep_rejection_event_count == 0
        assert summary.timestep_rejection_reason_counts == {}
        assert summary.last_timestep_rejection_event is None
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = run_sim.save_result(
            f"{tmpdir}/sim1d_smoke.h5",
            run_result,
        )
        with h5py.File(output_path, "r") as h5:
            assert h5.attrs["format"] == "sim1d-hdf5-v1"
            assert h5.attrs["solver"] == "LAPDSim1D"
            assert h5.attrs["steps"] == run_result.steps
            assert np.isclose(h5.attrs["final_time"], run_result.final_time)
            assert np.isnan(h5.attrs["t_prebreakdown_trigger"])
            assert np.isnan(h5.attrs["t_breakdown_trigger"])
            saved_params = json.loads(h5.attrs["params_json"])
            saved_flags = json.loads(h5.attrs["flags_json"])
            assert saved_params["dt_save"] == run_params["dt_save"]
            assert saved_flags["front_flux"] == flags["front_flux"]
            assert h5["time"].shape == run_result.time.shape
            assert h5["phase"].shape == run_result.phase.shape
            assert h5["phase_elapsed"].shape == run_result.phase_elapsed.shape
            assert h5["phase_cathode_enabled"].shape == run_result.phase.shape
            assert h5["phase_gas_puff_enabled"].shape == run_result.phase.shape
            assert h5["phase_floating"].shape == run_result.phase.shape
            assert h5["phase_events/time"].shape == (1,)
            assert h5["phase_events/phase"].shape == (1,)
            assert h5["phase_events/reason"].shape == (1,)
            assert h5["timestep_rejection_events/time"].shape == (0,)
            assert h5["timestep_rejection_events/reason"].shape == (0,)
            assert h5["current_trigger_samples/time"].shape == (0,)
            assert h5["current_trigger_samples/I_tot"].shape == (0,)
            assert h5["cathode_diagnostics/solve_enabled"].shape == (4,)
            assert h5["cathode_diagnostics/floating"].shape == (4,)
            assert all(
                value.decode("utf-8") == "pre_breakdown"
                for value in h5["phase"][()]
            )
            assert h5["y"].shape == run_result.y.shape
            assert h5["n"].shape == run_result.n.shape
            assert h5["geometry/cell_role"].shape == (geom.cells,)
            assert h5["geometry/cell_role"][0].decode("utf-8") == "source"
            assert h5["geometry/cell_role"][-1].decode("utf-8") == "end"
            assert h5["rhs_terms/pressure_work/Ee"].shape == (4, geom.cells)
            assert h5["total_rhs/Ee"].shape == (4, geom.cells)
            assert (
                h5["electron_energy_terms_W_cm3/pressure_work"].shape
                == (4, geom.cells)
            )
            assert (
                h5["ion_energy_terms_W_cm3/pressure_work"].shape
                == (4, geom.cells)
            )
            assert h5["diagnostics"].attrs["count"] == len(run_result.diagnostics)
            assert h5["diagnostics/dt"].shape == (len(run_result.diagnostics),)
            assert h5["diagnostics/accepted_dt"].shape == (
                len(run_result.diagnostics),
            )
            assert h5["diagnostics/step_cap"].shape == (len(run_result.diagnostics),)
            assert h5["diagnostics/active_constraint"].shape == (
                len(run_result.diagnostics),
            )
            assert h5["diagnostics/time"].shape == (len(run_result.diagnostics),)
            assert h5["diagnostics/phase"].shape == (len(run_result.diagnostics),)
            assert all(
                value.decode("utf-8") == "pre_breakdown"
                for value in h5["diagnostics/phase"][()]
            )
        loaded_result = load_result_hdf5(output_path)
        loaded_via_solver = LAPDSim1D.load_result(output_path)
        for loaded in (loaded_result, loaded_via_solver):
            assert loaded.path == output_path
            assert loaded.steps == run_result.steps
            assert np.isclose(loaded.final_time, run_result.final_time)
            assert np.isnan(loaded.t_prebreakdown_trigger)
            assert np.isnan(loaded.t_breakdown_trigger)
            assert np.isnan(loaded.t_breakdown)
            assert np.isnan(loaded.t_breakdown_ms)
            assert np.allclose(loaded.time_since_breakdown, loaded.time)
            assert np.allclose(
                loaded.time_ms_since_breakdown,
                1.0e3 * loaded.time,
            )
            assert np.allclose(loaded.phase_events["time"], [0.0])
            assert list(loaded.phase_events["phase"]) == ["pre_breakdown"]
            assert list(loaded.phase_events["reason"]) == ["initial"]
            assert np.allclose(loaded.timestep_rejection_events["time"], [])
            assert list(loaded.timestep_rejection_events["reason"]) == []
            assert np.allclose(loaded.current_trigger_samples["time"], [])
            assert np.allclose(loaded.current_trigger_samples["I_tot"], [])
            assert loaded.params["dt_save"] == run_params["dt_save"]
            assert loaded.flags["front_flux"] == flags["front_flux"]
            assert np.allclose(loaded.time, run_result.time)
            assert np.all(loaded.phase == run_result.phase)
            assert np.allclose(loaded.phase_elapsed, run_result.phase_elapsed)
            assert np.allclose(
                loaded.phase_cathode_enabled,
                run_result.phase_cathode_enabled,
            )
            assert np.allclose(
                loaded.phase_gas_puff_enabled,
                run_result.phase_gas_puff_enabled,
            )
            assert np.allclose(loaded.phase_floating, run_result.phase_floating)
            assert np.allclose(loaded.y, run_result.y)
            assert np.allclose(loaded.n, run_result.n)
            assert np.allclose(loaded.ne, run_result.ne)
            assert np.allclose(loaded.v_plasma, run_result.v_plasma)
            assert np.allclose(loaded.Ne_flux, run_result.Ne_flux)
            assert np.allclose(loaded.S_ion_bulk, run_result.S_ion_bulk)
            assert np.allclose(loaded.Qie, run_result.Qie)
            assert np.allclose(loaded.Qeb, run_result.Qeb)
            assert loaded.cathode.I_tot.shape == run_result.cathode.I_tot.shape
            assert np.allclose(loaded.Te, run_result.Te)
            assert np.all(loaded.cell_role == run_result.cell_role)
            assert set(loaded.rhs_terms) == expected_rhs_terms
            assert np.allclose(
                loaded.cathode_diagnostics["has_solution"],
                run_result.cathode_diagnostics["has_solution"],
            )
            assert np.allclose(
                loaded.cathode_diagnostics["solve_enabled"],
                run_result.cathode_diagnostics["solve_enabled"],
            )
            assert np.allclose(
                loaded.cathode_diagnostics["floating"],
                run_result.cathode_diagnostics["floating"],
            )
            assert np.all(
                loaded.cathode_diagnostics["source_regime"]
                == run_result.cathode_diagnostics["source_regime"]
            )
            assert np.allclose(
                loaded.rhs_terms["pressure_work"]["Ee"],
                run_result.rhs_terms["pressure_work"]["Ee"],
            )
            assert np.allclose(
                loaded.total_rhs["Ee"],
                run_result.total_rhs["Ee"],
            )
            assert np.allclose(
                loaded.electron_energy_terms_W_cm3["pressure_work"],
                run_result.electron_energy_terms_W_cm3["pressure_work"],
            )
            assert len(loaded.diagnostics) == len(run_result.diagnostics)
            assert np.isclose(loaded.diagnostics[0].dt, run_result.diagnostics[0].dt)
            assert np.isclose(
                loaded.diagnostics[0].accepted_dt,
                run_result.diagnostics[0].accepted_dt,
            )
            assert loaded.diagnostics[0].step_cap == run_result.diagnostics[0].step_cap
            assert np.isclose(
                loaded.diagnostics[0].time,
                run_result.diagnostics[0].time,
            )
            assert loaded.diagnostics[0].phase == run_result.diagnostics[0].phase
            assert np.isclose(
                loaded.diagnostics[0].phase_gas_puff_enabled,
                run_result.diagnostics[0].phase_gas_puff_enabled,
            )
            assert (
                loaded.diagnostics[0].active_constraint
                == run_result.diagnostics[0].active_constraint
            )

    cathode_run_params = dict(no_source_params)
    cathode_run_params["dt_save"] = 0.0
    cathode_run_flags = dict(flags)
    cathode_run_flags["cathode_coupling"] = True
    cathode_run_sim = LAPDSim1D(cathode_run_params, cathode_run_flags)
    cathode_run_result = cathode_run_sim.run(t_end=3.0e-10, dt=1.0e-10)
    assert cathode_run_result.steps == 3
    assert np.isclose(cathode_run_result.final_time, 3.0e-10)
    assert cathode_run_result.time.shape == (4,)
    assert np.all(np.isfinite(cathode_run_result.y))
    assert set(cathode_run_result.rhs_terms) == expected_rhs_terms
    assert np.allclose(cathode_run_result.phase_cathode_enabled, 1.0)
    assert np.allclose(cathode_run_result.phase_gas_puff_enabled, 0.0)
    assert np.allclose(cathode_run_result.phase_floating, 0.0)
    cathode_diag = cathode_run_result.cathode_diagnostics
    assert cathode_diag["enabled"].shape == (4,)
    assert np.allclose(cathode_diag["enabled"], 1.0)
    assert np.allclose(cathode_diag["configured"], 1.0)
    assert np.allclose(cathode_diag["phase_enabled"], 1.0)
    assert np.allclose(cathode_diag["rhs_enabled"], 1.0)
    assert np.allclose(cathode_diag["solve_enabled"], 1.0)
    assert np.allclose(cathode_diag["floating"], 0.0)
    assert np.allclose(cathode_diag["has_solution"], 1.0)
    assert np.allclose(cathode_diag["has_twin_solution"], 0.0)
    assert np.all(cathode_diag["source_phi_c"] > 0.0)
    assert np.all(cathode_diag["source_I_i"] > 0.0)
    assert np.all(cathode_diag["source_I_tot"] > 0.0)
    assert np.all(cathode_diag["source_P_prim"] >= 0.0)
    assert np.all(cathode_diag["source_P_ohmic"] >= 0.0)
    assert np.all(cathode_diag["source_P_loss"] > 0.0)
    assert np.all(np.isnan(cathode_diag["end_phi_c"]))
    assert np.all(
        np.isin(cathode_diag["source_regime"], ["classical", "virtual_cathode"])
    )
    assert np.all(cathode_diag["end_regime"] == "none")
    assert cathode_diag["beam_cross"].shape == (4, geom.cells)
    assert np.all(cathode_diag["beam_cross"][:, 0] > 0.0)
    assert np.allclose(cathode_diag["beam_cross"][:, 1:], 0.0)
    assert np.all(cathode_diag["n_beam"][:, 0] > 0.0)
    assert np.all(cathode_diag["v_beam"][:, 0] > 0.0)
    assert np.all(cathode_diag["l_b_profile"] > 0.0)
    assert np.allclose(cathode_diag["l_b_profile_twin"], 0.0)
    assert np.any(cathode_run_result.rhs_terms["cathode_surface_loss"]["n"] < 0.0)
    assert np.any(cathode_run_result.rhs_terms["cathode_surface_loss"]["nn"] > 0.0)
    assert np.any(cathode_run_result.rhs_terms["cathode_surface_loss"]["Ee"] < 0.0)
    assert np.any(cathode_run_result.rhs_terms["beam_ionization_birth"]["n"] > 0.0)
    assert np.all(
        cathode_run_result.rhs_terms["beam_power_deposition"]["Ee"] >= 0.0
    )
    assert np.any(
        cathode_run_result.rhs_terms["beam_power_deposition"]["Ee"] > 0.0
    )
    assert np.all(cathode_run_result.rhs_terms["beam_ionization_cost"]["Ee"] <= 0.0)
    assert np.any(cathode_run_result.rhs_terms["beam_ionization_cost"]["Ee"] < 0.0)
    assert np.allclose(cathode_run_result.rhs_terms["surface_loss"]["n"][:, 0], 0.0)
    assert np.all(cathode_run_result.cathode.I_tot[:4] > 0.0)
    assert np.allclose(
        cathode_run_result.S_ion_beam,
        cathode_run_result.rhs_terms["beam_ionization_birth"]["n"],
    )
    assert np.any(cathode_run_result.S_ion_beam > 0.0)
    assert np.allclose(
        cathode_run_result.Qeb,
        (
            cathode_run_result.electron_energy_terms_W_cm3[
                "beam_power_deposition"
            ]
            + cathode_run_result.electron_energy_terms_W_cm3[
                "beam_ionization_cost"
            ]
            + cathode_run_result.electron_energy_terms_W_cm3[
                "cathode_surface_loss"
            ]
        ),
    )
    assert np.any(cathode_run_result.Qeb > 0.0)
    cathode_saved_sum = np.zeros_like(cathode_run_result.y)
    for term_name in expected_rhs_terms:
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
            assert np.all(h5["cathode_diagnostics/source_I_tot"][()] > 0.0)
            assert all(
                value.decode("utf-8") in {"classical", "virtual_cathode"}
                for value in h5["cathode_diagnostics/source_regime"][()]
            )
            assert np.all(h5["cathode_diagnostics/beam_cross"][()][:, 0] > 0.0)
            assert np.any(h5["rhs_terms/cathode_surface_loss/n"][()] < 0.0)
            assert np.any(h5["rhs_terms/beam_power_deposition/Ee"][()] > 0.0)
            assert np.any(h5["rhs_terms/beam_ionization_cost/Ee"][()] < 0.0)
        loaded_cathode_result = load_result_hdf5(output_path)
        assert loaded_cathode_result.flags["cathode_coupling"]
        assert np.allclose(
            loaded_cathode_result.phase_cathode_enabled,
            cathode_run_result.phase_cathode_enabled,
        )
        assert set(loaded_cathode_result.rhs_terms) == expected_rhs_terms
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
        assert np.allclose(
            loaded_cathode_result.cathode.I_tot,
            cathode_run_result.cathode.I_tot,
        )
        assert np.allclose(
            loaded_cathode_result.S_ion_beam,
            cathode_run_result.S_ion_beam,
        )
        assert np.allclose(
            loaded_cathode_result.Qeb,
            cathode_run_result.Qeb,
        )
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
    assert growth_result.steps == 3
    assert np.allclose(growth_result.time, [0.0, 0.5e-6, 1.125e-6, 1.5e-6])
    assert [diag.step_cap for diag in growth_result.diagnostics] == [
        "phase_boundary",
        "dt_growth",
        "t_end",
    ]
    assert np.allclose(
        [diag.accepted_dt for diag in growth_result.diagnostics],
        [0.5e-6, 0.625e-6, 0.375e-6],
    )
    growth_summary = summarize_result(growth_result)
    assert growth_summary.step_cap_counts == {
        "dt_growth": 1,
        "phase_boundary": 1,
        "t_end": 1,
    }

    retry_params = dict(params)
    retry_flags = dict(flags)
    retry_flags["Plasma"] = False
    retry_flags["heat_conduction"] = False
    retry_params["dt_save"] = 0.0
    retry_params["pump_enabled"] = False
    retry_params["dt_max"] = 1.0e-6
    retry_params["neutral_dt_fraction"] = 100.0
    retry_params["max_neutral_step_fraction"] = 6.0
    retry_sim = LAPDSim1D(retry_params, retry_flags)
    retry_result = retry_sim.run(t_end=1.0e-6)
    assert retry_result.steps == 2
    assert np.allclose(retry_result.time, [0.0, 0.5e-6, 1.0e-6])
    assert retry_result.diagnostics[0].retry_count == 1
    assert retry_result.diagnostics[0].rejection_reason == "neutral_step_fraction"
    assert retry_result.diagnostics[0].step_cap == "retry"
    assert np.isclose(retry_result.diagnostics[0].accepted_dt, 0.5e-6)
    assert retry_result.diagnostics[1].retry_count == 0
    assert np.allclose(retry_result.timestep_rejection_events["time"], [0.0])
    assert np.allclose(
        retry_result.timestep_rejection_events["attempted_dt"],
        [1.0e-6],
    )
    assert np.allclose(retry_result.timestep_rejection_events["retry_index"], [0])
    assert list(retry_result.timestep_rejection_events["reason"]) == [
        "neutral_step_fraction"
    ]
    assert list(retry_result.timestep_rejection_events["phase"]) == [
        "equilibrium_puff"
    ]
    retry_summary = summarize_result(retry_result)
    assert retry_summary.step_cap_counts == {"retry": 1, "t_end": 1}
    assert retry_summary.retrying_step_count == 1
    assert retry_summary.total_retry_count == 1
    assert retry_summary.max_retry_count == 1
    assert retry_summary.rejection_reason_counts == {"neutral_step_fraction": 1}
    assert retry_summary.timestep_rejection_event_count == 1
    assert retry_summary.timestep_rejection_reason_counts == {
        "neutral_step_fraction": 1
    }
    assert retry_summary.last_timestep_rejection_event == {
        "time": 0.0,
        "attempted_dt": 1.0e-6,
        "retry_index": 0,
        "reason": "neutral_step_fraction",
        "phase": "equilibrium_puff",
        "active_constraint": "dt_max",
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        retry_output = retry_sim.save_result(
            f"{tmpdir}/sim1d_retry_smoke.h5",
            retry_result,
        )
        with h5py.File(retry_output, "r") as h5:
            assert h5["timestep_rejection_events/time"].shape == (1,)
            assert h5["timestep_rejection_events/attempted_dt"].shape == (1,)
            assert h5["timestep_rejection_events/reason"].shape == (1,)
        loaded_retry = load_result_hdf5(retry_output)
        assert np.allclose(
            loaded_retry.timestep_rejection_events["attempted_dt"],
            retry_result.timestep_rejection_events["attempted_dt"],
        )
        assert list(loaded_retry.timestep_rejection_events["reason"]) == list(
            retry_result.timestep_rejection_events["reason"]
        )
        assert loaded_retry.diagnostics[0].retry_count == 1
        assert loaded_retry.diagnostics[0].rejection_reason == (
            "neutral_step_fraction"
        )

    failed_retry_params = dict(retry_params)
    failed_retry_params["max_neutral_step_fraction"] = 1.0e-30
    failed_retry_params["max_step_retries"] = 1
    failed_retry_sim = LAPDSim1D(failed_retry_params, retry_flags)
    failed_retry_y0 = failed_retry_sim.get_initial_snapshot().y.copy()
    try:
        failed_retry_sim.run(t_end=1.0e-6)
    except TimestepRejectionError as exc:
        assert exc.reason == "neutral_step_fraction"
        assert exc.retry_count == 1
        assert np.isclose(exc.time, 0.0)
        assert np.isclose(exc.attempted_dt, 0.5e-6)
        assert np.isclose(exc.dt_min, failed_retry_params["dt_min"])
        assert exc.phase == "equilibrium_puff"
        assert exc.active_constraint == "dt_max"
        assert np.isclose(failed_retry_sim.time, 0.0)
        assert np.allclose(failed_retry_sim.get_initial_snapshot().y, failed_retry_y0)
    else:
        raise AssertionError("expected TimestepRejectionError")

    nonfinite_retry_params = dict(run_params)
    nonfinite_retry_params["max_step_retries"] = 1
    nonfinite_retry_params["dt_min"] = 1.0e-12
    nonfinite_retry_sim = LAPDSim1D(nonfinite_retry_params, flags)
    nonfinite_y = nonfinite_retry_sim.get_initial_snapshot().y.copy()
    nonfinite_index = STATE_NAMES_1D.index("Ee") * geom.cells + 2
    nonfinite_y[nonfinite_index] = np.nan

    def nonfinite_attempt(dt=None, operator_split=None):
        return SimpleNamespace(
            y=nonfinite_y.copy(),
            dt=float(dt),
            operator_split=bool(operator_split),
            solver_cache=nonfinite_retry_sim._step_cache_snapshot(),
        )

    nonfinite_retry_sim._attempt_step = nonfinite_attempt
    try:
        nonfinite_retry_sim.run(t_end=1.0e-10, dt=1.0e-10)
    except TimestepRejectionError as exc:
        assert exc.reason == "nonfinite_state"
        assert exc.rejection_detail["fields"]["Ee"]["indices"] == [2]
        assert np.isnan(exc.rejection_detail["fields"]["Ee"]["values"][0])
        assert "Ee" in str(exc)
    else:
        raise AssertionError("expected non-finite TimestepRejectionError")

    split_run_sim = LAPDSim1D(run_params, split_flags)
    split_run_result = split_run_sim.run(t_end=2.0e-10, dt=1.0e-10)
    assert split_run_result.steps == 2
    assert np.isclose(split_run_result.final_time, 2.0e-10)
    assert np.all(np.isfinite(split_run_result.y))

    phase_params = dict(no_source_params)
    phase_params["dt_save"] = 0.0
    phase_params["tau_prebreakdown"] = 1.0e-10
    phase_params["tau_discharge"] = 2.0e-10
    phase_params["tau_afterglow"] = 1.0e-10
    phase_sim = LAPDSim1D(phase_params, flags)
    phase_result = phase_sim.run(t_end=4.0e-10, dt=1.0e-10)
    assert np.allclose(
        phase_result.time,
        [0.0, 1.0e-10, 2.0e-10, 3.0e-10, 4.0e-10],
    )
    assert list(phase_result.phase) == [
        "pre_breakdown",
        "main_discharge",
        "main_discharge",
        "afterglow",
        "post_afterglow",
    ]
    assert np.allclose(
        phase_result.phase_elapsed,
        [0.0, 0.0, 1.0e-10, 0.0, 0.0],
    )
    assert np.allclose(
        phase_result.phase_cathode_enabled,
        [0.0, 0.0, 0.0, 0.0, 0.0],
    )
    assert np.allclose(
        phase_result.phase_gas_puff_enabled,
        [0.0, 0.0, 0.0, 0.0, 0.0],
    )
    assert np.allclose(
        phase_result.phase_floating,
        [0.0, 0.0, 0.0, 1.0, 0.0],
    )
    phase_summary = summarize_result(phase_result)
    assert phase_summary.phase_counts == {
        "afterglow": 1,
        "main_discharge": 2,
        "post_afterglow": 1,
        "pre_breakdown": 1,
    }
    assert phase_summary.diagnostic_phase_counts == {
        "afterglow": 1,
        "main_discharge": 2,
        "pre_breakdown": 1,
    }
    assert phase_summary.phase_switch_fractions == {
        "cathode_enabled": 0.0,
        "floating": 0.2,
        "gas_puff_enabled": 0.0,
    }

    phase_capped_sim = LAPDSim1D(phase_params, flags)
    phase_capped_result = phase_capped_sim.run(t_end=4.0e-10, dt=5.0e-10)
    assert phase_capped_result.steps == 3
    assert np.allclose(
        phase_capped_result.time,
        [0.0, 1.0e-10, 3.0e-10, 4.0e-10],
    )
    assert list(phase_capped_result.phase) == [
        "pre_breakdown",
        "main_discharge",
        "afterglow",
        "post_afterglow",
    ]
    assert np.allclose(
        phase_capped_result.phase_elapsed,
        [0.0, 0.0, 0.0, 0.0],
    )

    phase_cathode_flags = dict(flags)
    phase_cathode_flags["cathode_coupling"] = True
    phase_cathode_sim = LAPDSim1D(phase_params, phase_cathode_flags)
    phase_cathode_result = phase_cathode_sim.run(t_end=4.0e-10, dt=1.0e-10)
    assert np.allclose(
        phase_cathode_result.phase_cathode_enabled,
        [1.0, 1.0, 1.0, 0.0, 0.0],
    )
    assert np.allclose(
        phase_cathode_result.cathode_diagnostics["configured"],
        [1.0, 1.0, 1.0, 1.0, 1.0],
    )
    assert np.allclose(
        phase_cathode_result.cathode_diagnostics["phase_enabled"],
        [1.0, 1.0, 1.0, 0.0, 0.0],
    )
    assert np.allclose(
        phase_cathode_result.cathode_diagnostics["rhs_enabled"],
        [1.0, 1.0, 1.0, 0.0, 0.0],
    )
    assert np.allclose(
        phase_cathode_result.cathode_diagnostics["solve_enabled"],
        [1.0, 1.0, 1.0, 1.0, 0.0],
    )
    assert np.allclose(
        phase_cathode_result.cathode_diagnostics["floating"],
        [0.0, 0.0, 0.0, 1.0, 0.0],
    )
    assert np.allclose(
        phase_cathode_result.cathode_diagnostics["has_solution"],
        [1.0, 1.0, 1.0, 1.0, 0.0],
    )
    assert np.allclose(
        phase_cathode_result.rhs_terms["cathode_surface_loss"]["n"][3:],
        0.0,
    )
    assert np.allclose(
        phase_cathode_result.rhs_terms["beam_ionization_birth"]["n"][3:],
        0.0,
    )

    breakdown_params = dict(phase_params)
    breakdown_params["tau_breakdown"] = 1.0e-10
    breakdown_sim = LAPDSim1D(breakdown_params, flags)
    assert breakdown_sim.phase_at_time(0.0) == "pre_breakdown"
    assert breakdown_sim.phase_at_time(1.0e-10) == "breakdown"
    assert breakdown_sim.phase_at_time(2.0e-10) == "main_discharge"
    assert breakdown_sim.phase_at_time(4.0e-10) == "afterglow"
    assert breakdown_sim.phase_at_time(5.0e-10) == "post_afterglow"
    assert np.isclose(
        breakdown_sim.next_phase_boundary_after(0.0),
        1.0e-10,
    )
    assert np.isclose(
        breakdown_sim.next_phase_boundary_after(1.0e-10),
        2.0e-10,
    )
    breakdown_result = breakdown_sim.run(t_end=5.0e-10, dt=1.0e-10)
    assert np.allclose(
        breakdown_result.time,
        [0.0, 1.0e-10, 2.0e-10, 3.0e-10, 4.0e-10, 5.0e-10],
    )
    assert list(breakdown_result.phase) == [
        "pre_breakdown",
        "breakdown",
        "main_discharge",
        "main_discharge",
        "afterglow",
        "post_afterglow",
    ]
    assert np.allclose(
        breakdown_result.phase_elapsed,
        [0.0, 0.0, 0.0, 1.0e-10, 0.0, 0.0],
    )
    assert np.allclose(
        breakdown_result.phase_cathode_enabled,
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    assert np.allclose(
        breakdown_result.phase_gas_puff_enabled,
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    assert np.allclose(
        breakdown_result.phase_floating,
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    )
    assert np.allclose(
        breakdown_result.phase_events["time"],
        [0.0, 1.0e-10, 2.0e-10, 4.0e-10, 5.0e-10],
    )
    assert list(breakdown_result.phase_events["phase"]) == [
        "pre_breakdown",
        "breakdown",
        "main_discharge",
        "afterglow",
        "post_afterglow",
    ]
    assert list(breakdown_result.phase_events["reason"]) == [
        "initial",
        "tau_prebreakdown",
        "tau_breakdown",
        "tau_discharge",
        "tau_afterglow",
    ]
    breakdown_summary = summarize_result(breakdown_result)
    assert breakdown_summary.phase_counts == {
        "afterglow": 1,
        "breakdown": 1,
        "main_discharge": 2,
        "post_afterglow": 1,
        "pre_breakdown": 1,
    }
    assert breakdown_summary.diagnostic_phase_counts == {
        "afterglow": 1,
        "breakdown": 1,
        "main_discharge": 2,
        "pre_breakdown": 1,
    }
    assert breakdown_summary.phase_event_count == 5
    assert breakdown_summary.phase_event_phase_counts == {
        "afterglow": 1,
        "breakdown": 1,
        "main_discharge": 1,
        "post_afterglow": 1,
        "pre_breakdown": 1,
    }
    assert breakdown_summary.phase_event_reason_counts == {
        "initial": 1,
        "tau_afterglow": 1,
        "tau_breakdown": 1,
        "tau_discharge": 1,
        "tau_prebreakdown": 1,
    }
    assert breakdown_summary.last_phase_event == {
        "time": 5.0e-10,
        "phase": "post_afterglow",
        "reason": "tau_afterglow",
    }

    breakdown_capped_sim = LAPDSim1D(breakdown_params, flags)
    breakdown_capped_result = breakdown_capped_sim.run(
        t_end=5.0e-10,
        dt=1.0e-9,
    )
    assert breakdown_capped_result.steps == 4
    assert np.allclose(
        breakdown_capped_result.time,
        [0.0, 1.0e-10, 2.0e-10, 4.0e-10, 5.0e-10],
    )
    assert list(breakdown_capped_result.phase) == [
        "pre_breakdown",
        "breakdown",
        "main_discharge",
        "afterglow",
        "post_afterglow",
    ]
    assert np.allclose(
        breakdown_capped_result.phase_elapsed,
        [0.0, 0.0, 0.0, 0.0, 0.0],
    )
    assert np.allclose(
        breakdown_capped_result.phase_events["time"],
        [0.0, 1.0e-10, 2.0e-10, 4.0e-10, 5.0e-10],
    )

    breakdown_mid_sim = LAPDSim1D(breakdown_params, flags)
    breakdown_mid_sim.run(t_end=1.0e-10, dt=1.0e-10)
    breakdown_mid_result = breakdown_mid_sim.run(t_end=5.0e-10, dt=1.0e-10)
    assert np.allclose(
        breakdown_mid_result.phase_events["time"],
        [1.0e-10, 2.0e-10, 4.0e-10, 5.0e-10],
    )
    assert list(breakdown_mid_result.phase_events["phase"]) == [
        "breakdown",
        "main_discharge",
        "afterglow",
        "post_afterglow",
    ]
    assert list(breakdown_mid_result.phase_events["reason"]) == [
        "initial",
        "tau_breakdown",
        "tau_discharge",
        "tau_afterglow",
    ]

    breakdown_cathode_flags = dict(flags)
    breakdown_cathode_flags["cathode_coupling"] = True
    breakdown_cathode_sim = LAPDSim1D(
        breakdown_params,
        breakdown_cathode_flags,
    )
    breakdown_cathode_result = breakdown_cathode_sim.run(
        t_end=5.0e-10,
        dt=1.0e-10,
    )
    assert np.allclose(
        breakdown_cathode_result.phase_cathode_enabled,
        [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
    )

    current_phase_params = dict(no_source_params)
    current_phase_params["dt_save"] = 0.0
    current_phase_params["phase_transition_mode"] = "current"
    current_phase_params["tau_prebreakdown"] = 5.0e-10
    current_phase_params["tau_discharge"] = 2.0e-10
    current_phase_params["tau_afterglow"] = 1.0e-10
    current_phase_params["I_prebreakdown"] = 1.0e-9
    current_phase_params["I_breakdown"] = 1.0e-9
    current_phase_flags = dict(flags)
    current_phase_flags["cathode_coupling"] = True
    current_phase_sim = LAPDSim1D(current_phase_params, current_phase_flags)
    current_phase_result = current_phase_sim.run(t_end=5.0e-10, dt=1.0e-10)
    neutral_prebreakdown_params = dict(current_phase_params)
    neutral_prebreakdown_params["gas_puff_enabled"] = True
    neutral_prebreakdown_params["tau_neutral_prebreakdown"] = 2.0e-10
    neutral_prebreakdown_flags = dict(current_phase_flags)
    neutral_prebreakdown_flags["neutral_prebreakdown"] = True
    neutral_prebreakdown_sim = LAPDSim1D(
        neutral_prebreakdown_params,
        neutral_prebreakdown_flags,
    )
    neutral_prebreakdown_result = neutral_prebreakdown_sim.run(dt=1.0e-10)
    assert np.isclose(neutral_prebreakdown_result.final_time, 6.0e-10)
    assert np.isclose(
        neutral_prebreakdown_result.t_prebreakdown_trigger,
        2.0e-10,
    )
    assert np.isclose(neutral_prebreakdown_result.t_breakdown_trigger, 3.0e-10)
    assert list(neutral_prebreakdown_result.phase[:2]) == [
        "neutral_prebreakdown",
        "neutral_prebreakdown",
    ]
    assert np.allclose(neutral_prebreakdown_result.phase_cathode_enabled[:2], 0.0)
    assert np.allclose(neutral_prebreakdown_result.phase_gas_puff_enabled[:2], 1.0)
    assert np.allclose(
        neutral_prebreakdown_result.n[1],
        neutral_prebreakdown_result.n[0],
    )
    dynamic_current_phase_sim = LAPDSim1D(current_phase_params, current_phase_flags)
    dynamic_progress_snapshots = []
    dynamic_current_phase_initial_t_end = dynamic_current_phase_sim.default_t_end()
    dynamic_current_phase_result = dynamic_current_phase_sim.run(
        dt=1.0e-10,
        progress_tracker=dynamic_progress_snapshots.append,
        progress_interval_s=1.0,
    )
    assert np.isclose(dynamic_current_phase_result.final_time, 5.0e-10)
    assert np.isclose(dynamic_current_phase_result.t_breakdown_trigger, 2.0e-10)
    assert len(dynamic_progress_snapshots) == 3
    assert np.isclose(
        dynamic_progress_snapshots[0].t_end,
        dynamic_current_phase_initial_t_end,
    )
    assert np.isclose(dynamic_progress_snapshots[1].time, 2.0e-10)
    assert np.isclose(dynamic_progress_snapshots[1].t_end, 5.0e-10)
    assert np.isclose(dynamic_progress_snapshots[-1].fraction, 1.0)
    assert np.allclose(
        dynamic_current_phase_result.phase_events["time"],
        [0.0, 1.0e-10, 2.0e-10, 4.0e-10, 5.0e-10],
    )
    assert np.isclose(current_phase_sim._t_prebreakdown_trigger, 1.0e-10)
    assert np.isclose(current_phase_sim._t_breakdown_trigger, 2.0e-10)
    assert np.isclose(current_phase_result.t_prebreakdown_trigger, 1.0e-10)
    assert np.isclose(current_phase_result.t_breakdown_trigger, 2.0e-10)
    assert np.isclose(current_phase_result.t_breakdown, 2.0e-10)
    assert np.isclose(current_phase_result.t_breakdown_ms, 2.0e-7)
    assert np.allclose(
        current_phase_result.time,
        [0.0, 1.0e-10, 2.0e-10, 3.0e-10, 4.0e-10, 5.0e-10],
    )
    assert np.allclose(
        current_phase_result.time_since_breakdown,
        current_phase_result.time - 2.0e-10,
    )
    assert np.allclose(
        current_phase_result.time_ms_since_breakdown,
        1.0e3 * (current_phase_result.time - 2.0e-10),
    )
    assert list(current_phase_result.phase) == [
        "pre_breakdown",
        "breakdown",
        "main_discharge",
        "main_discharge",
        "afterglow",
        "post_afterglow",
    ]
    assert np.allclose(
        current_phase_result.phase_elapsed,
        [0.0, 0.0, 0.0, 1.0e-10, 0.0, 0.0],
    )
    assert np.allclose(
        current_phase_result.phase_cathode_enabled,
        [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
    )
    assert np.allclose(
        current_phase_result.phase_floating,
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    )
    assert np.all(
        current_phase_result.cathode_diagnostics["source_I_tot"][:4] > 0.0
    )
    assert np.all(
        np.isnan(current_phase_result.cathode_diagnostics["source_I_tot"][5:])
    )
    current_phase_summary = summarize_result(current_phase_result)
    assert current_phase_summary.phase_counts == {
        "afterglow": 1,
        "breakdown": 1,
        "main_discharge": 2,
        "post_afterglow": 1,
        "pre_breakdown": 1,
    }
    assert current_phase_summary.diagnostic_phase_counts == {
        "afterglow": 1,
        "breakdown": 1,
        "main_discharge": 2,
        "pre_breakdown": 1,
    }
    assert current_phase_summary.phase_event_count == 5
    assert current_phase_summary.phase_event_reason_counts == {
        "I_breakdown": 1,
        "I_prebreakdown": 1,
        "initial": 1,
        "tau_afterglow": 1,
        "tau_discharge": 1,
    }
    assert np.allclose(
        current_phase_result.phase_events["time"],
        [0.0, 1.0e-10, 2.0e-10, 4.0e-10, 5.0e-10],
    )
    assert np.allclose(
        current_phase_result.current_trigger_samples["time"],
        [1.0e-10, 2.0e-10],
    )
    assert np.allclose(
        current_phase_result.current_trigger_samples["I_tot"],
        current_phase_result.cathode_diagnostics["source_I_tot"][1:3],
    )
    assert current_phase_summary.current_trigger_sample_count == 2
    assert np.isclose(
        current_phase_summary.last_current_trigger_sample["time"],
        current_phase_result.current_trigger_samples["time"][-1],
    )
    assert np.isclose(
        current_phase_summary.last_current_trigger_sample["I_tot"],
        current_phase_result.current_trigger_samples["I_tot"][-1],
    )
    assert list(current_phase_result.phase_events["phase"]) == [
        "pre_breakdown",
        "breakdown",
        "main_discharge",
        "afterglow",
        "post_afterglow",
    ]
    assert list(current_phase_result.phase_events["reason"]) == [
        "initial",
        "I_prebreakdown",
        "I_breakdown",
        "tau_discharge",
        "tau_afterglow",
    ]
    current_sample_1 = current_phase_result.cathode_diagnostics["source_I_tot"][1]
    current_sample_2 = current_phase_result.cathode_diagnostics["source_I_tot"][2]
    assert current_sample_2 > current_sample_1
    interpolated_current_phase_params = dict(current_phase_params)
    interpolated_I_breakdown = 0.5 * (current_sample_1 + current_sample_2)
    interpolated_current_phase_params["I_breakdown"] = interpolated_I_breakdown
    interpolated_current_phase_sim = LAPDSim1D(
        interpolated_current_phase_params,
        current_phase_flags,
    )
    interpolated_current_phase_result = interpolated_current_phase_sim.run(
        t_end=5.0e-10,
        dt=1.0e-10,
    )
    expected_breakdown_time = 1.0e-10 + (
        (interpolated_I_breakdown - current_sample_1)
        / (current_sample_2 - current_sample_1)
    ) * 1.0e-10
    assert np.isclose(
        interpolated_current_phase_result.t_prebreakdown_trigger,
        1.0e-10,
    )
    assert np.isclose(
        interpolated_current_phase_result.t_breakdown_trigger,
        expected_breakdown_time,
    )
    assert np.allclose(
        interpolated_current_phase_result.phase_events["time"],
        [
            0.0,
            1.0e-10,
            expected_breakdown_time,
            expected_breakdown_time + current_phase_params["tau_discharge"],
            expected_breakdown_time
            + current_phase_params["tau_discharge"]
            + current_phase_params["tau_afterglow"],
        ],
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        current_phase_output = current_phase_sim.save_result(
            f"{tmpdir}/sim1d_current_phase_smoke.h5",
            current_phase_result,
        )
        with h5py.File(current_phase_output, "r") as h5:
            assert np.isclose(h5.attrs["t_prebreakdown_trigger"], 1.0e-10)
            assert np.isclose(h5.attrs["t_breakdown_trigger"], 2.0e-10)
            assert h5["phase_events/time"].shape == (5,)
            assert h5["current_trigger_samples/time"].shape == (2,)
            assert h5["current_trigger_samples/I_tot"].shape == (2,)
        loaded_current_phase = load_result_hdf5(current_phase_output)
        assert np.isclose(loaded_current_phase.t_prebreakdown_trigger, 1.0e-10)
        assert np.isclose(loaded_current_phase.t_breakdown_trigger, 2.0e-10)
        assert np.isclose(loaded_current_phase.t_breakdown, 2.0e-10)
        assert np.isclose(loaded_current_phase.t_breakdown_ms, 2.0e-7)
        assert np.allclose(
            loaded_current_phase.time_since_breakdown,
            current_phase_result.time_since_breakdown,
        )
        assert np.allclose(
            loaded_current_phase.time_ms_since_breakdown,
            current_phase_result.time_ms_since_breakdown,
        )
        assert np.allclose(
            loaded_current_phase.phase_events["time"],
            current_phase_result.phase_events["time"],
        )
        assert np.all(
            loaded_current_phase.phase_events["phase"]
            == current_phase_result.phase_events["phase"]
        )
        assert np.all(
            loaded_current_phase.phase_events["reason"]
            == current_phase_result.phase_events["reason"]
        )
        assert np.allclose(
            loaded_current_phase.current_trigger_samples["time"],
            current_phase_result.current_trigger_samples["time"],
        )
        assert np.allclose(
            loaded_current_phase.current_trigger_samples["I_tot"],
            current_phase_result.current_trigger_samples["I_tot"],
        )

    direct_current_phase_params = dict(current_phase_params)
    direct_current_phase_params["I_prebreakdown"] = 0.0
    direct_current_phase_sim = LAPDSim1D(
        direct_current_phase_params,
        current_phase_flags,
    )
    direct_current_phase_result = direct_current_phase_sim.run(
        t_end=4.0e-10,
        dt=1.0e-10,
    )
    assert direct_current_phase_sim._t_prebreakdown_trigger is None
    assert np.isclose(direct_current_phase_sim._t_breakdown_trigger, 1.0e-10)
    assert np.isnan(direct_current_phase_result.t_prebreakdown_trigger)
    assert np.isclose(direct_current_phase_result.t_breakdown_trigger, 1.0e-10)
    assert np.allclose(
        direct_current_phase_result.phase_events["time"],
        [0.0, 1.0e-10, 3.0e-10, 4.0e-10],
    )
    assert list(direct_current_phase_result.phase_events["phase"]) == [
        "pre_breakdown",
        "main_discharge",
        "afterglow",
        "post_afterglow",
    ]
    assert list(direct_current_phase_result.phase_events["reason"]) == [
        "initial",
        "I_breakdown",
        "tau_discharge",
        "tau_afterglow",
    ]
    assert list(direct_current_phase_result.phase) == [
        "pre_breakdown",
        "main_discharge",
        "main_discharge",
        "afterglow",
        "post_afterglow",
    ]

    failed_current_phase_params = dict(current_phase_params)
    failed_current_phase_params["I_prebreakdown"] = 1.0e30
    failed_current_phase_params["I_breakdown"] = 1.0e30
    failed_current_phase_sim = LAPDSim1D(
        failed_current_phase_params,
        current_phase_flags,
    )
    try:
        failed_current_phase_sim.run(t_end=5.0e-10, dt=1.0e-10)
    except BreakdownError as exc:
        assert "plasma failed to break down" in str(exc)
        assert exc.phase == "pre_breakdown"
        assert np.isclose(exc.time, failed_current_phase_params["tau_prebreakdown"])
        assert np.isfinite(exc.I_tot)
        assert exc.I_tot < failed_current_phase_params["I_prebreakdown"]
        assert np.isclose(exc.threshold, failed_current_phase_params["I_prebreakdown"])
        assert exc.threshold_name == "I_prebreakdown"
        assert np.isclose(
            exc.tau_prebreakdown,
            failed_current_phase_params["tau_prebreakdown"],
        )
        assert exc.details == {
            "phase": exc.phase,
            "time": exc.time,
            "I_tot": exc.I_tot,
            "threshold": exc.threshold,
            "threshold_name": exc.threshold_name,
            "tau_prebreakdown": exc.tau_prebreakdown,
        }
        assert np.allclose(exc.phase_events["time"], [0.0])
        assert list(exc.phase_events["phase"]) == ["pre_breakdown"]
        assert list(exc.phase_events["reason"]) == ["initial"]
        assert np.allclose(
            exc.current_trigger_samples["time"],
            [1.0e-10, 2.0e-10, 3.0e-10, 4.0e-10, 5.0e-10],
        )
        assert exc.current_trigger_samples["I_tot"].shape == (5,)
        assert np.all(np.isfinite(exc.current_trigger_samples["I_tot"]))
        assert exc.current_trigger_samples["I_tot"][-1] == exc.I_tot
    else:
        raise AssertionError("expected current-triggered run to fail breakdown")

    failed_breakdown_phase_params = dict(current_phase_params)
    failed_breakdown_phase_params["I_prebreakdown"] = 1.0e-9
    failed_breakdown_phase_params["I_breakdown"] = 1.0e30
    failed_breakdown_phase_sim = LAPDSim1D(
        failed_breakdown_phase_params,
        current_phase_flags,
    )
    try:
        failed_breakdown_phase_sim.run(t_end=5.0e-10, dt=1.0e-10)
    except BreakdownError as exc:
        assert "plasma failed to reach breakdown current" in str(exc)
        assert exc.phase == "breakdown"
        assert np.isclose(exc.time, failed_breakdown_phase_params["tau_prebreakdown"])
        assert exc.I_tot > 0.0
        assert np.isclose(exc.threshold, failed_breakdown_phase_params["I_breakdown"])
        assert exc.threshold_name == "I_breakdown"
        assert np.allclose(exc.phase_events["time"], [0.0, 1.0e-10])
        assert list(exc.phase_events["phase"]) == [
            "pre_breakdown",
            "breakdown",
        ]
        assert list(exc.phase_events["reason"]) == [
            "initial",
            "I_prebreakdown",
        ]
        assert np.allclose(
            exc.current_trigger_samples["time"],
            [1.0e-10, 2.0e-10, 3.0e-10, 4.0e-10, 5.0e-10],
        )
        assert exc.current_trigger_samples["I_tot"].shape == (5,)
        assert np.all(np.isfinite(exc.current_trigger_samples["I_tot"]))
        assert exc.current_trigger_samples["I_tot"][-1] == exc.I_tot
    else:
        raise AssertionError("expected current-triggered breakdown phase to fail")

    neutral_phase_run_params = dict(no_source_params)
    neutral_phase_run_params["dt_save"] = 0.0
    neutral_phase_run_params["tau_discharge"] = 2.0e-10
    neutral_phase_run_params["tau_cycle"] = 5.0e-10
    neutral_phase_run_params["cycles"] = 2
    neutral_phase_run_flags = dict(flags)
    neutral_phase_run_flags["Plasma"] = False
    neutral_phase_run_sim = LAPDSim1D(
        neutral_phase_run_params,
        neutral_phase_run_flags,
    )
    neutral_phase_result = neutral_phase_run_sim.run(t_end=4.0e-10, dt=1.0e-10)
    assert list(neutral_phase_result.phase) == [
        "equilibrium_puff",
        "equilibrium_puff",
        "equilibrium_off",
        "equilibrium_off",
        "equilibrium_off",
    ]
    assert np.allclose(
        neutral_phase_result.phase_gas_puff_enabled,
        [0.0, 0.0, 0.0, 0.0, 0.0],
    )
    assert np.allclose(
        neutral_phase_result.phase_events["time"],
        [0.0, 2.0e-10],
    )
    assert list(neutral_phase_result.phase_events["phase"]) == [
        "equilibrium_puff",
        "equilibrium_off",
    ]
    assert list(neutral_phase_result.phase_events["reason"]) == [
        "initial",
        "tau_discharge",
    ]
    neutral_phase_summary = summarize_result(neutral_phase_result)
    assert neutral_phase_summary.phase_event_count == 2
    assert neutral_phase_summary.phase_event_phase_counts == {
        "equilibrium_off": 1,
        "equilibrium_puff": 1,
    }
    assert neutral_phase_summary.phase_event_reason_counts == {
        "initial": 1,
        "tau_discharge": 1,
    }
    assert neutral_phase_summary.last_phase_event == {
        "time": 2.0e-10,
        "phase": "equilibrium_off",
        "reason": "tau_discharge",
    }
    assert neutral_phase_summary.current_trigger_sample_count == 0
    assert neutral_phase_summary.last_current_trigger_sample is None
    neutral_cycles_sim = LAPDSim1D(
        neutral_phase_run_params,
        neutral_phase_run_flags,
    )
    assert np.isclose(neutral_cycles_sim.default_t_end(), 1.0e-9)
    neutral_cycles_sim.start_simulation(dt=1.0e-9)
    neutral_cycles_result = neutral_cycles_sim.get_results()
    assert neutral_cycles_result.steps == 4
    assert np.isclose(neutral_cycles_result.final_time, 1.0e-9)
    assert np.allclose(
        neutral_cycles_result.time,
        [0.0, 2.0e-10, 5.0e-10, 7.0e-10, 1.0e-9],
    )
    assert [diag.step_cap for diag in neutral_cycles_result.diagnostics] == [
        "phase_boundary",
        "phase_boundary",
        "phase_boundary",
        "t_end",
    ]
    assert list(neutral_cycles_result.phase_events["reason"]) == [
        "initial",
        "tau_discharge",
        "tau_cycle",
        "tau_discharge",
        "tau_cycle",
    ]
    equilibration_params = dict(neutral_phase_run_params)
    equilibration_params["neutral_equilibration_cycles"] = 2
    equilibration_params["neutral_equilibration_dt"] = 1.0e-10
    equilibration_flags = dict(flags)
    equilibration_flags["neutral_equilibration"] = True
    equilibration_flags["launch_plasma_after_equilibration"] = False
    equilibration_sim = LAPDSim1D(equilibration_params, equilibration_flags)
    equilibration_sim.start_simulation(dt=1.0e-10)
    equilibration_result = equilibration_sim.get_results()
    equilibration_summary = equilibration_sim.get_neutral_equilibration_summary()
    assert equilibration_result is equilibration_sim.get_neutral_equilibration_results()
    assert equilibration_result.neutral_equilibration_summary is equilibration_summary
    assert equilibration_summary.cycles == 2
    assert np.isclose(equilibration_summary.final_time, 1.0e-9)
    assert np.isclose(equilibration_summary.mean_nn, np.mean(equilibration_result.nn[-1]))
    assert np.isclose(equilibration_summary.std_nn, np.std(equilibration_result.nn[-1]))
    assert not hasattr(equilibration_result, "neutral_equilibration")

    launch_flags = dict(equilibration_flags)
    launch_flags["launch_plasma_after_equilibration"] = True
    launch_sim = LAPDSim1D(equilibration_params, launch_flags)
    launch_sim.start_simulation(t_end=2.0e-10, dt=1.0e-10)
    launch_result = launch_sim.get_results()
    assert np.isclose(launch_result.final_time, 2.0e-10)
    assert hasattr(launch_result, "neutral_equilibration")
    assert launch_result.neutral_equilibration_summary.cycles == 2
    assert np.allclose(
        launch_result.nn[0],
        launch_result.neutral_equilibration.nn[-1],
    )
    neutral_phase_capped_sim = LAPDSim1D(
        neutral_phase_run_params,
        neutral_phase_run_flags,
    )
    neutral_phase_capped_result = neutral_phase_capped_sim.run(
        t_end=6.0e-10,
        dt=1.0e-9,
    )
    assert neutral_phase_capped_result.steps == 3
    assert np.allclose(
        neutral_phase_capped_result.time,
        [0.0, 2.0e-10, 5.0e-10, 6.0e-10],
    )
    assert list(neutral_phase_capped_result.phase) == [
        "equilibrium_puff",
        "equilibrium_off",
        "equilibrium_puff",
        "equilibrium_puff",
    ]
    assert np.allclose(
        neutral_phase_capped_result.phase_events["time"],
        [0.0, 2.0e-10, 5.0e-10],
    )
    assert list(neutral_phase_capped_result.phase_events["phase"]) == [
        "equilibrium_puff",
        "equilibrium_off",
        "equilibrium_puff",
    ]
    assert list(neutral_phase_capped_result.phase_events["reason"]) == [
        "initial",
        "tau_discharge",
        "tau_cycle",
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        from scripts.run_sim1d import _parse_args

        tmp_path = Path(tmpdir)
        assert (
            _parse_args(["--output", str(tmp_path / "default_operator_split.h5")])
            .operator_split
            is None
        )
        assert (
            _parse_args(
                [
                    "--output",
                    str(tmp_path / "forced_operator_split.h5"),
                    "--operator-split",
                ]
            ).operator_split
            is True
        )
        cli_config = tmp_path / "sim1d_cli_config.toml"
        cli_output = tmp_path / "sim1d_cli_output.h5"
        cli_config.write_text(
            "\n".join(
                [
                    "[params]",
                    "dt_save = 0.0",
                    "gas_puff_enabled = false",
                    "pump_enabled = false",
                    "b_ioniz = 0.0",
                    "b_rec_rad = 0.0",
                    "b_rec_3b = 0.0",
                    "b_Qei = 0.0",
                    "b_Qen = 0.0",
                    "b_Qcx = 0.0",
                    "b_ionization_energy_cost = 0.0",
                    "b_surface_loss = 0.0",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        cli_run = subprocess.run(
            [
                sys.executable,
                "scripts/run_sim1d.py",
                "--config",
                str(cli_config),
                "--output",
                str(cli_output),
                "--t-end",
                "2e-10",
                "--dt",
                "1e-10",
                "--max-steps",
                "10",
            ],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
        assert "sim1d run complete" in cli_run.stdout
        assert "sim1d health:" in cli_run.stdout
        assert "finite=True" in cli_run.stdout
        assert "steps=2" in cli_run.stdout
        cli_result = load_result_hdf5(cli_output)
        cli_summary = summarize_result(cli_result)
        assert cli_result.steps == 2
        assert np.isclose(cli_result.final_time, 2.0e-10)
        assert cli_result.time.shape == (3,)
        assert cli_result.params["dt_save"] == 0.0
        assert cli_result.params["b_surface_loss"] == 0.0
        assert np.all(np.isfinite(cli_result.y))
        assert cli_summary.finite
        assert np.isclose(
            cli_summary.total_particle_inventory_relative_drift,
            0.0,
            atol=1e-14,
        )
        cli_plot_dir = tmp_path / "sim1d_cli_plots"
        cli_plot_run = subprocess.run(
            [
                sys.executable,
                "scripts/plot_sim1d_run.py",
                str(cli_output),
                "--output-dir",
                str(cli_plot_dir),
                "--prefix",
                "cli",
            ],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
        assert "sim1d plots written:" in cli_plot_run.stdout
        assert (cli_plot_dir / "cli_summary.png").exists()
        assert (cli_plot_dir / "cli_densities.png").exists()

    ramp_y0 = pack_state(ramp_state)
    ramp_y1 = ssprk2_step(
        y0=ramp_y0,
        dt=1e-10,
        rhs_func=sim.rhs,
        floor_func=sim.floor_state_vector,
    )
    ramp_y1 = sim.floor_state_vector(ramp_y1)
    ramp_state_1 = unpack_state(ramp_y1, geom.cells)
    ramp_derived_1 = derive_state(ramp_state_1, sim.floors, sim.ion_mass_g)
    for values in (
        ramp_state_1.n,
        ramp_state_1.nn,
        ramp_state_1.M,
        ramp_state_1.Ee,
        ramp_state_1.Ei,
        ramp_derived_1.Te,
        ramp_derived_1.Ti,
    ):
        assert np.all(np.isfinite(values))
        assert np.all(values >= 0.0)
    ramp_after = sim.plasma_flux_rhs(y=ramp_y1, include_front=True)
    for values in (
        ramp_after.n,
        ramp_after.nn,
        ramp_after.M,
        ramp_after.Ee,
        ramp_after.Ei,
    ):
        assert np.all(np.isfinite(values))

    nn_ramp_y0 = pack_state(nn_ramp_state)
    nn_ramp_y1 = ssprk2_step(
        y0=nn_ramp_y0,
        dt=1e-10,
        rhs_func=sim.rhs,
        floor_func=sim.floor_state_vector,
    )
    nn_ramp_state_1 = unpack_state(sim.floor_state_vector(nn_ramp_y1), geom.cells)
    assert np.all(np.isfinite(nn_ramp_state_1.nn))
    assert np.all(nn_ramp_state_1.nn >= params["nn_floor"])

    print(
        "sim1d smoke ok: "
        f"cells={geom.cells}, dz={geom.dz_cm:g} cm, "
        f"Vp_total={geom.plasma_volume_cm3.sum():.6e} cm^3, "
        f"Vm_total={geom.neutral_volume_cm3.sum():.6e} cm^3"
    )


if __name__ == "__main__":
    main()
