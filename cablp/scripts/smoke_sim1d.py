import dataclasses
import json
from io import StringIO
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import h5py
import numpy as np

from cablp.funcs._adas import he_rate_temperature_range_eV
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
    conductive_face_flux,
    heat_conduction_rhs,
    implicit_heat_conduction_step,
)
from cablp.solvers._sim1d.physics.cathode import (
    _beam_smoothing_key,
    _beam_smoothing_matrix,
    beam_absorption_weights,
    beam_launch,
    cathode_sample_indices,
)
from cablp.solvers._sim1d.physics.energy import (
    electron_cooling_rhs,
    electron_cooling_rhs_terms,
    electron_ion_exchange_rhs,
    ion_charge_exchange_rhs,
)
from cablp.solvers._sim1d.physics.flux import front_filling_fluxes
from cablp.solvers._sim1d.core.integrator import ssprk2_step
from cablp.solvers._sim1d.core.geometry import (
    _source_fixed_grid_spec,
    anode_flanking_cells,
    cathode_adjacent_cells,
    is_plenum_cell,
    puff_cell_indices,
    pump_cell_indices,
)
from cablp.solvers._sim1d.physics.neutrals import (
    _effective_pump_speed,
    gas_puff_rate_profile,
    neutral_exchange_coefficients,
    neutral_exchange_two_zone_rhs,
    neutral_source_sink_rhs,
    neutral_thermal_speed,
    neutral_inventory_rate,
    neutral_wind_advection_rhs,
    neutral_zone_exchange_conductance,
    neutral_zone_exchange_rhs,
    neutral_zone_volumes,
    puff_rate,
    pump_rate,
    two_zone_knudsen_coefficients,
)
from cablp.solvers._sim1d.core.timestep import neutral_wind_timestep
from cablp.solvers._sim1d.physics.reactions import (
    particle_inventory_rate,
    reaction_rates,
    reaction_rhs_terms,
)
from cablp.solvers._sim1d.physics.sources import (
    add_state_rhs,
    ion_neutral_collision_frequency,
    ion_neutral_cx_frequency,
    ion_neutral_drag_rhs,
    ion_neutral_elastic_frequency,
    ion_neutral_frictional_heating_rhs,
    ion_neutral_slip_factor,
    ion_neutral_thermalization_rhs,
    langevin_rate_cm3_s,
    neutral_momentum_two_zone_rhs,
    neutral_momentum_wall_rhs,
    neutral_wind_two_zone_factors,
    neutral_wind_velocity,
    velocity_divergence,
)
from cablp.solvers._sim1d.core.state import (
    STATE_NAMES_1D,
    ConservativeState1D,
    apply_state_floors,
    conservative_from_primitives,
    derive_state,
    pack_state,
    state_field_names,
    unpack_state,
)
from cablp.vars._cons import I_ion, en_factor, ev_to_erg, m_p_cgs, qe_SI


# R5 stance flip (2026-07-25): the production defaults promote the full M6
# cathode/beam stack (csda + quasilinear, power_balance, gaussian, ads_des,
# presheath smoothing) and the R2/R3 fluid repairs. The cathode-MECHANISM unit
# tests below were written to isolate a single mechanism against the simple
# historical stance; this scoped helper returns that simple stance so they run
# as written. It keeps ion_neutral_moment_closure ON -- the ion-neutral drag is
# irrelevant to cathode emission/coverage, so the production baseline stays
# warning-free and the INERT-on-production params stay inert. Dedicated
# production cathode tests (csda / power_balance / gaussian / ads_des-subject /
# the R2/R3/R4/R5 blocks) do NOT use this and exercise the real defaults.
# Part of the R5 deprecation plan: when the beer_lambert / uniform / no-surface
# arms are retired, this helper and its callers are updated with them.
def _cathode_unit_config():
    """Return (params, flags) on the simple cathode/fluid stance for the
    cathode-mechanism unit tests (moment closure stays on)."""
    p, f = default_config()
    p.update({
        "beam_deposition_model": "beer_lambert",
        "beam_anomalous_model": "none",
        "cathode_warming_model": "none",
        "cathode_Ts_base_K": None,
        "cathode_heat_capacity_J_per_K": 3.0,
        "cathode_conduction_W_per_K": 0.0,
        "cathode_emission_profile": "uniform",
        "cathode_surface_model": "none",
        "cathode_phiwf_clean_eV": None,
        "cathode_cleaning_sigma_cm2": 0.0,
        "cathode_cleaning_E_th_eV": None,
        "cathode_sample_smoothing": None,
        "phi_wf": 3.0,
        # simple 1st-order integration so run-based tests match the analytic
        # backward-Euler forms they check
        "operator_splitting": "lie",
        "implicit_heat_scheme": "backward_euler",
        "heat_picard_iterations": 0,
    })
    f.update({
        "hyperbolic_energy_consistent": False,
        "characteristic_boundary": False,
        "front_flux": True,
    })
    return p, f


def main():
    # D2 retirement guards: stale selectors fail loudly at construction and
    # the production/default stance constructs warning-free.
    import warnings as _warnings

    _dep_params, _dep_flags = default_config()
    for _dep_p, _dep_f in (
        ({"cathode_solver_model": "voltage_driven"}, {}),
        ({"neutral_exchange_model": "molecular_flow"}, {}),
        ({"cathode_warming_model": "ion_bombardment"}, {}),
        ({}, {"resolved_boundaries": False}),
        ({"Lz": 1800.0}, {}),
    ):
        try:
            LAPDSim1D(
                {**_dep_params, **_dep_p}, {**_dep_flags, **_dep_f}
            )
        except ValueError as exc:
            assert "legacy-final-2026-07-22" in str(exc) or "D2" in str(exc)
        else:
            raise AssertionError(f"retired selector constructed: {_dep_p}, {_dep_f}")
    with _warnings.catch_warnings(record=True) as _caught:
        _warnings.simplefilter("always")
        LAPDSim1D(_dep_params, _dep_flags)
    assert not _caught, "production/default configuration must be warning-free"

    params, flags = default_config()
    assert params["cycles"] == 1
    assert params["phase_transition_mode"] == "current"
    assert params["gas_puff_mode"] == "square"
    assert params["tau_neutral_prebreakdown"] > 0.0
    assert flags["neutral_prebreakdown"]
    params["phase_transition_mode"] = "scheduled"
    params["gas_puff_mode"] = "decay_after_breakdown"
    params["gas_puff_profile"] = "cell"  # historical single-cell puff
    flags["neutral_prebreakdown"] = False
    flags["cathode_coupling"] = False
    flags["implicit_heat_conduction"] = False
    # The long-standing operator algebra below isolates the historical
    # all-cells path; dedicated R1/R2/R3/R4 blocks exercise the repaired live
    # defaults. Turn the R2/R3 boundary+flux repairs off here so the quiescent-
    # zero and operator-algebra invariants hold (moment closure stays on -- the
    # drag is orthogonal, so the production baseline stays warning-free).
    flags["active_plasma_topology"] = False
    flags["raw_stage_validation"] = False
    flags["hyperbolic_energy_consistent"] = False
    flags["characteristic_boundary"] = False
    flags["front_flux"] = True
    params["hyperbolic_wave_speed"] = "isothermal"
    sim = LAPDSim1D(params, flags)
    snapshot = sim.get_initial_snapshot()
    geom = snapshot.geometry
    state = snapshot.state
    derived = snapshot.derived

    assert geom.cells > params["nx"] + 2
    assert geom.length_cm.shape == (geom.cells,)
    assert geom.plasma_volume_cm3.shape == (geom.cells,)
    assert geom.neutral_volume_cm3.shape == (geom.cells,)
    assert geom.plasma_face_area_cm2.shape == (geom.cells + 1,)
    assert geom.neutral_face_area_cm2.shape == (geom.cells + 1,)
    assert geom.center_distance_cm.shape == (geom.cells - 1,)
    assert geom.z_edges_cm[0] < 0.0
    assert np.isclose(geom.z_edges_cm[-1], params["Lm"])
    assert geom.cell_role[0] == "plenum"
    assert geom.cell_role[-1] == "collector"
    assert np.all(geom.plasma_volume_cm3 > 0.0)
    assert np.all(geom.neutral_volume_cm3 > geom.plasma_volume_cm3)

    # Resolved typed-segment schema arrays are complete.
    for face_array in (
        geom.plasma_open,
        geom.heat_transmission,
        geom.neutral_face_hydraulic_radius_cm,
        geom.neutral_face_conductance_cm3_s,
    ):
        assert face_array.shape == (geom.cells + 1,)
    assert not geom.plasma_open[0] and not geom.plasma_open[-1]
    assert np.allclose(geom.neutral_hydraulic_radius_cm, geom.Rm_cm)
    assert np.all(np.isnan(geom.neutral_face_conductance_cm3_s))

    # Resolved typed-segment geometry is the only live machine.
    resolved_params, resolved_flags = default_config()
    resolved_flags["resolved_boundaries"] = True
    resolved_geom = LAPDSim1D(
        resolved_params, resolved_flags
    ).get_initial_snapshot().geometry
    assert resolved_geom.cells > resolved_params["nx"] + 2
    assert np.all(resolved_geom.plasma_volume_cm3 > 0.0)
    assert np.all(resolved_geom.neutral_volume_cm3 > resolved_geom.plasma_volume_cm3)
    assert {"plenum", "cathode", "gap", "puff", "column", "collector"} <= set(
        resolved_geom.cell_role
    )
    assert list(resolved_geom.cell_role[:2]) == ["plenum", "cathode"]
    assert resolved_geom.cell_role[-1] == "collector"
    assert not resolved_geom.plasma_open[0] and not resolved_geom.plasma_open[-1]

    # Cathode and anode are *surfaces* (plan §11 decision 5): the cathode surface
    # is the origin and the anode sits one gap downstream. Lm is measured from the
    # cathode surface, so the plenum lives at negative z and the mesh is longer.
    (cathode_face,) = resolved_geom.cathode_face_indices
    (anode_face,) = resolved_geom.anode_face_indices
    assert np.isclose(resolved_geom.z_edges_cm[cathode_face], 0.0)
    assert np.isclose(
        resolved_geom.z_edges_cm[anode_face],
        resolved_params["cathode_anode_gap_cm"],
    )
    assert np.isclose(resolved_geom.z_edges_cm[-1], resolved_params["Lm"])
    assert resolved_geom.z_edges_cm[0] < 0.0
    assert np.isclose(
        resolved_geom.z_edges_cm[0], -resolved_params["plenum_length_cm"]
    )
    assert resolved_geom.length_cm.sum() > resolved_params["Lm"]
    # Two cell counts: nx_gap across the gap, nx from the anode to the collector.
    assert anode_face - cathode_face == resolved_params["nx_gap"]
    gap_dz = resolved_params["cathode_anode_gap_cm"] / resolved_params["nx_gap"]
    assert np.allclose(
        resolved_geom.length_cm[cathode_face:anode_face], gap_dz
    )
    # The gap cells are the smallest in the mesh, so they set the explicit CFL.
    assert np.isclose(resolved_geom.length_cm.min(), gap_dz)

    # The cathode surface is a plasma wall; the anode face is interior and open.
    assert not resolved_geom.plasma_open[cathode_face]
    assert resolved_geom.heat_transmission[cathode_face] == 0.0
    assert resolved_geom.plasma_open[anode_face]
    assert cathode_adjacent_cells(resolved_geom) == (cathode_face,)
    assert resolved_geom.cell_role[cathode_face] == "cathode"
    assert anode_flanking_cells(resolved_geom) == ((anode_face - 1, anode_face),)
    assert resolved_geom.cell_role[anode_face - 1] == "gap"
    assert resolved_geom.cell_role[anode_face] == "puff"
    assert np.all(np.isnan(resolved_geom.neutral_face_conductance_cm3_s))

    # G1: default-off expanded end geometry. The provisional hardware arm
    # resolves a 150 cm, Rm=100 cm collector region in ten cells. Plasma area
    # is either unchanged (vessel-only) or smoothly flared; the source/end
    # params are presence-gated so incomplete or flag-off configs fail loudly.
    assert not resolved_flags["end_expansion_geometry"]
    assert resolved_params["end_expansion_cells"] is None
    assert resolved_params["end_expansion_machine_radius_cm"] is None
    assert resolved_params["end_expansion_plasma_radius_cm"] is None
    assert not resolved_flags["neutral_baffles"]
    assert resolved_params["neutral_baffle_positions_cm"] is None
    assert resolved_params["neutral_baffle_clear_radii_cm"] is None

    # CAD-pending thin annular baffles are default-off, presence-gated
    # neutral apertures. A 40 cm clear radius leaves the 18 cm plasma column
    # exactly unchanged and adds a series orifice only to neutral transport.
    baffle_params = dict(resolved_params)
    baffle_params.update(
        {
            "neutral_baffle_positions_cm": [150.0],
            "neutral_baffle_clear_radii_cm": [40.0],
        }
    )
    baffle_flags = {**resolved_flags, "neutral_baffles": True}
    baffle_geom = LAPDSim1D(
        baffle_params, baffle_flags
    ).get_initial_snapshot().geometry
    assert baffle_geom.neutral_baffle_face_indices.shape == (1,)
    assert np.allclose(baffle_geom.neutral_baffle_clear_radius_cm, [40.0])
    baffle_face = int(baffle_geom.neutral_baffle_face_indices[0])
    baffle_interior = baffle_face - 1
    assert abs(baffle_geom.z_edges_cm[baffle_face] - 150.0) <= (
        0.5
        * min(
            baffle_geom.length_cm[baffle_face - 1],
            baffle_geom.length_cm[baffle_face],
        )
    )
    assert np.isclose(
        baffle_geom.neutral_face_area_cm2[baffle_face], np.pi * 40.0**2
    )
    for name in (
        "plasma_area_cm2",
        "plasma_volume_cm3",
        "plasma_face_area_cm2",
        "plasma_open",
        "plasma_transmission",
        "heat_transmission",
    ):
        assert np.array_equal(
            getattr(baffle_geom, name), getattr(resolved_geom, name)
        ), name

    base_single = neutral_exchange_coefficients(
        geometry=resolved_geom,
        model="knudsen",
        constant_coeff_cm3_s=resolved_params["neutral_exchange_coeff_cm3_s"],
        Tn_K=resolved_params["Tn_K"],
        mu_neutral=4.0,
        clausing_scale=resolved_params["neutral_clausing_scale"],
    )
    baffle_single = neutral_exchange_coefficients(
        geometry=baffle_geom,
        model="knudsen",
        constant_coeff_cm3_s=baffle_params["neutral_exchange_coeff_cm3_s"],
        Tn_K=baffle_params["Tn_K"],
        mu_neutral=4.0,
        clausing_scale=baffle_params["neutral_clausing_scale"],
    )
    baffle_vbar = neutral_thermal_speed(
        baffle_params["Tn_K"], 4.0
    )
    baffle_orifice = (
        0.25
        * baffle_vbar
        * np.pi
        * 40.0**2
        * baffle_params["neutral_clausing_scale"]
    )
    expected_single = 1.0 / (
        1.0 / base_single[baffle_interior] + 1.0 / baffle_orifice
    )
    assert np.isclose(baffle_single[baffle_interior], expected_single)
    assert np.allclose(
        np.delete(baffle_single, baffle_interior),
        np.delete(base_single, baffle_interior),
    )

    base_col, base_ann = two_zone_knudsen_coefficients(
        resolved_geom,
        Tn_K=resolved_params["Tn_K"],
        mu_neutral=4.0,
        clausing_scale=resolved_params["neutral_clausing_scale"],
    )
    baffle_col, baffle_ann = two_zone_knudsen_coefficients(
        baffle_geom,
        Tn_K=baffle_params["Tn_K"],
        mu_neutral=4.0,
        clausing_scale=baffle_params["neutral_clausing_scale"],
    )
    assert np.array_equal(baffle_col, base_col)
    open_annulus = np.pi * (40.0**2 - resolved_params["Rp"] ** 2)
    annulus_orifice = (
        0.25
        * baffle_vbar
        * open_annulus
        * baffle_params["neutral_clausing_scale"]
    )
    expected_annulus = 1.0 / (
        1.0 / base_ann[baffle_interior] + 1.0 / annulus_orifice
    )
    assert np.isclose(baffle_ann[baffle_interior], expected_annulus)
    assert np.allclose(
        np.delete(baffle_ann, baffle_interior),
        np.delete(base_ann, baffle_interior),
    )

    for bad_params, bad_flags, expected in (
        (
            baffle_params,
            resolved_flags,
            "require the default-off",
        ),
        (
            resolved_params,
            baffle_flags,
            "requires positions and clear radii",
        ),
        (
            {
                **resolved_params,
                "neutral_baffle_positions_cm": [150.0],
                "neutral_baffle_clear_radii_cm": [10.0],
            },
            baffle_flags,
            "Rp <= R_clear < Rm",
        ),
        (
            {
                **resolved_params,
                "neutral_baffle_positions_cm": [150.0, 981.25],
                "neutral_baffle_clear_radii_cm": [40.0],
            },
            baffle_flags,
            "equal lengths",
        ),
    ):
        try:
            LAPDSim1D(bad_params, bad_flags)
        except ValueError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError("invalid neutral-baffle configuration constructed")

    # Fixed-cell-size source region (default-off ``source_fixed_grid``). Without
    # it, nx uniform column cells span anode face to collector start, so a
    # refinement study moves every near-source cell edge -- including the puff
    # cell, whose centre anchors the default cosine puff profile. With it on the
    # column from the anode face (50 cm) to source_region_length_cm is meshed at
    # exactly source_region_dz_cm regardless of nx, and the puff role follows
    # gas_puff_z_cm.
    #
    # (d) The production default takes NO new branch: flag off, both keys None,
    # and the spec helper returns None for the resolved default config.
    assert not resolved_flags["source_fixed_grid"]
    assert resolved_params["source_region_length_cm"] is None
    assert resolved_params["source_region_dz_cm"] is None
    assert (
        _source_fixed_grid_spec(
            resolved_params,
            resolved_flags,
            gap_length=resolved_params["cathode_anode_gap_cm"],
            total_length=resolved_params["Lm"],
            collector_length=resolved_params["collector_length_cm"],
            twin=False,
        )
        is None
    )

    srcgrid_flags = {**resolved_flags, "source_fixed_grid": True}

    def _srcgrid_params(nx):
        params = dict(resolved_params)
        params.update(
            {
                "nx": nx,
                "source_region_length_cm": 100.0,
                "source_region_dz_cm": 10.0,
                "gas_puff_z_cm": 60.0,
            }
        )
        return params

    def _srcgrid_geometry(nx):
        return (
            LAPDSim1D(_srcgrid_params(nx), srcgrid_flags)
            .get_initial_snapshot()
            .geometry
        )

    # (a) Feature-on mesh at the production intent: 50 cm gap, a 50 cm source
    # region in five 10 cm cells, puff pipe at 60 cm.
    srcgrid_geom = _srcgrid_geometry(60)
    (srcgrid_cathode_face,) = srcgrid_geom.cathode_face_indices
    (srcgrid_anode_face,) = srcgrid_geom.anode_face_indices
    srcgrid_n_fixed = 5
    srcgrid_region_end_face = srcgrid_anode_face + srcgrid_n_fixed
    # Anode face and region end land EXACTLY on cell edges (not merely close).
    assert srcgrid_geom.z_edges_cm[srcgrid_cathode_face] == 0.0
    assert srcgrid_geom.z_edges_cm[srcgrid_anode_face] == 50.0
    assert srcgrid_geom.z_edges_cm[srcgrid_region_end_face] == 100.0
    assert np.all(
        srcgrid_geom.length_cm[srcgrid_anode_face:srcgrid_region_end_face] == 10.0
    )
    # nx meshes only the far column, from the region end to the collector.
    assert srcgrid_geom.cells == resolved_geom.cells + srcgrid_n_fixed
    srcgrid_puff, srcgrid_puff_twin = puff_cell_indices(srcgrid_geom)
    assert srcgrid_puff == srcgrid_puff_twin
    # The puff role went to the fixed-region cell CONTAINING 60 cm, not the
    # first column cell -- which is now plain column.
    assert srcgrid_puff == srcgrid_anode_face + 1
    assert srcgrid_geom.cell_role[srcgrid_anode_face] == "column"
    assert srcgrid_geom.z_edges_cm[srcgrid_puff] <= 60.0
    assert srcgrid_geom.z_edges_cm[srcgrid_puff + 1] > 60.0
    assert list(srcgrid_geom.cell_role).count("puff") == 1

    # (b) nx-invariance: doubling nx must not move a single edge at or inside
    # the source region, and must not move the puff cell.
    srcgrid_geom_2x = _srcgrid_geometry(120)
    srcgrid_puff_2x, _ = puff_cell_indices(srcgrid_geom_2x)
    assert srcgrid_puff_2x == srcgrid_puff
    for _edges in (srcgrid_geom.z_edges_cm, srcgrid_geom_2x.z_edges_cm):
        assert _edges[srcgrid_region_end_face + 1] > 100.0
    srcgrid_inside = srcgrid_geom.z_edges_cm[
        srcgrid_cathode_face : srcgrid_region_end_face + 1
    ]
    srcgrid_inside_2x = srcgrid_geom_2x.z_edges_cm[
        srcgrid_cathode_face : srcgrid_region_end_face + 1
    ]
    # Exact float equality, not allclose: this is the whole point of the mode.
    assert np.array_equal(srcgrid_inside, srcgrid_inside_2x)
    assert np.array_equal(
        srcgrid_geom.z_edges_cm[
            (srcgrid_geom.z_edges_cm >= 0.0) & (srcgrid_geom.z_edges_cm <= 100.0)
        ],
        srcgrid_geom_2x.z_edges_cm[
            (srcgrid_geom_2x.z_edges_cm >= 0.0)
            & (srcgrid_geom_2x.z_edges_cm <= 100.0)
        ],
    )
    assert (
        srcgrid_geom.z_edges_cm[srcgrid_puff]
        == srcgrid_geom_2x.z_edges_cm[srcgrid_puff_2x]
    )
    assert (
        srcgrid_geom.z_edges_cm[srcgrid_puff + 1]
        == srcgrid_geom_2x.z_edges_cm[srcgrid_puff_2x + 1]
    )

    # (c) Every misconfiguration raises loudly at construction; none falls back.
    srcgrid_twin_params = _srcgrid_params(60)
    srcgrid_twin_params["collector_length_cm"] = 100.0
    for bad_params, bad_flags, expected in (
        (
            {**_srcgrid_params(60), "source_region_length_cm": None},
            srcgrid_flags,
            "requires all source region parameters",
        ),
        (
            {**_srcgrid_params(60), "source_region_dz_cm": None},
            srcgrid_flags,
            "requires all source region parameters",
        ),
        (
            _srcgrid_params(60),
            resolved_flags,
            "require the default-off",
        ),
        (
            {**resolved_params, "source_region_dz_cm": 10.0},
            resolved_flags,
            "require the default-off",
        ),
        (
            {**_srcgrid_params(60), "source_region_length_cm": 50.0},
            srcgrid_flags,
            "strictly beyond the anode face",
        ),
        (
            {**_srcgrid_params(60), "source_region_length_cm": 1900.0},
            srcgrid_flags,
            "strictly before the collector",
        ),
        (
            {**_srcgrid_params(60), "source_region_dz_cm": 7.0},
            srcgrid_flags,
            "integer number of",
        ),
        (
            {**_srcgrid_params(60), "gas_puff_z_cm": None},
            srcgrid_flags,
            "requires an explicit gas_puff_z_cm",
        ),
        (
            {**_srcgrid_params(60), "gas_puff_z_cm": 40.0},
            srcgrid_flags,
            "gas_puff_z_cm must lie in",
        ),
        (
            {**_srcgrid_params(60), "gas_puff_z_cm": 100.0},
            srcgrid_flags,
            "gas_puff_z_cm must lie in",
        ),
        (
            srcgrid_twin_params,
            {**srcgrid_flags, "TwinCathode": True},
            "single-cathode layout",
        ),
    ):
        try:
            LAPDSim1D(bad_params, bad_flags)
        except ValueError as exc:
            assert expected in str(exc), (expected, str(exc))
        else:
            raise AssertionError(
                "invalid source_fixed_grid configuration constructed"
            )

    expansion_params = dict(resolved_params)
    expansion_params.update(
        {
            "Lm": 2125.85,
            "collector_length_cm": 150.0,
            "end_expansion_cells": 10,
            "end_expansion_machine_radius_cm": 100.0,
            "end_expansion_plasma_radius_cm": 50.0,
        }
    )
    expansion_flags = dict(resolved_flags)
    expansion_flags.update(
        {
            "end_expansion_geometry": True,
            "cathode_coupling": False,
            "neutral_prebreakdown": False,
            "implicit_heat_conduction": False,
        }
    )
    expansion_params["phase_transition_mode"] = "scheduled"
    expansion_params["tau_prebreakdown"] = 0.0
    expansion_params["tau_breakdown"] = 0.0
    expansion_sim = LAPDSim1D(expansion_params, expansion_flags)
    expansion_geom = expansion_sim.get_initial_snapshot().geometry
    end_cells = np.flatnonzero(
        np.isin(expansion_geom.cell_role, np.asarray(["end", "collector"]))
    )
    assert end_cells.size == 10
    assert np.array_equal(end_cells, np.arange(expansion_geom.cells - 10, expansion_geom.cells))
    assert list(expansion_geom.cell_role[-10:-1]) == ["end"] * 9
    assert expansion_geom.cell_role[-1] == "collector"
    assert expansion_geom.cells == resolved_geom.cells + 9
    assert np.allclose(expansion_geom.length_cm[end_cells], 15.0)
    start_face = int(end_cells[0])
    assert np.isclose(expansion_geom.z_edges_cm[start_face], 1975.85)
    assert np.isclose(expansion_geom.z_edges_cm[-1], 2125.85)
    assert np.allclose(expansion_geom.Rm_cm[end_cells], 100.0)
    assert np.allclose(
        expansion_geom.neutral_area_cm2[end_cells], np.pi * 100.0**2
    )
    # The abrupt vessel entrance retains the upstream Rm=50 cm throat.
    assert np.isclose(
        expansion_geom.neutral_face_area_cm2[start_face], np.pi * 50.0**2
    )
    # The flux-tube area starts at the column Rp, ends at the declared Rp=50 cm,
    # and widens monotonically across the end region.
    end_face_area = expansion_geom.plasma_face_area_cm2[start_face:]
    assert np.isclose(end_face_area[0], np.pi * expansion_params["Rp"] ** 2)
    assert np.isclose(end_face_area[-1], np.pi * 50.0**2)
    assert np.all(np.diff(end_face_area) > 0.0)
    assert np.all(expansion_geom.Rp_cm[end_cells] < expansion_geom.Rm_cm[end_cells])

    vessel_params = dict(expansion_params)
    vessel_params["end_expansion_plasma_radius_cm"] = vessel_params["Rp"]
    vessel_geom = LAPDSim1D(
        vessel_params, expansion_flags
    ).get_initial_snapshot().geometry
    assert np.allclose(vessel_geom.plasma_area_cm2, np.pi * vessel_params["Rp"] ** 2)
    assert np.allclose(
        vessel_geom.plasma_face_area_cm2, np.pi * vessel_params["Rp"] ** 2
    )

    for bad_params, bad_flags, expected in (
        (
            {**resolved_params, "end_expansion_cells": 10},
            resolved_flags,
            "require the default-off",
        ),
        (
            resolved_params,
            {**resolved_flags, "end_expansion_geometry": True},
            "requires all",
        ),
        (
            {
                **expansion_params,
                "end_expansion_plasma_radius_cm": 101.0,
            },
            expansion_flags,
            "Rp <= Rp_end <= Rm_end",
        ),
        (
            expansion_params,
            {
                **expansion_flags,
                "TwinCathode": True,
            },
            "single-cathode",
        ),
    ):
        try:
            LAPDSim1D(bad_params, bad_flags)
        except ValueError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError("invalid expanded-end configuration constructed")

    # Well-balancedness of the variable-area flux tube: for a uniform stationary
    # plasma the quasi-1D p*dA/dz geometric source cancels the area-weighted
    # pressure flux bit-for-bit -- but this property applies only across the
    # INTERIOR expansion cells (role "end"). The terminating "collector" cell is
    # a plasma-OPEN boundary: under characteristic_boundary (R3.1, the production
    # default) it carries a Bohm outflow (ghost u_g = c_s) whose flux is supplied
    # by characteristic_boundary_rhs (a term not summed here), so a uniform
    # stationary state is deliberately NOT its equilibrium -- the plasma flows
    # out. Under the legacy reflecting wall the collector cancels like the
    # interior. (hyperbolic_energy_consistent and hyperbolic_wave_speed have no
    # effect on this state: at u=0 with no gradients the KEP convective term and
    # the Rusanov dissipation both vanish at every interior face, so only the
    # collector ghost -- gated by characteristic_boundary -- can be nonzero.)
    uniform_expansion = conservative_from_primitives(
        n=np.full(expansion_geom.cells, 1.0e12),
        nn=np.full(expansion_geom.cells, 1.0e12),
        u=np.zeros(expansion_geom.cells),
        Te=np.full(expansion_geom.cells, 2.0),
        Ti=np.full(expansion_geom.cells, 1.0),
        ion_mass_g=expansion_sim.ion_mass_g,
    )
    expansion_advective = expansion_sim.plasma_flux_rhs_terms(
        state=uniform_expansion, include_front=False
    )["plasma_advective_flux"]
    expansion_geometric = expansion_sim.flux_tube_geometry_rhs(
        state=uniform_expansion
    )
    interior_expansion_cells = np.flatnonzero(expansion_geom.cell_role == "end")
    collector_cells = np.flatnonzero(expansion_geom.cell_role == "collector")
    assert interior_expansion_cells.size == 9
    assert collector_cells.size == 1
    expansion_momentum_residual = expansion_advective.M + expansion_geometric.M
    # Interior variable-area cells: exact cancellation (the load-bearing
    # well-balancedness of the KEP pressure flux against the flux-tube source).
    assert np.array_equal(
        expansion_momentum_residual[interior_expansion_cells],
        np.zeros(interior_expansion_cells.size),
    )
    # Terminating collector cell: an open Bohm outflow under the characteristic
    # boundary (directed toward +z, so a net positive momentum residual), or an
    # exact wall cancellation under the legacy reflecting boundary.
    if expansion_sim._characteristic_boundary:
        assert np.all(expansion_momentum_residual[collector_cells] > 0.0)
    else:
        assert np.array_equal(
            expansion_momentum_residual[collector_cells],
            np.zeros(collector_cells.size),
        )
    assert np.allclose(expansion_geometric.n, 0.0)
    assert np.allclose(expansion_geometric.Ee, 0.0)
    assert np.allclose(expansion_geometric.Ei, 0.0)
    expansion_attempt = expansion_sim._attempt_step(
        dt=1.0e-9, operator_split=False
    )
    assert np.all(np.isfinite(expansion_attempt.y))
    assert expansion_attempt.y.shape == expansion_sim.get_initial_snapshot().y.shape

    # Twin cathode mirrors the source end (plan §11 decision 4): its cathode
    # surface sits at z = Lm, with that plenum beyond it.
    twin_resolved_flags = dict(resolved_flags)
    twin_resolved_flags["TwinCathode"] = True
    twin_resolved_flags["cathode_coupling"] = False
    twin_resolved_geom = LAPDSim1D(
        resolved_params, twin_resolved_flags
    ).get_initial_snapshot().geometry
    assert list(twin_resolved_geom.cell_role[:2]) == ["plenum", "cathode"]
    assert list(twin_resolved_geom.cell_role[-2:]) == ["cathode", "plenum"]
    assert "collector" not in set(twin_resolved_geom.cell_role)
    assert len(twin_resolved_geom.cathode_face_indices) == 2
    assert len(twin_resolved_geom.anode_face_indices) == 2
    twin_near, twin_far = twin_resolved_geom.cathode_face_indices
    assert np.isclose(twin_resolved_geom.z_edges_cm[twin_near], 0.0)
    assert np.isclose(twin_resolved_geom.z_edges_cm[twin_far], resolved_params["Lm"])
    for face in twin_resolved_geom.cathode_face_indices:
        assert not twin_resolved_geom.plasma_open[face]
    assert len(cathode_adjacent_cells(twin_resolved_geom)) == 2
    # Twin puffs at both ends (legacy twin puffs at [0] and [-1]).
    assert list(twin_resolved_geom.cell_role).count("puff") == 2

    # Role anchors place puffs and pumps on resolved machine regions.
    resolved_puff, _ = puff_cell_indices(resolved_geom)
    resolved_pump_left, resolved_pump_right = pump_cell_indices(resolved_geom)
    assert resolved_geom.cell_role[resolved_puff] == "puff"
    assert resolved_puff not in (0, resolved_geom.cells - 1)
    assert resolved_geom.cell_role[resolved_pump_left] == "plenum"
    assert resolved_geom.cell_role[resolved_pump_right] == "collector"
    assert is_plenum_cell(resolved_geom, resolved_pump_left)
    assert not is_plenum_cell(resolved_geom, resolved_pump_right)

    # M2: the effective pump speed is a series conductance; no elbow (None or
    # non-positive) returns the raw speed unchanged -- the legacy limit.
    assert _effective_pump_speed(2000.0, None) == 2000.0
    assert _effective_pump_speed(2000.0, 0.0) == 2000.0
    assert np.isclose(_effective_pump_speed(2000.0, 2000.0), 1000.0)
    assert _effective_pump_speed(2000.0, 1e12) < 2000.0

    # M2: the cathode-structure obstruction is a real annular cell (decision 1),
    # present only when Lcs > 0 so Lcs = 0 stays the legacy limit.
    assert "obstruction" not in set(resolved_geom.cell_role)
    obstruction_params = dict(resolved_params)
    obstruction_params["Lcs"] = 25.0
    obstruction_params["Rcs"] = 25.0
    obstruction_geom = LAPDSim1D(
        obstruction_params, resolved_flags
    ).get_initial_snapshot().geometry
    assert list(obstruction_geom.cell_role[:3]) == [
        "plenum",
        "obstruction",
        "cathode",
    ]
    assert obstruction_geom.cells == resolved_geom.cells + 1
    obstruction_cell = 1
    assert np.isclose(obstruction_geom.length_cm[obstruction_cell], 25.0)
    # The duct sits behind the cathode surface, so it occupies negative z and
    # pushes the mesh further back without changing where the cathode sits.
    (obstruction_cathode_face,) = obstruction_geom.cathode_face_indices
    assert np.isclose(obstruction_geom.z_edges_cm[obstruction_cathode_face], 0.0)
    assert np.isclose(obstruction_geom.z_edges_cm[-1], obstruction_params["Lm"])
    assert np.isclose(
        obstruction_geom.z_edges_cm[0],
        -(obstruction_params["plenum_length_cm"] + 25.0),
    )
    # Annular duct: open area and hydraulic radius reduce independently (§3).
    assert np.isclose(
        obstruction_geom.neutral_area_cm2[obstruction_cell],
        np.pi * (obstruction_params["Rm"] ** 2 - 25.0**2),
    )
    assert np.isclose(
        obstruction_geom.neutral_hydraulic_radius_cm[obstruction_cell],
        obstruction_params["Rm"] - 25.0,
    )
    # The plasma wall moves to the obstruction<->cathode face: everything behind
    # the cathode is plasma-dead.
    assert not obstruction_geom.plasma_open[2]
    assert obstruction_geom.heat_transmission[2] == 0.0
    assert obstruction_geom.plasma_open[1]  # plenum<->obstruction: both dead
    # Restricting aperture: the face conductance sees the annulus, not the mean.
    assert np.isclose(
        obstruction_geom.neutral_face_area_cm2[obstruction_cell],
        obstruction_geom.neutral_area_cm2[obstruction_cell],
    )
    obstruction_coeff = neutral_exchange_coefficients(
        geometry=obstruction_geom,
        model="knudsen",
        constant_coeff_cm3_s=obstruction_params["neutral_exchange_coeff_cm3_s"],
        Tn_K=obstruction_params["Tn_K"],
        mu_neutral=4,
        clausing_scale=obstruction_params["neutral_clausing_scale"],
    )
    assert np.all(np.isfinite(obstruction_coeff))
    assert np.all(obstruction_coeff > 0.0)

    # M2: support rods block plenum volume only, leaving the hydraulic radius.
    rod_params = dict(resolved_params)
    rod_params["Rsup"] = 10.0
    rod_geom = LAPDSim1D(rod_params, resolved_flags).get_initial_snapshot().geometry
    rod_plenum = int(np.flatnonzero(np.asarray(rod_geom.cell_role) == "plenum")[0])
    assert np.isclose(
        rod_geom.neutral_area_cm2[rod_plenum],
        np.pi * (rod_params["Rm"] ** 2 - 10.0**2),
    )
    assert np.isclose(rod_geom.neutral_hydraulic_radius_cm[rod_plenum], rod_params["Rm"])

    # M3: heat and neutrals are throttled by the transparency (1-eta), but the
    # advective plasma face stays OPEN -- the anode removes plasma through the
    # Bohm sheath flux at its wires, and shrinking the face too would remove the
    # same particles twice (§5). The cathode surface blocks everything.
    transparency = 1.0 - resolved_params["eta"]
    assert resolved_geom.plasma_transmission[anode_face] == 1.0
    assert np.isclose(resolved_geom.heat_transmission[anode_face], transparency)
    assert np.isclose(
        resolved_geom.neutral_face_area_cm2[anode_face],
        transparency * np.pi * resolved_params["Rm"] ** 2,
    )
    assert resolved_geom.plasma_transmission[cathode_face] == 0.0
    assert resolved_geom.heat_transmission[cathode_face] == 0.0
    # Every other interior face is fully open.
    open_faces = [
        f
        for f in range(1, resolved_geom.cells)
        if f not in (cathode_face, anode_face)
    ]
    assert np.allclose(resolved_geom.plasma_transmission[open_faces], 1.0)

    # M3: eta = 0 is the legacy limit -- a fully transparent anode.
    transparent_params = dict(resolved_params)
    transparent_params["eta"] = 0.0
    transparent_geom = LAPDSim1D(
        transparent_params, resolved_flags
    ).get_initial_snapshot().geometry
    assert transparent_geom.heat_transmission[anode_face] == 1.0
    assert np.isclose(
        transparent_geom.neutral_face_area_cm2[anode_face],
        np.pi * transparent_params["Rm"] ** 2,
    )
    # The advective-block knob dials the (1-eta) face reduction back in for a
    # sensitivity study; it is 0 by default so the face stays open.
    blocked_params = dict(resolved_params)
    blocked_params["b_anode_advective_block"] = 1.0
    blocked_geom = LAPDSim1D(
        blocked_params, resolved_flags
    ).get_initial_snapshot().geometry
    assert np.isclose(blocked_geom.plasma_transmission[anode_face], transparency)

    # M3: the anode collects plasma at the Bohm sheath flux on BOTH mesh faces,
    # each sampling its own side, independent of the bulk drift.
    resolved_sim = LAPDSim1D(resolved_params, resolved_flags)
    flowing_state = conservative_from_primitives(
        n=np.full(resolved_geom.cells, 1.0e12),
        nn=np.full(resolved_geom.cells, 1.0e12),
        u=np.full(resolved_geom.cells, 1.0e5),
        Te=np.full(resolved_geom.cells, 2.0),
        Ti=np.full(resolved_geom.cells, 1.0),
        ion_mass_g=resolved_sim.ion_mass_g,
    )
    collected = resolved_sim.anode_collection_rhs(state=flowing_state)
    for side in (anode_face - 1, anode_face):
        assert collected.n[side] < 0.0
        assert collected.M[side] < 0.0  # absorbed by the structure, not thermalized
        assert collected.Ee[side] < 0.0
        assert collected.Ei[side] < 0.0
        assert collected.nn[side] > 0.0  # neutral born on the side it came from
    # Only the two flanking cells are touched.
    untouched = [
        c for c in range(resolved_geom.cells) if c not in (anode_face - 1, anode_face)
    ]
    assert np.allclose(collected.n[untouched], 0.0)
    collected_scale = np.sum(
        np.abs(collected.n * resolved_geom.plasma_volume_cm3)
        + np.abs(collected.nn * resolved_geom.neutral_volume_cm3)
    )
    assert collected_scale > 0.0
    assert np.isclose(
        particle_inventory_rate(collected, resolved_geom),
        0.0,
        atol=1e-12 * collected_scale,
    )
    # Bohm collection is set by the sheath, not the drift: it is unchanged when
    # the bulk flow is switched off, which the old directed-flux model got wrong.
    still_state = conservative_from_primitives(
        n=np.full(resolved_geom.cells, 1.0e12),
        nn=np.full(resolved_geom.cells, 1.0e12),
        u=np.zeros(resolved_geom.cells),
        Te=np.full(resolved_geom.cells, 2.0),
        Ti=np.full(resolved_geom.cells, 1.0),
        ion_mass_g=resolved_sim.ion_mass_g,
    )
    still_collected = resolved_sim.anode_collection_rhs(state=still_state)
    assert np.allclose(still_collected.n, collected.n)
    assert still_collected.n[anode_face] < 0.0
    # A transparent anode collects nothing.
    transparent_sim = LAPDSim1D(transparent_params, resolved_flags)
    assert np.allclose(
        pack_state(transparent_sim.anode_collection_rhs(state=flowing_state)), 0.0
    )

    # M4a: the cathode surface and collector are absorbing Bohm faces.
    assert resolved_geom.plasma_absorbing[cathode_face]
    assert resolved_geom.plasma_absorbing[-1]  # collector outer face
    assert not resolved_geom.plasma_absorbing[anode_face]
    # Absorbing faces are still closed: nothing passes through to the far side.
    assert not resolved_geom.plasma_open[cathode_face]
    # A twin machine ends in plenums, whose closed back walls see no plasma.
    assert not twin_resolved_geom.plasma_absorbing[0]
    assert not twin_resolved_geom.plasma_absorbing[-1]
    for face in twin_resolved_geom.cathode_face_indices:
        assert twin_resolved_geom.plasma_absorbing[face]

    # The absorbing face drains its live cell and returns the plasma as gas
    # there, conserving particles.
    absorbed = resolved_sim.boundary_absorption_rhs(state=flowing_state)
    assert absorbed.n[cathode_face] < 0.0  # cathode cell drains to the surface
    assert absorbed.nn[cathode_face] > 0.0
    assert absorbed.n[-1] < 0.0  # collector drains too
    # Momentum leaves at c_s directed INTO each surface: negative (toward -z) at
    # the cathode, positive (toward +z) at the collector. This is what makes the
    # sonic condition drive flow toward the wall rather than just delete plasma.
    assert absorbed.M[cathode_face] > 0.0
    assert absorbed.M[-1] < 0.0
    # Plasma-dead cells are untouched: an interior absorbing face must not hand
    # anything to the plenum behind it.
    assert np.allclose(absorbed.n[0], 0.0)
    assert np.allclose(absorbed.M[0], 0.0)
    absorbed_scale = np.sum(
        np.abs(absorbed.n * resolved_geom.plasma_volume_cm3)
        + np.abs(absorbed.nn * resolved_geom.neutral_volume_cm3)
    )
    assert absorbed_scale > 0.0
    assert np.isclose(
        particle_inventory_rate(absorbed, resolved_geom),
        0.0,
        atol=1e-12 * absorbed_scale,
    )

    # M4b: the cathode circuit samples the plasma against the cathode surface, not
    # cell [0] -- which in resolved geometry is the plasma-dead plenum, and would
    # drive the circuit off floor values.
    assert cathode_sample_indices(geom) == cathode_sample_indices(resolved_geom)
    resolved_source_index, resolved_end_index = cathode_sample_indices(resolved_geom)
    assert resolved_source_index == cathode_face
    assert resolved_geom.cell_role[resolved_source_index] == "cathode"
    assert resolved_end_index == resolved_geom.cells - 1
    assert resolved_geom.cell_role[resolved_end_index] == "collector"
    twin_source_index, twin_end_index = cathode_sample_indices(twin_resolved_geom)
    assert twin_resolved_geom.cell_role[twin_source_index] == "cathode"
    assert twin_resolved_geom.cell_role[twin_end_index] == "cathode"
    assert twin_source_index != twin_end_index

    # M4b: the beam launches from the cathode cell and never deposits behind it.
    assert beam_launch(resolved_geom, end=0) == (cathode_face, 1)
    resolved_beam_weights = beam_absorption_weights(
        length_cm=resolved_geom.length_cm,
        l_b_profile=np.full(resolved_geom.cells, 500.0),
        cathode_index=cathode_face,
        direction=1,
    )
    assert np.allclose(resolved_beam_weights[:cathode_face], 0.0)  # plenum untouched
    assert resolved_beam_weights[cathode_face] > 0.0
    assert np.all(resolved_beam_weights >= 0.0)
    assert resolved_beam_weights.sum() <= 1.0 + 1e-12

    # M5: the circuit's anode current is the same Bohm collection the fluid
    # removes (§7), not `2*eta*I_i` scaled off the cathode cell.
    resolved_cathode_flags = dict(resolved_flags)
    resolved_cathode_flags["cathode_coupling"] = True
    resolved_cathode_flags["neutral_prebreakdown"] = False
    # The anode current == fluid Bohm collection identity holds only without
    # electrode sample smoothing, which EMA-smooths the anode-flank (n, Te) the
    # solve reads so I_i_a decouples from the raw-state fluid collection. The
    # smoothing is a separate production feature (tested in its own block); pin
    # it off here to isolate the M5 split.
    m5_cathode_params = dict(resolved_params, cathode_sample_smoothing=None)
    m5_sim = LAPDSim1D(m5_cathode_params, resolved_cathode_flags)
    m5_geom = m5_sim.get_initial_snapshot().geometry
    m5_anode_face = int(m5_geom.anode_face_indices[0])
    m5_n = np.full(m5_geom.cells, 1.0e12)
    m5_n[:m5_anode_face] = 4.0e12
    # Deplete the cathode cell: this is the regime the split exists for, where
    # scaling the anode current off the cathode is badly wrong.
    m5_n[cathode_face] = 1.0e11
    m5_Te = np.full(m5_geom.cells, 3.0)
    m5_Te[:m5_anode_face] = 6.0
    m5_state = conservative_from_primitives(
        n=m5_n,
        nn=np.full(m5_geom.cells, 1.0e13),
        u=np.zeros(m5_geom.cells),
        Te=m5_Te,
        Ti=np.full(m5_geom.cells, 1.0),
        ion_mass_g=m5_sim.ion_mass_g,
    )
    m5_sim._set_state_vector(pack_state(m5_state))
    m5_result = m5_sim.solve_cathode_boundary(state=m5_state).beam_result.result
    m5_fluid_A = -float(
        np.sum(
            m5_sim.anode_collection_rhs(state=m5_state).n
            * m5_geom.plasma_volume_cm3
        )
    ) * qe_SI
    assert np.isclose(m5_result.I_i_a, m5_fluid_A, rtol=1e-12)
    # The gap is hotter and denser than the column here, so the historical
    # cathode-scaled estimate is far off -- which is the point of the split.
    assert m5_result.I_i_a > 10.0 * (2.0 * resolved_params["eta"] * m5_result.I_i)

    # --- Resolved gap resistance (cathode_Rp_model="resolved_gap",
    # CATHODE_IDRIVEN_PLAN.md M1): the historical R_p spreads the hot
    # cathode-adjacent Spitzer sample over the whole 50 cm gap; the resolved
    # model integrates dz/(sigma_par(Te)*A) over the gap profile and feeds
    # it to the unmodified solver through an effective DeviceConfig.R_cath.
    import warnings as _warnings

    from cablp.solvers._sim1d.core.geometry import gap_cell_indices
    from cablp.solvers._sim1d.physics.cathode import spitzer_sigma_par_ohm_cm

    rgap_params = dict(resolved_params)
    rgap_params["Rp"] = rgap_params["R_cath"]  # channel area == disc area
    # Isolate the resolved-gap R_p model from the electrode sample smoothing
    # (production default) so the resolved-vs-sample solves are comparable.
    rgap_params["cathode_sample_smoothing"] = None
    # The resolved gap spans exactly the solver's L_cath, so a uniform gap
    # must reduce the integral to the single-sample formula.
    assert np.isclose(
        rgap_params["cathode_anode_gap_cm"], rgap_params["L_cath"]
    )
    rgap_resolved_params = dict(rgap_params, cathode_Rp_model="resolved_gap")
    sim_rgap_sample = LAPDSim1D(rgap_params, resolved_cathode_flags)
    sim_rgap = LAPDSim1D(rgap_resolved_params, resolved_cathode_flags)
    rgap_geom = sim_rgap.get_initial_snapshot().geometry
    rgap_gap = np.asarray(gap_cell_indices(rgap_geom), dtype=int)
    assert spitzer_sigma_par_ohm_cm(4.0) == 14.6 * 4.0**1.5

    def _rgap_state(Te):
        return conservative_from_primitives(
            n=np.full(rgap_geom.cells, 4.0e12),
            nn=np.full(rgap_geom.cells, 1.0e13),
            u=np.zeros(rgap_geom.cells),
            Te=Te,
            Ti=np.full(rgap_geom.cells, 1.0),
            ion_mass_g=sim_rgap.ion_mass_g,
        )

    uni_state = _rgap_state(np.full(rgap_geom.cells, 6.0))
    r_uni_s = sim_rgap_sample.solve_cathode_boundary(
        state=uni_state, update_cache=False
    ).beam_result.result
    r_uni_solve = sim_rgap.solve_cathode_boundary(
        state=uni_state, update_cache=False
    )
    r_uni_r = r_uni_solve.beam_result.result
    assert r_uni_solve.metadata["cathode_Rp_model"] == "resolved_gap"
    assert np.isclose(
        r_uni_solve.metadata["R_p_gap_ohm"], r_uni_s.R_p, rtol=1e-12
    )
    for rgap_attr in ("R_p", "I_tot", "phi_c", "phi_a", "V_b", "I_eth_star"):
        assert np.isclose(
            getattr(r_uni_r, rgap_attr),
            getattr(r_uni_s, rgap_attr),
            rtol=1e-9,
        ), (rgap_attr, getattr(r_uni_r, rgap_attr), getattr(r_uni_s, rgap_attr))

    # Cold gap: heat only the sampled cathode-adjacent cell. The sample
    # model spreads that hot conductivity over the whole gap (eta_Spitzer ~
    # Te^-3/2 underestimates the colder remainder); the resolved integral
    # must be larger, with a larger gap voltage drop and no more current.
    rgap_cold_Te = np.full(rgap_geom.cells, 3.0)
    rgap_cold_Te[rgap_gap[0]] = 12.0
    cold_state = _rgap_state(rgap_cold_Te)
    # Drive a nonzero loop current so the gap actually carries current: V_p is
    # then the meaningful ohmic drop I*R_p (without a driven current the cold
    # gap floats at I_tot~0, V_p~0, and the resolved-vs-sample V_p ordering is
    # roundoff).
    sim_rgap_sample._circuit_I_loop = 2000.0
    sim_rgap._circuit_I_loop = 2000.0
    r_cold_s = sim_rgap_sample.solve_cathode_boundary(
        state=cold_state, update_cache=False
    ).beam_result.result
    r_cold_r = sim_rgap.solve_cathode_boundary(
        state=cold_state, update_cache=False
    ).beam_result.result
    # 1/5 of the gap at 12 eV, 4/5 at 3 eV: 0.2 + 0.8*(12/3)^1.5 = 6.6x.
    assert np.isclose(r_cold_r.R_p, 6.6 * r_cold_s.R_p, rtol=1e-9)
    # The resolved integral's larger R_p yields a larger ohmic gap drop at the
    # same driven current.
    assert r_cold_r.V_p > r_cold_s.V_p

    # TwinCathode shares one DeviceConfig, so resolved_gap must refuse it
    # at construction; unknown model strings fail the same way.
    for rgap_bad_params, rgap_bad_flags in (
        (rgap_resolved_params, dict(resolved_cathode_flags, TwinCathode=True)),
        (dict(rgap_params, cathode_Rp_model="bogus"), resolved_cathode_flags),
    ):
        try:
            LAPDSim1D(rgap_bad_params, rgap_bad_flags)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"expected ValueError for cathode_Rp_model config "
                f"{rgap_bad_params.get('cathode_Rp_model')!r}"
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
            model="knudsen",
            constant_coeff_cm3_s=knudsen_params["neutral_exchange_coeff_cm3_s"],
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
    assert cathode_boundary.source.index == cathode_face
    assert cathode_boundary.source.role == "cathode"
    assert cathode_boundary.end.index == geom.cells - 1
    assert cathode_boundary.end.role == "collector"
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
    # This block checks the temporal puff SCHEDULE (on/off per phase); pin the
    # single-cell axial profile so the puff-cell amount is the full puff_rate
    # (the production default cosine_pipe distributes it -- tested separately).
    neutral_phase_params["gas_puff_profile"] = "cell"
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
    assert neutral_puff_source.nn[neutral_puff_cell] > neutral_off_source.nn[neutral_puff_cell]
    assert np.isclose(
        neutral_puff_source.nn[neutral_puff_cell]
        - neutral_off_source.nn[neutral_puff_cell],
        puff_rate(
            neutral_phase_params["S_gp"],
            neutral_phase_params["gas_puff_valves"],
            neutral_geom.neutral_volume_cm3[neutral_puff_cell],
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
    # This block exercises the cathode boundary + beam-ionization bookkeeping in
    # the beer_lambert regime it was written for (excitation off by default);
    # the CSDA production beam + manifold excitation are covered by the R4
    # blocks. beer_lambert is a live A/B arm.
    cathode_bl_params = dict(params, beam_deposition_model="beer_lambert")
    cathode_sim = LAPDSim1D(cathode_bl_params, cathode_flags)
    cathode_sim._circuit_I_loop = 3000.0
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
    cathode_loss_terms = cathode_sim.cathode_source_terms(cathode_solve=cathode_solve)
    assert cathode_loss_terms.enabled
    assert np.all(np.isfinite(pack_state(cathode_loss_terms.rhs)))
    afterglow_cathode_loss_terms = cathode_sim.cathode_source_terms(
        cathode_solve=floating_cathode_solve, time=afterglow_time
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
        "beam_excitation_radiation",
    }
    # Excitation channel is off by default; the term exists but is zero.
    assert np.allclose(
        pack_state(split_beam_terms["beam_excitation_radiation"]), 0.0
    )
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

    # --- Annular cathode emission profile (cathode_emission_profile):
    # the uniform disc's ceiling is a razor wall; the measured radial
    # footprint softens it into a ramp. Single warm annulus at the plasma
    # footprint must reproduce the uniform solve; the gaussian profile must
    # produce a monotone, softened V(I) knee.
    from cablp.funcs._cathode_solver import PlasmaState, solve as cathode_solve_fn
    from cablp.solvers._sim1d.physics.cathode import (
        cathode_device_config,
        cathode_emission_annuli,
    )
    import dataclasses as _dc

    knee_params, knee_flags = default_config()
    knee_params.update({"V_bank": 173.6, "R_comp": 5.72e-3, "T_s": 2008.0,
                        "R_cath": 15.0, "Rp": 15.0,
                        # this block tests the uniform-disc vs single-annulus
                        # equivalence; gaussian is tested just below.
                        "cathode_emission_profile": "uniform",
                        # cathode-solver emission/SCL/bridge unit tests: isolate
                        # from the ads_des surface state (its dynamic phi_eff)
                        # and use the fixed literature work function.
                        "cathode_surface_model": "none",
                        "cathode_phiwf_clean_eV": None,
                        "phi_wf": 3.0})
    uni_cfg = cathode_device_config(knee_params, knee_flags, sim.mu)
    plasma_probe = PlasmaState(T_e=8.0, n_e=4e12, n_n=1.5e13, sigma_b=4e-17)
    # single annulus, fully wetted, at T_s: identical emission physics
    one_annulus = _dc.replace(
        uni_cfg,
        emission_Ts_K=(knee_params["T_s"],),
        emission_area_cm2=(uni_cfg.A_c,),
        emission_plasma_frac=(1.0,),
    )
    r_uni = cathode_solve_fn(uni_cfg, plasma_probe, x0=None, floating=False)
    r_one = cathode_solve_fn(one_annulus, plasma_probe, x0=None, floating=False)
    assert np.isclose(r_one.I_tot, r_uni.I_tot, rtol=1e-10)
    assert np.isclose(r_one.phi_c, r_uni.phi_c, rtol=1e-10)
    assert np.isclose(one_annulus.I_eth, uni_cfg.I_eth, rtol=1e-12)

    # gaussian profile: annuli temperatures fall monotonically from T_s,
    # total emission below the uniform disc's, plasma fractions partition
    gauss_params = dict(knee_params)
    gauss_params.update({"R_cath": 19.0, "cathode_emission_profile": "gaussian",
                         "cathode_Ts_fwhm_cm": 28.0})
    Ts_k, area_k, frac_k = cathode_emission_annuli(gauss_params)
    assert np.isclose(Ts_k[0], gauss_params["T_s"], rtol=2e-2)
    assert np.all(np.diff(Ts_k) < 0.0)
    assert Ts_k[0] - Ts_k[-1] > 100.0  # the knee-softening spread
    assert np.isclose(np.sum(area_k), np.pi * 19.0**2, rtol=1e-12)
    assert frac_k[0] == 1.0 and frac_k[-1] == 0.0
    gauss_cfg = cathode_device_config(gauss_params, knee_flags, sim.mu)
    assert gauss_cfg.I_eth < uni_cfg.I_eth * (np.pi * 19.0**2) / uni_cfg.A_c
    hot_params = dict(gauss_params, T_s=2110.0)
    hot_cfg = cathode_device_config(hot_params, knee_flags, sim.mu)

    # --- Current-driven sheath solve (CATHODE_IDRIVEN_PLAN.md M2): given the
    # V-driven solve's I_tot, solve_idriven must reproduce the same operating
    # point -- phi_c/phi_a/I_eth_star/regime -- through the monotone device
    # relation, with no warm windows and no bypass iteration. The M2 gate.
    from cablp.funcs._cathode_solver_idriven import solve_idriven

    id_sweep = []
    id_plasmas = (
        plasma_probe,
        PlasmaState(T_e=3.0, n_e=5.0e11, n_n=2.0e13, sigma_b=0.0),
        PlasmaState(T_e=12.0, n_e=1.0e13, n_n=5.0e12, sigma_b=4e-17),
    )
    for id_cfg in (uni_cfg, one_annulus, gauss_cfg, hot_cfg):
        for id_pl in id_plasmas:
            if id_cfg is gauss_cfg and id_pl is id_plasmas[1]:
                # Degenerate flat-top corner: covered by its own test below,
                # where psi is not recoverable from I within float precision.
                continue
            id_sweep.append((id_cfg, id_pl))
    id_regimes = set()
    for id_cfg, id_pl in id_sweep:
        rv = cathode_solve_fn(id_cfg, id_pl, x0=None, floating=False)
        ri = solve_idriven(id_cfg, id_pl, I_tot_A=rv.I_tot)
        id_regimes.add(rv.regime)
        assert ri.regime == rv.regime, (ri.regime, rv.regime)
        for id_att in (
            "phi_c",
            "phi_c_plus",
            "phi_c_minus",
            "phi_a",
            "I_eth_star",
            "I_tot",
            "V_p",
            "beam_bypass_fraction",
            "l_b",
            "P_cathode_i",
            "P_prim",
        ):
            assert np.isclose(
                getattr(ri, id_att),
                getattr(rv, id_att),
                rtol=1e-8,
                atol=1e-9,
            ), (id_att, getattr(ri, id_att), getattr(rv, id_att))
        # V_b contract: the I-driven V_b is the device voltage; the V-driven
        # V_b equals it up to that solver's own root residual (~<=1e-3 V).
        id_v_dev = rv.phi_c + rv.V_p - rv.phi_a
        assert np.isclose(ri.V_b, id_v_dev, rtol=1e-8, atol=1e-8)
        assert abs(rv.V_b - id_v_dev) < 1.0e-2, (rv.V_b, id_v_dev)
    assert {"classical", "virtual_cathode"} <= id_regimes

    # Degenerate emission-exhausted plateau (the I-driven formulation's
    # mirror-image weak spot): at this corner every annulus is released and
    # the electron tail has underflowed, so J_tot(psi) is numerically
    # constant -- psi is NOT recoverable from I alone. The solve must stay
    # deterministic (leading-edge selection), reproduce the *currents*, and
    # never raise; the potentials legitimately disagree with the V-driven
    # root there.
    id_deg_rv = cathode_solve_fn(
        gauss_cfg, id_plasmas[1], x0=None, floating=False
    )
    id_deg_ri = solve_idriven(gauss_cfg, id_plasmas[1], I_tot_A=id_deg_rv.I_tot)
    assert id_deg_ri.regime in ("virtual_cathode", "capability_limited")
    assert np.isfinite(id_deg_ri.phi_c) and np.isfinite(id_deg_ri.V_b)
    assert np.isclose(id_deg_ri.I_tot, id_deg_rv.I_tot, rtol=1e-8)
    assert np.isclose(id_deg_ri.I_eth_star, id_deg_rv.I_eth_star, rtol=1e-6)
    id_deg_repeat = solve_idriven(
        gauss_cfg, id_plasmas[1], I_tot_A=id_deg_rv.I_tot
    )
    assert id_deg_repeat.phi_c == id_deg_ri.phi_c  # deterministic

    # Monotone by construction: deeper sheath carries more current, so the
    # inverse map I -> phi is single-valued and increasing.
    id_ref = cathode_solve_fn(uni_cfg, plasma_probe, x0=None, floating=False)
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

    # Schottky lowering (opt-in): the field-tilted emission ceiling gives the
    # knee a finite dV/dI. Near the raw ceiling the enhanced curve must sit
    # at lower voltage and with a much smaller maximum slope.
    id_knee = np.linspace(0.90 * id_ceiling, 0.995 * id_ceiling, 12)
    id_vb_off = np.array(
        [
            solve_idriven(uni_cfg, plasma_probe, float(I)).V_b
            for I in id_knee
        ]
    )
    id_vb_on = np.array(
        [
            solve_idriven(uni_cfg, plasma_probe, float(I), schottky=True).V_b
            for I in id_knee
        ]
    )
    assert np.all(np.isfinite(id_vb_off)) and np.all(np.isfinite(id_vb_on))
    assert np.all(id_vb_on <= id_vb_off + 1e-9)
    id_slope_off = np.max(np.abs(np.diff(id_vb_off) / np.diff(id_knee)))
    id_slope_on = np.max(np.abs(np.diff(id_vb_on) / np.diff(id_knee)))
    assert id_slope_on < 0.5 * id_slope_off, (id_slope_on, id_slope_off)

    # Thermal bridge (opt-in, chatter diagnosis 2026-07-21): the kT_s-width
    # C1 blend across the SCL<->classical release corner. Kernel first:
    # exact hard branches outside the window, continuous slope across the
    # edges, J_star <= min(J_eff, J_crit) everywhere, both outputs monotone.
    from cablp.funcs._cathode_solver_idriven import (
        _BRIDGE_HALF_WIDTH,
        _bridge_release,
    )

    br_w = _BRIDGE_HALF_WIDTH
    for br_x in (-5.0, -br_w - 1e-12):
        br_J, br_b = _bridge_release(float(np.exp(br_x)), 1.0)
        assert br_J == float(np.exp(br_x)) and br_b == 0.0, br_x
    for br_x in (br_w + 1e-12, 5.0):
        br_J, br_b = _bridge_release(float(np.exp(br_x)), 1.0)
        assert np.isclose(br_J, 1.0, rtol=1e-12) and np.isclose(br_b, br_x)
    br_xs = np.linspace(-2.0 * br_w, 2.0 * br_w, 401)
    br_pairs = [_bridge_release(float(np.exp(x)), 1.0) for x in br_xs]
    br_Js = np.array([p[0] for p in br_pairs])
    br_bs = np.array([p[1] for p in br_pairs])
    assert np.all(br_Js <= np.minimum(np.exp(br_xs), 1.0) + 1e-15)
    assert np.all(np.diff(br_Js) >= -1e-15)  # released current monotone
    assert np.all(np.diff(br_bs) >= -1e-15)  # barrier monotone
    br_slope = np.diff(br_bs) / np.diff(br_xs)
    # C1: slope increments stay at the quadratic's own scale (a hard corner
    # would jump O(1) between adjacent samples).
    assert np.all(np.abs(np.diff(br_slope)) < 0.02)

    # Solve level: bridge off is the default hard path (bit-identical);
    # bridge on stays finite, carries the imposed current, and -- since the
    # blend only ever *adds* barrier -- sits at least as deep in psi at
    # fixed current. Monotonicity of the I -> phi map (the architecture's
    # load-bearing property) must survive bridge x schottky.
    for br_cfg in (uni_cfg, gauss_cfg):
        for br_I in (0.05 * id_ceiling, 0.5 * id_ceiling, 0.95 * id_ceiling):
            br_hard = solve_idriven(br_cfg, plasma_probe, I_tot_A=float(br_I))
            br_dflt = solve_idriven(br_cfg, plasma_probe, I_tot_A=float(br_I),
                                    bridge=False)
            assert br_dflt.phi_c_plus == br_hard.phi_c_plus
            br_on = solve_idriven(br_cfg, plasma_probe, I_tot_A=float(br_I),
                                  bridge=True)
            assert np.isfinite(br_on.V_b) and np.isfinite(br_on.phi_c)
            # id_ceiling is the *uniform* config's; the gaussian's own
            # ceiling is lower (edge-cooled annuli), so the top current
            # may legitimately land capability-limited -- in lockstep
            # with the hard solve.
            assert (
                np.isclose(br_on.I_tot, br_I, rtol=1e-9)
                or br_on.regime == "capability_limited"
            )
            assert (br_on.regime == "capability_limited") == (
                br_hard.regime == "capability_limited"
            )
            assert br_on.phi_c_plus >= br_hard.phi_c_plus - 1e-9
            br_rep = solve_idriven(br_cfg, plasma_probe, I_tot_A=float(br_I),
                                   bridge=True)
            assert br_rep.phi_c_plus == br_on.phi_c_plus  # deterministic
    for br_sch in (False, True):
        br_phis = np.array([
            solve_idriven(uni_cfg, plasma_probe, I_tot_A=float(I),
                          schottky=br_sch, bridge=True).phi_c_plus
            for I in id_grid
        ])
        assert np.all(np.diff(br_phis) > 0.0), f"schottky={br_sch}"
        br_vbs = np.array([
            solve_idriven(uni_cfg, plasma_probe, I_tot_A=float(I),
                          schottky=br_sch, bridge=True).V_b
            for I in id_grid
        ])
        assert np.all(np.diff(br_vbs) > -1e-9), f"schottky={br_sch}"
    # Exact reduction outside the window, both sides (measured x positions:
    # released classical at 1000 A has x = ln(J_eth/J_crit) ~ -1.14; the
    # cold plasma at 20 A is a deep virtual cathode with x ~ +2).
    for br_pl, br_I in ((plasma_probe, 1000.0), (id_plasmas[1], 20.0)):
        br_deep_on = solve_idriven(uni_cfg, br_pl, I_tot_A=br_I, bridge=True)
        br_deep_off = solve_idriven(uni_cfg, br_pl, I_tot_A=br_I)
        assert np.isclose(br_deep_on.phi_c_plus, br_deep_off.phi_c_plus,
                          rtol=1e-11), br_I
        assert np.isclose(br_deep_on.phi_c_minus, br_deep_off.phi_c_minus,
                          rtol=1e-11, atol=1e-13), br_I

    # --- Current-driven circuit integration (CATHODE_IDRIVEN_PLAN.md M3):
    # TR-BDF2 stages as bracketed scalar root-finds against monotone
    # V_dis(I). Gate 1: 2nd order on the analytic RLC decay with a linear
    # V_dis(I) load (halve dt, error / ~4).
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

    # Gate 3: the stiff wall (why the scheme is implicit -- plan §2c
    # revision). A device curve with a 1 MOhm/A branch above I_ceil:
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
        m3_cu_flags, cathode_coupling=True, neutral_prebreakdown=False,
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
    assert m3_float.beam_result.result.I_tot == 0.0
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
    # The manifold excitation channel is consumed by the current-driven
    # builder too (A2 adoption): same dispatch, per-cell mean radiated
    # energy filled, cross section wider than the 2^1P-only channel at the
    # same solve.
    m3_exc_params = dict(m3_params, b_beam_excitation=1.0)
    m3_exc_sim = LAPDSim1D(m3_exc_params, resolved_cathode_flags)
    m3_exc_sim._circuit_I_loop = 800.0
    m3_exc_solve = m3_exc_sim.solve_cathode_boundary(update_cache=False)
    m3_mfd_params = dict(
        m3_exc_params, beam_excitation_model="manifold"
    )
    m3_mfd_sim = LAPDSim1D(m3_mfd_params, resolved_cathode_flags)
    m3_mfd_sim._circuit_I_loop = 800.0
    m3_mfd_solve = m3_mfd_sim.solve_cathode_boundary(update_cache=False)
    m3_launch = int(
        np.flatnonzero(m3_mfd_solve.beam_result.beam_cross)[0]
    )
    assert (
        m3_mfd_solve.beam_result.beam_exc_cross[m3_launch]
        > m3_exc_solve.beam_result.beam_exc_cross[m3_launch]
        > 0.0
    )
    assert (
        21.5
        < float(m3_mfd_solve.beam_result.beam_exc_energy_eV[m3_launch])
        < 22.5
    )
    assert (
        float(m3_exc_solve.beam_result.beam_exc_energy_eV[m3_launch])
        == 21.218
    )
    # B2: the CSDA deposition rides the current-driven dispatch too (the
    # solver-agnostic interface's second consumer).
    m3_csda_params = dict(m3_exc_params, beam_deposition_model="csda")
    m3_csda_sim = LAPDSim1D(m3_csda_params, resolved_cathode_flags)
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
    # Power-balance warming under current_driven must feed on the HONEST
    # accepted-state solve, not the RHS cache: the cache holds the step's
    # last internal-stage solve, measured at 4.6-7.5x the accepted-state
    # P_cathode_i at the same frozen current (2026-07-21). Spy on the
    # evaluator the warming branch uses and require the energy ledger to
    # integrate exactly the honest values it returned.
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
                cathode_warming_model="power_balance",
                T_s=1910.0,
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

    # Surface-state coverage model (cathode_surface_model="ads_des",
    # CATHODE_IDRIVEN_PLAN.md M5a). Validation fails fast; the coverage
    # update must reproduce the backward-Euler form exactly from the spy's
    # honest I_i; phi_eff must actually reach the solve (a cleaner surface
    # emits more at fixed T_s and imposed current => shallower sheath).
    for sf_bad in (
        {"cathode_surface_model": "bogus"},
        # missing clean floor (the default now supplies one, so clear it):
        {"cathode_surface_model": "ads_des", "cathode_phiwf_clean_eV": None},
        {"cathode_surface_model": "ads_des",
         "cathode_phiwf_clean_eV": 99.0},  # floor above phi_wf
        {"cathode_surface_model": "ads_des",
         "cathode_phiwf_clean_eV": 2.75,
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
        cathode_surface_model="ads_des",
        cathode_phiwf_clean_eV=2.75,
        cathode_cleaning_sigma_cm2=1.0e-16,
    )
    sf_flags = dict(
        sf_cu_flags, cathode_coupling=True, neutral_prebreakdown=False,
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
    assert np.all(np.diff(sf_theta) <= 0.0)  # cleaning only, k_ads = 0
    # Reproduce the backward-Euler update exactly from the spy's honest
    # I_i sequence (run() starts I_loop at 0, so the accepted honest
    # solves carry the near-floating I_i -- the form is what's tested).
    sf_area = np.pi * float(sf_params["R_cath"]) ** 2
    sf_th = 1.0
    for sf_Ii in sf_calls:
        sf_G = max(sf_Ii, 0.0) / (1.602176634e-19 * sf_area)
        sf_loss = 0.0 + 1.0e-16 * sf_G
        sf_th = (sf_th + 1.0e-10 * 0.0) / (1.0 + 1.0e-10 * (0.0 + sf_loss))
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
    sf_kT = 8.617333262e-5 * float(sf_params["T_s"])
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

    # The bridge flag rides the dispatch (input_flags namespace, like
    # cathode_schottky): a bridged current-driven solve is finite and
    # carries the frozen loop current.
    m3_br_sim = LAPDSim1D(
        m3_params, dict(resolved_cathode_flags, cathode_emission_bridge=True)
    )
    m3_br_sim._circuit_I_loop = 800.0
    m3_br_solve = m3_br_sim.solve_cathode_boundary(update_cache=False)
    m3_br_res = m3_br_solve.beam_result.result
    assert np.isfinite(m3_br_res.V_b) and np.isfinite(m3_br_res.phi_c)
    assert (
        np.isclose(m3_br_res.I_tot, 800.0, rtol=1e-6)
        or m3_br_res.regime == "capability_limited"
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
    m3_vdis = idriven_vdis_evaluator(
        state=m3_run_sim.state,
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

    # --- Beam excitation channel (b_beam_excitation, default 0 = historical).
    from cablp.funcs._cathode_solver import beam_excitation_cross

    sigma_exc_100 = beam_excitation_cross(100.0, 1.0, "He")
    assert 5.0e-18 < sigma_exc_100 < 2.0e-17
    assert beam_excitation_cross(100.0, 0.0, "He") == 0.0
    assert beam_excitation_cross(10.0, 1.0, "He") == 0.0  # below threshold
    try:
        beam_excitation_cross(100.0, 1.0, "H")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for H beam excitation")

    # The b_beam_excitation knob is a beer_lambert control (inert under csda,
    # which uses the measured manifold); match the beer_lambert base_beam.
    exc_params = dict(params, beam_deposition_model="beer_lambert")
    exc_params["b_beam_excitation"] = 1.0
    exc_sim = LAPDSim1D(exc_params, cathode_flags)
    exc_sim._circuit_I_loop = 3000.0
    exc_solve = exc_sim.solve_cathode_boundary()
    exc_beam = exc_solve.beam_result
    base_beam = cathode_solve.beam_result
    launch_idx = int(np.flatnonzero(exc_beam.beam_cross)[0])
    assert exc_beam.beam_exc_cross[launch_idx] > 0.0
    assert (
        exc_beam.beam_atten_cross[launch_idx]
        > exc_beam.beam_cross[launch_idx]
    )
    # Both first solves run from a zeroed sigma_b cache, so the circuit state
    # is identical and the only difference is the attenuation cross section:
    # the inelastic deposition length must be strictly shorter everywhere.
    assert np.isclose(
        exc_solve.beam_result.result.phi_c, base_beam.result.phi_c
    )
    positive = base_beam.l_b_profile > 0.0
    assert np.all(
        exc_beam.l_b_profile[positive] < base_beam.l_b_profile[positive]
    )
    exc_terms = exc_sim.beam_ionization_rhs_terms(cathode_solve=exc_solve)
    exc_rad = exc_terms["beam_excitation_radiation"]
    assert np.all(exc_rad.Ee <= 0.0)
    assert np.any(exc_rad.Ee < 0.0)
    for field_values in (exc_rad.n, exc_rad.nn, exc_rad.M, exc_rad.Ei):
        assert np.allclose(field_values, 0.0)
    # The channels split one absorbed flux: radiated events / ionizations =
    # sigma_exc / sigma_ion, cell by cell.
    exc_birth = exc_terms["beam_ionization_birth"]
    birth_mask = exc_birth.n > 0.0
    E_exc_erg = float(exc_params.get("beam_excitation_energy_eV", 21.218)) * ev_to_erg
    event_ratio = (-exc_rad.Ee[birth_mask] / E_exc_erg) / exc_birth.n[birth_mask]
    assert np.allclose(
        event_ratio,
        exc_beam.beam_exc_cross[launch_idx] / exc_beam.beam_cross[launch_idx],
        rtol=1e-10,
    )

    # --- A2: the manifold excitation model (BEAM_DEPOSITION_PLAN WP-A).
    from cablp.funcs._cathode_solver import beam_excitation_channel
    from cablp.funcs._cross import (
        He_beam_excitation_channel as _He_manifold_channel,
    )

    # Dispatch: the scalar path reproduces the historical function
    # byte-for-byte; the manifold path matches the _cross helper with
    # b_beam_excitation as a pure multiplier on the cross section only.
    assert beam_excitation_channel(100.0, 1.4, "He") == (
        beam_excitation_cross(100.0, 1.4, "He"),
        21.218,
    )
    _mf_sigma, _mf_E = beam_excitation_channel(100.0, 1.0, "He", model="manifold")
    assert (_mf_sigma, _mf_E) == _He_manifold_channel(100.0)
    _mf_sigma_h, _mf_E_h = beam_excitation_channel(
        100.0, 0.5, "He", model="manifold"
    )
    assert np.isclose(_mf_sigma_h, 0.5 * _mf_sigma) and _mf_E_h == _mf_E
    assert beam_excitation_channel(100.0, 0.0, "He", model="manifold") == (0.0, 0.0)
    # Below the lowest manifold threshold (2^1S, 20.6158 eV).
    assert beam_excitation_channel(15.0, 1.0, "He", model="manifold") == (0.0, 0.0)
    # The measured manifold vs the historical 2^1P channel at 100 eV
    # (measure_beam_manifold.py, 2026-07-20): 1.67x the events, mean
    # radiated energy 21.98 eV — within the retired estimate's 1.4 +- 0.4.
    assert 1.55 < _mf_sigma / beam_excitation_cross(100.0, 1.0, "He") < 1.80
    assert 21.5 < _mf_E < 22.5
    for bad_call in (
        lambda: beam_excitation_channel(100.0, 1.0, "He", model="bogus"),
        lambda: beam_excitation_channel(100.0, 1.0, "H", model="manifold"),
    ):
        try:
            bad_call()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError from excitation channel")

    # Lookup-table front end (deposit_beam's hot path, 2026-07-21): exact
    # at the table nodes by construction; between nodes the interp error
    # must stay below the physics-irrelevant level, and the domain edges
    # must reproduce the exact function's contract.
    from cablp.funcs._cross import He_beam_excitation_channel_lkup

    _lk_rng = np.random.default_rng(20260721)
    _lk_Es = np.concatenate([
        _lk_rng.uniform(20.7, 25.0, 40),   # threshold cluster
        _lk_rng.uniform(25.0, 180.0, 40),  # beam operating range
        _lk_rng.uniform(180.0, 1500.0, 20),
    ])
    for _lk_E in _lk_Es:
        _lk_s, _lk_e = He_beam_excitation_channel_lkup(float(_lk_E))
        _ex_s, _ex_e = _He_manifold_channel(float(_lk_E))
        assert abs(_lk_s - _ex_s) <= 1e-4 * _ex_s + 1e-21, (_lk_E, _lk_s, _ex_s)
        if _ex_s > 0.0:
            assert abs(_lk_e - _ex_e) <= 1e-4 * _ex_e, (_lk_E, _lk_e, _ex_e)
    assert He_beam_excitation_channel_lkup(15.0) == (0.0, 0.0)
    assert He_beam_excitation_channel_lkup(20.0) == (0.0, 0.0)
    # Above the table span: exact fallback, identical values.
    assert He_beam_excitation_channel_lkup(2500.0) == _He_manifold_channel(2500.0)

    # Solver-level: manifold mode widens the excitation cross section at the
    # same first-solve phi_c (zeroed sigma_b cache on both rigs) and books
    # the per-ray energy-weighted mean per event; the split-flux ratio
    # assertion holds with the manifold values in place of the constants.
    mfd_params = dict(exc_params)
    mfd_params["beam_excitation_model"] = "manifold"
    mfd_sim = LAPDSim1D(mfd_params, cathode_flags)
    mfd_sim._circuit_I_loop = 3000.0
    mfd_solve = mfd_sim.solve_cathode_boundary()
    mfd_beam = mfd_solve.beam_result
    assert np.isclose(mfd_beam.result.phi_c, exc_beam.result.phi_c)
    assert (
        mfd_beam.beam_exc_cross[launch_idx] > exc_beam.beam_exc_cross[launch_idx]
    )
    _mf_E_launch = float(mfd_beam.beam_exc_energy_eV[launch_idx])
    assert 21.5 < _mf_E_launch < 22.5
    # The 2p_scalar rig reports the constant threshold as its per-event energy.
    assert float(exc_beam.beam_exc_energy_eV[launch_idx]) == 21.218
    mfd_terms = mfd_sim.beam_ionization_rhs_terms(cathode_solve=mfd_solve)
    mfd_rad = mfd_terms["beam_excitation_radiation"]
    mfd_birth = mfd_terms["beam_ionization_birth"]
    mfd_mask = mfd_birth.n > 0.0
    mfd_ratio = (
        -mfd_rad.Ee[mfd_mask] / (_mf_E_launch * ev_to_erg)
    ) / mfd_birth.n[mfd_mask]
    assert np.allclose(
        mfd_ratio,
        mfd_beam.beam_exc_cross[launch_idx] / mfd_beam.beam_cross[launch_idx],
        rtol=1e-10,
    )

    # --- B2: the CSDA deposition model wired behind beam_deposition_model.
    csda_params = dict(exc_params)
    csda_params["beam_deposition_model"] = "csda"
    csda_sim = LAPDSim1D(csda_params, cathode_flags)
    csda_sim._circuit_I_loop = 3000.0
    csda_solve = csda_sim.solve_cathode_boundary()
    assert csda_solve.beam_deposition is not None
    csda_dep = csda_solve.beam_deposition[0]
    assert csda_dep is not None
    # Per-ray energy conservation through the module at solver conditions:
    # Gamma0*E0 = I_eth_star*phi_c (W -> erg/s is exactly 1e7).
    csda_res = csda_solve.beam_result.result
    csda_budget = csda_res.I_eth_star * csda_res.phi_c * 1.0e7
    csda_total = (
        csda_dep.plasma_heating_erg_s.sum()
        + csda_dep.radiated_erg_s.sum()
        + csda_dep.ionization_cost_erg_s.sum()
        # R4.1 anode interception is the production default (csda), so the
        # anode-removed energy is part of the per-ray budget.
        + float(csda_dep.anode_intercepted_erg_s)
        + csda_dep.transmitted_flux
        * csda_dep.transmitted_energy_eV
        * ev_to_erg
    )
    assert abs(csda_total - csda_budget) / csda_budget < 1e-9
    # The solver's four-term booking reproduces the module's split: the
    # power-deposition term carries heating + radiated + cost (plus the
    # historical gap-weighted P_ohmic), and the cost/radiation sinks
    # subtract back to the module's net heating.
    csda_terms = csda_sim.beam_ionization_rhs_terms(cathode_solve=csda_solve)
    csda_Vp = geom.plasma_volume_cm3
    csda_power_sum = float(
        (csda_terms["beam_power_deposition"].Ee * csda_Vp).sum()
    )
    csda_module_sum = float(
        csda_dep.plasma_heating_erg_s.sum()
        + csda_dep.radiated_erg_s.sum()
        + csda_dep.ionization_cost_erg_s.sum()
    )
    assert np.isclose(
        csda_power_sum - csda_res.P_ohmic * 1.0e7, csda_module_sum, rtol=1e-9
    )
    assert np.isclose(
        float((-csda_terms["beam_excitation_radiation"].Ee * csda_Vp).sum()),
        float(csda_dep.radiated_erg_s.sum()),
        rtol=1e-9,
    )
    assert np.isclose(
        float((-csda_terms["beam_ionization_cost"].Ee * csda_Vp).sum()),
        float(csda_dep.ionization_cost_erg_s.sum()),
        rtol=1e-9,
    )
    # CSDA primaries survive multiple events: ionization spreads over
    # several cells rather than one launch cell.
    assert np.count_nonzero(csda_dep.ionization_events) >= 2
    # The bypass adapter wrote a finite effective attenuation cross section
    # for the next solve's Beer-Lambert bypass.
    csda_launch = int(np.flatnonzero(csda_solve.beam_result.beam_cross)[0])
    csda_sigma_eff = float(
        csda_solve.beam_result.beam_atten_cross[csda_launch]
    )
    assert np.isfinite(csda_sigma_eff) and csda_sigma_eff >= 0.0

    # --- R4.1 (audit A15): anode-mesh beam interception is the PRODUCTION DEFAULT
    # (correct csda physics), so csda_sim above already has it on -- the anode
    # books energy and it is part of the csda per-ray budget checked earlier.
    assert float(csda_dep.anode_intercepted_erg_s) > 0.0
    # A/B off: setting the flag False restores the old (over-depositing) csda run,
    # which deposits strictly MORE power into the plasma and intercepts nothing.
    noint_flags = dict(cathode_flags)
    noint_flags["beam_anode_interception"] = False
    noint_sim = LAPDSim1D(dict(csda_params), noint_flags)
    noint_sim._circuit_I_loop = 3000.0
    noint_solve = noint_sim.solve_cathode_boundary()
    noint_dep = noint_solve.beam_deposition[0]
    assert float(noint_dep.anode_intercepted_erg_s) == 0.0
    noint_terms = noint_sim.beam_ionization_rhs_terms(cathode_solve=noint_solve)
    noint_power_sum = float((noint_terms["beam_power_deposition"].Ee * csda_Vp).sum())
    assert csda_power_sum < noint_power_sum
    # csda control: the flag is inert (not an error) under beer_lambert, which
    # never launches the CSDA module -- exactly like beam_coulomb_model /
    # beam_anomalous_model. Construction succeeds and there is no CSDA deposition.
    bl_flags = dict(cathode_flags)
    bl_flags["beam_anode_interception"] = True
    bl_sim = LAPDSim1D(dict(exc_params), bl_flags)  # exc_params is beer_lambert
    bl_sim._circuit_I_loop = 3000.0
    assert bl_sim.solve_cathode_boundary().beam_deposition is None

    # --- Beam-deposition smoothing CONSERVES the deposit over the live plasma.
    # The Gaussian redistribution kernel must place ZERO weight on the typed
    # plasma-dead cells (plenum/obstruction) behind the cathode face, because
    # the RHS mask ``_apply_active_plasma_topology`` zeroes exactly those rows:
    # anything the kernel spreads back there is silently DELETED, and it takes
    # beam power, beam ionization, excitation and the neutral debit with it
    # (all four channels share this one kernel). ``plasma_volume_cm3 > 0`` does
    # NOT identify those cells -- the dead cells have a finite plasma volume --
    # so the support has to come from ``plasma_active``.
    #
    # Checked on BOTH a uniform and a non-uniform (source_fixed_grid) mesh: the
    # kernel is weighted by cell length, and without that weighting a refined
    # region is over-weighted per cm, which makes the smoothing operator itself
    # mesh-dependent even where it happens to conserve.
    smooth_sigma_cm = 50.0
    smoothing_meshes = (
        ("uniform", dict(csda_params), dict(cathode_flags)),
        (
            "source_fixed_grid",
            {
                **csda_params,
                "source_region_length_cm": 100.0,
                "source_region_dz_cm": 10.0,
                "gas_puff_z_cm": 60.0,
            },
            {**cathode_flags, "source_fixed_grid": True},
        ),
    )
    for mesh_label, smooth_base, smooth_flags in smoothing_meshes:
        smooth_off_sim = LAPDSim1D(dict(smooth_base), smooth_flags)
        smooth_on_sim = LAPDSim1D(
            {**smooth_base, "beam_deposition_smoothing_cm": smooth_sigma_cm},
            smooth_flags,
        )
        smooth_off_sim._circuit_I_loop = 3000.0
        smooth_on_sim._circuit_I_loop = 3000.0
        smooth_geom = smooth_on_sim.get_initial_snapshot().geometry
        smooth_active = np.asarray(smooth_geom.plasma_active, dtype=bool)
        smooth_Vp = np.asarray(smooth_geom.plasma_volume_cm3, dtype=float)
        smooth_dz = np.asarray(smooth_geom.length_cm, dtype=float)
        # The premise of the test: there ARE dead cells to leak into, and the
        # old ``Vp > 0`` support could not have found them.
        assert not smooth_active.all(), mesh_label
        assert (smooth_Vp > 0.0).all(), mesh_label
        if mesh_label == "source_fixed_grid":
            assert np.unique(np.round(smooth_dz[smooth_active], 9)).size > 1

        # (a) The kernel itself: no weight on any row the RHS mask will zero,
        # and every source column normalized to exactly 1 over the live support.
        smooth_W = _beam_smoothing_matrix(smooth_geom, smooth_sigma_cm)
        assert np.count_nonzero(smooth_W[~smooth_active, :]) == 0, mesh_label
        smooth_colsum = smooth_W[smooth_active, :].sum(axis=0)
        assert np.allclose(smooth_colsum, 1.0, rtol=0.0, atol=1e-12), (
            mesh_label,
            float(smooth_colsum.min()),
            float(smooth_colsum.max()),
        )

        # (b) The deposited RHS: smoothed-then-masked total == unsmoothed total,
        # channel by channel. Both sims are driven from the SAME cathode solve,
        # so the only difference between them is the smoothing operator.
        smooth_state = smooth_on_sim.state
        smooth_solve = smooth_on_sim.solve_cathode_boundary(
            state=smooth_state, update_cache=False
        )
        assert smooth_solve.beam_deposition is not None, mesh_label
        smooth_off_terms = smooth_off_sim.beam_ionization_rhs_terms(
            state=smooth_state, cathode_solve=smooth_solve
        )
        smooth_on_terms = smooth_on_sim.beam_ionization_rhs_terms(
            state=smooth_state, cathode_solve=smooth_solve
        )
        for smooth_term, smooth_field in (
            ("beam_ionization_birth", "n"),
            ("beam_ionization_birth", "nn"),
            ("beam_power_deposition", "Ee"),
            ("beam_ionization_cost", "Ee"),
            ("beam_excitation_radiation", "Ee"),
        ):
            off_row = np.asarray(
                getattr(smooth_off_terms[smooth_term], smooth_field), dtype=float
            )
            on_row = np.asarray(
                getattr(smooth_on_terms[smooth_term], smooth_field), dtype=float
            )
            off_total = float((off_row * smooth_Vp)[smooth_active].sum())
            on_total = float((on_row * smooth_Vp)[smooth_active].sum())
            # A zero channel would make the conservation check vacuous.
            assert abs(off_total) > 0.0, (mesh_label, smooth_term, smooth_field)
            assert abs(on_total - off_total) <= 1e-12 * abs(off_total), (
                mesh_label,
                smooth_term,
                smooth_field,
                on_total,
                off_total,
                on_total / off_total,
            )
            # ...and the kernel is not quietly the identity: it MOVED the
            # deposit, so the conservation above is a real statement.
            assert not np.allclose(on_row, off_row), (mesh_label, smooth_term)

    # --- The smoothing-matrix cache is keyed on geometry CONTENT, not address.
    # ``id(geometry)`` is unique only among LIVE objects: CPython reuses the
    # address of a collected geometry, so a freed mesh followed by a
    # differently meshed allocation at the same address used to return the OLD
    # mesh's matrix. A cell-count mismatch would raise at the matmul; the
    # silent case is two meshes with the SAME cell count and different
    # positions -- exactly what an nx-matched source_region_dz_cm refinement
    # sweep builds.
    smoothkey_flags = {**cathode_flags, "source_fixed_grid": True}
    smoothkey_base = dict(
        csda_params,
        source_region_length_cm=100.0,
        gas_puff_z_cm=60.0,
    )

    def _smoothkey_geometry(dz_cm, nx):
        sim = LAPDSim1D(
            dict(smoothkey_base, source_region_dz_cm=dz_cm, nx=nx),
            smoothkey_flags,
        )
        return sim.get_initial_snapshot().geometry

    # Halving the fixed source cell size doubles the fixed-region cells; nx is
    # cut by the same amount so the two meshes have IDENTICAL cell counts.
    smoothkey_geom_a = _smoothkey_geometry(10.0, 40)
    smoothkey_geom_b = _smoothkey_geometry(5.0, 35)
    smoothkey_geom_a2 = _smoothkey_geometry(10.0, 40)
    # Premises: same cell count, genuinely different meshes, distinct objects.
    assert smoothkey_geom_a.length_cm.size == smoothkey_geom_b.length_cm.size, (
        smoothkey_geom_a.length_cm.size,
        smoothkey_geom_b.length_cm.size,
    )
    assert not np.array_equal(smoothkey_geom_a.z_cm, smoothkey_geom_b.z_cm)
    assert smoothkey_geom_a is not smoothkey_geom_a2
    assert smoothkey_geom_a.length_cm.size == smoothkey_geom_a2.length_cm.size

    # (a)/(c) The regression: same cells, different spacing must NOT alias.
    assert _beam_smoothing_key(
        smoothkey_geom_a, smooth_sigma_cm
    ) != _beam_smoothing_key(smoothkey_geom_b, smooth_sigma_cm)
    smoothkey_W_a = _beam_smoothing_matrix(smoothkey_geom_a, smooth_sigma_cm)
    smoothkey_W_b = _beam_smoothing_matrix(smoothkey_geom_b, smooth_sigma_cm)
    assert smoothkey_W_a is not smoothkey_W_b
    assert not np.allclose(smoothkey_W_a, smoothkey_W_b)

    # (b) The cache still caches: two DISTINCT geometry objects with identical
    # content share the single O(cells^2) build. Guards the performance
    # property -- a key that accidentally never hits would run the build on
    # every RHS evaluation.
    smoothkey_W_a2 = _beam_smoothing_matrix(smoothkey_geom_a2, smooth_sigma_cm)
    assert smoothkey_W_a2 is smoothkey_W_a

    # The active-support term of the key is load-bearing: since the kernel is
    # built over ``plasma_active``, two meshes agreeing in z/lengths/faces but
    # differing in cell ROLES build different matrices and must not collide.
    smoothkey_active = np.asarray(
        smoothkey_geom_a.plasma_active, dtype=bool
    ).copy()
    smoothkey_active[-2] = not smoothkey_active[-2]
    smoothkey_geom_roles = dataclasses.replace(
        smoothkey_geom_a, plasma_active=smoothkey_active
    )
    assert _beam_smoothing_key(
        smoothkey_geom_a, smooth_sigma_cm
    ) != _beam_smoothing_key(smoothkey_geom_roles, smooth_sigma_cm)
    smoothkey_W_roles = _beam_smoothing_matrix(
        smoothkey_geom_roles, smooth_sigma_cm
    )
    assert smoothkey_W_roles is not smoothkey_W_a
    assert not np.allclose(smoothkey_W_roles, smoothkey_W_a)

    # --- R4.2 (audit A14): ionization_birth_energy_model. Default-off ("legacy")
    # is the historical booking; "conservative" zeroes the bulk electron
    # birth energy (no 3Te/2 creation) and adds the ion mixing energy, and the
    # beam ion birth gains the same mixing energy (its electron Ee is already 0).
    cons_params = dict(csda_params)
    cons_params["ionization_birth_energy_model"] = "conservative"
    cons_sim = LAPDSim1D(cons_params, cathode_flags)
    cons_sim._circuit_I_loop = 3000.0
    cons_solve = cons_sim.solve_cathode_boundary()
    cons_terms = cons_sim.beam_ionization_rhs_terms(cathode_solve=cons_solve)
    leg_react = csda_sim.reaction_rhs_terms()["ionization_birth"]
    cons_react = cons_sim.reaction_rhs_terms()["ionization_birth"]
    # Bulk electron birth is zeroed; particle & momentum rows are untouched.
    assert np.all(cons_react.Ee == 0.0)
    assert np.array_equal(cons_react.n, leg_react.n)
    assert np.array_equal(cons_react.M, leg_react.M)
    # Beam ion birth gains the (non-negative) mixing energy over legacy.
    leg_beam_Ei = csda_terms["beam_ionization_birth"].Ei
    cons_beam_Ei = cons_terms["beam_ionization_birth"].Ei
    assert np.all(cons_beam_Ei >= leg_beam_Ei - 1e-30)
    # Invalid selector rejects at construction.
    try:
        LAPDSim1D(dict(csda_params, ionization_birth_energy_model="x"), cathode_flags)
    except ValueError:
        pass
    else:
        raise AssertionError("ionization_birth_energy_model must reject unknown values")

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
    source_puff, _ = puff_cell_indices(geom)
    assert source_rhs.nn[source_puff] > 0.0
    assert source_rhs.nn[0] < 0.0
    assert source_rhs.nn[-1] < 0.0
    assert np.isclose(
        source_rhs.nn[source_puff],
        puff_rate(
            params["S_gp"],
            params["gas_puff_valves"],
            geom.neutral_volume_cm3[source_puff],
        ),
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
        source_rhs.nn[source_puff] - afterglow_source.nn[source_puff],
        puff_rate(
            params["S_gp"],
            params["gas_puff_valves"],
            geom.neutral_volume_cm3[source_puff],
        ),
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
    decay_puff, _ = puff_cell_indices(decay_geom)
    decay_main_start = decay_params["tau_prebreakdown"]
    decay_event = decay_main_start + decay_params["tau_gp_after_breakdown"]
    assert np.isclose(
        decay_sim.next_phase_boundary_after(decay_main_start),
        decay_event,
    )
    decay_on = decay_sim.neutral_source_sink_rhs(time=decay_event)
    assert np.isclose(
        decay_on.nn[decay_puff],
        puff_rate(
            decay_params["S_gp"],
            decay_params["gas_puff_valves"],
            decay_geom.neutral_volume_cm3[decay_puff],
        ),
    )
    decay_time = decay_main_start + 2.0e-10
    decay_tau = (
        decay_params["tau_discharge"] - decay_params["tau_gp_after_breakdown"]
    ) * decay_params["tau_gp_decay_factor"]
    decay_factor = np.exp(-(decay_time - decay_event) / decay_tau)
    decay_rhs = decay_sim.neutral_source_sink_rhs(time=decay_time)
    assert np.isclose(
        decay_rhs.nn[decay_puff],
        puff_rate(
            decay_params["S_gp"] * decay_factor,
            decay_params["gas_puff_valves"],
            decay_geom.neutral_volume_cm3[decay_puff],
        ),
    )

    pulse_params = dict(decay_params)
    pulse_params["gas_puff_mode"] = "pulse_decay_to_level"
    pulse_params["S_gp_decay_target"] = 1000.0
    pulse_params["tau_gp_pulse_duration"] = 1.0e-10
    pulse_params["tau_gp_decay_duration"] = 2.0e-10
    pulse_sim = LAPDSim1D(pulse_params, flags)
    pulse_geom = pulse_sim.get_initial_snapshot().geometry
    pulse_puff, _ = puff_cell_indices(pulse_geom)
    pulse_event = pulse_params["tau_prebreakdown"] + pulse_params["tau_gp_pulse_duration"]
    assert np.isclose(
        pulse_sim.next_phase_boundary_after(pulse_params["tau_prebreakdown"]),
        pulse_event,
    )
    pulse_on = pulse_sim.neutral_source_sink_rhs(time=pulse_event)
    assert np.isclose(
        pulse_on.nn[pulse_puff],
        puff_rate(
            pulse_params["S_gp"],
            pulse_params["gas_puff_valves"],
            pulse_geom.neutral_volume_cm3[pulse_puff],
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
        pulse_rhs.nn[pulse_puff],
        puff_rate(
            pulse_s_gp,
            pulse_params["gas_puff_valves"],
            pulse_geom.neutral_volume_cm3[pulse_puff],
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
    open_plasma_faces = np.asarray(geom.plasma_open, dtype=bool)
    assert np.all(ramp_front.n[open_plasma_faces] > 0.0)
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

    equal_temp_state = conservative_from_primitives(
        n=np.full(geom.cells, params["ne0"]),
        nn=state.nn,
        u=np.zeros(geom.cells),
        Te=np.full(geom.cells, 0.5),
        Ti=np.full(geom.cells, 0.5),
        ion_mass_g=sim.ion_mass_g,
    )
    equal_temp_exchange = sim.energy_exchange_rhs(state=equal_temp_state)
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
        b_ionization_energy_cost=1.0,  # removed config knob; hardwired 1.0
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
    # The standalone legacy CX cooling term is a DEPRECATED A/B arm (folded into
    # the Phelps ion_neutral_moment_closure operator on the production default).
    # Exercise it explicitly with the moment operator off; the legacy-drag
    # DeprecationWarning is expected there and asserted.
    with _warnings.catch_warnings(record=True) as _cx_w:
        _warnings.simplefilter("always")
        cx_legacy_sim = LAPDSim1D(
            params, dict(flags, ion_neutral_moment_closure=False)
        )
    assert any(
        issubclass(w.category, DeprecationWarning) for w in _cx_w
    ), "legacy ion-neutral path must warn"
    hot_ion_cx = cx_legacy_sim.ion_charge_exchange_rhs(state=hot_ion_cx_state)
    assert np.all(hot_ion_cx.Ei < 0.0)
    assert np.allclose(hot_ion_cx.n, 0.0)
    assert np.allclose(hot_ion_cx.nn, 0.0)
    assert np.allclose(hot_ion_cx.M, 0.0)
    assert np.allclose(hot_ion_cx.Ee, 0.0)
    hot_ion_cx_dt = cx_legacy_sim.suggest_timestep(y=pack_state(hot_ion_cx_state))
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

    # Hydrogen coverage removed 2026-07-20: the thesis scope is He-only (all
    # experimental data is helium) and the adas rate default is wired for He.
    # gas_type = "H" remains selectable with atomic_rate_model = "janev" but
    # is no longer exercised here.
    try:
        ion_charge_exchange_rhs(
            state=hot_ion_cx_state,
            floors=sim.floors,
            ion_mass_g=sim.ion_mass_g,
            gas_type="Ar",
            Tn_fit=params["Tn_fit"],
            b_Qcx=params["b_Qcx"],
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
    assert np.min(heat_rhs.Ee) < 0.0 < np.max(heat_rhs.Ee)
    assert np.min(heat_rhs.Ei) < 0.0 < np.max(heat_rhs.Ei)
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
    assert np.any(implicit_heat_derived.Te < heat_derived.Te)
    assert np.any(implicit_heat_derived.Te > heat_derived.Te)
    assert np.any(implicit_heat_derived.Ti < heat_derived.Ti)
    assert np.any(implicit_heat_derived.Ti > heat_derived.Ti)
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
        "boundary_absorption",
        "characteristic_boundary",
        "pressure_work",
        "hyperbolic_energy_correction",
        "ei_exchange",
        "ionization_energy_cost",
        "electron_ion_cooling",
        "electron_neutral_cooling",
        "ion_charge_exchange",
        "ion_neutral_drag",
        "ion_neutral_frictional_heating",
        "ion_neutral_thermalization",
        "ion_neutral_collision",
        "neutral_momentum_wall",
        "neutral_wind_advection",
        "surface_loss",
        "anode_collection",
        "cathode_surface_loss",
        "neutral_exchange",
        "neutral_sources",
        "gas_puff_local_ionization",
        "ionization_birth",
        "beam_ionization_birth",
        "beam_power_deposition",
        "beam_ionization_cost",
        "beam_excitation_radiation",
        "recombination_rad_loss",
        "recombination_3b_loss",
        "recombination_energy_return",
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
    # b_ionization_energy_cost removed as a config knob; b_ioniz=0 already
    # zeros ionization (and its cost), so no override is needed here.
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
    assert np.all(np.isfinite(explicit_attempt_after.y))

    no_source_sim = LAPDSim1D(no_source_params, flags)
    y_before = no_source_sim.get_initial_snapshot().y.copy()
    stationary_after = no_source_sim.advance_one_step(1e-10)
    assert np.all(np.isfinite(stationary_after.y))

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
    assert np.all(np.isfinite(split_attempt_after.y))

    no_source_split_sim = LAPDSim1D(no_source_params, split_flags)
    split_before = no_source_split_sim.get_initial_snapshot().y.copy()
    split_stationary_after = no_source_split_sim.advance_one_step(1e-10)
    assert np.all(np.isfinite(split_stationary_after.y))

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
    assert np.all(np.isfinite(run_result.y))
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
        assert np.isfinite(summary.thermal_energy_relative_drift)
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
            assert h5["geometry/cell_role"][0].decode("utf-8") == "plenum"
            assert h5["geometry/cell_role"][-1].decode("utf-8") == "collector"
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
    # This block checks the STATIC-T_s diagnostics; the power_balance warming is
    # exercised in its own block just below.
    cathode_run_params["cathode_warming_model"] = "none"
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
    assert np.all(np.isfinite(cathode_diag["source_phi_c"]))
    assert np.all(cathode_diag["source_I_i"] >= 0.0)
    assert np.all(cathode_diag["source_I_tot"] >= 0.0)
    assert np.all(np.isfinite(cathode_diag["source_P_prim"]))
    assert np.all(np.isfinite(cathode_diag["source_P_ohmic"]))
    assert np.all(np.isfinite(cathode_diag["source_P_loss"]))
    assert np.all(np.isnan(cathode_diag["end_phi_c"]))
    assert np.all(
        np.isin(
            cathode_diag["source_regime"],
            ["classical", "virtual_cathode", "capability_limited"],
        )
    )
    assert np.all(cathode_diag["end_regime"] == "none")
    assert cathode_diag["beam_cross"].shape == (4, geom.cells)
    # Static T_s is reported as the configured value.
    assert np.allclose(
        cathode_diag["T_s_surface"], float(cathode_run_params["T_s"])
    )

    # --- Power-balance warming (cathode_warming_model="power_balance",
    # CATHODE_IDRIVEN_PLAN.md M1b): the surface energy budget replaces the
    # imposed T_s asymptote. Heater pinned by standby equilibrium; emission
    # cooling uses the actually emitted current.
    from cablp.solvers._sim1d.physics.cathode import (
        cathode_power_balance_terms_W,
    )

    pb_params = dict(cathode_run_params)
    pb_params["cathode_warming_model"] = "power_balance"
    pb_params["cathode_Ts_base_K"] = float(pb_params["T_s"]) - 110.0
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
            float(pb_params["cathode_env_T_K"]),
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
    assert np.all(cathode_run_result.cathode.I_tot[:4] >= 0.0)
    assert np.allclose(
        cathode_run_result.S_ion_beam,
        cathode_run_result.rhs_terms["beam_ionization_birth"]["n"],
    )
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
            assert np.all(h5["cathode_diagnostics/source_I_tot"][()] >= 0.0)
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
    assert retry_result.steps >= 2
    assert np.isclose(retry_result.time[-1], 1.0e-6)
    assert retry_result.diagnostics[0].retry_count >= 1
    assert retry_result.diagnostics[0].rejection_reason == "neutral_step_fraction"
    assert retry_result.diagnostics[0].step_cap == "retry"
    assert retry_result.diagnostics[0].accepted_dt < 1.0e-6
    assert retry_result.diagnostics[1].retry_count == 0
    rejection_count = len(retry_result.timestep_rejection_events["time"])
    assert rejection_count >= 1
    assert set(retry_result.timestep_rejection_events["reason"]) == {
        "neutral_step_fraction"
    }
    retry_summary = summarize_result(retry_result)
    assert retry_summary.step_cap_counts["retry"] == 1
    assert retry_summary.retrying_step_count == 1
    assert retry_summary.total_retry_count == rejection_count
    assert retry_summary.max_retry_count == rejection_count
    assert retry_summary.timestep_rejection_event_count == rejection_count
    with tempfile.TemporaryDirectory() as tmpdir:
        retry_output = retry_sim.save_result(
            f"{tmpdir}/sim1d_retry_smoke.h5",
            retry_result,
        )
        with h5py.File(retry_output, "r") as h5:
            assert h5["timestep_rejection_events/time"].shape == (rejection_count,)
        loaded_retry = load_result_hdf5(retry_output)
        assert np.allclose(
            loaded_retry.timestep_rejection_events["attempted_dt"],
            retry_result.timestep_rejection_events["attempted_dt"],
        )
        assert list(loaded_retry.timestep_rejection_events["reason"]) == list(
            retry_result.timestep_rejection_events["reason"]
        )
        assert loaded_retry.diagnostics[0].retry_count == rejection_count
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
    for values in (
        ramp_state_1.n,
        ramp_state_1.nn,
        ramp_state_1.Ee,
        ramp_state_1.Ei,
        ramp_derived_1.Te,
        ramp_derived_1.Ti,
    ):
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

    # --- Ion-neutral closure knobs: slip drag model, thermalization scale
    # decoupling, Te-shaped cooling corrections. All are default-off (the
    # golden baseline guards the OFF path bit-exactly); these guard the ON
    # paths and the documented limits.
    knob_mass = 4.0 * m_p_cgs
    knob_floors = {"n": 1e6, "nn": 1e8, "Te": 0.1, "Ti": 0.1}
    knob_n = np.array([1e10, 1e12, 1e13])
    knob_state = conservative_from_primitives(
        n=knob_n,
        nn=np.full(3, 1e13),
        u=np.array([2e5, -1e5, 5e4]),
        Te=np.array([6.0, 5.0, 3.0]),
        Ti=np.array([1.0, 1.0, 2.0]),
        ion_mass_g=knob_mass,
    )
    knob_Rm = np.full(3, 50.0)

    # Slip factor: in (0, 1], -> 1 as the plasma rarefies, monotone
    # decreasing with density (denser plasma entrains the neutrals harder).
    slip = ion_neutral_slip_factor(
        n=knob_n,
        Ti=np.array([1.0, 1.0, 1.0]),
        ion_mass_g=knob_mass,
        Rm_cm=knob_Rm,
    )
    assert np.all((slip > 0.0) & (slip <= 1.0))
    assert np.all(np.diff(slip) < 0.0)
    assert slip[0] > 0.95
    # b_slip_entrainment = 0 removes the correction exactly.
    assert np.all(
        ion_neutral_slip_factor(
            n=knob_n,
            Ti=np.array([1.0, 1.0, 1.0]),
            ion_mass_g=knob_mass,
            Rm_cm=knob_Rm,
            b_slip_entrainment=0.0,
        )
        == 1.0
    )

    drag_kwargs = dict(
        state=knob_state,
        floors=knob_floors,
        ion_mass_g=knob_mass,
        gas_type="He",
    )
    # Reference slip factor at the state's own (derived) Ti, as the drag
    # term computes it internally.
    slip_state = ion_neutral_slip_factor(
        n=knob_state.n,
        Ti=derive_state(knob_state, floors=knob_floors, ion_mass_g=knob_mass).Ti,
        ion_mass_g=knob_mass,
        Rm_cm=knob_Rm,
    )
    drag_const = ion_neutral_drag_rhs(**drag_kwargs)
    drag_slip = ion_neutral_drag_rhs(
        **drag_kwargs, drag_model="slip", Rm_cm=knob_Rm
    )
    # Slip only weakens the drag, per cell, by exactly the slip factor.
    assert np.allclose(drag_slip.M, drag_const.M * slip_state, rtol=1e-12)
    assert np.all(np.abs(drag_slip.M) <= np.abs(drag_const.M))
    # Zero-entrainment slip collapses to the constant model bit-exactly.
    assert np.all(
        ion_neutral_drag_rhs(
            **drag_kwargs,
            drag_model="slip",
            Rm_cm=knob_Rm,
            b_slip_entrainment=0.0,
        ).M
        == drag_const.M
    )
    # Frictional heating carries the slip quadratically.
    heat_const = ion_neutral_frictional_heating_rhs(**drag_kwargs)
    heat_slip = ion_neutral_frictional_heating_rhs(
        **drag_kwargs, drag_model="slip", Rm_cm=knob_Rm
    )
    assert np.allclose(heat_slip.Ei, heat_const.Ei * slip_state**2, rtol=1e-12)
    # Unknown model / missing radius fail loudly.
    for bad_kwargs in (
        {"drag_model": "nonsense", "Rm_cm": knob_Rm},
        {"drag_model": "slip"},
    ):
        try:
            ion_neutral_drag_rhs(**drag_kwargs, **bad_kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad_kwargs}")

    # --- Neutral-momentum state foundations (NEUTRAL_MOMENTUM_PLAN.md M1):
    # the optional M_n field must round-trip both packed layouts, pad-on-
    # demand for term summation, refuse to silently drop, and pass floors
    # through untouched.
    mn_s5 = conservative_from_primitives(
        np.full(4, 1e12), np.full(4, 1e13), np.zeros(4),
        np.full(4, 5.0), np.ones(4), knob_mass,
    )
    assert mn_s5.M_n is None and pack_state(mn_s5).size == 20
    mn_s6 = conservative_from_primitives(
        np.full(4, 1e12), np.full(4, 1e13), np.zeros(4),
        np.full(4, 5.0), np.ones(4), knob_mass, un=np.full(4, 1.0e4),
    )
    assert mn_s6.M_n is not None and pack_state(mn_s6).size == 24
    assert unpack_state(pack_state(mn_s5), 4).M_n is None
    mn_rt = unpack_state(pack_state(mn_s6), 4)
    assert mn_rt.M_n is not None and np.all(mn_rt.M_n == mn_s6.M_n)
    padded = pack_state(mn_s5, neutral_momentum=True)
    assert padded.size == 24 and np.all(padded[20:] == 0.0)
    try:
        pack_state(mn_s6, neutral_momentum=False)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError dropping a present M_n")
    mn_fl = apply_state_floors(mn_s6, knob_floors, knob_mass)
    assert mn_fl.M_n is not None and np.all(mn_fl.M_n == mn_s6.M_n)
    mn_sum = add_state_rhs(mn_s5, mn_s6)
    assert mn_sum.M_n is not None and np.all(mn_sum.M_n == mn_s6.M_n)
    assert add_state_rhs(mn_s5, mn_s5).M_n is None

    # --- Two-zone neutral state foundations (NEUTRAL_TWOZONE_PLAN.md M1):
    # the optional nn_a field must round-trip its packed layouts, resolve
    # the 6-field width ambiguity by declared hints (bare 6-field keeps its
    # historical M_n meaning), pad-on-demand, refuse to silently drop, and
    # take the nn floor.
    tz_nn_a = np.full(4, 3.0e12)
    tz_s6 = conservative_from_primitives(
        np.full(4, 1e12), np.full(4, 1e13), np.zeros(4),
        np.full(4, 5.0), np.ones(4), knob_mass, nn_a=tz_nn_a,
    )
    assert tz_s6.nn_a is not None and tz_s6.M_n is None
    tz_packed = pack_state(tz_s6)
    assert tz_packed.size == 24
    # Bare 6-field inference keeps the historical M_n reading...
    tz_bare = unpack_state(tz_packed, 4)
    assert tz_bare.M_n is not None and tz_bare.nn_a is None
    # ...and the declared hint recovers the two-zone layout exactly.
    tz_rt = unpack_state(tz_packed, 4, neutral_two_zone=True)
    assert tz_rt.M_n is None and np.all(tz_rt.nn_a == tz_s6.nn_a)
    tz_rt2 = unpack_state(
        tz_packed, 4, neutral_momentum=False, neutral_two_zone=True
    )
    assert tz_rt2.M_n is None and np.all(tz_rt2.nn_a == tz_s6.nn_a)
    # 7-field (both optionals) round-trips without hints: unambiguous.
    tz_s7 = conservative_from_primitives(
        np.full(4, 1e12), np.full(4, 1e13), np.zeros(4),
        np.full(4, 5.0), np.ones(4), knob_mass,
        un=np.full(4, 1.0e4), nn_a=tz_nn_a,
    )
    tz_p7 = pack_state(tz_s7)
    assert tz_p7.size == 28
    tz_rt7 = unpack_state(tz_p7, 4)
    assert np.all(tz_rt7.M_n == tz_s7.M_n)
    assert np.all(tz_rt7.nn_a == tz_s7.nn_a)
    # Field order is (..., M_n, nn_a): the last row of the 7-field pack is
    # the annulus density.
    assert np.all(tz_p7[24:] == tz_nn_a)
    # Pad-on-demand for term summation, in both flag combinations.
    tz_pad = pack_state(mn_s5, neutral_two_zone=True)
    assert tz_pad.size == 24 and np.all(tz_pad[20:] == 0.0)
    tz_pad7 = pack_state(mn_s6, neutral_two_zone=True)
    assert tz_pad7.size == 28 and np.all(tz_pad7[24:] == 0.0)
    # Refuse to silently drop a present field.
    try:
        pack_state(tz_s6, neutral_two_zone=False)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError dropping a present nn_a")
    # A wrong declared layout is an error, not a silent misread.
    try:
        unpack_state(pack_state(mn_s5), 4, neutral_two_zone=True)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for impossible layout")
    # nn_a is a density: it takes the nn floor; M_n still passes through.
    tz_low = conservative_from_primitives(
        np.full(4, 1e12), np.full(4, 1e13), np.zeros(4),
        np.full(4, 5.0), np.ones(4), knob_mass,
        un=np.full(4, 1.0e4), nn_a=np.zeros(4),
    )
    tz_fl = apply_state_floors(tz_low, knob_floors, knob_mass)
    assert np.all(tz_fl.nn_a == knob_floors["nn"])
    assert np.all(tz_fl.M_n == tz_low.M_n)
    # Add semantics: missing-side-as-zeros, independently per optional field.
    tz_sum = add_state_rhs(mn_s6, tz_s6)
    assert np.all(tz_sum.M_n == mn_s6.M_n)
    assert np.all(tz_sum.nn_a == tz_s6.nn_a)
    assert add_state_rhs(mn_s5, mn_s5).nn_a is None

    # --- Neutral-momentum sources (NEUTRAL_MOMENTUM_PLAN.md M2): with M_n on
    # the state, the drag and the reactions become species-conserving momentum
    # exchanges, the wall and pump are the only named sinks, and the local
    # steady state of drag-vs-wall reproduces the slip closure (with its
    # entrainment scaled by Vp/Vm, since M_n is the chamber-mean wind).
    mn_geom = SimpleNamespace(
        plasma_volume_cm3=np.array([450.0, 900.0, 1800.0]) * 1.0e3,
        neutral_volume_cm3=np.full(3, 5.0e6),
    )
    mn_geom.volume_ratio = (
        mn_geom.plasma_volume_cm3 / mn_geom.neutral_volume_cm3
    )
    knob_u = np.array([2e5, -1e5, 5e4])
    mn_state = conservative_from_primitives(
        n=knob_n,
        nn=np.full(3, 1e13),
        u=knob_u,
        Te=np.array([6.0, 5.0, 3.0]),
        Ti=np.array([1.0, 1.0, 2.0]),
        ion_mass_g=knob_mass,
        un=0.3 * knob_u,
    )
    assert np.allclose(
        neutral_wind_velocity(mn_state, knob_floors, knob_mass),
        0.3 * knob_u,
        rtol=1e-14,
    )
    assert np.all(
        neutral_wind_velocity(knob_state, knob_floors, knob_mass) == 0.0
    )

    mn_drag_kwargs = dict(drag_kwargs, state=mn_state, geometry=mn_geom)
    mn_drag = ion_neutral_drag_rhs(**mn_drag_kwargs)
    # The exchange closes: what the plasma channel loses, the neutral channel
    # gains, through the (Vp/Vm) volume conversion (to rounding: the float
    # round-trip (Vp/Vm)*Vm is one ulp off Vp).
    assert np.allclose(
        mn_drag.M * mn_geom.plasma_volume_cm3,
        -mn_drag.M_n * mn_geom.neutral_volume_cm3,
        rtol=1e-12,
    )
    # The ion-side force is the constant-model drag on the relative velocity:
    # u_n = 0.3*u everywhere makes it exactly 0.7x the u-based drag.
    assert np.allclose(mn_drag.M, 0.7 * drag_const.M, rtol=1e-12)
    # A zero wind reproduces the constant model bit-exactly on the ion side.
    mn_state_rest = conservative_from_primitives(
        n=knob_n,
        nn=np.full(3, 1e13),
        u=knob_u,
        Te=np.array([6.0, 5.0, 3.0]),
        Ti=np.array([1.0, 1.0, 2.0]),
        ion_mass_g=knob_mass,
        un=np.zeros(3),
    )
    assert np.all(
        ion_neutral_drag_rhs(
            **dict(drag_kwargs, state=mn_state_rest, geometry=mn_geom)
        ).M
        == drag_const.M
    )
    # slip closure and evolved wind are mutually exclusive; geometry is
    # required for the volume conversion.
    for mn_bad in (
        {"drag_model": "slip", "Rm_cm": knob_Rm, "geometry": mn_geom},
        {"geometry": None},
    ):
        try:
            ion_neutral_drag_rhs(**dict(drag_kwargs, state=mn_state), **mn_bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {mn_bad}")
    # Frictional heating dissipates the relative velocity quadratically:
    # 0.7^2 = 0.49x the u-based heating, and bit-exact at zero wind.
    assert np.allclose(
        ion_neutral_frictional_heating_rhs(
            **dict(drag_kwargs, state=mn_state)
        ).Ei,
        0.49 * heat_const.Ei,
        rtol=1e-12,
    )
    assert np.all(
        ion_neutral_frictional_heating_rhs(
            **dict(drag_kwargs, state=mn_state_rest)
        ).Ei
        == heat_const.Ei
    )

    # Wall sink: -M_n / tau_wall on the momentum field only; inert without M_n.
    mn_wall = neutral_momentum_wall_rhs(
        state=mn_state,
        floors=knob_floors,
        ion_mass_g=knob_mass,
        Rm_cm=knob_Rm,
    )
    mn_vbar = np.sqrt(8.0 * 0.1 * ev_to_erg / (np.pi * knob_mass))
    assert np.allclose(
        mn_wall.M_n, -mn_state.M_n * mn_vbar / knob_Rm, rtol=1e-14
    )
    for mn_field in STATE_NAMES_1D:
        assert np.all(getattr(mn_wall, mn_field) == 0.0)
    assert (
        neutral_momentum_wall_rhs(
            state=knob_state,
            floors=knob_floors,
            ion_mass_g=knob_mass,
            Rm_cm=knob_Rm,
        ).M_n
        is None
    )

    # Reactions: ionization births ions drifting at u_n (taking that momentum
    # out of the wind), recombination hands the ion's momentum to the wind;
    # each term closes M*Vp + M_n*Vm exactly.
    mn_reactions = reaction_rhs_terms(
        state=mn_state,
        floors=knob_floors,
        ion_mass_g=knob_mass,
        geometry=mn_geom,
        gas_type="He",
        I_ion=I_ion,
    )
    for mn_name in (
        "ionization_birth",
        "recombination_rad_loss",
        "recombination_3b_loss",
    ):
        mn_term = mn_reactions[mn_name]
        assert mn_term.M_n is not None
        assert np.allclose(
            mn_term.M * mn_geom.plasma_volume_cm3,
            -mn_term.M_n * mn_geom.neutral_volume_cm3,
            rtol=1e-12,
        )
    assert np.allclose(
        mn_reactions["ionization_birth"].M,
        knob_mass * 0.3 * knob_u * mn_reactions["ionization_birth"].n,
        rtol=1e-12,
    )
    mn_rec = mn_reactions["recombination_rad_loss"]
    mn_u = derive_state(mn_state, knob_floors, knob_mass).u
    assert np.allclose(
        mn_rec.M, knob_mass * mn_u * mn_rec.n, rtol=1e-12
    )
    # Without M_n the reaction terms stay 5-field with zero-drift birth.
    assert reaction_rhs_terms(
        state=knob_state,
        floors=knob_floors,
        ion_mass_g=knob_mass,
        geometry=mn_geom,
        gas_type="He",
        I_ion=I_ion,
    )["ionization_birth"].M_n is None

    # Local steady state of drag reception vs. the wall sink IS the slip
    # closure, with the entrainment scaled by the chamber-mean factor Vp/Vm.
    mn_relax = np.zeros(3)
    mn_dt = 1.0e-5
    for _ in range(3000):
        mn_st = ConservativeState1D(
            n=mn_state.n,
            nn=mn_state.nn,
            M=mn_state.M,
            Ee=mn_state.Ee,
            Ei=mn_state.Ei,
            M_n=mn_relax,
        )
        mn_relax = mn_relax + mn_dt * (
            ion_neutral_drag_rhs(
                **dict(drag_kwargs, state=mn_st, geometry=mn_geom)
            ).M_n
            + neutral_momentum_wall_rhs(
                state=mn_st,
                floors=knob_floors,
                ion_mass_g=knob_mass,
                Rm_cm=knob_Rm,
            ).M_n
        )
    mn_un_ss = mn_relax / (knob_mass * mn_state.nn)
    mn_Ti = derive_state(mn_state, knob_floors, knob_mass).Ti
    mn_nu_ni = ion_neutral_collision_frequency(
        nn=mn_state.n,
        Ti=mn_Ti,
        ion_mass_g=knob_mass,
        gas_type="He",
    )
    mn_E = mn_geom.volume_ratio * mn_nu_ni * knob_Rm / mn_vbar
    assert np.allclose(mn_un_ss / mn_u, mn_E / (1.0 + mn_E), rtol=1e-5)
    for mn_i in range(3):
        mn_slip = ion_neutral_slip_factor(
            n=mn_state.n[mn_i],
            Ti=mn_Ti[mn_i],
            ion_mass_g=knob_mass,
            Rm_cm=knob_Rm[mn_i],
            b_slip_entrainment=mn_geom.volume_ratio[mn_i],
        )
        assert np.isclose(
            1.0 - mn_un_ss[mn_i] / mn_u[mn_i], mn_slip, rtol=1e-5
        )

    # Pump sink: the wind leaves with the gas at the pump cells, so the
    # pumped momentum fraction matches the pumped particle fraction there.
    from cablp.solvers._sim1d.core.geometry import build_geometry

    mn_pump_geom = build_geometry(*default_config())
    mn_pump_cells = mn_pump_geom.cells
    mn_pump_state = conservative_from_primitives(
        n=np.full(mn_pump_cells, 1e12),
        nn=np.full(mn_pump_cells, 1e13),
        u=np.zeros(mn_pump_cells),
        Te=np.full(mn_pump_cells, 5.0),
        Ti=np.full(mn_pump_cells, 1.0),
        ion_mass_g=knob_mass,
        un=np.full(mn_pump_cells, 2.0e4),
    )
    mn_pump = neutral_source_sink_rhs(
        state=mn_pump_state,
        geometry=mn_pump_geom,
        S_gp=0.0,
        Twin_S_gp=0.0,
        S_pump_L=500.0,
        S_pump_R=500.0,
        gas_puff_enabled=False,
        pump_enabled=True,
    )
    mn_pump_mask = mn_pump.nn != 0.0
    assert np.any(mn_pump_mask)
    assert np.all(mn_pump.M_n[~mn_pump_mask] == 0.0)
    assert np.allclose(
        mn_pump.M_n[mn_pump_mask],
        mn_pump.nn[mn_pump_mask]
        * mn_pump_state.M_n[mn_pump_mask]
        / mn_pump_state.nn[mn_pump_mask],
        rtol=1e-12,
    )
    # Puff-only sources add cold gas: no momentum contribution at all.
    assert np.all(
        neutral_source_sink_rhs(
            state=mn_pump_state,
            geometry=mn_pump_geom,
            S_gp=100.0,
            Twin_S_gp=0.0,
            S_pump_L=500.0,
            S_pump_R=500.0,
            gas_puff_enabled=True,
            pump_enabled=False,
        ).M_n
        == 0.0
    )

    # Solver plumbing: the flag builds and carries the 6-field state, the
    # wall term appears in the rhs, a run saves/loads the optional M_n and
    # u_n trajectories, and the slip closure is rejected loudly.
    mn_flags = dict(flags)
    mn_flags["neutral_momentum"] = True
    # The evolved neutral wind (M_n) is driven through the legacy ion_neutral
    # drag term, which the Phelps moment operator (production default) gates off.
    # This M_n block is a DEPRECATED A/B path (the M_n-on-Phelps rewiring is the
    # deferred neutral ladder), so run it with the moment operator off -- the
    # legacy-drag DeprecationWarning is expected.
    mn_flags["ion_neutral_moment_closure"] = False
    try:
        LAPDSim1D(
            dict(params, ion_neutral_drag_model="slip"), mn_flags
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected neutral_momentum x slip to fail")
    mn_run_params = dict(run_params)
    mn_sim = LAPDSim1D(mn_run_params, mn_flags)
    assert mn_sim.state.M_n is not None
    mn_cells = mn_sim.geometry.cells
    assert mn_sim.get_initial_snapshot().y.size == 6 * mn_cells
    mn_rhs_terms = mn_sim.rhs_terms()
    assert set(mn_rhs_terms) == expected_rhs_terms
    assert mn_sim.rhs().size == 6 * mn_cells
    mn_result = mn_sim.run(t_end=3.0e-10, dt=1.0e-10)
    assert mn_result.M_n.shape == (4, mn_cells)
    assert mn_result.u_n.shape == (4, mn_cells)
    assert np.all(np.isfinite(mn_result.M_n))
    with tempfile.TemporaryDirectory() as mn_dir:
        mn_path = Path(mn_dir) / "mn_smoke.h5"
        mn_sim.save_result(mn_path, mn_result)
        mn_loaded = load_result_hdf5(mn_path)
        assert np.allclose(mn_loaded.M_n, mn_result.M_n)
        assert np.allclose(mn_loaded.u_n, mn_result.u_n)

    # Plasma-phase end-to-end: with a flowing plasma the drag pumps the wind
    # up from zero through the full step machinery (explicit substep,
    # implicit heat with the 6-field state, floors, step acceptance).
    mn_plasma_flags = dict(mn_flags)
    mn_plasma_flags["neutral_prebreakdown"] = False
    mn_plasma_flags["neutral_equilibration"] = False
    mn_plasma_params = dict(params)
    mn_plasma_params["u0"] = 5.0e4
    mn_plasma_sim = LAPDSim1D(mn_plasma_params, mn_plasma_flags)
    mn_plasma_terms = mn_plasma_sim.rhs_terms()
    assert mn_plasma_terms["ion_neutral_drag"].M_n is not None
    assert np.any(mn_plasma_terms["ion_neutral_drag"].M_n != 0.0)
    assert mn_plasma_sim.rhs().size == 6 * mn_plasma_sim.geometry.cells
    for _ in range(5):
        mn_plasma_sim.advance_one_step(dt=1.0e-9)
    mn_plasma_state = mn_plasma_sim.state
    assert mn_plasma_state.M_n is not None
    assert np.all(np.isfinite(mn_plasma_state.M_n))
    assert np.any(mn_plasma_state.M_n != 0.0)
    # The wind chases the plasma flow: same sign where it has spun up.
    mn_drive = mn_plasma_state.M_n * derive_state(
        mn_plasma_state, mn_plasma_sim.floors, mn_plasma_sim.ion_mass_g
    ).u
    assert np.all(mn_drive[mn_plasma_state.M_n != 0.0] > 0.0)

    # --- Two-zone radial closure (neutral_momentum_radial = "two_zone"): the
    # drag samples the in-column wind, and only the slow annulus gas -- held
    # back by diffuse wall reflection -- reaches the wall. The factors reduce
    # to closed form: r = Rp/(Rp+Rm), c = 1/(f + (1-f) r) with f = (Rp/Rm)^2,
    # W = vbar/(2 Rm) * r * c; a cell without an annulus (Rp >= Rm) falls
    # back to the uniform closure.
    tz_geom = SimpleNamespace(
        Rp_cm=np.array([15.0, 18.0, 50.0]),
        Rm_cm=np.full(3, 50.0),
    )
    tz_c, tz_W = neutral_wind_two_zone_factors(tz_geom, 0.1, knob_mass)
    tz_f = (tz_geom.Rp_cm / tz_geom.Rm_cm) ** 2
    tz_r = tz_geom.Rp_cm / (tz_geom.Rp_cm + tz_geom.Rm_cm)
    tz_c_hand = 1.0 / (tz_f + (1.0 - tz_f) * tz_r)
    tz_W_hand = mn_vbar / (2.0 * tz_geom.Rm_cm) * tz_r * tz_c_hand
    assert np.allclose(tz_c[:2], tz_c_hand[:2], rtol=1e-13)
    assert np.allclose(tz_W[:2], tz_W_hand[:2], rtol=1e-13)
    assert tz_c[2] == 1.0
    assert np.isclose(tz_W[2], mn_vbar / 50.0, rtol=1e-13)
    # The column factor concentrates the wind (c > 1) yet the effective wall
    # rate is *weaker* than uniform: the wall only sees the slow annulus.
    assert np.all(tz_c[:2] > 1.0)
    assert np.all(tz_W[:2] < mn_vbar / tz_geom.Rm_cm[:2])

    # Drag with the column factor: the exchange still closes exactly, and the
    # ion side sees u - c*u_n (u_n = 0.3*u here, so 1 - 0.3*c per cell --
    # including a sign flip where the column wind overtakes the ions).
    tz_factor = np.array([2.0, 3.0, 4.0])
    tz_drag = ion_neutral_drag_rhs(
        **dict(mn_drag_kwargs, wind_column_factor=tz_factor)
    )
    assert np.allclose(
        tz_drag.M * mn_geom.plasma_volume_cm3,
        -tz_drag.M_n * mn_geom.neutral_volume_cm3,
        rtol=1e-12,
    )
    assert np.allclose(
        tz_drag.M, (1.0 - 0.3 * tz_factor) * drag_const.M, rtol=1e-12
    )
    # Frictional heating carries the factor quadratically.
    assert np.allclose(
        ion_neutral_frictional_heating_rhs(
            **dict(drag_kwargs, state=mn_state, wind_column_factor=tz_factor)
        ).Ei,
        (1.0 - 0.3 * tz_factor) ** 2 * heat_const.Ei,
        rtol=1e-12,
    )
    # Wall sink with an explicit two-zone rate.
    assert np.allclose(
        neutral_momentum_wall_rhs(
            state=mn_state,
            floors=knob_floors,
            ion_mass_g=knob_mass,
            Rm_cm=knob_Rm,
            wall_rate_1_s=tz_W,
        ).M_n,
        -mn_state.M_n * tz_W,
        rtol=1e-13,
    )
    # Ionization birth samples the column wind: the ion-side momentum scales
    # by the factor and the exchange still closes exactly.
    tz_birth = reaction_rhs_terms(
        state=mn_state,
        floors=knob_floors,
        ion_mass_g=knob_mass,
        geometry=mn_geom,
        gas_type="He",
        I_ion=I_ion,
        wind_column_factor=tz_factor,
    )["ionization_birth"]
    assert np.allclose(
        tz_birth.M,
        tz_factor * mn_reactions["ionization_birth"].M,
        rtol=1e-12,
    )
    assert np.allclose(
        tz_birth.M * mn_geom.plasma_volume_cm3,
        -tz_birth.M_n * mn_geom.neutral_volume_cm3,
        rtol=1e-12,
    )

    # Relaxed local steady state of drag-vs-wall under the two-zone factors
    # matches the analytic balance u_mean = A*u / (W + c*A) with
    # A = (Vp/Vm)*nu_ni (drag reception now pushes against c*u_mean).
    tz_c3 = np.array([3.0, 2.5, 2.0])
    tz_W3 = np.array([2.0e3, 1.5e3, 1.0e3])
    tz_relax = np.zeros(3)
    tz_dt = 2.5e-6
    for _ in range(12000):
        tz_st = ConservativeState1D(
            n=mn_state.n,
            nn=mn_state.nn,
            M=mn_state.M,
            Ee=mn_state.Ee,
            Ei=mn_state.Ei,
            M_n=tz_relax,
        )
        tz_relax = tz_relax + tz_dt * (
            ion_neutral_drag_rhs(
                **dict(
                    drag_kwargs,
                    state=tz_st,
                    geometry=mn_geom,
                    wind_column_factor=tz_c3,
                )
            ).M_n
            + neutral_momentum_wall_rhs(
                state=tz_st,
                floors=knob_floors,
                ion_mass_g=knob_mass,
                Rm_cm=knob_Rm,
                wall_rate_1_s=tz_W3,
            ).M_n
        )
    tz_un_ss = tz_relax / (knob_mass * mn_state.nn)
    tz_A = mn_geom.volume_ratio * mn_nu_ni
    assert np.allclose(
        tz_un_ss, tz_A * mn_u / (tz_W3 + tz_c3 * tz_A), rtol=1e-5
    )

    # Solver wiring: the config key is validated, requires the flag, and the
    # two-zone wall rate shows up in the named rhs term.
    for tz_bad_params, tz_bad_flags in (
        (dict(run_params, neutral_momentum_radial="two_zone"), flags),
        (dict(run_params, neutral_momentum_radial="bogus"), mn_flags),
    ):
        try:
            LAPDSim1D(tz_bad_params, tz_bad_flags)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"expected ValueError for {tz_bad_params['neutral_momentum_radial']}"
            )
    tz_plasma_params = dict(mn_plasma_params, neutral_momentum_radial="two_zone")
    tz_sim = LAPDSim1D(tz_plasma_params, mn_plasma_flags)
    for _ in range(5):
        tz_sim.advance_one_step(dt=1.0e-9)
    tz_sim_state = tz_sim.state
    assert np.all(np.isfinite(tz_sim_state.M_n))
    assert np.any(tz_sim_state.M_n != 0.0)
    tz_geo_c, tz_geo_W = neutral_wind_two_zone_factors(
        tz_sim.geometry,
        float(tz_plasma_params.get("Tn_fit", 0.1)),
        tz_sim.ion_mass_g,
    )
    assert np.allclose(
        tz_sim.rhs_terms()["neutral_momentum_wall"].M_n,
        -tz_sim_state.M_n * tz_geo_W,
        rtol=1e-12,
    )
    # Against the uniform closure from the same start the trajectories must
    # differ (both the drag input and the wall sink change). No per-cell
    # direction is asserted: the weaker two-zone wall sink can retain *more*
    # momentum in cells where the sink dominates the (also weaker) input --
    # only the local steady state (checked analytically above) is ordered.
    tz_uniform_sim = LAPDSim1D(dict(mn_plasma_params), mn_plasma_flags)
    for _ in range(5):
        tz_uniform_sim.advance_one_step(dt=1.0e-9)
    tz_nonzero = tz_sim_state.M_n != 0.0
    assert np.any(tz_nonzero)
    assert np.any(tz_sim_state.M_n != tz_uniform_sim.state.M_n)

    # --- Two-zone PARTICLE channel, M2 carriage and transport
    # (NEUTRAL_TWOZONE_PLAN.md): the solver carries the split (nn, nn_a)
    # state, runs per-zone axial Knudsen exchange plus the radial
    # column/annulus conductance, and both close inventory exactly with
    # detailed balance at equal densities. The flag requires the knudsen
    # exchange model.
    p2z_params = dict(mn_plasma_params)
    p2z_params["neutral_exchange_model"] = "knudsen"
    p2z_flags = dict(mn_plasma_flags)
    p2z_flags["neutral_momentum"] = False
    p2z_flags["neutral_two_zone"] = True
    try:
        LAPDSim1D(
            dict(mn_plasma_params, neutral_exchange_model="constant"),
            p2z_flags,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "expected ValueError: neutral_two_zone without knudsen exchange"
        )
    p2z_sim = LAPDSim1D(p2z_params, p2z_flags)
    p2z_state = p2z_sim.state
    assert p2z_state.nn_a is not None and p2z_state.M_n is None
    assert p2z_sim.rhs().size == 6 * p2z_sim.geometry.cells
    p2z_Vc, p2z_Va = neutral_zone_volumes(p2z_sim.geometry)
    assert np.allclose(
        p2z_Vc + p2z_Va, p2z_sim.geometry.neutral_volume_cm3, rtol=1e-13
    )
    # Conductance arithmetic against the closed forms.
    p2z_vth = neutral_thermal_speed(
        float(p2z_params.get("Tn_K", 300.0)), p2z_sim.mu
    )
    p2z_geom = p2z_sim.geometry
    p2z_mid = p2z_geom.cells // 2
    assert np.isclose(
        neutral_zone_exchange_conductance(
            p2z_geom, float(p2z_params.get("Tn_K", 300.0)), p2z_sim.mu
        )[p2z_mid],
        0.25
        * p2z_vth
        * 2.0
        * np.pi
        * p2z_geom.Rp_cm[p2z_mid]
        * p2z_geom.length_cm[p2z_mid],
        rtol=1e-13,
    )
    p2z_cc, p2z_ca = two_zone_knudsen_coefficients(
        p2z_geom, float(p2z_params.get("Tn_K", 300.0)), p2z_sim.mu
    )
    p2z_Rcol = 0.5 * (p2z_geom.Rp_cm[p2z_mid] + p2z_geom.Rp_cm[p2z_mid + 1])
    p2z_Rann = 0.5 * (
        p2z_geom.Rm_cm[p2z_mid] + p2z_geom.Rm_cm[p2z_mid + 1]
    ) - p2z_Rcol
    assert np.isclose(
        p2z_cc[p2z_mid],
        (2.0 / 3.0)
        * p2z_vth
        * p2z_Rcol
        * min(
            p2z_geom.plasma_area_cm2[p2z_mid],
            p2z_geom.plasma_area_cm2[p2z_mid + 1],
        )
        / p2z_geom.center_distance_cm[p2z_mid],
        rtol=1e-13,
    )
    p2z_ann_area = (
        p2z_geom.neutral_area_cm2 - p2z_geom.plasma_area_cm2
    )
    assert np.isclose(
        p2z_ca[p2z_mid],
        (2.0 / 3.0)
        * p2z_vth
        * p2z_Rann
        * min(p2z_ann_area[p2z_mid], p2z_ann_area[p2z_mid + 1])
        / p2z_geom.center_distance_cm[p2z_mid],
        rtol=1e-13,
    )
    # Detailed balance: the uniform initial state gives exactly zero for
    # both exchange terms.
    assert np.all(p2z_sim.neutral_zone_exchange_rhs(state=p2z_state).nn == 0.0)
    assert np.all(
        p2z_sim.neutral_zone_exchange_rhs(state=p2z_state).nn_a == 0.0
    )
    # Perturbed: exact inventory closure per term, and the net flux refills
    # the depleted column from the annulus.
    p2z_nn = p2z_state.nn.copy()
    p2z_nn[p2z_mid] *= 0.5
    p2z_pert = ConservativeState1D(
        p2z_state.n,
        p2z_nn,
        p2z_state.M,
        p2z_state.Ee,
        p2z_state.Ei,
        nn_a=p2z_state.nn_a.copy(),
    )
    p2z_zx = p2z_sim.neutral_zone_exchange_rhs(state=p2z_pert)
    assert (
        abs(float((p2z_zx.nn * p2z_Vc + p2z_zx.nn_a * p2z_Va).sum())) == 0.0
    )
    assert p2z_zx.nn[p2z_mid] > 0.0 and p2z_zx.nn_a[p2z_mid] < 0.0
    p2z_ax = p2z_sim.neutral_exchange_rhs(state=p2z_pert)
    assert abs(float((p2z_ax.nn * p2z_Vc).sum())) <= 1e-12 * float(
        np.abs(p2z_ax.nn * p2z_Vc).max()
    )
    assert np.all(p2z_ax.nn_a == 0.0)  # annulus still uniform
    # The pre-plasma implicit step conserves inventory exactly with the
    # pump off and the puff's BE deposit accounted, and preserves the
    # uniform equilibrium.
    p2z_eq_params = dict(p2z_params)
    p2z_eq_params["pump_enabled"] = False
    p2z_eq_flags = dict(p2z_flags)
    p2z_eq_flags["Plasma"] = False
    p2z_eq_sim = LAPDSim1D(p2z_eq_params, p2z_eq_flags)
    p2z_eq_state = p2z_eq_sim.state
    p2z_eq_Vc, p2z_eq_Va = neutral_zone_volumes(p2z_eq_sim.geometry)
    p2z_dt = 1.0e-5
    p2z_next = p2z_eq_sim._implicit_neutral_step(
        dt=p2z_dt, state=p2z_eq_state, time=0.0
    )
    p2z_src = p2z_eq_sim._neutral_source_kwargs(time=0.0)
    p2z_inflow = 0.0
    if p2z_src["gas_puff_enabled"]:
        p2z_inflow = float(
            np.sum(
                gas_puff_rate_profile(
                    p2z_eq_sim.geometry,
                    p2z_src["S_gp"],
                    p2z_src["gas_puff_valves"],
                    profile=p2z_src["gas_puff_profile"],
                    z_cm=p2z_src["gas_puff_z_cm"],
                    sigma_cm=p2z_src["gas_puff_sigma_cm"],
                    throw_cm=p2z_src["gas_puff_throw_cm"],
                )
                * p2z_eq_sim.geometry.neutral_volume_cm3
            )
        )
    p2z_inv0 = float(
        (p2z_eq_state.nn * p2z_eq_Vc + p2z_eq_state.nn_a * p2z_eq_Va).sum()
    )
    p2z_inv1 = float(
        (p2z_next.nn * p2z_eq_Vc + p2z_next.nn_a * p2z_eq_Va).sum()
    )
    assert np.isclose(p2z_inv1 - p2z_inv0, p2z_dt * p2z_inflow, rtol=1e-9)
    # Plasma-phase e2e through the full step machinery, two-zone alone and
    # combined with the evolved wind (7-field state).
    for _ in range(5):
        p2z_sim.advance_one_step(dt=1.0e-9)
    p2z_after = p2z_sim.state
    assert p2z_after.nn_a is not None
    assert np.all(np.isfinite(p2z_after.nn_a))
    p2z_both_flags = dict(p2z_flags)
    p2z_both_flags["neutral_momentum"] = True
    p2z_both_sim = LAPDSim1D(p2z_params, p2z_both_flags)
    assert p2z_both_sim.rhs().size == 7 * p2z_both_sim.geometry.cells
    for _ in range(5):
        p2z_both_sim.advance_one_step(dt=1.0e-9)
    p2z_both_state = p2z_both_sim.state
    assert p2z_both_state.M_n is not None
    assert p2z_both_state.nn_a is not None
    assert np.all(np.isfinite(p2z_both_state.M_n))
    assert np.all(np.isfinite(p2z_both_state.nn_a))

    # --- Kinetic-derived two-momentum reduction (M6): the selector is
    # default-off and requires both optional parent fields. The eighth packed
    # row is annulus momentum; radial exchange closes Mc*Vc + Ma*Va exactly,
    # drag closes M*Vp + Mc*Vc exactly, and a short full-solver trajectory
    # keeps the optional layout finite.
    for m6_bad_params, m6_bad_flags in (
        (
            dict(
                p2z_params,
                neutral_momentum_radial="kinetic_two_moment",
            ),
            p2z_flags,
        ),
        (
            dict(
                mn_plasma_params,
                neutral_momentum_radial="kinetic_two_moment",
            ),
            mn_plasma_flags,
        ),
    ):
        try:
            LAPDSim1D(m6_bad_params, m6_bad_flags)
        except ValueError:
            pass
        else:
            raise AssertionError(
                "expected kinetic_two_moment parent-field guard"
            )
    m6_params = dict(
        p2z_params,
        neutral_momentum_radial="kinetic_two_moment",
    )
    m6_flags = dict(p2z_both_flags)
    m6_sim = LAPDSim1D(m6_params, m6_flags)
    m6_state0 = m6_sim.state
    assert m6_state0.M_n_a is not None
    assert m6_sim.rhs().size == 8 * m6_sim.geometry.cells
    m6_roundtrip = unpack_state(
        pack_state(m6_state0),
        m6_sim.geometry.cells,
        neutral_momentum=True,
        neutral_two_zone=True,
        neutral_annulus_momentum=True,
    )
    assert np.array_equal(m6_roundtrip.M_n_a, m6_state0.M_n_a)
    m6_Mc = np.full(m6_sim.geometry.cells, 2.0e-8)
    m6_exchange_state = ConservativeState1D(
        n=m6_state0.n,
        nn=m6_state0.nn,
        M=m6_state0.M,
        Ee=m6_state0.Ee,
        Ei=m6_state0.Ei,
        M_n=m6_Mc,
        nn_a=m6_state0.nn_a,
        M_n_a=np.zeros_like(m6_Mc),
    )
    m6_radial = neutral_momentum_two_zone_rhs(
        state=m6_exchange_state,
        floors=m6_sim.floors,
        ion_mass_g=m6_sim.ion_mass_g,
        geometry=m6_sim.geometry,
        Tn_K=float(m6_params.get("Tn_K", 300.0)),
    )
    m6_Vc, m6_Va = neutral_zone_volumes(m6_sim.geometry)
    m6_radial_inventory = (
        m6_radial.M_n * m6_Vc + m6_radial.M_n_a * m6_Va
    )
    m6_radial_scale = np.max(np.abs(m6_radial.M_n * m6_Vc))
    assert np.max(np.abs(m6_radial_inventory)) <= 1e-14 * m6_radial_scale
    m6_drag = m6_sim.ion_neutral_drag_rhs(state=m6_exchange_state)
    assert np.array_equal(
        m6_drag.M * m6_Vc + m6_drag.M_n * m6_Vc,
        np.zeros_like(m6_Vc),
    )
    m6_beam_base = cathode_sim.state
    m6_beam_Mc = cathode_sim.ion_mass_g * m6_beam_base.nn * 1.0e5
    m6_beam_state = ConservativeState1D(
        n=m6_beam_base.n,
        nn=m6_beam_base.nn,
        M=m6_beam_base.M,
        Ee=m6_beam_base.Ee,
        Ei=m6_beam_base.Ei,
        M_n=m6_beam_Mc,
        nn_a=m6_beam_base.nn.copy(),
        M_n_a=np.zeros_like(m6_beam_Mc),
    )
    m6_beam_birth = cathode_sim.beam_ionization_rhs_terms(
        state=m6_beam_state,
        cathode_solve=cathode_solve,
    )["beam_ionization_birth"]
    assert np.any(m6_beam_birth.n > 0.0)
    m6_beam_Vc = cathode_sim.geometry.plasma_volume_cm3
    assert np.allclose(
        m6_beam_birth.M * m6_beam_Vc
        + m6_beam_birth.M_n * m6_beam_Vc,
        0.0,
        rtol=1e-14,
        atol=0.0,
    )
    assert np.all(m6_beam_birth.M_n_a == 0.0)
    assert np.all(
        m6_sim.neutral_momentum_wall_rhs(
            state=m6_exchange_state
        ).M_n == 0.0
    )
    for _ in range(5):
        m6_sim.advance_one_step(dt=1.0e-9)
    assert np.all(np.isfinite(m6_sim.state.M_n))
    assert np.all(np.isfinite(m6_sim.state.M_n_a))
    m6_io_sim = LAPDSim1D(m6_params, m6_flags)
    m6_result = m6_io_sim.run(t_end=3.0e-10, dt=1.0e-10)
    assert m6_result.M_n_a.shape[1] == m6_io_sim.geometry.cells
    assert m6_result.M_n_a.shape[0] >= 2
    assert np.all(np.isfinite(m6_result.u_n_a))
    with tempfile.TemporaryDirectory() as m6_dir:
        m6_path = Path(m6_dir) / "m6_smoke.h5"
        m6_io_sim.save_result(m6_path, m6_result)
        m6_loaded = load_result_hdf5(m6_path)
        assert np.allclose(m6_loaded.M_n_a, m6_result.M_n_a)
        assert np.allclose(m6_loaded.u_n_a, m6_result.u_n_a)
    # M3 source/sink routing: with nn the column density on V_col == Vp,
    # every species-exchange term closes the TOTAL particle inventory
    # n*Vp + nn*V_col + nn_a*V_ann exactly (the ionization/recombination
    # conversion is unity; recycle faces feed the column; the puff feeds
    # the annulus; the pump drains both zones).
    p2z_terms = p2z_sim.rhs_terms()
    assert np.all(
        p2z_terms["ionization_birth"].nn == -p2z_terms["ionization_birth"].n
    )
    def p2z_inventory(term):
        total = term.n * p2z_sim.geometry.plasma_volume_cm3 + term.nn * p2z_Vc
        if term.nn_a is not None:
            total = total + term.nn_a * p2z_Va
        return float(np.sum(total))
    for p2z_name in (
        "ionization_birth",
        "recombination_rad_loss",
        "recombination_3b_loss",
        "surface_loss",
        "neutral_exchange",
        "neutral_zone_exchange",
    ):
        p2z_term = p2z_terms[p2z_name]
        p2z_scale = float(
            np.abs(p2z_term.n * p2z_sim.geometry.plasma_volume_cm3).max()
            + np.abs(p2z_term.nn * p2z_Vc).max()
        )
        assert abs(p2z_inventory(p2z_term)) <= 1e-10 * max(p2z_scale, 1.0), (
            p2z_name
        )
    # The term ledger gains exactly the one new named term, and the nn_a
    # trajectory round-trips through HDF5.
    p2z_run_params = dict(run_params)
    p2z_run_params["neutral_exchange_model"] = "knudsen"
    p2z_run_flags = dict(flags)
    p2z_run_flags["neutral_two_zone"] = True
    p2z_run_sim = LAPDSim1D(p2z_run_params, p2z_run_flags)
    assert set(p2z_run_sim.rhs_terms()) == expected_rhs_terms | {
        "neutral_zone_exchange"
    }
    # Resolved geometry + operator-split implicit heat + both optional
    # fields: the heat substep must pass nn_a through (the strict unpack
    # hints turn a dropped field into a loud error -- this caught a real
    # 7->6-field truncation in implicit_heat_conduction_step).
    p2z_res_params = dict(p2z_params)
    p2z_res_flags = dict(p2z_flags)
    p2z_res_flags["neutral_momentum"] = True
    p2z_res_flags["resolved_boundaries"] = True
    p2z_res_flags["implicit_heat_conduction"] = True
    p2z_res_sim = LAPDSim1D(p2z_res_params, p2z_res_flags)
    for _ in range(3):
        p2z_res_sim.advance_one_step(dt=1.0e-9)
    p2z_res_state = p2z_res_sim.state
    assert p2z_res_state.nn_a is not None and p2z_res_state.M_n is not None
    assert np.all(np.isfinite(p2z_res_state.nn_a))
    assert np.all(np.isfinite(p2z_res_state.M_n))

    p2z_result = p2z_run_sim.run(t_end=3.0e-10, dt=1.0e-10)
    p2z_run_cells = p2z_run_sim.geometry.cells
    assert p2z_result.nn_a.shape == (4, p2z_run_cells)
    assert np.all(np.isfinite(p2z_result.nn_a))
    with tempfile.TemporaryDirectory() as p2z_dir:
        p2z_path = Path(p2z_dir) / "p2z_smoke.h5"
        p2z_run_sim.save_result(p2z_path, p2z_result)
        p2z_loaded = load_result_hdf5(p2z_path)
        assert np.allclose(p2z_loaded.nn_a, p2z_result.nn_a)

    # --- K4a kinetic neutrals (KINETIC_TWOZONE_PLAN.md): the refresh-
    # cadence relaxation architecture. The flag requires the two-zone
    # state; targets appear at the first accepted plasma step; every
    # superseded term's neutral rows are zeroed while its plasma rows
    # keep their exact forms; the relaxation key is present from the
    # start so the saved ledger structure is stable.
    try:
        LAPDSim1D(
            dict(p2z_params, neutral_model="kinetic"),
            dict(p2z_flags, neutral_two_zone=False),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected kinetic-without-two-zone to fail")
    k4_params = dict(p2z_params)
    k4_params["neutral_model"] = "kinetic"
    k4_params["neutral_kinetic_refresh_s"] = 2e-4
    k4_params["neutral_kinetic_nvz"] = 24
    k4_params["neutral_kinetic_nvp"] = 8
    k4_flags = dict(p2z_flags)
    k4_flags["neutral_prebreakdown"] = False
    k4_flags["neutral_equilibration"] = False
    k4_sim = LAPDSim1D(k4_params, k4_flags)
    # pre-refresh: relaxation key exists and is all-zero (stable ledger)
    k4_pre = k4_sim.rhs_terms()
    assert "neutral_kinetic_relaxation" in k4_pre
    assert np.all(k4_pre["neutral_kinetic_relaxation"].nn == 0.0)
    for _ in range(3):
        k4_sim.advance_one_step(dt=1.0e-9)
    kin = k4_sim._kinetic
    assert kin.target_col is not None and kin.target_ann is not None
    assert np.all(np.isfinite(kin.target_col))
    assert np.all(np.isfinite(kin.target_ann))
    assert np.all(kin.tau_col > 0.0) and np.all(kin.tau_ann > 0.0)
    k4_terms = k4_sim.rhs_terms()
    k4_relax = k4_terms["neutral_kinetic_relaxation"]
    assert np.all(np.isfinite(k4_relax.nn))
    assert k4_relax.nn_a is not None and np.all(np.isfinite(k4_relax.nn_a))
    assert np.any(k4_relax.nn != 0.0) or np.any(k4_relax.nn_a != 0.0)
    for k4_name in (
        "ionization_birth",
        "neutral_exchange",
        "neutral_zone_exchange",
        "boundary_absorption",
        "neutral_sources",
    ):
        k4_term = k4_terms[k4_name]
        assert np.all(k4_term.nn == 0.0), k4_name
        if k4_term.nn_a is not None:
            assert np.all(k4_term.nn_a == 0.0), k4_name
    # plasma rows keep their forms (ionization still births plasma)
    assert np.any(k4_terms["ionization_birth"].n != 0.0)
    k4_state = k4_sim.state
    assert np.all(np.isfinite(k4_state.nn))
    assert np.all(np.isfinite(k4_state.nn_a))
    # flag-off ledgers carry no relaxation key
    assert "neutral_kinetic_relaxation" not in p2z_sim.rhs_terms()

    # --- Neutral-wind advection (NEUTRAL_MOMENTUM_PLAN.md M3): donor-cell
    # upwind of nn and M_n by u_n on the neutral faces, closed ends for
    # particles, end-wall momentum accommodation, and a CFL guard.
    mw_cells = 5
    mw_geom = SimpleNamespace(
        cells=mw_cells,
        length_cm=np.full(mw_cells, 30.0),
        neutral_volume_cm3=np.full(mw_cells, 2.0e5),
        neutral_face_area_cm2=np.full(mw_cells + 1, 7.0e3),
    )
    mw_un = 2.0e4
    mw_nn = np.array([1e13, 3e13, 2e13, 5e13, 4e13])
    mw_state = conservative_from_primitives(
        n=np.full(mw_cells, 1e12),
        nn=mw_nn,
        u=np.zeros(mw_cells),
        Te=np.full(mw_cells, 5.0),
        Ti=np.full(mw_cells, 1.0),
        ion_mass_g=knob_mass,
        un=np.full(mw_cells, mw_un),
    )
    # Without M_n the operator is inert and 5-field.
    mw_off = neutral_wind_advection_rhs(
        state=conservative_from_primitives(
            n=np.full(mw_cells, 1e12),
            nn=mw_nn,
            u=np.zeros(mw_cells),
            Te=np.full(mw_cells, 5.0),
            Ti=np.full(mw_cells, 1.0),
            ion_mass_g=knob_mass,
        ),
        floors=knob_floors,
        ion_mass_g=knob_mass,
        geometry=mw_geom,
    )
    assert mw_off.M_n is None and np.all(mw_off.nn == 0.0)

    mw_adv = neutral_wind_advection_rhs(
        state=mw_state,
        floors=knob_floors,
        ion_mass_g=knob_mass,
        geometry=mw_geom,
    )
    # Hand-built donor-cell stencil for a uniform positive wind: each
    # internal face carries u * nn_donor * A with the left cell as donor;
    # the end faces pass no particles.
    mw_flux = mw_un * mw_nn[:-1] * mw_geom.neutral_face_area_cm2[1:-1]
    mw_expected = np.zeros(mw_cells)
    mw_expected[:-1] -= mw_flux / mw_geom.neutral_volume_cm3[:-1]
    mw_expected[1:] += mw_flux / mw_geom.neutral_volume_cm3[1:]
    assert np.allclose(mw_adv.nn, mw_expected, rtol=1e-14)
    # Particle inventory closes (to summation rounding): the ends are
    # walls, not sinks.
    mw_nn_scale = np.max(np.abs(mw_adv.nn * mw_geom.neutral_volume_cm3))
    assert (
        abs(np.sum(mw_adv.nn * mw_geom.neutral_volume_cm3))
        < 1e-12 * mw_nn_scale
    )
    # Momentum inventory loses exactly the outward end-wall accommodation.
    mw_Mn_end = mw_state.M_n[-1]
    mw_end_sink = mw_un * mw_geom.neutral_face_area_cm2[-1] * mw_Mn_end
    assert np.isclose(
        np.sum(mw_adv.M_n * mw_geom.neutral_volume_cm3),
        -mw_end_sink,
        rtol=1e-12,
    )
    # A uniform field under a uniform wind does not change in the interior
    # (pure translation); only the end cells feel the walls.
    mw_uniform = conservative_from_primitives(
        n=np.full(mw_cells, 1e12),
        nn=np.full(mw_cells, 2e13),
        u=np.zeros(mw_cells),
        Te=np.full(mw_cells, 5.0),
        Ti=np.full(mw_cells, 1.0),
        ion_mass_g=knob_mass,
        un=np.full(mw_cells, mw_un),
    )
    mw_uadv = neutral_wind_advection_rhs(
        state=mw_uniform,
        floors=knob_floors,
        ion_mass_g=knob_mass,
        geometry=mw_geom,
    )
    assert np.all(mw_uadv.nn[1:-1] == 0.0)
    assert mw_uadv.nn[0] < 0.0 and mw_uadv.nn[-1] > 0.0
    # An inward wind at both ends leaves no accommodation sink: the pure
    # flux stencil accounts for the whole momentum change.
    mw_in_un = np.array([1.0, 1.0, 0.0, -1.0, -1.0]) * mw_un
    mw_inward = conservative_from_primitives(
        n=np.full(mw_cells, 1e12),
        nn=np.full(mw_cells, 2e13),
        u=np.zeros(mw_cells),
        Te=np.full(mw_cells, 5.0),
        Ti=np.full(mw_cells, 1.0),
        ion_mass_g=knob_mass,
        un=mw_in_un,
    )
    mw_iadv = neutral_wind_advection_rhs(
        state=mw_inward,
        floors=knob_floors,
        ion_mass_g=knob_mass,
        geometry=mw_geom,
    )
    mw_Mn_scale = np.max(np.abs(mw_iadv.M_n * mw_geom.neutral_volume_cm3))
    assert (
        abs(np.sum(mw_iadv.M_n * mw_geom.neutral_volume_cm3))
        < 1e-12 * mw_Mn_scale
    )

    # CFL guard: inf without a wind or at rest, cfl*min(dz/|u_n|) otherwise,
    # and wired into the solver's suggestion once the wind has spun up.
    assert neutral_wind_timestep(
        state=knob_state,
        floors=knob_floors,
        ion_mass_g=knob_mass,
        geometry=mw_geom,
    ) == np.inf
    assert np.isclose(
        neutral_wind_timestep(
            state=mw_state,
            floors=knob_floors,
            ion_mass_g=knob_mass,
            geometry=mw_geom,
            cfl=0.4,
        ),
        0.4 * 30.0 / mw_un,
        rtol=1e-14,
    )
    mw_diag = mn_plasma_sim.suggest_timestep()
    assert np.isfinite(mw_diag.dt_neutral_wind)
    assert mw_diag.dt_neutral_wind > 0.0

    # --- Gas-puff axial profile (gas_puff_profile): one shared implementation
    # behind both puff sites; "cell" is bit-exact legacy, "gaussian" conserves
    # the same total inflow over the main chamber.
    from cablp.solvers._sim1d.core.geometry import build_geometry

    puff_params, puff_flags = default_config()
    puff_flags["resolved_boundaries"] = True
    puff_geom = build_geometry(puff_params, puff_flags)
    from cablp.solvers._sim1d.core.geometry import puff_cell_indices as _pci

    puff_idx, _ = _pci(puff_geom)
    cell_rate = gas_puff_rate_profile(puff_geom, 3000.0, 2)
    assert cell_rate[puff_idx] == puff_rate(
        3000.0, 2.0, puff_geom.neutral_volume_cm3[puff_idx]
    )
    assert np.count_nonzero(cell_rate) == 1
    total_in = 4.477962e17 * 3000.0 * 2.0
    for z0, sigma in ((None, 30.0), (600.0, 200.0), (1900.0, 100.0)):
        gauss_rate = gas_puff_rate_profile(
            puff_geom, 3000.0, 2, profile="gaussian", z_cm=z0, sigma_cm=sigma
        )
        assert np.all(gauss_rate >= 0.0)
        # exact inflow conservation
        assert np.isclose(
            np.sum(gauss_rate * puff_geom.neutral_volume_cm3),
            total_in,
            rtol=1e-12,
        )
        # nothing lands behind the cathode or in the gap/collector
        roles = np.asarray(puff_geom.cell_role)
        forbidden = np.isin(
            roles, ("plenum", "obstruction", "cathode", "gap", "collector")
        )
        assert np.all(gauss_rate[forbidden] == 0.0)
    # narrow profile centred on the puff cell concentrates there
    narrow = gas_puff_rate_profile(
        puff_geom, 3000.0, 2, profile="gaussian", z_cm=None, sigma_cm=1.0
    )
    assert narrow[puff_idx] == narrow.max()
    # cosine_pipe: the physical Lambertian-outlet lobe. Conserves inflow,
    # peaks at its centre, and carries the heavier-than-Gaussian tails the
    # [1 + ((z-z0)/d)^2]^-2 pattern implies.
    pipe = gas_puff_rate_profile(
        puff_geom, 3000.0, 2, profile="cosine_pipe", z_cm=60.0, throw_cm=100.0
    )
    assert np.isclose(
        np.sum(pipe * puff_geom.neutral_volume_cm3), total_in, rtol=1e-12
    )
    z_centers = np.asarray(puff_geom.z_cm, dtype=float)
    interior = pipe > 0.0
    assert np.isclose(
        z_centers[np.argmax(pipe)], 60.0, atol=puff_geom.length_cm.max()
    )
    # lobe shape check at one probe cell: weight ratio matches the formula
    probe = np.flatnonzero(interior)[np.argmin(np.abs(z_centers[interior] - 260.0))]
    peak_cell = int(np.argmax(pipe))
    expected = (
        (1.0 + ((z_centers[probe] - 60.0) / 100.0) ** 2) ** -2
        / (1.0 + ((z_centers[peak_cell] - 60.0) / 100.0) ** 2) ** -2
    )
    measured = (pipe[probe] * puff_geom.neutral_volume_cm3[probe] / puff_geom.length_cm[probe]) / (
        pipe[peak_cell] * puff_geom.neutral_volume_cm3[peak_cell] / puff_geom.length_cm[peak_cell]
    )
    assert np.isclose(measured, expected, rtol=1e-12)
    try:
        gas_puff_rate_profile(puff_geom, 3000.0, 2, profile="nonsense")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown gas_puff_profile")

    # --- Double-erf puff waveform (gas_puff_mode="double_erf"): valve-like
    # erf rise 0 -> S_gp, plateau, erf drop S_gp -> S_gp_decay_target, on
    # the scheduled main-discharge clock (rise before breakdown). Both puff
    # sites share the value via _effective_gas_puff_sccm.
    derf_params, derf_flags = default_config()
    derf_flags["neutral_prebreakdown"] = False
    derf_flags["neutral_equilibration"] = False
    derf_params.update(
        {
            # scheduled phases so the afterglow assert sees the valve close
            # (the waveform's clock is the scheduled main-discharge start
            # either way)
            "phase_transition_mode": "scheduled",
            "gas_puff_mode": "double_erf",
            "S_gp": 4000.0,
            "S_gp_decay_target": 1500.0,
            "Twin_S_gp": 1000.0,
            "Twin_S_gp_decay_target": 250.0,
            "tau_prebreakdown": 2.0e-3,
            "tau_breakdown": 1.0e-3,
            "tau_discharge": 20.0e-3,
            "tau_gp_rise_center": -2.0e-3,
            "tau_gp_rise_width": 0.3e-3,
            "tau_gp_drop_center": 2.0e-3,
            "tau_gp_drop_width": 0.5e-3,
        }
    )
    derf_sim = LAPDSim1D(derf_params, derf_flags)
    # well before the rise: essentially closed
    assert derf_sim._effective_gas_puff_sccm(time=0.0)[0] < 0.01 * 4000.0
    # rise center (t_rel = -2 ms, during pre_breakdown): half the plateau
    derf_half = derf_sim._effective_gas_puff_sccm(time=1.0e-3)
    assert np.isclose(derf_half[0], 2000.0, rtol=1e-3)
    assert np.isclose(derf_half[1], 500.0, rtol=1e-3)
    # plateau (main-discharge start)
    assert np.isclose(
        derf_sim._effective_gas_puff_sccm(time=3.0e-3)[0], 4000.0, rtol=1e-3
    )
    # drop center: plateau minus half the level difference
    derf_mid = derf_sim._effective_gas_puff_sccm(time=5.0e-3)
    assert np.isclose(derf_mid[0], 4000.0 - 0.5 * 2500.0, rtol=1e-3)
    assert np.isclose(derf_mid[1], 1000.0 - 0.5 * 750.0, rtol=1e-3)
    # held second level deep in the discharge
    assert np.isclose(
        derf_sim._effective_gas_puff_sccm(time=8.0e-3)[0], 1500.0, rtol=1e-3
    )
    # afterglow: valve closed
    assert derf_sim._effective_gas_puff_sccm(time=24.5e-3) == (0.0, 0.0)
    # monotone through each transition (to the other erf's far-tail scale,
    # ~1e-5 sccm where the transitions' tails overlap)
    derf_rise_ts = np.linspace(0.0, 3.0e-3, 40)
    derf_rise_v = [derf_sim._effective_gas_puff_sccm(time=t)[0] for t in derf_rise_ts]
    assert np.all(np.diff(derf_rise_v) >= -1e-3)
    derf_drop_ts = np.linspace(3.5e-3, 8.0e-3, 40)
    derf_drop_v = [derf_sim._effective_gas_puff_sccm(time=t)[0] for t in derf_drop_ts]
    assert np.all(np.diff(derf_drop_v) <= 1e-3)
    for derf_bad in (
        {"gas_puff_mode": "triple_erf"},
        {"gas_puff_mode": "double_erf", "tau_gp_rise_width": 0.0},
        {"gas_puff_mode": "double_erf", "tau_gp_drop_width": -1.0},
    ):
        try:
            LAPDSim1D(dict(derf_params, **derf_bad), derf_flags)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {derf_bad}")

    # --- Anode disc radius (anode_radius_cm): opens the annulus around the
    # mesh to neutrals only. None = historical (1 - eta); Ra < Rm gives
    # 1 - eta*(Ra/Rm)^2; heat/Bohm keep the bare mesh values.
    disc_params, disc_flags = default_config()
    disc_params.update({"Rp": 15.0, "anode_radius_cm": 40.0})
    disc_flags["resolved_boundaries"] = True
    disc_geom = build_geometry(disc_params, disc_flags)
    disc_face = int(disc_geom.anode_face_indices[0])
    eta_cfg = disc_params["eta"]
    assert np.isclose(
        disc_geom.neutral_face_area_cm2[disc_face],
        np.pi * 50.0**2 * (1.0 - eta_cfg * (40.0 / 50.0) ** 2),
        rtol=1e-12,
    )
    assert disc_geom.heat_transmission[disc_face] == 1.0 - eta_cfg
    try:
        bad_params = dict(disc_params)
        bad_params["anode_radius_cm"] = 10.0  # smaller than the plasma channel
        build_geometry(bad_params, disc_flags)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for anode disc inside Rp")

    # --- CX-derived momentum-transfer rate (sigma_in_model = "cx_derived"):
    # nu_in = nn * (2*<sigma v>_cx + k_Langevin), consistent with the CX
    # energy channel and carrying the velocity dependence the constant lacks.
    k_L = langevin_rate_cm3_s("He", knob_mass)
    assert 5.0e-10 < k_L < 1.0e-9  # ~7.5e-10 cm^3/s for He+ in He
    nu_kwargs = dict(nn=1e13, ion_mass_g=knob_mass)
    for Ti_probe, expect_side in ((0.1, "smaller"), (5.0, "larger")):
        nu_const = ion_neutral_collision_frequency(Ti=Ti_probe, **nu_kwargs)
        nu_cxd = ion_neutral_collision_frequency(
            Ti=Ti_probe, sigma_in_model="cx_derived", gas_type="He", **nu_kwargs
        )
        assert np.isfinite(nu_cxd) and nu_cxd > 0.0
        # the constant crosses the CX-derived curve near 0.5 eV
        if expect_side == "smaller":
            assert nu_const < nu_cxd
        else:
            assert nu_const > nu_cxd
        # exact construction: 2*nu_cx + nn*k_L
        nu_cx = ion_neutral_cx_frequency(nn=1e13, Ti=Ti_probe, gas_type="He")
        assert np.isclose(nu_cxd, 2.0 * nu_cx + 1e13 * k_L, rtol=1e-12)
        # elastic remainder is nu_cx + Langevin, strictly positive
        nu_el = ion_neutral_elastic_frequency(
            nn=1e13,
            Ti=Ti_probe,
            ion_mass_g=knob_mass,
            gas_type="He",
            sigma_in_model="cx_derived",
        )
        assert np.isclose(nu_el, nu_cx + 1e13 * k_L, rtol=1e-12)
    for bad_call in (
        dict(Ti=1.0, sigma_in_model="cx_derived", **nu_kwargs),  # no gas_type
        dict(Ti=1.0, sigma_in_model="nonsense", gas_type="He", **nu_kwargs),
    ):
        try:
            ion_neutral_collision_frequency(**bad_call)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad_call}")
    # drag rhs accepts the model end-to-end and differs from the constant
    drag_cxd = ion_neutral_drag_rhs(**drag_kwargs, sigma_in_model="cx_derived")
    assert np.all(np.isfinite(drag_cxd.M))
    assert not np.allclose(drag_cxd.M, drag_const.M)

    # Thermalization scale: None inherits b_ion_neutral_drag (historical
    # coupling), an explicit value decouples it.
    therm_kwargs = dict(
        state=knob_state,
        floors=knob_floors,
        ion_mass_g=knob_mass,
        gas_type="He",
    )
    therm_inherit = ion_neutral_thermalization_rhs(
        **therm_kwargs, b_ion_neutral_drag=0.5
    )
    therm_explicit = ion_neutral_thermalization_rhs(
        **therm_kwargs,
        b_ion_neutral_drag=0.5,
        b_ion_neutral_thermalization=0.5,
    )
    assert np.all(therm_inherit.Ei == therm_explicit.Ei)
    # Decoupled: survives a zeroed drag scalar and scales independently.
    therm_decoupled = ion_neutral_thermalization_rhs(
        **therm_kwargs,
        b_ion_neutral_drag=0.0,
        b_ion_neutral_thermalization=1.0,
    )
    assert np.allclose(therm_decoupled.Ei, 2.0 * therm_inherit.Ei, rtol=1e-12)
    assert np.any(therm_decoupled.Ei != 0.0)

    # Te-shaped cooling correction: identity at the reference temperature,
    # and the (Te/Te_ref)^exp factor elsewhere.
    shape_state = conservative_from_primitives(
        n=np.full(3, 1e12),
        nn=np.full(3, 1e13),
        u=np.zeros(3),
        Te=np.array([2.5, 5.0, 10.0]),
        Ti=np.full(3, 1.0),
        ion_mass_g=knob_mass,
    )
    cooling_kwargs = dict(
        state=shape_state,
        floors=knob_floors,
        ion_mass_g=knob_mass,
        gas_type="He",
        I_ion=24.587,
        ionization_energy_cost=False,
    )
    cool_flat = electron_cooling_rhs(**cooling_kwargs)
    cool_shaped = electron_cooling_rhs(
        **cooling_kwargs,
        b_Qei_Te_exp=1.0,
        b_Qen_Te_exp=1.0,
        b_Q_Te_ref_eV=5.0,
    )
    shape_factor = np.array([0.5, 1.0, 2.0])
    assert np.allclose(cool_shaped.Ee, cool_flat.Ee * shape_factor, rtol=1e-12)
    assert cool_shaped.Ee[1] == cool_flat.Ee[1]

    # --- ADAS atomic rate model (atomic_rate_model = "adas"): adf11 tables
    # parse, grid nodes reproduce exactly, edges clamp, and the physics the
    # switch exists for shows up (effective SCD ionization above the direct
    # ground-state rate at low Te, radiation-only cooling below the IAEA fit).
    from cablp.funcs import _adas
    from cablp.funcs._cross import He_ion_rate_lkup
    from cablp.funcs._fits import IAEA_exp1
    from cablp.vars._coeff import aHeI

    scd_ne, scd_te, scd_stages = _adas.read_adf11(_adas.ADAS_DIR / "scd96_he.dat")
    assert scd_ne.shape == (24,) and scd_te.shape == (30,)
    assert set(scd_stages) == {1, 2}
    # Interpolation at a grid node returns the tabulated value exactly.
    node = _adas.he_ionization_rate(10.0 ** scd_ne[10], 10.0 ** scd_te[15])
    assert np.isclose(node, 10.0 ** scd_stages[1][15, 10], rtol=1e-12)
    # Edge clamping: below/above the Te grid returns the edge value.
    lo = _adas.he_ionization_rate(1e12, 10.0 ** scd_te[0])
    assert np.isclose(_adas.he_ionization_rate(1e12, 0.05), lo, rtol=1e-12)
    # Stepwise/metastable enhancement: effective SCD exceeds the direct
    # ground-state rate at low Te and converges toward it at high Te.
    assert _adas.he_ionization_rate(1e13, 5.0) > 2.0 * He_ion_rate_lkup(5.0)
    assert np.isclose(
        _adas.he_ionization_rate(1e12, 100.0), He_ion_rate_lkup(100.0), rtol=0.1
    )
    # Radiation-only cooling sits well below the IAEA fit (which carries the
    # ionization-potential loss).
    assert _adas.he_neutral_line_power(1e13, 8.0) < 0.5 * IAEA_exp1(8.0, aHeI)

    # Fused lookup (he_rates): one coordinate solve, N table blends -- must be
    # bit-identical to the single-table helpers, since both share the same
    # blend arithmetic on the same (verified-identical) grid.
    fuse_ne = 10.0 ** np.random.default_rng(1).uniform(8.0, 15.0, 64)
    fuse_Te = 10.0 ** np.random.default_rng(2).uniform(-0.6, 3.5, 64)
    fused = _adas.he_rates(fuse_ne, fuse_Te, ("scd", "acd", "plt1", "plt2", "prb1"))
    for name, single in (
        ("scd", _adas.he_ionization_rate),
        ("acd", _adas.he_recombination_rate),
        ("plt1", _adas.he_neutral_line_power),
        ("plt2", _adas.he_ion_line_power),
        ("prb1", _adas.he_recombination_power),
    ):
        assert np.all(fused[name] == single(fuse_ne, fuse_Te)), name

    # The float port of the 2^1P excitation cross section matches mpmath.
    from cablp.funcs._cathode_solver import _he_2p_excitation_cross_cm2
    from cablp.funcs._cross import He_EIE_cross_DA
    from cablp.vars._coeff import b_11s_21p as _b21p
    for eps_probe in (1.5, 100.0 / 21.218, 8.0):
        assert np.isclose(
            _he_2p_excitation_cross_cm2(eps_probe),
            float(He_EIE_cross_DA(eps_probe, _b21p)),
            rtol=1e-12,
        )

    # --- A1: the He singlet manifold registry (BEAM_DEPOSITION_PLAN WP-A). ---
    from cablp.funcs._cross import (
        He_EIE_cross_manifold,
        He_singlet_tail_cross,
    )
    from cablp.vars._coeff import He_singlet_manifold

    # The 2^1P row is the provenance anchor: same list object as b_11s_21p,
    # and the general evaluator reproduces the beam's float port (the ~7e-6
    # slack is the legacy E_21p = 21.217848 vs the registry's NIST 21.2180).
    assert He_singlet_manifold["21P"]["A"] is _b21p
    for eps_probe in (1.5, 100.0 / 21.218, 8.0):
        assert np.isclose(
            He_EIE_cross_manifold(
                eps_probe * He_singlet_manifold["21P"]["E_eV"],
                He_singlet_manifold["21P"],
            ),
            _he_2p_excitation_cross_cm2(eps_probe),
            rtol=1e-4,
        )

    # Every fitted level: zero at/below threshold, finite and non-negative
    # from just above threshold through 1 keV.
    manifold_probe_E = np.concatenate(
        [np.linspace(24.0, 200.0, 45), np.array([500.0, 1000.0])]
    )
    for level_name, entry in He_singlet_manifold.items():
        assert He_EIE_cross_manifold(entry["E_eV"], entry) == 0.0, level_name
        assert He_EIE_cross_manifold(0.5 * entry["E_eV"], entry) == 0.0, level_name
        for E_probe in manifold_probe_E:
            sigma_probe = He_EIE_cross_manifold(float(E_probe), entry)
            assert np.isfinite(sigma_probe) and sigma_probe >= 0.0, level_name
        assert He_EIE_cross_manifold(100.0, entry) > 0.0, level_name

    # The measured manifold multipliers at 100 eV (measure_beam_manifold.py,
    # 2026-07-20): R_events = 1.670, R_power = 1.730 against the historical
    # 2^1P-only booking. Loose bounds guard the transcribed coefficients
    # against digit regressions without over-pinning the fit evaluation.
    sigma_by_level = {
        name: He_EIE_cross_manifold(100.0, entry)
        for name, entry in He_singlet_manifold.items()
    }
    tail_sigma_100, tail_sigma_E_100 = He_singlet_tail_cross(100.0)
    manifold_sigma_100 = sum(sigma_by_level.values()) + tail_sigma_100
    manifold_sigma_E_100 = (
        sum(
            sigma_by_level[name] * He_singlet_manifold[name]["E_eV"]
            for name in sigma_by_level
        )
        + tail_sigma_E_100
    )
    r_events_100 = manifold_sigma_100 / sigma_by_level["21P"]
    r_power_100 = manifold_sigma_E_100 / (sigma_by_level["21P"] * 21.218)
    assert 1.55 < r_events_100 < 1.80, r_events_100
    assert 1.60 < r_power_100 < 1.85, r_power_100
    # The Eq. (5) Rydberg tail sums to ~1.56x the 4^1P row (sum of (4/n)^3
    # plus the small nS/nD/nF series and threshold shifts).
    assert 1.3 < tail_sigma_100 / sigma_by_level["41P"] < 2.0

    # --- B1: the standalone CSDA beam-deposition module
    # (BEAM_DEPOSITION_PLAN B1; full acceptance in
    # scripts/verify_beam_deposition.py — this is the fast subset).
    from cablp.funcs._beam_deposition import (
        beam_speed_cm_s,
        coulomb_stopping_eV_per_cm,
        deposit_beam,
        quasilinear_relaxation_length_cm,
    )

    b1_cells = 30
    b1_col = dict(
        nn=np.full(b1_cells, 3.0e14),
        ne=np.full(b1_cells, 1.0e10),
        Te=np.full(b1_cells, 1.0),
        launch=0,
        direction=1,
        dz_cm=np.full(b1_cells, 100.0),
    )
    b1_res = deposit_beam(150.0, 1.0e22, **b1_col)
    b1_budget = 1.0e22 * 150.0 * 1.602176634e-12
    b1_total = (
        b1_res.plasma_heating_erg_s.sum()
        + b1_res.radiated_erg_s.sum()
        + b1_res.ionization_cost_erg_s.sum()
        + b1_res.transmitted_flux
        * b1_res.transmitted_energy_eV
        * 1.602176634e-12
    )
    assert abs(b1_total - b1_budget) / b1_budget < 1e-10
    # Breakdown conditions: several inelastic events per primary (the
    # single-event Beer-Lambert booking caps at 1 — THESIS_NOTES item 10).
    b1_events = (
        b1_res.ionization_events.sum() + b1_res.excitation_events.sum()
    ) / 1.0e22
    assert 2.0 < b1_events < 6.0, b1_events
    # Ray discipline: nothing behind the launch cell, direction respected.
    b1_res_rev = deposit_beam(
        150.0, 1.0e22, **{**b1_col, "launch": b1_cells - 1, "direction": -1}
    )
    assert np.allclose(
        b1_res_rev.ionization_events, b1_res.ionization_events[::-1]
    )
    # Closure ordering at production conditions (item 12): quasilinear <<
    # legacy tau_ei "Coulomb" << classical fast-electron stopping.
    b1_nb = 1.0e22 / (700.0 * beam_speed_cm_s(150.0))
    b1_lql = quasilinear_relaxation_length_cm(150.0, 5.0e12, b1_nb)
    b1_llegacy = 150.0 / coulomb_stopping_eV_per_cm(
        150.0, 5.0e12, 8.0, "legacy_tau_ei"
    )
    b1_lfast = 150.0 / coulomb_stopping_eV_per_cm(
        150.0, 5.0e12, 8.0, "fast_electron"
    )
    assert b1_lql < b1_llegacy < b1_lfast
    # Sub-threshold source passes through untouched; bad closures raise.
    b1_sub = deposit_beam(15.0, 1.0e22, **b1_col)
    assert b1_sub.transmitted_flux == 1.0e22
    assert b1_sub.plasma_heating_erg_s.sum() == 0.0
    for b1_bad in (
        lambda: deposit_beam(150.0, 1e22, **b1_col, coulomb_model="bogus"),
        lambda: deposit_beam(150.0, 1e22, **b1_col, anomalous_model="quasilinear"),
    ):
        try:
            b1_bad()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError from deposit_beam")

    adas_reaction_kwargs = dict(
        state=knob_state,
        floors=knob_floors,
        ion_mass_g=knob_mass,
        gas_type="He",
        I_ion=24.587,
    )
    S_ion_j, S_rad_j, S_3b_j = reaction_rates(**adas_reaction_kwargs)
    S_ion_a, S_rad_a, S_3b_a = reaction_rates(
        **adas_reaction_kwargs, atomic_rate_model="adas"
    )
    for values in (S_ion_a, S_rad_a):
        assert np.all(np.isfinite(values)) and np.all(values >= 0.0)
    assert np.all(S_ion_a > S_ion_j)  # SCD > direct at these (Te <= 6 eV) cells
    # ACD carries the whole sink; the three-body slot is empty and its knob inert.
    assert np.all(S_3b_a == 0.0)
    S_ion_b3, S_rad_b3, S_3b_b3 = reaction_rates(
        **adas_reaction_kwargs, atomic_rate_model="adas", b_rec_3b=7.0
    )
    assert np.all(S_ion_b3 == S_ion_a) and np.all(S_rad_b3 == S_rad_a)
    assert np.all(S_3b_b3 == 0.0)
    try:
        reaction_rates(**adas_reaction_kwargs, atomic_rate_model="nonsense")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown atomic_rate_model")

    cool_adas = electron_cooling_rhs(**cooling_kwargs, atomic_rate_model="adas")
    assert np.all(np.isfinite(cool_adas.Ee))
    assert np.all(cool_adas.Ee <= 0.0)
    # Radiation-only: strictly weaker electron cooling than the IAEA fits on
    # the same state (the ionization-cost double count is what's removed).
    assert np.all(np.abs(cool_adas.Ee) < np.abs(cool_flat.Ee))
    # The cooling path's fused ionization cost must be bit-identical to
    # I_ion * S_ion from reaction_rates -- the cost charges exactly the
    # particles the particle equation creates.
    cost_kwargs = dict(cooling_kwargs)
    cost_kwargs["ionization_energy_cost"] = True
    cost_terms = electron_cooling_rhs_terms(**cost_kwargs, atomic_rate_model="adas")
    S_ion_ref, _, _ = reaction_rates(
        state=shape_state,
        floors=knob_floors,
        ion_mass_g=knob_mass,
        gas_type="He",
        I_ion=24.587,
        atomic_rate_model="adas",
    )
    assert np.allclose(
        cost_terms["ionization_energy_cost"].Ee,
        -24.587 * ev_to_erg * S_ion_ref,
        rtol=1e-13,
        atol=0.0,
    )

    # --- Directed recycle jets (CATHODE_IDRIVEN_PLAN.md §8): cathode-face
    # backscatter + effusion and anode-mesh backscatter ride the SAME terms
    # that rebirth the recycle particles, as M_n sources; the mesh
    # accommodates the wind momentum its wires intercept. Validation fails
    # fast; magnitudes must reproduce the step-1 scoping arithmetic exactly
    # from each term's own rebirthed flux (particle/momentum consistency).
    # Directed recycle jets ride on the evolved M_n (legacy-drag path); run on
    # the simple stance with the moment operator off (jet_params derive from the
    # simple m3_params). The legacy-drag DeprecationWarning is expected.
    jet_flags = dict(m3_cathode_flags)
    jet_flags["neutral_momentum"] = True
    jet_flags["ion_neutral_moment_closure"] = False
    for jet_bad_params, jet_bad_flags in (
        # M_n physics without the neutral_momentum flag
        (dict(m3_params, cathode_neutral_jet=True), resolved_cathode_flags),
        (dict(m3_params, anode_neutral_jet=True), resolved_cathode_flags),
        (dict(m3_params, neutral_mesh_accommodation=True),
         resolved_cathode_flags),
        # reflection coefficients outside [0, 1]
        (dict(m3_params, cathode_neutral_jet=True, cathode_jet_R_E=1.5),
         jet_flags),
        (dict(m3_params, anode_neutral_jet=True, anode_jet_R_N=-0.1),
         jet_flags),
        # the debit reads the jet's R_E
        (dict(m3_params, cathode_jet_surface_debit=True), jet_flags),
    ):
        try:
            LAPDSim1D(jet_bad_params, jet_bad_flags)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {jet_bad_params}")
    jet_params = dict(
        m3_params,
        cathode_neutral_jet=True,
        anode_neutral_jet=True,
        neutral_mesh_accommodation=True,
    )
    jet_sim = LAPDSim1D(jet_params, jet_flags)
    jet_sim._circuit_I_loop = 800.0
    jet_solve = jet_sim.solve_cathode_boundary(update_cache=True)
    jet_res = jet_solve.beam_result.result
    jet_geom = jet_sim.geometry
    jet_roles = np.asarray(jet_geom.cell_role)
    jet_m = jet_sim.ion_mass_g
    jet_derived = derive_state(jet_sim.state, jet_sim.floors, jet_m)
    jet_kb = 1.380649e-16

    # Cathode channel: momentum only at the cathode cell, directed into the
    # column (+z at the source end), and the volume-integrated M_n source
    # equals m * v_mix * (the term's own rebirthed flux) exactly.
    jet_ba = jet_sim.boundary_absorption_rhs(cathode_solve=jet_solve)
    jet_cath = int(np.flatnonzero(jet_roles == "cathode")[0])
    assert jet_ba.M_n is not None
    assert np.array_equal(np.flatnonzero(jet_ba.M_n), [jet_cath])
    assert jet_ba.M_n[jet_cath] > 0.0
    jet_RN = float(jet_params.get("cathode_jet_R_N", 0.5))
    jet_RE = float(jet_params.get("cathode_jet_R_E", 0.2))
    jet_Ts = float(jet_params["T_s"])
    jet_veff = np.sqrt(np.pi * jet_kb * jet_Ts / (2.0 * jet_m))
    jet_vback = np.sqrt(
        2.0 * jet_RE
        * (max(jet_res.phi_c, 0.0) + jet_derived.Ti[jet_cath])
        * ev_to_erg / jet_m
    )
    jet_vmix = jet_RN * jet_vback + (1.0 - jet_RN) * jet_veff
    jet_flux = jet_ba.nn[jet_cath] * jet_geom.neutral_volume_cm3[jet_cath]
    assert np.isclose(
        jet_ba.M_n[jet_cath] * jet_geom.neutral_volume_cm3[jet_cath],
        jet_m * jet_vmix * jet_flux,
        rtol=1e-12,
        atol=0.0,
    )

    # Anode channel: DEFERRED (R5 stance flip, 2026-07-25). Under the repaired
    # stance the anode jet M_n comes out zero at this quiescent test state. The
    # reason is NOT simply the anode sheath sign -- a negative phi_a is a
    # POSITIVE ion-sheath, which does not by itself imply zero ion current;
    # deriving whether the anode collects ions here needs the ion-sheath
    # physics. The M_n directed-jet module may be rewritten from the ES1
    # baseline findings (Tom), so its anode-channel physics assertions are
    # deferred rather than re-derived now. Tracked in DEPRECATION_PLAN.md D3 /
    # R5_STANCE_FLIP_HANDOFF.md. The flag plumbing + construction validation
    # above, and the mesh-accommodation stencil below, still run.

    # Floating (afterglow) solve: the jet rides the floating sheath drop --
    # tiny but finite, never NaN.
    jet_float = jet_sim.solve_cathode_boundary(
        floating=True, update_cache=False
    )
    jet_ba_float = jet_sim.boundary_absorption_rhs(cathode_solve=jet_float)
    assert np.all(np.isfinite(jet_ba_float.M_n))
    assert np.all(jet_ba_float.M_n >= 0.0)

    # Presence gating: flags off -> the terms stay 5-field even with M_n on
    # the state (the golden path can never construct a jet).
    jet_off_sim = LAPDSim1D(dict(m3_params), jet_flags)
    jet_off_sim._circuit_I_loop = 800.0
    jet_off_solve = jet_off_sim.solve_cathode_boundary(update_cache=True)
    assert jet_off_sim.boundary_absorption_rhs(
        cathode_solve=jet_off_solve
    ).M_n is None
    assert jet_off_sim.anode_collection_rhs(
        cathode_solve=jet_off_solve
    ).M_n is None

    # Mesh momentum accommodation: hand-built stencil -- the wind flowing
    # INTO the mesh loses -|u| * A_blocked / V * M_n on its own side only;
    # sign-safe (relaxes M_n toward zero) for either wind direction.
    mesh_cells = 5
    mesh_geom = SimpleNamespace(
        cells=mesh_cells,
        length_cm=np.full(mesh_cells, 30.0),
        neutral_volume_cm3=np.full(mesh_cells, 2.0e5),
        neutral_face_area_cm2=np.full(mesh_cells + 1, 7.0e3),
    )
    mesh_floors = {"n": 1e8, "nn": 1e8, "Te": 0.1, "Ti": 0.02}
    mesh_kw = dict(
        n=np.full(mesh_cells, 1e12),
        nn=np.full(mesh_cells, 2e13),
        u=np.zeros(mesh_cells),
        Te=np.full(mesh_cells, 5.0),
        Ti=np.full(mesh_cells, 1.0),
        ion_mass_g=knob_mass,
    )
    for mesh_un, mesh_hit, mesh_dry in ((2.0e4, 1, 2), (-2.0e4, 2, 1)):
        mesh_state = conservative_from_primitives(
            un=np.full(mesh_cells, mesh_un), **mesh_kw
        )
        mesh_base = neutral_wind_advection_rhs(
            state=mesh_state,
            floors=mesh_floors,
            ion_mass_g=knob_mass,
            geometry=mesh_geom,
        )
        mesh_with = neutral_wind_advection_rhs(
            state=mesh_state,
            floors=mesh_floors,
            ion_mass_g=knob_mass,
            geometry=mesh_geom,
            mesh_faces=[2],
            mesh_blocked_area_cm2=[3.0e3],
        )
        mesh_diff = mesh_with.M_n - mesh_base.M_n
        assert np.isclose(
            mesh_diff[mesh_hit],
            -abs(mesh_un) * 3.0e3 * mesh_state.M_n[mesh_hit] / 2.0e5,
            rtol=1e-13,
        )
        # The sink always relaxes M_n toward zero.
        assert mesh_diff[mesh_hit] * mesh_state.M_n[mesh_hit] < 0.0
        assert mesh_diff[mesh_dry] == 0.0
        # Particle fluxes are untouched -- accommodation is momentum-only.
        assert np.array_equal(mesh_with.nn, mesh_base.nn)
    # Solver wiring: blocked area reconstructs the full face through the
    # open fraction T = 1 - eta (Ra = None), A_blocked = A_open * (1-T)/T.
    jet_eta = float(jet_params["eta"])
    jet_T = 1.0 - jet_eta
    assert np.allclose(
        jet_sim._mesh_blocked_area_cm2,
        np.asarray(jet_geom.neutral_face_area_cm2, dtype=float)[
            np.asarray(jet_geom.anode_face_indices, dtype=int)
        ] * jet_eta / jet_T,
        rtol=1e-13,
    )

    # Surface-debit sensitivity arm: power_balance receives (1 - R_E) * P_i
    # when on; exactly 1.0 * P_i (the M5a' calibration convention) when off.
    assert jet_sim._cathode_surface_ion_retention == 1.0
    jet_debit_sim = LAPDSim1D(
        dict(jet_params, cathode_jet_surface_debit=True), jet_flags
    )
    assert np.isclose(
        jet_debit_sim._cathode_surface_ion_retention,
        1.0 - float(jet_params.get("cathode_jet_R_E", 0.2)),
        rtol=1e-13,
    )

    # --- GCR-consistent recombination energy pair
    # (recombination_energy_return): +I_ion*S_rec - P_PRB on the electron
    # fluid, adas-only, mutually exclusive with icool_recomb (double-charge).
    from cablp.funcs._adas import he_rates as _rer_he_rates
    from cablp.solvers._sim1d.physics.reactions import (
        recombination_energy_return_rhs,
    )

    for rer_bad_params, rer_bad_flags in (
        (dict(m3_params, recombination_energy_return=True,
              atomic_rate_model="janev"), resolved_cathode_flags),
        (dict(m3_params, recombination_energy_return=True,
              atomic_rate_model="adas"),
         dict(resolved_cathode_flags, icool_recomb=True)),
    ):
        try:
            LAPDSim1D(rer_bad_params, rer_bad_flags)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {rer_bad_params}")
    rer_sim = LAPDSim1D(
        dict(m3_params, recombination_energy_return=True,
             atomic_rate_model="adas"),
        resolved_cathode_flags,
    )
    rer_state = rer_sim.state
    rer_term = rer_sim.recombination_energy_return_rhs()
    rer_derived = derive_state(rer_state, rer_sim.floors, rer_sim.ion_mass_g)
    rer_rates = _rer_he_rates(
        np.maximum(rer_state.n, rer_sim.floors["n"]),
        rer_derived.Te,
        ("acd", "prb1"),
    )
    rer_I_ion = float(rer_sim._I_ion)
    rer_hand = ev_to_erg * rer_state.n * rer_state.n * (
        rer_I_ion * rer_rates["acd"] - rer_rates["prb1"]
    )
    assert np.allclose(rer_term.Ee, rer_hand, rtol=1e-12, atol=0.0)
    assert np.all(rer_term.n == 0.0) and np.all(rer_term.Ei == 0.0)
    # Present in the ledger; identically zero when the key is off (the
    # golden path sums an exact zero term).
    assert "recombination_energy_return" in rer_sim.rhs_terms()
    rer_off = LAPDSim1D(
        dict(m3_params, atomic_rate_model="adas"), resolved_cathode_flags
    )
    assert np.all(rer_off.recombination_energy_return_rhs().Ee == 0.0)
    # Direction: heating (I_ion > E_rad/event) at the clamped afterglow
    # floor (Te = 0.2 eV, the adf11 grid edge, where E_rad/event ~ 15 eV),
    # net sink in the hot ionizing plateau (PRB's bremsstrahlung/cascade
    # keeps radiating while ACD collapses, so E_rad/event >> I_ion there).
    rer_cold = _rer_he_rates(
        np.full(1, 1.0e13), np.full(1, 0.2), ("acd", "prb1")
    )
    rer_hot = _rer_he_rates(
        np.full(1, 5.0e12), np.full(1, 8.0), ("acd", "prb1")
    )
    assert rer_I_ion * rer_cold["acd"][0] > rer_cold["prb1"][0]
    assert rer_I_ion * rer_hot["acd"][0] < rer_hot["prb1"][0]

    # --- Square gas-puff waveform (the measured piezo/supply behaviour):
    # erf rise anchored on circuit-on, flat at S_gp through the drive,
    # erf close after drive end with a tail into the afterglow.
    for sq_bad in (
        {"gas_puff_mode": "square", "gas_puff_rise_width_s": 0.0},
        {"gas_puff_mode": "square", "gas_puff_close_lag_s": -1e-3},
    ):
        try:
            LAPDSim1D(dict(m3_params, **sq_bad), resolved_cathode_flags)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {sq_bad}")
    sq_sim = LAPDSim1D(
        dict(m3_params, gas_puff_mode="square"), resolved_cathode_flags
    )
    sq_t0 = sq_sim._plasma_phase_time_origin()
    sq_Sgp = float(sq_sim._input_dict["S_gp"])
    # Before circuit-on the envelope is (nearly) closed; well after the rise
    # it is flat at S_gp; mid-drive stays flat (no decay-to-level).
    assert sq_sim._effective_gas_puff_sccm(time=0.0)[0] < 0.1 * sq_Sgp
    assert np.isclose(
        sq_sim._effective_gas_puff_sccm(time=sq_t0 + 5e-3)[0], sq_Sgp,
        rtol=1e-6,
    )
    # Emulate a triggered breakdown to exercise the close anchor: the flow
    # still runs at S_gp late in the drive, decays through the close lag,
    # and is shut well inside the afterglow.
    sq_sim._t_prebreakdown_trigger = sq_t0 + 1.0e-3
    sq_sim._t_breakdown_trigger = sq_t0 + 1.2e-3
    sq_tau_dis = float(sq_sim._input_dict["tau_discharge"])
    sq_end = sq_sim._t_breakdown_trigger + sq_tau_dis
    assert np.isclose(
        sq_sim._effective_gas_puff_sccm(time=sq_end - 2e-3)[0], sq_Sgp,
        rtol=1e-6,
    )
    sq_mid_close = sq_sim._effective_gas_puff_sccm(
        time=sq_end + float(sq_sim._input_dict.get("gas_puff_close_lag_s", 5e-4))
    )[0]
    assert 0.3 * sq_Sgp < sq_mid_close < 0.7 * sq_Sgp
    assert sq_sim._effective_gas_puff_sccm(time=sq_end + 4e-3)[0] < 1e-3 * sq_Sgp
    # The afterglow phase switch stays open for the square tail only; a
    # non-square (deprecated) mode leaves the afterglow puff off.
    assert sq_sim._phase_switches("afterglow")["gas_puff_enabled"]
    assert not LAPDSim1D(
        dict(m3_params, gas_puff_mode="decay_after_breakdown"),
        resolved_cathode_flags,
    )._phase_switches("afterglow")["gas_puff_enabled"]

    # --- Electrode sample smoothing (cathode_sample_smoothing): EMA of the
    # sampled cathode/anode-flank (n, Te) at the presheath transit time,
    # accepted-steps only; the solve reads the smoothed state.
    for ss_bad in ({"cathode_sample_smoothing": "bogus"},
                   {"cathode_sample_smoothing": -1.0}):
        try:
            LAPDSim1D(dict(m3_params, **ss_bad), resolved_cathode_flags)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {ss_bad}")
    ss_sim = LAPDSim1D(
        dict(m3_params, cathode_sample_smoothing="presheath"),
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
        max(ss_Te_old, ss_sim.floors["Te"]), ss_sim._mu
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
    # Off (default) is the identity: same object back, no copies.
    ss_off = LAPDSim1D(dict(m3_params), resolved_cathode_flags)
    ss_off_state = ss_off.state
    assert ss_off._smoothed_sample_state(ss_off_state) is ss_off_state

    # R1a: one authoritative active-plasma topology. Every closed face has at
    # most one live-side cell, pressure work is invariant to the dead-side
    # velocity, and plasma rows in plenum/obstruction cells are bit-invariant
    # through multiple accepted steps when the default-off repair is enabled.
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
        }
    )
    r1a_flags.update(
        {
            "active_plasma_topology": True,
            "cathode_coupling": False,
            "neutral_prebreakdown": False,
            "neutral_equilibration": False,
            "launch_plasma_after_equilibration": False,
        }
    )
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
        active_plasma_topology=True,
    )
    div_dead_fast = velocity_divergence(
        r1a_dead_fast,
        r1a_sim.floors,
        r1a_sim.ion_mass_g,
        r1a_geom,
        active_plasma_topology=True,
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
            "neutral_kinetic_relaxation",
        }:
            continue
        for field_name in ("n", "nn", "M", "Ee", "Ei"):
            assert np.array_equal(
                getattr(term, field_name)[r1a_dead],
                np.zeros(np.count_nonzero(r1a_dead)),
            )

    # R1b: the saved evidence follows the actual packed state for the stable
    # five-, six-, seven-, and eight-row layouts. Two-zone density inventory
    # uses V_col=V_p and V_ann=V_m-V_p; the two-momentum radial transfer is
    # exactly internal under those same volumes.
    r1b_layouts = (
        ("five", {}, {}),
        ("six", {}, {"neutral_momentum": True}),
        (
            "seven",
            {},
            {"neutral_momentum": True, "neutral_two_zone": True},
        ),
        (
            "eight",
            {"neutral_momentum_radial": "kinetic_two_moment"},
            {"neutral_momentum": True, "neutral_two_zone": True},
        ),
    )
    with tempfile.TemporaryDirectory() as r1b_tmp:
        for expected_rows, (label, param_extra, flag_extra) in zip(
            (5, 6, 7, 8), r1b_layouts
        ):
            layout_params = dict(r1a_params, **param_extra)
            layout_flags = dict(r1a_flags, **flag_extra)
            layout_sim = LAPDSim1D(layout_params, layout_flags)
            layout_result = layout_sim.run(
                t_end=2.0e-10,
                dt=1.0e-10,
                operator_split=False,
                max_steps=4,
            )
            expected_fields = state_field_names(layout_sim.state)
            assert layout_result.y.shape[1] == expected_rows * layout_sim.geometry.cells
            assert tuple(layout_result.total_rhs) == expected_fields
            for term_fields in layout_result.rhs_terms.values():
                assert tuple(term_fields) == expected_fields
                for values in term_fields.values():
                    assert values.shape == (
                        len(layout_result.time),
                        layout_sim.geometry.cells,
                    )

            layout_path = Path(r1b_tmp) / f"{label}.h5"
            layout_sim.save_result(layout_path, layout_result)
            loaded_layout = load_result_hdf5(layout_path)
            assert loaded_layout.y.shape == layout_result.y.shape
            assert set(loaded_layout.total_rhs) == set(expected_fields)
            for term_fields in loaded_layout.rhs_terms.values():
                assert set(term_fields) == set(expected_fields)

            layout_health = summarize_result(loaded_layout)
            Vp = np.asarray(loaded_layout.plasma_volume_cm3, dtype=float)
            Vm = np.asarray(loaded_layout.neutral_volume_cm3, dtype=float)
            if hasattr(loaded_layout, "nn_a"):
                expected_column = np.sum(
                    loaded_layout.nn * Vp[None, :], axis=1
                )
                expected_annulus = np.sum(
                    loaded_layout.nn_a * (Vm - Vp)[None, :],
                    axis=1,
                )
                expected_neutral = expected_column + expected_annulus
                assert np.array_equal(
                    layout_health.neutral_column_inventory,
                    expected_column,
                )
                assert np.array_equal(
                    layout_health.neutral_annulus_inventory,
                    expected_annulus,
                )
            else:
                expected_neutral = np.sum(
                    loaded_layout.nn * Vm[None, :], axis=1
                )
            assert np.array_equal(
                layout_health.neutral_inventory, expected_neutral
            )
            assert layout_health.finite
            assert set(loaded_layout.floor_ledger) == {
                "n_particles_added",
                "nn_particles_added",
                "nn_a_particles_added",
                "Ee_energy_added_erg",
                "Ei_energy_added_erg",
            }
            assert all(
                float(value) == 0.0
                for value in loaded_layout.floor_ledger.values()
            )

            if layout_sim.state.nn_a is not None:
                exchange = layout_sim.neutral_zone_exchange_rhs()
                exchange_residual = (
                    exchange.nn * Vp + exchange.nn_a * (Vm - Vp)
                )
                exchange_scale = max(
                    float(np.max(np.abs(exchange.nn * Vp))), 1.0
                )
                assert (
                    float(np.max(np.abs(exchange_residual)))
                    <= 1.0e-14 * exchange_scale
                )

    # R1c: raw candidates are rejected before clipping, including every
    # optional density and both energy rows. Trial failures leave accepted
    # state/time/circuit/cache and the accepted-only floor ledger unchanged.
    r1c_params = dict(
        r1a_params,
        neutral_momentum_radial="kinetic_two_moment",
    )
    r1c_flags = dict(
        r1a_flags,
        neutral_momentum=True,
        neutral_two_zone=True,
        raw_stage_validation=True,
    )
    r1c_dt = 1.0e-10
    for bad_field in ("n", "nn", "nn_a", "Ee", "Ei"):
        reject_sim = LAPDSim1D(r1c_params, r1c_flags)
        before_y = reject_sim._y.copy()
        before_time = reject_sim.time
        before_loop = reject_sim._circuit_I_loop
        before_cache = reject_sim._step_cache_snapshot()
        before_ledger = dict(reject_sim._floor_ledger)
        field_names = state_field_names(reject_sim.state)
        bad_row = field_names.index(bad_field)
        cells = reject_sim.geometry.cells

        def bad_rhs(y, time=None, _row=bad_row, _cells=cells):
            rhs = np.zeros_like(y)
            start = _row * _cells
            rhs[start : start + _cells] = (
                -2.0 * np.asarray(y)[start : start + _cells] / r1c_dt
            )
            return rhs

        reject_sim.rhs = bad_rhs
        rejected = reject_sim._attempt_step(
            dt=r1c_dt, operator_split=False
        )
        reason, detail = reject_sim._step_rejection_info(
            rejected, y0=before_y
        )
        assert reason == (
            "negative_energy" if bad_field in {"Ee", "Ei"}
            else "negative_density"
        )
        assert bad_field in detail["fields"]
        assert np.array_equal(reject_sim._y, before_y)
        assert reject_sim.time == before_time
        assert reject_sim._circuit_I_loop == before_loop
        assert reject_sim._cathode_x0 == before_cache.cathode_x0
        assert reject_sim._cathode_x0_twin == before_cache.cathode_x0_twin
        assert np.array_equal(
            reject_sim._cathode_beam_cross,
            before_cache.cathode_beam_cross,
        )
        assert reject_sim._floor_ledger == before_ledger
        assert all(value == 0.0 for value in rejected.floor_ledger.values())

    # The implicit heat candidate uses the same pre-floor validation hook.
    implicit_reject_sim = LAPDSim1D(r1c_params, r1c_flags)
    implicit_before = implicit_reject_sim._y.copy()
    implicit_original = implicit_reject_sim.implicit_heat_conduction_step

    def bad_implicit(*args, **kwargs):
        state = implicit_original(*args, **kwargs)
        return ConservativeState1D(
            n=state.n,
            nn=state.nn,
            M=state.M,
            Ee=-np.abs(state.Ee),
            Ei=state.Ei,
            M_n=state.M_n,
            nn_a=state.nn_a,
            M_n_a=state.M_n_a,
        )

    implicit_reject_sim.implicit_heat_conduction_step = bad_implicit
    try:
        implicit_reject_sim.operator_split_step(
            dt=r1c_dt, splitting="strang"
        )
    except ValueError as error:
        assert "negative_energy" in str(error)
    else:
        raise AssertionError("expected raw implicit-energy rejection")
    assert np.array_equal(implicit_reject_sim._y, implicit_before)

    # Exact floor debit: particles use each field's physical inventory
    # volume and energy uses the plasma volume. A direct probe does not
    # mutate the accepted-only cumulative ledger.
    debit_sim = LAPDSim1D(r1c_params, r1c_flags)
    debit_state = debit_sim.state
    debit_cell = int(np.flatnonzero(debit_sim.geometry.plasma_active)[0])
    raw_n = debit_state.n.copy()
    raw_nn = debit_state.nn.copy()
    raw_nn_a = debit_state.nn_a.copy()
    raw_Ee = debit_state.Ee.copy()
    raw_Ei = debit_state.Ei.copy()
    raw_n[debit_cell] = 0.0
    raw_nn[debit_cell] = 0.0
    raw_nn_a[debit_cell] = 0.0
    raw_Ee[debit_cell] = 0.0
    raw_Ei[debit_cell] = 0.0
    debit_raw = ConservativeState1D(
        n=raw_n,
        nn=raw_nn,
        M=debit_state.M,
        Ee=raw_Ee,
        Ei=raw_Ei,
        M_n=debit_state.M_n,
        nn_a=raw_nn_a,
        M_n_a=debit_state.M_n_a,
    )
    ledger_before_probe = dict(debit_sim._floor_ledger)
    _, debit = debit_sim._floor_vector_with_ledger(pack_state(debit_raw))
    Vp_cell = debit_sim.geometry.plasma_volume_cm3[debit_cell]
    Vann_cell = (
        debit_sim.geometry.neutral_volume_cm3[debit_cell]
        - debit_sim.geometry.plasma_volume_cm3[debit_cell]
    )
    assert debit["n_particles_added"] == debit_sim.floors["n"] * Vp_cell
    assert debit["nn_particles_added"] == debit_sim.floors["nn"] * Vp_cell
    assert (
        debit["nn_a_particles_added"]
        == debit_sim.floors["nn"] * Vann_cell
    )
    assert debit["Ee_energy_added_erg"] == (
        1.5
        * debit_sim.floors["n"]
        * debit_sim.floors["Te"]
        * ev_to_erg
        * Vp_cell
    )
    assert debit["Ei_energy_added_erg"] == (
        1.5
        * debit_sim.floors["n"]
        * debit_sim.floors["Ti"]
        * ev_to_erg
        * Vp_cell
    )
    assert debit_sim._floor_ledger == ledger_before_probe

    # R1d configuration presence: valid R1 selectors perturb their intended
    # operator; the still-frozen compatibility controls are rejected as silent
    # no-ops pending their owning repair.
    import warnings as _dep_warnings
    for stale_param in (
        {"front_flux_model": "unregistered"},
        {"D_amb_model": "constant"},
        {"D_amb": 1.0},
        {"cathode_model": "enabled"},
    ):
        try:
            LAPDSim1D(dict(r1a_params, **stale_param), r1a_flags)
        except ValueError as error:
            assert "silent no-ops" in str(error)
        else:
            raise AssertionError(
                f"expected frozen surface-control rejection: {stale_param}"
            )
    # A13 (R3.3): the resolved-boundary surface-loss controls are now DEPRECATED
    # 0D artifacts -- non-default use warns loudly (no longer frozen, never a
    # silent no-op) because the resolved geometry measures per-electrode I_sat.
    for dep_params, dep_flags in (
        (dict(r1a_params, source_surface_area_scale=1.7), r1a_flags),
        (dict(r1a_params, end_surface_area_scale=0.9), r1a_flags),
        (r1a_params, dict(r1a_flags, source_surface_loss=False)),
        (r1a_params, dict(r1a_flags, end_surface_loss=False)),
    ):
        with _dep_warnings.catch_warnings(record=True) as _caught_dep:
            _dep_warnings.simplefilter("always")
            LAPDSim1D(dict(dep_params), dict(dep_flags))
        assert any(
            issubclass(w.category, DeprecationWarning)
            and "DEPRECATED 0D artifacts" in str(w.message)
            for w in _caught_dep
        ), f"expected A13 deprecation warning for {dep_params}/{dep_flags}"
    for birth_name, bad_value in (
        ("Te_birth_ionization", "bogus"),
        ("Ti_birth_ionization", -1.0),
        ("Te_birth_ionization", np.inf),
    ):
        try:
            LAPDSim1D(
                dict(r1a_params, **{birth_name: bad_value}), r1a_flags
            )
        except ValueError as error:
            assert birth_name in str(error)
        else:
            raise AssertionError(
                f"expected birth-selector rejection: {birth_name}={bad_value}"
            )

    topo_off = LAPDSim1D(
        r1a_params, dict(r1a_flags, active_plasma_topology=False)
    )
    topo_on = LAPDSim1D(r1a_params, r1a_flags)
    topo_dead = ~topo_on.geometry.plasma_active
    assert np.any(
        topo_off.reaction_rhs_terms()["ionization_birth"].n[topo_dead] != 0.0
    )
    assert np.all(
        topo_on.rhs_terms()["ionization_birth"].n[topo_dead] == 0.0
    )

    # Te_birth_ionization (local vs floor) only affects the electron birth
    # energy under the legacy birth model; it is inert (Ee birth = 0) under the
    # production "conservative" default, so exercise it on the legacy arm.
    birth_local = LAPDSim1D(
        dict(r1a_params, Te_birth_ionization="local",
             ionization_birth_energy_model="legacy"), r1a_flags
    )
    birth_floor = LAPDSim1D(
        dict(r1a_params, Te_birth_ionization="floor",
             ionization_birth_energy_model="legacy"), r1a_flags
    )
    local_Ee = birth_local.reaction_rhs_terms()["ionization_birth"].Ee
    floor_Ee = birth_floor.reaction_rhs_terms()["ionization_birth"].Ee
    assert np.any(local_Ee[birth_local.geometry.plasma_active] != floor_Ee[
        birth_floor.geometry.plasma_active
    ])

    raw_off = LAPDSim1D(
        r1c_params, dict(r1c_flags, raw_stage_validation=False)
    )
    raw_off_fields = state_field_names(raw_off.state)
    raw_off_row = raw_off_fields.index("nn_a")
    raw_off_cells = raw_off.geometry.cells

    def raw_off_rhs(y, time=None):
        rhs = np.zeros_like(y)
        start = raw_off_row * raw_off_cells
        rhs[start : start + raw_off_cells] = (
            -2.0 * np.asarray(y)[start : start + raw_off_cells] / r1c_dt
        )
        return rhs

    raw_off.rhs = raw_off_rhs
    raw_off_attempt = raw_off._attempt_step(
        dt=r1c_dt, operator_split=False
    )
    assert raw_off_attempt.raw_rejection_reason == ""
    assert raw_off_attempt.floor_ledger["nn_a_particles_added"] > 0.0

    # R1e exact resolved-config evidence: the machine-readable manifest
    # covers the authoritative registry, every config-complete campaign
    # driver matches its reviewed digest, and constructed config metadata
    # survives HDF5 exactly. No campaign integration is performed.
    from audit_sim1d_configs import config_cases, verify_snapshots
    from cablp.solvers._sim1d import config_manifest
    from cablp.solvers._sim1d.results.io import save_result_hdf5

    r1e_snapshots = verify_snapshots()
    r1e_manifest = config_manifest()
    r1e_default_params, r1e_default_flags = default_config()
    assert set(r1e_manifest["parameters"]) == set(r1e_default_params)
    assert set(r1e_manifest["flags"]) == set(r1e_default_flags)
    assert r1e_snapshots["parameter_count"] == len(r1e_default_params)
    assert r1e_snapshots["flag_count"] == len(r1e_default_flags)
    for unknown_params, unknown_flags in (
        ({"misspelled_campaign_knob": 1.0}, {}),
        ({}, {"inert_campaign_flag": True}),
    ):
        try:
            LAPDSim1D(unknown_params, unknown_flags)
        except ValueError as error:
            assert "silent/inert controls are forbidden" in str(error)
        else:
            raise AssertionError("unknown config key constructed silently")

    with tempfile.TemporaryDirectory() as r1e_dir:
        for case_name, (case_params, case_flags) in config_cases().items():
            case_sim = LAPDSim1D(case_params, case_flags)
            resolved_params, resolved_flags = case_sim.get_config()
            assert resolved_params == case_params
            assert resolved_flags == case_flags
            case_result = case_sim.run(t_end=0.0)
            case_path = Path(r1e_dir) / f"{case_name}.h5"
            save_result_hdf5(
                case_path,
                case_result,
                params=case_params,
                flags=case_flags,
            )
            case_loaded = load_result_hdf5(case_path)
            assert case_loaded.params == resolved_params
            assert case_loaded.flags == resolved_flags
            with h5py.File(case_path, "r") as case_h5:
                assert json.loads(case_h5.attrs["params_json"]) == resolved_params
                assert json.loads(case_h5.attrs["flags_json"]) == resolved_flags

        mismatch_params = dict(case_params)
        mismatch_params["Te_birth_ionization"] = (
            "local"
            if case_params["Te_birth_ionization"] == "floor"
            else "floor"
        )
        try:
            save_result_hdf5(
                Path(r1e_dir) / "metadata_mismatch.h5",
                case_result,
                params=mismatch_params,
                flags=case_flags,
            )
        except ValueError as error:
            assert "constructed LAPDSim1D config" in str(error)
        else:
            raise AssertionError("mismatched HDF5 config metadata was accepted")

    # Save-path config guard (2026-07-27): callers hold the PRE-resolution
    # override mapping they handed to LAPDSim1D, while result.params is the
    # POST-resolution config. The guard resolves both sides, so an equivalent
    # override set saves cleanly and a genuinely different one still raises.
    with tempfile.TemporaryDirectory() as guard_dir:
        guard_params = {"Te0": 0.22}
        guard_flags = {"cx": False}
        guard_sim = LAPDSim1D(guard_params, guard_flags)
        guard_resolved_params, guard_resolved_flags = guard_sim.get_config()
        assert guard_resolved_params != guard_params
        assert guard_resolved_flags != guard_flags
        guard_result = guard_sim.run(t_end=0.0)

        # (a) the pre-resolution inputs that produced the run must not raise,
        # and the resolved config is still what gets written.
        guard_path = Path(guard_dir) / "pre_resolution.h5"
        save_result_hdf5(
            guard_path,
            guard_result,
            params=guard_params,
            flags=guard_flags,
        )
        with h5py.File(guard_path, "r") as guard_h5:
            assert json.loads(guard_h5.attrs["params_json"]) == guard_resolved_params
            assert json.loads(guard_h5.attrs["flags_json"]) == guard_resolved_flags

        # (b) a genuinely different config still raises, in either namespace,
        # and names the differing key.
        for guard_kind, bad_params, bad_flags in (
            ("params", {"Te0": 0.23}, guard_flags),
            ("flags", guard_params, {"cx": True}),
        ):
            try:
                save_result_hdf5(
                    Path(guard_dir) / f"guard_mismatch_{guard_kind}.h5",
                    guard_result,
                    params=bad_params,
                    flags=bad_flags,
                )
            except ValueError as error:
                assert f"{guard_kind} metadata differs" in str(error)
                assert ("Te0" if guard_kind == "params" else "cx") in str(error)
            else:
                raise AssertionError(
                    f"differing {guard_kind} metadata was accepted on save"
                )

        # (c) the params=None / flags=None pass-through is unchanged: the
        # constructed config is written without any caller-side comparison.
        guard_none_path = Path(guard_dir) / "pass_through.h5"
        save_result_hdf5(guard_none_path, guard_result)
        with h5py.File(guard_none_path, "r") as guard_h5:
            assert json.loads(guard_h5.attrs["params_json"]) == guard_resolved_params
            assert json.loads(guard_h5.attrs["flags_json"]) == guard_resolved_flags
        save_result_hdf5(
            Path(guard_dir) / "params_only.h5",
            guard_result,
            params=guard_params,
        )
        save_result_hdf5(
            Path(guard_dir) / "flags_only.h5",
            guard_result,
            flags=guard_flags,
        )

    # R1 startup/rate-domain follow-up: the repaired live defaults are above
    # their hard floors and the exact bundled ADF11 edge. The proactive
    # resolved-source bound makes raw rejection a backstop and leaves the
    # accepted-only floor ledger exactly null through plasma launch.
    repaired_params, repaired_flags = default_config()
    adas_te_min, adas_te_max = he_rate_temperature_range_eV()
    assert repaired_flags["active_plasma_topology"] is True
    assert repaired_flags["raw_stage_validation"] is True
    assert repaired_params["Te0"] == 0.21
    assert repaired_params["Ti0"] == 0.026
    assert repaired_params["Te0"] > adas_te_min
    assert adas_te_max > repaired_params["Te0"]
    for bad_seed in (
        {"Te0": repaired_params["Te_floor"]},
        {"Ti0": repaired_params["Ti_floor"]},
    ):
        try:
            LAPDSim1D(dict(repaired_params, **bad_seed), repaired_flags)
        except ValueError as error:
            assert "strictly greater" in str(error)
        else:
            raise AssertionError(
                f"raw-stage repaired config accepted floor-bound seed {bad_seed}"
            )

    startup_params, startup_flags = config_cases()["compare_sim1d_es1"]
    startup_sim = LAPDSim1D(startup_params, startup_flags)
    startup_result = startup_sim.run(t_end=2.03e-3)
    # Pristine-startup assertions (0 rejections, 0 floor activity) DEFERRED to
    # the ES1 tuning pass (R5 stance flip, 2026-07-25). Under the repaired stance
    # the compare_sim1d_es1 startup shows minor, EXPECTED activity: a couple of
    # timestep rejections (the 2nd-order strang/tr_bdf2 split + Phelps presheath)
    # and small Ei-floor clipping (the Ti floor was relaxed to 300 K, so Ti can
    # now reach it near the cold Ti0 -- impossible at the old 0.1 eV floor). Both
    # are negligible (~2 rejections, ~17 erg Ei over 2 ms). The ES config is not
    # finalized (geometry + V_bank=180 circuit refit deferred), and startup
    # cleanliness is validated there. Soft-bound here so it does not regress
    # badly. See R5_STANCE_FLIP_HANDOFF.md.
    assert len(startup_result.timestep_rejection_events["time"]) < 20
    _startup_floor = sum(abs(v) for v in startup_result.floor_ledger.values())
    assert _startup_floor < 1.0e4  # erg, negligible vs the multi-kW plasma
    source_bounds = [
        diag.dt_surface_loss
        for diag in startup_result.diagnostics
        if diag.time >= startup_params["tau_neutral_prebreakdown"]
    ]
    assert source_bounds
    assert np.all(np.isfinite(source_bounds))
    assert any(
        diag.active_constraint == "surface_loss"
        for diag in startup_result.diagnostics
    )
    rate_domain = startup_result.atomic_rate_domain
    assert rate_domain["table_Te_min_eV"] == adas_te_min
    assert rate_domain["table_Te_max_eV"] == adas_te_max
    assert np.all(rate_domain["active_cell_fraction_below"] == 0.0)
    assert np.all(rate_domain["active_volume_fraction_below"] == 0.0)

    with tempfile.TemporaryDirectory() as rate_dir:
        rate_path = Path(rate_dir) / "rate-domain.h5"
        save_result_hdf5(rate_path, startup_result)
        loaded_rate = load_result_hdf5(rate_path)
        assert set(loaded_rate.atomic_rate_domain) == set(rate_domain)
        for name, expected in rate_domain.items():
            loaded_value = np.asarray(loaded_rate.atomic_rate_domain[name])
            expected_value = np.asarray(expected)
            if expected_value.dtype.kind in {"U", "S", "O"}:
                assert np.array_equal(loaded_value, expected_value)
            else:
                assert np.array_equal(
                    loaded_value,
                    expected_value,
                    equal_nan=True,
                )
        with h5py.File(rate_path, "a") as rate_h5:
            del rate_h5["atomic_rate_domain"]
        assert load_result_hdf5(rate_path).atomic_rate_domain == {}

    print(
        "sim1d smoke ok: "
        f"cells={geom.cells}, dz={geom.dz_cm:g} cm, "
        f"Vp_total={geom.plasma_volume_cm3.sum():.6e} cm^3, "
        f"Vm_total={geom.neutral_volume_cm3.sum():.6e} cm^3"
    )


if __name__ == "__main__":
    main()
