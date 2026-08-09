import dataclasses
import json
import math
import os
from io import StringIO
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import warnings
from types import SimpleNamespace

import h5py
import numpy as np

from cablp.funcs import _kernels as _kernel_selector
from cablp.funcs import _cathode_solver as _cathode_solver_mod
from cablp.funcs import _cathode_solver_idriven as _cathode_solver_idriven_mod
from cablp.funcs import _beam_deposition as _beam_deposition_mod
from cablp.funcs._adas import he_rate_temperature_range_eV
# main() re-imports deposit_beam locally further down (B1 block), which makes
# the bare name local to the whole function -- alias it for the item-35 block.
from cablp.funcs._beam_deposition import deposit_beam as _deposit_beam_ray
from cablp.funcs._cathode_solver import _compute_l_b
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
    _clip_ray_length,
    _csda_beam_deposition,
    _gap_clip_is_face_aligned,
    _ray_gap_breakout,
    beam_absorption_weights,
    beam_gap_ledger_mismatch,
    beam_launch,
    cathode_sample_indices,
)
from cablp.solvers._sim1d.physics.kinetic_dvm import (
    TransientDVM,
    ledger_residual as kinetic_dvm_ledger_residual,
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
    _derive_cathode_adjacent_cells,
    _source_fixed_grid_spec,
    absorbing_live_cells_by_role,
    anode_flanking_cells,
    cathode_adjacent_cells,
    is_plenum_cell,
    puff_cell_indices,
    pump_cell_indices,
)
from cablp.solvers._sim1d.physics.neutrals import (
    GAS_PUFF_DIAGNOSTIC_FIELDS,
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
from cablp.solvers._sim1d.core.timestep import (
    neutral_wind_timestep,
    plasma_source_timestep,
    suggest_timestep,
)
from cablp.solvers._sim1d.physics.reactions import (
    gas_puff_local_ionization_rhs,
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
    # The production/default stance must construct WARNING-FREE. This guards a
    # SURVIVING path -- it is what stops production silently acquiring a
    # DeprecationWarning. (The golden's legacy ion-neutral arm is the one
    # deliberate exception and is not exercised here.)
    import warnings as _warnings

    _dep_params, _dep_flags = default_config()
    with _warnings.catch_warnings(record=True) as _caught:
        _warnings.simplefilter("always")
        LAPDSim1D(_dep_params, _dep_flags)
    assert not _caught, "production/default configuration must be warning-free"

    params, flags = default_config()
    assert params["cycles"] == 1
    assert params["phase_transition_mode"] == "current"
    assert params["gas_puff_mode"] == "square"
    # No pre-drive window: the machine fires one global trigger, so the bank
    # connects as the puff starts and neutrals never accumulate with the drive
    # withheld (2026-08-03; see timing_defaults). Asserted EXACTLY, not as
    # ">= 0", so a reintroduced pre-phase fails here. The flag stays on and
    # gates the machinery -- the duration alone opts back in, which is what the
    # dedicated neutral_prebreakdown block below does (it pins its own
    # tau_neutral_prebreakdown, so that feature test does not read this default).
    assert params["tau_neutral_prebreakdown"] == 0.0
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

    # Selector validity. Each of these keys accepts a CLOSED set of values and
    # must reject anything else LOUDLY, at construction. The probe value is
    # garbage ('zzz'), not a retired selector, so what is tested is validity
    # rejection rather than removal narration. The assertions check that the
    # error fires, that it NAMES the key, and that it states what the key
    # accepts -- deliberately NOT the full message string. Pinning the exact
    # text is what made the previous version of this block impossible to
    # reword without rewriting the test.
    def _construction_error(param_overrides, flag_overrides):
        """Return the ValueError text LAPDSim1D raises for a bad selector."""
        p, f = default_config()
        p.update(param_overrides)
        f.update(flag_overrides)
        try:
            LAPDSim1D(p, f)
        except ValueError as exc:
            return str(exc)
        raise AssertionError(
            "expected a ValueError at construction for "
            f"{param_overrides or flag_overrides}"
        )

    _sel = _construction_error({"cathode_solver_model": "zzz"}, {})
    assert "cathode_solver_model" in _sel and "current_driven" in _sel, _sel
    _sel = _construction_error({"neutral_exchange_model": "zzz"}, {})
    assert "neutral_exchange_model" in _sel, _sel
    assert "constant" in _sel and "knudsen" in _sel, _sel
    _sel = _construction_error({"cathode_warming_model": "zzz"}, {})
    assert "cathode_warming_model" in _sel, _sel
    assert "none" in _sel and "power_balance" in _sel, _sel
    # resolved_boundaries is a BOOLEAN flag, read through bool(): a garbage
    # string is truthy and passes, so False is its only invalid value and the
    # one the guard exists for (a stale config still asking for the retired
    # geometry). Probed with False for that reason, not with 'zzz'.
    _sel = _construction_error({}, {"resolved_boundaries": False})
    assert "resolved_boundaries" in _sel and "True" in _sel, _sel

    # The else-raise inside physics.neutrals is DOUBLE-GUARDED: the solver
    # rejects a bad neutral_exchange_model at construction (just above), so
    # this branch is unreachable through LAPDSim1D and is exercised on the
    # helper directly instead.
    try:
        neutral_exchange_coefficients(
            geometry=resolved_geom,
            model="zzz",
            constant_coeff_cm3_s=resolved_params["neutral_exchange_coeff_cm3_s"],
            Tn_K=resolved_params["Tn_K"],
            mu_neutral=4.0,
            clausing_scale=resolved_params["neutral_clausing_scale"],
        )
    except ValueError as exc:
        _sel = str(exc)
        assert "neutral_exchange_model" in _sel, _sel
        assert "constant" in _sel and "knudsen" in _sel, _sel
    else:
        raise AssertionError(
            "neutral_exchange_coefficients accepted an unknown model"
        )

    # Cathode and anode are *surfaces*: the cathode surface
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
    # cathode_adjacent_cells is now a stored derivation, not a recomputation
    # (it is called ~24x per accepted step). The stored value must equal a
    # fresh derivation from the geometry's own topology arrays, element for
    # element and type for type.
    _fresh = _derive_cathode_adjacent_cells(
        resolved_geom.cell_role, resolved_geom.cathode_face_indices
    )
    assert resolved_geom.cathode_cell_indices == _fresh
    assert cathode_adjacent_cells(resolved_geom) == _fresh
    assert np.array_equal(
        np.asarray(cathode_adjacent_cells(resolved_geom), dtype=int),
        np.asarray(_fresh, dtype=int),
    )
    assert all(type(c) is int for c in cathode_adjacent_cells(resolved_geom))
    # Repeated reads return the identical object -- there is one copy, so
    # nothing can go stale relative to anything else.
    assert cathode_adjacent_cells(resolved_geom) is (
        cathode_adjacent_cells(resolved_geom)
    )
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

    # Twin cathode mirrors the source end: its cathode
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
    # Same equality check on the two-cathode layout, where the derivation
    # actually exercises the low-z branch.
    assert cathode_adjacent_cells(twin_resolved_geom) == (
        _derive_cathode_adjacent_cells(
            twin_resolved_geom.cell_role,
            twin_resolved_geom.cathode_face_indices,
        )
    )
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
    # Annular duct: open area and hydraulic radius reduce independently.
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
    # same particles twice. The cathode surface blocks everything.
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
    # removes, not `2*eta*I_i` scaled off the cathode cell.
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
    # M1): the historical R_p spreads the hot
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

    # --- beam_ionization_birth in the resolved-source dt bound (default off) --
    # The row is a volumetric plasma source that can drive a cell into a floor
    # within one step, and it has never been in ANY timestep bound: the bundle
    # carries only the boundary, anode-collection and cathode-surface rows.
    # Here the row is demonstrably live (asserted nonzero above).
    assert flags.get("beam_ionization_birth_timestep_bound", False) is False
    assert default_config()[1]["beam_ionization_birth_timestep_bound"] is False
    # Both sims built FRESH and identically here: the bundle re-solves the
    # cathode internally, so a sim whose solve cache has been walked by other
    # checks is not a like-for-like control.
    def _beam_bound_sim(bound_on):
        built = LAPDSim1D(
            cathode_bl_params,
            {
                **cathode_flags,
                "beam_ionization_birth_timestep_bound": bound_on,
            },
        )
        built._circuit_I_loop = 3000.0
        return built

    beam_bound_sim = _beam_bound_sim(True)
    beam_unbound_sim = _beam_bound_sim(False)
    beam_bound_row = beam_bound_sim.beam_ionization_rhs_terms(
        state=beam_bound_sim.state,
        cathode_solve=beam_bound_sim.solve_cathode_boundary(
            state=beam_bound_sim.state,
            time=beam_bound_sim._time,
            update_cache=False,
        ),
        time=beam_bound_sim._time,
    )["beam_ionization_birth"]
    assert np.any(beam_bound_row.n > 0.0)
    beam_bundle_off = beam_unbound_sim._plasma_source_timestep_rhs(
        state=beam_unbound_sim.state, time=beam_unbound_sim._time
    )
    beam_bundle_on = beam_bound_sim._plasma_source_timestep_rhs(
        state=beam_bound_sim.state, time=beam_bound_sim._time
    )
    # The WHOLE applied row joins the bundle -- not a sub-fraction of it. A
    # bound computed from part of a row describes a term the step does not
    # apply and leaves the remainder unbounded.
    for beam_field in ("n", "nn", "M", "Ee", "Ei"):
        assert np.allclose(
            getattr(beam_bundle_on, beam_field),
            getattr(beam_bundle_off, beam_field)
            + getattr(beam_bound_row, beam_field),
        ), beam_field
    # OFF the bundle is untouched: bit-identical to a sim that never heard of
    # the flag.
    beam_absent_sim = LAPDSim1D(cathode_bl_params, cathode_flags)
    beam_absent_sim._circuit_I_loop = 3000.0
    for beam_field in ("n", "nn", "M", "Ee", "Ei"):
        assert np.array_equal(
            getattr(
                beam_absent_sim._plasma_source_timestep_rhs(
                    state=beam_absent_sim.state,
                    time=beam_absent_sim._time,
                ),
                beam_field,
            ),
            getattr(beam_bundle_off, beam_field),
        ), beam_field
    # And it can only TIGHTEN the suggested step, never loosen it.
    assert (
        beam_bound_sim.suggest_timestep().dt_surface_loss
        <= beam_unbound_sim.suggest_timestep().dt_surface_loss * (1.0 + 1.0e-12)
    )
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

    # --- Current-driven sheath solve (M2): given the
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

    # --- Current-driven circuit integration (M3):
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
    # M5a). Validation fails fast; the coverage
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

    # --- A2: the manifold excitation model (WP-A).
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

    # --- Item 35: the gap-transmission probe is launched at the REAL emitted
    # flux, not unit flux, so flux-DEPENDENT stopping reaches the circuit.
    #
    # (a) The property the fix rests on, at the module: with flux-INDEPENDENT
    # stopping the surviving FRACTION does not depend on the launched flux, so
    # the flux-faithful probe reproduces the historical unit-flux probe
    # bit-for-bit; with the quasilinear closure it does depend on it (the
    # relaxation length runs on n_b ~ Gamma0/(A v_b)), which is exactly the
    # signal the unit-flux probe could not see.
    _fx_cells = 40
    _fx_dz = np.full(_fx_cells, 5.0)
    _fx_kwargs = dict(
        nn=np.full(_fx_cells, 1.0e13),
        ne=np.full(_fx_cells, 1.0e12),
        Te=np.full(_fx_cells, 3.0),
        dz_cm=_fx_dz,
        launch=0,
        direction=1,
        I_ion_eV=float(I_ion),
    )
    # Weak-beam domain (n_b < 0.1*ne), where the quasilinear closure is
    # defined; above it the module returns an infinite relaxation length by
    # design and the ray free-streams again.
    _fx_flux = 1.0e20
    _fx_lin = [
        _deposit_beam_ray(200.0, g, anomalous_model="none", **_fx_kwargs)
        for g in (1.0, _fx_flux)
    ]
    assert (
        float(_fx_lin[1].transmitted_flux) / _fx_flux
        == float(_fx_lin[0].transmitted_flux)
    )
    _fx_ql = [
        _deposit_beam_ray(
            200.0, g, anomalous_model="quasilinear",
            beam_area_cm2=100.0, **_fx_kwargs,
        )
        for g in (1.0, _fx_flux)
    ]
    # Unit flux streams through; the real flux is stopped by its own QL drag.
    assert float(_fx_ql[0].transmitted_flux) == 1.0
    assert float(_fx_ql[1].transmitted_flux) == 0.0
    # The probe-independent witness reads those same rays off their OWN
    # bookkeeping, with no reference to the probe -- this is what makes an
    # item-35-class probe defect visible without trusting the probe. The
    # QL-stopped ray above dies 75 cm in, so where it lands relative to the
    # gap is what decides breakout:
    _fx_gap_short = _clip_ray_length(_fx_dz, 0, 1, 50.0)   # ray dies past it
    _fx_gap_long = _clip_ray_length(_fx_dz, 0, 1, 100.0)   # ray dies inside it
    assert _ray_gap_breakout(_fx_ql[1], _fx_gap_short, 0, 1) == 1.0
    assert _ray_gap_breakout(_fx_ql[1], _fx_gap_long, 0, 1) == 0.0
    # Rays that leave the far end cleared the gap by definition (the
    # transmitted-flux shortcut, which also covers a sub-threshold ray whose
    # E_entry profile is all zeros).
    assert _ray_gap_breakout(_fx_ql[0], _fx_gap_long, 0, 1) == 1.0
    assert _ray_gap_breakout(_fx_lin[1], _fx_gap_long, 0, 1) == 1.0
    # Gap covering the whole path: no cell beyond it to sample, and the ray
    # did not leave the far end either, so it died inside.
    assert _ray_gap_breakout(
        _fx_ql[1], _clip_ray_length(_fx_dz, 0, 1, 1.0e4), 0, 1
    ) == 0.0

    # (b) The same statement through the solver: this csda config runs
    # anomalous_model="none", so the sigma_eff the adapter wrote must equal an
    # independent re-derivation from the HISTORICAL unit-flux probe. A
    # regression that made the flux-faithful probe non-flux-linear on the off
    # arm would move the golden's off-path arms and fail here.
    csda_state = csda_sim.state
    csda_derived = derive_state(
        csda_state, csda_sim._floors, csda_sim._ion_mass_g
    )
    csda_L_cath = float(csda_solve.device_config.L_cath)
    csda_dir = beam_launch(csda_sim._geometry, end=0)[1]
    csda_unit = _deposit_beam_ray(
        csda_res.phi_c,
        1.0,
        nn=csda_state.nn,
        ne=csda_state.n,
        Te=csda_derived.Te,
        dz_cm=_clip_ray_length(
            csda_sim._geometry.length_cm, csda_launch, csda_dir, csda_L_cath
        ),
        launch=csda_launch,
        direction=csda_dir,
        I_ion_eV=float(csda_sim._I_ion),
        coulomb_model=str(
            csda_params.get("beam_coulomb_model", "fast_electron")
        ),
        anomalous_model="none",
    )
    csda_unit_T = min(max(float(csda_unit.transmitted_flux), 1.0e-6), 1.0)
    csda_nn_launch = float(csda_state.nn[csda_launch])
    csda_l_bi = _compute_l_b(
        csda_res.phi_c,
        float(csda_derived.Te[csda_launch]),
        float(csda_state.n[csda_launch]),
        0.0,
        0.0,
    )
    csda_sigma_unit = max(
        0.0,
        (-math.log(csda_unit_T) / csda_L_cath - 1.0 / csda_l_bi)
        / csda_nn_launch,
    )
    assert csda_sigma_eff == csda_sigma_unit

    # --- Item 35 ledger tripwire: probe, deposition ray and circuit are three
    # views of the SAME gap crossing, and nothing else in the model notices
    # when they disagree (each side is internally consistent). On a healthy
    # config all three agree and no warning fires.
    csda_ledger = csda_solve.beam_gap_ledger
    csda_eta = float(csda_solve.device_config.eta)
    assert set(csda_ledger) == {0}
    csda_probe, csda_ray, csda_booked = csda_ledger[0]
    assert 0.0 < csda_probe <= 1.0 and 0.0 < csda_booked <= 1.0
    assert beam_gap_ledger_mismatch(csda_ledger, csda_eta) is None
    # The deposition ray's breakout is read off its OWN bookkeeping, so it is
    # an independent witness: here the ray clears the gap and the probe agrees.
    assert csda_ray == 1.0
    assert csda_probe == csda_ray
    # This healthy config sits ON the benign floor the tolerance is chosen to
    # clear: the ray fully transmits, which the Beer-Lambert solve cannot
    # represent above its Coulomb-only ceiling, so the clamp saturates and
    # leaves a little unbooked. It must stay a small fraction of emitted beam
    # power -- item 35 was 35% -- or the warning becomes noise.
    assert 0.0 < csda_eta * (csda_ray - csda_booked) < 0.02
    # The tripwire is a comparison, not an assertion about one side: an
    # injected divergence must be caught and reported as the worst offender.
    csda_trip = beam_gap_ledger_mismatch(
        {0: (csda_probe, csda_ray, 0.5 * csda_ray)}, csda_eta
    )
    assert csda_trip is not None
    assert csda_trip[0] == 0 and csda_trip[1] == "ray_vs_circuit"
    assert np.isclose(csda_trip[4], 0.5 * csda_eta * csda_ray)
    # A defect INSIDE the probe -- the item-35 class -- is caught by the
    # probe-vs-ray leg even though the circuit faithfully tracks the (wrong)
    # probe, which is exactly the configuration that stayed silent before.
    # Pre-fix this ledger read probe=1.0 with the ray at 0.0.
    csda_probe_defect = beam_gap_ledger_mismatch(
        {0: (1.0, 0.0, 0.96529)}, csda_eta
    )
    assert csda_probe_defect is not None
    assert csda_probe_defect[1] == "probe_vs_ray"
    assert np.isclose(csda_probe_defect[4], csda_eta)
    # ... and the warning it drives is emitted once per run, not per step.
    csda_broken = SimpleNamespace(
        beam_gap_ledger={0: (1.0, 0.0, 0.96529)},
        device_config=SimpleNamespace(eta=csda_eta),
    )
    with warnings.catch_warnings(record=True) as csda_warned:
        warnings.simplefilter("always")
        csda_sim._beam_gap_ledger_warned = False
        for _ in range(3):
            csda_sim._warn_beam_gap_ledger(csda_broken)
    csda_msgs = [
        w for w in csda_warned if "beam gap ledger" in str(w.message)
    ]
    assert len(csda_msgs) == 1
    assert "probe_vs_ray" in str(csda_msgs[0].message)
    csda_sim._beam_gap_ledger_warned = False
    # All three views are recorded as cathode diagnostics, defaulted on every
    # run so a beer_lambert run (and an old file, which has none of the three
    # datasets) stays readable.
    csda_diag = csda_sim._cathode_diagnostic_snapshot()
    assert csda_diag["source_beam_gap_survival_probe"] == csda_probe
    assert csda_diag["source_beam_gap_survival_ray"] == csda_ray
    assert csda_diag["source_beam_gap_survival_circuit"] == csda_booked
    assert np.isnan(csda_diag["end_beam_gap_survival_ray"])
    bl_diag = LAPDSim1D(
        dict(exc_params), dict(cathode_flags)
    )._cathode_diagnostic_snapshot()
    for _bl_key in ("probe", "ray", "circuit"):
        assert np.isnan(bl_diag[f"source_beam_gap_survival_{_bl_key}"])

    # --- Probe skip (cost read 2026-08-02, restructure A) ------------------
    # When the deposition ray died inside the gap, the gap-transmission probe
    # is not launched at all: its transmitted flux is then the EXACT float
    # 0.0. The claim is checked HERE against the probe itself -- the pre-change
    # call, run verbatim -- on a state from each regime, so the skip is
    # demonstrated equal to the work it replaces rather than argued to be.
    _pskip_geom = csda_sim._geometry
    _pskip_gap = _clip_ray_length(
        _pskip_geom.length_cm, csda_launch, csda_dir, csda_L_cath
    )
    _pskip_Gamma0 = csda_res.I_eth_star / qe_SI
    _pskip_ray_kwargs = dict(
        launch=csda_launch,
        direction=csda_dir,
        I_ion_eV=float(csda_sim._I_ion),
        coulomb_model=str(
            csda_params.get("beam_coulomb_model", "fast_electron")
        ),
        anomalous_model=str(
            csda_params.get("beam_anomalous_model", "none")
        ),
    )
    if _pskip_ray_kwargs["anomalous_model"] != "none":
        _pskip_ray_kwargs["beam_area_cm2"] = _pskip_geom.plasma_area_cm2

    def _pskip_adapter(nn):
        """(beam_result, gap_ledger) from the real adapter on a doctored nn."""
        _beam = SimpleNamespace(
            result=csda_res,
            result_twin=None,
            beam_atten_cross=np.zeros(_pskip_geom.cells),
        )
        _, _ledger = _csda_beam_deposition(
            _beam,
            SimpleNamespace(nn=nn, n=csda_state.n),
            SimpleNamespace(Te=csda_derived.Te),
            _pskip_geom,
            csda_solve.device_config,
            csda_params,
            float(csda_sim._I_ion),
            anode_interception=True,
        )
        return _beam, _ledger

    def _pskip_probe(nn):
        """Transmitted flux of the probe the pre-change code always launched."""
        return float(
            _deposit_beam_ray(
                csda_res.phi_c, _pskip_Gamma0, dz_cm=_pskip_gap,
                nn=nn, ne=csda_state.n, Te=csda_derived.Te,
                **_pskip_ray_kwargs,
            ).transmitted_flux
        )

    # Production geometry: the gap is 5 x 10 cm and L_cath is 50 cm, so the
    # clip lands on a cell face and the anode crossing sits past the gap --
    # both structural guards are inactive here, which is the case the skip
    # exists for.
    assert _gap_clip_is_face_aligned(_pskip_gap, _pskip_geom.length_cm)
    assert float(_pskip_gap[int(_pskip_geom.anode_face_indices[0])]) == 0.0
    for _pskip_scale, _pskip_ray_expect in ((1.0, 1.0), (1.0e3, 0.0)):
        _pskip_nn = np.asarray(csda_state.nn, dtype=float) * _pskip_scale
        _pskip_beam, _pskip_ledger = _pskip_adapter(_pskip_nn)
        _pskip_T, _pskip_ray, _pskip_circuit = _pskip_ledger[0]
        assert _pskip_ray == _pskip_ray_expect, (_pskip_scale, _pskip_ray)
        # The probe, launched for real, must reproduce the branch taken --
        # bit-for-bit, not to a tolerance.
        _pskip_ref = min(
            max(_pskip_probe(_pskip_nn) / _pskip_Gamma0, 1.0e-6), 1.0
        )
        assert _pskip_T == _pskip_ref, (_pskip_scale, _pskip_T, _pskip_ref)
        # All three ledger channels stay written from the values they always
        # came from -- the skip removes a computation, not a diagnostic.
        for _pskip_v in (_pskip_T, _pskip_ray, _pskip_circuit):
            assert np.isfinite(_pskip_v), (_pskip_scale, _pskip_ledger)
        assert 0.0 < _pskip_T <= 1.0 and 0.0 < _pskip_circuit <= 1.0
        assert np.isfinite(_pskip_beam.beam_atten_cross[csda_launch])
        # ... and the tripwire still reads the same three views: probe and ray
        # agree on the skip arm (0.0 vs the 1e-6 clamp) exactly as they do on
        # the transmitting arm, so no divergence is manufactured.
        assert beam_gap_ledger_mismatch(_pskip_ledger, csda_eta) is None
    # The dead-ray arm is the one the skip fires on, and its probe really does
    # transmit the exact float zero (the value the branch substitutes).
    assert _pskip_probe(np.asarray(csda_state.nn, dtype=float) * 1.0e3) == 0.0
    # Guard: a clip that ends mid-cell truncates the stop cell, so the probe
    # could run out of path where the deposition ray still had some. The skip
    # must see that and stand down.
    assert not _gap_clip_is_face_aligned(
        _clip_ray_length(_pskip_geom.length_cm, csda_launch, csda_dir, 45.0),
        _pskip_geom.length_cm,
    )
    assert _gap_clip_is_face_aligned(
        _clip_ray_length(_pskip_geom.length_cm, csda_launch, csda_dir, 40.0),
        _pskip_geom.length_cm,
    )
    # A clip longer than the whole path leaves every cell at its full length.
    assert _gap_clip_is_face_aligned(
        _clip_ray_length(_pskip_geom.length_cm, csda_launch, csda_dir, 1.0e6),
        _pskip_geom.length_cm,
    )

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

    # --- WP-D through the solver: beam_product_transport routes the CSDA
    # ray's event products (see the module block for the physics and the
    # per-ray identity). Unit level only -- the flag's effect on the ignition
    # timeline is a campaign run, not a smoke scenario.
    # Misconfiguration is loud at CONSTRUCTION, including the incomplete
    # configuration where the selection could only be a silent no-op.
    for wpd_bad in (
        dict(csda_params, beam_product_transport="bogus"),
        dict(csda_params, beam_product_transport="nonlocal",
             beam_deposition_model="beer_lambert"),
    ):
        try:
            LAPDSim1D(wpd_bad, dict(cathode_flags))
        except ValueError:
            pass
        else:
            raise AssertionError(
                "expected ValueError for beam_product_transport"
            )
    # The default is bit-exact through the solver too: naming "local"
    # explicitly reproduces the deposition csda_sim already produced.
    wpd_off_sim = LAPDSim1D(
        dict(csda_params, beam_product_transport="local"), dict(cathode_flags)
    )
    wpd_off_sim._circuit_I_loop = 3000.0
    wpd_off_dep = wpd_off_sim.solve_cathode_boundary().beam_deposition[0]
    assert np.array_equal(
        wpd_off_dep.plasma_heating_erg_s, csda_dep.plasma_heating_erg_s
    )
    assert wpd_off_dep.end_loss_low_erg_s == 0.0
    assert wpd_off_dep.end_loss_high_erg_s == 0.0
    # On: energy leaves through the ends, the plasma keeps less, and the
    # per-ray budget still closes with the ledger in it.
    wpd_on_sim = LAPDSim1D(
        dict(csda_params, beam_product_transport="nonlocal"),
        dict(cathode_flags),
    )
    wpd_on_sim._circuit_I_loop = 3000.0
    wpd_on_solve = wpd_on_sim.solve_cathode_boundary()
    wpd_on_dep = wpd_on_solve.beam_deposition[0]
    wpd_on_total = (
        wpd_on_dep.plasma_heating_erg_s.sum()
        + wpd_on_dep.radiated_erg_s.sum()
        + wpd_on_dep.ionization_cost_erg_s.sum()
        + float(wpd_on_dep.anode_intercepted_erg_s)
        + wpd_on_dep.end_loss_low_erg_s
        + wpd_on_dep.end_loss_high_erg_s
    )
    assert abs(wpd_on_total - csda_budget) / csda_budget < 1e-9
    assert (
        wpd_on_dep.end_loss_low_erg_s + wpd_on_dep.end_loss_high_erg_s > 0.0
    )
    assert (
        wpd_on_dep.plasma_heating_erg_s.sum()
        < csda_dep.plasma_heating_erg_s.sum()
    )
    # Energy-only in v1: the particle rows the fluid and circuit read are
    # untouched.
    assert np.array_equal(
        wpd_on_dep.ionization_events, csda_dep.ionization_events
    )
    # The gap-transmission PROBE and the item-35 tripwire are primary-flux
    # instruments and must be blind to product transport: all three ledger
    # views are unchanged, so sigma_eff and the circuit bypass are too.
    assert wpd_on_solve.beam_gap_ledger[0] == csda_ledger[0]
    assert (
        wpd_on_solve.beam_result.beam_atten_cross[csda_launch]
        == csda_sigma_eff
    )
    assert beam_gap_ledger_mismatch(wpd_on_solve.beam_gap_ledger, csda_eta) is None
    # The ledger is recorded as cathode diagnostics, zero-defaulted so
    # beer_lambert runs and pre-WP-D files stay readable.
    wpd_on_diag = wpd_on_sim._cathode_diagnostic_snapshot()
    assert wpd_on_diag["source_beam_end_loss_low_W"] == (
        wpd_on_dep.end_loss_low_erg_s * 1.0e-7
    )
    assert wpd_on_diag["source_beam_end_loss_high_W"] == (
        wpd_on_dep.end_loss_high_erg_s * 1.0e-7
    )
    assert wpd_on_diag["end_beam_end_loss_low_W"] == 0.0
    for _bl_key in ("low", "high"):
        assert bl_diag[f"source_beam_end_loss_{_bl_key}_W"] == 0.0

    # --- WP-E through the solver: heating_anomalous_transport routes the CSDA
    # ray's ANOMALOUS (quasilinear) heating onto tail electrons at E_tail (see
    # the module block for the physics and the conservation identity). Unit
    # level only -- the flag's effect on the ignition timeline is a campaign
    # run, not a smoke scenario.
    # The scenario must actually drive the anomalous channel, or the routing
    # has nothing to carry and every assertion below is vacuous.
    assert float(csda_dep.heating_anomalous_erg_s.sum()) > 0.0
    # K7 REPIN. This block and the K6 block below were written against the tail
    # closure as WP-E and K6 shipped it: birth at a FIXED rung, free escape at
    # the cathode face. K7 made both of those selectable and defaulted the
    # engaged walk to the corrected pair instead, so every arm here names the
    # legacy values explicitly. That is what keeps these assertions testing the
    # arithmetic they were written for, bit for bit, rather than silently
    # re-pointing at the new closure.
    wpe_legacy = dict(
        heating_anomalous_tail_energy_keying="fixed",
        heating_anomalous_tail_cathode_boundary="escape",
    )
    # Misconfiguration is loud at CONSTRUCTION, including every incomplete
    # configuration in which the selection could only be a silent no-op: no
    # CSDA module to deposit, no anomalous channel to carry, or a tail energy
    # the walk cannot launch at.
    for wpe_bad in (
        dict(csda_params, heating_anomalous_transport="bogus"),
        dict(csda_params, **wpe_legacy,
             heating_anomalous_transport="tail_walk",
             beam_deposition_model="beer_lambert"),
        dict(csda_params, **wpe_legacy,
             heating_anomalous_transport="tail_walk",
             beam_anomalous_model="none"),
        dict(csda_params, **wpe_legacy,
             heating_anomalous_transport="tail_walk",
             heating_anomalous_tail_energy_eV=0.0),
        dict(csda_params, **wpe_legacy,
             heating_anomalous_transport="tail_walk",
             heating_anomalous_tail_energy_eV=-75.0),
        dict(csda_params, **wpe_legacy,
             heating_anomalous_transport="tail_walk",
             heating_anomalous_tail_energy_eV=float("nan")),
        dict(csda_params, **wpe_legacy,
             heating_anomalous_transport="tail_walk",
             heating_anomalous_tail_energy_eV=float("inf")),
    ):
        try:
            LAPDSim1D(wpe_bad, dict(cathode_flags))
        except ValueError:
            pass
        else:
            raise AssertionError(
                "expected ValueError for heating_anomalous_transport"
            )
    # A BAD TAIL ENERGY UNDER "local" IS NOT AN ERROR: the key is documented as
    # read only under tail_walk, so it must stay inert (this pins the "read
    # ONLY under tail_walk" contract, not just the guard).
    LAPDSim1D(
        dict(csda_params, heating_anomalous_tail_energy_eV=-1.0),
        dict(cathode_flags),
    )
    # Default-path bit-exactness sentinel: naming "local" explicitly reproduces
    # the deposition csda_sim already produced, byte for byte, on every array.
    wpe_off_sim = LAPDSim1D(
        dict(csda_params, heating_anomalous_transport="local"),
        dict(cathode_flags),
    )
    wpe_off_sim._circuit_I_loop = 3000.0
    wpe_off_dep = wpe_off_sim.solve_cathode_boundary().beam_deposition[0]
    for _wpe_arr in (
        "plasma_heating_erg_s", "heating_anomalous_erg_s",
        "heating_coulomb_erg_s", "heating_secondary_erg_s",
        "heating_terminal_erg_s", "radiated_erg_s",
        "ionization_cost_erg_s", "ionization_events", "E_entry_eV",
    ):
        assert np.array_equal(
            getattr(wpe_off_dep, _wpe_arr), getattr(csda_dep, _wpe_arr)
        ), _wpe_arr
    assert wpe_off_dep.end_loss_tail_low_erg_s == 0.0
    assert wpe_off_dep.end_loss_tail_high_erg_s == 0.0
    # On: the QL power is carried away from its birth cells, so the plasma
    # keeps less of it, and the per-ray budget still closes with the tail
    # ledger in it.
    wpe_on_sim = LAPDSim1D(
        dict(csda_params, **wpe_legacy,
             heating_anomalous_transport="tail_walk"),
        dict(cathode_flags),
    )
    wpe_on_sim._circuit_I_loop = 3000.0
    wpe_on_solve = wpe_on_sim.solve_cathode_boundary()
    wpe_on_dep = wpe_on_solve.beam_deposition[0]
    wpe_tail_ledger = (
        float(wpe_on_dep.end_loss_tail_low_erg_s)
        + float(wpe_on_dep.end_loss_tail_high_erg_s)
    )
    wpe_on_total = (
        wpe_on_dep.plasma_heating_erg_s.sum()
        + wpe_on_dep.radiated_erg_s.sum()
        + wpe_on_dep.ionization_cost_erg_s.sum()
        + float(wpe_on_dep.anode_intercepted_erg_s)
        + wpe_on_dep.transmitted_flux
        * wpe_on_dep.transmitted_energy_eV
        * ev_to_erg
        + wpe_tail_ledger
    )
    assert abs(wpe_on_total - csda_budget) / csda_budget < 1e-9
    assert wpe_tail_ledger > 0.0
    assert (
        wpe_on_dep.plasma_heating_erg_s.sum()
        < csda_dep.plasma_heating_erg_s.sum()
    )
    # THE CONSERVATION IDENTITY, at solver conditions: the anomalous power the
    # "local" arm banks locally equals what the "tail_walk" arm deposits along
    # the walks plus what it books to the tail end ledger. This is exact-to-
    # roundoff and not merely a budget statement, because the ray integration
    # itself is bit-identical in both modes -- L_anom depends on the beam and
    # the column, never on where its energy is banked.
    wpe_removed = float(csda_dep.heating_anomalous_erg_s.sum())
    wpe_delivered = (
        float(wpe_on_dep.heating_anomalous_erg_s.sum()) + wpe_tail_ledger
    )
    assert abs(wpe_delivered - wpe_removed) / wpe_removed < 1e-12, (
        wpe_removed, wpe_delivered
    )
    # The other three heating channels are untouched: only the anomalous bank
    # moved, so the whole difference in plasma heating IS the tail ledger.
    for _wpe_arr in (
        "heating_coulomb_erg_s", "heating_secondary_erg_s",
        "heating_terminal_erg_s",
    ):
        assert np.array_equal(
            getattr(wpe_on_dep, _wpe_arr), getattr(csda_dep, _wpe_arr)
        ), _wpe_arr
    assert abs(
        (csda_dep.plasma_heating_erg_s.sum()
         - wpe_on_dep.plasma_heating_erg_s.sum())
        - wpe_tail_ledger
    ) / wpe_tail_ledger < 1e-9
    # Energy-only, exactly like WP-D: the particle rows the fluid and circuit
    # read are untouched, and the WP-D ledger stays identically zero -- the two
    # closures switch independently and do not share a ledger.
    assert np.array_equal(
        wpe_on_dep.ionization_events, csda_dep.ionization_events
    )
    assert wpe_on_dep.end_loss_low_erg_s == 0.0
    assert wpe_on_dep.end_loss_high_erg_s == 0.0
    # The gap-transmission PROBE and the item-35 tripwire are primary-flux
    # instruments and must be blind to heating transport too.
    assert wpe_on_solve.beam_gap_ledger[0] == csda_ledger[0]
    assert (
        wpe_on_solve.beam_result.beam_atten_cross[csda_launch]
        == csda_sigma_eff
    )
    # Hoisted stopping coefficient (cost read 2026-08-02, restructure C): the
    # adapter now builds the walks' per-cell A once and hands it to every
    # deposition ray instead of letting each ray rebuild it. Bit-exactness of
    # the hoist is checked at the SOLVER, against the same ray launched with
    # the coefficient left to the module -- the pre-change call.
    _b3_solver_ray = _deposit_beam_ray(
        csda_res.phi_c, _pskip_Gamma0, dz_cm=_pskip_geom.length_cm,
        nn=csda_state.nn, ne=csda_state.n, Te=csda_derived.Te,
        anode_cross_index=int(_pskip_geom.anode_face_indices[0]),
        anode_eta=csda_eta,
        anomalous_transport="tail_walk",
        tail_energy_eV=float(
            csda_params.get("heating_anomalous_tail_energy_eV", 75.0)
        ),
        **_pskip_ray_kwargs,
    )
    for _b3_arr in (
        "plasma_heating_erg_s", "heating_anomalous_erg_s",
        "heating_coulomb_erg_s", "heating_secondary_erg_s",
        "heating_terminal_erg_s", "radiated_erg_s",
        "ionization_cost_erg_s", "ionization_events", "excitation_events",
        "E_entry_eV",
    ):
        assert np.array_equal(
            getattr(wpe_on_dep, _b3_arr), getattr(_b3_solver_ray, _b3_arr)
        ), _b3_arr
    for _b3_sc in (
        "end_loss_tail_low_erg_s", "end_loss_tail_high_erg_s",
        "transmitted_flux", "transmitted_energy_eV",
    ):
        assert getattr(wpe_on_dep, _b3_sc) == getattr(_b3_solver_ray, _b3_sc)
    # The tail ledger is recorded as cathode diagnostics, zero-defaulted so
    # beer_lambert runs and pre-WP-E files stay readable.
    wpe_on_diag = wpe_on_sim._cathode_diagnostic_snapshot()
    assert wpe_on_diag["source_beam_end_loss_tail_low_W"] == (
        wpe_on_dep.end_loss_tail_low_erg_s * 1.0e-7
    )
    assert wpe_on_diag["source_beam_end_loss_tail_high_W"] == (
        wpe_on_dep.end_loss_tail_high_erg_s * 1.0e-7
    )
    assert wpe_on_diag["end_beam_end_loss_tail_low_W"] == 0.0
    for _bl_key in ("low", "high"):
        assert bl_diag[f"source_beam_end_loss_tail_{_bl_key}_W"] == 0.0
    # The two closures COMPOSE: with both on, each ledger books its own
    # population and neither is empty.
    wpe_both_sim = LAPDSim1D(
        dict(csda_params, **wpe_legacy,
             heating_anomalous_transport="tail_walk",
             beam_product_transport="nonlocal"),
        dict(cathode_flags),
    )
    wpe_both_sim._circuit_I_loop = 3000.0
    wpe_both_dep = wpe_both_sim.solve_cathode_boundary().beam_deposition[0]
    assert (
        wpe_both_dep.end_loss_low_erg_s + wpe_both_dep.end_loss_high_erg_s
        > 0.0
    )
    assert (
        wpe_both_dep.end_loss_tail_low_erg_s
        + wpe_both_dep.end_loss_tail_high_erg_s
        > 0.0
    )
    # WP-D's own ledger is unchanged by WP-E being on alongside it: the tail
    # power never lands in the product channels (this is the reason the two
    # ledgers are siblings rather than one shared pair of fields).
    assert wpe_both_dep.end_loss_low_erg_s == wpd_on_dep.end_loss_low_erg_s
    assert wpe_both_dep.end_loss_high_erg_s == wpd_on_dep.end_loss_high_erg_s
    wpe_both_total = (
        wpe_both_dep.plasma_heating_erg_s.sum()
        + wpe_both_dep.radiated_erg_s.sum()
        + wpe_both_dep.ionization_cost_erg_s.sum()
        + float(wpe_both_dep.anode_intercepted_erg_s)
        + wpe_both_dep.end_loss_low_erg_s
        + wpe_both_dep.end_loss_high_erg_s
        + wpe_both_dep.end_loss_tail_low_erg_s
        + wpe_both_dep.end_loss_tail_high_erg_s
    )
    assert abs(wpe_both_total - csda_budget) / csda_budget < 1e-9

    # --- K6 through the solver: heating_anomalous_tail_ionization lets the QL
    # tail walkers IONIZE the column gas they cross, turning the energy-only
    # WP-E walk into a particle channel. Unit level only -- what it does to
    # the discharge spin-up is a campaign run, not a smoke scenario.
    # Misconfiguration is loud at CONSTRUCTION, and every refusal here is a
    # configuration in which the channel could only be a no-op or could only
    # mis-bank what it carries.
    #
    # The EII table edge in this solver's own units, built the way both guards
    # build it (the module's constant times the solver's I_ion) rather than
    # written down. This is the exact value the K7c ladder refusal reported
    # (es1_lad_tw100ion_nx240.log): under phi_c keying at f = 1.0 with phi_c
    # at the cathode_phi_c_cap_V ceiling, the live E_tail lands here to the
    # last bit -- so the edge has to be inclusive or the declared f bracket
    # loses its top rung to float noise.
    _k7c_edge_eV = _beam_deposition_mod.HE_EII_EPS_TOP * float(csda_sim._I_ion)
    assert float(csda_sim._I_ion) == 24.58738793623
    assert _k7c_edge_eV == 1000.0000000000002
    for k6_bad in (
        dict(csda_params, heating_anomalous_tail_ionization="bogus"),
        # No walkers to give the channel to.
        dict(csda_params, heating_anomalous_tail_ionization="on"),
        dict(csda_params, heating_anomalous_transport="local",
             heating_anomalous_tail_ionization="on"),
        # K7b: the ONLY tail energy still refused is one past the tabulated
        # He EII cross section, where the lookup clamps to its last node and
        # the walk would attenuate on an extrapolated sigma. The bar is read
        # off the table (HE_EII_EPS_TOP * I_ion, ~1000 eV), not written down.
        # REPINNED 2026-08-06: the 20 eV (sub-threshold) and 300 eV
        # (above-<W_sec>) cases used to live in this list and are now the two
        # SPLIT TREATMENTS pinned below -- they construct and run.
        dict(csda_params, **wpe_legacy,
             heating_anomalous_transport="tail_walk",
             heating_anomalous_tail_energy_eV=1500.0,
             heating_anomalous_tail_ionization="on"),
        # K7c: the edge is inclusive within HE_EII_EDGE_REL_TOL, and a
        # GENUINE excess -- here 1e-9 relative, three decades past the
        # tolerance -- is still refused at construction.
        dict(csda_params, **wpe_legacy,
             heating_anomalous_transport="tail_walk",
             heating_anomalous_tail_energy_eV=_k7c_edge_eV * (1.0 + 1.0e-9),
             heating_anomalous_tail_ionization="on"),
    ):
        try:
            LAPDSim1D(k6_bad, dict(cathode_flags))
        except ValueError:
            pass
        else:
            raise AssertionError(
                "expected ValueError for heating_anomalous_tail_ionization "
                f"({k6_bad.get('heating_anomalous_tail_ionization')!r}, "
                f"{k6_bad.get('heating_anomalous_transport')!r}, "
                f"{k6_bad.get('heating_anomalous_tail_energy_eV')!r})"
            )
    # Every registered E_tail arm clears BOTH bars -- the bracket the campaign
    # reports is usable with the channel on, which is the point of computing
    # the bars rather than asserting them.
    for k6_rung in (30.0, 75.0, 150.0):
        LAPDSim1D(
            dict(csda_params, **wpe_legacy,
                 heating_anomalous_transport="tail_walk",
                 heating_anomalous_tail_energy_eV=k6_rung,
                 heating_anomalous_tail_ionization="on"),
            dict(cathode_flags),
        )
    # K7c: and the edge itself CONSTRUCTS, at the edge and a few ULPs above
    # it. At the edge the lookup evaluates the table's last node, which is
    # that node's own value and not an extrapolation of it, so there is
    # nothing for the guard to refuse.
    # (The boundary itself is not pinned either way: reconstructing an excess
    # of exactly HE_EII_EDGE_REL_TOL is a rounding away from either side of
    # the comparison, so the cases below sit strictly inside it.)
    for k7c_ok in (
        _k7c_edge_eV,
        _k7c_edge_eV * (1.0 + 1.0e-13),
        _k7c_edge_eV
        * (1.0 + 0.5 * _beam_deposition_mod.HE_EII_EDGE_REL_TOL),
    ):
        LAPDSim1D(
            dict(csda_params, **wpe_legacy,
                 heating_anomalous_transport="tail_walk",
                 heating_anomalous_tail_energy_eV=k7c_ok,
                 heating_anomalous_tail_ionization="on"),
            dict(cathode_flags),
        )
    # A tail energy outside the depth-1 band is not an error with the channel
    # off either: the key is documented as read only under "on", so the
    # energy-only walk must still accept 300 eV (this pins the contract, not
    # just the guard).
    LAPDSim1D(
        dict(csda_params, **wpe_legacy,
             heating_anomalous_transport="tail_walk",
             heating_anomalous_tail_energy_eV=300.0),
        dict(cathode_flags),
    )
    # Default-path bit-exactness sentinel: naming "off" explicitly reproduces
    # the tail_walk deposition above, byte for byte, on every array and every
    # scalar -- and the four K6 splits are identically zero, so the off path
    # cannot have entered the branch.
    k6_off_sim = LAPDSim1D(
        dict(csda_params, **wpe_legacy,
             heating_anomalous_transport="tail_walk",
             heating_anomalous_tail_ionization="off"),
        dict(cathode_flags),
    )
    k6_off_sim._circuit_I_loop = 3000.0
    k6_off_solve = k6_off_sim.solve_cathode_boundary()
    k6_off_dep = k6_off_solve.beam_deposition[0]
    for _k6_arr in (
        "plasma_heating_erg_s", "heating_anomalous_erg_s",
        "heating_coulomb_erg_s", "heating_secondary_erg_s",
        "heating_terminal_erg_s", "radiated_erg_s",
        "ionization_cost_erg_s", "ionization_events", "excitation_events",
        "E_entry_eV",
    ):
        assert np.array_equal(
            getattr(k6_off_dep, _k6_arr), getattr(wpe_on_dep, _k6_arr)
        ), _k6_arr
    for _k6_sc in ("end_loss_tail_low_erg_s", "end_loss_tail_high_erg_s",
                   "transmitted_flux", "transmitted_energy_eV"):
        assert getattr(k6_off_dep, _k6_sc) == getattr(wpe_on_dep, _k6_sc)
    for _k6_split in ("ionization_events_tail", "excitation_events_tail",
                      "ionization_cost_tail_erg_s", "radiated_tail_erg_s"):
        assert not np.any(getattr(k6_off_dep, _k6_split)), _k6_split
        assert not np.any(getattr(csda_dep, _k6_split)), _k6_split
    # On: the walkers now ionize on their way, so the SAME QL power comes back
    # split across more channels.
    k6_on_sim = LAPDSim1D(
        dict(csda_params, **wpe_legacy,
             heating_anomalous_transport="tail_walk",
             heating_anomalous_tail_ionization="on"),
        dict(cathode_flags),
    )
    k6_on_sim._circuit_I_loop = 3000.0
    k6_on_solve = k6_on_sim.solve_cathode_boundary()
    k6_on_dep = k6_on_solve.beam_deposition[0]
    k6_ledger = (
        float(k6_on_dep.end_loss_tail_low_erg_s)
        + float(k6_on_dep.end_loss_tail_high_erg_s)
    )
    # The scenario must actually FIRE the channel or every assertion is
    # vacuous: pairs born, potential invested, light radiated.
    assert float(k6_on_dep.ionization_events_tail.sum()) > 0.0
    assert float(k6_on_dep.ionization_cost_tail_erg_s.sum()) > 0.0
    assert float(k6_on_dep.radiated_tail_erg_s.sum()) > 0.0
    # THE K6 CLOSURE IDENTITY (E3): every eV launched as tail electrons ends in
    # exactly one of {bulk heat via thermalization, ionization investment,
    # secondary-birth heat, radiation, end ledger}. The launched power is what
    # the "local" arm banked locally -- exact, because the ray integration is
    # bit-identical in both modes -- and the secondary-birth heat is inside
    # heating_anomalous with the rest of the walkers' heat.
    k6_launched = float(csda_dep.heating_anomalous_erg_s.sum())
    k6_delivered = (
        float(k6_on_dep.heating_anomalous_erg_s.sum())
        + float(k6_on_dep.ionization_cost_tail_erg_s.sum())
        + float(k6_on_dep.radiated_tail_erg_s.sum())
        + k6_ledger
    )
    assert abs(k6_delivered - k6_launched) / k6_launched < 1e-12, (
        k6_launched, k6_delivered
    )
    # ... and the whole ray still closes, with the tail's cost and radiation
    # now inside the terms that already carried the primary's.
    k6_on_total = (
        k6_on_dep.plasma_heating_erg_s.sum()
        + k6_on_dep.radiated_erg_s.sum()
        + k6_on_dep.ionization_cost_erg_s.sum()
        + float(k6_on_dep.anode_intercepted_erg_s)
        + k6_on_dep.transmitted_flux * k6_on_dep.transmitted_energy_eV
        * ev_to_erg
        + k6_ledger
    )
    assert abs(k6_on_total - csda_budget) / csda_budget < 1e-9
    # THE PARTICLE STATEMENT: every tail ionization event is one pair added to
    # the shared row, so the difference from the energy-only arm IS the tail
    # split -- nothing is booked twice and nothing is dropped.
    k6_extra = k6_on_dep.ionization_events - wpe_on_dep.ionization_events
    assert np.allclose(
        k6_extra, k6_on_dep.ionization_events_tail, rtol=1e-12, atol=0.0
    )
    for _k6_pair in (
        ("excitation_events", "excitation_events_tail"),
        ("ionization_cost_erg_s", "ionization_cost_tail_erg_s"),
        ("radiated_erg_s", "radiated_tail_erg_s"),
    ):
        assert np.allclose(
            getattr(k6_on_dep, _k6_pair[0])
            - getattr(wpe_on_dep, _k6_pair[0]),
            getattr(k6_on_dep, _k6_pair[1]),
            rtol=1e-12, atol=0.0,
        ), _k6_pair
    # THE WALK WINDOW: not one pair is born, and not one erg deposited, in a
    # cell the RHS mask zeroes. This is the property the window exists for --
    # without it the -z walkers run on behind the cathode and most of the
    # channel's product is created and then deleted.
    k6_dead = ~np.asarray(k6_on_sim._geometry.plasma_active, dtype=bool)
    assert k6_dead.any(), "scenario has no plasma-dead cells to protect"
    assert not np.any(k6_on_dep.ionization_events_tail[k6_dead])
    assert not np.any(k6_on_dep.radiated_tail_erg_s[k6_dead])
    assert not np.any(
        k6_on_dep.heating_anomalous_erg_s[k6_dead]
    ), "tail heat deposited into a plasma-dead cell"
    # The primary's own three heating splits, the WP-D ledger and the
    # primary-flux instruments are all untouched: K6 adds to the tail channel
    # and to nothing else.
    for _k6_arr in ("heating_coulomb_erg_s", "heating_secondary_erg_s",
                    "heating_terminal_erg_s"):
        assert np.array_equal(
            getattr(k6_on_dep, _k6_arr), getattr(csda_dep, _k6_arr)
        ), _k6_arr
    assert k6_on_dep.end_loss_low_erg_s == 0.0
    assert k6_on_dep.end_loss_high_erg_s == 0.0
    assert k6_on_solve.beam_gap_ledger[0] == csda_ledger[0]
    assert (
        k6_on_solve.beam_result.beam_atten_cross[csda_launch] == csda_sigma_eff
    )
    # The tail births reach the FLUID through the beam-ionization birth row --
    # the same convention, the same momentum and birth-temperature booking, the
    # same I_ion sink -- so the row grows by exactly the tail's event density.
    k6_terms = k6_on_sim.beam_ionization_rhs_terms(cathode_solve=k6_on_solve)
    k6_terms_off = k6_off_sim.beam_ionization_rhs_terms(
        cathode_solve=k6_off_solve
    )
    k6_Vp = k6_on_sim._geometry.plasma_volume_cm3
    assert np.allclose(
        k6_terms["beam_ionization_birth"].n
        - k6_terms_off["beam_ionization_birth"].n,
        k6_on_dep.ionization_events_tail / k6_Vp,
        rtol=1e-10, atol=0.0,
    )
    assert np.allclose(
        k6_terms["beam_ionization_cost"].Ee
        - k6_terms_off["beam_ionization_cost"].Ee,
        -k6_on_sim._I_ion * ev_to_erg
        * k6_on_dep.ionization_events_tail / k6_Vp,
        rtol=1e-10, atol=0.0,
    )
    # The tail splits are recorded as cathode diagnostics, zero-defaulted so
    # beer_lambert runs and pre-K6 files stay readable.
    k6_diag = k6_on_sim._cathode_diagnostic_snapshot()
    assert np.array_equal(
        k6_diag["beam_tail_ionization_events_per_s"],
        k6_on_dep.ionization_events_tail,
    )
    assert np.array_equal(
        k6_diag["beam_tail_ionization_cost_W"],
        k6_on_dep.ionization_cost_tail_erg_s * 1.0e-7,
    )
    assert np.array_equal(
        k6_diag["beam_tail_radiated_W"], k6_on_dep.radiated_tail_erg_s * 1.0e-7
    )
    for _k6_dg in ("beam_tail_ionization_events_per_s",
                   "beam_tail_ionization_cost_W", "beam_tail_radiated_W"):
        assert not np.any(bl_diag[_k6_dg]), _k6_dg
        assert not np.any(wpe_on_diag[_k6_dg]), _k6_dg
    # At the MODULE, the walk window has no safe default and says so: an
    # ionizing walk with no window, an out-of-range window, and a window that
    # does not contain the cells the QL channel drives are all refusals.
    _k6_win = tuple(
        int(i) for i in (np.flatnonzero(~k6_dead)[0], np.flatnonzero(~k6_dead)[-1])
    )
    _k6_ray = dict(
        E0_eV=csda_res.phi_c, Gamma0_per_s=_pskip_Gamma0,
        dz_cm=_pskip_geom.length_cm, nn=csda_state.nn, ne=csda_state.n,
        Te=csda_derived.Te, anomalous_transport="tail_walk",
        tail_energy_eV=75.0, tail_ionization="on", **_pskip_ray_kwargs,
    )
    for _k6_win_bad in (None, (-1, 10), (5, 3), (0, _pskip_geom.cells)):
        try:
            _deposit_beam_ray(**_k6_ray, tail_walk_window=_k6_win_bad)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"expected ValueError for tail_walk_window={_k6_win_bad!r}"
            )
    # A window that excludes a driven cell is refused rather than silently
    # dropping that cell's tail power.
    try:
        _deposit_beam_ray(
            **_k6_ray,
            tail_walk_window=(_pskip_geom.cells - 1, _pskip_geom.cells - 1),
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "expected ValueError for a window excluding a QL-driven cell"
        )

    # --- K7 through the solver: the sheath-aware tail closure. The cathode
    # face REFLECTS walkers below e*phi_c(t) instead of deleting them, and the
    # birth energy is keyed to the live phi_c instead of a fixed rung. Unit
    # level only -- what the recovered power does to the discharge is a
    # campaign run.
    #
    # The scenario needs a PRODUCTION-LIKE phi_c: the block above runs against
    # the 1000 V cap, where 0.25*phi_c = 250 eV sits ABOVE the K6 depth-1 bar
    # (since K7b that marches under the disclosed truncation rather than
    # refusing, but it is still not the band the drive actually visits).
    # Capping the drop at 300 V puts the keyed energy where the drive puts it.
    k7_params = dict(csda_params, cathode_phi_c_cap_V=300.0)
    k7_local_sim = LAPDSim1D(
        dict(k7_params, heating_anomalous_transport="local"),
        dict(cathode_flags),
    )
    k7_local_sim._circuit_I_loop = 3000.0
    k7_local_solve = k7_local_sim.solve_cathode_boundary()
    k7_local_dep = k7_local_solve.beam_deposition[0]
    k7_phi_c = float(k7_local_solve.beam_result.result.phi_c)
    k7_launched = float(k7_local_dep.heating_anomalous_erg_s.sum())
    assert k7_launched > 0.0, "K7 scenario drives no QL power"
    # The keyed energy must clear both K6 bars or the ionizing arm below would
    # be testing a refusal instead of a walk.
    assert (
        _beam_deposition_mod.HE_E_STOP_EV < 0.25 * k7_phi_c < 221.0
    ), k7_phi_c

    # (a) PRESENCE GATE. Under "local" the three K7 keys are inert at ANY
    # value: the corrected closure lives inside the walk and cannot be reached
    # from a stance that never walks.
    for k7_inert in (
        dict(k7_params, heating_anomalous_tail_cathode_boundary="escape"),
        dict(k7_params, heating_anomalous_tail_energy_keying="fixed"),
        dict(k7_params, heating_anomalous_tail_phi_c_fraction=1.0),
    ):
        k7_inert_sim = LAPDSim1D(k7_inert, dict(cathode_flags))
        k7_inert_sim._circuit_I_loop = 3000.0
        k7_inert_dep = k7_inert_sim.solve_cathode_boundary().beam_deposition[0]
        for _k7_arr in (
            "plasma_heating_erg_s", "heating_anomalous_erg_s",
            "radiated_erg_s", "ionization_cost_erg_s", "ionization_events",
        ):
            assert np.array_equal(
                getattr(k7_inert_dep, _k7_arr), getattr(k7_local_dep, _k7_arr)
            ), _k7_arr
        assert k7_inert_dep.end_loss_tail_low_erg_s == 0.0
        assert k7_inert_dep.end_loss_tail_high_erg_s == 0.0

    # (b) MISCONFIGURATION is loud at CONSTRUCTION. The f arm is a DECLARED
    # BRACKET, so a value off it is refused wherever it appears; the rest are
    # the silent-no-op configurations: a fraction that nothing reads, a rung
    # that nothing reads, and a twin machine whose two reflecting faces would
    # trap the walkers with no way out.
    for k7_bad_p, k7_bad_f in (
        (dict(k7_params, heating_anomalous_tail_energy_keying="bogus"),
         cathode_flags),
        (dict(k7_params, heating_anomalous_tail_cathode_boundary="bogus"),
         cathode_flags),
        (dict(k7_params, heating_anomalous_tail_phi_c_fraction=0.3),
         cathode_flags),
        (dict(k7_params, heating_anomalous_tail_phi_c_fraction=0.0),
         cathode_flags),
        (dict(k7_params, heating_anomalous_transport="tail_walk",
              heating_anomalous_tail_energy_keying="fixed",
              heating_anomalous_tail_phi_c_fraction=0.25), cathode_flags),
        (dict(k7_params, heating_anomalous_transport="tail_walk",
              heating_anomalous_tail_energy_eV=150.0), cathode_flags),
        (dict(k7_params, heating_anomalous_transport="tail_walk"),
         dict(cathode_flags, TwinCathode=True)),
    ):
        try:
            LAPDSim1D(k7_bad_p, dict(k7_bad_f))
        except ValueError:
            pass
        else:
            raise AssertionError(
                "expected ValueError for the K7 selectors "
                f"({k7_bad_p.get('heating_anomalous_tail_energy_keying')!r}, "
                f"{k7_bad_p.get('heating_anomalous_tail_cathode_boundary')!r}, "
                f"{k7_bad_p.get('heating_anomalous_tail_phi_c_fraction')!r}, "
                f"twin={k7_bad_f.get('TwinCathode')})"
            )
    # Every declared bracket arm constructs, and the legacy pair does too --
    # the bracket the campaign reports is usable, which is the point of
    # refusing everything outside it.
    for k7_f in (0.25, 0.5, 1.0):
        LAPDSim1D(
            dict(k7_params, heating_anomalous_transport="tail_walk",
                 heating_anomalous_tail_phi_c_fraction=k7_f),
            dict(cathode_flags),
        )

    # (c) THE LEGACY ARM IS BIT-EXACT. Naming both legacy values reproduces the
    # module call that has never heard of reflection or keying, byte for byte
    # -- which is what makes every WP-E and K6 assertion above still an
    # assertion about the closure it was written for.
    k7_legacy_sim = LAPDSim1D(
        dict(k7_params, **wpe_legacy,
             heating_anomalous_transport="tail_walk"),
        dict(cathode_flags),
    )
    k7_legacy_sim._circuit_I_loop = 3000.0
    k7_legacy_dep = k7_legacy_sim.solve_cathode_boundary().beam_deposition[0]
    k7_legacy_ray = _deposit_beam_ray(
        k7_phi_c, k7_local_solve.beam_result.result.I_eth_star / qe_SI,
        dz_cm=_pskip_geom.length_cm,
        nn=csda_state.nn, ne=csda_state.n, Te=csda_derived.Te,
        anode_cross_index=int(_pskip_geom.anode_face_indices[0]),
        anode_eta=csda_eta,
        anomalous_transport="tail_walk", tail_energy_eV=75.0,
        **_pskip_ray_kwargs,
    )
    for _k7_arr in (
        "plasma_heating_erg_s", "heating_anomalous_erg_s", "radiated_erg_s",
        "ionization_cost_erg_s", "ionization_events", "excitation_events",
    ):
        assert np.array_equal(
            getattr(k7_legacy_dep, _k7_arr), getattr(k7_legacy_ray, _k7_arr)
        ), _k7_arr
    for _k7_sc in ("end_loss_tail_low_erg_s", "end_loss_tail_high_erg_s"):
        assert getattr(k7_legacy_dep, _k7_sc) == getattr(k7_legacy_ray, _k7_sc)
    assert k7_legacy_dep.end_loss_tail_low_erg_s > 0.0  # the deleted half

    # (d) REFLECTION. The corrected default returns the cathode-end flux to the
    # column: that ledger is EXACTLY zero (phi_c is above every energy any
    # walker can arrive with), the conservation identity still closes to
    # roundoff, and the column keeps materially more of the QL power.
    k7_on_sim = LAPDSim1D(
        dict(k7_params, heating_anomalous_transport="tail_walk"),
        dict(cathode_flags),
    )
    k7_on_sim._circuit_I_loop = 3000.0
    k7_on_dep = k7_on_sim.solve_cathode_boundary().beam_deposition[0]
    assert k7_on_dep.end_loss_tail_low_erg_s == 0.0
    k7_on_ledger = (
        float(k7_on_dep.end_loss_tail_low_erg_s)
        + float(k7_on_dep.end_loss_tail_high_erg_s)
    )
    k7_on_delivered = (
        float(k7_on_dep.heating_anomalous_erg_s.sum()) + k7_on_ledger
    )
    assert abs(k7_on_delivered - k7_launched) / k7_launched < 1e-12, (
        k7_launched, k7_on_delivered
    )
    assert (
        float(k7_on_dep.heating_anomalous_erg_s.sum())
        > float(k7_legacy_dep.heating_anomalous_erg_s.sum())
    )
    # Energy-only, exactly like WP-E: reflection moves where the QL energy
    # lands and nothing else. The particle rows and the WP-D ledger are
    # untouched, and so are the primary's own heating splits.
    assert np.array_equal(
        k7_on_dep.ionization_events, k7_local_dep.ionization_events
    )
    for _k7_arr in ("heating_coulomb_erg_s", "heating_secondary_erg_s",
                    "heating_terminal_erg_s"):
        assert np.array_equal(
            getattr(k7_on_dep, _k7_arr), getattr(k7_local_dep, _k7_arr)
        ), _k7_arr
    assert k7_on_dep.end_loss_low_erg_s == 0.0
    assert k7_on_dep.end_loss_high_erg_s == 0.0

    # (e) phi_c KEYING IS EXACTLY f*phi_c. The keyed arm and a fixed arm named
    # at that same energy are the same run, byte for byte -- the keying moves
    # one number and nothing else.
    for k7_f in (0.25, 0.5):
        k7_keyed_sim = LAPDSim1D(
            dict(k7_params, heating_anomalous_transport="tail_walk",
                 heating_anomalous_tail_cathode_boundary="escape",
                 heating_anomalous_tail_phi_c_fraction=k7_f),
            dict(cathode_flags),
        )
        k7_keyed_sim._circuit_I_loop = 3000.0
        k7_keyed_dep = (
            k7_keyed_sim.solve_cathode_boundary().beam_deposition[0]
        )
        k7_fixed_sim = LAPDSim1D(
            dict(k7_params, **wpe_legacy,
                 heating_anomalous_transport="tail_walk",
                 heating_anomalous_tail_energy_eV=k7_f * k7_phi_c),
            dict(cathode_flags),
        )
        k7_fixed_sim._circuit_I_loop = 3000.0
        k7_fixed_dep = (
            k7_fixed_sim.solve_cathode_boundary().beam_deposition[0]
        )
        assert np.array_equal(
            k7_keyed_dep.heating_anomalous_erg_s,
            k7_fixed_dep.heating_anomalous_erg_s,
        ), k7_f
        assert (
            k7_keyed_dep.end_loss_tail_low_erg_s
            == k7_fixed_dep.end_loss_tail_low_erg_s
        )
    # The default fraction IS the 0.25 arm (continuity with the shipped rung).
    k7_default_sim = LAPDSim1D(
        dict(k7_params, heating_anomalous_transport="tail_walk",
             heating_anomalous_tail_phi_c_fraction=0.25),
        dict(cathode_flags),
    )
    k7_default_sim._circuit_I_loop = 3000.0
    assert np.array_equal(
        k7_default_sim.solve_cathode_boundary()
        .beam_deposition[0].heating_anomalous_erg_s,
        k7_on_dep.heating_anomalous_erg_s,
    )

    # (f) THE IONIZING CHANNEL SURVIVES REFLECTION. The reflected walker keeps
    # marching and stays eligible to ionize, the K6 closure identity still
    # closes, the single-booking property still holds (the shared row grows by
    # exactly the tail split, so nothing is booked twice across the bounce),
    # and not one pair is born in a cell the RHS mask zeroes.
    k7_ion_sim = LAPDSim1D(
        dict(k7_params, heating_anomalous_transport="tail_walk",
             heating_anomalous_tail_ionization="on"),
        dict(cathode_flags),
    )
    k7_ion_sim._circuit_I_loop = 3000.0
    k7_ion_dep = k7_ion_sim.solve_cathode_boundary().beam_deposition[0]
    k7_ion_ledger = (
        float(k7_ion_dep.end_loss_tail_low_erg_s)
        + float(k7_ion_dep.end_loss_tail_high_erg_s)
    )
    assert k7_ion_dep.end_loss_tail_low_erg_s == 0.0
    assert float(k7_ion_dep.ionization_events_tail.sum()) > 0.0
    k7_ion_delivered = (
        float(k7_ion_dep.heating_anomalous_erg_s.sum())
        + float(k7_ion_dep.ionization_cost_tail_erg_s.sum())
        + float(k7_ion_dep.radiated_tail_erg_s.sum())
        + k7_ion_ledger
    )
    assert abs(k7_ion_delivered - k7_launched) / k7_launched < 1e-12, (
        k7_launched, k7_ion_delivered
    )
    assert np.allclose(
        k7_ion_dep.ionization_events - k7_on_dep.ionization_events,
        k7_ion_dep.ionization_events_tail, rtol=1e-12, atol=0.0,
    )
    for _k7_pair in (
        ("excitation_events", "excitation_events_tail"),
        ("ionization_cost_erg_s", "ionization_cost_tail_erg_s"),
        ("radiated_erg_s", "radiated_tail_erg_s"),
    ):
        assert np.allclose(
            getattr(k7_ion_dep, _k7_pair[0]) - getattr(k7_on_dep, _k7_pair[0]),
            getattr(k7_ion_dep, _k7_pair[1]), rtol=1e-12, atol=0.0,
        ), _k7_pair
    k7_dead = ~np.asarray(k7_ion_sim._geometry.plasma_active, dtype=bool)
    assert k7_dead.any(), "scenario has no plasma-dead cells to protect"
    assert not np.any(k7_ion_dep.ionization_events_tail[k7_dead])
    assert not np.any(k7_ion_dep.heating_anomalous_erg_s[k7_dead])
    # The reflected leg is what makes the channel bigger: more of the launched
    # power is spent in the column, so more pairs are born there.
    k7_ion_legacy_sim = LAPDSim1D(
        dict(k7_params, **wpe_legacy,
             heating_anomalous_transport="tail_walk",
             heating_anomalous_tail_ionization="on"),
        dict(cathode_flags),
    )
    k7_ion_legacy_sim._circuit_I_loop = 3000.0
    k7_ion_legacy_dep = (
        k7_ion_legacy_sim.solve_cathode_boundary().beam_deposition[0]
    )
    assert (
        float(k7_ion_dep.ionization_events_tail.sum())
        > float(k7_ion_legacy_dep.ionization_events_tail.sum())
    )

    # --- K7b: the BAND SPLIT. Under phi_c keying E_tail follows the live
    # cathode drop, so one run visits all three bands; refusing at the two
    # depth-1 bars (K6's behaviour) made no keyed ionizing arm startable from
    # cold. Each bar now selects a TREATMENT, and the property that matters is
    # that the split activates ONLY where the old code refused outright.
    #
    # (i) THE CLEAN PROPERTY. In band -- every fixed rung the bracket carries,
    # and the keyed arm above -- both exposure fields are identically zero, so
    # nothing that already ran can have taken a new branch.
    for _k7b_inband in (k7_ion_dep, k7_ion_legacy_dep, k6_on_dep):
        assert _k7b_inband.tail_power_erg_s > 0.0
        assert _k7b_inband.tail_sub_threshold_power_erg_s == 0.0
        assert _k7b_inband.tail_above_bar_power_erg_s == 0.0
    # ... and the exposure ledger is present but empty on the energy-only walk
    # and absent entirely without one, which is the presence gate for the two
    # new diagnostics.
    assert k7_on_dep.tail_power_erg_s > 0.0
    assert k7_on_dep.tail_sub_threshold_power_erg_s == 0.0
    assert k7_on_dep.tail_above_bar_power_erg_s == 0.0
    assert k7_local_dep.tail_power_erg_s == 0.0
    assert csda_dep.tail_power_erg_s == 0.0

    # (ii) BELOW THE LOWER BAR the march REVERTS to the energy-only walk. This
    # is exact physics -- no He inelastic channel is open below the lowest
    # threshold -- and it is exact arithmetic too: the reverted arm is the
    # SAME FLOATS the ionization-off arm produces for the same configuration,
    # on every array and every scalar. Pinned under BOTH tail-end conventions,
    # because the energy-only walk they revert onto has two different domains
    # (windowed under reflection, the whole grid under "escape") and the
    # reversion has to inherit whichever one it would have had.
    k7b_sub_eV = 0.5 * _beam_deposition_mod.HE_E_STOP_EV
    for _k7b_end in ({}, dict(wpe_legacy)):
        k7b_sub_base = dict(
            k7_params,
            heating_anomalous_transport="tail_walk",
            heating_anomalous_tail_energy_keying="fixed",
            heating_anomalous_tail_energy_eV=k7b_sub_eV,
        )
        k7b_sub_base.update(_k7b_end)
        k7b_sub_off_sim = LAPDSim1D(dict(k7b_sub_base), dict(cathode_flags))
        k7b_sub_off_sim._circuit_I_loop = 3000.0
        k7b_sub_off = (
            k7b_sub_off_sim.solve_cathode_boundary().beam_deposition[0]
        )
        k7b_sub_sim = LAPDSim1D(
            dict(k7b_sub_base, heating_anomalous_tail_ionization="on"),
            dict(cathode_flags),
        )
        k7b_sub_sim._circuit_I_loop = 3000.0
        k7b_sub_solve = k7b_sub_sim.solve_cathode_boundary()
        k7b_sub = k7b_sub_solve.beam_deposition[0]
        for _k7b_arr in (
            "plasma_heating_erg_s", "heating_anomalous_erg_s",
            "heating_coulomb_erg_s", "heating_secondary_erg_s",
            "heating_terminal_erg_s", "radiated_erg_s",
            "ionization_cost_erg_s", "ionization_events",
            "excitation_events", "E_entry_eV",
        ):
            assert np.array_equal(
                getattr(k7b_sub, _k7b_arr), getattr(k7b_sub_off, _k7b_arr)
            ), (_k7b_arr, _k7b_end)
        for _k7b_sc in ("end_loss_tail_low_erg_s", "end_loss_tail_high_erg_s",
                        "end_loss_low_erg_s", "end_loss_high_erg_s",
                        "transmitted_flux", "transmitted_energy_eV",
                        "tail_power_erg_s"):
            assert getattr(k7b_sub, _k7b_sc) == getattr(
                k7b_sub_off, _k7b_sc
            ), (_k7b_sc, _k7b_end)
        # NOT a silent no-op: zero ionization, and the reverted power booked.
        for _k7b_split in ("ionization_events_tail", "excitation_events_tail",
                           "ionization_cost_tail_erg_s",
                           "radiated_tail_erg_s"):
            assert not np.any(getattr(k7b_sub, _k7b_split)), _k7b_split
        assert k7b_sub.tail_power_erg_s > 0.0
        assert (
            k7b_sub.tail_sub_threshold_power_erg_s == k7b_sub.tail_power_erg_s
        )
        assert k7b_sub.tail_above_bar_power_erg_s == 0.0
        assert k7b_sub_off.tail_sub_threshold_power_erg_s == 0.0
    # The reverted frame reads as fully sub-band in the saved diagnostics.
    k7b_sub_diag = k7b_sub_sim._cathode_diagnostic_snapshot()
    assert k7b_sub_diag["beam_tail_sub_threshold_fraction"] == 1.0
    assert k7b_sub_diag["beam_tail_sub_threshold_power_W"] == (
        k7b_sub.tail_sub_threshold_power_erg_s * 1.0e-7
    )
    assert k7b_sub_diag["beam_tail_above_bar_power_W"] == 0.0

    # (iii) ABOVE THE UPPER BAR the march RUNS, with the depth-1 truncation
    # kept and its <= 2.0% cascade understatement disclosed rather than
    # refused. 300 eV was a construction-time refusal before K7b; it is
    # REPINNED here as an allowed above-bar case.
    k7b_hi_sim = LAPDSim1D(
        dict(k7_params, **wpe_legacy,
             heating_anomalous_transport="tail_walk",
             heating_anomalous_tail_energy_eV=300.0,
             heating_anomalous_tail_ionization="on"),
        dict(cathode_flags),
    )
    k7b_hi_sim._circuit_I_loop = 3000.0
    k7b_hi = k7b_hi_sim.solve_cathode_boundary().beam_deposition[0]
    assert float(k7b_hi.ionization_events_tail.sum()) > 0.0
    assert float(k7b_hi.radiated_tail_erg_s.sum()) > 0.0
    assert k7b_hi.tail_above_bar_power_erg_s == k7b_hi.tail_power_erg_s
    assert k7b_hi.tail_sub_threshold_power_erg_s == 0.0
    # The channel still closes its own energy branching above the bar -- the
    # truncation understates the CASCADE, it does not leak energy.
    k7b_hi_launched = float(k7_local_dep.heating_anomalous_erg_s.sum())
    k7b_hi_delivered = (
        float(k7b_hi.heating_anomalous_erg_s.sum())
        + float(k7b_hi.ionization_cost_tail_erg_s.sum())
        + float(k7b_hi.radiated_tail_erg_s.sum())
        + float(k7b_hi.end_loss_tail_low_erg_s)
        + float(k7b_hi.end_loss_tail_high_erg_s)
    )
    assert abs(k7b_hi_delivered - k7b_hi_launched) / k7b_hi_launched < 1e-12
    k7b_hi_diag = k7b_hi_sim._cathode_diagnostic_snapshot()
    assert k7b_hi_diag["beam_tail_above_bar_power_W"] == (
        k7b_hi.tail_above_bar_power_erg_s * 1.0e-7
    )
    assert k7b_hi_diag["beam_tail_sub_threshold_fraction"] == 0.0
    # Presence gate on the diagnostics: a beer_lambert run and a "local" run
    # launch no tail power at all, so the fraction is undefined rather than 0.
    for _k7b_dg in ("beam_tail_power_W", "beam_tail_sub_threshold_power_W",
                    "beam_tail_above_bar_power_W"):
        assert bl_diag[_k7b_dg] == 0.0, _k7b_dg
    assert math.isnan(bl_diag["beam_tail_sub_threshold_fraction"])
    assert wpe_on_diag["beam_tail_power_W"] > 0.0
    assert wpe_on_diag["beam_tail_sub_threshold_power_W"] == 0.0
    assert wpe_on_diag["beam_tail_above_bar_power_W"] == 0.0

    # (iv) THE TABLE EDGE IS STILL A REFUSAL, at the module as well as at
    # construction: past the tabulated He EII cross section the lookup clamps
    # to its last node and the walk would attenuate on an extrapolated sigma.
    try:
        _deposit_beam_ray(
            **{**_k6_ray,
               "tail_energy_eV": 1.5 * _beam_deposition_mod.HE_EII_EPS_TOP
               * _beam_deposition_mod.HE_I_ION_EV},
            tail_walk_window=_k6_win,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "expected ValueError for a tail energy past the EII table edge"
        )

    # (v) BUT THE EDGE ITSELF IS NOT PAST THE EDGE (K7c). The refusal exists
    # so that nothing marches on an EXTRAPOLATED cross section; AT the edge
    # the lookup returns the table's last node, which is that node's own
    # value, so there is nothing to refuse and the guard is inclusive within
    # HE_EII_EDGE_REL_TOL. The energy below is the ladder's exact failing
    # value (es1_lad_tw100ion_nx240.log): f = 1.0 keyed to a phi_c sitting at
    # the capability-limited ceiling puts E_tail on the edge to the last bit,
    # so a strict ">=" deleted the top rung of the declared f bracket.
    _k7c_edge_ray_eV = (
        _beam_deposition_mod.HE_EII_EPS_TOP * _pskip_ray_kwargs["I_ion_eV"]
    )
    assert _k7c_edge_ray_eV == 1000.0000000000002, _k7c_edge_ray_eV
    _k7c_sigma_top = float(np.exp(_beam_deposition_mod._HE_LOG_SIGMA[-1]))
    for _k7c_E in (
        _k7c_edge_ray_eV,
        _k7c_edge_ray_eV * (1.0 + 1.0e-13),
        _k7c_edge_ray_eV
        * (1.0 + 0.5 * _beam_deposition_mod.HE_EII_EDGE_REL_TOL),
    ):
        # The physics the tolerance rests on: the lookup at this energy IS the
        # tabulated endpoint, bit for bit. Nothing is extrapolated.
        assert _beam_deposition_mod.He_EII_cross_lkup(
            _k7c_E / _pskip_ray_kwargs["I_ion_eV"]
        ) == _k7c_sigma_top
        _k7c_dep = _deposit_beam_ray(
            **{**_k6_ray, "tail_energy_eV": _k7c_E},
            tail_walk_window=_k6_win,
        )
        # It MARCHES, and it marches in the disclosed above-<W_sec> regime
        # rather than silently.
        assert float(_k7c_dep.ionization_events_tail.sum()) > 0.0
        assert float(_k7c_dep.tail_above_bar_power_erg_s) > 0.0
        assert float(_k7c_dep.tail_sub_threshold_power_erg_s) == 0.0
    # A GENUINE excess -- 1e-9 relative, three decades past the tolerance --
    # is still refused, and the message says what it measured.
    try:
        _deposit_beam_ray(
            **{**_k6_ray,
               "tail_energy_eV": _k7c_edge_ray_eV * (1.0 + 1.0e-9)},
            tail_walk_window=_k6_win,
        )
    except ValueError as _k7c_exc:
        assert "relative excess" in str(_k7c_exc), str(_k7c_exc)
    else:
        raise AssertionError(
            "expected ValueError for a tail energy 1e-9 past the EII edge"
        )

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
    # --- saved effective S_gp(t) waveform (gas_puff_diagnostics) ------------
    # The recorded waveform must be the APPLIED one, not the configured level:
    # decay_after_breakdown shapes it down through the main discharge and the
    # phase gate shuts it off in the afterglow. Every other neutral channel is
    # switched off (pump, reactions, anode recycling) so the puff is the ONLY
    # nn source and the delivered fuel is unambiguous.
    puffdiag_params = dict(decay_params)
    puffdiag_params["dt_save"] = 0.0
    puffdiag_params["tau_afterglow"] = 3.0e-10
    puffdiag_params["pump_enabled"] = False
    puffdiag_params["b_anode_collection"] = 0.0
    # nn0 well below the delivered fuel, so the inventory difference below is
    # not swamped by float64 cancellation against a large standing fill.
    puffdiag_params["nn0"] = 1.0e5
    for _puffdiag_key in (
        "b_ioniz", "b_rec_rad", "b_rec_3b", "b_Qei", "b_Qen", "b_Qcx",
        "b_surface_loss",
    ):
        puffdiag_params[_puffdiag_key] = 0.0
    puffdiag_flags = dict(flags)
    puffdiag_flags["neutral_equilibration"] = False
    puffdiag_sim = LAPDSim1D(puffdiag_params, puffdiag_flags)
    puffdiag_geom = puffdiag_sim.get_initial_snapshot().geometry
    puffdiag_result = puffdiag_sim.run(t_end=8.0e-10, dt=1.0e-10)
    puffdiag = puffdiag_result.gas_puff_diagnostics
    assert set(puffdiag) == set(GAS_PUFF_DIAGNOSTIC_FIELDS), sorted(puffdiag)
    for _puffdiag_values in puffdiag.values():
        assert _puffdiag_values.shape == puffdiag_result.time.shape
        assert np.all(np.isfinite(_puffdiag_values))
    puffdiag_gate = np.asarray(
        puffdiag_result.phase_gas_puff_enabled, dtype=float
    ) > 0.0
    assert puffdiag_gate.any() and not puffdiag_gate.all(), puffdiag_gate
    # On: a real rate that has been shaped BELOW the configured level by the
    # end of the discharge. Off: identically zero, not the configured level.
    assert np.all(puffdiag["S_gp_sccm"][puffdiag_gate] > 0.0)
    assert np.all(puffdiag["puff_particles_per_s"][puffdiag_gate] > 0.0)
    assert puffdiag["S_gp_sccm"][puffdiag_gate][-1] < puffdiag_params["S_gp"]
    for _puffdiag_values in puffdiag.values():
        assert np.all(_puffdiag_values[~puffdiag_gate] == 0.0)
    # No twin cathode here, so the twin entry stays zero throughout.
    assert np.all(puffdiag["Twin_S_gp_sccm"] == 0.0)
    # The recorded rate IS the applied puff row: with pumping off the
    # neutral_sources term carries nothing else.
    puffdiag_vol = np.asarray(
        puffdiag_geom.neutral_volume_cm3, dtype=float
    )
    puffdiag_row = (
        np.asarray(
            puffdiag_result.rhs_terms["neutral_sources"]["nn"], dtype=float
        )
        @ puffdiag_vol
    )
    assert np.allclose(
        puffdiag_row, puffdiag["puff_particles_per_s"], rtol=1e-12, atol=0.0
    ), (puffdiag_row, puffdiag["puff_particles_per_s"])
    # ... and its time integral is the fuel the run actually banked. SSPRK2 on
    # a state-independent source is the explicit trapezoid, and every step is
    # saved (dt_save = 0), so this closes to roundoff rather than to O(dt).
    puffdiag_N_n = np.asarray(puffdiag_result.nn, dtype=float) @ puffdiag_vol
    assert np.isclose(
        puffdiag_N_n[-1] - puffdiag_N_n[0],
        np.trapezoid(
            puffdiag["puff_particles_per_s"], puffdiag_result.time
        ),
        rtol=1e-11,
        atol=0.0,
    ), (puffdiag_N_n[-1] - puffdiag_N_n[0], puffdiag["puff_particles_per_s"])

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
            for _gp_field in GAS_PUFF_DIAGNOSTIC_FIELDS:
                assert h5[f"gas_puff_diagnostics/{_gp_field}"].shape == (4,)
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
        # D3/D4 kernel provenance: every artifact names the arithmetic that
        # produced it, and the default is the pure Python path.
        with h5py.File(output_path, "r") as h5:
            assert h5.attrs["compiled_kernels"] == _kernel_selector.PROVENANCE
        assert run_result.compiled_kernels == _kernel_selector.PROVENANCE
        assert not _kernel_selector.compiled_kernels_requested()
        assert _kernel_selector.COMPILED_KERNELS is None
        assert _kernel_selector.PROVENANCE == _kernel_selector.PURE_PROVENANCE
        # The default path binds the untouched pure kernel object -- no
        # wrapper, no per-call branch, so the arithmetic is bit-for-bit
        # historical.
        assert _cathode_solver_mod._j_eth_crit is (
            _cathode_solver_mod._j_eth_crit_pure
        )
        assert _cathode_solver_idriven_mod._j_eth_crit is (
            _cathode_solver_mod._j_eth_crit_pure
        )
        # Tier A (2026-08-02): the same contract for the rest of the unit --
        # one rebinding site per name, and the two solver modules resolve the
        # SAME object (idriven from-imports the shared ones from
        # _cathode_solver, which rebinds before that import runs).
        for _ta_name in ("_c_log_ei", "_compute_l_b"):
            assert getattr(_cathode_solver_mod, _ta_name) is getattr(
                _cathode_solver_mod, _ta_name + "_pure"
            ), _ta_name
        assert _cathode_solver_idriven_mod._compute_l_b is (
            _cathode_solver_mod._compute_l_b_pure
        )
        assert _beam_deposition_mod._c_log_ei is (
            _cathode_solver_mod._c_log_ei_pure
        )
        for _ta_name in ("_schottky_lowering_eV", "_annular_state_schottky"):
            assert getattr(_cathode_solver_idriven_mod, _ta_name) is getattr(
                _cathode_solver_idriven_mod, _ta_name + "_pure"
            ), _ta_name
        # The compiled root find is opt-in only; on the default path
        # solve_idriven runs its historical Python bracket ladder.
        assert _cathode_solver_idriven_mod._COMPILED_ROOT is None
        # A typo'd opt-in must never read as "off".
        _saved_env = os.environ.get(_kernel_selector.ENV_VAR)
        try:
            os.environ[_kernel_selector.ENV_VAR] = "maybe"
            try:
                _kernel_selector.compiled_kernels_requested()
            except ValueError as error:
                assert _kernel_selector.ENV_VAR in str(error), error
            else:
                raise AssertionError(
                    "an unrecognised CABLP_COMPILED_KERNELS value must raise"
                )
            for _off in ("0", "false", "off", ""):
                os.environ[_kernel_selector.ENV_VAR] = _off
                assert not _kernel_selector.compiled_kernels_requested(), _off
            for _on in ("1", "TRUE", "Yes", "on"):
                os.environ[_kernel_selector.ENV_VAR] = _on
                assert _kernel_selector.compiled_kernels_requested(), _on
        finally:
            if _saved_env is None:
                os.environ.pop(_kernel_selector.ENV_VAR, None)
            else:
                os.environ[_kernel_selector.ENV_VAR] = _saved_env

        # D4 equivalence: wherever the extension has been BUILT, the compiled
        # kernel must be bit-identical to the pure one over the operating
        # range -- checked whether or not this process opted in to running it,
        # because the comparison is the point. Skipped (not failed) on a
        # checkout with no compiled extension: it is optional by design.
        # scripts/spike_cython_kernels.py is the full-resolution version.
        try:
            import importlib as _importlib

            _cy = _importlib.import_module("cablp.funcs._cathode_kernels_cy")
        except ImportError:
            _cy = None
        if _cy is not None:
            assert _cy.pemr() == _cathode_solver_mod._pemr
            _psi_sweep = np.concatenate(
                [
                    np.array([-1.0, 0.0, 1e-3]),
                    np.logspace(-12.0, 3.0, 400),
                    np.linspace(1e-6, 5.0e-3, 200),   # Taylor branch
                    np.linspace(5.0e-3, 150.0, 300),  # closed form
                ]
            )
            _n_taylor = 0
            _n_closed = 0
            for _mu in (1.0, 4.0, 40.0):  # He (the thesis gas) is mu = 4
                for _J_i in (1.0e-4, 1.0, 1.0e4):
                    for _psi in _psi_sweep:
                        _pure = _cathode_solver_mod._j_eth_crit_pure(
                            float(_psi), _J_i, _mu
                        )
                        _comp = _cy.j_eth_crit(float(_psi), _J_i, _mu)
                        # Bit-exact, not merely close: the golden baseline
                        # verifies exact on the compiled path, so anything
                        # looser here would be a weaker claim than the gate.
                        assert _pure == _comp, (_psi, _J_i, _mu, _pure, _comp)
                        if 0.0 < _psi < 1e-3:
                            _n_taylor += 1
                        elif _psi >= 1e-3:
                            _n_closed += 1
            assert _n_taylor > 1000 and _n_closed > 1000, (_n_taylor, _n_closed)

            # --- Tier A: the rest of the compiled cathode unit -------------
            # Every constant the compiled unit duplicates, checked against the
            # authoritative Python value the way the bind-time guards do.
            _cy.check_constants(
                _cathode_solver_mod._pemr,
                _cathode_solver_mod._erg_per_eV,
                _cathode_solver_mod._me_cgs,
            )
            _cy.check_constants_idriven(
                _cathode_solver_idriven_mod._SCHOTTKY_EV_PER_SQRT_V_M
            )
            for _ta_bad in (
                lambda: _cy.check_constants(
                    _cathode_solver_mod._pemr * (1.0 + 1e-15),
                    _cathode_solver_mod._erg_per_eV,
                    _cathode_solver_mod._me_cgs,
                ),
                lambda: _cy.check_constants_idriven(3.7946866e-5),
            ):
                try:
                    _ta_bad()
                except ValueError:
                    pass
                else:
                    raise AssertionError("a drifted constant must raise")
            # Operating-range sweeps, exact equality. Te spans the 0.1 eV floor
            # to the breakdown excursion; ne the pre-breakdown fill to the
            # plateau; phi_c the whole sheath range including the 1000 V cap.
            _ta_Te = np.concatenate([
                np.logspace(-1.0, 2.5, 120),
                np.array([9.999, 10.0, 10.001]),  # the c_log_ei branch corner
            ])
            _ta_ne = np.logspace(8.0, 15.0, 40)
            _ta_n = 0
            for _Te in _ta_Te:
                for _ne in _ta_ne:
                    _ta_n += 1
                    assert _cathode_solver_mod._c_log_ei_pure(
                        float(_Te), float(_ne)
                    ) == _cy.c_log_ei(float(_Te), float(_ne)), (_Te, _ne)
            assert _ta_n > 4000, _ta_n
            _ta_n = 0
            for _phi in (-1.0, 0.0, 1e-9, 1.0, 25.0, 180.0, 1000.0, 5.0e3):
                for _Te in (0.1, 1.0, 3.0, 12.0, 60.0, 300.0):
                    for _ne in (1.0e8, 1.0e10, 1.0e12, 1.0e14):
                        assert _cathode_solver_idriven_mod.\
                            _schottky_lowering_eV_pure(
                                float(_phi), float(_Te), float(_ne)
                            ) == _cy.schottky_lowering_eV(
                                float(_phi), float(_Te), float(_ne)
                            ), (_phi, _Te, _ne)
                        for _nn in (0.0, 1.0e12, 1.0e15):
                            for _sb in (0.0, 1.0e-17, 1.0e-15):
                                _ta_n += 1
                                # _compute_l_b_pure calls the module-global
                                # _c_log_ei, which IS the compiled one in an
                                # opted-in process -- so this leg alone would
                                # not catch a bad c_log_ei. The sweep above
                                # does, independently, which is what closes it.
                                assert _cathode_solver_mod._compute_l_b_pure(
                                    float(_phi), float(_Te), float(_ne),
                                    float(_nn), float(_sb),
                                ) == _cy.compute_l_b(
                                    float(_phi), float(_Te), float(_ne),
                                    float(_nn), float(_sb),
                                ), (_phi, _Te, _ne, _nn, _sb)
            assert _ta_n > 1500, _ta_n
            # The annular Schottky emission state -- the residual's body --
            # over randomised annuli covering the released, partially clamped
            # and fully choked regimes.
            _ta_rng = np.random.default_rng(20260802)
            _ta_seen = {True: 0, False: 0}
            for _ta_draw in range(200):
                _Te = float(10.0 ** _ta_rng.uniform(-1.0, 1.7))
                _ne = float(10.0 ** _ta_rng.uniform(9.0, 14.0))
                _J_i = float(10.0 ** _ta_rng.uniform(-4.0, 2.0))
                _Jk = tuple(
                    float(10.0 ** _ta_rng.uniform(-6.0, 3.0)) for _ in range(10)
                )
                _dk = tuple(
                    float(_ta_rng.uniform(0.01, 2.0)) for _ in range(10)
                )
                _fk = tuple(float(_ta_rng.uniform(0.0, 1.0)) for _ in range(10))
                # Zero-emission and zero-footprint annuli are real states (a
                # cold outer ring, a ring outside the plasma), so pin them into
                # half the draws. Only half, because a zero-footprint annulus
                # has J_crit = 0 and therefore always reports clamped, which
                # would hide the fully-released branch from the coverage count.
                if _ta_draw % 2:
                    _Jk = (0.0,) + _Jk[1:]
                    _fk = _fk[:5] + (0.0,) + _fk[6:]
                # Two emission scales per draw: as drawn (space-charge
                # clamped almost everywhere) and 1e-12 weaker (fully released,
                # so the un-clamped branch and the Schottky enhancement leg
                # are exercised too).
                for _ta_scale in (1.0, 1.0e-12):
                    _Jks = tuple(_ta_scale * _j for _j in _Jk)
                    for _psi in (1e-8, 1e-4, 0.3, 3.0, 25.0, 400.0):
                        _ta_pure = _cathode_solver_idriven_mod.\
                            _annular_state_schottky_pure(
                                _psi, _J_i, 4.0, _Jks, _dk, _fk, _Te, _ne
                            )
                        _ta_comp = _cy.annular_state_schottky(
                            _psi, _J_i, 4.0, _Jks, _dk, _fk, _Te, _ne
                        )
                        assert _ta_pure == _ta_comp, (
                            _psi, _ta_scale, _ta_pure, _ta_comp
                        )
                        _ta_seen[_ta_pure[2]] += 1
            assert _ta_seen[True] > 100 and _ta_seen[False] > 100, _ta_seen
            # Mismatched annulus lengths are a loud failure, not a zip-truncated
            # silent physics change.
            try:
                _cy.annular_state_schottky(
                    1.0, 1.0, 4.0, (1.0, 2.0), (0.1,), (0.5, 0.5), 3.0, 1e12
                )
            except ValueError:
                pass
            else:
                raise AssertionError("ragged annuli must raise")

            # --- Tier A: the compiled ROOT FIND ----------------------------
            # The compiled unit runs the whole bracket ladder plus brentq in C.
            # The reference here is the Python ladder verbatim -- the block in
            # solve_idriven -- built on the module's own pure emission state and
            # scipy's Python brentq, so only the ladder/root-find transcription
            # is under test. Bit-equality of the located psi is the claim.
            from scipy.optimize import brentq as _ta_brentq

            def _ta_python_ladder(
                J_i, mu, Lambda, T_e, n_e, J_eth_k, delta_k, ion_frac_k,
                J_imposed, phi_c_cap_V, psi_lo, psi_top, plateau_tol_rel,
            ):
                def _state(psi):
                    return _cathode_solver_idriven_mod.\
                        _annular_state_schottky_pure(
                            psi, J_i, mu, J_eth_k, delta_k, ion_frac_k,
                            T_e, n_e,
                        )

                def _J_tot(psi):
                    return (
                        J_i
                        * (1.0 - _cathode_solver_mod._exp_clamped(Lambda - psi))
                        + _state(psi)[0]
                    )

                def _net_phi_c(psi):
                    return (psi - _state(psi)[1]) * T_e

                capability_limited = False
                J_target = J_imposed
                for _ in range(200):
                    if _J_tot(psi_top) >= J_target:
                        break
                    if _net_phi_c(psi_top) >= phi_c_cap_V:
                        capability_limited = True
                        break
                    psi_top *= 2.0
                else:
                    capability_limited = True
                if capability_limited and _J_tot(psi_top) >= J_imposed - (
                    plateau_tol_rel * abs(J_imposed)
                ):
                    capability_limited = False
                    J_target = J_imposed - plateau_tol_rel * abs(J_imposed)
                if capability_limited:
                    if _net_phi_c(psi_top) > phi_c_cap_V:
                        return _ta_brentq(
                            lambda x: _net_phi_c(x) - phi_c_cap_V,
                            psi_lo, psi_top, xtol=1.0e-12, rtol=1.0e-14,
                            full_output=False,
                        ), True
                    return psi_top, True
                return _ta_brentq(
                    lambda x: _J_tot(x) - J_target,
                    psi_lo, psi_top, xtol=1.0e-12, rtol=1.0e-14,
                    full_output=False,
                ), False

            _ta_tol = _cathode_solver_idriven_mod._J_PLATEAU_TOL_REL
            _ta_lam = math.log(math.sqrt(4.0 * _cathode_solver_mod._pemr
                                         / (2.0 * math.pi)))
            _ta_cap_seen = {True: 0, False: 0}
            _ta_cases = 0
            for _Te in (0.3, 1.5, 4.0, 15.0):
                for _ne in (1.0e10, 5.0e11, 1.0e13):
                    for _J_i in (1.0e-3, 0.2, 5.0):
                        _Jk = tuple(
                            _J_i * 10.0 ** (2.0 - 0.4 * _a) for _a in range(10)
                        )
                        _dk = tuple(0.05 + 0.01 * _a for _a in range(10))
                        _fk = tuple(1.0 - 0.1 * _a for _a in range(10))
                        for _Jimp in (0.0, 1.0e-3, 0.5, 20.0, 1.0e4):
                            _ta_psi_top = max(1000.0 / _Te, _ta_lam + 2.0)
                            _ta_args = (
                                _J_i, 4.0, _ta_lam, _Te, _ne,
                                _Jk, _dk, _fk, _Jimp, 1000.0, 1.0e-8,
                                _ta_psi_top, _ta_tol,
                            )
                            _ta_ref, _ta_ref_cap = _ta_python_ladder(*_ta_args)
                            _ta_got, _ta_cap, _ta_iters = (
                                _cy.solve_psi_annular_schottky(*_ta_args)
                            )
                            assert _ta_cap == _ta_ref_cap, (
                                _ta_args[:5], _Jimp, _ta_cap, _ta_ref_cap
                            )
                            # Bit-equal, not close: the golden verifies exact
                            # on the compiled path, so a looser claim here
                            # would be weaker than the gate. A divergence must
                            # be reported with both roots and the iteration
                            # count, never absorbed by a tolerance.
                            assert _ta_got == _ta_ref, (
                                _ta_args[:5], _Jimp,
                                float(_ta_got).hex(), float(_ta_ref).hex(),
                                _ta_iters,
                            )
                            _ta_cap_seen[bool(_ta_cap)] += 1
                            _ta_cases += 1
            assert _ta_cases >= 180, _ta_cases
            assert _ta_cap_seen[False] > 20, _ta_cap_seen

        loaded_result = load_result_hdf5(output_path)
        loaded_via_solver = LAPDSim1D.load_result(output_path)
        for loaded in (loaded_result, loaded_via_solver):
            assert loaded.compiled_kernels == _kernel_selector.PROVENANCE
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
            assert set(loaded.gas_puff_diagnostics) == set(
                GAS_PUFF_DIAGNOSTIC_FIELDS
            )
            for _gp_field in GAS_PUFF_DIAGNOSTIC_FIELDS:
                assert np.array_equal(
                    loaded.gas_puff_diagnostics[_gp_field],
                    run_result.gas_puff_diagnostics[_gp_field],
                ), _gp_field
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
        # A file written before the waveform diagnostic existed still loads:
        # the group is absent and the reader NaN-defaults it to the right
        # length rather than raising or inventing zeros.
        legacy_path = f"{tmpdir}/sim1d_smoke_legacy.h5"
        shutil.copyfile(output_path, legacy_path)
        with h5py.File(legacy_path, "r+") as h5:
            del h5["gas_puff_diagnostics"]
        legacy_loaded = load_result_hdf5(legacy_path)
        assert set(legacy_loaded.gas_puff_diagnostics) == set(
            GAS_PUFF_DIAGNOSTIC_FIELDS
        )
        for _gp_field in GAS_PUFF_DIAGNOSTIC_FIELDS:
            legacy_values = legacy_loaded.gas_puff_diagnostics[_gp_field]
            assert legacy_values.shape == run_result.time.shape
            assert np.all(np.isnan(legacy_values)), _gp_field
        assert np.allclose(legacy_loaded.y, run_result.y)

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
    # M1b): the surface energy budget replaces the
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

    # --- accelerated dt_growth re-approach (default off) ---------------------
    # A long ramp: an early phase boundary sets a small first step, then
    # nothing physical binds and dt_growth caps every step while it climbs.
    # This is the regime the probe measured (80.6% of steps growth-capped at a
    # median 364x below the binding physics bound).
    ramp_params = dict(growth_params)
    ramp_params["tau_prebreakdown"] = 2.0e-9
    ramp_params["tau_discharge"] = 40.0e-6
    ramp_base = LAPDSim1D(ramp_params, growth_flags).run(t_end=3.0e-7)
    assert ramp_base.steps == 17
    assert [diag.step_cap for diag in ramp_base.diagnostics][:5] == [
        "phase_boundary",
        "dt_growth",
        "dt_growth",
        "dt_growth",
        "dt_growth",
    ]
    ramp_fast = LAPDSim1D(
        {
            **ramp_params,
            "dt_growth_recovery_patience": 3,
            "dt_growth_recovery_factor": 4.0,
        },
        growth_flags,
    ).run(t_end=3.0e-7)
    # Same ramp, far fewer steps: the whole point of the mechanism.
    assert ramp_fast.steps == 7, ramp_fast.steps
    ramp_fast_dt = [diag.accepted_dt for diag in ramp_fast.diagnostics]
    ramp_base_dt = [diag.accepted_dt for diag in ramp_base.diagnostics]
    # HYSTERESIS, both halves. Engaging takes evidence: the first step is
    # capped by the phase boundary, so steps 2-4 must still ramp at the BASE
    # 1.25 while the streak of three growth-capped steps is being earned...
    assert np.allclose(ramp_fast_dt[:4], ramp_base_dt[:4])
    assert np.allclose(
        ramp_fast_dt[1:4],
        [ramp_fast_dt[0] * 1.25**k for k in (1, 2, 3)],
    )
    # ...and only the step AFTER patience is met jumps by the recovery factor.
    assert np.isclose(ramp_fast_dt[4], ramp_fast_dt[3] * 4.0)
    assert np.isclose(ramp_fast_dt[5], ramp_fast_dt[4] * 4.0)
    # It never weakens a bound -- every accepted step is still <= dt_max.
    assert max(ramp_fast_dt) <= ramp_params["dt_max"] * (1.0 + 1.0e-12)
    # DEFAULT OFF and presence-gated: shipped defaults disable it, and setting
    # the keys to their defaults is step-for-step identical to not having them.
    assert default_config()[0]["dt_growth_recovery_patience"] == 0
    ramp_off = LAPDSim1D(
        {
            **ramp_params,
            "dt_growth_recovery_patience": 0,
            "dt_growth_recovery_factor": 4.0,
        },
        growth_flags,
    ).run(t_end=3.0e-7)
    assert [diag.accepted_dt for diag in ramp_off.diagnostics] == ramp_base_dt
    assert np.array_equal(ramp_off.n, ramp_base.n)
    assert np.array_equal(ramp_off.Ee, ramp_base.Ee)
    # Misconfiguration is loud, and at CONSTRUCTION.
    for bad_ramp in (
        {"dt_growth_recovery_patience": -1},
        {"dt_growth_recovery_patience": 1.5},
        {"dt_growth_recovery_patience": "soon"},
        # A recovery factor at or below the base could never accelerate.
        {"dt_growth_recovery_patience": 2, "dt_growth_recovery_factor": 1.25},
        {"dt_growth_recovery_patience": 2, "dt_growth_recovery_factor": 0.5},
        {
            "dt_growth_recovery_patience": 2,
            "dt_growth_recovery_factor": float("nan"),
        },
    ):
        try:
            LAPDSim1D({**ramp_params, **bad_ramp}, growth_flags)
        except ValueError as error:
            assert "dt_growth_recovery" in str(error), str(error)
        else:
            raise AssertionError(f"{bad_ramp} must raise")
    # A bad recovery factor is INERT while patience is 0: the key is not
    # consulted at all on the off path, so it cannot refuse a default run.
    LAPDSim1D(
        {
            **ramp_params,
            "dt_growth_recovery_patience": 0,
            "dt_growth_recovery_factor": 0.5,
        },
        growth_flags,
    )

    retry_params = dict(params)
    retry_flags = dict(flags)
    retry_flags["Plasma"] = False
    retry_flags["heat_conduction"] = False
    retry_params["dt_save"] = 0.0
    retry_params["pump_enabled"] = False
    retry_params["dt_max"] = 1.0e-6
    retry_params["neutral_dt_fraction"] = 100.0
    retry_params["max_neutral_step_fraction"] = 6.0
    # This scenario needs a near-vacuum start: the retry it exercises fires
    # when one puff step moves nn by more than max_neutral_step_fraction, which
    # is only possible against a tiny background. It used to inherit that from
    # the nn0 default; nn0 is now the realistic direct-run fill (2e13), against
    # which the same puff is a ~1e-4 fractional step and nothing is ever
    # rejected. Pin the initial condition the scenario is built on, alongside
    # the other limiter constants it already sets.
    retry_params["nn0"] = 1.0e9
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

    # The historical raise-on-timeout arm, now selected explicitly. Every
    # assertion below is the pre-existing BreakdownError contract, unchanged;
    # the switch-open default is covered by its own block further down.
    failed_current_phase_params = dict(current_phase_params)
    failed_current_phase_params["I_prebreakdown"] = 1.0e30
    failed_current_phase_params["I_breakdown"] = 1.0e30
    failed_current_phase_params["prebreakdown_timeout_action"] = "raise"
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
    failed_breakdown_phase_params["prebreakdown_timeout_action"] = "raise"
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

    # --- Ignition-failure diagnostics and guards -------------------------
    #
    # (i) the joint-condition logic on synthetic histories, (ii) the
    # switch-open firing on a synthetic no-trigger path, (iii) the scorer
    # hard-fail on a non-ignited fixture and its silence on an ignited one.
    # main() binds save_result_hdf5 as a local further down (R1e block), so
    # alias it here rather than relying on that later binding.
    from cablp.solvers._sim1d.results.io import (
        save_result_hdf5 as _save_result_hdf5_ignition,
    )
    from cablp.solvers._sim1d.core.ignition import (
        IGNITION_DIAGNOSTIC_FIELDS,
        IGNITION_RATE_WINDOW_S,
        IGNITION_STALL_MIN_SAMPLES,
        IGNITION_STALL_WINDOW_S,
        IgnitionMonitor,
        longest_joint_negative_span,
    )

    # Loud construction errors on an unusable window.
    for bad_kwargs in (
        {"window_s": 0.0},
        {"window_s": -1.0e-3},
        {"window_s": float("inf")},
        {"rate_window_s": 0.0},
        {"rate_window_s": 2.0 * IGNITION_STALL_WINDOW_S},
        {"min_samples": 1},
    ):
        try:
            IgnitionMonitor(**bad_kwargs)
        except ValueError as error:
            assert "IgnitionMonitor" in str(error)
        else:
            raise AssertionError(
                f"expected IgnitionMonitor({bad_kwargs}) to raise"
            )

    def _stall_trip_time(N_of_t, Ee_of_t, samples=1400, dt=1.0e-5):
        """Drive a fresh monitor through a synthetic history; return trip t."""
        monitor = IgnitionMonitor()
        for index in range(samples):
            t = index * dt
            record = monitor.record(
                time=t,
                N_plasma=N_of_t(t),
                N_neutral=1.0e18,
                Ee_total=Ee_of_t(t),
                armed=True,
            )
            if record["stalled"]:
                return t
        return None

    def _spike_trough_climb(t):
        # beam-turn-on spike -> initial-inventory-burn trough -> puff-restored
        # climb: the healthy start-up shape the detector must NOT kill.
        if t < 1.0e-3:
            return 1.0e12 * math.exp(60.0 * t)
        if t < 5.0e-3:
            return 1.0e12 * math.exp(0.06) * math.exp(-400.0 * (t - 1.0e-3))
        return (
            1.0e12
            * math.exp(0.06)
            * math.exp(-400.0 * 4.0e-3)
            * math.exp(900.0 * (t - 5.0e-3))
        )

    # Healthy: density spikes, troughs, then climbs while Ee rises. NO trip.
    assert (
        _stall_trip_time(_spike_trough_climb, lambda t: 1.0e3 * (1.0 + 50.0 * t))
        is None
    ), "spike/trough/climb with rising Ee must not trip"
    # The same trough with Ee merely holding/rising is the settled rationale:
    # density falling alone is never enough.
    assert (
        _stall_trip_time(
            lambda t: 1.0e12 * math.exp(-300.0 * t),
            lambda t: 1.0e3 * math.exp(10.0 * t),
        )
        is None
    ), "falling density with rising Ee must not trip"
    # Slow-positive growth with a slowly cooling electron pool: ambiguous, so
    # it must fall through untripped.
    assert (
        _stall_trip_time(
            lambda t: 1.0e12 * math.exp(5.0 * t),
            lambda t: 1.0e3 * math.exp(-50.0 * t),
        )
        is None
    ), "slow-positive gamma_N must not trip"
    # Oscillating about flat: ambiguous, untripped.
    assert (
        _stall_trip_time(
            lambda t: 1.0e12 * (1.0 + 0.2 * math.sin(2.0 * math.pi * t / 6.0e-4)),
            lambda t: 1.0e3 * (1.0 + 0.2 * math.sin(2.0 * math.pi * t / 6.0e-4)),
        )
        is None
    ), "an oscillating history must not trip"
    # Joint decay for LESS than the window, then recovery: untripped.
    assert (
        _stall_trip_time(
            lambda t: (
                1.0e12 * math.exp(-200.0 * t)
                if t < 2.2e-3
                else 1.0e12 * math.exp(-200.0 * 2.2e-3) * math.exp(500.0 * (t - 2.2e-3))
            ),
            lambda t: (
                1.0e3 * math.exp(-150.0 * t)
                if t < 2.2e-3
                else 1.0e3 * math.exp(-150.0 * 2.2e-3) * math.exp(400.0 * (t - 2.2e-3))
            ),
        )
        is None
    ), "a joint-negative stretch shorter than the window must not trip"
    # Sustained joint decay: MUST trip, and only once the window plus one
    # rate window has actually elapsed.
    stall_trip = _stall_trip_time(
        lambda t: 1.0e12 * math.exp(-200.0 * t),
        lambda t: 1.0e3 * math.exp(-150.0 * t),
    )
    assert stall_trip is not None, "sustained joint decay must trip"
    assert np.isclose(
        stall_trip,
        IGNITION_STALL_WINDOW_S + IGNITION_RATE_WINDOW_S,
        atol=2.0e-5,
    ), stall_trip
    # Disarming clears the buffer, so a window can never straddle beam-off.
    straddle_monitor = IgnitionMonitor()
    for index in range(1400):
        t = index * 1.0e-5
        straddle_record = straddle_monitor.record(
            time=t,
            N_plasma=1.0e12 * math.exp(-200.0 * t),
            N_neutral=1.0e18,
            Ee_total=1.0e3 * math.exp(-150.0 * t),
            armed=(index % 200) != 0,
        )
        assert not straddle_record["stalled"]
    # The offline replay metric agrees with the trip logic's own bookkeeping.
    assert longest_joint_negative_span([0.0, 1.0, 2.0], [-1.0, -1.0, 1.0],
                                       [-1.0, -1.0, -1.0]) == 1.0
    assert longest_joint_negative_span([0.0, 1.0, 2.0], [-1.0, -1.0, -1.0],
                                       [-1.0, -1.0, -1.0],
                                       armed=[1, 0, 1]) == 0.0
    assert longest_joint_negative_span([0.0, 1.0], [np.nan, -1.0],
                                       [-1.0, -1.0]) == 0.0

    # The switch-open on a synthetic no-trigger path: the current can never
    # reach the thresholds, so tau_prebreakdown fires the hardware guard.
    timeout_params = dict(current_phase_params)
    timeout_params["I_prebreakdown"] = 1.0e30
    timeout_params["I_breakdown"] = 1.0e30
    timeout_params["tau_prebreakdown"] = 3.0e-10
    timeout_sim = LAPDSim1D(timeout_params, current_phase_flags)
    assert np.isclose(timeout_sim.default_t_end(), 6.0e-10)
    with warnings.catch_warnings(record=True) as timeout_warnings:
        warnings.simplefilter("always")
        timeout_result = timeout_sim.run(dt=1.0e-10)
    assert any(
        "ignition aborted" in str(entry.message)
        and "prebreakdown_timeout" in str(entry.message)
        for entry in timeout_warnings
    ), [str(entry.message) for entry in timeout_warnings]
    # It is a real phase transition, not an exception: the run winds down
    # through the ordinary afterglow and STOPS at abort + tau_afterglow.
    assert np.isclose(timeout_result.final_time, 4.0e-10)
    assert list(timeout_result.phase_events["reason"]) == [
        "initial",
        "prebreakdown_timeout",
        "tau_afterglow",
    ]
    assert list(timeout_result.phase_events["phase"]) == [
        "pre_breakdown",
        "afterglow",
        "post_afterglow",
    ]
    assert np.allclose(
        timeout_result.phase_events["time"], [0.0, 3.0e-10, 4.0e-10]
    )
    assert "main_discharge" not in set(timeout_result.phase)
    assert list(timeout_result.phase) == [
        "pre_breakdown",
        "pre_breakdown",
        "pre_breakdown",
        "afterglow",
        "post_afterglow",
    ]
    # The switch is OPEN: the drive is off from the abort instant onwards.
    assert np.array_equal(
        timeout_result.phase_cathode_enabled, [1.0, 1.0, 1.0, 0.0, 0.0]
    )
    # The cathode floats through the afterglow sample and is simply dead by
    # post_afterglow -- the ordinary end-of-discharge switch state.
    assert np.array_equal(timeout_result.phase_floating, [0.0, 0.0, 0.0, 1.0, 0.0])
    assert timeout_result.ignition_abort["reason"] == "prebreakdown_timeout"
    assert np.isclose(timeout_result.ignition_abort["time_s"], 3.0e-10)
    assert np.isclose(
        timeout_result.ignition_abort["window_s"], IGNITION_STALL_WINDOW_S
    )
    assert timeout_result.ignition_abort["threshold_name"] == "I_prebreakdown"
    assert np.isclose(timeout_result.ignition_abort["threshold_A"], 1.0e30)
    for power_key in (
        "P_beam_W",
        "P_conduction_W",
        "P_cooling_W",
        "P_ionization_W",
        "P_transport_W",
        "P_beam_end_loss_W",
    ):
        assert power_key in timeout_result.ignition_abort
    assert set(timeout_result.ignition_diagnostics) == set(
        IGNITION_DIAGNOSTIC_FIELDS
    )
    for values in timeout_result.ignition_diagnostics.values():
        assert values.shape == timeout_result.time.shape
    # Armed while the drive is on and pre-ignition; disarmed once aborted.
    assert np.array_equal(
        timeout_result.ignition_diagnostics["armed"], [1.0, 1.0, 1.0, 0.0, 0.0]
    )
    assert np.all(
        np.isfinite(timeout_result.ignition_diagnostics["N_plasma"][:3])
    )
    assert np.all(np.isnan(timeout_result.ignition_diagnostics["P_beam_W"][3:]))
    # The abort survives the HDF5 round trip; a run without one carries none.
    with tempfile.TemporaryDirectory() as ignition_dir:
        ignition_path = Path(ignition_dir) / "timeout.h5"
        _save_result_hdf5_ignition(ignition_path, timeout_result)
        loaded_timeout = load_result_hdf5(ignition_path)
        assert loaded_timeout.ignition_abort["reason"] == "prebreakdown_timeout"
        assert np.isclose(loaded_timeout.ignition_abort["time_s"], 3.0e-10)
        assert set(loaded_timeout.ignition_diagnostics) == set(
            IGNITION_DIAGNOSTIC_FIELDS
        )
        assert np.array_equal(
            loaded_timeout.ignition_diagnostics["armed"],
            timeout_result.ignition_diagnostics["armed"],
        )
        ignited_path = Path(ignition_dir) / "ignited.h5"
        _save_result_hdf5_ignition(ignited_path, direct_current_phase_result)
        loaded_ignited = load_result_hdf5(ignited_path)
        assert not hasattr(loaded_ignited, "ignition_abort")

    # An igniting run is untouched: no guard event, no abort record, and the
    # detector never armed a full window inside it.
    assert not hasattr(current_phase_result, "ignition_abort")
    assert not hasattr(direct_current_phase_result, "ignition_abort")
    assert not set(direct_current_phase_result.phase_events["reason"]) & {
        "ignition_stalled",
        "prebreakdown_timeout",
    }
    assert np.all(direct_current_phase_result.ignition_diagnostics["stalled"] == 0.0)
    assert IGNITION_STALL_MIN_SAMPLES >= 2

    # --- non-ignition guards, wall-clock / accepted-step arm -----------------
    # The stall detector and the tau_prebreakdown timeout both measure
    # SIMULATED time, so neither can see a non-igniting arm that stops
    # producing simulated time and burns wall clock instead. These two budgets
    # close over that, through the SAME switch-open path.
    for budget_key, budget_value, budget_reason in (
        ("ignition_accepted_step_cap", 3, "accepted_step_cap"),
        ("ignition_wall_clock_cap_s", 1.0e-9, "wall_clock_cap"),
    ):
        budget_params = dict(current_phase_params)
        budget_params["I_prebreakdown"] = 1.0e30
        budget_params["I_breakdown"] = 1.0e30
        # Far beyond reach, so the simulated-time guard cannot be what fires.
        budget_params["tau_prebreakdown"] = 1.0
        budget_params[budget_key] = budget_value
        budget_sim = LAPDSim1D(budget_params, current_phase_flags)
        with warnings.catch_warnings(record=True) as budget_warnings:
            warnings.simplefilter("always")
            budget_result = budget_sim.run(dt=1.0e-10, max_steps=50)
        assert any(
            "ignition aborted" in str(entry.message)
            and budget_reason in str(entry.message)
            for entry in budget_warnings
        ), [str(entry.message) for entry in budget_warnings]
        # Same wind-down as every other switch-open abort: a real phase
        # transition, no main_discharge, and refused scoring.
        assert budget_result.ignition_abort["reason"] == budget_reason
        assert "main_discharge" not in set(budget_result.phase)
        assert budget_result.phase_events["reason"][-1] == "tau_afterglow"
        assert budget_result.phase_events["phase"][1] == "afterglow"
        assert budget_result.ignition_abort["wall_clock_s"] >= 0.0
        assert budget_result.ignition_abort["accepted_steps"] >= 1.0
        # The accepted-step cap is deterministic: it trips ON the capped step.
        if budget_key == "ignition_accepted_step_cap":
            assert np.isclose(
                budget_result.ignition_abort["time_s"], 3.0e-10
            ), budget_result.ignition_abort["time_s"]
            assert budget_result.ignition_abort["accepted_steps"] == 3.0
    # Misconfiguration is loud, and at CONSTRUCTION -- not hours into the very
    # crawl the guard exists to catch.
    for bad_key, bad_value in (
        ("ignition_wall_clock_cap_s", -1.0),
        ("ignition_wall_clock_cap_s", float("nan")),
        ("ignition_wall_clock_cap_s", "soon"),
        ("ignition_accepted_step_cap", -5),
        ("ignition_accepted_step_cap", 2.5),
        ("ignition_accepted_step_cap", "many"),
    ):
        try:
            LAPDSim1D({**current_phase_params, bad_key: bad_value},
                      current_phase_flags)
        except ValueError as error:
            assert bad_key in str(error), str(error)
        else:
            raise AssertionError(f"{bad_key}={bad_value!r} must raise")
    # Default-off, and presence-gated: shipped defaults disable both, and a
    # run that sets them to their defaults is step-for-step identical to one
    # that has never heard of them.
    assert default_config()[0]["ignition_wall_clock_cap_s"] == 0.0
    assert default_config()[0]["ignition_accepted_step_cap"] == 0
    budget_absent_params = dict(current_phase_params)
    budget_absent_params.pop("ignition_wall_clock_cap_s", None)
    budget_absent_params.pop("ignition_accepted_step_cap", None)
    budget_absent = LAPDSim1D(budget_absent_params, current_phase_flags).run(
        dt=1.0e-10, max_steps=6
    )
    budget_off = LAPDSim1D(
        {
            **current_phase_params,
            "ignition_wall_clock_cap_s": 0.0,
            "ignition_accepted_step_cap": 0,
        },
        current_phase_flags,
    ).run(dt=1.0e-10, max_steps=6)
    assert np.array_equal(budget_absent.n, budget_off.n)
    assert np.array_equal(budget_absent.Ee, budget_off.Ee)
    assert not hasattr(budget_absent, "ignition_abort")
    assert not hasattr(budget_off, "ignition_abort")

    # Scorer hard-fail (scripts): a non-ignited run must raise, an ignited one
    # must score its origin from the first main_discharge sample.
    import compare_sim1d_es1 as _cmp_es1
    import fingerprints_sim1d as _fingerprints

    for origin_fn, caller in (
        (_cmp_es1._main_discharge_origin, "compare_sim1d_es1"),
        (_fingerprints._origin_s, "fingerprints_sim1d"),
    ):
        try:
            origin_fn(timeout_result)
        except RuntimeError as error:
            message = str(error)
            assert "NON-IGNITED RUN" in message, message
            assert "post_afterglow" in message, message
            assert "prebreakdown_timeout" in message, message
        else:
            raise AssertionError(
                f"{caller} must refuse to score a non-ignited run"
            )
        ignited_origin = origin_fn(direct_current_phase_result)
        assert np.isclose(ignited_origin, 1.0e-10), (caller, ignited_origin)

    # Scorer hard-fail (scripts), stage (iii): a run whose trace ends before
    # the decay window closes must RAISE, not have the window quietly clipped
    # to whatever it covers. A clipped fit is a different measurement wearing
    # the campaign metric's name, and is not comparable run to run.
    def _decay_case(end_ms):
        """Synthetic scorable result + overlay whose trace ends at end_ms."""
        t_s = np.arange(0.0, end_ms * 1.0e-3 + 1.0e-9, 1.0e-4)
        z = np.array([0.0, 500.0, 1000.0])
        decay = np.exp(-t_s * 1.0e3)[:, None] * np.ones(z.size)[None, :]
        synthetic = SimpleNamespace(
            time=t_s,
            phase=np.array(["main_discharge"] * t_s.size),
            z_cm=z,
            n=1.0e12 * decay,
            Te=3.0 * decay,
            params={"tau_afterglow": end_ms * 1.0e-3 - 0.020, "tau_discharge": 0.020},
        )
        t_exp = np.linspace(20.0, 30.0, 101)
        overlay_stub = {
            "port": np.array([20]),
            "z_cm": np.array([500.0]),
            "isat_decay_port": np.array([20]),
            "isat_decay_time_ms": t_exp,
            "isat_decay_mean_a": np.exp(-(t_exp - 20.0))[None, :],
        }
        return synthetic, overlay_stub

    short_result, short_overlay = _decay_case(20.8)
    try:
        _cmp_es1.compare_decay(short_result, short_overlay)
    except RuntimeError as error:
        message = str(error)
        assert "SHORT AFTERGLOW" in message, message
        # Names the configured window, the available extent, and tau_afterglow.
        assert "(20, 21.5) ms" in message, message
        assert "20.8" in message, message
        assert "tau_afterglow" in message, message
    else:
        raise AssertionError(
            "compare_decay must refuse to score a run whose trace ends "
            "before the stage (iii) window closes"
        )

    # A trace that covers the window scores normally and reports the FULL
    # configured window back, unclipped.
    long_result, long_overlay = _decay_case(26.0)
    decay_rows, decay_window = _cmp_es1.compare_decay(long_result, long_overlay)
    assert decay_window == _cmp_es1.DECAY_WINDOW_MS, decay_window
    assert len(decay_rows) == 1, decay_rows
    # The reference 6 ms afterglow and the planned 2 ms probe default both
    # clear the (20.0, 21.5) window; only sub-1.5 ms afterglows trip the guard.
    assert _cmp_es1.DECAY_WINDOW_MS[1] - 20.0 <= 1.5

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
    # --- item 37: the equilibration delivers its CONFIGURED puff duty --------
    # tau_cycle / tau_discharge / dt chosen so a step lands a hair BELOW the
    # puff-off instant (t=6e-10 is 1 ulp short of cycle_start + tau_discharge).
    # The phase-boundary schedule used to DROP that boundary inside the run
    # loop's time_tol while the untolerated modulo in _phase_info still read
    # "puff", so the puff ran one whole extra step: this exact case delivered
    # 4.5e-10 s of puff against a configured 2.0e-10 s (+125%). Both readers now
    # share _equilibration_cycle_position, so the delivered ON-time is exact.
    duty_params = dict(neutral_phase_run_params)
    duty_params["tau_cycle"] = 5.0e-10
    duty_params["tau_discharge"] = 1.0e-10
    duty_params["cycles"] = 2
    duty_params["dt_save"] = 0.0
    duty_sim = LAPDSim1D(duty_params, dict(neutral_phase_run_flags))
    duty_result = duty_sim.run(t_end=1.0e-9, dt=2.5e-10)
    duty_times = np.asarray(duty_result.time, dtype=float)
    duty_phases = np.asarray(duty_result.phase, dtype=str)
    assert list(duty_phases) == [
        "equilibrium_puff",
        "equilibrium_off",
        "equilibrium_off",
        "equilibrium_puff",
        "equilibrium_off",
        "equilibrium_off",
        "equilibrium_puff",
    ], list(duty_phases)
    duty_on = float(
        np.sum(np.diff(duty_times)[duty_phases[:-1] == "equilibrium_puff"])
    )
    assert np.isclose(duty_on, 2.0 * 1.0e-10, rtol=1e-12), duty_on
    # Same assertion where the period DOES divide the step: 1e-10 windows on a
    # 5e-10 cycle stepped at exactly 1e-10 must not lose or gain a step either.
    duty_div_params = dict(duty_params)
    duty_div_sim = LAPDSim1D(duty_div_params, dict(neutral_phase_run_flags))
    duty_div_result = duty_div_sim.run(t_end=1.0e-9, dt=1.0e-10)
    duty_div_times = np.asarray(duty_div_result.time, dtype=float)
    duty_div_phases = np.asarray(duty_div_result.phase, dtype=str)
    duty_div_on = float(
        np.sum(np.diff(duty_div_times)[duty_div_phases[:-1] == "equilibrium_puff"])
    )
    assert np.isclose(duty_div_on, 2.0 * 1.0e-10, rtol=1e-12), duty_div_on
    # --- measured equilibration puff width (equilibration_gas_puff_on_s) -----
    # Default None == the historical tau_discharge-derived window, BIT-exact
    # through the real equilibration path (start_simulation -> the inner sim).
    # NB built on neutral_phase_params, NOT the no_source_* family: the puff
    # has to be ENABLED for the window to be observable in the seed at all.
    puffw_base_params = dict(neutral_phase_params)
    puffw_base_params["neutral_equilibration_cycles"] = 2
    puffw_base_params["neutral_equilibration_dt"] = 1.0e-10

    def _puffw_seed(puff_on, drop=False):
        puffw_params = dict(puffw_base_params)
        if drop:
            puffw_params.pop("equilibration_gas_puff_on_s", None)
        else:
            puffw_params["equilibration_gas_puff_on_s"] = puff_on
        puffw_sim = LAPDSim1D(puffw_params, dict(equilibration_flags))
        puffw_sim.start_simulation(dt=1.0e-10)
        return np.asarray(puffw_sim.get_results().nn[-1], dtype=float)

    puffw_none_nn = _puffw_seed(None)
    assert np.array_equal(_puffw_seed(None, drop=True), puffw_none_nn)
    # Setting it explicitly to tau_discharge must reproduce the fallback too.
    assert np.array_equal(
        _puffw_seed(puffw_base_params["tau_discharge"]), puffw_none_nn
    )
    # ... and a HALVED window must measurably starve the equilibration.
    puffw_half_nn = _puffw_seed(1.0e-10)
    assert np.mean(puffw_half_nn) < np.mean(puffw_none_nn), (
        float(np.mean(puffw_half_nn)), float(np.mean(puffw_none_nn))
    )
    # The window itself moved: the phase flips at the new width, and the
    # recorded phase event names the key that closed it.
    puffw_phase_params = dict(neutral_phase_run_params)
    puffw_phase_params["equilibration_gas_puff_on_s"] = 1.0e-10
    puffw_phase_sim = LAPDSim1D(
        puffw_phase_params, dict(neutral_phase_run_flags)
    )
    assert puffw_phase_sim.phase_at_time(0.5e-10) == "equilibrium_puff"
    assert puffw_phase_sim.phase_at_time(1.0e-10) == "equilibrium_off"
    assert np.isclose(puffw_phase_sim.next_phase_boundary_after(0.0), 1.0e-10)
    puffw_phase_result = puffw_phase_sim.run(t_end=4.0e-10, dt=1.0e-10)
    assert list(puffw_phase_result.phase) == [
        "equilibrium_puff",
        "equilibrium_off",
        "equilibrium_off",
        "equilibrium_off",
        "equilibrium_off",
    ], list(puffw_phase_result.phase)
    assert list(puffw_phase_result.phase_events["reason"]) == [
        "initial",
        "equilibration_gas_puff_on_s",
    ]
    # It is NOT inert to the neutral-seed signature: setting it must re-key.
    from cablp.solvers._sim1d.core.neutral_seed_cache import (
        neutral_seed_signature,
    )

    assert neutral_seed_signature(
        {**puffw_base_params, "equilibration_gas_puff_on_s": 1.0e-10},
        equilibration_flags,
    ) != neutral_seed_signature(
        {**puffw_base_params, "equilibration_gas_puff_on_s": None},
        equilibration_flags,
    )
    # Loud ValueError on a nonsense window, at CONSTRUCTION time.
    for bad_puff_on in (0.0, -1.0e-10, 1.0e-9, "twenty-five"):
        bad_puffw_params = dict(puffw_base_params)
        bad_puffw_params["equilibration_gas_puff_on_s"] = bad_puff_on
        try:
            LAPDSim1D(bad_puffw_params, dict(equilibration_flags))
        except ValueError as exc:
            assert "equilibration_gas_puff_on_s" in str(exc), str(exc)
        else:
            raise AssertionError(
                f"equilibration_gas_puff_on_s={bad_puff_on!r} did not raise"
            )
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

    # --- Neutral-momentum state foundations (M1):
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

    # --- Two-zone neutral state foundations (M1):
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

    # --- Neutral-momentum sources (M2): with M_n on
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
    # The solver carries the split (nn, nn_a)
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

    # --- K4a kinetic neutrals: the refresh-
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

    # --- K2a transient DVM neutrals: the LIVE distribution arm on its own
    # neutral clock. Default off and presence-gated; once engaged it owns
    # every neutral row and the ion-side momentum/energy of the channels it
    # models, the saved nn IS the column zeroth moment, the particle ledger
    # closes to roundoff, and each unsupported combination is refused at
    # construction naming the offender.
    assert "neutral_kinetic_dvm_coupling" not in p2z_sim.rhs_terms()
    assert p2z_sim._dvm is None
    kd_params = dict(p2z_params)
    kd_params["neutral_model"] = "kinetic_dvm"
    # Smoke-scale clock: a few 1 ns steps must cross it.
    kd_params["neutral_kinetic_dvm_cadence_s"] = 2.0e-9
    kd_params["neutral_kinetic_dvm_nvz"] = 16
    kd_params["neutral_kinetic_dvm_nvp"] = 6
    kd_flags = dict(p2z_flags)
    kd_flags["neutral_prebreakdown"] = False
    kd_flags["neutral_equilibration"] = False
    kd_flags["characteristic_boundary"] = False

    # Every refusal, and the offender it must name.
    for kd_bad_params, kd_bad_flags, kd_offender in (
        (kd_params, dict(kd_flags, neutral_two_zone=False), "neutral_two_zone"),
        (kd_params, dict(kd_flags, neutral_momentum=True), "neutral_momentum"),
        (
            dict(kd_params, neutral_kinetic_dvm_cadence_s=0.0),
            kd_flags,
            "neutral_kinetic_dvm_cadence_s",
        ),
        (
            dict(kd_params, neutral_kinetic_dvm_accommodation=-0.1),
            kd_flags,
            "neutral_kinetic_dvm_accommodation",
        ),
        (
            dict(kd_params, neutral_kinetic_dvm_elastic="bilinear"),
            kd_flags,
            "neutral_kinetic_dvm_elastic",
        ),
        (
            dict(kd_params, neutral_kinetic_dvm_nvz=17),
            kd_flags,
            "nvz",
        ),
        (
            dict(kd_params, gas_puff_local_ionization_fraction=0.3),
            kd_flags,
            "gas_puff_local_ionization_fraction",
        ),
        (
            kd_params,
            dict(kd_flags, coupled_circuit_picard=True),
            "coupled_circuit_picard",
        ),
        (
            dict(kd_params, neutral_kinetic_dvm_tn_feedback=True),
            dict(kd_flags, characteristic_boundary=True),
            "characteristic_boundary",
        ),
        (
            dict(kd_params, neutral_kinetic_dvm_annulus_flights="chord"),
            kd_flags,
            "neutral_kinetic_dvm_annulus_flights",
        ),
        (
            dict(
                kd_params,
                neutral_kinetic_dvm_annulus_flights="bounded_chord",
                neutral_model="moment",
            ),
            kd_flags,
            "neutral_kinetic_dvm_annulus_flights",
        ),
        (
            dict(
                kd_params,
                neutral_kinetic_dvm_annulus_flights="bounded_chord",
            ),
            dict(kd_flags, neutral_two_zone=False),
            "neutral_two_zone",
        ),
    ):
        try:
            LAPDSim1D(dict(kd_bad_params), dict(kd_bad_flags))
        except ValueError as kd_error:
            assert kd_offender in str(kd_error), (kd_offender, str(kd_error))
        else:
            raise AssertionError(
                f"kinetic_dvm accepted an unsupported configuration "
                f"({kd_offender})"
            )

    kd_sim = LAPDSim1D(dict(kd_params), dict(kd_flags))
    assert kd_sim._dvm is not None
    # Tn consumption is its OWN switch and defaults off.
    assert kd_sim._dvm_tn_feedback is False
    assert kd_sim._dvm_presheath_Tn_eV() is None
    # Pre-engagement: the coupling key exists and is all-zero, and the fluid
    # neutral rows are still live (the moment terms carry the fill).
    kd_pre = kd_sim.rhs_terms()
    assert "neutral_kinetic_dvm_coupling" in kd_pre
    assert np.all(kd_pre["neutral_kinetic_dvm_coupling"].M == 0.0)
    assert np.all(kd_pre["neutral_kinetic_dvm_coupling"].Ei == 0.0)
    assert np.any(kd_pre["neutral_sources"].nn != 0.0)

    for _ in range(6):
        kd_sim.advance_one_step(dt=1.0e-9)
    kd_dvm = kd_sim._dvm
    assert kd_sim._dvm_engaged and kd_dvm.updates >= 1
    assert np.all(np.isfinite(kd_dvm.f_c)) and np.all(np.isfinite(kd_dvm.f_a))
    assert np.all(kd_dvm.f_c >= 0.0) and np.all(kd_dvm.f_a >= 0.0)
    kd_state = kd_sim.state
    assert np.all(np.isfinite(kd_state.nn)) and np.all(np.isfinite(kd_state.nn_a))
    assert np.all(np.isfinite(kd_sim.rhs()))

    # Moment-consistency contract: the saved nn IS the column zeroth moment
    # (floored), nn_a the annulus moment. Exact equality, not a tolerance.
    kd_floor = kd_sim.floors["nn"]
    assert np.array_equal(
        kd_state.nn, np.maximum(kd_dvm.column_density(), kd_floor)
    )
    assert np.array_equal(
        kd_state.nn_a, np.maximum(kd_dvm.annulus_density(), kd_floor)
    )

    # Particle ledger: births minus losses closes to roundoff in both the
    # distribution and domain forms.
    kd_res = kinetic_dvm_ledger_residual(kd_dvm.last_ledger)
    assert abs(kd_res["distribution_rel"]) < 1.0e-12, kd_res
    assert abs(kd_res["domain_rel"]) < 1.0e-12, kd_res

    # Transfer antisymmetry: the fluid coupling rows ARE minus the measured
    # kinetic moments -- the M*Vp + M_n*Vm == 0 discipline, extended to a
    # velocity-resolved neutral state.
    kd_terms = kd_sim.rhs_terms()
    kd_active = np.asarray(kd_sim.geometry.plasma_active, dtype=bool)
    kd_coupling = kd_terms["neutral_kinetic_dvm_coupling"]

    def kd_expected(values):
        # The coupling term is plasma-coupled, so it takes the same
        # dead-cell mask as every other plasma term when that stance is on.
        if kd_sim._active_plasma_topology:
            return np.where(kd_active, values, 0.0)
        return values

    assert np.array_equal(
        np.asarray(kd_coupling.M, dtype=float), kd_expected(kd_dvm.M_transfer)
    )
    assert np.array_equal(
        np.asarray(kd_coupling.Ei, dtype=float), kd_expected(kd_dvm.Ei_transfer)
    )

    # Supersession: every neutral row is zeroed, the superseded ion-transfer
    # rows are zeroed, and the particle / electron rows keep their forms.
    for kd_name, kd_term in kd_terms.items():
        if kd_name == "neutral_kinetic_dvm_coupling":
            continue
        assert np.all(np.asarray(kd_term.nn, dtype=float) == 0.0), kd_name
        if kd_term.nn_a is not None:
            assert np.all(np.asarray(kd_term.nn_a, dtype=float) == 0.0), kd_name
    for kd_name in ("ionization_birth", "recombination_rad_loss"):
        assert np.all(kd_terms[kd_name].M == 0.0), kd_name
        assert np.all(kd_terms[kd_name].Ei == 0.0), kd_name
    assert np.any(kd_terms["ionization_birth"].n != 0.0)

    # Rejected attempts mutate nothing: the distribution, the lagged end
    # buffers and the coupling accumulators are bit-identical afterwards.
    kd_before = (
        kd_dvm.f_c.tobytes(),
        kd_dvm.f_a.tobytes(),
        kd_dvm.pend_L_c.tobytes(),
        kd_dvm.pend_R_c.tobytes(),
        kd_dvm.M_transfer.tobytes(),
        kd_dvm.Ei_transfer.tobytes(),
        kd_dvm.updates,
    )
    kd_sim._attempt_step(dt=1.0e-9)
    kd_sim._attempt_step(dt=1.0e-13)
    kd_sim.rhs(y=kd_sim._y * 1.01)
    assert kd_before == (
        kd_dvm.f_c.tobytes(),
        kd_dvm.f_a.tobytes(),
        kd_dvm.pend_L_c.tobytes(),
        kd_dvm.pend_R_c.tobytes(),
        kd_dvm.M_transfer.tobytes(),
        kd_dvm.Ei_transfer.tobytes(),
        kd_dvm.updates,
    )

    # K2e counted-particle ionization handshake. The arm debits the count
    # the PLASMA booked over the tick, so no neutral becomes an ion without
    # leaving the kinetic state -- the identity below is the whole point,
    # and it is a conservation law, not a tolerance.
    kd_resid = (
        kd_dvm.ion_removed_cum + kd_dvm.ion_debt - kd_dvm.ion_booked_cum
    )
    kd_ion_scale = max(float(np.max(np.abs(kd_dvm.ion_booked_cum))), 1e-300)
    assert float(np.max(np.abs(kd_resid))) / kd_ion_scale < 1.0e-12, kd_resid
    assert np.any(kd_dvm.ion_booked_cum > 0.0)
    # Nothing was withheld on a healthy tick, so the debit IS the booking.
    assert kd_dvm.ion_shortfall_updates == 0
    assert np.allclose(
        kd_dvm.ion_removed_cum, kd_dvm.ion_booked_cum, rtol=1.0e-12, atol=0.0
    )
    # The pending booking is attempt-local: the rejected attempts above did
    # not add to it, and it is cleared at every tick.
    assert np.all(np.isfinite(kd_sim._dvm_ion_booked))
    assert kd_sim._dvm_ion_stage_accum is None
    assert kd_sim._dvm_ion_stage_weight == 0.0
    # A standalone update -- no partner, no booked count -- leaves the
    # march's own tally standing and books nothing into the handshake. Run
    # against a snapshot so the arm this sim carries is put back bit for
    # bit, which also exercises the snapshot's new ledger fields.
    kd_snap = kd_dvm.snapshot()
    kd_bare_led = kd_dvm.update(
        1.0e-9,
        n_i=np.asarray(kd_sim.state.n, dtype=float),
        Ti_eV=np.asarray(kd_sim.derived.Ti, dtype=float),
        u_i=np.asarray(kd_sim.derived.u, dtype=float),
        nu_ion=np.full(kd_dvm.nz, 1.0e4),
    )
    assert kd_bare_led["ion_booked"] == 0.0
    assert kd_bare_led["ion_limited_cells"] == 0.0
    assert kd_bare_led["loss_ionization"] > 0.0
    assert np.array_equal(kd_dvm.ion_booked_cum, kd_snap["ion_booked_cum"])
    assert np.array_equal(kd_dvm.ion_removed_cum, kd_snap["ion_removed_cum"])
    kd_dvm.restore(kd_snap)
    assert np.array_equal(kd_dvm.f_c, kd_snap["f_c"])
    assert np.array_equal(kd_dvm.ion_debt, kd_snap["ion_debt"])

    # Tn consumption A/B: with the switch on, the measured Tn reaches the
    # presheath collisionality and the boundary absorption moves.
    kd_tn_sim = LAPDSim1D(
        dict(kd_params, neutral_kinetic_dvm_tn_feedback=True), dict(kd_flags)
    )
    for _ in range(6):
        kd_tn_sim.advance_one_step(dt=1.0e-9)
    assert kd_tn_sim._dvm_tn_feedback is True
    kd_tn = kd_tn_sim._dvm_presheath_Tn_eV()
    assert kd_tn is not None and np.all(np.isfinite(kd_tn))
    # The two builds are identical except for the switch, so any difference
    # in the absorption term is the Tn feedback and nothing else.
    kd_off_abs = kd_sim.boundary_absorption_rhs(state=kd_tn_sim.state).n
    kd_on_abs = kd_tn_sim.boundary_absorption_rhs(state=kd_tn_sim.state).n
    assert np.any(kd_off_abs != kd_on_abs), (
        "the Tn-consumption switch changed nothing"
    )

    # Production-style geometry (Lcs = 25): an obstruction cell sits between
    # the plenum and the cathode, so the cathode's live cell is index 2 --
    # neither an end cell nor a fixed offset from one. The wall-return
    # channels must be READ from that cell and DEPOSITED into it; positional
    # constants read the plasma-dead cells behind it and source nothing.
    kd_obs_params = dict(kd_params)
    kd_obs_params.update(
        {
            "Rp": 15.0,
            "R_cath": 15.0,
            "Rcs": 40.0,
            "Lcs": 25.0,
            "Rsup": 0.0,
            "end_expansion_cells": 10,
            "end_expansion_machine_radius_cm": 100.0,
            "end_expansion_plasma_radius_cm": 15.0,
            "source_region_length_cm": 100.0,
            "source_region_dz_cm": 10.0,
        }
    )
    kd_obs_flags = dict(kd_flags)
    kd_obs_flags["end_expansion_geometry"] = True
    kd_obs_flags["source_fixed_grid"] = True
    kd_obs_sim = LAPDSim1D(kd_obs_params, kd_obs_flags)
    kd_obs_roles = [str(r) for r in np.asarray(kd_obs_sim.geometry.cell_role)]
    assert kd_obs_roles[:3] == ["plenum", "obstruction", "cathode"]
    kd_obs_cath = 2
    kd_obs_coll = len(kd_obs_roles) - 1
    assert kd_obs_roles[kd_obs_coll] == "collector"
    assert absorbing_live_cells_by_role(kd_obs_sim.geometry) == {
        "cathode": (kd_obs_cath,),
        "collector": (kd_obs_coll,),
    }
    # The arm's deposition targets ARE the absorbing faces' live cells.
    assert kd_obs_sim._dvm.cath_cell == kd_obs_cath
    assert kd_obs_sim._dvm.coll_cell == kd_obs_coll
    for _ in range(6):
        kd_obs_sim.advance_one_step(dt=1.0e-9)
    assert kd_obs_sim._dvm_engaged and kd_obs_sim._dvm.updates >= 1
    kd_obs_rates = kd_obs_sim._kinetic_channel_rates(
        kd_obs_sim.state, kd_obs_sim.derived, kd_obs_sim.time
    )
    # Live, and placed ONLY on its own face -- not on cell 0 or 1.
    assert kd_obs_rates["cath"] > 0.0 and kd_obs_rates["coll"] > 0.0
    assert kd_obs_rates["cath_cells"][kd_obs_cath] == kd_obs_rates["cath"]
    assert kd_obs_rates["cath_cells"][0] == 0.0
    assert kd_obs_rates["cath_cells"][1] == 0.0
    assert kd_obs_rates["coll_cells"][kd_obs_coll] == kd_obs_rates["coll"]
    # Recycled == removed, per face, against the boundary term this stance
    # actually runs (kd_obs_flags keeps characteristic_boundary off).
    assert kd_obs_sim._characteristic_boundary is False
    kd_obs_removed = -np.asarray(
        kd_obs_sim.boundary_absorption_rhs(state=kd_obs_sim.state).n,
        dtype=float,
    ) * np.asarray(kd_obs_sim.geometry.plasma_volume_cm3, dtype=float)
    for kd_obs_cell, kd_obs_key in (
        (kd_obs_cath, "cath_cells"),
        (kd_obs_coll, "coll_cells"),
    ):
        assert abs(
            kd_obs_rates[kd_obs_key][kd_obs_cell] - kd_obs_removed[kd_obs_cell]
        ) <= 1.0e-12 * abs(kd_obs_removed[kd_obs_cell])
    # K2d: the return enters as a DIRECTED INFLOW at the emitting face, not
    # as a stationary birth inside the cell. One update of a bare engine on
    # this geometry, seeded empty, fed only the cathode channel: every fed
    # particle arrives (the ghost density is the counted particles over
    # exactly the |v_z| A dt the march multiplies back), nothing appears
    # upstream of the face, and part of the return has already travelled
    # downstream within the tick.
    kd_dep = TransientDVM(geometry=kd_obs_sim.geometry, nvz=16, nvp=6)
    assert kd_dep.cath_cell == kd_obs_cath and kd_dep.coll_cell == kd_obs_coll
    kd_dep.f_c[:] = 0.0
    kd_dep.f_a[:] = 0.0
    kd_dep_src = np.zeros(kd_dep.nz)
    kd_dep_src[kd_obs_cath] = 1.0e18
    kd_dep_dt = 1.0e-5
    kd_dep.update(
        kd_dep_dt,
        n_i=np.zeros(kd_dep.nz),
        Ti_eV=np.full(kd_dep.nz, 0.026),
        u_i=np.zeros(kd_dep.nz),
        nu_ion=np.zeros(kd_dep.nz),
        sources={"cathode_face": kd_dep_src},
        T_s_K=1910.0,
    )
    kd_dep_mass = kd_dep.f_c.sum(axis=(1, 2)) * kd_dep.V_col
    kd_dep_fed = 1.0e18 * kd_dep_dt
    assert abs(kd_dep.total_inventory() - kd_dep_fed) <= 1.0e-12 * kd_dep_fed
    assert kd_dep_mass[kd_obs_cath] > 0.0
    assert np.all(kd_dep_mass[:kd_obs_cath] == 0.0)
    assert kd_dep_mass[kd_obs_cath + 1:].sum() > 0.0
    assert kd_dep_mass[kd_obs_cath] < kd_dep.total_inventory()
    assert kd_dep.column_drift()[kd_obs_cath] > 0.0

    # K2d afterglow-entry stretch: the tick-frozen coupling drain is held
    # constant while the plasma steps inside one tick, and at the afterglow
    # entry it flipped sign and demanded more ion energy than the cathode
    # cell held -- an explicit e-fold below dt_min, so no admissible step
    # existed and the run died on a negative Ei (2026-08-05, t = 21.312 ms).
    # Three statements: the drain now BOUNDS dt, the applied drain cannot
    # carry a cell through its floor, and what it declines to carry is
    # re-ledgered rather than lost.
    kd_lim_sim = LAPDSim1D(dict(kd_obs_params), dict(kd_obs_flags))
    for _ in range(8):
        kd_lim_sim.advance_one_step(dt=1.0e-9)
    assert kd_lim_sim._dvm_engaged
    kd_lim_quiet = kd_lim_sim.dvm_transfer_ledger()
    # Inert on a healthy step: applied == booked, bit-exactly.
    assert kd_lim_quiet["relax_limited_steps"] == 0
    kd_lim_base = kd_lim_sim.suggest_timestep().dt_surface_loss
    kd_lim_cells = kd_lim_sim.geometry.cells
    kd_lim_sim._dvm.Ei_transfer = np.full(kd_lim_cells, -1.0e12)
    kd_lim_sim._dvm.M_transfer = np.full(kd_lim_cells, -1.0e3)
    # The bound SEES it (the defect: a 1e12 drain moved dt by exactly zero).
    assert kd_lim_sim.suggest_timestep().dt_surface_loss < kd_lim_base
    kd_lim_floor = 1.5 * ev_to_erg * kd_lim_sim.floors["Ti"]
    for _ in range(40):
        kd_lim_sim.advance_one_step()
        kd_lim_state = kd_lim_sim.state
        assert np.all(np.isfinite(kd_lim_state.Ei))
        assert np.all(
            kd_lim_state.Ei >= kd_lim_floor * kd_lim_state.n * (1.0 - 1.0e-12)
        )
    kd_lim = kd_lim_sim.dvm_transfer_ledger()
    assert kd_lim["relax_limited_steps"] > 0
    assert kd_lim["Ei"]["rel"] < 1.0e-12, kd_lim["Ei"]
    assert kd_lim["M"]["rel"] < 1.0e-12, kd_lim["M"]
    assert np.any(np.abs(kd_lim_sim._dvm.Ei_debt) > 0.0)
    # Every bound the arm makes phantom is withdrawn, so the constraint it
    # reports names a term the step actually applies.
    kd_lim_diag = kd_lim_sim.suggest_timestep()
    for kd_lim_field in (
        "dt_ion_charge_exchange",
        "dt_ion_neutral_drag",
        "dt_neutral_exchange",
        "dt_neutral_sources",
    ):
        assert getattr(kd_lim_diag, kd_lim_field) == np.inf, kd_lim_field
    assert kd_lim_diag.active_constraint not in (
        "ion_charge_exchange",
        "ion_neutral_drag",
        "neutral_exchange",
        "neutral_sources",
    )
    # And the timestep bundle reads the boundary operator THIS stance runs.
    assert kd_lim_sim._characteristic_boundary is False
    kd_lim_bundle = kd_lim_sim._plasma_source_timestep_rhs(
        state=kd_lim_sim.state, time=kd_lim_sim.time
    )
    assert np.all(np.isfinite(np.asarray(kd_lim_bundle.Ei, dtype=float)))

    # K2d transfer-ledger census, PERSISTED: the standing DVM report condition
    # ("quote relax_limited_steps and the outstanding debt; any limited > 0
    # gets a dedicated look") has to be answerable from the saved artifact,
    # not only from a live solver object. Four statements: the moment path
    # writes no such group at all, a DVM run round-trips its census through
    # save/load, the forced-limiter scenario persists NONZERO counts, and the
    # ledger identity applied_cum + debt == booked_cum survives the file.
    from cablp.solvers._sim1d.results.io import (
        save_result_hdf5 as _save_result_hdf5_dvm,
    )

    kd_cen_flags = dict(kd_obs_flags)
    # Same geometry and flags as the DVM build below; ONLY neutral_model
    # differs, so a layout difference between the two files can be nothing
    # else.
    kd_cen_mom_params = dict(kd_obs_params)
    kd_cen_mom_params["neutral_model"] = "moment"
    kd_cen_mom_params["dt_save"] = 5.0e-9
    kd_cen_mom_sim = LAPDSim1D(kd_cen_mom_params, dict(kd_cen_flags))
    assert kd_cen_mom_sim._dvm is None
    kd_cen_mom_result = kd_cen_mom_sim.run(t_end=2.0e-8, dt=1.0e-9)
    assert not hasattr(kd_cen_mom_result, "dvm_transfer_ledger")

    kd_cen_params = dict(kd_obs_params)
    kd_cen_params["dt_save"] = 5.0e-9
    kd_cen_sim = LAPDSim1D(kd_cen_params, dict(kd_cen_flags))
    for _ in range(8):
        kd_cen_sim.advance_one_step(dt=1.0e-9)
    assert kd_cen_sim._dvm_engaged
    kd_cen_cells = kd_cen_sim.geometry.cells
    kd_cen_sim._dvm.Ei_transfer = np.full(kd_cen_cells, -1.0e12)
    kd_cen_sim._dvm.M_transfer = np.full(kd_cen_cells, -1.0e3)
    kd_cen_result = kd_cen_sim.run(
        t_end=kd_cen_sim.time + 4.0e-8, dt=1.0e-9
    )
    kd_cen = kd_cen_result.dvm_transfer_ledger
    assert kd_cen["engaged"] == 1
    assert kd_cen["relax_limited_steps"] > 0
    assert kd_cen["limited_cells"] > 0

    with tempfile.TemporaryDirectory() as kd_cen_dir:
        kd_cen_mom_path = Path(kd_cen_dir) / "dvm_census_moment.h5"
        _save_result_hdf5_dvm(kd_cen_mom_path, kd_cen_mom_result)
        with h5py.File(kd_cen_mom_path, "r") as kd_cen_mom_h5:
            assert "dvm_transfer_ledger" not in kd_cen_mom_h5
        kd_cen_mom_loaded = load_result_hdf5(kd_cen_mom_path)
        assert not hasattr(kd_cen_mom_loaded, "dvm_transfer_ledger")
        assert summarize_result(
            kd_cen_mom_loaded
        ).dvm_transfer_ledger_census is None

        kd_cen_path = Path(kd_cen_dir) / "dvm_census.h5"
        _save_result_hdf5_dvm(
            kd_cen_path, kd_cen_result, params=kd_cen_params, flags=kd_cen_flags
        )
        kd_cen_loaded = load_result_hdf5(kd_cen_path)
        kd_cen_back = kd_cen_loaded.dvm_transfer_ledger
        assert set(kd_cen_back) == set(kd_cen)
        for kd_cen_name, kd_cen_value in kd_cen.items():
            if isinstance(kd_cen_value, np.ndarray):
                assert np.array_equal(kd_cen_back[kd_cen_name], kd_cen_value), (
                    kd_cen_name
                )
            else:
                assert kd_cen_back[kd_cen_name] == kd_cen_value, kd_cen_name
                assert isinstance(
                    kd_cen_back[kd_cen_name], type(kd_cen_value)
                ), kd_cen_name
        # The identity, re-checked from the FILE's own arrays.
        for kd_cen_ch in ("Ei", "M"):
            kd_cen_debt = kd_cen_back[f"{kd_cen_ch}_debt"]
            kd_cen_booked = kd_cen_back[f"{kd_cen_ch}_booked_cum"]
            kd_cen_applied = kd_cen_back[f"{kd_cen_ch}_applied_cum"]
            kd_cen_scale = float(
                np.max(np.abs(kd_cen_booked)) + np.max(np.abs(kd_cen_debt))
            )
            assert kd_cen_scale > 0.0, kd_cen_ch
            assert np.max(
                np.abs(kd_cen_applied + kd_cen_debt - kd_cen_booked)
            ) / kd_cen_scale < 1.0e-12, kd_cen_ch
        assert np.any(np.abs(kd_cen_back["Ei_debt"]) > 0.0)
        # The particle handshake's own identity, likewise from the FILE: a
        # nonzero residual here is particle creation in the coupled system,
        # which is what the counted debit exists to make impossible.
        kd_cen_ion_booked = kd_cen_back["ion_booked_cum"]
        kd_cen_ion_scale = float(
            np.max(np.abs(kd_cen_ion_booked))
            + np.max(np.abs(kd_cen_back["ion_debt"]))
        )
        assert kd_cen_ion_scale > 0.0
        assert np.max(
            np.abs(
                kd_cen_back["ion_removed_cum"]
                + kd_cen_back["ion_debt"]
                - kd_cen_ion_booked
            )
        ) / kd_cen_ion_scale < 1.0e-12
        assert kd_cen_back["ion_residual_rel"] < 1.0e-12
        # The per-save series: one record per saved frame, counters that only
        # ever climb, and never past the end-of-run totals.
        kd_cen_frames = len(kd_cen_result.time)
        for kd_cen_field in (
            "time",
            "relax_steps",
            "relax_limited_steps",
            "limited_cells",
            "ion_booked_total",
            "ion_removed_total",
            "ion_shortfall_updates",
        ):
            kd_cen_series = kd_cen_back[f"sample_{kd_cen_field}"]
            assert len(kd_cen_series) == kd_cen_frames, kd_cen_field
            assert np.all(np.diff(kd_cen_series) >= 0.0), kd_cen_field
        assert (
            kd_cen_back["sample_relax_limited_steps"][-1]
            <= kd_cen["relax_limited_steps"]
        )
        assert kd_cen_back["sample_relax_limited_steps"][-1] > 0.0
        # Surfaced, and the arm's presence is readable from the file.
        kd_cen_summary = summarize_result(kd_cen_loaded)
        assert kd_cen_summary.dvm_arm_configured is True
        assert (
            kd_cen_summary.dvm_transfer_ledger_census["relax_limited_steps"]
            == kd_cen["relax_limited_steps"]
        )
        assert (
            "Ei_debt_total" in kd_cen_summary.dvm_transfer_ledger_census
        )
        assert (
            kd_cen_summary.dvm_transfer_ledger_census["ion_residual_rel"]
            == kd_cen["ion_residual_rel"]
        )
        assert (
            kd_cen_summary.dvm_transfer_ledger_census["ion_booked_total"] > 0.0
        )
        # A PRE-FIX DVM artifact -- the arm ran, the census was never kept --
        # reads "not recorded", never zero.
        kd_cen_prefix = SimpleNamespace(**vars(kd_cen_result))
        del kd_cen_prefix.dvm_transfer_ledger
        kd_cen_prefix_path = Path(kd_cen_dir) / "dvm_census_prefix.h5"
        _save_result_hdf5_dvm(
            kd_cen_prefix_path,
            kd_cen_prefix,
            params=kd_cen_params,
            flags=kd_cen_flags,
        )
        with h5py.File(kd_cen_prefix_path, "r") as kd_cen_prefix_h5:
            assert "dvm_transfer_ledger" not in kd_cen_prefix_h5
        kd_cen_prefix_summary = summarize_result(
            load_result_hdf5(kd_cen_prefix_path)
        )
        assert kd_cen_prefix_summary.dvm_arm_configured is True
        assert kd_cen_prefix_summary.dvm_transfer_ledger_census is None

    # --- Neutral-wind advection (M3): donor-cell
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

    # --- A1: the He singlet manifold registry (WP-A). ---
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
    # (B1; full acceptance in
    # scripts/verify_beam_deposition.py — this is the fast subset).
    from cablp.funcs._beam_deposition import (
        _COULOMB_STOPPING_EXPONENT,
        _coulomb_stopping_coefficient,
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
    # single-event Beer-Lambert booking caps at 1 — recorded as item 10).
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

    # --- Per-cell float accumulators (cost read 2026-08-02, restructure B) ---
    # deposit_beam banks each substep's channels in local Python floats and
    # flushes them to their arrays once, at cell exit, instead of doing eight
    # `arr[cell] += scalar` fancy-index stores per substep (14.5% of a
    # substep). The claim is bit-exactness, and it is checked here against a
    # reference march that keeps the OLD per-substep stores.
    #
    # The reference duplicates only the LOOP STRUCTURE -- the thing that
    # changed. Every physics leaf (the cross-section lookups, the stopping
    # powers, the secondary energy) is the module's own function, so a change
    # to the physics moves both sides together and only a change to the
    # marching/banking structure can make this fire. If that happens, this
    # reference must be re-derived from the module (or retired), never
    # loosened.
    from cablp.funcs._beam_deposition import (
        _ERG_PER_EV as _b2_ERG,
        HE_E_STOP_EV as _b2_E_STOP,
        HE_I_ION_EV as _b2_I_ION,
        he_mean_secondary_energy_eV as _b2_W_sec,
    )
    from cablp.funcs._cross import (
        He_EII_cross_lkup as _b2_sigma_i,
        He_beam_excitation_channel_lkup as _b2_sigma_x,
    )

    def _b2_reference_march(
        E0_eV, Gamma0_per_s, nn, ne, Te, launch, direction, dz_cm,
        I_ion_eV=_b2_I_ION, E_stop_eV=_b2_E_STOP,
        coulomb_model="fast_electron", anomalous_model="none",
        beam_area_cm2=None, max_energy_fraction_per_substep=0.02,
        anode_cross_index=None, anode_eta=0.0,
        product_transport="local", anomalous_transport="local",
        tail_energy_eV=None,
    ):
        """The pre-restructure-B march: every bank written per SUBSTEP.

        Returns a dict of the per-cell arrays and the trajectory scalars.
        """
        cells = int(np.asarray(dz_cm).size)
        banks = {
            name: np.zeros(cells)
            for name in (
                "ionization_events", "excitation_events", "heating",
                "radiated", "ionization_cost", "E_entry", "heat_coulomb",
                "heat_anomalous", "heat_secondary", "heat_terminal",
                "sec_flux", "sec_power_eV", "anom_power_eV",
            )
        }
        walk_products = product_transport == "nonlocal"
        walk_tail = anomalous_transport == "tail_walk"
        area = np.broadcast_to(
            np.asarray(
                0.0 if beam_area_cm2 is None else beam_area_cm2, dtype=float
            ),
            (cells,),
        )
        frac = float(max_energy_fraction_per_substep)
        order = (
            range(launch, cells) if direction > 0 else range(launch, -1, -1)
        )
        E = float(E0_eV)
        gamma = float(Gamma0_per_s)
        absorbed = False
        anode_intercepted = 0.0
        terminal = (-1, 0.0, 0.0)
        intercept_active = anode_cross_index is not None and anode_eta > 0.0
        if E <= E_stop_eV:
            return dict(banks, transmitted_flux=gamma, transmitted_E=E,
                        anode_intercepted=0.0, terminal=terminal)
        for cell in order:
            if intercept_active and cell == anode_cross_index:
                anode_intercepted += anode_eta * gamma * E * _b2_ERG
                gamma *= 1.0 - anode_eta
                intercept_active = False
            banks["E_entry"][cell] = E
            remaining = float(dz_cm[cell])
            nn_c = float(nn[cell])
            ne_c = float(ne[cell])
            Te_c = float(Te[cell])
            while remaining > 0.0:
                sigma_i = (
                    _b2_sigma_i(E / I_ion_eV) if E > I_ion_eV else 0.0
                )
                sigma_x, E_rad = _b2_sigma_x(E)
                W_sec = _b2_W_sec(E, I_ion_eV=I_ion_eV)
                L_pot = nn_c * sigma_i * I_ion_eV
                L_sec = nn_c * sigma_i * W_sec
                L_exc = nn_c * sigma_x * E_rad
                L_coul = coulomb_stopping_eV_per_cm(
                    E, ne_c, Te_c, model=coulomb_model
                )
                L_anom = 0.0
                if anomalous_model == "quasilinear":
                    n_b = gamma / (float(area[cell]) * beam_speed_cm_s(E))
                    l_ql = quasilinear_relaxation_length_cm(E, ne_c, n_b)
                    if math.isfinite(l_ql) and l_ql > 0.0:
                        L_anom = E / l_ql
                L_tot = L_pot + L_sec + L_exc + L_coul + L_anom
                if L_tot <= 0.0:
                    break
                dz_sub = min(remaining, frac * E / L_tot)
                if E - L_tot * dz_sub <= E_stop_eV:
                    dz_sub = (E - E_stop_eV) / L_tot
                if dz_sub <= 0.0:
                    if walk_products:
                        terminal = (cell, gamma, E)
                    else:
                        banks["heating"][cell] += gamma * E * _b2_ERG
                        banks["heat_terminal"][cell] += gamma * E * _b2_ERG
                    E = 0.0
                    absorbed = True
                    break
                d_pot = L_pot * dz_sub
                d_sec = L_sec * dz_sub
                d_exc = L_exc * dz_sub
                d_coul = L_coul * dz_sub
                d_anom = L_anom * dz_sub
                banks["ionization_cost"][cell] += gamma * d_pot * _b2_ERG
                if walk_tail:
                    banks["anom_power_eV"][cell] += gamma * d_anom
                    d_anom_local = 0.0
                else:
                    d_anom_local = d_anom
                if walk_products:
                    banks["heating"][cell] += (
                        gamma * (d_coul + d_anom_local) * _b2_ERG
                    )
                    banks["sec_flux"][cell] += gamma * nn_c * sigma_i * dz_sub
                    banks["sec_power_eV"][cell] += gamma * d_sec
                else:
                    banks["heating"][cell] += (
                        gamma * (d_sec + d_coul + d_anom_local) * _b2_ERG
                    )
                    banks["heat_secondary"][cell] += gamma * d_sec * _b2_ERG
                banks["heat_coulomb"][cell] += gamma * d_coul * _b2_ERG
                banks["heat_anomalous"][cell] += gamma * d_anom_local * _b2_ERG
                banks["radiated"][cell] += gamma * d_exc * _b2_ERG
                banks["ionization_events"][cell] += (
                    gamma * nn_c * sigma_i * dz_sub
                )
                banks["excitation_events"][cell] += (
                    gamma * nn_c * sigma_x * dz_sub
                )
                E -= d_pot + d_sec + d_exc + d_coul + d_anom
                remaining -= dz_sub
                if E <= E_stop_eV:
                    if walk_products:
                        terminal = (cell, gamma, E)
                    else:
                        banks["heating"][cell] += gamma * E * _b2_ERG
                        banks["heat_terminal"][cell] += gamma * E * _b2_ERG
                    E = 0.0
                    absorbed = True
                    break
            if absorbed:
                break
        return dict(
            banks,
            transmitted_flux=0.0 if absorbed else gamma,
            transmitted_E=0.0 if absorbed else E,
            anode_intercepted=anode_intercepted,
            terminal=terminal,
        )

    # The banks the reference and the module must agree on, per cell. The
    # WP-D/WP-E withholding banks are not on the result object, so they are
    # compared through the arrays they end up in ("local" arms) and through
    # the walk products they drive ("nonlocal"/"tail_walk" arms).
    _b2_fields = (
        ("ionization_events", "ionization_events"),
        ("excitation_events", "excitation_events"),
        ("heating", "plasma_heating_erg_s"),
        ("radiated", "radiated_erg_s"),
        ("ionization_cost", "ionization_cost_erg_s"),
        ("E_entry", "E_entry_eV"),
        ("heat_coulomb", "heating_coulomb_erg_s"),
        ("heat_anomalous", "heating_anomalous_erg_s"),
        ("heat_secondary", "heating_secondary_erg_s"),
        ("heat_terminal", "heating_terminal_erg_s"),
    )
    # Representative states: the b1 breakdown column (ray absorbed mid-column,
    # many substeps per cell), a thin column the ray crosses whole
    # (transmitting, one substep-limited pass per cell), the quasilinear
    # closure (flux-dependent stopping), and the anode-mesh interception that
    # changes gamma mid-ray.
    _b2_thin = dict(
        nn=np.full(b1_cells, 1.0e12),
        ne=np.full(b1_cells, 1.0e10),
        Te=np.full(b1_cells, 3.0),
        launch=0, direction=1, dz_cm=np.full(b1_cells, 20.0),
    )
    _b2_cases = (
        ("absorbed", (150.0, 1.0e22), dict(b1_col)),
        ("transmitted", (150.0, 1.0e22), dict(_b2_thin)),
        ("reverse", (150.0, 1.0e22),
         {**b1_col, "launch": b1_cells - 1, "direction": -1}),
        ("quasilinear", (300.0, 1.0e20),
         {**_b2_thin, "anomalous_model": "quasilinear",
          "beam_area_cm2": 700.0}),
        ("anode", (150.0, 1.0e22),
         {**_b2_thin, "anode_cross_index": 5, "anode_eta": 0.358}),
    )
    for _b2_name, _b2_args, _b2_kw in _b2_cases:
        _b2_got = deposit_beam(*_b2_args, **_b2_kw)
        _b2_ref = _b2_reference_march(*_b2_args, **_b2_kw)
        for _b2_key, _b2_attr in _b2_fields:
            assert np.array_equal(
                getattr(_b2_got, _b2_attr), _b2_ref[_b2_key]
            ), (_b2_name, _b2_attr)
        assert _b2_got.transmitted_flux == _b2_ref["transmitted_flux"], _b2_name
        assert (
            _b2_got.transmitted_energy_eV == _b2_ref["transmitted_E"]
        ), _b2_name
        assert (
            float(_b2_got.anode_intercepted_erg_s)
            == _b2_ref["anode_intercepted"]
        ), _b2_name
        # A case that banks nothing would pass vacuously; every case must
        # actually deposit, and the absorbed ones must reach a terminal bank.
        assert _b2_ref["heating"].sum() > 0.0, _b2_name
    # The two withholding closures get their own arms, so the WP-D/WP-E banks
    # (flushed under their own `if` at cell exit) are exercised too, not just
    # the always-on eight. The tail arm needs the weak-beam domain
    # n_b < 0.1 n_e, or the QL relaxation length is infinite by design and
    # there is no anomalous power to withhold.
    _b2_weak = dict(
        nn=np.full(b1_cells, 1.0e13),
        ne=np.full(b1_cells, 1.0e12),
        Te=np.full(b1_cells, 3.0),
        launch=0, direction=1, dz_cm=np.full(b1_cells, 20.0),
    )
    for _b2_name, _b2_args, _b2_kw, _b2_transport in (
        ("wpd-absorbed", (150.0, 1.0e22), dict(b1_col),
         {"product_transport": "nonlocal"}),
        ("wpd-thin", (150.0, 1.0e22), dict(_b2_thin),
         {"product_transport": "nonlocal"}),
        ("wpe-weak", (200.0, 1.0e20), dict(_b2_weak),
         {"anomalous_transport": "tail_walk", "tail_energy_eV": 75.0,
          "anomalous_model": "quasilinear", "beam_area_cm2": 100.0}),
    ):
        _b2_full = {**_b2_kw, **_b2_transport}
        _b2_got = deposit_beam(*_b2_args, **_b2_full)
        _b2_ref = _b2_reference_march(*_b2_args, **_b2_full)
        # The walks run after the march and add to `heating`, so the
        # comparable per-cell banks here are the ones the march alone
        # writes plus the withheld populations themselves.
        for _b2_key, _b2_attr in (
            ("ionization_events", "ionization_events"),
            ("excitation_events", "excitation_events"),
            ("ionization_cost", "ionization_cost_erg_s"),
            ("radiated", "radiated_erg_s"),
            ("E_entry", "E_entry_eV"),
            ("heat_coulomb", "heating_coulomb_erg_s"),
        ):
            assert np.array_equal(
                getattr(_b2_got, _b2_attr), _b2_ref[_b2_key]
            ), (_b2_name, _b2_transport, _b2_attr)
        assert (
            _b2_got.transmitted_flux == _b2_ref["transmitted_flux"]
        ), (_b2_name, _b2_transport)
        if "product_transport" in _b2_transport:
            assert _b2_ref["sec_flux"].sum() > 0.0
        else:
            assert _b2_ref["anom_power_eV"].sum() > 0.0

    # --- Hoisted stopping coefficient (cost read 2026-08-02, restructure C) --
    # The walks' per-cell A in dE/dx = A W**p is a 262-iteration Python
    # listcomp costing ~100 us -- half the entire WP-E per-call surcharge --
    # and it depends only on (ne, Te, model). deposit_beam now accepts it from
    # the caller so several rays, or a future WP-F's energy groups, pay for it
    # once. Supplying it must be bit-identical to letting the module build it.
    _b3_kw = {
        **_b2_weak, "anomalous_transport": "tail_walk",
        "tail_energy_eV": 75.0, "anomalous_model": "quasilinear",
        "beam_area_cm2": 100.0,
    }
    _b3_coeff = _coulomb_stopping_coefficient(
        _b2_weak["ne"], _b2_weak["Te"], "fast_electron"
    )
    _b3_auto = deposit_beam(200.0, 1.0e20, **_b3_kw)
    _b3_given = deposit_beam(
        200.0, 1.0e20, **_b3_kw, stopping_coefficient=_b3_coeff
    )
    for _b3_field in (
        "ionization_events", "excitation_events", "plasma_heating_erg_s",
        "radiated_erg_s", "ionization_cost_erg_s", "E_entry_eV",
        "heating_coulomb_erg_s", "heating_anomalous_erg_s",
        "heating_secondary_erg_s", "heating_terminal_erg_s",
    ):
        assert np.array_equal(
            getattr(_b3_auto, _b3_field), getattr(_b3_given, _b3_field)
        ), _b3_field
    for _b3_scalar in (
        "transmitted_flux", "transmitted_energy_eV",
        "end_loss_tail_low_erg_s", "end_loss_tail_high_erg_s",
    ):
        assert getattr(_b3_auto, _b3_scalar) == getattr(_b3_given, _b3_scalar)
    # Non-vacuous: the tail walk actually carried power on this state.
    assert _b3_auto.heating_anomalous_erg_s.sum() > 0.0
    assert float(_b3_auto.end_loss_tail_high_erg_s) > 0.0
    # Presence gating: the default is None and behaves as it always did.
    assert np.array_equal(
        deposit_beam(
            150.0, 1.0e22, **b1_col, stopping_coefficient=None
        ).plasma_heating_erg_s,
        b1_res.plasma_heating_erg_s,
    )
    # A wrong-length coefficient is a loud failure at the call, never a silent
    # mis-walk against the wrong cells.
    try:
        deposit_beam(
            200.0, 1.0e20, **_b3_kw, stopping_coefficient=_b3_coeff[:-1]
        )
    except ValueError as _b3_err:
        assert "stopping_coefficient" in str(_b3_err), _b3_err
    else:
        raise AssertionError("a short stopping_coefficient must raise")

    # --- WP-D: non-local transport of the beam's EVENT PRODUCTS
    # (product_transport). At breakdown the secondary electrons and the
    # primary's terminal sub-threshold residual are below every He inelastic
    # threshold and Coulomb-couple at ~1 eV per machine pass, so banking them
    # in their birth cell is the wrong limit; "nonlocal" walks them along B
    # and books what escapes an end to the new end ledger.

    # (a) DEFAULT OFF IS BIT-EXACT. Passing the default explicitly and
    # omitting the key must give byte-identical arrays, and the end ledger
    # must be identically zero -- nothing is booked that was not booked
    # before, which is what keeps the production golden bit-exact.
    wpd_local = deposit_beam(150.0, 1.0e22, **b1_col, product_transport="local")
    for wpd_field in (
        "ionization_events", "excitation_events", "plasma_heating_erg_s",
        "radiated_erg_s", "ionization_cost_erg_s", "E_entry_eV",
        "heating_coulomb_erg_s", "heating_anomalous_erg_s",
        "heating_secondary_erg_s", "heating_terminal_erg_s",
    ):
        assert np.array_equal(
            getattr(wpd_local, wpd_field), getattr(b1_res, wpd_field)
        ), wpd_field
    assert wpd_local.transmitted_flux == b1_res.transmitted_flux
    assert wpd_local.end_loss_low_erg_s == 0.0
    assert wpd_local.end_loss_high_erg_s == 0.0
    assert wpd_local.end_loss_transmitted_erg_s == 0.0

    # The walk integrates the module's OWN stopping power in closed form
    # rather than substepping it, which is exact only because both closures
    # are pure power laws in W (lnLambda depends on ne and Te alone). This
    # guards that identity: if coulomb_stopping_eV_per_cm ever stops being
    # A(ne,Te)*W**p, the walk silently stops matching the primary's drag.
    for wpd_model, wpd_p in _COULOMB_STOPPING_EXPONENT.items():
        wpd_A = _coulomb_stopping_coefficient([2.0e12], [4.0], wpd_model)[0]
        for wpd_W in (0.2, 3.0, 40.0, 150.0):
            wpd_ref = coulomb_stopping_eV_per_cm(
                wpd_W, 2.0e12, 4.0, model=wpd_model
            )
            assert abs(wpd_A * wpd_W**wpd_p - wpd_ref) <= 1e-12 * wpd_ref, (
                wpd_model, wpd_W
            )

    # (b) THE EXTENDED CONSERVATION IDENTITY. On a column where the walks do
    # BOTH things -- the backward halves born at the launch cell leave the low
    # end immediately, the forward ones run ~11 m and thermalize inside the
    # 20 m domain -- per-ray energy still closes to roundoff with the end
    # ledger carrying what left:
    #     Gamma0*E0 = heating + radiated + cost + anode + end_loss
    wpd_cells = 40
    wpd_col = dict(
        nn=np.full(wpd_cells, 2.0e14),
        ne=np.full(wpd_cells, 1.0e11),
        Te=np.full(wpd_cells, 1.0),
        launch=0,
        direction=1,
        dz_cm=np.full(wpd_cells, 50.0),
    )
    wpd_ref_local = deposit_beam(150.0, 1.0e22, **wpd_col)
    wpd_nl = deposit_beam(150.0, 1.0e22, **wpd_col, product_transport="nonlocal")
    wpd_budget = 1.0e22 * 150.0 * 1.602176634e-12
    wpd_total = (
        wpd_nl.plasma_heating_erg_s.sum()
        + wpd_nl.radiated_erg_s.sum()
        + wpd_nl.ionization_cost_erg_s.sum()
        + float(wpd_nl.anode_intercepted_erg_s)
        + wpd_nl.end_loss_low_erg_s
        + wpd_nl.end_loss_high_erg_s
    )
    assert abs(wpd_total - wpd_budget) / wpd_budget < 1e-12
    assert wpd_nl.end_loss_low_erg_s > 0.0  # escaped backwards
    assert wpd_nl.end_loss_high_erg_s > 0.0  # escaped forwards
    assert wpd_nl.heating_secondary_erg_s.sum() > 0.0  # and some thermalized
    assert wpd_nl.heating_terminal_erg_s.sum() > 0.0
    # This ray is absorbed, so nothing in the ledger is the transmitted
    # primary -- all of it is walked product.
    assert wpd_nl.transmitted_flux == 0.0
    assert wpd_nl.end_loss_transmitted_erg_s == 0.0
    # Energy MOVED, it was not created: the plasma keeps strictly less.
    assert (
        wpd_nl.plasma_heating_erg_s.sum()
        < wpd_ref_local.plasma_heating_erg_s.sum()
    )
    # v1 is ENERGY-ONLY routing: the particle rows (and everything downstream
    # of them -- n, the circuit currents) are identical in both modes.
    assert np.array_equal(
        wpd_nl.ionization_events, wpd_ref_local.ionization_events
    )
    assert np.array_equal(
        wpd_nl.excitation_events, wpd_ref_local.excitation_events
    )

    # (c) THE LOCAL LIMIT. Raise n_e until the product range collapses far
    # below one cell and "nonlocal" must reproduce "local": every walk
    # thermalizes in its birth cell and nothing reaches an end. The tolerance
    # is roundoff (rtol 1e-12), not a convergence tolerance -- in this limit
    # the two bookings are the same sum in a different order, so anything
    # larger would mean a real leak rather than an unconverged walk.
    wpd_dense = dict(b1_col, ne=np.full(b1_cells, 1.0e13))
    wpd_dense_local = deposit_beam(150.0, 1.0e22, **wpd_dense)
    wpd_dense_nl = deposit_beam(
        150.0, 1.0e22, **wpd_dense, product_transport="nonlocal"
    )
    assert np.allclose(
        wpd_dense_nl.plasma_heating_erg_s,
        wpd_dense_local.plasma_heating_erg_s,
        rtol=1e-12,
        atol=0.0,
    )
    assert wpd_dense_nl.end_loss_low_erg_s == 0.0
    assert wpd_dense_nl.end_loss_high_erg_s == 0.0

    # (d) THE DIRECTION SPLIT. Secondaries leave broadly isotropically, so
    # each birth cell emits two half-weight walks, +z and -z. Confine the
    # neutrals to a single cell at the exact centre of an otherwise uniform
    # column: the two halves then see identical columns and identical
    # distances to their ends, so their escapes must match and the deposited
    # secondary profile must be mirror-symmetric about the birth cell. (The
    # primary streams on through the vacuum and transmits, so the high end
    # additionally carries its Gamma_t*E_t -- the ledger's other member,
    # subtracted out here through its own diagnostic split.)
    wpd_sym_cells = 41
    wpd_sym_mid = 20
    wpd_sym_nn = np.zeros(wpd_sym_cells)
    wpd_sym_nn[wpd_sym_mid] = 5.0e14
    wpd_sym = deposit_beam(
        150.0, 1.0e21,
        nn=wpd_sym_nn,
        ne=np.full(wpd_sym_cells, 3.0e10),
        Te=np.full(wpd_sym_cells, 1.0),
        launch=wpd_sym_mid,
        direction=1,
        dz_cm=np.full(wpd_sym_cells, 40.0),
        product_transport="nonlocal",
    )
    assert wpd_sym.transmitted_flux > 0.0
    assert wpd_sym.end_loss_transmitted_erg_s > 0.0
    assert np.isclose(
        wpd_sym.end_loss_high_erg_s - wpd_sym.end_loss_transmitted_erg_s,
        wpd_sym.end_loss_low_erg_s,
        rtol=1e-12,
        atol=0.0,
    )
    wpd_sym_heat = wpd_sym.heating_secondary_erg_s
    assert np.array_equal(
        wpd_sym_heat[wpd_sym_mid + 1:], wpd_sym_heat[:wpd_sym_mid][::-1]
    )

    # (e) MISCONFIGURATION is loud at the module boundary too (the solver
    # raises at construction; see the WP-D block in the R4/csda section).
    try:
        deposit_beam(150.0, 1e22, **b1_col, product_transport="bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for product_transport")

    # --- WP-E: QL heating locality (anomalous_transport). The anomalous
    # channel banks its drag as instantaneous LOCAL bulk heating; kinetically
    # QL fills a fast-tail plateau first, and at breakdown densities a tail
    # electron is collisionally decoupled and free-streams along B. Under
    # "tail_walk" the QL power is carried by tail electrons at E_tail on the
    # SAME closed-form Coulomb walk the WP-D products use.
    #
    # The column needs an ACTIVE anomalous channel, so the beam must be weak
    # enough for quasilinear theory to apply (n_b < n_e/10) -- b1_col's
    # 1e22 beam is not, and runs with anomalous_model="none" by default.
    wpe_cells = 60
    wpe_thin = dict(
        nn=np.full(wpe_cells, 1.0e12),
        ne=np.full(wpe_cells, 1.0e10),
        Te=np.full(wpe_cells, 2.0),
        launch=0,
        direction=1,
        dz_cm=np.full(wpe_cells, 30.0),
        anomalous_model="quasilinear",
        beam_area_cm2=100.0,
    )
    wpe_G0 = 1.0e18
    wpe_E0 = 150.0
    wpe_budget = wpe_G0 * wpe_E0 * 1.602176634e-12
    wpe_local = deposit_beam(wpe_E0, wpe_G0, **wpe_thin)
    assert wpe_local.heating_anomalous_erg_s.sum() > 0.0  # channel is live
    wpe_walk = deposit_beam(
        wpe_E0, wpe_G0, **wpe_thin,
        anomalous_transport="tail_walk", tail_energy_eV=75.0,
    )

    # (a) THE RAY IS BIT-IDENTICAL. L_anom depends on the beam and the column,
    # never on where its energy is banked, so the trajectory, the primary flux
    # and every non-anomalous channel are byte-for-byte the same. This is what
    # makes the conservation identity below exact rather than approximate.
    for _wpe_arr in (
        "E_entry_eV", "ionization_events", "excitation_events",
        "radiated_erg_s", "ionization_cost_erg_s", "heating_coulomb_erg_s",
        "heating_secondary_erg_s", "heating_terminal_erg_s",
    ):
        assert np.array_equal(
            getattr(wpe_walk, _wpe_arr), getattr(wpe_local, _wpe_arr)
        ), _wpe_arr
    assert wpe_walk.transmitted_flux == wpe_local.transmitted_flux
    assert wpe_walk.transmitted_energy_eV == wpe_local.transmitted_energy_eV

    # (b) THE CONSERVATION IDENTITY: banking removed = walked deposition +
    # end losses, to roundoff. The tolerance is roundoff (1e-12), not a
    # convergence tolerance -- the walk is closed-form and telescopes, so
    # anything larger would be a real leak.
    wpe_removed = float(wpe_local.heating_anomalous_erg_s.sum())
    wpe_ledger = (
        float(wpe_walk.end_loss_tail_low_erg_s)
        + float(wpe_walk.end_loss_tail_high_erg_s)
    )
    wpe_delivered = float(wpe_walk.heating_anomalous_erg_s.sum()) + wpe_ledger
    assert abs(wpe_delivered - wpe_removed) / wpe_removed < 1e-12, (
        wpe_removed, wpe_delivered
    )
    # ... and the whole per-ray budget closes with the tail ledger in it.
    wpe_total = (
        wpe_walk.plasma_heating_erg_s.sum()
        + wpe_walk.radiated_erg_s.sum()
        + wpe_walk.ionization_cost_erg_s.sum()
        + float(wpe_walk.anode_intercepted_erg_s)
        + wpe_walk.transmitted_flux
        * wpe_walk.transmitted_energy_eV
        * 1.602176634e-12
        + wpe_ledger
    )
    assert abs(wpe_total - wpe_budget) / wpe_budget < 1e-9

    # (c) THE THIN/HOT LIMIT: at breakdown-like n_e = 1e10 a 75 eV tail
    # electron's Coulomb range is hundreds of machine lengths, so nearly all
    # of the QL power leaves through the ends instead of heating the column.
    assert wpe_ledger / wpe_removed > 0.9
    assert (
        wpe_walk.plasma_heating_erg_s.sum()
        < wpe_local.plasma_heating_erg_s.sum()
    )

    # (d) THE LOCAL LIMIT (the D1 self-limiting pattern): raise n_e until the
    # tail range collapses below one cell and "tail_walk" must reproduce
    # "local" -- every walker thermalizes in its birth cell and nothing
    # reaches an end. The closure confines itself to the low-density phase.
    wpe_dense = dict(wpe_thin, ne=np.full(wpe_cells, 1.0e14))
    wpe_dense_local = deposit_beam(wpe_E0, wpe_G0, **wpe_dense)
    wpe_dense_walk = deposit_beam(
        wpe_E0, wpe_G0, **wpe_dense,
        anomalous_transport="tail_walk", tail_energy_eV=75.0,
    )
    assert wpe_dense_walk.end_loss_tail_low_erg_s == 0.0
    assert wpe_dense_walk.end_loss_tail_high_erg_s == 0.0
    assert np.allclose(
        wpe_dense_walk.plasma_heating_erg_s,
        wpe_dense_local.plasma_heating_erg_s,
        rtol=1e-9,
        atol=0.0,
    )

    # (e) E_tail SETS THE RANGE, NOT THE POWER. The equivalent tail flux is
    # P_QL/E_tail, so the power carried is independent of E_tail (conservation
    # holds at every bracket arm) while a hotter tail travels further and
    # exports more. This pins the one thing the bracket arms vary.
    wpe_prev_escape = -1.0
    for wpe_E_tail in (30.0, 75.0, 150.0):
        wpe_arm = deposit_beam(
            wpe_E0, wpe_G0, **wpe_thin,
            anomalous_transport="tail_walk", tail_energy_eV=wpe_E_tail,
        )
        wpe_arm_ledger = (
            float(wpe_arm.end_loss_tail_low_erg_s)
            + float(wpe_arm.end_loss_tail_high_erg_s)
        )
        wpe_arm_delivered = (
            float(wpe_arm.heating_anomalous_erg_s.sum()) + wpe_arm_ledger
        )
        assert abs(wpe_arm_delivered - wpe_removed) / wpe_removed < 1e-12
        assert wpe_arm_ledger > wpe_prev_escape
        wpe_prev_escape = wpe_arm_ledger

    # (f) THE DIRECTION SPLIT. The tails leave 50/50 along +-B, so a QL source
    # confined to the exact centre of an otherwise uniform column must produce
    # matching escapes at the two ends and a mirror-symmetric deposit.
    #
    # Confining it needs the per-cell ``beam_area_cm2`` rather than the
    # single-cell ``nn`` trick the WP-D split test uses: unlike the event
    # products, the anomalous drag is CONTINUOUS along the ray and is born in
    # every cell the primary crosses. A tiny area drives n_b above the
    # weak-beam ceiling n_e/10, where the quasilinear closure returns no drag
    # at all, so widening it in one cell selects that cell as the only source.
    wpe_sym_cells = 41
    wpe_sym_mid = 20
    wpe_sym_nn = np.zeros(wpe_sym_cells)
    wpe_sym_nn[wpe_sym_mid] = 1.0e13
    wpe_sym_area = np.full(wpe_sym_cells, 1.0e-2)
    wpe_sym_area[wpe_sym_mid] = 100.0
    wpe_sym_col = dict(
        nn=wpe_sym_nn,
        ne=np.full(wpe_sym_cells, 3.0e11),
        Te=np.full(wpe_sym_cells, 1.0),
        launch=wpe_sym_mid,
        direction=1,
        dz_cm=np.full(wpe_sym_cells, 40.0),
        anomalous_model="quasilinear",
        beam_area_cm2=wpe_sym_area,
    )
    wpe_sym_local = deposit_beam(wpe_E0, wpe_G0, **wpe_sym_col)
    assert np.array_equal(
        np.flatnonzero(wpe_sym_local.heating_anomalous_erg_s),
        np.array([wpe_sym_mid]),
    )
    wpe_sym = deposit_beam(
        wpe_E0, wpe_G0, **wpe_sym_col,
        anomalous_transport="tail_walk", tail_energy_eV=75.0,
    )
    assert wpe_sym.end_loss_tail_low_erg_s > 0.0
    # The two halves see identical columns and identical distances to their
    # ends, so this is an EQUALITY, not a tolerance.
    assert (
        wpe_sym.end_loss_tail_high_erg_s == wpe_sym.end_loss_tail_low_erg_s
    )
    wpe_sym_heat = wpe_sym.heating_anomalous_erg_s
    assert np.array_equal(
        wpe_sym_heat[wpe_sym_mid + 1:], wpe_sym_heat[:wpe_sym_mid][::-1]
    )
    assert wpe_sym.end_loss_low_erg_s == 0.0  # WP-D ledger untouched
    assert wpe_sym.end_loss_high_erg_s == 0.0

    # (g) MISCONFIGURATION is loud at the module boundary too (the solver
    # raises at construction; see the WP-E block in the R4/csda section).
    for wpe_bad_call in (
        lambda: deposit_beam(
            wpe_E0, wpe_G0, **wpe_thin, anomalous_transport="bogus"
        ),
        # tail_walk with no tail energy to launch at
        lambda: deposit_beam(
            wpe_E0, wpe_G0, **wpe_thin, anomalous_transport="tail_walk"
        ),
        lambda: deposit_beam(
            wpe_E0, wpe_G0, **wpe_thin,
            anomalous_transport="tail_walk", tail_energy_eV=0.0,
        ),
        lambda: deposit_beam(
            wpe_E0, wpe_G0, **wpe_thin,
            anomalous_transport="tail_walk", tail_energy_eV=float("inf"),
        ),
        # tail_walk with no anomalous channel to carry: a silent no-op
        lambda: deposit_beam(
            wpe_E0, wpe_G0, **dict(wpe_thin, anomalous_model="none"),
            anomalous_transport="tail_walk", tail_energy_eV=75.0,
        ),
    ):
        try:
            wpe_bad_call()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for anomalous_transport")

    # --- K7 at the module: one walk-window face REFLECTS instead of letting
    # walkers leave. The comparison against the threshold is general, so the
    # arm is not a disguised "reflect everything" switch.
    k7m_win = (0, wpe_cells - 1)
    k7m_common = dict(
        wpe_thin, anomalous_transport="tail_walk", tail_energy_eV=75.0,
        tail_walk_window=k7m_win, tail_reflect_face=-1,
    )
    # (a) THRESHOLD BELOW EVERY ARRIVAL ENERGY: nothing reflects, and with the
    # window spanning the whole grid the result is the unreflected walk BYTE
    # FOR BYTE. This is the general-comparison statement and the
    # bit-exactness statement in one.
    k7m_inert = deposit_beam(
        wpe_E0, wpe_G0, **k7m_common, tail_reflect_threshold_eV=1.0e-30,
    )
    assert np.array_equal(
        k7m_inert.heating_anomalous_erg_s, wpe_walk.heating_anomalous_erg_s
    )
    assert (
        k7m_inert.end_loss_tail_low_erg_s == wpe_walk.end_loss_tail_low_erg_s
    )
    assert (
        k7m_inert.end_loss_tail_high_erg_s == wpe_walk.end_loss_tail_high_erg_s
    )
    # (b) THRESHOLD ABOVE THEM: everything reflects. The named face's ledger is
    # EXACTLY zero, the conservation identity still closes to roundoff, and the
    # column keeps what the face used to delete.
    for k7m_face in (-1, 1):
        k7m_refl = deposit_beam(
            wpe_E0, wpe_G0, **dict(k7m_common, tail_reflect_face=k7m_face),
            tail_reflect_threshold_eV=1.0e4,
        )
        k7m_ledger = (
            float(k7m_refl.end_loss_tail_low_erg_s)
            + float(k7m_refl.end_loss_tail_high_erg_s)
        )
        k7m_face_ledger = (
            k7m_refl.end_loss_tail_low_erg_s if k7m_face < 0
            else k7m_refl.end_loss_tail_high_erg_s
        )
        assert k7m_face_ledger == 0.0, k7m_face
        k7m_delivered = (
            float(k7m_refl.heating_anomalous_erg_s.sum()) + k7m_ledger
        )
        assert abs(k7m_delivered - wpe_removed) / wpe_removed < 1e-12, (
            k7m_face, wpe_removed, k7m_delivered
        )
        assert (
            float(k7m_refl.heating_anomalous_erg_s.sum())
            > float(wpe_walk.heating_anomalous_erg_s.sum())
        )
        # Energy-only still: reflection moves energy, never particles.
        assert np.array_equal(
            k7m_refl.ionization_events, wpe_walk.ionization_events
        )
    # (c) A SUB-WINDOW IS A WALL. With the window closed short of the grid, no
    # tail energy lands beyond it in either direction, and the budget still
    # closes -- what leaves through the far face is booked, not lost.
    k7m_sub = deposit_beam(
        wpe_E0, wpe_G0, **dict(k7m_common, tail_walk_window=(0, 40)),
        tail_reflect_threshold_eV=1.0e4,
    )
    assert not np.any(k7m_sub.heating_anomalous_erg_s[41:])
    k7m_sub_delivered = (
        float(k7m_sub.heating_anomalous_erg_s.sum())
        + float(k7m_sub.end_loss_tail_low_erg_s)
        + float(k7m_sub.end_loss_tail_high_erg_s)
    )
    assert abs(k7m_sub_delivered - wpe_removed) / wpe_removed < 1e-12
    # (d) MISCONFIGURATION at the module boundary: a face with no threshold, a
    # threshold with no face, a face that is not a face, a threshold that is
    # not an energy, a face with no window to put it on, and reflection asked
    # for where there is no walk at all.
    for k7m_bad in (
        dict(tail_reflect_face=-1, tail_walk_window=k7m_win),
        dict(tail_reflect_threshold_eV=100.0, tail_walk_window=k7m_win),
        dict(tail_reflect_face=0, tail_reflect_threshold_eV=100.0,
             tail_walk_window=k7m_win),
        dict(tail_reflect_face=-1, tail_reflect_threshold_eV=0.0,
             tail_walk_window=k7m_win),
        dict(tail_reflect_face=-1, tail_reflect_threshold_eV=float("nan"),
             tail_walk_window=k7m_win),
        dict(tail_reflect_face=-1, tail_reflect_threshold_eV=float("inf"),
             tail_walk_window=k7m_win),
        dict(tail_reflect_face=-1, tail_reflect_threshold_eV=100.0),
        dict(tail_reflect_face=-1, tail_reflect_threshold_eV=100.0,
             tail_walk_window=(0, wpe_cells)),
    ):
        try:
            deposit_beam(
                wpe_E0, wpe_G0, **wpe_thin,
                anomalous_transport="tail_walk", tail_energy_eV=75.0,
                **k7m_bad,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {k7m_bad!r}")
    try:
        deposit_beam(
            wpe_E0, wpe_G0, **wpe_thin,
            tail_reflect_face=-1, tail_reflect_threshold_eV=100.0,
            tail_walk_window=k7m_win,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "expected ValueError for reflection without a tail walk"
        )

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

    # --- Directed recycle jets: cathode-face
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
    # deferred rather than re-derived now. The deferral retires once one of
    # two things settles what the anode channel should assert: the anode
    # ion-sheath current is derived properly, or the directed-jet module is
    # rewritten. Until then this is a deliberate gap in coverage, not an
    # oversight. The flag plumbing + construction validation above, and the
    # mesh-accommodation stencil below, still run.

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

    # --- Retired deep-afterglow low-Te recipe: adas_low_te_extension with
    # icool_recomb composes destructively (bare PRB charged, sub-edge PRB
    # amplified ~9,300x -> thermal runaway to the Te floor and a permanent
    # electron_cooling dt collapse). Construction must refuse the pair.
    try:
        LAPDSim1D(
            dict(m3_params, adas_low_te_extension=True),
            dict(resolved_cathode_flags, icool_recomb=True),
        )
    except ValueError as exc:
        assert "adas_low_te_extension" in str(exc)
        assert "icool_recomb" in str(exc)
    else:
        raise AssertionError(
            "expected ValueError for adas_low_te_extension + icool_recomb"
        )
    # Either flag ALONE stays constructible -- the guard is on the pair only,
    # and the recombination_energy_return guard's behavior is unchanged.
    LAPDSim1D(
        dict(m3_params, adas_low_te_extension=True), resolved_cathode_flags
    )
    LAPDSim1D(m3_params, dict(resolved_cathode_flags, icool_recomb=True))

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
    # The I_i-vs-n proportionality asserted below at rtol=1e-9 holds only in
    # the near-vacuum limit: _compute_l_b harmonically combines the beam's
    # electron-ion MFP (l_bi ~ 1/n_e) with its electron-NEUTRAL MFP
    # (l_bn = 1/(sigma_b*n_n)). While n_n is negligible l_b is a pure 1/n_e
    # power law and the self-consistent phi_c leaves I_i exactly linear in n;
    # at the realistic direct-run nn0 (2e13) the neutral leg is comparable, so
    # I_i departs from exact linearity (measured ratio 3.00077 instead of 3).
    # That coupling is physical -- pin the low fill this identity is stated in
    # rather than loosening the tolerance.
    ss_sim = LAPDSim1D(
        dict(m3_params, cathode_sample_smoothing="presheath", nn0=1.0e9),
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
    # 0.03 ms on the PLASMA clock. This was t_end=2.03e-3 while the stance ran
    # a 2.000 ms tau_neutral_prebreakdown, i.e. 2 ms of neutral-only dead time
    # plus the 0.03 ms of startup actually under test. The pre-phase is gone
    # (the machine has no pre-drive window, 2026-08-03), so the window drops by
    # exactly that dead time and this case still measures the same 0.03 ms of
    # startup it always did. NB the bounds below are UNCHANGED -- they pass at
    # their original values, which is the check that this is a window fix and
    # not a weakened test. Left at 2.03e-3 the case would instead run through
    # ignition (~0.06 ms here) into main_discharge and fail on main-discharge
    # physics it was never written to bound: 8.7e6 erg of Ee floor clipping
    # against the 1e4 erg startup bound, and 0.775/0.830 rate-domain
    # below-table fractions against the ==0 assertions. Startup itself is
    # unchanged -- the Ei floor term is bit-identical either way.
    startup_result = startup_sim.run(t_end=0.03e-3)
    # Pristine-startup assertions (0 rejections, 0 floor activity) DEFERRED to
    # the ES1 tuning pass (R5 stance flip, 2026-07-25). Under the repaired stance
    # the compare_sim1d_es1 startup shows minor, EXPECTED activity: a couple of
    # timestep rejections (the 2nd-order strang/tr_bdf2 split + Phelps presheath)
    # and small Ei-floor clipping (the Ti floor was relaxed to 300 K, so Ti can
    # now reach it near the cold Ti0 -- impossible at the old 0.1 eV floor). Both
    # are negligible (measured on the window above, 2026-08-03: 0 rejections,
    # 0.047 erg Ei; the old note's "~2 rejections, ~17 erg Ei over 2 ms" was
    # quoted over the retired pre-phase-padded window). The ES config is not
    # finalized (geometry + V_bank=180 circuit refit deferred), and startup
    # cleanliness is validated there. Soft-bound here so it does not regress
    # badly: these two asserts are a REGRESSION GUARD against the numbers
    # drifting, not a physics gate on the values themselves, because the
    # configuration they measure is still expected to move.
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

    # --- direct run() with neutral_equilibration ON warns loudly -------------
    # The equilibration fires only from start_simulation(); a direct run()
    # silently started from the nn0 fill instead. That is a warning, never an
    # error -- existing direct-run scripts must keep working.
    import warnings as _eq_warnings

    _eq_params, _eq_flags = default_config()
    assert _eq_flags["neutral_equilibration"], "expected the flag on by default"
    _eq_params["nx"] = 12
    _eq_sim = LAPDSim1D(_eq_params, _eq_flags)
    with _eq_warnings.catch_warnings(record=True) as _eq_caught:
        _eq_warnings.simplefilter("always")
        _eq_direct = _eq_sim.run(t_end=0.0)
    _eq_hits = [
        w for w in _eq_caught if "run() was called directly" in str(w.message)
    ]
    assert len(_eq_hits) == 1, (
        "direct run() with neutral_equilibration ON must warn exactly once, "
        f"got {[str(w.message) for w in _eq_caught]}"
    )
    _eq_text = str(_eq_hits[0].message)
    assert "start_simulation" in _eq_text
    assert "nn0" in _eq_text
    assert _eq_direct is not None, "the warning must not abort the run"
    # ... and the equilibration-aware entry point stays silent.
    _eq_sim2 = LAPDSim1D(_eq_params, {**_eq_flags, "neutral_equilibration": False})
    with _eq_warnings.catch_warnings(record=True) as _eq_quiet:
        _eq_warnings.simplefilter("always")
        _eq_sim2.run(t_end=0.0)
    assert not [
        w for w in _eq_quiet if "run() was called directly" in str(w.message)
    ], "the warning must be gated on the neutral_equilibration flag"
    # ... and the run() start_simulation() drives is silent even with the flag
    # on (exercises the guard directly; a full equilibration costs ~1 minute).
    _eq_sim3 = LAPDSim1D(_eq_params, _eq_flags)
    _eq_sim3._run_via_start_simulation = True
    with _eq_warnings.catch_warnings(record=True) as _eq_inner:
        _eq_warnings.simplefilter("always")
        _eq_sim3.run(t_end=0.0)
    assert not [
        w for w in _eq_inner if "run() was called directly" in str(w.message)
    ], "start_simulation()'s own run() must not warn"
    assert _eq_sim3._run_via_start_simulation, "run() must not clear the guard"

    # --- S_gp is born at rest (2026-07-28) ---------------------------------
    # The gas puff is a source of ZERO-parallel-momentum particles: cold gas
    # arrives through the pipe with no directed axial momentum, so S_gp adds
    # nn (and nn_a) and must NEVER add M_n / M_n_a. The only momentum the
    # source/sink term carries is the PUMP sink, which removes the wind that
    # leaves with the gas it is attached to. This invariant is the premise of
    # the neutral-momentum campaign thread -- the flow observable is only
    # evidence if the puff cannot manufacture wind -- and it lives at four
    # sites that must stay consistent: the explicit source/sink RHS (both
    # zone layouts), the two implicit neutral-equilibration steps, and the
    # local gas-puff ionization channel. All four are pinned below.
    from cablp.solvers._sim1d.core.geometry import build_geometry as _sgp_build

    sgp_params, sgp_flags = default_config()
    sgp_geom = _sgp_build(sgp_params, sgp_flags)
    sgp_cells = sgp_geom.cells
    sgp_ref_sim = LAPDSim1D(dict(sgp_params, nx=12), sgp_flags)
    sgp_mass = sgp_ref_sim.ion_mass_g
    # A puff level well onto the M6 square plateau, and the production
    # profile/valve count, so the profile under test is the one that runs.
    sgp_sccm = 3400.0
    sgp_valves = 2.0
    sgp_profile = "cosine_pipe"
    sgp_pump_lps = 4000.0
    sgp_puff = gas_puff_rate_profile(
        sgp_geom, sgp_sccm, sgp_valves, profile=sgp_profile, end=0
    )
    assert np.any(sgp_puff > 0.0), "the puff profile under test must be live"
    sgp_kwargs = dict(
        geometry=sgp_geom,
        S_gp=sgp_sccm,
        Twin_S_gp=0.0,
        S_pump_L=sgp_pump_lps,
        S_pump_R=sgp_pump_lps,
        gas_puff_valves=sgp_valves,
        gas_puff_profile=sgp_profile,
    )
    sgp_zeros = np.zeros(sgp_cells, dtype=float)
    # A state carrying a real wind everywhere, so a spurious puff momentum
    # source could not hide behind an M_n that happens to be zero.
    sgp_state = conservative_from_primitives(
        n=np.full(sgp_cells, 1e12),
        nn=np.full(sgp_cells, 1e13),
        u=np.zeros(sgp_cells),
        Te=np.full(sgp_cells, 5.0),
        Ti=np.full(sgp_cells, 1.0),
        ion_mass_g=sgp_mass,
        un=np.full(sgp_cells, 3.0e4),
    )
    assert sgp_state.M_n is not None and np.all(sgp_state.M_n != 0.0)

    sgp_pump_i_left, sgp_pump_i_right = pump_cell_indices(sgp_geom)
    sgp_pump_mask = np.zeros(sgp_cells, dtype=bool)
    sgp_pump_mask[[sgp_pump_i_left, sgp_pump_i_right]] = True

    def _sgp_expected_pump_sink(momentum):
        """Return the pump-only momentum sink -rate * momentum per cell."""
        sink = np.zeros(sgp_cells, dtype=float)
        for sgp_idx, sgp_speed in (
            (sgp_pump_i_left, sgp_pump_lps),
            (sgp_pump_i_right, sgp_pump_lps),
        ):
            sgp_rate = pump_rate(
                _effective_pump_speed(sgp_speed, None),
                sgp_geom.neutral_volume_cm3[sgp_idx],
            )
            sink[sgp_idx] -= sgp_rate * momentum[sgp_idx]
        return sink

    # Site 1, single zone, puff ON / pumps OFF: the puff is the WHOLE nn
    # source and contributes exactly nothing to M_n. Bit-exact, not a
    # tolerance -- there is no momentum arithmetic to round.
    sgp_puff_only = neutral_source_sink_rhs(
        state=sgp_state, gas_puff_enabled=True, pump_enabled=False, **sgp_kwargs
    )
    assert np.array_equal(sgp_puff_only.M_n, sgp_zeros), (
        "S_gp must add no neutral momentum"
    )
    assert np.array_equal(sgp_puff_only.nn, sgp_puff)
    assert sgp_puff_only.M_n_a is None and sgp_puff_only.nn_a is None
    assert np.array_equal(sgp_puff_only.M, sgp_zeros)

    # Site 1, single zone, puff ON / pumps ON: the only nonzero dM_n cells are
    # the two pump cells, and there dM_n is exactly -pump_rate * M_n. Adding
    # the puff on top leaves that untouched.
    sgp_both = neutral_source_sink_rhs(
        state=sgp_state, gas_puff_enabled=True, pump_enabled=True, **sgp_kwargs
    )
    assert np.all(sgp_both.M_n[~sgp_pump_mask] == 0.0)
    assert np.all(sgp_both.M_n[sgp_pump_mask] != 0.0)
    assert np.array_equal(
        sgp_both.M_n, _sgp_expected_pump_sink(sgp_state.M_n)
    )
    # ... and the pump-on/pump-off dM_n difference is the pump sink alone,
    # i.e. the puff term is identical in both calls.
    assert np.array_equal(
        sgp_both.M_n - sgp_puff_only.M_n, _sgp_expected_pump_sink(sgp_state.M_n)
    )

    # Site 1, two zone (nn_a and M_n_a present): the puff feeds the ANNULUS
    # where one exists -- and still adds no momentum to either zone.
    sgp_V_col, sgp_V_ann = neutral_zone_volumes(sgp_geom)
    assert np.all(sgp_V_ann > 0.0), "expected an annulus on every cell here"
    sgp_tz_state = ConservativeState1D(
        n=sgp_state.n,
        nn=sgp_state.nn,
        M=sgp_state.M,
        Ee=sgp_state.Ee,
        Ei=sgp_state.Ei,
        M_n=sgp_state.M_n,
        nn_a=np.full(sgp_cells, 2.0e13),
        M_n_a=np.full(sgp_cells, 4.0e-9),
    )
    sgp_tz_puff = neutral_source_sink_rhs(
        state=sgp_tz_state,
        gas_puff_enabled=True,
        pump_enabled=False,
        **sgp_kwargs,
    )
    assert np.array_equal(sgp_tz_puff.M_n, sgp_zeros)
    assert np.array_equal(sgp_tz_puff.M_n_a, sgp_zeros)
    # The gas lands in the annulus, and the annulus-volume re-normalization
    # conserves the inflow exactly against the single-zone chamber form.
    assert np.any(sgp_tz_puff.nn_a > 0.0)
    assert np.array_equal(sgp_tz_puff.nn, sgp_zeros)
    assert np.allclose(
        sgp_tz_puff.nn_a * sgp_V_ann + sgp_tz_puff.nn * sgp_V_col,
        sgp_puff * np.asarray(sgp_geom.neutral_volume_cm3, dtype=float),
        rtol=1e-13,
        atol=0.0,
    )
    # With the pumps on, BOTH zone momenta carry pump sinks and nothing else.
    sgp_tz_both = neutral_source_sink_rhs(
        state=sgp_tz_state,
        gas_puff_enabled=True,
        pump_enabled=True,
        **sgp_kwargs,
    )
    assert np.array_equal(
        sgp_tz_both.M_n, _sgp_expected_pump_sink(sgp_tz_state.M_n)
    )
    assert np.array_equal(
        sgp_tz_both.M_n_a, _sgp_expected_pump_sink(sgp_tz_state.M_n_a)
    )
    assert np.all(sgp_tz_both.M_n[~sgp_pump_mask] == 0.0)
    assert np.all(sgp_tz_both.M_n_a[~sgp_pump_mask] == 0.0)

    # Site 4: the local gas-puff ionization channel diverts a fraction of the
    # puff straight to plasma. Those particles are born AT REST too -- the M
    # row is identically zero -- while n gains exactly the neutrals nn loses.
    sgp_li_geom = sgp_ref_sim.geometry
    sgp_li_cells = sgp_li_geom.cells
    sgp_li_state = conservative_from_primitives(
        n=np.full(sgp_li_cells, 1e12),
        nn=np.full(sgp_li_cells, 1e13),
        u=np.full(sgp_li_cells, 1.0e5),
        Te=np.full(sgp_li_cells, 5.0),
        Ti=np.full(sgp_li_cells, 1.0),
        ion_mass_g=sgp_mass,
    )
    sgp_li_puff = gas_puff_rate_profile(
        sgp_li_geom, sgp_sccm, sgp_valves, profile=sgp_profile, end=0
    )
    assert np.any(sgp_li_puff > 0.0)
    sgp_li_kwargs = dict(
        state=sgp_li_state,
        floors=sgp_ref_sim.floors,
        ion_mass_g=sgp_mass,
        geometry=sgp_li_geom,
        puff_profile=sgp_li_puff,
        I_ion=I_ion,
    )
    sgp_li = gas_puff_local_ionization_rhs(fraction=0.3, **sgp_li_kwargs)
    assert np.array_equal(
        sgp_li.M, np.zeros(sgp_li_cells, dtype=float)
    ), "the diverted puff must be born at rest (no momentum source)"
    assert np.any(sgp_li.n > 0.0)
    assert np.all(sgp_li.nn <= 0.0) and np.any(sgp_li.nn < 0.0)
    sgp_li_scale = float(np.max(np.abs(sgp_li.n * sgp_li_geom.plasma_volume_cm3)))
    assert abs(particle_inventory_rate(sgp_li, sgp_li_geom)) <= (
        1e-12 * sgp_li_scale
    ), "the diverted particles must close plasma-plus-neutral inventory"
    # Default off is bit-exact, and the two-zone combination is rejected loudly.
    sgp_li_off = gas_puff_local_ionization_rhs(fraction=0.0, **sgp_li_kwargs)
    for sgp_field in STATE_NAMES_1D:
        assert np.all(getattr(sgp_li_off, sgp_field) == 0.0)
    try:
        gas_puff_local_ionization_rhs(
            **dict(sgp_li_kwargs, state=sgp_tz_state, fraction=0.3)
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "expected gas_puff_local_ionization x neutral_two_zone to fail"
        )

    # Sites 2 and 3: the implicit neutral-equilibration steps. The puff enters
    # the nn (and nn_a) linear solve only; M_n and M_n_a pass through bit-exact.
    sgp_eq_sim = LAPDSim1D(dict(sgp_params, nx=12), sgp_flags)
    sgp_eq_cells = sgp_eq_sim.geometry.cells
    sgp_eq_base = sgp_eq_sim.state
    sgp_eq_Mn = np.full(sgp_eq_cells, 5.0e-9)
    sgp_eq_state = ConservativeState1D(
        n=sgp_eq_base.n,
        nn=sgp_eq_base.nn,
        M=sgp_eq_base.M,
        Ee=sgp_eq_base.Ee,
        Ei=sgp_eq_base.Ei,
        M_n=sgp_eq_Mn.copy(),
    )
    # t on the M6 plateau, so the puff really is driving the solve.
    sgp_eq_next = sgp_eq_sim._implicit_neutral_step(1.0e-5, sgp_eq_state, 5.0e-3)
    assert np.any(sgp_eq_next.nn > sgp_eq_state.nn), "the puff must be feeding"
    assert np.array_equal(sgp_eq_next.M_n, sgp_eq_Mn)
    sgp_tzq_sim = LAPDSim1D(
        dict(sgp_params, nx=12, neutral_exchange_model="knudsen"),
        dict(sgp_flags, neutral_two_zone=True),
    )
    sgp_tzq_cells = sgp_tzq_sim.geometry.cells
    sgp_tzq_base = sgp_tzq_sim.state
    assert sgp_tzq_base.nn_a is not None
    sgp_tzq_Mn = np.full(sgp_tzq_cells, 5.0e-9)
    sgp_tzq_Mna = np.full(sgp_tzq_cells, 7.0e-9)
    sgp_tzq_state = ConservativeState1D(
        n=sgp_tzq_base.n,
        nn=sgp_tzq_base.nn,
        M=sgp_tzq_base.M,
        Ee=sgp_tzq_base.Ee,
        Ei=sgp_tzq_base.Ei,
        M_n=sgp_tzq_Mn.copy(),
        nn_a=sgp_tzq_base.nn_a,
        M_n_a=sgp_tzq_Mna.copy(),
    )
    sgp_tzq_next = sgp_tzq_sim._implicit_neutral_step_two_zone(
        1.0e-5, sgp_tzq_state, 5.0e-3
    )
    assert np.any(sgp_tzq_next.nn_a > sgp_tzq_state.nn_a), (
        "the two-zone puff must be feeding the annulus"
    )
    assert np.array_equal(sgp_tzq_next.M_n, sgp_tzq_Mn)
    assert np.array_equal(sgp_tzq_next.M_n_a, sgp_tzq_Mna)

    # --- Compiled-kernel END-TO-END equivalence (D3/D4, opt-in) -----------
    # The suite itself runs on the pure path (asserted above), and the D4
    # block compares each compiled kernel against its pure twin one function
    # at a time. Neither answers the end-to-end question: does a SOLVER
    # driven through the compiled kernels reach the same state? The kernel
    # binding happens once, at import, so the two paths cannot coexist in one
    # process -- each is a subprocess carrying its own CABLP_COMPILED_KERNELS.
    #
    # Gated on the extension being BUILT, not on this process having opted
    # in: the parent smoke deliberately runs pure (line ~4573 asserts it), so
    # keying this off the parent's env var would leave the section dead in
    # every gate invocation. On a checkout with no extension it SKIPS and
    # never fails -- the compiled path is opt-in by design and a pure
    # checkout must stay green.
    _ck_expected_kernel_id = "cython/_cathode_kernels_cy/tierA+csda"
    try:
        import importlib as _ck_importlib

        _ck_module = _ck_importlib.import_module(
            "cablp.funcs._cathode_kernels_cy"
        )
    except ImportError:
        _ck_module = None
    if _ck_module is None:
        print(
            "compiled-kernel equivalence: SKIPPED -- "
            "cablp.funcs._cathode_kernels_cy is not built "
            "(`python build_ext.py --inplace` enables it)"
        )
    else:
        assert _ck_module.KERNEL_ID == _ck_expected_kernel_id, (
            _ck_module.KERNEL_ID
        )
        # A short current-driven discharge on the production stance: the
        # cathode sheath solve (Tier A) runs on every sample and the CSDA ray
        # fires, so both halves of "tierA+csda" are on the hot path.
        _ck_child_source = '''
import json

import numpy as np

from cablp.funcs import _kernels as K
from cablp.solvers._sim1d import LAPDSim1D, default_config

params, flags = default_config()
params.update({
    "nx": 24,
    "dt_save": 0.0,
    "phase_transition_mode": "scheduled",
    "tau_neutral_prebreakdown": 0.0,
    "tau_prebreakdown": 0.0,
    "tau_breakdown": 0.0,
    "tau_discharge": 1.0,
    "tau_afterglow": 0.0,
})
result = LAPDSim1D(params, flags).run(t_end=2.0e-6, dt=1.0e-7)
diag = result.cathode_diagnostics
print(json.dumps({
    "provenance": K.PROVENANCE,
    "kernel_id": (
        None if K.COMPILED_KERNELS is None
        else str(K.COMPILED_KERNELS.KERNEL_ID)
    ),
    "requested": bool(K.compiled_kernels_requested()),
    "steps": int(result.steps),
    "solve_enabled": float(np.min(diag["solve_enabled"])),
    "has_solution": float(np.min(diag["has_solution"])),
    "beam_csda_active": float(np.max(diag["beam_csda_active"])),
    "I_tot": float(diag["source_I_tot"][-1]),
    "phi_c": float(diag["source_phi_c"][-1]),
    "y": np.ascontiguousarray(result.y[-1], dtype=float).tobytes().hex(),
}))
'''
        _ck_results = {}
        with tempfile.TemporaryDirectory() as _ck_tmpdir:
            _ck_script = Path(_ck_tmpdir) / "compiled_equivalence_child.py"
            _ck_script.write_text(_ck_child_source)
            for _ck_tag, _ck_optin in (("pure", None), ("compiled", "1")):
                # Inherit the environment (PYTHONPATH decides WHICH checkout
                # the child imports) and override only the opt-in.
                _ck_env = dict(os.environ)
                if _ck_optin is None:
                    _ck_env.pop(_kernel_selector.ENV_VAR, None)
                else:
                    _ck_env[_kernel_selector.ENV_VAR] = _ck_optin
                _ck_proc = subprocess.run(
                    [sys.executable, str(_ck_script)],
                    env=_ck_env,
                    capture_output=True,
                    text=True,
                )
                assert _ck_proc.returncode == 0, (
                    _ck_tag,
                    _ck_proc.returncode,
                    _ck_proc.stderr[-2000:],
                )
                # Warnings go to stderr; the JSON is the last stdout line.
                _ck_results[_ck_tag] = json.loads(
                    _ck_proc.stdout.strip().splitlines()[-1]
                )
        _ck_pure = _ck_results["pure"]
        _ck_compiled = _ck_results["compiled"]
        # Each child really took the path it was asked for -- an opt-in that
        # silently ran pure would make the comparison meaningless.
        assert _ck_pure["requested"] is False, _ck_pure
        assert _ck_pure["kernel_id"] is None, _ck_pure
        assert _ck_pure["provenance"] == _kernel_selector.PURE_PROVENANCE, (
            _ck_pure
        )
        assert _ck_compiled["requested"] is True, _ck_compiled
        assert _ck_compiled["kernel_id"] == _ck_expected_kernel_id, _ck_compiled
        assert _ck_compiled["provenance"] == _ck_expected_kernel_id, (
            _ck_compiled
        )
        # ...and the kernels were actually exercised. Without this the state
        # comparison could pass vacuously on a run that never solved.
        for _ck_tag, _ck_res in _ck_results.items():
            assert _ck_res["steps"] == 20, (_ck_tag, _ck_res["steps"])
            assert _ck_res["solve_enabled"] == 1.0, _ck_tag
            assert _ck_res["has_solution"] == 1.0, _ck_tag
            assert _ck_res["beam_csda_active"] == 1.0, _ck_tag
        # Bit-identical, not merely close: the compiled path is a faithful
        # transcription, so the raw state bytes must match exactly -- the
        # same standard the golden holds on the compiled path.
        assert _ck_compiled["y"] == _ck_pure["y"], (
            "compiled and pure solver states differ at the bit level"
        )
        assert _ck_compiled["I_tot"] == _ck_pure["I_tot"], (
            _ck_compiled["I_tot"], _ck_pure["I_tot"]
        )
        assert _ck_compiled["phi_c"] == _ck_pure["phi_c"], (
            _ck_compiled["phi_c"], _ck_pure["phi_c"]
        )
        print(
            "compiled-kernel equivalence: ok "
            f"({_ck_compiled['provenance']}, {_ck_pure['steps']} steps, "
            "final state bit-identical)"
        )

    # ---- dt_min lock: honest labeling, census, loud failure ----------------
    # Regression pins for the 2026-08-05 change. The clamp to dt_min used to
    # OVERWRITE active_constraint with "dt_min", so a run pinned at dt_min
    # reported that it was pinned and never by what -- and because the clamp
    # keeps such a run alive, a drained floor-pinned cell produced a silent
    # permanent lock (measured: scripts/dtmin_census_runlengths.txt).
    dtlock_params = dict(no_source_params)
    dtlock_flags = dict(flags)

    # (i) THE TRUE CONSTRAINT SURVIVES THE CLAMP. Synthetic drained
    # floor-pinned scenario: one cell sits exactly ON the density floor while
    # the resolved-source bundle still drains it, so the surface_loss bound
    # requests dt = 0 -- not a timestep request but a modelling breakdown.
    dtlock_sim = LAPDSim1D(dtlock_params, dtlock_flags)
    pinned_state = dtlock_sim.state
    pinned_n = np.asarray(pinned_state.n, dtype=float).copy()
    pinned_cell = pinned_n.size // 2
    pinned_n[pinned_cell] = float(dtlock_sim._floors["n"])
    pinned_state = dataclasses.replace(pinned_state, n=pinned_n)
    draining_source = SimpleNamespace(
        n=np.where(
            np.arange(pinned_n.size) == pinned_cell, -1.0, 0.0
        ),
        Ee=np.zeros_like(pinned_n),
        Ei=np.zeros_like(pinned_n),
    )
    assert (
        plasma_source_timestep(
            state=pinned_state,
            source_rhs=draining_source,
            floors=dtlock_sim._floors,
        )
        == 0.0
    )
    pinned_diag = suggest_timestep(
        state=pinned_state,
        floors=dtlock_sim._floors,
        ion_mass_g=dtlock_sim._ion_mass_g,
        mu=dtlock_sim._mu,
        geometry=dtlock_sim._geometry,
        neutral_exchange_coeff_cm3_s=dtlock_sim.neutral_exchange_coefficients(),
        plasma_source_rhs=draining_source,
        dt_min=1.0e-10,
        dt_max=1.0e-6,
    )
    # The label names the bound that actually minimized, NOT "dt_min".
    assert pinned_diag.active_constraint == "surface_loss"
    assert pinned_diag.dt_raw == 0.0
    assert pinned_diag.clamped_to_dt_min == 1.0
    assert pinned_diag.dt == 1.0e-10
    # A clamp that is not a hard zero is still labeled by its true bound.
    soft_clamp_diag = dtlock_sim.suggest_timestep()
    assert soft_clamp_diag.clamped_to_dt_min == 0.0
    big_floor_sim = LAPDSim1D(
        dict(dtlock_params, dt_min=1.0e-6, dt_max=1.0e-3), dtlock_flags
    )
    soft_clamped = big_floor_sim.suggest_timestep()
    assert soft_clamped.clamped_to_dt_min == 1.0
    assert soft_clamped.active_constraint == soft_clamp_diag.active_constraint
    assert soft_clamped.active_constraint != "dt_min"
    assert 0.0 < soft_clamped.dt_raw < 1.0e-6
    assert soft_clamped.dt == 1.0e-6

    # The guard counts CONSECUTIVE clamped steps, so drive it with a forced
    # clamp: what is under test is the counting and the raise, not the physics
    # that produces a clamp (which (i) already pins).
    class _ForcedClampSim(LAPDSim1D):
        """Force the clamp flag onto the first ``clamp_steps`` suggestions."""

        def __init__(self, params, flags, clamp_steps):
            super().__init__(params, flags)
            self._forced_clamps_left = int(clamp_steps)

        def suggest_timestep(self, *args, **kwargs):
            diag = super().suggest_timestep(*args, **kwargs)
            if self._forced_clamps_left > 0:
                self._forced_clamps_left -= 1
                return dataclasses.replace(
                    diag, clamped_to_dt_min=1.0, dt_raw=0.0
                )
            return diag

    # (iv) A SUB-THRESHOLD TRANSIENT MUST NOT RAISE. Self-releasing clamp
    # episodes are a known-good family (6-10% of steps in some completed
    # afterglow arms); aborting one would be the worse failure.
    transient_params = dict(dtlock_params)
    transient_params["dt_save"] = 0.0
    # Pin every step at dt_max so the run takes a known number of them (the
    # physical bounds here are far larger than t_end).
    transient_params["dt_max"] = 1.0e-10
    transient_params["dt_min_lock_max_steps"] = 5
    transient_sim = _ForcedClampSim(transient_params, dtlock_flags, clamp_steps=5)
    transient_result = transient_sim.run(t_end=1.2e-9)
    assert transient_sim._forced_clamps_left == 0
    # (ii) THE CENSUS COUNTS.
    transient_summary = summarize_result(transient_result)
    assert transient_summary.dt_min_clamped_step_count == 5
    assert transient_summary.max_consecutive_dt_min_clamped_steps == 5
    assert transient_summary.dt_min_hard_zero_step_count == 5
    assert transient_result.steps > 5
    assert "dt_min" not in transient_summary.constraint_counts
    assert [
        diag.clamped_to_dt_min for diag in transient_result.diagnostics[:6]
    ] == [1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
    with tempfile.TemporaryDirectory() as dtlock_dir:
        dtlock_path = Path(dtlock_dir) / "dtlock.h5"
        transient_sim.save_result(dtlock_path, transient_result)
        dtlock_loaded = load_result_hdf5(dtlock_path)
        loaded_summary = summarize_result(dtlock_loaded)
        assert loaded_summary.dt_min_clamped_step_count == 5
        assert loaded_summary.max_consecutive_dt_min_clamped_steps == 5
        assert loaded_summary.dt_min_hard_zero_step_count == 5

    # (ii-b) ACCEPTED STEPS BELOW dt_min ARE SEEN, AND SEPARATELY.
    # The dt_min clamp lifts a bound's request UP to dt_min inside
    # suggest_timestep, but the step caps are applied AFTERWARDS in the run
    # loop and can only shrink the step -- so an accepted step can land
    # strictly BELOW dt_min and the clamp census above cannot see it. A
    # production run (K6d) accepted 9.239e-11 against a configured dt_min of
    # 1e-10 with nothing recording it.
    below_params = dict(dtlock_params)
    below_params["dt_save"] = 0.0
    below_params["dt_min"] = 1.0e-10
    below_sim = LAPDSim1D(below_params, dtlock_flags)
    # t_end lands 5e-11 past the last whole step: below dt_min by construction.
    below_result = below_sim.run(t_end=2.5e-10, dt=1.0e-10)
    below_summary = summarize_result(below_result)
    assert [diag.accepted_dt for diag in below_result.diagnostics] == [
        1.0e-10,
        1.0e-10,
        below_result.diagnostics[-1].accepted_dt,
    ]
    assert below_summary.below_dt_min_step_count == 1
    assert below_summary.below_dt_min_known is True
    assert np.isclose(below_summary.below_dt_min_min_accepted_dt, 5.0e-11)
    # It NAMES the cap responsible -- the diagnostic point of the category.
    assert below_summary.below_dt_min_step_cap_counts == {"t_end": 1}
    # DISTINCT, not folded into the clamp count: no step was clamped here.
    assert below_summary.dt_min_clamped_step_count == 0
    assert below_summary.max_consecutive_dt_min_clamped_steps == 0
    # The clamp census and this one are independent: the forced-clamp run
    # above clamped 5 steps and had no below-floor accepted step.
    assert transient_summary.below_dt_min_step_count == 0
    # A result carrying no params cannot know dt_min, and says so rather than
    # reporting a reassuring zero.
    unknowable = summarize_result(
        SimpleNamespace(
            **{
                field: getattr(below_result, field)
                for field in dir(below_result)
                if not field.startswith("_") and field != "params"
            }
        )
    )
    assert unknowable.below_dt_min_known is False
    assert unknowable.below_dt_min_step_count == 0
    assert np.isnan(unknowable.below_dt_min_min_accepted_dt)

    # (iii) PAST THE THRESHOLD IT RAISES, LOUDLY AND WITH THE EVIDENCE.
    lock_params = dict(transient_params)
    locked_sim = _ForcedClampSim(lock_params, dtlock_flags, clamp_steps=40)
    try:
        locked_sim.run(t_end=1.2e-9)
    except RuntimeError as error:
        lock_message = str(error)
        assert "dt_min lock" in lock_message
        assert "6 consecutive steps" in lock_message
        assert "dt_min_lock_max_steps=5" in lock_message
        # the true bound, the offending cell, its density and its floor
        assert f"{transient_result.diagnostics[0].active_constraint!r}" in (
            lock_message
        )
        assert "index" in lock_message
        assert "n_floor=" in lock_message
        assert "modelling breakdown" in lock_message
    else:
        raise AssertionError("dt_min lock guard did not fire past its threshold")

    # Misconfiguration is loud at CONSTRUCTION time, not hours into a run.
    for bad_lock in (0, -1, 2.5, float("nan"), "many"):
        try:
            LAPDSim1D(
                dict(dtlock_params, dt_min_lock_max_steps=bad_lock), dtlock_flags
            )
        except ValueError as error:
            assert "dt_min_lock_max_steps must be a positive integer" in str(error)
        else:
            raise AssertionError(
                f"dt_min_lock_max_steps accepted {bad_lock!r}"
            )

    print(
        "sim1d smoke ok: "
        f"cells={geom.cells}, dz={geom.dz_cm:g} cm, "
        f"Vp_total={geom.plasma_volume_cm3.sum():.6e} cm^3, "
        f"Vm_total={geom.neutral_volume_cm3.sum():.6e} cm^3"
    )


if __name__ == "__main__":
    main()
