"""Smoke cases: the axial grid, area profiles, obstructions and the mirror-
field loader.
"""

import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

import h5py
import numpy as np

from cablp.constants import ev_to_erg, qe_SI
from cablp.solvers._sim1d import (
    LAPDSim1D,
    default_config,
    load_result_hdf5,
    summarize_result,
)
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
from cablp.solvers._sim1d.core.state import (
    ConservativeState1D,
    conservative_from_primitives,
    derive_state,
    pack_state,
)
from cablp.solvers._sim1d.physics.cathode import (
    beam_launch,
    cathode_sample_indices,
)
from cablp.solvers._sim1d.physics.kinetic_dvm import TransientDVM
from cablp.solvers._sim1d.physics.neutrals import (
    _effective_pump_speed,
    neutral_exchange_coefficients,
    neutral_thermal_speed,
    neutral_zone_volumes,
    two_zone_knudsen_coefficients,
)
from cablp.solvers._sim1d.physics.reactions import particle_inventory_rate
from cablp.solvers._sim1d.physics.sources import velocity_divergence

from ._harness import (
    _base_config,
    _base_sim,
    _case,
    _resolved_cathode_flags,
    _resolved_config,
    _resolved_geometry,
)


# --------------------------------------------------------------------
# shipped-defaults-and-base-geometry
# --------------------------------------------------------------------
@_case(
    "shipped-defaults-and-base-geometry",
    historical_stance=True,
    provides=("anode_face", "cathode_face"),
)
def _case_shipped_defaults_and_base_geometry():
    params, flags = default_config()
    # HELIUM MASS PIN (2026-08-21 unification). A LITERAL, deliberately: the
    # repo had carried three hand-made helium-mass products differing by up to
    # 0.9 ppm, none of them citable, and nothing caught it. The compiled
    # kernels never read a helium mass, so there is no .pyx constant guard to
    # extend -- this assertion is the guard. The value is Ar(4He)*u =
    # 4.00260325413 * 1.66053906892e-27 kg (CODATA 2022), cross-checked
    # against m(alpha) + 2 m_e - 79.005151 eV/c^2 to 5e-12 relative.
    from cablp.constants import m_He_SI as _m_He_SI
    from cablp.constants import m_He_cgs as _m_He_cgs

    assert _m_He_cgs == 6.6464790809e-24, _m_He_cgs
    assert _m_He_SI == 6.6464790809e-27, _m_He_SI
    assert LAPDSim1D(params, flags).ion_mass_g == 6.6464790809e-24
    assert params["cycles"] == 1
    assert params["phase_transition_mode"] == "current"
    # No pre-drive window: the machine fires one global trigger, so the bank
    # connects as the puff starts and neutrals never accumulate with the drive
    # withheld (2026-08-03; see timing_defaults). Asserted EXACTLY, not as
    # ">= 0", so a reintroduced pre-phase fails here. The duration alone opts
    # back in, which is what the dedicated neutral_prebreakdown block below
    # does (it pins its own tau_neutral_prebreakdown, so that feature test does
    # not read this default).
    assert params["tau_neutral_prebreakdown"] == 0.0
    # Everything below runs on the historical operator-algebra stance, which
    # the shared fixture pins (the quiescent-zero and operator-algebra
    # invariants hold only there).
    params, flags = _base_config()
    sim, snapshot = _base_sim()
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
    assert geom.cell_role[-1] == "end_wall"
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
    resolved_params, resolved_flags = _resolved_config()
    resolved_geom = _resolved_geometry()
    assert resolved_geom.cells > resolved_params["nx"] + 2
    assert np.all(resolved_geom.plasma_volume_cm3 > 0.0)
    assert np.all(resolved_geom.neutral_volume_cm3 > resolved_geom.plasma_volume_cm3)
    assert {"plenum", "cathode", "gap", "puff", "column", "end_wall"} <= set(
        resolved_geom.cell_role
    )
    assert list(resolved_geom.cell_role[:2]) == ["plenum", "cathode"]
    assert resolved_geom.cell_role[-1] == "end_wall"
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
    # Two cell counts: nx_gap across the gap, nx from the anode to the end wall.
    assert anode_face - cathode_face == resolved_params["nx_gap"]
    gap_dz = resolved_params["cathode_anode_gap_cm"] / resolved_params["nx_gap"]
    assert np.allclose(
        resolved_geom.length_cm[cathode_face:anode_face], gap_dz
    )
    # The smallest cell in the mesh sets the explicit CFL, and it is either a
    # gap cell or the end wall block -- every other segment (plenum, fixed
    # source region, far column) is longer than both on any shipped geometry.
    # Which of the two wins is a property of the machine, not of the mesher:
    # on the nominal 100 cm end wall it is the gap, and on the G1 measured
    # end wall (7.8 cm, a config default since the R2a fold-in) it is the
    # end wall block.
    assert np.isclose(
        resolved_geom.length_cm.min(),
        min(gap_dz, resolved_params["end_wall_length_cm"]),
    )

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
    # The puff role sits on the cell CONTAINING gas_puff_z_cm, wherever the
    # mesh puts it: on the nominal machine that is the first column cell past
    # the anode face, and under source_fixed_grid with the G1 puff position
    # (86.3 cm, both config defaults since the R2a fold-in) it is a fixed
    # source cell further downstream. Checked as containment, so this states
    # the rule rather than one machine's answer to it.
    _puff_first, _puff_last = puff_cell_indices(resolved_geom)
    assert _puff_first == _puff_last
    assert (
        resolved_geom.z_edges_cm[_puff_first]
        <= resolved_params["gas_puff_z_cm"]
        <= resolved_geom.z_edges_cm[_puff_first + 1]
    )
    assert _puff_first >= anode_face
    assert np.all(np.isnan(resolved_geom.neutral_face_conductance_cm3_s))

    # G1: default-off expanded end geometry. The provisional hardware arm
    # resolves a 150 cm, Rm=100 cm end wall region in ten cells. Plasma area
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
                Tn_K=resolved_params["Tn_K"],
        mu_neutral=4.0,
        clausing_scale=resolved_params["neutral_clausing_scale"],
    )
    baffle_single = neutral_exchange_coefficients(
        geometry=baffle_geom,
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
    return locals()


# --------------------------------------------------------------------
# source-fixed-grid
# --------------------------------------------------------------------
@_case(
    "source-fixed-grid",
    historical_stance=True,
    provides=(
        "expansion_geom", "expansion_sim", "srcgrid_off_flags",
        "srcgrid_off_params",
    ),
)
def _case_source_fixed_grid():
    # Fixed-cell-size source region (``source_fixed_grid``). Without it, nx
    # uniform column cells span anode face to end wall start, so a refinement
    # study moves every near-source cell edge -- including the puff cell, whose
    # centre anchors the default cosine puff profile. With it on the column from
    # the anode face to source_region_length_cm is meshed at exactly
    # source_region_dz_cm regardless of nx, and the puff role follows
    # gas_puff_z_cm.
    #
    # The gap, the region end and the puff position are PINNED below rather
    # than inherited. They were inherited until the 2026-08-24 CAD-span gap
    # adoption moved ``cathode_anode_gap_cm`` 50.0 -> 53.25, which changes the
    # source cell size (the span must stay a whole number of
    # source_region_dz_cm) and moves the puff into the first source cell --
    # neither of which this case is about. It exercises the MESHER, which is
    # gap-agnostic, so it now states the round geometry its hard-coded edge
    # positions below describe.
    #
    # (d) The OFF path takes no new branch: with the flag cleared and both keys
    # None the spec helper returns None. Since the R2a fold-in the flag and both
    # values are config defaults, so the off arm is constructed here rather than
    # read off default_config().
    resolved_params, resolved_flags = _resolved_config()
    resolved_geom = _resolved_geometry()
    srcgrid_off_flags = {**resolved_flags, "source_fixed_grid": False}
    srcgrid_off_params = dict(
        resolved_params,
        cathode_anode_gap_cm=50.0,
        source_region_length_cm=None,
        source_region_dz_cm=None,
    )
    assert (
        _source_fixed_grid_spec(
            srcgrid_off_params,
            srcgrid_off_flags,
            gap_length=srcgrid_off_params["cathode_anode_gap_cm"],
            total_length=srcgrid_off_params["Lm"],
            end_wall_length=srcgrid_off_params["end_wall_length_cm"],
            twin=False,
        )
        is None
    )
    srcgrid_off_geom = (
        LAPDSim1D(srcgrid_off_params, srcgrid_off_flags)
        .get_initial_snapshot()
        .geometry
    )

    srcgrid_flags = {**resolved_flags, "source_fixed_grid": True}

    def _srcgrid_params(nx):
        params = dict(resolved_params)
        params.update(
            {
                "nx": nx,
                "cathode_anode_gap_cm": 50.0,
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
    # nx meshes only the far column, from the region end to the end wall.
    assert srcgrid_geom.cells == srcgrid_off_geom.cells + srcgrid_n_fixed
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
    srcgrid_twin_params["end_wall_length_cm"] = 100.0
    # A source region reaching PAST the end wall block start, derived from the
    # machine rather than hardcoded (the G1 end wall is 7.8 cm, so a fixed
    # 1900 cm would now be comfortably inside the column) and rounded up to a
    # whole number of source cells so the integer-multiple check cannot fire
    # first and mask the one this case is about.
    srcgrid_past_end_wall = float(
        _srcgrid_params(60)["cathode_anode_gap_cm"]
        + 10.0
        * np.ceil(
            (
                resolved_params["Lm"]
                - resolved_params["end_wall_length_cm"]
                - _srcgrid_params(60)["cathode_anode_gap_cm"]
            )
            / 10.0
        )
    )
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
            srcgrid_off_flags,
            "source region parameters require the source_fixed_grid flag",
        ),
        (
            {**srcgrid_off_params, "source_region_dz_cm": 10.0},
            srcgrid_off_flags,
            "source region parameters require the source_fixed_grid flag",
        ),
        (
            {**_srcgrid_params(60), "source_region_length_cm": 50.0},
            srcgrid_flags,
            "strictly beyond the anode face",
        ),
        (
            {
                **_srcgrid_params(60),
                "source_region_length_cm": srcgrid_past_end_wall,
            },
            srcgrid_flags,
            "strictly before the end wall",
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
            "end_wall_length_cm": 150.0,
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
            "implicit_heat_conduction": False,
        }
    )
    expansion_params["phase_transition_mode"] = "scheduled"
    expansion_params["tau_prebreakdown"] = 0.0
    expansion_params["tau_breakdown"] = 0.0
    expansion_sim = LAPDSim1D(expansion_params, expansion_flags)
    expansion_geom = expansion_sim.get_initial_snapshot().geometry
    end_cells = np.flatnonzero(
        np.isin(expansion_geom.cell_role, np.asarray(["end", "end_wall"]))
    )
    assert end_cells.size == 10
    assert np.array_equal(end_cells, np.arange(expansion_geom.cells - 10, expansion_geom.cells))
    assert list(expansion_geom.cell_role[-10:-1]) == ["end"] * 9
    assert expansion_geom.cell_role[-1] == "end_wall"
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
    return locals()


# --------------------------------------------------------------------
# variable-area-well-balancedness
# --------------------------------------------------------------------
@_case(
    "variable-area-well-balancedness",
    historical_stance=True,
)
def _case_variable_area_well_balancedness(
    anode_face, cathode_face, expansion_geom, expansion_sim,
    srcgrid_off_flags, srcgrid_off_params
):
    # Well-balancedness of the variable-area flux tube: for a uniform stationary
    # plasma the quasi-1D p*dA/dz geometric source cancels the area-weighted
    # pressure flux bit-for-bit -- but this property applies only across the
    # INTERIOR expansion cells (role "end"). The terminating "end_wall" cell is
    # a plasma-OPEN boundary: it carries a Bohm outflow (ghost u_g = c_s) whose
    # flux is supplied by characteristic_boundary_rhs (a term not summed here),
    # so a uniform stationary state is deliberately NOT its equilibrium -- the
    # plasma flows out. (hyperbolic_energy_consistent and hyperbolic_wave_speed
    # have no effect on this state: at u=0 with no gradients the KEP convective
    # term and the Rusanov dissipation both vanish at every interior face, so
    # only the end wall ghost can be nonzero.) The legacy reflecting-wall
    # alternative, under which the end wall cancelled like the interior, was
    # retired; see commit 1fc05c9.
    resolved_params, resolved_flags = _resolved_config()
    sim, snapshot = _base_sim()
    geom = snapshot.geometry
    resolved_geom = _resolved_geometry()
    uniform_expansion = conservative_from_primitives(
        n=np.full(expansion_geom.cells, 1.0e12),
        nn=np.full(expansion_geom.cells, 1.0e12),
        u=np.zeros(expansion_geom.cells),
        Te=np.full(expansion_geom.cells, 2.0),
        Ti=np.full(expansion_geom.cells, 1.0),
        ion_mass_g=expansion_sim.ion_mass_g,
    )
    expansion_advective = expansion_sim.plasma_flux_rhs_terms(
        state=uniform_expansion
    )["plasma_advective_flux"]
    expansion_geometric = expansion_sim.flux_tube_geometry_rhs(
        state=uniform_expansion
    )
    interior_expansion_cells = np.flatnonzero(expansion_geom.cell_role == "end")
    end_wall_cells = np.flatnonzero(expansion_geom.cell_role == "end_wall")
    assert interior_expansion_cells.size == 9
    assert end_wall_cells.size == 1
    expansion_momentum_residual = expansion_advective.M + expansion_geometric.M
    # Interior variable-area cells: exact cancellation (the load-bearing
    # well-balancedness of the KEP pressure flux against the flux-tube source).
    assert np.array_equal(
        expansion_momentum_residual[interior_expansion_cells],
        np.zeros(interior_expansion_cells.size),
    )
    # Terminating end wall cell: an open Bohm outflow, directed toward +z, so
    # a net POSITIVE momentum residual -- the load-bearing contrast against the
    # interior cells' exact cancellation asserted just above.
    assert np.all(expansion_momentum_residual[end_wall_cells] > 0.0)
    assert np.allclose(expansion_geometric.n, 0.0)
    assert np.allclose(expansion_geometric.Ee, 0.0)
    assert np.allclose(expansion_geometric.Ei, 0.0)
    expansion_attempt = expansion_sim._attempt_step(
        dt=1.0e-9, operator_split=False
    )
    assert np.all(np.isfinite(expansion_attempt.y))
    assert expansion_attempt.y.shape == expansion_sim.get_initial_snapshot().y.shape

    # Twin cathode mirrors the source end: its cathode
    # surface sits at z = Lm, with that plenum beyond it. It builds on the
    # source_fixed_grid OFF arm because mirroring the fixed source region onto
    # a second cathode end is not implemented and the geometry refuses the
    # pair (checked in the refusal table above); since the R2a fold-in that
    # flag is a config default, so the twin layout has to clear it explicitly.
    twin_resolved_flags = dict(srcgrid_off_flags)
    twin_resolved_flags["TwinCathode"] = True
    twin_resolved_flags["cathode_coupling"] = False
    twin_resolved_sim = LAPDSim1D(srcgrid_off_params, twin_resolved_flags)
    twin_resolved_geom = twin_resolved_sim.get_initial_snapshot().geometry
    # PRESENCE GATE, armed side. The ``end_*`` cathode-result block exists
    # exactly where a twin solve can fill it; the single-cathode cases assert
    # its absence. Both directions, so a gate that silently stopped seeding
    # the block cannot pass.
    twin_cathode_diag = twin_resolved_sim._cathode_diagnostic_snapshot()
    assert "end_regime" in twin_cathode_diag
    assert "end_phi_c" in twin_cathode_diag
    assert "end_long_mfp" in twin_cathode_diag
    assert "end_phi_c_at_cap" in twin_cathode_diag
    # PRESENCE GATE, armed side, for the BEAM END-FACE rows -- the second
    # per-end block keyed on the same helper. The twin marches the ``-1``
    # ray, so every one of these has a filler here; the single-cathode cases
    # assert their absence. Named one by one rather than derived from the
    # ``source_`` set, because that set also carries the cathode-result key
    # ``beam_bypass_fraction``, which is not a beam end-face row.
    for _twin_beam_row in (
        "beam_anode_intercepted_W",
        "beam_transmitted_W",
        "beam_transmitted_flux_per_s",
        "beam_end_loss_low_W",
        "beam_end_loss_high_W",
        "beam_end_loss_tail_low_W",
        "beam_end_loss_tail_high_W",
        "beam_gap_survival_probe",
        "beam_gap_survival_ray",
        "beam_gap_survival_circuit",
    ):
        assert f"source_{_twin_beam_row}" in twin_cathode_diag, _twin_beam_row
        assert f"end_{_twin_beam_row}" in twin_cathode_diag, _twin_beam_row
    # The plateau-edge pair rides the multi-group closure, so it is present
    # per end exactly when that closure is armed -- both gates at once.
    for _twin_mg_row in (
        "beam_plateau_edge_eV", "beam_plateau_edge_clamped",
    ):
        for _twin_prefix in ("source", "end"):
            assert (
                (f"{_twin_prefix}_{_twin_mg_row}" in twin_cathode_diag)
                is bool(twin_resolved_sim._plateau_multigroup)
            ), (_twin_prefix, _twin_mg_row)
    assert list(twin_resolved_geom.cell_role[:2]) == ["plenum", "cathode"]
    assert list(twin_resolved_geom.cell_role[-2:]) == ["cathode", "plenum"]
    assert "end_wall" not in set(twin_resolved_geom.cell_role)
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
    assert resolved_geom.cell_role[resolved_pump_right] == "end_wall"
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
    # The anode's plasma face is always fully open: the mesh removes plasma
    # through the Bohm sheath flux at its wires, so shrinking the face too
    # would remove the same particles twice.
    assert resolved_geom.plasma_transmission[anode_face] == 1.0

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

    # M4a: the cathode surface and end wall are absorbing Bohm faces.
    assert resolved_geom.plasma_absorbing[cathode_face]
    assert resolved_geom.plasma_absorbing[-1]  # end wall outer face
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
    absorbed = resolved_sim.characteristic_boundary_rhs(state=flowing_state)
    assert absorbed.n[cathode_face] < 0.0  # cathode cell drains to the surface
    assert absorbed.nn[cathode_face] > 0.0
    assert absorbed.n[-1] < 0.0  # end wall drains too
    # Momentum leaves at c_s directed INTO each surface: negative (toward -z) at
    # the cathode, positive (toward +z) at the end wall. This is what makes the
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
    assert resolved_geom.cell_role[resolved_end_index] == "end_wall"
    twin_source_index, twin_end_index = cathode_sample_indices(twin_resolved_geom)
    assert twin_resolved_geom.cell_role[twin_source_index] == "cathode"
    assert twin_resolved_geom.cell_role[twin_end_index] == "cathode"
    assert twin_source_index != twin_end_index

    # M4b: the beam launches from the cathode cell.
    assert beam_launch(resolved_geom, end=0) == (cathode_face, 1)

    # M5: the circuit's anode current is the same Bohm collection the fluid
    # removes, not `2*eta*I_i` scaled off the cathode cell.
    resolved_cathode_flags = _resolved_cathode_flags()
    # The anode current == fluid Bohm collection identity holds on the state
    # the solve samples. The electrode sample smoothing EMA-smooths the
    # anode-flank (n, Te) the solve reads, so it is re-seeded from the probe
    # state below, which makes the smoothed sample that state itself.
    m5_cathode_params = dict(resolved_params)
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
        nn_a=np.full(m5_geom.cells, 1.0e13),
    )
    m5_sim._set_state_vector(pack_state(m5_state))
    m5_sim._init_sample_smoothing()
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


# --------------------------------------------------------------------
# obstruction-geometry-production-style
# --------------------------------------------------------------------
@_case(
    "obstruction-geometry-production-style",
    historical_stance=True,
)
def _case_obstruction_geometry_production_style(kd_flags, kd_params):
    # Production-style geometry (Lcs = 25): an obstruction cell sits between
    # the plenum and the cathode, so the cathode's live cell is index 2 --
    # neither an end cell nor a fixed offset from one. The wall-return
    # channels must be READ from that cell and DEPOSITED into it; positional
    # constants read the plasma-dead cells behind it and source nothing.
    # This is the PRE-G1 production geometry, reconstructed key by key (the
    # fitted 15 cm radii, the plenum-choke obstruction and the built-in end
    # flare, none of which the measured machine uses). The R2a fold-in moved
    # the four machine scalars into the config defaults, so they are named here
    # with the rest of the arm rather than inherited -- the tiny G1 end wall
    # block would otherwise put a 7.8 cm cell under this block's fixed
    # dt = 1 ns steps.
    kd_obs_params = dict(kd_params)
    kd_obs_params.update(
        {
            "Lm": 2000.0,
            "plenum_length_cm": 100.0,
            "end_wall_length_cm": 100.0,
            "gas_puff_z_cm": 60.0,
            "Rp": 15.0,
            "R_cath": 15.0,
            "Rcs": 40.0,
            "Lcs": 25.0,
            "Rsup": 0.0,
            "end_expansion_cells": 10,
            "end_expansion_machine_radius_cm": 100.0,
            "end_expansion_plasma_radius_cm": 15.0,
            "cathode_anode_gap_cm": 50.0,
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
    assert kd_obs_roles[kd_obs_coll] == "end_wall"
    assert absorbing_live_cells_by_role(kd_obs_sim.geometry) == {
        "cathode": (kd_obs_cath,),
        "end_wall": (kd_obs_coll,),
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
    # Recycled == removed, per face, against the boundary term that actually
    # runs -- the single plasma-terminating operator.
    kd_obs_removed = -np.asarray(
        kd_obs_sim.characteristic_boundary_rhs(state=kd_obs_sim.state).n,
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
    kd_lim_bundle = kd_lim_sim._plasma_source_timestep_rhs(
        state=kd_lim_sim.state, time=kd_lim_sim.time
    )
    assert np.all(np.isfinite(np.asarray(kd_lim_bundle.Ei, dtype=float)))

    # K2d transfer-ledger census, PERSISTED: the standing DVM report condition
    # ("quote relax_limited_steps and the outstanding debt; locate any
    # limited > 0 in the record -- a conducting-phase source-cell event is
    # the alarm, an afterglow far-column event is the bounded electron-ion
    # exchange regime") has to be answerable from the saved artifact,
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
    # NEVER LIMITED so far: the limiter's per-event rows are PRESENT and
    # EMPTY, so a reader never has to tell "the limiter never fired" from
    # "the amount was not recorded".
    assert kd_cen_sim._dvm.relax_limited_steps == 0
    kd_cen_empty = kd_cen_sim._dvm_limited_step_census()
    assert kd_cen_empty["limited_step_index"].shape == (0,)
    assert kd_cen_empty["limited_step_time_s"].shape == (0,)
    assert kd_cen_empty["limited_step_clamped_fraction_max"].shape == (0,)
    assert kd_cen_empty["limited_step_clamped_fraction_cells"].shape == (
        0, kd_cen_cells
    )
    assert kd_cen_empty["limited_clamped_fraction_max"] == 0.0
    assert kd_cen_empty["limited_clamped_fraction_mean"] == 0.0
    assert kd_cen_empty["limited_steps_dropped"] == 0
    kd_cen_sim._dvm.Ei_transfer = np.full(kd_cen_cells, -1.0e12)
    kd_cen_sim._dvm.M_transfer = np.full(kd_cen_cells, -1.0e3)
    kd_cen_result = kd_cen_sim.run(
        t_end=kd_cen_sim.time + 4.0e-8, dt=1.0e-9
    )
    kd_cen = kd_cen_result.dvm_transfer_ledger
    assert kd_cen["engaged"] == 1
    assert kd_cen["relax_limited_steps"] > 0
    assert kd_cen["limited_cells"] > 0
    # THE AMOUNT, beside the count. One entry per limited step: where in the
    # ledger it fell, when, the largest share of the desired transfer any
    # cell had withheld, and the per-cell profile of that share.
    assert kd_cen["limited_steps_dropped"] == 0
    assert (
        kd_cen["limited_steps_recorded"] == kd_cen["relax_limited_steps"]
    )
    assert (
        kd_cen["limited_steps_recorded"] < kd_cen["limited_steps_record_cap"]
    )
    kd_cen_frac = kd_cen["limited_step_clamped_fraction_cells"]
    assert kd_cen_frac.shape == (kd_cen["limited_steps_recorded"], kd_cen_cells)
    assert np.all(kd_cen_frac >= 0.0)
    assert np.all(kd_cen_frac <= 1.0)
    kd_cen_peak = kd_cen["limited_step_clamped_fraction_max"]
    assert np.array_equal(kd_cen_peak, kd_cen_frac.max(axis=1))
    assert np.all(kd_cen_peak > 0.0)
    # The cells the entry names are exactly the cells the limiter bound in,
    # so a look can go straight to them.
    assert np.all(kd_cen_frac.max(axis=0) > 0.0) == np.all(
        kd_cen["relax_cell_steps"] > 0.0
    )
    # Placed in the ledger and on the clock, both monotone.
    assert np.all(np.diff(kd_cen["limited_step_index"]) > 0.0)
    assert np.all(np.diff(kd_cen["limited_step_time_s"]) >= 0.0)
    # The quotable summary, over EVERY limited step.
    assert kd_cen["limited_clamped_fraction_max"] == float(np.max(kd_cen_peak))
    # The running sum accumulates in step order; ``np.mean`` sums pairwise,
    # so the two agree to rounding rather than to the bit.
    assert np.isclose(
        kd_cen["limited_clamped_fraction_mean"],
        float(np.mean(kd_cen_peak)),
        rtol=1.0e-12,
        atol=0.0,
    )
    assert 0.0 < kd_cen["limited_clamped_fraction_mean"] <= 1.0

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
        # The CX/elastic pair's two targets and the rate they were formed
        # with, per cell at every frame: nothing else records them and the
        # trajectory cannot recover them. Read AT the frame, so the last one
        # is the live solver's own array.
        for kd_cen_pc, kd_cen_live in (
            ("T_eff_eV", kd_cen_sim._dvm.T_eff_eV),
            ("u_n_eff", kd_cen_sim._dvm.u_n_eff),
            ("Ei_transfer_pair", kd_cen_sim._dvm.Ei_transfer_pair),
        ):
            kd_cen_rows = kd_cen_back[f"sample_{kd_cen_pc}"]
            assert kd_cen_rows.shape == (kd_cen_frames, kd_cen_cells), kd_cen_pc
            assert np.all(np.isfinite(kd_cen_rows)), kd_cen_pc
            assert np.array_equal(kd_cen_rows[-1], kd_cen_live), kd_cen_pc
        # A temperature is non-negative by construction (a second moment
        # about the ion drift); the pair rate is signed.
        assert np.all(kd_cen_back["sample_T_eff_eV"] >= 0.0)
        assert np.any(kd_cen_back["sample_T_eff_eV"] > 0.0)
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


# --------------------------------------------------------------------
# anode-disc-radius
# --------------------------------------------------------------------
@_case("anode-disc-radius")
def _case_anode_disc_radius(build_geometry):
    # --- Anode disc radius (anode_radius_cm): opens the annulus around the
    # mesh to neutrals only. None = historical (1 - eta); Ra < Rm gives
    # 1 - eta*(Ra/Rm)^2; heat/Bohm keep the bare mesh values.
    disc_params, disc_flags = default_config()
    disc_params.update({"Rp": 15.0, "anode_radius_cm": 40.0})
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
    except ValueError as disc_error:
        assert "anode_radius_cm must satisfy Rp <= Ra <= Rm" in str(
            disc_error
        ), str(disc_error)
    else:
        raise AssertionError("expected ValueError for anode disc inside Rp")


# --------------------------------------------------------------------
# prescribed-area-geometry
# --------------------------------------------------------------------
@_case(
    "prescribed-area-geometry",
    provides=(
        "_pa_Rm", "_pa_Rp", "_pa_base_p", "_pa_base_result",
        "_pa_cells", "_pa_geom0", "_pa_raw", "_pa_stance",
    ),
)
def _case_prescribed_area_geometry():
    # ---- pa: prescribed per-cell flux-tube / vessel geometry (default off) --
    # The capability replaces the uniform scalars Rp and Rm with per-cell
    # radius vectors computed OUTSIDE the solver, so the plasma areas, the cell
    # volumes, the face areas, the neutral conductances and the two-zone
    # annulus volume all follow a prescribed A(z) inside a prescribed bore; the
    # quasi-1D mirror force and the area-consistent pressure work come on with
    # it. Seven questions decide it: is the OFF path untouched, is a CONSTANT
    # profile pair bit-identical to no profile at all, is the mirror force
    # WELL-BALANCED (a static uniform-pressure plasma on a strongly varying
    # A(z) generates exactly no momentum), do both energy equations see the
    # same area, does a STEPPED bore land where it was asked to, do the
    # sliver-annulus guard and its declared cap behave as stated, and does
    # every misconfiguration raise.
    def _pa_stance(**over):
        params, flags = default_config()
        params.update({
            "nx": 12,
            "dt_save": 0.0,
            "phase_transition_mode": "scheduled",
            "tau_neutral_prebreakdown": 0.0,
            "tau_prebreakdown": 0.0,
            "tau_breakdown": 0.0,
            "tau_discharge": 1.0,
            "tau_afterglow": 0.0,
        })
        # The cached equilibrated seed is keyed on the geometry (a prescribed
        # profile re-keys it by design), so the comparison arms clear it and
        # the identity below is about the profile alone.
        params["initial_neutral_state"] = "fill"
        params.update(over)
        return params, flags

    _pa_base_p, _pa_base_f = _pa_stance()
    _pa_base_sim = LAPDSim1D(dict(_pa_base_p), dict(_pa_base_f))
    _pa_geom0 = _pa_base_sim.geometry
    _pa_cells = int(_pa_geom0.cells)
    _pa_Rp = np.asarray(_pa_geom0.Rp_cm, dtype=float)
    _pa_Rm = np.asarray(_pa_geom0.Rm_cm, dtype=float)

    def _pa_raw(result):
        return [
            np.ascontiguousarray(y, dtype=float).view(np.uint64).tobytes()
            for y in result.y
        ]

    # (a) PRESENCE GATE. With the flag off nothing is read, the column is the
    # uniform pi*Rp^2 inside the uniform Rm, and the quasi-1D geometric
    # momentum source is not even in the term ledger -- which is what keeps the
    # golden bit-exact. Naming the new keys at their off values must also
    # change nothing.
    assert not _pa_base_sim._variable_area_geometry
    assert "flux_tube_geometry" not in _pa_base_sim.rhs_terms()
    assert np.array_equal(_pa_geom0.plasma_area_cm2, np.pi * _pa_Rp**2)
    assert np.all(_pa_Rp == _pa_Rp[0]) and np.all(_pa_Rm == _pa_Rm[0])
    _pa_base_result = LAPDSim1D(
        dict(_pa_base_p), dict(_pa_base_f)
    ).run(t_end=1.0e-6, dt=1.0e-7)
    _pa_null_p, _pa_null_f = _pa_stance(
        plasma_radius_profile_cm=None,
        machine_radius_profile_cm=None,
        plasma_area_max_vessel_fraction=None,
    )
    _pa_null_f["prescribed_area_geometry"] = False
    _pa_null_result = LAPDSim1D(
        _pa_null_p, _pa_null_f
    ).run(t_end=1.0e-6, dt=1.0e-7)
    assert _pa_base_result.steps == _pa_null_result.steps > 0, (
        _pa_base_result.steps, _pa_null_result.steps
    )
    assert _pa_raw(_pa_base_result) == _pa_raw(_pa_null_result)
    return locals()


# --------------------------------------------------------------------
# prescribed-area-trivial-profile-identity
# --------------------------------------------------------------------
@_case("prescribed-area-trivial-profile-identity")
def _case_prescribed_area_trivial_profile_identity(
    _pa_Rm, _pa_Rp, _pa_base_result, _pa_cells, _pa_geom0, _pa_raw,
    _pa_stance
):
    # (b) THE TRIVIAL-PROFILE IDENTITY. Profiles holding the geometry's own
    # uniform Rp and Rm in every cell reproduce the no-profile run at the RAW
    # BIT level: each area is rebuilt with the same pi*R**2 expression the
    # uniform path uses, so the two reach the state through identical
    # arithmetic, and the geometric source the flag switches on is identically
    # +0.0, which cannot perturb the term sum. This is why the parameters are
    # RADII and not areas -- pi*r^2 does not round-trip through sqrt(A/pi) for
    # every r (18.415 does, 3.7 does not), so an area vector could only claim
    # the identity for lucky values. A ceiling that never binds is part of the
    # identity too: it re-derives nothing where it does not clip.
    _pa_flat_p, _pa_flat_f = _pa_stance(
        plasma_radius_profile_cm=_pa_Rp.tolist(),
        machine_radius_profile_cm=_pa_Rm.tolist(),
        plasma_area_max_vessel_fraction=1.0,
    )
    _pa_flat_f["prescribed_area_geometry"] = True
    _pa_flat_sim = LAPDSim1D(dict(_pa_flat_p), dict(_pa_flat_f))
    assert _pa_flat_sim._variable_area_geometry
    _pa_flat_terms = _pa_flat_sim.rhs_terms()
    assert "flux_tube_geometry" in _pa_flat_terms
    assert np.array_equal(
        _pa_flat_terms["flux_tube_geometry"].M, np.zeros(_pa_cells)
    )
    for _pa_field in (
        "Rp_cm", "Rm_cm", "plasma_area_cm2", "neutral_area_cm2",
        "plasma_volume_cm3", "neutral_volume_cm3", "volume_ratio",
        "plasma_face_area_cm2", "neutral_face_area_cm2",
        "neutral_hydraulic_radius_cm", "neutral_face_hydraulic_radius_cm",
    ):
        assert np.array_equal(
            np.asarray(getattr(_pa_flat_sim.geometry, _pa_field), dtype=float),
            np.asarray(getattr(_pa_geom0, _pa_field), dtype=float),
        ), _pa_field
    _pa_flat_result = LAPDSim1D(
        dict(_pa_flat_p), dict(_pa_flat_f)
    ).run(t_end=1.0e-6, dt=1.0e-7)
    assert _pa_raw(_pa_base_result) == _pa_raw(_pa_flat_result), (
        "a constant plasma_radius_profile_cm at Rp must be bit-identical to "
        "the uniform column"
    )
    # ...and the identity survives the duct and support-rod reductions, which
    # were rewritten against the per-cell bore to let a vessel profile compose
    # with them. Same values, same arithmetic, so same bytes.
    _pa_duct_p, _pa_duct_f = _pa_stance(Rcs=40.0, Lcs=25.0, Rsup=10.0)
    _pa_duct_geom = LAPDSim1D(
        dict(_pa_duct_p), dict(_pa_duct_f)
    ).geometry
    assert "obstruction" in set(_pa_duct_geom.cell_role)
    _pa_duct_flat_p, _pa_duct_flat_f = _pa_stance(
        Rcs=40.0,
        Lcs=25.0,
        Rsup=10.0,
        plasma_radius_profile_cm=np.asarray(
            _pa_duct_geom.Rp_cm, dtype=float
        ).tolist(),
        machine_radius_profile_cm=np.asarray(
            _pa_duct_geom.Rm_cm, dtype=float
        ).tolist(),
    )
    _pa_duct_flat_f["prescribed_area_geometry"] = True
    _pa_duct_flat_geom = LAPDSim1D(
        _pa_duct_flat_p, _pa_duct_flat_f
    ).geometry
    for _pa_field in (
        "neutral_area_cm2", "neutral_volume_cm3",
        "neutral_hydraulic_radius_cm", "neutral_face_area_cm2",
        "neutral_face_hydraulic_radius_cm", "volume_ratio",
    ):
        assert np.array_equal(
            np.asarray(getattr(_pa_duct_flat_geom, _pa_field), dtype=float),
            np.asarray(getattr(_pa_duct_geom, _pa_field), dtype=float),
        ), _pa_field

    # (b2) A STEPPED BORE. The point of the vessel vector: the machine radius
    # can change PART WAY along a block of cells, which neither the scalar Rm
    # nor end_expansion_machine_radius_cm (one value over the whole terminal
    # block) can express. The step lands exactly where it was asked to, the
    # open area and hydraulic radius follow it, and the neutral FACE at the
    # step stays a restricting aperture -- the narrow side, as at any other
    # change of bore.
    _pa_step_Rm = _pa_Rm.copy()
    _pa_step_Rm[-3:] = 76.2
    _pa_step_p, _pa_step_f = _pa_stance(
        plasma_radius_profile_cm=_pa_Rp.tolist(),
        machine_radius_profile_cm=_pa_step_Rm.tolist(),
    )
    _pa_step_f["prescribed_area_geometry"] = True
    _pa_step_geom = LAPDSim1D(_pa_step_p, _pa_step_f).geometry
    assert np.array_equal(
        np.asarray(_pa_step_geom.Rm_cm, dtype=float), _pa_step_Rm
    )
    assert np.array_equal(
        np.asarray(_pa_step_geom.neutral_area_cm2, dtype=float),
        np.pi * _pa_step_Rm**2,
    )
    assert np.array_equal(
        np.asarray(_pa_step_geom.neutral_hydraulic_radius_cm, dtype=float),
        _pa_step_Rm,
    )
    assert np.isclose(
        float(_pa_step_geom.neutral_face_area_cm2[_pa_cells - 3]),
        np.pi * float(_pa_Rm[0]) ** 2,
    ), "the face at a bore step must restrict to the narrow side"
    # The plasma is untouched by a vessel-only change.
    assert np.array_equal(
        np.asarray(_pa_step_geom.plasma_area_cm2, dtype=float),
        np.asarray(_pa_geom0.plasma_area_cm2, dtype=float),
    )

    # (b3) THE SLIVER-ANNULUS GUARD and its declared cap. A flux tube that
    # nearly fills the bore leaves an annulus that is positive -- so every
    # `V_ann > 0` gate in the annulus consumers passes -- yet tiny, and V_ann
    # is a DIVISOR (the zone exchange and the hot-channel deposit both scale
    # as 1/V_ann). The guard refuses that geometry at construction; the
    # declared area ceiling is the way to run it anyway, and it binds before
    # the sign refusal so a capped configuration cannot also trip that error.
    _pa_tight_rp = _pa_Rp.copy()
    _pa_tight_rp[-3:] = 76.17
    _pa_tight_Rm = _pa_Rm.copy()
    _pa_tight_Rm[-3:] = 76.2
    _pa_tight_p, _pa_tight_f = _pa_stance(
        plasma_radius_profile_cm=_pa_tight_rp.tolist(),
        machine_radius_profile_cm=_pa_tight_Rm.tolist(),
    )
    _pa_tight_f["prescribed_area_geometry"] = True
    try:
        LAPDSim1D(dict(_pa_tight_p), dict(_pa_tight_f))
    except ValueError as _pa_exc:
        assert "collapsed to a sliver" in str(_pa_exc), str(_pa_exc)
    else:
        raise AssertionError("a collapsed two-zone annulus must be refused")
    # The declared ceiling makes the two-zone case legal, at exactly the
    # stated fraction of the local vessel area.
    _pa_cap_p = dict(_pa_tight_p)
    _pa_cap_p["plasma_area_max_vessel_fraction"] = 0.95
    _pa_cap_geom = LAPDSim1D(_pa_cap_p, dict(_pa_tight_f)).geometry
    _pa_cap_ratio = (
        np.asarray(_pa_cap_geom.plasma_area_cm2, dtype=float)
        / np.asarray(_pa_cap_geom.neutral_area_cm2, dtype=float)
    )
    assert np.allclose(_pa_cap_ratio[-3:], 0.95, rtol=0.0, atol=1e-15)
    assert np.all(_pa_cap_ratio <= 0.95 + 1e-15)
    # ...and it leaves the cells it does not bind on exactly as supplied.
    assert np.array_equal(
        np.asarray(_pa_cap_geom.Rp_cm, dtype=float)[:-3], _pa_tight_rp[:-3]
    )
    _pa_cap_Vc, _pa_cap_Va = neutral_zone_volumes(_pa_cap_geom)
    assert np.allclose(
        (_pa_cap_Va / np.asarray(_pa_cap_geom.neutral_volume_cm3,
                                 dtype=float))[-3:],
        0.05,
        rtol=1e-12,
        atol=0.0,
    )
    # Cells with NO annulus at all are exempt: an absent zone is inert,
    # because every consumer already gates on V_ann > 0.
    _pa_full_rp = _pa_Rp.copy()
    _pa_full_rp[-3:] = float(_pa_Rm[0])
    _pa_full_p, _pa_full_f = _pa_stance(
        plasma_radius_profile_cm=_pa_full_rp.tolist(),
    )
    _pa_full_f["prescribed_area_geometry"] = True
    _pa_full_sim = LAPDSim1D(_pa_full_p, _pa_full_f)
    assert np.all(neutral_zone_volumes(_pa_full_sim.geometry)[1][-3:] == 0.0)


# --------------------------------------------------------------------
# prescribed-area-well-balancedness
# --------------------------------------------------------------------
@_case("prescribed-area-well-balancedness")
def _case_prescribed_area_well_balancedness(
    _pa_Rm, _pa_Rp, _pa_base_p, _pa_base_result, _pa_cells, _pa_geom0,
    _pa_raw, _pa_stance
):
    # (c) WELL-BALANCEDNESS -- the load-bearing property of the mirror force.
    # On a strongly varying A(z) a static uniform-pressure plasma must generate
    # EXACTLY no momentum: the quasi-1D p*dA/dz source is written with the same
    # multiply-then-subtract ordering as the area-weighted pressure flux it
    # pairs with, so the cancellation is bit-exact rather than merely
    # algebraic. The carve-out is the two plasma-TERMINATING live cells, where
    # the characteristic ghost-cell outflow (a term not summed here) carries
    # the face momentum instead of a reflecting wall pressure -- the same
    # carve-out the end_expansion block above makes.
    _pa_col = np.flatnonzero(
        np.isin(
            _pa_geom0.cell_role, np.asarray(["puff", "column"], dtype=object)
        )
    )
    assert _pa_col.size >= 8
    _pa_flare = _pa_Rp.copy()
    _pa_flare[_pa_col] = _pa_Rp[_pa_col] * (
        1.0 + 0.8 * np.sin(np.linspace(0.3, 3.0, _pa_col.size)) ** 2
    )
    _pa_var_p, _pa_var_f = _pa_stance(
        plasma_radius_profile_cm=_pa_flare.tolist()
    )
    _pa_var_f["prescribed_area_geometry"] = True
    _pa_var_sim = LAPDSim1D(dict(_pa_var_p), dict(_pa_var_f))
    _pa_var_geom = _pa_var_sim.geometry
    assert np.array_equal(_pa_var_geom.Rp_cm, _pa_flare)
    assert np.array_equal(_pa_var_geom.plasma_area_cm2, np.pi * _pa_flare**2)
    assert np.array_equal(
        _pa_var_geom.plasma_volume_cm3,
        np.pi * _pa_flare**2 * np.asarray(_pa_geom0.length_cm, dtype=float),
    )
    _pa_var_area = np.asarray(_pa_var_geom.plasma_area_cm2, dtype=float)
    assert float(np.ptp(_pa_var_area[_pa_col])) > 0.5 * float(
        np.max(_pa_var_area[_pa_col])
    ), "the synthetic flare must actually vary strongly"

    _pa_static = conservative_from_primitives(
        n=np.full(_pa_cells, 1.0e12),
        nn=np.full(_pa_cells, 1.0e12),
        u=np.zeros(_pa_cells),
        Te=np.full(_pa_cells, 2.0),
        Ti=np.full(_pa_cells, 1.0),
        ion_mass_g=_pa_var_sim.ion_mass_g,
    )
    _pa_adv = _pa_var_sim.plasma_flux_rhs_terms(
        state=_pa_static
    )["plasma_advective_flux"]
    _pa_geo = _pa_var_sim.flux_tube_geometry_rhs(state=_pa_static)
    _pa_terminating = sorted({
        int(cell)
        for cells in absorbing_live_cells_by_role(_pa_var_geom).values()
        for cell in cells
    })
    assert _pa_terminating, "the machine must have plasma-terminating faces"
    _pa_balanced = np.setdiff1d(np.arange(_pa_cells), _pa_terminating)
    assert set(_pa_col.tolist()) <= set(_pa_balanced.tolist())
    assert np.array_equal(
        (_pa_adv.M + _pa_geo.M)[_pa_balanced], np.zeros(_pa_balanced.size)
    ), "the mirror force is not well-balanced on a varying A(z)"
    # The source is momentum ONLY, and it is genuinely doing something: a
    # uniform column makes it vanish, so the flare is what is being measured
    # rather than an accident of the cancellation.
    for _pa_row in (_pa_geo.n, _pa_geo.nn, _pa_geo.Ee, _pa_geo.Ei):
        assert np.array_equal(np.asarray(_pa_row, dtype=float),
                              np.zeros(_pa_cells))
    assert np.any(_pa_geo.M[_pa_col] != 0.0)

    # (d) BOTH ENERGY EQUATIONS SEE THE SAME AREA. The pressure work is
    # -p_s * div u with div u the FACE-AREA-weighted d(Au)/dz / V, so a uniform
    # drift through the flare does expansion work on electrons and ions alike.
    # Checked against the closed form and against both rows.
    _pa_u0 = 3.0e5
    _pa_drift = conservative_from_primitives(
        n=np.full(_pa_cells, 1.0e12),
        nn=np.full(_pa_cells, 1.0e12),
        u=np.full(_pa_cells, _pa_u0),
        Te=np.full(_pa_cells, 2.0),
        Ti=np.full(_pa_cells, 1.0),
        ion_mass_g=_pa_var_sim.ion_mass_g,
    )
    _pa_divu = velocity_divergence(
        _pa_drift,
        _pa_var_sim.floors,
        _pa_var_sim.ion_mass_g,
        _pa_var_geom,
        active_plasma_topology=_pa_var_sim._active_plasma_topology,
    )
    _pa_face_A = np.asarray(_pa_var_geom.plasma_face_area_cm2, dtype=float)
    _pa_want_divu = (
        _pa_u0 * (_pa_face_A[1:] - _pa_face_A[:-1])
    ) / np.asarray(_pa_var_geom.plasma_volume_cm3, dtype=float)
    assert np.allclose(
        _pa_divu[_pa_col], _pa_want_divu[_pa_col], rtol=1e-12, atol=0.0
    )
    assert np.any(np.abs(_pa_divu[_pa_col]) > 0.0)
    _pa_pw = _pa_var_sim.pressure_work_rhs(state=_pa_drift)
    _pa_derived = derive_state(
        _pa_drift, _pa_var_sim.floors, _pa_var_sim.ion_mass_g
    )
    assert np.array_equal(_pa_pw.Ee, -_pa_derived.pe * _pa_divu)
    assert np.array_equal(_pa_pw.Ei, -_pa_derived.pi * _pa_divu)
    assert np.any(_pa_pw.Ee[_pa_col] != 0.0)
    assert np.any(_pa_pw.Ei[_pa_col] != 0.0)

    # (e) CONSERVATION UNDER A VARYING AREA. The area-weighted face inventory
    # and the cell volume are the same pairing, so the advective flux still
    # telescopes exactly across the closed domain; the reaction terms still
    # close in the volume-integrated inventory (their Vp/Vm conversion follows
    # the profile); the two-zone volumes still partition the chamber cell by
    # cell with V_ann > 0 everywhere the plasma does not fill it, and the zone
    # exchange conserves what it moves; and a stepped run stays finite.
    _pa_var_Vp = np.asarray(_pa_var_geom.plasma_volume_cm3, dtype=float)
    assert math.fsum((_pa_adv.n * _pa_var_Vp).tolist()) == 0.0
    _pa_react_terms = _pa_var_sim.reaction_rhs_terms(state=_pa_static)
    for _pa_name, _pa_term in _pa_react_terms.items():
        _pa_scale = float(
            np.sum(np.abs(_pa_term.n * _pa_var_Vp))
            + np.sum(np.abs(_pa_term.nn * _pa_var_geom.neutral_volume_cm3))
        )
        assert abs(
            particle_inventory_rate(_pa_term, _pa_var_geom)
        ) <= 1e-12 * _pa_scale, _pa_name

    _pa_tz_p, _pa_tz_f = _pa_stance(
        plasma_radius_profile_cm=_pa_flare.tolist(),
    )
    _pa_tz_f["prescribed_area_geometry"] = True
    _pa_tz_sim = LAPDSim1D(_pa_tz_p, _pa_tz_f)
    _pa_Vc, _pa_Va = neutral_zone_volumes(_pa_tz_sim.geometry)
    assert np.array_equal(
        _pa_Vc, np.asarray(_pa_tz_sim.geometry.plasma_volume_cm3, dtype=float)
    )
    assert np.all(_pa_Va > 0.0)
    assert np.allclose(
        _pa_Vc + _pa_Va,
        np.asarray(_pa_tz_sim.geometry.neutral_volume_cm3, dtype=float),
        rtol=1e-14,
        atol=0.0,
    )
    _pa_tz_state = _pa_tz_sim.state
    _pa_tz_nn = _pa_tz_state.nn.copy()
    _pa_tz_nn[_pa_col] *= 0.5
    _pa_tz_pert = ConservativeState1D(
        _pa_tz_state.n,
        _pa_tz_nn,
        _pa_tz_state.M,
        _pa_tz_state.Ee,
        _pa_tz_state.Ei,
        nn_a=_pa_tz_state.nn_a.copy(),
    )
    _pa_zx = _pa_tz_sim.neutral_zone_exchange_rhs(state=_pa_tz_pert)
    assert np.any(_pa_zx.nn[_pa_col] > 0.0)
    assert abs(
        float((_pa_zx.nn * _pa_Vc + _pa_zx.nn_a * _pa_Va).sum())
    ) <= 1e-12 * float(np.abs(_pa_zx.nn * _pa_Vc).max())
    _pa_var_result = LAPDSim1D(
        dict(_pa_var_p), dict(_pa_var_f)
    ).run(t_end=1.0e-6, dt=1.0e-7)
    assert _pa_var_result.steps > 0
    assert all(np.all(np.isfinite(_pa_y)) for _pa_y in _pa_var_result.y)
    assert _pa_raw(_pa_var_result) != _pa_raw(_pa_base_result), (
        "a strongly varying flux tube that changes nothing would mean the "
        "profile never reached the solver"
    )

    # (f) EVERY MISCONFIGURATION RAISES, at construction.
    def _pa_refuses(label, params_over=None, flags_over=None, expected=None):
        params, flags = _pa_stance(
            plasma_radius_profile_cm=_pa_flare.tolist()
        )
        flags["prescribed_area_geometry"] = True
        params.update(params_over or {})
        flags.update(flags_over or {})
        try:
            LAPDSim1D(params, flags)
        except ValueError as exc:
            if expected is not None:
                assert expected in str(exc), (label, str(exc))
            return
        raise AssertionError(f"prescribed_area_geometry must refuse: {label}")

    _pa_refuses(
        "a profile shorter than the mesh",
        params_over={"plasma_radius_profile_cm": _pa_flare[:-1].tolist()},
        expected="one entry per MESH cell",
    )
    _pa_refuses(
        "a profile the length of nx rather than of the mesh",
        params_over={"plasma_radius_profile_cm": [18.415] * 12},
        expected="one entry per MESH cell",
    )
    _pa_refuses(
        "a non-finite entry",
        params_over={
            "plasma_radius_profile_cm": _pa_flare[:-1].tolist() + [float("nan")]
        },
        expected="must be finite",
    )
    for _pa_bad in (0.0, -1.0):
        _pa_refuses(
            f"a non-positive entry ({_pa_bad}): a zero-area cell divides the "
            "flux divergence by a zero volume",
            params_over={
                "plasma_radius_profile_cm": _pa_flare[:-1].tolist() + [_pa_bad]
            },
            expected="must be > 0",
        )
    _pa_refuses(
        "an empty profile",
        params_over={"plasma_radius_profile_cm": []},
        expected="non-empty",
    )
    _pa_refuses(
        "a radius past the vessel WALL",
        params_over={
            "plasma_radius_profile_cm": (
                _pa_flare[:-1].tolist()
                + [float(_pa_base_p["Rm"]) + 1.0]
            )
        },
        expected="narrower than",
    )
    # The OPEN area is the binding one where a duct blocks part of the bore:
    # a plasma inside Rm can still be wider than the annular gap around the
    # cathode structure, and that is what would drive V_ann negative.
    _pa_duct_refuse_p, _pa_duct_refuse_f = _pa_stance(
        Rcs=40.0,
        Lcs=25.0,
        plasma_radius_profile_cm=np.full(_pa_cells + 1, 45.0).tolist(),
    )
    _pa_duct_refuse_f["prescribed_area_geometry"] = True
    try:
        LAPDSim1D(_pa_duct_refuse_p, _pa_duct_refuse_f)
    except ValueError as _pa_exc:
        assert "exceeds the local vessel open area" in str(_pa_exc), str(_pa_exc)
    else:
        raise AssertionError(
            "a plasma wider than a duct's open area must be refused"
        )
    _pa_refuses(
        "the flag armed with no profile at all",
        params_over={"plasma_radius_profile_cm": None},
        expected="requires plasma_radius_profile_cm",
    )
    _pa_refuses(
        "the built-in half-cosine flare configured alongside it -- two "
        "prescriptions of the same area with no composition rule",
        params_over={
            "Lm": 2125.85,
            "end_wall_length_cm": 150.0,
            "end_expansion_cells": 10,
            "end_expansion_machine_radius_cm": 100.0,
            "end_expansion_plasma_radius_cm": 50.0,
        },
        flags_over={"end_expansion_geometry": True},
        expected="cannot be combined with end_expansion_geometry",
    )
    # The vessel profile's own bad values, and the pair's consistency.
    _pa_refuses(
        "a vessel profile shorter than the mesh",
        params_over={"machine_radius_profile_cm": _pa_Rm[:-1].tolist()},
        expected="one entry per MESH cell",
    )
    _pa_refuses(
        "a non-positive vessel radius",
        params_over={
            "machine_radius_profile_cm": _pa_Rm[:-1].tolist() + [0.0]
        },
        expected="must be > 0",
    )
    _pa_refuses(
        "a non-finite vessel radius",
        params_over={
            "machine_radius_profile_cm": _pa_Rm[:-1].tolist() + [float("inf")]
        },
        expected="must be finite",
    )
    _pa_refuses(
        "a vessel narrower than the plasma it contains",
        params_over={
            "machine_radius_profile_cm": (
                _pa_Rm[:-1].tolist() + [float(_pa_flare[-1]) * 0.5]
            )
        },
        expected="narrower than",
    )
    for _pa_bad_cap in (0.0, -0.5, 1.5, float("nan")):
        _pa_refuses(
            f"an area ceiling outside (0, 1] ({_pa_bad_cap})",
            params_over={"plasma_area_max_vessel_fraction": _pa_bad_cap},
            expected="plasma_area_max_vessel_fraction must be finite and in",
        )
    for _pa_bad_thr in (-0.1, 1.0, 2.0, float("nan")):
        _pa_refuses(
            f"an annulus-fraction threshold outside [0, 1) ({_pa_bad_thr})",
            params_over={
                "neutral_annulus_volume_fraction_min": _pa_bad_thr,
            },
            expected="neutral_annulus_volume_fraction_min must be finite",
        )
    # ...and the presence gate the other way: ANY of the three parameters set
    # with the flag off is inert, so it raises rather than silently running the
    # uniform column inside the scalar bore.
    for _pa_off_key, _pa_off_value in (
        ("plasma_radius_profile_cm", _pa_flare.tolist()),
        ("machine_radius_profile_cm", _pa_Rm.tolist()),
        ("plasma_area_max_vessel_fraction", 0.95),
    ):
        _pa_off_p, _pa_off_f = _pa_stance(**{_pa_off_key: _pa_off_value})
        try:
            LAPDSim1D(_pa_off_p, _pa_off_f)
        except ValueError as _pa_exc:
            assert "require the default-off prescribed_area_geometry flag" in (
                str(_pa_exc)
            ), str(_pa_exc)
            assert _pa_off_key in str(_pa_exc), str(_pa_exc)
        else:
            raise AssertionError(
                f"{_pa_off_key} must be refused with the flag off"
            )


# --------------------------------------------------------------------
# The axial field-map loader (physics/mirror_field.py). There is NO mirror
# flag and NO mirror config key here, deliberately: the fluid mirror force is
# already in the model as the quasi-1D p dA/dz source
# (sources.flux_tube_geometry_rhs, armed by prescribed_area_geometry), and at
# A proportional to 1/B that source IS the isotropic average of
# -mu grad_par B, so a second term would double-count it exactly. The loader
# is a library function whose one in-tree consumer is
# scripts/characterise_mirror_fieldmap.py (at commit 48be9a4, retired
# 2026-09-03). These cases pin its arithmetic and
# its refusals.
# --------------------------------------------------------------------
def _mirror_synthetic_map(directory, *, drop=()):
    """Write a minimal census-shaped field map and return its path.

    Synthesized rather than read from ``scripts/lapd_end_field_Rp18p415.npz``
    so the gate does not depend on an ignored run artifact. The shape is
    deliberately analytic -- flat at B_bulk, then a linear ramp -- so every
    assertion below is a closed-form expectation, not a re-measurement of the
    real solve. ``drop`` omits named arrays, to exercise the malformed-map
    refusal.
    """
    path = os.path.join(directory, "mirror_synthetic_map.npz")
    z_m = np.linspace(14.0, 22.5, 4251)
    b_g = np.where(z_m < 19.0, 1400.0, 1400.0 - 200.0 * (z_m - 19.0))
    z_flux_m = np.linspace(14.0, 21.0, 701)
    r_flux_m = 0.18415 + 0.05 * np.maximum(z_flux_m - 19.0, 0.0)
    arrays = dict(
        z_axis_m=z_m,
        bulk_field_gauss=np.array(1400.0),
        plasma_radius_m=np.array(0.18415),
        interior_ripple_relative=np.array(5.430480263014005e-05),
        droop_min_bz_axis_gauss=b_g,
        droop_min_z_flux_m=z_flux_m,
        droop_min_flux_radius_m=r_flux_m,
        droop_min_crossing_z_m=np.array([20.0, 20.5]),
        droop_min_crossing_radii_m=np.array([0.5, 0.762]),
        droop_min_trace_end_z_m=np.array(21.0),
        off_bz_axis_gauss=0.5 * b_g,
        off_z_flux_m=z_flux_m,
        off_flux_radius_m=r_flux_m,
        off_crossing_z_m=np.array([19.5, 20.0]),
        off_crossing_radii_m=np.array([0.5, 0.762]),
        off_trace_end_z_m=np.array(21.0),
    )
    for name in drop:
        del arrays[name]
    np.savez_compressed(path, **arrays)
    return path


@_case("mirror-field-loader")
def _case_mirror_field_loader():
    """The map lands on the mesh in CGS, with the fill and masks declared."""
    from cablp.solvers._sim1d.core.geometry import build_geometry
    from cablp.solvers._sim1d.physics.mirror_field import load_mirror_field

    _mf_p, _mf_f = default_config()
    _mf_geom = build_geometry(_mf_p, _mf_f)
    with tempfile.TemporaryDirectory() as _mf_dir:
        _mf_map = _mirror_synthetic_map(_mf_dir)
        _mf = load_mirror_field(
            map_path=_mf_map,
            case="droop_min",
            geometry=_mf_geom,
            plasma_radius_cm=_mf_p["Rp"],
        )
        assert _mf.case == "droop_min"
        assert _mf.B_cell_gauss.shape == (_mf_geom.cells,)
        assert _mf.B_face_gauss.shape == (_mf_geom.cells + 1,)
        assert np.all(np.isfinite(_mf.B_cell_gauss))
        # Below the map the fill is the map's own bulk level, with exactly
        # zero gradient -- the declared approximation, not a solved field.
        assert _mf.interior_fill_gauss == 1400.0
        assert np.all(_mf.interior_fill_cell == (_mf.z_cm < _mf.map_z_min_cm))
        assert np.all(_mf.B_cell_gauss[_mf.interior_fill_cell] == 1400.0)
        assert np.all(
            _mf.dBdz_native_cell_gauss_per_cm[_mf.interior_fill_cell] == 0.0
        )
        # On the synthetic ramp the slope is -200 G/m = -2 G/cm exactly, and
        # both differencings must find it (the ramp is linear, so the native
        # and mesh gradients agree there).
        _mf_ramp = _mf.z_cm > 1910.0
        assert np.allclose(
            _mf.dBdz_native_cell_gauss_per_cm[_mf_ramp], -2.0, atol=1e-9
        ), "dB/dz must be gauss per CM"
        assert np.allclose(
            _mf.dBdz_mesh_cell_gauss_per_cm[_mf_ramp], -2.0, atol=1e-9
        )
        # Mirror ratio is a pure ratio of the same array.
        assert np.allclose(
            _mf.mirror_ratio_cell, _mf.B_cell_gauss / _mf.B_min_gauss
        )
        assert np.allclose(_mf.mirror_ratio_bulk_cell, _mf.B_cell_gauss / 1400.0)
        assert _mf.B_min_gauss == _mf.B_cell_gauss.min()
        assert _mf.B_max_gauss == _mf.B_cell_gauss.max()
        # The flux surface is NaN outside the trace and masked past first
        # wall contact -- past that point it is vacuum continuation, and the
        # loader says so instead of handing over a bare number.
        assert _mf.first_wall_contact_z_cm == 2000.0
        assert np.all(
            np.isnan(_mf.flux_radius_cell_cm[~_mf.flux_radius_valid_cell])
        )
        assert np.all(
            _mf.flux_radius_vacuum_continuation_cell
            <= _mf.flux_radius_valid_cell
        )
        assert np.all(
            _mf.z_cm[_mf.flux_radius_vacuum_continuation_cell] >= 2000.0
        )
        assert not np.any(
            _mf.flux_radius_vacuum_continuation_cell & (_mf.z_cm < 2000.0)
        )
        # The cell average is the exact mean of the same interpolant, so on
        # the flat interior it equals the point sample.
        _mf_flat = _mf.z_cm < 1800.0
        assert np.allclose(
            _mf.B_cell_average_gauss[_mf_flat], _mf.B_cell_gauss[_mf_flat]
        )
        # The two end-coil cases are NOT small perturbations of each other,
        # which is why 'case' has no default reading.
        _mf_off = load_mirror_field(
            map_path=_mf_map, case="off", geometry=_mf_geom
        )
        _mf_solved = ~_mf.interior_fill_cell
        assert np.allclose(
            _mf_off.B_cell_gauss[_mf_solved], 0.5 * _mf.B_cell_gauss[_mf_solved]
        )


@_case("mirror-field-loader-refusals")
def _case_mirror_field_loader_refusals():
    """Every way of asking the loader for a field it cannot supply."""
    from cablp.solvers._sim1d import config_manifest
    from cablp.solvers._sim1d.core.geometry import build_geometry
    from cablp.solvers._sim1d.physics import mirror_field as _mf_mod
    from cablp.solvers._sim1d.physics.mirror_field import load_mirror_field

    # The module must carry no force term, and the solver no mirror control:
    # the mirror force is already in the model and a sibling would
    # double-count it exactly.
    assert not hasattr(_mf_mod, "mirror_force_rhs")
    assert not hasattr(_mf_mod, "MIRROR_FORCE_PENDING")
    _mf_manifest = config_manifest()
    for _key in (
        "flux_tube_mirror",
        "mirror_field_map_path",
        "mirror_field_case",
        "mirror_field_interior_fill_gauss",
    ):
        assert _key not in _mf_manifest["parameters"], _key
        assert _key not in _mf_manifest["flags"], _key

    _mf_p, _mf_f = default_config()
    _mf_geom = build_geometry(_mf_p, _mf_f)
    with tempfile.TemporaryDirectory() as _mf_dir:
        _mf_map = _mirror_synthetic_map(_mf_dir)

        # An unknown end-coil case, and an undeclared one, both refuse before
        # the file is even opened.
        for _bad_case in ("droopmin", None, ""):
            try:
                load_mirror_field(
                    map_path=_mf_map, case=_bad_case, geometry=_mf_geom
                )
            except ValueError as exc:
                assert "must be one of" in str(exc), str(exc)
            else:
                raise AssertionError(f"case={_bad_case!r} must be refused")
        # Both required arguments have no default, so omitting either is a
        # TypeError at the call site rather than a guessed reading.
        for _kwargs in (
            {"map_path": _mf_map, "geometry": _mf_geom},
            {"case": "droop_min", "geometry": _mf_geom},
        ):
            try:
                load_mirror_field(**_kwargs)
            except TypeError as exc:
                assert "required keyword-only argument" in str(exc), str(exc)
            else:
                raise AssertionError(f"load_mirror_field({_kwargs}) must refuse")
        # A map path that names nothing readable.
        try:
            load_mirror_field(
                map_path=os.path.join(_mf_dir, "absent.npz"),
                case="off",
                geometry=_mf_geom,
            )
        except ValueError as exc:
            assert "does not name a readable file" in str(exc), str(exc)
        else:
            raise AssertionError("an unreadable map path must be refused")
        # A non-positive or non-finite interior fill.
        for _bad_fill in (-1.0, 0.0, float("nan")):
            try:
                load_mirror_field(
                    map_path=_mf_map, case="off", geometry=_mf_geom,
                    interior_fill_gauss=_bad_fill,
                )
            except ValueError as exc:
                assert "must be finite and positive" in str(exc), str(exc)
            else:
                raise AssertionError(f"fill={_bad_fill} must be refused")
        # A map whose flux surface was anchored on a different column.
        try:
            load_mirror_field(
                map_path=_mf_map, case="droop_min", geometry=_mf_geom,
                plasma_radius_cm=15.0,
            )
        except ValueError as exc:
            assert "traced its flux surface on a column of radius" in str(exc), (
                str(exc)
            )
        else:
            raise AssertionError("a map anchored on another column must refuse")
        # A mesh that runs past the map's high edge has nothing to read.
        _mf_long_p = dict(_mf_p)
        _mf_long_p["Lm"] = 2400.0
        _mf_long_geom = build_geometry(_mf_long_p, _mf_f)
        try:
            load_mirror_field(
                map_path=_mf_map, case="droop_min", geometry=_mf_long_geom
            )
        except ValueError as exc:
            assert "nothing to extend the field from" in str(exc), str(exc)
        else:
            raise AssertionError("a mesh past the map's high edge must refuse")

    # A file that is missing an array the loader needs is not a census output.
    with tempfile.TemporaryDirectory() as _mf_bad_dir:
        _mf_bad_map = _mirror_synthetic_map(
            _mf_bad_dir, drop=("droop_min_bz_axis_gauss",)
        )
        try:
            load_mirror_field(
                map_path=_mf_bad_map, case="droop_min", geometry=_mf_geom
            )
        except ValueError as exc:
            assert "has no array" in str(exc), str(exc)
            assert "solve_lapd_coil_field_census.py" in str(exc), str(exc)
        else:
            raise AssertionError("a map missing its Bz array must be refused")
