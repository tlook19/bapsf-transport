"""Smoke cases: the end-wall sheath, its face fluxes and its retired names."""

import math

import numpy as np

from cablp.cathode.circuit_common import sheath_lift_lambda
from cablp.constants import ev_to_erg, m_He_cgs
from cablp.solvers._sim1d import LAPDSim1D, default_config
from cablp.solvers._sim1d.core.geometry import absorbing_live_cells_by_role
from cablp.solvers._sim1d.core.state import derive_state
from cablp.solvers._sim1d.physics.sources import electrode_sheath_alpha
from cablp.solvers._sim1d.solver import (
    END_SHEATH_CATHODE_ROWS,
    END_SHEATH_DEBIT_ROWS,
    END_SHEATH_END_WALL_ROWS,
)

from ._harness import _base_config, _case, _resolved_config


# --------------------------------------------------------------------
# end-face-full-debit-split
# --------------------------------------------------------------------
@_case("end-face-full-debit-split", historical_stance=True)
def _case_end_face_full_debit_split():
    # THE TWO END FACES, each arming ITS OWN rows, and neither by a key. The
    # end wall's sheath-climb row is armed by the geometry's end wall face;
    # the emitting cathode face's three rows by the geometry's cathode face
    # with the cathode circuit solve running. This single-cathode layout has
    # both faces, so with the solve on it carries all four rows.
    _es_params, _es_flags = _base_config()
    _es_flags = dict(_es_flags)
    _es_flags["cathode_coupling"] = True

    # (i) WITHOUT THE CIRCUIT SOLVE the cathode face emits nothing and
    # collects nothing against a barrier, so its three rows are ABSENT --
    # which is a different statement from present-and-zero -- while the end
    # wall row, which rides the boundary operator's own flux, is present.
    _es_nocirc_flags = dict(_es_flags)
    _es_nocirc_flags["cathode_coupling"] = False
    _es_off = LAPDSim1D(dict(_es_params), _es_nocirc_flags)
    _es_off_terms = _es_off.rhs_terms()
    assert set(_es_off_terms) & set(END_SHEATH_DEBIT_ROWS) == set(
        END_SHEATH_END_WALL_ROWS
    )

    # (ii) WITH THE SOLVE, the cathode face arms its own rows and only those,
    # checked against the solve-off term set, so a row leaking across the
    # split shows up as a set difference rather than as a number nobody
    # looked at. (The solve also arms the electrode rows it always owns.)
    _es_both = LAPDSim1D(dict(_es_params), dict(_es_flags))
    _es_on_terms = _es_both.rhs_terms()
    _es_gained = set(_es_on_terms) - set(_es_off_terms)
    assert set(END_SHEATH_CATHODE_ROWS) <= _es_gained, sorted(_es_gained)
    assert not _es_gained & set(END_SHEATH_END_WALL_ROWS)
    assert set(END_SHEATH_DEBIT_ROWS) <= set(_es_on_terms)

    # (iii) ARMED, the four rows are ELECTRON ENERGY ONLY and land on the
    # faces they name.
    _es_geom = _es_both.geometry
    _es_roles = np.asarray(_es_geom.cell_role)
    _es_coll = int(np.flatnonzero(_es_roles == "end_wall")[0])
    _es_cath = int(np.flatnonzero(_es_roles == "cathode")[0])
    _es_rows = {}
    for _es_name in END_SHEATH_DEBIT_ROWS:
        _es_term = _es_on_terms[_es_name]
        for _es_field in ("n", "nn", "M", "Ei"):
            assert np.all(np.asarray(getattr(_es_term, _es_field)) == 0.0), (
                _es_name, _es_field
            )
        _es_row = np.asarray(_es_term.Ee, dtype=float)
        assert np.all(np.isfinite(_es_row)), _es_name
        _es_rows[_es_name] = _es_row
    _es_cell = {
        "end_wall_e_sheath_climb": _es_coll,
        "cathode_e_emitted_enthalpy": _es_cath,
        "cathode_e_emitted_fall": _es_cath,
        "cathode_e_collected_climb": _es_cath,
    }
    for _es_name, _es_row in _es_rows.items():
        _es_support = set(np.flatnonzero(_es_row).tolist())
        assert _es_support <= {_es_cell[_es_name]}, (_es_name, _es_support)

    # (iv) SIGNS, which are the physics and are not free. The end wall debit
    # comes OUT of the electron store; the emitted electrons' enthalpy and the
    # part of the fall the beam row does not carry go INTO it; the returning
    # electrons' barrier climb comes out. All four are nonzero on this stance
    # at its initial state -- a virtual cathode has formed there, so the fall
    # row is exercised rather than sitting at its zero branch.
    assert _es_rows["end_wall_e_sheath_climb"][_es_coll] < 0.0
    assert _es_rows["cathode_e_emitted_enthalpy"][_es_cath] > 0.0
    assert _es_rows["cathode_e_emitted_fall"][_es_cath] > 0.0
    assert _es_rows["cathode_e_collected_climb"][_es_cath] < 0.0

    # (v) THE CLOSED FORMS, against the solve and the boundary flux this very
    # evaluation used. The end wall face must debit the sheath-edge
    # (2 + Lambda_eff) Te per collected electron once the new row is added to
    # the 2 Te that characteristic_boundary always books, with Lambda_eff read
    # off the SAME alpha the boundary sampled its flux at.
    _es_state = _es_both.state
    _es_derived = derive_state(
        _es_state, floors=_es_both.floors, ion_mass_g=_es_both.ion_mass_g
    )
    _es_alpha = electrode_sheath_alpha(
        nn=float(_es_state.nn[_es_coll]),
        Te=float(_es_derived.Te[_es_coll]),
        Ti=float(_es_derived.Ti[_es_coll]),
        cell_length_cm=float(_es_geom.length_cm[_es_coll]),
        ion_mass_g=_es_both.ion_mass_g,
        alpha_isat=float(_es_params["alpha_isat"]),
        b_presheath_length=float(_es_params["b_presheath_length"]),
    )
    _es_lambda_eff = (
        sheath_lift_lambda(_es_both.ion_mass_g) - math.log(_es_alpha)
    )
    _es_gamma = float(_es_on_terms["characteristic_boundary"].n[_es_coll])
    _es_booked = (
        float(_es_on_terms["characteristic_boundary"].Ee[_es_coll])
        + _es_rows["end_wall_e_sheath_climb"][_es_coll]
    )
    _es_closed = (
        (2.0 + _es_lambda_eff)
        * float(_es_derived.Te[_es_coll])
        * ev_to_erg
        * _es_gamma
    )
    assert np.isclose(_es_booked, _es_closed, rtol=1e-12, atol=0.0), (
        _es_booked, _es_closed
    )
    # ... and the three cathode rows are their closed forms in watts, built
    # from the solve's own released and returning currents.
    _es_result = _es_both._cathode_solve.beam_result.result
    _es_Ts = float(_es_params["cathode_Ts_base_K"])
    _es_Vp = float(_es_geom.plasma_volume_cm3[_es_cath])
    _es_phi_plus = float(_es_result.phi_c_plus)
    _es_phi = float(_es_result.phi_c)
    assert float(_es_result.phi_c_minus) > 0.0, "expected a virtual cathode"
    for _es_name, _es_expect_W in (
        (
            "cathode_e_emitted_enthalpy",
            2.0 * 8.617333262e-5 * _es_Ts * float(_es_result.I_eth_star),
        ),
        (
            "cathode_e_emitted_fall",
            (_es_phi_plus - max(_es_phi, 0.0)) * float(_es_result.I_eth_star),
        ),
        (
            "cathode_e_collected_climb",
            -_es_phi_plus * float(_es_result.I_e_ret),
        ),
    ):
        _es_booked_W = _es_rows[_es_name][_es_cath] * _es_Vp / 1.0e7
        assert np.isclose(
            _es_booked_W, _es_expect_W, rtol=1e-12, atol=0.0
        ), (_es_name, _es_booked_W, _es_expect_W)

    # (vi) THE CATHODE FACE'S KEY IS RETIRED and refuses at the
    # configuration boundary at either value, naming what arms the rows now:
    # the name owns no read any more, so stating it at all would be the
    # silent inert control that boundary exists to forbid.
    for _es_value in (True, False):
        _es_retired_flags = dict(_es_flags)
        _es_retired_flags["cathode_face_full_debit"] = _es_value
        try:
            LAPDSim1D(dict(_es_params), _es_retired_flags)
        except ValueError as _es_exc:
            assert "unknown LAPDSim1D configuration keys" in str(_es_exc), (
                _es_exc
            )
            assert "cathode_face_full_debit is RETIRED" in str(_es_exc), (
                _es_exc
            )
            assert "cathode face" in str(_es_exc), _es_exc
        else:
            raise AssertionError(
                f"cathode_face_full_debit={_es_value} was accepted"
            )

    # (vii) THE MERGED KEY IS RETIRED and refuses at the configuration
    # boundary, naming its replacement -- the case a stored file written
    # before the split hits. Its old default value is refused too: the name
    # owns no read any more, so stating it at all would be the silent inert
    # control that boundary exists to forbid.
    for _es_value in (True, False):
        _es_retired_flags = dict(_es_flags)
        _es_retired_flags["end_sheath_full_debit"] = _es_value
        try:
            LAPDSim1D(dict(_es_params), _es_retired_flags)
        except ValueError as _es_exc:
            assert "unknown LAPDSim1D configuration keys" in str(_es_exc), (
                _es_exc
            )
            assert "end_sheath_full_debit is RETIRED" in str(_es_exc), _es_exc
            assert "cathode face" in str(_es_exc), _es_exc
            assert "end wall face" in str(_es_exc), _es_exc
        else:
            raise AssertionError(
                f"end_sheath_full_debit={_es_value} was accepted"
            )
    # ... and it is gone from the templates, so default_config() carries no
    # dead key that a reader could take for a live control.
    _es_default_params, _es_default_flags = default_config()
    assert "end_sheath_full_debit" not in _es_default_flags
    assert "end_sheath_full_debit" not in _es_default_params
    assert "end_wall_sheath_full_debit" not in _es_default_flags
    assert "cathode_face_full_debit" not in _es_default_flags
    assert "cathode_face_full_debit" not in _es_default_params


# --------------------------------------------------------------------
# end-wall-debit-armed-by-geometry-role
# --------------------------------------------------------------------
@_case("end-wall-debit-armed-by-geometry-role", historical_stance=True)
def _case_end_wall_debit_armed_by_geometry_role():
    # THE GEOMETRY KEYS ARE RETIRED and the end wall sheath debit is armed by
    # presence on ROLE: exactly when the geometry has a plasma-absorbing face
    # whose live cell carries the end wall role, never by a flag.
    from stance_config import load_configuration

    # (a) Every retired geometry key refuses at the configuration boundary
    # with the retired-key ValueError, in its own namespace.
    _rg_p, _rg_f = default_config()
    for _rg_ns, _rg_key, _rg_value in (
        ("flags", "end_expansion_geometry", False),
        ("flags", "prescribed_area_geometry", True),
        ("flags", "neutral_baffles", True),
        ("flags", "end_wall_sheath_full_debit", True),
        ("flags", "source_fixed_grid", True),
        ("params", "end_expansion_cells", 10),
        ("params", "end_expansion_machine_radius_cm", 100.0),
        ("params", "end_expansion_plasma_radius_cm", 50.0),
    ):
        assert _rg_key not in _rg_p and _rg_key not in _rg_f, _rg_key
        _rg_bad_p, _rg_bad_f = dict(_rg_p), dict(_rg_f)
        (_rg_bad_f if _rg_ns == "flags" else _rg_bad_p)[_rg_key] = _rg_value
        try:
            LAPDSim1D(_rg_bad_p, _rg_bad_f)
        except ValueError as _rg_exc:
            assert "unknown LAPDSim1D configuration keys" in str(_rg_exc), (
                _rg_exc
            )
            assert f"{_rg_key} is RETIRED" in str(_rg_exc), _rg_exc
        else:
            raise AssertionError(f"retired key {_rg_key} was accepted")

    # (b) A TwinCathode geometry, circuit off, as its fixtures build it: it
    # constructs on its mirrored fixed-source mesh, has no end wall face, and
    # so has no end wall debit armed and no end wall row in its ledger.
    _rg_res_p, _rg_res_f = _resolved_config()
    _rg_twin_p = dict(
        _rg_res_p,
        cathode_anode_gap_cm=50.0,
        source_region_length_cm=100.0,
        source_region_dz_cm=10.0,
        gas_puff_z_cm=60.0,
    )
    _rg_twin_f = dict(_rg_res_f, TwinCathode=True, cathode_coupling=False)
    _rg_twin = LAPDSim1D(_rg_twin_p, _rg_twin_f)
    assert "end_wall" not in absorbing_live_cells_by_role(_rg_twin.geometry)
    assert _rg_twin._end_wall_sheath_full_debit is False
    # ... and with the circuit off neither cathode face emits, so neither
    # books the cathode face's sheath rows.
    assert _rg_twin._cathode_face_full_debit is False
    assert not set(END_SHEATH_CATHODE_ROWS) & set(_rg_twin.rhs_terms())
    assert not set(END_SHEATH_END_WALL_ROWS) & set(_rg_twin.rhs_terms())

    # (c) The reference geometry has an end wall face and arms the debit.
    _rg_ref_p, _rg_ref_f, _ = load_configuration("g1atrim")
    _rg_ref = LAPDSim1D(_rg_ref_p, _rg_ref_f)
    assert absorbing_live_cells_by_role(_rg_ref.geometry).get("end_wall")
    assert _rg_ref._end_wall_sheath_full_debit is True
    # The reference's cathode face emits through the circuit solve and arms
    # the cathode face's sheath rows the same way.
    assert absorbing_live_cells_by_role(_rg_ref.geometry).get("cathode")
    assert _rg_ref._cathode_face_full_debit is True


# --------------------------------------------------------------------
# end-wall-lambda-eff-barrier-bracket
# --------------------------------------------------------------------
@_case("end-wall-lambda-eff-barrier-bracket", historical_stance=True)
def _case_end_wall_lambda_eff_barrier_bracket():
    # LAMBDA_EFF IS A STATE-DEPENDENT BARRIER AND ITS RANGE IS [Lambda,
    # Lambda + 1/2]. The end wall row books -Lambda_eff Te Gamma_coll beside
    # the boundary term's unconditional 2 Te on the same face and the same
    # flux, so the run's own two rows read the barrier back exactly:
    #
    #     Lambda_eff = 2 * end_wall_e_sheath_climb / characteristic_boundary
    #
    # at the end wall cell. Lambda_eff = Lambda + ln(1/alpha_se), and
    # alpha_se runs between exp(-1/2) (the presheath fits inside the sampling
    # cell: the cell is at the sheath edge and carries the whole Boltzmann
    # drop) and 1 (the presheath is longer than the cell: the cell sits inside
    # it, has already dropped part of the way from the reservoir, and its
    # electrons climb only the remainder). So the barrier moves with the
    # state, and a reading below Lambda + 1/2 in a cold thin end cell is the
    # second regime rather than a defect.
    #
    # The bracket is what is asserted here, not a value -- pinning a value
    # would pin a state.
    _le_params, _le_flags = _base_config()
    _le_flags = dict(_le_flags)
    _le_flags["cathode_coupling"] = True
    # This case reads a barrier off two RHS rows, not a neutral profile, and
    # ``run()`` performs no equilibration -- so the equilibration flag is
    # cleared rather than left on to warn that it did nothing.
    _le_params["initial_neutral_state"] = "fill"
    _le_sim = LAPDSim1D(dict(_le_params), _le_flags)
    _le_result = _le_sim.run(t_end=4.0e-10, dt=1.0e-10)

    _le_lambda = sheath_lift_lambda(_le_sim.ion_mass_g)
    _le_geom = _le_sim.geometry
    _le_roles = np.asarray(_le_geom.cell_role)
    _le_coll = int(np.flatnonzero(_le_roles == "end_wall")[0])
    _le_climb = np.asarray(
        _le_result.rhs_terms["end_wall_e_sheath_climb"]["Ee"], dtype=float
    )[:, _le_coll]
    _le_char = np.asarray(
        _le_result.rhs_terms["characteristic_boundary"]["Ee"], dtype=float
    )[:, _le_coll]
    _le_nn = np.asarray(_le_result.nn, dtype=float)[:, _le_coll]
    _le_Te = np.asarray(_le_result.Te, dtype=float)[:, _le_coll]
    _le_Ti = np.asarray(_le_result.Ti, dtype=float)[:, _le_coll]
    _le_len = float(_le_geom.length_cm[_le_coll])
    assert _le_climb.size >= 2, _le_climb.size

    # Both rows must be live on every save, or the ratio would be reading
    # nothing: the boundary row carries the 2 Te of the same collected flux.
    assert np.all(_le_char < 0.0), _le_char
    assert np.all(_le_climb < 0.0), _le_climb

    _le_read = 2.0 * _le_climb / _le_char
    for _le_i in range(_le_read.size):
        # (i) THE BRACKET, on every save.
        assert _le_lambda <= _le_read[_le_i] <= _le_lambda + 0.5, (
            _le_i, _le_read[_le_i], _le_lambda
        )
        # (ii) ... and it is the code's own alpha that puts it there, read
        # off the same saved state the boundary operator sampled its flux at.
        _le_alpha = electrode_sheath_alpha(
            nn=float(_le_nn[_le_i]),
            Te=float(_le_Te[_le_i]),
            Ti=float(_le_Ti[_le_i]),
            cell_length_cm=_le_len,
            ion_mass_g=_le_sim.ion_mass_g,
            alpha_isat=float(_le_params["alpha_isat"]),
            b_presheath_length=float(_le_params["b_presheath_length"]),
        )
        assert math.exp(-0.5) <= _le_alpha <= 1.0, (_le_i, _le_alpha)
        assert np.isclose(
            _le_read[_le_i],
            _le_lambda - math.log(_le_alpha),
            rtol=1e-11,
            atol=0.0,
        ), (_le_i, _le_read[_le_i], _le_alpha)

# ----------------------------------------------------------------------
# end-wall-rename-retired-names
# ----------------------------------------------------------------------
@_case("end-wall-rename-retired-names")
def _case_end_wall_rename_retired_names():
    # THE END-WALL RENAME. The model's far face IS the LAPD chamber's end
    # wall -- there is no distinct collector electrode -- so the cell role and
    # every configuration key that carried the old ``collector`` name were
    # renamed to ``end_wall``. Two halves are asserted here:
    #
    #   IN: a LIVE configuration naming a retired key, ``end_mode``
    #       included, is REFUSED at construction with the replacement
    #       named. A quietly accepted alias would be exactly the
    #       silent/inert control the config boundary exists to forbid.
    #   BACK: a SAVED artifact written before the rename still reads, because
    #       ``load_result_hdf5`` maps the stored ``cell_role`` string and says
    #       that it did.
    from cablp.solvers._sim1d.core.config import (
        LEGACY_CONFIG_KEY_ALIASES,
        RETIRED_FLAG_KEYS,
        RETIRED_PARAM_KEYS,
        apply_legacy_config_key_aliases,
        input_dict_template_1d,
        input_flags_template_1d,
    )
    from cablp.solvers._sim1d.results.io import (
        CELL_ROLE_SHIM_ATTR,
        LEGACY_CELL_ROLE_ALIASES,
        _apply_cell_role_aliases,
    )

    _ew_p, _ew_f = default_config()

    # (a) EVERY retired param key refuses in input_dict, naming its successor.
    _ew_retired_params = (
        ("collector_length_cm", 7.8, "end_wall_length_cm"),
        ("neutral_kinetic_dvm_collector_jet", True,
         "neutral_kinetic_dvm_end_wall_jet"),
        ("neutral_kinetic_dvm_collector_jet_R_N", 0.5,
         "neutral_kinetic_dvm_end_wall_jet_R_N"),
        ("neutral_kinetic_dvm_collector_jet_R_E", 0.6,
         "neutral_kinetic_dvm_end_wall_jet_R_E"),
        ("neutral_kinetic_dvm_collector_jet_T_launch_eV", 0.2,
         "neutral_kinetic_dvm_end_wall_jet_T_launch_eV"),
        ("neutral_kinetic_dvm_collector_jet_sheath_Te_multiple", 3.0,
         "neutral_kinetic_dvm_end_wall_jet_sheath_Te_multiple"),
    )
    for _ew_old, _ew_value, _ew_new in _ew_retired_params:
        assert _ew_old not in input_dict_template_1d, _ew_old
        assert _ew_old in RETIRED_PARAM_KEYS, _ew_old
        assert _ew_new in input_dict_template_1d, _ew_new
        try:
            LAPDSim1D(dict(_ew_p, **{_ew_old: _ew_value}), _ew_f)
        except ValueError as _ew_exc:
            _ew_msg = str(_ew_exc)
        else:
            raise AssertionError(f"{_ew_old} was ACCEPTED in input_dict")
        assert "unknown LAPDSim1D configuration keys" in _ew_msg, _ew_msg
        assert f"{_ew_old} is RETIRED" in _ew_msg, _ew_msg
        assert _ew_new in _ew_msg, _ew_msg

    # (b) the retired FLAG key, same statement in the other namespace.
    assert "collector_sheath_full_debit" not in input_flags_template_1d
    assert "collector_sheath_full_debit" in RETIRED_FLAG_KEYS
    assert "end_wall_sheath_full_debit" not in input_flags_template_1d
    try:
        LAPDSim1D(_ew_p, dict(_ew_f, collector_sheath_full_debit=True))
    except ValueError as _ew_fexc:
        _ew_fmsg = str(_ew_fexc)
    else:
        raise AssertionError(
            "collector_sheath_full_debit was ACCEPTED in input_flags"
        )
    assert "unknown LAPDSim1D configuration keys" in _ew_fmsg, _ew_fmsg
    assert "collector_sheath_full_debit is RETIRED" in _ew_fmsg, _ew_fmsg
    assert "end wall face" in _ew_fmsg, _ew_fmsg

    # (c) end_mode itself. The far face is the end wall unconditionally, so
    # the key is retired and every value of it, the old 'collector' included,
    # is refused with the retired-key message.
    for _ew_mode in ("collector", "end_wall"):
        try:
            LAPDSim1D(dict(_ew_p, end_mode=_ew_mode), _ew_f)
        except ValueError as _ew_vexc:
            _ew_vmsg = str(_ew_vexc)
        else:
            raise AssertionError(f"end_mode={_ew_mode!r} was ACCEPTED")
        assert "end_mode is RETIRED" in _ew_vmsg, _ew_vmsg

    # (d) THE READ SHIM. A stored role array is mapped, and the load reports
    # that it was; an array carrying none of the retired strings is returned
    # untouched and reports False.
    assert LEGACY_CELL_ROLE_ALIASES == {"collector": "end_wall"}
    _ew_old_roles = np.asarray(
        ["plenum", "cathode", "column", "collector"], dtype=object
    )
    _ew_mapped, _ew_fired = _apply_cell_role_aliases(_ew_old_roles)
    assert _ew_fired is True
    assert list(_ew_mapped) == ["plenum", "cathode", "column", "end_wall"]
    # the input is not mutated -- a caller holding the stored array keeps it
    assert list(_ew_old_roles) == [
        "plenum", "cathode", "column", "collector"
    ]
    _ew_new_roles = np.asarray(
        ["plenum", "cathode", "column", "end_wall"], dtype=object
    )
    _ew_kept, _ew_quiet = _apply_cell_role_aliases(_ew_new_roles)
    assert _ew_quiet is False
    assert _ew_kept is _ew_new_roles
    assert CELL_ROLE_SHIM_ATTR == "cell_role_legacy_alias_applied"

    # (e) THE SAVED-BLOCK KEY MAP, the read-side counterpart of (a)/(b). It
    # is for STORED blocks only and is never consulted by resolve_config --
    # which is what (a) and (b) just proved, since a mapped key would have
    # been accepted there.
    _ew_stored = {
        "collector_length_cm": 7.8,
        "collector_sheath_full_debit": True,
        "end_mode": "collector",
        "nx": 60,
    }
    _ew_current = apply_legacy_config_key_aliases(_ew_stored)
    assert _ew_current == {
        "end_wall_length_cm": 7.8,
        "end_wall_sheath_full_debit": True,
        "end_mode": "end_wall",
        "nx": 60,
    }, _ew_current
    # presence-gated: a block naming none of them is an unchanged copy
    _ew_plain = {"nx": 60, "end_mode": "end_wall"}
    assert apply_legacy_config_key_aliases(_ew_plain) == _ew_plain
    # and every retired name in the map is refused live, so the two tables
    # cannot drift apart into an accepted alias
    for _ew_old in LEGACY_CONFIG_KEY_ALIASES:
        assert (
            _ew_old in RETIRED_PARAM_KEYS or _ew_old in RETIRED_FLAG_KEYS
        ), _ew_old


# ----------------------------------------------------------------------
# end-wall-face-sheath-edge-flux
# ----------------------------------------------------------------------
@_case("end-wall-face-sheath-edge-flux", historical_stance=True)
def _case_end_wall_face_sheath_edge_flux():
    """The end wall removes the PHYSICAL flux at the sheath-edge state.

    (i) The particle sink is A_face alpha_se n c_s exactly -- no dissipative
    share on top of it -- and the momentum and ion-energy rows are that same
    face state's physical fluxes, while the two electron rows are their
    per-particle bookings times the one particle flux.
    (ii) Every plasma particle the face absorbs comes back as a neutral in
    the same cell.
    (iii) A stepped run stays positive and finite at the wall.
    """
    from cablp.solvers._sim1d.physics import flux as _ew_flux
    from cablp.solvers._sim1d.physics.sources import absorbing_face_states

    _ew_params, _ew_flags = _base_config()
    _ew_params = dict(_ew_params, max_steps_action="stop")
    _ew_flags = dict(_ew_flags)
    # run() is called directly below, so the equilibration pre-solve is
    # cleared rather than left on to warn that it did nothing.
    _ew_params["initial_neutral_state"] = "fill"
    _ew_lambda = sheath_lift_lambda(m_He_cgs)

    _ew_sim = LAPDSim1D(dict(_ew_params), dict(_ew_flags))
    _ew_geom = _ew_sim.geometry
    _ew_cell = int(absorbing_live_cells_by_role(_ew_geom)["end_wall"][0])
    _ew_faces = np.flatnonzero(
        np.asarray(_ew_geom.plasma_absorbing, dtype=bool)
    )
    _ew_face = int(
        [
            f for f in _ew_faces
            if int(_ew_geom.plasma_face_live_cell[f]) == _ew_cell
        ][0]
    )
    _ew_area = float(np.asarray(_ew_geom.plasma_face_area_cm2)[_ew_face])
    _ew_Vp = float(np.asarray(_ew_geom.plasma_volume_cm3)[_ew_cell])
    _ew_Vn = float(
        np.asarray(
            _ew_geom.plasma_volume_cm3
            if _ew_sim.state.nn_a is not None
            else _ew_geom.neutral_volume_cm3
        )[_ew_cell]
    )
    _ew_state = _ew_sim.state
    _ew_derived = derive_state(
        _ew_state, floors=_ew_sim.floors, ion_mass_g=_ew_sim.ion_mass_g
    )
    _ew_outward = -1.0 if _ew_cell == _ew_face else 1.0
    _ew_interior, _ew_ghost, _ew_alpha = absorbing_face_states(
        state=_ew_state,
        derived=_ew_derived,
        geometry=_ew_geom,
        live=_ew_cell,
        outward=_ew_outward,
        ion_mass_g=_ew_sim.ion_mass_g,
        alpha_isat=float(np.exp(-0.5)),
        b_presheath_length=float(
            _ew_sim._input_dict["b_presheath_length"]
        ),
    )
    _ew_Te = float(_ew_derived.Te[_ew_cell])
    _ew_Ti = float(_ew_derived.Ti[_ew_cell])
    _ew_cs = float(_ew_flux.ion_sound_speed(_ew_Te, _ew_sim.ion_mass_g))
    _ew_n_se = _ew_alpha * float(_ew_state.n[_ew_cell])

    _ew_climb_out = {}
    _ew_bnd = _ew_sim.characteristic_boundary_rhs(
        state=_ew_state, end_wall_climb_out=_ew_climb_out
    )

    # (i) THE PARTICLE SINK IS THE BOHM FLUX THROUGH THE FACE, and nothing
    # else: a Rusanov combination against this same ghost would add its
    # dissipative -a_max (n_R - n_L)/2 on top of it.
    _ew_removed = -float(np.asarray(_ew_bnd.n)[_ew_cell]) * _ew_Vp
    _ew_want = _ew_area * _ew_n_se * _ew_cs
    assert abs(_ew_removed / _ew_want - 1.0) <= 1.0e-14, (
        _ew_removed, _ew_want
    )

    # The momentum and ion-energy rows are the SAME face state's physical
    # fluxes through the same face, on the same one-sided divergence.
    _ew_scale = (1.0 if _ew_cell == _ew_face else -1.0) * _ew_area / _ew_Vp
    _ew_u = _ew_outward * _ew_cs
    _ew_f_M = (
        _ew_sim.ion_mass_g * _ew_n_se * _ew_u * _ew_u
        + _ew_n_se * (_ew_Te + _ew_Ti) * ev_to_erg
    )
    _ew_f_Ei = 1.5 * _ew_n_se * _ew_Ti * ev_to_erg * _ew_u
    assert abs(
        float(np.asarray(_ew_bnd.M)[_ew_cell]) / (_ew_scale * _ew_f_M) - 1.0
    ) <= 1.0e-14, np.asarray(_ew_bnd.M)[_ew_cell]
    assert abs(
        float(np.asarray(_ew_bnd.Ei)[_ew_cell]) / (_ew_scale * _ew_f_Ei) - 1.0
    ) <= 1.0e-14, np.asarray(_ew_bnd.Ei)[_ew_cell]

    # The two electron rows are per-particle bookings on that one flux:
    # 2 Te unconditionally, and Lambda_eff Te for the sheath the electrons
    # climbed, with Lambda_eff = Lambda + ln(1/alpha_se) at THIS face's own
    # sampling factor.
    _ew_sink = float(np.asarray(_ew_bnd.n)[_ew_cell])
    assert abs(
        float(np.asarray(_ew_bnd.Ee)[_ew_cell])
        / (2.0 * _ew_Te * ev_to_erg * _ew_sink) - 1.0
    ) <= 1.0e-14
    _ew_lambda_eff = _ew_lambda - math.log(_ew_alpha)
    assert abs(
        float(np.asarray(_ew_climb_out["Ee"])[_ew_cell])
        / (_ew_lambda_eff * _ew_Te * ev_to_erg * _ew_sink) - 1.0
    ) <= 1.0e-14

    # (ii) THE RECYCLE COUNT IS THE ION COUNT: every particle removed is
    # rebirthed as a neutral in the same cell.
    assert abs(
        float(np.asarray(_ew_bnd.nn)[_ew_cell]) * _ew_Vn / _ew_removed - 1.0
    ) <= 1.0e-12

    # (iii) A STEPPED RUN STAYS POSITIVE AND FINITE AT THE WALL.
    _ew_run = LAPDSim1D(dict(_ew_params), dict(_ew_flags))
    _ew_res = _ew_run.run(t_end=None, dt=None, max_steps=25)
    _ew_n = np.asarray(_ew_run.state.n, dtype=float)
    assert np.all(np.isfinite(_ew_n)) and np.all(_ew_n > 0.0)
    assert np.all(np.isfinite(np.asarray(_ew_run.state.Ee, dtype=float)))
    assert np.all(np.isfinite(np.asarray(_ew_run.state.Ei, dtype=float)))
    # The upstream neighbour is on the side the face is NOT on.
    _ew_prev = _ew_cell + 1 if _ew_cell == _ew_face else _ew_cell - 1
    print(
        "  end-wall face: n_wall/n_prev = "
        f"{_ew_n[_ew_cell] / _ew_n[_ew_prev]:.6f} over "
        f"{_ew_res.steps} steps"
    )
