"""Smoke cases: configuration files, derived configurations, key namespaces and
construction refusals.
"""

import argparse
import contextlib
from io import StringIO
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import h5py

from cablp.solvers._sim1d import LAPDSim1D, default_config, load_result_hdf5

from ._harness import _case


# --------------------------------------------------------------------
# production-construction-warning-free
# --------------------------------------------------------------------
@_case(
    "production-construction-warning-free",
    provides=("_warnings",),
)
def _case_production_construction_warning_free():
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

    # NEGATIVE CONTROL for the b_* removal (2026-08-28). The legacy rate,
    # cooling, conduction and anode scale factors are GONE from the config
    # surface; a caller that still supplies one must be told loudly at
    # construction, not silently ignored. Checked one key at a time so a
    # partially-completed removal cannot hide behind another name, and in the
    # params namespace only -- these were never flags.
    for _removed in (
        "b_ioniz", "b_Qei", "b_Qen", "b_Qcx", "b_Qie",
        "b_rec_rad", "b_rec_3b", "b_Qei_Te_exp", "b_Qen_Te_exp",
        "b_Q_Te_ref_eV", "b_epara", "b_ipara", "b_slip_entrainment",
        "b_anode_collection", "b_anode_advective_block",
    ):
        assert _removed not in _dep_params, _removed
        _stale_params, _stale_flags = default_config()
        _stale_params[_removed] = 1.0
        try:
            LAPDSim1D(_stale_params, _stale_flags)
        except ValueError as _exc:
            assert "unknown LAPDSim1D configuration keys" in str(_exc), _exc
            assert _removed in str(_exc), _exc
        else:
            raise AssertionError(
                f"removed config key {_removed!r} was accepted silently"
            )
    return locals()


# --------------------------------------------------------------------
# config-key-namespace-and-seed-cache
# --------------------------------------------------------------------
@_case("config-key-namespace-and-seed-cache")
def _case_config_key_namespace_and_seed_cache():
    # ---- the registered constants are exactly the registered constants ----
    # A key added to the wrong namespace silently does nothing (input_dict and
    # input_flags validate neither), so the split is asserted here.
    _r2_reg_p, _r2_reg_f = default_config()
    for _key in (
        "tracer_passivity_current_ratio",
        "tracer_passivity_thinness",
        "tracer_passivity_depletion",
        "tracer_passivity_hysteresis",
        "tracer_refresh_tol",
        "tracer_activation_ne",
        "tracer_overlap_band_ne",
        "tracer_overlap_rtol",
    ):
        assert _key in _r2_reg_p and _key not in _r2_reg_f, _key
    assert "regime_tracer" in _r2_reg_f and "regime_tracer" not in _r2_reg_p
    assert _r2_reg_f["regime_tracer"] is False, "regime_tracer must ship OFF"
    for _key in ("nn0_profile", "nn0_annulus_profile"):
        assert _key in _r2_reg_p and _key not in _r2_reg_f, _key
        assert _r2_reg_p[_key] is None, "a shaped IC ships no shape"
    assert (
        "initial_neutral_state" in _r2_reg_p
        and "initial_neutral_state" not in _r2_reg_f
    )
    assert _r2_reg_p["initial_neutral_state"] == "equilibrate", (
        "the shaped-fill route must ship OFF"
    )
    for _key in (
        "plasma_radius_profile_cm",
        "machine_radius_profile_cm",
        "plasma_area_max_vessel_fraction",
        "neutral_annulus_volume_fraction_min",
    ):
        assert _key in _r2_reg_p and _key not in _r2_reg_f, _key
    for _key in (
        "plasma_radius_profile_cm",
        "machine_radius_profile_cm",
        "plasma_area_max_vessel_fraction",
    ):
        assert _r2_reg_p[_key] is None, "a per-cell geometry ships no shape"
    # The sliver guard is the one key here that is NOT presence-gated on the
    # plasma profile -- it constrains any two-zone geometry -- so it ships a value, and
    # that value must be inert on a straight column (which leaves ~0.86) while
    # still above a capped 0.95-of-bore flux tube (0.05).
    assert 0.0 < _r2_reg_p["neutral_annulus_volume_fraction_min"] < 0.05
    # The prescribed profiles change the GEOMETRY, so they must re-key the
    # equilibrated neutral seed: no key may sit on the seed cache's inert
    # allowlists (the fail-closed rule -- a key leaves the hash only when it
    # provably cannot reach an equilibration, and Vp / V_ann / the zone
    # exchange conductance all can).
    from cablp.solvers._sim1d.core import neutral_seed_cache as _seed_cache_mod

    for _key in (
        "plasma_radius_profile_cm",
        "machine_radius_profile_cm",
        "plasma_area_max_vessel_fraction",
        "neutral_annulus_volume_fraction_min",
    ):
        assert _key not in _seed_cache_mod.INERT_PARAM_KEYS, _key
    _pa_sig_p, _pa_sig_f = default_config()
    _pa_sig_flare_f = dict(_pa_sig_f)
    for _key, _value in (
        ("plasma_radius_profile_cm", [18.415] * 3),
        ("machine_radius_profile_cm", [50.0] * 3),
        ("plasma_area_max_vessel_fraction", 0.95),
    ):
        _pa_sig_flare_p = dict(_pa_sig_p)
        _pa_sig_flare_p[_key] = _value
        assert _seed_cache_mod.neutral_seed_signature(
            _pa_sig_p, _pa_sig_f
        ) != _seed_cache_mod.neutral_seed_signature(
            _pa_sig_flare_p, _pa_sig_flare_f
        ), f"{_key} must invalidate a cached neutral seed"

    # ---- the cathode end-face booking flag is OUT of the signature ----
    # The other direction of the same fail-closed rule. It CANNOT
    # reach an equilibrated seed -- run_neutral_equilibration clears it
    # on the inner sim's config before it builds it -- so hashing it would
    # rotate every stored seed with no neutral content behind the
    # invalidation. The reference is loaded through build_baseline_config(),
    # so this tracks the stance of record instead of pinning a config.
    from baseline_sim1d import build_baseline_config as _sf_baseline_config

    _sf_keys = ("cathode_face_full_debit",)
    for _key in _sf_keys:
        assert _key in _seed_cache_mod.INERT_FLAG_KEYS, _key
        assert _key not in _seed_cache_mod.INERT_PARAM_KEYS, _key

    _sf_p, _sf_f = _sf_baseline_config()
    _sf_sig = _seed_cache_mod.neutral_seed_signature(_sf_p, _sf_f)
    for _key in _sf_keys:
        assert _key in _sf_f and _key not in _sf_p, _key
        _sf_toggled = {}
        for _value in (False, True):
            _sf_alt_f = dict(_sf_f)
            _sf_alt_f[_key] = _value
            _sf_toggled[_value] = _seed_cache_mod.neutral_seed_signature(
                _sf_p, _sf_alt_f
            )
        assert _sf_toggled[False] == _sf_toggled[True] == _sf_sig, (
            f"{_key} must NOT re-key the equilibrated neutral seed"
        )

    # ...and the exemption really did rotate the reference's signature once,
    # which is the disclosed cost of the change. Base is reconstructed by
    # importing the module source under its OWN name from a temp copy and
    # putting the key back: an isolated module object, so restoring
    # the pre-exemption key set cannot leak into any later case. Both
    # signatures are computed live -- pinning either hex would make this
    # clause stale at the next stance event, since every hashed key moves it.
    import importlib.util as _sf_importlib_util

    with tempfile.TemporaryDirectory() as _sf_dir:
        _sf_copy = os.path.join(_sf_dir, "neutral_seed_cache_base.py")
        shutil.copy2(_seed_cache_mod.__file__, _sf_copy)
        # The dotted name keeps __package__ on the real package, so the
        # module's relative import of SCCM_TO_PARTICLES_PER_S resolves; the
        # module object is never put in sys.modules, so nothing else sees it.
        _sf_spec = _sf_importlib_util.spec_from_file_location(
            "cablp.solvers._sim1d.core._smoke_neutral_seed_cache_base",
            _sf_copy,
        )
        _sf_base_mod = _sf_importlib_util.module_from_spec(_sf_spec)
        _sf_spec.loader.exec_module(_sf_base_mod)
        assert _sf_base_mod is not _seed_cache_mod
        _sf_base_mod.INERT_FLAG_KEYS = frozenset(
            _sf_base_mod.INERT_FLAG_KEYS - set(_sf_keys)
        )
        _sf_base_sig = _sf_base_mod.neutral_seed_signature(_sf_p, _sf_f)
    assert _sf_base_sig != _sf_sig, (
        "exempting the flag must rotate the reference's seed signature"
    )


# --------------------------------------------------------------------
# golden-baseline-config-constructs
# --------------------------------------------------------------------
@_case("golden-baseline-config-constructs")
def _case_golden_baseline_config_constructs():
    # THE GOLDEN'S OWN CONFIG MUST CONSTRUCT, and so must the neutral
    # equilibration pre-solve it stands on. This case TRACKS THE STANCE OF
    # RECORD by construction -- it builds no config of its own, it calls
    # baseline_sim1d.build_baseline_config(), so it follows every stance event
    # automatically and never needs re-pinning.
    #
    # Why the pre-solve half exists. The golden's re-cut drops the stance's
    # mesh-sized package and arms neutral_equilibration instead, so
    # start_simulation() builds an INNER neutrals-only LAPDSim1D whose flags
    # are not the outer run's. Constructing the outer sim alone does not reach
    # that inner construction, and the gap was not hypothetical: at the
    # 2026-09-02 kinetic stance event the outer sim constructed fine while the
    # inner one raised on two DVM jet guards, and the failure surfaced only
    # after the capture had been started.
    #
    # NO SOLVE. t_end = 0.0 runs the pre-solve to zero steps, so this exercises
    # the inner construction and nothing else; the whole case is a fraction of
    # a second. It deliberately does NOT assert what the equilibrated seed
    # contains -- that is the golden fixture's job.
    from baseline_sim1d import build_baseline_config

    _gb_params, _gb_flags = build_baseline_config()
    _gb_sim = LAPDSim1D(_gb_params, _gb_flags)
    _gb_result = _gb_sim.run_neutral_equilibration(t_end=0.0)
    assert _gb_result.time[-1] == 0.0

    # The pre-solve clears controls on ITS OWN copy of the config. Assert the
    # outer run is unchanged, because the outer run is what the fixture is
    # captured from: get_config() hands out copies, and a regression that made
    # it hand out the live dicts would silently disarm the golden's jets.
    for _gb_key in (
        "neutral_kinetic_dvm_cathode_jet",
        "neutral_kinetic_dvm_anode_jet",
    ):
        assert (
            _gb_sim._input_dict[_gb_key] == _gb_params[_gb_key]
        ), f"pre-solve mutated the outer run's {_gb_key}"
    for _gb_flag in ("Plasma", "cathode_coupling"):
        assert _gb_sim._flags[_gb_flag] == _gb_flags[_gb_flag], (
            f"pre-solve mutated the outer run's {_gb_flag}"
        )


# --------------------------------------------------------------------
# DERIVED CONFIGURATIONS (no default plasma: every run names its configuration).
#
# Every run names a configuration; an alternate the campaign runs against the
# reference is a FILE (base + declared deltas), not a command line. The four
# cases below hold the loader's contract: a derived file resolves to exactly
# what the equivalent hand-built configuration resolves to, and the three
# refusals that keep a derived file honest fire.
#
# Each fixture chain is built in a TEMPORARY stance directory -- a base
# resolves in ``stance_config.STANCE_DIR`` by definition, so the module global
# is redirected for the duration and restored after; nothing is written into
# the committed ``scripts/stances/``.
# --------------------------------------------------------------------
@contextlib.contextmanager
def _derived_fixture_dir():
    """Yield a temporary stance directory holding a copy of the reference."""
    import stance_config as _sc

    original = _sc.STANCE_DIR
    with tempfile.TemporaryDirectory() as _tmp:
        room = Path(_tmp)
        shutil.copy2(original / "g1atrim.toml", room / "g1atrim.toml")
        _sc.STANCE_DIR = room
        try:
            yield _sc, room
        finally:
            _sc.STANCE_DIR = original


@_case("configuration-derived-resolution")
def _case_configuration_derived_resolution():
    # A DERIVED configuration resolves to the same configuration a caller would
    # have built by hand from the base plus the same overrides -- same resolved
    # values, same identity. That equivalence is the form's whole claim: the
    # file is a way of NAMING a configuration, never a different one.
    from cablp.solvers._sim1d import config_identity, default_config

    with _derived_fixture_dir() as (_sc, _room):
        (_room / "derived.toml").write_text(
            'base = "g1atrim"\n'
            "\n"
            "[input_dict]\n"
            "nx = 42\n"
            "S_gp = 1234.0\n"
            "\n"
            "[input_flags]\n"
            "cathode_face_full_debit = true\n"
        )
        _dv_params, _dv_flags, _dv_lineage = _sc.load_configuration("derived")

        # The hand-built equivalent: default_config(), the base's whole delta,
        # then the same three overrides.
        _hand_p, _hand_f = default_config()
        _base = _sc.load_stance("g1atrim")
        _hand_p.update(_base.params)
        _hand_f.update(_base.flags)
        _hand_p["nx"] = 42
        _hand_p["S_gp"] = 1234.0
        _hand_f["cathode_face_full_debit"] = True

        assert _dv_params == _hand_p, sorted(
            k for k in set(_dv_params) | set(_hand_p)
            if _dv_params.get(k) != _hand_p.get(k)
        )
        assert _dv_flags == _hand_f
        assert _dv_lineage.identity == config_identity(_hand_p, _hand_f)

        # The lineage says what the file is, and says it by name only.
        assert _dv_lineage.name == "derived"
        assert _dv_lineage.base_chain == ("g1atrim",)
        assert len(_dv_lineage.file_sha256) == 2
        assert _dv_lineage.delta_keys == (
            "S_gp", "cathode_face_full_debit", "nx",
        ), _dv_lineage.delta_keys

        # NEGATIVE CONTROL. The identity is not a rubber stamp: move one delta
        # and it must move with it, or the check above proves nothing.
        (_room / "derived_moved.toml").write_text(
            'base = "g1atrim"\n'
            "\n"
            "[input_dict]\n"
            "nx = 42\n"
            "S_gp = 1235.0\n"
            "\n"
            "[input_flags]\n"
            "cathode_face_full_debit = true\n"
        )
        _, _, _moved = _sc.load_configuration("derived_moved")
        assert _moved.identity != _dv_lineage.identity


@_case("configuration-restated-delta-refusal")
def _case_configuration_restated_delta_refusal():
    # A delta must MOVE something. A line that restates the value it already
    # inherits reads as a decision and is not one, and it stops agreeing with
    # its base silently the first time the base moves.
    with _derived_fixture_dir() as (_sc, _room):
        _base_sgp = _sc.load_stance("g1atrim").params["S_gp"]
        (_room / "restated.toml").write_text(
            'base = "g1atrim"\n'
            "\n"
            "[input_dict]\n"
            f"S_gp = {_base_sgp!r}\n"
        )
        try:
            _sc.load_stance("restated")
        except ValueError as _rs_exc:
            assert "restates" in str(_rs_exc), str(_rs_exc)
            assert "S_gp" in str(_rs_exc), str(_rs_exc)
            assert "allow_restated" in str(_rs_exc), str(_rs_exc)
        else:
            raise AssertionError("a restated delta was ACCEPTED")

        # NEGATIVE CONTROL (a): the declared waiver accepts the same file.
        (_room / "restated_waived.toml").write_text(
            'base = "g1atrim"\n'
            "allow_restated = true\n"
            "\n"
            "[input_dict]\n"
            f"S_gp = {_base_sgp!r}\n"
        )
        assert _sc.load_stance("restated_waived").params["S_gp"] == _base_sgp

        # NEGATIVE CONTROL (b): a delta that MOVES the same key is accepted
        # without any waiver, so the refusal is about restatement and not
        # about the key.
        (_room / "moved.toml").write_text(
            'base = "g1atrim"\n'
            "\n"
            "[input_dict]\n"
            f"S_gp = {_base_sgp + 1.0!r}\n"
        )
        assert _sc.load_stance("moved").params["S_gp"] == _base_sgp + 1.0


@_case("configuration-chain-depth-refusal")
def _case_configuration_chain_depth_refusal():
    # Chains are allowed to MAX_CHAIN_FILES files and refused beyond: past that
    # depth a value cannot be traced to the file that chose it by reading.
    with _derived_fixture_dir() as (_sc, _room):
        assert _sc.MAX_CHAIN_FILES == 3, _sc.MAX_CHAIN_FILES
        (_room / "d2.toml").write_text(
            'base = "g1atrim"\n\n[input_dict]\nnx = 41\n'
        )
        (_room / "d3.toml").write_text(
            'base = "d2"\n\n[input_dict]\nnx = 42\n'
        )
        (_room / "d4.toml").write_text(
            'base = "d3"\n\n[input_dict]\nnx = 43\n'
        )

        # NEGATIVE CONTROL: the depth-3 chain is ACCEPTED, and carries the
        # whole chain in its lineage.
        _d3 = _sc.load_stance("d3")
        assert _d3.lineage.base_chain == ("d2", "g1atrim"), _d3.lineage.base_chain
        assert len(_d3.lineage.file_sha256) == 3
        assert _d3.params["nx"] == 42

        try:
            _sc.load_stance("d4")
        except ValueError as _cd_exc:
            assert "4 files deep" in str(_cd_exc), str(_cd_exc)
        else:
            raise AssertionError("a depth-4 chain was ACCEPTED")

        # A file that names itself is the same refusal about the same
        # structure, and must not be reachable by loading one more file.
        (_room / "loop.toml").write_text('base = "loop"\n\n[input_dict]\nnx = 44\n')
        try:
            _sc.load_stance("loop")
        except ValueError as _lp_exc:
            assert "its own base" in str(_lp_exc), str(_lp_exc)
        else:
            raise AssertionError("a self-referencing base was ACCEPTED")


@_case("configuration-unknown-key-delta-refusal")
def _case_configuration_unknown_key_delta_refusal():
    # A delta reaches the SAME refusals a base configuration's keys do: a key
    # no template owns, and a key filed in the wrong namespace. Neither may
    # survive as a silent inert control just because a base carried it.
    with _derived_fixture_dir() as (_sc, _room):
        (_room / "bogus.toml").write_text(
            'base = "g1atrim"\n\n[input_dict]\nnot_a_real_config_key = 1.0\n'
        )
        try:
            _sc.load_stance("bogus")
        except ValueError as _uk_exc:
            assert "not_a_real_config_key" in str(_uk_exc), str(_uk_exc)
            assert "no LAPDSim1D configuration template owns it" in str(_uk_exc)
        else:
            raise AssertionError("an unknown delta key was ACCEPTED")

        # Misfiled: an input_flags key stated in [input_dict]. The message
        # must name the namespace that DOES own it.
        (_room / "misfiled.toml").write_text(
            'base = "g1atrim"\n\n[input_dict]\ncathode_coupling = false\n'
        )
        try:
            _sc.load_stance("misfiled")
        except ValueError as _mf_exc:
            assert "input_flags" in str(_mf_exc), str(_mf_exc)
        else:
            raise AssertionError("a misfiled delta key was ACCEPTED")

        # NEGATIVE CONTROL: the correctly filed key in the correct namespace
        # loads, so the refusals above are about the key and not about the
        # derived form.
        (_room / "ok.toml").write_text(
            'base = "g1atrim"\n\n[input_flags]\ncathode_coupling = false\n'
        )
        assert _sc.load_stance("ok").flags["cathode_coupling"] is False


@_case("configuration-load-config-refuses-configuration-form")
def _case_configuration_load_config_refuses_configuration_form():
    # ``load_config`` reads [params]/[flags]/[models] and CANNOT resolve a
    # base -- the base directory is a scripts/ fact the solver package does not
    # know. Handed a configuration file it used to ignore every table it did
    # not recognise and resolve to bare defaults, which is the implied plasma
    # the ruling forbids. It refuses instead, naming the loader that can.
    from cablp.solvers._sim1d import load_config

    with tempfile.TemporaryDirectory() as _lc_tmp:
        _lc_room = Path(_lc_tmp)
        _lc_bad = _lc_room / "configuration_form.toml"
        _lc_bad.write_text('base = "g1atrim"\n\n[input_dict]\nnx = 42\n')
        try:
            load_config(_lc_bad)
        except ValueError as _lc_exc:
            assert "base" in str(_lc_exc), str(_lc_exc)
            assert "input_dict" in str(_lc_exc), str(_lc_exc)
            assert "stance_config" in str(_lc_exc), str(_lc_exc)
        else:
            raise AssertionError("a configuration-form file was ACCEPTED")

        # NEGATIVE CONTROL: the form load_config DOES own still loads.
        _lc_good = _lc_room / "params_form.toml"
        _lc_good.write_text("[params]\nnx = 42\n\n[flags]\ncathode_coupling = false\n")
        _lc_p, _lc_f = load_config(_lc_good)
        assert _lc_p["nx"] == 42 and _lc_f["cathode_coupling"] is False


@_case("configuration-hdf5-lineage-round-trip")
def _case_configuration_hdf5_lineage_round_trip():
    # A saved trajectory says WHICH configuration produced it. The lineage
    # round-trips through the HDF5 root attrs; a run that named none records
    # "<unnamed>"; and a file written before the attrs existed reads back None
    # for every one of them rather than a reconstructed answer.
    from cablp.solvers._sim1d import ConfigurationLineage, load_result_hdf5
    from cablp.solvers._sim1d.results.io import (
        UNNAMED_CONFIGURATION,
        _CONFIGURATION_ATTRS,
        save_result_hdf5,
    )

    _cl_lineage = ConfigurationLineage(
        name="smoke_derived",
        base_chain=("g1atrim",),
        file_sha256=("a" * 64, "b" * 64),
        delta_keys=("S_gp", "nx"),
        identity="c" * 64,
    )

    # NO SOLVE: t_end = 0.0 writes the initial state and stops, which is all a
    # metadata round-trip needs. The equilibration flag is cleared because
    # run() does not equilibrate and says so loudly; nothing here reads nn.
    def _cl_config():
        _p, _f = default_config()
        _p["initial_neutral_state"] = "fill"
        return _p, _f

    _cl_named = LAPDSim1D(*_cl_config(), configuration=_cl_lineage)
    assert _cl_named._configuration is _cl_lineage
    _cl_named_result = _cl_named.run(t_end=0.0)
    assert _cl_named_result.configuration is _cl_lineage

    _cl_unnamed_result = LAPDSim1D(*_cl_config()).run(t_end=0.0)
    assert _cl_unnamed_result.configuration is None

    with tempfile.TemporaryDirectory() as _cl_tmp:
        _cl_room = Path(_cl_tmp)

        _cl_named_h5 = _cl_room / "named.h5"
        save_result_hdf5(_cl_named_h5, _cl_named_result)
        _cl_back = load_result_hdf5(_cl_named_h5)
        assert _cl_back.configuration_name == "smoke_derived"
        assert _cl_back.configuration_base_chain == ["g1atrim"]
        assert _cl_back.configuration_file_sha256 == ["a" * 64, "b" * 64]
        assert _cl_back.configuration_delta_keys == ["S_gp", "nx"]
        assert _cl_back.configuration_identity == "c" * 64

        # A run that named no configuration says so, and says nothing else:
        # the four derived-form attrs are absent, not empty.
        _cl_unnamed_h5 = _cl_room / "unnamed.h5"
        save_result_hdf5(_cl_unnamed_h5, _cl_unnamed_result)
        with h5py.File(_cl_unnamed_h5, "r") as _cl_file:
            assert (
                _cl_file.attrs["configuration_name"] == UNNAMED_CONFIGURATION
            )
            for _cl_attr in _CONFIGURATION_ATTRS[1:]:
                assert _cl_attr not in _cl_file.attrs, _cl_attr
        _cl_unnamed_back = load_result_hdf5(_cl_unnamed_h5)
        assert _cl_unnamed_back.configuration_name == UNNAMED_CONFIGURATION
        for _cl_attr in _CONFIGURATION_ATTRS[1:]:
            assert getattr(_cl_unnamed_back, _cl_attr) is None, _cl_attr

        # LOAD -> SAVE MUST NOT DROP THE NAME. Re-saving a loaded artifact is
        # a routine step, and a named run that comes back "<unnamed>" is worse
        # than one that never carried a name: the loader reconstructs the
        # lineage and the writer carries it through unchanged.
        assert _cl_back.configuration == _cl_lineage, _cl_back.configuration
        _cl_resaved = _cl_room / "resaved.h5"
        save_result_hdf5(_cl_resaved, _cl_back)
        _cl_again = load_result_hdf5(_cl_resaved)
        for _cl_attr in _CONFIGURATION_ATTRS:
            assert getattr(_cl_again, _cl_attr) == getattr(_cl_back, _cl_attr), (
                _cl_attr
            )
        assert _cl_again.configuration == _cl_lineage

        # ... and an UNNAMED run stays unnamed rather than acquiring one.
        _cl_unnamed_resaved = _cl_room / "unnamed_resaved.h5"
        save_result_hdf5(_cl_unnamed_resaved, load_result_hdf5(_cl_unnamed_h5))
        with h5py.File(_cl_unnamed_resaved, "r") as _cl_file:
            assert (
                _cl_file.attrs["configuration_name"] == UNNAMED_CONFIGURATION
            )
            for _cl_attr in _CONFIGURATION_ATTRS[1:]:
                assert _cl_attr not in _cl_file.attrs, _cl_attr

        # NEGATIVE CONTROL (the pre-2026-09-03 file): strip every lineage attr
        # and the loader must report None for all five -- never "<unnamed>",
        # never an identity recomputed from params_json.
        _cl_old_h5 = _cl_room / "pre_lineage.h5"
        shutil.copy2(_cl_named_h5, _cl_old_h5)
        with h5py.File(_cl_old_h5, "a") as _cl_file:
            for _cl_attr in _CONFIGURATION_ATTRS:
                del _cl_file.attrs[_cl_attr]
        _cl_old_back = load_result_hdf5(_cl_old_h5)
        for _cl_attr in _CONFIGURATION_ATTRS:
            assert getattr(_cl_old_back, _cl_attr) is None, _cl_attr

    # A lineage is a record, not free text: the constructor refuses anything
    # that is not one, rather than storing a string that would later be
    # written into an artifact as if it named a committed file.
    try:
        LAPDSim1D(*_cl_config(), configuration="g1atrim")
    except ValueError as _cl_exc:
        assert "ConfigurationLineage" in str(_cl_exc), str(_cl_exc)
    else:
        raise AssertionError("a non-lineage configuration was ACCEPTED")


@_case("configuration-drivers-refuse-unnamed-runs")
def _case_configuration_drivers_refuse_unnamed_runs():
    # No driver runs an unnamed configuration. Each refusal is checked by
    # actually INVOKING the driver in a subprocess with no configuration named
    # -- not by reading its source -- because the failure this closes is a
    # command line that reads as a full package while standing on whatever the
    # shared driver dicts happened to hold, and only the real entry point can
    # say whether it still does.
    _dr_root = Path(__file__).resolve().parents[3]
    _dr_env = dict(os.environ)
    _dr_env["PYTHONPATH"] = str(_dr_root)

    def _dr_run(argv):
        return subprocess.run(
            [sys.executable, *argv],
            cwd=str(_dr_root), env=_dr_env,
            capture_output=True, text=True,
        )

    # Each row: the entry point's argv WITHOUT a configuration named, the
    # phrase its refusal must carry, and the switch that names one. The negative control
    # names a configuration that does not exist: that gets PAST the
    # missing-name refusal and dies on the unknown NAME instead, which is what
    # proves the first refusal is about the name being absent and not about
    # anything else on the command line. Neither invocation reaches a solve.
    _dr_missing = "no_such_configuration"
    _dr_drivers = (
        (["scripts/run/run_sim1d.py", "--output", "unused.h5"],
         "run_sim1d: name the configuration to run", "--config"),
        (["scripts/run/run_mechanism_ladder.py", "--es", "1",
          "--save-h5", "unused.h5"],
         "run_mechanism_ladder: name the configuration package", "--stance"),
        (["scripts/run/run_m6_point.py", "--es", "1", "--sgp", "9010",
          "--save-h5", "unused.h5"],
         "run_m6_point: name the configuration package", "--stance"),
        (["scripts/run/run_closure_ladder.py", "gen", "--root",
          str(Path(tempfile.gettempdir()) / "unused_closure_ladder_root")],
         "run_closure_ladder: name the configuration package", "--stance"),
        (["scripts/run/eqmap_make.py", "--out", "unused.npz"],
         "eqmap_make: name the configuration package", "--stance"),
        (["scripts/run/profile_sim1d.py", "--mode", "sample"],
         "profile_sim1d: name the configuration package", "--stance"),
        (["scripts/gates/audit_sim1d_equilibration_duty.py"],
         "audit_sim1d_equilibration_duty: name the configuration package",
         "--stance"),
        # The scorer's RUN route. Scoring --from-h5 is NOT covered and must not
        # be: it reads the configuration out of the artifact it scores, and a
        # name supplied there could only contradict it.
        (["scripts/score/compare_sim1d_es1.py", "--es", "1"],
         "compare_sim1d_es1: name the configuration package", "--stance"),
        # A SEED is an initial condition later runs stand on, so an unnamed one
        # feeding named runs is the unstanced divergence one layer down.
        (["scripts/run/build_neutral_seed_cache.py", "--es1",
          "--db-dir", "unused_seed_db"],
         "build_neutral_seed_cache: name the configuration package",
         "--stance"),
        # A stability verdict is a statement ABOUT a configuration: the same
        # corner is well behaved under one closure and marginal under another.
        (["scripts/run/sweep_sim1d_stability.py"],
         "sweep_sim1d_stability: name the configuration package", "--stance"),
    )

    for _dr_bare, _dr_phrase, _dr_switch in _dr_drivers:
        _dr_out = _dr_run(_dr_bare)
        _dr_text = _dr_out.stdout + _dr_out.stderr
        assert _dr_out.returncode != 0, (_dr_bare, _dr_text[-400:])
        assert _dr_phrase in _dr_text, (_dr_bare, _dr_text[-800:])

        _dr_named = _dr_run([*_dr_bare, _dr_switch, _dr_missing])
        _dr_named_text = _dr_named.stdout + _dr_named.stderr
        assert _dr_named.returncode != 0, (_dr_bare, _dr_named_text[-400:])
        assert _dr_phrase not in _dr_named_text, (
            _dr_bare, _dr_named_text[-800:]
        )
        assert _dr_missing in _dr_named_text, (_dr_bare, _dr_named_text[-800:])


@_case("configuration-drivers-refuse-rung-owned-supersession")
def _case_configuration_drivers_refuse_rung_owned_supersession():
    # A RUNG-OWNED key is the rung's to set, and the named configuration lands
    # ON TOP of the rung, so a file that carries one silently re-labels which
    # rung the run is: --es 2 would be scored against ES2 data while carrying
    # whatever drive the file names. Neither driver's departure report catches
    # it -- the file's value IS the configuration, so nothing departs from it.
    #
    # Both drivers now refuse it through ONE implementation,
    # run_mechanism_ladder.refuse_rung_supersession, which lives beside the
    # ES_OPERATING table whose membership RUNG_OWNED_LIVE describes. The check
    # runs the drivers' real entry points against fixture configurations in a
    # temporary stance directory: the refusal fires before anything is
    # constructed, and the positive controls stop at a patched run_model, so no
    # case here reaches a solve.
    import run_m6_point as _ro_m6
    import run_mechanism_ladder as _ro_ladder

    # ONE implementation, reached by both routes: the m6 driver holds the very
    # object the ladder defines, not a copy of it.
    assert (_ro_m6.refuse_rung_supersession
            is _ro_ladder.refuse_rung_supersession)
    assert _ro_m6.RUNG_OWNED_LIVE is _ro_ladder.RUNG_OWNED_LIVE
    assert _ro_ladder.RUNG_OWNED_LIVE == ("V_bank", "cathode_Ts_base_K")

    class _RoReached(Exception):
        """Raised in place of a solve, to mark the guard as passed."""

    def _ro_stub(*_args, **_kwargs):
        raise _RoReached

    def _ro_refused(entry, argv):
        """Return the refusal message ``entry(argv)`` raises, or fail."""
        try:
            entry(argv)
        except ValueError as _ro_exc:
            return str(_ro_exc)
        except _RoReached:
            raise AssertionError(f"guard did not fire: {argv}") from None
        raise AssertionError(f"no refusal and no run: {argv}")

    def _ro_passed(entry, argv):
        """Assert ``entry(argv)`` gets past the guard to the patched solve."""
        try:
            entry(argv)
        except _RoReached:
            return
        raise AssertionError(f"guard fired or run diverted: {argv}")

    _ro_saved = (_ro_ladder.run_model, _ro_m6.run_model)
    with _derived_fixture_dir() as (_ro_sc, _ro_room):
        _ro_ladder.run_model = _ro_stub
        _ro_m6.run_model = _ro_stub
        try:
            # The standby clash, on the route the ladder actually took: the
            # rung's ES1 standby is 1910.0 and this file names another.
            (_ro_room / "rungclash.toml").write_text(
                "[input_dict]\ncathode_Ts_base_K = 1234.0\n"
            )
            # The bank-voltage clash, the other member of the set.
            (_ro_room / "vbankclash.toml").write_text(
                "[input_dict]\nV_bank = 12.0\n"
            )
            # A configuration that names NO rung-owned key: the positive
            # control, and the shape every committed stance has.
            (_ro_room / "nonrung.toml").write_text(
                "[input_dict]\nS_gp = 1234.0\n"
            )
            _ro_clash_file = str((_ro_room / "rungclash.toml").resolve())
            _ro_h5 = str(_ro_room / "unused.h5")

            # (i) THE LADDER ROUTE, which is where the rung's standby was
            # being superseded.
            _ro_msg = _ro_refused(_ro_ladder.main, [
                "--es", "1",
                "--stance", "rungclash", "--save-h5", _ro_h5,
            ])
            assert "run_mechanism_ladder:" in _ro_msg, _ro_msg
            assert "cathode_Ts_base_K" in _ro_msg, _ro_msg
            assert "1234.0" in _ro_msg, _ro_msg       # the file's value
            assert "1910.0" in _ro_msg, _ro_msg       # the rung's value
            assert _ro_clash_file in _ro_msg, _ro_msg  # the file
            assert "--no-stance" in _ro_msg, _ro_msg
            # No restatement route on this driver, so none is offered.
            assert "--extra" not in _ro_msg, _ro_msg

            # (ii) THE OTHER RUNG-OWNED KEY, and a rung that is not ES1, so the
            # refusal is not reading one hard-coded operating point.
            _ro_msg_vb = _ro_refused(_ro_ladder.main, [
                "--es", "2",
                "--stance", "vbankclash", "--save-h5", _ro_h5,
            ])
            assert "V_bank" in _ro_msg_vb, _ro_msg_vb
            assert "12.0" in _ro_msg_vb, _ro_msg_vb
            assert "138.303" in _ro_msg_vb, _ro_msg_vb

            # (iii) THE M6 ROUTE, same helper, same key, its own name and its
            # own restatement switch.
            _ro_msg_m6 = _ro_refused(_ro_m6.main, [
                "--es", "1", "--stance", "rungclash", "--sgp", "9010",
                "--save-h5", _ro_h5,
            ])
            assert "run_m6_point:" in _ro_msg_m6, _ro_msg_m6
            assert "cathode_Ts_base_K" in _ro_msg_m6, _ro_msg_m6
            assert "1234.0" in _ro_msg_m6, _ro_msg_m6
            assert "1910.0" in _ro_msg_m6, _ro_msg_m6
            assert _ro_clash_file in _ro_msg_m6, _ro_msg_m6
            assert "--extra cathode_Ts_base_K=1910.0" in _ro_msg_m6, _ro_msg_m6

            # (iv) POSITIVE CONTROL. A configuration that names no rung-owned
            # key runs: the guard refuses supersession, not stances.
            _ro_passed(_ro_ladder.main, [
                "--es", "1",
                "--stance", "nonrung", "--save-h5", _ro_h5,
            ])
            _ro_passed(_ro_m6.main, [
                "--es", "1", "--stance", "nonrung", "--sgp", "9010",
                "--save-h5", _ro_h5,
            ])

            # (v) --no-stance ON THE LADDER is inert: with no stance layer
            # there is nothing above the rung to supersede it, and this
            # driver has no --extra with which a command line could put
            # something there, so its unstanced route cannot trip the guard.
            _ro_passed(_ro_ladder.main, [
                "--es", "1", "--no-stance",
                "--save-h5", _ro_h5,
            ])

            # (vi) CONSENT IS RESTATEMENT, on the driver that offers it. A
            # command line consents to the stance layering by naming the RUNG
            # value -- the one value that leaves the run's ES label true --
            # and gets past the guard the stance alone trips. The ladder
            # exposes no such switch, which is why its refusal above names
            # only --no-stance. Both rung-owned keys, on their own clashing
            # file.
            _ro_passed(_ro_m6.main, [
                "--es", "1", "--stance", "rungclash", "--sgp", "9010",
                "--extra", "cathode_Ts_base_K=1910.0", "--save-h5", _ro_h5,
            ])
            _ro_passed(_ro_m6.main, [
                "--es", "1", "--stance", "vbankclash", "--sgp", "9010",
                "--extra", "V_bank=177.843", "--save-h5", _ro_h5,
            ])

            # (vii) AND ANY OTHER VALUE IS REFUSED. --extra layers a key above
            # the stance; it does not re-choose the rung, so it can never move
            # a rung-owned key off the rung. Restating the STANCE's value is
            # the case that used to consent and is the one that matters: it is
            # exactly the arm that would be labelled ES1 while carrying
            # another rung's drive. A THIRD value is refused the same way.
            _ro_msg_st = _ro_refused(_ro_m6.main, [
                "--es", "1", "--stance", "rungclash", "--sgp", "9010",
                "--extra", "cathode_Ts_base_K=1234.0", "--save-h5", _ro_h5,
            ])
            assert "run_m6_point:" in _ro_msg_st, _ro_msg_st
            assert "cathode_Ts_base_K" in _ro_msg_st, _ro_msg_st
            assert "1234.0" in _ro_msg_st, _ro_msg_st   # the command line's
            assert "1910.0" in _ro_msg_st, _ro_msg_st   # the rung's
            assert _ro_clash_file in _ro_msg_st, _ro_msg_st  # the stance too
            assert "--extra cathode_Ts_base_K=1910.0" in _ro_msg_st, _ro_msg_st

            # On the other key, over a stance that does NOT name it: the
            # refusal is about the rung, not about the stance, so it fires
            # with no stance clause to report.
            _ro_msg_third = _ro_refused(_ro_m6.main, [
                "--es", "1", "--stance", "nonrung", "--sgp", "9010",
                "--extra", "V_bank=42.0", "--save-h5", _ro_h5,
            ])
            assert "V_bank" in _ro_msg_third, _ro_msg_third
            assert "42.0" in _ro_msg_third, _ro_msg_third
            assert "177.843" in _ro_msg_third, _ro_msg_third
            assert "names it too" not in _ro_msg_third, _ro_msg_third
            assert "--extra V_bank=177.843" in _ro_msg_third, _ro_msg_third

            # (viii) AND THE SAME ON --no-stance, WHICH DROPS THE
            # CONFIGURATION LAYER, NOT THE RUNG. The run is still labelled
            # ES1 and still scored against ES1 data, so an --extra that moves
            # a rung-owned key off its rung mislabels it exactly as it does
            # under a stance. Both keys, refused off the rung and consenting
            # at it, with a non-rung-owned key to show the guard still
            # refuses supersession rather than --extra.
            for _ro_key, _ro_off, _ro_rung in (
                ("V_bank", "42.0", "177.843"),
                ("cathode_Ts_base_K", "1500.0", "1910.0"),
            ):
                _ro_msg_ns = _ro_refused(_ro_m6.main, [
                    "--es", "1", "--no-stance", "--sgp", "9010",
                    "--extra", f"{_ro_key}={_ro_off}", "--save-h5", _ro_h5,
                ])
                assert "run_m6_point:" in _ro_msg_ns, _ro_msg_ns
                assert _ro_key in _ro_msg_ns, _ro_msg_ns
                assert _ro_off in _ro_msg_ns, _ro_msg_ns   # the command line's
                assert _ro_rung in _ro_msg_ns, _ro_msg_ns  # the rung's
                # No stance to name, so the refusal names none -- neither a
                # second holder of the key nor a stance layer to consent to.
                assert "names it too" not in _ro_msg_ns, _ro_msg_ns
                assert "stance" not in _ro_msg_ns, _ro_msg_ns
                assert (f"--extra {_ro_key}={_ro_rung}"
                        in _ro_msg_ns), _ro_msg_ns
                # Restating the rung value consents, unstanced as stanced.
                _ro_passed(_ro_m6.main, [
                    "--es", "1", "--no-stance", "--sgp", "9010",
                    "--extra", f"{_ro_key}={_ro_rung}", "--save-h5", _ro_h5,
                ])
            # A key the rung does not own is --extra's to set on an unstanced
            # run, at any value: the guard is about the ES label, not about
            # overrides.
            _ro_passed(_ro_m6.main, [
                "--es", "1", "--no-stance", "--sgp", "9010",
                "--extra", "S_gp=1234.0", "--save-h5", _ro_h5,
            ])
        finally:
            _ro_ladder.run_model, _ro_m6.run_model = _ro_saved

    # (ix) CONSENT AND IDENTITY AGREE ON THE SPELLING. The guard's equality
    # is exact and typeless, so ``1910`` consents to a rung value of
    # ``1910.0`` -- but ``config_identity`` hashes the canonical JSON of the
    # resolved configuration, where the two are different bytes. Consent
    # without this would accept a value that leaves the run carrying a second
    # identity for one physical configuration. ``--extra`` now types every
    # value from the owning template, so the two spellings ARE one value: same
    # float, same identity. Captured on the committed configuration through
    # the real driver, with LAPDSim1D stubbed, so nothing solves.
    import preflight_diffcfg as _xt_pre
    from cablp.solvers._sim1d import config_identity as _xt_identity

    def _xt_capture(spelling):
        captured = _xt_pre.capture(lambda: _ro_m6.main([
            "--es", "1", "--stance", "g1atrim", "--sgp", "9010",
            "--extra", f"cathode_Ts_base_K={spelling}",
            "--save-h5", "/dev/null",
        ]))
        return captured.params, captured.flags

    _xt_int_p, _xt_int_f = _xt_capture("1910")
    _xt_float_p, _xt_float_f = _xt_capture("1910.0")
    assert type(_xt_int_p["cathode_Ts_base_K"]) is float, (
        _xt_int_p["cathode_Ts_base_K"]
    )
    assert _xt_int_p["cathode_Ts_base_K"] == 1910.0
    assert (_xt_identity(_xt_int_p, _xt_int_f)
            == _xt_identity(_xt_float_p, _xt_float_f)), (
        _xt_identity(_xt_int_p, _xt_int_f),
        _xt_identity(_xt_float_p, _xt_float_f),
    )

    # ...and the two refusals that typing buys: a value that is not readable
    # as its key's type, and a key neither template owns. Both fire at the
    # parse layer, before the consent guard and before any construction --
    # captured the same stubbed way, so a regression here reports a missing
    # refusal rather than starting a solve.
    def _xt_refused(spelling):
        argv = [
            "--es", "1", "--stance", "g1atrim", "--sgp", "9010",
            "--extra", spelling, "--save-h5", "/dev/null",
        ]
        try:
            _xt_pre.capture(lambda: _ro_m6.main(argv))
        except ValueError as _xt_exc:
            return str(_xt_exc)
        raise AssertionError(f"--extra {spelling} was not refused")

    _xt_bad = _xt_refused("cathode_Ts_base_K=hot")
    assert "cathode_Ts_base_K" in _xt_bad, _xt_bad
    assert "carries float" in _xt_bad, _xt_bad
    assert "'hot'" in _xt_bad, _xt_bad
    _xt_unknown = _xt_refused("cathode_Ts_base=1910.0")
    assert "cathode_Ts_base" in _xt_unknown, _xt_unknown
    assert "NEITHER" in _xt_unknown, _xt_unknown


@_case("configuration-fluid-comparator-example")
def _case_configuration_fluid_comparator_example():
    # THE COMMITTED WORKED EXAMPLE. The fluid comparator is the campaign's
    # alternate closure written as a DERIVED FILE rather than as a heap of
    # --extra flags, and this case is the claim that the two are the same
    # configuration: it rebuilds the equivalent by hand -- the base
    # configuration plus exactly the file's own deltas, the way a command line
    # would supply them -- and compares the RESOLVED dicts key by key before
    # comparing identities, so a match is a fact about values and not about a
    # hash.
    #
    # It also constructs. A comparator that resolves but refuses at
    # construction is not an arm anyone can run, and the closure's
    # kinetic-only machinery (the DVM's own jets and baffles) is exactly what
    # a hand-written --extra list forgets.
    from cablp.solvers._sim1d import config_identity, default_config

    from stance_config import load_configuration, load_stance

    _fc_path = (
        Path(__file__).resolve().parents[2]
        / "stances" / "examples" / "g1atrim_fluid_comparator.toml"
    )
    _fc_params, _fc_flags, _fc_lineage = load_configuration(str(_fc_path))

    assert _fc_lineage.name == "g1atrim_fluid_comparator"
    assert _fc_lineage.base_chain == ("g1atrim",), _fc_lineage.base_chain
    assert len(_fc_lineage.file_sha256) == 2

    # The hand-built equivalent, stated here in full: this list IS the
    # --extra/--extra-flag command line the derived file replaces.
    _fc_extra_params = {
        "neutral_model": "moment",
        "cathode_neutral_jet": True,
        "cathode_jet_surface_debit": True,
        "cathode_jet_energy_convention": "total_reflected",
        "neutral_kinetic_dvm_cathode_jet": False,
        "neutral_kinetic_dvm_anode_jet": False,
        # The shared surface-jet launch width goes away with the jets: this
        # closure launches no such spectrum, so a set width is refused rather
        # than ignored. None is its unset value, which the derived file states
        # through [none_valued] because TOML has no null literal.
        "neutral_kinetic_dvm_jet_launch_width": None,
    }
    _fc_extra_flags = {
        "neutral_momentum": True,
        "neutral_energy": True,
        "neutral_hot_internal_wall": True,
    }
    assert _fc_lineage.delta_keys == tuple(
        sorted({*_fc_extra_params, *_fc_extra_flags})
    ), _fc_lineage.delta_keys

    _fc_hand_p, _fc_hand_f = default_config()
    _fc_base = load_stance("g1atrim")
    _fc_hand_p.update(_fc_base.params)
    _fc_hand_f.update(_fc_base.flags)
    _fc_hand_p.update(_fc_extra_params)
    _fc_hand_f.update(_fc_extra_flags)

    _fc_differ = [
        f"params:{k}" for k in sorted(set(_fc_params) | set(_fc_hand_p))
        if _fc_params.get(k) != _fc_hand_p.get(k)
    ] + [
        f"flags:{k}" for k in sorted(set(_fc_flags) | set(_fc_hand_f))
        if _fc_flags.get(k) != _fc_hand_f.get(k)
    ]
    assert not _fc_differ, _fc_differ
    assert _fc_lineage.identity == config_identity(_fc_hand_p, _fc_hand_f)

    LAPDSim1D(_fc_params, _fc_flags)

    # NEGATIVE CONTROL. Drop ONE delta from the hand-built equivalent and the
    # resolved dicts must part company, or the comparison above would pass for
    # a file that moved nothing at all.
    _fc_short_p, _fc_short_f = default_config()
    _fc_short_p.update(_fc_base.params)
    _fc_short_f.update(_fc_base.flags)
    _fc_short_p.update(
        {k: v for k, v in _fc_extra_params.items() if k != "neutral_model"}
    )
    _fc_short_f.update(_fc_extra_flags)
    assert _fc_short_p["neutral_model"] == "kinetic_dvm"
    assert _fc_lineage.identity != config_identity(_fc_short_p, _fc_short_f)


@_case("configuration-every-committed-example-constructs")
def _case_configuration_every_committed_example_constructs():
    # EVERY COMMITTED EXAMPLE, not a chosen one. scripts/stances/examples/ is
    # where the campaign's derived configurations live -- the alternate
    # closures, the per-rung reference arms, the comparators -- and until this
    # case existed almost none of them was loaded by anything that runs: a
    # renamed config key, a retired selector or a block whose family gained a
    # member would leave a committed file that no longer resolves, and nothing
    # would say so until somebody tried to run that arm.
    #
    # The case GLOBS the directory rather than listing it, so a file added
    # without a thought for this gate is covered the moment it lands. For each
    # file it resolves the configuration through the repo's own loader, checks
    # the lineage a saved trajectory would record, and CONSTRUCTS LAPDSim1D
    # from it. Construction, not a run: construction-time validation is where
    # a misconfigured arm raises, and a run would cost minutes per file.
    #
    # THE DISPOSITION TABLE below is the only way a file escapes either check,
    # and an entry is a SENTENCE, never a silent pass. Two kinds:
    #
    #   ("skip", "<why>")     construction needs an input that is not in this
    #                         repository (a measured trace, a per-cell profile
    #                         file). The file is still RESOLVED and its lineage
    #                         still checked; only the construction is skipped,
    #                         and the reason names the missing input.
    #   ("fails", "<text>")   the file does NOT construct today. This is a
    #                         FINDING, recorded here with the error text it
    #                         raises so the failure stays visible and the suite
    #                         stays green. The entry asserts the failure still
    #                         happens and still says that, so repairing the
    #                         file breaks this case until the row is removed --
    #                         a known failure cannot quietly become a mystery.
    #
    # The table is EMPTY: as it stands every committed example resolves and
    # constructs. A stale row is refused below, so a row cannot outlive the
    # file it names.
    from stance_config import available_stances, load_configuration

    _ex_dispositions = {}

    _ex_dir = Path(__file__).resolve().parents[2] / "stances" / "examples"
    _ex_files = sorted(_ex_dir.glob("*.toml"))
    assert _ex_files, f"no committed examples under {_ex_dir}"
    _ex_names = {path.name for path in _ex_files}
    _ex_stale = sorted(set(_ex_dispositions) - _ex_names)
    assert not _ex_stale, (
        f"the disposition table names files that are not in {_ex_dir}: "
        f"{_ex_stale}. A row outliving its file is a row nobody re-reads"
    )
    _ex_committed = set(available_stances())

    for _ex_path in _ex_files:
        _ex_params, _ex_flags, _ex_lineage = load_configuration(str(_ex_path))
        # THE LINEAGE a run built from this file would write into its HDF5
        # root: the file's own name, a base chain that bottoms out in a
        # COMMITTED stance (a derived file whose base chain ended anywhere else
        # would name a configuration a reader cannot resolve), and at least one
        # delta, because a derived file that moves nothing declares nothing.
        assert _ex_lineage.name == _ex_path.stem, (
            _ex_path.name, _ex_lineage.name
        )
        assert _ex_lineage.base_chain, _ex_path.name
        assert _ex_lineage.base_chain[-1] in _ex_committed, (
            _ex_path.name, _ex_lineage.base_chain
        )
        assert _ex_lineage.delta_keys, _ex_path.name
        assert len(_ex_lineage.file_sha256) == len(_ex_lineage.base_chain) + 1

        _ex_kind, _ex_why = _ex_dispositions.get(_ex_path.name, (None, None))
        if _ex_kind == "skip":
            continue
        if _ex_kind == "fails":
            try:
                LAPDSim1D(dict(_ex_params), dict(_ex_flags))
            except Exception as _ex_error:
                assert _ex_why in str(_ex_error), (
                    _ex_path.name, _ex_why, str(_ex_error)
                )
                continue
            raise AssertionError(
                f"{_ex_path.name} is recorded as a KNOWN FAILURE ({_ex_why}) "
                "and now constructs; delete its disposition row"
            )
        assert _ex_kind is None, (_ex_path.name, _ex_kind)
        LAPDSim1D(dict(_ex_params), dict(_ex_flags))


@_case("configuration-restated-block-refusal")
def _case_configuration_restated_block_refusal():
    # THE UNIT OF THE RESTATEMENT CHECK IS THE DELTA THE FILE WROTE, and a
    # declaration block is ONE delta. A block states a family's complete
    # membership regardless of value, so a member that agrees with the base is
    # the form working and must not be refused -- but the block as a whole must
    # still move something, or it re-declares a decision the base already made.
    #
    # The fixture is the base's own [models.beam_tail_closure] block, lifted
    # verbatim so completeness is the base's and not this case's guess, and
    # then used twice: once with one member moved, once untouched.
    import re as _rb_re

    with _derived_fixture_dir() as (_sc, _room):
        # Sliced on LINE-ANCHORED table headers: the block names also appear
        # inside the file's comments, so a bare substring search finds prose.
        _rb_lines = (_room / "g1atrim.toml").read_text().splitlines()
        _rb_start = _rb_lines.index("[models.beam_tail_closure]")
        _rb_end = next(
            i for i in range(_rb_start + 1, len(_rb_lines))
            if _rb_lines[i].startswith("[")
        )
        _rb_block = "\n".join(_rb_lines[_rb_start:_rb_end]).rstrip() + "\n"
        assert "ql_relaxation_coeff = 30.0" in _rb_block

        # NEGATIVE CONTROL (a): every member equal to the base. The block
        # declares a decision the base already made, and is refused BY FAMILY
        # NAME rather than by listing members that are individually blameless.
        (_room / "block_restated.toml").write_text(
            'base = "g1atrim"\n\n' + _rb_block
        )
        try:
            _sc.load_stance("block_restated")
        except ValueError as _rb_exc:
            assert "[models.beam_tail_closure]" in str(_rb_exc), str(_rb_exc)
            assert "restates" in str(_rb_exc), str(_rb_exc)
            assert "must move at least one" in str(_rb_exc), str(_rb_exc)
        else:
            raise AssertionError("a wholly restated block was ACCEPTED")

        # The same block with ONE member moved is a legitimate delta, and the
        # nineteen members that still equal the base do NOT make it one.
        _rb_moved_block = _rb_re.sub(
            r"^ql_relaxation_coeff = 30\.0$",
            "ql_relaxation_coeff = 31.0",
            _rb_block,
            count=1,
            flags=_rb_re.MULTILINE,
        )
        assert "ql_relaxation_coeff = 31.0" in _rb_moved_block
        (_room / "block_moved.toml").write_text(
            'base = "g1atrim"\n\n' + _rb_moved_block
        )
        _rb_derived = _sc.load_stance("block_moved")
        assert _rb_derived.params["ql_relaxation_coeff"] == 31.0

        # NEGATIVE CONTROL (b): the identity moves, and moves by exactly that
        # key -- so the accepted block is not quietly resolving to the base.
        _rb_base = _sc.load_stance("g1atrim")
        assert _rb_derived.lineage.identity != _rb_base.lineage.identity
        _rb_differ = [
            k for k in set(_rb_derived.params) | set(_rb_base.params)
            if _rb_derived.params.get(k) != _rb_base.params.get(k)
        ] + [
            f"flags:{k}" for k in set(_rb_derived.flags) | set(_rb_base.flags)
            if _rb_derived.flags.get(k) != _rb_base.flags.get(k)
        ]
        assert _rb_differ == ["ql_relaxation_coeff"], _rb_differ

        # A FLAT key restating the base is still refused, block or no block:
        # the exemption is about the form a delta is written in, not a licence.
        (_room / "block_moved_plus_flat.toml").write_text(
            'base = "g1atrim"\n\n'
            + _rb_moved_block
            + f"\n[input_dict]\nS_gp = {_rb_base.params['S_gp']!r}\n"
        )
        try:
            _sc.load_stance("block_moved_plus_flat")
        except ValueError as _rb_flat_exc:
            assert "input_dict:S_gp" in str(_rb_flat_exc), str(_rb_flat_exc)
        else:
            raise AssertionError("a restated flat key beside a block was ACCEPTED")


# --------------------------------------------------------------------
# floor-audit-names-its-configuration
# --------------------------------------------------------------------
@_case("floor-audit-names-its-configuration")
def _case_floor_audit_names_its_configuration():
    # The floor-activation audit is a run entry point, so it names the
    # configuration it measures: no bare mode, and the golden route resolves
    # to the golden's OWN configuration rather than to a hand-kept copy of it.
    import audit_sim1d_floor_activation as _fa

    # REFUSALS. argparse exits 2 on each; the message names what to pass.
    _fa_refused = (
        ([], "an unnamed configuration", "template of keys, not a plasma"),
        (["--max-steps", "5"], "an unnamed configuration with a step cap",
         "name the configuration package"),
        (["--no-stance", "--golden-route"], "the golden route without a name",
         "--golden-route is the golden gate's layering"),
        (["--stance", "g1atrim", "--nx", "40"], "--nx without the golden route",
         "--nx layers a mesh on the --golden-route treatment"),
    )
    for _fa_argv, _fa_what, _fa_says in _fa_refused:
        _fa_err = StringIO()
        try:
            with contextlib.redirect_stderr(_fa_err):
                _fa.main(_fa_argv)
        except SystemExit as _fa_exit:
            assert _fa_exit.code == 2, (_fa_what, _fa_exit.code)
            assert _fa_says in _fa_err.getvalue(), (
                _fa_what, _fa_err.getvalue()
            )
        else:
            raise AssertionError(f"the floor audit accepted {_fa_what}")

    # THE GOLDEN ROUTE, no solve: the configuration it assembles must be the
    # one the committed sidecar records, identity for identity. A drifting
    # hand-kept copy is exactly what this case exists to catch, so the
    # comparison is against the sidecar and not against a literal here.
    _fa_args = argparse.Namespace(
        stance="g1atrim", no_stance=False, golden_route=True, nx=None
    )
    _fa_params, _fa_flags, _fa_lineage = _fa.build_audit_config(_fa_args)
    _fa_sidecar = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "baselines"
            / "production_discharge.json"
        ).read_text()
    )["configuration"]
    assert _fa_lineage.name == _fa_sidecar["name"], _fa_lineage.name
    assert (
        _fa.config_identity(_fa_params, _fa_flags) == _fa_sidecar["identity"]
    ), (_fa.config_identity(_fa_params, _fa_flags), _fa_sidecar["identity"])

    # ...and the header says so, so a transcript carries the identity.
    _fa_out = StringIO()
    with contextlib.redirect_stdout(_fa_out):
        _fa.print_header(_fa_params, _fa_flags, _fa_lineage, None)
    _fa_header = _fa_out.getvalue()
    assert _fa_sidecar["identity"] in _fa_header, _fa_header
    assert "g1atrim" in _fa_header, _fa_header

    # --scheme defaults to the configuration's own value and announces an
    # explicit one as an override, so a verdict line cannot claim a scheme
    # the run did not use.
    assert "(the configuration's own)" in _fa_header, _fa_header
    _fa_out = StringIO()
    with contextlib.redirect_stdout(_fa_out):
        _fa.print_header(_fa_params, _fa_flags, _fa_lineage, "crank_nicolson")
    assert "OVERRIDDEN on the command line" in _fa_out.getvalue()

    # The unnamed route records itself as unnamed rather than borrowing a name.
    _fa_args = argparse.Namespace(
        stance=None, no_stance=True, golden_route=False, nx=None
    )
    _fa_params, _fa_flags, _fa_lineage = _fa.build_audit_config(_fa_args)
    assert _fa_lineage is None
    assert not hasattr(_fa, "PARAM_OVERRIDES")
    assert not hasattr(_fa, "FLAG_OVERRIDES")


# ----------------------------------------------------------------------
# configuration-file-value-typed-to-template
# ----------------------------------------------------------------------
@_case("configuration-file-value-typed-to-template")
def _case_configuration_file_value_typed_to_template():
    # ONE CONFIGURATION, ONE IDENTITY, HOWEVER ITS NUMBERS WERE SPELLED. TOML
    # distinguishes `1900` from `1900.0` and `config_identity` hashes the
    # canonical JSON text, where those are different bytes -- so a file that
    # spelled a float key's value as an integer used to resolve to a
    # configuration nothing else recognised, and wrote that spelling into
    # `params_json` for every run it produced. The `--extra` switches removed
    # the ambiguity at their parse layer; a configuration FILE is the other
    # place a value is written, and it now applies THE SAME rule, from the
    # same definition (`extra_overrides.coerce_value`).
    import re as _ct_re

    with _derived_fixture_dir() as (_sc, _room):
        _ct_base = _sc.load_stance("g1atrim")
        _ct_base_params, _ct_base_flags = default_config()
        _ct_base_params.update(_ct_base.params)
        _ct_base_flags.update(_ct_base.flags)
        # The pin the whole case turns on: the key is a FLOAT in the template
        # and the base leaves it there, so 1900 and 1900.0 are two spellings
        # of one move and neither is a restatement.
        assert type(_ct_base_params["cathode_Ts_base_K"]) is float
        assert _ct_base_params["cathode_Ts_base_K"] == 1910.0

        # (i) THE PAIR. Two files, one physical configuration.
        _ct_loaded = {}
        for _ct_label, _ct_spelling in (("int", "1900"), ("float", "1900.0")):
            (_room / f"typed_{_ct_label}.toml").write_text(
                'base = "g1atrim"\n'
                "\n"
                "[input_dict]\n"
                f"cathode_Ts_base_K = {_ct_spelling}\n"
            )
            _ct_loaded[_ct_label] = _sc.load_configuration(
                f"typed_{_ct_label}"
            )
        for _ct_label in ("int", "float"):
            _ct_value = _ct_loaded[_ct_label][0]["cathode_Ts_base_K"]
            assert type(_ct_value) is float, (_ct_label, type(_ct_value))
            assert _ct_value == 1900.0, (_ct_label, _ct_value)
        # Identical RESOLVED dicts first, then identical identity -- so the
        # match is a fact about values and not about a hash agreeing.
        assert _ct_loaded["int"][0] == _ct_loaded["float"][0]
        assert _ct_loaded["int"][1] == _ct_loaded["float"][1]
        assert (
            _ct_loaded["int"][2].identity == _ct_loaded["float"][2].identity
        ), (_ct_loaded["int"][2].identity, _ct_loaded["float"][2].identity)
        # NEGATIVE CONTROL: the shared identity is not the base's, so the
        # delta did move and the agreement above is not the agreement of two
        # files that both did nothing.
        assert _ct_loaded["int"][2].identity != _ct_base.lineage.identity

        # (ii) A BOOL KEY TAKES true/false AND NOTHING ELSE. `1` is a legible
        # integer and TOML hands it over as one; a flag is not a number.
        (_room / "typed_bool.toml").write_text(
            'base = "g1atrim"\n\n[input_flags]\ncathode_coupling = 1\n'
        )
        try:
            _sc.load_stance("typed_bool")
        except ValueError as _ct_bexc:
            _ct_bmsg = str(_ct_bexc)
        else:
            raise AssertionError("a bool key ACCEPTED an integer")
        assert "typed_bool" in _ct_bmsg, _ct_bmsg
        assert "cathode_coupling" in _ct_bmsg, _ct_bmsg
        assert "carries bool" in _ct_bmsg, _ct_bmsg
        assert "1" in _ct_bmsg, _ct_bmsg

        # ...and a STRING key likewise. This is the ONE place the file route
        # and the `--extra` route part company on purpose: a command-line
        # token is text and IS its own value, so `--extra gas_type=1` gives
        # the string "1", while a TOML integer is an integer and the file
        # is refused rather than quietly stringified.
        (_room / "typed_str.toml").write_text(
            'base = "g1atrim"\n\n[input_dict]\ngas_type = 1\n'
        )
        try:
            _sc.load_stance("typed_str")
        except ValueError as _ct_sexc:
            _ct_smsg = str(_ct_sexc)
        else:
            raise AssertionError("a str key ACCEPTED an integer")
        assert "gas_type" in _ct_smsg, _ct_smsg
        assert "carries str" in _ct_smsg, _ct_smsg

        # (iii) AN INT KEY TAKES A WHOLE FLOAT, and resolves to an int: `nx`
        # is a cell count, and 42.0 names the same mesh 42 does.
        (_room / "typed_int.toml").write_text(
            'base = "g1atrim"\n\n[input_dict]\nnx = 42.0\n'
        )
        _ct_nx = _sc.load_stance("typed_int").params["nx"]
        assert type(_ct_nx) is int, type(_ct_nx)
        assert _ct_nx == 42, _ct_nx
        # A fractional one is not a cell count at all.
        (_room / "typed_int_frac.toml").write_text(
            'base = "g1atrim"\n\n[input_dict]\nnx = 42.5\n'
        )
        try:
            _sc.load_stance("typed_int_frac")
        except ValueError as _ct_iexc:
            assert "carries int" in str(_ct_iexc), str(_ct_iexc)
        else:
            raise AssertionError("an int key ACCEPTED a fractional float")

        # (iv) THE RESTATED-DELTA CHECK READS THE COERCED VALUE. An integer
        # spelling of the base's float is the same value, so it is the same
        # restatement -- the coercion must not let a delta smuggle itself
        # past the check by changing type instead of value.
        (_room / "typed_restated.toml").write_text(
            'base = "g1atrim"\n\n[input_dict]\ncathode_Ts_base_K = 1910\n'
        )
        try:
            _sc.load_stance("typed_restated")
        except ValueError as _ct_rexc:
            _ct_rmsg = str(_ct_rexc)
        else:
            raise AssertionError("an int spelling of the base's value PASSED")
        assert "restates" in _ct_rmsg, _ct_rmsg
        assert "cathode_Ts_base_K" in _ct_rmsg, _ct_rmsg

        # (v) [none_valued] STAYS None whatever the template says. TOML has no
        # null literal, so a key named there is an explicit UNSET rather than
        # a value of the key's type, and coercing it to a float would be
        # coercing the absence of a value.
        (_room / "typed_unset.toml").write_text(
            'base = "g1atrim"\n\n'
            "[none_valued]\n"
            'input_dict = ["cathode_Ts_base_K"]\n'
        )
        assert (
            _sc.load_stance("typed_unset").params["cathode_Ts_base_K"] is None
        )

        # (vi) A DECLARATION BLOCK'S MEMBERS ARE FILE VALUES TOO, and are
        # typed exactly as a flat delta is, after the projection has filed
        # each member in its namespace. The block is lifted from the base so
        # its completeness is the base's, with one float member moved AND
        # spelled as an integer -- moved so the block is a delta at all,
        # spelled short so it exercises the projection's coercion.
        _ct_lines = (_room / "g1atrim.toml").read_text().splitlines()
        _ct_start = _ct_lines.index("[models.beam_tail_closure]")
        _ct_end = next(
            i for i in range(_ct_start + 1, len(_ct_lines))
            if _ct_lines[i].startswith("[")
        )
        _ct_block = "\n".join(_ct_lines[_ct_start:_ct_end]).rstrip() + "\n"
        _ct_block = _ct_re.sub(
            r"^ql_relaxation_coeff = 30\.0$",
            "ql_relaxation_coeff = 31",
            _ct_block,
            count=1,
            flags=_ct_re.MULTILINE,
        )
        assert "ql_relaxation_coeff = 31\n" in _ct_block
        (_room / "typed_block.toml").write_text(
            'base = "g1atrim"\n\n' + _ct_block
        )
        _ct_bparams = _sc.load_stance("typed_block").params
        assert type(_ct_bparams["ql_relaxation_coeff"]) is float
        assert _ct_bparams["ql_relaxation_coeff"] == 31.0

        # (vii) WHY NOTHING ROTATED. The rule is identity-NEUTRAL over the
        # committed set by construction, and this is that property stated as
        # a check rather than as a claim: every scalar the reference
        # configuration resolves to carries EXACTLY its template key's type,
        # so there was no int-for-float spelling anywhere for the coercion to
        # move. `None` is the explicit unset and a `None`/list template names
        # no scalar type, so both stand outside the statement.
        _ct_template_p, _ct_template_f = default_config()
        _ct_spelled = [
            f"{_ct_space}:{_ct_key}"
            for _ct_space, _ct_res, _ct_tpl in (
                ("input_dict", _ct_base_params, _ct_template_p),
                ("input_flags", _ct_base_flags, _ct_template_f),
            )
            for _ct_key, _ct_val in _ct_res.items()
            if _ct_val is not None
            and isinstance(_ct_tpl[_ct_key], (bool, int, float, str))
            and type(_ct_val) is not type(_ct_tpl[_ct_key])
        ]
        assert not _ct_spelled, _ct_spelled
