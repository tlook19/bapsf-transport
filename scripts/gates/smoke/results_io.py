"""Smoke cases: results, HDF5 I/O, restart, capture artifacts and the scoring
and comparison tools.
"""

import contextlib
from io import StringIO
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import h5py
import numpy as np

from cablp.atomic.adas import he_rate_temperature_range_eV
from cablp.cathode import (
    beam_deposition as _beam_deposition_mod,
    circuit_idriven as _cathode_solver_idriven_mod,
    circuit_common as _cathode_solver_mod,
    kernels as _kernel_selector,
)
from cablp.constants import ev_to_erg
from cablp.solvers._sim1d import (
    LAPDSim1D,
    ProgressPrinter1D,
    SimulationProgress1D,
    default_config,
    load_result_hdf5,
    summarize_result,
)
from cablp.solvers._sim1d.core.state import (
    ConservativeState1D,
    STATE_NAMES_1D,
    pack_state,
    state_field_names,
)
from cablp.solvers._sim1d.physics.neutrals import GAS_PUFF_DIAGNOSTIC_FIELDS
from cablp.solvers._sim1d.results.phase3_capture import (
    ARTIFACT_LOCATOR_BASE,
    ARTIFACT_LOCATOR_STATE,
    _LOCATOR_REQUIRED_FIELDS,
    configuration_identity as _phase3_configuration_identity,
    load_qualified_capture,
    reserve_run_id,
    write_qualified_capture,
)

from ._harness import (
    _CAPFIX_ESCAPE_CONFIG,
    _CAPFIX_ESCAPE_I_A,
    _CAPFIX_ESCAPE_KWARGS,
    _CAPFIX_ESCAPE_PLASMA,
    _base_config,
    _base_sim,
    _case,
)


# Every attribute the RETIRED ``results/compat.py`` used to attach to a result
# namespace, on both the run and the load path. The module aliased sim1d
# results into the shape of the 0D ``_sim3`` solver, which was removed at D2
# (2026-08-03); the aliases outlived their only reason to exist by two years of
# campaign and are retired with it.
#
# THE LIST IS THE GATE. A result is a plain ``SimpleNamespace``, so an absent
# alias raises ``AttributeError`` at the first read rather than returning
# anything -- there is no silent-wrong-value path to guard against, only a
# re-introduction to catch. The cases below assert every name is ABSENT, which
# is what makes a re-added shim fail here instead of in a plot months later.
_RETIRED_SIM3_COMPAT_ALIASES = (
    "ne",
    "v_plasma",
    "n_beam",
    "cathode",
    "cathode_twin",
    "t_breakdown",
    "t_breakdown_ms",
    "time_since_breakdown",
    "time_ms_since_breakdown",
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
    "sim3_compat_units",
    "sim3_compat_notes",
)


# --------------------------------------------------------------------
# no-source-run-and-results
# --------------------------------------------------------------------
@_case(
    "no-source-run-and-results",
    historical_stance=True,
    provides=(
        "cathode_diag", "cathode_run_flags", "cathode_run_params",
        "cathode_run_result", "cathode_run_sim", "run_params",
        "split_flags",
    ),
)
def _case_no_source_run_and_results(expected_rhs_terms, no_source_params):
    # The atomic reaction channels have no config-level off switch since the
    # b_* removal (2026-08-28); this case exercises the run/results plumbing,
    # and the fuelling and boundary sources are still off.
    params, flags = _base_config()
    sim, snapshot = _base_sim()
    geom = snapshot.geometry
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
    # The ``end_*`` cathode-result block is presence-gated on TwinCathode:
    # ABSENT here, not NaN. Only the twin solve fills it, so on a single
    # cathode it was a second copy of the block that never carried a number.
    assert "end_regime" not in run_result.cathode_diagnostics
    # Summed over the five base rows; the packed y also carries nn_a.
    saved_term_sum = np.zeros(
        (run_result.y.shape[0], len(STATE_NAMES_1D) * geom.cells)
    )
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
    # The _sim3 compatibility aliases are RETIRED with results/compat.py. What
    # used to stand here asserted that each alias equalled the field it aliased
    # -- ``Qei == -electron_ion_cooling``, ``S_ion_bulk == ionization_birth``,
    # ``ne == n`` -- so with the module gone every one of those is a statement
    # about a line that no longer exists. The assertions are replaced, not
    # dropped: the RUN path must now hand back a namespace carrying none of
    # them. Absence is the whole claim, and it is loud by construction, because
    # a result is a SimpleNamespace and a missing attribute is an
    # AttributeError at the first read.
    for field_name in _RETIRED_SIM3_COMPAT_ALIASES:
        assert not hasattr(run_result, field_name), field_name
    # The conservative fields the aliases were views OF are untouched, and the
    # block above this one already asserts their content term by term. The one
    # claim that only the alias carried was the cathode namespace's shape and
    # all-NaN content on a run with no cathode solve; it reads off the
    # diagnostics row the namespace was built from, which is unconditionally
    # seeded NaN per end.
    _source_I_tot = run_result.cathode_diagnostics["source_I_tot"]
    assert _source_I_tot.shape == run_result.time.shape
    assert np.all(np.isnan(_source_I_tot))

    entry_flags = dict(flags)
    entry_params = dict(run_params, initial_neutral_state="fill")
    entry_sim = LAPDSim1D(entry_params, entry_flags)
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
            assert h5["geometry/cell_role"][-1].decode("utf-8") == "end_wall"
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
        assert _cathode_solver_mod.j_eth_crit is (
            _cathode_solver_mod.j_eth_crit_pure
        )
        assert _cathode_solver_idriven_mod.j_eth_crit is (
            _cathode_solver_mod.j_eth_crit_pure
        )
        # Tier A (2026-08-02): the same contract for the rest of the unit --
        # one rebinding site per name, and the two solver modules resolve the
        # SAME object (idriven from-imports the shared ones from
        # circuit_common, which rebinds before that import runs).
        for _ta_name in ("c_log_ei", "compute_l_b"):
            assert getattr(_cathode_solver_mod, _ta_name) is getattr(
                _cathode_solver_mod, _ta_name + "_pure"
            ), _ta_name
        assert _cathode_solver_idriven_mod.compute_l_b is (
            _cathode_solver_mod.compute_l_b_pure
        )
        assert _beam_deposition_mod.c_log_ei is (
            _cathode_solver_mod.c_log_ei_pure
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
        # checkout with no compiled extension: it is optional by design. The
        # skip ANNOUNCES ITSELF -- a silent one reached review twice in one
        # cycle reading as a pass, which is the whole reason the solver-branch
        # rule asks for the equivalence lines by eye.
        # scripts/spike_cython_kernels.py (at commit 48be9a4, retired
        # 2026-09-03) is the full-resolution version.
        try:
            import importlib as _importlib

            _cy = _importlib.import_module("cablp.cathode._cathode_kernels_cy")
        except ImportError:
            _cy = None
        if _cy is not None:
            assert _cy.pemr() == _cathode_solver_mod.PEMR
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
                        _pure = _cathode_solver_mod.j_eth_crit_pure(
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
                _cathode_solver_mod.PEMR,
                _cathode_solver_mod.ERG_PER_EV,
                _cathode_solver_mod.ME_CGS,
            )
            _cy.check_constants_idriven(
                _cathode_solver_idriven_mod._SCHOTTKY_EV_PER_SQRT_V_M
            )
            for _ta_bad in (
                lambda: _cy.check_constants(
                    _cathode_solver_mod.PEMR * (1.0 + 1e-15),
                    _cathode_solver_mod.ERG_PER_EV,
                    _cathode_solver_mod.ME_CGS,
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
                    assert _cathode_solver_mod.c_log_ei_pure(
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
                                # compute_l_b_pure calls the module-global
                                # c_log_ei, which IS the compiled one in an
                                # opted-in process -- so this leg alone would
                                # not catch a bad c_log_ei. The sweep above
                                # does, independently, which is what closes it.
                                assert _cathode_solver_mod.compute_l_b_pure(
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
                        * (1.0 - _cathode_solver_mod.exp_clamped(Lambda - psi))
                        + _state(psi)[0]
                    )

                def _net_phi_c(psi):
                    return (psi - _state(psi)[1]) * T_e

                def _reported_phi_c(psi):
                    return psi * T_e - _state(psi)[1] * T_e

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
                psi_c_plus = None
                if not capability_limited:
                    psi_c_plus = _ta_brentq(
                        lambda x: _J_tot(x) - J_target,
                        psi_lo, psi_top, xtol=1.0e-12, rtol=1.0e-14,
                        full_output=False,
                    )
                    # The ceiling is enforced on the located root (2026-08-09).
                    if _reported_phi_c(psi_c_plus) >= phi_c_cap_V:
                        capability_limited = True
                if capability_limited:
                    if _net_phi_c(psi_top) > phi_c_cap_V:
                        return _ta_brentq(
                            lambda x: _net_phi_c(x) - phi_c_cap_V,
                            psi_lo, psi_top, xtol=1.0e-12, rtol=1.0e-14,
                            full_output=False,
                        ), True
                    return psi_top, True
                return psi_c_plus, False

            _ta_tol = _cathode_solver_idriven_mod._J_PLATEAU_TOL_REL
            _ta_lam = math.log(math.sqrt(4.0 * _cathode_solver_mod.PEMR
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

            # The returned-root ceiling (2026-08-09) on the case that motivated
            # it, end to end and on BOTH paths: the frozen escaping state, run
            # through solve_idriven with the compiled root find bound and then
            # with the pure ladder. The two must agree BIT for bit -- the
            # transcription is faithful or it is not -- and both must report
            # the ceiling rather than the far higher root the pre-fix ladder
            # returned. Also run at a current BELOW the window, so the
            # comparison covers the ordinary J-root path through the same
            # frozen state.
            _cap_cy_cfg = _cathode_solver_mod.DeviceConfig(
                **_CAPFIX_ESCAPE_CONFIG
            )
            _cap_cy_pl = _cathode_solver_mod.PlasmaState(
                **_CAPFIX_ESCAPE_PLASMA
            )
            _cap_saved_root = _cathode_solver_idriven_mod._COMPILED_ROOT
            try:
                for _cap_cy_I in (5.0, 5.45, _CAPFIX_ESCAPE_I_A, 6.0):
                    _cathode_solver_idriven_mod._COMPILED_ROOT = None
                    _cap_pure = _cathode_solver_idriven_mod.solve_idriven(
                        _cap_cy_cfg, _cap_cy_pl, I_tot_A=_cap_cy_I,
                        **_CAPFIX_ESCAPE_KWARGS
                    )
                    _cathode_solver_idriven_mod._COMPILED_ROOT = (
                        _cy.solve_psi_annular_schottky
                    )
                    _cap_comp = _cathode_solver_idriven_mod.solve_idriven(
                        _cap_cy_cfg, _cap_cy_pl, I_tot_A=_cap_cy_I,
                        **_CAPFIX_ESCAPE_KWARGS
                    )
                    assert _cap_comp.regime == _cap_pure.regime, (
                        _cap_cy_I, _cap_comp.regime, _cap_pure.regime
                    )
                    for _cap_att in (
                        "phi_c", "phi_c_plus", "phi_c_minus", "phi_a",
                        "I_eth_star", "I_tot", "V_p", "V_b", "l_b",
                        "beam_bypass_fraction",
                    ):
                        _cap_a = float(getattr(_cap_comp, _cap_att))
                        _cap_b = float(getattr(_cap_pure, _cap_att))
                        assert _cap_a == _cap_b, (
                            _cap_cy_I, _cap_att,
                            _cap_a.hex(), _cap_b.hex(),
                        )
                    if _cap_cy_I >= 5.46:
                        assert _cap_comp.regime == "capability_limited", (
                            _cap_cy_I, _cap_comp.regime
                        )
                        assert np.isclose(
                            _cap_comp.phi_c, 1000.0, rtol=1e-12, atol=0.0
                        ), _cap_comp.phi_c
                    else:
                        assert _cap_comp.regime == "virtual_cathode", (
                            _cap_cy_I, _cap_comp.regime
                        )
                        assert _cap_comp.phi_c < 1000.0, _cap_comp.phi_c
            finally:
                _cathode_solver_idriven_mod._COMPILED_ROOT = _cap_saved_root
        else:
            print(
                "compiled-kernel D4 unit equivalence: SKIPPED -- "
                "cablp.cathode._cathode_kernels_cy is not built "
                "(`python build_ext.py --inplace` enables it)"
            )

        loaded_result = load_result_hdf5(output_path)
        loaded_via_solver = LAPDSim1D.load_result(output_path)
        for loaded in (loaded_result, loaded_via_solver):
            assert loaded.compiled_kernels == _kernel_selector.PROVENANCE
            assert loaded.path == output_path
            assert loaded.steps == run_result.steps
            assert np.isclose(loaded.final_time, run_result.final_time)
            assert np.isnan(loaded.t_prebreakdown_trigger)
            assert np.isnan(loaded.t_breakdown_trigger)
            # The LOAD path is the second of the two sites that attached the
            # retired _sim3 aliases (results/io.py did it on every read, the
            # solver on every run), so it gets its own absence witness: a
            # result reconstructed from HDF5 must be as free of them as the
            # one that came straight off a run.
            for _retired_alias in _RETIRED_SIM3_COMPAT_ALIASES:
                assert not hasattr(loaded, _retired_alias), _retired_alias
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
            # The alias round-trips that stood here (ne, v_plasma, Ne_flux,
            # S_ion_bulk, Qie, Qeb, and the cathode namespace's shape) were
            # views of rows this same block round-trips directly a few lines
            # down: the rhs_terms key set and one of its fields, the electron
            # energy-term dict, and four cathode_diagnostics keys. Retiring
            # them costs the file format no coverage it does not still have.
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
    # This block checks the surface-temperature diagnostics at a HELD
    # surface temperature; the power-balance warming is exercised in its own
    # block just below. An emitting-layer heat capacity no step's increment
    # survives holds the surface at cathode_Ts_base_K to the bit.
    cathode_run_params["cathode_heat_capacity_J_per_K"] = 1.0e30
    cathode_run_params["cathode_Ts_base_K"] = 1998.15
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
    # source_I_tot's FIRST sample is the initial state's solve, whose net
    # current is zero by construction: what is stored there is the residual of
    # the root the solve returned, so it is bounded by the solve's own current
    # scale rather than by the sign of an epsilon. Every LATER sample has been
    # stepped and carries a real forward current, and keeps the strict clause.
    _ns_I_tot = np.asarray(cathode_diag["source_I_tot"], dtype=float)
    _ns_I_e = np.asarray(cathode_diag["source_I_e"], dtype=float)
    assert _ns_I_tot[0] >= -1.0e-12 * abs(_ns_I_e[0]), _ns_I_tot[0]
    assert np.all(_ns_I_tot[1:] > 0.0), _ns_I_tot[1:]
    assert np.all(np.isfinite(cathode_diag["source_P_prim"]))
    assert np.all(np.isfinite(cathode_diag["source_P_ohmic"]))
    # The pre-closure ``source_P_loss`` this once checked is retired from the
    # export; its successors are the closed audit rows.
    assert np.all(np.isfinite(cathode_diag["source_P_plasma_thermal_loss"]))
    assert np.all(np.isfinite(cathode_diag["source_P_into_plasma"]))
    # Single cathode: EVERY ``end_*`` dataset is ABSENT (presence-gated on
    # TwinCathode), not present-and-seeded. The carve-out that used to stand
    # here exempted ``end_beam_*``, which was seeded from a literal tuple and
    # so shipped on single-cathode runs as five NaN and seven 0.0 columns
    # nothing could fill. There is no carve-out now: no prefix, no rows.
    assert "end_phi_c" not in cathode_diag
    # ONE row in this group legitimately begins with ``end_`` and is not a
    # twin-cathode dataset: ``end_wall_surface_power_W``, the far face's
    # surface-power ledger line, which every run carries. It is excluded BY
    # NAME rather than by loosening the prefix, and asserted PRESENT, so the
    # exemption is pinned to that one row instead of opening the test to any
    # future ``end_*`` name.
    assert "end_wall_surface_power_W" in cathode_diag
    assert not [
        k for k in cathode_diag
        if k.startswith("end_") and k != "end_wall_surface_power_W"
    ]
    assert np.all(
        np.isin(
            cathode_diag["source_regime"],
            ["classical", "virtual_cathode", "capability_limited"],
        )
    )
    assert "end_regime" not in cathode_diag
    assert cathode_diag["beam_cross"].shape == (4, geom.cells)
    # The static surface temperature is reported as the configured value.
    assert np.allclose(
        cathode_diag["T_s_surface"],
        float(cathode_run_params["cathode_Ts_base_K"]),
    )
    return locals()


# --------------------------------------------------------------------
# restart-saved-evidence-r1b
# --------------------------------------------------------------------
@_case(
    "restart-saved-evidence-r1b",
    historical_stance=True,
)
def _case_restart_saved_evidence_r1b(r1a_flags, r1a_params):
    # R1b: the saved evidence follows the actual packed state for the stable
    # five-, six- and seven-row layouts. Two-zone density inventory uses
    # V_col=V_p and V_ann=V_m-V_p.
    r1b_layouts = (
        ("six", {}, {}),
        ("seven", {}, {"neutral_momentum": True}),
    )
    with tempfile.TemporaryDirectory() as r1b_tmp:
        for expected_rows, (label, param_extra, flag_extra) in zip(
            (6, 7), r1b_layouts
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
                "En_energy_added_erg",
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
    r1c_params = dict(r1a_params)
    r1c_flags = dict(
        r1a_flags,
        neutral_momentum=True,
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
    ):
        try:
            LAPDSim1D(dict(r1a_params, **stale_param), r1a_flags)
        except ValueError as error:
            assert "silent no-ops" in str(error)
        else:
            raise AssertionError(
                f"expected frozen surface-control rejection: {stale_param}"
            )
    # A13 (R3.3, deleted at D3 2026-08-21): the four resolved-boundary
    # surface-loss controls were 0D artifacts standing in for un-separated
    # cathode/anode I_sat, and the resolved geometry measures the Bohm I_sat
    # to each electrode face directly. They now name no key in either
    # namespace, so the unknown-key refusal owns them -- in BOTH namespaces,
    # which is what makes the deletion loud rather than silent.
    for dep_params, dep_flags in (
        (dict(r1a_params, source_surface_area_scale=1.7), r1a_flags),
        (dict(r1a_params, end_surface_area_scale=0.9), r1a_flags),
        (r1a_params, dict(r1a_flags, source_surface_loss=False)),
        (r1a_params, dict(r1a_flags, end_surface_loss=False)),
    ):
        try:
            LAPDSim1D(dict(dep_params), dict(dep_flags))
        except ValueError as error:
            assert "unknown LAPDSim1D configuration keys" in str(error), error
        else:
            raise AssertionError(
                f"expected unknown-key rejection for {dep_params}/{dep_flags}"
            )
    for birth_name, bad_value in (
        ("Ti_birth_ionization", "bogus"),
        ("Ti_birth_ionization", -1.0),
        ("Ti_birth_ionization", np.inf),
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

    # The masked reaction rows: the bare reaction operator books an
    # ionization birth on the plasma-dead cells, and the typed topology
    # removes it from the summed RHS there.
    topo_on = LAPDSim1D(r1a_params, r1a_flags)
    topo_dead = ~topo_on.geometry.plasma_active
    assert np.any(
        topo_on.reaction_rhs_terms()["ionization_birth"].n[topo_dead] != 0.0
    )
    assert np.all(
        topo_on.rhs_terms()["ionization_birth"].n[topo_dead] == 0.0
    )


# --------------------------------------------------------------------
# resolved-config-manifest-r1e
# --------------------------------------------------------------------
@_case("resolved-config-manifest-r1e")
def _case_resolved_config_manifest_r1e():
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
        mismatch_params["Ti_birth_ionization"] = (
            0.5
            if case_params["Ti_birth_ionization"] == "neutral"
            else "neutral"
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
        guard_flags = {"ionization_energy_cost": False}
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
            ("flags", guard_params, {"ionization_energy_cost": True}),
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
                assert (
                    "Te0" if guard_kind == "params" else "ionization_energy_cost"
                ) in str(error)
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


# --------------------------------------------------------------------
# golden-digest-gate-deterministic
# --------------------------------------------------------------------
@_case("golden-digest-gate-deterministic")
def _case_golden_digest_gate_deterministic():
    # scripts/gates/golden_digest_gate.py is the short-horizon complement to the
    # golden: it folds the packed state into a running SHA-256 after every
    # accepted step. Its own 4,000-step gate is a minutes-long run and is NOT
    # run here -- what this case owns is that the module imports and that the
    # digest is REPRODUCIBLE, because a digest that is not deterministic in
    # process would report every merge as a divergence. A deliberately tiny
    # config (nx=12, no neutral equilibration, 25 steps) makes that a
    # sub-second check.
    import golden_digest_gate as _gdg

    assert _gdg.DIGEST_STEPS == 4000
    assert _gdg.CHECKPOINT_INTERVAL == 1000
    assert _gdg.DEFAULT_REFERENCE.name == "golden_digest_4k.json"

    _gdg_params, _gdg_flags = default_config()
    _gdg_params["nx"] = 12
    _gdg_params["max_steps_action"] = "stop"
    _gdg_params["initial_neutral_state"] = "fill"
    _gdg_run_kwargs = {"t_end": None, "dt": None, "operator_split": None}
    _gdg_a = _gdg.compute_digest(
        _gdg_params,
        _gdg_flags,
        steps=25,
        checkpoint_interval=10,
        run_kwargs=_gdg_run_kwargs,
    )
    _gdg_b = _gdg.compute_digest(
        _gdg_params,
        _gdg_flags,
        steps=25,
        checkpoint_interval=10,
        run_kwargs=_gdg_run_kwargs,
    )
    assert _gdg_a["steps"] == 25
    assert len(_gdg_a["digest"]) == 64
    assert sorted(_gdg_a["checkpoints"], key=int) == ["0", "10", "20"]
    assert _gdg_b["digest"] == _gdg_a["digest"], (
        _gdg_a["digest"], _gdg_b["digest"]
    )
    assert _gdg_b["checkpoints"] == _gdg_a["checkpoints"]
    assert _gdg_b["config_identity"] == _gdg_a["config_identity"]


# --------------------------------------------------------------------
# phase3-artifact-locator-battery
# --------------------------------------------------------------------
@_case("phase3-artifact-locator-battery")
def _case_phase3_artifact_locator_battery():
    # The Phase 3 locator schema is what keeps the COMMITTED provenance
    # record honest about an artifact that lives OUTSIDE the repository: the
    # record names it by a locator relative to a frozen base, and the loader
    # must refuse any record that names it some other way. The battery this
    # case promotes was run once against the archived 42 MB capture; here it
    # runs against a synthetic qualified capture built in a tempdir, so the
    # gate carries no dependency on a machine-local archive and costs ~0.1 s.
    #
    # Every negative is asserted to raise ValueError AND to NAME the offending
    # field: a refusal that does not say what was wrong sends the reader to
    # the wrong place, and a refusal for an unrelated reason would pass a
    # bare "it raised" check while proving nothing.
    _p3_run = "urn:uuid:123e4567-e89b-42d3-a456-426614174000"
    _p3_stem = _p3_run.removeprefix("urn:uuid:")
    _p3_rows = ("n", "nn", "M", "Ee", "Ei")
    _p3_terms = ("synthetic_flux",)
    _p3_frames, _p3_cells = 2, 3
    _p3_zero = np.zeros((_p3_frames, _p3_cells), dtype=float)
    _p3_result = SimpleNamespace(
        run_id=_p3_run,
        params={
            "synthetic": 1,
            "max_steps_action": "stop",
            "neutral_momentum_radial": "uniform",
        },
        flags={
            "neutral_momentum": False,
            "neutral_two_zone": False,
            "neutral_energy": False,
        },
        compiled_kernels="pure",
        steps=4000,
        final_time=1.0e-5,
        run_status="max_steps_reached",
        time=np.asarray([0.0, 1.0e-5]),
        y=np.arange(
            _p3_frames * len(_p3_rows) * _p3_cells, dtype=float
        ).reshape(_p3_frames, -1),
        n=_p3_zero, nn=_p3_zero, M=_p3_zero, momentum=_p3_zero,
        Ee=_p3_zero, Ei=_p3_zero, u=_p3_zero, Te=_p3_zero, Ti=_p3_zero,
        pe=_p3_zero, pi=_p3_zero, p=_p3_zero,
        z_cm=np.asarray([1.0, 2.0, 3.0]),
        length_cm=np.ones(_p3_cells),
        Rp_cm=np.ones(_p3_cells),
        Rm_cm=np.full(_p3_cells, 2.0),
        plasma_volume_cm3=np.full(_p3_cells, 2.0),
        neutral_volume_cm3=np.full(_p3_cells, 5.0),
        volume_ratio=np.full(_p3_cells, 2.5),
        plasma_active=np.ones(_p3_cells, dtype=bool),
        cell_role=np.asarray(["column"] * _p3_cells, dtype=object),
        rhs_terms={
            term: {
                row: np.full((_p3_frames, _p3_cells), float(i), dtype=float)
                for i, row in enumerate(_p3_rows)
            }
            for term in _p3_terms
        },
        total_rhs={row: _p3_zero for row in _p3_rows},
        electron_energy_terms_W_cm3={term: _p3_zero for term in _p3_terms},
        ion_energy_terms_W_cm3={term: _p3_zero for term in _p3_terms},
        diagnostics=[],
    )

    with tempfile.TemporaryDirectory() as _p3_tmp:
        _p3_root = Path(_p3_tmp)
        _p3_out = _p3_root / "scripts/baselines/phase3_rhs"
        reserve_run_id(_p3_out, _p3_run, {"kind": "smoke-synthetic"})
        _p3_h5, _p3_provenance = write_qualified_capture(
            _p3_out,
            _p3_result,
            run_id=_p3_run,
            capture_revision="a" * 40,
            producer_path="scripts/run/capture_phase3_rhs.py",
            started_at="2026-08-24T12:00:00Z",
            completed_at="2026-08-24T12:00:01Z",
            configuration_identity_sha256=_phase3_configuration_identity(
                _p3_result.params, _p3_result.flags
            ),
            recipe_identity="synthetic-recipe",
            run_controls={"max_steps": 4000},
            invocation=["python", "scripts/run/capture_phase3_rhs.py",
                        "--synthetic"],
            producer_blobs={"cablp/synthetic.py": "b" * 40},
            environment_lock={"path": "poetry.lock",
                              "git_blob": "c" * 40},
            repository_root=_p3_root,
        )
        _p3_record = json.loads(
            Path(_p3_provenance).read_text(encoding="utf-8")
        )
        # The POSITIVE, and it is load-bearing: without it a schema that
        # refused everything would pass all 15 negatives.
        _p3_loaded, _p3_prov = load_qualified_capture(_p3_h5, _p3_provenance)
        assert _p3_loaded.steps == 4000
        assert _p3_prov["run_id"] == _p3_run
        # The record emits the canonical locator, measured from the frozen
        # base, never from where the file happens to sit.
        assert _p3_record["artifact_locator_base"] == ARTIFACT_LOCATOR_BASE
        assert _p3_record["locator_state"] == ARTIFACT_LOCATOR_STATE
        assert _p3_record["artifact_path"] == (
            f"artifacts/phase3/{_p3_stem}/{_p3_stem}.h5"
        )

        def _p3_mutate(**overrides):
            record = dict(_p3_record)
            record.update(overrides)
            return record

        def _p3_drop(field):
            record = dict(_p3_record)
            del record[field]
            return record

        _p3_negatives = [
            # locator vocabulary: both bases are frozen sets of one
            ("unknown-base",
             _p3_mutate(artifact_locator_base="bapsf-workspace"),
             "artifact_locator_base"),
            ("unknown-state", _p3_mutate(locator_state="tracked"),
             "locator_state"),
            # artifact_path form: anything that could resolve against a
            # machine root, escape the base, or fail to be a path at all
            ("absolute-path",
             _p3_mutate(artifact_path=(
                 f"/home/trloo/bapsf/artifacts/phase3/{_p3_stem}/"
                 f"{_p3_stem}.h5"
             )),
             "artifact_path"),
            ("tilde-path",
             _p3_mutate(artifact_path=(
                 f"~/bapsf/artifacts/phase3/{_p3_stem}/{_p3_stem}.h5"
             )),
             "artifact_path"),
            ("traversing-path",
             _p3_mutate(artifact_path=(
                 "artifacts/phase3/" + ("%s/" % "..") * 3 + f"{_p3_stem}.h5"
             )),
             "artifact_path"),
            ("empty-component",
             _p3_mutate(artifact_path=(
                 f"artifacts//phase3/{_p3_stem}/{_p3_stem}.h5"
             )),
             "artifact_path"),
            ("drive-path",
             _p3_mutate(artifact_path=(
                 f"C:\\bapsf\\artifacts\\phase3\\{_p3_stem}.h5"
             )),
             "artifact_path"),
            ("non-string-path", _p3_mutate(artifact_path=None),
             "artifact_path"),
            # execution fields that must agree with the HDF5's own attrs
            ("accepted-steps-mismatch", _p3_mutate(accepted_steps=3999),
             "accepted_steps"),
            ("run-status-mismatch", _p3_mutate(run_status="completed"),
             "run_status"),
        ] + [
            # every required field, absent
            (f"missing-{_field}", _p3_drop(_field), _field)
            for _field in _LOCATOR_REQUIRED_FIELDS
        ]
        assert len(_p3_negatives) == 15, len(_p3_negatives)

        _p3_scratch = _p3_root / "battery"
        _p3_scratch.mkdir()
        for _p3_name, _p3_bad, _p3_field in _p3_negatives:
            _p3_path = _p3_scratch / f"{_p3_name}.provenance.json"
            _p3_path.write_text(
                json.dumps(_p3_bad, indent=4) + "\n", encoding="utf-8"
            )
            try:
                load_qualified_capture(_p3_h5, _p3_path)
            except ValueError as _p3_exc:
                assert _p3_field in str(_p3_exc), (_p3_name, str(_p3_exc))
            else:
                raise AssertionError(
                    f"locator negative {_p3_name!r} was ACCEPTED"
                )


# --------------------------------------------------------------------
# golden-fixture-packed-row-count
# --------------------------------------------------------------------
@_case("golden-fixture-packed-row-count", historical_stance=True)
def _case_golden_fixture_packed_row_count():
    # STATE_NAMES_1D IS NOT A ROW COUNT. It names the five always-packed rows,
    # the fixed head of the layout; the optional rows that follow are packed
    # BY PRESENCE, so the packed width of any given run is a property of its
    # flags and not of the tuple. The golden fixtures record a
    # fields_per_cell of their own, and nothing tied the two together -- a
    # reader who took len(STATE_NAMES_1D) for the fixture's row count would be
    # wrong about the shipped reference configuration today, which packs a
    # sixth row.
    #
    # The tie asserted here follows the stance by construction: it builds no
    # config of its own (build_baseline_config()) and reads the widths the
    # sidecars actually recorded, so a stance event that adds or drops an
    # optional row moves both sides together and a change that moves only one
    # of them fails here.
    from baseline_sim1d import build_baseline_config

    _pr_params, _pr_flags = build_baseline_config()
    _pr_names = state_field_names(LAPDSim1D(_pr_params, _pr_flags).state)

    # The head is the five always-packed rows, in order.
    assert len(STATE_NAMES_1D) == 5, STATE_NAMES_1D
    assert _pr_names[: len(STATE_NAMES_1D)] == tuple(STATE_NAMES_1D), _pr_names
    # ... and the reference configuration packs strictly more than the head,
    # which is the whole reason the tuple cannot serve as the row count.
    assert len(_pr_names) > len(STATE_NAMES_1D), _pr_names

    _pr_baselines = Path(__file__).resolve().parents[2] / "baselines"
    _pr_widths = {}
    for _pr_side in ("production_discharge.json", "golden_digest_4k.json"):
        with open(_pr_baselines / _pr_side) as _pr_fh:
            _pr_widths[_pr_side] = int(json.load(_pr_fh)["fields_per_cell"])
        assert _pr_widths[_pr_side] == len(_pr_names), (
            f"{_pr_side} records fields_per_cell="
            f"{_pr_widths[_pr_side]} while the stance of record packs "
            f"{len(_pr_names)} rows {_pr_names}"
        )
    # The two fixtures are captured from the same configuration, so they can
    # never disagree with each other about the width either.
    assert len(set(_pr_widths.values())) == 1, _pr_widths


# ----------------------------------------------------------------------
# cell-role-whitelist-on-load
# ----------------------------------------------------------------------
@_case("cell-role-whitelist-on-load")
def _case_cell_role_whitelist_on_load():
    # THE ROLE VOCABULARY IS CLOSED, ON BOTH BOUNDARIES. `CELL_ROLES` names
    # every string a resolved mesh may carry in `cell_role`. A load that met
    # any other name used to return it in silence, and silence is the whole
    # hazard: every role-keyed selection downstream -- `== "end_wall"`, the
    # plasma-dead mask, the recycle routing -- then selects NOTHING, so the
    # artifact reads like a legal run with its far end quietly missing.
    #
    # The check runs AFTER the retired-name alias map, so the two statements
    # compose: a pre-rename artifact still reads through
    # `LEGACY_CELL_ROLE_ALIASES`, and a name that table does not map -- a name
    # never issued, or one whose alias entry regressed -- is REFUSED.
    import cablp.solvers._sim1d.results.io as _cw_io
    from cablp.solvers._sim1d.core.geometry import (
        CELL_ROLES,
        PLASMA_DEAD_ROLES,
        _assert_known_cell_roles,
    )
    from cablp.solvers._sim1d.results.io import (
        CELL_ROLE_SHIM_ATTR,
        save_result_hdf5,
    )

    # (iv) THE REGISTRY CONTRACT. The plasma-dead roles are a subset of the
    # accepted set -- a role added to one and forgotten in the other would
    # make a live mask name something no mesh can carry -- and the retired
    # far-end name is deliberately NOT a member: it reaches a current name
    # through the alias table or not at all.
    assert PLASMA_DEAD_ROLES <= CELL_ROLES, PLASMA_DEAD_ROLES - CELL_ROLES
    assert "collector" not in CELL_ROLES
    # ...and the builder-side half of "cannot desync" refuses too, which is
    # what keeps the assembly and the registry moving in one commit.
    try:
        _assert_known_cell_roles(np.asarray(["column", "sausage"], dtype=object))
    except ValueError as _cw_bexc:
        assert "sausage" in str(_cw_bexc), _cw_bexc
    else:
        raise AssertionError("the builder accepted an unregistered role")

    # NO SOLVE: t_end = 0.0 writes the initial state and stops, which is all a
    # role round-trip needs. The equilibration flag is cleared because run()
    # does not equilibrate and says so loudly; nothing here reads nn.
    _cw_params, _cw_flags = default_config()
    _cw_params["initial_neutral_state"] = "fill"
    _cw_result = LAPDSim1D(_cw_params, _cw_flags).run(t_end=0.0)

    with tempfile.TemporaryDirectory() as _cw_tmp:
        _cw_room = Path(_cw_tmp)

        def _cw_written(name, role=None):
            """Save the result, optionally overwriting its LAST stored role."""
            target = _cw_room / name
            save_result_hdf5(target, _cw_result)
            if role is not None:
                with h5py.File(target, "r+") as _cw_file:
                    _cw_file["geometry/cell_role"][-1] = role
            return target

        # (i) A FRESHLY WRITTEN FILE ROUND-TRIPS UNTOUCHED. Every role it
        # carries is registered, so neither the alias map nor the whitelist
        # has anything to do and the shim attribute says so.
        _cw_h5 = _cw_written("roles.h5")
        _cw_loaded = load_result_hdf5(_cw_h5)
        _cw_roles = {str(_cw_r) for _cw_r in _cw_loaded.cell_role}
        assert _cw_roles <= CELL_ROLES, _cw_roles - CELL_ROLES
        assert getattr(_cw_loaded, CELL_ROLE_SHIM_ATTR) is False
        assert str(_cw_loaded.cell_role[-1]) == "end_wall"

        # (ii) THE SAME FILE WITH ONE ROLE REWRITTEN IN PLACE to a string no
        # mesh ever issued. The refusal names the file, the offending string
        # and the accepted set, and points at the one route a retired name
        # has. `endwall` is chosen to be one underscore from a real role:
        # a typo is exactly how this arrives in practice.
        _cw_bad = _cw_written("unknown_role.h5", "endwall")
        try:
            load_result_hdf5(_cw_bad)
        except ValueError as _cw_exc:
            _cw_msg = str(_cw_exc)
        else:
            raise AssertionError("an unregistered cell_role was ACCEPTED")
        assert "endwall" in _cw_msg, _cw_msg
        assert str(_cw_bad) in _cw_msg, _cw_msg
        assert "'end_wall'" in _cw_msg, _cw_msg
        assert "LEGACY_CELL_ROLE_ALIASES" in _cw_msg, _cw_msg

        # (iii) A PRE-RENAME ARTIFACT still reads, through the alias table,
        # and the load reports that it did.
        _cw_old = _cw_written("pre_rename.h5", "collector")
        _cw_old_loaded = load_result_hdf5(_cw_old)
        assert getattr(_cw_old_loaded, CELL_ROLE_SHIM_ATTR) is True
        assert str(_cw_old_loaded.cell_role[-1]) == "end_wall"
        assert {
            str(_cw_r) for _cw_r in _cw_old_loaded.cell_role
        } <= CELL_ROLES

        # ...AND THE SAME FILE WITH THE ALIAS TABLE EMPTIED. This is the
        # scenario the whitelist exists for: a retired name whose alias entry
        # is gone is not a name the reader can act on, and before the
        # whitelist it loaded in silence. Emptying the table is the only way
        # to show that the refusal is what stands behind the alias map rather
        # than a second copy of it.
        assert _cw_io.LEGACY_CELL_ROLE_ALIASES == {"collector": "end_wall"}
        _cw_kept_aliases = _cw_io.LEGACY_CELL_ROLE_ALIASES
        _cw_io.LEGACY_CELL_ROLE_ALIASES = {}
        try:
            load_result_hdf5(_cw_old)
        except ValueError as _cw_rexc:
            _cw_rmsg = str(_cw_rexc)
        else:
            raise AssertionError(
                "'collector' was ACCEPTED with the alias table emptied"
            )
        finally:
            _cw_io.LEGACY_CELL_ROLE_ALIASES = _cw_kept_aliases
        assert "collector" in _cw_rmsg, _cw_rmsg
        assert "LEGACY_CELL_ROLE_ALIASES" in _cw_rmsg, _cw_rmsg

        # The restore is a fact to assert, not a hope: the table is back and
        # the very same file reads again.
        assert _cw_io.LEGACY_CELL_ROLE_ALIASES == {"collector": "end_wall"}
        assert getattr(
            load_result_hdf5(_cw_old), CELL_ROLE_SHIM_ATTR
        ) is True


# ----------------------------------------------------------------------
# far-end-double-ratio-area-cancels
# ----------------------------------------------------------------------
@_case("far-end-double-ratio-area-cancels")
def _case_far_end_double_ratio_area_cancels():
    """The probe-A area factor must divide OUT of the far-end double ratio.

    ``scripts/score/far_end_double_ratio.py`` reports
    D = (X50/X41) / (X11/X21) precisely because p11 and p50 carry ONE probe's
    area calibration: a common multiplicative factor on those two rows enters
    the numerator of both sub-ratios and cancels. That is the instrument's
    whole claim to being area-free, and it is a property of the arithmetic,
    so it is checked numerically rather than argued.

    The perturbation is the one a re-calibrated export would actually write:
    the p11 and p50 paired AREAS are scaled by a common factor (with their
    pairing strings rewritten to match, since the scorer's upstream-column
    invariant compares the two), and every already-area-normalized p11/p50
    row -- the two density conventions and the geomean current density -- is
    scaled by its reciprocal. Every metric's D must be unchanged to 1e-12.

    The NEGATIVE CONTROL scales p50 alone. That is not an area recalibration
    of one probe and must move D by exactly the factor, so a version of this
    case that compared nothing would fail it.
    """
    import far_end_double_ratio as _fedr

    overlay_npz = np.load(
        Path(__file__).resolve().parents[2] / "data" / "es1_sim1d_overlay.npz",
        allow_pickle=False,
    )
    base_overlay = {key: overlay_npz[key] for key in overlay_npz.files}

    # A synthetic trajectory whose port cells sit exactly on the overlay's own
    # z_cm, with a z- and t-dependent n and Te so no ratio is degenerate.
    z = np.asarray(base_overlay["z_cm"], dtype=float)
    t_s = np.arange(0.0, 25.0e-3 + 1.0e-9, 1.0e-4)
    shape = 1.0 + 0.3 * np.cos(z / 600.0)[None, :]
    ramp = (1.0 + 0.05 * np.sin(t_s * 400.0))[:, None]
    synthetic = SimpleNamespace(
        time=t_s,
        phase=np.array(["main_discharge"] * t_s.size),
        z_cm=z,
        n=1.0e13 * shape * ramp,
        Te=4.0 * shape**2 * ramp,
    )

    def _scaled(overlay, factor, ports):
        """Return a copy with a common area recalibration on ``ports``."""
        out = {k: (v.copy() if isinstance(v, np.ndarray) else v)
               for k, v in overlay.items()}
        port_list = [int(p) for p in np.asarray(overlay["port"])]
        geo_ports = [int(p) for p in
                     np.asarray(overlay["isat_ftavg_geomean_port"])]
        pairing = np.asarray(overlay["isat_ftavg_geomean_pairing"]).astype(object)
        areas = np.asarray(
            overlay["isat_ftavg_geomean_area_cm2"], dtype=float
        ).copy()
        for port in ports:
            i = geo_ports.index(port)
            areas[i] *= factor
            # The pairing string is the exporter's own record of those two
            # columns; the scorer checks the two against each other, so a
            # recalibration that moved only the array would be caught.
            segments = [s.strip() for s in str(pairing[i]).split(" x ")]
            rewritten = []
            for seg in segments:
                head, _, tail = seg.partition("=")
                value = float(tail.split(" ")[0]) * factor
                rewritten.append(f"{head}={value:.6f} cm2")
            pairing[i] = " x ".join(rewritten)
            for key in ("density_mean_cm3", "density_ftavg_cm3"):
                out[key] = np.asarray(out[key], dtype=float)
                out[key][port_list.index(port)] /= factor
            out["isat_ftavg_geomean_a_per_cm2"] = np.asarray(
                out["isat_ftavg_geomean_a_per_cm2"], dtype=float
            )
            out["isat_ftavg_geomean_a_per_cm2"][i] /= factor
        out["isat_ftavg_geomean_area_cm2"] = areas
        out["isat_ftavg_geomean_pairing"] = np.asarray(pairing, dtype=str)
        return out

    base_rows, base_skip = _fedr.double_ratio_rows(synthetic, base_overlay)
    assert base_skip is None, base_skip
    assert len(base_rows) == len(_fedr.METRIC_SPECS), base_rows
    assert all(np.isfinite(r["D_measured"]) and r["D_measured"] != 0.0
               for r in base_rows), base_rows

    factor = 1.7
    cancels = _scaled(base_overlay, factor, _fedr.NEAR_PAIR[1:] + _fedr.FAR_PAIR[1:])
    cancel_rows, cancel_skip = _fedr.double_ratio_rows(synthetic, cancels)
    assert cancel_skip is None, cancel_skip
    for base, moved in zip(base_rows, cancel_rows):
        assert moved["metric"] == base["metric"], (base, moved)
        for key in ("D_measured", "D_model", "D_ratio"):
            assert abs(moved[key] / base[key] - 1.0) <= 1.0e-12, (
                base["metric"], key, base[key], moved[key]
            )

    # Negative control: p50 alone is not one probe's recalibration, and every
    # metric's measured D must move by exactly 1/factor.
    one_port = _scaled(base_overlay, factor, _fedr.FAR_PAIR[1:])
    control_rows, control_skip = _fedr.double_ratio_rows(synthetic, one_port)
    assert control_skip is None, control_skip
    for base, moved in zip(base_rows, control_rows):
        assert abs(
            moved["D_measured"] / base["D_measured"] - 1.0 / factor
        ) <= 1.0e-12, (base["metric"], base["D_measured"], moved["D_measured"])



# ----------------------------------------------------------------------
# far-end-double-ratio-per-family-gating
# ----------------------------------------------------------------------
@_case("far-end-double-ratio-per-family-gating")
def _case_far_end_double_ratio_per_family_gating():
    """One overlay family's absence must not withhold the other's rows.

    ``double_ratio_rows`` used to gate the density and J (Isat) metric
    families as one all-or-nothing union: a missing J-family key dropped the
    density metrics too, against the function's own docstring ("rows is
    empty only when NO metric could be formed"). This checks the fix in both
    directions on synthetic overlays built from the real ES1 overlay's key
    set with one family's keys deleted.
    """
    import far_end_double_ratio as _fedr

    overlay_npz = np.load(
        Path(__file__).resolve().parents[2] / "data" / "es1_sim1d_overlay.npz",
        allow_pickle=False,
    )
    base_overlay = {key: overlay_npz[key] for key in overlay_npz.files}

    z = np.asarray(base_overlay["z_cm"], dtype=float)
    t_s = np.arange(0.0, 25.0e-3 + 1.0e-9, 1.0e-4)
    shape = 1.0 + 0.3 * np.cos(z / 600.0)[None, :]
    ramp = (1.0 + 0.05 * np.sin(t_s * 400.0))[:, None]
    synthetic = SimpleNamespace(
        time=t_s,
        phase=np.array(["main_discharge"] * t_s.size),
        z_cm=z,
        n=1.0e13 * shape * ramp,
        Te=4.0 * shape**2 * ramp,
    )

    # J family absent (density's port/z_cm identity columns kept, since they
    # are the shared z-lookup every metric reads, density or J alike).
    j_absent = {
        k: v for k, v in base_overlay.items() if k not in _fedr.J_OVERLAY_KEYS
    }
    j_absent_rows, j_absent_skip = _fedr.double_ratio_rows(synthetic, j_absent)
    assert {r["metric"] for r in j_absent_rows} == {"n", "n_ft"}, j_absent_rows
    assert j_absent_skip is not None and "J family" in j_absent_skip, (
        j_absent_skip
    )
    for key in _fedr.J_OVERLAY_KEYS:
        assert key in j_absent_skip, (key, j_absent_skip)

    # report_double_ratio must print the density rows without raising, even
    # though the overlay carries none of the J-family keys the legend used
    # to read unguarded.
    _out = StringIO()
    with contextlib.redirect_stdout(_out):
        _fedr.report_double_ratio(
            "smoke: J family absent", j_absent_rows, j_absent_skip,
            j_absent, _fedr._cmp.PLATEAU_MS,
        )
    _text = _out.getvalue()
    assert "(a) core-band density n" in _text, _text
    assert "(b) flux-tube density n_ft" in _text, _text
    assert "port rows and the faces behind them" not in _text, _text

    # Reverse: density's own (non-shared) measured fields absent, J intact.
    density_only_keys = tuple(
        k for k in _fedr.DENSITY_OVERLAY_KEYS if k not in ("port", "z_cm")
    )
    density_absent = {
        k: v for k, v in base_overlay.items() if k not in density_only_keys
    }
    density_absent_rows, density_absent_skip = _fedr.double_ratio_rows(
        synthetic, density_absent
    )
    assert {r["metric"] for r in density_absent_rows} == {
        "J_upstream", "J_geomean",
    }, density_absent_rows
    assert density_absent_skip is not None
    assert "density family" in density_absent_skip, density_absent_skip
    for key in density_only_keys:
        assert key in density_absent_skip, (key, density_absent_skip)


# ----------------------------------------------------------------------
# far-end-double-ratio-empty-legend-guard
# ----------------------------------------------------------------------
@_case("far-end-double-ratio-empty-legend-guard")
def _case_far_end_double_ratio_empty_legend_guard():
    """``report_double_ratio`` must not crash when no metric could be formed.

    ``report_double_ratio`` used to call ``_legend_lines`` -- which reads
    ``isat_ftavg_geomean_port`` unguarded -- BEFORE its ``if not rows`` skip
    branch, so an overlay carrying no usable rows raised instead of printing
    the skip line. This overlay has both metric families unusable (the J
    family's keys are absent outright, and density's own fields are absent
    too, though the shared port/z_cm columns survive) so ``double_ratio_rows``
    forms zero rows, and the report call must print the skip line and raise
    nothing.
    """
    import far_end_double_ratio as _fedr

    overlay_npz = np.load(
        Path(__file__).resolve().parents[2] / "data" / "es1_sim1d_overlay.npz",
        allow_pickle=False,
    )
    base_overlay = {key: overlay_npz[key] for key in overlay_npz.files}

    z = np.asarray(base_overlay["z_cm"], dtype=float)
    t_s = np.arange(0.0, 25.0e-3 + 1.0e-9, 1.0e-4)
    synthetic = SimpleNamespace(
        time=t_s,
        phase=np.array(["main_discharge"] * t_s.size),
        z_cm=z,
        n=1.0e13 * np.ones((t_s.size, z.size)),
        Te=4.0 * np.ones((t_s.size, z.size)),
    )

    density_only_keys = tuple(
        k for k in _fedr.DENSITY_OVERLAY_KEYS if k not in ("port", "z_cm")
    )
    no_rows_overlay = {
        k: v for k, v in base_overlay.items()
        if k not in _fedr.J_OVERLAY_KEYS and k not in density_only_keys
    }

    rows, skip_reason = _fedr.double_ratio_rows(synthetic, no_rows_overlay)
    assert rows == [], rows
    assert skip_reason is not None
    assert "J family" in skip_reason and "density family" in skip_reason, (
        skip_reason
    )

    _out = StringIO()
    with contextlib.redirect_stdout(_out):
        _fedr.report_double_ratio(
            "smoke: no usable rows", rows, skip_reason, no_rows_overlay,
            _fedr._cmp.PLATEAU_MS,
        )
    _text = _out.getvalue()
    assert "(no metric formed)" in _text, _text
    assert "J family" in _text and "density family" in _text, _text


# ----------------------------------------------------------------------
# far-end-double-ratio-shared-keys-skip
# ----------------------------------------------------------------------
@_case("far-end-double-ratio-shared-keys-skip", provides=())
def _case_far_end_double_ratio_shared_keys_skip():
    """A missing shared z-lookup key must skip cleanly, not raise.

    ``double_ratio_rows`` builds ``z_by_port`` from the overlay's ``port``
    and ``z_cm`` columns unconditionally -- every metric family reads it, so
    it is not one of the per-family ``FAMILY_OVERLAY_KEYS``. Before the
    shared-key prerequisite was hoisted out of ``DENSITY_OVERLAY_KEYS``, an
    overlay missing ``port`` alone would skip the density family (``port``
    sat in ``DENSITY_OVERLAY_KEYS``) but leave the J family looking present,
    so the function fell through to ``z_by_port`` and raised ``KeyError:
    'port'`` instead of returning a clean skip. This overlay keeps every
    other key (both families' own fields intact) and removes only ``port``.
    """
    import far_end_double_ratio as _fedr

    overlay_npz = np.load(
        Path(__file__).resolve().parents[2] / "data" / "es1_sim1d_overlay.npz",
        allow_pickle=False,
    )
    base_overlay = {key: overlay_npz[key] for key in overlay_npz.files}

    z = np.asarray(base_overlay["z_cm"], dtype=float)
    t_s = np.arange(0.0, 25.0e-3 + 1.0e-9, 1.0e-4)
    synthetic = SimpleNamespace(
        time=t_s,
        phase=np.array(["main_discharge"] * t_s.size),
        z_cm=z,
        n=1.0e13 * np.ones((t_s.size, z.size)),
        Te=4.0 * np.ones((t_s.size, z.size)),
    )

    port_absent = {k: v for k, v in base_overlay.items() if k != "port"}

    rows, skip_reason = _fedr.double_ratio_rows(synthetic, port_absent)
    assert rows == [], rows
    assert skip_reason is not None
    assert "port" in skip_reason, skip_reason


# ----------------------------------------------------------------------
# result-bitdiff-compare-synthetic
# ----------------------------------------------------------------------
@_case("result-bitdiff-compare-synthetic")
def _case_result_bitdiff_compare_synthetic():
    # The full-result bit-diff comparator on tiny synthetic files, no solve:
    # identical files pass, and a signed zero, an attribute and an extra
    # dataset are each reported as exactly one finding naming its path.
    import result_bitdiff

    def write(path, zero=0.0, attr=3, extra=False):
        str_dtype = h5py.string_dtype(encoding="utf-8")
        with h5py.File(path, "w") as h5:
            h5.attrs["format"] = "synthetic"
            h5.attrs["steps"] = 7
            h5.create_dataset("y", data=np.array([[1.0, zero], [2.5, -3.0]]))
            group = h5.create_group("rhs_terms/ionization")
            group.attrs["count"] = attr
            group.create_dataset("n", data=np.arange(4, dtype=np.int64))
            h5.create_dataset(
                "phase", data=np.asarray(["a", "b"], dtype=object),
                dtype=str_dtype,
            )
            if extra:
                h5.create_dataset("rhs_terms/extra", data=np.zeros(2))

    compare = result_bitdiff.compare_files

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        paths = {
            "base": tmp / "base.h5",
            "twin": tmp / "twin.h5",
            "negzero": tmp / "negzero.h5",
            "attr": tmp / "attr.h5",
            "extra": tmp / "extra.h5",
        }
        write(paths["base"])
        write(paths["twin"])
        write(paths["negzero"], zero=-0.0)
        write(paths["attr"], attr=4)
        write(paths["extra"], extra=True)

        report = compare(paths["base"], paths["twin"])
        assert report.identical, report.findings
        assert report.objects == 6 and report.attributes == 3, report
        # The value comparison np.array_equal would make passes the signed
        # zero; the raw-byte comparison must not.
        with h5py.File(paths["negzero"], "r") as h5:
            assert np.array_equal(h5["y"][()], np.array([[1.0, 0.0], [2.5, -3.0]]))
        expected = {
            "negzero": "/y BYTES differ at 1 of 4 elements; first at flat index 1",
            "attr": "/rhs_terms/ionization@count BYTES differ",
            "extra": "/rhs_terms/extra: dataset present in B only",
        }
        for name, prefix in expected.items():
            report = compare(paths["base"], paths[name])
            assert len(report.findings) == 1, (name, report.findings)
            assert report.findings[0].startswith(prefix), (
                name, report.findings)
        # The command-line mode exits on the same verdicts.
        for name, want in (("twin", 0), ("negzero", 1)):
            proc = subprocess.run(
                [sys.executable, str(Path(result_bitdiff.__file__)),
                 "compare", str(paths["base"]), str(paths[name])],
                capture_output=True, text=True,
            )
            assert proc.returncode == want, (name, proc.stdout, proc.stderr)
