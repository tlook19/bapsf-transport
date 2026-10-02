"""Smoke cases: breakdown, ignition, the prescribed drive and the afterglow
tail hand-off.
"""

import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import warnings

import h5py
import numpy as np

from cablp.constants import m_He_cgs
from cablp.solvers._sim1d import (
    BreakdownError,
    LAPDSim1D,
    TimestepRejectionError,
    default_config,
    load_result_hdf5,
    summarize_result,
)
from cablp.solvers._sim1d.core.state import STATE_NAMES_1D

from ._harness import (
    _TOL_ROUNDOFF,
    _base_config,
    _base_sim,
    _case,
    _cathode_unit_config,
)


# --------------------------------------------------------------------
# breakdown-retry-near-vacuum
# --------------------------------------------------------------------
@_case(
    "breakdown-retry-near-vacuum",
    historical_stance=True,
    provides=(
        "current_phase_flags", "current_phase_params",
        "current_phase_result", "direct_current_phase_result",
    ),
)
def _case_breakdown_retry_near_vacuum(
    no_source_params, retry_flags, retry_params, run_params, split_flags
):
    # This scenario needs a near-vacuum start: the retry it exercises fires
    # when one puff step moves nn by more than max_neutral_step_fraction, which
    # is only possible against a tiny background. It used to inherit that from
    # the nn0 default; nn0 is now the realistic direct-run fill (2e13), against
    # which the same puff is a ~1e-4 fractional step and nothing is ever
    # rejected. Pin the initial condition the scenario is built on, alongside
    # the other limiter constants it already sets.
    params, flags = _base_config()
    sim, snapshot = _base_sim()
    geom = snapshot.geometry
    retry_params["nn0"] = 1.0e9
    # The puff feeds the ANNULUS (the two-zone split is unconditional) and the
    # limiter reads the COLUMN, which fills only through the zone exchange: a
    # 1 us step moves it by 3.8e-2, the retried 0.5 us half by 9.8e-3 and the
    # second half by 2.8e-2. The threshold sits between them, so the first
    # step retries once and the second is accepted as it stands.
    retry_params["max_neutral_step_fraction"] = 0.033
    retry_sim = LAPDSim1D(retry_params, retry_flags)
    retry_result = retry_sim.run(t_end=1.0e-6)
    assert retry_result.steps >= 2
    assert np.isclose(retry_result.time[-1], 1.0e-6, **_TOL_ROUNDOFF)
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
            **_TOL_ROUNDOFF,
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
        assert np.isclose(exc.time, 0.0, **_TOL_ROUNDOFF)
        assert np.isclose(exc.attempted_dt, 0.5e-6, **_TOL_ROUNDOFF)
        assert np.isclose(exc.dt_min, failed_retry_params["dt_min"], **_TOL_ROUNDOFF)
        assert exc.phase == "equilibrium_puff"
        assert exc.active_constraint == "dt_max"
        assert np.isclose(failed_retry_sim.time, 0.0, **_TOL_ROUNDOFF)
        assert np.allclose(
            failed_retry_sim.get_initial_snapshot().y, failed_retry_y0, **_TOL_ROUNDOFF,
        )
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
    assert np.isclose(split_run_result.final_time, 2.0e-10, **_TOL_ROUNDOFF)
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
        **_TOL_ROUNDOFF,
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
        **_TOL_ROUNDOFF,
    )
    assert np.allclose(
        phase_result.phase_cathode_enabled,
        [0.0, 0.0, 0.0, 0.0, 0.0],
        **_TOL_ROUNDOFF,
    )
    assert np.allclose(
        phase_result.phase_gas_puff_enabled,
        [0.0, 0.0, 0.0, 0.0, 0.0],
        **_TOL_ROUNDOFF,
    )
    assert np.allclose(
        phase_result.phase_floating,
        [0.0, 0.0, 0.0, 1.0, 0.0],
        **_TOL_ROUNDOFF,
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
        **_TOL_ROUNDOFF,
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
        **_TOL_ROUNDOFF,
    )

    phase_cathode_flags = dict(flags)
    phase_cathode_flags["cathode_coupling"] = True
    phase_cathode_sim = LAPDSim1D(phase_params, phase_cathode_flags)
    phase_cathode_result = phase_cathode_sim.run(t_end=4.0e-10, dt=1.0e-10)
    assert np.allclose(
        phase_cathode_result.phase_cathode_enabled,
        [1.0, 1.0, 1.0, 0.0, 0.0],
        **_TOL_ROUNDOFF,
    )
    assert np.allclose(
        phase_cathode_result.cathode_diagnostics["configured"],
        [1.0, 1.0, 1.0, 1.0, 1.0],
        **_TOL_ROUNDOFF,
    )
    assert np.allclose(
        phase_cathode_result.cathode_diagnostics["phase_enabled"],
        [1.0, 1.0, 1.0, 0.0, 0.0],
        **_TOL_ROUNDOFF,
    )
    assert np.allclose(
        phase_cathode_result.cathode_diagnostics["rhs_enabled"],
        [1.0, 1.0, 1.0, 0.0, 0.0],
        **_TOL_ROUNDOFF,
    )
    assert np.allclose(
        phase_cathode_result.cathode_diagnostics["solve_enabled"],
        [1.0, 1.0, 1.0, 1.0, 0.0],
        **_TOL_ROUNDOFF,
    )
    assert np.allclose(
        phase_cathode_result.cathode_diagnostics["floating"],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        **_TOL_ROUNDOFF,
    )
    assert np.allclose(
        phase_cathode_result.cathode_diagnostics["has_solution"],
        [1.0, 1.0, 1.0, 1.0, 0.0],
        **_TOL_ROUNDOFF,
    )
    assert np.allclose(
        phase_cathode_result.rhs_terms["cathode_surface_loss"]["n"][3:],
        0.0,
        **_TOL_ROUNDOFF,
    )
    assert np.allclose(
        phase_cathode_result.rhs_terms["beam_ionization_birth"]["n"][3:],
        0.0,
        **_TOL_ROUNDOFF,
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
        **_TOL_ROUNDOFF,
    )
    assert np.isclose(
        breakdown_sim.next_phase_boundary_after(1.0e-10),
        2.0e-10,
        **_TOL_ROUNDOFF,
    )
    breakdown_result = breakdown_sim.run(t_end=5.0e-10, dt=1.0e-10)
    assert np.allclose(
        breakdown_result.time,
        [0.0, 1.0e-10, 2.0e-10, 3.0e-10, 4.0e-10, 5.0e-10],
        **_TOL_ROUNDOFF,
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
        **_TOL_ROUNDOFF,
    )
    assert np.allclose(
        breakdown_result.phase_cathode_enabled,
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        **_TOL_ROUNDOFF,
    )
    assert np.allclose(
        breakdown_result.phase_gas_puff_enabled,
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        **_TOL_ROUNDOFF,
    )
    assert np.allclose(
        breakdown_result.phase_floating,
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        **_TOL_ROUNDOFF,
    )
    assert np.allclose(
        breakdown_result.phase_events["time"],
        [0.0, 1.0e-10, 2.0e-10, 4.0e-10, 5.0e-10],
        **_TOL_ROUNDOFF,
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
        **_TOL_ROUNDOFF,
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
        **_TOL_ROUNDOFF,
    )
    assert np.allclose(
        breakdown_capped_result.phase_events["time"],
        [0.0, 1.0e-10, 2.0e-10, 4.0e-10, 5.0e-10],
        **_TOL_ROUNDOFF,
    )

    breakdown_mid_sim = LAPDSim1D(breakdown_params, flags)
    breakdown_mid_sim.run(t_end=1.0e-10, dt=1.0e-10)
    breakdown_mid_result = breakdown_mid_sim.run(t_end=5.0e-10, dt=1.0e-10)
    assert np.allclose(
        breakdown_mid_result.phase_events["time"],
        [1.0e-10, 2.0e-10, 4.0e-10, 5.0e-10],
        **_TOL_ROUNDOFF,
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
        **_TOL_ROUNDOFF,
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
    neutral_prebreakdown_sim = LAPDSim1D(
        neutral_prebreakdown_params,
        neutral_prebreakdown_flags,
    )
    neutral_prebreakdown_result = neutral_prebreakdown_sim.run(dt=1.0e-10)
    # Expected schedule, built from the parameters. The plasma phases start at
    # tau_neutral_prebreakdown; the first driven step runs from there to one
    # dt later and ends with the loop current above I_prebreakdown, so the
    # pre-breakdown trigger interpolates between the trigger samples at those
    # two times; the breakdown trigger is the next accepted step (its previous
    # sample is already above I_breakdown, so it is not interpolated); the
    # run ends tau_discharge + tau_afterglow after breakdown.
    _np_dt = 1.0e-10
    _np_origin = neutral_prebreakdown_params["tau_neutral_prebreakdown"]
    _np_I_pre = neutral_prebreakdown_params["I_prebreakdown"]
    _np_expected_breakdown = _np_origin + 2.0 * _np_dt
    _np_expected_end = (
        _np_expected_breakdown
        + neutral_prebreakdown_params["tau_discharge"]
        + neutral_prebreakdown_params["tau_afterglow"]
    )
    assert np.isclose(
        neutral_prebreakdown_result.t_breakdown_trigger,
        _np_expected_breakdown,
        **_TOL_ROUNDOFF,
    )
    assert np.isclose(
        neutral_prebreakdown_result.final_time, _np_expected_end, **_TOL_ROUNDOFF
    )
    _np_samples_t = neutral_prebreakdown_result.current_trigger_samples["time"]
    _np_samples_I = neutral_prebreakdown_result.current_trigger_samples["I_tot"]
    assert np.allclose(
        _np_samples_t[:2], [_np_origin, _np_origin + _np_dt], **_TOL_ROUNDOFF
    ), _np_samples_t
    assert _np_samples_I[0] < _np_I_pre <= _np_samples_I[1], _np_samples_I
    _np_expected_prebreakdown = _np_samples_t[0] + (
        (_np_I_pre - _np_samples_I[0]) / (_np_samples_I[1] - _np_samples_I[0])
    ) * (_np_samples_t[1] - _np_samples_t[0])
    assert np.isclose(
        neutral_prebreakdown_result.t_prebreakdown_trigger,
        _np_expected_prebreakdown,
        **_TOL_ROUNDOFF,
    )
    # One frame per accepted step: the neutral fill, the first driven step
    # (pre-breakdown), one breakdown step, then the discharge and afterglow
    # at tau / dt steps each, and the closing frame.
    assert list(neutral_prebreakdown_result.phase) == (
        ["neutral_prebreakdown"] * round(_np_origin / _np_dt)
        + ["pre_breakdown", "breakdown"]
        + ["main_discharge"]
        * round(neutral_prebreakdown_params["tau_discharge"] / _np_dt)
        + ["afterglow"]
        * round(neutral_prebreakdown_params["tau_afterglow"] / _np_dt)
        + ["post_afterglow"]
    ), list(neutral_prebreakdown_result.phase)
    assert list(neutral_prebreakdown_result.phase[:2]) == [
        "neutral_prebreakdown",
        "neutral_prebreakdown",
    ]
    assert np.allclose(
        neutral_prebreakdown_result.phase_cathode_enabled[:2], 0.0, **_TOL_ROUNDOFF,
    )
    assert np.allclose(
        neutral_prebreakdown_result.phase_gas_puff_enabled[:2], 1.0, **_TOL_ROUNDOFF,
    )
    assert np.allclose(
        neutral_prebreakdown_result.n[1],
        neutral_prebreakdown_result.n[0],
        **_TOL_ROUNDOFF,
    )
    dynamic_current_phase_sim = LAPDSim1D(current_phase_params, current_phase_flags)
    dynamic_progress_snapshots = []
    dynamic_current_phase_initial_t_end = dynamic_current_phase_sim.default_t_end()
    dynamic_current_phase_result = dynamic_current_phase_sim.run(
        dt=1.0e-10,
        progress_tracker=dynamic_progress_snapshots.append,
        progress_interval_s=1.0,
    )
    assert np.isclose(dynamic_current_phase_result.final_time, 5.0e-10, **_TOL_ROUNDOFF)
    assert np.isclose(
        dynamic_current_phase_result.t_breakdown_trigger, 2.0e-10, **_TOL_ROUNDOFF,
    )
    assert len(dynamic_progress_snapshots) == 3
    assert np.isclose(
        dynamic_progress_snapshots[0].t_end,
        dynamic_current_phase_initial_t_end,
        **_TOL_ROUNDOFF,
    )
    assert np.isclose(dynamic_progress_snapshots[1].time, 2.0e-10, **_TOL_ROUNDOFF)
    assert np.isclose(dynamic_progress_snapshots[1].t_end, 5.0e-10, **_TOL_ROUNDOFF)
    assert np.isclose(dynamic_progress_snapshots[-1].fraction, 1.0, **_TOL_ROUNDOFF)
    assert np.allclose(
        dynamic_current_phase_result.phase_events["time"],
        [0.0, 1.0e-10, 2.0e-10, 4.0e-10, 5.0e-10],
        **_TOL_ROUNDOFF,
    )
    assert np.isclose(
        current_phase_sim._t_prebreakdown_trigger, 1.0e-10, **_TOL_ROUNDOFF,
    )
    assert np.isclose(current_phase_sim._t_breakdown_trigger, 2.0e-10, **_TOL_ROUNDOFF)
    assert np.isclose(
        current_phase_result.t_prebreakdown_trigger, 1.0e-10, **_TOL_ROUNDOFF,
    )
    assert np.isclose(
        current_phase_result.t_breakdown_trigger, 2.0e-10, **_TOL_ROUNDOFF,
    )
    assert np.allclose(
        current_phase_result.time,
        [0.0, 1.0e-10, 2.0e-10, 3.0e-10, 4.0e-10, 5.0e-10],
        **_TOL_ROUNDOFF,
    )
    # t_breakdown / t_breakdown_ms / time_since_breakdown /
    # time_ms_since_breakdown were retired _sim3 aliases: a copy of
    # t_breakdown_trigger, its millisecond form, and the absolute time shifted
    # by it. The trigger and the absolute time are asserted directly above, so
    # what is gone is the unit conversion and the subtraction, not a fact
    # about the run.
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
        **_TOL_ROUNDOFF,
    )
    assert np.allclose(
        current_phase_result.phase_cathode_enabled,
        [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        **_TOL_ROUNDOFF,
    )
    assert np.allclose(
        current_phase_result.phase_floating,
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        **_TOL_ROUNDOFF,
    )
    # Sample 0 is the PRE-BREAKDOWN solve, whose net current is zero by
    # construction, so what is stored there is the residual of the root the
    # solve returned rather than a current: it is bounded by the solve's own
    # current scale. Samples 1-3 are driven and carry a real forward current.
    _bd_I_tot = np.asarray(
        current_phase_result.cathode_diagnostics["source_I_tot"], dtype=float
    )
    _bd_I_e = np.asarray(
        current_phase_result.cathode_diagnostics["source_I_e"], dtype=float
    )
    assert _bd_I_tot[0] >= -1.0e-12 * abs(_bd_I_e[0]), _bd_I_tot[0]
    assert np.all(_bd_I_tot[1:4] > 0.0), _bd_I_tot[1:4]
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
        **_TOL_ROUNDOFF,
    )
    assert np.allclose(
        current_phase_result.current_trigger_samples["time"],
        [1.0e-10, 2.0e-10],
        **_TOL_ROUNDOFF,
    )
    assert np.allclose(
        current_phase_result.current_trigger_samples["I_tot"],
        current_phase_result.cathode_diagnostics["source_I_tot"][1:3],
        **_TOL_ROUNDOFF,
    )
    assert current_phase_summary.current_trigger_sample_count == 2
    assert np.isclose(
        current_phase_summary.last_current_trigger_sample["time"],
        current_phase_result.current_trigger_samples["time"][-1],
        **_TOL_ROUNDOFF,
    )
    assert np.isclose(
        current_phase_summary.last_current_trigger_sample["I_tot"],
        current_phase_result.current_trigger_samples["I_tot"][-1],
        **_TOL_ROUNDOFF,
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
        **_TOL_ROUNDOFF,
    )
    assert np.isclose(
        interpolated_current_phase_result.t_breakdown_trigger,
        expected_breakdown_time,
        **_TOL_ROUNDOFF,
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
        **_TOL_ROUNDOFF,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        current_phase_output = current_phase_sim.save_result(
            f"{tmpdir}/sim1d_current_phase_smoke.h5",
            current_phase_result,
        )
        with h5py.File(current_phase_output, "r") as h5:
            assert np.isclose(
                h5.attrs["t_prebreakdown_trigger"], 1.0e-10, **_TOL_ROUNDOFF,
            )
            assert np.isclose(h5.attrs["t_breakdown_trigger"], 2.0e-10, **_TOL_ROUNDOFF)
            assert h5["phase_events/time"].shape == (5,)
            assert h5["current_trigger_samples/time"].shape == (2,)
            assert h5["current_trigger_samples/I_tot"].shape == (2,)
        loaded_current_phase = load_result_hdf5(current_phase_output)
        assert np.isclose(
            loaded_current_phase.t_prebreakdown_trigger, 1.0e-10, **_TOL_ROUNDOFF,
        )
        assert np.isclose(
            loaded_current_phase.t_breakdown_trigger, 2.0e-10, **_TOL_ROUNDOFF,
        )
        # Same four retired aliases as on the run path; the trigger they were
        # derived from is asserted on the line above, on both the raw HDF5
        # attribute and the loaded result.
        assert np.allclose(
            loaded_current_phase.time,
            current_phase_result.time,
            **_TOL_ROUNDOFF,
        )
        assert np.allclose(
            loaded_current_phase.phase_events["time"],
            current_phase_result.phase_events["time"],
            **_TOL_ROUNDOFF,
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
            **_TOL_ROUNDOFF,
        )
        assert np.allclose(
            loaded_current_phase.current_trigger_samples["I_tot"],
            current_phase_result.current_trigger_samples["I_tot"],
            **_TOL_ROUNDOFF,
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
    assert np.isclose(
        direct_current_phase_sim._t_breakdown_trigger, 1.0e-10, **_TOL_ROUNDOFF,
    )
    assert np.isnan(direct_current_phase_result.t_prebreakdown_trigger)
    assert np.isclose(
        direct_current_phase_result.t_breakdown_trigger, 1.0e-10, **_TOL_ROUNDOFF,
    )
    assert np.allclose(
        direct_current_phase_result.phase_events["time"],
        [0.0, 1.0e-10, 3.0e-10, 4.0e-10],
        **_TOL_ROUNDOFF,
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
    return locals()


# --------------------------------------------------------------------
# current-phase-raise-on-timeout
# --------------------------------------------------------------------
@_case(
    "current-phase-raise-on-timeout",
    historical_stance=True,
)
def _case_current_phase_raise_on_timeout(
    current_phase_flags, current_phase_params
):
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
        assert np.isclose(
            exc.time, failed_current_phase_params["tau_prebreakdown"], **_TOL_ROUNDOFF,
        )
        assert np.isfinite(exc.I_tot)
        assert exc.I_tot < failed_current_phase_params["I_prebreakdown"]
        assert np.isclose(
            exc.threshold, failed_current_phase_params["I_prebreakdown"], **_TOL_ROUNDOFF,
        )
        assert exc.threshold_name == "I_prebreakdown"
        assert np.isclose(
            exc.tau_prebreakdown,
            failed_current_phase_params["tau_prebreakdown"],
            **_TOL_ROUNDOFF,
        )
        assert exc.details == {
            "phase": exc.phase,
            "time": exc.time,
            "I_tot": exc.I_tot,
            "threshold": exc.threshold,
            "threshold_name": exc.threshold_name,
            "tau_prebreakdown": exc.tau_prebreakdown,
        }
        assert np.allclose(exc.phase_events["time"], [0.0], **_TOL_ROUNDOFF)
        assert list(exc.phase_events["phase"]) == ["pre_breakdown"]
        assert list(exc.phase_events["reason"]) == ["initial"]
        assert np.allclose(
            exc.current_trigger_samples["time"],
            [1.0e-10, 2.0e-10, 3.0e-10, 4.0e-10, 5.0e-10],
            **_TOL_ROUNDOFF,
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
        assert np.isclose(
            exc.time, failed_breakdown_phase_params["tau_prebreakdown"], **_TOL_ROUNDOFF,
        )
        assert exc.I_tot > 0.0
        assert np.isclose(
            exc.threshold, failed_breakdown_phase_params["I_breakdown"], **_TOL_ROUNDOFF,
        )
        assert exc.threshold_name == "I_breakdown"
        assert np.allclose(exc.phase_events["time"], [0.0, 1.0e-10], **_TOL_ROUNDOFF)
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
            **_TOL_ROUNDOFF,
        )
        assert exc.current_trigger_samples["I_tot"].shape == (5,)
        assert np.all(np.isfinite(exc.current_trigger_samples["I_tot"]))
        assert exc.current_trigger_samples["I_tot"][-1] == exc.I_tot
    else:
        raise AssertionError("expected current-triggered breakdown phase to fail")


# --------------------------------------------------------------------
# ignition-failure-diagnostics
# --------------------------------------------------------------------
@_case(
    "ignition-failure-diagnostics",
    historical_stance=True,
    provides=("timeout_result",),
)
def _case_ignition_failure_diagnostics(
    current_phase_flags, current_phase_params, current_phase_result,
    direct_current_phase_result
):
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
        rtol=1e-12,
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
    assert np.isclose(timeout_sim.default_t_end(), 6.0e-10, **_TOL_ROUNDOFF)
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
    assert np.isclose(timeout_result.final_time, 4.0e-10, **_TOL_ROUNDOFF)
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
        timeout_result.phase_events["time"], [0.0, 3.0e-10, 4.0e-10],
        **_TOL_ROUNDOFF,
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
    assert np.isclose(timeout_result.ignition_abort["time_s"], 3.0e-10, **_TOL_ROUNDOFF)
    assert np.isclose(
        timeout_result.ignition_abort["window_s"], IGNITION_STALL_WINDOW_S,
        **_TOL_ROUNDOFF,
    )
    assert timeout_result.ignition_abort["threshold_name"] == "I_prebreakdown"
    assert np.isclose(
        timeout_result.ignition_abort["threshold_A"], 1.0e30, **_TOL_ROUNDOFF,
    )
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
        assert np.isclose(
            loaded_timeout.ignition_abort["time_s"], 3.0e-10, **_TOL_ROUNDOFF,
        )
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
    return locals()


# --------------------------------------------------------------------
# non-ignition-guards
# --------------------------------------------------------------------
@_case(
    "non-ignition-guards",
    historical_stance=True,
    provides=(
        "equilibration_flags", "neutral_phase_run_flags",
        "neutral_phase_run_params",
    ),
)
def _case_non_ignition_guards(
    current_phase_flags, current_phase_params, direct_current_phase_result,
    no_source_params, timeout_result
):
    # --- non-ignition guards, wall-clock / accepted-step arm -----------------
    # The stall detector and the tau_prebreakdown timeout both measure
    # SIMULATED time, so neither can see a non-igniting arm that stops
    # producing simulated time and burns wall clock instead. These two budgets
    # close over that, through the SAME switch-open path.
    params, flags = _base_config()
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
                budget_result.ignition_abort["time_s"], 3.0e-10,
                **_TOL_ROUNDOFF,
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
        assert np.isclose(
            ignited_origin, 1.0e-10, **_TOL_ROUNDOFF,
        ), (caller, ignited_origin)

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
        **_TOL_ROUNDOFF,
    )
    assert np.allclose(
        neutral_phase_result.phase_events["time"],
        [0.0, 2.0e-10],
        **_TOL_ROUNDOFF,
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
    assert np.isclose(neutral_cycles_sim.default_t_end(), 1.0e-9, **_TOL_ROUNDOFF)
    neutral_cycles_sim.start_simulation(dt=1.0e-9)
    neutral_cycles_result = neutral_cycles_sim.get_results()
    assert neutral_cycles_result.steps == 4
    assert np.isclose(neutral_cycles_result.final_time, 1.0e-9, **_TOL_ROUNDOFF)
    assert np.allclose(
        neutral_cycles_result.time,
        [0.0, 2.0e-10, 5.0e-10, 7.0e-10, 1.0e-9],
        **_TOL_ROUNDOFF,
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
    equilibration_params["initial_neutral_state"] = "equilibrate_only"
    equilibration_flags = dict(flags)
    equilibration_sim = LAPDSim1D(equilibration_params, equilibration_flags)
    equilibration_sim.start_simulation(dt=1.0e-10)
    equilibration_result = equilibration_sim.get_results()
    equilibration_summary = equilibration_sim.get_neutral_equilibration_summary()
    assert equilibration_result is equilibration_sim.get_neutral_equilibration_results()
    assert equilibration_result.neutral_equilibration_summary is equilibration_summary
    assert equilibration_summary.cycles == 2
    assert np.isclose(equilibration_summary.final_time, 1.0e-9, **_TOL_ROUNDOFF)
    assert np.isclose(
        equilibration_summary.mean_nn, np.mean(equilibration_result.nn[-1]), **_TOL_ROUNDOFF,
    )
    assert np.isclose(
        equilibration_summary.std_nn, np.std(equilibration_result.nn[-1]), **_TOL_ROUNDOFF,
    )
    assert not hasattr(equilibration_result, "neutral_equilibration")

    launch_flags = dict(equilibration_flags)
    launch_params = dict(equilibration_params, initial_neutral_state="equilibrate")
    launch_sim = LAPDSim1D(launch_params, launch_flags)
    launch_sim.start_simulation(t_end=2.0e-10, dt=1.0e-10)
    launch_result = launch_sim.get_results()
    assert np.isclose(launch_result.final_time, 2.0e-10, **_TOL_ROUNDOFF)
    assert hasattr(launch_result, "neutral_equilibration")
    assert launch_result.neutral_equilibration_summary.cycles == 2
    assert np.allclose(
        launch_result.nn[0],
        launch_result.neutral_equilibration.nn[-1],
        **_TOL_ROUNDOFF,
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
        **_TOL_ROUNDOFF,
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
        **_TOL_ROUNDOFF,
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
    return locals()


# --------------------------------------------------------------------
# prescribed-drive-refusals
# --------------------------------------------------------------------
@_case("prescribed-drive-refusals")
def _case_prescribed_drive_refusals():
    # cathode_solver_model='prescribed_measured' REPLACES the predicted drive
    # with a measured one, and everything that could make that silently wrong
    # is refused at construction: a trace that cannot be read, a clock that
    # cannot be reconciled, a hand-off that cannot be honoured, a calibrated
    # key that nothing will read, and the three drive keys set under any other
    # solver model. Thirteen cases, all construction-only.
    #
    # The valid trace is the COMMITTED ES1 overlay, resolved from this file's
    # own tree -- the same artifact the scorer and the circuit fits read, so a
    # schema drift in it fails here. The malformed traces are written into a
    # temporary directory; none of them may land under scripts/.
    from cablp.solvers._sim1d.core.prescribed_drive import (
        PRESCRIBED_DRIVE_KEYS as _pr_DRIVE_KEYS,
        PRESCRIBED_MEASURED as _pr_MEASURED,
        TRACE_REQUIRED_KEYS as _pr_TRACE_KEYS,
    )
    from cablp.solvers._sim1d.physics.cathode import (
        CATHODE_SOLVER_MODELS as _pr_MODELS,
    )

    _pr_trace = str(
        Path(__file__).resolve().parents[2] / "data" / "es1_sim1d_overlay.npz"
    )
    assert Path(_pr_trace).is_file(), _pr_trace
    # The registered sets are what the refusal messages promise, so a model or
    # a required trace column added without a case here fails rather than
    # shipping untested.
    assert set(_pr_MODELS) == {"current_driven", _pr_MEASURED}, _pr_MODELS
    assert _pr_DRIVE_KEYS == (
        "cathode_prescribed_trace_path",
        "cathode_prescribed_t0_s",
        "cathode_prescribed_start_s",
    )
    assert _pr_TRACE_KEYS == (
        "discharge_time_ms",
        "discharge_current_mean_a",
        "discharge_voltage_positive_mean_v",
    )

    def _pr_config(_pr_flag_over=None, **over):
        _p, _f = default_config()
        _p.update({"nx": 12})
        _f["cathode_coupling"] = True
        _p["initial_neutral_state"] = "fill"
        _p.update(over)
        if _pr_flag_over:
            _f.update(_pr_flag_over)
        return _p, _f

    _pr_on = {"cathode_solver_model": _pr_MEASURED}

    def _pr_refuses(label, params, flags=None, needle=""):
        try:
            LAPDSim1D(*_pr_config(_pr_flag_over=flags, **params))
        except ValueError as error:
            assert needle in str(error), (label, str(error))
        else:
            raise AssertionError(
                f"prescribed_measured accepted {label}: {params!r}"
            )

    with tempfile.TemporaryDirectory() as _pr_dir:
        # Three malformed traces, each a different way an .npz can fail to be
        # a drive: the wrong schema, too few samples to interpolate between,
        # and a time base with no unique value at a given time.
        _pr_wrong = Path(_pr_dir) / "wrong_columns.npz"
        np.savez(
            _pr_wrong,
            time_s=np.linspace(0.0, 1.0, 8),
            current=np.linspace(0.0, 1.0, 8),
            voltage=np.linspace(0.0, 1.0, 8),
        )
        _pr_short = Path(_pr_dir) / "one_sample.npz"
        np.savez(
            _pr_short,
            discharge_time_ms=np.array([0.0]),
            discharge_current_mean_a=np.array([1.0]),
            discharge_voltage_positive_mean_v=np.array([1.0]),
        )
        _pr_unsorted = Path(_pr_dir) / "unsorted.npz"
        np.savez(
            _pr_unsorted,
            discharge_time_ms=np.array([0.0, 2.0, 1.0]),
            discharge_current_mean_a=np.array([1.0, 2.0, 3.0]),
            discharge_voltage_positive_mean_v=np.array([1.0, 2.0, 3.0]),
        )

        for _pr_label, _pr_params, _pr_flags, _pr_needle in (
            # (a) THE MODE IS THE TRACE, AND THE TWO CLOCKS. Each of the three
            # drive keys is required, and none has a defensible default: a
            # guessed origin would slide the whole measured drive against the
            # column.
            ("no trace path",
             dict(_pr_on, cathode_prescribed_t0_s=2.0e-3,
                  cathode_prescribed_start_s=2.0e-3),
             None, "cathode_prescribed_trace_path is REQUIRED and unset"),
            ("no t0",
             dict(_pr_on, cathode_prescribed_trace_path=_pr_trace,
                  cathode_prescribed_start_s=2.0e-3),
             None, "cathode_prescribed_t0_s is REQUIRED and unset"),
            ("no start",
             dict(_pr_on, cathode_prescribed_trace_path=_pr_trace,
                  cathode_prescribed_t0_s=2.0e-3),
             None, "cathode_prescribed_start_s is REQUIRED and unset"),
            # (b) A HAND-OFF THAT CANNOT BE HONOURED: below the origin it
            # would drive the column from the trace's quiescent pre-trigger
            # branch; past the trace's end it would never take over at all.
            ("start before t0",
             dict(_pr_on, cathode_prescribed_trace_path=_pr_trace,
                  cathode_prescribed_t0_s=2.0e-3,
                  cathode_prescribed_start_s=1.0e-3),
             None, "is before cathode_prescribed_t0_s"),
            ("start past the end of the trace",
             dict(_pr_on, cathode_prescribed_trace_path=_pr_trace,
                  cathode_prescribed_t0_s=2.0e-3,
                  cathode_prescribed_start_s=1.0),
             None, "is past the end of the trace on the model clock"),
            # (c) A FILE THAT IS NOT A TRACE.
            ("a trace with the wrong columns",
             dict(_pr_on, cathode_prescribed_trace_path=str(_pr_wrong),
                  cathode_prescribed_t0_s=2.0e-3,
                  cathode_prescribed_start_s=2.0e-3),
             None, "does not carry"),
            ("a trace with one sample",
             dict(_pr_on, cathode_prescribed_trace_path=str(_pr_short),
                  cathode_prescribed_t0_s=2.0e-3,
                  cathode_prescribed_start_s=2.0e-3),
             None, "interpolating a drive needs at least two"),
            ("a trace whose time base is not increasing",
             dict(_pr_on, cathode_prescribed_trace_path=str(_pr_unsorted),
                  cathode_prescribed_t0_s=2.0e-3,
                  cathode_prescribed_start_s=2.0e-3),
             None, "must be strictly increasing"),
            ("a trace file that does not exist",
             dict(_pr_on,
                  cathode_prescribed_trace_path="/nonexistent/es9_overlay.npz",
                  cathode_prescribed_t0_s=2.0e-3,
                  cathode_prescribed_start_s=2.0e-3),
             None, "does not exist"),
            # (d) KEYS NOTHING WILL READ. With the hand-off at or before the
            # model clock's origin there is no calibrated foot, so the
            # emission constant, the surface temperature, the bank loop and
            # the heater package are read by nothing for the whole run.
            ("a calibrated key with the foot disabled",
             dict(_pr_on, cathode_prescribed_trace_path=_pr_trace,
                  cathode_prescribed_t0_s=-1.0e-3,
                  cathode_prescribed_start_s=0.0, C_R=9.3),
             None, "are set away from their defaults"),
            # (f) THE PRESENCE GATE, the other way: a measured drive nothing
            # reads is exactly the silent inert control the config surface
            # forbids.
            ("a prescribed key while the mode is off",
             dict(cathode_prescribed_trace_path=_pr_trace),
             None, "are read only under cathode_solver_model"),
            # (g) A model that does not exist. Refused by name, never by
            # falling back to the calibrated solve.
            ("an unknown cathode_solver_model",
             dict(cathode_solver_model="prescribed"),
             None, "cathode_solver_model must be"),
        ):
            _pr_refuses(
                _pr_label, _pr_params, flags=_pr_flags, needle=_pr_needle
            )

        # Each of the three drive keys is refused on its own with the mode
        # off, not merely as a set: a run that sets one of them under the
        # calibrated cathode is as silently inert as one that sets all three.
        for _pr_off_key, _pr_off_value in (
            ("cathode_prescribed_trace_path", _pr_trace),
            ("cathode_prescribed_t0_s", 2.0e-3),
            ("cathode_prescribed_start_s", 2.0e-3),
        ):
            _pr_refuses(
                f"{_pr_off_key} while the mode is off",
                {_pr_off_key: _pr_off_value},
                needle="are read only under cathode_solver_model",
            )

    # THE POSITIVE CONTROL: the committed overlay, a hand-off inside its span
    # and after a calibrated foot, resolves -- and it resolves onto the MODEL
    # clock, carrying the digest of the file's own bytes so a saved artifact
    # names the measured product rather than a path.
    _pr_sim = LAPDSim1D(*_pr_config(
        cathode_solver_model=_pr_MEASURED,
        cathode_prescribed_trace_path=_pr_trace,
        cathode_prescribed_t0_s=2.0e-3,
        cathode_prescribed_start_s=2.0e-3,
    ))
    _pr_resolved = _pr_sim._prescribed_drive
    assert _pr_resolved is not None
    assert _pr_resolved.t0_s == 2.0e-3 and _pr_resolved.start_s == 2.0e-3
    assert len(_pr_resolved.sha256) == 64
    with np.load(_pr_trace, allow_pickle=False) as _pr_raw:
        assert np.array_equal(
            _pr_resolved.time_s,
            2.0e-3 + 1.0e-3 * np.asarray(
                _pr_raw["discharge_time_ms"], dtype=float
            ),
        )
        assert np.array_equal(
            _pr_resolved.current_A,
            np.asarray(_pr_raw["discharge_current_mean_a"], dtype=float),
        )
        assert np.array_equal(
            _pr_resolved.V_dis_V,
            np.asarray(
                _pr_raw["discharge_voltage_positive_mean_v"], dtype=float
            ),
        )
    assert not _pr_resolved.active(1.9e-3)
    assert _pr_resolved.active(2.0e-3)
    # The calibrated cathode is what the OFF path keeps: no trace resolved,
    # and the solver carries no hand-off to record.
    _pr_off_sim = LAPDSim1D(*_pr_config())
    assert _pr_off_sim._prescribed_drive is None
    assert _pr_off_sim._prescribed_handoff is None


# --------------------------------------------------------------------
# prescribed-drive-handoff
# --------------------------------------------------------------------
@_case("prescribed-drive-handoff")
def _case_prescribed_drive_handoff():
    # THE HAND-OFF ITSELF, over a short march that reaches the main discharge.
    # Three statements:
    #
    #   * past the hand-off the loop current IS the trace, interpolated onto
    #     the model clock -- bitwise, at every saved frame, not merely close;
    #   * the switch is RECORDED on the file: which trace drove the run (path
    #     and the digest of its bytes), when the switch happened, the two
    #     currents on either side of it, and their relative difference, which
    #     is the measurement of how far the calibrated cathode sat from this
    #     rung's measured drive. Nothing is smoothed across it;
    #   * the presence gate: with the mode off, the three drive keys sitting
    #     at their None defaults produce a run BIT-IDENTICAL to the same
    #     configuration that never names them, and a file that carries neither
    #     the drive record nor the hand-off record.
    #
    # The trace is a small synthetic overlay written into a temporary
    # directory: a coarse, strictly-increasing ramp, so the interpolation
    # between its samples is non-trivial and the identity below cannot pass on
    # a constant. It must never be written under scripts/.
    #
    # THE HAND-OFF JUMP LINE THIS CASE PRINTS IS EXPECTED, and is not a
    # failure: an invented current on a 12-cell fixture has nothing to do with
    # what an uncalibrated tiny cathode integrates to, so the switch is a large
    # one and the loud line fires. That it fires is part of what is exercised.
    from cablp.solvers._sim1d.core.prescribed_drive import (
        HANDOFF_JUMP_WARN_FRACTION as _ho_JUMP_WARN,
        PRESCRIBED_MEASURED as _ho_MEASURED,
    )
    from cablp.solvers._sim1d.results.io import save_result_hdf5 as _ho_save

    #: The phase schedule: long enough to break down and reach the main
    #: discharge (an aborted ignition is not the fixture this case wants),
    #: short enough to stay a smoke-scale run.
    _ho_tau_pre, _ho_tau_bd = 8.0e-5, 4.0e-5
    #: The hand-off, at the main discharge's own start; the trace's t = 0 is
    #: named to the same model time, so the trace is read from its trigger.
    _ho_t0 = _ho_tau_pre + _ho_tau_bd
    _ho_t_ms = np.array([-1.0, 0.0, 0.005, 0.01, 0.02, 0.05, 2.0])
    _ho_I_A = np.array([0.0, 50.0, 300.0, 700.0, 900.0, 950.0, 400.0])
    _ho_V_V = np.array([0.0, 40.0, 90.0, 120.0, 130.0, 128.0, 60.0])

    def _ho_config(trace_path=None, explicit_none=False):
        _p, _f = default_config()
        _p.update({
            "nx": 12,
            "dt_save": 0.0,
            "max_steps_action": "stop",
            "tau_prebreakdown": _ho_tau_pre,
            "tau_breakdown": _ho_tau_bd,
            "tau_discharge": 5.0e-4,
            "tau_afterglow": 1.0e-4,
        })
        _f["cathode_coupling"] = True
        _p["initial_neutral_state"] = "fill"
        if trace_path is not None:
            _p.update({
                "cathode_solver_model": _ho_MEASURED,
                "cathode_prescribed_trace_path": str(trace_path),
                "cathode_prescribed_t0_s": _ho_t0,
                "cathode_prescribed_start_s": _ho_t0,
            })
        if explicit_none:
            # The keys NAMED and left at their template defaults: the presence
            # gate is that naming them changes nothing at all.
            _p.update({
                "cathode_solver_model": "current_driven",
                "cathode_prescribed_trace_path": None,
                "cathode_prescribed_t0_s": None,
                "cathode_prescribed_start_s": None,
            })
        return _p, _f

    with tempfile.TemporaryDirectory() as _ho_dir:
        _ho_trace = Path(_ho_dir) / "synthetic_overlay.npz"
        np.savez(
            _ho_trace,
            discharge_time_ms=_ho_t_ms,
            discharge_current_mean_a=_ho_I_A,
            discharge_voltage_positive_mean_v=_ho_V_V,
        )
        _ho_sim = LAPDSim1D(*_ho_config(trace_path=_ho_trace))
        _ho_result = _ho_sim.run(t_end=_ho_t0 + 2.0e-5, dt=None)
        _ho_time = np.asarray(_ho_result.time, dtype=float)
        _ho_after = _ho_time >= _ho_t0
        # The march must actually cross the hand-off, and must have ignited:
        # a run that never reached the main discharge would be asserting on
        # the wind-down instead.
        assert _ho_after.sum() >= 8, int(_ho_after.sum())
        assert "main_discharge" in set(
            np.asarray(_ho_result.phase).tolist()
        ), sorted(set(np.asarray(_ho_result.phase).tolist()))

        # (i) I(t) IS THE TRACE past the hand-off. The loop current is a
        # measurement in this mode -- nothing is integrated -- so the saved
        # row equals the trace interpolated onto the model clock exactly.
        _ho_loop_A = np.asarray(
            _ho_result.cathode_diagnostics["circuit_I_loop"], dtype=float
        )
        _ho_want_A = np.interp(
            _ho_time, _ho_t0 + 1.0e-3 * _ho_t_ms, _ho_I_A
        )
        assert np.array_equal(_ho_loop_A[_ho_after], _ho_want_A[_ho_after])
        # ... and the trace is being READ, not held: the ramp gives a
        # different value at every frame, so a stuck sample-and-hold would
        # fail here rather than pass the identity above trivially.
        assert (
            np.unique(_ho_loop_A[_ho_after]).size == int(_ho_after.sum())
        )

        # (ii) THE SWITCH IS RECORDED, and the record reaches the FILE. The
        # relative jump is normalized on the MEASURED current, which is the
        # reference the calibrated foot is judged against.
        _ho_handoff = _ho_result.prescribed_handoff
        assert _ho_handoff["time_s"] >= _ho_t0
        _ho_jump = (
            _ho_handoff["current_trace_A"]
            - _ho_handoff["current_calibrated_A"]
        ) / max(abs(_ho_handoff["current_trace_A"]), 1.0e-12)
        assert _ho_handoff["relative_jump"] == _ho_jump
        assert _ho_handoff["current_trace_A"] == float(
            np.interp(
                _ho_handoff["time_s"],
                _ho_t0 + 1.0e-3 * _ho_t_ms,
                _ho_I_A,
            )
        )
        assert 0.0 < _ho_JUMP_WARN < 1.0

        _ho_path = Path(_ho_dir) / "prescribed_handoff.h5"
        _ho_save(_ho_path, _ho_result)
        with h5py.File(_ho_path, "r") as _ho_h5:
            _ho_attrs = dict(_ho_h5.attrs)
        assert _ho_attrs["cathode_prescribed_trace_path"] == str(_ho_trace)
        assert (
            _ho_attrs["cathode_prescribed_trace_sha256"]
            == _ho_sim._prescribed_drive.sha256
        )
        assert _ho_attrs["cathode_prescribed_t0_s"] == _ho_t0
        assert _ho_attrs["cathode_prescribed_start_s"] == _ho_t0
        assert (
            _ho_attrs["cathode_prescribed_handoff_time_s"]
            == _ho_handoff["time_s"]
        )
        assert (
            _ho_attrs["cathode_prescribed_handoff_current_calibrated_a"]
            == _ho_handoff["current_calibrated_A"]
        )
        assert (
            _ho_attrs["cathode_prescribed_handoff_current_trace_a"]
            == _ho_handoff["current_trace_A"]
        )
        assert (
            _ho_attrs["cathode_prescribed_handoff_relative_jump"]
            == _ho_handoff["relative_jump"]
        )

        # (iii) THE PRESENCE GATE. Two calibrated runs of the same length:
        # one whose config never names the three drive keys, one that names
        # all three at their None defaults. Bit-identical, field by field, at
        # raw uint64 -- and the OFF file carries none of the eight attributes
        # above, so an unarmed run's artifact is exactly the one it always was.
        _ho_bare = LAPDSim1D(*_ho_config()).run(
            t_end=None, dt=None, max_steps=30
        )
        _ho_named = LAPDSim1D(*_ho_config(explicit_none=True)).run(
            t_end=None, dt=None, max_steps=30
        )
        assert _ho_bare.steps == _ho_named.steps == 30

        def _ho_identical(a, b):
            """True when two saved rows agree BIT for bit.

            Float rows are compared as raw uint64 rather than by value: the
            cathode diagnostics carry NaN rows (``x0_twin_next`` on a
            single-cathode run), and NaN != NaN would report an unmoved row as
            a divergence. The regime row is a string row and is compared as
            one.
            """
            _a = np.ascontiguousarray(np.asarray(a))
            _b = np.ascontiguousarray(np.asarray(b))
            if _a.shape != _b.shape or _a.dtype != _b.dtype:
                return False
            if _a.dtype != np.float64:
                return np.array_equal(_a, _b)
            return np.array_equal(_a.view(np.uint64), _b.view(np.uint64))

        for _ho_field in ("time",) + tuple(STATE_NAMES_1D) + ("u", "Te", "Ti"):
            assert _ho_identical(
                getattr(_ho_bare, _ho_field), getattr(_ho_named, _ho_field)
            ), _ho_field
        assert set(_ho_bare.cathode_diagnostics) == set(
            _ho_named.cathode_diagnostics
        )
        for _ho_row in _ho_bare.cathode_diagnostics:
            assert _ho_identical(
                _ho_bare.cathode_diagnostics[_ho_row],
                _ho_named.cathode_diagnostics[_ho_row],
            ), _ho_row
        assert not hasattr(_ho_bare, "prescribed_drive")
        assert not hasattr(_ho_named, "prescribed_drive")
        assert not hasattr(_ho_named, "prescribed_handoff")

        _ho_off_path = Path(_ho_dir) / "calibrated_control.h5"
        _ho_save(_ho_off_path, _ho_named)
        with h5py.File(_ho_off_path, "r") as _ho_off_h5:
            _ho_off_attrs = set(_ho_off_h5.attrs)
        assert not {
            _ho_name for _ho_name in _ho_attrs
            if _ho_name.startswith("cathode_prescribed_")
        } & _ho_off_attrs


# ----------------------------------------------------------------------
# afterglow-tail-handoff-criterion
# ----------------------------------------------------------------------
@_case("afterglow-tail-handoff-criterion", historical_stance=True)
def _case_afterglow_tail_handoff_criterion():
    """The freewheel tail ends when the loop would need the load to drive it.

    The inductive tail is integrated at zero source voltage while the loop
    still has current AND the device voltage it integrated is positive. Two
    end conditions return it to open circuit, and both are checked here
    against the accept path itself (one step, and the loop current it leaves):

      HAND-OFF BY VOLTAGE  V_dis_step <= 0 at I = 800 A. The bank is open, a
        freewheel loop has no source, and a diode blocks the reversal a
        negative device voltage would drive. ``floating`` flips and the loop
        current goes to exactly zero.
      HAND-OFF BY CURRENT  I_prev <= 1 A at a positive device voltage. Same
        outcome; this is the condition that already existed.
      NEGATIVE CONTROL  I_prev = 800 A at V_dis_step = +10 V: no hand-off.
        ``inductive_tail`` stands, ``floating`` stays False, and the accept
        path integrates the loop rather than dropping it -- the measured
        freewheel is physics and is not switched off.
    """
    _th_p, _th_f = _cathode_unit_config()
    _th_p.update({
        "V_bank": 173.6,
        "R_comp": 5.72e-3,
        "L_parasitic_H": 6.6e-6,
        "cathode_solver_model": "current_driven",
        "dt_save": 0.0,
    })
    _th_f = dict(
        _th_f,
        cathode_coupling=True,
    )
    # This case steps the solver directly, so the equilibration seed is not
    # being asked for; the scalar-fill route keeps run() from warning that it
    # did not run one.
    _th_p = dict(_th_p, initial_neutral_state="fill")

    def _th_build(V_dis_step, I_prev):
        sim = LAPDSim1D(dict(_th_p), dict(_th_f))
        # Put the clock in the afterglow: this stance runs the current-mode
        # phase schedule, so the afterglow opens tau_discharge after the
        # breakdown trigger.
        sim._t_breakdown_trigger = 1.0e-3
        sim._time = 1.0e-3 + float(_th_p["tau_discharge"]) + 1.0e-6
        sim._circuit_I_prev = float(I_prev)
        sim._circuit_I_loop = float(I_prev)
        sim._circuit_V_dis_step = float(V_dis_step)
        assert sim.phase_switches_at_time(sim._time)["floating"], sim._time
        return sim

    # NEGATIVE CONTROL: the driven freewheel tail.
    _th_sim = _th_build(10.0, 800.0)
    _th_phase = _th_sim._cathode_phase_options()
    assert _th_phase["inductive_tail"] is True, _th_phase
    assert _th_phase["floating"] is False, _th_phase
    _th_sim.run(t_end=_th_sim._time + 2.0e-9, max_steps=1)
    assert _th_sim._circuit_I_loop > 0.0, _th_sim._circuit_I_loop

    # HAND-OFF, by either condition: floating, and the loop current is zero.
    for _th_V, _th_I in ((-0.05, 800.0), (10.0, 0.5)):
        _th_sim = _th_build(_th_V, _th_I)
        _th_phase = _th_sim._cathode_phase_options()
        assert _th_phase["floating"] is True, (_th_V, _th_I, _th_phase)
        assert _th_phase["inductive_tail"] is False, (_th_V, _th_I, _th_phase)
        assert _th_phase["solve_enabled"] is True, (_th_V, _th_I, _th_phase)
        # The RHS gate keeps the cathode ACTIVE across the hand-off, so the
        # electrode rows continue at I = 0 instead of dropping to zero.
        assert _th_sim._effective_cathode_flags(
            active_only=True
        )["cathode_coupling"] is True
        _th_solve = _th_sim.solve_cathode_boundary(update_cache=False)
        assert _th_solve.metadata["floating"] is True
        _th_res = _th_solve.beam_result.result
        assert abs(_th_res.I_tot) <= 1.0e-12 * max(_th_res.I_eth_star, 1.0)
        assert abs(_th_res.I_cathode_kirchhoff_residual) <= 1.0e-12
        _th_sim.run(t_end=_th_sim._time + 2.0e-9, max_steps=1)
        assert _th_sim._circuit_I_loop == 0.0, _th_sim._circuit_I_loop


# ----------------------------------------------------------------------
# tail-handoff-surface-continuity
# ----------------------------------------------------------------------
@_case("tail-handoff-surface-continuity")
def _case_tail_handoff_surface_continuity():
    """Evaporative emission cooling is continuous across the hand-off.

    An open circuit is zero NET current, not zero emission. At ``I_tot = 0``
    the emitter still releases its space-charge-limited ``I_eth_star`` and
    those electrons leave; the balance closes because the plasma returns a
    LARGER collected current over the barrier, which is a different flux
    arriving at the surface, not the emitted one turned back. So the surface's
    ``P_emis`` at ``I_tot = 0`` must be the limit of its value from the driven
    side, and the surface ledger takes no step at the hand-off.

    Checked on the late-afterglow state of the reference machine, where the
    emission is largest relative to the loop current:

      CONTINUITY   ``P_emis`` at ``I_tot = 0`` equals its value at
        ``I_tot = 1e-12`` and ``1e-9`` A to better than 1e-6 relative, and the
        approach from 1e-3 A is monotone and small. The formula is the shipped
        ``cathode_power_balance_terms_W``, driven by the shipped
        ``solve_idriven``, so this is the solver's own arithmetic.
      NEGATIVE CONTROL  the retired ``0.0 if floating`` form is discontinuous
        by the whole term: ``I_eth_star`` (61.895109 A here) times
        ``phi_wf + 2 k_B T_s`` (3.198777 eV) = 197.988671 W, dropped in one
        step. That is the defect this case exists to keep out.

    NOT ASSERTED, and NOT MODELLED: the energy the COLLECTED plasma electrons
    deposit back on the surface in return. ``cathode_power_balance_terms_W``
    (``cablp/solvers/_sim1d/physics/cathode.py``) states plainly that this
    deposit is not modelled, and while it is off the books ``P_emis`` is the
    whole of the face's electron-channel budget. At this state the unmodelled
    deposit would be the larger of the face's two electron-energy quantities,
    so this case's continuity check covers only the smaller, modelled one.
    """
    from cablp.cathode.circuit_common import (
        DeviceConfig as _sc_DeviceConfig,
        PlasmaState as _sc_PlasmaState,
    )
    from cablp.cathode.circuit_idriven import solve_idriven as _sc_solve
    from cablp.solvers._sim1d.physics.cathode import (
        cathode_power_balance_terms_W as _sc_terms,
    )

    _sc_cfg_kw = dict(
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
    _sc_Ts = 1913.453373340071
    _sc_cfg = _sc_DeviceConfig(T_s=_sc_Ts, **_sc_cfg_kw)
    _sc_pl = _sc_PlasmaState(
        T_e=0.1, n_e=3.424966858999728e11, n_n=4.701649450942045e13
    )
    _sc_alpha = math.exp(-0.5)
    # The surface-side input_dict the warming update passes: only these four
    # keys are read by the term.
    _sc_idict = {
        "R_cath": 18.415,
        "cathode_emissivity": 0.7,
        "cathode_Ts_base_K": 1910.0,
        "phi_wf": 2.869,
        "cathode_conduction_W_per_K": 0.0,
    }

    def _sc_P_emis(I_A, force_zero=False):
        r = _sc_solve(
            _sc_cfg, _sc_pl, I_A, anode_T_e=0.1, alpha_sheath=_sc_alpha,
            alpha_sheath_anode=_sc_alpha, phi_c_cap_V=1000.0,
        )
        I_emis = 0.0 if force_zero else float(r.I_eth_star)
        return _sc_terms(
            T_s_K=_sc_Ts,
            P_ion_W=float(r.P_cathode_i),
            I_eth_star_A=I_emis,
            input_dict=_sc_idict,
        )[3], float(r.I_eth_star)

    _sc_P0, _sc_I0 = _sc_P_emis(0.0)
    # The term is live and of the expected size at zero net current.
    assert _sc_P0 > 100.0, _sc_P0
    assert _sc_I0 > 10.0, _sc_I0
    # CONTINUITY from the driven side.
    for _sc_I, _sc_rtol in ((1.0e-12, 1.0e-12), (1.0e-9, 1.0e-9),
                            (1.0e-6, 1.0e-6), (1.0e-3, 1.0e-4)):
        _sc_P, _ = _sc_P_emis(_sc_I)
        assert abs(_sc_P - _sc_P0) <= _sc_rtol * _sc_P0, (
            _sc_I, _sc_P, _sc_P0
        )
    # NEGATIVE CONTROL: the retired form drops the whole term at the hand-off.
    _sc_P_old, _ = _sc_P_emis(0.0, force_zero=True)
    assert _sc_P_old == 0.0, _sc_P_old
    # ...and the size of that step is I_eth_star * (phi_wf + 2 k_B T_s).
    _sc_per_electron = 2.869 + 2.0 * 8.617333262e-5 * _sc_Ts
    assert np.isclose(
        _sc_P0 - _sc_P_old, _sc_I0 * _sc_per_electron, rtol=1e-12, atol=0.0
    ), (_sc_P0, _sc_I0, _sc_per_electron)
    assert _sc_P0 - _sc_P_old > 100.0, _sc_P0 - _sc_P_old


# --------------------------------------------------------------------
# phase-boundary-one-ulp-above-step
# --------------------------------------------------------------------
@_case("phase-boundary-one-ulp-above-step", historical_stance=True)
def _case_phase_boundary_one_ulp_above_step():
    """A boundary that lands an ulp above a step time is reached at that step.

    Current-driven phases with no neutral fill, dt = 1e-10 s: the triggers
    latch at the first two accepted steps, so the discharge starts at
    t_bd = 2e-10 s, and tau_discharge = 5e-10 s puts the afterglow boundary at
    fl(2e-10 + 5e-10) = 7.000000000000001e-10 s, one ulp above the step time
    7e-10 s that five 1e-10 s steps reach from t_bd. The run loop and the phase
    lookup must agree that the boundary is reached there: the discharge then
    runs exactly tau_discharge / dt steps, the frame after it starts the
    afterglow, and that afterglow step runs on the open circuit.
    """
    params, flags = _base_config()
    dt = 1.0e-10
    params.update({
        "gas_puff_enabled": False,
        "pump_enabled": False,
        "dt_save": 0.0,
        "phase_transition_mode": "current",
        "tau_prebreakdown": 5.0e-10,
        "tau_discharge": 5.0e-10,
        "tau_afterglow": 1.0e-10,
        "I_prebreakdown": 1.0e-9,
        "I_breakdown": 1.0e-9,
    })
    flags["cathode_coupling"] = True
    result = LAPDSim1D(params, flags).run(dt=dt)
    t_bd = 2.0 * dt
    afterglow_start = t_bd + params["tau_discharge"]
    # The premise: the boundary sum is not the step time it should coincide
    # with, but lies within an ulp above it.
    t_steps = t_bd
    for _ in range(round(params["tau_discharge"] / dt)):
        t_steps += dt
    assert afterglow_start != t_steps, (afterglow_start, t_steps)
    assert 0.0 < afterglow_start - t_steps <= np.spacing(t_steps), (
        afterglow_start, t_steps,
    )
    assert np.isclose(result.t_breakdown_trigger, t_bd, **_TOL_ROUNDOFF)

    phase = list(result.phase)
    # One frame per accepted step at dt_save = 0, so the main-discharge frames
    # count the driven steps.
    driven = [i for i, p in enumerate(phase) if p == "main_discharge"]
    assert len(driven) == round(params["tau_discharge"] / dt), phase
    first_after = driven[-1] + 1
    assert phase[first_after] == "afterglow", phase
    assert np.isclose(result.time[first_after], t_steps, **_TOL_ROUNDOFF)
    assert result.phase_floating[first_after] == 1.0
    assert result.phase_cathode_enabled[first_after] == 0.0
    # The afterglow's circuit state. The loop current the discharge hands on
    # is at most 1 A, so the hand-off rule opens the circuit for the
    # afterglow step and the loop carries exactly zero at its end.
    I_loop = np.asarray(result.cathode_diagnostics["circuit_I_loop"], float)
    assert 0.0 < I_loop[first_after] <= 1.0, I_loop
    assert I_loop[first_after + 1] == 0.0, I_loop
    assert phase[first_after + 1:] == ["post_afterglow"], phase
    assert np.isclose(
        result.final_time,
        afterglow_start + params["tau_afterglow"],
        **_TOL_ROUNDOFF,
    )
