"""Quick coarse _sim3/_sim1d benchmark probes.

The checks here compare signs, scales, shapes, and invariants.  They are not
intended to assert pointwise equality between the legacy primitive solver and
the conservative 1D solver.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass

import numpy as np

from cablp.solvers._sim1d import LAPDSim1D
from cablp.solvers._sim1d import default_config as default_config_1d
from cablp.solvers._sim3 import LAPDSim
from cablp.solvers._sim3 import default_config as default_config_3


SIM3_MAX_OUTPUT_STEPS = 50


@dataclass
class CaseReport:
    name: str
    detail: str


def main():
    reports = [
        neutral_puff_cycle_case(),
        sim1d_no_cathode_relaxation_case(),
        cathode_short_discharge_case(twin=False),
        cathode_short_discharge_case(twin=True),
    ]
    for report in reports:
        print(f"PASS {report.name}: {report.detail}")
    print(f"sim1d/sim3 benchmark ok: {len(reports)} cases")


def neutral_puff_cycle_case():
    sim3_params, sim3_flags = _sim3_base_config()
    sim1d_params, sim1d_flags = _sim1d_base_config()
    neutral_updates = {
        "cycles": 1,
        "tau_discharge": 2.0e-10,
        "tau_cycle": 5.0e-10,
        "dt_save": 0.0,
    }
    sim3_params.update(
        {
            **neutral_updates,
            "h0": 1.0e-10,
            "h_max_discharge": 1.0e-10,
            "h_max_afterglow": 1.0e-10,
            "max_output_steps": SIM3_MAX_OUTPUT_STEPS,
        }
    )
    sim3_flags["Plasma"] = False
    sim1d_params.update({**neutral_updates, "dt_max": 1.0e-10})
    sim1d_flags["Plasma"] = False

    sim3_result = _run_sim3(sim3_params, sim3_flags)
    sim1d_result = _run_sim1d(sim1d_params, sim1d_flags, t_end=None, dt=1.0e-10)

    sim3_nn = _finite_rows(sim3_result.nn)
    _assert_finite("sim3 neutral nn", sim3_nn)
    _assert_finite("sim1d neutral nn", sim1d_result.nn)
    _assert_positive("sim3 neutral density", sim3_nn)
    _assert_positive("sim1d neutral density", sim1d_result.nn)
    assert sim3_result.time.size >= 2
    assert sim1d_result.time.size >= 2
    assert np.all(np.diff(sim3_result.time) > 0.0)
    assert np.all(np.diff(sim1d_result.time) > 0.0)
    assert list(sim1d_result.phase_events["phase"]) == [
        "equilibrium_puff",
        "equilibrium_off",
        "equilibrium_puff",
    ]
    assert list(sim1d_result.phase_events["reason"]) == [
        "initial",
        "tau_discharge",
        "tau_cycle",
    ]
    assert np.any(np.abs(_finite_rows(sim3_result.Nn_flux)) > 0.0)
    assert np.any(np.abs(_finite_rows(sim1d_result.Nn_flux)) > 0.0)

    return CaseReport(
        "neutral puff/off cycle",
        (
            f"sim3_steps={sim3_result.time.size}, "
            f"sim1d_steps={sim1d_result.steps}, "
            f"sim1d_final_s={sim1d_result.final_time:.3e}"
        ),
    )


def sim1d_no_cathode_relaxation_case():
    params, flags = _sim1d_base_config()
    params.update(
        {
            "gas_puff_enabled": False,
            "pump_enabled": False,
            "b_ioniz": 0.0,
            "b_rec_rad": 0.0,
            "b_rec_3b": 0.0,
            "b_Qie": 0.0,
            "b_Qei": 0.0,
            "b_Qen": 0.0,
            "b_Qcx": 0.0,
            "b_epara": 0.0,
            "b_ipara": 0.0,
            "b_ionization_energy_cost": 0.0,
            "b_pressure_work_elec": 0.0,
            "b_pressure_work_ions": 0.0,
            "b_surface_loss": 0.0,
            "dt_save": 0.0,
        }
    )
    flags.update(
        {
            "cathode_coupling": False,
            "front_flux": False,
            "heat_conduction": False,
        }
    )
    sim = LAPDSim1D(params, flags)
    initial = sim.get_initial_snapshot().y.copy()
    result = sim.run(t_end=3.0e-10, dt=1.0e-10)

    _assert_finite("sim1d no-cathode state", result.y)
    assert np.allclose(result.y[-1], initial, rtol=0.0, atol=1.0e-20)
    assert np.allclose(result.phase_cathode_enabled, 0.0)
    assert np.allclose(result.cathode_diagnostics["enabled"], 0.0)

    return CaseReport(
        "sim1d no-cathode relaxation invariant",
        "_sim3 plasma phases always include cathode solving; sim1d state stayed fixed",
    )


def cathode_short_discharge_case(twin):
    sim3_params, sim3_flags = _sim3_base_config()
    sim1d_params, sim1d_flags = _sim1d_base_config()
    phase_updates = {
        "tau_prebreakdown": 5.0e-10,
        "tau_discharge": 2.0e-10,
        "tau_afterglow": 1.0e-10,
        "dt_save": 0.0,
        "I_prebreakdown": 1.0e-9,
        "I_breakdown": 1.0e-9,
    }
    sim3_params.update(
        {
            **phase_updates,
            "h0": 1.0e-10,
            "h_max_discharge": 1.0e-10,
            "h_max_afterglow": 1.0e-10,
            "max_output_steps": SIM3_MAX_OUTPUT_STEPS,
        }
    )
    sim3_flags["TwinCathode"] = bool(twin)
    sim1d_params.update(
        {
            **phase_updates,
            "tau_breakdown": 0.0,
            "phase_transition_mode": "current",
            "dt_max": 1.0e-10,
        }
    )
    sim1d_flags.update({"cathode_coupling": True, "TwinCathode": bool(twin)})

    sim3_result = _run_sim3(sim3_params, sim3_flags)
    sim1d_result = _run_sim1d(sim1d_params, sim1d_flags, t_end=5.0e-10, dt=1.0e-10)

    _assert_finite("sim3 cathode current", sim3_result.cathode.I_tot)
    _assert_finite("sim1d cathode current", sim1d_result.cathode.I_tot)
    assert np.nanmax(sim3_result.cathode.I_tot) > 0.0
    assert np.nanmax(sim1d_result.cathode.I_tot) > 0.0
    assert np.nanmax(sim3_result.S_ion_beam) > 0.0
    assert np.nanmax(sim1d_result.S_ion_beam) > 0.0
    assert sim3_result.t_breakdown is not None
    assert np.isfinite(sim1d_result.t_breakdown_ms)
    assert np.nanmin(sim3_result.time) <= 0.0 <= np.nanmax(sim3_result.time)
    assert np.nanmin(sim1d_result.time_ms_since_breakdown) <= 0.0
    assert np.nanmax(sim1d_result.time_ms_since_breakdown) >= 0.0

    if twin:
        _assert_finite("sim3 twin cathode current", sim3_result.cathode_twin.I_tot)
        _assert_finite("sim1d twin cathode current", sim1d_result.cathode_twin.I_tot)
        assert np.nanmax(sim3_result.cathode_twin.I_tot) > 0.0
        assert np.nanmax(sim1d_result.cathode_twin.I_tot) > 0.0
        sim3_beam = _finite_rows(sim3_result.S_ion_beam)
        sim1d_beam = _finite_rows(sim1d_result.S_ion_beam)
        assert np.nanmean(sim3_beam[:, 0]) > 0.0
        assert np.nanmean(sim3_beam[:, -1]) > 0.0
        assert np.nanmean(sim1d_beam[:, 0]) > 0.0
        assert np.nanmean(sim1d_beam[:, -1]) > 0.0

    label = "twin cathode short discharge" if twin else "cathode short discharge"
    return CaseReport(
        label,
        (
            f"sim3_Imax={np.nanmax(sim3_result.cathode.I_tot):.3e} A, "
            f"sim1d_Imax={np.nanmax(sim1d_result.cathode.I_tot):.3e} A"
        ),
    )


def _sim3_base_config():
    params, flags = default_config_3()
    params.update(
        {
            "Lm": 1800.0,
            "Lp": 1800.0,
            "cells": 3,
            "max_step_rejections": 20,
        }
    )
    return params, flags


def _sim1d_base_config():
    params, flags = default_config_1d()
    params.update(
        {
            "Lm": 1800.0,
            "Lz": 1600.0,
            "source_length_cm": 100.0,
            "end_length_cm": 100.0,
            "nx": 3,
            "max_step_retries": 20,
            "dt_growth_enabled": False,
        }
    )
    return params, flags


def _run_sim3(params, flags):
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        sim = LAPDSim(params, flags)
        sim.start_simulation()
    return sim.get_results()


def _run_sim1d(params, flags, *, t_end, dt):
    sim = LAPDSim1D(params, flags)
    sim.start_simulation(t_end=t_end, dt=dt, max_steps=200)
    return sim.get_results()


def _finite_rows(values):
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        return arr[np.isfinite(arr)]
    keep = np.any(np.isfinite(arr), axis=0)
    return arr[:, keep]


def _assert_finite(label, values):
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    assert finite.size > 0, f"{label} has no finite values"
    assert np.all(np.isfinite(finite)), f"{label} has non-finite values"


def _assert_positive(label, values):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    assert finite.size > 0, f"{label} has no finite values"
    assert np.all(finite > 0.0), f"{label} must be positive"


if __name__ == "__main__":
    main()
