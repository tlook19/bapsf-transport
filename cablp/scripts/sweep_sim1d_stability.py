"""Small sim1d stability sweep for LAPD-ish operating points."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cablp.solvers._sim1d import LAPDSim1D, default_config, summarize_result


@dataclass(frozen=True)
class StabilityCase:
    name: str
    ne0: float
    Te0: float
    Ti0: float
    t_end: float
    dt_max: float
    cathode: bool = False
    twin: bool = False
    nn0: float = 2.0e12
    updates: dict[str, float | int | str | bool] = field(default_factory=dict)


CASES = (
    StabilityCase(
        name="no_cathode_low_ne",
        ne0=1.0e12,
        Te0=3.0,
        Ti0=1.0,
        t_end=5.0e-7,
        dt_max=2.0e-8,
    ),
    StabilityCase(
        name="no_cathode_hot_dense",
        ne0=1.0e13,
        Te0=15.0,
        Ti0=3.0,
        t_end=2.5e-7,
        dt_max=1.0e-8,
    ),
    StabilityCase(
        name="cathode_low_ne",
        ne0=1.0e12,
        Te0=3.0,
        Ti0=1.0,
        t_end=5.0e-7,
        dt_max=2.0e-8,
        cathode=True,
    ),
    StabilityCase(
        name="cathode_hot_dense",
        ne0=1.0e13,
        Te0=15.0,
        Ti0=3.0,
        t_end=2.5e-7,
        dt_max=1.0e-8,
        cathode=True,
    ),
    StabilityCase(
        name="twin_cathode_midrange",
        ne0=5.0e12,
        Te0=8.0,
        Ti0=2.0,
        t_end=2.5e-7,
        dt_max=1.0e-8,
        cathode=True,
        twin=True,
    ),
)


def main():
    reports = []
    for case in CASES:
        result, summary = run_case(case)
        check_case(case, result, summary)
        reports.append(format_report(case, summary))

    for report in reports:
        print(report)
    print(f"sim1d stability sweep ok: {len(reports)} cases")


def run_case(case):
    params, flags = default_config()
    params.update(
        {
            "nx": 20,
            "ne0": case.ne0,
            "Te0": case.Te0,
            "Ti0": case.Ti0,
            "nn0": case.nn0,
            "tau_prebreakdown": 2.0e-7 if case.t_end >= 5.0e-7 else 1.0e-7,
            "tau_breakdown": 0.0,
            "tau_discharge": 2.0e-7 if case.t_end >= 5.0e-7 else 1.0e-7,
            "tau_afterglow": 1.0e-7 if case.t_end >= 5.0e-7 else 5.0e-8,
            "phase_transition_mode": "current" if case.cathode else "scheduled",
            "I_prebreakdown": 1.0e-9,
            "I_breakdown": 1.0e-9,
            "dt_save": case.dt_max * 2.5,
            "dt_max": case.dt_max,
            "dt_min": 1.0e-15,
            "dt_growth_factor": 1.4,
            "max_step_retries": 16,
        }
    )
    params.update(case.updates)
    flags.update(
        {
            "cathode_coupling": case.cathode,
            "TwinCathode": case.twin,
            "implicit_heat_conduction": True,
        }
    )

    sim = LAPDSim1D(params, flags)
    result = sim.run(t_end=case.t_end, dt=None, max_steps=10000)
    return result, summarize_result(result)


def check_case(case, result, summary):
    assert summary.finite, f"{case.name}: non-finite fields {summary.finite_fields}"
    assert result.steps > 0, f"{case.name}: no accepted steps"
    assert np.isclose(result.final_time, case.t_end), (
        f"{case.name}: final_time={result.final_time:.6e}, expected={case.t_end:.6e}"
    )
    assert summary.accepted_dt_min >= 0.99e-15, (
        f"{case.name}: accepted dt below configured floor"
    )
    assert summary.accepted_dt_max <= case.dt_max * (1.0 + 1.0e-12), (
        f"{case.name}: accepted dt exceeded dt_max"
    )
    assert summary.max_retry_count <= 4, (
        f"{case.name}: excessive per-step retries {summary.max_retry_count}"
    )
    assert summary.total_retry_count <= 20, (
        f"{case.name}: excessive total retries {summary.total_retry_count}"
    )
    assert summary.timestep_rejection_event_count <= 20, (
        f"{case.name}: excessive timestep rejections "
        f"{summary.timestep_rejection_event_count}"
    )
    assert summary.n_min >= 0.99e8, f"{case.name}: density fell below floor"
    assert summary.nn_min >= 0.99e8, f"{case.name}: neutral density fell below floor"
    assert summary.Te_min >= 0.099, f"{case.name}: Te fell below floor"
    assert summary.Ti_min >= 0.099, f"{case.name}: Ti fell below floor"
    assert summary.n_max < 5.0 * case.ne0, f"{case.name}: density runaway"
    assert summary.Te_max < max(5.0 * case.Te0, 30.0), f"{case.name}: Te runaway"
    assert abs(summary.total_particle_inventory_relative_drift) < 0.25, (
        f"{case.name}: large particle inventory drift "
        f"{summary.total_particle_inventory_relative_drift:.3e}"
    )
    assert abs(summary.thermal_energy_relative_drift) < 2.0, (
        f"{case.name}: large thermal energy drift "
        f"{summary.thermal_energy_relative_drift:.3e}"
    )
    if case.cathode:
        cathode_fractions = summary.cathode_diagnostic_fractions
        assert cathode_fractions.get("configured", 0.0) == 1.0
        assert cathode_fractions.get("has_solution", 0.0) > 0.0
        assert np.nanmax(result.cathode.I_tot) > 0.0
        if case.twin:
            assert cathode_fractions.get("has_twin_solution", 0.0) > 0.0
            assert np.nanmax(result.cathode_twin.I_tot) > 0.0


def format_report(case, summary):
    constraints = ",".join(
        f"{name}:{count}" for name, count in summary.constraint_counts.items()
    )
    if not constraints:
        constraints = "none"
    return (
        f"PASS {case.name}: steps={summary.steps}, "
        f"dt=[{summary.accepted_dt_min:.2e},{summary.accepted_dt_max:.2e}] s, "
        f"retries={summary.total_retry_count}, "
        f"rejects={summary.timestep_rejection_event_count}, "
        f"n=[{summary.n_min:.2e},{summary.n_max:.2e}] cm^-3, "
        f"Te=[{summary.Te_min:.2f},{summary.Te_max:.2f}] eV, "
        f"particle_drift={summary.total_particle_inventory_relative_drift:.2e}, "
        f"thermal_drift={summary.thermal_energy_relative_drift:.2e}, "
        f"constraints={constraints}"
    )


if __name__ == "__main__":
    main()
