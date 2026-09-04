"""Small sim1d stability sweep over a NAMED configuration's corners.

Four short fixed-corner cases -- seed density, temperatures, cathode on/off --
each run for a fraction of a microsecond and checked for finiteness and drift.
It answers "does the integrator stay well behaved near these corners", and the
answer is only meaningful about a configuration someone can name: the same
corner is stable under one closure and marginal under another.

``--stance NAME`` (or a configuration file's path) is that configuration and is
the BASE of the sweep; the per-case deltas below are applied over it, exactly
where a campaign driver applies its rung. A sweep that genuinely names none
says so with ``--no-stance``.

    python scripts/run/sweep_sim1d_stability.py --stance g1atrim
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np

from cablp.solvers._sim1d import LAPDSim1D, default_config, summarize_result

# scripts/ sibling imports: the seven purpose subdirectories on sys.path.
import sys as _sys
from pathlib import Path as _Path
for _sub in ("atomic", "gates", "kinetic", "run", "score", "stance",
             "verify"):
    _dir = str(_Path(__file__).resolve().parents[1] / _sub)
    if _dir not in _sys.path:
        _sys.path.insert(0, _dir)

from stance_config import (  # noqa: E402
    available_stances,
    load_named_configuration,
    without_mesh_sized_package,
)


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
    # twin_cathode_midrange retired 2026-09-03: it cannot be constructed on
    # the single-cathode source_fixed_grid layout (see RETIRED_CASES).
)

#: Corners removed from ``CASES``, and the reason ``--list``/``--only`` state
#: when the name is asked for. A retired corner is refused by name rather than
#: skipped, so a caller who names one is told why it is gone.
RETIRED_CASES = {
    "twin_cathode_midrange":
        "source_fixed_grid is defined only for the single-cathode layout, so "
        "a twin-cathode corner has no meaning on this source grid",
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    stance_group = ap.add_mutually_exclusive_group()
    stance_group.add_argument(
        "--stance", metavar="NAME", default=None,
        help="committed configuration file (scripts/stances/NAME.toml), or a "
             "configuration file's path, this sweep is based on. Available: "
             + (", ".join(available_stances()) or "(none committed)"))
    stance_group.add_argument(
        "--no-stance", action="store_true",
        help="acknowledge that this sweep names no configuration and runs "
             "this script's own corner definitions over the bare templates")
    ap.add_argument(
        "--list", action="store_true",
        help="print the corner names this sweep runs, and any retired name "
             "with the reason it was retired, then exit")
    ap.add_argument(
        "--only", action="append", default=[], metavar="NAME[,NAME...]",
        help="run only these corners (repeatable, comma-separated); a retired "
             "or unknown name is refused")
    args = ap.parse_args(argv)

    if args.list:
        for case in CASES:
            print(case.name)
        for name, reason in sorted(RETIRED_CASES.items()):
            print(f"{name}  RETIRED -- {reason}")
        return

    # Resolved BEFORE the configuration requirement below: a retired or
    # misspelled corner name is wrong however the sweep is invoked, and
    # answering it with the stance error instead would name the wrong fault.
    requested = [n for chunk in args.only for n in chunk.split(",") if n]
    retired = [n for n in requested if n in RETIRED_CASES]
    if retired:
        raise SystemExit(
            "sweep_sim1d_stability: "
            + "; ".join(
                f"corner {n} is RETIRED -- {RETIRED_CASES[n]}"
                for n in retired
            )
        )
    known = {case.name for case in CASES}
    unknown = [n for n in requested if n not in known]
    if unknown:
        raise SystemExit(
            "sweep_sim1d_stability: unknown corner name(s): "
            f"{', '.join(unknown)} (see --list)"
        )
    selected = ([case for case in CASES if case.name in requested]
                if requested else list(CASES))

    # A stability verdict is a statement ABOUT a configuration: the same corner
    # is well behaved under one closure and marginal under another, so a sweep
    # that names none reports a number nobody can place.
    if args.stance is None and not args.no_stance:
        raise SystemExit(
            "sweep_sim1d_stability: name the configuration package. Pass "
            "--stance <name> to sweep a committed stance file "
            "(available: "
            f"{', '.join(available_stances()) or '(none committed)'})"
            ", or --no-stance to acknowledge that this sweep has none and "
            "runs "
            "this script's own corner definitions over the bare templates."
        )

    base = (
        None if args.stance is None
        else load_named_configuration(args.stance)
    )
    print(
        "sim1d stability sweep configuration: "
        + ("<unnamed>" if base is None else f"{base.name} ({base.path})")
    )
    reports = []
    for case in selected:
        result, summary = run_case(case, base)
        check_case(case, result, summary)
        reports.append(format_report(case, summary))

    for report in reports:
        print(report)
    print(f"sim1d stability sweep ok: {len(reports)} cases")


def run_case(case, base=None):
    params, flags = default_config()
    if base is not None:
        # MINUS THE MESH-SIZED PACKAGE. These corners run twenty cells for a
        # fraction of a microsecond, and a per-cell profile sized for the
        # configuration's own mesh refuses any other one. What travels is the
        # operating point, which is what the corners are being asked about;
        # each case supplies its own seed fill below.
        base_params, base_flags = without_mesh_sized_package(
            base.params, base.flags
        )
        params.update(base_params)
        flags.update(base_flags)
        if not case.cathode:
            # A CATHODE-OFF CORNER HAS NO CATHODE RECYCLE. The reference
            # configuration arms the DVM's cathode jet, which launches the
            # recycle at the sheath energy phi_c the cathode solve supplies;
            # with cathode_coupling off there is no solve and the solver
            # refuses the pair rather than launching thermal atoms as if they
            # carried sheath energy. Turning the channel off is what "no
            # cathode" MEANS here, not a physics choice this sweep is making.
            # The ANODE jet goes with it: the cathode, anode and bank are one
            # system and one solve, so phi_a is unavailable for the same
            # reason phi_c is.
            params["neutral_kinetic_dvm_cathode_jet"] = False
            params["neutral_kinetic_dvm_anode_jet"] = False
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
        # Read the per-electrode loop current straight out of the saved
        # cathode diagnostics. This used to go through the retired
        # results/compat.py namespaces (``result.cathode`` /
        # ``result.cathode_twin``), which were nothing but views of these two
        # keys -- NaN-filled on frames with no solve, exactly as the keys
        # themselves are seeded, so nanmax reads the same numbers here.
        assert np.nanmax(result.cathode_diagnostics["source_I_tot"]) > 0.0
        if case.twin:
            assert cathode_fractions.get("has_twin_solution", 0.0) > 0.0
            assert np.nanmax(result.cathode_diagnostics["end_I_tot"]) > 0.0


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
