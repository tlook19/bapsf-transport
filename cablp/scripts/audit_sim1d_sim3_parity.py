"""Audit _sim3 defaults and flags against the _sim1d configuration surface.

The goal is not to make the two solvers expose identical knobs.  It is to keep
an explicit record of which _sim3 controls are implemented in _sim1d, which are
accepted only as scaffolding/diagnostics, which are still missing, and which do
not apply cleanly to the conservative 1D formulation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from cablp.solvers._sim1d import default_config as default_config_1d
from cablp.solvers._sim3 import default_config as default_config_3


IMPLEMENTED = "implemented"
SCAFFOLD = "accepted but diagnostic/scaffold only"
MISSING = "missing"
NOT_APPLICABLE = "intentionally not applicable to conservative 1D"

STATUSES = (IMPLEMENTED, SCAFFOLD, MISSING, NOT_APPLICABLE)


@dataclass(frozen=True)
class AuditEntry:
    status: str
    sim1d_key: str = ""
    note: str = ""


PARAM_AUDIT = {
    "gas_type": AuditEntry(IMPLEMENTED, "gas_type", "He/H species selector."),
    "ne0": AuditEntry(IMPLEMENTED, "ne0", "Uniform initial plasma density."),
    "Tn_fit": AuditEntry(IMPLEMENTED, "Tn_fit", "Reaction-rate neutral temperature."),
    "Te0": AuditEntry(IMPLEMENTED, "Te0", "Uniform initial electron temperature."),
    "Ti0": AuditEntry(IMPLEMENTED, "Ti0", "Uniform initial ion temperature."),
    "Lm": AuditEntry(IMPLEMENTED, "Lm", "Total 1D machine length."),
    "Rm": AuditEntry(IMPLEMENTED, "Rm", "Default neutral/machine radius."),
    "Rp": AuditEntry(IMPLEMENTED, "Rp", "Default plasma radius."),
    "Lp": AuditEntry(
        IMPLEMENTED,
        "Lz/source_length_cm/end_length_cm",
        "_sim1d tiles the full domain with explicit source/end lengths.",
    ),
    "cells": AuditEntry(
        IMPLEMENTED,
        "nx",
        "_sim1d uses resolved axial domain cells plus source/end cells.",
    ),
    "V_bank": AuditEntry(IMPLEMENTED, "V_bank", "Cathode circuit parameter."),
    "T_s": AuditEntry(IMPLEMENTED, "T_s", "Cathode circuit parameter."),
    "phi_wf": AuditEntry(IMPLEMENTED, "phi_wf", "Cathode circuit parameter."),
    "C_R": AuditEntry(IMPLEMENTED, "C_R", "Cathode circuit parameter."),
    "R_comp": AuditEntry(IMPLEMENTED, "R_comp", "Cathode circuit parameter."),
    "eta": AuditEntry(IMPLEMENTED, "eta", "Cathode/anode area ratio."),
    "L_cath": AuditEntry(IMPLEMENTED, "L_cath", "Cathode circuit parameter."),
    "R_cath": AuditEntry(IMPLEMENTED, "R_cath", "Cathode circuit parameter."),
    "gas_puff_mode": AuditEntry(IMPLEMENTED, "gas_puff_mode", "Phase-aware puff model."),
    "S_gp": AuditEntry(IMPLEMENTED, "S_gp", "Primary gas puff source."),
    "Twin_S_gp": AuditEntry(IMPLEMENTED, "Twin_S_gp", "Twin/end gas puff source."),
    "S_gp_decay_target": AuditEntry(IMPLEMENTED, "S_gp_decay_target", "Puff decay target."),
    "Twin_S_gp_decay_target": AuditEntry(
        IMPLEMENTED,
        "Twin_S_gp_decay_target",
        "Twin/end puff decay target.",
    ),
    "S_pump_L": AuditEntry(IMPLEMENTED, "S_pump_L", "Source-side pump."),
    "S_pump_R": AuditEntry(IMPLEMENTED, "S_pump_R", "End-side pump."),
    "b_epara": AuditEntry(
        IMPLEMENTED,
        "b_epara",
        "Electron heat-conduction scale, using conservative volumetric form.",
    ),
    "b_ipara": AuditEntry(
        IMPLEMENTED,
        "b_ipara",
        "Ion heat-conduction scale, using conservative volumetric form.",
    ),
    "b_ioniz": AuditEntry(IMPLEMENTED, "b_ioniz", "Bulk ionization scale."),
    "b_rec_rad": AuditEntry(IMPLEMENTED, "b_rec_rad", "Radiative recombination scale."),
    "b_rec_3b": AuditEntry(IMPLEMENTED, "b_rec_3b", "Three-body recombination scale."),
    "b_Qcx": AuditEntry(IMPLEMENTED, "b_Qcx", "Ion charge-exchange cooling scale."),
    "b_Qie": AuditEntry(IMPLEMENTED, "b_Qie", "Electron-ion energy exchange scale."),
    "b_Qei": AuditEntry(IMPLEMENTED, "b_Qei", "Electron-ion energy exchange scale."),
    "b_Qen": AuditEntry(IMPLEMENTED, "b_Qen", "Electron-neutral cooling scale."),
    "b_div_v_elec": AuditEntry(
        IMPLEMENTED,
        "b_pressure_work_elec",
        "_sim1d pressure work is conservative energy source bookkeeping.",
    ),
    "b_div_v_ions": AuditEntry(
        IMPLEMENTED,
        "b_pressure_work_ions",
        "_sim1d pressure work is conservative energy source bookkeeping.",
    ),
    "b_Te_conv": AuditEntry(
        NOT_APPLICABLE,
        "",
        "Primitive temperature convection is replaced by conservative energy flux.",
    ),
    "b_Ti_conv": AuditEntry(
        NOT_APPLICABLE,
        "",
        "Primitive temperature convection is replaced by conservative energy flux.",
    ),
    "b_source": AuditEntry(
        MISSING,
        "",
        "No direct _sim1d scale for cathode/source electron heating yet.",
    ),
    "cycles": AuditEntry(
        MISSING,
        "",
        "Neutral-only cycling currently uses t_end/tau_cycle rather than cycles.",
    ),
    "tau_prebreakdown": AuditEntry(IMPLEMENTED, "tau_prebreakdown", "Phase timing."),
    "tau_discharge": AuditEntry(IMPLEMENTED, "tau_discharge", "Phase timing."),
    "tau_gp_after_breakdown": AuditEntry(
        IMPLEMENTED,
        "tau_gp_after_breakdown",
        "Gas-puff decay timing.",
    ),
    "tau_gp_decay_factor": AuditEntry(
        IMPLEMENTED,
        "tau_gp_decay_factor",
        "Gas-puff decay timing.",
    ),
    "tau_gp_pulse_duration": AuditEntry(
        IMPLEMENTED,
        "tau_gp_pulse_duration",
        "Pulse-decay gas puff mode.",
    ),
    "tau_gp_decay_duration": AuditEntry(
        IMPLEMENTED,
        "tau_gp_decay_duration",
        "Pulse-decay gas puff mode.",
    ),
    "tau_afterglow": AuditEntry(IMPLEMENTED, "tau_afterglow", "Phase timing."),
    "tau_cycle": AuditEntry(IMPLEMENTED, "tau_cycle", "Neutral-only cycle timing."),
    "I_prebreakdown": AuditEntry(
        IMPLEMENTED,
        "I_prebreakdown",
        "Current-triggered phase transition threshold.",
    ),
    "I_breakdown": AuditEntry(
        IMPLEMENTED,
        "I_breakdown",
        "Current-triggered phase transition threshold.",
    ),
    "h0": AuditEntry(
        IMPLEMENTED,
        "dt",
        "Initial step is supplied to run(); adaptive suggestion uses dt_max.",
    ),
    "h_max_discharge": AuditEntry(
        IMPLEMENTED,
        "dt_max",
        "_sim1d uses one explicit maximum timestep control.",
    ),
    "h_max_afterglow": AuditEntry(
        IMPLEMENTED,
        "dt_max",
        "_sim1d uses one explicit maximum timestep control.",
    ),
    "max_step_rejections": AuditEntry(
        IMPLEMENTED,
        "max_step_retries",
        "Adaptive retry guard with conservative-step health checks.",
    ),
    "dt_save": AuditEntry(IMPLEMENTED, "dt_save", "Saved-output cadence."),
    "max_output_steps": AuditEntry(IMPLEMENTED, "max_output_steps", "Saved-output cap."),
    "rtol": AuditEntry(
        NOT_APPLICABLE,
        "",
        "_sim1d timestep selection is CFL/source-limited, not RK error-controlled.",
    ),
    "h_min": AuditEntry(IMPLEMENTED, "dt_min", "Minimum timestep guard."),
    "h_min_prebreakdown": AuditEntry(
        NOT_APPLICABLE,
        "",
        "_sim1d uses the same dt_min guard for all phases.",
    ),
    "prebreakdown_cfl_factor": AuditEntry(
        NOT_APPLICABLE,
        "",
        "_sim1d CFL limits are conservative finite-volume limits.",
    ),
    "ne_cfl_floor": AuditEntry(
        NOT_APPLICABLE,
        "",
        "_sim1d heat/CFL limits use conservative state floors directly.",
    ),
    "ne_floor": AuditEntry(IMPLEMENTED, "ne_floor", "Plasma density floor."),
    "nn_floor": AuditEntry(IMPLEMENTED, "nn_floor", "Neutral density floor."),
    "Te_floor": AuditEntry(IMPLEMENTED, "Te_floor", "Electron temperature floor."),
    "Ti_floor": AuditEntry(IMPLEMENTED, "Ti_floor", "Ion temperature floor."),
    "Te_reject_floor": AuditEntry(
        SCAFFOLD,
        "adaptive_retries_enabled",
        "Retries reject non-finite/negative states, but no per-variable reject floor.",
    ),
    "Ti_reject_floor": AuditEntry(
        SCAFFOLD,
        "adaptive_retries_enabled",
        "Retries reject non-finite/negative states, but no per-variable reject floor.",
    ),
    "ne_reject_floor": AuditEntry(
        SCAFFOLD,
        "adaptive_retries_enabled",
        "Retries reject non-finite/negative states, but no per-variable reject floor.",
    ),
    "nn_reject_floor": AuditEntry(
        SCAFFOLD,
        "adaptive_retries_enabled",
        "Retries reject non-finite/negative states, but no per-variable reject floor.",
    ),
    "debug_max_rel_step_change": AuditEntry(
        SCAFFOLD,
        "max_density_step_fraction/max_energy_step_fraction",
        "Related adaptive retry limits exist with different names and scope.",
    ),
    "debug_max_neighbor_ratio": AuditEntry(
        MISSING,
        "",
        "No adjacent-cell ratio debug guard yet.",
    ),
    "v_atol_cs_fraction": AuditEntry(
        NOT_APPLICABLE,
        "",
        "_sim1d does not use _sim3 RK component tolerances.",
    ),
    "debug_step_atol": AuditEntry(
        NOT_APPLICABLE,
        "",
        "_sim1d does not use _sim3 RK component tolerances.",
    ),
    "debug_check_start_time": AuditEntry(
        MISSING,
        "",
        "No delayed debug-check start time yet.",
    ),
    "debug_ignore_floor_neighbors": AuditEntry(
        MISSING,
        "",
        "No adjacent-cell ratio debug guard yet.",
    ),
    "alpha_ne_sonic_flux": AuditEntry(
        IMPLEMENTED,
        "alpha_front",
        "1D front-filling flux cap.",
    ),
    "beta_ne_sonic_flux": AuditEntry(
        IMPLEMENTED,
        "alpha_front",
        "Mapped to a single front-filling cap in _sim1d.",
    ),
    "hybrid_ne_taper_dn0": AuditEntry(
        MISSING,
        "",
        "No density-contrast taper for front flux yet.",
    ),
    "hybrid_ne_taper_power": AuditEntry(
        MISSING,
        "",
        "No density-contrast taper for front flux yet.",
    ),
    "hybrid_ne_taper_delay": AuditEntry(
        MISSING,
        "",
        "No post-breakdown delay taper for front flux yet.",
    ),
    "ion_pressure_weight": AuditEntry(
        NOT_APPLICABLE,
        "",
        "_sim1d momentum flux carries electron and ion pressure explicitly.",
    ),
}


FLAG_AUDIT = {
    "icool": AuditEntry(IMPLEMENTED, "icool", "Electron-ion cooling toggle."),
    "ncool": AuditEntry(IMPLEMENTED, "ncool", "Electron-neutral cooling toggle."),
    "cx": AuditEntry(IMPLEMENTED, "cx", "Charge-exchange cooling toggle."),
    "icool_recomb": AuditEntry(
        IMPLEMENTED,
        "icool_recomb",
        "Accepted by reaction/cooling paths.",
    ),
    "Plasma": AuditEntry(IMPLEMENTED, "Plasma", "Plasma vs neutral-only evolution."),
    "TwinCathode": AuditEntry(IMPLEMENTED, "TwinCathode", "Twin/end source behavior."),
    "Velocity": AuditEntry(
        NOT_APPLICABLE,
        "",
        "_sim1d always evolves conservative momentum.",
    ),
    "advection": AuditEntry(
        NOT_APPLICABLE,
        "",
        "_sim1d advects conservative state through baseline FV fluxes.",
    ),
    "adaptive_mesh": AuditEntry(
        NOT_APPLICABLE,
        "",
        "_sim1d currently uses a fixed resolved axial grid.",
    ),
    "hybrid_ne": AuditEntry(
        IMPLEMENTED,
        "front_flux",
        "Mapped to the explicit 1D front-filling flux toggle.",
    ),
    "debug_checks": AuditEntry(IMPLEMENTED, "debug_checks", "Basic debug checks."),
    "debug_raise_on_guard": AuditEntry(
        MISSING,
        "",
        "No separate raise-on-flooring guard yet.",
    ),
    "reject_floor_violations": AuditEntry(
        SCAFFOLD,
        "adaptive_retries_enabled",
        "Adaptive retries reject unhealthy states, but not this exact toggle.",
    ),
    "reject_large_step_changes": AuditEntry(
        SCAFFOLD,
        "adaptive_retries_enabled",
        "Adaptive retries use fractional-change guards with different names.",
    ),
}


def main(argv=None):
    args = _parse_args(argv)
    sim3_params, sim3_flags = default_config_3()
    sim1d_params, sim1d_flags = default_config_1d()

    param_rows = _audit_rows(
        label="param",
        sim3_defaults=sim3_params,
        sim1d_defaults=sim1d_params,
        audit=PARAM_AUDIT,
    )
    flag_rows = _audit_rows(
        label="flag",
        sim3_defaults=sim3_flags,
        sim1d_defaults=sim1d_flags,
        audit=FLAG_AUDIT,
    )
    rows = param_rows + flag_rows

    _validate_audit("param", sim3_params, PARAM_AUDIT)
    _validate_audit("flag", sim3_flags, FLAG_AUDIT)

    print("# _sim1d / _sim3 Config Parity Audit")
    print()
    print(_summary(rows))
    print()
    print(_table(rows))

    if args.show_sim1d_only:
        print()
        print("## _sim1d-only controls")
        print()
        print(_sim1d_only_table(sim3_params, sim3_flags, sim1d_params, sim1d_flags))

    if args.fail_on_missing and any(row["status"] == MISSING for row in rows):
        return 1
    return 0


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Classify _sim3 config defaults relative to _sim1d."
    )
    parser.add_argument(
        "--show-sim1d-only",
        action="store_true",
        help="Also list _sim1d-native params and flags not present in _sim3.",
    )
    parser.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="Exit with status 1 if any audited _sim3 key is classified missing.",
    )
    return parser.parse_args(argv)


def _audit_rows(label, sim3_defaults, sim1d_defaults, audit):
    rows = []
    for key in sorted(sim3_defaults):
        entry = audit[key]
        rows.append(
            {
                "type": label,
                "sim3_key": key,
                "sim1d_key": entry.sim1d_key,
                "status": entry.status,
                "accepted_same_name": "yes" if key in sim1d_defaults else "no",
                "sim3_default": _format_default(sim3_defaults[key]),
                "sim1d_default": _format_default(sim1d_defaults.get(key, "")),
                "note": entry.note,
            }
        )
    return rows


def _validate_audit(label, sim3_defaults, audit):
    sim3_keys = set(sim3_defaults)
    audit_keys = set(audit)
    missing_audit = sorted(sim3_keys - audit_keys)
    stale_audit = sorted(audit_keys - sim3_keys)
    bad_status = sorted(
        key for key, entry in audit.items() if entry.status not in STATUSES
    )
    errors = []
    if missing_audit:
        errors.append(f"{label} audit is missing keys: {', '.join(missing_audit)}")
    if stale_audit:
        errors.append(f"{label} audit has stale keys: {', '.join(stale_audit)}")
    if bad_status:
        errors.append(f"{label} audit has invalid statuses: {', '.join(bad_status)}")
    if errors:
        raise SystemExit("\n".join(errors))


def _summary(rows):
    total = len(rows)
    counts = {status: 0 for status in STATUSES}
    for row in rows:
        counts[row["status"]] += 1
    parts = [f"{status}: {counts[status]}" for status in STATUSES]
    return f"Audited {total} _sim3 controls. " + "; ".join(parts) + "."


def _table(rows):
    headers = (
        "type",
        "sim3_key",
        "status",
        "accepted_same_name",
        "sim1d_key",
        "sim3_default",
        "sim1d_default",
        "note",
    )
    return _markdown_table(headers, rows)


def _sim1d_only_table(sim3_params, sim3_flags, sim1d_params, sim1d_flags):
    rows = []
    for key in sorted(set(sim1d_params) - set(sim3_params)):
        rows.append(
            {
                "type": "param",
                "sim1d_key": key,
                "sim1d_default": _format_default(sim1d_params[key]),
            }
        )
    for key in sorted(set(sim1d_flags) - set(sim3_flags)):
        rows.append(
            {
                "type": "flag",
                "sim1d_key": key,
                "sim1d_default": _format_default(sim1d_flags[key]),
            }
        )
    return _markdown_table(("type", "sim1d_key", "sim1d_default"), rows)


def _markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(_escape_markdown(str(row.get(header, ""))) for header in headers)
            + " |"
        )
    return "\n".join(lines)


def _format_default(value):
    if value == "":
        return ""
    return repr(value)


def _escape_markdown(value):
    return value.replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    raise SystemExit(main())
