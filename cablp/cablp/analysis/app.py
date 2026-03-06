"""
LAPDSim Parameter Sweep GUI — Streamlit application.

Launch with:
    poetry run lapd-app
or:
    streamlit run cablp/cablp/analysis/app.py
"""
from __future__ import annotations

import json
import os
import pathlib
import queue
import threading
import time
from dataclasses import dataclass, field

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import psutil
import streamlit as st

matplotlib.use("Agg")  # non-interactive backend required for Streamlit

# Absolute imports (relative imports fail when Streamlit runs app.py as __main__)
from cablp.analysis.sweep import grid_sweep_parallel, param_combinations
from cablp.analysis.database import open_db, load_index, list_runs, load_run, rebuild_index, update_index
from cablp.analysis.plot import (
    plot_run,
    plot_run_comparison,
    plot_sweep_variance,
    plot_sweep_heatmap,
)

# ── Sweep progress manifest (persisted alongside the HDF5 database) ───────────

def _manifest_path(db_path: str) -> pathlib.Path:
    """Return the JSON manifest path for a given database path."""
    p = pathlib.Path(db_path).expanduser()
    return p.parent / (p.stem + ".progress.json")


def _save_manifest(db_path: str, data: dict) -> None:
    """Save sweep-progress state to a JSON file next to the database."""
    try:
        mp = _manifest_path(db_path)
        mp.parent.mkdir(parents=True, exist_ok=True)
        with open(mp, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass  # non-critical; never crash the sweep


def _load_manifest(db_path: str) -> dict | None:
    """Load sweep-progress state.  Returns None if not found or invalid."""
    try:
        mp = _manifest_path(db_path)
        if not mp.exists():
            return None
        with open(mp) as f:
            return json.load(f)
    except Exception:
        return None


# ── Directory / file browser widget ───────────────────────────────────────────

def _render_path_input(key: str, label: str, default: str, file_ext: str = ".h5") -> str:
    """
    Text input with a collapsible directory browser for selecting an HDF5 file path.
    Returns the current path value (unexpanded, as the user typed it).
    """
    txt_key = f"_pathtxt_{key}"
    dir_key = f"_pathdir_{key}"
    show_key = f"_pathshow_{key}"

    if txt_key not in st.session_state:
        st.session_state[txt_key] = default

    col_input, col_btn = st.columns([8, 1])
    with col_input:
        st.text_input(label, key=txt_key)
    if col_btn.button("📂", key=f"_pathbtn_{key}", help="Browse filesystem"):
        cur = pathlib.Path(st.session_state[txt_key]).expanduser()
        # Start browse from parent dir of current path, or best fallback
        candidates = [
            cur.parent if (cur.suffix == file_ext or cur.is_file()) else cur,
            pathlib.Path("~/lapd_data").expanduser(),
            pathlib.Path.home(),
        ]
        for c in candidates:
            if c.exists():
                st.session_state[dir_key] = str(c)
                break
        st.session_state[show_key] = not st.session_state.get(show_key, False)
        st.rerun()

    if st.session_state.get(show_key, False):
        cur_dir = pathlib.Path(st.session_state.get(dir_key, str(pathlib.Path.home())))
        st.caption(f"📁 `{cur_dir}`")

        parent = cur_dir.parent
        if parent != cur_dir:
            if st.button("⬆ Parent dir", key=f"_pathup_{key}"):
                st.session_state[dir_key] = str(parent)
                st.rerun()

        try:
            entries = sorted(
                cur_dir.iterdir(),
                key=lambda e: (e.is_file(), e.name.lower()),
            )
            dirs = [e for e in entries if e.is_dir() and not e.name.startswith(".")]
            files = [e for e in entries if e.is_file() and e.suffix == file_ext]

            if dirs:
                n_cols = min(4, len(dirs))
                dir_cols = st.columns(n_cols)
                for ci, d in enumerate(dirs[:12]):
                    if dir_cols[ci % n_cols].button(f"📁 {d.name}", key=f"_pd_{key}_{d.name}"):
                        st.session_state[dir_key] = str(d)
                        st.rerun()

            for f_entry in files:
                if st.button(f"📄 {f_entry.name}", key=f"_pf_{key}_{f_entry.name}"):
                    st.session_state[txt_key] = str(f_entry)
                    st.session_state[show_key] = False
                    st.rerun()

            nc1, nc2 = st.columns([5, 1])
            new_name = nc1.text_input(
                "New filename",
                key=f"_pnew_{key}",
                placeholder=f"filename{file_ext}",
                label_visibility="collapsed",
            )
            if nc2.button("Use", key=f"_puse_{key}") and new_name:
                fn = new_name if new_name.endswith(file_ext) else new_name + file_ext
                st.session_state[txt_key] = str(cur_dir / fn)
                st.session_state[show_key] = False
                st.rerun()

        except PermissionError:
            st.warning("Permission denied")

    return st.session_state.get(txt_key, default)


# ── Abort-helper: mark planned-but-not-started runs as failed in the DB ────────

def _mark_incomplete_as_failed(db_path: str, manifest: dict) -> None:
    """Write a 'failed' record for every planned run_id not yet in the database."""
    planned_ids = manifest.get("planned_run_ids", [])
    if not planned_ids:
        st.warning("Manifest has no planned run IDs; nothing to mark.")
        return
    try:
        with open_db(db_path, mode="a") as db:
            existing = set(list_runs(db))
            to_mark = [rid for rid in planned_ids if rid not in existing]
            for run_id in to_mark:
                grp = db.require_group("runs").require_group(run_id)
                grp.attrs["status"] = "failed"
                grp.attrs["error"] = "Sweep was aborted before this run could execute."
                update_index(db, run_id, {}, {}, {}, 0, status="failed")
        if to_mark:
            st.success(f"Marked {len(to_mark)} incomplete run(s) as failed.")
        else:
            st.info("All planned runs are already recorded in the database.")
    except Exception as exc:
        st.error(f"Failed to mark incomplete runs: {exc}")


# ── Parameter / flag metadata ─────────────────────────────────────────────────
# Each entry: key → {label, unit, default, type, group}
# type: "float" | "int" | "str" | "bool"
# str entries also have "choices": list

PARAM_META: dict[str, dict] = {
    # ── Gas & Initial Conditions ──────────────────────────────────────────────
    "gas_type": {
        "label": "Gas type", "unit": "", "default": "He",
        "type": "str", "group": "Gas & Initial Conditions",
        "choices": ["He", "H"],
    },
    "ne0": {
        "label": "Initial electron density (ne0)", "unit": "cm⁻³", "default": 1e9,
        "type": "float", "group": "Gas & Initial Conditions",
    },
    "nn0": {
        "label": "Initial neutral density (nn0)", "unit": "cm⁻³", "default": 5e12,
        "type": "float", "group": "Gas & Initial Conditions",
    },
    "Te0": {
        "label": "Initial electron temperature (Te0)", "unit": "eV", "default": 0.1,
        "type": "float", "group": "Gas & Initial Conditions",
    },
    "Ti0": {
        "label": "Initial ion temperature (Ti0)", "unit": "eV", "default": 0.1,
        "type": "float", "group": "Gas & Initial Conditions",
    },
    "Tn_fit": {
        "label": "Neutral temp for rate fits (Tn_fit)", "unit": "eV", "default": 0.1,
        "type": "float", "group": "Gas & Initial Conditions",
    },
    # ── Machine Geometry ──────────────────────────────────────────────────────
    "Lm": {
        "label": "Machine length (Lm)", "unit": "cm", "default": 1800.0,
        "type": "float", "group": "Machine Geometry",
    },
    "Rm": {
        "label": "Machine radius (Rm)", "unit": "cm", "default": 50.0,
        "type": "float", "group": "Machine Geometry",
    },
    "Lp": {
        "label": "Plasma length (Lp)", "unit": "cm", "default": 1800.0,
        "type": "float", "group": "Machine Geometry",
    },
    "Rp": {
        "label": "Plasma radius (Rp)", "unit": "cm", "default": 18.0,
        "type": "float", "group": "Machine Geometry",
    },
    "Rhf": {
        "label": "Radial heat flux scale length (Rhf)", "unit": "cm", "default": 50.0,
        "type": "float", "group": "Machine Geometry",
    },
    # ── Magnetic Field ────────────────────────────────────────────────────────
    "Bz0": {
        "label": "Axial magnetic field (Bz0)", "unit": "G", "default": 1500.0,
        "type": "float", "group": "Magnetic Field",
    },
    # ── Discharge (Primary Cathode) ───────────────────────────────────────────
    "Vd": {
        "label": "Discharge voltage (Vd)", "unit": "V", "default": 100.0,
        "type": "float", "group": "Discharge (Primary Cathode)",
    },
    "P_in": {
        "label": "Total Input Power (P_in)", "unit": "MW", "default": 0.5,
        "type": "float", "group": "Discharge (Primary Cathode)",
    },
    "S_gp": {
        "label": "Gas puff source rate (S_gp)", "unit": "", "default": 500.0,
        "type": "float", "group": "Discharge (Primary Cathode)",
    },
    "anode_transparency": {
        "label": "Anode transparency", "unit": "", "default": 1.0,
        "type": "float", "group": "Discharge (Primary Cathode)",
    },
    # ── Source / Sinks ────────────────────────────────────────────────────────
    "Source_nn0": {
        "label": "Source neutral density (Source_nn0)", "unit": "cm⁻³", "default": 1.2e13,
        "type": "float", "group": "Source / Sinks",
    },
    "S_pump_L": {
        "label": "Pump sink rate left (S_pump_L)", "unit": "s⁻¹", "default": 4000.0,
        "type": "float", "group": "Source / Sinks",
    },
    "S_pump_R": {
        "label": "Pump sink rate right (S_pump_R)", "unit": "s⁻¹", "default": 4000.0,
        "type": "float", "group": "Source / Sinks",
    },
    # ── Time & Solver ─────────────────────────────────────────────────────────
    "cells": {
        "label": "Number of cells", "unit": "", "default": 3,
        "type": "int", "group": "Time & Solver",
    },
    "end": {
        "label": "End time (end)", "unit": "s", "default": 21e-3,
        "type": "float", "group": "Time & Solver",
    },
    "d_off": {
        "label": "Discharge off time (d_off)", "unit": "s", "default": 20e-3,
        "type": "float", "group": "Time & Solver",
    },
    "dt_main": {
        "label": "Main time step (dt_main)", "unit": "s", "default": 3e-8,
        "type": "float", "group": "Time & Solver",
    },
    "dt_after": {
        "label": "Time step after discharge (dt_after)", "unit": "s", "default": 1e-7,
        "type": "float", "group": "Time & Solver",
    },
    "tau_I_on": {
        "label": "Beam current rise time (tau_I_on)", "unit": "s", "default": 0.001,
        "type": "float", "group": "Time & Solver",
    },
    "cycles": {
        "label": "Discharge cycles", "unit": "", "default": 1,
        "type": "int", "group": "Time & Solver",
    },
    "rtol": {
        "label": "Relative tolerance (rtol)", "unit": "", "default": 1e-3,
        "type": "float", "group": "Time & Solver",
    },
    "h_min": {
        "label": "Min step size (h_min)", "unit": "s", "default": 1e-12,
        "type": "float", "group": "Time & Solver",
    },
    # ── Transport Scaling ─────────────────────────────────────────────────────
    "b_epara": {"label": "e⁻ parallel scale (b_epara)", "unit": "", "default": 1.0, "type": "float", "group": "Transport Scaling"},
    "b_ipara": {"label": "Ion parallel scale (b_ipara)", "unit": "", "default": 1.0, "type": "float", "group": "Transport Scaling"},
    "b_eperp": {"label": "e⁻ perp scale (b_eperp)", "unit": "", "default": 1.0, "type": "float", "group": "Transport Scaling"},
    "b_iperp": {"label": "Ion perp scale (b_iperp)", "unit": "", "default": 1.0, "type": "float", "group": "Transport Scaling"},
    "b_ioniz": {"label": "Ionization scale (b_ioniz)", "unit": "", "default": 1.0, "type": "float", "group": "Transport Scaling"},
    "b_rec_rad": {"label": "Rad recombination scale (b_rec_rad)", "unit": "", "default": 1.0, "type": "float", "group": "Transport Scaling"},
    "b_rec_3b": {"label": "3-body recombination scale (b_rec_3b)", "unit": "", "default": 1.0, "type": "float", "group": "Transport Scaling"},
    "b_Qcx": {"label": "Charge exchange scale (b_Qcx)", "unit": "", "default": 1.0, "type": "float", "group": "Transport Scaling"},
    "b_source": {"label": "Source heating scale (b_source)", "unit": "", "default": 1.0, "type": "float", "group": "Transport Scaling"},
    "b_Qie": {"label": "Q_ie scale (b_Qie)", "unit": "", "default": 1.0, "type": "float", "group": "Transport Scaling"},
    "b_Qei": {"label": "Q_ei scale (b_Qei)", "unit": "", "default": 1.0, "type": "float", "group": "Transport Scaling"},
    "b_Qen": {"label": "Q_en scale (b_Qen)", "unit": "", "default": 1.0, "type": "float", "group": "Transport Scaling"},
}

# Twin cathode params rendered separately under Dual Cathode section
TWIN_META: dict[str, dict] = {
    "Twin_Vd": {"label": "Twin discharge voltage (Twin_Vd)", "unit": "V", "default": 100.0, "type": "float"},
    "Twin_Id": {"label": "Twin discharge current (Twin_Id)", "unit": "A", "default": 5000.0, "type": "float"},
    "Twin_nn0": {"label": "Twin neutral density (Twin_nn0)", "unit": "cm⁻³", "default": 1.2e13, "type": "float"},
    "Twin_S_gp": {"label": "Twin gas puff rate (Twin_S_gp)", "unit": "", "default": 500.0, "type": "float"},
}

FLAG_META: dict[str, dict] = {
    "eperp": {"label": "Electron perp transport", "default": False, "group": "Transport"},
    "iperp": {"label": "Ion perp transport", "default": False, "group": "Transport"},
    "icool": {"label": "Ion cooling", "default": True, "group": "Physics"},
    "ncool": {"label": "Neutral cooling", "default": True, "group": "Physics"},
    "cx": {"label": "Charge exchange", "default": True, "group": "Physics"},
    "icool_recomb": {"label": "Ion cooling from recombination", "default": False, "group": "Physics"},
    "mit_el": {"label": "MIT electron flag (mit_el)", "default": False, "group": "Physics"},
    "C_imp": {"label": "Carbon impurity (C_imp)", "default": False, "group": "Physics"},
    "O_imp": {"label": "Oxygen impurity (O_imp)", "default": False, "group": "Physics"},
    "Plasma": {"label": "Plasma physics", "default": True, "group": "Simulation"},
    "Velocity": {"label": "Plasma velocity", "default": True, "group": "Simulation"},
    "breakdown_vel": {"label": "Diffusive flux during breakdown", "default": True, "group": "Simulation"},
    "adaptive": {"label": "Adaptive time stepping (RK45)", "default": False, "group": "Simulation"},
}

PARAM_GROUP_ORDER = [
    "Gas & Initial Conditions",
    "Machine Geometry",
    "Magnetic Field",
    "Discharge (Primary Cathode)",
    "Source / Sinks",
    "Time & Solver",
    "Transport Scaling",
]

FLAG_GROUP_ORDER = ["Transport", "Physics", "Simulation"]


# ── Config persistence ─────────────────────────────────────────────────────────

_CONFIG_PATH = pathlib.Path.home() / ".lapd_app_config.json"

# All session-state keys that represent widget settings to save/restore
_PARAM_CFG_KEYS: list[str] = []
for _k in list(PARAM_META.keys()) + list(TWIN_META.keys()):
    _PARAM_CFG_KEYS += [
        f"pmode_{_k}", f"pfixed_{_k}",
        f"pmin_{_k}", f"pmax_{_k}", f"pstep_{_k}", f"pvary_{_k}",
    ]
_FLAG_CFG_KEYS: list[str] = [f"flag_{k}" for k in FLAG_META]
_MISC_CFG_KEYS: list[str] = ["dc_on_off", "dc_type", "pfixed_dt_max"]
_ALL_CFG_KEYS: list[str] = _PARAM_CFG_KEYS + _FLAG_CFG_KEYS + _MISC_CFG_KEYS


def _get_serializable_state() -> dict:
    """Extract serialisable widget state from session_state."""
    data: dict = {}
    for key in _ALL_CFG_KEYS:
        val = st.session_state.get(key)
        if val is None:
            continue
        if isinstance(val, np.integer):
            val = int(val)
        elif isinstance(val, np.floating):
            val = float(val)
        elif isinstance(val, np.ndarray):
            val = val.tolist()
        elif isinstance(val, list):
            val = [
                int(v) if isinstance(v, np.integer)
                else float(v) if isinstance(v, np.floating)
                else v
                for v in val
            ]
        data[key] = val
    return data


def _apply_state(data: dict) -> None:
    """Write config data to session_state so widgets pick it up on next render."""
    for key, val in data.items():
        st.session_state[key] = val


def _save_config(path: pathlib.Path = _CONFIG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(_get_serializable_state(), fh, indent=2)


def _load_config(path: pathlib.Path = _CONFIG_PATH) -> bool:
    """Load config from *path*.  Returns True on success."""
    try:
        with open(path) as fh:
            data = json.load(fh)
        _apply_state(data)
        return True
    except Exception:
        return False


def _load_defaults() -> None:
    """Reset all widget state to PARAM_META / FLAG_META built-in defaults."""
    for key, meta in PARAM_META.items():
        st.session_state[f"pmode_{key}"] = "Fixed"
        default = meta["default"]
        st.session_state[f"pfixed_{key}"] = default if meta["type"] == "str" else float(default)
    for key, meta in TWIN_META.items():
        st.session_state[f"pmode_{key}"] = "Fixed"
        st.session_state[f"pfixed_{key}"] = float(meta["default"])
    for key, meta in FLAG_META.items():
        st.session_state[f"flag_{key}"] = "True" if meta["default"] else "False"
    st.session_state["dc_on_off"] = "Off"
    st.session_state["dc_type"] = "Twin (symmetric)"


# ── Sweep state ────────────────────────────────────────────────────────────────

@dataclass
class SweepState:
    total: int = 0
    completed: int = 0
    failed: int = 0
    log: list = field(default_factory=list)
    running: bool = True
    done: bool = False
    error: str = ""
    total_time_s: float = 0.0
    db_path: str = ""
    planned_run_ids: list = field(default_factory=list)


# ── Session state helpers ──────────────────────────────────────────────────────

def _ss(key, default=None):
    """Get session state value with a default."""
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


def _set_ss(key, value):
    st.session_state[key] = value


# ── Range computation ─────────────────────────────────────────────────────────

def _arange_inclusive(min_val, max_val, step):
    """np.arange with inclusive upper bound."""
    if step <= 0:
        return np.array([min_val])
    vals = np.arange(min_val, max_val + step * 1e-9, step)
    return vals[vals <= max_val + step * 1e-9]


def _format_vals(vals):
    if len(vals) == 0:
        return "[]"
    def _fv(v):
        if isinstance(v, str):
            return v
        try:
            return f"{v:g}"
        except (TypeError, ValueError):
            return str(v)
    if len(vals) <= 6:
        return "[" + ", ".join(_fv(v) for v in vals) + "]"
    return f"[{_fv(vals[0])}, {_fv(vals[1])}, … {_fv(vals[-1])}]  ({len(vals)} values)"


# ── Widget renderers ───────────────────────────────────────────────────────────

def _num_format(value) -> str:
    """Return a printf format string for a number input.

    Uses scientific notation for values with magnitude >= 1e5 or < 1e-3.
    """
    if value is None or value == 0:
        return "%g"
    v = abs(float(value))
    if v >= 1e5 or v < 1e-3:
        return "%.3e"
    return "%g"


def _render_param_row(key: str, meta: dict) -> None:
    """Render a fixed/range selector for one numeric or string parameter."""
    param_type = meta["type"]
    label = meta["label"]
    unit = meta.get("unit", "")
    default = meta["default"]

    label_disp = f"**{label}**" + (f"  [{unit}]" if unit else "")
    st.markdown(label_disp)

    if param_type == "str":
        choices = meta.get("choices", [])
        mode = st.radio(
            f"##mode_{key}", ["Fixed", "Vary"], horizontal=True,
            label_visibility="collapsed", key=f"pmode_{key}",
        )
        if mode == "Fixed":
            idx_default = choices.index(default) if default in choices else 0
            val = st.selectbox(f"##fix_{key}", choices, index=idx_default,
                               label_visibility="collapsed", key=f"pfixed_{key}")
            _set_ss(f"param_{key}", {"mode": "fixed", "value": val})
        else:
            selected = st.multiselect(f"##vary_{key}", choices, default=choices,
                                      label_visibility="collapsed", key=f"pvary_{key}")
            _set_ss(f"param_{key}", {"mode": "range", "values": selected or choices})
        return

    # Numeric (float / int)
    mode = st.radio(
        f"##mode_{key}", ["Fixed", "Range"], horizontal=True,
        label_visibility="collapsed", key=f"pmode_{key}",
    )
    fmt = _num_format(default)
    if mode == "Fixed":
        val = st.number_input(
            f"##fix_{key}", value=float(default), format=fmt,
            label_visibility="collapsed", key=f"pfixed_{key}",
        )
        _set_ss(f"param_{key}", {"mode": "fixed", "value": val})
    else:
        c1, c2, c3 = st.columns(3)
        min_v = c1.number_input("Min", value=float(default), format=fmt, key=f"pmin_{key}")
        max_v = c2.number_input("Max", value=float(default) * 2, format=fmt, key=f"pmax_{key}")
        step_v = c3.number_input("Step", value=abs(float(default)) or 1.0, format=fmt,
                                 key=f"pstep_{key}", min_value=1e-30)
        vals = _arange_inclusive(min_v, max_v, step_v)
        if param_type == "int":
            vals = np.unique(vals.astype(int))
        st.caption(f"→ {_format_vals(vals)}")
        _set_ss(f"param_{key}", {"mode": "range", "values": vals.tolist()})
    st.divider()


def _render_flag_row(key: str, meta: dict) -> None:
    """Render a True / False / Both radio for one flag."""
    label = meta["label"]
    default = meta["default"]
    default_sel = "True" if default else "False"
    choice = st.radio(
        label, ["False", "True", "Both"], index=["False", "True", "Both"].index(default_sel),
        horizontal=True, key=f"flag_{key}",
    )
    _set_ss(f"flagcfg_{key}", choice)


# ── Sweep config builder ──────────────────────────────────────────────────────

def _build_sweep_config():
    """
    Read session state widgets and build
    (param_ranges, flag_ranges, fixed_params, fixed_flags, param_transforms).

    param_transforms is a callable ``(params, flags) -> params`` applied by the
    sweep engine after building each run's full params dict.  It derives ``Id``
    from the user-facing ``P_in`` (Total Input Power) and ``Vd``, and — in
    symmetric twin mode — splits power and neutral sources equally between the
    two cathodes.
    """
    param_ranges = {}
    fixed_params = {}
    flag_ranges = {}
    fixed_flags = {}

    for key in PARAM_META:
        cfg = st.session_state.get(f"param_{key}")
        ptype = PARAM_META[key]["type"]
        if cfg is None:
            # Widget not yet rendered; use default
            fixed_params[key] = PARAM_META[key]["default"]
            continue
        if cfg["mode"] == "fixed":
            val = cfg["value"]
            fixed_params[key] = int(val) if ptype == "int" else val
        else:
            vals = cfg.get("values", [])
            if len(vals) == 1:
                fixed_params[key] = int(vals[0]) if ptype == "int" else vals[0]
            elif len(vals) > 1:
                param_ranges[key] = [int(v) for v in vals] if ptype == "int" else list(vals)
            else:
                fixed_params[key] = PARAM_META[key]["default"]

    # Dual Cathode
    dc_on_off = st.session_state.get("dc_on_off", "Off")
    dc_type = st.session_state.get("dc_type", "Twin (symmetric)")

    if dc_on_off == "Off":
        fixed_flags["TwinCathode"] = False
    elif dc_on_off == "On":
        fixed_flags["TwinCathode"] = True
    else:  # Both
        flag_ranges["TwinCathode"] = [True, False]

    # Whether twin symmetric splitting applies (captured for the transform closure)
    _is_symmetric = (dc_on_off != "Off") and (dc_type == "Twin (symmetric)")

    if dc_on_off in ("On", "Both"):
        if dc_type != "Twin (symmetric)":  # Asymmetric — independent twin controls
            for key in TWIN_META:
                cfg = st.session_state.get(f"param_{key}")
                if cfg is None:
                    fixed_params[key] = TWIN_META[key]["default"]
                elif cfg["mode"] == "fixed":
                    fixed_params[key] = cfg["value"]
                else:
                    vals = cfg.get("values", [])
                    if len(vals) == 1:
                        fixed_params[key] = vals[0]
                    elif len(vals) > 1:
                        param_ranges[key] = list(vals)
                    else:
                        fixed_params[key] = TWIN_META[key]["default"]

    # When adaptive stepping is on, dt_max overrides dt_main and dt_after
    if st.session_state.get("flagcfg_adaptive") == "True":
        dt_max = st.session_state.get("param_dt_max", {}).get("value", 1e-2)
        fixed_params["dt_main"] = dt_max
        fixed_params["dt_after"] = dt_max
        # Remove stale values that PARAM_META processing may have set
        param_ranges.pop("dt_main", None)
        param_ranges.pop("dt_after", None)

    # Regular flags
    for key in FLAG_META:
        choice = st.session_state.get(f"flagcfg_{key}", "False" if not FLAG_META[key]["default"] else "True")
        if choice == "True":
            fixed_flags[key] = True
        elif choice == "False":
            fixed_flags[key] = False
        else:  # Both
            flag_ranges[key] = [True, False]

    # Build param_transforms closure.
    # Derives Id = P_in / Vd, and in symmetric twin mode splits power + sources equally.
    def _param_transform(params, flags, _sym=_is_symmetric):
        P_in_MW = params.pop("P_in", PARAM_META["P_in"]["default"])
        P_in_W = P_in_MW * 1e6  # convert MW → W
        Vd = params.get("Vd", PARAM_META["Vd"]["default"])
        twin_active = flags.get("TwinCathode", False)

        if twin_active and _sym:
            # Split total power equally; each cathode gets half the current
            Id = P_in_W / (2.0 * Vd)
            params["Id"] = Id
            params["Twin_Id"] = Id
            params["Twin_Vd"] = Vd
            # Split gas puff and neutral source equally between cathodes
            params["S_gp"] = params.get("S_gp", PARAM_META["S_gp"]["default"]) / 2.0
            params["Twin_S_gp"] = params["S_gp"]
            params["Source_nn0"] = params.get("Source_nn0", PARAM_META["Source_nn0"]["default"]) / 2.0
            params["Twin_nn0"] = params["Source_nn0"]
        else:
            # Single cathode or asymmetric twin — primary drives full P_in
            params["Id"] = P_in_W / Vd

        return params

    return param_ranges, flag_ranges, fixed_params, fixed_flags, _param_transform


def _count_combos(param_ranges, flag_ranges):
    n = 1
    for vals in param_ranges.values():
        n *= len(vals)
    for vals in flag_ranges.values():
        n *= len(vals)
    return n


def _describe_sweep(param_ranges, flag_ranges):
    parts = []
    for k, vals in sorted(param_ranges.items()):
        parts.append(f"{k} ∈ {_format_vals(vals)}")
    for k, vals in sorted(flag_ranges.items()):
        parts.append(f"{k} ∈ {vals}")
    return ";  ".join(parts) if parts else "No varied parameters"


def _fmt_val(v) -> str:
    """Format a parameter value, using scientific notation for large/small floats."""
    if isinstance(v, str):
        return v
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    fv = float(v)
    av = abs(fv)
    if av == 0:
        return "0"
    if av >= 1e5 or (0 < av < 1e-2):
        return f"{fv:.3e}"
    return f"{fv:g}"


# ── Sweep thread ──────────────────────────────────────────────────────────────

def _drain_queue():
    """Pull all pending messages from the sweep queue into sweep_state."""
    q: queue.Queue = st.session_state.get("sweep_queue")
    state: SweepState = st.session_state.get("sweep_state")
    if q is None or state is None:
        return
    changed = False
    while True:
        try:
            msg = q.get_nowait()
        except queue.Empty:
            break
        changed = True
        if "done" in msg:
            state.running = False
            state.done = True
            state.total_time_s = msg.get("total_time_s", 0.0)
            st.session_state["sweep_running"] = False
            if state.db_path:
                _save_manifest(state.db_path, {
                    "running": False,
                    "db_path": state.db_path,
                    "total": state.total,
                    "completed": state.completed,
                    "failed": state.failed,
                    "planned_run_ids": state.planned_run_ids,
                    "log": state.log[-50:],
                })
        elif "error" in msg:
            state.error = msg["error"]
            state.running = False
            state.done = True
            st.session_state["sweep_running"] = False
            if state.db_path:
                _save_manifest(state.db_path, {
                    "running": False,
                    "db_path": state.db_path,
                    "total": state.total,
                    "completed": state.completed,
                    "failed": state.failed,
                    "planned_run_ids": state.planned_run_ids,
                    "error": state.error,
                    "log": state.log[-50:],
                })
        else:
            state.completed = msg.get("i", state.completed)
            state.total = msg.get("total", state.total)
            run_id = msg.get("run_id", "")
            status = msg.get("status", "ok")
            if status == "failed":
                state.failed += 1
            stats = msg.get("stats", {})
            ne_var = stats.get("ne_var", float("nan"))
            run_time = stats.get("_run_time_s")
            equil_time = stats.get("_equil_time_s")
            equil_cache_hit = stats.get("_equil_cache_hit", False)
            equil_S_gp = stats.get("_equil_S_gp")
            equil_twin = stats.get("_equil_twin", False)
            equil_Twin_S_gp = stats.get("_equil_Twin_S_gp")
            do_equil = stats.get("_equilibrate_nn", False)

            line = f"[{state.completed}/{state.total}] {run_id} {status}"
            if not np.isnan(ne_var):
                line += f"  ne_var={ne_var:.3e}"
            if run_time is not None:
                line += f"  run={run_time:.1f}s"
            if do_equil and equil_S_gp is not None:
                twin_str = f"/twin={equil_Twin_S_gp:.0f}" if equil_twin else ""
                if equil_cache_hit:
                    line += f"  equil=cached(S_gp={equil_S_gp:.0f}{twin_str})"
                else:
                    line += f"  equil={equil_time:.1f}s(fresh,S_gp={equil_S_gp:.0f}{twin_str})"
            state.log.append(line)

    # Persist progress to manifest every time something changed
    if changed and state.running and state.db_path:
        _save_manifest(state.db_path, {
            "running": True,
            "db_path": state.db_path,
            "total": state.total,
            "completed": state.completed,
            "failed": state.failed,
            "planned_run_ids": state.planned_run_ids,
            "log": state.log[-50:],
        })


def _start_sweep_thread(db_path, n_workers, t_window, param_ranges, flag_ranges,
                        fixed_params, fixed_flags, param_transforms=None, equilibrate_nn=False):
    q: queue.Queue = queue.Queue()
    stop_event = threading.Event()

    # Compute all planned run IDs (for reconnect / abort marking)
    all_combos = param_combinations(param_ranges, flag_ranges)
    n_total = len(all_combos)
    planned_ids = [f"run_{i:04d}" for i in range(n_total)]

    state = SweepState(
        total=n_total,
        db_path=db_path,
        planned_run_ids=planned_ids,
    )

    # Persist the initial manifest so reconnect works immediately
    _save_manifest(db_path, {
        "running": True,
        "db_path": db_path,
        "total": n_total,
        "completed": 0,
        "failed": 0,
        "planned_run_ids": planned_ids,
        "log": [],
    })

    def progress_cb(i, total, run_id, status, stats):
        q.put({"i": i, "total": total, "run_id": run_id, "status": status, "stats": stats})

    def target():
        t_start = time.time()
        try:
            grid_sweep_parallel(
                param_ranges=param_ranges,
                flag_ranges=flag_ranges,
                fixed_params=fixed_params,
                fixed_flags=fixed_flags,
                db_path=db_path,
                t_window=t_window,
                n_workers=n_workers,
                progress_callback=progress_cb,
                param_transforms=param_transforms,
                equilibrate_nn=equilibrate_nn,
                verbose=False,
                verbose_equil=False,
                stop_event=stop_event,
            )
        except Exception as exc:
            q.put({"error": str(exc)})
        finally:
            q.put({"done": True, "total_time_s": time.time() - t_start})

    thread = threading.Thread(target=target, daemon=True)
    st.session_state["sweep_thread"] = thread
    st.session_state["sweep_queue"] = q
    st.session_state["sweep_state"] = state
    st.session_state["sweep_running"] = True
    st.session_state["sweep_stop_event"] = stop_event
    thread.start()


# ── Index → DataFrame ─────────────────────────────────────────────────────────

def _index_to_df(idx):
    import pandas as pd

    rows = []
    n = len(idx["run_ids"])
    for i in range(n):
        row = {
            "run_id": idx["run_ids"][i],
            "status": idx["status"][i],
            "n_cells": int(idx["n_cells"][i]) if i < len(idx["n_cells"]) else None,
        }
        for k, arr in idx["params"].items():
            row[f"p:{k}"] = arr[i] if i < len(arr) else None
        for k, arr in idx["flags"].items():
            row[f"f:{k}"] = bool(arr[i]) if i < len(arr) else None
        for k, arr in idx["stats_10_20ms"].items():
            row[f"s:{k}"] = float(arr[i]) if i < len(arr) else None
        # Derived: total input power in MW = (Id + Twin_Id) * Vd / 1e6
        def _f0(v):
            """Return float, treating None and NaN as 0."""
            try:
                f = float(v)
                return 0.0 if np.isnan(f) else f
            except (TypeError, ValueError):
                return 0.0
        Id = _f0(row.get("p:Id"))
        Twin_Id = _f0(row.get("p:Twin_Id"))
        Vd = _f0(row.get("p:Vd"))
        row["P_total_MW"] = (Id + Twin_Id) * Vd / 1e6
        rows.append(row)
    return pd.DataFrame(rows)


# ── Tab renderers ─────────────────────────────────────────────────────────────

def _render_configure_tab():
    st.header("Configure Parameter Sweep")

    # Config toolbar
    c_save, c_load, c_defaults, _ = st.columns([1, 1, 1, 5])
    if c_save.button("💾 Save Config", help=f"Save current configuration to {_CONFIG_PATH}"):
        _save_config()
        st.toast(f"Config saved to {_CONFIG_PATH}")
    if c_load.button("📂 Load Config", help=f"Reload configuration from {_CONFIG_PATH}"):
        if _CONFIG_PATH.exists():
            if _load_config():
                st.rerun()
        else:
            st.warning(f"No saved config found at {_CONFIG_PATH}")
    if c_defaults.button("🔄 Load Defaults", help="Reset all parameters to built-in defaults"):
        _load_defaults()
        st.rerun()

    col_params, col_flags = st.columns([3, 2])

    # Read adaptive flag state persisted from previous render
    _adaptive_on = st.session_state.get("flagcfg_adaptive", "False") == "True"

    with col_params:
        for group in PARAM_GROUP_ORDER:
            expanded = group in ("Discharge (Primary Cathode)", "Gas & Initial Conditions")
            with st.expander(group, expanded=expanded):
                for key, meta in PARAM_META.items():
                    if meta["group"] != group:
                        continue
                    # When adaptive stepping is active, dt_main and dt_after are
                    # replaced by a single dt_max input rendered below
                    if _adaptive_on and key in ("dt_main", "dt_after"):
                        continue
                    _render_param_row(key, meta)

                if group == "Time & Solver" and _adaptive_on:
                    st.markdown("**Max adaptive step (dt_max)**  [s]")
                    dt_max_val = st.number_input(
                        "##dt_max", value=1e-2, format="%.3e",
                        label_visibility="collapsed", key="pfixed_dt_max",
                        min_value=1e-30,
                    )
                    _set_ss("param_dt_max", {"mode": "fixed", "value": dt_max_val})
                    st.divider()

        # Dual Cathode section
        with st.expander("Dual Cathode", expanded=False):
            dc_on_off = st.radio(
                "Dual Cathode",
                ["Off", "On", "Both"],
                horizontal=True,
                key="dc_on_off",
                help=(
                    "**Off**: single cathode only.  "
                    "**On**: second cathode active.  "
                    "**Both**: sweep over single and dual cathode configurations."
                ),
            )
            if dc_on_off in ("On", "Both"):
                dc_type = st.radio(
                    "Second cathode type",
                    ["Twin (symmetric)", "Asymmetric"],
                    horizontal=True,
                    key="dc_type",
                    help=(
                        "**Twin**: total power (P_in) and sources (S_gp, Source_nn0) are split "
                        "equally between cathodes; Vd is shared.  "
                        "**Asymmetric**: second cathode has fully independent controls."
                    ),
                )
                if dc_type == "Twin (symmetric)":
                    # Show live splitting values
                    vd_mode = st.session_state.get("pmode_Vd", "Fixed")
                    p_in_mode = st.session_state.get("pmode_P_in", "Fixed")
                    s_gp_mode = st.session_state.get("pmode_S_gp", "Fixed")
                    snn0_mode = st.session_state.get("pmode_Source_nn0", "Fixed")

                    vd = float(st.session_state.get("pfixed_Vd", PARAM_META["Vd"]["default"]))
                    p_in = float(st.session_state.get("pfixed_P_in", PARAM_META["P_in"]["default"]))
                    s_gp = float(st.session_state.get("pfixed_S_gp", PARAM_META["S_gp"]["default"]))
                    s_nn0 = float(st.session_state.get("pfixed_Source_nn0", PARAM_META["Source_nn0"]["default"]))

                    lines = []
                    # Twin_Vd = Vd (unchanged)
                    if vd_mode == "Range":
                        vd_min = st.session_state.get("pmin_Vd", vd)
                        vd_max = st.session_state.get("pmax_Vd", vd)
                        lines.append(f"- **Twin_Vd** = **Vd**  *(range {vd_min:g} → {vd_max:g} V)*")
                    else:
                        lines.append(f"- **Twin_Vd** = {vd:g} V  *(= Vd)*")
                    # Id = Twin_Id = P_in / (2 × Vd)
                    if p_in_mode == "Range" or vd_mode == "Range":
                        lines.append("- **Id** = **Twin_Id** = P_in / (2×Vd)  *(computed per combination)*")
                    else:
                        id_split = p_in * 1e6 / (2.0 * vd)
                        lines.append(
                            f"- **Id** = **Twin_Id** = {_fmt_val(id_split)} A"
                            f"  *(= {p_in:.3f} MW / (2×{vd:g} V))*"
                        )
                    # S_gp split
                    if s_gp_mode == "Range":
                        sg_min = st.session_state.get("pmin_S_gp", s_gp)
                        sg_max = st.session_state.get("pmax_S_gp", s_gp)
                        lines.append(
                            f"- **S_gp** = **Twin_S_gp** = S_gp/2"
                            f"  *(range {sg_min/2:g} → {sg_max/2:g} per cathode)*"
                        )
                    else:
                        lines.append(
                            f"- **S_gp** = **Twin_S_gp** = {_fmt_val(s_gp/2)}"
                            f"  *(= {_fmt_val(s_gp)}/2 per cathode)*"
                        )
                    # Source_nn0 split
                    if snn0_mode == "Range":
                        sn_min = st.session_state.get("pmin_Source_nn0", s_nn0)
                        sn_max = st.session_state.get("pmax_Source_nn0", s_nn0)
                        lines.append(
                            f"- **Source_nn0** = **Twin_nn0** = Source_nn0/2"
                            f"  *(range {sn_min/2:.3e} → {sn_max/2:.3e} per cathode)*"
                        )
                    else:
                        lines.append(
                            f"- **Source_nn0** = **Twin_nn0** = {_fmt_val(s_nn0/2)}"
                            f"  *(= {_fmt_val(s_nn0)}/2 per cathode)*"
                        )
                    st.info(
                        "Twin mode — power and sources split equally between cathodes:\n"
                        + "\n".join(lines)
                    )
                else:
                    st.markdown("**Second cathode parameters:**")
                    for key, meta in TWIN_META.items():
                        _render_param_row(key, meta)

    with col_flags:
        for group in FLAG_GROUP_ORDER:
            with st.expander(group + " Flags", expanded=True):
                for key, meta in FLAG_META.items():
                    if meta["group"] == group:
                        _render_flag_row(key, meta)

    # Run count + parameter summary
    param_ranges, flag_ranges, fixed_params, fixed_flags, _transforms = _build_sweep_config()
    n_combos = _count_combos(param_ranges, flag_ranges)
    st.divider()
    col1, col2 = st.columns([1, 3])
    col1.metric("Total runs", n_combos)
    if n_combos > 0:
        col2.markdown(_describe_sweep(param_ranges, flag_ranges))

    with st.expander("Fixed Parameter Summary", expanded=True):
        import pandas as pd

        col_p, col_f = st.columns([3, 2])

        with col_p:
            st.markdown("**Fixed Parameters**")
            rows = []
            for group in PARAM_GROUP_ORDER:
                for k, v in fixed_params.items():
                    if k in PARAM_META and PARAM_META[k]["group"] == group:
                        meta = PARAM_META[k]
                        rows.append({
                            "Group": group,
                            "Parameter": meta["label"],
                            "Value": _fmt_val(v),
                            "Unit": meta.get("unit", ""),
                        })
            for k, v in fixed_params.items():
                if k in TWIN_META:
                    meta = TWIN_META[k]
                    rows.append({
                        "Group": "Dual Cathode",
                        "Parameter": meta["label"],
                        "Value": _fmt_val(v),
                        "Unit": meta.get("unit", ""),
                    })
            if rows:
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        with col_f:
            st.markdown("**Fixed Flags**")
            flag_rows = []
            for k, v in sorted(fixed_flags.items()):
                label = FLAG_META[k]["label"] if k in FLAG_META else k
                flag_rows.append({"Flag": label, "Value": str(v)})
            if flag_rows:
                st.dataframe(pd.DataFrame(flag_rows), width="stretch", hide_index=True)


def _render_run_tab():
    st.header("Run Parameter Sweep")

    db_path = _render_path_input("run_db", "Database path", "~/lapd_data/sweep.h5")
    db_path_exp = os.path.expanduser(db_path)

    col1, col2, _col3 = st.columns(3)
    max_workers = psutil.cpu_count(logical=True) or 4
    n_workers = col1.number_input("Workers", min_value=1, max_value=max_workers,
                                  value=1, key="run_n_workers")
    with col2:
        t_start = st.number_input("t_window start (ms)", min_value=0.0, value=10.0,
                                  key="run_t_start")
        t_end = st.number_input("t_window end (ms)", min_value=0.0, value=20.0,
                                key="run_t_end")

    # ── Reconnect / interrupt banner ──────────────────────────────────────────
    already_running = st.session_state.get("sweep_running", False)
    if not already_running:
        manifest = _load_manifest(db_path_exp)
        if manifest and manifest.get("running"):
            with st.container(border=True):
                st.warning(
                    f"⚠ A previous sweep was interrupted.  "
                    f"{manifest.get('completed', '?')}/{manifest.get('total', '?')} run(s) completed, "
                    f"{manifest.get('failed', 0)} failed."
                )
                c1, c2, c3 = st.columns(3)
                if c1.button("▶ Resume Sweep", help="Re-launch sweep — completed runs are automatically skipped"):
                    # Dismiss the banner then fall through to the normal launch flow below
                    manifest["running"] = False
                    _save_manifest(db_path_exp, manifest)
                    st.session_state["_resume_sweep"] = True
                    st.rerun()
                if c2.button("❌ Mark incomplete as failed",
                             help="Write 'failed' records for planned runs not yet in the database"):
                    _mark_incomplete_as_failed(db_path_exp, manifest)
                    manifest["running"] = False
                    _save_manifest(db_path_exp, manifest)
                if c3.button("✕ Dismiss", help="Hide this banner without taking action"):
                    manifest["running"] = False
                    _save_manifest(db_path_exp, manifest)
                    st.rerun()
                # Show last log lines
                last_log = manifest.get("log", [])[-5:]
                if last_log:
                    st.caption("Last log entries: " + " | ".join(last_log))

    if n_workers > 1:
        st.info(
            f"Parallel mode: {n_workers} workers.  "
            "Simulations run in separate processes; HDF5 writes are serialised on the main thread."
        )

    equilibrate_nn = st.checkbox(
        "Auto-equilibrate nn0",
        value=False,
        key="run_equilibrate_nn",
        help=(
            "Before each plasma-on run, automatically determine the equilibrium background "
            "neutral density by running 100 plasma-off cycles (3 s each, S_gp active first "
            "20 ms per cycle, adaptive dt ≤ 10 ms) starting from nn0 = 1×10⁸ cm⁻³.  "
            "Only nn0 (interior cells mean) is updated from the result.  "
            "Source_nn0 and Twin_nn0 are left as configured."
        ),
    )

    param_ranges, flag_ranges, fixed_params, fixed_flags, param_transforms = _build_sweep_config()
    n_combos = _count_combos(param_ranges, flag_ranges)
    st.markdown(f"**Ready to run {n_combos} combination(s).**  {_describe_sweep(param_ranges, flag_ranges)}")

    # ── Launch / Abort buttons ─────────────────────────────────────────────────
    btn_col1, btn_col2, _ = st.columns([2, 2, 6])
    launch = btn_col1.button("Launch Sweep", type="primary", disabled=already_running)
    abort_clicked = btn_col2.button(
        "⏹ Abort Sweep",
        type="secondary",
        disabled=not already_running,
        help="Signal the sweep to stop after finishing the current runs. Already-running worker processes will complete naturally.",
    )

    if abort_clicked and already_running:
        stop_ev = st.session_state.get("sweep_stop_event")
        if stop_ev is not None:
            stop_ev.set()
        st.toast("Abort signal sent — sweep will stop after current runs complete.")

    if st.session_state.pop("_resume_sweep", False):
        # User clicked "Resume Sweep" in the reconnect banner — treat like a normal launch
        launch = True

    if launch and not already_running:
        if n_combos == 0:
            st.warning("No parameter combinations — adjust configuration on the Configure tab.")
        else:
            # Clear previous sweep state so the "Sweep complete" banner resets
            st.session_state.pop("sweep_state", None)
            st.session_state.pop("sweep_queue", None)
            st.session_state.pop("sweep_stop_event", None)
            _start_sweep_thread(
                db_path=db_path_exp,
                n_workers=int(n_workers),
                t_window=(float(t_start), float(t_end)),
                param_ranges=param_ranges,
                flag_ranges=flag_ranges,
                fixed_params=fixed_params,
                fixed_flags=fixed_flags,
                param_transforms=param_transforms,
                equilibrate_nn=equilibrate_nn,
            )
            st.rerun()

    # ── Tab-close warning (beforeunload) when sweep is active ─────────────────
    if already_running:
        st.markdown(
            """
            <script>
            (function () {
                window._lapd_sweep_running = true;
                function _lapd_warn(e) {
                    if (window._lapd_sweep_running) {
                        e.preventDefault();
                        e.returnValue = '';
                        return '';
                    }
                }
                if (!window._lapd_warn_attached) {
                    window.addEventListener('beforeunload', _lapd_warn);
                    window._lapd_warn_attached = true;
                }
            })();
            </script>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Clear the flag so closing the tab no longer triggers the dialog
        st.markdown(
            "<script>window._lapd_sweep_running = false;</script>",
            unsafe_allow_html=True,
        )

    # ── Progress display ───────────────────────────────────────────────────────
    if "sweep_state" in st.session_state:
        _drain_queue()
        state: SweepState = st.session_state["sweep_state"]

        st.divider()

        progress_frac = state.completed / max(state.total, 1)
        st.progress(
            progress_frac,
            text=f"{state.completed}/{state.total} runs complete"
            + (f"  ({state.failed} failed)" if state.failed else ""),
        )

        # Memory / CPU monitor
        proc = psutil.Process()
        proc_ram = proc.memory_info().rss / 1e9
        sys_vm = psutil.virtual_memory()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Process RAM", f"{proc_ram:.2f} GB")
        m2.metric("System available", f"{sys_vm.available / 1e9:.2f} GB")
        m3.metric("System RAM used", f"{sys_vm.percent:.0f}%")
        m4.metric("CPU", f"{psutil.cpu_percent(interval=None):.0f}%")

        # Log
        log_text = "\n".join(state.log[-30:]) if state.log else "(waiting for first run…)"
        st.text_area("Run log (last 30)", log_text, height=200)

        if state.running:
            time.sleep(0.5)
            st.rerun()
        elif state.done:
            time_str = f"  Total: {state.total_time_s:.1f}s" if state.total_time_s > 0 else ""
            if state.error:
                st.error(f"Sweep failed: {state.error}")
            elif state.failed == 0:
                st.success(f"Sweep complete: {state.completed}/{state.total} runs succeeded.{time_str}")
            else:
                st.warning(
                    f"Sweep done: {state.completed - state.failed} succeeded, "
                    f"{state.failed} failed.{time_str}"
                )


def _render_explore_tab():
    st.header("Explore Database")
    db_path = _render_path_input("explore_db", "Database path", "~/lapd_data/sweep.h5")
    db_path = os.path.expanduser(db_path)
    if not os.path.exists(db_path):
        st.warning(f"File not found: `{db_path}`")
        return

    col_rebuild, col_info = st.columns([1, 4])
    if col_rebuild.button("Rebuild Index", help="Recompute the index from raw run data. Use this after partial failures or if variance plots show mismatched array sizes."):
        try:
            with open_db(db_path, mode="r+") as db:
                rebuild_index(db)
            col_info.success("Index rebuilt successfully.")
        except Exception as exc:
            col_info.error(f"Rebuild failed: {exc}")

    with open_db(db_path) as db:
        idx = load_index(db)
        all_run_ids = list_runs(db)

    if not all_run_ids:
        st.warning("Database contains no runs.")
        return

    ok_runs = [r for r, s in zip(idx["run_ids"], idx["status"]) if s == "ok"]

    # Build rich display labels for run-selection dropdowns
    def _run_display_labels(idx):
        labels = {}
        p = idx["params"]
        f = idx["flags"]
        for i, run_id in enumerate(idx["run_ids"]):
            def _p(key, default=0.0, _i=i):
                arr = p.get(key)
                if arr is None or _i >= len(arr):
                    return default
                v = arr[_i]
                try:
                    return default if np.isnan(float(v)) else v
                except (TypeError, ValueError):
                    return v if v else default
            def _f(key, default=False, _i=i):
                arr = f.get(key)
                return bool(arr[_i]) if arr is not None and _i < len(arr) else default

            twin = _f("TwinCathode")
            Id = _p("Id")
            Vd = _p("Vd")
            Twin_Id = _p("Twin_Id") if twin else 0.0
            S_gp = _p("S_gp")
            Twin_S_gp = _p("Twin_S_gp") if twin else 0.0
            gas = _p("gas_type", "?")
            if isinstance(gas, bytes):
                gas = gas.decode()
            P_MW = (Id + Twin_Id) * Vd / 1e6
            S_gp_total = S_gp + Twin_S_gp
            twin_str = "twin" if twin else "single"
            labels[run_id] = (
                f"{run_id}  |  {gas}  P={P_MW:.2f} MW  "
                f"S_gp={S_gp_total:.0f}  [{twin_str}]"
            )
        return labels
    run_labels = _run_display_labels(idx)

    sub_tabs = st.tabs(["📋 Table", "📊 Variance", "🔬 Inspector", "⚖️ Comparison"])

    # ── Table ─────────────────────────────────────────────────────────────────
    with sub_tabs[0]:
        st.subheader("Run Index")
        df = _index_to_df(idx)
        col_conf = {
            col: st.column_config.NumberColumn(format="%.3e")
            for col in df.columns
            if col.startswith(("p:", "s:"))
        }
        col_conf["P_total_MW"] = st.column_config.NumberColumn(
            "P_total [MW]", format="%.3f"
        )
        # Default visible columns; all columns still available in CSV export
        _DEFAULT_PARAMS = ["Vd", "Twin_Vd", "Id", "Twin_Id", "gas_type",
                           "nn0", "Source_nn0", "Twin_nn0"]
        default_cols = ["run_id", "status", "n_cells", "P_total_MW"]
        default_cols += [f"p:{k}" for k in _DEFAULT_PARAMS if f"p:{k}" in df.columns]
        default_cols += [c for c in df.columns if c.startswith("f:")]
        st.dataframe(df, width="stretch", height=500, column_config=col_conf,
                     column_order=default_cols)
        csv = df.to_csv(index=False).encode()
        st.download_button("Export CSV", csv, "run_index.csv", mime="text/csv")

    # ── Variance analysis ─────────────────────────────────────────────────────
    with sub_tabs[1]:
        st.subheader("Variance / Uniformity Analysis")

        if not ok_runs:
            st.warning("No successful runs in database.")
        else:
            ok_mask = np.array(idx["status"]) == "ok"
            param_keys = list(idx["params"].keys())
            flag_keys = list(idx["flags"].keys())

            # ── Filter controls (gas_type and TwinCathode) ────────────────────
            # These are used only as filters, not as x/hue axis options.
            var_filter_mask = ok_mask.copy()

            _gas_vals_all = np.asarray(idx["params"].get("gas_type", []))[ok_mask]
            _unique_gases = sorted(set(str(v) for v in _gas_vals_all)) if len(_gas_vals_all) else []
            _tc_vals_all = np.asarray(idx["flags"].get("TwinCathode", []))[ok_mask]
            _unique_tc = sorted(set(bool(v) for v in _tc_vals_all)) if len(_tc_vals_all) else []

            filt_cols = st.columns(2)
            if len(_unique_gases) > 1:
                sel_gas = filt_cols[0].multiselect(
                    "Filter gas type", _unique_gases, default=_unique_gases, key="var_gas_filter"
                )
                gas_arr = np.asarray(idx["params"].get("gas_type", []))
                if len(gas_arr) == len(ok_mask):
                    var_filter_mask &= np.array([str(v) in sel_gas for v in gas_arr])
            if len(_unique_tc) > 1:
                tc_labels = {True: "Twin", False: "Single"}
                sel_tc_label = filt_cols[1].radio(
                    "Filter cathode mode", ["All", "Single", "Twin"],
                    horizontal=True, key="var_tc_filter"
                )
                tc_arr = np.asarray(idx["flags"].get("TwinCathode", []))
                if len(tc_arr) == len(ok_mask) and sel_tc_label != "All":
                    sel_tc_bool = sel_tc_label == "Twin"
                    var_filter_mask &= np.array([bool(v) == sel_tc_bool for v in tc_arr])

            # Only keep keys where the value actually varies across filtered ok runs
            def _is_varied(arr, mask=var_filter_mask):
                vals = np.asarray(arr)[mask]
                if vals.dtype.kind in ("O", "U", "S"):
                    return len(set(str(v) for v in vals)) > 1
                try:
                    fv = vals.astype(float)
                    return len(np.unique(fv[~np.isnan(fv)])) > 1 if vals.size else False
                except (ValueError, TypeError):
                    return False

            # Exclude gas_type and TwinCathode — they are filter-only
            _filter_only = {"gas_type", "TwinCathode"}
            varied_param_keys = [
                k for k in param_keys
                if k not in _filter_only and _is_varied(idx["params"][k])
            ]
            varied_flag_keys = [
                k for k in flag_keys
                if k not in _filter_only and _is_varied(idx["flags"][k])
            ]
            all_keys = varied_param_keys + varied_flag_keys

            # Build filtered index for plot functions
            def _apply_mask(idx_src, mask):
                return {
                    "run_ids": [r for r, m in zip(idx_src["run_ids"], mask) if m],
                    "status": [s for s, m in zip(idx_src["status"], mask) if m],
                    "params": {k: np.asarray(v)[mask] for k, v in idx_src["params"].items()},
                    "flags":  {k: np.asarray(v)[mask] for k, v in idx_src["flags"].items()},
                    "stats_10_20ms": {
                        k: np.asarray(v)[mask] for k, v in idx_src["stats_10_20ms"].items()
                    },
                }

            filtered_idx = _apply_mask(idx, var_filter_mask)

            if not all_keys:
                st.info("No numeric varied parameters in the current selection.")
            else:
                c1, c2, c3, c4 = st.columns(4)
                x_param = c1.selectbox("X axis", all_keys, key="var_x")
                hue_opts = ["None"] + all_keys
                hue_param = c2.selectbox("Color by", hue_opts, key="var_hue")
                quantity = c3.radio("Quantity", ["ne", "Te"], key="var_qty", horizontal=True)
                plot_type = c4.radio("Plot type", ["Scatter", "Heatmap"], key="var_type", horizontal=True)

                try:
                    if plot_type == "Scatter":
                        fig = plot_sweep_variance(
                            filtered_idx, x_param,
                            hue_param=None if hue_param == "None" else hue_param,
                            quantity=quantity,
                        )
                        st.pyplot(fig, width="stretch")
                        plt.close(fig)
                    else:
                        y_param_opts = [k for k in all_keys if k != x_param]
                        if not y_param_opts:
                            st.info("Need at least 2 varied parameters for a heatmap.")
                        else:
                            y_param = st.selectbox("Y axis (heatmap)", y_param_opts, key="var_y")
                            qty_key = f"{quantity}_var"
                            fig = plot_sweep_heatmap(filtered_idx, x_param, y_param, quantity=qty_key)
                            st.pyplot(fig, width="stretch")
                            plt.close(fig)
                except Exception as exc:
                    st.error(f"Plot error: {exc}")

    # ── Inspector ─────────────────────────────────────────────────────────────
    with sub_tabs[2]:
        st.subheader("Single Run Inspector")

        if not ok_runs:
            st.warning("No successful runs to inspect.")
        else:
            col1, col2 = st.columns([2, 1])
            run_id = col1.selectbox("Select run", ok_runs, key="inspect_run",
                                    format_func=lambda r: run_labels.get(r, r))
            z_conv = col2.radio("Z convention", ["sim", "exp"], horizontal=True, key="inspect_z")

            try:
                with open_db(db_path) as db:
                    params, flags, results = load_run(db, run_id)

                figs = plot_run(results, params, flags, z_convention=z_conv)
                fig_name = st.selectbox("Figure", list(figs.keys()), key="inspect_fig")

                st.pyplot(figs[fig_name], width="stretch")
                for f in figs.values():
                    plt.close(f)

                with st.expander("Run parameters"):
                    col_p, col_f = st.columns(2)
                    with col_p:
                        st.markdown("**Parameters**")
                        for k, v in sorted(params.items()):
                            st.write(f"- `{k}` = `{_fmt_val(v)}`")
                    with col_f:
                        st.markdown("**Flags**")
                        for k, v in sorted(flags.items()):
                            st.write(f"- `{k}` = `{v}`")

            except Exception as exc:
                st.error(f"Error loading run: {exc}")

    # ── Comparison ────────────────────────────────────────────────────────────
    with sub_tabs[3]:
        st.subheader("Side-by-Side Run Comparison")

        if len(ok_runs) < 2:
            st.info("Need at least 2 successful runs in the database to compare.")
        else:
            col1, col2, col3 = st.columns(3)
            selected = col1.multiselect(
                "Select runs (2–4)", ok_runs,
                default=ok_runs[:min(2, len(ok_runs))],
                max_selections=4,
                key="compare_runs",
                format_func=lambda r: run_labels.get(r, r),
            )
            _COMPARE_QUANTITIES = ["ne", "nn", "Te", "Ti", "v_plasma", "isat", "ln_lambda",
                                   "primary_mfp", "bulk_mfp"]
            quantity = col2.selectbox("Quantity", _COMPARE_QUANTITIES, key="compare_qty")

            # Derive cell count limit from selected runs
            run_id_to_n_cells = dict(zip(idx["run_ids"], idx["n_cells"].tolist()))
            max_cells = max((int(run_id_to_n_cells.get(r, 1)) for r in selected), default=1) if selected else 1
            cell_idx = col3.number_input(
                "Cell index (−1 = all cells)", min_value=-1, max_value=max_cells - 1, value=-1,
                key="compare_cell",
            )

            if len(selected) >= 2:
                try:
                    fig = plot_run_comparison(db_path, selected, quantity, int(cell_idx))
                    st.pyplot(fig, width="stretch")
                    plt.close(fig)
                except Exception as exc:
                    st.error(f"Error generating comparison: {exc}")
            else:
                st.info("Select at least 2 runs above.")


# ── App entry ─────────────────────────────────────────────────────────────────

def main():
    try:
        import setproctitle
        setproctitle.setproctitle("lapd-app")
    except ImportError:
        pass

    st.set_page_config(
        page_title="LAPDSim Explorer",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Auto-load saved config once per session (before any widgets are rendered)
    if not st.session_state.get("_config_auto_loaded"):
        st.session_state["_config_auto_loaded"] = True
        if _CONFIG_PATH.exists():
            _load_config()

    st.title("⚡ LAPDSim Parameter Explorer")
    st.caption(
        "Configure a parameter sweep, launch it (optionally in parallel), "
        "then explore the results database."
    )

    tabs = st.tabs(["⚙️ Configure", "▶ Run", "🔍 Explore"])
    with tabs[0]:
        _render_configure_tab()
    with tabs[1]:
        _render_run_tab()
    with tabs[2]:
        _render_explore_tab()


if __name__ == "__main__":
    main()
