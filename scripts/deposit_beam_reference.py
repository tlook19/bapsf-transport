"""Capture/verify the ``deposit_beam`` REFERENCE CORPUS.

What this is
------------
``cablp.funcs._beam_deposition.deposit_beam`` is the CSDA beam-deposition hot
path. This script pins its answers: a committed fixture of argument tuples
paired with the outputs the implementation produces for them, compared at raw
uint64 and never with a tolerance. It is the ``interp_fused_reference`` pattern
applied to a whole function instead of one lerp -- a fixture that is just data,
checkable anywhere, and independent of which implementation is standing behind
the name.

The corpus has two halves:

* **REAL** entries -- argument tuples recorded IN-RUN from the golden fixture's
  own configuration (``baseline_sim1d``), one per sampled time across every
  runtime phase the run visits, in both call classes the solver makes
  (``deposition`` rays and gap-transmission ``probe`` rays), plus widest-ray
  and widest-tail champions. Recording the calls as they happen is exact;
  reconstructing them from a saved HDF5 would mean re-deriving the cathode
  solve outside the solver.
* **SYNTHETIC** entries -- an adversarial battery: the closure-family cross
  product, threshold-straddling walker energies, vacuum cells, sub-threshold
  and end-of-range landings, single-cell walk windows, reflecting-face bounce
  chains, every multi-group plateau rung, both ray directions, and the
  anode-interception corners.

Every entry additionally records, alongside the outputs:

* the measured CONSERVATION closure of the call (the per-ray energy identity,
  and where a tail is walked the tail-bank identity), as a relative residual;
* SUBSTEP and LEG counts -- the CSDA substep iterations at the top level and
  inside the recursive tail-walk legs, the number of those legs, and the
  number of closed-form product-walk integrations;
* the BRANCH/PATH selections that are cheaply readable from the result (cells
  reached, free-streamed cells, absorption, anode interception, the derived
  plateau rung energies).

Reference outputs are computed by the PURE implementation. The builder and the
verifier both REFUSE to run with the compiled CSDA march bound, because the
pure path is the reference implementation and a fixture captured through a
kernel would pin the kernel instead.

Usage::

    # 1. record real argument tuples from a full golden-config run
    python scripts/deposit_beam_reference.py --capture \\
        --output scripts/deposit_beam_corpus_raw.npz

    # 2. build the committed fixture (pure path only)
    python scripts/deposit_beam_reference.py --build \\
        --captured scripts/deposit_beam_corpus_raw.npz

    # 3. anywhere, any time
    python scripts/deposit_beam_reference.py --verify

``--capture`` may be run with ``CABLP_COMPILED_KERNELS=1``: it records the
solver's ARGUMENTS, and the compiled march is bit-exact against pure, so the
trajectory it samples is the same one. ``--build`` and ``--verify`` refuse the
opt-in.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_FIXTURE = SCRIPT_DIR / "data" / "deposit_beam_reference.npz"
FIXTURE_FORMAT = "deposit-beam-reference-v1"

_ERG_PER_EV = 1.602176634e-12

# ``deposit_beam`` keyword arguments that carry arrays; stored as their own
# fixture arrays. Everything else is JSON-serializable and rides the manifest.
ARRAY_KWARGS = (
    "nn",
    "ne",
    "Te",
    "dz_cm",
    "beam_area_cm2",
    "stopping_coefficient",
)

# ``BeamDepositionResult`` fields, split by kind. Every field is pinned.
RESULT_ARRAYS = (
    "ionization_events",
    "excitation_events",
    "plasma_heating_erg_s",
    "radiated_erg_s",
    "ionization_cost_erg_s",
    "E_entry_eV",
    "heating_coulomb_erg_s",
    "heating_anomalous_erg_s",
    "heating_secondary_erg_s",
    "heating_terminal_erg_s",
    "ionization_events_tail",
    "excitation_events_tail",
    "ionization_cost_tail_erg_s",
    "radiated_tail_erg_s",
)
RESULT_SCALARS = (
    "transmitted_flux",
    "transmitted_energy_eV",
    "anode_intercepted_erg_s",
    "end_loss_low_erg_s",
    "end_loss_high_erg_s",
    "end_loss_transmitted_erg_s",
    "terminal_escape_flux_per_s",
    "end_loss_tail_low_erg_s",
    "end_loss_tail_high_erg_s",
    "tail_power_erg_s",
    "tail_sub_threshold_power_erg_s",
    "tail_above_bar_power_erg_s",
    "plateau_wave_power_erg_s",
)

# Diagnostic keys whose recomputation --verify requires to reproduce exactly.
COUNT_KEYS = (
    "substeps_top",
    "substeps_nested",
    "legs",
    "walk_integrations",
    "max_leg_depth",
    "cells_reached",
    "free_stream_cells",
    "absorbed",
    "anode_intercepted_fired",
)
CLOSURE_KEYS = ("budget_erg_s", "closure_rel", "tail_closure_rel", "result_min")


# --- capture: real argument tuples from the golden configuration -----------

# Sample times [s] spanning the golden run's own phase timeline. At each, the
# FIRST call of each class at or after the target is recorded.
DEFAULT_TARGETS = (
    0.0,
    2.0e-5,
    5.0e-5,
    8.5e-5,
    1.2e-4,
    1.7e-4,
    2.0e-4,
    5.0e-4,
    1.0e-3,
    2.0e-3,
    5.0e-3,
    1.0e-2,
    1.5e-2,
    2.0e-2,
    2.02e-2,
    2.1e-2,
    2.3e-2,
    2.6e-2,
    2.6186e-2,
)


def _copy_kwargs(kwargs):
    return {
        k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v)
        for k, v in kwargs.items()
    }


def capture(output, t_end, targets=DEFAULT_TARGETS):
    """Run the golden configuration and record real ``deposit_beam`` calls."""
    sys.path.insert(0, str(SCRIPT_DIR))
    import baseline_sim1d as baseline
    import cablp.solvers._sim1d.physics.cathode as cathode_mod

    real_deposit = cathode_mod.deposit_beam
    pending = {"deposition": list(targets), "probe": list(targets)}
    captured = []
    phase_calls = {}
    phase_first = {}
    champions = {
        "widest_ray": {"score": -1, "entry": None},
        "widest_tail": {"score": -1, "entry": None},
    }
    counters = {"calls": 0}
    # The phase label depends only on the time, and a step makes several calls.
    phase_cache = {"t": None, "phase": ""}

    def snapshot(label, cls, now, phase, args, kwargs):
        return {
            "label": label,
            "cls": cls,
            "t_s": now,
            "phase": phase,
            "E0_eV": float(args[0]),
            "Gamma0_per_s": float(args[1]),
            "kwargs": _copy_kwargs(kwargs),
        }

    def recording_deposit(*args, **kwargs):
        counters["calls"] += 1
        now = float(sim._time)
        if now != phase_cache["t"]:
            phase_cache["t"] = now
            phase_cache["phase"] = str(sim.phase_at_time(now))
        phase = phase_cache["phase"]
        # A deposition ray carries the transport closure keywords; a
        # gap-transmission probe deliberately never does.
        cls = (
            "deposition"
            if ("anomalous_transport" in kwargs or "product_transport" in kwargs)
            else "probe"
        )
        phase_calls[(phase, cls)] = phase_calls.get((phase, cls), 0) + 1
        queue = pending[cls]
        if queue and now >= queue[0]:
            target = queue.pop(0)
            while queue and now >= queue[0]:
                queue.pop(0)
            captured.append(
                snapshot(
                    f"real_{cls}_t{target:.6e}", cls, now, phase, args, kwargs
                )
            )
        if (phase, cls) not in phase_first:
            phase_first[(phase, cls)] = snapshot(
                f"real_{cls}_first_{phase}", cls, now, phase, args, kwargs
            )
        result = real_deposit(*args, **kwargs)
        span = int(np.count_nonzero(result.E_entry_eV > 0.0))
        if span > champions["widest_ray"]["score"]:
            champions["widest_ray"] = {
                "score": span,
                "entry": snapshot(
                    "real_widest_ray", cls, now, phase, args, kwargs
                ),
            }
        tail_span = int(
            np.count_nonzero(result.heating_anomalous_erg_s > 0.0)
        )
        if tail_span > champions["widest_tail"]["score"]:
            champions["widest_tail"] = {
                "score": tail_span,
                "entry": snapshot(
                    "real_widest_tail", cls, now, phase, args, kwargs
                ),
            }
        return result

    params, flags = baseline.build_baseline_config()
    sim = baseline.LAPDSim1D(params, flags)
    cathode_mod.deposit_beam = recording_deposit
    try:
        run_kwargs = dict(baseline.BASELINE_RUN_KWARGS)
        if t_end is not None:
            run_kwargs["t_end"] = t_end
        sim.start_simulation(**run_kwargs)
    finally:
        cathode_mod.deposit_beam = real_deposit
    result = sim.get_results()

    entries = list(captured)
    seen = {(e["cls"], e["t_s"]) for e in entries}
    for key, entry in sorted(phase_first.items()):
        if (entry["cls"], entry["t_s"]) not in seen:
            entries.append(entry)
            seen.add((entry["cls"], entry["t_s"]))
    for name, champ in champions.items():
        entry = champ["entry"]
        if entry is not None and (entry["cls"], entry["t_s"]) not in seen:
            entries.append(entry)
            seen.add((entry["cls"], entry["t_s"]))

    census = {
        "deposit_beam_calls": counters["calls"],
        "steps": int(np.asarray(result.time).size),
        "final_time_s": float(np.asarray(result.time)[-1]),
        "phase_call_counts": {
            f"{phase}/{cls}": n for (phase, cls), n in sorted(phase_calls.items())
        },
        "widest_ray_cells": champions["widest_ray"]["score"],
        "widest_tail_cells": champions["widest_tail"]["score"],
        "compiled_kernels": os.environ.get("CABLP_COMPILED_KERNELS", ""),
    }
    _write_captured(Path(output), entries, census)
    print(
        f"captured {len(entries)} real call tuples from "
        f"{counters['calls']} deposit_beam calls over {census['steps']} saves "
        f"(t_end {census['final_time_s'] * 1e3:.4f} ms) -> {output}"
    )
    for key, n in census["phase_call_counts"].items():
        print(f"    calls {key:34s} {n}")
    for entry in entries:
        print(
            f"    {entry['label']:44s} {entry['phase']:16s} "
            f"t={entry['t_s'] * 1e3:9.4f} ms"
        )
    return 0


def _write_captured(path, entries, census):
    data = {}
    manifest = []
    for i, entry in enumerate(entries):
        scalar_kwargs = {}
        for key, value in entry["kwargs"].items():
            if isinstance(value, np.ndarray):
                data[f"c{i:04d}__{key}"] = np.ascontiguousarray(
                    value, dtype=float
                )
            elif isinstance(value, tuple):
                scalar_kwargs[key] = list(value)
            else:
                scalar_kwargs[key] = value
        manifest.append(
            {
                "label": entry["label"],
                "cls": entry["cls"],
                "t_s": entry["t_s"],
                "phase": entry["phase"],
                "E0_eV": entry["E0_eV"],
                "Gamma0_per_s": entry["Gamma0_per_s"],
                "scalar_kwargs": scalar_kwargs,
            }
        )
    data["__manifest__"] = np.array(json.dumps(manifest))
    data["__census__"] = np.array(json.dumps(census))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)


def _read_captured(path):
    z = np.load(path, allow_pickle=False)
    manifest = json.loads(str(z["__manifest__"]))
    census = json.loads(str(z["__census__"]))
    entries = []
    for i, meta in enumerate(manifest):
        arrays = {}
        for key in ARRAY_KWARGS:
            name = f"c{i:04d}__{key}"
            if name in z:
                arrays[key] = np.ascontiguousarray(z[name], dtype=float)
        entries.append(
            {
                "label": meta["label"],
                "cls": f"real_{meta['cls']}",
                "phase": meta["phase"],
                "t_s": meta["t_s"],
                "E0_eV": meta["E0_eV"],
                "Gamma0_per_s": meta["Gamma0_per_s"],
                "scalar_kwargs": dict(meta["scalar_kwargs"]),
                "arrays": arrays,
            }
        )
    return entries, census


# --- synthetic adversarial entries -----------------------------------------

# Two launched fluxes. The quasilinear drag runs on the beam density, so at
# the high flux every anomalous ray self-limits inside its launch cell (the
# short-range corner) while at the low flux the same closure marches the
# column; both are production regimes and the corpus carries both.
_GAMMA0 = 1.0e22
_GAMMA0_LOW = 1.0e18


def _entry(label, cls, E0, Gamma0, arrays, **scalar_kwargs):
    return {
        "label": label,
        "cls": cls,
        "phase": "synthetic",
        "t_s": float("nan"),
        "E0_eV": float(E0),
        "Gamma0_per_s": float(Gamma0),
        "scalar_kwargs": dict(scalar_kwargs),
        "arrays": {k: np.ascontiguousarray(v, dtype=float)
                   for k, v in arrays.items()},
    }


def _uniform(cells, dz, nn, ne, Te):
    ones = np.ones(cells)
    return {
        "nn": nn * ones,
        "ne": ne * ones,
        "Te": Te * ones,
        "dz_cm": dz * ones,
    }


def _w_sec_crossing_eV(B):
    """Energy where ``<W_sec>`` first reaches ``E_stop`` -- the K7b upper bar."""
    lo, hi = B.HE_E_STOP_EV, 100.0 * B.HE_I_ION_EV
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if B.he_mean_secondary_energy_eV(mid) >= B.HE_E_STOP_EV:
            hi = mid
        else:
            lo = mid
    return hi


def _closure_families():
    """Valid ``(anomalous_model, anomalous_transport, anomalous_disposal)``."""
    return (
        ("none", "local", "local"),
        ("quasilinear", "local", "local"),
        ("ql_relaxation", "local", "local"),
        ("quasilinear", "tail_walk", "local"),
        ("ql_relaxation", "tail_walk", "local"),
        ("quasilinear", "plateau_multigroup", "local"),
        ("ql_relaxation", "plateau_multigroup", "local"),
        ("quasilinear", "local", "landau_branched"),
        ("ql_relaxation", "local", "landau_branched"),
    )


def _family_kwargs(model, transport, disposal, E0, cells, tail_ion):
    """The keyword set one closure family requires, and nothing more."""
    kw = {
        "anomalous_model": model,
        "anomalous_transport": transport,
        "anomalous_disposal": disposal,
    }
    walkers = transport in ("tail_walk", "plateau_multigroup") or (
        disposal == "landau_branched"
    )
    arrays = {}
    if model != "none":
        arrays["beam_area_cm2"] = 700.0 * np.ones(cells)
    if model == "ql_relaxation":
        kw["ql_relaxation_coeff"] = 30.0
    if transport == "plateau_multigroup":
        kw["plateau_edge_eV"] = 0.35 * E0
    elif walkers:
        kw["tail_energy_eV"] = 75.0
    if walkers and tail_ion:
        kw["tail_ionization"] = "on"
        kw["tail_walk_window"] = [0, cells - 1]
    return kw, arrays, walkers


def _synthetic_entries(B, real_entries):
    """Build the adversarial set. Every energy here is FROZEN into the fixture."""
    out = []
    E_stop = B.HE_E_STOP_EV
    I_ion = B.HE_I_ION_EV
    E_table_top = B.HE_EII_EPS_TOP * I_ion
    E_cross = _w_sec_crossing_eV(B)

    # --- 1. the closure-family cross product, both directions --------------
    cells = 24
    col = _uniform(cells, 10.0, 2.0e13, 5.0e12, 8.0)
    E0 = 150.0
    for model, transport, disposal in _closure_families():
        for product_transport in ("local", "nonlocal", "terminal_nonlocal"):
            for coulomb in ("fast_electron", "legacy_tau_ei"):
                for tail_ion in (False, True):
                    kw, extra, walkers = _family_kwargs(
                        model, transport, disposal, E0, cells, tail_ion
                    )
                    if tail_ion and not walkers:
                        continue
                    for gamma0, flux in (
                        (_GAMMA0, "hi"), (_GAMMA0_LOW, "lo")
                    ):
                        for direction in (1, -1):
                            launch = 0 if direction > 0 else cells - 1
                            arrays = dict(col)
                            arrays.update(extra)
                            out.append(
                                _entry(
                                    f"family_{model}_{transport}_{disposal}_"
                                    f"{product_transport}_{coulomb}_"
                                    f"ion{int(tail_ion)}_{flux}_"
                                    f"d{direction:+d}",
                                    "family",
                                    E0,
                                    gamma0,
                                    arrays,
                                    launch=launch,
                                    direction=direction,
                                    coulomb_model=coulomb,
                                    product_transport=product_transport,
                                    **kw,
                                )
                            )

    # --- 2. threshold-straddling walker energies ---------------------------
    # Each selects a K7b band treatment: reverted (at or below E_stop), in
    # band, or above the <W_sec> bar; the tabulated EII edge is the last one
    # the module accepts at all.
    straddle = (
        ("below_estop", np.nextafter(E_stop, -np.inf)),
        ("at_estop", E_stop),
        ("above_estop", np.nextafter(E_stop, np.inf)),
        ("below_wsec_bar", np.nextafter(E_cross, -np.inf)),
        ("at_wsec_bar", E_cross),
        ("above_wsec_bar", np.nextafter(E_cross, np.inf)),
        ("below_eii_edge", np.nextafter(E_table_top, -np.inf)),
        ("at_eii_edge", E_table_top),
        ("inside_edge_tol", E_table_top * (1.0 + 0.5 * B.HE_EII_EDGE_REL_TOL)),
    )
    cells = 16
    col = _uniform(cells, 20.0, 5.0e13, 2.0e12, 4.0)
    col["beam_area_cm2"] = 700.0 * np.ones(cells)
    for name, E_tail in straddle:
        for direction in (1, -1):
            out.append(
                _entry(
                    f"straddle_{name}_d{direction:+d}",
                    "straddle",
                    max(400.0, 2.0 * E_tail),
                    _GAMMA0_LOW,
                    col,
                    launch=cells // 2,
                    direction=direction,
                    anomalous_model="quasilinear",
                    anomalous_transport="tail_walk",
                    tail_energy_eV=float(E_tail),
                    tail_ionization="on",
                    tail_walk_window=[0, cells - 1],
                )
            )

    # --- 3. vacuum cells ---------------------------------------------------
    # A cell with no neutrals and no electrons has no stopping power at all:
    # the substep loop breaks and the primary free-streams across it.
    cells = 20
    vac = _uniform(cells, 25.0, 3.0e13, 1.0e12, 3.0)
    vac["nn"][5:9] = 0.0
    vac["ne"][5:9] = 0.0
    vac_all = _uniform(cells, 25.0, 0.0, 0.0, 3.0)
    vac_ne_only = _uniform(cells, 25.0, 0.0, 1.0e12, 3.0)
    vac_nn_only = _uniform(cells, 25.0, 3.0e13, 0.0, 3.0)
    # Only a cell with BOTH densities at zero has no stopping power; the
    # single-species columns are the contrast entries that show the vacuum
    # break does not fire on one alone.
    for name, arrays in (
        ("interior_block", vac),
        ("whole_column", vac_all),
        ("nn_zero_only", vac_ne_only),
        ("ne_zero_only", vac_nn_only),
    ):
        for direction in (1, -1):
            out.append(
                _entry(
                    f"vacuum_{name}_d{direction:+d}",
                    "vacuum",
                    150.0,
                    _GAMMA0,
                    arrays,
                    launch=0 if direction > 0 else cells - 1,
                    direction=direction,
                )
            )

    # --- 4. E_stop landings and the sub-threshold source path --------------
    cells = 400
    long_col = _uniform(cells, 10.0, 1.0e13, 5.0e12, 6.0)
    out.append(
        _entry(
            "estop_long_range_absorption", "estop", 150.0, _GAMMA0, long_col,
            launch=0, direction=1,
        )
    )
    out.append(
        _entry(
            "estop_long_range_terminal_walk", "estop", 150.0, _GAMMA0,
            long_col, launch=0, direction=1,
            product_transport="terminal_nonlocal",
        )
    )
    cells = 12
    dense = _uniform(cells, 500.0, 5.0e14, 1.0e13, 2.0)
    out.append(
        _entry(
            "estop_single_cell_death", "estop", 150.0, _GAMMA0, dense,
            launch=0, direction=1,
        )
    )
    out.append(
        _entry(
            "estop_just_above_threshold", "estop",
            float(np.nextafter(E_stop, np.inf)), _GAMMA0, dense,
            launch=0, direction=1,
        )
    )
    short = _uniform(8, 10.0, 1.0e13, 1.0e12, 3.0)
    for pt in ("local", "nonlocal"):
        out.append(
            _entry(
                f"subthreshold_source_at_estop_{pt}", "estop", E_stop,
                _GAMMA0, short, launch=0, direction=1, product_transport=pt,
            )
        )
        out.append(
            _entry(
                f"subthreshold_source_below_estop_{pt}", "estop",
                float(np.nextafter(E_stop, -np.inf)), _GAMMA0, short,
                launch=7, direction=-1, product_transport=pt,
            )
        )

    # --- 5. single-cell walk windows ---------------------------------------
    # The window must contain every cell the QL channel drives, so the ray is
    # made to die inside its launch cell and the window is that one cell.
    cells = 8
    onecell = _uniform(cells, 400.0, 5.0e14, 1.0e13, 2.0)
    onecell["beam_area_cm2"] = 700.0 * np.ones(cells)
    for launch, direction in ((3, 1), (3, -1)):
        for reflect in (None, -1, 1):
            kw = {}
            if reflect is not None:
                kw["tail_reflect_face"] = reflect
                kw["tail_reflect_threshold_eV"] = 500.0
            out.append(
                _entry(
                    f"window_single_cell_l{launch}_d{direction:+d}_"
                    f"r{reflect}", "single_cell_window", 150.0, _GAMMA0,
                    onecell, launch=launch, direction=direction,
                    anomalous_model="quasilinear",
                    anomalous_transport="tail_walk",
                    tail_energy_eV=75.0,
                    tail_ionization="on",
                    tail_walk_window=[launch, launch],
                    **kw,
                )
            )

    # --- 6. reflecting-face bounce chains ----------------------------------
    # A threshold above the walker energy turns every arriving walker around,
    # so the unfolded two-leg path is exercised on both the marched and the
    # energy-only walk.
    cells = 14
    bounce = _uniform(cells, 30.0, 2.0e13, 3.0e12, 5.0)
    bounce["beam_area_cm2"] = 700.0 * np.ones(cells)
    for reflect in (-1, 1):
        for tail_ion in (False, True):
            for threshold in (500.0, 40.0, 10.0):
                kw = {"tail_walk_window": [0, cells - 1]}
                if tail_ion:
                    kw["tail_ionization"] = "on"
                out.append(
                    _entry(
                        f"bounce_r{reflect:+d}_ion{int(tail_ion)}_"
                        f"th{threshold:g}", "bounce", 150.0, _GAMMA0_LOW,
                        bounce,
                        launch=cells // 2, direction=1,
                        anomalous_model="quasilinear",
                        anomalous_transport="tail_walk",
                        tail_energy_eV=75.0,
                        tail_reflect_face=reflect,
                        tail_reflect_threshold_eV=threshold,
                        **kw,
                    )
                )

    # --- 7. every multi-group plateau rung ---------------------------------
    cells = 20
    mg = _uniform(cells, 30.0, 2.0e13, 3.0e12, 5.0)
    mg["beam_area_cm2"] = 700.0 * np.ones(cells)
    E_b = 150.0
    E_1 = 0.35 * E_b
    for groups in (1, 2, 4, 8, 16):
        for direction in (1, -1):
            for gamma0, flux in ((_GAMMA0, "hi"), (_GAMMA0_LOW, "lo")):
                out.append(
                    _entry(
                        f"multigroup_N{groups}_{flux}_d{direction:+d}",
                        "multigroup", E_b, gamma0, mg,
                        launch=0 if direction > 0 else cells - 1,
                        direction=direction,
                        anomalous_model="quasilinear",
                        anomalous_transport="plateau_multigroup",
                        plateau_edge_eV=E_1,
                        plateau_groups=groups,
                        tail_ionization="on",
                        tail_walk_window=[0, cells - 1],
                    )
                )
    # One leg entry per rung of the shipped group count: the argument list the
    # multi-group walk stage hands its own recursive march, at the derived
    # midpoint energies.
    _edges, midpoints = B.plateau_group_edges_eV(
        E_1, E_b, B.PLATEAU_GROUP_COUNT
    )
    for i, E_hat in enumerate(midpoints):
        for direction in (1, -1):
            out.append(
                _entry(
                    f"rung_synthetic_{i}_d{direction:+d}", "rung",
                    float(E_hat), 0.5 * _GAMMA0, mg,
                    launch=cells // 2, direction=direction,
                    anomalous_model="none",
                    product_transport="local",
                )
            )
    # The same, on a REAL multi-group column at its own state-solved edge.
    real_mg = _pick_real_multigroup(real_entries)
    if real_mg is not None:
        arrays = real_mg["arrays"]
        window = real_mg["scalar_kwargs"].get("tail_walk_window")
        lo, hi = (0, arrays["dz_cm"].size - 1) if window is None else window
        win = slice(int(lo), int(hi) + 1)
        real_col = {
            "nn": arrays["nn"][win],
            "ne": arrays["ne"][win],
            "Te": arrays["Te"][win],
            "dz_cm": arrays["dz_cm"][win],
        }
        _edges_r, mids_r = B.plateau_group_edges_eV(
            float(real_mg["scalar_kwargs"]["plateau_edge_eV"]),
            float(real_mg["E0_eV"]),
            int(real_mg["scalar_kwargs"].get(
                "plateau_groups", B.PLATEAU_GROUP_COUNT
            )),
        )
        n_w = real_col["dz_cm"].size
        for i, E_hat in enumerate(mids_r):
            for direction in (1, -1):
                out.append(
                    _entry(
                        f"rung_real_{i}_d{direction:+d}", "rung_real",
                        float(E_hat), 0.5 * _GAMMA0, real_col,
                        launch=n_w // 2, direction=direction,
                        anomalous_model="none",
                        product_transport="local",
                        coulomb_model=str(
                            real_mg["scalar_kwargs"].get(
                                "coulomb_model", "fast_electron"
                            )
                        ),
                        I_ion_eV=float(
                            real_mg["scalar_kwargs"].get("I_ion_eV", I_ion)
                        ),
                    )
                )

    # --- 8. anode-interception corners -------------------------------------
    cells = 30
    anode_col = _uniform(cells, 10.0, 5.0e12, 2.0e12, 6.0)
    anode_col["beam_area_cm2"] = 700.0 * np.ones(cells)
    corners = (
        ("at_launch", 0, 1, 0, 0.4),
        ("at_far_end", 0, 1, cells - 1, 0.4),
        ("eta_zero", 0, 1, 5, 0.0),
        ("eta_near_one", 0, 1, 5, 0.999),
        ("reverse_ray", cells - 1, -1, cells - 6, 0.4),
        ("reverse_at_launch", cells - 1, -1, cells - 1, 0.4),
    )
    for name, launch, direction, cross, eta in corners:
        out.append(
            _entry(
                f"anode_{name}", "anode", 150.0, _GAMMA0, anode_col,
                launch=launch, direction=direction,
                anode_cross_index=cross, anode_eta=eta,
            )
        )
    # A ray absorbed before it ever reaches the face intercepts nothing.
    dense_anode = _uniform(cells, 200.0, 5.0e14, 1.0e13, 2.0)
    out.append(
        _entry(
            "anode_absorbed_before_face", "anode", 150.0, _GAMMA0,
            dense_anode, launch=0, direction=1, anode_cross_index=cells - 1,
            anode_eta=0.4,
        )
    )
    # Interception composed with a walked tail, which is where the two
    # ledgers have to stay separable.
    for gamma0, flux, cross in (
        (_GAMMA0, "hi", 0), (_GAMMA0_LOW, "lo", 5)
    ):
        out.append(
            _entry(
                f"anode_with_tail_walk_{flux}", "anode", 150.0, gamma0,
                anode_col, launch=0, direction=1, anode_cross_index=cross,
                anode_eta=0.4, anomalous_model="quasilinear",
                anomalous_transport="tail_walk", tail_energy_eV=75.0,
                tail_ionization="on", tail_walk_window=[0, cells - 1],
            )
        )

    # --- 9. degenerate geometry --------------------------------------------
    one = _uniform(1, 10.0, 2.0e13, 5.0e12, 8.0)
    for direction in (1, -1):
        out.append(
            _entry(
                f"geometry_single_cell_d{direction:+d}", "geometry", 150.0,
                _GAMMA0, one, launch=0, direction=direction,
            )
        )
    edge = _uniform(10, 10.0, 2.0e13, 5.0e12, 8.0)
    out.append(
        _entry(
            "geometry_launch_at_high_end", "geometry", 150.0, _GAMMA0, edge,
            launch=9, direction=1,
        )
    )
    out.append(
        _entry(
            "geometry_launch_at_low_end", "geometry", 150.0, _GAMMA0, edge,
            launch=0, direction=-1,
        )
    )
    # An enormous collisional stopping power at an energy one ulp above
    # E_stop: the landing clamp lands the substep exactly on E_stop.
    crush = _uniform(4, 1.0e6, 1.0e18, 1.0e18, 1.0)
    out.append(
        _entry(
            "geometry_estop_landing_clamp", "geometry",
            float(np.nextafter(E_stop, np.inf)), _GAMMA0, crush,
            launch=0, direction=1,
        )
    )
    # The ZERO-LENGTH substep: the only route to it is a total loss rate that
    # overflows to infinity, which the anomalous channel reaches when the beam
    # cross-section collapses and the relaxation length underflows. The
    # residual is then absorbed whole at the energy the substep started at,
    # which is what distinguishes this branch from the landing clamp above.
    collapse = _uniform(4, 1.0e6, 1.0e13, 1.0e12, 1.0)
    collapse["beam_area_cm2"] = 1.0e-20 * np.ones(4)
    out.append(
        _entry(
            "geometry_zero_length_substep", "geometry",
            float(np.nextafter(E_stop, np.inf)), 1.0e30, collapse,
            launch=0, direction=1, anomalous_model="quasilinear",
        )
    )
    return out


def _pick_real_multigroup(real_entries):
    """The widest real multi-group deposition call, if the capture found one."""
    best = None
    best_span = -1
    for entry in real_entries:
        kw = entry["scalar_kwargs"]
        if kw.get("anomalous_transport") != "plateau_multigroup":
            continue
        if "plateau_edge_eV" not in kw:
            continue
        span = int(entry["arrays"]["dz_cm"].size)
        if span > best_span:
            best, best_span = entry, span
    return best


# --- invoking and instrumenting --------------------------------------------


def _call_kwargs(entry):
    kw = {}
    for key, value in entry["scalar_kwargs"].items():
        kw[key] = tuple(value) if isinstance(value, list) else value
    kw.update(entry["arrays"])
    return kw


def _require_pure(B):
    if B._CSDA_MARCH is not None:
        raise RuntimeError(
            "the compiled CSDA march is bound "
            f"(CABLP_COMPILED_KERNELS={os.environ.get('CABLP_COMPILED_KERNELS')!r}); "
            "the reference corpus is defined against the PURE implementation. "
            "Unset the opt-in and re-run."
        )


def _invoke(B, entry, fn=None):
    fn = B.deposit_beam if fn is None else fn
    return fn(entry["E0_eV"], entry["Gamma0_per_s"], **_call_kwargs(entry))


def _invoke_instrumented(B, entry):
    """Call ``deposit_beam`` counting substeps, recursive legs and walks.

    The substep counter keys off ``He_beam_excitation_channel_lkup``, which the
    march calls exactly once per substep iteration it enters; the leg counter
    keys off the module-global ``deposit_beam`` the tail-walk stage recurses
    through, and carries a depth so a leg's substeps are separable from the
    primary's.
    """
    state = {"legs": 0, "walks": 0, "depth": 0, "max_depth": 0}
    substeps = {}
    real_exc = B.He_beam_excitation_channel_lkup
    real_dep = B.deposit_beam
    real_walk = B._walk_products_forward

    def counting_exc(E):
        substeps[state["depth"]] = substeps.get(state["depth"], 0) + 1
        return real_exc(E)

    def counting_dep(*args, **kwargs):
        state["legs"] += 1
        state["depth"] += 1
        state["max_depth"] = max(state["max_depth"], state["depth"])
        try:
            return real_dep(*args, **kwargs)
        finally:
            state["depth"] -= 1

    def counting_walk(*args, **kwargs):
        state["walks"] += 1
        return real_walk(*args, **kwargs)

    B.He_beam_excitation_channel_lkup = counting_exc
    B.deposit_beam = counting_dep
    B._walk_products_forward = counting_walk
    try:
        result = _invoke(B, entry, fn=real_dep)
    finally:
        B.He_beam_excitation_channel_lkup = real_exc
        B.deposit_beam = real_dep
        B._walk_products_forward = real_walk
    counts = {
        "substeps_top": int(substeps.get(0, 0)),
        "substeps_nested": int(
            sum(n for d, n in substeps.items() if d > 0)
        ),
        "legs": int(state["legs"]),
        "walk_integrations": int(state["walks"]),
        "max_leg_depth": int(state["max_depth"]),
    }
    return result, counts


def _fields(res):
    arrays = {
        name: np.ascontiguousarray(getattr(res, name), dtype=float)
        for name in RESULT_ARRAYS
    }
    scalars = np.array(
        [float(getattr(res, name)) for name in RESULT_SCALARS], dtype=float
    )
    return arrays, scalars


def _diagnostics(entry, arrays, scalars, counts):
    """Closure residuals, path selections and the recorded counts, per entry."""
    kw = entry["scalar_kwargs"]
    scalar = dict(zip(RESULT_SCALARS, scalars.tolist()))
    budget = entry["Gamma0_per_s"] * entry["E0_eV"] * _ERG_PER_EV
    book_transmitted = kw.get("product_transport", "local") == "nonlocal"
    total = (
        arrays["plasma_heating_erg_s"].sum()
        + arrays["radiated_erg_s"].sum()
        + arrays["ionization_cost_erg_s"].sum()
        + scalar["anode_intercepted_erg_s"]
        + scalar["end_loss_low_erg_s"]
        + scalar["end_loss_high_erg_s"]
        + scalar["end_loss_tail_low_erg_s"]
        + scalar["end_loss_tail_high_erg_s"]
    )
    if not book_transmitted:
        total += (
            scalar["transmitted_flux"]
            * scalar["transmitted_energy_eV"]
            * _ERG_PER_EV
        )
    closure_rel = (
        abs(total - budget) / budget if budget > 0.0 else abs(total - budget)
    )
    # The tail bank's own identity, where the closure launches walkers whose
    # whole bank is accounted for: everything withheld is either deposited
    # into the anomalous split, spent on the walkers' own inelastic channels,
    # or booked to the tail end ledger.
    transport = kw.get("anomalous_transport", "local")
    tail_closure = None
    if transport in ("tail_walk", "plateau_multigroup"):
        launched = (
            scalar["tail_power_erg_s"] + scalar["plateau_wave_power_erg_s"]
        )
        landed = (
            arrays["heating_anomalous_erg_s"].sum()
            + arrays["ionization_cost_tail_erg_s"].sum()
            + arrays["radiated_tail_erg_s"].sum()
            + scalar["end_loss_tail_low_erg_s"]
            + scalar["end_loss_tail_high_erg_s"]
        )
        if launched > 0.0:
            tail_closure = abs(landed - launched) / launched
    reached = arrays["E_entry_eV"] > 0.0
    banked = (
        (arrays["plasma_heating_erg_s"] > 0.0)
        | (arrays["radiated_erg_s"] > 0.0)
        | (arrays["ionization_cost_erg_s"] > 0.0)
    )
    diag = {
        "budget_erg_s": float(budget),
        "closure_rel": float(closure_rel),
        "tail_closure_rel": (
            None if tail_closure is None else float(tail_closure)
        ),
        "result_min": float(
            min(
                min(float(a.min()) for a in arrays.values()),
                float(scalars.min()),
            )
        ),
        "cells_reached": int(np.count_nonzero(reached)),
        "free_stream_cells": int(np.count_nonzero(reached & ~banked)),
        "absorbed": bool(scalar["transmitted_flux"] == 0.0),
        "anode_intercepted_fired": bool(
            scalar["anode_intercepted_erg_s"] > 0.0
        ),
    }
    diag.update(counts)
    return diag


# --- fixture I/O ------------------------------------------------------------


def _pack(entries, records, provenance):
    data = {}
    manifest = []
    for i, (entry, (arrays, scalars, diag)) in enumerate(zip(entries, records)):
        for key, value in entry["arrays"].items():
            data[f"e{i:04d}__in__{key}"] = value
        for name, value in arrays.items():
            data[f"e{i:04d}__out__{name}"] = value
        data[f"e{i:04d}__out__scalars"] = scalars
        manifest.append(
            {
                "label": entry["label"],
                "cls": entry["cls"],
                "phase": entry["phase"],
                "t_s": entry["t_s"],
                "E0_eV": entry["E0_eV"],
                "Gamma0_per_s": entry["Gamma0_per_s"],
                "scalar_kwargs": entry["scalar_kwargs"],
                "arrays": sorted(entry["arrays"]),
                "diag": diag,
            }
        )
    data["__manifest__"] = np.array(json.dumps(manifest))
    data["__provenance__"] = np.array(json.dumps(provenance))
    data["__result_arrays__"] = np.array(list(RESULT_ARRAYS))
    data["__result_scalars__"] = np.array(list(RESULT_SCALARS))
    return data


def _unpack(z):
    manifest = json.loads(str(z["__manifest__"]))
    entries = []
    for i, meta in enumerate(manifest):
        arrays = {
            key: np.ascontiguousarray(z[f"e{i:04d}__in__{key}"], dtype=float)
            for key in meta["arrays"]
        }
        entries.append(
            {
                "index": i,
                "label": meta["label"],
                "cls": meta["cls"],
                "phase": meta["phase"],
                "t_s": meta["t_s"],
                "E0_eV": meta["E0_eV"],
                "Gamma0_per_s": meta["Gamma0_per_s"],
                "scalar_kwargs": meta["scalar_kwargs"],
                "arrays": arrays,
                "diag": meta["diag"],
            }
        )
    return entries


def _bitdiff(a, b):
    """Differing float64 values at raw uint64; both-NaN counts as equal."""
    a = np.ascontiguousarray(a, dtype=float)
    b = np.ascontiguousarray(b, dtype=float)
    if a.shape != b.shape:
        return max(a.size, b.size)
    both_nan = np.isnan(a) & np.isnan(b)
    return int((~both_nan & (a.view(np.uint64) != b.view(np.uint64))).sum())


# --- build ------------------------------------------------------------------


def build(captured_path, fixture_path):
    import cablp.funcs._beam_deposition as B

    _require_pure(B)
    real_entries, census = _read_captured(Path(captured_path))
    entries = real_entries + _synthetic_entries(B, real_entries)
    records = []
    for entry in entries:
        result, counts = _invoke_instrumented(B, entry)
        arrays, scalars = _fields(result)
        # The counters must not perturb what they count.
        plain_arrays, plain_scalars = _fields(_invoke(B, entry))
        differing = sum(
            _bitdiff(arrays[name], plain_arrays[name]) for name in RESULT_ARRAYS
        ) + _bitdiff(scalars, plain_scalars)
        if differing:
            raise RuntimeError(
                f"instrumentation perturbed entry {entry['label']!r} "
                f"({differing} differing values); the counts would not "
                "describe the reference outputs"
            )
        records.append(
            (arrays, scalars, _diagnostics(entry, arrays, scalars, counts))
        )
    provenance = {
        "format": FIXTURE_FORMAT,
        "reference_path": "pure",
        "numpy": np.__version__,
        "platform": sys.platform,
        "capture_census": census,
        "entries": len(entries),
        "note": (
            "Reference outputs computed by the pure-Python deposit_beam. The "
            "capture run that produced the real argument tuples records its "
            "own kernel selection in capture_census.compiled_kernels; the "
            "compiled march is bit-exact against pure, so the trajectory it "
            "sampled is the pure one."
        ),
    }
    data = _pack(entries, records, provenance)
    path = Path(fixture_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    _print_census(entries, records, census)
    print(
        f"built {len(entries)} corpus entries -> {path} "
        f"({path.stat().st_size / 1024.0:.1f} kB)"
    )
    return 0


def _print_census(entries, records, capture_census):
    by_class = {}
    for entry, (_a, _s, diag) in zip(entries, records):
        key = entry["cls"]
        bucket = by_class.setdefault(
            key, {"n": 0, "closure": [], "tail": [], "legs": 0, "substeps": 0}
        )
        bucket["n"] += 1
        bucket["closure"].append(diag["closure_rel"])
        if diag["tail_closure_rel"] is not None:
            bucket["tail"].append(diag["tail_closure_rel"])
        bucket["legs"] += diag["legs"]
        bucket["substeps"] += diag["substeps_top"] + diag["substeps_nested"]
    print("corpus census (class: entries, worst closure, worst tail closure):")
    for key in sorted(by_class):
        b = by_class[key]
        tail = max(b["tail"]) if b["tail"] else float("nan")
        print(
            f"    {key:22s} n={b['n']:4d}  closure<={max(b['closure']):.3e}  "
            f"tail<={tail:.3e}  legs={b['legs']:6d}  substeps={b['substeps']:8d}"
        )
    phases = {}
    for entry in entries:
        if entry["cls"].startswith("real"):
            phases[entry["phase"]] = phases.get(entry["phase"], 0) + 1
    if phases:
        print("real entries per phase:")
        for phase in sorted(phases):
            print(f"    {phase:22s} {phases[phase]}")
    # The capture run's own per-phase call census, carried into the fixture
    # provenance: a phase absent here made NO deposit_beam calls, and that
    # absence is a recorded fact rather than a gap in the corpus.
    calls = capture_census.get("phase_call_counts", {})
    print("deposit_beam calls made by the capture run, per phase/class:")
    for key in sorted(calls):
        print(f"    {key:34s} {calls[key]}")
    rungs = set()
    for entry in entries:
        if entry["cls"] in ("rung", "rung_real"):
            rungs.add((entry["cls"], round(entry["E0_eV"], 9)))
    if rungs:
        print(
            f"plateau rung leg entries: "
            f"{len({r for c, r in rungs if c == 'rung'})} synthetic + "
            f"{len({r for c, r in rungs if c == 'rung_real'})} real midpoints"
        )
    closure_all = [d["closure_rel"] for _e, (_a, _s, d) in
                   zip(entries, records)]
    tail_all = [d["tail_closure_rel"] for _e, (_a, _s, d) in
                zip(entries, records) if d["tail_closure_rel"] is not None]
    print(
        f"closure over all {len(closure_all)} entries: max "
        f"{max(closure_all):.6e}, median "
        f"{float(np.median(closure_all)):.6e}"
    )
    if tail_all:
        print(
            f"tail-bank closure over {len(tail_all)} walked entries: max "
            f"{max(tail_all):.6e}, median "
            f"{float(np.median(tail_all)):.6e}"
        )


# --- verify -----------------------------------------------------------------


def verify(fixture_path, quiet=False):
    import cablp.funcs._beam_deposition as B

    path = Path(fixture_path)
    if not path.exists():
        print(f"FAIL: fixture not found: {path}")
        return 1
    _require_pure(B)
    z = np.load(path, allow_pickle=False)
    entries = _unpack(z)
    if not entries:
        print("FAIL: fixture manifest is EMPTY -- refusing to report a pass")
        return 1
    provenance = json.loads(str(z["__provenance__"]))
    total = bad = compared_fields = 0
    diag_bad = []
    for entry in entries:
        i = entry["index"]
        result, counts = _invoke_instrumented(B, entry)
        arrays, scalars = _fields(result)
        for name in RESULT_ARRAYS:
            want = z[f"e{i:04d}__out__{name}"]
            got = arrays[name]
            d = _bitdiff(got, want)
            total += want.size
            compared_fields += 1
            bad += d
            if d and not quiet:
                print(
                    f"    MISMATCH {entry['label']:44s} {name:28s} "
                    f"{d:6d} / {want.size}"
                )
        want_s = z[f"e{i:04d}__out__scalars"]
        d = _bitdiff(scalars, want_s)
        total += want_s.size
        compared_fields += 1
        bad += d
        if d and not quiet:
            names = [
                RESULT_SCALARS[k]
                for k in range(len(RESULT_SCALARS))
                if _bitdiff(scalars[k:k + 1], want_s[k:k + 1])
            ]
            print(
                f"    MISMATCH {entry['label']:44s} scalars "
                f"{d:6d} / {want_s.size}  ({', '.join(names)})"
            )
        diag = _diagnostics(entry, arrays, scalars, counts)
        stored = entry["diag"]
        for key in COUNT_KEYS:
            if diag[key] != stored[key]:
                diag_bad.append(
                    f"{entry['label']}: {key} {diag[key]!r} != "
                    f"{stored[key]!r}"
                )
        for key in CLOSURE_KEYS:
            a, b = diag[key], stored[key]
            if a is None or b is None:
                if a is not b:
                    diag_bad.append(f"{entry['label']}: {key} {a!r} != {b!r}")
            elif _bitdiff(np.array([a]), np.array([b])):
                diag_bad.append(f"{entry['label']}: {key} {a!r} != {b!r}")
    if compared_fields == 0 or total == 0:
        print("FAIL: nothing was compared -- refusing to report a pass")
        return 1
    print(
        f"fixture {path.name}: {len(entries)} entries, format "
        f"{provenance.get('format')}, reference path "
        f"{provenance.get('reference_path')}"
    )
    print(
        f"compared {total} values across {compared_fields} result fields: "
        f"{bad} differing"
    )
    closures = [e["diag"]["closure_rel"] for e in entries]
    tails = [
        e["diag"]["tail_closure_rel"]
        for e in entries
        if e["diag"]["tail_closure_rel"] is not None
    ]
    print(
        f"recorded conservation closure: max {max(closures):.6e} over "
        f"{len(closures)} entries"
        + (
            f"; tail bank max {max(tails):.6e} over {len(tails)}"
            if tails
            else ""
        )
    )
    if diag_bad:
        print(
            f"DIAGNOSTIC MISMATCH -- {len(diag_bad)} recorded closure/count "
            "fields did not reproduce:"
        )
        for line in diag_bad[:20]:
            print(f"    {line}")
        if len(diag_bad) > 20:
            print(f"    ... and {len(diag_bad) - 20} more")
    if bad or diag_bad:
        print(
            "VERIFY FAILED -- the live implementation does not reproduce the "
            "reference corpus."
        )
        return 1
    print(
        "VERIFY OK -- the live pure implementation is bit-identical to the "
        "reference corpus, and its closure and substep/leg counts reproduce."
    )
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--capture", action="store_true",
        help="run the golden configuration and record real argument tuples",
    )
    g.add_argument(
        "--build", action="store_true",
        help="write the committed fixture from a capture plus the synthetic set",
    )
    g.add_argument(
        "--verify", action="store_true",
        help="check the live implementation against the fixture, at raw uint64",
    )
    p.add_argument("--output", help="--capture destination .npz")
    p.add_argument(
        "--captured", help="--build source, the .npz written by --capture"
    )
    p.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    p.add_argument(
        "--t-end", type=float, default=None,
        help="--capture: stop early (default: the fixture's own dynamic t_end)",
    )
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args(argv)
    if a.capture:
        if not a.output:
            p.error("--capture needs --output")
        return capture(a.output, a.t_end)
    if a.build:
        if not a.captured:
            p.error("--build needs --captured")
        return build(a.captured, a.fixture)
    return verify(a.fixture, a.quiet)


if __name__ == "__main__":
    raise SystemExit(main())
