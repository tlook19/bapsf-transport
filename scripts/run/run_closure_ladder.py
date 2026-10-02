#!/usr/bin/env python
"""Closure ladder: the kinetic reference, its parameter bands and the fluid closures.

One driver for the whole ladder, per rung (ES1 / ES2 / ES3): the kinetic
reference arm and its parameter-bracket band (the foot pair and three
fill-kernel arms included), the two fluid-closure toggle arms and each one's
own band. Every arm is a DERIVED configuration of one named base
configuration (``--stance``), under the prescribed-measured drive.

    python scripts/run/run_closure_ladder.py base --stance g1atrim --root DIR
    python scripts/run/run_closure_ladder.py feet --stance g1atrim --root DIR --es 1 2 3
    python scripts/run/run_closure_ladder.py gen  --stance g1atrim --root DIR --es 1
    python scripts/run/run_closure_ladder.py probe --stance g1atrim --root DIR --lanes 2 --es 1
    python scripts/run/run_closure_ladder.py run  --root DIR --lanes 4 --es 1
    python scripts/run/run_closure_ladder.py status|score|census|tabulate --root DIR --es 1

Run from the repository root. ``--root`` is the ladder tree (configs, runs,
feet, base, tabulations) and must lie outside the repository. ``--repo`` is
the checkout every child process runs in (default: the checkout holding this
file); a dedicated worktree keeps running arms insulated from edits to the
main checkout.

THE NAMED CONFIGURATION. ``base``, ``feet``, ``gen`` and ``probe`` read or
write configurations and REQUIRE ``--stance NAME_OR_PATH``: a committed
configuration name in ``<repo>/scripts/stances/`` or the path of such a file.
Every arm file declares ``base = "<name>"``, and a base resolves only among
the committed configurations, so a path outside that directory is refused.
The base must be a root configuration (no ``base`` key of its own): the
driver reads the fill block and the geometry rows from that file directly.
``--no-stance`` is refused -- every arm is a delta over the named base, so a
ladder without one has no configuration to derive from. ``run``, ``status``,
``score``, ``census`` and ``tabulate`` act on the arms already generated,
each of which names its own configuration file.

SUBCOMMANDS

``base``
    The 27 ms-duty equilibrated base: a derived file (equilibrate route on,
    profile route off) run for 2e-5 s on the compiled path, plus the geometry
    npz rebuilt from the base configuration's rows. With
    ``--compare-base-dir`` (another tree's ``base/``) the neutral rows and
    the geometry arrays are compared raw-uint64 against it.

``feet``
    The per-(rung, end) foot profiles via ``scripts/stance/sp3_build_nn0.py``.
    The ``ref`` end runs the builder's REGISTERED route (no ``--kernel``, no
    ``--dt-foot-s``); ``lo`` / ``hi`` state ``--dt-foot-s`` = registered
    foot -/+ the lead's shot sd; ``kslow`` / ``kfast`` / ``gapoff`` select
    the Knudsen member or switch the gap coupling off. The registered foot
    per rung is the measured circuit-on -> 1 kA lead minus the model's own
    1 kA time, rounded to 10 us (``FOOT``). ACCEPTANCE: the ``ref`` rows must
    reproduce, raw-uint64, the base configuration's fill rows (ES1) and the
    committed examples ``<base>_es{2,3}_reference.toml`` (ES2 / ES3), and
    with ``--compare-feet-dir`` also that directory's
    ``foot_es<N>_ref.npz``; a mismatch stops the driver.

``gen``
    Writes and pre-flights every arm configuration of the rung
    (``scripts/gates/preflight_diffcfg.py``). ES1's ``ref`` end inherits the
    base configuration's fill rows; every other arm-end restates the
    ``[models.initial_neutral_state]`` block carrying its rebuilt rows. A
    fill-carrying arm passes pre-flight when its only unexpected deltas are
    the two fill rows and the constructor probe is OK. A failing arm is
    marked ``configs/<arm>.BLOCKED``.

``probe``
    PER-ARM t0 REGISTRATION. The prescribed drive's hand-off instant
    ``cathode_prescribed_t0_s`` = ``cathode_prescribed_start_s`` is the
    arm's own scorer main-discharge origin: the arm's own model 1 kA time
    rounded UP to the 10 us save grid. Each arm is registered in three steps:
      (a) PROBE -- ``<arm>.probe.toml`` is the arm's configuration with
          t0 = start = ``PROBE_T0`` (past any crossing, so the cathode
          reaches 1 kA on its own self-consistent solve), run under a step
          cap with ``max_steps_action=stop``; the probe's saved h5 carries
          the model's own ``t_breakdown_trigger``.
      (b) REGISTER -- t0 = that time rounded up to the 10 us grid (a time
          within ``GRID_TOL`` of a grid point is treated as on the grid),
          written into the arm's final configuration as both keys, which is
          then re-pre-flighted.
      (c) CHECK -- after the full run (``score``), the scorer's
          main-discharge origin must equal t0 to within 1 ULP and the run's
          own ``t_breakdown_trigger`` must satisfy t_1kA <= t0 < t_1kA + 10 us.
    Probes are resumable, lane-counted alongside runs (both are
    ``run_m6_point`` processes) and cached: a registered arm is never probed
    again.

``run``
    Launches every registered, unblocked arm, at most ``--lanes`` live
    solver processes at a time counted machine-wide (``live_solvers``);
    resumable. Each arm writes ``<arm>.cmd`` / ``.start`` / ``.exit`` /
    ``.pid`` beside its h5 and log.

LAUNCH ROUTE FOR ``run`` AND ``probe``. Both start solver children and poll
them for hours. Every child stays in the driver's own process group (no new
session; ``timeout --foreground``, since a plain ``timeout`` moves its
command into a process group of its own), runs under its own ``timeout``
cap, and writes its stdout and
stderr to its own log. The two subcommands are an ORCHESTRATOR launch route:
the orchestrator starts the driver inside one detached tmux session, which
owns the driver and every arm it launches. Subagents must not run ``run`` or
``probe``.

``score``
    Runs the scorer over each finished arm twice (default window and
    ``--window plateau``, each with ``--json``) and the registration check (c).

``census``
    The afterglow tail hand-off census with
    ``--assert-no-handoff-before-ms 21.5`` over each finished arm, each
    reported PASS, FAIL or DID NOT RUN (census exit 2: no afterglow, or the
    record ends before 21.5 ms without a hand-off). The subcommand exits 1
    if any arm FAILED, else 2 if any DID NOT RUN, else 0.

``status`` / ``tabulate``
    Per-arm state, and the per-rung plateau stage-(ii) summary, band
    envelopes and t0 registration tables under ``tabulate/``.
"""
import argparse
import datetime as dt
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import numpy as np

# scripts/ sibling imports: the seven purpose subdirectories on sys.path.
import sys as _sys
from pathlib import Path as _Path
for _sub in ("atomic", "gates", "kinetic", "run", "score", "stance",
             "verify"):
    _dir = str(_Path(__file__).resolve().parents[1] / _sub)
    if _dir not in _sys.path:
        _sys.path.insert(0, _dir)

PROG = "run_closure_ladder"
THIS_CHECKOUT = Path(__file__).resolve().parents[2]

# Set by ``_configure`` from the command line before any subcommand runs.
ART = None
REPO = None
BASE_NAME = None
BASE_STANCE = None
EXAMPLES = None
CFG_DIR = None
RUN_DIR = None
FEET_DIR = None
BASE_DIR = None
TAB_DIR = None
COMPARE_BASE_DIR = None
COMPARE_FEET_DIR = None

PY = Path(sys.executable)
MAX_STEPS = 300000
PROBE_MAX_STEPS = 2500
PROBE_T0 = 5.0e-4
SAVE_DT = 1.0e-5
GRID_TOL = 1.0e-12
TIMEOUT_S = 43200
PROBE_TIMEOUT_S = 7200
FILL_KEYS = ("nn0_profile", "nn0_annulus_profile")
FILL_FAMILY = "initial_neutral_state"

# The measured-drive route per rung. t0 is registered per arm.
RUNGS = {
    1: {"trace": "scripts/data/es1_sim1d_overlay.npz", "V_bank": 177.843},
    2: {"trace": "scripts/data/es2_sim1d_overlay.npz", "V_bank": 138.303, "Ts": 1949.0},
    3: {"trace": "scripts/data/es3_sim1d_overlay.npz", "V_bank": 98.814, "Ts": 1972.0},
}
# The foot per rung: measured circuit-on -> 1 kA lead (s), its shot sd (s), the model's own 1 kA time (s).
FOOT = {1: (5.95e-3, 0.09e-3, 0.067e-3), 2: (6.75e-3, 0.02e-3, 0.089e-3), 3: (6.77e-3, 0.09e-3, 0.139e-3)}
S_GP0 = 9010.0


def _refuse(msg):
    raise SystemExit(f"{PROG}: {msg}")


def _resolve_stance(spec, repo):
    """Return ``(name, path)`` for a committed configuration in ``repo``.

    ``spec`` is a committed name or the path of a file in
    ``<repo>/scripts/stances/``. Refuses anything else, and a derived base.
    """
    stance_dir = (repo / "scripts" / "stances").resolve()
    available = ", ".join(sorted(p.stem for p in stance_dir.glob("*.toml"))) or "(none committed)"
    text = str(spec)
    if "/" in text or "\\" in text or text.endswith(".toml"):
        path = Path(text).resolve()
        if not path.is_file():
            _refuse(f"no configuration file at {text}. Available committed configurations: {available}")
        if path.parent != stance_dir:
            _refuse(f"--stance {text} is not a committed configuration in {stance_dir}: every arm "
                    f"declares `base = \"<name>\"`, and a base resolves only among the committed "
                    f"configurations. Available: {available}")
        name = path.stem
    else:
        name = text
        path = stance_dir / f"{name}.toml"
        if not path.is_file():
            _refuse(f"unknown stance {name!r}: no {name}.toml in {stance_dir}. Available: {available}")
    with open(path, "rb") as fh:
        doc = tomllib.load(fh)
    if "base" in doc:
        _refuse(f"--stance {name} is a derived configuration (base = {doc['base']!r}); the ladder reads "
                f"the fill block and the geometry rows from the base file itself, so its base must be a "
                f"root configuration")
    return name, path


def _configure(args):
    """Bind the module-level tree and checkout paths from the command line."""
    global ART, REPO, BASE_NAME, BASE_STANCE, EXAMPLES, CFG_DIR, RUN_DIR, FEET_DIR, BASE_DIR, TAB_DIR
    global COMPARE_BASE_DIR, COMPARE_FEET_DIR
    REPO = Path(args.repo).resolve()
    if getattr(args, "needs_stance", False):
        if getattr(args, "no_stance", False):
            _refuse("--no-stance is refused: every arm is a derived configuration of the named base "
                    "(`base = \"<name>\"`), so a ladder without one has nothing to derive from. "
                    "Pass --stance <name>.")
        if args.stance is None:
            _refuse("name the configuration package. Pass --stance <name> naming the committed base "
                    "configuration every arm derives from (a name in scripts/stances/ or the path of "
                    "such a file).")
        BASE_NAME, BASE_STANCE = _resolve_stance(args.stance, REPO)
    ART = Path(args.root).resolve()
    for checkout in (REPO, THIS_CHECKOUT):
        if ART == checkout or checkout in ART.parents:
            _refuse(f"--root {ART} lies inside the checkout {checkout}; the ladder tree holds run "
                    f"artifacts and belongs outside the repository")
    EXAMPLES = REPO / "scripts/stances/examples"
    CFG_DIR = ART / "configs"
    RUN_DIR = ART / "runs"
    FEET_DIR = ART / "feet"
    BASE_DIR = ART / "base"
    TAB_DIR = ART / "tabulate"
    COMPARE_BASE_DIR = Path(args.compare_base_dir) if getattr(args, "compare_base_dir", None) else None
    COMPARE_FEET_DIR = Path(args.compare_feet_dir) if getattr(args, "compare_feet_dir", None) else None


def grid_t0(t: float) -> float:
    """The 10 us save-grid instant at or above t, as the decimal the config carries."""
    k = t / SAVE_DT
    kr = round(k)
    k_int = int(kr) if abs(t - kr * SAVE_DT) <= GRID_TOL else int(math.ceil(k))
    return float(f"{k_int}e-05")


def nominal_t0(es: int) -> float:
    """The rung's nominal t0 -- the placeholder an arm carries until its own probe lands."""
    return grid_t0(FOOT[es][2])


def registered_foot_s(es: int) -> float:
    """The rung's registered foot: measured lead minus the model's 1 kA time, rounded to 10 us."""
    lead, _sd, t1ka = FOOT[es]
    return round(lead - t1ka, 5)


def foot_s(es: int, end: str) -> float:
    return registered_foot_s(es) + {"lo": -FOOT[es][1], "hi": +FOOT[es][1]}[end]


def foot_npz(es: int, end: str) -> Path:
    return FEET_DIR / f"foot_es{es}_{end}.npz"


# The builder arguments each foot end states beyond the common base/geometry set.
# The 'ref' end states NOTHING: the registered route is the builder's own default.
FOOT_ENDS = {
    "ref": [],
    "lo": None,       # filled per rung: --dt-foot-s (registered foot - sd)
    "hi": None,       # filled per rung: --dt-foot-s (registered foot + sd)
    "kslow": ["--knudsen-member", "slow"],
    "kfast": ["--knudsen-member", "fast"],
    "gapoff": ["--no-knudsen-gap-coupling"],
}


def foot_args(es: int, end: str):
    if end in ("lo", "hi"):
        return ["--dt-foot-s", f"{foot_s(es, end):.6g}"]
    return list(FOOT_ENDS[end])


# Kinetic band: (arm, key, value, fill end). The fill-end arms carry no scalar key:
# their delta IS the rebuilt fill block.
KIN_ARMS = [
    ("ref",        None,                                   None,   "ref"),
    ("tailfwd",    "heating_anomalous_tail_forward_fraction", 1.0, "ref"),
    ("f_lo",       "heat_flux_limiter_f",                  0.32,   "ref"),
    ("f_hi",       "heat_flux_limiter_f",                  1.5,    "ref"),
    ("sgp_lo",     "S_gp",                                 8650.0, "ref"),
    ("sgp_hi",     "S_gp",                                 9497.0, "ref"),
    ("acc_lo",     "neutral_kinetic_dvm_accommodation",    0.35,   "ref"),
    ("acc_hi",     "neutral_kinetic_dvm_accommodation",    0.46,   "ref"),
    ("pump_lo",    ("S_pump_L", "S_pump_R"),               2750.0, "ref"),
    ("pump_hi",    ("S_pump_L", "S_pump_R"),               3300.0, "ref"),
    ("beta_lo",    "neutral_kinetic_dvm_jet_launch_width", 0.085,  "ref"),
    ("beta_hi",    "neutral_kinetic_dvm_jet_launch_width", 0.19,   "ref"),
    ("foot_lo",    None,                                   None,   "lo"),
    ("foot_hi",    None,                                   None,   "hi"),
    ("kfill_slow", None,                                   None,   "kslow"),
    ("kfill_fast", None,                                   None,   "kfast"),
    ("gap_closed", None,                                   None,   "gapoff"),
]
# Fluid closures: their deltas over the base (the committed fluid comparator's, + the DVM put-aways).
CLOSURE_DELTAS = {
    "diffusive": {
        "dict": {"neutral_model": "moment", "neutral_kinetic_dvm_cathode_jet": False, "neutral_kinetic_dvm_anode_jet": False},
        "flags": {},
        "none": ["neutral_kinetic_dvm_jet_launch_width"],
    },
    "moment": {
        "dict": {"neutral_model": "moment", "cathode_neutral_jet": True, "cathode_jet_surface_debit": True,
                 "cathode_jet_energy_convention": "total_reflected",
                 "neutral_kinetic_dvm_cathode_jet": False, "neutral_kinetic_dvm_anode_jet": False},
        "flags": {"neutral_momentum": True, "neutral_energy": True, "neutral_hot_internal_wall": True},
        "none": ["neutral_kinetic_dvm_jet_launch_width"],
    },
}
FLUID_BANDS = {
    "diffusive": [("f_lo", "heat_flux_limiter_f", 0.32), ("f_hi", "heat_flux_limiter_f", 1.5),
                  ("sgp_lo", "S_gp", 8650.0), ("sgp_hi", "S_gp", 9497.0),
                  ("pump_lo", ("S_pump_L", "S_pump_R"), 2750.0), ("pump_hi", ("S_pump_L", "S_pump_R"), 3300.0)],
    "moment": [("f_lo", "heat_flux_limiter_f", 0.32), ("f_hi", "heat_flux_limiter_f", 1.5),
               ("sgp_lo", "S_gp", 8650.0), ("sgp_hi", "S_gp", 9497.0),
               ("pump_lo", ("S_pump_L", "S_pump_R"), 2750.0), ("pump_hi", ("S_pump_L", "S_pump_R"), 3300.0),
               ("acc_lo", "neutral_energy_wall_accommodation", 0.35), ("acc_hi", "neutral_energy_wall_accommodation", 0.46),
               ("jetB", ("cathode_jet_R_N", "cathode_jet_R_E"), (0.099, 0.0207)),
               ("jetLa", ("cathode_jet_R_N", "cathode_jet_R_E"), (0.572, 0.348))],
}


def arm_table():
    """Every arm as (name, closure|None, {key: value} flat deltas, fill end)."""
    out = []
    for name, key, val, end in KIN_ARMS:
        keys = key if isinstance(key, tuple) else ((key,) if key else ())
        out.append((name, None, {k: val for k in keys}, end))
    for closure in CLOSURE_DELTAS:
        out.append((closure, closure, {}, "ref"))
        for n, key, val in FLUID_BANDS[closure]:
            keys = key if isinstance(key, tuple) else (key,)
            vals = val if isinstance(val, tuple) else (val,) * len(keys)
            out.append((f"{closure}_{n}", closure, dict(zip(keys, vals)), "ref"))
    return out


ARMS = arm_table()
FEET_ENDS_USED = sorted({end for _, _, _, end in ARMS} | {"ref"})


def arm_id(name, es):
    return f"{name}_es{es}"


def paths(name, es):
    aid = arm_id(name, es)
    d = RUN_DIR / aid
    return {"id": aid, "dir": d, "cfg": CFG_DIR / f"{aid}.toml", "pre": CFG_DIR / f"{aid}.preflight.log",
            "blocked": CFG_DIR / f"{aid}.BLOCKED", "h5": d / f"{aid}.h5", "log": d / f"{aid}.log",
            "cmd": d / f"{aid}.cmd", "start": d / f"{aid}.start", "exit": d / f"{aid}.exit", "pid": d / f"{aid}.pid",
            "score": d / f"{aid}.score.txt", "score_json": d / f"{aid}.score.json",
            "score_pl": d / f"{aid}.score_plateau.txt", "score_pl_json": d / f"{aid}.score_plateau.json",
            "census": d / f"{aid}.tailcensus.txt",
            "reg": CFG_DIR / f"{aid}.registration.json", "check": d / f"{aid}.registration_check.json",
            "pcfg": CFG_DIR / f"{aid}.probe.toml", "ph5": d / f"{aid}.probe.h5", "plog": d / f"{aid}.probe.log",
            "pcmd": d / f"{aid}.probe.cmd", "pstart": d / f"{aid}.probe.start", "pexit": d / f"{aid}.probe.exit",
            "ppid": d / f"{aid}.probe.pid"}


def sgp_for(name):
    return 8650.0 if name.endswith("sgp_lo") else 9497.0 if name.endswith("sgp_hi") else S_GP0


def toml_scalar(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(float(v)) if isinstance(v, float) else str(v)
    return '"' + str(v) + '"'


def toml_array(a):
    vals = [repr(float(x)) for x in np.asarray(a, dtype=float)]
    rows = [", ".join(vals[i:i + 4]) for i in range(0, len(vals), 4)]
    return "[\n" + "".join(f"  {r},\n" for r in rows) + "]"


def load_base():
    with open(BASE_STANCE, "rb") as fh:
        return tomllib.load(fh)


def u64(a):
    return np.asarray(a, dtype=float).view(np.uint64)


def run_wt(argv, log: Path, env_extra=None, timeout=None):
    env = {**os.environ, "PYTHONPATH": str(REPO), "PYTHONDONTWRITEBYTECODE": "1"}
    env.pop("CABLP_COMPILED_KERNELS", None)
    if env_extra:
        env.update(env_extra)
    r = subprocess.run([str(x) for x in argv], cwd=REPO, capture_output=True, text=True, env=env, timeout=timeout)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("$ " + " ".join(shlex.quote(str(a)) for a in argv) + "\n\n" + r.stdout + r.stderr)
    return r


# ----------------------------------------------------------------------------- base / feet
def cmd_base(args):
    """The 27 ms equilibrated base: derived file (equilibrate route on, profile route off), 2e-5 s run."""
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    base = load_base()["models"][FILL_FAMILY]
    members = dict(base)
    members.update({"initial_neutral_state": "equilibrate"})
    for k in FILL_KEYS:
        members.pop(k, None)
    lines = ["# The 27 ms-duty equilibrated base for the ladder's foot profiles: the", "# equilibrate route replaces the profile route; nothing else moves.",
             f'base = "{BASE_NAME}"', "", f"[models.{FILL_FAMILY}]"]
    lines += [f"{k} = {toml_scalar(v)}" for k, v in members.items() if not isinstance(v, list)]
    # nn0 is the DIRECT-RUN fill and is inert on the equilibrate route; it carries the template's own value.
    lines += ["nn0 = 2.0e13",
              'none_valued = ["neutral_seed_cache_dir", "nn0_profile", "nn0_annulus_profile", "restart_from"]']
    cfg = BASE_DIR / "base27.toml"
    cfg.write_text("\n".join(lines) + "\n")
    h5 = BASE_DIR / "base27.h5"
    r = run_wt([PY, "scripts/run/run_sim1d.py", "--config", cfg, "--t-end", "2e-5", "--output", h5],
               BASE_DIR / "base27.log", env_extra={"CABLP_COMPILED_KERNELS": "1"}, timeout=7200)
    print(f"base27 rc={r.returncode} -> {h5}")
    if r.returncode:
        print(r.stdout[-2000:], r.stderr[-2000:])
        sys.exit(2)
    # geometry npz from the base configuration's rows (sp3's --extra-npz source)
    st = load_base()["input_dict"]
    keys = ["plasma_radius_profile_cm", "machine_radius_profile_cm", "neutral_baffle_positions_cm", "neutral_baffle_clear_radii_cm"]
    np.savez(BASE_DIR / "stance_geom.npz", **{k: np.asarray(st[k], dtype=float) for k in keys})
    print("geometry npz written")
    compare_base(h5, keys)


def compare_base(h5: Path, geom_keys):
    """Raw-uint64 identity of the neutral rows (and the geometry npz) against ``--compare-base-dir``."""
    import h5py
    if COMPARE_BASE_DIR is None:
        print("base identity: SKIPPED, no --compare-base-dir")
        return
    old = COMPARE_BASE_DIR / "base27.h5"
    report = {"old": str(old), "new": str(h5)}
    if not old.exists():
        print(f"base identity: SKIPPED, {old} missing")
        return
    with h5py.File(h5, "r") as a, h5py.File(old, "r") as b:
        for k in ("nn", "nn_a", "time"):
            if k not in a or k not in b:
                report[k] = "absent"
                continue
            x, y = a[k][...], b[k][...]
            if x.shape != y.shape:
                report[k] = f"SHAPE {x.shape} vs {y.shape}"
                continue
            d = int((u64(x.ravel()) != u64(y.ravel())).sum())
            report[k] = f"{d} of {x.size} differ"
            if x.ndim == 2:
                # the FOOT BUILDER reads the t=0 frame only (sp3 --base-from-h5), so that row is
                # the one the identity claim is about; later frames carry the plasma evolution.
                report[k + ":per_save"] = [f"save {i} (t={a['time'][i]:.1e}): "
                                           f"{int((u64(x[i]) != u64(y[i])).sum())} of {x.shape[1]} differ"
                                           for i in range(x.shape[0])]
    g_new = np.load(BASE_DIR / "stance_geom.npz")
    g_old_p = COMPARE_BASE_DIR / "stance_geom.npz"
    if g_old_p.exists():
        g_old = np.load(g_old_p)
        for k in geom_keys:
            d = int((u64(g_new[k]) != u64(g_old[k])).sum())
            report[f"geom:{k}"] = f"{d} of {g_new[k].size} differ"
    (BASE_DIR / "base_identity.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"base identity vs {COMPARE_BASE_DIR}:")
    for k, v in report.items():
        if k in ("old", "new"):
            continue
        if isinstance(v, list):
            for row in v:
                print(f"  {k:34s} {row}")
        else:
            print(f"  {k:34s} {v}")


def cmd_feet(args):
    FEET_DIR.mkdir(parents=True, exist_ok=True)
    g = BASE_DIR / "stance_geom.npz"
    h5 = BASE_DIR / "base27.h5"
    if not (g.exists() and h5.exists()):
        sys.exit(f"{PROG}: run the `base` subcommand first")
    for es in args.es:
        for end in FEET_ENDS_USED:
            out = foot_npz(es, end)
            argv = [PY, "scripts/stance/sp3_build_nn0.py", "--es", es, "--nx", 268, "--sgp", 9010,
                    "--base-from-h5", h5] + foot_args(es, end) + \
                   ["--extra", "gas_puff_orifice_id_cm=3.95", "gas_puff_orifice_length_cm=22.0",
                    "--extra-npz"] + [f"{k}={g}:{k}" for k in
                                      ("plasma_radius_profile_cm", "machine_radius_profile_cm", "neutral_baffle_positions_cm", "neutral_baffle_clear_radii_cm")] + \
                   ["--out", out]
            r = run_wt(argv, FEET_DIR / f"foot_es{es}_{end}.log", timeout=1800)
            m = re.search(r"S_gp x dt_foot = ([0-9.e+]+) atoms", r.stdout)
            d = re.search(r"dt_foot=([0-9.e-]+) s", r.stdout)
            print(f"foot es{es} {end:7s} dt={d.group(1) if d else '?':10s} rc={r.returncode} "
                  f"injected={m.group(1) if m else '?'} -> {out.name}")
            if r.returncode:
                print(r.stdout[-1500:], r.stderr[-1500:])
                sys.exit(2)
    accept_feet(args.es)


def accept_feet(es_list):
    """ACCEPTANCE: the `ref` rows reproduce the committed reference configurations, raw-uint64."""
    ok = True
    lines = []
    for es in es_list:
        p = foot_npz(es, "ref")
        if not p.exists():
            continue
        z = np.load(p)
        refs = []
        if es == 1:
            refs.append((f"committed base {BASE_STANCE.name}", load_base()["models"][FILL_FAMILY]))
        else:
            ex = EXAMPLES / f"{BASE_NAME}_es{es}_reference.toml"
            if not ex.is_file():
                sys.exit(f"{PROG}: feet acceptance needs the committed example {ex} -- STOP")
            with open(ex, "rb") as fh:
                refs.append((f"committed example {ex.name}", tomllib.load(fh)["models"][FILL_FAMILY]))
        if COMPARE_FEET_DIR is not None:
            kp = COMPARE_FEET_DIR / f"foot_es{es}_ref.npz"
            if kp.exists():
                refs.append((f"compare {kp.name}", {k: np.load(kp)[k] for k in FILL_KEYS}))
        for label, src in refs:
            for k in FILL_KEYS:
                d = int((u64(z[k]) != u64(src[k])).sum())
                lines.append(f"  es{es} ref {k:20s} vs {label:38s} {d} of {z[k].size} differ")
                ok = ok and d == 0
    print("feet acceptance (raw-uint64):")
    print("\n".join(lines))
    (FEET_DIR / "acceptance.txt").write_text("\n".join(lines) + f"\n\n{'PASS' if ok else 'FAIL'}\n")
    if not ok:
        sys.exit("feet acceptance FAILED -- STOP")
    print("feet acceptance: PASS")


# ----------------------------------------------------------------------------- gen
def fill_block_lines(base, es, end):
    """The restated initial_neutral_state block carrying the (es, end) foot."""
    z = np.load(foot_npz(es, end))
    members = dict(base["models"][FILL_FAMILY])
    lines = ["", f"[models.{FILL_FAMILY}]"]
    for k, v in members.items():
        if k in FILL_KEYS:
            lines.append(f"{k} = {toml_array(z[k])}")
        elif isinstance(v, list):
            lines.append(f"{k} = [" + ", ".join(toml_scalar(x) for x in v) + "]")
        else:
            lines.append(f"{k} = {toml_scalar(v)}")
    return lines


def registered_t0(name, es):
    """The arm's registered hand-off instant, or the rung's nominal until its probe lands."""
    p = paths(name, es)
    if p["reg"].exists():
        return float(json.loads(p["reg"].read_text())["t0_s"]), True
    return nominal_t0(es), False


def write_config(base, name, closure, deltas, end, es, t0, dest, header):
    rung = RUNGS[es]
    flat = {"cathode_solver_model": "prescribed_measured", "cathode_prescribed_trace_path": rung["trace"],
            "cathode_prescribed_t0_s": t0, "cathode_prescribed_start_s": t0}
    flags, none = {}, []
    if closure:
        c = CLOSURE_DELTAS[closure]
        flat.update(c["dict"]); flags.update(c["flags"]); none += c["none"]
    flat.update(deltas)
    lines = [header, f'base = "{BASE_NAME}"', "", "[input_dict]"] + [f"{k} = {toml_scalar(v)}" for k, v in flat.items()]
    if flags:
        lines += ["", "[input_flags]"] + [f"{k} = {toml_scalar(v)}" for k, v in flags.items()]
    if none:
        lines += ["", "[none_valued]", "input_dict = [" + ", ".join(f'"{k}"' for k in none) + "]"]
    carries_fill = (es != 1) or end != "ref"
    if carries_fill:
        lines += fill_block_lines(base, es, end)
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines) + "\n")
    return carries_fill


def write_arm_configs(base, name, closure, deltas, end, es):
    """The arm's final configuration (registered or nominal t0) and its t0-registration probe."""
    p = paths(name, es)
    t0, is_reg = registered_t0(name, es)
    tag = "registered" if is_reg else "nominal (probe pending)"
    cf = write_config(base, name, closure, deltas, end, es, t0, p["cfg"],
                      f"# ladder arm {p['id']} (generated {dt.date.today()}): derived from {BASE_NAME}; "
                      f"fill '{end}'; t0 = start = {t0!r} s, {tag}.")
    write_config(base, name, closure, deltas, end, es, PROBE_T0, p["pcfg"],
                 f"# ladder arm {p['id']} t0-REGISTRATION PROBE: the arm's configuration with the hand-off pushed to "
                 f"{PROBE_T0!r} s so the model reaches 1 kA on its own solve.")
    return cf, t0, is_reg


def expects(base, closure, deltas, es, t0):
    rung = RUNGS[es]
    e = [f"params:V_bank={rung['V_bank']}", "params:cathode_solver_model=prescribed_measured",
         f"params:cathode_prescribed_trace_path={rung['trace']}", f"params:cathode_prescribed_t0_s={t0!r}",
         f"params:cathode_prescribed_start_s={t0!r}"]
    if "Ts" in rung:
        e.append(f"params:cathode_Ts_base_K={rung['Ts']}")
    if closure:
        c = CLOSURE_DELTAS[closure]
        e += [f"params:{k}={toml_scalar(v).strip(chr(34))}" for k, v in c["dict"].items()]
        e += [f"flags:{k}={toml_scalar(v)}" for k, v in c["flags"].items()]
        e += [f"params:{k}=null" for k in c["none"]]
    flag_keys = set(base.get("input_flags", {}))
    for k, v in deltas.items():
        ns = "flags" if k in flag_keys else "params"
        e.append(f"{ns}:{k}={toml_scalar(v).strip(chr(34))}")
    return e


def preflight(name, closure, deltas, es, carries_fill, t0):
    p = paths(name, es)
    base = load_base()
    argv = [PY, "scripts/gates/preflight_diffcfg.py", "--stance", BASE_NAME]
    for x in expects(base, closure, deltas, es, t0):
        argv += ["--expect", x]
    argv += ["m6", "--", "--es", es, "--stance", p["cfg"].resolve(), "--sgp", sgp_for(name), "--save-h5", "/dev/null"]
    r = run_wt(argv, p["pre"])
    out = p["pre"].read_text()
    unexpected = set(re.findall(r"^\s*!!\s+\S+\s+(\S+?):", out, re.M))
    ctor_ok = "CONSTRUCTOR: OK" in out
    if carries_fill:
        ok = ctor_ok and unexpected == set(FILL_KEYS) and r.returncode in (0, 2)
        m = re.search(r"(\d+) discrepanc", out)
        ok = ok and m is not None and int(m.group(1)) == len(FILL_KEYS)
    else:
        ok = r.returncode == 0 and "PRE-FLIGHT: PASS" in out and ctor_ok and not unexpected
    for k in deltas:
        ok = ok and (k in out)
    # keep the log readable: truncate the two per-cell fill rows
    p["pre"].write_text(re.sub(r"(nn0(?:_annulus)?_profile: \[)[^\n]{200,}", r"\1... (array row truncated)]", out))
    if p["blocked"].exists():
        p["blocked"].unlink()
    if not ok:
        p["blocked"].write_text(f"preflight failed (rc={r.returncode}, unexpected={sorted(unexpected)}, ctor_ok={ctor_ok}); see {p['pre'].name}\n")
    return ok


def cmd_gen(args):
    base = load_base()
    n_ok = n_bad = 0
    for es in args.es:
        for name, closure, deltas, end in ARMS:
            if args.only and name not in args.only:
                continue
            cf, t0, is_reg = write_arm_configs(base, name, closure, deltas, end, es)
            ok = preflight(name, closure, deltas, es, cf, t0)
            n_ok += ok; n_bad += (not ok)
            print(f"{'ok     ' if ok else 'BLOCKED'} {arm_id(name, es):24s} fill={end:7s} "
                  f"t0={t0!r:9s} {'REG' if is_reg else 'nom'}")
    print(f"\n{n_ok} arms ready, {n_bad} blocked (see configs/*.BLOCKED and *.preflight.log)")


# ----------------------------------------------------------------------------- probe (t0 registration)
def probe_state(name, es):
    p = paths(name, es)
    if p["reg"].exists():
        return "registered"
    if not p["pcfg"].exists():
        return "nocfg"
    if p["pexit"].exists():
        return "harvest" if p["pexit"].read_text().strip() == "EXIT=0" and p["ph5"].exists() else "failed"
    if p["ppid"].exists():
        try:
            os.kill(int(p["ppid"].read_text().strip()), 0)
            return "running"
        except (ValueError, ProcessLookupError, PermissionError):
            return "stale"
    return "pending"


def launch_probe(name, es):
    p = paths(name, es)
    p["dir"].mkdir(parents=True, exist_ok=True)
    for f in (p["ph5"], p["plog"], p["pexit"], p["pstart"]):
        if f.exists():
            f.unlink()
    inner = (f"cd {shlex.quote(str(REPO))} && CABLP_COMPILED_KERNELS=1 PYTHONPATH={shlex.quote(str(REPO))} "
             f"PYTHONDONTWRITEBYTECODE=1 timeout --foreground {PROBE_TIMEOUT_S} {shlex.quote(str(PY))} scripts/run/run_m6_point.py "
             f"--stance {shlex.quote(str(p['pcfg'].resolve()))} --sgp {sgp_for(name)} --es {es} "
             f"--max-steps {PROBE_MAX_STEPS} --extra max_steps_action=stop "
             f"--save-h5 {shlex.quote(str(p['ph5']))} >> {shlex.quote(str(p['plog']))} 2>&1; "
             f"echo \"EXIT=$?\" > {shlex.quote(str(p['pexit']))}")
    p["pcmd"].write_text("#!/bin/bash\n# ladder t0 probe, generated " + dt.datetime.now().isoformat(timespec="seconds") + "\n" + inner + "\n")
    p["pstart"].write_text(dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") + "\n")
    with open(p["plog"], "ab") as log:
        proc = subprocess.Popen(["bash", str(p["pcmd"])], stdout=log, stderr=subprocess.STDOUT)
    p["ppid"].write_text(f"{proc.pid}\n")
    return proc


def harvest_probe(name, closure, deltas, end, es, base):
    """Read the probe's own 1 kA time, register t0, rewrite + re-pre-flight the arm's configuration."""
    import h5py
    p = paths(name, es)
    with h5py.File(p["ph5"], "r") as fh:
        t1ka = fh.attrs.get("t_breakdown_trigger", None)
        t_pre = fh.attrs.get("t_prebreakdown_trigger", None)
        t_fin = float(fh.attrs.get("final_time", np.nan))
        nsteps = int(fh.attrs.get("steps", -1))
    if t1ka is None or not np.isfinite(float(t1ka)):
        p["blocked"].write_text(f"t0 probe reached t={t_fin:.6g} s in {nsteps} steps without a 1 kA crossing; "
                                f"raise PROBE_MAX_STEPS (now {PROBE_MAX_STEPS})\n")
        return None
    t1ka = float(t1ka)
    t0 = grid_t0(t1ka)
    t_start = dt.datetime.fromisoformat(p["pstart"].read_text().strip().replace("Z", "+00:00"))
    wall = (dt.datetime.fromtimestamp(p["pexit"].stat().st_mtime, dt.timezone.utc) - t_start).total_seconds()
    reg = {"arm": p["id"], "probe_t0_s": PROBE_T0, "probe_max_steps": PROBE_MAX_STEPS,
           "probe_steps": nsteps, "probe_final_time_s": t_fin, "probe_wall_s": round(wall, 1),
           "t_breakdown_trigger_s": t1ka,
           "t_prebreakdown_trigger_s": (None if t_pre is None else float(t_pre)),
           "t0_s": t0, "grid_dt_s": SAVE_DT,
           "registered_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    p["reg"].write_text(json.dumps(reg, indent=2) + "\n")
    cf, t0b, _ = write_arm_configs(base, name, closure, deltas, end, es)
    ok = preflight(name, closure, deltas, es, cf, t0b)
    reg["preflight_ok"] = bool(ok)
    p["reg"].write_text(json.dumps(reg, indent=2) + "\n")
    return reg


def cmd_probe(args):
    base = load_base()
    queue = [(n, c, d, e, es) for es in args.es for n, c, d, e in ARMS if (not args.only or n in args.only)]
    while True:
        for n, c, d, e, es in queue:
            if probe_state(n, es) == "harvest":
                reg = harvest_probe(n, c, d, e, es, base)
                if reg:
                    print(f"[{dt.datetime.now():%H:%M:%S}] registered {arm_id(n, es):24s} "
                          f"t_1kA={reg['t_breakdown_trigger_s']:.6e} -> t0={reg['t0_s']!r} "
                          f"({reg['probe_wall_s']:.0f} s)", flush=True)
                else:
                    print(f"[{dt.datetime.now():%H:%M:%S}] BLOCKED {arm_id(n, es)}: no 1 kA crossing in the probe", flush=True)
        pending = [(n, es) for n, _c, _d, _e, es in queue if probe_state(n, es) in ("pending", "stale")]
        mine = [(n, es) for n, _c, _d, _e, es in queue if probe_state(n, es) == "running"]
        if not pending and not mine:
            print(f"[{dt.datetime.now():%H:%M:%S}] probe queue empty; done.", flush=True)
            return
        slots = args.lanes - live_solvers()
        for n, es in pending[:max(slots, 0)]:
            launch_probe(n, es)
            print(f"[{dt.datetime.now():%H:%M:%S}] probing {arm_id(n, es)}", flush=True)
            time.sleep(8)
        time.sleep(20)


# ----------------------------------------------------------------------------- run
def state(name, es):
    p = paths(name, es)
    if p["blocked"].exists():
        return "blocked"
    if not p["cfg"].exists():
        return "nocfg"
    if not p["reg"].exists():
        return "unreg"
    if p["exit"].exists():
        return "done" if p["exit"].read_text().strip() == "EXIT=0" and p["h5"].exists() else "failed"
    if p["pid"].exists():
        try:
            os.kill(int(p["pid"].read_text().strip()), 0)
            return "running"
        except (ValueError, ProcessLookupError, PermissionError):
            return "stale"
    return "pending"


def launch(name, es):
    p = paths(name, es)
    p["dir"].mkdir(parents=True, exist_ok=True)
    for f in (p["h5"], p["log"], p["exit"], p["start"]):
        if f.exists():
            f.unlink()
    inner = (f"cd {shlex.quote(str(REPO))} && CABLP_COMPILED_KERNELS=1 PYTHONPATH={shlex.quote(str(REPO))} "
             f"PYTHONDONTWRITEBYTECODE=1 timeout --foreground {TIMEOUT_S} {shlex.quote(str(PY))} scripts/run/run_m6_point.py "
             f"--stance {shlex.quote(str(p['cfg'].resolve()))} --sgp {sgp_for(name)} --es {es} "
             f"--max-steps {MAX_STEPS} --save-h5 {shlex.quote(str(p['h5']))} >> {shlex.quote(str(p['log']))} 2>&1; "
             f"echo \"EXIT=$?\" > {shlex.quote(str(p['exit']))}")
    p["cmd"].write_text("#!/bin/bash\n# ladder arm, generated " + dt.datetime.now().isoformat(timespec="seconds") + "\n" + inner + "\n")
    p["start"].write_text(dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") + "\n")
    with open(p["log"], "ab") as log:
        proc = subprocess.Popen(["bash", str(p["cmd"])], stdout=log, stderr=subprocess.STDOUT)
    p["pid"].write_text(f"{proc.pid}\n")
    return proc


#: The solver processes the lane cap covers: a python command line naming one
#: of these entry points or the solver class.
SOLVER_RE = re.compile(r"python.*(run_m6_point|run_sim1d|baseline_sim1d|golden_digest_gate|smoke_sim1d|"
                       r"verify_sim1d|audit_sim1d|_census|probe_|LAPDSim1D)")


def _own_tree():
    """This process and every ancestor, so a count never matches its own caller chain."""
    tree, pid = set(), os.getpid()
    while pid > 1:
        tree.add(pid)
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
        except OSError:
            break
        pid = int(stat.rsplit(")", 1)[1].split()[1])
    return tree


def live_solvers():
    """Count live solver processes machine-wide, excluding this process's own ancestor chain.

    A process counts when its command line matches ``SOLVER_RE`` and its
    ``argv[0]`` is a python interpreter, so a ``timeout`` or shell wrapper
    whose arguments name a solver is not counted beside the solver itself.
    """
    own = _own_tree()
    n = 0
    for d in Path("/proc").iterdir():
        if not d.name.isdigit() or int(d.name) in own:
            continue
        try:
            argv = (d / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        argv = [a.decode(errors="replace") for a in argv if a]
        if not argv or "python" not in Path(argv[0]).name:
            continue
        if SOLVER_RE.search(" ".join(argv)):
            n += 1
    return n


def cmd_run(args):
    queue = [(n, es) for es in args.es for n, *_ in ARMS if (not args.only or n in args.only)]
    while True:
        pending = [(n, es) for n, es in queue if state(n, es) in ("pending", "stale")]
        mine = [(n, es) for n, es in queue if state(n, es) == "running"]
        if not pending and not mine:
            print(f"[{dt.datetime.now():%H:%M:%S}] queue empty; done.", flush=True)
            return
        slots = args.lanes - live_solvers()
        for n, es in pending[:max(slots, 0)]:
            launch(n, es)
            print(f"[{dt.datetime.now():%H:%M:%S}] launched {arm_id(n, es)}", flush=True)
            time.sleep(8)
        time.sleep(30)


# ----------------------------------------------------------------------------- registration check
def registration_check(name, es):
    """(c) of the registration: the run's own origin, 1 kA time and hand-off jump against the registered t0."""
    import h5py
    p = paths(name, es)
    if not (p["h5"].exists() and p["reg"].exists()):
        return None
    reg = json.loads(p["reg"].read_text())
    t0 = float(reg["t0_s"])
    with h5py.File(p["h5"], "r") as fh:
        t = fh["time"][...]
        ph = np.asarray([x.decode() if isinstance(x, bytes) else x for x in fh["phase"][...]])
        t1ka = float(fh.attrs.get("t_breakdown_trigger", np.nan))
        jump = float(fh.attrs.get("cathode_prescribed_handoff_relative_jump", np.nan))
        hand = float(fh.attrs.get("cathode_prescribed_handoff_time_s", np.nan))
        cfg_t0 = float(fh.attrs.get("cathode_prescribed_t0_s", np.nan))
    hits = np.flatnonzero(ph == "main_discharge")
    origin = float(t[hits[0]]) if hits.size else float("nan")
    ulp = abs(np.float64(origin).view(np.int64) - np.float64(t0).view(np.int64)) if np.isfinite(origin) else -1
    out = {"arm": p["id"], "t0_s": t0, "config_t0_s": cfg_t0, "origin_s": origin, "origin_minus_t0_ulp": int(ulp),
           "t_breakdown_trigger_s": t1ka, "probe_t_breakdown_trigger_s": reg["t_breakdown_trigger_s"],
           "handoff_time_s": hand, "handoff_relative_jump": jump,
           "probe_wall_s": reg.get("probe_wall_s"),
           "origin_ok": bool(ulp >= 0 and ulp <= 1),
           "window_ok": bool(np.isfinite(t1ka) and t1ka <= t0 < t1ka + SAVE_DT)}
    out["PASS"] = bool(out["origin_ok"] and out["window_ok"])
    p["check"].write_text(json.dumps(out, indent=2) + "\n")
    return out


# ----------------------------------------------------------------------------- score / census / status / tabulate
def cmd_score(args):
    for es in args.es:
        for name, *_ in ARMS:
            if args.only and name not in args.only:
                continue
            p = paths(name, es)
            if state(name, es) != "done" or (p["score_pl"].exists() and not args.force):
                continue
            run_wt([PY, "scripts/score/compare_sim1d_es1.py", "--from-h5", p["h5"], "--es", es,
                    "--json", p["score_json"]], p["score"])
            r = run_wt([PY, "scripts/score/compare_sim1d_es1.py", "--from-h5", p["h5"], "--es", es,
                        "--window", "plateau", "--json", p["score_pl_json"]], p["score_pl"])
            chk = registration_check(name, es)
            print(f"scored {p['id']} rc={r.returncode} registration="
                  f"{'PASS' if chk and chk['PASS'] else 'FAIL' if chk else '-'}")


CENSUS_VERDICT = {0: "PASS", 2: "DID NOT RUN"}


def cmd_census(args):
    """Census every finished arm; exit 1 if any FAILED, else 2 if any DID NOT RUN.

    The census exits 0 PASS, 1 FAIL, 2 DID NOT RUN (the record has no
    afterglow, or ends before the assertion time without a hand-off); any
    other nonzero status is reported as FAIL.
    """
    failed = did_not_run = False
    for es in args.es:
        for name, *_ in ARMS:
            if args.only and name not in args.only:
                continue
            p = paths(name, es)
            if state(name, es) != "done" or (p["census"].exists() and not args.force):
                continue
            r = run_wt([PY, "scripts/verify/census_afterglow_tail_handoff.py", "--from-h5", p["h5"],
                        "--assert-no-handoff-before-ms", "21.5"], p["census"])
            verdict = CENSUS_VERDICT.get(r.returncode, "FAIL")
            failed |= verdict == "FAIL"
            did_not_run |= verdict == "DID NOT RUN"
            print(f"census {p['id']} rc={r.returncode} {verdict}")
    if failed:
        sys.exit(1)
    if did_not_run:
        sys.exit(2)


def cmd_status(args):
    now = dt.datetime.now(dt.timezone.utc)
    counts = {}
    for es in args.es:
        for name, *_ in ARMS:
            if args.only and name not in args.only:
                continue
            p = paths(name, es); st = state(name, es); pst = probe_state(name, es)
            counts[st] = counts.get(st, 0) + 1
            extra = ""
            if p["start"].exists():
                t0 = dt.datetime.fromisoformat(p["start"].read_text().strip().replace("Z", "+00:00"))
                t1 = dt.datetime.fromtimestamp(p["exit"].stat().st_mtime, dt.timezone.utc) if p["exit"].exists() else now
                extra = f"  {(t1 - t0).total_seconds() / 60:6.1f} min"
            reg = json.loads(p["reg"].read_text()) if p["reg"].exists() else None
            rt = f"  t0={reg['t0_s']!r:9s} t1kA={reg['t_breakdown_trigger_s']:.5e}" if reg else "  t0=(unregistered)"
            chk = json.loads(p["check"].read_text()) if p["check"].exists() else None
            ck = (f"  origin{'==' if chk['origin_ok'] else '!='}t0 ulp={chk['origin_minus_t0_ulp']} "
                  f"jump={chk['handoff_relative_jump']:+.3f} {'PASS' if chk['PASS'] else 'FAIL'}") if chk else ""
            print(f"{p['id']:24s} {st:8s} probe:{pst:10s}{rt}{extra}{ck}")
    print("\n" + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))


PL_RE = re.compile(r"^\s*(Te|n|Isat) plateau \(15-19\.5 ms\): mean ratio ([0-9.]+), mean rms rel ([0-9.]+), mean \|dev\|/sig ([0-9.]+)", re.M)


def plateau_rows(p):
    if not p["score_pl"].exists():
        return None
    d = {f: (r, s) for f, r, _, s in PL_RE.findall(p["score_pl"].read_text())}
    return d if len(d) == 3 else None


def cmd_tabulate(args):
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    groups = [("ref", ["ref"]), (f"kinetic band ({len(KIN_ARMS) - 1})", [n for n, *_ in KIN_ARMS if n != "ref"]),
              ("fluid closure (2)", ["diffusive", "moment"]),
              ("fluid band (diffusive)", ["diffusive"] + [f"diffusive_{n}" for n, *_ in FLUID_BANDS["diffusive"]]),
              ("fluid band (moment)", ["moment"] + [f"moment_{n}" for n, *_ in FLUID_BANDS["moment"]])]
    for es in args.es:
        lines = [f"# ES{es} plateau (15-19.5 ms) stage-(ii) summary — closure ladder", "",
                 "Per-field plateau mean ratio (arm/measured) and mean |dev|/sig, verbatim from the scorer's own "
                 "`<field> plateau (15-19.5 ms)` summary line (`--window plateau`), mean over the rung's ports.", "",
                 "| group | arm | Te ratio_pl | Te devsig_pl | n ratio_pl | n devsig_pl | Isat ratio_pl | Isat devsig_pl |",
                 "|---|---|---|---|---|---|---|---|"]
        for g, names in groups:
            for n in names:
                p = paths(n, es); d = plateau_rows(p)
                cells = " | ".join(f"{d[f][0]} | {d[f][1]}" for f in ("Te", "n", "Isat")) if d else "— | — | — | — | — | —"
                lines.append(f"| {g} | {p['id']} | {cells} |")
        # envelopes: min/max of ratio_pl over each band
        lines += ["", "## Band envelopes (min–max of the plateau mean ratio over the band's arms, the ref/toggle arm included)", "",
                  "| band | Te | n | Isat |", "|---|---|---|---|"]
        for g, names in groups[1:]:
            rows = [plateau_rows(paths(n, es)) for n in (["ref"] + names if g.startswith("kinetic") else names)]
            rows = [r for r in rows if r]
            if not rows:
                continue
            env = " | ".join(f"{min(float(r[f][0]) for r in rows):.2f}–{max(float(r[f][0]) for r in rows):.2f}" for f in ("Te", "n", "Isat"))
            lines.append(f"| {g} | {env} |")
        (TAB_DIR / f"es{es}_plateau_summary.md").write_text("\n".join(lines) + "\n")
        print(f"wrote {TAB_DIR / f'es{es}_plateau_summary.md'}")
        # per-arm t0 registration table
        rl = [f"# ES{es} per-arm t0 registration", "",
              "`probe t_1kA` is the model's own circuit-on → 1 kA time read from the registration probe (the arm's "
              "configuration with the hand-off pushed to 0.5 ms); `t0` is that time rounded UP to the 10 µs save "
              "grid and written into the arm as `cathode_prescribed_t0_s` = `cathode_prescribed_start_s`; `origin` "
              "is the scorer's main-discharge origin on the finished run; `jump` is the hand-off relative jump.", "",
              "| arm | fill | probe t_1kA [ms] | t0 [ms] | probe wall [s] | run t_1kA [ms] | origin [ms] | origin−t0 [ulp] | jump | check |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for name, _c, _d, end in ARMS:
            p = paths(name, es)
            reg = json.loads(p["reg"].read_text()) if p["reg"].exists() else None
            chk = json.loads(p["check"].read_text()) if p["check"].exists() else None
            if reg is None:
                rl.append(f"| {p['id']} | {end} | — | — | — | — | — | — | — | — |")
                continue
            c = (f"{chk['t_breakdown_trigger_s'] * 1e3:.5f} | {chk['origin_s'] * 1e3:.5f} | "
                 f"{chk['origin_minus_t0_ulp']} | {chk['handoff_relative_jump']:+.4f} | "
                 f"{'PASS' if chk['PASS'] else 'FAIL'}") if chk else "— | — | — | — | —"
            rl.append(f"| {p['id']} | {end} | {reg['t_breakdown_trigger_s'] * 1e3:.5f} | {reg['t0_s'] * 1e3:.3f} | "
                      f"{reg.get('probe_wall_s', '—')} | {c} |")
        (TAB_DIR / f"es{es}_t0_registration.md").write_text("\n".join(rl) + "\n")
        print(f"wrote {TAB_DIR / f'es{es}_t0_registration.md'}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, needs_stance):
        p.add_argument("--root", required=True,
                       help="the ladder tree (configs/, runs/, feet/, base/, tabulate/); outside the repository")
        p.add_argument("--repo", default=str(THIS_CHECKOUT),
                       help="the checkout every child process runs in (default: this file's checkout)")
        if needs_stance:
            g = p.add_mutually_exclusive_group()
            g.add_argument("--stance", metavar="NAME_OR_PATH", default=None,
                           help="the committed base configuration every arm derives from")
            g.add_argument("--no-stance", action="store_true",
                           help="refused: every arm derives from a named base")
        p.set_defaults(needs_stance=needs_stance)

    def es_arg(p):
        p.add_argument("--es", type=int, nargs="+", default=[1])
        p.add_argument("--only", nargs="*", default=None)

    b0 = sub.add_parser("base"); common(b0, True)
    b0.add_argument("--compare-base-dir", default=None,
                    help="another ladder tree's base/ to compare the new base against, raw-uint64")
    b0.set_defaults(fn=cmd_base)
    f = sub.add_parser("feet"); common(f, True); es_arg(f)
    f.add_argument("--compare-feet-dir", default=None,
                   help="a directory of foot_es<N>_ref.npz the ref rows must also reproduce, raw-uint64")
    f.set_defaults(fn=cmd_feet)
    g = sub.add_parser("gen"); common(g, True); es_arg(g); g.set_defaults(fn=cmd_gen)
    b = sub.add_parser("probe"); common(b, True); es_arg(b)
    b.add_argument("--lanes", type=int, default=2); b.set_defaults(fn=cmd_probe)
    r = sub.add_parser("run"); common(r, False); es_arg(r)
    r.add_argument("--lanes", type=int, default=4); r.set_defaults(fn=cmd_run)
    for nm, fn in (("score", cmd_score), ("census", cmd_census)):
        s = sub.add_parser(nm); common(s, False); es_arg(s)
        s.add_argument("--force", action="store_true"); s.set_defaults(fn=fn)
    s = sub.add_parser("status"); common(s, False); es_arg(s); s.set_defaults(fn=cmd_status)
    t = sub.add_parser("tabulate"); common(t, False); es_arg(t); t.set_defaults(fn=cmd_tabulate)
    a = ap.parse_args(argv)
    _configure(a)
    a.fn(a)


if __name__ == "__main__":
    main()
