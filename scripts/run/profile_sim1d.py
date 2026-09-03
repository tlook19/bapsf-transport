"""Detailed profiling harness for ``LAPDSim1D`` at the PRODUCTION configuration.

This is an *instrument*, not a gate: it runs the real production stance and
records where the wall clock goes, so the cost structure of a campaign run can
be analysed later without re-running it.

Config authority
----------------
The configuration is IMPORTED from ``compare_sim1d_es1.run_model`` (the same
no-drift rule ``baseline_sim1d.py`` follows), never reimplemented here.  That
function builds ``default_config()`` plus ``PARAM_OVERRIDES``/``FLAG_OVERRIDES``,
and ``PARAM_OVERRIDES`` reads its values from the STANCE OF RECORD,
``scripts/stances/g1atrim.toml``.  The default invocation is therefore::

    run_model(nx=PRODUCTION_NX, exchange_model="knudsen",
              extra={"tau_afterglow": PRODUCTION_TAU_AFTERGLOW})

which profiles the g1atrim stance's shared production package -- circuit, rate
model, numerics -- at the stance's OWN mesh.  The two keyword values supply
exactly the keys ``PARAM_OVERRIDES`` deliberately does NOT import from the
stance, because they are grid-coupled or run-cost settings that every
``run_model`` caller passes for itself; both are read from
``scripts/stances/g1atrim.toml`` here rather than transcribed, so they cannot
drift from it: ``nx = 268`` and ``tau_afterglow = 0.006`` (the latter against a
config default of 5e-3).

Note that this module's ``PRODUCTION_NX`` is the STANCE's nx, which is NOT
``compare_sim1d_es1.PRODUCTION_NX`` -- that name is the no-stance FALLBACK mesh
(240), which this instrument previously defaulted to, profiling the production
package on a mesh the production stance does not run.  Costs that scale with
cell count are therefore reported against 268.  Pass ``--nx 240`` to reproduce
the older fallback-mesh profiles.

Two profilers, because they answer different questions
------------------------------------------------------
``--mode sample`` (default)
    An in-process statistical sampler: a daemon thread snapshots the solver
    thread's Python stack at a fixed rate and folds the stacks into counts.
    Overhead is well under 1 %, so the wall-clock attribution it reports is
    the *undistorted* cost of a genuine production run.  Answers "where does
    the time actually go".

    (py-spy would be the usual tool, but it requires root on macOS.  The
    in-process sampler needs no privileges and buys something py-spy cannot
    give: every sample is tagged with the solver's own phase, so cost can be
    split across equilibration / pre-breakdown / main discharge / afterglow.)

``--mode cprofile``
    Deterministic instrumentation: exact call counts and per-call costs.
    Inflates the runtime substantially and penalises many-small-call functions,
    so its *timings* must not be read as production timings -- but its CALL
    COUNTS are exact, which sampling can never provide.  Answers "how many
    times per step is this called, and what does one call cost".

Read the two together: the sampler ranks the hot spots honestly, cProfile
explains whether a hot spot is one expensive call or a million cheap ones.

Outputs (all under ``--out-dir``, prefixed with ``--label``)
-----------------------------------------------------------
``<label>_meta.json``           run metadata: config, git commit, platform,
                                wall time, step count, throughput
``<label>_phase_trace.csv``     wall/sim-time/step/phase samples over the run
``<label>_folded.txt``          folded stacks ``frame;frame;... count``
                                (sample mode; the archival artifact -- every
                                other view can be regenerated from it, and it
                                loads directly in flamegraph/speedscope tools)
``<label>_folded_<phase>.txt``  the same, restricted to one solver phase
``<label>_sample_report.txt``   ranked self/total time tables (sample mode)
``<label>.prof``                binary ``pstats`` dump (cprofile mode)
``<label>_cprofile_*.txt``      ranked tottime/cumtime tables + callers

Usage::

    # honest production timing (costs one production run)
    python scripts/profile_sim1d.py --mode sample --label prod_sample_nx240

    # exact call counts (slower; timings are distorted by design)
    python scripts/profile_sim1d.py --mode cprofile --label prod_cprof_nx240

Artifacts are run outputs: they live in ``scripts/`` and are never
committed.
"""

import argparse
import cProfile
import io
import json
import platform
import pstats
import subprocess
import sys
import threading
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter

SCRIPT_DIR = Path(__file__).resolve().parents[1]
# scripts/ sibling imports: the seven purpose subdirectories on sys.path.
import sys as _sys
from pathlib import Path as _Path
for _sub in ("atomic", "gates", "kinetic", "run", "score", "stance",
             "verify"):
    _dir = str(_Path(__file__).resolve().parents[1] / _sub)
    if _dir not in _sys.path:
        _sys.path.insert(0, _dir)

from cablp.solvers._sim1d import LAPDSim1D  # noqa: E402
from cablp.solvers._sim1d.results.io import save_result_hdf5  # noqa: E402
from compare_sim1d_es1 import (  # noqa: E402
    PRODUCTION_STANCE,
    load_stance,
    run_model,
)

# The two keys compare_sim1d_es1.PARAM_OVERRIDES deliberately does NOT import
# from the stance -- its grid-coupled mesh and its run-cost budget, both left to
# each run_model caller -- are READ FROM THE STANCE FILE here rather than
# hand-copied, so this instrument cannot drift from
# scripts/stances/g1atrim.toml the way a transcribed literal can.
_STANCE_PARAMS = load_stance(PRODUCTION_STANCE).params

#: The MESH OF RECORD: the stance's own nx (g1atrim: 268). NOT
#: compare_sim1d_es1.PRODUCTION_NX, which is the no-stance FALLBACK mesh (240)
#: and which this instrument used to take by default -- profiling the production
#: package on a mesh the production stance does not use.
PRODUCTION_NX = int(_STANCE_PARAMS["nx"])

#: The stance of record's afterglow budget (g1atrim: 0.006, against a config
#: default of 5e-3).
PRODUCTION_TAU_AFTERGLOW = float(_STANCE_PARAMS["tau_afterglow"])

# Repo root, used only to shorten frame filenames in the folded stacks.
_REPO_ROOT = SCRIPT_DIR.parent


# --------------------------------------------------------------------------
# In-process statistical sampler
# --------------------------------------------------------------------------


# Frame labels are built once per code object and cached.  This matters: the
# sampler walks a stack tens of frames deep at every tick, so anything per-frame
# that touches the filesystem (an earlier version called Path.resolve here)
# costs seconds per sample and silently destroys the sampling rate.
_LABEL_CACHE = {}
_REPO_ROOT_STR = str(_REPO_ROOT) + "/"


def _frame_label(code):
    """Compact, stable frame label: ``qualname (path:firstline)``.

    Pure string work, no filesystem access, memoised on the code object.
    """
    label = _LABEL_CACHE.get(code)
    if label is not None:
        return label
    filename = code.co_filename
    if filename.startswith(_REPO_ROOT_STR):
        short = filename[len(_REPO_ROOT_STR):]
    else:
        # Stdlib / site-packages / <string>: keep the last two components only,
        # which is enough to tell numpy's linalg from our own code.
        parts = filename.rsplit("/", 2)
        short = "/".join(parts[-2:]) if len(parts) >= 2 else filename
    label = f"{code.co_qualname} ({short}:{code.co_firstlineno})"
    _LABEL_CACHE[code] = label
    return label


class StackSampler:
    """Sample one thread's Python stack at a fixed rate, folding into counts.

    The sampler runs in a daemon thread and reads ``sys._current_frames()``.
    When the target thread is inside a C call (numpy, scipy), the frame we see
    is the Python function that made that call -- which is exactly the
    attribution we want.

    Sampling is subject to the GIL: the sampler can only run when the target
    releases it, so at very high rates the effective rate falls below the
    requested one.  ``achieved_hz`` in the metadata reports what was actually
    obtained; treat a large shortfall as a caveat on fine-grained attribution,
    not on the broad ranking.
    """

    def __init__(self, target_tid, hz=100.0, phase_ref=None):
        self.target_tid = target_tid
        self.interval = 1.0 / float(hz)
        self.requested_hz = float(hz)
        self._phase_ref = phase_ref if phase_ref is not None else ["?"]
        self._stop = threading.Event()
        self._thread = None
        # folded stack -> sample count, overall and per solver phase
        self.counts = Counter()
        self.phase_counts = defaultdict(Counter)
        self.phase_samples = Counter()
        self.n_samples = 0
        self.n_missed = 0
        self.wall_sampling = 0.0

    def _capture(self):
        frame = sys._current_frames().get(self.target_tid)
        if frame is None:
            self.n_missed += 1
            return
        stack = []
        while frame is not None:
            stack.append(_frame_label(frame.f_code))
            frame = frame.f_back
        stack.reverse()
        key = ";".join(stack)
        phase = self._phase_ref[0]
        self.counts[key] += 1
        self.phase_counts[phase][key] += 1
        self.phase_samples[phase] += 1
        self.n_samples += 1

    def _run(self):
        t0 = perf_counter()
        next_tick = t0
        while not self._stop.is_set():
            next_tick += self.interval
            delay = next_tick - perf_counter()
            if delay > 0:
                if self._stop.wait(delay):
                    break
            else:
                # Fell behind (GIL contention or a slow capture): resync rather
                # than trying to catch up, which would burst-sample and bias.
                next_tick = perf_counter()
            self._capture()
        self.wall_sampling = perf_counter() - t0

    def start(self):
        self._thread = threading.Thread(
            target=self._run, name="stack-sampler", daemon=True
        )
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)

    @property
    def achieved_hz(self):
        if self.wall_sampling <= 0.0:
            return 0.0
        return self.n_samples / self.wall_sampling


# --------------------------------------------------------------------------
# Folded-stack reporting
# --------------------------------------------------------------------------


def write_folded(counts, path):
    """Write folded stacks, hottest first (``frame;frame;... count``)."""
    with open(path, "w") as fh:
        for stack, count in counts.most_common():
            fh.write(f"{stack} {count}\n")


def _self_and_total(counts):
    """Split folded stacks into leaf (self) and any-frame (total) tallies.

    A frame counts once per sample toward ``total`` even if it appears several
    times in the stack (recursion), so the totals stay interpretable as
    "fraction of wall time spent below this frame".
    """
    self_counts = Counter()
    total_counts = Counter()
    for stack, count in counts.items():
        frames = stack.split(";")
        if not frames:
            continue
        self_counts[frames[-1]] += count
        for frame in set(frames):
            total_counts[frame] += count
    return self_counts, total_counts


def sample_report(sampler, wall_s, top, stream):
    total = sampler.n_samples
    if total == 0:
        print("no samples captured", file=stream)
        return
    sec_per_sample = wall_s / total

    def _table(title, tally):
        print(f"\n{title}", file=stream)
        print(f"{'%wall':>7}  {'est_s':>9}  {'samples':>8}  frame", file=stream)
        print("-" * 100, file=stream)
        for frame, count in tally.most_common(top):
            pct = 100.0 * count / total
            print(
                f"{pct:7.2f}  {count * sec_per_sample:9.1f}  {count:8d}  {frame}",
                file=stream,
            )

    print("=" * 100, file=stream)
    print("SAMPLING PROFILE (in-process stack sampler)", file=stream)
    print("=" * 100, file=stream)
    print(f"wall time            : {wall_s:.1f} s", file=stream)
    print(f"samples              : {total}", file=stream)
    print(
        f"rate requested/achieved: {sampler.requested_hz:.1f} Hz / "
        f"{sampler.achieved_hz:.1f} Hz",
        file=stream,
    )
    print(f"seconds per sample   : {sec_per_sample * 1e3:.3f} ms", file=stream)
    print(f"missed captures      : {sampler.n_missed}", file=stream)

    print("\nWALL TIME BY SOLVER PHASE", file=stream)
    print(f"{'%wall':>7}  {'est_s':>9}  {'samples':>8}  phase", file=stream)
    print("-" * 60, file=stream)
    for phase, count in sampler.phase_samples.most_common():
        pct = 100.0 * count / total
        print(
            f"{pct:7.2f}  {count * sec_per_sample:9.1f}  {count:8d}  {phase}",
            file=stream,
        )

    self_counts, total_counts = _self_and_total(sampler.counts)
    _table(f"TOP {top} BY SELF TIME (leaf frame -- where the CPU actually is)",
           self_counts)
    _table(f"TOP {top} BY TOTAL TIME (frame anywhere on the stack)",
           total_counts)

    for phase, counts in sorted(
        sampler.phase_counts.items(),
        key=lambda kv: -sampler.phase_samples[kv[0]],
    ):
        n_phase = sampler.phase_samples[phase]
        if n_phase < max(20, 0.01 * total):
            continue
        phase_self, _ = _self_and_total(counts)
        print(
            f"\n\nPHASE {phase!r}: top {top} by self time "
            f"({n_phase} samples, {n_phase * sec_per_sample:.1f} s)",
            file=stream,
        )
        print(f"{'%phase':>7}  {'est_s':>9}  {'samples':>8}  frame", file=stream)
        print("-" * 100, file=stream)
        for frame, count in phase_self.most_common(top):
            print(
                f"{100.0 * count / n_phase:7.2f}  "
                f"{count * sec_per_sample:9.1f}  {count:8d}  {frame}",
                file=stream,
            )


# --------------------------------------------------------------------------
# Phase trace (read-only progress instrumentation)
# --------------------------------------------------------------------------


class PhaseTracer:
    """Record (wall, sim time, step, phase, dt) on a throttled cadence.

    Installed as a ``progress_tracker``; it only reads the progress object, so
    it cannot affect the physics.  The throttle keeps the per-step cost to a
    ``perf_counter`` and a comparison.
    """

    def __init__(self, phase_ref, every_s=1.0):
        self.phase_ref = phase_ref
        self.every_s = every_s
        self.rows = []
        self.t0 = perf_counter()
        self._last = -1e18
        self.max_step = 0

    def __call__(self, progress):
        phase = getattr(progress, "phase", "?")
        self.phase_ref[0] = phase
        step = int(getattr(progress, "step", 0) or 0)
        self.max_step = max(self.max_step, step)
        now = perf_counter()
        if now - self._last < self.every_s:
            return
        self._last = now
        self.rows.append(
            {
                "wall_s": now - self.t0,
                "step": step,
                "sim_time_s": float(getattr(progress, "time", float("nan"))),
                "fraction": float(getattr(progress, "fraction", float("nan"))),
                "phase": phase,
                "accepted_dt": float(
                    getattr(progress, "accepted_dt", float("nan"))
                ),
            }
        )

    def write_csv(self, path):
        cols = ["wall_s", "step", "sim_time_s", "fraction", "phase", "accepted_dt"]
        with open(path, "w") as fh:
            fh.write(",".join(cols) + "\n")
            for row in self.rows:
                fh.write(",".join(str(row[c]) for c in cols) + "\n")


def install_progress_tracker(tracer):
    """Patch ``start_simulation`` to carry our tracker; return an undo hook.

    ``run_model`` calls ``start_simulation`` with a fixed argument list, so a
    shim is the only way to attach instrumentation without forking the
    production driver.  Mirrors ``run_kn2z_promoted.py`` (at commit 48be9a4,
    retired 2026-09-03).
    """
    original = LAPDSim1D.start_simulation

    def patched(self, *args, **kwargs):
        kwargs.setdefault("progress_tracker", tracer)
        kwargs.setdefault("progress_interval_s", 0.0)
        return original(self, *args, **kwargs)

    LAPDSim1D.start_simulation = patched

    def undo():
        LAPDSim1D.start_simulation = original

    return undo


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def _git_commit():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _run_production(nx, tau_afterglow, exchange_model, t_end, extra_pairs):
    extra = {"tau_afterglow": tau_afterglow}
    for pair in extra_pairs or ():
        key, _, raw = pair.partition("=")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        extra[key] = value
    return run_model(
        nx=nx,
        exchange_model=exchange_model,
        extra=extra,
        t_end=t_end,
    )


def _write_cprofile_reports(profiler, out_dir, label, top):
    prof_path = out_dir / f"{label}.prof"
    profiler.dump_stats(str(prof_path))

    written = [prof_path]
    for sort_key, name in (("tottime", "tottime"), ("cumulative", "cumtime")):
        buf = io.StringIO()
        stats = pstats.Stats(profiler, stream=buf)
        stats.sort_stats(sort_key).print_stats(top)
        path = out_dir / f"{label}_cprofile_{name}.txt"
        path.write_text(buf.getvalue())
        written.append(path)

    # Callers of the hottest self-time functions: turns "this is expensive"
    # into "this is expensive AND here is who keeps calling it".
    buf = io.StringIO()
    stats = pstats.Stats(profiler, stream=buf)
    stats.sort_stats("tottime").print_callers(min(top, 40))
    path = out_dir / f"{label}_cprofile_callers.txt"
    path.write_text(buf.getvalue())
    written.append(path)
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("sample", "cprofile", "plain"),
        default="sample",
        help="sample: low-overhead wall-clock attribution (default). "
        "cprofile: exact call counts, distorted timings. "
        "plain: no profiler (control run for the overhead figure).",
    )
    parser.add_argument("--label", default=None, help="artifact filename prefix")
    parser.add_argument("--out-dir", default=str(SCRIPT_DIR))
    parser.add_argument("--nx", type=int, default=PRODUCTION_NX)
    parser.add_argument(
        "--tau-afterglow", type=float, default=PRODUCTION_TAU_AFTERGLOW
    )
    parser.add_argument("--exchange-model", default="knudsen")
    parser.add_argument(
        "--t-end",
        type=float,
        default=None,
        help="cap run length WITHOUT deforming the drive (run_model semantics); "
        "omit for the full production trajectory",
    )
    parser.add_argument("--sample-hz", type=float, default=100.0)
    parser.add_argument(
        "--switch-interval",
        type=float,
        default=None,
        help="sys.setswitchinterval for the run (sample mode). The sampler can "
        "only capture when the solver thread yields the GIL; shortening the "
        "interval raises the achieved rate at the cost of more GIL churn.",
    )
    parser.add_argument("--top", type=int, default=60)
    parser.add_argument("--trace-every-s", type=float, default=1.0)
    parser.add_argument(
        "--extra",
        nargs="*",
        default=(),
        help="extra params as key=value (JSON-parsed values)",
    )
    parser.add_argument("--save-h5", default=None, help="also save the trajectory")
    args = parser.parse_args(argv)

    label = args.label or f"prod_{args.mode}_nx{args.nx}"
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"profile_sim1d: mode={args.mode} label={label}", flush=True)
    print(
        f"config: nx={args.nx} exchange={args.exchange_model} "
        f"tau_afterglow={args.tau_afterglow} t_end={args.t_end} "
        f"extra={list(args.extra)}",
        flush=True,
    )
    print(f"out_dir: {out_dir}", flush=True)

    if args.switch_interval is not None:
        sys.setswitchinterval(args.switch_interval)
        print(f"switchinterval set to {sys.getswitchinterval()}", flush=True)

    phase_ref = ["startup"]
    tracer = PhaseTracer(phase_ref, every_s=args.trace_every_s)
    undo = install_progress_tracker(tracer)

    sampler = None
    profiler = None
    if args.mode == "sample":
        sampler = StackSampler(
            threading.get_ident(), hz=args.sample_hz, phase_ref=phase_ref
        )
    elif args.mode == "cprofile":
        profiler = cProfile.Profile()

    wall_start = perf_counter()
    try:
        if sampler is not None:
            sampler.start()
        if profiler is not None:
            profiler.enable()
        try:
            result, geometry, params, flags = _run_production(
                args.nx,
                args.tau_afterglow,
                args.exchange_model,
                args.t_end,
                args.extra,
            )
        finally:
            if profiler is not None:
                profiler.disable()
            if sampler is not None:
                sampler.stop()
    finally:
        undo()
    wall_s = perf_counter() - wall_start

    n_steps = tracer.max_step
    print(
        f"\nSOLVE DONE wall={wall_s:.1f}s steps={n_steps} "
        f"steps/s={n_steps / max(wall_s, 1e-9):.2f}",
        flush=True,
    )

    written = []

    trace_path = out_dir / f"{label}_phase_trace.csv"
    tracer.write_csv(trace_path)
    written.append(trace_path)

    if args.save_h5 is not None:
        save_result_hdf5(args.save_h5, result, params=params, flags=flags)
        print(f"saved result to {args.save_h5}", flush=True)

    if sampler is not None:
        folded = out_dir / f"{label}_folded.txt"
        write_folded(sampler.counts, folded)
        written.append(folded)
        for phase, counts in sampler.phase_counts.items():
            safe = "".join(c if c.isalnum() else "_" for c in phase)
            path = out_dir / f"{label}_folded_{safe}.txt"
            write_folded(counts, path)
            written.append(path)

        report_path = out_dir / f"{label}_sample_report.txt"
        buf = io.StringIO()
        sample_report(sampler, wall_s, args.top, buf)
        report_path.write_text(buf.getvalue())
        written.append(report_path)
        print(buf.getvalue(), flush=True)

    if profiler is not None:
        written.extend(_write_cprofile_reports(profiler, out_dir, label, args.top))
        buf = io.StringIO()
        stats = pstats.Stats(profiler, stream=buf)
        stats.sort_stats("tottime").print_stats(min(args.top, 40))
        print(buf.getvalue(), flush=True)

    meta = {
        "label": label,
        "mode": args.mode,
        "wall_s": wall_s,
        "steps": n_steps,
        "steps_per_s": n_steps / max(wall_s, 1e-9),
        "final_sim_time_s": float(result.time[-1]) if len(result.time) else None,
        "git_commit": _git_commit(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "config": {
            "nx": args.nx,
            "exchange_model": args.exchange_model,
            "tau_afterglow": args.tau_afterglow,
            "t_end": args.t_end,
            "extra": list(args.extra),
            "source": "compare_sim1d_es1.run_model (imported, no drift)",
        },
        "sampler": (
            {
                "requested_hz": sampler.requested_hz,
                "achieved_hz": sampler.achieved_hz,
                "samples": sampler.n_samples,
                "missed": sampler.n_missed,
                "phase_samples": dict(sampler.phase_samples),
            }
            if sampler is not None
            else None
        ),
        "artifacts": [str(p.name) for p in written],
    }
    meta_path = out_dir / f"{label}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
    written.append(meta_path)

    print("\nARTIFACTS", flush=True)
    for path in written:
        print(f"  {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
