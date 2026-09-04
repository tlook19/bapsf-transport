"""NO-SOLVE config-diff pre-flight: rebuild a run's config and diff it.

The standard M0 gate before spending compute on a campaign point: rebuild the
config the REAL driver would hand to ``LAPDSim1D``, diff it against a
reference run's recorded config, and refuse to proceed unless the difference
is exactly the intended one.

Mechanism (from the mn_p1/p2/p3 per-arm harnesses this replaces): the driver
is not re-implemented. ``compare_sim1d_es1.LAPDSim1D`` -- the single
construction site both supported entry points reach -- is stubbed with an
object that captures the fully-resolved ``params``/``flags`` and raises, so
the driver runs its own argument parsing, its own override precedence, and
its own defaults, and stops at the constructor. Nothing is solved and nothing
is written.

THE STUB'S BLIND SPOT, and the CONSTRUCTOR PROBE that closes it: stubbing the
constructor means the CONSTRUCTION GUARDS never run, so a config the real
solver refuses outright -- an unknown key, a member key a model family cannot
carry, a closure combination that raises -- sails through the diff and dies
at run time instead. That has happened twice (ab_ewp at 28f61fe; a beam-tail
closure arm re-briefed 2026-08-23). After the diff verdict this tool
therefore constructs the REAL
``LAPDSim1D`` from the captured config, replaying the driver's own call
argument for argument, and reports ``CONSTRUCTOR: OK`` or the constructor's
full refusal under a ``CONSTRUCTOR: RAISED`` banner. It is still NO-SOLVE:
construction validates and arms subsystems, and neutral equilibration and the
seed cache belong to ``start_simulation()``, not ``__init__``. Because the
probe replays the captured call rather than rebuilding a config of its own,
its input IS the real run's constructor input -- it cannot refuse something
the run would have accepted, or accept something the run would refuse. On
``--no-constructor-probe`` the stage is skipped and says so.

What this fixes: those per-arm copies hardcoded required delta COUNTS in
their verdict (control ``(0, 0)``, arm ``(0, 1)``), which is wrong the moment
an arm carries more than one delta, and which never checked that the deltas
were the INTENDED keys at the INTENDED values -- a count of one is satisfied
by the wrong key just as well as the right one. Here the verdict is an
EXPECTED-DELTA SET given on the command line: PASS iff every expected delta is
present with exactly the expected value AND there are no unexpected deltas.

An expectation's VALUE is read by the same layer that reads a driver's
``--extra`` token, so it is typed from the key's ``default_config()`` template
and reported in the one spelling the resolved config uses. A value that cannot
be read as its key's type, and a key owned by neither configuration template,
are refused before anything is rebuilt.

Entry points:

``m6``
    ``run_m6_point.main(argv)`` with the argv passed straight through, so the
    pre-flight and the real run are driven by the same command line.
``run-model``
    ``compare_sim1d_es1.run_model(**kwargs)`` from a JSON object, for arms
    expressed through that function's own knobs (``drag_closure``,
    ``flags_extra``, ``extra``, ...).

Reference sides, exactly one required:

``--reference REF.h5``
    the config a saved run recorded.
``--stance NAME``
    the config a committed stance file names (``scripts/stances/NAME.toml``,
    applied to ``default_config()``), so a candidate can be diffed against the
    stance of record without a saved run to point at.

Usage:

    python scripts/gates/preflight_diffcfg.py --reference REF.h5 \\
        --expect 'flags:neutral_momentum=true' \\
        m6 -- --es 2 --sgp 9010 --nx 240 --save-h5 /dev/null

    python scripts/gates/preflight_diffcfg.py --stance g1atrim \\
        m6 -- --es 1 --stance g1atrim --sgp 9010 --save-h5 /dev/null

    python scripts/gates/preflight_diffcfg.py --reference REF.h5 \\
        --expect 'params:b_ion_neutral_drag=1.0' \\
        --expect 'flags:neutral_momentum=true' \\
        run-model --kwargs '{"nx": 240, "drag_closure": "neutral_momentum"}'

Exit status: 0 on PASS, 2 on FAIL -- from EITHER stage: a constructor
rejection fails the pre-flight even when every delta was the intended one.
"""

import argparse
import json
import sys
from pathlib import Path

import h5py

_SCRIPTS = Path(__file__).resolve().parents[1]
# Resolve the package and the driver scripts from THIS file, never from a
# hardcoded checkout: a worktree that reached into the main checkout would
# pre-flight a different tree than the one it is about to run.
if str(_SCRIPTS.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS.parent))
# scripts/ sibling imports: the seven purpose subdirectories on sys.path.
import sys as _sys
from pathlib import Path as _Path
for _sub in ("atomic", "gates", "kinetic", "run", "score", "stance",
             "verify"):
    _dir = str(_Path(__file__).resolve().parents[1] / _sub)
    if _dir not in _sys.path:
        _sys.path.insert(0, _dir)

import compare_sim1d_es1 as cmp_es1  # noqa: E402
import run_m6_point  # noqa: E402
from extra_overrides import coerce_override  # noqa: E402
from stance_config import available_stances, stance_config  # noqa: E402

NAMESPACES = ("params", "flags")


class _Captured(Exception):
    """Carries the driver's whole constructor call out of the stub.

    The trailing ``args``/``kwargs`` are captured alongside the two config
    dicts so the constructor probe can REPLAY the call the driver actually
    made rather than inventing one.

    ``LAPDSim1D`` takes THREE config positionals since the declaration-block
    migration -- ``(input_dict, input_flags, input_models)`` -- and the splat
    is what keeps this stub faithful to that: a third positional lands in
    ``args`` and ``constructor_probe`` forwards it unchanged, so a driver
    that grew a block would be replayed with its block. Neither supported
    entry point passes one today (measured 2026-08-30 by an AST sweep of all
    487 ``LAPDSim1D`` call sites in the checkout: the only third-positional
    callers are this module's own replay and the declaration-block gate, and
    no driver is among them), but the probe does not depend on that staying
    true. The same splat covers a progress callback or any later keyword.
    """

    def __init__(self, params, flags, args, kwargs):
        super().__init__("captured resolved config")
        self.params = dict(params)
        self.flags = dict(flags)
        self.args = tuple(args)
        self.kwargs = dict(kwargs)


class _StubSim:
    """Stands in for LAPDSim1D: captures the resolved config, never solves."""

    def __init__(self, params, flags, *args, **kwargs):
        raise _Captured(params, flags, args, kwargs)


def capture(build):
    """Run ``build`` with LAPDSim1D stubbed; return the captured call."""
    real = cmp_es1.LAPDSim1D
    cmp_es1.LAPDSim1D = _StubSim
    try:
        build()
    except _Captured as captured:
        return captured
    finally:
        cmp_es1.LAPDSim1D = real
    raise SystemExit(
        "pre-flight ERROR: the driver returned without constructing "
        "LAPDSim1D, so no config was captured"
    )


def constructor_probe(captured):
    """Construct the REAL LAPDSim1D from ``captured``; report, never solve.

    Returns True when construction succeeded. The class is read off
    ``cmp_es1`` AFTER ``capture`` has restored it, so this is the same object
    the driver would have constructed and no second import path is involved.

    ``ValueError`` is the construction guards' declared refusal channel (the
    campaign rule is a loud ValueError at construction time on any
    misconfiguration), so it is reported as a verdict rather than a crash.
    Anything else is a genuine fault in the probe or the config machinery and
    is left to propagate with its traceback intact.
    """
    print("\n=== CONSTRUCTOR PROBE (real LAPDSim1D, no solve) ===")
    try:
        cmp_es1.LAPDSim1D(
            captured.params, captured.flags, *captured.args, **captured.kwargs
        )
    except ValueError as error:
        print(f"  CONSTRUCTOR: RAISED ({type(error).__name__})")
        for line in str(error).splitlines() or [""]:
            print(f"    {line}")
        return False
    print("  CONSTRUCTOR: OK")
    return True


def read_reference(path):
    """Return the (params, flags) recorded on a saved result file."""
    with h5py.File(path, "r") as h5:
        return (
            json.loads(_decode(h5.attrs["params_json"])),
            json.loads(_decode(h5.attrs["flags_json"])),
        )


def _decode(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _as_recorded(config):
    """Normalize a live config the way saving it to HDF5 would.

    The reference side has been through ``json.dumps``/``loads``; putting the
    rebuilt side through the same round trip keeps the comparison honest
    (a tuple that would be recorded as a list is not a difference).
    """
    return json.loads(json.dumps(config, sort_keys=True, default=_json_default))


def _json_default(value):
    item = getattr(value, "item", None)
    if item is not None:
        return item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def diff_config(reference, rebuilt):
    """Return {key: (kind, old, new)} for one namespace."""
    deltas = {}
    for key in sorted(set(reference) | set(rebuilt)):
        if key not in reference:
            deltas[key] = ("ADDED", None, rebuilt[key])
        elif key not in rebuilt:
            deltas[key] = ("REMOVED", reference[key], None)
        elif reference[key] != rebuilt[key]:
            deltas[key] = ("CHANGED", reference[key], rebuilt[key])
    return deltas


def parse_expectation(text):
    """Parse one ``namespace:key=value`` expectation, typed by the template.

    The value is read by the same layer that reads a driver's ``--extra``
    token (:func:`extra_overrides.coerce_override`), so an expectation carries
    the type its key's ``default_config()`` template value carries and is
    compared against a resolved config of that same type -- ``1910`` and
    ``1910.0`` name one float on a float key, and both are reported by the one
    spelling the resolved config uses.

    A value that cannot be read as its key's type, and a key owned by neither
    configuration template, are refused here -- before any config is rebuilt --
    rather than reaching the diff as a delta that can never match.
    """
    namespace, _, rest = text.partition(":")
    if namespace not in NAMESPACES or not rest:
        raise argparse.ArgumentTypeError(
            f"--expect must read '<{'|'.join(NAMESPACES)}>:key=value' "
            f"(got {text!r})"
        )
    key, sep, raw = rest.partition("=")
    if not sep or not key:
        raise argparse.ArgumentTypeError(
            f"--expect needs 'key=value' after the namespace (got {text!r})"
        )
    try:
        value = coerce_override(key, raw, "--expect")
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return namespace, key, value


def report(namespace, deltas, expected):
    """Print one namespace's deltas; return (missing, mismatched, unexpected)."""
    print(f"  {namespace}: {len(deltas)} delta(s), {len(expected)} expected")
    for key, (kind, old, new) in deltas.items():
        mark = "  " if key in expected else "!!"
        if kind == "ADDED":
            print(f"    {mark} ADDED    {key}: <absent> -> {new!r}")
        elif kind == "REMOVED":
            print(f"    {mark} REMOVED  {key}: {old!r} -> <absent>")
        else:
            print(f"    {mark} CHANGED  {key}: {old!r} -> {new!r}")
    missing, mismatched = [], []
    for key, want in expected.items():
        if key not in deltas:
            missing.append((key, want))
            continue
        kind, _, new = deltas[key]
        if kind == "REMOVED" or new != want:
            mismatched.append((key, want, None if kind == "REMOVED" else new))
    unexpected = [key for key in deltas if key not in expected]
    for key, want in missing:
        print(f"    !! MISSING  {key}: expected -> {want!r}, but it is unchanged")
    for key, want, got in mismatched:
        shown = "<removed>" if got is None else repr(got)
        print(f"    !! MISMATCH {key}: expected -> {want!r}, got -> {shown}")
    return missing, mismatched, unexpected


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "NO-SOLVE config diff of a rebuilt driver config against a "
            "reference run, with an explicit expected-delta set."
        )
    )
    reference = parser.add_mutually_exclusive_group(required=True)
    reference.add_argument(
        "--reference",
        help="reference result .h5 whose params_json/flags_json is the baseline",
    )
    reference.add_argument(
        "--stance",
        metavar="NAME",
        help=(
            "committed stance file (scripts/stances/NAME.toml) applied to "
            "default_config() as the baseline, instead of a saved run. "
            "Available: " + (", ".join(available_stances()) or "(none committed)")
        ),
    )
    parser.add_argument(
        "--expect",
        action="append",
        default=[],
        type=parse_expectation,
        metavar="NS:KEY=VALUE",
        help=(
            "an intended delta, repeatable; NS is 'params' or 'flags' and "
            "VALUE is the value the rebuilt config must carry, typed from the "
            "key's configuration template. Every delta not named here fails "
            "the pre-flight."
        ),
    )
    parser.add_argument(
        "--constructor-probe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "after the config diff, construct the REAL LAPDSim1D from the "
            "captured config (no solve) so the construction guards are "
            "checked here instead of at run time. On by default; "
            "--no-constructor-probe skips it and says so."
        ),
    )
    entry = parser.add_subparsers(dest="entry", required=True)
    m6 = entry.add_parser(
        "m6", help="drive run_m6_point.main with the argv that follows"
    )
    m6.add_argument("argv", nargs=argparse.REMAINDER)
    run_model = entry.add_parser(
        "run-model", help="drive compare_sim1d_es1.run_model from JSON kwargs"
    )
    run_model.add_argument(
        "--kwargs",
        required=True,
        help="JSON object of run_model keyword arguments",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    expected = {namespace: {} for namespace in NAMESPACES}
    for namespace, key, value in args.expect:
        expected[namespace][key] = value

    if args.entry == "m6":
        driver_argv = list(args.argv)
        if driver_argv and driver_argv[0] == "--":
            driver_argv = driver_argv[1:]
        description = "run_m6_point.main " + " ".join(driver_argv)
        build = lambda: run_m6_point.main(driver_argv)  # noqa: E731
    else:
        kwargs = json.loads(args.kwargs)
        if not isinstance(kwargs, dict):
            raise SystemExit("--kwargs must be a JSON object")
        description = f"compare_sim1d_es1.run_model(**{kwargs})"
        build = lambda: cmp_es1.run_model(**kwargs)  # noqa: E731

    if args.stance is not None:
        stance_params, stance_flags = stance_config(args.stance)
        reference_params = _as_recorded(stance_params)
        reference_flags = _as_recorded(stance_flags)
        source = f"stance {args.stance} (scripts/stances/{args.stance}.toml)"
    else:
        reference_params, reference_flags = read_reference(args.reference)
        source = args.reference
    print(f"reference : {source}")
    print(
        f"            params keys={len(reference_params)} "
        f"flags keys={len(reference_flags)}"
    )
    print(f"driver    : {description}")

    captured = capture(build)
    rebuilt = {
        "params": _as_recorded(captured.params),
        "flags": _as_recorded(captured.flags),
    }
    reference = {"params": reference_params, "flags": reference_flags}

    print("\n=== CONFIG DIFF (reference -> rebuilt) ===")
    failures = 0
    for namespace in NAMESPACES:
        deltas = diff_config(reference[namespace], rebuilt[namespace])
        missing, mismatched, unexpected = report(
            namespace, deltas, expected[namespace]
        )
        failures += len(missing) + len(mismatched) + len(unexpected)

    print("\n=== DIFF VERDICT ===")
    if failures:
        print(f"  {failures} discrepanc(ies) between the expected and actual deltas")
        print("  CONFIG DIFF: FAIL")
    else:
        print("  every expected delta present at its expected value, nothing else")
        print("  CONFIG DIFF: PASS")

    # The probe runs even when the diff already failed: the two stages catch
    # different faults, and one pre-flight that reports both beats two runs.
    if args.constructor_probe:
        constructed = constructor_probe(captured)
    else:
        print("\n=== CONSTRUCTOR PROBE (real LAPDSim1D, no solve) ===")
        print(
            "  CONSTRUCTOR: SKIPPED (--no-constructor-probe) -- the "
            "construction guards were NOT checked, so a config this "
            "pre-flight passes can still be refused at run time"
        )
        constructed = True

    print("\n=== VERDICT ===")
    if failures or not constructed:
        print("  PRE-FLIGHT: FAIL -- DO NOT RUN")
        return 2
    print("  PRE-FLIGHT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
