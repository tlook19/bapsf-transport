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

What this fixes: those per-arm copies hardcoded required delta COUNTS in
their verdict (control ``(0, 0)``, arm ``(0, 1)``), which is wrong the moment
an arm carries more than one delta, and which never checked that the deltas
were the INTENDED keys at the INTENDED values -- a count of one is satisfied
by the wrong key just as well as the right one. Here the verdict is an
EXPECTED-DELTA SET given on the command line: PASS iff every expected delta is
present with exactly the expected value AND there are no unexpected deltas.

Entry points:

``m6``
    ``run_m6_point.main(argv)`` with the argv passed straight through, so the
    pre-flight and the real run are driven by the same command line.
``run-model``
    ``compare_sim1d_es1.run_model(**kwargs)`` from a JSON object, for arms
    expressed through that function's own knobs (``drag_closure``,
    ``flags_extra``, ``extra``, ...).

Usage:

    python scripts/preflight_diffcfg.py --reference REF.h5 \\
        --expect 'flags:neutral_momentum=true' \\
        m6 -- --es 2 --sgp 3000 --nx 240 --save-h5 /dev/null

    python scripts/preflight_diffcfg.py --reference REF.h5 \\
        --expect 'params:b_ion_neutral_drag=1.0' \\
        --expect 'flags:neutral_momentum=true' \\
        run-model --kwargs '{"nx": 240, "drag_closure": "neutral_momentum"}'

Exit status: 0 on PASS, 2 on FAIL.
"""

import argparse
import json
import sys
from pathlib import Path

import h5py

_SCRIPTS = Path(__file__).resolve().parent
# Resolve the package and the driver scripts from THIS file, never from a
# hardcoded checkout: a worktree that reached into the main checkout would
# pre-flight a different tree than the one it is about to run.
for _entry in (str(_SCRIPTS.parent), str(_SCRIPTS)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

import compare_sim1d_es1 as cmp_es1  # noqa: E402
import run_m6_point  # noqa: E402

NAMESPACES = ("params", "flags")


class _Captured(Exception):
    """Carries the resolved config out of the stubbed constructor."""

    def __init__(self, params, flags):
        super().__init__("captured resolved config")
        self.params = dict(params)
        self.flags = dict(flags)


class _StubSim:
    """Stands in for LAPDSim1D: captures the resolved config, never solves."""

    def __init__(self, params, flags, *args, **kwargs):
        raise _Captured(params, flags)


def capture(build):
    """Run ``build`` with LAPDSim1D stubbed; return its resolved config."""
    real = cmp_es1.LAPDSim1D
    cmp_es1.LAPDSim1D = _StubSim
    try:
        build()
    except _Captured as captured:
        return captured.params, captured.flags
    finally:
        cmp_es1.LAPDSim1D = real
    raise SystemExit(
        "pre-flight ERROR: the driver returned without constructing "
        "LAPDSim1D, so no config was captured"
    )


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
    """Parse one ``namespace:key=<json value>`` expectation."""
    namespace, _, rest = text.partition(":")
    if namespace not in NAMESPACES or not rest:
        raise argparse.ArgumentTypeError(
            f"--expect must read '<{'|'.join(NAMESPACES)}>:key=<json value>' "
            f"(got {text!r})"
        )
    key, sep, raw = rest.partition("=")
    if not sep or not key:
        raise argparse.ArgumentTypeError(
            f"--expect needs 'key=<json value>' after the namespace (got {text!r})"
        )
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        # A bare word is the common case for selector strings; accept it
        # rather than making every caller quote JSON inside shell quotes.
        value = raw
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
    parser.add_argument(
        "--reference",
        required=True,
        help="reference result .h5 whose params_json/flags_json is the baseline",
    )
    parser.add_argument(
        "--expect",
        action="append",
        default=[],
        type=parse_expectation,
        metavar="NS:KEY=JSON",
        help=(
            "an intended delta, repeatable; NS is 'params' or 'flags' and "
            "JSON is the value the rebuilt config must carry. Every delta not "
            "named here fails the pre-flight."
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

    reference_params, reference_flags = read_reference(args.reference)
    print(f"reference : {args.reference}")
    print(
        f"            params keys={len(reference_params)} "
        f"flags keys={len(reference_flags)}"
    )
    print(f"driver    : {description}")

    params, flags = capture(build)
    rebuilt = {"params": _as_recorded(params), "flags": _as_recorded(flags)}
    reference = {"params": reference_params, "flags": reference_flags}

    print("\n=== CONFIG DIFF (reference -> rebuilt) ===")
    failures = 0
    for namespace in NAMESPACES:
        deltas = diff_config(reference[namespace], rebuilt[namespace])
        missing, mismatched, unexpected = report(
            namespace, deltas, expected[namespace]
        )
        failures += len(missing) + len(mismatched) + len(unexpected)

    print("\n=== VERDICT ===")
    if failures:
        print(f"  {failures} discrepanc(ies) between the expected and actual deltas")
        print("  PRE-FLIGHT: FAIL -- DO NOT RUN")
        return 2
    print("  every expected delta present at its expected value, nothing else")
    print("  PRE-FLIGHT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
