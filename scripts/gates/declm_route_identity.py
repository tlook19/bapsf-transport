"""Serialize the RESOLVED config surface of every representative route.

The [decl-migration] identity instrument. The declaration-block migration is a
RE-PLUMBING: it adds an explicit input form and changes nothing a solver reads.
The proof of that is this script -- it rebuilds each representative route the
way its real driver builds it, resolves the two namespaces exactly the way
``LAPDSim1D.__init__`` resolves them, and writes a canonical JSON serialization
plus a sha256 per route.

Run it at the BASE commit and again at the tip; the two files must agree route
for route. A route whose digest moves is a resolved-value movement, which the
migration's registration makes a stop-and-report.

WHAT "RESOLVED" MEANS HERE. Exactly the two statements at the head of
``_init_config_and_early_flags`` (solver.py), in that order:

    params, flags = resolve_config(supplied_params, supplied_flags)
    resolve_model_families(params, flags)

That pair is the whole surface every later construction phase and every RHS
term reads, so it is the surface identity is claimed over. Nothing is solved
and nothing is written; construction guards are NOT run (a route that resolves
identically but refuses at construction is still a change, which is what the
smoke and the k2_dvm suite are for).

THE ROUTES, and why each one is here:

``default``
    ``default_config()`` -- the package surface itself.
``golden``
    ``baseline_sim1d.build_baseline_config()`` -- the golden-at-stance config.
    Its digest moving means the golden's config moved.
``stance_g1atrim``
    ``stance_config('g1atrim')`` -- the stance of record on the campaign route,
    which differs from the golden by the mesh-sized package.
``m6_es1``
    ``run_m6_point.py --es 1 --stance g1atrim`` captured through
    ``preflight_diffcfg.capture`` -- the real campaign driver, running its own
    argument parsing and its own override precedence, stopped at the
    constructor. Not a re-implementation.
``ka1c``
    the same driver plus the THIRTEEN-MEMBER kinetic declaration on the command
    line -- ``neutral_model=kinetic_dvm`` with every member of
    ``KINETIC_DVM_INCOMPATIBLE_DEFAULTS`` named explicitly in its own
    namespace. This is the flat-namespace shape the declaration block replaces,
    so it is the one route whose identity proves the block and the flat route
    land on the same surface.
``k2_dvm``
    ``verify_sim1d_k2_dvm.arm_config()`` -- the DVM instrument fixture.
``b0c``
    ``verify_sim1d_b0c_cadence.arm_config()`` -- the cadence fixture.

Usage::

    PYTHONPATH=<checkout> python scripts/declm_route_identity.py --out FILE.json
    PYTHONPATH=<checkout> python scripts/declm_route_identity.py \
        --compare BASE.json --out TIP.json
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
# scripts/ sibling imports: the seven purpose subdirectories on sys.path.
import sys as _sys
from pathlib import Path as _Path
for _sub in ("atomic", "gates", "kinetic", "run", "score", "stance",
             "verify"):
    _dir = str(_Path(__file__).resolve().parents[1] / _sub)
    if _dir not in _sys.path:
        _sys.path.insert(0, _dir)
if str(_SCRIPTS.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS.parent))

import cablp  # noqa: E402
from cablp.solvers._sim1d.core.config import resolve_config  # noqa: E402
from cablp.solvers._sim1d.core.model_families import (  # noqa: E402
    KINETIC_DVM_INCOMPATIBLE_DEFAULTS,
    resolve_model_families,
)


def _canonical(value):
    """Return ``value`` in a form ``json.dumps`` serializes deterministically."""
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, float):
        # repr round-trips a float exactly; json's own float encoder does too,
        # but going through a string keeps -0.0 and the integral floats
        # distinguishable from ints in the serialization.
        return {"__float__": repr(value)}
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    return {"__repr__": repr(value)}


def resolved_surface(params, flags):
    """Resolve one route the way ``LAPDSim1D.__init__`` resolves it."""
    resolved_params, resolved_flags = resolve_config(params, flags)
    resolve_model_families(resolved_params, resolved_flags)
    return resolved_params, resolved_flags


def _serialize(params, flags):
    payload = {"params": _canonical(params), "flags": _canonical(flags)}
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return payload, hashlib.sha256(text.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ routes


def route_default():
    from cablp.solvers._sim1d import default_config

    return default_config()


def route_golden():
    import baseline_sim1d

    return baseline_sim1d.build_baseline_config()


def route_stance_g1atrim():
    from stance_config import stance_config

    return stance_config("g1atrim")


def _capture_m6(argv):
    """Run run_m6_point's own argument parsing and stop at the constructor."""
    import preflight_diffcfg
    import run_m6_point

    captured = preflight_diffcfg.capture(lambda: run_m6_point.main(argv))
    return captured.params, captured.flags


#: The command line every ``m6``-route case shares. ``--save-h5`` is required
#: by the driver and is never written: the constructor stub raises first.
_M6_BASE = [
    "--es", "1",
    "--stance", "g1atrim",
    "--sgp", "9010",
    "--save-h5", "/dev/null",
]


def route_m6_es1():
    return _capture_m6(list(_M6_BASE))


def route_ka1c():
    """The 13-member kinetic declaration as a command line.

    Every member of ``KINETIC_DVM_INCOMPATIBLE_DEFAULTS`` is named in ITS OWN
    namespace -- ``--extra`` for params, ``--extra-flag`` for flags -- which is
    the save-gate-probe discipline (adopted 2026-08-28 (Tom)) written out.
    """
    argv = list(_M6_BASE) + ["--extra", "neutral_model=kinetic_dvm"]
    params = [
        (key, value)
        for space, key, value, _why in KINETIC_DVM_INCOMPATIBLE_DEFAULTS
        if space == "params"
    ]
    flags = [
        (key, value)
        for space, key, value, _why in KINETIC_DVM_INCOMPATIBLE_DEFAULTS
        if space == "flags"
    ]
    argv += ["--extra"] + [f"{k}={json.dumps(v)}" for k, v in params]
    argv += ["--extra-flag"] + [f"{k}={json.dumps(v)}" for k, v in flags]
    return _capture_m6(argv)


def route_k2_dvm():
    import verify_sim1d_k2_dvm

    return verify_sim1d_k2_dvm.arm_config()


def route_b0c():
    import verify_sim1d_b0c_cadence

    return verify_sim1d_b0c_cadence.arm_config()


ROUTES = (
    ("default", route_default),
    ("golden", route_golden),
    ("stance_g1atrim", route_stance_g1atrim),
    ("m6_es1", route_m6_es1),
    ("ka1c", route_ka1c),
    ("k2_dvm", route_k2_dvm),
    ("b0c", route_b0c),
)


def collect(only=None):
    """Build every route and return ``{name: {digest, params, flags}}``."""
    out = {}
    for name, build in ROUTES:
        if only and name not in only:
            continue
        params, flags = build()
        resolved_params, resolved_flags = resolved_surface(params, flags)
        payload, digest = _serialize(resolved_params, resolved_flags)
        out[name] = {"digest": digest, **payload}
        print(f"  {name:16s} {digest}")
    return out


def compare(base, tip):
    """Print a per-route, per-key comparison. Return True when all agree."""
    ok = True
    names = sorted(set(base) | set(tip))
    print("\n=== ROUTE IDENTITY ===")
    for name in names:
        if name not in base or name not in tip:
            side = "tip" if name in tip else "base"
            print(f"  {name:16s} PRESENT ONLY AT {side.upper()}")
            ok = False
            continue
        if base[name]["digest"] == tip[name]["digest"]:
            print(f"  {name:16s} IDENTICAL  {base[name]['digest']}")
            continue
        ok = False
        print(f"  {name:16s} MOVED")
        print(f"      base {base[name]['digest']}")
        print(f"      tip  {tip[name]['digest']}")
        for space in ("params", "flags"):
            b, t = base[name][space], tip[name][space]
            for key in sorted(set(b) | set(t)):
                if key not in b:
                    print(f"      + {space}:{key} = {t[key]!r}  (ADDED)")
                elif key not in t:
                    print(f"      - {space}:{key} = {b[key]!r}  (REMOVED)")
                elif b[key] != t[key]:
                    print(f"      ~ {space}:{key}: {b[key]!r} -> {t[key]!r}")
    print("\nVERDICT:", "ALL ROUTES IDENTICAL" if ok else "ROUTE IDENTITY MOVED")
    return ok


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default=None, help="write the collected routes here")
    p.add_argument("--compare", default=None, help="a previously written --out")
    p.add_argument("--only", nargs="*", default=(), help="restrict to routes")
    args = p.parse_args(argv)

    print(f"import provenance: {cablp.__file__}")
    print("=== RESOLVED-SURFACE DIGESTS ===")
    collected = collect(only=set(args.only) or None)

    if args.out:
        Path(args.out).write_text(json.dumps(collected, indent=1, sort_keys=True))
        print(f"\nwrote {args.out}")

    if args.compare:
        base = json.loads(Path(args.compare).read_text())
        return 0 if compare(base, collected) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
