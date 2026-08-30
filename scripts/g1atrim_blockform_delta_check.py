"""G0: the block-form migration of g1atrim moved no VALUE.

The [g1atrim-block-form] delta-level gate. The stance of record was rewritten
from a flat delta into four declaration blocks plus a smaller flat delta. That
is a FORM migration, and this script is the measurement that says so at the
delta level -- one level below ``declm_route_identity.py``, which measures the
same claim at the resolved surface. Both are needed: a resolved surface can be
identical while the delta that produced it has quietly changed shape, and a
delta can look right while a consumer that reads only ``Stance.params`` sees
something else.

THREE ASSERTIONS, against the BASE file passed with ``--base``:

1. **Nothing dropped or moved.** Every ``(namespace, key, value)`` the BASE
   delta carries is present in the TIP delta at an EQUAL value
   (``model_families.values_equal``, which is safe for the per-cell lists).
2. **Nothing smuggled.** Every key the TIP delta carries that the BASE delta
   did not is a family member the block form now states explicitly, and its
   value EQUALS its ``default_config()`` value -- so projecting it onto the
   template is a no-op and no consumer can see it.
3. **The declared form is the intended one.** ``Stance.models`` carries exactly
   the four families this migration declares.

Assertion 2 is the one that carries the risk. A block is an INVENTORY, so it
necessarily states members the flat stance never named; each such member reaches
every consumer as a new delta entry. It is inert if and only if it equals the
config default, and that is what is measured here rather than assumed.

Usage::

    PYTHONPATH=<checkout> python scripts/g1atrim_blockform_delta_check.py \
        --base <path to the base g1atrim.toml>
"""

import argparse
import sys
import tomllib
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
if str(_SCRIPTS.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS.parent))

import cablp  # noqa: E402
from cablp.solvers._sim1d import default_config  # noqa: E402
from cablp.solvers._sim1d.core.model_families import values_equal  # noqa: E402
from stance_config import load_stance  # noqa: E402

STANCE = "g1atrim"

#: The families the migration declares in this stance file.
EXPECTED_BLOCKS = {
    "beam_tail_closure",
    "cathode_surface_recycle",
    "anode_surface_recycle",
    "initial_neutral_state",
}

NAMESPACES = ("input_dict", "input_flags")


def base_delta(path):
    """The BASE file's delta, as ``{namespace: {key: value}}``.

    The base form is flat: two tables plus a file-scope ``[none_valued]``. This
    reads it directly rather than through the loader, so the comparison does not
    depend on loader behaviour that the migration itself touches.
    """
    document = tomllib.loads(Path(path).read_text())
    unknown = sorted(set(document) - set(NAMESPACES) - {"none_valued", "models"})
    if unknown:
        raise SystemExit(f"base file has unexpected table(s): {unknown}")
    if "models" in document:
        raise SystemExit(
            "the --base file already carries [models.*] blocks; it is not the "
            "flat base this gate compares against"
        )
    out = {name: dict(document.get(name, {})) for name in NAMESPACES}
    none_valued = document.get("none_valued", {})
    for name in NAMESPACES:
        for key in none_valued.get(name, ()):
            out[name][key] = None
    return out


def brief(value):
    text = repr(value)
    return text if len(text) <= 58 else text[:55] + "..."


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--base",
        required=True,
        help="the base (flat) g1atrim.toml, e.g. from "
        "`git show ca444dd:scripts/stances/g1atrim.toml`",
    )
    args = parser.parse_args(argv)

    print(f"import provenance: {cablp.__file__}")

    base = base_delta(args.base)
    stance = load_stance(STANCE)
    tip = {"input_dict": dict(stance.params), "input_flags": dict(stance.flags)}
    defaults = dict(zip(NAMESPACES, default_config()))

    violations = []

    # (1) every BASE delta entry survives, at an equal value.
    print("\n=== (1) BASE delta entries preserved in the TIP delta ===")
    preserved = 0
    for name in NAMESPACES:
        for key in sorted(base[name]):
            want = base[name][key]
            if key not in tip[name]:
                violations.append(f"{name}:{key} DROPPED (base {brief(want)})")
                print(f"  FAIL  {name}:{key} DROPPED")
                continue
            got = tip[name][key]
            if not values_equal(got, want):
                violations.append(
                    f"{name}:{key} MOVED: base {brief(want)} -> tip {brief(got)}"
                )
                print(f"  FAIL  {name}:{key} base {brief(want)} -> tip {brief(got)}")
                continue
            preserved += 1
    print(f"  {preserved} base delta entries preserved, "
          f"{len(violations)} violation(s)")

    # (2) every key the TIP delta adds equals its config default.
    print("\n=== (2) TIP-only delta keys equal their config default ===")
    added = 0
    added_violations = 0
    for name in NAMESPACES:
        for key in sorted(tip[name]):
            if key in base[name]:
                continue
            added += 1
            got = tip[name][key]
            want = defaults[name].get(key)
            if key not in defaults[name]:
                added_violations += 1
                violations.append(f"{name}:{key} ADDED but no template owns it")
                print(f"  FAIL  {name}:{key} ADDED, no template owns it")
                continue
            if not values_equal(got, want):
                added_violations += 1
                violations.append(
                    f"{name}:{key} ADDED at {brief(got)}, but the config "
                    f"default is {brief(want)}"
                )
                print(f"  FAIL  {name}:{key} ADDED at {brief(got)} != "
                      f"default {brief(want)}")
                continue
            print(f"  PASS  {name}:{key} = {brief(got)} (== config default)")
    print(f"  {added} key(s) added by the block form, "
          f"{added_violations} not at their default")

    # (3) the declared families are the intended four.
    print("\n=== (3) Stance.models carries exactly the declared families ===")
    declared = set(stance.models)
    if declared == EXPECTED_BLOCKS:
        print(f"  PASS  {len(declared)} blocks: {', '.join(sorted(declared))}")
    else:
        violations.append(
            f"Stance.models is {sorted(declared)}, expected "
            f"{sorted(EXPECTED_BLOCKS)}"
        )
        print(f"  FAIL  {sorted(declared)} != {sorted(EXPECTED_BLOCKS)}")

    print(
        f"\nBASE delta: {sum(len(base[n]) for n in NAMESPACES)} entries; "
        f"TIP delta: {sum(len(tip[n]) for n in NAMESPACES)} entries "
        f"({added} added, all at config defaults when this gate passes)"
    )
    if violations:
        print(f"\nG0 FAIL: {len(violations)} violation(s)")
        for line in violations:
            print(f"  {line}")
        return 1
    print("\nG0 PASS: 0 violations -- the block form moved no value")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
