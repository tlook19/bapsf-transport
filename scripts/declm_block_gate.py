"""The declaration-block acceptance gate: equivalence, then every refusal.

Two questions, and the whole migration rests on them.

**EQUIVALENCE.** A configuration written as a declaration block and the same
configuration written flat must resolve to the identical surface -- byte for
byte, both namespaces. That is what makes the block a re-plumbing rather than a
change: if the two forms ever disagreed, adopting a block would silently move a
run. Every declared family is exercised, each one built twice from ONE source of
values (so the two arms cannot drift apart in the fixture itself) and compared
key by key.

**REFUSAL.** A block that is wrong must fail loudly at construction, naming the
offender and carrying the remedy -- never silently, never at run time. The cases
below are the ones the migration registered, plus the ones the form itself
implies: unknown family, unknown member, INCOMPLETE membership (the 24d option
(b) rule -- a block is an inventory, not a delta), a block for a family this
config does not select, two blocks claiming one key, a member also supplied
flat, and a family whose mutually exclusive routes are multiply armed.

Both halves run at the RESOLVER, and the equivalence half additionally runs a
real ``LAPDSim1D`` construction on one family so the claim covers the
constructor and not just the config boundary.

Usage::

    PYTHONPATH=<checkout> python scripts/declm_block_gate.py
"""

import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
if str(_SCRIPTS.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS.parent))

import cablp  # noqa: E402
from cablp.solvers._sim1d import default_config, resolve_config  # noqa: E402
from cablp.solvers._sim1d.core.model_families import (  # noqa: E402
    DECLARED_FAMILIES,
    FLAGS,
)

FAILURES = []
CHECKS = [0]


def check(label, condition, detail=""):
    CHECKS[0] += 1
    if condition:
        print(f"  PASS  {label}")
        return True
    print(f"  FAIL  {label}")
    if detail:
        for line in str(detail).splitlines():
            print(f"        {line}")
    FAILURES.append(label)
    return False


def refuses(label, build, *, must_name):
    """Assert ``build`` raises a ValueError naming every string in ``must_name``."""
    CHECKS[0] += 1
    try:
        build()
    except ValueError as error:
        text = str(error)
        missing = [needle for needle in must_name if needle not in text]
        if missing:
            print(f"  FAIL  {label}: message omits {missing}")
            print(f"        {text.splitlines()[0]}")
            FAILURES.append(label)
            return
        print(f"  PASS  {label}")
        print(f"        -> {text.splitlines()[0]}")
        return
    except Exception as error:  # pragma: no cover - a wrong exception type
        print(f"  FAIL  {label}: raised {type(error).__name__}, not ValueError")
        FAILURES.append(label)
        return
    print(f"  FAIL  {label}: did not raise")
    FAILURES.append(label)


# ------------------------------------------------------------- fixtures


#: A declared value per member, per family: the SOURCE both arms of the
#: equivalence test read. Values are chosen to be non-default where a default
#: would make the test vacuous, and to satisfy the family's own guards where
#: the constructor leg exercises them.
def family_values(family):
    """Return ``{member: value}`` for ``family``, from the config defaults.

    The config default IS a legal declared value for every member -- that is
    what "explicit regardless of value" means -- and starting from it keeps the
    fixture honest: the block arm and the flat arm are then testing the
    PLUMBING, not a hand-tuned config. The selector is forced to the engaged
    value where the family has one, because a block must select its family.
    """
    params, flags = default_config()
    values = {}
    for space, key in family.members:
        values[key] = (flags if space == FLAGS else params)[key]
    if family.selector is not None:
        values[family.selector] = family.engaged_value
    return values


def split_by_namespace(family, values):
    """Return ``(params, flags)`` -- the same values, filed by hand."""
    params, flags = {}, {}
    for space, key in family.members:
        (flags if space == FLAGS else params)[key] = values[key]
    return params, flags


def canonical(params, flags):
    return json.dumps(
        {
            "params": {k: repr(v) for k, v in sorted(params.items())},
            "flags": {k: repr(v) for k, v in sorted(flags.items())},
        },
        sort_keys=True,
    )


# ------------------------------------------------------- equivalence half


def gate_equivalence():
    print("\n=== BLOCK == FLAT (resolved surface, per family) ===")
    for family in DECLARED_FAMILIES:
        values = family_values(family)
        flat_params, flat_flags = split_by_namespace(family, values)

        block_side = resolve_config(models={family.name: dict(values)})
        flat_side = resolve_config(flat_params, flat_flags)

        same = canonical(*block_side) == canonical(*flat_side)
        detail = ""
        if not same:
            diffs = []
            for space, b, f in (
                ("params", block_side[0], flat_side[0]),
                ("flags", block_side[1], flat_side[1]),
            ):
                for key in sorted(set(b) | set(f)):
                    if b.get(key) != f.get(key):
                        diffs.append(
                            f"{space}:{key}: block={b.get(key)!r} "
                            f"flat={f.get(key)!r}"
                        )
            detail = "\n".join(diffs)
        check(
            f"{family.name}: block and flat resolve identically "
            f"({len(family.members)} members)",
            same,
            detail,
        )


def gate_equivalence_constructed():
    """The same claim one level up: a real solver built each way.

    The resolver test above compares the config boundary. This one compares
    what the CONSTRUCTOR made of it, so the claim covers the guards, the
    family resolver and every ``_init_*`` phase rather than stopping at
    ``resolve_config``.
    """
    print("\n=== BLOCK == FLAT (constructed solver) ===")
    from cablp.solvers._sim1d import LAPDSim1D

    family = next(
        f for f in DECLARED_FAMILIES if f.name == "cathode_surface_recycle"
    )
    values = family_values(family)
    flat_params, flat_flags = split_by_namespace(family, values)

    base_params, base_flags = default_config()
    base_params["nx"] = 8

    try:
        block_sim = LAPDSim1D(
            dict(base_params, **{}),
            dict(base_flags),
            {family.name: dict(values)},
        )
        flat_sim = LAPDSim1D(
            dict(base_params, **flat_params), dict(base_flags, **flat_flags)
        )
    except Exception as error:  # pragma: no cover - reported, not swallowed
        check(f"{family.name}: both forms construct", False, error)
        return

    check(
        f"{family.name}: constructed configs identical",
        canonical(block_sim._input_dict, block_sim._flags)
        == canonical(flat_sim._input_dict, flat_sim._flags),
    )


# ---------------------------------------------------------- refusal half


def gate_refusals():
    print("\n=== REFUSALS (loud ValueError, naming the offender) ===")
    dvm = next(f for f in DECLARED_FAMILIES if f.name == "neutral_closure")
    cathode = next(
        f for f in DECLARED_FAMILIES if f.name == "cathode_surface_recycle"
    )
    fill = next(
        f for f in DECLARED_FAMILIES if f.name == "initial_neutral_state"
    )

    refuses(
        "unknown declaration-block family",
        lambda: resolve_config(models={"neutral_clsoure": {}}),
        must_name=["neutral_clsoure", "Declarable families"],
    )

    bad_member = dict(family_values(cathode))
    bad_member["cathode_jet_R_M"] = 0.5
    refuses(
        "unknown key inside a block",
        lambda: resolve_config(models={cathode.name: bad_member}),
        must_name=["cathode_jet_R_M", "does not own"],
    )

    misfiled = dict(family_values(cathode))
    misfiled["neutral_equilibration"] = False
    refuses(
        "a block naming another family's member",
        lambda: resolve_config(models={cathode.name: misfiled}),
        must_name=["neutral_equilibration", "models.initial_neutral_state"],
    )

    incomplete = dict(family_values(cathode))
    del incomplete["cathode_jet_hot_carrier"]
    refuses(
        "INCOMPLETE membership (24d option (b): explicit regardless of value)",
        lambda: resolve_config(models={cathode.name: incomplete}),
        must_name=["INCOMPLETE", "cathode_jet_hot_carrier", "not declared"],
    )

    unselected = dict(family_values(dvm))
    unselected["neutral_model"] = "moment"
    refuses(
        "a block for a family this config does not select",
        lambda: resolve_config(models={dvm.name: unselected}),
        must_name=["does not select", "neutral_model", "kinetic_dvm"],
    )

    refuses(
        "two blocks claiming one key",
        lambda: resolve_config(
            models={
                dvm.name: dict(family_values(dvm)),
                cathode.name: dict(family_values(cathode)),
            }
        ),
        must_name=["declared by TWO model blocks", "models.neutral_closure"],
    )

    refuses(
        "a member also supplied FLAT",
        lambda: resolve_config(
            {"cathode_neutral_jet": False},
            None,
            {cathode.name: dict(family_values(cathode))},
        ),
        must_name=["cathode_neutral_jet", "one home", "flat"],
    )

    two_routes = dict(family_values(fill))
    two_routes["neutral_equilibration"] = True
    two_routes["neutral_initial_profile"] = True
    refuses(
        "two mutually exclusive fill routes armed at once",
        lambda: resolve_config(models={fill.name: two_routes}),
        must_name=["mutually exclusive", "equilibrate", "profile"],
    )

    both_valued = dict(family_values(cathode))
    both_valued["none_valued"] = ["cathode_jet_R_N"]
    refuses(
        "a member declared with a value AND in none_valued",
        lambda: resolve_config(models={cathode.name: both_valued}),
        must_name=["cathode_jet_R_N", "one declared value"],
    )


def gate_none_valued():
    """``none_valued`` declares a member as ``None``; TOML has no null."""
    print("\n=== none_valued (the TOML null affordance) ===")
    anode = next(
        f for f in DECLARED_FAMILIES if f.name == "anode_surface_recycle"
    )
    values = family_values(anode)
    block = {k: v for k, v in values.items() if k != "anode_jet_energy_convention"}
    block["none_valued"] = ["anode_jet_energy_convention"]
    params, _flags = resolve_config(models={anode.name: block})
    check(
        "none_valued sets the member to None",
        params["anode_jet_energy_convention"] is None,
        f"got {params['anode_jet_energy_convention']!r}",
    )


def gate_flat_route_untouched():
    """The flat route still works, and still refuses what it always refused."""
    print("\n=== FLAT ROUTE UNTOUCHED ===")
    base = resolve_config()
    check(
        "resolve_config() with no blocks equals default_config()",
        canonical(*base) == canonical(*default_config()),
    )
    refuses(
        "an unknown flat key is still refused",
        lambda: resolve_config({"not_a_key": 1}),
        must_name=["unknown LAPDSim1D configuration keys", "not_a_key"],
    )


def main():
    print(f"import provenance: {cablp.__file__}")
    gate_equivalence()
    gate_equivalence_constructed()
    gate_none_valued()
    gate_refusals()
    gate_flat_route_untouched()
    print(f"\n{CHECKS[0]} checks, {len(FAILURES)} failed")
    if FAILURES:
        for label in FAILURES:
            print(f"  FAILED: {label}")
        return 1
    print("declm block gate: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
