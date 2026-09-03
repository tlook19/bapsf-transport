"""The declaration-block acceptance gate: equivalence, then every refusal.

Two questions, and the whole migration rests on them.

**EQUIVALENCE.** A configuration written as a declaration block and the same
configuration written flat must resolve to the identical surface -- byte for
byte, both namespaces. That is what makes the block a re-plumbing rather than a
change: if the two forms ever disagreed, adopting a block would silently move a
run. Every declared family is exercised, each one built twice from ONE source of
values (so the two arms cannot drift apart in the fixture itself) and compared
key by key.

The equivalence fixture runs on NON-DEFAULT values (``PERTURBED``), one legal
alternative per member off the config default. On defaults the comparison is
weak in a specific way: a member the projection silently DROPPED would fall back
to the same default on both arms, the two sides would agree, and the leg would
pass while measuring nothing.

**REFUSAL.** A block that is wrong must fail loudly at construction, naming the
offender and carrying the remedy -- never silently, never at run time. ALL TEN
refusals the resolver carries are exercised here, and the count is the point:
``CONFIG_DECLARATIONS.md`` is the KB schema source for this form, so its table
and these checks are meant to be the same list. They are: a block that is not a
table; ``none_valued`` given something other than an array of names; a member
both valued and named in ``none_valued``; a key the family does not own (all
three owner branches -- another family, a config key no family owns in either
namespace, and no template at all); INCOMPLETE membership (the 24d option (b)
rule -- a block is an inventory, not a delta); a block for a family this config
does not select; a family whose mutually exclusive routes are multiply armed;
an unknown family; two blocks claiming one key; and a member also supplied
FLAT at a different value.

Both halves run at the RESOLVER, and the equivalence half additionally runs a
real ``LAPDSim1D`` construction on one family so the claim covers the
constructor and not just the config boundary. That constructor leg reads the
DEFAULT-valued fixture, because it needs a configuration that constructs. The
two TOML routes -- ``load_config`` and a committed stance file -- are exercised
against the SAME block and compared to the same flat form, so a divergence
between the file routes and the Python API cannot hide.

Last, the FIVE run-time-first guards hoisted with this migration are asserted to
refuse AT CONSTRUCTION -- the original four, ``dt_growth_factor``, the two
scheme selectors and the drag model (negative control at base commit aa65468),
and ``beam_deposition_model`` (negative control at base commit
ca444dd, where 'cdsa' constructs and silently runs beer_lambert). Both
reproduction recipes are in ``gate_hoisted_guards``.

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


def family_values(family):
    """Return ``{member: value}`` for ``family``, AT THE CONFIG DEFAULTS.

    The config default IS a legal declared value for every member -- that is
    what "explicit regardless of value" means. This is the fixture the
    CONSTRUCTOR leg and the refusal fixtures read, because both need a
    configuration that actually constructs, and the shipped defaults are the
    one value set guaranteed to.

    It is NOT what the equivalence leg reads. A block==flat comparison built
    from defaults is weak in a specific way: if the projection silently DROPPED
    a member, the flat arm would carry that member's default too, both sides
    would agree, and the test would pass while measuring nothing. See
    :func:`perturbed_family_values`.

    The selector is forced to the engaged value where the family has one,
    because a block must select its family.
    """
    params, flags = default_config()
    values = {}
    for space, key in family.members:
        values[key] = (flags if space == FLAGS else params)[key]
    if family.selector is not None:
        values[family.selector] = family.engaged_value
    return values


#: A legal NON-DEFAULT value for every member of every declared family, keyed by
#: config key. Each is a genuine alternative the solver accepts -- a bool
#: flipped, a string from the same validator's accepted set, a float moved off
#: its default, a None member given a legal value and vice versa -- read off the
#: domain each key's own guard enforces, not invented.
#:
#: This table is what makes the equivalence leg non-vacuous: with every member
#: at a value the config template does NOT hold, a member the projection failed
#: to carry shows up immediately as a difference against the flat arm, instead of
#: being masked by both sides falling back to the same default.
#:
#: A key appearing in two families takes one value here; where that key is the
#: OTHER family's selector, the selector force in
#: :func:`perturbed_family_values` overrides it, which is why
#: ``neutral_momentum_radial`` can be ``'two_zone'`` here and still be
#: ``'kinetic_two_moment'`` in its own family's block.
PERTURBED = {
    # beam_tail_closure
    "beam_deposition_model": "beer_lambert",
    "beam_coulomb_model": "legacy_tau_ei",
    "beam_anomalous_model": "none",
    "ql_relaxation_coeff": 60.0,
    "beam_product_transport": "nonlocal",
    "heating_anomalous_transport": "tail_walk",
    "heating_anomalous_disposal": "landau_branched",
    "heating_anomalous_tail_energy_eV": 150.0,
    "heating_anomalous_tail_ionization": "on",
    "heating_anomalous_tail_energy_keying": "fixed",
    "heating_anomalous_tail_phi_c_fraction": 0.25,
    "heating_anomalous_tail_cathode_boundary": "escape",
    "beam_tail_anode_reflected_particles": 0.5,
    "beam_tail_anode_reflected_energy": 0.5,
    "beam_clump_fraction": 0.5,
    "beam_clump_enhancement": 2.0,
    "beam_deposition_smoothing_cm": 25.0,
    "b_beam_excitation": 1.0,
    "beam_excitation_model": "manifold",
    "beam_excitation_energy_eV": 22.218,
    "beam_anode_interception": False,
    "beam_tail_anode_interception": True,
    # cathode_surface_recycle
    "cathode_neutral_jet": False,
    "cathode_jet_R_N": 0.5,
    "cathode_jet_R_E": 0.3,
    "cathode_jet_energy_convention": "legacy",
    "cathode_jet_surface_debit": False,
    "cathode_jet_hot_carrier": True,
    # anode_surface_recycle
    "anode_neutral_jet": True,
    "anode_jet_R_N": 0.5,
    "anode_jet_R_E": 0.3,
    "anode_jet_energy_convention": "total_reflected",
    "neutral_mesh_accommodation": True,
    # initial_neutral_state
    "neutral_equilibration": False,
    "launch_plasma_after_equilibration": False,
    "neutral_equilibration_cycles": 200,
    "neutral_equilibration_dt": 0.02,
    "equilibration_gas_puff_on_s": 0.025,
    "use_cached_neutral_seed": True,
    "neutral_seed_cache_dir": "/tmp/declm_block_gate_seed_cache",
    "neutral_initial_profile": True,
    "nn0": 4.0e13,
    "nn0_profile": [1.0, 2.0, 3.0, 4.0],
    "nn0_annulus_profile": [1.0, 2.0, 3.0, 4.0],
    # neutral_closure / neutral_radial_closure
    "neutral_model": "kinetic_dvm",
    "neutral_momentum": False,
    "neutral_energy": False,
    "neutral_hot_internal_wall": False,
    "neutral_momentum_radial": "two_zone",
    "neutral_hot_birth_drift": True,
    "neutral_knudsen_temperature": "local",
    "neutral_wall_momentum_partition": True,
    "neutral_wall_partition_sigma_hehe_cm2": 1.5e-15,
}

#: Members deliberately LEFT at their config default in the perturbed fixture,
#: with the reason. ``restart_from`` is a mutually exclusive ROUTE of
#: ``initial_neutral_state``: the fixture already arms the ``profile`` route, so
#: giving the restart payload a value would arm a second route and the block
#: would be refused before the equivalence comparison ever ran. The perturbation
#: is applied to the two route keys that CAN move together -- equilibrate is
#: disarmed and profile is armed, both away from their defaults.
PERTURB_KEEP_DEFAULT = {"restart_from"}


def perturbed_family_values(family):
    """Return ``{member: value}`` for ``family`` at legal NON-DEFAULT values.

    Every member is moved off its config default except those named in
    :data:`PERTURB_KEEP_DEFAULT`, and the selector is then forced to the engaged
    value (which is itself non-default for both selector families).

    This runs at the RESOLVER, where the per-key domain guards do not run; the
    values are nevertheless taken from the accepted sets those guards enforce,
    so the fixture reads as a configuration rather than as noise.
    """
    params, flags = default_config()
    values = {}
    for space, key in family.members:
        default = (flags if space == FLAGS else params)[key]
        if key in PERTURB_KEEP_DEFAULT:
            values[key] = default
            continue
        if key not in PERTURBED:
            raise KeyError(
                f"PERTURBED has no non-default value for {space}:{key}, a "
                f"member of {family.name!r}. Add one (a legal value off its "
                "config default) so the equivalence leg stays non-vacuous."
            )
        values[key] = PERTURBED[key]
    if family.selector is not None:
        values[family.selector] = family.engaged_value
    return values


def perturbation_report(family, values):
    """Return ``(moved, total)`` -- how many members sit off their default."""
    params, flags = default_config()
    moved = 0
    for space, key in family.members:
        default = (flags if space == FLAGS else params)[key]
        if repr(values[key]) != repr(default):
            moved += 1
    return moved, len(family.members)


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
    """Block == flat, per family, ON NON-DEFAULT VALUES.

    The values come from :func:`perturbed_family_values`, not from the config
    defaults. On defaults this comparison can pass while measuring nothing: a
    member the projection dropped would fall back to the same default on both
    arms. Every member here sits at a value the template does not hold (bar the
    one route key :data:`PERTURB_KEEP_DEFAULT` names), so a dropped member is a
    visible difference.
    """
    print("\n=== BLOCK == FLAT (resolved surface, per family, NON-DEFAULT) ===")
    for family in DECLARED_FAMILIES:
        values = perturbed_family_values(family)
        moved, total = perturbation_report(family, values)
        print(f"  [{family.name}: {moved}/{total} members off their default]")
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

    # _check_block_is_a_table. A block is a TABLE of members; there is no
    # shorthand form, so a scalar or a list where a table belongs is refused
    # rather than half-read.
    refuses(
        "a block that is not a table",
        lambda: resolve_config(models={cathode.name: ["cathode_neutral_jet"]}),
        must_name=["must be a table of member keys", "list"],
    )

    # _split_none_valued, first refusal. none_valued carries member key NAMES;
    # a bare string is the plausible mistake (TOML arrays and strings look alike
    # to a reader in a hurry) and it would otherwise iterate as characters.
    bad_none_valued = dict(family_values(cathode))
    bad_none_valued["none_valued"] = "cathode_jet_R_N"
    refuses(
        "none_valued given a string instead of an array of names",
        lambda: resolve_config(models={cathode.name: bad_none_valued}),
        must_name=["none_valued must be an array of member key", "cathode_jet_R_N"],
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

    # The same refusal's OTHER two owner branches: a real config key that no
    # family owns at all. The remedy differs per namespace ([params] vs
    # [flags]), and the message resolves the namespace itself -- which is the
    # branch a reader of the refusal table needs to be true.
    unowned = dict(family_values(cathode))
    unowned["C_R"] = 8.76
    unowned["neutral_baffles"] = True
    refuses(
        "a block naming config keys no family owns (both namespaces)",
        lambda: resolve_config(models={cathode.name: unowned}),
        must_name=[
            "C_R: an input_dict key no family owns",
            "state it flat under [params]",
            "neutral_baffles: an input_flags key no family owns",
            "state it flat under [flags]",
        ],
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

    # kinetic_two_moment's INTERNAL-MEMBER refusal, on the FLAT route. The two
    # internal members of the radial closure are armed while the selector is
    # left at its default, which is the shape the mistake actually takes: a
    # reader arms the partition and its cross section without engaging the
    # closure that gives them a reading. The refusal must collect BOTH members
    # in one message and name the selection that would make them readable --
    # the whole decision, not the first guard hit. Until this check, only
    # scripts/preset_resolver_demo.py (at commit 48be9a4, retired 2026-09-03)
    # (unregistered, so nothing runs it in a
    # gate) reached this branch.
    def _two_moment_members_armed():
        params, flags = default_config()
        params["nx"] = 60
        flags["neutral_wall_momentum_partition"] = True
        params["neutral_wall_partition_sigma_hehe_cm2"] = 1.26e-15
        from cablp.solvers._sim1d import LAPDSim1D

        LAPDSim1D(params, flags)

    refuses(
        "kinetic_two_moment internal members armed without the selection",
        _two_moment_members_armed,
        must_name=[
            "flags:neutral_wall_momentum_partition",
            "params:neutral_wall_partition_sigma_hehe_cm2",
            "neutral_momentum_radial='kinetic_two_moment'",
            "have no reading under",
            "engaging it is the whole decision",
        ],
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


def gate_toml_routes():
    """The two TOML routes: ``load_config`` and a committed-stance file.

    Both are exercised against the SAME block, and both are compared to the
    flat form of that block, so a divergence between the file routes and the
    Python API cannot hide.
    """
    print("\n=== TOML ROUTES ===")
    import tempfile

    from cablp.solvers._sim1d import load_config
    import stance_config

    anode = next(
        f for f in DECLARED_FAMILIES if f.name == "anode_surface_recycle"
    )
    values = family_values(anode)
    flat_params, flat_flags = split_by_namespace(anode, values)
    expected = canonical(*resolve_config(flat_params, flat_flags))

    body = "\n".join(
        f"{key} = {json.dumps(values[key])}"
        for _space, key in anode.members
        if values[key] is not None
    )
    nulls = [key for _space, key in anode.members if values[key] is None]
    block_text = f"[models.{anode.name}]\n{body}\n"
    if nulls:
        block_text += f"none_valued = {json.dumps(nulls)}\n"

    with tempfile.TemporaryDirectory() as tmp:
        config_path = Path(tmp) / "declm_block.toml"
        config_path.write_text(block_text)
        check(
            "load_config reads [models.<family>] and matches the flat form",
            canonical(*load_config(config_path)) == expected,
        )

        # The stance route, exercised against a real stance directory so the
        # loader's own name/table/namespace checks all run.
        stance_dir = Path(tmp) / "stances"
        stance_dir.mkdir()
        (stance_dir / "declmtest.toml").write_text(
            "[input_dict]\nnx = 8\n\n[input_flags]\n\n" + block_text
        )
        real_dir = stance_config.STANCE_DIR
        stance_config.STANCE_DIR = stance_dir
        try:
            stance = stance_config.load_stance("declmtest")
            projected = {
                key: stance.params[key]
                for _space, key in anode.members
                if key in stance.params
            }
            check(
                "a stance's block is projected into Stance.params",
                projected == {k: v for k, v in values.items()},
                f"got {projected}",
            )
            check(
                "the block is also kept as written on Stance.models",
                anode.name in stance.models,
            )
            refuses(
                "a stance whose block collides with its own flat table",
                lambda: _bad_stance(stance_config, stance_dir, block_text),
                must_name=["answered TWICE", "anode_neutral_jet"],
            )
        finally:
            stance_config.STANCE_DIR = real_dir


def _bad_stance(stance_config, stance_dir, block_text):
    (stance_dir / "declmbad.toml").write_text(
        "[input_dict]\nanode_neutral_jet = true\n\n[input_flags]\n\n" + block_text
    )
    stance_config.load_stance("declmbad")


def gate_hoisted_guards():
    """The FIVE run-time-first guards, now refused at construction.

    Each of these domains was first checked only once a run was already moving
    -- or, for the fifth, never checked at all.

    NEGATIVE CONTROL for the first four, run at base commit aa65468 before the
    hoist: all four of those configurations CONSTRUCTED, which is what made them
    run-time-first rather than merely redundant. Reproduce it with::

        git archive aa65468 cablp | tar -x -C <tmp> && PYTHONPATH=<tmp> ...

    NEGATIVE CONTROL for the fifth (``beam_deposition_model``, hoisted
    2026-08-30 with the g1atrim block-form migration), run the same way at base
    commit ca444dd::

        git archive ca444dd | tar -x -C <tmp> && PYTHONPATH=<tmp> \\
            python -c "from cablp.solvers._sim1d import LAPDSim1D, \\
                default_config; p, f = default_config(); p['nx'] = 8; \\
                p['beam_deposition_model'] = 'cdsa'; LAPDSim1D(p, f)"

    At ca444dd that CONSTRUCTS and runs, carrying 'cdsa' and silently selecting
    beer_lambert: every read of the key is an equality test against 'csda' with
    a beer_lambert fallback, in solver.py (five sites), physics/cathode.py and
    core/validation.py, so no per-call check refuses a name outside the domain.
    That makes this one worse than late -- there was no later check to reach.
    The domain now lives once, exported as
    ``physics.cathode.BEAM_DEPOSITION_MODELS``.

    The per-call checks of the first four are deliberately still in place; these
    are additional construction-time refusals, not replacements.
    """
    print("\n=== HOISTED RUN-TIME-FIRST GUARDS (the four + the fifth) ===")
    from cablp.solvers._sim1d import LAPDSim1D

    params, flags = default_config()
    params["nx"] = 8
    check("the shipped defaults still construct", _constructs(params, flags))
    for key, value, needle in (
        ("dt_growth_factor", 0.9, "dt_growth_factor must be > 1"),
        ("operator_splitting", "stang", "operator_splitting must be one of"),
        ("implicit_heat_scheme", "tr_bdf3", "implicit_heat_scheme must be one of"),
        ("ion_neutral_drag_model", "slipp", "ion_neutral_drag_model must be one of"),
        ("beam_deposition_model", "cdsa", "beam_deposition_model must be one of"),
    ):
        refuses(
            f"{key}={value!r} refused AT CONSTRUCTION",
            lambda k=key, v=value: LAPDSim1D(dict(params, **{k: v}), dict(flags)),
            must_name=[needle],
        )
    # POSITIVE controls for the fifth: both accepted names still construct, so
    # the new check refuses the typo and nothing else.
    for value in ("csda", "beer_lambert"):
        check(
            f"beam_deposition_model={value!r} still constructs",
            _constructs(dict(params, beam_deposition_model=value), flags),
        )


def _constructs(params, flags):
    from cablp.solvers._sim1d import LAPDSim1D

    try:
        LAPDSim1D(dict(params), dict(flags))
    except Exception as error:  # pragma: no cover - reported by the caller
        print(f"        {error}")
        return False
    return True


def main():
    print(f"import provenance: {cablp.__file__}")
    gate_equivalence()
    gate_equivalence_constructed()
    gate_none_valued()
    gate_toml_routes()
    gate_refusals()
    gate_hoisted_guards()
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
