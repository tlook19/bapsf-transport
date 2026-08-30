"""Declaration blocks: a model selection stated as its complete member set.

THE PROBLEM. Model-specific keys live in one flat top-level namespace, so a
config carries every family's keys at all times -- including the families it
did not select, at values that are meaningless or that the selected model
refuses outright. Nothing in the flat form says which keys belong together, so
a selection is assembled by hand, one key at a time, and verified by running
into the guards one refusal at a time. That is the cascade
:mod:`~cablp.solvers._sim1d.core.model_families` flattened for the two families
it owns; a DECLARATION BLOCK removes the shape that produces it.

THE FORM. A block names a family and states its COMPLETE membership::

    [models.cathode_surface_recycle]
    cathode_neutral_jet = true
    cathode_jet_R_N = 0.34
    cathode_jet_R_E = 0.18
    cathode_jet_energy_convention = "total_reflected"
    cathode_jet_surface_debit = true
    cathode_jet_hot_carrier = false

Three properties, and each one is the point of a different half of the ruling:

* **Explicit regardless of value** (24d option (b)). Every member is written,
  including members at their config default. A block is an INVENTORY, not a
  delta -- reading it tells you the whole decision, and a member that is
  missing is a refusal, not an inherited value.
* **Namespace-free.** ``cathode_neutral_jet`` is an ``input_dict`` key and
  ``neutral_equilibration`` is an ``input_flags`` key, and a block states
  neither fact: the family's membership carries the namespace, and this module
  files each member where it belongs. The driver-side hazard -- a key filed
  into the wrong namespace, silently inert before 2026-08-14 and a run-time
  refusal since -- cannot be expressed in a block at all.
* **One owner per key.** A member may not also appear flat, and two blocks may
  not claim the same key. A value has one home.

WHAT A BLOCK DOES NOT DO. It does not change any value, and it does not
validate physics. A block is projected onto the same two flat namespaces the
solver has always read, and then every construction guard runs exactly as
before -- the presence gates, the domain checks, the family resolver in
``model_families``. A configuration written as blocks and the same
configuration written flat resolve to the identical surface, byte for byte;
that equivalence is the migration's whole claim, and
``scripts/declm_route_identity.py`` is where it is measured.

THE FLAT ROUTE KEEPS WORKING. Blocks are an ADDITIONAL input form. A config
that names no block behaves exactly as it did -- the committed stances, the
campaign drivers and the 295 banked ``.cmd`` files are untouched and are not
rewritten. Where a flat route would become ambiguous under a block (the same
key in both), it fails loudly and the message carries the remedy.
"""

from .config import input_dict_template_1d, input_flags_template_1d
from .model_families import (
    DECLARED_FAMILIES,
    FLAGS,
    PARAMS,
    RESERVED_BLOCK_KEYS,
)

#: Every declared family by name.
FAMILIES_BY_NAME = {family.name: family for family in DECLARED_FAMILIES}


def _known_families():
    return ", ".join(sorted(FAMILIES_BY_NAME))


def _inventory_block(family):
    """The family's complete membership, for an error message's remedy half."""
    lines = [
        f"The complete membership of [models.{family.name}] -- "
        f"{family.summary} -- is the {len(family.members)} key(s) below, and a "
        "declaration block states every one of them (a member at its config "
        "default is still written, so a block reads as the whole decision):"
    ]
    lines.extend(f"  {key}" for _space, key in family.members)
    return lines


def _raise(lines):
    raise ValueError("\n".join(lines))


def _check_block_is_a_table(name, block):
    if not isinstance(block, dict):
        _raise(
            [
                f"[models.{name}] must be a table of member keys and their "
                f"values, not {type(block).__name__}. A declaration block "
                "states a family's complete membership; there is no shorthand "
                "form."
            ]
        )


def _split_none_valued(name, family, block):
    """Return the block's ``{key: value}`` with ``none_valued`` applied.

    ``none_valued`` is a block's TOML affordance and nothing else: TOML has no
    null literal, so a member whose declared value is ``None`` is named in this
    array instead of carrying a value. The Python API passes ``None`` directly
    and never needs it. (The committed stance files use the same convention at
    file scope, so a reader meets it once.)
    """
    values = {key: value for key, value in block.items()
              if key not in RESERVED_BLOCK_KEYS}
    named = block.get("none_valued", ())
    if isinstance(named, str) or not hasattr(named, "__iter__"):
        _raise(
            [
                f"[models.{name}] none_valued must be an array of member key "
                f"NAMES whose declared value is null (got {named!r}).",
                "",
                *_inventory_block(family),
            ]
        )
    for key in named:
        if key in values:
            _raise(
                [
                    f"[models.{name}] declares {key!r} both with a value "
                    f"({values[key]!r}) and in none_valued. A member has one "
                    "declared value, not two."
                ]
            )
        values[key] = None
    return values


def _check_membership(name, family, values):
    """Refuse a block that names a non-member, or omits a member."""
    declared = set(values)
    members = {key for _space, key in family.members}

    strangers = sorted(declared - members)
    if strangers:
        owners = []
        for key in strangers:
            other = next(
                (f.name for f in DECLARED_FAMILIES
                 if f.name != name and f.owns(key) is not None),
                None,
            )
            if other is not None:
                owners.append(f"  {key}: belongs to [models.{other}]")
            elif key in input_dict_template_1d:
                owners.append(f"  {key}: an input_dict key no family owns; "
                              "state it flat under [params]")
            elif key in input_flags_template_1d:
                owners.append(f"  {key}: an input_flags key no family owns; "
                              "state it flat under [flags]")
            else:
                owners.append(f"  {key}: no LAPDSim1D config template owns it")
        _raise(
            [
                f"[models.{name}] declares {len(strangers)} key(s) the family "
                "does not own. A declaration block states its family's "
                "membership and nothing else:",
                "",
                *owners,
                "",
                *_inventory_block(family),
            ]
        )

    missing = [key for _space, key in family.members if key not in declared]
    if missing:
        _raise(
            [
                f"[models.{name}] is INCOMPLETE: {len(missing)} member(s) of "
                f"{len(family.members)} are not declared. A model selection "
                "carries its complete member set -- a member at its config "
                "default is still written, so that a block reads as the full "
                "inventory rather than a delta against defaults that may "
                "move:",
                "",
                *(f"  {key}  (not declared)" for key in missing),
                "",
                *_inventory_block(family),
            ]
        )


def _check_selector_engaged(name, family, values):
    """Refuse a block for a family this config is not selecting."""
    if family.selector is None:
        return
    given = values[family.selector]
    if given == family.engaged_value:
        return
    _raise(
        [
            f"[models.{name}] declares the membership of a family it does not "
            f"select: the block sets {family.selector} = {given!r}, and this "
            f"family is engaged at {family.selector} = "
            f"{family.engaged_value!r}.",
            "",
            "Under a different selection these keys belong to whatever model "
            "IS selected, so declaring them here would claim a decision this "
            "config is not making. Either engage the family, or drop the "
            "block and let the keys stand flat.",
            "",
            *_inventory_block(family),
        ]
    )


def _check_routes(name, family, values):
    """Refuse a block that arms more than one mutually exclusive route."""
    armed = [
        (route, key, values[key], why)
        for route, (_space, key), off, why in family.routes
        if values[key] != off
    ]
    if len(armed) < 2:
        return
    lines = [
        f"[models.{name}] arms {len(armed)} mutually exclusive routes at once. "
        "Exactly one of them establishes the state; two would each overwrite "
        "the other's result, and which one won would depend on call order:",
        "",
    ]
    for route, key, given, why in armed:
        lines.append(f"  {route}: {key} = {given!r}")
        lines.append(f"      WHY {why}")
    lines.append("")
    lines.append(
        "Disarm all but one. The off value of each route is: "
        + ", ".join(
            f"{key}={off!r}"
            for _route, (_space, key), off, _why in family.routes
        )
    )
    _raise(lines)


def resolve_declaration_blocks(models, supplied_params, supplied_flags):
    """Project declaration blocks onto the two flat namespaces.

    ``models`` maps family name to that family's block. Returns
    ``(params, flags)`` -- the two namespaces' worth of member values the
    blocks declare, ready to be merged with the caller's flat overrides.

    Raises ``ValueError``, naming the offender and carrying the remedy, on an
    unknown family, a non-member key, an incomplete membership, a block for an
    unselected family, two blocks claiming one key, a member also supplied
    flat, and a family whose mutually exclusive routes are multiply armed.
    """
    if not models:
        return {}, {}

    unknown = sorted(set(models) - set(FAMILIES_BY_NAME))
    if unknown:
        _raise(
            [
                f"unknown model declaration block(s): "
                f"{', '.join(unknown)}. A block names a family whose "
                "membership the solver knows.",
                "",
                f"Declarable families: {_known_families()}",
            ]
        )

    out = {PARAMS: {}, FLAGS: {}}
    claimed = {}
    for name in sorted(models):
        family = FAMILIES_BY_NAME[name]
        block = models[name]
        _check_block_is_a_table(name, block)
        values = _split_none_valued(name, family, block)
        _check_membership(name, family, values)
        _check_selector_engaged(name, family, values)
        _check_routes(name, family, values)
        for space, key in family.members:
            if key in claimed:
                _raise(
                    [
                        f"{key!r} is declared by TWO model blocks: "
                        f"[models.{claimed[key]}] and [models.{name}]. The two "
                        "families overlap on this key, so they disagree about "
                        "which decision owns it, and only the caller can "
                        "settle that.",
                        "",
                        "Declare the family that owns the decision here and "
                        "drop the other block; the keys the dropped block "
                        "carried then stand flat, where the construction "
                        "guards read them as they always have.",
                    ]
                )
            claimed[key] = name
            out[space][key] = values[key]

    collisions = sorted(
        set(claimed).intersection(set(supplied_params) | set(supplied_flags))
    )
    if collisions:
        _raise(
            [
                f"{len(collisions)} key(s) are supplied BOTH flat and inside a "
                "declaration block. A value has one home:",
                "",
                *(
                    f"  {key}: flat, and in [models.{claimed[key]}]"
                    for key in collisions
                ),
                "",
                "Remove the flat setting -- the block is the whole decision "
                "and states this member itself -- or drop the block and keep "
                "the flat route.",
            ]
        )
    return out[PARAMS], out[FLAGS]
