"""Top-level model selections, the keys each one owns, and their resolver.

The user-facing config surface is MODEL SELECTIONS, not flag stacks. A
top-level selection -- ``neutral_model = "kinetic_dvm"``,
``neutral_momentum_radial = "kinetic_two_moment"`` -- OWNS a set of member
keys whose value the selection determines. This module carries those sets as
DATA, one per family, and resolves them at construction:

* a member left AT ITS CONFIG DEFAULT is set to the value the selection
  requires, automatically -- a stance or a brief never hand-clears it;
* a member the caller EXPLICITLY set (its value differs from the config
  template's) to something the selection refuses raises ONE ``ValueError``
  naming the selection, every offending key with its required-vs-given
  value and a one-line WHY, and the complete set the selection owns;
* a FAMILY-INTERNAL key -- one that has no meaning at all unless the
  selection is engaged -- raises the same single collected error when it is
  armed while the selection is not.

What this replaces is the one-guard-at-a-time cascade: engage a selection,
read the refusal, clear the key it names, run again, read the next refusal.
The member sets below ARE that cascade, flattened.

**Authority.** These sets are MEASURED -- obtained by constructing the arm,
reading what ``LAPDSim1D`` refuses, clearing the key that refusal names, and
repeating until it constructs. They are not invariants: a flipped default or
a new presence-gate changes them, and the way to re-measure is to run the
instrument suite (``scripts/verify_sim1d_k2_dvm.py`` for the DVM arm) and
read what the solver refuses. Do not extend them by guessing.

**The guards stay.** Every single-key guard these sets flatten is still in
``solver.py`` / ``validation.py`` and is untouched. The resolver runs FIRST,
so on a resolvable config the guards see values that already satisfy them;
they remain the authority on what each edge means, and this module's WHY
strings are summaries of them.

**Prerequisites are not members.** A selection that REQUIRES another control
to be ON (``kinetic_dvm`` requires ``neutral_two_zone``;
``kinetic_two_moment`` requires ``neutral_momentum`` and
``neutral_two_zone``) keeps its own standalone guard. Those are not
incompatibilities to be cleared away -- turning them on silently would arm
physics the caller did not ask for -- and every one of them is already ON in
the shipped defaults, so nothing is resolved there in practice.
"""

from .config import input_dict_template_1d, input_flags_template_1d

PARAMS = "params"
FLAGS = "flags"


# Package defaults a ``kinetic_dvm`` arm CANNOT carry, with the value that
# takes each one out of the branch the solver refuses. Entries are
# ``(namespace, key, required_value, why)`` -- the chain spans BOTH
# namespaces, so the namespace is carried explicitly rather than inferred,
# and the WHY rides with the entry so the collected refusal can quote it.
#
# These are refusals BY CONSTRUCTION, not preferences: the DVM state already
# owns the neutral first moment, and everything below is either that same
# ownership restated or a presence-gate standing on it.
#
# THE LIST IS THE MEASURED DEPTH OF THE CASCADE ON THIS HEAD. The first six
# entries are the measurement made from ``default_config()`` (the k2_dvm
# fixture's own base, transcribed here with its per-entry reasons); the
# remainder were measured the same way on 2026-08-23 from the g1atrim stance
# and from the guards that stand on the first six, and were absent from the
# earlier measurement only because their config defaults are ALREADY the
# value this arm requires. Every required value below equals its config
# default except the first six, so the later entries can only ever turn a
# cascade into one collected refusal -- they resolve nothing and change no
# configuration that constructs today.
KINETIC_DVM_INCOMPATIBLE_DEFAULTS = (
    (
        FLAGS, "neutral_momentum", False,
        "The DVM state carries the neutral momentum as the first moment of "
        "f, so an evolved M_n field would be a second, unowned copy of it.",
    ),
    (
        FLAGS, "neutral_energy", False,
        "The frictional half of the collisional energy is booked against "
        "the relative velocity u - u_n, which has no meaning without a "
        "neutral wind.",
    ),
    (
        FLAGS, "neutral_hot_internal_wall", False,
        "It walls the CX-born hot channel's ballistic flight; without "
        "neutral_energy there is no hot channel and the flag would be "
        "inert.",
    ),
    (
        PARAMS, "cathode_neutral_jet", False,
        "The cathode/anode jets and the mesh accommodation are M_n momentum "
        "physics and require the neutral_momentum flag.",
    ),
    (
        PARAMS, "cathode_jet_surface_debit", False,
        "It reads the cathode jet's R_E, so it requires cathode_neutral_jet.",
    ),
    (
        PARAMS, "cathode_jet_energy_convention", "legacy",
        "'total_reflected' rescales the cathode jet's launch energy and "
        "requires cathode_neutral_jet; 'legacy' is the historical reading "
        "that does not.",
    ),
    # --- measured 2026-08-23, same method, from the stance and the guards
    #     standing on the six above. Required value == config default for
    #     every one of them.
    (
        PARAMS, "neutral_momentum_radial", "uniform",
        "'two_zone' and 'kinetic_two_moment' close the radial profile of "
        "the EVOLVED wind and require the neutral_momentum flag.",
    ),
    (
        PARAMS, "anode_neutral_jet", False,
        "The cathode/anode jets and the mesh accommodation are M_n momentum "
        "physics and require the neutral_momentum flag.",
    ),
    (
        PARAMS, "neutral_mesh_accommodation", False,
        "The cathode/anode jets and the mesh accommodation are M_n momentum "
        "physics and require the neutral_momentum flag.",
    ),
    (
        PARAMS, "anode_jet_energy_convention", None,
        "It declares the reading of the ANODE jet's R_E and requires "
        "anode_neutral_jet; undeclared (None) is the only value that stands "
        "without the jet.",
    ),
    (
        FLAGS, "neutral_hot_birth_drift", False,
        "It directs the CX-born hot channel's birth kinematics; without "
        "neutral_energy there is no hot channel and the flag would be "
        "inert.",
    ),
    (
        PARAMS, "cathode_jet_hot_carrier", False,
        "It carries the cathode jet's backscatter share and needs that jet, "
        "its surface debit, and an En field for the CX partner atoms.",
    ),
    (
        PARAMS, "neutral_knudsen_temperature", "frozen",
        "'local' scales the Knudsen conductances by the evolved per-cell "
        "Tn, which only exists under neutral_energy.",
    ),
)


# The same measurement for ``neutral_momentum_radial = "kinetic_two_moment"``
# (2026-08-23ag; the later entries measured 2026-08-23 with the DVM set
# above). The reduction gives the annulus its own momentum row while nothing
# gives it an energy row, so the whole neutral-ENERGY package is what this
# closure refuses -- ``neutral_energy`` itself and every control standing on
# it. As above, only the first two differ from their config defaults.
KINETIC_TWO_MOMENT_INCOMPATIBLE_DEFAULTS = (
    (
        FLAGS, "neutral_energy", False,
        "The reduction gives the annulus its own momentum row while nothing "
        "gives it an energy row, so the single cold fluid the mini-flux "
        "transports would be split across two momenta and one energy.",
    ),
    (
        FLAGS, "neutral_hot_internal_wall", False,
        "It walls the CX-born hot channel's ballistic flight; without "
        "neutral_energy there is no hot channel and the flag would be "
        "inert.",
    ),
    (
        FLAGS, "neutral_hot_birth_drift", False,
        "It directs the CX-born hot channel's birth kinematics; without "
        "neutral_energy there is no hot channel and the flag would be "
        "inert.",
    ),
    (
        PARAMS, "cathode_jet_hot_carrier", False,
        "Every charge exchange along the beam returns an atom born at the "
        "local ion state, and without an En field there is nowhere to book "
        "the (3/2) k Ti it carries.",
    ),
    (
        PARAMS, "neutral_knudsen_temperature", "frozen",
        "'local' scales the Knudsen conductances by the evolved per-cell "
        "Tn, which only exists under neutral_energy.",
    ),
)


# Keys that belong to the two-moment closure and have no reading at all
# without it: the flag partitions the wall branch of the two-zone momentum
# operator, and the cross section is the number that partition is built
# from. Entries are ``(namespace, key, why)``.
KINETIC_TWO_MOMENT_INTERNAL_MEMBERS = (
    (
        FLAGS, "neutral_wall_momentum_partition",
        "It partitions the wall branch of the two-zone momentum operator, "
        "and no other radial closure carries an annulus momentum row for "
        "that branch to act on.",
    ),
    (
        PARAMS, "neutral_wall_partition_sigma_hehe_cm2",
        "The He-He elastic cross section sets the mean free path the "
        "partition's survival weight is built from; it is read only under "
        "the neutral_wall_momentum_partition flag.",
    ),
)


class ModelFamily:
    """One top-level model selection and the member keys it owns.

    ``selector_space``/``selector_key`` name the control that engages the
    family and ``engaged_value`` the value that engages it. ``members`` is
    the measured incompatibility set as
    ``(namespace, key, required_value, why)``; ``internal_members`` is
    ``(namespace, key, why)`` for keys that exist only under this selection.
    """

    __slots__ = (
        "name",
        "selector_space",
        "selector_key",
        "engaged_value",
        "members",
        "internal_members",
    )

    def __init__(
        self,
        name,
        selector_space,
        selector_key,
        engaged_value,
        members,
        internal_members=(),
    ):
        self.name = name
        self.selector_space = selector_space
        self.selector_key = selector_key
        self.engaged_value = engaged_value
        self.members = tuple(members)
        self.internal_members = tuple(internal_members)

    @property
    def selection(self):
        """The selection as it is written in a config, for error text."""
        return f"{self.selector_key}={self.engaged_value!r}"


#: Every family the resolver owns, in resolution order. ``kinetic_dvm`` runs
#: first: it forbids the radial closure the two-moment family selects, so a
#: config asking for both is refused by the DVM's own member set rather than
#: half-resolved by the other family first.
MODEL_FAMILIES = (
    ModelFamily(
        name="kinetic_dvm",
        selector_space=PARAMS,
        selector_key="neutral_model",
        engaged_value="kinetic_dvm",
        members=KINETIC_DVM_INCOMPATIBLE_DEFAULTS,
    ),
    ModelFamily(
        name="kinetic_two_moment",
        selector_space=PARAMS,
        selector_key="neutral_momentum_radial",
        engaged_value="kinetic_two_moment",
        members=KINETIC_TWO_MOMENT_INCOMPATIBLE_DEFAULTS,
        internal_members=KINETIC_TWO_MOMENT_INTERNAL_MEMBERS,
    ),
)


def _template(space):
    return input_flags_template_1d if space == FLAGS else input_dict_template_1d


def _default_of(space, key):
    return _template(space).get(key)


def _same(a, b):
    """Value equality that is safe for anything a config key can hold.

    Config values include bools, strings, ``None``, floats and per-cell
    lists; ``==`` on the last of those can return an array, so the result is
    forced to a plain bool and anything that will not compare is reported as
    different.
    """
    if a is b:
        return True
    try:
        return bool(a == b)
    except Exception:  # pragma: no cover - defensive, no config key does this
        return False


def _describe(space, key, value):
    return f"{space}:{key} = {value!r}"


def _owned_set_block(family):
    lines = [
        f"The complete set {family.selection} owns (required value in "
        "brackets; a key left at its config default is resolved to it "
        "automatically, so nothing here has to be cleared by hand):"
    ]
    for space, key, required, why in family.members:
        lines.append(f"  {space}:{key} [{required!r}]")
        lines.append(f"      WHY {why}")
    return lines


def _raise_member_conflicts(family, conflicts):
    lines = [
        f"{family.selection} owns configuration keys this config also sets "
        f"explicitly, to values the selection refuses ({len(conflicts)} of "
        "them). A model selection and its members cannot both be chosen: "
        "drop the selection, or drop these settings.",
        "",
    ]
    for space, key, required, given, why in conflicts:
        lines.append(f"  {space}:{key}: required {required!r}, given {given!r}")
        lines.append(f"      WHY {why}")
    lines.append("")
    lines.extend(_owned_set_block(family))
    raise ValueError("\n".join(lines))


def _raise_internal_members_armed(family, selector_given, armed):
    names = ", ".join(f"{space}:{key}" for space, key, _given, _why in armed)
    lines = [
        f"{names} belong to the {family.selection} closure and have no "
        f"reading under {family.selector_key}={selector_given!r}; they are "
        "armed here anyway.",
        "",
    ]
    for space, key, given, why in armed:
        lines.append(f"  {_describe(space, key, given)}")
        lines.append(f"      WHY {why}")
    lines.append("")
    lines.append(
        f"Set {family.selection} to use them. That selection additionally "
        "owns the keys below, so engaging it is the whole decision, not "
        "half of one:"
    )
    lines.extend(_owned_set_block(family)[1:])
    raise ValueError("\n".join(lines))


def resolve_model_families(params, flags):
    """Resolve every engaged model family IN PLACE, or raise once.

    ``params`` and ``flags`` are the RESOLVED config mappings (caller
    overrides already merged onto the templates by
    :func:`~cablp.solvers._sim1d.core.config.resolve_config`), which is what
    makes "explicitly set" decidable here: a member whose value equals its
    template default was not chosen by the caller, and a member whose value
    differs was.

    This runs BEFORE every other construction-time validator -- before
    ``validate_r1_configuration_presence``, before the deprecation register,
    and before each ``_init_*`` phase -- so the guards those phases carry see
    a configuration the selection has already made self-consistent. Nothing
    downstream is bypassed or deleted; the guards remain, and remain the
    authority on what each edge means.

    Returns the same two mappings, mutated. Raises ``ValueError`` naming one
    family, its complete member set and the offending keys.
    """
    spaces = {PARAMS: params, FLAGS: flags}
    for family in MODEL_FAMILIES:
        selector_given = spaces[family.selector_space].get(family.selector_key)
        if _same(selector_given, family.engaged_value):
            conflicts = []
            for space, key, required, why in family.members:
                given = spaces[space].get(key)
                if _same(given, required):
                    continue
                if _same(given, _default_of(space, key)):
                    spaces[space][key] = required
                    continue
                conflicts.append((space, key, required, given, why))
            if conflicts:
                _raise_member_conflicts(family, conflicts)
            continue
        armed = []
        for space, key, why in family.internal_members:
            given = spaces[space].get(key)
            if _same(given, _default_of(space, key)):
                continue
            armed.append((space, key, given, why))
        if armed:
            _raise_internal_members_armed(family, selector_given, armed)
    return params, flags
