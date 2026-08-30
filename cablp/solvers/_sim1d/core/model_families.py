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


# ------------------------------------------------------------------------
# DECLARATION-BLOCK FAMILIES
#
# The sets above answer "what does this SELECTION force?". The sets below
# answer the other half: "what is this family's COMPLETE MEMBERSHIP?" -- every
# key whose meaning belongs to the family, whether the family forces its value,
# merely reads it, or refuses it. That is the inventory a declaration block
# states, and the two halves are deliberately different data: a selection's
# forced set is a subset of a family's membership (the R coefficients a jet
# reads, for instance, are members that no selection forces).
#
# Membership is (space, key) only. A declaration block supplies every VALUE
# itself -- that is the 24d option (b) ruling, declaration entries explicit
# regardless of value -- so there is nothing here for a default to drift
# against. The space rides with the key for the same reason it does above: the
# families span BOTH namespaces, and a declaration block is written WITHOUT
# namespaces precisely so a member cannot be filed into the wrong one.
#
# EVERY SET BELOW IS MEASURED, by the same method as the sets above and on the
# same terms: read what the guards couple, not what a name suggests. Where the
# measurement disagreed with the 2026-08-23 census that commissioned this pass,
# the CODE won and the disagreement is recorded in the entry.

#: Family A, DVM half. The selection plus every key it forces.
NEUTRAL_CLOSURE_MEMBERS = ((PARAMS, "neutral_model"),) + tuple(
    (space, key) for space, key, _required, _why in KINETIC_DVM_INCOMPATIBLE_DEFAULTS
)

#: Family A, two-moment half. The selection, every key it forces, and the two
#: keys that have no reading without it.
NEUTRAL_RADIAL_CLOSURE_MEMBERS = (
    ((PARAMS, "neutral_momentum_radial"),)
    + tuple(
        (space, key)
        for space, key, _required, _why in KINETIC_TWO_MOMENT_INCOMPATIBLE_DEFAULTS
    )
    + tuple((space, key) for space, key, _why in KINETIC_TWO_MOMENT_INTERNAL_MEMBERS)
)

#: Family K -- beam deposition, the anomalous (quasilinear) channel, and the
#: walked tail. MEASURED 2026-08-30 at 22 keys, not the census's "~10": the
#: census counted the tail spine and left out the excitation trio, the clump
#: pair, the anode-reflect riders and the two interception FLAGS, all of which
#: the same guard block couples. The enforced chain is likewise 7 deep, not 4
#: (beam_anode_interception -> beam_tail_anode_interception -> the riders ->
#: tail_ionization -> the walked-tail selector -> beam_anomalous_model ->
#: beam_deposition_model); the census's "depth 4" is the middle spine read
#: upward. This is the family the 23ag detonation class came from -- change one
#: key and the guards refuse one at a time, which is exactly what declaring the
#: whole membership at once removes.
BEAM_TAIL_CLOSURE_MEMBERS = (
    (PARAMS, "beam_deposition_model"),
    (PARAMS, "beam_coulomb_model"),
    (PARAMS, "beam_anomalous_model"),
    (PARAMS, "ql_relaxation_coeff"),
    (PARAMS, "beam_product_transport"),
    (PARAMS, "heating_anomalous_transport"),
    (PARAMS, "heating_anomalous_disposal"),
    (PARAMS, "heating_anomalous_tail_energy_eV"),
    (PARAMS, "heating_anomalous_tail_ionization"),
    (PARAMS, "heating_anomalous_tail_energy_keying"),
    (PARAMS, "heating_anomalous_tail_phi_c_fraction"),
    (PARAMS, "heating_anomalous_tail_cathode_boundary"),
    (PARAMS, "beam_tail_anode_reflected_particles"),
    (PARAMS, "beam_tail_anode_reflected_energy"),
    (PARAMS, "beam_clump_fraction"),
    (PARAMS, "beam_clump_enhancement"),
    (PARAMS, "beam_deposition_smoothing_cm"),
    (PARAMS, "b_beam_excitation"),
    (PARAMS, "beam_excitation_model"),
    (PARAMS, "beam_excitation_energy_eV"),
    (FLAGS, "beam_anode_interception"),
    (FLAGS, "beam_tail_anode_interception"),
)

#: Family B, cathode half -- the cathode surface's directed-recycle channel.
#: The two R coefficients are members: they are the surface property the jet
#: reads, and a declaration that states the jet without them is not the whole
#: decision.
CATHODE_SURFACE_RECYCLE_MEMBERS = (
    (PARAMS, "cathode_neutral_jet"),
    (PARAMS, "cathode_jet_R_N"),
    (PARAMS, "cathode_jet_R_E"),
    (PARAMS, "cathode_jet_energy_convention"),
    (PARAMS, "cathode_jet_surface_debit"),
    (PARAMS, "cathode_jet_hot_carrier"),
)

#: Family B, anode half. ``neutral_mesh_accommodation`` is a member of THIS
#: half: it is armed with the anode jet, shares the jet's anode-face and
#: transparency requirements, and the stance arms the two together.
ANODE_SURFACE_RECYCLE_MEMBERS = (
    (PARAMS, "anode_neutral_jet"),
    (PARAMS, "anode_jet_R_N"),
    (PARAMS, "anode_jet_R_E"),
    (PARAMS, "anode_jet_energy_convention"),
    (PARAMS, "neutral_mesh_accommodation"),
)

#: Family F -- how the initial neutral state is established.
INITIAL_NEUTRAL_STATE_MEMBERS = (
    (FLAGS, "neutral_equilibration"),
    (FLAGS, "launch_plasma_after_equilibration"),
    (PARAMS, "neutral_equilibration_cycles"),
    (PARAMS, "neutral_equilibration_dt"),
    (PARAMS, "equilibration_gas_puff_on_s"),
    (FLAGS, "use_cached_neutral_seed"),
    (PARAMS, "neutral_seed_cache_dir"),
    (FLAGS, "neutral_initial_profile"),
    (PARAMS, "nn0"),
    (PARAMS, "nn0_profile"),
    (PARAMS, "nn0_annulus_profile"),
    (PARAMS, "restart_from"),
)

#: The mutually exclusive routes inside :data:`INITIAL_NEUTRAL_STATE_MEMBERS`,
#: as ``(route, (space, key), off_value, why)``: a route is ARMED when its key
#: holds anything other than ``off_value``, and at most one may be armed.
#:
#: THREE routes, not the census's four. MEASURED 2026-08-30: ``cached_seed`` is
#: NOT a fourth exclusive route -- it REQUIRES ``neutral_equilibration`` (the
#: co-requisite is validation.py's ``use_cached_neutral_seed is ON but the
#: configuration is incoherent`` refusal) and its dispatch is a hit/miss branch
#: INSIDE the equilibration path, so it is a modifier of ``equilibrate``, not
#: an alternative to it. That also settles the count: three routes have three
#: pairs, and the code carries exactly three direct pairwise refusals, not the
#: census's six. Collapsing the four-route reading into one selector would have
#: made cached_seed and equilibrate mutually exclusive -- a behaviour change,
#: and one this migration is forbidden to make.
INITIAL_NEUTRAL_STATE_ROUTES = (
    (
        "equilibrate", (FLAGS, "neutral_equilibration"), False,
        "start_simulation() runs the puff/off accumulation and seeds nn from "
        "it, overwriting whatever the initial condition put there.",
    ),
    (
        "profile", (FLAGS, "neutral_initial_profile"), False,
        "the shaped per-cell nn0_profile IS the initial fill, and it "
        "supersedes the scalar nn0 for both zones.",
    ),
    (
        "restart", (PARAMS, "restart_from"), None,
        "a restart payload replaces the whole initial condition, neutrals "
        "included.",
    ),
)


def _template(space):
    return input_flags_template_1d if space == FLAGS else input_dict_template_1d


def member_default(space, key):
    """The config template's value for one member, in its own namespace."""
    return _template(space).get(key)


def values_equal(a, b):
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
        if values_equal(selector_given, family.engaged_value):
            conflicts = []
            for space, key, required, why in family.members:
                given = spaces[space].get(key)
                if values_equal(given, required):
                    continue
                if values_equal(given, member_default(space, key)):
                    spaces[space][key] = required
                    continue
                conflicts.append((space, key, required, given, why))
            if conflicts:
                _raise_member_conflicts(family, conflicts)
            continue
        armed = []
        for space, key, why in family.internal_members:
            given = spaces[space].get(key)
            if values_equal(given, member_default(space, key)):
                continue
            armed.append((space, key, given, why))
        if armed:
            _raise_internal_members_armed(family, selector_given, armed)
    return params, flags


#: Names a declaration block reserves for itself, so they can never be members.
RESERVED_BLOCK_KEYS = frozenset({"none_valued"})


class DeclaredFamily:
    """One family as a DECLARATION BLOCK states it: a name and a membership.

    ``members`` is the family's COMPLETE inventory as ``(space, key)`` pairs,
    in the order a block should read. A block must state every one of them and
    nothing else; the values are the block's, not this object's.

    ``selector`` / ``engaged_value``, when present, name the member whose value
    engages the family. A block for a family with a selector must engage it --
    declaring the membership of a family you are not selecting is declaring
    keys that belong to whatever you selected instead.

    ``routes`` is the optional mutual-exclusion group described at
    :data:`INITIAL_NEUTRAL_STATE_ROUTES`.
    """

    __slots__ = ("name", "summary", "members", "selector", "engaged_value", "routes")

    def __init__(
        self, name, summary, members, selector=None, engaged_value=None, routes=()
    ):
        self.name = name
        self.summary = summary
        self.members = tuple(members)
        self.selector = selector
        self.engaged_value = engaged_value
        self.routes = tuple(routes)

    def owns(self, key):
        """Return the namespace of member ``key``, or ``None``."""
        for space, member in self.members:
            if member == key:
                return space
        return None


#: Every family a declaration block may name. A key appears in more than one
#: family here (``neutral_momentum_radial`` is the two-moment selector AND a
#: member the DVM selection forces; the jet keys are members of family B AND of
#: the DVM set that forbids them), which is why two blocks claiming one key is
#: refused rather than merged: overlapping membership means the two families
#: disagree about who owns the decision, and only the caller can settle it.
DECLARED_FAMILIES = (
    DeclaredFamily(
        name="neutral_closure",
        summary="the neutral closure: which model carries the neutral state",
        members=NEUTRAL_CLOSURE_MEMBERS,
        selector="neutral_model",
        engaged_value="kinetic_dvm",
    ),
    DeclaredFamily(
        name="neutral_radial_closure",
        summary="the radial closure of the evolved neutral wind",
        members=NEUTRAL_RADIAL_CLOSURE_MEMBERS,
        selector="neutral_momentum_radial",
        engaged_value="kinetic_two_moment",
    ),
    DeclaredFamily(
        name="beam_tail_closure",
        summary=(
            "beam deposition, the anomalous channel and the walked tail"
        ),
        members=BEAM_TAIL_CLOSURE_MEMBERS,
    ),
    DeclaredFamily(
        name="cathode_surface_recycle",
        summary="the cathode surface's directed-recycle channel",
        members=CATHODE_SURFACE_RECYCLE_MEMBERS,
    ),
    DeclaredFamily(
        name="anode_surface_recycle",
        summary="the anode mesh's directed-recycle channel",
        members=ANODE_SURFACE_RECYCLE_MEMBERS,
    ),
    DeclaredFamily(
        name="initial_neutral_state",
        summary="how the initial neutral state is established",
        members=INITIAL_NEUTRAL_STATE_MEMBERS,
        routes=INITIAL_NEUTRAL_STATE_ROUTES,
    ),
)


def _self_check():
    """Refuse a malformed family table AT IMPORT, not at first use.

    Every member must exist in the namespace its entry claims, every family
    name must be unique, a selector must be one of its own family's members,
    and a route key must be one too. A typo in the tables above is a defect in
    the config surface's own description, so it fails here rather than becoming
    a confusing refusal on somebody's arm.
    """
    seen = set()
    for family in DECLARED_FAMILIES:
        if family.name in seen:
            raise RuntimeError(f"duplicate declared family {family.name!r}")
        seen.add(family.name)
        if RESERVED_BLOCK_KEYS.intersection(m for _s, m in family.members):
            raise RuntimeError(
                f"declared family {family.name!r} owns a member whose name is "
                f"reserved inside a block: {sorted(RESERVED_BLOCK_KEYS)}"
            )
        for space, key in family.members:
            if key not in _template(space):
                raise RuntimeError(
                    f"declared family {family.name!r} names {space}:{key}, "
                    f"which the {space} template does not own"
                )
        if family.selector is not None and family.owns(family.selector) is None:
            raise RuntimeError(
                f"declared family {family.name!r} has selector "
                f"{family.selector!r} outside its own membership"
            )
        for _route, (space, key), _off, _why in family.routes:
            if (space, key) not in family.members:
                raise RuntimeError(
                    f"declared family {family.name!r} has a route on "
                    f"{space}:{key}, which is outside its own membership"
                )


_self_check()
