"""Load a named CONFIGURATION into an ``input_dict`` / ``input_flags`` pair.

Every run names a configuration. ``default_config()`` is the TEMPLATE of keys
and their classes -- never a plasma anyone runs -- and the configuration a run
names is a committed file: ``scripts/stances/g1atrim.toml`` is the LAPD
reference configuration, and the alternates the campaign compares against it
are DERIVED from it.

A BASE configuration is a named, complete, committed package: the whole delta
from :func:`default_config`, in one file, with nothing accreted on the command
line. It is what ``run_m6_point.py --stance`` applies, what
``preflight_diffcfg.py --stance`` diffs against, and what the golden gate is
captured at.

A DERIVED configuration states a ``base`` and the deltas that move it. It is a
first-class object, not a command line: an alternate closure, a reduced model,
a comparator arm -- anything the campaign runs AGAINST the reference -- is
declared as a file, so the run records what it was rather than what was typed.

File format. Two top-level keys and four tables, every one optional except
that a file with no delta at all declares nothing:

``base = "<name>"``
    The configuration this one is derived FROM: a committed file NAME, without
    path or suffix, resolved in ``scripts/stances/``. Chains are allowed to
    ``MAX_CHAIN_FILES`` files (this file and its bases) and refused beyond, and
    a cycle is refused.
``allow_restated = true``
    Permit flat deltas that restate the base's resolved value. Off by default:
    a delta must MOVE something, or it is a line that reads as a decision and
    changes nothing.
``[input_dict]``
    ``input_dict`` keys and their values.
``[input_flags]``
    ``input_flags`` keys and their values.
``[none_valued]``
    ``input_dict`` / ``input_flags`` arrays of key NAMES whose value is
    ``None``. TOML has no null literal, so a key that must be explicitly unset
    is named here instead of appearing in the table above.
``[models.<family>]``
    A DECLARATION BLOCK: one model family's COMPLETE membership, stated
    without namespaces, every member written regardless of value. See
    ``cablp/solvers/_sim1d/core/model_declarations.py`` for the form and its
    refusals. A block is PROJECTED onto the two tables above at load time, so
    ``Stance.params`` / ``Stance.flags`` carry the members like any other
    delta and every consumer of this loader reads them unchanged; the block as
    written is kept on ``Stance.models`` for reporting. A member may not also
    appear in ``[input_dict]`` / ``[input_flags]``.

Every key is checked against the config templates at load time, in the
namespace that owns it: the two namespaces are separate and neither accepts the
other's keys, so a misfiled key raises here, naming the namespace that does own
it, rather than reaching the solver's own construction-time refusal. A block's
members are exempt from that check by construction -- the family membership
carries the namespace, which is why a block cannot misfile a key at all.

Values are TOML-native (float, integer, string, boolean, array); arrays reach
the solver as lists, which is what the config keys that take per-cell profiles
expect.

RESOLUTION ORDER, and it is the whole contract::

    default_config()  ->  base chain, oldest base first  ->  this file's
    deltas  ->  the driver's nx / mesh package

A driver layers only its mesh on top; everything that decides what the plasma
IS comes from the named file. What a load returns alongside the two mappings is
a :class:`~cablp.solvers._sim1d.ConfigurationLineage` -- the name, the base
chain, each file's sha256, the declared delta keys and the resolved
configuration's identity -- which is what a run writes into its HDF5 root so a
saved trajectory can say which configuration produced it.
"""

import hashlib
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS.parent))

from cablp.solvers._sim1d import (  # noqa: E402
    ConfigurationLineage,
    config_identity,
    default_config,
    resolve_declaration_blocks,
)
from cablp.solvers._sim1d.core.model_declarations import (  # noqa: E402
    FAMILIES_BY_NAME,
)
from cablp.solvers._sim1d.core.model_families import (  # noqa: E402
    FLAGS,
    PARAMS,
    values_equal,
)

#: The two namespace tokens a declared family's membership carries, mapped onto
#: this loader's table names.
_SPACE_TABLE = {PARAMS: "input_dict", FLAGS: "input_flags"}

#: Directory holding the committed stance files.
STANCE_DIR = _SCRIPTS / "stances"
SUFFIX = ".toml"
NAMESPACES = ("input_dict", "input_flags")
TABLES = NAMESPACES + ("none_valued", "models")
#: Top-level (non-table) keys a configuration file may carry.
SCALARS = ("base", "allow_restated")

#: THE MESH-SIZED PACKAGE. These four params are per-cell arrays sized to the
#: configuration's OWN mesh, and the two flags below require them. They cannot
#: travel to another resolution and they are not resampled: the vessel profile
#: is a staircase whose steps interpolation would smear into a bore the
#: machine does not have, and the two nn0 profiles are an equilibrated foot
#: computed for that mesh, so resampling them is a new initial condition
#: rather than the configuration's.
#:
#: A caller that must run a named configuration at ITS OWN resolution -- the
#: golden gate on its coarse mesh, a corner sweep on twenty cells -- drops the
#: package WHOLE, with its flags, rather than half-applying it: a prescribed
#: geometry carrying a default fill is a hybrid corner of nobody's choosing.
#: What still travels is every mesh-independent key, which is the whole
#: operating point.
MESH_SIZED_PARAMS = (
    "plasma_radius_profile_cm",
    "machine_radius_profile_cm",
    "nn0_profile",
    "nn0_annulus_profile",
)
MESH_SIZED_FLAGS = (
    "prescribed_area_geometry",
    "neutral_initial_profile",
)


def without_mesh_sized_package(params, flags):
    """Return copies of a configuration's delta with the mesh package dropped.

    The params are REMOVED (so the templates' own values stand) and the flags
    are cleared, because each of those flags REQUIRES the array it reads. The
    caller supplies the initial neutral fill the dropped profile was carrying.
    """
    return (
        {k: v for k, v in params.items() if k not in MESH_SIZED_PARAMS},
        {**flags, **{k: False for k in MESH_SIZED_FLAGS}},
    )


#: How many files one chain may hold, this file included. Three is a base, a
#: derivation of it, and a derivation of that -- enough to say "the reference,
#: the campaign's variant of it, this arm's variant of that" and shallow enough
#: that a reader can still hold the whole chain in mind. A fourth file is
#: refused rather than silently resolved, because past that depth nobody can
#: say what a value came from without running the loader.
MAX_CHAIN_FILES = 3


@dataclass(frozen=True)
class Stance:
    """One loaded configuration: its name, its file, and its two deltas.

    ``params`` and ``flags`` are read-only mappings of the CUMULATIVE delta
    from :func:`default_config` -- this file's deltas over its base chain's,
    with any ``[models.*]`` declaration block already projected into them, so a
    consumer reads one delta whichever form the files used and whether or not
    the configuration is derived. ``models`` keeps the blocks AS WRITTEN (a
    derived file's block replacing its base's block for that family), for
    reports that want to say which decisions were declared rather than
    accreted. ``base`` is the name this file derives from, or ``None``.
    ``lineage`` is the record a run writes into its output.

    Use :func:`stance_config` for the resolved ``(params, flags)`` pair, or
    :func:`load_configuration` for that pair plus the lineage.
    """

    name: str
    path: Path
    params: MappingProxyType
    flags: MappingProxyType
    models: MappingProxyType
    base: str
    lineage: ConfigurationLineage


def available_stances():
    """Return the names of every committed stance file, sorted."""
    if not STANCE_DIR.is_dir():
        return ()
    return tuple(sorted(p.stem for p in STANCE_DIR.glob(f"*{SUFFIX}")))


def load_stance(name):
    """Return the :class:`Stance` named ``name``, a committed stance file.

    Raises ``ValueError`` on an unknown name (listing what is available), on an
    unknown table or top-level key, on a key its namespace does not own, on a
    key named in ``[none_valued]`` that also carries a value, and on anything a
    ``[models.<family>]`` declaration block gets wrong -- an unknown family, a
    key the family does not own, an incomplete membership, a block for a family
    the file does not select, two blocks claiming one key, or a member also
    stated flat. The block refusals carry the solver's own message, prefixed
    with this file's name.

    A DERIVED file additionally raises on an unknown ``base``, on a chain
    deeper than :data:`MAX_CHAIN_FILES`, on a cycle, and on a flat delta that
    restates its base's resolved value.
    """
    path = STANCE_DIR / f"{name}{SUFFIX}"
    if not path.is_file():
        known = ", ".join(available_stances()) or "(none committed)"
        raise ValueError(
            f"unknown stance {name!r}: no {path.name} in {STANCE_DIR}. "
            f"Available stances: {known}"
        )
    return _load(name, path, ())


def load_configuration(spec):
    """Return ``(params, flags, lineage)`` for a named or filed configuration.

    ``spec`` is either a committed stance NAME (``"g1atrim"``) or a PATH to a
    configuration file anywhere on disk (``scripts/stances/examples/x.toml``);
    a value carrying a path separator or the ``.toml`` suffix is read as a
    path. A derived file's ``base`` resolves in ``scripts/stances/`` whichever
    form was used, because a base is a COMMITTED configuration by definition.

    The returned pair is fully resolved -- ``default_config()`` with the base
    chain and this file's deltas applied -- and the lineage's ``identity`` is
    that configuration's. A driver that then layers its own mesh package
    restates the identity with ``lineage.with_identity(params, flags)``.
    """
    stance = load_named_configuration(spec)
    params, flags = default_config()
    params.update(stance.params)
    flags.update(stance.flags)
    return params, flags, stance.lineage


def load_named_configuration(spec):
    """Return the :class:`Stance` for a committed NAME or a file PATH.

    The name-or-path half of :func:`load_configuration`, for a caller that
    needs the file's cumulative DELTA rather than the resolved pair -- a driver
    that layers a named configuration over its own rung reads
    ``.params``/``.flags`` here and applies them where it applies a stance. A
    value carrying a path separator or the ``.toml`` suffix is read as a path;
    anything else is a committed configuration name. A derived file's ``base``
    resolves in ``scripts/stances/`` either way, because a base is a COMMITTED
    configuration by definition.
    """
    text = str(spec)
    if "/" in text or "\\" in text or text.endswith(SUFFIX):
        path = Path(text)
        if not path.is_file():
            raise ValueError(
                f"no configuration file at {path}. Name a committed "
                f"configuration ({', '.join(available_stances()) or 'none'}) "
                f"or the path of a configuration file; the LAPD reference "
                f"configuration is {STANCE_DIR / ('g1atrim' + SUFFIX)}"
            )
        return _load(path.stem, path, ())
    return load_stance(text)


def stance_config(name):
    """Return the resolved ``(params, flags)`` for a configuration.

    ``default_config()`` with the named configuration applied on top: the full
    configuration the file names, ready to hand to ``LAPDSim1D``.
    """
    stance = load_stance(name)
    params, flags = default_config()
    params.update(stance.params)
    flags.update(stance.flags)
    return params, flags


# --- the loader proper -------------------------------------------------
# ``_load`` is recursive over the base chain and carries ``chain`` -- the
# resolved paths already open, outermost first -- so a cycle and an over-deep
# chain are both refusals about the SAME structure rather than two unrelated
# guards, and neither can be reached by loading one more file.


def _load(name, path, chain):
    # RESOLVED on both sides. A chain reaches this function by name through
    # STANCE_DIR and by path through load_named_configuration, and a cycle
    # between the two spellings is still a cycle: comparing unresolved paths
    # would let one spelling of a file stand in for another.
    path = Path(path).resolve()
    document = _read_document(name, path)
    base_name = document.get("base")
    allow_restated = _read_allow_restated(name, path, document)

    if path in chain:
        cycle = " -> ".join(p.stem for p in (*chain, path))
        raise ValueError(
            f"configuration {name!r} ({path}) is its own base: {cycle}. A "
            "base chain is a chain, not a loop"
        )
    if len(chain) + 1 > MAX_CHAIN_FILES:
        deep = " -> ".join(p.stem for p in (*chain, path))
        raise ValueError(
            f"configuration base chain is {len(chain) + 1} files deep, and "
            f"{MAX_CHAIN_FILES} is the limit: {deep}. Past that depth a value "
            "cannot be traced to the file that chose it without running this "
            "loader; flatten the chain by writing the deltas into one derived "
            "file"
        )

    base = None
    if base_name is not None:
        if not isinstance(base_name, str):
            raise ValueError(
                f"configuration {name!r} ({path}) declares base = "
                f"{base_name!r}; base is the NAME of a committed "
                f"configuration file in {STANCE_DIR}, without path or suffix"
            )
        base_path = STANCE_DIR / f"{base_name}{SUFFIX}"
        if not base_path.is_file():
            known = ", ".join(available_stances()) or "(none committed)"
            raise ValueError(
                f"configuration {name!r} ({path}) declares base "
                f"{base_name!r}, and there is no {base_path.name} in "
                f"{STANCE_DIR}. Available: {known}"
            )
        base = _load(base_name, base_path, (*chain, path))
    elif allow_restated:
        raise ValueError(
            f"configuration {name!r} ({path}) sets allow_restated but "
            "declares no base. allow_restated waives the restated-delta "
            "refusal, which only a DERIVED configuration can trip"
        )

    own_params, own_flags, models = _read_deltas(name, path, document)

    if base is not None:
        _refuse_restated(
            name, path, base, own_params, own_flags, models, allow_restated
        )
        params = dict(base.params)
        flags = dict(base.flags)
        blocks = dict(base.models)
        params.update(own_params)
        flags.update(own_flags)
        blocks.update(models)
        base_chain = (base.name, *base.lineage.base_chain)
        file_sha256 = (_sha256(path), *base.lineage.file_sha256)
    else:
        params, flags, blocks = own_params, own_flags, models
        base_chain = ()
        file_sha256 = (_sha256(path),)

    resolved_params, resolved_flags = default_config()
    resolved_params.update(params)
    resolved_flags.update(flags)

    return Stance(
        name=name,
        path=path,
        params=MappingProxyType(params),
        flags=MappingProxyType(flags),
        models=MappingProxyType({k: dict(v) for k, v in blocks.items()}),
        base=base_name,
        lineage=ConfigurationLineage(
            name=name,
            base_chain=base_chain,
            file_sha256=file_sha256,
            delta_keys=tuple(sorted({*own_params, *own_flags})),
            identity=config_identity(resolved_params, resolved_flags),
        ),
    )


def _read_document(name, path):
    """Read the TOML and refuse any table or top-level key this form lacks."""
    with path.open("rb") as handle:
        document = tomllib.load(handle)
    unknown = sorted(set(document) - set(TABLES) - set(SCALARS))
    if unknown:
        raise ValueError(
            f"configuration {name!r} ({path}) has unknown table(s) or key(s) "
            f"{', '.join(unknown)}; a configuration file carries "
            f"{', '.join(TABLES)} and {', '.join(SCALARS)}"
        )
    return document


def _read_allow_restated(name, path, document):
    value = document.get("allow_restated", False)
    if not isinstance(value, bool):
        raise ValueError(
            f"configuration {name!r} ({path}) sets allow_restated = "
            f"{value!r}; it is a boolean waiver, not a value"
        )
    return value


def _read_deltas(name, path, document):
    """Return this file's OWN ``(params, flags, models)``, fully validated."""
    template = dict(zip(NAMESPACES, default_config()))
    resolved = {namespace: dict(document.get(namespace, {}))
                for namespace in NAMESPACES}
    none_valued = document.get("none_valued", {})
    unknown_none = sorted(set(none_valued) - set(NAMESPACES))
    if unknown_none:
        raise ValueError(
            f"configuration {name!r} ({path}) [none_valued] names unknown "
            f"namespace(s) {', '.join(unknown_none)}; it carries "
            f"{' and '.join(NAMESPACES)} arrays of key names"
        )
    for namespace in NAMESPACES:
        for key in none_valued.get(namespace, ()):
            if key in resolved[namespace]:
                raise ValueError(
                    f"configuration {name!r} ({path}) sets {namespace} key "
                    f"{key!r} both to a value and in [none_valued]; it has "
                    "one value, not two"
                )
            resolved[namespace][key] = None

    for namespace in NAMESPACES:
        for key in sorted(resolved[namespace]):
            if key in template[namespace]:
                continue
            other = next(
                (
                    n for n in NAMESPACES
                    if n != namespace and key in template[n]
                ),
                None,
            )
            owner = (
                f"; {namespace} and {other} are separate namespaces and the "
                f"solver refuses a key filed in the wrong one, so move it to "
                f"[{other}]"
                if other is not None
                else "; no LAPDSim1D configuration template owns it"
            )
            raise ValueError(
                f"configuration {name!r} ({path}) sets unknown {namespace} "
                f"key {key!r}{owner}"
            )

    # DECLARATION BLOCKS, projected into the two deltas above. The solver's
    # own resolver does the work, so a configuration file and a driver cannot
    # disagree about what a block means, and its refusals (unknown family,
    # incomplete membership, a member also stated flat) are raised here --
    # before anything is constructed -- naming this file.
    models = document.get("models", {})
    try:
        block_params, block_flags = resolve_declaration_blocks(
            models, resolved["input_dict"], resolved["input_flags"]
        )
    except ValueError as error:
        raise ValueError(f"configuration {name!r} ({path}): {error}") from None
    resolved["input_dict"].update(block_params)
    resolved["input_flags"].update(block_flags)
    return (
        resolved["input_dict"],
        resolved["input_flags"],
        {key: dict(value) for key, value in models.items()},
    )


def _family_members(family_name):
    """Return this loader's ``(table, key)`` pairs for a declared family.

    A block states its family's COMPLETE membership -- the declaration form
    enforces that before this runs -- so the family's membership IS the set of
    keys the block contributed, and it can be read from the family rather than
    recovered from the projection.
    """
    family = FAMILIES_BY_NAME[family_name]
    return tuple(
        (_SPACE_TABLE[space], key) for space, key in family.members
    )


def _refuse_restated(name, path, base, own_params, own_flags, models,
                     allow_restated):
    """Refuse a delta that repeats the base's resolved value.

    A derived configuration exists to state what MOVES. A line that restates
    the value it inherits reads as a decision and is not one: it survives a
    change to the base without announcing that it has stopped agreeing, and it
    puts a second home under a value that has one. ``allow_restated`` waives
    this for a file that deliberately pins a value against its base drifting.

    THE UNIT OF THE CHECK IS THE DELTA THE FILE WROTE, and a DECLARATION BLOCK
    is one delta, not a handful. A block is an INVENTORY of a family's complete
    membership, written out regardless of value, so an individual member that
    agrees with the base is the form working and is not restatement. What the
    block as a whole must do is still move something: a block whose every
    member equals the base's resolved value re-declares a decision the base
    already made, which is the same fault one flat line commits, and it is
    refused by family name.

    A flat key is checked on its own, as before. A key inside a block is
    checked only through its block -- it cannot be both, because a member also
    stated flat is already refused by the declaration form.
    """
    if allow_restated:
        return
    base_params, base_flags = default_config()
    base_params.update(base.params)
    base_flags.update(base.flags)
    own = {"input_dict": own_params, "input_flags": own_flags}
    resolved_base = {"input_dict": base_params, "input_flags": base_flags}

    def _restates(table, key):
        return key in resolved_base[table] and values_equal(
            own[table][key], resolved_base[table][key]
        )

    restated = []
    in_a_block = set()
    for family_name in sorted(models):
        members = [
            (table, key) for table, key in _family_members(family_name)
            if key in own[table]
        ]
        in_a_block.update(members)
        if members and all(_restates(table, key) for table, key in members):
            restated.append(
                f"[models.{family_name}] (all {len(members)} member(s) equal "
                "the base)"
            )
    for table in ("input_dict", "input_flags"):
        for key in sorted(own[table]):
            if (table, key) in in_a_block:
                continue
            if _restates(table, key):
                restated.append(f"{table}:{key} = {own[table][key]!r}")
    if not restated:
        return
    raise ValueError(
        f"configuration {name!r} ({path}) restates {len(restated)} "
        f"delta(s) its base {base.name!r} already resolves to, and a delta "
        f"must move something: {'; '.join(restated)}. Delete them to inherit "
        "the base's value, change them to state a different one, or set "
        "allow_restated = true at the top of the file to pin them "
        "deliberately against the base changing. (A declaration block is ONE "
        "delta: its members may agree with the base individually -- a block "
        "states a family's complete membership regardless of value -- but the "
        "block must move at least one of them.)"
    )


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
