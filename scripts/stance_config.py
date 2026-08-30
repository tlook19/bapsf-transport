"""Load a committed stance file into an ``input_dict`` / ``input_flags`` pair.

A STANCE is a named, complete, committed configuration package: the whole delta
from :func:`default_config`, in one file, with nothing accreted on the command
line. The files live in ``scripts/stances/<name>.toml`` and are the reference
every consumer reads -- the campaign driver (``run_m6_point.py --stance``), the
config-diff pre-flight (``preflight_diffcfg.py --stance``), and the scorer's
shared production values (``compare_sim1d_es1.PARAM_OVERRIDES``).

File format, all tables optional except ``[input_dict]`` and ``[input_flags]``:

``[input_dict]``
    ``input_dict`` keys and their values.
``[input_flags]``
    ``input_flags`` keys and their values.
``[none_valued]``
    ``input_dict`` / ``input_flags`` arrays of key NAMES whose stance value is
    ``None``. TOML has no null literal, so a key that must be explicitly unset
    is named here instead of appearing in the table above.
``[models.<family>]``
    A DECLARATION BLOCK: one model family's COMPLETE membership, stated
    without namespaces, every member written regardless of value. See
    ``cablp/solvers/_sim1d/core/model_declarations.py`` for the form and its
    refusals. A block is PROJECTED onto the two tables above at load time, so
    ``Stance.params`` / ``Stance.flags`` carry the members like any other
    stance delta and every consumer of this loader reads them unchanged; the
    block as written is kept on ``Stance.models`` for reporting. A member may
    not also appear in ``[input_dict]`` / ``[input_flags]``.

Every key is checked against the config templates at load time, in the
namespace that owns it: the two namespaces are separate and neither accepts the
other's keys, so a misfiled key raises here, naming the namespace that does own
it, rather than reaching the solver's own construction-time refusal. A block's
members are exempt from that check by construction -- the family membership
carries the namespace, which is why a block cannot misfile a key at all.

Values are TOML-native (float, integer, string, boolean, array); arrays reach
the solver as lists, which is what the config keys that take per-cell profiles
expect.
"""

import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS.parent))

from cablp.solvers._sim1d import (  # noqa: E402
    default_config,
    resolve_declaration_blocks,
)

#: Directory holding the committed stance files.
STANCE_DIR = _SCRIPTS / "stances"
SUFFIX = ".toml"
NAMESPACES = ("input_dict", "input_flags")
TABLES = NAMESPACES + ("none_valued", "models")


@dataclass(frozen=True)
class Stance:
    """One loaded stance: its name, its file, and its two override mappings.

    ``params`` and ``flags`` are read-only mappings of exactly what the file
    declares -- the delta, not a full config -- with any ``[models.*]``
    declaration block already projected into them, so a consumer reads one
    delta whichever form the file used. ``models`` keeps the blocks AS WRITTEN,
    for reports that want to say which decisions were declared rather than
    accreted. Use :func:`stance_config` for the resolved ``(params, flags)``
    pair.
    """

    name: str
    path: Path
    params: MappingProxyType
    flags: MappingProxyType
    models: MappingProxyType


def available_stances():
    """Return the names of every committed stance file, sorted."""
    if not STANCE_DIR.is_dir():
        return ()
    return tuple(sorted(p.stem for p in STANCE_DIR.glob(f"*{SUFFIX}")))


def load_stance(name):
    """Return the :class:`Stance` named ``name``.

    Raises ``ValueError`` on an unknown name (listing what is available), on an
    unknown table, on a key its namespace does not own, on a key named in
    ``[none_valued]`` that also carries a value, and on anything a
    ``[models.<family>]`` declaration block gets wrong -- an unknown family, a
    key the family does not own, an incomplete membership, a block for a family
    the file does not select, two blocks claiming one key, or a member also
    stated flat. The block refusals carry the solver's own message, prefixed
    with this file's name.
    """
    path = STANCE_DIR / f"{name}{SUFFIX}"
    if not path.is_file():
        known = ", ".join(available_stances()) or "(none committed)"
        raise ValueError(
            f"unknown stance {name!r}: no {path.name} in {STANCE_DIR}. "
            f"Available stances: {known}"
        )
    with path.open("rb") as handle:
        document = tomllib.load(handle)

    unknown_tables = sorted(set(document) - set(TABLES))
    if unknown_tables:
        raise ValueError(
            f"stance {name!r} ({path}) has unknown table(s) "
            f"{', '.join(unknown_tables)}; a stance file carries only "
            f"{', '.join(TABLES)}"
        )

    template = dict(zip(NAMESPACES, default_config()))
    resolved = {namespace: dict(document.get(namespace, {}))
                for namespace in NAMESPACES}
    none_valued = document.get("none_valued", {})
    unknown_none = sorted(set(none_valued) - set(NAMESPACES))
    if unknown_none:
        raise ValueError(
            f"stance {name!r} ({path}) [none_valued] names unknown "
            f"namespace(s) {', '.join(unknown_none)}; it carries "
            f"{' and '.join(NAMESPACES)} arrays of key names"
        )
    for namespace in NAMESPACES:
        for key in none_valued.get(namespace, ()):
            if key in resolved[namespace]:
                raise ValueError(
                    f"stance {name!r} ({path}) sets {namespace} key {key!r} "
                    "both to a value and in [none_valued]; it has one stance "
                    "value, not two"
                )
            resolved[namespace][key] = None

    for namespace in NAMESPACES:
        for key in sorted(resolved[namespace]):
            if key in template[namespace]:
                continue
            other = next(
                (n for n in NAMESPACES if n != namespace and key in template[n]),
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
                f"stance {name!r} ({path}) sets unknown {namespace} key "
                f"{key!r}{owner}"
            )

    # DECLARATION BLOCKS, projected into the two deltas above. The solver's
    # own resolver does the work, so a stance file and a driver cannot disagree
    # about what a block means, and its refusals (unknown family, incomplete
    # membership, a member also stated flat) are raised here -- before anything
    # is constructed -- naming this file.
    models = document.get("models", {})
    try:
        block_params, block_flags = resolve_declaration_blocks(
            models, resolved["input_dict"], resolved["input_flags"]
        )
    except ValueError as error:
        raise ValueError(f"stance {name!r} ({path}): {error}") from None
    resolved["input_dict"].update(block_params)
    resolved["input_flags"].update(block_flags)

    return Stance(
        name=name,
        path=path,
        params=MappingProxyType(resolved["input_dict"]),
        flags=MappingProxyType(resolved["input_flags"]),
        models=MappingProxyType({k: dict(v) for k, v in models.items()}),
    )


def stance_config(name):
    """Return the resolved ``(params, flags)`` for a stance.

    ``default_config()`` with the stance applied on top: the full configuration
    the stance names, ready to hand to ``LAPDSim1D``.
    """
    stance = load_stance(name)
    params, flags = default_config()
    params.update(stance.params)
    flags.update(stance.flags)
    return params, flags
