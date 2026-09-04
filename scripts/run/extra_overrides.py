"""The one reader of ``--extra KEY=VALUE`` overrides, shared by the drivers.

A command-line override arrives as TEXT, and the value a driver files into
``input_dict`` or ``input_flags`` is whatever that text parses to -- so
``cathode_Ts_base_K=1910`` delivers an ``int`` and ``cathode_Ts_base_K=1910.0``
delivers a ``float``. Arithmetic cannot tell the two apart; the CONFIGURATION
IDENTITY can. ``config_identity`` hashes the canonical JSON text of the
resolved configuration, and ``1910`` and ``1910.0`` are different bytes there,
so one physical configuration reached by the two spellings carries two
identities -- and an exact-equality consent guard reading the same value
accepts a spelling the identity will not recognise as the one it consented to.

The ambiguity is removed HERE, at the parse layer, because this is the one
place both of those readers sit above: every override is coerced to the type
the owning template key carries in :func:`default_config`, so the consent
guards above and the solver below see one type per key however the number was
typed. Values that cannot be read as the owning key's type, and keys that
neither template owns, raise ``ValueError`` at this layer, before anything is
constructed.

``input_dict`` and ``input_flags`` share no key, so the template that OWNS a
key is found by looking it up in both. Which namespace a value is FILED into
is still the switch's -- ``--extra`` for params, ``--extra-flag`` for flags --
and a key placed in the wrong one is refused by ``LAPDSim1D`` at construction
exactly as before: that refusal is about routing, and this one is about type.
"""

import json

from cablp.solvers._sim1d import default_config

#: Template value types this layer can name an expected type for. A key whose
#: template value is ``None`` or a list states no scalar type -- there is
#: nothing in ``default_config()`` to normalise such a value TO -- and its
#: parsed value is passed through unchanged for the solver's own validators
#: to judge.
_TYPE_NAMES = {bool: "bool", int: "int", float: "float", str: "str"}


def _read(raw):
    """Return ``raw`` as JSON where it is legible as JSON, else as itself.

    The bare tokens the config templates hold as strings -- ``He``,
    ``ads_des``, ``tr_bdf2`` -- are not JSON, and a caller types them without
    quotes; a JSON failure means the token IS the value.
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _coerce(key, raw, value, template, switch):
    """Return ``value`` as the type ``template`` carries, or raise.

    ``template`` is the key's value in :func:`default_config`, which is what
    states the type. ``raw`` is the token as typed, used for a string key --
    where the token itself is the value -- and in refusals.
    """
    kind = type(template)
    if kind not in _TYPE_NAMES:
        return value
    expected = _TYPE_NAMES[kind]
    if kind is bool:
        if value is True or value is False:
            return value
    elif kind is int:
        if type(value) is int:
            return value
        if type(value) is float and value.is_integer():
            return int(value)
    elif kind is float:
        if type(value) is float:
            return value
        if type(value) is int:
            return float(value)
    else:
        # A string key: every token is a legible value of its type, so the
        # token as typed is the value. JSON quoting is honoured where it was
        # used, so both ``k=ads_des`` and ``k="ads_des"`` give ``ads_des``.
        return value if isinstance(value, str) else raw
    raise ValueError(
        f"{switch} {key}={raw}: {key} carries {expected} -- the configuration "
        f"template gives it {template!r} -- and {value!r} cannot be read as "
        f"one. Spell the value as {expected}."
    )


def parse_extra_overrides(items, switch="--extra"):
    """Return ``{key: value}`` from ``KEY=VALUE`` strings, typed by template.

    ``items`` is the switch's raw argv list and ``switch`` names it in refusal
    messages. A key given more than once takes its last value, as a later
    layer overrides an earlier one.

    Each value is read as JSON where it is legible as JSON, kept as the bare
    token otherwise, and then coerced to the type its key's template value
    carries:

    ``float`` key
        an ``int`` or a ``float`` becomes ``float``.
    ``int`` key
        an ``int``, or a ``float`` with no fractional part, becomes ``int``.
    ``bool`` key
        ``true`` or ``false`` only.
    ``str`` key
        the token as typed, JSON quoting honoured where it was used.

    A key whose template value is ``None`` or a list names no scalar type and
    its parsed value passes through unchanged.

    Raises ``ValueError`` -- naming the key, the given value and the expected
    type -- when a value cannot be read as its key's type; when a key is owned
    by neither configuration template; and when an item is not ``KEY=VALUE``.
    """
    params_template, flags_template = default_config()
    out = {}
    for item in items or ():
        key, sep, raw = item.partition("=")
        if not sep or not key:
            raise ValueError(
                f"{switch} {item!r} is not of the form key=value."
            )
        if key in params_template:
            template = params_template[key]
        elif key in flags_template:
            template = flags_template[key]
        else:
            raise ValueError(
                f"{switch} {key}={raw}: {key} is owned by NEITHER "
                "configuration template -- it is in neither input_dict nor "
                "input_flags, so it would be a silent/inert control. Check "
                "which template owns the key in "
                "cablp/solvers/_sim1d/core/config.py and spell it exactly."
            )
        out[key] = _coerce(key, raw, _read(raw), template, switch)
    return out
