"""The cathode-diagnostics mapping a loaded result carries, and its refusals.

Six circuit scalars were exported to the HDF5 from the sim3 era and are not
exported any more: they were computed under a power closure that does not
close. ``P_cathode_i_pl`` carries the ANODE ion current under a cathode name;
``P_loss`` sums phi-inclusive electron powers against thermal-only ion powers
and the wrong cathode current; ``P_net`` and ``P_net2`` subtract full electrode
powers from a load term, mixing the circuit's field work with the plasma's
thermal book. ``P_comp`` and ``P_anode_i_pl`` were exported and never read.

Each has a successor in the closed audit set the circuit already computes, so
a reader that asks for a retired name is asking for a number that was wrong,
not for a number that is missing. :class:`CathodeDiagnostics` therefore refuses
the read and names the successor rather than raising a bare ``KeyError`` or --
worse -- letting a ``.get`` return a default that reads as a measurement.

Files saved BEFORE the retirement carry the datasets, and there the read
succeeds and returns what that file recorded: an old file is a record of what
the model then computed, and this module does not rewrite records.
"""


class RetiredCathodeDiagnosticError(KeyError):
    """A retired cathode-diagnostic name was read from a file that lacks it.

    A ``KeyError`` subclass so a caller that already guards a diagnostics read
    for absence keeps working, while a caller that guards nothing gets a
    message naming what to read instead.
    """

    def __str__(self):
        # KeyError.__str__ reprs its argument, which would quote the whole
        # message and bury the successor name in escapes.
        return self.args[0] if self.args else ""


#: Retired diagnostic name -> what to read instead. Keyed on the UNPREFIXED
#: circuit field name; the saved datasets carry a ``source_`` / ``end_``
#: prefix, which the lookup strips.
RETIRED_CATHODE_DIAGNOSTICS = {
    "P_cathode_i_pl": (
        "P_cathode_i_thermal -- the cathode's ion plasma-thermal power. "
        "P_cathode_i_pl was built from I_i_a, the ANODE ion current, under a "
        "cathode name"
    ),
    "P_anode_i_pl": (
        "P_anode_i_thermal -- the anode's ion plasma-thermal power"
    ),
    "P_loss": (
        "P_plasma_thermal_loss -- the total plasma-thermal power to the "
        "electrodes. P_loss added phi-inclusive electron powers to "
        "thermal-only ion powers and used the anode ion current at the "
        "cathode"
    ),
    "P_net": (
        "P_into_plasma for the power heating the plasma, or P_load_residual "
        "for the load-power closure check. P_net subtracted full electrode "
        "powers from the load field work, which books the sheath falls twice"
    ),
    "P_net2": (
        "P_into_plasma -- P_prim + P_ohmic minus the plasma-thermal loss, "
        "which is what P_net2 was reaching for"
    ),
    "P_comp": (
        "no successor: the compensating-resistor dissipation is I_tot**2 * "
        "R_comp, both of which are on the file (source_I_tot and the run's "
        "R_comp), and nothing read this row"
    ),
}

#: The dataset names the retirement removed, both prefixes.
RETIRED_CATHODE_DIAGNOSTIC_KEYS = frozenset(
    f"{prefix}_{name}"
    for prefix in ("source", "end")
    for name in RETIRED_CATHODE_DIAGNOSTICS
)


def _successor(name):
    """Return the successor advice for ``name``, or None if not retired."""
    for prefix in ("source_", "end_"):
        if name.startswith(prefix):
            return RETIRED_CATHODE_DIAGNOSTICS.get(name[len(prefix):])
    return RETIRED_CATHODE_DIAGNOSTICS.get(name)


class CathodeDiagnostics(dict):
    """The ``cathode_diagnostics`` mapping, refusing retired names loudly.

    Reads and iteration are a plain dict's. The one difference is a name in
    :data:`RETIRED_CATHODE_DIAGNOSTICS` that the file does NOT carry: that read
    raises :class:`RetiredCathodeDiagnosticError` naming the successor, through
    ``[]`` and through ``get`` alike -- a ``get`` that quietly returned its
    default would put a made-up number where a retired one used to be.

    ``in`` is left alone: the name really is absent, and a caller testing for
    presence deserves the truthful answer.
    """

    __slots__ = ()

    def __missing__(self, key):
        advice = _successor(key)
        if advice is None:
            raise KeyError(key)
        raise RetiredCathodeDiagnosticError(
            f"cathode diagnostic {key!r} was retired from the sim1d export "
            f"(it was computed under the pre-closure power book); read "
            f"{advice}. Files saved before the retirement still carry the "
            f"dataset and read normally -- this one does not."
        )

    def get(self, key, default=None):
        """Like ``dict.get``, except a retired absent name raises.

        The default would otherwise stand in for a diagnostic that was
        withdrawn for being wrong, which is the silent fallback this
        retirement exists to remove.
        """
        if key not in self and _successor(key) is not None:
            self.__missing__(key)
        return super().get(key, default)
