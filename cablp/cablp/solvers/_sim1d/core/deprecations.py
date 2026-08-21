"""Declarative deprecation register for the LAPDSim1D configuration surface.

This module holds ONE table and ONE emitter. A row says "using this key at a
value other than its canonical default is deprecated"; the emitter compares a
resolved config against the authoritative defaults in :mod:`.config` and raises
a single ``DeprecationWarning`` per offending key at solver construction.

**The lifecycle is warn -> drop from tests -> delete, and this module is the
first step only.** Nothing here refuses a run, changes a default, or alters a
computed value: every deprecated path stays runnable so old results remain
reproducible and A/B arms stay available. Deletion is a separate, focused pass
and replaces the path with a ``ValueError``, never a silent fallback.

**Canonical default construction is warning-free by CONSTRUCTION.** A row does
not carry its own copy of the default -- the default is read from
``input_dict_template_1d`` / ``input_flags_template_1d`` at emit time -- so a
row can never disagree with the shipped default and can never fire on
``default_config()``. The smoke suite asserts that property directly.

Two classes of row exist:

* **whole-key** (``values=None``) -- any value other than the canonical default
  is deprecated. Used for dead levers: controls no committed configuration
  sets, which are scheduled for removal.
* **value-scoped** (``values=(...)``) -- only the listed values are deprecated
  and the rest of the key stays fully supported. Used where a selector keeps
  live arms, e.g. ``neutral_model``: ``"kinetic"`` deprecates while
  ``"kinetic_dvm"`` does not.

Controls that already have their own louder guard are deliberately ABSENT.
``front_flux_model``, ``D_amb_model``, ``cathode_model`` and ``D_amb`` are
frozen by ``LAPDSim1D._validate_r1_configuration_presence`` and raise on
non-default use; the four resolved-boundary surface-loss controls
(``source_surface_loss``, ``end_surface_loss``, ``source_surface_area_scale``,
``end_surface_area_scale``), ``gas_puff_mode``'s retired waveform modes and the
legacy ion-neutral path (``ion_neutral_moment_closure=False``) already warn from
that same method. Adding a second warning for the same condition would only
duplicate it.

Closure families with a live A/B are APPARATUS, not legacy, and are absent for
that reason: the ion-neutral drag closures, the beam-deposition family, the
excitation models and ``atomic_rate_model="janev"`` all stay usable and
un-warned because the campaign quotes brackets and a bracket needs both arms.
"""

import warnings
from dataclasses import dataclass

import numpy as np

from .config import input_dict_template_1d, input_flags_template_1d

PARAMS = "params"
FLAGS = "flags"


@dataclass(frozen=True)
class DeprecatedControl:
    """One row of the register.

    Attributes
    ----------
    namespace:
        ``PARAMS`` if the key lives in ``input_dict``, ``FLAGS`` if it lives in
        ``input_flags``. The two are separate namespaces and a key filed under
        the wrong one would never be read, so this is checked at import.
    note:
        Why the control is deprecated. Appended to the warning verbatim; write
        it as a sentence fragment that follows "is DEPRECATED: ".
    replacement:
        What to do instead. Appended after the note.
    values:
        ``None`` for a whole-key row (any non-default value warns), or a tuple
        of the specific values that deprecate for a value-scoped row.
    """

    namespace: str
    note: str
    replacement: str
    values: tuple = None


# --- shared notes ---------------------------------------------------------
# Kept as module constants so a group of rows cannot drift apart in wording.

_DEAD_LEVER = (
    "it is a dead lever -- no committed configuration sets it (config surface "
    "audited 2026-08-19), so it is scheduled for removal"
)
_DEAD_LEVER_FIX = "drop the override and leave the key at its default"

_FREED_BRANCH = (
    "the golden fixture no longer pins it (re-anchored at the production "
    "stance, 2026-08-20), so the branch it selects is no longer exercised by "
    "any committed configuration and is scheduled for removal"
)
_FREED_BRANCH_FIX = (
    "drop the override; the path stays runnable meanwhile for A/B arms and "
    "for reproducing old results at their anchor tag"
)

_PUFF_WAVEFORM = (
    "it is read only by the retired gas_puff_mode waveforms "
    "('decay_after_breakdown', 'pulse_decay_to_level', 'double_erf'), which "
    "the measured square waveform superseded"
)
_PUFF_WAVEFORM_FIX = (
    "drop the override; the retired waveform modes stay runnable until the "
    "waveform-comparison figures freeze"
)


# --- the register ---------------------------------------------------------
# Ordered by group so the table reads as the audit that produced it.

DEPRECATED_CONTROLS = {
    # ==== branches freed by the 2026-08-20 golden re-anchor ================
    # Each of these was a live branch only because the retired fixture pinned
    # it away from the production value. With the fixture re-anchored at the
    # stance, nothing committed selects them any more.
    "active_plasma_topology": DeprecatedControl(
        FLAGS, _FREED_BRANCH, _FREED_BRANCH_FIX,
    ),
    "raw_stage_validation": DeprecatedControl(
        FLAGS, _FREED_BRANCH, _FREED_BRANCH_FIX,
    ),
    "hyperbolic_energy_consistent": DeprecatedControl(
        FLAGS, _FREED_BRANCH, _FREED_BRANCH_FIX,
    ),
    "characteristic_boundary": DeprecatedControl(
        FLAGS, _FREED_BRANCH, _FREED_BRANCH_FIX,
    ),
    "ion_neutral_thermalization": DeprecatedControl(
        FLAGS, _FREED_BRANCH, _FREED_BRANCH_FIX,
    ),
    "beam_anode_interception": DeprecatedControl(
        FLAGS, _FREED_BRANCH, _FREED_BRANCH_FIX,
    ),
    "front_flux": DeprecatedControl(
        FLAGS, _FREED_BRANCH, _FREED_BRANCH_FIX,
    ),
    "hyperbolic_wave_speed": DeprecatedControl(
        PARAMS, _FREED_BRANCH, _FREED_BRANCH_FIX, values=("isothermal",),
    ),
    "ionization_birth_energy_model": DeprecatedControl(
        PARAMS, _FREED_BRANCH, _FREED_BRANCH_FIX, values=("legacy",),
    ),
    "sigma_in_model": DeprecatedControl(
        PARAMS,
        "the pre-Phelps presheath cross-section arms are superseded by "
        "sigma_in_model='phelps', and " + _FREED_BRANCH,
        _FREED_BRANCH_FIX,
        values=("constant", "cx_derived"),
    ),
    "b_ion_neutral_drag": DeprecatedControl(
        PARAMS,
        "it scales the legacy ion-neutral drag term, which the moment-closed "
        "Phelps operator replaced, and " + _FREED_BRANCH,
        _FREED_BRANCH_FIX,
    ),
    "b_ion_neutral_thermalization": DeprecatedControl(
        PARAMS,
        "it scales the legacy ion-neutral thermalization term, which the "
        "moment-closed Phelps operator replaced, and " + _FREED_BRANCH,
        _FREED_BRANCH_FIX,
    ),
    "gas_puff_profile": DeprecatedControl(
        PARAMS,
        "the historical single-cell puff deposits the whole fuelling rate in "
        "one cell, which the resolved cosine pipe profile superseded, and "
        + _FREED_BRANCH,
        _FREED_BRANCH_FIX,
        values=("cell",),
    ),
    # ==== the kinetic neutral engine (selector-scoped) =====================
    # Scoped to the SELECTOR and its solver-side relaxation coupling. The
    # kinetic_neutrals module itself is NOT deprecated: it is the shared
    # library behind the kn2zone instrument and behind neutral_model=
    # "kinetic_dvm", which stays fully supported.
    "neutral_model": DeprecatedControl(
        PARAMS,
        "the relaxation-coupled kinetic neutral engine is superseded as an "
        "instrument by neutral_model='kinetic_dvm' and is quoted in no live "
        "claim",
        "select 'moment' (production) or 'kinetic_dvm' (the kinetic "
        "instrument); the module behind this selector stays as the shared "
        "kinetic library and is not affected",
        values=("kinetic",),
    ),
    # ==== dead levers: cathode ============================================
    "cathode_rad_area_cm2": DeprecatedControl(PARAMS, _DEAD_LEVER, _DEAD_LEVER_FIX),
    "cathode_ads_rate_per_s": DeprecatedControl(PARAMS, _DEAD_LEVER, _DEAD_LEVER_FIX),
    "cathode_desorption_prefactor_per_s": DeprecatedControl(
        PARAMS, _DEAD_LEVER, _DEAD_LEVER_FIX,
    ),
    "cathode_desorption_energy_eV": DeprecatedControl(
        PARAMS, _DEAD_LEVER, _DEAD_LEVER_FIX,
    ),
    "cathode_env_T_K": DeprecatedControl(PARAMS, _DEAD_LEVER, _DEAD_LEVER_FIX),
    "anode_radius_cm": DeprecatedControl(PARAMS, _DEAD_LEVER, _DEAD_LEVER_FIX),
    "coverage_backfill_time_s": DeprecatedControl(PARAMS, _DEAD_LEVER, _DEAD_LEVER_FIX),
    # ==== dead levers: timestep control ===================================
    "heat_dt_fraction": DeprecatedControl(PARAMS, _DEAD_LEVER, _DEAD_LEVER_FIX),
    "drag_dt_fraction": DeprecatedControl(PARAMS, _DEAD_LEVER, _DEAD_LEVER_FIX),
    "neutral_dt_fraction": DeprecatedControl(PARAMS, _DEAD_LEVER, _DEAD_LEVER_FIX),
    "dt_reject_factor": DeprecatedControl(PARAMS, _DEAD_LEVER, _DEAD_LEVER_FIX),
    "dt_growth_recovery_factor": DeprecatedControl(PARAMS, _DEAD_LEVER, _DEAD_LEVER_FIX),
    "surface_loss_floor_exempt_rtol": DeprecatedControl(
        PARAMS, _DEAD_LEVER, _DEAD_LEVER_FIX,
    ),
    "beam_ionization_birth_timestep_bound": DeprecatedControl(
        FLAGS, _DEAD_LEVER, _DEAD_LEVER_FIX,
    ),
    # ==== dead levers: the retired puff-waveform family ====================
    "Twin_S_gp_decay_target": DeprecatedControl(
        PARAMS, _PUFF_WAVEFORM, _PUFF_WAVEFORM_FIX,
    ),
    "tau_gp_after_breakdown": DeprecatedControl(
        PARAMS, _PUFF_WAVEFORM, _PUFF_WAVEFORM_FIX,
    ),
    "tau_gp_decay_factor": DeprecatedControl(
        PARAMS, _PUFF_WAVEFORM, _PUFF_WAVEFORM_FIX,
    ),
    "tau_gp_rise_center": DeprecatedControl(
        PARAMS, _PUFF_WAVEFORM, _PUFF_WAVEFORM_FIX,
    ),
    "tau_gp_rise_width": DeprecatedControl(
        PARAMS, _PUFF_WAVEFORM, _PUFF_WAVEFORM_FIX,
    ),
    "tau_gp_drop_center": DeprecatedControl(
        PARAMS, _PUFF_WAVEFORM, _PUFF_WAVEFORM_FIX,
    ),
    "tau_gp_drop_width": DeprecatedControl(
        PARAMS, _PUFF_WAVEFORM, _PUFF_WAVEFORM_FIX,
    ),
    # ==== dead levers: inert selectors ====================================
    "end_mode": DeprecatedControl(
        PARAMS,
        "the 'mirrored_source' end boundary is selected by no committed "
        "configuration (config surface audited 2026-08-19) and is scheduled "
        "for removal",
        _DEAD_LEVER_FIX,
        values=("mirrored_source",),
    ),
    "Ti_birth_ionization": DeprecatedControl(PARAMS, _DEAD_LEVER, _DEAD_LEVER_FIX),
    # ==== dead levers: scaling factors ====================================
    "b_anode_collection": DeprecatedControl(PARAMS, _DEAD_LEVER, _DEAD_LEVER_FIX),
    "b_anode_advective_block": DeprecatedControl(PARAMS, _DEAD_LEVER, _DEAD_LEVER_FIX),
    "alpha_front": DeprecatedControl(PARAMS, _DEAD_LEVER, _DEAD_LEVER_FIX),
    "b_rec_rad": DeprecatedControl(PARAMS, _DEAD_LEVER, _DEAD_LEVER_FIX),
    "b_rec_3b": DeprecatedControl(PARAMS, _DEAD_LEVER, _DEAD_LEVER_FIX),
    "b_Qie": DeprecatedControl(PARAMS, _DEAD_LEVER, _DEAD_LEVER_FIX),
    "b_Qei_Te_exp": DeprecatedControl(PARAMS, _DEAD_LEVER, _DEAD_LEVER_FIX),
    "b_Qen_Te_exp": DeprecatedControl(PARAMS, _DEAD_LEVER, _DEAD_LEVER_FIX),
    "b_Q_Te_ref_eV": DeprecatedControl(PARAMS, _DEAD_LEVER, _DEAD_LEVER_FIX),
    "b_epara": DeprecatedControl(PARAMS, _DEAD_LEVER, _DEAD_LEVER_FIX),
    "b_ipara": DeprecatedControl(PARAMS, _DEAD_LEVER, _DEAD_LEVER_FIX),
    "b_slip_entrainment": DeprecatedControl(PARAMS, _DEAD_LEVER, _DEAD_LEVER_FIX),
    "ln_lambda_min": DeprecatedControl(PARAMS, _DEAD_LEVER, _DEAD_LEVER_FIX),
    # ==== dead levers: neutral probe source ===============================
    "neutral_probe_profile": DeprecatedControl(PARAMS, _DEAD_LEVER, _DEAD_LEVER_FIX),
    "neutral_probe_waveform_table": DeprecatedControl(
        PARAMS, _DEAD_LEVER, _DEAD_LEVER_FIX,
    ),
    "neutral_probe_zone": DeprecatedControl(PARAMS, _DEAD_LEVER, _DEAD_LEVER_FIX),
    # ==== dead levers: the regime-tracer passivity constants ===============
    # The half of the tracer_* family nothing configures; the activation and
    # overlap keys ARE exercised and are absent from this table.
    "tracer_passivity_current_ratio": DeprecatedControl(
        PARAMS, _DEAD_LEVER, _DEAD_LEVER_FIX,
    ),
    "tracer_passivity_depletion": DeprecatedControl(
        PARAMS, _DEAD_LEVER, _DEAD_LEVER_FIX,
    ),
    "tracer_passivity_hysteresis": DeprecatedControl(
        PARAMS, _DEAD_LEVER, _DEAD_LEVER_FIX,
    ),
    "tracer_passivity_thinness": DeprecatedControl(
        PARAMS, _DEAD_LEVER, _DEAD_LEVER_FIX,
    ),
    "tracer_refresh_tol": DeprecatedControl(PARAMS, _DEAD_LEVER, _DEAD_LEVER_FIX),
    # ==== dead levers: miscellaneous ======================================
    "plasma_area_max_vessel_fraction": DeprecatedControl(
        PARAMS, _DEAD_LEVER, _DEAD_LEVER_FIX,
    ),
    "vessel_leak_resistance_ohm": DeprecatedControl(
        PARAMS, _DEAD_LEVER, _DEAD_LEVER_FIX,
    ),
    "beam_excitation_energy_eV": DeprecatedControl(
        PARAMS, _DEAD_LEVER, _DEAD_LEVER_FIX,
    ),
    "icool": DeprecatedControl(FLAGS, _DEAD_LEVER, _DEAD_LEVER_FIX),
    "ncool": DeprecatedControl(FLAGS, _DEAD_LEVER, _DEAD_LEVER_FIX),
}


def _validate_register():
    """Reject a row filed under the wrong namespace or naming no real key.

    ``input_dict`` and ``input_flags`` are separate namespaces and neither
    reads the other's keys, so a mis-filed row would be a warning that can
    never fire. Checked at import so the mistake cannot ship.
    """
    misfiled = []
    for name, row in DEPRECATED_CONTROLS.items():
        if row.namespace == PARAMS:
            owned = name in input_dict_template_1d
        elif row.namespace == FLAGS:
            owned = name in input_flags_template_1d
        else:
            raise ValueError(
                f"deprecation register row {name!r} has an unknown namespace "
                f"{row.namespace!r}; expected {PARAMS!r} or {FLAGS!r}"
            )
        if not owned:
            misfiled.append(f"{name} (filed as {row.namespace})")
    if misfiled:
        raise ValueError(
            "deprecation register rows name keys their declared namespace "
            "does not own (a warning that can never fire): "
            + ", ".join(sorted(misfiled))
        )


_validate_register()


def _differs(actual, default):
    """Return True when ``actual`` is not the canonical ``default`` value.

    Tolerant of the array-valued and ``None``-valued defaults on the config
    surface, where a bare ``!=`` is either ambiguous or raises.
    """
    if actual is default:
        return False
    if actual is None or default is None:
        return True
    try:
        return not bool(np.array_equal(np.asarray(actual), np.asarray(default)))
    except (TypeError, ValueError):
        try:
            return bool(actual != default)
        except (TypeError, ValueError):
            return True


def _brief(value, limit=60):
    """Return a repr short enough to sit inside a warning message."""
    text = repr(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def deprecation_messages(input_dict, input_flags):
    """Return one message per deprecated control the config actually uses.

    Pure: builds the message list without emitting anything, so tests and
    reports can read the register's verdict directly.

    Parameters
    ----------
    input_dict, input_flags:
        Resolved configuration mappings (every registered key present).

    Returns
    -------
    list[str]
        One message per offending key, in register order. Empty for the
        canonical defaults.
    """
    messages = []
    for name, row in DEPRECATED_CONTROLS.items():
        if row.namespace == PARAMS:
            supplied, template = input_dict, input_dict_template_1d
        else:
            supplied, template = input_flags, input_flags_template_1d
        if name not in supplied:
            continue
        actual = supplied[name]
        default = template[name]
        if row.values is None:
            if not _differs(actual, default):
                continue
        else:
            if not any(not _differs(actual, value) for value in row.values):
                continue
        messages.append(
            f"{name}={_brief(actual)} is DEPRECATED: {row.note}. "
            f"{row.replacement[:1].upper()}{row.replacement[1:]} "
            f"({name}={_brief(default)})."
        )
    return messages


def warn_deprecated_config(input_dict, input_flags, stacklevel=2):
    """Emit one ``DeprecationWarning`` per deprecated control in use.

    Called once at ``LAPDSim1D`` construction. Never raises, never changes a
    value: a deprecated path stays runnable and bit-identical.
    """
    for message in deprecation_messages(input_dict, input_flags):
        warnings.warn(message, DeprecationWarning, stacklevel=stacklevel + 1)
