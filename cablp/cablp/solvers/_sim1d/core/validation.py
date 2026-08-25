"""Construction-time configuration validation for :class:`LAPDSim1D`.

Every function here is a REFUSAL: it reads a resolved configuration (and, where
the answer depends on the machine, the built geometry) and either returns the
resolved record the solver arms itself with, or raises a ``ValueError`` naming
what the selector accepts.  Those messages are load-bearing documentation --
for many selectors they are the only in-repo statement of what is accepted --
so they are quoted verbatim from the solver they were extracted from and must
stay byte-identical.

Split out of ``solver.py`` at thread-24 phase R5.  The functions take explicit
inputs rather than a solver, so the refusals are readable, and testable,
without the object they used to hang off.
"""

import math
import warnings
from types import SimpleNamespace

import numpy as np

from .config import (
    coverage_closure_defaults,
    emitting_area_defaults,
    neutral_probe_source_defaults,
)
from .geometry import _anode_neutral_transparency
from ..physics.neutrals import (
    NEUTRAL_PROBE_WAVEFORMS,
    neutral_probe_profile_weights,
    neutral_probe_waveform_table,
)
from ..physics.sources import (
    ANODE_JET_ENERGY_CONVENTIONS,
    CATHODE_JET_ENERGY_CONVENTIONS,
)

#: Implemented operator-splitting compositions.
OPERATOR_SPLITTINGS = ("lie", "strang")


class _RawStageError(ValueError):
    def __init__(self, y, stage, reason, detail):
        super().__init__(f"{stage}: {reason}")
        self.y = np.asarray(y, dtype=float).copy()
        self.stage = str(stage)
        self.reason = str(reason)
        self.detail = dict(detail)


def _bad_array_summary(values, *, mode="nonfinite", max_indices=8):
    values = np.asarray(values, dtype=float)
    if mode == "negative":
        mask = values < 0.0
    else:
        mask = ~np.isfinite(values)
    bad = np.flatnonzero(mask)
    if bad.size == 0:
        return None
    finite = values[np.isfinite(values)]
    return {
        "count": int(bad.size),
        "indices": bad[:max_indices].astype(int).tolist(),
        "values": values[bad[:max_indices]].astype(float).tolist(),
        "nan_count": int(np.count_nonzero(np.isnan(values))),
        "posinf_count": int(np.count_nonzero(np.isposinf(values))),
        "neginf_count": int(np.count_nonzero(np.isneginf(values))),
        "finite_min": float(np.min(finite)) if finite.size else np.nan,
        "finite_max": float(np.max(finite)) if finite.size else np.nan,
    }


def validate_operator_splitting(splitting):
    """Return ``splitting`` unchanged if it names an implemented composition."""
    if splitting not in OPERATOR_SPLITTINGS:
        raise ValueError(
            "operator_splitting must be one of "
            f"{sorted(OPERATOR_SPLITTINGS)} (got {splitting!r})"
        )
    return splitting


def validate_r1_configuration_presence(
    input_dict,
    flags,
    *,
    geometry,
    ion_neutral_moment_closure,
    hyperbolic_wave_speed,
    characteristic_boundary,
    raw_stage_validation,
):
    """Reject R1-audited controls that would otherwise be silent no-ops."""
    frozen_controls = {
        "front_flux_model": (
            str(input_dict.get("front_flux_model")),
            "sonic_relaxation",
        ),
        "D_amb_model": (
            str(input_dict.get("D_amb_model")),
            "cs_dz",
        ),
        "D_amb": (
            float(input_dict.get("D_amb")),
            0.0,
        ),
        "cathode_model": (
            str(input_dict.get("cathode_model")),
            "disabled",
        ),
    }
    changed = [
        name
        for name, (actual, canonical) in frozen_controls.items()
        if actual != canonical
    ]
    if changed:
        raise ValueError(
            "R1-audited compatibility/boundary controls are frozen at "
            "their checkpoint values until their owning repair supplies "
            "a replacement operator; noncanonical values would be silent "
            "no-ops: "
            + ", ".join(changed)
        )
    # R5 stance flip (2026-07-25) deprecations. These paths remain runnable
    # (A/B arms + tag reproducibility) but are superseded by the repaired
    # production baseline; a non-default/active use warns.
    if not ion_neutral_moment_closure:
        warnings.warn(
            "the legacy ion-neutral drag/CX/thermalization path "
            "(ion_neutral_moment_closure=False, with b_ion_neutral_drag, "
            "ion_neutral_drag_model, b_ion_neutral_thermalization, and the "
            "Tn_fit collision temperature) is DEPRECATED: the Phelps "
            "moment-closed operator (ion_neutral_moment_closure) is the "
            "production drag baseline. Still runnable as an A/B arm and for "
            "reproducing old results at tag legacy-final-2026-07-22.",
            DeprecationWarning,
            stacklevel=2,
        )
    _gp_mode = str(input_dict.get("gas_puff_mode", "square"))
    if _gp_mode in ("pulse_decay_to_level", "decay_after_breakdown", "double_erf"):
        warnings.warn(
            f"gas_puff_mode={_gp_mode!r} is DEPRECATED (the measured "
            "waveform is 'square'); retained runnable only for the frozen "
            "waveform-comparison figures.",
            DeprecationWarning,
            stacklevel=2,
        )
    _deprecated_selectors = {
        "D_amb_model": (str(input_dict.get("D_amb_model", "cs_dz")), "cs_dz"),
        "cathode_model": (
            str(input_dict.get("cathode_model", "disabled")), "disabled",
        ),
    }
    _sel = [n for n, (a, d) in _deprecated_selectors.items() if a != d]
    if _sel:
        warnings.warn(
            "legacy-compat selectors " + ", ".join(_sel) + " are DEPRECATED "
            "and never consumed by the conservative solver (D_amb_model was "
            "a _sim3-compat knob; cathode_model is superseded by the "
            "cathode_coupling flag).",
            DeprecationWarning,
            stacklevel=2,
        )
    # "neutral" exists on the ION side only: it is the partner of the En
    # ionization sink, which debits the neutral energy field and has no
    # electron counterpart to pair with.
    for name, selectors, allowed in (
        ("Te_birth_ionization", ("local", "floor"), "'local' or 'floor'"),
        (
            "Ti_birth_ionization",
            ("local", "floor", "neutral"),
            "'local', 'floor', or 'neutral'",
        ),
    ):
        value = input_dict.get(name)
        if isinstance(value, str):
            if value not in set(selectors):
                raise ValueError(
                    f"{name} must be {allowed}, or a finite "
                    f"non-negative numeric eV value (got {value!r})"
                )
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = np.nan
        if not np.isfinite(numeric) or numeric < 0.0:
            raise ValueError(
                f"{name} must be {allowed}, or a finite "
                f"non-negative numeric eV value (got {value!r})"
            )
    birth_energy_model = str(
        input_dict.get("ionization_birth_energy_model", "legacy")
    )
    if birth_energy_model not in {"legacy", "conservative"}:
        raise ValueError(
            "ionization_birth_energy_model must be 'legacy' or "
            f"'conservative' (got {birth_energy_model!r})"
        )
    end_mode = str(input_dict.get("end_mode", "collector"))
    if end_mode != "collector":
        raise ValueError(
            f"end_mode={end_mode!r} is not available: the 'mirrored_source' "
            "end boundary was removed at D3, 2026-08-21 (it was a 0D-era "
            "selector that the conservative solver never branched on). "
            "Accepted: 'collector'."
        )
    if hyperbolic_wave_speed not in {"isothermal", "adiabatic"}:
        raise ValueError(
            "hyperbolic_wave_speed must be 'isothermal' or 'adiabatic' "
            f"(got {hyperbolic_wave_speed!r})"
        )
    if characteristic_boundary:
        # R3.1 characteristic ghost-cell Bohm outflow (audit A1/A16). It acts
        # only on plasma-terminating (absorbing) faces, which exist only in
        # the resolved geometry; without them the flag would be a silent
        # no-op (R1d discipline).
        absorbing = np.asarray(
            getattr(geometry, "plasma_absorbing", np.zeros(0)),
            dtype=bool,
        )
        if not np.any(absorbing):
            raise ValueError(
                "characteristic_boundary requires plasma-terminating "
                "(absorbing) faces, which exist only in the resolved "
                "geometry (resolved_boundaries=True); it would otherwise be "
                "a silent no-op"
            )
    # R4.1 anode-mesh beam interception (audit A15) is the production default
    # (correct csda physics). Like beam_coulomb_model / beam_anomalous_model it
    # is a csda control: it perturbs the operator under beam_deposition_model=
    # "csda" with resolved anode faces, and is inert under beer_lambert (which
    # never launches the CSDA module) or where no anode faces exist. The
    # _csda_beam_deposition wiring applies it only when eta>0 and anode faces
    # are present, so no construction rejection is needed.
    if raw_stage_validation and flags.get("Plasma", True):
        for initial_name, floor_name in (
            ("Te0", "Te_floor"),
            ("Ti0", "Ti_floor"),
        ):
            initial = float(input_dict[initial_name])
            floor = float(input_dict[floor_name])
            if not initial > floor:
                raise ValueError(
                    f"{initial_name} must be strictly greater than "
                    f"{floor_name} when raw_stage_validation=True "
                    f"(got {initial} <= {floor})"
                )


def validate_equilibration_gas_puff_on(input_dict):
    """Reject a nonsense equilibration puff width (loud, at construction).

    ``equilibration_gas_puff_on_s`` overrides the neutral-equilibration
    inner sim's per-cycle puff-ON window. ``None`` means "unset" (fall back
    to ``tau_discharge``); anything else must be a real, finite, positive
    duration that fits inside one puff/off cycle. A zero, negative, or
    longer-than-the-cycle value would silently produce a 0% or >100% duty
    instead of the measured window.
    """
    raw = input_dict.get("equilibration_gas_puff_on_s", None)
    if raw is None:
        return
    try:
        puff_on = float(raw)
    except (TypeError, ValueError):
        raise ValueError(
            "equilibration_gas_puff_on_s (the equilibration puff-ON window "
            f"[s]) must be a number or None (got {raw!r})"
        ) from None
    if not np.isfinite(puff_on) or puff_on <= 0.0:
        raise ValueError(
            "equilibration_gas_puff_on_s (the equilibration puff-ON window "
            f"[s]) must be finite and > 0 (got {puff_on!r}); use None to "
            "fall back to tau_discharge"
        )
    tau_cycle = float(input_dict.get("tau_cycle", 0.0))
    if tau_cycle > 0.0 and puff_on > tau_cycle:
        raise ValueError(
            "equilibration_gas_puff_on_s (the equilibration puff-ON window "
            f"[s]) must fit inside one puff/off cycle: got {puff_on!r} > "
            f"tau_cycle={tau_cycle!r}"
        )


def validate_neutral_seed_cache_config(input_dict, flags):
    """Reject an incoherent cached-neutral-seed configuration (loud, at build).

    ``use_cached_neutral_seed`` replaces the live neutral equilibration with a
    cached seed, so it requires the equilibration pipeline to be selected
    (``neutral_equilibration`` + ``launch_plasma_after_equilibration``) and a
    cache path. A missing path or a contradictory flag would otherwise be a
    silent no-op.
    """
    if not flags.get("use_cached_neutral_seed", False):
        return
    problems = []
    if not flags.get("neutral_equilibration", False):
        problems.append(
            "neutral_equilibration must be ON (the cache seeds that pipeline)"
        )
    if not flags.get("launch_plasma_after_equilibration", False):
        problems.append(
            "launch_plasma_after_equilibration must be ON (nothing to seed "
            "otherwise)"
        )
    if not input_dict.get("neutral_seed_cache_dir"):
        problems.append(
            "neutral_seed_cache_dir must be set to the seed-database directory"
        )
    if problems:
        raise ValueError(
            "use_cached_neutral_seed is ON but the configuration is "
            "incoherent: " + "; ".join(problems)
        )


def validate_phase_config(mode, action):
    """Reject unknown phase-transition / prebreakdown-timeout selectors."""
    if mode not in {"scheduled", "current"}:
        raise ValueError(
            "phase_transition_mode must be 'scheduled' or 'current' "
            f"(got {mode!r})"
        )
    if action not in {"switch_open", "raise"}:
        raise ValueError(
            "prebreakdown_timeout_action must be 'switch_open' or "
            f"'raise' (got {action!r})"
        )


def validate_gas_puff_config(input_dict):
    mode = input_dict.get("gas_puff_mode", "decay_after_breakdown")
    if mode not in {
        "decay_after_breakdown",
        "pulse_decay_to_level",
        "double_erf",
        "square",
    }:
        raise ValueError(
            "gas_puff_mode must be 'decay_after_breakdown', "
            "'pulse_decay_to_level', 'double_erf', or 'square' "
            f"(got {mode!r})"
        )
    if mode == "square":
        for key in ("gas_puff_rise_width_s",):
            width = float(input_dict.get(key, 5.0e-4))
            if width <= 0.0:
                raise ValueError(f"{key} must be positive (got {width})")
        for key in ("gas_puff_rise_center_s", "gas_puff_close_lag_s"):
            value = float(input_dict.get(key, 5.0e-4))
            if value < 0.0:
                raise ValueError(f"{key} must be >= 0 (got {value})")
    if mode == "double_erf":
        for key in ("tau_gp_rise_width", "tau_gp_drop_width"):
            width = float(input_dict.get(key, 1e-3))
            if width <= 0.0:
                raise ValueError(f"{key} must be positive (got {width})")
    tau_after_breakdown = input_dict.get("tau_gp_after_breakdown", None)
    if tau_after_breakdown is not None and float(tau_after_breakdown) < 0.0:
        raise ValueError(
            "tau_gp_after_breakdown must be >= 0 s, or None to keep S_gp "
            f"steady (got {tau_after_breakdown})"
        )
    tau_decay_factor = float(input_dict.get("tau_gp_decay_factor", 1.0))
    if tau_decay_factor <= 0.0:
        raise ValueError(
            f"tau_gp_decay_factor must be > 0 (got {tau_decay_factor})"
        )
    tau_pulse_duration = float(input_dict.get("tau_gp_pulse_duration", 0.0))
    if tau_pulse_duration < 0.0:
        raise ValueError(
            f"tau_gp_pulse_duration must be >= 0 (got {tau_pulse_duration})"
        )
    tau_decay_duration = float(input_dict.get("tau_gp_decay_duration", 1e-3))
    if tau_decay_duration <= 0.0:
        raise ValueError(
            f"tau_gp_decay_duration must be > 0 (got {tau_decay_duration})"
        )
    validate_gas_puff_orifice_config(input_dict)


def validate_gas_puff_orifice_config(input_dict):
    """Presence-gate the two feed-pipe keys against ``gas_puff_profile``.

    They belong to ``gas_puff_profile = "orifice"`` and to nothing else, so
    both directions raise: set without the profile they would be silently
    inert, and missing with it there is no aperture to derive a row from.
    Also refuses an aspect ratio the long-tube angular law has no branch for,
    and a shut valve, where the derivation would place no flow while still
    reporting itself as the injection geometry.

    The refusals that need the MESH -- a port off the grid, a plasma column
    that is not inside the vessel wall -- are raised by the row derivation
    itself, which the solver runs once at construction for that reason.
    """
    profile = input_dict.get("gas_puff_profile", "cell")
    keys = ("gas_puff_orifice_id_cm", "gas_puff_orifice_length_cm")
    values = {key: input_dict.get(key) for key in keys}
    for key, value in values.items():
        if value is not None and profile != "orifice":
            raise ValueError(
                f"{key} belongs to gas_puff_profile='orifice' and is inert "
                f"under {profile!r}; drop it or change the profile"
            )
        if value is None and profile == "orifice":
            raise ValueError(
                f"gas_puff_profile='orifice' requires {key}: the tube-beamed "
                "row is derived from the feed pipe's own bore and length, and "
                "there is no default aperture to fall back on"
            )
    if profile != "orifice":
        return
    bore = float(values["gas_puff_orifice_id_cm"])
    length = float(values["gas_puff_orifice_length_cm"])
    for key, value in (
        ("gas_puff_orifice_id_cm", bore),
        ("gas_puff_orifice_length_cm", length),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"{key} must be finite and positive (got {value})"
            )
    if length / bore < 4.0 / 3.0:
        raise ValueError(
            "gas_puff_orifice_length_cm / gas_puff_orifice_id_cm must be "
            f">= 4/3 (got {length} / {bore} = {length / bore}): the beaming "
            "law is a LONG-tube result whose end-effect prescription inverts "
            "below that ratio, and it has no short-tube branch"
        )
    if not bool(input_dict.get("gas_puff_enabled", True)):
        raise ValueError(
            "gas_puff_profile='orifice' derives the injection geometry of a "
            "puff that gas_puff_enabled=False never delivers; enable the puff "
            "or choose a profile that is not a derivation"
        )
    flow = (
        float(input_dict.get("S_gp", 0.0))
        * float(input_dict.get("gas_puff_valves", 2))
        * float(input_dict.get("gas_puff_delivery_fraction", 1.0))
    )
    if flow <= 0.0:
        raise ValueError(
            "gas_puff_profile='orifice' was configured but the puff delivers "
            f"no flow (S_gp x gas_puff_valves x gas_puff_delivery_fraction = "
            f"{flow}): there is nothing to place, and a derived row over a "
            "shut valve would report a geometry it never applies"
        )


def resolve_neutral_jet_config(
    input_dict, *, geometry, neutral_momentum, neutral_energy
):
    """Validate and RESOLVE the directed-recycle-jet configuration.

    The jets and the mesh accommodation are M_n physics: they require the
    neutral_momentum flag, and each channel requires the geometry feature
    it rides on (an absorbing cathode face; anode faces with eta > 0), so
    a misconfigured jet fails loudly instead of silently never firing.

    Returns the resolved record the solver arms its jet attributes from.
    """
    p = input_dict
    cathode_jet_enabled = bool(p.get("cathode_neutral_jet", False))
    anode_jet_enabled = bool(p.get("anode_neutral_jet", False))
    mesh_accommodation = bool(
        p.get("neutral_mesh_accommodation", False)
    )
    surface_debit = bool(p.get("cathode_jet_surface_debit", False))
    R_coeffs = {}
    for prefix, enabled in (
        ("cathode_jet", cathode_jet_enabled),
        ("anode_jet", anode_jet_enabled),
    ):
        R_N = float(p.get(f"{prefix}_R_N", 0.0))
        R_E = float(p.get(f"{prefix}_R_E", 0.0))
        if enabled and not (0.0 <= R_N <= 1.0 and 0.0 <= R_E <= 1.0):
            raise ValueError(
                f"{prefix}_R_N and {prefix}_R_E are particle/energy "
                "reflection coefficients and must lie in [0, 1] "
                f"(got R_N={R_N}, R_E={R_E})"
            )
        R_coeffs[f"{prefix}_R_N"] = R_N
        R_coeffs[f"{prefix}_R_E"] = R_E
    needs_mn = (
        cathode_jet_enabled
        or anode_jet_enabled
        or mesh_accommodation
    )
    if needs_mn and not neutral_momentum:
        raise ValueError(
            "cathode_neutral_jet / anode_neutral_jet / "
            "neutral_mesh_accommodation are M_n momentum physics and "
            "require the neutral_momentum flag"
        )
    roles = np.asarray(geometry.cell_role)
    absorbing = np.asarray(
        getattr(geometry, "plasma_absorbing", np.zeros(0)),
        dtype=bool,
    )
    if cathode_jet_enabled and not (
        np.any(absorbing) and np.any(roles == "cathode")
    ):
        raise ValueError(
            "cathode_neutral_jet requires an absorbing cathode face "
            "(resolved_boundaries geometry): the jet rides the "
            "boundary-absorption recycle flux"
        )
    anode_faces = np.asarray(
        getattr(geometry, "anode_face_indices", ()), dtype=int
    )
    eta = float(p.get("eta", 0.0))
    if (anode_jet_enabled or mesh_accommodation) and (
        anode_faces.size == 0 or eta <= 0.0
    ):
        raise ValueError(
            "anode_neutral_jet / neutral_mesh_accommodation require "
            "anode faces with eta > 0 (resolved geometry with a mesh)"
        )
    if surface_debit and not cathode_jet_enabled:
        raise ValueError(
            "cathode_jet_surface_debit reads the cathode jet's R_E and "
            "requires cathode_neutral_jet"
        )
    # The directed hot surface carrier: it takes over the backscatter share of
    # the cathode recycle, so it needs the jet that produces that share, the
    # debit that lets the surface give the energy up, and an En field for the
    # CX partner atoms to be born into. Each prerequisite raises on its own,
    # naming what is missing -- never a silent fallback to the v1 booking.
    carrier = bool(p.get("cathode_jet_hot_carrier", False))
    if carrier and not cathode_jet_enabled:
        raise ValueError(
            "cathode_jet_hot_carrier carries the CATHODE JET's backscatter "
            "share and requires cathode_neutral_jet: without that jet there "
            "is no R_N stream for it to own. Accepted: "
            "cathode_neutral_jet=True, or cathode_jet_hot_carrier=False"
        )
    if carrier and not surface_debit:
        raise ValueError(
            "cathode_jet_hot_carrier requires cathode_jet_surface_debit=True: "
            "the beam's launch energy is the R_E share of the ion bombardment "
            "power, and without the debit the surface keeps that power too, "
            "so the same R_E would be spent twice. This is not flipped for "
            "you -- the debit changes the cathode's power balance, which is a "
            "stance decision. Accepted: cathode_jet_surface_debit=True, or "
            "cathode_jet_hot_carrier=False"
        )
    if carrier and not neutral_energy:
        raise ValueError(
            "cathode_jet_hot_carrier requires the neutral_energy flag: every "
            "charge exchange along the beam returns an atom born at the LOCAL "
            "ION STATE, and without an En field there is nowhere to book the "
            "(3/2) k Ti it carries -- the ion debit would be one-sided. "
            "Accepted: neutral_energy=True, or cathode_jet_hot_carrier=False"
        )
    # Which convention R_E is read in when the jet's launch energy is
    # built. "legacy" is the historical reading and is bit-exact.
    convention = p.get("cathode_jet_energy_convention", "legacy")
    if convention not in CATHODE_JET_ENERGY_CONVENTIONS:
        raise ValueError(
            "cathode_jet_energy_convention must be one of "
            f"{CATHODE_JET_ENERGY_CONVENTIONS} (got {convention!r})"
        )
    cathode_jet_energy_convention = convention
    if convention == "total_reflected":
        if not cathode_jet_enabled:
            raise ValueError(
                "cathode_jet_energy_convention='total_reflected' rescales "
                "the cathode jet's launch energy and requires "
                "cathode_neutral_jet"
            )
        R_N = R_coeffs["cathode_jet_R_N"]
        R_E = R_coeffs["cathode_jet_R_E"]
        if not (0.0 < R_E <= R_N < 1.0):
            raise ValueError(
                "cathode_jet_energy_convention='total_reflected' reads "
                "cathode_jet_R_E as the TOTAL reflected energy fraction "
                "and gives each of the cathode_jet_R_N backscattered "
                "particles R_E/R_N of the incident energy, so it requires "
                "0 < cathode_jet_R_E <= cathode_jet_R_N < 1 (a reflected "
                "particle cannot carry more energy than it arrived with, "
                "and neither coefficient may be degenerate) -- got "
                f"cathode_jet_R_E={R_E}, cathode_jet_R_N={R_N}"
            )
    # The anode jet's own convention key. It ships UNDECLARED (``None``): the
    # tabulated reflection coefficients are published as TOTAL reflected
    # fractions while the channel was hard-coded to read R_E per backscattered
    # particle, so arming the jet without saying which reading applies runs the
    # momentum channel ~21 % low and says nothing about it. That is the failure
    # this guard exists to make impossible, which is why the key has no
    # default reading to fall back on.
    anode_convention = p.get("anode_jet_energy_convention", None)
    if anode_convention is not None and (
        anode_convention not in ANODE_JET_ENERGY_CONVENTIONS
    ):
        raise ValueError(
            "anode_jet_energy_convention must be None or one of "
            f"{ANODE_JET_ENERGY_CONVENTIONS} (got {anode_convention!r})"
        )
    if anode_jet_enabled and anode_convention is None:
        raise ValueError(
            "anode_neutral_jet is armed but anode_jet_energy_convention is "
            "undeclared (None). anode_jet_R_E can be read PER BACKSCATTERED "
            "PARTICLE ('legacy') or as the TOTAL reflected energy fraction "
            "('total_reflected', in which case each of the anode_jet_R_N "
            "backscattered particles carries R_E/R_N of the incident "
            "energy). The two give different launch speeds from the same "
            "number, so the reading is a stance decision and is not chosen "
            "for you"
        )
    if anode_convention == "total_reflected" and not anode_jet_enabled:
        raise ValueError(
            "anode_jet_energy_convention='total_reflected' rescales the "
            "anode jet's launch energy and requires anode_neutral_jet"
        )
    if anode_convention == "total_reflected":
        R_N = R_coeffs["anode_jet_R_N"]
        R_E = R_coeffs["anode_jet_R_E"]
        if not (0.0 < R_E <= R_N < 1.0):
            raise ValueError(
                "anode_jet_energy_convention='total_reflected' reads "
                "anode_jet_R_E as the TOTAL reflected energy fraction and "
                "gives each of the anode_jet_R_N backscattered particles "
                "R_E/R_N of the incident energy, so it requires "
                "0 < anode_jet_R_E <= anode_jet_R_N < 1 (a reflected "
                "particle cannot carry more energy than it arrived with, "
                "and neither coefficient may be degenerate) -- got "
                f"anode_jet_R_E={R_E}, anode_jet_R_N={R_N}"
            )
    if neutral_energy and cathode_jet_enabled and not surface_debit:
        raise ValueError(
            "cathode_neutral_jet with neutral_energy requires "
            "cathode_jet_surface_debit=True: the R_E share of the ion "
            "bombardment power is the energy the backscattered atoms "
            "carry away, and with an En field that energy is now BOOKED "
            "into the neutral gas. Without the debit the surface keeps it "
            "too, so the same R_E is spent twice. This is not flipped for "
            "you -- the debit changes the cathode's power balance, which "
            "is a stance decision, not a plumbing one. Accepted: "
            "cathode_jet_surface_debit=True, or neutral_energy without "
            "cathode_neutral_jet"
        )
    # Reflected-energy retention for the surface power balance:
    # (1 - R_E) of the ion bombardment power stays in the surface when
    # the debit sensitivity arm is on; 1.0 (the M5a' calibration
    # convention) otherwise.
    cathode_surface_ion_retention = (
        1.0 - R_coeffs["cathode_jet_R_E"] if surface_debit else 1.0
    )
    # Blocked mesh area for the wind's momentum accommodation: the open
    # fraction T = 1 - eta*(Ra/Rm)^2 already lives in the face area, so
    # A_blocked = A_open * (1 - T) / T.
    if mesh_accommodation:
        transparency = _anode_neutral_transparency(p)
        if transparency <= 0.0:
            raise ValueError(
                "neutral_mesh_accommodation requires a mesh with open "
                f"neutral area (transparency {transparency})"
            )
        open_area = np.asarray(
            geometry.neutral_face_area_cm2, dtype=float
        )[anode_faces]
        mesh_faces = anode_faces
        mesh_blocked_area_cm2 = (
            open_area * (1.0 - transparency) / transparency
        )
    else:
        mesh_faces = None
        mesh_blocked_area_cm2 = None
    return SimpleNamespace(
        cathode_jet_enabled=cathode_jet_enabled,
        anode_jet_enabled=anode_jet_enabled,
        mesh_accommodation=mesh_accommodation,
        cathode_jet_R_N=R_coeffs["cathode_jet_R_N"],
        cathode_jet_R_E=R_coeffs["cathode_jet_R_E"],
        anode_jet_R_N=R_coeffs["anode_jet_R_N"],
        anode_jet_R_E=R_coeffs["anode_jet_R_E"],
        cathode_jet_energy_convention=cathode_jet_energy_convention,
        anode_jet_energy_convention=anode_convention,
        cathode_jet_carrier=carrier,
        cathode_surface_ion_retention=cathode_surface_ion_retention,
        mesh_faces=mesh_faces,
        mesh_blocked_area_cm2=mesh_blocked_area_cm2,
    )


def resolve_coverage_config(input_dict, flags, *, geometry, neutral_model):
    """Validate and RESOLVE the clumpy-plasma coverage closure (v2).

    Every failure here is a construction-time ``ValueError``: an
    incomplete or unrepresentable coverage configuration must never reach
    the first cathode solve. With the flag off the four coverage keys
    must all sit at their defaults, so a run that configures the closure
    and forgets the flag is loud rather than silently mean-field.

    Returns the resolved record the solver arms its coverage attributes from.
    """
    enabled = bool(flags.get("coverage_closure", False))
    r = input_dict.get("coverage_growth_rate_per_s", 0.0)
    tau = input_dict.get("coverage_backfill_time_s", 0.0)
    f0 = input_dict.get("coverage_initial_fraction", None)
    profile = input_dict.get("coverage_initial_profile", None)
    if not enabled:
        defaults = coverage_closure_defaults()

        def _is_default(value, default):
            # coverage_initial_profile is sequence-valued, and ``!=`` on a
            # sequence is elementwise, so the comparison is reduced to one
            # bool here before it is used as a truth value.
            if value is None or default is None:
                return value is None and default is None
            return bool(np.array_equal(value, default))

        # coverage_growth_rate_per_s is the SHARED percolation clock: the
        # cathode emitting-area closure reads the same key rather than
        # minting a second rate, so it is live -- and a non-default value
        # legitimate -- whenever that flag is armed. The other three keys
        # are the column closure's alone and stay inert.
        inert = (
            ("coverage_backfill_time_s", tau),
            ("coverage_initial_fraction", f0),
            ("coverage_initial_profile", profile),
        )
        if not bool(flags.get("cathode_emitting_area", False)):
            inert = (("coverage_growth_rate_per_s", r),) + inert
        configured = [
            name
            for name, value in inert
            if not _is_default(value, defaults[name])
        ]
        if configured:
            raise ValueError(
                "the coverage-closure parameters "
                f"{sorted(configured)} were configured without the "
                "coverage_closure flag, where they are inert; set the "
                "flag or drop the parameters"
            )
        return SimpleNamespace(
            coverage=None,
            r=0.0,
            tau_s=0.0,
            f=None,
            deficit=None,
            burn_accum=None,
            burn_weight=0.0,
            w_accum=None,
            reservoir_debit=None,
            reservoir_burn_accum=None,
        )
    cells = geometry.cells
    if (f0 is None) == (profile is None):
        raise ValueError(
            "the coverage_closure flag requires EXACTLY ONE initial "
            "condition: coverage_initial_fraction (one uniform covered "
            "fraction in (0, 1]) or coverage_initial_profile (a per-cell "
            f"f_cov0 of length nx={cells}). "
            + (
                "Both were given; they are two spellings of the same "
                "initial condition and neither modifies the other, so "
                "there is no composition rule to apply -- drop one."
                if f0 is not None
                else "Neither was given; there is no neutral default -- "
                "1.0 is the fully-covered mean-field limit and would "
                "make the closure a silent no-op."
            )
        )
    if profile is not None:
        f_init = np.asarray(profile, dtype=float).reshape(-1)
        if f_init.size != cells:
            raise ValueError(
                "coverage_initial_profile must have one entry per grid "
                f"cell (nx={cells}); got {f_init.size}"
            )
        if not np.all(np.isfinite(f_init)) or np.any(
            f_init <= 0.0
        ) or np.any(f_init > 1.0):
            raise ValueError(
                "every coverage_initial_profile entry must be finite and "
                f"in (0, 1] (got min {float(np.min(f_init)):.6g}, max "
                f"{float(np.max(f_init)):.6g})"
            )
    else:
        f0 = float(f0)
        if not (math.isfinite(f0) and 0.0 < f0 <= 1.0):
            raise ValueError(
                "coverage_initial_fraction must be finite and in (0, 1] "
                f"(got {f0!r})"
            )
        f_init = np.full(cells, f0, dtype=float)
    r = float(r)
    if not (math.isfinite(r) and r >= 0.0):
        raise ValueError(
            "coverage_growth_rate_per_s (the column-mean logistic rate of "
            "df_cov/dt = r0*w*f_cov*(1-f_cov)) must be finite and >= 0 "
            f"(got {r!r})"
        )
    tau = float(tau)
    if not (math.isfinite(tau) and tau > 0.0):
        raise ValueError(
            "coverage_backfill_time_s (the reservoir->column neutral "
            f"refill time) must be finite and > 0 (got {tau!r})"
        )
    if str(
        input_dict.get("beam_deposition_model", "beer_lambert")
    ) != "csda":
        raise ValueError(
            "coverage_closure requires beam_deposition_model='csda': the "
            "closure splits the beam by area across the covered and "
            "reservoir media, and that split is built on the CSDA rays. "
            "Under 'beer_lambert' there is no second ray to give the "
            "reservoir, so the whole beam would be routed through the "
            "channels while the closure's own premise says only f_cov of "
            "it goes there -- a silently inconsistent model rather than a "
            "no-op, which is why this refuses instead of degrading"
        )
    if float(input_dict.get("beam_clump_fraction", 0.0)) > 0.0:
        raise ValueError(
            "coverage_closure is incompatible with beam_clump_fraction > "
            "0: both split the beam into rays over different neutral "
            "media, and their product is a four-ray composition this "
            "build does not define. Disable one"
        )
    if neutral_model != "moment":
        raise ValueError(
            "coverage_closure requires neutral_model='moment' (got "
            f"{neutral_model!r}): the kinetic arms take over the "
            "fluid nn rows once engaged, and the closure's covered-column "
            "burn is read from exactly those rows, so under a kinetic "
            "neutral model the column would never deplete and the "
            "backfill would be a silent no-op"
        )
    # NB there is deliberately NO refusal of the compiled kernels here.
    # v1 carried one, on the belief that the closure's beam split ran on
    # transcribed arithmetic that had never been bit-compared under
    # coverage. That is not what the opt-in reaches: the compiled march
    # (``_CSDA_MARCH``) is bound only inside ``deposit_beam``, the
    # SINGLE-MEDIUM ray, and ``deposit_beam_two_stream`` -- the closure's
    # own two-medium wrapper, its per-cell re-split, its re-mix and all of
    # its banking -- has no compiled branch at all. So under coverage the
    # opt-in accelerates exactly the nested single-medium walker marches
    # (the ray shape the tierA+csda transcription was bit-verified over)
    # plus the tier-A cathode kernels, and both paths were measured
    # raw-uint64 identical over coverage trajectories before the refusal
    # was lifted. Bit-identity, not the refusal, is the standing guard:
    # smoke's compiled-kernel equivalence block runs a beam-live coverage
    # arm both ways and asserts the raw state bytes match.
    return SimpleNamespace(
        coverage=True,
        r=r,
        tau_s=tau,
        # The coverage FIELD itself [1], per cell, in (0, 1]. v2 co-integrates
        # it (see _advance_coverage_fraction): its growth law is driven by the
        # beam ionization the coverage itself shapes, so there is no closed
        # form to evaluate and the field is carried as accepted-step state.
        f=f_init,
        # The covered column's neutral DEFICIT relative to the cell mean
        # [cm^-3], per cell. The mean field nn is untouched by the closure and
        # keeps every particle, so this auxiliary is a pure re-partition and
        # total inventory is conserved identically whatever happens to it.
        # It starts at zero: at the phase origin nothing has burnt yet.
        deficit=np.zeros(geometry.cells, dtype=float),
        burn_accum=None,
        burn_weight=0.0,
        # The stage-accumulated growth driver for the CURRENT attempt; armed by
        # _attempt_step and dropped with the attempt, exactly like the burn
        # tally above, so a rejected step cannot advance the field.
        w_accum=None,
        # The reservoir arm's neutral debit published by the beam terms of the
        # CURRENT RHS evaluation; reset by rhs_terms on every call so it can
        # never be read from a stale solve.
        reservoir_debit=None,
        reservoir_burn_accum=None,
    )


def resolve_emitting_area_config(input_dict, flags):
    """Validate and RESOLVE the cathode emitting-area closure (ea1).

    Every failure here is a construction-time ``ValueError``: a throttle
    that cannot be applied, or one that would be silently inert, must never
    reach the first cathode solve. With the flag off the seed key must sit
    at its shipped value, so a run that sets a seed and forgets the flag is
    loud rather than silently fully lit.

    Returns the lit-area fraction -- the closure's whole state -- or ``None``
    when the flag is off, which is the presence gate every consumer reads.
    """
    enabled = bool(flags.get("cathode_emitting_area", False))
    default_f0 = emitting_area_defaults()[
        "cathode_emitting_area_initial_fraction"
    ]
    f0 = input_dict.get(
        "cathode_emitting_area_initial_fraction", default_f0
    )
    if not enabled:
        if f0 != default_f0:
            raise ValueError(
                "cathode_emitting_area_initial_fraction was configured "
                f"({f0!r}) without the cathode_emitting_area flag, where "
                "it is inert; set the flag or drop the parameter"
            )
        return None
    if f0 is None or not (
        math.isfinite(float(f0)) and 0.0 < float(f0) <= 1.0
    ):
        raise ValueError(
            "cathode_emitting_area_initial_fraction (the lit fraction of "
            "the emitting face at the time origin) must be finite and in "
            f"(0, 1] (got {f0!r})"
        )
    if not bool(flags.get("cathode_coupling", False)):
        raise ValueError(
            "cathode_emitting_area requires cathode_coupling: the closure "
            "throttles the thermionic emission of the cathode solve, and "
            "with the coupling off there is no such solve, so the flag "
            "would be a silent no-op"
        )
    profile = str(
        input_dict.get("cathode_emission_profile", "uniform")
    )
    if profile != "gaussian":
        raise ValueError(
            "cathode_emitting_area requires "
            "cathode_emission_profile='gaussian' (got "
            f"{profile!r}): under 'uniform' the disc area A_c sets the "
            "Richardson emission AND collects the ion current, so a lit "
            "fraction applied to it would throttle the ion sink along "
            "with the emission -- the throttle is not expressible there"
        )
    # The growth rate is the SHARED percolation clock, read from the
    # coverage closure's key. It is validated here too because this flag
    # can be armed with that closure off, in which case nothing else
    # checks it.
    r = input_dict.get("coverage_growth_rate_per_s", 0.0)
    if not (math.isfinite(float(r)) and float(r) >= 0.0):
        raise ValueError(
            "coverage_growth_rate_per_s (the shared percolation clock of "
            "df_em/dt = r*f_em*(1-f_em)) must be finite and >= 0 "
            f"(got {r!r})"
        )
    return float(f0)


def resolve_neutral_probe_config(
    input_dict, flags, *, geometry, neutral_model, neutral_two_zone
):
    """Validate and RESOLVE the ad-hoc probe neutral source (v1).

    Every failure here is a construction-time ``ValueError``: an incomplete
    or unrepresentable probe configuration must never reach the first step.
    With the flag off all ten probe keys must sit at their ``None``
    defaults, so a run that configures a probe and forgets the flag is loud
    rather than silently unprobed.

    Returns the resolved instrument -- amplitude, normalized axial weights,
    waveform selector and its own parameters, and the two-zone target -- or
    ``None`` when the flag is off, which is the presence gate every consumer
    reads.
    """
    enabled = bool(flags.get("neutral_probe_source", False))
    defaults = neutral_probe_source_defaults()
    values = {
        name: input_dict.get(name, default)
        for name, default in defaults.items()
    }
    if not enabled:
        # Every default in this group is None -- the instrument ships no
        # number, deliberately -- so "at its default" is exactly "is
        # None", and the two sequence-valued keys need no elementwise
        # comparison here.
        configured = [
            name for name, value in values.items() if value is not None
        ]
        if configured:
            raise ValueError(
                "the probe-source parameters "
                f"{sorted(configured)} were configured without the "
                "neutral_probe_source flag, where they are inert; set the "
                "flag or drop the parameters"
            )
        return None
    if neutral_model != "moment":
        raise ValueError(
            "neutral_probe_source requires neutral_model='moment' (got "
            f"{neutral_model!r}): the kinetic arms take over the "
            "fluid nn rows once engaged, so a source written into those "
            "rows would be stripped or double-counted rather than felt -- "
            "the probe would silently inject nothing. Supporting the "
            "kinetic arms means injecting into their distribution "
            "function, which is a different instrument, not a flag"
        )
    amplitude = values["neutral_probe_amplitude_cm3_s"]
    if amplitude is None:
        raise ValueError(
            "the neutral_probe_source flag requires "
            "neutral_probe_amplitude_cm3_s (the volume-mean source rate "
            "[cm^-3 s^-1] at w = 1). There is no default: the amplitude is "
            "the hypothesis the arm states, and 0 -- an explicit null "
            "control -- is a value that has to be asked for"
        )
    amplitude = float(amplitude)
    if not (math.isfinite(amplitude) and amplitude >= 0.0):
        raise ValueError(
            "neutral_probe_amplitude_cm3_s must be finite and >= 0 (got "
            f"{values['neutral_probe_amplitude_cm3_s']!r})"
        )
    # The shape's own presence gating (exactly one of profile/family, and
    # the family's parameters required with it and forbidden without it)
    # lives with the shape builder, so the rule and its implementation
    # cannot drift apart.
    shape = values["neutral_probe_shape"]
    for name, value in (
        ("neutral_probe_center_cm", values["neutral_probe_center_cm"]),
        ("neutral_probe_width_cm", values["neutral_probe_width_cm"]),
    ):
        if shape is None and value is not None:
            raise ValueError(
                f"{name} is a parameter of the built-in profile family and "
                "has no meaning without neutral_probe_shape; drop it, or "
                "select a family (this run supplies its own "
                "neutral_probe_profile)"
            )
        if shape == "gaussian" and value is None:
            raise ValueError(
                f"neutral_probe_shape='gaussian' requires {name}"
            )
    weights = neutral_probe_profile_weights(
        geometry,
        profile=values["neutral_probe_profile"],
        shape=shape,
        center_cm=values["neutral_probe_center_cm"],
        width_cm=values["neutral_probe_width_cm"],
    )
    waveform = values["neutral_probe_waveform"]
    if waveform is None:
        raise ValueError(
            "the neutral_probe_source flag requires "
            "neutral_probe_waveform, one of "
            f"{list(NEUTRAL_PROBE_WAVEFORMS)}. There is no default: the "
            "waveform decides what the arm measured"
        )
    if waveform not in NEUTRAL_PROBE_WAVEFORMS:
        raise ValueError(
            "neutral_probe_waveform must be one of "
            f"{list(NEUTRAL_PROBE_WAVEFORMS)} (got {waveform!r})"
        )
    t_on = values["neutral_probe_t_on_s"]
    t_off = values["neutral_probe_t_off_s"]
    table = values["neutral_probe_waveform_table"]
    for name, value, owner in (
        ("neutral_probe_t_on_s", t_on, "square"),
        ("neutral_probe_t_off_s", t_off, "square"),
        ("neutral_probe_waveform_table", table, "table"),
    ):
        if value is not None and waveform != owner:
            raise ValueError(
                f"{name} belongs to neutral_probe_waveform={owner!r} and "
                f"is inert under {waveform!r}; drop it or change the "
                "waveform"
            )
        if value is None and waveform == owner:
            raise ValueError(
                f"neutral_probe_waveform={owner!r} requires {name}"
            )
    if waveform == "square":
        t_on = float(t_on)
        t_off = float(t_off)
        if not (math.isfinite(t_on) and math.isfinite(t_off)):
            raise ValueError(
                "neutral_probe_t_on_s and neutral_probe_t_off_s must be "
                f"finite (got {t_on!r}, {t_off!r})"
            )
        if not t_on < t_off:
            raise ValueError(
                "neutral_probe_t_on_s must be strictly less than "
                f"neutral_probe_t_off_s (got {t_on!r} >= {t_off!r}); the "
                "square window is the half-open [t_on, t_off), so an empty "
                "or inverted window injects nothing and is a "
                "misconfiguration rather than a null control"
            )
    table_cumulative = None
    if waveform == "table":
        table, table_cumulative = neutral_probe_waveform_table(table)
    zone = values["neutral_probe_zone"]
    if neutral_two_zone:
        if zone not in ("column", "annulus"):
            raise ValueError(
                "under the neutral_two_zone closure the probe source "
                "requires neutral_probe_zone = 'column' (the plasma "
                "column, nn) or 'annulus' (the surrounding chamber, "
                f"nn_a); got {zone!r}. There is no default: the two put "
                "the gas in different places and the plasma responds to "
                "them differently, which is precisely what a probe arm is "
                "measuring"
            )
    elif zone is not None:
        raise ValueError(
            "neutral_probe_zone selects between the two-zone closure's "
            "column and annulus neutral fields and has no meaning without "
            f"the neutral_two_zone flag (got {zone!r}); this run has one "
            "neutral field"
        )
    # COVERAGE COMPOSES, deliberately and without a refusal. The closure
    # partitions the MEAN nn into a covered column and a reservoir through
    # a deficit that only the COVERAGE_BURN_TERMS move -- terms whose rate
    # is set by a plasma or beam density. The probe is not one of those: it
    # is uniform across the cross-section by construction, exactly like the
    # gas puff and the pump, which that ledger already names as
    # deliberately absent. So a probe raises the covered column and the
    # reservoir by the same amount, leaves the deficit untouched, and the
    # partition identity f*col + (1-f)*res = nn keeps closing. The answer
    # to "does probe-injected inventory belong to the reservoir or the
    # column?" is therefore neither-and-both, in area proportion, and it is
    # forced rather than chosen -- which is why this is an allowance with a
    # statement and not a guess.
    return SimpleNamespace(
        amplitude_cm3_s=amplitude,
        weights=weights,
        waveform=waveform,
        t_on_s=None if waveform != "square" else t_on,
        t_off_s=None if waveform != "square" else t_off,
        table=None if waveform != "table" else table,
        table_cumulative=table_cumulative,
        zone=zone,
    )


def validate_raw_stage(y, stage, unpack):
    """Reject non-finite/negative raw candidates before floor clipping."""
    packed_summary = _bad_array_summary(y)
    if packed_summary is not None:
        raise _RawStageError(
            y,
            stage,
            "nonfinite_state",
            {"stage": stage, "fields": {"packed_y": packed_summary}},
        )
    state = unpack(y)
    fields = {
        "n": state.n,
        "nn": state.nn,
        "M": state.M,
        "Ee": state.Ee,
        "Ei": state.Ei,
    }
    if state.M_n is not None:
        fields["M_n"] = state.M_n
    if state.nn_a is not None:
        fields["nn_a"] = state.nn_a
    if state.M_n_a is not None:
        fields["M_n_a"] = state.M_n_a
    if state.En is not None:
        fields["En"] = state.En
    nonfinite = {
        name: summary
        for name, values in fields.items()
        if (summary := _bad_array_summary(values)) is not None
    }
    if nonfinite:
        raise _RawStageError(
            y,
            stage,
            "nonfinite_state",
            {"stage": stage, "fields": nonfinite},
        )
    negative_density = {
        name: summary
        for name, values in (
            ("n", state.n),
            ("nn", state.nn),
            ("nn_a", state.nn_a),
        )
        if values is not None
        and (
            summary := _bad_array_summary(values, mode="negative")
        )
        is not None
    }
    if negative_density:
        raise _RawStageError(
            y,
            stage,
            "negative_density",
            {"stage": stage, "fields": negative_density},
        )
    negative_energy = {
        name: summary
        for name, values in (("Ee", state.Ee), ("Ei", state.Ei))
        if (
            summary := _bad_array_summary(values, mode="negative")
        )
        is not None
    }
    if negative_energy:
        raise _RawStageError(
            y,
            stage,
            "negative_energy",
            {"stage": stage, "fields": negative_energy},
        )
