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
from types import SimpleNamespace

import numpy as np

from cablp.atomic.adas import he_rate_temperature_range_eV

from .config import (
    geometry_defaults,
    parallel_momentum_sink_defaults,
)
from .geometry import _anode_neutral_transparency, _far_end_is_mirror
from ..physics.sources import (
    ANODE_JET_ENERGY_CONVENTIONS,
    CATHODE_JET_ENERGY_CONVENTIONS,
)

#: Implemented operator-splitting compositions.
OPERATOR_SPLITTINGS = ("lie", "strang")


def validate_far_end_configuration(input_dict, flags):
    """Refuse, in ONE message, everything a mirror far end cannot carry.

    Returns ``None``. Under ``far_end = "end_wall"`` it reads nothing else and
    returns at once. Under ``far_end = "mirror"`` it collects every setting
    that presumes the end wall or cannot yet run at a mirror face and raises a
    single ``ValueError`` listing the complete incompatible set with the
    value each would have to take:

    * ``TwinCathode`` -- a different far end;
    * ``heating_anomalous_transport = "plateau_multigroup"`` together with
      ``cathode_coupling`` under ``anode_tail_booking = "lagged_current"`` --
      the walked tail with the lagged absolute tail current. The deposition
      module turns its walkers round at the mirror plane and reflects them at
      the cathode sheath, and the anode mesh is what removes them. The lagged
      booking counts the primary's full flux at the anode plane on top of the
      walkers its drag launched upstream of it, so at breakdown the anode
      books more fast electrons than the cathode emits, its sheath drop is
      driven far above the plateau and repels every walker, and nothing
      removes them: the walkers stay trapped between the two faces and the
      leg-cap residual (10-46 % of the tail power at 64 legs) raises the
      run-time residual bound. ``anode_tail_booking = "emission_fraction"``
      books the anode's direct collection per emitted electron and the
      walked tail constructs under it. The cathode circuit itself -- the CSDA
      primary, which turns round at the plane, and the ``"local"`` anomalous
      heating -- runs at a mirror;
    * ``neutral_momentum`` -- the neutral wind's wall-momentum sink at the far
      face treats it as a wall;
    * ``neutral_energy`` -- the end-face energy accommodation and the hot
      channel's end-plane landing treat the far face as a wall;
    * ``neutral_kinetic_dvm_end_wall_jet`` -- the kinetic closure's
      energetic end wall return, and a mirror has no end wall;
    * ``neutral_kinetic_dvm_annulus_flights = "bounded_chord"`` -- that
      annulus is carried as flights rather than marched, and returns its
      end-plane exits through the lagged end buffer a tick later, so it
      cannot reflect specularly at the mirror plane inside the march the
      way the marched ``"rates"`` annulus does (``neutral_model =
      "kinetic_dvm"`` itself runs at a mirror, its far plane specular);
    * ``S_pump_R != 0`` -- the right pump sits on the end wall cell, which a
      half column does not have;
    * a non-default ``end_wall_length_cm`` -- the half column has no end wall
      cell to give that length to.

    An invalid ``far_end`` value raises naming the accepted set.
    """
    if not _far_end_is_mirror(input_dict):
        return
    conflicts = []
    for flag in (
        "TwinCathode",
        "neutral_momentum",
        "neutral_energy",
    ):
        if bool(flags.get(flag)):
            conflicts.append(f"{flag}=False (got True)")
    if bool(input_dict.get("neutral_kinetic_dvm_end_wall_jet")):
        conflicts.append(
            "neutral_kinetic_dvm_end_wall_jet=False (got True: a mirror has "
            "no end wall to return from)"
        )
    flights = input_dict.get("neutral_kinetic_dvm_annulus_flights")
    if flights == "bounded_chord":
        conflicts.append(
            "neutral_kinetic_dvm_annulus_flights='rates' "
            "(got 'bounded_chord': its annulus is flown, not marched, and "
            "returns its end-plane exits through the lagged end buffer a "
            "tick later, so it cannot reflect specularly at the mirror "
            "plane)"
        )
    booking = input_dict.get("anode_tail_booking", "lagged_current")
    if (
        bool(flags.get("cathode_coupling"))
        and input_dict.get("heating_anomalous_transport")
        == "plateau_multigroup"
        and booking != "emission_fraction"
    ):
        conflicts.append(
            f"anode_tail_booking='emission_fraction' (got {booking!r} with "
            "heating_anomalous_transport='plateau_multigroup': the lagged "
            "booking counts the primary's full flux at the anode plane on "
            "top of the walkers launched upstream of it, so the anode books "
            "more fast electrons than the cathode emits, its sheath then "
            "repels every walker, the "
            "walkers stay trapped between the two faces, and the leg-cap "
            "residual (10-46 % of the tail power at 64 legs) raises the "
            "run-time residual bound)"
        )
    S_pump_R = float(input_dict.get("S_pump_R", 0.0))
    if S_pump_R != 0.0:
        conflicts.append(f"S_pump_R=0.0 (got {S_pump_R!r})")
    end_wall_default = geometry_defaults()["end_wall_length_cm"]
    end_wall_length = input_dict.get("end_wall_length_cm", end_wall_default)
    if end_wall_length != end_wall_default:
        conflicts.append(
            f"end_wall_length_cm={end_wall_default!r} (got "
            f"{end_wall_length!r})"
        )
    if conflicts:
        raise ValueError(
            "far_end='mirror' ends the column in a mirror face at Lm/2, with "
            "no end wall; this configuration sets keys that presume the end "
            "wall or cannot yet run at a mirror. Incompatible with "
            "far_end='mirror' (the complete set): TwinCathode, "
            "heating_anomalous_transport='plateau_multigroup' with "
            "cathode_coupling under anode_tail_booking='lagged_current', "
            "neutral_momentum, neutral_energy, "
            "neutral_kinetic_dvm_end_wall_jet, "
            "neutral_kinetic_dvm_annulus_flights='bounded_chord', "
            "S_pump_R != 0, end_wall_length_cm != "
            f"{end_wall_default!r}. Set: " + "; ".join(conflicts)
        )


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
    # The happy path is the overwhelming majority of the ~10^6 calls a
    # thousand steps make, and it needs the PREDICATE, not the index list:
    # answer it with a reduction over the mask already in hand rather than
    # materializing an empty index array first.
    if not mask.any():
        return None
    bad = np.flatnonzero(mask)
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
):
    """Reject R1-audited controls that would otherwise be silent no-ops."""
    # "neutral" is the partner of the En ionization sink, which debits the
    # neutral energy field.
    ti_birth = input_dict.get("Ti_birth_ionization")
    if isinstance(ti_birth, str):
        ti_birth_ok = ti_birth == "neutral"
    else:
        try:
            ti_birth_numeric = float(ti_birth)
        except (TypeError, ValueError):
            ti_birth_numeric = np.nan
        ti_birth_ok = (
            np.isfinite(ti_birth_numeric) and ti_birth_numeric >= 0.0
        )
    if not ti_birth_ok:
        raise ValueError(
            "Ti_birth_ionization must be 'neutral', or a finite "
            f"non-negative numeric eV value (got {ti_birth!r})"
        )
    if flags.get("Plasma", True):
        for initial_name, floor_name in (
            ("Te0", "Te_floor"),
            ("Ti0", "Ti_floor"),
        ):
            initial = float(input_dict[initial_name])
            floor = float(input_dict[floor_name])
            if not initial > floor:
                raise ValueError(
                    f"{initial_name} must be strictly greater than "
                    f"{floor_name}: raw-stage validation rejects a state "
                    f"at its floor (got {initial} <= {floor})"
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


def resolve_energy_exchange_rate_fraction(input_dict):
    """Return the exchange RATE bound's fraction, or None when unarmed.

    ``energy_exchange_rate_fraction`` is the ``c`` in ``dt <= c / nu_eq,max``
    that bounds the explicit electron-ion energy exchange by its RELAXATION
    RATE (``core.timestep.energy_exchange_rate_timestep``). ``None`` leaves
    the bound withdrawn. Anything else must be a real, finite number in
    ``(0, 1]``: zero or negative would stop the run dead, and above 1 puts the
    difference variable's ``z = -2 c`` below -2, outside SSPRK2's real-axis
    stability interval -- a fraction that arms this bound and is unstable at
    its own limit is the opposite of what arming it is for.
    """
    raw = input_dict.get("energy_exchange_rate_fraction", None)
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(
            "energy_exchange_rate_fraction (the electron-ion exchange rate "
            f"bound's fraction of 1/nu_eq) must be a real number or None "
            f"(got {raw!r})"
        )
    fraction = float(raw)
    if not np.isfinite(fraction) or fraction <= 0.0 or fraction > 1.0:
        raise ValueError(
            "energy_exchange_rate_fraction (the electron-ion exchange rate "
            f"bound's fraction of 1/nu_eq) must be finite and in (0, 1] "
            f"(got {raw!r}); above 1 the difference variable's z = -2 c falls "
            "outside SSPRK2's real-axis stability interval. Use None to leave "
            "the rate bound withdrawn"
        )
    return fraction


#: The accepted values of ``initial_neutral_state``, and what each one arms:
#: ``(equilibrate, launch, profile)``. ``equilibrate`` runs the pre-run
#: puff/off accumulation in ``start_simulation()``; ``launch`` proceeds into
#: the plasma run after it; ``profile`` starts from the per-cell
#: ``nn0_profile``. An equilibrated seed and a shaped profile would each
#: overwrite the other, so no value arms both.
INITIAL_NEUTRAL_STATES = {
    "equilibrate": (True, True, False),
    "equilibrate_only": (True, False, False),
    "fill": (False, False, False),
    "profile": (False, False, True),
}


def resolve_initial_neutral_state(input_dict):
    """Return ``(value, equilibrate, launch, profile)`` for the configuration.

    Raises ``ValueError`` naming the accepted values for any other value.
    """
    value = input_dict.get("initial_neutral_state")
    if value not in INITIAL_NEUTRAL_STATES:
        raise ValueError(
            "initial_neutral_state must be one of "
            f"{sorted(INITIAL_NEUTRAL_STATES)} (got {value!r})"
        )
    return (value, *INITIAL_NEUTRAL_STATES[value])


def validate_neutral_seed_cache_config(input_dict, flags):
    """Reject an incoherent cached-neutral-seed configuration (loud, at build).

    ``use_cached_neutral_seed`` replaces the live neutral equilibration with a
    cached seed, so it requires the equilibration pipeline to be selected
    (``initial_neutral_state = "equilibrate"``) and a cache path. A missing
    path or a contradictory selection would otherwise be a silent no-op.
    """
    if not flags.get("use_cached_neutral_seed", False):
        return
    problems = []
    if input_dict.get("initial_neutral_state") != "equilibrate":
        problems.append(
            "initial_neutral_state must be 'equilibrate' (the cache seeds "
            "that pipeline, and nothing is launched to seed otherwise)"
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
    """Refuse square-waveform edge timings and feed-pipe dimensions the puff
    cannot use: a non-positive edge width, a negative opening center or
    closing lag, and a feed pipe whose bore or length is not finite and
    positive or whose aspect ratio the long-tube angular law has no branch
    for.

    The refusals that need the MESH -- a port off the grid, a plasma column
    that is not inside the vessel wall -- are raised by the row derivation
    itself, which the solver runs once at construction for that reason.
    """
    for key in ("gas_puff_rise_width_s",):
        width = float(input_dict.get(key, 5.0e-4))
        if width <= 0.0:
            raise ValueError(f"{key} must be positive (got {width})")
    for key in ("gas_puff_rise_center_s", "gas_puff_close_lag_s"):
        value = float(input_dict.get(key, 5.0e-4))
        if value < 0.0:
            raise ValueError(f"{key} must be >= 0 (got {value})")
    pipe = {}
    for key in ("gas_puff_orifice_id_cm", "gas_puff_orifice_length_cm"):
        value = input_dict.get(key)
        if value is None or not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(
                f"{key} must be finite and positive (got {value})"
            )
        pipe[key] = float(value)
    bore = pipe["gas_puff_orifice_id_cm"]
    length = pipe["gas_puff_orifice_length_cm"]
    if length / bore < 4.0 / 3.0:
        raise ValueError(
            "gas_puff_orifice_length_cm / gas_puff_orifice_id_cm must be "
            f">= 4/3 (got {length} / {bore} = {length / bore}): the beaming "
            "law is a LONG-tube result whose end-effect prescription inverts "
            "below that ratio, and it has no short-tube branch"
        )


def resolve_jet_arming_criterion(input_dict):
    """Validate and RESOLVE the cathode-jet arming criterion.

    Returns ``(arm_A, disarm_A, active)``. ``active`` is the PRESENCE GATE:
    it is ``False`` for the inert declaration ``arm == 0``, and with it false
    the latch is never constructed, never evaluated, and the cathode jets keep
    the always-live behaviour that predates these keys.

    The criterion covers the two CATHODE channels (the fluid
    ``cathode_neutral_jet`` and the DVM ``neutral_kinetic_dvm_cathode_jet``)
    from ONE latch. The anode jets are driven by the anode-collected current
    rather than by the cathode's booked ``I_i`` and are outside it.

    Raises ``ValueError`` at construction on a non-finite or negative
    threshold, on a disarm threshold declared without an arm threshold, and on
    a band that is not ``0 <= disarm < arm``.
    """
    arm = float(input_dict.get("neutral_jet_arm_current_A", 0.0))
    disarm = float(input_dict.get("neutral_jet_disarm_current_A", 0.0))
    if not (np.isfinite(arm) and np.isfinite(disarm)):
        raise ValueError(
            "neutral_jet_arm_current_A and neutral_jet_disarm_current_A must "
            "be finite currents in amperes (got "
            f"arm={arm!r}, disarm={disarm!r})"
        )
    if arm < 0.0 or disarm < 0.0:
        raise ValueError(
            "neutral_jet_arm_current_A and neutral_jet_disarm_current_A are "
            "ion-current thresholds in amperes and cannot be negative (got "
            f"arm={arm}, disarm={disarm}). Accepted: arm = 0 for no arming "
            "criterion, or 0 <= disarm < arm"
        )
    if arm == 0.0:
        if disarm != 0.0:
            raise ValueError(
                "neutral_jet_disarm_current_A="
                f"{disarm} was declared while neutral_jet_arm_current_A is 0, "
                "which declares NO arming criterion: there is no latch for a "
                "disarm threshold to describe, so the value would sit inert "
                "and silently do nothing. Accepted: set "
                "neutral_jet_arm_current_A > 0 to arm the criterion, or leave "
                "both at 0"
            )
        return 0.0, 0.0, False
    if not disarm < arm:
        raise ValueError(
            "the cathode-jet arming criterion is a LATCHED HYSTERESIS and "
            "requires 0 <= neutral_jet_disarm_current_A < "
            f"neutral_jet_arm_current_A (got disarm={disarm}, arm={arm}). "
            "With disarm >= arm the band is empty or inverted and the latch "
            "would chatter on every step that crosses it, which is the one "
            "thing the hysteresis exists to prevent"
        )
    return arm, disarm, True


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
            "cathode_neutral_jet requires an absorbing cathode face: "
            "the jet rides the "
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


def refuse_cathode_backscatter_double_book(input_dict):
    """Raise when both cathode backscatter books are armed at once.

    ``neutral_kinetic_dvm_cathode_jet`` and ``cathode_jet_surface_debit``
    are two independent debits of the SAME quantity -- the ``R_E`` share of
    the ion bombardment energy the cathode collects. The fluid arm takes it
    off the surface as a retention factor on ``P_cathode_i``; the DVM arm
    takes it off as its own named ledger row, from the counted particles it
    hands the kinetic gas. Armed together the surface pays twice for one
    backscatter.

    Today the pair is unreachable through the model-family resolver, and the
    reason is DIRECT rather than inherited: ``cathode_jet_surface_debit`` is
    itself a member of ``KINETIC_DVM_INCOMPATIBLE_DEFAULTS`` with required
    value ``False``, and the resolver runs BEFORE the guards, so on any
    ``neutral_model = "kinetic_dvm"`` config it has already set the debit to
    ``False`` by the time this is asked. (The ``neutral_momentum`` /
    ``cathode_neutral_jet`` chain is that member's WHY string, not the
    mechanism -- the guard is not reached through it.) The refusal is in
    fact unconditional here: the config template ships the debit ``True``
    and the resolver requires ``False``, and the resolver reads "explicitly
    set" as "differs from the template", so neither of a bool's two values
    presents itself as a caller override -- both leave this looking at
    ``False``. Measured, not reasoned:
    ``scripts/dacc_pairing_mechanism_probe.py`` (at commit 48be9a4, retired
    2026-09-03).

    That is exactly why the guard is written as its own statement about the
    PAIR rather than left implicit in a prerequisite chain: relaxing any part
    of that resolver membership must not silently arm both books.
    """
    if not bool(input_dict.get("neutral_kinetic_dvm_cathode_jet", False)):
        return
    if not bool(input_dict.get("cathode_jet_surface_debit", False)):
        return
    raise ValueError(
        "neutral_kinetic_dvm_cathode_jet and cathode_jet_surface_debit both "
        "debit the cathode surface by the R_E share of the ion bombardment "
        "energy, and they are separate books: the fluid arm withholds it as "
        "a retention factor on P_cathode_i, the DVM arm withholds it as its "
        "own named 'backscatter' ledger row against the particles it counted. "
        "Armed together the surface pays for the same backscatter twice. "
        "Accepted: neutral_kinetic_dvm_cathode_jet=True with "
        "cathode_jet_surface_debit=False (the kinetic arm owns the recycle), "
        "or the fluid pair with neutral_kinetic_dvm_cathode_jet=False"
    )


def refuse_dvm_cathode_jet_without_cathode_coupling(input_dict, flags):
    """Raise when the DVM cathode jet is armed with no cathode solve behind it.

    The channel launches backscattered atoms at the energy an ion arrives
    with, ``phi_c + Te/2`` per collected ion. ``phi_c`` is the CATHODE
    SOLVE's sheath potential; without the ``cathode_coupling`` flag there is
    no such solve anywhere in the run, so the incident energy collapses to
    the presheath ``Te/2`` alone for the whole run and the channel silently
    stops being the energetic recycle it was armed to be. That is a configuration
    with no physical reading rather than a degraded one: an unconfigured
    cathode is not a cathode at zero sheath drop.

    The Te/2-only launch remains REACHABLE, and deliberately so -- a configured
    run whose cathode solve has not started, or whose solve returned a
    non-finite ``phi_c``, still books the ions that arrive with what the
    plasma gave them. What this refuses is the one corner where that reading
    would hold for an ENTIRE run because no solve was ever configured.
    """
    if not bool(input_dict.get("neutral_kinetic_dvm_cathode_jet", False)):
        return
    if bool(flags.get("cathode_coupling", False)):
        return
    raise ValueError(
        "neutral_kinetic_dvm_cathode_jet launches the cathode recycle at the "
        "incident ion energy phi_c + Te/2, and phi_c comes from the cathode "
        "solve the cathode_coupling flag configures. With cathode_coupling "
        "off there is no solve for the whole run, so every backscattered "
        "atom would launch at the presheath Te/2 alone -- the channel would "
        "be "
        "armed and silently carry no sheath energy. Accepted: "
        "neutral_kinetic_dvm_cathode_jet=True with cathode_coupling=True, "
        "or neutral_kinetic_dvm_cathode_jet=False"
    )


def refuse_anode_backscatter_double_book(input_dict):
    """Raise when both anode backscatter re-emissions are armed at once.

    ``neutral_kinetic_dvm_anode_jet`` and the fluid ``anode_neutral_jet`` are
    two independent directed re-emissions of the SAME collected stream: the
    ions the anode mesh neutralizes. The fluid arm launches the ``R_N`` share
    as a momentum source on ``M_n``; the DVM arm launches it as a directed
    volume birth on the velocity grid and books the energy that left with it
    against its own anode energy ledger. Armed together the mesh re-emits one
    backscatter twice, once into each representation of the neutral gas.

    Today the pair is unreachable through the model-family resolver:
    ``anode_neutral_jet`` is M_n momentum physics, it is its own member of
    ``KINETIC_DVM_INCOMPATIBLE_DEFAULTS`` at required value ``False``, and the
    resolver therefore either clears it or refuses it -- naming the whole
    member set -- before this is asked. That is exactly why the guard is
    written as its own statement about the PAIR rather than left implicit in a
    prerequisite chain: relaxing that membership must not silently arm both
    re-emissions.
    """
    if not bool(input_dict.get("neutral_kinetic_dvm_anode_jet", False)):
        return
    if not bool(input_dict.get("anode_neutral_jet", False)):
        return
    raise ValueError(
        "neutral_kinetic_dvm_anode_jet and anode_neutral_jet both re-emit the "
        "R_N share of the anode mesh's collected stream as a DIRECTED "
        "backscatter, and they are separate books: the fluid arm launches it "
        "as a momentum source on M_n, the DVM arm as a directed volume birth "
        "on the velocity grid with its own anode energy ledger row. Armed "
        "together the mesh re-emits one backscatter twice. Accepted: "
        "neutral_kinetic_dvm_anode_jet=True with anode_neutral_jet=False (the "
        "kinetic arm owns the mesh recycle), or the fluid jet with "
        "neutral_kinetic_dvm_anode_jet=False"
    )


def refuse_dvm_anode_jet_without_cathode_coupling(input_dict, flags):
    """Raise when the DVM anode jet is armed with no cathode solve behind it.

    The channel launches backscattered atoms at the energy an ion arrives
    with, ``phi_a + Ti`` per collected ion. ``phi_a`` is the anode sheath
    potential of the CATHODE SOLVE -- the same solve the fluid anode jet reads
    it from, since the cathode/anode/bank system is solved as one -- so
    without the ``cathode_coupling`` flag there is no such solve anywhere in
    the run, the incident energy collapses to the thermal ``Ti`` alone for the
    whole run, and the channel silently stops being the energetic recycle it
    was armed to be. That is a configuration with no physical reading rather
    than a degraded one.

    The Ti-only launch remains REACHABLE, and deliberately so -- a configured
    run whose cathode solve has not started, or whose solve returned a
    non-finite ``phi_a``, still books the ions that arrive with what the
    plasma gave them. What this refuses is the one corner where that reading
    would hold for an ENTIRE run because no solve was ever configured.
    """
    if not bool(input_dict.get("neutral_kinetic_dvm_anode_jet", False)):
        return
    if bool(flags.get("cathode_coupling", False)):
        return
    raise ValueError(
        "neutral_kinetic_dvm_anode_jet launches the anode-mesh recycle at the "
        "incident ion energy phi_a + Ti, and phi_a comes from the cathode "
        "solve the cathode_coupling flag configures -- the cathode, anode and "
        "bank are one system and one solve. With cathode_coupling off there "
        "is no solve for the whole run, so every backscattered atom would "
        "launch at the thermal Ti alone -- the channel would be armed and "
        "silently carry no sheath energy. Accepted: "
        "neutral_kinetic_dvm_anode_jet=True with cathode_coupling=True, or "
        "neutral_kinetic_dvm_anode_jet=False"
    )


def refuse_te_floor_above_adas_table_edge(input_dict):
    """Raise when ``Te_floor`` sits at or above the ADF11 low-Te grid edge.

    The bundled He adf11 tables are two-dimensional in ``(ne, Te)`` -- there
    is no ion-temperature axis, so ``Ti_floor`` has nothing to be ordered
    against and is out of scope here. On the Te axis the lookup CLAMPS: below
    the grid's first temperature every coefficient is held at its edge value
    (``_interp_coords`` clips the log coordinate to the grid), so the whole
    sub-edge region is a plateau rather than a rate curve.

    The floor is what makes that plateau harmless: with ``Te_floor`` strictly
    below the edge the clamped band is a band the state can enter and cool
    through, and the saved ``atomic_rate_domain`` ledger reports how much of
    the active plasma sits in it. A floor AT or ABOVE the edge inverts that:
    no cell can ever be recovered below the edge, the ledger's
    ``active_cell_fraction_below`` is identically zero by construction rather
    than by physics, and the standing statement that the floor sits below the
    table edge becomes false while every rate silently reads its edge value.

    The check is INDEPENDENT of ``adas_low_te_extension``. That key extends
    only ``acd`` and ``prb1`` below the edge, by the analytic recombination
    shape ratio, and it does not move the edge itself -- it reads the same grid
    edge and rescales beneath it. ``scd`` (ionization) and both ``plt`` line
    powers still clamp there either way, so the ordering claim is owed on the
    extended package exactly as it is on the clamped one.

    The edge is READ from the loaded table (``he_rate_temperature_range_eV``,
    the same source the ``atomic_rate_domain`` writer reads) rather than
    written down, so this refusal cannot drift from the bundled data.
    """
    te_edge_eV, _ = he_rate_temperature_range_eV()
    te_floor_eV = float(input_dict["Te_floor"])
    if te_floor_eV < te_edge_eV:
        return
    raise ValueError(
        f"Te_floor={te_floor_eV!r} eV sits at or above the bundled He adf11 "
        f"low-Te grid edge {te_edge_eV!r} eV, and the atomic rates read that "
        "grid. Below the edge every adf11 coefficient is CLAMPED "
        "to its edge value, so the floor is what keeps the clamped band a "
        "band the plasma cools through instead of the only band it occupies: "
        "with the floor at or above the edge no recovered Te can ever lie "
        "below the edge, the atomic_rate_domain ledger's "
        "active_cell_fraction_below is zero by construction rather than by "
        "physics, and the documented ordering (the floor sits below the "
        "table edge) is false. Accepted: Te_floor strictly below "
        f"{te_edge_eV!r} eV. adas_low_te_extension does not lift this: "
        "it rescales acd and prb1 beneath the same edge without moving it, "
        "and scd and both plt tables clamp there either way"
    )

def resolve_parallel_momentum_sink(input_dict, *, geometry):
    """Validate and RESOLVE the imposed parallel momentum sink.

    Every failure here is a construction-time ``ValueError``. With the sink
    off both of its numbers must sit at their ``None`` defaults, so a run
    that configures a response-map arm and forgets to arm it is loud rather
    than silently unforced; with it on both are required, and the axial
    position must land inside the plasma column so the term can neither
    describe a region that is not there nor reach no cell at all.

    Returns the resolved instrument -- the rate and the boolean per-cell
    mask of the column cells at or beyond the position -- or ``None`` when
    the sink is off, which is the presence gate every consumer reads.
    """
    defaults = parallel_momentum_sink_defaults()
    values = {
        name: input_dict.get(name, default)
        for name, default in defaults.items()
    }
    enabled = values["parallel_momentum_sink"]
    if not isinstance(enabled, bool):
        raise ValueError(
            "parallel_momentum_sink must be a bool (got "
            f"{enabled!r}); it is the arming gate of a response-map "
            "instrument, not a rate"
        )
    rate = values["parallel_momentum_sink_rate_s"]
    z_start = values["parallel_momentum_sink_z_start_cm"]
    if not enabled:
        configured = sorted(
            name
            for name in (
                "parallel_momentum_sink_rate_s",
                "parallel_momentum_sink_z_start_cm",
            )
            if values[name] is not None
        )
        if configured:
            raise ValueError(
                f"the parallel-momentum-sink parameters {configured} were "
                "configured without parallel_momentum_sink, where they are "
                "inert; arm the sink or drop the parameters"
            )
        return None
    if rate is None:
        raise ValueError(
            "parallel_momentum_sink requires parallel_momentum_sink_rate_s "
            "(the imposed damping rate nu_add [s^-1]). There is no default: "
            "this term has no physical owner, so the rate IS the hypothesis "
            "the arm states and nothing may supply one for it"
        )
    rate = float(rate)
    if not (math.isfinite(rate) and rate > 0.0):
        raise ValueError(
            "parallel_momentum_sink_rate_s must be finite and > 0 (got "
            f"{values['parallel_momentum_sink_rate_s']!r}); a zero rate is "
            "an unarmed sink, which is parallel_momentum_sink = false"
        )
    if z_start is None:
        raise ValueError(
            "parallel_momentum_sink requires "
            "parallel_momentum_sink_z_start_cm (the axial position [cm] at "
            "and beyond which the sink acts). There is no default: which "
            "part of the column sheds the momentum is the other half of the "
            "hypothesis"
        )
    z_start = float(z_start)
    if not math.isfinite(z_start):
        raise ValueError(
            "parallel_momentum_sink_z_start_cm must be finite (got "
            f"{values['parallel_momentum_sink_z_start_cm']!r})"
        )
    active = np.asarray(geometry.plasma_active, dtype=bool)
    z_cm = np.asarray(geometry.z_cm, dtype=float)
    column = z_cm[active]
    if column.size == 0:
        raise ValueError(
            "parallel_momentum_sink needs a plasma column to act on and "
            "this geometry has no plasma-active cell"
        )
    lo, hi = float(column.min()), float(column.max())
    if not (lo <= z_start <= hi):
        raise ValueError(
            "parallel_momentum_sink_z_start_cm must lie within the plasma "
            f"column's axial extent [{lo!r}, {hi!r}] cm (got {z_start!r}); "
            "below it the sink is not a statement about a region of the "
            "column, and above it the sink reaches no cell and would be "
            "silently inert"
        )
    cells = active & (z_cm >= z_start)
    return SimpleNamespace(rate_s=rate, z_start_cm=z_start, cells=cells)


def validate_raw_stage(y, stage, unpack):
    """Reject non-finite/negative raw candidates before floor clipping.

    Non-finiteness is decided ONCE, on the packed candidate: ``unpack_state``
    returns ``.copy()`` of the rows of ``y``, so every unpacked field is a
    bitwise copy of a value this scan has already seen, and a per-field rescan
    of a packed vector that passed cannot find anything. (It never could: the
    packed scan RAISES on the first bad value, so a per-field scan below it was
    only ever reachable if unpacking could invent one.) The negative-value
    scans below are a different predicate and are not covered by it.
    """
    packed_summary = _bad_array_summary(y)
    if packed_summary is not None:
        raise _RawStageError(
            y,
            stage,
            "nonfinite_state",
            {"stage": stage, "fields": {"packed_y": packed_summary}},
        )
    state = unpack(y)
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
