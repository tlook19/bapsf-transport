from dataclasses import dataclass

import numpy as np

from cablp.funcs._cross import phelps_momentum_transfer_rate_cm3_s
from cablp.vars._cons import ev_to_erg

from ..physics.cathode import circuit_relaxation_timestep
from ..physics.conduction import heat_conduction_timestep_bound
from ..physics.energy import (
    electron_cooling_rhs,
    electron_ion_exchange_rhs,
    ion_charge_exchange_rhs,
)
from ..physics.flux import ion_sound_speed, plasma_flux_rhs, plasma_wave_speed
from ..physics.neutrals import (
    NEUTRAL_GAMMA,
    neutral_exchange_rhs,
    neutral_source_sink_rhs,
)
from ..physics.reactions import reaction_rhs
from ..physics.sources import (
    ion_neutral_collision_frequency,
    neutral_energy_volume_ratio,
    neutral_temperature_eV,
    neutral_wind_velocity,
)
from .state import derive_state

#: Fraction [dimensionless] of the ion-neutral drag damping time 1/nu_in the
#: accepted step may take, so the explicit damping cannot overshoot: the
#: forward-Euler factor (1 - dt nu_in) stays at 1/2 or above.
DRAG_DT_FRACTION = 0.5


@dataclass(frozen=True)
class TimestepDiagnostics:
    dt: float
    dt_plasma_cfl: float
    dt_front_density: float
    dt_surface_loss: float
    dt_neutral_exchange: float
    dt_neutral_sources: float
    dt_reactions: float
    dt_energy_exchange: float
    dt_electron_cooling: float
    dt_ion_charge_exchange: float
    dt_heat_conduction: float
    dt_ion_neutral_drag: float
    dt_max: float
    active_constraint: str
    # Defaulted so results written before the neutral wind existed still
    # load; inf whenever the state carries no M_n (the historical case).
    dt_neutral_wind: float = np.inf
    # The current-driven loop's local relaxation bound. Defaulted (and inf)
    # so results written before it existed still load, and inf on every run
    # that does not arm cathode_circuit_voltage_bound -- the candidate is
    # presence-gated on the flag, so an unarmed run's dt sequence cannot move.
    dt_circuit: float = np.inf
    # The dt_min clamp, recorded as a FACT ABOUT THE STEP rather than as a
    # constraint name. ``active_constraint`` always names the bound that
    # actually minimized; ``clamped_to_dt_min`` (0.0/1.0, the float-flag idiom
    # used by the phase_* switches below) says whether that bound asked for
    # less than dt_min and was clamped up to it, and ``dt_raw`` records what it
    # asked for. ``dt_raw == 0.0`` is the drained floor-pinned signature from
    # ``_negative_margin_timestep``: a cell sitting ON a floor while still
    # draining, which is a modelling breakdown and not a timestep request.
    # Defaulted so pre-2026-08-05 results still load (see results/io.py for the
    # label-semantics boundary).
    clamped_to_dt_min: float = 0.0
    dt_raw: float = np.nan
    # The evolved neutral energy's relaxation bound. Defaulted (and inf) so
    # results written before the En field existed still load, and inf on every
    # run whose state carries no En.
    dt_neutral_energy: float = np.inf
    accepted_dt: float = np.nan
    # The dt_min clamp as a fact about the ACCEPTED step, which is a different
    # statement from ``clamped_to_dt_min`` above. That flag is computed from
    # the raw candidate minimum alone; the caps applied after it (dt_growth,
    # t_end, phase_boundary, save_time) and the retry ladder can each carry
    # the accepted dt BELOW dt_min while no candidate asked for less than
    # dt_min, so the two disagree exactly in the case the dt_min lock exists
    # to catch: a grind whose steps are set by the growth ramp re-approaching
    # from a sub-dt_min snap rather than by a physics bound. Set only when the
    # raw minimum is STRICTLY ABOVE dt_min, so it names a step a cap pushed
    # under the floor and never a run configured with dt_max at dt_min.
    # 0.0/1.0, the float-flag idiom; defaulted so pre-existing results load.
    clamped_to_dt_min_accepted: float = 0.0
    step_cap: str = ""
    retry_count: int = 0
    rejection_reason: str = ""
    time: float = np.nan
    phase: str = ""
    phase_cathode_enabled: float = 0.0
    phase_gas_puff_enabled: float = 0.0
    phase_floating: float = 0.0
    # The global dt-refinement instrument's factor, recorded as a fact about
    # the step so an instrumented run can PROVE the scale was applied rather
    # than inferring it from a dt trajectory. 1.0 is the unarmed value and the
    # default, so results written before the instrument existed still load.
    dt_global_scale: float = 1.0


def suggest_timestep(
    state,
    floors,
    ion_mass_g,
    mu,
    geometry,
    neutral_exchange_coeff_cm3_s,
    neutral_source_kwargs=None,
    reaction_kwargs=None,
    energy_exchange_kwargs=None,
    electron_cooling_kwargs=None,
    ion_charge_exchange_kwargs=None,
    heat_conduction_kwargs=None,
    ion_neutral_drag_kwargs=None,
    neutral_energy_kwargs=None,
    circuit_kwargs=None,
    plasma_source_rhs=None,
    source_floor_exempt_rtol=None,
    source_floor_exempt_exit_rtol=None,
    source_floor_exempt_latch=None,
    neutral_rows_superseded=False,
    cfl=0.4,
    density_dt_fraction=0.25,
    neutral_dt_fraction=0.25,
    circuit_dt_fraction=0.25,
    dt_min=1e-12,
    dt_max=1e-6,
    dt_global_scale=1.0,
    include_front=True,
    alpha_front=1.0,
    plasma_active=None,
    active_plasma_topology=False,
    wave_speed="isothermal",
):
    """Return a bounded explicit timestep and diagnostics.

    Every candidate here must be a bound on a rate the accepted step
    ACTUALLY APPLIES. A kinetic neutral arm supersedes whole rows of the
    fluid terms -- it zeroes them and carries them itself -- and a bound
    computed from a superseded row is a PHANTOM: it can name itself
    ``active_constraint`` and set the step while the row it describes is
    identically zero. ``neutral_rows_superseded`` withdraws the candidates
    that read a neutral row an arm has taken over (the pair-exchange and
    puff/pump bounds whole, and the reaction bound's neutral channel), and
    the caller withdraws the ion-transfer bounds by passing no kwargs for
    them. The replacement rows are bounded through ``plasma_source_rhs``,
    which is where the arm's own coupling term belongs.

    ``circuit_kwargs`` is the same kind of bundle for the current-driven
    loop's own ODE (see ``cathode.circuit_relaxation_timestep``). It is the
    one candidate that does not describe a fluid row, and it is
    presence-gated by the caller: ``None`` -- every run that does not arm
    ``cathode_circuit_voltage_bound``, and every phase with no live loop --
    withdraws it to ``inf``, so it cannot move an unarmed run's step.

    ``source_floor_exempt_exit_rtol`` and ``source_floor_exempt_latch`` are
    the ``surface_loss`` bound's optional exemption hysteresis, forwarded
    verbatim to ``plasma_source_timestep`` (which documents them). Both
    ``None`` -- the default -- leaves that bound's floor exemption
    single-threshold and this call free of side effects; with the band armed
    the call ADVANCES the caller's latch, so it is no longer a pure query.

    ``dt_global_scale`` is a measurement instrument, not a bound: it
    multiplies the returned step AFTER every candidate and after the
    dt_min/dt_max clamp (see ``apply_dt_global_scale``), so it refines the
    whole dt trajectory uniformly instead of tightening one channel. It does
    NOT participate in ``active_constraint`` or in ``clamped_to_dt_min``,
    which stay facts about the bounds.
    """
    if dt_min <= 0.0:
        raise ValueError(f"dt_min must be positive (got {dt_min})")
    if dt_max <= 0.0:
        raise ValueError(f"dt_max must be positive (got {dt_max})")
    if dt_min > dt_max:
        raise ValueError(f"dt_min must be <= dt_max (got {dt_min} > {dt_max})")

    dt_candidates = {
        "plasma_cfl": plasma_cfl_timestep(
            state=state,
            floors=floors,
            ion_mass_g=ion_mass_g,
            mu=mu,
            geometry=geometry,
            cfl=cfl,
            plasma_active=plasma_active,
            wave_speed=wave_speed,
        ),
        "front_density": front_density_timestep(
            state=state,
            floors=floors,
            ion_mass_g=ion_mass_g,
            mu=mu,
            geometry=geometry,
            density_dt_fraction=density_dt_fraction,
            include_front=include_front,
            alpha_front=alpha_front,
            plasma_active=plasma_active,
            active_plasma_topology=active_plasma_topology,
        ),
        # Retain the historical diagnostic key while assigning it to the live
        # resolved electrode/source bundle. The old volumetric endpoint loss
        # no longer exists.
        "surface_loss": plasma_source_timestep(
            state=state,
            source_rhs=plasma_source_rhs,
            floors=floors,
            fraction=density_dt_fraction,
            plasma_active=plasma_active,
            floor_exempt_rtol=source_floor_exempt_rtol,
            floor_exempt_exit_rtol=source_floor_exempt_exit_rtol,
            floor_exempt_latch=source_floor_exempt_latch,
        ),
        "neutral_exchange": (
            np.inf
            if neutral_rows_superseded
            else neutral_exchange_timestep(
                state=state,
                geometry=geometry,
                neutral_exchange_coeff_cm3_s=neutral_exchange_coeff_cm3_s,
                neutral_dt_fraction=neutral_dt_fraction,
                floors=floors,
            )
        ),
        "neutral_sources": (
            np.inf
            if neutral_rows_superseded
            else neutral_source_timestep(
                state=state,
                geometry=geometry,
                neutral_source_kwargs=neutral_source_kwargs,
                neutral_dt_fraction=neutral_dt_fraction,
            )
        ),
        "reactions": reaction_timestep(
            state=state,
            floors=floors,
            ion_mass_g=ion_mass_g,
            geometry=geometry,
            reaction_kwargs=reaction_kwargs,
            density_dt_fraction=density_dt_fraction,
            plasma_active=plasma_active,
            include_neutral_channel=not neutral_rows_superseded,
        ),
        "energy_exchange": energy_exchange_timestep(
            state=state,
            floors=floors,
            ion_mass_g=ion_mass_g,
            mu=mu,
            energy_exchange_kwargs=energy_exchange_kwargs,
            density_dt_fraction=density_dt_fraction,
            plasma_active=plasma_active,
        ),
        "electron_cooling": electron_cooling_timestep(
            state=state,
            floors=floors,
            ion_mass_g=ion_mass_g,
            electron_cooling_kwargs=electron_cooling_kwargs,
            density_dt_fraction=density_dt_fraction,
            plasma_active=plasma_active,
        ),
        "ion_charge_exchange": ion_charge_exchange_timestep(
            state=state,
            floors=floors,
            ion_mass_g=ion_mass_g,
            ion_charge_exchange_kwargs=ion_charge_exchange_kwargs,
            density_dt_fraction=density_dt_fraction,
            plasma_active=plasma_active,
        ),
        "heat_conduction": heat_conduction_timestep(
            state=state,
            floors=floors,
            ion_mass_g=ion_mass_g,
            mu=mu,
            geometry=geometry,
            heat_conduction_kwargs=heat_conduction_kwargs,
            plasma_active=plasma_active,
        ),
        "ion_neutral_drag": ion_neutral_drag_timestep(
            state=state,
            floors=floors,
            ion_mass_g=ion_mass_g,
            ion_neutral_drag_kwargs=ion_neutral_drag_kwargs,
            plasma_active=plasma_active,
        ),
        "neutral_wind": neutral_wind_timestep(
            state=state,
            floors=floors,
            ion_mass_g=ion_mass_g,
            geometry=geometry,
            cfl=cfl,
        ),
        "neutral_energy": neutral_energy_timestep(
            state=state,
            floors=floors,
            ion_mass_g=ion_mass_g,
            geometry=geometry,
            neutral_energy_kwargs=neutral_energy_kwargs,
            neutral_dt_fraction=neutral_dt_fraction,
        ),
        "circuit": circuit_timestep(
            circuit_kwargs=circuit_kwargs,
            circuit_dt_fraction=circuit_dt_fraction,
        ),
        "dt_max": float(dt_max),
    }
    active_constraint, raw_dt = min(dt_candidates.items(), key=lambda item: item[1])
    dt = min(max(raw_dt, dt_min), dt_max)
    # active_constraint keeps naming the bound that actually minimized. The
    # clamp is recorded as a separate fact, because relabelling it "dt_min"
    # hid the true bound exactly when a caller most needs it: a run pinned at
    # dt_min reported only that it was pinned, never by what.
    clamped_to_dt_min = dt == dt_min and raw_dt < dt_min
    dt = apply_dt_global_scale(dt, dt_global_scale)
    return TimestepDiagnostics(
        dt=float(dt),
        dt_plasma_cfl=float(dt_candidates["plasma_cfl"]),
        dt_front_density=float(dt_candidates["front_density"]),
        dt_surface_loss=float(dt_candidates["surface_loss"]),
        dt_neutral_exchange=float(dt_candidates["neutral_exchange"]),
        dt_neutral_sources=float(dt_candidates["neutral_sources"]),
        dt_reactions=float(dt_candidates["reactions"]),
        dt_energy_exchange=float(dt_candidates["energy_exchange"]),
        dt_electron_cooling=float(dt_candidates["electron_cooling"]),
        dt_ion_charge_exchange=float(dt_candidates["ion_charge_exchange"]),
        dt_heat_conduction=float(dt_candidates["heat_conduction"]),
        dt_ion_neutral_drag=float(dt_candidates["ion_neutral_drag"]),
        dt_max=float(dt_max),
        active_constraint=active_constraint,
        dt_neutral_wind=float(dt_candidates["neutral_wind"]),
        dt_circuit=float(dt_candidates["circuit"]),
        clamped_to_dt_min=float(clamped_to_dt_min),
        dt_raw=float(raw_dt),
        dt_neutral_energy=float(dt_candidates["neutral_energy"]),
        dt_global_scale=float(dt_global_scale),
    )


def apply_dt_global_scale(dt, dt_global_scale):
    """Return the final accepted ``dt`` scaled by the refinement instrument.

    The factor is applied AFTER every timestep candidate and after the
    dt_min/dt_max clamp, so the scaled step is deliberately allowed below
    ``dt_min``: the instrument answers "what does this run do at half the
    step it chose", which a re-clamp would silently refuse to ask. At the
    unarmed value 1.0 the multiply is SKIPPED rather than performed, so an
    unarmed run's dt arithmetic is bit-identical to one predating the key.
    """
    if dt_global_scale == 1.0:
        return dt
    return dt * dt_global_scale


def circuit_timestep(circuit_kwargs=None, circuit_dt_fraction=0.25):
    """Bound the step by the current-driven loop's local relaxation time.

    ``None`` withdraws the candidate (``inf``): the bound is presence-gated
    on ``cathode_circuit_voltage_bound`` and on a live loop, so an unarmed
    run never evaluates it. Otherwise the bundle is forwarded to
    ``cathode.circuit_relaxation_timestep``, which owns the physics.
    """
    if circuit_kwargs is None:
        return np.inf
    return circuit_relaxation_timestep(
        fraction=circuit_dt_fraction, **circuit_kwargs
    )


def plasma_cfl_timestep(
    state, floors, ion_mass_g, mu, geometry, cfl=0.4, plasma_active=None,
    wave_speed="isothermal",
):
    """Return the plasma wave CFL timestep [s]."""
    if cfl <= 0.0:
        raise ValueError(f"cfl must be positive (got {cfl})")
    derived = derive_state(state, floors=floors, ion_mass_g=ion_mass_g)
    cs = plasma_wave_speed(derived.Te, derived.Ti, mu, wave_speed)
    face_speed = 0.5 * (
        np.abs(derived.u[:-1])
        + np.abs(derived.u[1:])
        + cs[:-1]
        + cs[1:]
    )
    if plasma_active is not None:
        active = np.asarray(plasma_active, dtype=bool)
        face_active = (
            active[:-1]
            & active[1:]
            & np.asarray(geometry.plasma_open[1:-1], dtype=bool)
        )
        face_speed = np.where(face_active, face_speed, 0.0)
    return _distance_timestep(geometry.center_distance_cm, face_speed, cfl)


def plasma_source_timestep(
    state,
    source_rhs,
    floors,
    fraction=0.25,
    plasma_active=None,
    floor_exempt_rtol=None,
    floor_exempt_exit_rtol=None,
    floor_exempt_latch=None,
):
    """Bound resolved plasma sources against density/temperature floors.

    Electron and ion margins are formed in the conservative variables,
    including the change in the floor energy when ``n`` changes:

    ``d(E - 3/2 n T_floor)/dt = dE/dt - 3/2 T_floor dn/dt``.

    ``floor_exempt_rtol`` (default ``None`` = historical behavior) is the
    floor-aware drain exemption for the energy channels: a cell whose energy
    margin above its floor is at most ``floor_exempt_rtol`` times the per-cell
    floor energy ``3/2 n T_floor`` is excluded from THIS bound. Rationale: the
    accept-time floor clip resets a floor-pinned cell's margin to float
    residue every step, so a persistent drain re-trips this bound forever and
    pins dt at dt_min while the floor (not this bound) is what actually holds
    the cell. The exemption is one-sided (this drain bound only; every other
    bound still governs the cell) and by default knife-edge (recomputed from
    the current margin each call: any margin above the threshold re-admits the
    cell immediately). The density channel is never exempted.

    ``floor_exempt_exit_rtol`` (default ``None`` = the knife edge above) turns
    that single threshold into a two-threshold band. ``floor_exempt_rtol`` is
    then the ENTRY threshold and this is the RE-ADMISSION threshold, which must
    be the larger of the two; a cell whose margin lies between them keeps
    whichever state it held last, so a cell hovering at its floor cannot flap
    in and out of the exemption from one call to the next. The memory is
    ``floor_exempt_latch``, a caller-owned dict keyed by energy-channel name
    (``"Ee"``/``"Ei"``) holding the per-cell boolean exempt mask; this function
    READS the previous mask and WRITES the new one, so the call is stateful
    exactly while the band is armed and the caller owns the latch's lifetime.
    An absent, empty or wrongly-shaped entry starts the channel with every cell
    un-exempt.
    """
    if source_rhs is None:
        return np.inf
    if fraction <= 0.0:
        raise ValueError(f"fraction must be positive (got {fraction})")
    active = (
        np.ones_like(np.asarray(state.n), dtype=bool)
        if plasma_active is None
        else np.asarray(plasma_active, dtype=bool)
    )
    n = np.asarray(state.n, dtype=float)
    dn = np.asarray(source_rhs.n, dtype=float)
    candidates = [
        _negative_margin_timestep(
            n - float(floors["n"]),
            dn,
            fraction,
            active,
        )
    ]
    for energy_name, floor_name in (("Ee", "Te"), ("Ei", "Ti")):
        energy = np.asarray(getattr(state, energy_name), dtype=float)
        denergy = np.asarray(getattr(source_rhs, energy_name), dtype=float)
        floor_energy_per_particle = (
            1.5 * float(floors[floor_name]) * ev_to_erg
        )
        margin = energy - floor_energy_per_particle * n
        channel_active = active
        if floor_exempt_rtol is not None:
            # Floor-pinned exemption: exclude cells sitting (to within the
            # relative threshold) at the per-cell floor energy. Exempted
            # cells cannot set dt_surface_loss, so an exempted cell is never
            # reported as this bound's active constraint.
            if floor_exempt_exit_rtol is None:
                channel_active = active & (
                    margin
                    > float(floor_exempt_rtol) * floor_energy_per_particle * n
                )
            else:
                # Hysteresis band. Entry keeps the knife-edge threshold;
                # re-admission has to clear the wider one. A cell between the
                # two holds its previous verdict, which is the whole point:
                # the accept-time floor clip perturbs the margin by float
                # residue every step, and a single threshold turns that
                # residue into a per-step exempt/bound alternation.
                was_exempt = None
                if floor_exempt_latch is not None:
                    was_exempt = floor_exempt_latch.get(energy_name)
                if was_exempt is None or np.shape(was_exempt) != margin.shape:
                    was_exempt = np.zeros(margin.shape, dtype=bool)
                exempt = np.where(
                    was_exempt,
                    margin
                    <= float(floor_exempt_exit_rtol)
                    * floor_energy_per_particle
                    * n,
                    margin
                    <= float(floor_exempt_rtol) * floor_energy_per_particle * n,
                )
                if floor_exempt_latch is not None:
                    floor_exempt_latch[energy_name] = exempt
                channel_active = active & ~exempt
        candidates.append(
            _negative_margin_timestep(
                margin,
                denergy - floor_energy_per_particle * dn,
                fraction,
                channel_active,
            )
        )
    return float(min(candidates))


def front_density_timestep(
    state,
    floors,
    ion_mass_g,
    mu,
    geometry,
    density_dt_fraction=0.25,
    include_front=True,
    alpha_front=1.0,
    plasma_active=None,
    active_plasma_topology=False,
):
    """Return a fractional density-change timestep for front filling."""
    if not include_front:
        return np.inf
    if density_dt_fraction <= 0.0:
        raise ValueError(
            f"density_dt_fraction must be positive (got {density_dt_fraction})"
        )
    rhs_with_front = plasma_flux_rhs(
        state=state,
        floors=floors,
        ion_mass_g=ion_mass_g,
        mu=mu,
        geometry=geometry,
        include_front=True,
        alpha_front=alpha_front,
        active_plasma_topology=active_plasma_topology,
    )
    rhs_without_front = plasma_flux_rhs(
        state=state,
        floors=floors,
        ion_mass_g=ion_mass_g,
        mu=mu,
        geometry=geometry,
        include_front=False,
        alpha_front=alpha_front,
        active_plasma_topology=active_plasma_topology,
    )
    dn_front = rhs_with_front.n - rhs_without_front.n
    return _fractional_timestep(
        state.n,
        dn_front,
        density_dt_fraction,
        floors["n"],
        active_mask=plasma_active,
    )


def ion_neutral_drag_timestep(
    state,
    floors,
    ion_mass_g,
    ion_neutral_drag_kwargs=None,
    plasma_active=None,
):
    """Return an explicit-stability timestep for ion-neutral drag damping.

    The drag damps the flow at rate ``nu_in``; the accepted step keeps
    ``dt * max(nu_in)`` below ``DRAG_DT_FRACTION``.
    """
    if ion_neutral_drag_kwargs is None:
        return np.inf
    b_ion_neutral_drag = ion_neutral_drag_kwargs.get("b_ion_neutral_drag", 1.0)
    if b_ion_neutral_drag == 0.0:
        return np.inf
    derived = derive_state(state, floors=floors, ion_mass_g=ion_mass_g)
    nu_in = ion_neutral_collision_frequency(
        nn=state.nn,
        Ti=derived.Ti,
        gas_type=ion_neutral_drag_kwargs.get("gas_type"),
    )
    active = _active_values(nu_in, plasma_active)
    nu_max = (
        float(np.max(np.abs(active))) * abs(float(b_ion_neutral_drag))
        if active.size
        else 0.0
    )
    if nu_max <= 0.0:
        return np.inf
    return DRAG_DT_FRACTION / nu_max


def neutral_wind_timestep(state, floors, ion_mass_g, geometry, cfl=0.4):
    """Return the CFL bound for neutral-wind advection.

    ``cfl * min(dz / |u_n|)`` over cells with a moving wind; infinite when
    the state carries no ``M_n`` (the historical case) or the wind is
    everywhere still. The wind is deeply subsonic in practice (~1e-4 s
    against dt <= 1e-6), so this guard exists for pathological transients,
    not the design point.
    """
    if state.M_n is None:
        return np.inf
    speed = np.abs(
        neutral_wind_velocity(
            state, floors=floors, ion_mass_g=ion_mass_g, geometry=geometry
        )
    )
    if state.M_n_a is not None:
        if state.nn_a is None:
            raise ValueError("M_n_a requires nn_a")
        speed = np.maximum(
            speed,
            np.abs(state.M_n_a)
            / (
                ion_mass_g
                * np.maximum(np.asarray(state.nn_a, dtype=float), floors["nn"])
            ),
        )
    moving = speed > 0.0
    if not np.any(moving):
        return np.inf
    return float(cfl) * float(
        np.min(geometry.length_cm[moving] / speed[moving])
    )


def neutral_energy_timestep(
    state,
    floors,
    ion_mass_g,
    geometry,
    neutral_energy_kwargs=None,
    neutral_dt_fraction=0.25,
):
    """Bound the step by the evolved neutral energy's relaxation and transport.

    ``En`` is driven by two explicit relaxations: the ion-neutral collision
    operator pulls ``Tn`` toward ``Ti`` at ``(n/nn) nu_mt (Vp/V_En)`` (the
    per-NEUTRAL exchange rate -- the collision term's ``nu_mt`` is the rate a
    given ION collides, and the mirror lands in a reservoir of ``nn`` neutrals
    over its own volume), and the wall pulls it toward ``T_wall`` at
    ``alpha_E nu_wall``. The accepted step keeps ``dt`` times their sum below
    ``neutral_dt_fraction``.

    The cold fluid's mini-flux adds a hyperbolic bound on the same candidate:
    once the gas carries its own pressure the neutral signal speed is
    ``|u_n| + c_n`` rather than ``|u_n|`` alone, and the wind bound (which
    predates the pressure) does not see the acoustic part. Folding it in here
    keeps the neutral-energy arm on ONE named candidate instead of adding a
    diagnostic field that older results would not carry.

    ``None`` withdraws the candidate (``inf``): a state with no ``En`` has no
    such rate, so an unarmed run's step cannot move.
    """
    if neutral_energy_kwargs is None or state.En is None:
        return np.inf
    if neutral_dt_fraction <= 0.0:
        raise ValueError(
            f"neutral_dt_fraction must be positive (got {neutral_dt_fraction})"
        )
    derived = derive_state(state, floors=floors, ion_mass_g=ion_mass_g)
    Tn = neutral_temperature_eV(
        state, floors=floors, Tn_eV=neutral_energy_kwargs["Tn_eV"]
    )
    nn = np.maximum(np.asarray(state.nn, dtype=float), floors["nn"])
    nu_mt = nn * phelps_momentum_transfer_rate_cm3_s(
        0.5 * (derived.Ti + Tn), gas_type=neutral_energy_kwargs["gas_type"]
    )
    rate = (
        abs(float(neutral_energy_kwargs["b_ion_neutral_drag"]))
        * np.asarray(state.n, dtype=float)
        * nu_mt
        / nn
        * neutral_energy_volume_ratio(state, geometry)
    )
    wall_rate = neutral_energy_kwargs["wall_rate_1_s"]
    if wall_rate is None:
        vbar_n = np.sqrt(
            8.0
            * float(neutral_energy_kwargs["Tn_fit"])
            * ev_to_erg
            / (np.pi * ion_mass_g)
        )
        wall_rate = vbar_n / np.asarray(geometry.Rm_cm, dtype=float)
    rate = rate + abs(
        float(neutral_energy_kwargs["alpha_E"])
    ) * np.asarray(wall_rate, dtype=float)
    # Hyperbolic part: the cold fluid's own acoustic signal on its own cells.
    u_n = neutral_wind_velocity(
        state, floors=floors, ion_mass_g=ion_mass_g, geometry=geometry
    )
    c_n = np.sqrt(
        NEUTRAL_GAMMA * np.maximum(Tn, 0.0) * ev_to_erg / ion_mass_g
    )
    rate = rate + (np.abs(u_n) + c_n) / np.asarray(
        geometry.length_cm, dtype=float
    )
    rate_max = float(np.max(rate)) if rate.size else 0.0
    if rate_max <= 0.0:
        return np.inf
    return float(neutral_dt_fraction) / rate_max


def neutral_exchange_timestep(
    state,
    geometry,
    neutral_exchange_coeff_cm3_s,
    neutral_dt_fraction=0.25,
    floors=None,
):
    """Return a fractional neutral-density timestep for pair exchange.

    ``floors`` is threaded through only so the exchange term can build its
    donor-energy row on an ``En``-carrying state; this bound reads the density
    row alone, exactly as before.
    """
    if neutral_dt_fraction <= 0.0:
        raise ValueError(
            f"neutral_dt_fraction must be positive (got {neutral_dt_fraction})"
        )
    rhs = neutral_exchange_rhs(
        state=state,
        geometry=geometry,
        exchange_coeff_cm3_s=neutral_exchange_coeff_cm3_s,
        floors=floors,
    )
    return _fractional_timestep(state.nn, rhs.nn, neutral_dt_fraction, 0.0)


def neutral_source_timestep(
    state,
    geometry,
    neutral_source_kwargs=None,
    neutral_dt_fraction=0.25,
):
    """Return a fractional neutral-density timestep for puff/pump terms."""
    if neutral_source_kwargs is None:
        return np.inf
    if neutral_dt_fraction <= 0.0:
        raise ValueError(
            f"neutral_dt_fraction must be positive (got {neutral_dt_fraction})"
        )
    rhs = neutral_source_sink_rhs(
        state=state,
        geometry=geometry,
        **neutral_source_kwargs,
    )
    return _fractional_timestep(state.nn, rhs.nn, neutral_dt_fraction, 0.0)


def reaction_timestep(
    state,
    floors,
    ion_mass_g,
    geometry,
    reaction_kwargs=None,
    density_dt_fraction=0.25,
    plasma_active=None,
    include_neutral_channel=True,
):
    """Return a fractional density timestep for local plasma reactions.

    ``include_neutral_channel`` drops the ``nn`` half of the bound. The
    plasma half stays either way: a kinetic arm supersedes the reactions'
    neutral row but not the ions they birth.
    """
    if reaction_kwargs is None:
        return np.inf
    if density_dt_fraction <= 0.0:
        raise ValueError(
            f"density_dt_fraction must be positive (got {density_dt_fraction})"
        )
    rhs = reaction_rhs(
        state=state,
        floors=floors,
        ion_mass_g=ion_mass_g,
        geometry=geometry,
        **reaction_kwargs,
    )
    plasma_channel = _fractional_timestep(
        state.n,
        rhs.n,
        density_dt_fraction,
        floors["n"],
        active_mask=plasma_active,
    )
    if not include_neutral_channel:
        return plasma_channel
    return min(
        plasma_channel,
        _fractional_timestep(
            state.nn,
            rhs.nn,
            density_dt_fraction,
            0.0,
            active_mask=plasma_active,
        ),
    )


def energy_exchange_timestep(
    state,
    floors,
    ion_mass_g,
    mu,
    energy_exchange_kwargs=None,
    density_dt_fraction=0.25,
    plasma_active=None,
):
    """Return a fractional energy timestep for electron-ion exchange."""
    if energy_exchange_kwargs is None:
        return np.inf
    if density_dt_fraction <= 0.0:
        raise ValueError(
            f"density_dt_fraction must be positive (got {density_dt_fraction})"
        )
    rhs = electron_ion_exchange_rhs(
        state=state,
        floors=floors,
        ion_mass_g=ion_mass_g,
        mu=mu,
        **energy_exchange_kwargs,
    )
    return min(
        _fractional_timestep(
            state.Ee,
            rhs.Ee,
            density_dt_fraction,
            0.0,
            active_mask=plasma_active,
        ),
        _fractional_timestep(
            state.Ei,
            rhs.Ei,
            density_dt_fraction,
            0.0,
            active_mask=plasma_active,
        ),
    )


def electron_cooling_timestep(
    state,
    floors,
    ion_mass_g,
    electron_cooling_kwargs=None,
    density_dt_fraction=0.25,
    plasma_active=None,
):
    """Return a fractional electron-energy timestep for cooling losses."""
    if electron_cooling_kwargs is None:
        return np.inf
    if density_dt_fraction <= 0.0:
        raise ValueError(
            f"density_dt_fraction must be positive (got {density_dt_fraction})"
        )
    rhs = electron_cooling_rhs(
        state=state,
        floors=floors,
        ion_mass_g=ion_mass_g,
        **electron_cooling_kwargs,
    )
    return _fractional_timestep(
        state.Ee,
        rhs.Ee,
        density_dt_fraction,
        0.0,
        active_mask=plasma_active,
    )


def ion_charge_exchange_timestep(
    state,
    floors,
    ion_mass_g,
    ion_charge_exchange_kwargs=None,
    density_dt_fraction=0.25,
    plasma_active=None,
):
    """Return a fractional ion-energy timestep for charge exchange."""
    if ion_charge_exchange_kwargs is None:
        return np.inf
    if density_dt_fraction <= 0.0:
        raise ValueError(
            f"density_dt_fraction must be positive (got {density_dt_fraction})"
        )
    rhs = ion_charge_exchange_rhs(
        state=state,
        floors=floors,
        ion_mass_g=ion_mass_g,
        **ion_charge_exchange_kwargs,
    )
    return _fractional_timestep(
        state.Ei,
        rhs.Ei,
        density_dt_fraction,
        0.0,
        active_mask=plasma_active,
    )


def heat_conduction_timestep(
    state,
    floors,
    ion_mass_g,
    mu,
    geometry,
    heat_conduction_kwargs=None,
    plasma_active=None,
):
    """Return an explicit diffusion timestep bound for heat conduction."""
    if heat_conduction_kwargs is None:
        return np.inf
    return heat_conduction_timestep_bound(
        state=state,
        floors=floors,
        ion_mass_g=ion_mass_g,
        mu=mu,
        geometry=geometry,
        active_cells=plasma_active,
        **heat_conduction_kwargs,
    )


def _distance_timestep(distance, speed, fraction):
    active = speed > 0.0
    if not np.any(active):
        return np.inf
    return float(fraction * np.min(distance[active] / speed[active]))


def _active_values(values, active_mask):
    values = np.asarray(values, dtype=float)
    if active_mask is None:
        return values
    return values[np.asarray(active_mask, dtype=bool)]


def _negative_margin_timestep(margin, rate, fraction, active_mask):
    margin = np.asarray(margin, dtype=float)[active_mask]
    rate = np.asarray(rate, dtype=float)[active_mask]
    draining = rate < 0.0
    if not np.any(draining):
        return np.inf
    draining_margin = margin[draining]
    if np.any(draining_margin <= 0.0):
        return 0.0
    return float(fraction * np.min(draining_margin / -rate[draining]))


def _fractional_timestep(values, rates, fraction, floor, active_mask=None):
    values = np.asarray(values, dtype=float)
    rates = np.asarray(rates, dtype=float)
    if active_mask is not None:
        active_cells = np.asarray(active_mask, dtype=bool)
        values = values[active_cells]
        rates = rates[active_cells]
    active = np.abs(rates) > 0.0
    if not np.any(active):
        return np.inf
    margin = np.maximum(np.asarray(values, dtype=float) - floor, 0.0)
    positive_margin = margin[active] > 0.0
    if not np.any(positive_margin):
        return np.inf
    active_rates = np.abs(rates[active])[positive_margin]
    active_margin = margin[active][positive_margin]
    return float(fraction * np.min(active_margin / active_rates))
