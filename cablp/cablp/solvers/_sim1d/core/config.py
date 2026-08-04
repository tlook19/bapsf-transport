import tomllib

from cablp.vars._nn_table import lookup_nn0


def initial_condition_defaults():
    """Return defaults for species and initial primitive state.

    gas_type:
        Neutral/ion species selector. Options are ``"He"`` for helium atom/ion
        conventions and ``"H"`` for hydrogen neutral/proton conventions.
    ne0:
        Uniform initial plasma/electron density [cm^-3].
    nn0:
        Uniform initial neutral density [cm^-3]. If ``None``, the value is
        looked up from the gas-puff table via ``resolve_nn0``.

        This is the DIRECT-RUN fill only. The equilibrated path
        (``neutral_equilibration`` via ``start_simulation``) does NOT read this
        value: ``run_neutral_equilibration`` pins its inner sim's start at the
        nn_table generator's 1e8 and overwrites nn with the equilibrated
        profile, so the two paths are decoupled and this default can move
        without disturbing any equilibrated run.

        Provenance of the shipped value: ``config_defaults_provenance.md``.
    Te0:
        Uniform initial electron temperature [eV].
    Ti0:
        Uniform initial ion temperature [eV].
    u0:
        Uniform initial axial plasma velocity [cm/s].
    Tn_fit:
        DEPRECATED; superseded by the single cold-gas ``Tn_K``. Was the neutral
        collision temperature used by the legacy IAEA reaction-rate fits and the
        legacy ion-neutral drag/thermalization/CX quartet -- all retired under
        the Phelps ``ion_neutral_moment_closure`` baseline, so it is inert
        whenever that flag is on. The deferred M_n wall accommodation should
        read ``Tn_K``.
    """
    return {
        # --- ACTIVE (production) ---
        "gas_type": "He",
        "ne0": 1e9,
        # Pre-shot neutral background for DIRECT runs. The equilibrated path
        # never reads this (see the docstring above).
        "nn0": 2.0e13,
        # Te0 sits just above the bundled He ADF11 low-Te edge (~0.200092 eV),
        # below which the rate lookups clamp. Ti0 sits a hair above Ti_floor so
        # the raw-stage validator's strict Ti0 > Ti_floor holds (that floor is a
        # numerical positivity floor, not a temperature assertion).
        "Te0": 0.21,
        "Ti0": 0.026,
        "u0": 0.0,
        # --- DEPRECATED ---
        "Tn_fit": 0.1,
    }


def geometry_defaults():
    """Return defaults for the resolved 1D typed-segment geometry.

    Lm:
        Total machine length represented by the 1D mesh [cm].
    nx:
        Number of resolved column cells between anode and collector. With the
        default-off ``source_fixed_grid`` flag on it counts only the *far*
        column cells, between the source region end and the collector.
    Rm:
        Default neutral/machine radius [cm].
    Rp:
        Default plasma radius [cm].
    The remaining keys configure the resolved typed-segment geometry.
    D2 removed the legacy lumped geometry.

    In resolved mode the cathode surface defines the origin: it sits at ``z = 0``
    and the anode at ``z = cathode_anode_gap_cm``, with the plenum (and any
    obstruction) extending to *negative* z behind the cathode. ``Lm`` therefore
    spans the cathode surface to the far machine end; total mesh length is
    ``Lm + plenum_length_cm + Lcs``. Cathode and anode are **faces**, not
    cells, so they have positions but no length.

    plenum_length_cm:
        Length of each neutral-only plenum cell behind a cathode [cm].
    cathode_anode_gap_cm:
        Cathode-surface-to-anode distance [cm]; the anode face sits here.
    nx_gap:
        Number of resolved cells across the cathode-anode gap. These are the
        smallest cells in the mesh, so they set the explicit CFL timestep.
    collector_length_cm:
        Length of the collector cell at the non-cathode end (single-cathode
        layout only; the twin layout mirrors the source end instead) [cm].
    Rcs:
        Inner radius of the annular cathode-structure obstruction between plenum
        and cathode [cm]. ``0`` => full-bore (no obstruction). Consumed in M2.
    Lcs:
        Axial length of that obstruction [cm]. ``0`` => full aperture. Consumed
        in M2.
    Rsup:
        Effective blockage radius of plenum support rods [cm]. ``0`` => none;
        reduces plenum neutral volume only. Consumed in M2.
    end_expansion_cells:
        Number of cells resolving the collector/end expansion when the
        ``end_expansion_geometry`` flag is enabled. ``None`` when off.
    end_expansion_machine_radius_cm:
        Vessel/neutral radius [cm] throughout the expanded end region.
        Requires ``end_expansion_geometry``.
    end_expansion_plasma_radius_cm:
        Terminal plasma flux-tube radius [cm] at the end wall. The plasma
        cross-sectional area expands smoothly across the end region from
        ``Rp`` to this value. Setting it equal to ``Rp`` gives the
        vessel-only arm. Requires ``end_expansion_geometry``.
    neutral_baffle_positions_cm:
        Axial positions [cm] of optional thin annular baffles, measured from
        the cathode surface. A scalar or sequence is accepted. Requires the
        default-off ``neutral_baffles`` flag and matching clear radii.
    neutral_baffle_clear_radii_cm:
        Clear aperture radii [cm] for ``neutral_baffle_positions_cm``. Each
        aperture must leave the local plasma channel fully open and lie inside
        the local vessel radius. A scalar or sequence is accepted.
    source_region_length_cm:
        End of the fixed-cell-size source region [cm, measured from the cathode
        surface]; the region runs from the anode face at ``cathode_anode_gap_cm``
        to here and must lie strictly between the anode face and the collector
        block (``Lm - collector_length_cm``). ``None`` when off. Requires the
        default-off ``source_fixed_grid`` flag. For the value the campaign
        stance sets, see ``scripts/production_stance_provenance.md``.
    source_region_dz_cm:
        Cell size [cm] inside that source region, held fixed independently of
        ``nx``; the region length minus the anode gap must be an integer
        multiple of it (1e-9 relative tolerance). ``None`` when off. Requires
        the default-off ``source_fixed_grid`` flag.
    """
    return {
        "Lm": 2000.0,
        "nx": 60,
        "Rm": 50.0,
        "Rp": 15.0,
        "plenum_length_cm": 100.0,
        "cathode_anode_gap_cm": 50.0,
        "nx_gap": 5,
        "collector_length_cm": 100.0,
        "Rcs": 0.0,
        "Lcs": 0.0,
        "Rsup": 0.0,
        "end_expansion_cells": None,
        "end_expansion_machine_radius_cm": None,
        "end_expansion_plasma_radius_cm": None,
        "neutral_baffle_positions_cm": None,
        "neutral_baffle_clear_radii_cm": None,
        "source_region_length_cm": None,
        "source_region_dz_cm": None,
    }


def floor_defaults():
    """Return numerical floors applied to conservative state variables.

    ne_floor:
        Minimum plasma/electron density used when flooring state [cm^-3].
    nn_floor:
        Minimum neutral density used when flooring state [cm^-3].
    Te_floor:
        Minimum electron temperature recovered from conservative energy [eV].
    Ti_floor:
        Minimum ion temperature recovered from conservative energy [eV].
        The Phelps ``ion_neutral_moment_closure`` collision operator is
        thermal-valid with no 0.1 eV clamp; the only consumer that required
        0.1 eV was the retired legacy IAEA CX table. All remaining Ti consumers
        (kappa_par_ion, pressure, sound speed) need only Ti > 0.
    Te_floor:
        Minimum electron temperature recovered from conservative energy [eV].
        Sits below the ADF11 0.2 eV edge so the afterglow can cool. Lowering it
        toward the neutral-gas temperature is only meaningful together with the
        sub-edge ADAS extension -- see the RETIRED recipe in the module note
        below, which must not be run.

    Values and their provenance: ``config_defaults_provenance.md``.
    """
    return {
        "ne_floor": 1e8,
        "nn_floor": 1e8,
        "Te_floor": 0.1,
        "Ti_floor": 0.02585,
    }


# Deep-afterglow low-Te recipe -- RETIRED, DO NOT RUN.
#
# The recipe was: lower Te_floor to the neutral-gas temperature (0.02585) and
# set flags adas_low_te_extension=True, icool_recomb=True. The extension makes
# acd (recombination) and prb1 (recombination radiation) extend consistently
# below the 0.2 eV ADF11 edge; scd (ionization) and plt (line power) still
# clamp there but are exponentially dead at <0.2 eV, so a recombining 300 K
# afterglow would be well represented.
#
# DO NOT RUN icool_recomb TOGETHER WITH adas_low_te_extension. The two compose
# destructively: icool_recomb charges bare PRB (the double-charge warned about
# at recombination_energy_return below), and adas_low_te_extension amplifies
# the sub-edge PRB by ~9,300x, so the electron fluid runs away thermally to the
# floor and the electron_cooling timestep bound collapses permanently. The
# consistent net booking (I_ion*S_rec - P_PRB) that would make the pair sound
# is NOT built. Without it the afterglow validity window is Te > 0.2 eV (the
# ADF11 edge).


def neutral_source_defaults():
    """Return gas-puff, pump, and neutral-source defaults.

    S_gp:
        Source-side gas puff flow [sccm].
    Twin_S_gp:
        End-side gas puff flow used when ``TwinCathode`` is enabled [sccm].
    gas_puff_mode:
        Phase-dependent gas-puff schedule. ``"square"`` (default) holds the
        flow flat at ``S_gp`` between an opening and a closing erf edge (see
        below). The remaining modes are DEPRECATED (retained runnable for the
        frozen waveform-comparison figures; non-default use warns):
        ``"decay_after_breakdown"`` for steady puffing until optional decay
        after breakdown/main-discharge start,
        ``"pulse_decay_to_level"`` for a full-rate pulse followed by decay
        toward the configured target flow, and ``"double_erf"`` for a
        valve-like waveform: an erf (S-shaped) rise ``0 -> S_gp``, a
        plateau, then an erf drop ``S_gp -> S_gp_decay_target`` which is
        held for the rest of the discharge. Both transitions sit on the
        *scheduled* main-discharge clock (centers may be negative, i.e.
        during prebreakdown -- a real valve opens before breakdown), and
        the waveform replaces the other modes' full-rate prebreakdown
        behaviour. The twin puff shares the timing with its own levels.
        Smooth everywhere; clamped at zero if the transitions are set to
        overlap pathologically.
    tau_gp_rise_center:
        ``double_erf`` rise-transition center [s], relative to the
        scheduled main-discharge start (negative = before breakdown).
    tau_gp_rise_width:
        ``double_erf`` rise erf width scale [s]; the 10-90% rise time is
        ~1.81x this value.
    tau_gp_drop_center:
        ``double_erf`` drop-transition center [s], same clock as the rise.
    tau_gp_drop_width:
        ``double_erf`` drop erf width scale [s].
    S_gp_decay_target:
        Source-side target puff level for pulse decay modes [sccm].
    Twin_S_gp_decay_target:
        End-side target puff level for pulse decay modes [sccm].
    tau_gp_after_breakdown:
        Delay after breakdown/main-discharge start before puff decay begins [s].
        ``None`` keeps the puff steady through the main discharge.
    tau_gp_decay_factor:
        Multiplier applied to the main-discharge decay time constant.
    tau_gp_pulse_duration:
        Full-rate puff duration after breakdown for pulse decay modes [s].
    tau_gp_decay_duration:
        E-folding time toward the decay target for pulse decay modes [s].
    S_pump_L:
        Source-side vacuum pump speed [L/s].
    S_pump_R:
        End-side vacuum pump speed [L/s].
    gas_puff_enabled:
        Enables neutral gas-puff source terms.
    pump_enabled:
        Enables neutral pump sink terms.
    gas_puff_valves:
        Number of equivalent gas-puff valves used by the SCCM conversion.
    gas_puff_profile:
        Axial shape of the puff. ``"cell"`` (default, the historical
        behaviour) puts the whole flow in the role-tagged puff cell (in front
        of the anode in resolved geometry). ``"cosine_pipe"`` is the physical
        source -- a small pipe at the chamber wall ~10 cm in front of the
        anode, pointing radially inward with a Lambertian (cosine) outlet;
        its first-flight axial deposition is the cosine-lobe pattern
        ``[1 + ((z - z0)/d)^2]^-2`` with throw ``d ~ 2*Rm``, so centre and
        width both come from geometry rather than tuning. ``"gaussian"`` is
        the generic tunable shape. All distributed profiles conserve the
        total inflow exactly, and one shared implementation feeds both the
        explicit RHS and the implicit neutral matrix, so the two sites
        cannot desync.
    gas_puff_z_cm:
        Distributed-puff centre [cm, machine coordinates]; ``None`` falls back
        to whichever cell currently holds the ``puff`` role. Mirrored through
        the chamber midpoint for the twin puff. Pinning it in machine
        coordinates is what makes an nx refinement a resolution study: with
        ``None`` the source centre follows the puff cell's centre, so changing
        nx silently moves the source. Ignored by the ``"cell"`` profile, which
        puts the whole flow in the role-tagged cell.
    gas_puff_sigma_cm:
        Gaussian puff axial width [cm].
    gas_puff_throw_cm:
        Cosine-pipe throw distance ``d`` [cm], of order the chord across the
        chamber (~2*Rm). Sets the lobe's HWHM = 0.64*d.
    pump_elbow_conductance_lps:
        Conductance of the unmodeled pump elbow [L/s], combined in series with
        the pump speed as ``1/S_eff = 1/S_pump + 1/C_elbow``. Applies only to a
        pump sitting on a plenum cell, so it is inert in legacy geometry.
        ``None`` (default) or a
        non-positive value means no elbow restriction -- the legacy limit.

    Values and their provenance: ``config_defaults_provenance.md``.
    """
    return {
        # --- ACTIVE (square waveform + pump) ---
        # S_gp is the one free constant of the puff model; every other quantity
        # in the waveform is a hardware timing. It feeds back on the discharge
        # through S_gp -> ne -> current.
        "S_gp": 3400,
        "Twin_S_gp": 3400,
        "gas_puff_mode": "square",
        # "square" waveform edge timings. The piezo is driven by a square
        # voltage pulse from the SAME trigger that closes the cathode circuit
        # and is held for the discharge, so the flow is FLAT at S_gp with only
        # the piezo-opening/entry-transit erf edges. The rise ANCHORS ON
        # circuit-on (the end of the neutral-prebreakdown phase), not on
        # breakdown, so breakdown rides the inter-shot residual fill; the close
        # lag delays the closing edge past the end of the main discharge, so
        # the close tail runs into the afterglow. These three are hardware
        # timings, not fit knobs.
        "gas_puff_rise_center_s": 5e-4,
        "gas_puff_rise_width_s": 5e-4,
        "gas_puff_close_lag_s": 5e-4,
        # S_pump_L matches S_pump_R: the plenum aperture, not the pump speed,
        # throttles the source-side rate.
        "S_pump_L": 4000,
        "S_pump_R": 4000,
        "gas_puff_enabled": True,
        "pump_enabled": True,
        "gas_puff_valves": 2,
        "pump_elbow_conductance_lps": None,
        # Physical Lambertian pipe source ~10 cm in front of the anode; its
        # centre and width are geometry-derived, not tunable.
        "gas_puff_profile": "cosine_pipe",
        # The pipe position, in machine coordinates so it does not move with nx.
        "gas_puff_z_cm": 60.0,
        "gas_puff_sigma_cm": 50.0,
        "gas_puff_throw_cm": 100.0,
        # Fresh-puff fractional-coverage local ionization (default 0 = OFF,
        # bit-exact). Fraction of the localized gas-puff neutral source that is
        # ionized IN PLACE (the dense spotty jet has a short beam/bulk mfp, so
        # it burns to a localized plasma seed that launches the sonic
        # accumulation front) instead of spreading into the background nn. The
        # diverted
        # neutrals are debited from the puff and booked as ionization with the
        # bulk-reaction birth + I_ion cost (mass/energy conserving); it rides the
        # puff shape+waveform so it is auto-localized and relaxes with the ~1 ms
        # feed. Single-zone only (loud error with neutral_two_zone). In [0, 1).
        "gas_puff_local_ionization_fraction": 0.0,
        # --- DEPRECATED (only read by the retired pulse/decay/double_erf puff
        # modes; kept runnable for the frozen waveform-comparison figures) ---
        "S_gp_decay_target": 1500,
        "Twin_S_gp_decay_target": 0.0,
        "tau_gp_after_breakdown": None,
        "tau_gp_decay_factor": 1.0,
        "tau_gp_pulse_duration": 1e-3,
        "tau_gp_decay_duration": 5e-3,
        "tau_gp_rise_center": -5e-3,
        "tau_gp_rise_width": 1e-3,
        "tau_gp_drop_center": 1e-3,
        "tau_gp_drop_width": 1e-3,
    }


def timing_defaults():
    """Return phase timing and current-trigger defaults.

    tau_prebreakdown:
        Maximum pre-breakdown duration or scheduled pre-breakdown phase [s].
    tau_neutral_prebreakdown:
        Neutral-only accumulation duration before the plasma/cathode
        current-triggered phases begin [s]. The plasma clock starts at the end
        of this window, so a positive value delays the whole discharge by
        exactly that much.

        A POSITIVE VALUE IS AN OPT-IN for studies that specifically want a
        neutral-only accumulation phase. Zero disables the pre-phase entirely.
        The ``neutral_prebreakdown`` flag stays on by default and gates the
        machinery, so setting this duration alone is enough to get the phase
        back.

        For why the default is what it is, see
        ``config_defaults_provenance.md``.
    tau_breakdown:
        Scheduled breakdown duration before main discharge when not using
        current-triggered transitions [s].
    tau_discharge:
        Main-discharge duration [s].
    tau_afterglow:
        Afterglow duration after the main discharge [s].
    tau_cycle:
        Neutral-only puff/off cycle duration [s].
    equilibration_gas_puff_on_s:
        Per-cycle gas-puff ON window of the neutral-equilibration inner sim [s].
        ``None`` (the default) keeps the historical behaviour exactly: the
        window is ``tau_discharge``, i.e. the equilibration inherits the
        MAIN-DISCHARGE duration as its puff width.

        That inheritance is a double duty with no physical basis: the
        equilibration's puff window is the machine's total gas-puff pulse
        width, an independent hardware quantity. Set it explicitly to decouple
        the two (see ``scripts/production_stance_provenance.md`` for the value
        the campaign stance uses).

        Read ONLY by the ``Plasma=False`` equilibration inner sim; the main
        run's puff is closed by its own waveform envelope, never by this
        window. Must be > 0 and (when ``tau_cycle`` > 0) <= ``tau_cycle``;
        anything else raises at construction.
    cycles:
        Number of neutral-only cycles used by the default run duration.
    neutral_equilibration_cycles:
        Number of puff/off cycles used by the optional neutral pre-equilibration
        run.
    neutral_equilibration_dt:
        Fixed timestep for optional neutral pre-equilibration [s]. ``None`` uses
        the adaptive timestep selector, which may be much slower.
    phase_transition_mode:
        Phase scheduler mode. Options are ``"scheduled"`` to use configured
        phase durations and ``"current"`` to use cathode ``I_tot`` thresholds.
    I_prebreakdown:
        Cathode total-current threshold for leaving pre-breakdown [A].
    I_breakdown:
        Cathode total-current threshold for entering main discharge [A].
    prebreakdown_timeout_action:
        What happens when ``tau_prebreakdown`` elapses without a breakdown
        trigger. ``"switch_open"`` (default) mirrors the machine's own hardware
        guard: the cathode switch OPENS, a ``"prebreakdown_timeout"`` phase
        event is recorded, and the run winds down through the existing
        afterglow machinery to a finite end time instead of crawling at a
        collapsed timestep. ``"raise"`` is the historical behavior -- a
        ``BreakdownError`` is raised and the in-progress trajectory is lost;
        it is retained for the sweep drivers that classify a point from that
        exception. Only consulted under
        ``phase_transition_mode="current"`` (the scheduled scheduler has no
        breakdown trigger to miss).

    Values and their provenance: ``config_defaults_provenance.md``.
    """
    return {
        "tau_prebreakdown": 0.05,
        # 0.0 disables the neutral-only pre-drive window entirely.
        "tau_neutral_prebreakdown": 0.0,
        "tau_breakdown": 0.0,
        "tau_discharge": 20e-3,
        "tau_afterglow": 5e-3,
        "tau_cycle": 3.0,
        "equilibration_gas_puff_on_s": None,
        "cycles": 1,
        "neutral_equilibration_cycles": 100,
        "neutral_equilibration_dt": 1e-2,
        "phase_transition_mode": "current",
        "I_prebreakdown": 150.0,
        "I_breakdown": 1000.0,
        "prebreakdown_timeout_action": "switch_open",
    }


def output_defaults():
    """Return saved-output cadence and cap defaults.

    dt_save:
        Minimum time between saved trajectory samples [s]. Non-positive values
        save every accepted step.
    t_save_start:
        Earliest simulation time to start saving trajectory samples [s].
    max_output_steps:
        Maximum number of saved trajectory samples. Zero means unlimited.
    neutral_seed_cache_dir:
        Directory of the neutral-equilibration seed DATABASE used when the
        ``use_cached_neutral_seed`` flag is on. Each distinct neutral-flow
        configuration (geometry / puffing / pumping / neutral physics) gets one
        entry ``neutral_seed_<signature>.npz``, auto-populated on first use and
        reused thereafter (a browsable fill-rate table). ``None`` (default) means
        no database is configured. See ``core/neutral_seed_cache.py`` and
        ``scripts/build_neutral_seed_cache.py``.
    """
    return {
        "dt_save": 1e-5,
        "t_save_start": 0.0,
        "max_output_steps": 0,
        "neutral_seed_cache_dir": None,
    }


def model_mode_defaults():
    """Return string-valued model selector defaults.

    front_flux_model:
        Axial plasma front-filling flux closure. The implemented option is
        ``"sonic_relaxation"``; the ``front_flux`` flag enables or disables the
        closure.
    hyperbolic_wave_speed:
        Signal speed used by both the Rusanov dissipation ``a_max`` and the
        plasma CFL. ``"isothermal"`` is the historical gamma=1 Bohm
        speed ``sqrt(Te/m_i)``; ``"adiabatic"`` (default) is the exact linear
        acoustic speed of the implemented gamma=5/3 two-species ideal-gas
        energy system, ``sqrt((5/3)(Te+Ti)/m_i)``.
    D_amb_model:
        Ambipolar diffusion coefficient model. ``"cs_dz"`` is retained for
        _sim3 compatibility; the current conservative flux closure does not
        directly use this selector.
    end_mode:
        End boundary behavior. Options are ``"collector"`` and
        ``"mirrored_source"``.
    cathode_model:
        Cathode model selector retained for configuration compatibility. The
        current option is ``"disabled"``; actual cathode coupling is controlled
        by the ``cathode_coupling`` flag.
    Te_birth_ionization:
        Electron birth temperature model for ionization. Options are
        ``"local"`` to use the local electron temperature, ``"floor"`` to use
        the electron temperature floor, or a numeric eV value.
    Ti_birth_ionization:
        Ion birth temperature model for ionization. Options are ``"local"`` to
        use the local ion temperature, ``"floor"`` to use the ion temperature
        floor, or a numeric eV value.
    ionization_birth_energy_model:
        How ionization births book their energy moments. ``"legacy"``
        (historical): the electron birth adds ``3/2 Te_birth S_ion`` to ``Ee``
        and the ion birth adds ``3/2 Ti_birth S_ion`` to ``Ei``; under
        ``Te_birth_ionization="local"`` the electron term creates ``3 Te/2`` of
        thermal energy per new electron, cancelling most of the ionization
        potential cost -- unphysical (a new electron carries no kinetic energy).
        ``"conservative"`` (default): reconciles bulk (and beam) births to the
        defensible
        ``Ee = 0`` convention the beam already uses -- the new electron is born
        cold, so ``Te`` falls by dilution -- and books the ion mass-loading
        relative-drift mixing energy ``1/2 m (u_i - u_n)^2 S_ion`` to ``Ei``
        explicitly, so ion total energy (internal + kinetic) closes to the
        consumed neutral's energy instead of losing the drift energy through the
        bulk kinetic derivative. Under ``"conservative"`` the
        ``Te_birth_ionization`` selector is inert (the electron birth energy is
        physically zero).
    neutral_exchange_model:
        Axial neutral transport model. ``"constant"`` uses a fixed coefficient.

        ``"knudsen"`` (default) treats cell-to-cell exchange as Fickian transport with the
        Knudsen diffusivity ``D = (2/3)*v_th*R``, i.e. ``C = D*A/dz``. This is
        mesh-independent and reproduces the textbook long-tube conductance
        ``(2*pi/3)*v_th*R^3/L`` exactly. Thin apertures (the anode mesh) keep an
        orifice conductance in series. Prefer this for resolved runs, where the
        puff-to-pump back-path is the physics of interest and the historical model
        under-predicts it by 2-14x depending on cell size.
    operator_splitting:
        How the operator-split path composes the explicit non-heat operator A
        with the implicit heat operator B. ``"lie"`` does ``A(dt)`` then
        ``B(dt)`` and is first-order in dt however accurate the two
        sub-integrators are, because the splitting error goes as dt*[A,B].
        ``"strang"`` does ``B(dt/2)``, ``A(dt)``, ``B(dt/2)``, whose symmetry
        cancels that leading term and leaves O(dt^2), at the cost of one extra
        heat substep per step -- B is halved rather than A because it is the
        cheap operator. Second-order overall also requires a second-order
        ``implicit_heat_scheme`` and a positive ``heat_picard_iterations``;
        Strang alone only removes the splitting term. Ignored when the
        operator-split path is disabled.
    implicit_heat_scheme:
        Time-discretization of the implicit heat-conduction substep used by the
        operator-split path (``implicit_heat_conduction`` flag). Options are
        ``"backward_euler"`` (theta=1; unconditionally monotone, so it cannot
        undershoot the temperature floors), ``"crank_nicolson"`` (theta=1/2;
        second-order in the substep but leaves stiff modes ringing at undamped
        amplitude), ``"shifted"`` (theta=0.6; first-order with roughly a fifth
        of backward Euler's error constant, and damps ringing by ~2/3 per
        step), and ``"tr_bdf2"`` (a trapezoidal stage followed by a BDF2 stage;
        second-order *and* L-stable, so it rings far less than Crank-Nicolson
        at twice the solve cost, though it is not monotone like backward
        Euler). Ignored when the operator-split path is disabled.

    Values and their provenance: ``config_defaults_provenance.md``.
    """
    return {
        # --- ACTIVE ---
        "front_flux_model": "sonic_relaxation",
        # "adiabatic" PAIRS with hyperbolic_energy_consistent: same gamma=5/3
        # energy system, so the signal speed matches the flux.
        "hyperbolic_wave_speed": "adiabatic",
        "end_mode": "collector",
        "Ti_birth_ionization": "floor",
        # "conservative": no spurious 3Te/2 electron birth energy; ion
        # mass-loading mixing energy booked explicitly.
        "ionization_birth_energy_model": "conservative",
        "neutral_exchange_model": "knudsen",
        "neutral_model": "moment",
        # 2nd-order operator-split pair; both are needed together with
        # heat_picard_iterations > 0 for the step to reach second order.
        "operator_splitting": "strang",
        "implicit_heat_scheme": "tr_bdf2",
        # --- INERT under these defaults (kept for the A/B arms) ---
        # Dead under ionization_birth_energy_model="conservative" (electron
        # birth energy is physically zero):
        "Te_birth_ionization": "local",
        # Dead under neutral_model="moment" (the K4a kinetic engine, gated on
        # neutral_two_zone, is the only consumer):
        "neutral_kinetic_refresh_s": 5e-4,
        "neutral_kinetic_refresh_tol": 0.2,
        "neutral_kinetic_nvz": 48,
        "neutral_kinetic_nvp": 12,
        # Bucket-2 default-off closure instrument (low-Te ADAS extension; only
        # active with icool_recomb, sub-0.2 eV):
        "adas_low_te_extension": False,
        # --- DEPRECATED (legacy-compat selectors; superseded, non-default
        # warns; retained for reproducibility at the tag) ---
        # D_amb_model: _sim3-compat; the conservative flux closure never uses it.
        "D_amb_model": "cs_dz",
        # cathode_model: compat only; actual coupling is the cathode_coupling flag.
        "cathode_model": "disabled",
    }


def fudge_factor_defaults():
    """Return physics scale factors and boundary geometry multipliers.

    alpha_front:
        Multiplier for the front-filling/sonic relaxation flux.
    D_amb:
        Constant ambipolar diffusion coefficient when selected [cm^2/s].
    atomic_rate_model:
        Source of the He atomic rate coefficients. ``"adas"`` (default) uses
        the OPEN-ADAS GCR '96 effective
        coefficients (``cablp/vars/adas``, see its README): SCD ionization
        (includes the stepwise/metastable channel the direct rate lacks --
        up to ~3-6x at 3-5 eV, LAPD densities), ACD recombination (includes
        three-body, so ``b_rec_3b`` is inert), and PLT/PRB radiated power for
        the ``b_Qei``/``b_Qen`` cooling terms. The ADAS cooling coefficients
        are radiation-only and therefore consistent with the separate
        ``ionization_energy_cost`` term; the IAEA He I fit is not -- it
        already contains the ionization-potential loss, which ``"janev"``
        with ``b_Qen`` near 1 double-counts. ``"janev"`` (the historical
        behaviour) uses the direct ground-state ionization rate, the separate
        radiative/three-body recombination coefficients, and the IAEA cooling
        fits. ``"adas"`` is wired for ``gas_type = "He"`` only -- hydrogen
        configs must set ``"janev"`` or the solver raises at construction.
    b_ioniz:
        Bulk ionization particle source scale factor.
    b_rec_rad:
        Radiative recombination particle sink scale factor. Under
        ``atomic_rate_model = "adas"`` this scales the whole (ACD) sink.
    b_rec_3b:
        Three-body recombination particle sink scale factor. Inert under
        ``atomic_rate_model = "adas"`` (ACD already includes three-body).
    b_Qie:
        Electron-ion thermal exchange scale factor.
    b_Qei:
        Electron-ion inelastic/radiative cooling scale factor.
    b_Qen:
        Electron-neutral inelastic cooling scale factor.
    b_Qei_Te_exp, b_Qen_Te_exp:
        Optional Te-dependent shape for the corresponding correction: a nonzero
        exponent multiplies the term by ``(Te / b_Q_Te_ref_eV) ** exp``. The
        IAEA cooling fits carry a factor ~2 uncertainty across the 2-12 eV
        discharge range, and a constant scalar cannot express an error that
        varies over that range; this hook admits a literature- or
        decay-calibrated shape without touching the fits. ``0`` (default)
        skips the factor entirely.
    b_Q_Te_ref_eV:
        Reference temperature for the Te-dependent shape [eV]; the correction
        equals the bare ``b_Q*`` scalar exactly at this Te.
    b_Qcx:
        Ion charge-exchange cooling scale factor.
    b_epara:
        Electron axial heat-conduction scale factor.
    b_ipara:
        Ion axial heat-conduction scale factor.
    b_surface_loss:
        Plasma surface neutralization/loss scale factor.
    b_ion_neutral_drag:
        Ion-neutral drag (friction) momentum-sink scale factor. With the
        ``constant`` drag model this is the whole neutral-flow closure,
        asserting a fixed velocity slip ``u_n = (1 - b)*u`` everywhere; with
        the ``slip`` model it remains as an overall multiplier (leave at 1
        unless doing a sensitivity study).
    ion_neutral_drag_model:
        Closure for the neutral flow the drag acts against. ``"constant"``
        (default) uses ``b_ion_neutral_drag`` alone. ``"slip"`` computes a
        per-cell slip factor ``s = 1/(1 + E)`` from the entrainment balance
        ``E = nu_ni * tau_wall`` (ions entrain neutrals at ``n*sigma_in*v_ti``;
        neutrals lose the momentum to the wall in ``Rm / vbar_n``), so the slip
        sweeps from ~1 in rarefied plasma to ~0 at full entrainment instead of
        being asserted constant. Applies to the drag and frictional-heating
        terms (the latter quadratically). The ``neutral_momentum`` flag
        replaces both closures with an evolved neutral wind and is mutually
        exclusive with ``"slip"`` (which is that equation's own steady state).
    b_slip_entrainment:
        Multiplier on the entrainment parameter ``E`` of the slip closure;
        absorbs the O(1) geometric factors the balance ignores. Inert with the
        ``constant`` drag model.
    neutral_momentum_radial:
        Radial closure for the evolved neutral wind (requires the
        ``neutral_momentum`` flag; inert without it and errors if set to
        ``"two_zone"`` while the flag is off). ``"uniform"`` (default, the
        original M2 behaviour) treats ``M_n`` as radially uniform: the drag
        pushes against the chamber-mean wind and the wall sink is
        ``vbar_n/Rm``. ``"two_zone"`` closes the radial profile
        algebraically (``neutral_wind_two_zone_factors``): drag acts only in
        the plasma column, whose gas is entrained faster than it can escape
        radially, while the annulus gas is held slow by diffuse wall
        reflection -- so the drag, frictional heating, and ionization birth
        sample the *column* wind (chamber mean times ~3.3 on production
        geometry) and only the slow annulus gas reaches the wall (effective
        sink ~1.9e3 1/s vs the uniform 4.9e3 1/s). Net effect: less drag
        input, slower chamber-mean wind. ``"kinetic_two_moment"`` requires
        ``neutral_two_zone`` and evolves separate column and annulus
        momenta. Their radial exchange rates are fixed by the fast-ion and
        300-K free-molecular crossing times; annulus momentum alone reaches
        the vessel and optional baffles. This selector has no fitted
        coefficient.
    b_ion_neutral_thermalization:
        Scale factor for the elastic ion-neutral thermal-equilibration term.
        ``None`` (default) inherits ``b_ion_neutral_drag`` -- the historical
        coupling, kept for reproducibility -- but that term relaxes
        *temperature*, not momentum, so a slip-motivated drag scalar has no
        physical business scaling it. Set explicitly (e.g. 1.0) to decouple;
        an explicit value also frees the term from the ``ion_neutral_drag``
        flag's zeroing.
    b_presheath_length:
        Scale factor on the collisional presheath depth `c_s / nu_in` used to
        sample the upstream density for the Bohm flux at plasma-terminating
        surfaces. `alpha_isat` converts
        the *presheath-entrance* density to the sheath edge, so it must be applied
        to an upstream sample. `0` collapses the sample to the adjacent cell,
        recovering the historical behaviour; `1` (default) uses the physical
        depth. Inert in legacy geometry, which has no absorbing faces.
    sigma_in_cm2:
        Ion-neutral momentum-transfer cross section [cm^2]. Only used by the
        ``constant`` sigma_in_model.
    sigma_in_model:
        Source of the ion-neutral momentum-transfer rate, which feeds the
        drag, the slip closure's entrainment, thermalization, the drag
        timestep bound, and the presheath depth. ``"constant"`` (default, the
        historical behaviour) uses ``sigma_in_cm2`` at the ion thermal speed.
        ``"cx_derived"`` builds it from the same resonant charge-exchange
        table the CX energy channel uses -- ``nu_in = nn * (2*<sigma v>_cx(Ti)
        + k_Langevin)`` -- since for a symmetric pair each exchange transfers
        essentially the full momentum (``sigma_mt ~ 2*sigma_cx``), plus the
        velocity-independent Langevin polarization floor. This restores the
        factor ~2 velocity dependence a constant cannot express: the constant
        crosses the CX-derived curve near 0.5 eV (too small in the afterglow,
        ~1.5-1.8x at 0.1 eV; too large in the warm column, ~1.3x at 5-10 eV).
    alpha_isat:
        Ion-saturation/surface-loss coefficient.
    source_surface_area_scale:
        DEPRECATED: 0D artifact that stood in for un-separated cathode/anode
        I_sat. The resolved geometry measures the Bohm I_sat to each electrode
        face directly, so this multiplier has no operator to control and is
        never consumed; non-default use warns.
    end_surface_area_scale:
        DEPRECATED: 0D artifact, see ``source_surface_area_scale``.
    b_anode_collection:
        Multiplier on the resolved anode collection sink. This was formerly
        available only through an unregistered ``dict.get`` fallback.
    b_anode_advective_block:
        Fraction of the anode face treated as blocked by the mesh for
        advective transport. This was formerly available only through an
        unregistered ``dict.get`` fallback.

    Values and their provenance: ``config_defaults_provenance.md``.
    """
    # The atomic-rate / cooling / conduction scale factors are INERT under the
    # shipped defaults: ADAS and the Phelps collision operator supply those
    # channels directly, and a single uniform multiplier is not a physical
    # knob (the Te-dependent b_Q*_Te_exp hooks are the honest correction and
    # ship off). They remain READABLE via the solver's .get(key, 1.0), so the
    # "janev" A/B arm and the "=0 to disable a term" diagnostics still work.
    #
    # The must-be-1 STRUCTURAL constants (b_pressure_work_elec,
    # b_pressure_work_ions, b_ionization_energy_cost) are NOT registered at
    # all -- the solver hardwires 1.0, and resolve_config rejects them as
    # unknown keys. Exposing a knob that must be 1 is a footgun.
    return {
        # --- ACTIVE coefficients ---
        "atomic_rate_model": "adas",
        "b_surface_loss": 1.0,      # functional: =0 disables the boundary sink
        "b_presheath_length": 1.0,  # presheath depth (load-bearing)
        "alpha_isat": 0.6065306597126334,
        "b_anode_collection": 1.0,
        "b_anode_advective_block": 0.0,
        # alpha_front is ACTIVE only if the front_flux flag is on; front_flux
        # ships OFF, so this is inert by default.
        "alpha_front": 1.0,
        # Ion-neutral momentum-transfer rate, which also feeds the presheath
        # depth (electrode_sheath_alpha). "phelps" is the same cross section
        # ion_neutral_moment_closure uses (He-only); "constant"/"cx_derived"
        # are the A/B arms, and sigma_in_cm2 is read by "constant" only.
        "sigma_in_cm2": 5.0e-15,
        "sigma_in_model": "phelps",
        # --- INERT: superseded rate/cooling/conduction scales, locked at 1
        # (kept readable for the janev A/B and the =0 disable diagnostics) ---
        "b_ioniz": 1.0,
        "b_rec_rad": 1.0,
        "b_rec_3b": 1.0,       # also: ACD already includes three-body under adas
        "b_Qie": 1.0,
        "b_Qei": 1.0,
        "b_Qen": 1.0,
        "b_Qcx": 1.0,          # also dead under ion_neutral_moment_closure
        "b_Qei_Te_exp": 0.0,   # the honest Te-dependent hooks, off
        "b_Qen_Te_exp": 0.0,
        "b_Q_Te_ref_eV": 5.0,
        "b_epara": 1.0,        # the real conduction knob is the flux limiter
        "b_ipara": 1.0,
        "D_amb": 0.0,          # dead with the deprecated D_amb_model
        # GCR-consistent recombination energy booking (default-off closure
        # instrument; only active with icool_recomb, sub-0.2 eV). Per
        # recombination event, credit the binding energy I_ion to the electron
        # fluid (paid at ionization via I_ion*S_ion and never returned) AND
        # charge the full ADAS PRB (recombination radiation + bremsstrahlung +
        # cascade). Net = I_ion - E_rad. The PAIR is the consistent unit;
        # enabling PRB alone double-charges (why icool_recomb stays off) --
        # construction refuses the combination and requires atomic_rate_model=
        # "adas". adf11 grid bottoms at 0.2 eV; lookups clamp there.
        #
        # NOT BUILT: the consistent net booking (I_ion*S_rec - P_PRB), so
        # icool_recomb still charges bare PRB. Paired with
        # adas_low_te_extension -- which amplifies the sub-edge PRB by
        # ~9,300x -- that double-charge drives a thermal runaway to the Te
        # floor and a permanent electron_cooling timestep-bound collapse. DO
        # NOT RUN icool_recomb TOGETHER WITH adas_low_te_extension; see the
        # retired recipe in the module note above.
        "recombination_energy_return": False,
        # --- INERT: default-off instruments / neutral ladder ---
        # Electron heat-flux limiter (read only when the
        # electron_heat_flux_limit flag is on): free-streaming fraction f in
        # q_sat = f*n*Te*v_the, the harmonic (Cowie-McKee) saturation cap.
        "heat_flux_limiter_f": 0.3,
        # Non-local Knudsen exponent p for that limiter (read only when
        # electron_heat_flux_limit is on). lambda = 1/(1+Kn^p)
        # with Kn = q_SH/q_sat. p=1.0 (default) is the harmonic Cowie-McKee
        # form. p>1 suppresses the steep-gradient (high-Kn, non-local)
        # startup flux much harder while leaving the shallow-gradient established
        # column near-Spitzer -- the startup-front pre-heating vs established-
        # column trade a single free-streaming factor cannot separate.
        "heat_flux_limiter_exponent": 1.0,
        # Radial closure for the evolved neutral wind (needs the
        # neutral_momentum flag; the deferred ladder).
        "neutral_momentum_radial": "uniform",
        # --- DEPRECATED: legacy ion-neutral drag (superseded by the Phelps
        # ion_neutral_moment_closure) + the 0D area scales. Warn on
        # non-default/active use; retained runnable for reproducibility. ---
        "b_ion_neutral_drag": 1.0,
        "ion_neutral_drag_model": "constant",
        "b_slip_entrainment": 1.0,
        "b_ion_neutral_thermalization": None,
        "source_surface_area_scale": 1.8,
        "end_surface_area_scale": 1.0,
    }


def cathode_defaults():
    """Return LaB6 cathode/device circuit defaults.

    V_bank:
        Cathode power-supply bank voltage [V].
    T_s:
        Cathode surface temperature [K].
    phi_wf:
        Cathode work function [eV].
    C_R:
        EFFECTIVE Richardson emission constant [A cm^-2 K^-2] in
        ``J = C_R T^2 exp(-e phi_wf/(kB T))``. Not the Richardson-Dushman
        universal (120): the cathode literature treats this prefactor as an
        effective constant absorbing surface state, patch fields and the
        non-ideal emitting fraction.

        ``C_R`` and ``cathode_Ts_base_K`` are DEGENERATE in this expression --
        a change in the prefactor trades against a change in surface
        temperature along one flat direction -- so a configuration must not
        move both to represent the same emission. Values and their provenance:
        ``config_defaults_provenance.md`` and, for the campaign stance,
        ``scripts/production_stance_provenance.md``.
    R_comp:
        External/compliance resistance [Ohm]. The full loop series resistance.
        It does NOT set the discharge current -- the emission ceiling does;
        ``R_comp`` sets the voltage headroom, and the loop current is only
        weakly sensitive to it. ``R_comp`` and ``C_bank_F`` are jointly
        determined and must move together; see
        ``config_defaults_provenance.md``.
    R_comp_partition:
        Voltage-probe partition fraction ``x`` of ``R_comp``. ``R_comp`` is
        split into an external part ``x*R_comp`` (bank side of the probe) and
        an internal part ``(1-x)*R_comp`` (probe->plasma). The reported
        ``V_dis = V_bank - I*(x*R_comp) - L*dI/dt``; the plasma sees
        ``V_b = V_dis - I*((1-x)*R_comp + R_mesh)``.

        This parameter is DYNAMICALLY INERT, OBSERVATIONALLY ACTIVE, and
        therefore a calibration knob. Read all three together -- the first
        alone reads as "ignore this parameter", and it is not ignorable.

        1. DYNAMICALLY INERT. ``x`` cancels identically from the loop
           equation. The circuit is handed ``R_comp_ohm = x*R_comp``
           (``solver.py``) while ``vdis_of_I(I) = V_b(I) + I*((1-x)*R_comp +
           R_mesh)`` (``cathode.py``), so the integrand that
           ``advance_circuit_current_driven`` integrates,

               f(I) = (V_src - I*x*R_comp - vdis_of_I(I)) / L
                    = (V_src - I*R_comp - V_b(I) - I*R_mesh) / L

           contains no ``x``. The loop current responds only to the TOTAL
           ``R_comp`` plus ``R_mesh``, and nothing physical consumes the
           REPORTED ``V_dis`` (the beam energy comes from ``phi_c``, off the
           cathode solve). The loop current, ``V_b``, ``phi_c``, the beam
           deposition and the whole plasma trajectory are all x-independent.
        2. OBSERVATIONALLY ACTIVE. ``x`` does set the REPORTED ``V_dis``, at
           ``dV_dis/dx = -I*R_comp``, and reported ``V_dis`` is a scored
           observable. So ``x`` changes what a run reports without changing
           what it simulates.
        3. THEREFORE A CALIBRATION KNOB. It decouples the total series
           resistance from the reported ``V_dis``, which is what lets the
           total ``R_comp`` -- which genuinely does throttle the current --
           change while the ``V_dis`` comparison stays matched.

        Do NOT use it to represent real internal resistance. Resistance
        between the probe and the plasma does lower the current, and that is
        ``R_mesh_ohm``, which is genuinely additional resistance rather than a
        relabelling of ``R_comp``. Correspondingly there is no "fit
        ``R_comp`` for the current, then derive ``x``" recipe: ``V_dis`` pins
        the product ``x*R_comp``, the current pins the emission, and the
        internal resistance is bounded independently.

        Default ``1.0`` (all external, internal part 0) is bit-exact with the
        historical behaviour. Must be in ``[0, 1]``. Adopted values:
        ``scripts/production_stance_provenance.md``.
    R_mesh_ohm:
        Anode-mesh series resistance [Ohm], separate from ``R_comp`` and on the
        internal (plasma) side of the probe, so it is invisible to the V_dis
        formula. Physically the Mo anode-mesh wire, order 1 mOhm and rising
        with anode temperature; only a CONSTANT value is implemented, so any
        ``R_mesh(T_anode)`` dependence must be approximated by that constant.
        Unlike ``R_comp_partition`` this is a real series resistance and does
        reach the loop current. Default ``0.0`` is bit-exact. Must be ``>= 0``.
        Measured bounds: ``config_defaults_provenance.md``.
    eta:
        Anode-to-cathode area ratio.
    anode_radius_cm:
        Radius of the anode mesh disc [cm]. ``None`` (default) spans the
        chamber, giving the historical neutral transparency ``1 - eta``. A
        smaller disc opens the annulus around it to free neutral flow, so
        the face's neutral open fraction becomes ``1 - eta*(Ra/Rm)^2``.
        Heat transmission and Bohm collection keep the bare ``1 - eta`` /
        ``eta``; the disc must still cover the plasma channel (``Ra >= Rp``).
        Resolved geometry only.
    L_cath:
        Cathode-to-anode distance used by the cathode solver [cm].
    R_cath:
        Cathode radius used to compute cathode area [cm].
    C_bank_F:
        Effective capacitance of the discharge bank [F]. ``None`` (default)
        is the historical infinite bank. When set, the bank voltage starts at
        ``V_bank`` and drains by the drawn charge during drive phases
        (backward-Euler, folded into the circuit solve as a ``dt/C`` term on
        the effective resistance); the tail and floating phases leave it
        inert.

        Moves jointly with ``R_comp``: the two are determined together, so a
        configuration must not change one alone. Value, hardware bounds and
        provenance: ``config_defaults_provenance.md``.
    L_parasitic_H:
        Parasitic series inductance in the current-driven discharge circuit
        [H], in series with ``R_comp``. The loop current is advanced once per
        accepted step by TR-BDF2. It must be positive when cathode coupling is
        enabled.

        L is inert for the sigma-scored discharge quantities and shows up in
        the current-rise shape and the ignition time. Value, bracket and
        provenance: ``config_defaults_provenance.md``.
    cathode_warming_model:
        Slow evolution of the emitter surface temperature within a shot.
        ``"none"`` (default) holds ``T_s`` constant, so the
        emission ceiling -- and with it the discharge current -- saturates
        on the circuit timescale (~1-2 ms), where the measured current rises
        for ~15-20 ms. ``"power_balance"`` (M1b) uses
        imposed asymptote with the surface energy balance

            C_th dT_s/dt = P_heater + P_cathode_i
                           - eps*sigma*A*(T_s^4 - T_env^4)
                           - (I_eth_star/e)*(phi_wf + 2 k_B T_s)

        The last term is the emission cooling the relaxation model lacks:
        each *actually emitted* electron (``I_eth_star`` from the accepted
        solve, not the Richardson ceiling) carries away the work function
        plus its ~2kT_s of thermal energy. Space-charge clamping therefore
        suppresses cooling early (faster warm-up) and releases it near the
        ceiling (harder cap). ``P_heater`` is pinned by the pre-discharge
        equilibrium ``P_heater = eps*sigma*A*(T_base^4 - T_env^4)`` (open
        circuit => no net emission), so the heater is not a free parameter.
        The steady state -- and with it the plateau current -- becomes an
        *output* of the balance, independent of ``C_th``; the configured
        ``T_s`` is no longer an asymptote and only sets the static-model
        fallback. During floating phases the emitted electrons return to
        the surface, so the emission-cooling term is dropped there. The
        update is semi-implicit in the linearized loss (unconditionally
        stable for any ``C_th``), floored at ``cathode_env_T_K``, accepted
        steps only.
    cathode_Ts_base_K:
        Heater-maintained standby surface temperature [K] for
        ``cathode_warming_model = "power_balance"`` -- the temperature the
        cathode sits at before the discharge, i.e. an operational machine
        setpoint. Required when that model is on; also its initial condition.
        Per-run operating points live in
        ``run_mechanism_ladder.ES_OPERATING[es]["Ts_standby_K"]``. Note the
        degeneracy with ``C_R`` documented above: the two describe one flat
        direction, so a configuration must not move both.
    cathode_heat_capacity_J_per_K:
        Effective thermal mass of the *emitting layer* [J/K] for
        ``"power_balance"``. NB this is the thermal skin depth reached over
        the discharge (sqrt(alpha*t) ~ 0.3-0.5 mm of LaB6), not the disc's
        bulk heat capacity (~hundreds of J/K) -- it shapes only the ramp
        timescale; the steady state is independent of it.
    cathode_emissivity:
        Total hemispherical emissivity of the emitting surface for the
        radiation term (LaB6 ~0.7).
    cathode_rad_area_cm2:
        Radiating area [cm^2] for ``"power_balance"``. ``None`` (default)
        uses the cathode disc area ``pi*R_cath^2``.
    cathode_env_T_K:
        Environment temperature [K] the surface radiates against
        (chamber walls); negligible against T_s^4, kept for completeness.
    cathode_conduction_W_per_K:
        Conductance [W/K] from the emitting skin layer into the
        heater-held substrate at ``cathode_Ts_base_K`` -- the "heater
        maintains the lower end" restoring term,
        ``P_cond = G_cond*(T_s - T_base)``. Vanishes at standby, so the
        heater pinning is unchanged. **This term is what stabilizes the
        balance at the LAPD operating point**: without it (0, the
        pure-radiation limit) the bombardment feedback gain d(P_ion)/dT_s
        through the emission loop exceeds the radiation+emission stiffness and
        the discharge runs away to several times the physical current.
        Physical scale: quasi-static ``kappa*A/delta`` for LaB6 is ~10 kW/K at
        a 0.4 mm skin depth; the effective value over a ~20 ms transient is
        lower. This term sets the plateau surface-temperature rise, and the
        plateau *current* then follows from the balance. Adopted values:
        ``config_defaults_provenance.md`` and
        ``scripts/production_stance_provenance.md``.
    cathode_emission_profile:
        Radial structure of the thermionic emitter. ``"uniform"`` (default,
        historical) is a single-temperature disc, whose emission ceiling is a
        razor wall in the discharge V(I) curve -- the operating point riding
        that wall is what makes the circuit-coupled current/voltage noisy.
        ``"gaussian"`` gives the cathode a radial falloff: the
        emission-current footprint ``exp(-4 ln2 r^2/FWHM^2)``,
        Richardson-inverted into a local surface temperature profile. The
        implied centre-to-edge temperature drop of order 150-200 K softens the
        ceiling into a stable ramp. Use with the physical cathode ``R_cath``
        and keep ``Rp`` at the plasma-channel value.
    cathode_Ts_fwhm_cm:
        Emission-footprint FWHM [cm] for the gaussian profile. Value and
        measured range: ``config_defaults_provenance.md``.
    cathode_emission_annuli:
        Number of annuli discretizing the profile.
    cathode_solver_model:
        The live ``"current_driven"`` formulation carries the loop current
        ``I_loop`` (and the
        bank voltage when ``C_bank_F`` is set) as explicit solver state,
        advanced once per *accepted* step by a TR-BDF2 step of
        ``dI/dt = (V_src − I·R_comp − V_dis(I))/L``; each stage is a
        bracketed scalar root-find over the monotone current-driven sheath
        solve (`solve_idriven`), which is well-posed at the ceiling.
        Within a step every RHS call sees the frozen ``I_loop``. Requires
        ``L_parasitic_H > 0`` and a single
        cathode (``TwinCathode`` raises). Floating phases route to the
        historical open-circuit solve. The trapezoidal circuit fold and
        its guards are inert in this mode.
    cathode_phi_c_cap_V:
        Physical ceiling [V] on the *net* cathode sheath drop in the
        current-driven solve. An imposed current the sheath cannot carry
        below it returns the ceiling solution tagged
        ``regime = "capability_limited"`` with the correspondingly large
        V_dis, and the circuit ramps the current down at ~V/L — the
        well-posed version of the inductive kick.
    cathode_Rp_model:
        How the cathode solver's parallel plasma (gap) resistance ``R_p`` is
        built. ``"sample"`` (default, historical) is the solver's internal
        ``R_p = L_cath / (pi R_cath^2 sigma_par(Te_sample))`` with the
        Spitzer conductivity evaluated at the *one* cathode-adjacent sampled
        cell -- which underestimates the resistance of a gap colder than
        that sample (eta_Spitzer ~ Te^-3/2), and so can bias V_dis(t) over a
        discharge in which the gap cools away from the sampled cell.
        ``"resolved_gap"`` integrates ``R_p = sum_k dz_k / (sigma_par(Te_k)
        * A_k)`` over the resolved cathode-anode gap cells with each cell's
        own Te and plasma-channel area -- the same per-cell weighting the
        ohmic deposition already uses. Fed to the sheath evaluator through an
        effective ``DeviceConfig.R_cath`` chosen so the
        solver's internal formula reproduces the integrated value exactly
        (``R_cath`` is used nowhere else in the solve). Requires a single
        cathode (``TwinCathode`` raises: one shared DeviceConfig
        cannot carry two gaps sampled at different Te). NB with ``Rp !=
        R_cath`` the two models differ even for a uniform gap -- conduction
        through the plasma channel, not the cathode disc.
    b_beam_excitation:
        Scale on the neutral-excitation cross section added to the primary
        beam's inelastic channels. ``0`` (default) is the historical beam:
        ionization-only attenuation and every deposited eV heating the
        plasma. Nonzero adds beam-driven neutral excitation, whose ~21-22 eV
        per event radiates away promptly as He I light (the
        ``beam_excitation_radiation`` term) and whose cross section shortens
        the beam's inelastic deposition length. What the scale multiplies
        depends on ``beam_excitation_model``: under ``"2p_scalar"`` it scales
        the 2^1P cross section alone, so ``1.0`` books that channel and a
        larger value stands in for the rest of the singlet manifold; under
        ``"manifold"`` it scales the measured manifold sum, so it is a pure
        sensitivity multiplier whose benchmark value is ``1.0``.
        Triplet/metastable excitation is exchange-driven and collapses above
        ~50 eV, so it is deliberately absent. He-only.
    beam_excitation_model:
        Which cross-section set the beam's excitation channel uses.
        ``"2p_scalar"`` (default, historical): the single 2^1P cross section
        with ``beam_excitation_energy_eV`` radiated per event.
        ``"manifold"``: the summed Ralchenko et al. (2008) singlet manifold
        (fitted n <= 4 levels plus the Eq. (5) n >= 5 tail,
        ``vars._coeff.He_singlet_manifold``) with the energy-weighted mean
        radiated energy per event computed at the beam energy. Over 60-180 eV
        the manifold gives 1.65-1.75x the 2^1P events and 1.71-1.81x its
        radiated power, so it is knob-free where ``"2p_scalar"`` needs
        ``b_beam_excitation`` to stand in for the missing levels. The
        current-driven sheath consumes the channel through
        ``beam_excitation_channel``.
    beam_excitation_energy_eV:
        Threshold and radiated energy per beam excitation event [eV]
        (the 2^1P excitation energy). Used by ``"2p_scalar"`` only; under
        ``"manifold"`` the thresholds and radiated energies come from the
        manifold registry and this key is inert.
    beam_deposition_model:
        How the primary beam deposits along the column.
        ``"beer_lambert"`` (default, historical): single-event absorption
        over the mixed Coulomb/inelastic profile (``l_b_profile`` +
        ``beam_absorption_weights``). ``"csda"``: the deterministic
        slowing-down module (``funcs/_beam_deposition.deposit_beam``, a pure
        function of the beam and the column — B2):
        primaries survive multiple inelastic events, per-cell ionization/
        excitation/heating/radiation come from the integrated ray, and the
        sheath solve's bypass fraction is driven by the module's gap
        transmission through an effective attenuation cross section at the
        launch cell (exact when the transmission is at or below the frozen
        solve's Coulomb-only ceiling ``exp(-L_cath/l_bi)``; clamped to the
        ceiling otherwise). Under ``"csda"`` the ``b_beam_excitation`` and
        ``beam_excitation_model`` knobs are inert — the module always uses
        the measured manifold, knob-free. He-only.
    beam_coulomb_model:
        Coulomb drag closure for the CSDA module (inert under
        ``"beer_lambert"``). ``"fast_electron"`` (default): the physical
        stopping power ``dE/dx = 2 pi e^4 n_e lnL / E`` (~30 m e-fold at
        150 eV, n_e = 5e12). ``"legacy_tau_ei"``: the historical
        ``v(E) tau_ei(Te)`` form (~1 m; overestimates classical drag ~30x).
        Both parameter-free.
    beam_anomalous_model:
        Anomalous (beam-plasma instability) drag for the CSDA module (inert
        under ``"beer_lambert"``). ``"none"`` (default) or
        ``"quasilinear"``: mean-energy relaxation over
        ``l_QL = (n_e/n_b)(v_b/w_pe) ln(n_e/n_b)`` (~5-10 cm at production
        parameters), energy to local electron heating — the
        Langmuir-turbulence picture behind primaries not surviving
        downstream. Weak-beam domain only (returns no drag when
        ``n_b >= n_e/10``); parameter-free.
    beam_product_transport:
        Where the CSDA ray's event PRODUCTS deposit (inert under
        ``"beer_lambert"``, which never launches the module; selecting the
        non-default value there raises). ``"local"`` (default, historical,
        bit-exact): the mean secondary energy ``<W_sec>`` per ionization and
        the primary's terminal sub-threshold residual are banked as plasma
        heating in the cell where the event happened — perfect local
        confinement. ``"nonlocal"``: each product
        instead walks along B from its birth cell on its own mini-CSDA
        Coulomb slowing integral (the SAME ``beam_coulomb_model`` the primary
        uses), depositing until it thermalizes at the local Maxwellian mean
        ``1.5*Te`` or leaves an end, where its remaining energy is booked to
        the new END LEDGER and leaves the system. Secondaries split 50/50
        into +z/-z half-weight walks (broadly isotropic OPB emission);
        the terminal residual keeps the primary's direction. Motivation: at
        breakdown both products sit BELOW every He inelastic threshold and
        Coulomb-couple at ~1 eV per machine pass (n_e ~ 1e10), i.e. they are
        near-collisionless along B exactly where ``"local"`` assumes perfect
        confinement. The end ledger also books the transmitted PRIMARY's
        ``Gamma_t*E_t``, closing a standing hole (computed since B1, never
        banked). Parameter-free; no pitch-angle diffusion and no elastic
        e-He channel (~5 meV/collision) — stated limitations. ENERGY-ONLY in
        v1: ionization events, the particle rows and the circuit currents are
        identical in both modes. The two settings are a bracket, not a
        prediction, so a result must state which one it used.
    heating_anomalous_transport:
        Where the CSDA ray's ANOMALOUS (quasilinear) heating lands (inert
        under ``"beer_lambert"``, and requires an active anomalous channel;
        selecting the non-default value without either raises). ``"local"``
        (default, historical, bit-exact): the QL drag is banked as
        instantaneous local bulk electron heating in the cell that drove it —
        the Langmuir turbulence Landau-damps near where it grows, so its
        energy is handed to the background there. ``"tail_walk"``:
        quasilinear diffusion does not warm a
        Maxwellian in place, it fills a fast-tail plateau first, and at
        breakdown densities a tail electron is collisionally decoupled
        (Coulomb range ~km at n_e ~ 1e10, hundreds of machine lengths) and
        free-streams along B. Under ``"tail_walk"`` each cell's QL power is
        withheld and carried by tail electrons at
        ``heating_anomalous_tail_energy_eV``, launched 50/50 along +-B and
        walked on the SAME closed-form Coulomb machinery the
        ``beam_product_transport`` product walks use (the ray's own
        ``beam_coulomb_model``, the same ``1.5*Te`` thermalization floor) — no
        new physics parameters beyond the tail energy. Energy still hot at a
        domain end goes to a SEPARATE tail end ledger (kept apart from the
        product ledger so both stay readable when the two closures are on
        together) and leaves the system.
        Motivation: this is an effective heating lag plus an end loss during
        exactly the e-folds that set the avalanche growth rate. ``"tail_walk"``
        is the FREE-ESCAPE bound (no sheath/ambipolar throttle), so
        {local, tail_walk} is a bracket, not a prediction, and a result must
        state which one it used. ENERGY-ONLY: ionization events, the particle
        rows and the circuit currents are identical in both modes.
    heating_anomalous_tail_energy_eV:
        QL plateau energy ``E_tail`` [eV] the tail electrons are launched at.
        **Read ONLY under ``heating_anomalous_transport="tail_walk"``** --
        inert otherwise. Must be finite and > 0. It sets the walkers' Coulomb
        range and therefore how far the QL power travels before thermalizing;
        the equivalent tail flux is ``P_QL / E_tail``, so the power carried is
        independent of it. The plateau energy is a kinetic quantity a fluid
        model cannot pin, so this is an ASSUMED value and a run that uses it
        must report a bracket rather than a single number. Arms and values:
        ``config_defaults_provenance.md``.
    beam_clump_fraction:
        Fractional-coverage beam-neutral closure (default 0.0 = OFF, bit-exact).
        The fresh gas puff is a dense, SPOTTY cloud sitting on the uniform
        equilibration seed (the residual inter-shot background), so the beam is
        BIMODAL: a fraction ``f`` of its flux meets dense clumps (short l_b ->
        deposits locally near the source, seeding the sonic accumulation front)
        while ``1-f`` streams through the thin gaps at the background density
        (long l_b -> penetrates to the far end, the fast interferometer
        "pedestal"). The radially-uniform single-l_b deposition is neither. When
        ``f>0`` (and ``beam_clump_enhancement>1``) the CSDA ray is split into a
        clump ray (flux ``f*Gamma0`` against ``nn*chi``) and a gap ray (flux
        ``(1-f)*Gamma0`` against the background ``nn``), and the two per-cell
        depositions are summed; the beam stays energy-limited so totals are
        bounded. ``f`` is a physical cloud area-coverage fraction, ``[0,1)``.
    beam_clump_enhancement:
        Clump neutral-density enhancement ``chi`` over the local background for
        the clump ray (default 1.0 = OFF). ``nn_clump = chi*nn`` shortens the
        clump-ray l_b, controlling how LOCALIZED the clump deposition is (higher
        chi -> shorter deposition -> stronger front seed). Represents the fresh
        puff's density above the equilibration seed; ``>= 1``. Requires
        ``beam_clump_fraction>0`` to act (both default to the off/uniform value).
    beam_deposition_smoothing_cm:
        Physical Gaussian width [cm] for a conservative spatial smoothing of the
        CSDA beam source terms (ionization, excitation, radiated, and heating
        densities) before they enter the fluid. ``0.0`` (default) is OFF and
        bit-exact. A nonzero width redistributes each cell's beam deposition to
        its axial neighbours with a mass/energy-conserving column-normalized
        Gaussian over the live plasma cells, so the deposited totals are
        unchanged. Because the width is a FIXED length (not a cell count) the
        deposition profile is mesh-convergent, which removes the grid-scale
        current-step artifact where the beam range crossing a cell boundary
        kicks the sheath solve. CSDA only (inert under ``beer_lambert``). Must
        be ``>= 0``.

    Values and their provenance: ``config_defaults_provenance.md``.
    """
    # These defaults ship the full cathode stack: power_balance warming, CSDA
    # beam + quasilinear anomalous drag, gaussian emission profile, ads_des
    # surface state, and presheath sample smoothing. Rp_model stays "sample".
    #
    # The circuit values here are mirrored EXACTLY by the campaign stance in
    # ``scripts/compare_sim1d_es1.PARAM_OVERRIDES``; the duplication is
    # deliberate (that dict is the campaign stance record, and removing the
    # pins would change resolution order for the other run drivers). Values,
    # provenance classes and bars: config_defaults_provenance.md.
    return {
        # --- ACTIVE: circuit hardware ---
        # V_bank here is the SUPPLY SETPOINT, which is a DIFFERENT QUANTITY
        # from the measured pre-shot open-circuit bank voltage and is
        # deliberately NOT replaced by it. The per-run open-circuit readings
        # live in ``scripts/run_mechanism_ladder.ES_OPERATING``; any run that
        # means the machine rather than the dial must set V_bank from there.
        "V_bank": 180.0,
        # T_s is only the static-model fallback under power_balance (the input
        # is cathode_Ts_base_K).
        "T_s": 1998.15,
        # phi_wf is the contaminated SHOT-START work function read by ads_des.
        "phi_wf": 2.869,
        "C_R": 29.0,
        "R_comp": 7.2244e-3,
        "R_comp_partition": 1.0,
        "R_mesh_ohm": 0.0,
        "L_parasitic_H": 8.1e-6,
        "C_bank_F": 9.5,
        # Gated fluid<->circuit Picard (only read when the
        # coupled_circuit_picard flag is on): relative loop-current change that
        # triggers a re-run, and the iteration cap.
        "circuit_picard_tol_rel": 1.0e-2,
        "circuit_picard_max_iter": 3,
        "eta": 0.358,
        "anode_radius_cm": None,
        "L_cath": 50.0,
        "R_cath": 15.0,
        # --- ACTIVE: beam deposition (CSDA production stack; b_beam_excitation
        # + beam_excitation_model are INERT under csda -- the module uses the
        # measured manifold, knob-free -- and matter only for the beer_lambert
        # A/B arm) ---
        "beam_deposition_model": "csda",
        "beam_coulomb_model": "fast_electron",
        "beam_anomalous_model": "quasilinear",
        # Non-local product transport: DEFAULT OFF (bit-exact).
        "beam_product_transport": "local",
        # QL heating locality: DEFAULT OFF (bit-exact). The tail energy is
        # inert under "local".
        "heating_anomalous_transport": "local",
        "heating_anomalous_tail_energy_eV": 75.0,
        "beam_clump_fraction": 0.0,
        "beam_clump_enhancement": 1.0,
        "beam_deposition_smoothing_cm": 0.0,
        "b_beam_excitation": 0.0,
        "beam_excitation_model": "2p_scalar",
        "beam_excitation_energy_eV": 21.218,
        # --- ACTIVE: cathode warming (power_balance) ---
        "cathode_warming_model": "power_balance",
        "cathode_Ts_base_K": 1910.0,
        "cathode_heat_capacity_J_per_K": 120.0,
        "cathode_conduction_W_per_K": 1200.0,
        "cathode_emissivity": 0.7,
        "cathode_rad_area_cm2": None,
        "cathode_env_T_K": 300.0,
        # --- ACTIVE: gaussian emission profile (radial falloff) ---
        "cathode_emission_profile": "gaussian",
        "cathode_Ts_fwhm_cm": 28.0,
        "cathode_emission_annuli": 10,
        "cathode_Rp_model": "sample",
        "cathode_solver_model": "current_driven",
        "cathode_phi_c_cap_V": 1000.0,
        # Surface-state coverage model:
        # "ads_des" evolves contaminant coverage theta with
        # dtheta/dt = k_ads(1-theta) - [nu0 e^(-E_des/kTs) + sigma Gamma_i] theta
        # and substitutes phi_eff = phi_clean + (phi_wf - phi_clean)*theta
        # everywhere phi_wf is read (emission, Schottky, cooling, gaussian
        # inversion -- every consumer reads the one substituted
        # value, never a mix of phi_eff and phi_wf). phi_wf keeps its
        # meaning as the contaminated SHOT-START value; the clean floor is
        # the per-shot-accessible depth of the re-adsorbed layer, not the
        # literature clean-LaB6 value. In-shot the ion-stimulated term
        # dominates (fluence-cleaning limit); adsorption and thermal
        # desorption are carried for the between-shot cycle map and default to
        # zero (inert).
        # --- ACTIVE: ads_des surface state (in-shot ion-stimulated cleaning;
        # the between-shot ads/desorption params default to zero, inert) ---
        "cathode_surface_model": "ads_des",
        "cathode_phiwf_clean_eV": 2.809,
        "cathode_cleaning_sigma_cm2": 3.5e-16,
        # Ion-stimulated desorption threshold [eV]: scales sigma by the
        # near-threshold Bohdansky factor (1-(Eth/E)^(2/3))(1-Eth/E)^2 at the
        # per-ion energy E = P_cathode_i/I_i. None = the energy-independent
        # fluence limit.
        "cathode_cleaning_E_th_eV": 20.0,
        "cathode_ads_rate_per_s": 0.0,
        "cathode_desorption_prefactor_per_s": 0.0,
        "cathode_desorption_energy_eV": 3.0,
        # Directed neutral recycle jets: with
        # the neutral_momentum flag on, the surface recycle fluxes carry
        # directed momentum into M_n instead of rebirthing at rest.
        # Momentum-only first pass -- the reflected atoms' kinetic energy
        # is not booked (neutrals have no energy field; standing M2
        # convention). (R_N, R_E) are the particle and energy reflection
        # coefficients of the surface -- literature quantities, not fit knobs
        # (cathode = He->LaB6, anode = He->Mo; see
        # config_defaults_provenance.md for the values and their brackets).
        # The cathode channel splits R_N fast backscatter at
        # sqrt(2 R_E (phi_c + Ti)/m) + (1-R_N) directed effusion at the
        # surface T_s; the anode channel is backscatter-only (wire
        # re-emission has no net axial direction), per collected side, at
        # the solve's phi_a.
        "cathode_neutral_jet": False,
        "cathode_jet_R_N": 0.5,
        "cathode_jet_R_E": 0.2,
        "anode_neutral_jet": False,
        "anode_jet_R_N": 0.5,
        "anode_jet_R_E": 0.25,
        # Sensitivity arm: debit the cathode surface's ion heating by the
        # reflected-energy fraction (power_balance receives
        # (1 - R_E) * P_cathode_i). Off by default, since the jet's first pass
        # is momentum-only. Requires cathode_neutral_jet.
        "cathode_jet_surface_debit": False,
        # Mesh momentum accommodation for the evolved wind: the momentum
        # the anode wires intercept lands on the anode structure instead
        # of staying in the gas (the open-area throttle alone leaves the
        # gap recirculation artificially elastic). Requires
        # neutral_momentum and anode faces.
        "neutral_mesh_accommodation": False,
        # Electrode sample smoothing: the sheath solve's inputs are the
        # instantaneous cathode-cell and anode-flank (n, Te) cell averages,
        # which carry grid-level explicit-step noise the physical supply
        # integrates over -- the presheath delivers flux averaged over an ion
        # transit time tau ~ l_cell/c_s. Because V(I) is nearly flat, that
        # sampling noise amplifies into per-solve V_b and leaks into physics
        # through the beam energy (phi_c per solve) and the trapezoidal fold's
        # EMF residual. The anode sample matters equally: J_i_a and Te_anode
        # enter the residual through tau_a*ln(1 + J_anode/J_i_a), so anode-side
        # noise flaps phi_a and drags phi_c with it. "presheath" computes
        # tau = l_cell/c_s(Te_ema) per sampled cell, so it is a derived
        # physical timescale rather than a knob; a float is a fixed tau [s];
        # None disables bit-exactly. EMA updates on accepted steps only.
        "cathode_sample_smoothing": "presheath",
    }


def physics_fit_defaults():
    """Return auxiliary physical fit and neutral transport defaults.

    heat_picard_iterations:
        Picard iterations used to evaluate the conductivity in the implicit
        heat-conduction substep. Zero freezes the Braginskii conductivity
        (roughly proportional to T^2.5) at the incoming state, which is
        first-order accurate in dt however accurate the substep scheme is, so
        ``crank_nicolson`` and ``tr_bdf2`` cannot express their second order.
        A positive value re-evaluates the conductivity at the scheme's own flux
        evaluation point until the temperature converges, at the cost of one
        extra banded solve per species per iteration. Note that Lie splitting
        in the operator-split path is an independent first-order term, so a
        converged Picard alone does not make the whole step second-order.
    heat_picard_tol:
        Relative temperature-change tolerance ending the Picard iteration early.
    ln_lambda_min:
        Minimum Coulomb logarithm used by transport and exchange estimates.
    Tn_K:
        Neutral gas temperature used by molecular-flow neutral exchange [K].
    neutral_exchange_coeff_cm3_s:
        Constant neutral exchange coefficient for the constant model [cm^3/s].
    neutral_clausing_scale:
        Scale factor applied to molecular-flow Clausing conductance.

    Values and their provenance: ``config_defaults_provenance.md``.
    """
    return {
        # --- ACTIVE ---
        # The third leg of the 2nd-order operator-split defaults: a positive
        # value is required for tr_bdf2 + strang to express second order.
        "heat_picard_iterations": 2,
        "heat_picard_tol": 1e-10,
        "ln_lambda_min": 1.0,
        "Tn_K": 300.0,  # single cold-gas neutral temperature (Phelps T_eff)
        # --- INERT under these defaults ---
        # Only the "constant" neutral_exchange_model reads this (the default is
        # "knudsen"):
        "neutral_exchange_coeff_cm3_s": 1.0e5,
        # molecular-flow Clausing conductance -- molecular_flow is deprecated:
        "neutral_clausing_scale": 1.0,
    }


def timestep_defaults():
    """Return explicit timestep, growth, and retry-control defaults.

    cfl:
        CFL fraction for wave/advection timestep constraints.
    density_dt_fraction:
        Fractional density-change limit for source/reaction timestep estimates.
    neutral_dt_fraction:
        Fractional neutral-density change limit for neutral source estimates.
    heat_dt_fraction:
        Fractional thermal-energy change limit for heat/source estimates.
    dt_min:
        Minimum allowed timestep [s].
    dt_max:
        Maximum allowed timestep [s].
    max_steps:
        Maximum accepted timesteps for a run. Zero means unlimited.
    max_steps_action:
        What ``run()`` does when ``max_steps`` is reached before ``t_end``.
        ``"raise"`` (default, historical behavior) raises RuntimeError and the
        in-progress trajectory is lost; ``"stop"`` ends the run cleanly and
        returns the partial trajectory with ``run_status =
        "max_steps_reached"`` (a completed opt-in run carries ``run_status =
        "completed"``) so the caller can inspect and save it.
    surface_loss_floor_exempt_rtol:
        Relative threshold for the floor-aware drain exemption on the
        "surface_loss" (resolved electrode/source bundle) timestep bound,
        active only when the ``surface_loss_floor_exempt`` flag is on. A cell
        whose electron (ion) energy margin above the per-cell floor energy
        ``3/2 n Te_floor`` (``3/2 n Ti_floor``) is at most this fraction OF
        that floor energy is excluded from the drain-margin bound: the
        accept-time floor clip re-pins the cell every step while the sink
        keeps draining, so the bound re-trips forever and dt collapses (the
        diagnosed afterglow crawl). The threshold is relative to the floor
        energy, not an absolute magnitude. The scale it has to separate: a
        pinned cell HOVERS at a small positive relative margin -- the clip
        plus one step of re-heating residue, not float round-off -- while a
        healthy drained cell sits orders of magnitude above that. A cell
        whose drain dominates its heating self-drives its hover margin below
        any fixed threshold within a few steps (hover ~ heating*dt shrinks
        with dt), so the exemption engages exactly for genuinely pinned
        cells. Readmission is knife-edge: any margin above the threshold
        re-admits the cell immediately (no hysteresis). Must be in (0, 1)
        when the flag is on.
    adaptive_retries_enabled:
        Enables retrying a rejected step with a smaller timestep.
    max_step_retries:
        Maximum retry attempts for one accepted step.
    dt_reject_factor:
        Timestep multiplier applied after a rejected attempt.
    dt_growth_enabled:
        Enables limiting timestep growth between accepted steps.
    dt_growth_factor:
        Maximum timestep growth factor between accepted steps.
    max_density_step_fraction:
        Optional accepted-step density fractional-change guard. Zero disables it.
    max_neutral_step_fraction:
        Optional accepted-step neutral fractional-change guard. Zero disables it.
    max_energy_step_fraction:
        Optional accepted-step thermal-energy fractional-change guard. Zero
        disables it.
    drag_dt_fraction:
        Maximum ion-neutral drag relaxation fraction per explicit step.
        This was formerly available only through an unregistered
        ``dict.get`` fallback.

    Values and their provenance: ``config_defaults_provenance.md``.
    """
    return {
        "cfl": 0.4,
        "density_dt_fraction": 0.25,
        "neutral_dt_fraction": 0.25,
        "heat_dt_fraction": 0.25,
        "dt_min": 1e-10,
        "dt_max": 1e-4,
        "max_steps": 0,
        "max_steps_action": "raise",
        "surface_loss_floor_exempt_rtol": 1e-3,
        "adaptive_retries_enabled": True,
        "max_step_retries": 8,
        "dt_reject_factor": 0.5,
        "dt_growth_enabled": True,
        "dt_growth_factor": 1.25,
        "max_density_step_fraction": 0.0,
        "max_neutral_step_fraction": 0.0,
        "max_energy_step_fraction": 0.0,
        "drag_dt_fraction": 0.5,
    }


_PARAMETER_DEFAULT_GROUPS = (
    initial_condition_defaults,
    geometry_defaults,
    floor_defaults,
    neutral_source_defaults,
    timing_defaults,
    output_defaults,
    model_mode_defaults,
    fudge_factor_defaults,
    cathode_defaults,
    physics_fit_defaults,
    timestep_defaults,
)


def build_input_dict_template_1d():
    """Compose the public flat input-default dictionary from grouped defaults."""
    input_dict = {}
    for defaults in _PARAMETER_DEFAULT_GROUPS:
        group = defaults()
        duplicate = set(input_dict).intersection(group)
        if duplicate:
            raise RuntimeError(
                f"duplicate LAPDSim1D defaults in {defaults.__name__}: "
                f"{sorted(duplicate)}"
            )
        input_dict.update(group)
    return input_dict


input_dict_template_1d = build_input_dict_template_1d()


# Flag defaults. Which flags a specific committed configuration pins, and why,
# is recorded in config_defaults_provenance.md and the per-config notes under
# scripts/.
input_flags_template_1d = {
    "Plasma": True,
    "TwinCathode": False,
    # Typed plasma topology, and rejection of raw invalid stages before any
    # floor projection.
    "active_plasma_topology": True,
    "raw_stage_validation": True,
    # Retained as a stale-config guard; False raises at construction.
    "resolved_boundaries": True,
    # End-vessel / magnetic-flare geometry. Presence gated in core.geometry:
    # all three end_expansion_* parameters are required when on and forbidden
    # when off. Bit-exact off.
    "end_expansion_geometry": False,
    # Thin annular apertures. Positions and clear radii are required together
    # when on and forbidden when off; the plasma channel stays open.
    "neutral_baffles": False,
    # Fixed-cell-size source region, so a mesh refinement study is not
    # self-confounding: the column between the anode face and
    # source_region_length_cm is meshed at exactly source_region_dz_cm
    # regardless of nx, which then refines only the far column, and the puff
    # role follows gas_puff_z_cm instead of the first column cell. Presence
    # gated in core.geometry in both directions (both parameters required when
    # on, forbidden when off) and incompatible with TwinCathode. Default OFF
    # and structurally bit-exact off.
    "source_fixed_grid": False,
    "heat_conduction": True,
    "implicit_heat_conduction": True,
    # Flux-limited electron heat conduction. The classical Spitzer-Harm flux
    # can exceed the free-streaming scale n*Te*v_the at resolved gap faces,
    # i.e. exceed the physical ceiling. When ON, the electron conductivity is
    # scaled per cell by lambda = q_sat/(q_sat+q_SH) (harmonic Cowie-McKee),
    # q_sat = heat_flux_limiter_f * n * Te * v_the -- so the flux caps at
    # free-streaming where gradients are steep and recovers Spitzer where they
    # are shallow. Electron only; ion conduction unchanged. Default OFF and
    # bit-exact off; a declared closure-family A/B instrument.
    "electron_heat_flux_limit": False,
    # Sonic front-filling closure, OFF by default: the mesh A/B found the front
    # to be a numerical artifact (its L1 activity and Rusanov numerical
    # diffusion vanish under refinement). OFF renders alpha_front inert.
    "front_flux": False,
    # Conservative hyperbolic core (R2):
    # kinetic-energy-preserving convective momentum flux, plus deposit of the
    # Rusanov (n,M) numerical kinetic-energy dissipation into ion internal
    # energy, plus a KEP pressure-work discretization -- so the closed-domain
    # total plasma energy K+Ee+Ei is conserved to machine precision.
    "hyperbolic_energy_consistent": True,
    # Characteristic material boundaries (R3): replace
    # the closed-reflecting-face + one-sided volumetric absorber at the
    # plasma-terminating (cathode/collector) surfaces with a one-sided
    # characteristic ghost-cell Bohm outflow -- the KEP/Rusanov flux evaluated
    # against a Bohm ghost state (n_se = n*presheath_alpha, u = c_s into the
    # wall, Te, Ti). Fixes a wrong-sign momentum so the wall is a net energy
    # sink instead of a kinetic source. Requires resolved geometry (absorbing
    # faces); rejects at construction otherwise.
    "characteristic_boundary": True,
    # Anode-mesh beam interception (R4): the CSDA beam
    # ray launches the full emitted flux Gamma0 = I_eth_star/e through the
    # whole column, so without this the fluid deposits the entire emitted beam
    # while the circuit books only the (1 - eta*beam_bypass_fraction) fraction
    # into the plasma. This adds the missing interception event at the
    # anode-face crossing: the mesh solid fraction eta of the flux surviving
    # the gap is removed (the anode surface takes I_bypass*V_b, the sheath
    # returns I_bypass*phi_a to the circuit) and only (1 - eta) transmits
    # downstream. Like beam_coulomb_model / beam_anomalous_model it is a csda
    # control: inert under beam_deposition_model="beer_lambert" (which never
    # launches the CSDA module) and where the resolved geometry has no anode
    # faces. Set False for the with/without-interception A/B.
    "beam_anode_interception": True,
    # DEPRECATED: 0D-artifact per-electrode surface-loss enables. The resolved
    # geometry's plasma-terminating (absorbing) faces are a geometry fact, not
    # a config toggle; these are never consumed and non-default use warns.
    # Retained at their canonical True so a default run is warning-free.
    "source_surface_loss": True,
    "end_surface_loss": True,
    "ion_neutral_drag": True,
    "ion_neutral_drag_cx_only": False,
    # Evolve axial neutral momentum M_n as a sixth conservative field:
    # the drag deposits its momentum into the
    # neutral wind instead of a closure, ionization/recombination exchange
    # momentum between species, and the wall/pump remove it. Mutually
    # exclusive with ion_neutral_drag_model="slip", whose closure is this
    # equation's own local steady state. Off => the 5-field state.
    "neutral_momentum": False,
    # Split the neutral density into plasma-column and annulus zones:
    # an optional conservative field nn_a
    # carries the annulus density and nn becomes the COLUMN density. Axial
    # Knudsen transport runs per zone (the annulus is a free conduit), the
    # zones exchange free-molecularly at the column surface, and the
    # plasma only ever absorbs column gas. Requires
    # neutral_exchange_model="knudsen" (the per-zone conductances have no
    # constant counterpart). Off => single-field
    # chamber-mean nn.
    "neutral_two_zone": False,
    "ion_neutral_thermalization": False,
    # Replace the drag + frictional-heating + elastic thermalization +
    # CX-cooling quartet with ONE moment-closed reduced ion-neutral collision
    # operator (Phelps He+/He isotropic+backscatter rates, T_eff=(Ti+Tn)/2).
    # Presence-gated: when ON the four legacy ion-neutral terms are forced to
    # zero and this single operator runs; when OFF it is a strict no-op.
    # He-only. Uses the single cold-gas Tn_K for the neutral temperature,
    # ending the Tn_K/Tn_fit term-by-term mix. With this ON the ad-hoc
    # b_ion_neutral_drag / sigma_in constant/cx_derived / slip closures are
    # superseded and DEPRECATED.
    "ion_neutral_moment_closure": True,
    # Gated fluid<->circuit Picard. The fluid step runs at a loop current
    # frozen over the step, then the circuit advances from the accepted plasma
    # -- a frozen-current lag that can fail to converge at the emission knee.
    # When ON, the accepted step is re-run (<= circuit_picard_max_iter times)
    # with the updated loop current whenever |dI/dt| is large (a driven phase
    # and the loop current moved more than circuit_picard_tol_rel), so
    # fluid+T_s+circuit share one self-consistent I_loop. Default OFF and a
    # strict no-op where the trigger does not fire (one pass == the sequential
    # advance, bit-exact). Incompatible with the kinetic neutral engine.
    "coupled_circuit_picard": False,
    "cathode_coupling": True,
    # Schottky barrier lowering in the *current-driven* sheath solve only:
    # the extracting sheath field lowers the
    # effective work function, tilting the emission ceiling into a sloped
    # line. It collapses the per-solve V_b two-state chatter into a steady
    # band and restores current that the gaussian emission profile's edge
    # cooling costs. Because it shifts the effective barrier, phi_wf and this
    # flag are only meaningful together: a configuration quoting phi_wf must
    # state this flag's value.
    "cathode_schottky": True,
    # kT_s-width thermal bridge across the SCL<->classical emission-release
    # corner, *current-driven* sheath solve only:
    # the emitted Maxwellian's kT_s energy spread smooths the razor
    # min(J_eth, J_crit) corner that turns boundary-cell Te noise into V_b
    # chatter. C1 blend with exact hard-branch reduction outside the window,
    # monotonicity of J_tot(psi) preserved by construction (convex combination
    # of branch slopes -- see funcs._cathode_solver_idriven._bridge_release).
    # Off => bit-exact hard branches.
    "cathode_emission_bridge": False,
    # Gates the neutral-only pre-drive phase. DELIBERATELY LEFT ON while
    # tau_neutral_prebreakdown defaults to 0.0: the duration alone decides
    # whether the phase runs, so ON + zero duration is already inert
    # (`_neutral_prebreakdown_duration` returns 0.0 either way, which is why
    # flipping this to False would be bit-exact and buys nothing). What it
    # would cost is the opt-in: a study that sets a positive
    # tau_neutral_prebreakdown would then get NO phase and no error -- exactly
    # the silent fallback the house rules forbid. Keeping it True leaves the
    # duration as the single sufficient control.
    "neutral_prebreakdown": True,
    "neutral_equilibration": True,
    "launch_plasma_after_equilibration": True,
    # Reuse a cached neutral-equilibration seed (the equilibrated nn/nn_a
    # profile) instead of re-running the ~1-min 100-cycle equilibration every
    # run. Default OFF and bit-exact off.
    # When ON, requires neutral_equilibration + launch_plasma_after_equilibration
    # ON and a neutral_seed_cache_dir (the signature-keyed seed DATABASE):
    # a miss (new neutral-flow config) equilibrates once and stores it. See
    # core/neutral_seed_cache.py and scripts/build_neutral_seed_cache.py.
    "use_cached_neutral_seed": False,
    # Floor-aware drain exemption on the "surface_loss" timestep bound
    # (the afterglow dt-collapse fix): cells pinned at the Te/Ti floor
    # (energy margin within surface_loss_floor_exempt_rtol of the per-cell
    # floor energy) are excluded from the drain-margin bound ONLY -- the
    # accept-time floor clip resets their margin to float residue every step,
    # so a persistent drain otherwise pins dt at dt_min indefinitely. One-sided
    # (all other bounds still govern the cell) and knife-edge (any real margin
    # re-admits the cell immediately). ON by default because a run that reaches
    # a floor-pinned afterglow otherwise cannot finish in finite time. Set it
    # False to recover the historical bound.
    "surface_loss_floor_exempt": True,
    "ionization_energy_cost": True,
    "icool": True,
    "ncool": True,
    "cx": True,
    "icool_recomb": False,
    "debug_checks": False,
}


def load_config(path):
    """
    Load 1D solver parameters and flags from a TOML file.

    The file may contain ``[params]`` and ``[flags]`` sections. Missing values
    fall back to ``input_dict_template_1d`` and ``input_flags_template_1d``.
    """
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return resolve_config(raw.get("params", {}), raw.get("flags", {}))


def default_config():
    """Return copies of the default 1D input dictionary and flags."""
    return dict(input_dict_template_1d), dict(input_flags_template_1d)


def resolve_config(params=None, flags=None):
    """Resolve caller overrides against the one authoritative default registry.

    Unknown keys fail at this boundary so misspelled or retired campaign
    controls cannot survive as silent metadata-only settings.
    """
    supplied_params = {} if params is None else dict(params)
    supplied_flags = {} if flags is None else dict(flags)
    if "Lz" in supplied_params:
        raise ValueError(
            "Lz belongs to the legacy lumped geometry, which has been "
            "removed; use the resolved geometry, or reproduce the old "
            "behaviour at tag legacy-final-2026-07-22"
        )
    unknown_params = sorted(set(supplied_params) - set(input_dict_template_1d))
    unknown_flags = sorted(set(supplied_flags) - set(input_flags_template_1d))
    if unknown_params or unknown_flags:
        details = []
        if unknown_params:
            details.append(f"params={unknown_params}")
        if unknown_flags:
            details.append(f"flags={unknown_flags}")
        raise ValueError(
            "unknown LAPDSim1D configuration keys (silent/inert controls are "
            f"forbidden): {', '.join(details)}"
        )
    resolved_params = dict(input_dict_template_1d)
    resolved_params.update(supplied_params)
    resolved_flags = dict(input_flags_template_1d)
    resolved_flags.update(supplied_flags)
    return resolved_params, resolved_flags


def config_manifest():
    """Return a machine-readable manifest of every registered default."""
    parameters = {}
    for defaults in _PARAMETER_DEFAULT_GROUPS:
        for name, value in defaults().items():
            parameters[name] = {
                "default": value,
                "source": defaults.__name__,
            }
    flags = {
        name: {
            "default": value,
            "source": "input_flags_template_1d",
        }
        for name, value in input_flags_template_1d.items()
    }
    return {
        "schema": "lapdsim1d-config-manifest-v1",
        "parameters": parameters,
        "flags": flags,
    }


def resolve_nn0(input_dict, input_flags):
    """Return configured or table-derived initial neutral density [cm^-3]."""
    nn0 = input_dict.get("nn0")
    if nn0 is not None:
        return nn0
    return lookup_nn0(input_dict["S_gp"], twin=input_flags["TwinCathode"])
