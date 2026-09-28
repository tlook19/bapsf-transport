"""The ``LAPDSim1D`` configuration surface: the ``input_dict`` and
``input_flags`` templates, TOML loading, and the lineage and identity a run
records.

Each ``*_defaults()`` function below returns one group of the template, and its
docstring is the authoritative statement of what every key in that group MEANS
-- the quantity, its units, sign convention, valid range, which term consumes
it, which flag gates it and what it raises. That is deliberately all a
docstring here says. A configured value's CLASS (measured, derived, fitted or
assumed), its honest bar, and the measurement or fit standing behind it are
recorded OUTSIDE this repository, because that record rests on measurement
memos and run artifacts a reader of this public repository cannot obtain. What
can be read here is the shipped value itself, in this file, and the value a
configuration adopts, in its own file under ``scripts/stances/``.
"""

import dataclasses
import hashlib
import json
import tomllib


def initial_condition_defaults():
    """Return defaults for the initial primitive state.

    The species is helium, unconditionally.

    ne0:
        Uniform initial plasma/electron density [cm^-3].
    initial_neutral_state:
        How the initial neutral state is established. One of:

        ``"equilibrate"`` (default): ``start_simulation()`` runs the pre-run
        puff/off neutral accumulation -- an inner neutral-only sim with
        ``Plasma`` and ``cathode_coupling`` off, for
        ``neutral_equilibration_cycles`` at ``neutral_equilibration_dt`` --
        seeds ``nn`` (and ``nn_a``) from its settled state, and proceeds into
        the plasma run.

        ``"equilibrate_only"``: the same accumulation, after which
        ``start_simulation()`` stops and returns the equilibration result
        itself. This is how the neutral-only seed is produced.

        ``"fill"``: no accumulation; the run starts from the uniform scalar
        ``nn0`` fill.

        ``"profile"``: the run starts from the per-cell ``nn0_profile`` (and
        optionally ``nn0_annulus_profile``), which supersedes the scalar
        ``nn0``.

        Calling ``run()`` directly under the two equilibrating values performs
        NO equilibration and WARNS rather than raising, since the run is well
        defined -- it starts from the direct ``nn0`` fill. The equilibrating
        values and ``"profile"`` each refuse ``restart_from`` at construction:
        a restart payload replaces the whole initial condition after
        construction, so either would be silently overwritten. Any other value
        raises at construction naming the four accepted.
    nn0:
        Uniform initial neutral density [cm^-3]. REQUIRED, except under
        ``initial_neutral_state = "profile"``, which supersedes it and requires
        ``None``; a ``None`` reaching ``resolve_nn0`` any other way raises,
        the frozen gas-puff table that used to fill one in being retired.

        This is the DIRECT-RUN fill only. The equilibrated path does NOT read
        this value: ``run_neutral_equilibration`` pins its inner sim's start
        at 1e8 and overwrites nn with the equilibrated profile, so the two
        paths are decoupled and this default can move without disturbing any
        equilibrated run. Since ``initial_neutral_state`` ships at
        ``"equilibrate"``, the uniform value is a PLACEHOLDER that no shipped
        configuration reads -- equilibration is the convention for the fill a
        run starts from.
    nn0_profile:
        PER-CELL initial neutral density [cm^-3]: a sequence of length ``nx``
        (the grid's cell count), every entry finite and ``> 0``. Read ONLY
        under ``initial_neutral_state = "profile"``, and REQUIRED by it; under
        any other value it must be ``None`` or construction raises.

        Supplied as VALUES, not as a shape: these are the absolute densities
        the run starts from, cell by cell, and nothing rescales or normalizes
        them. This is the externally-computed-profile hook for the INITIAL
        CONDITION -- the solver does no file I/O, so a hypothesized axial fill
        is built outside and passed here.

        It supersedes the scalar ``nn0`` for BOTH zones, so ``nn0`` must be
        ``None`` under ``"profile"`` -- a non-``None`` scalar there raises
        rather than establishing a silent precedence. ``resolve_nn0`` is not
        consulted on that path.
    nn0_annulus_profile:
        PER-CELL initial ANNULUS neutral density [cm^-3]: same length,
        finiteness and positivity rules as ``nn0_profile``. Read ONLY under
        ``initial_neutral_state = "profile"``.

        OPTIONAL under ``"profile"``: omitted, the annulus starts at ``nn0_profile``,
        which is the shaped form of the shipped convention that both zones
        start at the same fill density. Supplying it addresses the two zones
        separately, which is what a construction that routes its inventory
        radially needs.
    Te0:
        Uniform initial electron temperature [eV].
    Ti0:
        Uniform initial ion temperature [eV].
    u0:
        Uniform initial axial plasma velocity [cm/s].
    Tn_fit:
        DEPRECATED; superseded by the single cold-gas ``Tn_K``. Was the neutral
        collision temperature used by the legacy IAEA reaction-rate fits and the
        legacy ion-neutral drag/thermalization/CX quartet -- all retired in
        favour of the Phelps moment-closed ion-neutral collision operator, so
        it is inert. The deferred M_n wall accommodation should read
        ``Tn_K``.
    """
    return {
        # --- ACTIVE (production) ---
        "ne0": 1e9,
        "initial_neutral_state": "equilibrate",
        # Pre-shot neutral background for DIRECT runs. The equilibrated path
        # never reads this (see the docstring above).
        "nn0": 2.0e13,
        # Shaped initial neutral fill (initial_neutral_state = "profile"). Both
        # are None on every shipped configuration: a per-cell IC has no default
        # shape to inherit, and the route's whole content is what the caller
        # computed outside.
        "nn0_profile": None,
        "nn0_annulus_profile": None,
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
        Number of resolved column cells. On the single-cathode layout it
        counts only the *far* column cells, between the fixed source region's
        end and the end wall (or the mirror face); on the ``TwinCathode``
        layout, the far cells on EACH side of the mid-plane, so the twin
        column carries ``2 nx`` far cells.
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
    end_wall_length_cm:
        Length of the end wall cell at the non-cathode end (single-cathode
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
    plasma_radius_profile_cm:
        PER-CELL effective plasma flux-tube radius [cm]: a sequence with one
        entry per MESH cell (``geometry.cells`` -- the plenum, gap, column and
        end cells, not just the ``nx`` column cells), every entry finite and
        ``> 0``, or ``None`` for the uniform ``pi Rp^2`` column. Its presence
        is what arms the prescribed per-cell geometry and the quasi-1D
        flux-tube terms that come with it.

        It replaces the uniform scalar ``Rp`` cell by cell, so the plasma
        cross-section is ``pi r(z)^2``, the cell volume ``pi r(z)^2 dz``, and
        the face area the average of the two adjacent cells -- the same
        expressions the uniform column uses, evaluated on a vector. A profile
        holding ``Rp`` in every cell is therefore bit-identical to no profile
        at all.

        The quantity the profile prescribes is the AREA ``A(z)`` (the flux-tube
        variable, ``A B = const``); the radius ``sqrt(A/pi)`` is how it is
        supplied, because that parameterization is what makes the constant
        profile exact rather than exact-to-a-rounding. Any conversion from a
        solved ``B(z)`` happens outside the solver, which does no file I/O.

        Supplied as VALUES, not as a shape: nothing rescales or normalizes
        them, and no cell is masked by role. Every entry must satisfy
        ``pi r^2 <= `` the local vessel open area, since the column zone
        cannot be larger than the chamber holding it (the two-zone annulus
        volume ``V_ann = Vm - Vp`` would go negative and be clipped to zero
        silently).
    machine_radius_profile_cm:
        PER-CELL vessel/neutral radius [cm], the same per-mesh-cell form and
        the same finiteness/positivity rules as ``plasma_radius_profile_cm``.
        Read ONLY with ``plasma_radius_profile_cm`` supplied (without it,
        construction raises), where it is OPTIONAL: omitted, every cell keeps
        the scalar ``Rm``. It replaces that scalar cell by cell, setting the
        neutral open area ``pi Rm(z)^2``, the neutral cell volume, and the
        hydraulic radius that sets the free-molecular face conductance -- so a
        vessel whose bore STEPS partway along a cell block is expressible,
        which a single ``Rm`` is not.

        Composes with the annular-duct and support-rod reductions rather than
        overriding them: an obstruction cell keeps its open area
        ``pi (Rm(z)^2 - Rcs^2)`` and hydraulic radius ``Rm(z) - Rcs``, and a
        plenum keeps ``pi (Rm(z)^2 - Rsup^2)``. Every entry must be ``>=`` the
        local ``plasma_radius_profile_cm`` entry; the vessel cannot be
        narrower than the plasma it contains.
    plasma_area_max_vessel_fraction:
        Optional ceiling on the prescribed plasma area as a fraction of the
        local vessel open area, in ``(0, 1]``. ``None`` (the default) applies
        no ceiling. Read ONLY with ``plasma_radius_profile_cm`` supplied
        (without it, construction raises).

        When set, each cell's plasma area is clipped to
        ``fraction * A_vessel(z)``. This is a DECLARED regularization, not a
        geometry: its purpose is to keep the two-zone annulus a real volume
        where a solved flux tube would otherwise fill the bore, since the
        annulus row's sources divide by ``V_ann`` (the hot-channel deposit is
        ``landed * Vp / V_ann``) and a sliver annulus makes those divisions
        stiff. A configuration that sets it is stating that cap as part of its
        closure. It binds before the vessel-area check, so in cells where it
        binds the hard refusal cannot fire.
    neutral_annulus_volume_fraction_min:
        Minimum ``V_ann / V_neutral`` allowed in any cell that HAS an annulus,
        [1]. Cells with no annulus at
        all (``V_ann = 0`` exactly -- the plenum, and any cell the plasma
        fills) are untouched: every annulus consumer already gates on
        ``V_ann > 0``, so an absent zone is inert by construction.

        What this refuses is the zone that EXISTS but has collapsed to a
        sliver, which nothing gates on and which enters as a divisor: the
        two-zone exchange and the hot-channel deposit both scale as
        ``1 / V_ann``, so a vanishing annulus does not switch off, it
        stiffens. Checked at construction against the built zone volumes and
        raised as a ``ValueError`` naming the offending cells. ``0.0``
        disables the check; the shipped value is far below any uniform-column
        geometry (a straight ``Rp`` inside ``Rm`` leaves ~0.86) and far above
        the collapse it exists to catch.
    neutral_baffle_positions_cm:
        Axial positions [cm] of optional thin annular baffles, measured from
        the cathode surface. A scalar or sequence is accepted, or ``None``
        for no baffles. Supplied together with matching clear radii (one
        without the other raises); their presence is what places the baffles.
    neutral_baffle_clear_radii_cm:
        Clear aperture radii [cm] for ``neutral_baffle_positions_cm``. Each
        aperture must leave the local plasma channel fully open and lie inside
        the local vessel radius. A scalar or sequence is accepted.
    source_region_length_cm:
        End of the fixed-cell-size source region [cm, measured from the cathode
        surface]; the region runs from the anode face at ``cathode_anode_gap_cm``
        to here and must lie strictly between the anode face and the end wall
        block (``Lm - end_wall_length_cm``), or the mid-plane ``Lm/2`` under
        ``far_end = "mirror"`` or ``TwinCathode``. Required on every layout;
        ``TwinCathode`` mirrors the region onto its far cathode end.
    source_region_dz_cm:
        Cell size [cm] inside that source region, held fixed independently of
        ``nx``; the region length minus the anode gap must be an integer
        multiple of it (1e-9 relative tolerance). Required on every layout.
    far_end:
        What ends the column at the far machine end, one of
        ``"end_wall"`` (default) and ``"mirror"``; any other value raises.
        ``"end_wall"``: the column runs to ``Lm - end_wall_length_cm`` and
        the end wall cell closes the machine, its outer face plasma-absorbing.
        ``"mirror"``: the HALF column. The mesh stops at the mid-plane
        ``z = Lm/2`` in a MIRROR face -- the symmetry plane of a two-source
        machine whose image cathode-anode source sits at ``z = Lm`` -- with
        no end wall cell; under it ``Lm`` is the mirror configuration's own
        length, cathode to image cathode, and ``nx`` counts the far column
        cells between the fixed source region and ``Lm/2``.
        The mirror face is closed and not absorbing: its fluid face flux is
        the ordinary face kernel against the mirror ghost state
        ``(n, -M, Ee, Ei)`` of the cell beside it, so it carries no particle
        or energy flux and the momentum flux ``p + a_max M`` (see
        ``NUMERICS.md``), and no heat crosses it. Nothing armed by the end
        wall role exists under it. Construction raises, naming the complete
        set, when ``"mirror"`` is combined with anything that presumes the end
        wall or cannot yet run at a mirror: ``TwinCathode``,
        ``heating_anomalous_transport = "plateau_multigroup"`` together with
        ``cathode_coupling`` (the walkers bounce between the cathode sheath
        and the plane, and once the anode sheath repels them all nothing
        removes them), ``neutral_momentum`` and
        ``neutral_energy`` (their far-face wall sinks and the hot channel's
        end-plane landing), ``neutral_kinetic_dvm_end_wall_jet`` (there is
        no end wall to return from),
        ``neutral_kinetic_dvm_annulus_flights = "bounded_chord"`` (its
        flown annulus returns its end exits through the lagged end buffer,
        so it cannot reflect specularly at the plane), ``S_pump_R != 0``
        (there is no right pump) and a non-default ``end_wall_length_cm``.
        ``neutral_model = "kinetic_dvm"`` runs at a mirror: its velocity
        distributions reflect specularly there, ``f(-v_z) = f(v_z)``,
        within each march. ``cathode_coupling`` runs at a mirror: the CSDA
        primary that reaches the plane turns round there, marches back with
        the anode interception re-armed, is turned back by the cathode
        sheath, and is booked to a leg-cap residual row if it has not stopped
        within the module's leg budget; the beam smoothing folds about the
        plane.
    """
    return {
        "Lm": 2117.8,
        "nx": 60,
        "Rm": 50.0,
        "Rp": 18.415,
        "plenum_length_cm": 166.0,
        "cathode_anode_gap_cm": 53.25,
        "nx_gap": 5,
        "end_wall_length_cm": 7.8,
        "Rcs": 0.0,
        "Lcs": 0.0,
        "Rsup": 0.0,
        # Prescribed per-cell flux-tube and vessel radii, plus the optional
        # area ceiling, armed by the plasma profile's presence. All None in
        # the template: a per-cell geometry has no default shape to inherit,
        # and its whole content is what the caller computed outside.
        "plasma_radius_profile_cm": None,
        "machine_radius_profile_cm": None,
        "plasma_area_max_vessel_fraction": None,
        # Two-zone sliver-annulus guard. Not presence-gated on the prescribed
        # geometry: it constrains ANY two-zone geometry, and the shipped value
        # is inert on every uniform column (which leaves ~0.86) while still an
        # order of magnitude above a capped 0.95-of-bore flux tube (0.05).
        "neutral_annulus_volume_fraction_min": 1.0e-2,
        "neutral_baffle_positions_cm": None,
        "neutral_baffle_clear_radii_cm": None,
        "source_region_length_cm": 103.25,
        "source_region_dz_cm": 10.0,
        "far_end": "end_wall",
    }


def floor_defaults():
    """Return numerical floors applied to conservative state variables.

    ne_floor:
        Minimum plasma/electron density used when flooring state [cm^-3].
    nn_floor:
        Minimum neutral density used when flooring state [cm^-3].
    Te_floor:
        Minimum electron temperature recovered from conservative energy [eV].
        Sits below the ADF11 0.2 eV edge so the afterglow can cool. Lowering it
        toward the neutral-gas temperature is only meaningful together with the
        sub-edge ADAS extension -- see the RETIRED recipe in the module note
        below, which must not be run.
    Ti_floor:
        Minimum ion temperature recovered from conservative energy [eV].
        The Phelps moment-closed ion-neutral collision operator is
        thermal-valid with no 0.1 eV clamp; the only consumer that required
        0.1 eV was the retired legacy IAEA CX table. All remaining Ti consumers
        (kappa_par_ion, pressure, sound speed) need only Ti > 0.
    """
    return {
        "ne_floor": 1e8,
        "nn_floor": 1e8,
        "Te_floor": 0.1,
        "Ti_floor": 0.02585,
    }


def neutral_source_defaults():
    """Return gas-puff, pump, and neutral-source defaults.

    S_gp:
        Source-side gas puff flow [sccm].
    Twin_S_gp:
        End-side gas puff flow used when ``TwinCathode`` is enabled [sccm].
    gas_puff_rise_center_s:
        The puff waveform is a square valve pulse: the flow is flat at
        ``S_gp`` between an opening and a closing erf edge, and the three
        timings below set those edges.

        Opening-edge center [s], measured from the end of the
        neutral-prebreakdown phase (the instant the cathode circuit closes),
        so the opening edge does not wait for breakdown. Must be ``>= 0``.
    gas_puff_rise_width_s:
        Erf width scale [s] shared by BOTH edges -- the opening
        edge and the closing edge are built with this one width. The 10-90%
        transition time is ~1.81x this value. Must be positive.
    gas_puff_close_lag_s:
        Delay [s] from the end of the main discharge (``tau_discharge`` after
        the main-discharge start) to the closing-edge center, so
        the closing tail runs on past the drive. Must be ``>= 0``.

        A bad value of any of the three raises at construction. The envelope is
        ``max(rise - fall, 0)``, so edges configured to overlap clamp at zero
        flow rather than going negative.
    S_pump_L:
        Source-side vacuum pump speed [L/s], lumped per END: the whole
        pumping speed seen by that end cell, ducting included.
    S_pump_R:
        End-side vacuum pump speed [L/s], lumped per END, same convention as
        ``S_pump_L``.
    gas_puff_enabled:
        Enables neutral gas-puff source terms.
    pump_enabled:
        Enables neutral pump sink terms.
    gas_puff_valves:
        Number of equivalent gas-puff valves used by the SCCM conversion.
    gas_puff_delivery_fraction:
        Dimensionless delivery/entry efficiency [1] multiplying the gas puff
        at the single shared sccm-to-particles conversion, so ``S_gp`` means
        the flow delivered AT THE VALVE and the flow injected into the model
        volume is ``S_gp * gas_puff_delivery_fraction``. It enters exactly
        where ``gas_puff_valves`` does, so it scales the puff magnitude
        without touching its axial shape or its waveform, and it applies to
        the source-end and twin-end puffs alike. Consumed by the neutral
        gas-puff source term in ``physics.neutrals`` -- the explicit RHS, the
        implicit backward-Euler neutral matrices and the saved
        ``puff_particles_per_s`` diagnostic all read the same value, so none
        can desync. Must be in ``(0, 1]`` and finite; a
        value outside that range raises at construction. ``1.0`` (the default)
        is the identity and is bit-exact.
    gas_puff_z_cm:
        The axial shape of the puff is the tube-beamed injection row: the
        feed pipe at the mid-plane puff ports is treated as a collimating
        tube in free-molecular flow, and the row is the ray-optics
        first-flight landing distribution of its exit distribution on the
        plasma column, with the wall and column radii read off the grid at
        the port cell. The row is not re-weighted by cell length and not
        masked to the main-chamber roles -- it lands where the rays land. It
        conserves the total inflow exactly, and one shared implementation
        feeds both the explicit RHS and the implicit neutral matrix, so the
        two sites cannot desync.

        Puff-port centre [cm, machine coordinates]; ``None`` falls back
        to whichever cell currently holds the ``puff`` role. Mirrored through
        the chamber midpoint for the twin puff. Pinning it in machine
        coordinates is what makes an nx refinement a resolution study: with
        ``None`` the source centre follows the puff cell's centre, so changing
        nx silently moves the source.
    gas_puff_orifice_id_cm:
        Inner diameter of the collimating feed pipe [cm], the emitting
        aperture of the injection row. Must be finite and positive.
    gas_puff_orifice_length_cm:
        Length of that same feed pipe [cm]. Only its ratio to
        ``gas_puff_orifice_id_cm`` enters -- that aspect ratio is the beaming
        parameter of the tube's exit distribution, and the row narrows as it
        grows. Must be finite, positive, and at least 4/3 of the bore, below
        which the long-tube expression has no branch and construction raises.
    pump_elbow_conductance_lps:
        Conductance of the unmodeled pump elbow [L/s], combined in series with
        the pump speed as ``1/S_eff = 1/S_pump + 1/C_elbow``. Applies only to a
        pump sitting on a plenum cell, so it is inert in legacy geometry.
        ``None`` (default) or a
        non-positive value means no elbow restriction -- the legacy limit.
        Because ``S_pump_L``/``S_pump_R`` are lumped per-end speeds that
        already carry their own ducting, setting this alongside them applies
        the same restriction twice on the source side.
    """
    return {
        # --- ACTIVE (square waveform + pump) ---
        # S_gp is the one free constant of the puff model; every other quantity
        # in the waveform is a hardware timing. It feeds back on the discharge
        # through S_gp -> ne -> current. FITTED-FLUX class: these two levels
        # were fitted under the retired 0 C sccm convention, so the 2026-08-21
        # meter changeover rescaled their digits by 1.0734834 (3400 ->
        # 3649.84) to hold the fitted particle flux fixed.
        "S_gp": 3649.84,
        "Twin_S_gp": 3649.84,
        # Square-waveform edge timings. The piezo is driven by a square
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
        # S_pump_L matches S_pump_R: each end carries the same lumped pumping
        # speed, the series conductance of that end's turbo through its own
        # elbow. The elbow is already inside this number, so
        # pump_elbow_conductance_lps stays None -- setting both would count the
        # elbow twice on the source side.
        "S_pump_L": 3000.0,
        "S_pump_R": 3000.0,
        "gas_puff_enabled": True,
        "pump_enabled": True,
        "gas_puff_valves": 2,
        # Delivery/entry efficiency of the puff: S_gp is the flow AT THE VALVE
        # and this fraction is the share that reaches the modelled volume. 1.0
        # is the identity, so the shipped configuration is unchanged; the
        # decomposition exists so the valve level and the delivered level are
        # separate quantities rather than one lumped constant.
        "gas_puff_delivery_fraction": 1.0,
        "pump_elbow_conductance_lps": None,
        # The puff-port position, in machine coordinates so it does not move
        # with nx.
        "gas_puff_z_cm": 86.3,
        # The collimating feed pipe behind those ports: bore and length [cm].
        "gas_puff_orifice_id_cm": 3.95,
        "gas_puff_orifice_length_cm": 22.0,
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
        the two.

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
    ignition_wall_clock_cap_s:
        Wall-clock budget [s] for reaching breakdown, measured from the start
        of the ``run()`` call. Zero (the default) disables the guard.

        Every OTHER non-ignition guard is expressed in SIMULATED time -- the
        stall window and ``tau_prebreakdown`` both are -- so all of them
        assume simulated time keeps advancing. A run that fails to ignite
        can instead destroy simulated time per wall-second: the timestep
        collapses and the arm crawls for hours without ever reaching the
        simulated instant at which a guard would fire. This cap is the arm
        that closes over that mode. It trips the SAME switch-open path as
        the stall trip and the ``tau_prebreakdown`` timeout, with reason
        ``"wall_clock_cap"``.

        Checked only while the run has not yet broken down, so it can never
        interrupt an igniting or ignited run however long it takes. Must be
        finite and non-negative; anything else raises at construction.
    ignition_accepted_step_cap:
        Accepted-step budget for reaching breakdown, counted from the start
        of the ``run()`` call. Zero (the default) disables the guard.

        The hardware-independent companion to
        ``ignition_wall_clock_cap_s``: it bounds the same crawl by work done
        rather than by time taken, so it is reproducible across machines
        and is the form to prefer for a gate. Trips the same switch-open
        path with reason ``"accepted_step_cap"``. Distinct from
        ``max_steps``, which bounds the WHOLE run and whose action is a
        RuntimeError or a truncated trajectory rather than a physical
        wind-down.

        Checked only while the run has not yet broken down. Must be a
        non-negative integer; anything else raises at construction.
    """
    return {
        "tau_prebreakdown": 0.05,
        # Both default-off: 0 disables the guard entirely and no wall clock
        # is ever read, so an unset run is bit-exact with a run predating
        # these keys.
        "ignition_wall_clock_cap_s": 0.0,
        "ignition_accepted_step_cap": 0,
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
        ``scripts/run/build_neutral_seed_cache.py``.
    """
    return {
        "dt_save": 1e-5,
        "t_save_start": 0.0,
        "max_output_steps": 0,
        "neutral_seed_cache_dir": None,
    }


def model_mode_defaults():
    """Return string-valued model selector defaults.

    Ti_birth_ionization:
        Ion birth temperature model for ionization -- the temperature the ion
        BORN by bulk ionization, by a beam ionization, and by the gas-puff
        local-ionization channel carries. Options are ``"neutral"`` (the
        default) or a numeric eV value.

        ``"neutral"`` is the option that PAIRS with the neutral energy field:
        the ion is born at the local neutral temperature
        ``Tn = (2/3) En / (nn k)`` of the very population the ``En``
        ionization sink debits (the column ``nn``),
        so the ion gains exactly the ``(3/2) k Tn`` per particle the neutral
        gas gives up and the pair conserves energy by construction. With
        ``neutral_energy`` off the state carries no ``En``, there is no local
        neutral temperature, and the birth falls back to the cold-gas scalar
        ``Tn_K``.

        A numeric value is NON-CONSERVING against an evolved ``En``: the sink
        still removes ``(3/2) k Tn`` per ionized atom while the ion is born at
        an unrelated temperature, and the difference leaves the model. That
        difference is reported per cell and per save by the
        ``ionization_birth_thermal_deficit_*_W_cm3`` diagnostic rows, which
        read zero to roundoff under ``"neutral"``.

        Ionization births book their energy moments on one convention, for
        bulk, beam and gas-puff births alike: the new electron is born cold
        (``Ee`` gains nothing, so ``Te`` falls by dilution), and the ion gains
        ``3/2 Ti_birth S_ion`` plus the mass-loading relative-drift mixing
        energy ``1/2 m (u_i - u_n)^2 S_ion``, so ion total energy (internal +
        kinetic) closes to the consumed neutral's energy.
    neutral_exchange_model:
        Axial neutral transport model. ``"constant"`` uses a fixed coefficient.

        ``"knudsen"`` (default) treats cell-to-cell exchange as Fickian transport with the
        Knudsen diffusivity ``D = (2/3)*v_th*R``, i.e. ``C = D*A/dz``. This is
        mesh-independent and reproduces the textbook long-tube conductance
        ``(2*pi/3)*v_th*R^3/L`` exactly. Thin apertures (the anode mesh) keep an
        orifice conductance in series. Prefer this for resolved runs, where the
        puff-to-pump back-path is the physics of interest and the historical model
        under-predicts it by 2-14x depending on cell size.
    neutral_model:
        Which engine carries the neutral population. ``"moment"`` integrates
        the fluid neutral density (and, with the ``neutral_momentum`` flag,
        its momentum) directly from the conservative RHS terms.
        ``"kinetic_dvm"`` carries LIVE transient column and annulus
        distributions ``f(z, v_z, v_perp)`` and advances them with one split
        implicit transport/collision step per neutral-clock tick, at step
        ACCEPTANCE only. The fluid neutral rows (``nn``, ``nn_a``, and any
        neutral-momentum rows) are then carried by the kinetic state instead
        of by the RHS, and ``nn`` IS the zeroth moment of the column
        distribution; the ion-side momentum and energy transfer of the
        ionization, charge-exchange, elastic and recombination channels is
        minus the corresponding moment of the kinetic operator, so the two
        sides are antisymmetric by construction. Electron-side costs
        (ionization potential, radiation, excitation) stay on the plasma
        book unchanged. It rides on the two-zone state and engages only once
        the plasma phase is live, so the pre-breakdown fill
        and the neutral equilibration stay on the moment terms.

        ``"kinetic_dvm"`` is a top-level MODEL SELECTION and OWNS a member
        set -- every control in it is M_n or En physics the kinetic state
        already carries, so the two cannot both own it. The measured set is
        ``core/model_families.KINETIC_DVM_INCOMPATIBLE_DEFAULTS``, which is
        the authority here: the ``neutral_momentum``, ``neutral_energy`` and
        ``neutral_hot_internal_wall`` flags, the cathode jet (``cathode_neutral_jet``,
        ``cathode_jet_surface_debit``, ``cathode_jet_energy_convention``,
        ``cathode_jet_hot_carrier``) and the anode-side momentum channel
        (``anode_neutral_jet``, ``anode_jet_energy_convention``,
        ``neutral_mesh_accommodation``). A member left at its config default
        is set to the value this selection requires AUTOMATICALLY, before
        any validator runs -- nothing has to be hand-cleared. A member the
        caller set EXPLICITLY to a value the selection refuses raises one
        ``ValueError`` at construction naming the selection, the whole
        member set and every offending key.

        Any other value raises at construction.
    neutral_kinetic_dvm_cadence_s:
        Neutral-clock interval [s] between transient DVM updates under
        ``neutral_model = "kinetic_dvm"``. The kinetic state advances on this
        clock, not on the plasma step, and the plasma-side transfer terms are
        held constant between ticks. Must be positive; raises at construction
        otherwise. Inert under the other neutral models. The shipped value is
        PROVISIONAL and was NOT chosen from an accuracy study -- the
        multirate convergence measurement that selects it has not been run.
    neutral_kinetic_dvm_nvz:
        Number of axial-velocity bins in the transient DVM's velocity grid.
        Must be EVEN: an odd count places a bin at exactly ``v_z = 0``, which
        neither transports nor mirrors under end-wall reflection. Raises at
        construction otherwise.
    neutral_kinetic_dvm_nvp:
        Number of perpendicular-speed bins in that same grid (positive-only,
        carrying the 2D perpendicular speed measure).
    neutral_kinetic_dvm_vmax_cm_s:
        Half-extent [cm/s] of that same grid: the axial axis spans
        ``(-vmax, +vmax)`` and the perpendicular-speed axis ``(0, vmax)``,
        both sinh-stretched about one fine scale, so this is what fixes
        which velocities the engine can represent at all. A drift the grid
        cannot reach is not approximated -- no non-negative weighting of
        bins has a mean beyond the last bin CENTER -- so a directed spectrum
        asked for past that point RAISES at the tick.

        ``None`` (the default) DERIVES the extent from the launch band the
        configuration can actually produce:

        * with a surface jet armed, ``1.25 * sqrt(2 e_max / m_He)``, where
          ``e_max = max(R_E / R_N) * (cathode_phi_c_cap_V + 10 eV)`` over
          the armed jets, each reading its own
          ``neutral_kinetic_dvm_{cathode,anode}_jet_R_E`` and ``_R_N``. The
          1.25 places the last bin CENTER above the drift at ``e_max`` --
          by 6.4% on a 48-bin axis and 10.8% on a 64-bin one -- rather than
          at it, and the validator refuses a grid where it does not.
        * with neither armed, the thermal/sonic sizing the engine has always
          used: four ion thermal speeds at a 10 eV ion cap plus 1.5 sonic
          drifts at 2e6 cm/s, which is what the CX tail and the column's own
          flows need and is unrelated to any jet.

        The ``cathode_phi_c_cap_V`` in that expression is the RAW atomic-data
        cap, taken because it is resolvable at construction; the circuit's
        own ``min(cap, V_avail(I))`` bound is tighter but current-dependent
        and therefore is not. The same ``cathode_phi_c_cap_V`` allowance is
        applied to the ANODE fall as a stated assumption about the band the
        construction check covers -- ``phi_a`` carries no cap of its own and
        none is implied here -- so the check is conservative on that side
        rather than predictive, and the tick's own moment refusal remains the
        backstop for anything the band did not anticipate.

        A positive float pins the extent instead, which is how the historical
        thermal/sonic sizing is reproduced with the jets armed. Raises at
        construction on a non-finite or non-positive value, and on a value
        BELOW that thermal/sonic sizing while a jet is armed, where the grid
        would be too small for the flows the arm carries irrespective of the
        launch band. Read only under ``neutral_model = "kinetic_dvm"``.
    neutral_kinetic_dvm_accommodation:
        The THERMAL (energy) accommodation coefficient ``alpha_E`` of the
        vessel's room-temperature technical-metal surfaces, in ``[0, 1]``.
        Under this engine's Maxwell specular-diffuse kernel the diffuse
        fraction IS ``alpha_E`` exactly: the accommodated share re-emits
        cosine-distributed at the surface temperature and therefore carries
        that surface's own mean energy, while the remainder exchanges no
        energy at all, so one visit transfers ``alpha_E`` of the available
        energy difference by construction.

        It is read at the CYLINDER and the END PLATES only. The anode mesh
        and the interior closed faces re-emit everything they intercept at a
        surface temperature -- they are fully accommodating by construction
        and do not read this key. It applies uniformly to every wall
        incidence, including the charge-exchange tail: the kernel makes no
        distinction by incident energy.

        The non-accommodated share is returned at the incident energy. At
        the CYLINDRICAL wall it is returned, per cell, on a cosine-wall
        spectrum whose temperature parameter is solved so that the
        spectrum's DISCRETE mean energy equals the retained share's own
        incident mean energy per atom: the count and the energy are both
        exact and the return carries zero net axial momentum, so the surface
        randomizes the direction while exchanging no energy. The solve is a
        bracketed bisection on a monotone function and RAISES at the tick
        rather than falling back when the target lies outside what the
        velocity grid can re-emit. At an end wall it is always the exact
        ``v_z`` bin mirror, which the symmetric stretched axis represents
        without re-projection error; the anode mesh and the interior closed
        faces use their own channels.

        The arm also carries the polarization-elastic ion-neutral channel: a
        BGK-like relaxation toward the local ion Maxwellian at the Phelps
        isotropic rate, alongside the charge-exchange channel at the Phelps
        backscatter rate, so the arm's momentum-transfer cross section is the
        fluid operator's ``Qi + 2 Qb``.

        Behavioural note on the LEFT end: the accommodated share there
        re-emits at the live cathode surface temperature ``T_s`` rather than
        at the wall temperature, so a room-temperature coefficient is being
        extrapolated to a hot surface. That end plate is area-subdominant
        against the cylinder.

        A boxed surface property, never a fit parameter. Raises at
        construction outside ``[0, 1]``.
    neutral_kinetic_dvm_exchange:
        Column/annulus zone-exchange closure of the transient DVM: the
        per-``(cell, v_perp)`` frequencies at which a neutral crosses
        ``r = Rp`` in either direction and strikes the vessel wall at
        ``r = Rm``. ``"cauchy_chord"`` uses the three-dimensional Cauchy
        mean chord ``4V/S = 2 (Rm - Rp)`` at the perpendicular speed and
        splits one surface encounter between the two cylinders as
        ``Rp/Rm : (1 - Rp/Rm)``. ``"geometric"`` uses the mean chord of the
        cell CROSS-SECTION, ``pi A / P = pi (Rm - Rp) / 2`` -- the crossings
        of two coaxial cylinders are decided entirely by the motion in the
        ``(x, y)`` plane, so the chord is a planar one -- and splits the
        encounter between the two circles in proportion to their
        PERIMETERS, giving ``nu_a->c = 2 vp Rp / (pi (Rm^2 - Rp^2))``,
        ``nu_a->wall = 2 vp Rm / (pi (Rm^2 - Rp^2))`` and
        ``nu_c->a = 2 vp / (pi Rp)``, the last of which averages over a
        Maxwellian to the free-molecular ``vbar / (2 Rp)`` the fluid arm's
        zone-exchange conductance already carries. Both branches impose
        ``V_col nu_c->a == V_ann nu_a->c`` on the actual cell volumes, so
        the particle ledger's zone channel cancels exactly either way. Any
        other value raises at construction.
    neutral_kinetic_dvm_annulus_flights:
        How the transient DVM takes the ANNULUS zone's wall-interaction and
        radial-exchange flights. ``"rates"`` uses the algebraic
        ``neutral_kinetic_dvm_exchange`` rates in the implicit march, with
        the annulus advected axially by the same upwind sweep as the
        column; the flight-time distribution each rate implies is
        exponential. ``"bounded_chord"`` replaces the annulus-side wall and
        annulus-to-column rates with three deterministic flight classes
        whose mean chords are derived numerically from the local
        ``(Rp, Rm)``: a wall launch reaches the inner surface with the view
        factor ``Rp/Rm`` at chord ``c_wi`` and the wall otherwise at
        ``c_ww``, and a column escape reaches the wall at ``c_io``. Each
        flight displaces the atom axially by exactly ``v_z c / v_perp`` and
        lasts ``c / v_perp``, so the axial step per surface encounter is
        bounded rather than exponentially tailed. Under that branch the
        annulus is not advected by the march -- the jump is its whole axial
        motion -- and the annulus distribution is the sum of the three
        in-flight populations; the column keeps the rate treatment, so
        ``neutral_kinetic_dvm_exchange`` still sets its escape rate
        ``nu_c->a`` and is not ignored. Neither branch has a free
        parameter. Inert unless ``neutral_model = "kinetic_dvm"``;
        selecting ``"bounded_chord"`` without that model, or any other
        value, raises at construction.
    neutral_kinetic_dvm_cathode_jet:
        Whether the transient DVM splits the counted cathode recycle into an
        ENERGETIC BACKSCATTER share and a thermal remainder. Off, every
        recycled atom leaves the cathode on the thermal cosine half-flux at
        the live surface temperature, which is the shipped reading. On, the
        ``neutral_kinetic_dvm_cathode_jet_R_N`` share is instead born as a
        directed volume birth in the cell the recycle was counted into,
        carrying ``(R_E/R_N)(phi_c + Te/2)`` of kinetic energy per atom -- the
        ``"total_reflected"`` reading of the reflection coefficients -- and
        the remainder keeps the thermal inflow. The energy the share carries
        is DEBITED from the cathode surface's own power balance in the same
        accepted step that counted it, as the named ``backscatter`` row of
        the surface energy ledger, so the reflected energy is created once
        rather than by both books. Inert unless
        ``neutral_model = "kinetic_dvm"``; arming it under any other neutral
        model raises at construction, and a standing guard refuses it
        together with the fluid channel's ``cathode_jet_surface_debit``,
        which would debit the same ``R_E`` a second time. That guard is
        unreachable through this model selection -- the debit is its own
        member of the kinetic_dvm family resolver, cleared to ``False``
        before any guard runs -- and is kept as a live statement about the
        PAIR against a future relaxation of that membership.
    neutral_kinetic_dvm_cathode_jet_R_N:
        Particle reflection coefficient of the transient DVM's cathode
        backscatter: the share of the collected ion flux that returns as
        energetic neutrals rather than desorbing thermally. Read only when
        ``neutral_kinetic_dvm_cathode_jet`` is on, and required there to
        satisfy ``0 < R_E <= R_N < 1``.
    neutral_kinetic_dvm_cathode_jet_R_E:
        TOTAL reflected energy fraction of that same channel: reflected
        energy over incident energy, summed over all particles. The
        ``R_N`` backscattered atoms carry all of it, so each leaves with
        ``R_E/R_N`` of the incident ``phi_c + Te/2``, and the cathode surface
        is debited exactly ``R_E`` of the ion bombardment energy the same
        particles delivered. Read only when
        ``neutral_kinetic_dvm_cathode_jet`` is on.
    neutral_kinetic_dvm_cathode_jet_T_launch_eV:
        NUMERICS parameter of that channel: the width of the smear the
        monoenergetic backscatter beam is represented by on the discrete
        velocity grid. ``None`` ties it to the grid -- the axial bin
        containing the launch speed, expressed as a temperature -- which is
        the narrowest spectrum the grid resolves there and the value the
        channel is specified at; a positive float pins it instead, as an A/B
        instrument. It is not a physical gas temperature: the launch
        spectrum's drift is solved from the ENERGY, so the spectrum's
        discrete mean energy is the launch energy whatever this is set to,
        and what this changes is only how wide a bundle of bins carries it. A
        spectrum this leaves too narrow or too fast for the grid to project
        raises at the tick rather than launching at the wrong energy. Read
        only when ``neutral_kinetic_dvm_cathode_jet`` is on; must be ``None``
        or positive.
    neutral_kinetic_dvm_anode_jet:
        Whether the transient DVM splits the counted anode-mesh collection
        into an ENERGETIC BACKSCATTER share and a thermal remainder. Off,
        every neutralized ion leaves the mesh at rest on the wall
        distribution in the cell it was counted into, which is the shipped
        reading. On, the ``neutral_kinetic_dvm_anode_jet_R_N`` share is
        instead born as a directed volume birth in that same cell, carrying
        ``(R_E/R_N)(phi_a + Ti)`` of kinetic energy per atom -- the
        ``"total_reflected"`` reading of the reflection coefficients -- and
        directed AWAY from the mesh on the side it was collected from, which
        is the fluid channel's own placement rule. The remainder keeps the
        thermal rebirth. A cell whose incident energy comes out at exactly
        zero -- which the fluid-parity clamp ``max(phi_a + Ti, 0)`` reaches
        while the anode sheath is electron-attracting before breakdown --
        launches NOTHING and is born wholly thermal, per cell rather than per
        tick: the same ion gives ``v_back = 0`` under the fluid spec, which
        is thermal desorption, and this channel books it as such rather than
        inventing energy the ion did not bring. Wire-INTERCEPTED neutrals are
        untouched: they keep the at-rest re-emission, and the axial momentum
        they arrived with is booked to the structure as a named diagnostic
        row. Inert unless
        ``neutral_model = "kinetic_dvm"``; arming it under any other neutral
        model raises at construction, as does arming it with no cathode
        solve to read ``phi_a`` from, or on a geometry whose anode mesh is
        not exactly one face. A standing guard refuses it together with the
        fluid channel's ``anode_neutral_jet``, which would re-emit the same
        collected stream directed a second time; that guard is unreachable
        through this model selection -- the fluid jet is its own member of
        the kinetic_dvm family resolver -- and is kept as a live statement
        about the PAIR against a future relaxation of that membership.
    neutral_kinetic_dvm_anode_jet_R_N:
        Particle reflection coefficient of the transient DVM's anode
        backscatter: the share of the collected ion flux that returns as
        energetic neutrals rather than being implanted and desorbing at
        rest. Read only when ``neutral_kinetic_dvm_anode_jet`` is on, and
        required there to satisfy ``0 < R_E <= R_N < 1``.
    neutral_kinetic_dvm_anode_jet_R_E:
        TOTAL reflected energy fraction of that same channel: reflected
        energy over incident energy, summed over all particles. The ``R_N``
        backscattered atoms carry all of it, so each leaves with ``R_E/R_N``
        of the incident ``phi_a + Ti``, and the anode energy book records
        exactly ``R_E`` of the ion bombardment energy the same particles
        delivered as having left with them. Read only when
        ``neutral_kinetic_dvm_anode_jet`` is on.
    neutral_kinetic_dvm_anode_jet_T_launch_eV:
        NUMERICS parameter of that channel: the width of the smear the
        monoenergetic backscatter beam is represented by on the discrete
        velocity grid. ``None`` ties it to the grid -- the axial bin
        containing the launch speed, expressed as a temperature -- which is
        the narrowest spectrum the grid resolves there and the value the
        channel is specified at; a positive float pins it instead, as an A/B
        instrument. It is not a physical gas temperature: the launch
        spectrum's drift is solved from the ENERGY, so the spectrum's
        discrete mean energy is the launch energy whatever this is set to,
        and what this changes is only how wide a bundle of bins carries it. A
        spectrum this leaves too narrow or too fast for the grid to project
        raises at the tick rather than launching at the wrong energy. Read
        only when ``neutral_kinetic_dvm_anode_jet`` is on; must be ``None``
        or positive.
    neutral_kinetic_dvm_end_wall_jet:
        Whether the transient DVM splits the counted END WALL return into an
        ENERGETIC FAST SHARE and a thermal remainder. Off, every ion the
        far-end characteristic boundary removes comes back as a cosine
        half-flux directed into the column at the 300 K wall temperature,
        which is the shipped reading. On, the
        ``neutral_kinetic_dvm_end_wall_jet_R_N`` share is instead born as a
        ``-z``-directed volume birth in the end cell the return was counted
        into, carrying ``(R_E/R_N)`` of the arrival energy per atom, and the
        remaining ``1 - R_N`` keeps the thermal face inflow. The end wall
        carries neither a sheath solve nor an energy book in this
        model, so two things follow that its cathode and anode twins do not
        share: the per-ion arrival energy is PRESCRIBED, through
        ``neutral_kinetic_dvm_end_wall_jet_sheath_Te_multiple``, rather than
        read from a solve; and the energy the launched atoms carry is debited
        from no surface book, exactly as the ``(3/2) k T_wall`` per atom the
        thermal return already carries is debited from none. Inert unless
        ``neutral_model = "kinetic_dvm"``; arming it under any other neutral
        model raises at construction, as does arming it without all three of
        its required numbers, or naming any of its four numbers while it is
        off.
    neutral_kinetic_dvm_end_wall_jet_R_N:
        Particle share of the counted end wall return that leaves the plate
        as the energetic directed launch rather than desorbing thermally.
        ``None`` until a configuration names it: required when
        ``neutral_kinetic_dvm_end_wall_jet`` is on, refused when it is off,
        and required there to satisfy ``0 < R_N <= 1``.
    neutral_kinetic_dvm_end_wall_jet_R_E:
        TOTAL returned energy fraction of that same channel: the energy that
        leaves with the launched atoms over the energy the collected ions
        arrived with. The ``R_N`` launched atoms carry all of it, so each
        leaves with ``R_E/R_N`` of the prescribed per-ion arrival energy.
        ``R_E`` above ``R_N`` is accepted here and means each launched atom
        leaves with MORE than the mean arrival energy per collected ion,
        which is a statement about a prescribed channel rather than about a
        reflection coefficient. ``None`` until a configuration names it:
        required when the channel is on, refused when it is off, and required
        there to satisfy ``0 < R_E <= 1``.
    neutral_kinetic_dvm_end_wall_jet_T_launch_eV:
        NUMERICS parameter of that channel: the width of the smear the
        monoenergetic launch beam is represented by on the discrete velocity
        grid. ``None`` ties it to the grid -- the axial bin containing the
        launch speed, expressed as a temperature -- which is the narrowest
        spectrum the grid resolves there; a positive float pins it instead,
        as an A/B instrument. It is not a physical gas temperature: the
        launch spectrum's drift is solved from the ENERGY, so the spectrum's
        discrete mean energy is the launch energy whatever this is set to,
        and what this changes is only how wide a bundle of bins carries it. A
        spectrum this leaves too narrow or too fast for the grid to project
        raises at the tick rather than launching at the wrong energy. Read
        only when ``neutral_kinetic_dvm_end_wall_jet`` is on, refused when
        it is off; must be ``None`` or positive.
    neutral_kinetic_dvm_end_wall_jet_sheath_Te_multiple:
        The multiple of the end cell's ``Te`` which, plus that cell's ``Ti``,
        is the kinetic energy ONE collected ion is taken to arrive at the
        end wall with. It is the FLOATING-SHEATH CONVENTION THE CALLER
        STATES, not a solved potential: this model carries no end wall
        sheath and no end wall circuit, so nothing here computes a drop and
        this number is the whole statement about one. ``None`` until a
        configuration names it: required when
        ``neutral_kinetic_dvm_end_wall_jet`` is on, refused when it is off,
        and required there to be a positive finite float.
    neutral_kinetic_dvm_jet_launch_width:
        ONE dimensionless width, shared by every armed surface jet, tying the
        launch smear to the launch ENERGY rather than to the velocity grid:
        ``T_launch = beta * e_launch``, so the fractional energy width a
        launch is smeared over is ``(3/2) beta`` at every launch energy and
        every velocity resolution. ``None`` (the shipped value) leaves each
        jet on its grid-tied width -- the axial bin containing the launch
        speed, expressed as a temperature -- whose fractional width instead
        halves with each doubling of ``neutral_kinetic_dvm_nvz``, so the
        launched SHAPE is a property of the grid there and of the surface
        here. Set, the grid-tied width remains a FLOOR: it is used wherever
        it exceeds ``beta * e_launch``, because a smear narrower than the
        local bin collapses onto one bin and cannot be projected at all. It
        is not a physical gas temperature -- the launch spectrum's drift is
        solved from the energy, so the discrete mean energy is the launch
        energy whatever this is -- and it carries no low-energy refusal band:
        ``u^2 = v_back^2 (1 - 3 beta / 2)`` is positive for every launch
        energy. Read only under ``neutral_model = "kinetic_dvm"``; must be
        ``None`` or a float in ``(0, 2/3)``, is refused with every surface jet
        off (nothing would read it), and is refused together with any
        ``neutral_kinetic_dvm_*_jet_T_launch_eV`` (two smear rules for one
        spectrum). Each armed tick reports how often the floor bound, as the
        ``launch_projections``, ``launch_projections_on_floor`` and
        ``launch_floor_fraction`` rows of the engine's per-tick ledger.
    neutral_kinetic_dvm_transfer_hold:
        How the plasma applies the transient DVM's tick-booked CX/elastic
        transfer between neutral clock ticks. ``"exponential"`` (the
        resolved default) treats that pair as the linear relaxation it is
        and integrates it exactly over each plasma step at the tick's frozen
        rate and target: ``Ei <- Ei_eq + (Ei - Ei_eq) exp(-nu dt)``, and the
        momentum row at the same ``nu`` towards the same lost-population
        drift. ``"zoh"`` holds the booked RATE constant across the tick
        instead, which is unconditionally unstable once ``nu dt_tick``
        exceeds 2 and is retained only as a negative control and to
        reproduce artifacts of runs made before the exponential hold. The
        ionization and recombination rows are a source under either value
        and are never relaxed. The difference between what the plasma
        applied and what the tick booked is carried as a per-cell HOLD DEBT,
        separate from the floor debt of
        ``neutral_kinetic_dvm_transfer_relax_fraction`` and repaid as a
        constant source over the following tick; it is the cadence meter.
        Read only under ``neutral_model = "kinetic_dvm"`` -- setting it
        under any other neutral model raises at construction, as does any
        value outside the accepted pair.
    neutral_kinetic_dvm_transfer_relax_fraction:
        Share of a cell's ion-energy margin above its ``Ti`` floor that the
        transient DVM's tick-frozen coupling drain may consume in ONE plasma
        step, in ``(0, 1]``. The transfer is held constant between neutral
        clock ticks while the plasma steps many times inside one, so at a
        collapsing cell the frozen drain can demand more energy than the cell
        holds; this caps what is APPLIED. The withheld energy and momentum
        are not dropped -- they are held as a per-cell debt and re-offered on
        later steps, so ``applied + debt == booked`` per cell at every
        accepted step. A value of ``1.0`` permits a drain that lands exactly
        on the floor within the step and leaves nothing for the other terms.
        Inert unless ``neutral_model = "kinetic_dvm"``; raises at
        construction outside the interval.
    adas_low_te_extension:
        Extends the ADAS ``acd`` (recombination) and ``prb1`` (recombination
        radiated power) coefficients consistently below the bundled ADF11
        low-Te edge at 0.2 eV, where the lookups otherwise clamp to the edge
        value. ``False`` keeps the clamp. Read by the reaction and energy
        terms; ``scd`` (ionization) and ``plt`` (line power) clamp at the edge
        either way.
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
    """
    return {
        # --- ACTIVE ---
        "Ti_birth_ionization": "neutral",
        "neutral_model": "moment",
        # 2nd-order operator-split pair; both are needed together with
        # heat_picard_iterations > 0 for the step to reach second order.
        "operator_splitting": "strang",
        "implicit_heat_scheme": "tr_bdf2",
        # --- INERT under these defaults (kept for the A/B arms) ---
        # Dead under neutral_model="moment" (the K2a transient DVM arm is the
        # only consumer). The cadence is
        # PROVISIONAL -- a conservative placeholder, not an accuracy result:
        "neutral_kinetic_dvm_cadence_s": 2.5e-5,
        "neutral_kinetic_dvm_nvz": 48,
        "neutral_kinetic_dvm_nvp": 12,
        # None = "size the grid to the launch band the armed jets can
        # produce"; a positive float pins the half-extent [cm/s] instead:
        "neutral_kinetic_dvm_vmax_cm_s": None,
        "neutral_kinetic_dvm_accommodation": 0.40,
        "neutral_kinetic_dvm_exchange": "cauchy_chord",
        "neutral_kinetic_dvm_annulus_flights": "rates",
        "neutral_kinetic_dvm_transfer_relax_fraction": 0.5,
        # Cathode-side energetic recycle, default OFF. The two reflection
        # coefficients MIRROR the fluid channel's cathode_jet_R_N /
        # cathode_jet_R_E so the two arms describe the same surface; None
        # ties the launch smear to the local velocity-grid bin:
        "neutral_kinetic_dvm_cathode_jet": False,
        "neutral_kinetic_dvm_cathode_jet_R_N": 0.34,
        "neutral_kinetic_dvm_cathode_jet_R_E": 0.18,
        "neutral_kinetic_dvm_cathode_jet_T_launch_eV": None,
        # Anode-side energetic recycle, default OFF. The two reflection
        # coefficients MIRROR the fluid channel's anode_jet_R_N /
        # anode_jet_R_E so the two arms describe the same mesh; None ties
        # the launch smear to the local velocity-grid bin:
        "neutral_kinetic_dvm_anode_jet": False,
        "neutral_kinetic_dvm_anode_jet_R_N": 0.63,
        "neutral_kinetic_dvm_anode_jet_R_E": 0.41,
        "neutral_kinetic_dvm_anode_jet_T_launch_eV": None,
        # End-wall-side energetic return, default OFF. Unlike the two
        # surfaces above there is no fluid channel to mirror and no sheath
        # solve to read, so every one of its four numbers is None -- "not
        # named" -- until a configuration names it, and naming one while the
        # channel is off is refused rather than left inert:
        "neutral_kinetic_dvm_end_wall_jet": False,
        "neutral_kinetic_dvm_end_wall_jet_R_N": None,
        "neutral_kinetic_dvm_end_wall_jet_R_E": None,
        "neutral_kinetic_dvm_end_wall_jet_T_launch_eV": None,
        "neutral_kinetic_dvm_end_wall_jet_sheath_Te_multiple": None,
        # Shared ENERGY-TIED launch smear for every armed surface jet.
        # None = "keep the grid-tied width", the shipped behaviour; a float
        # in (0, 2/3) smears every jet at T = beta * e_launch instead:
        "neutral_kinetic_dvm_jet_launch_width": None,
        # None = "not named"; resolved to "exponential" by the arm, and
        # refused outright by every other neutral model, so the key can
        # never be a silently inert control:
        "neutral_kinetic_dvm_transfer_hold": None,
        # Bucket-2 default-off closure instrument: extends acd/prb1 below the
        # 0.2 eV adf11 edge; the prb1 half is booked through
        # recombination_energy_return:
        "adas_low_te_extension": False,
    }


def fudge_factor_defaults():
    """Return physics scale factors and boundary geometry multipliers.

    recombination_energy_return:
        Books the GCR-consistent recombination energy PAIR on the electron
        fluid: per recombination event credit the binding energy ``I_ion``
        (paid at ionization via ``I_ion*S_ion`` and never returned by the
        standard booking) and charge the full ADAS ``prb1`` radiated power,
        adding ``I_ion*S_rec - P_PRB`` to ``Ee`` on top of the ordinary
        recombination terms. Both halves are evaluated from the same ACD sink
        the particle equation applies; the
        ``3/2 Te S_rec`` capture-kinetic-energy loss stays booked where it is
        and cancels in the net. The sign of the net follows the conditions --
        heating where the radiated energy per event is below ``I_ion``, an
        extra sink where it is above. ``False`` returns a zero source without
        evaluating the term. The pair is the consistent unit. Lookups clamp at
        the ADF11 grid edges.
    heat_flux_limiter_f:
        Free-streaming fraction ``f`` setting the electron heat-flux
        saturation ceiling ``q_sat = f n Te v_the`` (``Te`` in erg,
        ``v_the = sqrt(Te/m_e)``), against which the classical Spitzer-Harm
        parallel flux ``q_SH`` is capped. The limiter scales the conductivity
        per cell, so the operator stays a conservative flux divergence, and
        is frozen at the incoming ``Te`` like ``kappa`` itself. A smaller
        ``f`` lowers the ceiling and suppresses more. The limiter is always
        on and ``f`` must be ``> 0``; raises at construction otherwise.
    heat_flux_limiter_exponent:
        Knudsen exponent ``p`` in that limiter's suppression factor
        ``lambda = 1/(1 + (q_SH/q_sat)^p)``, applied as
        ``kappa_eff = lambda*kappa_e``. The ratio ``q_SH/q_sat`` plays the
        role of a Knudsen number. ``1.0`` is the harmonic form
        ``lambda = q_sat/(q_sat + q_SH)`` (Malone, McCrory & Morse, PRL 34
        (1975) 721; equivalently Fundamenski, PPCF 47 (2005) R163, eq. 10a) and
        takes its own code branch, so it is bit-exact with the pre-exponent
        limiter. ``p > 1`` suppresses
        the steep-gradient (high-ratio, non-local) flux much harder while
        leaving the shallow-gradient limit near-Spitzer -- a separation a
        single free-streaming fraction cannot express. Must be ``> 0``;
        raises at construction otherwise.
    b_surface_loss:
        Plasma surface neutralization/loss scale factor.
    b_ion_neutral_drag:
        Ion-neutral drag (friction) momentum-sink scale factor. Without an
        evolved neutral momentum this is the whole neutral-flow closure of the
        legacy drag term, asserting a fixed velocity slip ``u_n = (1 - b)*u``
        everywhere (leave at 1 unless doing a sensitivity study).
    b_presheath_length:
        Scale factor on the collisional presheath depth `c_s / nu_in` used to
        sample the upstream density for the Bohm flux at plasma-terminating
        surfaces. `alpha_isat` converts
        the *presheath-entrance* density to the sheath edge, so it must be applied
        to an upstream sample. `0` collapses the sample to the adjacent cell,
        recovering the historical behaviour; `1` (default) uses the physical
        depth. Inert in legacy geometry, which has no absorbing faces.
    alpha_isat:
        Ion-saturation/surface-loss coefficient.
    """
    # The atomic-rate / cooling / conduction scale factors that used to live
    # here were REMOVED (2026-08-28): ADAS and the Phelps collision operator
    # supply those channels directly, atomic rates are fixed inputs and not
    # knobs (standing policy 2026-07-20), and a uniform multiplier locked at 1
    # is not a physical control. The solver now hardwires the unit scale, and
    # resolve_config rejects the retired names as unknown keys -- as it already
    # does for the must-be-1 STRUCTURAL constants (b_pressure_work_elec,
    # b_pressure_work_ions, b_ionization_energy_cost). Exposing a knob that
    # must be 1 is a footgun. A future sensitivity instrument is a new build,
    # not a resurrection of these.
    return {
        # --- ACTIVE coefficients ---
        "b_surface_loss": 1.0,      # functional: =0 disables the boundary sink
        "b_presheath_length": 1.0,  # presheath depth (load-bearing)
        "alpha_isat": 0.6065306597126334,
        # GCR-consistent recombination energy booking (default-off closure
        # instrument; sub-0.2 eV). Per
        # recombination event, credit the binding energy I_ion to the electron
        # fluid (paid at ionization via I_ion*S_ion and never returned) AND
        # charge the full ADAS PRB (recombination radiation + bremsstrahlung +
        # cascade). Net = I_ion - E_rad. The PAIR is the consistent unit.
        # adf11 grid bottoms at 0.2 eV; lookups clamp there.
        "recombination_energy_return": False,
        # --- Electron heat-flux limiter (always on) ---
        # Free-streaming fraction f in q_sat = f*n*Te*v_the -- the ceiling
        # (Cowie & McKee, ApJ 211 (1977) 135, eq. 7) that the harmonic cap
        # saturates toward. Convention: v_the = sqrt(Te/m_e), as in Malone
        # 1975 / Fundamenski 2005; a coefficient quoted in the Cowie & McKee
        # convention needs *sqrt(2/pi) = 0.7979 to be read as an f here.
        "heat_flux_limiter_f": 0.45,
        # Non-local Knudsen exponent p for that limiter.
        # lambda = 1/(1+Kn^p)
        # with Kn = q_SH/q_sat. p=1.0 (default) is the harmonic form of
        # Malone 1975 / Fundamenski 2005 eq. 10a. p>1 suppresses the
        # steep-gradient (high-Kn, non-local)
        # startup flux much harder while leaving the shallow-gradient established
        # column near-Spitzer -- the startup-front pre-heating vs established-
        # column trade a single free-streaming factor cannot separate.
        "heat_flux_limiter_exponent": 1.0,
        # Multiplier on the moment-closed ion-neutral collision coefficient
        # and the drag timestep bound. Warns on a non-default value.
        "b_ion_neutral_drag": 1.0,
    }


def cathode_defaults():
    """Return LaB6 cathode/device circuit defaults.

    V_bank:
        Cathode power-supply bank voltage [V].
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
        move both to represent the same emission.
    R_comp:
        External/compliance resistance [Ohm]. The full loop series resistance.
        It does NOT set the discharge current -- the emission ceiling does;
        ``R_comp`` sets the voltage headroom, and the loop current is only
        weakly sensitive to it. ``R_comp`` and ``C_bank_F`` are jointly
        determined and must move together.
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
        historical behaviour. Must be in ``[0, 1]``.
    R_mesh_ohm:
        Anode-mesh series resistance [Ohm], separate from ``R_comp`` and on the
        internal (plasma) side of the probe, so it is invisible to the V_dis
        formula. Physically the Mo anode-mesh wire, order 1 mOhm and rising
        with anode temperature; only a CONSTANT value is implemented, so any
        ``R_mesh(T_anode)`` dependence must be approximated by that constant.
        Unlike ``R_comp_partition`` this is a real series resistance and does
        reach the loop current. Default ``0.0`` is bit-exact. Must be ``>= 0``.
    eta:
        Anode-mesh solid fraction (opacity) [dimensionless]: the share of the
        anode face its wires occupy, so ``1 - eta`` transmits. ``eta`` sets the
        anode's Bohm ion collection area (``2*eta*I_i``, both mesh faces) and
        the share of the gap-surviving thermionic beam the mesh intercepts;
        ``1 - eta`` is the face's neutral transparency and the beam's geometric
        survival. Must lie in ``[0, 1]``.
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
        Effective capacitance of the discharge bank [F]. ``None``
        is the historical infinite bank; the default is the hardware value
        ``9.5``. When set, the bank voltage starts at
        ``V_bank`` and drains by the drawn charge during drive phases
        (backward-Euler, folded into the circuit solve as a ``dt/C`` term on
        the effective resistance); the tail and floating phases leave it
        inert.

        Moves jointly with ``R_comp``: the two are determined together, so a
        configuration must not change one alone.
    L_parasitic_H:
        Parasitic series inductance in the current-driven discharge circuit
        [H], in series with ``R_comp``. The loop current is advanced once per
        accepted step by TR-BDF2. It must be positive when cathode coupling is
        enabled.

        L is inert for the sigma-scored discharge quantities and shows up in
        the current-rise shape and the ignition time.
    cathode_Ts_base_K:
        Heater-maintained standby surface temperature [K] -- the temperature
        the cathode sits at before the discharge. DERIVED, not measured: it
        is the operator-set heater current read through the Fig-10
        heater-current -> surface-temperature map.

        It is the initial condition of the evolving emitter surface
        temperature and the substrate temperature of the conduction term.
        Within a shot the surface temperature evolves by the surface energy
        balance

            C_th dT_s/dt = P_heater + P_cathode_i
                           - eps*sigma*A*(T_s^4 - T_env^4)
                           - (I_eth_star/e)*(phi_wf + 2 k_B T_s)
                           - G_cond*(T_s - T_base)

        where each *actually emitted* electron (``I_eth_star`` from the
        accepted solve, not the Richardson ceiling) carries away the work
        function plus its ~2kT_s of thermal energy. Space-charge clamping
        therefore suppresses cooling early (faster warm-up) and releases it
        near the ceiling (harder cap). ``P_heater`` is pinned by the
        pre-discharge equilibrium ``P_heater = eps*sigma*A*(T_base^4 -
        T_env^4)`` (open circuit => no net emission), so the heater is not a
        free parameter. The steady state -- and with it the plateau current --
        is an *output* of the balance, independent of ``C_th``. The update is
        semi-implicit in the linearized loss (unconditionally stable for any
        ``C_th``), floored at the 300 K chamber-wall temperature the surface
        radiates against, accepted steps only.

        Required, and refused as ``None`` at construction -- there is no
        other configured surface temperature to fall back on, and the TPMC
        kinetic background reads it too.
        Per-run operating points live in
        ``run_mechanism_ladder.ES_OPERATING[es]["Ts_standby_K"]``. Note the
        degeneracy with ``C_R`` documented above: the two describe one flat
        direction, so a configuration must not move both.
    cathode_heat_capacity_J_per_K:
        Effective thermal mass of the *emitting layer* [J/K] in the surface
        energy balance. NB this is the thermal skin depth reached over
        the discharge (sqrt(alpha*t) ~ 0.3-0.5 mm of LaB6), not the disc's
        bulk heat capacity (~hundreds of J/K) -- it shapes only the ramp
        timescale; the steady state is independent of it.
    cathode_emissivity:
        Total hemispherical emissivity of the emitting surface for the
        radiation term (LaB6 ~0.7).
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
        plateau *current* then follows from the balance.
    cathode_phiwf_clean_eV:
        Work function [eV] of the fully cleaned surface -- the ``theta -> 0``
        floor of ``phi_eff``, i.e. the per-shot-accessible depth of the
        removable layer rather than a literature clean-surface value.
        REQUIRED and must be strictly below ``phi_wf``; raises at
        construction when missing or not below it.

        The cathode work function evolves with the coverage ``theta`` in
        ``[0, 1]`` of the contaminant layer, initialized fully covered
        (``theta = 1``, so the shot starts at ``phi_wf`` exactly), evolving
        as

            dtheta/dt = -sigma_cl Gamma_i theta

        (ion-stimulated desorption, the only coverage channel, so ``theta``
        is monotonically non-increasing through a shot), and substitutes
        ``phi_eff = phi_clean + (phi_wf - phi_clean)*theta`` wherever the
        work function is read -- Richardson emission, Schottky lowering and
        emission cooling all take the one substituted value, never a mix of
        ``phi_eff`` and ``phi_wf``. ``theta`` advances by a backward-Euler
        update on accepted steps only. ``phi_wf`` keeps its meaning as the
        fully-covered shot-start work function.
    cathode_cleaning_sigma_cm2:
        Ion-stimulated desorption cross section [cm^2] in the coverage loss
        term ``sigma_cl*Gamma_i``, where the ion flux density onto the
        cathode is ``Gamma_i = I_i/(e*pi*R_cath^2)`` taken from the
        accepted-state sheath solve. Must be non-negative; raises at
        construction otherwise. ``0`` removes the ion-stimulated channel.
    cathode_cleaning_E_th_eV:
        Threshold energy [eV] for that desorption cross section. When set,
        ``cathode_cleaning_sigma_cm2`` is scaled by the near-threshold
        Bohdansky factor ``(1 - (E_th/E)^(2/3))*(1 - E_th/E)^2`` at the mean
        deposited energy per ion ``E = P_cathode_i/I_i`` from the same
        accepted-state solve, and the channel is switched off entirely for
        ``E <= E_th``. ``None`` leaves the cross section energy-independent
        (the pure fluence limit).
    cathode_solver_model:
        Which formulation supplies the discharge drive.

        ``"current_driven"`` (default) carries the loop current
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

        ``"prescribed_measured"`` PRESCRIBES both drive quantities from a
        measured trace instead of predicting either: the discharge current
        ``I(t)`` and the discharge voltage ``V_dis(t)`` are interpolated from
        the rung's own overlay file, and the cathode fall follows from the
        loop bookkeeping the result is already assembled with,
        ``phi_c = (V_dis − V_series) + phi_a − V_p``, with ``V_p = I·R_p`` the
        gap drop the model carries and ``V_series`` the internal series drop
        the ``V_dis`` probe does not see (identically zero at the shipped
        ``R_comp_partition = 1``, ``R_mesh_ohm = 0``, where the relation
        reduces to ``phi_c = V_dis + phi_a − V_p``). The emitted electron
        current is ``max(I − I_i, 0)`` against the plasma's own Bohm ion
        current at the cathode cell. Richardson emission, the surface
        temperature and the bank loop are therefore NOT consulted BY THE
        DRIVE while the prescribed drive is in force, and the
        cathode-warming ledger rows accumulate nothing over those steps.
        The surface temperature is FROZEN rather than retired: it holds the
        value it carried into the hand-off (the last warmed value) and stays
        load-bearing on the NEUTRALS, since an engaged kinetic neutral closure
        reads it as the cathode-end wall re-emission temperature. Requires all
        three ``cathode_prescribed_*`` keys below. See
        ``cablp/cathode/circuit_prescribed.py``.

        BRACKET AXIS (advisor, 2026-09-05). Under this mode the beam's birth
        energy is ``e·phi_c`` with ``phi_c`` assembled from the measured
        ``V_dis`` and the model's OWN ``phi_a`` and ``V_p`` — so those two
        become load-bearing on the beam energy in a way they are not under
        ``"current_driven"``, where the emission solve fixes ``phi_c``
        directly. No new control is exposed for either: the anode fall is the
        anode sheath the model already solves and the gap drop is the resolved
        Spitzer column. What follows is a REPORTING obligation, not a knob —
        a beam energy quoted from a prescribed-measured run is to be quoted as
        a BRACKET over ``phi_a`` and ``V_p``, never as a single number.
    cathode_prescribed_trace_path:
        Path to the measured discharge trace that drives
        ``cathode_solver_model = "prescribed_measured"``. An ``.npz`` in the ES
        overlay schema (``scripts/data/es{N}_sim1d_overlay.npz``), which must
        carry ``discharge_time_ms`` [ms], ``discharge_current_mean_a`` [A] and
        ``discharge_voltage_positive_mean_v`` [V]; a file missing any of the
        three, carrying non-finite samples, or whose time base is not strictly
        increasing is refused at construction. The voltage column is the
        overlay's POSITIVE convention, which is the sign the model's ``V_dis``
        carries, and is read as-is. The run records the file's sha256 in its
        HDF5 root attributes, so a saved trajectory names the exact measured
        product that drove it. REQUIRED under that mode and refused under any
        other; ``None`` (default) is the off path.

        Load-bearing on the beam energy through ``phi_c`` — see the bracket
        note under ``cathode_solver_model``.
    cathode_prescribed_t0_s:
        Model time [s] that the trace's own ``t = 0`` names, i.e. the origin
        that reconciles the two clocks:
        ``t_trace_ms = (t_model_s − cathode_prescribed_t0_s)·1e3``. This is
        the convention ``scripts/score/compare_sim1d_es1.py`` aligns with
        post-hoc, where the origin is the model time of the first
        ``main_discharge`` frame; a solver stepping forward cannot search a
        saved trajectory for it, so it is supplied here and measured from a
        calibrated reference run of the same configuration. REQUIRED under
        ``cathode_solver_model = "prescribed_measured"`` and refused under any
        other; there is no defensible default, and a guessed origin slides the
        whole measured drive against the column.
    cathode_prescribed_start_s:
        Model time [s] at which the prescribed drive TAKES OVER. Before it the
        run is the calibrated cathode exactly as configured — Richardson
        emission, the bank loop, the warming model — so a run in this mode
        needs a valid CALIBRATED configuration as well as a trace. The
        hand-off exists because prescribing from ``t = 0`` would impose the
        plateau current on a column that has not broken down, where the sheath
        cannot carry it and the solve sits at ``cathode_phi_c_cap_V`` for the
        whole build leg. Nothing is smoothed across the switch: the switch
        time and the two currents on either side of it are recorded in the
        HDF5 root attributes and printed, and a relative discontinuity above
        ``HANDOFF_JUMP_WARN_FRACTION`` (``core/prescribed_drive.py``) is
        announced loudly rather than hidden. Must be at or after
        ``cathode_prescribed_t0_s`` and inside the trace's span. REQUIRED
        under that mode and refused under any other.

        At or below zero it disables the foot entirely — the prescribed drive
        is then in force from the first step, the calibrated cathode never
        runs, and construction refuses any non-default value among the keys
        only that cathode reads (``CALIBRATED_ONLY_KEYS`` in
        ``core/prescribed_drive.py``: the emission constant, the surface
        temperature, the bank loop and the heater package), because on such a
        run they are read by nothing. That is the regime the foot exists to
        avoid, and it is available-but-checked rather than forbidden.

        Load-bearing on the beam energy through ``phi_c`` — see the bracket
        note under ``cathode_solver_model``.
    cathode_phi_c_cap_V:
        Physical ceiling [V] on the *net* cathode sheath drop in the
        current-driven solve. An imposed current the sheath cannot carry
        below it returns the ceiling solution tagged
        ``regime = "capability_limited"`` with the correspondingly large
        V_dis, and the circuit ramps the current down at ~V/L — the
        well-posed version of the inductive kick. It bounds a REGIME of the
        solve rather than describing a drop the device sustains, so the
        ceiling value is reported as ``phi_c`` for as long as that regime
        holds, and every consumer keyed to ``phi_c`` (notably the top of the
        multi-group plateau spectrum) sees it.
        This cap is a domain guard on the atomic data and holds in every
        regime.
    beam_anomalous_model:
        Anomalous (beam-plasma instability) drag for the CSDA deposition
        module (``cathode/beam_deposition.deposit_beam``). A declared closure
        BRACKET of three arms; a result states which one produced it.
        ``"none"``.
        ``"quasilinear"`` (default): mean-energy relaxation over
        ``l_QL = (n_e/n_b)(v_b/w_pe) ln(n_e/n_b)`` (~5-10 cm at production
        parameters), energy to local electron heating — the
        Langmuir-turbulence picture behind primaries not surviving
        downstream. Weak-beam domain only (returns no drag when
        ``n_b >= n_e/10``); parameter-free.
        ``"ql_relaxation"``: the same instability booked on its relaxation
        physics rather than by fiat. Reactive trapping extracts
        ``f_ext = min(n_b/2n_e, 1)^(1/3)`` of the beam energy, spread over the
        plateau-formation length ``L_rel = ql_relaxation_coeff (n_e/n_b) v_b /
        w_pe``, delivered to BULK electrons where the waves collisionally damp;
        and the booking is gated per cell on the boxed onset inequality
        ``0.687 w_pe min(n_b/n_e,1)^(1/3) > nu_en/2`` with ``w_pe > nu_en``,
        ``nu_en = nn K_m(Te)`` on the He e-n momentum-transfer table. No
        weak-beam cutoff (the caps carry the ``n_b >~ n_e`` corner). Requires
        ``ql_relaxation_coeff``. Not available on the compiled kernel, which
        takes the anomalous channel as a boolean and would run the fiat arm;
        selecting it takes the Python march.
    ql_relaxation_coeff:
        The O(10-100) coefficient in the quasilinear plateau-formation time
        ``tau_QL = c (n_e/n_b)/w_pe``, and so the length the extracted beam
        power is spread over. Read ONLY under
        ``beam_anomalous_model="ql_relaxation"`` and inert under every other
        value. Must be finite and > 0 or construction raises. It is a
        REGISTERED BRACKET rather than a tuned number, and results under this
        closure are quoted at the bracket endpoints, not at the default alone.
    heating_anomalous_transport:
        Where the CSDA ray's ANOMALOUS (quasilinear) heating lands. Selecting
        the non-default value without an active anomalous channel raises.
        ``"local"`` (default): the QL drag is banked as instantaneous local
        bulk electron heating in the cell that drove it — the Langmuir
        turbulence Landau-damps near where it grows, so its energy is handed
        to the background there.
        ``"plateau_multigroup"``: quasilinear diffusion does not warm a
        Maxwellian in place, it fills a fast-tail plateau, and the plateau is
        not one energy, so this value carries the SPECTRUM. In the flux frame
        the relaxed distribution is flat over the resonant band, so
        ``dGamma/dE`` is flat and ``dP/dE`` goes as ``E`` from the plateau
        EDGE ``E_1`` up to the beam energy ``E_b = e*phi_c``. ``E_1`` is a
        state-dependent solve, not a dial: it is where the flat plateau meets
        the launch cell's own 1D-reduced Maxwellian while carrying the emitted
        beam flux ``j_b = I_eth*/(e A_cell)``, found by bisection at every
        extraction solve and clamped to the inelastic floor with a counted
        census (``plateau_edge_clamped_steps`` in the cathode diagnostics) on
        any frame that hits it. The bank then splits into its two heirs -- a
        WAVE/BULK share ``(E_b - E_1)/2E_b`` banked as local bulk heat in the
        extraction cells, and a STREAMING share ``(E_b + E_1)/2E_b`` split
        into ``N`` equal-power groups with ``E^2``-uniform edges (equal power
        AND equal classical range by construction), each launched at its
        arithmetic-midpoint energy along +-B on the
        ``heating_anomalous_tail_forward_fraction`` split and walked on the
        CSDA module's Coulomb slowing machinery (the fast-electron
        stopping power, a ``1.5*Te`` thermalization floor). Nothing is fitted
        and no new parameter appears: the shares, edges and weights all
        follow from the flat plateau. Energy still hot at a domain end goes to
        a SEPARATE tail end ledger and leaves the system.
        The walkers IONIZE and EXCITE the column gas they pass through: each
        group is marched on the CSDA module's own integration, attenuating on
        the local COLUMN neutral density (the column channel ``nn``) with the He ionization and excitation cross
        sections at the walker's CURRENT energy, simultaneously with its
        Coulomb slowing. Each ionization event births one ion/electron pair at
        the event cell on the beam's own birth convention, invests ``I_ion``,
        banks the mean secondary ``<W_sec>`` as local electron heat and each
        excitation threshold as radiation; what still reaches a domain end
        goes to the tail end ledger. The two depth-1 truncation bars are
        evaluated PER GROUP on each group's midpoint energy: at or below the
        lowest inelastic threshold a group reverts to the energy-only walk
        (exact, since no inelastic channel is open there), and above the
        ``<W_sec>(E)`` crossing it marches with the depth-1 truncation, which
        there understates the tail's ionization by a MEASURED <= 2.0%. The
        power in each band is carried in the tail diagnostics
        (``beam_tail_sub_threshold_power_W`` /
        ``beam_tail_sub_threshold_fraction`` /
        ``beam_tail_above_bar_power_W``), so neither regime is silent. A
        group energy past the tabulated He EII cross section is refused at
        every cathode solve; the edge is INCLUSIVE within a relative tolerance
        of 1e-12 (``_beam_deposition.HE_EII_EDGE_REL_TOL``), because
        ``phi_c`` at ``cathode_phi_c_cap_V`` can put the top of the spectrum
        on the edge to the last bit.
        The range law is classical Coulomb.
    heating_anomalous_tail_forward_fraction:
        The share of each launched tail population sent along +z -- the
        direction from the cathode toward the end wall, the one the beam
        itself travels and the one the ``_tail_high`` end-loss row books. The
        remaining ``1 - f`` is launched along -z. **Read ONLY when the QL tail
        is WALKED** (``heating_anomalous_transport="plateau_multigroup"``) --
        inert otherwise, and a non-default value without it raises rather than
        being silently ignored. Dimensionless, in ``[0.5, 1.0]``; anything
        outside that range, and any non-finite value, raises at construction.
        ``0.5`` (default): the symmetric launch. ``1.0``: no -z walker is
        launched at all. It applies to every plateau group. The launched POWER
        is ``flux * E`` at any split, so this key moves where the tail power is
        delivered and never how much of it there is; the cathode-boundary,
        ionization and end-ledger conventions are untouched.
    heating_anomalous_tail_cathode_boundary:
        What the CATHODE end does to a tail walker that reaches it. **Read
        ONLY when the QL tail is WALKED**
        (``heating_anomalous_transport="plateau_multigroup"``) -- inert
        otherwise. ``"reflect"`` (default): a walker arriving at the cathode
        face of the plasma-active window with energy below ``e*phi_c(t)`` is
        turned around at the same energy and keeps walking; only a walker at or
        above that drop escapes. ``"escape"``: the free-escape convention, in
        which every walker reaching the face leaves and its energy is booked
        to the tail end ledger. The cathode sits at an accelerating drop of a
        few hundred volts through drive, at or above every plateau group
        energy, so free escape there deletes tail power the sheath in fact
        returns to the column. Under ``"reflect"`` the cathode-face row of the
        tail end ledger (``source_beam_end_loss_tail_low_W``) is therefore
        EXACTLY ZERO -- every group is born below ``e*phi_c`` and walkers only
        lose energy. A reader deriving the escaping fraction must still sum
        the whole end ledger rather than name the far-end row alone.
        Reflection is total by construction, with no partial-reflection
        coefficient -- the radial fraction of the returning tail that misses
        the emitting disc is UNSIZED in 1D and is a documented limitation, not
        a knob. Requires a single cathode: with ``TwinCathode`` both window
        faces reflect, trapping the walkers, and that raises.
    beam_tail_anode_reflected_particles:
        Reversed-walker rider, PARTICLE half (default 0.0 = the rider OFF,
        bit-exact). The share ``R_e`` of the QL tail walkers the anode mesh
        intercepts that come back off it, PER INCIDENT walker. Read ONLY
        where the anode tail cull fires -- a resolved mesh and a walked tail
        -- and refused with a non-zero value anywhere else. Dimensionless, in
        ``[0, 1]``. At 0.0 nothing returns and the whole culled share lands on
        the anode, which is the cull with no rider on top. A crossing whose
        incident energy is below the module's rider energy floor returns
        nothing whatever this value says: the walker is absorbed there.
    beam_tail_anode_reflected_energy:
        Reversed-walker rider, ENERGY half (default 0.0 = OFF, bit-exact).
        The share ``eta_E`` of the intercepted walkers' incident ENERGY that
        comes back, PER INCIDENT walker -- the same normalization the particle
        half uses, which is what keeps the pair free of any separate mean
        energy ratio. Dimensionless, in ``[0, 1]``, and it must not exceed
        ``beam_tail_anode_reflected_particles``: the mean energy per returned
        walker is the ratio of the two in units of the incident energy, and a
        ratio above one would return more than arrived. Refused rather than
        clamped.
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
        kicks the sheath solve. Must be ``>= 0``.
    cathode_neutral_jet:
        Gives the neutral flux recycled at an absorbing CATHODE face directed
        axial momentum instead of rebirthing it at rest: a fraction
        ``cathode_jet_R_N`` backscatters and the implanted remainder desorbs
        as a directed effusive flux off the hot surface. The momentum rides
        in the SAME term that rebirths the particles, so the two are
        consistent by construction, and the surface absorbs the difference
        between the incoming sonic momentum and the re-emitted jet momentum.
        End wall faces stay momentum-free. ``False`` rebirths at rest.
        Requires the ``neutral_momentum`` flag (there is no ``M_n`` field for
        the momentum to land in otherwise) and a geometry with an absorbing
        cathode face; raises at construction otherwise. The reflected atoms'
        kinetic energy beyond the mean-flow momentum is NOT booked --
        neutrals carry no energy field.
    cathode_jet_R_N:
        Particle reflection coefficient of the cathode surface: the
        backscattered fraction, leaving at
        ``v_back = sqrt(2 R_E (phi_c + Ti)/m)``. The remaining ``1 - R_N`` is
        implanted and re-emitted effusively at
        ``v_eff = sqrt(pi k_B T_s/(2 m))``, the per-particle directed
        momentum of a cosine-law effusive flux, so the mixed jet speed is
        ``R_N*v_back + (1 - R_N)*v_eff``. Must lie in ``[0, 1]`` when
        ``cathode_neutral_jet`` is on; raises at construction otherwise.
        Inert when that jet is off.
    cathode_jet_R_E:
        Energy reflection coefficient of the cathode surface, setting the
        backscatter speed ``v_back`` above. Must lie in ``[0, 1]`` when
        ``cathode_neutral_jet`` is on; raises at construction otherwise. Also
        read by ``cathode_jet_surface_debit``.
        ``cathode_jet_energy_convention`` fixes whether it is read per
        backscattered particle or as the total reflected energy fraction.
    cathode_jet_energy_convention:
        What ``cathode_jet_R_E`` MEANS when the backscattered atoms' launch
        speed is built, and therefore how much of the incident ion power the
        cathode jet hands the neutral gas.

        ``"legacy"`` reads it per backscattered particle:
        ``v_back = sqrt(2 R_E (phi_c + Ti)/m)``, carried by the ``R_N``
        reflected fraction alone, so the gas receives ``R_N R_E`` of the
        incident ion power while ``cathode_jet_surface_debit`` removes
        ``R_E`` of it from the surface.

        ``"total_reflected"`` reads it as the TOTAL reflected energy fraction
        (reflected energy over incident, summed over all particles -- the
        convention the surface debit is written in), so each of the ``R_N``
        backscattered particles leaves with ``R_E/R_N`` of the incident
        energy, ``v_back = sqrt(2 (R_E/R_N) (phi_c + Ti)/m)``, and the gas
        receives exactly the ``R_E`` the surface gave up.

        Consumed by the jet's ``M_n`` momentum booking and by the
        ``cathode_jet_neutral_energy`` term through one shared spec, so the
        two can never disagree. ``"total_reflected"`` requires
        ``cathode_neutral_jet`` and
        ``0 < cathode_jet_R_E <= cathode_jet_R_N < 1``; any other string, or
        those bounds violated, raises at construction. Inert when the cathode
        jet is off.
    anode_neutral_jet:
        The same directed-recycle treatment at the ANODE faces, applied per
        collected side: the backscattered fraction ``anode_jet_R_N`` is
        re-emitted back toward the side it was collected from, at
        the launch speed ``anode_jet_energy_convention`` builds from ``R_E``
        and the solve's anode drop ``phi_a``.
        The remaining ``1 - R_N`` re-emits from thin cylindrical wires with
        no net axial direction, so the anode channel is backscatter-only.
        ``False`` rebirths at rest. Requires the ``neutral_momentum`` flag,
        anode faces with ``eta > 0``, and a declared
        ``anode_jet_energy_convention``; raises at construction otherwise.
    anode_jet_R_N:
        Particle reflection coefficient of the anode surface -- the
        backscattered fraction. Must lie in ``[0, 1]`` when
        ``anode_neutral_jet`` is on; raises at construction otherwise. Inert
        when that jet is off.
    anode_jet_R_E:
        Energy reflection coefficient of the anode surface, setting the anode
        backscatter speed. Must lie in ``[0, 1]`` when ``anode_neutral_jet``
        is on; raises at construction otherwise.
        ``anode_jet_energy_convention`` fixes whether it is read per
        backscattered particle or as the total reflected energy fraction.
    anode_jet_energy_convention:
        What ``anode_jet_R_E`` MEANS when the backscattered atoms' launch
        speed is built, and therefore how fast the anode jet launches them.

        ``"legacy"`` reads it per backscattered particle,
        ``v_back = sqrt(2 R_E (phi_a + Ti)/m)`` -- the reading the anode
        channel was hard-coded to before this key existed.

        ``"total_reflected"`` reads it as the TOTAL reflected energy fraction
        (reflected energy over incident, summed over all particles -- the
        convention tabulated reflection coefficients are published in), so
        each of the ``R_N`` backscattered particles leaves with ``R_E/R_N``
        of the incident energy,
        ``v_back = sqrt(2 (R_E/R_N) (phi_a + Ti)/m)``.

        ``None`` (the default) is UNDECLARED, not a reading: arming
        ``anode_neutral_jet`` while it is ``None`` raises at construction,
        because the two readings launch the same coefficients at different
        speeds and the choice is a stance decision. ``"total_reflected"``
        additionally requires ``anode_neutral_jet`` and
        ``0 < anode_jet_R_E <= anode_jet_R_N < 1``; any other value raises.
        Inert when the anode jet is off.
    cathode_jet_surface_debit:
        Debits the cathode surface energy balance by the reflected-energy
        fraction: the warming model receives
        ``(1 - cathode_jet_R_E)*P_cathode_i`` in place of the full ion
        bombardment power, so the energy carried off by reflected atoms stops
        heating the surface. ``False`` retains all of it. Requires
        ``cathode_neutral_jet`` (it reads that jet's ``R_E``); raises at
        construction otherwise.
    cathode_jet_hot_carrier:
        Gives the cathode jet's BACKSCATTER share its own directed hot
        population instead of dumping it, cold, into the one cathode-adjacent
        cell. ``False`` (the default) is the v1 booking and is bit-exact.

        On, the ``cathode_jet_R_N`` share of the cathode recycle flux leaves
        the surface as an algebraic quasi-static beam at the one-spec
        ``v_back`` and is attenuated along the column by three channels --
        charge exchange at the relative collision energy, electron-impact
        ionization at the local ``Te``, and a geometric escape across the
        column boundary -- with a ``(1 - eta)`` first-crossing cull at each
        anode face. No new state field and no new saved row: the profile is
        rebuilt from the state on every evaluation. A CX event makes the fast
        atom an ion and returns the exchanged ion to the gas at the LOCAL ion
        state; an in-beam ionization is a plasma source paying the standard
        binding cost; escaped, culled and end-lost atoms are named leaks.

        The three v1 bookings the beam replaces are WITHHELD when it is armed
        (the cathode cell's ``R_N`` neutral rebirth, that share of the
        ``cathode_jet_neutral_energy`` excess, and the ``R_N v_back`` share of
        the jet momentum), so no channel is booked twice.

        Requires ``cathode_neutral_jet`` (it carries that jet's backscatter
        share), ``cathode_jet_surface_debit`` (the surface must give the
        energy up) and the ``neutral_energy`` flag (the partner atoms need an
        energy field to be born into); raises at construction otherwise.
    neutral_jet_arm_current_A:
        Arming threshold [A] of the cathode-jet ARMING CRITERION: the cathode
        jets launch only while the latch is ARMED, and it arms on the first
        accepted step whose cathode solve booked an ion current ``I_i`` at or
        above this value. ``0.0`` (the default) declares NO criterion -- the
        jets are then live whenever their own selectors are on, which is the
        behaviour that predates this key and is bit-identical to it.

        Covers BOTH cathode channels from one latch: the fluid
        ``cathode_neutral_jet`` and the DVM
        ``neutral_kinetic_dvm_cathode_jet``. The ANODE jets are not covered --
        they are driven by the anode-collected current, not by this one.

        While DISARMED the jets launch nothing AND the cathode surface is not
        debited for them: both sides of that pair are gated by the same latch
        state read from the same solve, so the surface is never debited for
        atoms that were never born. The DVM launch representability guard is
        unaffected and still fires on every armed step.

        The latch this key arms is RUN STATE on the solver instance, and it IS
        part of the restart record: a resumed run comes up with the arming
        state, the censored-step count, the transition count and the last
        transition time the producing run ended on, so a mid-run handoff does
        not re-censor jets the discharge had already brought into existence.
        The rows are presence-gated on a criterion being declared, so a
        payload from a run at the default carries none. One written before the
        latch was carried has none either: it still LOADS, and the resumed run
        starts disarmed and warns that it did.

        Must be >= 0. A positive value additionally requires
        ``0 <= neutral_jet_disarm_current_A < neutral_jet_arm_current_A``;
        anything else raises at construction.
    neutral_jet_disarm_current_A:
        Disarming threshold [A] of the same latch: once armed, the jets stay
        armed until an accepted step's booked ``I_i`` falls BELOW this value.
        That is what makes the criterion a latched hysteresis rather than a
        per-step comparison, and it is why a current dwelling near the arming
        threshold cannot chatter the jets on and off.

        The latch state this threshold advances is carried across a restart
        together with the rest of the criterion's census, so a resumed run
        holds an armed jet armed instead of re-crossing the band from scratch;
        ``neutral_jet_arm_current_A`` states the carriage in full.

        Must be ``0.0`` when ``neutral_jet_arm_current_A`` is ``0.0`` (no
        criterion is declared, so there is no band to describe); otherwise
        ``0 <= disarm < arm``. Negative values raise at construction.
    neutral_mesh_accommodation:
        Accommodates the evolved neutral wind's momentum on the anode mesh
        WIRES. The mesh's open area already throttles what the wind carries
        across, but the momentum the wires intercept has to land on the anode
        structure rather than stay in the gas; without this sink the gap
        recirculation set up by opposing surface jets is artificially
        elastic. For each anode face, wind flowing INTO the mesh from either
        flanking cell loses ``-max(+/-u_n, 0)*A_blocked/V*M_n`` -- the same
        free-molecular form the end walls use -- with
        ``A_blocked = A_open*(1 - T)/T`` for neutral transparency ``T``.
        ``False`` is off. Requires the ``neutral_momentum`` flag and anode
        faces with ``eta > 0`` and positive neutral transparency; raises at
        construction otherwise.
    """
    # These defaults ship the full cathode stack: CSDA beam + quasilinear
    # anomalous drag.
    #
    # The circuit values here are mirrored EXACTLY by the campaign stance in
    # ``scripts/score/compare_sim1d_es1.PARAM_OVERRIDES``; the duplication is
    # deliberate (that dict is the campaign stance record, and removing the
    # pins would change resolution order for the other run drivers).
    return {
        # --- ACTIVE: circuit hardware ---
        # V_bank here is the SUPPLY SETPOINT, which is a DIFFERENT QUANTITY
        # from the measured pre-shot open-circuit bank voltage and is
        # deliberately NOT replaced by it. The per-run open-circuit readings
        # live in ``scripts/run/run_mechanism_ladder.ES_OPERATING``; any run that
        # means the machine rather than the dial must set V_bank from there.
        "V_bank": 180.0,
        # phi_wf is the contaminated SHOT-START work function the coverage
        # model starts from.
        "phi_wf": 2.869,
        "C_R": 29.0,
        "R_comp": 7.2244e-3,
        "R_comp_partition": 1.0,
        "R_mesh_ohm": 0.0,
        "L_parasitic_H": 8.1e-6,
        "C_bank_F": 9.5,
        "eta": 0.358,
        "anode_radius_cm": None,
        "L_cath": 53.25,
        "R_cath": 18.415,
        # --- ACTIVE: beam deposition (the CSDA march) ---
        "beam_anomalous_model": "quasilinear",
        # ql_relaxation's plateau-formation bracket constant. INERT unless that
        # closure is selected; the shipped value is the bracket's geometric
        # centre and every headline under the closure is quoted at 10 and 100
        # as well.
        "ql_relaxation_coeff": 30.0,
        # QL heating locality: DEFAULT local (bit-exact).
        "heating_anomalous_transport": "local",
        # Launch-direction split of the walked tail: DEFAULT SYMMETRIC
        # (bit-exact). Inert unless the tail is walked.
        "heating_anomalous_tail_forward_fraction": 0.5,
        # The cathode face of the tail walk. Inert unless the tail is walked;
        # "escape" is the free-escape arm.
        "heating_anomalous_tail_cathode_boundary": "reflect",
        # Reversed-walker rider on the anode tail cull: DEFAULT OFF
        # (bit-exact). Both are read only where the cull fires.
        # The declared box the campaign brackets these across is NOT here --
        # the arms state their own values.
        "beam_tail_anode_reflected_particles": 0.0,
        "beam_tail_anode_reflected_energy": 0.0,
        "beam_clump_fraction": 0.0,
        "beam_clump_enhancement": 1.0,
        "beam_deposition_smoothing_cm": 0.0,
        # --- cathode surface power balance ---
        "cathode_Ts_base_K": 1910.0,
        "cathode_heat_capacity_J_per_K": 120.0,
        "cathode_conduction_W_per_K": 1200.0,
        "cathode_emissivity": 0.7,
        "cathode_solver_model": "current_driven",
        # --- OFF: prescribed measured drive (cathode_solver_model=
        # "prescribed_measured"). All three are None on the off path and are
        # REFUSED at construction under any other solver model, so a measured
        # trace can never be configured into a run that would ignore it.
        "cathode_prescribed_trace_path": None,
        "cathode_prescribed_t0_s": None,
        "cathode_prescribed_start_s": None,
        "cathode_phi_c_cap_V": 1000.0,
        # Surface-state coverage: the contaminant coverage theta evolves with
        # dtheta/dt = -sigma Gamma_i theta
        # and phi_eff = phi_clean + (phi_wf - phi_clean)*theta is substituted
        # everywhere phi_wf is read (emission, Schottky, cooling -- every
        # consumer reads the one substituted value, never a mix of phi_eff
        # and phi_wf). phi_wf keeps its meaning as the contaminated
        # SHOT-START value; the clean floor is the per-shot-accessible depth
        # of the re-adsorbed layer, not the literature clean-LaB6 value.
        # Ion-stimulated desorption is the only coverage-loss channel: the
        # coverage is monotonically non-increasing through a shot.
        "cathode_phiwf_clean_eV": 2.809,
        "cathode_cleaning_sigma_cm2": 3.5e-16,
        # Ion-stimulated desorption threshold [eV]: scales sigma by the
        # near-threshold Bohdansky factor (1-(Eth/E)^(2/3))(1-Eth/E)^2 at the
        # per-ion energy E = P_cathode_i/I_i. None = the energy-independent
        # fluence limit.
        "cathode_cleaning_E_th_eV": 20.0,
        # Directed neutral recycle jets: with
        # the neutral_momentum flag on, the surface recycle fluxes carry
        # directed momentum into M_n instead of rebirthing at rest.
        # Momentum-only first pass -- the reflected atoms' kinetic energy
        # is not booked (neutrals have no energy field; standing M2
        # convention). (R_N, R_E) are the particle and energy reflection
        # coefficients of the surface -- literature quantities, not fit knobs
        # (cathode = He->LaB6, anode = He->Mo).
        # The cathode channel splits R_N fast backscatter at
        # sqrt(2 R_E (phi_c + Ti)/m) + (1-R_N) directed effusion at the
        # surface T_s; the anode channel is backscatter-only, per collected
        # side, at the solve's phi_a (wire re-emission has no net axial
        # direction).
        "cathode_neutral_jet": True,
        "cathode_jet_R_N": 0.34,
        "cathode_jet_R_E": 0.18,
        # Which convention R_E is read in when the cathode backscatter speed
        # is built. "legacy" reads it per backscattered particle (the gas gets
        # R_N*R_E of the incident ion power while the surface debit removes
        # R_E); "total_reflected" reads it as the TRIM total reflected-energy
        # fraction, so the R_N reflected particles carry R_E/R_N each and the
        # exported power matches the debit.
        "cathode_jet_energy_convention": "total_reflected",
        "anode_neutral_jet": False,
        "anode_jet_R_N": 0.63,
        "anode_jet_R_E": 0.41,
        # Which convention anode_jet_R_E is read in. Ships UNDECLARED (None):
        # arming the jet without declaring it raises, because the same number
        # read per backscattered particle rather than as the total reflected
        # fraction launches the atoms ~21% slow and says nothing about it.
        # "legacy" is the per-particle reading the channel was hard-coded to
        # before this key existed; "total_reflected" is the convention the
        # tabulated coefficients above are published in.
        "anode_jet_energy_convention": None,
        # Debit the cathode surface's ion heating by the reflected-energy
        # fraction (the power balance receives (1 - R_E) * P_cathode_i); off,
        # the jet is momentum-only and the surface keeps that power. Requires
        # cathode_neutral_jet, and is REQUIRED by neutral_energy with the jet
        # armed -- with an En field the reflected power is booked into the gas,
        # so without the debit the same R_E would be spent twice.
        "cathode_jet_surface_debit": True,
        # Directed hot surface carrier for the backscatter share: DEFAULT OFF
        # (bit-exact). On, the R_N share leaves as its own attenuated beam
        # instead of rebirthing cold at the cathode cell, and the three v1
        # bookings it replaces are withheld. Requires cathode_neutral_jet,
        # cathode_jet_surface_debit and the neutral_energy flag.
        "cathode_jet_hot_carrier": False,
        # Cathode-jet arming criterion, ONE latch for both cathode channels
        # (fluid and DVM). Ships INERT: arm = 0 declares no criterion, so the
        # jets are live whenever their own selectors are on and the shipped
        # stance is bit-identical to the behaviour that predates these keys.
        # A run that wants the criterion sets both explicitly.
        "neutral_jet_arm_current_A": 0.0,
        "neutral_jet_disarm_current_A": 0.0,
        # Mesh momentum accommodation for the evolved wind: the momentum
        # the anode wires intercept lands on the anode structure instead
        # of staying in the gas (the open-area throttle alone leaves the
        # gap recirculation artificially elastic). Requires
        # neutral_momentum and anode faces.
        "neutral_mesh_accommodation": False,
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
    Tn_K:
        Neutral gas temperature setting the neutral thermal speed [K].
        Superseded as the collision operator's neutral temperature wherever
        the ``neutral_energy`` flag evolves ``En``, which carries a per-cell
        ``Tn`` instead.
    neutral_energy_wall_accommodation:
        Thermal accommodation coefficient ``alpha_E`` for neutral energy at
        the vessel surfaces, read only under the ``neutral_energy`` flag. It
        scales the free-molecular wall-visit rate in the ``En`` sink
        ``-alpha_E nu_wall (En - (3/2) nn k T_wall)``: ``0`` is perfectly
        specular (no energy exchange at the wall) and ``1`` is full
        accommodation in a single visit. Must lie in ``[0, 1]``; anything
        outside raises at construction.
    neutral_clausing_scale:
        Scale factor applied to the Knudsen tube and orifice conductances.
    """
    return {
        # --- ACTIVE ---
        # The third leg of the 2nd-order operator-split defaults: a positive
        # value is required for tr_bdf2 + strang to express second order.
        "heat_picard_iterations": 2,
        "heat_picard_tol": 1e-10,
        "Tn_K": 300.0,  # single cold-gas neutral temperature (Phelps T_eff)
        # --- INERT under these defaults ---
        # Neutral-energy wall accommodation (read only when the
        # neutral_energy flag is on, which ships ON -- so this key IS read
        # under the shipped defaults):
        "neutral_energy_wall_accommodation": 0.40,
        # LIVE (it scales every Knudsen tube and orifice conductance); inert
        # only because the default multiplier is 1.0:
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
    energy_exchange_rate_fraction:
        Fraction ``c`` [dimensionless] of the electron-ion thermal relaxation
        time ``1/nu_eq`` the accepted step may take, as a RATE bound
        ``dt <= c / nu_eq,max`` over the plasma-active cells, taken as the min
        with the fractional-change ``energy_exchange`` bound. ``nu_eq`` is the
        rate at which the exchange term relaxes one species' temperature
        toward the other (``physics.energy.electron_ion_relaxation_rate``,
        read back out of the same ``Q_ie`` the term calls); the DIFFERENCE
        ``Te - Ti`` relaxes at ``2 nu_eq``, so the explicit SSPRK2 advance of
        that difference has ``z = -2 c``, which stays inside the scheme's
        real-axis stability interval ``z >= -2`` exactly for ``c <= 1``.
        Stability, not accuracy: the
        fractional-change bound alone vanishes as ``Te -> Ti`` and stops
        bounding the exchange exactly where it is stiffest. ``None`` -- the
        default -- withdraws the bound entirely, so an unarmed run's dt
        arithmetic is bit-identical to one predating this key. Anything else
        must be a real, finite number in ``(0, 1]``; zero, negative, above
        one, non-finite or non-numeric raises ValueError at construction. The
        bound rides the timestep diagnostics as ``dt_energy_exchange_rate``
        and names itself ``energy_exchange_rate`` when it binds.
    dt_min:
        Minimum allowed timestep [s].
    dt_min_lock_max_steps:
        Maximum number of CONSECUTIVE adaptive steps whose timestep may be
        clamped up to ``dt_min`` before ``run()`` raises RuntimeError. Guards
        the dt_min lock: when a bound requests ``dt <= 0`` -- the signature of
        a cell sitting ON a floor while a term still drains it -- the clamp
        keeps the run alive at ``dt_min`` forever, so the run never finishes
        and never reports why. Consecutiveness is the discriminator: clamp
        episodes that release on their own are a normal, known-good family
        and are not bounded by this key; only an unbroken run of clamped
        steps is. The counter resets on the first unclamped step, and a
        caller-supplied fixed ``dt`` is never counted (the clamp does not set
        the step there, so such a run cannot lock). The raised error names the
        true active constraint, what it asked for, and the cell closest to the
        density floor. Must be a positive integer; anything else (zero,
        negative, non-integer, NaN) raises ValueError at construction.
    dt_max:
        Maximum allowed timestep [s].
    dt_global_scale:
        Uniform multiplier [dimensionless] on the FINAL accepted timestep,
        applied after every timestep candidate and after the dt_min/dt_max
        clamp, so it refines the whole dt trajectory by one factor instead of
        tightening one channel (scaling ``cfl`` refines only the CFL-bound
        phases). Must satisfy ``0 < dt_global_scale <= 1.0``; anything else --
        zero, negative, above one, non-finite, non-numeric -- raises
        ValueError at construction. It is a MEASUREMENT knob: it never
        loosens a bound, it does not name itself ``active_constraint``, and
        the scaled step is deliberately not re-clamped to ``dt_min``. The
        applied factor rides the timestep diagnostics as ``dt_global_scale``.
        The default 1.0 skips the multiply entirely, so an unarmed run is
        bit-exact with one predating this key.
    max_steps:
        Maximum accepted timesteps for a run. Zero means unlimited.
    max_steps_action:
        What ``run()`` does when ``max_steps`` is reached before ``t_end``.
        ``"raise"`` (default, historical behavior) raises RuntimeError and the
        in-progress trajectory is lost; ``"stop"`` ends the run cleanly and
        returns the partial trajectory with ``run_status =
        "max_steps_reached"`` (a completed opt-in run carries ``run_status =
        "completed"``) so the caller can inspect and save it.
    adaptive_retries_enabled:
        Enables retrying a rejected step with a smaller timestep.
    max_step_retries:
        Maximum retry attempts for one accepted step.
    dt_growth_enabled:
        Enables limiting timestep growth between accepted steps.
    dt_growth_factor:
        Maximum timestep growth factor between accepted steps.
    dt_growth_recovery_patience:
        Number of CONSECUTIVE accepted steps that must be capped by
        ``dt_growth`` before the accelerated re-approach engages. The default
        is 4. Zero disables the mechanism entirely and the ramp is uniformly
        ``dt_growth_factor``.

        What it is for: after a collapse the ramp re-approaches the physics
        bound geometrically, so recovering from a factor F below it costs
        ``log F / log(dt_growth_factor)`` steps -- at the shipped 1.25 that is
        ~26 steps from 364x below, and in knife-edge ``surface_loss`` regimes
        such episodes recur often enough to dominate the step count (measured
        in one probe: 80.6% of steps capped by ``dt_growth``, at a median 364x
        below the binding physics bound).

        Being capped by ``dt_growth`` for many steps in a row is evidence that
        the controller is merely ramping rather than tracking anything: no
        physical bound has bound in all that time. This key is how long to
        require that evidence. It is a PATIENCE, not a threshold on dt --
        nothing here inspects how far below the bound the step is, so the
        mechanism cannot mistake a genuinely small physics bound for a ramp.
    dt_growth_recovery_factor:
        Growth factor used once the accelerated re-approach has engaged.
        Consulted ONLY when ``dt_growth_recovery_patience`` > 0, which the
        shipped default satisfies, so this key is live at its own default.
        Must be greater than ``dt_growth_factor``; anything else raises at
        construction.

        The asymmetry between engaging and releasing is the hysteresis:
        engaging takes ``dt_growth_recovery_patience`` consecutive
        growth-capped steps, releasing takes ONE step capped by anything else
        (a physics bound, an output cadence, or a retry after a rejection).
        Re-approach is therefore fast while nothing is binding and instantly
        conservative again the moment something is. It does not weaken any
        bound: every step is still the minimum over all candidates, and this
        only widens the ceiling the ramp itself imposes.
    surface_loss_floor_exempt_exit_rtol:
        Outer (re-admission) threshold [dimensionless] of a two-threshold band
        on the ``surface_loss`` floor-aware drain exemption. The default is
        0.1. Zero
        disables the band entirely: the exemption stays single-threshold and
        knife-edge, and a run is bit-exact with one predating this key.

        With the band armed, an energy channel's cell is EXEMPTED when its
        margin above the per-cell floor energy ``3/2 n T_floor`` falls to
        within ``solver.SURFACE_LOSS_FLOOR_EXEMPT_RTOL`` of it -- the inner
        entry threshold, unchanged -- and is RE-ADMITTED only once that margin
        rises above this key's fraction of the same floor energy. Between the
        two thresholds a cell keeps whichever state it was last in, so a cell
        hovering at its temperature floor cannot alternate between exempt and
        bound from one step to the next. The band is one-sided and per-channel
        exactly as the single-threshold exemption is: it changes only which
        cells this drain bound reads, never a floor, a rate or any other
        bound, and the density channel is still never exempted.

        What the width MEANS, and it is the reason for the ceiling below: an
        exempted cell is not re-admitted to this drain bound until its margin
        exceeds this fraction of its floor energy, so the temperature interval
        ``[T_floor, (1 + exit_rtol) * T_floor]`` is drain-unthrottled by
        design once a cell has entered the exemption. The floor, not this
        bound, holds those cells; the density channel is never exempted at any
        width.

        Must be strictly greater than ``solver.SURFACE_LOSS_FLOOR_EXEMPT_RTOL``
        and strictly less than 1.0 when nonzero. A value at or below the inner
        threshold
        (which would be no band at all), a value at or above 1.0 (which would
        demand a margin larger than the floor energy itself before a cell can
        come back, so the exemption is no longer a hysteresis band around the
        floor and in the limit is permanent), and a negative or non-finite or
        non-numeric value each raise ValueError at construction.

        The exemption latch it introduces is per-cell, per-energy-channel RUN
        STATE: it lives on the solver instance, advances on every
        ``suggest_timestep`` call that actually evaluates this bound, and is
        not part of the restart record, so a resumed run starts with every
        cell un-exempt.
    max_density_step_fraction:
        Optional accepted-step density fractional-change guard. Zero disables it.
    max_neutral_step_fraction:
        Optional accepted-step neutral fractional-change guard. Zero disables it.
    max_energy_step_fraction:
        Optional accepted-step thermal-energy fractional-change guard. Zero
        disables it.
    """
    return {
        "cfl": 0.4,
        "density_dt_fraction": 0.25,
        "neutral_dt_fraction": 0.25,
        # Default-off stability bound: None withdraws the candidate before any
        # state is read, so an unarmed run's dt sequence is untouched.
        "energy_exchange_rate_fraction": None,
        "dt_min": 1e-10,
        "dt_min_lock_max_steps": 250000,
        "dt_max": 1e-4,
        # Default-off instrument: 1.0 skips the multiply entirely, so an
        # unarmed run's dt arithmetic is untouched.
        "dt_global_scale": 1.0,
        "max_steps": 0,
        "max_steps_action": "raise",
        "adaptive_retries_enabled": True,
        "max_step_retries": 8,
        "dt_growth_enabled": True,
        "dt_growth_factor": 1.25,
        # ARMED: after 4 consecutive dt_growth-capped steps the ramp switches
        # to the recovery factor, which makes that factor live at its own
        # default. Patience 0 still skips the branch entirely, so the ramp is
        # uniformly dt_growth_factor and the off path is bit-exact with a run
        # predating these keys.
        "dt_growth_recovery_patience": 4,
        "dt_growth_recovery_factor": 4.0,
        # ARMED: the floor-exempt test is the two-threshold band, entry at
        # SURFACE_LOSS_FLOOR_EXEMPT_RTOL and re-admission at this width. Zero
        # still skips the band entirely, leaving the single-threshold
        # expression and allocating no latch.
        "surface_loss_floor_exempt_exit_rtol": 0.1,
        "max_density_step_fraction": 0.0,
        "max_neutral_step_fraction": 0.0,
        "max_energy_step_fraction": 0.0,
    }


def restart_defaults():
    """Continuation of a previous run from an exported end state.

    restart_from:
        Path to a ``sim1d-restart-v1`` payload written by
        ``results.restart.save_restart_state``, or ``None`` (the default) for a
        run that builds its own initial condition. When set, the payload's
        instant replaces the whole initial condition: the conserved state, the
        simulation clock, every continuation cache and latch, and the run
        loop's own controller state, so the resumed run's saved frames are
        raw-byte identical to those of an unsplit run over the same window.

        The load raises ``ValueError`` at construction if the file is missing,
        carries another format, or was produced under a different grid, packed
        state layout, or structural closure key; and if the run also requests
        an equilibrating ``initial_neutral_state`` (which would overwrite the
        restored state) or
        a ``neutral_model`` whose distribution function the payload does not
        carry. The full inventory, and the justification for each deliberately
        dropped member, is ``_sim1d/results/restart.py`` together with the
        restart state carried in ``_sim1d/solver.py``; the resume contract is
        ``_sim1d/NUMERICS.md``, section "Restart".
    """
    return {
        "restart_from": None,
    }


def parallel_momentum_sink_defaults():
    """Imposed parallel momentum sink beyond a stated axial position.

    A RESPONSE-MAP INSTRUMENT, and nothing else. Its VALUE CLASS is PROBE:
    the rate below has NO PHYSICAL OWNER in this model -- no collision
    process supplies it, no measurement pins it, and nothing derives it.
    What it is for is the inverse question: how much parallel momentum
    loss, imposed over which length of column, the machine's observed
    profiles would require. The deliverable is the REQUIREMENT (a force in
    dyn and the momentum-loss length that goes with it), not the number
    that produced it.

    **It must never appear in a stance.** A configuration that names a
    plasma states physics with owners; this term has none, so arming it in
    a base configuration would put an unowned force inside the reference
    everything else is compared against. Response-map arms are DERIVED
    configurations that arm it, run, and are read as a map.

    The term is a linear damping of the evolved parallel momentum density
    on the column cells at or beyond ``parallel_momentum_sink_z_start_cm``,

        F = -nu_add * M = -nu_add * m_i n u   [g cm^-2 s^-2],

    booked as its own RHS row ``parallel_momentum_sink``, with the
    frictional work it does,

        Q_i = +nu_add * M * u = nu_add * m_i n u^2   [erg cm^-3 s^-1],

    booked into the ION internal energy as the row
    ``parallel_momentum_sink_heating``. The full dissipated drift energy is
    booked, not half: the boxed ion-neutral drag gives its other half to a
    neutral population that exists, and this term has no partner species to
    give anything to, so a half-booking would be an unowned energy leak
    rather than a convention. Both rows are absent from the ledger entirely
    unless the term is armed, so an unarmed run's saved term structure and
    trajectory are bit-identical to a checkout that has never heard of it.

    The rate does NOT enter the timestep ladder, and the margin behind that
    ruling is MEASURED rather than asserted. The term is an explicit linear
    damping, so the accepted step must satisfy ``nu_add * dt << 1`` on its
    own, and the reference ES1 run the three response-map arms are derived
    from (49,415 accepted steps) says by how much. Its largest accepted step
    on the drive plateau is 6.03e-7 s, giving ``nu_add * dt <= 2.8e-3`` on
    the largest of the three arms; its largest step anywhere, taken in the
    afterglow, is 5.96e-6 s, giving ``nu_add * dt <= 2.8e-2``. SSPRK2
    applied to ``y' = -nu_add * y`` has amplification ``1 - x + x^2/2`` with
    ``x = nu_add * dt``, which is stable for ``0 <= x <= 2``, so the margin
    at the largest step the run takes is 72.5x and the step would have to
    reach 4.3e-4 s before this term bound anything. That is the ruling: no
    ladder entry is needed, because the bounds the limiter already carries
    hold the term well inside its own stability limit without knowing it
    exists. A rate large enough to bind would be a statement about the
    machine no response map is asking.

    parallel_momentum_sink:
        Whether the sink is armed. ``False`` (the default) is the whole
        shipped package: the term is never constructed, neither row is
        emitted, and naming either number below while this is off raises at
        construction rather than leaving an inert control.
    parallel_momentum_sink_rate_s:
        The rate ``nu_add`` [s^-1]. ``None`` until a configuration names it:
        REQUIRED when the sink is armed, refused when it is off, and
        required there to be a finite float strictly greater than zero.
        There is no default -- the rate IS the hypothesis the arm states,
        and inheriting one would make the map read as a physical claim.
        With the local flow speed ``u`` it states a momentum-loss length
        ``L = u / nu_add``, which is the form the requirement is quoted in.
    parallel_momentum_sink_z_start_cm:
        Axial position [cm] on the same coordinate as ``geometry.z_cm``,
        at and beyond which the sink acts. ``None`` until a configuration
        names it: REQUIRED when the sink is armed, refused when it is off,
        and required there to be finite and to lie WITHIN the plasma
        column's own axial extent. A position outside it raises: below the
        column the term is not a statement about a region at all, and above
        it the term reaches no cell and would be silently inert, which is
        the one thing a presence gate exists to prevent.
    """
    return {
        "parallel_momentum_sink": False,
        "parallel_momentum_sink_rate_s": None,
        "parallel_momentum_sink_z_start_cm": None,
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
    restart_defaults,
    parallel_momentum_sink_defaults,
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


# Flag defaults. Which flags a specific committed configuration pins is stated
# by that configuration file itself, under scripts/stances/.
input_flags_template_1d = {
    # The plasma solve itself. OFF puts the run on the neutral-only implicit
    # stepper: the plasma RHS returns a zero state, the kinetic and DVM refresh
    # loops never run, the phase machine follows the equilibration cycle lattice
    # instead of the discharge schedule, the gas puff loses its waveform, and
    # default_t_end becomes cycles * tau_cycle (which raises unless cycles is
    # positive). run_neutral_equilibration pins it off on its inner sim.
    # A structural restart key -- a payload whose run had it set differently is
    # refused rather than restored.
    "Plasma": True,
    # Two-cathode layout: a cathode at BOTH ends, both plasma-terminating faces
    # mirrored, and the end-side puff Twin_S_gp carrying the second source.
    # Its mesh is the far_end='mirror' half column reflected about Lm/2: the
    # fixed source region and puff cell at both ends and nx far cells on each
    # side. It has no end wall, so the end wall sheath debit is absent.
    # Two construction-time refusals, each where the twin geometry leaves a
    # single-valued quantity undefined: cathode_solver_model='current_driven';
    # and heating_anomalous_tail_cathode_boundary='reflect' (both walls of
    # the walk window would be reflecting cathodes, trapping the walkers, and the
    # walk has no termination convention for that). A structural restart key.
    "TwinCathode": False,
    # Axial (Spitzer-Harm) electron and ion heat conduction: the RHS term, its
    # parabolic timestep bound, and the conductivity of the implicit substep.
    # OFF returns a zero conduction RHS and withdraws the bound (it returns
    # infinity, so conduction stops constraining dt). Note the implicit substep
    # still runs with the flag off -- it applies its handed-in source at K = 0,
    # an exact dt*S, rather than dropping the source.
    "heat_conduction": True,
    # The operator split. ON steps explicit SSPRK2 for everything but heat
    # (operator A) and then an implicit heat substep (operator B); the explicit
    # stage runs with the conduction bound withdrawn, since B is unconditionally
    # stable in it. OFF takes one fully explicit SSPRK2 step and dt carries the
    # parabolic conduction bound. The scheme, splitting order and Picard count
    # of the substep are the implicit_heat_scheme / operator_splitting /
    # heat_picard_iterations parameters. ON also re-homes the beam
    # electron-energy deposition and the anode electron-sheath debit out of A
    # and into B, so a caller asking a single step for operator_split=False
    # while it is on is refused: those rows would have nowhere to land.
    "implicit_heat_conduction": True,
    # Evolve axial neutral momentum M_n as a sixth conservative field:
    # the drag deposits its momentum into the
    # neutral wind instead of a closure, ionization/recombination exchange
    # momentum between species, and the wall/pump remove it. Off => the
    # 5-field state.
    "neutral_momentum": True,
    # Evolve the neutral thermal energy density En as an optional conservative
    # field, packed last, AND with it the decoupled two-channel neutral gas the
    # field only makes sense inside.
    #
    # COLD CHANNEL. The neutral temperature becomes the per-cell field value
    # Tn = (2/3) En / (nn k) instead of the config scalar Tn_K. (nn, M_n, En)
    # are transported as one fluid by a Rusanov mini-flux carrying the COLD
    # gas's own pressure p_n = (2/3) En, which SUPERSEDES the donor-cell M_n
    # self-advection -- exactly one advection operator runs. The Knudsen
    # exchanges carry the donor cell's energy per atom; the puff arrives at the
    # wall temperature, the pump leaves at the local one, ionization debits the
    # local one, and the surfaces accommodate En toward (3/2) nn k T_wall at
    # neutral_energy_wall_accommodation times the free-molecular wall-visit
    # rate.
    #
    # HOT CHANNEL. The CX-born minority sits at the local ion temperature and
    # is collisionally decoupled from the cold bulk (the gas-gas mean free path
    # is far longer than the column radius), so its pressure never enters a
    # force on the fluid. It is algebraic -- no packed row -- with a ballistic
    # redistribution kernel: atoms are eroded out of nn at their own energy,
    # fly, and end on the column boundary (mass moved axially, energy left on
    # the wall), in re-CX (momentum and energy handed to the ions where they
    # got to), or ionized in flight.
    #
    # Requires neutral_momentum; refuses every
    # kinetic neutral model (which carries the neutral energy as a moment of f). With
    # cathode_neutral_jet it additionally requires cathode_jet_surface_debit,
    # so the backscatter energy is booked once rather than twice. Each is a
    # construction-time ValueError. Off => the historical layout, bit-exact.
    "neutral_energy": True,
    # Wall the hot channel's ballistic flight at the INTERNAL plasma
    # boundaries, not only at the two global end planes. The walls are the
    # closed plasma faces (geometry.plasma_open false: every face where a
    # plasma-dead cell -- plenum, obstruction -- abuts a live one, plus the two
    # end planes) together with the plasma-absorbing faces, which are a
    # refinement of that set. A flight reaching one is clipped to the wall
    # plane and the atom is booked in the cell on its OWN side of it, which is
    # exactly the fold/absorb treatment the end planes already get; the landed
    # atoms rejoin the COLD neutral books (nn, or the annulus nn_a)
    # at that boundary-adjacent cell, at the unchanged landing energy.
    #
    # PER-CELL BEHAVIOUR. Every cell is confined to its own contiguous run of
    # same-topology cells. A LIVE cell's flights stay in its live segment, so
    # its landings never fall on a plasma-dead cell -- with the flag off they
    # do, and the caller's plasma-topology mask (which the hot channel's rows
    # are subject to) then deletes those deposits, so atoms leave the inventory
    # with no surface having absorbed them. A PLASMA-DEAD cell's flights stay
    # in the dead block they were born in, so its (floor-density) births can no
    # longer deposit out of a masked cell into a live one either. A BOUNDARY
    # cell -- the live cell against a cathode disc or an end wall -- is the
    # cell that receives everything folded at that wall, on both counts. The
    # mask itself is untouched; the flag only stops feeding it rows to delete.
    # Cells with no column (Rp = 0) keep the in-place identity row they already
    # had.
    #
    # Consumed by the neutral_hot_channel term alone (physics.hot_neutrals):
    # it is passed to ballistic_flight_kernels. hot_end_fraction then reads "folded at a wall"
    # rather than "folded at an end plane".
    #
    # Requires neutral_energy -- there is no hot channel to wall without it --
    # as a construction-time ValueError. Bit-exact when off (presence-gated:
    # the off path's wall bounds ARE the two end planes, so every clip reduces
    # to the historical one).
    "neutral_hot_internal_wall": True,
    # The cathode/anode/bank circuit solve. OFF, no cathode solve is produced
    # for the whole run: the boundary carries no device current or voltage, the
    # cathode and anode jets return nothing. run_neutral_equilibration pins it
    # off on its inner sim. Two construction-time refusals of things that need
    # a solve that would not exist: the two DVM jets, whose launch energies are
    # the sheath potentials phi_c and phi_a -- armed without a solve they would
    # silently launch at the thermal Ti alone. With the flag ON, a zero anode ion
    # current is a runtime error rather than a clamp: the circuit cannot close,
    # and the message names clearing this flag as the way to model a machine
    # with no anode collection.
    # A structural restart key.
    "cathode_coupling": True,
    # Reuse a cached neutral-equilibration seed (the equilibrated nn/nn_a
    # profile) instead of re-running the ~1-min 100-cycle equilibration every
    # run. Default OFF and bit-exact off.
    # When ON, requires initial_neutral_state = "equilibrate" and a
    # neutral_seed_cache_dir (the signature-keyed seed DATABASE):
    # a miss (new neutral-flow config) equilibrates once and stores it. See
    # core/neutral_seed_cache.py and scripts/run/build_neutral_seed_cache.py.
    "use_cached_neutral_seed": False,
    # END-FACE SHEATH ELECTRON-ENERGY BOOKING, one closure per axial end. The
    # two ends are separate faces carrying separate fluxes in separate
    # regimes -- the end wall row is live in every phase, the cathode fall
    # row is identically zero until a virtual cathode forms -- and nothing
    # couples them except the column, so each end is armed on its own. Every
    # row either end adds is ELECTRON ENERGY ONLY ([erg cm^-3 s^-1] on the
    # plasma cell volume, positive = into the electron store) and is
    # PRESENCE-GATED: unarmed, its rows do not exist at all.
    #
    # END WALL -- no key. ONE row, ``end_wall_e_sheath_climb``, negative,
    # armed by the GEOMETRY: present exactly when the mesh carries a
    # plasma-absorbing face whose live cell has the end wall role, and absent
    # otherwise (the TwinCathode layout ends in a second cathode and has
    # none). It does NOT require the cathode circuit solve. The end wall is a
    # floating exhaust with no circuit branch, so the fall its collected
    # electrons climb comes out of the plasma electron store and is handed to
    # the ions, which deposit it on the surface. The row is
    # -Lambda_eff Te Gamma_coll, which with the unconditional 2 Te of
    # ``characteristic_boundary`` makes the face debit the sheath-edge
    # (2 + Lambda_eff) Te per collected electron. Lambda_eff = Lambda +
    # ln(1/alpha) is the barrier at a surface drawing no net current: the
    # circuit's sheath lift for the configured gas plus the presheath drop
    # implied by the very alpha that face samples its Bohm flux at, so the two
    # cannot describe different sheath edges. Neither number is written here;
    # both are read from the code that already owns them.
    #
    # LAMBDA_EFF IS A STATE-DEPENDENT BARRIER, NOT A CONSTANT, and its range
    # is [Lambda, Lambda + 1/2]. The two limits are the two limits of the same
    # alpha. When the collisional presheath is SHORTER than the sampling cell
    # the cell sits at the sheath edge, alpha -> exp(-1/2) carries the whole
    # Boltzmann drop, and Lambda_eff -> Lambda + 1/2. When the presheath is
    # LONGER than the cell -- which is the pre-breakdown state, where a cold
    # thin end cell puts L_ps at tens to hundreds of cm against a cell of a
    # few cm -- the cell is INSIDE the presheath, its density has already
    # dropped part of the way from the reservoir, and the electrons collected
    # from it climb only the remaining ln(1/alpha) of presheath on top of the
    # Lambda of sheath; the rest of the drop is resolved by the interior
    # cells. alpha -> 1 there and Lambda_eff -> Lambda. A reading below
    # Lambda + 1/2 in the early phases is that regime and is correct; a
    # reading outside the bracket is not, and the smoke suite asserts the
    # bracket rather than a value.
    #
    # CATHODE -- no key. THREE rows at the emitting face, armed by the
    # GEOMETRY and the circuit: present exactly when the mesh carries a
    # plasma-absorbing face whose live cell has the cathode role and the
    # cathode circuit solve (``cathode_coupling``) runs, because the solve is
    # the source of I_eth_star, I_e_ret and the sheath potentials. The
    # TwinCathode layout books each emitting face from its own circuit result.
    # With the end wall row this is ONE sheath-edge rule at every electrode
    # face. The three rows are kept apart because they are three different
    # physical channels with two different signs:
    #   ``cathode_e_emitted_enthalpy``  +2 k_B T_s Gamma_em, positive. The
    #       enthalpy the released electrons carry in, off a half-Maxwellian at
    #       the emitter surface temperature. Gamma_em = I_eth_star/e is the
    #       SPACE-CHARGE-RELEASED flux, not the Richardson ceiling.
    #   ``cathode_e_emitted_fall``      +e (phi_c_plus - max(phi_c, 0))
    #       Gamma_em, positive. The part of the fall those same electrons drop
    #       through that the beam row does not already carry: the beam row
    #       distributes the NET phi_c, so this is identically zero while no
    #       virtual cathode has formed (phi_c_minus = 0) and positive once one
    #       has.
    #   ``cathode_e_collected_climb``   -e phi_c_plus Gamma_ec, negative. The
    #       barrier the returning plasma electrons climbed, charged to their
    #       own store -- the anode flag's plasma-pays convention at the
    #       cathode. Gamma_ec = I_e_ret/e.
    # The beam deposition row is UNTOUCHED: it already distributes the net-phi_c
    # energy of the emitted electrons and nothing here re-books it. The
    # 2 (Te - T_s) Gamma_em form is NOT what this books.
    #
    # A non-finite current or potential from the solve raises RuntimeError
    # rather than planting a NaN in an energy row.
    # The electron-energy sink charged per ionization event, I_ion * S_ion. Off
    # zeroes that cooling row, so ionizations cost the electrons nothing. This
    # flag is the whole on/off: the companion scale is hardwired to 1.0 and is
    # not a config knob.
    "ionization_energy_cost": True,
    # Non-finite state assertions, checked at the end of construction and after
    # every state-vector set. On, a non-finite value in any state field
    # (n, nn, M, Ee, Ei, M_n, nn_a, M_n_a, En) or any derived field
    # (u, Te, Ti, pe, pi, p) raises a ValueError naming that field. Off, the
    # checks never run. A debugging instrument: it changes no physics, only how
    # early and how loudly a corrupted state is caught.
    "debug_checks": False,
}


def load_config(path):
    """
    Load 1D solver parameters and flags from a TOML file.

    The file may contain ``[params]`` and ``[flags]`` sections. Missing values
    fall back to ``input_dict_template_1d`` and ``input_flags_template_1d``.

    It may additionally contain ``[models.<family>]`` DECLARATION BLOCKS, each
    stating one model family's complete membership without namespaces; see
    :mod:`~cablp.solvers._sim1d.core.model_declarations`. Blocks and the flat
    tables resolve onto the same surface, and a key may appear in only one of
    them.

    Any other top-level table or key RAISES. This loader reads three tables and
    nothing else, and a file carrying a fourth is a file written for a different
    reader -- most often the CONFIGURATION form, whose ``base`` /
    ``[input_dict]`` / ``[input_flags]`` this function cannot resolve, because a
    ``base`` names a committed file in a directory the solver package does not
    know about. Silently ignoring such a file would resolve it to bare defaults,
    which is precisely the implied plasma the refusal exists to stop.
    """
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    unknown = sorted(set(raw) - {"params", "flags", "models"})
    if unknown:
        raise ValueError(
            f"{path}: load_config reads [params], [flags] and "
            f"[models.<family>] only; it does not own {', '.join(unknown)}. "
            "A CONFIGURATION file (base = \"<name>\", [input_dict], "
            "[input_flags]) is read by the configuration loader "
            "scripts/stance/stance_config.py, which resolves its base chain "
            "against the committed stance directory; this loader would resolve "
            "it to bare defaults instead."
        )
    return resolve_config(
        raw.get("params", {}), raw.get("flags", {}), raw.get("models", {})
    )


def default_config():
    """Return copies of the default 1D input dictionary and flags."""
    return dict(input_dict_template_1d), dict(input_flags_template_1d)


# Shared successor texts for keys retired together, so a group cannot drift
# apart in wording.
_FRONT_FLUX_RETIRED = (
    "nothing: the sonic front-filling flux is removed; the Rusanov face "
    "flux is the only plasma face flux"
)
_AMBIPOLAR_RETIRED = (
    "nothing: the conservative solver has no ambipolar-diffusion closure; "
    "the Rusanov face flux carries the plasma"
)
_ELECTRON_DRIFT_RETIRED = (
    "nothing: the electron drift-transport operator is removed; the "
    "electron pressure work is booked with the ion velocity"
)
_REGIME_TRACER_RETIRED = (
    "nothing: the pre-breakdown passive tracer is removed; the plasma fluid "
    "owns every typed-active cell from the first step"
)


#: Keys REMOVED from the templates, each mapped to the successor that took
#: over its reads. A configuration naming one is refused like any other
#: unknown key -- the entry only makes the refusal say what replaced it,
#: because a stored file written before the removal is the case that hits
#: this path and "unknown key" alone does not tell its author where to go.
#: A name stays here once removed: the message is the only thing a retired
#: key still owns.
RETIRED_PARAM_KEYS = {
    # The end-wall rename. The model's far face IS the LAPD chamber's end
    # wall -- there is no distinct collector electrode -- so the role and
    # every key that carried its name were renamed. A pre-rename
    # configuration file is the case that reaches this path.
    "collector_length_cm": (
        "end_wall_length_cm, the same length of the same cell under the "
        "face's own name"
    ),
    "neutral_kinetic_dvm_collector_jet": (
        "neutral_kinetic_dvm_end_wall_jet, the same channel under the "
        "face's own name"
    ),
    "neutral_kinetic_dvm_collector_jet_R_N": (
        "neutral_kinetic_dvm_end_wall_jet_R_N"
    ),
    "neutral_kinetic_dvm_collector_jet_R_E": (
        "neutral_kinetic_dvm_end_wall_jet_R_E"
    ),
    "neutral_kinetic_dvm_collector_jet_T_launch_eV": (
        "neutral_kinetic_dvm_end_wall_jet_T_launch_eV"
    ),
    "neutral_kinetic_dvm_collector_jet_sheath_Te_multiple": (
        "neutral_kinetic_dvm_end_wall_jet_sheath_Te_multiple"
    ),
    "T_s": (
        "cathode_Ts_base_K, the heater-maintained standby surface "
        "temperature the cathode power balance evolves from"
    ),
    "end_wall_face_riemann_solver": (
        "nothing: the end wall face removes the PHYSICAL flux at the "
        "sheath-edge state it samples (n_se = alpha_se n, u = c_s), which "
        "is a single state and so poses no Riemann problem for a solver to "
        "resolve"
    ),
    # Circuit and cathode selectors and their parameters, removed with the
    # closures they served. Selectors whose one surviving value is now
    # unconditional name that behaviour; the rest name nothing.
    "cathode_model": (
        "nothing: the cathode solve is controlled by the cathode_coupling "
        "flag alone"
    ),
    "cathode_warming_model": (
        "nothing: the surface power balance ('power_balance') is "
        "unconditional"
    ),
    "cathode_surface_model": (
        "nothing: the ads/des contaminant-coverage surface state "
        "('ads_des') is unconditional"
    ),
    "cathode_sample_smoothing": (
        "nothing: the presheath-transit electrode sample smoothing "
        "('presheath') is unconditional"
    ),
    "cathode_emission_profile": (
        "nothing: the uniform emitting disc ('uniform') is unconditional"
    ),
    "cathode_Ts_fwhm_cm": (
        "nothing: it was read only by the removed gaussian emission profile"
    ),
    "cathode_emission_annuli": (
        "nothing: it was read only by the removed gaussian emission profile"
    ),
    "cathode_emitting_area_initial_fraction": (
        "nothing: the emitting-area closure it seeded is removed"
    ),
    "cathode_Rp_model": (
        "nothing: the gap resistance is the sampled-cell form ('sample'), "
        "unconditionally"
    ),
    "cathode_lnL_model": (
        "nothing: the Coulomb logarithm is the local electron-ion form "
        "('nrl_ei'), unconditionally"
    ),
    "cathode_circuit_sample": (
        "nothing: the circuit advance reads the raw accepted sample ('raw'), "
        "unconditionally"
    ),
    "cathode_circuit_bound_object": (
        "nothing: the circuit voltage bound it configured is removed"
    ),
    "circuit_dt_fraction": (
        "nothing: the circuit relaxation timestep bound it scaled is removed "
        "with the circuit voltage bound"
    ),
    "circuit_picard_tol_rel": (
        "nothing: the fluid-circuit Picard coupling it configured is removed"
    ),
    "circuit_picard_max_iter": (
        "nothing: the fluid-circuit Picard coupling it configured is removed"
    ),
    "cathode_ion_secondary_emission_yield": (
        "nothing: ion-induced secondary emission at the cathode face is "
        "removed"
    ),
    "vessel_capacitance_F": (
        "nothing: the vessel common-mode node it configured is removed"
    ),
    "vessel_leak_resistance_ohm": (
        "nothing: the vessel common-mode node it configured is removed"
    ),
    # Beam-deposition and hot-tail selectors and their parameters, removed
    # with the closures they served. Selectors whose one surviving value is
    # now unconditional name that behaviour; the rest name nothing.
    "beam_deposition_model": (
        "nothing: the beam deposits by the CSDA slowing-down march "
        "('csda'), unconditionally"
    ),
    "beam_coulomb_model": (
        "nothing: the CSDA Coulomb drag is the fast-electron stopping power "
        "('fast_electron'), unconditionally"
    ),
    "beam_excitation_model": (
        "nothing: the sheath solve carries no beam excitation channel; the "
        "cathode-anode gap attenuation it feeds back is the effective cross "
        "section inverted from the CSDA deposition march, and the march "
        "books the beam's excitation of the gas"
    ),
    "b_beam_excitation": (
        "nothing: the sheath solve carries no beam excitation channel; the "
        "cathode-anode gap attenuation it feeds back is the effective cross "
        "section inverted from the CSDA deposition march, and the march "
        "books the beam's excitation of the gas"
    ),
    "beam_excitation_energy_eV": (
        "nothing: the sheath solve carries no beam excitation channel, and "
        "the CSDA deposition march reads each excitation's radiated energy "
        "from the helium singlet manifold"
    ),
    "beam_product_transport": (
        "nothing: the CSDA ray's event products are banked in their birth "
        "cell ('local'), unconditionally"
    ),
    "heating_anomalous_disposal": (
        "nothing: the extracted anomalous power is disposed of as "
        "heating_anomalous_transport says, with no per-cell split ('local')"
    ),
    "heating_anomalous_tail_energy_keying": (
        "nothing: the only walked tail left, "
        "heating_anomalous_transport='plateau_multigroup', keys its spectrum "
        "to the live e*phi_c and the solved plateau edge"
    ),
    "heating_anomalous_tail_energy_eV": (
        "nothing: it was the fixed-rung birth energy of the removed "
        "single-line tail walk"
    ),
    "heating_anomalous_tail_phi_c_fraction": (
        "nothing: it was the f in E_tail = f*e*phi_c of the removed "
        "single-line tail walk; the plateau spectrum spans the whole band"
    ),
    "heating_anomalous_tail_ionization": (
        "nothing: the walked tail "
        "(heating_anomalous_transport='plateau_multigroup') ionizes and "
        "excites the column gas ('on'), unconditionally"
    ),
    "ionization_birth_energy_model": (
        "nothing: ionization births book the cold-electron convention with "
        "the ion mass-loading mixing energy ('conservative'), "
        "unconditionally"
    ),
    "Te_birth_ionization": (
        "nothing: a new electron is born cold, so no birth temperature is "
        "read"
    ),
    # Neutral and ion-neutral experiment keys, removed with the closures they
    # served. Selectors whose one surviving value is now unconditional name
    # that behaviour; the rest name nothing.
    "neutral_probe_amplitude_cm3_s": (
        "nothing: the ad-hoc probe neutral source is removed"
    ),
    "neutral_probe_profile": (
        "nothing: the ad-hoc probe neutral source is removed"
    ),
    "neutral_probe_shape": (
        "nothing: the ad-hoc probe neutral source is removed"
    ),
    "neutral_probe_center_cm": (
        "nothing: the ad-hoc probe neutral source is removed"
    ),
    "neutral_probe_width_cm": (
        "nothing: the ad-hoc probe neutral source is removed"
    ),
    "neutral_probe_waveform": (
        "nothing: the ad-hoc probe neutral source is removed"
    ),
    "neutral_probe_t_on_s": (
        "nothing: the ad-hoc probe neutral source is removed"
    ),
    "neutral_probe_t_off_s": (
        "nothing: the ad-hoc probe neutral source is removed"
    ),
    "neutral_probe_waveform_table": (
        "nothing: the ad-hoc probe neutral source is removed"
    ),
    "neutral_probe_zone": (
        "nothing: the ad-hoc probe neutral source is removed"
    ),
    "neutral_wall_partition_sigma_hehe_cm2": (
        "nothing: the wall-branch momentum partition it weighted is removed"
    ),
    "neutral_knudsen_temperature": (
        "nothing: the Knudsen conductances are evaluated once at Tn_K "
        "('frozen'), unconditionally"
    ),
    "neutral_momentum_radial": (
        "nothing: the evolved neutral wind is radially uniform ('uniform'), "
        "unconditionally"
    ),
    "neutral_kinetic_refresh_s": (
        "nothing: the relaxation-coupled kinetic neutral engine "
        "(neutral_model='kinetic') is removed; neutral_model='kinetic_dvm' "
        "is the kinetic neutral closure"
    ),
    "neutral_kinetic_refresh_tol": (
        "nothing: the relaxation-coupled kinetic neutral engine "
        "(neutral_model='kinetic') is removed; neutral_model='kinetic_dvm' "
        "is the kinetic neutral closure"
    ),
    "neutral_kinetic_nvz": (
        "nothing: the relaxation-coupled kinetic neutral engine "
        "(neutral_model='kinetic') is removed; neutral_model='kinetic_dvm' "
        "is the kinetic neutral closure"
    ),
    "neutral_kinetic_nvp": (
        "nothing: the relaxation-coupled kinetic neutral engine "
        "(neutral_model='kinetic') is removed; neutral_model='kinetic_dvm' "
        "is the kinetic neutral closure"
    ),
    "ion_neutral_drag_model": (
        "nothing: the legacy drag's neutral flow is closed by the constant "
        "b_ion_neutral_drag ('constant'), unconditionally"
    ),
    "b_ion_neutral_thermalization": (
        "nothing: the elastic ion-neutral thermalization term it scaled is "
        "removed"
    ),
    "coverage_growth_rate_per_s": (
        "nothing: the clumpy-plasma coverage closure is removed"
    ),
    "coverage_backfill_time_s": (
        "nothing: the clumpy-plasma coverage closure is removed"
    ),
    "coverage_initial_fraction": (
        "nothing: the clumpy-plasma coverage closure is removed"
    ),
    "coverage_initial_profile": (
        "nothing: the clumpy-plasma coverage closure is removed"
    ),
    # The adopted neutral, ion-neutral and gas-puff selections, whose keys
    # had one legal value left, and the puff shapes and waveforms they
    # retired.
    "neutral_exchange_model": (
        "nothing: axial neutral exchange is Knudsen transport "
        "('knudsen'), unconditionally"
    ),
    "neutral_exchange_coeff_cm3_s": (
        "nothing: the constant axial neutral exchange model it fed is "
        "removed"
    ),
    "neutral_kinetic_dvm_elastic": (
        "nothing: the kinetic_dvm arm carries the polarization-elastic "
        "channel ('phelps_iso'), unconditionally"
    ),
    "neutral_kinetic_dvm_wall_reflection": (
        "nothing: the kinetic_dvm arm returns the non-accommodated wall "
        "share on the energy-matched cosine spectrum "
        "('diffuse_elastic'), unconditionally"
    ),
    "gas_puff_mode": (
        "nothing: the gas puff is the square valve pulse, "
        "unconditionally; its edges are gas_puff_rise_center_s, "
        "gas_puff_rise_width_s and gas_puff_close_lag_s"
    ),
    "gas_puff_profile": (
        "nothing: the puff's axial shape is the tube-beamed orifice "
        "row, unconditionally; its pipe is gas_puff_orifice_id_cm and "
        "gas_puff_orifice_length_cm"
    ),
    "gas_puff_sigma_cm": (
        "nothing: the gaussian puff shape it widened is removed"
    ),
    "gas_puff_throw_cm": (
        "nothing: the cosine_pipe puff shape it set is removed"
    ),
    "gas_puff_local_ionization_fraction": (
        "nothing: the in-place puff ionization channel is removed (the "
        "two-zone puff routes through the annulus)"
    ),
    "S_gp_decay_target": (
        "nothing: the pulse, decay and double_erf puff waveforms it "
        "served are removed; the gas puff is the square valve pulse"
    ),
    "Twin_S_gp_decay_target": (
        "nothing: the pulse, decay and double_erf puff waveforms it "
        "served are removed; the gas puff is the square valve pulse"
    ),
    "tau_gp_after_breakdown": (
        "nothing: the pulse, decay and double_erf puff waveforms it "
        "served are removed; the gas puff is the square valve pulse"
    ),
    "tau_gp_decay_factor": (
        "nothing: the pulse, decay and double_erf puff waveforms it "
        "served are removed; the gas puff is the square valve pulse"
    ),
    "tau_gp_pulse_duration": (
        "nothing: the pulse, decay and double_erf puff waveforms it "
        "served are removed; the gas puff is the square valve pulse"
    ),
    "tau_gp_decay_duration": (
        "nothing: the pulse, decay and double_erf puff waveforms it "
        "served are removed; the gas puff is the square valve pulse"
    ),
    "tau_gp_rise_center": (
        "nothing: the pulse, decay and double_erf puff waveforms it "
        "served are removed; the gas puff is the square valve pulse"
    ),
    "tau_gp_rise_width": (
        "nothing: the pulse, decay and double_erf puff waveforms it "
        "served are removed; the gas puff is the square valve pulse"
    ),
    "tau_gp_drop_center": (
        "nothing: the pulse, decay and double_erf puff waveforms it "
        "served are removed; the gas puff is the square valve pulse"
    ),
    "tau_gp_drop_width": (
        "nothing: the pulse, decay and double_erf puff waveforms it "
        "served are removed; the gas puff is the square valve pulse"
    ),
    # The hydrogen path and the analytic-fit rate model, deleted. Helium and
    # the ADAS effective coefficients are unconditional.
    "gas_type": (
        "nothing: the species is helium, unconditionally; the hydrogen "
        "path is removed"
    ),
    "atomic_rate_model": (
        "nothing: the OPEN-ADAS effective coefficients ('adas') are "
        "unconditional; the analytic-fit 'janev' arm is removed"
    ),
    # Closed experiments and one-value legacy selectors, removed with the
    # code they gated. A one-value selector's behaviour is unconditional.
    "front_flux_model": _FRONT_FLUX_RETIRED,
    "alpha_front": _FRONT_FLUX_RETIRED,
    "D_amb_model": _AMBIPOLAR_RETIRED,
    "D_amb": _AMBIPOLAR_RETIRED,
    "sigma_in_model": (
        "nothing: the ion-neutral momentum-transfer rate is the Phelps "
        "He+/He cross section of the moment-closed collision operator, "
        "unconditionally"
    ),
    "end_mode": (
        "nothing: the far face is the chamber end wall, unconditionally"
    ),
    "electron_drift_charge_death": _ELECTRON_DRIFT_RETIRED,
    "electron_drift_anode_handshake": _ELECTRON_DRIFT_RETIRED,
    "tracer_passivity_current_ratio": _REGIME_TRACER_RETIRED,
    "tracer_passivity_thinness": _REGIME_TRACER_RETIRED,
    "tracer_passivity_depletion": _REGIME_TRACER_RETIRED,
    "tracer_passivity_hysteresis": _REGIME_TRACER_RETIRED,
    "tracer_refresh_tol": _REGIME_TRACER_RETIRED,
    "tracer_activation_ne": _REGIME_TRACER_RETIRED,
    "tracer_overlap_band_ne": _REGIME_TRACER_RETIRED,
    "tracer_overlap_rtol": _REGIME_TRACER_RETIRED,
    # The built-in end-expansion flare, deleted with its flag. A variable
    # area is the prescribed per-cell geometry.
    "end_expansion_cells": (
        "nothing: the end-expansion flare is removed; a variable-area end "
        "block is expressed by plasma_radius_profile_cm and "
        "machine_radius_profile_cm"
    ),
    "end_expansion_machine_radius_cm": (
        "nothing: the end-expansion flare is removed; a stepped vessel bore "
        "is expressed by machine_radius_profile_cm"
    ),
    "end_expansion_plasma_radius_cm": (
        "nothing: the end-expansion flare is removed; a flaring flux tube "
        "is expressed by plasma_radius_profile_cm"
    ),
    # The adopted numerics selector: its one surviving value is
    # unconditional.
    "hyperbolic_wave_speed": (
        "nothing: the Rusanov a_max and the plasma CFL use the adiabatic "
        "signal speed sqrt((5/3)(Te+Ti)/m_i), unconditionally"
    ),
}


#: The same register for ``input_flags``. Separate because the two namespaces
#: are separate: a retired flag name resurfacing in ``params`` is a misfiled
#: key, not a retired one, and must keep reading as the plain unknown key it
#: is.
RETIRED_FLAG_KEYS = {
    # The end-wall rename; see RETIRED_PARAM_KEYS above.
    "collector_sheath_full_debit": (
        "nothing: the end wall's sheath-climb row is armed wherever the "
        "geometry has an end wall face, unconditionally"
    ),
    "end_sheath_full_debit": (
        "nothing: the emitting cathode face's three sheath rows are armed "
        "wherever the geometry has a cathode face and the cathode circuit "
        "solve runs, and the end wall's sheath-climb row wherever the "
        "geometry has an end wall face, unconditionally"
    ),
    "cathode_face_full_debit": (
        "nothing: the emitting cathode face's three sheath rows (emitted "
        "enthalpy, virtual-cathode fall, collected climb) are armed wherever "
        "the geometry has a cathode face and the cathode circuit solve runs, "
        "unconditionally"
    ),
    "beam_tail_anode_interception": (
        "nothing: the QL tail walkers are culled at the anode mesh wherever "
        "the mesh is resolved and the closure walks a tail, on the same "
        "solid fraction eta and into the same anode_intercepted row as the "
        "primary's anode interception -- a mesh opaque to the streaming "
        "beam is opaque to its tail, so there was no second decision for a "
        "flag to carry"
    ),
    "end_wall_face_riemann_flux": (
        "nothing: the end wall face removes the PHYSICAL flux at the "
        "sheath-edge state it samples, unconditionally -- a sheath sends no "
        "wave back into the plasma, so there is no face kernel left to "
        "select between"
    ),
    # Circuit and cathode flags. The adopted ones name the behaviour that is
    # now unconditional; the deleted ones name nothing.
    "cathode_schottky": (
        "nothing: Schottky barrier lowering in the current-driven sheath "
        "solve is unconditional"
    ),
    "anode_sheath_full_debit": (
        "nothing: the anode sheath debit (2 Te + phi_a per collected "
        "electron at an electron-repelling anode, sheath-edge collection "
        "rows) is unconditional"
    ),
    "cathode_emission_bridge": (
        "nothing: the emission release keeps its hard space-charge corner"
    ),
    "cathode_emitting_area": (
        "nothing: the cathode emitting-area closure is removed"
    ),
    "cathode_enthalpy_on_beam": (
        "nothing: the emitted electrons' launch enthalpy stays in the "
        "cathode_e_emitted_enthalpy row at the cathode cell"
    ),
    "cathode_ion_secondary_emission": (
        "nothing: ion-induced secondary emission at the cathode face is "
        "removed"
    ),
    "coupled_circuit_picard": (
        "nothing: the fluid-circuit Picard coupling is removed; the circuit "
        "advances once per accepted step"
    ),
    "cathode_circuit_voltage_bound": (
        "nothing: the sheath ceiling is cathode_phi_c_cap_V alone"
    ),
    "cathode_circuit_project_over_wall": (
        "nothing: the over-wall projection of the circuit advance is removed"
    ),
    "regime_vessel_node": (
        "nothing: the vessel common-mode node is removed"
    ),
    # Beam flags. The adopted ones name the behaviour that is now
    # unconditional; the deleted one names nothing.
    "beam_anode_interception": (
        "nothing: the anode mesh intercepts its solid fraction eta of the "
        "CSDA beam at the anode-face crossing wherever the geometry resolves "
        "an anode face, unconditionally"
    ),
    "beam_deposition_in_heat_substep": (
        "nothing: the beam electron-energy deposition is applied by the "
        "implicit heat substep whenever implicit_heat_conduction is on, "
        "unconditionally"
    ),
    "beam_ionization_birth_timestep_bound": (
        "nothing: the beam ionization-birth row is in no timestep bound"
    ),
    # Neutral and ion-neutral experiment flags, removed with the closures
    # they served.
    "end_recycle_to_annulus": (
        "nothing: the end wall recycle is rebirthed on the column row, "
        "unconditionally"
    ),
    "neutral_hot_birth_drift": (
        "nothing: the hot channel's CX-born atoms launch at the local Ti "
        "alone, unconditionally"
    ),
    "neutral_probe_source": (
        "nothing: the ad-hoc probe neutral source is removed"
    ),
    "neutral_wall_momentum_partition": (
        "nothing: the wall-branch momentum partition is removed with the "
        "kinetic_two_moment radial closure"
    ),
    "ion_neutral_drag_cx_only": (
        "nothing: the legacy drag is driven by the total ion-neutral "
        "momentum-transfer frequency, unconditionally"
    ),
    "ion_neutral_thermalization": (
        "nothing: the elastic ion-neutral thermalization term is removed"
    ),
    "coverage_closure": (
        "nothing: the clumpy-plasma coverage closure is removed"
    ),
    # The initial-neutral-state flags, folded into one selector.
    "neutral_equilibration": (
        "initial_neutral_state: 'equilibrate' (or 'equilibrate_only' where "
        "launch_plasma_after_equilibration was off) for the ON value, and "
        "'fill' (or 'profile' where neutral_initial_profile was on) for OFF"
    ),
    "launch_plasma_after_equilibration": (
        "initial_neutral_state: 'equilibrate' launches the plasma run after "
        "the accumulation, 'equilibrate_only' stops there"
    ),
    "neutral_initial_profile": (
        "initial_neutral_state='profile'"
    ),
    # The adopted neutral and ion-neutral flags: the behaviour is
    # unconditional.
    "cx": (
        "nothing: charge-exchange cooling is carried by the "
        "moment-closed ion-neutral collision operator, unconditionally"
    ),
    "ion_neutral_drag": (
        "nothing: ion-neutral friction is carried by the moment-closed "
        "ion-neutral collision operator, unconditionally"
    ),
    "ion_neutral_moment_closure": (
        "nothing: the moment-closed ion-neutral collision operator is "
        "unconditional"
    ),
    "neutral_two_zone": (
        "nothing: the neutral gas is split into column and annulus "
        "zones, unconditionally"
    ),
    "neutral_kinetic_dvm_baffles": (
        "nothing: the kinetic_dvm arm applies the neutral baffle "
        "geometry to its annulus wherever the geometry carries baffles"
    ),
    "neutral_prebreakdown": (
        "tau_neutral_prebreakdown: the phase runs whenever its duration "
        "is positive"
    ),
    # Closed experiments and one-value legacy flags; see RETIRED_PARAM_KEYS.
    "regime_tracer": _REGIME_TRACER_RETIRED,
    "front_flux": _FRONT_FLUX_RETIRED,
    "electron_drift_transport": _ELECTRON_DRIFT_RETIRED,
    "rates_at_accepted_state": (
        "nothing: the rate-freezing instrument is removed; the reaction "
        "terms are evaluated at the stage state"
    ),
    "resolved_boundaries": (
        "nothing: the resolved typed-segment geometry is unconditional"
    ),
    "icool_recomb": (
        "recombination_energy_return, which charges the ADAS PRB together "
        "with the recombination binding-energy credit; bare PRB charging is "
        "removed"
    ),
    # The geometry flags. The deleted one names nothing; the adopted ones
    # name what now arms the behaviour.
    "end_expansion_geometry": (
        "nothing: the end-expansion flare is removed; a variable-area "
        "machine is plasma_radius_profile_cm (with machine_radius_profile_cm "
        "for the vessel)"
    ),
    "prescribed_area_geometry": (
        "nothing: the prescribed per-cell geometry is armed by the presence "
        "of plasma_radius_profile_cm"
    ),
    "neutral_baffles": (
        "nothing: the baffles are placed by the presence of "
        "neutral_baffle_positions_cm and neutral_baffle_clear_radii_cm"
    ),
    "end_wall_sheath_full_debit": (
        "nothing: the end wall's sheath-climb row is armed wherever the "
        "geometry has an end wall face, unconditionally"
    ),
    "source_fixed_grid": (
        "nothing: the mesh always carries the fixed source region "
        "(source_region_length_cm, source_region_dz_cm), mirrored onto the "
        "far cathode end under TwinCathode"
    ),
    # The adopted numerics flags: the behaviour is unconditional.
    "active_plasma_topology": (
        "nothing: the typed plasma topology (closed faces carry their live "
        "cell's pressure, plasma-dead cells are masked out of every "
        "plasma-coupled term) is unconditional"
    ),
    "hyperbolic_energy_consistent": (
        "nothing: the kinetic-energy-preserving momentum flux and the "
        "hyperbolic_dissipation_heating deposit are unconditional"
    ),
    "raw_stage_validation": (
        "nothing: raw invalid stages are rejected before any floor "
        "projection, unconditionally"
    ),
    "surface_loss_floor_exempt": (
        "nothing: the floor-aware drain exemption on the surface_loss "
        "timestep bound is unconditional; surface_loss_floor_exempt_exit_rtol "
        "still sets its re-admission band"
    ),
    "electron_heat_flux_limit": (
        "nothing: the electron heat-flux limiter is unconditional; "
        "heat_flux_limiter_f and heat_flux_limiter_exponent still set it"
    ),
}


#: SAVED-ARTIFACT key aliases, retired name -> current name. Used ONLY when a
#: params/flags block is read back off a stored file that was written before a
#: rename, so a reader can address it under the names the code now uses.
#:
#: It is deliberately NOT consulted by :func:`resolve_config`: a LIVE
#: configuration naming a retired key is refused there, with its successor
#: named, because a silently accepted alias is exactly the inert control that
#: boundary exists to forbid. Reading a stored file is the opposite case --
#: the file cannot be edited to say something else, and the value in it is a
#: fact about a run that already happened.
LEGACY_CONFIG_KEY_ALIASES = {
    "collector_length_cm": "end_wall_length_cm",
    "collector_sheath_full_debit": "end_wall_sheath_full_debit",
    "neutral_kinetic_dvm_collector_jet": "neutral_kinetic_dvm_end_wall_jet",
    "neutral_kinetic_dvm_collector_jet_R_N":
        "neutral_kinetic_dvm_end_wall_jet_R_N",
    "neutral_kinetic_dvm_collector_jet_R_E":
        "neutral_kinetic_dvm_end_wall_jet_R_E",
    "neutral_kinetic_dvm_collector_jet_T_launch_eV":
        "neutral_kinetic_dvm_end_wall_jet_T_launch_eV",
    "neutral_kinetic_dvm_collector_jet_sheath_Te_multiple":
        "neutral_kinetic_dvm_end_wall_jet_sheath_Te_multiple",
}

#: The retired VALUES a stored params block may carry, per key.
LEGACY_CONFIG_VALUE_ALIASES = {
    "end_mode": {"collector": "end_wall"},
}


def apply_legacy_config_key_aliases(mapping):
    """Return a stored params/flags block under the CURRENT key names.

    Presence-gated in both directions: a block naming none of the retired keys
    is returned as an unchanged copy, and a retired key whose current name is
    ALSO present is left alone rather than overwriting the current one -- two
    spellings of one control in one stored block is a corrupt file, not an
    alias to resolve.
    """
    out = dict(mapping)
    for old, new in LEGACY_CONFIG_KEY_ALIASES.items():
        if old in out and new not in out:
            out[new] = out.pop(old)
    for key, values in LEGACY_CONFIG_VALUE_ALIASES.items():
        if key in out and out[key] in values:
            out[key] = values[out[key]]
    return out


def resolve_config(params=None, flags=None, models=None):
    """Resolve caller overrides against the one authoritative default registry.

    Unknown keys fail at this boundary so misspelled or retired campaign
    controls cannot survive as silent metadata-only settings. A key listed in
    :data:`RETIRED_PARAM_KEYS` or :data:`RETIRED_FLAG_KEYS` is refused with
    its successor named, so a configuration file written before the removal
    reports where its value should go rather than only that the key is gone.
    The refusal fires on ANY use of a retired name, its old default included:
    the key owns no read any more, so stating it would be exactly the silent,
    inert control this boundary exists to forbid.

    ``models`` carries DECLARATION BLOCKS -- ``{family: {member: value}}`` --
    which are projected onto the two flat namespaces before the merge, so
    everything downstream reads the one flat surface it always has. Omitting
    it (the default) leaves this function's behaviour and its output
    bit-identical to the flat-only form.
    """
    # Imported here rather than at module scope: model_declarations reads the
    # templates this module builds, so a top-level import would close a cycle
    # through a half-initialised config module.
    from .model_declarations import resolve_declaration_blocks

    supplied_params = {} if params is None else dict(params)
    supplied_flags = {} if flags is None else dict(flags)
    block_params, block_flags = resolve_declaration_blocks(
        models, supplied_params, supplied_flags
    )
    unknown_params = sorted(set(supplied_params) - set(input_dict_template_1d))
    unknown_flags = sorted(set(supplied_flags) - set(input_flags_template_1d))
    if unknown_params or unknown_flags:
        details = []
        if unknown_params:
            details.append(f"params={unknown_params}")
        if unknown_flags:
            details.append(f"flags={unknown_flags}")
        message = (
            "unknown LAPDSim1D configuration keys (silent/inert controls are "
            f"forbidden): {', '.join(details)}"
        )
        retired = [
            f"{key} is RETIRED; use {successor}"
            for key, successor in sorted(RETIRED_PARAM_KEYS.items())
            if key in unknown_params
        ] + [
            f"{key} is RETIRED; use {successor}"
            for key, successor in sorted(RETIRED_FLAG_KEYS.items())
            if key in unknown_flags
        ]
        if retired:
            message = f"{message}. {'. '.join(retired)}"
        raise ValueError(message)
    resolved_params = dict(input_dict_template_1d)
    resolved_params.update(supplied_params)
    resolved_params.update(block_params)
    resolved_flags = dict(input_flags_template_1d)
    resolved_flags.update(supplied_flags)
    resolved_flags.update(block_flags)
    return resolved_params, resolved_flags


def canonical_config_payload(params, flags):
    """Return the canonical JSON text of one resolved ``(params, flags)`` pair.

    Sorted keys, no whitespace, no NaN: the same text whatever order the two
    mappings were built in, so two configurations that resolve to the same
    values produce the same bytes. This is the payload
    :func:`config_identity` hashes.
    """
    return json.dumps(
        {"params": params, "flags": flags},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def config_identity(params, flags):
    """Return the sha256 of a resolved configuration.

    The identity of a CONFIGURATION: two configurations share it exactly when
    every resolved parameter and flag agrees, whatever route built them -- a
    stance file, a derived file's deltas, or command-line overrides. It is the
    digest ``scripts/gates/audit_sim1d_configs.py`` pins its reviewed snapshots
    with, and it is what a run records so a saved trajectory can be matched to
    the configuration that produced it.
    """
    return hashlib.sha256(
        canonical_config_payload(params, flags).encode()
    ).hexdigest()


@dataclasses.dataclass(frozen=True)
class ConfigurationLineage:
    """WHICH configuration a run names, and what it was derived from.

    Every field answers one question a reader of a saved artifact asks:

    ``name``
        The configuration's name -- the stem of the file that declares it.
    ``base_chain``
        The names it is derived FROM, nearest base first, empty for a base
        configuration that derives from nothing.
    ``file_sha256``
        The sha256 of each file in the chain, this file first and then
        ``base_chain`` in order, so the chain can be checked byte for byte
        against the committed files.
    ``delta_keys``
        The names -- names only, never values -- of the keys this file moves
        relative to its base. Values live in the recorded config itself.
    ``identity``
        The :func:`config_identity` of the RESOLVED configuration. Two runs
        that agree here ran the same configuration.

    Frozen, and carried by value: a run's lineage is a statement about what it
    was, and nothing downstream may edit it. Use :meth:`with_identity` to state
    the identity of a configuration a driver finished resolving.
    """

    name: str
    base_chain: tuple
    file_sha256: tuple
    delta_keys: tuple
    identity: str

    def with_identity(self, params, flags):
        """Return this lineage with ``identity`` taken from a resolved config.

        For a driver that layers its own mesh package over the named
        configuration: the name, chain and deltas are unchanged -- they are
        facts about the FILE -- while the identity becomes that of the
        configuration the driver actually constructs.
        """
        return dataclasses.replace(
            self, identity=config_identity(params, flags)
        )


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


def resolve_nn0(input_dict):
    """Return the configured uniform initial neutral density [cm^-3].

    ``nn0`` has no fallback. It used to have one -- a ``None`` was resolved
    from the frozen gas-puff lookup table, keyed on ``S_gp`` and
    ``TwinCathode`` -- and that table is RETIRED. It could not be regenerated
    in-tree (its generator drove the removed 0D _sim3 solver), and its keys
    were pre-2026-08-21 0 C-sccm while ``S_gp`` has meant meter-sccm since,
    so every lookup was off by the ~7% conversion ratio and could not be
    converted without inventing an interpolation of data that was never
    computed.

    It was NOT unreached, and the retirement record should not pretend
    otherwise: the golden gate's own configuration arrived here with a
    ``None`` and took the table's answer as its uniform neutral fill. The
    stance pins a per-cell profile and ``nn0 = None`` to go with it, and the
    gate's coarse-mesh re-cut dropped the profile without restoring a scalar,
    which reopened this branch. That value is now an explicit literal in the
    golden builder, pinned before this fallback was removed, so the fill the
    gate starts from is written down instead of looked up.

    Raises ``ValueError`` on a ``None``, which under the solver's call order
    is a construction-time refusal. ``None`` is still the REQUIRED value under
    ``initial_neutral_state = "profile"`` -- that path supersedes the scalar
    with a per-cell array and does not call this function at all.
    """
    nn0 = input_dict.get("nn0")
    if nn0 is None:
        raise ValueError(
            "nn0 is None and there is no table to resolve it from (the "
            "frozen gas-puff nn0 table was retired). nn0 accepts a uniform "
            "initial neutral density in cm^-3; None is accepted ONLY under "
            "initial_neutral_state='profile', which supersedes the scalar "
            "with the per-cell nn0_profile array."
        )
    return nn0
