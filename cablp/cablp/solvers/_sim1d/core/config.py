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
        without disturbing any equilibrated run. Since ``neutral_equilibration``
        ships ON, the uniform value is a PLACEHOLDER that no shipped
        configuration reads -- equilibration is the convention for the fill a
        run starts from.

        Provenance of the shipped value: ``config_defaults_provenance.md``.
    nn0_profile:
        PER-CELL initial neutral density [cm^-3]: a sequence of length ``nx``
        (the grid's cell count), every entry finite and ``> 0``. Read ONLY
        under the ``neutral_initial_profile`` flag, and REQUIRED by it; with
        the flag off it must be ``None`` or construction raises.

        Supplied as VALUES, not as a shape: these are the absolute densities
        the run starts from, cell by cell, and nothing rescales or normalizes
        them. This is the externally-computed-profile hook for the INITIAL
        CONDITION -- the solver does no file I/O, so a hypothesized axial fill
        is built outside and passed here. (Its source-side counterpart,
        ``neutral_probe_profile``, is a normalized SHAPE; the two are not
        interchangeable.)

        It supersedes the scalar ``nn0`` for BOTH zones, so ``nn0`` must be
        ``None`` when the flag is armed -- an armed flag with a non-``None``
        scalar raises rather than establishing a silent precedence. Neither
        ``resolve_nn0`` nor the ``nn_table`` lookup is consulted on the armed
        path.
    nn0_annulus_profile:
        PER-CELL initial ANNULUS neutral density [cm^-3] under the
        ``neutral_two_zone`` closure: same length, finiteness and positivity
        rules as ``nn0_profile``. Read ONLY under the
        ``neutral_initial_profile`` flag, and valid only with
        ``neutral_two_zone`` on -- setting it without that closure raises,
        because there is no second neutral field for it to land in.

        OPTIONAL when armed: omitted, the annulus starts at ``nn0_profile``,
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
        # Shaped initial neutral fill (neutral_initial_profile flag). Both are
        # None on every shipped configuration: a per-cell IC has no default
        # shape to inherit, and the flag's whole content is what the caller
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
        Number of resolved column cells between anode and collector. Under the
        ``source_fixed_grid`` flag it counts only the *far* column cells,
        between the source region end and the collector.
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
    plasma_radius_profile_cm:
        PER-CELL effective plasma flux-tube radius [cm]: a sequence with one
        entry per MESH cell (``geometry.cells`` -- the plenum, gap, column and
        end cells, not just the ``nx`` column cells), every entry finite and
        ``> 0``. Read ONLY under the ``prescribed_area_geometry`` flag, and
        REQUIRED by it; with the flag off it must be ``None`` or construction
        raises.

        It replaces the uniform scalar ``Rp`` cell by cell, so the plasma
        cross-section is ``pi r(z)^2``, the cell volume ``pi r(z)^2 dz``, and
        the face area the average of the two adjacent cells -- the same
        expressions the uniform column uses, evaluated on a vector. A profile
        holding ``Rp`` in every cell is therefore bit-identical to no profile
        at all.

        The quantity the flag prescribes is the AREA ``A(z)`` (the flux-tube
        variable, ``A B = const``); the radius ``sqrt(A/pi)`` is how it is
        supplied, because that parameterization is what makes the constant
        profile exact rather than exact-to-a-rounding. Any conversion from a
        solved ``B(z)`` happens outside the solver, which does no file I/O.

        Supplied as VALUES, not as a shape: nothing rescales or normalizes
        them, and no cell is masked by role. Every entry must satisfy
        ``pi r^2 <= `` the local vessel open area, since the column zone
        cannot be larger than the chamber holding it (the two-zone annulus
        volume ``V_ann = Vm - Vp`` would go negative and be clipped to zero
        silently). REFUSES ``end_expansion_geometry``: both prescribe the
        end-block flux-tube area and there is no composition rule.
    machine_radius_profile_cm:
        PER-CELL vessel/neutral radius [cm], the same per-mesh-cell form and
        the same finiteness/positivity rules as ``plasma_radius_profile_cm``.
        Read ONLY under the ``prescribed_area_geometry`` flag, where it is
        OPTIONAL: omitted, every cell keeps the scalar ``Rm`` exactly as
        before. It replaces that scalar cell by cell, setting the neutral open
        area ``pi Rm(z)^2``, the neutral cell volume, and the hydraulic radius
        that sets the free-molecular face conductance -- so a vessel whose
        bore STEPS partway along a cell block is expressible, which a single
        ``Rm`` (or ``end_expansion_machine_radius_cm``, one value over the
        whole terminal block) is not.

        Composes with the annular-duct and support-rod reductions rather than
        overriding them: an obstruction cell keeps its open area
        ``pi (Rm(z)^2 - Rcs^2)`` and hydraulic radius ``Rm(z) - Rcs``, and a
        plenum keeps ``pi (Rm(z)^2 - Rsup^2)``. Every entry must be ``>=`` the
        local ``plasma_radius_profile_cm`` entry; the vessel cannot be
        narrower than the plasma it contains.
    plasma_area_max_vessel_fraction:
        Optional ceiling on the prescribed plasma area as a fraction of the
        local vessel open area, in ``(0, 1]``. ``None`` (the default) applies
        no ceiling. Read ONLY under the ``prescribed_area_geometry`` flag.

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
        under the ``neutral_two_zone`` closure [1]. Cells with no annulus at
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
        ``source_fixed_grid`` flag, and is required by it. For the value's
        provenance see ``scripts/production_stance_provenance.md``.
    source_region_dz_cm:
        Cell size [cm] inside that source region, held fixed independently of
        ``nx``; the region length minus the anode gap must be an integer
        multiple of it (1e-9 relative tolerance). ``None`` when off. Requires
        the ``source_fixed_grid`` flag, and is required by it.
    """
    return {
        "Lm": 2117.8,
        "nx": 60,
        "Rm": 50.0,
        "Rp": 18.415,
        "plenum_length_cm": 166.0,
        "cathode_anode_gap_cm": 53.25,
        "nx_gap": 5,
        "collector_length_cm": 7.8,
        "Rcs": 0.0,
        "Lcs": 0.0,
        "Rsup": 0.0,
        "end_expansion_cells": None,
        "end_expansion_machine_radius_cm": None,
        "end_expansion_plasma_radius_cm": None,
        # Prescribed per-cell flux-tube and vessel radii, plus the optional
        # area ceiling (prescribed_area_geometry flag). All None on every
        # shipped configuration: a per-cell geometry has no default shape to
        # inherit, and the flag's whole content is what the caller computed
        # outside.
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
# set the input_dict key adas_low_te_extension=True together with the
# input_flags key icool_recomb=True. The extension makes
# acd (recombination) and prb1 (recombination radiation) extend consistently
# below the 0.2 eV ADF11 edge; scd (ionization) and plt (line power) still
# clamp there but are exponentially dead at <0.2 eV, so a recombining 300 K
# afterglow would be well represented.
#
# icool_recomb TOGETHER WITH adas_low_te_extension RAISES at construction. The
# two compose destructively: icool_recomb charges bare PRB (the double-charge
# warned about at recombination_energy_return below), and adas_low_te_extension
# amplifies the sub-edge PRB by ~9,300x, so the electron fluid runs away
# thermally to the floor and the electron_cooling timestep bound collapses
# permanently. The consistent net booking (I_ion*S_rec - P_PRB) that would make
# the pair sound is NOT built. Without it the afterglow validity window is
# Te > 0.2 eV (the ADF11 edge).


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
    gas_puff_rise_center_s:
        ``"square"`` opening-edge center [s], measured from the end of the
        neutral-prebreakdown phase (the instant the cathode circuit closes),
        so the opening edge does not wait for breakdown. Must be ``>= 0``.
    gas_puff_rise_width_s:
        Erf width scale [s] shared by BOTH ``"square"`` edges -- the opening
        edge and the closing edge are built with this one width. The 10-90%
        transition time is ~1.81x this value. Must be positive.
    gas_puff_close_lag_s:
        Delay [s] from the end of the main discharge (``tau_discharge`` after
        the main-discharge start) to the ``"square"`` closing-edge center, so
        the closing tail runs on past the drive. Must be ``>= 0``.

        The three ``"square"`` timings above are read and validated only in
        that mode; a bad value raises at construction. The envelope is
        ``max(rise - fall, 0)``, so edges configured to overlap clamp at zero
        flow rather than going negative.
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
        where ``gas_puff_valves`` does, at both conversion sites (the
        ``"cell"`` profile and the distributed profiles), so it scales the
        puff magnitude without touching its axial shape or its waveform, and
        it applies to the source-end and twin-end puffs alike. Consumed by the
        neutral gas-puff source term in ``physics.neutrals`` -- the explicit
        RHS, the implicit backward-Euler neutral matrices, the local-ionization
        channel, and the saved ``puff_particles_per_s`` diagnostic all read the
        same value, so none can desync. Must be in ``(0, 1]`` and finite; a
        value outside that range raises at construction. ``1.0`` (the default)
        is the identity and is bit-exact.
    gas_puff_profile:
        Axial shape of the puff. ``"cell"`` (the historical
        behaviour) puts the whole flow in the role-tagged puff cell, which
        under ``source_fixed_grid`` follows ``gas_puff_z_cm`` and otherwise
        is the column cell against the anode face. ``"cosine_pipe"``
        (default) is the physical source -- a small pipe at the chamber wall,
        at the measured mid-plane puff ports on the anode stack, pointing
        radially inward with a Lambertian (cosine) outlet;
        its first-flight axial deposition is the cosine-lobe pattern
        ``[1 + ((z - z0)/d)^2]^-2`` with throw ``d ~ 2*Rm``, so centre and
        width both come from geometry rather than tuning. ``"gaussian"`` is
        the generic tunable shape. ``"orifice"`` is the tube-beamed injection
        row: the feed pipe at the same ports is treated as a collimating tube
        in free-molecular flow, and the row is the ray-optics first-flight
        landing distribution of its exit distribution on the plasma column,
        with the wall and column radii read off the grid at the port cell. It
        requires ``gas_puff_orifice_id_cm`` and ``gas_puff_orifice_length_cm``
        and rejects them under any other profile. Unlike the other
        distributed shapes it is not re-weighted by cell length and not masked
        to the main-chamber roles -- it lands where the rays land. All
        distributed profiles conserve the total inflow exactly, and one shared
        implementation feeds both the explicit RHS and the implicit neutral
        matrix, so the two sites cannot desync.
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
    gas_puff_orifice_id_cm:
        Inner diameter of the collimating feed pipe [cm], the emitting
        aperture of the ``"orifice"`` profile. ``None`` (the default) is the
        only value accepted under any other profile, and ``"orifice"`` refuses
        to construct without it. Must be finite and positive.
    gas_puff_orifice_length_cm:
        Length of that same feed pipe [cm]. Only its ratio to
        ``gas_puff_orifice_id_cm`` enters -- that aspect ratio is the beaming
        parameter of the tube's exit distribution, and the row narrows as it
        grows. Must be finite, positive, and at least 4/3 of the bore, below
        which the long-tube expression has no branch and construction raises.
        Same presence gating as ``gas_puff_orifice_id_cm``.
    gas_puff_local_ionization_fraction:
        Fraction of the gas-puff neutral source ionized IN PLACE rather than
        added to the background neutral density. The diverted neutrals are
        debited from the puff and booked as an ionization source under the
        same birth-temperature and ionization-cost conventions as bulk
        ionization (``Te_birth_ionization``, ``Ti_birth_ionization``,
        ``ionization_birth_energy_model``), so the term conserves mass and
        energy. It is built from the configured puff shape and waveform, so
        it is localized wherever the puff is and follows the same time
        dependence. Must be in ``[0, 1)``; ``0`` returns a zero source
        without evaluating the term. Raises at construction outside that
        range, or if the ``neutral_two_zone`` flag is on (which routes the
        puff through the annulus zone instead). Inert while the gas puff is
        disabled.
    pump_elbow_conductance_lps:
        Conductance of the unmodeled pump elbow [L/s], combined in series with
        the pump speed as ``1/S_eff = 1/S_pump + 1/C_elbow``. Applies only to a
        pump sitting on a plenum cell, so it is inert in legacy geometry.
        ``None`` (default) or a
        non-positive value means no elbow restriction -- the legacy limit.
        Because ``S_pump_L``/``S_pump_R`` are lumped per-end speeds that
        already carry their own ducting, setting this alongside them applies
        the same restriction twice on the source side.

    Values and their provenance: ``config_defaults_provenance.md``.
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
        # Physical Lambertian pipe source at the measured mid-plane puff ports
        # on the anode stack; its centre is measured and its width is
        # geometry-derived, and neither is tunable.
        "gas_puff_profile": "cosine_pipe",
        # The pipe position, in machine coordinates so it does not move with nx.
        "gas_puff_z_cm": 86.3,
        "gas_puff_sigma_cm": 50.0,
        "gas_puff_throw_cm": 100.0,
        # The collimating feed pipe behind those ports. Both None off the
        # "orifice" profile, which is the only consumer and requires both.
        "gas_puff_orifice_id_cm": None,
        "gas_puff_orifice_length_cm": None,
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
        # Same FITTED-FLUX rescale as S_gp above (1500 -> 1610.23); the twin
        # target is zero and is invariant under any conversion.
        "S_gp_decay_target": 1610.23,
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

    Values and their provenance: ``config_defaults_provenance.md``.
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
        End boundary behaviour, carried into the cathode-boundary
        diagnostics as a label. ``"collector"`` is the only accepted value;
        the 0D-era ``"mirrored_source"`` alternative, which the conservative
        solver never branched on, was removed at D3 (2026-08-21) and raises.
    cathode_model:
        Cathode model selector retained for configuration compatibility. The
        current option is ``"disabled"``; actual cathode coupling is controlled
        by the ``cathode_coupling`` flag.
    Te_birth_ionization:
        Electron birth temperature model for ionization. Options are
        ``"local"`` to use the local electron temperature, ``"floor"`` to use
        the electron temperature floor, or a numeric eV value.
    Ti_birth_ionization:
        Ion birth temperature model for ionization -- the temperature the ion
        BORN by bulk ionization, by a beam ionization, and by the gas-puff
        local-ionization channel carries. Options are ``"local"`` (the local
        ion temperature), ``"floor"`` (the ion temperature floor),
        ``"neutral"`` (the default), or a numeric eV value.

        ``"neutral"`` is the option that PAIRS with the neutral energy field:
        the ion is born at the local neutral temperature
        ``Tn = (2/3) En / (nn k)`` of the very population the ``En``
        ionization sink debits (the column ``nn`` under ``neutral_two_zone``),
        so the ion gains exactly the ``(3/2) k Tn`` per particle the neutral
        gas gives up and the pair conserves energy by construction. With
        ``neutral_energy`` off the state carries no ``En``, there is no local
        neutral temperature, and the birth falls back to the cold-gas scalar
        ``Tn_K``.

        ``"floor"``, ``"local"`` and a numeric value are NON-CONSERVING
        against an evolved ``En``: the sink still removes ``(3/2) k Tn`` per
        ionized atom while the ion is born at an unrelated temperature, and
        the difference leaves the model. That difference is reported per cell
        and per save by the ``ionization_birth_thermal_deficit_*_W_cm3``
        diagnostic rows, which read zero to roundoff under ``"neutral"``. They
        stay selectable, and warn, so a pre-adoption artifact can be reproduced
        bit-for-bit.
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
    neutral_knudsen_temperature:
        Which temperature the Knudsen conductances take their thermal speed
        from. Read only under the ``neutral_energy`` flag, which is the only
        setting in which a second answer exists.

        ``"frozen"`` (default, and the ratified v1-primary) evaluates every
        conductance once at the configured ``Tn_K``, so axial and radial
        neutral transport is a fixed property of the geometry.

        ``"local"`` scales each conductance by ``sqrt(Tn_local / Tn_K)`` from
        the evolved per-cell neutral temperature -- the thermal-transpiration
        arm, in which a hot patch of gas conducts faster than a cold one.

        WHAT THIS ARM DOES AND DOES NOT DO. It scales the RATE, not the
        equilibrium. The driving potential stays the density difference, so a
        temperature gradient at uniform density still drives no flow and the
        steady state is still uniform ``nn``: this is not the textbook
        transpiration relation ``n ~ 1/sqrt(T)``, which would require the
        driving potential itself to change and is NOT built. It is a DISCLOSED
        sensitivity arm on the transport timescale, not the production closure.
        Selecting it without ``neutral_energy`` raises at construction, because
        there is no per-cell ``Tn`` for it to read.
    neutral_model:
        Which engine carries the neutral population. ``"moment"`` integrates
        the fluid neutral density (and, with the ``neutral_momentum`` flag,
        its momentum) directly from the conservative RHS terms.
        ``"kinetic"`` drives the neutrals toward per-cell targets and
        relaxation times produced by a velocity-space kinetic solve of the
        two-zone neutral state. Those solves run at step ACCEPTANCE only --
        never inside a trial RHS evaluation, so step retries are
        deterministic -- and only once the plasma phase is live; the
        neutral-prebreakdown fill is carried by the moment terms in both
        settings, as is the whole run until the first refresh completes.
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
        book unchanged. Like ``"kinetic"`` it rides on the two-zone state and
        engages only once the plasma phase is live, so the pre-breakdown fill
        and the neutral equilibration stay on the moment terms.

        ``"kinetic_dvm"`` is a top-level MODEL SELECTION and OWNS a member
        set -- every control in it is M_n or En physics the kinetic state
        already carries, so the two cannot both own it. The measured set is
        ``core/model_families.KINETIC_DVM_INCOMPATIBLE_DEFAULTS``, which is
        the authority here: the ``neutral_momentum``, ``neutral_energy``,
        ``neutral_hot_internal_wall`` and ``neutral_hot_birth_drift`` flags,
        the ``neutral_momentum_radial`` and ``neutral_knudsen_temperature``
        selectors, the cathode jet (``cathode_neutral_jet``,
        ``cathode_jet_surface_debit``, ``cathode_jet_energy_convention``,
        ``cathode_jet_hot_carrier``) and the anode-side momentum channel
        (``anode_neutral_jet``, ``anode_jet_energy_convention``,
        ``neutral_mesh_accommodation``). A member left at its config default
        is set to the value this selection requires AUTOMATICALLY, before
        any validator runs -- nothing has to be hand-cleared. A member the
        caller set EXPLICITLY to a value the selection refuses raises one
        ``ValueError`` at construction naming the selection, the whole
        member set and every offending key.

        Outside that set the arm still requires the ``neutral_two_zone``
        flag (a prerequisite, never resolved for you) and refuses
        ``coupled_circuit_picard``, a nonzero
        ``gas_puff_local_ionization_fraction``, and ``gas_type`` other than
        ``"He"``. Any other value, or ``"kinetic"``/``"kinetic_dvm"`` without
        the two-zone flag, raises at construction.
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
    neutral_kinetic_dvm_accommodation:
        Thermal accommodation coefficient of the chamber surfaces, in
        ``[0, 1]``. The accommodated fraction is re-emitted cosine-distributed
        at the wall temperature (300 K, or the live cathode surface
        temperature on the cathode-adjacent end); the remaining fraction is
        reflected at the incident energy, which on this axisymmetric grid is
        bin-preserving at the cylindrical wall and an exact bin mirror at an
        end wall. A boxed surface property, never a fit parameter. Raises at
        construction outside ``[0, 1]``.
    neutral_kinetic_dvm_elastic:
        Polarization-elastic ion-neutral channel of the transient DVM.
        ``"phelps_iso"`` adds a BGK-like relaxation toward the local ion
        Maxwellian at the Phelps isotropic rate, alongside the charge-exchange
        channel at the Phelps backscatter rate; ``"off"`` drops it, leaving
        charge exchange to carry all ion-neutral momentum transfer. The arm
        supersedes the fluid ion-neutral collision family wholesale and that
        operator's momentum-transfer cross section is ``Qi + 2 Qb``, so
        ``"off"`` deliberately omits the ``Qi`` half. Any other value raises
        at construction.
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
        selecting ``"bounded_chord"`` without that model and the
        ``neutral_two_zone`` flag, or any other value, raises at
        construction.
    neutral_kinetic_dvm_tn_feedback:
        Whether the DVM's measured neutral temperature ``Tn(z)`` FEEDS the
        fluid evaluations that otherwise assume a fixed cold gas. The
        temperature moment is computed as a diagnostic whenever the arm is
        on; this switch controls consumption only, so that the
        assumed-300 K versus measured comparison is a clean A/B. Under this
        arm the single surviving consumer is the collisional presheath depth
        behind the sheath-edge sampling, whose ``T_eff = (Ti + Tn)/2`` sets
        ``nu_in``. Requires ``characteristic_boundary`` to be OFF: in the
        characteristic stance the circuit's cathode sheath factor samples the
        same presheath through a path that does not carry ``Tn``, and letting
        only one of the two move would break the shared sheath-edge density
        those two deliberately agree on. Raises at construction otherwise.
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
    neutral_kinetic_refresh_s:
        Maximum interval [s] between full kinetic solves under
        ``neutral_model = "kinetic"``. Between full refreshes the targets are
        updated from the response functions frozen at the last one. Must be
        positive; raises at construction otherwise. Inert under ``"moment"``.
    neutral_kinetic_refresh_tol:
        Relative drift in the neutral absorption field that forces a full
        refresh early, ahead of ``neutral_kinetic_refresh_s``. The absorption
        field is the response functions' only staleness channel; the test is
        the maximum over cells of ``|nu_ion - nu_ref| / max(|nu_ref|, 1e2)``
        against this value, so a larger tolerance permits staler response
        functions between refreshes. Inert under ``"moment"``.
    neutral_kinetic_nvz:
        Number of axial-velocity (``v_z``) bins in the kinetic engine's
        shared velocity grid. The axis is signed and stretched, placing bins
        finely near zero. Inert under ``"moment"``.
    neutral_kinetic_nvp:
        Number of perpendicular-speed (``v_perp``) bins in that same grid.
        This axis is positive-only -- it carries the 2D perpendicular speed
        measure -- and is likewise stretched. Inert under ``"moment"``.
    adas_low_te_extension:
        Extends the ADAS ``acd`` (recombination) and ``prb1`` (recombination
        radiated power) coefficients consistently below the bundled ADF11
        low-Te edge at 0.2 eV, where the lookups otherwise clamp to the edge
        value. ``False`` keeps the clamp. Read by the reaction and energy
        terms only under ``atomic_rate_model = "adas"``; ``scd`` (ionization)
        and ``plt`` (line power) clamp at the edge either way. Raises at
        construction when combined with the ``icool_recomb`` flag: the two
        compose destructively -- see the module note above
        ``neutral_source_defaults``.
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
        "Ti_birth_ionization": "neutral",
        # "conservative": no spurious 3Te/2 electron birth energy; ion
        # mass-loading mixing energy booked explicitly.
        "ionization_birth_energy_model": "conservative",
        "neutral_exchange_model": "knudsen",
        # The ratified v1-primary: conductances frozen at Tn_K. Read only under
        # the neutral_energy flag, which ships ON.
        "neutral_knudsen_temperature": "frozen",
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
        # Dead under neutral_model="moment" (the K2a transient DVM arm, also
        # gated on neutral_two_zone, is the only consumer). The cadence is
        # PROVISIONAL -- a conservative placeholder, not an accuracy result:
        "neutral_kinetic_dvm_cadence_s": 2.5e-5,
        "neutral_kinetic_dvm_nvz": 48,
        "neutral_kinetic_dvm_nvp": 12,
        "neutral_kinetic_dvm_accommodation": 1.0,
        "neutral_kinetic_dvm_elastic": "phelps_iso",
        "neutral_kinetic_dvm_exchange": "cauchy_chord",
        "neutral_kinetic_dvm_annulus_flights": "rates",
        "neutral_kinetic_dvm_tn_feedback": False,
        "neutral_kinetic_dvm_transfer_relax_fraction": 0.5,
        # None = "not named"; resolved to "exponential" by the arm, and
        # refused outright by every other neutral model, so the key can
        # never be a silently inert control:
        "neutral_kinetic_dvm_transfer_hold": None,
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
    recombination_energy_return:
        Books the GCR-consistent recombination energy PAIR on the electron
        fluid: per recombination event credit the binding energy ``I_ion``
        (paid at ionization via ``I_ion*S_ion`` and never returned by the
        standard booking) and charge the full ADAS ``prb1`` radiated power,
        adding ``I_ion*S_rec - P_PRB`` to ``Ee`` on top of the ordinary
        recombination terms. Both halves scale with ``b_rec_rad``, so the
        credit tracks the sink the particle equation actually applies; the
        ``3/2 Te S_rec`` capture-kinetic-energy loss stays booked where it is
        and cancels in the net. The sign of the net follows the conditions --
        heating where the radiated energy per event is below ``I_ion``, an
        extra sink where it is above. ``False`` returns a zero source without
        evaluating the term. Requires ``atomic_rate_model = "adas"`` (the
        janev path has no PRB booking) and raises otherwise; the pair is the
        consistent unit, so it also raises when combined with the
        ``icool_recomb`` flag, which charges PRB on its own. Lookups clamp at
        the ADF11 grid edges.
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
    heat_flux_limiter_f:
        Free-streaming fraction ``f`` setting the electron heat-flux
        saturation ceiling ``q_sat = f n Te v_the`` (``Te`` in erg,
        ``v_the = sqrt(Te/m_e)``), against which the classical Spitzer-Harm
        parallel flux ``q_SH`` is capped. The limiter scales the conductivity
        per cell, so the operator stays a conservative flux divergence, and
        is frozen at the incoming ``Te`` like ``kappa`` itself. A smaller
        ``f`` lowers the ceiling and suppresses more. Read only when the
        ``electron_heat_flux_limit`` flag is on, where it must be ``> 0``;
        raises at construction otherwise.
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
        single free-streaming fraction cannot express. Read only when the
        ``electron_heat_flux_limit`` flag is on, where it must be ``> 0``;
        raises at construction otherwise.
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
    sigma_in_model:
        Source of the ion-neutral momentum-transfer rate, which feeds the
        drag, the slip closure's entrainment, thermalization, the drag
        timestep bound, and the presheath depth.

        ``"phelps"`` is the only accepted value: the definitive
        momentum-transfer rate, the same Phelps He+/He isotropic +
        backscatter cross section the ``ion_neutral_moment_closure`` operator
        uses, ``nu_in = nn * (k_b + 1/2 k_iso)(T_eff)`` with
        ``T_eff = (Ti + Tn)/2`` at the single cold-gas ``Tn`` (300 K). This
        ties the presheath sampling to the same collision physics as the
        drag. He-only, gated at construction; the legacy ``"constant"`` and
        ``"cx_derived"`` arms, which were the solver's only non-helium path,
        were removed at D3 (2026-08-21) and raise.
    alpha_isat:
        Ion-saturation/surface-loss coefficient.
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
        # ion_neutral_moment_closure uses, and since D3 the only accepted
        # value: the solver is helium-only.
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
        # floor and a permanent electron_cooling timestep-bound collapse.
        # Construction refuses icool_recomb TOGETHER WITH
        # adas_low_te_extension; see the retired recipe in the module note
        # above.
        "recombination_energy_return": False,
        # --- Electron heat-flux limiter (read only when the
        # electron_heat_flux_limit flag is on, which is a shipped default) ---
        # Free-streaming fraction f in q_sat = f*n*Te*v_the -- the ceiling
        # (Cowie & McKee, ApJ 211 (1977) 135, eq. 7) that the harmonic cap
        # saturates toward. Convention: v_the = sqrt(Te/m_e), as in Malone
        # 1975 / Fundamenski 2005; a coefficient quoted in the Cowie & McKee
        # convention needs *sqrt(2/pi) = 0.7979 to be read as an f here.
        "heat_flux_limiter_f": 0.45,
        # Non-local Knudsen exponent p for that limiter (read only when
        # electron_heat_flux_limit is on). lambda = 1/(1+Kn^p)
        # with Kn = q_SH/q_sat. p=1.0 (default) is the harmonic form of
        # Malone 1975 / Fundamenski 2005 eq. 10a. p>1 suppresses the
        # steep-gradient (high-Kn, non-local)
        # startup flux much harder while leaving the shallow-gradient established
        # column near-Spitzer -- the startup-front pre-heating vs established-
        # column trade a single free-streaming factor cannot separate.
        "heat_flux_limiter_exponent": 1.0,
        # Radial closure for the evolved neutral wind (needs the
        # neutral_momentum flag; the deferred ladder).
        "neutral_momentum_radial": "uniform",
        # --- DEPRECATED: legacy ion-neutral drag (superseded by the Phelps
        # ion_neutral_moment_closure). Warn on non-default/active use;
        # retained runnable for reproducibility. ---
        "b_ion_neutral_drag": 1.0,
        "ion_neutral_drag_model": "constant",
        "b_slip_entrainment": 1.0,
        "b_ion_neutral_thermalization": None,
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
        Anode-mesh solid fraction (opacity) [dimensionless]: the share of the
        anode face its wires occupy, so ``1 - eta`` transmits. ``eta`` sets the
        anode's Bohm ion collection area (``2*eta*I_i``, both mesh faces) and
        the share of the gap-surviving thermionic beam the mesh intercepts;
        ``1 - eta`` is the face's neutral transparency and the beam's geometric
        survival. Must lie in ``[0, 1]``. Value and class:
        ``config_defaults_provenance.md``.
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
    circuit_picard_tol_rel:
        Relative convergence tolerance on the loop current for the
        fluid<->circuit Picard iteration. A step is re-run when
        ``|I_new - I_frozen| > tol * max(|I_new|, 1)``, until the current a
        step PRODUCES matches the frozen one it was RUN at -- so the fluid,
        the surface temperature and the circuit end the step sharing one
        self-consistent ``I_loop`` instead of the fluid seeing a lagged
        current. Re-runs restore an exact snapshot first, so discarded
        iterations mutate no accepted-step state. Read only when the
        ``coupled_circuit_picard`` flag is on, where it must be ``> 0``;
        raises at construction otherwise.
    circuit_picard_max_iter:
        Cap on those re-runs per step; ``1`` permits a single attempt and so
        reproduces the uniterated behaviour. Only driven phases iterate --
        floating phases break after the first attempt regardless. Read only
        when the ``coupled_circuit_picard`` flag is on, where it must be
        ``>= 1``; raises at construction otherwise.
    cathode_warming_model:
        Slow evolution of the emitter surface temperature within a shot.
        ``"none"`` holds ``T_s`` constant, so the
        emission ceiling -- and with it the discharge current -- saturates
        on the circuit timescale (~1-2 ms), where the measured current rises
        for ~15-20 ms. ``"power_balance"`` (default, M1b) uses
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
        stable for any ``C_th``), floored at the 300 K chamber-wall
        temperature the surface radiates against, accepted steps only.
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
        Radial structure of the thermionic emitter. ``"uniform"``
        (default) is a single-temperature disc, whose emission ceiling is a
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
    cathode_surface_model:
        Whether the cathode work function evolves with the coverage of the
        contaminant layer. ``"none"`` holds ``phi_wf`` static for the shot.
        ``"ads_des"`` carries a coverage ``theta`` in ``[0, 1]``, initialized
        fully covered (``theta = 1``, so the shot starts at ``phi_wf``
        exactly), evolving as

            dtheta/dt = -sigma_cl Gamma_i theta

        (ion-stimulated desorption, the only coverage channel, so ``theta``
        is monotonically non-increasing through a shot), and
        substitutes ``phi_eff = phi_clean + (phi_wf - phi_clean)*theta``
        wherever the work function is read -- Richardson emission, Schottky
        lowering, emission cooling and the gaussian profile's Richardson
        inversion all take the one substituted value, never a mix of
        ``phi_eff`` and ``phi_wf``. ``theta`` advances by a backward-Euler
        update on accepted steps only. Any other value raises at
        construction. Under ``"ads_des"``, ``phi_wf`` keeps its meaning as
        the fully-covered shot-start work function.
    cathode_phiwf_clean_eV:
        Work function [eV] of the fully cleaned surface -- the ``theta -> 0``
        floor of ``phi_eff``, i.e. the per-shot-accessible depth of the
        removable layer rather than a literature clean-surface value.
        REQUIRED under ``cathode_surface_model = "ads_des"`` and must be
        strictly below ``phi_wf``; raises at construction when missing or not
        below it. Inert under ``"none"``.
    cathode_cleaning_sigma_cm2:
        Ion-stimulated desorption cross section [cm^2] in the coverage loss
        term ``sigma_cl*Gamma_i``, where the ion flux density onto the
        cathode is ``Gamma_i = I_i/(e*pi*R_cath^2)`` taken from the
        accepted-state sheath solve. Must be non-negative; raises at
        construction otherwise. ``0`` removes the ion-stimulated channel.
        Inert unless ``cathode_surface_model = "ads_des"``.
    cathode_cleaning_E_th_eV:
        Threshold energy [eV] for that desorption cross section. When set,
        ``cathode_cleaning_sigma_cm2`` is scaled by the near-threshold
        Bohdansky factor ``(1 - (E_th/E)^(2/3))*(1 - E_th/E)^2`` at the mean
        deposited energy per ion ``E = P_cathode_i/I_i`` from the same
        accepted-state solve, and the channel is switched off entirely for
        ``E <= E_th``. ``None`` leaves the cross section energy-independent
        (the pure fluence limit). Inert unless
        ``cathode_surface_model = "ads_des"``.
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
        well-posed version of the inductive kick. It bounds a REGIME of the
        solve rather than describing a drop the device sustains, so the
        ceiling value is reported as ``phi_c`` for as long as that regime
        holds, and every consumer keyed to ``phi_c`` (notably the tail birth
        energy under ``heating_anomalous_tail_energy_keying="phi_c"``) sees it.
        Under the ``cathode_circuit_voltage_bound`` flag this cap is not the
        whole ceiling: the solve is run against ``min`` of it and the circuit
        member selected by ``cathode_circuit_bound_object``, and the
        ``bound_active`` diagnostic says which of the two the solve sat on.
        This cap alone is a domain guard on the atomic data and holds in every
        regime; the restrictions on the composition belong to the circuit
        member and are stated at that key and at the flag.
    cathode_circuit_bound_object:
        Which quantity the circuit-available voltage bounds under the
        ``cathode_circuit_voltage_bound`` flag. Read only while that flag is
        on; inert otherwise, and construction raises on an unknown value when
        it is on.

        ``"device_voltage"`` (default) makes the object the DEVICE voltage
        ``V_b = phi_c - phi_a + V_p``, which is the quantity the loop equation
        contains. The circuit member of the composed ceiling is then the net
        cathode drop at which ``V_b`` reaches the available voltage, located
        by a bracketed solve on the same monotone device relation the current
        root uses, with ``phi_a`` and ``V_p`` evaluated by the identical
        expressions that assemble the returned result. Because the anode fall
        SUBTRACTS, ``phi_c`` may legitimately exceed the available voltage
        while ``V_b`` does not, and this object permits exactly that.

        ``"phi_c"`` makes the object the net cathode drop itself: the circuit
        member IS the available voltage. This is the R1 composition, bit for
        bit, retained as an A/B arm. It coincides with ``"device_voltage"``
        only where ``phi_a`` and ``V_p`` are negligible — the near-vacuum
        build leg — and mis-clamps a correct plateau solve elsewhere,
        silently except for the ``bound_active`` census.

        NEITHER object changes what the bound leaves alone: the inductor's
        back-EMF is not counted as supply, so on a FALLING leg the physical
        ``V_b`` exceeds the available voltage and the bound engages -- but
        the loop current is not held there, because the circuit integrates
        the sheath's unbounded demand rather than the clamped ``V_b``. Only
        the reported/beam-facing objects are clamped.
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
    cathode_lnL_model:
        Which Coulomb logarithm the parallel Spitzer conductivity
        ``sigma_par`` is built from, in BOTH sheath solvers and in every
        sim1d-side consumer of ``sigma_par`` (the ``"resolved_gap"`` R_p
        integral and the ohmic gap deposition weights).

        ``"nrl_ei"`` (default) evaluates the electron-ion Coulomb logarithm at
        the local ``(Te, n)`` and floors it at ``LN_LAMBDA_MIN``, the same
        convention the conduction and electron-ion exchange terms use, giving
        ``sigma_par = (1.96/(1.03e-2 lnLambda)) Te^1.5`` [Ohm^-1 cm^-1].

        ``"fixed_14p6"`` restores the frozen-coefficient form
        ``sigma_par = 14.6 Te^1.5``, i.e. ``"nrl_ei"`` evaluated at
        ``lnLambda = 13.03`` and held there regardless of state. It is an
        ATTRIBUTION-ONLY comparison arm: it exists so a result can be split
        between the lnLambda correction and everything else, and it is not a
        physical alternative. Any other value raises at construction.
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
        ``"beer_lambert"`` (historical): single-event absorption
        over the mixed Coulomb/inelastic profile (``l_b_profile`` +
        ``beam_absorption_weights``). ``"csda"`` (default): the deterministic
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
        under ``"beer_lambert"``). A declared closure BRACKET of three arms; a
        result states which one produced it.
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
        REGISTERED BRACKET rather than a tuned number — see the provenance
        note — and results under this closure are quoted at the bracket
        endpoints, not at the default alone.
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
        the terminal residual keeps the primary's direction.
        ``"terminal_nonlocal"``: the MIDDLE point -- the terminal residual
        walks exactly as under ``"nonlocal"`` (same machinery, same
        thermalization rule, same end ledger) while every ALONG-RAY product,
        the secondaries included, is banked in its birth cell exactly as under
        ``"local"``. It differs from ``"nonlocal"`` in two further bookings:
        the transmitted primary keeps its own term instead of joining the end
        ledger (so under this value the ledger holds the walked terminal
        escape alone), and the escaping terminal electrons' CURRENT is added
        to the vessel node's wall electron channel when ``regime_vessel_node``
        is also armed -- charge to the node, energy to the ledger, neither
        counted twice. Not available on the compiled kernel, which takes
        product transport as a single boolean covering both populations and
        so cannot express one walking without the other; selecting it takes
        the Python march. Motivation: at
        breakdown both products sit BELOW every He inelastic threshold and
        Coulomb-couple at ~1 eV per machine pass (n_e ~ 1e10), i.e. they are
        near-collisionless along B exactly where ``"local"`` assumes perfect
        confinement. Under ``"nonlocal"`` the end ledger also books the
        transmitted PRIMARY's ``Gamma_t*E_t``, closing a standing hole
        (computed since B1, never
        banked). Parameter-free; no pitch-angle diffusion and no elastic
        e-He channel (~5 meV/collision) — stated limitations. ENERGY-ONLY:
        ionization events and the particle rows are
        identical under all three values, and so are the circuit currents
        except for the one charge channel ``"terminal_nonlocal"`` adds at an
        armed vessel node. The three settings are a bracket, not a
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
        ``"plateau_multigroup"``: the quasilinear plateau is not one energy,
        and this value carries the SPECTRUM instead of a line. In the flux
        frame the relaxed distribution is flat over the resonant band, so
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
        arithmetic-midpoint energy and walked by exactly the machinery
        ``"tail_walk"`` uses. Nothing is fitted and no new parameter appears:
        the shares, edges and weights all follow from the flat plateau. The
        two single-line arms are its two heirs taken one at a time, which is
        why the ``f`` dial, the fixed rung and the keying selector are all
        INERT under it and are REFUSED at construction rather than ignored.
        The range law is unchanged (classical Coulomb). ENERGY-ONLY in the
        same sense as ``"tail_walk"``, and it inherits that value's tail
        ionization, cathode-boundary and end-ledger conventions unchanged.
        Not supported under ``coverage_closure`` (the two-stream march shares
        one withholding bank and its reservoir carries the density floor, so
        an edge solved there would be a floor artifact) -- that raises.
    heating_anomalous_disposal:
        How each cell's extracted anomalous power is SPLIT between the local
        bulk and the walked tail (inert under ``"beer_lambert"``, and requires
        an active anomalous channel; selecting the non-default value without
        either raises). ``"local"`` (default, bit-exact): no split — the
        disposition is whatever ``heating_anomalous_transport`` says, which is
        all-or-nothing. ``"landau_branched"``: the wave the beam drives loses
        its energy through two channels at once, Landau damping on the resonant
        electrons (which makes a nonlocal tail) and collisional damping of the
        wave (which makes local bulk heat), and their ratio is a COMPUTED
        property of each cell rather than a choice —
        ``f_Landau = gamma_L / (gamma_L + nu_en/2)`` with ``nu_en = nn*K_m(Te)``
        on the boxed He e-n momentum-transfer coefficient and ``gamma_L`` the
        Maxwellian Landau rate at the beam-resonant phase velocity
        (``funcs._beam_deposition.landau_branching_fraction``, which carries the
        formula and its ``v_phi/v_te`` validity caveat). That share of each
        cell's power is walked exactly as ``"tail_walk"`` walks all of it —
        same birth energy, launch, Coulomb machinery, cathode and collector
        conventions and tail end ledger — and ``1 - f_Landau`` is banked
        locally exactly as ``"local"`` banks all of it. NO new physical
        constant: the branching is computed from boxed inputs and the birth
        energy is the existing ``phi_c`` keying.
        The ``heating_anomalous_transport`` values ``"tail_walk"`` and
        ``"local"`` are the ``f_Landau ≡ 1`` and ``f_Landau ≡ 0`` corners of
        this one, so selecting ``"landau_branched"`` together with any
        non-``"local"`` transport RAISES — both name a disposition for the
        same bank. ``"landau_branched"`` requires
        ``heating_anomalous_tail_energy_keying="phi_c"`` (the registered birth
        energy is the live cathode drop, and the fixed rung is an assumed
        constant this closure does not carry) with
        ``heating_anomalous_tail_phi_c_fraction`` stated explicitly, and it
        RAISES under ``coverage_closure``: the two-stream march shares one
        withholding bank between the channel and reservoir arms and the
        reservoir carries the density FLOOR, so a branching there would be an
        artifact of the floor convention (see
        ``funcs._beam_deposition.deposit_beam_two_stream``). ENERGY-ONLY,
        exactly like ``heating_anomalous_transport``.
    heating_anomalous_tail_energy_eV:
        QL plateau energy ``E_tail`` [eV] the tail electrons are launched at.
        **Read ONLY when the QL tail is WALKED (``heating_anomalous_transport
        ="tail_walk"``; ``heating_anomalous_disposal="landau_branched"`` walks
        only the Landau share) WITH
        ``heating_anomalous_tail_energy_keying="fixed"``** -- inert otherwise,
        and supplying a value other than the shipped one under
        ``"phi_c"`` keying or under
        ``heating_anomalous_transport="plateau_multigroup"`` (where the birth
        energies are the derived group midpoints) raises rather than being
        silently ignored. Must be
        finite and > 0. It sets the walkers' Coulomb
        range and therefore how far the QL power travels before thermalizing;
        the equivalent tail flux is ``P_QL / E_tail``, so the power carried is
        independent of it. The plateau energy is a kinetic quantity a fluid
        model cannot pin, so this is an ASSUMED value and a run that uses it
        must report a bracket rather than a single number. Arms and values:
        ``config_defaults_provenance.md``.
    heating_anomalous_tail_energy_keying:
        How the tail birth energy ``E_tail`` is set. **Read ONLY when the QL
        tail is WALKED** (``heating_anomalous_transport="tail_walk"`` or
        ``heating_anomalous_disposal="landau_branched"``) -- inert otherwise,
        and the branched disposal accepts ``"phi_c"`` alone.
        ``heating_anomalous_transport="plateau_multigroup"`` keys the
        spectrum's TOP to the live ``e*phi_c`` and its BOTTOM to the solved
        plateau edge, so there is no rung to select: a non-default keying
        raises there rather than being ignored.
        ``"phi_c"`` (default): ``E_tail = f * e*phi_c(t)``, keyed to the LIVE
        cathode accelerating drop of the ray that drove the QL power, with
        ``f`` from ``heating_anomalous_tail_phi_c_fraction``. ``"fixed"``: the
        constant ``heating_anomalous_tail_energy_eV``, the WP-E/K6 behaviour,
        bit-exact when selected. The plateau is filled by a beam whose energy
        IS ``phi_c``, so a fixed rung makes the walkers' reflection margin at
        the sheath an accident of how far the drive happens to sit from that
        rung; keying removes that dependence. Under ``"phi_c"`` with
        ``heating_anomalous_tail_ionization="on"`` the two depth-1 truncation
        bars are evaluated on the LIVE ``E_tail`` at every solve, so which
        band the walkers march in is a per-frame property of the DRIVE rather
        than of the configuration: a cold foot sits below the lower bar and
        the march reverts to the energy-only walk there, and ``f = 1.0``
        crosses the upper bar at production drive and marches under the
        disclosed depth-1 understatement. Both are recorded per frame in the
        tail diagnostics (``beam_tail_sub_threshold_power_W`` /
        ``beam_tail_sub_threshold_fraction`` /
        ``beam_tail_above_bar_power_W``), so neither regime is silent.
    heating_anomalous_tail_phi_c_fraction:
        The fraction ``f`` in ``E_tail = f * e*phi_c(t)``. **Read ONLY under
        ``heating_anomalous_tail_energy_keying="phi_c"``**; must be ``None``
        under ``"fixed"``, where supplying one would silently do nothing, and
        must be ``None`` under
        ``heating_anomalous_transport="plateau_multigroup"``, whose derived
        spectrum spans the whole band and carries BOTH ends of this bracket
        at once.
        ``None`` (default) selects the shipped arm ``f = 0.25``, except under
        ``heating_anomalous_disposal="landau_branched"``, which requires the
        arm to be stated and raises on ``None``. The only
        accepted values are the DECLARED BRACKET ``{0.25, 0.5, 1.0}`` -- any
        other value raises, because ``f`` is a bracket the campaign reports
        across and never a fitted number. Values and class:
        ``config_defaults_provenance.md``.
    heating_anomalous_tail_cathode_boundary:
        What the CATHODE end does to a tail walker that reaches it. **Read
        ONLY when the QL tail is WALKED** (``heating_anomalous_transport=
        "tail_walk"`` or ``"plateau_multigroup"``, or
        ``heating_anomalous_disposal="landau_branched"``)
        -- inert otherwise. ``"reflect"`` (default): a walker arriving at the cathode
        face of the plasma-active window with energy below ``e*phi_c(t)`` is
        turned around at the same energy and keeps walking; only a walker at or
        above that drop escapes. ``"escape"``: the WP-E/K6 free-escape
        convention, in which every walker reaching the face leaves and its
        energy is booked to the tail end ledger -- selectable, and bit-exact
        when selected. The cathode sits at an accelerating drop of a few
        hundred volts through drive, above every plateau energy the bracket
        carries, so free escape there deletes tail power the sheath in fact
        returns to the column. Under ``"reflect"`` the cathode-face row of the
        tail end ledger (``source_beam_end_loss_tail_low_W``) is therefore
        EXACTLY ZERO for the whole of a ``"phi_c"``-keyed run -- birth energy
        ``f*e*phi_c`` with ``f <= 1`` against a threshold of ``e*phi_c``, and
        walkers only lose energy -- but NOT for a ``"fixed"``-keyed one, where
        the rung is decoupled from the drive and any frame with
        ``e*phi_c`` below the rung lets walkers out through that face. A reader
        deriving the escaping fraction must sum the whole end ledger rather
        than name the far-end row alone. Selecting ``"reflect"`` also makes the
        plasma-active window bound the ENERGY-ONLY walk (which otherwise runs
        the whole grid): the reflecting face has to be a face the walk stops
        at. Reflection is total by construction, with no partial-reflection
        coefficient -- the radial fraction of the returning tail that misses
        the emitting disc is UNSIZED in 1D and is a documented limitation, not
        a knob. Requires a single cathode: with ``TwinCathode`` both window
        faces reflect, trapping the walkers, and that raises.
    heating_anomalous_tail_ionization:
        Whether the QL tail walkers may IONIZE and EXCITE the column gas they
        pass through. **Read ONLY when the QL tail is WALKED**
        (``heating_anomalous_transport="tail_walk"`` or
        ``"plateau_multigroup"``, or
        ``heating_anomalous_disposal="landau_branched"``) -- inert otherwise,
        and selecting ``"on"`` without one of them raises. Under
        ``"plateau_multigroup"`` the two depth-1 bars below are evaluated PER
        GROUP, on each group's own midpoint energy, so the band exposures
        become power-weighted shares of the launched streaming bank rather
        than all-or-nothing. ``"off"`` (default, bit-exact):
        the walk is energy-only, the walkers Coulomb-slow and nothing else, and
        every particle row is what it would be under ``"local"``. ``"on"``: each
        tail population is marched on the CSDA module's own integration, so it
        attenuates on the local COLUMN neutral density (under
        ``neutral_two_zone`` that is the column channel ``nn``, the only gas on
        the walker's field line) with the He ionization and excitation cross
        sections evaluated at the walker's CURRENT energy, simultaneously with
        its Coulomb slowing. Each ionization event births one ion/electron pair
        at the event cell on the same convention the beam's own births use,
        invests ``I_ion``, banks the mean secondary ``<W_sec>`` as local
        electron heat and each excitation threshold as radiation; the walker
        continues on the reduced energy, and what still reaches a domain end
        goes to the same tail end ledger the energy-only walk uses. So
        ``"on"`` adds a PARTICLE channel where the rest of WP-E is
        energy-only, and both settings share one end convention -- the flag
        moves one thing.
        Motivation: the omitted channel is negligible in the main discharge
        (Coulomb blocking, thin target) but brushes materiality in the
        breakdown foot, where it feeds back on the very density that
        suppresses it.
        BAND TREATMENT (K7b): the two depth-1 bars are still computed from the
        thresholds themselves, but each selects a treatment for the ray rather
        than refusing it, so a ``phi_c``-keyed arm can run from cold. At or
        below the lowest inelastic threshold the ionizing march REVERTS to the
        energy-only walk -- exact, since no inelastic channel is open there,
        and bit-identical to what ``"off"`` would do for that frame. Above the
        ``<W_sec>(E_tail)`` crossing the march RUNS with the depth-1
        truncation, which there understates the tail's ionization by a
        MEASURED <= 2.0%. Neither regime is silent: the tail diagnostics carry
        the power marched in each. The one refusal left is a tail energy past
        the tabulated He EII cross section, where the lookup would clamp to its
        last node and the walk would attenuate on an extrapolated cross
        section. That edge is checked against the table itself, and it is
        checked in TWO places because they see different values: construction
        tests ``heating_anomalous_tail_energy_eV``, which under ``"phi_c"``
        keying is the inert fixed rung, and the deposition module tests the
        LIVE ``E_tail = f*phi_c(t)`` at every cathode solve. The runtime one is
        the binding check under keying, and it is REACHABLE: with ``f = 1.0``
        and ``phi_c`` at ``cathode_phi_c_cap_V`` (the capability-limited
        ceiling, a numerical bound on the sheath solve rather than a drop the
        device sustains) ``E_tail`` lands on the edge to the last bit. The edge
        is therefore INCLUSIVE within a relative tolerance of 1e-12
        (``_beam_deposition.HE_EII_EDGE_REL_TOL``): within it the lookup is
        clamped to the table's last node, which AT the edge is that node's own
        value and not an extrapolation, and beyond it the march raises and the
        message reports the measured relative excess.
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
    cathode_neutral_jet:
        Gives the neutral flux recycled at an absorbing CATHODE face directed
        axial momentum instead of rebirthing it at rest: a fraction
        ``cathode_jet_R_N`` backscatters and the implanted remainder desorbs
        as a directed effusive flux off the hot surface. The momentum rides
        in the SAME term that rebirths the particles, so the two are
        consistent by construction, and the surface absorbs the difference
        between the incoming sonic momentum and the re-emitted jet momentum.
        Collector faces stay momentum-free. ``False`` rebirths at rest.
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
    cathode_sample_smoothing:
        Exponential-moving-average smoothing of the ``(n, Te)`` the sheath
        solve samples, covering the cathode sample cell and the two cells
        flanking the first anode face -- every cell the solve reads. The EMA
        is seeded from the initial state and advances on accepted steps only,
        so dt-retries never move it, and the sampled ``Ee`` is rebuilt from
        the smoothed pair. ``"presheath"`` derives the time constant per cell
        as the ion transit across it, ``tau = l_cell/c_s(Te_ema)``, so it is
        a local physical timescale rather than a configured number; a float
        is a fixed ``tau`` [s] and must be positive; ``None`` disables the
        smoothing bit-exactly, passing the raw state through. Any other
        string, or a non-positive float, raises at construction.

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
        "L_cath": 53.25,
        "R_cath": 18.415,
        # --- ACTIVE: beam deposition (CSDA production stack; b_beam_excitation
        # + beam_excitation_model are INERT under csda -- the module uses the
        # measured manifold, knob-free -- and matter only for the beer_lambert
        # A/B arm) ---
        "beam_deposition_model": "csda",
        "beam_coulomb_model": "fast_electron",
        "beam_anomalous_model": "quasilinear",
        # ql_relaxation's plateau-formation bracket constant. INERT unless that
        # closure is selected; the shipped value is the bracket's geometric
        # centre and every headline under the closure is quoted at 10 and 100
        # as well.
        "ql_relaxation_coeff": 30.0,
        # Non-local product transport: DEFAULT OFF (bit-exact).
        "beam_product_transport": "local",
        # QL heating locality: DEFAULT OFF (bit-exact). The tail energy is
        # inert under "local".
        "heating_anomalous_transport": "local",
        # Branched disposal of the extracted QL power: DEFAULT OFF
        # (bit-exact). Inert under "local"; see the docstring above.
        "heating_anomalous_disposal": "local",
        "heating_anomalous_tail_energy_eV": 75.0,
        "heating_anomalous_tail_ionization": "off",
        # K7 sheath-aware tail closure. Both keys are inert unless the walk is
        # engaged, and when it is they DEFAULT TO THE CORRECTED closure; the
        # WP-E/K6 arms stay reachable, and bit-exact, by naming "fixed" and
        # "escape" explicitly.
        "heating_anomalous_tail_energy_keying": "phi_c",
        "heating_anomalous_tail_phi_c_fraction": None,
        "heating_anomalous_tail_cathode_boundary": "reflect",
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
        # --- ACTIVE: uniform emission profile ---
        "cathode_emission_profile": "uniform",
        "cathode_Ts_fwhm_cm": 28.0,
        "cathode_emission_annuli": 10,
        "cathode_Rp_model": "sample",
        # Coulomb logarithm behind sigma_par. "nrl_ei" reads it at the local
        # (Te, n); "fixed_14p6" freezes it at the historical 13.03 and is an
        # attribution-only comparison arm.
        "cathode_lnL_model": "nrl_ei",
        "cathode_solver_model": "current_driven",
        "cathode_phi_c_cap_V": 1000.0,
        "cathode_circuit_bound_object": "device_voltage",
        # Surface-state coverage model:
        # "ads_des" evolves contaminant coverage theta with
        # dtheta/dt = -sigma Gamma_i theta
        # and substitutes phi_eff = phi_clean + (phi_wf - phi_clean)*theta
        # everywhere phi_wf is read (emission, Schottky, cooling, gaussian
        # inversion -- every consumer reads the one substituted
        # value, never a mix of phi_eff and phi_wf). phi_wf keeps its
        # meaning as the contaminated SHOT-START value; the clean floor is
        # the per-shot-accessible depth of the re-adsorbed layer, not the
        # literature clean-LaB6 value. Ion-stimulated desorption is the only
        # coverage-loss channel: the coverage is monotonically non-increasing
        # through a shot.
        # --- ACTIVE: ads_des surface state (in-shot ion-stimulated cleaning) ---
        "cathode_surface_model": "ads_des",
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
        # (cathode = He->LaB6, anode = He->Mo; see
        # config_defaults_provenance.md for the values and their brackets).
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
        # fraction (power_balance receives (1 - R_E) * P_cathode_i); off, the
        # jet is momentum-only and the surface keeps that power. Requires
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
    neutral_wall_partition_sigma_hehe_cm2:
        He--He MOMENTUM-TRANSFER cross section ``sigma_mt`` [cm^2] (the
        ``Omega^(1,1)``-derived moment, NOT a total elastic one) setting the
        neutral-neutral mean free path ``1/(nn_a sigma)`` that the
        ``neutral_wall_momentum_partition`` flag uses to weight the two-zone
        wall branch. The partition attenuates DIRECTED MOMENTUM, so the
        forward-peaked small-angle encounters a total cross section counts at
        full weight barely remove any, and a total would over-suppress the
        wall branch. REQUIRED by that flag and read by nothing else: it has no
        default, so arming the flag without it raises at construction, and
        supplying it without the flag raises as well. Must be finite and
        strictly positive.
    neutral_exchange_coeff_cm3_s:
        Constant neutral exchange coefficient for the constant model [cm^3/s].
    neutral_clausing_scale:
        Scale factor applied to the Knudsen tube and orifice conductances.

    Values and their provenance: ``config_defaults_provenance.md``.
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
        # REQUIRED by the neutral_wall_momentum_partition flag (which ships
        # off) and forbidden without it. None is the "not supplied" sentinel,
        # not a physical value -- there is no defaulted He-He cross section.
        "neutral_wall_partition_sigma_hehe_cm2": None,
        # Only the "constant" neutral_exchange_model reads this (the default is
        # "knudsen"):
        "neutral_exchange_coeff_cm3_s": 1.0e5,
        # LIVE on the default "knudsen" path (it scales every tube and orifice
        # conductance); inert only because the default multiplier is 1.0:
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
        ``dt_growth`` before the accelerated re-approach engages. Zero (the
        default) disables the mechanism entirely and the ramp is uniformly
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
        Consulted ONLY when ``dt_growth_recovery_patience`` > 0. Must be
        greater than ``dt_growth_factor``; anything else raises at
        construction.

        The asymmetry between engaging and releasing is the hysteresis:
        engaging takes ``dt_growth_recovery_patience`` consecutive
        growth-capped steps, releasing takes ONE step capped by anything else
        (a physics bound, an output cadence, or a retry after a rejection).
        Re-approach is therefore fast while nothing is binding and instantly
        conservative again the moment something is. It does not weaken any
        bound: every step is still the minimum over all candidates, and this
        only widens the ceiling the ramp itself imposes.
    max_density_step_fraction:
        Optional accepted-step density fractional-change guard. Zero disables it.
    max_neutral_step_fraction:
        Optional accepted-step neutral fractional-change guard. Zero disables it.
    max_energy_step_fraction:
        Optional accepted-step thermal-energy fractional-change guard. Zero
        disables it.
    circuit_dt_fraction:
        Fraction of the current-driven loop's LOCAL relaxation time
        ``tau_circuit = L_parasitic_H / (R_comp + R_mesh_ohm + dV_dis/dI)``
        allowed per accepted step. Read only while
        ``cathode_circuit_voltage_bound`` is armed and a live loop exists;
        the candidate is withdrawn to ``inf`` otherwise, so this key cannot
        move an unarmed run. Bounds the sheath's capability wall, whose
        device slope reaches ~2 kOhm (``tau_circuit`` ~ 4 ns) while ``L/R``
        is 1.12 ms -- the wall, not the loop's bulk time constant, is the
        stiff feature. Accuracy, not stability: the TR-BDF2 advance is
        L-stable. See ``cathode.circuit_relaxation_timestep``.

    Values and their provenance: ``config_defaults_provenance.md``.
    """
    return {
        "cfl": 0.4,
        "density_dt_fraction": 0.25,
        "neutral_dt_fraction": 0.25,
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
        # Default-off: patience 0 skips the branch entirely, so the ramp is
        # uniformly dt_growth_factor and a run is bit-exact with one predating
        # these keys. NO default flip -- that decision is not the code's.
        "dt_growth_recovery_patience": 0,
        "dt_growth_recovery_factor": 4.0,
        "max_density_step_fraction": 0.0,
        "max_neutral_step_fraction": 0.0,
        "max_energy_step_fraction": 0.0,
        "circuit_dt_fraction": 0.25,
    }


def coverage_closure_defaults():
    """Return clumpy-plasma coverage-closure defaults (v2, z-resolved).

    Every key here is read ONLY under the ``coverage_closure`` flag and is
    inert otherwise. The closure carries a PER-CELL coverage fraction
    ``f_cov(z, t) in (0, 1]``: the plasma occupies that fraction of the column
    cross-section at each axial position, so channel-local densities are the
    mean divided by ``f_cov(z)`` and the remaining ``1 - f_cov(z)`` is a
    neutral reservoir. See ``MODEL.md`` for the term placement.

    coverage_growth_rate_per_s:
        Column-mean logistic growth rate ``r0`` [s^-1] of the coverage field,
        which evolves as
        ``df_cov(z)/dt = r0 * w(z, t) * f_cov(z) * (1 - f_cov(z))`` from its
        initial condition at the plasma-phase time origin. ``w(z, t)`` is the
        local beam-ionization rate normalized to its own volume-weighted
        column mean, so ``<w> = 1`` by construction and ``r0`` keeps its
        meaning as the mean rate -- it introduces no constant of its own and
        is the calibration target. Must be finite and ``>= 0``; ``0`` freezes
        the field at its initial condition. The law takes feedback from the
        state (deposition depends on the coverage it drives), so the solver
        CO-INTEGRATES it on the step's stage structure rather than evaluating
        a closed form.

        SHARED CLOCK: this key is ALSO the growth rate of the cathode
        emitting-area fraction under the ``cathode_emitting_area`` flag (see
        ``emitting_area_defaults``), which reads it rather than carrying a
        rate of its own. The two closures describe one percolation clock seen
        from two surfaces, so the constant is fitted once and has one owner;
        the key is therefore live -- and non-default values are accepted --
        whenever EITHER flag is armed.
    coverage_backfill_time_s:
        Relaxation time ``tau_backfill`` [s] over which the uncovered
        reservoir refills the covered column's neutral density toward the
        cell mean. Must be finite and ``> 0``. The exchange moves no particles
        between cells and none into or out of the conserved mean neutral
        field, so total particle inventory is unaffected by construction.
    coverage_initial_fraction:
        UNIFORM initial coverage fraction ``f_cov0`` at the plasma-phase time
        origin, applied to every cell. ``None`` (the default) is the only
        value permitted with the flag off; with the flag on it must lie in
        ``(0, 1]``. This is an initial condition, not a physical constant.
    coverage_initial_profile:
        PER-CELL initial coverage ``f_cov0(z)``: a sequence of length ``nx``
        (the grid's cell count) with every entry finite and in ``(0, 1]``.
        ``None`` (the default) is the only value permitted with the flag off.
        With the flag on, EXACTLY ONE of this and
        ``coverage_initial_fraction`` must be given -- they are two spellings
        of the same initial condition and neither modifies the other, so
        supplying both is a construction-time ``ValueError`` rather than a
        rule the reader has to remember. This is the per-realization ensemble
        hook: the solver contains no randomness, and an ensemble is generated
        by building profiles externally and passing them here, one run per
        realization.

    Values and their provenance: ``config_defaults_provenance.md``.
    """
    return {
        "coverage_growth_rate_per_s": 1390.0,
        "coverage_backfill_time_s": 3.0e-5,
        "coverage_initial_fraction": None,
        "coverage_initial_profile": None,
    }


def emitting_area_defaults():
    """Return cathode emitting-area percolation defaults (ea1).

    Every key here is read ONLY under the ``cathode_emitting_area`` flag and is
    inert otherwise. The closure carries ONE scalar ``f_em(t) in (0, 1]``: the
    fraction of the cathode's emitting face that is actually lit. The annular
    emission tuples are scaled by it at the single device-config seam --
    ``area_k -> f_em*area_k`` (which throttles each annulus's Richardson
    emission) and ``frac_k -> f_em*frac_k`` (which throttles the share of the
    Bohm ion current attributed to the lit patches, and hence each patch's
    space-charge release limit). Electron repulsion, the full-disc ion sink,
    the anode sample and the warming ion power stay FULL-DISC.

    cathode_emitting_area_initial_fraction:
        Initial lit fraction ``f_em0`` at the run's time origin. Must be finite
        and in ``(0, 1]``; with the flag off it must sit at its shipped value,
        so a run that sets a seed and forgets the flag raises rather than
        running mean-field. This is an initial condition AND a physical
        estimate of the machine's window-start emitting fraction: its shipped
        value carries a bracket, recorded with its class in
        ``config_defaults_provenance.md``.

    The growth law is the logistic ``df_em/dt = r*f_em*(1 - f_em)``, advanced
    on ACCEPTED steps only in the exactly-integrated form, so ``f_em`` is
    monotone non-decreasing for ``r >= 0``, never leaves ``(0, 1]``, and never
    falls below its seed. The rate ``r`` is NOT a key of this group: it is
    ``coverage_growth_rate_per_s``, the SAME disclosed percolation constant the
    column coverage closure uses, read here so the two surfaces share one clock
    with one owner and one fit.

    Values and their provenance: ``config_defaults_provenance.md``.
    """
    return {
        "cathode_emitting_area_initial_fraction": 0.0075,
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
        ``neutral_equilibration`` (which would overwrite the restored state) or
        a ``neutral_model`` whose distribution function the payload does not
        carry. The full inventory, and the justification for each deliberately
        dropped member, is ``_sim1d/RESTART.md``.
    """
    return {
        "restart_from": None,
    }


def neutral_probe_source_defaults():
    """Ad-hoc probe neutral source ``S_probe(z, t)`` (v1, moment model only).

    Every key here is read ONLY under the ``neutral_probe_source`` flag and is
    inert otherwise; with the flag off each must sit at its ``None`` default or
    construction raises. Nothing in this group has a shipped value: the source
    is an INSTRUMENT whose whole content is what the caller asks for, so there
    is no default amplitude, shape, waveform or placement to inherit.

    The term is a volumetric particle source on the neutral density equation,

        S_probe(z, t) = A * p(z) * w(t)   [cm^-3 s^-1],

    added to ``dn_n/dt`` as its own named RHS row (``neutral_probe_source``).
    Its injection conventions -- zero net momentum, and the single cold-gas
    ``Tn_K`` population -- are documented on
    ``physics.neutrals.neutral_probe_source_rhs``, which is the term.

    The source is a PLASMA-RUN source. Whenever the solver is on the
    neutral-only implicit stepper (the ``Plasma`` flag off, or the
    ``neutral_prebreakdown`` phase) the term is identically zero, so a probe
    can neither fuel a pre-shot fill nor reach a cached neutral-equilibration
    seed.

    neutral_probe_amplitude_cm3_s:
        Amplitude ``A`` [cm^-3 s^-1]: the source rate at ``w = 1``, averaged
        over the whole grid weighted by chamber (neutral) cell volume. The
        axial profile is normalized so that mean is exactly ``A``, which makes
        the volume-integrated influx ``A * w(t) * sum(V_chamber)``
        [particles/s] independent of the grid and of the profile's own scale.
        Must be finite and ``>= 0``; ``0`` is an explicit null-control arm, not
        a default.
    neutral_probe_profile:
        PER-CELL axial shape ``p(z)``: a sequence of length ``nx`` (the grid's
        cell count), every entry finite and ``>= 0``, not all zero. Supplied
        as a SHAPE -- its overall scale is divided out by the normalization
        above, so only its relative form matters. This is the
        externally-computed-profile hook: the solver contains no randomness and
        does no file I/O, so an arbitrary hypothesized axial source is built
        outside and passed here. EXACTLY ONE of this and
        ``neutral_probe_shape`` must be given with the flag on.
    neutral_probe_shape:
        Built-in parametric profile family, for arms that need no profile
        file. ``"gaussian"`` is the only member:
        ``p(z) = exp(-(z - z0)^2 / (2 sigma^2))`` sampled at cell centres, with
        ``z0 = neutral_probe_center_cm`` and ``sigma = neutral_probe_width_cm``,
        both then required. EXACTLY ONE of this and ``neutral_probe_profile``
        must be given with the flag on.
    neutral_probe_center_cm:
        Gaussian centre ``z0`` [cm] on the same axial coordinate as
        ``geometry.z_cm``. Required with ``neutral_probe_shape="gaussian"`` and
        forbidden otherwise. Must be finite; a centre outside the grid is
        permitted (it is a one-sided tail, not an error).
    neutral_probe_width_cm:
        Gaussian standard deviation ``sigma`` [cm]. Required with
        ``neutral_probe_shape="gaussian"`` and forbidden otherwise. Must be
        finite and ``> 0``.
    neutral_probe_waveform:
        Time dependence ``w(t)``, dimensionless, on the ABSOLUTE solver clock
        (seconds since the start of the run, the same ``time`` the RHS is
        evaluated at). One of:

        * ``"const"`` -- ``w = 1`` for all time; no further keys.
        * ``"square"`` -- ``w = 1`` on ``[t_on, t_off)`` and ``0`` outside;
          ``neutral_probe_t_on_s`` and ``neutral_probe_t_off_s`` required.
          The edges are hard: nothing is smoothed and no smoothing constant
          exists. Both edges are registered as step boundaries, which keeps
          the APPLIED rate a square; the delivered inventory does not depend
          on that and is exact on any lattice.
        * ``"table"`` -- ``neutral_probe_waveform_table`` required; linear
          interpolation between its nodes and exactly ``0`` outside their
          span.

        There is no default: the waveform decides what a probe arm measured,
        so it is stated rather than inherited.

        Whatever the form, the integration stages consume the waveform's EXACT
        AVERAGE over the step being taken, not its value at the stage times,
        so the inventory a run delivers is ``A * int w dt * sum(V)`` exactly
        -- the hypothesis as stated, independent of the step lattice. See
        ``physics.neutrals.neutral_probe_waveform_mean`` for why a pointwise
        waveform would not be.
    neutral_probe_t_on_s:
        Square-waveform rising edge [s], absolute solver clock. Required with
        ``neutral_probe_waveform="square"`` and forbidden otherwise. Must be
        finite and strictly less than ``neutral_probe_t_off_s``.
    neutral_probe_t_off_s:
        Square-waveform falling edge [s], absolute solver clock. Required with
        ``neutral_probe_waveform="square"`` and forbidden otherwise.
    neutral_probe_waveform_table:
        Tabulated ``w(t)`` as a sequence of ``[t_s, w]`` pairs: at least two
        rows, ``t`` strictly increasing, every entry finite and every ``w``
        ``>= 0``. Required with ``neutral_probe_waveform="table"`` and
        forbidden otherwise. ``w`` is ``0`` strictly outside the tabulated
        span -- a table states what the source does at the times it lists, and
        holding an end value indefinitely would deliver inventory nobody asked
        for.
    neutral_probe_zone:
        Which neutral zone the source feeds under the ``neutral_two_zone``
        closure: ``"column"`` (the plasma column, ``nn``) or ``"annulus"``
        (the surrounding chamber, ``nn_a``). Required when that flag is on --
        the two land the gas in different places and the plasma's response to
        them differs, so there is no defensible default -- and forbidden when
        it is off, where there is only one neutral field. Where a cell has no
        annulus (``V_ann = 0``) an annulus-routed source falls back to the
        column for that cell, exactly as the gas puff does.
    """
    return {
        "neutral_probe_amplitude_cm3_s": None,
        "neutral_probe_profile": None,
        "neutral_probe_shape": None,
        "neutral_probe_center_cm": None,
        "neutral_probe_width_cm": None,
        "neutral_probe_waveform": None,
        "neutral_probe_t_on_s": None,
        "neutral_probe_t_off_s": None,
        "neutral_probe_waveform_table": None,
        "neutral_probe_zone": None,
    }


def regime_tracer_defaults():
    """Pre-breakdown passive-tracer bridge (regime R2); ``regime_tracer`` flag.

    Every key here is read ONLY under the ``regime_tracer`` flag and is inert
    otherwise. With the flag off the tracer object is never constructed and no
    branch below it is reachable, so the trajectory is bit-identical to a
    checkout that has never heard of the feature.

    On a cell the tracer owns, the plasma density is the exact integral of the
    affine ODE ``dn/dt = gamma(z)*n + S(z, t)``: ``gamma`` a Picard-frozen
    functional of the slow background (bulk ionization minus recombination
    minus surface absorption) and ``S`` the n-independent beam-impact
    ionization birth. ``Te`` on those cells is the root of a quasi-static local
    electron energy balance rather than an integrated field. The method, the
    passive/active interface, and the neglect bounds are ``_sim1d/NUMERICS.md``
    (section "Regime-R2 pre-breakdown passive-tracer bridge"). Values and their
    provenance classes are ``config_defaults_provenance.md``.

    A cell is PASSIVE while all three criteria below hold; it activates (and
    the fluid takes it over) when any of them fails AND its density has reached
    ``tracer_activation_ne``. Each criterion is expressed as a ratio that must
    stay ``<= 1`` after division by its constant, so the three are directly
    comparable and the census can name the one that binds.

    tracer_passivity_current_ratio:
        Criterion (a). Largest share of the loop current the cell may CONDUCT
        and still count as passive, dimensionless in ``(0, 1]``. The conducted
        current is ``I_cond = sigma_par(n, Te) * A_plasma * V_dev / L_plasma``
        with the Spitzer parallel conductivity and the R1-bounded device
        voltage; it is the current the plasma actually passes under the applied
        drop, NOT the cathode's emission capability. Raises at construction
        outside ``(0, 1]``.

        ``I_cond`` is an UPPER BOUND, so this criterion is one-sided:
        satisfying it establishes passivity, failing it does not establish the
        converse. The bound puts the WHOLE device drop across the column, while
        most of that drop is the cathode sheath fall, so the axial field and
        therefore ``I_cond`` are overstated. Two consequences for a caller:
        cells are handed to the fluid earlier than the physics alone requires
        (the safe direction), and at low density it is the
        ``tracer_activation_ne`` gate rather than this criterion that binds.
    tracer_passivity_thinness:
        Criterion (b). Largest cumulative single-pass fraction of the beam's
        energy the plasma may absorb and still count as passive, dimensionless
        in ``(0, 1]``. Accumulated along each cathode's ray from the launch end
        as ``sum (dE/dx)_plasma * dz / E_beam`` with the Coulomb slowing rate
        on plasma electrons; the max over ends is the cell's value. Raises at
        construction outside ``(0, 1]``.
    tracer_passivity_depletion:
        Criterion (c). Largest fraction of the local neutral density the
        plasma's own bulk ionization may burn and still count as passive,
        dimensionless in ``(0, 1]``. The beam's neutral debit is NOT counted --
        the beam is background, and criterion (c) measures the plasma's
        back-reaction on the neutrals, not the discharge's. Raises at
        construction outside ``(0, 1]``.
    tracer_passivity_hysteresis:
        Enter/exit ratio on all three criteria, ``> 1``. A cell activates when
        its worst ratio exceeds ``1`` and can only return to passive when that
        ratio falls below ``1 / tracer_passivity_hysteresis``. All three ratios
        are monotone increasing while the discharge builds, so re-entry is not
        an expected event; the width exists so that a cell sitting exactly on a
        criterion cannot chatter between descriptions on round-off and make the
        step sequence irreproducible. Raises at construction at ``<= 1``.
    tracer_refresh_tol:
        Picard cadence for ``gamma`` and the quasi-static ``Te``: both are
        frozen until the largest relative change in the background they are
        built from (``n``, ``nn``, ``S``) exceeds this, then both are rebuilt.
        ``0`` refreshes every step. A numerics tolerance in the same family as
        ``circuit_picard_tol_rel``, not a description-selecting constant.
        Raises at construction if negative.
    tracer_activation_ne:
        Handoff density [cm^-3]: the density at or above which the FLUID
        description is usable, so a cell that has failed passivity may be given
        to it. Must sit far above ``ne_floor`` -- handing the fluid a cell
        whose density the floor clip is holding up would reproduce exactly the
        floor-poisoned regime the tracer exists to skip -- and construction
        raises unless it is at least ten times ``ne_floor``.
    tracer_overlap_band_ne:
        Two-element ``[low, high]`` density band [cm^-3] over which BOTH
        descriptions are valid and must therefore agree: at or above
        ``tracer_activation_ne`` (fluid valid) and below the density at which
        passivity fails (tracer valid). Read by
        ``scripts/regime_r2_overlap_gate.py``, which is the two-sided gate.
        Construction raises unless ``0 < low < high``.
    tracer_overlap_rtol:
        Relative agreement the two descriptions must reach inside that band for
        the overlap gate to PASS. Raises at construction if not positive.
    """
    return {
        "tracer_passivity_current_ratio": 0.01,
        "tracer_passivity_thinness": 0.01,
        "tracer_passivity_depletion": 0.01,
        "tracer_passivity_hysteresis": 3.0,
        "tracer_refresh_tol": 0.01,
        "tracer_activation_ne": 1.0e10,
        # A LIST, not a tuple: the resolved config round-trips through JSON in
        # the HDF5 result header, and a tuple comes back as a list, so a tuple
        # default would fail the saved-vs-rebuilt config identity check.
        "tracer_overlap_band_ne": [1.0e10, 1.0e11],
        "tracer_overlap_rtol": 0.05,
    }


def regime_vessel_node_defaults():
    """Vessel / common-mode node constants (flag ``regime_vessel_node``).

    Inert unless the flag is on, in which case each value below is validated
    at construction. The node itself is ONE state variable ``V_cm``, the
    anode-to-wall (common-mode) potential, obeying
    ``C_total dV_cm/dt = I_wall_net`` with ``I_wall_net`` the electron current
    landing on wall-connected surfaces minus the ion wall flux from the column
    minus the leak. Method of record: ``_sim1d/NUMERICS.md``, section "Vessel
    common-mode node"; hardware provenance and its class are
    ``config_defaults_provenance.md``.

    vessel_capacitance_F:
        ``C_total`` [F]: the total capacitance bridging the floating
        cathode/anode system to the vessel wall. The LAPD cathode/anode system
        floats with respect to the machine wall, the whole electrically
        connected stainless vessel is ONE wall conductor, and the anode is
        referenced to it only through four feedthrough capacitors across the
        ceramic gap insulators, so ``C_total`` is their parallel sum. Must be
        finite and positive; construction raises otherwise. This value is
        ESTIMATED and the BRACKET is the claim, not the shipped number — see
        the provenance note, and sweep it rather than quoting it.
    vessel_leak_resistance_ohm:
        ``R_leak`` [Ohm]: the resistive tie from the same node to the wall,
        draining ``V_cm/R_leak``. Must be positive and finite; construction
        raises on zero, negative or non-finite. ``None`` is accepted and means
        the idealized HARD FLOAT (no DC path at all) — an explicit A/B arm.

        The capacitor TYPE is visually UNRESOLVED, so the value is ESTIMATED
        over a bracket spanning both readings (2.5e7 Ohm, the aged-electrolytic
        low edge, to 1e11 Ohm, the polypropylene-film insulation class); the
        shipped default takes the second-look FILM reading. The bench
        measurement resolves it. See the provenance note, and sweep it rather
        than quoting it.

        **The structural fact the model rests on does not depend on the type.**
        ``R_leak*C_total`` is at least ~10 s at BOTH bracket edges, against a
        ~25 ms discharge, so within a shot the node is hard-float **in kind**
        either way and the leak moves nothing that a run measures.

        Two documented model deviations, neither of them built. (i) POLARITY,
        conditional on the unresolved type: IF the capacitors are electrolytic
        they are polarized and conduct asymmetrically under reverse bias
        (diode-like above ~1-2 V), which this node can reach because the
        machine's plateau bias is observed at EITHER sign — the shipped leak is
        SYMMETRIC, a linear resistor in both directions. IF they are film there
        is no polarity nuance at all, and the black band on one side of the
        cylinder is the conventional OUTER-FOIL marking (a shielding
        convention; electrolytics mark polarity with explicit -/+ symbols).
        (ii) INTER-SHOT MEMORY: with a leak timescale far longer than the ~3 s
        shot period under either reading, the capacitors cannot discharge the
        node between shots, so the physical reset path is the afterglow plasma
        conductance, not the leak. Runs here are single-shot and start from
        ``V_cm = 0``.
    """
    return {
        "vessel_capacitance_F": 1.3e-6,
        "vessel_leak_resistance_ohm": 1.0e10,
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
    coverage_closure_defaults,
    emitting_area_defaults,
    restart_defaults,
    neutral_probe_source_defaults,
    regime_tracer_defaults,
    regime_vessel_node_defaults,
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
    # The resolved typed-segment geometry is the only geometry. Retained as a
    # stale-config guard; False raises at construction.
    "resolved_boundaries": True,
    # End-vessel / magnetic-flare geometry. Presence gated in core.geometry:
    # all three end_expansion_* parameters are required when on and forbidden
    # when off. Bit-exact off.
    "end_expansion_geometry": False,
    # Prescribed per-cell geometry: the plasma flux-tube area A(z), supplied
    # as the radius vector plasma_radius_profile_cm (sqrt(A/pi)), and
    # optionally the vessel bore as machine_radius_profile_cm. They replace
    # the uniform scalars Rp and Rm cell by cell, so the cell volumes, the
    # face areas, the neutral conductances and the two-zone annulus volume
    # V_ann = Vm - Vp all follow the profiles; the quasi-1D
    # flux_tube_geometry momentum source (the well-balanced p dA/dz mirror
    # force paired with the area-weighted pressure flux) and the
    # area-consistent d(Au)/dz pressure work in both energy equations come on
    # with it. It exists because the built-in end_expansion_geometry flare is
    # a half-cosine with zero slope at BOTH ends (which no solved convex B(z)
    # has) applied against ONE vessel radius over the whole terminal block
    # (which a stepped bore is not); the profiles are computed offline from
    # the field and the machine drawing and passed in.
    # Presence gated in core.geometry in both directions (the plasma profile
    # is required when on and every profile key is forbidden when off) and
    # REFUSED together with end_expansion_geometry -- two prescriptions of the
    # same area with no composition rule. Default OFF and bit-exact off; a
    # constant plasma profile at Rp with no vessel profile is bit-identical to
    # it being off.
    "prescribed_area_geometry": False,
    # Thin annular apertures. Positions and clear radii are required together
    # when on and forbidden when off; the plasma channel stays open.
    "neutral_baffles": False,
    # Fixed-cell-size source region, so a mesh refinement study is not
    # self-confounding: the column between the anode face and
    # source_region_length_cm is meshed at exactly source_region_dz_cm
    # regardless of nx, which then refines only the far column, and the puff
    # role follows gas_puff_z_cm instead of the first column cell. Presence
    # gated in core.geometry in both directions (both parameters required when
    # on, forbidden when off) and incompatible with TwinCathode. Structurally
    # bit-exact when off.
    "source_fixed_grid": True,
    "heat_conduction": True,
    "implicit_heat_conduction": True,
    # Flux-limited electron heat conduction. The classical Spitzer-Harm flux
    # can exceed the free-streaming scale n*Te*v_the at resolved gap faces,
    # i.e. exceed the physical ceiling. When ON, the electron conductivity is
    # scaled per cell by lambda = q_sat/(q_sat+q_SH) -- the harmonic form of
    # Malone 1975 / Fundamenski 2005 eq. 10a, riding on the Cowie & McKee 1977
    # free-streaming ceiling q_sat = heat_flux_limiter_f * n * Te * v_the -- so
    # the flux caps at
    # free-streaming where gradients are steep and recovers Spitzer where they
    # are shallow. Electron only; ion conduction unchanged. Bit-exact when
    # off; a declared closure-family A/B instrument. The cap COEFFICIENT
    # heat_flux_limiter_f is a separate input_dict key with its own default.
    "electron_heat_flux_limit": True,
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
    "ion_neutral_drag": True,
    "ion_neutral_drag_cx_only": False,
    # Evolve axial neutral momentum M_n as a sixth conservative field:
    # the drag deposits its momentum into the
    # neutral wind instead of a closure, ionization/recombination exchange
    # momentum between species, and the wall/pump remove it. Mutually
    # exclusive with ion_neutral_drag_model="slip", whose closure is this
    # equation's own local steady state. Off => the 5-field state.
    "neutral_momentum": True,
    # Split the neutral density into plasma-column and annulus zones:
    # an optional conservative field nn_a
    # carries the annulus density and nn becomes the COLUMN density. Axial
    # Knudsen transport runs per zone (the annulus is a free conduit), the
    # zones exchange free-molecularly at the column surface, and the
    # plasma only ever absorbs column gas. Requires
    # neutral_exchange_model="knudsen" (the per-zone conductances have no
    # constant counterpart). Off => single-field
    # chamber-mean nn.
    "neutral_two_zone": True,
    # Route the END-REGION recycle stream into the annulus. The plasma-
    # terminating faces whose live cell has the COLLECTOR role rebirth their
    # absorbed flux as thermal diffuse gas in that cell's annulus row nn_a
    # (dN_loss / V_ann) instead of in its column row nn; the CATHODE faces are
    # untouched, so the ratified jet/debit closure over them is unchanged. The
    # plasma-side rows (n, M, Ee, Ei and the sonic momentum debit) are
    # unchanged either way -- this moves where the returning atoms land, not
    # how much plasma the surface takes. The routed atoms carry NO directed
    # momentum, on M_n or M_n_a: a diffuse thermal re-emission has none, and
    # the chamber-mean wind is left alone.
    #
    # ENERGY. The recycled atoms are booked at the wall temperature exactly
    # once. The routed stream leaves the column nn row, and with it the
    # (3/2) k T_wall column-En credit the neutral-energy routing table grants
    # every "wall" source; the annulus carries no energy field, and the
    # zone-exchange convention already re-supplies wall-temperature enthalpy
    # when annulus gas re-enters the column. Booking both would plant the same
    # energy twice.
    #
    # Requires neutral_two_zone (the destination row must exist), and refuses
    # any geometry whose routed collector cell has no annulus (V_ann = 0);
    # both are construction-time ValueErrors. Default OFF and bit-exact off
    # (presence-gated: the off path passes no annulus volume to either
    # boundary term and adds no nn_a row). Applies to whichever of the two
    # plasma-terminating discretizations the run configured --
    # characteristic_boundary or the volumetric absorber.
    "end_recycle_to_annulus": False,
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
    # Requires ion_neutral_moment_closure and neutral_momentum; refuses
    # coverage_closure (its deficit partitions nn only, so a mean En under
    # concentration would be an unstated closure), the two-momentum reduction
    # (an annulus momentum row with no annulus energy row), and every kinetic
    # neutral model (which carries the neutral energy as a moment of f). With
    # cathode_neutral_jet it additionally requires cathode_jet_surface_debit,
    # so the backscatter energy is booked once rather than twice. Each is a
    # construction-time ValueError. Off => the historical layout, bit-exact.
    "neutral_energy": True,
    # Launch the hot channel's CX-born atoms at the local (Ti, u_i) instead of
    # at Ti alone: the ion drift enters the ballistic flight kinematics as
    # v_z = v_hot*mu + u_i, so the axial hop becomes
    # dz = chord*(mu + m)/sqrt(1 - mu^2) with m = u_i/v_hot, and the landing,
    # residence and end-plane matrices become row-wise drift-asymmetric.
    # Consumed by the neutral_hot_channel term alone (physics.hot_neutrals):
    # it selects directed_flight_kernels over ballistic_flight_kernels and
    # additionally computes the hot_n_flight / hot_flux_z streaming
    # diagnostics, which read zero when this flag is off.
    #
    # mu stays uniform on [-1, 1] and v_perp is untouched (the drift is axial),
    # so nu_ball = v_hot/Rp, every branching ratio and the standing population
    # are unchanged; only WHERE the flights get to moves. The momentum a hot
    # atom carries, p_hot = m*u_i, is the launch MEAN and already directed, so
    # no new momentum source is booked and the ion/cold/hot closure is the same
    # one. No new constant: m is local state.
    #
    # COST: the kernel stops being a pure function of the geometry (the speed
    # no longer cancels from dz), so it is rebuilt on every RHS evaluation
    # rather than once per run.
    #
    # Requires neutral_energy -- there is no hot channel to launch without it
    # -- as a construction-time ValueError. Default OFF and bit-exact off
    # (presence-gated: the off path builds no drift kernel and the isotropic
    # one is untouched).
    "neutral_hot_birth_drift": False,
    # Partition the two-zone WALL BRANCH of the neutral momentum ledger. The
    # free-molecular wall sink -nu_wall*M_n_a assumes every annulus atom
    # reaches the vessel wall; at finite gas density a He-He elastic collision
    # can intercept it first, and that momentum stays in the annulus gas
    # instead of accommodating on the surface. The surviving fraction is the
    # cosine-averaged slab transmission 2*E_3(tau) across the annulus radial
    # thickness, tau = (Rm - Rp)*nn_a*sigma_HeHe (see
    # physics.sources.neutral_wall_partition_survival). MOMENTUM ONLY: the
    # particle and energy channels are untouched.
    #
    # Requires neutral_momentum_radial='kinetic_two_moment' (the only closure
    # that owns a wall branch on its own annulus momentum row) and REQUIRES
    # neutral_wall_partition_sigma_hehe_cm2, which has no default -- arming the
    # flag without it raises at construction, and setting the cross section
    # without the flag raises too. Default OFF and bit-exact off (presence
    # gated: the off path passes sigma_hehe_cm2=None and the operator's
    # arithmetic is unchanged).
    "neutral_wall_momentum_partition": False,
    # Wall the hot channel's ballistic flight at the INTERNAL plasma
    # boundaries, not only at the two global end planes. The walls are the
    # closed plasma faces (geometry.plasma_open false: every face where a
    # plasma-dead cell -- plenum, obstruction -- abuts a live one, plus the two
    # end planes) together with the plasma-absorbing faces, which are a
    # refinement of that set. A flight reaching one is clipped to the wall
    # plane and the atom is booked in the cell on its OWN side of it, which is
    # exactly the fold/absorb treatment the end planes already get; the landed
    # atoms rejoin the COLD neutral books (nn, or nn_a under neutral_two_zone)
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
    # cell -- the live cell against a cathode disc or a collector -- is the
    # cell that receives everything folded at that wall, on both counts. The
    # mask itself is untouched; the flag only stops feeding it rows to delete.
    # Cells with no column (Rp = 0) keep the in-place identity row they already
    # had.
    #
    # Consumed by the neutral_hot_channel term alone (physics.hot_neutrals):
    # it is passed to ballistic_flight_kernels, and to directed_flight_kernels
    # under neutral_hot_birth_drift so the two kernels cannot disagree about
    # where the walls are. hot_end_fraction then reads "folded at a wall"
    # rather than "folded at an end plane".
    #
    # Requires neutral_energy -- there is no hot channel to wall without it --
    # as a construction-time ValueError. Bit-exact when off (presence-gated:
    # the off path's wall bounds ARE the two end planes, so every clip reduces
    # to the historical one).
    "neutral_hot_internal_wall": True,
    # Shaped initial neutral fill. The run's neutral IC comes from a PER-CELL
    # profile of absolute densities (nn0_profile, and optionally
    # nn0_annulus_profile under neutral_two_zone) instead of the uniform
    # scalar nn0. Values, not a shape: nothing is rescaled or normalized, so
    # the array IS the initial condition. Requires nn0_profile, requires
    # nn0 = None (the scalar and the table lookup are superseded, and an
    # armed flag with an explicit scalar raises rather than establishing a
    # silent precedence), and REFUSES neutral_equilibration and restart_from
    # -- both of those overwrite nn after construction, so a shaped IC under
    # either would be silently discarded. Each is a construction-time
    # ValueError, as is either profile key set with this flag off. Default
    # OFF and bit-exact off (presence-gated: the off path builds no profile
    # and the initial condition is the historical uniform fill).
    "neutral_initial_profile": False,
    "ion_neutral_thermalization": False,
    # Replace the drag + frictional-heating + elastic thermalization +
    # CX-cooling quartet with ONE moment-closed reduced ion-neutral collision
    # operator (Phelps He+/He isotropic+backscatter rates, T_eff=(Ti+Tn)/2).
    # Presence-gated: when ON the four legacy ion-neutral terms are forced to
    # zero and this single operator runs; when OFF it is a strict no-op.
    # He-only. Uses the single cold-gas Tn_K for the neutral temperature,
    # ending the Tn_K/Tn_fit term-by-term mix. With this ON the ad-hoc
    # b_ion_neutral_drag / slip closures are
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
    # Bound the device voltage by what the CIRCUIT can supply, *current-driven*
    # sheath solve only. The
    # ceiling the sheath root is solved against becomes
    # min(cathode_phi_c_cap_V, <the circuit member>) -- the atomic-data cap
    # composed with the loop equation V_src - I*(R_comp + R_mesh_ohm) read at
    # dI/dt = 0 -- so the returned phi_c, the beam birth energy keyed to it,
    # and the capability-limited device voltage V_b are all held at or below
    # the supply. Which quantity the available voltage bounds is
    # cathode_circuit_bound_object's choice. Without the flag the
    # capability-limited branch floors V_b at the data cap, which on the
    # pre-breakdown build leg reports ~1000 V and a ~keV beam against a bank
    # supplying ~178 V. The cap itself is untouched and still composes as the
    # other upper bound (it is the He EII table top, an atomic-data domain
    # guard). The inductor's back-EMF is deliberately NOT counted as available
    # voltage. Requires cathode_solver_model='current_driven',
    # cathode_coupling and V_bank > 0; inactive (ceiling falls back to the
    # data cap) wherever the available voltage is not positive, notably the
    # zero-bank inductive tail. Default OFF and bit-exact off.
    #
    # WHAT THE BOUND DOES NOT BOUND is the loop current. The circuit
    # integrates the sheath's UNBOUNDED demand, not this clamped V_b, so the
    # restoring force survives the clamp. Feeding the clamped value back in
    # was the ratchet defect (2026-08-12): the loop residual went identically
    # zero above the capability wall, dI/dt >= 0 everywhere, and I_loop
    # became the running maximum of the TR stage's explicit overshoot --
    # 156.7 A in one 2e-5 s step against a converged 0.9 A. See
    # cathode.idriven_vdis_evaluator.
    #
    # SCOPE. A FULL-WINDOW RUN WITH THIS FLAG ON IS IN CONTRACT (2026-08-12);
    # both of the exclusions that once narrowed it are gone. The phi_c/V_b
    # OBJECT mismatch went with cathode_circuit_bound_object='device_voltage',
    # which bounds V_b itself, so phi_c may legitimately exceed the available
    # voltage where the anode fall subtracts. The BACK-EMF exclusion went with
    # the integrand: it held that while the bound binds the loop residual is
    # identically zero, hence dI/dt = 0, hence a frozen main-discharge decay.
    # That was true of the old integrand and is not true of this one --
    # measured on the ON-probe build leg, where the current FALLS on 12 of 33
    # bound saves. Nothing raises when the bound engages; only the
    # bound_active census shows it, and the probe window reaches no plateau
    # decay, so reading the census on a run that does remains worthwhile.
    "cathode_circuit_voltage_bound": False,
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
    # (electron/ion energy margin within solver.SURFACE_LOSS_FLOOR_EXEMPT_RTOL
    # of the per-cell floor energy 3/2 n Te_floor, i.e. Te within 0.1% of the
    # floor) are excluded from the drain-margin bound ONLY -- the
    # accept-time floor clip resets their margin to float residue every step,
    # so a persistent drain otherwise pins dt at dt_min indefinitely. One-sided
    # (all other bounds still govern the cell) and knife-edge (any real margin
    # re-admits the cell immediately). ON by default because a run that reaches
    # a floor-pinned afterglow otherwise cannot finish in finite time. Set it
    # False to recover the historical bound.
    "surface_loss_floor_exempt": True,
    # Include the beam_ionization_birth row in the resolved electrode/source
    # ("surface_loss") timestep bound. Default OFF and bit-exact off.
    #
    # The row is in NO timestep bound today: the bundle carries only the
    # boundary, anode-collection and cathode-surface rows, so beam-driven
    # birth -- a volumetric plasma source of unbounded magnitude that CAN
    # drive a cell into a floor within one step -- has never constrained dt.
    # That is pre-existing and solver-wide, not specific to any arm, and the
    # row is measured healthy on the current arms; this is insurance, not a
    # hot fix.
    #
    # Turning it ON changes the suggested timestep wherever the row is live,
    # so it MOVES THE GOLDEN and the default-flip decision is deliberately
    # left open rather than taken here.
    "beam_ionization_birth_timestep_bound": False,
    # Clumpy-plasma coverage closure v1. Breakdown in the machine is
    # azimuthally patchy -- discrete channels carry the discharge -- which the
    # 1D mean-field solver azimuthally averages away. When ON, a scalar
    # coverage fraction f_cov(t) in (0, 1] splits the beam by AREA between the
    # covered channels (concentrated plasma, n -> n/f_cov) and the uncovered
    # reservoir (tenuous, its own neutrals), and splits the neutrals into a
    # burnt covered column and a reservoir that refills it. The MEAN equations
    # are untouched, so total particle inventory is conserved identically; at
    # f_cov = 1 every factor reduces to the shipped model. Default OFF and
    # bit-exact off (presence-gated: the off path never builds the coverage
    # view and every consumer keeps its historical argument list). Requires
    # coverage_initial_fraction, beam_deposition_model="csda",
    # neutral_model="moment", no beam clumping, and the pure-Python kernels;
    # each is a construction-time ValueError.
    "coverage_closure": False,
    # Cathode emitting-area percolation (ea1). Thermionic release in the
    # machine's current foot is patchy: only lit patches of the emitting face
    # carry it, and the lit area percolates outward. When ON, one scalar
    # f_em(t) in (0, 1] throttles the emission at the annuli seam -- area_k ->
    # f_em*area_k and the ion attribution frac_k -> f_em*frac_k, unrenormalized
    # -- so each patch keeps its clamp ratio and its barrier and the whole
    # space-charge release curve rescales as f_em * (the full-disc curve),
    # leaving phi_c and the beam launch energy invariant at a fixed sheath
    # state. Full-disc quantities (electron repulsion, the ion sink, the anode
    # sample, the warming ion power) are NOT scaled. f_em grows logistically on
    # accepted steps at coverage_growth_rate_per_s -- the SHARED percolation
    # clock, not a second constant. Requires cathode_coupling,
    # cathode_emission_profile="gaussian" (under "uniform" A_c is dual-use for
    # emission AND ion collection, so the throttle is not expressible) and a
    # seed in (0, 1]; each is a construction-time ValueError, as is setting the
    # seed key with this flag off. Default OFF and bit-exact off
    # (presence-gated: the off path passes no override, scales no tuple and
    # leaves every device config identical). COMPOSES with coverage_closure:
    # the two describe different surfaces (the cathode face and the column
    # cross-section), share no state, and their only common object is the
    # growth constant -- so the composition is permitted rather than refused,
    # and a composed arm must disclose that it is one.
    "cathode_emitting_area": False,
    # Ad-hoc probe neutral source S_probe(z,t) = A p(z) w(t), a volumetric
    # particle source on the neutral density equation. An INFERENCE
    # INSTRUMENT: an arm with this on measures the plasma's response to a
    # hypothesized neutral source, so it is never a validation channel and a
    # run carrying it must say so. Supports the moment neutral model only
    # (the kinetic arms take over the fluid nn rows). Requires an amplitude,
    # exactly one of neutral_probe_profile / neutral_probe_shape, a waveform
    # with its own keys, and -- under neutral_two_zone -- an explicit
    # neutral_probe_zone; each is a construction-time ValueError, as is any of
    # the ten keys set with this flag off. Default OFF and bit-exact off
    # (presence-gated: the off path builds no profile and adds no RHS row).
    "neutral_probe_source": False,
    # Pre-breakdown PASSIVE-TRACER bridge (regime R2). On a cell that is still
    # passive -- conducting a negligible share of the loop current, absorbing a
    # negligible share of the beam's single pass, and burning a negligible
    # share of the local neutrals -- the plasma feeds back on nothing, so its
    # density is integrated as the EXACT solution of the affine scalar ODE
    # dn/dt = gamma*n + S while the background (circuit ramp, cathode thermal,
    # coverage, neutrals) owns the timestep. That removes the floor-poisoned
    # dt collapse the fluid solver suffers when n sits near ne_floor, and makes
    # n = 0 a regular state, so ne0 = 0 is a legitimate initial condition.
    # Cells hand back to the full solver individually, at a closed interface
    # face; the whole-column handoff is a restart state transfer. Requires
    # cathode_coupling (the beam birth S is the cathode solve's) and refuses
    # the kinetic/kinetic_dvm neutral models (R2 is fluid-arms only) -- both
    # construction-time ValueErrors, as is any out-of-range criterion constant.
    # Default OFF, presence-gated and bit-exact off.
    "regime_tracer": False,
    # Vessel / common-mode node. The cathode/anode system FLOATS with respect
    # to the machine wall (the whole electrically connected stainless vessel is
    # one conductor; the anode is tied to it only through four feedthrough
    # electrolytic capacitors across the ceramic gap insulators). This adds ONE state
    # variable V_cm, the anode-to-wall potential, obeying
    # C_total dV_cm/dt = I_wall_net -- electron current landing on
    # wall-connected surfaces (the transmitted beam terminates on the far end,
    # which IS the vessel) minus the column's ion wall flux minus V_cm/R_leak.
    # V_cm is the potential a transmitted beam electron must CLIMB from the
    # mesh into the column, so the energy reaching column physics is
    # phi_c - max(V_cm, 0): the node throttles the beam, ionization feeds the
    # ion wall flux back, and the floating constraint (zero net system-to-wall
    # current) is what lets beam leakage into the column grow. Advanced once
    # per ACCEPTED step, in closed form over the step's frozen currents.
    # Requires cathode_coupling, Plasma, cathode_circuit_voltage_bound (the
    # beam energy the climb is subtracted from must be the circuit-bounded
    # one, never the atomic-data cap), beam_deposition_model='csda' (only the
    # CSDA rays book a transmitted flux at a terminating surface) and a
    # plasma-terminating collector face (the ion wall channel) -- each a
    # construction-time ValueError, as are a non-positive vessel_capacitance_F
    # and a non-positive vessel_leak_resistance_ohm. V_cm(t) and the three
    # current channels ride the cathode diagnostics; nothing here is scored.
    # Default OFF, presence-gated and bit-exact off.
    "regime_vessel_node": False,
    # Rate-freezing INSTRUMENT, default OFF. When on, the bulk reaction
    # source terms (ionization birth and both recombination losses) inside the
    # explicit non-heat operator are evaluated at the step-START accepted
    # state instead of at the current SSPRK2 stage state, so the rates are
    # frozen across the step. That deliberately caps the step at first order
    # in the rates: it isolates the stage-state channel for measurement and is
    # NOT an accuracy improvement. Scope is the explicit operator's reaction
    # terms only -- the implicit heat substep, the conductivity Picard
    # iteration and the kinetic DVM are untouched. Must be a real bool;
    # anything else (0/1, a string) raises ValueError at construction.
    # Bit-exact when off.
    "rates_at_accepted_state": False,
    # Anode sheath electron-energy booking, default OFF. Armed, the plasma
    # electron store is debited (2 Te + phi_a) per electron the anode
    # collects -- the sheath-edge energy flux of the truncated Maxwellian
    # whose zeroth moment the sheath solve already closes on -- rather than
    # the plasma-thermal 2 Te alone. The added phi_a * I_e_coll lands on the
    # anode-flanking cells under the SAME Bohm split weights the thermal part
    # uses, and the circuit/load ledger is untouched: the phi_a those
    # electrons pay is the field energy the loop and the anode ions already
    # book. It also re-cuts the anode mesh's own Bohm collection rows to
    # their sheath-edge values -- Te/2 per collected ion on Ee (the presheath
    # work; those electrons' thermal transport is carried by the sheath term)
    # in place of 3/2 Te, and 5/2 Ti on Ei (the enthalpy flux) in place of
    # 3/2 Ti. Requires ``characteristic_boundary``, whose thermal-only
    # electrode routing this completes: with that routing off the full
    # P_anode_e is already deposited and arming this would debit the fall
    # twice, so the combination is a construction-time ValueError. TWO
    # REGIMES, both booked: the increment above is the electron-REPELLING
    # anode (phi_a > 0), where the collected electrons climbed the fall and
    # the plasma paid; at an electron-ATTRACTING anode (phi_a <= 0 -- an
    # anode demanding at or above electron saturation) the field does work ON
    # the electrons, the BANK pays the fall, and the plasma-side debit stays
    # the thermal-only 2 Te, so no increment is applied and the booking is
    # the unarmed one. That branch is counted rather than silent: an armed
    # run's cathode diagnostics carry ``anode_attracting_steps`` (accepted
    # steps that took it, cumulative) and ``anode_attracting_last_time_s``.
    # A non-finite phi_a belongs to neither regime and raises RuntimeError.
    # Must be a real bool. Bit-exact when off.
    "anode_sheath_full_debit": False,
    "ionization_energy_cost": True,
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
    """Return configured or table-derived initial neutral density [cm^-3].

    CONVENTION INCONSISTENCY on the fallback branch, documented rather than
    silently patched: ``nn_table``'s keys are pre-2026-08-21 0 C-sccm while
    ``S_gp`` is now meter-sccm, so the lookup is off by the ~7% conversion
    ratio. It is not converted here because the frozen table cannot be
    regenerated on the new convention (its generator retired with _sim3) and a
    lookup-time conversion would invent an interpolation of data that was
    never computed. Production never reaches this branch: both the config
    default and the stance of record pin ``nn0`` explicitly, so the line above
    short-circuits.
    """
    nn0 = input_dict.get("nn0")
    if nn0 is not None:
        return nn0
    return lookup_nn0(input_dict["S_gp"], twin=input_flags["TwinCathode"])
