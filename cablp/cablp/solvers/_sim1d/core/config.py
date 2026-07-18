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
    Tn_fit:
        Neutral temperature used in reaction-rate fits [eV].
    Te0:
        Uniform initial electron temperature [eV].
    Ti0:
        Uniform initial ion temperature [eV].
    u0:
        Uniform initial axial plasma velocity [cm/s].
    """
    return {
        "gas_type": "He",
        "ne0": 1e9,
        "nn0": 1e9,
        "Tn_fit": 0.1,
        "Te0": 0.1,
        "Ti0": 0.1,
        "u0": 0.0,
    }


def geometry_defaults():
    """Return defaults for the 1D axial source-domain-end geometry.

    Lm:
        Total machine length represented by the 1D mesh [cm].
    Lz:
        Resolved axial domain length between the source and end cells [cm].
    source_length_cm:
        Length of the lumped source/cathode boundary cell [cm].
    end_length_cm:
        Length of the lumped end/anode or collector boundary cell [cm].
    nx:
        Number of resolved axial cells between source and end cells.
    Rm:
        Default neutral/machine radius [cm].
    Rp:
        Default plasma radius [cm].
    source_Rm, end_Rm:
        Optional neutral/machine radii for source and end cells [cm]. ``None``
        means use ``Rm``.
    source_Rp, end_Rp:
        Optional plasma radii for source and end cells [cm]. ``None`` means use
        ``Rp``.

    The keys below configure the *resolved* typed-segment geometry, selected by
    the ``resolved_boundaries`` flag (BOUNDARY_REGIONS_PLAN.md §3). They are
    inert while that flag is off. Each defaults to its legacy limit so a resolved
    run with the obstruction/rod knobs at zero collapses toward the lump (§13).

    plenum_length_cm:
        Length of each neutral-only plenum cell behind a cathode [cm].
    cathode_length_cm:
        Length of each cathode boundary cell [cm].
    anode_length_cm:
        Length of each anode boundary cell [cm].
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
    """
    return {
        "Lm": 2000.0,
        "Lz": 1800.0,
        "source_length_cm": 100.0,
        "end_length_cm": 100.0,
        "nx": 60,
        "Rm": 50.0,
        "Rp": 18.0,
        "source_Rm": None,
        "end_Rm": None,
        "source_Rp": None,
        "end_Rp": None,
        # Resolved typed-segment geometry (inert while resolved_boundaries off).
        "plenum_length_cm": 100.0,
        "cathode_length_cm": 30.0,
        "anode_length_cm": 10.0,
        "collector_length_cm": 100.0,
        "Rcs": 0.0,
        "Lcs": 0.0,
        "Rsup": 0.0,
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
    """
    return {
        "ne_floor": 1e8,
        "nn_floor": 1e8,
        "Te_floor": 0.1,
        "Ti_floor": 0.1,
    }


def neutral_source_defaults():
    """Return gas-puff, pump, and neutral-source defaults.

    S_gp:
        Source-side gas puff flow [sccm].
    Twin_S_gp:
        End-side gas puff flow used when ``TwinCathode`` is enabled [sccm].
    gas_puff_mode:
        Phase-dependent gas-puff schedule. Options are
        ``"decay_after_breakdown"`` for steady puffing until optional decay
        after breakdown/main-discharge start, and
        ``"pulse_decay_to_level"`` for a full-rate pulse followed by decay
        toward the configured target flow.
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
    pump_elbow_conductance_lps:
        Conductance of the unmodeled pump elbow [L/s], combined in series with
        the pump speed as ``1/S_eff = 1/S_pump + 1/C_elbow``
        (BOUNDARY_REGIONS_PLAN.md §4). Applies only to a pump sitting on a plenum
        cell, so it is inert in legacy geometry. ``None`` (default) or a
        non-positive value means no elbow restriction -- the legacy limit.
    """
    return {
        "S_gp": 6500,
        "Twin_S_gp": 5500,
        "gas_puff_mode": "pulse_decay_to_level",
        "S_gp_decay_target": 1500,
        "Twin_S_gp_decay_target": 0.0,
        "tau_gp_after_breakdown": None,
        "tau_gp_decay_factor": 1.0,
        "tau_gp_pulse_duration": 1e-3,
        "tau_gp_decay_duration": 5e-3,
        "S_pump_L": 2000,
        "S_pump_R": 4000,
        "gas_puff_enabled": True,
        "pump_enabled": True,
        "gas_puff_valves": 2,
        "pump_elbow_conductance_lps": None,
    }


def timing_defaults():
    """Return phase timing and current-trigger defaults.

    tau_prebreakdown:
        Maximum pre-breakdown duration or scheduled pre-breakdown phase [s].
    tau_neutral_prebreakdown:
        Optional neutral-only accumulation duration before plasma/cathode
        current-triggered phases begin [s].
    tau_breakdown:
        Scheduled breakdown duration before main discharge when not using
        current-triggered transitions [s].
    tau_discharge:
        Main-discharge duration [s].
    tau_afterglow:
        Afterglow duration after the main discharge [s].
    tau_cycle:
        Neutral-only puff/off cycle duration [s].
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
    """
    return {
        "tau_prebreakdown": 0.05,
        "tau_neutral_prebreakdown": 0.002,
        "tau_breakdown": 0.0,
        "tau_discharge": 20e-3,
        "tau_afterglow": 5e-3,
        "tau_cycle": 3.0,
        "cycles": 1,
        "neutral_equilibration_cycles": 100,
        "neutral_equilibration_dt": 1e-2,
        "phase_transition_mode": "current",
        "I_prebreakdown": 150.0,
        "I_breakdown": 1000.0,
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
    """
    return {
        "dt_save": 1e-5,
        "t_save_start": 0.0,
        "max_output_steps": 0,
    }


def model_mode_defaults():
    """Return string-valued model selector defaults.

    front_flux_model:
        Axial plasma front-filling flux closure. The implemented option is
        ``"sonic_relaxation"``; the ``front_flux`` flag enables or disables the
        closure.
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
    neutral_exchange_model:
        Neutral axial exchange model. Options are ``"constant"`` and
        ``"molecular_flow"``.
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
        "front_flux_model": "sonic_relaxation",
        "D_amb_model": "cs_dz",
        "end_mode": "collector",
        "cathode_model": "disabled",
        "Te_birth_ionization": "local",
        "Ti_birth_ionization": "floor",
        "neutral_exchange_model": "molecular_flow",
        "operator_splitting": "lie",
        "implicit_heat_scheme": "backward_euler",
    }


def fudge_factor_defaults():
    """Return physics scale factors and boundary geometry multipliers.

    alpha_front:
        Multiplier for the front-filling/sonic relaxation flux.
    D_amb:
        Constant ambipolar diffusion coefficient when selected [cm^2/s].
    b_ioniz:
        Bulk ionization particle source scale factor.
    b_rec_rad:
        Radiative recombination particle sink scale factor.
    b_rec_3b:
        Three-body recombination particle sink scale factor.
    b_Qie:
        Electron-ion thermal exchange scale factor.
    b_Qei:
        Electron-ion inelastic/radiative cooling scale factor.
    b_Qen:
        Electron-neutral inelastic cooling scale factor.
    b_Qcx:
        Ion charge-exchange cooling scale factor.
    b_epara:
        Electron axial heat-conduction scale factor.
    b_ipara:
        Ion axial heat-conduction scale factor.
    b_ionization_energy_cost:
        Electron energy cost scale for ionization potential losses.
    b_pressure_work_elec:
        Electron pressure-work source scale factor.
    b_pressure_work_ions:
        Ion pressure-work source scale factor.
    b_surface_loss:
        Plasma surface neutralization/loss scale factor.
    b_ion_neutral_drag:
        Ion-neutral drag (friction) momentum-sink scale factor.
    sigma_in_cm2:
        Ion-neutral momentum-transfer cross section [cm^2].
    alpha_isat:
        Ion-saturation/surface-loss coefficient.
    source_surface_area_scale:
        Surface-loss area multiplier for the source boundary cell.
    end_surface_area_scale:
        Surface-loss area multiplier for the end boundary cell.
    """
    return {
        "alpha_front": 1.0,
        "D_amb": 0.0,
        "b_ioniz": 1.0,
        "b_rec_rad": 1.0,
        "b_rec_3b": 1.0,
        "b_Qie": 1.0,
        "b_Qei": 1.0,
        "b_Qen": 1.0,
        "b_Qcx": 1.0,
        "b_epara": 1.0,
        "b_ipara": 1.0,
        "b_ionization_energy_cost": 1.0,
        "b_pressure_work_elec": 1.0,
        "b_pressure_work_ions": 1.0,
        "b_surface_loss": 1.0,
        "b_ion_neutral_drag": 1.0,
        "sigma_in_cm2": 5.0e-15,
        "alpha_isat": 0.6065306597126334,
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
        Richardson constant [A cm^-2 K^-2].
    R_comp:
        External/compliance resistance [Ohm].
    eta:
        Anode-to-cathode area ratio.
    L_cath:
        Cathode-to-anode distance used by the cathode solver [cm].
    R_cath:
        Cathode radius used to compute cathode area [cm].
    """
    return {
        "V_bank": 180.0,
        "T_s": 1975,
        "phi_wf": 3.0,
        "C_R": 29.0,
        "R_comp": 0.010,
        "eta": 0.358,
        "L_cath": 50.0,
        "R_cath": 18.0,
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
    """
    return {
        "heat_picard_iterations": 0,
        "heat_picard_tol": 1e-10,
        "ln_lambda_min": 1.0,
        "Tn_K": 300.0,
        "neutral_exchange_coeff_cm3_s": 1.0e5,
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
    """
    return {
        "cfl": 0.4,
        "density_dt_fraction": 0.25,
        "neutral_dt_fraction": 0.25,
        "heat_dt_fraction": 0.25,
        "dt_min": 1e-10,
        "dt_max": 1e-4,
        "max_steps": 0,
        "adaptive_retries_enabled": True,
        "max_step_retries": 8,
        "dt_reject_factor": 0.5,
        "dt_growth_enabled": True,
        "dt_growth_factor": 1.25,
        "max_density_step_fraction": 0.0,
        "max_neutral_step_fraction": 0.0,
        "max_energy_step_fraction": 0.0,
    }


def build_input_dict_template_1d():
    """Compose the public flat input-default dictionary from grouped defaults."""
    input_dict = {}
    for defaults in (
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
    ):
        input_dict.update(defaults())
    return input_dict


input_dict_template_1d = build_input_dict_template_1d()


input_flags_template_1d = {
    "Plasma": True,
    "TwinCathode": False,
    # Master switch for the resolved source/end boundary redesign
    # (BOUNDARY_REGIONS_PLAN.md §13). Off => the legacy source/domain/end lump,
    # reproducing today's behavior bit-for-bit. On => the typed-segment machine.
    "resolved_boundaries": False,
    "heat_conduction": True,
    "implicit_heat_conduction": True,
    "front_flux": True,
    "source_surface_loss": True,
    "end_surface_loss": True,
    "ion_neutral_drag": True,
    "ion_neutral_drag_cx_only": False,
    "ion_neutral_thermalization": False,
    "cathode_coupling": True,
    "neutral_prebreakdown": True,
    "neutral_equilibration": True,
    "launch_plasma_after_equilibration": True,
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
    input_dict = {**input_dict_template_1d, **raw.get("params", {})}
    input_flags = {**input_flags_template_1d, **raw.get("flags", {})}
    return input_dict, input_flags


def default_config():
    """Return copies of the default 1D input dictionary and flags."""
    return dict(input_dict_template_1d), dict(input_flags_template_1d)


def resolve_nn0(input_dict, input_flags):
    """Return configured or table-derived initial neutral density [cm^-3]."""
    nn0 = input_dict.get("nn0")
    if nn0 is not None:
        return nn0
    return lookup_nn0(input_dict["S_gp"], twin=input_flags["TwinCathode"])
