import tomllib

from cablp.vars._nn_table import lookup_nn0


def initial_condition_defaults():
    return {
        "gas_type": "He",
        "ne0": 1e10,
        "nn0": None,
        "Tn_fit": 0.1,
        "Te0": 0.1,
        "Ti0": 0.1,
        "u0": 0.0,
    }


def geometry_defaults():
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
    }


def floor_defaults():
    return {
        "ne_floor": 1e8,
        "nn_floor": 1e8,
        "Te_floor": 0.1,
        "Ti_floor": 0.1,
    }


def neutral_source_defaults():
    return {
        "S_gp": 8000,
        "Twin_S_gp": 8000,
        "gas_puff_mode": "decay_after_breakdown",
        "S_gp_decay_target": 0.0,
        "Twin_S_gp_decay_target": 0.0,
        "tau_gp_after_breakdown": None,
        "tau_gp_decay_factor": 1.0,
        "tau_gp_pulse_duration": 0.0,
        "tau_gp_decay_duration": 1e-3,
        "S_pump_L": 4000,
        "S_pump_R": 4000,
        "gas_puff_enabled": True,
        "pump_enabled": True,
        "gas_puff_valves": 2,
    }


def timing_defaults():
    return {
        "tau_prebreakdown": 0.05,
        "tau_breakdown": 0.0,
        "tau_discharge": 20e-3,
        "tau_afterglow": 5e-3,
        "tau_cycle": 3.0,
        "cycles": 1,
        "phase_transition_mode": "scheduled",
        "I_prebreakdown": 100.0,
        "I_breakdown": 1000.0,
    }


def output_defaults():
    return {
        "dt_save": 1e-5,
        "t_save_start": 0.0,
        "max_output_steps": 0,
    }


def model_mode_defaults():
    return {
        "front_flux_model": "sonic_relaxation",
        "D_amb_model": "cs_dz",
        "end_mode": "collector",
        "cathode_model": "disabled",
        "Te_birth_ionization": "local",
        "Ti_birth_ionization": "floor",
        "neutral_exchange_model": "molecular_flow",
    }


def fudge_factor_defaults():
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
        "alpha_isat": 0.6065306597126334,
        "source_surface_area_scale": 2.0,
        "end_surface_area_scale": 1.0,
    }


def cathode_defaults():
    return {
        "V_bank": 100.0,
        "T_s": 1973.15,
        "phi_wf": 3.0,
        "C_R": 29.0,
        "R_comp": 0.004,
        "eta": 0.358,
        "L_cath": 50.0,
        "R_cath": 18.0,
    }


def physics_fit_defaults():
    return {
        "ln_lambda_min": 1.0,
        "Tn_K": 300.0,
        "neutral_exchange_coeff_cm3_s": 1.0e5,
        "neutral_clausing_scale": 1.0,
    }


def timestep_defaults():
    return {
        "cfl": 0.4,
        "density_dt_fraction": 0.25,
        "neutral_dt_fraction": 0.25,
        "heat_dt_fraction": 0.25,
        "dt_min": 1e-12,
        "dt_max": 1e-6,
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
    "heat_conduction": True,
    "implicit_heat_conduction": False,
    "front_flux": True,
    "source_surface_loss": True,
    "end_surface_loss": True,
    "cathode_coupling": False,
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
