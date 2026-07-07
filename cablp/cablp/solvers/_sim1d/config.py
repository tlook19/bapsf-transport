import tomllib

from cablp.vars._nn_table import lookup_nn0


input_dict_template_1d = {
    "gas_type": "He",
    "ne0": 1e10,
    "nn0": None,
    "Tn_fit": 0.1,
    "Te0": 0.1,
    "Ti0": 0.1,
    "u0": 0.0,
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
    "ne_floor": 1e8,
    "nn_floor": 1e8,
    "Te_floor": 0.1,
    "Ti_floor": 0.1,
    "S_gp": 8000,
    "Twin_S_gp": 8000,
    "dt_save": 1e-5,
    "t_save_start": 0.0,
    "max_output_steps": 0,
    "front_flux_model": "sonic_relaxation",
    "alpha_front": 1.0,
    "D_amb_model": "cs_dz",
    "D_amb": 0.0,
    "b_pressure_work_elec": 1.0,
    "b_pressure_work_ions": 1.0,
    "Tn_K": 300.0,
    "neutral_exchange_model": "molecular_flow",
    "neutral_exchange_coeff_cm3_s": 1.0e5,
    "neutral_clausing_scale": 1.0,
    "cfl": 0.4,
    "density_dt_fraction": 0.25,
    "neutral_dt_fraction": 0.25,
    "dt_min": 1e-12,
    "dt_max": 1e-6,
}


input_flags_template_1d = {
    "Plasma": True,
    "TwinCathode": False,
    "heat_conduction": True,
    "front_flux": True,
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
