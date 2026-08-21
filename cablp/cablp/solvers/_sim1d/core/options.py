"""Per-subsystem option bundles for :class:`LAPDSim1D`.

Every bundle here is RUN-CONSTANT: it reads the resolved configuration (and a
handful of quantities the solver derives from it once) and returns the keyword
set one physics subsystem is called with for the whole run.  They used to be
``*_kwargs`` methods on the solver, rebuilt from ``input_dict`` on every RHS
evaluation; :func:`build_solver_options` builds them ONCE at construction, so
the config surface is read in one place at one time instead of being
re-interpreted thousands of times per step.

This is a regrouping, not a change: the same keys carry the same values from
the same defaults.  The solver's ``*_kwargs`` accessors survive as thin
readers -- several verification drivers call them -- and each hands back a
fresh ``dict`` exactly as before, so a caller may still mutate what it gets.

Bundles that depend on the STEP rather than the run stay on the solver:
``_neutral_source_kwargs`` (phase switches and the gas-puff waveform at
``time``), ``_circuit_timestep_kwargs`` (the loop state and a per-step device
relation), and the ``_tracer_*`` builders (derived views, two of them keyed on
the live state and cathode solve).
"""

from dataclasses import dataclass

import numpy as np

from cablp.vars._cons import ev_to_erg, kb_cgs


def collision_operator_kwargs(input_dict, flags, *, gas_type):
    """Return the rate bundle every ion-neutral collision channel shares.

    The moment-closed operator, the CX decoupling correction, and the hot
    channel must all read ONE gas, ONE reference neutral temperature, and
    ONE drag scale, or their split stops being a split.
    """
    drag_enabled = bool(flags.get("ion_neutral_drag", True))
    return {
        "gas_type": gas_type,
        "Tn_eV": float(input_dict.get("Tn_K", 300.0))
        * kb_cgs
        / ev_to_erg,
        "b_ion_neutral_drag": (
            float(input_dict.get("b_ion_neutral_drag", 1.0))
            if drag_enabled
            else 0.0
        ),
    }


def energy_exchange_kwargs(input_dict):
    return {
        "b_Qie": float(input_dict.get("b_Qie", 1.0)),
    }


def surface_loss_kwargs(input_dict):
    # The resolved boundary terms read only ``alpha_isat`` and
    # ``b_surface_loss``. The former per-face source/end enables and area
    # scales were A13 no-ops (never consumed) and are DEPRECATED 0D artifacts
    # (R3.3): the resolved geometry measures the Bohm I_sat to each electrode
    # face directly. ``validate_r1_configuration_presence`` warns on their
    # non-default use.
    return {
        "alpha_isat": float(input_dict.get("alpha_isat", np.exp(-0.5))),
        "end_mode": input_dict.get("end_mode", "collector"),
        "b_surface_loss": float(input_dict.get("b_surface_loss", 1.0)),
    }


def ion_neutral_drag_kwargs(input_dict, flags, *, gas_type):
    drag_enabled = bool(flags.get("ion_neutral_drag", True))
    return {
        "gas_type": gas_type,
        "b_ion_neutral_drag": (
            float(input_dict.get("b_ion_neutral_drag", 1.0))
            if drag_enabled
            else 0.0
        ),
        "cx_only": bool(flags.get("ion_neutral_drag_cx_only", False)),
    }


def slip_closure_kwargs(input_dict, *, geometry):
    """Extra kwargs for the drag/frictional-heating slip closure."""
    return {
        "drag_model": str(
            input_dict.get("ion_neutral_drag_model", "constant")
        ),
        "b_slip_entrainment": float(
            input_dict.get("b_slip_entrainment", 1.0)
        ),
        "Rm_cm": geometry.Rm_cm,
        "Tn_fit": float(input_dict.get("Tn_fit", 0.1)),
    }


def electron_cooling_kwargs(input_dict, flags, *, gas_type, I_ion):
    return {
        "gas_type": gas_type,
        "I_ion": I_ion,
        "b_ioniz": float(input_dict.get("b_ioniz", 1.0)),
        "b_rec_rad": float(input_dict.get("b_rec_rad", 1.0)),
        "b_rec_3b": float(input_dict.get("b_rec_3b", 1.0)),
        # b_ionization_energy_cost removed as a config knob (R5 stance flip):
        # must be 1 for conservative energy booking, and the on/off is the
        # ionization_energy_cost flag. Hardwired 1.0.
        "b_ionization_energy_cost": 1.0,
        "b_Qei": float(input_dict.get("b_Qei", 1.0)),
        "b_Qen": float(input_dict.get("b_Qen", 1.0)),
        "b_Qei_Te_exp": float(input_dict.get("b_Qei_Te_exp", 0.0)),
        "b_Qen_Te_exp": float(input_dict.get("b_Qen_Te_exp", 0.0)),
        "b_Q_Te_ref_eV": float(input_dict.get("b_Q_Te_ref_eV", 5.0)),
        "atomic_rate_model": str(
            input_dict.get("atomic_rate_model", "adas")
        ),
        "ionization_energy_cost": bool(
            flags.get("ionization_energy_cost", True)
        ),
        "icool_recomb": bool(flags.get("icool_recomb", False)),
        # A18/R5.3: the low-Te extension defines ONE consistent atomic
        # package -- the electron-cooling prb1 honors it just like the
        # particle-rate acd. Default off => golden bit-exact.
        "adas_low_te_extension": bool(
            input_dict.get("adas_low_te_extension", False)
        ),
    }


def ion_charge_exchange_kwargs(input_dict, flags, *, gas_type):
    return {
        "gas_type": gas_type,
        "Tn_fit": float(input_dict.get("Tn_fit", 0.1)),
        "b_Qcx": float(input_dict.get("b_Qcx", 1.0)),
        "cx": bool(flags.get("cx", True)),
    }


def heat_conduction_kwargs(
    input_dict,
    flags,
    *,
    electron_heat_flux_limit,
    heat_flux_limiter_f,
    heat_flux_limiter_exponent,
):
    return {
        "b_epara": float(input_dict.get("b_epara", 1.0)),
        "b_ipara": float(input_dict.get("b_ipara", 1.0)),
        "heat_conduction": bool(flags.get("heat_conduction", True)),
        "electron_heat_flux_limit": electron_heat_flux_limit,
        "heat_flux_limiter_f": heat_flux_limiter_f,
        "heat_flux_limiter_exponent": heat_flux_limiter_exponent,
    }


def neutral_energy_timestep_kwargs(
    input_dict,
    flags,
    *,
    gas_type,
    neutral_energy,
    neutral_energy_alpha,
    neutral_energy_wall_Tn_eV,
    neutral_energy_wall_rate,
):
    """Return the bundle the En relaxation bound reads, or None.

    ``None`` where no ``En`` field exists, which withdraws the candidate.
    The values are the ones the collision operator and the wall sink
    actually apply, so the bound describes the applied rate rather than a
    nominal one.
    """
    if not neutral_energy:
        return None
    drag_enabled = bool(flags.get("ion_neutral_drag", True))
    return {
        "gas_type": gas_type,
        "Tn_eV": float(input_dict.get("Tn_K", 300.0))
        * kb_cgs
        / ev_to_erg,
        "b_ion_neutral_drag": (
            float(input_dict.get("b_ion_neutral_drag", 1.0))
            if drag_enabled
            else 0.0
        ),
        "alpha_E": neutral_energy_alpha,
        "Tn_fit": neutral_energy_wall_Tn_eV,
        "wall_rate_1_s": neutral_energy_wall_rate,
    }


def reaction_kwargs(input_dict, *, gas_type, I_ion, wind_column_factor):
    return {
        "gas_type": gas_type,
        "I_ion": I_ion,
        "b_ioniz": float(input_dict.get("b_ioniz", 1.0)),
        "b_rec_rad": float(input_dict.get("b_rec_rad", 1.0)),
        "b_rec_3b": float(input_dict.get("b_rec_3b", 1.0)),
        "atomic_rate_model": str(
            input_dict.get("atomic_rate_model", "adas")
        ),
        "adas_low_te_extension": bool(
            input_dict.get("adas_low_te_extension", False)
        ),
        "Te_birth_ionization": input_dict.get(
            "Te_birth_ionization", "local"
        ),
        "Ti_birth_ionization": input_dict.get(
            "Ti_birth_ionization", "floor"
        ),
        "ionization_birth_energy_model": str(
            input_dict.get("ionization_birth_energy_model", "legacy")
        ),
        "wind_column_factor": wind_column_factor,
    }


@dataclass(frozen=True)
class SolverOptions:
    """The run-constant keyword bundles, resolved once at construction."""

    collision_operator: dict
    energy_exchange: dict
    surface_loss: dict
    ion_neutral_drag: dict
    slip_closure: dict
    electron_cooling: dict
    ion_charge_exchange: dict
    heat_conduction: dict
    neutral_energy_timestep: dict | None
    reaction: dict


def build_solver_options(
    input_dict,
    flags,
    *,
    geometry,
    gas_type,
    I_ion,
    electron_heat_flux_limit,
    heat_flux_limiter_f,
    heat_flux_limiter_exponent,
    neutral_energy,
    neutral_energy_alpha,
    neutral_energy_wall_Tn_eV,
    neutral_energy_wall_rate,
    wind_column_factor,
):
    """Resolve every run-constant subsystem bundle in one pass.

    Called from ``LAPDSim1D.__init__`` once all of its keyword arguments have
    themselves been resolved -- which is why it takes them explicitly rather
    than reaching into a half-built solver.
    """
    return SolverOptions(
        collision_operator=collision_operator_kwargs(
            input_dict, flags, gas_type=gas_type
        ),
        energy_exchange=energy_exchange_kwargs(input_dict),
        surface_loss=surface_loss_kwargs(input_dict),
        ion_neutral_drag=ion_neutral_drag_kwargs(
            input_dict, flags, gas_type=gas_type
        ),
        slip_closure=slip_closure_kwargs(input_dict, geometry=geometry),
        electron_cooling=electron_cooling_kwargs(
            input_dict, flags, gas_type=gas_type, I_ion=I_ion
        ),
        ion_charge_exchange=ion_charge_exchange_kwargs(
            input_dict, flags, gas_type=gas_type
        ),
        heat_conduction=heat_conduction_kwargs(
            input_dict,
            flags,
            electron_heat_flux_limit=electron_heat_flux_limit,
            heat_flux_limiter_f=heat_flux_limiter_f,
            heat_flux_limiter_exponent=heat_flux_limiter_exponent,
        ),
        neutral_energy_timestep=neutral_energy_timestep_kwargs(
            input_dict,
            flags,
            gas_type=gas_type,
            neutral_energy=neutral_energy,
            neutral_energy_alpha=neutral_energy_alpha,
            neutral_energy_wall_Tn_eV=neutral_energy_wall_Tn_eV,
            neutral_energy_wall_rate=neutral_energy_wall_rate,
        ),
        reaction=reaction_kwargs(
            input_dict,
            gas_type=gas_type,
            I_ion=I_ion,
            wind_column_factor=wind_column_factor,
        ),
    )
