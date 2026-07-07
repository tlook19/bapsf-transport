from dataclasses import dataclass

import numpy as np

from .energy import (
    electron_cooling_rhs,
    electron_ion_exchange_rhs,
    ion_charge_exchange_rhs,
)
from .flux import ion_sound_speed, plasma_flux_rhs
from .neutrals import neutral_exchange_rhs, neutral_source_sink_rhs
from .reactions import reaction_rhs
from .state import derive_state


@dataclass(frozen=True)
class TimestepDiagnostics:
    dt: float
    dt_plasma_cfl: float
    dt_front_density: float
    dt_neutral_exchange: float
    dt_neutral_sources: float
    dt_reactions: float
    dt_energy_exchange: float
    dt_electron_cooling: float
    dt_ion_charge_exchange: float
    dt_max: float
    active_constraint: str


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
    cfl=0.4,
    density_dt_fraction=0.25,
    neutral_dt_fraction=0.25,
    dt_min=1e-12,
    dt_max=1e-6,
    include_front=True,
    alpha_front=1.0,
):
    """Return a bounded explicit timestep and diagnostics."""
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
        ),
        "neutral_exchange": neutral_exchange_timestep(
            state=state,
            geometry=geometry,
            neutral_exchange_coeff_cm3_s=neutral_exchange_coeff_cm3_s,
            neutral_dt_fraction=neutral_dt_fraction,
        ),
        "neutral_sources": neutral_source_timestep(
            state=state,
            geometry=geometry,
            neutral_source_kwargs=neutral_source_kwargs,
            neutral_dt_fraction=neutral_dt_fraction,
        ),
        "reactions": reaction_timestep(
            state=state,
            floors=floors,
            ion_mass_g=ion_mass_g,
            geometry=geometry,
            reaction_kwargs=reaction_kwargs,
            density_dt_fraction=density_dt_fraction,
        ),
        "energy_exchange": energy_exchange_timestep(
            state=state,
            floors=floors,
            ion_mass_g=ion_mass_g,
            mu=mu,
            energy_exchange_kwargs=energy_exchange_kwargs,
            density_dt_fraction=density_dt_fraction,
        ),
        "electron_cooling": electron_cooling_timestep(
            state=state,
            floors=floors,
            ion_mass_g=ion_mass_g,
            electron_cooling_kwargs=electron_cooling_kwargs,
            density_dt_fraction=density_dt_fraction,
        ),
        "ion_charge_exchange": ion_charge_exchange_timestep(
            state=state,
            floors=floors,
            ion_mass_g=ion_mass_g,
            ion_charge_exchange_kwargs=ion_charge_exchange_kwargs,
            density_dt_fraction=density_dt_fraction,
        ),
        "dt_max": float(dt_max),
    }
    active_constraint, raw_dt = min(dt_candidates.items(), key=lambda item: item[1])
    dt = min(max(raw_dt, dt_min), dt_max)
    if dt == dt_min and raw_dt < dt_min:
        active_constraint = "dt_min"
    return TimestepDiagnostics(
        dt=float(dt),
        dt_plasma_cfl=float(dt_candidates["plasma_cfl"]),
        dt_front_density=float(dt_candidates["front_density"]),
        dt_neutral_exchange=float(dt_candidates["neutral_exchange"]),
        dt_neutral_sources=float(dt_candidates["neutral_sources"]),
        dt_reactions=float(dt_candidates["reactions"]),
        dt_energy_exchange=float(dt_candidates["energy_exchange"]),
        dt_electron_cooling=float(dt_candidates["electron_cooling"]),
        dt_ion_charge_exchange=float(dt_candidates["ion_charge_exchange"]),
        dt_max=float(dt_max),
        active_constraint=active_constraint,
    )


def plasma_cfl_timestep(state, floors, ion_mass_g, mu, geometry, cfl=0.4):
    """Return the plasma wave CFL timestep [s]."""
    if cfl <= 0.0:
        raise ValueError(f"cfl must be positive (got {cfl})")
    derived = derive_state(state, floors=floors, ion_mass_g=ion_mass_g)
    cs = ion_sound_speed(derived.Te, mu)
    face_speed = 0.5 * (
        np.abs(derived.u[:-1])
        + np.abs(derived.u[1:])
        + cs[:-1]
        + cs[1:]
    )
    return _distance_timestep(geometry.center_distance_cm, face_speed, cfl)


def front_density_timestep(
    state,
    floors,
    ion_mass_g,
    mu,
    geometry,
    density_dt_fraction=0.25,
    include_front=True,
    alpha_front=1.0,
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
    )
    rhs_without_front = plasma_flux_rhs(
        state=state,
        floors=floors,
        ion_mass_g=ion_mass_g,
        mu=mu,
        geometry=geometry,
        include_front=False,
        alpha_front=alpha_front,
    )
    dn_front = rhs_with_front.n - rhs_without_front.n
    return _fractional_timestep(state.n, dn_front, density_dt_fraction, floors["n"])


def neutral_exchange_timestep(
    state,
    geometry,
    neutral_exchange_coeff_cm3_s,
    neutral_dt_fraction=0.25,
):
    """Return a fractional neutral-density timestep for pair exchange."""
    if neutral_dt_fraction <= 0.0:
        raise ValueError(
            f"neutral_dt_fraction must be positive (got {neutral_dt_fraction})"
        )
    rhs = neutral_exchange_rhs(
        state=state,
        geometry=geometry,
        exchange_coeff_cm3_s=neutral_exchange_coeff_cm3_s,
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
):
    """Return a fractional density timestep for local plasma reactions."""
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
    return min(
        _fractional_timestep(state.n, rhs.n, density_dt_fraction, floors["n"]),
        _fractional_timestep(state.nn, rhs.nn, density_dt_fraction, 0.0),
    )


def energy_exchange_timestep(
    state,
    floors,
    ion_mass_g,
    mu,
    energy_exchange_kwargs=None,
    density_dt_fraction=0.25,
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
        _fractional_timestep(state.Ee, rhs.Ee, density_dt_fraction, 0.0),
        _fractional_timestep(state.Ei, rhs.Ei, density_dt_fraction, 0.0),
    )


def electron_cooling_timestep(
    state,
    floors,
    ion_mass_g,
    electron_cooling_kwargs=None,
    density_dt_fraction=0.25,
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
    return _fractional_timestep(state.Ee, rhs.Ee, density_dt_fraction, 0.0)


def ion_charge_exchange_timestep(
    state,
    floors,
    ion_mass_g,
    ion_charge_exchange_kwargs=None,
    density_dt_fraction=0.25,
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
    return _fractional_timestep(state.Ei, rhs.Ei, density_dt_fraction, 0.0)


def _distance_timestep(distance, speed, fraction):
    active = speed > 0.0
    if not np.any(active):
        return np.inf
    return float(fraction * np.min(distance[active] / speed[active]))


def _fractional_timestep(values, rates, fraction, floor):
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
