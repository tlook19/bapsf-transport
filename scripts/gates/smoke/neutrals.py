"""Smoke cases: neutral state, gas puff, fill, equilibration and the neutral
closures.
"""

import argparse
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np

from cablp.constants import I_ion, ev_to_erg, kb_cgs, m_He_cgs, m_p_cgs
from cablp.solvers._sim1d import LAPDSim1D, default_config, load_result_hdf5
from cablp.solvers._sim1d.core.geometry import pump_cell_indices
from cablp.solvers._sim1d.core.integrator import ssprk2_step
from cablp.solvers._sim1d.core.state import (
    ConservativeState1D,
    STATE_NAMES_1D,
    apply_state_floors,
    conservative_from_primitives,
    derive_state,
    pack_state,
    unpack_state,
)
from cablp.solvers._sim1d.core.timestep import neutral_wind_timestep
from cablp.solvers._sim1d.physics.energy import (
    electron_cooling_rhs_terms,
    ion_charge_exchange_rhs,
)
from cablp.solvers._sim1d.physics.neutrals import (
    GAS_PUFF_DIAGNOSTIC_FIELDS,
    _effective_pump_speed,
    gas_puff_rate_profile,
    neutral_source_sink_rhs,
    neutral_thermal_speed,
    neutral_wind_advection_rhs,
    neutral_zone_exchange_conductance,
    neutral_zone_volumes,
    puff_rate,
    pump_rate,
    two_zone_knudsen_coefficients,
)
from cablp.solvers._sim1d.physics.reactions import reaction_rhs_terms
from cablp.solvers._sim1d.physics.sources import (
    IONIZATION_BIRTH_DEFICIT_DIAGNOSTIC_FIELDS,
    add_state_rhs,
    cathode_jet_backscatter_speed,
    neutral_energy_volume_ratio,
    neutral_momentum_wall_rhs,
    neutral_temperature_eV,
    neutral_wind_velocity,
    velocity_divergence,
)

from ._harness import (
    _base_config,
    _base_sim,
    _case,
    _pin_pre_r2a_neutral_stance,
    _resolved_cathode_flags,
)


def _zone_particle_rate(rhs, geometry):
    """Return the plasma-plus-neutral particle rate [1/s] on the two-zone books.

    ``n`` is booked on the plasma volume, ``nn`` on the COLUMN volume and
    ``nn_a`` (when the term carries one) on the annulus volume.
    """
    V_col, V_ann = neutral_zone_volumes(geometry)
    terms = (
        np.asarray(rhs.n, dtype=float) * geometry.plasma_volume_cm3
        + np.asarray(rhs.nn, dtype=float) * V_col
    )
    if getattr(rhs, "nn_a", None) is not None:
        terms = terms + np.asarray(rhs.nn_a, dtype=float) * V_ann
    return math.fsum(terms.tolist())


# --------------------------------------------------------------------
# gas-puff-diagnostics-and-fluid-operators
# --------------------------------------------------------------------
@_case(
    "gas-puff-diagnostics-and-fluid-operators",
    historical_stance=True,
    provides=("hot_ion_cx_state", "nn_ramp_state", "ramp_state"),
)
def _case_gas_puff_diagnostics_and_fluid_operators(
    _warnings, short_phase_params, dt_default, source_rhs
):
    # --- saved effective S_gp(t) waveform (gas_puff_diagnostics) ------------
    # The recorded waveform must be the APPLIED one, not the configured level:
    # the square envelope shapes it and the phase gate shuts it off in the
    # afterglow. The puff is not isolated here:
    # the inventory closure below is a FULL budget over every booked nn row, so
    # the fixture runs the anode at its natural eta rather than arranging the
    # channel quiet.
    params, flags = _base_config()
    sim, snapshot = _base_sim()
    geom = snapshot.geometry
    state = snapshot.state
    puffdiag_params = dict(short_phase_params)
    puffdiag_params["dt_save"] = 0.0
    # The square valve's closing tail keeps the puff switch open through the
    # afterglow, so the window must reach post_afterglow for the gate to shut
    # inside it.
    puffdiag_params["tau_afterglow"] = 1.0e-10
    # The square edges at fixture scale: the hardware timings are ~0.5 ms,
    # which on this ns window would leave the valve all but shut and the puff
    # far below the channels it is meant to dominate.
    puffdiag_params["gas_puff_rise_center_s"] = 0.0
    puffdiag_params["gas_puff_rise_width_s"] = 1.0e-10
    puffdiag_params["gas_puff_close_lag_s"] = 0.0
    puffdiag_params["pump_enabled"] = False
    # nn0 well below the delivered fuel, so the inventory difference below is
    # not swamped by float64 cancellation against a large standing fill.
    puffdiag_params["nn0"] = 1.0e5
    puffdiag_params["b_surface_loss"] = 0.0
    puffdiag_flags = dict(flags)
    puffdiag_params["initial_neutral_state"] = "fill"
    puffdiag_sim = LAPDSim1D(puffdiag_params, puffdiag_flags)
    puffdiag_geom = puffdiag_sim.get_initial_snapshot().geometry
    puffdiag_result = puffdiag_sim.run(t_end=8.0e-10, dt=1.0e-10)
    puffdiag = puffdiag_result.gas_puff_diagnostics
    assert set(puffdiag) == set(GAS_PUFF_DIAGNOSTIC_FIELDS), sorted(puffdiag)
    for _puffdiag_values in puffdiag.values():
        assert _puffdiag_values.shape == puffdiag_result.time.shape
        assert np.all(np.isfinite(_puffdiag_values))
    puffdiag_gate = np.asarray(
        puffdiag_result.phase_gas_puff_enabled, dtype=float
    ) > 0.0
    assert puffdiag_gate.any() and not puffdiag_gate.all(), puffdiag_gate
    # On: a real rate that has been shaped BELOW the configured level by the
    # end of the discharge. Off: identically zero, not the configured level.
    assert np.all(puffdiag["S_gp_sccm"][puffdiag_gate] > 0.0)
    assert np.all(puffdiag["puff_particles_per_s"][puffdiag_gate] > 0.0)
    assert puffdiag["S_gp_sccm"][puffdiag_gate][-1] < puffdiag_params["S_gp"]
    for _puffdiag_values in puffdiag.values():
        assert np.all(_puffdiag_values[~puffdiag_gate] == 0.0)
    # No twin cathode here, so the twin entry stays zero throughout.
    assert np.all(puffdiag["Twin_S_gp_sccm"] == 0.0)
    # The recorded rate IS the applied puff row: with pumping off the
    # neutral_sources term carries nothing else.
    # Two-zone books: nn on the column volume, nn_a on the annulus volume.
    puffdiag_Vc, puffdiag_Va = neutral_zone_volumes(puffdiag_geom)

    def _puffdiag_particles(fields):
        total = np.asarray(fields["nn"], dtype=float) @ puffdiag_Vc
        if "nn_a" in fields:
            total = total + np.asarray(fields["nn_a"], dtype=float) @ puffdiag_Va
        return total

    puffdiag_row = _puffdiag_particles(
        puffdiag_result.rhs_terms["neutral_sources"]
    )
    assert np.allclose(
        puffdiag_row, puffdiag["puff_particles_per_s"], rtol=1e-12, atol=0.0
    ), (puffdiag_row, puffdiag["puff_particles_per_s"])
    # ... and the neutral inventory closes against the FULL ledger of nn rows,
    # with the physics ON. Every term the run books that carries an nn row is
    # summed -- the sum is taken over the saved rhs_terms themselves, so no
    # channel can be silently omitted by hand-listing. Same provenance class as
    # the puff row above: the run's own saved diagnostics, never a re-derived
    # rate. dt_save = 0 saves every step, so the time integral is over the full
    # step sequence rather than a subsample.
    #
    # Rows this fixture makes ACTIVE, as a fraction of the inventory change:
    #   neutral_sources         ~ +1.0        the puff itself
    #   anode_collection        ~ +7.3e-5     mesh recycling at the natural eta
    #   recombination_rad_loss  ~ +3.2e-9     neutrals returned by recombination
    #   neutral_exchange        ~ -1.8e-37    inventory-conserving by construction
    #   ionization_birth        ~ -9.9e-55    plasma is cold here, so fuel burnt is nil
    # Every other booked term either exposes no nn row or is identically zero.
    # The last two are formally nonzero but numerically inert -- they sit far
    # below the closure's own roundoff floor, so the negative control below can
    # only demonstrate sensitivity to the three rows that carry weight.
    puffdiag_booked = {
        _term_name: np.trapezoid(
            _puffdiag_particles(_term_fields),
            puffdiag_result.time,
        )
        for _term_name, _term_fields in puffdiag_result.rhs_terms.items()
        if "nn" in _term_fields
    }
    puffdiag_N_n = (
        np.asarray(puffdiag_result.nn, dtype=float) @ puffdiag_Vc
        + np.asarray(puffdiag_result.nn_a, dtype=float) @ puffdiag_Va
    )
    puffdiag_dN_n = puffdiag_N_n[-1] - puffdiag_N_n[0]
    puffdiag_budget = sum(puffdiag_booked.values())
    # SSPRK2 on the state-INDEPENDENT puff row is the explicit trapezoid, so
    # that row closes exactly; the residual is set by the state-DEPENDENT
    # recombination row, whose stage-vs-saved-state gap is O(dt^2) on a row
    # that is itself only 3.2e-9 of the total. Measured 2.2e-12 -- the same
    # roundoff class as the puff-only tie this replaced, but a strictly
    # stronger statement, since it ties every channel rather than one.
    assert np.isclose(
        puffdiag_budget, puffdiag_dN_n, rtol=1e-11, atol=0.0
    ), (puffdiag_budget, puffdiag_dN_n, puffdiag_booked)
    # The rows that can actually move the budget at this tolerance, pinned so a
    # newly activated neutral channel cannot slip into the sum unnoticed. The
    # inert rows above are 25+ orders below this threshold, so it is not a
    # knife edge.
    puffdiag_carrying = {
        _term_name
        for _term_name, _term_integral in puffdiag_booked.items()
        if abs(_term_integral) > 1e-12 * abs(puffdiag_dN_n)
    }
    assert puffdiag_carrying == {
        "neutral_sources",
        "anode_collection",
        "recombination_rad_loss",
    }, puffdiag_carrying
    # NEGATIVE CONTROL: the budget must be sensitive to every row it claims to
    # carry, or the closure would be tying fewer channels than it advertises.
    # Each delta is chosen just above that row's detectability threshold,
    # rtol * |dN_n| / |row integral|; perturbing the row must break the closure.
    for _neg_name, _neg_delta in (
        ("neutral_sources", 1.0e-9),
        ("anode_collection", 1.0e-5),
        ("recombination_rad_loss", 1.0e-1),
    ):
        assert not np.isclose(
            puffdiag_budget + _neg_delta * puffdiag_booked[_neg_name],
            puffdiag_dN_n,
            rtol=1e-11,
            atol=0.0,
        ), (_neg_name, _neg_delta)

    assert np.allclose(source_rhs.n, 0.0)
    assert np.allclose(source_rhs.M, 0.0)
    assert np.allclose(source_rhs.Ee, 0.0)
    assert np.allclose(source_rhs.Ei, 0.0)

    disabled_params = dict(params)
    disabled_params["gas_puff_enabled"] = False
    disabled_params["pump_enabled"] = False
    disabled_sim = LAPDSim1D(disabled_params, flags)
    disabled_source = disabled_sim.neutral_source_sink_rhs()
    for values in (
        disabled_source.n,
        disabled_source.nn,
        disabled_source.M,
        disabled_source.Ee,
        disabled_source.Ei,
    ):
        assert np.allclose(values, 0.0, atol=1e-20)

    reaction_rhs = sim.reaction_rhs()
    for values in (
        reaction_rhs.n,
        reaction_rhs.nn,
        reaction_rhs.M,
        reaction_rhs.Ee,
        reaction_rhs.Ei,
    ):
        assert np.all(np.isfinite(values))
    reaction_Vc, _reaction_Va = neutral_zone_volumes(geom)
    reaction_inventory_scale = np.sum(
        np.abs(reaction_rhs.n * geom.plasma_volume_cm3)
        + np.abs(reaction_rhs.nn * reaction_Vc)
    )
    reaction_inventory_tol = 1e-12 * reaction_inventory_scale
    assert np.isclose(
        _zone_particle_rate(reaction_rhs, geom),
        0.0,
        atol=reaction_inventory_tol,
    )
    reaction_terms = sim.reaction_rhs_terms()
    assert set(reaction_terms) == {
        "ionization_birth",
        "recombination_rad_loss",
        "recombination_3b_loss",
    }
    reaction_term_sum = np.zeros_like(pack_state(reaction_rhs))
    for term in reaction_terms.values():
        for field_name in STATE_NAMES_1D:
            assert np.all(np.isfinite(getattr(term, field_name)))
        assert np.isclose(
            _zone_particle_rate(term, geom),
            0.0,
            atol=reaction_inventory_tol,
        )
        reaction_term_sum = reaction_term_sum + pack_state(term)
    assert np.allclose(reaction_term_sum, pack_state(reaction_rhs))
    recomb_state = conservative_from_primitives(
        n=np.full(geom.cells, params["ne0"]),
        nn=state.nn,
        nn_a=state.nn_a,
        u=np.full(geom.cells, 1.0e5),
        Te=np.full(geom.cells, 1.0),
        Ti=np.full(geom.cells, 1.0),
        ion_mass_g=sim.ion_mass_g,
    )
    recomb_terms = sim.reaction_rhs_terms(state=recomb_state)
    for _recomb_name in ("recombination_rad_loss", "recombination_3b_loss"):
        _recomb_term = recomb_terms[_recomb_name]
        assert np.all(_recomb_term.n <= 0.0)
        assert np.all(_recomb_term.nn >= 0.0)
        assert np.all(_recomb_term.M <= 0.0)
        assert np.all(_recomb_term.Ee <= 0.0)
        assert np.all(_recomb_term.Ei <= 0.0)
    assert np.all(recomb_terms["recombination_rad_loss"].n < 0.0)

    density_ramp = np.linspace(2.0, 1.0, geom.cells) * params["ne0"]
    ramp_state = conservative_from_primitives(
        n=density_ramp,
        nn=state.nn,
        nn_a=state.nn_a,
        u=np.zeros(geom.cells),
        Te=np.full(geom.cells, params["Te0"]),
        Ti=np.full(geom.cells, params["Ti0"]),
        ion_mass_g=sim.ion_mass_g,
    )
    ramp_rhs = sim.plasma_flux_rhs(y=pack_state(ramp_state))
    for values in (ramp_rhs.n, ramp_rhs.nn, ramp_rhs.M, ramp_rhs.Ee, ramp_rhs.Ei):
        assert np.all(np.isfinite(values))
    ramp_flux_terms = sim.plasma_flux_rhs_terms(state=ramp_state)
    assert set(ramp_flux_terms) == {"plasma_advective_flux"}
    ramp_flux_sum = np.zeros_like(pack_state(ramp_rhs))
    for term in ramp_flux_terms.values():
        for field_name in STATE_NAMES_1D:
            assert np.all(np.isfinite(getattr(term, field_name)))
        ramp_flux_sum = ramp_flux_sum + pack_state(term)
    assert np.allclose(ramp_flux_sum, pack_state(ramp_rhs))

    nn_ramp_state = conservative_from_primitives(
        n=state.n,
        nn=np.linspace(2.0, 1.0, geom.cells) * state.nn[0],
        nn_a=np.linspace(2.0, 1.0, geom.cells) * state.nn[0],
        u=np.zeros(geom.cells),
        Te=np.full(geom.cells, params["Te0"]),
        Ti=np.full(geom.cells, params["Ti0"]),
        ion_mass_g=sim.ion_mass_g,
    )
    nn_ramp_rhs = sim.neutral_exchange_rhs(state=nn_ramp_state)
    assert nn_ramp_rhs.nn[0] < 0.0
    assert nn_ramp_rhs.nn[-1] > 0.0
    nn_ramp_Vc, nn_ramp_Va = neutral_zone_volumes(geom)
    inventory_terms = np.concatenate(
        [nn_ramp_rhs.nn * nn_ramp_Vc, nn_ramp_rhs.nn_a * nn_ramp_Va]
    )
    inventory_tol = 1e-12 * np.sum(np.abs(inventory_terms))
    assert np.isclose(
        _zone_particle_rate(nn_ramp_rhs, geom), 0.0, atol=inventory_tol
    )
    assert np.allclose(nn_ramp_rhs.n, 0.0)
    assert np.allclose(nn_ramp_rhs.M, 0.0)
    assert np.allclose(nn_ramp_rhs.Ee, 0.0)
    assert np.allclose(nn_ramp_rhs.Ei, 0.0)

    nn_ramp_dt = sim.suggest_timestep(y=pack_state(nn_ramp_state))
    assert np.isfinite(nn_ramp_dt.dt_neutral_exchange)
    assert nn_ramp_dt.dt_neutral_exchange < dt_default.dt_neutral_exchange

    expanding_state = conservative_from_primitives(
        n=np.full(geom.cells, params["ne0"]),
        nn=state.nn,
        nn_a=state.nn_a,
        u=np.linspace(-1.0e4, 1.0e4, geom.cells),
        Te=np.full(geom.cells, params["Te0"]),
        Ti=np.full(geom.cells, params["Ti0"]),
        ion_mass_g=sim.ion_mass_g,
    )
    expanding_div_u = velocity_divergence(
        expanding_state, sim.floors, sim.ion_mass_g, geom
    )
    expanding_pressure = sim.pressure_work_rhs(state=expanding_state)
    assert np.all(expanding_div_u[2:-2] > 0.0)
    assert np.all(expanding_pressure.Ee[2:-2] < 0.0)
    assert np.all(expanding_pressure.Ei[2:-2] < 0.0)

    compressing_state = conservative_from_primitives(
        n=np.full(geom.cells, params["ne0"]),
        nn=state.nn,
        nn_a=state.nn_a,
        u=np.linspace(1.0e4, -1.0e4, geom.cells),
        Te=np.full(geom.cells, params["Te0"]),
        Ti=np.full(geom.cells, params["Ti0"]),
        ion_mass_g=sim.ion_mass_g,
    )
    compressing_div_u = velocity_divergence(
        compressing_state, sim.floors, sim.ion_mass_g, geom
    )
    compressing_pressure = sim.pressure_work_rhs(state=compressing_state)
    assert np.all(compressing_div_u[2:-2] < 0.0)
    assert np.all(compressing_pressure.Ee[2:-2] > 0.0)
    assert np.all(compressing_pressure.Ei[2:-2] > 0.0)

    hot_e_state = conservative_from_primitives(
        n=np.full(geom.cells, params["ne0"]),
        nn=state.nn,
        nn_a=state.nn_a,
        u=np.zeros(geom.cells),
        Te=np.full(geom.cells, 2.0),
        Ti=np.full(geom.cells, 0.5),
        ion_mass_g=sim.ion_mass_g,
    )
    hot_e_exchange = sim.energy_exchange_rhs(state=hot_e_state)
    assert np.all(hot_e_exchange.Ee < 0.0)
    assert np.all(hot_e_exchange.Ei > 0.0)
    assert np.allclose(hot_e_exchange.Ee + hot_e_exchange.Ei, 0.0)
    hot_e_dt = sim.suggest_timestep(y=pack_state(hot_e_state))
    assert np.isfinite(hot_e_dt.dt_energy_exchange)

    hot_i_state = conservative_from_primitives(
        n=np.full(geom.cells, params["ne0"]),
        nn=state.nn,
        nn_a=state.nn_a,
        u=np.zeros(geom.cells),
        Te=np.full(geom.cells, 0.5),
        Ti=np.full(geom.cells, 2.0),
        ion_mass_g=sim.ion_mass_g,
    )
    hot_i_exchange = sim.energy_exchange_rhs(state=hot_i_state)
    assert np.all(hot_i_exchange.Ee > 0.0)
    assert np.all(hot_i_exchange.Ei < 0.0)
    assert np.allclose(hot_i_exchange.Ee + hot_i_exchange.Ei, 0.0)

    equal_temp_state = conservative_from_primitives(
        n=np.full(geom.cells, params["ne0"]),
        nn=state.nn,
        nn_a=state.nn_a,
        u=np.zeros(geom.cells),
        Te=np.full(geom.cells, 0.5),
        Ti=np.full(geom.cells, 0.5),
        ion_mass_g=sim.ion_mass_g,
    )
    equal_temp_exchange = sim.energy_exchange_rhs(state=equal_temp_state)
    assert np.allclose(equal_temp_exchange.Ee, 0.0, atol=1e-30)
    assert np.allclose(equal_temp_exchange.Ei, 0.0, atol=1e-30)
    cooling_state = conservative_from_primitives(
        n=np.full(geom.cells, 1.0e12),
        nn=np.full(geom.cells, 1.0e12),
        nn_a=np.full(geom.cells, 1.0e12),
        u=np.zeros(geom.cells),
        Te=np.full(geom.cells, 10.0),
        Ti=np.full(geom.cells, 1.0),
        ion_mass_g=sim.ion_mass_g,
    )
    cooling_rhs = sim.electron_cooling_rhs(state=cooling_state)
    cooling_terms = sim.electron_cooling_rhs_terms(state=cooling_state)
    assert set(cooling_terms) == {
        "ionization_energy_cost",
        "electron_ion_cooling",
        "electron_neutral_cooling",
    }
    cooling_term_sum = np.zeros_like(pack_state(cooling_rhs))
    for term in cooling_terms.values():
        cooling_term_sum = cooling_term_sum + pack_state(term)
    assert np.allclose(cooling_term_sum, pack_state(cooling_rhs))
    assert np.any(cooling_terms["ionization_energy_cost"].Ee < 0.0)
    assert np.any(cooling_terms["electron_ion_cooling"].Ee < 0.0)
    assert np.any(cooling_terms["electron_neutral_cooling"].Ee < 0.0)
    assert np.all(cooling_rhs.Ee < 0.0)
    assert np.allclose(cooling_rhs.n, 0.0)
    assert np.allclose(cooling_rhs.nn, 0.0)
    assert np.allclose(cooling_rhs.M, 0.0)
    assert np.allclose(cooling_rhs.Ei, 0.0)
    cooling_dt = sim.suggest_timestep(y=pack_state(cooling_state))
    assert np.isfinite(cooling_dt.dt_electron_cooling)

    # Every cooling channel is a strict electron-energy sink.
    assert np.all(cooling_rhs.Ee < 0.0)

    # b_ionization_energy_cost survives as a function-level kwarg (it is not a
    # config key): zeroing it empties its OWN term and leaves the two
    # radiative terms untouched.
    costless_terms = electron_cooling_rhs_terms(
        state=cooling_state,
        floors=sim.floors,
        ion_mass_g=sim.ion_mass_g,
        gas_type=params["gas_type"],
        I_ion=sim.I_ion,
        b_ionization_energy_cost=0.0,
        atomic_rate_model=params["atomic_rate_model"],
        ionization_energy_cost=True,
    )
    assert np.allclose(costless_terms["ionization_energy_cost"].Ee, 0.0)
    assert np.all(cooling_terms["ionization_energy_cost"].Ee < 0.0)
    for _cool_name in ("electron_ion_cooling", "electron_neutral_cooling"):
        assert np.array_equal(
            costless_terms[_cool_name].Ee, cooling_terms[_cool_name].Ee
        )

    hot_ion_cx_state = conservative_from_primitives(
        n=np.full(geom.cells, 1.0e12),
        nn=np.full(geom.cells, 1.0e12),
        nn_a=np.full(geom.cells, 1.0e12),
        u=np.zeros(geom.cells),
        Te=np.full(geom.cells, 1.0),
        Ti=np.full(geom.cells, 10.0),
        ion_mass_g=sim.ion_mass_g,
    )
    # The standalone CX cooling function is no longer a saved term (the
    # moment-closed collision operator carries CX cooling); it still sizes
    # the ion charge-exchange timestep bound, so its sign and its bound are
    # checked here.
    hot_ion_cx = ion_charge_exchange_rhs(
        state=hot_ion_cx_state,
        floors=sim.floors,
        ion_mass_g=sim.ion_mass_g,
        gas_type=params["gas_type"],
        Tn_fit=params["Tn_fit"],
    )
    assert np.all(hot_ion_cx.Ei < 0.0)
    assert np.allclose(hot_ion_cx.n, 0.0)
    assert np.allclose(hot_ion_cx.nn, 0.0)
    assert np.allclose(hot_ion_cx.M, 0.0)
    assert np.allclose(hot_ion_cx.Ee, 0.0)
    hot_ion_cx_dt = sim.suggest_timestep(
        y=pack_state(
            ConservativeState1D(
                n=hot_ion_cx_state.n,
                nn=hot_ion_cx_state.nn,
                M=hot_ion_cx_state.M,
                Ee=hot_ion_cx_state.Ee,
                Ei=hot_ion_cx_state.Ei,
                nn_a=hot_ion_cx_state.nn.copy(),
            )
        )
    )
    assert np.isfinite(hot_ion_cx_dt.dt_ion_charge_exchange)

    warm_neutral_cx = ion_charge_exchange_rhs(
        state=hot_ion_cx_state,
        floors=sim.floors,
        ion_mass_g=sim.ion_mass_g,
        gas_type=params["gas_type"],
        Tn_fit=20.0,
    )
    assert np.all(warm_neutral_cx.Ei > 0.0)
    return locals()


# --------------------------------------------------------------------
# equilibration-puff-duty
# --------------------------------------------------------------------
@_case(
    "equilibration-puff-duty",
    historical_stance=True,
)
def _case_equilibration_puff_duty(
    neutral_phase_run_flags, neutral_phase_run_params
):
    # --- the equilibration delivers its CONFIGURED puff duty -----------------
    # tau_cycle / tau_discharge / dt chosen so a step lands a hair BELOW the
    # puff-off instant (t=6e-10 is 1 ulp short of cycle_start + tau_discharge).
    # The phase-boundary schedule used to DROP that boundary inside the run
    # loop's time_tol while the untolerated modulo in _phase_info still read
    # "puff", so the puff ran one whole extra step: this exact case delivered
    # 4.5e-10 s of puff against a configured 2.0e-10 s (+125%). Both readers now
    # share _equilibration_cycle_position, so the delivered ON-time is exact.
    duty_params = dict(neutral_phase_run_params)
    duty_params["tau_cycle"] = 5.0e-10
    duty_params["tau_discharge"] = 1.0e-10
    duty_params["cycles"] = 2
    duty_params["dt_save"] = 0.0
    duty_sim = LAPDSim1D(duty_params, dict(neutral_phase_run_flags))
    duty_result = duty_sim.run(t_end=1.0e-9, dt=2.5e-10)
    duty_times = np.asarray(duty_result.time, dtype=float)
    duty_phases = np.asarray(duty_result.phase, dtype=str)
    assert list(duty_phases) == [
        "equilibrium_puff",
        "equilibrium_off",
        "equilibrium_off",
        "equilibrium_puff",
        "equilibrium_off",
        "equilibrium_off",
        "equilibrium_puff",
    ], list(duty_phases)
    duty_on = float(
        np.sum(np.diff(duty_times)[duty_phases[:-1] == "equilibrium_puff"])
    )
    assert np.isclose(duty_on, 2.0 * 1.0e-10, rtol=1e-12), duty_on
    # Same assertion where the period DOES divide the step: 1e-10 windows on a
    # 5e-10 cycle stepped at exactly 1e-10 must not lose or gain a step either.
    duty_div_params = dict(duty_params)
    duty_div_sim = LAPDSim1D(duty_div_params, dict(neutral_phase_run_flags))
    duty_div_result = duty_div_sim.run(t_end=1.0e-9, dt=1.0e-10)
    duty_div_times = np.asarray(duty_div_result.time, dtype=float)
    duty_div_phases = np.asarray(duty_div_result.phase, dtype=str)
    duty_div_on = float(
        np.sum(np.diff(duty_div_times)[duty_div_phases[:-1] == "equilibrium_puff"])
    )
    assert np.isclose(duty_div_on, 2.0 * 1.0e-10, rtol=1e-12), duty_div_on


# --------------------------------------------------------------------
# equilibration-puff-width
# --------------------------------------------------------------------
@_case(
    "equilibration-puff-width",
    historical_stance=True,
)
def _case_equilibration_puff_width(
    equilibration_flags, neutral_phase_params, neutral_phase_run_flags,
    neutral_phase_run_params, nn_ramp_state, ramp_state
):
    # --- measured equilibration puff width (equilibration_gas_puff_on_s) -----
    # Default None == the historical tau_discharge-derived window, BIT-exact
    # through the real equilibration path (start_simulation -> the inner sim).
    # NB built on neutral_phase_params, NOT the no_source_* family: the puff
    # has to be ENABLED for the window to be observable in the seed at all.
    params, flags = _base_config()
    sim, snapshot = _base_sim()
    geom = snapshot.geometry
    puffw_base_params = dict(neutral_phase_params)
    # The equilibration-only route the equilibration case above measured
    # through: run the accumulation, return its result, launch nothing.
    puffw_base_params["initial_neutral_state"] = "equilibrate_only"
    puffw_base_params["neutral_equilibration_cycles"] = 2
    puffw_base_params["neutral_equilibration_dt"] = 1.0e-10

    def _puffw_seed(puff_on, drop=False):
        puffw_params = dict(puffw_base_params)
        if drop:
            puffw_params.pop("equilibration_gas_puff_on_s", None)
        else:
            puffw_params["equilibration_gas_puff_on_s"] = puff_on
        puffw_sim = LAPDSim1D(puffw_params, dict(equilibration_flags))
        puffw_sim.start_simulation(dt=1.0e-10)
        return np.asarray(puffw_sim.get_results().nn[-1], dtype=float)

    puffw_none_nn = _puffw_seed(None)
    assert np.array_equal(_puffw_seed(None, drop=True), puffw_none_nn)
    # Setting it explicitly to tau_discharge must reproduce the fallback too.
    assert np.array_equal(
        _puffw_seed(puffw_base_params["tau_discharge"]), puffw_none_nn
    )
    # ... and a HALVED window must measurably starve the equilibration.
    puffw_half_nn = _puffw_seed(1.0e-10)
    assert np.mean(puffw_half_nn) < np.mean(puffw_none_nn), (
        float(np.mean(puffw_half_nn)), float(np.mean(puffw_none_nn))
    )
    # The window itself moved: the phase flips at the new width, and the
    # recorded phase event names the key that closed it.
    puffw_phase_params = dict(neutral_phase_run_params)
    puffw_phase_params["equilibration_gas_puff_on_s"] = 1.0e-10
    puffw_phase_sim = LAPDSim1D(
        puffw_phase_params, dict(neutral_phase_run_flags)
    )
    assert puffw_phase_sim.phase_at_time(0.5e-10) == "equilibrium_puff"
    assert puffw_phase_sim.phase_at_time(1.0e-10) == "equilibrium_off"
    assert np.isclose(puffw_phase_sim.next_phase_boundary_after(0.0), 1.0e-10)
    puffw_phase_result = puffw_phase_sim.run(t_end=4.0e-10, dt=1.0e-10)
    assert list(puffw_phase_result.phase) == [
        "equilibrium_puff",
        "equilibrium_off",
        "equilibrium_off",
        "equilibrium_off",
        "equilibrium_off",
    ], list(puffw_phase_result.phase)
    assert list(puffw_phase_result.phase_events["reason"]) == [
        "initial",
        "equilibration_gas_puff_on_s",
    ]
    # It is NOT inert to the neutral-seed signature: setting it must re-key.
    from cablp.solvers._sim1d.core.neutral_seed_cache import (
        neutral_seed_signature,
    )

    assert neutral_seed_signature(
        {**puffw_base_params, "equilibration_gas_puff_on_s": 1.0e-10},
        equilibration_flags,
    ) != neutral_seed_signature(
        {**puffw_base_params, "equilibration_gas_puff_on_s": None},
        equilibration_flags,
    )
    # Loud ValueError on a nonsense window, at CONSTRUCTION time.
    for bad_puff_on in (0.0, -1.0e-10, 1.0e-9, "twenty-five"):
        bad_puffw_params = dict(puffw_base_params)
        bad_puffw_params["equilibration_gas_puff_on_s"] = bad_puff_on
        try:
            LAPDSim1D(bad_puffw_params, dict(equilibration_flags))
        except ValueError as exc:
            assert "equilibration_gas_puff_on_s" in str(exc), str(exc)
        else:
            raise AssertionError(
                f"equilibration_gas_puff_on_s={bad_puff_on!r} did not raise"
            )
    ramp_y0 = pack_state(ramp_state)
    ramp_y1 = ssprk2_step(
        y0=ramp_y0,
        dt=1e-10,
        rhs_func=sim.rhs,
        floor_func=sim.floor_state_vector,
    )
    ramp_y1 = sim.floor_state_vector(ramp_y1)
    ramp_state_1 = unpack_state(ramp_y1, geom.cells)
    ramp_derived_1 = derive_state(ramp_state_1, sim.floors, sim.ion_mass_g)
    for values in (
        ramp_state_1.n,
        ramp_state_1.nn,
        ramp_state_1.M,
        ramp_state_1.Ee,
        ramp_state_1.Ei,
        ramp_derived_1.Te,
        ramp_derived_1.Ti,
    ):
        assert np.all(np.isfinite(values))
    for values in (
        ramp_state_1.n,
        ramp_state_1.nn,
        ramp_state_1.Ee,
        ramp_state_1.Ei,
        ramp_derived_1.Te,
        ramp_derived_1.Ti,
    ):
        assert np.all(values >= 0.0)
    ramp_after = sim.plasma_flux_rhs(y=ramp_y1)
    for values in (
        ramp_after.n,
        ramp_after.nn,
        ramp_after.M,
        ramp_after.Ee,
        ramp_after.Ei,
    ):
        assert np.all(np.isfinite(values))

    nn_ramp_y0 = pack_state(nn_ramp_state)
    nn_ramp_y1 = ssprk2_step(
        y0=nn_ramp_y0,
        dt=1e-10,
        rhs_func=sim.rhs,
        floor_func=sim.floor_state_vector,
    )
    nn_ramp_state_1 = unpack_state(sim.floor_state_vector(nn_ramp_y1), geom.cells)
    assert np.all(np.isfinite(nn_ramp_state_1.nn))
    assert np.all(nn_ramp_state_1.nn >= params["nn_floor"])


# --------------------------------------------------------------------
# ion-neutral-closure-knobs
# --------------------------------------------------------------------
@_case(
    "ion-neutral-closure-knobs",
    provides=(
        "knob_Rm", "knob_floors", "knob_mass", "knob_n", "knob_state",
    ),
)
def _case_ion_neutral_closure_knobs():
    # --- Ion-neutral closure knobs: a three-cell reference state, handed to
    # the neutral-momentum cases below.
    knob_mass = 4.0 * m_p_cgs
    knob_floors = {"n": 1e6, "nn": 1e8, "Te": 0.1, "Ti": 0.1}
    knob_n = np.array([1e10, 1e12, 1e13])
    knob_state = conservative_from_primitives(
        n=knob_n,
        nn=np.full(3, 1e13),
        u=np.array([2e5, -1e5, 5e4]),
        Te=np.array([6.0, 5.0, 3.0]),
        Ti=np.array([1.0, 1.0, 2.0]),
        ion_mass_g=knob_mass,
    )
    knob_Rm = np.full(3, 50.0)
    return locals()


# --------------------------------------------------------------------
# neutral-momentum-state-foundations
# --------------------------------------------------------------------
@_case(
    "neutral-momentum-state-foundations",
    provides=("mn_s5", "mn_s6"),
)
def _case_neutral_momentum_state_foundations(knob_floors, knob_mass):
    # --- Neutral-momentum state foundations (M1):
    # the optional M_n field must round-trip both packed layouts, pad-on-
    # demand for term summation, refuse to silently drop, and pass floors
    # through untouched.
    mn_s5 = conservative_from_primitives(
        np.full(4, 1e12), np.full(4, 1e13), np.zeros(4),
        np.full(4, 5.0), np.ones(4), knob_mass,
    )
    assert mn_s5.M_n is None and pack_state(mn_s5).size == 20
    mn_s6 = conservative_from_primitives(
        np.full(4, 1e12), np.full(4, 1e13), np.zeros(4),
        np.full(4, 5.0), np.ones(4), knob_mass, un=np.full(4, 1.0e4),
    )
    assert mn_s6.M_n is not None and pack_state(mn_s6).size == 24
    assert unpack_state(pack_state(mn_s5), 4).M_n is None
    mn_rt = unpack_state(pack_state(mn_s6), 4)
    assert mn_rt.M_n is not None and np.all(mn_rt.M_n == mn_s6.M_n)
    padded = pack_state(mn_s5, neutral_momentum=True)
    assert padded.size == 24 and np.all(padded[20:] == 0.0)
    try:
        pack_state(mn_s6, neutral_momentum=False)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError dropping a present M_n")
    mn_fl = apply_state_floors(mn_s6, knob_floors, knob_mass)
    assert mn_fl.M_n is not None and np.all(mn_fl.M_n == mn_s6.M_n)
    mn_sum = add_state_rhs(mn_s5, mn_s6)
    assert mn_sum.M_n is not None and np.all(mn_sum.M_n == mn_s6.M_n)
    assert add_state_rhs(mn_s5, mn_s5).M_n is None
    return locals()


# --------------------------------------------------------------------
# two-zone-neutral-state-foundations
# --------------------------------------------------------------------
@_case("two-zone-neutral-state-foundations")
def _case_two_zone_neutral_state_foundations(
    knob_floors, knob_mass, mn_s5, mn_s6
):
    # --- Two-zone neutral state foundations (M1):
    # the optional nn_a field must round-trip its packed layouts, resolve
    # the 6-field width ambiguity by declared hints (bare 6-field keeps its
    # historical M_n meaning), pad-on-demand, refuse to silently drop, and
    # take the nn floor.
    tz_nn_a = np.full(4, 3.0e12)
    tz_s6 = conservative_from_primitives(
        np.full(4, 1e12), np.full(4, 1e13), np.zeros(4),
        np.full(4, 5.0), np.ones(4), knob_mass, nn_a=tz_nn_a,
    )
    assert tz_s6.nn_a is not None and tz_s6.M_n is None
    tz_packed = pack_state(tz_s6)
    assert tz_packed.size == 24
    # Bare 6-field inference keeps the historical M_n reading...
    tz_bare = unpack_state(tz_packed, 4)
    assert tz_bare.M_n is not None and tz_bare.nn_a is None
    # ...and the declared hint recovers the two-zone layout exactly.
    tz_rt = unpack_state(tz_packed, 4, neutral_two_zone=True)
    assert tz_rt.M_n is None and np.all(tz_rt.nn_a == tz_s6.nn_a)
    tz_rt2 = unpack_state(
        tz_packed, 4, neutral_momentum=False, neutral_two_zone=True
    )
    assert tz_rt2.M_n is None and np.all(tz_rt2.nn_a == tz_s6.nn_a)
    # 7-field (both optionals) round-trips without hints: unambiguous.
    tz_s7 = conservative_from_primitives(
        np.full(4, 1e12), np.full(4, 1e13), np.zeros(4),
        np.full(4, 5.0), np.ones(4), knob_mass,
        un=np.full(4, 1.0e4), nn_a=tz_nn_a,
    )
    tz_p7 = pack_state(tz_s7)
    assert tz_p7.size == 28
    tz_rt7 = unpack_state(tz_p7, 4)
    assert np.all(tz_rt7.M_n == tz_s7.M_n)
    assert np.all(tz_rt7.nn_a == tz_s7.nn_a)
    # Field order is (..., M_n, nn_a): the last row of the 7-field pack is
    # the annulus density.
    assert np.all(tz_p7[24:] == tz_nn_a)
    # Pad-on-demand for term summation, in both flag combinations.
    tz_pad = pack_state(mn_s5, neutral_two_zone=True)
    assert tz_pad.size == 24 and np.all(tz_pad[20:] == 0.0)
    tz_pad7 = pack_state(mn_s6, neutral_two_zone=True)
    assert tz_pad7.size == 28 and np.all(tz_pad7[24:] == 0.0)
    # Refuse to silently drop a present field.
    try:
        pack_state(tz_s6, neutral_two_zone=False)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError dropping a present nn_a")
    # A wrong declared layout is an error, not a silent misread.
    try:
        unpack_state(pack_state(mn_s5), 4, neutral_two_zone=True)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for impossible layout")
    # nn_a is a density: it takes the nn floor; M_n still passes through.
    tz_low = conservative_from_primitives(
        np.full(4, 1e12), np.full(4, 1e13), np.zeros(4),
        np.full(4, 5.0), np.ones(4), knob_mass,
        un=np.full(4, 1.0e4), nn_a=np.zeros(4),
    )
    tz_fl = apply_state_floors(tz_low, knob_floors, knob_mass)
    assert np.all(tz_fl.nn_a == knob_floors["nn"])
    assert np.all(tz_fl.M_n == tz_low.M_n)
    # Add semantics: missing-side-as-zeros, independently per optional field.
    tz_sum = add_state_rhs(mn_s6, tz_s6)
    assert np.all(tz_sum.M_n == mn_s6.M_n)
    assert np.all(tz_sum.nn_a == tz_s6.nn_a)
    assert add_state_rhs(mn_s5, mn_s5).nn_a is None


# --------------------------------------------------------------------
# neutral-momentum-sources
# --------------------------------------------------------------------
@_case(
    "neutral-momentum-sources",
    historical_stance=True,
    provides=(
        "build_geometry", "mn_flags", "mn_geom",
        "mn_plasma_flags", "mn_plasma_params",
        "mn_plasma_sim", "mn_reactions", "mn_state", "mn_u", "mn_vbar",
    ),
)
def _case_neutral_momentum_sources(
    expected_rhs_terms, knob_Rm,
    knob_floors, knob_mass, knob_n, knob_state, run_params
):
    # --- Neutral-momentum sources (M2): with M_n on the state, the
    # reactions become species-conserving momentum exchanges and the wall and
    # pump are the only named sinks.
    params, flags = _base_config()
    mn_geom = SimpleNamespace(
        plasma_volume_cm3=np.array([450.0, 900.0, 1800.0]) * 1.0e3,
        neutral_volume_cm3=np.full(3, 5.0e6),
    )
    mn_geom.volume_ratio = (
        mn_geom.plasma_volume_cm3 / mn_geom.neutral_volume_cm3
    )
    knob_u = np.array([2e5, -1e5, 5e4])
    mn_state = conservative_from_primitives(
        n=knob_n,
        nn=np.full(3, 1e13),
        u=knob_u,
        Te=np.array([6.0, 5.0, 3.0]),
        Ti=np.array([1.0, 1.0, 2.0]),
        ion_mass_g=knob_mass,
        un=0.3 * knob_u,
    )
    assert np.allclose(
        neutral_wind_velocity(mn_state, knob_floors, knob_mass),
        0.3 * knob_u,
        rtol=1e-14,
    )
    assert np.all(
        neutral_wind_velocity(knob_state, knob_floors, knob_mass) == 0.0
    )

    # Wall sink: -M_n / tau_wall on the momentum field only; inert without M_n.
    mn_wall = neutral_momentum_wall_rhs(
        state=mn_state,
        floors=knob_floors,
        ion_mass_g=knob_mass,
        Rm_cm=knob_Rm,
    )
    mn_vbar = np.sqrt(8.0 * 0.1 * ev_to_erg / (np.pi * knob_mass))
    assert np.allclose(
        mn_wall.M_n, -mn_state.M_n * mn_vbar / knob_Rm, rtol=1e-14
    )
    for mn_field in STATE_NAMES_1D:
        assert np.all(getattr(mn_wall, mn_field) == 0.0)
    assert (
        neutral_momentum_wall_rhs(
            state=knob_state,
            floors=knob_floors,
            ion_mass_g=knob_mass,
            Rm_cm=knob_Rm,
        ).M_n
        is None
    )

    # Reactions: ionization births ions drifting at u_n (taking that momentum
    # out of the wind), recombination hands the ion's momentum to the wind;
    # each term closes M*Vp + M_n*Vm exactly.
    mn_reactions = reaction_rhs_terms(
        state=mn_state,
        floors=knob_floors,
        ion_mass_g=knob_mass,
        geometry=mn_geom,
        gas_type="He",
        I_ion=I_ion,
    )
    for mn_name in (
        "ionization_birth",
        "recombination_rad_loss",
        "recombination_3b_loss",
    ):
        mn_term = mn_reactions[mn_name]
        assert mn_term.M_n is not None
        assert np.allclose(
            mn_term.M * mn_geom.plasma_volume_cm3,
            -mn_term.M_n * mn_geom.neutral_volume_cm3,
            rtol=1e-12,
        )
    assert np.allclose(
        mn_reactions["ionization_birth"].M,
        knob_mass * 0.3 * knob_u * mn_reactions["ionization_birth"].n,
        rtol=1e-12,
    )
    mn_rec = mn_reactions["recombination_rad_loss"]
    mn_u = derive_state(mn_state, knob_floors, knob_mass).u
    assert np.allclose(
        mn_rec.M, knob_mass * mn_u * mn_rec.n, rtol=1e-12
    )
    # Without M_n the reaction terms stay 5-field with zero-drift birth.
    assert reaction_rhs_terms(
        state=knob_state,
        floors=knob_floors,
        ion_mass_g=knob_mass,
        geometry=mn_geom,
        gas_type="He",
        I_ion=I_ion,
    )["ionization_birth"].M_n is None

    # Pump sink: the wind leaves with the gas at the pump cells, so the
    # pumped momentum fraction matches the pumped particle fraction there.
    from cablp.solvers._sim1d.core.geometry import build_geometry

    mn_pump_geom = build_geometry(*default_config())
    mn_pump_cells = mn_pump_geom.cells
    mn_pump_state = conservative_from_primitives(
        n=np.full(mn_pump_cells, 1e12),
        nn=np.full(mn_pump_cells, 1e13),
        u=np.zeros(mn_pump_cells),
        Te=np.full(mn_pump_cells, 5.0),
        Ti=np.full(mn_pump_cells, 1.0),
        ion_mass_g=knob_mass,
        un=np.full(mn_pump_cells, 2.0e4),
    )
    mn_pump = neutral_source_sink_rhs(
        state=mn_pump_state,
        geometry=mn_pump_geom,
        S_gp=0.0,
        Twin_S_gp=0.0,
        S_pump_L=500.0,
        S_pump_R=500.0,
        gas_puff_enabled=False,
        pump_enabled=True,
    )
    mn_pump_mask = mn_pump.nn != 0.0
    assert np.any(mn_pump_mask)
    assert np.all(mn_pump.M_n[~mn_pump_mask] == 0.0)
    assert np.allclose(
        mn_pump.M_n[mn_pump_mask],
        mn_pump.nn[mn_pump_mask]
        * mn_pump_state.M_n[mn_pump_mask]
        / mn_pump_state.nn[mn_pump_mask],
        rtol=1e-12,
    )
    # Puff-only sources add cold gas: no momentum contribution at all.
    assert np.all(
        neutral_source_sink_rhs(
            state=mn_pump_state,
            geometry=mn_pump_geom,
            S_gp=100.0,
            Twin_S_gp=0.0,
            S_pump_L=500.0,
            S_pump_R=500.0,
            gas_puff_enabled=True,
            pump_enabled=False,
            gas_puff_orifice_id_cm=3.95,
            gas_puff_orifice_length_cm=22.0,
        ).M_n
        == 0.0
    )

    # Solver plumbing: the flag builds and carries the 6-field state, the
    # wall term appears in the rhs, a run saves/loads the optional M_n and
    # u_n trajectories.
    mn_flags = dict(flags)
    mn_flags["neutral_momentum"] = True
    # The evolved neutral wind (M_n) is driven by the moment-closed
    # ion-neutral collision operator's neutral mirror row.
    mn_run_params = dict(run_params)
    mn_sim = LAPDSim1D(mn_run_params, mn_flags)
    assert mn_sim.state.M_n is not None
    mn_cells = mn_sim.geometry.cells
    assert mn_sim.get_initial_snapshot().y.size == 7 * mn_cells
    mn_rhs_terms = mn_sim.rhs_terms()
    assert set(mn_rhs_terms) == expected_rhs_terms
    assert mn_sim.rhs().size == 7 * mn_cells
    mn_result = mn_sim.run(t_end=3.0e-10, dt=1.0e-10)
    assert mn_result.M_n.shape == (4, mn_cells)
    assert mn_result.u_n.shape == (4, mn_cells)
    assert np.all(np.isfinite(mn_result.M_n))
    with tempfile.TemporaryDirectory() as mn_dir:
        mn_path = Path(mn_dir) / "mn_smoke.h5"
        mn_sim.save_result(mn_path, mn_result)
        mn_loaded = load_result_hdf5(mn_path)
        assert np.allclose(mn_loaded.M_n, mn_result.M_n)
        assert np.allclose(mn_loaded.u_n, mn_result.u_n)

    # Plasma-phase end-to-end: with a flowing plasma the collision operator
    # pumps the wind up from zero through the full step machinery (explicit
    # substep, implicit heat with the 7-field state, floors, step acceptance).
    mn_plasma_flags = dict(mn_flags)
    mn_plasma_params = dict(params)
    mn_plasma_params["initial_neutral_state"] = "fill"
    mn_plasma_params["u0"] = 5.0e4
    mn_plasma_sim = LAPDSim1D(mn_plasma_params, mn_plasma_flags)
    mn_plasma_terms = mn_plasma_sim.rhs_terms()
    assert mn_plasma_terms["ion_neutral_collision"].M_n is not None
    assert np.any(mn_plasma_terms["ion_neutral_collision"].M_n != 0.0)
    assert mn_plasma_sim.rhs().size == 7 * mn_plasma_sim.geometry.cells
    for _ in range(5):
        mn_plasma_sim.advance_one_step(dt=1.0e-9)
    mn_plasma_state = mn_plasma_sim.state
    assert mn_plasma_state.M_n is not None
    assert np.all(np.isfinite(mn_plasma_state.M_n))
    assert np.any(mn_plasma_state.M_n != 0.0)
    # The wind chases the plasma flow: same sign where it has spun up.
    mn_drive = mn_plasma_state.M_n * derive_state(
        mn_plasma_state, mn_plasma_sim.floors, mn_plasma_sim.ion_mass_g
    ).u
    assert np.all(mn_drive[mn_plasma_state.M_n != 0.0] > 0.0)
    return locals()


# --------------------------------------------------------------------
# neutral-two-zone-particle-channel
# --------------------------------------------------------------------
@_case(
    "neutral-two-zone-particle-channel",
    historical_stance=True,
    provides=(
        "p2z_Va", "p2z_Vc", "p2z_both_flags", "p2z_flags", "p2z_params",
        "p2z_sim",
    ),
)
def _case_neutral_two_zone_particle_channel(mn_plasma_flags, mn_plasma_params):
    # --- Two-zone PARTICLE channel, M2 carriage and transport
    # The solver carries the split (nn, nn_a)
    # state, runs per-zone axial Knudsen exchange plus the radial
    # column/annulus conductance, and both close inventory exactly with
    # detailed balance at equal densities.
    p2z_params = dict(mn_plasma_params)
    p2z_flags = dict(mn_plasma_flags)
    p2z_flags["neutral_momentum"] = False
    p2z_sim = LAPDSim1D(p2z_params, p2z_flags)
    p2z_state = p2z_sim.state
    assert p2z_state.nn_a is not None and p2z_state.M_n is None
    assert p2z_sim.rhs().size == 6 * p2z_sim.geometry.cells
    p2z_Vc, p2z_Va = neutral_zone_volumes(p2z_sim.geometry)
    assert np.allclose(
        p2z_Vc + p2z_Va, p2z_sim.geometry.neutral_volume_cm3, rtol=1e-13
    )
    # Conductance arithmetic against the closed forms.
    p2z_vth = neutral_thermal_speed(
        float(p2z_params.get("Tn_K", 300.0)), p2z_sim.mu
    )
    p2z_geom = p2z_sim.geometry
    p2z_mid = p2z_geom.cells // 2
    assert np.isclose(
        neutral_zone_exchange_conductance(
            p2z_geom, float(p2z_params.get("Tn_K", 300.0)), p2z_sim.mu
        )[p2z_mid],
        0.25
        * p2z_vth
        * 2.0
        * np.pi
        * p2z_geom.Rp_cm[p2z_mid]
        * p2z_geom.length_cm[p2z_mid],
        rtol=1e-13,
    )
    p2z_cc, p2z_ca = two_zone_knudsen_coefficients(
        p2z_geom, float(p2z_params.get("Tn_K", 300.0)), p2z_sim.mu
    )
    p2z_Rcol = 0.5 * (p2z_geom.Rp_cm[p2z_mid] + p2z_geom.Rp_cm[p2z_mid + 1])
    p2z_Rann = 0.5 * (
        p2z_geom.Rm_cm[p2z_mid] + p2z_geom.Rm_cm[p2z_mid + 1]
    ) - p2z_Rcol
    assert np.isclose(
        p2z_cc[p2z_mid],
        (2.0 / 3.0)
        * p2z_vth
        * p2z_Rcol
        * min(
            p2z_geom.plasma_area_cm2[p2z_mid],
            p2z_geom.plasma_area_cm2[p2z_mid + 1],
        )
        / p2z_geom.center_distance_cm[p2z_mid],
        rtol=1e-13,
    )
    p2z_ann_area = (
        p2z_geom.neutral_area_cm2 - p2z_geom.plasma_area_cm2
    )
    assert np.isclose(
        p2z_ca[p2z_mid],
        (2.0 / 3.0)
        * p2z_vth
        * p2z_Rann
        * min(p2z_ann_area[p2z_mid], p2z_ann_area[p2z_mid + 1])
        / p2z_geom.center_distance_cm[p2z_mid],
        rtol=1e-13,
    )
    # Detailed balance: the uniform initial state gives exactly zero for
    # both exchange terms.
    assert np.all(p2z_sim.neutral_zone_exchange_rhs(state=p2z_state).nn == 0.0)
    assert np.all(
        p2z_sim.neutral_zone_exchange_rhs(state=p2z_state).nn_a == 0.0
    )
    # Perturbed: exact inventory closure per term, and the net flux refills
    # the depleted column from the annulus.
    p2z_nn = p2z_state.nn.copy()
    p2z_nn[p2z_mid] *= 0.5
    p2z_pert = ConservativeState1D(
        p2z_state.n,
        p2z_nn,
        p2z_state.M,
        p2z_state.Ee,
        p2z_state.Ei,
        nn_a=p2z_state.nn_a.copy(),
    )
    p2z_zx = p2z_sim.neutral_zone_exchange_rhs(state=p2z_pert)
    # Closure is antisymmetric by construction in the scalar flow; the volume
    # re-multiplication here is not a guaranteed identity, so the honest claim
    # is closure to round-off (1 ULP at Rp = 18.415), not an exact zero.
    assert abs(
        float((p2z_zx.nn * p2z_Vc + p2z_zx.nn_a * p2z_Va).sum())
    ) <= 1e-12 * float(np.abs(p2z_zx.nn * p2z_Vc).max())
    assert p2z_zx.nn[p2z_mid] > 0.0 and p2z_zx.nn_a[p2z_mid] < 0.0
    p2z_ax = p2z_sim.neutral_exchange_rhs(state=p2z_pert)
    assert abs(float((p2z_ax.nn * p2z_Vc).sum())) <= 1e-12 * float(
        np.abs(p2z_ax.nn * p2z_Vc).max()
    )
    assert np.all(p2z_ax.nn_a == 0.0)  # annulus still uniform
    # The pre-plasma implicit step conserves inventory exactly with the
    # pump off and the puff's BE deposit accounted, and preserves the
    # uniform equilibrium.
    p2z_eq_params = dict(p2z_params)
    p2z_eq_params["pump_enabled"] = False
    p2z_eq_flags = dict(p2z_flags)
    p2z_eq_flags["Plasma"] = False
    p2z_eq_sim = LAPDSim1D(p2z_eq_params, p2z_eq_flags)
    p2z_eq_state = p2z_eq_sim.state
    p2z_eq_Vc, p2z_eq_Va = neutral_zone_volumes(p2z_eq_sim.geometry)
    p2z_dt = 1.0e-5
    p2z_next = p2z_eq_sim._implicit_neutral_step(
        dt=p2z_dt, state=p2z_eq_state, time=0.0
    )
    p2z_src = p2z_eq_sim._neutral_source_kwargs(time=0.0)
    p2z_inflow = 0.0
    if p2z_src["gas_puff_enabled"]:
        p2z_inflow = float(
            np.sum(
                gas_puff_rate_profile(
                    p2z_eq_sim.geometry,
                    p2z_src["S_gp"],
                    p2z_src["gas_puff_valves"],
                    z_cm=p2z_src["gas_puff_z_cm"],
                    orifice_id_cm=p2z_src["gas_puff_orifice_id_cm"],
                    orifice_length_cm=p2z_src["gas_puff_orifice_length_cm"],
                )
                * p2z_eq_sim.geometry.neutral_volume_cm3
            )
        )
    p2z_inv0 = float(
        (p2z_eq_state.nn * p2z_eq_Vc + p2z_eq_state.nn_a * p2z_eq_Va).sum()
    )
    p2z_inv1 = float(
        (p2z_next.nn * p2z_eq_Vc + p2z_next.nn_a * p2z_eq_Va).sum()
    )
    assert np.isclose(p2z_inv1 - p2z_inv0, p2z_dt * p2z_inflow, rtol=1e-9)
    # Plasma-phase e2e through the full step machinery, two-zone alone and
    # combined with the evolved wind (7-field state).
    for _ in range(5):
        p2z_sim.advance_one_step(dt=1.0e-9)
    p2z_after = p2z_sim.state
    assert p2z_after.nn_a is not None
    assert np.all(np.isfinite(p2z_after.nn_a))
    p2z_both_flags = dict(p2z_flags)
    p2z_both_flags["neutral_momentum"] = True
    p2z_both_sim = LAPDSim1D(p2z_params, p2z_both_flags)
    assert p2z_both_sim.rhs().size == 7 * p2z_both_sim.geometry.cells
    for _ in range(5):
        p2z_both_sim.advance_one_step(dt=1.0e-9)
    p2z_both_state = p2z_both_sim.state
    assert p2z_both_state.M_n is not None
    assert p2z_both_state.nn_a is not None
    assert np.all(np.isfinite(p2z_both_state.M_n))
    assert np.all(np.isfinite(p2z_both_state.nn_a))
    return locals()


# --------------------------------------------------------------------
# neutral-wind-advection
# --------------------------------------------------------------------
@_case("neutral-wind-advection")
def _case_neutral_wind_advection(
    knob_floors, knob_mass, knob_state, mn_plasma_sim
):
    # --- Neutral-wind advection (M3): donor-cell
    # upwind of nn and M_n by u_n on the neutral faces, closed ends for
    # particles, end-wall momentum accommodation, and a CFL guard.
    mw_cells = 5
    mw_geom = SimpleNamespace(
        cells=mw_cells,
        length_cm=np.full(mw_cells, 30.0),
        neutral_volume_cm3=np.full(mw_cells, 2.0e5),
        neutral_face_area_cm2=np.full(mw_cells + 1, 7.0e3),
    )
    mw_un = 2.0e4
    mw_nn = np.array([1e13, 3e13, 2e13, 5e13, 4e13])
    mw_state = conservative_from_primitives(
        n=np.full(mw_cells, 1e12),
        nn=mw_nn,
        u=np.zeros(mw_cells),
        Te=np.full(mw_cells, 5.0),
        Ti=np.full(mw_cells, 1.0),
        ion_mass_g=knob_mass,
        un=np.full(mw_cells, mw_un),
    )
    # Without M_n the operator is inert and 5-field.
    mw_off = neutral_wind_advection_rhs(
        state=conservative_from_primitives(
            n=np.full(mw_cells, 1e12),
            nn=mw_nn,
            u=np.zeros(mw_cells),
            Te=np.full(mw_cells, 5.0),
            Ti=np.full(mw_cells, 1.0),
            ion_mass_g=knob_mass,
        ),
        floors=knob_floors,
        ion_mass_g=knob_mass,
        geometry=mw_geom,
    )
    assert mw_off.M_n is None and np.all(mw_off.nn == 0.0)

    mw_adv = neutral_wind_advection_rhs(
        state=mw_state,
        floors=knob_floors,
        ion_mass_g=knob_mass,
        geometry=mw_geom,
    )
    # Hand-built donor-cell stencil for a uniform positive wind: each
    # internal face carries u * nn_donor * A with the left cell as donor;
    # the end faces pass no particles.
    mw_flux = mw_un * mw_nn[:-1] * mw_geom.neutral_face_area_cm2[1:-1]
    mw_expected = np.zeros(mw_cells)
    mw_expected[:-1] -= mw_flux / mw_geom.neutral_volume_cm3[:-1]
    mw_expected[1:] += mw_flux / mw_geom.neutral_volume_cm3[1:]
    assert np.allclose(mw_adv.nn, mw_expected, rtol=1e-14)
    # Particle inventory closes (to summation rounding): the ends are
    # walls, not sinks.
    mw_nn_scale = np.max(np.abs(mw_adv.nn * mw_geom.neutral_volume_cm3))
    assert (
        abs(np.sum(mw_adv.nn * mw_geom.neutral_volume_cm3))
        < 1e-12 * mw_nn_scale
    )
    # Momentum inventory loses exactly the outward end-wall accommodation.
    mw_Mn_end = mw_state.M_n[-1]
    mw_end_sink = mw_un * mw_geom.neutral_face_area_cm2[-1] * mw_Mn_end
    assert np.isclose(
        np.sum(mw_adv.M_n * mw_geom.neutral_volume_cm3),
        -mw_end_sink,
        rtol=1e-12,
    )
    # A uniform field under a uniform wind does not change in the interior
    # (pure translation); only the end cells feel the walls.
    mw_uniform = conservative_from_primitives(
        n=np.full(mw_cells, 1e12),
        nn=np.full(mw_cells, 2e13),
        u=np.zeros(mw_cells),
        Te=np.full(mw_cells, 5.0),
        Ti=np.full(mw_cells, 1.0),
        ion_mass_g=knob_mass,
        un=np.full(mw_cells, mw_un),
    )
    mw_uadv = neutral_wind_advection_rhs(
        state=mw_uniform,
        floors=knob_floors,
        ion_mass_g=knob_mass,
        geometry=mw_geom,
    )
    assert np.all(mw_uadv.nn[1:-1] == 0.0)
    assert mw_uadv.nn[0] < 0.0 and mw_uadv.nn[-1] > 0.0
    # An inward wind at both ends leaves no accommodation sink: the pure
    # flux stencil accounts for the whole momentum change.
    mw_in_un = np.array([1.0, 1.0, 0.0, -1.0, -1.0]) * mw_un
    mw_inward = conservative_from_primitives(
        n=np.full(mw_cells, 1e12),
        nn=np.full(mw_cells, 2e13),
        u=np.zeros(mw_cells),
        Te=np.full(mw_cells, 5.0),
        Ti=np.full(mw_cells, 1.0),
        ion_mass_g=knob_mass,
        un=mw_in_un,
    )
    mw_iadv = neutral_wind_advection_rhs(
        state=mw_inward,
        floors=knob_floors,
        ion_mass_g=knob_mass,
        geometry=mw_geom,
    )
    mw_Mn_scale = np.max(np.abs(mw_iadv.M_n * mw_geom.neutral_volume_cm3))
    assert (
        abs(np.sum(mw_iadv.M_n * mw_geom.neutral_volume_cm3))
        < 1e-12 * mw_Mn_scale
    )

    # CFL guard: inf without a wind or at rest, cfl*min(dz/|u_n|) otherwise,
    # and wired into the solver's suggestion once the wind has spun up.
    assert neutral_wind_timestep(
        state=knob_state,
        floors=knob_floors,
        ion_mass_g=knob_mass,
        geometry=mw_geom,
    ) == np.inf
    assert np.isclose(
        neutral_wind_timestep(
            state=mw_state,
            floors=knob_floors,
            ion_mass_g=knob_mass,
            geometry=mw_geom,
            cfl=0.4,
        ),
        0.4 * 30.0 / mw_un,
        rtol=1e-14,
    )
    mw_diag = mn_plasma_sim.suggest_timestep()
    assert np.isfinite(mw_diag.dt_neutral_wind)
    assert mw_diag.dt_neutral_wind > 0.0


# --------------------------------------------------------------------
# gas-puff-orifice-profile
# --------------------------------------------------------------------
@_case(
    "gas-puff-orifice-profile",
    provides=("build_geometry",),
)
def _case_gas_puff_orifice_profile():
    # --- Tube-beamed injection row (the puff's axial shape): the SAME row
    # the kinetic instruments launch, read by the fluid solver as its
    # deposition profile. Three things are owned here: the row the shared
    # implementation returns is bit-for-bit the row scripts/stance/puff_orifice.py
    # derives on the same inputs (one derivation, not two), it conserves the
    # total inflow exactly, and every misconfiguration raises at CONSTRUCTION.
    from cablp.solvers._sim1d.core.geometry import build_geometry

    import puff_orifice as _porf

    orf_params, orf_flags = default_config()
    orf_params["nx"] = 60
    orf_geom = build_geometry(orf_params, orf_flags)
    orf_z = float(orf_params["gas_puff_z_cm"])
    ORF_ID, ORF_LEN = 3.95, 22.0

    orf_row = gas_puff_rate_profile(
        orf_geom,
        3000.0,
        2,
        z_cm=orf_z,
        orifice_id_cm=ORF_ID,
        orifice_length_cm=ORF_LEN,
    )
    # BIT-FOR-BIT against the scripts-side derivation, at the port cell the
    # kinetic instruments index (searchsorted on the edges, clipped). The
    # comparison is on raw bytes: "close" would not catch a re-derivation.
    orf_i = int(np.searchsorted(orf_geom.z_edges_cm, orf_z) - 1)
    orf_i = min(max(orf_i, 0), int(orf_geom.Rm_cm.size) - 1)
    orf_ref_row, orf_meta = _porf.launch_row(
        orf_geom.z_edges_cm,
        pipe_id_cm=ORF_ID,
        aspect_ratio=ORF_LEN / ORF_ID,
        r_wall_cm=float(orf_geom.Rm_cm[orf_i]),
        r_edge_cm=float(orf_geom.Rp_cm[orf_i]),
        z_port_cm=orf_z,
    )
    orf_total_in = 4.171431e17 * 3000.0 * 2.0
    orf_expected = (
        orf_total_in
        * orf_ref_row
        / np.asarray(orf_geom.neutral_volume_cm3, dtype=float)
    )
    assert orf_expected.tobytes() == orf_row.tobytes()
    # exact inflow conservation, same bar as the other distributed profiles
    assert np.isclose(
        np.sum(orf_row * orf_geom.neutral_volume_cm3), orf_total_in, rtol=1e-12
    )
    assert np.all(orf_row >= 0.0)
    assert np.isclose(orf_ref_row.sum(), 1.0, rtol=1e-12)
    # the row is run-constant and memoised; a second call must return the same
    # values, not a re-derivation that could drift
    orf_again = gas_puff_rate_profile(
        orf_geom,
        3000.0,
        2,
        z_cm=orf_z,
        orifice_id_cm=ORF_ID,
        orifice_length_cm=ORF_LEN,
    )
    assert orf_again.tobytes() == orf_row.tobytes()

    def _orf_refused(label, **overrides):
        bad_params = dict(orf_params)
        bad_params.update(overrides)
        try:
            LAPDSim1D(bad_params, dict(orf_flags))
        except ValueError:
            return
        raise AssertionError(f"expected ValueError at construction: {label}")

    orf_armed = {
        "gas_puff_orifice_id_cm": ORF_ID,
        "gas_puff_orifice_length_cm": ORF_LEN,
    }
    _orf_refused(
        "orifice without the bore",
        **dict(orf_armed, gas_puff_orifice_id_cm=None),
    )
    _orf_refused(
        "orifice without the length",
        **dict(orf_armed, gas_puff_orifice_length_cm=None),
    )
    _orf_refused("non-positive bore", **dict(orf_armed, gas_puff_orifice_id_cm=-1.0))
    _orf_refused(
        "aspect ratio below 4/3",
        **dict(orf_armed, gas_puff_orifice_length_cm=4.0),
    )
    _orf_refused("port off the grid", **dict(orf_armed, gas_puff_z_cm=99999.0))
    # the derivation's own refusals, exercised directly: on this stance the
    # config gate above fires first, so these are the backstop and their
    # existence is what this block owns.
    for orf_label, orf_kwargs in (
        ("port off the grid", {"z_port_cm": 1.0e9}),
        ("column outside the wall", {"r_edge_cm": 1.0e4}),
    ):
        orf_call = {
            "r_edge_cm": float(orf_geom.Rp_cm[orf_i]),
            "z_port_cm": orf_z,
        }
        orf_call.update(orf_kwargs)
        try:
            _porf.launch_row(
                orf_geom.z_edges_cm,
                pipe_id_cm=ORF_ID,
                aspect_ratio=ORF_LEN / ORF_ID,
                r_wall_cm=float(orf_geom.Rm_cm[orf_i]),
                **orf_call,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"launch_row accepted {orf_label}")
    try:
        _porf.clausing_intensity(np.array([2.0]), ORF_LEN / ORF_ID)
    except ValueError:
        pass
    else:
        raise AssertionError("clausing_intensity accepted theta outside [0, pi/2]")
    try:
        _porf.clausing_intensity(np.array([0.1]), 1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("clausing_intensity accepted Gamma < 4/3")

    # and the armed configuration constructs and carries the keys to the puff
    # sites, so the profile is reachable end to end rather than only in the
    # builder.
    orf_sim = LAPDSim1D(dict(orf_params, **orf_armed), dict(orf_flags))
    orf_nk = orf_sim._neutral_source_kwargs(time=0.0)
    assert orf_nk["gas_puff_orifice_id_cm"] == ORF_ID
    assert orf_nk["gas_puff_orifice_length_cm"] == ORF_LEN
    return locals()


# --------------------------------------------------------------------
# directed-recycle-jets
# --------------------------------------------------------------------
@_case(
    "directed-recycle-jets",
    historical_stance=True,
)
def _case_directed_recycle_jets(knob_mass, m3_cathode_flags, m3_params):
    # --- Directed recycle jets: cathode-face
    # backscatter + effusion and anode-mesh backscatter ride the SAME terms
    # that rebirth the recycle particles, as M_n sources; the mesh
    # accommodates the wind momentum its wires intercept. Validation fails
    # fast; magnitudes must reproduce the step-1 scoping arithmetic exactly
    # from each term's own rebirthed flux (particle/momentum consistency).
    # Directed recycle jets ride on the evolved M_n; run on the simple stance
    # (jet_params derive from the simple m3_params).
    resolved_cathode_flags = _resolved_cathode_flags()
    jet_flags = dict(m3_cathode_flags)
    jet_flags["neutral_momentum"] = True
    for jet_bad_params, jet_bad_flags in (
        # M_n physics without the neutral_momentum flag
        (dict(m3_params, cathode_neutral_jet=True), resolved_cathode_flags),
        (dict(m3_params, anode_neutral_jet=True), resolved_cathode_flags),
        (dict(m3_params, neutral_mesh_accommodation=True),
         resolved_cathode_flags),
        # reflection coefficients outside [0, 1]
        (dict(m3_params, cathode_neutral_jet=True, cathode_jet_R_E=1.5),
         jet_flags),
        (dict(m3_params, anode_neutral_jet=True, anode_jet_R_N=-0.1),
         jet_flags),
        # the debit reads the jet's R_E
        (dict(m3_params, cathode_jet_surface_debit=True), jet_flags),
        # the anode jet must DECLARE which convention its R_E is read in:
        # undeclared, unknown, declared-without-the-jet, and the
        # total_reflected ordering requirement 0 < R_E <= R_N < 1.
        (dict(m3_params, anode_neutral_jet=True), jet_flags),
        (dict(m3_params, anode_neutral_jet=True,
              anode_jet_energy_convention="bogus"), jet_flags),
        (dict(m3_params, anode_jet_energy_convention="total_reflected"),
         jet_flags),
        (dict(m3_params, anode_neutral_jet=True,
              anode_jet_energy_convention="total_reflected",
              anode_jet_R_E=0.9), jet_flags),
    ):
        try:
            LAPDSim1D(jet_bad_params, jet_bad_flags)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {jet_bad_params}")
    jet_params = dict(
        m3_params,
        cathode_neutral_jet=True,
        anode_neutral_jet=True,
        anode_jet_energy_convention="total_reflected",
        neutral_mesh_accommodation=True,
    )
    jet_sim = LAPDSim1D(jet_params, jet_flags)
    jet_sim._circuit_I_loop = 800.0
    jet_solve = jet_sim.solve_cathode_boundary(update_cache=True)
    jet_res = jet_solve.beam_result.result
    jet_geom = jet_sim.geometry
    jet_roles = np.asarray(jet_geom.cell_role)
    jet_m = jet_sim.ion_mass_g
    jet_derived = derive_state(jet_sim.state, jet_sim.floors, jet_m)
    jet_kb = 1.380649e-16

    # Cathode channel: momentum only at the cathode cell, directed into the
    # column (+z at the source end), and the volume-integrated M_n source
    # equals m * v_mix * (the term's own rebirthed flux) exactly.
    jet_ba = jet_sim.characteristic_boundary_rhs(cathode_solve=jet_solve)
    jet_cath = int(np.flatnonzero(jet_roles == "cathode")[0])
    assert jet_ba.M_n is not None
    assert np.array_equal(np.flatnonzero(jet_ba.M_n), [jet_cath])
    assert jet_ba.M_n[jet_cath] > 0.0
    jet_RN = float(jet_params.get("cathode_jet_R_N", 0.5))
    jet_RE = float(jet_params.get("cathode_jet_R_E", 0.2))
    jet_Ts = float(jet_params["cathode_Ts_base_K"])
    jet_veff = np.sqrt(np.pi * jet_kb * jet_Ts / (2.0 * jet_m))
    # The incident per-ion energy is the CIRCUIT's own phi_c + Te/2 -- the
    # half-Te the presheath gave the Bohm ion plus the fall it drops through
    # -- restated here rather than read from the helper, so a change to the
    # helper has to be made twice to pass.
    jet_vback = np.sqrt(
        2.0 * jet_RE
        * max(
            max(jet_res.phi_c, 0.0) + 0.5 * jet_derived.Te[jet_cath], 0.0
        )
        * ev_to_erg / jet_m
    )
    jet_vmix = jet_RN * jet_vback + (1.0 - jet_RN) * jet_veff
    # nn is the COLUMN density (booked on V_col); M_n is the chamber-mean
    # wind (booked on the chamber volume).
    jet_Vc, _jet_Va = neutral_zone_volumes(jet_geom)
    jet_flux = jet_ba.nn[jet_cath] * jet_Vc[jet_cath]
    assert np.isclose(
        jet_ba.M_n[jet_cath] * jet_geom.neutral_volume_cm3[jet_cath],
        jet_m * jet_vmix * jet_flux,
        rtol=1e-12,
        atol=0.0,
    )

    # Anode channel: DEFERRED (R5 stance flip, 2026-07-25). Under the repaired
    # stance the anode jet M_n comes out zero at this quiescent test state. The
    # reason is NOT simply the anode sheath sign -- a negative phi_a is a
    # POSITIVE ion-sheath, which does not by itself imply zero ion current;
    # deriving whether the anode collects ions here needs the ion-sheath
    # physics. The M_n directed-jet module may be rewritten from the ES1
    # baseline findings, so its anode-channel physics assertions are
    # deferred rather than re-derived now. The deferral retires once one of
    # two things settles what the anode channel should assert: the anode
    # ion-sheath current is derived properly, or the directed-jet module is
    # rewritten. Until then this is a deliberate gap in coverage, not an
    # oversight. The flag plumbing + construction validation above, and the
    # mesh-accommodation stencil below, still run.

    # Floating (afterglow) solve: the jet rides the floating sheath drop --
    # tiny but finite, never NaN.
    jet_float = jet_sim.solve_cathode_boundary(
        floating=True, update_cache=False
    )
    jet_ba_float = jet_sim.characteristic_boundary_rhs(cathode_solve=jet_float)
    assert np.all(np.isfinite(jet_ba_float.M_n))
    assert np.all(jet_ba_float.M_n >= 0.0)

    # Presence gating: flags off -> the terms stay 5-field even with M_n on
    # the state (the golden path can never construct a jet).
    jet_off_sim = LAPDSim1D(dict(m3_params), jet_flags)
    jet_off_sim._circuit_I_loop = 800.0
    jet_off_solve = jet_off_sim.solve_cathode_boundary(update_cache=True)
    assert jet_off_sim.characteristic_boundary_rhs(
        cathode_solve=jet_off_solve
    ).M_n is None
    assert jet_off_sim.anode_collection_rhs(
        cathode_solve=jet_off_solve
    ).M_n is None

    # Mesh momentum accommodation: hand-built stencil -- the wind flowing
    # INTO the mesh loses -|u| * A_blocked / V * M_n on its own side only;
    # sign-safe (relaxes M_n toward zero) for either wind direction.
    mesh_cells = 5
    mesh_geom = SimpleNamespace(
        cells=mesh_cells,
        length_cm=np.full(mesh_cells, 30.0),
        neutral_volume_cm3=np.full(mesh_cells, 2.0e5),
        neutral_face_area_cm2=np.full(mesh_cells + 1, 7.0e3),
    )
    mesh_floors = {"n": 1e8, "nn": 1e8, "Te": 0.1, "Ti": 0.02}
    mesh_kw = dict(
        n=np.full(mesh_cells, 1e12),
        nn=np.full(mesh_cells, 2e13),
        u=np.zeros(mesh_cells),
        Te=np.full(mesh_cells, 5.0),
        Ti=np.full(mesh_cells, 1.0),
        ion_mass_g=knob_mass,
    )
    for mesh_un, mesh_hit, mesh_dry in ((2.0e4, 1, 2), (-2.0e4, 2, 1)):
        mesh_state = conservative_from_primitives(
            un=np.full(mesh_cells, mesh_un), **mesh_kw
        )
        mesh_base = neutral_wind_advection_rhs(
            state=mesh_state,
            floors=mesh_floors,
            ion_mass_g=knob_mass,
            geometry=mesh_geom,
        )
        mesh_with = neutral_wind_advection_rhs(
            state=mesh_state,
            floors=mesh_floors,
            ion_mass_g=knob_mass,
            geometry=mesh_geom,
            mesh_faces=[2],
            mesh_blocked_area_cm2=[3.0e3],
        )
        mesh_diff = mesh_with.M_n - mesh_base.M_n
        assert np.isclose(
            mesh_diff[mesh_hit],
            -abs(mesh_un) * 3.0e3 * mesh_state.M_n[mesh_hit] / 2.0e5,
            rtol=1e-13,
        )
        # The sink always relaxes M_n toward zero.
        assert mesh_diff[mesh_hit] * mesh_state.M_n[mesh_hit] < 0.0
        assert mesh_diff[mesh_dry] == 0.0
        # Particle fluxes are untouched -- accommodation is momentum-only.
        assert np.array_equal(mesh_with.nn, mesh_base.nn)
    # Solver wiring: blocked area reconstructs the full face through the
    # open fraction T = 1 - eta (Ra = None), A_blocked = A_open * (1-T)/T.
    jet_eta = float(jet_params["eta"])
    jet_T = 1.0 - jet_eta
    assert np.allclose(
        jet_sim._mesh_blocked_area_cm2,
        np.asarray(jet_geom.neutral_face_area_cm2, dtype=float)[
            np.asarray(jet_geom.anode_face_indices, dtype=int)
        ] * jet_eta / jet_T,
        rtol=1e-13,
    )

    # Surface-debit sensitivity arm: power_balance receives (1 - R_E) * P_i
    # when on; exactly 1.0 * P_i (the M5a' calibration convention) when off.
    assert jet_sim._cathode_surface_ion_retention == 1.0
    jet_debit_sim = LAPDSim1D(
        dict(jet_params, cathode_jet_surface_debit=True), jet_flags
    )
    assert np.isclose(
        jet_debit_sim._cathode_surface_ion_retention,
        1.0 - float(jet_params.get("cathode_jet_R_E", 0.2)),
        rtol=1e-13,
    )

    # Reflected-energy CONVENTION. The debit above is written in the TRIM
    # convention (R_E = total reflected energy / total incident, so the
    # surface keeps 1 - R_E), while the launch reads R_E per backscattered
    # particle and only the R_N reflected fraction carries it -- the gas
    # receives R_N*R_E of what the surface gave up. "total_reflected" reads
    # R_E the way the debit does: R_E/R_N per reflected particle, so the
    # exported energy per RECYCLED particle is exactly the R_E the debit
    # removed. "legacy" (the shipped default) keeps the historical launch.
    for jet_conv_bad in (
        # unknown selector
        dict(jet_params, cathode_jet_energy_convention="bogus"),
        # the corrected convention rescales the jet and needs one to rescale
        dict(m3_params, cathode_jet_energy_convention="total_reflected"),
        # 0 < R_E <= R_N < 1: a reflected particle cannot leave with more
        # energy than it arrived with, and neither coefficient is degenerate
        dict(jet_params, cathode_jet_energy_convention="total_reflected",
             cathode_jet_R_E=0.8, cathode_jet_R_N=0.5),
        dict(jet_params, cathode_jet_energy_convention="total_reflected",
             cathode_jet_R_E=0.0, cathode_jet_R_N=0.5),
        dict(jet_params, cathode_jet_energy_convention="total_reflected",
             cathode_jet_R_E=0.5, cathode_jet_R_N=1.0),
    ):
        try:
            LAPDSim1D(jet_conv_bad, jet_flags)
        except ValueError as exc:
            assert "cathode_jet_energy_convention" in str(exc)
        else:
            raise AssertionError(
                "expected ValueError for cathode_jet_energy_convention="
                f"{jet_conv_bad.get('cathode_jet_energy_convention')!r}"
            )
    # Both happy directions construct: the shipped default, an explicit
    # "legacy", and "total_reflected" at the interior point and at the
    # R_E == R_N boundary (fully inelastic capture of the reflected share).
    for jet_conv_ok in (
        dict(jet_params, cathode_jet_energy_convention="legacy"),
        dict(jet_params, cathode_jet_energy_convention="total_reflected"),
        dict(jet_params, cathode_jet_energy_convention="total_reflected",
             cathode_jet_R_E=0.5, cathode_jet_R_N=0.5),
    ):
        LAPDSim1D(jet_conv_ok, jet_flags)
    # An explicit "legacy" is the default's own path: the jet's M_n row is
    # byte-identical to the key being absent entirely.
    jet_legacy_sim = LAPDSim1D(
        dict(jet_params, cathode_jet_energy_convention="legacy"), jet_flags
    )
    jet_legacy_sim._circuit_I_loop = 800.0
    jet_legacy_ba = jet_legacy_sim.characteristic_boundary_rhs(
        cathode_solve=jet_legacy_sim.solve_cathode_boundary(update_cache=True)
    )
    assert np.array_equal(jet_legacy_ba.M_n, jet_ba.M_n)

    # THE CONSERVATION IDENTITY, on a short jet-armed run with the En field
    # present. Per RECYCLED particle the surface debit gives up
    # R_E*(phi_c + Ti) and, under "total_reflected", the backscatter share of
    # the jet's En term delivers exactly that -- to machine precision, on the
    # evolved state, from the term's own rebirthed flux. Under "legacy" the
    # same read returns R_N times it, which is the energy hole this convention
    # closes. NAMED RESIDUAL, and it is not closed by this identity: the
    # cathode solve debits R_E * P_cathode_i, whose ion flux is the solve's
    # own Bohm current and whose per-ion energy is (phi_c + Te/2), while the
    # jet rides the fluid boundary term's recycle flux at (phi_c + Ti).
    jet_en_flags = dict(jet_flags, neutral_energy=True)
    jet_en_ref = None
    for jet_conv, jet_conv_share in (
        ("legacy", jet_RN),
        ("total_reflected", 1.0),
    ):
        jet_en_sim = LAPDSim1D(
            dict(
                jet_params,
                cathode_jet_surface_debit=True,
                cathode_jet_energy_convention=jet_conv,
            ),
            jet_en_flags,
        )
        jet_en_sim._circuit_I_loop = 800.0
        jet_en_sim.run(t_end=3.0e-10, dt=1.0e-10)
        jet_en_state = jet_en_sim.state
        assert jet_en_state.En is not None
        jet_en_solve = jet_en_sim.solve_cathode_boundary(
            state=jet_en_state, update_cache=False
        )
        jet_en_spec = jet_en_sim._cathode_jet_spec(jet_en_solve)
        assert jet_en_spec["energy_convention"] == jet_conv
        jet_en_ba = jet_en_sim.characteristic_boundary_rhs(
            state=jet_en_state, cathode_solve=jet_en_solve
        )
        jet_en_term = jet_en_sim.cathode_jet_neutral_energy_rhs(
            state=jet_en_state,
            cathode_solve=jet_en_solve,
            recycle_nn_row=jet_en_ba.nn,
        )
        jet_en_derived = derive_state(
            jet_en_state, jet_en_sim.floors, jet_en_sim.ion_mass_g
        )
        jet_en_cath = np.asarray(jet_en_sim.geometry.cell_role) == "cathode"
        jet_en_vback = cathode_jet_backscatter_speed(
            jet_en_spec, jet_en_derived.Te, jet_en_sim.ion_mass_g
        )
        # Per-particle: what the backscattered share actually carries, and
        # what the surface debit gave up for it.
        jet_en_carried = (
            jet_RN * 0.5 * jet_en_sim.ion_mass_g * jet_en_vback**2
        )[jet_en_cath]
        jet_en_debited = (
            jet_RE
            * (
                jet_en_spec["phi_c_V"]
                + 0.5 * jet_en_derived.Te[jet_en_cath]
            )
            * ev_to_erg
        )
        assert np.all(jet_en_debited > 0.0)
        assert np.allclose(
            jet_en_carried,
            jet_conv_share * jet_en_debited,
            rtol=1e-13,
            atol=0.0,
        )
        # The debit fraction the surface balance loses IS R_E, both arms.
        assert np.isclose(
            1.0 - jet_en_sim._cathode_surface_ion_retention,
            jet_RE,
            rtol=1e-13,
        )
        # The term itself is the excess over the wall credit the generic
        # surface booking already granted, on the cathode cells alone.
        assert np.array_equal(
            np.flatnonzero(jet_en_term.En != 0.0),
            np.flatnonzero(
                jet_en_cath & (np.maximum(jet_en_ba.nn, 0.0) != 0.0)
            ),
        )
        jet_en_ref = jet_en_carried if jet_en_ref is None else jet_en_ref
    # The two conventions really do separate: total_reflected launches the
    # backscatter with 1/R_N times the legacy energy.
    assert np.all(jet_en_carried > jet_en_ref)


# --------------------------------------------------------------------
# cathode-jet-hot-carrier
# --------------------------------------------------------------------
@_case("cathode-jet-hot-carrier")
def _case_cathode_jet_hot_carrier():
    # --- The DIRECTED hot surface carrier (thread-23 phase 1). The R_N
    # backscatter share leaves the cathode as its own attenuated beam instead
    # of rebirthing cold at the cathode cell. Three things are gated here:
    # the construction refusals, the OFF path's bit-exactness, and the term's
    # own particle/energy/momentum closure with the three v1 withholdings
    # checked against the launch BY NAME. The stance-point magnitudes are
    # scripts/verify/t23c_pairwise_audit.py's job; this is the plumbing gate.
    from cablp.solvers._sim1d.core.state import NEUTRAL_ENERGY_FLOOR_T_K
    from cablp.solvers._sim1d.physics.sources import (
        cathode_jet_backscatter_speed as _hc_vback,
    )

    hc_kb = 1.380649e-16
    hc_params, hc_flags = default_config()
    hc_params["nx"] = 10
    hc_params["initial_neutral_state"] = "fill"

    # (1) Construction refusals, one per prerequisite, each naming the flag.
    for hc_bad_params, hc_bad_flags, hc_missing in (
        (
            dict(
                hc_params,
                cathode_jet_hot_carrier=True,
                cathode_neutral_jet=False,
                cathode_jet_surface_debit=False,
                cathode_jet_energy_convention="legacy",
            ),
            hc_flags,
            "cathode_neutral_jet",
        ),
        (
            dict(
                hc_params,
                cathode_jet_hot_carrier=True,
                cathode_jet_surface_debit=False,
            ),
            hc_flags,
            "cathode_jet_surface_debit",
        ),
        (
            dict(hc_params, cathode_jet_hot_carrier=True),
            dict(
                hc_flags,
                neutral_energy=False,
                neutral_hot_internal_wall=False,
            ),
            "neutral_energy",
        ),
    ):
        try:
            LAPDSim1D(hc_bad_params, dict(hc_bad_flags))
        except ValueError as exc:
            assert "cathode_jet_hot_carrier" in str(exc)
            assert hc_missing in str(exc)
        else:
            raise AssertionError(
                f"expected ValueError for a carrier missing {hc_missing}"
            )

    # (2) PRESENCE GATING AND THE OFF PATH. With the flag off the term is
    # absent from the ledger entirely, and an explicit False is byte-identical
    # to the key being absent -- state vector included, after a step.
    hc_off = LAPDSim1D(dict(hc_params), dict(hc_flags))
    hc_off._circuit_I_loop = 800.0
    assert hc_off._cathode_jet_carrier is False
    assert "cathode_jet_hot_carrier" not in hc_off.rhs_terms()
    hc_absent = dict(hc_params)
    hc_absent.pop("cathode_jet_hot_carrier")
    hc_ref = LAPDSim1D(hc_absent, dict(hc_flags))
    hc_ref._circuit_I_loop = 800.0
    for hc_sim in (hc_off, hc_ref):
        hc_sim.run(t_end=3.0e-10, dt=1.0e-10)
    assert np.array_equal(hc_off._y, hc_ref._y)
    # The boundary term's new keyword defaults to the historical booking: on a
    # carrier-ARMED solver, calling it WITHOUT the launch channel reproduces
    # the off solver's row bit for bit.
    hc_on = LAPDSim1D(
        dict(hc_params, cathode_jet_hot_carrier=True), dict(hc_flags)
    )
    hc_on._circuit_I_loop = 800.0
    hc_on.run(t_end=3.0e-10, dt=1.0e-10)
    hc_state = hc_off.state
    hc_solve = hc_off.solve_cathode_boundary(
        state=hc_state, update_cache=False
    )
    hc_bnd_off = hc_off.characteristic_boundary_rhs(
        state=hc_state, cathode_solve=hc_solve
    )
    hc_bnd_inert = hc_on.characteristic_boundary_rhs(
        state=hc_state, cathode_solve=hc_solve
    )
    assert np.array_equal(hc_bnd_off.nn, hc_bnd_inert.nn)
    assert np.array_equal(hc_bnd_off.M_n, hc_bnd_inert.M_n)

    # (3) THE CLOSURE, on the armed solver's own evolved state, with the three
    # withholdings checked against the launch by name. A sum identity alone
    # cannot catch a compensated double-book, which is why each pair is
    # compared directly.
    hc_g = hc_on.geometry
    hc_state = hc_on.state
    hc_Vp = np.asarray(hc_g.plasma_volume_cm3, dtype=float)
    hc_Vm = np.asarray(hc_g.neutral_volume_cm3, dtype=float)
    hc_Vnn = hc_Vp if hc_state.nn_a is not None else hc_Vm
    hc_VMn = hc_Vp if hc_state.M_n_a is not None else hc_Vm
    hc_Vann = np.maximum(hc_Vm - hc_Vp, 1e-300)
    hc_solve = hc_on.solve_cathode_boundary(
        state=hc_state, update_cache=False
    )
    hc_out = {}
    hc_on_bnd = hc_on.characteristic_boundary_rhs(
        state=hc_state, cathode_solve=hc_solve, carrier_out=hc_out
    )
    hc_v1_bnd = hc_on.characteristic_boundary_rhs(
        state=hc_state, cathode_solve=hc_solve
    )
    hc_launch = hc_out["launch_per_s"]
    hc_reaction = hc_on.reaction_rhs_terms(state=hc_state)
    hc_term = hc_on.cathode_jet_hot_carrier_rhs(
        state=hc_state,
        cathode_solve=hc_solve,
        launch_per_s=hc_launch,
        ionization_rate=np.asarray(
            hc_reaction["ionization_birth"].n, dtype=float
        )
        / np.maximum(
            np.asarray(hc_state.nn, dtype=float), hc_on.floors["nn"]
        ),
    )
    hc_led = hc_on._jet_carrier_diagnostics
    hc_launched = float(np.sum(hc_launch))
    assert hc_launched > 0.0
    assert np.isclose(hc_led["launch_per_s"], hc_launched, rtol=1e-13)

    # (i) the cathode cell's nn rebirth <-> the launch rate
    assert np.isclose(
        float(np.sum((hc_v1_bnd.nn - hc_on_bnd.nn) * hc_Vnn)),
        hc_launched,
        rtol=1e-10,
        atol=0.0,
    )
    # (ii) the R_N v_back share of jet_M_n <-> the launch momentum
    assert np.isclose(
        float(np.sum((hc_v1_bnd.M_n - hc_on_bnd.M_n) * hc_VMn)),
        hc_led["launch_dyn"],
        rtol=1e-10,
        atol=0.0,
    )
    # (iii) the v1 En pair (surface wall credit + jet excess) <-> the launch
    # power, with the v1 side rebuilt from the documented formula.
    hc_spec = hc_on._cathode_jet_spec(hc_solve)
    hc_der = derive_state(hc_state, hc_on.floors, hc_on.ion_mass_g)
    hc_RN = float(hc_spec["R_N"])
    hc_vback = _hc_vback(hc_spec, hc_der.Te, hc_on.ion_mass_g)
    hc_ejet = hc_RN * 0.5 * hc_on.ion_mass_g * hc_vback**2 + (
        1.0 - hc_RN
    ) * (1.5 * hc_kb * max(float(hc_spec["T_s_K"]), 0.0))
    hc_wall = 1.5 * hc_kb * NEUTRAL_ENERGY_FLOOR_T_K
    hc_cath = np.asarray(hc_g.cell_role) == "cathode"
    hc_v1_En = float(
        np.sum(
            np.where(
                hc_cath,
                np.maximum(hc_v1_bnd.nn, 0.0) * (hc_ejet - hc_wall),
                0.0,
            )
            * hc_Vnn
        )
    ) + hc_wall * float(np.sum(np.maximum(hc_v1_bnd.nn, 0.0) * hc_Vnn))
    hc_on_En = float(
        np.sum(
            hc_on.cathode_jet_neutral_energy_rhs(
                state=hc_state,
                cathode_solve=hc_solve,
                recycle_nn_row=hc_on_bnd.nn,
            ).En
            * hc_Vnn
        )
    ) + hc_wall * float(np.sum(np.maximum(hc_on_bnd.nn, 0.0) * hc_Vnn))
    assert np.isclose(
        (hc_v1_En - hc_on_En) * 1e-7,
        hc_led["launch_W"],
        rtol=1e-10,
        atol=0.0,
    )

    # Sum closure, all three conserved quantities, at machine precision.
    hc_fates = (
        hc_led["partner_exchange_per_s"]
        + hc_led["jet_ionization_per_s"]
        + hc_led["wall_leak_per_s"]
        + hc_led["end_leak_per_s"]
        + hc_led["mesh_cull_per_s"]
    )
    hc_rows = (
        float(np.sum(hc_term.n * hc_Vp))
        + float(np.sum(hc_term.nn * hc_Vnn))
        + float(np.sum(hc_term.nn_a * hc_Vann))
    )
    assert np.isclose(hc_fates, hc_launched, rtol=1e-10, atol=0.0)
    assert np.isclose(hc_rows, hc_launched, rtol=1e-10, atol=0.0)
    hc_spent_W = (
        float(np.sum(hc_term.Ei * hc_Vp)) * 1e-7
        + float(np.sum(hc_term.En * hc_Vnn)) * 1e-7
        + hc_led["wall_leak_W"]
        + hc_led["end_leak_W"]
        + hc_led["mesh_cull_W"]
    )
    assert np.isclose(hc_spent_W, hc_led["launch_W"], rtol=1e-10, atol=0.0)
    hc_spent_dyn = (
        float(np.sum(hc_term.M * hc_Vp))
        + float(np.sum(hc_term.M_n * hc_VMn))
        + hc_led["leak_dyn"]
    )
    assert np.isclose(
        hc_spent_dyn, hc_led["launch_dyn"], rtol=1e-10, atol=0.0
    )
    # Charge exchange is a SWAP: it moves no net plasma density, and every
    # deposit lands on a plasma-active cell (a masked deposit would be a
    # silent particle loss rather than the named leak the ledger reports).
    assert np.isclose(
        float(np.sum(hc_term.n * hc_Vp)),
        hc_led["jet_ionization_per_s"],
        rtol=1e-10,
        atol=0.0,
    )
    hc_live = np.asarray(hc_g.plasma_active, dtype=bool)
    for hc_row in (hc_term.n, hc_term.nn, hc_term.nn_a, hc_term.Ei):
        assert np.all(np.asarray(hc_row, dtype=float)[~hc_live] == 0.0)
    # The beam is directed AWAY from the source cathode, into the column.
    assert hc_led["launch_dyn"] > 0.0
    assert hc_led["E_fast_eV"] > 1.0
    # The anode mesh really culls: eta of the flux crossing the anode face.
    assert hc_led["mesh_cull_per_s"] > 0.0
    # The geometric escape length is the mean interior-point ray of the column
    # disc, 8 Rp/(3 pi), times <cot(theta)> = pi/2 over the Lambert launch:
    # lambda_esc = 4 Rp / 3. Pinned as a NUMBER, because the first cut of this
    # kernel compounded a chord-vs-ray error with a <cot> -> 1/<tan> swap and
    # landed a third short (advisor correction 2026-08-21).
    from cablp.solvers._sim1d.physics.jet_carrier import (
        carrier_escape_length_cm as _hc_lesc,
    )

    hc_Rp = np.asarray(hc_g.Rp_cm, dtype=float)
    hc_lam = _hc_lesc(hc_g)
    assert np.allclose(
        hc_lam[hc_Rp > 0.0],
        (4.0 / 3.0) * hc_Rp[hc_Rp > 0.0],
        rtol=1e-14,
        atol=0.0,
    )
    assert np.all(np.isinf(hc_lam[hc_Rp <= 0.0]))
    # The dt bundle OWNS the carrier: the bounded rows are the applied rows
    # (the withheld boundary plus the beam), not the v1 rows the step no
    # longer books. Off, the bundle is untouched.
    hc_bundle_on = hc_on._plasma_source_timestep_rhs(
        state=hc_state, time=hc_on.time
    )
    hc_bundle_off = hc_off._plasma_source_timestep_rhs(
        state=hc_off.state, time=hc_off.time
    )
    assert np.any(hc_bundle_on.Ei != 0.0)
    assert np.all(np.isfinite(hc_bundle_on.Ei))
    assert np.all(np.isfinite(hc_bundle_off.Ei))
    # A dt PROBE must not rewrite the ledger the last accepted evaluation
    # left, exactly as it re-solves the cathode with update_cache=False.
    hc_probe_ref = hc_on._jet_carrier_diagnostics
    hc_on._plasma_source_timestep_rhs(state=hc_state, time=hc_on.time)
    assert hc_on._jet_carrier_diagnostics is hc_probe_ref
    # The birth-convention debts are REPORTED as numbers, not left as prose:
    # neither is visible to any identity above, because both halves of every
    # pair are booked in one convention.
    for hc_debt in (
        "u_dM_partner_exchange_W",
        "u_dM_jet_ionization_W",
        "u_dM_ion_total_W",
        "u_dM_partner_neutral_W",
        "q_mix_missing_W",
        "electron_birth_convention_W",
    ):
        assert hc_debt in hc_led
        assert np.isfinite(hc_led[hc_debt])
    assert np.isclose(
        hc_led["u_dM_ion_total_W"],
        hc_led["u_dM_partner_exchange_W"] + hc_led["u_dM_jet_ionization_W"],
        rtol=1e-12,
    )
    # Q_mix is a squared magnitude summed over births: never negative, and
    # strictly positive wherever the beam deposited anything at all.
    assert hc_led["q_mix_missing_W"] > 0.0
    # The carrier is constructible on the bulk birth convention, on which
    # the electron side AGREES: the bulk books Ee_birth = 0 exactly as the
    # carrier does.
    LAPDSim1D(dict(hc_params, cathode_jet_hot_carrier=True), dict(hc_flags))


# --------------------------------------------------------------------
# square-gas-puff-waveform
# --------------------------------------------------------------------
@_case(
    "square-gas-puff-waveform",
    historical_stance=True,
)
def _case_square_gas_puff_waveform(m3_params):
    # --- Square gas-puff waveform (the measured piezo/supply behaviour):
    # erf rise anchored on circuit-on, flat at S_gp through the drive,
    # erf close after drive end with a tail into the afterglow.
    resolved_cathode_flags = _resolved_cathode_flags()
    for sq_bad in (
        {"gas_puff_rise_width_s": 0.0},
        {"gas_puff_close_lag_s": -1e-3},
    ):
        try:
            LAPDSim1D(dict(m3_params, **sq_bad), resolved_cathode_flags)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {sq_bad}")
    sq_sim = LAPDSim1D(dict(m3_params), resolved_cathode_flags)
    sq_t0 = sq_sim._plasma_phase_time_origin()
    sq_Sgp = float(sq_sim._input_dict["S_gp"])
    # Before circuit-on the envelope is (nearly) closed; well after the rise
    # it is flat at S_gp; mid-drive stays flat (no decay-to-level).
    assert sq_sim._effective_gas_puff_sccm(time=0.0)[0] < 0.1 * sq_Sgp
    assert np.isclose(
        sq_sim._effective_gas_puff_sccm(time=sq_t0 + 5e-3)[0], sq_Sgp,
        rtol=1e-6,
    )
    # Emulate a triggered breakdown to exercise the close anchor: the flow
    # still runs at S_gp late in the drive, decays through the close lag,
    # and is shut well inside the afterglow.
    sq_sim._t_prebreakdown_trigger = sq_t0 + 1.0e-3
    sq_sim._t_breakdown_trigger = sq_t0 + 1.2e-3
    sq_tau_dis = float(sq_sim._input_dict["tau_discharge"])
    sq_end = sq_sim._t_breakdown_trigger + sq_tau_dis
    assert np.isclose(
        sq_sim._effective_gas_puff_sccm(time=sq_end - 2e-3)[0], sq_Sgp,
        rtol=1e-6,
    )
    sq_mid_close = sq_sim._effective_gas_puff_sccm(
        time=sq_end + float(sq_sim._input_dict.get("gas_puff_close_lag_s", 5e-4))
    )[0]
    assert 0.3 * sq_Sgp < sq_mid_close < 0.7 * sq_Sgp
    assert sq_sim._effective_gas_puff_sccm(time=sq_end + 4e-3)[0] < 1e-3 * sq_Sgp
    # The afterglow phase switch stays open for the square tail.
    assert sq_sim._phase_switches("afterglow")["gas_puff_enabled"]


# --------------------------------------------------------------------
# neutral-equilibration-run-warning
# --------------------------------------------------------------------
@_case("neutral-equilibration-run-warning")
def _case_neutral_equilibration_run_warning():
    # --- direct run() with neutral_equilibration ON warns loudly -------------
    # The equilibration fires only from start_simulation(); a direct run()
    # silently started from the nn0 fill instead. That is a warning, never an
    # error -- existing direct-run scripts must keep working.
    import warnings as _eq_warnings

    _eq_params, _eq_flags = default_config()
    assert _eq_params["initial_neutral_state"] == "equilibrate", (
        "expected the equilibrate route by default"
    )
    _eq_params["nx"] = 12
    _eq_sim = LAPDSim1D(_eq_params, _eq_flags)
    with _eq_warnings.catch_warnings(record=True) as _eq_caught:
        _eq_warnings.simplefilter("always")
        _eq_direct = _eq_sim.run(t_end=0.0)
    _eq_hits = [
        w for w in _eq_caught if "run() was called directly" in str(w.message)
    ]
    assert len(_eq_hits) == 1, (
        "direct run() with neutral_equilibration ON must warn exactly once, "
        f"got {[str(w.message) for w in _eq_caught]}"
    )
    _eq_text = str(_eq_hits[0].message)
    assert "start_simulation" in _eq_text
    assert "nn0" in _eq_text
    assert _eq_direct is not None, "the warning must not abort the run"
    # ... and the equilibration-aware entry point stays silent.
    _eq_sim2 = LAPDSim1D(
        {**_eq_params, "initial_neutral_state": "fill"}, _eq_flags
    )
    with _eq_warnings.catch_warnings(record=True) as _eq_quiet:
        _eq_warnings.simplefilter("always")
        _eq_sim2.run(t_end=0.0)
    assert not [
        w for w in _eq_quiet if "run() was called directly" in str(w.message)
    ], "the warning must be gated on the neutral_equilibration flag"
    # ... and the run() start_simulation() drives is silent even with the flag
    # on (exercises the guard directly; a full equilibration costs ~1 minute).
    _eq_sim3 = LAPDSim1D(_eq_params, _eq_flags)
    _eq_sim3._run_via_start_simulation = True
    with _eq_warnings.catch_warnings(record=True) as _eq_inner:
        _eq_warnings.simplefilter("always")
        _eq_sim3.run(t_end=0.0)
    assert not [
        w for w in _eq_inner if "run() was called directly" in str(w.message)
    ], "start_simulation()'s own run() must not warn"
    assert _eq_sim3._run_via_start_simulation, "run() must not clear the guard"


# --------------------------------------------------------------------
# gas-puff-source-born-at-rest
# --------------------------------------------------------------------
@_case("gas-puff-source-born-at-rest")
def _case_gas_puff_source_born_at_rest():
    # --- S_gp is born at rest (2026-07-28) ---------------------------------
    # The gas puff is a source of ZERO-parallel-momentum particles: cold gas
    # arrives through the pipe with no directed axial momentum, so S_gp adds
    # nn (and nn_a) and must NEVER add M_n / M_n_a. The only momentum the
    # source/sink term carries is the PUMP sink, which removes the wind that
    # leaves with the gas it is attached to. This invariant is the premise of
    # the neutral-momentum campaign thread -- the flow observable is only
    # evidence if the puff cannot manufacture wind -- and it lives at two
    # sites that must stay consistent: the explicit source/sink RHS (both
    # zone layouts) and the implicit neutral-equilibration step. Both are
    # pinned below.
    from cablp.solvers._sim1d.core.geometry import build_geometry as _sgp_build

    sgp_params, sgp_flags = default_config()
    sgp_geom = _sgp_build(sgp_params, sgp_flags)
    sgp_cells = sgp_geom.cells
    sgp_ref_sim = LAPDSim1D(dict(sgp_params, nx=12), sgp_flags)
    sgp_mass = sgp_ref_sim.ion_mass_g
    # A puff level well onto the M6 square plateau, and the production
    # row/valve count, so the row under test is the one that runs.
    sgp_sccm = 3400.0
    sgp_valves = 2.0
    sgp_orifice = dict(
        orifice_id_cm=sgp_params["gas_puff_orifice_id_cm"],
        orifice_length_cm=sgp_params["gas_puff_orifice_length_cm"],
    )
    sgp_pump_lps = 4000.0
    sgp_puff = gas_puff_rate_profile(
        sgp_geom, sgp_sccm, sgp_valves, end=0, **sgp_orifice
    )
    assert np.any(sgp_puff > 0.0), "the puff profile under test must be live"
    sgp_kwargs = dict(
        geometry=sgp_geom,
        S_gp=sgp_sccm,
        Twin_S_gp=0.0,
        S_pump_L=sgp_pump_lps,
        S_pump_R=sgp_pump_lps,
        gas_puff_valves=sgp_valves,
        gas_puff_orifice_id_cm=sgp_orifice["orifice_id_cm"],
        gas_puff_orifice_length_cm=sgp_orifice["orifice_length_cm"],
    )
    sgp_zeros = np.zeros(sgp_cells, dtype=float)
    # A state carrying a real wind everywhere, so a spurious puff momentum
    # source could not hide behind an M_n that happens to be zero.
    sgp_state = conservative_from_primitives(
        n=np.full(sgp_cells, 1e12),
        nn=np.full(sgp_cells, 1e13),
        u=np.zeros(sgp_cells),
        Te=np.full(sgp_cells, 5.0),
        Ti=np.full(sgp_cells, 1.0),
        ion_mass_g=sgp_mass,
        un=np.full(sgp_cells, 3.0e4),
    )
    assert sgp_state.M_n is not None and np.all(sgp_state.M_n != 0.0)

    sgp_pump_i_left, sgp_pump_i_right = pump_cell_indices(sgp_geom)
    sgp_pump_mask = np.zeros(sgp_cells, dtype=bool)
    sgp_pump_mask[[sgp_pump_i_left, sgp_pump_i_right]] = True

    def _sgp_expected_pump_sink(momentum):
        """Return the pump-only momentum sink -rate * momentum per cell."""
        sink = np.zeros(sgp_cells, dtype=float)
        for sgp_idx, sgp_speed in (
            (sgp_pump_i_left, sgp_pump_lps),
            (sgp_pump_i_right, sgp_pump_lps),
        ):
            sgp_rate = pump_rate(
                _effective_pump_speed(sgp_speed, None),
                sgp_geom.neutral_volume_cm3[sgp_idx],
            )
            sink[sgp_idx] -= sgp_rate * momentum[sgp_idx]
        return sink

    # Site 1, single zone, puff ON / pumps OFF: the puff is the WHOLE nn
    # source and contributes exactly nothing to M_n. Bit-exact, not a
    # tolerance -- there is no momentum arithmetic to round.
    sgp_puff_only = neutral_source_sink_rhs(
        state=sgp_state, gas_puff_enabled=True, pump_enabled=False, **sgp_kwargs
    )
    assert np.array_equal(sgp_puff_only.M_n, sgp_zeros), (
        "S_gp must add no neutral momentum"
    )
    assert np.array_equal(sgp_puff_only.nn, sgp_puff)
    assert sgp_puff_only.M_n_a is None and sgp_puff_only.nn_a is None
    assert np.array_equal(sgp_puff_only.M, sgp_zeros)

    # Site 1, single zone, puff ON / pumps ON: the only nonzero dM_n cells are
    # the two pump cells, and there dM_n is exactly -pump_rate * M_n. Adding
    # the puff on top leaves that untouched.
    sgp_both = neutral_source_sink_rhs(
        state=sgp_state, gas_puff_enabled=True, pump_enabled=True, **sgp_kwargs
    )
    assert np.all(sgp_both.M_n[~sgp_pump_mask] == 0.0)
    assert np.all(sgp_both.M_n[sgp_pump_mask] != 0.0)
    assert np.array_equal(
        sgp_both.M_n, _sgp_expected_pump_sink(sgp_state.M_n)
    )
    # ... and the pump-on/pump-off dM_n difference is the pump sink alone,
    # i.e. the puff term is identical in both calls.
    assert np.array_equal(
        sgp_both.M_n - sgp_puff_only.M_n, _sgp_expected_pump_sink(sgp_state.M_n)
    )

    # Site 1, two zone (nn_a and M_n_a present): the puff feeds the ANNULUS
    # where one exists -- and still adds no momentum to either zone.
    sgp_V_col, sgp_V_ann = neutral_zone_volumes(sgp_geom)
    assert np.all(sgp_V_ann > 0.0), "expected an annulus on every cell here"
    sgp_tz_state = ConservativeState1D(
        n=sgp_state.n,
        nn=sgp_state.nn,
        M=sgp_state.M,
        Ee=sgp_state.Ee,
        Ei=sgp_state.Ei,
        M_n=sgp_state.M_n,
        nn_a=np.full(sgp_cells, 2.0e13),
        M_n_a=np.full(sgp_cells, 4.0e-9),
    )
    sgp_tz_puff = neutral_source_sink_rhs(
        state=sgp_tz_state,
        gas_puff_enabled=True,
        pump_enabled=False,
        **sgp_kwargs,
    )
    assert np.array_equal(sgp_tz_puff.M_n, sgp_zeros)
    assert np.array_equal(sgp_tz_puff.M_n_a, sgp_zeros)
    # The gas lands in the annulus, and the annulus-volume re-normalization
    # conserves the inflow exactly against the single-zone chamber form.
    assert np.any(sgp_tz_puff.nn_a > 0.0)
    assert np.array_equal(sgp_tz_puff.nn, sgp_zeros)
    assert np.allclose(
        sgp_tz_puff.nn_a * sgp_V_ann + sgp_tz_puff.nn * sgp_V_col,
        sgp_puff * np.asarray(sgp_geom.neutral_volume_cm3, dtype=float),
        rtol=1e-13,
        atol=0.0,
    )
    # With the pumps on, BOTH zone momenta carry pump sinks and nothing else.
    sgp_tz_both = neutral_source_sink_rhs(
        state=sgp_tz_state,
        gas_puff_enabled=True,
        pump_enabled=True,
        **sgp_kwargs,
    )
    assert np.array_equal(
        sgp_tz_both.M_n, _sgp_expected_pump_sink(sgp_tz_state.M_n)
    )
    assert np.array_equal(
        sgp_tz_both.M_n_a, _sgp_expected_pump_sink(sgp_tz_state.M_n_a)
    )
    assert np.all(sgp_tz_both.M_n[~sgp_pump_mask] == 0.0)
    assert np.all(sgp_tz_both.M_n_a[~sgp_pump_mask] == 0.0)

    # Site 2: the implicit neutral-equilibration step. The puff enters the
    # nn and nn_a linear solve only; M_n and M_n_a pass through bit-exact.
    sgp_tzq_sim = LAPDSim1D(
        dict(sgp_params, nx=12),
        dict(sgp_flags),
    )
    sgp_tzq_cells = sgp_tzq_sim.geometry.cells
    sgp_tzq_base = sgp_tzq_sim.state
    assert sgp_tzq_base.nn_a is not None
    sgp_tzq_Mn = np.full(sgp_tzq_cells, 5.0e-9)
    sgp_tzq_Mna = np.full(sgp_tzq_cells, 7.0e-9)
    sgp_tzq_state = ConservativeState1D(
        n=sgp_tzq_base.n,
        nn=sgp_tzq_base.nn,
        M=sgp_tzq_base.M,
        Ee=sgp_tzq_base.Ee,
        Ei=sgp_tzq_base.Ei,
        M_n=sgp_tzq_Mn.copy(),
        nn_a=sgp_tzq_base.nn_a,
        M_n_a=sgp_tzq_Mna.copy(),
    )
    sgp_tzq_next = sgp_tzq_sim._implicit_neutral_step_two_zone(
        1.0e-5, sgp_tzq_state, 5.0e-3
    )
    assert np.any(sgp_tzq_next.nn_a > sgp_tzq_state.nn_a), (
        "the two-zone puff must be feeding the annulus"
    )
    assert np.array_equal(sgp_tzq_next.M_n, sgp_tzq_Mn)
    assert np.array_equal(sgp_tzq_next.M_n_a, sgp_tzq_Mna)


# --------------------------------------------------------------------
# shaped-initial-neutral-fill-sp3
# --------------------------------------------------------------------
@_case("shaped-initial-neutral-fill-sp3")
def _case_shaped_initial_neutral_fill_sp3():
    # ---- sp3: shaped initial neutral fill (initial_neutral_state="profile") -
    # The capability replaces the uniform scalar nn0 with a per-cell array of
    # ABSOLUTE densities. Four questions decide it: does the off path still
    # build exactly the old initial condition, is a UNIFORM profile at the
    # scalar's own value bit-identical to the scalar run (the
    # null-construction identity -- the load-bearing check, because it is the
    # only one that says the array reaches the state through the same
    # arithmetic rather than merely near it), does a shaped profile arrive
    # unaltered in both zones, and does every misconfiguration raise.
    def _sp3_stance(**over):
        params, flags = default_config()
        params.update({
            "nx": 12,
            "dt_save": 0.0,
            "phase_transition_mode": "scheduled",
            "tau_neutral_prebreakdown": 0.0,
            "tau_prebreakdown": 0.0,
            "tau_breakdown": 0.0,
            "tau_discharge": 1.0,
            "tau_afterglow": 0.0,
            "beam_anomalous_model": "quasilinear",
            # The surface held: no step's temperature increment survives
            # this heat capacity, and nothing cleans at a zero cross section.
            "cathode_Ts_base_K": 1998.15,
            "cathode_heat_capacity_J_per_K": 1.0e30,
            "cathode_cleaning_sigma_cm2": 0.0,
            "cathode_cleaning_E_th_eV": None,
        })
        # The shaped IC and the equilibrated seed are alternative statements of
        # the same initial condition, so the comparison stance takes the
        # scalar-fill route on the scalar arm and the profile route on the
        # shaped one -- never the equilibrated seed.
        params["initial_neutral_state"] = "fill"
        # The scalar arm below compares the two arms at the raw bit level, so
        # the stance names the cold layout rather than inheriting it.
        _pin_pre_r2a_neutral_stance(params, flags)
        # A UNIFORM nx=12 column. The spreading-kernel checks in (e) state
        # their widths in CELLS and convert with the mesh's MEAN cell length,
        # which only means "cells" on a uniform mesh; the fixed source region
        # (a config default since the R2a fold-in) makes cell sizes differ by
        # a factor of several, and a 2-cell kernel would then be sub-cell where
        # it lands. The IC construction under test is mesh-agnostic.
        flags["source_fixed_grid"] = False
        params["source_region_length_cm"] = None
        params["source_region_dz_cm"] = None
        params.update(over)
        return params, flags

    _sp3_scalar_p, _sp3_scalar_f = _sp3_stance()
    _sp3_scalar_sim = LAPDSim1D(dict(_sp3_scalar_p), dict(_sp3_scalar_f))
    _sp3_cells = int(_sp3_scalar_sim.geometry.cells)
    _sp3_nn0 = float(_sp3_scalar_p["nn0"])

    # (a) OFF PATH. No profile object is built and the initial fill is the
    # historical uniform array, cell for cell.
    assert _sp3_scalar_sim._nn0_profile is None
    assert _sp3_scalar_sim._nn0_annulus_profile is None
    assert np.array_equal(
        _sp3_scalar_sim.state.nn, np.full(_sp3_cells, _sp3_nn0)
    )
    assert np.array_equal(
        _sp3_scalar_sim.state.nn_a, np.full(_sp3_cells, _sp3_nn0)
    )

    # (b) THE NULL-CONSTRUCTION IDENTITY. A uniform profile at the scalar's own
    # value must reproduce the scalar run at the RAW BIT level, not merely
    # close: the array path and the scalar path have to reach the state
    # through the same arithmetic.
    def _sp3_raw(result):
        return [
            np.ascontiguousarray(y, dtype=float).view(np.uint64).tobytes()
            for y in result.y
        ]

    _sp3_scalar_result = LAPDSim1D(
        dict(_sp3_scalar_p), dict(_sp3_scalar_f)
    ).run(t_end=1.0e-6, dt=1.0e-7)
    _sp3_uniform_p, _sp3_uniform_f = _sp3_stance(
        nn0=None, nn0_profile=[_sp3_nn0] * _sp3_cells
    )
    _sp3_uniform_p["initial_neutral_state"] = "profile"
    _sp3_uniform_result = LAPDSim1D(
        _sp3_uniform_p, _sp3_uniform_f
    ).run(t_end=1.0e-6, dt=1.0e-7)
    assert _sp3_scalar_result.steps == _sp3_uniform_result.steps > 0, (
        _sp3_scalar_result.steps, _sp3_uniform_result.steps
    )
    assert _sp3_raw(_sp3_scalar_result) == _sp3_raw(_sp3_uniform_result), (
        "a uniform nn0_profile at the scalar's own value must be bit-identical "
        "to the scalar run"
    )

    # (c) A SHAPED PROFILE ROUND-TRIPS. state.nn at t = 0 IS the supplied
    # array -- no normalization, no rescaling, no role masking -- and the same
    # for the annulus.
    _sp3_shape = (
        _sp3_nn0 * (1.5 + np.sin(np.arange(_sp3_cells, dtype=float)))
    ).tolist()
    _sp3_ann_shape = (
        _sp3_nn0 * (2.5 + np.cos(np.arange(_sp3_cells, dtype=float)))
    ).tolist()
    _sp3_shaped_p, _sp3_shaped_f = _sp3_stance(
        nn0=None, nn0_profile=_sp3_shape
    )
    _sp3_shaped_p["initial_neutral_state"] = "profile"
    _sp3_shaped_sim = LAPDSim1D(dict(_sp3_shaped_p), dict(_sp3_shaped_f))
    assert np.array_equal(_sp3_shaped_sim.state.nn, np.array(_sp3_shape))

    _sp3_tz_f = dict(_sp3_shaped_f)
    _sp3_tz_p = dict(_sp3_shaped_p)
    _sp3_tz_sim = LAPDSim1D(dict(_sp3_tz_p), dict(_sp3_tz_f))
    # Omitted annulus profile => the shipped convention that both zones start
    # at the same fill, in its shaped form.
    assert np.array_equal(_sp3_tz_sim.state.nn, np.array(_sp3_shape))
    assert np.array_equal(_sp3_tz_sim.state.nn_a, np.array(_sp3_shape))
    _sp3_tz_p["nn0_annulus_profile"] = _sp3_ann_shape
    _sp3_tz2_sim = LAPDSim1D(dict(_sp3_tz_p), dict(_sp3_tz_f))
    assert np.array_equal(_sp3_tz2_sim.state.nn, np.array(_sp3_shape))
    assert np.array_equal(_sp3_tz2_sim.state.nn_a, np.array(_sp3_ann_shape))

    # (d) EVERY MISCONFIGURATION RAISES, at construction.
    def _sp3_refuses(label, params_over=None, flags_over=None):
        params, flags = _sp3_stance(nn0=None, nn0_profile=_sp3_shape)
        params["initial_neutral_state"] = "profile"
        params.update(params_over or {})
        flags.update(flags_over or {})
        try:
            LAPDSim1D(params, flags)
        except ValueError:
            return
        raise AssertionError(f"the profile route must refuse: {label}")

    _sp3_refuses(
        "a wrong-length profile",
        params_over={"nn0_profile": _sp3_shape[:-1]},
    )
    _sp3_refuses(
        "a non-finite entry",
        params_over={"nn0_profile": _sp3_shape[:-1] + [float("nan")]},
    )
    _sp3_refuses(
        "a zero entry (a density is positive; nn_floor would paper it over)",
        params_over={"nn0_profile": _sp3_shape[:-1] + [0.0]},
    )
    _sp3_refuses(
        "a negative entry",
        params_over={"nn0_profile": _sp3_shape[:-1] + [-1.0]},
    )
    _sp3_refuses(
        "the route armed alongside an explicit scalar nn0",
        params_over={"nn0": _sp3_nn0},
    )
    # The profile and the equilibrated seed are values of one selector, so
    # the pair the solver used to refuse is unrepresentable; an unknown
    # selector value is the refusal that remains.
    _sp3_refuses(
        "an initial_neutral_state value the selector does not accept",
        params_over={"initial_neutral_state": "profiled"},
    )
    _sp3_refuses(
        "restart_from, which replaces the whole initial condition",
        params_over={"restart_from": "nonexistent.h5"},
    )
    _sp3_refuses(
        "the route armed with no profile at all",
        params_over={"nn0_profile": None},
    )
    # ...and the presence gate the other way: either key set off the profile
    # route is inert, so it raises rather than running the uniform fill
    # silently.
    for _sp3_key, _sp3_value in (
        ("nn0_profile", _sp3_shape),
        ("nn0_annulus_profile", _sp3_ann_shape),
    ):
        _sp3_off_p, _sp3_off_f = _sp3_stance(**{_sp3_key: _sp3_value})
        try:
            LAPDSim1D(_sp3_off_p, _sp3_off_f)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"{_sp3_key} must be refused off the profile route"
            )
    # The complementary presence gate, and the witness for the RETIRED
    # gas-puff nn0 table. ``nn0 = None`` is the profile route's own
    # requirement, but off that route it used to fall through to a frozen
    # lookup keyed on
    # S_gp -- ungenerable, on a superseded sccm convention, reached by nothing
    # that ships. The table is gone, so the only remaining reading of a None
    # here is "no initial neutral density was configured", and resolve_nn0
    # refuses it. The solver resolves the fill inside __init__, so the refusal
    # is a construction-time ValueError.
    _sp3_no_nn0_p, _sp3_no_nn0_f = _sp3_stance(nn0=None)
    try:
        LAPDSim1D(_sp3_no_nn0_p, _sp3_no_nn0_f)
    except ValueError as _sp3_no_nn0_error:
        assert "nn0 accepts" in str(_sp3_no_nn0_error), _sp3_no_nn0_error
        assert "initial_neutral_state='profile'" in str(_sp3_no_nn0_error)
    else:
        raise AssertionError(
            "nn0 = None off the profile route must be refused"
        )

    # (e) THE CONSTRUCTION SCRIPT. It is an instrument in scripts/, not repo
    # physics, so it is imported HERE rather than at module scope -- the smoke
    # suite's import block stays package-only. Two properties are asserted:
    # every spreading kernel conserves the injected inventory on the grid
    # exactly, and the ledger's throughput arithmetic reproduces the sp2
    # bridge numbers under BOTH stated conventions.
    import sp3_build_nn0 as _sp3_mod

    _sp3_geom = _sp3_scalar_sim.geometry
    # The orifice row is not masked to the main-chamber roles (it lands where
    # the rays land); the spreading kernels act on those roles only, so the
    # conservation statement is made on the row's part inside them.
    _sp3_deposit = gas_puff_rate_profile(
        _sp3_geom, 5200.0, 2, z_cm=60.0, orifice_id_cm=3.95,
        orifice_length_cm=22.0,
    ) * np.asarray(_sp3_geom.neutral_volume_cm3, dtype=float)
    _sp3_deposit = _sp3_deposit * np.array(
        [
            _sp3_role in _sp3_mod._PUFF_ELIGIBLE_ROLES
            for _sp3_role in _sp3_geom.cell_role
        ],
        dtype=float,
    )
    assert float(_sp3_deposit.sum()) > 0.0
    # Widths are stated in CELLS, not centimetres: this smoke geometry is far
    # coarser than a production grid, and a kernel narrower than one cell is
    # the identity on any grid -- it would conserve trivially and say nothing.
    _sp3_dz = float(np.mean(np.asarray(_sp3_geom.length_cm, dtype=float)))
    for _sp3_kernel in _sp3_mod.KERNELS:
        for _sp3_cells_wide in (0.4, 2.0, 6.0):
            _sp3_width = _sp3_cells_wide * _sp3_dz
            _sp3_W = _sp3_mod.spread_matrix(_sp3_geom, _sp3_kernel, _sp3_width)
            _sp3_out = _sp3_W @ _sp3_deposit
            _sp3_rel = abs(
                float(_sp3_out.sum()) - float(_sp3_deposit.sum())
            ) / float(_sp3_deposit.sum())
            assert _sp3_rel < 1e-12, (_sp3_kernel, _sp3_width, _sp3_rel)
            if _sp3_cells_wide >= 2.0:
                # ...and it really moved gas: a spread that never moved
                # anything conserves trivially.
                assert float(np.max(_sp3_out)) < float(
                    np.max(_sp3_deposit)
                ), (_sp3_kernel, _sp3_cells_wide)
    # ...and the h5 BASE reader, which is what makes the verdict arm's single
    # delta the foot addition: with the sp1 reference as base, the arm starts
    # from the reference's OWN equilibrated initial profile rather than from
    # the uniform convention (7.5x denser). The reader must return those
    # frames exactly, and refuse a grid mismatch.
    _sp3_base_p, _sp3_base_f = _sp3_stance()
    _sp3_base_p["nn0"] = 3.3e12
    _sp3_base_sim = LAPDSim1D(dict(_sp3_base_p), dict(_sp3_base_f))
    _sp3_base_result = _sp3_base_sim.run(t_end=1.0e-6, dt=1.0e-7)
    with tempfile.TemporaryDirectory() as _sp3_tmpdir:
        _sp3_h5 = str(Path(_sp3_tmpdir) / "sp3_base_source.h5")
        _sp3_base_sim.save_result(_sp3_h5, _sp3_base_result)
        _sp3_got_col, _sp3_got_ann = _sp3_mod.base_profiles_from_h5(
            _sp3_h5, _sp3_base_sim.geometry.cells
        )
        # THE NULL CONSTRUCTION at the reader: the base IS the source's t=0
        # frames, bit for bit, in both zones.
        assert np.array_equal(
            _sp3_got_col, np.asarray(_sp3_base_result.nn[0], dtype=float)
        )
        assert np.array_equal(
            _sp3_got_ann, np.asarray(_sp3_base_result.nn_a[0], dtype=float)
        )
        # ...and it really is a profile from the run, not the scalar echoed:
        # the equilibration-free stance starts uniform, so this fixture is
        # checked for having been READ rather than reconstructed.
        assert np.array_equal(
            _sp3_got_col, np.full(_sp3_got_col.size, 3.3e12)
        )
        for _sp3_bad_cells, _sp3_why in (
            (_sp3_base_sim.geometry.cells - 1, "a cell-count mismatch"),
        ):
            try:
                _sp3_mod.base_profiles_from_h5(_sp3_h5, _sp3_bad_cells)
            except ValueError:
                pass
            else:
                raise AssertionError(
                    f"base_profiles_from_h5 must refuse {_sp3_why}"
                )
    # A zero-width kernel is not a spread; the null control is dt_foot = 0,
    # which never reaches the kernel at all.
    for _sp3_bad_width in (0.0, -1.0):
        try:
            _sp3_mod.spread_matrix(_sp3_geom, "diffusive", _sp3_bad_width)
        except ValueError:
            pass
        else:
            raise AssertionError("spread_matrix must refuse a non-positive width")

    # THE FOOT REGISTRATION. dt_foot is MEASURED per rung -- the machine's
    # circuit-on -> 1 kA lead minus the model's own circuit-on -> 1 kA time,
    # rounded to the 10 us the leads are quoted at -- and its bracket is that
    # lead's shot-to-shot sd, so the bracket is an error bar centred on the
    # registered foot rather than a pair of choices. Both properties are
    # asserted off the registered constants, so a rung added or a time
    # re-measured cannot silently break the arithmetic. The ROUNDING is pinned
    # too, because it is what lets an omitted --dt-foot-s reproduce a
    # committed fill: the raw subtraction would miss it in the last digits.
    for _sp3_es in (1, 2, 3):
        _sp3_foot = _sp3_mod.registered_foot_s(_sp3_es)
        _sp3_lo, _sp3_hi = _sp3_mod.dt_foot_bracket_s(_sp3_es)
        _sp3_sd = _sp3_mod.MEASURED_LEAD_SD_S[_sp3_es]
        assert _sp3_foot == round(
            _sp3_mod.MEASURED_LEAD_S[_sp3_es]
            - _sp3_mod.MODEL_1KA_S[_sp3_es],
            _sp3_mod.FOOT_QUANTUM_DECIMALS,
        ), (_sp3_es, _sp3_foot)
        assert _sp3_foot > 0.0, (_sp3_es, _sp3_foot)
        assert abs((_sp3_lo + _sp3_hi) / 2.0 - _sp3_foot) < 1e-15, _sp3_es
        assert abs((_sp3_hi - _sp3_lo) - 2.0 * _sp3_sd) < 1e-15, _sp3_es

    # ------------------------------------------------------------------
    # THE REGISTRATION ITSELF, PINNED AS LITERALS.
    #
    # Everything above checks the registration is SELF-CONSISTENT: the foot is
    # the rounded subtraction, the bracket is centred on it. That says nothing
    # about WHICH numbers are registered, and the committed initial-fill rows
    # are built from exactly those numbers -- so restoring a superseded time,
    # or flipping the builder's registered spreading member back, moves the
    # fill of every rung while every self-consistency check above still passes.
    # The only instrument that catches either otherwise needs an equilibrated
    # base and a per-cell geometry that do not live in this repository, so it
    # cannot ride a per-merge gate.
    #
    # These literals are therefore the per-merge gate on the registration.
    # EVERY ONE OF THEM IS A STANCE EVENT TO MOVE: the committed
    # nn0_profile / nn0_annulus_profile rows are rebuilt with it, the
    # configuration identity rotates, and the golden is re-anchored. Changing a
    # number here without doing that is the mistake this block exists to make
    # loud.
    # ------------------------------------------------------------------
    # The REGISTERED spreading member. Moving it rebuilds every committed fill
    # row through a different operator: a stance event.
    assert _sp3_mod.KERNEL_REGISTERED == "knudsen", _sp3_mod.KERNEL_REGISTERED
    assert _sp3_mod.KNUDSEN_KERNEL == "knudsen"
    # ...and the CLI's own default, which is what an omitted --kernel builds and
    # therefore what every reproduction invocation relies on. argparse carries
    # it separately from the constant above, so it is read from the parser the
    # builder actually constructs -- captured by standing in for parse_args, so
    # main() gets no further and nothing is built. Moving it: a stance event.
    _sp3_parsers = []
    _sp3_real_parse_args = argparse.ArgumentParser.parse_args

    def _sp3_capture_parser(self, *args, **kwargs):
        _sp3_parsers.append(self)
        raise SystemExit(0)

    argparse.ArgumentParser.parse_args = _sp3_capture_parser
    try:
        _sp3_mod.main([])
    except SystemExit:
        pass
    finally:
        argparse.ArgumentParser.parse_args = _sp3_real_parse_args
    assert len(_sp3_parsers) == 1, len(_sp3_parsers)
    assert _sp3_parsers[0].get_default("kernel") == "knudsen", (
        _sp3_parsers[0].get_default("kernel")
    )
    # The registered coefficient and the two ends of its closure bracket, in
    # D_i = kappa R_i vbar. Moving the reference rebuilds every row; moving
    # either end restates the bracket a result is quoted with. Stance events.
    assert _sp3_mod.KNUDSEN_KAPPA_REFERENCE == 2.0 / 3.0
    assert _sp3_mod.KNUDSEN_KAPPA_SLOW == 0.45
    assert _sp3_mod.KNUDSEN_KAPPA_FAST == 0.90
    assert _sp3_mod.KNUDSEN_MEMBERS == {
        "reference": 2.0 / 3.0, "slow": 0.45, "fast": 0.90
    }, _sp3_mod.KNUDSEN_MEMBERS
    assert _sp3_mod.KNUDSEN_MEMBER_DEFAULT == "reference"
    # The substep count the foot is integrated over. Fixed so a rebuild writes
    # the same bytes, so moving it moves those bytes: a stance event.
    assert _sp3_mod.KNUDSEN_SUBSTEPS_DEFAULT == 600
    # Gap coupling is REGISTERED ON: the region behind the anode mesh is
    # carried, and the operator may place gas there. Turning it off is a
    # disclosed alternate, not a default -- flipping the registration is a
    # stance event.
    assert _sp3_mod.KNUDSEN_GAP_COUPLING_REGISTERED is True
    # The registered source convention is the FIRST of the two, the deposit
    # released continuously over the foot. Swapping the order would silently
    # re-register it, so the order is pinned too: a stance event either way.
    assert _sp3_mod.KNUDSEN_SOURCE_CONVENTIONS == (
        "continuous", "deposit_t0"
    ), _sp3_mod.KNUDSEN_SOURCE_CONVENTIONS
    # THE THREE REGISTERED FEET [s] and their brackets. The foot is what an
    # omitted --dt-foot-s supplies, so these three numbers ARE the committed
    # fills' durations; a superseded measured lead or model 1 kA time restored
    # upstream lands here. Each is a stance event.
    assert _sp3_mod.registered_foot_s(1) == 0.00588
    assert _sp3_mod.registered_foot_s(2) == 0.00666
    assert _sp3_mod.registered_foot_s(3) == 0.00663
    # The brackets are the registered foot +- the lead's shot-to-shot sd. They
    # are compared to their literals within 1e-15 s rather than exactly,
    # because the addition is done in float64 and one end is not representable.
    for _sp3_es, _sp3_want in (
        (1, (0.00579, 0.00597)),
        (2, (0.00664, 0.00668)),
        (3, (0.00654, 0.00672)),
    ):
        _sp3_got = _sp3_mod.dt_foot_bracket_s(_sp3_es)
        assert all(abs(g - w) < 1e-15 for g, w in zip(_sp3_got, _sp3_want)), (
            _sp3_es, _sp3_got, _sp3_want
        )

    # vbar at 300 K helium, and the ballistic reach of the ES1 registered foot.
    _sp3_vbar = _sp3_mod.mean_speed_cm_s(300.0, m_He_cgs)
    assert 1.25e5 < _sp3_vbar < 1.27e5, _sp3_vbar
    _sp3_reach = _sp3_vbar * _sp3_mod.registered_foot_s(1)
    assert 7.2e2 < _sp3_reach < 7.5e2, _sp3_reach
    # The sp2 bridge numbers, 4.7e18--2.1e19 atoms, are HISTORICAL: they are
    # the sp2 leg's own foot times, 2.0 ms and 4.5 ms, read under the two
    # throughput conventions -- the low end per-valve-nominal at 2.0 ms, the
    # high end as-applied at 4.5 ms. Those foot times are properties of that
    # banked leg and are written here rather than read from the live
    # registration, which no longer carries them. puff_rate(..., 1.0) is the
    # repo's own throughput constant per unit volume, so this is the solver's
    # arithmetic and not a restatement of it. The 5200 sccm the sp2 leg ran is
    # a FITTED-FLUX quantity: it was fitted under the retired 0 C sccm
    # convention, so the 2026-08-21 meter changeover rescales its digits by
    # 1.0734834 (-> 5582.11 meter-sccm) to hold the delivered particle flux
    # fixed. The banked ATOM counts below are physical and do not move.
    _sp3_sp2_feet_s = (2.0e-3, 4.5e-3)
    _sp3_nominal = puff_rate(5582.11, 1, 1.0) * min(_sp3_sp2_feet_s)
    _sp3_applied = puff_rate(5582.11, 2, 1.0) * max(_sp3_sp2_feet_s)
    assert abs(_sp3_nominal / 4.7e18 - 1.0) < 0.02, _sp3_nominal
    assert abs(_sp3_applied / 2.1e19 - 1.0) < 0.02, _sp3_applied
    # The two ends differ by exactly the valve factor times the foot ratio, so
    # the pair is one leg read under two conventions and not two independent
    # numbers.
    assert abs(_sp3_applied / _sp3_nominal - (2.0 * 4.5) / 2.0) < 1e-12


# --------------------------------------------------------------------
# fill-spreading-knudsen-operator
# --------------------------------------------------------------------
@_case("fill-spreading-knudsen-operator")
def _case_fill_spreading_knudsen_operator():
    # ---- the initial fill's wall-limited spreading operator ----------------
    # sp3_build_nn0.py's third spreading member is a conservative
    # finite-volume diffusion solve rather than a stencil, and its acceptance
    # instrument is scripts/verify/verify_fill_spreading.py, which carries the
    # whole gate set. What runs HERE is that instrument's own FAST SUBSET --
    # the gates decidable on a synthetic mesh in milliseconds: that a source
    # uniform per unit volume produces a density continuous across a bore step
    # (with the legacy length-weighted route as the negative control that must
    # miss it by the area ratio), that the long-time limit is the uniform
    # density the inventory allows, that the one-substep propagator obeys
    # detailed balance with respect to VOLUME, and that a free-space spread
    # carries the diffusivity it was given. The gates that need a production
    # configuration built, or a file from outside the repository, are the
    # verifier's own to run.
    #
    # The verifier is an instrument in scripts/, not repo physics, so it is
    # imported HERE rather than at module scope, following the sp3 precedent
    # above. Its subset is called rather than re-implemented: one statement of
    # each property, in the file that owns it.
    import verify_fill_spreading as _fill_mod

    _fill_results = _fill_mod.fast_gates()
    assert _fill_results, "the fast subset must run at least one gate"
    for _fill_name, _fill_ok, _fill_lines in _fill_results:
        assert _fill_ok, (_fill_name, _fill_lines)
        assert _fill_lines, _fill_name


# --------------------------------------------------------------------
# equilibration-map-slicer
# --------------------------------------------------------------------
@_case("equilibration-map-slicer")
def _case_equilibration_map_slicer():
    # ---- eqmap: the equilibration map's slicer ----------------------------
    # eqmap_make.py runs a 101st cycle whose puff is open for the foot fill
    # time and keeps nn(z,t) as a map of starting distributions; eqmap_slice.py
    # cuts a pre-fill time out of it into an sp3 shaped-fill npz. The PRODUCER
    # carries its own two checks (its t=0 row is the equilibrated seed; the
    # inventory books as puff minus pump) and needs a full equilibration to run,
    # which is far too slow for a smoke suite -- so what is asserted here is the
    # SLICER, whose properties are decidable on a synthetic map in milliseconds.
    # Both are instruments in scripts/, so they are imported HERE rather than at
    # module scope, following the sp3 precedent above.
    import eqmap_make as _eq_make_mod
    import eqmap_slice as _eq_mod

    # A synthetic map at a real stance, so the construction check below has a
    # geometry to land in. The rows are arbitrary but positive and far above
    # nn_floor; nothing here asserts physics, only the slicer's arithmetic.
    _eq_p, _eq_f = default_config()
    _eq_p["nx"] = 12
    _eq_cells = int(LAPDSim1D(dict(_eq_p), dict(_eq_f)).geometry.cells)
    _eq_z = np.asarray(
        LAPDSim1D(dict(_eq_p), dict(_eq_f)).geometry.z_cm, dtype=float
    )
    _eq_t = np.array([0.0, 1e-3, 2e-3, 3e-3], dtype=float)
    _eq_nn = 1e12 * (
        1.0 + np.outer(np.arange(1.0, 5.0), np.linspace(1.0, 2.0, _eq_cells)) ** 2
    )
    _eq_header = {
        "es": None, "nx": 12, "cells": _eq_cells, "two_zone": False,
        "S_gp_sccm": float(_eq_p["S_gp"]), "foot_s": 3e-3, "cadence_s": 1e-3,
        "stance_extra": {}, "stance_extra_flag": {},
    }
    with tempfile.TemporaryDirectory() as _eq_dir:
        _eq_map = str(Path(_eq_dir) / "map.npz")
        np.savez(
            _eq_map,
            format=_eq_make_mod.MAP_FORMAT,
            t_s=_eq_t,
            nn=_eq_nn,
            z_cm=_eq_z,
            provenance=json.dumps(_eq_header, sort_keys=True),
        )
        _eq_loaded = _eq_mod.load_map(_eq_map)
        assert np.array_equal(_eq_loaded["nn"], _eq_nn)

        # (i) AN EXACT SAMPLE IS COPIED, NOT INTERPOLATED. This is what makes
        # the producer's null survive the slicer: a map's t=0 row IS the
        # standard equilibrated seed, so slicing at t=0 must reproduce it at
        # the raw bit level rather than merely to rounding.
        for _eq_k, _eq_time in enumerate(_eq_t):
            _eq_cut, _eq_mode, _, _, _ = _eq_mod.interpolate(
                _eq_t, _eq_nn, float(_eq_time), 1e-12
            )
            assert _eq_mode == "exact_sample", (_eq_k, _eq_mode)
            assert _eq_cut.tobytes() == _eq_nn[_eq_k].tobytes(), (
                "an exact pre-fill time must return the recorded sample "
                "verbatim, bit for bit"
            )

        # (ii) A MIDPOINT IS THE LINEAR BLEND, and the bracket it reports is
        # the bracket it used.
        _eq_cut, _eq_mode, _eq_lo, _eq_hi, _eq_w = _eq_mod.interpolate(
            _eq_t, _eq_nn, 1.5e-3, 1e-12
        )
        assert (_eq_mode, _eq_lo, _eq_hi) == ("linear", 1, 2)
        assert abs(_eq_w - 0.5) < 1e-15, _eq_w
        assert np.allclose(
            _eq_cut, 0.5 * (_eq_nn[1] + _eq_nn[2]), rtol=0.0, atol=0.0
        )
        # ...and a linear blend of positive rows is positive, so the slicer
        # cannot manufacture a value the sp3 positivity validator would refuse.
        assert np.all(_eq_cut > 0.0)

        # (iii) THE ERROR FIGURE IS A BOUND WHERE IT CLAIMS TO BE ONE, and
        # DISCLAIMS ITSELF ACROSS THE VALVE-OPENING CORNER. The map's t=0 is a
        # corner in nn(t) -- flat before the valve opens, rising after -- and no
        # stencil inside the map spans it, so a bracket touching sample 0 is
        # reported as an estimate rather than a bound. That distinction is the
        # instrument's honesty and is asserted, not assumed.
        _eq_abs, _eq_rel, _eq_valid, _eq_why = _eq_mod.interpolation_error(
            _eq_t, _eq_nn, 1, 2
        )
        assert _eq_abs > 0.0 and _eq_rel > 0.0
        assert _eq_valid, "an interior bracket's figure must be a bound"
        assert "NOT A BOUND" not in _eq_why
        # These rows are quadratic in the sample index, so the second difference
        # is exact and the true error must actually sit under the bound.
        _eq_true = float(np.max(np.abs(_eq_cut - 0.5 * (_eq_nn[1] + _eq_nn[2]))))
        assert _eq_true <= _eq_abs * (1.0 + 1e-9), (_eq_true, _eq_abs)
        _, _, _eq_valid0, _eq_why0 = _eq_mod.interpolation_error(
            _eq_t, _eq_nn, 0, 1
        )
        assert not _eq_valid0, (
            "a bracket spanning the map's t=0 corner must NOT claim a bound"
        )
        assert "NOT A BOUND" in _eq_why0

        # (iv) EVERY MISUSE RAISES rather than writing a quietly wrong fill.
        _eq_bad = str(Path(_eq_dir) / "notamap.npz")
        np.savez(_eq_bad, nn=_eq_nn)
        try:
            _eq_mod.load_map(_eq_bad)
        except ValueError:
            pass
        else:
            raise AssertionError("load_map must refuse a non-eqmap npz")
        for _eq_label, _eq_values in (
            ("a non-positive entry", np.append(_eq_nn[0][:-1], 0.0)),
            ("a non-finite entry", np.append(_eq_nn[0][:-1], np.nan)),
            ("an entry below nn_floor", np.append(_eq_nn[0][:-1], 1.0)),
        ):
            try:
                _eq_mod.validate_profile(_eq_values, "nn0_profile", 1e8)
            except ValueError:
                continue
            raise AssertionError(f"the slicer must refuse {_eq_label}")

        # (v) THE ROUND TRIP. A written slice loads through run_m6_point's own
        # reading of the file, passes every sp3 construction-time validator,
        # and arrives in the state unaltered -- which is the whole delivery
        # contract, asserted end to end.
        _eq_out = str(Path(_eq_dir) / "slice.npz")
        assert _eq_mod.main([
            "--map", _eq_map, "--prefill-s", "1.5e-3", "--out", _eq_out,
        ]) == 0
        with np.load(_eq_out, allow_pickle=False) as _eq_data:
            assert "nn0_profile" in _eq_data.files, (
                "the slice must carry the key run_m6_point.py looks for"
            )
            assert np.array_equal(
                np.asarray(_eq_data["nn0_profile"], dtype=float), _eq_cut
            )
            assert "nn0_annulus_profile" not in _eq_data.files, (
                "a single-zone map must not emit an annulus array"
            )
            assert json.loads(str(_eq_data["provenance"]))["prefill_s"] == 1.5e-3
        _eq_check = _eq_mod.selfcheck(_eq_out, _eq_header)
        assert _eq_check["pass"], _eq_check
        assert _eq_check["cells"] == _eq_cells

        # (vi) EXTRAPOLATION IS REFUSED. The map's axis is the foot window that
        # was actually run; a pre-fill time past it has no recorded fill and
        # inventing one would be a silent fabrication.
        for _eq_bad_t in ("-1e-3", "4e-3"):
            try:
                _eq_mod.main([
                    "--map", _eq_map, "--prefill-s", _eq_bad_t,
                    "--out", str(Path(_eq_dir) / "never.npz"),
                ])
            except ValueError:
                continue
            raise AssertionError(
                f"a pre-fill time of {_eq_bad_t} s is off the map and must raise"
            )


# --------------------------------------------------------------------
# hot-channel-internal-wall
# --------------------------------------------------------------------
@_case("hot-channel-internal-wall")
def _case_hot_channel_internal_wall():
    # ==================================================================
    # HOT-CHANNEL INTERNAL WALL (neutral_hot_internal_wall, default on).
    # The ballistic flight kernel clips at the closed/absorbing plasma faces
    # as well as the two global end planes, so a live cell's landings can no
    # longer fall on a plasma-dead cell (where the topology mask would delete
    # the deposit) and a dead cell's can no longer fall on a live one.
    # ==================================================================
    from cablp.solvers._sim1d.physics.hot_neutrals import (
        ballistic_flight_kernels as _hiw_kernels,
        flight_wall_bounds as _hiw_bounds,
    )

    _hiw_p, _hiw_f = default_config()
    _hiw_geom = LAPDSim1D(dict(_hiw_p), dict(_hiw_f)).geometry
    _hiw_dead = ~np.asarray(_hiw_geom.plasma_active, dtype=bool)
    assert _hiw_dead.any(), "the default machine must carry a plasma-dead plenum"

    # Off, the bounds ARE the two end planes -- which is what makes the off
    # path's clips the historical ones.
    _hiw_zlo, _hiw_zhi, _hiw_clo, _hiw_chi = _hiw_bounds(_hiw_geom, internal_wall=False)
    assert np.all(_hiw_zlo == _hiw_geom.z_edges_cm[0])
    assert np.all(_hiw_zhi == _hiw_geom.z_edges_cm[-1])
    assert np.all(_hiw_clo == 0) and np.all(_hiw_chi == _hiw_geom.cells - 1)

    _hiw_off = _hiw_kernels(_hiw_geom, samples=401, internal_wall=False)
    _hiw_on = _hiw_kernels(_hiw_geom, samples=401, internal_wall=True)
    for _hiw_k in (0, 1):
        # The solid-angle normalization identity survives the extra walls.
        assert np.allclose(_hiw_off[_hiw_k].sum(axis=1), 1.0, rtol=0.0, atol=1e-12)
        assert np.allclose(_hiw_on[_hiw_k].sum(axis=1), 1.0, rtol=0.0, atol=1e-12)
    # THE GATE: with the wall on, no live birth cell puts any landing (or any
    # residence) mass over a plasma-dead cell, and no dead one puts any over a
    # live cell. Exactly zero, not small -- the flights never cross the face.
    for _hiw_k, _hiw_name in ((0, "landing"), (1, "residence")):
        _hiw_m = _hiw_on[_hiw_k]
        assert np.all(_hiw_m[~_hiw_dead][:, _hiw_dead] == 0.0), _hiw_name
        assert np.all(_hiw_m[_hiw_dead][:, ~_hiw_dead] == 0.0), _hiw_name
    # ... and it is a real discriminator: off, the live cells against the
    # cathode disc DO deposit over the plenum, which is the deleted stream.
    assert _hiw_off[0][~_hiw_dead][:, _hiw_dead].sum() > 0.0

    # The flag is presence-gated on the channel it walls.
    _hiw_bad_p, _hiw_bad_f = default_config()
    _hiw_bad_f["neutral_hot_internal_wall"] = True
    _hiw_bad_f["neutral_energy"] = False
    try:
        LAPDSim1D(dict(_hiw_bad_p), dict(_hiw_bad_f))
    except ValueError as _hiw_exc:
        assert "neutral_hot_internal_wall" in str(_hiw_exc), str(_hiw_exc)
        assert "neutral_energy" in str(_hiw_exc), str(_hiw_exc)
    else:
        raise AssertionError(
            "neutral_hot_internal_wall without neutral_energy must be refused"
        )


@_case("kinetic-geff-thermal-floor")
def _case_kinetic_geff_thermal_floor():
    """Pin the ASSEMBLED ``g_eff`` thermal floor, as a value, once.

    ``ion_thermal_g_eff_floor_cm2_s2`` is the single definition of
    ``8 k Ti / (pi m)`` that the DVM collision operator, the hot surface
    carrier, the TPMC fast-reflected arm and the E2 comparison all import,
    so a transcription of it cannot drift between the four consumers.

    The pin is the assembled floor -- coefficient AND mass together -- and
    not the bare coefficient, because the defect class it guards is their
    PRODUCT: ``8/mu`` is identically ``16/m_He`` for an equal-mass pair, so
    a coefficient pin alone passes the two-Maxwellian (reduced-mass) form
    unchanged. Pinning the drift-free mean relative speed separates them by
    ``sqrt(2)``. Which consumer actually calls the helper is C5(a)'s job in
    ``scripts/verify/verify_sim1d_k2_dvm.py``; this case pins the number.
    """
    from cablp.solvers._sim1d.physics.kinetic_neutrals import (
        ion_thermal_g_eff_floor_cm2_s2 as _gf,
    )
    from cablp.constants import ev_to_erg as _gf_ev, m_He_cgs as _gf_m

    # The drift-free mean speed of the ion Maxwellian at Ti = 1 eV, with the
    # repo's m_He: sqrt(8 k Ti / (pi m_He)) [cm/s].
    _gf_speed = float(np.sqrt(_gf(1.0)))
    assert np.isclose(_gf_speed, 783482.7390046517, rtol=1e-12, atol=0.0), (
        _gf_speed
    )
    # The form this must NOT be: the reduced mass mu = m_He/2 folds in a
    # second thermal spread that no consumer's projectile has.
    _gf_mu_speed = float(np.sqrt(_gf(1.0, 0.5 * _gf_m)))
    assert np.isclose(
        _gf_mu_speed, 1108011.9153855983, rtol=1e-12, atol=0.0
    ), _gf_mu_speed
    assert np.isclose(
        _gf_mu_speed / _gf_speed, np.sqrt(2.0), rtol=1e-12, atol=0.0
    )

    # The explicit-argument call (what jet_carrier makes, passing its own
    # two constants) is bit-identical to the default one.
    assert _gf(1.0, _gf_m, _gf_ev) == _gf(1.0)
    # Linear in Ti, and shape-preserving on an array argument.
    _gf_Ti = np.array([0.25, 1.0, 4.0])
    assert np.allclose(_gf(_gf_Ti), _gf_Ti * _gf(1.0), rtol=1e-12, atol=0.0)


# --------------------------------------------------------------------
# ionization-birth-neutral-temperature
# --------------------------------------------------------------------
@_case("ionization-birth-neutral-temperature")
def _case_ionization_birth_neutral_temperature():
    # ONE EVENT, BOOKED TWICE. At ionization the En sink removes the local
    # (3/2) k Tn per consumed atom while the Ei birth adds (3/2) k T_birth per
    # born ion; the pair conserves energy only at T_birth = Tn. This block pins
    # the option that makes them agree ("neutral"), the diagnostic rows that
    # DISCLOSE the gap when they do not, and the no-En fallback.
    from cablp.solvers._sim1d.results.io import save_result_hdf5 as _nb_save
    from cablp.solvers._sim1d.core.deprecations import deprecation_messages

    # (a) THE SHIPPED DEFAULT IS THE CONSERVING BIRTH (adopted 2026-08-23,
    # with the C_R one-knob re-trim; golden recaptured in the same change).
    _nb_params, _nb_flags = default_config()
    assert _nb_params["Ti_birth_ionization"] == "neutral"
    assert _nb_flags["neutral_energy"] is True
    # Neither the default nor a numeric arm warns.
    _nb_dep = dict(_nb_params)
    assert deprecation_messages(_nb_dep, _nb_flags) == []
    _nb_dep["Ti_birth_ionization"] = 0.5
    assert not [
        _m for _m in deprecation_messages(_nb_dep, _nb_flags)
        if _m.startswith("Ti_birth_ionization=")
    ]

    # The selector refuses what it does not implement.
    for _nb_key, _nb_bad in (
        ("Ti_birth_ionization", "wall"),
        ("Ti_birth_ionization", "floor"),
        ("Ti_birth_ionization", "local"),
    ):
        try:
            LAPDSim1D(dict(_nb_params, **{_nb_key: _nb_bad}), dict(_nb_flags))
        except ValueError as _nb_exc:
            assert _nb_key in str(_nb_exc), str(_nb_exc)
        else:
            raise AssertionError(f"{_nb_key}={_nb_bad!r} must be refused")

    def _nb_build(_birth):
        # A jet-hot source region without paying for the run that makes one:
        # the state's own En is scaled where the jet deposits, so Tn there is
        # ~10 eV against a 300 K ion floor and the two bookings visibly part.
        _sim = LAPDSim1D(
            dict(_nb_params, Ti_birth_ionization=_birth), dict(_nb_flags)
        )
        _sim.run(t_end=3.0e-10, dt=1.0e-10)
        _sim._circuit_I_loop = 800.0   # arm the beam so its birth row is live
        _hot = _sim.state
        _hot.En[1:6] *= 400.0
        _sim._y[:] = pack_state(_hot)   # .state unpacks a COPY; write it back
        return _sim

    # "floor": a numeric arm AT the ion temperature floor, which parts from
    # the neutral temperature wherever the gas is hotter than that floor.
    _nb_sims = {}
    _nb_Ti_floor = None
    for _nb_birth in ("neutral", "floor"):
        _nb_sim = _nb_build(
            "neutral" if _nb_birth == "neutral" else _nb_Ti_floor
        )
        if _nb_birth == "neutral":
            _nb_Ti_floor = float(_nb_sim.floors["Ti"])
        _nb_sims[_nb_birth] = _nb_sim
        _nb_terms = _nb_sim.rhs_terms()
        _nb_state = _nb_sim.state
        _nb_rows = _nb_sim._birth_deficit_diagnostics
        _nb_Tn = neutral_temperature_eV(
            _nb_state, floors=_nb_sim.floors, Tn_eV=np.nan
        )
        assert np.max(_nb_Tn) > 5.0, np.max(_nb_Tn)   # the hot region is hot
        _nb_Vp = _nb_sim.geometry.plasma_volume_cm3
        # En rides nn's volume; put both bookings on the plasma volume, which
        # is the one Ei lives on, so the two are directly comparable in watts.
        _nb_V_En = _nb_Vp / neutral_energy_volume_ratio(
            _nb_state, _nb_sim.geometry
        )
        _nb_Ti_birth = (
            _nb_Tn
            if _nb_birth == "neutral"
            else np.full_like(_nb_Tn, _nb_sim.floors["Ti"])
        )
        # The summed row is exactly the three per-site rows.
        assert np.array_equal(
            _nb_rows[IONIZATION_BIRTH_DEFICIT_DIAGNOSTIC_FIELDS[0]],
            sum(
                _nb_rows[_name]
                for _name in IONIZATION_BIRTH_DEFICIT_DIAGNOSTIC_FIELDS[1:]
            ),
        )
        _nb_live = 0
        for _nb_term_name, _nb_field in zip(
            (
                "ionization_birth",
                "beam_ionization_birth",
                "gas_puff_local_ionization",
            ),
            IONIZATION_BIRTH_DEFICIT_DIAGNOSTIC_FIELDS[1:],
        ):
            _nb_term = _nb_terms[_nb_term_name]
            _nb_S = np.asarray(_nb_term.n, dtype=float)
            _nb_en_W = np.asarray(_nb_term.En, dtype=float) * _nb_V_En * 1.0e-7
            _nb_ei_W = 1.5 * ev_to_erg * _nb_Ti_birth * _nb_S * _nb_Vp * 1.0e-7
            _nb_scale = np.abs(_nb_en_W) + np.abs(_nb_ei_W)
            _nb_scale = np.where(_nb_scale > 0.0, _nb_scale, 1.0)
            # (c) THE ROW IS THE GAP: deficit = -(En sink + Ei birth thermal),
            # per cell, whichever selector is in force.
            assert np.all(
                np.abs(_nb_rows[_nb_field] * _nb_Vp + _nb_en_W + _nb_ei_W)
                <= 1.0e-12 * _nb_scale
            ), (_nb_birth, _nb_term_name)
            _nb_hot = (_nb_Tn > _nb_sim.floors["Ti"]) & (_nb_S > 0.0)
            if _nb_birth == "neutral":
                # (b) THE PAIR CLOSES. Booked at the neutral temperature the
                # sink debits, the two sides cancel to roundoff per cell...
                assert np.all(
                    np.abs(_nb_en_W + _nb_ei_W) <= 1.0e-12 * _nb_scale
                ), _nb_term_name
                # ...and the disclosure row reads zero there.
                assert np.all(
                    np.abs(_nb_rows[_nb_field]) * _nb_Vp <= 1.0e-12 * _nb_scale
                ), _nb_term_name
            else:
                # (c) ...and at the floor it is POSITIVE wherever the gas is
                # hotter than the ion floor: energy leaving the model.
                assert np.all(_nb_rows[_nb_field][_nb_hot] > 0.0), _nb_term_name
            if _nb_hot.any():
                _nb_live += 1
        # Both the bulk and the beam channel were actually exercised.
        assert _nb_live >= 2, _nb_live

    # The two selectors are not the same run: "floor" really does delete power
    # here, so the block above is not vacuous.
    _nb_floor_total = _nb_sims["floor"]._birth_deficit_diagnostics[
        IONIZATION_BIRTH_DEFICIT_DIAGNOSTIC_FIELDS[0]
    ]
    assert (
        np.sum(_nb_floor_total * _nb_sims["floor"].geometry.plasma_volume_cm3)
        > 0.1
    )

    # The rows are ADDITIVE state on the artifact: they round-trip through
    # sim1d-hdf5-v1 unchanged.
    _nb_result = _nb_sims["floor"].run(t_end=4.0e-10, dt=1.0e-10)
    with tempfile.TemporaryDirectory() as _nb_dir:
        _nb_path = Path(_nb_dir) / "birth_deficit.h5"
        _nb_save(_nb_path, _nb_result)
        _nb_loaded = load_result_hdf5(_nb_path)
        for _nb_field in IONIZATION_BIRTH_DEFICIT_DIAGNOSTIC_FIELDS:
            assert np.array_equal(
                getattr(_nb_loaded, _nb_field), getattr(_nb_result, _nb_field)
            ), _nb_field

    # (d) NO En FIELD, NO LOCAL NEUTRAL TEMPERATURE: the birth falls back to
    # the cold-gas scalar Tn_K, bit-for-bit the numeric selector at that value.
    _nb_off_p, _nb_off_f = _pin_pre_r2a_neutral_stance(*default_config())
    _nb_TnK_eV = float(_nb_off_p.get("Tn_K", 300.0)) * kb_cgs / ev_to_erg
    _nb_off_neutral = LAPDSim1D(
        dict(_nb_off_p, Ti_birth_ionization="neutral"), dict(_nb_off_f)
    )
    _nb_off_numeric = LAPDSim1D(
        dict(_nb_off_p, Ti_birth_ionization=_nb_TnK_eV), dict(_nb_off_f)
    )
    assert _nb_off_neutral.state.En is None
    assert np.array_equal(
        _nb_off_neutral.rhs_terms()["ionization_birth"].Ei,
        _nb_off_numeric.rhs_terms()["ionization_birth"].Ei,
    )
    # With no En field there is no sink to pair with, so no rows are recorded.
    assert _nb_off_neutral._birth_deficit_diagnostics == {}


@_case("neutral-retired-keys-refuse")
def _case_neutral_retired_keys_refuse():
    # The neutral, gas and ion-neutral experiment keys removed with the
    # closures they served. Each is gone from its template, is on the retired
    # register of ITS OWN namespace, and a configuration naming it -- at any
    # value, the old default included -- is refused at construction with the
    # key named as RETIRED. A retired name filed in the OTHER namespace reads
    # as the plain unknown key it is there.
    from cablp.solvers._sim1d.core.config import (
        RETIRED_FLAG_KEYS,
        RETIRED_PARAM_KEYS,
        input_dict_template_1d,
        input_flags_template_1d,
    )

    _nr_params = {
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
        "neutral_wall_partition_sigma_hehe_cm2": None,
        "neutral_knudsen_temperature": "frozen",
        "neutral_momentum_radial": "uniform",
        "neutral_kinetic_refresh_s": 5e-4,
        "neutral_kinetic_refresh_tol": 0.2,
        "neutral_kinetic_nvz": 48,
        "neutral_kinetic_nvp": 12,
        "ion_neutral_drag_model": "constant",
        "b_ion_neutral_thermalization": None,
        "coverage_growth_rate_per_s": 1390.0,
        "coverage_backfill_time_s": 3.0e-5,
        "coverage_initial_fraction": None,
        "coverage_initial_profile": None,
        # Adopted selections whose keys had one legal value left, and the
        # puff shapes and waveforms they retired.
        "neutral_exchange_model": "knudsen",
        "neutral_exchange_coeff_cm3_s": 1.0e5,
        "neutral_kinetic_dvm_elastic": "phelps_iso",
        "neutral_kinetic_dvm_wall_reflection": "diffuse_elastic",
        "gas_puff_mode": "square",
        "gas_puff_profile": "orifice",
        "gas_puff_sigma_cm": 50.0,
        "gas_puff_throw_cm": 100.0,
        "gas_puff_local_ionization_fraction": 0.0,
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
    _nr_flags = {
        "end_recycle_to_annulus": False,
        "neutral_hot_birth_drift": False,
        "neutral_probe_source": False,
        "neutral_wall_momentum_partition": False,
        "ion_neutral_drag_cx_only": False,
        "ion_neutral_thermalization": False,
        "coverage_closure": False,
        # Folded into initial_neutral_state.
        "neutral_equilibration": True,
        "launch_plasma_after_equilibration": True,
        "neutral_initial_profile": False,
        # Adopted: the behaviour is unconditional.
        "cx": True,
        "ion_neutral_drag": True,
        "ion_neutral_moment_closure": True,
        "neutral_two_zone": True,
        "neutral_kinetic_dvm_baffles": False,
        "neutral_prebreakdown": True,
    }
    _nr_base_p, _nr_base_f = default_config()
    for _nr_key, _nr_value in _nr_params.items():
        assert _nr_key not in input_dict_template_1d, _nr_key
        assert _nr_key not in input_flags_template_1d, _nr_key
        assert _nr_key in RETIRED_PARAM_KEYS, _nr_key
        try:
            LAPDSim1D(dict(_nr_base_p, **{_nr_key: _nr_value}), _nr_base_f)
        except ValueError as _nr_exc:
            assert f"{_nr_key} is RETIRED" in str(_nr_exc), str(_nr_exc)
        else:
            raise AssertionError(f"retired params key {_nr_key} ACCEPTED")
    for _nr_key, _nr_value in _nr_flags.items():
        assert _nr_key not in input_dict_template_1d, _nr_key
        assert _nr_key not in input_flags_template_1d, _nr_key
        assert _nr_key in RETIRED_FLAG_KEYS, _nr_key
        try:
            LAPDSim1D(_nr_base_p, dict(_nr_base_f, **{_nr_key: _nr_value}))
        except ValueError as _nr_exc:
            assert f"{_nr_key} is RETIRED" in str(_nr_exc), str(_nr_exc)
        else:
            raise AssertionError(f"retired flags key {_nr_key} ACCEPTED")
    # A retired FLAG name in params is a misfiled key, not a retired one.
    try:
        LAPDSim1D(dict(_nr_base_p, coverage_closure=False), _nr_base_f)
    except ValueError as _nr_exc:
        assert "unknown LAPDSim1D configuration keys" in str(_nr_exc)
        assert "RETIRED" not in str(_nr_exc), str(_nr_exc)
    else:
        raise AssertionError("a misfiled retired flag name was ACCEPTED")
    # The removed selector VALUE of the surviving neutral_model selector is
    # refused, and the refusal states what the selector accepts.
    try:
        LAPDSim1D(dict(_nr_base_p, neutral_model="kinetic"), _nr_base_f)
    except ValueError as _nr_exc:
        assert "neutral_model must be 'moment' or 'kinetic_dvm'" in str(
            _nr_exc
        ), str(_nr_exc)
    else:
        raise AssertionError("neutral_model='kinetic' ACCEPTED")
