import numpy as np

from cablp.funcs._heat import elec_par_heat_div, ion_par_heat_div
from cablp.funcs._plasmaparams import c_log
from cablp.solvers._sim1d import LAPDSim1D, default_config
from cablp.solvers._sim1d.conduction import (
    heat_conduction_rhs,
    implicit_heat_conduction_step,
)
from cablp.solvers._sim1d.energy import (
    electron_cooling_rhs,
    electron_ion_exchange_rhs,
    ion_charge_exchange_rhs,
)
from cablp.solvers._sim1d.flux import front_filling_fluxes
from cablp.solvers._sim1d.integrator import ssprk2_step
from cablp.solvers._sim1d.neutrals import (
    neutral_exchange_coefficients,
    neutral_inventory_rate,
    puff_rate,
    pump_rate,
)
from cablp.solvers._sim1d.reactions import (
    particle_inventory_rate,
    reaction_rates,
)
from cablp.solvers._sim1d.sources import velocity_divergence
from cablp.solvers._sim1d.state import (
    conservative_from_primitives,
    derive_state,
    pack_state,
    unpack_state,
)
from cablp.vars._cons import en_factor, ev_to_erg


def main():
    params, flags = default_config()
    sim = LAPDSim1D(params, flags)
    snapshot = sim.get_initial_snapshot()
    geom = snapshot.geometry
    state = snapshot.state
    derived = snapshot.derived

    assert geom.cells == params["nx"] + 2
    assert geom.length_cm.shape == (geom.cells,)
    assert geom.plasma_volume_cm3.shape == (geom.cells,)
    assert geom.neutral_volume_cm3.shape == (geom.cells,)
    assert geom.plasma_face_area_cm2.shape == (geom.cells + 1,)
    assert geom.neutral_face_area_cm2.shape == (geom.cells + 1,)
    assert geom.center_distance_cm.shape == (geom.cells - 1,)
    assert np.isclose(geom.z_edges_cm[0], 0.0)
    assert np.isclose(geom.z_edges_cm[-1], params["Lm"])
    assert geom.cell_role[0] == "source"
    assert geom.cell_role[-1] == "end"
    assert np.all(geom.cell_role[1:-1] == "domain")
    assert np.all(geom.plasma_volume_cm3 > 0.0)
    assert np.all(geom.neutral_volume_cm3 > geom.plasma_volume_cm3)

    for values in (
        state.n,
        state.nn,
        state.M,
        state.Ee,
        state.Ei,
        derived.u,
        derived.Te,
        derived.Ti,
        derived.pe,
        derived.pi,
        derived.p,
    ):
        assert np.all(np.isfinite(values))

    neutral_coeff = sim.neutral_exchange_coefficients()
    assert neutral_coeff.shape == (geom.cells - 1,)
    assert np.all(np.isfinite(neutral_coeff))
    assert np.all(neutral_coeff >= 0.0)
    assert np.any(neutral_coeff > 0.0)

    constant_coeff = neutral_exchange_coefficients(
        geometry=geom,
        model="constant",
        constant_coeff_cm3_s=params["neutral_exchange_coeff_cm3_s"],
        Tn_K=params["Tn_K"],
        mu_neutral=4,
        clausing_scale=params["neutral_clausing_scale"],
    )
    assert constant_coeff.shape == (geom.cells - 1,)
    assert np.allclose(constant_coeff, params["neutral_exchange_coeff_cm3_s"])

    dt_default = sim.suggest_timestep()
    assert np.isfinite(dt_default.dt)
    assert dt_default.dt > 0.0
    assert dt_default.dt <= params["dt_max"]
    assert dt_default.dt >= params["dt_min"]
    assert dt_default.active_constraint in {
        "plasma_cfl",
        "front_density",
        "neutral_exchange",
        "neutral_sources",
        "reactions",
        "energy_exchange",
        "electron_cooling",
        "ion_charge_exchange",
        "heat_conduction",
        "dt_max",
        "dt_min",
    }
    assert np.isfinite(dt_default.dt_neutral_sources)
    assert dt_default.dt_reactions > 0.0
    assert dt_default.dt_energy_exchange > 0.0
    assert dt_default.dt_electron_cooling > 0.0
    assert dt_default.dt_ion_charge_exchange > 0.0
    assert dt_default.dt_heat_conduction > 0.0

    rhs = sim.plasma_flux_rhs(include_front=False)
    for values in (rhs.n, rhs.nn, rhs.M, rhs.Ee, rhs.Ei):
        assert np.allclose(values, 0.0, atol=1e-20)
    pressure_rhs = sim.pressure_work_rhs()
    for values in (
        pressure_rhs.n,
        pressure_rhs.nn,
        pressure_rhs.M,
        pressure_rhs.Ee,
        pressure_rhs.Ei,
    ):
        assert np.allclose(values, 0.0, atol=1e-20)
    neutral_rhs = sim.neutral_exchange_rhs()
    for values in (
        neutral_rhs.n,
        neutral_rhs.nn,
        neutral_rhs.M,
        neutral_rhs.Ee,
        neutral_rhs.Ei,
    ):
        assert np.allclose(values, 0.0, atol=1e-20)
    source_rhs = sim.neutral_source_sink_rhs()
    assert source_rhs.nn[0] > 0.0
    assert source_rhs.nn[-1] < 0.0
    assert np.isclose(
        source_rhs.nn[0],
        puff_rate(params["S_gp"], params["gas_puff_valves"], geom.neutral_volume_cm3[0])
        - pump_rate(params["S_pump_L"], geom.neutral_volume_cm3[0]) * state.nn[0],
    )
    assert np.isclose(
        source_rhs.nn[-1],
        -pump_rate(params["S_pump_R"], geom.neutral_volume_cm3[-1]) * state.nn[-1],
    )
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
    reaction_inventory_scale = np.sum(
        np.abs(reaction_rhs.n * geom.plasma_volume_cm3)
        + np.abs(reaction_rhs.nn * geom.neutral_volume_cm3)
    )
    reaction_inventory_tol = 1e-12 * reaction_inventory_scale
    assert np.isclose(
        particle_inventory_rate(reaction_rhs, geom),
        0.0,
        atol=reaction_inventory_tol,
    )
    S_ion_off, _, _ = reaction_rates(
        state=state,
        floors=sim.floors,
        ion_mass_g=sim.ion_mass_g,
        gas_type=params["gas_type"],
        I_ion=sim.I_ion,
        b_ioniz=0.0,
        b_rec_rad=params["b_rec_rad"],
        b_rec_3b=params["b_rec_3b"],
    )
    assert np.allclose(S_ion_off, 0.0)

    recomb_state = conservative_from_primitives(
        n=np.full(geom.cells, params["ne0"]),
        nn=state.nn,
        u=np.full(geom.cells, 1.0e5),
        Te=np.full(geom.cells, 1.0),
        Ti=np.full(geom.cells, 1.0),
        ion_mass_g=sim.ion_mass_g,
    )
    recomb_params = dict(params)
    recomb_params["b_ioniz"] = 0.0
    recomb_sim = LAPDSim1D(recomb_params, flags)
    recomb_rhs = recomb_sim.reaction_rhs(state=recomb_state)
    assert np.all(recomb_rhs.M < 0.0)
    assert np.all(recomb_rhs.Ee < 0.0)
    assert np.all(recomb_rhs.Ei < 0.0)

    density_ramp = np.linspace(2.0, 1.0, geom.cells) * params["ne0"]
    ramp_state = conservative_from_primitives(
        n=density_ramp,
        nn=state.nn,
        u=np.zeros(geom.cells),
        Te=np.full(geom.cells, params["Te0"]),
        Ti=np.full(geom.cells, params["Ti0"]),
        ion_mass_g=sim.ion_mass_g,
    )
    ramp_front = front_filling_fluxes(
        state=ramp_state,
        floors=sim.floors,
        ion_mass_g=sim.ion_mass_g,
        mu=sim.mu,
        geometry=geom,
        alpha_front=params["alpha_front"],
    )
    assert np.all(ramp_front.n[1:-1] > 0.0)
    ramp_rhs = sim.plasma_flux_rhs(y=pack_state(ramp_state), include_front=True)
    for values in (ramp_rhs.n, ramp_rhs.nn, ramp_rhs.M, ramp_rhs.Ee, ramp_rhs.Ei):
        assert np.all(np.isfinite(values))

    nn_ramp_state = conservative_from_primitives(
        n=state.n,
        nn=np.linspace(2.0, 1.0, geom.cells) * state.nn[0],
        u=np.zeros(geom.cells),
        Te=np.full(geom.cells, params["Te0"]),
        Ti=np.full(geom.cells, params["Ti0"]),
        ion_mass_g=sim.ion_mass_g,
    )
    nn_ramp_rhs = sim.neutral_exchange_rhs(state=nn_ramp_state)
    assert nn_ramp_rhs.nn[0] < 0.0
    assert nn_ramp_rhs.nn[-1] > 0.0
    inventory_terms = nn_ramp_rhs.nn * geom.neutral_volume_cm3
    inventory_tol = 1e-12 * np.sum(np.abs(inventory_terms))
    assert np.isclose(
        neutral_inventory_rate(nn_ramp_rhs, geom), 0.0, atol=inventory_tol
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
        u=np.zeros(geom.cells),
        Te=np.full(geom.cells, 0.5),
        Ti=np.full(geom.cells, 2.0),
        ion_mass_g=sim.ion_mass_g,
    )
    hot_i_exchange = sim.energy_exchange_rhs(state=hot_i_state)
    assert np.all(hot_i_exchange.Ee > 0.0)
    assert np.all(hot_i_exchange.Ei < 0.0)
    assert np.allclose(hot_i_exchange.Ee + hot_i_exchange.Ei, 0.0)

    equal_temp_exchange = sim.energy_exchange_rhs()
    assert np.allclose(equal_temp_exchange.Ee, 0.0, atol=1e-30)
    assert np.allclose(equal_temp_exchange.Ei, 0.0, atol=1e-30)
    disabled_exchange = electron_ion_exchange_rhs(
        state=hot_e_state,
        floors=sim.floors,
        ion_mass_g=sim.ion_mass_g,
        mu=sim.mu,
        b_Qie=0.0,
        ln_lambda_min=params["ln_lambda_min"],
    )
    assert np.allclose(disabled_exchange.Ee, 0.0)
    assert np.allclose(disabled_exchange.Ei, 0.0)

    cooling_state = conservative_from_primitives(
        n=np.full(geom.cells, 1.0e12),
        nn=np.full(geom.cells, 1.0e12),
        u=np.zeros(geom.cells),
        Te=np.full(geom.cells, 10.0),
        Ti=np.full(geom.cells, 1.0),
        ion_mass_g=sim.ion_mass_g,
    )
    cooling_rhs = sim.electron_cooling_rhs(state=cooling_state)
    assert np.all(cooling_rhs.Ee < 0.0)
    assert np.allclose(cooling_rhs.n, 0.0)
    assert np.allclose(cooling_rhs.nn, 0.0)
    assert np.allclose(cooling_rhs.M, 0.0)
    assert np.allclose(cooling_rhs.Ei, 0.0)
    cooling_dt = sim.suggest_timestep(y=pack_state(cooling_state))
    assert np.isfinite(cooling_dt.dt_electron_cooling)

    ionization_only_cooling = electron_cooling_rhs(
        state=cooling_state,
        floors=sim.floors,
        ion_mass_g=sim.ion_mass_g,
        gas_type=params["gas_type"],
        I_ion=sim.I_ion,
        b_ioniz=params["b_ioniz"],
        b_rec_rad=params["b_rec_rad"],
        b_rec_3b=params["b_rec_3b"],
        b_ionization_energy_cost=params["b_ionization_energy_cost"],
        b_Qei=0.0,
        b_Qen=0.0,
        ionization_energy_cost=True,
        icool=True,
        ncool=True,
        icool_recomb=flags["icool_recomb"],
    )
    assert np.all(ionization_only_cooling.Ee < 0.0)

    disabled_cooling = electron_cooling_rhs(
        state=cooling_state,
        floors=sim.floors,
        ion_mass_g=sim.ion_mass_g,
        gas_type=params["gas_type"],
        I_ion=sim.I_ion,
        b_ioniz=params["b_ioniz"],
        b_rec_rad=params["b_rec_rad"],
        b_rec_3b=params["b_rec_3b"],
        b_ionization_energy_cost=0.0,
        b_Qei=0.0,
        b_Qen=0.0,
        ionization_energy_cost=True,
        icool=True,
        ncool=True,
        icool_recomb=flags["icool_recomb"],
    )
    assert np.allclose(disabled_cooling.Ee, 0.0)
    assert np.allclose(disabled_cooling.Ei, 0.0)

    hot_ion_cx_state = conservative_from_primitives(
        n=np.full(geom.cells, 1.0e12),
        nn=np.full(geom.cells, 1.0e12),
        u=np.zeros(geom.cells),
        Te=np.full(geom.cells, 1.0),
        Ti=np.full(geom.cells, 10.0),
        ion_mass_g=sim.ion_mass_g,
    )
    hot_ion_cx = sim.ion_charge_exchange_rhs(state=hot_ion_cx_state)
    assert np.all(hot_ion_cx.Ei < 0.0)
    assert np.allclose(hot_ion_cx.n, 0.0)
    assert np.allclose(hot_ion_cx.nn, 0.0)
    assert np.allclose(hot_ion_cx.M, 0.0)
    assert np.allclose(hot_ion_cx.Ee, 0.0)
    hot_ion_cx_dt = sim.suggest_timestep(y=pack_state(hot_ion_cx_state))
    assert np.isfinite(hot_ion_cx_dt.dt_ion_charge_exchange)

    warm_neutral_cx = ion_charge_exchange_rhs(
        state=hot_ion_cx_state,
        floors=sim.floors,
        ion_mass_g=sim.ion_mass_g,
        gas_type=params["gas_type"],
        Tn_fit=20.0,
        b_Qcx=params["b_Qcx"],
        cx=True,
    )
    assert np.all(warm_neutral_cx.Ei > 0.0)

    disabled_cx = ion_charge_exchange_rhs(
        state=hot_ion_cx_state,
        floors=sim.floors,
        ion_mass_g=sim.ion_mass_g,
        gas_type=params["gas_type"],
        Tn_fit=params["Tn_fit"],
        b_Qcx=0.0,
        cx=True,
    )
    assert np.allclose(disabled_cx.Ei, 0.0)

    heat_state = conservative_from_primitives(
        n=np.full(geom.cells, 1.0e12),
        nn=state.nn,
        u=np.zeros(geom.cells),
        Te=np.linspace(2.0, 1.0, geom.cells),
        Ti=np.linspace(1.5, 0.5, geom.cells),
        ion_mass_g=sim.ion_mass_g,
    )
    heat_rhs = sim.heat_conduction_rhs(state=heat_state)
    for values in (heat_rhs.n, heat_rhs.nn, heat_rhs.M):
        assert np.allclose(values, 0.0)
    assert np.all(np.isfinite(heat_rhs.Ee))
    assert np.all(np.isfinite(heat_rhs.Ei))
    assert heat_rhs.Ee[0] < 0.0
    assert heat_rhs.Ee[-1] > 0.0
    assert heat_rhs.Ei[0] < 0.0
    assert heat_rhs.Ei[-1] > 0.0
    heat_energy_tol = 1e-12 * np.sum(
        np.abs(heat_rhs.Ee * geom.plasma_volume_cm3)
    )
    assert np.isclose(
        np.sum(heat_rhs.Ee * geom.plasma_volume_cm3),
        0.0,
        atol=heat_energy_tol,
    )
    heat_ion_energy_tol = 1e-12 * np.sum(
        np.abs(heat_rhs.Ei * geom.plasma_volume_cm3)
    )
    assert np.isclose(
        np.sum(heat_rhs.Ei * geom.plasma_volume_cm3),
        0.0,
        atol=heat_ion_energy_tol,
    )
    heat_dt = sim.suggest_timestep(y=pack_state(heat_state))
    assert np.isfinite(heat_dt.dt_heat_conduction)
    heat_derived = derive_state(heat_state, sim.floors, sim.ion_mass_g)
    heat_ln_lambda = np.maximum(
        c_log(heat_derived.Te, heat_state.n, kind="ei"),
        params["ln_lambda_min"],
    )
    dTe_dt = heat_rhs.Ee / (1.5 * heat_state.n * ev_to_erg)
    dTi_dt = heat_rhs.Ei / (1.5 * heat_state.n * ev_to_erg)
    legacy_dTe_dt = en_factor * elec_par_heat_div(
        heat_derived.Te,
        heat_state.n,
        geom.length_cm,
        heat_ln_lambda,
    )
    legacy_dTi_dt = en_factor * ion_par_heat_div(
        heat_derived.Ti,
        heat_state.n,
        geom.length_cm,
        sim.mu,
        heat_ln_lambda,
    )
    assert np.allclose(dTe_dt, legacy_dTe_dt)
    assert np.allclose(dTi_dt, legacy_dTi_dt)

    disabled_heat = heat_conduction_rhs(
        state=heat_state,
        floors=sim.floors,
        ion_mass_g=sim.ion_mass_g,
        mu=sim.mu,
        geometry=geom,
        b_epara=0.0,
        b_ipara=0.0,
        heat_conduction=True,
        ln_lambda_min=params["ln_lambda_min"],
    )
    assert np.allclose(disabled_heat.Ee, 0.0)
    assert np.allclose(disabled_heat.Ei, 0.0)

    implicit_heat_state = sim.implicit_heat_conduction_step(
        dt=heat_dt.dt_heat_conduction,
        state=heat_state,
    )
    implicit_heat_derived = derive_state(
        implicit_heat_state, sim.floors, sim.ion_mass_g
    )
    assert np.all(np.isfinite(implicit_heat_state.Ee))
    assert np.all(np.isfinite(implicit_heat_state.Ei))
    assert np.all(implicit_heat_derived.Te >= params["Te_floor"])
    assert np.all(implicit_heat_derived.Ti >= params["Ti_floor"])
    assert implicit_heat_derived.Te[0] < heat_derived.Te[0]
    assert implicit_heat_derived.Te[-1] > heat_derived.Te[-1]
    assert implicit_heat_derived.Ti[0] < heat_derived.Ti[0]
    assert implicit_heat_derived.Ti[-1] > heat_derived.Ti[-1]
    assert np.isclose(
        np.sum((implicit_heat_state.Ee - heat_state.Ee) * geom.plasma_volume_cm3),
        0.0,
        atol=heat_energy_tol,
    )
    assert np.isclose(
        np.sum((implicit_heat_state.Ei - heat_state.Ei) * geom.plasma_volume_cm3),
        0.0,
        atol=heat_ion_energy_tol,
    )

    small_dt = min(1.0e-11, 1.0e-4 * heat_dt.dt_heat_conduction)
    small_implicit = sim.implicit_heat_conduction_step(dt=small_dt, state=heat_state)
    assert np.allclose(
        (small_implicit.Ee - heat_state.Ee) / small_dt,
        heat_rhs.Ee,
        rtol=5.0e-4,
        atol=1.0e-8,
    )
    assert np.allclose(
        (small_implicit.Ei - heat_state.Ei) / small_dt,
        heat_rhs.Ei,
        rtol=5.0e-4,
        atol=1.0e-8,
    )

    disabled_implicit = implicit_heat_conduction_step(
        state=heat_state,
        floors=sim.floors,
        ion_mass_g=sim.ion_mass_g,
        mu=sim.mu,
        geometry=geom,
        dt=heat_dt.dt_heat_conduction,
        b_epara=0.0,
        b_ipara=0.0,
        heat_conduction=True,
        ln_lambda_min=params["ln_lambda_min"],
    )
    assert np.allclose(disabled_implicit.Ee, heat_state.Ee)
    assert np.allclose(disabled_implicit.Ei, heat_state.Ei)

    nonheat_rhs = sim.rhs(pack_state(heat_state), include_heat_conduction=False)
    full_rhs = sim.rhs(pack_state(heat_state), include_heat_conduction=True)
    heat_rhs_y = pack_state(heat_rhs)
    assert np.allclose(full_rhs - nonheat_rhs, heat_rhs_y)

    split_dt = min(1.0e-10, 0.1 * heat_dt.dt_heat_conduction)
    manual_explicit_y = ssprk2_step(
        y0=pack_state(heat_state),
        dt=split_dt,
        rhs_func=lambda yy: sim.rhs(yy, include_heat_conduction=False),
        floor_func=sim.floor_state_vector,
    )
    manual_heat_state = sim.implicit_heat_conduction_step(
        dt=split_dt,
        y=manual_explicit_y,
    )
    manual_split_y = sim.floor_state_vector(pack_state(manual_heat_state))
    split_y = sim.operator_split_step(y=pack_state(heat_state), dt=split_dt)
    assert np.allclose(split_y, manual_split_y)

    no_heat_bound_dt = sim.suggest_timestep(
        y=pack_state(heat_state),
        include_heat_conduction=False,
    )
    assert np.isinf(no_heat_bound_dt.dt_heat_conduction)
    assert no_heat_bound_dt.active_constraint != "heat_conduction"

    fast_state = conservative_from_primitives(
        n=np.full(geom.cells, params["ne0"]),
        nn=state.nn,
        u=np.full(geom.cells, 1.0e7),
        Te=np.full(geom.cells, params["Te0"]),
        Ti=np.full(geom.cells, params["Ti0"]),
        ion_mass_g=sim.ion_mass_g,
    )
    fast_dt = sim.suggest_timestep(y=pack_state(fast_state))
    assert fast_dt.dt_plasma_cfl < dt_default.dt_plasma_cfl

    no_source_params = dict(params)
    no_source_params["gas_puff_enabled"] = False
    no_source_params["pump_enabled"] = False
    no_source_params["b_ioniz"] = 0.0
    no_source_params["b_rec_rad"] = 0.0
    no_source_params["b_rec_3b"] = 0.0
    no_source_params["b_Qei"] = 0.0
    no_source_params["b_Qen"] = 0.0
    no_source_params["b_Qcx"] = 0.0
    no_source_params["b_ionization_energy_cost"] = 0.0
    no_source_sim = LAPDSim1D(no_source_params, flags)
    y_before = no_source_sim.get_initial_snapshot().y.copy()
    stationary_after = no_source_sim.advance_one_step(1e-10)
    assert np.allclose(stationary_after.y, y_before, rtol=0.0, atol=1e-20)

    split_flags = dict(flags)
    split_flags["implicit_heat_conduction"] = True
    no_source_split_sim = LAPDSim1D(no_source_params, split_flags)
    split_before = no_source_split_sim.get_initial_snapshot().y.copy()
    split_stationary_after = no_source_split_sim.advance_one_step(1e-10)
    assert np.allclose(split_stationary_after.y, split_before, rtol=0.0, atol=1e-20)

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
        assert np.all(values >= 0.0)
    ramp_after = sim.plasma_flux_rhs(y=ramp_y1, include_front=True)
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

    print(
        "sim1d smoke ok: "
        f"cells={geom.cells}, dz={geom.dz_cm:g} cm, "
        f"Vp_total={geom.plasma_volume_cm3.sum():.6e} cm^3, "
        f"Vm_total={geom.neutral_volume_cm3.sum():.6e} cm^3"
    )


if __name__ == "__main__":
    main()
