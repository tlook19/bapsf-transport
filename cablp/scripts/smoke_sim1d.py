import numpy as np

from cablp.solvers._sim1d import LAPDSim1D, default_config
from cablp.solvers._sim1d_flux import front_filling_fluxes
from cablp.solvers._sim1d_integrator import ssprk2_step
from cablp.solvers._sim1d_neutrals import neutral_inventory_rate
from cablp.solvers._sim1d_sources import velocity_divergence
from cablp.solvers._sim1d_state import (
    conservative_from_primitives,
    derive_state,
    pack_state,
    unpack_state,
)


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
    assert np.isclose(neutral_inventory_rate(nn_ramp_rhs, geom), 0.0, atol=1e-6)
    assert np.allclose(nn_ramp_rhs.n, 0.0)
    assert np.allclose(nn_ramp_rhs.M, 0.0)
    assert np.allclose(nn_ramp_rhs.Ee, 0.0)
    assert np.allclose(nn_ramp_rhs.Ei, 0.0)

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

    y_before = snapshot.y.copy()
    stationary_after = sim.advance_one_step(1e-10)
    assert np.allclose(stationary_after.y, y_before, rtol=0.0, atol=1e-20)

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
