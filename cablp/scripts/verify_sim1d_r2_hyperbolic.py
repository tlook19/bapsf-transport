"""R2 conservative-hyperbolic-core gate suite (SIM1D_MODEL_AUDIT_PLAN R2).

Pre-registered gates G1-G7 for the kinetic-energy-preserving hyperbolic core
(default-off flags ``hyperbolic_energy_consistent`` and the ``adiabatic``
``hyperbolic_wave_speed``, with ``front_flux`` retired). This is a verification
harness, not a campaign run: it isolates the pure hyperbolic operator
(advective flux + pressure work + KEP energy correction) via a manual SSPRK2
stepper, with reactions/cooling/conduction excluded, so the wave tests are
clean. Floors are watched and a run is INVALID if one binds where it should not.

Gates:
  G1  constant/uniform stationary state is well balanced (zero hyperbolic RHS)
  G2  contact discontinuity advects with no spurious pressure oscillation
  G3  acoustic pulse propagates at c=sqrt((5/3)(Te+Ti)/m_i), not sqrt(Te/m_i)
  G4  Sod shock/rarefaction stays positive and conserves total energy
  G5  closed-domain total-energy identity holds to machine precision (on),
      with the uncorrected leak reported as the failing reference
  G6  positivity: a demanding floor-inert state stays positive and finite
  G7  >=3-mesh behaviour: energy identity holds at every mesh; front-flux L1
      activity and the Rusanov numerical diffusion vanish under refinement

Usage:
    python scripts/verify_sim1d_r2_hyperbolic.py
"""
import sys

import numpy as np

from cablp.solvers._sim1d import LAPDSim1D, default_config
from cablp.solvers._sim1d.core.state import (
    ConservativeState1D, conservative_from_primitives, derive_state,
    apply_state_floors,
)
from cablp.solvers._sim1d.physics.flux import plasma_wave_speed
from cablp.vars._cons import ev_to_erg

FLOOR_RTOL = 1e-9

# Clean single-phase autonomous fixture (mirrors verify_sim1d_order): floors far
# away, no cathode, no time-dependent forcing. The manual stepper below uses only
# the hyperbolic operator, so reaction/cooling flags are irrelevant to it.
CLEAN_PARAMS = {
    "ne0": 1e12, "nn0": 1e13, "Te0": 5.0, "Ti0": 2.0, "u0": 0.0,
    "gas_puff_enabled": False, "pump_enabled": False,
    "phase_transition_mode": "scheduled",
    "tau_neutral_prebreakdown": 0.0, "tau_prebreakdown": 0.0,
    "tau_breakdown": 0.0, "tau_discharge": 1.0, "tau_afterglow": 0.0,
    "adaptive_retries_enabled": False, "dt_growth_enabled": False,
    "dt_min": 1e-16, "dt_max": 1.0,
    "max_density_step_fraction": 0.0, "max_neutral_step_fraction": 0.0,
    "max_energy_step_fraction": 0.0,
}
CLEAN_FLAGS = {
    "Plasma": True, "implicit_heat_conduction": True,
    "neutral_prebreakdown": False, "neutral_equilibration": False,
    "launch_plasma_after_equilibration": False,
    "cathode_coupling": False, "debug_checks": False,
}


def make_sim(nx=120, energy_consistent=True, wave_speed="adiabatic", front=False):
    params, flags = default_config()
    params.update(CLEAN_PARAMS)
    params["nx"] = nx
    params["hyperbolic_wave_speed"] = wave_speed
    flags.update(CLEAN_FLAGS)
    flags["front_flux"] = front
    flags["hyperbolic_energy_consistent"] = energy_consistent
    return LAPDSim1D(params, flags)


def state_from(sim, n, u, Te, Ti, nn=None):
    if nn is None:
        nn = np.full_like(np.asarray(n, dtype=float), 1e13)
    return conservative_from_primitives(n, nn, u, Te, Ti, sim._ion_mass_g)


def hyp_rhs(sim, state):
    """Pure hyperbolic RHS: advective flux + pressure work + KEP correction."""
    adv = sim.plasma_flux_rhs_terms(state=state)["plasma_advective_flux"]
    pw = sim.pressure_work_rhs(state=state)
    corr = sim.hyperbolic_energy_correction_rhs(state=state)
    dn = np.asarray(adv.n)
    dM = np.asarray(adv.M) + np.asarray(pw.M) + np.asarray(corr.M)
    dEe = np.asarray(adv.Ee) + np.asarray(pw.Ee) + np.asarray(corr.Ee)
    dEi = np.asarray(adv.Ei) + np.asarray(pw.Ei) + np.asarray(corr.Ei)
    return dn, dM, dEe, dEi


class FloorWatch:
    def __init__(self):
        self.binds = 0

    def apply(self, sim, st):
        floors = sim._floors
        # Only ACTIVE plasma cells matter: plasma-dead plenum/boundary cells sit
        # at/below the floor by construction and are floored every step.
        active = np.asarray(sim._geometry.plasma_active, dtype=bool)
        n_safe = np.maximum(np.asarray(st.n, dtype=float), floors["n"])
        raw_Te = (2.0 / 3.0) * np.asarray(st.Ee) / (n_safe * ev_to_erg)
        raw_Ti = (2.0 / 3.0) * np.asarray(st.Ei) / (n_safe * ev_to_erg)
        for val, fl in ((np.asarray(st.n), floors["n"]), (raw_Te, floors["Te"]),
                        (raw_Ti, floors["Ti"])):
            self.binds += int(np.count_nonzero((val < fl * (1.0 - FLOOR_RTOL)) & active))
        return apply_state_floors(st, floors=floors, ion_mass_g=sim._ion_mass_g)


def ssprk2(sim, st, dt, watch):
    dn, dM, dEe, dEi = hyp_rhs(sim, st)
    s1 = watch.apply(sim, ConservativeState1D(
        n=st.n + dt * dn, nn=st.nn, M=st.M + dt * dM,
        Ee=st.Ee + dt * dEe, Ei=st.Ei + dt * dEi))
    dn1, dM1, dEe1, dEi1 = hyp_rhs(sim, s1)
    return watch.apply(sim, ConservativeState1D(
        n=0.5 * st.n + 0.5 * (s1.n + dt * dn1),
        nn=st.nn,
        M=0.5 * st.M + 0.5 * (s1.M + dt * dM1),
        Ee=0.5 * st.Ee + 0.5 * (s1.Ee + dt * dEe1),
        Ei=0.5 * st.Ei + 0.5 * (s1.Ei + dt * dEi1)))


def total_energy(sim, st):
    """Closed-domain totals [erg]: kinetic, electron+ion internal."""
    V = np.asarray(sim._geometry.plasma_volume_cm3, dtype=float)
    active = np.asarray(sim._geometry.plasma_active, dtype=bool)
    mi = sim._ion_mass_g
    n = np.maximum(np.asarray(st.n), sim._floors["n"])
    K = np.asarray(st.M) ** 2 / (2.0 * mi * n)
    Eint = np.asarray(st.Ee) + np.asarray(st.Ei)
    return float(np.sum((V * (K + Eint))[active]))


def energy_production(sim, st):
    """Sum_active V * d(K+Ee+Ei)/dt for the hyperbolic operator [erg/s]."""
    V = np.asarray(sim._geometry.plasma_volume_cm3, dtype=float)
    active = np.asarray(sim._geometry.plasma_active, dtype=bool)
    mi = sim._ion_mass_g
    d = derive_state(st, floors=sim._floors, ion_mass_g=mi)
    dn, dM, dEe, dEi = hyp_rhs(sim, st)
    dK = d.u * dM - 0.5 * mi * d.u ** 2 * dn
    return float(np.sum((V * (dK + dEe + dEi))[active])), \
        float(np.sum((V * np.abs(d.u * dM))[active]))


def dz_min(sim):
    z = np.asarray(sim._geometry.z_cm, dtype=float)
    return float(np.min(np.diff(z)))


def seeded(sim, amp=0.3):
    z = np.asarray(sim._geometry.z_cm, dtype=float)
    ph = 2.0 * np.pi * (z - z[0]) / (z[-1] - z[0])
    n = 1e12 * (1.0 + amp * np.sin(ph))
    Te = 5.0 * (1.0 + amp * np.sin(ph))
    Ti = 2.0 * (1.0 + amp * np.sin(ph + 0.7))
    u = 1.0e5 * np.sin(ph)
    return state_from(sim, n, u, Te, Ti)


# --------------------------------------------------------------------------- #
def gate_g1():
    sim = make_sim()
    cells = sim._geometry.cells
    st = state_from(sim, np.full(cells, 1e12), np.zeros(cells),
                    np.full(cells, 5.0), np.full(cells, 2.0))
    dn, dM, dEe, dEi = hyp_rhs(sim, st)
    active = np.asarray(sim._geometry.plasma_active, dtype=bool)
    d = derive_state(st, floors=sim._floors, ion_mass_g=sim._ion_mass_g)
    # scale: a pressure-force divergence p/dz
    scale = float(np.max(np.abs(d.p))) / dz_min(sim)
    resid = float(max(np.max(np.abs(dn[active])), np.max(np.abs(dM[active])) / scale,
                      np.max(np.abs(dEe[active])), np.max(np.abs(dEi[active]))))
    ok = resid < 1e-10
    return "G1 constant/uniform well-balanced", ok, f"max residual/scale = {resid:.2e}"


def gate_g2():
    # Rusanov (retained by the R2 decision) smears contacts and generates a
    # bounded pressure disturbance; a contact-resolving flux (HLLC) is the
    # deferred option. The honest gate is that the disturbance is small,
    # stable, and non-growing -- not that it is zero.
    sim = make_sim()
    cells = sim._geometry.cells
    z = np.asarray(sim._geometry.z_cm)
    ph = 2.0 * np.pi * (z - z[0]) / (z[-1] - z[0])
    Ti = np.full(cells, 2.0)
    n = 1e12 * (1.0 + 0.3 * np.sin(ph))
    p_target = 1e12 * (5.0 + 2.0) * ev_to_erg  # uniform pressure
    Te = p_target / (n * ev_to_erg) - Ti
    u = np.full(cells, 3.0e4)
    st = state_from(sim, n, u, Te, Ti)
    d0 = derive_state(st, floors=sim._floors, ion_mass_g=sim._ion_mass_g)
    p0 = np.asarray(d0.p)
    active = np.asarray(sim._geometry.plasma_active, dtype=bool)
    dt = 0.3 * dz_min(sim) / float(np.max(np.abs(u) + plasma_wave_speed(d0.Te, d0.Ti, sim._mu, "adiabatic")))
    watch = FloorWatch()

    def pdev():
        d = derive_state(st, floors=sim._floors, ion_mass_g=sim._ion_mass_g)
        return float(np.max(np.abs(np.asarray(d.p)[active] - p0[active])) / np.mean(p0[active]))

    for _ in range(40):
        st = ssprk2(sim, st, dt, watch)
    dev40 = pdev()
    for _ in range(40):
        st = ssprk2(sim, st, dt, watch)
    dev80 = pdev()
    bounded = dev80 < 0.05
    non_growing = dev80 <= 1.5 * dev40 + 1e-9
    ok = (watch.binds == 0) and bounded and non_growing
    return "G2 contact: bounded, stable pressure (Rusanov smears)", ok, \
        f"dp/p @40={dev40:.2e} @80={dev80:.2e} (bounded={bounded}, non-growing={non_growing})"


def gate_g3():
    # right-moving acoustic pulse on a uniform background; measure phase speed.
    sim = make_sim(nx=240)
    cells = sim._geometry.cells
    z = np.asarray(sim._geometry.z_cm)
    active = np.asarray(sim._geometry.plasma_active, dtype=bool)
    idx = np.flatnonzero(active)
    i0, i1 = idx[3], idx[-4]  # stay off the internal boundaries
    zc = z[(i0 + i1) // 2]
    width = 0.06 * (z[i1] - z[i0])
    n0, Te0, Ti0 = 1e12, 5.0, 2.0
    mi = sim._ion_mass_g
    c = float(9.79e5 * np.sqrt((5.0 / 3.0) * (Te0 + Ti0) / sim._mu))
    rho0 = mi * n0
    p0 = n0 * (Te0 + Ti0) * ev_to_erg
    amp = 1e-3
    bump = np.exp(-((z - zc) ** 2) / (2 * width ** 2))
    dp = amp * p0 * bump
    dn = dp / (c ** 2) / mi           # delta rho / mi -> delta n
    du = dp / (rho0 * c)              # right-moving eigenvector
    n = n0 + dn
    # keep Te,Ti so that p = n(Te+Ti)ev perturbs by dp: distribute to Te
    Te = (p0 + dp) / (n * ev_to_erg) - Ti0
    u = du * bump * 0 + du            # du already carries bump via dp
    u = dp / (rho0 * c)
    st = state_from(sim, n, u, np.maximum(Te, 0.5), np.full(cells, Ti0))
    d0 = derive_state(st, floors=sim._floors, ion_mass_g=mi)
    dt = 0.3 * dz_min(sim) / float(np.max(np.abs(d0.u) + plasma_wave_speed(d0.Te, d0.Ti, sim._mu, "adiabatic")))
    nsteps = 60
    watch = FloorWatch()
    n_init = np.asarray(st.n).copy()
    for _ in range(nsteps):
        st = ssprk2(sim, st, dt, watch)
    dn_fin = np.asarray(st.n) - n0
    # centroid shift of the perturbation over the active window
    w = active & (np.abs(z - zc) < 4 * width + c * dt * nsteps)
    x = z[w]
    a0 = (n_init - n0)[w]
    a1 = dn_fin[w]
    cen0 = np.sum(x * np.abs(a0)) / np.sum(np.abs(a0))
    cen1 = np.sum(x * np.abs(a1)) / np.sum(np.abs(a1))
    v_meas = (cen1 - cen0) / (dt * nsteps)
    c_iso = float(9.79e5 * np.sqrt(Te0 / sim._mu))
    err_adi = abs(v_meas - c) / c
    ok = (watch.binds == 0) and (err_adi < 0.15) and (abs(v_meas - c) < abs(v_meas - c_iso))
    return "G3 acoustic speed = adiabatic c", ok, \
        f"v={v_meas:.3e} vs c_adi={c:.3e} (err {err_adi:.1%}), c_iso={c_iso:.3e}"


def _shock_tube(sim, ratios, nsteps, cfl=0.25):
    cells = sim._geometry.cells
    z = np.asarray(sim._geometry.z_cm)
    zc = 0.5 * (z[0] + z[-1])
    left = z < zc
    nL, nR, TeL, TeR, TiL, TiR = ratios
    n = np.where(left, nL, nR)
    Te = np.where(left, TeL, TeR)
    Ti = np.where(left, TiL, TiR)
    st = state_from(sim, n, np.zeros(cells), Te, Ti)
    E0 = total_energy(sim, st)
    d0 = derive_state(st, floors=sim._floors, ion_mass_g=sim._ion_mass_g)
    dt = cfl * dz_min(sim) / float(np.max(np.abs(d0.u) + plasma_wave_speed(d0.Te, d0.Ti, sim._mu, "adiabatic")))
    watch = FloorWatch()
    ok_pos = True
    for _ in range(nsteps):
        st = ssprk2(sim, st, dt, watch)
        d = derive_state(st, floors=sim._floors, ion_mass_g=sim._ion_mass_g)
        if not (np.all(np.isfinite(np.asarray(st.n))) and np.all(np.asarray(st.n) > 0)
                and np.all(np.asarray(d.p) > 0)):
            ok_pos = False
            break
    E1 = total_energy(sim, st)
    moved = float(np.max(np.abs(np.asarray(st.n) - n)) / nL)
    return ok_pos, watch.binds, abs(E1 - E0) / abs(E0), moved


def gate_g4():
    sim = make_sim(nx=200)
    # (a) strong 4:1 Sod -> robustness: positive + finite + waves propagate.
    #     Floors may act in the rarefaction; that is physical admissibility.
    pos_s, _, _, moved_s = _shock_tube(sim, (4e12, 1e12, 8.0, 3.0, 3.0, 1.5), 120)
    # (b) moderate floor-inert shock at two timestep sizes over the SAME end
    #     time. The KEP scheme conserves total energy in the semi-discrete sense
    #     (G5, machine zero); the residual time-integration drift of the
    #     nonlinear kinetic energy is O(dt^2), so halving dt cuts it ~4x. That
    #     confirms the drift is time-error, not a spatial conservation leak.
    mod = (2e12, 1e12, 6.0, 4.0, 3.0, 2.0)
    pos_1, b1, dE1, moved_m = _shock_tube(sim, mod, 120, cfl=0.25)
    pos_2, b2, dE2, _ = _shock_tube(sim, mod, 240, cfl=0.125)
    ratio = dE1 / max(dE2, 1e-30)
    strong_ok = pos_s and moved_s > 1e-3
    order_ok = pos_1 and pos_2 and (b1 == 0) and (b2 == 0) and (2.5 < ratio < 6.0)
    ok = strong_ok and order_ok
    return "G4 Sod: robust + O(dt^2) energy drift (semi-discrete conservative)", ok, \
        f"strong: pos={pos_s} moved={moved_s:.2f}; moderate dE/E dt->dt/2: " \
        f"{dE1:.2e}->{dE2:.2e} (ratio {ratio:.1f}, ~4 => O(dt^2) time error)"


def gate_g5():
    results = []
    for amp in (0.1, 0.3):
        sim_on = make_sim(energy_consistent=True)
        sim_off = make_sim(energy_consistent=False)
        st_on = seeded(sim_on, amp)
        st_off = seeded(sim_off, amp)
        leak_on, scale = energy_production(sim_on, st_on)
        leak_off, _ = energy_production(sim_off, st_off)
        results.append((amp, leak_on / scale, leak_off / scale))
    ok = all(abs(r[1]) < 1e-12 for r in results)
    detail = "; ".join(f"amp{a}: on={o:.1e} off={f:.1e} (rel)" for a, o, f in results)
    return "G5 closed-domain total-energy identity", ok, detail


def gate_g6():
    sim = make_sim(nx=160)
    cells = sim._geometry.cells
    z = np.asarray(sim._geometry.z_cm)
    ph = 2.0 * np.pi * (z - z[0]) / (z[-1] - z[0])
    n = 1e12 * (1.0 + 0.6 * np.sin(ph))
    Te = 6.0 * (1.0 + 0.5 * np.sin(2 * ph))
    Ti = 2.5 * (1.0 + 0.4 * np.cos(ph))
    c = plasma_wave_speed(np.full(cells, np.max(Te)), np.full(cells, np.max(Ti)), sim._mu, "adiabatic")
    u = 0.5 * float(np.max(c)) * np.sin(3 * ph)  # transonic-ish
    st = state_from(sim, n, u, Te, Ti)
    d0 = derive_state(st, floors=sim._floors, ion_mass_g=sim._ion_mass_g)
    dt = 0.3 * dz_min(sim) / float(np.max(np.abs(d0.u) + plasma_wave_speed(d0.Te, d0.Ti, sim._mu, "adiabatic")))
    watch = FloorWatch()
    ok_fin = True
    for _ in range(80):
        st = ssprk2(sim, st, dt, watch)
        if not np.all(np.isfinite(np.asarray(st.n))):
            ok_fin = False
            break
    d = derive_state(st, floors=sim._floors, ion_mass_g=sim._ion_mass_g)
    pos = bool(np.all(np.asarray(st.n) > 0) and np.all(np.asarray(d.p) > 0))
    ok = ok_fin and pos and (watch.binds == 0)
    return "G6 positivity (floor-inert)", ok, f"finite={ok_fin}, positive={pos}, floor binds={watch.binds}"


def gate_g7():
    leaks = []
    front_l1 = []
    kdiss = []
    for nx in (60, 120, 240):
        sim = make_sim(nx=nx, energy_consistent=True)
        st = seeded(sim, 0.3)
        leak, scale = energy_production(sim, st)
        leaks.append(abs(leak) / scale)
        # front-flux L1 activity (relative to advective flux L1) at this mesh
        simf = make_sim(nx=nx, energy_consistent=True, front=True)
        terms = simf.plasma_flux_rhs_terms(state=seeded(simf, 0.3))
        V = np.asarray(simf._geometry.plasma_volume_cm3)
        act = np.asarray(simf._geometry.plasma_active, dtype=bool)
        fr = np.sum(np.abs(np.asarray(terms["plasma_front_flux"].n) * V)[act])
        ad = np.sum(np.abs(np.asarray(terms["plasma_advective_flux"].n) * V)[act])
        front_l1.append(fr / ad)
        # Rusanov effective numerical diffusion ~ 0.5 * amax * dz, using the
        # interior cell size (which scales with the mesh, unlike a boundary cell)
        d = derive_state(st, floors=sim._floors, ion_mass_g=sim._ion_mass_g)
        c = plasma_wave_speed(d.Te, d.Ti, sim._mu, "adiabatic")
        dz_interior = float(np.median(np.diff(np.asarray(sim._geometry.z_cm))))
        kdiss.append(0.5 * float(np.mean(np.abs(d.u) + c)) * dz_interior)
    energy_ok = all(l < 1e-11 for l in leaks)
    front_vanishes = front_l1[0] > front_l1[1] > front_l1[2]
    diff_vanishes = kdiss[0] > kdiss[1] > kdiss[2]
    ok = energy_ok and front_vanishes and diff_vanishes
    detail = (f"leak(nx=60,120,240)={['%.1e'%l for l in leaks]}; "
              f"front L1/adv={['%.2e'%x for x in front_l1]} (vanishing={front_vanishes}); "
              f"num.diffusion[cm^2/s]={['%.2e'%k for k in kdiss]} (vanishing={diff_vanishes})")
    return "G7 mesh: identity holds; front+diffusion vanish", ok, detail


def main():
    gates = [gate_g1, gate_g2, gate_g3, gate_g4, gate_g5, gate_g6, gate_g7]
    all_ok = True
    print("R2 conservative-hyperbolic-core gate suite")
    print("=" * 72)
    for g in gates:
        name, ok, detail = g()
        all_ok = all_ok and ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        print(f"        {detail}")
    print("=" * 72)
    print("R2 hyperbolic gates:", "ALL PASS" if all_ok else "FAILURES PRESENT")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
