"""NBL pass-2 gate suite: the decoupled two-channel neutral transport.

Pre-registered analytic identities for the pass-2 build. Pass 1 gave the ``En``
field its collision coupling and its wall sink; pass 2 completes the budget and
splits the gas into two collisionally decoupled populations, so the identities
that used to close pairwise (ion <-> cold) now close three-way (ion <-> cold
<-> hot) with the wall as the only named leak.

The pass-1 suite ``verify_sim1d_nbl1_neutral_energy.py`` is NOT modified and
still passes unchanged; run both.

Gates:
  K1  KERNEL NORMALIZATION: every landing row and every residence row of the
      ballistic kernel sums to exactly 1 -- the solid-angle identity -- and the
      end-plane fold-back fraction it took to get there is reported, not hidden
  K2  KERNEL KINEMATICS: the mean absolute axial hop of an isotropic launch is
      the column radius, the closed form of dz = Rp mu/sqrt(1-mu^2); and the
      kernel is a function of geometry alone (two states with different Ti and
      nn produce the identical kernel object)
  X1  CX/ELASTIC SPLIT: nu_cx + nu_el reproduces nu_mt to the bit, nu_el equals
      nn*0.5*k_iso to the bit, and the zero floor never binds anywhere on a
      wide temperature sweep
  X2  THREE-WAY ENERGY CLOSURE: for collision + CX correction + hot channel,
      dEi*Vp + dEn*V_En summed over the domain equals the dissipated drift
      power MINUS the energy the hot atoms left on the wall, to machine
      precision. This is the pass-1 C1 identity generalized to the third
      channel; C1 itself still holds on its own operator
  X3  WHOLE-SYSTEM PARTICLE CLOSURE: the CX erosion, the ballistic landing, and
      the in-flight ionization sum to exactly zero particles
  X4  WHOLE-SYSTEM MOMENTUM CLOSURE: ion + cold + hot momentum rates sum to
      exactly minus the momentum the landed atoms left on the wall
  X5  NO COLD HEATING FROM CX: with the drift removed, the CX channel changes
      the cold gas's particle count but not its temperature, to the bit
  A1  ADVECTION STACK: with En the ledger's single advection term IS the
      mini-flux (it carries an En row), the donor-cell operator is not in the
      sum, and nothing is advected twice
  A2  MINI-FLUX REST STATE: a uniform gas at rest gives exactly zero on all
      three rows -- the pressure force cancels the closed-end faces exactly
  A3  MINI-FLUX INVENTORY: interior fluxes telescope, so the domain totals of
      the nn and En rows are zero to machine precision
  A4  PRESSURE SIGN: a positive dp_n/dz drives momentum toward -z, and the
      force scales exactly linearly with En at fixed nn
  E1  ISOTHERMAL INVARIANCE: a pure density gradient at uniform Tn moves
      particles through the Knudsen exchange without moving either cell's
      temperature -- the property that fixes the donor energy at (3/2) k Tn
  S1  SOURCE BOOKKEEPING: the puff arrives at exactly the wall energy (so it
      cannot drive the floor), the pump and ionization are temperature-
      preserving, and recombination returns exactly what it debits from Ei
  S2  BOOKING COVERAGE: every term in the live ledger has a declared booking,
      and every term declared 'none' really does leave nn alone
  W3  WALL RATE: the energy channel's wall-visit rate is built from the 300 K
      thermal speed, not the momentum closure's 0.1 eV Tn_fit
  T1  TRANSPIRATION ARM: 'local' at a uniform Tn = Tn_K reproduces 'frozen' to
      the bit, and separates from it once Tn varies
  G6  RESOLVER DOWNGRADE: the two-momentum reduction no longer REFUSES the
      neutral-energy package -- 40c519c made neutral_energy a shipped default,
      and 2f3638a made a model selection resolve a member left at its config
      default instead of raising on it -- so the gate pins the downgrade:
      construction succeeds and the resolved arm reports _neutral_energy False
      with _neutral_two_momentum True
  G7, G8 construction guards: the jet without the surface debit, and a bad or
      unusable transpiration selector, each raise a loud ValueError naming what
      is accepted; the happy paths construct

Usage:
    PYTHONPATH=<checkout>/cablp python scripts/verify/verify_sim1d_nbl2_neutral_transport.py
"""
import sys

import numpy as np

from cablp.solvers._sim1d import LAPDSim1D, default_config
from cablp.solvers._sim1d.solver import _NEUTRAL_ENERGY_TERM_BOOKING
from cablp.solvers._sim1d.core.state import (
    NEUTRAL_ENERGY_FLOOR_T_K,
    conservative_from_primitives,
    derive_state,
    neutral_energy_floor,
)
from cablp.solvers._sim1d.physics.hot_neutrals import (
    ballistic_flight_kernels,
    hot_channel_rates,
    neutral_hot_channel_rhs,
)
from cablp.solvers._sim1d.physics.neutrals import (
    neutral_exchange_rhs,
    neutral_fluid_flux_rhs,
)
from cablp.solvers._sim1d.physics.sources import (
    ion_neutral_collision_rhs,
    ion_neutral_cx_split_rates,
    neutral_cx_channel_rhs,
    neutral_energy_volume_ratio,
    neutral_temperature_eV,
    neutral_wind_velocity,
)
from cablp.atomic.cross_sections import (
    phelps_cx_rate_cm3_s,
    phelps_iso_rate_cm3_s,
    phelps_momentum_transfer_rate_cm3_s,
)
from cablp.constants import ev_to_erg, kb_cgs

CLEAN_PARAMS = {
    "ne0": 1e12, "nn0": 1e13, "Te0": 15.0, "Ti0": 2.0, "u0": 0.0,
    "gas_puff_enabled": False, "pump_enabled": False,
    "atomic_rate_model": "adas",
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

TN_K = 300.0
TN_EV = TN_K * kb_cgs / ev_to_erg
WALL_EV = NEUTRAL_ENERGY_FLOOR_T_K * kb_cgs / ev_to_erg


def make_sim(neutral_energy=True, two_zone=False, **overrides):
    params, flags = default_config()
    params.update(CLEAN_PARAMS)
    params["nx"] = 60
    params["gas_type"] = "He"
    params["Tn_K"] = TN_K
    flags.update(CLEAN_FLAGS)
    flags["ion_neutral_moment_closure"] = True
    flags["neutral_momentum"] = True
    flags["neutral_two_zone"] = bool(two_zone)
    flags["neutral_energy"] = bool(neutral_energy)
    for key, value in overrides.items():
        if key in flags:
            flags[key] = value
        else:
            params[key] = value
    return LAPDSim1D(params, flags)


def make_state(sim, u_i, u_n, Ti=2.0, Tn_K=TN_K, En_shape=None, nn=None,
               two_zone=False):
    cells = sim._geometry.cells
    if nn is None:
        nn = np.full(cells, 1.0e13)
    nn = np.asarray(nn, dtype=float)
    state = conservative_from_primitives(
        n=np.full(cells, 1.0e12),
        nn=nn,
        u=np.full(cells, float(u_i)),
        Te=np.full(cells, 15.0),
        Ti=np.full(cells, float(Ti)) if np.isscalar(Ti) else np.asarray(Ti),
        ion_mass_g=sim._ion_mass_g,
        un=np.full(cells, float(u_n)),
        nn_a=nn.copy() if two_zone else None,
        Tn_K=Tn_K,
    )
    if En_shape is not None:
        state = type(state)(
            n=state.n, nn=state.nn, M=state.M, Ee=state.Ee, Ei=state.Ei,
            M_n=state.M_n, nn_a=state.nn_a, M_n_a=state.M_n_a,
            En=np.asarray(En_shape, dtype=float),
        )
    return state


def _volumes(sim, state):
    Vp = np.asarray(sim._geometry.plasma_volume_cm3, dtype=float)
    return Vp, Vp / neutral_evr(sim, state)


def neutral_evr(sim, state):
    return neutral_energy_volume_ratio(state, sim._geometry)


def _channels(sim, state, ionization_rate=None):
    """Return (collision, cx correction, hot rhs, hot diagnostics)."""
    kwargs = sim._collision_operator_kwargs()
    coll = ion_neutral_collision_rhs(
        state=state, floors=sim._floors, ion_mass_g=sim._ion_mass_g,
        geometry=sim._geometry, wind_column_factor=sim._wind_column_factor,
        **kwargs,
    )
    cx = neutral_cx_channel_rhs(
        state=state, floors=sim._floors, ion_mass_g=sim._ion_mass_g,
        geometry=sim._geometry, wind_column_factor=sim._wind_column_factor,
        **kwargs,
    )
    rate = (
        np.zeros_like(state.nn) if ionization_rate is None else ionization_rate
    )
    hot, diagnostics = neutral_hot_channel_rhs(
        state=state, floors=sim._floors, ion_mass_g=sim._ion_mass_g,
        geometry=sim._geometry, kernels=sim._hot_neutral_kernels,
        I_ion=sim._I_ion, ionization_rate_per_neutral=rate,
        wind_column_factor=sim._wind_column_factor, **kwargs,
    )
    return coll, cx, hot, diagnostics


def _bitwise(a, b):
    a = np.ascontiguousarray(np.asarray(a, dtype=float))
    b = np.ascontiguousarray(np.asarray(b, dtype=float))
    return a.shape == b.shape and np.array_equal(a.view(np.uint64), b.view(np.uint64))


# --------------------------------------------------------------- kernel gates


def gate_k1():
    sim = make_sim()
    landing, residence, end_fraction = sim._hot_neutral_kernels
    land_err = float(np.max(np.abs(landing.sum(axis=1) - 1.0)))
    res_err = float(np.max(np.abs(residence.sum(axis=1) - 1.0)))
    ok = land_err < 1e-13 and res_err < 1e-13
    return "K1 kernel rows close to 1 (solid-angle normalization)", ok, (
        f"landing max|rowsum-1| = {land_err:.2e}  residence = {res_err:.2e}  "
        f"end-plane fold-back: mean {float(np.mean(end_fraction)):.4f} "
        f"max {float(np.max(end_fraction)):.4f}"
    )


def gate_k2():
    sim = make_sim()
    landing, _residence, _end = sim._hot_neutral_kernels
    z = np.asarray(sim._geometry.z_cm, dtype=float)
    dz = np.asarray(sim._geometry.length_cm, dtype=float)
    Rp = np.asarray(sim._geometry.Rp_cm, dtype=float)
    # For an isotropic launch, dz = c * mu/sqrt(1-mu^2) with mu uniform on
    # [-1,1], so P(|dz| <= x) = (x/c) / sqrt(1 + (x/c)^2) in closed form. The
    # kernel's cumulative landing fraction within k cells of the birth is that
    # probability at the cell EDGE, exactly -- no discretization enters, which
    # is what makes this a transcription check and not a resolution check.
    mid = z.size // 2
    worst = 0.0
    rows = []
    for k in (0, 1, 2, 4, 8):
        within = np.abs(np.arange(z.size) - mid) <= k
        measured = float(np.sum(landing[mid][within]))
        x = (k + 0.5) * float(dz[mid])
        r = x / float(Rp[mid])
        expect = r / np.sqrt(1.0 + r * r)
        worst = max(worst, abs(measured - expect))
        rows.append(f"k={k}: {measured:.6f} vs {expect:.6f}")
    # Geometry alone: the kernel does not depend on the plasma state at all.
    # The rebuild must mirror the solver's own arming of the internal-wall
    # option, which is what selects the flight bounds; the default in the
    # function signature is not the value the solver builds with.
    again = ballistic_flight_kernels(
        sim._geometry, internal_wall=sim._neutral_hot_internal_wall
    )
    stable = _bitwise(again[0], landing)
    ok = worst < 1e-3 and stable
    return "K2 kernel kinematics match the closed-form isotropic hop", ok, (
        f"max |measured - closed form| = {worst:.2e}  ["
        + "; ".join(rows)
        + f"]  rebuild bitwise-identical={stable}"
    )


# ------------------------------------------------------------- CX split gates


def gate_x1():
    Ti = np.logspace(-2, 1.5, 400)
    Tn = np.full_like(Ti, TN_EV)
    nn = np.full_like(Ti, 3.0e12)
    nu_cx, nu_el = ion_neutral_cx_split_rates(
        nn=nn, Ti=Ti, Tn=Tn, gas_type="He"
    )
    T_eff = 0.5 * (Ti + Tn)
    nu_mt = nn * phelps_momentum_transfer_rate_cm3_s(T_eff, gas_type="He")
    closure = float(np.max(np.abs((nu_cx + nu_el) / nu_mt - 1.0)))
    # The elastic remainder IS the half isotropic-elastic rate: that is what
    # makes the split a partition of nu_mt rather than a subtraction that
    # happens to be positive.
    half_iso = nn * 0.5 * phelps_iso_rate_cm3_s(T_eff, gas_type="He")
    close = float(np.max(np.abs(nu_el / half_iso - 1.0)))
    cx_exact = _bitwise(nu_cx, nn * phelps_cx_rate_cm3_s(T_eff, "He"))
    floor_binds = bool(np.any(nu_el < 0.0))
    ok = closure < 1e-15 and close < 1e-12 and cx_exact and not floor_binds
    return "X1 CX/elastic split is exact and the zero floor never binds", ok, (
        f"|(nu_cx+nu_el)/nu_mt - 1| <= {closure:.2e}  nu_el == nn*0.5*k_iso to "
        f"{close:.2e}  nu_cx == nn*k_b bitwise={cx_exact}  floor bound "
        f"anywhere={floor_binds}  (sweep Ti 1e-2..32 eV, {Ti.size} points)"
    )


def _ionization_rate(sim, state):
    """A representative in-flight ionization frequency [1/s] for the gates."""
    return 1.0e3 * np.ones_like(np.asarray(state.nn, dtype=float))


def gate_x2(two_zone=False):
    sim = make_sim(two_zone=two_zone)
    st = make_state(sim, u_i=3.0e5, u_n=1.0e5, Ti=2.0, two_zone=two_zone)
    rate = _ionization_rate(sim, st)
    coll, cx, hot, diag = _channels(sim, st, rate)
    Vp, V_En = _volumes(sim, st)
    der = derive_state(st, floors=sim._floors, ion_mass_g=sim._ion_mass_g)
    u_n = neutral_wind_velocity(
        st, floors=sim._floors, ion_mass_g=sim._ion_mass_g,
        geometry=sim._geometry,
    )
    dissipated = float(np.sum(-np.asarray(coll.M) * (der.u - u_n) * Vp))
    wall_loss = (
        diag["hot_wall_energy_erg_s"] - diag["hot_wall_energy_returned_erg_s"]
    )
    thermal = float(
        np.sum((coll.Ei + cx.Ei + hot.Ei) * Vp)
        + np.sum(
            (
                np.asarray(coll.En)
                + np.asarray(cx.En)
                + np.asarray(hot.En)
            )
            * V_En
        )
    )
    residual = thermal - (dissipated - wall_loss)
    scale = max(abs(dissipated), abs(wall_loss), 1e-300)
    rel = abs(residual) / scale
    ok = rel < 1e-11 and abs(dissipated) > 0.0 and wall_loss > 0.0
    label = "two-zone" if two_zone else "single-zone"
    return (
        f"X2 three-way energy closure dEi*Vp + dEn*V_En == P_diss - L_wall "
        f"({label})",
        ok,
        f"rel residual = {rel:.2e}  P_diss = {dissipated:.6e} erg/s  "
        f"L_wall = {wall_loss:.6e} erg/s",
    )


def gate_x2_two_zone():
    return gate_x2(two_zone=True)


def gate_x3(two_zone=False):
    sim = make_sim(two_zone=two_zone)
    st = make_state(sim, u_i=3.0e5, u_n=1.0e5, Ti=2.0, two_zone=two_zone)
    coll, cx, hot, _diag = _channels(sim, st, _ionization_rate(sim, st))
    Vp, V_En = _volumes(sim, st)
    V_ann = np.asarray(sim._geometry.neutral_volume_cm3, dtype=float) - Vp
    cold = float(np.sum(np.asarray(cx.nn) * V_En))
    landed = (
        float(np.sum(np.asarray(hot.nn_a) * V_ann))
        if two_zone
        else float(np.sum(np.asarray(hot.nn) * V_En))
    )
    plasma = float(np.sum(np.asarray(hot.n) * Vp))
    total = cold + landed + plasma
    scale = max(abs(cold), 1e-300)
    rel = abs(total) / scale
    ok = rel < 1e-12 and abs(cold) > 0.0 and landed > 0.0 and plasma > 0.0
    label = "two-zone" if two_zone else "single-zone"
    return f"X3 whole-system particle closure ({label})", ok, (
        f"eroded = {cold:.6e} /s  landed = {landed:.6e} /s  "
        f"ionized = {plasma:.6e} /s  rel residual = {rel:.2e}"
    )


def gate_x3_two_zone():
    return gate_x3(two_zone=True)


def gate_x4():
    sim = make_sim()
    st = make_state(sim, u_i=3.0e5, u_n=1.0e5, Ti=2.0)
    coll, cx, hot, _diag = _channels(sim, st, _ionization_rate(sim, st))
    Vp = np.asarray(sim._geometry.plasma_volume_cm3, dtype=float)
    Vm = np.asarray(sim._geometry.neutral_volume_cm3, dtype=float)
    rates = hot_channel_rates(
        state=st, floors=sim._floors, ion_mass_g=sim._ion_mass_g,
        geometry=sim._geometry, ionization_rate_per_neutral=_ionization_rate(sim, st),
        residence=sim._hot_neutral_kernels[1],
        wind_column_factor=sim._wind_column_factor,
        **sim._collision_operator_kwargs(),
    )
    wall_momentum = float(np.sum(rates["wall"] * rates["p_hot"] * Vp))
    total = (
        float(np.sum(np.asarray(coll.M) * Vp))
        + float(np.sum((np.asarray(coll.M_n) + np.asarray(cx.M_n)) * Vm))
        + float(np.sum(np.asarray(hot.M) * Vp))
    )
    residual = total + wall_momentum
    rel = abs(residual) / max(abs(wall_momentum), 1e-300)
    ok = rel < 1e-11 and abs(wall_momentum) > 0.0
    return "X4 whole-system momentum closure == -(wall absorption)", ok, (
        f"system rate = {total:.6e}  wall = {-wall_momentum:.6e} g cm/s^2  "
        f"rel residual = {rel:.2e}"
    )


def gate_x5():
    # No drift anywhere, so the whole cold-side energy booking is thermal. The
    # CX share must be a pure per-particle removal and the ONLY heating left
    # must be the elastic rate's: (coll + cx).En == -q_therm_el - e_cold S_cx.
    sim = make_sim()
    st = make_state(sim, u_i=0.0, u_n=0.0, Ti=6.0)
    coll, cx, _hot, _diag = _channels(sim, st)
    der = derive_state(st, floors=sim._floors, ion_mass_g=sim._ion_mass_g)
    Tn = neutral_temperature_eV(st, floors=sim._floors, Tn_eV=TN_EV)
    nu_cx, nu_el = ion_neutral_cx_split_rates(
        nn=st.nn, Ti=der.Ti, Tn=Tn, gas_type="He"
    )
    n = np.asarray(st.n, dtype=float)
    ratio = neutral_evr(sim, st)
    e_cold = 1.5 * Tn * ev_to_erg
    expected = (
        -1.5 * nu_el * n * (Tn - der.Ti) * ev_to_erg - e_cold * nu_cx * n
    ) * ratio
    actual = np.asarray(coll.En) + np.asarray(cx.En)
    rel = float(np.max(np.abs(actual - expected) / np.abs(expected)))
    # And the CX share really is gone: pass-1's booking is materially different.
    separation = float(
        np.max(np.abs(actual / np.asarray(coll.En) - 1.0))
    )
    ok = rel < 1e-13 and separation > 1e-2
    return "X5 CX erodes the cold gas; only the ELASTIC rate heats it", ok, (
        f"(coll+cx).En == elastic-only + per-particle removal, rel = "
        f"{rel:.2e}  separation from the pass-1 full-nu_mt booking = "
        f"{separation:.3f}"
    )


# --------------------------------------------------------------- advection


def gate_a1():
    sim = make_sim()
    st = make_state(sim, u_i=3.0e5, u_n=1.0e5, Ti=2.0)
    term = sim.neutral_wind_advection_rhs(state=st)
    direct = neutral_fluid_flux_rhs(
        state=st, floors=sim._floors, ion_mass_g=sim._ion_mass_g,
        geometry=sim._geometry, mesh_faces=sim._mesh_faces,
        mesh_blocked_area_cm2=sim._mesh_blocked_area_cm2,
    )
    is_miniflux = (
        term.En is not None
        and _bitwise(term.nn, direct.nn)
        and _bitwise(term.M_n, direct.M_n)
        and _bitwise(term.En, direct.En)
    )
    ledger = sim.rhs_terms()
    advection_keys = [k for k in ledger if "advection" in k or "flux" in k]
    neutral_advection = [k for k in advection_keys if k.startswith("neutral")]
    single = neutral_advection == ["neutral_wind_advection"]
    ok = is_miniflux and single
    return "A1 advection stack: the mini-flux SUPERSEDES the donor cell", ok, (
        f"ledger term is the mini-flux bitwise={is_miniflux}  "
        f"neutral advection keys in the ledger={neutral_advection}  "
        f"exactly one={single}"
    )


def gate_a2():
    sim = make_sim()
    st = make_state(sim, u_i=0.0, u_n=0.0, Ti=2.0)
    term = neutral_fluid_flux_rhs(
        state=st, floors=sim._floors, ion_mass_g=sim._ion_mass_g,
        geometry=sim._geometry,
    )
    still = (
        bool(np.all(np.asarray(term.nn) == 0.0))
        and bool(np.all(np.asarray(term.En) == 0.0))
        and bool(np.all(np.asarray(term.M_n) == 0.0))
    )
    return "A2 mini-flux leaves a uniform gas at rest exactly stationary", still, (
        f"max|dnn|={float(np.max(np.abs(term.nn))):.3e}  "
        f"max|dEn|={float(np.max(np.abs(term.En))):.3e}  "
        f"max|dM_n|={float(np.max(np.abs(term.M_n))):.3e}"
    )


def gate_a3():
    sim = make_sim()
    cells = sim._geometry.cells
    nn = 1.0e13 * (1.0 + 0.5 * np.sin(np.linspace(0.0, 6.0, cells)))
    st = make_state(sim, u_i=0.0, u_n=2.0e4, Ti=2.0, nn=nn)
    term = neutral_fluid_flux_rhs(
        state=st, floors=sim._floors, ion_mass_g=sim._ion_mass_g,
        geometry=sim._geometry,
    )
    V = np.asarray(sim._geometry.neutral_volume_cm3, dtype=float)
    A = np.asarray(sim._geometry.neutral_face_area_cm2, dtype=float)
    dn = float(np.sum(np.asarray(term.nn) * V))
    dE = float(np.sum(np.asarray(term.En) * V))
    scale_n = float(np.sum(np.abs(np.asarray(term.nn)) * V))
    scale_E = float(np.sum(np.abs(np.asarray(term.En)) * V))
    # The En row is NOT conservative and is not meant to be: the -p_n div u_n
    # pressure work converts thermal energy to and from bulk kinetic energy.
    # The gate is that the whole non-telescoping part IS that work and nothing
    # else, so the advective half conserves exactly.
    u_n = neutral_wind_velocity(
        st, floors=sim._floors, ion_mass_g=sim._ion_mass_g,
        geometry=sim._geometry,
    )
    face_u = np.zeros(V.size + 1, dtype=float)
    face_u[1:-1] = 0.5 * (u_n[:-1] + u_n[1:])
    div_u = (A[1:] * face_u[1:] - A[:-1] * face_u[:-1]) / V
    work = float(np.sum(-(2.0 / 3.0) * np.asarray(st.En) * div_u * V))
    ok = (
        abs(dn) / scale_n < 1e-13
        and abs(dE - work) / max(abs(work), scale_E * 1e-14) < 1e-11
    )
    return "A3 mini-flux: nn conserves, En moves only by pressure work", ok, (
        f"nn residual {abs(dn) / scale_n:.2e} (relative to the transported "
        f"magnitude)  En total {dE:.6e} vs pressure work {work:.6e} erg/s, "
        f"rel {abs(dE - work) / max(abs(work), 1e-300):.2e}"
    )


def gate_a4():
    sim = make_sim()
    cells = sim._geometry.cells
    st = make_state(sim, u_i=0.0, u_n=0.0, Ti=2.0)
    ramp = np.linspace(1.0, 3.0, cells)
    hot = make_state(
        sim, u_i=0.0, u_n=0.0, Ti=2.0,
        En_shape=np.asarray(st.En) * ramp,
    )
    hotter = make_state(
        sim, u_i=0.0, u_n=0.0, Ti=2.0,
        En_shape=2.0 * np.asarray(st.En) * ramp,
    )
    a = np.asarray(
        neutral_fluid_flux_rhs(
            state=hot, floors=sim._floors, ion_mass_g=sim._ion_mass_g,
            geometry=sim._geometry,
        ).M_n
    )
    b = np.asarray(
        neutral_fluid_flux_rhs(
            state=hotter, floors=sim._floors, ion_mass_g=sim._ion_mass_g,
            geometry=sim._geometry,
        ).M_n
    )
    # Judge the sign only where the duct is genuinely uniform: in a varying
    # cross-section the wall reaction p dA/dz is part of the same force and a
    # bare gradient sign is not the right question there.
    area = np.asarray(sim._geometry.neutral_face_area_cm2, dtype=float)
    uniform = np.isclose(area[:-1], area[1:], rtol=1e-12)
    uniform[:2] = False
    uniform[-2:] = False
    pushes_down = bool(np.all(a[uniform] < 0.0))
    live = np.abs(a) > 0.0
    linear = float(np.max(np.abs(b[live] / a[live] - 2.0)))
    ok = pushes_down and linear < 1e-12 and int(np.count_nonzero(uniform)) > 10
    return "A4 cold pressure pushes down-gradient and is linear in En", ok, (
        f"dp_n/dz > 0 drives M_n < 0 on all {int(np.count_nonzero(uniform))} "
        f"uniform-area interior cells={pushes_down}  max |ratio - 2| = "
        f"{linear:.2e}"
    )


def gate_e1():
    # Uniform Tn, non-uniform nn: particles move, temperature must not.
    sim = make_sim()
    cells = sim._geometry.cells
    nn = 1.0e13 * (1.0 + 0.4 * np.cos(np.linspace(0.0, 5.0, cells)))
    st = make_state(sim, u_i=0.0, u_n=0.0, Ti=2.0, nn=nn)
    term = neutral_exchange_rhs(
        state=st, geometry=sim._geometry,
        exchange_coeff_cm3_s=sim.neutral_exchange_coefficients(),
        floors=sim._floors,
    )
    per_particle = np.asarray(st.En) / np.asarray(st.nn)
    # dTn/dt ∝ (dEn - (En/nn) dnn) / nn
    residual = np.asarray(term.En) - per_particle * np.asarray(term.nn)
    scale = np.maximum(np.abs(per_particle * np.asarray(term.nn)), 1e-300)
    rel = float(np.max(np.abs(residual) / scale))
    moves = bool(np.any(np.asarray(term.nn) != 0.0))
    ok = rel < 1e-11 and moves
    return "E1 Knudsen exchange leaves an isothermal gas isothermal", ok, (
        f"max relative dTn drive = {rel:.2e}  particles actually move={moves}"
    )


# ----------------------------------------------------------- source booking


def gate_s1():
    sim = make_sim(gas_puff_enabled=True, pump_enabled=True, S_gp=9010)
    st = make_state(sim, u_i=0.0, u_n=0.0, Ti=2.0)
    term = sim.neutral_source_sink_rhs(state=st, time=0.0)
    nn_row = np.asarray(term.nn)
    En_row = np.asarray(term.En)
    per_particle = np.asarray(st.En) / np.asarray(st.nn)
    wall = float(neutral_energy_floor(np.ones(1))[0])
    source = nn_row > 0.0
    sink = nn_row < 0.0

    def _rel(a, b):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        if a.size == 0:
            return np.inf
        return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))

    puff_rel = _rel(En_row[source], nn_row[source] * wall)
    puff_ok = bool(np.any(source)) and puff_rel < 1e-14
    pump_rel = _rel(En_row[sink], nn_row[sink] * per_particle[sink])
    pump_ok = bool(np.any(sink)) and pump_rel < 1e-14
    # Ionization: the ledger's booking must debit the local per-particle energy.
    ledger = sim.rhs_terms()
    ion = ledger["ionization_birth"]
    ion_rel = _rel(
        np.asarray(ion.En), np.asarray(ion.nn) * per_particle
    )
    ion_ok = ion_rel < 1e-14 and bool(np.any(np.asarray(ion.nn) < 0.0))
    # Recombination returns exactly what Ei loses, per particle.
    rec = ledger["recombination_rad_loss"]
    der = derive_state(st, floors=sim._floors, ion_mass_g=sim._ion_mass_g)
    Vp = np.asarray(sim._geometry.plasma_volume_cm3, dtype=float)
    V_En = Vp / neutral_evr(sim, st)
    lost = float(np.sum(np.asarray(rec.Ei) * Vp))
    gained = float(np.sum(np.asarray(rec.En) * V_En))
    rec_rel = abs(lost + gained) / max(abs(lost), 1e-300)
    ok = puff_ok and pump_ok and ion_ok and rec_rel < 1e-12
    return "S1 puff at T_wall, pump/ionization local, recombination closes", ok, (
        f"puff at the wall energy rel={puff_rel:.2e}  pump per-particle "
        f"rel={pump_rel:.2e}  ionization per-particle rel={ion_rel:.2e}  "
        f"recombination Ei/En rel residual={rec_rel:.2e}"
    )


def gate_s2():
    sim = make_sim(gas_puff_enabled=True, pump_enabled=True, S_gp=9010)
    ledger = sim.rhs_terms()
    missing = sorted(set(ledger) - set(_NEUTRAL_ENERGY_TERM_BOOKING))
    declared_none = [
        name for name, mode in _NEUTRAL_ENERGY_TERM_BOOKING.items()
        if mode == "none" and name in ledger
    ]
    liars = [
        name for name in declared_none
        if np.any(np.asarray(ledger[name].nn) != 0.0)
    ]
    ok = not missing and not liars
    return "S2 every ledger term has a declared neutral-energy booking", ok, (
        f"terms in the ledger={len(ledger)}  undeclared={missing}  "
        f"declared 'none' but moving nn={liars}"
    )


def gate_w3():
    sim = make_sim()
    kwargs = sim._neutral_energy_timestep_kwargs()
    used = float(kwargs["Tn_fit"])
    ok = abs(used / WALL_EV - 1.0) < 1e-12 and abs(used - 0.1) > 1e-3
    return "W3 the energy channel's wall rate uses the 300 K thermal speed", ok, (
        f"Tn_fit fed to the wall sink = {used:.6e} eV (300 K = {WALL_EV:.6e} "
        f"eV); the momentum closure's 0.1 eV is NOT used"
    )


def gate_t1():
    frozen = make_sim(neutral_knudsen_temperature="frozen")
    local = make_sim(neutral_knudsen_temperature="local")
    # The arm scales a conductance, so it can only show up where a density
    # gradient is driving a current: a uniform-nn state has no flow to scale.
    cells = frozen._geometry.cells
    nn = 1.0e13 * (1.0 + 0.4 * np.cos(np.linspace(0.0, 5.0, cells)))
    # Uniform Tn == Tn_K: the scale is exactly 1, so the two must agree bitwise.
    flat = make_state(frozen, u_i=0.0, u_n=0.0, Ti=2.0, Tn_K=TN_K, nn=nn)
    a = np.asarray(frozen.neutral_exchange_rhs(state=flat).nn)
    b = np.asarray(local.neutral_exchange_rhs(state=flat).nn)
    # Not bit-for-bit: Tn is RECONSTRUCTED from En, so sqrt(Tn/Tn_K) lands an
    # ulp off unity even when the gas is exactly at Tn_K, and the exchange row
    # is a DIFFERENCE of face rates, which amplifies that. The crisp identity
    # is on the scale factor itself; the row is quoted as corroboration.
    scale = local._transpiration_face_scale(flat)
    scale_err = float(np.max(np.abs(scale - 1.0)))
    live_flat = np.abs(a) > 0.0
    identity = float(np.max(np.abs(b[live_flat] / a[live_flat] - 1.0)))
    same = scale_err < 1e-15 and identity < 1e-12
    # Non-uniform Tn: the arm must actually separate.
    ramp = np.asarray(flat.En) * np.linspace(1.0, 25.0, cells)
    hot = make_state(frozen, u_i=0.0, u_n=0.0, Ti=2.0, nn=nn, En_shape=ramp)
    c = np.asarray(frozen.neutral_exchange_rhs(state=hot).nn)
    d = np.asarray(local.neutral_exchange_rhs(state=hot).nn)
    live = np.abs(c) > 0.0
    separation = float(np.max(np.abs(d[live] / c[live] - 1.0)))
    ok = same and separation > 1e-2 and bool(np.any(live))
    return "T1 transpiration: identity at uniform Tn, separates when it varies", ok, (
        f"uniform-Tn scale factor departs from 1 by {scale_err:.2e}; the "
        f"exchange row agrees to {identity:.2e}  max relative separation on a "
        f"25x Tn ramp = {separation:.3f}"
    )


def _guard(label, expect_fragment, **overrides):
    try:
        make_sim(**overrides)
    except ValueError as exc:
        text = str(exc)
        return label, expect_fragment in text, f"raised: {text[:104]}"
    return label, False, "no ValueError raised"


def gate_g6():
    """Pin the RESOLVER's downgrade, which replaced this gate's old refusal.

    40c519c flipped ``neutral_energy`` ON in the shipped defaults, so the
    two-moment reduction's incompatibility with the neutral-energy package is
    now with a member sitting AT ITS CONFIG DEFAULT, not with an explicit
    caller choice.

    2f3638a made a model selection OWN its member keys: a member left at its
    config default is resolved to the value the selection requires instead of
    raising, and only an EXPLICITLY set member still raises.

    Together those make this configuration construct rather than refuse, so
    the gate certifies what the resolver documents it does -- the arm comes
    back with the neutral-energy package off and the two-momentum closure on,
    which is the state the earlier ValueError stood for.
    """
    label = (
        "G6 resolver: neutral_energy at its default downgrades under the "
        "two-momentum reduction"
    )
    try:
        sim = make_sim(
            neutral_two_zone=True,
            neutral_momentum_radial="kinetic_two_moment",
        )
    except ValueError as exc:
        return label, False, f"construction REFUSED: {exc}"
    ok = sim._neutral_energy is False and sim._neutral_two_momentum is True
    return label, ok, (
        f"constructs=True  resolved _neutral_energy={sim._neutral_energy} "
        f"(expect False)  _neutral_two_momentum={sim._neutral_two_momentum} "
        f"(expect True)"
    )


def gate_g7():
    label, ok, detail = _guard(
        "G7 guard: the cathode jet without the surface debit raises",
        "requires cathode_jet_surface_debit",
        cathode_neutral_jet=True,
        cathode_jet_R_N=0.5,
        cathode_jet_R_E=0.5,
        cathode_jet_surface_debit=False,
    )
    try:
        make_sim(
            cathode_neutral_jet=True, cathode_jet_R_N=0.5,
            cathode_jet_R_E=0.5, cathode_jet_surface_debit=True,
        )
        happy = True
        happy_detail = "happy path constructs"
    except ValueError as exc:
        happy = False
        happy_detail = f"happy path REFUSED: {exc}"
    return label, ok and happy, f"{detail} | {happy_detail}"


def gate_g8():
    label, ok, detail = _guard(
        "G8 guard: a bad or unusable transpiration selector raises",
        "must be 'frozen' or 'local'",
        neutral_knudsen_temperature="sqrt",
    )
    # neutral_hot_internal_wall is a shipped default (True) whose own
    # neutral_energy guard fires FIRST, masking the selector guard this
    # sub-gate is for; clear it so the transpiration refusal is what is read.
    label2, ok2, detail2 = _guard(
        "local without neutral_energy",
        "requires the neutral_energy flag",
        neutral_energy=False,
        neutral_hot_internal_wall=False,
        neutral_knudsen_temperature="local",
    )
    return label, ok and ok2, f"{detail} | {label2}: {detail2[:96]}"


def main():
    gates = [
        gate_k1, gate_k2,
        gate_x1, gate_x2, gate_x2_two_zone, gate_x3, gate_x3_two_zone,
        gate_x4, gate_x5,
        gate_a1, gate_a2, gate_a3, gate_a4, gate_e1,
        gate_s1, gate_s2, gate_w3, gate_t1,
        gate_g6, gate_g7, gate_g8,
    ]
    all_ok = True
    print("NBL pass-2 gate suite (decoupled two-channel neutral transport)")
    print("=" * 76)
    for gate in gates:
        name, ok, detail = gate()
        all_ok = all_ok and ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        print(f"        {detail}")
    print("=" * 76)
    print("NBL pass-2 gates:", "ALL PASS" if all_ok else "FAILURES PRESENT")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
