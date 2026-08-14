"""NBL pass-1 neutral-energy gate suite (En field core).

Pre-registered analytic identities and construction guards for the optional
``En`` conservative field (flag ``neutral_energy``, default off). Pass 1 builds
the field, the collision coupling, and the wall accommodation ONLY: there is no
pressure force, no En advection, no Knudsen enthalpy carriage, and no
jet/puff/pump/reaction En bookkeeping, so the En budget is deliberately
incomplete and a flag-on run is not a physics arm.

Extends the ``verify_sim1d_r4_collision.py`` pattern (pure-RHS harness, no time
advance).

Gates:
  C1  PAIRWISE ENERGY CONSERVATION: for the extended moment-closed operator,
      dEi*Vp + dEn*V_En == -dM * u_rel * Vp per cell -- the ion rows plus the
      En rows sum to exactly the dissipated drift power -- to machine
      precision, with both channels (friction and thermal) live
  C2  thermal-only limit: at u == u_n the frictional halves vanish and the
      thermal exchange is exactly antisymmetric in the volume-integrated
      energy, dEi*Vp == -dEn*V_En
  C3  friction-only limit: at Ti == Tn the thermal channel vanishes and the two
      species take exactly HALF the dissipated power each (equal-mass split)
  C4  per-cell Tn: the operator reads Tn = (2/3) En / (nn k) from the field,
      not the config scalar -- at rest the ion row equals the closed form
      1.5 n nu_mt(T_eff(Tn_cell)) (Tn_cell - Ti) k evaluated cell by cell (an
      exact identity, and materially different from the same form at the
      scalar Tn), and setting En to the scalar's own energy reproduces the
      scalar result bit-for-bit
  W1  wall sink sign and equilibrium: the sink is <= 0 above the wall energy,
      >= 0 below it, and identically zero AT (3/2) nn k T_wall -- the same
      energy apply_state_floors clips to
  W2  wall rate scaling: the sink is exactly linear in alpha_E, and alpha_E = 0
      makes it identically zero
  F1  floor: apply_state_floors clips En up to (3/2) nn k T_wall against the
      FLOORED nn and leaves an above-floor En untouched
  P1  presence-off: with the flag off no state carries En, every term's En row
      is absent, and the packed width is the historical one
  G1..G5 construction guards: missing moment closure, missing neutral momentum,
      coverage_closure, each kinetic neutral model, and alpha_E outside [0, 1]
      each raise a loud ValueError naming what is accepted; the happy path
      constructs and packs En last

Usage:
    PYTHONPATH=<checkout>/cablp python scripts/verify_sim1d_nbl1_neutral_energy.py
"""
import sys

import numpy as np

from cablp.solvers._sim1d import LAPDSim1D, default_config
from cablp.solvers._sim1d.core.state import (
    NEUTRAL_ENERGY_FLOOR_T_K,
    apply_state_floors,
    conservative_from_primitives,
    derive_state,
    neutral_energy_floor,
    state_field_names,
)
from cablp.solvers._sim1d.physics.sources import (
    ion_neutral_collision_rhs,
    neutral_energy_volume_ratio,
    neutral_energy_wall_rhs,
    neutral_temperature_eV,
    neutral_wind_velocity,
)
from cablp.funcs._cross import phelps_momentum_transfer_rate_cm3_s
from cablp.vars._cons import ev_to_erg, kb_cgs

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


def make_state(sim, u_i, u_n, Ti=2.0, Tn_K=TN_K, En_shape=None, two_zone=False):
    cells = sim._geometry.cells
    nn = np.full(cells, 1.0e13)
    state = conservative_from_primitives(
        n=np.full(cells, 1.0e12),
        nn=nn,
        u=np.full(cells, float(u_i)),
        Te=np.full(cells, 15.0),
        Ti=np.full(cells, float(Ti)),
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


def _active(sim):
    return np.asarray(sim._geometry.plasma_active, dtype=bool)


def _volumes(sim, state):
    Vp = np.asarray(sim._geometry.plasma_volume_cm3, dtype=float)
    V_En = Vp * neutral_energy_volume_ratio(state, sim._geometry) ** -1.0
    return Vp, V_En


def _collision(sim, state):
    return ion_neutral_collision_rhs(
        state=state,
        floors=sim._floors,
        ion_mass_g=sim._ion_mass_g,
        gas_type="He",
        Tn_eV=TN_EV,
        geometry=sim._geometry,
    )


def _u_rel(sim, state):
    der = derive_state(state, floors=sim._floors, ion_mass_g=sim._ion_mass_g)
    u_n = neutral_wind_velocity(
        state, floors=sim._floors, ion_mass_g=sim._ion_mass_g,
        geometry=sim._geometry,
    )
    return der.u - u_n


def _max_rel(residual, scale, mask):
    scale = np.maximum(np.abs(scale), 1e-300)
    return float(np.max(np.abs(residual[mask]) / scale[mask]))


def gate_c1(two_zone=False):
    sim = make_sim(two_zone=two_zone)
    st = make_state(sim, u_i=3.0e5, u_n=1.0e5, Ti=2.0, two_zone=two_zone)
    term = _collision(sim, st)
    Vp, V_En = _volumes(sim, st)
    u_rel = _u_rel(sim, st)
    lhs = term.Ei * Vp + np.asarray(term.En) * V_En
    rhs = -np.asarray(term.M) * u_rel * Vp
    act = _active(sim)
    rel = _max_rel(lhs - rhs, rhs, act)
    live = np.any(term.Ei[act] != 0.0) and np.any(np.asarray(term.En)[act] != 0.0)
    ok = rel < 1e-13 and live
    label = "two-zone" if two_zone else "single-zone"
    return (
        f"C1 pairwise energy conservation dEi*Vp + dEn*V_En == -dM*u_rel*Vp "
        f"({label})",
        ok,
        f"max rel residual = {rel:.2e}  both channels live={live}",
    )


def gate_c1_two_zone():
    return gate_c1(two_zone=True)


def gate_c2():
    # u == u_n: no friction anywhere, so the thermal exchange must be exactly
    # antisymmetric in the volume-integrated energy.
    sim = make_sim()
    st = make_state(sim, u_i=2.0e5, u_n=2.0e5, Ti=5.0)
    term = _collision(sim, st)
    Vp, V_En = _volumes(sim, st)
    act = _active(sim)
    resid = term.Ei * Vp + np.asarray(term.En) * V_En
    rel = _max_rel(resid, term.Ei * Vp, act)
    ok = rel < 1e-13 and np.any(term.Ei[act] != 0.0)
    return "C2 thermal-only limit: dEi*Vp == -dEn*V_En", ok, (
        f"max rel residual = {rel:.2e}"
    )


def gate_c3():
    # Ti == Tn (both above the Ti floor): the thermal channel vanishes and the
    # equal-mass split must give each species exactly half the dissipation.
    sim = make_sim()
    Ti_eV = 2.0
    st = make_state(sim, u_i=4.0e5, u_n=0.0, Ti=Ti_eV, Tn_K=Ti_eV * ev_to_erg / kb_cgs)
    term = ion_neutral_collision_rhs(
        state=st, floors=sim._floors, ion_mass_g=sim._ion_mass_g,
        gas_type="He", Tn_eV=TN_EV, geometry=sim._geometry,
    )
    Vp, V_En = _volumes(sim, st)
    u_rel = _u_rel(sim, st)
    total = -np.asarray(term.M) * u_rel * Vp
    act = _active(sim)
    ion_half = _max_rel(term.Ei * Vp - 0.5 * total, total, act)
    neu_half = _max_rel(np.asarray(term.En) * V_En - 0.5 * total, total, act)
    ok = ion_half < 1e-13 and neu_half < 1e-13 and np.any(total[act] != 0.0)
    return "C3 friction-only limit: each species takes exactly half", ok, (
        f"ion half rel={ion_half:.2e}  neutral half rel={neu_half:.2e}"
    )


def gate_c4():
    sim = make_sim()
    # (a) At rest (no friction) the ion row is the thermal channel alone, so
    #     it must equal the closed form evaluated at the PER-CELL Tn the field
    #     carries -- rates included, since nu_mt reads T_eff = (Ti + Tn)/2.
    #     This is an exact identity, not a magnitude heuristic.
    flat = make_state(sim, u_i=0.0, u_n=0.0, Ti=2.0)
    ramp_En = np.asarray(flat.En, dtype=float) * np.linspace(
        1.0, 40.0, flat.nn.shape[0]
    )
    ramped = make_state(sim, u_i=0.0, u_n=0.0, Ti=2.0, En_shape=ramp_En)
    Tn_ramp = neutral_temperature_eV(ramped, floors=sim._floors, Tn_eV=TN_EV)
    varies = float(np.max(Tn_ramp) / np.min(Tn_ramp))
    der = derive_state(ramped, floors=sim._floors, ion_mass_g=sim._ion_mass_g)
    nu_mt = np.asarray(ramped.nn, dtype=float) * (
        phelps_momentum_transfer_rate_cm3_s(
            0.5 * (der.Ti + Tn_ramp), gas_type="He"
        )
    )
    expected = 1.5 * nu_mt * ramped.n * (Tn_ramp - der.Ti) * ev_to_erg
    term_ramp = _collision(sim, ramped)
    act = _active(sim)
    rel = _max_rel(term_ramp.Ei - expected, expected, act)
    # The same closed form at the SCALAR Tn is a materially different answer,
    # which is what makes the identity above evidence that the field is read.
    scalar_nu = np.asarray(ramped.nn, dtype=float) * (
        phelps_momentum_transfer_rate_cm3_s(
            0.5 * (der.Ti + TN_EV), gas_type="He"
        )
    )
    scalar_form = 1.5 * scalar_nu * ramped.n * (TN_EV - der.Ti) * ev_to_erg
    separation = _max_rel(term_ramp.Ei - scalar_form, scalar_form, act)
    # (b) En set to exactly the scalar's own energy must reproduce the scalar
    #     result to the bit.
    no_field = make_state(sim, u_i=3.0e5, u_n=1.0e5, Ti=2.0)
    no_field = type(no_field)(
        n=no_field.n, nn=no_field.nn, M=no_field.M, Ee=no_field.Ee,
        Ei=no_field.Ei, M_n=no_field.M_n, En=None,
    )
    scalar_term = _collision(sim, no_field)
    matched_En = 1.5 * np.asarray(no_field.nn, dtype=float) * kb_cgs * TN_K
    matched = make_state(sim, u_i=3.0e5, u_n=1.0e5, Ti=2.0, En_shape=matched_En)
    field_term = _collision(sim, matched)
    bitwise = bool(
        np.array_equal(
            np.asarray(scalar_term.Ei).view(np.uint64),
            np.asarray(field_term.Ei).view(np.uint64),
        )
        and np.array_equal(
            np.asarray(scalar_term.M).view(np.uint64),
            np.asarray(field_term.M).view(np.uint64),
        )
    )
    ok = varies > 30.0 and rel < 1e-13 and separation > 1e-3 and bitwise
    return "C4 Tn is the per-cell field, and matches the scalar when equal", ok, (
        f"Tn max/min={varies:.1f}  per-cell closed-form rel={rel:.2e}  "
        f"separation from the scalar form={separation:.2e}  "
        f"scalar-match bitwise={bitwise}"
    )


def _wall(sim, state, alpha_E=None):
    return neutral_energy_wall_rhs(
        state=state,
        floors=sim._floors,
        ion_mass_g=sim._ion_mass_g,
        geometry=sim._geometry,
        Rm_cm=sim._geometry.Rm_cm,
        alpha_E=sim._neutral_energy_alpha if alpha_E is None else alpha_E,
        Tn_fit=float(sim._input_dict.get("Tn_fit", 0.1)),
        wall_rate_1_s=sim._wind_wall_rate,
    )


def gate_w1():
    sim = make_sim()
    at_wall = make_state(sim, u_i=0.0, u_n=0.0, Tn_K=NEUTRAL_ENERGY_FLOOR_T_K)
    hot = make_state(sim, u_i=0.0, u_n=0.0, Tn_K=4.0 * NEUTRAL_ENERGY_FLOOR_T_K)
    cold_En = 0.5 * neutral_energy_floor(at_wall.nn)
    cold = make_state(sim, u_i=0.0, u_n=0.0, En_shape=cold_En)
    zero = np.asarray(_wall(sim, at_wall).En)
    sink = np.asarray(_wall(sim, hot).En)
    source = np.asarray(_wall(sim, cold).En)
    exactly_zero = bool(np.all(zero == 0.0))
    ok = (
        exactly_zero
        and bool(np.all(sink <= 0.0))
        and bool(np.any(sink < 0.0))
        and bool(np.all(source >= 0.0))
        and bool(np.any(source > 0.0))
    )
    return "W1 wall sink: zero AT the floor energy, sink above, source below", ok, (
        f"at-wall all-zero={exactly_zero}  max sink={float(np.min(sink)):.3e}  "
        f"max source={float(np.max(source)):.3e}"
    )


def gate_w2():
    sim = make_sim()
    hot = make_state(sim, u_i=0.0, u_n=0.0, Tn_K=4.0 * NEUTRAL_ENERGY_FLOOR_T_K)
    base = np.asarray(_wall(sim, hot, alpha_E=0.40).En)
    doubled = np.asarray(_wall(sim, hot, alpha_E=0.80).En)
    off = np.asarray(_wall(sim, hot, alpha_E=0.0).En)
    live = np.abs(base) > 0.0
    rel = float(np.max(np.abs(doubled[live] / base[live] - 2.0)))
    ok = rel < 1e-14 and bool(np.all(off == 0.0)) and bool(np.any(live))
    return "W2 wall sink is exactly linear in alpha_E; alpha_E=0 is zero", ok, (
        f"max |ratio - 2| = {rel:.2e}  alpha_E=0 all-zero="
        f"{bool(np.all(off == 0.0))}"
    )


def gate_f1():
    sim = make_sim()
    cells = sim._geometry.cells
    nn = np.full(cells, 1.0e13)
    nn[0] = 0.5 * sim._floors["nn"]          # nn itself gets floored up
    below = neutral_energy_floor(nn) * 0.25  # En starts a quarter of the way
    above = neutral_energy_floor(nn) * 7.0
    En = np.where(np.arange(cells) % 2 == 0, below, above)
    state = conservative_from_primitives(
        n=np.full(cells, 1.0e12), nn=nn, u=np.zeros(cells),
        Te=np.full(cells, 15.0), Ti=np.full(cells, 2.0),
        ion_mass_g=sim._ion_mass_g, un=np.zeros(cells), Tn_K=TN_K,
    )
    state = type(state)(
        n=state.n, nn=state.nn, M=state.M, Ee=state.Ee, Ei=state.Ei,
        M_n=state.M_n, En=En,
    )
    floored = apply_state_floors(state, sim._floors, sim._ion_mass_g)
    expect = neutral_energy_floor(
        np.maximum(nn, sim._floors["nn"])
    )
    clipped = np.arange(cells) % 2 == 0
    clipped_ok = bool(
        np.array_equal(
            np.asarray(floored.En)[clipped].view(np.uint64),
            expect[clipped].view(np.uint64),
        )
    )
    kept_ok = bool(
        np.array_equal(
            np.asarray(floored.En)[~clipped].view(np.uint64),
            En[~clipped].view(np.uint64),
        )
    )
    Tn = neutral_temperature_eV(floored, floors=sim._floors, Tn_eV=np.nan)
    wall_eV = NEUTRAL_ENERGY_FLOOR_T_K * kb_cgs / ev_to_erg
    ok = clipped_ok and kept_ok and bool(np.all(Tn >= wall_eV * (1.0 - 1e-12)))
    return "F1 En floor clips to (3/2) nn_floored k T_wall, leaves the rest", ok, (
        f"clipped exact={clipped_ok}  above-floor untouched={kept_ok}  "
        f"min Tn={float(np.min(Tn)):.6e} eV vs wall {wall_eV:.6e} eV"
    )


def gate_p1():
    sim = make_sim(neutral_energy=False)
    st = sim.state
    names = state_field_names(st)
    terms = sim.rhs_terms()
    no_rows = all(term.En is None for term in terms.values())
    ok = (
        st.En is None
        and "En" not in names
        and no_rows
        and "neutral_energy_wall" not in terms
        and sim.rhs().size == len(names) * sim._geometry.cells
    )
    return "P1 flag off: no En field, no En rows, historical packed width", ok, (
        f"packed fields={names}  any En row={not no_rows}  "
        f"wall term registered={'neutral_energy_wall' in terms}"
    )


def _guard(label, expect_fragment, **overrides):
    try:
        make_sim(**overrides)
    except ValueError as exc:
        text = str(exc)
        ok = expect_fragment in text
        return label, ok, f"raised: {text[:96]}"
    return label, False, "no ValueError raised"


def gate_g1():
    return _guard(
        "G1 guard: neutral_energy without ion_neutral_moment_closure raises",
        "requires ion_neutral_moment_closure",
        ion_neutral_moment_closure=False,
    )


def gate_g2():
    return _guard(
        "G2 guard: neutral_energy without neutral_momentum raises",
        "requires neutral_momentum",
        neutral_momentum=False,
    )


def gate_g3():
    return _guard(
        "G3 guard: neutral_energy with coverage_closure raises",
        "incompatible with coverage_closure",
        coverage_closure=True,
    )


def gate_g4():
    results = []
    for model in ("kinetic", "kinetic_dvm"):
        label, ok, detail = _guard(
            f"neutral_model={model!r}",
            "incompatible with",
            neutral_model=model,
            neutral_two_zone=True,
        )
        results.append((ok, f"{label}: {detail}"))
    ok = all(entry[0] for entry in results)
    return "G4 guard: neutral_energy with either kinetic neutral model raises", ok, (
        " | ".join(entry[1] for entry in results)
    )


def gate_g5():
    outcomes = []
    for value in (-0.01, 1.5, float("nan")):
        label, ok, detail = _guard(
            f"alpha_E={value}",
            "must be in [0, 1]",
            neutral_energy_wall_accommodation=value,
        )
        outcomes.append((ok, f"{label}: {detail[:52]}"))
    happy = make_sim(neutral_energy_wall_accommodation=1.0)
    constructs = state_field_names(happy.state)[-1] == "En"
    ok = all(entry[0] for entry in outcomes) and constructs
    return "G5 guard: alpha_E outside [0, 1] raises; happy path packs En last", ok, (
        " | ".join(entry[1] for entry in outcomes)
        + f" | happy path packs En last={constructs}"
    )


def main():
    gates = [
        gate_c1, gate_c1_two_zone, gate_c2, gate_c3, gate_c4,
        gate_w1, gate_w2, gate_f1, gate_p1,
        gate_g1, gate_g2, gate_g3, gate_g4, gate_g5,
    ]
    all_ok = True
    print("NBL pass-1 neutral-energy gate suite (En field core)")
    print("=" * 76)
    for gate in gates:
        name, ok, detail = gate()
        all_ok = all_ok and ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        print(f"        {detail}")
    print("=" * 76)
    print("NBL pass-1 gates:", "ALL PASS" if all_ok else "FAILURES PRESENT")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
