"""Smoke cases: atomic rate models and cross sections."""

import numpy as np

from cablp.atomic.cross_sections import phelps_momentum_transfer_rate_cm3_s
from cablp.constants import ev_to_erg
from cablp.solvers._sim1d import LAPDSim1D
from cablp.solvers._sim1d.core.integrator import ssprk2_step
from cablp.solvers._sim1d.core.state import (
    STATE_NAMES_1D,
    conservative_from_primitives,
    derive_state,
    pack_state,
)
from cablp.solvers._sim1d.physics.conduction import (
    heat_conduction_rhs,
    implicit_heat_conduction_step,
)
from cablp.solvers._sim1d.physics.sources import (
    ion_neutral_collision_frequency,
)

from ._harness import _base_config, _base_sim, _case, _resolved_cathode_flags


# --------------------------------------------------------------------
# helium-only-reaction-rates
# --------------------------------------------------------------------
@_case(
    "helium-only-reaction-rates",
    historical_stance=True,
    provides=("expected_rhs_terms", "no_source_params"),
)
def _case_helium_only_reaction_rates(dt_default):
    params, flags = _base_config()
    sim, snapshot = _base_sim()
    geom = snapshot.geometry
    state = snapshot.state
    heat_state = conservative_from_primitives(
        n=np.full(geom.cells, 1.0e12),
        nn=state.nn,
        nn_a=state.nn_a,
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
    assert np.min(heat_rhs.Ee) < 0.0 < np.max(heat_rhs.Ee)
    assert np.min(heat_rhs.Ei) < 0.0 < np.max(heat_rhs.Ei)
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
    heat_dt = sim.suggest_timestep(y=pack_state(heat_state, neutral_two_zone=True))
    assert np.isfinite(heat_dt.dt_heat_conduction)
    heat_derived = derive_state(heat_state, sim.floors, sim.ion_mass_g)

    disabled_heat = heat_conduction_rhs(
        state=heat_state,
        floors=sim.floors,
        ion_mass_g=sim.ion_mass_g,
        mu=sim.mu,
        geometry=geom,
        heat_conduction=False,
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
    assert np.any(implicit_heat_derived.Te < heat_derived.Te)
    assert np.any(implicit_heat_derived.Te > heat_derived.Te)
    assert np.any(implicit_heat_derived.Ti < heat_derived.Ti)
    assert np.any(implicit_heat_derived.Ti > heat_derived.Ti)
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

    disabled_implicit = implicit_heat_conduction_step(
        state=heat_state,
        floors=sim.floors,
        ion_mass_g=sim.ion_mass_g,
        mu=sim.mu,
        geometry=geom,
        dt=heat_dt.dt_heat_conduction,
        heat_conduction=False,
    )
    assert np.allclose(disabled_implicit.Ee, heat_state.Ee)
    assert np.allclose(disabled_implicit.Ei, heat_state.Ei)

    nonheat_rhs = sim.rhs(pack_state(heat_state, neutral_two_zone=True), include_heat_conduction=False)
    full_rhs = sim.rhs(pack_state(heat_state, neutral_two_zone=True), include_heat_conduction=True)
    heat_rhs_y = pack_state(heat_rhs, neutral_two_zone=True)
    assert np.allclose(full_rhs - nonheat_rhs, heat_rhs_y)
    rhs_terms = sim.rhs_terms(pack_state(heat_state, neutral_two_zone=True), include_heat_conduction=True)
    expected_rhs_terms = {
        # The column/annulus zone exchange: the two-zone split is
        # unconditional, so its term is always present.
        "neutral_zone_exchange",
        "plasma_advective_flux",
        "boundary_absorption",
        "characteristic_boundary",
        # The end wall sheath debit: armed by the geometry's end wall face,
        # which this single-cathode layout carries.
        "end_wall_e_sheath_climb",
        "pressure_work",
        "hyperbolic_dissipation_heating",
        "ei_exchange",
        "ionization_energy_cost",
        "electron_ion_cooling",
        "electron_neutral_cooling",
        "ion_charge_exchange",
        "ion_neutral_drag",
        "ion_neutral_frictional_heating",
        "ion_neutral_thermalization",
        "ion_neutral_collision",
        "neutral_momentum_wall",
        "neutral_wind_advection",
        "surface_loss",
        "anode_collection",
        "cathode_surface_loss",
        "anode_e_sheath_loss",
        "neutral_exchange",
        "neutral_sources",
        "gas_puff_local_ionization",
        "ionization_birth",
        "beam_ionization_birth",
        "beam_power_deposition",
        "beam_ionization_cost",
        "beam_excitation_radiation",
        "recombination_rad_loss",
        "recombination_energy_return",
        "heat_conduction",
    }
    assert set(rhs_terms) == expected_rhs_terms
    term_sum = np.zeros_like(full_rhs)
    for term in rhs_terms.values():
        for field_name in STATE_NAMES_1D:
            assert np.all(np.isfinite(getattr(term, field_name)))
        term_sum = term_sum + pack_state(term, neutral_two_zone=True)
    assert np.allclose(term_sum, full_rhs)
    nonheat_terms = sim.rhs_terms(
        pack_state(heat_state, neutral_two_zone=True),
        include_heat_conduction=False,
    )
    assert np.allclose(pack_state(nonheat_terms["heat_conduction"], neutral_two_zone=True), 0.0)
    assert np.allclose(
        pack_state(rhs_terms["heat_conduction"], neutral_two_zone=True),
        full_rhs - nonheat_rhs,
    )
    # The electrode electron sheath pair: with no cathode solve BOTH rows are
    # zero, and the anode row is energy-only (Ee) in every configuration.
    assert np.allclose(pack_state(rhs_terms["cathode_surface_loss"], neutral_two_zone=True), 0.0)
    assert np.allclose(pack_state(rhs_terms["anode_e_sheath_loss"], neutral_two_zone=True), 0.0)
    for _zero_field in ("n", "nn", "M", "Ei"):
        assert np.allclose(
            getattr(rhs_terms["anode_e_sheath_loss"], _zero_field), 0.0
        )
    assert np.allclose(pack_state(rhs_terms["beam_ionization_birth"], neutral_two_zone=True), 0.0)
    assert np.allclose(pack_state(rhs_terms["beam_power_deposition"], neutral_two_zone=True), 0.0)
    assert np.allclose(pack_state(rhs_terms["beam_ionization_cost"], neutral_two_zone=True), 0.0)

    split_dt = min(1.0e-10, 0.1 * heat_dt.dt_heat_conduction)
    manual_explicit_y = ssprk2_step(
        y0=pack_state(heat_state, neutral_two_zone=True),
        dt=split_dt,
        rhs_func=lambda yy: sim.rhs(yy, include_heat_conduction=False),
        floor_func=sim.floor_state_vector,
    )
    manual_heat_state = sim.implicit_heat_conduction_step(
        dt=split_dt,
        y=manual_explicit_y,
    )
    manual_split_y = sim.floor_state_vector(pack_state(manual_heat_state, neutral_two_zone=True))
    split_y = sim.operator_split_step(y=pack_state(heat_state, neutral_two_zone=True), dt=split_dt)
    assert np.allclose(split_y, manual_split_y)

    no_heat_bound_dt = sim.suggest_timestep(
        y=pack_state(heat_state, neutral_two_zone=True),
        include_heat_conduction=False,
    )
    assert np.isinf(no_heat_bound_dt.dt_heat_conduction)
    assert no_heat_bound_dt.active_constraint != "heat_conduction"

    fast_state = conservative_from_primitives(
        n=np.full(geom.cells, params["ne0"]),
        nn=state.nn,
        nn_a=state.nn_a,
        u=np.full(geom.cells, 1.0e7),
        Te=np.full(geom.cells, params["Te0"]),
        Ti=np.full(geom.cells, params["Ti0"]),
        ion_mass_g=sim.ion_mass_g,
    )
    fast_dt = sim.suggest_timestep(y=pack_state(fast_state, neutral_two_zone=True))
    assert fast_dt.dt_plasma_cfl < dt_default.dt_plasma_cfl

    no_source_params = dict(params)
    no_source_params["gas_puff_enabled"] = False
    no_source_params["pump_enabled"] = False
    return locals()


# --------------------------------------------------------------------
# sigma-in-phelps
# --------------------------------------------------------------------
@_case(
    "sigma-in-phelps",
    provides=("cooling_kwargs", "shape_state"),
)
def _case_sigma_in_phelps(knob_floors, knob_mass, knob_state):
    # --- Momentum-transfer rate. The two legacy arms ("constant",
    # "cx_derived") were removed at D3, so the Phelps rate is the whole
    # function: nu_in = nn * (k_b + 1/2 k_iso)((Ti + Tn)/2). It must be
    # positive, finite, and exactly the tabulated rate at the effective
    # temperature.
    for Ti_probe in (0.1, 5.0):
        nu_in = ion_neutral_collision_frequency(nn=1e13, Ti=Ti_probe)
        assert np.isfinite(nu_in) and nu_in > 0.0
        assert np.isclose(
            nu_in,
            1e13
            * phelps_momentum_transfer_rate_cm3_s(
                0.5 * (Ti_probe + 0.025851)
            ),
            rtol=0.0,
        )

    # Reference state the ADAS cooling checks downstream read.
    shape_state = conservative_from_primitives(
        n=np.full(3, 1e12),
        nn=np.full(3, 1e13),
        u=np.zeros(3),
        Te=np.array([2.5, 5.0, 10.0]),
        Ti=np.full(3, 1.0),
        ion_mass_g=knob_mass,
    )
    cooling_kwargs = dict(
        state=shape_state,
        floors=knob_floors,
        ion_mass_g=knob_mass,
        I_ion=24.587,
        ionization_energy_cost=False,
    )
    return locals()


# --------------------------------------------------------------------
# adas-atomic-rate-model
# --------------------------------------------------------------------
@_case(
    "adas-atomic-rate-model",
    provides=("_b21p", "_he_2p_excitation_cross_cm2"),
)
def _case_adas_atomic_rate_model():
    # --- ADAS atomic rates: adf11 tables parse, grid nodes reproduce
    # exactly, edges clamp, and the physics the effective coefficients carry
    # shows up (effective SCD ionization above the direct ground-state rate at
    # low Te, radiation-only cooling below the IAEA fit).
    from cablp.atomic import adas as _adas
    from cablp.atomic.cross_sections import He_ion_rate_lkup
    from cablp.atomic.fits import IAEA_exp1
    from cablp.atomic.coefficients import aHeI

    scd_ne, scd_te, scd_stages = _adas.read_adf11(_adas.ADAS_DIR / "scd96_he.dat")
    assert scd_ne.shape == (24,) and scd_te.shape == (30,)
    assert set(scd_stages) == {1, 2}
    # Interpolation at a grid node returns the tabulated value exactly.
    node = _adas.he_ionization_rate(10.0 ** scd_ne[10], 10.0 ** scd_te[15])
    assert np.isclose(node, 10.0 ** scd_stages[1][15, 10], rtol=1e-12)
    # Edge clamping: below/above the Te grid returns the edge value.
    lo = _adas.he_ionization_rate(1e12, 10.0 ** scd_te[0])
    assert np.isclose(_adas.he_ionization_rate(1e12, 0.05), lo, rtol=1e-12)
    # Stepwise/metastable enhancement: effective SCD exceeds the direct
    # ground-state rate at low Te and converges toward it at high Te.
    assert _adas.he_ionization_rate(1e13, 5.0) > 2.0 * He_ion_rate_lkup(5.0)
    assert np.isclose(
        _adas.he_ionization_rate(1e12, 100.0), He_ion_rate_lkup(100.0), rtol=0.1
    )
    # Radiation-only cooling sits well below the IAEA fit (which carries the
    # ionization-potential loss).
    assert _adas.he_neutral_line_power(1e13, 8.0) < 0.5 * IAEA_exp1(8.0, aHeI)

    # Fused lookup (he_rates): one coordinate solve, N table blends -- must be
    # bit-identical to the single-table helpers, since both share the same
    # blend arithmetic on the same (verified-identical) grid.
    fuse_ne = 10.0 ** np.random.default_rng(1).uniform(8.0, 15.0, 64)
    fuse_Te = 10.0 ** np.random.default_rng(2).uniform(-0.6, 3.5, 64)
    fused = _adas.he_rates(fuse_ne, fuse_Te, ("scd", "acd", "plt1", "plt2", "prb1"))
    for name, single in (
        ("scd", _adas.he_ionization_rate),
        ("acd", _adas.he_recombination_rate),
        ("plt1", _adas.he_neutral_line_power),
        ("plt2", _adas.he_ion_line_power),
        ("prb1", _adas.he_recombination_power),
    ):
        assert np.all(fused[name] == single(fuse_ne, fuse_Te)), name

    # The float port of the 2^1P excitation cross section matches mpmath.
    from cablp.cathode.circuit_common import _he_2p_excitation_cross_cm2
    from cablp.atomic.cross_sections import He_EIE_cross_DA
    from cablp.atomic.coefficients import b_11s_21p as _b21p
    for eps_probe in (1.5, 100.0 / 21.218, 8.0):
        assert np.isclose(
            _he_2p_excitation_cross_cm2(eps_probe),
            float(He_EIE_cross_DA(eps_probe, _b21p)),
            rtol=1e-12,
        )
    return locals()


# --------------------------------------------------------------------
# retired-gas-type-and-rate-model-keys
# --------------------------------------------------------------------
@_case("retired-gas-type-and-rate-model-keys")
def _case_retired_gas_type_and_rate_model_keys():
    # The species is helium and the atomic rates are the OPEN-ADAS effective
    # coefficients, unconditionally. gas_type and atomic_rate_model are
    # retired keys: a configuration naming either is refused at construction
    # whatever value it carries -- the former defaults included, since the key
    # owns no read -- and the refusal names the key as retired and states
    # what is now unconditional.
    from cablp.solvers._sim1d.core.config import (
        RETIRED_PARAM_KEYS,
        default_config,
        input_dict_template_1d,
    )

    for _rk_key in ("gas_type", "atomic_rate_model"):
        assert _rk_key not in input_dict_template_1d, _rk_key
        assert _rk_key in RETIRED_PARAM_KEYS, _rk_key
    _rk_params, _rk_flags = default_config()
    for _rk_key, _rk_value in (
        ("gas_type", "He"),
        ("gas_type", "H"),
        ("atomic_rate_model", "adas"),
        ("atomic_rate_model", "janev"),
    ):
        try:
            LAPDSim1D(dict(_rk_params, **{_rk_key: _rk_value}), _rk_flags)
        except ValueError as _rk_exc:
            _rk_msg = str(_rk_exc)
            assert f"{_rk_key} is RETIRED" in _rk_msg, _rk_msg
            assert "unconditional" in _rk_msg, _rk_msg
        else:
            raise AssertionError(
                f"{_rk_key}={_rk_value!r} was accepted; a retired key must "
                "be refused at construction"
            )
    # Misfiled into the flag namespace it is a plain unknown key, not a
    # retired one: the retired register is per namespace.
    try:
        LAPDSim1D(_rk_params, dict(_rk_flags, gas_type="He"))
    except ValueError as _rk_exc:
        assert "flags=['gas_type']" in str(_rk_exc), str(_rk_exc)
        assert "RETIRED" not in str(_rk_exc), str(_rk_exc)
    else:
        raise AssertionError("gas_type in input_flags was accepted")


# --------------------------------------------------------------------
# he-singlet-manifold-registry
# --------------------------------------------------------------------
@_case("he-singlet-manifold-registry")
def _case_he_singlet_manifold_registry(_b21p, _he_2p_excitation_cross_cm2):
    # --- A1: the He singlet manifold registry (WP-A). ---
    from cablp.atomic.cross_sections import (
        He_EIE_cross_manifold,
        He_singlet_tail_cross,
    )
    from cablp.atomic.coefficients import He_singlet_manifold

    # The 2^1P row is the provenance anchor: same list object as b_11s_21p,
    # and the general evaluator reproduces the beam's float port (the ~7e-6
    # slack is the legacy E_21p = 21.217848 vs the registry's NIST 21.2180).
    assert He_singlet_manifold["21P"]["A"] is _b21p
    for eps_probe in (1.5, 100.0 / 21.218, 8.0):
        assert np.isclose(
            He_EIE_cross_manifold(
                eps_probe * He_singlet_manifold["21P"]["E_eV"],
                He_singlet_manifold["21P"],
            ),
            _he_2p_excitation_cross_cm2(eps_probe),
            rtol=1e-4,
        )

    # Every fitted level: zero at/below threshold, finite and non-negative
    # from just above threshold through 1 keV.
    manifold_probe_E = np.concatenate(
        [np.linspace(24.0, 200.0, 45), np.array([500.0, 1000.0])]
    )
    for level_name, entry in He_singlet_manifold.items():
        assert He_EIE_cross_manifold(entry["E_eV"], entry) == 0.0, level_name
        assert He_EIE_cross_manifold(0.5 * entry["E_eV"], entry) == 0.0, level_name
        for E_probe in manifold_probe_E:
            sigma_probe = He_EIE_cross_manifold(float(E_probe), entry)
            assert np.isfinite(sigma_probe) and sigma_probe >= 0.0, level_name
        assert He_EIE_cross_manifold(100.0, entry) > 0.0, level_name

    # The measured manifold multipliers at 100 eV (measure_beam_manifold.py,
    # 2026-07-20): R_events = 1.670, R_power = 1.730 against the historical
    # 2^1P-only booking. Loose bounds guard the transcribed coefficients
    # against digit regressions without over-pinning the fit evaluation.
    sigma_by_level = {
        name: He_EIE_cross_manifold(100.0, entry)
        for name, entry in He_singlet_manifold.items()
    }
    tail_sigma_100, tail_sigma_E_100 = He_singlet_tail_cross(100.0)
    manifold_sigma_100 = sum(sigma_by_level.values()) + tail_sigma_100
    manifold_sigma_E_100 = (
        sum(
            sigma_by_level[name] * He_singlet_manifold[name]["E_eV"]
            for name in sigma_by_level
        )
        + tail_sigma_E_100
    )
    r_events_100 = manifold_sigma_100 / sigma_by_level["21P"]
    r_power_100 = manifold_sigma_E_100 / (sigma_by_level["21P"] * 21.218)
    assert 1.55 < r_events_100 < 1.80, r_events_100
    assert 1.60 < r_power_100 < 1.85, r_power_100
    # The Eq. (5) Rydberg tail sums to ~1.56x the 4^1P row (sum of (4/n)^3
    # plus the small nS/nD/nF series and threshold shifts).
    assert 1.3 < tail_sigma_100 / sigma_by_level["41P"] < 2.0


# --------------------------------------------------------------------
# adas-low-te-extension-retired
# --------------------------------------------------------------------
@_case(
    "adas-low-te-extension-retired",
    historical_stance=True,
)
def _case_adas_low_te_extension_retired(m3_params):
    # --- The low-Te extension alone stays constructible.
    resolved_cathode_flags = _resolved_cathode_flags()
    LAPDSim1D(
        dict(m3_params, adas_low_te_extension=True), resolved_cathode_flags
    )

    # --- Te_floor must stay BELOW the adf11 low-Te grid edge. Below that edge every coefficient is clamped
    # to its edge value, so a floor at or above it makes the clamped band the
    # only band the plasma can occupy -- the atomic_rate_domain ledger's
    # "fraction below the table" is then zero by construction, and the
    # standing floor-below-the-edge ordering is false. The edge is read off
    # the loaded table, not written down, so this cannot drift from the data.
    from cablp.atomic.adas import (
        he_rate_temperature_range_eV as _tf_te_range,
    )

    _tf_edge_eV, _ = _tf_te_range()
    assert 0.0 < _tf_edge_eV < 1.0, _tf_edge_eV
    # Te0 is raised alongside the floor because the Te0 > Te_floor guard in
    # validate_r1_configuration_presence runs FIRST and would otherwise be the
    # refusal seen: this clause has to reach the table-edge guard itself.
    try:
        LAPDSim1D(
            dict(m3_params, Te_floor=0.25, Te0=0.5), resolved_cathode_flags
        )
    except ValueError as exc:
        assert "Te_floor" in str(exc), str(exc)
        assert repr(_tf_edge_eV) in str(exc), str(exc)
        assert "adf11" in str(exc), str(exc)
    else:
        raise AssertionError(
            "expected ValueError for Te_floor above the adf11 low-Te edge"
        )
    # The shipped floor is below the edge and constructs.
    assert float(m3_params["Te_floor"]) < _tf_edge_eV
    LAPDSim1D(dict(m3_params, Te_floor=0.1), resolved_cathode_flags)
    # The stance of record constructs unchanged -- built from
    # build_baseline_config(), so it follows every stance event automatically.
    from baseline_sim1d import build_baseline_config as _tf_baseline

    _tf_params, _tf_flags = _tf_baseline()
    assert float(_tf_params["Te_floor"]) < _tf_edge_eV
    LAPDSim1D(_tf_params, _tf_flags)


# --------------------------------------------------------------------
# gcr-recombination-energy-pair
# --------------------------------------------------------------------
@_case(
    "gcr-recombination-energy-pair",
    historical_stance=True,
)
def _case_gcr_recombination_energy_pair(m3_params):
    # --- GCR-consistent recombination energy pair
    # (recombination_energy_return): +I_ion*S_rec - P_PRB on the electron
    # fluid.
    resolved_cathode_flags = _resolved_cathode_flags()
    from cablp.atomic.adas import he_rates as _rer_he_rates
    from cablp.solvers._sim1d.physics.reactions import (
        recombination_energy_return_rhs,
    )

    rer_sim = LAPDSim1D(
        dict(m3_params, recombination_energy_return=True),
        resolved_cathode_flags,
    )
    rer_state = rer_sim.state
    rer_term = rer_sim.recombination_energy_return_rhs()
    rer_derived = derive_state(rer_state, rer_sim.floors, rer_sim.ion_mass_g)
    rer_rates = _rer_he_rates(
        np.maximum(rer_state.n, rer_sim.floors["n"]),
        rer_derived.Te,
        ("acd", "prb1"),
    )
    rer_I_ion = float(rer_sim._I_ion)
    rer_hand = ev_to_erg * rer_state.n * rer_state.n * (
        rer_I_ion * rer_rates["acd"] - rer_rates["prb1"]
    )
    assert np.allclose(rer_term.Ee, rer_hand, rtol=1e-12, atol=0.0)
    assert np.all(rer_term.n == 0.0) and np.all(rer_term.Ei == 0.0)
    # Present in the ledger; identically zero when the key is off (the
    # golden path sums an exact zero term).
    assert "recombination_energy_return" in rer_sim.rhs_terms()
    rer_off = LAPDSim1D(m3_params, resolved_cathode_flags)
    assert np.all(rer_off.recombination_energy_return_rhs().Ee == 0.0)
    # Direction: heating (I_ion > E_rad/event) at the clamped afterglow
    # floor (Te = 0.2 eV, the adf11 grid edge, where E_rad/event ~ 15 eV),
    # net sink in the hot ionizing plateau (PRB's bremsstrahlung/cascade
    # keeps radiating while ACD collapses, so E_rad/event >> I_ion there).
    rer_cold = _rer_he_rates(
        np.full(1, 1.0e13), np.full(1, 0.2), ("acd", "prb1")
    )
    rer_hot = _rer_he_rates(
        np.full(1, 5.0e12), np.full(1, 8.0), ("acd", "prb1")
    )
    assert rer_I_ion * rer_cold["acd"][0] > rer_cold["prb1"][0]
    assert rer_I_ion * rer_hot["acd"][0] < rer_hot["prb1"][0]
