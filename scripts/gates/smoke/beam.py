"""Smoke cases: the primary beam, CSDA deposition, the walked hot tail and the
quasilinear relaxation closure.
"""

import dataclasses
import math
from types import SimpleNamespace
import warnings

import numpy as np

from cablp.atomic import cross_sections as _cross_mod
from cablp.cathode import beam_deposition as _beam_deposition_mod
from cablp.cathode.beam_deposition import deposit_beam as _deposit_beam_ray
from cablp.cathode.circuit_common import compute_l_b
from cablp.constants import I_ion, ev_to_erg, qe_SI
from cablp.solvers._sim1d import LAPDSim1D, default_config
from cablp.solvers._sim1d.core.geometry import puff_cell_indices
from cablp.solvers._sim1d.core.state import derive_state
from cablp.solvers._sim1d.physics import cathode as _cathode_mod
from cablp.solvers._sim1d.physics.cathode import (
    _beam_smoothing_key,
    _beam_smoothing_matrix,
    _clip_ray_length,
    _csda_beam_deposition,
    _gap_clip_is_face_aligned,
    _ray_gap_breakout,
    beam_gap_ledger_mismatch,
    beam_launch,
)
from cablp.solvers._sim1d.physics.energy import (
    electron_cooling_rhs,
    electron_cooling_rhs_terms,
)
from cablp.solvers._sim1d.physics.neutrals import (
    gas_puff_rate_profile,
    neutral_zone_volumes,
    pump_rate,
)
from cablp.solvers._sim1d.physics.reactions import reaction_rates

from ._harness import (
    _base_config,
    _base_sim,
    _case,
    _cathode_flags,
    _cathode_unit_config,
    _pin_pre_r2a_neutral_stance,
    _tracking_electrode_sample,
)


def _qlr_config(**overrides):
    """Return a 12-cell, already-emitting current-driven cathode pair.

    The surface is held hot (no step's temperature increment survives this
    heat capacity, and nothing cleans at a zero cross section), so a beam
    launches inside a smoke-sized window and the anomalous closures have
    power to act on.
    """
    params, flags = default_config()
    params["nx"] = 12
    params["cathode_solver_model"] = "current_driven"
    params["initial_neutral_state"] = "fill"
    flags["cathode_coupling"] = True
    _pin_pre_r2a_neutral_stance(params, flags)
    params.update({
        "cathode_Ts_base_K": 1998.15,
        "cathode_heat_capacity_J_per_K": 1.0e30,
        "cathode_cleaning_sigma_cm2": 0.0,
        "cathode_cleaning_E_th_eV": None,
    })
    params.update(overrides)
    return params, flags


# --------------------------------------------------------------------
# beam-manifold-excitation-model
# --------------------------------------------------------------------
@_case(
    "beam-manifold-excitation-model",
    historical_stance=True,
)
def _case_beam_manifold_excitation_model():
    # --- A2: the manifold excitation channel the CSDA march books (WP-A).
    from cablp.atomic.cross_sections import (
        He_beam_excitation_channel as _He_manifold_channel,
    )
    from cablp.cathode.circuit_common import _he_2p_excitation_cross_cm2

    _mf_sigma, _mf_E = _He_manifold_channel(100.0)
    # Below the lowest manifold threshold (2^1S, 20.6158 eV).
    assert _He_manifold_channel(15.0) == (0.0, 0.0)
    # The measured manifold vs the 2^1P channel alone at 100 eV
    # (measure_beam_manifold.py, 2026-07-20): 1.67x the events, mean
    # radiated energy 21.98 eV.
    assert 1.55 < _mf_sigma / _he_2p_excitation_cross_cm2(100.0 / 21.218) < 1.80
    assert 21.5 < _mf_E < 22.5

    # Lookup-table front end (deposit_beam's hot path, 2026-07-21): exact
    # at the table nodes by construction; between nodes the interp error
    # must stay below the physics-irrelevant level, and the domain edges
    # must reproduce the exact function's contract.
    from cablp.atomic.cross_sections import He_beam_excitation_channel_lkup

    _lk_rng = np.random.default_rng(20260721)
    _lk_Es = np.concatenate([
        _lk_rng.uniform(20.7, 25.0, 40),   # threshold cluster
        _lk_rng.uniform(25.0, 180.0, 40),  # beam operating range
        _lk_rng.uniform(180.0, 1500.0, 20),
    ])
    for _lk_E in _lk_Es:
        _lk_s, _lk_e = He_beam_excitation_channel_lkup(float(_lk_E))
        _ex_s, _ex_e = _He_manifold_channel(float(_lk_E))
        assert abs(_lk_s - _ex_s) <= 1e-4 * _ex_s + 1e-21, (_lk_E, _lk_s, _ex_s)
        if _ex_s > 0.0:
            assert abs(_lk_e - _ex_e) <= 1e-4 * _ex_e, (_lk_E, _lk_e, _ex_e)
    assert He_beam_excitation_channel_lkup(15.0) == (0.0, 0.0)
    assert He_beam_excitation_channel_lkup(20.0) == (0.0, 0.0)
    # Above the table span: exact fallback, identical values.
    assert He_beam_excitation_channel_lkup(2500.0) == _He_manifold_channel(2500.0)


# --------------------------------------------------------------------
# beam-csda-deposition-model
# --------------------------------------------------------------------
@_case(
    "beam-csda-deposition-model",
    historical_stance=True,
    provides=(
        "csda_Vp", "csda_budget", "csda_dep", "csda_launch",
        "csda_params", "csda_power_sum", "csda_res", "csda_sigma_eff",
        "csda_sim", "csda_solve", "csda_terms",
    ),
)
def _case_beam_csda_deposition_model():
    # --- B2: the CSDA deposition module wired into the cathode solve.
    sim, snapshot = _base_sim()
    geom = snapshot.geometry
    cathode_flags = _cathode_flags()
    csda_params = dict(_base_config()[0])
    csda_sim = LAPDSim1D(csda_params, cathode_flags)
    csda_sim._circuit_I_loop = 3000.0
    csda_solve = csda_sim.solve_cathode_boundary()
    assert csda_solve.beam_deposition is not None
    csda_dep = csda_solve.beam_deposition[0]
    assert csda_dep is not None
    # Per-ray energy conservation through the module at solver conditions:
    # Gamma0*E0 = I_eth_star*phi_c (W -> erg/s is exactly 1e7).
    csda_res = csda_solve.beam_result.result
    csda_budget = csda_res.I_eth_star * csda_res.phi_c * 1.0e7
    csda_total = (
        csda_dep.plasma_heating_erg_s.sum()
        + csda_dep.radiated_erg_s.sum()
        + csda_dep.ionization_cost_erg_s.sum()
        # R4.1 anode interception is the production default (csda), so the
        # anode-removed energy is part of the per-ray budget.
        + float(csda_dep.anode_intercepted_erg_s)
        + csda_dep.transmitted_flux
        * csda_dep.transmitted_energy_eV
        * ev_to_erg
    )
    assert abs(csda_total - csda_budget) / csda_budget < 1e-9
    # The solver's four-term booking reproduces the module's split: the
    # power-deposition term carries heating + radiated + cost (plus the
    # historical gap-weighted P_ohmic), and the cost/radiation sinks
    # subtract back to the module's net heating.
    csda_terms = csda_sim.beam_ionization_rhs_terms(cathode_solve=csda_solve)
    csda_Vp = geom.plasma_volume_cm3
    csda_power_sum = float(
        (csda_terms["beam_power_deposition"].Ee * csda_Vp).sum()
    )
    csda_module_sum = float(
        csda_dep.plasma_heating_erg_s.sum()
        + csda_dep.radiated_erg_s.sum()
        + csda_dep.ionization_cost_erg_s.sum()
    )
    assert np.isclose(
        csda_power_sum - csda_res.P_ohmic * 1.0e7, csda_module_sum, rtol=1e-9
    )
    assert np.isclose(
        float((-csda_terms["beam_excitation_radiation"].Ee * csda_Vp).sum()),
        float(csda_dep.radiated_erg_s.sum()),
        rtol=1e-9,
    )
    assert np.isclose(
        float((-csda_terms["beam_ionization_cost"].Ee * csda_Vp).sum()),
        float(csda_dep.ionization_cost_erg_s.sum()),
        rtol=1e-9,
    )
    # CSDA primaries survive multiple events: ionization spreads over
    # several cells rather than one launch cell.
    assert np.count_nonzero(csda_dep.ionization_events) >= 2
    # The bypass adapter wrote a finite effective attenuation cross section
    # for the next solve's Beer-Lambert bypass.
    csda_launch = int(np.flatnonzero(csda_solve.beam_result.beam_cross)[0])
    csda_sigma_eff = float(
        csda_solve.beam_result.beam_atten_cross[csda_launch]
    )
    assert np.isfinite(csda_sigma_eff) and csda_sigma_eff >= 0.0
    return locals()


# --------------------------------------------------------------------
# beam-gap-transmission-probe
# --------------------------------------------------------------------
@_case(
    "beam-gap-transmission-probe",
    provides=("csda_L_cath", "csda_derived", "csda_dir", "csda_state"),
)
def _case_beam_gap_transmission_probe(
    csda_launch, csda_params, csda_res, csda_sigma_eff, csda_sim,
    csda_solve
):
    # --- Item 35: the gap-transmission probe is launched at the REAL emitted
    # flux, not unit flux, so flux-DEPENDENT stopping reaches the circuit.
    #
    # (a) The property the fix rests on, at the module: with flux-INDEPENDENT
    # stopping the surviving FRACTION does not depend on the launched flux, so
    # the flux-faithful probe reproduces the historical unit-flux probe
    # bit-for-bit; with the quasilinear closure it does depend on it (the
    # relaxation length runs on n_b ~ Gamma0/(A v_b)), which is exactly the
    # signal the unit-flux probe could not see.
    _fx_cells = 40
    _fx_dz = np.full(_fx_cells, 5.0)
    _fx_kwargs = dict(
        nn=np.full(_fx_cells, 1.0e13),
        ne=np.full(_fx_cells, 1.0e12),
        Te=np.full(_fx_cells, 3.0),
        dz_cm=_fx_dz,
        launch=0,
        direction=1,
        I_ion_eV=float(I_ion),
    )
    # Weak-beam domain (n_b < 0.1*ne), where the quasilinear closure is
    # defined; above it the module returns an infinite relaxation length by
    # design and the ray free-streams again.
    _fx_flux = 1.0e20
    _fx_lin = [
        _deposit_beam_ray(200.0, g, anomalous_model="none", **_fx_kwargs)
        for g in (1.0, _fx_flux)
    ]
    assert (
        float(_fx_lin[1].transmitted_flux) / _fx_flux
        == float(_fx_lin[0].transmitted_flux)
    )
    _fx_ql = [
        _deposit_beam_ray(
            200.0, g, anomalous_model="quasilinear",
            beam_area_cm2=100.0, **_fx_kwargs,
        )
        for g in (1.0, _fx_flux)
    ]
    # Unit flux streams through; the real flux is stopped by its own QL drag.
    assert float(_fx_ql[0].transmitted_flux) == 1.0
    assert float(_fx_ql[1].transmitted_flux) == 0.0
    # The probe-independent witness reads those same rays off their OWN
    # bookkeeping, with no reference to the probe -- this is what makes an
    # item-35-class probe defect visible without trusting the probe. The
    # QL-stopped ray above dies 75 cm in, so where it lands relative to the
    # gap is what decides breakout:
    _fx_gap_short = _clip_ray_length(_fx_dz, 0, 1, 50.0)   # ray dies past it
    _fx_gap_long = _clip_ray_length(_fx_dz, 0, 1, 100.0)   # ray dies inside it
    assert _ray_gap_breakout(_fx_ql[1], _fx_gap_short, 0, 1) == 1.0
    assert _ray_gap_breakout(_fx_ql[1], _fx_gap_long, 0, 1) == 0.0
    # Rays that leave the far end cleared the gap by definition (the
    # transmitted-flux shortcut, which also covers a sub-threshold ray whose
    # E_entry profile is all zeros).
    assert _ray_gap_breakout(_fx_ql[0], _fx_gap_long, 0, 1) == 1.0
    assert _ray_gap_breakout(_fx_lin[1], _fx_gap_long, 0, 1) == 1.0
    # Gap covering the whole path: no cell beyond it to sample, and the ray
    # did not leave the far end either, so it died inside.
    assert _ray_gap_breakout(
        _fx_ql[1], _clip_ray_length(_fx_dz, 0, 1, 1.0e4), 0, 1
    ) == 0.0

    # (b) The same statement through the solver: this csda config runs
    # anomalous_model="none", so the sigma_eff the adapter wrote must equal an
    # independent re-derivation from the HISTORICAL unit-flux probe. A
    # regression that made the flux-faithful probe non-flux-linear on the off
    # arm would move the golden's off-path arms and fail here.
    csda_state = csda_sim.state
    csda_derived = derive_state(
        csda_state, csda_sim._floors, csda_sim._ion_mass_g
    )
    csda_L_cath = float(csda_solve.device_config.L_cath)
    csda_dir = beam_launch(csda_sim._geometry, end=0)[1]
    csda_unit = _deposit_beam_ray(
        csda_res.phi_c,
        1.0,
        nn=csda_state.nn,
        ne=csda_state.n,
        Te=csda_derived.Te,
        dz_cm=_clip_ray_length(
            csda_sim._geometry.length_cm, csda_launch, csda_dir, csda_L_cath
        ),
        launch=csda_launch,
        direction=csda_dir,
        I_ion_eV=float(csda_sim._I_ion),
        coulomb_model="fast_electron",
        anomalous_model="none",
    )
    csda_unit_T = min(max(float(csda_unit.transmitted_flux), 1.0e-6), 1.0)
    csda_nn_launch = float(csda_state.nn[csda_launch])
    csda_l_bi = compute_l_b(
        csda_res.phi_c,
        float(csda_derived.Te[csda_launch]),
        float(csda_state.n[csda_launch]),
        0.0,
        0.0,
    )
    csda_sigma_unit = max(
        0.0,
        (-math.log(csda_unit_T) / csda_L_cath - 1.0 / csda_l_bi)
        / csda_nn_launch,
    )
    assert csda_sigma_eff == csda_sigma_unit
    return locals()


# --------------------------------------------------------------------
# beam-gap-ledger-tripwire
# --------------------------------------------------------------------
@_case(
    "beam-gap-ledger-tripwire",
    historical_stance=True,
    provides=("bl_diag", "csda_eta", "csda_ledger"),
)
def _case_beam_gap_ledger_tripwire(csda_sim, csda_solve, csda_params):
    # --- Item 35 ledger tripwire: probe, deposition ray and circuit are three
    # views of the SAME gap crossing, and nothing else in the model notices
    # when they disagree (each side is internally consistent). On a healthy
    # config all three agree and no warning fires.
    cathode_flags = _cathode_flags()
    csda_ledger = csda_solve.beam_gap_ledger
    csda_eta = float(csda_solve.device_config.eta)
    assert set(csda_ledger) == {0}
    csda_probe, csda_ray, csda_booked, csda_ceiling = csda_ledger[0]
    assert 0.0 < csda_probe <= 1.0 and 0.0 < csda_booked <= 1.0
    assert 0.0 < csda_ceiling <= 1.0
    assert beam_gap_ledger_mismatch(csda_ledger, csda_eta) is None
    # The deposition ray's breakout is read off its OWN bookkeeping, so it is
    # an independent witness: here the ray clears the gap and the probe agrees.
    assert csda_ray == 1.0
    assert csda_probe == csda_ray
    # This healthy config sits ON the benign floor the tolerance is chosen to
    # clear: the ray fully transmits, which the Beer-Lambert solve cannot
    # represent above its Coulomb-only ceiling, so the clamp saturates and
    # leaves a little unbooked. It must stay a small fraction of emitted beam
    # power -- the unit-flux probe defect reached 35% -- or the warning
    # becomes noise.
    assert 0.0 < csda_eta * (csda_ray - csda_booked) < 0.02
    # ... and the ceiling column says WHY it is unbooked rather than leaving
    # it to be re-derived: with the ray transmitting whole, the sigma_eff >= 0
    # clamp pins the circuit to the Coulomb-only ceiling EXACTLY, so the two
    # are the same number here and the whole shortfall is representability.
    assert csda_booked == csda_ceiling
    # The third case names that shortfall for what it is. It is opt-in, so
    # first: with it OFF the instrument is unchanged on a ledger that now
    # carries a ceiling -- this healthy one still reads clean.
    assert beam_gap_ledger_mismatch(
        csda_ledger, csda_eta, separate_representability=True
    ) is None
    # The MARGINAL-TRANSMISSION regime the case exists for: a broken-out ray
    # against a low ceiling, with the circuit clamped onto that ceiling. The
    # SAME excursion is reported either way and by the same power -- only the
    # label moves, from a divergence that is not happening to the
    # representability gap that is.
    csda_marginal = {0: (1.0, 1.0, 0.5, 0.5)}
    csda_marg_off = beam_gap_ledger_mismatch(csda_marginal, csda_eta)
    csda_marg_on = beam_gap_ledger_mismatch(
        csda_marginal, csda_eta, separate_representability=True
    )
    assert csda_marg_off[1] == "ray_vs_circuit"
    assert csda_marg_on[1] == "ray_vs_ceiling"
    assert csda_marg_on[4] == csda_marg_off[4]
    assert np.isclose(csda_marg_on[4], 0.5 * csda_eta)
    # The separation has to cut BOTH ways or it is just a rename: a ray that
    # breaks out while the circuit sits well BELOW the ceiling is a genuine
    # divergence (a broken probe propagated into sigma_eff), and it must stay
    # labelled one even with the case armed. Here the representability gap is
    # real but sub-tolerance, and the divergence is what trips.
    csda_breakout_hole = {0: (1.0, 1.0, 0.2, 0.9)}
    assert beam_gap_ledger_mismatch(
        csda_breakout_hole, csda_eta, separate_representability=True
    )[1] == "ray_vs_circuit"
    # A ray that did NOT break out is outside the case entirely -- the
    # ray_vs_circuit signature keeps its own label with the case armed.
    assert beam_gap_ledger_mismatch(
        {0: (1.0, 0.0, 0.96529, 0.99)}, csda_eta,
        separate_representability=True,
    )[1] == "probe_vs_ray"
    # Pre-ceiling ledger entries stay readable: no fourth element, no third
    # case, and the answer is the one the two-case instrument always gave.
    assert beam_gap_ledger_mismatch(
        {0: (1.0, 1.0, 0.5)}, csda_eta, separate_representability=True
    )[1] == "ray_vs_circuit"
    # The tripwire is a comparison, not an assertion about one side: an
    # injected divergence must be caught and reported as the worst offender.
    csda_trip = beam_gap_ledger_mismatch(
        {0: (csda_probe, csda_ray, 0.5 * csda_ray)}, csda_eta
    )
    assert csda_trip is not None
    assert csda_trip[0] == 0 and csda_trip[1] == "ray_vs_circuit"
    assert np.isclose(csda_trip[4], 0.5 * csda_eta * csda_ray)
    # A defect INSIDE the probe -- the item-35 class -- is caught by the
    # probe-vs-ray leg even though the circuit faithfully tracks the (wrong)
    # probe, which is exactly the configuration that stayed silent before.
    # Pre-fix this ledger read probe=1.0 with the ray at 0.0.
    csda_probe_defect = beam_gap_ledger_mismatch(
        {0: (1.0, 0.0, 0.96529)}, csda_eta
    )
    assert csda_probe_defect is not None
    assert csda_probe_defect[1] == "probe_vs_ray"
    assert np.isclose(csda_probe_defect[4], csda_eta)
    # ... and the warning it drives is emitted once per run, not per step.
    csda_broken = SimpleNamespace(
        beam_gap_ledger={0: (1.0, 0.0, 0.96529)},
        device_config=SimpleNamespace(eta=csda_eta),
    )
    with warnings.catch_warnings(record=True) as csda_warned:
        warnings.simplefilter("always")
        csda_sim._beam_gap_ledger_warned = False
        for _ in range(3):
            csda_sim._warn_beam_gap_ledger(csda_broken)
    csda_msgs = [
        w for w in csda_warned if "beam gap ledger" in str(w.message)
    ]
    assert len(csda_msgs) == 1
    assert "probe_vs_ray" in str(csda_msgs[0].message)
    csda_sim._beam_gap_ledger_warned = False
    # All three views are recorded as cathode diagnostics, defaulted on every
    # run so a frame with no CSDA ray (and an old file, which has none of the
    # three datasets) stays readable.
    csda_diag = csda_sim._cathode_diagnostic_snapshot()
    assert csda_diag["source_beam_gap_survival_probe"] == csda_probe
    assert csda_diag["source_beam_gap_survival_ray"] == csda_ray
    assert csda_diag["source_beam_gap_survival_circuit"] == csda_booked
    # Single cathode: the ``end_`` gap-survival views are ABSENT, not NaN.
    # Only the ``-1`` ray fills them and that ray is marched only under
    # ``TwinCathode``; the armed side is asserted at the twin fixture.
    assert "end_beam_gap_survival_ray" not in csda_diag
    bl_diag = LAPDSim1D(
        dict(csda_params), dict(cathode_flags)
    )._cathode_diagnostic_snapshot()
    for _bl_key in ("probe", "ray", "circuit"):
        assert np.isnan(bl_diag[f"source_beam_gap_survival_{_bl_key}"])
    return locals()


# --------------------------------------------------------------------
# beam-probe-skip
# --------------------------------------------------------------------
@_case(
    "beam-probe-skip",
    provides=("_pskip_Gamma0", "_pskip_geom", "_pskip_ray_kwargs"),
)
def _case_beam_probe_skip(
    csda_L_cath, csda_derived, csda_dir, csda_eta, csda_launch,
    csda_params, csda_res, csda_sim, csda_solve, csda_state
):
    # --- Probe skip (cost read 2026-08-02, restructure A) ------------------
    # When the deposition ray died inside the gap, the gap-transmission probe
    # is not launched at all: its transmitted flux is then the EXACT float
    # 0.0. The claim is checked HERE against the probe itself -- the pre-change
    # call, run verbatim -- on a state from each regime, so the skip is
    # demonstrated equal to the work it replaces rather than argued to be.
    _pskip_geom = csda_sim._geometry
    _pskip_gap = _clip_ray_length(
        _pskip_geom.length_cm, csda_launch, csda_dir, csda_L_cath
    )
    _pskip_Gamma0 = csda_res.I_eth_star / qe_SI
    _pskip_ray_kwargs = dict(
        launch=csda_launch,
        direction=csda_dir,
        I_ion_eV=float(csda_sim._I_ion),
        coulomb_model="fast_electron",
        anomalous_model=str(
            csda_params.get("beam_anomalous_model", "none")
        ),
    )
    if _pskip_ray_kwargs["anomalous_model"] != "none":
        _pskip_ray_kwargs["beam_area_cm2"] = _pskip_geom.plasma_area_cm2

    def _pskip_adapter(nn):
        """(beam_result, gap_ledger) from the real adapter on a doctored nn."""
        _beam = SimpleNamespace(
            result=csda_res,
            result_twin=None,
            beam_atten_cross=np.zeros(_pskip_geom.cells),
        )
        _, _ledger, _ = _csda_beam_deposition(
            _beam,
            SimpleNamespace(nn=nn, n=csda_state.n),
            SimpleNamespace(Te=csda_derived.Te),
            _pskip_geom,
            csda_solve.device_config,
            csda_params,
            float(csda_sim._I_ion),
        )
        return _beam, _ledger

    def _pskip_probe(nn):
        """Transmitted flux of the probe the pre-change code always launched."""
        return float(
            _deposit_beam_ray(
                csda_res.phi_c, _pskip_Gamma0, dz_cm=_pskip_gap,
                nn=nn, ne=csda_state.n, Te=csda_derived.Te,
                **_pskip_ray_kwargs,
            ).transmitted_flux
        )

    # The two structural guards are checked BOTH on the production machine and
    # on a pinned 50 cm fixture, and the pair is the point.
    #
    # The exact-zero skip is only VALID where the L_cath clip lands on a cell
    # face. Production lands on one: the CAD-span gap is 5 x 10.65 == 53.25
    # and L_cath is the same distance, and ``_clip_ray_length`` accumulates
    # forward so it hits the anode face exactly. That was NOT true between the
    # CAD-span adoption and the exactness fix of the same event -- the clip
    # decremented a running remainder, left a 3.55e-15 cm sliver on the
    # anode-crossing cell, and opened the item-35 gap ledger by 35.8 % of
    # emitted beam power. The pinned fixture is kept alongside because a
    # SECOND face-aligned mesh, arrived at by different arithmetic
    # (50.0 / 5 == 10.0 is exact in binary), keeps this case honest if the
    # production gap ever moves again.
    _pskip_fixture_params = dict(csda_params)
    _pskip_fixture_params["cathode_anode_gap_cm"] = 50.0
    _pskip_fixture_params["L_cath"] = 50.0
    # The fixed source region runs from the anode face outward, so its far end
    # rides the pinned gap or the span stops being a whole number of cells.
    _pskip_fixture_params["source_region_length_cm"] = 100.0
    _pskip_fixture_sim = LAPDSim1D(_pskip_fixture_params, _cathode_flags())
    _pskip_fixture_geom = _pskip_fixture_sim._geometry
    _pskip_fixture_launch, _pskip_fixture_dir = beam_launch(
        _pskip_fixture_geom, end=0
    )
    _pskip_fixture_gap = _clip_ray_length(
        _pskip_fixture_geom.length_cm,
        _pskip_fixture_launch,
        _pskip_fixture_dir,
        float(_pskip_fixture_params["L_cath"]),
    )
    assert _gap_clip_is_face_aligned(
        _pskip_fixture_gap, _pskip_fixture_geom.length_cm
    )
    assert float(
        _pskip_fixture_gap[int(_pskip_fixture_geom.anode_face_indices[0])]
    ) == 0.0
    # ... and so does the PRODUCTION machine, which is the regression guard for
    # the exactness fix: if the clip ever goes back to leaving a rounding
    # sliver at the anode face, these two fail here instead of surfacing as an
    # item-35 ledger warning buried in a capture log.
    assert _gap_clip_is_face_aligned(_pskip_gap, _pskip_geom.length_cm)
    assert float(_pskip_gap[int(_pskip_geom.anode_face_indices[0])]) == 0.0
    for _pskip_scale, _pskip_ray_expect in ((1.0, 1.0), (1.0e3, 0.0)):
        _pskip_nn = np.asarray(csda_state.nn, dtype=float) * _pskip_scale
        _pskip_beam, _pskip_ledger = _pskip_adapter(_pskip_nn)
        (
            _pskip_T, _pskip_ray, _pskip_circuit, _pskip_ceiling,
        ) = _pskip_ledger[0]
        assert _pskip_ray == _pskip_ray_expect, (_pskip_scale, _pskip_ray)
        # The probe, launched for real, must reproduce the branch taken --
        # bit-for-bit, not to a tolerance.
        _pskip_ref = min(
            max(_pskip_probe(_pskip_nn) / _pskip_Gamma0, 1.0e-6), 1.0
        )
        assert _pskip_T == _pskip_ref, (_pskip_scale, _pskip_T, _pskip_ref)
        # All the ledger channels stay written from the values they always
        # came from -- the skip removes a computation, not a diagnostic. The
        # ceiling rides along: it is a property of the STATE (Coulomb-only
        # mfp against the gap length), so the skip cannot reach it at all.
        for _pskip_v in (
            _pskip_T, _pskip_ray, _pskip_circuit, _pskip_ceiling,
        ):
            assert np.isfinite(_pskip_v), (_pskip_scale, _pskip_ledger)
        assert 0.0 < _pskip_T <= 1.0 and 0.0 < _pskip_circuit <= 1.0
        assert np.isfinite(_pskip_beam.beam_atten_cross[csda_launch])
        # ... and the tripwire still reads the same three views: probe and ray
        # agree on the skip arm (0.0 vs the 1e-6 clamp) exactly as they do on
        # the transmitting arm, so no divergence is manufactured.
        assert beam_gap_ledger_mismatch(_pskip_ledger, csda_eta) is None
    # The dead-ray arm is the one the skip fires on, and its probe really does
    # transmit the exact float zero (the value the branch substitutes).
    assert _pskip_probe(np.asarray(csda_state.nn, dtype=float) * 1.0e3) == 0.0
    # Guard: a clip that ends mid-cell truncates the stop cell, so the probe
    # could run out of path where the deposition ray still had some. The skip
    # must see that and stand down. Run on the PINNED fixture: these three
    # clip lengths are chosen against a 10 cm gap cell (45 lands mid-cell, 40
    # lands on a face), so they only mean what they say on that mesh.
    assert not _gap_clip_is_face_aligned(
        _clip_ray_length(
            _pskip_fixture_geom.length_cm,
            _pskip_fixture_launch, _pskip_fixture_dir, 45.0,
        ),
        _pskip_fixture_geom.length_cm,
    )
    assert _gap_clip_is_face_aligned(
        _clip_ray_length(
            _pskip_fixture_geom.length_cm,
            _pskip_fixture_launch, _pskip_fixture_dir, 40.0,
        ),
        _pskip_fixture_geom.length_cm,
    )
    # A clip longer than the whole path leaves every cell at its full length.
    assert _gap_clip_is_face_aligned(
        _clip_ray_length(
            _pskip_fixture_geom.length_cm,
            _pskip_fixture_launch, _pskip_fixture_dir, 1.0e6,
        ),
        _pskip_fixture_geom.length_cm,
    )
    return locals()


# --------------------------------------------------------------------
# beam-anode-mesh-interception
# --------------------------------------------------------------------
@_case(
    "beam-anode-mesh-interception",
    historical_stance=True,
)
def _case_beam_anode_mesh_interception(csda_dep):
    # --- R4.1 (audit A15): anode-mesh beam interception is unconditional
    # wherever the geometry resolves an anode face, so csda_sim above already
    # has it on -- the anode books energy and it is part of the csda per-ray
    # budget checked earlier.
    assert float(csda_dep.anode_intercepted_erg_s) > 0.0


# --------------------------------------------------------------------
# beam-walked-tail-fixtures
# --------------------------------------------------------------------
@_case(
    "beam-walked-tail-fixtures",
    historical_stance=True,
    provides=("k7_local_dep", "k7_local_diag", "k7_params"),
)
def _case_beam_walked_tail_fixtures(csda_params):
    # --- The walked-tail scenario the tail-forward and plateau cases share,
    # and the cathode-boundary key's own contract.
    #
    # The scenario needs a PRODUCTION-LIKE phi_c: capping the drop at 300 V
    # puts the plateau's top where the drive puts it.
    cathode_flags = _cathode_flags()
    k7_params = dict(csda_params, cathode_phi_c_cap_V=300.0)
    k7_local_sim = LAPDSim1D(
        dict(k7_params, heating_anomalous_transport="local"),
        dict(cathode_flags),
    )
    k7_local_sim._circuit_I_loop = 3000.0
    k7_local_solve = k7_local_sim.solve_cathode_boundary()
    k7_local_dep = k7_local_solve.beam_deposition[0]
    k7_local_diag = k7_local_sim._cathode_diagnostic_snapshot()
    assert float(k7_local_dep.heating_anomalous_erg_s.sum()) > 0.0, (
        "the scenario drives no QL power"
    )
    assert k7_local_dep.end_loss_tail_low_erg_s == 0.0
    assert k7_local_dep.end_loss_tail_high_erg_s == 0.0

    # (a) PRESENCE GATE. Under "local" the cathode-boundary key is inert at
    # either value: the boundary lives inside the walk and cannot be reached
    # from a configuration that never walks.
    k7_inert_sim = LAPDSim1D(
        dict(k7_params, heating_anomalous_tail_cathode_boundary="escape"),
        dict(cathode_flags),
    )
    k7_inert_sim._circuit_I_loop = 3000.0
    k7_inert_dep = k7_inert_sim.solve_cathode_boundary().beam_deposition[0]
    for _k7_arr in (
        "plasma_heating_erg_s", "heating_anomalous_erg_s",
        "radiated_erg_s", "ionization_cost_erg_s", "ionization_events",
    ):
        assert np.array_equal(
            getattr(k7_inert_dep, _k7_arr), getattr(k7_local_dep, _k7_arr)
        ), _k7_arr

    # (b) MISCONFIGURATION is loud at CONSTRUCTION: an unknown boundary, and
    # a twin machine whose two reflecting faces would trap the walkers with
    # no way out.
    for k7_bad_p, k7_bad_f in (
        (dict(k7_params, heating_anomalous_tail_cathode_boundary="bogus"),
         cathode_flags),
        (dict(k7_params, heating_anomalous_transport="plateau_multigroup"),
         dict(cathode_flags, TwinCathode=True)),
    ):
        try:
            LAPDSim1D(k7_bad_p, dict(k7_bad_f))
        except ValueError:
            pass
        else:
            raise AssertionError(
                "expected ValueError for the cathode-boundary selector "
                f"({k7_bad_p.get('heating_anomalous_tail_cathode_boundary')!r}, "
                f"twin={k7_bad_f.get('TwinCathode')})"
            )
    return locals()


# --------------------------------------------------------------------
# tail-forward-* : the walked tail's launch-direction split
# --------------------------------------------------------------------
#: The walked-tail ROUTES the split has to reach, as ``(label, extra
#: params)`` over ``k7_params`` + ``plateau_multigroup``: the march bounded by
#: the reflecting cathode face, and the one that free-escapes at both ends.
_TF_ROUTES = (
    ("reflect", {}),
    ("escape", {"heating_anomalous_tail_cathode_boundary": "escape"}),
)
#: Every per-cell row a tail launch can move.
_TF_ARRAYS = (
    "plasma_heating_erg_s", "heating_anomalous_erg_s",
    "heating_coulomb_erg_s", "heating_secondary_erg_s",
    "heating_terminal_erg_s", "radiated_erg_s", "ionization_cost_erg_s",
    "ionization_events", "excitation_events", "E_entry_eV",
    "ionization_events_tail", "excitation_events_tail",
    "ionization_cost_tail_erg_s", "radiated_tail_erg_s",
)
_TF_SCALARS = (
    "end_loss_tail_low_erg_s", "end_loss_tail_high_erg_s",
    "end_loss_low_erg_s", "end_loss_high_erg_s",
    "transmitted_flux", "transmitted_energy_eV",
)
_TF_KEY = "heating_anomalous_tail_forward_fraction"


def _tf_dep(k7_params, route_extra, forward=None):
    """One walked-tail deposition at this route, optionally at a stated split.

    ``forward=None`` leaves the key out of the supplied params entirely, which
    is the arm the default-inert case compares against.
    """
    tf_p = dict(k7_params, heating_anomalous_transport="plateau_multigroup")
    tf_p.update(route_extra)
    if forward is not None:
        tf_p[_TF_KEY] = forward
    tf_sim = LAPDSim1D(tf_p, dict(_cathode_flags()))
    tf_sim._circuit_I_loop = 3000.0
    return tf_sim.solve_cathode_boundary().beam_deposition[0]


def _tf_identical(dep_a, dep_b):
    """Byte-for-byte on every row and scalar a tail launch can move."""
    for tf_arr in _TF_ARRAYS:
        if not np.array_equal(getattr(dep_a, tf_arr), getattr(dep_b, tf_arr)):
            return tf_arr
    for tf_sc in _TF_SCALARS:
        if getattr(dep_a, tf_sc) != getattr(dep_b, tf_sc):
            return tf_sc
    return None


def _tf_centroid(row):
    """Cell-index centroid of a per-cell deposition row."""
    return float((np.arange(row.size) * row).sum() / row.sum())


# --------------------------------------------------------------------
# tail-forward-default-inert
# --------------------------------------------------------------------
@_case("tail-forward-default-inert", historical_stance=True)
def _case_tail_forward_default_inert(k7_params):
    # --- The launch-direction split at its symmetric default is INERT, byte
    # for byte, on every route that launches a tail walker. The default path
    # takes the historical branch verbatim (the same 0.5*flux expression, the
    # same two legs in the same order), so this is an equality on raw floats
    # and not a tolerance.
    for tf_label, tf_extra in _TF_ROUTES:
        tf_absent = _tf_dep(k7_params, tf_extra)
        tf_stated = _tf_dep(k7_params, tf_extra, forward=0.5)
        tf_diff = _tf_identical(tf_absent, tf_stated)
        assert tf_diff is None, (tf_label, tf_diff)
    return locals()


# --------------------------------------------------------------------
# tail-forward-energy-closure
# --------------------------------------------------------------------
@_case("tail-forward-energy-closure", historical_stance=True)
def _case_tail_forward_energy_closure(k7_params):
    # --- THE SPLIT MOVES NO POWER. The equivalent tail flux is P / E, so the
    # two directions' fluxes sum to the same total at any split and flux * E
    # returns the launched power to roundoff. Every eV the plateau groups
    # launch still ends in exactly one of {bulk heat via thermalization,
    # ionization investment, radiation, the tail end ledger, the anode mesh}.
    # The closure is stated against the groups' own launched power (the
    # STREAMING share of the bank).
    for tf_f in (0.5, 0.75, 1.0):
        tf_mg_dep = _tf_dep(k7_params, {}, forward=tf_f)
        tf_mg_launched = float(tf_mg_dep.tail_power_erg_s)
        assert tf_mg_launched > 0.0, tf_f
        tf_mg_culled = float(tf_mg_dep.tail_anode_culled_erg_s)
        tf_mg_delivered = (
            float(tf_mg_dep.heating_anomalous_erg_s.sum())
            + float(tf_mg_dep.ionization_cost_tail_erg_s.sum())
            + float(tf_mg_dep.radiated_tail_erg_s.sum())
            + float(tf_mg_dep.end_loss_tail_low_erg_s)
            + float(tf_mg_dep.end_loss_tail_high_erg_s)
            - float(tf_mg_dep.plateau_wave_power_erg_s)
        )
        assert tf_mg_culled > 0.0, tf_f
        assert abs(
            tf_mg_delivered + tf_mg_culled - tf_mg_launched
        ) / tf_mg_launched < 1e-12, (
            tf_f, tf_mg_launched, tf_mg_delivered, tf_mg_culled
        )
    return locals()


# --------------------------------------------------------------------
# tail-forward-direction
# --------------------------------------------------------------------
@_case("tail-forward-direction", historical_stance=True)
def _case_tail_forward_direction(k7_params):
    # --- WHERE the power lands is what the split moves. At f = 1.0 nothing
    # travels -z at all, so on the free-escape route the cathode-face row of
    # the tail end ledger is EXACTLY zero -- non-vacuously, because at the
    # symmetric launch it is positive.
    tf_escape = {"heating_anomalous_tail_cathode_boundary": "escape"}
    tf_esc_half = _tf_dep(k7_params, tf_escape, forward=0.5)
    tf_esc_full = _tf_dep(k7_params, tf_escape, forward=1.0)
    assert float(tf_esc_half.end_loss_tail_low_erg_s) > 0.0
    assert float(tf_esc_full.end_loss_tail_low_erg_s) == 0.0
    # THE CATHODE FACE IS NEVER REACHED. Under "reflect" there is no
    # reflection counter to read, and the cathode-face ledger row is zero at
    # every split by construction (that is what reflecting means), so the
    # statement is made where it is observable: at f = 1.0 the reflecting and
    # free-escaping arms are the SAME RUN, byte for byte, because no walker
    # arrives at that face for the convention to act on. At the symmetric
    # launch they differ, so the equality is a measurement and not an
    # identity.
    tf_ref_full = _tf_dep(k7_params, {}, forward=1.0)
    assert _tf_identical(tf_ref_full, tf_esc_full) is None
    assert _tf_identical(
        _tf_dep(k7_params, {}, forward=0.5), tf_esc_half
    ) is not None
    # ... and the deposited-power centroid moves toward +z as f rises.
    for tf_label, tf_extra in (("reflect", {}),):
        tf_centroids = [
            _tf_centroid(
                _tf_dep(k7_params, tf_extra, forward=tf_f)
                .heating_anomalous_erg_s
            )
            for tf_f in (0.5, 0.75, 1.0)
        ]
        assert (
            tf_centroids[0] < tf_centroids[1] < tf_centroids[2]
        ), (tf_label, tf_centroids)
    return locals()


# --------------------------------------------------------------------
# tail-forward-refusals
# --------------------------------------------------------------------
@_case("tail-forward-refusals", historical_stance=True)
def _case_tail_forward_refusals(k7_params):
    # --- Misconfiguration is loud at CONSTRUCTION, and every refusal names
    # the key. The domain is checked whether or not the walk is engaged (a
    # value off the range is wrong either way); the inert-use refusal is what
    # keeps the key from being a silent no-op under a stance that never walks.
    tf_walk = dict(k7_params, heating_anomalous_transport="plateau_multigroup")
    for tf_bad in (
        dict(k7_params, **{_TF_KEY: 1.0}),
        dict(k7_params, **{_TF_KEY: 0.75}),
        dict(tf_walk, **{_TF_KEY: 0.4}),
        dict(tf_walk, **{_TF_KEY: 0.0}),
        dict(tf_walk, **{_TF_KEY: -1.0}),
        dict(tf_walk, **{_TF_KEY: 1.1}),
        dict(tf_walk, **{_TF_KEY: float("nan")}),
        dict(tf_walk, **{_TF_KEY: float("inf")}),
    ):
        try:
            LAPDSim1D(tf_bad, dict(_cathode_flags()))
        except ValueError as tf_exc:
            assert _TF_KEY in str(tf_exc), str(tf_exc)
        else:
            raise AssertionError(
                f"expected ValueError for {_TF_KEY}={tf_bad[_TF_KEY]!r} under "
                f"heating_anomalous_transport="
                f"{tf_bad.get('heating_anomalous_transport')!r}"
            )
    # The default constructs under a stance that never walks -- the key is
    # inert there, not refused -- and every accepted value constructs under
    # the walked selection.
    LAPDSim1D(dict(k7_params, **{_TF_KEY: 0.5}), dict(_cathode_flags()))
    for tf_f in (0.5, 0.75, 1.0):
        LAPDSim1D(dict(tf_walk, **{_TF_KEY: tf_f}), dict(_cathode_flags()))
    return locals()


# --------------------------------------------------------------------
# beam-plateau-multigroup
# --------------------------------------------------------------------
@_case(
    "beam-plateau-multigroup",
    historical_stance=True,
)
def _case_beam_plateau_multigroup(k7_local_dep, k7_local_diag, k7_params):
    # --- The multi-group plateau closure: the QL bank carries a SPECTRUM,
    # not a line. The single-energy arms are its two heirs taken one at a
    # time (the shipped f = 0.25 line stood in for the WAVE share, f = 1.0
    # for the streaming one), so this value has to reproduce both at once and
    # conserve while it does. Unit level only -- what the recovered reach does
    # to the discharge is a campaign run.
    cathode_flags = _cathode_flags()

    # (a) MISCONFIGURATION IS LOUD AT CONSTRUCTION, in both namespaces.
    for _mg_bad_p, _mg_bad_f in (
        # an unknown selector string
        (dict(k7_params, heating_anomalous_transport="plateau_multigroups"),
         dict(cathode_flags)),
        # no anomalous channel: no power for the groups to carry
        (dict(k7_params, heating_anomalous_transport="plateau_multigroup",
              beam_anomalous_model="none"),
         dict(cathode_flags)),
        # WRONG NAMESPACE: a params key filed into flags is silent-inert and
        # is refused by the unknown-key guard on both sides.
        (dict(k7_params),
         dict(cathode_flags, heating_anomalous_transport="plateau_multigroup")),
    ):
        try:
            LAPDSim1D(_mg_bad_p, _mg_bad_f)
        except ValueError:
            pass
        else:
            raise AssertionError(
                "expected ValueError for "
                f"{_mg_bad_p.get('heating_anomalous_transport')!r} / "
                f"{sorted(set(_mg_bad_f) - set(cathode_flags))}"
            )

    # (b) THE DERIVED SPECTRUM. The edge solve is a genuine root of a MONOTONE
    # residual, the E^2-uniform edges are equal-power by construction, and the
    # two shares partition the bank exactly.
    _mg_ne, _mg_Te, _mg_Eb = 4.6e12, 9.6, 177.0
    _mg_E1, _mg_clamp = _beam_deposition_mod.plateau_edge_energy_eV(
        _mg_Eb, 1.17e19, _mg_ne, _mg_Te
    )
    assert _mg_clamp == 0
    assert _beam_deposition_mod.HE_E_STOP_EV < _mg_E1 < _mg_Eb, _mg_E1

    def _mg_residual(E1):
        # f_M(v_1) - m j_b / ((E_b - E_1) erg); the solve's own equation,
        # written out here so the root is checked against the STATEMENT and
        # not against the solver that produced it.
        f_M = _mg_ne * math.sqrt(
            _beam_deposition_mod._ME_CGS
            / (2.0 * math.pi * _mg_Te * _beam_deposition_mod._ERG_PER_EV)
        ) * math.exp(-E1 / _mg_Te)
        demand = (
            _beam_deposition_mod._ME_CGS * 1.17e19
            / ((_mg_Eb - E1) * _beam_deposition_mod._ERG_PER_EV)
        )
        return f_M - demand

    assert _mg_residual(_mg_E1 - 1.0e-6) > 0.0 > _mg_residual(_mg_E1 + 1.0e-6)
    # The clamp is REACHABLE and REPORTED, never silent: a beam flux the bulk
    # Maxwellian cannot supply at any edge above the floor lands on the floor.
    _mg_floor_E1, _mg_floor_clamp = (
        _beam_deposition_mod.plateau_edge_energy_eV(
            _mg_Eb, 1.0e21, _mg_ne, _mg_Te
        )
    )
    assert _mg_floor_clamp == -1
    assert _mg_floor_E1 == _beam_deposition_mod.HE_E_STOP_EV
    _mg_edges, _mg_mids = _beam_deposition_mod.plateau_group_edges_eV(
        _mg_E1, _mg_Eb, _beam_deposition_mod.PLATEAU_GROUP_COUNT
    )
    _mg_N = _beam_deposition_mod.PLATEAU_GROUP_COUNT
    _mg_w = np.diff(_mg_edges ** 2) / (_mg_Eb ** 2 - _mg_E1 ** 2)
    assert _mg_edges[0] == _mg_E1 and _mg_edges[-1] == _mg_Eb
    assert np.allclose(_mg_w, 1.0 / _mg_N, rtol=0.0, atol=1e-14), _mg_w
    assert np.all(_mg_edges[:-1] < _mg_mids) and np.all(_mg_mids < _mg_edges[1:])
    assert (
        (_mg_Eb + _mg_E1) / (2.0 * _mg_Eb)
        + (_mg_Eb - _mg_E1) / (2.0 * _mg_Eb)
    ) == 1.0

    # (c) THE CLOSURE THROUGH THE SOLVER: it conserves, and it carries BOTH
    # heirs. The withheld bank is measured independently as the anomalous
    # power the SAME ray banks under "local" -- the march is bit-identical in
    # both arms, so that sum IS P_QL.
    _mg_bank = float(k7_local_dep.heating_anomalous_erg_s.sum())
    assert _mg_bank > 0.0, "scenario drives no QL power"
    mg_sim = LAPDSim1D(
        dict(k7_params, heating_anomalous_transport="plateau_multigroup"),
        dict(cathode_flags),
    )
    mg_sim._circuit_I_loop = 3000.0
    mg_solve = mg_sim.solve_cathode_boundary()
    mg_dep = mg_solve.beam_deposition[0]
    mg_culled = float(mg_dep.tail_anode_culled_erg_s)
    mg_delivered = (
        float(mg_dep.heating_anomalous_erg_s.sum())
        + float(mg_dep.ionization_cost_tail_erg_s.sum())
        + float(mg_dep.radiated_tail_erg_s.sum())
        + float(mg_dep.end_loss_tail_low_erg_s)
        + float(mg_dep.end_loss_tail_high_erg_s)
    )
    assert mg_culled > 0.0, mg_culled
    assert abs(mg_delivered + mg_culled - _mg_bank) / _mg_bank < 1e-12, (
        _mg_bank, mg_delivered, mg_culled
    )
    # The two heirs partition the bank: the wave share is banked locally, the
    # streaming share is what was launched as walkers, and nothing else exists.
    mg_wave = float(mg_dep.plateau_wave_power_erg_s)
    mg_stream = float(mg_dep.tail_power_erg_s)
    assert mg_wave > 0.0 and mg_stream > 0.0
    assert abs((mg_wave + mg_stream) - _mg_bank) / _mg_bank < 1e-12
    # The edge is a property of the EXTRACTION and is carried per end, with
    # its clamp verdict beside it.
    mg_edge = mg_solve.beam_plateau_edge
    assert set(mg_edge) == {0}, mg_edge
    # THIS FIXTURE CLAMPS, and that is the physics rather than a defect: it is
    # a cold pre-breakdown cathode (launch cell ne ~ 1e9, Te ~ 0.2 eV against
    # a 300 V drop), where the bulk Maxwellian cannot reach the plateau level
    # the emitted flux demands at ANY edge above the inelastic floor. The
    # closure lands on the floor and SAYS SO -- which is the whole point of
    # the clamp being counted rather than swallowed.
    assert mg_edge[0][0] == _beam_deposition_mod.HE_E_STOP_EV
    assert mg_edge[0][0] < float(mg_solve.beam_result.result.phi_c)
    assert mg_edge[0][1] == -1
    # The census rows are PRESENCE-GATED: absent on every other value, so an
    # unarmed run's saved diagnostic structure is untouched.
    mg_diag = mg_sim._cathode_diagnostic_snapshot()
    assert mg_diag["source_beam_plateau_edge_eV"] == mg_edge[0][0]
    assert mg_diag["source_beam_plateau_edge_clamped"] == -1.0
    assert mg_diag["beam_plateau_wave_power_W"] > 0.0
    # The clamp CENSUS counts ACCEPTED STEPS, so it is zero until one is
    # taken and non-zero after -- never silent about a frame that clamped.
    assert mg_diag["plateau_edge_clamped_steps"] == 0.0
    assert math.isnan(mg_diag["plateau_edge_clamped_last_time_s"])
    mg_sim.advance_one_step()
    mg_stepped = mg_sim._cathode_diagnostic_snapshot()
    assert mg_stepped["plateau_edge_clamped_steps"] == 1.0, mg_stepped[
        "plateau_edge_clamped_steps"
    ]
    assert mg_stepped["plateau_edge_clamped_last_time_s"] == mg_sim._time
    for _mg_key in (
        "plateau_edge_clamped_steps", "plateau_edge_clamped_last_time_s",
        "beam_plateau_wave_power_W", "source_beam_plateau_edge_eV",
        "source_beam_plateau_edge_clamped",
    ):
        assert _mg_key not in k7_local_diag, _mg_key

    # (d) The streaming heir is a strictly interior share of the bank: this
    # closure neither banks all of it locally nor walks all of it.
    assert 0.0 < mg_stream < _mg_bank
    return locals()


# --------------------------------------------------------------------
# anode-tail-cull-crossing-rule
# --------------------------------------------------------------------
@_case("anode-tail-cull-crossing-rule", historical_stance=True)
def _case_anode_tail_cull_crossing_rule(
    _pskip_Gamma0, _pskip_geom, _pskip_ray_kwargs, csda_derived, csda_eta,
    csda_res, csda_state,
):
    """D1 (i)+(ii): WHERE the anode mesh takes its share of the tail.

    (i) at the frozen solver state, one deposition evaluation with the cull
    and one without: the cull removes exactly ``eta`` of what crosses and
    leaves ``1 - eta`` walking; (ii) on synthetic marches, a leg is culled
    when -- and only when -- its own direction carries it across the plane,
    strictly after its birth.
    """
    # --- (i) THE SHARE, at the frozen state the solver just solved. ONE ray,
    # evaluated twice: the ONLY difference between the two calls is whether
    # the tail cull kwargs are present, so nothing but the cull can move a
    # float between them. (Turning the whole mesh off instead would also move
    # the PRIMARY's interception and with it the QL power the tail is built
    # from, which is a different A/B.)
    ac_face = int(_pskip_geom.anode_face_indices[0])
    ac_eta = float(csda_eta)
    assert 0.0 < ac_eta < 1.0, ac_eta
    ac_ray_kwargs = dict(
        dz_cm=_pskip_geom.length_cm, nn=csda_state.nn, ne=csda_state.n,
        Te=csda_derived.Te, anode_cross_index=ac_face, anode_eta=ac_eta,
        anomalous_transport="tail_walk",
        tail_energy_eV=75.0,
        **_pskip_ray_kwargs,
    )
    ac_off = _deposit_beam_ray(
        csda_res.phi_c, _pskip_Gamma0, **ac_ray_kwargs
    )
    ac_on = _deposit_beam_ray(
        csda_res.phi_c, _pskip_Gamma0,
        tail_anode_cross_index=ac_face, tail_anode_eta=ac_eta,
        **ac_ray_kwargs,
    )
    assert float(ac_off.tail_anode_culled_erg_s) == 0.0
    ac_culled = float(ac_on.tail_anode_culled_erg_s)
    assert ac_culled > 0.0, "the cull never fired: the case is vacuous"
    # THE CULL IS THE ONLY MOVER, and what it moves it MOVES: the disarmed
    # ray's delivery is the armed ray's delivery plus the culled bank, to
    # roundoff, on one ray at one frozen state. This is the identity the
    # closure cases assert arm-to-arm, stated here where the two arms differ
    # in nothing but the cull kwargs.
    def _ac_delivered(ray):
        return (
            float(ray.heating_anomalous_erg_s.sum())
            + float(ray.ionization_cost_tail_erg_s.sum())
            + float(ray.radiated_tail_erg_s.sum())
            + float(ray.end_loss_tail_low_erg_s)
            + float(ray.end_loss_tail_high_erg_s)
        )

    assert abs(
        (_ac_delivered(ac_on) + ac_culled) / _ac_delivered(ac_off) - 1.0
    ) < 1e-12, (_ac_delivered(ac_off), _ac_delivered(ac_on), ac_culled)
    # ... and it takes it from the COLUMN SIDE. The walkers this scenario
    # launches are born on both sides of the plane, so the column-side
    # deposit is not simply (1 - eta) of the disarmed ray's -- the
    # walkers born past the plane never cross it -- but every walker that
    # DOES cross loses eta there, which is what the per-crossing equality
    # in (ii) below states exactly.
    ac_col_on = float(ac_on.heating_anomalous_erg_s[ac_face:].sum())
    ac_col_off = float(ac_off.heating_anomalous_erg_s[ac_face:].sum())
    assert ac_col_off > 0.0, "no tail power reaches the column side"
    assert ac_col_on < ac_col_off, (ac_col_on, ac_col_off)
    assert (1.0 - ac_eta) <= ac_col_on / ac_col_off < 1.0, (
        ac_col_on, ac_col_off, ac_eta
    )
    # FIRST CROSSING ONLY: no walker is culled twice, so the culled bank
    # cannot reach eta of the whole launched tail -- a walker culled at a
    # second crossing would take it past that bound.
    ac_launched = (
        float(ac_off.tail_power_erg_s)
        + float(ac_off.plateau_wave_power_erg_s)
    )
    assert 0.0 < ac_culled / ac_launched < ac_eta, (
        ac_culled, ac_launched, ac_eta
    )

    # --- (ii) THE CROSSING RULE, on synthetic marched TAIL legs. The plane
    # sits between cells X-1 and X; a leg crosses it by ENTERING the cell on
    # the far side in its own direction, and entering the cell it was born in
    # is not a crossing. Eight rows, both directions, both flanking cells and
    # one cell either side of them.
    #
    # Marched exactly as ``_tail_recursive_chains`` marches one: the leg's
    # own crossing cell, handed over through ``_leg_cull_kwargs``, which is
    # where the strictly-after-birth rule lives. It is stated there rather
    # than inside the march because the STREAMING PRIMARY has always been
    # intercepted on its own launch cell when asked to be, and the
    # deposit-beam corpus pins that arithmetic; the tail legs simply do not
    # ask.
    from cablp.cathode.beam_deposition import (
        _leg_cull_kwargs as _ac_leg_kwargs,
    )
    ac_cells, ac_X = 12, 6
    ac_nn = np.full(ac_cells, 1.0e13)
    ac_ne = np.full(ac_cells, 1.0e12)
    ac_Te = np.full(ac_cells, 3.0)
    ac_dz = np.full(ac_cells, 10.0)
    ac_kw = dict(
        I_ion_eV=24.587, E_stop_eV=25.0, coulomb_model="fast_electron",
        anomalous_model="none",
    )
    # The helper itself: withheld on the launch cell, handed over otherwise.
    assert _ac_leg_kwargs({"a": 1}, 6, 6) == {}
    assert _ac_leg_kwargs({"a": 1}, 6, 5) == {"a": 1}
    for ac_dir, ac_cell in ((+1, ac_X), (-1, ac_X - 1)):
        for ac_born in (ac_X - 2, ac_X - 1, ac_X, ac_X + 1):
            ac_ray = _deposit_beam_ray(
                75.0, 1.0e18, ac_nn, ac_ne, ac_Te, ac_born, ac_dir, ac_dz,
                **_ac_leg_kwargs(
                    dict(anode_cross_index=ac_cell, anode_eta=ac_eta),
                    ac_cell, ac_born,
                ),
                **ac_kw,
            )
            ac_frac = float(ac_ray.anode_intercepted_erg_s) / (
                1.0e18 * 75.0 * ev_to_erg
            )
            ac_expect = (
                (ac_dir > 0 and ac_born < ac_X)
                or (ac_dir < 0 and ac_born >= ac_X)
            )
            assert (ac_frac > 0.0) == ac_expect, (
                ac_dir, ac_born, ac_frac, ac_expect
            )
            if ac_expect:
                # ... and it takes exactly eta of the flux that reached the
                # plane, at the energy it arrived with.
                assert abs(
                    ac_frac
                    - ac_eta * float(ac_ray.E_entry_eV[ac_cell]) / 75.0
                ) < 1e-12, (ac_dir, ac_born, ac_frac)
    print(
        "anode tail cull crossing rule: ok (eta "
        f"{ac_eta:.4f}, column-side survival {ac_col_on / ac_col_off:.6f}, "
        f"culled share of launch {ac_culled / ac_launched:.4f})"
    )
    return locals()


# --------------------------------------------------------------------
# anode-tail-sheath-reflection
# --------------------------------------------------------------------
@_case("anode-tail-sheath-reflection", historical_stance=True)
def _case_anode_tail_sheath_reflection(csda_params):
    """The wires' sheath: absorbed at or above the drop, reflected below it.

    Synthetic walks either side of ``e*phi_a`` from BOTH directions, then the
    two shares summing to the whole intercepted bank on a plateau-like state
    at a stated drop.
    """
    ar_eta = float(csda_params["eta"])
    ar_cells, ar_X = 24, 8
    ar_nn = np.full(ar_cells, 3.0e13)
    ar_ne = np.full(ar_cells, 1.0e12)
    ar_Te = np.full(ar_cells, 3.0)
    ar_dz = np.full(ar_cells, 10.0)
    ar_common = dict(
        I_ion_eV=24.587, E_stop_eV=25.0, coulomb_model="fast_electron",
        anomalous_model="quasilinear", ql_relaxation_coeff=0.5,
        beam_area_cm2=np.full(ar_cells, 100.0),
        anomalous_transport="tail_walk", tail_energy_eV=60.0,
        product_transport="nonlocal",
    )

    def _ar_ray(phi_eV, direction, ionize):
        kw = dict(ar_common)
        kw["tail_ionization"] = "on" if ionize else "off"
        if ionize:
            kw["tail_walk_window"] = (0, ar_cells - 1)
        return _deposit_beam_ray(
            300.0, 1.0e18, ar_nn, ar_ne, ar_Te,
            0 if direction > 0 else ar_cells - 1, direction, ar_dz,
            tail_anode_cross_index=ar_X, tail_anode_eta=ar_eta,
            tail_anode_phi_eV=phi_eV, **kw,
        )

    def _ar_closure(ray):
        """Launched == delivered + culled, the tail bank's own identity."""
        launched = (
            float(ray.tail_power_erg_s)
            + float(ray.plateau_wave_power_erg_s)
        )
        delivered = (
            float(ray.heating_anomalous_erg_s.sum())
            + float(ray.ionization_cost_tail_erg_s.sum())
            + float(ray.radiated_tail_erg_s.sum())
            + float(ray.end_loss_tail_low_erg_s)
            + float(ray.end_loss_tail_high_erg_s)
        )
        return abs(
            delivered + float(ray.tail_anode_culled_erg_s) - launched
        ) / launched

    # --- BOTH SIDES OF THE BARRIER, both directions, both tail routes. The
    # SUM of what the wires keep and what their sheath turns back is the
    # whole intercepted bank and does not move with the drop; only the split
    # does. A drop far above every arrival reflects all of it, a drop below
    # every arrival keeps all of it, and 0.0 is the historical statement.
    for ar_ionize in (False, True):
        for ar_dir in (+1, -1):
            ar_base = _ar_ray(0.0, ar_dir, ar_ionize)
            ar_bank = float(ar_base.tail_anode_culled_erg_s)
            assert ar_bank > 0.0, (ar_ionize, ar_dir)
            assert float(
                ar_base.tail_anode_sheath_reflected_erg_s
            ) == 0.0, "a non-positive drop must reflect nothing"
            for ar_phi, ar_all_reflected in (
                (5.0, False), (1.0e4, True),
            ):
                ar_ray = _ar_ray(ar_phi, ar_dir, ar_ionize)
                ar_kept = float(ar_ray.tail_anode_culled_erg_s)
                ar_back = float(ar_ray.tail_anode_sheath_reflected_erg_s)
                assert abs(
                    (ar_kept + ar_back) / ar_bank - 1.0
                ) < 1e-12, (ar_ionize, ar_dir, ar_phi, ar_kept, ar_back)
                if ar_all_reflected:
                    # Far above every arrival: nothing reaches a wire.
                    assert ar_kept == 0.0, (ar_ionize, ar_dir, ar_kept)
                    assert ar_back > 0.0
                else:
                    # Far below every arrival: every one lands.
                    assert ar_back == 0.0, (ar_ionize, ar_dir, ar_back)
                    assert ar_kept == ar_bank
                # ENERGY IS CONSERVED EITHER WAY: a reflected walker stays in
                # the plasma, so the tail bank's closure holds at roundoff at
                # every drop.
                assert _ar_closure(ar_ray) < 1e-12, (
                    ar_ionize, ar_dir, ar_phi, _ar_closure(ar_ray)
                )
            # ... and a drop INSIDE the arrival spectrum splits it, which is
            # what makes the two corners above a bracket rather than a pair
            # of trivial cases.
            ar_mid = _ar_ray(58.0, ar_dir, ar_ionize)
            assert float(ar_mid.tail_anode_culled_erg_s) > 0.0
            assert float(ar_mid.tail_anode_sheath_reflected_erg_s) > 0.0
            assert abs(
                (float(ar_mid.tail_anode_culled_erg_s)
                 + float(ar_mid.tail_anode_sheath_reflected_erg_s))
                / ar_bank - 1.0
            ) < 1e-12
            assert _ar_closure(ar_mid) < 1e-12

    # --- THE REFLECTED LEG WALKS BACK, onto the side it came from. Read at
    # ``tail_forward_fraction = 1.0``, where every walker is launched +z, so
    # the only walkers that meet the plane are the ones born below it and
    # "the side it came from" is one side of the grid rather than two.
    # The (1 - eta) share passes either way, so the column side past the
    # plane is BIT-IDENTICAL between a keeping mesh and a mirroring one --
    # the whole difference is the eta share, and a mirror puts it back on
    # the gap side.
    def _ar_fwd_ray(phi_eV):
        kw = dict(ar_common)
        kw["tail_ionization"] = "off"
        kw["tail_forward_fraction"] = 1.0
        return _deposit_beam_ray(
            300.0, 1.0e18, ar_nn, ar_ne, ar_Te, 0, +1, ar_dz,
            tail_anode_cross_index=ar_X, tail_anode_eta=ar_eta,
            tail_anode_phi_eV=phi_eV, **kw,
        )

    ar_keep = _ar_fwd_ray(5.0)
    ar_mirror = _ar_fwd_ray(1.0e4)
    assert float(ar_keep.tail_anode_sheath_reflected_erg_s) == 0.0
    assert float(ar_mirror.tail_anode_culled_erg_s) == 0.0
    assert np.array_equal(
        ar_mirror.heating_anomalous_erg_s[ar_X:],
        ar_keep.heating_anomalous_erg_s[ar_X:],
    )
    assert float(
        ar_mirror.heating_anomalous_erg_s[:ar_X].sum()
    ) > float(ar_keep.heating_anomalous_erg_s[:ar_X].sum())
    # ... and the energy the mirror kept in the plasma is exactly the bank
    # the keeping mesh took out of it: what lands on the gap side plus what
    # leaves through the gap-side end.
    ar_back = (
        float(ar_mirror.heating_anomalous_erg_s[:ar_X].sum())
        - float(ar_keep.heating_anomalous_erg_s[:ar_X].sum())
        + float(ar_mirror.end_loss_tail_low_erg_s)
        - float(ar_keep.end_loss_tail_low_erg_s)
    )
    assert abs(
        ar_back / float(ar_keep.tail_anode_culled_erg_s) - 1.0
    ) < 1e-12, (ar_back, float(ar_keep.tail_anode_culled_erg_s))

    # --- REFUSALS. The drop is read only where there is a cull to apply it
    # to, and it has to be a number.
    for ar_bad in (
        dict(tail_anode_phi_eV=21.0),
        dict(tail_anode_cross_index=ar_X, tail_anode_eta=ar_eta,
             tail_anode_phi_eV=float("nan")),
    ):
        try:
            _deposit_beam_ray(
                300.0, 1.0e18, ar_nn, ar_ne, ar_Te, 0, +1, ar_dz,
                tail_ionization="off", **ar_common, **ar_bad,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"expected ValueError for tail_anode_phi_eV ({ar_bad})"
            )
    print(
        "anode wire-sheath reflection: ok (intercepted bank invariant in "
        "phi_a on both routes and both directions)"
    )
    return locals()


# --------------------------------------------------------------------
# anode-tail-circuit-coupling
# --------------------------------------------------------------------
@_case("anode-tail-circuit-coupling", historical_stance=True)
def _case_anode_tail_circuit_coupling():
    """The tail current in the loop, and the anode's electron cap.

    The cull's current reaches the circuit through ``tail_anode_current_A``,
    ONE STEP LAGGED: the deposition that measures it is solved after the
    circuit within a step, so a solve reads the previous accepted step's
    cull. At a frozen state the sheath answers monotonically, and the two
    sheath-fall powers add up to the whole current the anode collects times
    the drop.
    """
    from cablp.cathode.circuit_idriven import solve_idriven as _tc_idriven
    from cablp.cathode.circuit_prescribed import (
        solve_prescribed as _tc_prescribed,
    )
    from cablp.cathode import circuit_common as _tc_circ
    from cablp.plasma.params import (
        bohm_sound_speed as _tc_cs, electron_mean_speed as _tc_ve,
    )
    from cablp.constants import m_He_cgs as _tc_mi, qe_SI as _tc_e

    tc_params, tc_flags = _cathode_unit_config()
    from cablp.solvers._sim1d.physics.cathode import (
        cathode_device_config as _tc_device,
    )
    tc_cfg = _tc_device(tc_params, tc_flags, 4.0, _tc_mi)
    tc_Te, tc_ne = 4.0, 2.0e12
    tc_plasma = _tc_circ.PlasmaState(
        T_e=tc_Te, n_e=tc_ne, n_n=1.0e13, sigma_b=1.0e-16
    )
    # The anode sample the solver hands over, built by hand from the ONE
    # analytic form, so the identity below is a statement about the two
    # expressions and not about a sampled state.
    tc_A = 2.0 * tc_cfg.eta * tc_cfg.A_c
    tc_I_i_a = tc_A * _tc_e * tc_ne * _tc_cs(tc_Te, _tc_mi) * math.exp(-0.5)
    tc_I_e_sat = 0.25 * tc_ne * _tc_ve(tc_Te) * tc_A * _tc_e

    tc_variants = (
        ("idriven", _tc_idriven, dict(I_tot_A=3000.0)),
    )

    # --- (e2) THE ALGEBRAIC IDENTITY. Where ``I_i_a`` IS the analytic
    # e^(-1/2) n c_s collection on the wire area, the explicit random flux
    # and the implicit ``I_i_a exp(Lambda_a)`` are the SAME number, so the
    # rewritten sheath relation answers with the same phi_a. That is also the
    # live configuration: the solver's anode sample hands over exactly that
    # analytic collection, so the two forms agree to roundoff there too and
    # the explicit member is carried as the physical statement of the cap,
    # not as a change of value.
    assert abs(
        tc_I_e_sat / (tc_I_i_a * math.exp(tc_cfg.Lambda + 0.5)) - 1.0
    ) < 1e-13, (tc_I_e_sat, tc_I_i_a)
    for tc_name, tc_solve, tc_extra in tc_variants:
        tc_old = tc_solve(
            tc_cfg, tc_plasma, anode_current_A=tc_I_i_a, anode_T_e=tc_Te,
            **tc_extra,
        )
        tc_new = tc_solve(
            tc_cfg, tc_plasma, anode_current_A=tc_I_i_a, anode_T_e=tc_Te,
            anode_electron_saturation_A=tc_I_e_sat, **tc_extra,
        )
        assert abs(
            tc_new.phi_a - tc_old.phi_a
        ) <= 1e-12 * abs(tc_old.phi_a), (tc_name, tc_old.phi_a, tc_new.phi_a)
        assert abs(
            tc_new.P_anode_e - tc_old.P_anode_e
        ) <= 1e-12 * abs(tc_old.P_anode_e), tc_name

    # --- (d4) THE COUPLING IS LIVE, AND MONOTONE. At a frozen state, more
    # current arriving straight off the tail is less current the sheath has
    # to pass, so the anode sits higher. Read across a ladder rather than a
    # pair, so a non-monotone relation cannot slip through on two points.
    for tc_name, tc_solve, tc_extra in tc_variants:
        tc_phi = []
        for tc_I_tail in (0.0, 100.0, 300.0, 600.0):
            tc_res = tc_solve(
                tc_cfg, tc_plasma, anode_current_A=tc_I_i_a,
                anode_T_e=tc_Te, anode_electron_saturation_A=tc_I_e_sat,
                tail_anode_current_A=tc_I_tail, **tc_extra,
            )
            tc_phi.append(float(tc_res.phi_a))
            # THE NEW ROW: the circuit pays phi_a for the tail current the
            # wires took, and the two sheath-fall powers together are the
            # drop times the WHOLE electron current the anode collects.
            tc_I_e_a = float(tc_res.P_anode_e_thermal) / (
                2.0 * float(tc_res.T_e_anode)
            )
            assert abs(
                (float(tc_res.P_anode_e_phi) + float(tc_res.P_tail_phi))
                - float(tc_res.phi_a) * (tc_I_e_a + tc_I_tail)
            ) <= 1e-9 * abs(
                float(tc_res.P_anode_e_phi) + float(tc_res.P_tail_phi)
            ), (tc_name, tc_I_tail, tc_res.P_anode_e_phi, tc_res.P_tail_phi)
            assert float(tc_res.P_tail_phi) == (
                max(float(tc_res.phi_a), 0.0) * tc_I_tail
            ), (tc_name, tc_I_tail)
        assert all(
            tc_phi[k] < tc_phi[k + 1] for k in range(len(tc_phi) - 1)
        ), (tc_name, tc_phi)
        assert tc_phi[0] > 0.0, (tc_name, tc_phi)

    # The prescribed variant carries the same two members; it takes the
    # measured drive rather than solving for the current, so it is read on
    # its own call.
    tc_pre = _tc_prescribed(
        tc_cfg, tc_plasma, I_tot_A=3000.0, V_dis_V=90.0,
        anode_current_A=tc_I_i_a, anode_T_e=tc_Te,
        anode_electron_saturation_A=tc_I_e_sat, tail_anode_current_A=400.0,
    )
    assert float(tc_pre.P_tail_phi) == (
        max(float(tc_pre.phi_a), 0.0) * 400.0
    )
    print(
        "anode tail circuit coupling: ok (phi_a rises with I_tail,a; "
        "P_anode_e_phi + P_tail_phi closes; the coupling is ONE STEP "
        "LAGGED by construction)"
    )
    return locals()
# --------------------------------------------------------------------
# beam-deposition-smoothing-conservation
# --------------------------------------------------------------------
@_case(
    "beam-deposition-smoothing-conservation",
    historical_stance=True,
    provides=("smooth_sigma_cm",),
)
def _case_beam_deposition_smoothing_conservation(csda_params):
    # --- Beam-deposition smoothing CONSERVES the deposit over the live plasma.
    # The Gaussian redistribution kernel must place ZERO weight on the typed
    # plasma-dead cells (plenum/obstruction) behind the cathode face, because
    # the RHS mask ``_apply_active_plasma_topology`` zeroes exactly those rows:
    # anything the kernel spreads back there is silently DELETED, and it takes
    # beam power, beam ionization, excitation and the neutral debit with it
    # (all four channels share this one kernel). ``plasma_volume_cm3 > 0`` does
    # NOT identify those cells -- the dead cells have a finite plasma volume --
    # so the support has to come from ``plasma_active``.
    #
    # Checked on two meshes, the default one and one with a pinned fixed
    # source region (non-uniform: the source cells are shorter than the far
    # column's): the kernel is weighted by cell length, and without that
    # weighting a refined region is over-weighted per cm, which makes the
    # smoothing operator itself mesh-dependent even where it happens to
    # conserve.
    cathode_flags = _cathode_flags()
    smooth_sigma_cm = 50.0
    smoothing_meshes = (
        ("default", dict(csda_params), dict(cathode_flags)),
        (
            "pinned_source_grid",
            {
                **csda_params,
                # Gap pinned with the region: see _case_source_fixed_grid.
                "cathode_anode_gap_cm": 50.0,
                "source_region_length_cm": 100.0,
                "source_region_dz_cm": 10.0,
                "gas_puff_z_cm": 60.0,
            },
            dict(cathode_flags),
        ),
    )
    for mesh_label, smooth_base, smooth_flags in smoothing_meshes:
        smooth_off_sim = LAPDSim1D(dict(smooth_base), smooth_flags)
        smooth_on_sim = LAPDSim1D(
            {**smooth_base, "beam_deposition_smoothing_cm": smooth_sigma_cm},
            smooth_flags,
        )
        smooth_off_sim._circuit_I_loop = 3000.0
        smooth_on_sim._circuit_I_loop = 3000.0
        smooth_geom = smooth_on_sim.get_initial_snapshot().geometry
        smooth_active = np.asarray(smooth_geom.plasma_active, dtype=bool)
        smooth_Vp = np.asarray(smooth_geom.plasma_volume_cm3, dtype=float)
        smooth_dz = np.asarray(smooth_geom.length_cm, dtype=float)
        # The premise of the test: there ARE dead cells to leak into, and the
        # old ``Vp > 0`` support could not have found them.
        assert not smooth_active.all(), mesh_label
        assert (smooth_Vp > 0.0).all(), mesh_label
        if mesh_label == "pinned_source_grid":
            assert np.unique(np.round(smooth_dz[smooth_active], 9)).size > 1

        # (a) The kernel itself: no weight on any row the RHS mask will zero,
        # and every source column normalized to exactly 1 over the live support.
        smooth_W = _beam_smoothing_matrix(smooth_geom, smooth_sigma_cm)
        assert np.count_nonzero(smooth_W[~smooth_active, :]) == 0, mesh_label
        smooth_colsum = smooth_W[smooth_active, :].sum(axis=0)
        assert np.allclose(smooth_colsum, 1.0, rtol=0.0, atol=1e-12), (
            mesh_label,
            float(smooth_colsum.min()),
            float(smooth_colsum.max()),
        )

        # (b) The deposited RHS: smoothed-then-masked total == unsmoothed total,
        # channel by channel. Both sims are driven from the SAME cathode solve,
        # so the only difference between them is the smoothing operator.
        smooth_state = smooth_on_sim.state
        smooth_solve = smooth_on_sim.solve_cathode_boundary(
            state=smooth_state, update_cache=False
        )
        assert smooth_solve.beam_deposition is not None, mesh_label
        smooth_off_terms = smooth_off_sim.beam_ionization_rhs_terms(
            state=smooth_state, cathode_solve=smooth_solve
        )
        smooth_on_terms = smooth_on_sim.beam_ionization_rhs_terms(
            state=smooth_state, cathode_solve=smooth_solve
        )
        for smooth_term, smooth_field in (
            ("beam_ionization_birth", "n"),
            ("beam_ionization_birth", "nn"),
            ("beam_power_deposition", "Ee"),
            ("beam_ionization_cost", "Ee"),
            ("beam_excitation_radiation", "Ee"),
        ):
            off_row = np.asarray(
                getattr(smooth_off_terms[smooth_term], smooth_field), dtype=float
            )
            on_row = np.asarray(
                getattr(smooth_on_terms[smooth_term], smooth_field), dtype=float
            )
            off_total = float((off_row * smooth_Vp)[smooth_active].sum())
            on_total = float((on_row * smooth_Vp)[smooth_active].sum())
            # A zero channel would make the conservation check vacuous.
            assert abs(off_total) > 0.0, (mesh_label, smooth_term, smooth_field)
            assert abs(on_total - off_total) <= 1e-12 * abs(off_total), (
                mesh_label,
                smooth_term,
                smooth_field,
                on_total,
                off_total,
                on_total / off_total,
            )
            # ...and the kernel is not quietly the identity: it MOVED the
            # deposit, so the conservation above is a real statement.
            assert not np.allclose(on_row, off_row), (mesh_label, smooth_term)
    return locals()


# --------------------------------------------------------------------
# beam-smoothing-matrix-cache
# --------------------------------------------------------------------
@_case(
    "beam-smoothing-matrix-cache",
    historical_stance=True,
)
def _case_beam_smoothing_matrix_cache(csda_params, smooth_sigma_cm):
    # --- The smoothing-matrix cache is keyed on geometry CONTENT, not address.
    # ``id(geometry)`` is unique only among LIVE objects: CPython reuses the
    # address of a collected geometry, so a freed mesh followed by a
    # differently meshed allocation at the same address used to return the OLD
    # mesh's matrix. A cell-count mismatch would raise at the matmul; the
    # silent case is two meshes with the SAME cell count and different
    # positions -- exactly what an nx-matched source_region_dz_cm refinement
    # sweep builds.
    cathode_flags = _cathode_flags()
    smoothkey_flags = dict(cathode_flags)
    smoothkey_base = dict(
        csda_params,
        # Gap pinned with the region: see _case_source_fixed_grid.
        cathode_anode_gap_cm=50.0,
        source_region_length_cm=100.0,
        gas_puff_z_cm=60.0,
    )

    def _smoothkey_geometry(dz_cm, nx):
        sim = LAPDSim1D(
            dict(smoothkey_base, source_region_dz_cm=dz_cm, nx=nx),
            smoothkey_flags,
        )
        return sim.get_initial_snapshot().geometry

    # Halving the fixed source cell size doubles the fixed-region cells; nx is
    # cut by the same amount so the two meshes have IDENTICAL cell counts.
    smoothkey_geom_a = _smoothkey_geometry(10.0, 40)
    smoothkey_geom_b = _smoothkey_geometry(5.0, 35)
    smoothkey_geom_a2 = _smoothkey_geometry(10.0, 40)
    # Premises: same cell count, genuinely different meshes, distinct objects.
    assert smoothkey_geom_a.length_cm.size == smoothkey_geom_b.length_cm.size, (
        smoothkey_geom_a.length_cm.size,
        smoothkey_geom_b.length_cm.size,
    )
    assert not np.array_equal(smoothkey_geom_a.z_cm, smoothkey_geom_b.z_cm)
    assert smoothkey_geom_a is not smoothkey_geom_a2
    assert smoothkey_geom_a.length_cm.size == smoothkey_geom_a2.length_cm.size

    # (a)/(c) The regression: same cells, different spacing must NOT alias.
    assert _beam_smoothing_key(
        smoothkey_geom_a, smooth_sigma_cm
    ) != _beam_smoothing_key(smoothkey_geom_b, smooth_sigma_cm)
    smoothkey_W_a = _beam_smoothing_matrix(smoothkey_geom_a, smooth_sigma_cm)
    smoothkey_W_b = _beam_smoothing_matrix(smoothkey_geom_b, smooth_sigma_cm)
    assert smoothkey_W_a is not smoothkey_W_b
    assert not np.allclose(smoothkey_W_a, smoothkey_W_b)

    # (b) The cache still caches: two DISTINCT geometry objects with identical
    # content share the single O(cells^2) build. Guards the performance
    # property -- a key that accidentally never hits would run the build on
    # every RHS evaluation.
    smoothkey_W_a2 = _beam_smoothing_matrix(smoothkey_geom_a2, smooth_sigma_cm)
    assert smoothkey_W_a2 is smoothkey_W_a

    # The active-support term of the key is load-bearing: since the kernel is
    # built over ``plasma_active``, two meshes agreeing in z/lengths/faces but
    # differing in cell ROLES build different matrices and must not collide.
    smoothkey_active = np.asarray(
        smoothkey_geom_a.plasma_active, dtype=bool
    ).copy()
    smoothkey_active[-2] = not smoothkey_active[-2]
    smoothkey_geom_roles = dataclasses.replace(
        smoothkey_geom_a, plasma_active=smoothkey_active
    )
    assert _beam_smoothing_key(
        smoothkey_geom_a, smooth_sigma_cm
    ) != _beam_smoothing_key(smoothkey_geom_roles, smooth_sigma_cm)
    smoothkey_W_roles = _beam_smoothing_matrix(
        smoothkey_geom_roles, smooth_sigma_cm
    )
    assert smoothkey_W_roles is not smoothkey_W_a
    assert not np.allclose(smoothkey_W_roles, smoothkey_W_a)


# --------------------------------------------------------------------
# ionization-birth-energy-model
# --------------------------------------------------------------------
@_case(
    "ionization-birth-energy-model",
    historical_stance=True,
    provides=("short_phase_params", "source_rhs"),
)
def _case_ionization_birth_energy_model(csda_sim):
    # --- R4.2 (audit A14): ionization births book the cold-electron
    # convention -- the bulk electron birth energy is zero (no 3Te/2
    # creation) -- and the particle rows carry the births.
    params, flags = _base_config()
    sim, snapshot = _base_sim()
    geom = snapshot.geometry
    state = snapshot.state
    cons_react = csda_sim.reaction_rhs_terms()["ionization_birth"]
    assert np.all(cons_react.Ee == 0.0)
    assert np.any(cons_react.n > 0.0)

    rhs = sim.plasma_flux_rhs()
    # A uniform stationary plasma has no advective divergence -- exactly, on
    # every row, everywhere EXCEPT the momentum row at the plasma-terminating
    # faces. There the advective flux is deliberately zeroed and the ghost
    # boundary supplies the complete face condition (including its own
    # pressure), so the reflecting-wall pressure that used to cancel here is
    # gone. That is the retirement of the legacy closed-wall alternative
    # (see commit 1fc05c9), not a loss of well-balancedness: the interior
    # still cancels bit-for-bit, which is what the assertions below pin.
    _ib_absorbing = np.asarray(geom.plasma_absorbing, dtype=bool)
    _ib_live = np.asarray(geom.plasma_face_live_cell)
    _ib_term = sorted(
        {int(_ib_live[f]) for f in np.flatnonzero(_ib_absorbing)
         if int(_ib_live[f]) >= 0}
    )
    assert len(_ib_term) == 2, _ib_term
    # The plenum cell behind the cathode's interior absorbing face loses the
    # same wall pressure, so it moves too and is excluded with them.
    _ib_moved = set(_ib_term) | {
        int(f) - 1 for f in np.flatnonzero(_ib_absorbing) if int(f) - 1 >= 0
    }
    _ib_interior = np.array(
        [c for c in range(geom.cells) if c not in _ib_moved], dtype=int
    )
    for values in (rhs.n, rhs.nn, rhs.Ee, rhs.Ei):
        assert np.allclose(values, 0.0, atol=1e-20)
    _ib_M = np.asarray(rhs.M, dtype=float)
    assert np.allclose(_ib_M[_ib_interior], 0.0, atol=1e-20)
    # Non-vacuous, and directed OUT of the domain at each terminating cell:
    # -z at the cathode (plasma on its high-z side), +z at the end wall.
    _ib_cath, _ib_coll = _ib_term
    assert str(geom.cell_role[_ib_cath]) == "cathode"
    assert str(geom.cell_role[_ib_coll]) == "end_wall"
    assert _ib_M[_ib_cath] < 0.0
    assert _ib_M[_ib_coll] > 0.0
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
    source_puff, _ = puff_cell_indices(geom)
    # The puff is the orifice row scaled by the square envelope at the
    # evaluation time, booked in particles across the two zones (it feeds
    # the annulus wherever there is one).
    source_Vc, source_Va = neutral_zone_volumes(geom)

    def _source_particles(term):
        return term.nn * source_Vc + term.nn_a * source_Va

    def _puff_particles(time):
        return gas_puff_rate_profile(
            geom,
            sim._effective_gas_puff_sccm(time=time)[0],
            params["gas_puff_valves"],
            z_cm=params["gas_puff_z_cm"],
            orifice_id_cm=params["gas_puff_orifice_id_cm"],
            orifice_length_cm=params["gas_puff_orifice_length_cm"],
        ) * np.asarray(geom.neutral_volume_cm3, dtype=float)

    source_particles = _source_particles(source_rhs)
    assert source_particles[source_puff] > 0.0
    assert source_rhs.nn[0] < 0.0
    assert source_rhs.nn[-1] < 0.0
    assert np.isclose(
        source_particles[source_puff], _puff_particles(0.0)[source_puff]
    )
    assert np.isclose(
        source_rhs.nn[-1],
        -pump_rate(params["S_pump_R"], geom.neutral_volume_cm3[-1]) * state.nn[-1],
    )
    afterglow_time = params["tau_prebreakdown"] + params["tau_discharge"]
    afterglow_source = sim.neutral_source_sink_rhs(time=afterglow_time)
    assert np.isclose(
        afterglow_source.nn[0],
        -pump_rate(params["S_pump_L"], geom.neutral_volume_cm3[0]) * state.nn[0],
    )
    assert np.isclose(
        source_particles[source_puff]
        - _source_particles(afterglow_source)[source_puff],
        _puff_particles(0.0)[source_puff]
        - _puff_particles(afterglow_time)[source_puff],
    )
    assert np.isclose(afterglow_source.nn[-1], source_rhs.nn[-1])
    afterglow_source_terms = sim.rhs_terms(
        include_heat_conduction=False,
        time=params["tau_prebreakdown"] + params["tau_discharge"],
    )
    afterglow_dt_diag = sim.suggest_timestep(
        time=params["tau_prebreakdown"] + params["tau_discharge"]
    )
    assert np.isclose(
        afterglow_dt_diag.time,
        params["tau_prebreakdown"] + params["tau_discharge"],
    )
    assert afterglow_dt_diag.phase == "afterglow"
    assert afterglow_dt_diag.phase_cathode_enabled == 0.0
    # The square valve's closing tail keeps the afterglow puff switch open.
    assert afterglow_dt_diag.phase_gas_puff_enabled == 1.0
    assert afterglow_dt_diag.phase_floating == 1.0
    assert np.allclose(
        afterglow_source_terms["neutral_sources"].nn,
        afterglow_source.nn,
    )
    assert np.allclose(afterglow_source_terms["neutral_sources"].n, 0.0)
    assert (
        afterglow_dt_diag.dt_neutral_sources >= sim.suggest_timestep().dt_neutral_sources
    )
    # A short scheduled phase set with the pump off, reused by the gas-puff
    # diagnostics case.
    short_phase_params = dict(params)
    short_phase_params["pump_enabled"] = False
    short_phase_params["tau_prebreakdown"] = 1.0e-10
    short_phase_params["tau_discharge"] = 4.0e-10
    short_phase_params["tau_afterglow"] = 1.0e-10
    return locals()


# --------------------------------------------------------------------
# csda-module-standalone
# --------------------------------------------------------------------
@_case(
    "csda-module-standalone",
    provides=(
        "_COULOMB_STOPPING_EXPONENT", "_coulomb_stopping_coefficient",
        "b1_cells", "b1_col", "b1_res", "beam_speed_cm_s",
        "coulomb_stopping_eV_per_cm", "deposit_beam",
        "quasilinear_relaxation_length_cm",
    ),
)
def _case_csda_module_standalone():
    # --- B1: the standalone CSDA beam-deposition module
    # (B1; full acceptance in
    # scripts/verify/verify_beam_deposition.py — this is the fast subset).
    from cablp.cathode.beam_deposition import (
        _COULOMB_STOPPING_EXPONENT,
        _coulomb_stopping_coefficient,
        beam_speed_cm_s,
        coulomb_stopping_eV_per_cm,
        deposit_beam,
        quasilinear_relaxation_length_cm,
    )

    b1_cells = 30
    b1_col = dict(
        nn=np.full(b1_cells, 3.0e14),
        ne=np.full(b1_cells, 1.0e10),
        Te=np.full(b1_cells, 1.0),
        launch=0,
        direction=1,
        dz_cm=np.full(b1_cells, 100.0),
    )
    b1_res = deposit_beam(150.0, 1.0e22, **b1_col)
    b1_budget = 1.0e22 * 150.0 * 1.602176634e-12
    b1_total = (
        b1_res.plasma_heating_erg_s.sum()
        + b1_res.radiated_erg_s.sum()
        + b1_res.ionization_cost_erg_s.sum()
        + b1_res.transmitted_flux
        * b1_res.transmitted_energy_eV
        * 1.602176634e-12
    )
    assert abs(b1_total - b1_budget) / b1_budget < 1e-10
    # Breakdown conditions: several inelastic events per primary (the
    # single-event Beer-Lambert booking caps at 1).
    b1_events = (
        b1_res.ionization_events.sum() + b1_res.excitation_events.sum()
    ) / 1.0e22
    assert 2.0 < b1_events < 6.0, b1_events
    # Ray discipline: nothing behind the launch cell, direction respected.
    b1_res_rev = deposit_beam(
        150.0, 1.0e22, **{**b1_col, "launch": b1_cells - 1, "direction": -1}
    )
    assert np.allclose(
        b1_res_rev.ionization_events, b1_res.ionization_events[::-1]
    )
    # Closure ordering at production conditions: quasilinear <<
    # legacy tau_ei "Coulomb" << classical fast-electron stopping.
    b1_nb = 1.0e22 / (700.0 * beam_speed_cm_s(150.0))
    b1_lql = quasilinear_relaxation_length_cm(150.0, 5.0e12, b1_nb)
    b1_llegacy = 150.0 / coulomb_stopping_eV_per_cm(
        150.0, 5.0e12, 8.0, "legacy_tau_ei"
    )
    b1_lfast = 150.0 / coulomb_stopping_eV_per_cm(
        150.0, 5.0e12, 8.0, "fast_electron"
    )
    assert b1_lql < b1_llegacy < b1_lfast
    # Sub-threshold source passes through untouched; bad closures raise.
    b1_sub = deposit_beam(15.0, 1.0e22, **b1_col)
    assert b1_sub.transmitted_flux == 1.0e22
    assert b1_sub.plasma_heating_erg_s.sum() == 0.0
    for b1_bad, b1_says in (
        (
            lambda: deposit_beam(150.0, 1e22, **b1_col, coulomb_model="bogus"),
            "unknown coulomb_model 'bogus'",
        ),
        (
            lambda: deposit_beam(150.0, 1e22, **b1_col, anomalous_model="quasilinear"),
            "anomalous_model='quasilinear' needs beam_area_cm2",
        ),
    ):
        try:
            b1_bad()
        except ValueError as b1_error:
            assert b1_says in str(b1_error), (b1_says, str(b1_error))
        else:
            raise AssertionError("expected ValueError from deposit_beam")
    return locals()


# --------------------------------------------------------------------
# csda-per-cell-accumulators
# --------------------------------------------------------------------
@_case(
    "csda-per-cell-accumulators",
    provides=("_b2_weak",),
)
def _case_csda_per_cell_accumulators(
    b1_cells, b1_col, beam_speed_cm_s, coulomb_stopping_eV_per_cm,
    deposit_beam, quasilinear_relaxation_length_cm
):
    # --- Per-cell float accumulators (cost read 2026-08-02, restructure B) ---
    # deposit_beam banks each substep's channels in local Python floats and
    # flushes them to their arrays once, at cell exit, instead of doing eight
    # `arr[cell] += scalar` fancy-index stores per substep (14.5% of a
    # substep). The claim is bit-exactness, and it is checked here against a
    # reference march that keeps the OLD per-substep stores.
    #
    # The reference duplicates only the LOOP STRUCTURE -- the thing that
    # changed. Every physics leaf (the cross-section lookups, the stopping
    # powers, the secondary energy) is the module's own function, so a change
    # to the physics moves both sides together and only a change to the
    # marching/banking structure can make this fire. If that happens, this
    # reference must be re-derived from the module (or retired), never
    # loosened.
    from cablp.cathode.beam_deposition import (
        _ERG_PER_EV as _b2_ERG,
        HE_E_STOP_EV as _b2_E_STOP,
        HE_I_ION_EV as _b2_I_ION,
        he_mean_secondary_energy_eV as _b2_W_sec,
    )
    from cablp.atomic.cross_sections import (
        He_EII_cross_lkup as _b2_sigma_i,
        He_beam_excitation_channel_lkup as _b2_sigma_x,
    )

    def _b2_reference_march(
        E0_eV, Gamma0_per_s, nn, ne, Te, launch, direction, dz_cm,
        I_ion_eV=_b2_I_ION, E_stop_eV=_b2_E_STOP,
        coulomb_model="fast_electron", anomalous_model="none",
        beam_area_cm2=None, max_energy_fraction_per_substep=0.02,
        anode_cross_index=None, anode_eta=0.0,
        product_transport="local", anomalous_transport="local",
        tail_energy_eV=None,
    ):
        """The pre-restructure-B march: every bank written per SUBSTEP.

        Returns a dict of the per-cell arrays and the trajectory scalars.
        """
        cells = int(np.asarray(dz_cm).size)
        banks = {
            name: np.zeros(cells)
            for name in (
                "ionization_events", "excitation_events", "heating",
                "radiated", "ionization_cost", "E_entry", "heat_coulomb",
                "heat_anomalous", "heat_secondary", "heat_terminal",
                "sec_flux", "sec_power_eV", "anom_power_eV",
            )
        }
        walk_products = product_transport == "nonlocal"
        walk_tail = anomalous_transport == "tail_walk"
        area = np.broadcast_to(
            np.asarray(
                0.0 if beam_area_cm2 is None else beam_area_cm2, dtype=float
            ),
            (cells,),
        )
        frac = float(max_energy_fraction_per_substep)
        order = (
            range(launch, cells) if direction > 0 else range(launch, -1, -1)
        )
        E = float(E0_eV)
        gamma = float(Gamma0_per_s)
        absorbed = False
        anode_intercepted = 0.0
        terminal = (-1, 0.0, 0.0)
        intercept_active = anode_cross_index is not None and anode_eta > 0.0
        if E <= E_stop_eV:
            return dict(banks, transmitted_flux=gamma, transmitted_E=E,
                        anode_intercepted=0.0, terminal=terminal)
        for cell in order:
            if intercept_active and cell == anode_cross_index:
                anode_intercepted += anode_eta * gamma * E * _b2_ERG
                gamma *= 1.0 - anode_eta
                intercept_active = False
            banks["E_entry"][cell] = E
            remaining = float(dz_cm[cell])
            nn_c = float(nn[cell])
            ne_c = float(ne[cell])
            Te_c = float(Te[cell])
            while remaining > 0.0:
                sigma_i = (
                    _b2_sigma_i(E / I_ion_eV) if E > I_ion_eV else 0.0
                )
                sigma_x, E_rad = _b2_sigma_x(E)
                W_sec = _b2_W_sec(E, I_ion_eV=I_ion_eV)
                L_pot = nn_c * sigma_i * I_ion_eV
                L_sec = nn_c * sigma_i * W_sec
                L_exc = nn_c * sigma_x * E_rad
                L_coul = coulomb_stopping_eV_per_cm(
                    E, ne_c, Te_c, model=coulomb_model
                )
                L_anom = 0.0
                if anomalous_model == "quasilinear":
                    n_b = gamma / (float(area[cell]) * beam_speed_cm_s(E))
                    l_ql = quasilinear_relaxation_length_cm(E, ne_c, n_b)
                    if math.isfinite(l_ql) and l_ql > 0.0:
                        L_anom = E / l_ql
                L_tot = L_pot + L_sec + L_exc + L_coul + L_anom
                if L_tot <= 0.0:
                    break
                dz_sub = min(remaining, frac * E / L_tot)
                if E - L_tot * dz_sub <= E_stop_eV:
                    dz_sub = (E - E_stop_eV) / L_tot
                if dz_sub <= 0.0:
                    if walk_products:
                        terminal = (cell, gamma, E)
                    else:
                        banks["heating"][cell] += gamma * E * _b2_ERG
                        banks["heat_terminal"][cell] += gamma * E * _b2_ERG
                    E = 0.0
                    absorbed = True
                    break
                d_pot = L_pot * dz_sub
                d_sec = L_sec * dz_sub
                d_exc = L_exc * dz_sub
                d_coul = L_coul * dz_sub
                d_anom = L_anom * dz_sub
                banks["ionization_cost"][cell] += gamma * d_pot * _b2_ERG
                if walk_tail:
                    banks["anom_power_eV"][cell] += gamma * d_anom
                    d_anom_local = 0.0
                else:
                    d_anom_local = d_anom
                if walk_products:
                    banks["heating"][cell] += (
                        gamma * (d_coul + d_anom_local) * _b2_ERG
                    )
                    banks["sec_flux"][cell] += gamma * nn_c * sigma_i * dz_sub
                    banks["sec_power_eV"][cell] += gamma * d_sec
                else:
                    banks["heating"][cell] += (
                        gamma * (d_sec + d_coul + d_anom_local) * _b2_ERG
                    )
                    banks["heat_secondary"][cell] += gamma * d_sec * _b2_ERG
                banks["heat_coulomb"][cell] += gamma * d_coul * _b2_ERG
                banks["heat_anomalous"][cell] += gamma * d_anom_local * _b2_ERG
                banks["radiated"][cell] += gamma * d_exc * _b2_ERG
                banks["ionization_events"][cell] += (
                    gamma * nn_c * sigma_i * dz_sub
                )
                banks["excitation_events"][cell] += (
                    gamma * nn_c * sigma_x * dz_sub
                )
                E -= d_pot + d_sec + d_exc + d_coul + d_anom
                remaining -= dz_sub
                if E <= E_stop_eV:
                    if walk_products:
                        terminal = (cell, gamma, E)
                    else:
                        banks["heating"][cell] += gamma * E * _b2_ERG
                        banks["heat_terminal"][cell] += gamma * E * _b2_ERG
                    E = 0.0
                    absorbed = True
                    break
            if absorbed:
                break
        return dict(
            banks,
            transmitted_flux=0.0 if absorbed else gamma,
            transmitted_E=0.0 if absorbed else E,
            anode_intercepted=anode_intercepted,
            terminal=terminal,
        )

    # The banks the reference and the module must agree on, per cell. The
    # WP-D/WP-E withholding banks are not on the result object, so they are
    # compared through the arrays they end up in ("local" arms) and through
    # the walk products they drive ("nonlocal"/"tail_walk" arms).
    _b2_fields = (
        ("ionization_events", "ionization_events"),
        ("excitation_events", "excitation_events"),
        ("heating", "plasma_heating_erg_s"),
        ("radiated", "radiated_erg_s"),
        ("ionization_cost", "ionization_cost_erg_s"),
        ("E_entry", "E_entry_eV"),
        ("heat_coulomb", "heating_coulomb_erg_s"),
        ("heat_anomalous", "heating_anomalous_erg_s"),
        ("heat_secondary", "heating_secondary_erg_s"),
        ("heat_terminal", "heating_terminal_erg_s"),
    )
    # Representative states: the b1 breakdown column (ray absorbed mid-column,
    # many substeps per cell), a thin column the ray crosses whole
    # (transmitting, one substep-limited pass per cell), the quasilinear
    # closure (flux-dependent stopping), and the anode-mesh interception that
    # changes gamma mid-ray.
    _b2_thin = dict(
        nn=np.full(b1_cells, 1.0e12),
        ne=np.full(b1_cells, 1.0e10),
        Te=np.full(b1_cells, 3.0),
        launch=0, direction=1, dz_cm=np.full(b1_cells, 20.0),
    )
    _b2_cases = (
        ("absorbed", (150.0, 1.0e22), dict(b1_col)),
        ("transmitted", (150.0, 1.0e22), dict(_b2_thin)),
        ("reverse", (150.0, 1.0e22),
         {**b1_col, "launch": b1_cells - 1, "direction": -1}),
        ("quasilinear", (300.0, 1.0e20),
         {**_b2_thin, "anomalous_model": "quasilinear",
          "beam_area_cm2": 700.0}),
        ("anode", (150.0, 1.0e22),
         {**_b2_thin, "anode_cross_index": 5, "anode_eta": 0.358}),
    )
    for _b2_name, _b2_args, _b2_kw in _b2_cases:
        _b2_got = deposit_beam(*_b2_args, **_b2_kw)
        _b2_ref = _b2_reference_march(*_b2_args, **_b2_kw)
        for _b2_key, _b2_attr in _b2_fields:
            assert np.array_equal(
                getattr(_b2_got, _b2_attr), _b2_ref[_b2_key]
            ), (_b2_name, _b2_attr)
        assert _b2_got.transmitted_flux == _b2_ref["transmitted_flux"], _b2_name
        assert (
            _b2_got.transmitted_energy_eV == _b2_ref["transmitted_E"]
        ), _b2_name
        assert (
            float(_b2_got.anode_intercepted_erg_s)
            == _b2_ref["anode_intercepted"]
        ), _b2_name
        # A case that banks nothing would pass vacuously; every case must
        # actually deposit, and the absorbed ones must reach a terminal bank.
        assert _b2_ref["heating"].sum() > 0.0, _b2_name
    # The two withholding closures get their own arms, so the WP-D/WP-E banks
    # (flushed under their own `if` at cell exit) are exercised too, not just
    # the always-on eight. The tail arm needs the weak-beam domain
    # n_b < 0.1 n_e, or the QL relaxation length is infinite by design and
    # there is no anomalous power to withhold.
    _b2_weak = dict(
        nn=np.full(b1_cells, 1.0e13),
        ne=np.full(b1_cells, 1.0e12),
        Te=np.full(b1_cells, 3.0),
        launch=0, direction=1, dz_cm=np.full(b1_cells, 20.0),
    )
    for _b2_name, _b2_args, _b2_kw, _b2_transport in (
        ("wpd-absorbed", (150.0, 1.0e22), dict(b1_col),
         {"product_transport": "nonlocal"}),
        ("wpd-thin", (150.0, 1.0e22), dict(_b2_thin),
         {"product_transport": "nonlocal"}),
        ("wpe-weak", (200.0, 1.0e20), dict(_b2_weak),
         {"anomalous_transport": "tail_walk", "tail_energy_eV": 75.0,
          "anomalous_model": "quasilinear", "beam_area_cm2": 100.0}),
    ):
        _b2_full = {**_b2_kw, **_b2_transport}
        _b2_got = deposit_beam(*_b2_args, **_b2_full)
        _b2_ref = _b2_reference_march(*_b2_args, **_b2_full)
        # The walks run after the march and add to `heating`, so the
        # comparable per-cell banks here are the ones the march alone
        # writes plus the withheld populations themselves.
        for _b2_key, _b2_attr in (
            ("ionization_events", "ionization_events"),
            ("excitation_events", "excitation_events"),
            ("ionization_cost", "ionization_cost_erg_s"),
            ("radiated", "radiated_erg_s"),
            ("E_entry", "E_entry_eV"),
            ("heat_coulomb", "heating_coulomb_erg_s"),
        ):
            assert np.array_equal(
                getattr(_b2_got, _b2_attr), _b2_ref[_b2_key]
            ), (_b2_name, _b2_transport, _b2_attr)
        assert (
            _b2_got.transmitted_flux == _b2_ref["transmitted_flux"]
        ), (_b2_name, _b2_transport)
        if "product_transport" in _b2_transport:
            assert _b2_ref["sec_flux"].sum() > 0.0
        else:
            assert _b2_ref["anom_power_eV"].sum() > 0.0
    return locals()


# --------------------------------------------------------------------
# csda-hoisted-stopping-coefficient
# --------------------------------------------------------------------
@_case("csda-hoisted-stopping-coefficient")
def _case_csda_hoisted_stopping_coefficient(
    _COULOMB_STOPPING_EXPONENT, _b2_weak, _coulomb_stopping_coefficient,
    b1_cells, b1_col, b1_res, coulomb_stopping_eV_per_cm, deposit_beam
):
    # --- Hoisted stopping coefficient (cost read 2026-08-02, restructure C) --
    # The walks' per-cell A in dE/dx = A W**p is a 262-iteration Python
    # listcomp costing ~100 us -- half the entire WP-E per-call surcharge --
    # and it depends only on (ne, Te, model). deposit_beam now accepts it from
    # the caller so several rays, or a future WP-F's energy groups, pay for it
    # once. Supplying it must be bit-identical to letting the module build it.
    _b3_kw = {
        **_b2_weak, "anomalous_transport": "tail_walk",
        "tail_energy_eV": 75.0, "anomalous_model": "quasilinear",
        "beam_area_cm2": 100.0,
    }
    _b3_coeff = _coulomb_stopping_coefficient(
        _b2_weak["ne"], _b2_weak["Te"], "fast_electron"
    )
    _b3_auto = deposit_beam(200.0, 1.0e20, **_b3_kw)
    _b3_given = deposit_beam(
        200.0, 1.0e20, **_b3_kw, stopping_coefficient=_b3_coeff
    )
    for _b3_field in (
        "ionization_events", "excitation_events", "plasma_heating_erg_s",
        "radiated_erg_s", "ionization_cost_erg_s", "E_entry_eV",
        "heating_coulomb_erg_s", "heating_anomalous_erg_s",
        "heating_secondary_erg_s", "heating_terminal_erg_s",
    ):
        assert np.array_equal(
            getattr(_b3_auto, _b3_field), getattr(_b3_given, _b3_field)
        ), _b3_field
    for _b3_scalar in (
        "transmitted_flux", "transmitted_energy_eV",
        "end_loss_tail_low_erg_s", "end_loss_tail_high_erg_s",
    ):
        assert getattr(_b3_auto, _b3_scalar) == getattr(_b3_given, _b3_scalar)
    # Non-vacuous: the tail walk actually carried power on this state.
    assert _b3_auto.heating_anomalous_erg_s.sum() > 0.0
    assert float(_b3_auto.end_loss_tail_high_erg_s) > 0.0
    # Presence gating: the default is None and behaves as it always did.
    assert np.array_equal(
        deposit_beam(
            150.0, 1.0e22, **b1_col, stopping_coefficient=None
        ).plasma_heating_erg_s,
        b1_res.plasma_heating_erg_s,
    )
    # A wrong-length coefficient is a loud failure at the call, never a silent
    # mis-walk against the wrong cells.
    try:
        deposit_beam(
            200.0, 1.0e20, **_b3_kw, stopping_coefficient=_b3_coeff[:-1]
        )
    except ValueError as _b3_err:
        assert "stopping_coefficient" in str(_b3_err), _b3_err
    else:
        raise AssertionError("a short stopping_coefficient must raise")

    # --- WP-D: non-local transport of the beam's EVENT PRODUCTS
    # (product_transport). At breakdown the secondary electrons and the
    # primary's terminal sub-threshold residual are below every He inelastic
    # threshold and Coulomb-couple at ~1 eV per machine pass, so banking them
    # in their birth cell is the wrong limit; "nonlocal" walks them along B
    # and books what escapes an end to the new end ledger.

    # (a) DEFAULT OFF IS BIT-EXACT. Passing the default explicitly and
    # omitting the key must give byte-identical arrays, and the end ledger
    # must be identically zero -- nothing is booked that was not booked
    # before, which is what keeps the production golden bit-exact.
    wpd_local = deposit_beam(150.0, 1.0e22, **b1_col, product_transport="local")
    for wpd_field in (
        "ionization_events", "excitation_events", "plasma_heating_erg_s",
        "radiated_erg_s", "ionization_cost_erg_s", "E_entry_eV",
        "heating_coulomb_erg_s", "heating_anomalous_erg_s",
        "heating_secondary_erg_s", "heating_terminal_erg_s",
    ):
        assert np.array_equal(
            getattr(wpd_local, wpd_field), getattr(b1_res, wpd_field)
        ), wpd_field
    assert wpd_local.transmitted_flux == b1_res.transmitted_flux
    assert wpd_local.end_loss_low_erg_s == 0.0
    assert wpd_local.end_loss_high_erg_s == 0.0
    assert wpd_local.end_loss_transmitted_erg_s == 0.0

    # The walk integrates the module's OWN stopping power in closed form
    # rather than substepping it, which is exact only because both closures
    # are pure power laws in W (lnLambda depends on ne and Te alone). This
    # guards that identity: if coulomb_stopping_eV_per_cm ever stops being
    # A(ne,Te)*W**p, the walk silently stops matching the primary's drag.
    for wpd_model, wpd_p in _COULOMB_STOPPING_EXPONENT.items():
        wpd_A = _coulomb_stopping_coefficient([2.0e12], [4.0], wpd_model)[0]
        for wpd_W in (0.2, 3.0, 40.0, 150.0):
            wpd_ref = coulomb_stopping_eV_per_cm(
                wpd_W, 2.0e12, 4.0, model=wpd_model
            )
            assert abs(wpd_A * wpd_W**wpd_p - wpd_ref) <= 1e-12 * wpd_ref, (
                wpd_model, wpd_W
            )

    # (b) THE EXTENDED CONSERVATION IDENTITY. On a column where the walks do
    # BOTH things -- the backward halves born at the launch cell leave the low
    # end immediately, the forward ones run ~11 m and thermalize inside the
    # 20 m domain -- per-ray energy still closes to roundoff with the end
    # ledger carrying what left:
    #     Gamma0*E0 = heating + radiated + cost + anode + end_loss
    wpd_cells = 40
    wpd_col = dict(
        nn=np.full(wpd_cells, 2.0e14),
        ne=np.full(wpd_cells, 1.0e11),
        Te=np.full(wpd_cells, 1.0),
        launch=0,
        direction=1,
        dz_cm=np.full(wpd_cells, 50.0),
    )
    wpd_ref_local = deposit_beam(150.0, 1.0e22, **wpd_col)
    wpd_nl = deposit_beam(150.0, 1.0e22, **wpd_col, product_transport="nonlocal")
    wpd_budget = 1.0e22 * 150.0 * 1.602176634e-12
    wpd_total = (
        wpd_nl.plasma_heating_erg_s.sum()
        + wpd_nl.radiated_erg_s.sum()
        + wpd_nl.ionization_cost_erg_s.sum()
        + float(wpd_nl.anode_intercepted_erg_s)
        + wpd_nl.end_loss_low_erg_s
        + wpd_nl.end_loss_high_erg_s
    )
    assert abs(wpd_total - wpd_budget) / wpd_budget < 1e-12
    assert wpd_nl.end_loss_low_erg_s > 0.0  # escaped backwards
    assert wpd_nl.end_loss_high_erg_s > 0.0  # escaped forwards
    assert wpd_nl.heating_secondary_erg_s.sum() > 0.0  # and some thermalized
    assert wpd_nl.heating_terminal_erg_s.sum() > 0.0
    # This ray is absorbed, so nothing in the ledger is the transmitted
    # primary -- all of it is walked product.
    assert wpd_nl.transmitted_flux == 0.0
    assert wpd_nl.end_loss_transmitted_erg_s == 0.0
    # Energy MOVED, it was not created: the plasma keeps strictly less.
    assert (
        wpd_nl.plasma_heating_erg_s.sum()
        < wpd_ref_local.plasma_heating_erg_s.sum()
    )
    # v1 is ENERGY-ONLY routing: the particle rows (and everything downstream
    # of them -- n, the circuit currents) are identical in both modes.
    assert np.array_equal(
        wpd_nl.ionization_events, wpd_ref_local.ionization_events
    )
    assert np.array_equal(
        wpd_nl.excitation_events, wpd_ref_local.excitation_events
    )

    # (c) THE LOCAL LIMIT. Raise n_e until the product range collapses far
    # below one cell and "nonlocal" must reproduce "local": every walk
    # thermalizes in its birth cell and nothing reaches an end. The tolerance
    # is roundoff (rtol 1e-12), not a convergence tolerance -- in this limit
    # the two bookings are the same sum in a different order, so anything
    # larger would mean a real leak rather than an unconverged walk.
    wpd_dense = dict(b1_col, ne=np.full(b1_cells, 1.0e13))
    wpd_dense_local = deposit_beam(150.0, 1.0e22, **wpd_dense)
    wpd_dense_nl = deposit_beam(
        150.0, 1.0e22, **wpd_dense, product_transport="nonlocal"
    )
    assert np.allclose(
        wpd_dense_nl.plasma_heating_erg_s,
        wpd_dense_local.plasma_heating_erg_s,
        rtol=1e-12,
        atol=0.0,
    )
    assert wpd_dense_nl.end_loss_low_erg_s == 0.0
    assert wpd_dense_nl.end_loss_high_erg_s == 0.0

    # (d) THE DIRECTION SPLIT. Secondaries leave broadly isotropically, so
    # each birth cell emits two half-weight walks, +z and -z. Confine the
    # neutrals to a single cell at the exact centre of an otherwise uniform
    # column: the two halves then see identical columns and identical
    # distances to their ends, so their escapes must match and the deposited
    # secondary profile must be mirror-symmetric about the birth cell. (The
    # primary streams on through the vacuum and transmits, so the high end
    # additionally carries its Gamma_t*E_t -- the ledger's other member,
    # subtracted out here through its own diagnostic split.)
    wpd_sym_cells = 41
    wpd_sym_mid = 20
    wpd_sym_nn = np.zeros(wpd_sym_cells)
    wpd_sym_nn[wpd_sym_mid] = 5.0e14
    wpd_sym = deposit_beam(
        150.0, 1.0e21,
        nn=wpd_sym_nn,
        ne=np.full(wpd_sym_cells, 3.0e10),
        Te=np.full(wpd_sym_cells, 1.0),
        launch=wpd_sym_mid,
        direction=1,
        dz_cm=np.full(wpd_sym_cells, 40.0),
        product_transport="nonlocal",
    )
    assert wpd_sym.transmitted_flux > 0.0
    assert wpd_sym.end_loss_transmitted_erg_s > 0.0
    assert np.isclose(
        wpd_sym.end_loss_high_erg_s - wpd_sym.end_loss_transmitted_erg_s,
        wpd_sym.end_loss_low_erg_s,
        rtol=1e-12,
        atol=0.0,
    )
    wpd_sym_heat = wpd_sym.heating_secondary_erg_s
    assert np.array_equal(
        wpd_sym_heat[wpd_sym_mid + 1:], wpd_sym_heat[:wpd_sym_mid][::-1]
    )

    # (e) MISCONFIGURATION is loud at the module boundary too (the solver
    # raises at construction; see the WP-D block in the R4/csda section).
    try:
        deposit_beam(150.0, 1e22, **b1_col, product_transport="bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for product_transport")


# --------------------------------------------------------------------
# csda-ql-heating-locality
# --------------------------------------------------------------------
@_case(
    "csda-ql-heating-locality",
    provides=(
        "wpe_E0", "wpe_G0", "wpe_cells", "wpe_removed", "wpe_thin",
        "wpe_walk",
    ),
)
def _case_csda_ql_heating_locality(deposit_beam):
    # --- WP-E: QL heating locality (anomalous_transport). The anomalous
    # channel banks its drag as instantaneous LOCAL bulk heating; kinetically
    # QL fills a fast-tail plateau first, and at breakdown densities a tail
    # electron is collisionally decoupled and free-streams along B. Under
    # "tail_walk" the QL power is carried by tail electrons at E_tail on the
    # SAME closed-form Coulomb walk the WP-D products use.
    #
    # The column needs an ACTIVE anomalous channel, so the beam must be weak
    # enough for quasilinear theory to apply (n_b < n_e/10) -- b1_col's
    # 1e22 beam is not, and runs with anomalous_model="none" by default.
    wpe_cells = 60
    wpe_thin = dict(
        nn=np.full(wpe_cells, 1.0e12),
        ne=np.full(wpe_cells, 1.0e10),
        Te=np.full(wpe_cells, 2.0),
        launch=0,
        direction=1,
        dz_cm=np.full(wpe_cells, 30.0),
        anomalous_model="quasilinear",
        beam_area_cm2=100.0,
    )
    wpe_G0 = 1.0e18
    wpe_E0 = 150.0
    wpe_budget = wpe_G0 * wpe_E0 * 1.602176634e-12
    wpe_local = deposit_beam(wpe_E0, wpe_G0, **wpe_thin)
    assert wpe_local.heating_anomalous_erg_s.sum() > 0.0  # channel is live
    wpe_walk = deposit_beam(
        wpe_E0, wpe_G0, **wpe_thin,
        anomalous_transport="tail_walk", tail_energy_eV=75.0,
    )

    # (a) THE RAY IS BIT-IDENTICAL. L_anom depends on the beam and the column,
    # never on where its energy is banked, so the trajectory, the primary flux
    # and every non-anomalous channel are byte-for-byte the same. This is what
    # makes the conservation identity below exact rather than approximate.
    for _wpe_arr in (
        "E_entry_eV", "ionization_events", "excitation_events",
        "radiated_erg_s", "ionization_cost_erg_s", "heating_coulomb_erg_s",
        "heating_secondary_erg_s", "heating_terminal_erg_s",
    ):
        assert np.array_equal(
            getattr(wpe_walk, _wpe_arr), getattr(wpe_local, _wpe_arr)
        ), _wpe_arr
    assert wpe_walk.transmitted_flux == wpe_local.transmitted_flux
    assert wpe_walk.transmitted_energy_eV == wpe_local.transmitted_energy_eV

    # (b) THE CONSERVATION IDENTITY: banking removed = walked deposition +
    # end losses, to roundoff. The tolerance is roundoff (1e-12), not a
    # convergence tolerance -- the walk is closed-form and telescopes, so
    # anything larger would be a real leak.
    wpe_removed = float(wpe_local.heating_anomalous_erg_s.sum())
    wpe_ledger = (
        float(wpe_walk.end_loss_tail_low_erg_s)
        + float(wpe_walk.end_loss_tail_high_erg_s)
    )
    wpe_delivered = float(wpe_walk.heating_anomalous_erg_s.sum()) + wpe_ledger
    assert abs(wpe_delivered - wpe_removed) / wpe_removed < 1e-12, (
        wpe_removed, wpe_delivered
    )
    # ... and the whole per-ray budget closes with the tail ledger in it.
    wpe_total = (
        wpe_walk.plasma_heating_erg_s.sum()
        + wpe_walk.radiated_erg_s.sum()
        + wpe_walk.ionization_cost_erg_s.sum()
        + float(wpe_walk.anode_intercepted_erg_s)
        + wpe_walk.transmitted_flux
        * wpe_walk.transmitted_energy_eV
        * 1.602176634e-12
        + wpe_ledger
    )
    assert abs(wpe_total - wpe_budget) / wpe_budget < 1e-9

    # (c) THE THIN/HOT LIMIT: at breakdown-like n_e = 1e10 a 75 eV tail
    # electron's Coulomb range is hundreds of machine lengths, so nearly all
    # of the QL power leaves through the ends instead of heating the column.
    assert wpe_ledger / wpe_removed > 0.9
    assert (
        wpe_walk.plasma_heating_erg_s.sum()
        < wpe_local.plasma_heating_erg_s.sum()
    )

    # (d) THE LOCAL LIMIT (the D1 self-limiting pattern): raise n_e until the
    # tail range collapses below one cell and "tail_walk" must reproduce
    # "local" -- every walker thermalizes in its birth cell and nothing
    # reaches an end. The closure confines itself to the low-density phase.
    wpe_dense = dict(wpe_thin, ne=np.full(wpe_cells, 1.0e14))
    wpe_dense_local = deposit_beam(wpe_E0, wpe_G0, **wpe_dense)
    wpe_dense_walk = deposit_beam(
        wpe_E0, wpe_G0, **wpe_dense,
        anomalous_transport="tail_walk", tail_energy_eV=75.0,
    )
    assert wpe_dense_walk.end_loss_tail_low_erg_s == 0.0
    assert wpe_dense_walk.end_loss_tail_high_erg_s == 0.0
    assert np.allclose(
        wpe_dense_walk.plasma_heating_erg_s,
        wpe_dense_local.plasma_heating_erg_s,
        rtol=1e-9,
        atol=0.0,
    )

    # (e) E_tail SETS THE RANGE, NOT THE POWER. The equivalent tail flux is
    # P_QL/E_tail, so the power carried is independent of E_tail (conservation
    # holds at every bracket arm) while a hotter tail travels further and
    # exports more. This pins the one thing the bracket arms vary.
    wpe_prev_escape = -1.0
    for wpe_E_tail in (30.0, 75.0, 150.0):
        wpe_arm = deposit_beam(
            wpe_E0, wpe_G0, **wpe_thin,
            anomalous_transport="tail_walk", tail_energy_eV=wpe_E_tail,
        )
        wpe_arm_ledger = (
            float(wpe_arm.end_loss_tail_low_erg_s)
            + float(wpe_arm.end_loss_tail_high_erg_s)
        )
        wpe_arm_delivered = (
            float(wpe_arm.heating_anomalous_erg_s.sum()) + wpe_arm_ledger
        )
        assert abs(wpe_arm_delivered - wpe_removed) / wpe_removed < 1e-12
        assert wpe_arm_ledger > wpe_prev_escape
        wpe_prev_escape = wpe_arm_ledger

    # (f) THE DIRECTION SPLIT. The tails leave 50/50 along +-B, so a QL source
    # confined to the exact centre of an otherwise uniform column must produce
    # matching escapes at the two ends and a mirror-symmetric deposit.
    #
    # Confining it needs the per-cell ``beam_area_cm2`` rather than the
    # single-cell ``nn`` trick the WP-D split test uses: unlike the event
    # products, the anomalous drag is CONTINUOUS along the ray and is born in
    # every cell the primary crosses. A tiny area drives n_b above the
    # weak-beam ceiling n_e/10, where the quasilinear closure returns no drag
    # at all, so widening it in one cell selects that cell as the only source.
    wpe_sym_cells = 41
    wpe_sym_mid = 20
    wpe_sym_nn = np.zeros(wpe_sym_cells)
    wpe_sym_nn[wpe_sym_mid] = 1.0e13
    wpe_sym_area = np.full(wpe_sym_cells, 1.0e-2)
    wpe_sym_area[wpe_sym_mid] = 100.0
    wpe_sym_col = dict(
        nn=wpe_sym_nn,
        ne=np.full(wpe_sym_cells, 3.0e11),
        Te=np.full(wpe_sym_cells, 1.0),
        launch=wpe_sym_mid,
        direction=1,
        dz_cm=np.full(wpe_sym_cells, 40.0),
        anomalous_model="quasilinear",
        beam_area_cm2=wpe_sym_area,
    )
    wpe_sym_local = deposit_beam(wpe_E0, wpe_G0, **wpe_sym_col)
    assert np.array_equal(
        np.flatnonzero(wpe_sym_local.heating_anomalous_erg_s),
        np.array([wpe_sym_mid]),
    )
    wpe_sym = deposit_beam(
        wpe_E0, wpe_G0, **wpe_sym_col,
        anomalous_transport="tail_walk", tail_energy_eV=75.0,
    )
    assert wpe_sym.end_loss_tail_low_erg_s > 0.0
    # The two halves see identical columns and identical distances to their
    # ends, so this is an EQUALITY, not a tolerance.
    assert (
        wpe_sym.end_loss_tail_high_erg_s == wpe_sym.end_loss_tail_low_erg_s
    )
    wpe_sym_heat = wpe_sym.heating_anomalous_erg_s
    assert np.array_equal(
        wpe_sym_heat[wpe_sym_mid + 1:], wpe_sym_heat[:wpe_sym_mid][::-1]
    )
    assert wpe_sym.end_loss_low_erg_s == 0.0  # WP-D ledger untouched
    assert wpe_sym.end_loss_high_erg_s == 0.0

    # (g) MISCONFIGURATION is loud at the module boundary too (the solver
    # raises at construction; see the WP-E block in the R4/csda section).
    for wpe_bad_call in (
        lambda: deposit_beam(
            wpe_E0, wpe_G0, **wpe_thin, anomalous_transport="bogus"
        ),
        # tail_walk with no tail energy to launch at
        lambda: deposit_beam(
            wpe_E0, wpe_G0, **wpe_thin, anomalous_transport="tail_walk"
        ),
        lambda: deposit_beam(
            wpe_E0, wpe_G0, **wpe_thin,
            anomalous_transport="tail_walk", tail_energy_eV=0.0,
        ),
        lambda: deposit_beam(
            wpe_E0, wpe_G0, **wpe_thin,
            anomalous_transport="tail_walk", tail_energy_eV=float("inf"),
        ),
        # tail_walk with no anomalous channel to carry: a silent no-op
        lambda: deposit_beam(
            wpe_E0, wpe_G0, **dict(wpe_thin, anomalous_model="none"),
            anomalous_transport="tail_walk", tail_energy_eV=75.0,
        ),
    ):
        try:
            wpe_bad_call()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for anomalous_transport")
    return locals()


# --------------------------------------------------------------------
# csda-walk-window-reflection-k7
# --------------------------------------------------------------------
@_case("csda-walk-window-reflection-k7")
def _case_csda_walk_window_reflection_k7(
    cooling_kwargs, deposit_beam, knob_floors, knob_mass,
    knob_state, shape_state, wpe_E0, wpe_G0, wpe_cells, wpe_removed,
    wpe_thin, wpe_walk
):
    # --- K7 at the module: one walk-window face REFLECTS instead of letting
    # walkers leave. The comparison against the threshold is general, so the
    # arm is not a disguised "reflect everything" switch.
    k7m_win = (0, wpe_cells - 1)
    k7m_common = dict(
        wpe_thin, anomalous_transport="tail_walk", tail_energy_eV=75.0,
        tail_walk_window=k7m_win, tail_reflect_face=-1,
    )
    # (a) THRESHOLD BELOW EVERY ARRIVAL ENERGY: nothing reflects, and with the
    # window spanning the whole grid the result is the unreflected walk BYTE
    # FOR BYTE. This is the general-comparison statement and the
    # bit-exactness statement in one.
    k7m_inert = deposit_beam(
        wpe_E0, wpe_G0, **k7m_common, tail_reflect_threshold_eV=1.0e-30,
    )
    assert np.array_equal(
        k7m_inert.heating_anomalous_erg_s, wpe_walk.heating_anomalous_erg_s
    )
    assert (
        k7m_inert.end_loss_tail_low_erg_s == wpe_walk.end_loss_tail_low_erg_s
    )
    assert (
        k7m_inert.end_loss_tail_high_erg_s == wpe_walk.end_loss_tail_high_erg_s
    )
    # (b) THRESHOLD ABOVE THEM: everything reflects. The named face's ledger is
    # EXACTLY zero, the conservation identity still closes to roundoff, and the
    # column keeps what the face used to delete.
    for k7m_face in (-1, 1):
        k7m_refl = deposit_beam(
            wpe_E0, wpe_G0, **dict(k7m_common, tail_reflect_face=k7m_face),
            tail_reflect_threshold_eV=1.0e4,
        )
        k7m_ledger = (
            float(k7m_refl.end_loss_tail_low_erg_s)
            + float(k7m_refl.end_loss_tail_high_erg_s)
        )
        k7m_face_ledger = (
            k7m_refl.end_loss_tail_low_erg_s if k7m_face < 0
            else k7m_refl.end_loss_tail_high_erg_s
        )
        assert k7m_face_ledger == 0.0, k7m_face
        k7m_delivered = (
            float(k7m_refl.heating_anomalous_erg_s.sum()) + k7m_ledger
        )
        assert abs(k7m_delivered - wpe_removed) / wpe_removed < 1e-12, (
            k7m_face, wpe_removed, k7m_delivered
        )
        assert (
            float(k7m_refl.heating_anomalous_erg_s.sum())
            > float(wpe_walk.heating_anomalous_erg_s.sum())
        )
        # Energy-only still: reflection moves energy, never particles.
        assert np.array_equal(
            k7m_refl.ionization_events, wpe_walk.ionization_events
        )
    # (c) A SUB-WINDOW IS A WALL. With the window closed short of the grid, no
    # tail energy lands beyond it in either direction, and the budget still
    # closes -- what leaves through the far face is booked, not lost.
    k7m_sub = deposit_beam(
        wpe_E0, wpe_G0, **dict(k7m_common, tail_walk_window=(0, 40)),
        tail_reflect_threshold_eV=1.0e4,
    )
    assert not np.any(k7m_sub.heating_anomalous_erg_s[41:])
    k7m_sub_delivered = (
        float(k7m_sub.heating_anomalous_erg_s.sum())
        + float(k7m_sub.end_loss_tail_low_erg_s)
        + float(k7m_sub.end_loss_tail_high_erg_s)
    )
    assert abs(k7m_sub_delivered - wpe_removed) / wpe_removed < 1e-12
    # (d) MISCONFIGURATION at the module boundary: a face with no threshold, a
    # threshold with no face, a face that is not a face, a threshold that is
    # not an energy, a face with no window to put it on, and reflection asked
    # for where there is no walk at all.
    for k7m_bad in (
        dict(tail_reflect_face=-1, tail_walk_window=k7m_win),
        dict(tail_reflect_threshold_eV=100.0, tail_walk_window=k7m_win),
        dict(tail_reflect_face=0, tail_reflect_threshold_eV=100.0,
             tail_walk_window=k7m_win),
        dict(tail_reflect_face=-1, tail_reflect_threshold_eV=0.0,
             tail_walk_window=k7m_win),
        dict(tail_reflect_face=-1, tail_reflect_threshold_eV=float("nan"),
             tail_walk_window=k7m_win),
        dict(tail_reflect_face=-1, tail_reflect_threshold_eV=float("inf"),
             tail_walk_window=k7m_win),
        dict(tail_reflect_face=-1, tail_reflect_threshold_eV=100.0),
        dict(tail_reflect_face=-1, tail_reflect_threshold_eV=100.0,
             tail_walk_window=(0, wpe_cells)),
    ):
        try:
            deposit_beam(
                wpe_E0, wpe_G0, **wpe_thin,
                anomalous_transport="tail_walk", tail_energy_eV=75.0,
                **k7m_bad,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {k7m_bad!r}")
    try:
        deposit_beam(
            wpe_E0, wpe_G0, **wpe_thin,
            tail_reflect_face=-1, tail_reflect_threshold_eV=100.0,
            tail_walk_window=k7m_win,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "expected ValueError for reflection without a tail walk"
        )

    S_ion_a, S_rad_a = reaction_rates(
        state=knob_state,
        floors=knob_floors,
        ion_mass_g=knob_mass,
    )
    for values in (S_ion_a, S_rad_a):
        assert np.all(np.isfinite(values)) and np.all(values >= 0.0)

    cool_adas = electron_cooling_rhs(**cooling_kwargs)
    assert np.all(np.isfinite(cool_adas.Ee))
    assert np.all(cool_adas.Ee <= 0.0)
    # The cooling path's fused ionization cost must be bit-identical to
    # I_ion * S_ion from reaction_rates -- the cost charges exactly the
    # particles the particle equation creates.
    cost_kwargs = dict(cooling_kwargs)
    cost_kwargs["ionization_energy_cost"] = True
    cost_terms = electron_cooling_rhs_terms(**cost_kwargs)
    S_ion_ref, _ = reaction_rates(
        state=shape_state,
        floors=knob_floors,
        ion_mass_g=knob_mass,
    )
    assert np.allclose(
        cost_terms["ionization_energy_cost"].Ee,
        -24.587 * ev_to_erg * S_ion_ref,
        rtol=1e-13,
        atol=0.0,
    )


# --------------------------------------------------------------------
# ql-relaxation-onset-gate
# --------------------------------------------------------------------
@_case(
    "ql-relaxation-onset-gate",
    provides=("_qlr_ray",),
)
def _case_ql_relaxation_onset_gate():
    # ================= ql_relaxation: the anomalous middle leg =============
    # Pre-registered with the closure. Four things have to hold and each has an
    # anti-vacuity twin: the boxed onset gate actually gates (and is open where
    # the memo says it is), the bracket constant is load-bearing when selected
    # and INERT when it is not, the compiled kernel never runs this closure,
    # and a passive cell BOOKS its power (the option-3 refusal is keyed to the
    # fiat arm alone).
    _qlr_cells = 20
    _qlr_dz = np.full(_qlr_cells, 10.0)
    _qlr_ray = dict(
        nn=np.full(_qlr_cells, 2.0e13),
        ne=np.full(_qlr_cells, 1.0e10),
        Te=np.full(_qlr_cells, 2.0),
        launch=0,
        direction=1,
        dz_cm=_qlr_dz,
        beam_area_cm2=100.0,
    )

    # ---- (a) the boxed onset inequality, and that it is a GATE ----
    # Memo statement, recomputed rather than quoted: over the working range the
    # linear onset is always open, by a wide margin. This is what makes onset
    # NOT the gating physics and relaxation the currency.
    _qlr_nb = 1.0e19 / (
        100.0 * _beam_deposition_mod.beam_speed_cm_s(150.0)
    )
    for _qlr_ne in (1.0e8, 1.0e9, 1.0e10, 1.0e11):
        assert _beam_deposition_mod.ql_onset_open(
            _qlr_ne, 2.0e13, 2.0, _qlr_nb
        ), (
            "the boxed QL onset must be open across the working range "
            f"(closed at n_e = {_qlr_ne:g})"
        )
    # ANTI-VACUITY: the gate CAN close, and closes for the right reason. Two
    # ways, one per conjunct -- an absurd neutral density kills growth against
    # damping, and a vacuum-class plasma leaves no wave to damp.
    assert not _beam_deposition_mod.ql_onset_open(1.0e10, 1.0e22, 2.0, _qlr_nb)
    assert not _beam_deposition_mod.ql_onset_open(1.0e2, 2.0e13, 2.0, _qlr_nb)
    # ... and a closed gate books EXACTLY zero, not merely little.
    _qlr_shut = _beam_deposition_mod.deposit_beam(
        150.0,
        1.0e19,
        **dict(_qlr_ray, nn=np.full(_qlr_cells, 1.0e22)),
        anomalous_model="ql_relaxation",
        ql_relaxation_coeff=30.0,
    )
    assert not np.any(_qlr_shut.heating_anomalous_erg_s), (
        "with the onset gate closed the ql_relaxation channel must book "
        "identically zero"
    )
    assert (
        _beam_deposition_mod.ql_relaxation_stopping_eV_per_cm(
            150.0, 1.0e10, 1.0e22, 2.0, _qlr_nb, 30.0
        )
        == 0.0
    )
    return locals()


# --------------------------------------------------------------------
# ql-relaxation-books-and-conserves
# --------------------------------------------------------------------
@_case("ql-relaxation-books-and-conserves")
def _case_ql_relaxation_books_and_conserves(_qlr_ray):
    # ---- (b) the closure books, conserves, and the bracket moves it ----
    _qlr_by_coeff = {}
    for _qlr_c in (10.0, 30.0, 100.0):
        _qlr_by_coeff[_qlr_c] = _beam_deposition_mod.deposit_beam(
            150.0,
            1.0e19,
            **_qlr_ray,
            anomalous_model="ql_relaxation",
            ql_relaxation_coeff=_qlr_c,
        )
    _qlr_anom = {
        c: float(r.heating_anomalous_erg_s.sum())
        for c, r in _qlr_by_coeff.items()
    }
    assert _qlr_anom[10.0] > _qlr_anom[30.0] > _qlr_anom[100.0] > 0.0, (
        "the plateau-formation bracket must be load-bearing and monotone "
        f"(a longer relaxation length is weaker drag): {_qlr_anom}"
    )
    # Energy: extracted + retained-and-carried-out = launched, to roundoff. The
    # ledger is the module's own per-ray identity, which the new channel joins
    # rather than sits beside.
    _qlr_ref = _qlr_by_coeff[30.0]
    _qlr_launched = 1.0e19 * 150.0 * ev_to_erg
    _qlr_booked = (
        float(_qlr_ref.plasma_heating_erg_s.sum())
        + float(_qlr_ref.radiated_erg_s.sum())
        + float(_qlr_ref.ionization_cost_erg_s.sum())
        + _qlr_ref.transmitted_flux * _qlr_ref.transmitted_energy_eV * ev_to_erg
    )
    assert abs(_qlr_booked / _qlr_launched - 1.0) < 1e-12, (
        f"ql_relaxation broke per-ray energy conservation: {_qlr_booked} vs "
        f"{_qlr_launched}"
    )
    # The extracted power lands in the anomalous bank and therefore in the
    # LUMPED plasma-heating bank the RHS consumes -- bulk electrons, where the
    # waves damp -- not in a separate ledger.
    assert float(_qlr_ref.heating_anomalous_erg_s.sum()) > 0.0
    assert float(_qlr_ref.plasma_heating_erg_s.sum()) > float(
        _qlr_ref.heating_anomalous_erg_s.sum()
    )
    # The middle leg is a MIDDLE leg at this state: strictly between refusing
    # the channel and the fiat closure's near-total absorption.
    _qlr_none = _beam_deposition_mod.deposit_beam(
        150.0, 1.0e19, **_qlr_ray, anomalous_model="none"
    )
    _qlr_fiat = _beam_deposition_mod.deposit_beam(
        150.0, 1.0e19, **_qlr_ray, anomalous_model="quasilinear"
    )
    assert (
        float(_qlr_none.heating_anomalous_erg_s.sum())
        < _qlr_anom[30.0]
        < float(_qlr_fiat.heating_anomalous_erg_s.sum())
    )


# --------------------------------------------------------------------
# ql-relaxation-module-refusals
# --------------------------------------------------------------------
@_case("ql-relaxation-module-refusals")
def _case_ql_relaxation_module_refusals(_qlr_ray):
    # ---- (c) the module's own refusals ----
    for _qlr_bad in (None,):
        try:
            _beam_deposition_mod.deposit_beam(
                150.0, 1.0e19, **_qlr_ray, anomalous_model="ql_relaxation",
                ql_relaxation_coeff=_qlr_bad,
            )
        except ValueError as _qlr_error:
            assert (
                "anomalous_model='ql_relaxation' needs ql_relaxation_coeff"
                in str(_qlr_error)
            ), str(_qlr_error)
        else:
            raise AssertionError(
                "ql_relaxation must refuse an unregistered bracket arm"
            )
    for _qlr_bad in (0.0, -1.0, float("nan"), float("inf")):
        try:
            _beam_deposition_mod.deposit_beam(
                150.0, 1.0e19, **_qlr_ray, anomalous_model="ql_relaxation",
                ql_relaxation_coeff=_qlr_bad,
            )
        except ValueError as _qlr_error:
            assert (
                "ql_relaxation_coeff must be finite and > 0" in str(_qlr_error)
            ), (_qlr_bad, str(_qlr_error))
        else:
            raise AssertionError(
                f"ql_relaxation_coeff={_qlr_bad} must raise"
            )


# --------------------------------------------------------------------
# ql-relaxation-compiled-kernel-refusal
# --------------------------------------------------------------------
@_case("ql-relaxation-compiled-kernel-refusal")
def _case_ql_relaxation_compiled_kernel_refusal(_qlr_ray):
    # ---- (d) the compiled kernel must NEVER run this closure ----
    # It takes the anomalous channel as a BOOLEAN and applies the fiat drag, so
    # offering it ql_relaxation would silently run the wrong physics. Tested by
    # binding a march that explodes if reached, which works on a pure checkout
    # too -- the point is the precondition, not the extension.
    class _QlrKernelReached(RuntimeError):
        pass

    class _QlrFakeTables:
        # Only ``exc_top`` is read before the march is called.
        exc_top = 1.0e9

    def _qlr_boom(*_args, **_kwargs):
        raise _QlrKernelReached("the compiled march was offered this ray")

    _qlr_saved = (
        _beam_deposition_mod._CSDA_MARCH, _beam_deposition_mod._csda_tables
    )
    try:
        _beam_deposition_mod._CSDA_MARCH = _qlr_boom
        _beam_deposition_mod._csda_tables = lambda: _QlrFakeTables()
        _beam_deposition_mod.deposit_beam(
            150.0, 1.0e19, **_qlr_ray, anomalous_model="ql_relaxation",
            ql_relaxation_coeff=30.0,
        )
        # ANTI-VACUITY: the same harness DOES reach the kernel for the two
        # closures the transcription reproduces, so the pass above is the
        # precondition doing work and not the fake march being unreachable.
        for _qlr_ok in ("none", "quasilinear"):
            try:
                _beam_deposition_mod.deposit_beam(
                    150.0, 1.0e19, **_qlr_ray, anomalous_model=_qlr_ok,
                )
            except _QlrKernelReached:
                pass
            else:
                raise AssertionError(
                    f"the compiled-march precondition test is vacuous for "
                    f"{_qlr_ok!r}: the kernel was never offered the ray"
                )
    finally:
        (
            _beam_deposition_mod._CSDA_MARCH,
            _beam_deposition_mod._csda_tables,
        ) = _qlr_saved


# --------------------------------------------------------------------
# ql-relaxation-presence-gating
# --------------------------------------------------------------------
@_case("ql-relaxation-presence-gating")
def _case_ql_relaxation_presence_gating():
    # ---- (e) PRESENCE GATING: byte-identity with ql_relaxation unselected ---
    # The key must not reach deposit_beam, and sweeping it must not move a
    # single bit of a run on either of the other two arms.
    def _qlr_unselected_bytes(model, coeff):
        params, flags = _qlr_config()
        params["beam_anomalous_model"] = model
        params["ql_relaxation_coeff"] = coeff
        seen = []
        # Wrapped in the CATHODE module's namespace, not the defining one:
        # cathode.py binds ``deposit_beam`` by name at import, so the
        # top-level deposition rays -- the ones that carry the closure's
        # keywords -- resolve there and nowhere else.
        _real = _cathode_mod.deposit_beam

        def _watch(*args, **kwargs):
            seen.append("ql_relaxation_coeff" in kwargs)
            return _real(*args, **kwargs)

        _cathode_mod.deposit_beam = _watch
        try:
            out = _tracking_electrode_sample(
                LAPDSim1D(params, flags)
            ).run(t_end=1.0e-6, dt=1.0e-7)
        finally:
            _cathode_mod.deposit_beam = _real
        return np.asarray(out.n, dtype=float).tobytes(), seen

    for _qlr_model in ("none", "quasilinear"):
        _qlr_ref_bytes, _qlr_seen = _qlr_unselected_bytes(_qlr_model, 30.0)
        assert _qlr_seen and not any(_qlr_seen), (
            f"beam_anomalous_model={_qlr_model!r} must not carry "
            "ql_relaxation_coeff into deposit_beam"
        )
        for _qlr_sweep in (10.0, 100.0, 1.0e4):
            _qlr_moved, _ = _qlr_unselected_bytes(_qlr_model, _qlr_sweep)
            assert _qlr_moved == _qlr_ref_bytes, (
                f"ql_relaxation_coeff moved a {_qlr_model!r} run "
                f"(swept to {_qlr_sweep})"
            )
    # ANTI-VACUITY: with the closure SELECTED the identical sweep must move the
    # run, and the key must reach the module.
    _qlr_sel_ref, _qlr_sel_seen = _qlr_unselected_bytes("ql_relaxation", 30.0)
    assert _qlr_sel_seen and all(_qlr_sel_seen), (
        "ql_relaxation must carry its bracket constant into deposit_beam"
    )
    _qlr_sel_swept, _ = _qlr_unselected_bytes("ql_relaxation", 100.0)
    assert _qlr_sel_swept != _qlr_sel_ref, (
        "the presence-gating test is vacuous: the bracket constant does not "
        "move a run even when its closure is selected"
    )


# --------------------------------------------------------------------
# ql-relaxation-solver-refusals
# --------------------------------------------------------------------
@_case("ql-relaxation-solver-refusals")
def _case_ql_relaxation_solver_refusals():
    # ---- (f) construction-time refusals, at the SOLVER ----
    def _qlr_refuses(says, **overrides):
        params, flags = _qlr_config()
        params["beam_anomalous_model"] = "ql_relaxation"
        params.update(overrides)
        try:
            LAPDSim1D(params, flags)
        except ValueError as error:
            assert says in str(error), (overrides, str(error))
            return
        raise AssertionError(f"LAPDSim1D must refuse {overrides}")

    _qlr_refuses(
        "beam_anomalous_model='ql_relaxation' requires ql_relaxation_coeff",
        ql_relaxation_coeff=None,
    )
    _qlr_bad_coeff = "ql_relaxation_coeff must be finite and > 0"
    _qlr_refuses(_qlr_bad_coeff, ql_relaxation_coeff=0.0)
    _qlr_refuses(_qlr_bad_coeff, ql_relaxation_coeff=-30.0)
    _qlr_refuses(_qlr_bad_coeff, ql_relaxation_coeff=float("nan"))
    _qlr_ok_params, _qlr_ok_flags = _qlr_config()
    _qlr_ok_params["beam_anomalous_model"] = "ql_relaxation"
    LAPDSim1D(_qlr_ok_params, _qlr_ok_flags)
    # The selector domain is closed.
    _qlr_zzz_params, _qlr_zzz_flags = _qlr_config()
    _qlr_zzz_params["beam_anomalous_model"] = "zzz"
    try:
        LAPDSim1D(_qlr_zzz_params, _qlr_zzz_flags)
    except ValueError as _qlr_zzz_error:
        assert "beam_anomalous_model must be one of" in str(_qlr_zzz_error), (
            str(_qlr_zzz_error)
        )
    else:
        raise AssertionError("beam_anomalous_model must reject 'zzz'")


# --------------------------------------------------------------------
# ql-relaxation-km-table
# --------------------------------------------------------------------
@_case("ql-relaxation-km-table")
def _case_ql_relaxation_km_table():
    # ---- (h) the K_m table is the boxed table ----
    # Two nodes. The numbers are DERIVED from the three published LXCat He
    # elastic momentum-transfer sets (Biagi / IST-Lisbon / Morgan, retrieved
    # 2026-08-13): each node is the three-set arithmetic centre and each
    # bracket is [min, max] over the three. Rotated 2026-08-30 from the memo's
    # earlier pair by [km-node-boxing-decision] -- nodes (6.0, 2.1)e-16 ->
    # (6.280, 1.992)e-16, the 25 eV bracket (1.6, 2.6)e-16 ->
    # (1.950, 2.067)e-16. These literals pin the SHIPPED table, so they rotate
    # with it and only with it. Interpolation is log-log
    # inside the span and CLAMPED outside it, so no structure is manufactured
    # where the table has none.
    # The nodes round-trip through the log-log interpolation, so they are
    # compared to the published constants relatively (1 ULP of an exp/log pair,
    # not a tolerance on the physics); the CLAMPS are exact, being the function
    # compared against itself.
    for _qlr_node, _qlr_sigma in zip(
        _cross_mod.HE_EN_MT_NODE_EV, _cross_mod.HE_EN_MT_SIGMA_CM2
    ):
        assert abs(
            _cross_mod.he_electron_momentum_transfer_cm2(_qlr_node)
            / _qlr_sigma
            - 1.0
        ) < 1e-12, _qlr_node
    assert _cross_mod.HE_EN_MT_SIGMA_CM2 == (6.280e-16, 1.992e-16)
    assert _cross_mod.HE_EN_MT_NODE_EV == (5.0, 25.0)
    assert _cross_mod.he_electron_momentum_transfer_cm2(
        1.0
    ) == _cross_mod.he_electron_momentum_transfer_cm2(5.0)
    assert _cross_mod.he_electron_momentum_transfer_cm2(
        500.0
    ) == _cross_mod.he_electron_momentum_transfer_cm2(25.0)
    assert (
        1.992e-16
        < _cross_mod.he_electron_momentum_transfer_cm2(11.18)
        < 6.280e-16
    )
    # The 25 eV row is carried AS a bracket, and the shipped value sits in it.
    _qlr_lo, _qlr_hi = _cross_mod.HE_EN_MT_SIGMA_BRACKET_CM2[1]
    assert _qlr_lo == 1.950e-16 and _qlr_hi == 2.067e-16
    assert _qlr_lo < _cross_mod.HE_EN_MT_SIGMA_CM2[1] < _qlr_hi
    _qlr_lo5, _qlr_hi5 = _cross_mod.HE_EN_MT_SIGMA_BRACKET_CM2[0]
    assert _qlr_lo5 < _cross_mod.HE_EN_MT_SIGMA_CM2[0] < _qlr_hi5
    # nu_en = nn * K_m is a rate: positive, finite, and rising with the fill.
    assert (
        0.0
        < _cross_mod.he_electron_momentum_transfer_rate_cm3_s(2.0)
        < 1.0e-6
    )
    assert _cross_mod.he_electron_momentum_transfer_rate_cm3_s(
        8.0
    ) > _cross_mod.he_electron_momentum_transfer_rate_cm3_s(0.5)


# --------------------------------------------------------------------
# beam-tail-retired-keys-refuse
# --------------------------------------------------------------------
@_case("beam-tail-retired-keys-refuse")
def _case_beam_tail_retired_keys_refuse():
    # The beam-deposition and hot-tail selectors, flags and parameters removed
    # with the closures they served. Each is gone from its template, is on the
    # retired register of ITS OWN namespace, and a configuration naming it --
    # at any value, the old default included -- is refused at construction
    # with the key named as RETIRED. A retired name filed in the OTHER
    # namespace reads as the plain unknown key it is there.
    from cablp.solvers._sim1d.core.config import (
        RETIRED_FLAG_KEYS,
        RETIRED_PARAM_KEYS,
        input_dict_template_1d,
        input_flags_template_1d,
    )

    _rb_params = {
        "beam_deposition_model": "csda",
        "beam_coulomb_model": "fast_electron",
        "beam_excitation_model": "2p_scalar",
        "beam_product_transport": "local",
        "heating_anomalous_disposal": "local",
        "heating_anomalous_tail_energy_keying": "phi_c",
        "heating_anomalous_tail_energy_eV": 75.0,
        "heating_anomalous_tail_phi_c_fraction": None,
        "heating_anomalous_tail_ionization": "on",
        "ionization_birth_energy_model": "conservative",
        "Te_birth_ionization": "local",
        "b_beam_excitation": 1.4,
        "beam_excitation_energy_eV": 21.218,
    }
    _rb_flags = {
        "beam_anode_interception": True,
        "beam_deposition_in_heat_substep": True,
        "beam_ionization_birth_timestep_bound": False,
    }
    _rb_base_p, _rb_base_f = default_config()
    for _rb_key, _rb_value in _rb_params.items():
        assert _rb_key not in input_dict_template_1d, _rb_key
        assert _rb_key not in input_flags_template_1d, _rb_key
        assert _rb_key in RETIRED_PARAM_KEYS, _rb_key
        try:
            LAPDSim1D(dict(_rb_base_p, **{_rb_key: _rb_value}), _rb_base_f)
        except ValueError as _rb_exc:
            assert f"{_rb_key} is RETIRED" in str(_rb_exc), str(_rb_exc)
        else:
            raise AssertionError(f"retired params key {_rb_key} ACCEPTED")
    for _rb_key, _rb_value in _rb_flags.items():
        assert _rb_key not in input_dict_template_1d, _rb_key
        assert _rb_key not in input_flags_template_1d, _rb_key
        assert _rb_key in RETIRED_FLAG_KEYS, _rb_key
        try:
            LAPDSim1D(_rb_base_p, dict(_rb_base_f, **{_rb_key: _rb_value}))
        except ValueError as _rb_exc:
            assert f"{_rb_key} is RETIRED" in str(_rb_exc), str(_rb_exc)
        else:
            raise AssertionError(f"retired flags key {_rb_key} ACCEPTED")
    # A retired FLAG name in params is a misfiled key, not a retired one.
    try:
        LAPDSim1D(dict(_rb_base_p, beam_anode_interception=True), _rb_base_f)
    except ValueError as _rb_exc:
        assert "unknown LAPDSim1D configuration keys" in str(_rb_exc)
        assert "RETIRED" not in str(_rb_exc), str(_rb_exc)
    else:
        raise AssertionError("a misfiled retired flag name was ACCEPTED")
    # The removed selector VALUES of the surviving selectors are refused.
    for _rb_key, _rb_value in (
        ("heating_anomalous_transport", "tail_walk"),
        ("Ti_birth_ionization", "floor"),
        ("Ti_birth_ionization", "local"),
    ):
        try:
            LAPDSim1D(dict(_rb_base_p, **{_rb_key: _rb_value}), _rb_base_f)
        except ValueError as _rb_exc:
            assert _rb_key in str(_rb_exc), str(_rb_exc)
        else:
            raise AssertionError(f"{_rb_key}={_rb_value!r} ACCEPTED")


# --------------------------------------------------------------------
# beam-l-b-profile-at-fed-back-cross
# --------------------------------------------------------------------
@_case("beam-l-b-profile-at-fed-back-cross", historical_stance=True)
def _case_beam_l_b_profile_at_fed_back_cross(
    csda_sim, csda_solve, csda_launch, csda_sigma_eff
):
    # The saved l_b_profile is the primary beam's per-cell mean free path at
    # the attenuation cross section the solve FEEDS BACK -- the launch cell's
    # beam_atten_cross after the CSDA gap-transmission inversion overwrote
    # it -- not at the ionization cross section the beam assembly started
    # from. Same function, same inputs, so the comparison is exact.
    from cablp.cathode.circuit_common import compute_l_b
    from cablp.solvers._sim1d.core.state import derive_state

    _lp_beam = csda_solve.beam_result
    assert float(_lp_beam.beam_atten_cross[csda_launch]) == csda_sigma_eff
    _lp_state = csda_sim._smoothed_sample_state(csda_sim.state)
    _lp_derived = derive_state(
        _lp_state, floors=csda_sim.floors, ion_mass_g=csda_sim.ion_mass_g
    )
    _lp_phi_c = _lp_beam.result.phi_c

    def _lp_profile(sigma):
        return np.array([
            compute_l_b(
                _lp_phi_c, _lp_derived.Te[j], _lp_state.n[j],
                _lp_state.nn[j], sigma,
            )
            for j in range(csda_sim.geometry.cells)
        ])

    _lp_expected = _lp_profile(_lp_beam.beam_atten_cross[csda_launch])
    assert np.all(_lp_expected > 0.0)
    assert _lp_beam.l_b_profile.tobytes() == _lp_expected.tobytes()
    # NEGATIVE CONTROL: at this state the inversion moved the cross section,
    # so the profile at the pre-overwrite (ionization) cross section is a
    # different array -- the check above can fail.
    _lp_pre = _lp_profile(_lp_beam.beam_cross[csda_launch])
    assert _lp_beam.beam_cross[csda_launch] != csda_sigma_eff
    assert _lp_pre.tobytes() != _lp_expected.tobytes()


# ====================================================================
# The mirror far end: the CSDA primary, the tail walkers, the smoothing
# ====================================================================
_MIRROR_ETA = 0.358  # the anode mesh solid fraction the mirror cases cull at


def _mirror_column(cells):
    """(nn, ne, Te, dz) of a synthetic non-uniform half column."""
    return (
        np.full(cells, 1.0e11),
        np.full(cells, 1.0e10) * np.linspace(1.0, 2.0, cells),
        np.linspace(3.0, 2.0, cells),
        np.linspace(8.0, 12.0, cells),
    )


def _mirror_full(*arrays):
    """The half column's arrays reflected about the mirror plane."""
    return tuple(np.concatenate([a, a[::-1]]) for a in arrays)


def _mirror_fold(values, cells):
    """The full column's per-cell values folded onto the half column."""
    values = np.asarray(values, dtype=float)
    return values[:cells] + values[cells:][::-1]


def _mirror_fold_worst(half, full, cells):
    """Worst per-cell relative difference of a half bank and a folded one.

    Cells whose folded value is zero must be exactly zero in the half bank
    too; the rest are compared relative to the folded value.
    """
    fold = _mirror_fold(full, cells)
    half = np.asarray(half, dtype=float)
    zero = fold == 0.0
    if np.any(half[zero] != 0.0):
        return math.inf
    if not np.any(~zero):
        return 0.0
    return float(
        np.max(np.abs(half[~zero] - fold[~zero]) / np.abs(fold[~zero]))
    )


_MIRROR_BANKS = (
    "plasma_heating_erg_s",
    "ionization_events",
    "excitation_events",
    "radiated_erg_s",
    "ionization_cost_erg_s",
    "heating_anomalous_erg_s",
)


def _mirror_mg_kwargs(cells, **extra):
    """The walked plateau tail on the synthetic column, ionizing."""
    kwargs = dict(
        anomalous_model="quasilinear",
        beam_area_cm2=1000.0,
        anomalous_transport="plateau_multigroup",
        plateau_edge_eV=30.0,
        tail_ionization="on",
        tail_walk_window=(0, cells - 1),
    )
    kwargs.update(extra)
    return kwargs


def _mirror_march_kwargs():
    """The march configuration the tail legs run in."""
    return dict(
        I_ion_eV=I_ion,
        E_stop_eV=_beam_deposition_mod.HE_E_STOP_EV,
        coulomb_model="fast_electron",
        anomalous_model="none",
        max_energy_fraction_per_substep=0.02,
    )


def _mirror_residual_bound_lifted():
    """Lift the run-time mirror residual bound for a synthetic ray.

    The mirror cases below that exercise a configuration where nothing
    removes the walkers (a reflecting cathode and no absorbing anode) do so
    to test a per-face rule or the capped branch itself, which
    ``MIRROR_RESIDUAL_MAX_FRACTION`` refuses on a solver-facing call; the
    bound tests the result and changes no float the march produces.
    """
    from unittest import mock

    return mock.patch.object(
        _beam_deposition_mod, "MIRROR_RESIDUAL_MAX_FRACTION", math.inf
    )


# --------------------------------------------------------------------
# mirror-tail-per-face-rule
# --------------------------------------------------------------------
@_case("mirror-tail-per-face-rule")
def _case_mirror_tail_per_face_rule():
    """Each walk-window face has ONE rule under a mirror far end.

    The cathode face (``tail_reflect_face``) turns back a walker below its
    threshold and lets one at or above it escape; the mirror face
    (``mirror_face``) turns every walker round, so its end-loss row is exactly
    0.0 while walker power arrives there. Naming one face for both, a mirror
    with no walk window, a window stopping short of the grid end, the
    energy-only walk, a walking ``product_transport`` and an unknown face are
    each refused. NEGATIVE CONTROL: the same call without ``mirror_face`` lets
    the walkers out through the far face, so "the mirror row is 0.0" fails on
    it.
    """
    cells = 24
    nn, ne, Te, dz = _mirror_column(cells)
    args = (200.0, 1.0e20, nn, ne, Te, 0, 1, dz)
    reflect = dict(tail_reflect_face=-1, tail_reflect_threshold_eV=200.0)
    # Nothing removes these walkers (both faces turn them), so the residual
    # bound is lifted for this one call; mirror-residual-bound-raises owns it.
    with _mirror_residual_bound_lifted():
        turned = _deposit_beam_ray(
            *args, **_mirror_mg_kwargs(cells, mirror_face=1, **reflect)
        )
    assert turned.tail_power_erg_s > 0.0
    assert turned.tail_mirror_erg_s > 0.0
    assert turned.tail_mirror_flux_per_s > 0.0
    assert turned.end_loss_tail_high_erg_s == 0.0
    # Every walker is born below e*phi_c = 200 eV and only loses energy, so
    # the cathode face turns all of them back as well.
    assert turned.end_loss_tail_low_erg_s == 0.0
    # A cathode threshold below the walkers: the cathode face lets them out,
    # the mirror face still turns every one round.
    leaky = _deposit_beam_ray(
        *args,
        **_mirror_mg_kwargs(
            cells, mirror_face=1, tail_reflect_face=-1,
            tail_reflect_threshold_eV=1.0,
        ),
    )
    assert leaky.end_loss_tail_low_erg_s > 0.0
    assert leaky.end_loss_tail_high_erg_s == 0.0
    # NEGATIVE CONTROL: no mirror, and the far face is an exit.
    plain = _deposit_beam_ray(*args, **_mirror_mg_kwargs(cells, **reflect))
    assert plain.end_loss_tail_high_erg_s > 0.0
    assert plain.tail_mirror_erg_s == 0.0
    # The refusals, each by its own message.
    no_window = dict(mirror_face=1)
    for kwargs, needle in (
        (_mirror_mg_kwargs(cells, mirror_face=1, tail_reflect_face=1,
                           tail_reflect_threshold_eV=200.0),
         "name the same window face"),
        (no_window, "mirror_face needs tail_walk_window"),
        (_mirror_mg_kwargs(cells, mirror_face=1,
                           tail_walk_window=(0, cells - 2)),
         "must end there too"),
        (dict(_mirror_mg_kwargs(cells, mirror_face=1),
              tail_ionization="off"),
         "requires tail_ionization='on'"),
        (dict(mirror_face=1, tail_walk_window=(0, cells - 1),
              product_transport="nonlocal"),
         "product_transport='nonlocal'"),
        (_mirror_mg_kwargs(cells, mirror_face=2), "mirror_face must be"),
    ):
        try:
            _deposit_beam_ray(*args, **kwargs)
        except ValueError as exc:
            assert needle in str(exc), (needle, str(exc))
        else:
            raise AssertionError(f"{needle}: ACCEPTED")


# --------------------------------------------------------------------
# mirror-tail-full-window-fold
# --------------------------------------------------------------------
@_case("mirror-tail-full-window-fold")
def _case_mirror_tail_full_window_fold():
    """The mirror turn IS the image half: walkers on the full window, folded.

    The walked plateau tail of one ray on the half column with the mirror
    turn, against the same ray on the FULL window -- the half column and its
    reflection, no mirror, both ends free exits -- folded onto the half: every
    per-cell bank (heating, the anomalous delivery, ionization, excitation,
    radiation, cost) equal to 1e-12 relative, per cell. The walkers turn round
    at unchanged energy, so each returning leg marches the floats the full
    window's walker marches in the image half. What the full window lets out
    through its far end is what the half lets out through the cathode face
    after the turn. The primary stops short of the plane here, so the walkers
    are the only population that turns. NEGATIVE CONTROL: without the mirror
    the half column's walkers leave through the far face and the fold
    differs by far more than the bar.
    """
    cells = 24
    nn, ne, Te, dz = _mirror_column(cells)
    half = _deposit_beam_ray(
        200.0, 1.0e20, nn, ne, Te, 0, 1, dz,
        **_mirror_mg_kwargs(cells, mirror_face=1),
    )
    full = _deposit_beam_ray(
        200.0, 1.0e20, *_mirror_full(nn, ne, Te), 0, 1, *_mirror_full(dz),
        **_mirror_mg_kwargs(2 * cells),
    )
    # Non-vacuity: walkers reach the plane; the primary does not.
    assert half.tail_mirror_erg_s > 0.01 * half.tail_power_erg_s, (
        half.tail_mirror_erg_s, half.tail_power_erg_s
    )
    assert half.primary_mirror_flux_per_s == 0.0
    assert float(full.E_entry_eV[cells]) == 0.0
    worst = {
        name: _mirror_fold_worst(
            getattr(half, name), getattr(full, name), cells
        )
        for name in _MIRROR_BANKS
    }
    print(
        "mirror-tail-full-window-fold: worst per-cell relative difference "
        + ", ".join(f"{k}={v:.2e}" for k, v in worst.items())
    )
    assert max(worst.values()) <= 1.0e-12, worst
    exits_full = full.end_loss_tail_low_erg_s + full.end_loss_tail_high_erg_s
    assert math.isclose(
        half.end_loss_tail_low_erg_s, exits_full, rel_tol=1.0e-12
    ), (half.end_loss_tail_low_erg_s, exits_full)
    assert half.end_loss_tail_high_erg_s == 0.0
    assert half.tail_leg_cap_residual_erg_s == 0.0
    # NEGATIVE CONTROL.
    plain = _deposit_beam_ray(
        200.0, 1.0e20, nn, ne, Te, 0, 1, dz, **_mirror_mg_kwargs(cells),
    )
    control = max(
        _mirror_fold_worst(getattr(plain, name), getattr(full, name), cells)
        for name in _MIRROR_BANKS
    )
    assert control > 1.0e-3, control


# --------------------------------------------------------------------
# mirror-tail-cull-rearmed
# --------------------------------------------------------------------
@_case("mirror-tail-cull-rearmed")
def _case_mirror_tail_cull_rearmed():
    """A walker returning from the mirror is a NEW walker for the anode cull.

    One walker population, born gap-side of the anode plane and launched
    toward the mirror on a column with no neutrals and almost no plasma,
    crosses the plane (the mesh takes ``eta``), turns round at the plane at
    UNCHANGED energy, is re-armed, crosses the plane again (the mesh takes
    ``eta`` of what is left) and leaves through the cathode face. The mesh
    therefore takes ``eta (2 - eta)`` of the launched flux, the mirror sees
    ``(1 - eta)`` of it arrive, and the returning leg starts at the exact
    energy the first leg arrived with. NEGATIVE CONTROL: the end wall chains
    (first crossing only, no turn) take ``eta`` alone.
    """
    cells = 20
    nn = np.zeros(cells)
    ne = np.full(cells, 1.0e2)
    Te = np.full(cells, 1.0)
    dz = np.full(cells, 10.0)
    flux = np.zeros(cells)
    flux[4] = 1.0e18
    plans = [(100.0, flux, None, True)]
    cull = (8, _MIRROR_ETA, 0.0, 0.0, 0.0)
    layout, ledger = _beam_deposition_mod._tail_mirror_chains(
        plans, nn, ne, Te, dz, _mirror_march_kwargs(), 0, cells - 1, None,
        0.0, 1, cull=cull,
    )
    (chain,) = layout[0]
    assert [leg[3] for leg in chain] == [1, -1], [leg[3] for leg in chain]
    f0 = 1.0e18
    expected = f0 * _MIRROR_ETA + (1.0 - _MIRROR_ETA) * f0 * _MIRROR_ETA
    assert math.isclose(ledger["culled_flux"], expected, rel_tol=1e-12), (
        ledger["culled_flux"], expected
    )
    assert math.isclose(
        ledger["mirror_flux"], (1.0 - _MIRROR_ETA) * f0, rel_tol=1e-12
    )
    # Unchanged energy at the turn: what arrived is what the return carries.
    E_arrived = chain[0][2]
    assert ledger["mirror_eV"] == ledger["mirror_flux"] * E_arrived
    assert ledger["escape_high_eV"] == 0.0
    assert ledger["escape_low_eV"] > 0.0 and ledger["cap_flux"] == 0.0
    # NEGATIVE CONTROL: the end wall's first-crossing cull.
    _wall, wall_take = _beam_deposition_mod._tail_recursive_chains(
        plans, nn, ne, Te, dz, _mirror_march_kwargs(), 0, cells - 1, None,
        0.0, cull=cull,
    )
    assert math.isclose(wall_take[0], f0 * _MIRROR_ETA, rel_tol=1e-12)
    assert not math.isclose(wall_take[0], expected, rel_tol=1e-6)


# --------------------------------------------------------------------
# mirror-tail-leg-cap-residual
# --------------------------------------------------------------------
@_case("mirror-tail-leg-cap-residual")
def _case_mirror_tail_leg_cap_residual():
    """The leg cap is spent exactly, and its residual is BOOKED, not dropped.

    (a) On a vacuum column (no neutrals, no plasma) a walker between the
    cathode face (which reflects it) and the mirror (which turns it) never
    loses energy: its tree marches exactly ``MIRROR_MAX_LEGS`` legs and the
    residual row carries its whole launched flux and power, exactly.
    (b) On a walked-tail ray whose anode sheath repels every walker (a drop
    far above the plateau), nothing removes the walkers, the cap binds, and
    the tail identity
        P_tail = heating_anomalous + cost_tail + radiated_tail
                 + end_loss_tail + residual + (culled - returned)
    still closes to 1e-12, the residual a non-zero term of it.
    NEGATIVE CONTROL: without the mirror the same ray books no residual and
    its walkers leave through the far face instead.
    """
    cells = 12
    vacuum = (
        np.zeros(cells), np.zeros(cells), np.full(cells, 1.0),
        np.full(cells, 10.0),
    )
    flux = np.zeros(cells)
    flux[3] = 2.0e17
    layout, ledger = _beam_deposition_mod._tail_mirror_chains(
        [(80.0, flux, None, True)], *vacuum, _mirror_march_kwargs(), 0,
        cells - 1, -1, 1.0e9, 1,
    )
    legs = sum(len(chain) for chain in layout[0])
    assert legs == _beam_deposition_mod.MIRROR_MAX_LEGS, legs
    assert ledger["cap_flux"] == 2.0e17, ledger["cap_flux"]
    assert ledger["cap_eV"] == 2.0e17 * 80.0, ledger["cap_eV"]

    nn, ne, Te, dz = _mirror_column(cells)
    trap = dict(
        tail_reflect_face=-1, tail_reflect_threshold_eV=200.0,
        tail_anode_cross_index=5, tail_anode_eta=_MIRROR_ETA,
        tail_anode_phi_eV=1.0e4, plateau_groups=2,
    )
    # The capped branch itself: the residual bound is lifted so the budget's
    # booking can be read (mirror-residual-bound-raises owns the bound).
    with _mirror_residual_bound_lifted():
        res = _deposit_beam_ray(
            200.0, 1.0e20, nn, ne, Te, 0, 1, dz,
            **_mirror_mg_kwargs(cells, mirror_face=1, **trap),
        )

    def tail_gap(r):
        bank = r.tail_power_erg_s + r.plateau_wave_power_erg_s
        booked = (
            math.fsum(
                (r.heating_anomalous_erg_s + r.ionization_cost_tail_erg_s
                 + r.radiated_tail_erg_s).tolist()
            )
            + r.end_loss_tail_low_erg_s + r.end_loss_tail_high_erg_s
            + r.tail_leg_cap_residual_erg_s
            + r.tail_anode_culled_erg_s - r.tail_anode_returned_erg_s
        )
        return abs(booked - bank) / bank

    assert res.tail_anode_sheath_reflected_flux_per_s > 0.0
    frac = res.tail_leg_cap_residual_erg_s / res.tail_power_erg_s
    print(
        "mirror-tail-leg-cap-residual: trapped tail residual "
        f"{frac:.3e} of the launched tail power, identity "
        f"{tail_gap(res):.2e}"
    )
    assert res.tail_leg_cap_residual_erg_s > 1.0e-6 * res.tail_power_erg_s
    assert res.tail_leg_cap_residual_flux_per_s > 0.0
    assert tail_gap(res) <= 1.0e-12, tail_gap(res)
    assert res.end_loss_tail_high_erg_s == 0.0
    # NEGATIVE CONTROL.
    wall = _deposit_beam_ray(
        200.0, 1.0e20, nn, ne, Te, 0, 1, dz,
        **_mirror_mg_kwargs(cells, **trap),
    )
    assert wall.tail_leg_cap_residual_erg_s == 0.0
    assert wall.end_loss_tail_high_erg_s > 0.0
    assert tail_gap(wall) <= 1.0e-12


# --------------------------------------------------------------------
# mirror-beam-smoothing-fold
# --------------------------------------------------------------------
@_case("mirror-beam-smoothing-fold")
def _case_mirror_beam_smoothing_fold():
    """The beam smoothing folds its Gaussian about the mirror plane.

    On the template at nx = 40 the TwinCathode machine is the half column
    reflected about Lm/2. A mirror-symmetric deposit smoothed by the twin's
    matrix, restricted to the half, equals the half column's own smoothing
    of the half deposit to 1e-12 relative per live cell at a 50 cm width, and
    the half matrix still conserves (every live column sums to 1).
    NEGATIVE CONTROL: the half column's matrix built without the mirror fold
    (the same geometry with no mirror face) differs by more than 1e-3 near
    the plane.
    """
    from cablp.solvers._sim1d.core.geometry import build_geometry

    params, flags = default_config()
    params["nx"] = 40
    half = build_geometry(dict(params, far_end="mirror"), flags)
    twin = build_geometry(params, dict(flags, TwinCathode=True))
    cells = int(half.cells)
    assert int(twin.cells) == 2 * cells
    live = np.asarray(half.plasma_active, dtype=bool)
    rng = np.random.default_rng(7)
    ext = np.where(live, rng.uniform(0.5, 1.5, cells), 0.0)
    ext_full = np.concatenate([ext, ext[::-1]])
    W_half = _beam_smoothing_matrix(half, 50.0)
    W_twin = _beam_smoothing_matrix(twin, 50.0)
    got = W_half @ ext
    want = (W_twin @ ext_full)[:cells]
    worst = float(
        np.max(np.abs(got[live] - want[live]) / np.abs(want[live]))
    )
    print(f"mirror-beam-smoothing-fold: worst relative difference {worst:.2e}")
    assert worst <= 1.0e-12, worst
    assert np.allclose(
        W_half[:, live].sum(axis=0), 1.0, rtol=0.0, atol=1.0e-13
    )
    # NEGATIVE CONTROL.
    bare = dataclasses.replace(
        half, mirror_face_indices=np.zeros(0, dtype=int)
    )
    assert _beam_smoothing_key(bare, 50.0) != _beam_smoothing_key(half, 50.0)
    control = _beam_smoothing_matrix(bare, 50.0) @ ext
    assert float(
        np.max(np.abs(control[live] - want[live]) / np.abs(want[live]))
    ) > 1.0e-3


# --------------------------------------------------------------------
# mirror-csda-primary-turn
# --------------------------------------------------------------------
@_case("mirror-csda-primary-turn")
def _case_mirror_csda_primary_turn():
    """The CSDA primary turns round at the mirror plane and is booked there.

    (a) A primary that reaches the plane on the half column and stops on its
    way back, against the same ray on the full window (the half and its
    reflection) folded onto the half: every per-cell bank equal to 1e-12
    relative per cell; nothing is transmitted; the flux arriving at the plane
    is the launched flux and its power is that flux at the full window's
    entry energy of the first image cell, exactly; the per-ray energy
    identity closes to 1e-12.
    (b) On a column with no neutrals and almost no plasma the primary
    bounces between the cathode sheath and the plane; the anode mesh is
    re-armed on every return from the plane (33 interceptions in 64 legs),
    so the residual flux is exactly ``(1 - eta)**33`` of the launch, booked
    in its row, and the identity still closes.
    NEGATIVE CONTROL: without ``mirror_face`` the half ray transmits the
    surviving flux out of the far end and is intercepted once.
    """
    cells = 30
    nn = np.full(cells, 3.5e14) * np.linspace(1.2, 0.8, cells)
    ne = np.full(cells, 1.0e10)
    Te = np.linspace(3.0, 2.0, cells)
    dz = np.linspace(8.0, 12.0, cells)
    window = dict(tail_walk_window=(0, cells - 1))
    half = _deposit_beam_ray(
        150.0, 1.0e18, nn, ne, Te, 0, 1, dz, mirror_face=1, **window,
    )
    full = _deposit_beam_ray(
        150.0, 1.0e18, *_mirror_full(nn, ne, Te), 0, 1, *_mirror_full(dz),
    )
    assert float(full.transmitted_flux) == 0.0  # it stops in the image half
    assert float(full.E_entry_eV[cells]) > 0.0  # ...after crossing the plane
    worst = max(
        _mirror_fold_worst(getattr(half, name), getattr(full, name), cells)
        for name in _MIRROR_BANKS
    )
    print(
        f"mirror-csda-primary-turn: worst per-cell fold difference {worst:.2e}"
    )
    assert worst <= 1.0e-12, worst
    assert half.transmitted_flux == 0.0 and half.transmitted_energy_eV == 0.0
    assert half.primary_mirror_flux_per_s == 1.0e18
    assert half.primary_mirror_erg_s == (
        1.0e18 * float(full.E_entry_eV[cells]) * ev_to_erg
    )

    def ray_gap(r, E0, G0):
        booked = (
            math.fsum(
                (r.plasma_heating_erg_s + r.radiated_erg_s
                 + r.ionization_cost_erg_s).tolist()
            )
            + r.anode_intercepted_erg_s
            + r.transmitted_flux * r.transmitted_energy_eV * ev_to_erg
            + r.primary_mirror_residual_erg_s
        )
        return abs(booked - G0 * E0 * ev_to_erg) / (G0 * E0 * ev_to_erg)

    assert ray_gap(half, 150.0, 1.0e18) <= 1.0e-12
    # (b) the bouncing primary.
    thin = (np.zeros(cells), np.full(cells, 1.0e2), np.full(cells, 1.0))
    bounce = _deposit_beam_ray(
        150.0, 1.0e18, *thin, 0, 1, dz, mirror_face=1,
        anode_cross_index=5, anode_eta=_MIRROR_ETA, **window,
    )
    expected = 1.0e18
    for _ in range(33):
        expected *= 1.0 - _MIRROR_ETA
    assert bounce.primary_mirror_residual_flux_per_s == expected, (
        bounce.primary_mirror_residual_flux_per_s, expected
    )
    assert bounce.primary_mirror_residual_erg_s > 0.0
    assert ray_gap(bounce, 150.0, 1.0e18) <= 1.0e-12
    # NEGATIVE CONTROL.
    plain = _deposit_beam_ray(
        150.0, 1.0e18, *thin, 0, 1, dz, anode_cross_index=5,
        anode_eta=_MIRROR_ETA,
    )
    assert plain.transmitted_flux == (1.0 - _MIRROR_ETA) * 1.0e18
    assert plain.primary_mirror_residual_flux_per_s == 0.0
    assert plain.anode_intercepted_erg_s < bounce.anode_intercepted_erg_s


# --------------------------------------------------------------------
# mirror-cathode-coupling-constructs
# --------------------------------------------------------------------
def _mirror_circuit_config(far_end):
    """A short scheduled discharge with the cathode circuit on, 24 far cells."""
    params, flags = default_config()
    flags.update(
        neutral_momentum=False, neutral_energy=False,
        neutral_hot_internal_wall=False,
    )
    params.update({
        "cathode_neutral_jet": False,
        "cathode_jet_surface_debit": False,
        "cathode_jet_energy_convention": "legacy",
        "dt_save": 0.0,
        "phase_transition_mode": "scheduled",
        "tau_neutral_prebreakdown": 0.0,
        "tau_prebreakdown": 0.0,
        "tau_breakdown": 0.0,
        "tau_discharge": 1.0,
        "tau_afterglow": 0.0,
        "nx": 24,
        "beam_deposition_smoothing_cm": 50.0,
        "initial_neutral_state": "fill",
    })
    if far_end == "mirror":
        params.update(far_end="mirror", S_pump_R=0.0)
    return params, flags


@_case("mirror-cathode-coupling-constructs")
def _case_mirror_cathode_coupling_constructs():
    """The half column runs with the cathode circuit on.

    ``far_end = "mirror"`` with ``cathode_coupling`` constructs and runs a
    short discharge: the CSDA ray fires, the primary reaches the plane and
    turns round there (its arrival rows fill), nothing is transmitted out of
    the far end and no end-loss row fills, the leg-cap rows are saved, and the
    mirror rows survive a save and reload. The walked tail with the circuit
    is refused at the mirror under the default ``anode_tail_booking``, naming
    its reason.
    NEGATIVE CONTROL: the end wall run carries none of the mirror rows, so
    their presence is the mirror's and not a seeded default.
    """
    import tempfile
    from pathlib import Path

    from cablp.solvers._sim1d.results.io import load_result_hdf5

    params, flags = _mirror_circuit_config("mirror")
    assert flags["cathode_coupling"] is True
    sim = LAPDSim1D(params, flags)
    result = sim.run(t_end=2.0e-6, dt=1.0e-7)
    diag = result.cathode_diagnostics
    assert float(np.max(diag["beam_csda_active"])) == 1.0
    assert float(np.max(diag["source_beam_mirror_primary_flux_per_s"])) > 0.0
    assert float(np.max(diag["source_beam_mirror_primary_W"])) > 0.0
    for name in (
        "source_beam_transmitted_W",
        "source_beam_transmitted_flux_per_s",
        "source_beam_end_loss_high_W",
        "source_beam_end_loss_tail_high_W",
        "source_beam_mirror_primary_residual_W",
        "source_beam_tail_leg_cap_residual_W",
    ):
        assert float(np.max(np.abs(diag[name]))) == 0.0, name
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "mirror.h5"
        sim.save_result(path, result)
        loaded = load_result_hdf5(path)
    assert np.array_equal(
        loaded.cathode_diagnostics["source_beam_mirror_primary_W"],
        diag["source_beam_mirror_primary_W"],
    )
    try:
        LAPDSim1D(
            dict(params, heating_anomalous_transport="plateau_multigroup"),
            flags,
        )
    except ValueError as exc:
        assert (
            "anode_tail_booking='emission_fraction' (got 'lagged_current' "
            "with heating_anomalous_transport='plateau_multigroup'"
            in str(exc)
        ), str(exc)
    else:
        raise AssertionError("the walked tail ACCEPTED at a mirror")
    # NEGATIVE CONTROL.
    wall_params, wall_flags = _mirror_circuit_config("end_wall")
    wall = LAPDSim1D(wall_params, wall_flags).run(t_end=3.0e-7, dt=1.0e-7)
    assert not any(
        "mirror" in name or "leg_cap" in name
        for name in wall.cathode_diagnostics
    )


# --------------------------------------------------------------------
# mirror-primary-needs-absorbing-anode
# --------------------------------------------------------------------
@_case("mirror-primary-needs-absorbing-anode")
def _case_mirror_primary_needs_absorbing_anode():
    """A mirror with the circuit on is refused without a mesh that absorbs.

    Under ``far_end = "mirror"`` with ``cathode_coupling`` the CSDA primary
    bounces between the cathode sheath and the plane, and the leg series
    converges only because the anode mesh takes ``eta`` of it on every
    return. Both ways of losing that are refused at construction, each
    exercised: ``eta = 0`` and a geometry with no anode face.
    NEGATIVE CONTROL: the same settings construct where no primary is walked
    at a mirror -- ``eta = 0`` on the end wall, and ``eta = 0`` at a mirror
    with the circuit off -- and the mirror with the circuit on and the
    shipped ``eta`` constructs, so the refusal is the combination's.
    """
    from unittest import mock

    from cablp.solvers._sim1d import solver as _solver_mod

    needle = "converges only while the anode mesh absorbs"
    params, flags = _mirror_circuit_config("mirror")
    assert float(params["eta"]) > 0.0
    try:
        LAPDSim1D(dict(params, eta=0.0), flags)
    except ValueError as exc:
        assert needle in str(exc), str(exc)
        assert "eta=0.0" in str(exc), str(exc)
    else:
        raise AssertionError("eta = 0 ACCEPTED at a mirror with the circuit")
    real_build = _solver_mod.build_geometry

    def no_anode(*args, **kwargs):
        geometry = real_build(*args, **kwargs)
        return dataclasses.replace(
            geometry, anode_face_indices=np.zeros(0, dtype=int)
        )

    with mock.patch.object(_solver_mod, "build_geometry", no_anode):
        try:
            LAPDSim1D(params, flags)
        except ValueError as exc:
            assert needle in str(exc), str(exc)
            assert "0 anode face(s)" in str(exc), str(exc)
        else:
            raise AssertionError("no anode face ACCEPTED at a mirror")
    # NEGATIVE CONTROL.
    LAPDSim1D(params, flags)
    wall_params, wall_flags = _mirror_circuit_config("end_wall")
    LAPDSim1D(dict(wall_params, eta=0.0), wall_flags)
    LAPDSim1D(dict(params, eta=0.0), dict(flags, cathode_coupling=False))


# --------------------------------------------------------------------
# mirror-residual-bound-raises
# --------------------------------------------------------------------
@_case("mirror-residual-bound-raises")
def _case_mirror_residual_bound_raises():
    """A mirrored ray whose leg budget leaves power unmarched is refused.

    ``deposit_beam(mirror_face=...)`` raises when the tail leg-cap residual
    plus the primary's residual exceeds ``MIRROR_RESIDUAL_MAX_FRACTION`` of
    the ray's launched power ``Gamma0 * E0``. Both components are exercised:
    (a) a primary bouncing on a near-vacuum column behind a thin mesh
    (``eta = 0.05``: ``0.95**33`` of it is left after 64 legs), and
    (b) walkers between a reflecting cathode and the mirror with no anode
    to remove them. Each raises a RuntimeError naming the bound.
    NEGATIVE CONTROL: with the bound lifted -- the pre-bound behaviour -- the
    same two calls return and book a residual above the bound, so the
    refusal is the bound's; and the primary behind the shipped mesh
    (``eta = 0.358``) converges under the bound and returns.
    """
    cells = 30
    dz = np.linspace(8.0, 12.0, cells)
    thin = (np.zeros(cells), np.full(cells, 1.0e2), np.full(cells, 1.0))
    window = dict(tail_walk_window=(0, cells - 1), mirror_face=1)
    bound = _beam_deposition_mod.MIRROR_RESIDUAL_MAX_FRACTION
    assert bound == 1.0e-4, bound
    primary = (
        (150.0, 1.0e18, *thin, 0, 1, dz),
        dict(window, anode_cross_index=5, anode_eta=0.05),
    )
    walker_cells = 24
    nn, ne, Te, wdz = _mirror_column(walker_cells)
    walkers = (
        (200.0, 1.0e20, nn, ne, Te, 0, 1, wdz),
        _mirror_mg_kwargs(
            walker_cells, mirror_face=1, tail_reflect_face=-1,
            tail_reflect_threshold_eV=200.0,
        ),
    )
    shares = {}
    for label, (args, kwargs), row in (
        ("primary", primary, "primary_mirror_residual_erg_s"),
        ("walkers", walkers, "tail_leg_cap_residual_erg_s"),
    ):
        try:
            _deposit_beam_ray(*args, **kwargs)
        except RuntimeError as exc:
            assert "MIRROR_RESIDUAL_MAX_FRACTION" in str(exc), str(exc)
        else:
            raise AssertionError(f"{label}: an unconverged mirror ray RETURNED")
        # NEGATIVE CONTROL: the pre-bound behaviour.
        with _mirror_residual_bound_lifted():
            res = _deposit_beam_ray(*args, **kwargs)
        share = getattr(res, row) / (args[0] * args[1] * ev_to_erg)
        assert share > bound, (label, share)
        shares[label] = share
    converged = _deposit_beam_ray(
        150.0, 1.0e18, *thin, 0, 1, dz,
        **dict(window, anode_cross_index=5, anode_eta=_MIRROR_ETA),
    )
    conv_share = converged.primary_mirror_residual_erg_s / (
        150.0 * 1.0e18 * ev_to_erg
    )
    assert 0.0 < conv_share < bound, conv_share
    print(
        "mirror-residual-bound-raises: refused at residual shares "
        + ", ".join(f"{k}={v:.3e}" for k, v in shares.items())
        + f"; eta=0.358 primary returns at {conv_share:.3e}"
    )


# --------------------------------------------------------------------
# mirror-tail-sheath-share-merged
# --------------------------------------------------------------------
@_case("mirror-tail-sheath-share-merged")
def _case_mirror_tail_sheath_share_merged():
    """The sheath-turned share rejoins its parent at the anode plane.

    In the reflecting regime (the wires' sheath repels every walker) a walker
    crossing the anode plane toward the cathode splits: the sheath turns
    ``eta`` of it back at the plane and the rest walks on to the cathode,
    which turns it back through the gap. The two are superposed at the plane
    (fluxes summed, energy flux-weighted) into ONE walker, so each launched
    walker is a single chain and not a tree. On a column with no neutrals and
    almost no plasma nothing is lost: each launched walker marches one chain, the
    gap return carries ``(1 - eta)`` of the flux to the plane, the merged
    walker reaches the mirror carrying the whole launched flux (1e-12) at the
    launch energy (1e-9, the column's Coulomb loss), and the leg cap books the
    whole launch.
    NEGATIVE CONTROL: a crossing toward the MIRROR (a gap-born walker) has no
    parent coming back through the same plane, so its turned share is still
    a walker of its own -- that launch holds a second chain entry.
    """
    cells = 20
    nn = np.zeros(cells)
    ne = np.full(cells, 1.0e2)
    Te = np.full(cells, 1.0)
    dz = np.full(cells, 10.0)
    f0 = 1.0e18
    E0 = 100.0
    cull = (8, _MIRROR_ETA, 0.0, 0.0, 1.0e4)  # the sheath repels at 1e4 eV
    flux = np.zeros(cells)
    flux[12] = f0
    layout, ledger = _beam_deposition_mod._tail_mirror_chains(
        [(E0, flux, flux.copy(), True)], nn, ne, Te, dz,
        _mirror_march_kwargs(), 0, cells - 1, -1, 1.0e9, 1, cull=cull,
    )
    # One marched chain per launched walker. A turned share still held when
    # the leg budget runs out is released and capped at once, which leaves an
    # empty chain and no legs.
    chains = [chain for chain in layout[0] if chain]
    assert len(chains) == 2, [len(c) for c in layout[0]]
    assert len(layout[0]) <= 4, [len(c) for c in layout[0]]
    assert ledger["sheath_flux"] > 0.0
    merges = 0
    for chain in chains:
        dirs = [leg[3] for leg in chain]
        for k in range(len(chain) - 2):
            # crossing leg to the cathode face, the gap return, then the
            # merged walker from the plane to the mirror
            if dirs[k] == -1 and dirs[k + 1] == 1 and dirs[k + 2] == 1:
                merges += 1
                assert math.isclose(
                    chain[k + 1][1], (1.0 - _MIRROR_ETA) * f0, rel_tol=1e-12
                ), (chain[k + 1][1], f0)
                gap_banks = np.concatenate(chain[k + 1][0])
                assert np.all(gap_banks.reshape(5, cells)[:, 8:] == 0.0)
                assert math.isclose(chain[k + 2][1], f0, rel_tol=1e-12), (
                    chain[k + 2][1], f0
                )
                # The column's Coulomb loss is ~1e-11 of E per leg here.
                assert math.isclose(chain[k + 2][2], E0, rel_tol=1e-9)
    assert merges >= 2, merges
    assert math.isclose(ledger["cap_flux"], 2.0 * f0, rel_tol=1e-12), (
        ledger["cap_flux"]
    )
    assert ledger["escape_low_eV"] == 0.0 and ledger["escape_high_eV"] == 0.0
    # NEGATIVE CONTROL: a gap-born walker heading for the mirror.
    gap_flux = np.zeros(cells)
    gap_flux[4] = f0
    control, _ledger = _beam_deposition_mod._tail_mirror_chains(
        [(E0, gap_flux, None, True)], nn, ne, Te, dz,
        _mirror_march_kwargs(), 0, cells - 1, -1, 1.0e9, 1, cull=cull,
    )
    # The launch holds its own chain and the turned share's (which the shared
    # budget, spent by the parent's bounces, caps at once).
    assert len(control[0]) == 2, [len(c) for c in control[0]]
    assert len(control[0][0]) == _beam_deposition_mod.MIRROR_MAX_LEGS
    print(
        f"mirror-tail-sheath-share-merged: {merges} merges in "
        f"{sum(len(c) for c in chains)} legs on two chains; the gap-born "
        f"control holds {len(control[0])} chain entries"
    )


def _booking_unit_solve_inputs():
    """(config, plasma, I_i_a, I_e_sat) for the anode-booking unit solves.

    A long-mean-free-path gap (no neutrals, no beam cross section), so the
    beam's gap survival ``beta`` is order one and the ``w_gap`` term is
    resolvable in ``phi_a``.
    """
    from cablp.cathode import circuit_common as _circ
    from cablp.constants import m_He_cgs as _mi
    from cablp.plasma.params import (
        bohm_sound_speed as _cs, electron_mean_speed as _ve,
    )
    from cablp.solvers._sim1d.physics.cathode import cathode_device_config

    params, flags = _cathode_unit_config()
    cfg = cathode_device_config(params, flags, 4.0, _mi)
    Te, ne = 4.0, 2.0e12
    plasma = _circ.PlasmaState(T_e=Te, n_e=ne, n_n=0.0, sigma_b=0.0)
    area = 2.0 * cfg.eta * cfg.A_c
    I_i_a = area * qe_SI * ne * _cs(Te, _mi) * math.exp(-0.5)
    I_e_sat = 0.25 * ne * _ve(Te) * area * qe_SI
    return cfg, plasma, I_i_a, I_e_sat


# --------------------------------------------------------------------
# anode-tail-booking-identity
# --------------------------------------------------------------------
@_case("anode-tail-booking-identity")
def _case_anode_tail_booking_identity():
    """``anode_tail_booking = "emission_fraction"`` books what it states.

    On a synthetic solve with known ``w_gap``, ``c_ret`` and ``c_tail``, where
    the beam clears the anode sheath (``phi_a < phi_c``), the anode sheath
    passes ``I_i,a + I_tot - (eta*beta*(1 - w_gap) + c_ret + c_tail)*I_star``,
    ``phi_a = T_e,a ln(I_e,sat / that)``, the whole direct term is booked
    (``anode_direct_collected_fraction == 1``), and ``P_tail_phi`` is
    ``phi_a * c_tail * I_star``. The circuit refuses mixed or out-of-range
    booking inputs.
    NEGATIVE CONTROL: the default path with no booking arguments is
    bit-identical to an explicit ``"lagged_current"``, carries no collected
    fraction, and at ``w_gap = c_ret = 0`` the booking equals it fed
    ``c_tail * I_star`` as its absolute tail current; a non-zero ``w_gap`` or
    ``c_ret`` moves ``phi_a``.
    """
    from cablp.cathode.circuit_idriven import solve_idriven

    cfg, plasma, I_i_a, I_e_sat = _booking_unit_solve_inputs()
    common = dict(
        anode_current_A=I_i_a, anode_T_e=plasma.T_e,
        anode_electron_saturation_A=I_e_sat, I_tot_A=3000.0,
    )
    c_tail, w_gap, c_ret = 0.21, 0.37, 0.05
    r = solve_idriven(
        cfg, plasma, anode_tail_booking="emission_fraction",
        tail_anode_coefficient=c_tail, anode_gap_walker_fraction=w_gap,
        primary_return_coefficient=c_ret, **common,
    )
    beta = r.beam_bypass_fraction
    assert beta > 0.05, beta
    I_e_a = I_i_a + r.I_tot - (
        cfg.eta * beta * (1.0 - w_gap) + c_ret + c_tail
    ) * r.I_eth_star
    assert I_e_a > 0.0, I_e_a
    want = plasma.T_e * math.log(I_e_sat / I_e_a)
    assert r.phi_a < r.phi_c, (r.phi_a, r.phi_c)
    assert abs(r.phi_a - want) <= 1e-9 * abs(want), (r.phi_a, want)
    assert r.anode_direct_collected_fraction == 1.0
    assert abs(
        r.P_tail_phi - max(r.phi_a, 0.0) * c_tail * r.I_eth_star
    ) <= 1e-12 * abs(r.P_tail_phi)
    # NEGATIVE CONTROL.
    base = solve_idriven(cfg, plasma, **common)
    explicit = solve_idriven(
        cfg, plasma, anode_tail_booking="lagged_current", **common
    )
    assert base.phi_a == explicit.phi_a and base.V_b == explicit.V_b
    assert math.isnan(base.anode_direct_collected_fraction)
    at_zero = solve_idriven(
        cfg, plasma, anode_tail_booking="emission_fraction",
        tail_anode_coefficient=c_tail, **common,
    )
    lagged = solve_idriven(
        cfg, plasma, tail_anode_current_A=c_tail * at_zero.I_eth_star,
        **common,
    )
    assert abs(at_zero.phi_a - lagged.phi_a) <= 1e-9 * abs(lagged.phi_a), (
        at_zero.phi_a, lagged.phi_a,
    )
    assert abs(r.phi_a - at_zero.phi_a) > 1e-6 * abs(at_zero.phi_a)
    for kwargs, needle in (
        (dict(anode_tail_booking="bogus"), "must be 'lagged_current' or"),
        (dict(tail_anode_coefficient=0.1), "belongs to 'emission_fraction'"),
        (dict(primary_return_coefficient=0.1),
         "belongs to 'emission_fraction'"),
        (dict(anode_tail_booking="emission_fraction",
              tail_anode_current_A=1.0), "belongs to 'lagged_current'"),
        (dict(anode_tail_booking="emission_fraction",
              anode_gap_walker_fraction=1.5), "must be in [0, 1]"),
        (dict(anode_tail_booking="emission_fraction",
              primary_return_coefficient=-0.1), "finite and >= 0"),
    ):
        try:
            solve_idriven(cfg, plasma, **common, **kwargs)
        except ValueError as exc:
            assert needle in str(exc), str(exc)
        else:
            raise AssertionError(f"{kwargs} ACCEPTED")


# --------------------------------------------------------------------
# anode-tail-booking-conservation-assert
# --------------------------------------------------------------------
@_case("anode-tail-booking-conservation-assert")
def _case_anode_tail_booking_conservation_assert():
    """The booked direct collection may not exceed the emission.

    ``anode_tail_booking_coefficients`` forms ``c_tail = I_tail / I_emit``,
    ``w_gap = e * gap_born_flux / I_emit`` and
    ``c_ret = e * primary_net_return_flux / I_emit`` and raises a RuntimeError
    when ``eta*beta*(1 - w_gap) + c_ret + c_tail > 1`` (over-counting inputs:
    the tail beside an un-netted primary, and a return interception pushing
    an otherwise admissible total over one) or when ``w_gap`` leaves
    ``[0, 1]``. On a walked ray with the net-basis ledger, births + net
    interceptions + net remnant equal ``Gamma0`` to roundoff, and the
    primary-side births equal the launched walker flux. THE RETURN RULE: a
    mirrored primary whose returns reach the anode plane below the wires'
    sheath has every return's eta share turned back rather than booked -- no
    net return interception, the anode row carries the outbound crossing
    alone, more flux reaches the mirror -- and the ledger and the ray's power
    identity still close.
    NEGATIVE CONTROL: netted inputs pass and return the three coefficients,
    no emission returns zeros, the same ray without ``primary_net_basis``
    leaves the net rows at zero with every other row bit-identical, and the
    mirrored primary against an attracting anode (``phi = 0``) books its
    returns.
    """
    import dataclasses as _dc

    from cablp.solvers._sim1d.physics.cathode import (
        anode_tail_booking_coefficients,
    )

    eta, beta, I_emit = 0.358, 1.0, 10.0
    for args, needle in (
        ((9.0, I_emit, 0.0, eta, beta), "exceeds the emission"),
        ((1.0, I_emit, 11.0 / qe_SI, eta, beta), "w_gap in [0, 1]"),
        ((9.0, I_emit, 8.0 / qe_SI, eta, beta, 0.4 / qe_SI),
         "exceeds the emission"),
    ):
        try:
            anode_tail_booking_coefficients(*args)
        except RuntimeError as exc:
            assert needle in str(exc), str(exc)
        else:
            raise AssertionError(f"over-count {args} ACCEPTED")
    # NEGATIVE CONTROL.
    c_tail, w_gap, c_ret = anode_tail_booking_coefficients(
        9.0, I_emit, 8.0 / qe_SI, eta, beta, 0.02 / qe_SI
    )
    assert abs(c_tail - 0.9) < 1e-15 and abs(w_gap - 0.8) < 1e-12
    assert abs(c_ret - 0.002) < 1e-12
    assert eta * beta * (1.0 - w_gap) + c_ret + c_tail <= 1.0
    assert anode_tail_booking_coefficients(0.0, 0.0, 0.0, eta, beta) == (
        0.0, 0.0, 0.0,
    )
    cells = 24
    nn, ne, Te, dz = _mirror_column(cells)
    args = (300.0, 1.0e18, nn, ne, Te, 0, 1, dz)
    kwargs = _mirror_mg_kwargs(cells, anode_cross_index=6, anode_eta=_MIRROR_ETA)
    net = _deposit_beam_ray(*args, primary_net_basis=True,
                            primary_anode_collected_fraction=1.0, **kwargs)
    G0 = args[1]
    total = (
        net.primary_births_flux_per_s + net.primary_net_direct_flux_per_s
        + net.primary_net_return_flux_per_s
        + net.primary_net_remnant_flux_per_s
    )
    assert abs(total - G0) <= 1e-12 * G0, (total, G0)
    assert net.primary_net_direct_flux_per_s > 0.0
    assert abs(
        net.primary_births_flux_per_s - net.tail_launched_flux_per_s
    ) <= 1e-9 * net.tail_launched_flux_per_s
    assert 0.0 < net.tail_gap_born_flux_per_s <= net.tail_launched_flux_per_s
    # NEGATIVE CONTROL.
    gross = _deposit_beam_ray(*args, **kwargs)
    for field in _dc.fields(gross):
        a, b = getattr(gross, field.name), getattr(net, field.name)
        if field.name.startswith("primary_births") or field.name.startswith(
            "primary_net_"
        ):
            assert a == 0.0, field.name
            continue
        assert np.array_equal(np.asarray(a), np.asarray(b)), field.name
    no_plane = _deposit_beam_ray(*args, **_mirror_mg_kwargs(cells))
    assert no_plane.tail_gap_born_flux_per_s == 0.0
    # THE RETURN RULE, on a mirrored primary that stops within its budget.
    m_nn, m_ne = np.full(cells, 3.0e12), np.full(cells, 3.0e11) * np.linspace(
        1.0, 2.0, cells
    )
    m_args = (60.0, 1.0e18, m_nn, m_ne, Te, 0, 1, dz)

    def mirrored(phi):
        return _deposit_beam_ray(*m_args, **_mirror_mg_kwargs(
            cells, mirror_face=1, tail_reflect_face=-1,
            tail_reflect_threshold_eV=60.0, anode_cross_index=5,
            anode_eta=_MIRROR_ETA, tail_anode_cross_index=5,
            tail_anode_eta=_MIRROR_ETA, tail_anode_phi_eV=phi,
            primary_net_basis=True, primary_anode_collected_fraction=1.0,
        ))

    turned, kept = mirrored(1.0e4), mirrored(0.0)
    for res in (turned, kept):
        total = (
            res.primary_births_flux_per_s + res.primary_net_direct_flux_per_s
            + res.primary_net_return_flux_per_s
            + res.primary_net_remnant_flux_per_s
        )
        assert abs(total - m_args[1]) <= 1e-12 * m_args[1], total
        booked = (
            math.fsum((res.plasma_heating_erg_s + res.radiated_erg_s
                       + res.ionization_cost_erg_s).tolist())
            + res.anode_intercepted_erg_s
            + res.end_loss_low_erg_s + res.end_loss_high_erg_s
            + res.end_loss_tail_low_erg_s + res.end_loss_tail_high_erg_s
            + res.primary_mirror_residual_erg_s
            + res.tail_leg_cap_residual_erg_s
        )
        power = m_args[0] * m_args[1] * ev_to_erg
        assert abs(booked - power) <= 1e-12 * power, (booked, power)
    assert turned.primary_net_return_flux_per_s == 0.0
    assert turned.primary_mirror_flux_per_s > kept.primary_mirror_flux_per_s
    assert turned.anode_intercepted_erg_s < kept.anode_intercepted_erg_s
    # NEGATIVE CONTROL.
    assert kept.primary_net_return_flux_per_s > 0.0


# --------------------------------------------------------------------
# anode-balance-floor-probe-vs-dispatched
# --------------------------------------------------------------------
@_case("anode-balance-floor-probe-vs-dispatched")
def _case_anode_balance_floor_probe_vs_dispatched():
    """Under ``"emission_fraction"`` a balance with no floating solution
    raises, except at the circuit's I = 0 probe; the default floors it and
    reports it.

    A solve at ``I_tot = 0`` whose directly collected tail exceeds what the
    anode sheath can pass books none of the fast term under
    ``"emission_fraction"``, dispatched or probe alike, at
    ``phi_a = T_e,a ln(I_e,sat / I_i,a)``. The balance helper handed a
    non-positive current with no fast electron taken out raises ValueError
    naming the balance, and under ``probe`` returns the floored value
    ``T_e,a ln(I_e,sat / 1e-300)``. The circuit advance's stages evaluate
    their ``I = 0`` endpoint through ``vdis_bracket_probe`` only.
    NEGATIVE CONTROL: under the default the same infeasible dispatched solve
    returns the floored value with ``anode_floor_fired == 1`` (the probe with
    0), on the current-driven and the prescribed solve alike; a feasible solve
    is the same number with or without the flag, reports no firing; the
    routed advance lands on the same current as the unrouted one.
    """
    from cablp.cathode.circuit_idriven import solve_idriven
    from cablp.cathode.circuit_prescribed import solve_prescribed
    from cablp.solvers._sim1d.physics.cathode import (
        advance_circuit_current_driven,
    )

    cfg, plasma, I_i_a, I_e_sat = _booking_unit_solve_inputs()
    common = dict(
        anode_current_A=I_i_a, anode_T_e=plasma.T_e,
        anode_electron_saturation_A=I_e_sat,
    )
    from cablp.cathode.circuit_common import (
        ANODE_FAST_BRANCH_NONE, emission_fraction_anode_balance,
    )

    floored = math.log(I_e_sat / 1e-300) * plasma.T_e
    without = plasma.T_e * math.log(I_e_sat / I_i_a)
    for flag in (False, True):
        r = solve_idriven(cfg, plasma, I_tot_A=0.0,
                          anode_tail_booking="emission_fraction",
                          tail_anode_coefficient=0.9,
                          anode_balance_probe=flag, **common)
        assert r.anode_fast_branch == ANODE_FAST_BRANCH_NONE, flag
        assert r.anode_direct_collected_fraction == 0.0, flag
        assert abs(r.phi_a - without) <= 1e-12 * without, (r.phi_a, without)
    try:
        emission_fraction_anode_balance(
            0.0, 1.0, I_e_sat, plasma.T_e, 10.0, False, lambda: "unit"
        )
    except ValueError as exc:
        assert "anode sheath balance is infeasible" in str(exc), str(exc)
    else:
        raise AssertionError("an infeasible dispatched balance RETURNED")
    phi_p, f_p, branch_p = emission_fraction_anode_balance(
        0.0, 1.0, I_e_sat, plasma.T_e, 10.0, True, lambda: "unit"
    )
    assert phi_p == floored and f_p == 0.0, (phi_p, floored)
    assert branch_p == ANODE_FAST_BRANCH_NONE
    # NEGATIVE CONTROL.
    tail = 10.0 * I_i_a
    default = solve_idriven(cfg, plasma, I_tot_A=0.0,
                            tail_anode_current_A=tail, **common)
    assert default.phi_a == floored and default.anode_floor_fired == 1.0
    default_probe = solve_idriven(cfg, plasma, I_tot_A=0.0,
                                  tail_anode_current_A=tail,
                                  anode_balance_probe=True, **common)
    assert default_probe.phi_a == floored
    assert default_probe.anode_floor_fired == 0.0
    presc = solve_prescribed(cfg, plasma, I_tot_A=0.0, V_dis_V=40.0,
                             tail_anode_current_A=tail, **common)
    assert presc.anode_floor_fired == 1.0
    a = solve_idriven(cfg, plasma, I_tot_A=3000.0, **common)
    b = solve_idriven(cfg, plasma, I_tot_A=3000.0, anode_balance_probe=True,
                      **common)
    assert a.phi_a == b.phi_a and a.V_b == b.V_b
    assert a.anode_floor_fired == 0.0
    # The stage probe routing.
    seen = []

    def vdis(I):
        if I == 0.0:
            raise AssertionError("the I = 0 endpoint reached vdis_of_I")
        seen.append(I)
        return 40.0 + 0.01 * I

    def vdis_probe(I):
        assert I == 0.0, I
        return 40.0

    I_new, _, _ = advance_circuit_current_driven(
        100.0, 1.0e-6, 170.0, 5.0e-3, 6.6e-6, vdis,
        vdis_bracket_probe=vdis_probe,
    )
    ref, _, _ = advance_circuit_current_driven(
        100.0, 1.0e-6, 170.0, 5.0e-3, 6.6e-6, lambda I: 40.0 + 0.01 * I,
    )
    assert I_new == ref and seen, (I_new, ref)


# --------------------------------------------------------------------
# anode-tail-booking-mirror-walked-tail
# --------------------------------------------------------------------
@_case("anode-tail-booking-mirror-walked-tail")
def _case_anode_tail_booking_mirror_walked_tail():
    """The walked tail with the circuit at a mirror is gated on the booking.

    ``far_end = "mirror"`` with ``cathode_coupling`` and
    ``heating_anomalous_transport = "plateau_multigroup"`` constructs and
    steps under ``anode_tail_booking = "emission_fraction"``, committing the
    three lagged coefficients; under ``"lagged_current"`` it is refused,
    naming the booking and the trapped residual. ``"emission_fraction"``
    without the walked tail, or with the prescribed drive, is refused.
    NEGATIVE CONTROL: the same walked tail under ``"lagged_current"`` on the
    end wall constructs, so the refusal is the mirror's.
    """
    params, flags = _mirror_circuit_config("mirror")
    walked = dict(params, heating_anomalous_transport="plateau_multigroup")
    sim = LAPDSim1D(dict(walked, anode_tail_booking="emission_fraction"),
                    flags)
    for _ in range(3):
        sim.advance_one_step()
    assert 0.0 <= sim._cathode_anode_gap_walker_frac <= 1.0
    assert sim._cathode_tail_anode_coef >= 0.0
    assert sim._cathode_primary_return_coef >= 0.0
    try:
        LAPDSim1D(walked, flags)
    except ValueError as exc:
        assert (
            "anode_tail_booking='emission_fraction' (got 'lagged_current' "
            "with heating_anomalous_transport='plateau_multigroup'"
            in str(exc)
        ), str(exc)
        assert "10-46 % of the tail power at 64 legs" in str(exc), str(exc)
    else:
        raise AssertionError("the lagged walked tail ACCEPTED at a mirror")
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        trace = Path(tmp) / "trace.npz"
        np.savez(
            trace,
            discharge_time_ms=np.linspace(0.0, 10.0, 11),
            discharge_current_mean_a=np.full(11, 3000.0),
            discharge_voltage_positive_mean_v=np.full(11, 60.0),
        )
        prescribed = dict(
            walked, anode_tail_booking="emission_fraction",
            cathode_solver_model="prescribed_measured",
            cathode_prescribed_trace_path=str(trace),
            cathode_prescribed_t0_s=1.0e-3,
            cathode_prescribed_start_s=2.0e-3,
        )
        for bad, needle in (
            (dict(params, anode_tail_booking="emission_fraction"),
             "silent no-op"),
            (prescribed,
             "cathode_solver_model='prescribed_measured' solves phi_c FROM"),
        ):
            try:
                LAPDSim1D(bad, flags)
            except ValueError as exc:
                assert needle in str(exc), str(exc)
            else:
                raise AssertionError(f"{needle}: ACCEPTED")
    # NEGATIVE CONTROL.
    wall_params, wall_flags = _mirror_circuit_config("end_wall")
    LAPDSim1D(
        dict(wall_params, heating_anomalous_transport="plateau_multigroup"),
        wall_flags,
    )


# --------------------------------------------------------------------
# anode-direct-booking-sheath-rule
# --------------------------------------------------------------------
@_case("anode-direct-booking-sheath-rule")
def _case_anode_direct_booking_sheath_rule():
    """The primary's direct interception is booked only where the beam
    clears the anode sheath: three branches, no root-find.

    (a) THE FIXTURE STATE: the first dispatched solve of the anode-sink
    fixture -- a virtual-cathode beam at ``phi_c`` ~ 4.3 V, all of its
    ~496 A emission returned, the loop at zero current -- re-solved under
    ``"emission_fraction"`` books none of the direct term (a 4.3 eV beam
    cannot climb the ~12.1 V sheath the balance without it sets):
    ``phi_a = T_e,a ln(I_e,sat / I_i,a)`` and the collected fraction is 0,
    no raise. (b) The pinned branch: with the term the sheath would sit above
    the beam energy and without it below, so ``phi_a`` equals the beam energy
    and the collected share is strictly between 0 and 1, reproducing it.
    (c) A beam that clears the sheath is booked whole.
    NEGATIVE CONTROL: the same fixture solve under the default booking books
    the whole term, floors the balance and reports the firing.
    """
    from unittest import mock

    from cablp.cathode import circuit_idriven as _ci
    from cablp.cathode.circuit_common import emission_fraction_anode_balance
    from ._harness import _anode_sink_config

    captured = []
    real = _ci.solve_idriven

    def spy(config, plasma, *args, **kwargs):
        if not captured and not kwargs.get("anode_balance_probe", False):
            captured.append((config, plasma, dict(kwargs)))
        return real(config, plasma, *args, **kwargs)

    params, flags = _anode_sink_config()
    with mock.patch.object(_ci, "solve_idriven", spy):
        LAPDSim1D(params, flags).advance_one_step()
    config, plasma, kwargs = captured[0]
    assert kwargs["I_tot_A"] == 0.0 and kwargs["tail_anode_current_A"] == 0.0
    r = real(config, plasma, **dict(
        kwargs, anode_tail_booking="emission_fraction",
    ))
    I_i_a = kwargs["anode_current_A"]
    I_e_sat = kwargs["anode_electron_saturation_A"]
    T_ea = kwargs["anode_T_e"]
    assert r.regime == "virtual_cathode" and 3.5 < r.phi_c < 5.0, r.phi_c
    assert r.I_eth_star > 400.0 and r.beam_bypass_fraction > 0.2
    without = T_ea * math.log(I_e_sat / (I_i_a + r.I_tot))
    assert 11.5 < without < 12.7, without
    assert abs(r.phi_a - without) <= 1e-9 * without, (r.phi_a, without)
    assert r.anode_direct_collected_fraction == 0.0
    # (b) the pinned branch, and (c) the cleared branch.
    I_esat, T = 1000.0, 3.0
    I_rest = I_esat * math.exp(-10.0 / T)  # phi_a = 10 V without the term
    I_direct = 0.5 * I_rest
    phi, f, _ = emission_fraction_anode_balance(
        I_rest, I_direct, I_esat, T, 11.0, False, lambda: ""
    )
    assert phi == 11.0 and 0.0 < f < 1.0, (phi, f)
    back = T * math.log(I_esat / (I_rest - f * I_direct))
    assert abs(back - 11.0) < 1e-12, back
    phi_c, f_c, _ = emission_fraction_anode_balance(
        I_rest, I_direct, I_esat, T, 20.0, False, lambda: ""
    )
    assert f_c == 1.0
    assert abs(phi_c - T * math.log(I_esat / (I_rest - I_direct))) < 1e-12
    # NEGATIVE CONTROL.
    d = real(config, plasma, **kwargs)
    assert d.anode_floor_fired == 1.0 and d.phi_a > 1000.0, d.phi_a


# --------------------------------------------------------------------
# anode-booking-evaluators-plumbed
# --------------------------------------------------------------------
@_case("anode-booking-evaluators-plumbed")
def _case_anode_booking_evaluators_plumbed():
    """The circuit advance and the accepted-state re-solve book the anode as
    the dispatched solve does under ``"emission_fraction"``.

    On a walked-tail half column stepped under ``"emission_fraction"``, every
    sheath solve -- dispatched, circuit-advance and accepted-state -- carries
    the booking and the solver's committed lagged coefficients at the moment
    of the call, and none carries an absolute tail current.
    NEGATIVE CONTROL: under the default booking (the end wall) the evaluators
    keep their standing behaviour -- no booking keywords, a zero tail current
    -- while the dispatched solve carries the lagged one.
    """
    from unittest import mock

    from cablp.cathode import circuit_idriven as _ci
    from cablp.solvers._sim1d.physics import cathode as _pc

    real = _ci.solve_idriven

    def run(params, flags, steps):
        calls = []
        holder = {}

        def spy(config, plasma, *args, **kwargs):
            calls.append((dict(kwargs), holder["lag"]()))
            return real(config, plasma, *args, **kwargs)

        sim = LAPDSim1D(params, flags)
        for _ in range(steps):
            sim.advance_one_step()
        def lag():
            return (sim._cathode_tail_anode_coef,
                    sim._cathode_anode_gap_walker_frac,
                    sim._cathode_primary_return_coef, sim._cathode_tail_anode_I)

        holder["lag"] = lag
        with mock.patch.object(_ci, "solve_idriven", spy), \
                mock.patch.object(_pc, "solve_idriven", spy):
            before = lag()
            sim.advance_one_step()
        return calls, before, lag()

    params, flags = _mirror_circuit_config("mirror")
    calls, coef, _ = run(dict(
        params, heating_anomalous_transport="plateau_multigroup",
        anode_tail_booking="emission_fraction",
    ), flags, 4)
    assert coef[0] > 0.0, coef
    assert len(calls) > 3, len(calls)
    for c, now in calls:
        assert c.get("anode_tail_booking") == "emission_fraction", c
        assert (
            c["tail_anode_coefficient"], c["anode_gap_walker_fraction"],
            c["primary_return_coefficient"],
        ) == now[:3], (c, now)
        assert c.get("tail_anode_current_A", 0.0) == 0.0
    # NEGATIVE CONTROL.
    wall_params, wall_flags = _mirror_circuit_config("end_wall")
    calls, coef, _ = run(dict(
        wall_params, heating_anomalous_transport="plateau_multigroup",
    ), wall_flags, 4)
    assert coef[3] > 0.0, coef
    for c, now in calls:
        assert c.get("tail_anode_current_A", 0.0) in (0.0, now[3]), (c, now)
        assert c.get("anode_tail_booking", "lagged_current") == (
            "lagged_current"
        )
    assert any(c.get("tail_anode_current_A", 0.0) > 0.0 for c, _ in calls)
    assert any(c.get("tail_anode_current_A", 0.0) == 0.0 for c, _ in calls)


def _returning_mirror_ray(phi, **extra):
    """A mirrored walked-tail primary whose returns reach the anode plane.

    ``(args, kwargs)`` for ``deposit_beam``: a 60 eV primary on a dense
    synthetic half column with the plane at cell 5, a reflecting cathode face
    and the wires' sheath at ``phi``. ``extra`` joins the keywords.
    """
    cells = 24
    _nn, _ne, Te, dz = _mirror_column(cells)
    nn = np.full(cells, 3.0e12)
    ne = np.full(cells, 3.0e11) * np.linspace(1.0, 2.0, cells)
    args = (60.0, 1.0e18, nn, ne, Te, 0, 1, dz)
    kwargs = _mirror_mg_kwargs(
        cells, mirror_face=1, tail_reflect_face=-1,
        tail_reflect_threshold_eV=60.0, anode_cross_index=5,
        anode_eta=_MIRROR_ETA, tail_anode_cross_index=5,
        tail_anode_eta=_MIRROR_ETA, tail_anode_phi_eV=phi,
    )
    kwargs.update(extra)
    return args, kwargs


# --------------------------------------------------------------------
# anode-gap-born-outbound-only
# --------------------------------------------------------------------
@_case("anode-gap-born-outbound-only")
def _case_anode_gap_born_outbound_only():
    """Under ``primary_net_basis`` the gap-born walker count is the OUTBOUND
    leg's, so the circuit's direct term and the deposition's agree.

    On a mirrored ray whose returns cross the anode plane into the gap (an
    attracting anode, so the returns are intercepted and their transmitted
    share walks the gap), the circuit's ``eta * (1 - w_gap)`` with
    ``w_gap = tail_gap_born_flux / Gamma0`` equals the deposition's net
    direct interception ``primary_net_direct_flux / Gamma0`` to roundoff,
    while the returns do add births in the gap.
    NEGATIVE CONTROL: the same ray without the net basis counts every walker
    born in the gap, the returns' included, and that count misses the net
    direct interception by far more than roundoff.
    """
    args, kwargs = _returning_mirror_ray(0.0)
    G0 = args[1]
    net = _deposit_beam_ray(*args, primary_net_basis=True,
                            primary_anode_collected_fraction=1.0, **kwargs)
    assert net.primary_net_return_flux_per_s > 0.0
    assert net.primary_net_direct_flux_per_s > 0.0
    w_gap = net.tail_gap_born_flux_per_s / G0
    direct = net.primary_net_direct_flux_per_s / G0
    assert abs(_MIRROR_ETA * (1.0 - w_gap) - direct) <= 1e-15, (
        _MIRROR_ETA * (1.0 - w_gap), direct,
    )
    # NEGATIVE CONTROL.
    gross = _deposit_beam_ray(*args, **kwargs)
    w_all = gross.tail_gap_born_flux_per_s / G0
    assert w_all > w_gap * (1.0 + 1e-3), (w_all, w_gap)
    assert abs(_MIRROR_ETA * (1.0 - w_all) - direct) > 1e-6, (w_all, direct)


# --------------------------------------------------------------------
# anode-booking-consumer-assert
# --------------------------------------------------------------------
@_case("anode-booking-consumer-assert")
def _case_anode_booking_consumer_assert():
    """The solve that APPLIES the lagged coefficients re-asserts the bound at
    its own ``beta``.

    Coefficients the producing deposition admitted at a smaller ``beta``
    (``eta*beta_prod*(1 - w_gap) + c_ret + c_tail <= 1``) exceed 1 at the
    consuming solve's ``beta``, and ``solve_idriven`` raises a
    ``RuntimeError`` naming the total.
    NEGATIVE CONTROL: the same ``w_gap`` and ``c_ret`` with an admissible
    tail coefficient solve at the same ``beta``.
    """
    from cablp.cathode.circuit_idriven import solve_idriven
    from cablp.solvers._sim1d.physics.cathode import (
        anode_tail_booking_coefficients,
    )

    cfg, plasma, I_i_a, I_e_sat = _booking_unit_solve_inputs()
    common = dict(
        anode_current_A=I_i_a, anode_T_e=plasma.T_e,
        anode_electron_saturation_A=I_e_sat, I_tot_A=3000.0,
        anode_tail_booking="emission_fraction",
    )
    beta = solve_idriven(cfg, plasma, **common).beam_bypass_fraction
    assert beta > 0.05, beta
    w_gap, c_ret = 0.37, 0.05
    direct = cfg.eta * beta * (1.0 - w_gap)
    c_tail = 1.0 - c_ret - 0.5 * direct
    # The producer, at half this beta, admits them.
    I_emit = 10.0
    anode_tail_booking_coefficients(
        c_tail * I_emit, I_emit, w_gap * I_emit / qe_SI, cfg.eta, 0.5 * beta,
        c_ret * I_emit / qe_SI,
    )
    try:
        solve_idriven(
            cfg, plasma, tail_anode_coefficient=c_tail,
            anode_gap_walker_fraction=w_gap, primary_return_coefficient=c_ret,
            **common,
        )
    except RuntimeError as exc:
        assert "in the solve that applies it" in str(exc), str(exc)
    else:
        raise AssertionError("consumer-side over-count ACCEPTED")
    # NEGATIVE CONTROL.
    r = solve_idriven(
        cfg, plasma, tail_anode_coefficient=0.21,
        anode_gap_walker_fraction=w_gap, primary_return_coefficient=c_ret,
        **common,
    )
    assert r.beam_bypass_fraction == beta
    assert direct + c_ret + 0.21 <= 1.0


# --------------------------------------------------------------------
# walker-fate-assert-negative-control
# --------------------------------------------------------------------
@_case("walker-fate-assert-negative-control")
def _case_walker_fate_assert_negative_control():
    """The walker-fate assertion fires when the wires' sheath-turned share is
    booked into the anode's kept row.

    On a mirrored walked-tail ray under ``primary_net_basis`` whose wire
    sheath turns part of the walkers back, a wrapped
    ``_tail_mirror_chains`` adds the turned flux to the culled (kept) flux,
    the over-count the assertion exists to catch; ``deposit_beam`` raises
    the fate ``RuntimeError``.
    NEGATIVE CONTROL: the unwrapped ray runs, turns a share back, and its
    kept tail plus leg-cap residual stays within the walkers launched.
    """
    from unittest import mock

    cells = 24
    nn, ne, Te, dz = _mirror_column(cells)
    args = (60.0, 1.0e18, 3.0 * nn, 3.0 * ne, Te, 0, 1, dz)
    kwargs = _mirror_mg_kwargs(
        cells, mirror_face=1, tail_reflect_face=-1,
        tail_reflect_threshold_eV=60.0, anode_cross_index=5,
        anode_eta=_MIRROR_ETA, tail_anode_cross_index=5,
        tail_anode_eta=_MIRROR_ETA, tail_anode_phi_eV=30.0,
        primary_net_basis=True, primary_anode_collected_fraction=1.0,
    )
    real = _beam_deposition_mod._tail_mirror_chains

    def booked_as_kept(*a, **k):
        chains, ledger = real(*a, **k)
        ledger = dict(ledger)
        ledger["culled_flux"] = ledger["culled_flux"] + ledger["sheath_flux"]
        return chains, ledger

    with mock.patch.object(
        _beam_deposition_mod, "_tail_mirror_chains", booked_as_kept
    ):
        try:
            _deposit_beam_ray(*args, **kwargs)
        except RuntimeError as exc:
            assert "the walkers' fates exceed their births" in str(exc), (
                str(exc)
            )
        else:
            raise AssertionError("turned share booked as kept ACCEPTED")
    # NEGATIVE CONTROL.
    r = _deposit_beam_ray(*args, **kwargs)
    assert r.tail_anode_sheath_reflected_flux_per_s > 0.0
    assert (
        r.tail_anode_culled_flux_per_s - r.tail_anode_returned_flux_per_s
        + r.tail_leg_cap_residual_flux_per_s
    ) <= r.tail_launched_flux_per_s



# --------------------------------------------------------------------
# anode-booking-diagnostics-saved
# --------------------------------------------------------------------
@_case("anode-booking-diagnostics-saved")
def _case_anode_booking_diagnostics_saved():
    """Under ``"emission_fraction"`` the saved file carries the anode floor
    census, the fast term's collected fraction and its branch.

    A walked-tail half column run under the booking saves
    ``cathode_diagnostics/anode_floor_dispatched_solves`` (the as-of-save
    count, zero: the balance raises rather than floors),
    ``cathode_diagnostics/source_anode_direct_collected_fraction`` (in
    ``[0, 1]`` on every frame) and
    ``cathode_diagnostics/source_anode_fast_branch`` (0, 1 or 2 on every
    frame), and the run-level attribute ``anode_floor_dispatched_solves``.
    NEGATIVE CONTROL: the same walked tail at the end wall under the default
    booking saves none of the four.
    """
    import tempfile

    import h5py

    def saved(params, flags):
        sim = LAPDSim1D(params, flags)
        result = sim.run(t_end=3.0e-9, dt=1.0e-9)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = sim.save_result(f"{tmpdir}/booking.h5", result)
            with h5py.File(path, "r") as h5:
                diag = h5["cathode_diagnostics"]
                return (
                    {name: diag[name][()] for name in diag
                     if name.endswith("anode_floor_dispatched_solves")
                     or name.endswith("anode_direct_collected_fraction")
                     or name.endswith("anode_fast_branch")},
                    h5.attrs.get("anode_floor_dispatched_solves"),
                )

    params, flags = _mirror_circuit_config("mirror")
    rows, attr = saved(dict(
        params, heating_anomalous_transport="plateau_multigroup",
        anode_tail_booking="emission_fraction",
    ), flags)
    assert set(rows) == {
        "anode_floor_dispatched_solves",
        "source_anode_direct_collected_fraction",
        "source_anode_fast_branch",
    }, sorted(rows)
    assert np.all(rows["anode_floor_dispatched_solves"] == 0.0)
    fraction = rows["source_anode_direct_collected_fraction"]
    assert fraction.size > 1 and np.all((fraction >= 0.0) & (fraction <= 1.0))
    branch = rows["source_anode_fast_branch"]
    assert branch.size == fraction.size and np.all(np.isin(branch, (0, 1, 2)))
    assert attr == 0, attr
    # NEGATIVE CONTROL.
    wall_params, wall_flags = _mirror_circuit_config("end_wall")
    rows, attr = saved(dict(
        wall_params, heating_anomalous_transport="plateau_multigroup",
    ), wall_flags)
    assert rows == {} and attr is None, (sorted(rows), attr)



# --------------------------------------------------------------------
# anode-outbound-primary-sheath-rule
# --------------------------------------------------------------------
@_case("anode-outbound-primary-sheath-rule")
def _case_anode_outbound_primary_sheath_rule():
    """Under the net basis the outbound primary's anode interception follows
    the circuit's sheath rule: the anode collects the fraction the balance
    sets, and the sheath turns the rest back into the gap.

    (a) THE FIXTURE: the anode-sink fixture's first dispatched solve under
    ``"emission_fraction"`` -- a virtual-cathode beam at ``phi_c`` ~ 4.3 V
    against the ~12.1 V sheath the balance sets -- collects none of the
    direct term, and its beam, below the ionization energy, launches no
    deposition ray. (b) That
    verdict on a marching walked-tail ray (300 eV, end wall, no tail cull):
    the anode row is 0, the net direct interception is 0, the turned share is
    in the column and in the transmitted flux, and the ray's power identity
    and the net ledger (births + direct + return + remnant = Gamma0) close to
    roundoff; a pinned fraction books exactly that share of the whole
    interception. (c) The net basis refuses a missing or out-of-range
    fraction, and the fraction without the net basis.
    NEGATIVE CONTROL: the same ray at fraction 1 books the whole
    ``anode_eta * Gamma0 * E`` interception at the plane.
    """
    from unittest import mock

    from cablp.cathode import circuit_idriven as _ci
    from ._harness import _anode_sink_config

    captured = []
    real = _ci.solve_idriven

    def spy(config, plasma, *args, **kwargs):
        if not captured and not kwargs.get("anode_balance_probe", False):
            captured.append((config, plasma, dict(kwargs)))
        return real(config, plasma, *args, **kwargs)

    params, flags = _anode_sink_config()
    with mock.patch.object(_ci, "solve_idriven", spy):
        LAPDSim1D(params, flags).advance_one_step()
    config, plasma, kwargs = captured[0]
    fixture = real(config, plasma, **dict(
        kwargs, anode_tail_booking="emission_fraction",
    ))
    assert 3.5 < fixture.phi_c < 5.0 and 11.5 < fixture.phi_a < 12.7, (
        fixture.phi_c, fixture.phi_a,
    )
    f_sh = fixture.anode_direct_collected_fraction
    assert f_sh == 0.0, f_sh
    cells = 24
    nn, ne, Te, dz = _mirror_column(cells)
    ray_kwargs = _mirror_mg_kwargs(
        cells, anode_cross_index=6, anode_eta=_MIRROR_ETA,
        primary_net_basis=True,
    )
    # The fixture's own beam is below the ionization energy, where the
    # cathode solve launches no deposition ray at all: nothing reaches the
    # anode from it on either side.
    assert fixture.phi_c <= I_ion, fixture.phi_c

    # (b) the verdict on a marching ray.
    args = (300.0, 1.0e18, nn, ne, Te, 0, 1, dz)

    def ray(f):
        res = _deposit_beam_ray(
            *args, primary_anode_collected_fraction=f, **ray_kwargs,
        )
        power = args[0] * args[1] * ev_to_erg
        booked = (
            math.fsum((res.plasma_heating_erg_s + res.radiated_erg_s
                       + res.ionization_cost_erg_s).tolist())
            + res.anode_intercepted_erg_s
            + res.end_loss_low_erg_s + res.end_loss_high_erg_s
            + res.end_loss_tail_low_erg_s + res.end_loss_tail_high_erg_s
            + res.primary_mirror_residual_erg_s
            + res.tail_leg_cap_residual_erg_s
            + res.transmitted_flux * res.transmitted_energy_eV * ev_to_erg
        )
        assert abs(booked - power) <= 1e-12 * power, (f, booked, power)
        total = math.fsum((
            res.primary_births_flux_per_s, res.primary_net_direct_flux_per_s,
            res.primary_net_return_flux_per_s,
            res.primary_net_remnant_flux_per_s,
        ))
        assert abs(total - args[1]) <= 1e-12 * args[1], (f, total)
        return res

    turned, pinned, whole = ray(f_sh), ray(0.5), ray(1.0)
    assert turned.anode_intercepted_erg_s == 0.0
    assert turned.primary_net_direct_flux_per_s == 0.0
    assert turned.transmitted_flux > whole.transmitted_flux
    assert math.fsum(turned.plasma_heating_erg_s.tolist()) > math.fsum(
        whole.plasma_heating_erg_s.tolist()
    )
    assert pinned.anode_intercepted_erg_s == 0.5 * whole.anode_intercepted_erg_s
    assert pinned.primary_net_direct_flux_per_s == (
        0.5 * whole.primary_net_direct_flux_per_s
    )
    # (c) the refusals.
    for extra, needle in (
        ({}, "give primary_anode_collected_fraction"),
        ({"primary_anode_collected_fraction": 1.5}, "must be in [0, 1]"),
        ({"primary_anode_collected_fraction": float("nan")},
         "must be in [0, 1]"),
    ):
        try:
            _deposit_beam_ray(*args, **ray_kwargs, **extra)
        except ValueError as exc:
            assert needle in str(exc), str(exc)
        else:
            raise AssertionError(f"{extra} ACCEPTED")
    gross_kwargs = dict(ray_kwargs, primary_net_basis=False)
    try:
        _deposit_beam_ray(
            *args, primary_anode_collected_fraction=0.0, **gross_kwargs,
        )
    except ValueError as exc:
        assert "belongs to primary_net_basis" in str(exc), str(exc)
    else:
        raise AssertionError("fraction without the net basis ACCEPTED")
    # NEGATIVE CONTROL.
    E_cross = float(whole.E_entry_eV[6])
    assert E_cross > 0.0
    want = _MIRROR_ETA * args[1] * E_cross * ev_to_erg
    assert abs(whole.anode_intercepted_erg_s - want) <= 1e-12 * want, (
        whole.anode_intercepted_erg_s, want,
    )
    assert whole.primary_net_direct_flux_per_s > 0.0


# --------------------------------------------------------------------
# anode-fast-term-sheath-rule
# --------------------------------------------------------------------
@_case("anode-fast-term-sheath-rule")
def _case_anode_fast_term_sheath_rule():
    """Under ``"emission_fraction"`` the whole fast term the anode books
    (direct, return and tail together) passes one sheath rule at the
    applying solve's own ``phi_c``.

    (a) A solve whose sheath without the term sits at or above ``phi_c``
    books none of it: branch NONE, collected fraction 0,
    ``phi_a = T_e,a ln(I_e,sat / (I_i,a + I_tot)) >= phi_c`` and no tail
    sheath-fall power. (b) THE RAISING STATE, synthetic: a virtual-cathode
    solve at ``phi_c`` ~ 81 V whose emission exceeds the loop current, with
    ``c_tail = 0.88`` and ``w_gap = 0.977``, takes the pinned branch:
    ``phi_a = phi_c``, the fraction in ``[0, 1)`` and a positive current
    ``I_i,a + I_tot - f (eta beta (1 - w_gap) + c_tail) I_star`` that
    reproduces ``phi_a``. (c) The recorded step-18 circuit-advance solve of
    the half-column twin (the balance's own inputs: ``I_i,a`` 8.7313 A,
    ``I_tot`` 129.1288 A, ``I_star`` 165.5871 A, ``c_direct`` 8.3214e-9,
    ``c_tail`` 0.88073, ``phi_c`` 80.9525 V, ``T_e,a`` 37.8942 eV,
    ``I_e,sat`` 490.556 A) replayed through the balance returns the pinned
    branch with a positive sheath current.
    NEGATIVE CONTROL: (b) and (c) under the pre-change split -- the return
    and tail subtracted before the rule, the rule applied to the direct term
    alone -- raise the infeasible-balance error.
    """
    from cablp.cathode import circuit_common as _cc
    from cablp.cathode.circuit_idriven import solve_idriven

    cfg, _, _, _ = _booking_unit_solve_inputs()
    T_ea, I_esat, I_ia = 37.89, 490.56, 8.73
    common = dict(
        anode_current_A=I_ia, anode_T_e=T_ea,
        anode_electron_saturation_A=I_esat,
        anode_tail_booking="emission_fraction",
        tail_anode_coefficient=0.88, anode_gap_walker_fraction=0.977,
    )
    # (a) no booking.
    r = solve_idriven(
        cfg, _cc.PlasmaState(T_e=38.75, n_e=2.12e10, n_n=0.0, sigma_b=0.0),
        I_tot_A=50.0, **common,
    )
    assert r.anode_fast_branch == _cc.ANODE_FAST_BRANCH_NONE
    assert r.anode_direct_collected_fraction == 0.0
    without = T_ea * math.log(I_esat / (I_ia + r.I_tot))
    assert abs(r.phi_a - without) <= 1e-12 * without, (r.phi_a, without)
    assert r.phi_a >= r.phi_c, (r.phi_a, r.phi_c)
    assert r.P_tail_phi == 0.0, r.P_tail_phi
    # (b) the raising state.
    r = solve_idriven(
        cfg, _cc.PlasmaState(T_e=38.75, n_e=3.7e10, n_n=0.0, sigma_b=0.0),
        I_tot_A=129.13, **common,
    )
    assert r.regime == "virtual_cathode" and 80.0 < r.phi_c < 82.0, (
        r.regime, r.phi_c,
    )
    assert r.I_eth_star > r.I_tot, (r.I_eth_star, r.I_tot)
    assert r.anode_fast_branch == _cc.ANODE_FAST_BRANCH_PINNED
    f = r.anode_direct_collected_fraction
    assert 0.0 <= f < 1.0 and r.phi_a == r.phi_c, (f, r.phi_a, r.phi_c)
    c_fast = cfg.eta * r.beam_bypass_fraction * (1.0 - 0.977) + 0.88
    I_e_a = I_ia + r.I_tot - f * c_fast * r.I_eth_star
    assert I_e_a > 0.0, I_e_a
    back = T_ea * math.log(I_esat / I_e_a)
    assert abs(back - r.phi_a) <= 1e-9 * r.phi_a, (back, r.phi_a)
    assert abs(
        r.P_tail_phi - r.phi_a * f * 0.88 * r.I_eth_star
    ) <= 1e-12 * r.P_tail_phi
    # (c) the recorded step-18 advance solve, through the balance.
    I_ia18, I_tot18, I_star18 = 8.731328467152306, 129.12876566786295, (
        165.58705846843623
    )
    c_dir18, c_tail18 = 8.321426379939418e-09, 0.8807342998848132
    phi_c18, T_ea18, I_esat18 = 80.95249107818451, 37.89423544883266, (
        490.5564117505911
    )
    phi18, f18, branch18 = _cc.emission_fraction_anode_balance(
        I_ia18 + I_tot18, (c_dir18 + c_tail18) * I_star18, I_esat18, T_ea18,
        phi_c18, False, lambda: "step 18",
    )
    assert branch18 == _cc.ANODE_FAST_BRANCH_PINNED and phi18 == phi_c18
    assert 0.0 <= f18 < 1.0, f18
    assert I_ia18 + I_tot18 - f18 * (c_dir18 + c_tail18) * I_star18 > 0.0
    # NEGATIVE CONTROL: the pre-change split.
    for rest, direct, E, T, sat in (
        (I_ia + r.I_tot - 0.88 * r.I_eth_star,
         cfg.eta * r.beam_bypass_fraction * (1.0 - 0.977) * r.I_eth_star,
         r.phi_c, T_ea, I_esat),
        (I_ia18 + I_tot18 - c_tail18 * I_star18, c_dir18 * I_star18,
         phi_c18, T_ea18, I_esat18),
    ):
        try:
            _cc.emission_fraction_anode_balance(
                rest, direct, sat, T, E, False, lambda: "pre-change split"
            )
        except ValueError as exc:
            assert "anode sheath balance is infeasible" in str(exc), str(exc)
        else:
            raise AssertionError(
                f"the pre-change split RETURNED at I_rest={rest!r} A"
            )


# --------------------------------------------------------------------
# anode-pinned-share-knife-edge
# --------------------------------------------------------------------
@_case("anode-pinned-share-knife-edge")
def _case_anode_pinned_share_knife_edge():
    """The pinned branch returns its collected share within ``[0, 1]`` at
    the knife-edge, and a gross excess raises.

    (a) With ``E_beam`` equal to ``phi_a`` with the term exactly, and one ulp
    either side of it, over a sweep of balances: every share is in
    ``[0, 1]``, the ulp above is booked whole, and at least one exact
    knife-edge share, which the raw balance puts above 1 by roundoff, is
    clamped to exactly 1.0. (b) A share already inside ``[0, 1]`` is the raw
    balance, bit for bit. (c) An ill-conditioned balance (the fast term
    3e-12 of the current the sheath passes) at the knife-edge puts the raw
    share ~2.7e-4 above 1, beyond the clamp: it raises ``ValueError``.
    NEGATIVE CONTROL: the raw balance itself exceeds 1 at the swept
    knife-edges and at (c), so the clamp and the raise are exercised.
    """
    import sys as _sys

    from cablp.cathode.circuit_common import (
        ANODE_FAST_BRANCH_BOOKED,
        ANODE_FAST_BRANCH_PINNED,
        emission_fraction_anode_balance,
    )

    eps = _sys.float_info.epsilon
    I_esat, T = 1000.0, 3.0

    def raw(I_rest, I_fast, E):
        return (I_rest - I_esat * math.exp(-E / T)) / I_fast

    clamped = 0
    for k in range(1, 401):
        I_rest = I_esat * math.exp(-10.0 / T) * (1.0 + k * 1e-5)
        I_fast = 0.5 * I_rest
        E = T * math.log(I_esat / (I_rest - I_fast))
        up, down = math.nextafter(E, math.inf), math.nextafter(E, -math.inf)
        _, f_up, b_up = emission_fraction_anode_balance(
            I_rest, I_fast, I_esat, T, up, False, lambda: ""
        )
        assert b_up == ANODE_FAST_BRANCH_BOOKED and f_up == 1.0, (k, f_up)
        for E_beam in (E, down):
            phi, f, b = emission_fraction_anode_balance(
                I_rest, I_fast, I_esat, T, E_beam, False, lambda: ""
            )
            assert b == ANODE_FAST_BRANCH_PINNED and phi == E_beam, (k, b)
            assert 0.0 <= f <= 1.0, (k, E_beam, f)
            r = raw(I_rest, I_fast, E_beam)
            if 0.0 <= r <= 1.0:
                assert f == r, (k, f, r)
            else:
                assert abs(r - f) <= 64.0 * eps, (k, f, r)
                clamped += 1
    # NEGATIVE CONTROL: roundoff did put raw shares outside [0, 1].
    assert clamped > 0, clamped
    # (c) the ill-conditioned knife-edge.
    I_rest, I_fast = 1.0, 3e-12
    E = T * math.log(I_esat / (I_rest - I_fast))
    r = raw(I_rest, I_fast, E)
    assert r - 1.0 > 1e-4, r
    try:
        emission_fraction_anode_balance(
            I_rest, I_fast, I_esat, T, E, False, lambda: "fixture"
        )
    except ValueError as exc:
        assert "outside [0, 1]" in str(exc), str(exc)
    else:
        raise AssertionError(f"a share of {r!r} did not raise")
