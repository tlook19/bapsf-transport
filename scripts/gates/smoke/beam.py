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
# beam-excitation-channel
# --------------------------------------------------------------------
@_case(
    "beam-excitation-channel",
    historical_stance=True,
    provides=(
        "beam_excitation_cross", "exc_beam", "exc_params", "launch_idx",
    ),
)
def _case_beam_excitation_channel(cathode_solve):
    # --- Beam excitation channel (b_beam_excitation, default 0 = historical).
    params, flags = _base_config()
    cathode_flags = _cathode_flags()
    from cablp.cathode.circuit_common import beam_excitation_cross

    sigma_exc_100 = beam_excitation_cross(100.0, 1.0)
    assert 5.0e-18 < sigma_exc_100 < 2.0e-17
    assert beam_excitation_cross(100.0, 0.0) == 0.0
    assert beam_excitation_cross(10.0, 1.0) == 0.0  # below threshold

    # The b_beam_excitation knob scales the sheath solve's excitation
    # channel.
    exc_params = dict(params)
    exc_params["b_beam_excitation"] = 1.0
    exc_sim = LAPDSim1D(exc_params, cathode_flags)
    exc_sim._circuit_I_loop = 3000.0
    exc_solve = exc_sim.solve_cathode_boundary()
    exc_beam = exc_solve.beam_result
    base_beam = cathode_solve.beam_result
    launch_idx = int(np.flatnonzero(exc_beam.beam_cross)[0])
    assert exc_beam.beam_exc_cross[launch_idx] > 0.0
    # Both first solves run from a zeroed sigma_b cache, so the circuit state
    # is identical and the only difference is the attenuation cross section:
    # the inelastic deposition length must be strictly shorter everywhere.
    assert np.isclose(
        exc_solve.beam_result.result.phi_c, base_beam.result.phi_c
    )
    positive = base_beam.l_b_profile > 0.0
    assert np.all(
        exc_beam.l_b_profile[positive] < base_beam.l_b_profile[positive]
    )
    exc_terms = exc_sim.beam_ionization_rhs_terms(cathode_solve=exc_solve)
    exc_rad = exc_terms["beam_excitation_radiation"]
    assert np.all(exc_rad.Ee <= 0.0)
    assert np.any(exc_rad.Ee < 0.0)
    for field_values in (exc_rad.n, exc_rad.nn, exc_rad.M, exc_rad.Ei):
        assert np.allclose(field_values, 0.0)
    # The 2^1P channel reports the constant threshold as its per-event
    # energy.
    assert float(exc_beam.beam_exc_energy_eV[launch_idx]) == 21.218
    return locals()


# --------------------------------------------------------------------
# beam-manifold-excitation-model
# --------------------------------------------------------------------
@_case(
    "beam-manifold-excitation-model",
    historical_stance=True,
)
def _case_beam_manifold_excitation_model(beam_excitation_cross):
    # --- A2: the manifold excitation channel (WP-A).
    from cablp.cathode.circuit_common import beam_excitation_channel
    from cablp.atomic.cross_sections import (
        He_beam_excitation_channel as _He_manifold_channel,
    )

    # Dispatch: the scalar path reproduces the historical function
    # byte-for-byte; the manifold path matches the _cross helper with
    # b_beam_excitation as a pure multiplier on the cross section only.
    assert beam_excitation_channel(100.0, 1.4) == (
        beam_excitation_cross(100.0, 1.4),
        21.218,
    )
    _mf_sigma, _mf_E = beam_excitation_channel(100.0, 1.0, model="manifold")
    assert (_mf_sigma, _mf_E) == _He_manifold_channel(100.0)
    _mf_sigma_h, _mf_E_h = beam_excitation_channel(
        100.0, 0.5, model="manifold"
    )
    assert np.isclose(_mf_sigma_h, 0.5 * _mf_sigma) and _mf_E_h == _mf_E
    assert beam_excitation_channel(100.0, 0.0, model="manifold") == (0.0, 0.0)
    # Below the lowest manifold threshold (2^1S, 20.6158 eV).
    assert beam_excitation_channel(15.0, 1.0, model="manifold") == (0.0, 0.0)
    # The measured manifold vs the historical 2^1P channel at 100 eV
    # (measure_beam_manifold.py, 2026-07-20): 1.67x the events, mean
    # radiated energy 21.98 eV — within the retired estimate's 1.4 +- 0.4.
    assert 1.55 < _mf_sigma / beam_excitation_cross(100.0, 1.0) < 1.80
    assert 21.5 < _mf_E < 22.5
    for bad_call in (
        lambda: beam_excitation_channel(100.0, 1.0, model="bogus"),
    ):
        try:
            bad_call()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError from excitation channel")

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
def _case_beam_csda_deposition_model(exc_params):
    # --- B2: the CSDA deposition module wired into the cathode solve.
    sim, snapshot = _base_sim()
    geom = snapshot.geometry
    cathode_flags = _cathode_flags()
    csda_params = dict(exc_params)
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
def _case_beam_gap_ledger_tripwire(csda_sim, csda_solve, exc_params):
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
        dict(exc_params), dict(cathode_flags)
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
    # Checked on BOTH a uniform and a non-uniform (source_fixed_grid) mesh: the
    # kernel is weighted by cell length, and without that weighting a refined
    # region is over-weighted per cm, which makes the smoothing operator itself
    # mesh-dependent even where it happens to conserve.
    cathode_flags = _cathode_flags()
    smooth_sigma_cm = 50.0
    smoothing_meshes = (
        ("uniform", dict(csda_params), dict(cathode_flags)),
        (
            "source_fixed_grid",
            {
                **csda_params,
                # Gap pinned with the region: see _case_source_fixed_grid.
                "cathode_anode_gap_cm": 50.0,
                "source_region_length_cm": 100.0,
                "source_region_dz_cm": 10.0,
                "gas_puff_z_cm": 60.0,
            },
            {**cathode_flags, "source_fixed_grid": True},
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
        if mesh_label == "source_fixed_grid":
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
    smoothkey_flags = {**cathode_flags, "source_fixed_grid": True}
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

    S_ion_a, S_rad_a, S_3b_a = reaction_rates(
        state=knob_state,
        floors=knob_floors,
        ion_mass_g=knob_mass,
    )
    for values in (S_ion_a, S_rad_a):
        assert np.all(np.isfinite(values)) and np.all(values >= 0.0)
    # ACD carries the whole sink; the three-body slot is empty.
    assert np.all(S_3b_a == 0.0)

    cool_adas = electron_cooling_rhs(**cooling_kwargs)
    assert np.all(np.isfinite(cool_adas.Ee))
    assert np.all(cool_adas.Ee <= 0.0)
    # The cooling path's fused ionization cost must be bit-identical to
    # I_ion * S_ion from reaction_rates -- the cost charges exactly the
    # particles the particle equation creates.
    cost_kwargs = dict(cooling_kwargs)
    cost_kwargs["ionization_energy_cost"] = True
    cost_terms = electron_cooling_rhs_terms(**cost_kwargs)
    S_ion_ref, _, _ = reaction_rates(
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
