"""Smoke cases: the regime tracer."""

import numpy as np

from cablp.solvers._sim1d import LAPDSim1D, default_config
from cablp.solvers._sim1d.physics.cathode import beam_ionization_rhs_terms

from ._harness import (
    _case,
    _pin_pre_r2a_neutral_stance,
    _tracking_electrode_sample,
)


# --------------------------------------------------------------------
# tracer-affine-update-identity
# --------------------------------------------------------------------
@_case(
    "tracer-affine-update-identity",
    provides=("_r2", "solver_module"),
)
def _case_tracer_affine_update_identity():
    # ==================================================================
    # REGIME-R2 PRE-BREAKDOWN PASSIVE TRACER (default off, bit-exact off)
    #
    # Four blocks, matching the pre-registered gate: (i) affine-update
    # exactness against the closed form, (ii) presence gating, (iii) every
    # construction-time refusal, (iv) ANTI-VACUITY -- each guard is shown to
    # FAIL on a deliberately broken variant, because a guard that cannot fire
    # is not a guard.
    # ==================================================================
    from cablp.solvers._sim1d import solver as solver_module
    from cablp.solvers._sim1d.physics import tracer as _r2

    # ---- (i) the affine update IS the closed-form solution ----
    # Two cells, one growing and one decaying, plus the two removable
    # singularities the expm1 form exists to handle.
    _r2_n0 = np.array([3.0e9, 7.0e11])
    _r2_gamma = np.array([2.5e4, -1.1e5])
    _r2_S = np.array([4.0e13, 9.0e12])
    _r2_dt = 3.7e-6
    _r2_closed = (
        (_r2_n0 + _r2_S / _r2_gamma) * np.exp(_r2_gamma * _r2_dt)
        - _r2_S / _r2_gamma
    )
    _r2_got = _r2.affine_update(_r2_n0, _r2_gamma, _r2_S, _r2_dt)
    assert np.max(np.abs(_r2_got / _r2_closed - 1.0)) < 1e-14, (
        "R2 affine update must reproduce the closed-form solution",
        _r2_got,
        _r2_closed,
    )
    # gamma -> 0 is exact, not merely close: the update degenerates to n + S dt.
    assert _r2.affine_update(_r2_n0, np.zeros(2), _r2_S, _r2_dt).tolist() == (
        _r2_n0 + _r2_S * _r2_dt
    ).tolist(), "R2 affine update at gamma = 0 must be exactly n + S*dt"
    # ... and stays exact for a gamma so small that exp(x) - 1 would cancel to
    # nothing. This is the whole reason the expm1 form is used.
    _r2_tiny = np.full(2, 1.0e-13)
    assert np.max(
        np.abs(
            _r2.affine_update(_r2_n0, _r2_tiny, _r2_S, _r2_dt)
            / (_r2_n0 + _r2_S * _r2_dt)
            - 1.0
        )
    ) < 1e-12, "R2 affine update must stay regular as gamma -> 0"
    # n = 0 is a REGULAR state: a true-vacuum cell fills from the source alone.
    _r2_vac = _r2.affine_update(np.zeros(2), _r2_gamma, _r2_S, _r2_dt)
    assert np.allclose(
        _r2_vac,
        _r2_S * np.expm1(_r2_gamma * _r2_dt) / _r2_gamma,
        rtol=1e-14,
        atol=0.0,
    ), "R2 affine update must run from a true-vacuum initial condition"
    assert np.all(np.isfinite(_r2_vac)) and np.all(_r2_vac > 0.0)
    # A decaying cell relaxes onto the exact equilibrium -S/gamma, with no
    # floor race and no overshoot.
    _r2_eq = _r2.affine_update(1.0e14, -1.0e7, 1.0e9, 1.0)
    assert abs(float(_r2_eq) / (1.0e9 / 1.0e7) - 1.0) < 1e-14
    # The time integral is the closed form too (it feeds criterion (c)).
    _r2_int_closed = (
        (_r2_n0 + _r2_S / _r2_gamma)
        * np.expm1(_r2_gamma * _r2_dt)
        / _r2_gamma
        - _r2_S / _r2_gamma * _r2_dt
    )
    assert np.max(
        np.abs(
            _r2.affine_time_integral(_r2_n0, _r2_gamma, _r2_S, _r2_dt)
            / _r2_int_closed
            - 1.0
        )
    ) < 1e-12
    assert np.allclose(
        _r2.affine_time_integral(_r2_n0, np.zeros(2), _r2_S, _r2_dt),
        _r2_n0 * _r2_dt + 0.5 * _r2_S * _r2_dt * _r2_dt,
        rtol=1e-14,
        atol=0.0,
    )
    # ANTI-VACUITY for (i): the tolerance above must REJECT the naive
    # implementation (n*exp(x) + S*dt), which is the mistake the expm1 form is
    # there to avoid. If this passed, the exactness assertions would be empty.
    _r2_naive = _r2_n0 * np.exp(_r2_gamma * _r2_dt) + _r2_S * _r2_dt
    assert np.max(np.abs(_r2_naive / _r2_closed - 1.0)) > 1e-14, (
        "R2 exactness assertions are vacuous: the naive update passes them too"
    )
    return locals()


# --------------------------------------------------------------------
# tracer-fluid-n-row-identity
# --------------------------------------------------------------------
@_case("tracer-fluid-n-row-identity")
def _case_tracer_fluid_n_row_identity(_r2):
    # ---- (i, continued) gamma*n + S IS the fluid's own n row ----
    # The tracer claims to duplicate no physics: it recovers each channel's
    # coefficient from the solver's own term function by homogeneity degree.
    # That claim is checkable, and this is the check.
    _r2_id_p, _r2_id_f = default_config()
    _r2_id_p["nx"] = 12
    _r2_id_p["ne0"] = 5.0e10  # above ne_floor, so the probe ratio is exactly 1
    _r2_id_p["initial_neutral_state"] = "fill"
    _r2_id_f["cathode_coupling"] = False
    _r2_id_sim = LAPDSim1D(_r2_id_p, _r2_id_f)
    _r2_id_state = _r2_id_sim.state
    _r2_id_n = np.asarray(_r2_id_state.n, dtype=float)
    _r2_id_Te = np.asarray(_r2_id_sim.derived.Te, dtype=float)
    _r2_id_Ti = np.asarray(_r2_id_sim.derived.Ti, dtype=float)
    _r2_id_gamma = _r2.growth_rate(
        state=_r2_id_state,
        n_true=_r2_id_n,
        n_probe=np.maximum(_r2_id_n, _r2_id_sim.floors["n"]),
        Te_eV=_r2_id_Te,
        Ti_eV=_r2_id_Ti,
        floors=_r2_id_sim.floors,
        ion_mass_g=_r2_id_sim.ion_mass_g,
        reaction_kwargs=_r2_id_sim._tracer_reaction_kwargs(),
        boundary_rhs=_r2_id_sim._tracer_boundary_rhs(None, 0.0),
    )
    _r2_id_terms = _r2_id_sim.rhs_terms(include_heat_conduction=False)
    # The four n-row channels gamma is built from. Both boundary spellings are
    # summed because exactly one of them is live: boundary_absorption is
    # identically zero everywhere since the legacy absorber was retired
    # (see commit 1fc05c9), and the row is kept only for saved-ledger schema
    # stability. Summing both is what keeps this identity readable against
    # artifacts written on either side of that retirement.
    _r2_id_fluid = sum(
        np.asarray(_r2_id_terms[name].n, dtype=float)
        for name in (
            "ionization_birth",
            "recombination_rad_loss",
            "recombination_3b_loss",
            "boundary_absorption",
            "characteristic_boundary",
        )
    )
    # Plasma-dead cells are masked out of the fluid rows by the typed topology
    # and have no tracer either, so the identity is asserted where plasma lives.
    _r2_id_live = np.asarray(_r2_id_sim.geometry.plasma_active, dtype=bool)
    _r2_id_scale = np.maximum(np.abs(_r2_id_fluid), 1.0)
    _r2_id_err = np.max(
        (np.abs(_r2_id_gamma * _r2_id_n - _r2_id_fluid) / _r2_id_scale)[
            _r2_id_live
        ]
    )
    assert _r2_id_err < 1e-10, (
        "R2 gamma must reproduce the fluid's own n row (no duplicated physics)",
        _r2_id_err,
    )
    # ANTI-VACUITY: a 1e-7 relative error in gamma -- far smaller than picking
    # the wrong boundary discretization or mis-scaling a homogeneity degree --
    # must break it. Without this the tolerance could be vacuously loose.
    assert np.max(
        (
            np.abs(_r2_id_gamma * 1.0000001 * _r2_id_n - _r2_id_fluid)
            / _r2_id_scale
        )[_r2_id_live]
    ) > 1e-10, "R2 gamma identity is vacuous: a 1e-7 error passes it"


# --------------------------------------------------------------------
# tracer-presence-gating
# --------------------------------------------------------------------
@_case(
    "tracer-presence-gating",
    provides=("_r2_on_config",),
)
def _case_tracer_presence_gating():
    # ---- (ii) PRESENCE GATING: the off path cannot read the tracer keys ----
    # Sweeping every registered criterion constant by 3x and 1/3 must leave the
    # flag-off trajectory raw-byte identical. If the off path touched any of
    # them, this moves.
    def _r2_off_bytes(scale):
        params, flags = default_config()
        params["nx"] = 12
        params["initial_neutral_state"] = "fill"
        flags["cathode_coupling"] = False
        for key in (
            "tracer_passivity_current_ratio",
            "tracer_passivity_thinness",
            "tracer_passivity_depletion",
            "tracer_refresh_tol",
        ):
            params[key] = params[key] * scale
        params["tracer_passivity_hysteresis"] = 1.0 + (
            params["tracer_passivity_hysteresis"] - 1.0
        ) * scale
        params["tracer_activation_ne"] = params["tracer_activation_ne"] * scale
        params["tracer_overlap_rtol"] = params["tracer_overlap_rtol"] * scale
        sim_off = LAPDSim1D(params, flags)
        assert sim_off._tracer is None, (
            "regime_tracer off must build no tracer object"
        )
        # The off path must also hand back the BASE geometry object itself, not
        # a copy: a view would be a branch the golden could see.
        assert sim_off._plasma_geometry() is sim_off._geometry
        result = sim_off.run(t_end=4.0e-10, dt=1.0e-10)
        return np.asarray(result.n, dtype=float).tobytes()

    _r2_off_ref = _r2_off_bytes(1.0)
    assert _r2_off_bytes(3.0) == _r2_off_ref, (
        "regime_tracer OFF must not read tracer_* constants (3x sweep moved it)"
    )
    assert _r2_off_bytes(1.0 / 3.0) == _r2_off_ref, (
        "regime_tracer OFF must not read tracer_* constants (1/3 sweep moved it)"
    )

    def _r2_on_config(**overrides):
        params, flags = default_config()
        params["nx"] = 12
        params["cathode_solver_model"] = "current_driven"
        params["initial_neutral_state"] = "fill"
        flags["cathode_coupling"] = True
        flags["regime_tracer"] = True
        # The refusal table below arms one offending key at a time and asserts
        # WHICH refusal fires, so the base must not carry a second conflict of
        # its own (neutral_energy refuses the kinetic neutral models).
        _pin_pre_r2a_neutral_stance(params, flags)
        params.update(overrides)
        return params, flags

    # ANTI-VACUITY for (ii): with the flag ON the same sweep MUST move the run.
    # Otherwise the byte-identity above would be satisfied by a feature that
    # reads nothing at all, and would prove nothing about gating.
    def _r2_on_bytes(activation_ne):
        params, flags = _r2_on_config(tracer_activation_ne=activation_ne)
        sim_on = LAPDSim1D(params, flags)
        assert sim_on._tracer is not None
        return np.asarray(
            sim_on.run(t_end=6.0e-8, dt=2.0e-8).n, dtype=float
        ).tobytes()

    assert _r2_on_bytes(1.0e10) != _r2_off_ref, (
        "regime_tracer ON must change the trajectory"
    )

    # The ON path owns the cells it claims: the fluid's active mask excludes
    # them, the geometry view closes the interface, and the density floor is
    # skipped there so n = 0 stays exactly 0.
    _r2_on_params, _r2_on_flags = _r2_on_config()
    _r2_on_sim = LAPDSim1D(_r2_on_params, _r2_on_flags)
    assert bool(np.any(_r2_on_sim._tracer_passive))
    assert not np.any(
        _r2_on_sim._plasma_active_mask() & _r2_on_sim._tracer_passive
    ), "the fluid must not own a cell the tracer owns"
    _r2_on_view = _r2_on_sim._plasma_geometry()
    assert _r2_on_view is not _r2_on_sim._geometry
    _r2_on_dead = ~_r2_on_sim._plasma_active_mask()
    for _face in range(1, _r2_on_sim.geometry.cells):
        if _r2_on_dead[_face - 1] != _r2_on_dead[_face]:
            assert not _r2_on_view.plasma_open[_face], (
                "a passive/active interface face must be closed"
            )
            assert _r2_on_view.plasma_transmission[_face] == 0.0
            assert _r2_on_view.heat_transmission[_face] == 0.0
    # A closed face has at most one live cell, so each cell has exactly one
    # owner -- the property the closed-face treatment was chosen for.
    for _face in np.flatnonzero(~np.asarray(_r2_on_view.plasma_open, dtype=bool)):
        _live = int(_r2_on_view.plasma_face_live_cell[int(_face)])
        assert _live < 0 or not _r2_on_sim._tracer_passive[_live]

    # ne0 = 0 is a legal initial condition, and the floor leaves it alone.
    _r2_vac_params, _r2_vac_flags = _r2_on_config(ne0=0.0)
    _r2_vac_sim = LAPDSim1D(_r2_vac_params, _r2_vac_flags)
    # Exactly the tracer's own cells hold a true vacuum. The plasma-DEAD cells
    # are not the tracer's and keep their ordinary floor, which is why this is
    # asserted on the passive mask rather than on the whole grid.
    assert float(
        np.max(np.asarray(_r2_vac_sim.state.n)[_r2_vac_sim._tracer_passive])
    ) == 0.0, "ne0 = 0 must survive construction under the tracer"
    _r2_vac_floored = _r2_vac_sim.floor_state_vector(_r2_vac_sim._y)
    _r2_vac_n = _r2_vac_floored[: _r2_vac_sim.geometry.cells]
    assert float(np.max(_r2_vac_n[_r2_vac_sim._tracer_passive])) == 0.0, (
        "the density floor must skip tracer cells; n = 0 is a regular state"
    )
    # ANTI-VACUITY: the very same vector floored with the tracer disengaged IS
    # clipped, so the exemption above is doing something.
    _r2_vac_sim._tracer_passive = np.zeros_like(_r2_vac_sim._tracer_passive)
    assert float(
        np.min(_r2_vac_sim.floor_state_vector(_r2_vac_sim._y)[
            : _r2_vac_sim.geometry.cells
        ])
    ) == _r2_vac_sim.floors["n"], (
        "the floor-exemption assertion is vacuous: nothing was being clipped"
    )
    return locals()


# --------------------------------------------------------------------
# tracer-passive-anomalous-leak-phase-gated-solve
# --------------------------------------------------------------------
@_case("tracer-passive-anomalous-leak-phase-gated-solve", provides=())
def _case_tracer_passive_anomalous_leak_phase_gated_solve():
    """``tracer_passive_anomalous_leak`` must not dispatch off-phase.

    Its sibling ``_tracer_prepare`` gates its own re-solve on
    ``self._flags.get("cathode_coupling")`` before calling
    ``solve_cathode_boundary``; ``tracer_passive_anomalous_leak`` called it
    unconditionally when ``self._cathode_solve is None``. Fixed to build
    ``cathode_flags = self._effective_cathode_flags(time=time,
    active_only=True)`` first and dispatch only when
    ``cathode_flags["cathode_coupling"]`` is True -- the same pattern
    ``_jet_cathode_solve`` (:10411) and ``cathode_source_terms`` (:10980) use.

    NOTE ON THE RECONSTRUCTED FAILURE MODE. In a ``neutral_prebreakdown``
    phase (``tau_neutral_prebreakdown`` > 0, the tracer engaged) the
    unconditional pre-fix call does NOT raise:
    ``cathode_boundary_state.enabled`` (``physics/cathode.py``) reads the
    SAME phase-aware ``cathode_coupling`` the gate above reads, and
    ``solve_cathode_boundary``'s module function returns a disabled,
    no-op ``CathodeSolve1D`` -- measured directly below. So the pre-fix
    call was silently WASTEFUL in this phase rather than a hard refusal;
    the fix is for phase-consistency with the other solve sites. What IS
    tested, both ways: the fixed method dispatches a solve only in a phase
    that has one.
    """
    def _tpal_build():
        params, flags = default_config()
        params = dict(params)
        flags = dict(flags)
        params["nx"] = 16
        params["cathode_solver_model"] = "current_driven"
        params["initial_neutral_state"] = "fill"
        flags["cathode_coupling"] = True
        flags["regime_tracer"] = True
        params["phase_transition_mode"] = "scheduled"
        params["tau_neutral_prebreakdown"] = 1.0e-6
        params["tau_prebreakdown"] = 1.0e-6
        params["tau_breakdown"] = 0.0
        params["tau_discharge"] = 1.0e-6
        params["tau_afterglow"] = 1.0e-6
        _pin_pre_r2a_neutral_stance(params, flags)
        return params, flags

    _tpal_params, _tpal_flags = _tpal_build()
    _tpal_sim = LAPDSim1D(_tpal_params, _tpal_flags)
    assert _tpal_sim._tracer_engaged

    _tpal_prebreak_t = 0.5 * float(_tpal_params["tau_neutral_prebreakdown"])
    assert _tpal_sim.phase_at_time(_tpal_prebreak_t) == "neutral_prebreakdown"
    _tpal_discharge_t = (
        float(_tpal_params["tau_neutral_prebreakdown"])
        + float(_tpal_params["tau_prebreakdown"])
        + float(_tpal_params["tau_breakdown"])
        + 0.5 * float(_tpal_params["tau_discharge"])
    )
    assert _tpal_sim.phase_at_time(_tpal_discharge_t) == "main_discharge"

    def _tpal_call_with_spy(time):
        calls = []
        orig = _tpal_sim.solve_cathode_boundary

        def _tpal_spy(**kw):
            calls.append(kw)
            return orig(**kw)

        _tpal_sim.solve_cathode_boundary = _tpal_spy
        try:
            out = _tpal_sim.tracer_passive_anomalous_leak(time=time)
        finally:
            _tpal_sim.solve_cathode_boundary = orig
        return out, calls

    # (i) neutral_prebreakdown has no cathode solve: the fix must not
    # dispatch one.
    _tpal_out_pre, _tpal_calls_pre = _tpal_call_with_spy(_tpal_prebreak_t)
    assert len(_tpal_calls_pre) == 0, (
        "tracer_passive_anomalous_leak dispatched a cathode solve in a "
        "phase with no cathode solve"
    )
    assert np.all(_tpal_out_pre == 0.0), _tpal_out_pre

    # (ii) ANTI-VACUITY: main_discharge DOES have a cathode solve, so the
    # gate must not suppress every dispatch -- only the phase-inappropriate
    # one.
    _tpal_out_main, _tpal_calls_main = _tpal_call_with_spy(_tpal_discharge_t)
    assert len(_tpal_calls_main) == 1, (
        "tracer_passive_anomalous_leak must still dispatch a cathode solve "
        "in a phase that has one"
    )

    # (iii) THE RECONSTRUCTED PRE-FIX CALL, direct: unconditional dispatch
    # in neutral_prebreakdown returns a disabled solve rather than raising,
    # which is why this case tests call suppression rather than a refusal.
    _tpal_recon = _tpal_sim.solve_cathode_boundary(
        state=_tpal_sim.state, time=_tpal_prebreak_t, update_cache=False
    )
    assert _tpal_recon.metadata["enabled"] is False, _tpal_recon.metadata
    assert _tpal_recon.beam_result is None


# --------------------------------------------------------------------
# tracer-construction-refusals
# --------------------------------------------------------------------
@_case(
    "tracer-construction-refusals",
    historical_stance=True,
    provides=("_r2_refuses",),
)
def _case_tracer_construction_refusals(_r2_on_config):
    # ---- (iii) every construction-time refusal, and (iv) its anti-vacuity ----
    # Each case: the broken config RAISES naming the offending key, and the
    # SAME config with only that key repaired constructs. The second half is
    # the anti-vacuity check -- without it a raise could be coming from
    # anywhere in the configuration.
    def _r2_refuses(fragment, **overrides):
        params, flags = _r2_on_config()
        flag_keys = {key for key in overrides if key in flags}
        for key, value in overrides.items():
            (flags if key in flag_keys else params)[key] = value
        try:
            LAPDSim1D(params, flags)
        except ValueError as error:
            assert fragment in str(error), (fragment, str(error))
        else:
            raise AssertionError(
                f"regime_tracer must refuse this configuration ({fragment})"
            )
        # anti-vacuity: repair only the offending key and it must construct
        params, flags = _r2_on_config()
        LAPDSim1D(params, flags)

    _r2_refuses("cathode_coupling on", cathode_coupling=False)
    _r2_refuses("Plasma on", Plasma=False)
    _r2_refuses("R2 is fluid-arms", neutral_model="kinetic_dvm")
    _r2_refuses("restart_from", restart_from="/nonexistent/payload.h5")
    for _key in (
        "tracer_passivity_current_ratio",
        "tracer_passivity_thinness",
        "tracer_passivity_depletion",
    ):
        _r2_refuses(_key, **{_key: 0.0})
        _r2_refuses(_key, **{_key: 1.5})
    _r2_refuses("tracer_passivity_hysteresis", tracer_passivity_hysteresis=1.0)
    _r2_refuses("tracer_refresh_tol", tracer_refresh_tol=-1.0)
    _r2_refuses("tracer_activation_ne", tracer_activation_ne=1.0e8)
    _r2_refuses("tracer_overlap_band_ne", tracer_overlap_band_ne=(1e11, 1e10))
    _r2_refuses("tracer_overlap_band_ne", tracer_overlap_band_ne=None)
    _r2_refuses("tracer_overlap_rtol", tracer_overlap_rtol=0.0)
    return locals()


# --------------------------------------------------------------------
# tracer-census-and-criterion
# --------------------------------------------------------------------
@_case("tracer-census-and-criterion")
def _case_tracer_census_and_criterion(_r2, _r2_on_config):
    # ---- the census exists from day one, and names a binding criterion ----
    _r2_cen_params, _r2_cen_flags = _r2_on_config()
    _r2_cen_sim = LAPDSim1D(_r2_cen_params, _r2_cen_flags)
    _r2_cen_result = _r2_cen_sim.run(t_end=6.0e-8, dt=2.0e-8)
    _r2_cen = getattr(_r2_cen_result, "tracer_criterion_census", None)
    assert _r2_cen is not None, "a tracer run must carry its criterion census"
    assert set(_r2_cen["ratios"]) == set(_r2.CRITERION_NAMES)
    assert _r2_cen["criterion"].shape == (_r2_cen_sim.geometry.cells,)
    assert _r2_cen["passive"].shape == (_r2_cen_sim.geometry.cells,)
    assert _r2_cen["refreshes"] >= 1
    assert "transport_ratio" in _r2_cen
    assert _r2_cen_sim._tracer_census_line().startswith("regime_r2 tracer census")
    # ANTI-VACUITY: a run WITHOUT the flag carries no census at all.
    _r2_nocen_p, _r2_nocen_f = default_config()
    _r2_nocen_p["nx"] = 12
    _r2_nocen_p["initial_neutral_state"] = "fill"
    _r2_nocen_sim = LAPDSim1D(_r2_nocen_p, _r2_nocen_f)
    assert not hasattr(
        _r2_nocen_sim.run(t_end=2.0e-10, dt=1.0e-10), "tracer_criterion_census"
    )
    assert _r2_nocen_sim._tracer_census_line() is None


# --------------------------------------------------------------------
# tracer-ql-booking-passive-cells
# --------------------------------------------------------------------
@_case(
    "tracer-ql-booking-passive-cells",
    provides=(
        "_r2ql_beam_kwargs", "_r2ql_config", "_r2ql_sim", "_r2ql_solve",
    ),
)
def _case_tracer_ql_booking_passive_cells(_r2, _r2_on_config, solver_module):
    # ---- (v) the QL/anomalous booking is REFUSED on passive cells ----
    # Quasilinear absorption is a beam-PLASMA instability, so on a cell the
    # tracer owns -- a cell that by definition carries no plasma worth speaking
    # of -- there is no wave medium and the channel does not exist. The gate is
    # the passive mask and nothing else: no density threshold is introduced, so
    # the tracer-to-fluid handoff and the onset of QL absorption are one event.
    # The tracer configs above start from a cold cathode, which does not launch
    # a beam inside a smoke-sized window -- and with no beam there is no
    # anomalous power, so every assertion here would pass on an empty set. The
    # already-emitting cathode the CSDA blocks use is what makes the checks
    # bite; the precondition below is what enforces that it did.
    def _r2ql_config(**overrides):
        params, flags = _r2_on_config()
        params.update({
            # The surface held: no step's temperature increment survives
            # this heat capacity, and nothing cleans at a zero cross section.
            "cathode_Ts_base_K": 1998.15,
            "cathode_heat_capacity_J_per_K": 1.0e30,
            "cathode_cleaning_sigma_cm2": 0.0,
            "cathode_cleaning_E_th_eV": None,
        })
        params.update(overrides)
        return params, flags

    _r2ql_params, _r2ql_flags = _r2ql_config()
    # The electrode sample follows the accepted state: over this 1 us window
    # the supply-averaged sample would still sit at the cold start and no
    # beam would launch.
    _r2ql_sim = _tracking_electrode_sample(
        LAPDSim1D(_r2ql_params, _r2ql_flags)
    )
    _r2ql_sim.run(t_end=1.0e-6, dt=1.0e-7)
    _r2ql_passive = _r2ql_sim._tracer_passive
    _r2ql_solve = _r2ql_sim.solve_cathode_boundary(
        state=_r2ql_sim.state, time=_r2ql_sim._time, update_cache=False
    )
    _r2ql_S, _r2ql_net, _r2ql_full = _r2ql_sim._tracer_beam_rows(
        _r2ql_sim.state, _r2ql_solve, _r2ql_sim._time
    )
    _r2ql_beam_kwargs = _r2ql_sim._tracer_beam_kwargs(
        _r2ql_sim.state, _r2ql_solve, _r2ql_sim._time
    )
    _r2ql_power = _r2.beam_anomalous_power_density(**_r2ql_beam_kwargs)
    # PRECONDITION, and the reason the assertions below are not vacuous: there
    # IS anomalous power on passive cells at this state. Without it a refusal
    # that does nothing would pass everything that follows.
    assert float(np.max(np.abs(_r2ql_power[_r2ql_passive]))) > 0.0, (
        "the QL-refusal assertions are vacuous: no anomalous power is booked "
        "on any passive cell at this state, so nothing is being refused"
    )
    # The refusal removes exactly the anomalous share, and only there.
    assert np.array_equal(
        _r2ql_net, _r2ql_full - np.where(_r2ql_passive, _r2ql_power, 0.0)
    ), "the passive-cell booking must be the full beam power minus QL exactly"
    assert np.array_equal(
        _r2ql_net[~_r2ql_passive], _r2ql_full[~_r2ql_passive]
    ), "an ACTIVE cell's beam power booking must be untouched"
    # The auditable invariant: zero, exactly, on every passive cell.
    assert float(
        np.max(np.abs(_r2ql_sim.tracer_passive_anomalous_leak()))
    ) == 0.0, "QL power leaked into a passive cell's booking"

    # ANTI-VACUITY: break the refusal and the invariant must catch it. Rebinding
    # the solver module's own reference is the smallest faithful stand-in for
    # deleting the subtraction -- the audit recomputes the anomalous share
    # through physics.tracer's reference, so it does not travel through the path
    # it is checking and still sees the truth.
    _r2ql_real = solver_module.beam_anomalous_power_density
    try:
        solver_module.beam_anomalous_power_density = (
            lambda *_args, **kwargs: np.zeros(
                int(kwargs["geometry"].cells), dtype=float
            )
        )
        _r2ql_leak = _r2ql_sim.tracer_passive_anomalous_leak()
    finally:
        solver_module.beam_anomalous_power_density = _r2ql_real
    assert np.array_equal(
        _r2ql_leak, np.where(_r2ql_passive, _r2ql_power, 0.0)
    ), (
        "the passive-cell QL leak check is vacuous: with the refusal removed "
        "it must report the whole anomalous power on the passive cells"
    )
    assert float(np.max(np.abs(_r2ql_leak))) > 0.0
    # ... and the real refusal is back, so the invariant holds again.
    assert float(
        np.max(np.abs(_r2ql_sim.tracer_passive_anomalous_leak()))
    ) == 0.0

    # The production stance smooths the deposition over a fixed physical width
    # (50 cm), and the smoothing is applied to the LUMPED power -- so the
    # anomalous share has to go through the same kernel or the subtraction
    # removes a differently-shaped profile from the one that was booked. This
    # case is here because getting it wrong is quiet: the arrays still have the
    # right units and the right total, only the profile is wrong.
    _r2ql_sm_params, _r2ql_sm_flags = _r2ql_config(
        beam_deposition_smoothing_cm=50.0
    )
    _r2ql_sm_sim = _tracking_electrode_sample(
        LAPDSim1D(_r2ql_sm_params, _r2ql_sm_flags)
    )
    _r2ql_sm_sim.run(t_end=1.0e-6, dt=1.0e-7)
    _r2ql_sm_solve = _r2ql_sm_sim.solve_cathode_boundary(
        state=_r2ql_sm_sim.state, time=_r2ql_sm_sim._time, update_cache=False
    )
    _r2ql_sm_rows = _r2ql_sm_sim._tracer_beam_rows(
        _r2ql_sm_sim.state, _r2ql_sm_solve, _r2ql_sm_sim._time
    )
    _r2ql_sm_kwargs = _r2ql_sm_sim._tracer_beam_kwargs(
        _r2ql_sm_sim.state, _r2ql_sm_solve, _r2ql_sm_sim._time
    )
    _r2ql_sm_power = _r2.beam_anomalous_power_density(**_r2ql_sm_kwargs)
    # The unsmoothed share is a DIFFERENT array, so the assertion below is
    # about the kernel and not merely about subtracting something.
    _r2ql_sm_kwargs_raw = dict(
        _r2ql_sm_kwargs,
        input_dict=dict(
            _r2ql_sm_kwargs["input_dict"], beam_deposition_smoothing_cm=0.0
        ),
    )
    _r2ql_sm_raw = _r2.beam_anomalous_power_density(**_r2ql_sm_kwargs_raw)
    assert not np.array_equal(_r2ql_sm_power, _r2ql_sm_raw), (
        "the smoothing case is vacuous: the kernel changed nothing"
    )
    assert np.array_equal(
        _r2ql_sm_rows[1],
        _r2ql_sm_rows[2]
        - np.where(_r2ql_sm_sim._tracer_passive, _r2ql_sm_power, 0.0),
    ), "the refused QL share must go through the booking's own smoothing kernel"
    assert float(
        np.max(np.abs(_r2ql_sm_sim.tracer_passive_anomalous_leak()))
    ) == 0.0
    # Conservative kernel: smoothing moves the anomalous power around, it does
    # not create or destroy it.
    _r2ql_sm_Vp = np.asarray(
        _r2ql_sm_sim._geometry.plasma_volume_cm3, dtype=float
    )
    assert abs(
        float(np.sum(_r2ql_sm_power * _r2ql_sm_Vp))
        / float(np.sum(_r2ql_sm_raw * _r2ql_sm_Vp))
        - 1.0
    ) < 1e-12, "the smoothing kernel must conserve the anomalous power total"
    return locals()


# --------------------------------------------------------------------
# tracer-owner-state-criteria
# --------------------------------------------------------------------
@_case("tracer-owner-state-criteria")
def _case_tracer_owner_state_criteria(
    _r2, _r2_on_config, _r2_refuses, _r2ql_beam_kwargs, _r2ql_config,
    _r2ql_sim, _r2ql_solve
):
    # ---- (vi) the criteria read the OWNER's state, per cell ----
    # The quasi-static balance is solved on the passive set only, so off that
    # set its Te is a floor-by-convention filler; and the affine update's
    # density on a cell the fluid owns is that cell's step-START density
    # advanced by a description that does not own it. Both are composed against
    # the fluid's own state before any criterion reads them.
    _r2own_params, _r2own_flags = _r2ql_config()
    _r2own_sim = _tracking_electrode_sample(
        LAPDSim1D(_r2own_params, _r2own_flags)
    )
    _r2own_sim.run(t_end=1.0e-6, dt=1.0e-7)
    _r2own_cells = int(_r2own_sim.geometry.cells)
    # Hand two cells to the fluid by hand. Every tracer config starts with the
    # whole plasma passive, and the criteria's active-cell branch is exactly
    # what is under test, so it has to be reached deliberately here.
    _r2own_mask = _r2own_sim._tracer_passive.copy()
    _r2own_live = np.flatnonzero(_r2own_mask)
    assert _r2own_live.size >= 4, "need passive cells to hand over"
    _r2own_handed = _r2own_live[:2]
    _r2own_mask[_r2own_handed] = False
    _r2own_sim._tracer_passive = _r2own_mask
    _r2own_fluid_n = np.asarray(_r2own_sim.state.n, dtype=float)
    _r2own_fluid_Te = np.asarray(_r2own_sim.derived.Te, dtype=float)
    # Distinct stand-ins for what the tracer would have said, so "took the
    # fluid's value" and "took the tracer's" cannot be confused.
    _r2own_tracer_n = _r2own_fluid_n * 3.0 + 1.0
    _r2own_tracer_Te = _r2own_fluid_Te * 5.0 + 2.0
    _r2own_got_n = _r2own_sim._tracer_criteria_n_cm3(_r2own_tracer_n)
    _r2own_got_Te = _r2own_sim._tracer_criteria_Te_eV(_r2own_tracer_Te)
    assert np.array_equal(
        _r2own_got_n[_r2own_mask], _r2own_tracer_n[_r2own_mask]
    ), "a PASSIVE cell's criteria must read the tracer's own density"
    assert np.array_equal(
        _r2own_got_Te[_r2own_mask], _r2own_tracer_Te[_r2own_mask]
    ), "a PASSIVE cell's criteria must read the quasi-static Te"
    assert np.array_equal(
        _r2own_got_n[~_r2own_mask], _r2own_fluid_n[~_r2own_mask]
    ), "an ACTIVE cell's criteria must read the FLUID's own density"
    assert np.array_equal(
        _r2own_got_Te[~_r2own_mask], _r2own_fluid_Te[~_r2own_mask]
    ), "an ACTIVE cell's criteria must read the FLUID's own Te"
    # ANTI-VACUITY: the composition has to CHANGE something on the handed
    # cells, or the two assertions above would hold for a build that composed
    # nothing at all.
    assert not np.array_equal(
        _r2own_got_n[_r2own_handed], _r2own_tracer_n[_r2own_handed]
    ), "the density composition is vacuous: it returned the tracer's values"
    assert not np.array_equal(
        _r2own_got_Te[_r2own_handed], _r2own_tracer_Te[_r2own_handed]
    ), "the Te composition is vacuous: it returned the balance's values"
    # ... and the balance's SOLVE DOMAIN is the passive set, so a handed-over
    # cell is never asked for a quasi-static Te in the first place.
    _r2own_solve = _r2own_sim.solve_cathode_boundary(
        state=_r2own_sim.state, time=_r2own_sim._time, update_cache=False
    )
    _r2own_S, _r2own_P, _r2own_Pf = _r2own_sim._tracer_beam_rows(
        _r2own_sim.state, _r2own_solve, _r2own_sim._time
    )
    _r2own_Te_qs, _r2own_sc = _r2.quasistatic_Te_eV(
        state=_r2own_sim.state,
        n_true=_r2own_fluid_n,
        n_probe=np.maximum(_r2own_fluid_n, _r2own_sim.floors["n"]),
        Ti_eV=np.full(_r2own_cells, float(_r2own_sim.floors["Ti"])),
        S_beam=_r2own_S,
        P_beam_net=_r2own_P,
        floors=_r2own_sim.floors,
        ion_mass_g=_r2own_sim.ion_mass_g,
        mu=_r2own_sim._mu,
        cooling_kwargs=_r2own_sim._electron_cooling_kwargs(),
        exchange_kwargs=_r2own_sim._tracer_exchange_kwargs(),
        boundary_rhs=_r2own_sim._tracer_boundary_rhs(
            _r2own_solve, _r2own_sim._time
        ),
        active=_r2own_mask & (
            (_r2own_fluid_n > 0.0) | (_r2own_S > 0.0)
        ),
        Te_ceiling_eV=_r2own_sim._tracer_beam_energy_eV(_r2own_solve),
    )
    assert np.all(
        _r2own_Te_qs[_r2own_handed] == _r2own_sim.floors["Te"]
    ), "a cell outside the solve domain must not come back solved"

    # PRESENCE GATE for the new code: with the flag OFF there are no passive
    # cells, the beam booking the tracer helpers report is the fluid's own
    # untouched row, and the audit is identically zero.
    _r2ql_off_params, _r2ql_off_flags = _r2ql_config()
    _r2ql_off_flags["regime_tracer"] = False
    _r2ql_off_sim = _tracking_electrode_sample(
        LAPDSim1D(_r2ql_off_params, _r2ql_off_flags)
    )
    _r2ql_off_sim.run(t_end=1.0e-6, dt=1.0e-7)
    _r2ql_off_solve = _r2ql_off_sim.solve_cathode_boundary(
        state=_r2ql_off_sim.state,
        time=_r2ql_off_sim._time,
        update_cache=False,
    )
    _r2ql_off_rows = _r2ql_off_sim._tracer_beam_rows(
        _r2ql_off_sim.state, _r2ql_off_solve, _r2ql_off_sim._time
    )
    assert np.array_equal(_r2ql_off_rows[1], _r2ql_off_rows[2]), (
        "with regime_tracer off nothing may be subtracted from the beam power"
    )
    assert float(
        np.max(np.abs(_r2ql_off_sim.tracer_passive_anomalous_leak()))
    ) == 0.0

    # With no cathode solve there is nothing to read and the accessor is zero
    # rather than guessing.
    assert not np.any(
        _r2.beam_anomalous_power_density(
            **dict(_r2ql_beam_kwargs, cathode_solve=None)
        )
    )
    # And the accessor shares the BOOKING's gate, not the deposition object's
    # presence: read back through flags that switch the beam rows off, the same
    # live solve yields zero. Without this the subtraction could remove power
    # from a row that booked none.
    _r2ql_gate_kwargs = dict(
        _r2ql_beam_kwargs,
        input_flags=dict(_r2ql_beam_kwargs["input_flags"],
                         cathode_coupling=False),
    )
    assert _r2ql_solve.beam_deposition is not None
    assert not np.any(_r2.beam_anomalous_power_density(**_r2ql_gate_kwargs))
    assert not np.any(
        np.asarray(
            beam_ionization_rhs_terms(
                I_ion=_r2ql_sim._I_ion, **_r2ql_gate_kwargs
            )["beam_power_deposition"].Ee,
            dtype=float,
        )
    )
