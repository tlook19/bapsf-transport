"""Smoke cases: time integration, timestep bounds, heat conduction, fluid
operators and the implicit sinks.
"""

import dataclasses
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np

from cablp.cathode import (
    circuit_idriven as _cathode_solver_idriven_mod,
    circuit_common as _cathode_solver_mod,
)
from cablp.constants import He_e_mass_ratio, ev_to_erg, m_He_cgs
from cablp.plasma.params import LN_LAMBDA_MIN, c_log, time_elec_coll
from cablp.solvers._sim1d import (
    LAPDSim1D,
    default_config,
    load_result_hdf5,
    summarize_result,
)
from cablp.solvers._sim1d.core.geometry import anode_flanking_cells
from cablp.solvers._sim1d.core.state import (
    conservative_from_primitives,
    derive_state,
    pack_state,
)
from cablp.solvers._sim1d.core.timestep import (
    ELECTRODE_SINK_DT_FRACTION,
    electrode_sink_rate_timestep,
    plasma_source_timestep,
    suggest_timestep,
)
from cablp.solvers._sim1d.core.validation import (
    resolve_energy_exchange_rate_fraction,
)
from cablp.solvers._sim1d.physics.conduction import (
    heat_conduction_timestep_bound,
    implicit_heat_conduction_step,
)
from cablp.solvers._sim1d.physics.energy import electron_ion_relaxation_rate
from cablp.solvers._sim1d.physics.sources import velocity_divergence
from cablp.solvers._sim1d.solver import _timestep_limiters

from ._harness import (
    _TOL_ROUNDOFF,
    _anode_sink_config,
    _anode_sink_sim,
    _base_config,
    _base_sim,
    _case,
)


# --------------------------------------------------------------------
# energy-exchange-rate-bound
# --------------------------------------------------------------------
@_case(
    "energy-exchange-rate-bound",
    historical_stance=True,
)
def _case_energy_exchange_rate_bound():
    # The electron-ion exchange's RATE bound (default off). The fractional
    # bound above vanishes as Te -> Ti, so a cold dense column sitting at
    # Te ~= Ti is bounded by nothing while nu_eq is at its stiffest; the rate
    # bound is the stability complement. Built on an exchange-dominated state
    # at that operating point: n ~ 8e12 cm^-3, Te = Ti ~ 0.1 eV.
    params, flags = _base_config()
    exchange_rate_sim, exchange_rate_snapshot = _base_sim()
    geom = exchange_rate_snapshot.geometry
    stiff_state = conservative_from_primitives(
        n=np.full(geom.cells, 8.0e12),
        nn=exchange_rate_snapshot.state.nn,
        nn_a=exchange_rate_snapshot.state.nn,
        u=np.zeros(geom.cells),
        Te=np.full(geom.cells, 0.11),
        Ti=np.full(geom.cells, 0.11),
        ion_mass_g=exchange_rate_sim.ion_mass_g,
    )
    stiff_y = pack_state(stiff_state)

    # (a) The bound resolves to c / nu_eq,max, against nu_eq evaluated
    # INDEPENDENTLY from the Braginskii collision time rather than from the
    # helper the bound itself calls.
    stiff_derived = derive_state(
        stiff_state,
        floors=exchange_rate_sim.floors,
        ion_mass_g=exchange_rate_sim.ion_mass_g,
    )
    stiff_n = np.maximum(stiff_state.n, exchange_rate_sim.floors["n"])
    stiff_ln_lambda = np.maximum(
        c_log(stiff_derived.Te, stiff_n, kind="ei"), LN_LAMBDA_MIN
    )
    independent_nu_eq = 2.0 / (
        time_elec_coll(stiff_derived.Te, stiff_n, stiff_ln_lambda)
        * He_e_mass_ratio
    )
    helper_nu_eq = electron_ion_relaxation_rate(
        state=stiff_state,
        floors=exchange_rate_sim.floors,
        ion_mass_g=exchange_rate_sim.ion_mass_g,
        mu=exchange_rate_sim._mu,
    )
    assert np.all(np.abs(helper_nu_eq / independent_nu_eq - 1.0) < 1.0e-12)
    # The regime this bound exists for: nu_eq is stiff on the ~1 us scale.
    assert np.all(independent_nu_eq > 1.0e5)

    for fraction in (0.5, 1.0):
        armed_params = dict(params)
        armed_params["energy_exchange_rate_fraction"] = fraction
        armed_sim = LAPDSim1D(armed_params, flags)
        armed_dt = armed_sim.suggest_timestep(y=stiff_y)
        assert np.isclose(
            armed_dt.dt_energy_exchange_rate,
            fraction / np.max(independent_nu_eq),
            rtol=1.0e-12,
            atol=0.0,
        )
        # It is a candidate of its own, so it can only tighten the step and it
        # names itself when it binds.
        assert armed_dt.dt <= armed_dt.dt_energy_exchange_rate
        assert armed_dt.active_constraint == "energy_exchange_rate"

    # (b) With the key at None the bound is absent from the dt census: the
    # candidate is withdrawn to inf, it cannot be the active constraint, and
    # the step it would have set is left to the other bounds.
    assert params["energy_exchange_rate_fraction"] is None
    assert default_config()[0]["energy_exchange_rate_fraction"] is None
    unarmed_sim = LAPDSim1D(params, flags)
    unarmed_dt = unarmed_sim.suggest_timestep(y=stiff_y)
    assert unarmed_dt.dt_energy_exchange_rate == np.inf
    assert unarmed_dt.active_constraint != "energy_exchange_rate"
    assert unarmed_dt.dt != armed_dt.dt
    assert "energy_exchange_rate" not in dict(
        _timestep_limiters(unarmed_dt, count=len(dataclasses.fields(unarmed_dt)))
    )

    # An int is a real number and is accepted, resolved to its float; the
    # upper end of the admissible range is exactly c = 1, where z = -2.
    assert resolve_energy_exchange_rate_fraction(
        {"energy_exchange_rate_fraction": 1}
    ) == 1.0

    # (c) The refusals, at construction. 1.5 is refused because z = -2 c = -3
    # falls outside SSPRK2's real-axis stability interval.
    for bad_fraction in (0, 1.5, "x"):
        try:
            LAPDSim1D(
                {**params, "energy_exchange_rate_fraction": bad_fraction},
                flags,
            )
        except ValueError as exc:
            assert "energy_exchange_rate_fraction" in str(exc)
        else:
            raise AssertionError(
                "energy_exchange_rate_fraction="
                f"{bad_fraction!r} was accepted at construction"
            )
    return locals()


# --------------------------------------------------------------------
# timestep-dt-growth-reapproach
# --------------------------------------------------------------------
@_case(
    "timestep-dt-growth-reapproach",
    historical_stance=True,
    provides=("retry_flags", "retry_params"),
)
def _case_timestep_dt_growth_reapproach(growth_flags, growth_params):
    # --- accelerated dt_growth re-approach (armed at the shipped default) ----
    # A long ramp: an early phase boundary sets a small first step, then
    # nothing physical binds and dt_growth caps every step while it climbs.
    # This is the regime the probe measured (80.6% of steps growth-capped at a
    # median 364x below the binding physics bound).
    params, flags = _base_config()
    ramp_params = dict(growth_params)
    ramp_params["tau_prebreakdown"] = 2.0e-9
    ramp_params["tau_discharge"] = 40.0e-6
    # The UNACCELERATED reference arm. Patience presence-gates the whole
    # mechanism, so 0 skips the branch and this is the uniform
    # dt_growth_factor ramp a run predating these keys took -- the same 17
    # steps this case pinned while 0 was the shipped default.
    ramp_slow_params = {**ramp_params, "dt_growth_recovery_patience": 0}
    ramp_base = LAPDSim1D(ramp_slow_params, growth_flags).run(t_end=3.0e-7)
    assert ramp_base.steps == 17
    assert [diag.step_cap for diag in ramp_base.diagnostics][:5] == [
        "phase_boundary",
        "dt_growth",
        "dt_growth",
        "dt_growth",
        "dt_growth",
    ]
    ramp_fast = LAPDSim1D(
        {
            **ramp_params,
            "dt_growth_recovery_patience": 3,
            "dt_growth_recovery_factor": 4.0,
        },
        growth_flags,
    ).run(t_end=3.0e-7)
    # Same ramp, far fewer steps: the whole point of the mechanism.
    assert ramp_fast.steps == 7, ramp_fast.steps
    ramp_fast_dt = [diag.accepted_dt for diag in ramp_fast.diagnostics]
    ramp_base_dt = [diag.accepted_dt for diag in ramp_base.diagnostics]
    # HYSTERESIS, both halves. Engaging takes evidence: the first step is
    # capped by the phase boundary, so steps 2-4 must still ramp at the BASE
    # 1.25 while the streak of three growth-capped steps is being earned...
    assert np.allclose(ramp_fast_dt[:4], ramp_base_dt[:4], **_TOL_ROUNDOFF)
    assert np.allclose(
        ramp_fast_dt[1:4],
        [ramp_fast_dt[0] * 1.25**k for k in (1, 2, 3)],
        **_TOL_ROUNDOFF,
    )
    # ...and only the step AFTER patience is met jumps by the recovery factor.
    assert np.isclose(ramp_fast_dt[4], ramp_fast_dt[3] * 4.0, **_TOL_ROUNDOFF)
    assert np.isclose(ramp_fast_dt[5], ramp_fast_dt[4] * 4.0, **_TOL_ROUNDOFF)
    # It never weakens a bound -- every accepted step is still <= dt_max.
    assert max(ramp_fast_dt) <= ramp_params["dt_max"] * (1.0 + 1.0e-12)
    # DEFAULT ON and presence-gated. The shipped patience ARMS the mechanism,
    # so a run that says nothing about these keys accelerates; and a config
    # carrying them at their defaults is step-for-step identical to one with
    # them absent from the params dict entirely, which is what presence-gating
    # has to mean now that the gate is open by default.
    assert default_config()[0]["dt_growth_recovery_patience"] == 4
    assert default_config()[0]["dt_growth_recovery_factor"] == 4.0
    assert ramp_params["dt_growth_recovery_patience"] == 4
    ramp_default = LAPDSim1D(ramp_params, growth_flags).run(t_end=3.0e-7)
    ramp_absent_params = dict(ramp_params)
    ramp_absent_params.pop("dt_growth_recovery_patience")
    ramp_absent_params.pop("dt_growth_recovery_factor")
    ramp_absent = LAPDSim1D(ramp_absent_params, growth_flags).run(t_end=3.0e-7)
    ramp_default_dt = [diag.accepted_dt for diag in ramp_default.diagnostics]
    assert [diag.accepted_dt for diag in ramp_absent.diagnostics] == ramp_default_dt
    assert np.array_equal(ramp_absent.n, ramp_default.n)
    assert np.array_equal(ramp_absent.Ee, ramp_default.Ee)
    # The armed default is the mechanism, at ITS patience: the same ramp as the
    # unaccelerated arm while a streak of four growth-capped steps is earned,
    # then the recovery factor -- one step later than the patience-3 arm, and
    # over the same horizon in fewer steps than the unaccelerated arm.
    assert np.allclose(ramp_default_dt[:5], ramp_base_dt[:5], **_TOL_ROUNDOFF)
    assert np.isclose(ramp_default_dt[5], ramp_default_dt[4] * 4.0, **_TOL_ROUNDOFF)
    assert ramp_default.steps < ramp_base.steps, (
        ramp_default.steps, ramp_base.steps
    )
    # ...and turning it off is still bit-exact: patience 0 recovers the
    # unaccelerated ramp exactly, keys present or not.
    ramp_off = LAPDSim1D(
        {
            **ramp_params,
            "dt_growth_recovery_patience": 0,
            "dt_growth_recovery_factor": 4.0,
        },
        growth_flags,
    ).run(t_end=3.0e-7)
    assert [diag.accepted_dt for diag in ramp_off.diagnostics] == ramp_base_dt
    assert np.array_equal(ramp_off.n, ramp_base.n)
    assert np.array_equal(ramp_off.Ee, ramp_base.Ee)
    # Misconfiguration is loud, and at CONSTRUCTION.
    for bad_ramp in (
        {"dt_growth_recovery_patience": -1},
        {"dt_growth_recovery_patience": 1.5},
        {"dt_growth_recovery_patience": "soon"},
        # A recovery factor at or below the base could never accelerate.
        {"dt_growth_recovery_patience": 2, "dt_growth_recovery_factor": 1.25},
        # ...and the factor is LIVE at the shipped patience, so a bad one is
        # refused without any patience override to make it so.
        {"dt_growth_recovery_factor": 1.25},
        {"dt_growth_recovery_patience": 2, "dt_growth_recovery_factor": 0.5},
        {
            "dt_growth_recovery_patience": 2,
            "dt_growth_recovery_factor": float("nan"),
        },
    ):
        try:
            LAPDSim1D({**ramp_params, **bad_ramp}, growth_flags)
        except ValueError as error:
            assert "dt_growth_recovery" in str(error), str(error)
        else:
            raise AssertionError(f"{bad_ramp} must raise")
    # A bad recovery factor is INERT while patience is 0: the key is not
    # consulted at all on the off path, so it cannot refuse a default run.
    LAPDSim1D(
        {
            **ramp_params,
            "dt_growth_recovery_patience": 0,
            "dt_growth_recovery_factor": 0.5,
        },
        growth_flags,
    )

    retry_params = dict(params)
    retry_flags = dict(flags)
    retry_flags["Plasma"] = False
    retry_flags["heat_conduction"] = False
    retry_params["dt_save"] = 0.0
    retry_params["pump_enabled"] = False
    retry_params["dt_max"] = 1.0e-6
    retry_params["neutral_dt_fraction"] = 100.0
    retry_params["max_neutral_step_fraction"] = 6.0
    return locals()


# --------------------------------------------------------------------
# timestep-surface-loss-floor-exempt-hysteresis
# --------------------------------------------------------------------
@_case(
    "timestep-surface-loss-floor-exempt-hysteresis",
    historical_stance=True,
)
def _case_timestep_surface_loss_floor_exempt_hysteresis(
    growth_flags, growth_params
):
    # --- hysteresis band on the surface_loss floor exemption (armed) ---------
    # A single-threshold exemption is knife-edge: the accept-time floor clip
    # perturbs a floor-pinned cell's margin by float residue every step, so one
    # threshold lets such a cell alternate between exempt and bound. The band
    # keeps the 1e-3 ENTRY threshold and widens RE-ADMISSION. It is armed at
    # the shipped default; 0.0 selects the knife edge.
    from cablp.solvers._sim1d.solver import SURFACE_LOSS_FLOOR_EXEMPT_RTOL

    hyst_inner = SURFACE_LOSS_FLOOR_EXEMPT_RTOL
    hyst_outer = default_config()[0]["surface_loss_floor_exempt_exit_rtol"]
    assert hyst_outer == 1.0e-1
    hyst_floors = {"n": 1.0e8, "Te": 0.1, "Ti": 0.1}
    hyst_n = np.array([1.0e10, 1.0e10])
    hyst_floor_energy = 1.5 * hyst_floors["Te"] * ev_to_erg * hyst_n

    def _hyst_state(rel_margin):
        # Cell 0 sits at rel_margin of its per-cell floor energy; cell 1 is a
        # healthy cell an order of magnitude clear of every threshold.
        margin = np.array([rel_margin, 1.0]) * hyst_floor_energy
        return SimpleNamespace(
            n=hyst_n,
            Ee=hyst_floor_energy + margin,
            Ei=hyst_floor_energy + margin,
        )

    # Only the electron channel drains, and cell 0 drains 1e6x harder, so
    # whether cell 0 is exempt IS the bound this function returns.
    hyst_rhs = SimpleNamespace(
        n=np.zeros(2),
        Ee=np.array([-1.0e-3, -1.0e-9]),
        Ei=np.zeros(2),
    )

    def _hyst_dt(rel_margin, latch):
        return plasma_source_timestep(
            state=_hyst_state(rel_margin),
            source_rhs=hyst_rhs,
            floors=hyst_floors,
            fraction=0.25,
            floor_exempt_rtol=hyst_inner,
            floor_exempt_exit_rtol=None if latch is None else hyst_outer,
            floor_exempt_latch=latch,
        )

    # Below the inner threshold, inside the band, then clear of the outer one.
    hyst_walk = (0.5 * hyst_inner, 10.0 * hyst_inner, 0.5)
    hyst_latch = {}
    hyst_band = [_hyst_dt(rel, hyst_latch) for rel in hyst_walk]
    hyst_knife = [_hyst_dt(rel, None) for rel in hyst_walk]
    # Entry and full recovery are the same verdict either way...
    assert hyst_band[0] == hyst_knife[0]
    assert hyst_band[2] == hyst_knife[2]
    # ...and inside the band the knife edge re-admits cell 0 (collapsing the
    # bound by ~8 orders) while the band holds the exemption it granted.
    assert hyst_band[1] > hyst_knife[1] * 1.0e6, (hyst_band[1], hyst_knife[1])
    # Both energy channels latch, independently and per cell.
    assert set(hyst_latch) == {"Ee", "Ei"}
    assert hyst_latch["Ee"].tolist() == [False, False]
    hyst_step_latch = {}
    _hyst_dt(hyst_walk[0], hyst_step_latch)
    assert hyst_step_latch["Ee"].tolist() == [True, False]
    _hyst_dt(hyst_walk[1], hyst_step_latch)
    assert hyst_step_latch["Ee"].tolist() == [True, False]
    _hyst_dt(hyst_walk[2], hyst_step_latch)
    assert hyst_step_latch["Ee"].tolist() == [False, False]
    # The band is not an entry threshold: a cell that never cleared the INNER
    # threshold is not exempted merely by lying inside the band.
    hyst_cold_latch = {}
    assert _hyst_dt(hyst_walk[1], hyst_cold_latch) == hyst_knife[1]
    assert hyst_cold_latch["Ee"].tolist() == [False, False]

    # DEFAULT ON and presence-gated. Two separate properties, and the flip
    # made them testable only as a pair:
    #   (1) the ARMED default builds the latch and selects the band, and a run
    #       with the key at its default is step-for-step identical to one with
    #       the key absent from the params dict entirely -- the template is
    #       what fills it, so "absent" now means 0.1, not 0.0;
    #   (2) the OFF path is still reachable and still presence-gated: an
    #       explicit 0.0 allocates no latch and leaves the single-threshold
    #       expression in place (the knife edge exercised above through
    #       floor_exempt_exit_rtol=None).
    assert default_config()[0]["surface_loss_floor_exempt_exit_rtol"] == hyst_outer
    hyst_params = dict(growth_params)
    hyst_params["tau_prebreakdown"] = 2.0e-9
    hyst_params["tau_discharge"] = 40.0e-6
    assert hyst_params["surface_loss_floor_exempt_exit_rtol"] == hyst_outer
    hyst_default_sim = LAPDSim1D(hyst_params, growth_flags)
    assert hyst_default_sim._surface_loss_floor_exempt_exit_rtol == hyst_outer
    assert hyst_default_sim._surface_loss_floor_exempt_latch == {}
    hyst_absent_params = dict(hyst_params)
    hyst_absent_params.pop("surface_loss_floor_exempt_exit_rtol")
    hyst_absent_sim = LAPDSim1D(hyst_absent_params, growth_flags)
    assert hyst_absent_sim._surface_loss_floor_exempt_exit_rtol == hyst_outer
    hyst_default = hyst_default_sim.run(t_end=3.0e-7)
    hyst_absent = hyst_absent_sim.run(t_end=3.0e-7)
    assert [d.accepted_dt for d in hyst_absent.diagnostics] == [
        d.accepted_dt for d in hyst_default.diagnostics
    ]
    for hyst_field in ("n", "nn", "M", "Ee", "Ei"):
        assert np.array_equal(
            getattr(hyst_absent, hyst_field), getattr(hyst_default, hyst_field)
        ), hyst_field
    # The off path, explicitly selected: no latch, no band.
    hyst_off_sim = LAPDSim1D(
        {**hyst_params, "surface_loss_floor_exempt_exit_rtol": 0.0},
        growth_flags,
    )
    assert hyst_off_sim._surface_loss_floor_exempt_exit_rtol is None
    assert hyst_off_sim._surface_loss_floor_exempt_latch is None
    hyst_off = hyst_off_sim.run(t_end=3.0e-7)
    for hyst_field in ("n", "nn", "M", "Ee", "Ei"):
        assert np.all(np.isfinite(getattr(hyst_off, hyst_field))), hyst_field

    # BOTH knobs armed on the same short run -- as the shipped defaults arm
    # them, with no override at all: the band plus the accelerated dt_growth
    # re-approach. Finite, complete, and the latch is live.
    hyst_on_sim = LAPDSim1D(dict(hyst_params), growth_flags)
    assert hyst_on_sim._surface_loss_floor_exempt_exit_rtol == hyst_outer
    assert hyst_on_sim._surface_loss_floor_exempt_latch == {}
    assert hyst_on_sim._dt_growth_recovery_patience == 4
    hyst_on = hyst_on_sim.run(t_end=3.0e-7)
    assert hyst_on.steps > 0
    for hyst_field in ("n", "nn", "M", "Ee", "Ei"):
        assert np.all(np.isfinite(getattr(hyst_on, hyst_field))), hyst_field
    assert np.all(np.isfinite([d.accepted_dt for d in hyst_on.diagnostics]))
    assert np.all(np.array([d.accepted_dt for d in hyst_on.diagnostics]) > 0.0)

    # Misconfiguration is loud, and at CONSTRUCTION.
    for bad_hyst in (
        {"surface_loss_floor_exempt_exit_rtol": -1.0e-2},
        {"surface_loss_floor_exempt_exit_rtol": float("nan")},
        {"surface_loss_floor_exempt_exit_rtol": float("inf")},
        {"surface_loss_floor_exempt_exit_rtol": "wide"},
        # An outer threshold at or below the inner one is not a band.
        {"surface_loss_floor_exempt_exit_rtol": SURFACE_LOSS_FLOOR_EXEMPT_RTOL},
        {"surface_loss_floor_exempt_exit_rtol": 1.0e-4},
        # ...and one at or above 1.0 is not a band either: re-admission would
        # need a margin exceeding the floor energy itself, which in the large
        # limit is a permanent exemption from this bound.
        {"surface_loss_floor_exempt_exit_rtol": 1.0},
        {"surface_loss_floor_exempt_exit_rtol": 25.0},
    ):
        try:
            LAPDSim1D({**hyst_params, **bad_hyst}, growth_flags)
        except ValueError as error:
            assert "surface_loss_floor_exempt_exit_rtol" in str(error), str(error)
        else:
            raise AssertionError(f"{bad_hyst} must raise")
    return locals()


# --------------------------------------------------------------------
# dt-min-lock
# --------------------------------------------------------------------
@_case(
    "dt-min-lock",
    historical_stance=True,
    provides=("dt_min_lock_snap_result", "dt_min_lock_transient_result"),
)
def _case_dt_min_lock(no_source_params):
    # ---- dt_min lock: honest labeling, census, loud failure ----------------
    # Regression pins for the 2026-08-05 change. The clamp to dt_min used to
    # OVERWRITE active_constraint with "dt_min", so a run pinned at dt_min
    # reported that it was pinned and never by what -- and because the clamp
    # keeps such a run alive, a drained floor-pinned cell produced a silent
    # permanent lock (measured: scripts/dtmin_census_runlengths.txt).
    params, flags = _base_config()
    dtlock_params = dict(no_source_params)
    dtlock_flags = dict(flags)

    # (i) THE TRUE CONSTRAINT SURVIVES THE CLAMP. Synthetic drained
    # floor-pinned scenario: one cell sits exactly ON the density floor while
    # the resolved-source bundle still drains it, so the surface_loss bound
    # requests dt = 0 -- not a timestep request but a modelling breakdown.
    dtlock_sim = LAPDSim1D(dtlock_params, dtlock_flags)
    pinned_state = dtlock_sim.state
    pinned_n = np.asarray(pinned_state.n, dtype=float).copy()
    pinned_cell = pinned_n.size // 2
    pinned_n[pinned_cell] = float(dtlock_sim._floors["n"])
    pinned_state = dataclasses.replace(pinned_state, n=pinned_n)
    draining_source = SimpleNamespace(
        n=np.where(
            np.arange(pinned_n.size) == pinned_cell, -1.0, 0.0
        ),
        Ee=np.zeros_like(pinned_n),
        Ei=np.zeros_like(pinned_n),
    )
    assert (
        plasma_source_timestep(
            state=pinned_state,
            source_rhs=draining_source,
            floors=dtlock_sim._floors,
        )
        == 0.0
    )
    pinned_diag = suggest_timestep(
        state=pinned_state,
        floors=dtlock_sim._floors,
        ion_mass_g=dtlock_sim._ion_mass_g,
        mu=dtlock_sim._mu,
        geometry=dtlock_sim._geometry,
        neutral_exchange_coeff_cm3_s=dtlock_sim.neutral_exchange_coefficients(),
        plasma_source_rhs=draining_source,
        dt_min=1.0e-10,
        dt_max=1.0e-6,
    )
    # The label names the bound that actually minimized, NOT "dt_min".
    assert pinned_diag.active_constraint == "surface_loss"
    assert pinned_diag.dt_raw == 0.0
    assert pinned_diag.clamped_to_dt_min == 1.0
    assert pinned_diag.dt == 1.0e-10
    # A clamp that is not a hard zero is still labeled by its true bound.
    soft_clamp_diag = dtlock_sim.suggest_timestep()
    assert soft_clamp_diag.clamped_to_dt_min == 0.0
    big_floor_sim = LAPDSim1D(
        dict(dtlock_params, dt_min=1.0e-6, dt_max=1.0e-3), dtlock_flags
    )
    soft_clamped = big_floor_sim.suggest_timestep()
    assert soft_clamped.clamped_to_dt_min == 1.0
    assert soft_clamped.active_constraint == soft_clamp_diag.active_constraint
    assert soft_clamped.active_constraint != "dt_min"
    assert 0.0 < soft_clamped.dt_raw < 1.0e-6
    assert soft_clamped.dt == 1.0e-6

    # The guard counts CONSECUTIVE clamped steps, so drive it with a forced
    # clamp: what is under test is the counting and the raise, not the physics
    # that produces a clamp (which (i) already pins).
    class _ForcedClampSim(LAPDSim1D):
        """Force the clamp flag onto the first ``clamp_steps`` suggestions."""

        def __init__(self, params, flags, clamp_steps):
            super().__init__(params, flags)
            self._forced_clamps_left = int(clamp_steps)

        def suggest_timestep(self, *args, **kwargs):
            diag = super().suggest_timestep(*args, **kwargs)
            if self._forced_clamps_left > 0:
                self._forced_clamps_left -= 1
                return dataclasses.replace(
                    diag, clamped_to_dt_min=1.0, dt_raw=0.0
                )
            return diag

    # (iv) A SUB-THRESHOLD TRANSIENT MUST NOT RAISE. Self-releasing clamp
    # episodes are a known-good family (6-10% of steps in some completed
    # afterglow arms); aborting one would be the worse failure.
    transient_params = dict(dtlock_params)
    transient_params["dt_save"] = 0.0
    # Pin every step at dt_max so the run takes a known number of them (the
    # physical bounds here are far larger than t_end).
    transient_params["dt_max"] = 1.0e-10
    transient_params["dt_min_lock_max_steps"] = 5
    transient_sim = _ForcedClampSim(transient_params, dtlock_flags, clamp_steps=5)
    transient_result = transient_sim.run(t_end=1.2e-9)
    assert transient_sim._forced_clamps_left == 0
    # (ii) THE CENSUS COUNTS.
    transient_summary = summarize_result(transient_result)
    assert transient_summary.dt_min_clamped_step_count == 5
    assert transient_summary.max_consecutive_dt_min_clamped_steps == 5
    assert transient_summary.dt_min_hard_zero_step_count == 5
    assert transient_result.steps > 5
    assert "dt_min" not in transient_summary.constraint_counts
    assert [
        diag.clamped_to_dt_min for diag in transient_result.diagnostics[:6]
    ] == [1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
    with tempfile.TemporaryDirectory() as dtlock_dir:
        dtlock_path = Path(dtlock_dir) / "dtlock.h5"
        transient_sim.save_result(dtlock_path, transient_result)
        dtlock_loaded = load_result_hdf5(dtlock_path)
        loaded_summary = summarize_result(dtlock_loaded)
        assert loaded_summary.dt_min_clamped_step_count == 5
        assert loaded_summary.max_consecutive_dt_min_clamped_steps == 5
        assert loaded_summary.dt_min_hard_zero_step_count == 5

    # (ii-b) ACCEPTED STEPS BELOW dt_min ARE SEEN, AND SEPARATELY.
    # The dt_min clamp lifts a bound's request UP to dt_min inside
    # suggest_timestep, but the step caps are applied AFTERWARDS in the run
    # loop and can only shrink the step -- so an accepted step can land
    # strictly BELOW dt_min and the clamp census above cannot see it. A
    # production run (K6d) accepted 9.239e-11 against a configured dt_min of
    # 1e-10 with nothing recording it.
    below_params = dict(dtlock_params)
    below_params["dt_save"] = 0.0
    below_params["dt_min"] = 1.0e-10
    below_sim = LAPDSim1D(below_params, dtlock_flags)
    # t_end lands 5e-11 past the last whole step: below dt_min by construction.
    below_result = below_sim.run(t_end=2.5e-10, dt=1.0e-10)
    below_summary = summarize_result(below_result)
    assert [diag.accepted_dt for diag in below_result.diagnostics] == [
        1.0e-10,
        1.0e-10,
        below_result.diagnostics[-1].accepted_dt,
    ]
    assert below_summary.below_dt_min_step_count == 1
    assert below_summary.below_dt_min_known is True
    assert np.isclose(
        below_summary.below_dt_min_min_accepted_dt, 5.0e-11, **_TOL_ROUNDOFF,
    )
    # It NAMES the cap responsible -- the diagnostic point of the category.
    assert below_summary.below_dt_min_step_cap_counts == {"t_end": 1}
    # DISTINCT, not folded into the clamp count: no step was clamped here.
    assert below_summary.dt_min_clamped_step_count == 0
    assert below_summary.max_consecutive_dt_min_clamped_steps == 0
    # The clamp census and this one are independent: the forced-clamp run
    # above clamped 5 steps and had no below-floor accepted step.
    assert transient_summary.below_dt_min_step_count == 0
    # A result carrying no params cannot know dt_min, and says so rather than
    # reporting a reassuring zero.
    unknowable = summarize_result(
        SimpleNamespace(
            **{
                field: getattr(below_result, field)
                for field in dir(below_result)
                if not field.startswith("_") and field != "params"
            }
        )
    )
    assert unknowable.below_dt_min_known is False
    assert unknowable.below_dt_min_step_count == 0
    assert np.isnan(unknowable.below_dt_min_min_accepted_dt)

    # (ii-c) A SAVE SNAP BELOW dt_min IS A CLAMP THE LOCK COUNTS.
    # The 2026-08-24 defect. ``clamped_to_dt_min`` is computed inside
    # suggest_timestep from the RAW candidate minimum, before the caps; a
    # save-cadence snap below dt_min is therefore invisible to it, the lock
    # counter RESETS on every such step, and the accepted sub-dt_min step then
    # anchors the dt-growth ramp -- so a run whose every step is set by the
    # ramp re-approaching from a sub-dt_min snap can crawl indefinitely with
    # dt_min_lock_max_steps never firing. The accepted-dt signal is what
    # closes it, and the two flags must stay DISTINCT: this run sets the new
    # one on every step and the old one on none.
    snap_params = dict(dtlock_params)
    snap_params["dt_min"] = 1.0e-10
    snap_params["dt_max"] = 1.0e-9
    snap_params["dt_save"] = 2.0e-11
    snap_params["dt_min_lock_max_steps"] = 250000
    snap_sim = LAPDSim1D(snap_params, dtlock_flags)
    snap_result = snap_sim.run(t_end=3.0e-10)
    assert snap_result.steps >= 6, snap_result.steps
    for snap_diag in snap_result.diagnostics:
        # An output-cadence snap on every step (the final one coincides with
        # t_end, which caps it under that name instead).
        assert snap_diag.step_cap in ("save_time", "t_end"), snap_diag.step_cap
        # The raw bound never asked for less than dt_min ...
        assert snap_diag.dt_raw >= snap_params["dt_min"], snap_diag.dt_raw
        assert snap_diag.clamped_to_dt_min == 0.0
        # ... and yet the ACCEPTED step is below it, on every step.
        assert snap_diag.accepted_dt < snap_params["dt_min"]
        assert snap_diag.clamped_to_dt_min_accepted == 1.0
    # And it is what the lock counts: the same run raises once the threshold
    # is small enough to be crossed, on the accepted-dt signal alone.
    snap_lock_params = dict(snap_params, dt_min_lock_max_steps=5)
    snap_locked = LAPDSim1D(snap_lock_params, dtlock_flags)
    try:
        snap_locked.run(t_end=3.0e-10)
    except RuntimeError as error:
        snap_message = str(error)
        assert "dt_min lock" in snap_message
        assert "6 consecutive steps" in snap_message
        assert "ACCEPTED step at dt_min" in snap_message, snap_message
        assert "'save_time'" in snap_message, snap_message
    else:
        raise AssertionError(
            "dt_min lock did not fire on accepted sub-dt_min save snaps"
        )
    # The round trip carries the new flag.
    with tempfile.TemporaryDirectory() as snap_dir:
        snap_path = Path(snap_dir) / "snap.h5"
        snap_sim.save_result(snap_path, snap_result)
        snap_loaded = load_result_hdf5(snap_path)
        assert [
            diag.clamped_to_dt_min_accepted for diag in snap_loaded.diagnostics
        ] == [1.0] * snap_result.steps

    # (iii) PAST THE THRESHOLD IT RAISES, LOUDLY AND WITH THE EVIDENCE.
    lock_params = dict(transient_params)
    locked_sim = _ForcedClampSim(lock_params, dtlock_flags, clamp_steps=40)
    try:
        locked_sim.run(t_end=1.2e-9)
    except RuntimeError as error:
        lock_message = str(error)
        assert "dt_min lock" in lock_message
        assert "6 consecutive steps" in lock_message
        assert "dt_min_lock_max_steps=5" in lock_message
        # the true bound, the offending cell, its density and its floor
        assert f"{transient_result.diagnostics[0].active_constraint!r}" in (
            lock_message
        )
        assert "index" in lock_message
        assert "n_floor=" in lock_message
        assert "modelling breakdown" in lock_message
    else:
        raise AssertionError("dt_min lock guard did not fire past its threshold")

    # Misconfiguration is loud at CONSTRUCTION time, not hours into a run.
    for bad_lock in (0, -1, 2.5, float("nan"), "many"):
        try:
            LAPDSim1D(
                dict(dtlock_params, dt_min_lock_max_steps=bad_lock), dtlock_flags
            )
        except ValueError as error:
            assert "dt_min_lock_max_steps must be a positive integer" in str(error)
        else:
            raise AssertionError(
                f"dt_min_lock_max_steps accepted {bad_lock!r}"
            )

    # The two runs above are the only places the suite drives each half of the
    # lock's signal on its own -- (ii-c) sets the accepted flag and nothing
    # else, the forced-clamp transient the raw flag and nothing else -- so the
    # union census is asserted against THEM rather than against a second pair
    # built to the same recipe, which could drift away from these.
    return {
        "dt_min_lock_snap_result": snap_result,
        "dt_min_lock_transient_result": transient_result,
    }


# --------------------------------------------------------------------
# parallel-momentum-sink-refusals
# --------------------------------------------------------------------
@_case("parallel-momentum-sink-refusals")
def _case_parallel_momentum_sink_refusals():
    # THE RESPONSE-MAP SINK'S CONSTRUCTION-TIME REFUSALS. This term has no
    # physical owner: nothing in the model supplies its damping rate, and
    # nothing decides which part of the column sheds the momentum, so those
    # two numbers ARE the hypothesis an arm states. Neither may be defaulted,
    # guessed, or left configured-but-inert. Fifteen misconfigurations must
    # each raise a ValueError before the first step, a complete configuration
    # must construct, and the presence gate must be visible in the LEDGER --
    # the two booked rows exist only when the sink is armed.
    #
    # A deliberately tiny mesh (nx=12, no equilibration): every statement here
    # is about the configuration surface and the resolved mask, and none of it
    # is mesh-sized. Construction only, no solve.
    def _ms_config(**over):
        _p, _f = default_config()
        _p.update({"nx": 12, "max_steps_action": "stop"})
        _p["initial_neutral_state"] = "fill"
        _p.update(over)
        return _p, _f

    _ms_off_sim = LAPDSim1D(*_ms_config())
    assert _ms_off_sim._momentum_sink is None
    _ms_z = np.asarray(_ms_off_sim.geometry.z_cm, dtype=float)
    _ms_active = np.asarray(_ms_off_sim.geometry.plasma_active, dtype=bool)
    _ms_column = _ms_z[_ms_active]
    _ms_lo, _ms_hi = float(_ms_column.min()), float(_ms_column.max())
    _ms_mid = float(_ms_column[_ms_column.size // 2])
    #: The imposed damping rate [s^-1] the response map was cut at. Any
    #: positive number serves here: what these cases assert is the refusal,
    #: never the value.
    _ms_rate = 2312.88

    # (i) THE input_dict REFUSALS. Three shapes of failure: parameters
    # configured while the sink is off (inert), an armed sink missing half of
    # its statement, and a number that cannot mean what it claims to.
    for _ms_bad, _ms_needle in (
        # Configured-but-unarmed, in every combination. An arm that forgets to
        # set the flag would otherwise run as its own control and be reported
        # as a null.
        ({"parallel_momentum_sink_rate_s": _ms_rate},
         "were configured without parallel_momentum_sink"),
        ({"parallel_momentum_sink_z_start_cm": _ms_mid},
         "were configured without parallel_momentum_sink"),
        ({"parallel_momentum_sink_rate_s": _ms_rate,
          "parallel_momentum_sink_z_start_cm": _ms_mid},
         "were configured without parallel_momentum_sink"),
        # Armed with half a hypothesis, or none of it.
        ({"parallel_momentum_sink": True},
         "requires parallel_momentum_sink_rate_s"),
        ({"parallel_momentum_sink": True,
          "parallel_momentum_sink_z_start_cm": _ms_mid},
         "requires parallel_momentum_sink_rate_s"),
        ({"parallel_momentum_sink": True,
          "parallel_momentum_sink_rate_s": _ms_rate},
         "requires parallel_momentum_sink_z_start_cm"),
        # A rate that is not a rate. Zero is an UNARMED sink spelled as an
        # armed one, which is the silent no-op the config surface forbids.
        ({"parallel_momentum_sink": True,
          "parallel_momentum_sink_rate_s": 0.0,
          "parallel_momentum_sink_z_start_cm": _ms_mid},
         "must be finite and > 0"),
        ({"parallel_momentum_sink": True,
          "parallel_momentum_sink_rate_s": -_ms_rate,
          "parallel_momentum_sink_z_start_cm": _ms_mid},
         "must be finite and > 0"),
        ({"parallel_momentum_sink": True,
          "parallel_momentum_sink_rate_s": float("inf"),
          "parallel_momentum_sink_z_start_cm": _ms_mid},
         "must be finite and > 0"),
        # A position outside the column: below it the sink is not a statement
        # about a region that exists, above it the mask is empty and the arm
        # would be inert while claiming to be armed.
        ({"parallel_momentum_sink": True,
          "parallel_momentum_sink_rate_s": _ms_rate,
          "parallel_momentum_sink_z_start_cm": _ms_lo - 1.0},
         "must lie within the plasma column's axial extent"),
        ({"parallel_momentum_sink": True,
          "parallel_momentum_sink_rate_s": _ms_rate,
          "parallel_momentum_sink_z_start_cm": _ms_hi + 1.0},
         "must lie within the plasma column's axial extent"),
        ({"parallel_momentum_sink": True,
          "parallel_momentum_sink_rate_s": _ms_rate,
          "parallel_momentum_sink_z_start_cm": float("nan")},
         "parallel_momentum_sink_z_start_cm must be finite"),
        # The arming gate is a bool, not a magnitude: a truthy float here
        # reads as a rate that is silently discarded.
        ({"parallel_momentum_sink": 1.0,
          "parallel_momentum_sink_rate_s": _ms_rate,
          "parallel_momentum_sink_z_start_cm": _ms_mid},
         "parallel_momentum_sink must be a bool"),
    ):
        try:
            LAPDSim1D(*_ms_config(**_ms_bad))
        except ValueError as error:
            assert _ms_needle in str(error), (_ms_bad, str(error))
        else:
            raise AssertionError(
                f"parallel_momentum_sink accepted {_ms_bad!r}"
            )

    # (ii) THE NAMESPACE REFUSALS. Every one of these keys belongs to
    # input_dict; filed into input_flags they would be unread. The namespace
    # guard is what makes that loud, and it is asserted HERE because a
    # response-map arm is written as a command line and the arming key reads
    # like a flag.
    for _ms_flag_key, _ms_flag_value in (
        ("parallel_momentum_sink", True),
        ("parallel_momentum_sink_rate_s", _ms_rate),
    ):
        _ms_p, _ms_f = _ms_config()
        _ms_f[_ms_flag_key] = _ms_flag_value
        try:
            LAPDSim1D(_ms_p, _ms_f)
        except ValueError as error:
            assert "unknown LAPDSim1D configuration keys" in str(error), (
                _ms_flag_key, str(error)
            )
            assert _ms_flag_key in str(error), (_ms_flag_key, str(error))
        else:
            raise AssertionError(
                f"input_flags accepted the input_dict key {_ms_flag_key!r}"
            )

    # (iii) THE POSITIVE CONTROL. A complete, in-column configuration
    # constructs, and the mask it resolves to is exactly the plasma-active
    # cells at or beyond the stated position -- no boundary cell, nothing
    # below z_start.
    _ms_on_sim = LAPDSim1D(*_ms_config(
        parallel_momentum_sink=True,
        parallel_momentum_sink_rate_s=_ms_rate,
        parallel_momentum_sink_z_start_cm=_ms_mid,
    ))
    _ms_sink = _ms_on_sim._momentum_sink
    assert _ms_sink is not None
    assert _ms_sink.rate_s == _ms_rate
    assert _ms_sink.z_start_cm == _ms_mid
    _ms_mask = np.asarray(_ms_sink.cells, dtype=bool)
    assert np.array_equal(_ms_mask, _ms_active & (_ms_z >= _ms_mid))
    assert bool(_ms_active[_ms_mask].all())
    assert bool((_ms_z[_ms_mask] >= _ms_sink.z_start_cm).all())
    # A mask that reached nothing, or reached everything, would make the
    # mechanism case's off-support statement vacuous.
    _ms_on_cells = np.flatnonzero(_ms_mask)
    assert _ms_on_cells.size >= 4 and (~_ms_mask).sum() >= 4
    assert np.array_equal(
        _ms_on_cells,
        np.arange(_ms_on_cells[0], _ms_on_cells[-1] + 1),
    ), _ms_on_cells

    # (iv) THE PRESENCE GATE, read off the LEDGER. Arming the sink adds
    # exactly two named rows and removes none; with it off neither row is
    # built, so the off path cannot enter the branch and a saved trajectory
    # from an unarmed run carries the row set it always did.
    _ms_off_rows = set(_ms_off_sim.rhs_terms())
    _ms_on_rows = set(_ms_on_sim.rhs_terms())
    assert _ms_on_rows - _ms_off_rows == {
        "parallel_momentum_sink", "parallel_momentum_sink_heating"
    }, sorted(_ms_on_rows - _ms_off_rows)
    assert not _ms_off_rows - _ms_on_rows
    assert len(_ms_on_rows) == len(_ms_off_rows) + 2


# --------------------------------------------------------------------
# parallel-momentum-sink-mechanism
# --------------------------------------------------------------------
@_case("parallel-momentum-sink-mechanism")
def _case_parallel_momentum_sink_mechanism():
    # WHAT THE ARMED SINK ACTUALLY DOES, over a short march. Four statements,
    # and all four are exact rather than tolerant:
    #
    #   * the M row IS -nu_add * M on the mask and bitwise zero off it, so the
    #     instrument acts on the stated cells and nowhere else;
    #   * the Ei row IS -(M row) * u -- the PAIRWISE PARTNER of the momentum
    #     book, which is what makes the term energy-closing: the whole of the
    #     destroyed drift energy reappears as ion heating and none of it is
    #     dropped (sum-closure alone cannot see a compensated double-book);
    #   * the sink writes M and Ei and NOTHING else -- arming it moves the
    #     applied RHS in exactly those two fields, by exactly the two booked
    #     rows, and leaves every other field bitwise unchanged;
    #   * the saved ledger closes: the named rows sum to the exported total.
    #
    # Tiny mesh, 40 steps, every step saved. The physics of the response map
    # is not this case's subject -- the BOOKING is.
    def _mm_config(**over):
        _p, _f = default_config()
        _p.update({"nx": 12, "max_steps_action": "stop", "dt_save": 0.0})
        _p["initial_neutral_state"] = "fill"
        _p.update(over)
        return _p, _f

    _mm_probe = LAPDSim1D(*_mm_config())
    _mm_z = np.asarray(_mm_probe.geometry.z_cm, dtype=float)
    _mm_active = np.asarray(_mm_probe.geometry.plasma_active, dtype=bool)
    _mm_column = _mm_z[_mm_active]
    #: The stated cell range: the plasma column from its own mid-cell centre
    #: to the far end. On this fixture that is the outer half of the column,
    #: with the source-region cells left OUT so "exactly zero off it" has
    #: cells to be true on.
    _mm_z_start = float(_mm_column[_mm_column.size // 2])
    _mm_rate = 2312.88
    _mm_on_config = dict(
        parallel_momentum_sink=True,
        parallel_momentum_sink_rate_s=_mm_rate,
        parallel_momentum_sink_z_start_cm=_mm_z_start,
    )

    _mm_sim = LAPDSim1D(*_mm_config(**_mm_on_config))
    _mm_mask = np.asarray(_mm_sim._momentum_sink.cells, dtype=bool)
    assert _mm_mask.any() and not _mm_mask.all()
    _mm_result = _mm_sim.run(t_end=None, dt=None, max_steps=40)
    assert _mm_result.steps == 40
    assert len(_mm_result.time) > 1

    _mm_row_M = np.asarray(
        _mm_result.rhs_terms["parallel_momentum_sink"]["M"], dtype=float
    )
    _mm_row_Ei = np.asarray(
        _mm_result.rhs_terms["parallel_momentum_sink_heating"]["Ei"],
        dtype=float,
    )
    _mm_M = np.asarray(_mm_result.M, dtype=float)
    _mm_u = np.asarray(_mm_result.u, dtype=float)
    # The term has to have DONE something, or every identity below is a
    # statement about zeros.
    assert np.max(np.abs(_mm_row_M)) > 0.0
    assert np.max(np.abs(_mm_row_Ei)) > 0.0

    # ON the mask: -nu_add * M, at raw float64. OFF it: exactly zero, in both
    # rows -- not small, zero.
    assert np.array_equal(
        _mm_row_M[:, _mm_mask], -_mm_rate * _mm_M[:, _mm_mask]
    )
    assert np.all(_mm_row_M[:, ~_mm_mask] == 0.0)
    assert np.all(_mm_row_Ei[:, ~_mm_mask] == 0.0)
    # THE PAIRWISE PARTNER: Q_i = +nu_add * M * u = -(M row) * u, everywhere,
    # so the momentum book and the energy book are one statement.
    assert np.array_equal(_mm_row_Ei, -_mm_row_M * _mm_u)
    # The sink is a MOMENTUM term with a heating partner and nothing else: no
    # particles are born or lost, no electron energy is touched.
    for _mm_row_name in (
        "parallel_momentum_sink", "parallel_momentum_sink_heating"
    ):
        for _mm_field, _mm_values in _mm_result.rhs_terms[_mm_row_name].items():
            if _mm_row_name == "parallel_momentum_sink" and _mm_field == "M":
                continue
            if (
                _mm_row_name == "parallel_momentum_sink_heating"
                and _mm_field == "Ei"
            ):
                continue
            assert np.all(np.asarray(_mm_values, dtype=float) == 0.0), (
                _mm_row_name, _mm_field
            )

    # THE EXPORTED LEDGER CLOSES: the named rows sum to total_rhs, reported
    # row-relative as well as absolute (a misbooking that moves its own row by
    # O(1) can read as O(1e-2) throughput-normalized).
    for _mm_field, _mm_total in _mm_result.total_rhs.items():
        _mm_acc = np.zeros_like(np.asarray(_mm_total, dtype=float))
        _mm_biggest = 0.0
        for _mm_row in _mm_result.rhs_terms.values():
            if _mm_field in _mm_row:
                _mm_values = np.asarray(_mm_row[_mm_field], dtype=float)
                _mm_acc = _mm_acc + _mm_values
                _mm_biggest = max(
                    _mm_biggest, float(np.max(np.abs(_mm_values)))
                )
        _mm_resid = float(
            np.max(np.abs(_mm_acc - np.asarray(_mm_total, dtype=float)))
        )
        _mm_rel = _mm_resid / _mm_biggest if _mm_biggest > 0.0 else 0.0
        assert _mm_rel <= 1.0e-12, (_mm_field, _mm_resid, _mm_rel)

    # WHAT ARMING THE SINK MOVES, at ONE fixed state. Two freshly-constructed
    # sims -- identical but for the sink keys -- evaluate the packed RHS the
    # integrator applies at the same y. Every field except M and Ei is bitwise
    # unchanged, and so is every cell off the mask; on the mask the two moved
    # fields differ from the control by the booked rows themselves, to
    # roundoff (the rows are summed into the accumulator, so the SUMMATION
    # ORDER of the other terms moves in the last bits -- that is the only
    # reason this leg is not bit-exact).
    #
    # The residual is normalized ROW-RELATIVE, on the booked row's own scale:
    # against the whole field's scale the M row is O(1e-17) here and a
    # misbooking of the entire term would still read as passing.
    _mm_y = np.array(_mm_sim._y, dtype=float, copy=True)
    _mm_fresh_on = LAPDSim1D(*_mm_config(**_mm_on_config))
    _mm_fresh_off = LAPDSim1D(*_mm_config())
    _mm_rhs_on = _mm_fresh_on._unpack(_mm_fresh_on.rhs(y=_mm_y))
    _mm_rhs_off = _mm_fresh_off._unpack(_mm_fresh_off.rhs(y=_mm_y))
    _mm_terms_on = _mm_fresh_on.rhs_terms(y=_mm_y)
    _mm_booked = {
        "M": np.asarray(
            _mm_terms_on["parallel_momentum_sink"].M, dtype=float
        ),
        "Ei": np.asarray(
            _mm_terms_on["parallel_momentum_sink_heating"].Ei, dtype=float
        ),
    }
    for _mm_packed in dataclasses.fields(_mm_rhs_on):
        _mm_field = _mm_packed.name
        _mm_on_values = getattr(_mm_rhs_on, _mm_field)
        _mm_off_values = getattr(_mm_rhs_off, _mm_field)
        # Optional rows (the neutral-momentum, two-zone and neutral-energy
        # fields) are None on both sides at this stance; a row present on one
        # side only would be a configuration difference this case never made.
        assert (_mm_on_values is None) == (_mm_off_values is None), _mm_field
        if _mm_on_values is None:
            continue
        _mm_delta = (
            np.asarray(_mm_on_values, dtype=float)
            - np.asarray(_mm_off_values, dtype=float)
        )
        assert np.all(_mm_delta[~_mm_mask] == 0.0), _mm_field
        if _mm_field not in _mm_booked:
            assert np.all(_mm_delta == 0.0), _mm_field
            continue
        _mm_want = _mm_booked[_mm_field]
        _mm_row_scale = float(np.max(np.abs(_mm_want)))
        assert _mm_row_scale > 0.0, _mm_field
        _mm_row_rel = (
            float(np.max(np.abs(_mm_delta - _mm_want))) / _mm_row_scale
        )
        assert _mm_row_rel <= 1.0e-12, (_mm_field, _mm_row_rel)


# ----------------------------------------------------------------------
# dt-min-lock-union-summary
# ----------------------------------------------------------------------
@_case("dt-min-lock-union-summary")
def _case_dt_min_lock_union_summary(
    dt_min_lock_snap_result, dt_min_lock_transient_result
):
    # THE REPORTED CENSUS MUST BE THE SIGNAL THE GUARD COUNTS. The run loop's
    # dt_min lock fires on the OR of two disjoint per-step flags, but the
    # health summary reported only the raw one -- so the (ii-c) grind, whose
    # every step sets the accepted flag and none the raw one, read out as
    # "clamped_steps=0": a clean run, in exactly the failure mode the lock
    # exists to catch. The union is now its own field beside the two parts,
    # additive and folded into neither.
    snap = summarize_result(dt_min_lock_snap_result)
    # (i) THE GRIND THE RAW COUNT CANNOT SEE. Every step is an accepted step
    # below dt_min, no step is a raw clamp, and the union is the whole run.
    assert snap.dt_min_clamped_step_count == 0
    assert snap.below_dt_min_step_count == dt_min_lock_snap_result.steps
    assert (
        snap.dt_min_accepted_clamped_step_count
        == dt_min_lock_snap_result.steps
    )
    assert snap.dt_min_lock_step_count == dt_min_lock_snap_result.steps
    assert snap.dt_min_lock_accepted_signal_present is True
    # DISJOINTNESS IS CHECKED, NOT ASSUMED. The raw flag needs the raw bound
    # below dt_min and the accepted flag needs it strictly above, so no step
    # can set both; the census reports a violation count so that construction
    # property is a reading rather than a claim.
    assert snap.dt_min_lock_signal_overlap_count == 0

    transient = summarize_result(dt_min_lock_transient_result)
    # (ii) THE MIRROR: the forced-clamp run drives the other half alone.
    assert transient.dt_min_clamped_step_count == 5
    assert transient.dt_min_accepted_clamped_step_count == 0
    assert transient.dt_min_lock_step_count == 5
    assert transient.dt_min_lock_signal_overlap_count == 0
    for summary in (snap, transient):
        # The union covers each part and, the two being disjoint, is exactly
        # their sum.
        assert summary.dt_min_lock_step_count >= (
            summary.dt_min_clamped_step_count
        )
        assert summary.dt_min_lock_step_count >= (
            summary.dt_min_accepted_clamped_step_count
        )
        assert summary.dt_min_lock_step_count == (
            summary.dt_min_clamped_step_count
            + summary.dt_min_accepted_clamped_step_count
        )
        assert summary.dt_min_lock_signal_overlap_count == 0

    # (iii) A RESULT THAT CANNOT REPORT THE ACCEPTED HALF SAYS SO. Stripping
    # the flag off the diagnostics leaves the union readable only as the raw
    # count; the presence field is what keeps that partial census from passing
    # as the whole signal.
    stripped = SimpleNamespace(
        **{
            field: getattr(dt_min_lock_transient_result, field)
            for field in dir(dt_min_lock_transient_result)
            if not field.startswith("_") and field != "diagnostics"
        },
        diagnostics=[
            SimpleNamespace(
                **{
                    name: value
                    for name, value in dataclasses.asdict(diag).items()
                    if name != "clamped_to_dt_min_accepted"
                }
            )
            for diag in dt_min_lock_transient_result.diagnostics
        ],
    )
    stripped_summary = summarize_result(stripped)
    assert stripped_summary.dt_min_lock_accepted_signal_present is False
    assert stripped_summary.dt_min_accepted_clamped_step_count == 0
    assert (
        stripped_summary.dt_min_lock_step_count
        == stripped_summary.dt_min_clamped_step_count
        == 5
    )


# ----------------------------------------------------------------------
# kep-pressure-work-closure
# ----------------------------------------------------------------------
@_case(
    "kep-pressure-work-closure",
    provides=(
        "_kep_flare", "_kep_flat", "_kep_state", "_kep_rows", "_kep_faces",
        "_kep_closure",
    ),
)
def _case_kep_pressure_work_closure():
    """G1. The energy-consistent core closes PER CELL, in moving flow, in a flare.

    The claim the energy-consistent hyperbolic core makes is LOCAL, not
    merely global: for every cell with two open faces the hyperbolic operator's
    total-energy rate equals the net total-energy flux through that cell's two
    faces, where the discrete total-energy face flux is

        ``G_f = A_f [ F^Ee_f + F^Ei_f + 0.5{M}_f u_L u_R + Pi_f ]``,
        ``Pi_f = 0.5 (p_L u_R + u_L p_R)``,

    i.e. the flux carries the ENTHALPY, ``A u (K + E_e + E_i + p)``. The
    pressure member ``Pi`` is what the ``-p_s div u`` row pairs with, cell by
    cell, for general states and variable area; a closed-domain sum is blind to
    the difference and closes either way.

    The deciding instrument is the FOLDED rows the solver integrates --
    ``rhs_terms(...)["pressure_work"]`` and ``["hyperbolic_dissipation_heating"]``
    -- never the bare ``sources.pressure_work_rhs``, because what is at stake
    is exactly what the fold puts in the row.

    Two states on the real flared geometry: a smooth one whose velocity changes
    sign, and a seeded rough one (log-normal n, T_e, T_i; normal u). The POWER
    GUARD on each asserts the state actually exercises the variable area, so a
    vacuously satisfied closure cannot pass.
    """
    from stance_config import (  # noqa: E402
        load_configuration as _kep_load,
        without_mesh_sized_package as _kep_drop_mesh,
    )
    from cablp.solvers._sim1d.physics.flux import (  # noqa: E402
        rusanov_fluxes as _kep_rusanov,
    )

    #: The scalar neutral fill the golden runs, which stands in for the
    #: mesh-sized fill profile on the constant-area twin. Neutrals are inert in
    #: every clause below; the fill only has to be a legal one.
    _KEP_FLAT_NN0 = 2725059978765.871

    def _kep_build(flared):
        """The reference configuration, flared or with the mesh package dropped."""
        params, flags, _ = _kep_load("g1atrim")
        params, flags = dict(params), dict(flags)
        if not flared:
            params, flags = _kep_drop_mesh(params, flags)
            params, flags = dict(params), dict(flags)
            params["nn0"] = _KEP_FLAT_NN0
        return LAPDSim1D(params, flags)

    def _kep_state(sim, n, u, Te, Ti):
        """A conservative state on ``sim``'s mesh from primitive profiles."""
        m = sim.ion_mass_g
        n = np.asarray(n, dtype=float)
        u = np.asarray(u, dtype=float)
        return dataclasses.replace(
            sim.state,
            n=n.copy(),
            M=m * n * u,
            Ee=1.5 * n * np.asarray(Te, dtype=float) * ev_to_erg,
            Ei=1.5 * n * np.asarray(Ti, dtype=float) * ev_to_erg,
        )

    def _kep_rows(sim, state):
        """The FOLDED hyperbolic rows, from the solver's own term ledger."""
        terms = sim.rhs_terms(y=pack_state(state))
        zero = np.zeros(sim.geometry.cells, dtype=float)
        return {
            "adv": terms["plasma_advective_flux"],
            "geomM": (
                terms["flux_tube_geometry"].M
                if "flux_tube_geometry" in terms else zero
            ),
            "pw": terms["pressure_work"],
            "diss": terms["hyperbolic_dissipation_heating"],
            "cb": terms["characteristic_boundary"],
        }

    def _kep_faces(sim, state):
        """``G_f``: the discrete total-energy face flux, area-weighted [erg/s]."""
        geom = sim._plasma_geometry()
        cells = geom.cells
        area = np.asarray(geom.plasma_face_area_cm2, dtype=float)
        derived = derive_state(
            state, floors=sim.floors, ion_mass_g=sim.ion_mass_g
        )
        u = derived.u
        fluxes = _kep_rusanov(state, sim.floors, sim.ion_mass_g, geom)
        internal = area * (fluxes.Ee + fluxes.Ei)
        kinetic = np.zeros(cells + 1, dtype=float)
        kinetic[1:-1] = (
            area[1:-1] * 0.5 * (state.M[:-1] + state.M[1:])
            * 0.5 * u[:-1] * u[1:]
        )
        pressure = np.zeros(cells + 1, dtype=float)
        pressure[1:-1] = area[1:-1] * 0.5 * (
            u[:-1] * derived.p[1:] + derived.p[:-1] * u[1:]
        )
        closed = ~np.asarray(geom.plasma_open, dtype=bool)
        kinetic[closed] = 0.0
        pressure[closed] = 0.0
        return internal + kinetic + pressure

    def _kep_closure(sim, state, with_boundary=False):
        """Per-cell ``V d(K+Ee+Ei)/dt + G_out - G_in`` [erg/s], and the row scale.

        With ``with_boundary`` the terminating boundary operator is included;
        without it, only the hyperbolic operator is, which is what makes the
        terminating-cell reading a statement about this fix rather than a
        restatement of what the boundary row books.
        """
        geom = sim._plasma_geometry()
        volume = np.asarray(geom.plasma_volume_cm3, dtype=float)
        mass = sim.ion_mass_g
        derived = derive_state(
            state, floors=sim.floors, ion_mass_g=mass
        )
        u = derived.u
        rows = _kep_rows(sim, state)
        adv, pw, diss = rows["adv"], rows["pw"], rows["diss"]
        dM = adv.M + rows["geomM"]
        dn = np.asarray(adv.n, dtype=float)
        dEe = np.asarray(adv.Ee, dtype=float) + pw.Ee
        dEi = np.asarray(adv.Ei, dtype=float) + pw.Ei + diss.Ei
        if with_boundary:
            cb = rows["cb"]
            dM = dM + cb.M
            dn = dn + cb.n
            dEe = dEe + cb.Ee
            dEi = dEi + cb.Ei
        rate = volume * (
            u * dM - 0.5 * mass * u**2 * dn + dEe + dEi
        )
        faces = _kep_faces(sim, state)
        scale = np.abs(volume * (pw.Ee + pw.Ei))
        return rate + faces[1:] - faces[:-1], scale

    _kep_flare = _kep_build(flared=True)
    _kep_flat = _kep_build(flared=False)

    # The two meshes are what the clauses below assume: the flare has a
    # varying area and an armed quasi-1D momentum source, the twin neither.
    assert _kep_flare._variable_area_geometry
    assert not _kep_flat._variable_area_geometry
    _kep_cells = int(_kep_flare.geometry.cells)
    _kep_geom = _kep_flare._plasma_geometry()
    _kep_area = np.asarray(_kep_geom.plasma_face_area_cm2, dtype=float)
    _kep_dA = _kep_area[1:] - _kep_area[:-1]
    assert np.count_nonzero(_kep_dA) >= 20, np.count_nonzero(_kep_dA)

    # Cells with two OPEN faces: the closure below is theirs. The terminating
    # cells are G4's.
    _kep_open = np.asarray(_kep_geom.plasma_open, dtype=bool)
    _kep_interior = np.flatnonzero(_kep_open[:-1] & _kep_open[1:])
    assert _kep_interior.size > 200, _kep_interior.size

    _kep_zz = np.arange(_kep_cells, dtype=float) / _kep_cells
    _kep_smooth = _kep_state(
        _kep_flare,
        1.0e13 * (1.0 + 0.3 * np.sin(2.0 * np.pi * 3.0 * _kep_zz)),
        3.0e5 + 4.0e5 * np.cos(2.0 * np.pi * 5.0 * _kep_zz),
        4.0 + 2.0 * np.sin(2.0 * np.pi * 3.0 * _kep_zz),
        2.0 + np.cos(2.0 * np.pi * 5.0 * _kep_zz),
    )
    # The velocity really does change sign, so the smooth state is a moving
    # one in both directions rather than a one-way drift.
    _kep_u_smooth = derive_state(
        _kep_smooth, _kep_flare.floors, _kep_flare.ion_mass_g
    ).u
    assert _kep_u_smooth.max() > 0.0 > _kep_u_smooth.min()

    _kep_rng = np.random.default_rng(7)
    _kep_rough = _kep_state(
        _kep_flare,
        1.0e13 * np.exp(_kep_rng.normal(0.0, 0.5, _kep_cells)),
        _kep_rng.normal(0.0, 6.0e5, _kep_cells),
        np.exp(_kep_rng.normal(np.log(3.0), 0.6, _kep_cells)),
        np.exp(_kep_rng.normal(np.log(1.5), 0.6, _kep_cells)),
    )

    for _kep_label, _kep_st in (
        ("smooth", _kep_smooth), ("rough", _kep_rough)
    ):
        _kep_resid, _kep_scale = _kep_closure(_kep_flare, _kep_st)
        _kep_tol = 1.0e-11 * _kep_scale[_kep_interior].max()
        _kep_worst = np.abs(_kep_resid[_kep_interior]).max()
        assert _kep_worst <= _kep_tol, (_kep_label, _kep_worst, _kep_tol)
        # POWER GUARD. The non-telescoping source the folded form used to
        # carry is ``u p dA`` per cell; the closure is only meaningful where
        # that is enormous against the tolerance it is asserted at.
        _kep_dp = derive_state(
            _kep_st, _kep_flare.floors, _kep_flare.ion_mass_g
        )
        _kep_guard = np.abs(
            _kep_dp.u * _kep_dp.p * _kep_dA
        )[_kep_interior].max()
        assert _kep_guard >= 1.0e6 * _kep_tol, (
            _kep_label, _kep_guard, _kep_tol
        )

    # And the fold that makes the closure above a statement about the ROW the
    # solver integrates: with the selector armed, "pressure_work" IS
    # pressure_work_rhs, bit for bit, on the plasma-ACTIVE cells. (The term
    # ledger masks the typed plasma-dead plenum, which is a property of the
    # ledger and not of this row.) Everything the later cases read off
    # sources-level helpers rests on this.
    _kep_folded = _kep_rows(_kep_flare, _kep_smooth)["pw"]
    _kep_bare = _kep_flare.pressure_work_rhs(state=_kep_smooth)
    _kep_live = np.flatnonzero(
        np.asarray(_kep_geom.plasma_active, dtype=bool)
    )
    assert _kep_live.size == _kep_cells - 1, _kep_live.size
    for _kep_field in ("Ee", "Ei"):
        assert (
            np.asarray(getattr(_kep_folded, _kep_field))[_kep_live].tobytes()
            == np.asarray(getattr(_kep_bare, _kep_field))[_kep_live].tobytes()
        ), _kep_field
    return locals()


# ----------------------------------------------------------------------
# kep-constant-area-discriminators
# ----------------------------------------------------------------------
@_case("kep-constant-area-discriminators")
def _case_kep_constant_area_discriminators(_kep_flat, _kep_rows, _kep_state):
    """G2. Two constant-area states that tell ``-p div u`` from ``+u dz p``.

    They are the whole difference between the two forms, in a column where no
    area term can hide it:

    (A) uniform ``n`` and ``T`` with a LINEAR velocity. ``dz p_s = 0``, so
        ``+u dz p_s`` reads exactly zero while ``-p_s dz u`` is a strictly
        negative expansion cooling whose implied rate is the adiabatic
        ``-(2/3) T dz u``.
    (B) uniform ``n`` and ``u`` with a LINEAR temperature. ``dz u = 0``, so
        ``-p_s dz u`` is exactly zero while ``+u dz p_s`` reads a per-cell
        ``V u dz p_s`` that is not small.
    """
    geom = _kep_flat._plasma_geometry()
    z = np.asarray(geom.z_cm, dtype=float)
    volume = np.asarray(geom.plasma_volume_cm3, dtype=float)
    cells = int(geom.cells)
    ones = np.ones(cells, dtype=float)
    column = slice(50, 201)
    n0, T0 = 1.0e13, 5.0

    # (A) linear u.
    _g2_s = (8.0e5 - 2.0e5) / (z[-1] - z[0])
    _g2_a = _kep_state(
        _kep_flat, n0 * ones, 2.0e5 + _g2_s * (z - z[0]), T0 * ones, T0 * ones
    )
    _g2_row_a = _kep_rows(_kep_flat, _g2_a)["pw"]
    _g2_p = n0 * T0 * ev_to_erg
    _g2_want_a = -_g2_p * _g2_s
    assert _g2_want_a < 0.0
    for _g2_side in (_g2_row_a.Ee, _g2_row_a.Ei):
        _g2_got = np.asarray(_g2_side, dtype=float)[column]
        assert np.all(_g2_got < 0.0)
        assert np.allclose(_g2_got, _g2_want_a, rtol=1.0e-12, atol=0.0), (
            _g2_got.min(), _g2_got.max(), _g2_want_a
        )
    # The implied temperature rate is the adiabatic one. The row is the only
    # term that touches E_s here that the advective dilution does not, so the
    # rate is read off the row against the local n.
    _g2_dTdt = (2.0 / 3.0) * _g2_row_a.Ee[100] / (n0 * ev_to_erg)
    assert np.isclose(
        _g2_dTdt, -(2.0 / 3.0) * T0 * _g2_s, rtol=1.0e-12, atol=0.0
    ), (_g2_dTdt, -(2.0 / 3.0) * T0 * _g2_s)

    # (B) linear T at a uniform velocity.
    _g2_u0 = 5.0e5
    _g2_gT = (2.0 - 8.0) / (z[-1] - z[0])
    _g2_T = 8.0 + _g2_gT * (z - z[0])
    _g2_b = _kep_state(_kep_flat, n0 * ones, _g2_u0 * ones, _g2_T, _g2_T)
    _g2_row_b = _kep_rows(_kep_flat, _g2_b)["pw"]
    # The scale the zero is asserted against is the magnitude the OTHER form
    # would have put in this row: u p / dz, per unit volume.
    _g2_dz = float(z[101] - z[100])
    _g2_scale = _g2_u0 * (n0 * 8.0 * ev_to_erg) / _g2_dz
    for _g2_side in (_g2_row_b.Ee, _g2_row_b.Ei):
        _g2_got = np.asarray(_g2_side, dtype=float)[column]
        assert np.all(np.abs(_g2_got) <= 1.0e-12 * _g2_scale), (
            np.abs(_g2_got).max(), _g2_scale
        )
    # Non-vacuity: what the row would have carried is far above that bound.
    _g2_would = abs(_g2_u0 * n0 * _g2_gT * ev_to_erg * volume[100])
    assert _g2_would >= 1.0e6 * (1.0e-12 * _g2_scale * volume[100]), _g2_would


# ----------------------------------------------------------------------
# kep-flare-adiabat
# ----------------------------------------------------------------------
@_case("kep-flare-adiabat")
def _case_kep_flare_adiabat(_kep_flare, _kep_rows, _kep_state):
    """G3. Through a flare, at uniform pressure, the row COOLS.

    A uniform state drifting at ``u > 0`` into an expanding tube does
    ``-u p_s dA/V`` of work per unit volume -- expansion cooling, sign pinned
    -- and the implied temperature rate is the quasi-1D adiabat
    ``-(2/3) T u dA/V``. The sign is the reading: the pre-fix row carried the
    same magnitude with the opposite sign, which reads as a flare that HEATS.
    """
    geom = _kep_flare._plasma_geometry()
    area = np.asarray(geom.plasma_face_area_cm2, dtype=float)
    volume = np.asarray(geom.plasma_volume_cm3, dtype=float)
    cells = int(geom.cells)
    ones = np.ones(cells, dtype=float)
    n0, T0, u0 = 1.0e13, 5.0, 5.0e5
    dA = area[1:] - area[:-1]

    _g3_state = _kep_state(
        _kep_flare, n0 * ones, u0 * ones, T0 * ones, T0 * ones
    )
    _g3_row = _kep_rows(_kep_flare, _g3_state)["pw"]
    _g3_p = n0 * T0 * ev_to_erg
    _g3_want = -u0 * _g3_p * dA / volume

    # The expanding cells only: dA > 0 is where the adiabat cools.
    _g3_flare = np.flatnonzero(dA > 0.0)
    assert _g3_flare.size >= 20, _g3_flare.size
    for _g3_side in (_g3_row.Ee, _g3_row.Ei):
        _g3_got = np.asarray(_g3_side, dtype=float)[_g3_flare]
        assert np.all(_g3_got < 0.0), _g3_got.max()
        assert np.allclose(
            _g3_got, _g3_want[_g3_flare], rtol=1.0e-12, atol=0.0
        )
    # The implied adiabat, at the cell where the area step is largest.
    _g3_cell = int(_g3_flare[np.argmax(dA[_g3_flare])])
    _g3_dTdt = (2.0 / 3.0) * _g3_row.Ee[_g3_cell] / (n0 * ev_to_erg)
    _g3_adiabat = -(2.0 / 3.0) * T0 * u0 * dA[_g3_cell] / volume[_g3_cell]
    assert _g3_adiabat < 0.0
    assert np.isclose(_g3_dTdt, _g3_adiabat, rtol=1.0e-12, atol=0.0), (
        _g3_dTdt, _g3_adiabat
    )


# ----------------------------------------------------------------------
# kep-terminating-cells
# ----------------------------------------------------------------------
@_case("kep-terminating-cells")
def _case_kep_terminating_cells(
    _kep_flare, _kep_closure, _kep_rows, _kep_state
):
    """G4. Nothing but the boundary operator crosses a terminating face.

    At an absorbing face the advective momentum flux carries nothing, so the
    only total energy that may cross is what ``characteristic_boundary_rhs``
    books. Three clauses:

    (a) with the boundary row EXCLUDED, the hyperbolic operator's own
        total-energy rate in each terminating cell closes against that cell's
        ONE open face -- residual zero, at a guard of ``|p u A_f|`` at the
        absorbing face, which is the power the pre-fix pressure member passed
        through it unpaid;
    (b) with the boundary row INCLUDED, the whole difference between the cell's
        rate and its open-face influx IS that row's own booking, and it is a
        net sink;
    (c) a SYNTHETIC reflecting face -- closed, with a live cell, but NOT
        plasma-absorbing, which the stance geometry has none of -- carries a
        face velocity of zero, so no pressure work crosses a wall the fluid
        does not move through.
    """
    geom = _kep_flare._plasma_geometry()
    area = np.asarray(geom.plasma_face_area_cm2, dtype=float)
    volume = np.asarray(geom.plasma_volume_cm3, dtype=float)
    mass = _kep_flare.ion_mass_g
    cells = int(geom.cells)
    absorbing = np.asarray(geom.plasma_absorbing, dtype=bool)
    live = np.asarray(geom.plasma_face_live_cell, dtype=int)

    _g4_faces = np.flatnonzero(absorbing)
    assert _g4_faces.size == 2, _g4_faces
    _g4_cells = [int(live[f]) for f in _g4_faces]
    assert all(c >= 0 for c in _g4_cells), _g4_cells

    _g4_rng = np.random.default_rng(11)
    _g4_state = _kep_state(
        _kep_flare,
        1.0e13 * np.exp(_g4_rng.normal(0.0, 0.5, cells)),
        _g4_rng.normal(0.0, 6.0e5, cells),
        np.exp(_g4_rng.normal(np.log(3.0), 0.6, cells)),
        np.exp(_g4_rng.normal(np.log(1.5), 0.6, cells)),
    )
    _g4_derived = derive_state(_g4_state, _kep_flare.floors, mass)

    # (a) the hyperbolic operator alone.
    _g4_resid, _g4_scale = _kep_closure(_kep_flare, _g4_state)
    _g4_open = np.asarray(geom.plasma_open, dtype=bool)
    _g4_interior = np.flatnonzero(_g4_open[:-1] & _g4_open[1:])
    _g4_tol = 1.0e-11 * _g4_scale[_g4_interior].max()
    for _g4_face, _g4_cell in zip(_g4_faces, _g4_cells):
        assert abs(_g4_resid[_g4_cell]) <= _g4_tol, (
            _g4_cell, _g4_resid[_g4_cell], _g4_tol
        )
        _g4_guard = abs(
            _g4_derived.p[_g4_cell] * _g4_derived.u[_g4_cell] * area[_g4_face]
        )
        assert _g4_guard >= 1.0e6 * _g4_tol, (_g4_cell, _g4_guard, _g4_tol)

    # (b) with the boundary row in, the difference is that row and nothing else.
    _g4_full, _ = _kep_closure(_kep_flare, _g4_state, with_boundary=True)
    _g4_cb = _kep_rows(_kep_flare, _g4_state)["cb"]
    _g4_cb_energy = volume * (
        _g4_derived.u * _g4_cb.M
        - 0.5 * mass * _g4_derived.u**2 * _g4_cb.n
        + _g4_cb.Ee + _g4_cb.Ei
    )
    for _g4_cell in _g4_cells:
        assert abs(
            _g4_full[_g4_cell] - _g4_cb_energy[_g4_cell]
        ) <= _g4_tol, (_g4_cell, _g4_full[_g4_cell], _g4_cb_energy[_g4_cell])
        # It is a SINK, and not a small one: the terminating faces are where
        # the plasma leaves.
        assert _g4_cb_energy[_g4_cell] < -1.0e6 * _g4_tol, _g4_cell

    # (c) the synthetic reflecting face. Built by clearing ABSORBING on a face
    # that stays closed and keeps its live cell -- the one topology the
    # reference geometry does not contain.
    _g4_reflect_face = int(_g4_faces[0])
    _g4_reflect_cell = int(live[_g4_reflect_face])
    _g4_absorb = absorbing.copy()
    _g4_absorb[_g4_reflect_face] = False
    _g4_geom = dataclasses.replace(geom, plasma_absorbing=_g4_absorb)
    _g4_divu = velocity_divergence(
        _g4_state, _kep_flare.floors, mass, _g4_geom,
    )
    # A reflecting face passes no inventory: the cell's divergence is its one
    # open face alone.
    _g4_other = _g4_reflect_face + 1
    # Built with the operator's own association (face velocity, then area,
    # then the zero the reflecting face contributes) so the reading is exact
    # rather than within an ulp.
    _g4_face_u = 0.5 * (
        _g4_derived.u[_g4_reflect_cell] + _g4_derived.u[_g4_reflect_cell + 1]
    )
    _g4_want = (
        area[_g4_other] * _g4_face_u - 0.0
    ) / volume[_g4_reflect_cell]
    assert _g4_divu[_g4_reflect_cell] == _g4_want, (
        _g4_divu[_g4_reflect_cell], _g4_want
    )
    # Non-vacuity: the live cell is moving, so the absorbing rule would have
    # put a materially different number here.
    assert abs(_g4_derived.u[_g4_reflect_cell]) > 1.0e4
    _g4_absorbing_rule = (
        area[_g4_other] * _g4_face_u
        - area[_g4_reflect_face] * _g4_derived.u[_g4_reflect_cell]
    ) / volume[_g4_reflect_cell]
    assert not np.isclose(
        _g4_want, _g4_absorbing_rule, rtol=1.0e-6, atol=0.0
    ), (_g4_want, _g4_absorbing_rule)


# ----------------------------------------------------------------------
# kep-acoustic-symbol
# ----------------------------------------------------------------------
@_case("kep-acoustic-symbol")
def _case_kep_acoustic_symbol(_kep_flat, _kep_rows, _kep_state):
    """G5. The assembled operator's sound speed is the ADIABATIC one, and it
    is Galilean invariant.

    The Fourier symbol of the semi-discrete hyperbolic core on a uniform
    constant-area state must have eigen-phase-speeds ``u0 +- c_ad`` and ``u0``
    twice, with ``c_ad = sqrt((5/3)(T_e+T_i)/m_i)`` -- the same gamma the
    ``adiabatic`` wave speed the Rusanov ``a_max`` and the CFL use assumes.
    Measured at 80 cells per wavelength, at rest and in a moving frame. This is
    the only clause that sees Galilean invariance: a form whose modes are not
    ``u0 +- c`` for ANY single ``c`` fails it at ``u0 != 0`` while passing at
    rest.
    """
    geom = _kep_flat._plasma_geometry()
    z = np.asarray(geom.z_cm, dtype=float)
    cells = int(geom.cells)
    mass = _kep_flat.ion_mass_g
    n0, Te0, Ti0 = 1.0e13, 5.0, 2.0
    c_ad = math.sqrt((5.0 / 3.0) * (Te0 + Ti0) * ev_to_erg / mass)

    def _g5_rhs(fields):
        """The four hyperbolic rows, as the solver folds them."""
        n, M, Ee, Ei = fields
        state = dataclasses.replace(
            _kep_flat.state, n=n.copy(), M=M.copy(), Ee=Ee.copy(), Ei=Ei.copy()
        )
        rows = _kep_rows(_kep_flat, state)
        adv, pw, diss = rows["adv"], rows["pw"], rows["diss"]
        return np.array([
            adv.n, adv.M, adv.Ee + pw.Ee, adv.Ei + pw.Ei + diss.Ei
        ])

    window = slice(60, 220)
    k = 2.0 * math.pi / (80.0 * float(z[101] - z[100]))
    cosine, sine = np.cos(k * z), np.sin(k * z)
    scales = np.array([
        n0, mass * n0 * 1.0e6,
        1.5 * n0 * Te0 * ev_to_erg, 1.5 * n0 * Ti0 * ev_to_erg,
    ])
    for u0 in (0.0, 4.0e5):
        base = np.array([
            np.full(cells, n0), np.full(cells, mass * n0 * u0),
            np.full(cells, 1.5 * n0 * Te0 * ev_to_erg),
            np.full(cells, 1.5 * n0 * Ti0 * ev_to_erg),
        ])
        symbol = np.zeros((4, 4), dtype=complex)
        for col in range(4):
            eps = 1.0e-5 * scales[col]
            response = []
            for shape in (cosine, sine):
                bump = np.zeros((4, cells), dtype=float)
                bump[col] = eps * shape
                response.append(
                    (_g5_rhs(base + bump) - _g5_rhs(base - bump)) / (2.0 * eps)
                )
            real, imag = response
            for row in range(4):
                projected = (
                    (real[row] + 1j * imag[row])[window]
                    * np.exp(-1j * k * z[window])
                )
                symbol[row, col] = (
                    projected.mean() * scales[col] / scales[row]
                )
        speeds = np.sort(-np.linalg.eigvals(symbol).imag / k)
        want = np.sort(np.array([u0 - c_ad, u0, u0, u0 + c_ad]))
        # 0.5% of the fastest signal in the frame -- an absolute bound, since
        # the two contact modes sit at u0 = 0 in the rest frame.
        tol = 0.005 * (abs(u0) + c_ad)
        assert np.all(np.abs(speeds - want) <= tol), (u0, speeds, want, tol)


# ----------------------------------------------------------------------
# sound-speed-true-ion-mass
# ----------------------------------------------------------------------
@_case("sound-speed-true-ion-mass")
def _case_sound_speed_true_ion_mass():
    """The sound speed is sqrt(Te/m_i) on the TRUE ion mass, everywhere.

    Two clauses. (i) The fluid's ``ion_sound_speed`` is the closed form
    ``sqrt(Te [erg] / m_He)`` to round-off, and the adiabatic branch of
    ``plasma_wave_speed`` is the same construction at gamma = 5/3 on the same
    mass. (ii) The cathode circuit's own sound speed is that IDENTICAL number
    at the same Te -- not a second expression that agrees to a tolerance --
    which is what lets the circuit's ion current and the fluid's face loss be
    one book. The sheath lift rides the same mass.
    """
    from cablp.solvers._sim1d.physics.flux import (
        ion_sound_speed as _ss_cs_fn,
        plasma_wave_speed as _ss_wave,
    )
    from cablp.cathode.circuit_common import sheath_lift_lambda as _ss_lambda

    for _ss_Te in (0.05, 1.0, 4.0, 17.5, 120.0):
        _ss_want = math.sqrt(_ss_Te * ev_to_erg / m_He_cgs)
        _ss_got = float(_ss_cs_fn(_ss_Te, m_He_cgs))
        assert abs(_ss_got / _ss_want - 1.0) <= 1.0e-15, (_ss_Te, _ss_got)
        # The signal speed is the same mass with gamma = 5/3 on (Te + Ti).
        _ss_ad = float(_ss_wave(_ss_Te, 0.3 * _ss_Te, m_He_cgs))
        _ss_ad_want = math.sqrt(
            (5.0 / 3.0) * 1.3 * _ss_Te * ev_to_erg / m_He_cgs
        )
        assert abs(_ss_ad / _ss_ad_want - 1.0) <= 1.0e-15, (_ss_Te, _ss_ad)

    # The circuit reads the same spec, so the two sound speeds are the same
    # float -- asserted with == rather than a tolerance.
    _ss_dev = _cathode_solver_mod.DeviceConfig(
        A_c=706.8583470577034,
        mu=4,
        ion_mass_g=m_He_cgs,
        T_s=1910.0,
    )
    for _ss_Te in (0.05, 1.0, 4.0, 17.5, 120.0):
        _ss_circuit = float(
            _cathode_solver_idriven_mod._bohm_sound_speed(
                _ss_Te, _ss_dev.ion_mass_g
            )
        )
        assert _ss_circuit == float(_ss_cs_fn(_ss_Te, m_He_cgs)), _ss_Te

    # Lambda = ln sqrt(m_i / 2 pi m_e), on the same ion mass the collection
    # currents it throttles are built from.
    _ss_lam_want = math.log(
        math.sqrt(m_He_cgs / (2.0 * math.pi * 9.1093837015e-28))
    )
    assert abs(_ss_lambda(m_He_cgs) - _ss_lam_want) <= 1.0e-15
    assert _ss_dev.Lambda == _ss_lambda(m_He_cgs)


def _ee_sink_fixture():
    """(sim, state, geometry, floors, capacity, nu) for the substep sink cases.

    The floors are pushed to -inf on both temperatures so the substep's one
    clip site is inert: the realised-increment ledger is defined against the
    PRE-clip temperature, so the closure identity is only exact where nothing
    is clipped.
    """
    sim, snapshot = _base_sim()
    geometry = snapshot.geometry
    floors = dict(sim.floors)
    floors["Te"] = -np.inf
    floors["Ti"] = -np.inf
    # A conducting, non-uniform plateau-like column, so the conduction
    # operator is genuinely live beside the reaction term (the base
    # snapshot's quiescent 0.21 eV / 1e9 cm^-3 state has no gradient at all
    # and would leave K exactly zero).
    axis = np.linspace(0.0, 1.0, geometry.cells)
    n_profile = 1.0e12 * (1.0 + 0.4 * np.sin(2.0 * np.pi * axis))
    Te_profile = 5.0 + 2.5 * np.cos(3.0 * np.pi * axis)
    Ti_profile = 1.5 + 0.4 * np.sin(5.0 * np.pi * axis)
    state = conservative_from_primitives(
        n=n_profile,
        nn=np.full(geometry.cells, 1.0e13),
        u=np.zeros(geometry.cells),
        Te=Te_profile,
        Ti=Ti_profile,
        ion_mass_g=sim.ion_mass_g,
    )
    n_floored = np.maximum(state.n, floors["n"])
    capacity = 1.5 * n_floored * ev_to_erg
    # A two-cell profile in the anode's own shape: a strong rate on the two
    # flanking cells, exact zero everywhere else.
    nu = np.zeros(geometry.cells, dtype=float)
    pairs = anode_flanking_cells(geometry)
    assert pairs, "the base stance must resolve an anode"
    gap_side, column_side = pairs[0]
    nu[gap_side] = 3.6e5
    nu[column_side] = 3.8e5
    return sim, state, geometry, floors, capacity, nu


def _ee_sink_step(sim, state, geometry, floors, dt, scheme, picard, **kwargs):
    """One bare ``implicit_heat_conduction_step`` on the fixture's arguments."""
    return implicit_heat_conduction_step(
        state=state,
        floors=floors,
        ion_mass_g=sim.ion_mass_g,
        mu=sim.mu,
        geometry=geometry,
        dt=dt,
        implicit_heat_scheme=scheme,
        heat_picard_iterations=picard,
        heat_picard_tol=1.0e-12,
        **kwargs,
    )


# ----------------------------------------------------------------------
# implicit-ee-sink-substep-identity
# ----------------------------------------------------------------------
@_case("implicit-ee-sink-substep-identity", historical_stance=True)
def _case_implicit_ee_sink_substep_identity():
    sim, state, geometry, floors, capacity, nu = _ee_sink_fixture()
    dt = 6.0e-7
    Te_old = np.asarray(state.Ee, dtype=float) / capacity
    # Sized against the store the substep is moving, so neither the source
    # nor the sink drives the temperature through zero.
    source = (
        np.linspace(-0.05, 0.10, geometry.cells) * np.asarray(state.Ee) / dt
    )

    for scheme in ("backward_euler", "shifted", "crank_nicolson", "tr_bdf2"):
        for picard in (0, 2):
            # (a) the per-cell closure: the three realised rows sum to the
            # substep's own pre-clip electron-energy increment.
            ledger = {}
            stepped = _ee_sink_step(
                sim, state, geometry, floors, dt, scheme, picard,
                ee_source=source,
                ee_sink_rate=nu,
                ledger_out=ledger,
            )
            increment = np.asarray(stepped.Ee, dtype=float) - capacity * Te_old
            rows = (
                ledger["Ee_conduction_erg_cm3"]
                + ledger["Ee_sink_erg_cm3"]
                + ledger["Ee_source_erg_cm3"]
            )
            scale = np.maximum(
                np.abs(ledger["Ee_conduction_erg_cm3"]),
                np.maximum(
                    np.abs(ledger["Ee_sink_erg_cm3"]),
                    np.abs(ledger["Ee_source_erg_cm3"]),
                ),
            )
            scale = np.maximum(scale, np.max(np.abs(increment)))
            assert np.max(np.abs(increment - rows) / scale) < 1.0e-12, (
                scheme, picard, np.max(np.abs(increment - rows) / scale)
            )
            # The sink is a LOSS: never positive, and exactly zero off the
            # two cells the rate profile names.
            assert np.all(ledger["Ee_sink_erg_cm3"] <= 0.0), (scheme, picard)
            assert np.all(ledger["Ee_sink_erg_cm3"][nu == 0.0] == 0.0), (
                scheme, picard
            )
            assert np.any(ledger["Ee_sink_erg_cm3"] < 0.0), (scheme, picard)

            # (b) no sink handed in reproduces the historical substep byte for
            # byte, and an all-zero rate array is the same float arithmetic.
            historical = _ee_sink_step(
                sim, state, geometry, floors, dt, scheme, picard,
                ee_source=source,
            )
            none_rate = _ee_sink_step(
                sim, state, geometry, floors, dt, scheme, picard,
                ee_source=source, ee_sink_rate=None,
            )
            zero_rate = _ee_sink_step(
                sim, state, geometry, floors, dt, scheme, picard,
                ee_source=source, ee_sink_rate=np.zeros(geometry.cells),
            )
            for field in ("Ee", "Ei"):
                base_bytes = np.asarray(
                    getattr(historical, field), dtype=float
                ).tobytes()
                assert np.asarray(
                    getattr(none_rate, field), dtype=float
                ).tobytes() == base_bytes, (scheme, picard, field)
                assert np.asarray(
                    getattr(zero_rate, field), dtype=float
                ).tobytes() == base_bytes, (scheme, picard, field)
            # ... and the sink actually moved the answer, so (b) is not
            # passing because the whole term is inert.
            assert not np.array_equal(
                np.asarray(stepped.Ee, dtype=float),
                np.asarray(historical.Ee, dtype=float),
            ), (scheme, picard)

    # A negative or non-finite rate is a misconfiguration, not a source.
    for bad in (-1.0, np.nan, np.inf):
        bad_rate = nu.copy()
        bad_rate[0] = bad
        try:
            _ee_sink_step(
                sim, state, geometry, floors, dt, "tr_bdf2", 2,
                ee_sink_rate=bad_rate,
            )
        except ValueError as exc:
            assert "ee_sink_rate" in str(exc), exc
        else:
            raise AssertionError(f"ee_sink_rate={bad!r} was accepted")


# ----------------------------------------------------------------------
# implicit-ee-sink-pure-decay-exact
# ----------------------------------------------------------------------
@_case("implicit-ee-sink-pure-decay-exact", historical_stance=True)
def _case_implicit_ee_sink_pure_decay_exact():
    sim, state, geometry, floors, capacity, _nu = _ee_sink_fixture()
    Te_old = np.asarray(state.Ee, dtype=float) / capacity
    gamma = 2.0 - math.sqrt(2.0)
    implicit_weight = gamma / 2.0
    tr_a = 1.0 / (gamma * (2.0 - gamma))
    tr_b = -((1.0 - gamma) ** 2) / (gamma * (2.0 - gamma))

    def amplification(scheme, z):
        """The scheme's stability function at ``z = -nu*dt``."""
        if scheme == "tr_bdf2":
            m = -implicit_weight * z
            t_gamma = (1.0 - m) / (1.0 + m)
            return (tr_a * t_gamma + tr_b) / (1.0 + m)
        theta = {
            "backward_euler": 1.0, "shifted": 0.6, "crank_nicolson": 0.5,
        }[scheme]
        return (1.0 + (1.0 - theta) * z) / (1.0 - theta * z)

    for nu_dt in (0.05, 0.4, 1.0):
        dt = 5.0e-7
        rate = np.full(geometry.cells, nu_dt / dt, dtype=float)
        for scheme in ("backward_euler", "shifted", "crank_nicolson", "tr_bdf2"):
            ledger = {}
            stepped = _ee_sink_step(
                sim, state, geometry, floors, dt, scheme, 0,
                heat_conduction=False,
                ee_sink_rate=rate,
                ledger_out=ledger,
            )
            Te_new = np.asarray(stepped.Ee, dtype=float) / capacity
            expected = Te_old * amplification(scheme, -nu_dt)
            assert np.max(np.abs(Te_new / expected - 1.0)) < 1.0e-13, (
                scheme, nu_dt, np.max(np.abs(Te_new / expected - 1.0))
            )
            # With no conduction and no source, the realised debit IS the
            # whole increment.
            realised = -ledger["Ee_sink_erg_cm3"]
            assert np.allclose(
                realised, capacity * (Te_old - Te_new), rtol=1.0e-13, atol=0.0
            ), (scheme, nu_dt)
            assert np.all(ledger["Ee_conduction_erg_cm3"] == 0.0), (scheme, nu_dt)
            assert np.all(ledger["Ee_source_erg_cm3"] == 0.0), (scheme, nu_dt)
            # The ION solve is never handed a sink.
            assert np.allclose(
                np.asarray(stepped.Ei, dtype=float),
                np.asarray(state.Ei, dtype=float),
                rtol=1.0e-14, atol=0.0,
            ), (scheme, nu_dt)

    # Monotonicity at a rate the accuracy bound would never let through:
    # backward Euler is the one unconditionally monotone member, because the
    # enlarged operator stays an M-matrix. The second-order schemes ring
    # negative there exactly as they do on a stiff conduction mode -- asserted
    # so the distinction cannot be misread as a defect later.
    dt = 5.0e-7
    stiff = np.full(geometry.cells, 50.0 / dt, dtype=float)
    monotone = _ee_sink_step(
        sim, state, geometry, floors, dt, "backward_euler", 0,
        heat_conduction=False, ee_sink_rate=stiff,
    )
    assert np.all(np.asarray(monotone.Ee, dtype=float) > 0.0)
    assert np.all(
        np.asarray(monotone.Ee, dtype=float) < np.asarray(state.Ee, dtype=float)
    )
    for ringing in ("shifted", "crank_nicolson", "tr_bdf2"):
        rung = _ee_sink_step(
            sim, state, geometry, floors, dt, ringing, 0,
            heat_conduction=False, ee_sink_rate=stiff,
        )
        assert np.all(np.asarray(rung.Ee, dtype=float) < 0.0), ringing


# ----------------------------------------------------------------------
# implicit-ee-sink-substep-order
# ----------------------------------------------------------------------
@_case("implicit-ee-sink-substep-order", historical_stance=True)
def _case_implicit_ee_sink_substep_order():
    sim, state, geometry, floors, capacity, nu_shape = _ee_sink_fixture()
    # The refinement triplet is taken in the RESOLVED regime of BOTH terms --
    # an order read where either one is stiff is meaningless (every L-stable
    # substep reads ~1 there). The coarsest step is the state's own explicit
    # conduction bound, and the rate is scaled so nu*dt is 0.3 on that step,
    # which puts the error's argmax on a sink cell.
    bound = heat_conduction_timestep_bound(
        state=state,
        floors=floors,
        ion_mass_g=sim.ion_mass_g,
        mu=sim.mu,
        geometry=geometry,
    )
    window = 4.0 * bound
    nu = np.zeros(geometry.cells, dtype=float)
    sink_cells = np.flatnonzero(nu_shape > 0.0)
    nu[sink_cells] = 0.30 / (window / 4.0)
    # The electron heat-flux limiter is always on. The limited conduction
    # operator is second order on this fixture: against a converged
    # reference its error envelope has slope ~2 over 8 octaves of dt for
    # TR-BDF2 and Crank-Nicolson. A reference-free Richardson triplet
    # (4, 8, 16 steps) at the reference f = 0.45 is contaminated, though:
    # the sink drives the limiter's gradient through a sign change at cell
    # 7, and under the shared-midpoint conductivity the step that crosses it
    # carries a phase-dependent O(dt^2) residual, so the triplet reads an
    # erratic order at any dt. This case is about the sink, so the
    # conduction runs in the limiter's Spitzer limit, f = 1e8: at that
    # free-streaming fraction the suppression factor on this state is 1 to
    # within 5.5e-9 (against 0.55 at f = 0.45).
    spitzer_limit_f = 1.0e8

    def integrate(scheme, picard, steps):
        current = state
        sub_dt = window / steps
        for _ in range(steps):
            current = _ee_sink_step(
                sim, current, geometry, floors, sub_dt, scheme, picard,
                ee_sink_rate=nu, heat_flux_limiter_f=spitzer_limit_f,
            )
        return np.asarray(current.Ee, dtype=float) / capacity

    orders = {}
    argmax_cells = {}
    for scheme, picard in (
        ("backward_euler", 2), ("crank_nicolson", 2), ("tr_bdf2", 2),
    ):
        coarse = integrate(scheme, picard, 4)
        medium = integrate(scheme, picard, 8)
        fine = integrate(scheme, picard, 16)
        num = np.max(np.abs(coarse - medium))
        den = np.max(np.abs(medium - fine))
        orders[scheme] = math.log2(num / den)
        argmax_cells[scheme] = int(np.argmax(np.abs(coarse - medium)))
    print(
        "  implicit ee-sink substep order (dt*lambda_max = %.3f, nu*dt = 0.30): "
        % (0.25 * window / 4.0 / bound)
        + ", ".join(f"{k} {v:.2f}" for k, v in orders.items())
    )
    # The reading has to be ABOUT the sink: the refinement error must peak on
    # a cell the rate profile names.
    for scheme, cell in argmax_cells.items():
        assert cell in set(sink_cells.tolist()), (scheme, cell, sink_cells)
    assert orders["crank_nicolson"] >= 1.9, orders
    assert orders["tr_bdf2"] >= 1.9, orders
    assert 0.8 <= orders["backward_euler"] <= 1.2, orders


# ----------------------------------------------------------------------
# order-gate-envelope-fit
# ----------------------------------------------------------------------
@_case("order-gate-envelope-fit")
def _case_order_gate_envelope_fit():
    # The order gate quotes the least-squares slope of log(error) against
    # log(dt); the triplet ratio is a screen. Pinned on synthetic error
    # sequences whose slope is known exactly.
    from verify_sim1d_order import (
        DRIFT_SPAN_BOUND,
        DT_LAMBDA_FLAG,
        MIN_LEVELS,
        ORDER_SE_BOUND,
        REF_RESOLVE_FACTOR,
        drift_label,
        envelope_fit,
        pre_asymptotic_reasons,
        triplet_screens,
    )

    assert MIN_LEVELS == 4, MIN_LEVELS

    # A clean power law: the slope is returned exactly, with no spread, and
    # the triplet screen on solutions u = u* + C dt**p reads the same p.
    dts = 1.0e-8 * 2.0 ** -np.arange(6, dtype=float)
    clean = 3.0e-2 * (dts / dts[0]) ** 1.7
    fit = envelope_fit(dts, clean)
    assert abs(fit["order"] - 1.7) < 1e-12, fit
    assert fit["se"] < 1e-12 and fit["max_resid"] < 1e-12, fit
    assert np.allclose(fit["local"], 1.7, rtol=0.0, atol=1e-12), fit
    assert pre_asymptotic_reasons(fit, clean, 0.0, 1.0) == []
    assert drift_label(fit) is None, fit
    exact = np.array([1.0, 2.0])
    shape = np.array([1.0, -0.5])
    solutions = [exact + 0.1 * (dt / dts[0]) ** 1.7 * shape for dt in dts]
    assert np.allclose(triplet_screens(solutions), 1.7, rtol=0.0, atol=1e-9)

    # Fewer than MIN_LEVELS levels is refused, not fitted.
    try:
        envelope_fit(dts[: MIN_LEVELS - 1], clean[: MIN_LEVELS - 1])
    except ValueError:
        pass
    else:
        raise AssertionError("envelope_fit accepted fewer than MIN_LEVELS levels")

    # A phase-dependent wobble of +/-0.3 in log2 about slope 2. Over four
    # levels the slope's standard error exceeds the bound: PRE-ASYMPTOTIC.
    # Over nine levels the same wobble averages out -- the slope is 2 exactly
    # (the alternation is orthogonal to log dt over an odd count) and is
    # quoted -- while the triplet screens on the same sequence read erratically.
    def wobbled(levels):
        d = 1.0e-8 * 2.0 ** -np.arange(levels, dtype=float)
        sign = (-1.0) ** np.arange(levels)
        return d, (d / d[0]) ** 2 * 2.0 ** (0.3 * sign)

    d4, e4 = wobbled(4)
    fit4 = envelope_fit(d4, e4)
    # Analytic: 0.3 * sqrt(0.32) with n - 2 degrees of freedom; n - 1 and n
    # would read 0.14 and 0.12.
    assert abs(fit4["se"] - 0.3 * math.sqrt(0.32)) < 1e-9, fit4
    assert fit4["se"] > ORDER_SE_BOUND, fit4
    reasons4 = pre_asymptotic_reasons(fit4, e4, 0.0, 1.0)
    assert len(reasons4) == 1 and "standard error" in reasons4[0], reasons4
    d9, e9 = wobbled(9)
    fit9 = envelope_fit(d9, e9)
    assert abs(fit9["order"] - 2.0) < 1e-12, fit9
    assert fit9["se"] < ORDER_SE_BOUND, fit9
    assert pre_asymptotic_reasons(fit9, e9, 0.0, 1.0) == []
    # The alternating local slopes are a wobble, not a drift.
    assert drift_label(fit9) is None, fit9
    screens9 = triplet_screens([np.array([e]) for e in e9])
    assert max(screens9) - min(screens9) > 1.0, screens9

    # A monotone drift of the local slope (1.6 -> 1.4 -> 1.2, a crossover band)
    # fits with a small standard error, is not PRE-ASYMPTOTIC, and is labelled
    # DRIFTING instead of quoted.
    drift_local = np.array([1.6, 1.4, 1.2])
    e_drift = 1.0e-3 * 2.0 ** -np.concatenate(([0.0], np.cumsum(drift_local)))
    fit_drift = envelope_fit(dts[:4], e_drift)
    assert np.allclose(fit_drift["local"], drift_local, rtol=0.0, atol=1e-12)
    assert fit_drift["se"] < ORDER_SE_BOUND, fit_drift
    assert pre_asymptotic_reasons(fit_drift, e_drift, 0.0, 1.0) == []
    label = drift_label(fit_drift)
    assert label is not None and label.startswith("DRIFTING"), label

    # The span half of the rule: monotone local slopes spanning 0.10 stay an
    # ORDER, and spanning 0.25 are labelled, either side of the 0.2 bound.
    assert DRIFT_SPAN_BOUND == 0.2, DRIFT_SPAN_BOUND
    for local, drifting in (
        (np.array([1.10, 1.05, 1.00]), False),
        (np.array([1.25, 1.125, 1.00]), True),
    ):
        e_mono = 1.0e-3 * 2.0 ** -np.concatenate(([0.0], np.cumsum(local)))
        fit_mono = envelope_fit(dts[:4], e_mono)
        assert np.allclose(fit_mono["local"], local, rtol=0.0, atol=1e-12)
        assert pre_asymptotic_reasons(fit_mono, e_mono, 0.0, 1.0) == []
        mono_label = drift_label(fit_mono)
        if drifting:
            assert mono_label is not None and mono_label.startswith(
                "DRIFTING"
            ), (local, mono_label)
        else:
            assert mono_label is None, (local, mono_label)

    # The smallest level error must clear the reference's error bound by
    # REF_RESOLVE_FACTOR; below it the envelope is PRE-ASYMPTOTIC.
    floor = float(np.min(clean))
    unresolved = pre_asymptotic_reasons(
        fit, clean, 2.0 * floor / REF_RESOLVE_FACTOR, 1.0
    )
    assert len(unresolved) == 1 and "not resolved" in unresolved[0], unresolved
    assert pre_asymptotic_reasons(
        fit, clean, 0.5 * floor / REF_RESOLVE_FACTOR, 1.0
    ) == []

    # An unresolved stiff mode, and a level with no error, are PRE-ASYMPTOTIC.
    stiff = pre_asymptotic_reasons(fit, clean, 0.0, 1.5 * DT_LAMBDA_FLAG)
    assert len(stiff) == 1 and "dt*lambda_max" in stiff[0], stiff
    zeroed = clean.copy()
    zeroed[-1] = 0.0
    fit0 = envelope_fit(dts, zeroed)
    assert not np.isfinite(fit0["order"]), fit0
    assert pre_asymptotic_reasons(fit0, zeroed, 0.0, 1.0), fit0
    print(
        "  order-gate envelope: clean slope %.2f; wobbled slope %.2f +/- %.2f "
        "(4 levels, PRE-ASYMPTOTIC), %.2f +/- %.2f (9 levels, quoted); "
        "triplet screens %s"
        % (fit["order"], fit4["order"], fit4["se"], fit9["order"], fit9["se"],
           ", ".join(f"{s:.2f}" for s in screens9))
    )


# ----------------------------------------------------------------------
# dt-not-bound-by-anode-row
# ----------------------------------------------------------------------
@_case("dt-not-bound-by-anode-row")
def _case_dt_not_bound_by_anode_row():
    sim, pair = _anode_sink_sim()
    state = sim.state
    bundle = sim._plasma_source_timestep_rhs(state=state, time=sim._time)
    row = np.asarray(sim.rhs_terms()["anode_e_sheath_loss"].Ee, dtype=float)
    assert np.all(np.abs(row[pair]) > 0.0), row[pair]
    # The bundle the surface_loss bound reads no longer contains the row.
    bundle_Ee = np.asarray(bundle.Ee, dtype=float)
    with_row = bundle_Ee + row
    assert np.all(
        np.abs(bundle_Ee[pair]) < np.abs(with_row[pair])
    ), (bundle_Ee[pair], with_row[pair])
    # ... and the row's absence is exact: adding it back reproduces the
    # pre-withdrawal bundle at those cells.
    cache = sim._step_cache_snapshot()
    sim._electrode_sink_in_heat_substep = False
    try:
        legacy_bundle = np.asarray(
            sim._plasma_source_timestep_rhs(
                state=state, time=sim._time
            ).Ee,
            dtype=float,
        )
    finally:
        sim._electrode_sink_in_heat_substep = True
        sim._restore_step_cache(cache)
    assert np.allclose(legacy_bundle, with_row, rtol=1.0e-12, atol=0.0), (
        legacy_bundle[pair], with_row[pair]
    )

    # The replacement candidate exists and is finite. It is the shared
    # electrode-sink bound over BOTH implicit sink rates -- the anode's and
    # the cathode face's collected-electron climb, which joins it.
    diag = sim.suggest_timestep(include_heat_conduction=False)
    nu, _ = sim.electrode_ee_sink_rate()
    nu_climb, _ = sim.cathode_climb_ee_sink_rate()
    assert np.isfinite(diag.dt_electrode_sink_rate)
    assert np.isclose(
        diag.dt_electrode_sink_rate,
        ELECTRODE_SINK_DT_FRACTION / float(np.max(nu + nu_climb)),
        rtol=1.0e-12, atol=0.0,
    )
    # The ANODE's share of it, through the solver: with the climb's gate
    # turned off the candidate is the anode rate alone, and it is exactly the
    # anode-only formula, finite, and inert here.
    cache = sim._step_cache_snapshot()
    sim._cathode_climb_in_heat_substep = False
    try:
        anode_diag = sim.suggest_timestep(include_heat_conduction=False)
    finally:
        sim._cathode_climb_in_heat_substep = True
        sim._restore_step_cache(cache)
    assert np.isclose(
        anode_diag.dt_electrode_sink_rate,
        ELECTRODE_SINK_DT_FRACTION / float(np.max(nu)),
        rtol=1.0e-12, atol=0.0,
    )
    assert anode_diag.dt_electrode_sink_rate > anode_diag.dt, (
        anode_diag.dt_electrode_sink_rate, anode_diag.dt
    )
    assert anode_diag.active_constraint != "electrode_sink_rate"
    # ... and it BINDS on a state whose rate is scaled up past every other
    # candidate, so the candidate is not merely inert-by-construction.
    scaled = ELECTRODE_SINK_DT_FRACTION / (0.01 * diag.dt)
    bound = electrode_sink_rate_timestep(
        electrode_sink_rate=np.full(sim.geometry.cells, scaled),
    )
    assert bound < diag.dt, (bound, diag.dt)
    # With the operator split off the candidate is withdrawn entirely.
    assert electrode_sink_rate_timestep(electrode_sink_rate=None) == np.inf


# ----------------------------------------------------------------------
# implicit-ee-sink-no-solve-bit-identity
# ----------------------------------------------------------------------
@_case("implicit-ee-sink-no-solve-bit-identity")
def _case_implicit_ee_sink_no_solve_bit_identity():
    # A phase with no cathode solve hands the substep an all-zero rate, and
    # an all-zero rate is exact-zero arithmetic on every float the substep
    # touches -- so those steps are byte-identical to a build with no sink.
    params, flags = _anode_sink_config()
    params["tau_discharge"] = 0.0
    params["tau_afterglow"] = 0.0
    quiet = LAPDSim1D(params, flags)
    nu, booked_W = quiet.electrode_ee_sink_rate()
    assert np.all(nu == 0.0), nu[nu != 0.0]
    assert np.all(booked_W == 0.0)
    state = quiet.state
    with_zero = quiet.implicit_heat_conduction_step(
        dt=1.0e-10, state=state, ee_sink_rate=nu, ledger_out={},
    )
    without = quiet.implicit_heat_conduction_step(dt=1.0e-10, state=state)
    for field in ("Ee", "Ei"):
        assert np.asarray(
            getattr(with_zero, field), dtype=float
        ).tobytes() == np.asarray(
            getattr(without, field), dtype=float
        ).tobytes(), field
    # A post-drive phase reaches the same place through the phase gate.
    cold = LAPDSim1D(*_anode_sink_config())
    cold_flags_off = dict(cold._flags)
    assert cold_flags_off["Plasma"]
    late_rate, late_power = cold.electrode_ee_sink_rate(
        time=cold._time, state=cold.state,
    )
    assert late_rate.shape == (cold.geometry.cells,)
    assert np.all(np.isfinite(late_rate)) and np.all(late_rate >= 0.0)


# ----------------------------------------------------------------------
# numerics-retired-keys-refuse
# ----------------------------------------------------------------------
@_case("numerics-retired-keys-refuse")
def _case_numerics_retired_keys_refuse():
    # The numerics flags and the wave-speed selector whose one surviving
    # behaviour is now unconditional. Each is gone from both templates, is on
    # the retired register of ITS OWN namespace, and a configuration naming
    # it -- at any value, the old default included -- is refused at
    # construction with the key named as RETIRED and the refusal stating the
    # unconditional behaviour ("nothing: ..."). A retired name filed in the
    # OTHER namespace reads as the plain unknown key it is there.
    from cablp.solvers._sim1d.core.config import (
        RETIRED_FLAG_KEYS,
        RETIRED_PARAM_KEYS,
        input_dict_template_1d,
        input_flags_template_1d,
    )

    _nr_params = {
        "hyperbolic_wave_speed": ("adiabatic", "isothermal"),
    }
    _nr_flags = {
        "active_plasma_topology": (True, False),
        "hyperbolic_energy_consistent": (True, False),
        "raw_stage_validation": (True, False),
        "surface_loss_floor_exempt": (True, False),
        "electron_heat_flux_limit": (True, False),
    }
    _nr_base_p, _nr_base_f = default_config()
    for _nr_key, _nr_values in _nr_params.items():
        assert _nr_key not in input_dict_template_1d, _nr_key
        assert _nr_key not in input_flags_template_1d, _nr_key
        assert _nr_key in RETIRED_PARAM_KEYS, _nr_key
        for _nr_value in _nr_values:
            try:
                LAPDSim1D(dict(_nr_base_p, **{_nr_key: _nr_value}), _nr_base_f)
            except ValueError as _nr_exc:
                assert f"{_nr_key} is RETIRED; use nothing: " in str(
                    _nr_exc
                ), str(_nr_exc)
            else:
                raise AssertionError(
                    f"retired params key {_nr_key}={_nr_value!r} ACCEPTED"
                )
    for _nr_key, _nr_values in _nr_flags.items():
        assert _nr_key not in input_dict_template_1d, _nr_key
        assert _nr_key not in input_flags_template_1d, _nr_key
        assert _nr_key in RETIRED_FLAG_KEYS, _nr_key
        for _nr_value in _nr_values:
            try:
                LAPDSim1D(_nr_base_p, dict(_nr_base_f, **{_nr_key: _nr_value}))
            except ValueError as _nr_exc:
                assert f"{_nr_key} is RETIRED; use nothing: " in str(
                    _nr_exc
                ), str(_nr_exc)
            else:
                raise AssertionError(
                    f"retired flags key {_nr_key}={_nr_value!r} ACCEPTED"
                )
    # A retired FLAG name in params is a misfiled key, not a retired one.
    try:
        LAPDSim1D(dict(_nr_base_p, raw_stage_validation=True), _nr_base_f)
    except ValueError as _nr_exc:
        assert "unknown LAPDSim1D configuration keys" in str(_nr_exc)
        assert "RETIRED" not in str(_nr_exc), str(_nr_exc)
    else:
        raise AssertionError("a misfiled retired flag name was ACCEPTED")
    # The limiter's coefficient and the exemption's band are what stay
    # configurable, and each still refuses a value that cannot be one.
    for _nr_bad in (
        {"heat_flux_limiter_f": 0.0},
        {"heat_flux_limiter_exponent": -1.0},
    ):
        try:
            LAPDSim1D(dict(_nr_base_p, **_nr_bad), _nr_base_f)
        except ValueError as _nr_exc:
            assert next(iter(_nr_bad)) in str(_nr_exc), str(_nr_exc)
        else:
            raise AssertionError(f"{_nr_bad} ACCEPTED")


# ----------------------------------------------------------------------
# mirror-face-flux-unit
# ----------------------------------------------------------------------
def _mirror_sim(**overrides):
    """A half-column (far_end = "mirror") sim on the operator-algebra stance."""
    # Copies: the fixture hands every caller the SAME two dicts.
    params, flags = (dict(part) for part in _base_config())
    params["far_end"] = "mirror"
    params["S_pump_R"] = 0.0
    params["initial_neutral_state"] = "fill"
    params.update(overrides)
    return LAPDSim1D(params, flags)


@_case("mirror-face-flux-unit", historical_stance=True)
def _case_mirror_face_flux_unit():
    """The mirror face flux, pinned where the formula is the claim.

    (a) Against the mirror ghost (n, -M, Ee, Ei) the face kernel gives
        F_n = F_Ee = F_Ei = 0 and F_M = p_L + a_max M_L with
        a_max = |u_L| + sqrt((5/3)(Te+Ti)/m_i), exactly; at the live state
        n = 2e12 cm^-3, u = 1.5e5 cm/s, Te = 4 eV, Ti = 1 eV that is
        p_L = 16.02176634 and F_M = 19.14691966571864 (erg cm^-3). The
        pressure-only closed-face rule would give p_L, so the dissipation
        a_max M_L = 3.12515332... is what the mirror adds.
    (b) Well balanced at rest: u = 0 gives F_M = p_L exactly.
    (c) q = 0 at the mirror: the conductive face flux vanishes there against
        a temperature gradient that drives the interior faces.
    (d) No total energy crosses the mirror: in the mirror cell,
        V d(K + Ee + Ei)/dt from the hyperbolic rows (advective flux,
        pressure work and the dissipation deposit) equals the inflow G through
        its one open face -- which holds only because the deposit returns the
        a_max M_L dissipation to Ei.
    """
    from cablp.solvers._sim1d.physics.conduction import conductive_face_flux
    from cablp.solvers._sim1d.physics.flux import (
        plasma_wave_speed,
        rusanov_fluxes,
    )

    sim = _mirror_sim()
    geom = sim._plasma_geometry()
    cells = int(geom.cells)
    face = int(geom.mirror_face_indices[0])
    assert face == cells
    live = cells - 1
    mass = sim.ion_mass_g
    z = np.asarray(geom.z_cm, dtype=float)
    ramp = z / z[-1]
    n = 1.0e12 * (1.0 + ramp)
    u = 1.5e5 * ramp
    Te = 2.0 + 2.0 * ramp
    Ti = 0.5 + 0.5 * ramp
    n[live], u[live], Te[live], Ti[live] = 2.0e12, 1.5e5, 4.0, 1.0
    state = dataclasses.replace(
        sim.state,
        n=n.copy(),
        M=mass * n * u,
        Ee=1.5 * n * Te * ev_to_erg,
        Ei=1.5 * n * Ti * ev_to_erg,
    )
    derived = derive_state(state, floors=sim.floors, ion_mass_g=mass)

    # (a)
    faces = rusanov_fluxes(state, sim.floors, mass, geom)
    assert faces.n[face] == 0.0, faces.n[face]
    assert faces.Ee[face] == 0.0, faces.Ee[face]
    assert faces.Ei[face] == 0.0, faces.Ei[face]
    p_L = float(derived.p[live])
    a_max = abs(float(derived.u[live])) + float(
        plasma_wave_speed(derived.Te[live], derived.Ti[live], mass)
    )
    assert faces.M[face] == p_L + a_max * float(state.M[live]), (
        faces.M[face], p_L, a_max
    )
    assert math.isclose(p_L, 16.02176634, rel_tol=1e-12, abs_tol=0.0), p_L
    assert math.isclose(
        float(faces.M[face]), 19.14691966571864, rel_tol=1e-12, abs_tol=0.0,
    )
    assert faces.M[face] - p_L > 3.1, "the mirror dissipation is the claim"

    # (b)
    rest = dataclasses.replace(state, M=np.zeros(cells))
    rest_faces = rusanov_fluxes(rest, sim.floors, mass, geom)
    rest_p = float(derive_state(rest, sim.floors, mass).p[live])
    assert rest_faces.M[face] == rest_p, (rest_faces.M[face], rest_p)
    assert rest_faces.n[face] == 0.0

    # (c)
    q = conductive_face_flux(Te, np.full(cells, 1.0e20), geom)
    assert q[face] == 0.0, q[face]
    assert abs(q[face - 1]) > 0.0, "the interior face must carry heat"

    # (d)
    terms = sim.rhs_terms(y=pack_state(state))
    adv = terms["plasma_advective_flux"]
    pw = terms["pressure_work"]
    diss = terms["hyperbolic_dissipation_heating"]
    assert "flux_tube_geometry" not in terms  # a uniform column
    volume = np.asarray(geom.plasma_volume_cm3, dtype=float)
    area = np.asarray(geom.plasma_face_area_cm2, dtype=float)
    uu = derived.u
    rate = volume[live] * (
        uu[live] * adv.M[live]
        - 0.5 * mass * uu[live] ** 2 * adv.n[live]
        + adv.Ee[live] + pw.Ee[live]
        + adv.Ei[live] + pw.Ei[live] + diss.Ei[live]
    )
    left = face - 1
    g_in = area[left] * (
        faces.Ee[left] + faces.Ei[left]
        + 0.5 * (state.M[live - 1] + state.M[live])
        * 0.5 * uu[live - 1] * uu[live]
        + 0.5 * (uu[live - 1] * derived.p[live] + derived.p[live - 1] * uu[live])
    )
    tol = 1.0e-11 * float(np.max(np.abs(volume * (pw.Ee + pw.Ei))))
    assert abs(rate - g_in) <= tol, (rate, g_in, tol)
    # Non-vacuity: the kinetic energy the mirror dissipation removes, which
    # the deposit must return, is far above the tolerance -- a deposit blind
    # to the mirror face would leave exactly this residual.
    guard = a_max * float(state.M[live]) * float(uu[live]) * area[face]
    assert abs(guard) >= 1.0e6 * tol, (guard, tol)


# ----------------------------------------------------------------------
# mirror-fluid-march
# ----------------------------------------------------------------------
@_case("mirror-fluid-march", historical_stance=True)
def _case_mirror_fluid_march():
    """A short fluid march on the half column stays finite and conserving.

    Circuit off, fluid neutrals, the default template on the operator-algebra
    stance, a flow driven at the mirror (u0 = 2e5 cm/s toward it). With the
    puff and the pump off, nothing but the booked boundary rows exchanges
    particles, and all of them recycle: the plasma-plus-neutral inventory is
    conserved to roundoff over 200 steps, with no floor addition. The mirror
    cell's flow is braked by the face (its u falls below the cell behind it),
    and the saved result carries the mirror face in its geometry group and no
    end wall surface-power line.
    """
    sim = _mirror_sim(
        ne0=1.0e12, Te0=3.0, Ti0=1.0, u0=2.0e5,
        gas_puff_enabled=False, pump_enabled=False,
    )
    geom = sim._plasma_geometry()
    col, ann = sim._zone_volumes
    volume = np.asarray(geom.plasma_volume_cm3, dtype=float)

    def inventory(state):
        return math.fsum(
            (state.n * volume).tolist()
            + (state.nn * col).tolist()
            + (state.nn_a * ann).tolist()
        )

    before = inventory(sim.state)
    for _ in range(200):
        sim.advance_one_step()
    after = sim.state
    for name in ("n", "nn", "nn_a", "M", "Ee", "Ei"):
        assert np.all(np.isfinite(getattr(after, name))), name
    assert abs(inventory(after) - before) <= 1.0e-13 * before, (
        inventory(after), before
    )
    assert all(value == 0.0 for value in sim._floor_ledger.values()), (
        sim._floor_ledger
    )
    u_after = after.M / (sim.ion_mass_g * after.n)
    assert u_after[-1] < u_after[-2] < 2.0e5, u_after[-3:]

    # The saved result: the mirror face is written, the end wall line is not.
    run_sim = _mirror_sim(dt_save=0.0)
    result = run_sim.run(t_end=3.0e-10, dt=1.0e-10)
    assert "end_wall_surface_power_W" not in result.cathode_diagnostics
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "mirror.h5"
        run_sim.save_result(path, result)
        loaded = load_result_hdf5(path)
    assert list(loaded.mirror_face_indices) == [int(geom.cells)]
    assert abs(float(loaded.mirror_face_z_cm[0]) - 1058.9) <= 1.0e-9
    wall_params, wall_flags = _base_config()
    wall_sim = LAPDSim1D(
        dict(wall_params, dt_save=0.0, initial_neutral_state="fill"),
        wall_flags,
    )
    wall_result = wall_sim.run(t_end=3.0e-10, dt=1.0e-10)
    assert "end_wall_surface_power_W" in wall_result.cathode_diagnostics
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "wall.h5"
        wall_sim.save_result(path, wall_result)
        wall_loaded = load_result_hdf5(path)
    assert not hasattr(wall_loaded, "mirror_face_indices")
    assert not hasattr(wall_loaded, "mirror_face_z_cm")
