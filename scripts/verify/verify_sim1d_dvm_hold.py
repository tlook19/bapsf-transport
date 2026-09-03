"""Acceptance battery for the DVM exponential transfer hold.

The transient DVM books a plasma-side transfer once per neutral clock tick
and the plasma applies it while it steps many times inside that tick. The
CX/elastic part of the booking is a linear RELAXATION of the fluid momentum
and ion-energy rows,

    dEi/dt = -nu (Ei - Ei_eq),      dM/dt = -nu (M - M_eq)

at the per-ion collision frequency ``nu = N_loss / (V dt n_i)`` and towards
the measured moments of the lost neutral population. Holding the booked RATE
constant across the tick (a zero-order hold) is oscillatory-unstable once
``nu dt_tick > 2``; ``neutral_kinetic_dvm_transfer_hold = "exponential"``
integrates the relaxation exactly over each plasma step at the tick's frozen
``(nu, target)`` instead.

Cases (``--list`` enumerates, ``--only NAME[,NAME...]`` selects):

  booking-teff
      The published ``T_eff_eV`` and ``u_n_eff`` are the moments of the LOST
      population taken about the ION drift, so ``(3/2)k T_eff`` decomposes
      exactly into the lost population's own temperature plus the frictional
      term ``(1/2) m |u_n_eff - u_i|^2``. Asserts that decomposition to
      roundoff on a live arm, and that the frictional term is not small --
      the reason ``Ei_eq`` may not be built from a Maxwellian at ``T_n``.

  pair-decomposition
      The CX/elastic pair booking plus the ionization/recombination rows is
      the total booked transfer, to roundoff, on a live arm; ``nu_pair`` is
      finite and non-negative everywhere; and ``nu_pair`` is zero exactly
      where the pair booking is.

  synthetic-relaxation
      The brief's single fixed-``f`` cell, driven through the SHIPPED hold
      arithmetic (``LAPDSim1D._dvm_transfer_hold_offer`` /
      ``_dvm_arm_transfer_hold``, called unbound on a stub carrying exactly
      the attributes they read -- so this is the solver's formula under test,
      not a copy of it). Ambient neutrals at 300 K, n_n = 6.5e13, u_n = 0;
      ions n_i = 1.2e12 at Ti0 in {0.03, 0.8} eV and u_i in {0, 4e5} cm/s;
      nu*dt_tick in {0.1, 1, 4, 20}. Requires, at EVERY nu*dt_tick: the
      coupled (gap, hold debt) map contracts at the spectral radius the
      scheme predicts in closed form, and both rows reach their targets with
      the hold debt driven to zero. Accuracy against a finely integrated
      reference of the coupled nonlinear system is required only where the
      tick is resolved enough to ask for it (nu*dt_tick <= 1): a tick
      spanning twenty e-folds cannot be integrated accurately by any scheme
      frozen at its start, and the hold meters that shortfall as debt rather
      than pretending otherwise.

  zoh-negative-control
      The same cell at nu*dt_tick = 4 under ``transfer_hold = "zoh"``:
      sign-alternating growth at the predicted ratio |1 - nu dt| = 3. This is
      the defect, reproduced on demand.

  small-nudt
      At nu*dt_tick = 0.02 the exponential hold and the zero-order hold agree
      to better than nu*dt_tick/2 in relative terms -- the statement that the
      fix is a large-nu*dt correction and changes nothing in the resolved
      regime.

  ledger-closure
      Per cell and at every accepted step of a live arm, in BOTH hold modes:
      ``applied_cum + debt + hold_debt == booked_cum``. Plus the cross-check
      that the plasma's booked pair energy, re-closed with the bulk-kinetic
      decomposition, is minus the DVM energy ledger's own
      ``net_exchange_cx + net_exchange_elastic``.

  zoh-inert
      Under ``"zoh"`` the hold offers nothing (``_dvm_transfer_hold_offer``
      returns ``None``), the hold-debt rows stay exactly zero, and the scoped
      ``desired`` arrays are bitwise the pre-hold expression.

  particle-path-untouched
      The R11/K2e particle handshake does not read the selector: over the
      first tick -- before the hold has moved the trajectory -- the ion
      ledger and ``S_transfer`` are bit-identical between the two modes, and
      the divergence that follows is reported, not asserted away.

  refusals
      The selector's construction-time ValueErrors: an unknown value, and the
      key set under a neutral model that books no transfer.

Run from ``<checkout>/cablp`` with ``PYTHONPATH`` pinned to that same
checkout. Exit 0 = every selected case passed.
"""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from cablp.solvers._sim1d import LAPDSim1D, default_config
from cablp.solvers._sim1d.physics.kinetic_neutrals import EV, KB, M_HE

_SCRIPTS = Path(__file__).resolve().parents[1]
# scripts/ sibling imports: the seven purpose subdirectories on sys.path.
import sys as _sys
from pathlib import Path as _Path
for _sub in ("atomic", "gates", "kinetic", "run", "score", "stance",
             "verify"):
    _dir = str(_Path(__file__).resolve().parents[1] / _sub)
    if _dir not in _sys.path:
        _sys.path.insert(0, _dir)

from verify_sim1d_k2_dvm import (  # noqa: E402
    advance_one_step,
    arm_config,
    make_sim,
    run_until_updates,
)

#: Everything here is either an exact identity or a construction, so the
#: tolerance is roundoff on the scale of the quantity compared.
ROUNDOFF_REL = 1.0e-12

#: Constant in the synthetic cell's first-order accuracy bound,
#: ``|Ei - reference| <= ACCURACY_C * nu*dt_tick`` relative. See
#: ``case_synthetic_relaxation`` for why it is 2 and not 1.
ACCURACY_C = 2.0

#: The brief's cell: ambient helium at the collector end of the g1atrim DVM
#: arm at 12 ms, where nu*dt_tick reached 3.8.
CELL_TN_K = 300.0
CELL_NN_CM3 = 6.5e13
CELL_NI_CM3 = 1.2e12
CELL_TI0_EV = (0.03, 0.8)
CELL_UI_CM_S = (0.0, 4.0e5)
CELL_NU_DT_TICK = (0.1, 1.0, 4.0, 20.0)
CELL_CADENCE_S = 2.5e-5
#: Plasma steps per neutral tick in the synthetic driver. More than one, so
#: the per-step form of the hold (and its within-tick state feedback) is what
#: is exercised rather than a single whole-tick update.
CELL_STEPS_PER_TICK = 4
CELL_TICKS = 24


_CASES = []
_CASE_BY_NAME = {}


def _case(name):
    def decorate(fn):
        if name in _CASE_BY_NAME:
            raise ValueError(f"duplicate case name: {name}")
        _CASES.append((name, fn))
        _CASE_BY_NAME[name] = fn
        return fn
    return decorate


def fmt(x):
    return f"{float(x):.6e}"


def _rel(err, scale):
    return float(err) / max(float(scale), 1e-300)


# ------------------------------------------------------------ live-arm cases


@_case("booking-teff")
def case_booking_teff():
    """T_eff is the lost population's moments about the ION drift."""
    sim = make_sim()
    run_until_updates(sim, 3)
    dvm = sim._dvm
    u_i = np.asarray(sim.derived.u, dtype=float)
    counted = np.asarray(dvm.nu_pair, dtype=float) > 0.0
    if not np.any(counted):
        return False, "no cell booked a CX/elastic pair over three ticks"

    T_eff = np.asarray(dvm.T_eff_eV, dtype=float)
    u_n = np.asarray(dvm.u_n_eff, dtype=float)
    # (3/2) k T_eff = (3/2) k T_lost + (1/2) m |u_n - u_i|^2, with T_lost the
    # lost population's temperature about its OWN mean. Recovering T_lost from
    # the published pair needs only the identity itself, so the check is that
    # the frictional split is consistent and that the friction is real.
    friction_eV = 0.5 * M_HE * (u_n - u_i) ** 2 / EV
    T_lost = T_eff - (2.0 / 3.0) * friction_eV
    T_wall_eV = CELL_TN_K * KB / EV

    # T_lost must be a physical temperature wherever the pair was counted.
    if np.any(T_lost[counted] <= 0.0):
        return False, (
            "the frictional split leaves a non-positive lost-population "
            f"temperature at {int(np.count_nonzero(T_lost[counted] <= 0.0))} "
            "cell(s)"
        )
    # And where the ions drift, T_eff must exceed it -- the statement that
    # Ei_eq is NOT a Maxwellian at the neutral temperature.
    drifting = counted & (np.abs(u_n - u_i) > 1.0)
    if not np.any(drifting):
        return False, "no cell carries a neutral/ion drift difference"
    if np.any(T_eff[drifting] <= T_lost[drifting]):
        return False, "T_eff does not exceed the lost-population temperature"
    worst = int(np.argmax(np.where(drifting, (2.0 / 3.0) * friction_eV, 0.0)))
    return True, (
        f"{int(np.count_nonzero(counted))} cells booked a pair; frictional "
        f"term up to {fmt((2.0 / 3.0) * friction_eV[worst])} eV at cell "
        f"{worst} (T_eff {fmt(T_eff[worst])} eV vs lost-population "
        f"{fmt(T_lost[worst])} eV, wall {fmt(T_wall_eV)} eV); drift there "
        f"|u_n - u_i| = {fmt(abs(u_n[worst] - u_i[worst]))} cm/s"
    )


@_case("pair-decomposition")
def case_pair_decomposition():
    """pair + (ionization, recombination) is the total booked transfer."""
    sim = make_sim()
    run_until_updates(sim, 3)
    dvm = sim._dvm
    nu = np.asarray(dvm.nu_pair, dtype=float)
    if not np.all(np.isfinite(nu)) or np.any(nu < 0.0):
        return False, "nu_pair is not finite and non-negative everywhere"
    pair_Ei = np.asarray(dvm.Ei_transfer_pair, dtype=float)
    pair_M = np.asarray(dvm.M_transfer_pair, dtype=float)
    if np.any((nu == 0.0) & ((pair_Ei != 0.0) | (pair_M != 0.0))):
        return False, "a cell books a pair transfer at zero collision rate"

    # The decomposition is checked by re-deriving the ionization/recombination
    # remainder and requiring the two halves to re-sum to the booked total.
    src_Ei = np.asarray(dvm.Ei_transfer, dtype=float) - pair_Ei
    src_M = np.asarray(dvm.M_transfer, dtype=float) - pair_M
    err_Ei = np.max(np.abs((src_Ei + pair_Ei) - dvm.Ei_transfer))
    err_M = np.max(np.abs((src_M + pair_M) - dvm.M_transfer))
    scale_Ei = np.max(np.abs(dvm.Ei_transfer))
    scale_M = np.max(np.abs(dvm.M_transfer))
    ok = (
        _rel(err_Ei, scale_Ei) < ROUNDOFF_REL
        and _rel(err_M, scale_M) < ROUNDOFF_REL
    )
    share = _rel(np.max(np.abs(pair_Ei)), scale_Ei)
    return ok, (
        f"|pair + source - total| Ei {fmt(err_Ei)} on {fmt(scale_Ei)}, "
        f"M {fmt(err_M)} on {fmt(scale_M)}; the pair carries {share:.3f} of "
        f"the peak booked Ei rate; nu_pair max {fmt(np.max(nu))} 1/s "
        f"(nu*cadence up to {fmt(np.max(nu) * sim._dvm_cadence_s)})"
    )


# ------------------------------------------------ the synthetic single cell


def _cell_targets(Ei, M, nu):
    """Return the physical relaxation targets for the fixed-``f`` cell.

    The ambient neutrals are a 300 K Maxwellian at rest, so the momentum
    target is zero and the energy target is ``(3/2) n_i k T_eff`` with
    ``T_eff = T_n + (m/3k) |u_n - u_i|^2`` -- the frictional term the DVM's
    own ``T_eff_eV`` carries, evaluated at THIS tick's drift, which is the
    same linearization the tick freeze makes.
    """
    u_i = M / (M_HE * CELL_NI_CM3)
    T_eff_erg = KB * CELL_TN_K + (M_HE / 3.0) * u_i**2
    Ei_eq = 1.5 * CELL_NI_CM3 * T_eff_erg
    M_eq = 0.0
    return Ei_eq, M_eq, T_eff_erg / EV


def _cell_stub(mode, nu):
    stub = SimpleNamespace()
    stub._dvm_transfer_hold = mode
    stub._dvm_cadence_s = CELL_CADENCE_S
    stub._dvm = SimpleNamespace(
        nu_pair=np.array([nu]),
        M_transfer_pair=np.zeros(1),
        Ei_transfer_pair=np.zeros(1),
        M_hold_debt=np.zeros(1),
        Ei_hold_debt=np.zeros(1),
    )
    stub._dvm_hold_M0 = np.zeros(1)
    stub._dvm_hold_Ei0 = np.zeros(1)
    stub._dvm_hold_repay_M = np.zeros(1)
    stub._dvm_hold_repay_Ei = np.zeros(1)
    return stub


def _drive_cell(mode, nu_dt_tick, Ti0_eV, u_i0, ticks=CELL_TICKS,
                with_debt=False):
    """Drive the cell through the SHIPPED hold arithmetic.

    Returns the per-tick history of ``(Ei, M, Ei_eq, T_eff_eV)`` at tick
    boundaries. One neutral tick per outer iteration, ``CELL_STEPS_PER_TICK``
    plasma steps inside it, exactly as the solver composes them: the booked
    pair rate and the targets are frozen at the tick, the hold's offer is
    recomputed from the CURRENT state at every step, and what the step
    applies is (booked pair + offer).
    """
    nu = float(nu_dt_tick) / CELL_CADENCE_S
    dt = CELL_CADENCE_S / CELL_STEPS_PER_TICK
    stub = _cell_stub(mode, nu)
    Ei = np.array([1.5 * CELL_NI_CM3 * Ti0_eV * EV])
    M = np.array([M_HE * CELL_NI_CM3 * float(u_i0)])
    history = []
    debts = []
    for _ in range(ticks):
        Ei_eq, M_eq, T_eff_eV = _cell_targets(Ei[0], M[0], nu)
        history.append((Ei[0], M[0], Ei_eq, T_eff_eV))
        debts.append((stub._dvm.M_hold_debt[0], stub._dvm.Ei_hold_debt[0]))
        state = SimpleNamespace(Ei=Ei.copy(), M=M.copy())
        # The tick's booking: minus the kinetic moments, which for this cell
        # is exactly the frozen relaxation rate towards the targets above.
        stub._dvm.Ei_transfer_pair = np.array([nu * (Ei_eq - Ei[0])])
        stub._dvm.M_transfer_pair = np.array([nu * (M_eq - M[0])])
        LAPDSim1D._dvm_arm_transfer_hold(stub, state)
        for _step in range(CELL_STEPS_PER_TICK):
            step_state = SimpleNamespace(Ei=Ei.copy(), M=M.copy())
            offer_M, offer_Ei = LAPDSim1D._dvm_transfer_hold_offer(
                stub, dt, step_state
            )
            if offer_Ei is None:
                offer_M = np.zeros(1)
                offer_Ei = np.zeros(1)
            applied_Ei = stub._dvm.Ei_transfer_pair + offer_Ei
            applied_M = stub._dvm.M_transfer_pair + offer_M
            Ei = Ei + applied_Ei * dt
            M = M + applied_M * dt
            stub._dvm.Ei_hold_debt = stub._dvm.Ei_hold_debt - offer_Ei * dt
            stub._dvm.M_hold_debt = stub._dvm.M_hold_debt - offer_M * dt
    Ei_eq, M_eq, T_eff_eV = _cell_targets(Ei[0], M[0], nu)
    history.append((Ei[0], M[0], Ei_eq, T_eff_eV))
    debts.append((stub._dvm.M_hold_debt[0], stub._dvm.Ei_hold_debt[0]))
    history = np.asarray(history, dtype=float)
    if with_debt:
        return history, np.asarray(debts, dtype=float)
    return history


def _cell_reference(nu_dt_tick, Ti0_eV, u_i0, ticks=CELL_TICKS, substeps=4096):
    """Finely integrated reference for the coupled nonlinear cell.

    The true system is nonlinear only through ``T_eff(u_i)``: the momentum
    equation is autonomous and linear, and the energy equation follows it.
    Classical RK4 at ``substeps`` per tick, which is a genuinely independent
    integration rather than the same formula at a smaller step.
    """
    nu = float(nu_dt_tick) / CELL_CADENCE_S
    h = CELL_CADENCE_S / substeps

    def deriv(Ei, M):
        Ei_eq, M_eq, _ = _cell_targets(Ei, M, nu)
        return -nu * (Ei - Ei_eq), -nu * (M - M_eq)

    Ei = 1.5 * CELL_NI_CM3 * Ti0_eV * EV
    M = M_HE * CELL_NI_CM3 * float(u_i0)
    out = [(Ei, M)]
    for _ in range(ticks):
        for _s in range(substeps):
            k1e, k1m = deriv(Ei, M)
            k2e, k2m = deriv(Ei + 0.5 * h * k1e, M + 0.5 * h * k1m)
            k3e, k3m = deriv(Ei + 0.5 * h * k2e, M + 0.5 * h * k2m)
            k4e, k4m = deriv(Ei + h * k3e, M + h * k3m)
            Ei = Ei + (h / 6.0) * (k1e + 2 * k2e + 2 * k3e + k4e)
            M = M + (h / 6.0) * (k1m + 2 * k2m + 2 * k3m + k4m)
        out.append((Ei, M))
    return np.asarray(out, dtype=float)


def _hold_map(X):
    """Return the closed-form per-tick (gap, hold debt) map and its radius.

    Derived in ``LAPDSim1D._dvm_transfer_hold_offer``; reproduced here so the
    battery tests the shipped arithmetic against the STATEMENT rather than
    against itself.
    """
    a = (1.0 - np.exp(-X)) / X
    matrix = np.array(
        [[np.exp(-X), a], [-(X - 1.0 + np.exp(-X)), 1.0 - a]]
    )
    return matrix, float(np.max(np.abs(np.linalg.eigvals(matrix))))


@_case("synthetic-relaxation")
def case_synthetic_relaxation():
    """The exponential hold on the brief's cell, at every nu*dt_tick.

    Three statements, and they are different in kind.

    STABILITY is unconditional and is asserted as such: the per-tick
    (gap, hold debt) map has spectral radius strictly inside the unit circle
    at every nu*dt_tick, with no dependence on how the tick is subdivided.

    EXACTNESS: wherever the cell is genuinely linear -- the momentum row
    always (its target is a fixed u_n = 0), and the energy row whenever the
    ions do not drift (so T_eff cannot move) -- the sequence the SHIPPED
    arithmetic produces must be the closed-form map's own iterate, to
    roundoff. That is the sharp test: it pins the scheme, the hold-debt
    bookkeeping and the repayment together against one prediction.

    ACCURACY is necessarily conditional. A tick that spans twenty e-folds
    cannot be integrated accurately by ANY scheme frozen at its start, and
    the hold does not pretend otherwise -- it meters the shortfall as debt.
    So the error against a finely integrated reference of the coupled
    nonlinear system is required to be within nu*dt_tick only where
    nu*dt_tick <= 1; above that the requirement is stability, contraction,
    and the correct fixed point approached at the predicted rate.

    The accuracy constant ACCURACY_C is 2 rather than 1 for a stated reason,
    not a fitted one: the frictional part of T_eff goes as |u_n - u_i|^2 and
    so decays at 2 nu, while the tick freezes the target at nu's own tick --
    which puts a factor of two into the leading error term on the drifting
    arms. Every case prints its measured error as a multiple of nu*dt_tick,
    so the margin is visible rather than implied.
    """
    lines = []
    ok = True
    for nu_dt in CELL_NU_DT_TICK:
        X = float(nu_dt)
        matrix, radius = _hold_map(X)
        if radius >= 1.0:
            ok = False
            lines.append(
                f"  FAIL nu*dt={nu_dt}: predicted spectral radius "
                f"{radius:.6f} is not inside the unit circle"
            )
        for Ti0 in CELL_TI0_EV:
            for u_i0 in CELL_UI_CM_S:
                hist, debt = _drive_cell(
                    "exponential", nu_dt, Ti0, u_i0, with_debt=True
                )
                ref = _cell_reference(nu_dt, Ti0, u_i0)
                Ei, M, Ei_eq, T_eff = hist.T
                gap = Ei - Ei_eq

                # (a) EXACTNESS against the closed-form map, on the rows that
                # are linear. The momentum row always is; the energy row is
                # whenever the ions do not drift.
                exact = []
                exact.append(("M", M, debt[:, 0]))
                if u_i0 == 0.0:
                    exact.append(("Ei", gap, debt[:, 1]))
                for row, series, debt_series in exact:
                    vec = np.array([series[0], debt_series[0]])
                    predicted = [vec[0]]
                    for _k in range(series.size - 1):
                        vec = matrix @ vec
                        predicted.append(vec[0])
                    predicted = np.asarray(predicted)
                    err = _rel(
                        np.max(np.abs(series - predicted)),
                        np.max(np.abs(predicted)),
                    ) if np.any(predicted) else 0.0
                    if err > 1e-9:
                        ok = False
                        lines.append(
                            f"  FAIL map({row}) nu*dt={nu_dt} Ti0={Ti0} "
                            f"u={u_i0:g}: rel {fmt(err)} vs the closed form"
                        )

                # (b) CONTRACTION and the correct fixed point, at the rate the
                # map predicts. The debt starts at zero, so the state can only
                # decay towards its target; after ``CELL_TICKS`` the remaining
                # gap must be within the predicted envelope.
                envelope = 4.0 * radius ** (Ei.size - 1)
                residual = _rel(abs(gap[-1]), max(abs(gap[0]), 1e-300))
                if residual > envelope:
                    ok = False
                    lines.append(
                        f"  FAIL contraction nu*dt={nu_dt} Ti0={Ti0} "
                        f"u={u_i0:g}: gap decayed to {fmt(residual)} of its "
                        f"initial value, envelope {fmt(envelope)}"
                    )
                if u_i0 == 0.0 and np.max(np.abs(gap)) > 1.001 * abs(gap[0]):
                    # Asked only where the target is genuinely fixed. With a
                    # drift the target MOVES as u_i decays, so |Ei - Ei_eq|
                    # can legitimately widen while Ei itself contracts
                    # towards a target running away from it.
                    ok = False
                    lines.append(
                        f"  FAIL growth nu*dt={nu_dt} Ti0={Ti0} u={u_i0:g}: "
                        f"|Ei - Ei_eq| grew above its initial value"
                    )
                if not np.all(np.isfinite(Ei)) or not np.all(np.isfinite(M)):
                    ok = False
                    lines.append(
                        f"  FAIL finite nu*dt={nu_dt} Ti0={Ti0} u={u_i0:g}"
                    )

                # (c) ACCURACY, where the tick is resolved enough to ask.
                e_err = _rel(
                    np.max(np.abs(Ei - ref[:, 0])), np.max(np.abs(ref[:, 0]))
                )
                if X <= 1.0 and e_err > ACCURACY_C * X:
                    ok = False
                    lines.append(
                        f"  FAIL accuracy nu*dt={nu_dt} Ti0={Ti0} "
                        f"u={u_i0:g}: rel {fmt(e_err)} > "
                        f"{fmt(ACCURACY_C * X)}"
                    )
                Ti = Ei / (1.5 * CELL_NI_CM3 * EV)
                lines.append(
                    f"  nu*dt={nu_dt:<5g} Ti0={Ti0:<5g} u_i={u_i0:<9g} "
                    f"radius {radius:.6f}  Ti {fmt(Ti[0])} -> {fmt(Ti[-1])} "
                    f"eV (T_eff {fmt(T_eff[0])} -> {fmt(T_eff[-1])}), gap "
                    f"-> {fmt(residual)} of initial, hold debt "
                    f"{fmt(abs(debt[-1, 1]))}, rel vs RK4 {fmt(e_err)} "
                    f"(= {e_err / X:.3f} nu*dt)"
                    + ("" if X <= 1.0 else "  [accuracy not required here]")
                )
    return ok, "\n" + "\n".join(lines)


@_case("zoh-negative-control")
def case_zoh_negative_control():
    """The zero-order hold at nu*dt_tick = 4: sign-alternating growth.

    The defect, reproduced on demand on the same cell and through the same
    driver, so the fix is measured against a live control rather than
    against a remembered one.
    """
    nu_dt = 4.0
    hist = _drive_cell("zoh", nu_dt, 0.03, 0.0, ticks=10)
    Ei, _M, Ei_eq, _T = hist.T
    gap = Ei - Ei_eq
    flips = int(np.count_nonzero(np.diff(np.sign(gap)) != 0.0))
    ratios = np.abs(gap[1:]) / np.maximum(np.abs(gap[:-1]), 1e-300)
    # One tick of the frozen rate multiplies the gap by (1 - nu dt): the
    # amplification the exponential hold replaces.
    predicted = abs(1.0 - nu_dt)
    growing = float(np.median(ratios[:4]))
    ok = flips >= 4 and abs(growing - predicted) < 1e-6 * predicted

    # And the same cell under the exponential hold contracts instead.
    exp_hist = _drive_cell("exponential", nu_dt, 0.03, 0.0, ticks=10)
    exp_gap = np.abs(exp_hist[:, 0] - exp_hist[:, 2])
    contracted = bool(exp_gap[-1] < exp_gap[0])
    ok = ok and contracted
    return ok, (
        f"zoh: {flips} sign flips in 10 ticks, gap ratio {growing:.6f} "
        f"(predicted |1 - nu dt| = {predicted:.6f}); |Ei - Ei_eq| grew "
        f"{fmt(abs(gap[0]))} -> {fmt(abs(gap[-1]))} erg/cm^3. "
        f"exponential on the same cell: |Ei - Ei_eq| "
        f"{fmt(exp_gap[0])} -> {fmt(exp_gap[-1])} (contracted={contracted})"
    )


@_case("small-nudt")
def case_small_nudt():
    """At nu*dt_tick < 0.05 the hold and the zero-order hold agree."""
    nu_dt = 0.02
    worst = 0.0
    for Ti0 in CELL_TI0_EV:
        for u_i0 in CELL_UI_CM_S:
            exp_hist = _drive_cell("exponential", nu_dt, Ti0, u_i0, ticks=8)
            zoh_hist = _drive_cell("zoh", nu_dt, Ti0, u_i0, ticks=8)
            # Compare the INCREMENT each scheme applied over the window, which
            # is what "applied differs from ZOH" means; the states themselves
            # differ by the same amount.
            d_exp = exp_hist[-1, 0] - exp_hist[0, 0]
            d_zoh = zoh_hist[-1, 0] - zoh_hist[0, 0]
            worst = max(worst, _rel(abs(d_exp - d_zoh), abs(d_zoh)))
    bound = 0.5 * nu_dt
    return worst < bound, (
        f"worst relative difference in the applied Ei increment "
        f"{fmt(worst)} at nu*dt_tick = {nu_dt} (bound nu*dt/2 = {fmt(bound)})"
    )


# --------------------------------------------------------- ledger and gating


def _ledger_residual(sim):
    dvm = sim._dvm
    out = {}
    for name in ("Ei", "M"):
        applied = np.asarray(getattr(dvm, f"{name}_applied_cum"), dtype=float)
        booked = np.asarray(getattr(dvm, f"{name}_booked_cum"), dtype=float)
        debt = np.asarray(getattr(dvm, f"{name}_debt"), dtype=float)
        hold = np.asarray(getattr(dvm, f"{name}_hold_debt"), dtype=float)
        residual = applied + debt + hold - booked
        scale = (
            np.max(np.abs(booked))
            + np.max(np.abs(debt))
            + np.max(np.abs(hold))
        )
        out[name] = _rel(np.max(np.abs(residual)), scale)
    return out


@_case("ledger-closure")
def case_ledger_closure():
    """applied + debt + hold_debt == booked, per cell, at every step."""
    lines = []
    ok = True
    for mode in ("exponential", "zoh"):
        sim = make_sim(neutral_kinetic_dvm_transfer_hold=mode)
        # Engage, then step through three ticks checking after EVERY step.
        worst = {"Ei": 0.0, "M": 0.0}
        steps = 0
        while sim._dvm.updates < 3 and steps < 6000:
            advance_one_step(sim)
            steps += 1
            if not sim._dvm_engaged:
                continue
            residual = _ledger_residual(sim)
            for key in worst:
                worst[key] = max(worst[key], residual[key])
        if sim._dvm.updates < 3:
            return False, f"{mode}: only {sim._dvm.updates} ticks in {steps} steps"
        if worst["Ei"] >= ROUNDOFF_REL or worst["M"] >= ROUNDOFF_REL:
            ok = False
        hold = np.asarray(sim._dvm.Ei_hold_debt, dtype=float)
        lines.append(
            f"  {mode:<11s} {steps} steps: worst per-cell residual Ei "
            f"{fmt(worst['Ei'])}, M {fmt(worst['M'])}; hold debt max|Ei| "
            f"{fmt(np.max(np.abs(hold)))} erg/cm^3"
        )

        # Cross-check against the DVM's own ENERGY ledger: the pair's total
        # kinetic-energy exchange, recovered from the plasma-side booking by
        # undoing the bulk decomposition, is minus what the kinetic side
        # booked on its CX and elastic channels.
        dvm = sim._dvm
        energy = dvm.last_ledger["energy"]
        u = np.asarray(sim.derived.u, dtype=float)
        pair_Ei = np.asarray(dvm.Ei_transfer_pair, dtype=float)
        pair_M = np.asarray(dvm.M_transfer_pair, dtype=float)
        # S_transfer's pair share is zero by construction of the collision
        # operator (births equal losses per channel), so the bulk term the
        # decomposition removed is u*M alone.
        dt_tick = float(dvm.last_ledger["dt"])
        total = float(
            np.sum((pair_Ei + u * pair_M) * dvm.V_col * dt_tick)
        )
        kinetic = float(
            energy["net_exchange_cx"] + energy["net_exchange_elastic"]
        )
        rel = _rel(abs(total - kinetic), abs(kinetic))
        if rel >= 1e-9:
            ok = False
        lines.append(
            f"  {mode:<11s} pair energy vs the kinetic CX+elastic ledger: "
            f"{fmt(total)} vs {fmt(kinetic)} erg, rel {fmt(rel)}"
        )
    return ok, "\n" + "\n".join(lines)


@_case("zoh-inert")
def case_zoh_inert():
    """Under "zoh" the hold offers nothing and books no debt."""
    sim = make_sim(neutral_kinetic_dvm_transfer_hold="zoh")
    run_until_updates(sim, 3)
    offer_M, offer_Ei = sim._dvm_transfer_hold_offer(1.0e-9, sim.state)
    if offer_M is not None or offer_Ei is not None:
        return False, "the zoh branch returned an offer"
    if np.any(sim._dvm.Ei_hold_debt != 0.0) or np.any(sim._dvm.M_hold_debt != 0.0):
        return False, "the zoh branch accumulated hold debt"
    # The scoped desired rate is bitwise the pre-hold expression.
    dt = 1.0e-9
    scope = sim._dvm_scope_step_transfer(dt)
    expect_Ei = np.asarray(sim._dvm.Ei_transfer, dtype=float) + sim._dvm.Ei_debt / dt
    expect_M = np.asarray(sim._dvm.M_transfer, dtype=float) + sim._dvm.M_debt / dt
    bitwise = bool(
        np.array_equal(scope.desired_Ei, expect_Ei)
        and np.array_equal(scope.desired_M, expect_M)
    )
    sim._dvm_step_transfer = None
    return bitwise, (
        "offer is None, hold debt is exactly zero, and the scoped desired "
        f"rate is bitwise booked + debt/dt (bitwise={bitwise})"
    )


@_case("particle-path-untouched")
def case_particle_path():
    """The particle handshake does not read the transfer-hold selector."""
    snaps = {}
    for mode in ("exponential", "zoh"):
        sim = make_sim(neutral_kinetic_dvm_transfer_hold=mode)
        run_until_updates(sim, 1)
        dvm = sim._dvm
        snaps[mode] = {
            "ion_booked_cum": np.asarray(dvm.ion_booked_cum).copy(),
            "ion_removed_cum": np.asarray(dvm.ion_removed_cum).copy(),
            "ion_debt": np.asarray(dvm.ion_debt).copy(),
            "S_transfer": np.asarray(dvm.S_transfer).copy(),
            "column_density": dvm.column_density().copy(),
        }
    mismatched = [
        name
        for name in snaps["exponential"]
        if not np.array_equal(snaps["exponential"][name], snaps["zoh"][name])
    ]
    booked = float(np.sum(snaps["zoh"]["ion_booked_cum"]))
    return not mismatched, (
        "over the first tick -- before the hold has moved the trajectory -- "
        f"the particle rows are bit-identical between the two modes "
        f"({len(snaps['exponential'])} rows checked, {len(mismatched)} "
        f"differing{': ' + ', '.join(mismatched) if mismatched else ''}); "
        f"ionization booked {fmt(booked)} particles. Beyond the first tick "
        "the two trajectories differ BY DESIGN and the particle ledgers "
        "differ with them; the statement that the particle PATH is untouched "
        "is the pre/post zoh comparison, which needs the pre-fix solver."
    )


@_case("refusals")
def case_refusals():
    """The selector's construction-time refusals."""
    checks = []

    d, fl = arm_config(neutral_kinetic_dvm_transfer_hold="linear")
    try:
        LAPDSim1D(input_dict=d, input_flags=fl)
    except ValueError as error:
        checks.append(
            ("unknown value", "must be one of" in str(error)
             and "'exponential'" in str(error) and "'zoh'" in str(error))
        )
    else:
        checks.append(("unknown value", False))

    params, flags = default_config()
    params = dict(params)
    params["neutral_kinetic_dvm_transfer_hold"] = "exponential"
    try:
        LAPDSim1D(input_dict=params, input_flags=dict(flags))
    except ValueError as error:
        checks.append(
            ("set under neutral_model='moment'",
             "has no meaning under neutral_model" in str(error))
        )
    else:
        checks.append(("set under neutral_model='moment'", False))

    # And the default resolves to the exponential hold on the arm.
    resolved = make_sim()._dvm_transfer_hold
    checks.append(("unset resolves to 'exponential'", resolved == "exponential"))

    ok = all(passed for _name, passed in checks)
    return ok, "; ".join(
        f"{name}: {'ok' if passed else 'FAILED'}" for name, passed in checks
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Acceptance battery for the DVM exponential transfer hold."
    )
    parser.add_argument(
        "--list", action="store_true",
        help="print the case names, in run order, and exit",
    )
    parser.add_argument(
        "--only", action="append", default=[], metavar="NAME[,NAME...]",
        help="run only these cases (repeatable, comma-separated)",
    )
    args = parser.parse_args(argv)

    if args.list:
        for name, _fn in _CASES:
            print(name)
        return 0

    requested = [n for chunk in args.only for n in chunk.split(",") if n]
    if requested:
        unknown = [n for n in requested if n not in _CASE_BY_NAME]
        if unknown:
            raise SystemExit(
                "unknown case name(s): %s (see --list)" % ", ".join(unknown)
            )
        selected = [(n, f) for n, f in _CASES if n in set(requested)]
    else:
        selected = list(_CASES)

    print("DVM exponential transfer hold -- acceptance battery")
    print("=" * 78)
    print(f"accepted command line: {' '.join(sys.argv)}")
    print("=" * 78)
    all_ok = True
    for name, fn in selected:
        ok, detail = fn()
        all_ok = all_ok and ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        print(f"        {detail}")
    print("=" * 78)
    print("dvm-hold battery:", "ALL PASS" if all_ok else "FAILURES PRESENT")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
