"""PAIRWISE-PARTNER audit for a transferred-channel build, run on a live state.

WHY THIS EXISTS, AND WHY A SUM CLOSURE IS NOT ENOUGH. When a build moves a
channel from one owner to another it books a DEBIT on the old owner and a
CREDIT on the new one. A single sum identity -- "everything in equals
everything out" -- passes just as happily when BOTH halves of a pair are wrong
by the same amount: a share left in place on the old owner and planted again on
the new one closes the sum and doubles the channel. That failure mode is the
reason this instrument exists. Every transfer is therefore checked as a NAMED
PAIR: this quantity, moved off that owner, must appear on this one, and the two
are compared directly rather than through a total.

The instrument is deliberately generic. It carries a registry of PAIRS (a
partner identity between two independently computed quantities) and a registry
of CLOSURES (a conservation identity over one term's own rows and its named
leaks), evaluates them against a context built from a live solver, and prints
one table. The thread-23 directed hot surface carrier
(``cathode_jet_hot_carrier``) is its first client.

EXTENDING THIS INSTRUMENT (the shared-instrument mandate). A second build adds
its own checks WITHOUT touching anything above the registry section:

    from t23c_pairwise_audit import register_pair, register_closure

    @register_pair("wall_partition", "momentum",
                   partners=("hot-channel debit", "wall sink credit"),
                   description="...one sentence naming both owners...")
    def _pair_wall_partition(ctx):
        return debit_value, credit_value

    Return ``(left, right)`` in ONE unit, or ``None`` to declare the pair
    inapplicable to this context (it is reported as SKIP, never as a pass).

``ctx`` is the :class:`AuditContext` below: the solver, the state, the RHS
terms, the carrier's named ledger, and the geometry volumes. Add fields to it
rather than re-deriving them per pair, so two clients cannot disagree about
what state they audited.

USAGE

    python scripts/t23c_pairwise_audit.py --t-end 5e-3 --nx 60

builds the golden's own operating point (``default_config()`` + the committed
stance minus its mesh-sized package + ``nx``), arms the carrier, runs to
``--t-end``, and audits the end state. ``--tol`` sets the relative bar
(default 1e-10). Exit status is 0 only when every applicable pair and closure
passes.
"""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
if str(_SCRIPTS.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS.parent))

from baseline_sim1d import build_baseline_config  # noqa: E402
from cablp.solvers._sim1d import LAPDSim1D  # noqa: E402
from cablp.solvers._sim1d.core.state import (  # noqa: E402
    NEUTRAL_ENERGY_FLOOR_T_K,
    derive_state,
)
from cablp.solvers._sim1d.physics.sources import (  # noqa: E402
    cathode_jet_backscatter_speed,
)
from cablp.constants import ev_to_erg, kb_cgs  # noqa: E402

#: erg/s per watt.
ERG_S_PER_W = 1.0e7


# ---------------------------------------------------------------- registries

PAIR_REGISTRY = []
CLOSURE_REGISTRY = []


def register_pair(name, quantity, partners, description):
    """Register a named partner identity. See the module docstring."""

    def decorate(fn):
        PAIR_REGISTRY.append(
            SimpleNamespace(
                name=name,
                quantity=quantity,
                partners=tuple(partners),
                description=description,
                evaluate=fn,
            )
        )
        return fn

    return decorate


def register_closure(name, quantity, description):
    """Register a conservation identity over one term's own rows and leaks."""

    def decorate(fn):
        CLOSURE_REGISTRY.append(
            SimpleNamespace(
                name=name,
                quantity=quantity,
                description=description,
                evaluate=fn,
            )
        )
        return fn

    return decorate


class AuditContext(SimpleNamespace):
    """The evaluated state every registered check reads. Built once."""


def build_context(sim):
    """Return the :class:`AuditContext` for a solver's CURRENT state."""
    state = sim.state
    geometry = sim.geometry
    derived = derive_state(state, sim.floors, sim.ion_mass_g)
    Vp = np.asarray(geometry.plasma_volume_cm3, dtype=float)
    Vm = np.asarray(geometry.neutral_volume_cm3, dtype=float)
    V_nn = Vp if state.nn_a is not None else Vm
    V_Mn = Vp if state.M_n_a is not None else Vm
    V_ann = np.maximum(Vm - Vp, 1e-300)

    # ONE cathode solve feeds every term below. The boundary term is then
    # evaluated BOTH ways against it -- the historical path and the armed one
    # -- which is what makes the withholding pairs a comparison of two
    # independently produced numbers rather than a restatement of one.
    solve = sim.solve_cathode_boundary(state=state, update_cache=False)
    spec = sim._cathode_jet_spec(solve)
    carrier_out = {}
    if sim._characteristic_boundary:
        boundary = sim.characteristic_boundary_rhs
    else:
        boundary = sim.boundary_absorption_rhs
    boundary_on = boundary(
        state=state, cathode_solve=solve, carrier_out=carrier_out
    )
    boundary_off = boundary(state=state, cathode_solve=solve)
    launch_per_s = carrier_out.get("launch_per_s")
    jet_energy_on = sim.cathode_jet_neutral_energy_rhs(
        state=state, cathode_solve=solve, recycle_nn_row=boundary_on.nn
    )
    # The per-neutral ionization frequency the bulk reaction term is using --
    # the same array the solver threads into the carrier on a live step.
    reaction = sim.reaction_rhs_terms(state=state)
    ionization_rate = np.asarray(
        reaction["ionization_birth"].n, dtype=float
    ) / np.maximum(np.asarray(state.nn, dtype=float), sim.floors["nn"])
    carrier_term = sim.cathode_jet_hot_carrier_rhs(
        state=state,
        cathode_solve=solve,
        launch_per_s=launch_per_s,
        ionization_rate=ionization_rate,
    )
    ledger = dict(sim._jet_carrier_diagnostics)
    return AuditContext(
        sim=sim,
        state=state,
        geometry=geometry,
        derived=derived,
        Vp=Vp,
        Vm=Vm,
        V_nn=V_nn,
        V_Mn=V_Mn,
        V_ann=V_ann,
        carrier_term=carrier_term,
        ledger=ledger,
        cathode_solve=solve,
        cathode_jet_spec=spec,
        boundary_on=boundary_on,
        boundary_off=boundary_off,
        jet_energy_on=jet_energy_on,
        launch_per_s=launch_per_s,
        cathode_mask=np.asarray(geometry.cell_role) == "cathode",
        plasma_active=np.asarray(geometry.plasma_active, dtype=bool),
    )


# ----------------------------------------------------------- carrier helpers


def _row_total(row, volume):
    """Volume-integrate one RHS row; ``None`` (an absent field) is zero."""
    if row is None:
        return 0.0
    return float(np.sum(np.asarray(row, dtype=float) * volume))


def _v1_jet_excess_off(ctx):
    """Return the v1 ``cathode_jet_neutral_energy`` EXCESS row total [erg/s].

    Rebuilt from the DOCUMENTED v1 formula rather than read back off the code
    this build changed, so the pair below compares two independent statements
    of the same quantity: per recycled particle the cathode jet delivers
    ``R_N (1/2) m v_back^2 + (1 - R_N) (3/2) k T_s``, of which the generic
    surface booking has already granted ``(3/2) k T_wall``, and the term
    supplies the difference on the cathode cells alone.
    """
    spec = ctx.cathode_jet_spec
    R_N = float(spec["R_N"])
    v_back = cathode_jet_backscatter_speed(
        spec, ctx.derived.Ti, ctx.sim.ion_mass_g
    )
    e_jet = R_N * 0.5 * ctx.sim.ion_mass_g * v_back**2 + (1.0 - R_N) * (
        1.5 * kb_cgs * max(float(spec["T_s_K"]), 0.0)
    )
    excess = e_jet - 1.5 * kb_cgs * NEUTRAL_ENERGY_FLOOR_T_K
    recycle = np.maximum(np.asarray(ctx.boundary_off.nn, dtype=float), 0.0)
    return float(
        np.sum(np.where(ctx.cathode_mask, recycle * excess, 0.0) * ctx.V_nn)
    )


# --------------------------------------------------------------- the pairs


@register_pair(
    "surface_debit_vs_carrier_income",
    "energy per recycled particle [erg]",
    partners=("cathode surface debit", "carrier launch energy"),
    description=(
        "The energy the cathode surface gives up per recycled particle "
        "(cathode_jet_surface_debit's R_E share of the incident ion energy) "
        "and the energy the carrier's R_N launched atoms leave with. Under "
        "'total_reflected' these are the same number; under 'legacy' the "
        "carrier receives R_N times it, which is the pre-existing convention "
        "shortfall the smoke suite already documents."
    ),
)
def _pair_surface_debit(ctx):
    spec = ctx.cathode_jet_spec
    if spec is None:
        return None
    R_N = float(spec["R_N"])
    R_E = float(spec["R_E"])
    convention = spec.get("energy_convention", "legacy")
    share = R_E if convention == "total_reflected" else R_N * R_E
    cell = int(np.flatnonzero(ctx.cathode_mask)[0])
    Ti = float(ctx.derived.Ti[cell])
    v_back = float(
        cathode_jet_backscatter_speed(spec, Ti, ctx.sim.ion_mass_g)
    )
    income = R_N * 0.5 * ctx.sim.ion_mass_g * v_back**2
    debit = share * (float(spec["phi_c_V"]) + Ti) * ev_to_erg
    return debit, income


@register_pair(
    "cell1_nn_withholding_vs_launch",
    "particles [s^-1]",
    partners=("cathode-cell nn rebirth withheld", "carrier launch rate"),
    description=(
        "The neutral rebirth the boundary term stops booking at the cathode "
        "cell when the carrier is armed, against the rate the carrier "
        "launches. Evaluated by calling the boundary term BOTH ways on the "
        "same state and the same cathode solve, so the left side is the "
        "historical code path rather than a restatement of the new one."
    ),
)
def _pair_nn_withholding(ctx):
    if ctx.launch_per_s is None:
        return None
    withheld = _row_total(ctx.boundary_off.nn, ctx.V_nn) - _row_total(
        ctx.boundary_on.nn, ctx.V_nn
    )
    return withheld, float(np.sum(ctx.launch_per_s))


@register_pair(
    "cell1_En_withholding_vs_launch",
    "power [W]",
    partners=("cathode-cell En booking withheld", "carrier launch power"),
    description=(
        "The neutral ENERGY the v1 pair (the surface booking's wall credit "
        "plus cathode_jet_neutral_energy's excess) would have planted in the "
        "cathode cell, less what it still plants with the carrier armed, "
        "against the carrier's launch power. The v1 side is rebuilt from the "
        "documented formula, not read off the changed code."
    ),
)
def _pair_En_withholding(ctx):
    if ctx.launch_per_s is None or ctx.cathode_jet_spec is None:
        return None
    wall = 1.5 * kb_cgs * NEUTRAL_ENERGY_FLOOR_T_K
    v1_off = _v1_jet_excess_off(ctx) + wall * _row_total(
        np.maximum(np.asarray(ctx.boundary_off.nn, dtype=float), 0.0),
        ctx.V_nn,
    )
    v1_on = _row_total(ctx.jet_energy_on.En, ctx.V_nn) + wall * _row_total(
        np.maximum(np.asarray(ctx.boundary_on.nn, dtype=float), 0.0),
        ctx.V_nn,
    )
    return (v1_off - v1_on) / ERG_S_PER_W, ctx.ledger["launch_W"]


@register_pair(
    "jet_Mn_withholding_vs_launch",
    "momentum [dyn]",
    partners=("jet_M_n backscatter share withheld", "carrier launch momentum"),
    description=(
        "The R_N v_back share of the boundary term's directed jet momentum, "
        "withheld when the carrier is armed (the implanted (1 - R_N) v_eff "
        "effusive share stays behind), against the momentum the carrier "
        "launches. Same OFF/ON evaluation as the particle pair."
    ),
)
def _pair_Mn_withholding(ctx):
    if ctx.launch_per_s is None or ctx.boundary_on.M_n is None:
        return None
    withheld = _row_total(ctx.boundary_off.M_n, ctx.V_Mn) - _row_total(
        ctx.boundary_on.M_n, ctx.V_Mn
    )
    return withheld, ctx.ledger["launch_dyn"]


@register_pair(
    "ion_cx_swap_momentum",
    "momentum [dyn]",
    partners=("jet-in minus partner-out", "carrier ion momentum row"),
    description=(
        "A charge exchange is a swap: the fast atom's momentum arrives on the "
        "ions and the exchanged ion's leaves with the partner atom. The net "
        "the ion row may keep is the difference, and the partner's share must "
        "appear on M_n -- booking only one side closes a sum while violating "
        "conservation, which is precisely what a pairwise check catches."
    ),
)
def _pair_cx_momentum(ctx):
    if ctx.carrier_term is None or "partner_exchange_dyn" not in ctx.ledger:
        return None
    jet_in_minus_partner_out = (
        ctx.ledger["partner_exchange_dyn"]
        + ctx.ledger["jet_ionization_dyn"]
    )
    return jet_in_minus_partner_out, _row_total(ctx.carrier_term.M, ctx.Vp)


@register_pair(
    "cx_partner_En_credit_vs_Ei_debit",
    "power [W]",
    partners=("Ei debit (3/2 k Ti per event)", "En partner credit"),
    description=(
        "The CX partner atom is born at the LOCAL ION STATE, so the ions are "
        "debited (3/2) k Ti per event and the cold gas is credited the same. "
        "The debit is read as the energy the beam carried in less what the "
        "ion row kept; the credit as the En row less its wall-temperature "
        "return. The two are built from different rows on purpose."
    ),
)
def _pair_partner_En(ctx):
    if ctx.carrier_term is None or ctx.carrier_term.En is None:
        return None
    carried = (
        ctx.ledger["partner_exchange_carried_W"]
        + ctx.ledger["jet_ionization_W"]
    )
    debit = carried - _row_total(ctx.carrier_term.Ei, ctx.Vp) / ERG_S_PER_W
    credit = (
        _row_total(ctx.carrier_term.En, ctx.V_nn) / ERG_S_PER_W
        - ctx.ledger["column_return_W"]
    )
    return debit, credit


# ------------------------------------------------------------ the closures


@register_closure(
    "carrier_particles",
    "particles [s^-1]",
    "Every launched atom ends in exactly one fate, and every fate deposits "
    "its particle on a row: the ionization row, the cold gas, or the annulus.",
)
def _closure_particles(ctx):
    if ctx.carrier_term is None or ctx.launch_per_s is None:
        return None
    launched = float(np.sum(ctx.launch_per_s))
    fates = (
        ctx.ledger["partner_exchange_per_s"]
        + ctx.ledger["jet_ionization_per_s"]
        + ctx.ledger["wall_leak_per_s"]
        + ctx.ledger["end_leak_per_s"]
        + ctx.ledger["mesh_cull_per_s"]
    )
    rows = (
        _row_total(ctx.carrier_term.n, ctx.Vp)
        + _row_total(ctx.carrier_term.nn, ctx.V_nn)
        + _row_total(ctx.carrier_term.nn_a, ctx.V_ann)
    )
    return [
        ("launch", launched),
        ("named fates", fates),
        ("deposited rows", rows),
    ]


@register_closure(
    "carrier_energy",
    "power [W]",
    "The launch power is spent on the ion row, the neutral energy row and the "
    "three named leaks. The electron binding cost is NOT part of this "
    "identity: it is the standard ADAS cost the bulk channel pays too, booked "
    "on Ee exactly as ionization_energy_cost books it.",
)
def _closure_energy(ctx):
    if ctx.carrier_term is None:
        return None
    spent = (
        _row_total(ctx.carrier_term.Ei, ctx.Vp) / ERG_S_PER_W
        + _row_total(ctx.carrier_term.En, ctx.V_nn) / ERG_S_PER_W
        + ctx.ledger["wall_leak_W"]
        + ctx.ledger["end_leak_W"]
        + ctx.ledger["mesh_cull_W"]
    )
    return [("launch", ctx.ledger["launch_W"]), ("spent", spent)]


@register_closure(
    "carrier_momentum",
    "momentum [dyn]",
    "The launched directed momentum arrives on the ion row, on the neutral "
    "momentum row (the CX partners' share) or on a surface with a leaked "
    "atom.",
)
def _closure_momentum(ctx):
    if ctx.carrier_term is None:
        return None
    spent = (
        _row_total(ctx.carrier_term.M, ctx.Vp)
        + _row_total(ctx.carrier_term.M_n, ctx.V_Mn)
        + ctx.ledger["leak_dyn"]
    )
    return [("launch", ctx.ledger["launch_dyn"]), ("spent", spent)]


@register_closure(
    "carrier_topology",
    "particles [s^-1]",
    "Nothing the carrier deposits may land on a plasma-dead cell. The "
    "caller's topology mask would DELETE such a deposit, turning a leak the "
    "ledger reports into a silent particle loss it does not, so the beam is "
    "confined to its own plasma segment and this is the check on it.",
)
def _closure_topology(ctx):
    if ctx.carrier_term is None:
        return None
    rows = (
        _row_total(ctx.carrier_term.n, ctx.Vp)
        + _row_total(ctx.carrier_term.nn, ctx.V_nn)
        + _row_total(ctx.carrier_term.nn_a, ctx.V_ann)
    )
    live = ctx.plasma_active.astype(float)
    masked = (
        _row_total(ctx.carrier_term.n, ctx.Vp * live)
        + _row_total(ctx.carrier_term.nn, ctx.V_nn * live)
        + _row_total(ctx.carrier_term.nn_a, ctx.V_ann * live)
    )
    return [("all cells", rows), ("plasma-active cells", masked)]


# ------------------------------------------------------------------ running


def _relative(values):
    """Return the worst pairwise relative spread of a list of numbers."""
    scale = max(abs(v) for _, v in values)
    if scale == 0.0:
        return 0.0
    lo = min(v for _, v in values)
    hi = max(v for _, v in values)
    return abs(hi - lo) / scale


def run_audit(ctx, tol, stream=sys.stdout):
    """Evaluate every registered pair and closure; return ``True`` on pass."""
    ok = True
    print("=" * 78, file=stream)
    print("PAIRWISE-PARTNER AUDIT", file=stream)
    print("=" * 78, file=stream)
    print(f"  relative tolerance   {tol:.1e}", file=stream)
    print(file=stream)
    for pair in PAIR_REGISTRY:
        result = pair.evaluate(ctx)
        print(f"[pair] {pair.name}   ({pair.quantity})", file=stream)
        print(f"       {pair.partners[0]}  <->  {pair.partners[1]}",
              file=stream)
        if result is None:
            print("       SKIP -- not applicable to this context", file=stream)
            print(file=stream)
            continue
        left, right = result
        values = [(pair.partners[0], float(left)),
                  (pair.partners[1], float(right))]
        rel = _relative(values)
        for label, value in values:
            print(f"       {label:44s} {value: .12e}", file=stream)
        verdict = "PASS" if rel <= tol else "FAIL"
        ok = ok and rel <= tol
        print(f"       relative difference {rel: .3e}   {verdict}", file=stream)
        print(file=stream)
    for closure in CLOSURE_REGISTRY:
        result = closure.evaluate(ctx)
        print(f"[closure] {closure.name}   ({closure.quantity})", file=stream)
        if result is None:
            print("       SKIP -- not applicable to this context", file=stream)
            print(file=stream)
            continue
        rel = _relative(result)
        for label, value in result:
            print(f"       {label:44s} {value: .12e}", file=stream)
        verdict = "PASS" if rel <= tol else "FAIL"
        ok = ok and rel <= tol
        print(f"       relative spread     {rel: .3e}   {verdict}", file=stream)
        print(file=stream)
    return ok


def print_ledger(ctx, stream=sys.stdout):
    """Print the carrier's named ledger rows and its TPMC-comparable reads."""
    ledger = ctx.ledger
    if not ledger:
        print("carrier ledger EMPTY -- the flag is not armed", file=stream)
        return
    print("=" * 78, file=stream)
    print("NAMED LEDGER ROWS (directed hot surface carrier)", file=stream)
    print("=" * 78, file=stream)
    rows = (
        ("launch", "launch_per_s", "launch_W"),
        ("jet-ionization", "jet_ionization_per_s", "jet_ionization_W"),
        ("partner-exchange", "partner_exchange_per_s", "partner_exchange_W"),
        ("wall-leak", "wall_leak_per_s", "wall_leak_W"),
        ("end-leak", "end_leak_per_s", "end_leak_W"),
        ("mesh-cull", "mesh_cull_per_s", "mesh_cull_W"),
        ("v1-withdrawal", "v1_withdrawal_per_s", "v1_withdrawal_W"),
    )
    print(f"  {'row':20s} {'rate [/s]':>16s} {'power [kW]':>14s}", file=stream)
    for label, rate_key, power_key in rows:
        print(
            f"  {label:20s} {ledger[rate_key]:16.6e} "
            f"{ledger[power_key] * 1e-3:14.6f}",
            file=stream,
        )
    delivered = (
        _row_total(ctx.carrier_term.Ei, ctx.Vp) / ERG_S_PER_W
        + _row_total(ctx.carrier_term.En, ctx.V_nn) / ERG_S_PER_W
    )
    print(file=stream)
    print(f"  delivered to the fluids      {delivered * 1e-3:14.6f} kW",
          file=stream)
    print(
        "  delivered fraction           "
        f"{delivered / max(ledger['launch_W'], 1e-300):14.6f}",
        file=stream,
    )
    print(
        "  electron binding cost        "
        f"{ledger['jet_ionization_cost_W'] * 1e-3:14.6f} kW  "
        "(booked on Ee, outside the beam energy identity)",
        file=stream,
    )
    print(file=stream)
    print("  CONVENTION DEBTS (invisible to the identities below)", file=stream)
    print(
        "  u.dM partner-exchange        "
        f"{ledger['u_dM_partner_exchange_W'] * 1e-3:14.6f} kW",
        file=stream,
    )
    print(
        "  u.dM jet-ionization          "
        f"{ledger['u_dM_jet_ionization_W'] * 1e-3:14.6f} kW",
        file=stream,
    )
    print(
        "  u.dM ion total               "
        f"{ledger['u_dM_ion_total_W'] * 1e-3:14.6f} kW",
        file=stream,
    )
    print(
        "  u.dM partner (neutral row)   "
        f"{ledger['u_dM_partner_neutral_W'] * 1e-3:14.6f} kW",
        file=stream,
    )
    print(
        "    the bulk-kinetic cross term this booking creates and debits "
        "nowhere; both halves of",
        file=stream,
    )
    print(
        "    every pair below are booked in it, so no identity can see it.",
        file=stream,
    )
    print(
        "  Q_mix missing (ion side)     "
        f"{ledger['q_mix_missing_W'] * 1e-3:14.6f} kW",
        file=stream,
    )
    print(
        "    (1/2) m (u_i - v_fast)^2 over the beam's own births: the term "
        "the stance's",
        file=stream,
    )
    print(
        "    ionization_birth_energy_model='conservative' books for a birth "
        "and the carrier",
        file=stream,
    )
    print(
        "    does not. THE LIVE DEBT; which of the two rows the carrier ought "
        "to book is a",
        file=stream,
    )
    print(
        "    birth-convention ruling and is deliberately not taken in the "
        "code.",
        file=stream,
    )
    print(
        "  electron-birth gap           "
        f"{ledger['electron_birth_convention_W'] * 1e-3:14.6f} kW",
        file=stream,
    )
    print(
        "    (3/2) k Te per beam ionization. NOT a mismatch at the stance: "
        "'conservative'",
        file=stream,
    )
    print(
        "    books Ee_birth = 0 for the bulk too, so the carrier AGREES. "
        "Reported as the size",
        file=stream,
    )
    print(
        "    a disagreement would have (a deprecated 'legacy' bulk arm would "
        "carry it).",
        file=stream,
    )
    print(
        "  bulk birth model             "
        f"{ctx.sim._input_dict.get('ionization_birth_energy_model', 'legacy')!r:>14}",
        file=stream,
    )
    print(file=stream)
    print("  KINEMATICS AND TPMC-COMPARABLE READS", file=stream)
    print(f"  E_fast [eV]                  {ledger['E_fast_eV']:14.6f}",
          file=stream)
    print(f"  v_fast [cm/s]                {ledger['v_fast_cm_s']:14.6e}",
          file=stream)
    print(f"  f_dep (CX + ionization)      {ledger['f_dep']:14.6f}",
          file=stream)
    print(f"  CX share of interactions     {ledger['cx_share']:14.6f}",
          file=stream)
    print(file=stream)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t-end", type=float, default=5.0e-3,
                        help="simulated time to run before auditing [s]")
    parser.add_argument("--nx", type=int, default=60,
                        help="far-column cell count (the golden mesh is 60)")
    parser.add_argument("--tol", type=float, default=1.0e-10,
                        help="relative tolerance for every identity")
    parser.add_argument("--progress", type=float, default=30.0,
                        help="progress interval [s]; 0 disables")
    parser.add_argument("--out", default=None,
                        help="also write the report to this file")
    args = parser.parse_args()

    params, flags = build_baseline_config(
        param_overrides={"nx": args.nx, "cathode_jet_hot_carrier": True}
    )
    sim = LAPDSim1D(
        params,
        flags,
        progress_interval_s=(args.progress if args.progress > 0 else None),
    )
    # start_simulation, NOT run: the baseline configuration asks for the
    # equilibrated neutral seed, and only this entry point builds it. Calling
    # run() directly would start from the bare nn0 fill and audit a different
    # machine from the one the golden's operating point describes.
    sim.start_simulation(t_end=args.t_end, max_steps=150000)

    ctx = build_context(sim)
    streams = [sys.stdout]
    handle = open(args.out, "w") if args.out else None
    if handle is not None:
        streams.append(handle)
    try:
        for stream in streams:
            print(
                f"solver time {sim.time:.6e} s   nx={args.nx}   "
                f"carrier armed={sim._cathode_jet_carrier}",
                file=stream,
            )
            print_ledger(ctx, stream=stream)
        ok = all([run_audit(ctx, args.tol, stream=s) for s in streams])
        for stream in streams:
            print("AUDIT " + ("PASS" if ok else "FAIL"), file=stream)
    finally:
        if handle is not None:
            handle.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
