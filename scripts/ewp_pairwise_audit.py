"""Pairwise-partner momentum audit, as a registry of named momentum pairs.

Sum-closure alone cannot certify a term that MOVES momentum. A term that
fabricates momentum and a term that relocates it can both leave a global sum
looking healthy -- if the fabricated part is small, or if some other sink
happens to absorb it. So this audit is PAIRWISE: for each registered
mechanism, every increment it puts on a receiver row must be matched by an
equal-and-opposite decrement of its OWN named partner term, and no other row
of the ledger may move at all.

REGISTRY. Each mechanism registers one ``MomentumPair`` under a name via the
``@_pair`` decorator. A pair supplies a builder returning ``PairEvidence`` --
the flag-off and flag-on operator outputs, the partitioned pool with its
absorbed/retained split, INDEPENDENTLY recomputed reference rows, the volumes
the conservation sum is taken on, and a zero-strength limit case. The five
checks below are generic and run unchanged on every registered pair, so adding
a mechanism means writing one builder, not one audit.

CHECKS (all must pass; exit 1 on any failure):

  P1  ROW ISOLATION -- with the mechanism on, every ledger row except the
      declared receiver is bit-identical to the mechanism-off operator.
  P2  PAIRWISE PARTNER -- two bit-exact statements together excluding
      fabrication: (a) absorbed + retained is exactly the pool they are drawn
      from, so the split creates nothing; (b) the operator's receiver row is
      exactly the reference row rebuilt with the partner term replaced by its
      absorbed part, on BOTH settings. A differenced RHS cannot make this
      statement -- its cancellation is only roundoff-accurate -- which is why
      the reference rows are recomputed independently here.
  P3  POOL BOUND -- the re-routed amount is non-negative and never exceeds the
      pool it is drawn from.
  P4  CONSERVATION (mechanism on) -- the volume-weighted rows plus the booked
      loss sum to zero to roundoff, and the booked loss is non-trivial and
      strictly smaller than the un-partitioned one.
  P5  ZERO-STRENGTH LIMIT -- where the mechanism's strength goes to zero its
      weight is exactly 1 and the whole operator reproduces the mechanism-off
      result bit-for-bit, so the change is a strict refinement of the certified
      ledger rather than a replacement for it.

Registered pairs:

  ``wall-branch-momentum-partition``
      ``neutral_wall_momentum_partition``: the two-zone wall branch
      ``-nu_wall*M_n_a`` assumes free-molecular flight to the vessel wall. At
      finite gas density a He--He elastic collision can intercept it, and that
      momentum stays in the annulus gas. Receiver ``M_n_a``; partner the
      wall-absorption term.

This audit does NOT re-run the old X4 whole-system closure gate
(``verify_sim1d_nbl2_neutral_transport.py::gate_x4``), which is untouched.

Usage (from <checkout>/cablp, PYTHONPATH set to that same cablp):

    python scripts/ewp_pairwise_audit.py [--pair NAME] [--list] [--demo]
                                         [--sigma-hehe-cm2 S]

``--sigma-hehe-cm2`` supplies the He--He elastic cross section [cm^2] for the
wall-branch pair. There is deliberately NO default in the solver: the value is
owed a literature box. The audit's own default is the repo's one existing
boxed number, the hard-sphere Lennard-Jones value in
``scripts/sp3_build_nn0.py`` (2.044e-15 cm^2, collision diameter 2.551
Angstrom, Hirschfelder/Curtiss/Bird), used here for DEMONSTRATION ONLY.
"""

import argparse
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Callable

import numpy as np

from cablp.solvers._sim1d.core.state import ConservativeState1D, derive_state
from cablp.solvers._sim1d.physics.sources import (
    neutral_momentum_two_zone_rhs,
    neutral_wall_partition_survival,
)
from cablp.vars._cons import ev_to_erg, kb_cgs

ROW_NAMES = ("n", "nn", "M", "Ee", "Ei", "M_n", "nn_a", "M_n_a")


# ------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------
@dataclass
class PairEvidence:
    """Everything the generic checks need about one mechanism."""

    receiver: str                 # state row that GAINS
    partner: str                  # human name of the term that LOSES
    rhs_off: ConservativeState1D  # operator with the mechanism off
    rhs_on: ConservativeState1D   # operator with the mechanism on
    pool: np.ndarray              # the partitioned quantity, receiver units
    absorbed: np.ndarray          # part still booked to the partner
    retained: np.ndarray          # part re-routed to the receiver
    ref_row_off: np.ndarray       # independently rebuilt receiver row, off
    ref_row_on: np.ndarray        # independently rebuilt receiver row, on
    volumes: dict                 # row name -> volume for the closure sum
    sink_volume: np.ndarray       # volume the booked loss is charged on
    limit_off: ConservativeState1D   # zero-strength limit, mechanism off
    limit_on: ConservativeState1D    # zero-strength limit, mechanism on
    limit_weight: np.ndarray         # the weight at zero strength (must be 1)
    notes: dict = field(default_factory=dict)


@dataclass
class MomentumPair:
    name: str
    summary: str
    build: Callable            # (args) -> PairEvidence
    demo: Callable | None = None


PAIRS: dict[str, MomentumPair] = {}


def _pair(name, summary, demo=None):
    def wrap(fn):
        if name in PAIRS:
            raise ValueError(f"duplicate momentum pair: {name}")
        PAIRS[name] = MomentumPair(name=name, summary=summary, build=fn, demo=demo)
        return fn
    return wrap


# ------------------------------------------------------------------
# Generic checks
# ------------------------------------------------------------------
def _rows(rhs):
    return {name: getattr(rhs, name) for name in ROW_NAMES}


def _same(a, b):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return np.array_equal(np.asarray(a), np.asarray(b))


def check_pair(ev):
    """Run P1-P5 on one PairEvidence. Return (ok, lines)."""
    lines = []
    ok = True

    # --- P1 ROW ISOLATION ---------------------------------------------------
    moved = [
        name for name, a in _rows(ev.rhs_off).items()
        if name != ev.receiver and not _same(a, _rows(ev.rhs_on)[name])
    ]
    p1 = not moved
    ok &= p1
    lines.append(
        f"P1 ROW ISOLATION            {'PASS' if p1 else 'FAIL'}  "
        f"rows moved besides {ev.receiver!r}: {moved or 'none'}"
    )

    # --- P2 PAIRWISE PARTNER ------------------------------------------------
    p2a = np.array_equal(ev.absorbed + ev.retained, ev.pool)
    p2b = _same(getattr(ev.rhs_on, ev.receiver), ev.ref_row_on) and _same(
        getattr(ev.rhs_off, ev.receiver), ev.ref_row_off
    )
    p2 = bool(p2a and p2b)
    ok &= p2
    increment = np.asarray(getattr(ev.rhs_on, ev.receiver)) - np.asarray(
        getattr(ev.rhs_off, ev.receiver)
    )
    resid = float(np.max(np.abs(increment - ev.retained)))
    scale = float(np.max(np.abs(ev.pool)))
    lines.append(
        f"P2 PAIRWISE PARTNER         {'PASS' if p2 else 'FAIL'}  "
        f"partition identity absorbed+retained==pool: "
        f"{'exact' if p2a else 'BROKEN'}; partner substitution "
        f"({ev.receiver} vs {ev.partner}) on/off rows: "
        f"{'exact' if p2b else 'BROKEN'}"
    )
    lines.append(
        f"     (diagnostic) differenced increment vs partner decrement: "
        f"max|d| = {resid:.3e} on a pool of {scale:.3e}, "
        f"rel = {resid / max(scale, 1e-300):.3e} -- roundoff of the summed "
        f"RHS, not a ledger discrepancy"
    )

    # --- P3 POOL BOUND ------------------------------------------------------
    frac = np.divide(
        ev.retained, ev.pool,
        out=np.zeros_like(ev.retained), where=ev.pool != 0.0,
    )
    p3 = bool(np.all(ev.retained >= 0.0) and np.all(frac <= 1.0))
    ok &= p3
    lines.append(
        f"P3 POOL BOUND               {'PASS' if p3 else 'FAIL'}  "
        f"re-routed fraction of the {ev.partner} pool in "
        f"[{frac.min():.6f}, {frac.max():.6f}]"
    )

    # --- P4 CONSERVATION (mechanism on) -------------------------------------
    net = sum(
        float(np.sum(np.asarray(getattr(ev.rhs_on, row)) * vol))
        for row, vol in ev.volumes.items()
    )
    booked = float(np.sum(ev.absorbed * ev.sink_volume))
    unpartitioned = float(np.sum(ev.pool * ev.sink_volume))
    rel = abs(net + booked) / max(abs(booked), 1e-300)
    p4 = bool(rel < 1e-12 and booked > 0.0 and booked < unpartitioned)
    ok &= p4
    lines.append(
        f"P4 CONSERVATION (on)        {'PASS' if p4 else 'FAIL'}  "
        f"net={net:.9e} booked={booked:.9e} rel_residual={rel:.3e} "
        f"(tol 1e-12); booked < unpartitioned {unpartitioned:.9e}: "
        f"{booked < unpartitioned}"
    )

    # --- P5 ZERO-STRENGTH LIMIT ---------------------------------------------
    p5 = bool(
        np.all(ev.limit_weight == 1.0)
        and all(
            _same(v, _rows(ev.limit_on)[k])
            for k, v in _rows(ev.limit_off).items()
        )
    )
    ok &= p5
    lines.append(
        f"P5 ZERO-STRENGTH LIMIT      {'PASS' if p5 else 'FAIL'}  "
        f"weight is exactly 1 and the operator is bit-identical to "
        f"mechanism-off on every row"
    )
    return ok, lines


# ------------------------------------------------------------------
# Pair: wall-branch momentum partition
# ------------------------------------------------------------------
# Demonstration-only cross section: the repo's single boxed He-He elastic
# value, defined in scripts/sp3_build_nn0.py (hard-sphere pi*sigma_LJ^2,
# sigma_LJ = 2.551 A; Hirschfelder/Curtiss/Bird). NOT a solver default.
DEMO_SIGMA_HEHE_CM2 = 2.044e-15
DEMO_SIGMA_SOURCE = (
    "scripts/sp3_build_nn0.py SIGMA_HE_HE_CM2 (hard-sphere LJ, 2.551 A; "
    "Hirschfelder/Curtiss/Bird) -- DEMONSTRATION ONLY, not a boxed solver value"
)

# Production-class annulus geometry: LAPD column radius and vessel bore [cm].
RP_CM = 15.0
RM_CM = 50.0
ION_MASS_G = 6.6464731e-24  # He


def _wall_fixture(nn_a_cm3):
    """Return (state, floors, geometry) for a drifting two-zone gas."""
    nn_a = np.asarray(nn_a_cm3, dtype=float)
    cells = nn_a.size
    Rp = np.full(cells, RP_CM)
    Rm = np.full(cells, RM_CM)
    geometry = SimpleNamespace(
        Rp_cm=Rp,
        Rm_cm=Rm,
        plasma_volume_cm3=np.pi * Rp**2 * 10.0,
        neutral_volume_cm3=np.pi * Rm**2 * 10.0,
    )
    ones = np.ones(cells)
    state = ConservativeState1D(
        n=1.0e12 * ones,
        nn=5.0e11 * ones,
        M=ION_MASS_G * 1.0e12 * 3.0e5 * ones,
        Ee=1.5 * 1.0e12 * 3.0 * 1.602176634e-12 * ones,
        Ei=1.5 * 1.0e12 * 2.0 * 1.602176634e-12 * ones,
        M_n=ION_MASS_G * 5.0e11 * 1.0e5 * ones,
        nn_a=nn_a,
        M_n_a=ION_MASS_G * nn_a * 4.0e4,
    )
    floors = {"n": 1.0e6, "nn": 1.0e6, "Te": 0.1, "Ti": 0.1}
    return state, floors, geometry


def _wall_reference(state, floors, geometry, Tn_K=300.0):
    """Independently rebuild the two-zone momentum operator's pieces.

    The audit must not certify the operator against itself, so this recomputes
    the radial transfer, the wall pool and the zone volumes from the closure's
    stated definitions rather than importing them.
    """
    derived = derive_state(state, floors=floors, ion_mass_g=ION_MASS_G)
    Rp = np.asarray(geometry.Rp_cm, dtype=float)
    Rm = np.asarray(geometry.Rm_cm, dtype=float)
    Vc = np.asarray(geometry.plasma_volume_cm3, dtype=float)
    Va = np.maximum(np.asarray(geometry.neutral_volume_cm3, dtype=float) - Vc, 0.0)
    live = Va > 0.0
    vbar_i = np.sqrt(
        8.0 * np.asarray(derived.Ti, dtype=float) * ev_to_erg
        / (np.pi * ION_MASS_G)
    )
    vbar_n = np.sqrt(8.0 * float(Tn_K) * kb_cgs / (np.pi * ION_MASS_G))
    ann_area = np.maximum(Rm**2 - Rp**2, 1e-300)
    nu_ca = np.where(live, vbar_i / (2.0 * Rp), 0.0)
    nu_ac = np.where(live, vbar_n * Rp / (2.0 * ann_area), 0.0)
    nu_wall = np.where(live, vbar_n * Rm / (2.0 * ann_area), 0.0)
    Mc = np.asarray(state.M_n, dtype=float)
    Ma = np.asarray(state.M_n_a, dtype=float)
    transfer = -Vc * nu_ca * Mc + Va * nu_ac * Ma
    return SimpleNamespace(
        pool=nu_wall * Ma, transfer=transfer, Vc=Vc, Va=Va, live=live
    )


def _wall_row(ref, absorbed):
    """Return the reference ``M_n_a`` row for a given wall-absorption term."""
    row = -ref.transfer / np.maximum(ref.Va, 1e-300) - absorbed
    return np.where(ref.live, row, 0.0)


def _wall_demo(args):
    sigma = args.sigma_hehe_cm2
    dens = np.array([1.0e11, 3.0e11, 5.0e11, 1.0e12, 3.0e12, 1.0e13])
    state, floors, geometry = _wall_fixture(dens)
    survival, tau, mfp = neutral_wall_partition_survival(geometry, dens, sigma)
    pool = _wall_reference(state, floors, geometry).pool
    frac = (pool - survival * pool) / pool
    print()
    print(f"DEMONSTRATION  sigma_HeHe = {sigma:.4e} cm^2")
    print(f"  source: {DEMO_SIGMA_SOURCE}")
    print(f"  annulus radial thickness d = Rm - Rp = {RM_CM - RP_CM:g} cm")
    print()
    print(
        f"  {'nn_a [cm^-3]':>14}  {'mfp [cm]':>12}  {'tau=d/mfp':>10}  "
        f"{'survival 2E3':>13}  {'re-routed':>10}"
    )
    for i in range(dens.size):
        print(
            f"  {dens[i]:14.3e}  {mfp[i]:12.4g}  {tau[i]:10.5f}  "
            f"{survival[i]:13.6f}  {frac[i] * 100:9.3f}%"
        )
    print()
    print(
        "  're-routed' = fraction of the wall-branch momentum pool "
        "nu_wall*M_n_a kept by the annulus gas instead of the wall."
    )


@_pair(
    "wall-branch-momentum-partition",
    "neutral_wall_momentum_partition: He-He collisions intercept the "
    "free-molecular flight to the vessel wall, so part of the wall branch "
    "stays in the annulus gas (receiver M_n_a, partner wall absorption).",
    demo=_wall_demo,
)
def _build_wall_pair(args):
    sigma = args.sigma_hehe_cm2
    # A spread of annulus densities, from the nearly free-molecular corner to
    # the strongly attenuating one.
    nn_a = np.array([1.0e11, 5.0e11, 1.0e12, 5.0e12, 2.0e13])
    state, floors, geometry = _wall_fixture(nn_a)
    kw = dict(
        state=state, floors=floors, ion_mass_g=ION_MASS_G, geometry=geometry
    )
    off = neutral_momentum_two_zone_rhs(**kw, sigma_hehe_cm2=None)
    on = neutral_momentum_two_zone_rhs(**kw, sigma_hehe_cm2=sigma)

    ref = _wall_reference(state, floors, geometry)
    survival, tau, mfp = neutral_wall_partition_survival(
        geometry, state.nn_a, sigma
    )
    absorbed = survival * ref.pool
    retained = ref.pool - absorbed

    # Zero-strength limit: an empty annulus has zero optical depth.
    lim_state, lim_floors, lim_geom = _wall_fixture(np.zeros_like(nn_a))
    lim_kw = dict(
        state=lim_state, floors=lim_floors, ion_mass_g=ION_MASS_G,
        geometry=lim_geom,
    )
    lim_weight, _, _ = neutral_wall_partition_survival(
        lim_geom, lim_state.nn_a, sigma
    )

    return PairEvidence(
        receiver="M_n_a",
        partner="wall absorption",
        rhs_off=off,
        rhs_on=on,
        pool=ref.pool,
        absorbed=absorbed,
        retained=retained,
        ref_row_off=_wall_row(ref, ref.pool),
        ref_row_on=_wall_row(ref, absorbed),
        volumes={"M_n": ref.Vc, "M_n_a": ref.Va},
        sink_volume=ref.Va,
        limit_off=neutral_momentum_two_zone_rhs(**lim_kw, sigma_hehe_cm2=None),
        limit_on=neutral_momentum_two_zone_rhs(**lim_kw, sigma_hehe_cm2=sigma),
        limit_weight=lim_weight,
        notes={"tau": tau, "survival": survival, "mfp": mfp},
    )


# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true", help="list registered pairs")
    ap.add_argument(
        "--pair", action="append", default=None,
        help="audit only this pair (repeatable); default is all",
    )
    ap.add_argument(
        "--demo", action="store_true",
        help="also print each pair's magnitude table",
    )
    ap.add_argument(
        "--sigma-hehe-cm2", type=float, default=DEMO_SIGMA_HEHE_CM2,
        help="He-He elastic cross section [cm^2] for the wall-branch pair "
             "(default is the repo's sp3_build_nn0 hard-sphere value, "
             "demonstration only)",
    )
    args = ap.parse_args()

    if args.list:
        for name, pair in PAIRS.items():
            print(f"{name}\n    {pair.summary}")
        return

    names = args.pair or list(PAIRS)
    unknown = [n for n in names if n not in PAIRS]
    if unknown:
        raise SystemExit(f"unknown pair(s): {unknown}; known: {list(PAIRS)}")

    print("ewp_pairwise_audit: pairwise-partner momentum audit")
    all_ok = True
    for name in names:
        pair = PAIRS[name]
        print()
        print(f"=== {name} ===")
        ev = pair.build(args)
        ok, lines = check_pair(ev)
        all_ok &= ok
        for line in lines:
            print(line)
        if "tau" in ev.notes:
            print(
                f"     optical depth tau spanned: "
                f"[{ev.notes['tau'].min():.5f}, {ev.notes['tau'].max():.5f}]; "
                f"weight spanned: [{ev.notes['survival'].min():.6f}, "
                f"{ev.notes['survival'].max():.6f}]"
            )
        if args.demo and pair.demo is not None:
            pair.demo(args)

    print()
    print("RESULT:", "PASS" if all_ok else "FAIL")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
