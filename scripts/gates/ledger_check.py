"""Conservation-receipt checker for ``sim1d-hdf5-v1`` result files.

A result file may carry a conservation RECEIPT: for each saved interval, every
amount a term moved between two accounts. This checker reads the receipt and
the saved state, recomputes each account's inventory from the state, and
decides whether every account closes. It is written from the model's stated
conservation laws and the result file's layout only; it imports nothing from
the solver, so a receipt the solver writes is checked by arithmetic the solver
did not supply.

Usage::

    python scripts/gates/ledger_check.py RUN.h5 [--stage particles|energy|momentum ...] [--margin M]
    python scripts/gates/ledger_check.py --self-test

Exit status: 0 every requested check passes; 1 a closure, leg or stage-total
failure or a census gap; 2 the check could not run (no ``receipt/`` group, an
unknown schema, a requested stage the receipt does not hold, intervals that do
not tile the saves). A missing receipt is exit 2, never a pass.

RECEIPT SCHEMA (``receipt-v1``)::

    receipt/
      attrs: schema = "receipt-v1", cadence ("save" | "step"), stages_present (e.g. ["particles"])
      interval_t0, interval_t1        (n_intervals,) float64, seconds
      interval_steps                  (n_intervals,) int64, accepted steps accumulated in the interval
      entries/<term>/<quantity>       (n_intervals,) float64, volume-integrated total over the interval
          attrs: debit, credit (account names), site, leg_of (optional), units
      state/<name>                    (n_saves,) extra state the saved fields do not hold
      census/<term>                   attrs: status = "entered" | "zero" | "not_tracked", reason

* ``quantity`` is one of ``particles``, ``momentum``, ``energy_e``,
  ``energy_i``, ``energy_k``, ``energy_n``, in particles, g cm/s and erg.
  Stages: ``particles`` (quantity particles), ``energy`` (the four energy
  quantities), ``momentum``.
* An entry moves its amount from the ``debit`` account to the ``credit``
  account: a positive amount raises the credit account's inventory and lowers
  the debit account's.
* Cadence ``save``: one interval per pair of consecutive saves, interval k
  running from save k to save k+1 (``interval_t0``/``interval_t1`` equal to the
  saved ``time`` exactly). Cadence ``step``: the intervals tile the saves
  contiguously, and the checker sums those between each pair of saves.
* Where two code sites each compute one side of an exchange, each writes its
  own entry and both carry the same ``leg_of`` name. The checker compares the
  legs, and counts the exchange ONCE in the account closures, at the legs'
  mean (assumption A7 below).
* Accounts with an inventory: ``plasma_particles``, ``neutral_particles``,
  ``electron_energy``, ``ion_energy``, ``plasma_kinetic_energy``,
  ``plasma_momentum``, ``neutral_energy``, ``neutral_momentum``, ``circuit``,
  ``cathode_surface``. External accounts (entries only, no inventory):
  ``cathode``, ``anode``, ``end_wall``, ``radiation``,
  ``ionization_potential``, ``pump``, ``puff``, ``numerical_floors``,
  ``bank``. Any other account name is a failure (and is treated as external
  in the arithmetic, so the failure names the entry, not its neighbours).
* ``state/neutral_particles_kinetic`` with the 0/1 flag
  ``state/neutral_kinetic_engaged`` per save holds the neutral particle
  inventory once the kinetic neutrals engage; ``state/circuit``,
  ``state/cathode_surface``, ``state/neutral_energy`` and
  ``state/neutral_momentum`` hold those inventories.
* A census key may also name an inventoried account; ``not_tracked`` on it
  marks the account NOT TRACKED: it is reported, never counted as closed, and
  in its stage total it stands on the boundary like an external account.

WHAT IT CHECKS, per requested stage (default: every stage in
``stages_present``):

1. Account closure: for each inventoried account and each save interval, the
   inventory change recomputed from the saved state equals the entries
   credited to it minus those debited from it.
2. Leg agreement: entries sharing a ``leg_of`` agree.
3. Stage total: summed over the stage's inventoried accounts, the change
   equals the entries from accounts outside that set minus the entries to
   them.
4. Census (once, over all terms): every census term has a valid status, an
   ``entered`` term has at least one entry, and every entry's term is in the
   census.

THE BAR for 1-3 is a roundoff bound fixed before any run, never a physical
scale or a fitted number and never relative to the net change::

    bound = margin * count * 2**-53 * gross

``gross`` is the sum of the magnitudes of everything entering the comparison
(the entries' magnitudes plus, for each inventory, the sum of its summands'
magnitudes at both saves) and ``count`` the floating-point operations that
produced the two sides (derived beside :func:`_closure_count`).

INVENTORIES. The model documents state the plasma lives in the column of each
cell, the neutrals in the column and the annulus, and the conservative fields
are densities; how the saved arrays combine with the saved volumes is taken
under these named assumptions, which the documents do not state:

* A1 (plasma zone): a plasma inventory is the sum over ``plasma_active``
  cells of the field times ``geometry/plasma_volume_cm3`` (the column volume
  V_col = A dz).
* A2 (neutral zones): ``nn`` is the column density and ``nn_a`` the annulus
  density; the annulus volume is ``neutral_volume_cm3 - plasma_volume_cm3``;
  the neutral inventory is ``nn*V_col + nn_a*V_ann`` summed over ALL cells.
* A3 (field units): the saved fields are CGS per cm^3 (``n``, ``nn``,
  ``nn_a`` in cm^-3; ``momentum`` in g cm^-2 s^-1; ``Ee``, ``Ei`` in
  erg cm^-3), so a field times a volume in cm^3 is in the receipt's units.
* A4 (kinetic energy): the plasma kinetic energy density is
  ``0.5 * momentum * u`` with ``u`` the saved velocity field (cm/s).
* A5 (state accounts): ``neutral_energy``, ``neutral_momentum``, ``circuit``
  and ``cathode_surface`` come from ``receipt/state/<account>``, a single
  summand each, counted as a sum over the file's cells.
* A6 (stage of the state accounts): ``circuit`` and ``cathode_surface`` hold
  energy and belong to the ``energy`` stage.
* A7 (legs in a closure): a ``leg_of`` group enters the account closures once,
  at the mean of its legs, with the first leg's debit and credit.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

import h5py
import numpy as np

SCHEMA = "receipt-v1"

# Unit roundoff of IEEE binary64 under round-to-nearest.
UNIT_ROUNDOFF = 2.0 ** -53

# Safety factor on the operation count. Each counted operation is taken to
# round once, by at most UNIT_ROUNDOFF relative to a magnitude ``gross``
# bounds. Default 4 = 2 * 2: a counted state update is a multiply-add
# (state + dt*rate) that rounds twice, and the magnitude each rounding is
# relative to is an intermediate (a mid-interval state, a partial sum) that
# the endpoint magnitudes in ``gross`` bound only to within the same factor
# when the interval's change is comparable to the inventory.
MARGIN = 4.0

QUANTITY_STAGE = {
    "particles": "particles",
    "momentum": "momentum",
    "energy_e": "energy",
    "energy_i": "energy",
    "energy_k": "energy",
    "energy_n": "energy",
}
QUANTITY_UNITS = {
    "particles": "particles",
    "momentum": "g cm/s",
    "energy_e": "erg",
    "energy_i": "erg",
    "energy_k": "erg",
    "energy_n": "erg",
}
STAGES = ("particles", "energy", "momentum")

INVENTORIED = {
    "plasma_particles": "particles",
    "neutral_particles": "particles",
    "electron_energy": "energy",
    "ion_energy": "energy",
    "plasma_kinetic_energy": "energy",
    "neutral_energy": "energy",
    "circuit": "energy",            # A6
    "cathode_surface": "energy",    # A6
    "plasma_momentum": "momentum",
    "neutral_momentum": "momentum",
}
EXTERNAL = frozenset({
    "cathode", "anode", "end_wall", "radiation", "ionization_potential",
    "pump", "puff", "numerical_floors", "bank",
})
STATE_ACCOUNTS = ("neutral_energy", "neutral_momentum", "circuit", "cathode_surface")
CENSUS_STATUSES = ("entered", "zero", "not_tracked")


class CannotRun(Exception):
    """The check could not run (exit 2)."""


def _s(v):
    return v.decode() if isinstance(v, bytes) else str(v)


def _str_list(v):
    if v is None:
        return []
    if isinstance(v, (str, bytes)):
        return [_s(v)]
    return [_s(x) for x in np.asarray(v).ravel()]


# ---------------------------------------------------------------- inventories

class Inventory:
    """An account's inventory per save: value, sum of summand magnitudes, summand count."""

    def __init__(self, values, gross, summands):
        self.values = np.asarray(values, dtype=np.float64)
        self.gross = np.asarray(gross, dtype=np.float64)
        self.summands = int(summands)


def _summed(parts, summands):
    total = parts.sum(axis=1)
    return Inventory(total, np.abs(parts).sum(axis=1), summands)


def _load_inventories(f, rc):
    """Every inventory the file can supply, keyed by account name."""
    geo = f["geometry"] if "geometry" in f else None
    state = rc["state"] if "state" in rc else None
    inv = {}
    n_cells = None
    if geo is not None and "plasma_volume_cm3" in geo:
        vp = geo["plasma_volume_cm3"][()].astype(np.float64)
        n_cells = vp.size
        active = (geo["plasma_active"][()].astype(bool) if "plasma_active" in geo
                  else np.ones(n_cells, bool))
        n_act = int(active.sum())

        def plasma(field):  # A1, A3
            return _summed(field[:, active] * vp[active], n_act)

        if "n" in f:
            inv["plasma_particles"] = plasma(f["n"][()])
        if "Ee" in f:
            inv["electron_energy"] = plasma(f["Ee"][()])
        if "Ei" in f:
            inv["ion_energy"] = plasma(f["Ei"][()])
        if "momentum" in f:
            inv["plasma_momentum"] = plasma(f["momentum"][()])
            if "u" in f:  # A4
                inv["plasma_kinetic_energy"] = plasma(0.5 * f["momentum"][()] * f["u"][()])
        if "nn" in f and "nn_a" in f and "neutral_volume_cm3" in geo:  # A2
            va = geo["neutral_volume_cm3"][()].astype(np.float64) - vp
            parts = np.concatenate([f["nn"][()] * vp, f["nn_a"][()] * va], axis=1)
            neu = _summed(parts, 2 * n_cells)
            if state is not None and "neutral_particles_kinetic" in state:
                if "neutral_kinetic_engaged" not in state:
                    raise CannotRun("state/neutral_particles_kinetic without state/neutral_kinetic_engaged")
                eng = state["neutral_kinetic_engaged"][()].astype(bool)
                kin = state["neutral_particles_kinetic"][()].astype(np.float64)
                neu.values = np.where(eng, kin, neu.values)
                neu.gross = np.where(eng, np.abs(kin), neu.gross)
            inv["neutral_particles"] = neu
    if state is not None:
        for name in STATE_ACCOUNTS:  # A5
            if name in state:
                v = state[name][()].astype(np.float64)
                inv[name] = Inventory(v, np.abs(v), n_cells or 1)
    return inv, (n_cells or 1)


# ---------------------------------------------------------------- receipt

class Entry:
    def __init__(self, term, quantity, debit, credit, leg_of, values, members=1):
        self.term = term
        self.quantity = quantity
        self.stage = QUANTITY_STAGE.get(quantity)
        self.debit = debit
        self.credit = credit
        self.leg_of = leg_of
        self.values = values
        self.members = members  # real entries this one stands for (a leg group > 1)

    @property
    def label(self):
        return f"{self.term}/{self.quantity}"


def _save_groups(rc, time):
    """Receipt intervals belonging to each save interval, and steps per save interval."""
    for key in ("interval_t0", "interval_t1", "interval_steps"):
        if key not in rc:
            raise CannotRun(f"receipt/{key} missing")
    t0 = rc["interval_t0"][()]
    t1 = rc["interval_t1"][()]
    steps = rc["interval_steps"][()].astype(np.int64)
    cadence = _s(rc.attrs.get("cadence", ""))
    n_saves = time.size
    if cadence == "save":
        if t0.size != n_saves - 1 or np.any(t0 != time[:-1]) or np.any(t1 != time[1:]):
            raise CannotRun("cadence 'save' intervals do not equal the saved time pairs")
        groups = [[k] for k in range(n_saves - 1)]
    elif cadence == "step":
        groups, j = [], 0
        for k in range(n_saves - 1):
            if j >= t0.size or t0[j] != time[k]:
                raise CannotRun(f"cadence 'step' intervals do not start at save {k}")
            g = [j]
            while t1[j] != time[k + 1]:
                if t1[j] > time[k + 1] or j + 1 >= t0.size or t0[j + 1] != t1[j]:
                    raise CannotRun(f"cadence 'step' intervals do not tile save interval {k}")
                j += 1
                g.append(j)
            groups.append(g)
            j += 1
        if j != t0.size:
            raise CannotRun("cadence 'step' intervals extend past the last save")
    else:
        raise CannotRun(f"unknown cadence {cadence!r}")
    save_steps = np.array([int(steps[g].sum()) for g in groups], dtype=np.int64)
    n_sub = np.array([len(g) for g in groups], dtype=np.int64)
    return groups, save_steps, n_sub


def _load_entries(rc, groups, failures):
    entries = []
    if "entries" not in rc:
        return entries
    known = set(INVENTORIED) | EXTERNAL
    for term in sorted(rc["entries"].keys()):
        tg = rc["entries"][term]
        for quantity in sorted(tg.keys()):
            ds = tg[quantity]
            label = f"{term}/{quantity}"
            if quantity not in QUANTITY_STAGE:
                failures.append(("quantity", label, None))
                continue
            if "debit" not in ds.attrs or "credit" not in ds.attrs:
                failures.append(("entry", label, None))
                continue
            debit, credit = _s(ds.attrs["debit"]), _s(ds.attrs["credit"])
            for acct in (debit, credit):
                if acct not in known:
                    failures.append(("account", label, None))
            raw = ds[()].astype(np.float64)
            values = np.array([sum(raw[g].tolist()) if len(g) > 1 else raw[g[0]] for g in groups])
            leg = _s(ds.attrs["leg_of"]) if "leg_of" in ds.attrs else None
            entries.append(Entry(term, quantity, debit, credit, leg, values))
    return entries


def _load_census(rc):
    census = {}
    if "census" in rc:
        for term in rc["census"].keys():
            status = rc["census"][term].attrs.get("status")
            census[term] = _s(status) if status is not None else None
    return census


# ---------------------------------------------------------------- bounds

def _bound(margin, count, gross):
    return margin * count * UNIT_ROUNDOFF * gross


def _closure_count(steps, n_entries, summands, n_sub):
    # Operations producing the two sides of one account's closure in one save
    # interval:
    #   steps*(n_entries+1): each accepted step adds at most n_entries term
    #       increments to each cell's state and stores it once; each rounding is
    #       bounded relative to that cell's state, and the cells' state
    #       magnitudes sum to the inventory gross;
    #   4*summands: each saved summand is a density times a volume (one
    #       rounding at each of the two saves) and the two inventory sums
    #       (summands-1 additions each);
    #   n_entries*(steps + summands + n_sub): each entry is accumulated over
    #       the interval's steps, each step's amount a sum over at most
    #       `summands` cells, and the checker sums n_sub receipt intervals;
    #   n_entries: the checker sums the entries;
    #   2: the inventory difference and the residual.
    return steps * (n_entries + 1) + 4 * summands + n_entries * (steps + summands + n_sub) + n_entries + 2


def _leg_count(steps, n_legs, n_cells, n_sub):
    # Each leg is accumulated over the interval's steps from per-step amounts
    # summed over at most n_cells cells, then summed over n_sub receipt
    # intervals; one subtraction compares two legs.
    return n_legs * (steps + n_cells + n_sub) + 1


def _ratio(resid, bound):
    if bound > 0:
        return abs(resid) / bound
    return 0.0 if resid == 0 else float("inf")


def _verdict(resid, bounds):
    ratios = np.array([_ratio(r, b) for r, b in zip(resid, bounds)])
    worst = int(np.argmax(ratios)) if ratios.size else 0
    n_fail = int(np.sum(ratios > 1.0))
    return worst, n_fail, ratios


def _fmt(tag, ok, what, worst, resid, bounds, ratios, n_fail):
    if not len(resid):
        return f"  {tag:4s} {what}: no intervals"
    return (f"  {'PASS' if ok else 'FAIL'} {what}: worst interval {worst} residual "
            f"{resid[worst]:.3e} bound {bounds[worst]:.3e} ratio {ratios[worst]:.3e} "
            f"({n_fail}/{len(resid)} intervals over)")


# ---------------------------------------------------------------- check

class Report:
    def __init__(self):
        self.lines = []
        self.failures = []      # (kind, name, interval or None)
        self.closed = []        # "stage/account" that closed
        self.not_tracked = []   # account names
        self.exit_code = 0


def check(path, stages=None, margin=MARGIN):
    rep = Report()
    with h5py.File(path, "r") as f:
        if "receipt" not in f:
            raise CannotRun("no receipt/ group")
        rc = f["receipt"]
        schema = _s(rc.attrs.get("schema", ""))
        if schema != SCHEMA:
            raise CannotRun(f"unknown receipt schema {schema!r}")
        present = _str_list(rc.attrs.get("stages_present"))
        for s in present:
            if s not in STAGES:
                raise CannotRun(f"unknown stage {s!r} in stages_present")
        stages = list(stages) if stages else present
        for s in stages:
            if s not in present:
                raise CannotRun(f"requested stage {s!r} not in stages_present {present}")
        if "time" not in f:
            raise CannotRun("no time dataset")
        time = f["time"][()]
        groups, steps, n_sub = _save_groups(rc, time)
        failures = rep.failures
        entries = _load_entries(rc, groups, failures)
        census = _load_census(rc)
        inv, n_cells = _load_inventories(f, rc)

    rep.lines.append(f"ledger_check: {path}")
    rep.lines.append(f"  schema {SCHEMA}, {len(groups)} save intervals, stages {stages}, margin {margin:g}")
    for label in sorted({x[1] for x in failures if x[0] in ("quantity", "entry", "account")}):
        kinds = sorted({x[0] for x in failures if x[1] == label})
        rep.lines.append(f"  FAIL entry {label}: invalid {', '.join(kinds)}")

    not_tracked = sorted(a for a, st in census.items() if a in INVENTORIED and st == "not_tracked")
    rep.not_tracked = not_tracked

    for stage in stages:
        rep.lines.append(f"stage {stage}")
        stage_entries = [e for e in entries if e.stage == stage]
        for e in stage_entries:
            for acct in (e.debit, e.credit):
                if acct in INVENTORIED and INVENTORIED[acct] != stage:
                    failures.append(("stage-mismatch", e.label, None))
                    rep.lines.append(f"  FAIL entry {e.label}: account {acct} holds stage "
                                     f"{INVENTORIED[acct]}, entry is stage {stage}")

        # Legs: compare, then stand each group in the closures once (A7).
        effective, legs = [], {}
        for e in stage_entries:
            if e.leg_of is None:
                effective.append(e)
            else:
                legs.setdefault((e.leg_of, e.quantity), []).append(e)
        for (leg, quantity), members in sorted(legs.items()):
            name = f"{leg}/{quantity}"
            if len(members) < 2:
                failures.append(("leg", name, None))
                rep.lines.append(f"  FAIL leg {name}: one leg only ({members[0].term})")
            else:
                if len({(m.debit, m.credit) for m in members}) > 1:
                    failures.append(("leg", name, None))
                    rep.lines.append(f"  FAIL leg {name}: legs name different accounts")
                vals = np.array([m.values for m in members])
                resid = np.max(np.abs(vals - vals[0]), axis=0)
                bounds = np.array([_bound(margin, _leg_count(steps[k], len(members), n_cells, n_sub[k]),
                                          np.abs(vals[:, k]).sum()) for k in range(len(groups))])
                worst, n_fail, ratios = _verdict(resid, bounds)
                if n_fail:
                    failures.append(("leg", name, worst))
                rep.lines.append(_fmt("leg", n_fail == 0, f"leg {name} ({', '.join(m.term for m in members)})",
                                      worst, resid, bounds, ratios, n_fail))
            mean = np.sum([m.values for m in members], axis=0) / len(members)
            effective.append(Entry(f"[{leg}]", quantity, members[0].debit, members[0].credit,
                                   leg, mean, members=len(members)))

        # Account closure.
        tracked = []
        for acct, acct_stage in INVENTORIED.items():
            if acct_stage != stage:
                continue
            touching = [e for e in effective if acct in (e.debit, e.credit)]
            if acct in not_tracked:
                rep.lines.append(f"  NOT TRACKED {acct} ({len(touching)} entries touch it; "
                                 f"census status not_tracked)")
                continue
            if acct not in inv:
                if touching:
                    failures.append(("no-inventory", f"{stage}/{acct}", None))
                    rep.lines.append(f"  FAIL closure {acct}: entries touch it but the file holds no inventory")
                continue
            tracked.append(acct)
            I = inv[acct]
            n_ent = sum(e.members for e in touching)
            resid, bounds = [], []
            for k in range(len(groups)):
                flow = 0.0
                mag = 0.0
                for e in touching:
                    sign = (e.credit == acct) - (e.debit == acct)
                    flow += sign * e.values[k]
                    mag += abs(e.values[k])
                resid.append((I.values[k + 1] - I.values[k]) - flow)
                gross = I.gross[k] + I.gross[k + 1] + mag
                bounds.append(_bound(margin, _closure_count(steps[k], n_ent, I.summands, n_sub[k]), gross))
            resid, bounds = np.array(resid), np.array(bounds)
            worst, n_fail, ratios = _verdict(resid, bounds)
            if n_fail:
                failures.append(("closure", f"{stage}/{acct}", worst))
            else:
                rep.closed.append(f"{stage}/{acct}")
            rep.lines.append(_fmt("", n_fail == 0, f"closure {acct}", worst, resid, bounds, ratios, n_fail))

        # Stage total. The stage identity is the sum of the account identities
        # over the tracked set; an entry with BOTH accounts in the set enters
        # that sum once with + and once with -, so it is left out here rather
        # than added and subtracted: internal transfers cancel exactly, with no
        # rounding, and only boundary-crossing entries are summed.
        if tracked:
            tset = set(tracked)
            touching = [e for e in effective if e.debit in tset or e.credit in tset]
            n_ent = sum(e.members for e in touching)
            summands = sum(inv[a].summands for a in tracked)
            resid, bounds = [], []
            for k in range(len(groups)):
                change = 0.0
                gross = 0.0
                for a in tracked:
                    change += inv[a].values[k + 1] - inv[a].values[k]
                    gross += inv[a].gross[k] + inv[a].gross[k + 1]
                ext = 0.0
                for e in touching:
                    sign = (e.credit in tset) - (e.debit in tset)
                    if sign:
                        ext += sign * e.values[k]
                    gross += abs(e.values[k])
                resid.append(change - ext)
                count = _closure_count(steps[k], n_ent, summands, n_sub[k]) + 2 * len(tracked)
                bounds.append(_bound(margin, count, gross))
            resid, bounds = np.array(resid), np.array(bounds)
            worst, n_fail, ratios = _verdict(resid, bounds)
            if n_fail:
                failures.append(("stage", stage, worst))
            rep.lines.append(_fmt("", n_fail == 0, f"stage total {stage} over {len(tracked)} accounts",
                                  worst, resid, bounds, ratios, n_fail))

    # Census.
    terms_with_entries = {e.term for e in entries}
    for label in sorted({x[1] for x in failures if x[0] in ("quantity", "entry")}):
        terms_with_entries.add(label.split("/")[0])
    census_fail = []
    for term, status in sorted(census.items()):
        if status not in CENSUS_STATUSES:
            census_fail.append(("census-status", term, None))
            rep.lines.append(f"  FAIL census {term}: status {status!r} is not one of {CENSUS_STATUSES}")
        elif status == "entered" and term not in terms_with_entries:
            census_fail.append(("census-entered-empty", term, None))
            rep.lines.append(f"  FAIL census {term}: status entered but no entry")
        elif status == "not_tracked":
            rep.lines.append(f"  NOT TRACKED census {term}")
    for term in sorted(terms_with_entries):
        if term not in census:
            census_fail.append(("census-missing", term, None))
            rep.lines.append(f"  FAIL census {term}: has entries but no census status")
    failures.extend(census_fail)
    n_status = {s: sum(1 for v in census.values() if v == s) for s in CENSUS_STATUSES}
    rep.lines.append(f"census: {'FAIL' if census_fail else 'PASS'} ({len(census)} terms: "
                     f"{n_status['entered']} entered, {n_status['zero']} zero, "
                     f"{n_status['not_tracked']} not_tracked; {len(census_fail)} gaps)")
    rep.lines.append(f"accounts closed: {len(rep.closed)} ({', '.join(rep.closed) or 'none'}); "
                     f"NOT TRACKED: {', '.join(not_tracked) or 'none'}")
    if failures:
        rep.exit_code = 1
        rep.lines.append(f"LEDGER: FAIL ({len(failures)} failures)")
    else:
        rep.lines.append("LEDGER: PASS")
    return rep


def run(path, stages=None, margin=MARGIN, out=print):
    try:
        rep = check(path, stages, margin)
    except CannotRun as exc:
        out(f"ledger_check: {path}")
        out(f"LEDGER: CANNOT RUN ({exc})")
        return 2, None
    for line in rep.lines:
        out(line)
    return rep.exit_code, rep


# ---------------------------------------------------------------- self-test
#
# Synthetic result files with planted closures and planted breaks. The state
# is built from the entries: per accepted step, each term draws a positive
# increment per cell; the debit account's cell inventories lose it and the
# credit account's gain it; the receipt entry is the sum of those increments.
# So closure holds to roundoff by construction, and each break is planted at a
# known account, interval and kind.

N_CELLS = 24
N_SAVES = 7
ENGAGE_SAVE = 3          # kinetic neutrals engage at this save
ACTIVE = np.arange(1, N_CELLS)   # cell 0 is a neutral-only plenum

# (term, quantity, debit, credit, cells, log10 lo, log10 hi, leg_of)
# cells: index array, or "scalar" for a state account on both sides of the
# per-cell draw; increments are per cell per step.
_TERMS = [
    ("ionization", "particles", "neutral_particles", "plasma_particles", ACTIVE, 12, 14, None),
    ("recombination", "particles", "plasma_particles", "neutral_particles", ACTIVE, 10, 11, None),
    ("anode_collection", "particles", "plasma_particles", "anode", np.arange(2, 5), 11, 12, None),
    ("end_wall_loss", "particles", "plasma_particles", "end_wall", np.array([N_CELLS - 1]), 9, 10, None),
    ("gas_puff", "particles", "puff", "neutral_particles", np.arange(3, 9), 12, 13, None),
    ("pump", "particles", "neutral_particles", "pump", np.arange(N_CELLS - 3, N_CELLS), 11, 12, None),
    ("floor_density", "particles", "numerical_floors", "plasma_particles", np.array([5]), 8, 9, None),
    ("ei_exchange", "energy_e", "electron_energy", "ion_energy", ACTIVE, 1, 3, None),
    ("inelastic", "energy_e", "electron_energy", "radiation", ACTIVE, 1, 2, None),
    ("ionization_cost", "energy_e", "electron_energy", "ionization_potential", ACTIVE, 0, 2, None),
    ("ohmic", "energy_e", "circuit", "electron_energy", ACTIVE, 1, 2, None),
    ("bank_supply", "energy_e", "bank", "circuit", np.array([0]), 3, 3.5, None),
    ("cathode_sheath_heat", "energy_e", "electron_energy", "cathode_surface", np.array([1]), 2, 3, None),
    ("rusanov_dissipation", "energy_k", "plasma_kinetic_energy", "ion_energy", ACTIVE, -2, 0, None),
    ("kinetic_drive", "energy_k", "ion_energy", "plasma_kinetic_energy", ACTIVE, -1, 0, None),
    ("cx_heating", "energy_i", "ion_energy", "neutral_energy", ACTIVE, 0, 1, None),
    ("ion_end_wall", "energy_i", "ion_energy", "end_wall", np.array([N_CELLS - 1]), 1, 2, None),
    ("neutral_wall_loss", "energy_n", "neutral_energy", "end_wall", np.array([0]), 1, 2, None),
    ("floor_energy", "energy_e", "numerical_floors", "electron_energy", np.array([5]), -4, -3, None),
    ("cx_friction_plasma", "momentum", "plasma_momentum", "neutral_momentum", ACTIVE, -4, -3, "cx_friction"),
    ("cx_friction_neutral", "momentum", "plasma_momentum", "neutral_momentum", ACTIVE, -4, -3, "cx_friction"),
    ("pressure_end_wall", "momentum", "plasma_momentum", "end_wall", np.array([N_CELLS - 1]), -3, -2, None),
    ("neutral_wall_drag", "momentum", "neutral_momentum", "end_wall", np.array([0]), -4, -3, None),
    ("source_push", "momentum", "cathode", "plasma_momentum", np.array([1]), -3, -2, None),
]
# The planted unbooked term of the "missing term" scenario.
_MISSING = ("wall_recycle", "particles", "end_wall", "neutral_particles", np.array([N_CELLS - 1]), 13, 13.5, None)
_MISSING_INTERVAL = 1
# Neutral zone each particle term acts on (A2's column/annulus split).
_NEUTRAL_ZONE = {"ionization": "col", "recombination": "col", "gas_puff": "ann", "pump": "ann",
                 "wall_recycle": "ann"}


def _build(path, variant=None, seed=20261002):
    """Write one synthetic result file; returns construction facts the scenarios use."""
    v = dict(variant or {})
    rng = np.random.default_rng(seed)
    vp = 10 ** rng.uniform(4, 5.3, N_CELLS)
    vn = vp / rng.uniform(0.2, 0.5, N_CELLS)
    va = vn - vp
    steps = rng.integers(4, 12, N_SAVES - 1)
    dt = 1e-7 * rng.uniform(0.5, 1.5, int(steps.sum()))
    tb = np.concatenate([[0.0], np.cumsum(dt)])
    save_idx = np.concatenate([[0], np.cumsum(steps)])
    time = tb[save_idx]

    # Cell inventories (particles per cell, erg per cell, g cm/s per cell).
    act = np.zeros(N_CELLS, bool)
    act[ACTIVE] = True
    cell = {
        "plasma_particles": np.where(act, 10 ** rng.uniform(16, 18, N_CELLS), 0.0),
        "neutral_col": 10 ** rng.uniform(16, 18, N_CELLS),
        "neutral_ann": 10 ** rng.uniform(16.5, 18.5, N_CELLS),
        "electron_energy": np.where(act, 10 ** rng.uniform(4, 7, N_CELLS), 0.0),
        "ion_energy": np.where(act, 10 ** rng.uniform(3, 6, N_CELLS), 0.0),
        "plasma_kinetic_energy": np.where(act, 10 ** rng.uniform(2, 5, N_CELLS), 0.0),
        "plasma_momentum": np.where(act, rng.choice([-1.0, 1.0], N_CELLS) * 10 ** rng.uniform(-1, 2, N_CELLS), 0.0),
    }
    scalar = {"circuit": 1.3e6, "cathode_surface": 2.1e9, "neutral_energy": 4.7e7, "neutral_momentum": -3.1e1}

    stages_present = v.get("stages", list(STAGES))
    terms = [t for t in _TERMS if QUANTITY_STAGE[t[1]] in stages_present]
    if v.get("missing_term"):
        terms = terms + [_MISSING]

    def apply(acct, cells, incr, sign, term):
        if acct in scalar:
            scalar[acct] += sign * incr.sum()
        elif acct == "neutral_particles":
            cell["neutral_" + _NEUTRAL_ZONE[term]][cells] += sign * incr
        elif acct in cell:
            cell[acct][cells] += sign * incr
        # external accounts hold no inventory

    snap = {k: [a.copy()] for k, a in cell.items()}
    ssnap = {k: [s] for k, s in scalar.items()}
    step_amount = {t[0]: [] for t in terms}     # per step, summed over cells
    cell_acc = {t[0]: None for t in terms}      # per cell, summed over steps (the neutral-side leg)
    neutral_leg_acc = []
    for k in range(N_SAVES - 1):
        for _ in range(steps[k]):
            for term, quantity, debit, credit, cells, lo, hi, leg in terms:
                if term == "cx_friction_neutral":
                    continue  # the second site's leg of the cx_friction draw below
                incr = 10 ** rng.uniform(lo, hi, len(cells))
                if term == _MISSING[0] and k != _MISSING_INTERVAL:
                    incr = np.zeros(len(cells))
                apply(debit, cells, incr, -1.0, term)
                apply(credit, cells, incr, +1.0, term)
                step_amount[term].append(incr.sum())
                if term == "cx_friction_plasma":
                    neutral_leg_acc.append((k, incr.copy()))
        for key, a in cell.items():
            snap[key].append(a.copy())
        for key, s in scalar.items():
            ssnap[key].append(s)

    # Receipt totals per step, then per save interval.
    per_step = {term: np.array(a) for term, a in step_amount.items()}
    if "cx_friction_plasma" in per_step:
        # The neutral site sums the same increments in the other order: per
        # cell over the steps of the interval, then over the cells.
        nl = []
        for k in range(N_SAVES - 1):
            acc = np.zeros(len(ACTIVE))
            for kk, incr in neutral_leg_acc:
                if kk == k:
                    acc += incr
            nl.append(acc.sum())
        per_save_neutral_leg = np.array(nl)
    edges = save_idx
    per_save = {term: np.array([a[edges[k]:edges[k + 1]].sum() for k in range(N_SAVES - 1)])
                for term, a in per_step.items()}
    if "cx_friction_plasma" in per_step:
        per_save["cx_friction_neutral"] = per_save_neutral_leg
        # The step-cadence leg: per step, the same increments in the same order.
        per_step["cx_friction_neutral"] = per_step["cx_friction_plasma"].copy()

    inv_p = np.array(snap["plasma_particles"])
    facts = {"steps": steps, "plasma_inv": inv_p, "per_save": per_save, "time": time}

    # Planted breaks applied to the state.
    if "closure_break" in v:
        acct, kb, factor = v["closure_break"]
        assert acct == "plasma_particles"
        # The bound, written here independently of the checker from its
        # stated formula: margin * count * 2**-53 * gross with
        # count = S*(E+1) + 4*C + E*(S+C+1) + E + 2.
        touching = [t for t in terms if acct in (t[2], t[3])]
        E, S, C = len(touching), int(steps[kb]), len(ACTIVE)
        gross = (np.abs(inv_p[kb][ACTIVE]).sum() + np.abs(inv_p[kb + 1][ACTIVE]).sum()
                 + sum(abs(per_save[t[0]][kb]) for t in touching))
        count = S * (E + 1) + 4 * C + E * (S + C + 1) + E + 2
        bound = 4.0 * count * 2.0 ** -53 * gross
        for j in range(kb + 1, N_SAVES):
            snap["plasma_particles"][j][5] += factor * bound
        facts["planted_delta"] = factor * bound
    if "leg_disagree" in v:
        kl = v["leg_disagree"]
        mom_gross = sum(np.abs(a).sum() for a in snap["plasma_momentum"])
        D = 1e-6 * mom_gross   # far above any roundoff bound (~1e-13 relative)
        per_save["cx_friction_neutral"][kl] += D
        for j in range(kl + 1, N_SAVES):
            ssnap["neutral_momentum"][j] += D
        facts["leg_D"] = D

    # Saved fields (A1-A4).
    def field(key, vol, mask=None):
        a = np.array(snap[key]) / vol
        if mask is not None:
            a[:, ~mask] = 0.0
        return a

    n = field("plasma_particles", vp, act)
    n[:, 0] = 10 ** rng.uniform(5, 9, N_SAVES)   # plenum garbage the plasma_active mask must drop
    nn = field("neutral_col", vp)
    nn_a = field("neutral_ann", va)
    # After engagement the saved neutral fields are republished moments that
    # need not hold the inventory; the receipt's state does.
    nn[ENGAGE_SAVE:] *= 1.0 + 1e-6
    nn_a[ENGAGE_SAVE:] *= 1.0 - 1e-6
    kin = np.array([snap["neutral_col"][j].sum() + snap["neutral_ann"][j].sum() for j in range(N_SAVES)])
    engaged = (np.arange(N_SAVES) >= ENGAGE_SAVE).astype(np.int8)
    Ee = field("electron_energy", vp, act)
    Ei = field("ion_energy", vp, act)
    mom = field("plasma_momentum", vp, act)
    kdens = field("plasma_kinetic_energy", vp, act)
    u = np.zeros_like(mom)
    u[:, act] = 2.0 * kdens[:, act] / mom[:, act]

    with h5py.File(path, "w") as f:
        f.attrs["format"] = "sim1d-hdf5-v1"
        f["time"] = time
        for name, a in (("n", n), ("nn", nn), ("nn_a", nn_a), ("momentum", mom), ("Ee", Ee),
                        ("Ei", Ei), ("u", u)):
            f[name] = a
        g = f.create_group("geometry")
        g["plasma_volume_cm3"] = vp
        g["neutral_volume_cm3"] = vn
        g["volume_ratio"] = vp / vn
        g["plasma_active"] = act
        g["cell_role"] = np.array([b"plenum"] + [b"column"] * (N_CELLS - 1), dtype=object)
        g["length_cm"] = np.full(N_CELLS, 10.0)
        if v.get("no_receipt"):
            return facts
        rc = f.create_group("receipt")
        rc.attrs["schema"] = SCHEMA
        cadence = v.get("cadence", "save")
        rc.attrs["cadence"] = cadence
        rc.attrs["stages_present"] = np.array(stages_present, dtype=object)
        if cadence == "save":
            rc["interval_t0"], rc["interval_t1"] = time[:-1], time[1:]
            rc["interval_steps"] = steps.astype(np.int64)
            data = per_save
        else:
            rc["interval_t0"], rc["interval_t1"] = tb[:-1], tb[1:]
            rc["interval_steps"] = np.ones(tb.size - 1, np.int64)
            data = per_step
        st = rc.create_group("state")
        st["neutral_particles_kinetic"] = kin
        st["neutral_kinetic_engaged"] = engaged
        for key in scalar:
            st[key] = np.array(ssnap[key])
        if v.get("not_tracked") == "circuit":
            st["circuit"][...] = st["circuit"][()] * (1.0 + rng.uniform(-1e-3, 1e-3, N_SAVES))
        eg = rc.create_group("entries")
        census = {}
        for term, quantity, debit, credit, cells, lo, hi, leg in terms:
            if term == _MISSING[0]:
                continue  # applied to the state, absent from the receipt
            if v.get("swap") == term:
                debit, credit = credit, debit
            if v.get("unknown_account") == term:
                credit = credit + "p"   # the external side, so no inventory moves
            ds = eg.create_group(term).create_dataset(quantity, data=data[term])
            ds.attrs["debit"], ds.attrs["credit"] = debit, credit
            ds.attrs["site"] = "synthetic"
            ds.attrs["units"] = QUANTITY_UNITS[quantity]
            if leg:
                ds.attrs["leg_of"] = leg
            census[term] = "entered"
        census["recycling_jet"] = "zero"
        if v.get("census_missing"):
            del census[v["census_missing"]]
        if v.get("census_entered_empty"):
            census[v["census_entered_empty"]] = "entered"
        if v.get("not_tracked"):
            census[v["not_tracked"]] = "not_tracked"
        cg = rc.create_group("census")
        for term, status in census.items():
            c = cg.create_group(term)
            if v.get("census_nostatus") != term:
                c.attrs["status"] = status
            c.attrs["reason"] = "synthetic"
    return facts


def self_test(verbose=False):
    # (name, variant, requested stages, expected exit, expected failures,
    #  extra expectations). A failure is (kind, name, interval); interval None
    #  means "any interval" (a break planted in every interval).
    swap_I = None   # the swap is planted in every interval
    scenarios = [
        ("clean", {}, None, 0, [], {}),
        ("clean, step cadence", {"cadence": "step"}, None, 0, [], {}),
        ("closure break x100 (plasma_particles, interval 2)",
         {"closure_break": ("plasma_particles", 2, 100.0)}, None, 1,
         [("closure", "particles/plasma_particles", 2), ("stage", "particles", 2)], {}),
        ("closure break x0.1 (plasma_particles, interval 2)",
         {"closure_break": ("plasma_particles", 2, 0.1)}, None, 0, [], {}),
        ("debit and credit swapped (ionization)", {"swap": "ionization"}, None, 1,
         [("closure", "particles/plasma_particles", swap_I),
          ("closure", "particles/neutral_particles", swap_I)], {}),
        ("term applied but not booked (wall_recycle, interval 1)", {"missing_term": True}, None, 1,
         [("closure", "particles/neutral_particles", _MISSING_INTERVAL),
          ("stage", "particles", _MISSING_INTERVAL)], {}),
        ("legs disagree (cx_friction, interval 4)", {"leg_disagree": 4}, None, 1,
         [("leg", "cx_friction/momentum", 4), ("closure", "momentum/plasma_momentum", 4),
          ("closure", "momentum/neutral_momentum", 4), ("stage", "momentum", 4)], {}),
        ("census term with no status (inelastic)", {"census_nostatus": "inelastic"}, None, 1,
         [("census-status", "inelastic", None)], {}),
        ("entered term with no entry (beam_ionization)", {"census_entered_empty": "beam_ionization"},
         None, 1, [("census-entered-empty", "beam_ionization", None)], {}),
        ("entry whose term is not in the census (gas_puff)", {"census_missing": "gas_puff"}, None, 1,
         [("census-missing", "gas_puff", None)], {}),
        ("unknown account name (pump -> pumpp)", {"unknown_account": "pump"}, None, 1,
         [("account", "pump/particles", None)], {}),
        ("requested stage absent (momentum)", {"stages": ["particles"]}, ["momentum"], 2, [], {}),
        ("no receipt", {"no_receipt": True}, None, 2, [], {}),
        ("not_tracked account (circuit, its state corrupted)", {"not_tracked": "circuit"}, None, 0, [],
         {"not_tracked": ["circuit"], "closed_excludes": "energy/circuit",
          "closed_includes": "energy/electron_energy"}),
    ]
    bad = 0
    with tempfile.TemporaryDirectory(prefix="ledger_selftest_") as tmp:
        for i, (name, variant, stages, want_exit, want_fail, extra) in enumerate(scenarios):
            path = os.path.join(tmp, f"scenario_{i:02d}.h5")
            _build(path, variant)
            lines = []
            code, rep = run(path, stages, MARGIN, out=lines.append)
            got = sorted(rep.failures, key=str) if rep else []
            ok = code == want_exit
            if ok and rep is not None:
                unmatched = list(got)
                for kind, nm, iv in want_fail:
                    hit = [g for g in unmatched if g[0] == kind and g[1] == nm and (iv is None or g[2] == iv)]
                    if not hit:
                        ok = False
                    for h in hit:
                        unmatched.remove(h)
                if unmatched:
                    ok = False
                if "not_tracked" in extra and rep.not_tracked != extra["not_tracked"]:
                    ok = False
                if "closed_excludes" in extra and extra["closed_excludes"] in rep.closed:
                    ok = False
                if "closed_includes" in extra and extra["closed_includes"] not in rep.closed:
                    ok = False
            bad += not ok
            want_txt = f"exit {want_exit} {[(k, n, '*' if iv is None else iv) for k, n, iv in want_fail]}"
            got_txt = f"exit {code} {got}"
            if extra.get("not_tracked") and rep is not None:
                got_txt += f" not_tracked={rep.not_tracked} closed={len(rep.closed)}"
                want_txt += f" not_tracked={extra['not_tracked']}"
            print(f"{'OK  ' if ok else 'MISMATCH'} {name}: expected {want_txt}; got {got_txt}")
            if verbose or not ok:
                for line in lines:
                    print("      " + line)
    print(f"SELF-TEST: {'PASS' if bad == 0 else f'FAIL ({bad} scenarios differ)'}")
    return 0 if bad == 0 else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("result", nargs="?", help="a sim1d-hdf5-v1 result file")
    ap.add_argument("--stage", action="append", choices=STAGES,
                    help="stage to check (repeatable); default every stage in stages_present")
    ap.add_argument("--margin", type=float, default=MARGIN,
                    help=f"safety factor on the roundoff count (default {MARGIN:g})")
    ap.add_argument("--self-test", action="store_true",
                    help="build synthetic files with planted closures and breaks and check each")
    ap.add_argument("--verbose", action="store_true", help="self-test: print every report")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test(args.verbose)
    if not args.result:
        ap.error("a result file or --self-test is required")
    code, _ = run(args.result, args.stage, args.margin)
    return code


if __name__ == "__main__":
    sys.exit(main())
