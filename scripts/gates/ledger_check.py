"""Conservation-receipt checker for ``sim1d-hdf5-v1`` result files.

A result file may carry a conservation RECEIPT: for each saved interval, every
amount a term moved between two accounts. This checker reads the receipt and
the saved state, recomputes each account's inventory from the state, and
decides whether every account closes. It is written from the model's stated
conservation laws and the result file's layout only; it imports nothing from
the solver, so a receipt the solver writes is checked by arithmetic the solver
did not supply.

Usage::

    python scripts/gates/ledger_check.py RUN.h5 [--stage particles|energy|momentum ...]
                                                [--margin M] [--ceiling R]
    python scripts/gates/ledger_check.py --self-test

Exit status: 0 every requested check passes; 1 a closure, exchange,
stage-total or bound-ceiling failure, an invalid entry, a steps mismatch, an
orphan exchange state, or a census gap; 2 the check could not run (no
``receipt/`` group, an unknown schema, a requested stage the receipt does not
hold, intervals that do not tile the saves, no ``geometry/plasma_volume_cm3``
or ``geometry/plasma_active``). A missing receipt is exit 2, never a pass.

RECEIPT SCHEMA (``receipt-v1``)::

    receipt/
      attrs: schema = "receipt-v1", cadence ("save" | "step"), stages_present (e.g. ["particles"]),
             updates_per_step (int, optional, default 1)
      interval_t0, interval_t1        (n_intervals,) float64, seconds
      interval_steps                  (n_intervals,) int64, accepted steps accumulated in the interval
      entries/<term>/<quantity>       (n_intervals,) float64, volume-integrated total over the interval
          attrs: debit, credit (account names), site, units, sign ("one-signed" | "signed")
      entries/<term>/<quantity>_gross (n_intervals,) float64, accumulated sum of the magnitudes
                                      of the contributions that made the entry
      state/<name>                    (n_saves,) extra state the saved fields do not hold
      census/<term>                   attrs: status = "entered" | "zero" | "not_tracked", reason

* ``quantity`` is one of ``particles``, ``momentum``, ``energy_e``,
  ``energy_i``, ``energy_k``, ``energy_n``; each entry's ``units`` attr must
  be its quantity's unit: ``"particles"``, ``"g cm/s"``, ``"erg"`` (the four
  energies). Stages: ``particles`` (quantity particles), ``energy`` (the four
  energy quantities), ``momentum``.
* An entry moves its amount from the ``debit`` account to the ``credit``
  account: a positive amount raises the credit account's inventory and lowers
  the debit account's.
* SELF-ENTRIES: an entry whose debit and credit are the same account books
  movement inside that account (cell-to-cell transport). It is valid only on
  an account with a cell-resolved inventory (the plasma and neutral accounts),
  contributes zero to the account's closure, and contributes its gross to the
  account's bound: transport inside an account rounds relative to fluxes that
  no boundary entry carries. On a clearing, circuit, cathode-surface or
  external account it fails.
* Every entry has its ``<quantity>_gross`` companion, and the bounds below use
  it rather than the entry's magnitude: an entry whose contributions cancel
  carries rounding of the size of its gross, not of its net. Every entry
  declares ``sign``: a ``one-signed`` entry's gross equals its magnitude
  within ``GROSS_ULPS`` ulps in both directions; a ``signed`` entry's gross
  may exceed its magnitude (the report prints gross/|net|) but never falls
  below it by more than ``GROSS_ULPS`` ulps.
* Cadence ``save``: one interval per pair of consecutive saves, interval k
  running from save k to save k+1 (``interval_t0``/``interval_t1`` equal to the
  saved ``time`` exactly). Cadence ``step``: the intervals tile the saves
  contiguously, and the checker sums those between each pair of saves. When
  the file has a root attr ``steps``, the sum of ``interval_steps`` must equal
  it.
* An exchange computed at two code sites is booked as two ordinary entries
  through a CLEARING account ``exchange:<name>``: the site that removes the
  quantity writes debit = the account it changed, credit = ``exchange:<name>``;
  the site that adds it writes debit = ``exchange:<name>``, credit = the
  account it changed. Each entry touches only the state its own site changed.
  The clearing account's inventory is ``receipt/state/exchange_<name>`` when
  that dataset exists (a declared carried debt) and identically zero
  otherwise; its closure is the agreement of the two sites. A clearing account
  belongs to the stage of the quantity booked through it; one with entries on
  only one side is a failure, and a ``state/exchange_<name>`` with a nonzero
  value that no entry books is a failure.
* Accounts with an inventory: ``plasma_particles``, ``neutral_particles``,
  ``electron_energy``, ``ion_energy``, ``plasma_kinetic_energy``,
  ``plasma_momentum``, ``neutral_energy``, ``neutral_momentum``, ``circuit``,
  ``cathode_surface``, and every ``exchange:<name>``. External accounts
  (entries only, no inventory): ``cathode``, ``anode``, ``end_wall``,
  ``radiation``, ``ionization_potential``, ``pump``, ``puff``,
  ``numerical_floors``, ``bank``. Any other account name is a failure (and is
  treated as external in the arithmetic, so the failure names the entry, not
  its neighbours).
* ``state/neutral_kinetic_engaged`` is a 0/1 flag per save. Once the kinetic
  neutrals engage, ``state/neutral_particles_kinetic``,
  ``state/neutral_momentum`` and ``state/neutral_energy`` hold those
  inventories; before, the saved fields do (``nn``/``nn_a``, ``M_n``/``M_n_a``,
  ``En``). Where a neutral momentum or energy field is absent, the state
  dataset holds the inventory at every save; with neither, the account must be
  declared ``not_tracked`` in the census or it fails. ``state/circuit`` and
  ``state/cathode_surface`` hold those inventories; when the energy stage is
  checked and either has no state series, it must likewise be declared
  ``not_tracked`` or it fails.
* A census key may also name an inventoried account; ``not_tracked`` on it
  marks the account NOT TRACKED: it is reported, never counted as closed, and
  in its stage total it stands on the boundary like an external account.

WHAT IT CHECKS, per requested stage (default: every stage in
``stages_present``):

1. Account closure: for each inventoried account (clearing accounts included,
   reported as exchanges) and each save interval, the inventory change
   recomputed from the saved state equals the entries credited to it minus
   those debited from it.
2. Bound ceiling: an account whose bound exceeds ``CEILING`` times its
   inventory's summand magnitudes at the two saves is CANNOT CERTIFY; a bound
   that loose decides nothing, and the ceiling caps what an inflated gross or
   step count can buy. A clearing account with no carried debt is held against
   the inventory magnitudes of its tracked site accounts, or, with none, its
   entries' summed |net|.
3. Stage total: summed over the stage's tracked inventoried accounts, the
   change equals the entries from accounts outside that set minus the entries
   to them. When every account is tracked this is the sum of the closures of
   check 1 and adds nothing they do not already decide; it adds information
   only where an account is not tracked.
4. Entries and steps (once): known quantity, debit and credit present,
   accounts known, ``units`` equal to the quantity's unit, ``sign`` declared,
   a gross present and consistent with the entry and its sign; the steps
   cross-check; no orphan clearing-account state.
5. Census (once, over all terms): every census term has a valid status; an
   ``entered`` term has at least one entry; a ``zero`` or ``not_tracked`` term
   has no entry with a nonzero value; every entry's term is in the census. An
   ``entered`` term whose entries are all zero is named in the report.

THE BAR for 1 and 3 is a roundoff bound fixed before any run, never a
physical scale or a fitted number and never relative to the net change::

    bound = margin * count * 2**-53 * gross

``gross`` is the sum of the magnitudes of everything entering the comparison
(each entry's ``_gross`` plus, for each inventory, the sum of its summands'
magnitudes at both saves) and ``count`` the floating-point operations that
produced the two sides (derived beside :func:`_closure_count`).

WHAT THE COUNT ASSUMES about the writer's arithmetic: each accepted step
updates each cell's state ``updates_per_step`` times (1 unless the receipt
declares more), each update adding at most E increments per cell; the saved
fields are float64; each entry is accumulated beside its gross, over the same
contributions in the same order; a saved density is the stored state itself,
not the result of a division. A writer using multi-stage or split updates
must either show its arithmetic stays inside the count or declare the number
of state updates per step in ``updates_per_step``, which multiplies the
per-step terms of the count.

INVENTORIES. The model documents state the plasma lives in the column of each
cell, the neutrals in the column and the annulus, and the conservative fields
are densities; how the saved arrays combine with the saved volumes is as
follows:

* A1 (plasma zone): a plasma inventory is the sum over ``plasma_active``
  cells of the field times ``geometry/plasma_volume_cm3`` (the column volume
  V_col = A dz).
* A2 (neutral zones): with an ``nn_a`` dataset the neutrals are two-zone,
  ``nn`` the column density and ``nn_a`` the annulus density, and the
  inventory is ``nn*plasma_volume_cm3 + nn_a*(neutral_volume_cm3 -
  plasma_volume_cm3)`` summed over ALL cells; without ``nn_a`` they are
  single-zone and the inventory is ``nn*neutral_volume_cm3`` over all cells.
  Neutral momentum: with ``M_n`` and ``M_n_a`` the inventory is
  ``M_n*plasma_volume_cm3 + M_n_a*(neutral_volume_cm3 - plasma_volume_cm3)``;
  with ``M_n`` alone, ``M_n`` is a chamber-mean field and the inventory is
  ``M_n*neutral_volume_cm3``, whatever the density layout. Neutral energy:
  ``En`` is a column field, ``En*plasma_volume_cm3`` in a two-zone file and
  ``En*neutral_volume_cm3`` in a single-zone file. All over ALL cells.
* A3 (field units): the saved fields are CGS per cm^3 (``n``, ``nn``,
  ``nn_a`` in cm^-3; ``momentum``, ``M_n`` in g cm^-2 s^-1; ``Ee``, ``Ei``,
  ``En`` in erg cm^-3), so a field times a volume in cm^3 is in the receipt's
  units.
* A4 (kinetic energy): the plasma kinetic energy density is
  ``0.5 * momentum * u`` with ``u`` the saved velocity field (cm/s).
* A5 (state accounts): an inventory taken from ``receipt/state/`` is a single
  summand, counted in the inventory terms as a sum over the file's cells.
* A6 (stage of the state accounts): ``circuit`` and ``cathode_surface`` hold
  energy and belong to the ``energy`` stage.
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

# Largest bound, relative to the inventory's summand magnitudes at the two
# saves, at which a closure still certifies anything. 1e-9 is about 1e7 unit
# roundoffs: a count of 1e3 operations at margin 4 against a gross up to ~2e3
# times the inventory. Beyond it the account is CANNOT CERTIFY.
CEILING = 1e-9

# Gross against entry. With the gross accumulated beside the entry over the
# same contributions in the same order, |fl(sum c)| <= fl(sum |c|) holds
# exactly, for signed contributions too: by induction |s_k| <= t_k, because
# |s_{k-1} + c_k| <= t_{k-1} + |c_k| and round-to-nearest is monotone. For
# one-signed contributions the two accumulations are the same computation up
# to sign and agree to the bit. GROSS_ULPS ulps of |entry| allow only for a
# final step the writer does outside that order (a conversion, or summing the
# receipt intervals of a save interval in another order).
GROSS_ULPS = 4

EXCHANGE_PREFIX = "exchange:"
# Accounts on which a self-entry (debit == credit, transport inside the
# account) has a meaning: those inventoried from cell-resolved stores.
SELF_ENTRY_ACCOUNTS = frozenset({
    "plasma_particles", "neutral_particles", "electron_energy", "ion_energy",
    "plasma_kinetic_energy", "plasma_momentum", "neutral_energy", "neutral_momentum",
})
SIGNS = ("one-signed", "signed")

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
# Accounts that must have an inventory or be declared not_tracked.
REQUIRED_INVENTORY = ("neutral_momentum", "neutral_energy", "circuit", "cathode_surface")
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


def _is_exchange(acct):
    return acct.startswith(EXCHANGE_PREFIX) and len(acct) > len(EXCHANGE_PREFIX)


def _known_account(acct):
    return acct in INVENTORIED or acct in EXTERNAL or _is_exchange(acct)


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


def _neutral_field(f, name, vp, vn, two_zone, n_cells):
    """A neutral inventory from the saved fields, or None (A2).

    ``nn``: two-zone (``nn_a`` present) column plus annulus, else chamber.
    ``M_n``: with ``M_n_a``, column plus annulus; without it, ``M_n`` is a
    chamber-mean field on the neutral volume, whatever the density layout.
    ``En``: a column field, on the plasma volume in a two-zone file and on the
    neutral volume in a single-zone file; there is no annulus energy field.
    """
    if name not in f:
        return None
    x = f[name][()]
    if name == "nn":
        if two_zone:
            parts = np.concatenate([x * vp, f["nn_a"][()] * (vn - vp)], axis=1)
            return _summed(parts, 2 * n_cells)
        return _summed(x * vn, n_cells)
    if name == "M_n":
        if "M_n_a" in f:
            parts = np.concatenate([x * vp, f["M_n_a"][()] * (vn - vp)], axis=1)
            return _summed(parts, 2 * n_cells)
        return _summed(x * vn, n_cells)  # chamber-mean M_n
    if name == "En":
        return _summed(x * (vp if two_zone else vn), n_cells)
    raise ValueError(name)


def _engaged_merge(field, state_vals, engaged, n_cells):
    """Saved field before the kinetic neutrals engage, receipt state from then on."""
    if field is None:
        return None if state_vals is None else Inventory(state_vals, np.abs(state_vals), n_cells)
    if engaged is None or not engaged.any():
        return field
    if state_vals is None:
        return None
    field.values = np.where(engaged, state_vals, field.values)
    field.gross = np.where(engaged, np.abs(state_vals), field.gross)
    return field


def _load_inventories(f, rc):
    """Every inventory the file can supply, the cell count, and the clearing-account
    state series keyed by exchange name."""
    if "geometry" not in f or "plasma_volume_cm3" not in f["geometry"]:
        raise CannotRun("no geometry/plasma_volume_cm3")
    geo = f["geometry"]
    if "plasma_active" not in geo:
        raise CannotRun("no geometry/plasma_active")
    state = rc["state"] if "state" in rc else None

    def st(name):
        return state[name][()].astype(np.float64) if state is not None and name in state else None

    inv = {}
    vp = geo["plasma_volume_cm3"][()].astype(np.float64)
    n_cells = vp.size
    active = geo["plasma_active"][()].astype(bool)
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

    eng = st("neutral_kinetic_engaged")
    engaged = eng.astype(bool) if eng is not None else None
    if st("neutral_particles_kinetic") is not None and engaged is None:
        raise CannotRun("state/neutral_particles_kinetic without state/neutral_kinetic_engaged")
    vn = geo["neutral_volume_cm3"][()].astype(np.float64) if "neutral_volume_cm3" in geo else None
    two_zone = "nn_a" in f
    for acct, field, state_name in (("neutral_particles", "nn", "neutral_particles_kinetic"),
                                    ("neutral_momentum", "M_n", "neutral_momentum"),
                                    ("neutral_energy", "En", "neutral_energy")):
        fld = _neutral_field(f, field, vp, vn, two_zone, n_cells) if vn is not None else None
        merged = _engaged_merge(fld, st(state_name), engaged, n_cells)
        if merged is not None:
            inv[acct] = merged
    for name in ("circuit", "cathode_surface"):  # A5
        v = st(name)
        if v is not None:
            inv[name] = Inventory(v, np.abs(v), n_cells)
    debts = {}
    if state is not None:
        for key in state.keys():
            if key.startswith("exchange_"):
                debts[key[len("exchange_"):]] = state[key][()].astype(np.float64)
    return inv, n_cells, debts


def _clearing_inventory(acct, debts, n_saves):
    """A clearing account's inventory: its declared carried debt, or identically zero."""
    name = acct[len(EXCHANGE_PREFIX):]
    if name in debts:
        v = debts[name]
        return Inventory(v, np.abs(v), 1)
    return Inventory(np.zeros(n_saves), np.zeros(n_saves), 0)


# ---------------------------------------------------------------- receipt

class Entry:
    def __init__(self, term, quantity, debit, credit, values, gross):
        self.term = term
        self.quantity = quantity
        self.stage = QUANTITY_STAGE.get(quantity)
        self.debit = debit
        self.credit = credit
        self.values = values
        self.gross = gross

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
    return groups, save_steps, n_sub, int(steps.sum())


def _per_save(raw, groups):
    return np.array([sum(raw[g].tolist()) if len(g) > 1 else raw[g[0]] for g in groups])


def _load_entries(rc, groups, failures, lines):
    """Entries per save interval; every invalid entry is recorded in ``failures``.
    Also returns, per term, whether any of its entries holds a nonzero value."""
    entries, nonzero = [], {}
    if "entries" not in rc:
        return entries, nonzero

    def fail(kind, label, why):
        failures.append((kind, label, None))
        lines.append(f"  FAIL entry {label}: {why}")

    for term in sorted(rc["entries"].keys()):
        tg = rc["entries"][term]
        keys = set(tg.keys())
        for quantity in sorted(keys):
            if quantity.endswith("_gross") and quantity[:-len("_gross")] in keys:
                continue
            ds = tg[quantity]
            label = f"{term}/{quantity}"
            raw = ds[()].astype(np.float64)
            nonzero[term] = nonzero.get(term, False) or bool(np.any(raw != 0.0))
            if quantity not in QUANTITY_STAGE:
                fail("quantity", label, f"unknown quantity {quantity!r}")
                continue
            if "debit" not in ds.attrs or "credit" not in ds.attrs:
                fail("entry", label, "debit or credit attr missing")
                continue
            debit, credit = _s(ds.attrs["debit"]), _s(ds.attrs["credit"])
            for acct in (debit, credit):
                if not _known_account(acct):
                    fail("account", label, f"unknown account {acct!r}")
            if debit == credit and _known_account(debit) and debit not in SELF_ENTRY_ACCOUNTS:
                fail("self-entry-invalid", label,
                     f"self-entry on {debit!r}, which holds no cell-resolved inventory")
            units = _s(ds.attrs["units"]) if "units" in ds.attrs else None
            if units != QUANTITY_UNITS[quantity]:
                fail("units", label, f"units {units!r}, quantity {quantity} is in "
                                     f"{QUANTITY_UNITS[quantity]!r}")
            sign = _s(ds.attrs["sign"]) if "sign" in ds.attrs else None
            if sign not in SIGNS:
                fail("sign", label, f"sign {sign!r} is not one of {SIGNS}")
            values = _per_save(raw, groups)
            gname = quantity + "_gross"
            if gname not in keys:
                fail("gross-missing", label, f"no {gname} companion; bounds use |entry|")
                gross = np.abs(values)
            else:
                graw = tg[gname][()].astype(np.float64)
                tol = GROSS_ULPS * np.spacing(np.abs(raw))
                bad = ~(graw >= np.abs(raw) - tol)
                if sign == "one-signed":
                    bad |= ~(graw <= np.abs(raw) + tol)
                if np.any(bad):
                    k = int(np.argmax(bad))
                    fail("gross-inconsistent", label,
                         f"gross {graw[k]:.17e} against |entry| {abs(raw[k]):.17e} in receipt "
                         f"interval {k} ({sign})")
                gross = _per_save(graw, groups)
                if sign == "signed":
                    with np.errstate(divide="ignore", invalid="ignore"):
                        r = gross / np.abs(values)
                    lines.append(f"  signed entry {label}: gross/|net| from {np.nanmin(r):.3e} "
                                 f"to {np.nanmax(r):.3e}")
            entries.append(Entry(term, quantity, debit, credit, values, gross))
    return entries, nonzero


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


def _closure_count(steps, n_entries, summands, cells, n_sub, updates):
    # Operations producing the two sides of one account's closure in one save
    # interval (`updates` = updates_per_step):
    #   updates*steps*(n_entries+1): each state update adds at most n_entries
    #       term increments to each cell's state and stores it once; each
    #       rounding is bounded relative to that cell's state or to the
    #       increment, and the cells' state magnitudes sum to the inventory
    #       gross, the increments' to the entries' gross;
    #   4*summands: each saved summand is a density times a volume (one
    #       rounding at each of the two saves) and the two inventory sums
    #       (summands-1 additions each); a clearing account has summands = 1
    #       with a declared debt and 0 when its inventory is identically zero;
    #   n_entries*(updates*steps + cells + n_sub): each entry is accumulated
    #       over the interval's state updates, each update's amount a sum over
    #       at most `cells` cells (the file's cell count, whatever the
    #       account's inventory looks like: a clearing account's entries are
    #       still per-cell sums at their sites), and the checker sums n_sub
    #       receipt intervals;
    #   n_entries: the checker sums the entries;
    #   2: the inventory difference and the residual.
    return (updates * steps * (n_entries + 1) + 4 * summands
            + n_entries * (updates * steps + cells + n_sub) + n_entries + 2)


def _ratio(resid, bound):
    if bound > 0:
        return abs(resid) / bound
    return 0.0 if resid == 0 else float("inf")


def _verdict(resid, bounds):
    ratios = np.array([_ratio(r, b) for r, b in zip(resid, bounds)])
    worst = int(np.argmax(ratios)) if ratios.size else 0
    n_fail = int(np.sum(ratios > 1.0))
    return worst, n_fail, ratios


def _fmt(ok, what, worst, resid, bounds, ratios, n_fail):
    if not len(resid):
        return f"  PASS {what}: no intervals"
    return (f"  {'PASS' if ok else 'FAIL'} {what}: worst interval {worst} residual "
            f"{resid[worst]:.3e} bound {bounds[worst]:.3e} ratio {ratios[worst]:.3e} "
            f"({n_fail}/{len(resid)} intervals over)")


def _closure(I, acct, touching, steps, n_sub, cells, updates, margin):
    resid, bounds = [], []
    for k in range(len(steps)):
        flow = 0.0
        mag = 0.0
        for e in touching:
            sign = (e.credit == acct) - (e.debit == acct)   # 0 for a self-entry
            flow += sign * e.values[k]
            mag += e.gross[k]
        resid.append((I.values[k + 1] - I.values[k]) - flow)
        gross = I.gross[k] + I.gross[k + 1] + mag
        count = _closure_count(steps[k], len(touching), I.summands, cells, n_sub[k], updates)
        bounds.append(_bound(margin, count, gross))
    return np.array(resid), np.array(bounds)


def _ceiling(ref, bounds, ceiling):
    """Worst interval of bound / reference magnitude and whether it exceeds the ceiling."""
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(ref > 0, bounds / ref, 0.0)
    worst = int(np.argmax(rel)) if rel.size else 0
    return worst, (rel[worst] if rel.size else 0.0), bool(np.any(rel > ceiling))


# ---------------------------------------------------------------- check

class Report:
    def __init__(self):
        self.lines = []
        self.failures = []        # (kind, name, interval or None)
        self.closed = []          # "stage/account" that closed
        self.not_tracked = []     # account names
        self.bounds = {}          # "stage/account" -> per-interval bounds
        self.entered_all_zero = []
        self.exit_code = 0


def check(path, stages=None, margin=MARGIN, ceiling=CEILING):
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
        updates = int(rc.attrs.get("updates_per_step", 1))
        if updates < 1:
            raise CannotRun(f"updates_per_step {updates} < 1")
        if "time" not in f:
            raise CannotRun("no time dataset")
        time = f["time"][()]
        groups, steps, n_sub, total_steps = _save_groups(rc, time)
        inv, n_cells, debts = _load_inventories(f, rc)
        rep.lines.append(f"ledger_check: {path}")
        rep.lines.append(f"  schema {SCHEMA}, {len(groups)} save intervals, stages {stages}, "
                         f"margin {margin:g}, ceiling {ceiling:g}, updates_per_step {updates}")
        failures = rep.failures
        if "steps" in f.attrs and int(f.attrs["steps"]) != total_steps:
            failures.append(("steps-mismatch", "steps", None))
            rep.lines.append(f"  FAIL steps: receipt interval_steps sum to {total_steps}, "
                             f"the file's steps attr is {int(f.attrs['steps'])}")
        entries, nonzero = _load_entries(rc, groups, failures, rep.lines)
        census = _load_census(rc)

    not_tracked = sorted(a for a, st in census.items() if a in INVENTORIED and st == "not_tracked")
    rep.not_tracked = not_tracked

    # Clearing accounts, the stage of each, and orphan clearing-account state.
    clearing_stages = {}
    for e in entries:
        for acct in (e.debit, e.credit):
            if _is_exchange(acct):
                clearing_stages.setdefault(acct, set()).add(e.stage)
    for acct, sts in sorted(clearing_stages.items()):
        if len(sts) > 1:
            failures.append(("stage-mismatch", acct, None))
            rep.lines.append(f"  FAIL exchange {acct}: booked in stages {sorted(sts)}")
    for name, v in sorted(debts.items()):
        acct = EXCHANGE_PREFIX + name
        if acct not in clearing_stages and np.any(v != 0.0):
            failures.append(("exchange-orphan-state", acct, None))
            rep.lines.append(f"  FAIL exchange {acct}: state/exchange_{name} holds nonzero values "
                             f"but no entry books the clearing account")

    for stage in stages:
        rep.lines.append(f"stage {stage}")
        stage_entries = [e for e in entries if e.stage == stage]
        for e in stage_entries:
            for acct in (e.debit, e.credit):
                if acct in INVENTORIED and INVENTORIED[acct] != stage:
                    failures.append(("stage-mismatch", e.label, None))
                    rep.lines.append(f"  FAIL entry {e.label}: account {acct} holds stage "
                                     f"{INVENTORIED[acct]}, entry is stage {stage}")

        tracked = []

        def certify(acct, label, I, touching, ref=None, ref_name="inventory magnitude"):
            name = f"{stage}/{acct}"
            resid, bounds = _closure(I, acct, touching, steps, n_sub, n_cells, updates, margin)
            rep.bounds[name] = bounds
            worst, n_fail, ratios = _verdict(resid, bounds)
            if ref is None:
                ref = I.gross[:-1] + I.gross[1:]
            cw, crel, over = _ceiling(ref, bounds, ceiling)
            rep.lines.append(_fmt(n_fail == 0, label, worst, resid, bounds, ratios, n_fail))
            if over:
                failures.append(("bound-ceiling", name, cw))
                rep.lines.append(f"  CANNOT CERTIFY {acct}: bound / {ref_name} {crel:.3e} "
                                 f"exceeds the ceiling {ceiling:g} (worst interval {cw})")
            return n_fail, over, worst

        # Account closure.
        for acct, acct_stage in INVENTORIED.items():
            if acct_stage != stage:
                continue
            touching = [e for e in stage_entries if acct in (e.debit, e.credit)]
            if acct in not_tracked:
                rep.lines.append(f"  NOT TRACKED {acct} ({len(touching)} entries touch it; "
                                 f"census status not_tracked)")
                continue
            if acct not in inv:
                if touching or acct in REQUIRED_INVENTORY:
                    failures.append(("no-inventory", f"{stage}/{acct}", None))
                    rep.lines.append(f"  FAIL closure {acct}: the file holds no inventory for it and "
                                     f"the census does not declare it not_tracked")
                continue
            tracked.append(acct)
            n_fail, over, worst = certify(acct, f"closure {acct}", inv[acct], touching)
            if n_fail:
                failures.append(("closure", f"{stage}/{acct}", worst))
            elif not over:
                rep.closed.append(f"{stage}/{acct}")

        # Exchanges: clearing-account closure is the agreement of the two sites.
        for acct in sorted(a for a, sts in clearing_stages.items() if stage in sts):
            touching = [e for e in stage_entries if acct in (e.debit, e.credit)]
            sides = ", ".join(f"{e.term}: {e.debit} -> {e.credit}" for e in touching)
            # A self-entry on a clearing account is invalid (reported with the
            # entries) and never counts as a side.
            sided = [e for e in touching if e.debit != e.credit]
            in_side = any(e.credit == acct for e in sided)
            out_side = any(e.debit == acct for e in sided)
            if not (in_side and out_side):
                failures.append(("clearing-one-sided", f"{stage}/{acct}", None))
                rep.lines.append(f"  FAIL exchange {acct}: entries on one side only ({sides})")
            I = _clearing_inventory(acct, debts, len(time))
            tracked.append(acct)
            inv[acct] = I
            debt = "declared carried debt" if I.summands else "no carried debt"
            ref, ref_name = None, "inventory magnitude"
            if not I.summands:
                # A zero-inventory clearing account has no inventory of its own to
                # set the ceiling against: use the summed inventory magnitudes of
                # the tracked site accounts on the other side of its entries. If
                # every such account is not tracked or external, fall back to the
                # entries' summed |net|: CANNOT CERTIFY only when the bound exceeds
                # the ceiling times what the sites declared they moved.
                sites = {e.debit if e.credit == acct else e.credit for e in sided}
                sites = [s for s in sites if s in inv and s not in not_tracked and not _is_exchange(s)]
                if sites:
                    ref = sum(inv[s].gross[:-1] + inv[s].gross[1:] for s in sites)
                    ref_name = f"site inventory magnitude ({', '.join(sorted(sites))})"
                else:
                    ref = sum(np.abs(e.values) for e in sided) if sided else np.zeros(len(groups))
                    ref_name = "entries' summed |net| (no tracked site account)"
            n_fail, over, worst = certify(acct, f"exchange {acct} ({debt}; {sides})", I, touching,
                                          ref, ref_name)
            if n_fail:
                failures.append(("exchange", f"{stage}/{acct}", worst))
            elif in_side and out_side and not over:
                rep.closed.append(f"{stage}/{acct}")

        # Stage total. The stage identity is the sum of the account identities
        # over the tracked set (clearing accounts included); an entry with BOTH
        # accounts in the set enters that sum once with + and once with -, so it
        # is left out here rather than added and subtracted: internal transfers
        # cancel exactly, with no rounding, and only boundary-crossing entries
        # are summed. With every account tracked this restates the closures; it
        # adds information only where an account is not tracked.
        if tracked:
            tset = set(tracked)
            touching = [e for e in stage_entries if e.debit in tset or e.credit in tset]
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
                    gross += e.gross[k]
                resid.append(change - ext)
                count = (_closure_count(steps[k], len(touching), summands, n_cells, n_sub[k], updates)
                         + 2 * len(tracked))
                bounds.append(_bound(margin, count, gross))
            resid, bounds = np.array(resid), np.array(bounds)
            rep.bounds[f"stage/{stage}"] = bounds
            worst, n_fail, ratios = _verdict(resid, bounds)
            if n_fail:
                failures.append(("stage", stage, worst))
            rep.lines.append(_fmt(n_fail == 0, f"stage total {stage} over {len(tracked)} accounts",
                                  worst, resid, bounds, ratios, n_fail))

    # Census.
    census_fail = []
    for term, status in sorted(census.items()):
        if status not in CENSUS_STATUSES:
            census_fail.append(("census-status", term, None))
            rep.lines.append(f"  FAIL census {term}: status {status!r} is not one of {CENSUS_STATUSES}")
        elif status == "entered" and term not in nonzero:
            census_fail.append(("census-entered-empty", term, None))
            rep.lines.append(f"  FAIL census {term}: status entered but no entry")
        elif status == "entered" and not nonzero[term]:
            rep.entered_all_zero.append(term)
        elif status == "zero" and nonzero.get(term):
            census_fail.append(("census-zero-has-entry", term, None))
            rep.lines.append(f"  FAIL census {term}: status zero but an entry holds a nonzero value")
        elif status == "not_tracked" and nonzero.get(term):
            census_fail.append(("census-not-tracked-has-entry", term, None))
            rep.lines.append(f"  FAIL census {term}: status not_tracked but an entry holds a nonzero value")
        elif status == "not_tracked":
            rep.lines.append(f"  NOT TRACKED census {term}")
    for term in sorted(nonzero):
        if term not in census:
            census_fail.append(("census-missing", term, None))
            rep.lines.append(f"  FAIL census {term}: has entries but no census status")
    failures.extend(census_fail)
    if rep.entered_all_zero:
        rep.lines.append(f"  entered terms whose entries are all zero: {', '.join(rep.entered_all_zero)}")
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


def run(path, stages=None, margin=MARGIN, ceiling=CEILING, out=print):
    try:
        rep = check(path, stages, margin, ceiling)
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
# credit account's gain it; the receipt entry is the sum of those increments
# and its gross the sum of their magnitudes, accumulated in the same order.
# So closure holds to roundoff by construction, and each break is planted at a
# known account, interval and kind.

SEED = 7340117
N_CELLS = 24
N_SAVES = 7
ENGAGE_SAVE = 3          # kinetic neutrals engage at this save
ACTIVE = np.arange(1, N_CELLS)   # cell 0 is a neutral-only plenum
CX = "exchange:cx_friction"
_BOUND_RTOL = 1e-12      # report bound vs the independent formula: the two sum
                         # at most ~40 non-negative gross terms in different
                         # orders, so they differ by <= 40 * 2**-53 ~ 4.4e-15
                         # relative; 1e-12 leaves two orders of headroom.

# (term, quantity, debit, credit, cells, log10 lo, log10 hi); increments are
# per cell per step. A state account (circuit, cathode_surface,
# neutral_energy, neutral_momentum) receives the sum over the cells.
_TERMS = [
    ("ionization", "particles", "neutral_particles", "plasma_particles", ACTIVE, 12, 14),
    ("recombination", "particles", "plasma_particles", "neutral_particles", ACTIVE, 10, 11),
    ("anode_collection", "particles", "plasma_particles", "anode", np.arange(2, 5), 11, 12),
    ("end_wall_loss", "particles", "plasma_particles", "end_wall", np.array([N_CELLS - 1]), 9, 10),
    ("gas_puff", "particles", "puff", "neutral_particles", np.arange(3, 9), 12, 13),
    ("pump", "particles", "neutral_particles", "pump", np.arange(N_CELLS - 3, N_CELLS), 11, 12),
    ("floor_density", "particles", "numerical_floors", "plasma_particles", np.array([5]), 8, 9),
    ("ei_exchange", "energy_e", "electron_energy", "ion_energy", ACTIVE, 1, 3),
    ("inelastic", "energy_e", "electron_energy", "radiation", ACTIVE, 1, 2),
    ("ionization_cost", "energy_e", "electron_energy", "ionization_potential", ACTIVE, 0, 2),
    ("ohmic", "energy_e", "circuit", "electron_energy", ACTIVE, 1, 2),
    ("bank_supply", "energy_e", "bank", "circuit", np.array([0]), 3, 3.5),
    ("cathode_sheath_heat", "energy_e", "electron_energy", "cathode_surface", np.array([1]), 2, 3),
    ("rusanov_dissipation", "energy_k", "plasma_kinetic_energy", "ion_energy", ACTIVE, -2, 0),
    ("kinetic_drive", "energy_k", "ion_energy", "plasma_kinetic_energy", ACTIVE, -1, 0),
    ("cx_heating", "energy_i", "ion_energy", "neutral_energy", ACTIVE, 0, 1),
    ("ion_end_wall", "energy_i", "ion_energy", "end_wall", np.array([N_CELLS - 1]), 1, 2),
    ("neutral_wall_loss", "energy_n", "neutral_energy", "end_wall", np.array([0]), 1, 2),
    ("floor_energy", "energy_e", "numerical_floors", "electron_energy", np.array([5]), -4, -3),
    # The cx friction exchange: the plasma site removes momentum into the
    # clearing account, the neutral site adds its own computation of it.
    ("cx_friction_plasma", "momentum", "plasma_momentum", CX, ACTIVE, -4, -3),
    ("cx_friction_neutral", "momentum", CX, "neutral_momentum", ACTIVE, -4, -3),
    ("pressure_end_wall", "momentum", "plasma_momentum", "end_wall", np.array([N_CELLS - 1]), -3, -2),
    ("neutral_wall_drag", "momentum", "neutral_momentum", "end_wall", np.array([0]), -4, -3),
    ("source_push", "momentum", "cathode", "plasma_momentum", np.array([1]), -3, -2),
    # An entered term that books zero in every interval (named in the report).
    ("jet_rebirth", "particles", "end_wall", "neutral_particles", np.array([N_CELLS - 1]), None, None),
]
# The planted unbooked term of the "missing term" scenario.
_MISSING = ("wall_recycle", "particles", "end_wall", "neutral_particles", np.array([N_CELLS - 1]), 13, 13.5)
_MISSING_INTERVAL = 1
# Neutral zone each particle term acts on (A2's column/annulus split).
_NEUTRAL_ZONE = {"ionization": "col", "recombination": "col", "gas_puff": "ann", "pump": "ann",
                 "wall_recycle": "ann", "jet_rebirth": "ann"}
_CARRY = 0.3   # carried-debt scenarios: the neutral site applies 70% of its leg, the rest is carried


def _write_entry(eg, term, quantity, debit, credit, values, gross, sign="one-signed", units=None):
    tg = eg.require_group(term)
    ds = tg.create_dataset(quantity, data=values)
    ds.attrs["debit"], ds.attrs["credit"] = debit, credit
    ds.attrs["site"] = "synthetic"
    ds.attrs["units"] = QUANTITY_UNITS[quantity] if units is None else units
    if sign is not None:
        ds.attrs["sign"] = sign
    if gross is not None:
        tg.create_dataset(quantity + "_gross", data=gross)


def _write_census(rc, census, no_status=None):
    cg = rc.create_group("census")
    for term, status in census.items():
        c = cg.create_group(term)
        if no_status != term:
            c.attrs["status"] = status
        c.attrs["reason"] = "synthetic"


def _build(path, variant=None, seed=SEED):
    """Write one synthetic result file; returns the independently derived bound
    of the plasma_particles closure in interval 2 (or None)."""
    v = dict(variant or {})
    rng = np.random.default_rng(seed)
    U = int(v.get("updates_per_step", 1))
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
    p_sign = rng.choice([-1.0, 1.0], N_CELLS)
    cell = {
        "plasma_particles": np.where(act, 10 ** rng.uniform(16, 18, N_CELLS), 0.0),
        "neutral_col": 10 ** rng.uniform(16, 18, N_CELLS),
        "neutral_ann": 10 ** rng.uniform(16.5, 18.5, N_CELLS),
        "electron_energy": np.where(act, 10 ** rng.uniform(4, 7, N_CELLS), 0.0),
        "ion_energy": np.where(act, 10 ** rng.uniform(3, 6, N_CELLS), 0.0),
        "plasma_kinetic_energy": np.where(act, 10 ** rng.uniform(2, 5, N_CELLS), 0.0),
        "plasma_momentum": np.where(act, p_sign * 10 ** rng.uniform(-1, 2, N_CELLS), 0.0),
    }
    scalar = {"circuit": 1.3e6, "cathode_surface": 2.1e9, "neutral_energy": 4.7e7, "neutral_momentum": -3.1e1}
    debt = 0.0
    carry = _CARRY if v.get("carried_debt") else 0.0

    stages_present = v.get("stages", list(STAGES))
    terms = [t for t in _TERMS if QUANTITY_STAGE[t[1]] in stages_present]
    if v.get("drop_account"):
        # An account the run neither books nor stores: no entry touches it.
        terms = [t for t in terms if v["drop_account"] not in (t[2], t[3])]
    if v.get("missing_term"):
        terms = terms + [_MISSING]

    def apply(acct, cells, incr, sign, term):
        if acct in scalar:
            scalar[acct] += sign * incr.sum()
        elif acct == "neutral_particles":
            cell["neutral_" + _NEUTRAL_ZONE[term]][cells] += sign * incr
        elif acct in cell:
            cell[acct][cells] += sign * incr
        # external and clearing accounts hold no cell inventory

    snap = {k: [a.copy()] for k, a in cell.items()}
    ssnap = {k: [s] for k, s in scalar.items()}
    debt_snap = [debt]
    amount = {t[0]: [] for t in terms}     # per step, summed over cells
    gross = {t[0]: [] for t in terms}
    for k in range(N_SAVES - 1):
        for _ in range(steps[k]):
            for term, quantity, debit, credit, cells, lo, hi in terms:
                if term == "cx_friction_neutral":
                    continue  # the neutral site's leg of the cx_friction draw, below
                if lo is None:
                    incr = np.zeros(len(cells))
                else:
                    incr = 10 ** rng.uniform(lo, hi, len(cells))
                if term == _MISSING[0] and k != _MISSING_INTERVAL:
                    incr = np.zeros(len(cells))
                # updates_per_step state updates, each applying its share.
                parts = [incr] if U == 1 else [incr / U] * (U - 1) + [incr - (U - 1) * (incr / U)]
                a_step = 0.0
                g_step = 0.0
                for p in parts:
                    apply(debit, cells, p, -1.0, term)
                    apply(credit, cells, p, +1.0, term)
                    a_step += p.sum()
                    g_step += np.abs(p).sum()
                amount[term].append(a_step)
                gross[term].append(g_step)
                if term == "cx_friction_plasma":
                    # The neutral site computes the same exchange in its own
                    # order (sequential, last cell first) and applies what it
                    # computed, less the share it carries as debt.
                    b = sum(incr[::-1].tolist()) * (1.0 - carry)
                    scalar["neutral_momentum"] += b
                    amount["cx_friction_neutral"].append(b)
                    gross["cx_friction_neutral"].append(b)
                    debt += a_step - b
        for key, a in cell.items():
            snap[key].append(a.copy())
        for key, s in scalar.items():
            ssnap[key].append(s)
        debt_snap.append(debt)

    per_step = {term: np.array(a) for term, a in amount.items()}
    per_step_g = {term: np.array(a) for term, a in gross.items()}
    edges = save_idx
    per_save = {term: np.array([a[edges[k]:edges[k + 1]].sum() for k in range(N_SAVES - 1)])
                for term, a in per_step.items()}
    per_save_g = {term: np.array([a[edges[k]:edges[k + 1]].sum() for k in range(N_SAVES - 1)])
                  for term, a in per_step_g.items()}
    inv_p = np.array(snap["plasma_particles"])

    # The bound of the plasma_particles closure in interval 2, written here
    # independently of the checker from its stated formula:
    #   margin * count * 2**-53 * gross,
    #   count = U*S*(E+1) + 4*C + E*(U*S + N_CELLS + 1) + E + 2,
    # C the active cells (the inventory's summands), N_CELLS the per-cell sum
    # behind each entry's per-step amount, gross the inventory magnitudes at
    # both saves plus the entries' gross.
    kb = 2
    touching = [t for t in terms if "plasma_particles" in (t[2], t[3])]
    E, S, C = len(touching), int(steps[kb]), len(ACTIVE)
    g = (np.abs(inv_p[kb][ACTIVE]).sum() + np.abs(inv_p[kb + 1][ACTIVE]).sum()
         + sum(per_save_g[t[0]][kb] for t in touching))
    count = U * S * (E + 1) + 4 * C + E * (U * S + N_CELLS + 1) + E + 2
    bound_pp = 4.0 * count * 2.0 ** -53 * g

    # Planted breaks applied to the state.
    if "closure_break" in v:
        acct, kbreak, factor = v["closure_break"]
        assert acct == "plasma_particles" and kbreak == kb
        for j in range(kb + 1, N_SAVES):
            snap["plasma_particles"][j][5] += factor * bound_pp
    if "leg_disagree" in v:
        # The neutral site computes, books and applies D more than the plasma
        # site removed in one interval: each site account closes on its own
        # entry, the clearing account does not.
        kl = v["leg_disagree"]
        D = 1e-6 * sum(np.abs(a).sum() for a in snap["plasma_momentum"])   # far above roundoff
        per_save["cx_friction_neutral"][kl] += D
        per_save_g["cx_friction_neutral"][kl] += D
        for j in range(kl + 1, N_SAVES):
            ssnap["neutral_momentum"][j] += D
    if "wrong_debt" in v:
        kd = v["wrong_debt"]
        offset = 1e-3 * abs(debt_snap[kd + 1])
        for j in range(kd + 1, N_SAVES):
            debt_snap[j] += offset

    # Saved fields (A1-A4).
    def field(key, vol, mask=None):
        a = np.array(snap[key]) / vol
        if mask is not None:
            a[:, ~mask] = 0.0
        return a

    n = field("plasma_particles", vp, act)
    n[:, 0] = 10 ** rng.uniform(5, 9, N_SAVES)   # plenum values the plasma_active mask must drop
    if v.get("single_zone"):
        nn = (np.array(snap["neutral_col"]) + np.array(snap["neutral_ann"])) / vn
        nn_a = None
    else:
        nn = field("neutral_col", vp)
        nn_a = field("neutral_ann", va)
    # After engagement the saved neutral fields are republished moments that
    # need not hold the inventory; the receipt's state does.
    nn[ENGAGE_SAVE:] *= 1.0 + 1e-6
    if nn_a is not None:
        nn_a[ENGAGE_SAVE:] *= 1.0 - 1e-6
    kin = np.array([snap["neutral_col"][j].sum() + snap["neutral_ann"][j].sum() for j in range(N_SAVES)])
    engaged = (np.arange(N_SAVES) >= ENGAGE_SAVE).astype(np.int8)
    Ee = field("electron_energy", vp, act)
    Ei = field("ion_energy", vp, act)
    mom = field("plasma_momentum", vp, act)
    kdens = field("plasma_kinetic_energy", vp, act)
    u = np.zeros_like(mom)
    u[:, act] = 2.0 * kdens[:, act] / mom[:, act]
    neutral_fields = {}
    layout = v.get("neutral_fields")
    if layout:
        # Saved neutral momentum or energy fields holding the inventory, spread
        # over the cells with weights w, in one of the solver's layouts (this
        # file is two-zone in density):
        #   mn_split   M_n on the column volume + M_n_a on the annulus (40/60);
        #   mn_chamber M_n alone, a chamber mean on the neutral volume;
        #   en_column  En, a column field on the plasma volume.
        # Any other volume than the layout's gives a different inventory
        # (sum(w*vp/vn) != 1 and so on), so the wrong volume fails the closure.
        w = rng.uniform(0.5, 1.5, N_CELLS)
        w /= w.sum()
        acct = "neutral_energy" if layout == "en_column" else "neutral_momentum"
        tot = np.array(ssnap[acct])[:, None]
        if layout == "mn_split":
            arrays = {"M_n": 0.4 * w * tot / vp, "M_n_a": 0.6 * w * tot / va}
        elif layout == "mn_chamber":
            arrays = {"M_n": w * tot / vn}
        else:
            arrays = {"En": w * tot / vp}
        for name, a in arrays.items():
            # Republished after engagement: the receipt's state holds the inventory.
            a[ENGAGE_SAVE:] *= 1.0 + 1e-6
            neutral_fields[name] = a
        # Before engagement the state is not the inventory; the fields are.
        ssnap[acct] = [s * (1.01 if j < ENGAGE_SAVE else 1.0) for j, s in enumerate(ssnap[acct])]

    with h5py.File(path, "w") as f:
        f.attrs["format"] = "sim1d-hdf5-v1"
        f.attrs["steps"] = int(steps.sum()) + (1 if v.get("steps_mismatch") else 0)
        f["time"] = time
        for name, a in (("n", n), ("nn", nn), ("nn_a", nn_a), ("momentum", mom), ("Ee", Ee),
                        ("Ei", Ei), ("u", u), *neutral_fields.items()):
            if a is not None:
                f[name] = a
        g = f.create_group("geometry")
        g["plasma_volume_cm3"] = vp
        g["neutral_volume_cm3"] = vn
        g["volume_ratio"] = vp / vn
        if not v.get("no_plasma_active"):
            g["plasma_active"] = act
        g["cell_role"] = np.array([b"plenum"] + [b"column"] * (N_CELLS - 1), dtype=object)
        g["length_cm"] = np.full(N_CELLS, 10.0)
        if v.get("no_receipt"):
            return bound_pp
        rc = f.create_group("receipt")
        rc.attrs["schema"] = SCHEMA
        cadence = v.get("cadence", "save")
        rc.attrs["cadence"] = cadence
        rc.attrs["stages_present"] = np.array(stages_present, dtype=object)
        if U != 1:
            rc.attrs["updates_per_step"] = U
        if cadence == "save":
            rc["interval_t0"], rc["interval_t1"] = time[:-1], time[1:]
            rc["interval_steps"] = steps.astype(np.int64)
            data, data_g = per_save, per_save_g
        else:
            rc["interval_t0"], rc["interval_t1"] = tb[:-1], tb[1:]
            rc["interval_steps"] = np.ones(tb.size - 1, np.int64)
            data, data_g = per_step, per_step_g
        st = rc.create_group("state")
        st["neutral_particles_kinetic"] = kin
        st["neutral_kinetic_engaged"] = engaged
        for key in scalar:
            if key != v.get("drop_account"):
                st[key] = np.array(ssnap[key])
        if v.get("carried_debt"):
            st["exchange_cx_friction"] = np.array(debt_snap)
        if v.get("orphan_state"):
            st["exchange_ghost"] = np.where(np.arange(N_SAVES) >= 2, 1e-3, 0.0)
        if v.get("not_tracked") == "circuit":
            st["circuit"][...] = st["circuit"][()] * (1.0 + rng.uniform(-1e-3, 1e-3, N_SAVES))
        eg = rc.create_group("entries")
        census = {}
        for term, quantity, debit, credit, cells, lo, hi in terms:
            if term == _MISSING[0]:
                continue  # applied to the state, absent from the receipt
            if v.get("swap") == term:
                debit, credit = credit, debit
            if v.get("unknown_account") == term:
                credit = credit + "p"   # the external side, so no inventory moves
            gvals = data_g[term]
            sign = "one-signed"
            if v.get("gross_inconsistent") == term:
                gvals = 0.5 * np.abs(data[term])
            if v.get("one_signed_gross_high") == term:
                gvals = np.abs(data[term]) * (1.0 + 1e-9)
            if v.get("inflate_gross") == term:
                # Declared signed with a gross far above the inventory: it
                # would carry a planted break under its bound.
                gvals = np.full_like(data[term], 1e5 * np.abs(inv_p).sum(axis=1).max())
                sign = "signed"
            if v.get("missing_sign") == term:
                sign = None
            _write_entry(eg, term, quantity, debit, credit, data[term],
                         None if v.get("gross_missing") == term else gvals, sign=sign,
                         units="erg" if v.get("bad_units") == term else None)
            census[term] = "entered"
        census["recycling_jet"] = "zero"
        for key in ("census_zero", "census_not_tracked_term"):
            if v.get(key):
                census[v[key]] = "zero" if key == "census_zero" else "not_tracked"
        if v.get("census_missing"):
            del census[v["census_missing"]]
        if v.get("census_entered_empty"):
            census[v["census_entered_empty"]] = "entered"
        if v.get("not_tracked"):
            census[v["not_tracked"]] = "not_tracked"
        _write_census(rc, census, no_status=v.get("census_nostatus"))
    return bound_pp


def _build_small(path, mode, factor=None):
    """Small momentum-only files whose rounding is planted exactly.

    ``cancel``/``cancel_understated``: a signed term on 2 cells, per step and
    cell a push of X then a pull of -(X - d) then a drain e, with X = 1.5*2**30
    on the grid g = 2**-22, cell momenta ~X/1024 on that grid, d on the grid and
    e = 0.375 g below it. Each push X rounds away the e the previous step left,
    so the state loses e per cell-step (3 g per interval) while the entry books
    it; the entry's gross is ~2e6 times its net.
    ``transport``/``transport_unbooked``: cell-to-cell moves of X out of cell 0
    into cell 1 and back each step, booked as a self-entry on plasma_momentum,
    with a one-signed drain e from each cell; each move rounds away the drain's
    sub-grid bits (0.75 g per step) and only the self-entry's gross covers it.
    ``exchange``/``exchange_break``: 130 cells, one step; one site sums the
    per-cell exchange contributions forward, the other in reverse, with one
    dominant cell of 1 and 129 cells of 0.99 * 2**-53; returns the independently
    derived clearing-account bound.
    """
    exchange = mode.startswith("exchange")
    n_cells = 130 if exchange else 2
    S = 1 if exchange else 4
    n_int = 1 if exchange else 2
    vp = np.ones(n_cells)
    act = np.ones(n_cells, bool)
    steps = np.full(n_int, S, np.int64)
    time = np.arange(n_int + 1) * 1e-6
    k = 30
    X = 1.5 * 2.0 ** k
    g = 2.0 ** (k - 52)
    e = 0.375 * g
    P = np.full(n_cells, np.round((X / 1024) / g) * g) if not exchange else np.full(n_cells, 10.0)
    snaps = [P.copy()]
    neutral = [5.0]
    entries = {}   # term -> [quantity, debit, credit, per-interval values, gross, sign]

    def book(term, quantity, debit, credit, sign):
        entries.setdefault(term, [quantity, debit, credit, np.zeros(n_int), np.zeros(n_int), sign])
        return entries[term]

    bound_cx = None
    if exchange:
        x = np.array([1.0] + [0.99 * 2.0 ** -53] * (n_cells - 1))
        a = 0.0
        for c in x:          # site 1: forward, and its gross beside it
            a += c
        b = 0.0
        for c in x[::-1]:    # site 2: reverse
            b += c
        P = P - x
        # Independent bound of the clearing closure (no carried debt):
        # count = S*(E+1) + 4*0 + E*(S + n_cells + 1) + E + 2, E = 2 entries.
        E = 2
        count = S * (E + 1) + E * (S + n_cells + 1) + E + 2
        bound_cx = 4.0 * count * 2.0 ** -53 * (a + b)
        D = 0.0 if factor is None else factor * bound_cx
        bs = b + D
        book("cx_plasma", "momentum", "plasma_momentum", "exchange:cx", "one-signed")[3:5] = [
            np.array([a]), np.array([a])]
        if mode == "exchange_self":
            # The second site is replaced by a self-entry on the clearing
            # account, which must not pass for the missing side.
            book("cx_self", "momentum", "exchange:cx", "exchange:cx", "one-signed")[3:5] = [
                np.array([b]), np.array([b])]
            bs = 0.0
        else:
            book("cx_neutral", "momentum", "exchange:cx", "neutral_momentum", "one-signed")[3:5] = [
                np.array([bs]), np.array([bs])]
        snaps.append(P.copy())
        neutral.append(5.0 + bs)
    else:
        d = np.round(1e-6 * X / g) * g
        for kk in range(n_int):
            for _ in range(S):
                if mode.startswith("cancel"):
                    ent = book("cancel_push", "momentum", "plasma_momentum", "end_wall", "signed")
                    net = 0.0
                    gr = 0.0
                    for c in range(n_cells):
                        cn = 0.0
                        cg = 0.0
                        for contrib in (X, -(X - d), e):
                            P[c] -= contrib
                            cn += contrib
                            cg += abs(contrib)
                        net += cn
                        gr += cg
                    ent[3][kk] += net
                    ent[4][kk] += gr
                else:
                    if mode == "transport":
                        adv = book("advection", "momentum", "plasma_momentum", "plasma_momentum",
                                   "one-signed")
                    drain = book("fine_drain", "momentum", "plasma_momentum", "end_wall", "one-signed")
                    P[0] -= X
                    P[1] += X
                    P[1] -= X
                    P[0] += X
                    if mode == "transport":
                        adv[3][kk] += 2 * X
                        adv[4][kk] += 2 * X
                    for c in range(n_cells):
                        P[c] -= e
                    drain[3][kk] += 2 * e
                    drain[4][kk] += 2 * e
            snaps.append(P.copy())
            neutral.append(5.0)
        if mode == "cancel_understated":
            entries["cancel_push"][4] = np.abs(entries["cancel_push"][3])

    with h5py.File(path, "w") as f:
        f.attrs["format"] = "sim1d-hdf5-v1"
        f.attrs["steps"] = int(steps.sum())
        f["time"] = time
        f["momentum"] = np.array(snaps) / vp
        geo = f.create_group("geometry")
        geo["plasma_volume_cm3"] = vp
        geo["neutral_volume_cm3"] = 2 * vp
        geo["plasma_active"] = act
        geo["cell_role"] = np.array([b"column"] * n_cells, dtype=object)
        geo["length_cm"] = np.ones(n_cells)
        rc = f.create_group("receipt")
        rc.attrs["schema"] = SCHEMA
        rc.attrs["cadence"] = "save"
        rc.attrs["stages_present"] = np.array(["momentum"], dtype=object)
        rc["interval_t0"], rc["interval_t1"] = time[:-1], time[1:]
        rc["interval_steps"] = steps
        rc.create_group("state")["neutral_momentum"] = np.array(neutral)
        eg = rc.create_group("entries")
        for term, (quantity, debit, credit, vals, gross, sign) in entries.items():
            _write_entry(eg, term, quantity, debit, credit, vals, gross, sign=sign)
        _write_census(rc, {term: "entered" for term in entries})
    return bound_cx


def self_test(verbose=False):
    # (name, builder call, requested stages, expected exit, expected failures,
    #  extra expectations). A failure is (kind, name, interval); interval None
    #  means "any interval" (a break planted in every interval).
    ANY = None
    MX = f"momentum/{CX}"
    M = lambda v: (lambda p: _build(p, v))               # noqa: E731
    SM = lambda mode, fac=None: (lambda p: _build_small(p, mode, fac))   # noqa: E731
    PP2 = ("particles/plasma_particles", 2)
    scenarios = [
        ("clean", M({}), None, 0, [],
         {"entered_zero": ["jet_rebirth"], "bound_check": PP2}),
        ("clean, step cadence", M({"cadence": "step"}), None, 0, [], {}),
        ("clean, single-zone neutrals", M({"single_zone": True}), None, 0, [], {}),
        ("clean, two-zone M_n with M_n_a (column + annulus) before engagement",
         M({"neutral_fields": "mn_split"}), None, 0, [], {}),
        ("clean, two-zone M_n with no M_n_a (chamber mean) before engagement",
         M({"neutral_fields": "mn_chamber"}), None, 0, [], {}),
        ("clean, two-zone En (column field) before engagement",
         M({"neutral_fields": "en_column"}), None, 0, [], {}),
        ("clean, updates_per_step = 3", M({"updates_per_step": 3}), None, 0, [],
         {"bound_check": PP2}),
        ("clean, exchange with a carried debt", M({"carried_debt": True}), None, 0, [], {}),
        ("clean, signed term cancelling to ~1e-6 of its gross", SM("cancel"), None, 0, [], {}),
        ("clean, internal transport booked as a self-entry", SM("transport"), None, 0, [], {}),
        ("clean, two sites summing an exchange in opposite orders (130 cells)", SM("exchange"),
         None, 0, [], {"bound_check": (MX.replace(CX, "exchange:cx"), 0)}),
        ("exchange sites differ by 0.1 x the clearing bound", SM("exchange_break", 0.1), None, 0, [],
         {"bound_check": (MX.replace(CX, "exchange:cx"), 0)}),
        # The stage total does not see this break: its gross carries the plasma
        # momentum inventory (130 cells of 10, ~1.3e3 times the exchange's
        # gross of ~2) and its count the inventory terms (1317 against 271 for
        # the clearing account, gross 2613 against 2), so its bound is ~6.3e3
        # times the clearing bound and 100 x the clearing bound sits under it.
        # The clearing account's own bound is what decides two-site agreement.
        ("exchange sites differ by 100 x the clearing bound", SM("exchange_break", 100.0), None, 1,
         [("exchange", "momentum/exchange:cx", 0)], {}),
        ("closure break x100 (plasma_particles, interval 2)",
         M({"closure_break": ("plasma_particles", 2, 100.0)}), None, 1,
         [("closure", "particles/plasma_particles", 2), ("stage", "particles", 2)], {}),
        ("closure break x0.1 (plasma_particles, interval 2)",
         M({"closure_break": ("plasma_particles", 2, 0.1)}), None, 0, [], {}),
        ("closure break x100 under an inflated signed gross (floor_density)",
         M({"closure_break": ("plasma_particles", 2, 100.0), "inflate_gross": "floor_density"}), None, 1,
         [("bound-ceiling", "particles/plasma_particles", ANY)], {}),
        ("debit and credit swapped (ionization)", M({"swap": "ionization"}), None, 1,
         [("closure", "particles/plasma_particles", ANY),
          ("closure", "particles/neutral_particles", ANY)], {}),
        ("term applied but not booked (wall_recycle, interval 1)", M({"missing_term": True}), None, 1,
         [("closure", "particles/neutral_particles", _MISSING_INTERVAL),
          ("stage", "particles", _MISSING_INTERVAL)], {}),
        ("internal transport with no self-entry", SM("transport_unbooked"), None, 1,
         [("closure", "momentum/plasma_momentum", ANY), ("stage", "momentum", ANY)], {}),
        ("exchange sites disagree (cx_friction, interval 4)", M({"leg_disagree": 4}), None, 1,
         [("exchange", MX, 4), ("stage", "momentum", 4)], {}),
        ("carried debt series wrong from save 4 (cx_friction)", M({"carried_debt": True, "wrong_debt": 3}),
         None, 1, [("exchange", MX, 3), ("stage", "momentum", 3)], {}),
        ("exchange booked on one side only (cx_friction_neutral debit and credit swapped)",
         M({"swap": "cx_friction_neutral"}), None, 1,
         [("clearing-one-sided", MX, None), ("exchange", MX, ANY),
          ("closure", "momentum/neutral_momentum", ANY)], {}),
        ("orphan clearing-account state (state/exchange_ghost)", M({"orphan_state": True}), None, 1,
         [("exchange-orphan-state", "exchange:ghost", None)], {}),
        # The second site replaced by a self-entry on the clearing account: the
        # self-entry is invalid, the account one-sided, and the plasma site's
        # removal of ~1 is unmatched in both the clearing closure and the
        # stage total (whose bound, ~1.5e-9, is far below 1).
        ("self-entry on a clearing account standing in for its second site", SM("exchange_self"),
         None, 1, [("self-entry-invalid", "cx_self/momentum", None),
                   ("clearing-one-sided", "momentum/exchange:cx", None),
                   ("exchange", "momentum/exchange:cx", 0), ("stage", "momentum", 0)], {}),
        # An inflated signed gross on the neutral site of a clearing account
        # with no carried debt: the clearing account is held against its site
        # accounts' inventories and refused, as is neutral_momentum itself.
        ("inflated signed gross on a clearing account (cx_friction_neutral)",
         M({"inflate_gross": "cx_friction_neutral"}), None, 1,
         [("bound-ceiling", MX, ANY), ("bound-ceiling", "momentum/neutral_momentum", ANY)], {}),
        ("cathode_surface with no state series and no census declaration",
         M({"drop_account": "cathode_surface"}), None, 1,
         [("no-inventory", "energy/cathode_surface", None)], {}),
        ("cathode_surface with no state series, declared not_tracked",
         M({"drop_account": "cathode_surface", "not_tracked": "cathode_surface"}), None, 0, [],
         {"not_tracked": ["cathode_surface"]}),
        ("signed term with gross understated to |entry|", SM("cancel_understated"), None, 1,
         [("closure", "momentum/plasma_momentum", ANY), ("stage", "momentum", ANY)], {}),
        ("entry with no gross (pump)", M({"gross_missing": "pump"}), None, 1,
         [("gross-missing", "pump/particles", None)], {}),
        ("gross below |entry| (pump)", M({"gross_inconsistent": "pump"}), None, 1,
         [("gross-inconsistent", "pump/particles", None)], {}),
        ("one-signed gross above |entry| (pump)", M({"one_signed_gross_high": "pump"}), None, 1,
         [("gross-inconsistent", "pump/particles", None)], {}),
        ("entry with no sign (pump)", M({"missing_sign": "pump"}), None, 1,
         [("sign", "pump/particles", None)], {}),
        ("units wrong (pump in erg)", M({"bad_units": "pump"}), None, 1,
         [("units", "pump/particles", None)], {}),
        ("steps attr disagrees with interval_steps", M({"steps_mismatch": True}), None, 1,
         [("steps-mismatch", "steps", None)], {}),
        ("census term with no status (inelastic)", M({"census_nostatus": "inelastic"}), None, 1,
         [("census-status", "inelastic", None)], {}),
        ("entered term with no entry (beam_ionization)", M({"census_entered_empty": "beam_ionization"}),
         None, 1, [("census-entered-empty", "beam_ionization", None)], {}),
        ("zero term with a nonzero entry (floor_density)", M({"census_zero": "floor_density"}), None, 1,
         [("census-zero-has-entry", "floor_density", None)], {}),
        ("not_tracked term with a nonzero entry (end_wall_loss)",
         M({"census_not_tracked_term": "end_wall_loss"}), None, 1,
         [("census-not-tracked-has-entry", "end_wall_loss", None)], {}),
        ("entry whose term is not in the census (gas_puff)", M({"census_missing": "gas_puff"}), None, 1,
         [("census-missing", "gas_puff", None)], {}),
        ("unknown account name (pump -> pumpp)", M({"unknown_account": "pump"}), None, 1,
         [("account", "pump/particles", None)], {}),
        ("requested stage absent (momentum)", M({"stages": ["particles"]}), ["momentum"], 2, [], {}),
        ("no receipt", M({"no_receipt": True}), None, 2, [], {}),
        ("no geometry/plasma_active", M({"no_plasma_active": True}), None, 2, [], {}),
        ("not_tracked account (circuit, its state corrupted)", M({"not_tracked": "circuit"}), None, 0, [],
         {"not_tracked": ["circuit"], "closed_excludes": "energy/circuit",
          "closed_includes": "energy/electron_energy"}),
    ]
    bad = 0
    with tempfile.TemporaryDirectory(prefix="ledger_selftest_") as tmp:
        for i, (name, build, stages, want_exit, want_fail, extra) in enumerate(scenarios):
            path = os.path.join(tmp, f"scenario_{i:02d}.h5")
            indep = build(path)
            lines = []
            code, rep = run(path, stages, MARGIN, CEILING, out=lines.append)
            got = sorted(rep.failures, key=str) if rep else []
            ok = code == want_exit
            notes = []
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
                if "entered_zero" in extra:
                    notes.append(f"all-zero entered={rep.entered_all_zero}")
                    ok &= rep.entered_all_zero == extra["entered_zero"]
                if "bound_check" in extra:
                    key, kk = extra["bound_check"]
                    got_b = rep.bounds.get(key, [np.nan] * (kk + 1))[kk]
                    rel = abs(got_b - indep) / indep
                    notes.append(f"{key} bound[{kk}] {got_b:.6e} vs independent {indep:.6e} (rel {rel:.1e})")
                    ok &= bool(rel <= _BOUND_RTOL)
            bad += not ok
            want_txt = f"exit {want_exit} {[(k, n, '*' if iv is None else iv) for k, n, iv in want_fail]}"
            got_txt = f"exit {code} {got}"
            if extra.get("not_tracked") and rep is not None:
                got_txt += f" not_tracked={rep.not_tracked} closed={len(rep.closed)}"
                want_txt += f" not_tracked={extra['not_tracked']}"
            if notes:
                got_txt += "; " + "; ".join(notes)
            print(f"{'OK  ' if ok else 'MISMATCH'} {name}: expected {want_txt}; got {got_txt}")
            if verbose or not ok:
                for line in lines:
                    print("      " + line)
    print(f"SELF-TEST: PASS ({len(scenarios)} scenarios)" if bad == 0
          else f"SELF-TEST: FAIL ({bad} of {len(scenarios)} scenarios differ)")
    return 0 if bad == 0 else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("result", nargs="?", help="a sim1d-hdf5-v1 result file")
    ap.add_argument("--stage", action="append", choices=STAGES,
                    help="stage to check (repeatable); default every stage in stages_present")
    ap.add_argument("--margin", type=float, default=MARGIN,
                    help=f"safety factor on the roundoff count (default {MARGIN:g})")
    ap.add_argument("--ceiling", type=float, default=CEILING,
                    help=f"largest bound / inventory magnitude that still certifies (default {CEILING:g})")
    ap.add_argument("--self-test", action="store_true",
                    help="build synthetic files with planted closures and breaks and check each")
    ap.add_argument("--verbose", action="store_true", help="self-test: print every report")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test(args.verbose)
    if not args.result:
        ap.error("a result file or --self-test is required")
    code, _ = run(args.result, args.stage, args.margin, args.ceiling)
    return code


if __name__ == "__main__":
    sys.exit(main())
