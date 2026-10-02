"""The conservation receipt: what each term moved, between which accounts.

A result carries, beside its saved state, a RECEIPT for the particle stage:
for every save interval and every named term that moved particles, the
volume-integrated, time-integrated amount the term moved from a ``debit``
account to a ``credit`` account, with the accumulated magnitude of the
contributions that made it (its ``_gross``). The schema is ``receipt-v1``;
``scripts/gates/ledger_check.py`` documents it and is the reader that closes
it against the saved state, so this module is the writer's side only.

WHAT IS BOOKED, AND WHERE. Each code site books only what it itself applied:

* the explicit stages (:meth:`ParticleReceipt.book_rhs`), once per ``rhs``
  call inside a step attempt, at the stage's weight in the accepted state
  (``dt / 2`` for both SSPRK2 stages), from the very rows the stage sums;
* the floors (:meth:`ParticleReceipt.book_floor`), at the weight the floor
  ledger books the same call at;
* the backward-Euler neutral-only step (:meth:`ParticleReceipt.book_neutral_step`);
* the kinetic neutral engine's tick (:meth:`ParticleReceipt.book_tick`), from
  the per-tick particle ledger the engine returns.

Attempt tallies are armed per attempt and committed only on acceptance, so a
rejected attempt books nothing; engine ticks run only on accepted steps and
book straight into the interval. Each save closes the interval.

Accounts are the checker's: ``plasma_particles`` (``n`` on the plasma volume
over the plasma-active cells), ``neutral_particles`` (the fluid ``nn`` on the
column volume plus ``nn_a`` on the annulus volume until the kinetic neutrals
engage, the engine's own inventory after), the clearing accounts
``exchange:<name>`` through which two sites book the two sides of one
exchange, and the external accounts ``cathode``, ``anode``, ``end_wall``,
``pump``, ``puff`` and ``numerical_floors``.

Nothing here writes to a solver state, a cache, or a ledger the solver reads:
every input is a row or a count the solver has already computed.
"""

from __future__ import annotations

import numpy as np

SCHEMA = "receipt-v1"
STAGE = "particles"
UNITS = "particles"

PLASMA = "plasma_particles"
NEUTRAL = "neutral_particles"
FLOORS = "numerical_floors"
PUFF = "puff"
PUMP = "pump"

X_IONIZATION = "exchange:ionization"
X_RECOMBINATION = "exchange:recombination"
X_ANODE = "exchange:anode_return"
X_CATHODE_FACE = "exchange:cathode_face_recycle"
X_END_WALL = "exchange:end_wall_recycle"

#: Clearing account of each counted kinetic source channel (the keys the
#: solver's ``_dvm_source_booked`` tally carries).
CHANNEL_CLEARING = {
    "cathode_face": X_CATHODE_FACE,
    "end_wall_face": X_END_WALL,
    "recombination": X_RECOMBINATION,
    "anode": X_ANODE,
}

# How many times a cell's particle state is updated per accepted step: the
# ``updates_per_step`` the checker's operation count multiplies by.
#   * Explicit path and operator split: the particle rows are integrated by
#     exactly one SSPRK2 step at the full dt, which updates each cell twice
#     (y1 = y0 + dt*k0, then y2 = 0.5*y0 + 0.5*(y1 + dt*k1)), each update
#     adding that stage's summed term rows. The implicit heat substeps copy
#     the particle rows unchanged, and the floors after them find nothing
#     below the floor that the stage floors did not already lift.
#   * Neutral-only step: one backward-Euler solve, one update.
#   * Kinetic engine: at most one tick per accepted step (a tick fires on an
#     accepted step whose end time reaches the neutral clock), each one
#     update of the engine's inventory.
# The largest of these is 2.
UPDATES_PER_STEP = 2

# rhs term -> how its particle rows are booked. ``self`` books the row as
# transport inside one account; ``exchange`` books the plasma ``n`` row
# against a clearing account (sign: +1 the plasma gains, -1 it loses) and,
# where the fluid neutral rows are live, the ``nn``/``nn_a`` rows against the
# same clearing account from the other side.
_RHS_SITES = {
    "plasma_advective_flux": "physics/flux.py:plasma_flux_rhs_terms",
    "characteristic_boundary": "physics/sources.py:characteristic_boundary_rhs",
    "ionization_birth": "physics/reactions.py:reaction_rhs_terms",
    "beam_ionization_birth": "physics/cathode.py:beam_ionization_rhs_terms",
    "recombination_rad_loss": "physics/reactions.py:reaction_rhs_terms",
    "anode_collection": "physics/sources.py:anode_collection_rhs",
    "neutral_exchange": "physics/neutrals.py (axial Knudsen exchange)",
    "neutral_zone_exchange": "physics/neutrals.py (column/annulus exchange)",
    "neutral_sources": "physics/neutrals.py:neutral_source_sink_rhs",
}
_EXCHANGE_TERMS = {
    # term: (clearing account, plasma sign)
    "ionization_birth": (X_IONIZATION, +1.0),
    "beam_ionization_birth": (X_IONIZATION, +1.0),
    "recombination_rad_loss": (X_RECOMBINATION, -1.0),
    "anode_collection": (X_ANODE, -1.0),
}
#: rhs terms the booking table covers.
BOOKED_RHS_TERMS = frozenset(_RHS_SITES)

_ENGINE_SITE = "physics/kinetic_dvm.py:TransientDVM.update (tick ledger)"
_SOLVER_SITE = "solver.py"


class ParticleReceipt:
    """Accumulate one run's particle receipt, save interval by save interval.

    ``geometry`` supplies the plasma and neutral volumes and the
    plasma-active mask, ``zone_volumes`` the ``(V_col, V_ann)`` the two-zone
    neutral rows are written against, ``recycle_cells`` the live cells of
    each absorbing face by role, ``kinetic`` whether a kinetic neutral engine
    will own the neutral inventory once it engages. ``declared`` maps a
    term this receipt does not book on this configuration to the reason, and
    ``accounts_not_tracked`` an account whose closure that leaves wrong to
    the reason; both are written into the census.
    """

    def __init__(
        self,
        geometry,
        zone_volumes,
        recycle_cells,
        kinetic=False,
        declared=None,
        accounts_not_tracked=None,
    ):
        self._Vp = np.asarray(geometry.plasma_volume_cm3, dtype=float)
        self._Vc = np.asarray(zone_volumes[0], dtype=float)
        self._Va = np.asarray(zone_volumes[1], dtype=float)
        self._active = np.asarray(geometry.plasma_active, dtype=bool)
        cells = self._Vp.size
        self._cells = cells
        self._role_masks = {}
        for role, members in dict(recycle_cells).items():
            mask = np.zeros(cells, dtype=bool)
            mask[list(members)] = True
            self._role_masks[str(role)] = mask
        known = self._role_masks.get("cathode", np.zeros(cells, bool)) | (
            self._role_masks.get("end_wall", np.zeros(cells, bool))
        )
        self._boundary_other = ~known
        self.kinetic = bool(kinetic)
        self.declared = dict(declared or {})
        self.accounts_not_tracked = dict(accounts_not_tracked or {})
        self._specs = {}
        self._interval = {}
        self._interval_steps = 0
        self._attempt = None
        self._weight = 0.0
        # Per save frame: the closed interval's values, and the state rows.
        self.frames = []
        self.rhs_keys = set()
        self.unbooked = {}
        self.handover = None
        self.leading_steps_not_covered = 0
        self._resumed = False

    # ------------------------------------------------------------ booking

    def _spec(self, term, debit, credit, sign, site):
        spec = (debit, credit, sign, site)
        known = self._specs.get(term)
        if known is None:
            self._specs[term] = spec
        elif known != spec:
            raise ValueError(
                f"receipt entry {term!r} booked as {spec} after {known}"
            )

    def _add(self, target, term, debit, credit, sign, site, x, weight):
        """Book ``weight * sum(x)`` with gross ``|weight| * sum(|x|)``."""
        self._spec(term, debit, credit, sign, site)
        x = np.asarray(x, dtype=float)
        value = float(x.sum())
        gross = float(np.abs(x).sum())
        slot = target.setdefault(term, [0.0, 0.0])
        slot[0] += weight * value
        slot[1] += abs(weight) * gross

    def _add_scalar(self, term, debit, credit, sign, site, value, gross):
        self._spec(term, debit, credit, sign, site)
        slot = self._interval.setdefault(term, [0.0, 0.0])
        slot[0] += float(value)
        slot[1] += float(gross)

    def arm_attempt(self, weight):
        """Open one step attempt's tally; ``weight`` is each rhs call's."""
        self._attempt = {}
        self._weight = float(weight)

    def drop_attempt(self):
        """Close the attempt's tally and return it (to commit or discard)."""
        tally = self._attempt
        self._attempt = None
        self._weight = 0.0
        return tally

    def commit(self, tally):
        """Fold an ACCEPTED attempt's tally into the interval: one step."""
        for term, (value, gross) in (tally or {}).items():
            slot = self._interval.setdefault(term, [0.0, 0.0])
            slot[0] += value
            slot[1] += gross
        self._interval_steps += 1

    def book_rhs(self, terms, applied, engaged, source_parts=None):
        """Book one rhs call's applied terms at the attempt's stage weight.

        ``source_parts`` holds the puff and pump rows ``neutral_sources``
        is the sum of, as that term's builder returned them.
        """
        self.rhs_keys.update(terms)
        tally = self._attempt
        if tally is None:
            return
        w = self._weight
        Vp, Vc, Va = self._Vp, self._Vc, self._Va
        for name in applied:
            term = terms[name]
            n = np.asarray(term.n, dtype=float)
            xp = np.where(self._active, Vp * n, 0.0)
            nn = np.asarray(term.nn, dtype=float)
            xn = Vc * nn
            if term.nn_a is not None:
                xn = xn + Va * np.asarray(term.nn_a, dtype=float)
            site = _RHS_SITES.get(name)
            if name == "plasma_advective_flux":
                self._add(tally, name, PLASMA, PLASMA, "signed", site, xp, w)
                self._unbooked(name, xn, w)
            elif name in ("neutral_exchange", "neutral_zone_exchange"):
                self._add(tally, name, NEUTRAL, NEUTRAL, "signed", site, xn, w)
                self._unbooked(name, xp, w)
            elif name == "neutral_sources":
                self._unbooked(name, xp, w)
                if engaged:
                    # The engine owns the neutrals: the term's rows are
                    # stripped and the puff and pumps are the engine's own.
                    self._unbooked(name, xn, w)
                else:
                    self._book_sources(tally, name, site, source_parts, w)
            elif name in _EXCHANGE_TERMS:
                clearing, sign = _EXCHANGE_TERMS[name]
                if sign > 0:
                    self._add(tally, name, clearing, PLASMA, "one-signed",
                              site, xp, w)
                else:
                    self._add(tally, name, PLASMA, clearing, "one-signed",
                              site, -xp, w)
                if not engaged:
                    if sign > 0:
                        self._add(tally, f"{name}.neutral", NEUTRAL, clearing,
                                  "one-signed", site, -xn, w)
                    else:
                        self._add(tally, f"{name}.neutral", clearing, NEUTRAL,
                                  "one-signed", site, xn, w)
                else:
                    self._unbooked(name, xn, w)
            elif name == "characteristic_boundary":
                for role, part, clearing in (
                    ("end_wall", name, X_END_WALL),
                    ("cathode", f"{name}.cathode_face", X_CATHODE_FACE),
                ):
                    mask = self._role_masks.get(role)
                    if mask is None:
                        continue
                    self._add(tally, part, PLASMA, clearing, "one-signed",
                              site, np.where(mask, -xp, 0.0), w)
                    if not engaged:
                        self._add(tally, f"{part}.neutral" if part != name
                                  else f"{name}.end_wall_neutral",
                                  clearing, NEUTRAL, "one-signed", site,
                                  np.where(mask, xn, 0.0), w)
                self._unbooked(name, np.where(self._boundary_other, xp, 0.0), w)
                self._unbooked(
                    name,
                    np.where(self._boundary_other, xn, 0.0) if not engaged
                    else xn,
                    w,
                )
            else:
                self._unbooked(name, xp, w)
                self._unbooked(name, xn, w)

    def _unbooked(self, name, x, w):
        g = float(np.abs(np.asarray(x, dtype=float)).sum())
        if g:
            self.unbooked[name] = self.unbooked.get(name, 0.0) + abs(w) * g

    def _book_sources(self, tally, name, site, parts, w):
        Vc, Va = self._Vc, self._Va
        puff = Vc * parts["puff_nn"]
        pump = Vc * parts["pump_nn"]
        if parts.get("puff_nn_a") is not None:
            puff = puff + Va * parts["puff_nn_a"]
            pump = pump + Va * parts["pump_nn_a"]
        self._add(tally, name, PUFF, NEUTRAL, "one-signed", site, puff, w)
        self._add(tally, f"{name}.pump", NEUTRAL, PUMP, "one-signed", site,
                  pump, w)

    def book_floor(self, raw, floored, floors, weight, engaged):
        """Book one floor call's particle additions at ``weight``."""
        tally = self._attempt
        if tally is None:
            return
        site = "core/state.py:apply_state_floors"
        n_raw = np.asarray(raw.n, dtype=float)
        x = np.where(
            self._active & (n_raw < floors["n"]),
            (np.asarray(floored.n, dtype=float) - n_raw) * self._Vp,
            0.0,
        )
        self._add(tally, "floor_density", FLOORS, PLASMA, "one-signed", site,
                  x, weight)
        if engaged:
            return
        nn_raw = np.asarray(raw.nn, dtype=float)
        xn = np.where(
            nn_raw < floors["nn"],
            (np.asarray(floored.nn, dtype=float) - nn_raw) * self._Vc,
            0.0,
        )
        if raw.nn_a is not None:
            na_raw = np.asarray(raw.nn_a, dtype=float)
            xn = xn + np.where(
                na_raw < floors["nn"],
                (np.asarray(floored.nn_a, dtype=float) - na_raw) * self._Va,
                0.0,
            )
        self._add(tally, "floor_neutral_density", FLOORS, NEUTRAL,
                  "one-signed", site, xn, weight)

    def book_neutral_step(self, parts):
        """Book one backward-Euler neutral-only step (weight one)."""
        tally = self._attempt
        if tally is None:
            return
        site = "solver.py:_implicit_neutral_step_two_zone"
        Vc, Va = self._Vc, self._Va
        x_c = parts["nn_next"]
        x_a = parts["nn_a_next"]
        name = "implicit_neutral_step"
        self._add(tally, name, PUFF, NEUTRAL, "one-signed", site,
                  Vc * parts["puff_c"] + Va * parts["puff_a"], 1.0)
        pump = np.zeros(self._cells)
        for index, coeff in parts["pump"]:
            pump[index] += coeff * (x_c[index] * Vc[index] + x_a[index] * Va[index])
        self._add(tally, f"{name}.pump", NEUTRAL, PUMP, "one-signed", site,
                  pump, 1.0)
        dt = parts["dt"]
        gross = 0.0
        for coeff, x in ((parts["column_coeff"], x_c), (parts["annulus_coeff"], x_a)):
            c = np.asarray(coeff, dtype=float)
            gross += float(np.sum(2.0 * dt * np.where(c > 0.0, c, 0.0)
                                  * (np.abs(x[:-1]) + np.abs(x[1:]))))
        kr = np.asarray(parts["zone_exchange"], dtype=float)
        gross += float(np.sum(2.0 * dt * np.where(kr > 0.0, kr, 0.0)
                              * (np.abs(x_c) + np.abs(x_a))))
        self._spec(f"{name}.transport", NEUTRAL, NEUTRAL, "signed", site)
        slot = tally.setdefault(f"{name}.transport", [0.0, 0.0])
        slot[1] += gross

    def book_tick(self, ledger, external_births):
        """Book one kinetic engine tick from its particle ledger (weight one).

        ``external_births`` is the engine's own list of births that come
        from outside its inventory; every other birth and every loss but
        ionization and the two pumps is internal to the account.
        """
        mapping = {
            "recombination": ("kinetic_neutrals.recombination", X_RECOMBINATION),
            "cathode_face": ("kinetic_neutrals.cathode_face", X_CATHODE_FACE),
            "cathode_jet": ("kinetic_neutrals.cathode_face", X_CATHODE_FACE),
            "end_wall_face": ("kinetic_neutrals.end_wall", X_END_WALL),
            "end_wall_jet": ("kinetic_neutrals.end_wall", X_END_WALL),
            "anode": ("kinetic_neutrals.anode", X_ANODE),
            "anode_jet": ("kinetic_neutrals.anode", X_ANODE),
            "puff": ("kinetic_neutrals.puff", PUFF),
        }
        missing = sorted(set(external_births) - set(mapping))
        if missing:
            raise ValueError(
                f"kinetic external birth channel(s) {missing} have no receipt "
                "booking"
            )
        sums = {}
        for channel in external_births:
            term, source = mapping[channel]
            v = float(ledger[f"birth_{channel}"])
            if source == X_END_WALL and "end_wall" not in self._role_masks and (
                v == 0.0
            ):
                # No end wall face on this geometry (a mirror or twin): the
                # plasma books nothing on the channel, so neither side does.
                # A non-zero birth is still booked, and fails the check.
                continue
            s = sums.setdefault(term, [source, 0.0, 0.0])
            s[1] += v
            s[2] += abs(v)
        for term, (source, value, gross) in sums.items():
            self._add_scalar(term, source, NEUTRAL, "one-signed", _ENGINE_SITE,
                             value, gross)
        v = float(ledger["loss_ionization"])
        self._add_scalar("kinetic_neutrals.ionization", NEUTRAL, X_IONIZATION,
                         "signed", _ENGINE_SITE, v, abs(v))
        for side in ("L", "R"):
            v = float(ledger[f"loss_pump_{side}"])
            self._add_scalar(f"kinetic_neutrals.pump_{side}", NEUTRAL, PUMP,
                             "one-signed", _ENGINE_SITE, v, abs(v))
        # Everything else moves atoms inside the engine's inventory: the
        # charge-exchange, elastic, wall, mesh, baffle and closed-face pairs,
        # and the end planes' outflow into the lagged return buffers (the
        # buffers are part of the inventory) less the pumped share, against
        # the buffers' release. Net, it is the change of the buffers.
        net = 0.0
        gross = 0.0
        for key, value in ledger.items():
            if key.startswith("birth_"):
                if key[len("birth_"):] not in external_births:
                    net += float(value)
                    gross += abs(float(value))
            elif key.startswith("loss_") and key not in (
                "loss_ionization", "loss_pump_L", "loss_pump_R"
            ):
                net -= float(value)
                gross += abs(float(value))
        net += float(ledger["loss_pump_L"]) + float(ledger["loss_pump_R"])
        self._add_scalar("kinetic_neutrals.internal", NEUTRAL, NEUTRAL,
                         "signed", _ENGINE_SITE, net, gross)

    # ------------------------------------------------------------ saves

    def start_run(self):
        """Begin a new trajectory: its saves are this receipt's frames.

        The open interval's accumulators are kept; the run's first save
        closes them into its first frame, which opens the receipt and is no
        interval of its own.
        """
        self.frames = []

    def save(self, state_rows):
        """Close the interval at a save; ``state_rows`` is this save's state."""
        frame = {
            "entries": self._interval,
            "steps": self._interval_steps,
            "state": dict(state_rows),
        }
        if self._resumed:
            # The first save after a restart has no frame at the hand-off in
            # this run's trajectory, so its partial interval is not closable.
            self.leading_steps_not_covered = self._interval_steps
            frame["entries"] = {}
            frame["steps"] = 0
            self._resumed = False
        self.frames.append(frame)
        self._interval = {}
        self._interval_steps = 0

    # ------------------------------------------------------------ restart

    def restart_members(self):
        """The accumulators a restart payload carries, as flat floats."""
        out = {"interval_steps": float(self._interval_steps)}
        for term, (value, gross) in self._interval.items():
            out[f"value:{term}"] = float(value)
            out[f"gross:{term}"] = float(gross)
        return out

    def load_restart_members(self, members):
        """Restore :meth:`restart_members`; an older payload has none."""
        self._resumed = True
        members = dict(members or {})
        self._interval_steps = int(members.pop("interval_steps", 0.0))
        for key, value in members.items():
            kind, term = key.split(":", 1)
            slot = self._interval.setdefault(term, [0.0, 0.0])
            slot[0 if kind == "value" else 1] = float(value)

    # ------------------------------------------------------------ result

    def result(self, rhs_keys=()):
        """Return the receipt as plain arrays for the result writer.

        The first save opens the receipt; each later save closes one
        interval, so ``n_saves - 1`` intervals.
        """
        frames = self.frames
        n_int = max(len(frames) - 1, 0)
        entries = {}
        for term, (debit, credit, sign, site) in sorted(self._specs.items()):
            values = np.zeros(n_int)
            gross = np.zeros(n_int)
            for k, frame in enumerate(frames[1:]):
                slot = frame["entries"].get(term)
                if slot is not None:
                    values[k], gross[k] = slot
            entries[term] = {
                "values": values,
                "gross": gross,
                "debit": debit,
                "credit": credit,
                "sign": sign,
                "site": site,
            }
        steps = np.asarray([f["steps"] for f in frames[1:]], dtype=np.int64)
        state_names = sorted({k for f in frames for k in f["state"]})
        state = {
            name: np.asarray([f["state"].get(name, 0.0) for f in frames],
                             dtype=float)
            for name in state_names
        }
        census = self._census(set(rhs_keys) | self.rhs_keys, entries)
        return {
            "schema": SCHEMA,
            "cadence": "save",
            "stages_present": [STAGE],
            "updates_per_step": UPDATES_PER_STEP,
            "interval_steps": steps,
            "entries": entries,
            "state": state,
            "census": census,
            "leading_steps_not_covered": int(self.leading_steps_not_covered),
        }

    def _census(self, rhs_keys, entries):
        census = {}
        parts_of = {}
        for term in entries:
            base = term.split(".", 1)[0]
            parts_of.setdefault(base, []).append(term)
        for key in sorted(rhs_keys):
            if key.startswith("_"):
                continue
            if key in self.declared:
                census[key] = ("not_tracked", self.declared[key])
            elif key in BOOKED_RHS_TERMS and key in entries:
                census[key] = (
                    "entered",
                    "booked as " + ", ".join(sorted(parts_of.get(key, [key])))
                    + "; " + _RHS_REASON.get(key, "")
                )
            elif key in self.unbooked:
                census[key] = (
                    "not_tracked",
                    "carries particle rows this receipt does not book "
                    f"(gross {self.unbooked[key]:.6e} particles)",
                )
            elif key in BOOKED_RHS_TERMS:
                census[key] = ("zero", "never applied inside a step attempt")
            else:
                census[key] = ("zero", "carries no particle row")
        for term in entries:
            if term not in census:
                census[term] = ("entered", _PART_REASON.get(
                    term, f"part of {term.split('.', 1)[0]}"))
        for term, reason in _OUTSIDE_RHS.items():
            if term not in census:
                census[term] = (
                    "entered" if term in entries else "zero", reason
                )
        if self.kinetic:
            delta = self.handover
            census["kinetic_neutrals.engagement_handover"] = (
                "zero",
                "the engine is seeded from the fluid column and annulus "
                "densities and is not booked; measured kinetic minus fluid "
                "inventory at engagement: "
                + ("not engaged" if delta is None else
                   f"{delta[0]:.6e} particles of {delta[1]:.6e}"),
            )
            census["kinetic_neutrals.republish"] = (
                "zero",
                "the fluid nn/nn_a rows are rewritten from the engine's "
                "moments (one-sided at the floor); they are not the "
                "neutral inventory once the engine owns it",
            )
            census["floor_neutral_density.after_engagement"] = (
                "zero",
                "neutral floor additions after engagement land on the "
                "republished fluid rows, not on the engine's inventory",
            )
        if self.leading_steps_not_covered:
            census["restart_resume"] = (
                "not_tracked",
                "a resumed run saves no frame at the hand-off, so the "
                f"{self.leading_steps_not_covered} accepted steps before its "
                "first save fall in no interval",
            )
        for account, reason in self.accounts_not_tracked.items():
            census[account] = ("not_tracked", reason)
        return census


_RHS_REASON = {
    "plasma_advective_flux": "interior faces move plasma between cells and "
    "closed faces carry none: a self-entry on plasma_particles",
    "characteristic_boundary": "the plasma removed at each absorbing face, "
    "by role (characteristic_boundary: end wall; .cathode_face: cathode), "
    "booked to that face's recycle clearing account; the fluid nn rows "
    "rebirth it from the other side",
    "ionization_birth": "bulk ionization through exchange:ionization",
    "beam_ionization_birth": "beam ionization through exchange:ionization",
    "recombination_rad_loss": "recombination through exchange:recombination",
    "anode_collection": "anode-mesh collection through exchange:anode_return",
    "neutral_exchange": "axial neutral exchange: a self-entry on "
    "neutral_particles",
    "neutral_zone_exchange": "column/annulus exchange: a self-entry on "
    "neutral_particles",
    "neutral_sources": "the gas puff (neutral_sources) and the end pumps "
    "(neutral_sources.pump)",
}
_PART_REASON = {
    "floor_density": "plasma density floor additions at the floor ledger's "
    "weights",
    "floor_neutral_density": "column and annulus neutral density floor "
    "additions at the floor ledger's weights, before engagement",
}
_PART_REASON.update({
    "kinetic_neutrals.ionization": "the engine's ionization debit at its "
    "tick, the side of exchange:ionization opposite the plasma's births",
    "kinetic_neutrals.recombination": "recombination births at the tick, "
    "from the counted recombination channel",
    "kinetic_neutrals.cathode_face": "cathode-face recycle births at the "
    "tick (thermal and jet shares), from the counted channel",
    "kinetic_neutrals.end_wall": "end wall recycle births at the tick "
    "(thermal and jet shares), from the counted channel",
    "kinetic_neutrals.anode": "anode return births at the tick (thermal and "
    "jet shares), from the counted channel",
    "kinetic_neutrals.puff": "the gas puff the engine births at its tick",
    "kinetic_neutrals.pump_L": "the pumped share of the left end plane's "
    "outflow",
    "kinetic_neutrals.pump_R": "the pumped share of the right end plane's "
    "outflow",
    "kinetic_neutrals.internal": "transport inside the engine's inventory "
    "(collision rebirths, wall, mesh, baffle and closed-face returns, end "
    "buffers): a self-entry on neutral_particles",
})
_OUTSIDE_RHS = {
    "floor_density": _PART_REASON["floor_density"],
    "floor_neutral_density": _PART_REASON["floor_neutral_density"],
    "implicit_neutral_step": "the backward-Euler neutral-only step: puff "
    "(this entry), pumps (.pump) and exchange (.transport, a self-entry)",
    "implicit_heat_substep": "the implicit heat substep copies the particle "
    "rows unchanged",
}
