"""LANE-MARCH equivalence: per-walker A/B of the two tail-leg routes.

``deposit_beam``'s ionizing tail walk has two implementations of the same
legs -- ``_tail_recursive_chains`` (one recursive ``deposit_beam`` per leg, the
route it has always taken) and ``_tail_lane_chains`` (all legs marched together
by ``cathode.beam_lane_march``). Which one runs is a cost threshold,
``beam_deposition.LANE_MARCH_MIN_LEGS``, and nothing else, so the claim under
test is that they produce THE SAME FLOATS.

This runs both on the same inputs and compares them WALKER BY WALKER, which is
what the threshold-cliff hazard needs: a 1-ULP reordering does not show up as a
small per-cell deviation, it shows up as a walker taking a different branch --
one more or one fewer substep, absorbing instead of transmitting, reflecting
instead of escaping. So the census counts FLIPPED WALKERS and weighs them by
the energy they deposited, rather than reporting a per-cell tolerance.

Two modes, both reporting at raw uint64:

* ``--corpus`` replays every entry of the committed ``deposit_beam`` reference
  fixture, intercepting each ray's tail-leg batch and running it both ways;
* ``--random N`` marches N randomized lane batteries directly against
  per-lane ``deposit_beam`` calls, covering vacuum cells, zero-length cells,
  both Coulomb closures, both directions, and launch energies straddling the
  stop threshold and the ionization potential.

Usage::

    python scripts/r3lane_equivalence.py --corpus
    python scripts/r3lane_equivalence.py --random 400
    python scripts/r3lane_equivalence.py --corpus --random 400
"""

import argparse
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import cablp.cathode.beam_deposition as B  # noqa: E402
from cablp.cathode.beam_lane_march import lane_march  # noqa: E402

#: Pre-registered bar (R3 sub-event 2): the energy-weighted effect of ALL
#: flipped walkers, as a fraction of the per-call deposited energy.
FLIP_ENERGY_BAR = 1.0e-9


def _bits(a):
    return np.ascontiguousarray(a, dtype=float).view(np.uint64)


def _differs(a, b):
    a = np.atleast_1d(np.asarray(a, dtype=float))
    b = np.atleast_1d(np.asarray(b, dtype=float))
    if a.shape != b.shape:
        return True
    return bool(np.any(_bits(a) != _bits(b)))


def _leg_energy(banks):
    """Energy [erg/s] a leg deposited: its heating, radiation and ionization cost."""
    _ion, _exc, cost, rad, heat = banks
    return float(np.sum(cost) + np.sum(rad) + np.sum(heat))


class Census:
    """Flipped walkers and their energy weight, accumulated over many calls."""

    def __init__(self):
        self.walkers = 0
        self.flipped = 0
        self.flipped_energy = 0.0
        self.total_energy = 0.0
        self.calls = 0
        self.worst_call_ratio = 0.0
        self.takes = 0
        self.take_diffs = 0
        self.notes = []
        # Reflecting-face arrivals, which is what makes the per-birth exit
        # state load-bearing: a batch is only evidence about it if the corpus
        # actually presents BOTH sides of the threshold at the same face.
        self.face_arrivals = 0
        self.face_reflected = 0
        self.face_escaped = 0

    def face_census(self, ref_chains, reflect_face):
        if reflect_face is None:
            return
        for chains in ref_chains:
            if chains is None:
                continue
            for chain in chains:
                banks, flux, energy, direction = chain[0]
                if direction != reflect_face or flux <= 0.0:
                    continue
                self.face_arrivals += 1
                if len(chain) > 1:
                    self.face_reflected += 1
                else:
                    self.face_escaped += 1

    def compare_take(self, lane_take, ref_take, label):
        """The four-scalar anode take both routes return beside their layout."""
        self.takes += 1
        if _differs(lane_take, ref_take):
            self.take_diffs += 1
            if len(self.notes) < 20:
                self.notes.append(
                    f"{label}: anode take differs -- lane {tuple(lane_take)} "
                    f"vs recursive {tuple(ref_take)}"
                )

    def compare(self, lane_chains, ref_chains, label):
        self.calls += 1
        call_flip = 0.0
        call_total = 0.0
        for slot, (got, want) in enumerate(zip(lane_chains, ref_chains)):
            if (got is None) != (want is None):
                self.notes.append(f"{label}: population {slot} ionization flag")
                self.flipped += 1
                continue
            if got is None:
                continue
            if len(got) != len(want):
                self.notes.append(
                    f"{label}: population {slot} walker count "
                    f"{len(got)} != {len(want)}"
                )
                self.flipped += abs(len(got) - len(want))
                continue
            for chain_got, chain_want in zip(got, want):
                self.walkers += 1
                energy = sum(_leg_energy(leg[0]) for leg in chain_want)
                call_total += energy
                flip = len(chain_got) != len(chain_want)
                if not flip:
                    for lg, lw in zip(chain_got, chain_want):
                        if lg[3] != lw[3]:
                            flip = True
                        elif _differs(lg[1], lw[1]) or _differs(lg[2], lw[2]):
                            flip = True
                        else:
                            for bg, bw in zip(lg[0], lw[0]):
                                if _differs(bg, bw):
                                    flip = True
                                    break
                        if flip:
                            break
                if flip:
                    self.flipped += 1
                    call_flip += energy
                    if len(self.notes) < 20:
                        self.notes.append(f"{label}: walker flipped")
        self.flipped_energy += call_flip
        self.total_energy += call_total
        if call_total > 0.0:
            self.worst_call_ratio = max(
                self.worst_call_ratio, call_flip / call_total
            )

    def report(self):
        ratio = (
            self.flipped_energy / self.total_energy
            if self.total_energy > 0.0 else 0.0
        )
        print(
            f"branch-flip census: {self.walkers} walkers over {self.calls} "
            f"calls, {self.flipped} flipped"
        )
        print(
            f"  energy-weighted effect of flipped walkers: {ratio:.6e} of "
            f"{self.total_energy:.6e} erg/s deposited "
            f"(worst single call {self.worst_call_ratio:.6e}); "
            f"bar {FLIP_ENERGY_BAR:.1e}"
        )
        print(
            f"  reflecting-face arrivals: {self.face_arrivals} "
            f"({self.face_reflected} below threshold and turned around, "
            f"{self.face_escaped} above it and free-escaping)"
        )
        print(
            f"  anode take: {self.takes} compared, {self.take_diffs} differing"
        )
        for line in self.notes[:20]:
            print(f"    {line}")
        return (
            self.flipped == 0
            and self.take_diffs == 0
            and ratio <= FLIP_ENERGY_BAR
        )


def run_corpus(census):
    """Replay the reference corpus, running each tail-leg batch BOTH ways."""
    import deposit_beam_reference as R

    fixture = SCRIPT_DIR / "data" / "deposit_beam_reference.npz"
    z = np.load(fixture, allow_pickle=False)
    entries = R._unpack(z)
    real_lane = B._tail_lane_chains
    real_min = B.LANE_MARCH_MIN_LEGS
    state = {"label": ""}
    batches = [0]
    batched = [0]

    def dual(plans, *args, **kwargs):
        before = B.LANE_MARCH_COUNTS["legs"]
        got, got_take = real_lane(plans, *args, **kwargs)
        # A batch that fell back to the recursive route would compare itself
        # against itself, which is no evidence at all; count the ones that
        # really took the lane route so the census cannot be vacuous.
        if B.LANE_MARCH_COUNTS["legs"] > before:
            batched[0] += 1
        want, want_take = B._tail_recursive_chains(plans, *args, **kwargs)
        batches[0] += 1
        census.compare(got, want, state["label"])
        # The anode take is route output too, so a silent divergence there
        # fails the equivalence exactly as a flipped walker does.
        census.compare_take(got_take, want_take, state["label"])
        census.face_census(want, args[7])
        return got, got_take

    # Threshold forced to 1 so EVERY batch takes the lane route, including the
    # small ones the shipped threshold sends down the recursive one -- the
    # question here is equivalence, not which route is cheaper.
    B.LANE_MARCH_MIN_LEGS = 1
    B._tail_lane_chains = dual
    try:
        for entry in entries:
            state["label"] = entry["label"]
            R._invoke(B, entry)
    finally:
        B._tail_lane_chains = real_lane
        B.LANE_MARCH_MIN_LEGS = real_min
    print(
        f"corpus: {len(entries)} entries, {batches[0]} tail-leg batches, "
        f"{batched[0]} of them actually marched as lanes"
    )


def run_random(census, trials, seed=20260827):
    """Randomized lane batteries against per-lane ``deposit_beam`` calls."""
    rng = np.random.default_rng(seed)
    I_ion = B.HE_I_ION_EV
    E_stop = B.HE_E_STOP_EV
    values = 0
    bad = 0
    lanes_done = 0
    for _ in range(trials):
        C = int(rng.integers(1, 40))
        nn = 10.0 ** rng.uniform(11, 15, C)
        ne = 10.0 ** rng.uniform(8, 13, C)
        Te = rng.uniform(0.1, 20.0, C)
        dz = rng.uniform(0.5, 30.0, C)
        if rng.random() < 0.3:
            k = int(rng.integers(0, C))
            nn[k] = 0.0
            ne[k] = 0.0
        if rng.random() < 0.15:
            dz[int(rng.integers(0, C))] = 0.0
        model = "fast_electron" if rng.random() < 0.7 else "legacy_tau_ei"
        frac = float(rng.choice([0.02, 0.05, 0.2, 0.005]))
        L = int(rng.integers(1, 40))
        E0 = rng.uniform(E_stop * 1.0000001, 400.0, L)
        m = max(1, L // 3)
        if rng.random() < 0.4:
            E0[:m] = E_stop * (1.0 + rng.uniform(0.0, 1e-9, m))
        if rng.random() < 0.3:
            E0[:m] = I_ion * (1.0 + rng.uniform(-1e-9, 1e-9, m))
            E0 = np.maximum(E0, E_stop * 1.0000001)
        g0 = 10.0 ** rng.uniform(10, 17, L)
        launch = rng.integers(0, C, L)
        dirn = rng.choice([-1, 1], L)
        res = lane_march(
            E0, g0, launch, dirn, nn, ne, Te, dz,
            I_ion_eV=I_ion, E_stop_eV=E_stop, coulomb_model=model,
            max_energy_fraction_per_substep=frac,
        )
        real_exc = B.He_beam_excitation_channel_lkup
        for k in range(L):
            n_sub = [0]

            def counting(E, _r=real_exc, _n=n_sub):
                _n[0] += 1
                return _r(E)

            B.He_beam_excitation_channel_lkup = counting
            try:
                want = B.deposit_beam(
                    float(E0[k]), float(g0[k]), nn, ne, Te, int(launch[k]),
                    int(dirn[k]), dz, I_ion_eV=I_ion, E_stop_eV=E_stop,
                    coulomb_model=model, anomalous_model="none",
                    max_energy_fraction_per_substep=frac,
                )
            finally:
                B.He_beam_excitation_channel_lkup = real_exc
            lanes_done += 1
            pairs = (
                (res.ionization_events[k], want.ionization_events),
                (res.excitation_events[k], want.excitation_events),
                (res.plasma_heating_erg_s[k], want.plasma_heating_erg_s),
                (res.radiated_erg_s[k], want.radiated_erg_s),
                (res.ionization_cost_erg_s[k], want.ionization_cost_erg_s),
                ([res.transmitted_flux[k]], [want.transmitted_flux]),
                ([res.transmitted_energy_eV[k]], [want.transmitted_energy_eV]),
            )
            for got, ref in pairs:
                got = np.atleast_1d(np.asarray(got, dtype=float))
                ref = np.atleast_1d(np.asarray(ref, dtype=float))
                values += ref.size
                bad += int(np.count_nonzero(_bits(got) != _bits(ref)))
            values += 1
            if int(res.substeps[k]) != n_sub[0]:
                bad += 1
    print(
        f"random battery: {trials} batteries, {lanes_done} lanes, "
        f"{values} values, {bad} differing"
    )
    return bad == 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--corpus", action="store_true")
    p.add_argument("--random", type=int, default=0, metavar="N")
    a = p.parse_args(argv)
    if not (a.corpus or a.random):
        p.error("nothing to do: pass --corpus and/or --random N")
    ok = True
    census = Census()
    if a.corpus:
        run_corpus(census)
        ok = census.report() and ok
    if a.random:
        ok = run_random(census, a.random) and ok
    print("LANE EQUIVALENCE OK" if ok else "LANE EQUIVALENCE FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
