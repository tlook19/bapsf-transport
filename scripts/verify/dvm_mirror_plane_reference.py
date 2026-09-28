"""Capture/verify the DVM mirror-plane fixture corpus.

The corpus ``scripts/data/dvm_mirror_plane_reference.npz`` pins the transient
DVM's MIRROR-PLANE branch -- ``TransientDVM._march`` re-injecting the ``+z``
sweep's right-end outflow, bin-mirrored, as the ``-z`` sweep's inflow, and
``TransientDVM.update`` booking that same-tick return in its particle and
energy ledgers -- at raw float64, so the branch is pinned by data rather than
by a second implementation of it.

WHAT IS IN IT. Three arms, each a synthetic tube ending in a mirror face
(``mirror_face_indices == [nz]``), a velocity grid and a sequence of TICKS:

``oneside16``
    a 6-cell tube on a 16 x 6 grid, gas only in its last three cells and only
    in the ``+v_z`` bins of both zones, collisionless and sourceless -- the
    plane taking the whole traffic;
``collide48``
    a 10-cell tube on a 48 x 12 grid, seeded at rest, with charge exchange,
    elastic scattering and ionization against a drifting ion profile, a puff
    into the annulus, a recombination birth and a live cathode-surface
    temperature on the left end;
``closed16``
    a 6-cell tube on a 16 x 6 grid with an interior closed face, a partly
    transparent mesh face and a counted cathode-face recycle entering at the
    closed face.

Each tick stores the column and annulus distributions after the update and
the ledger's numeric rows (particles, then energy), in sorted key order.
Ticks are replayed in order from the stored initial state, so a tick's answer
depends on every tick before it; the arms are replayed exactly as stored.

Usage (from the repo root, PYTHONPATH set to the repo root)::

    python scripts/verify/dvm_mirror_plane_reference.py --capture
    python scripts/verify/dvm_mirror_plane_reference.py --verify
    python scripts/verify/dvm_mirror_plane_reference.py --verify --impl perturbed

``--verify`` reports the number of DIFFERING RAW UINT64 values per arm and
array -- float64 bit patterns compared as integers, so a one-ulp move is a
difference. The bar is zero. ``--impl perturbed`` is the NEGATIVE CONTROL: it
moves the last value of every ghost density the march receives by one ulp and
nothing else; every arm must then report a non-zero count and the script
exits 1. ``--capture`` rewrites the corpus and is a recapture-class event.
"""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cablp.solvers._sim1d.physics import kinetic_dvm as kd  # noqa: E402

DEFAULT_REFERENCE = (
    Path(__file__).resolve().parents[1] / "data"
    / "dvm_mirror_plane_reference.npz"
)
TICKS = 3


def _tube(nz, closed_face=None):
    """A synthetic tube of ``nz`` cells ending in a mirror face."""
    lengths = np.linspace(4.0, 7.0, nz)
    Rp = np.linspace(1.5, 2.5, nz)
    Rm = np.full(nz, 5.0)
    geometry = SimpleNamespace(
        length_cm=lengths,
        plasma_volume_cm3=np.pi * Rp**2 * lengths,
        neutral_volume_cm3=np.pi * Rm**2 * lengths,
        Rp_cm=Rp,
        Rm_cm=Rm,
        mirror_face_indices=np.array([nz]),
    )
    if closed_face is not None:
        plasma_open = np.ones(nz + 1, dtype=bool)
        plasma_open[[0, closed_face, nz]] = False
        geometry.plasma_open = plasma_open
    return geometry


def _arms():
    """Return ``{name: (engine, initial (f_c, f_a), per-tick update kwargs)}``."""
    arms = {}

    engine = kd.TransientDVM(geometry=_tube(6), nvz=16, nvp=6)
    nz, g = engine.nz, engine.g
    one_sided = engine.M_cold * (g.vz > 0.0)[:, None]
    f_c = np.zeros((nz, g.nvz, g.nvp))
    f_a = np.zeros_like(f_c)
    f_c[-3:] = np.array([1.0e12, 2.0e12, 3.0e12])[:, None, None] * one_sided
    f_a[-3:] = np.array([3.0e11, 2.0e11, 1.0e11])[:, None, None] * one_sided
    quiet = dict(
        n_i=np.zeros(nz), Ti_eV=np.ones(nz), u_i=np.zeros(nz),
        nu_ion=np.zeros(nz),
    )
    arms["oneside16"] = (engine, (f_c, f_a), [(1.0e-4, quiet)] * TICKS)

    engine = kd.TransientDVM(geometry=_tube(10), nvz=48, nvp=12)
    nz = engine.nz
    engine.seed_from_density(
        np.linspace(1.0e12, 3.0e12, nz), np.linspace(4.0e11, 2.0e11, nz)
    )
    f_c, f_a = engine.f_c.copy(), engine.f_a.copy()
    kwargs = dict(
        n_i=np.linspace(5.0e11, 2.0e12, nz),
        Ti_eV=np.linspace(0.5, 3.0, nz),
        u_i=np.linspace(-2.0e5, 3.0e5, nz),
        nu_ion=np.linspace(1.0e3, 5.0e3, nz),
        sources={
            "puff": np.linspace(0.0, 1.0e17, nz),
            "recombination": np.linspace(1.0e15, 0.0, nz),
        },
        T_s_K=1500.0,
    )
    arms["collide48"] = (engine, (f_c, f_a), [(3.0e-5, kwargs)] * TICKS)

    engine = kd.TransientDVM(
        geometry=_tube(6, closed_face=1), nvz=16, nvp=6,
        mesh_face=4, transparency=0.6,
    )
    nz = engine.nz
    engine.seed_from_density(np.full(nz, 1.0e12), np.full(nz, 3.0e11))
    f_c, f_a = engine.f_c.copy(), engine.f_a.copy()
    recycle = np.zeros(nz)
    recycle[1] = 2.0e13
    kwargs = dict(
        n_i=np.full(nz, 1.0e12), Ti_eV=np.full(nz, 1.0),
        u_i=np.zeros(nz), nu_ion=np.full(nz, 2.0e3),
        source_counts={"cathode_face": recycle},
    )
    arms["closed16"] = (engine, (f_c, f_a), [(5.0e-5, kwargs)] * TICKS)
    return arms


def _ledger_vector(ledger):
    """The ledger's numeric rows, particles then energy, sorted by key."""
    rows = [float(ledger[k]) for k in sorted(ledger) if k != "energy"]
    energy = ledger["energy"]
    rows += [float(energy[k]) for k in sorted(energy)]
    return np.asarray(rows, dtype=float)


def _replay():
    """Replay every arm; return ``{key: array}`` in the corpus layout."""
    out = {}
    for name, (engine, (f_c, f_a), ticks) in _arms().items():
        assert engine.mirror_plane, name
        engine.f_c, engine.f_a = f_c.copy(), f_a.copy()
        out[f"{name}/f_c_initial"] = f_c
        out[f"{name}/f_a_initial"] = f_a
        for k, (dt, kwargs) in enumerate(ticks):
            ledger = engine.update(dt, **kwargs)
            out[f"{name}/tick{k}/f_c"] = engine.f_c.copy()
            out[f"{name}/tick{k}/f_a"] = engine.f_a.copy()
            out[f"{name}/tick{k}/ledger"] = _ledger_vector(ledger)
            assert not np.any(engine.pend_R_c) and not np.any(engine.pend_R_a)
    return out


def _differing(got, want):
    got = np.ascontiguousarray(np.asarray(got, dtype=float))
    want = np.ascontiguousarray(np.asarray(want, dtype=float))
    if got.shape != want.shape:
        return max(got.size, want.size)
    return int(np.count_nonzero(got.view(np.uint64) != want.view(np.uint64)))


def capture(path):
    corpus = _replay()
    np.savez_compressed(path, **corpus)
    total = sum(np.asarray(v).size for v in corpus.values())
    print(f"captured {len(corpus)} arrays, {total} values -> {path}")
    return 0


def verify(path, impl="live"):
    stored = np.load(path)
    original = kd._ghost_density
    if impl == "perturbed":
        def perturbed(*args, **kwargs):
            dens = np.array(original(*args, **kwargs), dtype=float)
            flat = dens.reshape(-1)
            flat[-1] = np.nextafter(flat[-1], np.inf)
            return dens

        kd._ghost_density = perturbed
    try:
        live = _replay()
    finally:
        kd._ghost_density = original
    if sorted(live) != sorted(stored.files):
        print(
            "CORPUS LAYOUT DIFFERS: live "
            f"{sorted(set(live) - set(stored.files))} not stored, stored "
            f"{sorted(set(stored.files) - set(live))} not live"
        )
        return 2
    total_values = 0
    total_diff = 0
    per_arm = {}
    for key in sorted(live):
        diff = _differing(live[key], stored[key])
        total_values += stored[key].size
        total_diff += diff
        arm = key.split("/")[0]
        per_arm[arm] = per_arm.get(arm, 0) + diff
        if diff:
            print(f"  {key}: {diff} of {stored[key].size} differ")
    for arm, diff in per_arm.items():
        print(f"arm {arm}: {diff} differing")
    print(
        f"dvm_mirror_plane_reference ({impl}): {total_diff} differing of "
        f"{total_values} raw uint64 values"
    )
    if impl == "perturbed":
        ok = all(diff > 0 for diff in per_arm.values())
        print(
            "negative control: every arm differs"
            if ok else "NEGATIVE CONTROL FAILED: an arm did not move"
        )
        return 1
    return 0 if total_diff == 0 else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--capture", action="store_true",
                      help="rewrite the corpus (a recapture-class event)")
    mode.add_argument("--verify", action="store_true",
                      help="replay the corpus and count differing values")
    parser.add_argument("--impl", choices=("live", "perturbed"),
                        default="live")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    args = parser.parse_args(argv)
    if args.capture:
        return capture(args.reference)
    return verify(args.reference, args.impl)


if __name__ == "__main__":
    sys.exit(main())
