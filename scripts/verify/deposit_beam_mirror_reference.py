"""Capture/verify the deposit_beam MIRROR fixture corpus.

The corpus ``scripts/data/deposit_beam_mirror_reference.npz`` pins the CSDA
module's MIRROR branch -- ``deposit_beam(mirror_face=...)`` turning the
primary and the tail walkers round at the mirror plane, the mirror chains of
``_tail_mirror_chains`` (the re-armed anode cull, the wire sheath and the
merge of its turned share with its parent at the anode plane, the shared leg
budget and its booked residual), the returning primary's hand-off of its
anomalous drag (``anomalous_bank_eV``), and the beam smoothing's fold about
the mirror face -- at raw float64, so the branch is pinned by data rather than
by a second implementation of it. It is a SEPARATE corpus from
``deposit_beam_reference.npz``: no entry of that corpus passes a mirror face,
and none of its entries is touched here.

WHAT IS IN IT. Nine arms on synthetic half columns:

``fold_walkers``
    the walked plateau tail (quasilinear drag, eight groups, ionizing),
    walkers turning at the mirror, the cathode face a free exit;
``reflect_walkers``
    the same with the cathode face reflecting below ``e*phi_c``, the anode
    cull armed and the wire sheath turning back the walkers below a 60 eV
    drop;
``trapped_walkers``
    the anode sheath repelling every walker (a 1e4 eV drop), two groups: the
    leg budget binds and the residual rows fill;
``primary_turn``
    a primary that reaches the plane and stops on its way back;
``primary_bounce``
    a primary on a near-vacuum column bouncing between the cathode sheath
    and the plane, the anode interception re-armed on every return, until the
    budget books its residual;
``primary_bank``
    a primary reaching the plane under weak quasilinear drag with the walked
    tail on: its returning legs hand their drag to the walked bank;
``primary_net_basis``
    the primary's particle ledger on the net basis (``primary_net_basis``)
    with the walked tail, the wires' sheath at 40 V and half of the
    outbound interception collected: the outbound turned share, the returns'
    sheath rule, the outbound-only gap-born count and the net rows;
``chains_rearm``
    ``_tail_mirror_chains`` called directly on one gap-side walker: its legs'
    banks, directions and transmitted flux/energy, and the chain ledger;
``smoothing``
    the beam-smoothing matrix of the template's half column at nx = 40 and a
    50 cm width.

Every ``BeamDepositionResult`` field of a ray arm is stored, arrays and
scalars alike, as float64.

Usage (from the repo root, PYTHONPATH set to the repo root)::

    python scripts/verify/deposit_beam_mirror_reference.py --capture
    python scripts/verify/deposit_beam_mirror_reference.py --verify
    python scripts/verify/deposit_beam_mirror_reference.py --verify --impl perturbed

``--verify`` reports the number of DIFFERING RAW UINT64 values per arm --
float64 bit patterns compared as integers, so a one-ulp move is a difference.
The bar is zero. The corpus is defined against the pure implementation, so
``--capture`` and ``--verify`` refuse ``CABLP_COMPILED_KERNELS=1``; the
smoke suite's compiled-equivalence case compares the two paths on these arms
directly. ``--impl perturbed`` is the NEGATIVE CONTROL: it moves the launch
energy of every leg the mirror branch marches (every ``deposit_beam`` call
made without a mirror face) and the smoothing width by one ulp and nothing
else; every arm must then report a non-zero count. The control exits 0 when
every arm moved (the control passed) and 1 when an arm did not.
``--capture`` rewrites the corpus and is a recapture-class event.
"""

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cablp.cathode import beam_deposition as bd  # noqa: E402
from cablp.cathode import kernels as _kernels  # noqa: E402
from cablp.constants import I_ion  # noqa: E402
from cablp.solvers._sim1d import default_config  # noqa: E402
from cablp.solvers._sim1d.core.geometry import build_geometry  # noqa: E402
from cablp.solvers._sim1d.physics import cathode as _cathode  # noqa: E402

DEFAULT_REFERENCE = (
    Path(__file__).resolve().parents[1] / "data"
    / "deposit_beam_mirror_reference.npz"
)
ETA = 0.358


def _column(cells):
    return (
        np.full(cells, 1.0e11),
        np.full(cells, 1.0e10) * np.linspace(1.0, 2.0, cells),
        np.linspace(3.0, 2.0, cells),
        np.linspace(8.0, 12.0, cells),
    )


def _mg(cells, **extra):
    kwargs = dict(
        anomalous_model="quasilinear", beam_area_cm2=1000.0,
        anomalous_transport="plateau_multigroup", plateau_edge_eV=30.0,
        tail_ionization="on", tail_walk_window=(0, cells - 1), mirror_face=1,
    )
    kwargs.update(extra)
    return kwargs


def _ray_arms():
    """``{name: (args, kwargs)}`` of every deposit_beam arm."""
    arms = {}
    cells = 24
    nn, ne, Te, dz = _column(cells)
    base = (200.0, 1.0e20, nn, ne, Te, 0, 1, dz)
    arms["fold_walkers"] = (base, _mg(cells))
    arms["reflect_walkers"] = (base, _mg(
        cells, tail_reflect_face=-1, tail_reflect_threshold_eV=200.0,
        tail_anode_cross_index=5, tail_anode_eta=ETA,
        tail_anode_phi_eV=60.0,
    ))
    small = 12
    nn_s, ne_s, Te_s, dz_s = _column(small)
    arms["trapped_walkers"] = ((200.0, 1.0e20, nn_s, ne_s, Te_s, 0, 1, dz_s),
                               _mg(small, tail_reflect_face=-1,
                                   tail_reflect_threshold_eV=200.0,
                                   tail_anode_cross_index=5,
                                   tail_anode_eta=ETA,
                                   tail_anode_phi_eV=1.0e4,
                                   plateau_groups=2))
    cells = 30
    dz = np.linspace(8.0, 12.0, cells)
    Te = np.linspace(3.0, 2.0, cells)
    window = dict(tail_walk_window=(0, cells - 1), mirror_face=1)
    arms["primary_turn"] = (
        (150.0, 1.0e18, np.full(cells, 3.5e14) * np.linspace(1.2, 0.8, cells),
         np.full(cells, 1.0e10), Te, 0, 1, dz),
        dict(window),
    )
    arms["primary_bounce"] = (
        (150.0, 1.0e18, np.zeros(cells), np.full(cells, 1.0e2),
         np.full(cells, 1.0), 0, 1, dz),
        dict(window, anode_cross_index=5, anode_eta=ETA),
    )
    arms["primary_bank"] = (
        (200.0, 1.0e20, np.full(cells, 1.0e11), np.full(cells, 1.0e12), Te,
         0, 1, dz),
        _mg(cells, tail_reflect_face=-1, tail_reflect_threshold_eV=200.0,
            anode_cross_index=5, anode_eta=ETA, tail_anode_cross_index=5,
            tail_anode_eta=ETA),
    )
    arms["primary_net_basis"] = (
        (60.0, 1.0e18, np.full(cells, 3.0e12),
         np.full(cells, 3.0e11) * np.linspace(1.0, 2.0, cells), Te,
         0, 1, dz),
        _mg(cells, tail_reflect_face=-1, tail_reflect_threshold_eV=60.0,
            anode_cross_index=5, anode_eta=ETA, tail_anode_cross_index=5,
            tail_anode_eta=ETA, tail_anode_phi_eV=40.0,
            primary_net_basis=True, primary_anode_collected_fraction=0.5),
    )
    return arms


def _result_arrays(res):
    """Every field of a BeamDepositionResult, as float64 arrays by name."""
    return {
        f.name: np.atleast_1d(np.asarray(getattr(res, f.name), dtype=float))
        for f in dataclasses.fields(res)
    }


def _chains_arm():
    cells = 20
    flux = np.zeros(cells)
    flux[4] = 1.0e18
    layout, ledger = bd._tail_mirror_chains(
        [(100.0, flux, None, True)], np.zeros(cells), np.full(cells, 1.0e2),
        np.full(cells, 1.0), np.full(cells, 10.0),
        dict(I_ion_eV=I_ion, E_stop_eV=bd.HE_E_STOP_EV,
             coulomb_model="fast_electron", anomalous_model="none",
             max_energy_fraction_per_substep=0.02),
        0, cells - 1, None, 0.0, 1, cull=(8, ETA, 0.0, 0.0, 0.0),
    )
    out = {}
    for c, chain in enumerate(layout[0]):
        for k, (banks, t_flux, t_E, direction) in enumerate(chain):
            out[f"chain{c}/leg{k}/banks"] = np.concatenate(banks)
            out[f"chain{c}/leg{k}/scalars"] = np.array(
                [t_flux, t_E, float(direction)]
            )
    out["ledger"] = np.array(
        [ledger[name] for name in bd.MIRROR_CHAIN_LEDGER], dtype=float
    )
    return out


def _smoothing_arm(sigma):
    params, flags = default_config()
    params["nx"] = 40
    geom = build_geometry(dict(params, far_end="mirror"), flags)
    _cathode._BEAM_SMOOTH_CACHE.clear()
    _cathode._BEAM_SMOOTH_KEY_CACHE.clear()
    return {"W": np.array(_cathode._beam_smoothing_matrix(geom, sigma))}


def _replay(sigma=50.0):
    """Replay every arm; return ``{key: array}`` in the corpus layout.

    The run-time residual bound (``MIRROR_RESIDUAL_MAX_FRACTION``) is lifted
    for the ray arms: the arms whose walkers nothing removes exist to pin the
    capped branch and its booked residual, which a solver-facing call refuses.
    The bound tests the result; it changes no float the march produces.
    """
    out = {}
    bound = bd.MIRROR_RESIDUAL_MAX_FRACTION
    bd.MIRROR_RESIDUAL_MAX_FRACTION = float("inf")
    try:
        for name, (args, kwargs) in _ray_arms().items():
            res = bd.deposit_beam(*args, **kwargs)
            for field, value in _result_arrays(res).items():
                out[f"{name}/{field}"] = value
    finally:
        bd.MIRROR_RESIDUAL_MAX_FRACTION = bound
    for key, value in _chains_arm().items():
        out[f"chains_rearm/{key}"] = value
    for key, value in _smoothing_arm(sigma).items():
        out[f"smoothing/{key}"] = value
    return out


def _differing(got, want):
    got = np.ascontiguousarray(np.asarray(got, dtype=float))
    want = np.ascontiguousarray(np.asarray(want, dtype=float))
    if got.shape != want.shape:
        return max(got.size, want.size)
    return int(np.count_nonzero(got.view(np.uint64) != want.view(np.uint64)))


def _require_pure():
    if _kernels.COMPILED_KERNELS is not None:
        raise SystemExit(
            "deposit_beam_mirror_reference: the corpus is defined against the "
            "pure implementation; unset CABLP_COMPILED_KERNELS"
        )


def capture(path):
    _require_pure()
    corpus = _replay()
    np.savez_compressed(path, **corpus)
    total = sum(np.asarray(v).size for v in corpus.values())
    print(f"captured {len(corpus)} arrays, {total} values -> {path}")
    return 0


def verify(path, impl="live"):
    _require_pure()
    stored = np.load(path)
    original = bd.deposit_beam
    sigma = 50.0
    if impl == "perturbed":
        def perturbed(*args, **kwargs):
            if kwargs.get("mirror_face") is None:
                args = (np.nextafter(float(args[0]), np.inf),) + args[1:]
            return original(*args, **kwargs)

        bd.deposit_beam = perturbed
        sigma = float(np.nextafter(50.0, np.inf))
    try:
        live = _replay(sigma)
    finally:
        bd.deposit_beam = original
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
        if diff and impl == "live":
            print(f"  {key}: {diff} of {stored[key].size} differ")
    for arm, diff in per_arm.items():
        print(f"arm {arm}: {diff} differing")
    print(
        f"deposit_beam_mirror_reference ({impl}): {total_diff} differing of "
        f"{total_values} raw uint64 values"
    )
    if impl == "perturbed":
        ok = all(diff > 0 for diff in per_arm.values())
        print(
            "negative control: every arm differs"
            if ok else "NEGATIVE CONTROL FAILED: an arm did not move"
        )
        return 0 if ok else 1
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
