"""Bit-inertness A/B for the DVM arm itself, at the shipped wall reflection.

``dvm_b1_bitinert_ab.py`` proves the ``moment`` and ``kinetic`` arms do not
move, which is the statement about the plumbing and about the golden. It
cannot say anything about ``neutral_model = "kinetic_dvm"``, because that arm
is not one of its arms -- and the wall-reflection member rewrote lines
INSIDE that arm's update: the non-accommodated wall share moved out of the
two annulus branches into ``TransientDVM._wall_return_counts``, and the
energy row that books it grew a branch.

So this walks the transient engine directly at the shipped
``wall_reflection = "specular"`` -- never naming the member, so the same
file runs unchanged at the base commit -- and digests the accepted state
after EVERY tick at raw ``uint64``. Both annulus treatments are walked,
because the jump arm places the same array through a different route, and a
live plasma is supplied so the charge-exchange and elastic channels are
armed rather than dormant.

Usage (from the checkout root, PYTHONPATH set to it)::

    python scripts/k2_dvm_wall_reflection_specular_ab.py \
        --out scripts/k2_dvm_specular_ab_<label>.json
    python scripts/k2_dvm_wall_reflection_specular_ab.py --compare A.json B.json
"""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from cablp.solvers._sim1d.physics.kinetic_dvm import TransientDVM

#: Ticks each arm walks. The digest is running, so a divergence at any tick
#: reaches the final value.
DEFAULT_TICKS = 40

#: Accommodation the arms run at. Interior, so BOTH the accommodated and the
#: non-accommodated wall shares are live -- at 0 or 1 one of the two is
#: switched off and the statement would be half of itself.
ALPHA = 0.40


def uniform_tube(nz, length_cm=1600.0, Rp=15.0, Rm=50.0):
    """Return the strictly-uniform coaxial tube the gate suite uses."""
    dz = np.full(nz, length_cm / nz)
    Rp_cm = np.full(nz, Rp)
    Rm_cm = np.full(nz, Rm)
    return SimpleNamespace(
        cells=nz,
        length_cm=dz,
        Rp_cm=Rp_cm,
        Rm_cm=Rm_cm,
        plasma_volume_cm3=np.pi * Rp_cm**2 * dz,
        neutral_volume_cm3=np.pi * Rm_cm**2 * dz,
    )


def capture_arm(annulus_flights, exchange_model, ticks, nz=12):
    """Return the running digest of one arm's tick sequence."""
    dvm = TransientDVM(
        geometry=uniform_tube(nz),
        nvz=16,
        nvp=6,
        accommodation=ALPHA,
        exchange_model=exchange_model,
        annulus_flights=annulus_flights,
        mesh_face=nz // 2,
        transparency=0.642,
        s_L=0.3,
        s_R=0.3,
    )
    dvm.seed_from_density(np.full(nz, 1.0e13), np.full(nz, 1.0e13))
    plasma = {
        "n_i": np.linspace(1.0e12, 5.0e12, nz),
        "Ti_eV": np.linspace(0.5, 3.0, nz),
        "u_i": np.linspace(-1.0e5, 1.0e5, nz),
        "nu_ion": np.full(nz, 1.0e3),
    }
    puff = np.zeros(nz)
    puff[3] = 3.0e17
    running = hashlib.sha256()
    ledger = hashlib.sha256()
    for _ in range(ticks):
        led = dvm.update(
            2.5e-5,
            sources={"puff": puff / 2.5e-5},
            T_s_K=1910.0,
            **plasma,
        )
        for field in (dvm.f_c, dvm.f_a, dvm.pend_L_c, dvm.pend_R_c,
                      dvm.pend_L_a, dvm.pend_R_a):
            running.update(np.asarray(field, dtype=np.float64).tobytes())
        for key in sorted(k for k in led if k != "energy"):
            ledger.update(
                np.asarray(led[key], dtype=np.float64).tobytes()
            )
        for key in sorted(led["energy"]):
            ledger.update(
                np.asarray(led["energy"][key], dtype=np.float64).tobytes()
            )
    return {
        "state_sha256": running.hexdigest(),
        "ledger_sha256": ledger.hexdigest(),
        "inventory": float(dvm.total_inventory()),
        "energy": float(dvm.total_energy()),
        "updates": int(dvm.updates),
    }


def capture(ticks):
    """Return every arm's digest, keyed by annulus treatment and closure."""
    out = {"ticks": int(ticks), "alpha": ALPHA, "arms": {}}
    for flights in ("rates", "bounded_chord"):
        for exchange in ("cauchy_chord", "geometric"):
            out["arms"][f"{flights}/{exchange}"] = capture_arm(
                flights, exchange, ticks
            )
    return out


def compare(a_path, b_path):
    """Print whether two captures are bit-identical; 0 when they are."""
    a = json.loads(Path(a_path).read_text())
    b = json.loads(Path(b_path).read_text())
    same = a["ticks"] == b["ticks"] and a["alpha"] == b["alpha"]
    print(f"ticks {a['ticks']} | {b['ticks']}, alpha {a['alpha']} | "
          f"{b['alpha']}")
    for name in sorted(set(a["arms"]) | set(b["arms"])):
        left = a["arms"].get(name, {})
        right = b["arms"].get(name, {})
        for key in ("state_sha256", "ledger_sha256", "inventory", "energy",
                    "updates"):
            match = left.get(key) == right.get(key)
            same = same and match
            print(f"{name:26s} {key:14s} {'==' if match else '!='}  "
                  f"{left.get(key)}  |  {right.get(key)}")
    print("BIT-IDENTICAL" if same else "DIVERGED")
    return 0 if same else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticks", type=int, default=DEFAULT_TICKS)
    parser.add_argument("--out")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = parser.parse_args(argv)
    if args.compare:
        return compare(*args.compare)
    if not args.out:
        parser.error("--out is required unless --compare is given")
    record = capture(args.ticks)
    Path(args.out).write_text(json.dumps(record, indent=2, sort_keys=True))
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
