"""Quantify the wall channel's move across the accommodation adoption.

The kinetic_dvm arm is EXPECTED to change under
``neutral_kinetic_dvm_accommodation`` 1.0 -> 0.40: that is what the adoption
is for. ``b5cj_bitinert_ab.py --compare`` answers only "did it move", which
here is the uninteresting half. This reads the two captured records and
reports, per DVM tick, WHAT moved -- the wall rows first, since alpha_E is
read at the wall, then everything else ranked by relative move -- so the
report can state the size and the direction of the adoption rather than the
fact of it.

At alpha_E = 1 the reflected share is empty and ``birth_wall_reflected`` is
identically zero; at 0.40 it carries 60 % of every wall encounter. The
particle sum ``birth_wall_accommodated + birth_wall_reflected`` is the
conserved quantity across the swap -- the wall returns the same ATOMS either
way -- while the ENERGY split is the physics that moved.

Usage (from the worktree root)::

    python scripts/dacc_wall_ledger_delta.py BASE.json CANDIDATE.json
"""

import argparse
import json
from pathlib import Path

#: Rows the accommodation coefficient is read at, reported first and in full.
WALL_ROWS = (
    "birth_wall_accommodated",
    "birth_wall_reflected",
    "loss_wall",
    "energy.birth_wall_accommodated",
    "energy.birth_wall_reflected",
    "energy.loss_wall",
    "energy.net_surface_wall",
)


def _rel(a, b):
    """Relative move of ``b`` from ``a``; ``None`` where undefined."""
    if a == b:
        return 0.0
    if a == 0.0:
        return None
    return (b - a) / abs(a)


def _fmt(value):
    return "None" if value is None else f"{value:+.4e}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("base")
    ap.add_argument("candidate")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args(argv)

    a = json.loads(Path(args.base).read_text())
    b = json.loads(Path(args.candidate).read_text())

    print("DVM wall-channel ledger delta across the accommodation adoption")
    print("base      alpha_E = 1.0   (every wall return re-emitted at T_wall)")
    print("candidate alpha_E = 0.40  (60 % returned at the incident energy)")
    print("=" * 78)
    print(f"state digest   base {a['digest_sha256']}")
    print(f"               cand {b['digest_sha256']}")
    print(
        "               "
        f"{'DIFFER (expected -- this is the adoption)' if a['digest_sha256'] != b['digest_sha256'] else 'IDENTICAL (UNEXPECTED)'}"
    )
    print(f"steps {a['steps_taken']} vs {b['steps_taken']};  "
          f"ticks {a['dvm_ticks']} vs {b['dvm_ticks']};  "
          f"t_end {a['time_s']:.12e} vs {b['time_s']:.12e}")
    print("=" * 78)

    for i, (ra, rb) in enumerate(
        zip(a["dvm_ledgers"], b["dvm_ledgers"]), start=1
    ):
        print(f"--- DVM tick {i} : wall rows ---")
        for key in WALL_ROWS:
            va, vb = ra.get(key), rb.get(key)
            if va is None or vb is None:
                print(f"  {key:34s} MISSING on one side")
                continue
            print(
                f"  {key:34s} {va:+.6e} -> {vb:+.6e}   "
                f"rel {_fmt(_rel(va, vb))}"
            )
        # The atoms the wall returns are conserved across the split; only
        # their energy is re-partitioned. Stated so the reader can see the
        # particle book is untouched by the coefficient.
        for tag, rows in (("base", ra), ("cand", rb)):
            tot = (
                rows["birth_wall_accommodated"] + rows["birth_wall_reflected"]
            )
            print(f"  [{tag}] wall birth accommodated+reflected = {tot:.6e}")
        print()

    print("--- largest relative moves anywhere in tick 1 ---")
    ra, rb = a["dvm_ledgers"][0], b["dvm_ledgers"][0]
    ranked = []
    for key in sorted(set(ra) & set(rb)):
        r = _rel(ra[key], rb[key])
        if r is not None:
            ranked.append((abs(r), key, ra[key], rb[key], r))
    ranked.sort(reverse=True)
    for _, key, va, vb, r in ranked[: args.top]:
        print(f"  {key:34s} {va:+.6e} -> {vb:+.6e}   rel {r:+.4e}")

    print()
    print("--- cathode surface energy ledger [J] ---")
    la, lb = a["cathode_energy_ledger_J"], b["cathode_energy_ledger_J"]
    for key in sorted(set(la) | set(lb)):
        va, vb = la.get(key), lb.get(key)
        print(f"  {key:12s} {va!r:26s} -> {vb!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
