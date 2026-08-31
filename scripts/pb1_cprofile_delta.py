"""[perf-batch-1] Per-item base-vs-tip comparison of two cProfile dumps.

Reads the two ``profile_sim1d.py --mode cprofile`` dumps this batch took over
the compiled production arm and prints, for every site the batch touches, the
exact CALL COUNT and the tottime/cumtime on each side.

Call counts are exact and load-independent, which is why they are the
authoritative column here; cProfile timings are inflated by instrumentation
and both runs were taken on a shared box, so the time columns rank the work
rather than measure it. The honest wall figure is the un-profiled
``--mode sample`` pair taken on a quiet box.

    python scripts/pb1_cprofile_delta.py BASE.prof TIP.prof
"""

import argparse
import pstats

SITES = [
    "_interp_blend",
    "_interp_coords",
    "he_rates",
    "_array_fingerprint",
    "_beam_smoothing_key",
    "_bad_array_summary",
    "validate_raw_stage",
    "_step_rejection_info",
    "c_log",
    "unpack_state",
    "_resolve_optional_layout",
    "_implicit_neutral_step_two_zone",
    "_two_zone_implicit_matrix",
    "_zero_rhs_state",
    "add_state_rhs",
    "eye",
    "solve",
]


def grab(stats):
    out = {}
    for (_fn, _ln, name), (_cc, nc, tt, ct, _callers) in stats.stats.items():
        if name in SITES:
            row = out.setdefault(name, [0, 0.0, 0.0])
            row[0] += nc
            row[1] += tt
            row[2] += ct
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("tip")
    args = ap.parse_args(argv)

    base = pstats.Stats(args.base)
    tip = pstats.Stats(args.tip)
    print(
        f"TOTAL tottime   base={base.total_tt:10.3f} s  tip={tip.total_tt:10.3f} s"
        f"   {100 * (tip.total_tt - base.total_tt) / base.total_tt:+7.2f}%"
    )
    print(
        f"TOTAL calls     base={base.total_calls:10d}    tip={tip.total_calls:10d}"
        f"     {100 * (tip.total_calls - base.total_calls) / base.total_calls:+7.2f}%"
    )
    print()
    b, t = grab(base), grab(tip)
    header = (
        f"{'site':34s} {'calls base':>11s} {'calls tip':>11s} {'d%':>8s} "
        f"{'tot base':>9s} {'tot tip':>9s} {'d%':>8s}"
    )
    print(header)
    print("-" * len(header))
    for name in sorted(set(b) | set(t)):
        bb = b.get(name, [0, 0.0, 0.0])
        tt = t.get(name, [0, 0.0, 0.0])
        dc = 100 * (tt[0] - bb[0]) / bb[0] if bb[0] else float("nan")
        dt = 100 * (tt[1] - bb[1]) / bb[1] if bb[1] else float("nan")
        print(
            f"{name:34s} {bb[0]:11d} {tt[0]:11d} {dc:+8.1f} "
            f"{bb[1]:9.3f} {tt[1]:9.3f} {dt:+8.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
