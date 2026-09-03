"""K2: the PRE-REGISTERED acceptance gate for the geometric zone-exchange rates.

Runs the committed E2 comparison (``neutral_arch_e2_compare``) at REDUCED Monte
Carlo statistics, twice: once with the shipped ``cauchy_chord`` zone-exchange
closure and once with the derived ``geometric`` one, against the SAME reduced
reference. Recording the reduced baseline first is the point -- the full-run
+184 % headline was measured at 16 x 500,000 histories, and comparing a reduced
derived arm to it would confound the closure change with the statistics change.

The gate, written before the derived rates were run (a miss is a NULL RESULT to
be reported, never a licence to adjust the rates):

  (a) the worst matched-time mid-machine annulus-density (``n_ann``) relative
      deviation shrinks by AT LEAST 3x against the reduced-statistics baseline
      of the same quantity;
  (b) no (quantity, region, time-bin) that was WITHIN its band under the
      baseline leaves it under the derived rates, where "within its band" is
      ``|dev| <= 1 sigma_MC`` OR ``|dev %| <= the DVM's own dt band for that
      quantity`` -- the two-sided noise statement the E2 summary already
      reports, with the dt band measured here on the BASELINE arm at dt and
      dt/2, exactly as the E2 driver measures it.

Nothing in this script is tunable against the result. It imports the committed
driver and subclasses the engine; it does not modify either.

Usage:

    PYTHONPATH=<checkout>/cablp python scripts/verify/k2_dvm_exchange_acceptance.py \
        --run <saved nx=240 run>.h5 \
        --out ~/bapsf/artifacts/<event>/k2_dvm_exchange_acceptance.txt

``--run`` and ``--out`` are required: run artifacts are read from and written
under the artifacts root, never beside this script.
"""

import argparse
import os
import platform
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

# scripts/ sibling imports: the seven purpose subdirectories on sys.path.
import sys as _sys
from pathlib import Path as _Path
for _sub in ("atomic", "gates", "kinetic", "run", "score", "stance",
             "verify"):
    _dir = str(_Path(__file__).resolve().parents[1] / _sub)
    if _dir not in _sys.path:
        _sys.path.insert(0, _dir)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import neutral_arch_e2_compare as e2  # noqa: E402
from mc_neutrals import KB, M_HE, T_WALL_K, load_background  # noqa: E402

from cablp.solvers._sim1d.physics.kinetic_dvm import TransientDVM  # noqa: E402

# Quantities the gate reads. wstep_* is reference-only (the DVM has no such
# surface) and is excluded here as it is in the E2 summary's own reading.
KEYS = ("n_col", "n_ann", "p_col", "p_ann", "e_col", "e_ann",
        "exch_ca", "exch_ac", "wrad_inc", "wrad_ret",
        "wend_inc", "wend_ret")
MID = "mid-machine 500-1000"
SHRINK_REQUIRED = 3.0


class GeometricDVM(TransientDVM):
    """The engine, pinned to the derived zone-exchange closure."""

    def __init__(self, **kwargs):
        kwargs["exchange_model"] = "geometric"
        super().__init__(**kwargs)


def run_arm(shared, args, exchange, dt):
    """Run the committed E2 DVM arm under one zone-exchange closure."""
    original = e2.TransientDVM
    if exchange == "geometric":
        e2.TransientDVM = GeometricDVM
    try:
        diag, obj = e2.run_dvm(
            shared, dt, args.nvz, args.nvp, args.accommodation,
            args.elastic_model,
        )
    finally:
        e2.TransientDVM = original
    got = getattr(obj, "exchange_model", None)
    if got != exchange:
        raise RuntimeError(
            f"the arm was asked for exchange_model={exchange!r} but the "
            f"engine reports {got!r} -- the selector did not take"
        )
    return diag, obj


def dt_bands(shared, dvm_a, dvm_h):
    """The DVM's own dt band per quantity, exactly as the E2 driver forms it."""
    band = {}
    for key in ("n_col", "n_ann", "p_col", "p_ann", "e_col", "e_ann",
                "exch_ca", "exch_ac", "wrad_inc", "wrad_ret"):
        for lab, m in e2.region_masks(shared["z_cm"]):
            if not m.any():
                continue
            w = (shared["V_col"] if key.endswith("col")
                 else shared["V_ann"] if key.endswith("ann") else None)
            if w is not None and w[m].sum() <= 0.0:
                continue
            a = e2.agg(dvm_a[key], m, w)
            b = e2.agg(dvm_h[key], m, w)
            r = np.abs(e2.rel_dev(a, b)) * 100.0
            band[key] = max(band.get(key, 0.0), float(np.nanmax(r)))
    for j in (0, 1):
        for key in ("wend_inc", "wend_ret"):
            r = np.abs(e2.rel_dev(dvm_a[key][:, j], dvm_h[key][:, j])) * 100.0
            band[key] = max(band.get(key, 0.0), float(np.nanmax(r)))
    return band


def cell_table(rows, band):
    """Flatten the comparison rows into a ``(region, key, bin)`` dict.

    Each entry is ``(dev %, dev/sigma, within-band)``.
    """
    out = {}
    for lab, key, _kind, d, r, e, _ref in rows:
        for b in range(np.asarray(d).size):
            s = e2.dev_sigma(d[b], r[b], e[b])
            p = e2.rel_dev(d[b], r[b])
            pct = float(p) * 100.0 if np.isfinite(p) else np.nan
            sig = float(s) if np.isfinite(s) else np.nan
            within = (
                (np.isfinite(sig) and abs(sig) <= 1.0)
                or (np.isfinite(pct) and abs(pct) <= band.get(key, 0.0))
            )
            out[(lab, key, b)] = (pct, sig, bool(within))
    return out


def worst_pct(table, key, region=None):
    """Worst |dev %| for a quantity, optionally inside one region."""
    best = (0.0, None)
    for (lab, k, b), (pct, _s, _w) in table.items():
        if k != key or (region is not None and lab != region):
            continue
        if np.isfinite(pct) and abs(pct) > abs(best[0]):
            best = (pct, (lab, b))
    return best


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Pre-registered acceptance gate for the geometric DVM "
                    "zone-exchange rates, on a reduced E2 rerun."
    )
    ap.add_argument(
        "--run",
        required=True,
        help="saved nx=240 production background (read in place, geometry "
             "only); a saved sim1d run under the artifacts root, e.g. "
             "~/bapsf/artifacts/<event>/....h5",
    )
    ap.add_argument("--window", nargs=2, type=float, default=(5.0, 19.5))
    ap.add_argument("--nvz", type=int, default=48)
    ap.add_argument("--nvp", type=int, default=12)
    ap.add_argument("--dvm-dt", type=float, default=2.5e-5)
    ap.add_argument("--t-end-ms", type=float, default=6.0)
    ap.add_argument("--t-switch-ms", type=float, default=3.0)
    ap.add_argument("--bin-ms", type=float, default=0.5)
    ap.add_argument("--particles", type=int, default=500_000)
    ap.add_argument("--batches", type=int, default=8,
                    help="REDUCED from the full run's 16; both arms are "
                         "scored against this one reference")
    ap.add_argument("--seed", type=int, default=20260805)
    ap.add_argument("--accommodation", type=float, default=1.0)
    ap.add_argument("--elastic-model", default="phelps_iso")
    ap.add_argument("--progress", type=int, default=0)
    ap.add_argument(
        "--out",
        required=True,
        help="acceptance transcript path; write under the artifacts root, "
             "e.g. ~/bapsf/artifacts/<event>/k2_dvm_exchange_acceptance.txt",
    )
    args = ap.parse_args(argv)
    args.seed_state = True

    cmdline = " ".join(
        [f"PYTHONPATH={os.environ.get('PYTHONPATH', '')}",
         sys.executable, str(Path(sys.argv[0]))] + list(sys.argv[1:])
    )
    t_all = time.perf_counter()

    print(f"K2 acceptance: loading {args.run}", flush=True)
    bg = load_background(args.run, tuple(args.window))
    shared = e2.build_shared(bg, args)

    print("DVM arm, cauchy_chord (baseline) ...", flush=True)
    dvm_base, obj_base = run_arm(shared, args, "cauchy_chord", args.dvm_dt)
    dvm_vmax = float(obj_base.g.vz.max())
    print("DVM arm, cauchy_chord at dt/2 (the dt band) ...", flush=True)
    dvm_half, _ = run_arm(shared, args, "cauchy_chord", 0.5 * args.dvm_dt)
    print("DVM arm, geometric (derived) ...", flush=True)
    dvm_geo, obj_geo = run_arm(shared, args, "geometric", args.dvm_dt)

    print(
        f"MC reference, true kinematics, REDUCED: {args.batches} batches x "
        f"{args.particles} histories ...", flush=True,
    )
    mc_m, mc_s, mc_meta = e2.run_mc(
        shared, args, "kinetic", args.accommodation, args.elastic_model,
        dvm_vmax,
    )

    ref_label = "transient full-particle TPMC, TRUE two-body kinematics"
    rows_base = e2.compare_rows(shared, dvm_base, mc_m, mc_s, ref_label)
    rows_geo = e2.compare_rows(shared, dvm_geo, mc_m, mc_s, ref_label)
    band = dt_bands(shared, dvm_base, dvm_half)
    tab_base = cell_table(rows_base, band)
    tab_geo = cell_table(rows_geo, band)

    lines = build_report(
        args, cmdline, shared, bg, obj_base, obj_geo, band, tab_base, tab_geo,
        mc_meta, time.perf_counter() - t_all,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}", flush=True)
    return 0


def build_report(args, cmdline, shared, bg, obj_base, obj_geo, band,
                 tab_base, tab_geo, mc_meta, wall_s):
    L = []
    W = 108

    def hdr(t):
        L.append("")
        L.append(t)
        L.append("-" * W)

    L.append("K2 -- ACCEPTANCE GATE FOR THE GEOMETRIC ZONE-EXCHANGE RATES")
    L.append("=" * W)
    L.append("")
    L.append(
        "Reduced-statistics rerun of the committed E2 comparison, with the "
        "shipped cauchy_chord closure and the derived geometric closure "
        "scored against the SAME reduced reference at identical elapsed "
        "times. The gate below was written before the derived rates were run."
    )
    L.append("")
    L.append(f"background      : {args.run}, window {tuple(args.window)} ms")
    L.append(
        f"schedule        : sources ON 0-{args.t_switch_ms:.2f} ms, OFF to "
        f"{args.t_end_ms:.2f} ms; {int(round(args.t_end_ms / args.bin_ms))} "
        f"report bins of {args.bin_ms:.2f} ms"
    )
    L.append(
        f"DVM             : nz={shared['nz']}, grid {args.nvz}x{args.nvp}, "
        f"neutral clock dt={args.dvm_dt:g} s (dt band from the same arm at "
        f"dt/2)"
    )
    L.append(
        f"MC reference    : {args.batches} independent batches x "
        f"{args.particles} histories, base seed {args.seed} "
        f"(REDUCED from the full run's 16 x 500000)"
    )
    L.append(
        f"closures        : baseline {obj_base.exchange_model!r}, derived "
        f"{obj_geo.exchange_model!r} (probed on the built engine, not assumed)"
    )
    L.append(
        f"machine         : {platform.platform()}; python "
        f"{platform.python_version()}, numpy {np.__version__}; total wall "
        f"{wall_s:.1f} s"
    )
    L.append("")
    L.append("Full command (reruns end to end):")
    L.append("")
    L.append(f"    {cmdline}")

    hdr("MC INTEGRITY (a nonzero majorant-violation count invalidates the run)")
    L.append("")
    viol = sum(m["violations"] for m in mc_meta)
    seg = sum(m["segments"] for m in mc_meta)
    closure = [
        (m["lost_ion"] + m["lost_pump"] + m["resident"] + m["stuck"])
        / max(m["launched_atoms"], 1e-300)
        for m in mc_meta
    ]
    L.append(
        f"  {seg / 1e6:.1f}e6 flight segments over {len(mc_meta)} batches; "
        f"majorant violations {viol}; worst ledger closure ratio "
        f"{min(closure):.10f} / {max(closure):.10f}"
    )

    hdr("GATE (a): worst matched-time annulus-density deviation")
    L.append("")
    L.append(
        "The full 16-batch E2 run reported n_ann worst +184.19 % at "
        "(mid-machine 500-1000, bin 11). The reduced-statistics BASELINE below "
        "is the number the derived arm must beat by 3x -- the full-run value "
        "is quoted only for context and is NOT the denominator of the gate."
    )
    L.append("")
    L.append(
        f"  {'statistic':>44s} {'baseline':>26s} {'derived':>26s} {'shrink':>9s}"
    )
    verdict_a = True
    for label, region in (
        (f"n_ann worst |dev %| in {MID}", MID),
        ("n_ann worst |dev %| over all regions", None),
    ):
        b_pct, b_at = worst_pct(tab_base, "n_ann", region)
        g_pct, g_at = worst_pct(tab_geo, "n_ann", region)
        shrink = abs(b_pct) / max(abs(g_pct), 1e-300)
        if region == MID and shrink < SHRINK_REQUIRED:
            verdict_a = False
        L.append(
            f"  {label:>44s} "
            f"{f'{b_pct:+.2f}% (bin {b_at[1] if b_at else -1})':>26s} "
            f"{f'{g_pct:+.2f}% (bin {g_at[1] if g_at else -1})':>26s} "
            f"{shrink:9.3f}x"
        )
    L.append("")
    L.append(
        f"  GATE (a) [>= {SHRINK_REQUIRED:.0f}x shrink of the mid-machine "
        f"n_ann worst deviation]: "
        f"{'PASS' if verdict_a else 'MISS (null result)'}"
    )

    hdr("GATE (b): nothing that was inside its band leaves it")
    L.append("")
    L.append(
        "'Inside its band' = |dev| <= 1 sigma_MC OR |dev %| <= the DVM's own "
        "dt band for that quantity (measured on the baseline arm at dt vs "
        "dt/2). The dt bands used, in %:"
    )
    L.append("")
    L.append("    " + ", ".join(f"{k} {band.get(k, 0.0):.1f}" for k in KEYS))
    L.append("")
    regressions = [
        (lab, key, b, tab_base[(lab, key, b)], tab_geo[(lab, key, b)])
        for (lab, key, b) in sorted(tab_base)
        if key in KEYS
        and (lab, key, b) in tab_geo
        and tab_base[(lab, key, b)][2]
        and not tab_geo[(lab, key, b)][2]
    ]
    inside_base = sum(
        1 for k, v in tab_base.items() if k[1] in KEYS and v[2]
    )
    inside_geo = sum(
        1 for k, v in tab_geo.items() if k[1] in KEYS and v[2]
    )
    total = sum(1 for k in tab_base if k[1] in KEYS)
    L.append(
        f"  cells inside the band: baseline {inside_base}/{total}, derived "
        f"{inside_geo}/{total}; regressions {len(regressions)}"
    )
    if regressions:
        L.append("")
        L.append(
            f"  {'region':>24s} {'quantity':>10s} {'bin':>4s} "
            f"{'base dev %':>12s} {'base dev/sig':>13s} "
            f"{'geo dev %':>12s} {'geo dev/sig':>13s}"
        )
        for lab, key, b, (bp, bs, _), (gp, gs, _) in regressions:
            L.append(
                f"  {lab:>24s} {key:>10s} {b:4d} {bp:12.2f} {bs:13.2f} "
                f"{gp:12.2f} {gs:13.2f}"
            )
    L.append("")
    verdict_b = not regressions
    L.append(
        f"  GATE (b) [no in-band quantity leaves its band]: "
        f"{'PASS' if verdict_b else 'MISS (null result)'}"
    )

    hdr("WORST MATCHED-TIME DEVIATION, BOTH ARMS, SAME REFERENCE")
    L.append("")
    L.append(
        f"  {'quantity':>10s} {'dt band %':>10s} "
        f"{'baseline worst dev %':>34s} {'derived worst dev %':>34s} "
        f"{'ratio':>8s}"
    )
    for key in KEYS:
        b_pct, b_at = worst_pct(tab_base, key)
        g_pct, g_at = worst_pct(tab_geo, key)
        ratio = abs(g_pct) / max(abs(b_pct), 1e-300)
        L.append(
            f"  {key:>10s} {band.get(key, 0.0):10.1f} "
            f"{f'{b_pct:+.2f} ({b_at[0]}, bin {b_at[1]})' if b_at else 'n/a':>34s} "
            f"{f'{g_pct:+.2f} ({g_at[0]}, bin {g_at[1]})' if g_at else 'n/a':>34s} "
            f"{ratio:8.3f}"
        )

    hdr("PER-REGION n_ann AND exch_ac, EVERY TIME BIN (the exchange channel "
        "the closure moves)")
    L.append("")
    for key in ("n_ann", "exch_ac", "exch_ca", "wrad_inc"):
        L.append("")
        L.append(f"  {key}   (dev % vs the same reduced reference)")
        L.append(
            f"  {'region':>24s} " + " ".join(f"{b:>8d}" for b in range(
                int(round(args.t_end_ms / args.bin_ms))))
        )
        for lab, _m in e2.region_masks(shared["z_cm"]):
            cells = [
                (b, tab_base.get((lab, key, b)), tab_geo.get((lab, key, b)))
                for b in range(int(round(args.t_end_ms / args.bin_ms)))
            ]
            if not any(c[1] for c in cells):
                continue
            L.append(
                f"  {lab + ' base':>24s} "
                + " ".join(
                    f"{c[1][0]:8.1f}" if c[1] else f"{'-':>8s}" for c in cells
                )
            )
            L.append(
                f"  {lab + ' geom':>24s} "
                + " ".join(
                    f"{c[2][0]:8.1f}" if c[2] else f"{'-':>8s}" for c in cells
                )
            )

    hdr("VERDICT")
    L.append("")
    L.append(
        f"  (a) {'PASS' if verdict_a else 'MISS'}    "
        f"(b) {'PASS' if verdict_b else 'MISS'}    "
        f"=> acceptance {'PASSES' if verdict_a and verdict_b else 'is a NULL RESULT'}"
    )
    L.append("")
    L.append(
        "A miss is reported as measured. No rate, floor or threshold in this "
        "comparison was adjusted after seeing it, and the derived expression "
        "carries no free constant that could be."
    )
    L.append("")
    return L


if __name__ == "__main__":
    raise SystemExit(main())
