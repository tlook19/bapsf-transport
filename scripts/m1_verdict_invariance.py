"""Apply the M1 verdict-invariance gate (registered 2026-08-31) to three runs.

The gate asks whether an M1 retest verdict survives a halving of the neutral
cadence. It scores three saved runs through ``compare_sim1d_es1`` -- a FLUID
comparator, a DVM arm at neutral cadence h, and the SAME DVM arm at h/2 -- and
reports, per scored row and then as one verdict:

Cadence condition, per scored row, with O the row's scored observable::

    Delta_cad = |O(h) - O(h/2)|  <=  min(|Delta_AB|, sigma_tot) / 3
    Delta_AB  =  O(h) - O(fluid)

O is the row's window-mean model value (``compare``'s ``model``), so
``Delta_AB``, ``Delta_cad``, ``sigma_tot`` and the threshold all carry the
row's own units and are directly comparable. ``sigma_tot`` is the row's
window-mean ``sqrt(SEM^2 + sigma_sys^2)`` -- the scorer's own error model (SEM
(+) sweep systematics), measurement-side, depending on the run only through
which samples its time coverage admits.

EXACT-ZERO CORNER. A row on which the DVM arm reproduces the fluid comparator
exactly has ``Delta_AB = 0``, hence ``threshold = min(0, sigma_tot/3) = 0``,
and the single ``Delta_cad <= threshold`` rule then passes it only when
``Delta_cad`` is exactly 0.0. That is the intended reading, not an edge case to
be softened: the gate's premise is that the cadence must not move a row by an
appreciable fraction of the A/B separation it is asked to resolve, and where
there is no separation there is no room to move. Such rows are marked ``[0]``.

UNGATED ROWS. A row whose ``sigma_tot`` is absent, non-finite or non-positive
carries no usable uncertainty, so the threshold cannot be formed. It is
reported UNGATED and counted separately -- never silently passed, and never
counted as a failure either.

Verdict quantity: the mean mid-port density ratio model/measured, over ports
11, 21 and 29. PORT -> ROW MAPPING: the overlay carries ``port`` = [11, 21, 29,
41, 50] paired positionally with ``z_cm`` = [470.05, 789.55, 1045.15, 1428.55,
1716.10] cm, and ``compare`` emits one row per (field, port) carrying that port
number as a string; the three mid-ports are therefore the ``field == "n"`` rows
whose ``port`` is "11", "21" or "29", at z = 470.05, 789.55 and 1045.15 cm. The
ratio itself is the row's ``ratio``, the window mean of model/measured.

Bins, evaluated per cadence against the fluid comparator:

- RESOLVED -- mid-port ratio >= 0.85;
- IMPROVED -- otherwise, mid-port ratio at least +0.10 above the fluid's AND
  the Isat rows not degrading;
- NULL -- otherwise.

DEGRADE is defined from the scorer's Isat rows: the mean over the five
``field == "Isat"`` rows of ``sigma`` (the row's mean ``|dev|/sigma_tot``) must
not INCREASE relative to the fluid comparator. The comparison is exact -- any
increase, however small, degrades -- because the quantity is already a mean
over five rows and a tolerance here would be an unregistered free parameter.

The overall verdict is one of::

    M1 VERDICT: <bin> (cadence-invariant)
    M1 VERDICT: CADENCE-LIMITED NULL (bin h=<..> h/2=<..>)
    FAIL: <n> rows exceed the cadence threshold

reported in that order of precedence: a row-level cadence failure means the
gate's own premise is unmet and no verdict is read off it, a bin disagreement
means the verdict exists but does not survive the cadence, and only a run of
the table with neither yields a verdict.

This module reads only the five artifact fields ``compare`` touches (``time``,
``n``, ``Te``, ``phase`` and ``geometry/z_cm``) rather than loading whole
trajectories: a saved run's bulk is its ``rhs_terms`` and diagnostics, which no
scoring stage reads. Rows are never re-implemented here -- every number in the
table comes from ``compare_sim1d_es1.compare``.

Usage::

    python scripts/m1_verdict_invariance.py --fluid F.h5 \\
        --dvm-h H.h5 --dvm-h2 H2.h5 --es 1
    python scripts/m1_verdict_invariance.py --self-test
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from compare_sim1d_es1 import OVERLAY, compare  # noqa: E402

# Imported rather than re-spelled: these two decide how a stored dataset
# becomes an array, and a second copy of that decision here could drift from
# the writer's without either side failing. _read_dataset dispatches on the
# stored dtype so the variable-length `phase` strings decode exactly as
# load_result_hdf5 decodes them.
from cablp.solvers._sim1d.results.io import (  # noqa: E402
    RESULT_VERSION,
    _decode_string,
    _read_dataset,
)

# The fields `compare` reads off a result, and the only ones this loads.
SCORING_FIELDS = ("time", "n", "Te", "phase")

MID_PORTS = ("11", "21", "29")
CADENCE_DIVISOR = 3.0
RESOLVED_RATIO = 0.85
IMPROVED_DELTA = 0.10


def load_scoring_fields(path):
    """Return the minimal result namespace ``compare`` reads from an artifact.

    Loads ``time``, ``n``, ``Te``, ``phase`` and ``geometry/z_cm`` and nothing
    else. Raises ``ValueError`` on a file whose format attribute is not the
    supported result version, matching ``load_result_hdf5``'s refusal rather
    than reading unknown layout by luck.
    """
    path = Path(path)
    with h5py.File(path, "r") as h5:
        file_format = _decode_string(h5.attrs.get("format", ""))
        if file_format != RESULT_VERSION:
            raise ValueError(
                f"unsupported sim1d result format {file_format!r} in {path}; "
                f"expected {RESULT_VERSION!r}"
            )
        fields = {name: _read_dataset(h5[name]) for name in SCORING_FIELDS}
        fields["z_cm"] = _read_dataset(h5["geometry/z_cm"])
    return SimpleNamespace(**fields)


def score_run(path, overlay):
    """Return ``compare``'s stage (ii) rows for one saved run."""
    return compare(load_scoring_fields(path), None, overlay)


def _row_key(row):
    return (row["field"], row["port"])


def gate_table(rows_fluid, rows_h, rows_h2):
    """Return one gate record per scored row, in the h arm's row order.

    Raises ``ValueError`` when the three runs do not score the identical set of
    rows: a row present at one cadence and absent at the other is a coverage or
    port-set difference, and pairing across it would compare two different
    questions.
    """
    by_fluid = {_row_key(r): r for r in rows_fluid}
    by_h = {_row_key(r): r for r in rows_h}
    by_h2 = {_row_key(r): r for r in rows_h2}
    if not (set(by_fluid) == set(by_h) == set(by_h2)):
        missing = sorted(
            (set(by_fluid) | set(by_h) | set(by_h2))
            - (set(by_fluid) & set(by_h) & set(by_h2))
        )
        raise ValueError(
            "the three runs did not score the same rows; rows present in some "
            f"and not others: {missing}. A row scored at one cadence and not "
            "the other cannot be gated for cadence invariance"
        )

    records = []
    for row in rows_h:
        key = _row_key(row)
        o_fluid = float(by_fluid[key]["model"])
        o_h = float(by_h[key]["model"])
        o_h2 = float(by_h2[key]["model"])
        sigma_tot = by_h[key].get("sigma_tot")
        sigma_tot = np.nan if sigma_tot is None else float(sigma_tot)
        delta_ab = o_h - o_fluid
        delta_cad = abs(o_h - o_h2)
        gated = bool(np.isfinite(sigma_tot) and sigma_tot > 0.0)
        if gated:
            threshold = min(abs(delta_ab), sigma_tot) / CADENCE_DIVISOR
            status = "PASS" if delta_cad <= threshold else "FAIL"
        else:
            threshold = np.nan
            status = "UNGATED"
        records.append(
            {
                "field": key[0],
                "port": key[1],
                "o_fluid": o_fluid,
                "o_h": o_h,
                "o_h2": o_h2,
                "sigma_tot": sigma_tot,
                "delta_ab": delta_ab,
                "delta_cad": delta_cad,
                "threshold": threshold,
                "exact_zero": bool(delta_ab == 0.0),
                "status": status,
            }
        )
    return records


def mid_port_n_ratio(rows):
    """Return the mean model/measured density ratio over ports 11, 21 and 29."""
    sub = [r for r in rows if r["field"] == "n" and r["port"] in MID_PORTS]
    if len(sub) != len(MID_PORTS):
        raise ValueError(
            f"expected one 'n' row per mid-port {MID_PORTS}, found "
            f"{sorted(r['port'] for r in sub)}; the verdict quantity is "
            "undefined without all three"
        )
    return float(np.mean([r["ratio"] for r in sub]))


def isat_deviation(rows):
    """Return the mean ``|dev|/sigma_tot`` over the scored Isat rows."""
    sub = [r for r in rows if r["field"] == "Isat"]
    if not sub:
        raise ValueError(
            "no 'Isat' rows were scored, so the degrade criterion that the "
            "IMPROVED bin depends on cannot be evaluated"
        )
    return float(np.mean([r["sigma"] for r in sub]))


def verdict_bin(ratio, ratio_fluid, isat, isat_fluid):
    """Return the verdict bin for one cadence against the fluid comparator."""
    if ratio >= RESOLVED_RATIO:
        return "RESOLVED"
    if (ratio - ratio_fluid) >= IMPROVED_DELTA and isat <= isat_fluid:
        return "IMPROVED"
    return "NULL"


def verdict_block(rows_fluid, rows_h, rows_h2):
    """Return the G-verdict quantities for the fluid and both cadences."""
    ratio_fluid = mid_port_n_ratio(rows_fluid)
    isat_fluid = isat_deviation(rows_fluid)
    ratio_h = mid_port_n_ratio(rows_h)
    ratio_h2 = mid_port_n_ratio(rows_h2)
    isat_h = isat_deviation(rows_h)
    isat_h2 = isat_deviation(rows_h2)
    return {
        "ratio_fluid": ratio_fluid,
        "ratio_h": ratio_h,
        "ratio_h2": ratio_h2,
        "isat_fluid": isat_fluid,
        "isat_h": isat_h,
        "isat_h2": isat_h2,
        "isat_degrades_h": bool(isat_h > isat_fluid),
        "isat_degrades_h2": bool(isat_h2 > isat_fluid),
        "bin_h": verdict_bin(ratio_h, ratio_fluid, isat_h, isat_fluid),
        "bin_h2": verdict_bin(ratio_h2, ratio_fluid, isat_h2, isat_fluid),
    }


def overall_verdict(records, block):
    """Return the one-line overall verdict for a gate table and verdict block."""
    fails = [r for r in records if r["status"] == "FAIL"]
    if fails:
        return f"FAIL: {len(fails)} rows exceed the cadence threshold"
    bin_h, bin_h2 = block["bin_h"], block["bin_h2"]
    if bin_h != bin_h2:
        return (
            f"M1 VERDICT: CADENCE-LIMITED NULL (bin h={bin_h} h/2={bin_h2})"
        )
    return f"M1 VERDICT: {bin_h} (cadence-invariant)"


def _report_gate(records):
    print("\n--- cadence gate: per scored row ---")
    print(
        "  (O is the row's window-mean model value; sigma_tot is the row's "
        "window-mean"
    )
    print("   sqrt(SEM^2 + sigma_sys^2).  threshold = min(|Delta_AB|, sigma_tot)/3.")
    print("   '[0]' marks Delta_AB = 0, where the threshold is exactly 0 and only")
    print("   Delta_cad = 0 passes.  UNGATED rows carry no usable sigma_tot.)")
    header = (
        f"{'field':>5} {'port':>5} {'O_fluid':>12} {'O_h':>12} {'O_h2':>12} "
        f"{'sigma_tot':>12} {'Delta_AB':>12} {'Delta_cad':>12} "
        f"{'threshold':>12} {'verdict':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in records:
        mark = "[0]" if r["exact_zero"] else ""
        print(
            f"{r['field']:>5} {r['port']:>5} {r['o_fluid']:12.5g} "
            f"{r['o_h']:12.5g} {r['o_h2']:12.5g} {r['sigma_tot']:12.5g} "
            f"{r['delta_ab']:12.5g} {r['delta_cad']:12.5g} "
            f"{r['threshold']:12.5g} {r['status']:>8}{mark}"
        )
    n_pass = sum(1 for r in records if r["status"] == "PASS")
    n_fail = sum(1 for r in records if r["status"] == "FAIL")
    n_ungated = sum(1 for r in records if r["status"] == "UNGATED")
    print(
        f"  {n_pass} PASS, {n_fail} FAIL, {n_ungated} UNGATED, "
        f"over {len(records)} scored row(s)"
    )


def _report_verdict(block):
    print("\n--- G-verdict: mid-port density ratio ---")
    print(
        f"  ports {', '.join(MID_PORTS)} (the 'n' rows at z = 470.05, 789.55, "
        "1045.15 cm)"
    )
    print(f"  mean mid-port n ratio, fluid: {block['ratio_fluid']:.4f}")
    print(f"  mean mid-port n ratio, h:     {block['ratio_h']:.4f}")
    print(f"  mean mid-port n ratio, h/2:   {block['ratio_h2']:.4f}")
    print(
        f"  Isat mean |dev|/sigma_tot: fluid {block['isat_fluid']:.4f} | "
        f"h {block['isat_h']:.4f} | h/2 {block['isat_h2']:.4f}"
    )
    print(
        f"  Isat degrades vs fluid: h {block['isat_degrades_h']} | "
        f"h/2 {block['isat_degrades_h2']}"
    )
    print(f"  bin at h:   {block['bin_h']}")
    print(f"  bin at h/2: {block['bin_h2']}")
    print(f"  bins agree: {block['bin_h'] == block['bin_h2']}")


def run_gate(fluid, dvm_h, dvm_h2, es, json_path=None):
    """Score three runs, print the gate table and verdict block, return records."""
    overlay_path = (
        OVERLAY if es == 1 else OVERLAY.parent / f"es{es}_sim1d_overlay.npz"
    )
    overlay = np.load(overlay_path, allow_pickle=False)

    print(f"\n=== M1 verdict-invariance gate (ES{es}) ===")
    print(f"  fluid:  {fluid}")
    print(f"  DVM h:  {dvm_h}")
    print(f"  DVM h/2:{dvm_h2}")
    # Scored one at a time: each result namespace falls out of scope before the
    # next is read, so only one run's fields are resident at a time.
    print("\n  scoring fluid ...")
    rows_fluid = score_run(fluid, overlay)
    print("  scoring DVM at h ...")
    rows_h = score_run(dvm_h, overlay)
    print("  scoring DVM at h/2 ...")
    rows_h2 = score_run(dvm_h2, overlay)

    records = gate_table(rows_fluid, rows_h, rows_h2)
    block = verdict_block(rows_fluid, rows_h, rows_h2)
    _report_gate(records)
    _report_verdict(block)
    verdict = overall_verdict(records, block)
    print(f"\n{verdict}")

    if json_path is not None:
        payload = {
            "es": int(es),
            "fluid": str(fluid),
            "dvm_h": str(dvm_h),
            "dvm_h2": str(dvm_h2),
            "rows": records,
            "verdict_block": block,
            "verdict": verdict,
        }
        with open(json_path, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        print(f"wrote gate result to {json_path}")
    return records, block, verdict


# --- self-test ------------------------------------------------------------
# Synthetic row tables in the shape `compare` returns, exercising each branch
# of the gate without touching an artifact. Only the keys the gate reads are
# populated; the numbers are chosen to sit unambiguously on one side of each
# threshold except where the case is ABOUT the threshold.


def _row(field, port, model, ratio=1.0, sigma=1.0, sigma_tot=1.0):
    row = {
        "field": field,
        "port": port,
        "model": float(model),
        "ratio": float(ratio),
        "sigma": float(sigma),
    }
    if sigma_tot is not None:
        row["sigma_tot"] = float(sigma_tot)
    return row


def _table(n_models, n_ratios, isat_sigmas, sigma_tot=1.0):
    """Build a five-port row table: three mid-ports plus ports 41 and 50."""
    ports = ("11", "21", "29", "41", "50")
    rows = []
    for port, model, ratio in zip(ports, n_models, n_ratios):
        rows.append(
            _row("n", port, model, ratio=ratio, sigma=1.0, sigma_tot=sigma_tot)
        )
    for port, sig in zip(ports, isat_sigmas):
        rows.append(
            _row("Isat", port, 10.0, ratio=1.0, sigma=sig, sigma_tot=sigma_tot)
        )
    return rows


def _case_pass():
    """All rows well inside threshold; both cadences RESOLVED."""
    fluid = _table([1.0] * 5, [0.60] * 5, [2.0] * 5)
    h = _table([2.0] * 5, [0.90] * 5, [2.0] * 5)
    h2 = _table([2.01] * 5, [0.90] * 5, [2.0] * 5)
    return fluid, h, h2, "M1 VERDICT: RESOLVED (cadence-invariant)"


def _case_threshold_fail():
    """One row moves further than min(|Delta_AB|, sigma_tot)/3."""
    fluid = _table([1.0] * 5, [0.60] * 5, [2.0] * 5)
    h = _table([2.0] * 5, [0.90] * 5, [2.0] * 5)
    # Delta_AB = 1.0, sigma_tot = 1.0 -> threshold 0.3333; move port 11 by 0.9.
    h2 = _table([2.9, 2.0, 2.0, 2.0, 2.0], [0.90] * 5, [2.0] * 5)
    return fluid, h, h2, "FAIL: 1 rows exceed the cadence threshold"


def _case_bin_flip():
    """Rows pass the cadence gate, but the mid-port ratio crosses 0.85."""
    fluid = _table([1.0] * 5, [0.60] * 5, [2.0] * 5)
    # Mid-port ratios average 0.8501 at h and 0.8499 at h/2 -- astride the
    # RESOLVED boundary -- while the model values move by 0.001, far inside
    # the 0.3333 threshold. At h/2 the ratio is +0.2499 over the fluid with
    # Isat unchanged, so that cadence lands IMPROVED, not NULL.
    h = _table([2.0] * 5, [0.8501] * 5, [2.0] * 5)
    h2 = _table([2.001] * 5, [0.8499] * 5, [2.0] * 5)
    return (
        fluid,
        h,
        h2,
        "M1 VERDICT: CADENCE-LIMITED NULL (bin h=RESOLVED h/2=IMPROVED)",
    )


def _case_isat_degrade():
    """Ratio clears +0.10 but stays under 0.85, and Isat degrades -> NULL."""
    fluid = _table([1.0] * 5, [0.60] * 5, [2.0] * 5)
    # Isat mean |dev|/sigma_tot rises 2.0 -> 2.5 at both cadences, so the
    # IMPROVED branch is refused despite the +0.20 ratio gain.
    h = _table([2.0] * 5, [0.80] * 5, [2.5] * 5)
    h2 = _table([2.001] * 5, [0.80] * 5, [2.5] * 5)
    return fluid, h, h2, "M1 VERDICT: NULL (cadence-invariant)"


def _case_exact_zero_pass():
    """Delta_AB = 0 on every row; Delta_cad is exactly 0, so all rows pass."""
    fluid = _table([2.0] * 5, [0.90] * 5, [2.0] * 5)
    h = _table([2.0] * 5, [0.90] * 5, [2.0] * 5)
    h2 = _table([2.0] * 5, [0.90] * 5, [2.0] * 5)
    return fluid, h, h2, "M1 VERDICT: RESOLVED (cadence-invariant)"


def _case_exact_zero_fail():
    """Delta_AB = 0 but Delta_cad is not: threshold is 0, so the row fails."""
    fluid = _table([2.0] * 5, [0.90] * 5, [2.0] * 5)
    h = _table([2.0] * 5, [0.90] * 5, [2.0] * 5)
    h2 = _table([2.0 + 1e-12, 2.0, 2.0, 2.0, 2.0], [0.90] * 5, [2.0] * 5)
    return fluid, h, h2, "FAIL: 1 rows exceed the cadence threshold"


def _case_ungated():
    """A row with no usable sigma_tot is UNGATED, not passed and not failed."""
    fluid = _table([1.0] * 5, [0.60] * 5, [2.0] * 5)
    h = _table([2.0] * 5, [0.90] * 5, [2.0] * 5)
    h2 = _table([2.0] * 5, [0.90] * 5, [2.0] * 5)
    for row in (h[0], h[1]):
        row.pop("sigma_tot")
    h[2]["sigma_tot"] = float("nan")
    h[3]["sigma_tot"] = 0.0
    return fluid, h, h2, "M1 VERDICT: RESOLVED (cadence-invariant)"


SELF_TEST_CASES = (
    ("PASS: all rows inside threshold", _case_pass),
    ("FAIL: one row exceeds the threshold", _case_threshold_fail),
    ("bin flip -> CADENCE-LIMITED NULL", _case_bin_flip),
    ("Isat degrades -> IMPROVED refused", _case_isat_degrade),
    ("Delta_AB = 0, Delta_cad = 0 -> PASS", _case_exact_zero_pass),
    ("Delta_AB = 0, Delta_cad > 0 -> FAIL", _case_exact_zero_fail),
    ("sigma_tot missing/non-finite/zero -> UNGATED", _case_ungated),
)


def self_test():
    """Run the synthetic cases, printing expected vs got. Returns an exit code."""
    print("\n=== M1 verdict-invariance gate: self-test ===")
    failures = 0
    for name, build in SELF_TEST_CASES:
        fluid, h, h2, expected = build()
        records = gate_table(fluid, h, h2)
        block = verdict_block(fluid, h, h2)
        got = overall_verdict(records, block)
        ok = got == expected
        failures += 0 if ok else 1
        print(f"\n  case: {name}")
        print(f"    expected: {expected}")
        print(f"    got:      {got}")
        counts = {
            status: sum(1 for r in records if r["status"] == status)
            for status in ("PASS", "FAIL", "UNGATED")
        }
        print(
            f"    rows: {counts['PASS']} PASS, {counts['FAIL']} FAIL, "
            f"{counts['UNGATED']} UNGATED | bins h={block['bin_h']} "
            f"h/2={block['bin_h2']}"
        )
        print(f"    {'ok' if ok else 'MISMATCH'}")
    print(
        f"\nself-test: {len(SELF_TEST_CASES) - failures} of "
        f"{len(SELF_TEST_CASES)} cases passed"
    )
    return 1 if failures else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fluid", type=Path, default=None, help="fluid comparator h5")
    parser.add_argument(
        "--dvm-h", type=Path, default=None, help="DVM arm at neutral cadence h"
    )
    parser.add_argument(
        "--dvm-h2", type=Path, default=None, help="the same DVM arm at h/2"
    )
    parser.add_argument("--es", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument(
        "--json", type=Path, default=None, help="write the gate result to PATH"
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run the synthetic gate cases and exit; needs no artifact",
    )
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()

    missing = [
        name
        for name, value in (
            ("--fluid", args.fluid),
            ("--dvm-h", args.dvm_h),
            ("--dvm-h2", args.dvm_h2),
        )
        if value is None
    ]
    if missing:
        parser.error(
            f"the gate needs all three runs; missing {', '.join(missing)} "
            "(or pass --self-test)"
        )
    run_gate(args.fluid, args.dvm_h, args.dvm_h2, args.es, json_path=args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
