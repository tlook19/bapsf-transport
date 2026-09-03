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

WHICH ROWS THE GATE BINDS ON. The verdict rests on the mid-port ``n`` rows and
the ``Isat`` rows, so a cadence difference can only move a verdict through one
of those. They are the VERDICT-BEARING rows and the gate FAILS on them. Every
other scored row is still computed and reported -- a cadence excursion there
prints as ``FAIL-UNGATED`` and does not fail the run, because it cannot change
the answer. ``--all-rows`` prints the informational rows alongside.

PORT -> ROW MAPPING: the overlay carries ``port`` = [11, 21, 29, 41, 50] paired
positionally with ``z_cm`` = [470.05, 789.55, 1045.15, 1428.55, 1716.10] cm,
and ``compare`` emits one row per (field, port) carrying that port number as a
string; the three mid-ports are therefore the ``field == "n"`` rows whose
``port`` is "11", "21" or "29", at z = 470.05, 789.55 and 1045.15 cm.

Bins, evaluated per cadence against the fluid comparator (REGISTERED
2026-08-31 (Tom); these REPLACE the ``RESOLVED_RATIO = 0.85`` /
mean-Isat form this module shipped with, which was stale against the
registration):

- RESOLVED -- the model is within 1 ``sigma_tot`` of the measurement at ALL
  THREE mid ports, PER PORT (the p11/p21/p29 ``n`` rows);
- IMPROVED -- otherwise, G at least +0.10 above the comparator's G AND the
  Isat rows not degrading;
- NULL -- otherwise.

G is the mean mid-port density ratio model/measured, the row's own ``ratio``
averaged over the three mid ports. It survives the re-registration as the
IMPROVED bin's quantity, but it is NO LONGER what decides RESOLVED: a mean over
three ports can be satisfied by compensating over- and under-shoots while no
individual port agrees with the measurement, which is exactly what a
per-port test excludes.

NON-DEGRADATION, stated precisely: the Isat rows degrade if ANY SINGLE
``field == "Isat"`` row's ``sigma`` (its window-mean ``|dev|/sigma_tot``)
INCREASES relative to the fluid comparator. Per row, not the mean, and exact --
any increase, however small, degrades. The mean form this replaces let an arm
buy a large improvement at one port by giving some back at another and still
read as undegraded; a mean is the statistic that hides precisely that.

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

_SCRIPTS = Path(__file__).resolve().parents[1]
# scripts/ sibling imports: the seven purpose subdirectories on sys.path.
import sys as _sys
from pathlib import Path as _Path
for _sub in ("atomic", "gates", "kinetic", "run", "score", "stance",
             "verify"):
    _dir = str(_Path(__file__).resolve().parents[1] / _sub)
    if _dir not in _sys.path:
        _sys.path.insert(0, _dir)

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

#: RESOLVED: the model sits within this many ``sigma_tot`` of the measurement
#: at EVERY mid port. The bins were re-registered 2026-08-31 (Tom) and this is
#: the registered form; the ``RESOLVED_RATIO = 0.85`` mean-ratio test this
#: replaces was a DIFFERENT statement -- a mean over the three ports, which a
#: pair of compensating over- and under-shoots can satisfy while no individual
#: port agrees with the measurement at all.
RESOLVED_SIGMA = 1.0

#: IMPROVED: how far the mid-port density figure of merit G must exceed the
#: comparator's before an arm counts as improved on it. Unchanged in value by
#: the re-registration; what changed is the companion non-degradation test.
IMPROVED_DELTA = 0.10

#: The fields whose rows the invariance gate is BINDING on: the verdict rests
#: on the mid-port ``n`` rows and the ``Isat`` rows, so those are the rows a
#: cadence difference could move a verdict through. Every other scored row is
#: still computed and REPORTED (``--all-rows``), but a cadence excursion there
#: cannot change the verdict and is not a gate failure.
VERDICT_BEARING_FIELDS = ("n", "Isat")


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
        # VERDICT-BEARING (2026-08-31 (Tom)): the gate BINDS only on the rows
        # a verdict rests on -- the mid-port n rows and the Isat rows. A
        # cadence excursion anywhere else is still measured and printed, but
        # it cannot move a verdict, so failing the gate on it would stop an
        # arm for a reason the verdict does not depend on.
        verdict_bearing = key[0] in VERDICT_BEARING_FIELDS and (
            key[0] != "n" or key[1] in MID_PORTS
        )
        if status == "FAIL" and not verdict_bearing:
            status = "FAIL-UNGATED"
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
                "verdict_bearing": verdict_bearing,
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
    """Return the mean ``|dev|/sigma_tot`` over the scored Isat rows.

    REPORTED, and no longer the non-degradation test: see
    :func:`isat_degrades`, which is the registered per-row form.
    """
    sub = [r for r in rows if r["field"] == "Isat"]
    if not sub:
        raise ValueError(
            "no 'Isat' rows were scored, so the degrade criterion that the "
            "IMPROVED bin depends on cannot be evaluated"
        )
    return float(np.mean([r["sigma"] for r in sub]))


def isat_rows_by_port(rows):
    """Return ``{port: |dev|/sigma_tot}`` for the scored Isat rows."""
    sub = {r["port"]: float(r["sigma"]) for r in rows if r["field"] == "Isat"}
    if not sub:
        raise ValueError(
            "no 'Isat' rows were scored, so the degrade criterion that the "
            "IMPROVED bin depends on cannot be evaluated"
        )
    return sub


def isat_degrades(rows, rows_comparator):
    """Return ``(degrades, worst_port, worst_increase)`` for the Isat rows.

    THE REGISTERED NON-DEGRADATION TEST (2026-08-31 (Tom)), stated precisely:
    the Isat rows degrade if ANY single row's ``|dev|/sigma_tot`` INCREASES
    against the comparator. Not the mean, and not a tolerance -- a strict
    per-row increase on one port is a degradation even where the mean falls.

    That is a deliberately stricter reading than the mean comparison it
    replaces, and it is the one the bins were registered on: an arm that
    buys a large improvement at one port by giving some back at another has
    not left the Isat rows undegraded, and a mean is exactly the statistic
    that hides it.

    Raises ``ValueError`` when the two row sets do not cover the same ports:
    a port scored on one side only cannot be compared, and silently skipping
    it would answer a different question than the one the bin asks.
    """
    here = isat_rows_by_port(rows)
    there = isat_rows_by_port(rows_comparator)
    if set(here) != set(there):
        raise ValueError(
            "the arm and its comparator scored different Isat ports "
            f"({sorted(here)} vs {sorted(there)}); the per-row "
            "non-degradation test is undefined across a differing port set"
        )
    worst_port, worst_increase = None, 0.0
    for port, value in here.items():
        increase = value - there[port]
        if increase > worst_increase:
            worst_port, worst_increase = port, increase
    return bool(worst_port is not None), worst_port, float(worst_increase)


def mid_port_within_sigma(rows):
    """Return ``(all_within, {port: |dev|/sigma_tot})`` for the mid-port n rows.

    The registered RESOLVED test: the model is within ``RESOLVED_SIGMA``
    ``sigma_tot`` of the measurement at ALL THREE mid ports, per port.
    """
    sub = {
        r["port"]: float(r["sigma"])
        for r in rows
        if r["field"] == "n" and r["port"] in MID_PORTS
    }
    if set(sub) != set(MID_PORTS):
        raise ValueError(
            f"expected one 'n' row per mid-port {MID_PORTS}, found "
            f"{sorted(sub)}; the RESOLVED bin is undefined without all three"
        )
    return bool(all(v <= RESOLVED_SIGMA for v in sub.values())), sub


def verdict_bin(rows, rows_comparator):
    """Return the verdict bin for one cadence against the fluid comparator.

    THE REGISTERED BINS (2026-08-31 (Tom)):

    * RESOLVED -- the model is within ``RESOLVED_SIGMA`` sigma_tot at ALL
      THREE mid ports (the p11/p21/p29 ``n`` rows), per port;
    * IMPROVED -- G is at least ``IMPROVED_DELTA`` above the comparator's G,
      AND no Isat row's ``|dev|/sigma_tot`` increases against the comparator;
    * NULL -- otherwise.

    Both halves of IMPROVED must hold. An arm that lifts G while pushing a
    single Isat row further from the measurement is NULL, not IMPROVED.
    """
    resolved, _ = mid_port_within_sigma(rows)
    if resolved:
        return "RESOLVED"
    g = mid_port_n_ratio(rows)
    g_comparator = mid_port_n_ratio(rows_comparator)
    degrades, _, _ = isat_degrades(rows, rows_comparator)
    if (g - g_comparator) >= IMPROVED_DELTA and not degrades:
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
    resolved_h, sigma_h = mid_port_within_sigma(rows_h)
    resolved_h2, sigma_h2 = mid_port_within_sigma(rows_h2)
    resolved_fluid, sigma_fluid = mid_port_within_sigma(rows_fluid)
    degr_h, degr_h_port, degr_h_amount = isat_degrades(rows_h, rows_fluid)
    degr_h2, degr_h2_port, degr_h2_amount = isat_degrades(rows_h2, rows_fluid)
    return {
        "ratio_fluid": ratio_fluid,
        "ratio_h": ratio_h,
        "ratio_h2": ratio_h2,
        "isat_fluid": isat_fluid,
        "isat_h": isat_h,
        "isat_h2": isat_h2,
        # Per-port mid-port deviations, the RESOLVED bin's own quantity.
        "mid_sigma_fluid": sigma_fluid,
        "mid_sigma_h": sigma_h,
        "mid_sigma_h2": sigma_h2,
        "resolved_fluid": resolved_fluid,
        "resolved_h": resolved_h,
        "resolved_h2": resolved_h2,
        # Per-ROW Isat degradation, the IMPROVED bin's own quantity, with the
        # port and size of the worst increase so a NULL can be read.
        "isat_degrades_h": degr_h,
        "isat_degrades_h2": degr_h2,
        "isat_worst_degrade_port_h": degr_h_port,
        "isat_worst_degrade_port_h2": degr_h2_port,
        "isat_worst_degrade_h": degr_h_amount,
        "isat_worst_degrade_h2": degr_h2_amount,
        "bin_h": verdict_bin(rows_h, rows_fluid),
        "bin_h2": verdict_bin(rows_h2, rows_fluid),
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


def _report_gate(records, all_rows=False):
    shown = records if all_rows else [
        r for r in records if r["verdict_bearing"]
    ]
    hidden = len(records) - len(shown)
    print("\n--- cadence gate: per scored row ---")
    if hidden:
        print(
            f"  showing the {len(shown)} VERDICT-BEARING row(s); {hidden} "
            f"further scored row(s) are gated as informational -- pass "
            f"--all-rows to print them"
        )
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
    for r in shown:
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
    n_soft = sum(1 for r in records if r["status"] == "FAIL-UNGATED")
    n_bearing = sum(1 for r in records if r["verdict_bearing"])
    print(
        f"  {n_pass} PASS, {n_fail} FAIL, {n_soft} FAIL-UNGATED, "
        f"{n_ungated} UNGATED, over {len(records)} scored row(s)"
    )
    print(
        f"  the gate BINDS on the {n_bearing} verdict-bearing row(s) "
        f"(n at ports {'/'.join(MID_PORTS)} + every Isat row); the rest are "
        f"reported and not gated, and a cadence excursion there prints as "
        f"FAIL-UNGATED"
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
        f"  Isat degrades vs fluid (PER ROW -- any row's |dev|/sigma_tot "
        f"increasing): h {block['isat_degrades_h']} | "
        f"h/2 {block['isat_degrades_h2']}"
    )
    for arm in ("h", "h2"):
        port = block[f"isat_worst_degrade_port_{arm}"]
        if port is not None:
            print(
                f"    worst Isat degradation at {arm}: port {port}, "
                f"+{block[f'isat_worst_degrade_{arm}']:.4f} sigma_tot"
            )
    print(
        f"\n  RESOLVED test -- model within {RESOLVED_SIGMA:.1f} sigma_tot at "
        f"EVERY mid port (|dev|/sigma_tot per port):"
    )
    for label, key in (("fluid", "mid_sigma_fluid"), ("h", "mid_sigma_h"),
                       ("h/2", "mid_sigma_h2")):
        per_port = block[key]
        cells = "  ".join(
            f"p{p} {per_port[p]:.3f}" for p in MID_PORTS
        )
        flag = block[
            "resolved_fluid" if label == "fluid"
            else ("resolved_h" if label == "h" else "resolved_h2")
        ]
        print(f"    {label:>5}: {cells}   -> within at all three: {flag}")
    print(f"\n  bin at h:   {block['bin_h']}")
    print(f"  bin at h/2: {block['bin_h2']}")
    print(f"  bins agree: {block['bin_h'] == block['bin_h2']}")
    print(
        f"  BINS (registered 2026-08-31): RESOLVED = within "
        f"{RESOLVED_SIGMA:.1f} sigma_tot at all three mid ports; IMPROVED = "
        f"G >= comparator G + {IMPROVED_DELTA:.2f} AND no Isat row's "
        f"|dev|/sigma_tot increasing; NULL otherwise."
    )


def run_gate(fluid, dvm_h, dvm_h2, es, json_path=None, all_rows=False):
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
    _report_gate(records, all_rows=all_rows)
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


def _table(n_models, n_ratios, isat_sigmas, sigma_tot=1.0, n_sigmas=None):
    """Build a five-port row table: three mid-ports plus ports 41 and 50.

    ``n_sigmas`` is the per-port ``|dev|/sigma_tot`` of the ``n`` rows, which
    is the RESOLVED bin's own quantity. It defaults to 0.5 -- comfortably
    inside the bin -- so a case that is not about RESOLVED does not have to
    say anything about it.
    """
    ports = ("11", "21", "29", "41", "50")
    if n_sigmas is None:
        n_sigmas = [0.5] * len(ports)
    rows = []
    for port, model, ratio, nsig in zip(ports, n_models, n_ratios, n_sigmas):
        rows.append(
            _row("n", port, model, ratio=ratio, sigma=nsig, sigma_tot=sigma_tot)
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
    """Rows pass the cadence gate, but the two cadences land in DIFFERENT bins.

    RESOLVED at h (every mid port inside 1 sigma_tot) and IMPROVED at h/2
    (one mid port outside, but G is +0.25 over the fluid with no Isat row
    degrading).
    """
    fluid = _table([1.0] * 5, [0.60] * 5, [2.0] * 5, n_sigmas=[2.0] * 5)
    h = _table([2.0] * 5, [0.85] * 5, [2.0] * 5, n_sigmas=[0.5] * 5)
    h2 = _table(
        [2.001] * 5, [0.85] * 5, [2.0] * 5, n_sigmas=[0.5, 0.5, 1.5, 0.5, 0.5]
    )
    return (
        fluid,
        h,
        h2,
        "M1 VERDICT: CADENCE-LIMITED NULL (bin h=RESOLVED h/2=IMPROVED)",
    )


def _case_isat_degrade():
    """G clears +0.10, no port RESOLVED, and every Isat row degrades -> NULL."""
    fluid = _table([1.0] * 5, [0.60] * 5, [2.0] * 5, n_sigmas=[2.0] * 5)
    h = _table([2.0] * 5, [0.80] * 5, [2.5] * 5, n_sigmas=[2.0] * 5)
    h2 = _table([2.001] * 5, [0.80] * 5, [2.5] * 5, n_sigmas=[2.0] * 5)
    return fluid, h, h2, "M1 VERDICT: NULL (cadence-invariant)"


def _case_resolved_needs_all_three():
    """RESOLVED refused when ONE mid port sits outside 1 sigma_tot.

    THE RE-REGISTRATION'S POINT, as a case, and it is built to FLIP: G is 0.90
    here, so the retired ``ratio >= 0.85`` test would have called this
    RESOLVED, while port 29 sits 1.8 sigma_tot from the measurement and the
    registered per-port test refuses it. G is only +0.05 over the comparator,
    so IMPROVED is refused too and the bin is NULL. ``_retired_bin`` below
    asserts that flip rather than leaving it as a claim in this docstring.
    """
    fluid = _table([1.0] * 5, [0.85] * 5, [2.0] * 5, n_sigmas=[2.0] * 5)
    h = _table(
        [2.0] * 5, [0.90] * 5, [2.0] * 5, n_sigmas=[0.2, 0.2, 1.8, 0.2, 0.2]
    )
    h2 = _table(
        [2.001] * 5, [0.90] * 5, [2.0] * 5, n_sigmas=[0.2, 0.2, 1.8, 0.2, 0.2]
    )
    return fluid, h, h2, "M1 VERDICT: NULL (cadence-invariant)"


def _case_isat_one_row_degrades():
    """IMPROVED refused when ONE Isat row degrades although the MEAN improves.

    The other half of the re-registration. Isat goes 2.0 -> [1.0, 1.0, 1.0,
    1.0, 2.5]: the mean falls 2.0 -> 1.30, so the mean comparison this
    replaces would have allowed IMPROVED, while port 50 has moved 0.5
    sigma_tot FURTHER from the measurement. G clears +0.10 and no mid port is
    inside 1 sigma, so the bin turns entirely on the non-degradation test.
    """
    fluid = _table([1.0] * 5, [0.60] * 5, [2.0] * 5, n_sigmas=[2.0] * 5)
    isat = [1.0, 1.0, 1.0, 1.0, 2.5]
    h = _table([2.0] * 5, [0.80] * 5, isat, n_sigmas=[2.0] * 5)
    h2 = _table([2.001] * 5, [0.80] * 5, isat, n_sigmas=[2.0] * 5)
    return fluid, h, h2, "M1 VERDICT: NULL (cadence-invariant)"


def _case_non_verdict_row_excursion():
    """A cadence excursion on a NON-verdict-bearing row does not fail the gate.

    Port 41's ``n`` row is scored but is not one of the three mid ports, so it
    cannot move a verdict. It moves far past its threshold here and must print
    FAIL-UNGATED while the run's verdict still stands.
    """
    fluid = _table([1.0] * 5, [0.60] * 5, [2.0] * 5)
    h = _table([2.0] * 5, [0.90] * 5, [2.0] * 5)
    h2 = _table([2.0, 2.0, 2.0, 2.9, 2.0], [0.90] * 5, [2.0] * 5)
    return fluid, h, h2, "M1 VERDICT: RESOLVED (cadence-invariant)"


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


#: The RETIRED bins (``RESOLVED_RATIO = 0.85`` and the MEAN-Isat
#: non-degradation test), kept only so the two cases below can PROVE they
#: decide differently. Nothing else calls this, and it is not a fallback.
_RETIRED_RESOLVED_RATIO = 0.85


def _retired_bin(rows, rows_comparator):
    """Return the bin the RETIRED (pre-2026-08-31) rules would have given."""
    g = mid_port_n_ratio(rows)
    if g >= _RETIRED_RESOLVED_RATIO:
        return "RESOLVED"
    g_comp = mid_port_n_ratio(rows_comparator)
    isat = isat_deviation(rows)
    isat_comp = isat_deviation(rows_comparator)
    if (g - g_comp) >= IMPROVED_DELTA and isat <= isat_comp:
        return "IMPROVED"
    return "NULL"


#: Cases that must be decided DIFFERENTLY by the registered bins than by the
#: retired ones, as ``{case name: (retired bin, registered bin)}``. This is the
#: negative control on the re-registration itself: without it, "the old rule
#: would have got this wrong" is an assertion in a docstring rather than a
#: measurement, and a re-registration that quietly changed nothing would pass
#: the suite unnoticed.
BIN_FLIP_CASES = {
    "RESOLVED refused: one mid port outside 1 sigma_tot": (
        "RESOLVED",
        "NULL",
    ),
    "IMPROVED refused: one Isat row degrades though the mean improves": (
        "IMPROVED",
        "NULL",
    ),
}


SELF_TEST_CASES = (
    ("PASS: all rows inside threshold", _case_pass),
    ("FAIL: one row exceeds the threshold", _case_threshold_fail),
    ("bin flip -> CADENCE-LIMITED NULL", _case_bin_flip),
    ("Isat degrades on every row -> IMPROVED refused", _case_isat_degrade),
    ("Delta_AB = 0, Delta_cad = 0 -> PASS", _case_exact_zero_pass),
    ("Delta_AB = 0, Delta_cad > 0 -> FAIL", _case_exact_zero_fail),
    ("sigma_tot missing/non-finite/zero -> UNGATED", _case_ungated),
    # The three cases the 2026-08-31 re-registration adds. The first two are
    # the ones the RETIRED bins would have decided differently, which is what
    # makes them worth their place.
    (
        "RESOLVED refused: one mid port outside 1 sigma_tot",
        _case_resolved_needs_all_three,
    ),
    (
        "IMPROVED refused: one Isat row degrades though the mean improves",
        _case_isat_one_row_degrades,
    ),
    (
        "non-verdict-bearing row excursion -> FAIL-UNGATED, verdict stands",
        _case_non_verdict_row_excursion,
    ),
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
            for status in ("PASS", "FAIL", "FAIL-UNGATED", "UNGATED")
        }
        print(
            f"    rows: {counts['PASS']} PASS, {counts['FAIL']} FAIL, "
            f"{counts['FAIL-UNGATED']} FAIL-UNGATED, "
            f"{counts['UNGATED']} UNGATED | bins h={block['bin_h']} "
            f"h/2={block['bin_h2']}"
        )
        # NEGATIVE CONTROL on the re-registration: where a case is registered
        # as a bin FLIP, the retired rules must actually give the other answer.
        if name in BIN_FLIP_CASES:
            want_retired, want_now = BIN_FLIP_CASES[name]
            got_retired = _retired_bin(h, fluid)
            got_now = block["bin_h"]
            flip_ok = got_retired == want_retired and got_now == want_now
            failures += 0 if flip_ok else 1
            print(
                f"    BIN FLIP: retired rules give {got_retired} "
                f"(expected {want_retired}), registered rules give "
                f"{got_now} (expected {want_now}) -- "
                f"{'ok' if flip_ok else 'MISMATCH'}"
            )
        print(f"    {'ok' if ok else 'MISMATCH'}")
    print(
        f"\nself-test: {len(SELF_TEST_CASES)} cases, "
        f"{len(BIN_FLIP_CASES)} of them additionally asserting a bin FLIP "
        f"against the retired rules; {failures} failure(s)"
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
        "--all-rows",
        action="store_true",
        help=(
            "print every scored row, not only the verdict-bearing ones. "
            "INFORMATIONAL: it changes what is shown, never what is gated"
        ),
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
    run_gate(
        args.fluid,
        args.dvm_h,
        args.dvm_h2,
        args.es,
        json_path=args.json,
        all_rows=args.all_rows,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
