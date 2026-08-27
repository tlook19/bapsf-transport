"""Synthetic unit-check for the stage (iii) decay-observability disclosure.

Two independent checks, both on hand-built rows -- no HDF5, no solver:

1. CRITERION. ``_decay_observability`` is exercised on a normally-decaying
   port, a port whose ``tau_exp`` sits just under the window span, one just
   over it, the ES3 port-50 case (601.60 ms, a trace that does not decay
   inside the window at all), and a NaN row. Each case asserts the expected
   OBSERVED/EXTRAPOLATED verdict and the expected ``D_exp``.

2. NO-REGRESSION. The pre-change ``_report_decay`` is exec'd out of the
   baseline commit and run on the same rows, and every line the old printer
   emitted is checked to survive verbatim in the new output -- the row lines
   as exact prefixes (the new columns are appended), the header/rule and the
   existing mean line byte-for-byte. This is the mechanical form of the
   "every number the script prints today must still print, unchanged, with
   the same label" constraint.

Run from ``<checkout>/cablp`` with ``PYTHONPATH=<checkout>/cablp``.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import math
import pathlib
import subprocess
import sys

import numpy as np

BASELINE_REF = "64e9565"  # agent-staging tip this branch was cut from
# DELIBERATELY the pre-flatten path, and not a stale one. It is looked up with
# `git show BASELINE_REF:REPO_PATH` against a PINNED HISTORICAL revision, where
# the sim1d scripts lived under `cablp/scripts/`. Q5 disposition (review, 26dw;
# `restructure/RENAME_MAP.md` section 8): recipes resolved against a base that
# predates the R2 flatten keep the old path, and the flatten commit is the named
# boundary. Rewriting this to `scripts/...` breaks the lookup it exists to do.
REPO_PATH = "cablp/scripts/compare_sim1d_es1.py"
WINDOW = (20.0, 21.5)
SPAN = WINDOW[1] - WINDOW[0]


def _load_current():
    here = pathlib.Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "_cmp_es1_current", here / "compare_sim1d_es1.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_baseline_report_decay():
    """Return the pre-change ``_report_decay`` from the baseline commit."""
    src = subprocess.run(
        ["git", "show", f"{BASELINE_REF}:{REPO_PATH}"],
        cwd=pathlib.Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    ns: dict = {
        "__file__": str(pathlib.Path(__file__).resolve()),
        "__name__": "_cmp_es1_baseline",
    }
    exec(compile(src, f"{BASELINE_REF}:{REPO_PATH}", "exec"), ns)
    return ns["_report_decay"]


# --- the synthetic rows ---------------------------------------------------
# tau_exp values chosen to straddle the criterion: the window span is the only
# scale in it, so the interesting points are span-epsilon and span+epsilon.
CASES = [
    # (label, tau_model_ms, tau_exp_ms, expect_extrapolated)
    ("normal decay (ES3 port 11)", 0.84, 0.69, False),
    ("tau_exp just UNDER the span", 1.40, 1.49, False),
    ("tau_exp EQUAL to the span", 1.40, 1.50, False),
    ("tau_exp just OVER the span", 1.40, 1.51, True),
    ("ES3 port 41 (1.74 ms)", 1.40, 1.74, True),
    ("ES3 port 50 (601.60 ms, flat trace)", 1.92, 601.60, True),
    ("no measured e-fold at all (NaN)", 1.50, float("nan"), False),
]


def check_criterion(mod) -> None:
    print("=== 1. criterion, on synthetic tau_exp values "
          f"(window span {SPAN:.1f} ms) ===")
    print(f"{'case':>38} {'tau_exp':>9} {'D_exp [%]':>10} {'verdict':>13}")
    failures = 0
    for label, _tau_model, tau_exp, expect_extrap in CASES:
        extrap, d_exp = mod._decay_observability(tau_exp, SPAN)
        if math.isnan(tau_exp):
            expect_d = float("nan")
            ok_d = math.isnan(d_exp)
        else:
            expect_d = 1.0 - math.exp(-SPAN / tau_exp)
            ok_d = abs(d_exp - expect_d) < 1.0e-12
        ok = (extrap == expect_extrap) and ok_d
        failures += 0 if ok else 1
        if math.isnan(tau_exp):
            verdict = "unscored"  # no measured e-fold to classify either way
        else:
            verdict = "EXTRAPOLATED" if extrap else "observed"
        print(
            f"{label:>38} {tau_exp:9.2f} {100.0 * d_exp:10.4f} "
            f"{verdict:>13}   {'ok' if ok else 'MISMATCH'}"
        )
    print(f"  criterion mismatches: {failures}")
    assert failures == 0, "criterion did not match the expected classification"


def build_rows(mod):
    rows = []
    for i, (_label, tau_model, tau_exp, _expect) in enumerate(CASES):
        extrapolated, decay_frac_exp = mod._decay_observability(tau_exp, SPAN)
        rows.append(
            {
                "port": 10 + i,
                "z": 400.0 + 100.0 * i,
                "tau_exp_ms": tau_exp,
                "tau_model_ms": tau_model,
                "ratio": tau_model / tau_exp if np.isfinite(tau_exp) else np.nan,
                "extrapolated": extrapolated,
                "decay_frac_exp": decay_frac_exp,
            }
        )
    return rows


def _capture(fn, *args) -> list[str]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(*args)
    return buf.getvalue().splitlines()


def check_no_regression(mod, rows) -> None:
    old_lines = _capture(_load_baseline_report_decay(), rows, WINDOW)
    new_lines = _capture(mod._report_decay, rows, WINDOW)

    print("\n=== 2a. baseline (pre-change) printer, same rows ===")
    print("\n".join(old_lines))
    print("\n=== 2b. current printer, same rows ===")
    print("\n".join(new_lines))

    print("\n=== 2c. every baseline line survives in the new output ===")
    failures = 0
    for line in old_lines:
        # Row lines gained appended columns, so they survive as PREFIXES; every
        # other line (header, rule, the existing mean) must survive
        # BYTE-IDENTICAL. Try identity first, fall back to prefix.
        hit = line in new_lines
        kind = "verbatim"
        if not hit:
            hit = any(n.startswith(line) for n in new_lines)
            kind = "prefix (columns appended)"
        failures += 0 if hit else 1
        print(f"  [{'ok' if hit else 'LOST'}] {kind:>26} | {line!r}")
    print(f"  lost baseline lines: {failures}")
    assert failures == 0, "a pre-existing printed line did not survive"

    old_means = [ln for ln in old_lines if "mean tau_model/tau_exp" in ln]
    new_means = [ln for ln in new_lines if "mean tau_model/tau_exp" in ln]
    print("\n=== 2d. the existing all-port mean line is byte-identical ===")
    for ln in old_means:
        print(f"  baseline: {ln!r}")
        assert ln in new_means, "the all-port mean line changed"
    print(f"  new mean lines printed: {len(new_means)} "
          f"(baseline printed {len(old_means)})")


def check_all_marked(mod) -> None:
    print("\n=== 3. every scored port extrapolated -> no empty mean ===")
    rows = [
        {
            "port": 50,
            "z": 1716.0,
            "tau_exp_ms": 601.60,
            "tau_model_ms": 1.92,
            "ratio": 1.92 / 601.60,
            "extrapolated": True,
            "decay_frac_exp": 1.0 - math.exp(-SPAN / 601.60),
        }
    ]
    lines = _capture(mod._report_decay, rows, WINDOW)
    print("\n".join(lines))
    tail = [ln for ln in lines if "OBSERVED ports only" in ln]
    assert tail and "none" in tail[0], "expected an explicit 'none', not a mean"


def main() -> int:
    mod = _load_current()
    check_criterion(mod)
    rows = build_rows(mod)
    check_no_regression(mod, rows)
    check_all_marked(mod)
    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
