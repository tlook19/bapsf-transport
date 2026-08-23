"""Exercise ``compare_sim1d_es1.py --es 4`` WITHOUT reading the ES4 holdout.

ES4 is a deliberate holdout: the M4 ES4 comparison is frozen until the thesis
writing stretch, so the scoring path has to be built and exercised without any
ES4 measured value being read, printed, or written anywhere. This check does
that by scoring against a STAND-IN overlay -- the ES1 product, copied under
the file name ``es4_sim1d_overlay.npz`` into a scratch directory (with its
discharge-current trace scaled, see below) -- and redirecting the scorer's
overlay root there. Every number this prints is therefore an ES1 number, or a
known multiple of one; the real ``scripts/data/es4_sim1d_overlay.npz`` is only
ever tested for existence, never opened.

What it proves:

* the ``--es 4`` overlay selection resolves to ``es4_sim1d_overlay.npz`` and
  does NOT fall back to the ES1 file: the stand-in's discharge-current trace
  is scaled by a known factor, and the measured peak the scorer prints at
  rung 4 is asserted to carry that factor. A silent fallback to the ES1
  product would print the unscaled peak;
* all three scoring stages -- (i) discharge current, (ii) port Te / density /
  Isat, (iii) per-port Isat decay -- run to completion at rung 4 and render
  the rung in the stage (ii) header;
* the sigma_tot error model and both semi-quantitative criteria are reached
  at rung 4 (the same code path as every other rung: nothing in the error
  model is keyed on the rung);
* the guard on the fresh-run path fires for every rung other than 1, so a
  rung can never be scored against a model this driver built at the ES1
  operating point.

Usage::

    python scripts/es4_scorer_path_check.py --from-h5 scripts/RUN.h5

Any full-cycle artifact whose trace covers the stage (iii) window works; the
stand-in is ES1 data, so the printed deviations mean nothing physically and
are not to be quoted.
"""

import argparse
import contextlib
import io
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import compare_sim1d_es1 as SCORER  # noqa: E402

#: factor applied to the stand-in's discharge-current trace.  It has no
#: physical meaning -- it is the tracer that distinguishes "read the rung-4
#: file" from "silently fell back to ES1".
STANDIN_CURRENT_SCALE = 2.0


def _build_standin(tmp_dir):
    """Write the ES1 overlay into ``tmp_dir`` under both rung names.

    The ES4 copy has its discharge-current trace scaled by
    ``STANDIN_CURRENT_SCALE``, so the stage (i) measured peak says which file
    was actually read.
    """
    data_dir = tmp_dir / "data"
    data_dir.mkdir(parents=True)
    src = SCORER.OVERLAY
    if not src.exists():
        raise SystemExit(f"missing the ES1 overlay to copy: {src}")
    shutil.copyfile(src, data_dir / "es1_sim1d_overlay.npz")
    with np.load(src, allow_pickle=False) as ov:
        payload = {k: ov[k] for k in ov.files}
    es1_peak = float(np.nanmax(payload["discharge_current_mean_a"]))
    for key in ("discharge_current_mean_a", "discharge_current_sem_a"):
        payload[key] = payload[key] * STANDIN_CURRENT_SCALE
    np.savez_compressed(data_dir / "es4_sim1d_overlay.npz", **payload)
    return data_dir, es1_peak


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-h5",
        type=Path,
        required=True,
        help="any saved full-cycle sim1d result to score the stand-in against",
    )
    args = parser.parse_args(argv)

    real_es4 = SCORER.OVERLAY.parent / "es4_sim1d_overlay.npz"
    print(
        f"real ES4 product present on disk: {real_es4.exists()}"
        f" ({real_es4})"
    )
    print("  -- existence only; this script never opens it.")

    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        data_dir, es1_peak = _build_standin(Path(tmp))
        real_overlay = SCORER.OVERLAY
        SCORER.OVERLAY = data_dir / "es1_sim1d_overlay.npz"
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = SCORER.main(["--from-h5", str(args.from_h5), "--es", "4"])
            text = buf.getvalue()
            print(text)

            # No silent ES1 fallback: the peak printed at rung 4 must be
            # the SCALED one. Compared at the printed precision (4 s.f.).
            hit = re.search(r"measured\s+([0-9.eE+-]+)\s+\+/-", text)
            printed_peak = float(hit.group(1)) if hit else float("nan")
            expected = es1_peak * STANDIN_CURRENT_SCALE
            no_fallback = bool(
                np.isfinite(printed_peak)
                and abs(printed_peak - expected) <= 1e-3 * abs(expected)
            )
            print(
                "stage (i) measured peak printed at rung 4: "
                f"{printed_peak:.6g} A"
                f" | stand-in expects {expected:.6g} A"
                f" | an ES1 fallback would print {es1_peak:.6g} A"
            )

            checks = [
                ("rung-4 scoring returns 0", rc == 0),
                (
                    "rung 4 read the rung-4 file (no silent ES1 fallback)",
                    no_fallback,
                ),
                (
                    "stage (i) discharge current ran",
                    "stage (i): discharge current" in text,
                ),
                (
                    "stage (ii) ran and names rung 4",
                    "stage (ii): bulk Te / density at the ES4 ports" in text,
                ),
                (
                    "stage (ii) reports the sigma_tot column",
                    "|dev|/sig" in text,
                ),
                (
                    "stage (ii) reached the semi-quantitative criteria",
                    "semi-quantitative marks" in text,
                ),
                (
                    "stage (iii) decay e-folds ran",
                    "stage (iii): Isat decay e-fold times" in text,
                ),
            ]
        finally:
            SCORER.OVERLAY = real_overlay

    # Fresh-run guard: no rung other than 1 may be scored against a model this
    # driver builds, because it only ever builds the ES1 operating point.
    for rung in (2, 3, 4):
        try:
            SCORER.main(["--es", str(rung)])
        except ValueError as exc:
            hit = "ES1 operating point" in str(exc)
        else:
            hit = False
        checks.append((f"fresh-run guard fires for --es {rung}", hit))

    print("\n--- checks ---")
    for name, passed in checks:
        ok = ok and passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    print(
        "\nNo ES4 measured value was read, printed or written: every number "
        "above came from the ES1 product copied under the ES4 file name."
    )
    print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
