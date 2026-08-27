"""Certify pure == compiled on a configuration that actually INTERPOLATES.

The production golden returns ``exact=True`` on both the pure and the compiled
path while saying nothing whatever about ``_interp_scalar``. The reason is NOT
that its configuration never reaches a table lookup -- it reaches millions of
them, and the interpolation-fusion work of 2026-08-17 measured lerp results
changing bitwise underneath a trajectory that stayed bit-identical. The reason
is that those differences never propagate into the state the fixture compares:
**a bit-exactness gate certifies only what reaches its SAVED state**, not
every line it executes. A second gate is therefore needed for interpolation
behaviour the golden runs but cannot witness.

This is that gate. It reuses the frozen K7c arms
(``k7cbuild_frozen_bitexact.py``), which carry ``beam_deposition_model =
"csda"`` and therefore drive the compiled march's three cross-section table
interpolations:

- ``tw``    -- tail_walk at the shipped rung, energy-only, legacy-pinned;
- ``twion`` -- the same plus the ionizing tail channel.

Each arm is run twice, once per kernel path, in a SEPARATE process -- the
compiled/pure choice is bound at import, so it cannot be switched inside one
interpreter -- and the two digest streams are compared line by line.

Two refusals, both earned:

- **Every digest file must carry a non-zero digest-line count.** Comparing two
  empty files reports agreement, and an arm that crashed early produces
  exactly that. A file with no digest lines fails the gate rather than passing
  it.
- **Every child reports its own kernel provenance, in-process.** A run that
  opted in to the compiled kernels but loaded the pure ones would compare pure
  against pure and call it a match. The child checks
  ``cablp.cathode.kernels.PROVENANCE`` against what was asked of it and exits
  non-zero on a mismatch; the parent re-checks the reported line before
  scoring anything.

Usage (from <checkout>/cablp, with PYTHONPATH set to that same cablp, and the
extension built with ``python build_ext.py --inplace``)::

    python scripts/interp_bitexact_gate.py
    python scripts/interp_bitexact_gate.py --steps 400 --report-every 25
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

#: What ``_kernels.PROVENANCE`` must read as on each path.
PURE_PROVENANCE = "pure"
COMPILED_PROVENANCE = "cython/_cathode_kernels_cy/tierA+csda"

#: A digest line is an indented label followed by the sha256 hex k7cbuild
#: prints. Step headers and the arm banner deliberately do NOT match.
_DIGEST_LINE = re.compile(r"^\s+\w+\s+[0-9a-f]{64}\s*$")

_PROVENANCE_TAG = "PROVENANCE "


def _digest_lines(text):
    """The comparable lines of a child's output."""
    return [ln for ln in text.splitlines() if _DIGEST_LINE.match(ln)]


def child(arm, steps, report_every, want):
    """Run one arm in this process and print provenance, then its digests."""
    from cablp.cathode import kernels as _kernels

    provenance = str(_kernels.PROVENANCE)
    print(f"{_PROVENANCE_TAG}{provenance}", flush=True)
    if provenance != want:
        print(
            f"PROVENANCE MISMATCH: this process reports {provenance!r} but was "
            f"asked for {want!r}. Refusing to produce digests -- a gate that "
            f"never loaded the path it claims to test proves nothing.",
            file=sys.stderr,
        )
        return 2

    import k7cbuild_frozen_bitexact as k7c

    k7c.main(["--steps", str(steps), "--report-every", str(report_every),
              "--arm", arm])
    return 0


def _run_one(arm, path, steps, report_every, outdir):
    """Spawn one child, capture its output, return (path_to_file, text, rc)."""
    want = COMPILED_PROVENANCE if path == "compiled" else PURE_PROVENANCE
    env = dict(os.environ)
    if path == "compiled":
        env["CABLP_COMPILED_KERNELS"] = "1"
    else:
        env.pop("CABLP_COMPILED_KERNELS", None)
    cmd = [
        sys.executable, str(SCRIPT_DIR / "interp_bitexact_gate.py"),
        "--child", "--arm", arm, "--steps", str(steps),
        "--report-every", str(report_every), "--want", want,
    ]
    print(f"  running {arm}/{path} ...", flush=True)
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    text = proc.stdout + proc.stderr
    out = outdir / f"t22ifuse_{arm}_{path}.txt"
    out.write_text(text)
    return out, text, proc.returncode


def _check(out, text, rc, arm, path):
    """Provenance, exit status and non-emptiness, before anything is compared."""
    want = COMPILED_PROVENANCE if path == "compiled" else PURE_PROVENANCE
    reported = None
    for ln in text.splitlines():
        if ln.startswith(_PROVENANCE_TAG):
            reported = ln[len(_PROVENANCE_TAG):].strip()
            break
    lines = _digest_lines(text)
    print(f"  {arm}/{path}: rc={rc} provenance={reported!r} "
          f"digest_lines={len(lines)} -> {out.name}")
    if rc != 0:
        print(f"  REFUSING: {arm}/{path} exited {rc}; see {out}")
        return None
    if reported != want:
        print(f"  REFUSING: {arm}/{path} reported provenance {reported!r}, "
              f"expected {want!r}")
        return None
    if not lines:
        print(f"  REFUSING: {arm}/{path} produced ZERO digest lines. Two empty "
              f"streams compare equal, which is how an empty comparison once "
              f"read as a PASS.")
        return None
    return lines


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--arms", nargs="+", default=["tw", "twion"])
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--report-every", type=int, default=25)
    p.add_argument("--outdir", type=Path, default=SCRIPT_DIR)
    # Child-mode plumbing; not for direct use.
    p.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--arm", help=argparse.SUPPRESS)
    p.add_argument("--want", help=argparse.SUPPRESS)
    a = p.parse_args(argv)

    if a.child:
        return child(a.arm, a.steps, a.report_every, a.want)

    a.outdir.mkdir(parents=True, exist_ok=True)
    print(f"interp bit-exactness gate: arms={a.arms} steps={a.steps} "
          f"report_every={a.report_every}")
    failures = []
    for arm in a.arms:
        streams = {}
        for path in ("pure", "compiled"):
            out, text, rc = _run_one(arm, path, a.steps, a.report_every,
                                     a.outdir)
            streams[path] = _check(out, text, rc, arm, path)
        if streams["pure"] is None or streams["compiled"] is None:
            failures.append(f"{arm}: REFUSED (see above)")
            continue
        pure, comp = streams["pure"], streams["compiled"]
        if len(pure) != len(comp):
            print(f"  {arm}: LINE COUNT DIFFERS pure={len(pure)} "
                  f"compiled={len(comp)}")
            failures.append(f"{arm}: digest-line counts differ")
            continue
        differing = [i for i, (x, y) in enumerate(zip(pure, comp)) if x != y]
        print(f"  {arm}: compared {len(pure)} digest lines, "
              f"{len(differing)} differing")
        if differing:
            for i in differing[:6]:
                print(f"      line {i}:\n        pure     {pure[i].strip()}\n"
                      f"        compiled {comp[i].strip()}")
            if len(differing) > 6:
                print(f"      ... and {len(differing) - 6} more")
            failures.append(f"{arm}: {len(differing)}/{len(pure)} digest lines "
                            f"differ")

    if failures:
        print("\nGATE FAILED:")
        for f in failures:
            print(f"  {f}")
        return 1
    print("\nGATE OK — pure and compiled are bit-identical on every arm.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
