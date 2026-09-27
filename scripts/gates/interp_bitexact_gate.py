r"""Certify pure == compiled on a configuration that actually INTERPOLATES.

The production golden returns ``exact=True`` on both the pure and the compiled
path while saying nothing whatever about ``_interp_scalar``. The reason is NOT
that its configuration never reaches a table lookup -- it reaches millions of
them, and the interpolation-fusion work of 2026-08-17 measured lerp results
changing bitwise underneath a trajectory that stayed bit-identical. The reason
is that those differences never propagate into the state the fixture compares:
**a bit-exactness gate certifies only what reaches its SAVED state**, not
every line it executes. A second gate is therefore needed for interpolation
behaviour the golden runs but cannot witness.

This is that gate, and it now has TWO legs, one per fused-lerp implementation.
``cablp.numerics.interp`` ships two of them -- the scalar
``interp_scalar_fused`` and the array ``interp_array_fused`` (through
``_interp_array_unchecked_multi`` and the :func:`fma_array` reconstruction
under it) -- and they are separate source paths that reach the same tables. A
gate that exercised only one would leave the other uncertified against exactly
the ISA-baseline contraction question this gate exists to answer.

**Leg A -- the SCALAR path, pure vs compiled on two frozen probe arms.** The
arms are defined in this file (``PROBE_PARAMS``, ``PROBE_FLAGS``, ``EXTRA``,
``ARMS``); they deposit the beam by the CSDA march with the quasilinear
channel and a walked, ionizing plateau tail, and therefore drive the compiled
march's three cross-section table interpolations along the primary and along
every plateau group:

- ``mg``    -- ``heating_anomalous_transport = "plateau_multigroup"`` with the
  reflecting cathode face;
- ``mgesc`` -- the same with the free-escaping cathode face.

The probe configuration is FROZEN: every value it stands on is written out in
this file and moves only when someone edits it, so two checkouts compared by
this gate run the same configuration. It names no committed stance and does
not track the reference one; each child prints the :func:`config_identity` of
the arm it ran, so two transcripts can be seen to share a configuration.
SHA-256 digests are taken over the raw little-endian float64 bytes of every
piece of state the arm carries (the fluid state and the kinetic-neutral
distributions, pending fluxes, transfers and debt ledgers).

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

**Leg B -- the ARRAY path, over the committed ``deposit_beam`` corpus.** The
sole array-fused-lerp consumer is ``cablp.cathode.beam_lane_march``, and the
committed instrument that A/Bs it against the scalar-fused recursive route is
``scripts/verify/r3lane_equivalence.py --corpus``: it replays every corpus entry,
runs each tail-leg batch BOTH ways, and censuses FLIPPED WALKERS at raw
uint64. This gate runs it as a subprocess and scores three things, refusing on
any of them -- the same fail-closed discipline leg A applies to its digest
streams:

- the child exits 0 and prints ``LANE EQUIVALENCE OK``;
- the branch-flip census reports ZERO flipped walkers;
- **the corpus line reports more than zero batches actually marched as
  lanes.** A run in which every batch fell back to the recursive route would
  compare that route against itself and report perfect agreement, which is the
  array-path form of leg A's empty-stream trap: it is a vacuous pass, and it
  fails the gate rather than passing it.

Both legs leave capture files behind -- one per arm per kernel path, plus the
array leg's own transcript -- so ``--outdir`` is REQUIRED and names a
directory OUTSIDE the repository, whose ``scripts/`` tree holds code only.
There is deliberately no default: a bare invocation refuses rather than
scattering run artifacts into the checkout.

Usage (from the checkout ROOT, with PYTHONPATH set to that same root, and the
extension built with ``python build_ext.py --inplace``)::

    python scripts/gates/interp_bitexact_gate.py --outdir <artifacts-dir>
    python scripts/gates/interp_bitexact_gate.py --outdir <artifacts-dir> \
        --steps 400 --report-every 25
"""

import argparse
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

#: ``scripts/`` itself -- the parent of the seven purpose subdirectories,
#: used to address leg B's sibling instrument. It is NOT an output
#: location: captured output goes where ``--outdir`` says, outside the repo.
SCRIPTS_ROOT = Path(__file__).resolve().parents[1]

#: This file's own directory, ``scripts/gates/``. The child relaunch
#: below re-invokes THIS script, so it is addressed where the script
#: actually lives rather than at the flat ``scripts/`` root.
GATE_DIR = Path(__file__).resolve().parent

#: What ``_kernels.PROVENANCE`` must read as on each path.
PURE_PROVENANCE = "pure"
COMPILED_PROVENANCE = "cython/_cathode_kernels_cy/tierA+csda"

#: A digest line is an indented label followed by the sha256 hex the child
#: prints. Step headers and the arm banner deliberately do NOT match.
_DIGEST_LINE = re.compile(r"^\s+\w+\s+[0-9a-f]{64}\s*$")

_PROVENANCE_TAG = "PROVENANCE "

#: Leg B's non-vacuity line: ``corpus: N entries, M tail-leg batches, K of
#: them actually marched as lanes``. ``K`` is the count that must exceed zero
#: -- it is the number of batches that really took the array-fused lane route
#: rather than falling back to the scalar recursive one.
_LANE_CORPUS_LINE = re.compile(
    r"^corpus:\s+(\d+)\s+entries,\s+(\d+)\s+tail-leg batches,\s+(\d+)\s+of "
    r"them actually marched as lanes\s*$"
)

#: Leg B's agreement line: ``branch-flip census: W walkers over C calls, F
#: flipped``. ``F`` must be zero.
_LANE_CENSUS_LINE = re.compile(
    r"^branch-flip census:\s+(\d+)\s+walkers over\s+(\d+)\s+calls,\s+(\d+)\s+"
    r"flipped\s*$"
)

_LANE_VERDICT_OK = "LANE EQUIVALENCE OK"


# --------------------------------------------------------------------------
# THE FROZEN PROBE CONFIGURATION of leg A. These values are owned by this
# file and held still on purpose; they are not a view of any live
# configuration and nothing keeps them in step with one. Some coincide with
# values the reference stance carries; they must never become a live read,
# because a probe whose configuration drifts underneath it cannot certify
# that two checkouts carried the same trajectory.
PROBE_PARAMS = {
    "V_bank": 177.843,
    "R_comp": 0.0072244,
    "L_parasitic_H": 8.1e-06,
    "C_bank_F": 9.5,
    "equilibration_gas_puff_on_s": 0.025,
    "S_gp": 9010.0,
    "b_beam_excitation": 1.4,
    "Rp": 18.415,
    "R_cath": 18.415,
    "implicit_heat_scheme": "tr_bdf2",
    "operator_splitting": "strang",
    "heat_picard_iterations": 2,
    "heat_picard_tol": 1e-10,
    "Rsup": 0.0,
    "cathode_conduction_W_per_K": 12058.0,
    "cathode_heat_capacity_J_per_K": 181.0,
    "C_R": 8.76,
    "beam_deposition_smoothing_cm": 50.0,
}
PROBE_FLAGS = {}

#: Applied over ``PROBE_PARAMS`` (and over the ES1 operating point below);
#: where a key appears in both, this value wins. ``S_gp`` is in sccm.
EXTRA = {
    "cathode_solver_model": "current_driven",
    "beam_anomalous_model": "quasilinear",
    "cathode_heat_capacity_J_per_K": 120.0,
    "cathode_emissivity": 0.7,
    "phi_wf": 2.869,
    "cathode_phiwf_clean_eV": 2.809,
    "cathode_cleaning_sigma_cm2": 3.5e-16,
    "cathode_cleaning_E_th_eV": 20.0,
    "S_gp": 3000.0,
    "nx": 240,
    "tau_afterglow": 0.006,
    "neutral_model": "kinetic_dvm",
    "neutral_kinetic_dvm_cadence_s": 1.0e-5,
    "neutral_kinetic_dvm_nvz": 96,
    "neutral_kinetic_dvm_nvp": 32,
    "neutral_kinetic_dvm_exchange": "cauchy_chord",
    "neutral_kinetic_dvm_annulus_flights": "bounded_chord",
    "C_R": 12.96,
}

#: The two leg-A arms.
ARMS = {
    "mg": {
        "heating_anomalous_transport": "plateau_multigroup",
        "heating_anomalous_tail_cathode_boundary": "reflect",
    },
    "mgesc": {
        "heating_anomalous_transport": "plateau_multigroup",
        "heating_anomalous_tail_cathode_boundary": "escape",
    },
}
# --------------------------------------------------------------------------


def build_arm(arm):
    """Return ``(sim, identity)`` for one leg-A arm.

    ``identity`` is the :func:`config_identity` of the assembled
    ``(params, flags)`` pair.
    """
    from run_mechanism_ladder import ES_OPERATING

    from cablp.solvers._sim1d import (
        LAPDSim1D,
        config_identity,
        default_config,
    )

    params, flags = default_config()
    params.update(PROBE_PARAMS)
    flags.update(PROBE_FLAGS)
    op = ES_OPERATING[1]
    params["V_bank"] = op["V_bank"]
    params["cathode_Ts_base_K"] = op["Ts_standby_K"]
    params.update(EXTRA)
    params.update(ARMS[arm])
    return LAPDSim1D(params, flags), config_identity(params, flags)


def state_digest(*arrays):
    """SHA-256 hex over the raw little-endian float64 bytes of ``arrays``."""
    import numpy as np

    h = hashlib.sha256()
    for arr in arrays:
        a = np.ascontiguousarray(np.asarray(arr, dtype="<f8"))
        h.update(a.tobytes())
    return h.hexdigest()


def run_arm(arm, steps, report_every):
    """Advance one arm ``steps`` steps, printing state digests periodically."""
    sim, identity = build_arm(arm)
    print(
        "frozen probe configuration (names no stance; does not track one)\n"
        f"config_identity={identity}"
    )
    print(
        f"arm={arm} cells={sim.geometry.cells} dvm={sim._dvm is not None} "
        f"anom_transport="
        f"{sim._input_dict.get('heating_anomalous_transport', 'local')!r} "
        f"cathode_boundary="
        f"{sim._input_dict.get('heating_anomalous_tail_cathode_boundary')!r}"
    )
    for step in range(1, steps + 1):
        sim.advance_one_step()
        if step % report_every == 0 or step == steps:
            dvm = sim._dvm
            print(
                f"step {step:6d}  t={sim.time:.9e}  updates={dvm.updates:6d}\n"
                f"    y      {state_digest(sim._y)}\n"
                f"    f_c    {state_digest(dvm.f_c)}\n"
                f"    f_a    {state_digest(dvm.f_a)}\n"
                f"    pend   {state_digest(dvm.pend_L_c, dvm.pend_R_c, dvm.pend_L_a, dvm.pend_R_a)}\n"
                f"    xfer   {state_digest(dvm.M_transfer, dvm.Ei_transfer, dvm.S_transfer, dvm.Tn_col_eV)}\n"
                f"    debt   {state_digest(dvm.M_debt, dvm.Ei_debt, dvm.M_applied_cum, dvm.Ei_applied_cum, dvm.M_booked_cum, dvm.Ei_booked_cum)}",
                flush=True,
            )
    print(f"frozen-arm digests printed (arm={arm})")



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

    # scripts/ sibling imports: the seven purpose subdirectories on sys.path.
    import sys as _sys
    from pathlib import Path as _Path
    for _sub in ("atomic", "gates", "kinetic", "run", "score", "stance",
                 "verify"):
        _dir = str(_Path(__file__).resolve().parents[1] / _sub)
        if _dir not in _sys.path:
            _sys.path.insert(0, _dir)

    run_arm(arm, steps, report_every)
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
        sys.executable, str(GATE_DIR / "interp_bitexact_gate.py"),
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


def array_leg(outdir):
    """Leg B: certify the ARRAY fused lerp over the ``deposit_beam`` corpus.

    Runs ``r3lane_equivalence.py --corpus`` -- the committed instrument that
    marches every corpus tail-leg batch through ``beam_lane_march`` (array
    fused) and through the recursive ``deposit_beam`` route (scalar fused) and
    censuses flipped walkers at raw uint64.

    Returns ``None`` on success, or a one-line failure string. Fails CLOSED:
    a non-zero exit, a missing or unparseable census line, any flipped walker,
    or a corpus in which no batch actually took the lane route is a failure,
    not a pass.
    """
    env = dict(os.environ)
    # The lane march and the fma_array reconstruction under it are pure
    # Python; the corpus instrument calls ``beam_deposition`` directly rather
    # than through the kernel selector, so the compiled opt-in is irrelevant
    # here. Cleared anyway so the leg reads the same whatever the caller's
    # environment carries.
    env.pop("CABLP_COMPILED_KERNELS", None)
    cmd = [
        sys.executable,
        str(SCRIPTS_ROOT / "verify" / "r3lane_equivalence.py"),
        "--corpus",
    ]
    print("  running array-path corpus leg (r3lane_equivalence --corpus) ...",
          flush=True)
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    text = proc.stdout + proc.stderr
    out = outdir / "interp_gate_array_leg.txt"
    out.write_text(text)

    entries = batches = lanes = None
    walkers = flipped = None
    for ln in text.splitlines():
        m = _LANE_CORPUS_LINE.match(ln)
        if m:
            entries, batches, lanes = (int(g) for g in m.groups())
            continue
        m = _LANE_CENSUS_LINE.match(ln)
        if m:
            walkers, _calls, flipped = (int(g) for g in m.groups())
    verdict_ok = any(ln.strip() == _LANE_VERDICT_OK for ln in text.splitlines())

    print(f"  array/corpus: rc={proc.returncode} entries={entries} "
          f"batches={batches} lane_marched={lanes} walkers={walkers} "
          f"flipped={flipped} verdict_ok={verdict_ok} -> {out.name}")

    if proc.returncode != 0:
        print(f"  REFUSING: array leg exited {proc.returncode}; see {out}")
        return f"array: child exited {proc.returncode}"
    if lanes is None:
        print(f"  REFUSING: array leg printed no parseable corpus line; the "
              f"non-vacuity count cannot be read. See {out}")
        return "array: corpus line missing or unparseable"
    if flipped is None:
        print(f"  REFUSING: array leg printed no parseable branch-flip census; "
              f"see {out}")
        return "array: branch-flip census missing or unparseable"
    if lanes <= 0:
        print(f"  REFUSING: array leg engaged ZERO lane legs over {entries} "
              f"corpus entries and {batches} tail-leg batches. Every batch "
              f"fell back to the recursive route, so the comparison ran the "
              f"scalar path against itself -- the array-path form of an empty "
              f"digest stream, and a vacuous pass.")
        return "array: zero lane legs engaged (vacuous)"
    if flipped != 0:
        print(f"  {flipped} of {walkers} walkers FLIPPED between the array and "
              f"scalar routes; see {out}")
        return f"array: {flipped}/{walkers} walkers differ"
    if not verdict_ok:
        print(f"  REFUSING: array leg did not print {_LANE_VERDICT_OK!r}; see "
              f"{out}")
        return "array: child withheld its OK verdict"
    print(f"  array: compared {walkers} walkers over {batches} batches "
          f"({lanes} lane-marched), 0 flipped")
    return None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--arms", nargs="+", choices=sorted(ARMS),
                   default=["mg", "mgesc"])
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--report-every", type=int, default=25)
    p.add_argument("--outdir", type=Path, default=None,
                   help="directory to write the capture files into; REQUIRED, "
                        "and must be outside the repository")
    # Child-mode plumbing; not for direct use.
    p.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--arm", help=argparse.SUPPRESS)
    p.add_argument("--want", help=argparse.SUPPRESS)
    a = p.parse_args(argv)

    if a.child:
        return child(a.arm, a.steps, a.report_every, a.want)

    if a.outdir is None:
        print("interp_bitexact_gate.py writes run artifacts (one capture file "
              "per arm per kernel path, plus the array leg's transcript) and "
              "so requires --outdir naming a directory outside the "
              "repository, whose scripts/ tree holds code only.",
              file=sys.stderr)
        return 2

    a.outdir.mkdir(parents=True, exist_ok=True)
    print(f"interp bit-exactness gate: arms={a.arms} steps={a.steps} "
          f"report_every={a.report_every}")
    print("  leg A: scalar fused lerp, pure vs compiled on the frozen probe "
          "arms")
    print("  leg B: array fused lerp, lane march vs recursive route on the "
          "deposit_beam corpus")
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

    array_failure = array_leg(a.outdir)
    if array_failure is not None:
        failures.append(array_failure)

    if failures:
        print("\nGATE FAILED:")
        for f in failures:
            print(f"  {f}")
        return 1
    print("\nGATE OK — pure and compiled are bit-identical on every arm, and "
          "the array fused lerp agrees with the scalar one over the corpus.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
