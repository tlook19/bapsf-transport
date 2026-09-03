"""Execute the locked Phase 3 RHS qualification recipe when authorized.

Importing this module is read-only.  Calling :func:`capture_phase3_rhs` or the
CLI constructs and executes ``LAPDSim1D``; those entry points are intentionally
not used by the source-implementation tests.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from cablp.solvers._sim1d.results.phase3_capture import (
    PHASE3_CAPTURE_PROFILE,
    configuration_identity,
    reserve_run_id,
    validate_run_id,
    write_qualified_capture,
)


SCRIPT_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = SCRIPT_DIR.parents[0]
OUTPUT_DIRECTORY = SCRIPT_DIR / "baselines" / "phase3_rhs"
PRODUCER_PATH = "scripts/capture_phase3_rhs.py"
MINIMUM_CAPTURE_ANCESTOR = "5911bc18a3b1f065dfff351d00190aba0e2f4e26"
EXPECTED_CONFIGURATION_IDENTITY = (
    "91e19ac5a7eb11c21ce0c38ab36cb60f948c420edc8ae0a1642e80095cb0eec6"
)
RECIPE_IDENTITY = "golden-digest-first-4000-accepted-steps-v1"
PRODUCER_INPUT_PATHS = (
    "cablp/solvers/_sim1d/core/config.py",
    "cablp/solvers/_sim1d/core/state.py",
    "cablp/solvers/_sim1d/results/io.py",
    "cablp/solvers/_sim1d/results/phase3_capture.py",
    "cablp/solvers/_sim1d/solver.py",
    "scripts/baseline_sim1d.py",
    "scripts/golden_digest_gate.py",
    "scripts/stances/g1atrim.toml",
    "scripts/baselines/golden_digest_4k.json",
    "scripts/baselines/production_discharge.json",
    PRODUCER_PATH,
)


def capture_phase3_rhs(run_id, capture_revision, invocation):
    """Run and capture the locked recipe after all fail-closed preflights."""
    run_id = validate_run_id(run_id)
    capture_revision = _verify_source_boundary(capture_revision)
    invocation = list(invocation)
    producer_blobs = _producer_blobs()

    # Import the recipe owners only after identity/output/source preflight.
    # Neither import constructs a solver; construction occurs below, after the
    # persistent allocation record has been installed.
    # scripts/ sibling imports: the seven purpose subdirectories on sys.path.
    import sys as _sys
    from pathlib import Path as _Path
    for _sub in ("atomic", "gates", "kinetic", "run", "score", "stance",
                 "verify"):
        _dir = str(_Path(__file__).resolve().parents[1] / _sub)
        if _dir not in _sys.path:
            _sys.path.insert(0, _dir)
    import baseline_sim1d
    import golden_digest_gate

    params, flags = baseline_sim1d.build_baseline_config(
        golden_digest_gate.DIGEST_PARAM_OVERRIDES
    )
    actual_identity = configuration_identity(params, flags)
    if actual_identity != EXPECTED_CONFIGURATION_IDENTITY:
        raise RuntimeError(
            f"resolved configuration identity {actual_identity} differs from "
            f"committed selection {EXPECTED_CONFIGURATION_IDENTITY}"
        )
    if flags.get("use_cached_neutral_seed") is not False:
        raise RuntimeError(
            "qualification recipe requires use_cached_neutral_seed=false"
        )
    if "CABLP_COMPILED_KERNELS" in os.environ:
        raise RuntimeError(
            "qualification recipe requires CABLP_COMPILED_KERNELS absent"
        )
    run_kwargs = dict(baseline_sim1d.BASELINE_RUN_KWARGS)
    run_kwargs["max_steps"] = golden_digest_gate.DIGEST_STEPS
    expected_controls = {
        "t_end": None,
        "dt": None,
        "operator_split": None,
        "max_steps": 4000,
    }
    if run_kwargs != expected_controls:
        raise RuntimeError(
            f"ordered run controls drifted: {run_kwargs!r} != {expected_controls!r}"
        )
    if golden_digest_gate.DIGEST_STEPS != 4000:
        raise RuntimeError("DIGEST_STEPS drifted from the locked 4000-step horizon")

    started_at = _utc_now()
    reserve_run_id(
        OUTPUT_DIRECTORY,
        run_id,
        {
            "allocated_at": started_at,
            "capture_revision": capture_revision,
            "capture_profile": PHASE3_CAPTURE_PROFILE,
            "invocation": invocation,
        },
    )

    # This is the first solver construction in the entry point.  The explicit
    # identity already exists durably and is supplied in the constructor.
    sim = baseline_sim1d.LAPDSim1D(params, flags, run_id=run_id)
    sim.start_simulation(**run_kwargs)
    result = sim.get_results()
    completed_at = _utc_now()
    return write_qualified_capture(
        OUTPUT_DIRECTORY,
        result,
        run_id=run_id,
        capture_revision=capture_revision,
        producer_path=PRODUCER_PATH,
        started_at=started_at,
        completed_at=completed_at,
        configuration_identity_sha256=EXPECTED_CONFIGURATION_IDENTITY,
        recipe_identity=RECIPE_IDENTITY,
        run_controls=run_kwargs,
        invocation=invocation,
        producer_blobs=producer_blobs,
        environment_lock={
            "path": "poetry.lock",
            "git_blob": _git("rev-parse", "HEAD:poetry.lock"),
        },
        repository_root=REPOSITORY_ROOT,
    )


def _verify_source_boundary(capture_revision):
    if not isinstance(capture_revision, str) or len(capture_revision) != 40:
        raise ValueError("capture_revision must be an explicit full Git commit")
    head = _git("rev-parse", "HEAD")
    if capture_revision != head:
        raise RuntimeError(
            f"capture revision mismatch: requested {capture_revision}, HEAD is {head}"
        )
    for advanced_ref in ("campaign", "origin/campaign"):
        advanced_revision = _git("rev-parse", advanced_ref)
        if advanced_revision != capture_revision:
            raise RuntimeError(
                f"{advanced_ref} is {advanced_revision}, not the authorized "
                f"capture revision {capture_revision}"
            )
    ancestor = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            MINIMUM_CAPTURE_ANCESTOR,
            head,
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
    )
    if ancestor.returncode != 0:
        raise RuntimeError(
            f"capture revision does not descend from {MINIMUM_CAPTURE_ANCESTOR}"
        )
    dirty = _git("status", "--porcelain=v1", "--untracked-files=no")
    if dirty:
        raise RuntimeError(f"tracked source tree is dirty:\n{dirty}")
    return head


def _producer_blobs():
    """Return a complete tracked package + locked recipe blob census.

    "Complete" means complete over what git TRACKS: the census is
    ``git ls-tree`` of ``cablp`` plus the locked recipe inputs, so a
    file that is not in the tree cannot appear here. The OPEN-ADAS ``.dat``
    data files under ``cablp/atomic/data/adas`` are untracked as of
    2026-08-26 and are therefore ABSENT from this census, even though the
    atomic rates they carry are an input to the captured RHS. Their
    integrity is not lost, it is carried elsewhere: the data-block sha256
    table in ``cablp/atomic/data/adas/README.md`` pins them, and that README
    IS tracked and so IS censused here. A capture's provenance record must
    be read with both halves together.
    """
    package_paths = _git(
        "ls-tree", "-r", "--name-only", "HEAD", "--", "cablp"
    ).splitlines()
    paths = sorted(set(package_paths) | set(PRODUCER_INPUT_PATHS))
    return {path: _git("rev-parse", f"HEAD:{path}") for path in paths}


def _git(*arguments):
    return subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--capture-revision", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    raw_argv = sys.argv[1:] if argv is None else list(argv)
    invocation = ["python", "scripts/capture_phase3_rhs.py", *raw_argv]
    h5_path, provenance_path = capture_phase3_rhs(
        args.run_id,
        args.capture_revision,
        invocation,
    )
    print(h5_path.relative_to(REPOSITORY_ROOT))
    print(provenance_path.relative_to(REPOSITORY_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
