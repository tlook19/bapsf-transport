"""sp3 gate (a): flag-OFF bit-exactness, base HEAD vs branch, both paths.

ARTIFACT, not a repo gate. Runs a short production-stance point in a base
checkout and in the sp3 branch, on the pure AND compiled kernel paths, and
compares the final solver state at the RAW UINT64 level. The sp3 flag is
never armed: the question is whether a checkout that has merely GAINED the
capability still reproduces the trajectory of one that has never heard of it.

Child mode (one tree, one path) prints a JSON line; driver mode runs the four
children and compares.

    python scripts/sp3_bitexact_off.py --driver \
        --base <base tree root> --branch <branch tree root>
"""

import argparse
import json
import os
import subprocess
import sys


def child(tree, nx, max_steps, equilibration):
    sys.path.insert(0, os.path.join(tree, "cablp", "scripts"))
    sys.path.insert(0, os.path.join(tree, "cablp"))

    import numpy as np

    import cablp
    from cablp.funcs import _kernels as K
    from compare_sim1d_es1 import run_model

    extra = {
        "cathode_solver_model": "current_driven",
        "beam_deposition_model": "csda",
        "beam_anomalous_model": "quasilinear",
        "cathode_emission_profile": "gaussian",
        "cathode_warming_model": "power_balance",
        "cathode_sample_smoothing": "presheath",
        "gas_puff_mode": "square",
        "S_gp": 5200.0,
        "max_steps_action": "stop",
    }
    flags_extra = {"neutral_two_zone": True}
    if not equilibration:
        flags_extra["neutral_equilibration"] = False
    result, geometry, params, flags = run_model(
        nx=nx, extra=extra, flags_extra=flags_extra, max_steps=max_steps
    )
    y = np.ascontiguousarray(result.y[-1], dtype=float)
    print(json.dumps({
        "cablp_file": cablp.__file__,
        "provenance": K.PROVENANCE,
        "kernel_id": (
            None if K.COMPILED_KERNELS is None
            else str(K.COMPILED_KERNELS.KERNEL_ID)
        ),
        "requested": bool(K.compiled_kernels_requested()),
        "steps": int(result.steps),
        "cells": int(geometry.cells),
        "has_sp3_key": "nn0_profile" in params,
        "has_sp3_flag": "neutral_initial_profile" in flags,
        "y_uint64": y.view(np.uint64).tobytes().hex(),
    }))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--driver", action="store_true")
    p.add_argument("--tree")
    p.add_argument("--base")
    p.add_argument("--branch")
    p.add_argument("--nx", type=int, default=24)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--no-equilibration", action="store_true")
    args = p.parse_args(argv)

    if not args.driver:
        child(args.tree, args.nx, args.max_steps, not args.no_equilibration)
        return

    results = {}
    for tree_tag, tree in (("base", args.base), ("branch", args.branch)):
        for path_tag, optin in (("pure", None), ("compiled", "1")):
            env = dict(os.environ)
            env["PYTHONPATH"] = os.path.join(tree, "cablp")
            if optin is None:
                env.pop("CABLP_COMPILED_KERNELS", None)
            else:
                env["CABLP_COMPILED_KERNELS"] = optin
            cmd = [
                sys.executable, os.path.abspath(__file__),
                "--tree", tree, "--nx", str(args.nx),
                "--max-steps", str(args.max_steps),
            ]
            if args.no_equilibration:
                cmd.append("--no-equilibration")
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
            if proc.returncode != 0:
                raise SystemExit(
                    f"{tree_tag}/{path_tag} failed:\n{proc.stderr[-4000:]}"
                )
            results[tree_tag, path_tag] = json.loads(
                proc.stdout.strip().splitlines()[-1]
            )

    ok = True
    for tag, res in sorted(results.items()):
        print(
            f"{tag[0]:>6}/{tag[1]:<8} steps={res['steps']} cells={res['cells']} "
            f"provenance={res['provenance']} "
            f"sp3_key={res['has_sp3_key']} sp3_flag={res['has_sp3_flag']} "
            f"cablp={res['cablp_file']}"
        )
    # The branch must CARRY the capability and the base must NOT -- otherwise
    # the comparison is between two identical checkouts and proves nothing.
    for path_tag in ("pure", "compiled"):
        assert results["base", path_tag]["has_sp3_key"] is False
        assert results["base", path_tag]["has_sp3_flag"] is False
        assert results["branch", path_tag]["has_sp3_key"] is True
        assert results["branch", path_tag]["has_sp3_flag"] is True
        assert results["branch", path_tag]["steps"] > 0
    assert results["base", "compiled"]["kernel_id"] is not None
    assert results["branch", "compiled"]["kernel_id"] is not None
    assert results["base", "pure"]["kernel_id"] is None
    assert results["branch", "pure"]["kernel_id"] is None

    for path_tag in ("pure", "compiled"):
        same = (
            results["base", path_tag]["y_uint64"]
            == results["branch", path_tag]["y_uint64"]
        )
        ok = ok and same
        print(
            f"flag-off bit-exactness [{path_tag}]: "
            f"{'IDENTICAL' if same else 'DIFFER'} (raw uint64, "
            f"{len(results['base', path_tag]['y_uint64']) // 2} bytes)"
        )
    # ...and the compiled path really matched the pure one in each tree, which
    # is what makes "both paths" two independent statements.
    for tree_tag in ("base", "branch"):
        same = (
            results[tree_tag, "pure"]["y_uint64"]
            == results[tree_tag, "compiled"]["y_uint64"]
        )
        ok = ok and same
        print(
            f"compiled-vs-pure within {tree_tag}: "
            f"{'IDENTICAL' if same else 'DIFFER'}"
        )
    print("RESULT:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
