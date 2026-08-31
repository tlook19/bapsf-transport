"""In-process import-provenance attestation for [perf-batch-1] worktree gates.

Imported (``-c "import scripts.pb1_provenance"``) or run as ``python -m`` before
a gate in the SAME process, this asserts that ``cablp`` resolved inside this
worktree rather than the main checkout (the editable install's ``cablp.pth``
can silently serve main-checkout code under a worktree ``PYTHONPATH``) and
prints the resolved path together with the loaded kernel identity.
"""

import os
import pathlib
import sys

WORKTREE = pathlib.Path(__file__).resolve().parent.parent


def attest():
    """Assert cablp resolves inside this worktree; print path and KERNEL_ID."""
    import cablp

    resolved = pathlib.Path(cablp.__file__).resolve()
    if WORKTREE not in resolved.parents:
        raise SystemExit(
            f"PROVENANCE FAIL: cablp.__file__={resolved} is outside the "
            f"worktree root {WORKTREE}"
        )
    from cablp.cathode.kernels import KERNEL_ID

    print(f"[provenance] cablp.__file__ = {resolved}")
    print(f"[provenance] KERNEL_ID      = {KERNEL_ID}")
    print(
        "[provenance] CABLP_COMPILED_KERNELS = "
        f"{os.environ.get('CABLP_COMPILED_KERNELS', '<unset>')}"
    )
    return resolved, KERNEL_ID


def main(argv):
    """Attest, then exec the gate named by ``argv`` in this same process."""
    attest()
    if not argv:
        return 0
    script = argv[0]
    sys.argv = list(argv)
    source = pathlib.Path(script).read_text()
    namespace = {"__name__": "__main__", "__file__": script}
    exec(compile(source, script, "exec"), namespace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
