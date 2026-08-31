"""[perf-batch-1] item 1 -- bit-exactness proof for the ``he_rates`` coordinate share.

Two independent proofs that sharing one bilinear coordinate solve across the
``he_rates`` CALL SITES changes no floating-point operation:

``--ast``
    AST identity of the two functions that own the arithmetic,
    ``cablp.atomic.adas._interp_coords`` and ``._interp_blend``, against the
    pre-change source pinned verbatim in this file. The share is a memoization
    of their RESULTS; if either expression tree moved, the change was not a
    memoization and the raw-uint64 leg below is no longer a sufficient proof.

``--capture CORPUS.npz``
    Record every ``(ne, Te)`` argument pair ``he_rates`` is actually called
    with over N golden-config steps -- real solver states, not synthetic
    sweeps -- and write them to ``CORPUS.npz``.

``--reference CORPUS.npz REF.npz`` / ``--verify CORPUS.npz REF.npz``
    Evaluate every table in ``_HE_QUANTITIES`` at every captured pair, both
    with and without ``low_te_extension``, and write / compare the results as
    RAW uint64 (``ndarray.view(np.uint64)``), so the comparison is bitwise and
    not tolerance-based. ``--verify`` prints the number of differing uint64
    words; the gate is 0.

Artifacts (``*.npz``) are run outputs and are never committed; only this
script is.
"""

import argparse
import ast
import inspect
import sys
import textwrap
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from cablp.atomic import adas  # noqa: E402

# Verbatim pre-change source of the two arithmetic-owning helpers, at the
# batch base commit 75a2fa1. Compared by AST, so comments and formatting are
# free to move; an operand, an operator or an evaluation order is not.
BASE_INTERP_COORDS = '''
def _interp_coords(log_x_grid, log_y_grid, log_x, log_y):
    """Return clamped bilinear indices and weights for the shared grid."""
    x = np.clip(log_x, log_x_grid[0], log_x_grid[-1])
    y = np.clip(log_y, log_y_grid[0], log_y_grid[-1])
    ix = np.clip(np.searchsorted(log_x_grid, x) - 1, 0, log_x_grid.size - 2)
    iy = np.clip(np.searchsorted(log_y_grid, y) - 1, 0, log_y_grid.size - 2)
    fx = (x - log_x_grid[ix]) / (log_x_grid[ix + 1] - log_x_grid[ix])
    fy = (y - log_y_grid[iy]) / (log_y_grid[iy + 1] - log_y_grid[iy])
    return ix, iy, fx, fy
'''

BASE_INTERP_BLEND = '''
def _interp_blend(table, ix, iy, fx, fy):
    """Blend one table at precomputed bilinear coordinates."""
    c00 = table[iy, ix]
    c01 = table[iy, ix + 1]
    c10 = table[iy + 1, ix]
    c11 = table[iy + 1, ix + 1]
    return (
        c00 * (1.0 - fy) * (1.0 - fx)
        + c01 * (1.0 - fy) * fx
        + c10 * fy * (1.0 - fx)
        + c11 * fy * fx
    )
'''


def _strip_docstrings(tree):
    """Drop docstring nodes so a comment/docstring edit is not a difference."""
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            continue
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:]
    return tree


def _dump(source):
    return ast.dump(_strip_docstrings(ast.parse(textwrap.dedent(source).strip())))


def check_ast():
    """Assert the two arithmetic helpers are AST-identical to the base."""
    failures = []
    for func, base in (
        (adas._interp_coords, BASE_INTERP_COORDS),
        (adas._interp_blend, BASE_INTERP_BLEND),
    ):
        moved = _dump(inspect.getsource(func)) != _dump(base)
        if moved:
            failures.append(func.__name__)
        print(f"[ast] {func.__name__}: {'MOVED' if moved else 'identical'}")
    if failures:
        print(f"AST IDENTITY FAIL: {', '.join(failures)}")
        return 1
    print("AST IDENTITY OK: _interp_coords and _interp_blend unchanged")
    return 0


def capture(corpus_path, steps):
    """Record the real ``(ne, Te)`` pairs ``he_rates`` is called with."""
    from baseline_sim1d import BASELINE_RUN_KWARGS, build_baseline_config
    from cablp.solvers._sim1d import LAPDSim1D

    pairs_ne = []
    pairs_te = []
    original = adas.he_rates

    def recording(ne_cm3, Te_eV, quantities, low_te_extension=False):
        pairs_ne.append(np.asarray(ne_cm3, dtype=float).copy())
        pairs_te.append(np.asarray(Te_eV, dtype=float).copy())
        return original(ne_cm3, Te_eV, quantities, low_te_extension=low_te_extension)

    params, flags = build_baseline_config({"max_steps_action": "stop"})
    sim = LAPDSim1D(params, flags)
    kwargs = dict(BASELINE_RUN_KWARGS)
    kwargs["max_steps"] = int(steps)

    # Patch every module that bound the name at import time, plus the origin.
    from cablp.solvers._sim1d.physics import energy as _energy
    from cablp.solvers._sim1d.physics import reactions as _reactions
    from cablp.solvers._sim1d import solver as _solver

    holders = [adas, _energy, _reactions, _solver]
    for holder in holders:
        holder.he_rates = recording
    try:
        sim.start_simulation(**kwargs)
    finally:
        for holder in holders:
            holder.he_rates = original

    ne = np.concatenate(pairs_ne)
    te = np.concatenate(pairs_te)
    np.savez_compressed(corpus_path, ne=ne, Te=te, calls=np.array(len(pairs_ne)))
    print(
        f"corpus captured: {corpus_path} calls={len(pairs_ne)} pairs={ne.size} "
        f"ne=[{ne.min():.6e}, {ne.max():.6e}] Te=[{te.min():.6e}, {te.max():.6e}]"
    )
    return 0


def _evaluate_corpus(corpus_path):
    """Return {label: raw uint64 view} over every table and both extension arms."""
    data = np.load(corpus_path)
    ne = np.asarray(data["ne"], dtype=float)
    te = np.asarray(data["Te"], dtype=float)
    names = sorted(adas._HE_QUANTITIES)
    out = {}
    for extension in (False, True):
        # One fused call carrying EVERY quantity, and one call per quantity on
        # its own: the share must not make a table's value depend on which
        # other tables were asked for in the same call.
        fused = adas.he_rates(ne, te, names, low_te_extension=extension)
        for name in names:
            solo = adas.he_rates(ne, te, (name,), low_te_extension=extension)
            for tag, values in (("fused", fused[name]), ("solo", solo[name])):
                key = f"{name}|{'ext' if extension else 'clamp'}|{tag}"
                out[key] = np.ascontiguousarray(
                    np.asarray(values, dtype=float)
                ).view(np.uint64)
    return out


def reference(corpus_path, ref_path):
    """Write the raw-uint64 reference for the corpus at the current source."""
    values = _evaluate_corpus(corpus_path)
    np.savez_compressed(ref_path, **values)
    total = sum(v.size for v in values.values())
    print(f"reference written: {ref_path} arms={len(values)} words={total}")
    return 0


def verify(corpus_path, ref_path):
    """Recompute and compare against the reference, bitwise."""
    fresh = _evaluate_corpus(corpus_path)
    ref = np.load(ref_path)
    keys = sorted(set(fresh) | set(ref.files))
    differing = 0
    total = 0
    missing = []
    for key in keys:
        if key not in fresh or key not in ref.files:
            missing.append(key)
            continue
        a = np.asarray(ref[key], dtype=np.uint64)
        b = np.asarray(fresh[key], dtype=np.uint64)
        if a.shape != b.shape:
            print(f"  {key}: SHAPE {a.shape} vs {b.shape}")
            differing += max(a.size, b.size)
            continue
        bad = int(np.count_nonzero(a != b))
        total += a.size
        differing += bad
        if bad:
            idx = int(np.flatnonzero(a != b)[0])
            print(
                f"  {key}: {bad} differing; first at {idx} "
                f"ref={a[idx]:#018x} fresh={b[idx]:#018x}"
            )
    if missing:
        print(f"  MISSING ARMS: {missing}")
    print(
        f"raw-uint64 comparison: arms={len(keys)} words={total} "
        f"differing={differing}"
    )
    if differing or missing:
        print("BIT-EXACTNESS FAIL")
        return 1
    print("BIT-EXACTNESS OK: 0 differing uint64 over the corpus")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ast", action="store_true")
    ap.add_argument("--capture", metavar="CORPUS")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--reference", nargs=2, metavar=("CORPUS", "REF"))
    ap.add_argument("--verify", nargs=2, metavar=("CORPUS", "REF"))
    args = ap.parse_args(argv)

    status = 0
    if args.ast:
        status |= check_ast()
    if args.capture:
        status |= capture(args.capture, args.steps)
    if args.reference:
        status |= reference(*args.reference)
    if args.verify:
        status |= verify(*args.verify)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
