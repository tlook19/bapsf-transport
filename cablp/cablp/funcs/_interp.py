"""Scalar linear interpolation with an explicitly FUSED lerp.

``numpy.interp`` computes its interior lerp as ``slope*(x - xp[j]) + fp[j]``,
and whether that expression rounds ONCE (a fused multiply-add) or TWICE is
decided by the compiler that built numpy, not by numpy's source: the x86-64
baseline numpy's linux wheels target has no FMA instruction, while the aarch64
base ISA does. The two forms disagree by 1 ULP on a few parts in ten thousand
of queries, which is enough to separate two implementations that are supposed
to be bit-identical.

:func:`interp_scalar_fused` removes the ambiguity by writing the fusion out
with :func:`math.fma`, so the result is the single-rounding answer on every
platform. The compiled kernel module transcribes the same expression with C's
``fma()``; both therefore agree with each other, and with a contracted
``numpy.interp``, by construction rather than by luck.
"""

import math

__all__ = ["interp_scalar_fused"]


def interp_scalar_fused(x, xp, fp, left=None, right=None):
    """``numpy.interp(x, xp, fp, left=left, right=right)`` for a scalar ``x``.

    Parameters
    ----------
    x : float
        The query point. Units are whatever ``xp`` carries. Coerced with
        ``float()``, so a non-scalar argument raises here rather than being
        interpolated elementwise.
    xp : sequence of float
        Monotonically increasing sample points, length >= 1. Not checked --
        as in ``numpy.interp``, a non-increasing ``xp`` gives an undefined
        answer rather than an error.
    fp : sequence of float
        Sample values, same length as ``xp``. Units are whatever the caller's
        table carries; this function is units-agnostic.
    left : float, optional
        Returned for ``x < xp[0]``. Defaults to ``fp[0]``.
    right : float, optional
        Returned for ``x > xp[-1]``. Defaults to ``fp[-1]``.

    Returns
    -------
    float
        The interpolated value.

    Notes
    -----
    Edge cases, all matching ``numpy.interp``:

    * ``x`` NaN returns ``x`` itself, before any table access.
    * ``x < xp[0]`` returns ``left``; ``x > xp[-1]`` returns ``right``. The
      comparisons are strict, so ``x`` exactly equal to either endpoint
      interpolates (and lands on that endpoint's ``fp``) rather than taking a
      fill value.
    * ``x`` exactly on a node returns that node's ``fp`` without evaluating a
      slope, which keeps a non-finite neighbour from contaminating an exact
      hit.
    * When the lerp is non-finite it is retried from the right-hand node, and
      a flat segment (``fp[j] == fp[j+1]``) falls back to that shared value.
    * ``len(xp) == 1`` is a three-way ``left`` / ``right`` / ``fp[0]`` answer
      with NO NaN short-circuit -- a NaN query returns ``fp[0]``, because both
      of numpy's comparisons are false for it. This is the one edge case where
      the single-node shape is not the general path's limit, and it is the
      shape's own rule, not an approximation of it.
    * ``len(xp) == 2`` has a single segment and is otherwise ordinary. Neither
      degenerate shape has a lerp to fuse for an out-of-range or on-node query,
      so agreement there is independent of the fusion.

    The interior lerp uses :func:`math.fma` and so rounds once. That is the
    single behavioural difference from writing ``slope*(x - xj) + fj``, and it
    is what makes the answer independent of how the interpreter's numpy was
    compiled.
    """
    n = len(xp)
    x = float(x)

    if n == 1:
        # numpy runs a separate loop for a single sample point, and it has no
        # NaN short-circuit: both comparisons are false for a NaN query, so it
        # falls through to fp[0] instead of propagating the NaN. Written out
        # because the general path below would propagate it.
        if x < xp[0]:
            return float(fp[0]) if left is None else float(left)
        if x > xp[0]:
            return float(fp[0]) if right is None else float(right)
        return float(fp[0])

    if math.isnan(x):
        return x
    if x > xp[n - 1]:
        return float(fp[n - 1]) if right is None else float(right)
    if x < xp[0]:
        return float(fp[0]) if left is None else float(left)

    # Largest j with xp[j] <= x, by the same bisection numpy's
    # binary_search_with_guess settles on.
    imin = 0
    imax = n
    while imin < imax:
        imid = imin + ((imax - imin) >> 1)
        if x >= xp[imid]:
            imin = imid + 1
        else:
            imax = imid
    j = imin - 1

    if j == n - 1:
        return float(fp[j])
    xj = float(xp[j])
    if xj == x:
        return float(fp[j])

    xj1 = float(xp[j + 1])
    fj = float(fp[j])
    fj1 = float(fp[j + 1])
    slope = (fj1 - fj) / (xj1 - xj)
    res = math.fma(slope, x - xj, fj)
    if math.isnan(res):
        res = math.fma(slope, x - xj1, fj1)
        if math.isnan(res) and fj == fj1:
            res = fj
    return res
