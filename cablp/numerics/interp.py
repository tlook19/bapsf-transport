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

:func:`interp_array_fused` is the same lerp for a whole array of queries, for
callers that evaluate one table at many points at once. numpy exposes no FMA
ufunc, so the single rounding is reconstructed with the error-free transforms
:func:`fma_array` is built from; the result is bit-identical to calling
:func:`interp_scalar_fused` on each element, so the two are interchangeable
and both belong to the bit-exactness gate inventory.
"""

import math
from bisect import bisect_right

import numpy as np

__all__ = [
    "interp_scalar_fused",
    "interp_array_fused",
    "fma_array",
    "check_fma_domain",
    "check_interp_table",
    "FMA_ARRAY_MAX_ABS",
]


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
    # binary_search_with_guess settles on. ``bisect_right`` splits on the
    # identical predicate -- it descends on ``x < xp[mid]``, the exact negation
    # of the ``x >= xp[mid]`` this search is defined by, and NaN is already
    # gone -- so it returns the same index, in C rather than in the
    # interpreter. The search is the whole cost of this function: it runs
    # ~200k times per solver step from the CSDA march.
    j = bisect_right(xp, x) - 1

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


# --- the same lerp, one table and many queries ------------------------------

#: Dekker's splitting constant, ``2**27 + 1``: it cuts a float64 significand
#: into two 26-bit halves whose products are each exactly representable.
_SPLITTER = float(2 ** 27 + 1)

#: Magnitude ceiling on every argument of :func:`fma_array`. Above it the
#: splitting step ``_SPLITTER * a`` or the partial product ``a*b`` can
#: overflow, and the error-free transforms below stop being error-free. It is
#: enforced rather than assumed, so an out-of-domain caller raises instead of
#: receiving a silently double-rounded answer.
FMA_ARRAY_MAX_ABS = 1e150


def _two_product(a, b):
    """``(p, e)`` with ``p = a*b`` rounded and ``a*b == p + e`` exactly.

    Dekker's algorithm. Exact for arguments within
    :data:`FMA_ARRAY_MAX_ABS`; ``e`` is the rounding error of the product and
    is zero when the product is exact.
    """
    p = a * b
    ca = _SPLITTER * a
    ah = ca - (ca - a)
    al = a - ah
    cb = _SPLITTER * b
    bh = cb - (cb - b)
    bl = b - bh
    return p, ((ah * bh - p) + ah * bl + al * bh) + al * bl


def _two_sum(a, b):
    """``(s, e)`` with ``s = a+b`` rounded and ``a + b == s + e`` exactly.

    Knuth's two-sum; unlike the "fast" variant it needs no ordering of the
    arguments by magnitude.
    """
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def fma_array(a, b, c):
    """Elementwise ``math.fma(a, b, c)``: ``a*b + c`` with ONE rounding.

    Parameters
    ----------
    a, b, c : ndarray of float64
        Broadcastable operands. Every element must be finite and within
        :data:`FMA_ARRAY_MAX_ABS` in magnitude.

    Returns
    -------
    ndarray of float64
        ``round_to_nearest(a*b + c)``, elementwise and bit-identical to
        :func:`math.fma` on the same operands.

    Raises
    ------
    ValueError
        If any operand is non-finite or exceeds :data:`FMA_ARRAY_MAX_ABS`.

    Notes
    -----
    numpy has no FMA ufunc and its x86-64 baseline does not contract
    ``a*b + c``, so the single rounding is reconstructed rather than emitted:
    ``a*b`` is split into an exact double-double by :func:`_two_product`, the
    high part is summed with ``c`` by :func:`_two_sum`, and the two residuals
    are combined and ROUNDED TO ODD before the final add. Round-to-odd on the
    tail is what makes the closing addition round exactly once overall -- a
    plain ``s + (t + t_err)`` would round twice and can disagree with
    ``math.fma`` at a tie.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    for name, arr in (("a", a), ("b", b), ("c", c)):
        check_fma_domain(arr, name)
    return _fma_array_unchecked(a, b, c)


def check_fma_domain(arr, name):
    """Raise unless every element of ``arr`` is a legal :func:`fma_array` operand.

    Split out so a caller with a FIXED operand -- an interpolation table, say
    -- can check it once instead of on every call.
    """
    arr = np.asarray(arr, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(
            f"fma_array operand {name!r} has a non-finite element; the "
            "error-free transforms it is built from are defined on finite "
            "operands only"
        )
    if arr.size and np.max(np.abs(arr)) > FMA_ARRAY_MAX_ABS:
        raise ValueError(
            f"fma_array operand {name!r} exceeds FMA_ARRAY_MAX_ABS="
            f"{FMA_ARRAY_MAX_ABS:g}; above it Dekker splitting overflows "
            "and the reconstruction is no longer exact"
        )


def _fma_array_unchecked(a, b, c):
    """:func:`fma_array` without the domain checks; operands are assumed legal."""
    p, p_err = _two_product(a, b)
    s, s_err = _two_sum(p, c)
    t, t_err = _two_sum(s_err, p_err)
    # Round the tail to odd: nudge ``t`` one ULP toward the residual whenever
    # the residual is non-zero and ``t``'s last significand bit is even.
    t = np.array(t, dtype=np.float64)
    nudge = (t_err != 0.0) & ((t.view(np.uint64) & np.uint64(1)) == 0)
    t = np.where(nudge, np.nextafter(t, np.copysign(np.inf, t_err)), t)
    return s + t


def interp_array_fused(x, xp, fp, left=None, right=None):
    """``numpy.interp(x, xp, fp, left=left, right=right)`` for an array ``x``.

    Parameters
    ----------
    x : ndarray of float64
        Query points, any shape. Must be finite.
    xp : ndarray of float64
        STRICTLY increasing sample points, length >= 2. Checked, because the
        vectorised path has no per-element fallback for a degenerate segment.
    fp : ndarray of float64
        Sample values, same length as ``xp``. Must be finite.
    left, right : float, optional
        Returned for ``x < xp[0]`` / ``x > xp[-1]``. Default ``fp[0]`` /
        ``fp[-1]``.

    Returns
    -------
    ndarray of float64
        The interpolated values, bit-identical elementwise to
        :func:`interp_scalar_fused` on the same table.

    Raises
    ------
    ValueError
        If ``xp`` is shorter than 2, not strictly increasing, if ``xp``/``fp``
        disagree in length, or if any input is non-finite.

    Notes
    -----
    The interior lerp is :func:`fma_array`, so it rounds once, exactly as the
    scalar form does. The scalar form's NaN-retry and flat-segment fallbacks
    are unreachable here: they exist for a non-finite or degenerate table, and
    both are refused above rather than repaired.
    """
    x = np.asarray(x, dtype=float)
    check_interp_table(xp, fp)
    if not np.all(np.isfinite(x)):
        raise ValueError(
            "interp_array_fused needs finite query points; the scalar form's "
            "NaN passthrough is deliberately not reproduced here"
        )
    return _interp_array_unchecked(x, np.asarray(xp, dtype=float),
                                   np.asarray(fp, dtype=float), left, right)


def check_interp_table(xp, fp):
    """Raise unless ``(xp, fp)`` is a legal :func:`interp_array_fused` table.

    Split out so a caller holding a FIXED table can check it once instead of
    on every lookup. The requirements are stricter than the scalar form's: a
    strictly increasing, finite table, at least two nodes.
    """
    xp = np.asarray(xp, dtype=float)
    fp = np.asarray(fp, dtype=float)
    n = xp.size
    if n < 2:
        raise ValueError(
            f"interp_array_fused needs at least 2 sample points (got {n}); "
            "the single-node shape has its own rule and is scalar-only"
        )
    if fp.size != n:
        raise ValueError(
            f"xp and fp must have the same length (got {n} and {fp.size})"
        )
    if not np.all(np.diff(xp) > 0.0):
        raise ValueError("interp_array_fused needs a strictly increasing xp")
    if not (np.all(np.isfinite(xp)) and np.all(np.isfinite(fp))):
        raise ValueError("interp_array_fused needs a finite table")
    check_fma_domain(fp, "fp")


def _interp_array_unchecked(x, xp, fp, left=None, right=None):
    """:func:`interp_array_fused` with the table and query checks hoisted out."""
    return _interp_array_unchecked_multi([(x, xp, fp, left, right)])[0]


def _interp_array_unchecked_multi(queries):
    """Several unchecked lookups, sharing ONE fused multiply-add.

    ``queries`` is a sequence of ``(x, xp, fp, left, right)``. Each is
    interpolated exactly as :func:`_interp_array_unchecked` would; the only
    difference is that the interior lerps are concatenated and rounded in a
    SINGLE :func:`fma_array` call. The FMA is elementwise, so concatenating
    changes no result -- it removes per-call numpy dispatch, which dominates at
    the array sizes the CSDA lane march runs at.

    Consecutive queries sharing the same ``x`` and ``xp`` OBJECTS reuse the
    first one's bracketing search, which is the two-table-one-grid case
    (sigma and sigma*E_rad on one energy grid).
    """
    parts = []
    slopes = []
    dxs = []
    fjs = []
    prev = None
    for x, xp, fp, left, right in queries:
        if prev is not None and prev[0] is x and prev[1] is xp:
            j, jj, xj = prev[2], prev[3], prev[4]
        else:
            n = xp.size
            j = np.searchsorted(xp, x, side="right") - 1
            jj = np.clip(j, 0, n - 2)
            xj = xp[jj]
            prev = (x, xp, j, jj, xj)
        fj = fp[jj]
        slopes.append((fp[jj + 1] - fj) / (xp[jj + 1] - xj))
        dxs.append(x - xj)
        fjs.append(fj)
        parts.append((x, xp, fp, left, right, j, xj, fj))
    if len(parts) == 1:
        lerps = [_fma_array_unchecked(slopes[0], dxs[0], fjs[0])]
    else:
        sizes = [np.size(v) for v in fjs]
        flat = _fma_array_unchecked(
            np.concatenate([np.broadcast_to(v, (s,))
                            for v, s in zip(slopes, sizes)]),
            np.concatenate([np.broadcast_to(v, (s,))
                            for v, s in zip(dxs, sizes)]),
            np.concatenate([np.broadcast_to(v, (s,))
                            for v, s in zip(fjs, sizes)]),
        )
        lerps = []
        at = 0
        for s in sizes:
            lerps.append(flat[at:at + s])
            at += s
    out = []
    for lerp, (x, xp, fp, left, right, j, xj, fj) in zip(lerps, parts):
        n = xp.size
        # On-node queries take the node's own value without evaluating a slope;
        # the last node and the two out-of-range fills are the scalar form's.
        res = np.where(xj == x, fj, lerp)
        res = np.where(j >= n - 1, fp[n - 1], res)
        lo = float(fp[0]) if left is None else float(left)
        hi = float(fp[n - 1]) if right is None else float(right)
        res = np.where(x < xp[0], lo, res)
        out.append(np.where(x > xp[n - 1], hi, res))
    return out
