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
    "FMA_ARRAY_MIN_ABS",
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

#: Magnitude FLOOR on every NONZERO element of :func:`fma_array`'s MULTIPLIED
#: arguments ``a`` and ``b`` -- the lower half of the domain
#: :data:`FMA_ARRAY_MAX_ABS` bounds from above. The addend ``c`` is exempt: it
#: is never multiplied.
#:
#: Dekker's two-product is error-free only ABSENT UNDERFLOW. The precondition
#: is on the exponents of the factors -- ``e_a + e_b >= e_min + p - 1`` (Muller
#: et al., *Handbook of Floating-Point Arithmetic*, 2nd ed. 4.4.1) -- which for
#: binary64 (``e_min = -1022``, ``p = 53``) is ``|a*b| >= 2**-970 ~
#: 1.0021e-292``. Below that the split partial products ``al*bl`` and friends
#: round in the subnormal range, ``e`` stops being the whole rounding error of
#: ``a*b``, and :func:`fma_array` silently stops agreeing with
#: :func:`math.fma`.
#:
#: The floor is the cheap conservative form of that condition, applied to each
#: operand separately: with every nonzero element at or above ``1e-145``, the
#: product of two of them is at or above ``1e-290``, clearing ``2**-970`` by a
#: factor of ~100; a zero operand needs no floor because ``0*x`` is exactly
#: ``0``. Veltkamp's splitter cannot underflow on it either
#: (``_SPLITTER * 1e-145 ~ 1.3e-137``), and the value mirrors the ceiling --
#: ``1e150**2 = 1e300`` sits under the overflow threshold exactly as
#: ``1e-145**2 = 1e-290`` sits over the underflow one.
#:
#: It is NON-BINDING on everything the CSDA lane march can reach. Over the
#: committed ``deposit_beam`` corpus plus 2,000 randomized lane batteries
#: (5,889,066 operand elements, ``scripts/r3fma_domain_probe.py`` (at commit
#: 48be9a4, retired 2026-09-03)) the smallest
#: nonzero operand magnitude was ``4.66e-23`` and the smallest nonzero
#: ``|a*b|`` was ``9.45e-28`` -- 122 and 265 decades above this floor and the
#: exactness threshold respectively.
FMA_ARRAY_MIN_ABS = 1e-145


def _two_product(a, b):
    """``(p, e)`` with ``p = a*b`` rounded and ``a*b == p + e`` exactly.

    Dekker's algorithm. Exact for arguments inside :func:`check_fma_domain`'s
    domain -- within :data:`FMA_ARRAY_MAX_ABS`, so the split cannot overflow,
    and, when nonzero, at or above :data:`FMA_ARRAY_MIN_ABS`, so the product
    cannot underflow. There ``e`` is the rounding error of the product and is
    zero when the product is exact. Outside either fence ``e`` is NOT the whole
    error and the caller gets a double-rounded answer.
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
    a, b : ndarray of float64
        The MULTIPLIED operands, inside the checked domain: every element
        finite, and every nonzero element between :data:`FMA_ARRAY_MIN_ABS`
        and :data:`FMA_ARRAY_MAX_ABS` in magnitude.
    c : ndarray of float64
        The addend, broadcastable against ``a*b``. Finite and within
        :data:`FMA_ARRAY_MAX_ABS`; it carries NO lower fence, because it is
        never multiplied.

    Returns
    -------
    ndarray of float64
        ``round_to_nearest(a*b + c)``, elementwise and bit-identical to
        :func:`math.fma` FOR OPERANDS WITHIN THE CHECKED DOMAIN. Both fences
        are enforced on every call, so an out-of-domain operand raises rather
        than returning a silently different answer.

    Raises
    ------
    ValueError
        If any operand is non-finite or exceeds :data:`FMA_ARRAY_MAX_ABS`, or
        if ``a`` or ``b`` has a nonzero element below
        :data:`FMA_ARRAY_MIN_ABS`.

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

    The one place the reconstruction is NOT :func:`math.fma` is where its
    building block stops being error-free: **when the product ``a*b``
    underflows into the subnormal range**. :func:`_two_product` is Dekker's,
    whose residual is the true rounding error of ``a*b`` only while the split
    partial products stay normal (``|a*b| >= 2**-970``); below that ``e`` loses
    bits and the answer can differ from :func:`math.fma`. It is a real
    difference, not a theoretical one: on triples whose product is forced below
    ``2**-970`` and whose addend exposes the residual, 64,599 of 300,000
    disagree. :data:`FMA_ARRAY_MIN_ABS` is the fence that keeps callers out of
    it. Three neighbouring cases are NOT affected and are deliberately left
    inside the domain: a subnormal ADDEND
    (``c`` only ever passes through the error-free :func:`_two_sum`), normal
    products that CANCEL into a subnormal SUM (the closing add is a plain
    rounding), and zero factors (``0*x`` is exactly ``0``, which is why the
    floor exempts zeros). All three measure 0 differing from :func:`math.fma`;
    ``scripts/r3fma_underflow_fence.py`` is the reproduction.

    The SIGNED ZERO of an exact-zero product is the one remaining departure,
    and it is a sign question only. **When ``a*b`` is an exact zero, the sign
    of a resulting ``±0.0`` may differ from :func:`math.fma`**: the closing
    two-sum add loses the zero's sign, so ``(+0.0)*(-x) + (-0.0)`` returns
    ``+0.0`` where :func:`math.fma` returns ``-0.0``. It is left in place, for
    three reasons. It is NUMERICALLY INERT -- ``+0.0 == -0.0``, and the two
    agree under every arithmetic and comparison operation except ``signbit``,
    ``copysign``, ``atan2`` and division, none of which any caller applies to
    an :func:`fma_array` result. It is IN-DOMAIN rather than fenced, because
    zeros are deliberately exempt from :data:`FMA_ARRAY_MIN_ABS` for the reason
    just given (``0*x`` is exactly ``0`` and cannot underflow), and fencing
    them would refuse legal calls to suppress a sign bit. And it is
    UNREACHABLE on the march: over the committed ``deposit_beam`` corpus plus
    randomized lane batteries, the MULTIPLIED operand streams ``a`` and ``b``
    contain ZERO exact zeros (``scripts/r3fma_domain_probe.py``, which counts
    them per operand), so no reachable call forms an exact-zero product at all.
    Disclosed rather than fixed, and it is the LAST known gap between this
    reconstruction and :func:`math.fma`.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    for name, arr, factor in (("a", a, True), ("b", b, True), ("c", c, False)):
        check_fma_domain(arr, name, factor=factor)
    return _fma_array_unchecked(a, b, c)


def check_fma_domain(arr, name, *, factor=True):
    """Raise unless every element of ``arr`` is a legal :func:`fma_array` operand.

    Two fences, one on each end of Dekker's error-free window: no element above
    :data:`FMA_ARRAY_MAX_ABS`, where the split overflows, and no NONZERO
    element below :data:`FMA_ARRAY_MIN_ABS`, where a product of two such
    operands underflows. Exact zeros pass both, since ``0*x`` is exactly ``0``.

    Parameters
    ----------
    arr : ndarray of float64
        The operand to check.
    name : str
        Its name in the caller's signature, for the error message.
    factor : bool, optional
        Whether ``arr`` is one of the MULTIPLIED operands. The floor applies
        only to those: the addend ``c`` reaches the answer through
        :func:`_two_sum`, which is error-free for any finite operands, so a
        tiny or even subnormal ``c`` is legal and is NOT fenced (measured: 0
        differing from :func:`math.fma` over 300,000 subnormal-addend triples,
        ``scripts/r3fma_underflow_fence.py``). Fencing it as well would refuse
        a call the reconstruction handles exactly.

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
    if arr.size:
        mag = np.abs(arr)
        if np.max(mag) > FMA_ARRAY_MAX_ABS:
            raise ValueError(
                f"fma_array operand {name!r} exceeds FMA_ARRAY_MAX_ABS="
                f"{FMA_ARRAY_MAX_ABS:g}; above it Dekker splitting overflows "
                "and the reconstruction is no longer exact"
            )
        if factor:
            nonzero = mag[mag > 0.0]
            if nonzero.size and np.min(nonzero) < FMA_ARRAY_MIN_ABS:
                raise ValueError(
                    f"fma_array operand {name!r} has a nonzero element of "
                    f"magnitude {float(np.min(nonzero)):g}, below "
                    f"FMA_ARRAY_MIN_ABS={FMA_ARRAY_MIN_ABS:g}: the Dekker "
                    "two-product is error-free only absent underflow "
                    "(|a*b| >= 2**-970 ~ 1.0e-292), and below this floor a "
                    "product of two operands can land in the subnormal "
                    "range, where fma_array stops matching math.fma. The "
                    "reachable CSDA march domain clears the floor by 122 "
                    "decades (smallest nonzero operand 4.66e-23, smallest "
                    "nonzero |a*b| 9.45e-28, 265 decades above the exactness "
                    "threshold), so an operand that trips this fence is a "
                    "caller defect, not a tight bound"
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
    strictly increasing, finite table, at least two nodes, with BOTH axes
    inside :func:`check_fma_domain`'s magnitude domain. ``xp`` is checked
    because it reaches the fused multiply-add as well as ``fp`` does -- the
    lerp's ``b`` operand is ``x - xp[j]`` and its ``a`` operand carries
    ``xp[j+1] - xp[j]`` in the denominator -- so a table checked here needs no
    per-lookup check in the hot loop.
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
    check_fma_domain(xp, "xp")
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
