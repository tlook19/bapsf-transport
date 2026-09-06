"""Tube-beamed gas-puff launch row derived from the CAD port geometry.

The KINETIC neutral instruments (`mc_neutrals.py` TPMC, `kn2zone.py`, and the
DVM-side consumers of `kinetic_neutrals.puff_launch_bins`) place the puff by an
AXIAL ROW of birth rates, `bg["sources"]["puff_cells"]`, and then transport the
atoms themselves. What that row must carry is therefore the INJECTION geometry
-- where gas from the feed line first reaches the plasma column -- not a
deposition envelope. The fluid `gas_puff_profile = "cosine_pipe"` row is the
other thing: with no neutral transport of its own the fluid model has to spread
the source itself, so its `throw_cm` is an end-state closure.

TWO CONSUMERS, ONE ROW. `gas_puff_profile = "orifice"` hands this same row to
the FLUID solver as its per-cell deposition profile, through the shared
`neutrals.gas_puff_rate_profile`. That is a DISCLOSED CLOSURE, not an identity:
a kinetic FIRST-FLIGHT row is being read as a fluid DEPOSITION row, so the
transport the kinetic engines would apply after first arrival is absent, and
what the fluid model then spreads is the injection footprint rather than an
end-state envelope. The substitution is deliberate -- it puts the foot fill,
the equilibrated seed and the in-solver puff on ONE shape at one geometry --
and the difference against `"cosine_pipe"` is a registered finding, not an
error. This module is where that disclosure is stated.

GEOMETRY (all CAD unless marked). Gas enters at the anode-stack station through
two azimuthally opposed mid-plane ports. It arrives there through a feed line
that adapts the wide port stub down to CF35/KF40 pipe, so the collimating
element is the NARROW PIPE and the effective source aperture is the pipe exit
at the vessel wall. A pipe of inner diameter ``d`` and length ``L`` in
free-molecular flow does not emit with the thin-aperture cosine law: it BEAMS,
with an exit distribution narrower than cosine that sharpens as the aspect
ratio ``Gamma = L / d`` grows.

THE ANGULAR DISTRIBUTION is the standard transparent-regime (collisionless)
Clausing long-tube form, in the Olander-Kruger parameterisation as collected by
Ashkarin et al., arXiv:2605.12212 Eqs. (17), (22)-(25), whose ``ftr`` reduces to
the Clausing model under the long-tube end-effect prescription ``alpha =
2/(3 Gamma)``, ``zeta_0 = alpha``, ``zeta_1 = 1 - alpha``.  With

    q      = Gamma tan(theta),      theta_o = arctan(1 / Gamma)
    R(q)   = arccos(q) - q sqrt(1 - q^2)                        (q <= 1)

the normalised angular intensity (``f(0) = 1``) is

    f(theta) = zeta_0 cos(theta)
             + (2/pi) cos(theta) [ (1 - zeta_0) R(q)
                                   + (zeta_1 - zeta_0) (2/(3q))
                                     (1 - (1 - q^2)^(3/2)) ]     q <= 1

    f(theta) = zeta_0 cos(theta)
             + (4/(3 pi q)) (zeta_1 - zeta_0) cos(theta)         q >= 1

(the two branches agree at ``q = 1``). SOURCE CLASS: analytic literature
result for free-molecular flow through a straight tube with diffuse wall
re-emission; no fitted quantity enters it, and its only input is ``Gamma``.
Its own stated regime of best justification is ``Gamma >~ 10``, above the
aspect ratios reached here -- see the disclosure list below.

THE ROW is then pure ray optics: emit uniformly over the pipe-exit disc at the
vessel wall, weight directions by ``f(theta) sin(theta)``, fly straight, and
record the axial coordinate at which each ray first reaches the plasma-column
radius (its perigee, for a ray whose closest approach stays outside the
column -- the two definitions coincide at grazing, so the row is continuous
across that boundary).

THE BRACKET. The feed pipe is not in the CAD export, so its two numbers are
hardware brackets rather than pins: inner diameter ``d`` in [3.8, 4.1] cm
(CF35/KF40 class) and length ``L >= 22 cm``, the latter ONE-SIDED, from the
cathode-side yellow coil stack the flange must clear. A one-sided length bound
gives a one-sided spread bracket, and the endpoints are unambiguous because
both numbers push the footprint the same way:

    WIDE   d = 4.1 cm, L = 22 cm   (Gamma ~ 5.37) -- the WIDEST footprint the
                                    hardware permits; a longer pipe only
                                    narrows it.
    NARROW d = 3.8 cm, L -> inf                   -- the fully-beamed limit,
                                    where the row degenerates to the geometric
                                    image of the aperture.

Both endpoints are returned together: a consumer takes the row it runs plus its
bracket partner, so the spread is always disclosed as a bracket and never
quoted as a single derived number.

DISCLOSURES (all live, none of them free parameters):

* The ~10.2 cm port bore is a SHORT NON-LIMITING STUB. The CAD gives it a
  through-wall length equal to the 0.952 cm shell thickness, aspect ratio
  ~0.09, so it collimates nothing; the pipe behind it sets the distribution.
* ``Gamma`` at the wide endpoint is ~5.4, below the ``Gamma >~ 10`` where the
  Clausing closure's linear-density assumption is best justified. The error
  runs toward the cosine law, i.e. toward a WIDER footprint, which is the
  endpoint that is already labelled the widest.
* The two mid-plane ports sit at the SAME axial station on opposite sides of
  the machine, so they contribute no extra axial span: rotating one onto the
  other leaves both the plasma cylinder and z unchanged, and the two rows are
  identical. The "two-port span" is the station's own axial extent.
* The kinetic engines launch the row's atoms with a WALL (cosine) velocity
  spectrum. This module folds the beaming into the axial placement only; the
  birth velocity distribution is unchanged, so the beam's forward momentum is
  not represented.
* The flight is treated as collisionless from the pipe exit to the column.
  Puff-phase densities are low enough for that between the wall and the
  column, and everything downstream of first arrival belongs to the engine.
"""

import functools

import numpy as np

# ---------------------------------------------------------------- CAD pins
#
# All from the engineer-supplied SolidWorks export of 2026-08-18 (assembly
# LAPD_TomLook-Magnets.STEP, per-part global bounding boxes in
# tree_bbox_mag.txt, analytic cylindrical
# faces in part_measurements.txt).  Model z = (-4560 mm - CAD z) / 1000, i.e.
# distance from the LaB6 emitting face.

#: Axial centre of the injection station [cm].  ``940206.STEP``, the port weld
#: collar, appears three times at this station -- azimuth 0 deg, 90 deg and
#: 180 deg -- each spanning z_model 0.8121-0.9137 m.  The two mid-plane
#: copies (0 deg, 180 deg) are the gas-puff ports; the four 45 deg-offset
#: ports at the same station carry the anode bias feeds.  This value is also
#: the shipped ``gas_puff_z_cm``.
PORT_CENTER_Z_CM = 86.3

#: Full axial extent of the port collar [cm] (the "two-port span" of record).
PORT_SPAN_Z_CM = (81.21, 91.37)

#: Port bore diameter [cm].  ``Main-cyl`` carries through-wall cylindrical
#: faces ``r = 50.86 mm`` on the mid-plane at both x = +-409.6 mm.  Short and
#: non-limiting: its length is the 0.952 cm shell thickness.
PORT_BORE_DIAMETER_CM = 10.172

#: Vessel inner radius at the injection station [cm].  ``Main-cyl`` (the source
#: chamber) has analytic inner/outer cylinder faces r = 400.05 / 409.57 mm.
#: The stance's ``machine_radius_profile_cm`` carries 40.0 cm in these cells,
#: so model and machine already agree here.
WALL_RADIUS_CM = 40.005

#: Feed-pipe inner diameter bracket [cm].  NOT in the CAD -- nothing is modelled
#: outboard of the port pad ``110026.STEP`` (r = 429.4 mm) at either mid-plane
#: azimuth -- so the CF35/KF40 class bracket stands as ruled.
PIPE_ID_CM_BRACKET = (3.8, 4.1)

#: Feed-pipe length lower bound [cm], ONE-SIDED.  The cathode-side yellow coil
#: stack (``Magnet-yellow``, r_in 565.1 / r_out 720.7 mm) is the ruler: the
#: flange sits outboard of it.  Measured from the 500 mm main-chamber radius
#: this gives 22 cm; measured from the 400.05 mm source-chamber wall the port
#: actually sits in it would give 32 cm, so 22 cm is the conservative (widest)
#: choice.  The coils do not obstruct the port -- the station at 0.863 m lies
#: in the gap between the source frames (end 0.747 m) and the yellow stack
#: (start 1.112 m).
PIPE_LENGTH_CM_MIN = 22.0

#: Aspect ratio standing in for ``L -> inf`` at the narrow endpoint.  The row
#: there is the geometric image of the aperture and is insensitive to the
#: exact value: the off-axis wing carries weight O(1/Gamma).
NARROW_ASPECT_RATIO = 1.0e4

#: Quadrature node counts: (aperture radial, aperture azimuthal, polar inside
#: the direct cone, polar outside it, direction azimuthal).
QUADRATURE = (8, 16, 200, 400, 180)


# --------------------------------------------------- the angular distribution


def clausing_intensity(theta, aspect_ratio):
    """Normalised Clausing angular intensity ``f(theta)``, ``f(0) = 1``.

    ``theta`` [rad] is measured from the tube axis and may be an array;
    ``aspect_ratio`` is ``Gamma = L / d``.  Returns the differential flux per
    unit solid angle relative to the axial value, for the transparent
    (collisionless) molecular-flow regime with diffuse wall re-emission.  The
    formula and its source class are stated in the module docstring.

    Raises ``ValueError`` for a non-positive aspect ratio, and for a
    ``theta`` outside the forward hemisphere, where the expression has no
    meaning.
    """
    gamma = float(aspect_ratio)
    if not np.isfinite(gamma) or gamma <= 0.0:
        raise ValueError(
            f"aspect_ratio must be finite and positive (got {aspect_ratio!r}); "
            "it is Gamma = L / d of the collimating pipe."
        )
    if gamma < 4.0 / 3.0:
        raise ValueError(
            f"aspect_ratio must be >= 4/3 (got {gamma}): below it the long-tube "
            "end-effect prescription zeta_0 = 2/(3 Gamma), zeta_1 = 1 - zeta_0 "
            "inverts (zeta_1 < zeta_0) and the intensity goes negative. This "
            "expression is a LONG-tube result and has no short-tube branch."
        )
    th = np.asarray(theta, dtype=float)
    if np.any(th < 0.0) or np.any(th > 0.5 * np.pi):
        raise ValueError(
            "theta must lie in [0, pi/2]: the Clausing intensity is defined "
            "on the forward hemisphere of the tube exit only."
        )
    alpha = 2.0 / (3.0 * gamma)
    zeta0, zeta1 = alpha, 1.0 - alpha
    cos_t = np.cos(th)
    q = gamma * np.tan(th)
    out = zeta0 * cos_t
    near = q <= 1.0
    qn = np.where(near, q, 1.0)  # keep the sqrt real off-branch
    root = np.sqrt(np.maximum(1.0 - qn**2, 0.0))
    # (2/(3q))(1 - (1-q^2)^{3/2}) -> q as q -> 0; take the limit explicitly.
    tiny = qn < 1.0e-8
    wall_near = np.where(
        tiny, qn, (2.0 / (3.0 * np.where(tiny, 1.0, qn))) * (1.0 - root**3)
    )
    direct = np.arccos(np.clip(qn, -1.0, 1.0)) - qn * root
    out = out + np.where(
        near,
        (2.0 / np.pi) * cos_t * ((1.0 - zeta0) * direct + (zeta1 - zeta0) * wall_near),
        (4.0 / (3.0 * np.pi * np.where(near, 1.0, q))) * (zeta1 - zeta0) * cos_t,
    )
    return out


# ------------------------------------------------------------- the launch row


def _first_arrival_z(z_emit, offset_y, cos_t, sin_t, cos_p, sin_p, r_wall, r_edge):
    """Axial coordinate where one ray reaches the plasma column.

    The ray leaves ``(r_wall, offset_y, z_emit)`` in machine coordinates with
    direction ``(-cos_t, sin_t cos_p, sin_t sin_p)``.  Returns the axial
    coordinate of the first crossing of ``r = r_edge``, or of the perigee when
    the ray stays outside the column, together with a boolean marking the
    crossing rays.  All arguments broadcast.
    """
    # radial quadratic: a t^2 + 2 b t + c = |r|^2 - r_edge^2
    a = cos_t**2 + (sin_t * cos_p) ** 2
    b = -r_wall * cos_t + offset_y * sin_t * cos_p
    c = r_wall**2 + offset_y**2 - r_edge**2
    disc = b**2 - a * c
    root = np.sqrt(np.maximum(disc, 0.0))
    t_hit = (-b - root) / a
    t_perigee = -b / a
    hit = (disc > 0.0) & (t_hit > 0.0)
    t = np.where(hit, t_hit, np.maximum(t_perigee, 0.0))
    return z_emit + t * sin_t * sin_p, hit


def launch_row(
    z_edges_cm,
    *,
    pipe_id_cm,
    aspect_ratio,
    r_wall_cm=WALL_RADIUS_CM,
    r_edge_cm,
    z_port_cm=PORT_CENTER_Z_CM,
    quadrature=QUADRATURE,
):
    """Per-cell axial launch weights on ``z_edges_cm``, summing to 1.

    ``z_edges_cm`` are the consuming engine's own cell edges [cm] in the frame
    the port position is quoted in.  ``pipe_id_cm`` and ``aspect_ratio`` fix the
    source aperture and its beaming; ``r_wall_cm`` and ``r_edge_cm`` are the
    vessel wall and the plasma-column radius [cm] that bound the flight.

    Returns ``(weights, meta)``.  ``weights`` is one non-negative entry per
    cell.  ``meta`` reports the fractions that are diagnostic rather than
    physical -- ``clipped`` (rays landing off the grid, folded into the end
    cells so no fuel is deleted) and ``missed`` (rays whose closest approach
    stays outside the column, placed at their perigee) -- plus the aspect
    ratio and the direct-cone half-angle actually used.

    Raises ``ValueError`` on a grid that is not increasing, on a non-positive
    aperture or radius, on a plasma radius that is not inside the wall, and on
    a port position outside the grid: each would place fuel somewhere the
    geometry does not describe.
    """
    edges = np.asarray(z_edges_cm, dtype=float)
    if edges.ndim != 1 or edges.size < 2 or not np.all(np.diff(edges) > 0.0):
        raise ValueError(
            "z_edges_cm must be a strictly increasing 1-D array of cell edges."
        )
    d = float(pipe_id_cm)
    r_wall = float(r_wall_cm)
    r_edge = float(r_edge_cm)
    z0 = float(z_port_cm)
    if not d > 0.0:
        raise ValueError(f"pipe_id_cm must be positive (got {pipe_id_cm!r}).")
    if not 0.0 < r_edge < r_wall:
        raise ValueError(
            f"need 0 < r_edge_cm < r_wall_cm (got r_edge_cm={r_edge_cm!r}, "
            f"r_wall_cm={r_wall_cm!r}): the puff flies inward from the vessel "
            "wall to the plasma column, so the column must lie inside it."
        )
    if not edges[0] <= z0 <= edges[-1]:
        raise ValueError(
            f"z_port_cm={z0} lies outside the grid [{edges[0]}, {edges[-1]}]: "
            "the injection station must be on the consuming engine's mesh."
        )
    n_u, n_psi, n_th_in, n_th_out, n_phi = (int(v) for v in quadrature)
    if min(n_u, n_psi, n_th_in, n_th_out, n_phi) < 1:
        raise ValueError(f"quadrature node counts must be >= 1 (got {quadrature!r}).")

    gamma = float(aspect_ratio)
    theta_o = np.arctan(1.0 / gamma)

    # exit-disc nodes: equal-area midpoints in (rho^2, psi)
    u = (np.arange(n_u) + 0.5) / n_u
    psi = 2.0 * np.pi * (np.arange(n_psi) + 0.5) / n_psi
    rho = 0.5 * d * np.sqrt(u)
    a_y = np.ravel(rho[:, None] * np.sin(psi)[None, :])
    b_z = np.ravel(rho[:, None] * np.cos(psi)[None, :])

    phi = 2.0 * np.pi * (np.arange(n_phi) + 0.5) / n_phi
    cos_p, sin_p = np.cos(phi), np.sin(phi)

    # polar nodes: midpoints, split at the direct-cone edge where f kinks
    d_in = theta_o / n_th_in
    d_out = (0.5 * np.pi - theta_o) / n_th_out
    thetas = np.concatenate(
        (
            d_in * (np.arange(n_th_in) + 0.5),
            theta_o + d_out * (np.arange(n_th_out) + 0.5),
        )
    )
    d_theta = np.concatenate((np.full(n_th_in, d_in), np.full(n_th_out, d_out)))
    f_theta = clausing_intensity(thetas, gamma)

    row = np.zeros(edges.size - 1)
    total = 0.0
    clipped = 0.0
    missed = 0.0
    disc_w = 1.0 / a_y.size
    for th, dth, ft in zip(thetas, d_theta, f_theta):
        w = ft * np.sin(th) * dth * (2.0 * np.pi / n_phi) * disc_w
        if w <= 0.0:
            continue
        z_land, hit = _first_arrival_z(
            (z0 + b_z)[:, None],
            a_y[:, None],
            np.cos(th),
            np.sin(th),
            cos_p[None, :],
            sin_p[None, :],
            r_wall,
            r_edge,
        )
        z_flat = np.ravel(z_land)
        n_ray = z_flat.size
        idx = np.searchsorted(edges, z_flat, side="right") - 1
        off = (idx < 0) | (idx >= row.size)
        clipped += w * float(np.count_nonzero(off))
        missed += w * float(np.count_nonzero(~hit))
        row += w * np.bincount(
            np.clip(idx, 0, row.size - 1), minlength=row.size
        ).astype(float)
        total += w * n_ray
    if total <= 0.0:  # unreachable for a valid Gamma; never return a silent zero
        raise ValueError(
            f"the Clausing quadrature collected no flux at Gamma={gamma}: "
            "the angular distribution integrated to zero."
        )
    row /= total
    meta = {
        "aspect_ratio": gamma,
        "theta_o_deg": float(np.degrees(theta_o)),
        "clipped_fraction": clipped / total,
        "missed_fraction": missed / total,
        "pipe_id_cm": d,
        "r_wall_cm": r_wall,
        "r_edge_cm": r_edge,
        "z_port_cm": z0,
    }
    return row, meta


def launch_row_bracket(
    z_edges_cm,
    *,
    r_edge_cm,
    r_wall_cm=WALL_RADIUS_CM,
    z_port_cm=PORT_CENTER_Z_CM,
    quadrature=QUADRATURE,
):
    """Both endpoints of the one-sided hardware bracket, as one object.

    Returns ``{"wide": (row, meta), "narrow": (row, meta)}``.  ``"wide"`` is the
    widest axial footprint the feed line permits -- the largest bracketed pipe
    bore at the shortest bracketed length -- and ``"narrow"`` is the
    fully-beamed ``L -> inf`` limit at the smallest bore.  Consumers take the
    endpoint they run TOGETHER with its partner: the derived spread is a
    bracket, and quoting one endpoint alone would state a pin the hardware
    does not support.
    """
    d_lo, d_hi = PIPE_ID_CM_BRACKET
    common = dict(
        r_wall_cm=r_wall_cm,
        r_edge_cm=r_edge_cm,
        z_port_cm=z_port_cm,
        quadrature=quadrature,
    )
    return {
        "wide": launch_row(
            z_edges_cm,
            pipe_id_cm=d_hi,
            aspect_ratio=PIPE_LENGTH_CM_MIN / d_hi,
            **common,
        ),
        "narrow": launch_row(
            z_edges_cm,
            pipe_id_cm=d_lo,
            aspect_ratio=NARROW_ASPECT_RATIO,
            **common,
        ),
    }


# ------------------------------------------------- the run-constant accessor


@functools.lru_cache(maxsize=8)
def _launch_row_cached(
    edges_bytes, pipe_id_cm, aspect_ratio, r_wall_cm, r_edge_cm, z_port_cm
):
    """Memoised :func:`launch_row`, keyed on the raw bytes of the cell edges.

    Returns a READ-ONLY array: it is shared between every caller that asks for
    the same grid and geometry, so a mutation would reach all of them.
    """
    row, _meta = launch_row(
        np.frombuffer(edges_bytes, dtype=float),
        pipe_id_cm=pipe_id_cm,
        aspect_ratio=aspect_ratio,
        r_wall_cm=r_wall_cm,
        r_edge_cm=r_edge_cm,
        z_port_cm=z_port_cm,
    )
    row.setflags(write=False)
    return row


def launch_row_for_grid(
    geometry, *, pipe_id_cm, pipe_length_cm, z_port_cm
):
    """The launch row for one solver geometry, as per-cell mass fractions.

    The flight bounds are read off the grid itself at the cell containing
    ``z_port_cm``: the plasma-column radius ``Rp_cm`` there is the target and
    the vessel radius ``Rm_cm`` there is the emitting wall, indexed exactly as
    the kinetic instruments index them so the two derive the same row. The
    aperture is the feed pipe, ``pipe_id_cm`` wide and ``pipe_length_cm`` long,
    whose ratio is the beaming parameter ``Gamma``.

    Every input is run-constant, so the result is MEMOISED and returned
    read-only; the quadrature behind it costs a fraction of a second and the
    puff row is otherwise rebuilt on every right-hand side.

    Raises ``ValueError`` on a non-positive pipe diameter or length, and
    carries every refusal of :func:`launch_row` and
    :func:`clausing_intensity` -- an aspect ratio below ``4/3``, a port
    outside the grid, and a column radius that is not inside the wall.
    """
    d = float(pipe_id_cm)
    length = float(pipe_length_cm)
    if not np.isfinite(d) or d <= 0.0:
        raise ValueError(
            f"gas_puff_orifice_id_cm must be finite and positive (got "
            f"{pipe_id_cm!r}): it is the inner diameter of the collimating "
            "feed pipe."
        )
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError(
            f"gas_puff_orifice_length_cm must be finite and positive (got "
            f"{pipe_length_cm!r}): it is the length of the collimating feed "
            "pipe, and only its ratio to the bore enters."
        )
    edges = np.ascontiguousarray(geometry.z_edges_cm, dtype=float)
    z0 = float(z_port_cm)
    i_port = int(np.searchsorted(edges, z0) - 1)
    i_port = min(max(i_port, 0), int(np.asarray(geometry.Rm_cm).size) - 1)
    return _launch_row_cached(
        edges.tobytes(),
        d,
        length / d,
        float(np.asarray(geometry.Rm_cm)[i_port]),
        float(np.asarray(geometry.Rp_cm)[i_port]),
        z0,
    )


# ---------------------------------------------------------------- reporting


def mass_span(row, z_edges_cm, quantiles=(0.05, 0.95)):
    """Inter-quantile axial span of a per-cell row [cm], with its two edges.

    The span of record for a puff row: the 5%-95% interval, read off the
    cumulative mass with linear interpolation WITHIN the straddling cell so
    the number does not jump with the mesh.  Returns ``(span, lo, hi)``.
    """
    w = np.asarray(row, dtype=float)
    edges = np.asarray(z_edges_cm, dtype=float)
    if w.size != edges.size - 1:
        raise ValueError(
            f"row has {w.size} entries for {edges.size - 1} cells: the span is "
            "read on the row's own grid."
        )
    tot = w.sum()
    if tot <= 0.0:
        raise ValueError("cannot take the mass span of an all-zero row.")
    cum = np.concatenate(([0.0], np.cumsum(w) / tot))
    lo, hi = (float(np.interp(q, cum, edges)) for q in quantiles)
    return hi - lo, lo, hi


def describe(rows, z_edges_cm, cosine_pipe_row=None):
    """Announce-itself lines: what was derived, and how it compares.

    ``rows`` is the mapping returned by :func:`launch_row_bracket`.  When
    ``cosine_pipe_row`` is given (the fluid stance's own per-cell row on the
    same grid) the fluid deposition envelope is reported beside the kinetic
    bracket, so the registered kinetic-vs-fluid difference is visible on every
    run that uses this row rather than inferred later.
    """
    edges = np.asarray(z_edges_cm, dtype=float)
    lines = [
        "[puff-orifice] kinetic launch row DERIVED from the CAD port geometry",
        f"  station           z = {PORT_CENTER_Z_CM:.4g} cm "
        f"(CAD collar span {PORT_SPAN_Z_CM[0]:.4g}-{PORT_SPAN_Z_CM[1]:.4g} cm, "
        "two opposed mid-plane ports at one station)",
        f"  flight            wall r = {WALL_RADIUS_CM:.5g} cm -> column "
        f"r = {rows['wide'][1]['r_edge_cm']:.5g} cm",
        f"  port bore         {PORT_BORE_DIAMETER_CM:.5g} cm, non-limiting "
        "(through-wall length = 0.952 cm shell)",
        "  angular law       Clausing transparent-regime long-tube "
        "(Ashkarin et al. arXiv:2605.12212 Eqs. 17, 22-25)",
    ]
    for key, label in (("wide", "WIDE  "), ("narrow", "NARROW")):
        row, meta = rows[key]
        span, lo, hi = mass_span(row, edges)
        lines.append(
            f"  {label} d = {meta['pipe_id_cm']:.4g} cm, Gamma = "
            f"{meta['aspect_ratio']:.4g} (theta_o = {meta['theta_o_deg']:.3g} deg): "
            f"5-95% span {span:.4g} cm [{lo:.4g}, {hi:.4g}], "
            f"perigee-placed {meta['missed_fraction']:.3g}, "
            f"off-grid {meta['clipped_fraction']:.3g}"
        )
    if cosine_pipe_row is not None:
        span, lo, hi = mass_span(cosine_pipe_row, edges)
        lines.append(
            f"  FLUID  cosine_pipe deposition envelope: 5-95% span {span:.4g} cm "
            f"[{lo:.4g}, {hi:.4g}] -- a CLOSURE, not this geometry; the "
            "difference is the registered finding, not an error"
        )
    return lines
