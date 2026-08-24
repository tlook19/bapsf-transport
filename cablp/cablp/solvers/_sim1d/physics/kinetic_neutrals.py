"""Kinetic two-zone neutral engine (K1/K1b/K4).

The deterministic velocity-resolved column/annulus neutral model: shared
sinh-stretched (v_z, v_perp) grid, moment-exact shifted Maxwellians,
KN1D-class generation iteration, and the bounded-chord JUMP kernel for the
annulus (K1b -- gated against the TPMC to within MC statistics, zero free
parameters). Lives in the package so the offline CLI (`scripts/kn2zone.py`)
and the in-solver K4a quasi-static mode share ONE implementation.

The engine consumes a plain background dict (the offline wrapper builds it
from a saved run via `mc_neutrals.load_background`; the solver builds it
from its live fields): z_edges, Rp, Rm [cm], nu_ion, nu_cx [1/s], Ti [eV],
u [cm/s], T_s [K], S_pump_L/R [L/s], eta, mesh_edge, sources (the ledger
menu), optional rec_cell.

Constants match `scripts/mc_neutrals.py` EXACTLY (the instrument suite
must agree to the bit on inputs; disagreement between instruments is
method error, never input error).
"""

import numpy as np

from cablp.vars._cons import m_He_cgs

EV = 1.602176634e-12
KB = 1.380649e-16
# Helium mass: imported, never re-derived. cablp.vars._cons is THE
# definition point (Ar(4He)*u, CODATA 2022); the hand-made
# 4.002602 * 1.66053907e-24 product this replaced was 0.31 ppm low.
M_HE = m_He_cgs
T_WALL_K = 300.0


# ------------------------------------------------------------ g_eff thermal floor

def ion_thermal_g_eff_floor_cm2_s2(Ti_eV, ion_mass_g=M_HE, ev_to_erg=EV):
    """Return the ion thermal floor of ``g_eff^2`` [cm^2/s^2].

        floor = 8 k Ti / (pi m)

    ``Ti_eV`` is the ion temperature [eV] and may be any shape numpy
    broadcasts; ``ion_mass_g`` [g] and ``ev_to_erg`` [erg/eV] default to this
    module's helium constants. The result is non-negative for ``Ti_eV >= 0``
    and nothing is raised: the temperature clamp belongs to the caller, and
    every caller applies one before this call.

    THE ONE DEFINITION of the floor, so that a transcription of it cannot
    drift. Callers form the mean relative speed of a projectile against a
    drifting ion Maxwellian as

        g_eff^2 = |v - u_i|^2 + floor(Ti)

    -- the standard interpolation between the drift-dominated and the
    thermal-dominated limits -- and consume ``g_eff`` twice: as the rate's
    relative speed in ``n_i sigma(E_rel) g_eff``, and through
    ``E_rel = (1/2) mu g_eff^2`` as the Phelps cross-section argument.

    The floor carries the mass of whichever collider is MAXWELLIAN, and in
    every caller of this helper that is the IONS ALONE: the projectile's
    velocity is resolved exactly -- a velocity-grid bin, a monoenergetic beam
    atom, or a tracked MC particle -- and is already inside ``|v - u_i|^2``,
    so the drift-free limit of ``<g>`` is the mean speed of the ion Maxwellian
    by itself, ``sqrt(8 k Ti / (pi m))``. The reduced-mass form
    ``8 k Ti / (pi mu) = 16 k Ti / (pi m)`` is the TWO-Maxwellian expression
    -- ``mu`` is what folds two INDEPENDENT thermal spreads into one -- and
    used against a resolved projectile it counts a thermal spread that does
    not exist, inflating ``g_eff`` by up to ``sqrt(2)``. The reduced mass DOES
    belong in ``E_rel``: that is two-body kinematics, unaffected by which
    species is Maxwellian.

    ``sqrt(floor)`` at ``Ti_eV = 1`` with this module's ``M_HE`` is
    783482.7390046517 cm/s, and 1108011.9... cm/s under the reduced-mass
    form; the value is pinned by the ``kinetic-geff-thermal-floor`` case of
    ``scripts/smoke_sim1d.py``.
    """
    return 8.0 * Ti_eV * ev_to_erg / (np.pi * ion_mass_g)


# ---------------------------------------------------------------- velocity grid

def stretched_axis(vmax, n, v_fine):
    """Return a symmetric sinh-stretched axis of n bin CENTERS in (-vmax, vmax).

    Resolution ~v_fine near zero (the 300 K wall gas -- the duct-tail
    population) coarsening toward vmax (the CX tail). No bin sits at
    exactly zero: centers are at half-offsets, so the upwind march never
    divides by zero and every bin transports.
    """
    a = np.arcsinh(vmax / v_fine)
    u = (np.arange(n) + 0.5) / n * 2.0 - 1.0  # (-1, 1), half-offset
    return v_fine * np.sinh(a * np.abs(u)) * np.sign(u)


def stretched_positive_axis(vmax, n, v_fine):
    """Return n positive bin centers in (0, vmax), fine near zero."""
    a = np.arcsinh(vmax / v_fine)
    u = (np.arange(n) + 0.5) / n
    return v_fine * np.sinh(a * u)


def bin_edges(centers, lo=None, hi=None):
    e = np.empty(centers.size + 1)
    e[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    e[0] = lo if lo is not None else centers[0] - (e[1] - centers[0])
    e[-1] = hi if hi is not None else centers[-1] + (centers[-1] - e[-2])
    return e


class VGrid:
    """Shared (v_z, v_perp) grid with analytic-bin-mass Maxwellian projection."""

    def __init__(self, vmax_z, vmax_p, nvz, nvp, v_fine):
        self.vz = stretched_axis(vmax_z, nvz, v_fine)
        self.vp = stretched_positive_axis(vmax_p, nvp, v_fine)
        self.vz_edges = bin_edges(self.vz)
        self.vp_edges = bin_edges(self.vp, lo=0.0)
        self.nvz = nvz
        self.nvp = nvp
        # broadcastable views
        self.VZ = self.vz[:, None]      # (nvz, 1)
        self.VP = self.vp[None, :]      # (1, nvp)
        self.V2 = self.VZ**2 + self.VP**2

    def maxwellian(self, T_eV, u_drift, exact_moments=True):
        """Return bin masses of a drifting Maxwellian, summing to 1.

        Analytic per-bin masses: erf differences along v_z, Rayleigh CDF
        differences along v_perp (the 2D perpendicular speed measure), so
        the density is exact by construction. ``exact_moments`` applies the
        two-basis compensation (KN1D's scheme, reimplemented) so the
        DISCRETE drift and energy moments -- evaluated with bin centers,
        the way the transport uses them -- also hit their targets.
        """
        s = np.sqrt(max(T_eV, 1e-6) * EV / M_HE)  # 1D thermal spread
        from math import erf, sqrt

        ez = (self.vz_edges - u_drift) / (s * np.sqrt(2.0))
        wz = 0.5 * np.diff(np.array([erf(x) for x in ez]))
        ep = self.vp_edges / s
        wp = np.diff(-np.exp(-0.5 * ep**2))
        f = wz[:, None] * wp[None, :]
        total = f.sum()
        if total <= 0:
            # drift beyond the grid edge: clamp to the boundary half-space
            # rather than dying mid-run (the grid should be sized to make
            # this unreachable; the clamp keeps a pathological transient
            # cell from killing a discharge)
            edge = self.vz_edges[-2] if u_drift > 0 else self.vz_edges[1]
            ez = (self.vz_edges - edge) / (s * np.sqrt(2.0))
            wz = 0.5 * np.diff(np.array([erf(x) for x in ez]))
            f = wz[:, None] * wp[None, :]
            total = f.sum()
        f /= total
        if not exact_moments:
            return f
        # Two-basis compensation: f' = f + a*(vz - u)*f + b*(V2 - <V2>)*f
        # solved so the discrete first (vz) and second (energy) moments hit
        # the analytic targets exactly; density is preserved because both
        # basis functions are built moment-free about the current state.
        target_u = u_drift
        target_e = u_drift**2 + 3.0 * s**2
        for _ in range(4):
            m1 = float((f * self.VZ).sum())
            m2 = float((f * self.V2).sum())
            b1 = (self.VZ - m1) * f
            b2 = (self.V2 - m2) * f
            A = np.array(
                [
                    [float((b1 * self.VZ).sum()), float((b2 * self.VZ).sum())],
                    [float((b1 * self.V2).sum()), float((b2 * self.V2).sum())],
                ]
            )
            rhs = np.array([target_u - m1, target_e - m2])
            try:
                ab = np.linalg.solve(A, rhs)
            except np.linalg.LinAlgError:
                break
            f_new = f + ab[0] * b1 + ab[1] * b2
            if f_new.min() < -1e-12 * f.max():
                # clip-and-retry keeps positivity; one pass is enough in
                # practice on adequately resolved grids
                f_new = np.maximum(f_new, 0.0)
                f_new /= f_new.sum()
            f = f_new
            if (
                abs(float((f * self.VZ).sum()) - target_u)
                <= 1e-10 * max(abs(target_u), s)
                and abs(float((f * self.V2).sum()) - target_e) <= 1e-10 * target_e
            ):
                break
        return f

    def wall_emission_spectrum(self, T_K):
        """Diffuse (cosine) re-emission spectrum off the CYLINDRICAL wall.

        The flux weighting acts along the radial surface normal, so the
        re-emitted perpendicular-speed marginal is ~ vp^2 exp(-vp^2/2s^2)
        (one power of vp from the 2D measure, one from the cosine law) --
        NOT the volume Maxwellian's Rayleigh. Detailed balance demands it:
        the wall absorbs at nu_w ~ vp, so the equilibrium (volume
        Maxwellian) is stationary only if re-emission is the vp-flux-
        weighted spectrum. An isotropic re-emission would over-populate
        grazing atoms every bounce and over-carry the duct tail.
        v_z stays a plain Gaussian (tangential to the wall).
        """
        s = np.sqrt(KB * T_K / M_HE)
        from math import erf

        ez = self.vz_edges / (s * np.sqrt(2.0))
        wz = 0.5 * np.diff(np.array([erf(x) for x in ez]))
        # per-bin mass of vp^2 exp(-vp^2 / 2 s^2): fine subsampled quadrature
        wp = np.empty(self.nvp)
        for k in range(self.nvp):
            x = np.linspace(self.vp_edges[k], self.vp_edges[k + 1], 64)
            y = x**2 * np.exp(-0.5 * (x / s) ** 2)
            wp[k] = np.trapezoid(y, x)
        f = wz[:, None] * wp[None, :]
        return f / f.sum()

    def half_flux_spectrum(self, T_K, sign):
        """Cosine-flux (surface-emission) bin masses on the half-space.

        Flux-weighted Maxwellian at wall temperature: mass per bin
        proportional to |v_z| * M restricted to sign(v_z) = sign; sums to 1.
        """
        T_eV = KB * T_K / EV
        m = self.maxwellian(T_eV, 0.0, exact_moments=False)
        w = np.where(np.sign(self.VZ) == sign, np.abs(self.VZ) * m, 0.0)
        return w / w.sum()


# ------------------------------------------------------------ puff placement


def puff_launch_bins(sources, z_edges, nz):
    """Return the gas puff as ``[(z-bin index, rate [atoms/s]), ...]``.

    The puff is a DISTRIBUTION over z, not a point. ``sources["puff_cells"]``
    is the per-cell puff row [s^-1] on this engine's own cells, assembled by
    the loader from the run's own neutral ledger (or, on an artifact whose
    ledger cannot carry it, derived from the resolved config through the
    solver's own ``gas_puff_rate_profile``). Honouring it seeds the fuel
    exactly where the solver seeded it: under the config of record the
    ``"cosine_pipe"`` profile centred on ``gas_puff_z_cm``, whose 86.3 cm is
    the CAD-measured mid-plane injection station -- the two CF6000-class
    ports on the anode stack, z_model 0.812-0.914 m, centre 0.863 m. The
    configured profile, not the bare CAD span, is what is placed: it is the
    shape the solver applied, so the kinetic instrument and the fluid model
    fuel the same cells and a disagreement between them stays method error
    rather than input error.

    HISTORICAL, superseded 2026-08-24: this engine seeded the WHOLE puff into
    the single bin containing ``sources["puff_z"]``. That convention dates
    from when ``"cell"`` was the only shipped ``gas_puff_profile``, where the
    single bin and the per-cell row are the same bin at the same rate and
    nothing moves; on ``"gaussian"`` or ``"cosine_pipe"`` it collapsed a
    distributed source to a point. ``puff_z`` remains the fallback for a
    background carrying no per-cell row -- the synthetic slab fixtures, and
    any caller assembling ``sources`` by hand.

    Raises ``ValueError`` when the per-cell row is not one entry per cell:
    a length mismatch would silently place fuel in the wrong cells.
    """
    total = float(sources.get("puff", 0.0))
    if total <= 0.0:
        return []
    cells = sources.get("puff_cells")
    if cells is None:
        iz = int(np.searchsorted(z_edges, sources["puff_z"]) - 1)
        return [(min(max(iz, 0), nz - 1), total)]
    row = np.asarray(cells, dtype=float)
    if row.size != nz:
        raise ValueError(
            f"sources['puff_cells'] must have one entry per engine cell "
            f"(nz={nz}); got {row.size}. It is a per-cell puff row on this "
            "engine's own grid, not a shape to be resampled."
        )
    return [(int(i), float(row[i])) for i in np.flatnonzero(row > 0.0)]


# ---------------------------------------------------------------- the solver


class KN2Zone:
    def __init__(self, bg, nvz=80, nvp=24, truncate=1e-3, max_gen=400,
                 verbose=True, grid=None):
        self.bg = bg
        self.truncate = truncate
        self.max_gen = max_gen
        self.verbose = verbose
        ze = bg["z_edges"]
        self.dz = np.diff(ze)
        self.nz = self.dz.size
        self.Rp = bg["Rp"]
        self.Rm = bg["Rm"]
        self.A_col = np.pi * self.Rp**2
        self.A_ann = np.pi * (self.Rm**2 - self.Rp**2)
        self.V_col = self.A_col * self.dz
        self.V_ann = self.A_ann * self.dz
        self.nu_ion = bg["nu_ion"]
        self.nu_cx = bg["nu_cx"]
        if grid is not None:
            # frozen shared grid (K4a-t: the compiled flight kernels bind
            # to the grid, so refreshes must reuse it)
            self.g = grid
        else:
            # size from BOTH the thermal spread and the drift: a sonic
            # CX-source cell must project inside the grid
            Ti_max = float(np.max(bg["Ti"]))
            u_max = float(np.max(np.abs(bg["u"]))) if "u" in bg else 0.0
            vmax = 4.0 * np.sqrt(max(Ti_max, 0.5) * EV / M_HE) + 1.5 * u_max
            v_fine = 0.25 * np.sqrt(KB * T_WALL_K / M_HE)
            self.g = VGrid(vmax, vmax, nvz, nvp, v_fine)
        g = self.g
        # Velocity-dependent zone rates, (nz, nvp). Two closed geometric
        # operator sets, no free parameters in either:
        #
        # "flux": rates that reproduce the volume-mixed one-way fluxes
        # exactly at the Maxwellian zeroth moment (the moment model's K_r).
        # Correct at equilibrium, but the TRANSPORT population is
        # wall-emission dominated, and per surface flight the branching
        # into the column is the view factor Rp/Rm, not the volume-mixed
        # Rp/(Rp+Rm) -- 30% low per bounce, compounding down the duct.
        #
        # "cauchy" (default): mean surface-to-surface flight time from the
        # cavity Cauchy chord <l> = 4V/S = 2(Rm-Rp), branched by the
        # coaxial-cylinder view factor F = Rp/Rm; the column return rate
        # follows from detailed balance (V_col nu_x = V_ann nu_x'). Its
        # Maxwellian-averaged column-entry flux lands within ~2% of the
        # exact equilibrium value -- both operator sets are honest, they
        # privilege different measures; the TPMC adjudicates.
        vp = g.vp[None, :]
        Rp2 = self.Rp[:, None]
        Rm2 = self.Rm[:, None]
        ann_area = np.maximum(self.Rm**2 - self.Rp**2, 1e-12)[:, None]
        if bg.get("zone_rates", "cauchy") == "flux":
            self.nux = 2.0 * vp / (np.pi * Rp2)
            self.nuxp = 2.0 * vp * Rp2 / (np.pi * ann_area)
            self.nuw = 2.0 * vp * Rm2 / (np.pi * ann_area)
        else:
            nu_total = vp / (2.0 * np.maximum(Rm2 - Rp2, 1e-9))
            self.nuxp = (Rp2 / Rm2) * nu_total
            self.nuw = (1.0 - Rp2 / Rm2) * nu_total
            self.nux = self.nuxp * ann_area / np.maximum(Rp2**2, 1e-12)
        # fixed re-emission spectra: the cylindrical wall re-emits
        # flux-weighted (cosine about the radial normal); volume processes
        # (CX) stay isotropic Maxwellians.
        self.M_wall = g.wall_emission_spectrum(T_WALL_K)
        self.M_cx = np.empty((self.nz, g.nvz, g.nvp))
        for i in range(self.nz):
            self.M_cx[i] = g.maxwellian(
                max(bg["Ti"][i], 0.02), bg["u"][i]
            )
        # end pump sticking (TPMC convention: S_pump over the end-plane
        # one-way thermal flux)
        vbar = np.sqrt(8.0 * KB * T_WALL_K / (np.pi * M_HE))
        A_end = np.pi * self.Rm[-1] ** 2
        self.s_R = bg["S_pump_R"] * 1e3 / (A_end * vbar / 4.0)
        self.s_L = bg["S_pump_L"] * 1e3 / (A_end * vbar / 4.0)
        self.mesh_face = bg["mesh_edge"]  # z-edge index
        self.transparency = 1.0 - bg["eta"]

    # -- one generation: implicit-upwind march with 2x2 zone coupling
    def sweep(self, Sc, Sa, Fc_in_L, Fa_in_L, Fc_in_R, Fa_in_R):
        """March one generation. S* are volume sources [cm^-3 s^-1 / bin];
        F*_in are inflow densities [cm^-3 / bin] at the domain ends
        (cathode side L, collector side R; only the inward-moving half of
        each is used). Returns (Fc, Fa) [cm^-3 / bin] and the intercepted
        collided-flux tallies for the next generation."""
        g = self.g
        nz, nvz, nvp = self.nz, g.nvz, g.nvp
        Fc = np.zeros((nz, nvz, nvp))
        Fa = np.zeros((nz, nvz, nvp))
        mesh_cx_c = np.zeros(nz)  # intercepted mesh flux, re-emitted later
        mesh_cx_a = np.zeros(nz)
        for direction in (+1, -1):
            if direction > 0:
                order = range(nz)
                sel = g.vz > 0
                F_c_prev = Fc_in_L[sel]
                F_a_prev = Fa_in_L[sel]
            else:
                order = range(nz - 1, -1, -1)
                sel = g.vz < 0
                F_c_prev = Fc_in_R[sel]
                F_a_prev = Fa_in_R[sel]
            vz = np.abs(g.vz[sel])[:, None]  # (nsel, 1)
            for i in order:
                # anode-mesh interception on the upstream face of this cell
                face = i if direction > 0 else i + 1
                if face == self.mesh_face:
                    blocked_c = (1.0 - self.transparency) * F_c_prev
                    blocked_a = (1.0 - self.transparency) * F_a_prev
                    # intercepted atoms re-emit on the incident side, next
                    # generation; tally their one-way flux [atoms/s /
                    # (zone volume)] into the emitting cell
                    j = i - direction if direction > 0 else i + 1
                    j = min(max(j, 0), nz - 1)
                    mesh_cx_c[j] += float(
                        (blocked_c * vz).sum() * self.A_col[j] / self.V_col[j]
                    )
                    mesh_cx_a[j] += float(
                        (blocked_a * vz).sum() * self.A_ann[j] / self.V_ann[j]
                    ) if self.V_ann[j] > 0 else 0.0
                    F_c_prev = self.transparency * F_c_prev
                    F_a_prev = self.transparency * F_a_prev
                # Zone coupling with the volume conversion folded in: the
                # column gains nu_x * Fa (annulus outflow nu_x' * Fa * V_ann
                # converted per column volume equals nu_x * Fa exactly), the
                # annulus gains nu_x' * Fc -- symmetric in density space,
                # equilibrium at EQUAL densities, and the per-bin conductance
                # V_col * nu_x = 2 v_perp Rp dz Maxwellian-averages to the
                # moment model's K_r identically.
                lam = vz / self.dz[i]
                a11 = lam + (
                    self.nu_ion[i] + self.nu_cx[i] + self.nux[i][None, :]
                )
                a12 = -self.nux[i][None, :] * np.ones_like(vz)
                a21 = -self.nuxp[i][None, :] * np.ones_like(vz)
                a22 = lam + (self.nuxp[i] + self.nuw[i])[None, :]
                r1 = Sc[i][sel] + lam * F_c_prev
                r2 = Sa[i][sel] + lam * F_a_prev
                det = a11 * a22 - a12 * a21
                fc = (r1 * a22 - a12 * r2) / det
                fa = (a11 * r2 - a21 * r1) / det
                Fc[i][sel] = fc
                Fa[i][sel] = fa
                F_c_prev, F_a_prev = fc, fa
        return Fc, Fa, mesh_cx_c, mesh_cx_a

    def outgoing_flux(self, F, zone_area, end):
        """One-way outgoing flux [atoms/s] of F at a domain end."""
        g = self.g
        i = -1 if end > 0 else 0
        sel = g.vz > 0 if end > 0 else g.vz < 0
        vz = np.abs(g.vz[sel])[:, None]
        return float((F[i][sel] * vz).sum() * zone_area[i])

    def solve(self):
        """Generation-iterate to the steady state.

        Primary sources come from ``self.bg["sources"]`` (the run's own
        ledger, exactly the TPMC's menu): cathode/collector faces as
        boundary inflows (cosine at T_s / 300 K), the puff as an annulus
        volume source at 300 K over the configured axial profile
        (:func:`puff_launch_bins`), volume recombination in the column at the
        local ion Maxwellian, and the anode-mesh rebirths as directed
        300 K half-Maxwellians in the flanking cells.
        """
        g = self.g
        nz = self.nz
        bgs = self.bg["sources"]
        T_s = self.bg["T_s"]

        def inflow(rate, spectrum, area):
            # boundary density per bin from a face flux with unit spectrum
            with np.errstate(divide="ignore", invalid="ignore"):
                dens = np.where(
                    np.abs(g.VZ) > 0, spectrum / (np.abs(g.VZ) * area), 0.0
                )
            return rate * dens

        Fc_in_L = inflow(
            bgs.get("cathode_face", 0.0),
            g.half_flux_spectrum(T_s, +1),
            self.A_col[0],
        )
        Fc_in_R = inflow(
            bgs.get("collector_face", 0.0),
            g.half_flux_spectrum(T_WALL_K, -1),
            self.A_col[-1],
        )
        Fa_in_L = np.zeros_like(Fc_in_L)
        Fa_in_R = np.zeros_like(Fc_in_R)

        Sc = np.zeros((nz, g.nvz, g.nvp))
        Sa = np.zeros((nz, g.nvz, g.nvp))
        for iz, rate in puff_launch_bins(bgs, self.bg["z_edges"], nz):
            Sa[iz] += rate / self.V_ann[iz] * self.M_wall
        rec = self.bg.get("rec_cell")
        if rec is not None and rec.sum() > 0:
            scale = bgs.get("vol_rec", rec.sum()) / rec.sum()
            for i in range(nz):
                if rec[i] > 0:
                    Sc[i] += scale * rec[i] / self.V_col[i] * self.M_cx[i]
        for name, sign in (("anode_left", -1), ("anode_right", +1)):
            rate = bgs.get(name, 0.0)
            if rate <= 0:
                continue
            j = self.mesh_face - 1 if sign < 0 else self.mesh_face
            j = min(max(j, 0), nz - 1)
            Sc[j] += (
                rate / self.V_col[j] * g.half_flux_spectrum(T_WALL_K, sign)
            )

        F_tot_c = np.zeros((nz, g.nvz, g.nvp))
        F_tot_a = np.zeros((nz, g.nvz, g.nvp))
        inv_tot = 0.0
        gen = 0
        while gen < self.max_gen:
            Fc, Fa, mesh_c, mesh_a = self.sweep(
                Sc, Sa, Fc_in_L, Fa_in_L, Fc_in_R, Fa_in_R
            )
            F_tot_c += Fc
            F_tot_a += Fa
            inv = float(
                (Fc.sum(axis=(1, 2)) * self.V_col).sum()
                + (Fa.sum(axis=(1, 2)) * self.V_ann).sum()
            )
            inv_tot += inv
            gen += 1
            if self.verbose and (gen <= 3 or gen % 25 == 0):
                print(f"  gen {gen:3d}: inventory {inv:.3e} "
                      f"(total {inv_tot:.3e})")
            if inv <= self.truncate * inv_tot:
                break
            # ---- collided fluxes -> next generation sources
            nn_c = Fc.sum(axis=(1, 2))
            R_cx = self.nu_cx * nn_c                          # (nz,)
            R_w = (Fa * self.nuw[:, None, :]).sum(axis=(1, 2))  # (nz,)
            Sc = R_cx[:, None, None] * self.M_cx
            Sa = R_w[:, None, None] * self.M_wall[None, :, :]
            # mesh interception re-emits at 300 K on the incident side:
            Sc += mesh_c[:, None, None] * self.M_wall[None, :, :]
            Sa += mesh_a[:, None, None] * self.M_wall[None, :, :]
            # ---- end walls: re-emit (1 - sticking), thermal, same zone
            Fc_in_L = np.zeros_like(Fc_in_L)
            Fc_in_R = np.zeros_like(Fc_in_R)
            Fa_in_L = np.zeros_like(Fa_in_L)
            Fa_in_R = np.zeros_like(Fa_in_R)
            for F, area, zone_in_L, zone_in_R, T_L in (
                (Fc, self.A_col, "cL", "cR", T_s),
                (Fa, self.A_ann, "aL", "aR", T_s),
            ):
                out_L = self.outgoing_flux(F, area, -1)
                out_R = self.outgoing_flux(F, area, +1)
                back_L = (1.0 - self.s_L) * out_L
                back_R = (1.0 - self.s_R) * out_R
                spec_L = g.half_flux_spectrum(T_L, +1)
                spec_R = g.half_flux_spectrum(T_WALL_K, -1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    dL = np.where(
                        np.abs(g.VZ) > 0,
                        spec_L / (np.abs(g.VZ) * area[0]),
                        0.0,
                    )
                    dR = np.where(
                        np.abs(g.VZ) > 0,
                        spec_R / (np.abs(g.VZ) * area[-1]),
                        0.0,
                    )
                if zone_in_L == "cL":
                    Fc_in_L = back_L * dL
                    Fc_in_R = back_R * dR
                else:
                    Fa_in_L = back_L * dL
                    Fa_in_R = back_R * dR
        return {
            "Fc": F_tot_c,
            "Fa": F_tot_a,
            "generations": gen,
            "nn_col": F_tot_c.sum(axis=(1, 2)),
            "nn_ann": F_tot_a.sum(axis=(1, 2)),
            "un_col": _drift(F_tot_c, self.g),
            "un_ann": _drift(F_tot_a, self.g),
        }


def _drift(F, g):
    n = F.sum(axis=(1, 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(
            n > 0, (F * g.VZ[None, :, :]).sum(axis=(1, 2)) / np.maximum(n, 1e-300), 0.0
        )


class KineticEngineFast:
    """Compiled jump-kernel engine for in-solver use (K4a).

    Physics identical to ``KN2ZoneJump`` up to two documented
    aggregations; the flight geometry is generation-independent, so it is
    COMPILED once at construction:

    - wall launches are rank-1 in velocity (rate x cosine-wall spectrum),
      so the wall->wall and wall->inner classes reduce to
      spectrum-aggregated kernels: landing matrices, residence matrices
      [atom-seconds per unit launch], end-crossing masses. EXACT for
      wall->wall (landing re-emission resamples the spectrum anyway);
      for wall->inner the entrants are assigned the wall spectrum at the
      landing cell (the flights are short, ~0.75 Rm, so the
      position-velocity correlation lost is minor -- validated against
      KN2ZoneJump).
    - column escapes (inner->outer) stay bin-resolved: precompiled
      landing cells + flight times; residence split half source / half
      landing cell (short flights; minor inventory share).

    Each generation is then a few (nz x nz) matvecs plus one
    gather/scatter, so the refresh-cadence architecture fits the runtime
    budget at the full grid.
    """

    def __init__(self, jump):
        self.j = jump
        nz = jump.nz
        g = jump.g
        wall_spec = jump.M_wall
        self.P_ww = np.zeros((nz, nz))
        self.P_wi = np.zeros((nz, nz))
        self.R_ww = np.zeros((nz, nz))
        self.R_wi = np.zeros((nz, nz))
        self.E_ww = np.zeros((nz, 2))
        self.E_wi = np.zeros((nz, 2))
        for i in range(nz):
            for P, R, E, chord in (
                (self.P_ww, self.R_ww, self.E_ww, jump.c_ww),
                (self.P_wi, self.R_wi, self.E_wi, jump.c_wi),
            ):
                launch = np.zeros((nz, g.nvz, g.nvp))
                launch[i] = wall_spec
                tal = np.zeros(nz)
                talv = np.zeros(nz)
                land, eL, eR = jump._fly(launch, chord, tal, talv)
                P[i] = land.sum(axis=(1, 2))
                R[i] = tal * np.maximum(jump.V_ann, 0.0)
                E[i, 0] = float(eL.sum())
                E[i, 1] = float(eR.sum())
        ze = jump.z_edges
        zc = jump.zc
        with np.errstate(divide="ignore", invalid="ignore"):
            tof = jump.c_io[:, None, None] / np.maximum(
                g.VP[None, :, :], 1e-30
            )
        dz = g.VZ[None, :, :] * tof
        z_land = zc[:, None, None] + dz
        self.io_cross_L = z_land < ze[0]
        self.io_cross_R = z_land > ze[-1]
        self.io_j = np.clip(
            np.searchsorted(ze, np.clip(z_land, ze[0], ze[-1])) - 1,
            0,
            nz - 1,
        )
        self.io_t = np.where(
            self.io_cross_L | self.io_cross_R, 0.0, tof
        )  # clipped flights' residence folded into end handling (small)
        self.F = jump.F_inner
        self.wall_spec = wall_spec
        # END-emission kernels: atoms re-emitted from the z-normal end
        # walls carry the HALF-FLUX spectrum (plain-Maxwellian v_perp
        # marginal -- grazing-rich, the NBL's duct feed), NOT the
        # cylindrical wall's vp^2-weighted spectrum. Compiled per end
        # through the same two chord classes.
        self.end_kernels = {}
        for endcell, sign, T in ((0, +1, jump.bg["T_s"]), (nz - 1, -1, T_WALL_K)):
            spec = g.half_flux_spectrum(T, sign)
            entry = {}
            for key, chord, frac_arr in (
                ("ww", jump.c_ww, 1.0 - jump.F_inner),
                ("wi", jump.c_wi, jump.F_inner),
            ):
                launch = np.zeros((nz, g.nvz, g.nvp))
                launch[endcell] = frac_arr[endcell] * spec
                tal = np.zeros(nz)
                talv = np.zeros(nz)
                land, eL, eR = jump._fly(launch, chord, tal, talv)
                entry[key] = {
                    "P": land.sum(axis=(1, 2)),
                    "R": tal * np.maximum(jump.V_ann, 0.0),
                    "EL": float(eL.sum()),
                    "ER": float(eR.sum()),
                }
            self.end_kernels[endcell] = entry

    def solve(self, Sc, Fc_in_L, Fc_in_R, wall_rate0, truncate=1e-3,
              max_gen=600):
        """Generation-iterate with compiled flight classes.

        ``Sc``: column volume source per bin [cm^-3 s^-1]; ``Fc_in_*``:
        boundary inflow densities per bin; ``wall_rate0``: primary annulus
        wall-launch rates per cell [atoms/s]. Returns the moments dict.
        """
        j = self.j
        g = j.g
        nz = j.nz
        tal_t = np.zeros(nz)
        arr = np.zeros(nz)  # per-cell annulus launch turnover (tau source)
        F_tot_c = np.zeros((nz, g.nvz, g.nvp))
        wall_rate = wall_rate0.copy()
        inner = np.zeros((nz, g.nvz, g.nvp))
        inv_tot = 0.0
        launch_peak = 1e-300
        gen = 0
        spec_L = g.half_flux_spectrum(j.bg["T_s"], +1)
        spec_R = g.half_flux_spectrum(T_WALL_K, -1)
        while gen < max_gen:
            Fc = np.zeros((nz, g.nvz, g.nvp))
            for direction in (+1, -1):
                order = range(nz) if direction > 0 else range(nz - 1, -1, -1)
                sel = g.vz > 0 if direction > 0 else g.vz < 0
                F_prev = (Fc_in_L if direction > 0 else Fc_in_R)[sel]
                vz = np.abs(g.vz[sel])[:, None]
                for i in order:
                    face = i if direction > 0 else i + 1
                    if face == j.mesh_face:
                        F_prev = j.transparency * F_prev
                    lam = vz / j.dz[i]
                    denom = lam + (
                        j.nu_ion[i] + j.nu_cx[i] + j.nux[i][None, :]
                    )
                    fc = (Sc[i][sel] + lam * F_prev) / denom
                    Fc[i][sel] = fc
                    F_prev = fc
            F_tot_c += Fc
            inv = float((Fc.sum(axis=(1, 2)) * j.V_col).sum())
            inner += Fc * j.nux[:, None, :] * j.V_col[:, None, None]
            R_cx = j.nu_cx * Fc.sum(axis=(1, 2))
            Sc = R_cx[:, None, None] * j.M_cx
            out_L = j.outgoing_flux(Fc, j.A_col, -1)
            out_R = j.outgoing_flux(Fc, j.A_col, +1)
            Fc_in_L = _inflow(
                (1.0 - j.s_L) * out_L, spec_L, j.A_col[0], g
            )
            Fc_in_R = _inflow(
                (1.0 - j.s_R) * out_R, spec_R, j.A_col[-1], g
            )
            w = wall_rate
            arr += w
            wall_rate = np.zeros(nz)
            il = inner
            inner = np.zeros_like(il)
            launches = float(w.sum() + il.sum())
            w_ww = w * (1.0 - self.F)
            w_wi = w * self.F
            tal_t += (w_ww @ self.R_ww + w_wi @ self.R_wi) / np.maximum(
                j.V_ann, 1e-30
            )
            wall_rate += w_ww @ self.P_ww
            Sc = Sc + (
                (w_wi @ self.P_wi) / j.V_col
            )[:, None, None] * self.wall_spec
            end_L = float(w_ww @ self.E_ww[:, 0] + w_wi @ self.E_wi[:, 0])
            end_R = float(w_ww @ self.E_ww[:, 1] + w_wi @ self.E_wi[:, 1])
            if np.any(il):
                interior = ~(self.io_cross_L | self.io_cross_R)
                res = il * self.io_t
                tal_src = 0.5 * res.sum(axis=(1, 2))
                tal_land = np.zeros(nz)
                np.add.at(
                    tal_land, self.io_j[interior], 0.5 * res[interior]
                )
                tal_t += (tal_src + tal_land) / np.maximum(j.V_ann, 1e-30)
                landed = np.zeros(nz)
                np.add.at(landed, self.io_j[interior], il[interior])
                wall_rate += landed
                end_L += float(il[self.io_cross_L].sum())
                end_R += float(il[self.io_cross_R].sum())
            # end re-emission through the compiled END kernels (z-normal
            # half-flux spectra), not the cylindrical wall class
            for endcell, back in ((0, (1.0 - j.s_L) * end_L),
                                  (nz - 1, (1.0 - j.s_R) * end_R)):
                if back <= 0:
                    continue
                ek = self.end_kernels[endcell]
                tal_t += back * (ek["ww"]["R"] + ek["wi"]["R"]) / np.maximum(
                    j.V_ann, 1e-30
                )
                wall_rate += back * ek["ww"]["P"]
                Sc = Sc + (
                    back * ek["wi"]["P"] / j.V_col
                )[:, None, None] * self.wall_spec
                # secondary end crossings recirculate locally (small): fold
                # the surviving fraction back onto the same end next pass
                wall_rate[0] += back * (1.0 - j.s_L) * (
                    ek["ww"]["EL"] + ek["wi"]["EL"]
                )
                wall_rate[-1] += back * (1.0 - j.s_R) * (
                    ek["ww"]["ER"] + ek["wi"]["ER"]
                )
            inv_tot += max(inv, 0.0)
            gen += 1
            next_launch = float(wall_rate.sum() + inner.sum())
            launch_peak = max(launch_peak, launches, next_launch)
            if (
                gen > 5
                and inv <= truncate * max(inv_tot, 1e-300)
                and next_launch <= truncate * launch_peak
            ):
                break
        return {
            "generations": gen,
            "nn_col": F_tot_c.sum(axis=(1, 2)),
            "nn_ann": tal_t,
            "un_col": _drift_engine(F_tot_c, g),
            # cumulative wall-launch turnover per cell [atoms/s summed over
            # generations] -- at steady state the per-cell throughput, so
            # tau_ann = inventory / arrival is the K0-honest buildup time
            "ann_arrival": arr,
        }


def _inflow(rate, spectrum, area, g):
    with np.errstate(divide="ignore", invalid="ignore"):
        dens = np.where(
            np.abs(g.VZ) > 0, spectrum / (np.abs(g.VZ) * area), 0.0
        )
    return rate * dens


def _drift_engine(F, g):
    n = F.sum(axis=(1, 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(
            n > 0,
            (F * g.VZ[None, :, :]).sum(axis=(1, 2))
            / np.maximum(n, 1e-300),
            0.0,
        )


CHORD_CLASS_SAMPLES = 20001


def annulus_chord_classes(Rp_cm, Rm_cm, samples=CHORD_CLASS_SAMPLES):
    """Return the bounded-chord flight classes of a coaxial annulus.

    Three cosine-weighted chord classes, derived numerically from the local
    ``(Rp, Rm)`` alone -- no free parameters. ``s`` is the sine of the
    emission angle to the surface normal, uniformly sampled because the
    cosine law makes it uniform on ``[0, 1]``:

    ``c_ww``
        outer wall to outer wall, the emission directions that miss the
        inner cylinder (``s >= Rp/Rm``);
    ``c_wi``
        outer wall to the inner cylinder, the directions that hit it
        (``s < Rp/Rm``); the fraction taking this branch is the view factor
        ``F_inner = Rp/Rm``;
    ``c_io``
        inner cylinder outward to the outer wall, over all directions.

    Returns ``(F_inner, c_ww, c_wi, c_io, var_ww, var_wi, var_io)``, all
    per-cell arrays [cm] except the dimensionless view factor and the three
    chord VARIANCES [cm^2] of the sampled distributions, which state how
    much of the class each mean stands for.
    """
    Rp = np.asarray(Rp_cm, dtype=float)
    Rm = np.asarray(Rm_cm, dtype=float)
    nz = Rp.size
    F_inner = Rp / Rm
    s = np.linspace(0.0, 1.0, int(samples))
    c_ww = np.empty(nz)
    c_wi = np.empty(nz)
    c_io = np.empty(nz)
    v_ww = np.zeros(nz)
    v_wi = np.zeros(nz)
    v_io = np.zeros(nz)
    for i in range(nz):
        mu = F_inner[i]
        Rmi = Rm[i]
        Rpi = Rp[i]
        outer = s >= mu
        cw = 2.0 * Rmi * np.sqrt(1.0 - s[outer] ** 2)
        if cw.size:
            c_ww[i] = cw.mean()
            v_ww[i] = cw.var()
        else:
            c_ww[i] = 2.0 * (Rmi - Rpi)
        si = s[~outer]
        if si.size:
            ci = Rmi * np.sqrt(1.0 - si**2) - np.sqrt(
                np.maximum(Rpi**2 - (Rmi * si) ** 2, 0.0)
            )
            c_wi[i] = ci.mean()
            v_wi[i] = ci.var()
        else:
            c_wi[i] = Rmi - Rpi
        cio = np.sqrt(Rmi**2 - (Rpi * s) ** 2) - Rpi * np.sqrt(1.0 - s**2)
        c_io[i] = cio.mean()
        v_io[i] = cio.var()
    return F_inner, c_ww, c_wi, c_io, v_ww, v_wi, v_io


class KN2ZoneJump(KN2Zone):
    """K1b: bounded-chord (jump) annulus kernel.

    The rate-based annulus operators give EXPONENTIAL flight-time
    distributions; the measured chord statistics of the annular duct
    (cosine emission, 2D perpendicular projection) are ten times narrower
    (wall->wall mean^2/var ~ 10, wall->inner ~ 200) -- the exponential's
    long-flight tail is fictitious and over-carries the duct. Here annulus
    flights are DETERMINISTIC jumps at the class-mean chords, all derived
    numerically from the local (Rp, Rm) -- no free parameters:

      wall -> {inner: view factor Rp/Rm, else wall}, chords c_wi / c_ww
      inner (column escapes) -> wall, chord c_io

    Axial displacement per flight: dz = v_z * c/v_perp. Residence is
    apportioned along the traversed cells; end-plane crossings clip, stick
    with the pump probability, and re-emit thermally. The column keeps the
    rate treatment (its gas is volume-mixed by CX).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # per-cell chord classes from the local geometry, cosine-weighted
        (
            self.F_inner,
            self.c_ww,
            self.c_wi,
            self.c_io,
            _,
            _,
            _,
        ) = annulus_chord_classes(self.Rp, self.Rm)
        self.z_edges = self.bg["z_edges"]
        self.zc = 0.5 * (self.z_edges[:-1] + self.z_edges[1:])

    def _fly(self, launch, chord_cm, tal_t, tal_tv):
        """Propagate one flight class. ``launch``: (nz, nvz, nvp) rates
        [atoms/s]. Returns (landing rates per cell (nz, nvz, nvp), end
        losses dict) and accumulates annulus residence tallies."""
        g = self.g
        nz = self.nz
        land = np.zeros_like(launch)
        end_L = np.zeros((g.nvz, g.nvp))
        end_R = np.zeros((g.nvz, g.nvp))
        ze = self.z_edges
        with np.errstate(divide="ignore", invalid="ignore"):
            tof = chord_cm[:, None, None] / np.maximum(g.VP[None, :, :], 1e-30)
        dz = g.VZ[None, :, :] * tof  # (nz, nvz, nvp)
        for i in range(nz):
            rates = launch[i]
            if not np.any(rates):
                continue
            z0 = self.zc[i]
            z1 = np.clip(z0 + dz[i], ze[0], ze[-1])
            crossed_L = (z0 + dz[i]) < ze[0]
            crossed_R = (z0 + dz[i]) > ze[-1]
            # residence: uniform along [min(z0,z1), max(z0,z1)] at speed --
            # time in cell j = overlap_j / |vz| (clipped path)
            lo = np.minimum(z0, z1)
            hi = np.maximum(z0, z1)
            # overlaps: (nvz, nvp, nz)
            ov = np.clip(
                np.minimum(hi[..., None], ze[1:][None, None, :])
                - np.maximum(lo[..., None], ze[:-1][None, None, :]),
                0.0,
                None,
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                t_cell = ov / np.maximum(np.abs(g.VZ[..., None]), 1e-30)
            w = rates[..., None] * t_cell  # atom-seconds per cell
            tal_t += w.sum(axis=(0, 1)) / np.maximum(self.V_ann, 1e-30)
            tal_tv += (w * g.VZ[..., None]).sum(axis=(0, 1)) / np.maximum(
                self.V_ann, 1e-30
            )
            interior = ~(crossed_L | crossed_R)
            if np.any(interior):
                j = np.clip(np.searchsorted(ze, z1) - 1, 0, nz - 1)
                idx = np.nonzero(interior)
                np.add.at(land, (j[idx], idx[0], idx[1]), rates[idx])
            end_L += np.where(crossed_L, rates, 0.0)
            end_R += np.where(crossed_R, rates, 0.0)
        return land, end_L, end_R

    def solve(self):
        g = self.g
        nz = self.nz
        bgs = self.bg["sources"]
        T_s = self.bg["T_s"]
        wall_spec = self.M_wall  # cosine-wall spectrum (vp^2-weighted)

        def inflow(rate, spectrum, area):
            with np.errstate(divide="ignore", invalid="ignore"):
                dens = np.where(
                    np.abs(g.VZ) > 0, spectrum / (np.abs(g.VZ) * area), 0.0
                )
            return rate * dens

        # column boundary inflows (faces are column-radius discs)
        Fc_in_L = inflow(
            bgs.get("cathode_face", 0.0),
            g.half_flux_spectrum(T_s, +1),
            self.A_col[0],
        )
        Fc_in_R = inflow(
            bgs.get("collector_face", 0.0),
            g.half_flux_spectrum(T_WALL_K, -1),
            self.A_col[-1],
        )
        Sc = np.zeros((nz, g.nvz, g.nvp))
        rec = self.bg.get("rec_cell")
        if rec is not None and rec.sum() > 0:
            scale = bgs.get("vol_rec", rec.sum()) / rec.sum()
            for i in range(nz):
                if rec[i] > 0:
                    Sc[i] += scale * rec[i] / self.V_col[i] * self.M_cx[i]
        for name, sign in (("anode_left", -1), ("anode_right", +1)):
            rate = bgs.get(name, 0.0)
            if rate <= 0:
                continue
            j = self.mesh_face - 1 if sign < 0 else self.mesh_face
            j = min(max(j, 0), nz - 1)
            Sc[j] += (
                rate / self.V_col[j] * g.half_flux_spectrum(T_WALL_K, sign)
            )
        # annulus wall-launch rates [atoms/s per bin]: the puff
        wall_launch = np.zeros((nz, g.nvz, g.nvp))
        for iz, rate in puff_launch_bins(bgs, self.z_edges, nz):
            wall_launch[iz] += rate * wall_spec
        inner_launch = np.zeros_like(wall_launch)  # column escapes

        tal_t = np.zeros(nz)
        tal_tv = np.zeros(nz)
        F_tot_c = np.zeros((nz, g.nvz, g.nvp))
        inv_tot = 0.0
        gen = 0
        while gen < self.max_gen:
            inv = 0.0
            # ---- column: 1-zone implicit sweep (no annulus rate coupling)
            Fc = np.zeros((nz, g.nvz, g.nvp))
            for direction in (+1, -1):
                order = range(nz) if direction > 0 else range(nz - 1, -1, -1)
                sel = g.vz > 0 if direction > 0 else g.vz < 0
                F_prev = (Fc_in_L if direction > 0 else Fc_in_R)[sel]
                vz = np.abs(g.vz[sel])[:, None]
                for i in order:
                    face = i if direction > 0 else i + 1
                    if face == self.mesh_face:
                        F_prev = self.transparency * F_prev
                    lam = vz / self.dz[i]
                    denom = lam + (
                        self.nu_ion[i] + self.nu_cx[i] + self.nux[i][None, :]
                    )
                    fc = (Sc[i][sel] + lam * F_prev) / denom
                    Fc[i][sel] = fc
                    F_prev = fc
            F_tot_c += Fc
            inv += float((Fc.sum(axis=(1, 2)) * self.V_col).sum())
            # column escapes -> annulus inner-launches (bin-preserving)
            esc = Fc * self.nux[:, None, :] * self.V_col[:, None, None]
            inner_launch += esc
            # CX relay -> next column generation
            R_cx = self.nu_cx * Fc.sum(axis=(1, 2))
            Sc = R_cx[:, None, None] * self.M_cx
            # column end outgoing -> re-emit with sticking
            Fc_in_L = np.zeros_like(Fc_in_L)
            Fc_in_R = np.zeros_like(Fc_in_R)
            out_L = self.outgoing_flux(Fc, self.A_col, -1)
            out_R = self.outgoing_flux(Fc, self.A_col, +1)
            Fc_in_L += inflow(
                (1.0 - self.s_L) * out_L,
                g.half_flux_spectrum(T_s, +1),
                self.A_col[0],
            )
            Fc_in_R += inflow(
                (1.0 - self.s_R) * out_R,
                g.half_flux_spectrum(T_WALL_K, -1),
                self.A_col[-1],
            )
            # ---- annulus flights this generation
            wl, il = wall_launch, inner_launch
            wall_launch = np.zeros_like(wl)
            inner_launch = np.zeros_like(il)
            inv_flight = float(wl.sum() + il.sum())
            # wall launches branch: F_inner -> inner (chord c_wi), rest -> wall
            landings_w = np.zeros((nz, g.nvz, g.nvp))
            for launch, chord, to_inner in (
                (wl * (1.0 - self.F_inner)[:, None, None], self.c_ww, False),
                (wl * self.F_inner[:, None, None], self.c_wi, True),
                (il, self.c_io, False),
            ):
                if not np.any(launch):
                    continue
                land, end_L, end_R = self._fly(launch, chord, tal_t, tal_tv)
                if to_inner:
                    # entering the column: bin-preserving volume source
                    Sc += land / self.V_col[:, None, None]
                else:
                    landings_w += land
                # end crossings: stick or thermally re-emit into the annulus
                back_L = (1.0 - self.s_L) * end_L.sum()
                back_R = (1.0 - self.s_R) * end_R.sum()
                if back_L > 0:
                    wall_launch[0] += back_L * g.half_flux_spectrum(T_s, +1)
                if back_R > 0:
                    wall_launch[-1] += back_R * g.half_flux_spectrum(
                        T_WALL_K, -1
                    )
            # landed wall atoms re-emit next generation, cosine-wall spectrum
            wall_launch += landings_w.sum(axis=(1, 2))[:, None, None] * wall_spec
            inv_tot += max(inv, 0.0)
            gen += 1
            next_launch = float(wall_launch.sum() + inner_launch.sum())
            if gen == 1:
                self._launch_peak = max(inv_flight, next_launch, 1e-300)
            else:
                self._launch_peak = max(self._launch_peak, next_launch)
            if self.verbose and (gen <= 3 or gen % 25 == 0):
                print(f"  gen {gen:3d}: column inventory {inv:.3e} "
                      f"(total {inv_tot:.3e}), next flights {next_launch:.3e}")
            if (
                gen > 5
                and inv <= self.truncate * max(inv_tot, 1e-300)
                and next_launch <= self.truncate * self._launch_peak
            ):
                break
        nn_ann = tal_t
        with np.errstate(invalid="ignore", divide="ignore"):
            un_ann = np.where(tal_t > 0, tal_tv / np.maximum(tal_t, 1e-300), 0.0)
        return {
            "Fc": F_tot_c,
            "generations": gen,
            "nn_col": F_tot_c.sum(axis=(1, 2)),
            "nn_ann": nn_ann,
            "un_col": _drift(F_tot_c, self.g),
            "un_ann": un_ann,
        }


