"""KN2Zone: deterministic kinetic two-zone neutrals (KINETIC_TWOZONE_PLAN.md K1).

The synthesis instrument between the moment two-zone (NEUTRAL_TWOZONE_PLAN.md)
and the TPMC (`mc_neutrals.py`): two distribution functions on a shared
velocity grid,

    f_c(z, v_z, v_perp)   column   (r < Rp)
    f_a(z, v_z, v_perp)   annulus  (Rp < r < Rm)

with velocity-DEPENDENT radial reduction -- zone exchange and wall collision
rates scale with v_perp, so grazing atoms (small v_perp) stream axially with
few interactions: the duct tail the Fickian moment closure cannot represent
is representable here by construction.

Rates (flux-equivalent forms; angle-averaging a 2D in-plane isotropic
distribution gives <v_perp> = (pi/4) vbar, so each reproduces the
free-molecular flux rate exactly at the Maxwellian zeroth moment -- the
moment model's K_r and the M5 factors are these operators' zeroth moments,
a consistency check, not an input):

    nu_x  (column -> annulus)  = 2 v_perp / (pi Rp)
    nu_x' (annulus -> column)  = 2 v_perp Rp / (pi (Rm^2 - Rp^2))
    nu_w  (annulus wall)       = 2 v_perp Rm / (pi (Rm^2 - Rp^2))

Steady state by generation iteration (KN1D's algorithm class): each
generation is an implicit-upwind march in z per velocity bin with the 2x2
zone coupling solved per cell; the collided fluxes (CX in the column at the
local ion Maxwellian -- the relay; wall accommodation in the annulus at
300 K; end-wall thermal re-emission with pump sticking; anode-mesh
interception) re-enter as the next generation's sources. Ionization
absorbs. Atomic data and the source ledger are the run's own, via
`mc_neutrals.load_background` -- disagreement between instruments is method
error, never input error.

Discrete shifted Maxwellians use a moment-exact projection: analytic bin
masses, then a two-basis linear compensation (KN1DPy
`create_shifted_maxwellian` algorithm, reimplemented; LGPL clone consulted
in scratch only) so the numerically evaluated drift and energy moments hit
their targets to rounding.

Usage:
    python scripts/kn2zone.py RUN.h5 [--window 5 19.5] [--nvz 80 --nvp 24]
        [--truncate 1e-3] [--max-gen 400] [--out PREFIX]
    python scripts/kn2zone.py --selftest
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mc_neutrals import EV, KB, M_HE, T_WALL_K, load_background  # noqa: E402


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
            raise ValueError("empty Maxwellian projection; widen the grid")
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


# ---------------------------------------------------------------- the solver


class KN2Zone:
    def __init__(self, bg, nvz=80, nvp=24, truncate=1e-3, max_gen=400,
                 verbose=True):
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
        Ti_max = float(np.max(bg["Ti"]))
        vmax = 4.0 * np.sqrt(max(Ti_max, 0.5) * EV / M_HE)
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
        volume source at 300 K, volume recombination in the column at the
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
        if bgs.get("puff", 0.0) > 0:
            iz = int(
                np.searchsorted(self.bg["z_edges"], bgs["puff_z"]) - 1
            )
            iz = min(max(iz, 0), nz - 1)
            Sa[iz] += bgs["puff"] / self.V_ann[iz] * self.M_wall
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


# ---------------------------------------------------------------- self-tests


def selftest():
    ok = True
    # 1. Moment-exact shifted Maxwellians across the temperature range.
    g = VGrid(4e6, 4e6, 80, 24, 0.25 * np.sqrt(KB * T_WALL_K / M_HE))
    for T, u in ((0.0259, 0.0), (0.0259, 3e4), (1.0, 5e4), (3.0, -2e5)):
        f = g.maxwellian(T, u)
        s2 = T * EV / M_HE
        n = float(f.sum())
        m1 = float((f * g.VZ).sum())
        m2 = float((f * g.V2).sum())
        e_t = u**2 + 3.0 * s2
        if not (
            abs(n - 1) < 1e-12
            and abs(m1 - u) <= 1e-8 * max(abs(u), np.sqrt(s2))
            and abs(m2 - e_t) <= 1e-8 * e_t
        ):
            print(f"FAIL maxwellian moments T={T} u={u}: "
                  f"n-1={n-1:.2e} du={m1-u:.3e} dE/E={(m2-e_t)/e_t:.2e}")
            ok = False
    print("shifted-Maxwellian moments:", "ok" if ok else "FAIL")

    # 2. Streaming-only attenuation: single-bin beam through uniform
    # absorber matches the discrete closed form (1 + nu dz/vz)^-i, and the
    # closed form converges to exp(-nu z / vz).
    nz, nu, dz = 40, 5.0e3, 10.0
    bg = _slab_background(nz=nz, dz=dz, nu_ion=nu)
    kn = KN2Zone(bg, nvz=16, nvp=6, verbose=False, max_gen=1)
    g = kn.g
    ivz = int(np.argmin(np.abs(g.vz - 2.0e5)))
    ivp = 0
    Fin = np.zeros((g.nvz, g.nvp))
    Fin[ivz, ivp] = 1.0
    kn.nux[:] = 0.0
    kn.nuxp[:] = 0.0
    kn.nuw[:] = 0.0
    Fc, _, _, _ = kn.sweep(
        np.zeros((nz, g.nvz, g.nvp)), np.zeros((nz, g.nvz, g.nvp)),
        Fin, np.zeros_like(Fin), np.zeros_like(Fin), np.zeros_like(Fin),
    )
    vz = g.vz[ivz]
    expected = (1.0 + nu * dz / vz) ** -(np.arange(nz) + 1)
    got = Fc[:, ivz, ivp]
    err = np.max(np.abs(got - expected) / expected)
    print(f"streaming attenuation vs discrete closed form: "
          f"max rel {err:.2e}", "ok" if err < 1e-12 else "FAIL")
    ok = ok and err < 1e-12

    # 3. Closed box: no sinks, reflective ends, walls + zone exchange only.
    # The generation-iterated equilibrium must be axially flat with EQUAL
    # zone densities (detailed balance of nu_x / nu_x') -- the kinetic
    # counterpart of the TPMC's closed-box test.
    bg2 = _slab_background(nz=12, dz=30.0, nu_ion=0.0)
    kn2 = KN2Zone(bg2, nvz=24, nvp=8, verbose=False, max_gen=4000,
                  truncate=1e-7)
    kn2.s_L = 0.0
    kn2.s_R = 0.0
    src = dict(kn2.bg["sources"])
    kn2.bg["sources"] = {
        "puff": 1.0e20, "puff_z": 0.5 * 12 * 30.0,
        "cathode_face": 0.0, "collector_face": 0.0,
    }
    res = kn2.solve()
    # normalized flatness and zone balance over the interior
    nc, na = res["nn_col"], res["nn_ann"]
    flat = np.max(np.abs(na / na.mean() - 1.0))
    balance = np.max(np.abs(nc / na - 1.0))
    print(f"closed box: annulus flatness {flat:.3f}, "
          f"col/ann balance dev {balance:.3f}",
          "ok" if flat < 0.05 and balance < 0.05 else "FAIL (loose gate)")
    ok = ok and flat < 0.05 and balance < 0.05
    kn2.bg["sources"] = src
    return 0 if ok else 1


def _slab_background(nz, dz, nu_ion):
    z_edges = np.arange(nz + 1) * dz
    return {
        "z_edges": z_edges,
        "Rp": np.full(nz, 15.0),
        "Rm": np.full(nz, 50.0),
        "nu_ion": np.full(nz, nu_ion),
        "nu_cx": np.zeros(nz),
        "Ti": np.full(nz, 0.05),
        "u": np.zeros(nz),
        "T_s": T_WALL_K,
        "S_pump_L": 0.0,
        "S_pump_R": 0.0,
        "eta": 0.0,
        "mesh_edge": -999,
        "sources": {},
    }


# ---------------------------------------------------------------- main


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--window", nargs=2, type=float, default=(5.0, 19.5))
    ap.add_argument("--nvz", type=int, default=80)
    ap.add_argument("--nvp", type=int, default=24)
    ap.add_argument("--zone-rates", choices=("cauchy", "flux"),
                    default="cauchy")
    ap.add_argument("--truncate", type=float, default=1e-3)
    ap.add_argument("--max-gen", type=int, default=400)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    if args.selftest:
        sys.exit(selftest())
    if args.run is None:
        ap.error("RUN.h5 required unless --selftest")

    bg = load_background(args.run, tuple(args.window))
    bg["zone_rates"] = args.zone_rates
    kn = KN2Zone(bg, nvz=args.nvz, nvp=args.nvp, truncate=args.truncate,
                 max_gen=args.max_gen)
    res = kn.solve()
    print(f"converged in {res['generations']} generations")

    ze = bg["z_edges"]
    zc = 0.5 * (ze[:-1] + ze[1:])
    print(f"\n{'z[cm]':>7} {'nn_col':>10} {'nn_ann':>10} "
          f"{'un_col[km/s]':>12} {'un_ann':>8}")
    for i in range(0, zc.size, max(1, zc.size // 18)):
        print(f"{zc[i]:7.0f} {res['nn_col'][i]:10.3g} "
              f"{res['nn_ann'][i]:10.3g} {res['un_col'][i]/1e5:12.2f} "
              f"{res['un_ann'][i]/1e5:8.2f}")

    out = args.out or (Path(args.run).stem + "_kn2zone")
    np.savez(
        Path(args.run).parent / f"{out}.npz",
        z=zc, nn_col=res["nn_col"], nn_ann=res["nn_ann"],
        un_col=res["un_col"], un_ann=res["un_ann"],
        generations=res["generations"],
    )
    print(f"\nsaved {out}.npz")


if __name__ == "__main__":
    main()
