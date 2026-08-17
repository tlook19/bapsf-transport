"""K5 end-zone entry gate probe (pre-registered read; diagnostician instrument).

Imports the E2 classes UNCHANGED from scripts/neutral_arch_e2_compare.py and
adds, in a subclass only:
  (i)  end-region (z > Z_B) entry/exit tracking -> residence-time and
       return-event statistics, plus death censoring by channel;
  (ii) origin tagging (birth region AND last-wall-interaction region) and the
       origin split of the annulus->column crossing flux in the mid-machine
       band (500-1000 cm).
The DVM arm is the merged TransientDVM driven exactly as E2's run_dvm drives
it, with a non-invasive _march wrapper adding the face-flux and end-inventory
accumulators.  No repo file is modified.
"""
import sys, time, json
from pathlib import Path
from types import SimpleNamespace
import numpy as np

CABLP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CABLP / "scripts"))
sys.path.insert(0, str(CABLP))

import neutral_arch_e2_compare as e2
from neutral_arch_e2_compare import (
    TransientMC, build_shared, blank_diag, run_dvm, mc_reduce,
    M_HE, EV, KB, wall_emit_cyl, cosine_z, cylinder_spectrum,
)
from mc_neutrals import load_background, T_WALL_K
from cablp.solvers._sim1d.physics.kinetic_dvm import TransientDVM

# ---------------------------------------------------------------- constants
NCLS = 4          # 0 source(z<100) 1 duct-annulus 2 end(z>=Z_B) 3 duct-column
CLS_NAMES = ["source(z<100)", "duct-annulus", "end(z>=1800)", "duct-column"]
MID_LO, MID_HI = 500.0, 1000.0
RES_EDGES = np.concatenate(([0.0], np.logspace(-5, np.log10(3e-2), 26)))


class K5MC(TransientMC):
    """TransientMC + origin tagging + end-region residence tracking."""

    def __init__(self, shared, mode, rng, n_particles, accommodation,
                 elastic_model, dvm_grid_vmax, z_b, ib):
        self.z_b = float(z_b)
        self.ib = int(ib)
        super().__init__(shared, mode, rng, n_particles, accommodation,
                         elastic_model, dvm_grid_vmax)

    # region class from position (r needed only to split duct col/ann)
    def _cls_pos(self, z, in_col):
        c = np.full(z.shape, 1, dtype=np.int8)          # duct-annulus
        c[in_col] = 3                                    # duct-column
        c[z < 100.0] = 0                                 # source region
        c[z >= self.z_b] = 2                             # end region
        return c

    def _launch(self, n_particles):
        super()._launch(n_particles)
        n = self.wgt.size
        z = self.pos[:, 2]
        ic = np.clip(np.searchsorted(self.ze, z) - 1, 0, self.nz - 1)
        r2 = self.pos[:, 0] ** 2 + self.pos[:, 1] ** 2
        in_col = r2 < self.Rp[ic] ** 2
        self.origin_birth = self._cls_pos(z, in_col)
        self.origin_wall = self.origin_birth.copy()
        self.chan = np.repeat(np.arange(len(self.channels)),
                              [c for _, _, c in self.channels]).astype(np.int8)
        self.t_enter = np.where(z >= self.z_b, self.clock, np.nan)
        # region-visit provenance: 0 = never left the duct, 1 = source
        # region (z<100) most recently, 2 = end region most recently
        self.last_special = np.zeros(n, dtype=np.int8)
        self.last_special[z < 100.0] = 1
        self.last_special[z >= self.z_b] = 2
        self.ever_end = (z >= self.z_b).copy()
        # tallies
        nb = self.nbin
        self.k5 = {
            "ac_mid_tot": np.zeros(nb),
            "ac_mid_birth": np.zeros((nb, NCLS)),
            "ac_mid_wall": np.zeros((nb, NCLS)),
            "ac_mid_chan": np.zeros((nb, len(self.channels))),
            "ac_mid_lastvisit": np.zeros((nb, 3)),
            "ac_mid_everend": np.zeros((nb, 2)),
            "end_entry_w": np.zeros(nb),          # atoms entering z>=z_b
            "end_return_w": np.zeros(nb),         # atoms returning to z<z_b
            "res_hist": np.zeros(RES_EDGES.size - 1),   # completed stays
            "res_sum_w": 0.0, "res_sum_wt": 0.0,
            "cens": {k: [0.0, 0.0] for k in ("pump", "ion", "trunc")},
            "seed_end_atoms": float(self.wgt[self.t_enter == 0.0].sum()),
        }

    def _close_res(self, idx, t_now, kind):
        """Close residence for particles idx (censored channel `kind`)."""
        m = np.isfinite(self.t_enter[idx])
        if not m.any():
            return
        j = idx[m]
        tau = t_now[m] - self.t_enter[j]
        w = self.wgt[j]
        acc = self.k5["cens"][kind]
        acc[0] += float(w.sum()); acc[1] += float((w * tau).sum())
        self.t_enter[j] = np.nan

    # ---------------- copied _step with marked K5 insertions ----------------
    def _step(self, rng, ze, Rp, Rm):
        pos, vel, wgt, clock = self.pos, self.vel, self.wgt, self.clock
        N = wgt.size
        nz = self.nz
        prev_z = pos[:, 2].copy()                                   # K5
        v2 = (vel * vel).sum(axis=1)
        speed = np.maximum(np.sqrt(v2), 1.0)
        icell = np.clip(np.searchsorted(ze, pos[:, 2]) - 1, 0, nz - 1)
        r2 = pos[:, 0] ** 2 + pos[:, 1] ** 2
        inside = r2 < Rp[icell] ** 2

        with np.errstate(divide="ignore", invalid="ignore"):
            d_z = np.where(
                vel[:, 2] > 0,
                (ze[icell + 1] - pos[:, 2]) / vel[:, 2],
                np.where(
                    vel[:, 2] < 0, (ze[icell] - pos[:, 2]) / vel[:, 2], np.inf
                ),
            ) * speed
        vxy2 = vel[:, 0] ** 2 + vel[:, 1] ** 2
        b = pos[:, 0] * vel[:, 0] + pos[:, 1] * vel[:, 1]
        Rw = Rm[icell]
        disc = b**2 + vxy2 * (Rw**2 - r2)
        with np.errstate(divide="ignore", invalid="ignore"):
            t_wall = (-b + np.sqrt(np.maximum(disc, 0.0))) / np.where(
                vxy2 > 0, vxy2, np.inf
            )
        d_wall = np.where(vxy2 > 0, t_wall * speed, np.inf)
        Rp_here = Rp[icell]
        disc_p = b**2 + vxy2 * (Rp_here**2 - r2)
        sq_p = np.sqrt(np.maximum(disc_p, 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            t_exit = (-b + sq_p) / np.where(vxy2 > 0, vxy2, np.inf)
            t_enter = (-b - sq_p) / np.where(vxy2 > 0, vxy2, np.inf)
        t_rp = np.where(inside, t_exit, np.where(t_enter > 0, t_enter, np.inf))
        d_rp = np.where(
            (vxy2 > 0) & (disc_p > 0) & (t_rp > 1e-12), t_rp * speed, np.inf
        )
        nu_here = np.where(inside, self.nu_maj[icell], 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            d_coll = np.where(
                nu_here > 0.0,
                -np.log(rng.random(N)) * speed / np.maximum(nu_here, 1e-300),
                np.inf,
            )

        d_geom = np.minimum(np.minimum(d_z, d_wall), np.minimum(d_coll, d_rp))
        dt_geom = d_geom / speed
        dt_left = self.t_end - clock
        dt = np.minimum(dt_geom, dt_left)
        truncated = dt_left <= dt_geom

        zone = np.where(inside, 0, 1).astype(np.int64)
        self._deposit_tracks(
            clock, clock + dt, icell * 2 + zone, wgt, vel[:, 2], v2
        )
        self.n_segments += N

        pos = pos + vel * dt[:, None]
        pos = pos + (vel / speed[:, None]) * 1e-7
        t_ev = clock + dt
        dead = truncated.copy()
        self.resident += float(wgt[truncated].sum())

        # ---- collision events (column only)
        hit_c = (~truncated) & (d_coll <= np.minimum(
            np.minimum(d_z, d_wall), d_rp
        ))
        if hit_c.any():
            idx = np.flatnonzero(hit_c)
            ii = icell[idx]
            u = rng.random(idx.size) * self.nu_maj[ii]
            ionz = u < self.nu_ion[ii]
            dead[idx[ionz]] = True
            self.lost_ion += float(wgt[idx[ionz]].sum())
            self._close_res(idx[ionz], t_ev[idx[ionz]], "ion")       # K5
            rest = idx[~ionz]
            if rest.size:
                self._collide(rng, rest, icell[rest], vel, u[~ionz])

        # ---- cylindrical wall
        hit_w = (~truncated) & (~hit_c) & (d_wall <= np.minimum(d_z, d_rp))
        if hit_w.any():
            idx = np.flatnonzero(hit_w)
            E_in = 0.5 * M_HE * (vel[idx] * vel[idx]).sum(axis=1)
            r_now = np.sqrt(pos[idx, 0] ** 2 + pos[idx, 1] ** 2)
            shrink = (Rm[icell[idx]] * 0.9999) / np.maximum(r_now, 1e-9)
            pos[idx, 0] *= shrink
            pos[idx, 1] *= shrink
            acc = rng.random(idx.size) < self.alpha
            new = vel[idx].copy()
            ia = idx[acc]
            if ia.size:
                new[acc] = wall_emit_cyl(
                    rng, pos[ia, 0], pos[ia, 1], self.sh["T_wall_K"]
                )
            ir = np.flatnonzero(~acc)
            if ir.size:
                j = idx[~acc]
                rr = np.maximum(np.sqrt(pos[j, 0] ** 2 + pos[j, 1] ** 2), 1e-12)
                nx, ny = pos[j, 0] / rr, pos[j, 1] / rr
                vn = new[ir, 0] * nx + new[ir, 1] * ny
                new[ir, 0] -= 2.0 * vn * nx
                new[ir, 1] -= 2.0 * vn * ny
            vel[idx] = new
            # K5: last-wall tag update (any radial-wall interaction)
            zw = pos[idx, 2]
            cw = np.full(idx.size, 1, dtype=np.int8)
            cw[zw < 100.0] = 0
            cw[zw >= self.z_b] = 2
            self.origin_wall[idx] = cw
            E_out = 0.5 * M_HE * (new * new).sum(axis=1)
            kk = self._tbin(t_ev[idx])
            np.add.at(self.diag["wrad_inc"], (kk, icell[idx]), wgt[idx] * E_in)
            np.add.at(self.diag["wrad_ret"], (kk, icell[idx]), wgt[idx] * E_out)
            ph = self._phase(t_ev[idx])
            self._spec_add(
                "spec_rad_acc", "specE_rad_acc", ph[acc], wgt[idx][acc],
                E_out[acc],
            )
            self._spec_add(
                "spec_rad_ref", "specE_rad_ref", ph[~acc], wgt[idx][~acc],
                E_out[~acc],
            )

        # ---- column-surface crossings: the radial exchange channel
        hit_rp = (~truncated) & (~hit_c) & (~hit_w) & (d_rp < d_z)
        if hit_rp.any():
            idx = np.flatnonzero(hit_rp)
            kk = self._tbin(t_ev[idx])
            out_going = inside[idx]
            np.add.at(
                self.diag["exch_ca"], (kk[out_going], icell[idx][out_going]),
                wgt[idx][out_going],
            )
            np.add.at(
                self.diag["exch_ac"], (kk[~out_going], icell[idx][~out_going]),
                wgt[idx][~out_going],
            )
            # K5: mid-band annulus->column crossing flux by origin
            ein = idx[~out_going]
            if ein.size:
                zc_in = pos[ein, 2]
                mid = (zc_in >= MID_LO) & (zc_in < MID_HI)
                if mid.any():
                    m = ein[mid]
                    kb_ = self._tbin(t_ev[m])
                    np.add.at(self.k5["ac_mid_tot"], kb_, wgt[m])
                    np.add.at(self.k5["ac_mid_birth"],
                              (kb_, self.origin_birth[m]), wgt[m])
                    np.add.at(self.k5["ac_mid_wall"],
                              (kb_, self.origin_wall[m]), wgt[m])
                    np.add.at(self.k5["ac_mid_chan"],
                              (kb_, self.chan[m]), wgt[m])
                    np.add.at(self.k5["ac_mid_lastvisit"],
                              (kb_, self.last_special[m]), wgt[m])
                    np.add.at(self.k5["ac_mid_everend"],
                              (kb_, self.ever_end[m].astype(np.int64)),
                              wgt[m])

        # ---- z-edge crossings: ends, mesh, and the annular step face
        hit_z = (~truncated) & (~hit_c) & (~hit_w) & (~hit_rp)
        if hit_z.any():
            idx = np.flatnonzero(hit_z)
            zdir = np.sign(vel[idx, 2])
            edge = np.where(zdir > 0, icell[idx] + 1, icell[idx])
            self._ends(rng, idx, edge, pos, vel, wgt, t_ev, dead)
            self._mesh(rng, idx, edge, icell, pos, vel)
            self._step_face(rng, idx, edge, zdir, icell, pos, vel, wgt, t_ev)

        # ---- K5: end-region entry/exit bookkeeping (before compaction)
        new_z = pos[:, 2]
        tr_idx = np.flatnonzero(truncated); self._close_res(tr_idx, t_ev[tr_idx], "trunc")
        live = ~dead
        entered = live & ~np.isfinite(self.t_enter) & (new_z >= self.z_b)
        if entered.any():
            self.t_enter[entered] = t_ev[entered]
            np.add.at(self.k5["end_entry_w"], self._tbin(t_ev[entered]),
                      wgt[entered])
        exited = live & np.isfinite(self.t_enter) & (new_z < self.z_b)
        if exited.any():
            j = np.flatnonzero(exited)
            tau = t_ev[j] - self.t_enter[j]
            w = wgt[j]
            self.k5["res_hist"] += np.histogram(
                tau, bins=RES_EDGES, weights=w)[0]
            self.k5["res_sum_w"] += float(w.sum())
            self.k5["res_sum_wt"] += float((w * tau).sum())
            np.add.at(self.k5["end_return_w"], self._tbin(t_ev[j]), w)
            self.t_enter[j] = np.nan

        # K5: region-visit provenance update
        in_src = new_z < 100.0
        in_end = new_z >= self.z_b
        self.last_special[in_src] = 1
        self.last_special[in_end] = 2
        self.ever_end |= in_end

        alive = ~dead
        self.pos = pos[alive]
        self.vel = vel[alive]
        self.wgt = wgt[alive]
        self.clock = t_ev[alive]
        self.origin_birth = self.origin_birth[alive]                 # K5
        self.origin_wall = self.origin_wall[alive]                   # K5
        self.chan = self.chan[alive]                                 # K5
        self.t_enter = self.t_enter[alive]                           # K5
        self.last_special = self.last_special[alive]                 # K5
        self.ever_end = self.ever_end[alive]                         # K5

    # ---------------- copied _ends with pump-censoring + wall tags ----------
    def _ends(self, rng, idx, edge, pos, vel, wgt, t_ev, dead):
        for j, (at, sign) in enumerate(
            ((edge == 0, +1.0), (edge == self.nz, -1.0))
        ):
            e = idx[at]
            if e.size == 0:
                continue
            E_in = 0.5 * M_HE * (vel[e] * vel[e]).sum(axis=1)
            kk = self._tbin(t_ev[e])
            np.add.at(self.diag["wend_inc"], (kk, j), wgt[e] * E_in)
            stick = rng.random(e.size) < self.s_end[j]
            dead[e[stick]] = True
            self.lost_pump += float(wgt[e[stick]].sum())
            if j == 1:                                               # K5
                self._close_res(e[stick], t_ev[e[stick]], "pump")
            keep = e[~stick]
            if keep.size == 0:
                continue
            acc = rng.random(keep.size) < self.alpha
            new = vel[keep].copy()
            if acc.any():
                new[acc] = cosine_z(
                    rng, int(acc.sum()), self.T_end[j], sign
                )
            if (~acc).any():
                new[~acc, 2] = -new[~acc, 2]
            vel[keep] = new
            pos[keep, 2] = np.clip(pos[keep, 2], 1e-6, self.ze[-1] - 1e-6)
            self.origin_wall[keep] = 0 if j == 0 else 2              # K5
            E_out = 0.5 * M_HE * (new * new).sum(axis=1)
            np.add.at(
                self.diag["wend_ret"], (self._tbin(t_ev[keep]), j),
                wgt[keep] * E_out,
            )
            ph = self._phase(t_ev[keep])
            sub = np.full(keep.size, j, dtype=np.int64)
            self._spec_add(
                "spec_end_acc", "specE_end_acc", ph[acc], wgt[keep][acc],
                E_out[acc], sub=sub[acc],
            )
            self._spec_add(
                "spec_end_ref", "specE_end_ref", ph[~acc], wgt[keep][~acc],
                E_out[~acc], sub=sub[~acc],
            )

    def _mesh(self, rng, idx, edge, icell, pos, vel):
        m = (edge == self.mesh_face) & (edge != 0) & (edge != self.nz)
        e = idx[m]
        if e.size == 0:
            return
        blocked = rng.random(e.size) > self.transparency
        b = e[blocked]
        if b.size == 0:
            return
        back_cell = icell[b]
        pos[b] = self._in_cell(back_cell, self.Rp)
        vel[b] = cylinder_spectrum(rng, b.size, self.sh["T_wall_K"])
        self.origin_wall[b] = 0                                      # K5

    def _step_face(self, rng, idx, edge, zdir, icell, pos, vel, wgt, t_ev):
        interior = (edge > 0) & (edge < self.nz)
        e = idx[interior]
        if e.size == 0:
            return
        dest = np.where(zdir[interior] > 0, edge[interior], edge[interior] - 1)
        r = np.sqrt(pos[e, 0] ** 2 + pos[e, 1] ** 2)
        hit = r > self.Rm[dest]
        h = e[hit]
        if h.size == 0:
            return
        sign = -zdir[interior][hit]
        E_in = 0.5 * M_HE * (vel[h] * vel[h]).sum(axis=1)
        acc = rng.random(h.size) < self.alpha
        new = vel[h].copy()
        if acc.any():
            new[acc] = cosine_z(
                rng, int(acc.sum()), self.sh["T_wall_K"], sign[acc]
            )
        if (~acc).any():
            new[~acc, 2] = -new[~acc, 2]
        vel[h] = new
        pos[h, 2] += sign * 1e-6
        self.origin_wall[h] = 2                                      # K5
        E_out = 0.5 * M_HE * (new * new).sum(axis=1)
        kk = self._tbin(t_ev[h])
        np.add.at(self.diag["wstep_inc"], (kk, icell[h]), wgt[h] * E_in)
        np.add.at(self.diag["wstep_ret"], (kk, icell[h]), wgt[h] * E_out)


# ------------------------------------------------------------- DVM driver
def run_dvm_k5(shared, dt, nvz, nvp, ib, ib2, end_mask, end_mask2, mid_mask,
               seed_col=None, seed_ann=None, sources=None):
    """E2's run_dvm plus per-bin end-region flux/inventory accumulators.

    ``seed_col``/``seed_ann``/``sources`` override the shared inputs, so the
    LINEAR (frozen-background) engine can be decomposed by origin piece.
    """
    diag, dvm_holder = {}, {}
    nz, nbin, bin_s = shared["nz"], shared["nbin"], shared["bin_s"]
    t_end, t_switch = shared["t_end"], shared["t_switch"]
    dvm = TransientDVM(
        geometry=shared["geometry"], nvz=nvz, nvp=nvp,
        accommodation=1.0, elastic_model="phelps_iso",
        transparency=shared["transparency"], mesh_face=shared["mesh_face"],
        s_L=shared["s_L"], s_R=shared["s_R"], T_wall_K=shared["T_wall_K"],
        Ti_cap_eV=shared["Ti_cap_eV"], u_cap_cm_s=shared["u_cap_cm_s"],
    )
    sc = shared["seed_col"] if seed_col is None else seed_col
    sa = shared["seed_ann"] if seed_ann is None else seed_ann
    src_menu = shared["sources"] if sources is None else sources
    dvm.seed_from_density(sc, sa)
    g = dvm.g
    captured = {}
    real_march = dvm._march

    def march(*a, **kw):
        out = real_march(*a, **kw)
        captured["res"] = out
        return out

    dvm._march = march
    neg = g.vz < 0
    pos_ = g.vz > 0
    wneg = np.abs(g.vz[neg])[:, None]
    wpos = g.vz[pos_][:, None]
    acc = {k: np.zeros(nbin) for k in (
        "N_end", "N_end2", "phi_ret", "phi_in", "phi_ret2", "phi_in2",
        "pump_R", "ion_end", "exch_ac_mid", "exch_ac_mid2")}
    nu_ion = shared["plasma"]["nu_ion"]
    Vc, Va = dvm.V_col, dvm.V_ann

    def inv(mask):
        return float(((dvm.f_c.sum(axis=(1, 2)) * Vc
                       + dvm.f_a.sum(axis=(1, 2)) * Va)[mask]).sum())

    prevN, prevN2 = inv(end_mask), inv(end_mask2)
    nsteps = int(round(t_end / dt))
    for step in range(nsteps):
        t0 = step * dt
        k = min(int(t0 / bin_s), nbin - 1)
        src = src_menu if t0 < t_switch - 1e-15 else None
        led = dvm.update(dt, sources=src, T_s_K=shared["T_s_K"],
                         **shared["plasma"])
        f_c_m, f_a_m, mesh_c, mesh_a, out = captured["res"]
        curN, curN2 = inv(end_mask), inv(end_mask2)
        acc["N_end"][k] += 0.5 * (prevN + curN) * dt
        acc["N_end2"][k] += 0.5 * (prevN2 + curN2) * dt
        prevN, prevN2 = curN, curN2
        # face fluxes at the marched state (the flux the march actually took)
        for tag, i_b in (("", ib), ("2", ib2)):
            phi_r = float((f_c_m[i_b][neg] * wneg).sum() * dvm.face_c[i_b]
                          + (f_a_m[i_b][neg] * wneg).sum() * dvm.face_a[i_b])
            phi_i = float((f_c_m[i_b - 1][pos_] * wpos).sum() * dvm.face_c[i_b]
                          + (f_a_m[i_b - 1][pos_] * wpos).sum()
                          * dvm.face_a[i_b])
            acc["phi_ret" + tag][k] += phi_r * dt
            acc["phi_in" + tag][k] += phi_i * dt
        acc["pump_R"][k] += led["loss_pump_R"]
        acc["ion_end"][k] += float(
            (nu_ion[end_mask] * f_c_m[end_mask].sum(axis=(1, 2))
             * Vc[end_mask]).sum() * dt)
        acc["exch_ac_mid"][k] += float(
            (dvm.nuxp[mid_mask][:, None, :] * f_a_m[mid_mask]).sum(
                axis=(1, 2)).dot(Va[mid_mask]) * dt)
    for k_ in acc:
        acc[k_] /= bin_s
    return acc, dvm


# ------------------------------------------------------------------- main
def main():
    t_wall0 = time.perf_counter()
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--particles", type=int, default=500_000)
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--nvz", type=int, default=96)
    ap.add_argument("--nvp", type=int, default=32)
    ap.add_argument("--dvm-dt", type=float, default=1.0e-5)
    ap.add_argument("--seed", type=int, default=20260805)
    ap.add_argument("--out", default=str(CABLP / "scripts/k5gate_raw.npz"))
    ap.add_argument("--progress", type=int, default=0)
    a = ap.parse_args()

    args = SimpleNamespace(
        t_end_ms=6.0, t_switch_ms=3.0, bin_ms=0.5, seed_state=True,
        particles=a.particles, batches=a.batches, seed=a.seed,
        progress=a.progress,
    )
    bg = load_background(str(CABLP / "scripts/es1_k3a_cal2_nx240.h5"),
                         (5.0, 19.5))
    shared = build_shared(bg, args)
    ze, zc = shared["z_edges"], shared["z_cm"]
    ib = int(np.argmin(np.abs(ze - 1800.0)))
    ib2 = int(np.argmin(np.abs(ze - 1900.0)))
    z_b, z_b2 = float(ze[ib]), float(ze[ib2])
    end_mask = zc >= z_b
    end_mask2 = zc >= z_b2
    mid_mask = (zc >= MID_LO) & (zc < MID_HI)
    print(f"z_b={z_b} (edge {ib}), z_b2={z_b2} (edge {ib2}), "
          f"end cells {end_mask.sum()}/{end_mask2.sum()}, "
          f"total source {shared['total_rate']:.4g} atoms/s", flush=True)

    print("DVM arm (full) ...", flush=True)
    t0 = time.perf_counter()
    dvm_acc, dvm = run_dvm_k5(shared, a.dvm_dt, a.nvz, a.nvp, ib, ib2,
                              end_mask, end_mask2, mid_mask)
    print(f"  DVM full done in {time.perf_counter() - t0:.1f} s", flush=True)
    dvm_vmax = float(dvm.g.vz.max())

    # ---- linear origin decomposition of the SAME DVM configuration.
    # The frozen-background TransientDVM update is affine in (f, sources):
    # every operator (march, zone exchange, wall/end returns, CX/elastic
    # rebirth at the frozen local Maxwellian, mesh, pump) is linear in f,
    # and the external sources add.  So full = A + B + C exactly, checked
    # below to roundoff.
    z0 = np.zeros(shared["nz"])
    m_src = zc < 100.0
    m_duct = (zc >= 100.0) & (zc < z_b)
    m_end = zc >= z_b
    empty_src = {"puff": z0, "recombination": z0, "anode": z0,
                 "cathode_face": 0.0, "collector_face": 0.0}
    full_src = shared["sources"]
    pieces = {}
    # A: source-region origin = seeds with z<100 + all z<100 source channels
    #    (+ the distributed recombination channel, 0.05% of the menu; stated)
    srcA = dict(full_src); srcA["collector_face"] = 0.0
    pieces["A_source"] = dict(
        seed_col=np.where(m_src, shared["seed_col"], 0.0),
        seed_ann=np.where(m_src, shared["seed_ann"], 0.0), sources=srcA)
    # B: duct-seed origin = seeds with 100 <= z < z_b, no sources
    pieces["B_duct"] = dict(
        seed_col=np.where(m_duct, shared["seed_col"], 0.0),
        seed_ann=np.where(m_duct, shared["seed_ann"], 0.0),
        sources=empty_src)
    # C: end origin = seeds with z >= z_b + the collector-face recycle source
    srcC = dict(empty_src); srcC["collector_face"] = full_src["collector_face"]
    pieces["C_end"] = dict(
        seed_col=np.where(m_end, shared["seed_col"], 0.0),
        seed_ann=np.where(m_end, shared["seed_ann"], 0.0), sources=srcC)
    dvm_pieces = {}
    for name, kw in pieces.items():
        t0 = time.perf_counter()
        pa, _ = run_dvm_k5(shared, a.dvm_dt, a.nvz, a.nvp, ib, ib2,
                           end_mask, end_mask2, mid_mask, **kw)
        dvm_pieces[name] = pa
        print(f"  DVM piece {name} done in {time.perf_counter()-t0:.1f} s",
              flush=True)
    lin = max(
        float(np.max(np.abs(sum(p["exch_ac_mid"] for p in
                                dvm_pieces.values())
                            - dvm_acc["exch_ac_mid"]))
              / max(np.max(np.abs(dvm_acc["exch_ac_mid"])), 1e-300)),
        float(np.max(np.abs(sum(p["N_end"] for p in dvm_pieces.values())
                            - dvm_acc["N_end"]))
              / max(np.max(np.abs(dvm_acc["N_end"])), 1e-300)))
    print(f"  linearity residual (max rel, exch_ac_mid & N_end): {lin:.2e}",
          flush=True)

    batches, k5s, metas = [], [], []
    for kb in range(a.batches):
        seed = a.seed + 1000 * kb
        rng = np.random.default_rng(seed)
        mc = K5MC(shared, "kinetic", rng, a.particles, 1.0, "phelps_iso",
                  dvm_vmax, z_b, ib)
        t0 = time.perf_counter()
        diag = mc.run(progress=a.progress)
        wall = time.perf_counter() - t0
        closure = (mc.lost_ion + mc.lost_pump + mc.resident + mc.stuck) \
            / max(mc.launched_atoms, 1e-300)
        print(f"  MC batch {kb+1}/{a.batches} seed={seed}: "
              f"{mc.n_segments/1e6:.1f}e6 seg in {wall:.1f} s, "
              f"viol {mc.majorant_violations}, closure {closure:.9f}",
              flush=True)
        batches.append(diag)
        k5s.append(mc.k5)
        metas.append({"seed": seed, "segments": mc.n_segments,
                      "violations": mc.majorant_violations,
                      "closure": closure,
                      "channels": [(n, c) for n, _, c in mc.channels]})
    mean, sem = mc_reduce(batches)

    np.savez(a.out,
             dvm_acc=json.dumps({k: v.tolist() for k, v in dvm_acc.items()}),
             dvm_pieces=json.dumps({n: {k: v.tolist() for k, v in p.items()}
                                    for n, p in dvm_pieces.items()}),
             linearity=lin,
             k5=json.dumps([{k: (v.tolist() if isinstance(v, np.ndarray)
                                 else v) for k, v in d.items()} for d in k5s]),
             mc_n_col=np.stack([b["n_col"] for b in batches]),
             mc_n_ann=np.stack([b["n_ann"] for b in batches]),
             mc_exch_ac=np.stack([b["exch_ac"] for b in batches]),
             V_col=shared["V_col"], V_ann=shared["V_ann"],
             z_cm=zc, z_edges=ze, meta=json.dumps(metas),
             params=json.dumps(vars(a)),
             z_b=z_b, z_b2=z_b2, ib=ib, ib2=ib2)
    print(f"saved {a.out}; total wall {time.perf_counter()-t_wall0:.1f} s",
          flush=True)


if __name__ == "__main__":
    main()
