"""K5 build read: the bounded-chord annulus vs the step-face ray-traced MC.

Reports (does NOT gate) the E2 numbers the K5 brief names, for the rate arm
and for the bounded-chord arm, on the SAME frozen background and the same
schedule the pre-registered K5 probe used:

  (1) mid-machine (500-1000 cm) matched-time n_ann deviation vs the MC
  (2) mid-band annulus->column crossing flux, DVM/MC ratio
  (3) source-region provenance fraction of that mid-band flux, from the
      exact linear origin decomposition of the frozen-background engine
  (4) end-region Little's-law residence N_end / return flux

The MC reference is arm-independent, so it is read from the probe's frozen
raw file rather than re-run: scripts/k5gate_raw.npz (8 x 500000 histories,
base seed 20260805).

ESTIMATOR NOTE. Under the rate arm the annulus crosses a plane by ADVECTION,
so the end-plane fluxes are the marched face fluxes -- the probe's own
estimator, reproduced here as a control. Under the bounded-chord arm the
annulus has no advective flux: it crosses a plane by JUMPING over it, so the
annulus half of the end-plane flux is counted as flight displacements that
straddle the plane. Both count one-way particle crossings of z = z_b per
second, which is what the MC's own entry/return tallies count.

Usage (from <checkout>/cablp, with PYTHONPATH set to that same cablp):
    python scripts/k5build_measure.py --out k5build_report.txt
"""

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

# The solver import comes FIRST and from THIS checkout: the E2 scripts put
# their own checkout's package at sys.path[0] on import, which is exactly the
# PYTHONPATH trap. Importing the package first pins it in sys.modules.
from cablp.solvers._sim1d.physics.kinetic_dvm import (  # noqa: E402
    FLIGHT_CLASSES,
    TransientDVM,
)

MAIN = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import neutral_arch_e2_compare as e2  # noqa: E402
from neutral_arch_e2_compare import build_shared  # noqa: E402
from mc_neutrals import load_background  # noqa: E402

BACKGROUND = MAIN / "scripts/es1_k3a_cal2_nx240.h5"
MC_RAW = MAIN / "scripts/k5gate_raw.npz"
WINDOW = (5.0, 19.5)
MID_LO, MID_HI = 500.0, 1000.0
ON = slice(4, 6)          # the developed-ON window, bins 2.0-3.0 ms
T95 = 2.365               # two-sided 95% Student-t, 7 dof (8 batches)


def mstats(vals):
    v = np.asarray(vals, dtype=float)
    return v.mean(axis=0), T95 * v.std(axis=0, ddof=1) / np.sqrt(v.shape[0])


def crossing_masks(flights, ib):
    """Boolean straddle masks of face ``ib`` for the two half displacements.

    A half displacement from cell ``i`` to cell ``j`` crosses face ``ib``
    forward when ``i < ib <= j`` and backward when ``j < ib <= i``. The
    kernel stores both maps as flat destination indices, so the cell is
    recovered by integer division.
    """
    out = {}
    nz = flights.nz
    nb = flights.n_flat // nz
    cell = np.arange(nz)[:, None, None]
    for name in FLIGHT_CLASSES:
        m = {}
        for tag, flat in (("place", flights.hold_flat[name]),
                          ("route", flights.dest_flat[name])):
            j = (flat // nb).reshape(flights.shape)
            m[tag + "_fwd"] = (cell < ib) & (j >= ib)
            m[tag + "_bwd"] = (j < ib) & (cell >= ib)
        out[name] = m
    return out


def run_arm(shared, dt, nvz, nvp, ib, end_mask, mid_mask, flights_mode,
            seed_col=None, seed_ann=None, sources=None):
    """Advance one DVM arm over the E2 schedule and accumulate the read."""
    nz, nbin, bin_s = shared["nz"], shared["nbin"], shared["bin_s"]
    t_end, t_switch = shared["t_end"], shared["t_switch"]
    dvm = TransientDVM(
        geometry=shared["geometry"], nvz=nvz, nvp=nvp,
        accommodation=1.0, elastic_model="phelps_iso",
        annulus_flights=flights_mode,
        transparency=shared["transparency"], mesh_face=shared["mesh_face"],
        s_L=shared["s_L"], s_R=shared["s_R"], T_wall_K=shared["T_wall_K"],
        Ti_cap_eV=shared["Ti_cap_eV"], u_cap_cm_s=shared["u_cap_cm_s"],
    )
    jump = dvm.flights is not None
    dvm.seed_from_density(
        shared["seed_col"] if seed_col is None else seed_col,
        shared["seed_ann"] if seed_ann is None else seed_ann,
    )
    src_menu = shared["sources"] if sources is None else sources
    g = dvm.g
    captured = {}
    real_march = dvm._march

    def march(*a, **kw):
        out = real_march(*a, **kw)
        captured["res"] = out
        return out

    dvm._march = march

    tally = {"ret": 0.0, "in": 0.0}
    if jump:
        masks = crossing_masks(dvm.flights, ib)
        real_place, real_route = dvm.flights.place, dvm.flights.route

        def place(name, launched):
            m = masks[name]
            tally["in"] += float(launched[m["place_fwd"]].sum())
            tally["ret"] += float(launched[m["place_bwd"]].sum())
            return real_place(name, launched)

        def route(name, counts):
            passed = counts * dvm.flights.w_pass[name]
            m = masks[name]
            tally["in"] += float(passed[m["route_fwd"]].sum())
            tally["ret"] += float(passed[m["route_bwd"]].sum())
            return real_route(name, counts)

        dvm.flights.place = place
        dvm.flights.route = route

    neg, pos = g.vz < 0, g.vz > 0
    wneg = np.abs(g.vz[neg])[:, None]
    wpos = g.vz[pos][:, None]
    acc = {k: np.zeros(nbin) for k in
           ("N_end", "phi_ret", "phi_in", "pump_R", "ion_end", "exch_ac_mid")}
    n_ann = np.zeros((nbin, nz))
    n_col = np.zeros((nbin, nz))
    nu_ion = shared["plasma"]["nu_ion"]
    Vc, Va = dvm.V_col, dvm.V_ann

    def end_inventory():
        return float(((dvm.f_c.sum(axis=(1, 2)) * Vc
                       + dvm.f_a.sum(axis=(1, 2)) * Va)[end_mask]).sum())

    prevN = end_inventory()
    prev_na = dvm.annulus_density()
    prev_nc = dvm.column_density()
    nsteps = int(round(t_end / dt))
    for step in range(nsteps):
        t0 = step * dt
        k = min(int(t0 / bin_s), nbin - 1)
        src = src_menu if t0 < t_switch - 1e-15 else None
        tally["ret"] = tally["in"] = 0.0
        led = dvm.update(dt, sources=src, T_s_K=shared["T_s_K"],
                         **shared["plasma"])
        f_c_m, f_a_m, mesh_c, mesh_a, out = captured["res"]
        curN = end_inventory()
        acc["N_end"][k] += 0.5 * (prevN + curN) * dt
        prevN = curN
        cur_na, cur_nc = dvm.annulus_density(), dvm.column_density()
        n_ann[k] += 0.5 * (prev_na + cur_na) * dt
        n_col[k] += 0.5 * (prev_nc + cur_nc) * dt
        prev_na, prev_nc = cur_na, cur_nc
        # column half of the end-plane flux: the advective face flux the
        # march actually took, in both arms
        phi_r = float((f_c_m[ib][neg] * wneg).sum() * dvm.face_c[ib])
        phi_i = float((f_c_m[ib - 1][pos] * wpos).sum() * dvm.face_c[ib])
        if jump:
            phi_r += tally["ret"] / dt
            phi_i += tally["in"] / dt
            acc["exch_ac_mid"][k] += float(
                dvm.last_flight["annulus_to_column"][mid_mask].sum()
            )
        else:
            phi_r += float((f_a_m[ib][neg] * wneg).sum() * dvm.face_a[ib])
            phi_i += float((f_a_m[ib - 1][pos] * wpos).sum() * dvm.face_a[ib])
            acc["exch_ac_mid"][k] += float(
                (dvm.nuxp[mid_mask][:, None, :] * f_a_m[mid_mask]).sum(
                    axis=(1, 2)).dot(Va[mid_mask]) * dt)
        acc["phi_ret"][k] += phi_r * dt
        acc["phi_in"][k] += phi_i * dt
        acc["pump_R"][k] += led["loss_pump_R"]
        acc["ion_end"][k] += float(
            (nu_ion[end_mask] * f_c_m[end_mask].sum(axis=(1, 2))
             * Vc[end_mask]).sum() * dt)
    for key in acc:
        acc[key] /= bin_s
    acc["n_ann"] = n_ann / bin_s
    acc["n_col"] = n_col / bin_s
    return acc


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--nvz", type=int, default=96)
    p.add_argument("--nvp", type=int, default=32)
    p.add_argument("--dvm-dt", type=float, default=1.0e-5)
    p.add_argument("--out", default="k5build_report.txt")
    a = p.parse_args(argv)

    raw = np.load(MC_RAW, allow_pickle=True)
    k5 = json.loads(str(raw["k5"]))
    for b in k5:
        for key, v in b.items():
            if isinstance(v, list):
                b[key] = np.array(v)
    zc, ze = raw["z_cm"], raw["z_edges"]
    z_b, ib = float(raw["z_b"]), int(raw["ib"])
    V_ann = raw["V_ann"]
    end_mask = zc >= z_b
    mid_mask = (zc >= MID_LO) & (zc < MID_HI)

    args = SimpleNamespace(t_end_ms=6.0, t_switch_ms=3.0, bin_ms=0.5,
                           seed_state=True, particles=0, batches=0, seed=0,
                           progress=0)
    bg = load_background(str(BACKGROUND), WINDOW)
    shared = build_shared(bg, args)
    assert shared["nz"] == zc.size, (shared["nz"], zc.size)
    bw = 0.5e-3

    # --- MC reference quantities
    mc_n_ann = raw["mc_n_ann"]                              # (B, nbin, nz)
    mc_inv = (raw["mc_n_col"][:, :, end_mask] * raw["V_col"][end_mask]
              + mc_n_ann[:, :, end_mask] * V_ann[end_mask]).sum(axis=2)
    mc_ret = np.stack([b["end_return_w"] for b in k5]) / bw
    mc_ac = np.stack([b["ac_mid_tot"] for b in k5]) / bw
    mc_ann_mid = np.stack([
        (mc_n_ann[i][:, mid_mask] * V_ann[mid_mask]).sum(axis=1)
        / V_ann[mid_mask].sum() for i in range(mc_n_ann.shape[0])
    ])
    mc_ann_mid_m, mc_ann_mid_h = mstats(mc_ann_mid)
    mc_ac_m, mc_ac_h = mstats(mc_ac[:, ON].mean(axis=1))
    mc_inv_m, mc_inv_h = mstats(mc_inv[:, ON].mean(axis=1))
    mc_ret_m, mc_ret_h = mstats(mc_ret[:, ON].mean(axis=1))
    mc_tau_m, mc_tau_h = mstats(
        mc_inv[:, ON].mean(axis=1) / mc_ret[:, ON].mean(axis=1)
    )

    L = []

    def say(line=""):
        print(line, flush=True)
        L.append(line)

    say("K5 BUILD READ -- bounded-chord annulus vs the step-face MC "
        "(REPORT, not a gate)")
    say(f"background {BACKGROUND.name}, frozen over the breakdown-relative "
        f"window {WINDOW[0]}-{WINDOW[1]} ms; E2 schedule sources ON 0-3 ms, "
        f"OFF 3-6 ms, 12 x 0.5 ms bins")
    say(f"DVM {a.nvz}x{a.nvp} dt={a.dvm_dt:g}; MC reference read frozen from "
        f"{MC_RAW.name} (8 x 500000 histories, base seed 20260805)")
    say(f"end-region boundary z_b={z_b:.1f} cm (edge {ib}), "
        f"{int(end_mask.sum())} end cells; mid band {MID_LO:.0f}-{MID_HI:.0f} "
        f"cm, {int(mid_mask.sum())} cells")
    say("acceptance numbers are RATIFICATION-PENDING: reported for the "
        "record, nothing was tuned to them")
    say()

    results = {}
    for mode in ("rates", "bounded_chord"):
        t0 = time.perf_counter()
        full = run_arm(shared, a.dvm_dt, a.nvz, a.nvp, ib, end_mask,
                       mid_mask, mode)
        say(f"[{mode}] full arm in {time.perf_counter() - t0:.1f} s")
        z0 = np.zeros(shared["nz"])
        m_src = zc < 100.0
        m_duct = (zc >= 100.0) & (zc < z_b)
        m_end = zc >= z_b
        empty = {"puff": z0, "recombination": z0, "anode": z0,
                 "cathode_face": 0.0, "collector_face": 0.0}
        menu = shared["sources"]
        srcA = dict(menu)
        srcA["collector_face"] = 0.0
        srcC = dict(empty)
        srcC["collector_face"] = menu["collector_face"]
        pieces = {
            "A_source": dict(seed_col=np.where(m_src, shared["seed_col"], 0.0),
                             seed_ann=np.where(m_src, shared["seed_ann"], 0.0),
                             sources=srcA),
            "B_duct": dict(seed_col=np.where(m_duct, shared["seed_col"], 0.0),
                           seed_ann=np.where(m_duct, shared["seed_ann"], 0.0),
                           sources=empty),
            "C_end": dict(seed_col=np.where(m_end, shared["seed_col"], 0.0),
                          seed_ann=np.where(m_end, shared["seed_ann"], 0.0),
                          sources=srcC),
        }
        parts = {}
        for name, kw in pieces.items():
            t0 = time.perf_counter()
            parts[name] = run_arm(shared, a.dvm_dt, a.nvz, a.nvp, ib,
                                  end_mask, mid_mask, mode, **kw)
            say(f"[{mode}] piece {name} in {time.perf_counter() - t0:.1f} s")
        lin = max(
            float(np.max(np.abs(sum(p["exch_ac_mid"] for p in parts.values())
                                - full["exch_ac_mid"]))
                  / max(np.max(np.abs(full["exch_ac_mid"])), 1e-300)),
            float(np.max(np.abs(sum(p["N_end"] for p in parts.values())
                                - full["N_end"]))
                  / max(np.max(np.abs(full["N_end"])), 1e-300)),
        )
        results[mode] = (full, parts, lin)
        say(f"[{mode}] linearity residual (max rel, exch_ac_mid & N_end): "
            f"{lin:.2e}")
    say()

    say("(1) MID-MACHINE (500-1000 cm) ANNULUS DENSITY, matched time")
    say(f"    V_ann-weighted band mean; deviation = (DVM - MC) / MC")
    hdr = (f"{'bin ms':>9} | {'MC n_ann':>10} {'+-95%':>9} "
           f"{'rates':>10} {'dev %':>8} | {'bnd-chord':>10} {'dev %':>8}")
    say("    " + hdr)
    say("    " + "-" * len(hdr))
    worst = {}
    for mode in ("rates", "bounded_chord"):
        dv = np.array([
            (results[mode][0]["n_ann"][b][mid_mask] * V_ann[mid_mask]).sum()
            / V_ann[mid_mask].sum() for b in range(mc_ann_mid.shape[1])
        ])
        results[mode][0]["mid_ann"] = dv
        pct = 100.0 * (dv - mc_ann_mid_m) / mc_ann_mid_m
        worst[mode] = (float(np.max(np.abs(pct))),
                       int(np.argmax(np.abs(pct))), pct)
    for b in range(mc_ann_mid.shape[1]):
        say(f"    {b*0.5:.1f}-{(b+1)*0.5:.1f}".ljust(13)
            + f"| {mc_ann_mid_m[b]:10.3e} {mc_ann_mid_h[b]:9.1e} "
            f"{results['rates'][0]['mid_ann'][b]:10.3e} "
            f"{worst['rates'][2][b]:+8.2f} | "
            f"{results['bounded_chord'][0]['mid_ann'][b]:10.3e} "
            f"{worst['bounded_chord'][2][b]:+8.2f}")
    for mode in ("rates", "bounded_chord"):
        w, at, pct = worst[mode]
        say(f"    WORST matched-time |dev| [{mode:>13}]: {pct[at]:+.2f} % "
            f"at bin {at} ({at*0.5:.1f}-{(at+1)*0.5:.1f} ms)")
    say()

    say("(2) MID-BAND ANNULUS->COLUMN CROSSING FLUX, bins 2.0-3.0 ms")
    say(f"    MC {mc_ac_m:.4e} +-{mc_ac_h:.1e} atoms/s")
    for mode in ("rates", "bounded_chord"):
        v = results[mode][0]["exch_ac_mid"][ON].mean()
        say(f"    {mode:>13}  {v:.4e} atoms/s   ratio {v / mc_ac_m:.3f}")
    say()

    say("(3) SOURCE-REGION PROVENANCE of that mid-band flux "
        "(exact linear origin decomposition)")
    birth = np.stack([b["ac_mid_birth"] for b in k5])[:, ON].sum(axis=1)
    tot = np.stack([b["ac_mid_tot"] for b in k5])[:, ON].sum(axis=1)
    frac_m, frac_h = mstats(birth / tot[:, None])
    say(f"    MC birth-region tag, source(z<100): {frac_m[0]:.4f} "
        f"+- {frac_h[0]:.4f}  (duct-annulus {frac_m[1]:.4f}, "
        f"end {frac_m[2]:.4f}, duct-column {frac_m[3]:.4f})")
    for mode in ("rates", "bounded_chord"):
        full, parts, _ = results[mode]
        den = full["exch_ac_mid"][ON].mean()
        row = "    " + f"{mode:>13} "
        for name in ("A_source", "B_duct", "C_end"):
            v = parts[name]["exch_ac_mid"][ON].mean()
            row += f" {name} {v / den:.4f}"
        say(row)
    say()

    say("(4) END-REGION LITTLE'S-LAW RESIDENCE, bins 2.0-3.0 ms")
    say(f"    MC  N_end {mc_inv_m:.4e} +-{mc_inv_h:.1e} atoms, "
        f"return {mc_ret_m:.4e} +-{mc_ret_h:.1e} atoms/s, "
        f"tau {1e3*mc_tau_m:.3f} +- {1e3*mc_tau_h:.3f} ms")
    for mode in ("rates", "bounded_chord"):
        full = results[mode][0]
        N = full["N_end"][ON].mean()
        r = full["phi_ret"][ON].mean()
        allx = r + full["pump_R"][ON].mean() + full["ion_end"][ON].mean()
        say(f"    {mode:>13}  N_end {N:.4e} (ratio {N / mc_inv_m:.3f}), "
            f"return {r:.4e} (ratio {r / mc_ret_m:.3f}), "
            f"tau {1e3*N/r:.3f} ms; all-exit tau {1e3*N/allx:.3f} ms")
    say()
    say("estimator note: the bounded-chord annulus has no advective face "
        "flux -- its half of the end-plane flux counts flight displacements "
        "that straddle z_b; the rate arm's is the marched face flux (the "
        "probe's own estimator).")

    Path(a.out).write_text("\n".join(L) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
