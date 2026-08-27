#!/usr/bin/env python
"""Read-only estimate of the ionization a QL fast-tail channel would supply.

Offline quadrature/walk estimate over SAVED backgrounds; it runs no solver and
mutates no state. ``walk_estimate`` is the reference arithmetic the in-solver
tail-ionization cross-check compares against, so this module is imported as a
library as well as run as a script.

Instrument: the repo's own CSDA march
(``cathode.beam_deposition.deposit_beam``). Tail electrons at ``E_tail`` are
launched 50/50 +/-z from the cells carrying ``beam_heat_anomalous_W``, with
flux ``P_QL / E_tail``, and marched with the ``fast_electron`` Coulomb model,
the He EII ionization cross section (``He_EII_cross_lkup``, Janev) and the
summed singlet excitation manifold (``He_beam_excitation_channel_lkup``) --
the same arithmetic the production beam uses.

Two secondary-electron treatments:

``A``  Secondaries carry the module's ``<W_sec>`` OPB mean. For
       ``E_tail <= 150`` eV that mean lies below the 20.6158 eV inelastic
       floor, so they bank as local bulk heat and the cascade truncates at
       depth 1.
``B``  The OPB spectrum is resolved per birth cell into sub-threshold,
       excitation-only and ionizing groups; the above-threshold groups walk
       recursively.

Boundaries: -z walks truncate at the cathode disc (``WALL``); a walker
reaching either end above ``min(4*Te(end cell), 30 eV)`` is lost, and below it
banks as end-cell heat. Free-escape -- no anode-mesh interception.

Backgrounds and report/profile destinations are resolved relative to this
file's directory and can be overridden on the command line; run with
``--help`` for the current defaults.
"""
import os, sys, math, json, argparse
from pathlib import Path
os.environ.setdefault('CABLP_COMPILED_KERNELS', '1')
# Resolve the package from THIS checkout, not a hardcoded one, so the module
# (and the gate that imports it) works from a clean clone. Appended rather
# than inserted: an explicit PYTHONPATH still wins, which is what keeps a
# worktree gate testing the worktree's own build.
_REPO_CABLP = str(Path(__file__).resolve().parent.parent)
if _REPO_CABLP not in sys.path:
    sys.path.append(_REPO_CABLP)
import numpy as np
import h5py
from cablp.cathode import beam_deposition as bd
from cablp.cathode.kernels import PROVENANCE

SCRIPTS_DIR = Path(__file__).resolve().parent

ERG_PER_EV = 1.602176634e-12
J_PER_EV = 1.602176634e-19
E_STOP = bd.HE_E_STOP_EV      # 20.6158 eV, lowest inelastic threshold (2^1S)
I_ION = bd.HE_I_ION_EV        # 24.58738793623 eV
EBAR = bd.HE_OPB_EBAR_EV      # 15.8 eV OPB shape parameter
RUNGS = (30.0, 75.0, 150.0)
WALL = 2                      # cathode cell index: -z walks truncate here
REFLECT_TE_MULT = 4.0         # end reflection threshold = 4*Te(end cell)
TRUNC_FRAC = 1e-9             # walks below this fraction of P_QL bank locally
MID_LO, MID_HI = 500.0, 1000.0

# Default backgrounds, resolved beside this file. They are gitignored campaign
# artifacts, so a clean clone has the CODE but must be pointed at runs: use
# --kin/--fld. The kinetic default is shot1, NOT shot0: the shot0 background
# this estimate was first run against is superseded, and every later read used
# shot1, so defaulting to shot0 silently produced numbers that were not
# comparable with the ones being quoted.
DEFAULT_RUNS = {
    'KIN': SCRIPTS_DIR / 'es1_k5a_shot1_nx240.h5',
    'FLD': SCRIPTS_DIR / 'es1_prod_circuit_nx240.h5',
}
RUNS = {tag: str(path) for tag, path in DEFAULT_RUNS.items()}
TRELS = {
    'KIN': [-0.5, -0.3, -0.1, 0.5, 1.0, 2.0, 3.4, 8.0, 13.5, 18.0],
    'FLD': [-0.3, -0.2, -0.1, 0.5, 1.0, 2.0, 3.4, 18.0],
}
# negative t_rel = EXTRA (beyond registered windows): breakdown-phase foot probes

def opb_cdf(w):
    return math.atan(w / EBAR)

def opb_cond_mean(a, b):
    den = opb_cdf(b) - opb_cdf(a)
    if den <= 0.0:
        return 0.5 * (a + b)
    return EBAR * (math.log1p((b/EBAR)**2) - math.log1p((a/EBAR)**2)) / (2.0*den)

def walk_estimate(E_tail, P_QL_W, nn, ne, Te, dz, variant):
    """One snapshot, one rung. Returns dict with S_tail (ions/s per full-grid
    cell), buckets [W], counters."""
    ncell = nn.size
    P_eVs = P_QL_W / J_PER_EV          # eV/s per cell
    P_tot_eVs = float(P_eVs.sum())
    S_tail = np.zeros(ncell)
    heat_cells = np.zeros(ncell)       # local bulk-heat deposition, eV/s
    buckets = dict(ion_invest=0.0, radiation=0.0, coulomb=0.0, sec_local=0.0,
                   terminal=0.0, end_reflect=0.0, end_loss_cath=0.0,
                   end_loss_far=0.0, trunc_local=0.0)
    n_ion_gen = [0.0, 0.0]             # gen0, cascade
    launched = 0.0                     # tail electrons /s (gen0)

    # sliced domain: full grid from WALL onward; -z exit = cathode disc
    sl = slice(WALL, ncell)
    nn_s, ne_s, Te_s, dz_s = nn[sl], ne[sl], Te[sl], dz[sl]
    off = WALL

    stack = []
    for c in np.flatnonzero(P_eVs > 0.0):
        assert c >= WALL, f"QL power in cell {c} behind the cathode wall"
        flux = P_eVs[c] / E_tail
        launched += flux
        for d in (+1, -1):
            stack.append((c - off, d, E_tail, 0.5 * flux, 0))

    while stack:
        c, d, E0, flux, gen = stack.pop()
        pw = E0 * flux
        if pw <= 0.0:
            continue
        if E0 <= E_STOP or pw < TRUNC_FRAC * P_tot_eVs:
            # sub-threshold (Coulomb-couples locally) or negligible: bank local
            key = 'sec_local' if E0 <= E_STOP else 'trunc_local'
            buckets[key] += pw
            heat_cells[c + off] += pw
            continue
        res = bd.deposit_beam(
            E0, flux, nn_s, ne_s, Te_s, c, d, dz_cm=dz_s,
            coulomb_model='fast_electron', anomalous_model='none',
        )
        ev = res.ionization_events            # ions/s per sliced cell
        S_tail[sl] += ev
        n_ion_gen[1 if gen else 0] += float(ev.sum())
        buckets['ion_invest'] += float(res.ionization_cost_erg_s.sum()) / ERG_PER_EV
        buckets['radiation'] += float(res.radiated_erg_s.sum()) / ERG_PER_EV
        buckets['coulomb'] += float(res.heating_coulomb_erg_s.sum()) / ERG_PER_EV
        buckets['terminal'] += float(res.heating_terminal_erg_s.sum()) / ERG_PER_EV
        heat_cells[sl] += (res.heating_coulomb_erg_s
                           + res.heating_terminal_erg_s) / ERG_PER_EV
        # transmitted primary -> end rule
        if res.transmitted_flux > 0.0:
            Et, ft = res.transmitted_energy_eV, res.transmitted_flux
            endc = (ncell - 1) if d > 0 else WALL
            if Et > min(REFLECT_TE_MULT * Te[endc], 30.0):
                buckets['end_loss_far' if d > 0 else 'end_loss_cath'] += ft * Et
            else:
                buckets['end_reflect'] += ft * Et
                heat_cells[endc] += ft * Et
        # secondaries
        sec_eVs = res.heating_secondary_erg_s / ERG_PER_EV   # eV/s per cell
        if variant == 'A':
            tot = float(sec_eVs.sum())
            buckets['sec_local'] += tot
            heat_cells[sl] += sec_eVs
            continue
        # variant B: OPB-resolved per birth cell
        for j in np.flatnonzero(ev > 0.0):
            evj = float(ev[j]); sej = float(sec_eVs[j])
            E_rep = float(res.E_entry_eV[j])
            wmax = 0.5 * (E_rep - I_ION)
            if wmax <= E_STOP:
                buckets['sec_local'] += sej
                heat_cells[j + off] += sej
                continue
            # partition [0,wmax] at E_STOP and I_ION
            edges = [0.0, E_STOP] + ([I_ION, wmax] if wmax > I_ION else [wmax])
            norm = opb_cdf(wmax)
            ws, es = [], []
            for a, b in zip(edges[:-1], edges[1:]):
                ws.append((opb_cdf(b) - opb_cdf(a)) / norm)
                es.append(opb_cond_mean(a, b))
            mean_part = sum(w*e for w, e in zip(ws, es))
            scale = (sej / evj) / mean_part if mean_part > 0 else 1.0
            es = [min(e * scale, wmax) for e in es]
            # group 0: sub-threshold -> local
            buckets['sec_local'] += evj * ws[0] * es[0]
            heat_cells[j + off] += evj * ws[0] * es[0]
            for w, e in zip(ws[1:], es[1:]):
                if w <= 0.0 or e <= 0.0:
                    continue
                if e <= E_STOP:
                    buckets['sec_local'] += evj * w * e
                    heat_cells[j + off] += evj * w * e
                    continue
                for dd in (+1, -1):
                    stack.append((j, dd, e, 0.5 * evj * w, gen + 1))

    tot = sum(buckets.values())
    return dict(S_tail=S_tail, heat_cells=heat_cells * J_PER_EV,
                buckets={k: v * J_PER_EV for k, v in buckets.items()},
                P_QL_W=P_tot_eVs * J_PER_EV,
                closure=(tot - P_tot_eVs) / P_tot_eVs if P_tot_eVs else 0.0,
                launched_per_s=launched, n_ion_gen0=n_ion_gen[0],
                n_ion_casc=n_ion_gen[1])

def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--kin', default=str(DEFAULT_RUNS['KIN']),
                    help='kinetic-arm background h5 (default: %(default)s)')
    ap.add_argument('--fld', default=str(DEFAULT_RUNS['FLD']),
                    help='fluid-arm background h5 (default: %(default)s)')
    ap.add_argument('--report', default=str(SCRIPTS_DIR / 'tailion_report.txt'),
                    help='text report destination (default: %(default)s)')
    ap.add_argument('--profiles',
                    default=str(SCRIPTS_DIR / 'tailion_profiles.npz'),
                    help='npz profile destination (default: %(default)s)')
    return ap.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    runs = {'KIN': args.kin, 'FLD': args.fld}
    missing = [f'{tag}={path}' for tag, path in runs.items()
               if not os.path.exists(path)]
    if missing:
        raise SystemExit(
            'background run(s) not found: ' + ', '.join(missing) +
            '\nThese are gitignored campaign artifacts and are not part of a '
            'clean clone. Point --kin/--fld at saved runs.'
        )
    out_lines = []
    def P(*a):
        s = ' '.join(str(x) for x in a)
        print(s); out_lines.append(s)

    P(f"# tailion_estimate — TAIL-IONIZATION BRIDGE discriminator")
    P(f"# kernels: {PROVENANCE}")
    P(f"# conventions: coulomb=fast_electron (dE/dx=2 pi e^4 ne lnL/E, "
      f"lnL=_c_log_ei(max(Te,0.1),ne), NRL 2019 2-3/2-4)")
    P(f"# sigma_ion=He_EII_cross_lkup (Janev, threshold {I_ION} eV); "
      f"sigma_exc=He_beam_excitation_channel_lkup (Ralchenko singlet manifold + n>=5 tail)")
    P(f"# <W_sec>=OPB mean, Ebar={EBAR} eV; E_stop={E_STOP} eV; "
      f"end reflection threshold min({REFLECT_TE_MULT}*Te(end cell), 30 eV); "
      f"-z walks truncate at cathode cell {WALL}; free-escape (no anode-mesh interception; "
      f"mesh variant = +z arm x (1-eta)=0.642, exact by flux linearity)")
    P("")
    profiles = {}
    for tag, fn in runs.items():
        f = h5py.File(fn, 'r')
        t = f['time'][:]
        tbd = float(f.attrs['t_breakdown_trigger'])
        z = f['geometry/z_cm'][:]
        dz = f['geometry/length_cm'][:]
        vol = f['geometry/plasma_volume_cm3'][:]
        role = np.array([r.decode() for r in f['geometry/cell_role'][:]])
        colmask = np.isin(role, ('column', 'puff'))
        midmask = (z >= MID_LO) & (z <= MID_HI)
        n_all = f['n'][:]                     # (frames, cells) — 5.6 MB, fine
        # front position per frame: first column cell (ascending z) with
        # n < 0.1 * max(n over column cells); front z = z of preceding cell
        zi = np.argsort(z[colmask]); zcol = z[colmask][zi]
        zf = np.full(t.size, zcol[0])
        ncol = n_all[:, colmask][:, zi]
        for k in range(t.size):
            nm = np.nanmax(ncol[k])
            if nm <= 0:
                continue
            below = np.flatnonzero(ncol[k] < 0.1 * nm)
            zf[k] = zcol[below[0] - 1] if below.size and below[0] > 0 else (
                zcol[-1] if not below.size else zcol[0])
        P(f"== RUN {tag}  {os.path.basename(fn)}  t_breakdown={tbd*1e3:.4f} ms")
        for trel in TRELS[tag]:
            fr = int(np.argmin(np.abs(t - (tbd + trel * 1e-3))))
            tt = t[fr]
            nn = np.nan_to_num(f['nn'][fr]); ne = np.nan_to_num(f['n'][fr])
            Te = np.nan_to_num(f['Te'][fr])
            P_QL = np.nan_to_num(f['cathode_diagnostics/beam_heat_anomalous_W'][fr])
            act_n = (np.nan_to_num(f['rhs_terms/ionization_birth/n'][fr])
                     + np.nan_to_num(f['rhs_terms/beam_ionization_birth/n'][fr]))
            actual = act_n * np.nan_to_num(vol)      # ions/s per cell
            act_mid = float(actual[midmask].sum())
            act_tot = float(actual.sum())
            act_prefront = float(actual[(z > zf[fr])].sum())
            nn_mid = float(np.mean(nn[midmask]))
            P(f"-- t_rel=+{trel} ms (frame {fr}, t={tt*1e3:.3f} ms, phase="
              f"{f['phase'][fr].decode()})  front z_f={zf[fr]:.0f} cm")
            P(f"   P_QL={P_QL.sum()/1e3:.2f} kW in cells {np.flatnonzero(P_QL>0).tolist()}"
              f"  nn_mid={nn_mid:.3e} cm^-3  Te_ends=({Te[WALL]:.2f},{Te[-1]:.2f}) eV")
            P(f"   ACTUAL ionization: total={act_tot:.3e}/s  mid-band={act_mid:.3e}/s"
              f"  pre-front={act_prefront:.3e}/s")

            # mechanism addendum: single +z marker ray per rung from the peak
            # QL cell; z at E crossings and S_tail band split
            bcell = int(np.argmax(P_QL))
            for E_tail in RUNGS:
                mres = bd.deposit_beam(E_tail, 1.0, nn[WALL:], ne[WALL:],
                                       Te[WALL:], bcell - WALL, 1,
                                       dz_cm=dz[WALL:],
                                       coulomb_model='fast_electron',
                                       anomalous_model='none')
                Ez = mres.E_entry_eV; zz = z[WALL:]
                live = np.flatnonzero(Ez > 0)
                marks = {}
                for thr in (50.0, 30.0, I_ION):
                    idx = live[Ez[live] > thr]
                    marks[thr] = zz[idx[-1]] if idx.size else float('nan')
                zstop = zz[live[-1]] if live.size else float('nan')
                stat = ('exits far end E=%.1f eV' % mres.transmitted_energy_eV
                        if mres.transmitted_flux > 0 else 'stops at z=%.0f cm' % zstop)
                P(f"   marker +z ray E={int(E_tail)}: z(E>50)={marks[50.0]:.0f} "
                  f"z(E>30)={marks[30.0]:.0f} z(E>I_ion)={marks[I_ION]:.0f} cm; {stat}")
            for zprobe in (250.0, 750.0, 1500.0):
                jp = int(np.argmin(np.abs(z - zprobe)))
                P(f"   state at z={zprobe:.0f}: ne={ne[jp]:.3e} nn={nn[jp]:.3e} Te={Te[jp]:.2f}")
            for E_tail in RUNGS:
                for variant in ('A', 'B'):
                    r = walk_estimate(E_tail, P_QL, nn, ne, Te, dz, variant)
                    S = r['S_tail']
                    S_mid = float(S[midmask].sum()); S_tot = float(S.sum())
                    S_pre = float(S[(z > zf[fr])].sum())
                    b = r['buckets']; PQ = r['P_QL_W']
                    fr_ = {k: (v / PQ if PQ else 0.0) for k, v in b.items()}
                    yield_per_e = S_tot / r['launched_per_s'] if r['launched_per_s'] else 0
                    eV_per_ion = (PQ / J_PER_EV) / S_tot if S_tot else float('inf')
                    ratio_mid = S_mid / act_mid if act_mid else float('inf')
                    key = f"{tag}_t{trel}_E{int(E_tail)}_{variant}"
                    profiles[key] = S
                    P(f"   E={int(E_tail):>3} eV [{variant}] S_tail: tot={S_tot:.3e}/s"
                      f" mid={S_mid:.3e}/s  RATIO_mid={ratio_mid:.3f}"
                      f"  prefront={S_pre:.3e}/s ({(S_pre/S_tot if S_tot else 0):.2f} of S_tail)"
                      f"  yield/e-={yield_per_e:.3f} (W-value exp {E_tail/46.0:.2f})"
                      f"  eV/ion={eV_per_ion:.0f}")
                    b0 = float(S[(z >= 0) & (z < MID_LO)].sum())
                    b2 = float(S[z > MID_HI].sum())
                    P(f"        bands ions/s: z<500={b0:.2e} mid={S_mid:.2e} z>1000={b2:.2e}")
                    P(f"        branching: ion={fr_['ion_invest']:.3f} rad={fr_['radiation']:.3f}"
                      f" coul={fr_['coulomb']:.3f} secH={fr_['sec_local']:.3f}"
                      f" term={fr_['terminal']:.3f} endC={fr_['end_loss_cath']:.3f}"
                      f" endF={fr_['end_loss_far']:.3f} refl={fr_['end_reflect']:.3f}"
                      f" trunc={fr_['trunc_local']:.4f}  closure={r['closure']:.2e}"
                      f"  casc_ions={r['n_ion_casc']:.2e}/s")
            profiles[f"{tag}_t{trel}_actual"] = actual
            profiles[f"{tag}_t{trel}_nn"] = nn
        profiles[f"{tag}_zf_cm"] = zf
        profiles[f"{tag}_time_s"] = t
        f.close()
    profiles['z_cm'] = z
    np.savez_compressed(args.profiles, **profiles)
    with open(args.report, 'w') as fh:
        fh.write('\n'.join(out_lines) + '\n')
    P(f"\n# wrote {args.report} and {args.profiles}")

if __name__ == '__main__':
    main()
