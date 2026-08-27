"""fts_ -- TAIL PRESSURE-SHARE SIZING READ (fueling-anchor / NBL, pre-fa4).

READ-ONLY reducer over EXISTING kinetic-era artifacts. Runs no solver, opens
no file for write except its own report. It answers one question: what is the
NEUTRAL PRESSURE PER ATOM the kinetic era measured, and how does it compare
with the cold-gas pressure per atom the NBL fluid build assumes
(``Tn_K = 300 K``, core/config.py:2205)?

Sources (all already on disk in cablp/scripts/):
  - e2_regen_true_sources/neutral_arch_e2_compare_nx240_truesrc.txt
        E2 matched-time DVM vs true-kinematics TPMC; carries column/annulus
        DENSITY, AXIAL MOMENTUM DENSITY and KINETIC ENERGY DENSITY per
        region, per 0.5 ms bin, per source phase. The second moment is what
        makes the pressure read possible.
        REPOINTED 2026-08-24 (Tom's ruling, log 24ab). This read was
        originally computed on the flat neutral_arch_e2_compare_nx240.txt,
        which ran SOURCE-STARVED: its menu totalled 4.507e21 atoms/s against
        the corrected 1.28923e22 (2.86x, e2_regen_true_sources/
        leg_truesrc.cmd execution record), because the two-zone puff and the
        anode-collection channel were still being read off the column rows
        alone. That product is INVALIDATED. The _truesrc file is the same E2
        comparison at the same decided config over the corrected menu, and
        it is the only E2 file this script reads.
  - neutral_arch_e0_summary.md          E0 bench alpha + T_par/T_perp table
        (NB: the copy on disk is the 2026-08-06 SMALLBATCH rerun, background
        sb_bg.h5, which no longer exists; the nx240 T table was overwritten).
  - neutral_arch_e0_rerun_check.txt     preserves the nx240 alpha medians.
  - neutral_arch_e1_vgrid_nx240.txt     local closed-tube T_par/T_perp fixed
        points + the grid-limited/converged verdicts.

The pressure identity used throughout, stated so the arithmetic is auditable:
for a population split into a cold bulk (fraction f_c at T_c) and a hot tail
(fraction f_h at T_h), the MEASURED scalar moment temperature obeys
    T_meas = f_c*T_c + f_h*T_h,
so the tail's pressure share against the cold bulk is
    S = f_h*T_h / (f_c*T_c) = (T_meas - f_c*T_c) / (f_c*T_c),
which needs only T_meas and f_c -- T_h and f_h are NOT separately required,
and are not separately available from any saved artifact.
"""
import re
import sys

E2 = 'e2_regen_true_sources/neutral_arch_e2_compare_nx240_truesrc.txt'
M_HE = 4.002602 * 1.66053906660e-24      # g
EV = 1.602176634e-12                     # erg/eV
KB = 1.380649e-16                        # erg/K
T300 = 300.0 * KB / EV                   # eV

ROW = re.compile(
    r'^(.{22})\s+([\d.]+-[\d.]+)\s+(tran|relx)\s+'
    r'(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$')


def parse(path):
    txt = open(path).read()
    cfgs = re.split(r'^={100}\n# (Configuration .*?)\n={100}\n', txt, flags=re.M)
    out = {}
    for i in range(1, len(cfgs), 2):
        secs = re.split(r'^### (.*?)$', cfgs[i + 1], flags=re.M)
        d = {}
        for j in range(1, len(secs), 2):
            rows = []
            for ln in secs[j + 1].splitlines():
                m = ROW.match(ln)
                if m:
                    rows.append(dict(region=m.group(1).strip(), t=m.group(2),
                                     phase=m.group(3), dvm=float(m.group(4)),
                                     ref=float(m.group(5)), sem=float(m.group(6))))
            if rows:
                d[secs[j].split('  ')[0].strip()] = rows
        out[cfgs[i]] = d
    return out


def temps(dens, mom, ener, key):
    """Mean energy per atom, bulk energy, thermal T [eV] for column 'key'."""
    n, p, e = dens[key], mom[key], ener[key]
    E_tot = e / n / EV
    u = p / (M_HE * n)
    E_bulk = 0.5 * M_HE * u * u / EV
    return E_tot, u, E_bulk, (2.0 / 3.0) * (E_tot - E_bulk)


def share(T, f_c):
    return (T - f_c * T300) / (f_c * T300)


def main():
    L = []
    W = L.append
    W('TAIL PRESSURE-SHARE SIZING READ  (fts_)')
    W('=' * 100)
    W('read-only reduction over existing artifacts; no solver invoked.')
    W(f'cold reference   : Tn_K = 300 K  ->  T_cold = {T300:.6f} eV'
      '   (core/config.py:2205, class MEASURED)')
    W(f'He atom mass     : {M_HE:.6e} g')
    W('')
    W('SECTION 1 -- E2 nx240: neutral pressure per atom, per region, per phase')
    W('-' * 100)
    W(f'source : {E2}, Configuration A')
    W('         (accommodation 1.0; reference = transient full-particle TPMC,')
    W('          TRUE two-body kinematics -- the file\'s own truth column).')
    W('CONDITIONALITY: this is a NEUTRAL-ONLY transient on the FROZEN plasma')
    W('  background es1_kn2z_promoted_nx240.h5 (plateau 5.0-19.5 ms). "tran" =')
    W('  neutral SOURCES ON, "relx" = SOURCES OFF. Neither phase is a plasma')
    W('  discharge/afterglow regime: the plasma is the same frozen plateau in')
    W('  both. Era: 2026-08-05, PRE the 2026-08-06 handshake conservation fix.')
    W('')
    cfg = parse(E2)
    A = cfg[[k for k in cfg if k.startswith('Configuration A:')][0]]
    dens = A['nn_col']
    mom = A['column axial momentum density']
    ener = A['column kinetic energy density']
    adens = A['nn_ann']
    amom = A['annulus axial momentum density']
    aener = A['annulus kinetic energy density']
    assert [(r['region'], r['t']) for r in dens] == [(r['region'], r['t']) for r in ener]

    for zone, D, P, E in (('COLUMN', dens, mom, ener),
                          ('ANNULUS', adens, amom, aener)):
        W(f'### {zone}')
        W(f"{'region':<22}{'t [ms]':>11}{'ph':>5}"
          f"{'<E>ref':>9}{'u_ref':>11}{'Ebulk':>8}{'T_ref':>9}{'T_dvm':>9}"
          f"{'T/T300':>9}")
        W(f"{'':<22}{'':>11}{'':>5}{'[eV]':>9}{'[cm/s]':>11}{'[eV]':>8}"
          f"{'[eV]':>9}{'[eV]':>9}{'':>9}")
        W('-' * 100)
        prev = None
        for k in range(len(D)):
            assert D[k]['region'] == E[k]['region'] == P[k]['region']
            if prev is not None and D[k]['region'] != prev:
                W('')
            prev = D[k]['region']
            Er, ur, Eb, Tr = temps(D[k], P[k], E[k], 'ref')
            _, _, _, Td = temps(D[k], P[k], E[k], 'dvm')
            W(f"{D[k]['region']:<22}{D[k]['t']:>11}{D[k]['phase']:>5}"
              f"{Er:>9.4f}{ur:>11.3e}{Eb:>8.4f}{Tr:>9.4f}{Td:>9.4f}"
              f"{Tr / T300:>9.1f}")
        W('')

    W('SECTION 2 -- pressure share of the hot tail against the cold bulk')
    W('-' * 100)
    W('S = (T_meas - f_c*T_cold)/(f_c*T_cold), evaluated on the TPMC reference')
    W('column temperature, at the last bin of each phase. f_c is the COLD')
    W('NUMBER fraction; the only measured constraint on it in the record is the')
    W('E0 reviewer statement "67-85 % of column mass sits within 3 thermal')
    W('spreads of 300 K" (CAMPAIGN_LOG_2026-08-05_to_2026-08-10_kinetic.md:291),')
    W('so f_c = 0.67 / 0.85 bracket it and f_c = 1.0 is the HARD LOWER BOUND on')
    W('S (no cold gas can be missing).')
    W('')
    W(f"{'zone/region':<30}{'phase':>6}{'T_meas':>9}"
      f"{'S(f=1.0)':>10}{'S(f=.85)':>10}{'S(f=.67)':>10}")
    W('-' * 100)
    for zone, D, P, E in (('col', dens, mom, ener), ('ann', adens, amom, aener)):
        seen = set()
        for ph in ('tran', 'relx'):
            for k in reversed(range(len(D))):
                if D[k]['phase'] != ph:
                    continue
                key = (D[k]['region'], ph)
                if D[k]['region'] in seen and key[1] == ph and \
                        any(s == key for s in seen):
                    continue
                if key in seen:
                    continue
                seen.add(key)
                _, _, _, Tr = temps(D[k], P[k], E[k], 'ref')
                W(f"{zone + ' ' + D[k]['region']:<30}{ph:>6}{Tr:>9.4f}"
                  f"{share(Tr, 1.0):>10.1f}{share(Tr, 0.85):>10.1f}"
                  f"{share(Tr, 0.67):>10.1f}")
    W('')

    W('SECTION 3 -- E0 / E1 corroboration (quoted, not recomputed)')
    W('-' * 100)
    W('E0 bench, deviational mass fraction alpha = sum|f - M[n,u,T]|/sum f, on')
    W('the nx240 background (neutral_arch_e0_rerun_check.txt, bit-identical')
    W('across three runs, no RNG on that path):')
    W('    column  1.7043 mid-machine / 1.7146 max / 1.2810 min')
    W('    annulus 0.9184 mid-machine / 1.1619 max / 0.3348 min')
    W('alpha is bounded by 2 (disjoint f and M). alpha ~ 1.7 means the true f')
    W('and its own best isotropic Maxwellian barely overlap.')
    W('')
    W('E0 moment temperatures. The nx240 T_par/T_perp TABLE IS LOST: the file')
    W('neutral_arch_e0_summary.md on disk was overwritten on 2026-08-06 by the')
    W('smallbatch smoke rerun against background sb_bg.h5, which is itself no')
    W('longer present. The surviving (smallbatch) table, column zone:')
    W(f"{'z [cm]':>8}{'T_par':>9}{'T_perp':>9}{'T_scalar':>10}{'T/T300':>9}"
      f"{'T_par/T_perp':>14}")
    e0 = [(5, .5249, .2447), (224, .6692, .5580), (419, 1.2718, .8504),
          (614, 1.7295, 1.0410), (809, 1.9732, 1.1253), (1004, 2.0416, 1.1387),
          (1199, 1.9757, 1.1049), (1394, 1.8202, 1.0339), (1589, 1.5676, .9058),
          (1784, 1.1718, .6653)]
    for z, tp, tq in e0:
        ts = (tp + 2 * tq) / 3.0
        W(f'{z:>8}{tp:>9.4f}{tq:>9.4f}{ts:>10.4f}{ts / T300:>9.1f}'
          f'{tp / tq:>14.2f}')
    W('')
    W('E1 local closed-tube relaxed fixed points, nx240 background')
    W('(neutral_arch_e1_vgrid_nx240.txt, 128x48 anchor column):')
    W(f"{'cell':<38}{'T_par':>9}{'T_perp':>9}{'T_scalar':>10}{'T/T300':>9}")
    e1 = [('cold wall-fed  z=111.2 Ti=1.72 eV', 1.604164, 1.145557),
          ('mid-column CX  z=1026.2 Ti=9.65 eV', 6.092358, 6.020288),
          ('source-adjacent z=5.0 Ti=4.59 eV', 7.559342, 2.063629)]
    for nm, tp, tq in e1:
        ts = (tp + 2 * tq) / 3.0
        W(f'{nm:<38}{tp:>9.4f}{tq:>9.4f}{ts:>10.4f}{ts / T300:>9.1f}')
    W('  NB these are LOCAL CLOSED TUBES (12 cells, no wall-gas resupply,')
    W('  ionization balanced tick-by-tick by an equal-rate recombination')
    W('  rebirth). They are an upper envelope on the CX relay, not the open')
    W('  column. T_par and T_perp are GRID-LIMITED at the production 48x12')
    W('  grid (worst 1.165e-2 / 1.843e-2 vs the 128x48 anchor).')
    W('')
    out = 'fts_read.txt'
    open(out, 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))
    print(f'\n[written: {out}]', file=sys.stderr)


main()


# ---------------------------------------------------------------------------
# Appended sections 4-6: mechanism decomposition and era transfer.
# ---------------------------------------------------------------------------
import math
import h5py
import numpy as np

RP_CM = 15.0          # geometry/Rp_cm, uniform, es1_kn2z_promoted_nx240.h5
REGIONS = {'near-anode  z<100': (0, 100), 'mid-machine 500-1000': (500, 1000),
           'far column 1000-1800': (1000, 1800), 'end-expansion z>1800': (1800, 2000)}
# E2 CX-channel table, true-kinematics MC, neutral Maxwellian at 300 K
# (neutral_arch_e2_cx_channel.txt, "Channel rate coefficients", k_cx MC column)
KCX_TI = [1.59, 3.60, 5.51, 7.23, 8.38, 9.09, 9.50, 9.63]
KCX_MC = [2.4365e-9, 3.1605e-9, 3.7088e-9, 4.1338e-9, 4.3944e-9,
          4.5488e-9, 4.6297e-9, 4.657e-9]


def k_cx(Ti):
    return float(np.interp(Ti, KCX_TI, KCX_MC))


def append():
    L = []
    W = L.append
    cfg = parse(E2)
    A = cfg[[k for k in cfg if k.startswith('Configuration A:')][0]]
    D, X = A['nn_col'], A['radial exchange, column -> annulus']
    E, P = A['column kinetic energy density'], A['column axial momentum density']

    W('')
    W('SECTION 4 -- what sets the column moment temperature (measured rates)')
    W('-' * 100)
    W('The column loss channel is the measured radial exchange row. tau_exch =')
    W('N_col / exch_ca with N_col = <nn_col> * pi*Rp^2*L_region, Rp = 15 cm')
    W('(geometry/Rp_cm, uniform). Last bin of each phase.')
    W(f"{'region':<24}{'ph':>5}{'nn_col':>11}{'exch_ca':>11}"
      f"{'nu_exch':>10}{'tau[us]':>9}")
    W('-' * 100)
    for ph in ('tran', 'relx'):
        for r, (lo, hi) in REGIONS.items():
            k = [i for i in range(len(D))
                 if D[i]['region'] == r and D[i]['phase'] == ph][-1]
            V = math.pi * RP_CM ** 2 * (hi - lo)
            N = D[k]['ref'] * V
            nu = X[k]['ref'] / N
            W(f"{r:<24}{ph:>5}{D[k]['ref']:>11.3e}{X[k]['ref']:>11.3e}"
              f"{nu:>10.3e}{1e6 / nu:>9.1f}")
    W('')
    W('ARITHMETIC (labelled as such, not a measurement). A CX-born atom is')
    W('born at the local ion energy and crosses the column radius ballistically,')
    W('so it lives tau_hot = Rp/v_hot, v_hot = sqrt(2*T_i/m). Its steady number')
    W('fraction follows the saturating two-state balance')
    W('    f_hot = nu_cx*tau_hot / (1 + nu_cx*tau_hot),  nu_cx = n_i*k_cx(T_i),')
    W('which cannot exceed 1 when the CX rate outruns the escape rate.')
    W('The implied moment temperature is T_pred = f_hot*T_i + (1-f_hot)*T_cold.')
    W('Comparing T_pred against the TPMC-measured T tests the mechanism.')
    with h5py.File('es1_kn2z_promoted_nx240.h5', 'r') as f:
        t = f['time'][:]
        z = f['geometry/z_cm'][:]
        kk = int(np.argmin(abs(t - 0.0142)))
        ni_row = f['n'][kk]
        Ti_row = f['Ti'][kk]
    W(f"{'region':<24}{'n_i':>11}{'T_i':>8}{'k_cx':>11}{'nu_cx':>10}"
      f"{'tau_h[us]':>10}{'f_hot':>8}{'T_pred':>8}{'T_meas':>8}")
    W('-' * 100)
    for r, (lo, hi) in REGIONS.items():
        m = (z >= lo) & (z < hi)
        ni = float(np.mean(ni_row[m]))
        Ti = float(np.mean(Ti_row[m]))
        kc = k_cx(Ti)
        nu = ni * kc
        v = math.sqrt(2.0 * Ti * EV / M_HE)
        th = RP_CM / v
        fh = nu * th / (1.0 + nu * th)   # saturating two-state balance
        Tp = fh * Ti + (1 - fh) * T300
        k = [i for i in range(len(D))
             if D[i]['region'] == r and D[i]['phase'] == 'tran'][-1]
        _, _, _, Tm = temps(D[k], P[k], E[k], 'ref')
        W(f'{r:<24}{ni:>11.3e}{Ti:>8.2f}{kc:>11.3e}{nu:>10.3e}'
          f'{th * 1e6:>10.2f}{fh:>8.3f}{Tp:>8.3f}{Tm:>8.3f}')
    W('')
    W('SECTION 5 -- transferability to the CURRENT (fueling-anchor) stance')
    W('-' * 100)
    W('Mid-machine 500-1000, t = 14.2 ms, plasma state of each run as saved.')
    W('The E0/E1/E2 background is the 2026-07-27 KN2Zone quasi-static arm; the')
    W('fa* arms are the live NBL-stage fueling-anchor arms.')
    W(f"{'run':<24}{'n_i':>11}{'T_i':>8}{'nn_col':>11}{'n_i/nn':>9}"
      f"{'nu_cx':>10}{'f_hot':>8}{'T_pred':>8}{'S(f_c=1)':>10}")
    W('-' * 100)
    for p, lab in (('es1_kn2z_promoted_nx240.h5', 'KN2Z-BG (E0/E1/E2)'),
                   ('fa1_arm.h5', 'fa1_arm'), ('fa2_arm.h5', 'fa2_arm'),
                   ('fa3_arm.h5', 'fa3_arm'), ('faj_arm.h5', 'faj_arm')):
        try:
            with h5py.File(p, 'r') as f:
                t = f['time'][:]
                z = f['geometry/z_cm'][:]
                kk = int(np.argmin(abs(t - 0.0142)))
                m = (z >= 500) & (z < 1000)
                ni = float(np.mean(f['n'][kk][m]))
                Ti = float(np.mean(f['Ti'][kk][m]))
                nn = float(np.mean(f['nn'][kk][m]))
        except Exception as exc:
            W(f'{lab:<24}  unreadable: {exc}')
            continue
        kc = k_cx(Ti)
        nu = ni * kc
        v = math.sqrt(2.0 * Ti * EV / M_HE)
        x = nu * RP_CM / v; fh = x / (1.0 + x)   # saturating
        Tp = fh * Ti + (1 - fh) * T300
        W(f'{lab:<24}{ni:>11.3e}{Ti:>8.2f}{nn:>11.3e}{ni / nn:>9.2f}'
          f'{nu:>10.3e}{fh:>8.3f}{Tp:>8.3f}{share(Tp, 1.0):>10.1f}')
    W('  k_cx below 1.59 eV is an EXTRAPOLATION of the E2 MC table (fa1 only).')
    W('  Neutral-neutral mfp check, sigma_nn ~ 1e-15 cm^2 for He:')
    for lab, nn in (('KN2Z-BG', 1.167e10), ('fa2/fa3', 3.86e11),
                    ('faj', 1.085e12), ('fa1', 1.099e13)):
        W(f'    {lab:<10} nn = {nn:.3e} cm^-3  ->  mfp = {1.0 / (nn * 1e-15):.3e} cm'
          f'   (column radius 15 cm)')
    W('')
    with open('fts_read.txt', 'a') as fh:
        fh.write('\n'.join(L) + '\n')
    print('\n'.join(L))


append()
