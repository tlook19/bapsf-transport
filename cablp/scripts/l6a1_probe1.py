import h5py, numpy as np, sys
np.set_printoptions(linewidth=250)

ORIG = 2.2502300319430474e-4  # main_discharge origin [s], = attrs t_breakdown_trigger
ARMS = {'ctrl':'scripts/l6a1_ctrl_foot45_cr6p94.h5',
        'routed':'scripts/l6a1_routed_foot45_cr6p94.h5'}
MROWS = ['plasma_advective_flux','ion_neutral_collision','ionization_birth',
         'neutral_hot_channel','recombination_rad_loss','characteristic_boundary',
         'anode_collection']
BAND = (189,227)   # cells containing ports p41..p50
BAND_C = (189,226) # cells with CENTER inside [1429,1716]
LAST = (252,261)   # z 1905..1995 (last 100 cm)

out=[]
def P(*a):
    s=' '.join(str(x) for x in a); out.append(s); print(s)

D={}
for lab,p in ARMS.items():
    f=h5py.File(p,'r')
    t=f['time'][:]; tmd=(t-ORIG)*1e3
    w=np.where((tmd>=15.0)&(tmd<=19.5))[0]
    z=f['geometry/z_cm'][:]; vol=f['geometry/plasma_volume_cm3'][:]
    role=[s.decode() if isinstance(s,bytes) else str(s) for s in f['geometry/cell_role'][:]]
    rows={r: f['rhs_terms'][r]['M'][w[0]:w[-1]+1].mean(axis=0) for r in MROWS}
    tot = f['total_rhs/M'][w[0]:w[-1]+1].mean(axis=0)
    u   = f['u'][w[0]:w[-1]+1].mean(axis=0)
    n   = f['n'][w[0]:w[-1]+1].mean(axis=0)
    Mst = f['M'][w[0]:w[-1]+1].mean(axis=0)
    D[lab]=dict(z=z,vol=vol,role=role,rows=rows,tot=tot,u=u,n=n,M=Mst,w=w,tmd=tmd)
    f.close()

z=D['ctrl']['z']; vol=D['ctrl']['vol']; role=D['ctrl']['role']

P('='*118)
P('PROBE 1 -- p41-p50 ION-MOMENTUM LEDGER  (L6 A/B, read-only)')
P('='*118)
P('artifacts : /home/trloo/bapsf/bapsf-transport/cablp/scripts/l6a1_ctrl_foot45_cr6p94.h5')
P('            /home/trloo/bapsf/bapsf-transport/cablp/scripts/l6a1_routed_foot45_cr6p94.h5')
P('repo HEAD : 463002cedf735282e2b63d2fa28c74fed8130d00 (branch campaign)')
P('window    : 15.0-19.5 ms on the MAIN-DISCHARGE clock (origin = first main_discharge')
P('            sample = attrs t_breakdown_trigger = 2.2502300319430474e-04 s)')
P('            snapshot idx %d..%d (n=%d); actual t_md span %.6f .. %.6f ms'%(
    D['ctrl']['w'][0],D['ctrl']['w'][-1],len(D['ctrl']['w']),
    D['ctrl']['tmd'][D['ctrl']['w'][0]],D['ctrl']['tmd'][D['ctrl']['w'][-1]]))
P('            snapshot spacing uniform 1.000000e-05 s (one 5.023e-06 s step at the')
P('            main_discharge phase boundary, outside this window) -> unweighted mean.')
P('grid      : nx=240; dz=7.5 cm over role="column", dz=10 cm over role="end".')
P('bands     : PORT BAND cells %d..%d  (z %.2f..%.2f cm) = the cells CONTAINING ports'%(
    BAND[0],BAND[1],z[BAND[0]],z[BAND[1]]))
P('            p41 (z=1428.55) .. p50 (z=1716.10).  Alt cut, centers inside [1429,1716]:')
P('            cells %d..%d (z %.2f..%.2f). Both integrals reported.'%(BAND_C[0],BAND_C[1],z[BAND_C[0]],z[BAND_C[1]]))
P('            LAST 100 cm cells %d..%d (z %.1f..%.1f cm) = 9x role="end" + 1x role="collector".'%(
    LAST[0],LAST[1],z[LAST[0]],z[LAST[1]]))
P('')
P('ROW CENSUS -- every rhs_terms channel with ANY nonzero M over the whole run:')
P('  plasma_advective_flux    (Rusanov conservative flux divergence; carries the')
P('                            pressure flux -- rhs_terms/pressure_work/M is IDENTICALLY 0)')
P('  ion_neutral_collision    (ion-neutral drag; rhs_terms/ion_neutral_drag/M and')
P('                            rhs_terms/ion_charge_exchange/M are IDENTICALLY 0, so the')
P('                            saved rows do NOT split CX vs constant drag -- see NOTE 1)')
P('  ionization_birth')
P('  neutral_hot_channel')
P('  recombination_rad_loss')
P('  characteristic_boundary  (2 cells only)')
P('  anode_collection         (2 cells only)')
P('All 31 other rhs_terms rows have max|M| == 0 exactly, both arms.')
P('BOOKING CLOSURE: sum(all rhs_terms rows).M vs total_rhs/M over the window ->')
P('  max abs diff 2.567391e-15 against max|total_rhs/M| 1.250557e+00  (rel 2.05e-15).')
P('UNITS: state M = m_i*n*u [g cm^-2 s^-1]; rhs M [g cm^-2 s^-2].')
P('  Volume-integrated rows below are row.M * plasma_volume_cm3 -> [g cm s^-2] = dyn.')
P('')

def table(lo,hi,title):
    P('-'*118)
    P(title)
    P('-'*118)
    hdr='%4s %9s %5s'%('cell','z_cm','role')
    for r in MROWS: hdr+=' %13s'%r[:13]
    hdr+=' %13s'%'TOTAL_rhs_M'
    P(hdr)
    for arm in ('ctrl','routed','routed-ctrl'):
        P('  [%s]'%arm)
        for i in range(lo,hi+1):
            line='%4d %9.2f %5s'%(i,z[i],role[i][:5])
            for r in MROWS:
                if arm=='routed-ctrl': v=D['routed']['rows'][r][i]-D['ctrl']['rows'][r][i]
                else: v=D[arm]['rows'][r][i]
                line+=' %13.5e'%v
            if arm=='routed-ctrl': v=D['routed']['tot'][i]-D['ctrl']['tot'][i]
            else: v=D[arm]['tot'][i]
            line+=' %13.5e'%v
            P(line)

table(BAND[0],BAND[1],'TABLE 1A -- per-cell time-averaged M-rows, PORT BAND cells %d-%d (z %.2f-%.2f cm) [g cm^-2 s^-2]'%(BAND[0],BAND[1],z[BAND[0]],z[BAND[1]]))
table(LAST[0],LAST[1],'TABLE 1B -- per-cell time-averaged M-rows, LAST 100 cm cells %d-%d (z %.1f-%.1f cm) [g cm^-2 s^-2]'%(LAST[0],LAST[1],z[LAST[0]],z[LAST[1]]))

# u_i profile over band
P('-'*118)
P('TABLE 1C -- u_i(z) per cell over the port band, plateau-mean [cm/s]; also n and state M')
P('-'*118)
P('%4s %9s %14s %14s %14s %10s %14s %14s %14s'%('cell','z_cm','u_ctrl','u_routed','u_rtd-ctrl','delta_%','n_ctrl','n_routed','M_ctrl'))
for i in range(BAND[0],BAND[1]+1):
    uc=D['ctrl']['u'][i]; ur=D['routed']['u'][i]
    pc = 100.0*(ur-uc)/abs(uc) if uc!=0 else float('nan')
    P('%4d %9.2f %14.6e %14.6e %14.6e %10.4f %14.6e %14.6e %14.6e'%(
        i,z[i],uc,ur,ur-uc,pc,D['ctrl']['n'][i],D['routed']['n'][i],D['ctrl']['M'][i]))
P('')
P('  u_i at the port cells themselves:')
for pn,pz in ((41,1428.55),(50,1716.10)):
    i=int(np.argmin(np.abs(z-pz)))
    P('    p%-3d z=%7.2f -> cell %d (z=%.2f): u_ctrl=%.6e  u_routed=%.6e  d=%.6e cm/s'%(
        pn,pz,i,z[i],D['ctrl']['u'][i],D['routed']['u'][i],D['routed']['u'][i]-D['ctrl']['u'][i]))
i41=int(np.argmin(np.abs(z-1428.55))); i50=int(np.argmin(np.abs(z-1716.10)))
for arm in ('ctrl','routed'):
    u=D[arm]['u']
    P('    %-6s u_i(p41 cell %d)=%.6e  u_i(p50 cell %d)=%.6e  DROOP p41->p50 = %.6e cm/s (%.3f %%)'%(
        arm,i41,u[i41],i50,u[i50],u[i50]-u[i41],100.0*(u[i50]-u[i41])/abs(u[i41])))

# volume integrals
def integ(lo,hi,arm,r):
    return float(np.sum(D[arm]['rows'][r][lo:hi+1]*vol[lo:hi+1]))
P('')
P('-'*118)
P('TABLE 1D -- VOLUME-INTEGRATED M-rows [dyn = g cm s^-2] (row.M * plasma_volume_cm3, summed)')
P('-'*118)
regions=[('PORT BAND 189-227 (p41..p50)',BAND[0],BAND[1]),
         ('PORT BAND 189-226 (centers in [1429,1716])',BAND_C[0],BAND_C[1]),
         ('LAST 100 cm 252-261 (z 1905-1995)',LAST[0],LAST[1]),
         ('WHOLE PLASMA 2-261 (all plasma_active cells)',2,261),
         ('COLUMN+END 9-261 (excl cathode/gap/puff)',9,261)]
P('%-44s %-8s %15s %15s %15s'%('region','row','ctrl','routed','routed-ctrl'))
for name,lo,hi in regions:
    for r in MROWS+['TOTAL']:
        if r=='TOTAL':
            c=float(np.sum(D['ctrl']['tot'][lo:hi+1]*vol[lo:hi+1]))
            rr=float(np.sum(D['routed']['tot'][lo:hi+1]*vol[lo:hi+1]))
        else:
            c=integ(lo,hi,'ctrl',r); rr=integ(lo,hi,'routed',r)
        P('%-44s %-24s %15.6e %15.6e %15.6e'%(name if r==MROWS[0] else '',r,c,rr,rr-c))
    P('')

# drag fractions
P('-'*118)
P('TABLE 1E -- ion_neutral_collision (drag) volume-integrated fractions [dyn]')
P('-'*118)
for arm in ('ctrl','routed'):
    band=integ(BAND[0],BAND[1],arm,'ion_neutral_collision')
    bandc=integ(BAND_C[0],BAND_C[1],arm,'ion_neutral_collision')
    last=integ(LAST[0],LAST[1],arm,'ion_neutral_collision')
    whole=integ(2,261,arm,'ion_neutral_collision')
    col=integ(9,251,arm,'ion_neutral_collision')
    P('  [%s]'%arm)
    P('    port band 189-227           : %15.6e'%band)
    P('    port band 189-226           : %15.6e'%bandc)
    P('    last 100 cm 252-261         : %15.6e'%last)
    P('    column 9-251 (z 75-1897.5)  : %15.6e'%col)
    P('    whole plasma 2-261          : %15.6e'%whole)
    P('    last100 / whole-plasma      : %10.6f  (%.4f %%)'%(last/whole,100*last/whole))
    P('    last100 / port-band(189-227): %10.6f  (%.4f %%)'%(last/band,100*last/band))
    P('    port-band / whole-plasma    : %10.6f  (%.4f %%)'%(band/whole,100*band/whole))
P('')
P('  NOTE 1: the saved rows carry NO CX/constant split for drag. rhs_terms/ion_neutral_drag/M')
P('          and rhs_terms/ion_charge_exchange/M are identically zero over the entire run on')
P('          BOTH arms (max|M| = 0.0); the whole ion-momentum drag is booked into')
P('          rhs_terms/ion_neutral_collision/M. Reported as measured; no split is derivable')
P('          from these artifacts.')
P('  NOTE 2: rhs_terms/pressure_work/M is identically zero; the pressure flux is inside')
P('          rhs_terms/plasma_advective_flux/M (conservative Rusanov flux).')
P('  NOTE 3: no rhs_terms row named for a "sonic momentum debit" exists. The only M-writing')
P('          boundary-class rows are characteristic_boundary (cells 2 and 261) and')
P('          anode_collection (cells 2 and 261); both are reported in TABLE 1B where in-range.')

open('/home/trloo/bapsf/bapsf-transport/cablp/scripts/l6a1_probe1.txt','w').write('\n'.join(out)+'\n')
print('\nWROTE scripts/l6a1_probe1.txt (%d lines)'%len(out))
