import h5py, numpy as np
ORIG=2.2502300319430474e-4
ARMS={'ctrl':'scripts/l6a1_ctrl_foot45_cr6p94.h5','routed':'scripts/l6a1_routed_foot45_cr6p94.h5'}
CELLS=[259,260,261]
out=[]
def P(*a):
    s=' '.join(str(x) for x in a); out.append(s); print(s)

D={}
for lab,p in ARMS.items():
    f=h5py.File(p,'r')
    t=f['time'][:]; tmd=(t-ORIG)*1e3
    w=np.where((tmd>=20.0)&(tmd<=25.0))[0]
    z=f['geometry/z_cm'][:]
    role=[s.decode() if isinstance(s,bytes) else str(s) for s in f['geometry/cell_role'][:]]
    rows=sorted(f['rhs_terms'].keys())
    d={'z':z,'role':role,'w':w,'tmd':tmd,'rows':{}}
    for r in rows:
        d['rows'][r]={fld: f['rhs_terms'][r][fld][w[0]:w[-1]+1].mean(axis=0) for fld in ('nn','nn_a','n','En','Ei','M_n')}
    for k in ('nn','nn_a','nn_hot','n','Tn','f_hot','tau_hot','hot_births','hot_recx','hot_S_cx',
              'hot_ionized','hot_wall','hot_flux_z','hot_n_flight','hot_end_fraction',
              'hot_Ei_ionization','hot_Ei_recx','Te','Ti'):
        if k in f: d[k]=f[k][w[0]:w[-1]+1].mean(axis=0)
    d['vol_n']=f['geometry/neutral_volume_cm3'][:]
    d['vol_p']=f['geometry/plasma_volume_cm3'][:]
    d['rowlist']=rows
    D[lab]=d
    f.close()

z=D['ctrl']['z']; role=D['ctrl']['role']; rows=D['ctrl']['rowlist']
P('='*126)
P('PROBE 2 -- WHAT FEEDS neutral_hot_channel AT THE COLLECTOR CELL  (L6 A/B, read-only)')
P('='*126)
P('artifacts : /home/trloo/bapsf/bapsf-transport/cablp/scripts/l6a1_{ctrl,routed}_foot45_cr6p94.h5')
P('repo HEAD : 463002cedf735282e2b63d2fa28c74fed8130d00 (branch campaign)')
P('window    : 20.0-25.0 ms MAIN-DISCHARGE clock (origin 2.2502300319430474e-04 s)')
P('            snapshot idx %d..%d (n=%d), t_md %.6f..%.6f ms'%(
   D['ctrl']['w'][0],D['ctrl']['w'][-1],len(D['ctrl']['w']),
   D['ctrl']['tmd'][D['ctrl']['w'][0]],D['ctrl']['tmd'][D['ctrl']['w'][-1]]))
P('            NB phase_events: afterglow begins at t=0.020225023 s = 20.000 ms t_md, and')
P('            post_afterglow at 0.026225023 s = 26.000 ms t_md. This window is ENTIRELY')
P('            inside the afterglow phase.')
P('cells     : 259 (z=%.1f, role=%s), 260 (z=%.1f, role=%s), 261 (z=%.1f, role=%s)'%(
   z[259],role[259],z[260],role[260],z[261],role[261]))
P('columns   : "nn" = COLUMN zone neutral field; "nn_a" = ANNULUS zone neutral field.')
P('units     : rhs nn/nn_a [cm^-3 s^-1]; volume-integrated = row * neutral_volume_cm3 [s^-1].')
P('            neutral_volume_cm3[259..261] = %.6e %.6e %.6e'%tuple(D['ctrl']['vol_n'][259:262]))
P('')

# 1. the target row
P('-'*126)
P('TABLE 2A -- rhs_terms/neutral_hot_channel, time-averaged, cells 259-261')
P('-'*126)
P('%5s %8s %6s %14s %14s %14s %14s %14s'%('arm','cell','role','nn','nn_a','n','En','Ei'))
for arm in ('ctrl','routed'):
    for i in CELLS:
        r=D[arm]['rows']['neutral_hot_channel']
        P('%5s %8d %6s %14.6e %14.6e %14.6e %14.6e %14.6e'%(arm,i,role[i][:6],r['nn'][i],r['nn_a'][i],r['n'][i],r['En'][i],r['Ei'][i]))
for i in CELLS:
    rc=D['ctrl']['rows']['neutral_hot_channel']; rr=D['routed']['rows']['neutral_hot_channel']
    P('%5s %8d %6s %14.6e %14.6e %14.6e %14.6e %14.6e'%('rtd-ct',i,role[i][:6],
      rr['nn'][i]-rc['nn'][i],rr['nn_a'][i]-rc['nn_a'][i],rr['n'][i]-rc['n'][i],
      rr['En'][i]-rc['En'][i],rr['Ei'][i]-rc['Ei'][i]))
P('')
P('  volume-integrated (row * neutral_volume_cm3) [s^-1]:')
vn=D['ctrl']['vol_n']
for arm in ('ctrl','routed'):
    for i in CELLS:
        r=D[arm]['rows']['neutral_hot_channel']
        P('    %-6s cell %3d : nn %15.6e   nn_a %15.6e'%(arm,i,r['nn'][i]*vn[i],r['nn_a'][i]*vn[i]))
P('')

# 2. full census of every row nonzero in nn or nn_a at these cells
P('-'*126)
P('TABLE 2B -- FULL CENSUS: every rhs_terms row with a nonzero time-averaged nn or nn_a at cells 259-261')
P('-'*126)
P('%-32s %5s %14s %14s %14s %14s'%('row','cell','nn_ctrl','nn_routed','nn_a_ctrl','nn_a_routed'))
for r in rows:
    hit=False
    for i in CELLS:
        for arm in ('ctrl','routed'):
            if D[arm]['rows'][r]['nn'][i]!=0 or D[arm]['rows'][r]['nn_a'][i]!=0: hit=True
    if not hit: continue
    for i in CELLS:
        P('%-32s %5d %14.6e %14.6e %14.6e %14.6e'%(r,i,
          D['ctrl']['rows'][r]['nn'][i],D['routed']['rows'][r]['nn'][i],
          D['ctrl']['rows'][r]['nn_a'][i],D['routed']['rows'][r]['nn_a'][i]))
    P('')
P('  Rows with IDENTICALLY ZERO nn and nn_a at cells 259-261 (both arms):')
zr=[r for r in rows if not any(D[a]['rows'][r]['nn'][i]!=0 or D[a]['rows'][r]['nn_a'][i]!=0
                               for i in CELLS for a in ('ctrl','routed'))]
P('    '+', '.join(zr))
P('')

# 3. hot-population diagnostics
P('-'*126)
P('TABLE 2C -- hot-neutral population diagnostics (top-level datasets), time-averaged, cells 259-261')
P('-'*126)
keys=['nn','nn_a','nn_hot','f_hot','tau_hot','hot_births','hot_recx','hot_S_cx','hot_ionized',
      'hot_wall','hot_flux_z','hot_n_flight','hot_end_fraction','hot_Ei_ionization','hot_Ei_recx',
      'n','Tn','Te','Ti']
P('%-20s %14s %14s %14s | %14s %14s %14s'%('dataset','259 ctrl','260 ctrl','261 ctrl','259 routed','260 routed','261 routed'))
for k in keys:
    if k not in D['ctrl']: continue
    P('%-20s %14.6e %14.6e %14.6e | %14.6e %14.6e %14.6e'%(k,
      D['ctrl'][k][259],D['ctrl'][k][260],D['ctrl'][k][261],
      D['routed'][k][259],D['routed'][k][260],D['routed'][k][261]))
P('')
P('  routed - ctrl:')
for k in keys:
    if k not in D['ctrl']: continue
    P('%-20s %14.6e %14.6e %14.6e'%(k,
      D['routed'][k][259]-D['ctrl'][k][259],D['routed'][k][260]-D['ctrl'][k][260],
      D['routed'][k][261]-D['ctrl'][k][261]))

open('/home/trloo/bapsf/bapsf-transport/cablp/scripts/l6a1_probe2.txt','w').write('\n'.join(out)+'\n')
print('\nWROTE scripts/l6a1_probe2.txt (%d lines)'%len(out))
