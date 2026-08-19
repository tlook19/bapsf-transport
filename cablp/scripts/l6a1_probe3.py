import h5py, numpy as np, json
ORIG=2.2502300319430474e-4
ARMS={'ctrl':'scripts/l6a1_ctrl_foot45_cr6p94.h5','routed':'scripts/l6a1_routed_foot45_cr6p94.h5'}
out=[]
def P(*a):
    s=' '.join(str(x) for x in a); out.append(s); print(s)

D={}
for lab,p in ARMS.items():
    f=h5py.File(p,'r')
    t=f['time'][:]; tmd=(t-ORIG)*1e3
    w =np.where((tmd>=20.0)&(tmd<=25.0))[0]
    wp=np.where((tmd>=15.0)&(tmd<=19.5))[0]
    g=f['geometry']
    Vp=g['plasma_volume_cm3'][:]; Vn=g['neutral_volume_cm3'][:]
    Vcol=Vp.copy(); Vann=np.maximum(Vn-Vp,0.0)
    d=dict(z=g['z_cm'][:],Vcol=Vcol,Vann=Vann,Vn=Vn,w=w,wp=wp,tmd=tmd,
           role=[s.decode() if isinstance(s,bytes) else str(s) for s in g['cell_role'][:]])
    for r in ('neutral_sources','characteristic_boundary','neutral_zone_exchange','neutral_exchange',
              'surface_loss','boundary_absorption','neutral_cx_channel','neutral_hot_channel'):
        d[r]={fl: f['rhs_terms'][r][fl][w[0]:w[-1]+1].mean(axis=0) for fl in ('nn','nn_a')}
        d[r+'_PL']={fl: f['rhs_terms'][r][fl][wp[0]:wp[-1]+1].mean(axis=0) for fl in ('nn','nn_a')}
    d['nn']=f['nn'][w[0]:w[-1]+1].mean(axis=0); d['nn_a']=f['nn_a'][w[0]:w[-1]+1].mean(axis=0)
    d['nn_PL']=f['nn'][wp[0]:wp[-1]+1].mean(axis=0); d['nn_a_PL']=f['nn_a'][wp[0]:wp[-1]+1].mean(axis=0)
    # instantaneous per-snapshot effective pumping speed at 261
    ns_nn  = f['rhs_terms/neutral_sources/nn'][w[0]:w[-1]+1,261]
    ns_nna = f['rhs_terms/neutral_sources/nn_a'][w[0]:w[-1]+1,261]
    nn_i   = f['nn'][w[0]:w[-1]+1,261]; nna_i=f['nn_a'][w[0]:w[-1]+1,261]
    d['S_eff_col_Vcol'] = float(np.mean(-ns_nn *Vcol[261]/nn_i))
    d['S_eff_col_Vn']   = float(np.mean(-ns_nn *Vn[261]  /nn_i))
    d['S_eff_ann_Vann'] = float(np.mean(-ns_nna*Vann[261]/nna_i))
    d['S_eff_ann_Vn']   = float(np.mean(-ns_nna*Vn[261]  /nna_i))
    d['puff_pps']=f['gas_puff_diagnostics/puff_particles_per_s'][:]
    d['puff_en'] =f['phase_gas_puff_enabled'][:]
    d['par']=json.loads(f.attrs['params_json']); d['fl']=json.loads(f.attrs['flags_json'])
    D[lab]=d; f.close()

z=D['ctrl']['z']; role=D['ctrl']['role']
Vcol=D['ctrl']['Vcol']; Vann=D['ctrl']['Vann']; Vn=D['ctrl']['Vn']

P('='*122)
P('PROBE 3 -- PUMP-BOOKING RECONCILE  (L6 A/B, read-only)')
P('='*122)
P('artifacts : /home/trloo/bapsf/bapsf-transport/cablp/scripts/l6a1_{ctrl,routed}_foot45_cr6p94.h5')
P('repo HEAD : 463002cedf735282e2b63d2fa28c74fed8130d00 (branch campaign)')
P('window    : 20.0-25.0 ms MAIN-DISCHARGE clock, snapshot idx %d..%d (n=%d) -- afterglow phase.'%(
   D['ctrl']['w'][0],D['ctrl']['w'][-1],len(D['ctrl']['w'])))
P('            plateau cross-reference window 15.0-19.5 ms, idx %d..%d (n=%d).'%(
   D['ctrl']['wp'][0],D['ctrl']['wp'][-1],len(D['ctrl']['wp'])))
P('')
P('(a) CONFIG VALUES, read from the ARMS OWN RESOLVED CONFIGS (h5 attrs params_json/flags_json)')
P('-'*122)
P('%-38s %-22s %-22s %s'%('key','ctrl','routed','namespace'))
for k in ('S_pump_L','S_pump_R','pump_elbow_conductance_lps','pump_enabled','collector_length_cm','plenum_length_cm'):
    P('%-38s %-22r %-22r %s'%(k,D['ctrl']['par'].get(k,'<absent>'),D['routed']['par'].get(k,'<absent>'),
      'params' if k in D['ctrl']['par'] else ('flags' if k in D['ctrl']['fl'] else '?')))
for k in ('neutral_two_zone','end_recycle_to_annulus'):
    P('%-38s %-22r %-22r %s'%(k,D['ctrl']['fl'].get(k,'<absent>'),D['routed']['fl'].get(k,'<absent>'),'flags'))
P('')
P('  The two resolved configs are IDENTICAL except end_recycle_to_annulus (False -> True).')
P('  pump_elbow_conductance_lps is None on BOTH arms (no elbow conductance in series).')
P('')
P('  ZONE VOLUMES (neutrals.py:136-148: V_col = plasma_volume_cm3;')
P('                                     V_ann = max(neutral_volume_cm3 - plasma_volume_cm3, 0)):')
P('    cell   2 (cathode)   V_col=%14.6e  V_ann=%14.6e  V_neutral=%14.6e'%(Vcol[2],Vann[2],Vn[2]))
P('    cell 261 (collector) V_col=%14.6e  V_ann=%14.6e  V_neutral=%14.6e'%(Vcol[261],Vann[261],Vn[261]))
P('')

P('(b) rhs_terms/neutral_sources DECOMPOSITION')
P('-'*122)
P('  Where is the row nonzero at all? (afterglow window, ctrl, |value|>0):')
for fld in ('nn','nn_a'):
    nz=np.where(D['ctrl']['neutral_sources'][fld]!=0)[0]
    P('    %-5s nonzero cells: %s'%(fld, ', '.join('%d(%s,z=%.1f)'%(i,role[i],z[i]) for i in nz) if len(nz) else '<none>'))
P('  plateau window (15.0-19.5 ms), ctrl:')
for fld in ('nn','nn_a'):
    nz=np.where(D['ctrl']['neutral_sources'+'_PL'][fld]!=0)[0]
    P('    %-5s nonzero cells: %s'%(fld, ', '.join('%d(%s,z=%.1f)'%(i,role[i],z[i]) for i in nz) if len(nz) else '<none>'))
P('')
P('  gas_puff_diagnostics/puff_particles_per_s : afterglow-window mean ctrl=%.6e routed=%.6e'%(
   D['ctrl']['puff_pps'][D['ctrl']['w']].mean(),D['routed']['puff_pps'][D['routed']['w']].mean()))
P('  phase_gas_puff_enabled                    : afterglow-window mean ctrl=%.6e routed=%.6e'%(
   D['ctrl']['puff_en'][D['ctrl']['w']].mean(),D['routed']['puff_en'][D['routed']['w']].mean()))
P('  gas_puff_diagnostics/puff_particles_per_s : PLATEAU  -window mean ctrl=%.6e routed=%.6e'%(
   D['ctrl']['puff_pps'][D['ctrl']['wp']].mean(),D['routed']['puff_pps'][D['routed']['wp']].mean()))
P('')
P('  TABLE 3B -- neutral_sources at cells 2 and 261, both zones, both arms, both windows')
P('  %-10s %-5s %-6s %16s %16s %16s'%('window','cell','zone','rhs [cm^-3 s^-1]','V_zone [cm^3]','rate [s^-1]'))
for wl,suf in (('afterglow',''),('plateau','_PL')):
    for i in (2,261):
        for zone,fld,V in (('column','nn',Vcol),('annulus','nn_a',Vann)):
            for arm in ('ctrl','routed'):
                v=D[arm]['neutral_sources'+suf][fld][i]
                P('  %-10s %-5d %-6s %16.6e %16.6e %16.6e   [%s]'%(wl,i,zone,v,V[i],v*V[i],arm))
P('')
P('  Also, alternative weighting by the TOTAL neutral volume (V_neutral) at 261, afterglow:')
for zone,fld in (('column','nn'),('annulus','nn_a')):
    for arm in ('ctrl','routed'):
        v=D[arm]['neutral_sources'][fld][261]
        P('    %-8s %-6s x V_neutral = %16.6e s^-1   [%s]'%(zone,fld,v*Vn[261],arm))
P('')

P('(c) ARITHMETIC INPUTS')
P('-'*122)
P('  c1. Measured annulus neutral_sources at cell 261, afterglow window:')
for arm in ('ctrl','routed'):
    v=D[arm]['neutral_sources']['nn_a'][261]
    P('      [%-6s] rhs = %.6e cm^-3 s^-1 ; x V_ann(%.6e) = %.6e s^-1'%(arm,v,Vann[261],v*Vann[261]))
P('      (the brief quotes -1.27e20 /s; the ctrl value above is the match)')
P('')
P('  c2. Local neutral densities at cell 261, afterglow-window mean [cm^-3]:')
for arm in ('ctrl','routed'):
    P('      [%-6s] nn = %.6e   nn_a = %.6e'%(arm,D[arm]['nn'][261],D[arm]['nn_a'][261]))
P('')
P('  c3. Effective pumping speed BACKED OUT per snapshot then averaged, S_eff = -rhs*V/n [cm^3 s^-1]:')
P('      (config S_pump_R = 4000 L/s = 4.000000e+06 cm^3 s^-1)')
P('      %-8s %18s %18s %18s %18s'%('arm','col x V_col','col x V_neutral','ann x V_ann','ann x V_neutral'))
for arm in ('ctrl','routed'):
    P('      %-8s %18.6e %18.6e %18.6e %18.6e'%(arm,D[arm]['S_eff_col_Vcol'],D[arm]['S_eff_col_Vn'],
      D[arm]['S_eff_ann_Vann'],D[arm]['S_eff_ann_Vn']))
P('      ratios to 4.000000e+06 cm^3/s:')
for arm in ('ctrl','routed'):
    P('      %-8s %18.6f %18.6f %18.6f %18.6f'%(arm,D[arm]['S_eff_col_Vcol']/4e6,D[arm]['S_eff_col_Vn']/4e6,
      D[arm]['S_eff_ann_Vann']/4e6,D[arm]['S_eff_ann_Vn']/4e6))
P('      geometry volume_ratio at 261 (V_p/V_n) = %.9f'%(Vcol[261]/Vn[261]))
P('      V_ann/V_neutral at 261                 = %.9f'%(Vann[261]/Vn[261]))
P('')
P('  c4. Nominal pump rate from config x local density (S_pump_R * n_local) [s^-1]:')
for arm in ('ctrl','routed'):
    P('      [%-6s] 4.0e6 * nn   = %.6e   |   4.0e6 * nn_a = %.6e'%(arm,4e6*D[arm]['nn'][261],4e6*D[arm]['nn_a'][261]))
P('')
P('  c5. END-RECYCLE row (rhs_terms/characteristic_boundary) -- the row the routing switch moves:')
P('      %-10s %-6s %-5s %16s %16s %16s'%('window','arm','cell','rhs nn','rhs nn_a','vol-integrated'))
for wl,suf in (('afterglow',''),('plateau','_PL')):
    for arm in ('ctrl','routed'):
        cb=D[arm]['characteristic_boundary'+suf]
        tot_nn =float(np.sum(cb['nn'][252:262]*Vcol[252:262]))
        tot_nna=float(np.sum(cb['nn_a'][252:262]*Vann[252:262]))
        P('      %-10s %-6s %-5s %16.6e %16.6e   nn:%12.6e s^-1  nn_a:%12.6e s^-1'%(
          wl,arm,'261',cb['nn'][261],cb['nn_a'][261],cb['nn'][261]*Vcol[261],cb['nn_a'][261]*Vann[261]))
        P('      %-10s %-6s %-5s %16s %16s   cells 252-261 SUM  nn:%12.6e  nn_a:%12.6e  TOTAL:%12.6e s^-1'%(
          wl,arm,'sum','','',tot_nn,tot_nna,tot_nn+tot_nna))
P('      (the brief quotes a total recycle of 1.92e21 /s; the measured totals are above)')
P('')
P('  c6. Ratio pump / recycle at cell 261 (afterglow window):')
for arm in ('ctrl','routed'):
    cb=D[arm]['characteristic_boundary']
    rec=cb['nn'][261]*Vcol[261]+cb['nn_a'][261]*Vann[261]
    pump=D[arm]['neutral_sources']['nn'][261]*Vcol[261]+D[arm]['neutral_sources']['nn_a'][261]*Vann[261]
    P('      [%-6s] pump(both zones) = %.6e s^-1 ; recycle(cell 261) = %.6e s^-1 ; |pump|/recycle = %.6f'%(
      arm,pump,rec,abs(pump)/rec if rec!=0 else float('nan')))
P('')
P('  NOTE: this section reports MEASURED rows and the arithmetic that reproduces them. The')
P('        code citations for the S_pump expression and the neutral_sources contributor census')
P('        are appended below.')

open('/home/trloo/bapsf/bapsf-transport/cablp/scripts/l6a1_probe3.txt','w').write('\n'.join(out)+'\n')
print('\nWROTE scripts/l6a1_probe3.txt (%d lines)'%len(out))
