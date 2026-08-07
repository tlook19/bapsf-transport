"""ghostinflow probe 3: whole-system particle balance from saved trajectories."""
import sys, json
import numpy as np, h5py

for path in sys.argv[1:]:
    f = h5py.File(path, 'r')
    t = f['time'][:]
    g = f['geometry']
    Vn = g['neutral_volume_cm3'][:]
    Vc = g['plasma_volume_cm3'][:]
    Va = np.maximum(Vn - Vc, 0.0)
    params = json.loads(f.attrs['params_json'])
    two_zone = 'nn_a' in f and np.abs(f['nn_a'][:]).max() > 0
    nn = f['nn'][:]
    if two_zone:
        N_n = (nn * Vc).sum(1) + (f['nn_a'][:] * Va).sum(1)
    else:
        N_n = (nn * Vn).sum(1)
    N_p = (f['n'][:] * Vc).sum(1)
    puff = np.trapezoid(f['gas_puff_diagnostics/puff_particles_per_s'][:], t)

    def integ(term, field='n', vol=Vc, clip=None):
        a = f['rhs_terms/%s/%s' % (term, field)][:]
        if clip == '+': a = np.clip(a, 0.0, None)
        if clip == '-': a = np.clip(a, None, 0.0)
        return np.trapezoid((a * vol).sum(1), t)

    ion = integ('ionization_birth') + integ('beam_ionization_birth')
    rec = -(integ('recombination_rad_loss') + integ('recombination_3b_loss'))
    bnd = -integ('characteristic_boundary')
    ano = -integ('anode_collection')
    adv = integ('plasma_advective_flux')
    print("=" * 96)
    print(path, " model=", params.get('neutral_model'), " S_gp=", params.get('S_gp'), " two_zone=", two_zone)
    print("  puff delivered            %+.5g" % puff)
    print("  dN_neutral                %+.5g" % (N_n[-1] - N_n[0]))
    print("  dN_plasma                 %+.5g" % (N_p[-1] - N_p[0]))
    print("  (dN_n+dN_p)/puff          %.4f" % ((N_n[-1]-N_n[0]+N_p[-1]-N_p[0]) / puff))
    print("  ionization (bulk+beam)    %+.5g   (= %.2f x puff)" % (ion, ion / puff))
    print("  recombination             %+.5g" % rec)
    print("  boundary absorption loss  %+.5g" % bnd)
    print("  anode collection loss     %+.5g" % ano)
    print("  plasma advective net      %+.5g" % adv)
    print("  plasma balance check: ion - rec - bnd - ano + adv = %+.5g  vs dN_p %+.5g   (resid/ion %.4f)"
          % (ion - rec - bnd - ano + adv, N_p[-1]-N_p[0],
             (ion - rec - bnd - ano + adv - (N_p[-1]-N_p[0])) / ion))
    excess = (N_n[-1]-N_n[0]) + (N_p[-1]-N_p[0]) - puff
    print("  UNEXPLAINED EXCESS        %+.5g  = %.4f x puff = %.4f x ionization"
          % (excess, excess / puff, excess / ion))
