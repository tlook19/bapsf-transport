"""pd0 READ B: Landau-vs-collisional damping branching, measured on the
recorded per-cell states of the a1/a4/bx30 arms (pd0_endvent_*.npz), plus the
memo cross-checks and the READ C walker-reach numbers.

FORMULAS AND PROVENANCE (no new constants; everything from the repo or the
memo's own cited standard results):
  * nu_en = nn * K_m(Te), K_m = the SHIPPED
    cablp.atomic.cross_sections.he_electron_momentum_transfer_rate_cm3_s (two-node
    boxed table, sigma(1.5Te)*<v>). Collisional Langmuir AMPLITUDE damping
    = nu_en/2 [Ginzburg 1970; memo Key formulas].
  * Landau damping of a Langmuir wave on the local Maxwellian, evaluated at
    the beam-resonant phase velocity v_phi = v_b (k = omega/v_b):
        gamma_L = sqrt(pi/8) * omega_pe * (v_phi/v_te)^3
                  * exp(-v_phi^2/(2 v_te^2) - 3/2)
    with v_te^2 = Te/m_e, omega_pe = 5.64e4 sqrt(ne) [Krall & Trivelpiece
    Sec. 8.6 standard result, the Bohm-Gross -3/2 term included; the memo's
    Sec.4 anchors (e^-37 at Te 5, Landau-limited threshold ~4e14 at Te 25)
    reproduce under exactly this expression -- shown below before use].
  * v_b from the LIVE cathode drop of the same trajectory step,
    v_b = beam_speed_cm_s(phi_c(t)) (repo function; CSDA slowing along the
    column neglected for v_phi -- disclosed).
  * f_Landau = gamma_L / (gamma_L + nu_en/2).

Reads only pd0_endvent_*.npz; writes only this text. Read-only w.r.t. repo.
"""
import numpy as np
from cablp.atomic.cross_sections import (
    he_electron_momentum_transfer_rate_cm3_s as K_m,
    HE_EN_MT_SIGMA_CM2, HE_EN_MT_SIGMA_BRACKET_CM2,
)
from cablp.cathode.beam_deposition import (
    beam_speed_cm_s, coulomb_stopping_eV_per_cm,
)
from cablp.constants import ev_to_erg, m_e_cgs

OMEGA = 5.64e4  # memo's Omega: omega_pe = OMEGA*sqrt(ne) [rad/s]


def gamma_landau(ne, Te, E_beam_eV):
    ne = np.asarray(ne, float); Te = np.asarray(Te, float)
    r2 = np.maximum(2.0 * E_beam_eV / np.maximum(Te, 1e-12), 1e-12)  # (v_phi/v_te)^2
    return (np.sqrt(np.pi / 8.0) * OMEGA * np.sqrt(np.maximum(ne, 0.0))
            * r2 ** 1.5 * np.exp(-0.5 * r2 - 1.5))


def nu_en_half(nn, Te, km_scale=1.0):
    return 0.5 * np.asarray(nn, float) * K_m(np.asarray(Te, float)) * km_scale


def f_landau(ne, Te, nn, E_beam_eV, km_scale=1.0):
    gL = gamma_landau(ne, Te, E_beam_eV)
    nc = nu_en_half(nn, Te, km_scale)
    return gL / (gL + nc)


print("== pd0_branching: memo cross-checks first (stance n_b=2.95e8, "
      "E=177.6 eV, nn=2e13) ==")
for Te in (5.0, 25.0):
    nu = 2.0 * nu_en_half(2e13, Te)
    print(f"   Te={Te:g}: nu_en = {float(nu):.3e} 1/s  "
          f"(memo: 1.8e6 at Te 5, 1.4e6 at Te 25)")
E = 177.6
for Te in (5.0, 25.0):
    expo = E / Te + 1.5
    print(f"   Te={Te:g}: Landau exponent = e^-{expo:.1f} "
          f"(memo Te 5: 'e^-37')")
# Landau-limited threshold: gamma_r = 0.687 omega_pe (n_b/ne)^(1/3) = gamma_L
nb = 2.95e8
r2 = 2.0 * E / 25.0
gL_over_wpe = float(np.sqrt(np.pi / 8.0) * r2 ** 1.5 * np.exp(-0.5 * r2 - 1.5))
ne_thr = nb / (gL_over_wpe / 0.687) ** 3
print(f"   Te=25: gamma_L/omega_pe = {gL_over_wpe:.3e}; Landau-limited "
      f"threshold ne = {ne_thr:.2e} (memo: ~4e14) ")
for Te in (5.0, 25.0):
    nc = float(nu_en_half(2e13, Te))
    g1 = gL_over_wpe if Te == 25.0 else float(
        np.sqrt(np.pi/8.0)*(2*E/Te)**1.5*np.exp(-E/Te-1.5))
    ne_cross = (nc / (g1 * OMEGA)) ** 2
    print(f"   Te={Te:g}: gamma_L = nu_en/2 crossing at ne = {ne_cross:.3e} "
          f"cm^-3 (E=177.6)")
print("   NOTE: memo Sec.4 says Landau 'dominates above a few x1e8' at Te 25;"
      " under the same formula that reproduces its other two anchors the"
      " crossing is ~4e6 -- Landau dominance sets in LOWER than the memo's"
      " phrasing (conservative direction). Flagged, not resolved here.")

print("\n== branching at stance conditions (Te 25, nn 2e13, E 177.6) ==")
for ne in (1e8, 1e9, 1e10, 1e11):
    fl = float(f_landau(ne, 25.0, 2e13, E))
    lo = float(f_landau(ne, 25.0, 2e13, E,
                        km_scale=HE_EN_MT_SIGMA_BRACKET_CM2[1][0]
                        / HE_EN_MT_SIGMA_CM2[1]))
    hi = float(f_landau(ne, 25.0, 2e13, E,
                        km_scale=HE_EN_MT_SIGMA_BRACKET_CM2[1][1]
                        / HE_EN_MT_SIGMA_CM2[1]))
    print(f"   ne={ne:.0e}: f_Landau = {fl:.4f}  "
          f"[K_m 25eV bracket: {min(lo,hi):.4f}..{max(lo,hi):.4f}]")

print("\n== 50% crossing surface gamma_L(Te,ne) = nu_en(Te)/2 at nn=2e13 ==")
for Eb, tag in ((177.6, "stance E=177.6"), (60.0, "foot-typical E=60"),
                (30.0, "cold-foot E=30")):
    line = []
    for Te in (2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0):
        nc = float(nu_en_half(2e13, Te))
        g1 = float(np.sqrt(np.pi/8.0)*(2*Eb/Te)**1.5*np.exp(-Eb/Te-1.5))
        ne_c = (nc / (g1 * OMEGA)) ** 2 if g1 > 0 else np.inf
        line.append(f"Te{Te:g}:{ne_c:.1e}")
    print(f"   {tag}: ne_50% = " + "  ".join(line))

print("\n== trajectory branching on the recorded arm states ==")
for arm, path in (("a1", "scripts/pd0_endvent_a1.npz"),
                  ("a4", "scripts/pd0_endvent_a4.npz"),
                  ("bx30", "scripts/pd0_endvent_bx30.npz")):
    z = np.load(path, allow_pickle=True)
    import json as _json
    meta = _json.loads(str(z["meta"]))
    t = z["t"]; phic = z["phi_c"]
    n = z["cell_n"]; Te = z["cell_Te"]; nn = z["cell_nn"]
    act = z["active"].astype(bool)
    i_on = int(meta["i_on"])
    steps = {"onset": i_on, "mid": (i_on + len(t) - 1) // 2, "end": len(t) - 1}
    print(f"\n   -- {arm} (t_on={meta['t_on']:.3e} s) --")
    for nm, i in steps.items():
        Eb = float(phic[i])
        fl = f_landau(n[i, act], Te[i, act], nn[i, act], Eb)
        j_pk = int(np.argmax(n[i, act]))
        print(f"   {nm} t={t[i]:.3e}: phi_c={Eb:.1f} V  "
              f"f_Landau[active cells] min/median/max = "
              f"{fl.min():.3f}/{np.median(fl):.3f}/{fl.max():.3f}  "
              f"at n-argmax cell: {fl[j_pk]:.3f}  "
              f"(Te there {Te[i, act][j_pk]:.2f} eV, "
              f"ne {n[i, act][j_pk]:.2e}, nn {nn[i, act][j_pk]:.2e})")
    # thermostat trace: f_Landau at the running density-argmax cell
    idx = np.linspace(i_on, len(t) - 1, 8).astype(int)
    trace = []
    for i in idx:
        j = int(np.argmax(n[i, act]))
        trace.append(f"{t[i]*1e6:.1f}us:{float(f_landau(n[i,act][j], Te[i,act][j], nn[i,act][j], float(phic[i]))):.3f}")
    print("   thermostat trace (t : f_Landau @ n-argmax): " + "  ".join(trace))

print("\n== READ C numbers: tail-walker Coulomb reach (fast_electron model) ==")
print("   range R(E_tail) = int dE / (dE/dx), dE/dx = "
      "coulomb_stopping_eV_per_cm(E, ne, Te) [repo function]; floor 1.5*Te")
for ne in (1e8, 1e9, 1e10, 1e11):
    row = []
    for Et in (30.0, 75.0, 150.0):
        Te = 5.0
        Egrid = np.linspace(1.5 * Te, Et, 4000)
        L = np.array([coulomb_stopping_eV_per_cm(float(e), ne, Te) for e in Egrid])
        R_cm = float(np.trapezoid(1.0 / np.maximum(L, 1e-300), Egrid))
        row.append(f"E{Et:g}:{R_cm/100.0:.3g} m")
    print(f"   ne={ne:.0e} (Te 5): " + "  ".join(row))
print("   (machine column in the r2 arm geometry: ~17 m; foot densities "
      "1e8-1e10 -> reach km-to-Mm class, i.e. machine-transparent)")
