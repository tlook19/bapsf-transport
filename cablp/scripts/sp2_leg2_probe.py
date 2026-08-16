# sp1 LEG 2 -- THE ACCOUNTING (diagnostician probe; NO SOLVES, arithmetic +
# h5 reads only). Reads sp1_ref.h5 (leg-1 fluid REF, campaign 7bd4041 era)
# and reproduces/extends the leg-1 throughput ledger:
#   (a) required influx vs shipped S_gp throughput, BOTH denominator
#       conventions (nominal-sccm 5200*4.477962e17 used by the leg-1 .cmd
#       arithmetic, and the AS-APPLIED solver puff diagnostic
#       puff_particles_per_s, which carries gas_puff_valves);
#   (b) where the shipped cosine_pipe routing deposits (first-flight),
#       fraction landing in the required band;
#   (c) the flow demand: flux through the band's bounding faces and the
#       implied directed velocity at the REF's own nn(z), vs He thermal.
# Registration: CAMPAIGN_LOG "sp1 LEG 1 EXECUTED TO PRODUCTS" + Tom's leg-2
# dispatch. Artifact: sp2_leg2_probe.txt (this script's stdout).
import json, math
import numpy as np
import h5py

H5 = "/Users/tlook/bapsf/bapsf-transport/cablp/scripts/sp1_ref.h5"
SCCM = 4.477962e17          # particles/s per sccm (neutrals.py:760)
A_1045 = 7.1828e13          # cm^-3 s^-1, sp1_p1045r.cmd (frozen linear rule)
A_630  = 9.4510e13          # cm^-3 s^-1, sp1_p630r.cmd
SGP = 5200.0                # sccm, frozen calibration
PLATEAU = (0.015, 0.025)    # s, labeled plateau window for nn(z) means

with h5py.File(H5, "r") as f:
    p = json.loads(f.attrs["params_json"])
    fl = json.loads(f.attrs["flags_json"])
    g = f["geometry"]
    z = g["z_cm"][:]; V = g["neutral_volume_cm3"][:]
    L = g["length_cm"][:]; roles = [r.decode() if isinstance(r, bytes) else str(r) for r in g["cell_role"][:]]
    Rp = g["Rp_cm"][:]; Rm = g["Rm_cm"][:]
    t = f["time"][:]
    nn = f["nn"][:]; nn_a = f["nn_a"][:] if "nn_a" in f else None
    puff_pps = f["gas_puff_diagnostics"]["puff_particles_per_s"][:]
    sgp_sccm = f["gas_puff_diagnostics"]["S_gp_sccm"][:]

print("== sp2 LEG 2 ledger probe over sp1_ref.h5 ==")
print(f"config: S_gp={p['S_gp']}, gas_puff_valves={p['gas_puff_valves']}, "
      f"profile={p['gas_puff_profile']!r}, z0={p['gas_puff_z_cm']} cm, "
      f"throw={p['gas_puff_throw_cm']} cm, Tn_K={p['Tn_K']}, "
      f"two_zone={fl.get('neutral_two_zone')}, Rp={Rp.max()}, Rm={Rm.max()}")

# ---------- (a) throughput ledger ----------
nominal = SGP * SCCM                      # leg-1 .cmd denominator
applied_max = float(np.max(puff_pps))     # as-applied solver diagnostic
mask = (t >= PLATEAU[0]) & (t <= PLATEAU[1])
applied_plateau = float(np.mean(puff_pps[mask]))
sumV = float(np.sum(V))
print(f"\nsum(V_neutral) = {sumV:.6e} cm^3   (leg-1 quoted 1.892024e+07)")
print(f"nominal throughput 5200 sccm x 4.477962e17 = {nominal:.4e} /s")
print(f"AS-APPLIED puff_particles_per_s: max={applied_max:.4e} /s, "
      f"plateau mean [{PLATEAU[0]}-{PLATEAU[1]} s]={applied_plateau:.4e} /s, "
      f"S_gp_sccm applied max={np.max(sgp_sccm)}")
print(f"ratio applied/nominal = {applied_max/nominal:.4f}  "
      f"(gas_puff_valves={p['gas_puff_valves']})")
for name, A in (("p1045r", A_1045), ("p630r", A_630)):
    R = A * sumV
    print(f"required influx {name}: A={A:.4e} -> R = {R:.4e} /s | "
          f"{100*R/nominal:.1f}% of nominal | {100*R/applied_max:.1f}% of as-applied")

# ---------- (b) shipped first-flight deposition ----------
ELIG = {"puff", "column", "source", "domain", "end"}
elig = np.array([r in ELIG for r in roles])
z0 = float(p["gas_puff_z_cm"]); d = float(p["gas_puff_throw_cm"])
w = np.zeros_like(z)
w[elig] = (1.0 / (1.0 + ((z[elig] - z0) / d) ** 2) ** 2) * L[elig]
w /= w.sum()
for lo, hi, tag in ((790.0, 1045.0, "required band 790-1045"),
                    (1045 - 64, 1045 + 64, "1045 +/- 2sig(32)"),
                    (790 - 64, 790 + 64, "790 +/- 2sig(32)"),
                    (0.0, 300.0, "source region z<300")):
    frac = w[(z >= lo) & (z <= hi)].sum()
    print(f"shipped cosine_pipe first-flight fraction in [{lo:.0f},{hi:.0f}] "
          f"({tag}): {100*frac:.4f}%")

# ---------- (c) flow demand ----------
nn_bar = nn[mask].mean(axis=0)
nna_bar = nn_a[mask].mean(axis=0) if nn_a is not None else np.zeros_like(nn_bar)
def at(zq):
    i = int(np.argmin(np.abs(z - zq)))
    return i
print("\nREF plateau-mean nn(z) [column] / nn_a(z) [annulus], cm^-3:")
for zq in (614, 790, 918, 981, 1045, 1109, 1300, 1700):
    i = at(zq)
    print(f"  z={z[i]:7.1f}  nn={nn_bar[i]:.3e}  nn_a={nna_bar[i]:.3e}")
# per-cell areas: the production geometry carries an end flare (Rm=100 cm
# cells); use the LOCAL radii at each band face, not the global max.
print(f"\nRm profile: min {Rm.min()}, max {Rm.max()} (end-flare cells); "
      f"Rm at z=1045: {Rm[at(1045)]}")
for name, R, zc in (("p1045r", A_1045 * sumV, 1045.0), ("p630r", A_630 * sumV, 630.0)):
    # band faces at +/-2 sigma of the Gaussian footprint
    for face in (zc - 64, zc + 64):
        i = at(face)
        ncol, nann = nn_bar[i], nna_bar[i]
        A_col = math.pi * float(Rp[i]) ** 2
        A_ann = math.pi * (float(Rm[i]) ** 2 - float(Rp[i]) ** 2)
        # column-only supply (probe fed the column zone)
        v2 = R / (2 * A_col * ncol)
        v1 = R / (A_col * ncol)
        # chamber-wide supply weighted by both zone densities
        veff2 = R / (2 * (A_col * ncol + A_ann * nann))
        veff1 = R / (A_col * ncol + A_ann * nann)
        print(f"{name}: face z={face:.0f} (Rp={Rp[i]:.0f}/Rm={Rm[i]:.0f}, A_col={A_col:.0f}, A_ann={A_ann:.0f} cm^2; nn={ncol:.2e}, nn_a={nann:.2e}) | "
              f"column-only v: 2-sided {v2/1e5:.1f} km/s, 1-sided {v1/1e5:.1f} km/s | "
              f"chamber v_eff: 2-sided {veff2/1e5:.2f} km/s, 1-sided {veff1/1e5:.2f} km/s")
kB = 1.380649e-16; mHe = 6.6464731e-24
for T in (300.0, 1000.0, 2000.0):
    vmean = math.sqrt(8 * kB * T / (math.pi * mHe))
    print(f"He mean thermal speed at {T:.0f} K: {vmean/1e5:.2f} km/s")

# ---------- leg-3 bridge ----------
print("\nfoot-accumulation bridge (order of magnitude):")
for dt_foot in (2.0e-3, 4.5e-3):
    for tag, thr in (("nominal", nominal), ("as-applied", applied_max)):
        N = thr * dt_foot
        for name, R in (("p1045r", A_1045 * sumV),):
            print(f"  foot {1e3*dt_foot:.1f} ms x {tag} throughput = {N:.3e} particles"
                  f" -> sustains required {name} rate for {1e3*N/R:.2f} ms")
print("thermal spread in foot: 1.26 km/s x 2 / 4.5 ms = "
      f"{1.26e5*2e-3/100:.1f} / {1.26e5*4.5e-3/100:.1f} m")
