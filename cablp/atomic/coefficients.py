# The ONE import in this otherwise pure-data module, and it is deliberate: the
# He I ionization limit below is the SAME physical quantity as
# ``constants.I_ion``, and this module carried its own 4-decimal spelling of it
# until they were unified. Bound to a private alias so a data module that
# everything imports ``from`` does not also re-export ``I_ion``.
from ..constants import I_ion as _I_ION_EV

# IAEA EXPRESSION 1, He I Electron Cooling Rate Coefficients
aHeI = [0.6623e3, 0.9476e-1, 0.7456, -0.2592, 3.8098, 0.4026]

# IAEA EXPRESSION 4, He II Electron Cooling Rate Coefficients
aHeII = [0.3476e3, 0.1214, 0.7974, 0.4819, 1.4066, -0.3639e-2, 0.9720e-3, 0.4078]

# IAEA EXPRESSION 1, H I Electron Cooling Rate Coefficients
# Hydrogen-arm coefficients: QUARANTINED domain (untested; ruled 2026-08-27) --
# the guarded entry points refuse H; direct evaluation via IAEA_exp1(Te, aHI)
# is the caller's responsibility.
aHI = [0.1247e3, 0.2111, 0.5982, 0.4063, 0.9640e-3, 1.4523]

# IAEA EXPRESSION 6, H II Electron Cooling Rate Coefficients
# Hydrogen-arm coefficients: QUARANTINED domain (untested; ruled 2026-08-27) --
# the guarded entry points refuse H; direct evaluation via IAEA_exp6(Te, aHII)
# is the caller's responsibility.
aHII = [0.3180e-3, 0.2332, -0.9388e-2, 0.8617, 0.9663e-02, 0.8537]

# fitting parameters for ionization cross sections a_i
a_11s = [5.857e-1, -4.457e-1, 7.680e-1, -2.521e0, 3.317e0, 0.000e0]
# DEAD IN THE TRACKED TREE: the double-ionization row has no importer in this
# repository. It is RETAINED rather than deleted because the gitignored
# exploratory notebook `scripts/cross_sections.ipynb` imports and uses it on
# the working machine. Delete it with that notebook's disposition, not before.
a_11s_double = [1.323e-6, 8.208e-3, -6.676e-2, 2.978e-1, -1.925e-1, 0.000e0]

# fitting parameters for dipole-allowed excitation cross sections b_i_f
b_11s_21p = [7.087e-1, -9.347e-2, -1.598e0, 2.986e0, -1.293e0, 3.086e-1]

# --- He I ground-state singlet excitation manifold (cathode beam channel) ---
# Ralchenko, Janev, Kato, Fursa, Bray, de Heer, At. Data Nucl. Data Tables 94
# (2008) 603-622, doi:10.1016/j.adt.2007.11.003. Collision-strength fits
# Omega(x), x = E/E_th, converted to cross sections via the paper's Eq. (1):
#   sigma = pi*a0^2 * Ry / (g_i * E) * Omega(x),  g_i = 1 for 1^1S.
# form "allowed"  -> Table 1 / Eq. (2), 6 coefficients (n^1P levels)
# form "forbidden"-> Table 2 / Eq. (3), 5 coefficients (n^1S, n^1D, n^1F)
# Coefficients transcribed from the published tables and verified against the
# rendered PDF pages digit by digit (2026-07-20); the 2^1P row is identical to
# the in-repo b_11s_21p (provenance anchor for the whole set). Fit accuracy is
# 5-10% outside the threshold-resonance region (paper Sec. 2); underlying
# cross sections carry the 10-30% assessment of the DeltaS = 0 group.
#
# E_eV is the excitation threshold = radiated energy booked per event (NIST
# ASD level energies, 4 dp; the legacy 2^1P constant E_21p = 21.217848 in
# _cons.py differs in the 5th decimal and is left untouched). 2^1S is
# metastable (~20 ms two-photon lifetime), but at column densities electron
# collisions transfer it to 2^1P long before the beam-heating timescales
# care; its 20.6 eV is booked as radiated (WP-A).
#
# Triplet levels are deliberately absent: exchange-driven excitation
# collapses above ~50 eV, below the 60-180 eV beam range.
# The n >= 5 nP/nS/nD Rydberg tail is not in this registry; it is computed
# from the n = 4 rows by the paper's Eq. (5) scaling in
# atomic.cross_sections.He_singlet_tail_cross.
He_singlet_manifold = {
    "21S": {
        "E_eV": 20.6158,
        "form": "forbidden",
        "A": [1.888e-1, -5.754e-1, 3.439e0, -2.088e0, 2.544e1],
    },
    "21P": {
        "E_eV": 21.2180,
        "form": "allowed",
        "A": b_11s_21p,
    },
    "31S": {
        "E_eV": 22.9203,
        "form": "forbidden",
        "A": [4.033e-2, -1.872e-2, 2.368e0, -1.379e0, 1.258e2],
    },
    "31D": {
        "E_eV": 23.0736,
        "form": "forbidden",
        "A": [9.708e-3, 2.855e-2, -8.265e-2, 4.944e-2, 1.992e-1],
    },
    "31P": {
        "E_eV": 23.0870,
        "form": "allowed",
        "A": [1.730e-1, 2.410e-2, -4.709e-1, 7.690e-1, -3.216e-1, 8.568e-1],
    },
    "41S": {
        "E_eV": 23.6736,
        "form": "forbidden",
        "A": [1.613e-2, -5.564e-2, 2.943e-1, -2.024e-1, 2.342e1],
    },
    "41D": {
        "E_eV": 23.7361,
        "form": "forbidden",
        "A": [5.420e-3, 1.198e-2, -3.173e-2, 1.606e-2, 1.060e-1],
    },
    "41F": {
        "E_eV": 23.7370,
        "form": "forbidden",
        "A": [4.383e-5, -1.033e-4, 3.772e-3, 1.631e-2, 5.644e1],
    },
    "41P": {
        "E_eV": 23.7421,
        "form": "allowed",
        "A": [6.923e-2, 6.893e-3, -2.079e-1, 3.508e-1, -1.497e-1, 4.280e-2],
    },
}

# He I ionization limit [eV], taken FROM the canonical constant rather than
# re-typed, and the per-series quantum defects implied by the
# NIST n = 4 singlet levels above (E_n = E_lim - Ry/(n - delta)^2); used by the
# Eq. (5) Rydberg-tail scaling. The P defect is negative (singlet nP sits
# slightly above hydrogenic), consistent across n = 2/3/4 to < 0.0022. The
# shipped values reproduce the NIST-ASD-derived n = 4 defects (S 0.1417,
# P -0.0116, D 0.0019, F 0.0003) to three decimals, on the reduced-mass R_He;
# the n = 4 boxing is preferred over the asymptotic delta_0 because the n^-3
# Rydberg tail is dominated by the lowest n it scales from.
# UNCHECKED, flagged rather than corrected: the 41D and 41F thresholds in the
# manifold above sit 1.9 and 6.6 cm^-1 off their NIST ASD levels. Neither has
# been digit-proofed against the source, and no coefficient here is changed on
# the strength of that gap.
He_ionization_limit_eV = _I_ION_EV
He_singlet_quantum_defect = {"S": 0.142, "P": -0.012, "D": 0.002, "F": 0.000}
