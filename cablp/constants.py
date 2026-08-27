from mpmath import mp

en_factor = 2 / 3
ev_to_erg = 1.602176634e-12  # qe_SI * 1e-7

I_ion = 24.58738793623
I_21p = 21.217848
I_double = 79.005151
I_Ry = 13.6056931

kb_SI = 1.380649e-23  # in SI
m_p_SI = 1.6726219e-27  # kg
m_He_SI = 6.6464790809e-27  # kg  (see m_He_cgs below for the derivation)
m_e_SI = 9.1093837e-31  # kg
qe_SI = 1.602176634e-19  # electron charge in SI

kb_cgs = 1.380649e-16  # in cgs
m_p_cgs = 1.6726219e-24  # grams
# Neutral helium-4 ATOM mass. THE single definition point for the helium mass:
# every consumer imports one of these two spellings rather than re-deriving the
# product (unified 2026-08-21 -- the repo had carried three different hand-made
# products differing by up to 0.9 ppm, none of them citable).
# NIST/CODATA publishes no helium-atom mass directly, so this is a DERIVED
# product of two published constants:
#   Ar(4He) * u = 4.00260325413 u * 1.66053906892e-27 kg/u
#               = 6.646479080869e-27 kg
# Cross-checked against m(alpha) + 2 m_e - 79.005151 eV/c^2 (the double
# ionization energy), which agrees to 5e-12 relative.
m_He_cgs = 6.6464790809e-24  # grams
m_e_cgs = 9.1093837e-28  # grams
qe_cgs = 4.803e-10  # electron charge in cgs

H_e_mass_ratio = 1.8362e3

E_ion = mp.mpf("24.58738793623")  # Example value in eV
E_21p = mp.mpf("21.217848")  # Example value in eV
E_double = mp.mpf("79.005151")  # Example value in eV
M_e_eV = mp.mpf("510998.95069")  # Electron mass in eV/c^2
c_SI = mp.mpf("299792458")
c_cgs = mp.mpf("29979245800")  # Speed of light in cm/s
atm_cross_SI = mp.mpf(8.7974e-21)
atm_cross_cgs = mp.mpf(8.7974e-17)  # pi * a_0^2 in cm^2
Ry_eV = mp.mpf("13.605693122990")  # Rydberg energy in eV
