"""ea1x: per-arm no-solve config diff -- PREFLIGHT for the ms-class B2 extension.

Same construction as scripts/ea1v_diffcfg.py (which is NOT modified): builds the
efold1 BASELINE config (regime_r2_overlap_gate.build_config(20, False), the a1
arm's config verbatim) and the ARMED config for each ea1x arm through the SAME
namespace resolver the run harnesses use (recertdiag2_traj.apply_set), then
prints the key-by-key diff with recertdiag2_configdiff.diff.

Standing rule (the diffcfg technique): expected deltas only, or stop.

Six arms: the four stage-1 arms carried to t-target 5e-3 s, plus the two B4
emission-insensitivity nulls at the central seed:
  * T_s 1998.15 -> 1910.0        (INITIAL cathode surface temperature; the
    warming model evolves it from cathode_Ts_base_K, which is untouched)
  * Te_birth_ionization "local" -> "floor"
Both are the EXACT keys/values the efold1-era discriminator measured as
bit-identical nulls (recertdiag2_r2_Ts1910.txt / recertdiag2_r2_tebirthfloor.txt).

Constructs nothing, integrates nothing. Writes only stdout.
"""
import sys

from recertdiag2_configdiff import diff
from recertdiag2_traj import apply_set
from regime_r2_overlap_gate import build_config

SEED = ["cathode_emitting_area=true",
        "cathode_emitting_area_initial_fraction=0.0075"]

ARMS = {
    "ea1x_seed": SEED,
    "ea1x_seed_lo": ["cathode_emitting_area=true",
                     "cathode_emitting_area_initial_fraction=0.0063"],
    "ea1x_seed_hi": ["cathode_emitting_area=true",
                     "cathode_emitting_area_initial_fraction=0.0087"],
    "ea1x_disposal": SEED + ['heating_anomalous_disposal="landau_branched"',
                             "heating_anomalous_tail_phi_c_fraction=1.0"],
    "ea1x_b4_ts": SEED + ["T_s=1910.0"],
    "ea1x_b4_tebirth": SEED + ['Te_birth_ionization="floor"'],
}


def main():
    print("== ea1x_diffcfg: efold1 a1 BASELINE config  vs  each ea1x arm's ARMED "
          "config, both built at HEAD 1abe696, no solve\n")
    for label, sets in ARMS.items():
        pa, fa = build_config(20, False)
        pb, fb = build_config(20, False)
        applied = apply_set(pb, fb, sets)
        print(f"--- {label}: --set {' '.join(sets)}")
        print(f"    apply_set resolved: {applied}")
        for k in applied:
            ns = "FLAGS" if k in fb else ("PARAMS" if k in pb else "??")
            print(f"    namespace[{k}] = {ns}")
        diff("a1_baseline", pa, fa, label, pb, fb)
    return 0


if __name__ == "__main__":
    sys.exit(main())
