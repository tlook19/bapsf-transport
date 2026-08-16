"""ea1v: per-arm no-solve config diff -- PREFLIGHT for the ea1 stage-1 verdict runs.

Constructs nothing, integrates nothing. For each of the four ea1v arms it builds
the efold1 BASELINE config (regime_r2_overlap_gate.build_config(20, False), the
a1 arm's config verbatim) and the ARMED config (the same builder + the arm's
--set list routed through recertdiag2_traj.apply_set, i.e. THE EXACT SAME
namespace resolver efold1_traj.py / pd0_endvent_traj.py use), and prints the
key-by-key diff.

Purpose (the diffcfg technique, standing rule): expected deltas only, or stop.
The flag-vs-param trap is exactly what this catches: cathode_emitting_area is a
FLAG, cathode_emitting_area_initial_fraction is a PARAM.

Reuses recertdiag2_configdiff.diff verbatim as the printer and
recertdiag2_traj.apply_set verbatim as the resolver -- neither is modified.

NB apply_set json.loads() each value, so a value that EQUALS the shipped default
correctly prints no diff line while still appearing in the 'applied' echo: the
0.0075 seed IS the shipped default (the derived bracket midpoint), so the two
central arms arm it explicitly and it shows in 'applied', not in the diff.
"""
import sys

from recertdiag2_configdiff import diff
from recertdiag2_traj import apply_set
from regime_r2_overlap_gate import build_config

ARMS = {
    "ea1v_seed": ["cathode_emitting_area=true",
                  "cathode_emitting_area_initial_fraction=0.0075"],
    "ea1v_seed_lo": ["cathode_emitting_area=true",
                     "cathode_emitting_area_initial_fraction=0.0063"],
    "ea1v_seed_hi": ["cathode_emitting_area=true",
                     "cathode_emitting_area_initial_fraction=0.0087"],
    "ea1v_disposal": ["cathode_emitting_area=true",
                      "cathode_emitting_area_initial_fraction=0.0075",
                      'heating_anomalous_disposal="landau_branched"',
                      "heating_anomalous_tail_phi_c_fraction=1.0"],
}


def main():
    print("== ea1v_diffcfg: efold1 a1 BASELINE config  vs  each ea1v arm's ARMED "
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
