"""pd1: per-arm no-solve config diff -- PREFLIGHT for the disposal leg.

Constructs nothing, integrates nothing. Identical in form to sc1_diffcfg.py:
for each of the three pd1 arms it builds the efold1 BASELINE config
(regime_r2_overlap_gate.build_config(20, False), the a1 arm's config verbatim)
and the ARMED config (the same builder + the arm's --set list routed through
recertdiag2_traj.apply_set, i.e. THE EXACT SAME namespace resolver
efold1_traj.py / pd0_endvent_traj.py use), and prints the key-by-key diff.

Purpose (the diffcfg technique, standing rule): expected deltas only, or stop.
Both pd1 keys are input_dict/params keys, so a key that landed in the wrong
namespace -- or a --set that silently moved a second key -- shows here BEFORE
any solve is spent.

Reuses recertdiag2_configdiff.diff verbatim as the printer and
recertdiag2_traj.apply_set verbatim as the resolver -- neither is modified.
"""
import sys

from recertdiag2_configdiff import diff
from recertdiag2_traj import apply_set
from regime_r2_overlap_gate import build_config

ARMS = {
    "pd1_f100": ['heating_anomalous_disposal="landau_branched"',
                 "heating_anomalous_tail_phi_c_fraction=1.0"],
    "pd1_f050": ['heating_anomalous_disposal="landau_branched"',
                 "heating_anomalous_tail_phi_c_fraction=0.5"],
    "pd1_f025": ['heating_anomalous_disposal="landau_branched"',
                 "heating_anomalous_tail_phi_c_fraction=0.25"],
}


def main():
    print("== pd1_diffcfg: efold1 a1 BASELINE config  vs  each pd1 arm's ARMED "
          "config, both built at HEAD 7bd4041, no solve\n")
    for label, sets in ARMS.items():
        pa, fa = build_config(20, False)
        pb, fb = build_config(20, False)
        applied = apply_set(pb, fb, sets)
        print(f"--- {label}: --set {' '.join(sets)}")
        print(f"    apply_set resolved: {applied}")
        print(f"    namespace check: "
              f"{[(k, 'params' if k in pb else 'flags') for k in applied]}")
        diff("a1_baseline", pa, fa, label, pb, fb)
    return 0


if __name__ == "__main__":
    sys.exit(main())
