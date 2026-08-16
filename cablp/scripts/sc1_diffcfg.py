"""sc1: per-arm no-solve config diff -- PREFLIGHT for the structural-channel leg.

Constructs nothing, integrates nothing. For each of the four sc1 arms it builds
the efold1 BASELINE config (regime_r2_overlap_gate.build_config(20, False), the
a1 arm's config verbatim) and the ARMED config (the same builder + the arm's
--set list routed through recertdiag2_traj.apply_set, i.e. THE EXACT SAME
namespace resolver efold1_traj.py uses), and prints the key-by-key diff.

Purpose (the diffcfg technique, standing rule): expected deltas only, or stop.
A key that lands in the wrong namespace, or a --set that silently moves a
second key, shows here BEFORE any solve is spent.

Reuses recertdiag2_configdiff.diff verbatim as the printer and
recertdiag2_traj.apply_set verbatim as the resolver -- neither is modified.

NB apply_set json.loads() each value, so string values must be quoted twice on
the command line (as efold1_a3_*.cmd did) and an integer literal 30 compares
EQUAL to the shipped float default 30.0 -- an unchanged value correctly prints
no diff line while still appearing in the 'applied' echo.
"""
import sys

from recertdiag2_configdiff import diff
from recertdiag2_traj import apply_set
from regime_r2_overlap_gate import build_config

ARMS = {
    "sc1_b30": ['beam_anomalous_model="ql_relaxation"', "ql_relaxation_coeff=30"],
    "sc1_b100": ['beam_anomalous_model="ql_relaxation"', "ql_relaxation_coeff=100"],
    "sc1_bx30": ['beam_anomalous_model="ql_relaxation"', "ql_relaxation_coeff=30",
                 "coverage_closure=true", "coverage_initial_fraction=0.05",
                 "coverage_growth_rate_per_s=1390"],
    "sc1_bx100": ['beam_anomalous_model="ql_relaxation"', "ql_relaxation_coeff=100",
                  "coverage_closure=true", "coverage_initial_fraction=0.05",
                  "coverage_growth_rate_per_s=1390"],
}


def main():
    print("== sc1_diffcfg: efold1 a1 BASELINE config  vs  each sc1 arm's ARMED "
          "config, both built at HEAD 57ba63e, no solve\n")
    for label, sets in ARMS.items():
        pa, fa = build_config(20, False)
        pb, fb = build_config(20, False)
        applied = apply_set(pb, fb, sets)
        print(f"--- {label}: --set {' '.join(sets)}")
        print(f"    apply_set resolved: {applied}")
        diff("a1_baseline", pa, fa, label, pb, fb)
    return 0


if __name__ == "__main__":
    sys.exit(main())
