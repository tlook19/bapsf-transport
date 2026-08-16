"""pd1: anti-vacuity, BPD landmarks and the foot/finish tau split.

NO SOLVE, NO NEW IMPLEMENTATION. This file only names the pd1 arms and then
runs ``sc1_reads.main()`` -- the sc1 instrument of record, imported and
executed unmodified, so every definition (the I_wall anti-vacuity reference,
the 1e11 BPD crossings raw + ln-interpolated, the 30%/30% time-split
foot/finish least-squares taus) is byte-for-byte the one the sc1 leg reported
under. The module's own docstring carries those definitions and their caveats,
including the standing BPD label.

The efold1 a1 baseline row is REUSED, not re-run (tau_F2 6.8595 us of record).
"""
import sys

import sc1_reads

sc1_reads.ARMS = [
    ("efold1_a1_baseline", "REUSED efold1 baseline (tau 6.86 us of record)"),
    ("pd1_f100", "landau_branched, phi_c_fraction 1.0 (CENTRAL ARM)"),
    ("pd1_f050", "landau_branched, phi_c_fraction 0.5"),
    ("pd1_f025", "landau_branched, phi_c_fraction 0.25"),
]

if __name__ == "__main__":
    sys.exit(sc1_reads.main())
