"""efold1: no-solve config diff -- the 2026-08-11 F2 shots' RECORDED stance
(params_json/flags_json read straight out of each shot's h5, run at HEAD
4e4dd27) vs what the SAME recipe builds at the current HEAD 57ba63e.

Purpose: decide, before spending ~3 h of solve, whether the e-fold-owner
decomposition (covcal_efold_read.py) has an honest-clock re-run to do at all,
or whether the intervening circuit work is bit-exact-off at this stance.

Reuses recertdiag2_configdiff.diff verbatim as the printer.
"""
import json
import sys
from pathlib import Path

import h5py

from covbuild_run_conducting_phase import build_config as covbuild_config
from recertdiag2_configdiff import diff

HERE = Path(__file__).resolve().parent

SHOTS = {"covdecide_twion_f005": 1390.0,
         "covcal_f2_shot1": 179.9,
         "covcal_f2_shot2": 1.996e-4}

EXTRA = {"max_steps_action": "stop",
         "heating_anomalous_transport": "tail_walk",
         "heating_anomalous_tail_ionization": "on"}


def main():
    print("== efold1_diffcfg: F2 shot h5 stance (HEAD 4e4dd27) vs the same "
          "recipe rebuilt at HEAD 57ba63e\n")
    for stem, r in SHOTS.items():
        h5 = HERE / f"{stem}.h5"
        if not h5.exists():
            print(f"{stem}: ABSENT")
            continue
        with h5py.File(h5, "r") as f:
            pa = json.loads(f.attrs["params_json"])
            fa = json.loads(f.attrs["flags_json"])
        pb, fb = covbuild_config(60, coverage=(0.05, r), extra=dict(EXTRA))
        diff(f"{stem}.h5 (RECORDED, 4e4dd27)", pa, fa,
             f"{stem} recipe REBUILT (57ba63e)", dict(pb), dict(fb))
    return 0


if __name__ == "__main__":
    sys.exit(main())
