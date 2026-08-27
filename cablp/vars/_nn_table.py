"""
Equilibrated background neutral density (nn0) lookup table.

Data is loaded from nn_table.csv in the same directory.

*** THIS TABLE IS FROZEN and cannot be regenerated in-tree (2026-08-03). ***
Its generator (the untracked scripts/generate_nn_table.py) drove the 0D
_sim3 solver, which has been removed. The DATA below survives and this
lookup still works; the ability to REGENERATE it does not. Reproduce the
generator at tag legacy-final-2026-07-22 if the table must ever change.

Note the production stance never reaches this table at all: both the default
and golden configs pin nn0 = 2.0e13 cm^-3 explicitly, so
core.config.resolve_nn0 short-circuits before calling lookup_nn0.

Values came from a 100-cycle plasma-off equilibration, the same one used by
the bapsf-app "Auto-equilibrate nn0" feature:

  - 100 cycles × 3 s each, Plasma=False, adaptive RK45, dt_max=1e-2 s
  - Gas puff active for first 20 ms of each cycle
  - All cells start at nn0_init = 1e8 cm^-3

The equilibrated profile is uniform to <0.6 % across cells for all S_gp
configurations, so a single scalar applies for any cell count.

Fixed conditions (defaults from input_dict_template):
  S_pump_L = S_pump_R = 4000 L/s
  Lm = Lp = 1800 cm,  Rm = 50 cm
  gas_type = "He"

Keys are per-cathode S_gp in SCCM (100 – 16000):
  single cathode: one source at cell 0
  twin cathode  : S_gp = Twin_S_gp (app splits total S_gp equally)

*** LEGACY SCCM CONVENTION. *** The keys are pre-2026-08-21 0 C-sccm, while
the repo's sccm now MEANS meter-sccm (Sensirion SFM5500, 20 C / 1013 mbar) --
a ~7% different particle throughput at the same numeric key. The table's own
generator retired with _sim3, so it cannot be regenerated on the new
convention, and the keys are ANNOTATED rather than rescaled: rescaling frozen
data would forge an interpolation that was never computed.

**Nothing in the repo converts, and that is deliberate.** The only in-repo
caller is the fallback branch of ``core.config.resolve_nn0``, which passes the
configured meter-sccm ``S_gp`` straight through and carries the ~7%
inconsistency as a documented note. Production never reaches it: both the
config default and the stance of record pin ``nn0`` explicitly, so
``resolve_nn0`` short-circuits first. A conversion is therefore owed by any
NEW caller that actually depends on this table -- it is not applied here, and
it is not applied for you.

Usage
-----
    from cablp.vars._nn_table import lookup_nn0

    nn0 = lookup_nn0(500)            # single, 500 SCCM
    nn0 = lookup_nn0(500, twin=True) # twin, 500 SCCM per cathode (1000 total)
"""

import math
from pathlib import Path

import numpy as np

# ── Load table from CSV at import time ───────────────────────────────────────
_CSV = Path(__file__).parent / "nn_table.csv"
_data = np.loadtxt(_CSV, delimiter=",", comments="#")
# columns: s_gp_sccm, nn0_single_cm3, nn0_twin_cm3
_SGP_ARR    = _data[:, 0]
_SINGLE_ARR = _data[:, 1]
_TWIN_ARR   = _data[:, 2]

_SGP_MIN = float(_SGP_ARR[0])
_SGP_MAX = float(_SGP_ARR[-1])


def lookup_nn0(s_gp, twin=False):
    """
    Return the equilibrated background neutral density for a given S_gp.

    Interpolates log-linearly between tabulated values.  The relationship is
    nearly exactly linear (nn0 ∝ S_gp at fixed pumping), so interpolation
    error is negligible within the tabulated range.

    Parameters
    ----------
    s_gp : float
        Per-cathode gas-puff rate [SCCM].  For a twin discharge where the app
        splits the total S_gp equally, pass the per-cathode value (= total / 2).
    twin : bool
        True for twin-cathode configuration, False (default) for single.

    Returns
    -------
    float
        Equilibrated nn0 [cm^-3], valid as nn0, Source_nn0, and Twin_nn0.

    Raises
    ------
    ValueError
        If s_gp is outside the tabulated range [100, 16000] SCCM.
    """
    if s_gp < _SGP_MIN or s_gp > _SGP_MAX:
        raise ValueError(
            f"s_gp={s_gp} SCCM is outside the tabulated range "
            f"[{_SGP_MIN:.0f}, {_SGP_MAX:.0f}] SCCM."
        )

    arr = _TWIN_ARR if twin else _SINGLE_ARR

    # Exact hit (integer S_gp)
    idx = np.searchsorted(_SGP_ARR, s_gp)
    if _SGP_ARR[idx] == s_gp:
        return float(arr[idx])

    # Log-linear interpolation between neighbouring tabulated points
    lo, hi = _SGP_ARR[idx - 1], _SGP_ARR[idx]
    t = math.log(s_gp / lo) / math.log(hi / lo)
    return math.exp((1 - t) * math.log(arr[idx - 1]) + t * math.log(arr[idx]))
